import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, logging
from tqdm.auto import tqdm
import pickle
import os

# Setting logging level to error to reduce warnings
logging.set_verbosity_error()

class TextDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

class TextClassificationPipeline:
    def __init__(self, model_name='distilbert-base-uncased', num_labels=4, device=None):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Using device: {self.device}")

    def train(self, train_loader, val_loader=None, epochs=5, lr=5e-5):
        optimizer = AdamW(self.model.parameters(), lr=lr)
        scaler = GradScaler(enabled=(self.device == 'cuda'))
        best_val_accuracy = 0

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            train_preds, train_labels = [], []

            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with autocast(enabled=(self.device == 'cuda')):
                    outputs = self.model(**batch)
                    loss = outputs.loss

                if self.device == 'cuda':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

                total_loss += loss.item()
                train_preds.extend(outputs.logits.argmax(dim=-1).cpu().numpy())
                train_labels.extend(batch['labels'].cpu().numpy())

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            train_accuracy = accuracy_score(train_labels, train_preds)
            print(f'Epoch {epoch+1} - Train Loss: {total_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.4f}')

            if val_loader:
                val_accuracy = self.evaluate(val_loader)
                print(f'Validation Accuracy: {val_accuracy:.4f}')
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    self.save_model("best_model", "best_tokenizer")
                    print("Best model saved!")

    def evaluate(self, data_loader):
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                all_preds.extend(outputs.logits.argmax(dim=-1).cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())

        return accuracy_score(all_labels, all_preds)

    def predict(self, texts):
        dataset = TextDataset(texts, tokenizer=self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=32)
        
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                all_preds.extend(outputs.logits.argmax(dim=-1).cpu().numpy())

        return all_preds

    def save_model(self, model_path, tokenizer_path):
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(tokenizer_path)
        print(f"Model saved to {model_path} and tokenizer saved to {tokenizer_path}")

    def load_model(self, model_path, tokenizer_path):
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
        self.model.to(self.device)
        print(f"Model loaded from {model_path} and tokenizer loaded from {tokenizer_path}")

def prepare_data(df, text_column, target_column, label_encoder):
    texts = df[text_column].tolist()
    labels = label_encoder.transform(df[target_column])
    return texts, labels

if __name__ == "__main__":
    # Loading and preprocess data
    train_df = pd.read_csv("data/raw/train_data_extracted.csv").dropna()
    test_df = pd.read_csv("data/raw/test_data_extracted.csv").dropna()

    with open("models/label_encoder.pkl", "rb") as f:
        label_encoder_array = pickle.load(f)

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.unique(label_encoder_array)

    target_column = 'target_col'
    text_column = 'text'

    # Preparing data
    train_texts, train_labels = prepare_data(train_df, text_column, target_column, label_encoder)
    test_texts, test_labels = prepare_data(test_df, text_column, target_column, label_encoder)

    # Initializing pipeline
    pipeline = TextClassificationPipeline(num_labels=len(label_encoder.classes_))

    # Creating datasets and dataloaders
    train_dataset = TextDataset(train_texts, train_labels, pipeline.tokenizer)
    test_dataset = TextDataset(test_texts, test_labels, pipeline.tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Training the model
    pipeline.train(train_loader, val_loader=test_loader, epochs=5)

    # Evaluating on test set
    test_accuracy = pipeline.evaluate(test_loader)
    print(f'Final Test Accuracy: {test_accuracy:.4f}')

    # Making predictions on test set
    test_preds = pipeline.predict(test_texts)
    print(classification_report(test_labels, test_preds, target_names=label_encoder.classes_))

    # Saving the final model
    pipeline.save_model("models/distilbert_model", "models/distilbert_tokenizer")