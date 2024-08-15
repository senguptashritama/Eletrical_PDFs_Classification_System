import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from tqdm.auto import tqdm
import pickle

class TextClassificationPipeline:
    def __init__(self, model_name='distilbert-base-uncased', num_labels=4, device=None):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def preprocess_text(self, text):
        return self.tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')

    def prepare_data(self, texts, labels=None):
        encodings = [self.preprocess_text(text) for text in texts]
        input_ids = torch.stack([enc['input_ids'].squeeze(0) for enc in encodings])
        attention_masks = torch.stack([enc['attention_mask'].squeeze(0) for enc in encodings])

        if labels is not None:
            labels = torch.tensor(labels)
            return TensorDataset(input_ids, attention_masks, labels)
        return TensorDataset(input_ids, attention_masks)

    def train(self, train_loader, epochs=5, lr=5e-5):
        optimizer = AdamW(self.model.parameters(), lr=lr)
        scaler = GradScaler()

        for epoch in range(epochs):
            total_loss = 0
            all_labels, all_preds = [], []

            pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}', unit='batch')
            self.model.train()

            for batch in train_loader:
                input_ids, attention_mask, labels = [item.to(self.device) for item in batch]
                labels = labels.long()

                with autocast():
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                total_loss += loss.item()
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
                
                pbar.update(1)

            pbar.close()
            avg_loss = total_loss / len(train_loader)
            accuracy = accuracy_score(torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy())

            print(f'Epoch {epoch+1} completed with average loss: {avg_loss:.4f} and accuracy: {accuracy:.4f}')

    def evaluate(self, test_loader):
        self.model.eval()
        test_preds = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids, attention_mask = [item.to(self.device) for item in batch]
                outputs = self.model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                test_preds.extend(preds)

        return test_preds

    def save_model(self, model_path, tokenizer_path):
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(tokenizer_path)
        print("Model and tokenizer saved successfully!")


train_df = pd.read_csv("train_data_extracted.csv").dropna()
test_df = pd.read_csv("test_data_extracted.csv").dropna()


with open("label_encoder.pkl", "rb") as f:
    label_encoder_array = pickle.load(f)


unique_labels = np.unique(label_encoder_array)
label_encoder = LabelEncoder()
label_encoder.classes_ = unique_labels


target_column = 'target_col'  
text_column = 'text'  

train_df['encoded_target'] = label_encoder.transform(train_df[target_column])
test_df['encoded_target'] = label_encoder.transform(test_df[target_column])


if __name__ == "__main__":
    pipeline = TextClassificationPipeline()

    train_dataset = pipeline.prepare_data(train_df[text_column], train_df['encoded_target'].values)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    test_dataset = pipeline.prepare_data(test_df[text_column])
    test_loader = DataLoader(test_dataset, batch_size=16)

 
    pipeline.train(train_loader, epochs=5)

    test_preds = pipeline.evaluate(test_loader)
    test_df['predictions'] = test_preds

    accuracy = accuracy_score(test_df['encoded_target'], test_df['predictions'])
    print(f'Test Accuracy: {accuracy:.2f}')
    print(classification_report(test_df['encoded_target'], test_df['predictions']))

    pipeline.save_model("./distilbert_model", "./distilbert_tokenizer")