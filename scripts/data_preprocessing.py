import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class DataPreprocessor:
    def __init__(self, train_file, test_file):
        self.train_df = pd.read_csv(train_file)
        self.test_df = pd.read_csv(test_file)
        self.label_encoder = LabelEncoder()
        self.word2vec_model = None

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\d', '', text)
        text = re.sub(r'(?<=\b\w)(\s+\w\b)+', lambda match: match.group(0).replace(" ", ""), text)
        words = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        words = [word for word in words if len(word) > 2]
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        return words

    def process_data(self):
        self.train_df.dropna(inplace=True)
        self.test_df.dropna(inplace=True)
        self.train_df['processed_text'] = self.train_df['text'].apply(self.preprocess_text)
        self.test_df['processed_text'] = self.test_df['text'].apply(self.preprocess_text)

        self.word2vec_model = Word2Vec(sentences=self.train_df['processed_text'], vector_size=100, window=5, min_count=1, workers=4)
        self.train_df['text_vectors'] = self.train_df['processed_text'].apply(lambda x: [self.word2vec_model.wv[word] for word in x if word in self.word2vec_model.wv])
        self.test_df['text_vectors'] = self.test_df['processed_text'].apply(lambda x: [self.word2vec_model.wv[word] for word in x if word in self.word2vec_model.wv])

        self.label_encoder.fit(self.train_df['target_col'])
        self.train_df['encoded_target'] = self.label_encoder.transform(self.train_df['target_col'])
        self.test_df['encoded_target'] = self.label_encoder.transform(self.test_df['target_col'])
        
        self.word2vec_model.save("models/word2vec_model.bin")
        joblib.dump(self.label_encoder, 'models/label_encoder.pkl')

    def pad_vectors(self, vectors, max_len=100):
        if len(vectors) > max_len:
            vectors = vectors[:max_len]
        else:
            vectors += [np.zeros(vectors[0].shape)] * (max_len - len(vectors))
        return np.array(vectors)

    def flatten_vectors(self, vectors, max_len=100):
        padded_vectors = self.pad_vectors(vectors, max_len)
        return padded_vectors.reshape(-1)

    def prepare_data(self):
        max_len = 100
        self.train_df['processed_text_vectors'] = self.train_df['text_vectors'].apply(lambda x: self.flatten_vectors(x, max_len))
        self.test_df['processed_text_vectors'] = self.test_df['text_vectors'].apply(lambda x: self.flatten_vectors(x, max_len))
        X_train = np.array(list(self.train_df['processed_text_vectors']))
        X_test = np.array(list(self.test_df['processed_text_vectors']))
        y_train = self.train_df['encoded_target'].values
        y_test = self.test_df['encoded_target'].values
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocessor = DataPreprocessor('data/raw/train_data_extracted.csv', 'data/raw/test_data_extracted.csv')
    preprocessor.process_data()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data()
    np.save('data/processed/X_train.npy', X_train)
    np.save('data/processed/X_test.npy', X_test)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/y_test.npy', y_test)
    print("Preprocessing complete. Data saved.")