import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib

class SVM_ModelTrainer:
    def __init__(self):
        self.svm_model = None

    def load_data(self):
        X_train = np.load('data/processed/X_train.npy')
        X_test = np.load('data/processed/X_test.npy')
        y_train = np.load('data/processed/y_train.npy')
        y_test = np.load('data/processed/y_test.npy')
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        self.svm_model = SVC(kernel='linear', probability=True)
        self.svm_model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        y_pred = self.svm_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f'Accuracy: {accuracy:.2f}')
        print('Classification Report:')
        print(report)

    def cross_validate(self, X_train, y_train):
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.svm_model, X_train, y_train, cv=kf)
        print(f'Cross-validation scores: {cv_scores}')
        print(f'Mean cross-validation score: {cv_scores.mean():.2f}')
        print(f'Standard deviation of cross-validation scores: {cv_scores.std():.2f}')

    def save_model(self):
        joblib.dump(self.svm_model, 'models/svm_model.pkl')
        print("Model saved as svm_model.pkl")

if __name__ == "__main__":
    trainer = SVM_ModelTrainer()
    X_train, X_test, y_train, y_test = trainer.load_data()
    trainer.train_model(X_train, y_train)
    trainer.evaluate_model(X_test, y_test)
    trainer.cross_validate(X_train, y_train)
    trainer.save_model()