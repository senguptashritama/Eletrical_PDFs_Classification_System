Here's your text in Markdown format:

```markdown
# Electrical PDFs Classification System

## Project Overview
The Electrical PDFs Classification System is a machine learning-powered solution for classifying electrical engineering documents. Given a PDF document, the system can automatically determine the document's category or classification.

## Data Preprocessing
The data preprocessing steps are as follows:

- **Data Loading**: The project loads the training and test data from CSV files into Pandas DataFrames.
- **Text Extraction**: The PDF documents are downloaded from the provided URLs, and their text content is extracted using the Doctr library. This text is stored in the `text` column of the DataFrames.
- **Text Cleaning and Preprocessing**: The extracted text is preprocessed by:
  - Converting to lowercase
  - Removing non-alphanumeric characters
  - Removing stopwords
  - Lemmatizing the words
- **Word Embedding**: The preprocessed text is converted into numerical vectors using the Word2Vec model, which is trained on the training data.
- **Label Encoding**: The target labels are encoded using a `LabelEncoder`.
- **Data Splitting**: The data is split into training and test sets, with the processed text vectors and encoded labels saved to disk for use in the model training.

## Model Training
Two models were trained and evaluated for this project:

- **SVM Model**:
  - An SVM (Support Vector Machine) model with a linear kernel was trained on the preprocessed text vectors.
  - The model was evaluated using the held-out test set, achieving an accuracy of 96%.
  - 5-fold cross-validation was performed, resulting in a mean accuracy of 96% and a standard deviation of 2%.
  - The trained SVM model was saved to disk for later use.

- **DistilBERT Model**:
  - A DistilBERT-based sequence classification model was trained on the preprocessed text and labels.
  - The model was trained for 5 epochs using the AdamW optimizer and gradient scaling (for improved performance on GPUs).
  - The trained model achieved a test accuracy of 99%.
  - The trained DistilBERT model and tokenizer were saved to disk.

## Model Evaluation
The performance of the models was evaluated using the following metrics:

- **Accuracy**: Both the SVM and DistilBERT models achieved high accuracy scores, with the DistilBERT model outperforming the SVM model.
- **Precision, Recall, and F1-score**: The classification reports for both models showed high scores for these metrics, indicating the models were able to accurately classify the documents across all classes.
- **Cross-Validation**: The SVM model's cross-validation scores were consistent with the test set performance, further validating the model's robustness.

## Conclusion
The Electrical PDFs Classification System demonstrates the effective use of machine learning techniques to automate the classification of electrical engineering documents. The combination of data preprocessing, SVM modeling, and DistilBERT modeling resulted in a highly accurate and reliable system for classifying these types of documents.
```


![image](https://github.com/user-attachments/assets/57786672-65c0-44f8-a18a-8aa89f742b90)
