# Electrical PDFs Classification System 
[Link to App](https://parspecassignment.streamlit.app/)


## Project Overview
The Electrical PDFs Classification System is an advanced machine learning application designed to categorize electrical engineering documents automatically. Leveraging state-of-the-art models and a robust preprocessing pipeline, this system accurately classifies PDF documents by extracting and analyzing their textual content. The project employs a combination of traditional machine learning techniques, such as Support Vector Machines (SVM), and modern deep learning approaches, including DistilBERT, to deliver high-performance results.

The system's architecture is built to handle large volumes of unstructured text data, efficiently converting them into meaningful insights. This solution is particularly useful for organizations or individuals needing to organize and manage large collections of technical documents. By automating the classification process, the Electrical PDFs Classification System saves time and reduces the potential for human error, making it a reliable tool for document management in the electrical engineering domain.

## User Interface

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

## Application Overview
The project also includes a Streamlit-based web application that allows users to classify PDF documents using the trained models. The app provides a user-friendly interface where users can input a PDF URL and choose between the SVM and DistilBERT models for classification.

### App Features
- **PDF Text Extraction**: The app uses the Doctr library to extract text from the provided PDF document.
- **Model Selection**: Users can choose between the SVM and DistilBERT models for classification.
- **Prediction Display**: The app displays the predicted document label along with the probability of the prediction.

## Conclusion
The Electrical PDFs Classification System exemplifies the successful deployment of machine learning models in a real-world scenario. It combines comprehensive data preprocessing, sophisticated model training, and a user-friendly deployment platform to deliver a high-performance solution for classifying electrical engineering documents. This project not only showcases the potential of AI in automating complex tasks but also sets a benchmark for future projects in the domain.



![image](https://github.com/user-attachments/assets/57786672-65c0-44f8-a18a-8aa89f742b90)
