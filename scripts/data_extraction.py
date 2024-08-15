import os
import gdown
import requests
import pandas as pd
import numpy as np
import logging
import threading
import time
import re
from io import BytesIO
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Suppressing deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Setting up logging
logging.basicConfig(filename='text_extraction.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PDFTextExtractor:
    def __init__(self, model, timeout=20):
        self.model = model
        self.timeout = timeout
        self.pdf_counter = 0

    def fix_url(self, url):
        """Fix URLs that are missing a scheme."""
        if not re.match(r'https?:', url):
            url = 'https://' + url.lstrip('/')
        return url

    def extract_text_from_pdf(self, pdf_url):
        self.pdf_counter += 1
        row_number = self.pdf_counter  # Use the global counter to get the current row number
        logging.info(f"Processing row {row_number} with URL: {pdf_url}...")

        # Fixing the URL if needed
        pdf_url = self.fix_url(pdf_url)

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._extract_text, pdf_url)
                return future.result(timeout=self.timeout)
        except TimeoutError:
            logging.warning(f"Timeout reached for row {row_number}. Skipping PDF: {pdf_url}")
            return ""  # Returning an empty string or placeholder for timeout cases
        except Exception as e:
            logging.error(f"Error processing PDF at row {row_number}: {e}")
            return ""  # Returning an empty string or placeholder for other errors

    def _extract_text(self, pdf_url):
        # Downloading the PDF from the URL
        pdf_path = gdown.download(pdf_url, 'temp.pdf', quiet=True)

        # Using Doctr to extract text
        document = DocumentFile.from_pdf(pdf_path)
        result = self.model(document)
        json_response = result.export()

        # Extracting text from the first two pages of the OCR result
        values = []
        num_pages_to_process = min(2, len(json_response['pages']))  # Ensure we don't go beyond available pages
        for page_index in range(num_pages_to_process):
            page = json_response['pages'][page_index]
            for block in page['blocks']:
                for line in block['lines']:
                    for word in line['words']:
                        values.append(word['value'])

        return " ".join(values)

class DataProcessor:
    def __init__(self, train_file, test_file, model):
        self.train_file = train_file
        self.test_file = test_file
        self.model = model

    def load_data(self):
        # Loading the Excel file
        excel_data = pd.ExcelFile(self.train_file)

        # Loading the sheets into separate DataFrames
        self.train_df = pd.read_excel(excel_data, sheet_name='train_data')
        self.test_df = pd.read_excel(excel_data, sheet_name='test_data')

    def preprocess_data(self):
        logging.info("Starting data preprocessing...")

        # Preprocessing train data
        initial_train_rows = len(self.train_df)
        
        # Drop duplicates
        self.train_df.drop_duplicates(subset='datasheet_link', inplace=True)
        after_duplicate_removal = len(self.train_df)
        
        # Drop rows with missing values
        self.train_df.dropna(subset=['datasheet_link'], inplace=True)
        after_na_removal = len(self.train_df)

        logging.info(f"Train data: Initial rows: {initial_train_rows}")
        logging.info(f"Train data: Rows after removing duplicates: {after_duplicate_removal}")
        logging.info(f"Train data: Rows after removing missing values: {after_na_removal}")

        # Preprocessing test data
        initial_test_rows = len(self.test_df)
        
        # Dropping duplicates in test data
        self.test_df.drop_duplicates(subset='datasheet_link', inplace=True)
        after_test_duplicate_removal = len(self.test_df)
        
        # Dropping rows with missing values in test data
        self.test_df.dropna(subset=['datasheet_link'], inplace=True)
        after_test_na_removal = len(self.test_df)

        logging.info(f"Test data: Initial rows: {initial_test_rows}")
        logging.info(f"Test data: Rows after removing duplicates: {after_test_duplicate_removal}")
        logging.info(f"Test data: Rows after removing missing values: {after_test_na_removal}")

        # Splitting the train dataframe into chunks
        self.train_dfs = np.array_split(self.train_df, 3)
        logging.info(f"Train DataFrames split into {len(self.train_dfs)} chunks.")

    def extract_text(self):
        logging.info("Starting text extraction...")

        extractor = PDFTextExtractor(self.model)
        # Initializing an empty list to collect all chunks
        all_chunks = []

        for i, chunk in enumerate(self.train_dfs):
            logging.info(f"Processing chunk {i+1}/{len(self.train_dfs)}...")
            chunk['text'] = chunk['datasheet_link'].apply(lambda link: extractor.extract_text_from_pdf(link))
            all_chunks.append(chunk)
            logging.info(f"Chunk {i+1} processed.")

        # Combining all chunks into a single DataFrame
        combined_df = pd.concat(all_chunks, ignore_index=True)
        combined_df.to_csv('data/raw/train_data_extracted.csv', index=False)
        logging.info("Train data processed and saved as train_data_extracted.csv")

        # Processing test data
        logging.info("Processing test data...")
        self.test_df['text'] = self.test_df['datasheet_link'].apply(lambda link: extractor.extract_text_from_pdf(link))
        self.test_df.to_csv('data/raw/test_data_extracted.csv', index=False)
        logging.info("Test data processed and saved as test_data_extracted.csv")

if __name__ == "__main__":
    # Initializing the OCR predictor
    model = ocr_predictor(pretrained=True)

    # Initializing DataProcessor and run the pipeline
    processor = DataProcessor('data/raw/data.xlsx', None, model)
    processor.load_data()
    processor.preprocess_data()
    processor.extract_text()

    logging.info("Pipeline completed.")