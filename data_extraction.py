import os
import gdown
import requests
import pandas as pd
import numpy as np
import logging
import torch
import signal
import re
from io import BytesIO
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# Set up logging
logging.basicConfig(filename='text_extraction.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PDFTextExtractor:
    def __init__(self, model, timeout=10):
        self.model = model
        self.timeout = timeout
        self.pdf_counter = 0

        # Set up signal handler for timeout
        signal.signal(signal.SIGALRM, self.timeout_handler)

    class TimeoutException(Exception):
        pass

    def timeout_handler(self, signum, frame):
        raise self.TimeoutException()

    def fix_url(self, url):
        """Fix URLs that are missing a scheme."""
        if not re.match(r'https?:', url):
            url = 'https://' + url.lstrip('/')
        return url

    def extract_text_from_pdf(self, pdf_url):
        self.pdf_counter += 1
        row_number = self.pdf_counter  # Use the global counter to get the current row number
        logging.info(f"Processing row {row_number} with URL: {pdf_url}...")

        # Fix the URL if needed
        pdf_url = self.fix_url(pdf_url)

        # Set a timeout
        signal.alarm(self.timeout)

        try:
            # Download the PDF from the URL
            pdf_path = gdown.download(pdf_url, 'temp.pdf', quiet=True)

            # Use Doctr to extract text
            document = DocumentFile.from_pdf(pdf_path)
            result = self.model(document)
            json_response = result.export()

            # Extract text from the first two pages of the OCR result
            values = []
            num_pages_to_process = min(2, len(json_response['pages']))  # Ensure we don't go beyond available pages
            for page_index in range(num_pages_to_process):
                page = json_response['pages'][page_index]
                for block in page['blocks']:
                    for line in block['lines']:
                        for word in line['words']:
                            values.append(word['value'])

            # Reset the alarm
            signal.alarm(0)

            return " ".join(values)
        except self.TimeoutException:
            logging.warning(f"Timeout reached for row {row_number}. Skipping PDF: {pdf_url}")
            return ""  # Return an empty string or placeholder for timeout cases
        except Exception as e:
            # In case of an error, check if the URL needs correction
            if 'Invalid URL' in str(e):
                suggested_url = re.search(r'Perhaps you meant (https?:[^\s]+)', str(e))
                if suggested_url:
                    suggested_url = suggested_url.group(1)
                    logging.info(f"Trying suggested URL: {suggested_url}")
                    return self.extract_text_from_pdf(suggested_url)  # Retry with suggested URL
            logging.error(f"Error processing PDF at row {row_number}: {e}")
            return ""  # Return an empty string or placeholder for other errors

class DataProcessor:
    def __init__(self, train_file, test_file, model):
        self.train_file = train_file
        self.test_file = test_file
        self.model = model

    def load_data(self):
        # Load the Excel file
        excel_data = pd.ExcelFile(self.train_file)

        # Load the sheets into separate DataFrames
        self.train_df = pd.read_excel(excel_data, sheet_name='train_data')
        self.test_df = pd.read_excel(excel_data, sheet_name='test_data')

    def preprocess_data(self):
        logging.info("Starting data preprocessing...")

        # Drop duplicates and check for missing values
        duplicate_count = self.train_df['datasheet_link'].duplicated().sum()
        logging.info(f"Number of duplicate values: {duplicate_count}")
        self.train_df.drop_duplicates(subset='datasheet_link', inplace=True)

        missing_links = self.train_df['datasheet_link'].isnull()
        num_missing_links = missing_links.sum()
        logging.info(f"Number of missing values in 'datasheet_link' column: {num_missing_links}")

        # Split the train dataframe into chunks
        self.train_dfs = np.array_split(self.train_df, 3)
        logging.info(f"Train DataFrames split into {len(self.train_dfs)} chunks.")

    def extract_text(self):
        logging.info("Starting text extraction...")

        extractor = PDFTextExtractor(self.model)
        # Initialize an empty list to collect all chunks
        all_chunks = []

        for i, chunk in enumerate(self.train_dfs):
            logging.info(f"Processing chunk {i+1}/{len(self.train_dfs)}...")
            chunk['text'] = chunk['datasheet_link'].apply(lambda link: extractor.extract_text_from_pdf(link))
            all_chunks.append(chunk)
            logging.info(f"Chunk {i+1} processed.")

        # Combine all chunks into a single DataFrame
        combined_df = pd.concat(all_chunks, ignore_index=True)
        combined_df.to_csv('train_data_extracted.csv', index=False)
        logging.info("Train data processed and saved as train_data_extracted.csv")

        # Process test data
        logging.info("Processing test data...")
        self.test_df['text'] = self.test_df['datasheet_link'].apply(lambda link: extractor.extract_text_from_pdf(link))
        self.test_df.to_csv('test_data_extracted.csv', index=False)
        logging.info("Test data processed and saved as test_data_extracted.csv")

if __name__ == "__main__":
    # Set up device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ocr_predictor(pretrained=True)
    model.to(device)

    # Initialize DataProcessor and run the pipeline
    processor = DataProcessor('data.xlsx', None, model)
    processor.load_data()
    processor.preprocess_data()
    processor.extract_text()

    logging.info("Pipeline completed.")
