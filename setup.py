import os
from pathlib import Path
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = 'mlops_pipeline'

list_of_files = [
    "data/raw/.gitignore",
    "data/processed/.gitignore",
    "models/.gitkeep",
    "scripts/data_preprocessing.py",
    "scripts/model_training.py",
    "notebooks/.gitkeep",
    "requirements.txt",
    "main.py",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir:
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")
