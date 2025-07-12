import os
from pathlib import Path
import logging
import sys

# --- Configuration for Logging to Terminal ---
# We configure the logger to ensure messages are outputted to the console (sys.stdout).
# This addresses the issue where logging.basicConfig() might not display output in some environments.

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create a StreamHandler that writes to standard output (the terminal)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)

# Define the format of the log messages
formatter = logging.Formatter('%(levelname)s: %(message)s')
handler.setFormatter(formatter)

# Add the handler to the logger
if not logger.handlers:
    logger.addHandler(handler)
# ---------------------------------------------


pro_name="dsproject"

list_file=[
    f"src/{pro_name}/__init__.py",
    f"src/{pro_name}/components/__init__.py",
    f"src/{pro_name}/components/data_ingestion.py",
    f"src/{pro_name}/components/data_transformation.py",
    f"src/{pro_name}/components/model_trainer.py",
    f"src/{pro_name}/components/model_monitoring.py",
    f"src/{pro_name}/pipeline/__init__.py",
    f"src/{pro_name}/pipeline/training_pipeline.py",
    f"src/{pro_name}/exception.py",
    f"src/{pro_name}/logger.py",
    f"src/{pro_name}/utils.py",
    "main.py"
    "app.py",
    "Dockerfile"
]

for filepath in list_file:
    # Use Pathlib for easier path handling and normalization
    filepath = Path(filepath)
    # Extract directory and filename
    fildir,filname = os.path.split(filepath)

    # 1. Create directories if they don't exist
    if fildir != "":
        os.makedirs(fildir, exist_ok=True)
        logging.info(f"Creating directory: {fildir} for the file {filname}")

    # 2. Create the file if it doesn't exist or is empty
    # We use the corrected syntax with parentheses for os.path.exists() and os.path.getsize()
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath,'w') as f:
            pass # Creates an empty file
            logging.info(f"Creating empty file: {filepath}")

    # 3. Log if the file already exists and is not empty
    else:
        logging.info(f"{filname} is already exists")