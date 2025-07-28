from src.dsproject.logger import logging
from src.dsproject.exception import CustonExecption
from src.dsproject.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.dsproject.components.data_transformation import DataTransFormationConfig
from src.dsproject.components.data_transformation import Datatransfor
from src.dsproject.components.model_trainer import ModelTrainer,ModelTrainerConfig # Import ModelTrainer
import sys


if __name__ == "__main__":
    logging.info("The Execution has started")

    try:
        # --- Data Ingestion Step ---
        logging.info("Starting Data Ingestion...")
        data_ingestion = DataIngestion()
        # This method should return the paths to the raw train and test data
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data Ingestion completed. Train data at: {train_data_path}, Test data at: {test_data_path}")

        # --- Data Transformation Step ---
        logging.info("Starting Data Transformation...")
        data_transformation = Datatransfor()
        # This method should now return the preprocessed train and test arrays
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_trans(
            train_data_path,
            test_data_path
        )
        logging.info("Data Transformation completed. Transformed data arrays are ready.")

        # --- Model Training Step ---
        logging.info("Starting Model Training...")
        model_trainer = ModelTrainer()
        # Pass the transformed arrays to the model trainer
        r2_score_value = model_trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info(f"Model Training completed. Best model R2 Score: {r2_score_value}")

    except Exception as e:
        logging.error(f"An error occurred during pipeline execution: {e}")
        raise CustonExecption(e, sys)

