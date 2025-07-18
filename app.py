from src.dsproject.logger import logging
from src.dsproject.exception import CustonExecption
from src.dsproject.components.data_ingestion import DataIngestion
from src.dsproject.components.data_ingestion import DataIngestionConfig
import sys



if __name__=="__main__":
    logging.info("The Execution has started")

    try:
        #data_ingestion_config=DataIngestionConfig()
        data_ingestion=DataIngestion()
        data_ingestion.initiate_data_ingestion()
        
    except Exception as e:
        logging.info("Custom Exception")
        raise CustonExecption(e,sys)