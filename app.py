from src.dsproject.logger import logging
from src.dsproject.exception import CustonExecption
from src.dsproject.components.data_ingestion import DataIngestion,DataIngestionConfig
from src.dsproject.components.data_transformation import DataTransFormationConfig
from src.dsproject.components.data_transformation import Datatransfor
import sys



if __name__=="__main__":
    logging.info("The Execution has started")

    try:
        #data_ingestion_config=DataIngestionConfig()
        data_ingestion=DataIngestion()
        train_data_path,test_data_path= data_ingestion.initiate_data_ingestion()
        
        #data_transformation_config=DataTransFormationConfig()
        data_transformation=Datatransfor()
        data_transformation.initiate_data_trans(train_data_path,test_data_path)
        
    except Exception as e:
        logging.info("Custom Exception")
        raise CustonExecption(e,sys)