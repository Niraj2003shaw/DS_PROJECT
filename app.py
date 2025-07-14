from src.dsproject.logger import logging
from src.dsproject.exception import CustonExecption
import sys



if __name__=="__main__":
    logging.info("The Execution has started")

    try:
        a=1/0
    except Exception as e:
        logging.info("Custom Exception")
        raise CustonExecption(e,sys)