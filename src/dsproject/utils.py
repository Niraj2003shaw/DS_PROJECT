import os
import sys
import pymysql
from src.dsproject.exception import CustonExecption
from src.dsproject.logger import logging
import pandas as pd
from dotenv import load_dotenv

import pickle
import numpy as np



load_dotenv()

host=os.getenv("host")
user=os.getenv("user")
passs=os.getenv("password")
db=os.getenv("db")

def read_sql_data():
    logging.info("Reading Sql database started")
    mydb = None # Initialize mydb outside try to ensure it's defined for finally block
    try:
        mydb=pymysql.connect(
            host=host,
            user=user,
            password=passs,
            db=db
        )
        # --- FIX IS HERE ---
        # Change this line:
        # logging.info("Connection established",mydb)
        # To one of these:
        logging.info("Connection established successfully.") # Cleanest option
        # Or if you want to see the connection object's representation:
        # logging.info(f"Connection established: {mydb}")
        # -------------------

        df=pd.read_sql_query("Select *from Students", con=mydb)

        print(df.head())
        return df

    except Exception as ex:
        raise CustonExecption(ex,sys)
    finally:
        if mydb is not None and mydb.open: # Check if mydb was created and is open
            mydb.close()
            logging.info("Database connection closed.")
            
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj,file_obj)
    
    except Exception as e:
        raise CustonExecption(e,sys)