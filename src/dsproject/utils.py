import os
import sys
import pymysql
from src.dsproject.exception import CustonExecption
from src.dsproject.logger import logging
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

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
    
def evaluate_models(X_train, Y_train,X_test,Y_test,models,params):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=params[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,Y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,Y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(Y_train, y_train_pred)

            test_model_score = r2_score(Y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)