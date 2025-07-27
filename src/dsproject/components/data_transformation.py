import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.dsproject.utils import save_object

from src.dsproject.exception import CustonExecption
from src.dsproject.logger import logging
import os




@dataclass
class DataTransFormationConfig:
    pre_obj_file_path=os.path.join('artifact','preprocessor.pkl')
    
class Datatransfor:
    def __init__(self):
        self.data_trans_config=DataTransFormationConfig()
        
    def get_data_trans_obj(self):
        #this function is for data transformation
        try:
            df = pd.read_csv(os.path.join('notebook\data','raw.csv'))
            X = df.drop(columns=['math score'],axis=1)
            num_features = X.select_dtypes(exclude="object").columns
            cat_features = X.select_dtypes(include="object").columns
            num_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
            ])
            cat_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
            ])
            
            logging.info(f"Categorical columns:{cat_features}")
            logging.info(f"Numerical column:{num_features}")
            
            
            preprocessor= ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,num_features),
                    ("cat_pipeline",cat_pipeline,cat_features)
                ]
            )
            return preprocessor
        
        
        except Exception as e:
            raise CustonExecption(e,sys)
    
    
    def initiate_data_trans(self,train_path,test_path):
        try:
            df = pd.read_csv(os.path.join('notebook\data','raw.csv'))
            X = df.drop(columns=['math score'],axis=1)
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info(f"Reading the train and test file")
            
            preprocessing_obj=self.get_data_trans_obj()
            
            target_col_name="math score"
            num_features = X.select_dtypes(exclude="object").columns
            
            ## divide the train dataset to independent and dependent dataset
            
            input_feature_train_df=train_df.drop(columns=[target_col_name],axis=1)
            target_feature_train_df=train_df[target_col_name]
            
            ## divide the test dataset to independent and dependent dataset
            
            input_feature_test_df=test_df.drop(columns=[target_col_name],axis=1)
            target_feature_test_df=test_df[target_col_name]
            
            logging.info(f"Applying Preprocessing on training and test dataframe")
            
            input_feature_train=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test=preprocessing_obj.transform(input_feature_test_df)
            
            train_arr= np.c_[
                input_feature_train,np.array(target_feature_train_df)
            ]
            test_arr= np.c_[
                input_feature_test,np.array(target_feature_test_df)
            ]
            
            logging.info(f"Preprocessing object")
            
            save_object(
                file_path=self.data_trans_config.pre_obj_file_path,
                obj=preprocessing_obj
            )
            
            return(
                train_arr,
                test_arr,
                self.data_trans_config.pre_obj_file_path
            )
            
        except Exception as e:
            raise CustonExecption(e,sys)