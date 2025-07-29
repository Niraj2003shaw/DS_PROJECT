import os
import sys
from dataclasses import dataclass
from urllib.parse import urlparse # Still useful for debugging/understanding URI type
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor, # Added to models dictionary for consistency
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Assuming these are correctly defined in your project structure
from src.dsproject.exception import CustonExecption
from src.dsproject.logger import logging
# Assuming you have a utils.py with a save_object function
from src.dsproject.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    # Path where the trained model will be saved locally
    trained_model_file_path: str = os.path.join("artifact", 'model.pkl')
    # Path where the MLflow logged model will be temporarily saved before logging as artifact
    mlflow_model_temp_path: str = os.path.join("artifact", 'mlflow_model_temp')


class ModelTrainer:
    def __init__(self):
        # Initialize ModelTrainerConfig to get the model file paths
        self.model_train_con = ModelTrainerConfig()
    
    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    
    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info('Splitting training and test data into features (X) and target (Y)')
            # Separate features (X) and target (Y) from the training and test arrays
            X_train, Y_train, X_test, Y_test = (
                train_arr[:, :-1],   # All columns except the last for training features
                train_arr[:,-1],   # Last column for training target
                test_arr[:, :-1],   # All columns except the last for testing features
                test_arr[:, -1],    # Last column for testing target
            )

            # Define a dictionary of models to be trained and evaluated
            models = {
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(), # Added GradientBoostingRegressor to models
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False), # verbose=False to suppress CatBoost output
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            # Define parameters for hyperparameter tuning
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest Regressor": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [.1, .05, .02, .01],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [.1, .05, .02, .01],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .05, .02, .01],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "K-Neighbors Regressor": {}
            }

            logging.info('Evaluating models...')
            # Evaluate each model and get their performance scores
            model_report: dict = evaluate_models(X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test, models=models, params=params)

            # Get the best model score from the evaluation report
            best_model_score = max(sorted(model_report.values()))

            # Get the name of the best model
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            print("This is the Best model::")
            print(best_model_name)
            
            # Ensure `actual_model` correctly maps to the best model's parameters
            # This loop assumes a perfect match between model names and param keys.
            actual_model = ""
            for model_key in params.keys():
                if best_model_name == model_key:
                    actual_model = model_key
                    break
            
            best_params = params[actual_model] # Get parameters for the best model

            # IMPORTANT: Remove mlflow.set_registry_uri if you are not using MLflow Model Registry features
            # and are only logging artifacts to DagsHub's tracking server.
            # The MLFLOW_TRACKING_URI environment variable should be set externally for this.
            # mlflow.set_registry_uri("https://dagshub.com/Niraj2003shaw/DS_PROJECT.mlflow")
            
            # The tracking URI should be set as an environment variable (MLFLOW_TRACKING_URI)
            # as discussed in previous responses. This line is just to get the scheme for the check.
            tracking_uri = mlflow.get_tracking_uri()
            tracking_url_type_store = urlparse(tracking_uri).scheme
            logging.info(f"MLflow Tracking URI: {tracking_uri}, Scheme: {tracking_url_type_store}")

            
            # Start an MLflow run to log metrics, parameters, and the model
            with mlflow.start_run():
                predicted_qualities = best_model.predict(X_test)
                (rmse, mae, r2) = self.eval_metrics(Y_test, predicted_qualities)

                mlflow.log_params(best_params)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)

                # Explicitly save the model locally first, then log it as a generic artifact.
                # This bypasses the Model Registry API which seems to be causing the error.
                # Create the directory if it doesn't exist
                os.makedirs(self.model_train_con.mlflow_model_temp_path, exist_ok=True)
                
                # Save the model in MLflow format to a temporary local path
                mlflow.sklearn.save_model(
                    sk_model=best_model, 
                    path=self.model_train_con.mlflow_model_temp_path
                )
                
                # Log the entire directory containing the saved model as an artifact
                mlflow.log_artifacts(
                    local_dir=self.model_train_con.mlflow_model_temp_path, 
                    artifact_path="best_model" # This will be the folder name in DagsHub artifacts
                )
                logging.info(f"Model logged as artifact to DagsHub at: {mlflow.active_run().info.artifact_uri}/best_model")

            # If the best model score is below a certain threshold, raise an exception
            if best_model_score < 0.6: # You can adjust this threshold based on your project needs
                raise CustonExecption("No best model found with sufficient R2 score")

            logging.info(f"Best model found: {best_model_name} with R2 score: {best_model_score}")

            # Save the best model locally using the save_object utility function
            save_object(
                file_path=self.model_train_con.trained_model_file_path,
                obj=best_model
            )

            # Make predictions with the best model on the test set
            predicted = best_model.predict(X_test)

            # Calculate the R2 score for the best model's predictions
            r2_square = r2_score(Y_test, predicted)
            logging.info(f"R2 score of the best model on test data: {r2_square}")

            return r2_square

        except Exception as e:
            # Catch any exceptions and raise a custom exception
            raise CustonExecption(e, sys)

