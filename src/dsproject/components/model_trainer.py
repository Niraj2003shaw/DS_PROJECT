import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
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
    # Path where the trained model will be saved
    trained_model_file_path: str = os.path.join("artifact", 'model.pkl')

class ModelTrainer:
    def __init__(self):
        # Initialize ModelTrainerConfig to get the model file path
        self.model_train_con = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info('Splitting training and test data into features (X) and target (Y)')
            # Separate features (X) and target (Y) from the training and test arrays
            X_train, Y_train, X_test, Y_test = (
                train_arr[:, :-1],  # All columns except the last for training features
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
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False), # verbose=False to suppress CatBoost output
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            # Define parameters for hyperparameter tuning (example, you might expand this)
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
            # Assuming evaluate_models is a function that trains and evaluates models
            # and returns a dictionary of model names to their R2 scores
            model_report: dict = evaluate_models(X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test, models=models, params=params)

            # Get the best model score from the evaluation report
            best_model_score = max(sorted(model_report.values()))

            # Get the name of the best model
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            # If the best model score is below a certain threshold, raise an exception
            if best_model_score < 0.6: # You can adjust this threshold
                raise CustonExecption("No best model found with sufficient R2 score")

            logging.info(f"Best model found: {best_model_name} with R2 score: {best_model_score}")

            # Save the best model using the save_object utility function
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
            # The sys.exc_info() is used to get details about the exception
            raise CustonExecption(e, sys)

