import os
from sklearn.metrics import root_mean_squared_error,mean_absolute_error,r2_score
import joblib
from src import logger
import pandas as pd
from src.config.configuration import ModelEvaluationConfig
from src.utils.common import save_json
from pathlib import Path

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self,actual,pred):
        rmse= root_mean_squared_error(actual,pred)
        mae=mean_absolute_error(actual,pred)
        r2=r2_score(actual,pred)
        return rmse,mae,r2
    
    def save_results(self):

        test_data= pd.read_csv(self.config.test_data_path)
        model= joblib.load(self.config.model_path)

        test_x= test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        predicted_qualities = model.predict(test_x)

        (rmse,mae,r2) = self.eval_metrics(test_y,predicted_qualities)

        scores= {"RMSE": rmse, "MAE" : mae, "R2_Score": r2 }
        save_json(path=Path(self.config.metric_file_name), data=scores)
