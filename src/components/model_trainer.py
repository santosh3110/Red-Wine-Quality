import os
from catboost import CatBoostRegressor
import joblib
from src import logger
import pandas as pd
from src.config.configuration import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        
        X_train=train_data.drop([self.config.target_column],axis=1)
        X_test=test_data.drop([self.config.target_column], axis=1)
        y_train=train_data[self.config.target_column]
        y_test=test_data[self.config.target_column]


        catr = CatBoostRegressor(iterations=self.config.iterations, depth=self.config.depth, learning_rate=self.config.learning_rate, random_state=42)
        catr.fit(X_train,y_train)

        joblib.dump(catr,os.path.join(self.config.root_dir,self.config.model_name))