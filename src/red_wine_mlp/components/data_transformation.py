import pandas as pd
import os
from red_wine_mlp import logger
from sklearn.model_selection import train_test_split
from red_wine_mlp.entity.config_entity import DataTransformationConfig



class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_splitting(self):
        data = pd.read_csv(self.config.data_path)
        train,test=train_test_split(data,test_size=0.25)

        train.to_csv(os.path.join(self.config.root_dir,"train.csv"),index=False)
        test.to_csv(os.path.join(self.config.root_dir,"test.csv"), index=False)

        logger.info("train_test_split performed!")
        logger.info(f'Train_Data_Shape:{train.shape}')
        logger.info(f'Test_Data_Shape:{test.shape}')

        print(f'Train_Data_Shape:{train.shape}')
        print(f'Test_Data_Shape:{test.shape}')