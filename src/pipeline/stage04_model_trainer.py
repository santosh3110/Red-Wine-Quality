from src.config.configuration import ConfigurationManager
from src.components.model_trainer import ModelTrainer
from src import logger

STAGE_NAME = "Model Training Stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config()
            model_trainer_config = ModelTrainer(config=model_trainer_config)
            model_trainer_config.train()
        except Exception as e:
            raise e