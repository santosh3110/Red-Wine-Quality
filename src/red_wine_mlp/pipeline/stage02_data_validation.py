from src.red_wine_mlp.config.configuration import ConfigurationManager
from src.red_wine_mlp.components.data_validation import DataValidation
from src.red_wine_mlp import logger

STAGE_NAME = "Data Validation Stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_validation_config = config.get_data_validation_config()
            data_validation = DataValidation(config=data_validation_config)
            data_validation.validate_all_columns()

        except Exception as e:
            raise e