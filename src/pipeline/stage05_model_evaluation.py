from src.config.configuration import ConfigurationManager
from src.components.model_evaluation import ModelEvaluation
from src import logger

STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            model_eval_config = config.get_model_evaluation_config()
            model_eval_config = ModelEvaluation(config=model_eval_config)
            model_eval_config.save_results()
        except Exception as e:
            raise e