from src import logger
from src.pipeline.stage01_data_ingestion import DataIngestionTrainingPipeline
from src.pipeline.stage02_data_validation import DataValidationTrainingPipeline
from src.pipeline.stage03_data_transformation import DataTransformationTrainingPipeline
from src.pipeline.stage04_model_trainer import ModelTrainingPipeline
from src.pipeline.stage05_model_evaluation import ModelEvaluationPipeline

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>> stage {STAGE_NAME} started <<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx=====x")

except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Validation Stage"

try:
    logger.info(f">>>> stage {STAGE_NAME} started <<<<")
    data_validation = DataValidationTrainingPipeline()
    data_validation.main()
    logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx=====x")

except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Transformation Stage"

try:
    logger.info(f">>>> stage {STAGE_NAME} started <<<<")
    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.main()
    logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx=====x")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Training Stage"

try:
    logger.info(f">>>> stage {STAGE_NAME} started <<<<")
    model_trainer = ModelTrainingPipeline()
    model_trainer.main()
    logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx=====x")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Evaluation Stage"

try:
    logger.info(f">>>> stage {STAGE_NAME} started <<<<")
    model_evaluation = ModelEvaluationPipeline()
    model_evaluation.main()
    logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx=====x")

except Exception as e:
    logger.exception(e)
    raise e