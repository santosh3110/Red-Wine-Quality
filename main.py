from src.red_wine_mlp import logger
from src.red_wine_mlp.pipeline.stage01_data_ingestion import DataIngestionTrainingPipeline
from src.red_wine_mlp.pipeline.stage02_data_validation import DataValidationTrainingPipeline
from src.red_wine_mlp.pipeline.stage03_data_transformation import DataTransformationTrainingPipeline
from src.red_wine_mlp.pipeline.stage04_model_trainer import ModelTrainingPipeline

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