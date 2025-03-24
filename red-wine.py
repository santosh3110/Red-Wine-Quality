import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV,StratifiedShuffleSplit,RepeatedStratifiedKFold

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.linear_model import ElasticNet

from urllib.parse import urlparse
import mlflow.sklearn 
import mlflow
from mlflow.models.signature import infer_signature
import logging

import dagshub
dagshub.init(repo_owner='santoshkumarguntupalli', repo_name='Red-Wine-Quality', mlflow=True)

logging.basicConfig(level=logging.WARN)
logger=logging.getLogger(__name__)

def eval_metrics(actual,pred):
    rmse= np.sqrt(mean_squared_error(actual,pred))
    r2=r2_score(actual,pred)
    mae=mean_absolute_error(actual,pred)
    return rmse,mae,r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Read the wine-quality csv file from the URL
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    shuffle_split = StratifiedShuffleSplit(test_size=0.25,train_size=0.75,n_splits=35,random_state=42)
    
    with mlflow.start_run():

        param_grid = {
            'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 10, 50, 100],  #alpha values
            'l1_ratio': np.arange(0.0, 1.1, 0.05),  #range for l1_ratio
            'tol': [1e-06, 1e-05, 1e-04, 0.0001, 0.001, 0.01]  #range for tolerance
            }

        enet= ElasticNet(max_iter=1000)

        gs= GridSearchCV(enet,param_grid=param_grid,scoring="r2",cv = shuffle_split, return_train_score=True, n_jobs = -1)

        gs.fit(train_x,train_y)

        # alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
        # l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3

        alpha=gs.best_params_['alpha']
        l1_ratio=gs.best_params_['l1_ratio']
        tol=gs.best_params_['tol']

        predicted_qualities = gs.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha={:f}, l1_ratio={:f}), tolerance={:f}:".format(alpha,l1_ratio,tol))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_param("tol",tol)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        
        # For remote server only (Dagshub)
        remote_server_uri = "https://dagshub.com/santoshkumarguntupalli/Red-Wine-Quality.mlflow"
        mlflow.set_tracking_uri(remote_server_uri)



        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                gs, "model", registered_model_name="ElasticnetWineModel")
        else:
            mlflow.sklearn.log_model(gs, "model")