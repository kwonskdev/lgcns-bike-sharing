import os
import sys
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from src.common.constants import (
    ARTIFACT_PATH,
    DATA_PATH,
    LOG_FILEPATH,
    PREDICTION_PATH,
)
from src.common.logger import handle_exception, set_logger

logger = set_logger(os.path.join(LOG_FILEPATH, "logs.log"))
sys.excepthook = handle_exception
warnings.filterwarnings(action="ignore")

if __name__ == "__main__":
    DATE = datetime.now().strftime("%Y%m%d")
    logger.info("Loading the test data...")
    test = pd.read_csv(os.path.join(DATA_PATH, "bike_sharing_test.csv"))

    logger.info("Loading a pre-trained pipeline")
    model = joblib.load(os.path.join(ARTIFACT_PATH, "model.pkl"))

    X = test.drop(["datetime", "count"], axis=1, inplace=False)
    id_ = test["datetime"].to_numpy()

    logger.info("Saving a feature data for the test data...")
    model["preprocessor"].transform(X=X).to_csv(
        os.path.join(DATA_PATH, "storage", "bike_sharing_test_features.csv"),
        index=False,
    )

    pred_df = pd.DataFrame({"datetime": id_, "count": model.predict(X)})
    logger.info(f"Batch prediction for {len(pred_df)} times is created.")

    save_path = os.path.join(PREDICTION_PATH, f"{DATE}_count_prediction.csv")
    pred_df.to_csv(save_path, index=False)

    logger.info(f"Prediction can be found in the following path: \n{save_path}")
