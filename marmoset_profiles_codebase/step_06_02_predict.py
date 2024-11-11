import argparse
import logging
import os

import numpy as np
import pandas as pd
from tensorflow import compat
from tensorflow.keras.models import load_model

from cnn_models.simple_model import simple_model
from cnn_models.multi_branch_binary_model import multi_branch_binary_model
from read_json import read_json

C_LOGGER_NAME = "get_holdout"
logging.basicConfig(
    level=getattr(logging, "DEBUG"),
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(C_LOGGER_NAME)


def read_data(paths):
    logger.info("Loading data...")

    # Loading array with profiles
    profiles = np.load(paths.profiles_npy)
    logger.info("%s", paths.profiles_npy)

    # Loading .csv file with information about profiles
    profiles_df = pd.read_csv(paths.profiles_csv)
    logger.info("%s", paths.profiles_csv)

    model_order = read_json(paths.models_order)
    models_df = pd.read_csv(paths.models)

    models_path = models_df[models_df.set.isin(model_order[paths.holdout_id])].path

    models_dict = {}
    for set_id, m_path in zip(models_path, model_order[paths.holdout_id]):
        models_dict[set_id] = m_path

    logger.info("Loading data... Done!")

    return profiles, profiles_df, models_dict, models_df


def process(paths):
    profiles, profiles_df, models_dict, models_df = read_data(paths)

    x_test = profiles.reshape((profiles.shape[0], profiles.shape[1], 1))

    for m_path, set_id in models_dict.items():
        logger.info("Model path: %s", m_path)

        set_no = m_path.split("/")[-1]

        output_path = os.path.join(paths.output, paths.holdout_id, set_no)
        logger.info("Output dir: %s", output_path)

        if os.path.exists(output_path) is False:
            os.mkdir(output_path)

        pred_path = os.path.join(output_path, f"pred_y.npy")

        logger.info("\nCLASSIFING...\n")
        model = load_model(m_path)
        predictions = model.predict(x_test)
        logger.info("\nCLASSIFING... Done!\n")

        np.save(pred_path, predictions)
        logger.info("Predictions have saved in: %s", pred_path)

        predicted_classes = np.argmax(predictions, axis=1)
        predicted_confidence = np.max(predictions, axis=1)

        profiles_df[f"pred_y"] = predicted_classes
        profiles_df[f"pred_confidence"] = predicted_confidence

        del model


    profiles_df = profiles_df.loc[:, ~profiles_df.columns.str.contains("^Unnamed")]

    gb = profiles_df.groupby("area")
    profiles_df["pred_area"] = np.nan
    for _area, _df in gb:
        _pred_areas = pd.unique(_df.label)
        profiles_df.loc[profiles_df.pred_y.isin(_pred_areas), "pred_area"] = _area

    gb = profiles_df.groupby("region")
    profiles_df["pred_region"] = np.nan
    for _reg, _df in gb:
        _pred_reg = pd.unique(_df.label)
        profiles_df.loc[profiles_df.pred_y.isin(_pred_reg), "pred_region"] = _reg

    df_path = os.path.join(output_path, "results.csv")

    profiles_df.to_csv(df_path)
    logger.info("Dataframe has saved in: %s", pred_path)


def parse_args():
    """ """
    parser = argparse.ArgumentParser(
        description=process.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-x",
        "--profiles",
        required=True,
        dest="profiles_npy",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    )

    parser.add_argument(
        "-i",
        "--holdout-id",
        required=True,
        dest="holdout_id",
        type=str,
        metavar="FILENAME",
        help="Path to  directory",
    )

    parser.add_argument(
        "-c",
        "--profiles-csv",
        required=True,
        dest="profiles_csv",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    )

    parser.add_argument(
        "-m",
        "--models-info",
        required=True,
        dest="models",
        type=str,
        metavar="FILENAME",
        help="Path to  directory",
    )

    parser.add_argument(
        "-j",
        "--models-order",
        required=True,
        dest="models_order",
        type=str,
        metavar="FILENAME",
        help="Path to output",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        dest="output",
        type=str,
        metavar="FILENAME",
        help="Path to output directory",
    )

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    # tf_config = compat.v1.ConfigProto()
    # tf_config.gpu_options.per_process_gpu_memory_fraction = (
    #     0.33  # 0.33  # 0.6 sometimes works better for folks
    # )
    # compat.v1.keras.backend.set_session(compat.v1.Session(config=tf_config))
    # os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    input_options = parse_args()
    process(input_options)
