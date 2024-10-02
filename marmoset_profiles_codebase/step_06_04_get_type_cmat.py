import argparse
import logging
import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

C_LOGGER_NAME = "type_cmat"
logging.basicConfig(
    level=getattr(logging, "DEBUG"),
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(C_LOGGER_NAME)

AREAS_TYPES = [
    "Kon",
    "Eu3",
    "Eu2",
    "Eu1",
    "Dys",
    "Agr"
]


def read_data(paths):
    logger.info("Loading data...")

    # Loading .csv file with information about profiles
    profiles_df = pd.read_csv(paths.validation_csv)
    logger.info("%s", paths.validation_csv)

    labels_processed = pd.read_csv(paths.labels_processed)
    logger.info("%s", paths.labels_processed)

    logger.info("Loading data... Done!")

    return profiles_df, labels_processed


def check_all_areas(df, areas_df):
    diif_col = np.setdiff1d(pd.unique(areas_df.area_id), pd.unique(df.area_id))
    if len(diif_col) == 0:
        return []
    else:
        logger.info("Missing areas: %s", diif_col)
        miss_idx = areas_df[areas_df.area_id.isin(diif_col)].idx_in_model.values
        logger.info("Missing idx: %s", miss_idx)
        return miss_idx


def process(paths):
    df, labels_processed = read_data(paths)

    df.dropna(inplace=True)
    y_real = np.array(df.type_id, dtype='int8')

    req_order = pd.unique(df.area_order)

    y_pred = np.array(df.pred_type_id, dtype='int8')

    labels = np.unique(y_real)

    confmat = confusion_matrix(y_true=y_real, y_pred=y_pred, labels=labels)

    confmat_path = os.path.join(paths.output, "typesxtypes_cmat.npy")
    np.save(confmat_path, confmat)
    logger.info("Confusion matrix saved... %s", confmat_path)

    y_real = df.idx_in_model
    y_pred = df.pred_y
    for area_type in AREAS_TYPES:
        _type_df = df[df.type == area_type]

        req_type_order = pd.unique(_type_df.area_order)
        order_in_type_df = pd.unique(_type_df.idx_in_model)
        type_labels = [
            x
            for _, x in sorted(
                zip(req_type_order, order_in_type_df), key=lambda pair: pair[0]
            )
        ]

        logger.info("Area type: %s", area_type)
        logger.info("LABELS:")
        logger.info("%s", type_labels)

        type_cmat = confusion_matrix(y_true=y_real, y_pred=y_pred, labels=type_labels)
        logger.info("The confmat shape:")
        logger.info("%s", type_cmat.shape)

        confmat_path = os.path.join(paths.output, f"{area_type}_cmat.npy")
        np.save(confmat_path, type_cmat)
        logger.info("Confusion matrix saved... %s", confmat_path)


def parse_args():
    """ """
    parser = argparse.ArgumentParser(
        description=process.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-v",
        "--validation-csv",
        required=True,
        dest="validation_csv",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    )

    parser.add_argument(
        "-l",
        "--labels-processed",
        required=True,
        dest="labels_processed",
        type=str,
        metavar="FILENAME",
        help="Path to output csv file with ",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        dest="output",
        type=str,
        metavar="FILENAME",
        help="Path to output npy file with",
    )

    parser.add_argument(
        "-e",
        "--one-vs-all",
        required=False,
        action="store_true",
        dest="binary",
        help="Path to output directory",
    )

    arguments = parser.parse_args()

    return arguments


if __name__ == "__main__":
    input_options = parse_args()
    process(input_options)
