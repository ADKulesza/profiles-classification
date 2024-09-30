import argparse
import logging
import os
from sys import platform

import numpy as np
import pandas as pd
from sklearn import metrics

C_LOGGER_NAME = "metrics"
logging.basicConfig(
    level=getattr(logging, "DEBUG"),
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(C_LOGGER_NAME)


def read_data(paths):
    logger.info("Loading data...")

    # Loading .csv file with information about profiles
    profiles_df = pd.read_csv(paths.validation_csv)
    logger.info("%s", paths.validation_csv)

    logger.info("Loading data... Done!")

    return profiles_df


def check_all_areas(df, label_names, sort_map):
    diif_col = np.setdiff1d(label_names.area_id, df.area)
    if len(diif_col) == 0:
        return []
    else:
        miss_areas = label_names[label_names.area_id.isin(diif_col)].area
        miss_idx = sort_map[sort_map.index.isin(miss_areas)].values[0]
        return miss_idx


def process(paths):
    df = read_data(paths)
    df.dropna(inplace=True)

    y_true = np.array(df.idx_in_model)

    macro_metrics = {
        "dataset": [],
        "accuracy": [],
        "recall": [],
        "precision": [],
        "f1": [],
    }

    y_pred = np.array(df.pred_y)
    if platform == "linux":
        model_dataset = paths.validation_csv.split("/")[-2]
    else:
        model_dataset = paths.validation_csv.split("//")[-2]

    macro_metrics["dataset"].append(model_dataset)
    macro_metrics["accuracy"].append(metrics.accuracy_score(y_true, y_pred))
    macro_metrics["recall"].append(
        metrics.recall_score(y_true, y_pred, average="macro")
    )
    macro_metrics["precision"].append(
        metrics.precision_score(y_true, y_pred, average="macro")
    )
    macro_metrics["f1"].append(metrics.f1_score(y_true, y_pred, average="macro"))

    macro_metrics_df = pd.DataFrame(macro_metrics)
    macro_mtr_path = os.path.join(paths.output, f"macro_metrics.csv")
    macro_metrics_df.to_csv(macro_mtr_path)

    if paths.binary:
        return 0

    label_metrics = {
        "area": [],
        "area_id": [],
        "idx_in_model": [],
        "accuracy": [],
        "recall": [],
        "precision": [],
        "f1": [],
        "area_order": [],
        "region": [],
        "color_r": [],
        "color_g": [],
        "color_b": [],
    }

    area_list_gb = df.groupby(["area_order", "area", "area_id", "idx_in_model", "region",
                               "color_r", "color_g", "color_b"])

    logger.info("\nArea metrics processing...", )
    for area_tuple in area_list_gb:
        area_info = area_tuple[0]
        logger.info("Processing area... %s", area_info[1])

        label_metrics["area_order"].append(area_info[0])
        label_metrics["area"].append(area_info[1])
        label_metrics["area_id"].append(area_info[2])
        label_metrics["idx_in_model"].append(area_info[3])
        label_metrics["region"].append(area_info[4])
        label_metrics["color_r"].append(area_info[5])
        label_metrics["color_g"].append(area_info[6])
        label_metrics["color_b"].append(area_info[7])

        _y_true = np.zeros(y_true.shape[0])
        _y_true[y_true == area_info[3]] = 1

        _y_pred = np.zeros(y_true.shape[0])
        _y_pred[y_pred == area_info[3]] = 1

        label_metrics["accuracy"].append(metrics.accuracy_score(_y_true, _y_pred))
        label_metrics["f1"].append(metrics.f1_score(_y_true, _y_pred))
        label_metrics["recall"].append(metrics.recall_score(_y_true, _y_pred))
        label_metrics["precision"].append(metrics.precision_score(_y_true, _y_pred))

    logger.info("Area metrics processing... Done!\n", )
    label_mtr_df = pd.DataFrame(label_metrics)
    label_mtr_path = os.path.join(paths.output, f"area_metrics.csv")
    label_mtr_df.to_csv(label_mtr_path)

    logger.info("Results saved to... %s", macro_mtr_path)


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
