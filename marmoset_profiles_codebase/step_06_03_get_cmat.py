import argparse
import logging
import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

C_LOGGER_NAME = "cmat"
logging.basicConfig(
    level=getattr(logging, "DEBUG"),
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(C_LOGGER_NAME)

REGION_ORDER = [
    "Dorsolateral prefrontal",
    "Ventrolateral prefrontal",
    "Orbitofrontal",
    "Medial prefrontal",
    "Motor and premotor",
    "Insula and rostral lateral sulcus",
    "Somatosensory cortex",
    "Auditory",
    "Lateral and inferior temporal",
    "Ventral temporal",
    "Posterior cingulate, medial and retrosplenial",
    "Posterior parietal",
    "Visual cortex",
]


def read_data(paths):
    logger.info("Loading data...")

    # Loading .csv file with information about profiles
    profiles_df = pd.read_csv(paths.validation_csv)
    logger.info("%s", paths.validation_csv)

    logger.info("Loading data... Done!")

    return profiles_df


def check_all_areas(df, areas_df):
    diif_col = np.setdiff1d(pd.unique(areas_df.area_id), pd.unique(df.area_id))
    if len(diif_col) == 0:
        return []
    else:
        logger.info("Missing areas: %s", diif_col)
        miss_idx = areas_df[areas_df.area_id.isin(diif_col)].label.values
        logger.info("Missing idx: %s", miss_idx)
        return miss_idx


def process(paths):
    df = read_data(paths)

    df.dropna(inplace=True)
    y_real = np.array(df.label)

    req_order = pd.unique(df.label)

    y_pred = np.array(df["pred_y"])

    order_in_df = pd.unique(df.label)
    labels = [
        x for _, x in sorted(zip(req_order, order_in_df), key=lambda pair: pair[0])
    ]

    logger.info("LAAABELS %s", labels)
    order_path = os.path.join(paths.output, f"idx_in_model_order.npy")
    np.save(order_path, labels)

    confmat = confusion_matrix(y_true=y_real, y_pred=y_pred, labels=labels)

    logger.info("The confmat shape:")
    logger.info("%s", confmat.shape)

    # It is very ugly workaround
    if paths.binary:
        pass
    else:
        if confmat.shape[0] != 115:
            logger.error("Conusion matrix shape invalid!!!")

    cmat_path = os.path.join(paths.output, f"cmat.npy")
    np.save(cmat_path, confmat)
    logger.info("Results saved to... %s", cmat_path)

    if paths.binary:
        return 0

    region_list = pd.unique(df.region)

    regions_cmat_path = os.path.join(paths.output, "region_cmat")
    if os.path.exists(regions_cmat_path) is False:
        os.mkdir(regions_cmat_path)

    for region in region_list:
        logger.info("Cmat for region: %s", region)
        _reg_df = df[df.region == region]

        req_reg_order = pd.unique(_reg_df.label)
        order_in_reg_df = pd.unique(_reg_df.label)
        reg_labels = [
            x
            for _, x in sorted(
                zip(req_reg_order, order_in_reg_df), key=lambda pair: pair[0]
            )
        ]

        reg_cmat = confusion_matrix(y_true=y_real, y_pred=y_pred, labels=reg_labels)

        _reg = region.replace(" ", "_")
        _reg_cmat_path = os.path.join(regions_cmat_path, _reg + ".npy")

        np.save(_reg_cmat_path, reg_cmat)
        logger.info("Results saved to... %s", _reg_cmat_path)

    true_reg = df.region
    pred_reg = df.pred_region
    regxreg_confmat = confusion_matrix(
        y_true=true_reg, y_pred=pred_reg, labels=REGION_ORDER
    )

    reg_cmat_path = os.path.join(paths.output, f"regionxregion_cmat.npy")
    np.save(reg_cmat_path, regxreg_confmat)


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
