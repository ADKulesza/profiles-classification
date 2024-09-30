import argparse
import os
import re

import numpy as np

from plot_methods.plot_logger import get_logger

C_LOGGER_NAME = "mean_cmat"

RE_DIR = re.compile(".*holdout_\d*")
RE_SET = re.compile(".*_set_\d*$")


def read_data(paths):
    logger.info("Loading data...")

    confmat = np.array([])
    cmat_ctr = 0
    for root, dirs, files in os.walk(paths.evaluation_dir):
        if RE_DIR.match(root):
            if not RE_SET.match(root):
                continue
        else:
            continue

        cmat_fname = "cmat.npy"

        _path = os.path.join(root, cmat_fname)
        logger.info("PATH %s", _path)
        if confmat.shape[0] == 0:
            confmat = np.load(_path)
        else:
            confmat += np.load(_path)

        cmat_ctr += 1

    confmat = confmat / cmat_ctr

    logger.info("Loading data... Done!")

    return confmat


def process(paths):
    confmat = read_data(paths)

    np.save(paths.output, confmat)
    logger.info("Results saved to... %s", paths.output)


def parse_args():
    """ """
    parser = argparse.ArgumentParser(
        description=process.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-e",
        "--evaluation-dir",
        required=True,
        dest="evaluation_dir",
        type=str,
        metavar="FILENAME",
        help="Path to output npy file with",
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        dest="output",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    )

    arguments = parser.parse_args()

    return arguments


if __name__ == "__main__":
    logger = get_logger(C_LOGGER_NAME)
    input_options = parse_args()

    process(input_options)
