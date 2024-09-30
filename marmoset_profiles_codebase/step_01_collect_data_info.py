import argparse
import glob
import logging
import os

import nibabel
import numpy as np
import pandas as pd

from dataset_configuration import DatasetConfiguration

C_LOGGER_NAME = "data_collector"
logging.basicConfig(
    level=getattr(logging, "DEBUG"),
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(C_LOGGER_NAME)


def find_sections_range(ctx_thick_path):
    seg = nibabel.load(ctx_thick_path)
    segd = seg.get_fdata().astype(int)
    segd_mid = np.unique(np.where(segd == 3)[1])
    return segd_mid[0], segd_mid[-1]


def scoop_data(paths):
    cases = os.listdir(paths.input_dir)

    ddf = {"case": [], "stack_name": [], "st_section_idx": [], "end_section_idx": []}

    for case in cases:
        logger.info("Case... %s", case)
        ctx_paths = [paths.input_dir + "/" + "average_cortical_thickness_mask.nii.gz"]

        if len(ctx_paths) != 1:
            raise ValueError("Incomplete data!")
        fname = ctx_paths[0].rsplit("/", 1)[-1]
        stack_name = fname.rsplit("_")[0]
        st_sec_idx, end_sec_idx = find_sections_range(ctx_paths[0])

        ddf["case"].append(case)
        ddf["stack_name"].append(stack_name)
        ddf["st_section_idx"].append(st_sec_idx)
        ddf["end_section_idx"].append(end_sec_idx)

    output_df = pd.DataFrame(ddf)
    output_df.to_csv(paths.output_csv)
    logger.info("File has saved to... %s", paths.output_csv)


def parse_args():
    """ """
    parser = argparse.ArgumentParser(
        description=scoop_data.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input-dir",
        required=True,
        dest="input_dir",
        type=str,
        metavar="FILENAME",
        help="Path to the directory",
    )

    parser.add_argument(
        "-s",
        "--ctx-suffix",
        required=True,
        dest="ctx_suffix",
        type=str,
        metavar="FILENAME",
        help="",
    )

    parser.add_argument(
        "-o",
        "--output-csv",
        required=True,
        dest="output_csv",
        type=str,
        metavar="FILENAME",
        help="",
    )

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    input_options = parse_args()
    scoop_data(input_options)
