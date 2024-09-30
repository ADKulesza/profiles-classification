import argparse
import os

import nibabel
import numpy as np
import pandas as pd

from dataset_configuration import DatasetConfiguration


def generate_label_file(config, input_path, output_path):
    """
    Generate csv file with list of area_id in all cases

    Parameters
    ----------
    config : instance of DatasetConfiguration class
    input_path : path to segmentation stack (nifti)
    output_path : path to output csv file
    """

    seg_path = input_path
    seg = nibabel.load(seg_path)
    segd = seg.get_fdata().astype(int)
    labels = np.unique(segd)

    if os.path.exists(output_path):
        _df = pd.read_csv(output_path)
        df_labels = np.array(_df.area_id, dtype=int)
        labels = np.union1d(df_labels, labels)

    if config("do_exclude_zero"):
        labels = labels[labels != 0]

    labels = np.sort(labels)
    print(labels)
    labels_dict = {
        "area_id": [*labels],
        "label": [None] * labels.size,
        "idx_in_model": [None] * labels.size,
    }

    df = pd.DataFrame(labels_dict)
    df.to_csv(output_path, index=False)


def parse_args():
    """
    Provides command-line interface
    """
    parser = argparse.ArgumentParser(
        description=generate_label_file.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-c",
        "--config-fname",
        required=True,
        dest="config_fname",
        type=str,
        metavar="FILENAME",
        help="Path to file with configuration",
    )

    parser.add_argument(
        "-i",
        "--input-dir",
        required=True,
        dest="input_dir",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    ),

    parser.add_argument(
        "-o",
        "--output-path",
        required=True,
        dest="output",
        type=str,
        metavar="FILENAME",
        help="Path to output csv file with labels",
    )

    arguments = parser.parse_args()

    return arguments


if __name__ == "__main__":
    input_options = parse_args()
    data_settings = DatasetConfiguration(input_options.config_fname)
    generate_label_file(data_settings, input_options.input_dir, input_options.output)
