import argparse
import json
import logging

import pandas as pd

from dataset_configuration import DatasetConfiguration

C_LOGGER_NAME = "report"
logging.basicConfig(
    level=getattr(logging, "DEBUG"),
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(C_LOGGER_NAME)


def read_data(paths):
    logger.info("Loading data...")

    # Loading .csv file with information about profiles
    logger.info("%s", paths.profiles_csv)
    prfoiles_df = pd.read_csv(paths.profiles_csv)

    logger.info("Loading data... Done!")

    return prfoiles_df


def process(config, paths):
    df = read_data(paths)
    report_dict = {"cases": list(pd.unique(df.case))}

    for case in report_dict["cases"]:
        df_case = df[df.case == case]
        trial_df = df_case.groupby("accept").size()
        df_case = df_case[df_case.accept]
        labels_list = list(map(int, sorted(pd.unique(df_case.area_id))))
        sections_list = list(map(int, sorted(pd.unique(df_case.section))))
        report_dict[case] = {
            "accepted_profiles": int(trial_df[True]),
            "rejected_profiles": int(trial_df[False] if False in trial_df else 0),
            "labels": labels_list,
            "sections": sections_list,
        }

    out_dict = {**config.settings_in_dict, "OUTPUT": {**report_dict}}
    with open(paths.output_report_path, "w") as f:
        json.dump(out_dict, f, indent=2)
    f.close()


def parse_args(doc_source):
    """ """
    parser = argparse.ArgumentParser(
        description=doc_source.__doc__,
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
        "-f",
        "--profiles-info-csv",
        required=True,
        dest="profiles_csv",
        type=str,
        metavar="FILENAME",
        help="Path to csv file with info about accepted profiles",
    )

    parser.add_argument(
        "-r",
        "--output-report",
        required=True,
        dest="output_report_path",
        type=str,
        metavar="FILENAME",
        help="Path to output json file with summary report",
    )

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    input_options = parse_args(process)
    data_settings = DatasetConfiguration(input_options.config_fname)
    process(data_settings, input_options)
