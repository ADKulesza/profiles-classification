import argparse
import logging

import networkx as nx
import numpy as np
import pandas as pd

from dataset_configuration import DatasetConfiguration
from read_json import read_json
from step_00_graph_of_labels import graph_weight

C_LOGGER_NAME = "label_prob"
logging.basicConfig(
    level=getattr(logging, "DEBUG"),
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(C_LOGGER_NAME)


def read_data(paths):
    logger.info("Loading data...")

    # Loading .csv file with information about profiles
    profiles_df = pd.read_csv(paths.profiles_csv)
    logger.info("%s", paths.profiles_csv)

    # Loading .csv file with label assignment
    labels_df = pd.read_csv(paths.labels_idx)
    logger.info("%s", paths.labels_idx)

    data_graph = read_json(paths.graph)
    graph = nx.node_link_graph(data_graph)

    logger.info("Loading data... Done!")

    return profiles_df, labels_df, graph


def process(config, paths):
    logger.info("Get probabilities... ")
    profiles_df, labels_df, graph = read_data(paths)
    n_profiles = profiles_df.shape[0]
    model_area_id = config("specific_area_id_list")[0]
    prob_dict = {}
    areas_ctr = profiles_df.groupby("area_id").size()

    for _area in labels_df.area_id:
        area_w = graph_weight(
            graph, str(model_area_id), str(_area), config("reduce_label_impact")
        )

        # TODO look into
        if _area not in areas_ctr.keys():
            logger.warning("Area %s does not occur in dataset!!!", _area)
            continue
        else:
            prob_dict[_area] = area_w * areas_ctr[_area] / n_profiles

    prob_modulo = (
        1 - sum(np.array([*prob_dict.values()]) * areas_ctr.array)
    ) / n_profiles
    logger.info("Prob modulo... %s", prob_modulo)
    for _area, _prob in prob_dict.items():
        mod_prob = prob_dict[_area] + prob_modulo
        if mod_prob > 0:
            prob_dict[_area] = mod_prob

    profiles_prob = np.vectorize(lambda _a: prob_dict.get(_a))(profiles_df.area_id)
    profiles_df["prob"] = profiles_prob
    profiles_df = profiles_df.loc[:, ~profiles_df.columns.str.contains("^Unnamed")]
    profiles_df.to_csv(paths.output_prob_csv)
    logger.info("Get probabilities... Done!")
    logger.info("Results saved in... %s", paths.output_prob_csv)


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
        "-d",
        "--labels",
        required=True,
        dest="labels_idx",
        type=str,
        metavar="FILENAME",
        help="Path to csv file with info about area_id",
    )

    parser.add_argument(
        "-g",
        "--graph-path",
        required=True,
        dest="graph",
        type=str,
        metavar="FILENAME",
        help="Path to json file with graph of connected labels",
    )

    parser.add_argument(
        "-s",
        "--split-profiles-csv",
        required=True,
        dest="profiles_csv",
        type=str,
        metavar="FILENAME",
        help="Path to output csv file with labels dataset",
    )

    parser.add_argument(
        "-p",
        "--output-prob-split-profiles-csv",
        required=True,
        dest="output_prob_csv",
        type=str,
        metavar="FILENAME",
        help="Path to output npy file with",
    )

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    input_options = parse_args(process)
    data_settings = DatasetConfiguration(input_options.config_fname)
    process(data_settings, input_options)
