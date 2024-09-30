import argparse
import json
import logging

import networkx as nx
import nibabel
import numpy as np
from skimage.morphology import cube, dilation

C_LOGGER_NAME = "build_graph"
logging.basicConfig(
    level=getattr(logging, "DEBUG"),
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(C_LOGGER_NAME)


def read_data(seg_path):
    logger.info("Loading segmentation... %s", seg_path)
    seg = nibabel.load(seg_path)
    segd = seg.get_fdata()
    logger.info("Loading segmentation... Done!")

    return segd


def graph_weight(graph, start, goal, a=8):
    if start == "0" or goal == "0":
        return 1.0

    try:
        shortest = nx.shortest_path(graph, start, goal)
    except:
        logger.warning("Check input! Label %s or %s are not in graph", start, goal)
        return 0

    path_len = len(shortest) - 2
    weight = (a - path_len) / a
    if weight < 0:
        weight = 0
    return weight


def build_graph(seg_path, output_path):
    """
    Creates a graph representing the neighborhood between areas.
    Graph is created based on only one case.

    Parameters
    ----------
    seg_path : path to nifti segmentation
    output_path : path to jason file output path
    """

    segd = read_data(seg_path)
    labels = np.unique(segd)

    logger.info("Build graph...")

    graph = nx.Graph()

    for _l in labels[1:].astype(int):
        logger.info("Build graph... label %s", _l)
        mask = np.zeros(segd.shape)
        mask[segd == _l] = 1
        mask = dilation(mask, cube(3))
        connections = np.unique(segd[mask == 1]).astype(int)
        connections = connections[connections != 0]
        connections = connections[connections != _l]
        graph.add_edges_from([str(_l), str(c)] for c in connections)

    data = nx.node_link_data(graph)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
    f.close()

    logger.info("Build graph... Done! Result has saved to %s", output_path)


def parse_args():
    """
    Provides command-line interface
    """
    parser = argparse.ArgumentParser(
        description=build_graph.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-s",
        "--seg-path",
        required=True,
        dest="seg_path",
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
        help="Path to ",
    ),

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    input_options = parse_args()
    build_graph(input_options.seg_path, input_options.output)
