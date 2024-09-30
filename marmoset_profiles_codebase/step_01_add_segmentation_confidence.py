#!/usr/bin/env python
# encoding: utf-8

import argparse
import copy
import logging

import numpy as np
import vtk

import vtk_core

C_LOGGER_NAME = ""
logging.basicConfig(
    level=getattr(logging, "DEBUG"),
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(C_LOGGER_NAME)

C_POINTDATA_SEGMENTATION = "Segmentation"
C_CELLDATA_PROBABILITY = "Segmentation Confidence"
C_CELLDATA_SEGMENTATION = "Segmentation"

C_HARD_SEG_POINT_DATA_CLEANUP = [C_POINTDATA_SEGMENTATION]


class StreamlinesSegmentation:
    """
    Compute hard (binary) segmentations of streamlines.

    The purpose of this class is to get
    streamlines_confidence.vtk file with hard segmentation
    and probability of it of every profile.

    Usage example:

    python3 add_segmentation_confidence.py \\
    --streamlines-input ~/atlas/0070/0070_streamlines.vtk
    --output-dir ~/atlas/0070/0070_streamlines_confidence.vtk
    """

    def __init__(self, polydata):
        self.__polydata = polydata
        self.__cells_no = int(polydata.GetNumberOfCells())

        self.__cell_data_arrays = {}

        self.__build_cell_data_arrays()
        self.__process_cell_data_arrays()

    def __call__(self):
        return copy.deepcopy(self.__process_cell_data_arrays())

    def __fill_cell_data(self, arr_dtype, components_no, arr_name):
        self.__cell_data_arrays[arr_name] = arr_dtype()
        self.__cell_data_arrays[arr_name].SetNumberOfComponents(components_no)
        self.__cell_data_arrays[arr_name].SetNumberOfTuples(self.__cells_no)
        self.__cell_data_arrays[arr_name].SetName(arr_name)

    def __build_cell_data_arrays(self):
        self.__fill_cell_data(vtk.vtkFloatArray, 1, C_CELLDATA_PROBABILITY)
        self.__fill_cell_data(vtk.vtkUnsignedCharArray, 1, C_CELLDATA_SEGMENTATION)

    def __get_point_seg(self, point_id):
        return (
            self.__polydata.GetPointData()
            .GetArray(C_POINTDATA_SEGMENTATION)
            .GetTuple1(point_id)
        )

    def __set_probability(self, cell, probability):
        return self.__cell_data_arrays[C_CELLDATA_PROBABILITY].SetTuple1(
            cell, probability
        )

    def __set_segmentation(self, cell, most_likely_area):
        return self.__cell_data_arrays[C_CELLDATA_SEGMENTATION].SetTuple1(
            cell, most_likely_area
        )

    def __generate_segmentation_probability(self):
        for i in range(self.__cells_no):
            cell = self.__polydata.GetCell(i)
            points_no = cell.GetNumberOfPoints()
            get_point_id = cell.GetPointIds().GetId

            # Get the coordinates of all points comprising given cell:
            points_to_test = map(
                lambda x: self.__get_point_seg(get_point_id(x)), range(points_no)
            )
            points_to_test = list(points_to_test)

            # Try to establish the most likely area and its probability.
            # Upon failure, just assume that we are outside the brain.
            values, counts = np.unique(points_to_test, return_counts=True)
            most_likely_area = values[np.argmax(counts)]
            probability = float(counts[np.argmax(counts)]) / float(points_no)

            self.__set_probability(i, probability)
            self.__set_segmentation(i, most_likely_area)

    def __process_cell_data_arrays(self):
        self.__generate_segmentation_probability()
        for cell_array_name in [C_CELLDATA_PROBABILITY, C_CELLDATA_SEGMENTATION]:
            self.__polydata.GetCellData().AddArray(
                self.__cell_data_arrays[cell_array_name]
            )

        for point_data_array_name in C_HARD_SEG_POINT_DATA_CLEANUP:
            self.__polydata.GetPointData().RemoveArray(point_data_array_name)

    def save(self, output_path):
        vtk_core.save_vtk_polydata(output_path, self.__polydata)


def process_profiles(arguments):
    streamlines = vtk_core.load_vtk_polydata(arguments.streamlines_path)
    logger.info("Getting streamlines confidence: %s", arguments.streamlines_path)
    hard_segmentation = StreamlinesSegmentation(streamlines)
    hard_segmentation.save(arguments.output)
    logger.info("Done! Results saved to: %s", arguments.output)


def parse_args():
    """ """
    parser = argparse.ArgumentParser(
        description=StreamlinesSegmentation.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-s",
        "--streamlines-input",
        required=True,
        dest="streamlines_path",
        type=str,
        metavar="FILENAME",
        help="Path to section streamlines",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        dest="output",
        type=str,
        metavar="FILENAME",
        help="Path to output confidence streamlines",
    )

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    input_options = parse_args()
    process_profiles(input_options)
