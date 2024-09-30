#!/usr/bin/env python
# encoding: utf-8

import argparse
import copy
import glob
import logging
import os

import numpy as np

import vtk_core
from dataset_configuration import DatasetConfiguration
from read_json import read_json

C_LOGGER_NAME = "generate_streamlines_profiles"
logging.basicConfig(
    level=getattr(logging, "DEBUG"),
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(C_LOGGER_NAME)

C_POINTDATA_IMAGE = "Image"
C_POINTDATA_NORM_THICKNESS = "Normalized thickness"
C_CELLDATA_PROBABILITY = "Segmentation Confidence"
C_CELLDATA_SEGMENTATION = "Segmentation"


class NormalizedProfiles:
    """
    Getting cortical profiles.

    The purpose of this class is to get npy files.
    First with point values of normalized profiles
    and second with assigned segemntation value
    to the profile

    Usage example:

    python3 extract_profiles.py \\
    --confidence-input ~/atlas/0070/0070_streamlines_confidence.vtk \\
    --output-dir ~/atlas/0070/
    """

    def __init__(self, streamlines, profile_length, profile_domain):
        self.__streamlines = streamlines
        self.__profile_length = profile_length
        self.__profile_domain = profile_domain

        self.__streamlines_no = int(self.__streamlines.GetNumberOfCells())

        # An alias / reference
        self.__get_point_image_value = (
            self.__streamlines.GetPointData().GetArray(C_POINTDATA_IMAGE).GetTuple1
        )
        self.__get_point_normt_value = (
            self.__streamlines.GetPointData()
            .GetArray(C_POINTDATA_NORM_THICKNESS)
            .GetTuple1
        )

        self.__segmentation_arr = np.zeros((self.__streamlines_no, 2), float)
        self.__interpolated_profile = np.zeros(
            (self.__streamlines_no, self.__profile_length)
        )

        self._process_all_streamlines()

    def _process_all_streamlines(self):

        for streamline_id in range(self.__streamlines_no):
            streamline = self.__streamlines.GetCell(streamline_id)
            self.__interpolated_profile[streamline_id, :] = self.__get_point_values(
                streamline
            )
            self.__get_streamline_segmentation(streamline_id)

    def __get_point_values(self, streamline):
        points_no = streamline.GetNumberOfPoints()
        current_profile = np.zeros(points_no)
        current_normthick = np.zeros(points_no)
        for point_id in range(points_no):
            streamline_point = streamline.GetPointIds().GetId(point_id)
            current_profile[point_id] = self.__get_point_image_value(streamline_point)
            current_normthick[point_id] = self.__get_point_normt_value(streamline_point)

        interpolated_profile = np.interp(
            self.__profile_domain, current_normthick, current_profile
        )

        return interpolated_profile

    def __get_streamline_segmentation(self, streamline_id):
        area_id = int(
            self.__streamlines.GetCellData()
            .GetArray(C_CELLDATA_SEGMENTATION)
            .GetTuple(streamline_id)[0]
        )
        area_confidence = (
            self.__streamlines.GetCellData()
            .GetArray(C_CELLDATA_PROBABILITY)
            .GetTuple(streamline_id)[0]
        )
        #
        self.__segmentation_arr[streamline_id] = (area_id, area_confidence)

    @property
    def profiles(self):
        return copy.deepcopy(self.__interpolated_profile)

    @property
    def segmentation(self):
        return copy.deepcopy(self.__segmentation_arr)


class SaveProfiles:
    """
    Saves profiles in numpy arrays

    Arguments:
    dir_path -- path to directory with section's contents
                and where the npy profiles will saved
    norm_profiles -- instance of class NormalizedProfiles
    """

    def __init__(self, dir_path, norm_profiles: NormalizedProfiles):
        self.__path = dir_path
        self.__profiles = norm_profiles.profiles
        self.__segmentation = norm_profiles.segmentation

        self.__save()

    def __get_section_number(self):
        json_path = glob.glob(self.__path + "/*.json")[0]
        data = read_json(json_path)

        prefix = data["sections"][0]["output_prefix"]
        return prefix

    def __save(self):
        prefix = self.__get_section_number()
        np.save(
            os.path.join(self.__path, f"{prefix}_norm_profiles.npy"), self.__profiles
        )
        np.save(
            os.path.join(self.__path, f"{prefix}_segmentation.npy"), self.__segmentation
        )
        logger.info("Done! Results saved to: %s", self.__path)


def process_all_profiles(arguments):
    logger.info("Processing profiles: %s", arguments.probability)
    streamlines = vtk_core.load_vtk_polydata(arguments.probability)

    conf_handler = DatasetConfiguration(arguments.config_fname)

    profile_len = conf_handler("profile_length")

    profile_domain = np.linspace(0.0, 1.0, profile_len)
    norm_profiles = NormalizedProfiles(streamlines, profile_len, profile_domain)
    SaveProfiles(arguments.output, norm_profiles)


def parse_args():
    """ """
    parser = argparse.ArgumentParser(
        description=NormalizedProfiles.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-s",
        "--confidence-input",
        required=True,
        dest="probability",
        type=str,
        metavar="FILENAME",
        help="Path to section streamlines confidence",
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
        "-o",
        "--output-dir",
        required=True,
        dest="output",
        type=str,
        metavar="FILENAME",
        help="Path to output directory",
    )

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    input_options = parse_args()
    process_all_profiles(input_options)
