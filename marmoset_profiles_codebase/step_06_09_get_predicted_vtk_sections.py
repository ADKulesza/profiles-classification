import argparse
import glob
import logging
import os

import numpy as np
import pandas as pd
import vtk
from tqdm import tqdm

import vtk_core as vtk_core
from dataset_configuration import DatasetConfiguration

C_LOGGER_NAME = "generate samples"
logging.basicConfig(level=getattr(logging, "DEBUG"),
                    format='%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(C_LOGGER_NAME)


def read_data(paths):
    logger.info("Loading data...")

    profiles_df = pd.read_csv(paths.profiles_csv)
    logger.info("%s", paths.profiles_csv)

    logger.info("Loading data... Done!")

    return profiles_df


def fill_streamline_data(streamlines, streamline_id,
                         predictions, pred_conf, colors, pred_colors,
                         pred_area_id, pred_confidence,
                         true_area_color,
                         pred_area_color):
    predicted_area = int(pred_area_id)
    predicted_area_confidence = float(pred_confidence)

    predictions.SetTuple1(streamline_id, predicted_area)
    pred_conf.SetTuple1(streamline_id, predicted_area_confidence)

    colors.SetTuple3(streamline_id, *true_area_color)
    pred_colors.SetTuple3(streamline_id, *pred_area_color)

    streamlines.GetCellData().AddArray(predictions)
    streamlines.GetCellData().AddArray(pred_conf)
    streamlines.GetCellData().AddArray(colors)
    streamlines.GetCellData().AddArray(pred_colors)

    return streamlines


def process(paths, config):
    profiles_df = read_data(paths)

    sections = config("holdout_sections")

    sections_in_df = pd.unique(profiles_df.section)
    for section in sections:
        if section not in sections_in_df:
            raise ValueError(f"Section {section} does not occur in cases")

    # ----
    case_list = pd.unique(profiles_df.case)

    for case in case_list:
        case_df = profiles_df[profiles_df.case == case]

        for i_sec in sections:
            _df = case_df[case_df.section == i_sec]
            # pred_area_id = _df.pred_area_id.values
            # pred_conf_arr = _df.pred_confidence.values

            _input_path = os.path.join(paths.streamlines_dir, case)
            fn_input = glob.glob(
                _input_path + f'/*{i_sec}/*{i_sec}_streamlines_confidence.vtk')

            fn_input = fn_input[0]

            streamlines = vtk_core.load_vtk_polydata(fn_input)
            number_of_streamlines = int(streamlines.GetNumberOfCells())

            logger.info("Number of streamlines... %s", number_of_streamlines)
            #
            predictions = vtk.vtkUnsignedCharArray()
            predictions.SetNumberOfTuples(number_of_streamlines)
            predictions.SetNumberOfComponents(1)
            predictions.SetName("Predicted Segmentation")
            #
            pred_conf = vtk.vtkFloatArray()
            pred_conf.SetNumberOfTuples(number_of_streamlines)
            pred_conf.SetNumberOfComponents(1)
            pred_conf.SetName("Predicted Confidence")

            # Makes area good colors xD
            colors = vtk.vtkUnsignedCharArray()
            colors.SetNumberOfTuples(number_of_streamlines * 3)
            colors.SetNumberOfComponents(3)
            colors.SetName("Area Colors")
            #
            pred_colors = vtk.vtkUnsignedCharArray()
            pred_colors.SetNumberOfTuples(number_of_streamlines * 3)
            pred_colors.SetNumberOfComponents(3)
            pred_colors.SetName("Predicted Area Colors")

            id_array = np.arange(number_of_streamlines, dtype=np.uint)
            pred_id_array = pd.unique(_df.profile_id)
            ignored_ids = np.setdiff1d(id_array, pred_id_array)
            for index, row in tqdm(_df.iterrows()):
                streamline_id = row.profile_id

                true_area_color = (row.color_r, row.color_g, row.color_b)
                pred_area_color = (row.pred_color_r, row.pred_color_g, row.pred_color_b)

                streamlines = fill_streamline_data(streamlines, streamline_id,
                                                   predictions, pred_conf, colors, pred_colors,
                                                   row.pred_area_id, row.pred_confidence,
                                                   true_area_color,
                                                   pred_area_color)

            for streamline_id in ignored_ids:
                pred_area_id = int(-1)
                pred_confidence = 0.
                true_area_color = np.array([255, 255, 255], dtype=np.uint8)
                pred_area_color = np.array([255, 255, 255], dtype=np.uint8)

                streamlines = fill_streamline_data(streamlines, streamline_id,
                                                   predictions, pred_conf, colors, pred_colors,
                                                   pred_area_id, pred_confidence,
                                                   true_area_color,
                                                   pred_area_color)

            # "{:.1f}".format(x)
            fn_output = os.path.join(paths.output, "{:03d}_streamlines_confidence_pred.vtk".format(i_sec))
            vtk_core.save_vtk_polydata(fn_output,
                                       streamlines)


def parse_args():
    """
    """
    parser = argparse.ArgumentParser(
        description=process.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        "-c",
        "--config-fname",
        required=True,
        dest="config_fname",
        type=str,
        metavar="FILENAME",
        help="Path to file with configuration")

    parser.add_argument(
        "-p",
        "--profiles-csv",
        required=True,
        dest="profiles_csv",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    )

    parser.add_argument(
        "-s",
        "--streamlines-dir",
        required=True,
        dest="streamlines_dir",
        type=str,
        metavar="FILENAME",
        help="Path to json file with")

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
    dataset_settings = DatasetConfiguration(input_options.config_fname)
    process(input_options, dataset_settings)
