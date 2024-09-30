import argparse
import logging
import os

import numpy as np
import pandas as pd
import vtk
from tqdm import tqdm

import vtk_core as vtk_core
from dataset_configuration import DatasetConfiguration

C_LOGGER_NAME = "generate vtk"
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

    logger.info("Loading data... Done!")

    return profiles_df


def load_vtk_section(vtk_dir, case, i_section):
    # TODO 0
    fn_input = os.path.join(vtk_dir, case, str(i_section), f"{i_section}_streamlines_confidence.vtk")

    logger.info("Loading vtk section... %s", fn_input)
    streamlines = vtk_core.load_vtk_polydata(fn_input)

    number_of_streamlines = int(streamlines.GetNumberOfCells())
    logger.info("Number of streamlines... %s", number_of_streamlines)

    logger.info("Loading vtk section... Done!")

    return streamlines, number_of_streamlines


def new_setting_streamlines(number_of_streamlines):
    predictions = vtk.vtkUnsignedCharArray()
    predictions.SetNumberOfTuples(number_of_streamlines)
    predictions.SetNumberOfComponents(1)
    predictions.SetName("Predicted Segmentation")

    pred_conf = vtk.vtkFloatArray()
    pred_conf.SetNumberOfTuples(number_of_streamlines)
    pred_conf.SetNumberOfComponents(1)
    pred_conf.SetName("Predicted Confidence")

    # Makes area good colors
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfTuples(number_of_streamlines * 3)
    colors.SetNumberOfComponents(3)
    colors.SetName("Area")

    pred_colors = vtk.vtkUnsignedCharArray()
    pred_colors.SetNumberOfTuples(number_of_streamlines * 3)
    pred_colors.SetNumberOfComponents(3)
    pred_colors.SetName("Predicted_area")

    return predictions, pred_conf, colors, pred_colors


def process(config, paths):
    profiles_df = read_data(paths)

    sections = config("holdout_sections")

    case_list = pd.unique(profiles_df.case)

    for case in case_list:
        case_df = profiles_df[profiles_df.case == case]
        for i_sec in sections:
            _df = case_df[case_df.section == i_sec]

            streamlines, number_of_streamlines = load_vtk_section(paths.streamlines_dir, case, i_sec)

            predictions, pred_conf, colors, pred_colors = new_setting_streamlines(number_of_streamlines)

            id_array = np.arange(number_of_streamlines, dtype=np.uint)

            for streamline_id in tqdm(id_array):
                logger.info("Profile id... %s", streamline_id)

                row = _df[_df.profile_id == streamline_id]

                if row.empty:
                    _predicted_area = int(0)
                    _predicted_area_confidence = 0.0
                    _true_color = np.array([255, 255, 255], dtype=np.uint8)
                    _predicted_color = np.array([255, 255, 255], dtype=np.uint8)

                else:
                    _pred_y = row.pred_area_id.iloc[0]
                    _pred_confidence = row.pred_confidence.iloc[0]

                    colors_df = row[["color_r", "color_g", "color_b"]].iloc[0]
                    colors_df = [int(c) for c in colors_df]

                    pred_colors_df = row[["pred_color_r", "pred_color_g", "pred_color_b"]].iloc[0]
                    pred_colors_df = [int(c) for c in pred_colors_df]

                    _predicted_area = int(_pred_y)
                    _predicted_area_confidence = float(_pred_confidence)

                    _true_color = colors_df
                    _predicted_color = pred_colors_df

                predictions.SetTuple1(streamline_id, _predicted_area)
                pred_conf.SetTuple1(streamline_id, _predicted_area_confidence)
                colors.SetTuple3(streamline_id, *_true_color)
                pred_colors.SetTuple3(streamline_id, *_predicted_color)

                streamlines.GetCellData().AddArray(predictions)
                streamlines.GetCellData().AddArray(pred_conf)
                streamlines.GetCellData().AddArray(colors)
                streamlines.GetCellData().AddArray(pred_colors)

            vtk_core.save_vtk_polydata(
                f"{paths.output_streamlines}/{case}_{i_sec}_streamlines_confidence_pred.vtk",
                streamlines,
            )


def parse_args():
    """ """
    parser = argparse.ArgumentParser(
        description=process.__doc__,
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
        help="Path to ",
    ),

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        dest="output_streamlines",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    ),

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    input_options = parse_args()
    data_settings = DatasetConfiguration(input_options.config_fname)
    process(data_settings, input_options)
