import copy
import glob
import logging
import os
import re

from read_json import read_json

C_LOGGER_NAME = "Dataset Configuration"
logging.basicConfig(
    level=getattr(logging, "DEBUG"),
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(C_LOGGER_NAME)


class DatasetConfiguration(object):
    """
    Profile processing information.

    The purpose of this class is to read dataset preprocessing
    configuration from json file and handle it.

    Json file must contain certain fields:
    ----------
    {
        "profile_length": 300,
        "case_list": [
            "atlas"
        ],
        "stack_names": [
            "average"
        ],
        "specific_labels_list": [],
        "min_confidence_level": 0.67,
        "max_confidence_level": 1.0,
        "specific_sections_list": [],
        "drop_underrepresented_labels": true,
        "labels_drop_rate": 0.1,
        "adapt_label_amount": false,
        "max_label_amount": 5000,
        "adapt_to_least_amount_label": false,
        "allow_replicate_profiles": false,
        "test_size": 0.2,
        "norm_by_mean_std": true,
        "norm_to_new_range": false,
        "new_range_min": 0,
        "new_range_max": 1,
        "norm_among_labels": false,
        "norm_among_all_profiles": false,
        "add_noise": true,
        "add_artifacts": false,
        "max_bar_in_plot": 30
    }

    Notes
    ----------
    profile_length : int
        length of profile in samples
    case_list : list
        preprocessing cases
    stack_names : list
        prefix of slice stack
        [!] len(case_list) == len(stack_names)
    specific_labels_list : list
        labels to train the model
        type one number in a LIST to binarize the model
        number from file or itk ??????
    min_confidence_level : float
        minimum level of profile confidence
        segmentation confidence;
        value between 0 and 1
    max_confidence_level : float
        maximum level of profile confidence

    test_size : float
        proportion of test dataset profile


    Methods
    ----------
    sections(case_dir, report):
        return list of preprocessing sections

    labels(labels_dict_fname):
        return list of preprocessing sections

    """

    def __init__(self, conf_fname):
        self._config_dict = read_json(conf_fname)

    def __call__(self, param_name):
        return copy.deepcopy(self._config_dict[param_name])

    @property
    def settings_in_dict(self):
        return copy.deepcopy(self._config_dict)

    def sections(self, case_dir):
        if len(self._config_dict["specific_sections_list"]) == 0:

            sections_path_list = glob.glob(case_dir + "/[0-9]*")
            sections_path_list.sort()
            sections_list = [
                re.search(r"[0-9]*$", path).group(0) for path in sections_path_list
            ]
            return sections_list, sections_path_list

        else:
            sections_list = []
            sections_path_list = []
            for section in self._config_dict["specific_sections_list"]:
                sections_list.append(f"%04.f" % section)
                path = os.path.join(case_dir, f"%04.f" % section)
                if os.path.exists(path):
                    sections_path_list.append(path)
                else:
                    logger.warning("Path does not exist... %s", path)
            return sections_list, sections_path_list
