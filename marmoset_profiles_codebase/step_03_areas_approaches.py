import copy
import logging

import numpy as np
import pandas as pd

C_LOGGER_NAME = "create_datasets"
logging.basicConfig(
    level=getattr(logging, "DEBUG"),
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(C_LOGGER_NAME)


def get_approach(config):

    if config("do_all_areas"):
        if config("other_to_zero"):
            logger.warning("Option other_to_zero is True")

        return AllvsAll

    if len(config("specific_area_id_list")) == 1:
        return OneVsAll

    if len(config("specific_area_id_list")) > 1:
        return SelectedAreas

    raise ValueError("Invalid data configuration! Check dataset_settings")


class PickAreas:
    def __init__(self, config, profiles_df):
        logger.info("Pick profiles due the area_id...")

        self._config = config
        self._df = profiles_df
        self._all_labels = pd.unique(profiles_df.area_id)
        self._areas_to_process = self._get_area_id_to_process()

    def _get_area_id_to_process(self):
        return copy.copy(self._all_labels)

    def _check_area_id_values(self, all_labels):
        pass

    def process(self):
        return copy.copy(self._df)

    @property
    def areas_to_process(self):
        return copy.copy(self._areas_to_process)

    @property
    def reject_areas(self):
        all_areas_id = pd.unique(self._df)
        reject_areas = np.setdiff1d(all_areas_id, self._areas_to_process)
        return reject_areas


class AllvsAll(PickAreas):
    def __init__(self, config, profiles_df):
        super().__init__(config, profiles_df)
        logger.info("All vs All approach!")

    # def _check_area_id_values(self): # z labels
    #     if (all_area_id == area_profiles_df).all():
    #         logger.warning("Check input! %s", np.setxor1d(all_area_id, area_profiles_df))


class OneVsAll(PickAreas):
    def __init__(self, config, profiles_df):
        super().__init__(config, profiles_df)
        logger.info("One vs All approach!")

    def _get_area_id_to_process(self):
        # _areas_to_process
        self._check_area_id_values(self._all_labels)
        areas_id = np.array(self._config("specific_area_id_list"))
        if self._config("other_to_zero"):
            areas_id = np.insert(areas_id, 0, 0)

        return areas_id

    def _check_area_id_values(self, all_labels):
        if self._config("other_to_zero"):
            if self._config("zero_profiles_number ") == 0:
                raise ValueError("Number of to_zero_labeled profiles is 0!")

        if self._config("specific_area_id_list")[0] not in all_labels:
            raise ValueError("Area id does not occur in dataset!")

    def process(self):
        to_zero_area_id = np.setdiff1d(self._all_labels, self._areas_to_process)
        to_zero_df = self._df.area_id.isin(to_zero_area_id)
        if self._config("other_to_zero"):
            self._df.loc[to_zero_df, "label"] = 0
        else:
            self._df.loc[to_zero_df, "accept"] = False

        return copy.copy(self._df)


class SelectedAreas(PickAreas):
    pass
