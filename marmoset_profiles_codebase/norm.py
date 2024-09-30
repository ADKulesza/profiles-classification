import copy
import logging

import numpy as np

C_LOGGER_NAME = "norm"
logging.basicConfig(
    level=getattr(logging, "DEBUG"),
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(C_LOGGER_NAME)


class NormProfiles(object):
    """ """

    def __init__(self, config, profiles_arr):

        self._config = config
        self._profiles = profiles_arr

        if self._config("norm_by_mean_std"):
            logger.info("Norm by mean std...")
            self._norm_profiles = self._norm_by_mean_std()

        else:
            logger.info("No single profile normalization...")

    @property
    def norm_profiles(self):
        return copy.deepcopy(self._norm_profiles)

    def _norm_by_mean_std(self):
        norm_profiles = self._profiles - np.mean(self._profiles, axis=0)
        norm_profiles = norm_profiles / np.std(self._profiles, axis=0)
        return norm_profiles
