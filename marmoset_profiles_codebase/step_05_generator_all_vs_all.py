import logging
import math

import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence

C_LOGGER_NAME = "training_model"
logging.basicConfig(
    level=getattr(logging, "DEBUG"),
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(C_LOGGER_NAME)


class BatchNormAugmentationSequence(Sequence):
    """
    This class is a data generator
    for training deep neural network models in TensorFlow.
    It inherits the TensorFlow Sequence class,
    providing an efficient way to load
    and augment batches of data during model training.

    Parameters:
        - x_set: Input data.
        - y_set: Labels corresponding to the input data.
        - batch_size: Size of each data batch.
        - df: DataFrame containing additional
            information about model indices.
        - add_noise: Boolean flag
            indicating whether to add noise
            to the data (default: False).
        - noise_ratio: The ratio of noise
            to be added to the data (default: 0.2).
        - add_artifacts: Boolean flag
            indicating whether to add artifacts
            to the data (default: False).
        - artifacts_ratio: The ratio of batches
            with added artifacts (default: 0.2).

    Methods:
        - __len__(self): Returns the number of batches in the sequence.
        - __getitem__(self, idx): Returns a batch of data
            and labels for a given batch index.
        - select_profiles(self, idx_pool):
            Randomly selects profile indices from a given pool.

    Class Variables:
        - _df_dict: Dictionary storing profile indices for each class.
        - _add_noise, _noise_ratio:
            Flags and ratio for adding noise to the data.
        - _add_artifacts, _artifacts_distrib:
            Flags and ratio for adding artifacts to the data.
        - _art_switch:
            Boolean variable tracking the state of the artifact switch.
        - _rng: NumPy random number generator.

    Note:
        It allows flexibility in modifying data by adding noise or artifacts,
        aiding in better generalization to diverse conditions.
    """

    def __init__(
        self,
        x_set,
        y_set,
        batch_size,
        df,
        add_noise=False,
        noise_ratio=0.2,
        add_artifacts=False,
        artifacts_ratio=0.2,
    ):

        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

        self.df = df
        self.class_list = pd.unique(df.label)
        self._df_dict = {}
        for _class in self.class_list:
            _df = self.df[self.df.label == _class]
            self._df_dict[_class] = np.array(_df.new_index.astype(int), dtype=np.uint)

        self._add_noise = add_noise
        self._noise_ratio = noise_ratio
        self._add_artifacts = add_artifacts
        self._artifacts_distrib = artifacts_ratio
        self._profile_len = x_set.shape[1]

        self._art_switch = False

        self._rng = np.random.default_rng(seed=42)

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):

        # Equal number of examples in each class
        classes_in_batch = self._rng.choice(
            self.class_list, self.batch_size, replace=True
        )
        batch_idx = np.vectorize(
            lambda _class: self.select_profiles(self._df_dict[_class])
        )(classes_in_batch)
        batch_x = self.x[batch_idx]
        batch_y = self.y[batch_idx]

        if self._add_noise:
            # Please make note that is compatible to data with only 1 channel
            x = batch_x[:, :, 0]
            mu = np.zeros(x.shape[0])
            std = self._noise_ratio * np.std(x, axis=1)

            noise = self._rng.normal(
                np.expand_dims(mu, axis=1), np.expand_dims(std, axis=1), size=x.shape
            )
            x += noise
            batch_x[:, :, 0] = x

        if self._add_artifacts:
            artifacts_chance = self._rng.random(self.batch_size)
            where_artifacts = np.where(artifacts_chance < self._artifacts_distrib)[0]
            for i in where_artifacts:
                st_idx = np.random.randint(0, batch_x.shape[1] - 1)
                end_idx = np.random.randint(st_idx, batch_x.shape[1])
                if self._art_switch:
                    batch_x[i, st_idx:end_idx, 0] -= 2 * np.mean(
                        batch_x[i, st_idx:end_idx, 0]
                    )
                else:
                    batch_x[i, st_idx:end_idx, 0] += 4 * np.mean(
                        batch_x[i, st_idx:end_idx, 0]
                    )

                self._art_switch = ~self._art_switch

        return batch_x, batch_y

    @staticmethod
    def select_profiles(idx_pool):
        return np.random.choice(idx_pool, 1)
