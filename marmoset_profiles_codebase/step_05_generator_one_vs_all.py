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


class OneVsAllAugmentationSequence(Sequence):

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
        self.half_bs = int(batch_size // 2)

        self.df = df

        self.class_list = pd.unique(df.idx_in_model)
        self._df_dict = {}
        self.prob_dict = {}
        for _class in self.class_list:
            _df = self.df[self.df.idx_in_model == _class]
            self._df_dict[_class] = np.array(_df.new_index.astype(int), dtype=np.uint)
            self.prob_dict[_class] = _df.prob.array

        # in binary model; to speed up calculations
        self.prob_dict_0 = self.prob_dict[0]
        self.prob_dict_0 = self.prob_dict_0 / np.sum(self.prob_dict_0)
        self.prob_dict_1 = self.prob_dict[1]
        self.prob_dict_1 = self.prob_dict_1 / np.sum(self.prob_dict_1)

        self._add_noise = add_noise
        self._noise_ratio = noise_ratio
        self._add_artifacts = add_artifacts
        self._artifacts_distrib = artifacts_ratio
        self._profile_len = x_set.shape[1]

        self._art_switch = False

        self._rng = np.random.default_rng(seed=42)  # TODO

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):

        # Equal number of examples in each class
        # classes_in_batch = self._rng.choice(self.class_list, self.batch_size, replace=True)
        # batch_idx = np.vectorize(lambda _class: self.select_profiles(self._df_dict[_class],
        #                                                              self.prob_dict[_class]))(classes_in_batch)
        batch_idx = np.zeros(self.batch_size, dtype=int)
        batch_idx[: self.half_bs] = self.select_profiles(
            self._df_dict[0], self.prob_dict_0, self.half_bs
        )
        batch_idx[self.half_bs :] = self.select_profiles(
            self._df_dict[1], self.prob_dict_1, self.half_bs
        )
        self._rng.shuffle(batch_idx)
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

    def select_profiles(self, idx_pool, prob, sample_size):
        return self._rng.choice(idx_pool, sample_size, p=prob)
