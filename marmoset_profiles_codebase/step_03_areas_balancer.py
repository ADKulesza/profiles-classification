import logging

import networkx as nx
import numpy as np
import pandas as pd

from read_json import read_json
from step_00_graph_of_labels import graph_weight

C_LOGGER_NAME = "balance_profiles"
logging.basicConfig(
    level=getattr(logging, "DEBUG"),
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(C_LOGGER_NAME)


class GraphBalance:
    def __init__(
        self,
        config,
        graph_path,
        profile_df,
        label_to_prcs,
        zero_labels,
        label_weigths_path,
    ):
        self._config = config
        data_graph = read_json(graph_path)
        self._graph = nx.node_link_graph(data_graph)
        self._profile_df = profile_df
        self._label_to_prcs = label_to_prcs
        self._zero_labels = zero_labels
        self._label_weigths = label_weigths_path

        self._prof_calc = self.get_number_of_profiles()

    def get_weight(self):
        weights = dict.fromkeys(self._zero_labels, 0)
        n_labels = self._label_to_prcs.shape[0]
        for z_label in self._zero_labels:
            n_p = self._profile_df[self._profile_df.area_id == z_label].shape[0]

            if n_p < 0.01 * self._config("zero_profiles_number "):
                weights[z_label] = 1
                continue
            for _label in self._label_to_prcs[1:]:
                _weight = graph_weight(
                    self._graph,
                    str(int(_label)),
                    str(int(z_label)),
                    self._config("reduce_label_impact"),
                )
                weights[z_label] += _weight
            weights[z_label] /= n_labels

        # zabezpieczenie na wypadek, gdyby ratio było źle podane
        for z_label, _weight in weights.items():
            if _weight > 1:
                weights[z_label] = 1

        df = pd.DataFrame({"label": weights.keys(), "weight": weights.values()})

        df.to_csv(self._label_weigths)

        return weights

    def get_number_of_profiles(self):
        weights = self.get_weight()
        profiles_number = dict.fromkeys(self._zero_labels, 0)
        number_all = self._profile_df[self._profile_df.label == 0].shape[0]
        label_n = self._zero_labels.shape[0]

        # if number_all > sum_all ! weź wszystkie dostępne
        for z_label in self._zero_labels:
            n_p = self._profile_df[self._profile_df.area_id == z_label].shape[0]
            profiles_number[z_label] = int(
                n_p * weights[z_label] / np.exp(6.54 * n_p / number_all)
            )
        _sum = sum(profiles_number.values())

        # Jeśli za mało dodaj równomiernie
        if _sum < self._config("zero_profiles_number "):

            profile_in_add = self._config("zero_profiles_number ") - _sum
            prof_per_label = int(profile_in_add // (label_n + 1))

            backlog_profiles = 0
            for z_label in self._zero_labels:
                n_p = self._profile_df[self._profile_df.area_id == z_label].shape[0]
                if n_p - profiles_number[z_label] - prof_per_label > 0:
                    profiles_number[z_label] += prof_per_label
                    if (
                        backlog_profiles != 0
                        and profiles_number[z_label] - backlog_profiles > 0
                    ):
                        profiles_number[z_label] += backlog_profiles
                        backlog_profiles = 0
                else:
                    backlog_profiles += prof_per_label

                if profiles_number[z_label] > n_p:
                    profiles_number[z_label] = n_p

                if sum(profiles_number.values()) >= number_all:
                    break

        return profiles_number

    def balance_by_graph(self):

        for z_label in self._zero_labels:
            _df = self._profile_df[self._profile_df.area_id == z_label]
            reject_profile_number = _df.shape[0] - self._prof_calc[z_label]
            _sample_df = _df.sample(n=reject_profile_number)
            _df_idx = np.array(_sample_df.index, dtype=int)
            self._profile_df.loc[_df_idx, "accept"] = False

        return self._profile_df


def balance_dataset(config, profile_df, labels_to_process, graph_path, label_weights):

    if config("other_to_zero") and len(config("specific_area_id_list")) != 0:

        zero_labels = pd.unique(profile_df[profile_df.label == 0].area_id)

        graph = GraphBalance(
            config,
            graph_path,
            profile_df,
            labels_to_process,
            zero_labels,
            label_weights,
        )

        profile_df = graph.balance_by_graph()

    for _label in labels_to_process:
        if _label == 0:
            l_num = config("zero_profiles_number ")
        else:
            l_num = config("max_label_amount")
        _df = profile_df[(profile_df.label == _label) & profile_df.accept]
        if _df.shape[0] < l_num:
            continue

        to_false_num = _df.shape[0] - l_num
        _sample_df = _df.sample(n=to_false_num)

        _df_idx = np.array(_sample_df.index, dtype=int)
        profile_df.loc[_df_idx, "accept"] = False
    return profile_df
