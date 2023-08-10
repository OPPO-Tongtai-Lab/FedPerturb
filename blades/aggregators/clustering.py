# add Defense scheme's detection results for malicious clients by OPPO

from typing import List, Union

import numpy as np
import torch
from numpy import inf
from sklearn.cluster import AgglomerativeClustering

from blades.clients.client import BladesClient
from .mean import _BaseAggregator


class Clustering(_BaseAggregator):
    r"""A robust aggregator from paper `On the byzantine robustness of clustered
    federated learning.
    ICASSP 2020, CCF B
    <https://ieeexplore.ieee.org/abstract/document/9054676>`_.

    It separates the client population into two groups based on the cosine
    similarities.
    """

    def __init__(self, debug_loger):
        # super(Clustering, self).__init__()
        self.debug_loger = debug_loger
        self.byze_encount = []
        self.byze_notIdent = []

    def __call__(
        self, inputs: Union[List[BladesClient], List[torch.Tensor], torch.Tensor]
    ):
        updates = self._get_updates(inputs)

        byze = 0
        for c in inputs:
            if c.is_byzantine():
                byze = byze + 1
        self.byze_encount.append(byze)

        num = len(updates)
        dis_max = np.zeros((num, num))
        for i in range(num):
            for j in range(i + 1, num):
                dis_max[i, j] = 1 - torch.nn.functional.cosine_similarity(
                    updates[i, :], updates[j, :], dim=0
                )
                dis_max[j, i] = dis_max[i, j]
        dis_max[dis_max == -inf] = -1
        dis_max[dis_max == inf] = 1
        dis_max[np.isnan(dis_max)] = -1
        # with open('../notebooks/updates_fedsgd_ipm.npy', 'wb') as f:
        #     np.save(f, dis_max)
        clustering = AgglomerativeClustering(
            affinity="precomputed", linkage="complete", n_clusters=2
        )
        clustering.fit(dis_max)
        flag = 1 if np.sum(clustering.labels_) > num // 2 else 0
        values = torch.vstack(
            list(
                model
                for model, label in zip(updates, clustering.labels_)
                if label == flag
            )
        ).mean(dim=0)

        chose_id = []
        byze_count = 0
        for index, label in enumerate(clustering.labels_):
            if label != flag:
                continue
            chose_id.append(int(inputs[index]._id))
            if inputs[index].is_byzantine():
                byze_count = byze_count + 1
        self.byze_notIdent.append(byze_count)
        if len(self.byze_notIdent)> 5 and np.sum(self.byze_encount[-5:]) > 0:
            fnr = np.sum(self.byze_notIdent[-5:]) / np.sum(self.byze_encount[-5:])
            self.debug_loger.info(f"Recent 5 round FNR is {fnr:.4f}")
        if np.sum(self.byze_encount) > 0:
            fnr = np.sum(self.byze_notIdent) / np.sum(self.byze_encount)
            self.debug_loger.info(f"Total FNR is {fnr:.4f}") 
        self.debug_loger.info(f"clustering  result ID is {chose_id}")

        return values
