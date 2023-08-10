# add Defense scheme's detection results for malicious clients by OPPO
from typing import List, Union

import numpy as np
import torch
from sklearn.cluster import KMeans, MeanShift

from blades.clients.client import BladesClient
from blades.utils import torch_utils
from .mean import Mean, _BaseAggregator
from .median import Median


class Signguard(_BaseAggregator):
    r"""A robust aggregator from paper `Xu et al.

    SignGuard: Byzantine-robust Federated
    Learning through Collaborative Malicious Gradient
    Filtering <https://arxiv.org/abs/2109.05872>`_.
    """

    def __init__(
        self, debug_loger, 
        agg="mean", 
        max_tau=1e5, 
        linkage="average"
    ) -> None:
        super(Signguard, self).__init__()

        assert linkage in ["average", "single"]
        self.tau = max_tau
        self.linkage = linkage
        self.l2norm_his = []
        if agg == "mean":
            self.agg = Mean()
        elif agg == "median":
            self.agg = Median()
        else:
            raise NotImplementedError(f"{agg} is not supported yet.")
        self.debug_loger = debug_loger
        self.byze_encount = []
        self.byze_notIdent = []

    def __call__(
        self, inputs: Union[List[BladesClient], List[torch.Tensor], torch.Tensor]
    ):
        updates = self._get_updates(inputs)
        num = len(updates)
        l2norms = [torch.norm(update).item() for update in updates]
        M = np.median(l2norms)
        L = 0.1
        R = 3.0
        S1_idxs = []
        for idx, (l2norm, update) in enumerate(zip(l2norms, updates)):
            if l2norm >= L * M and l2norm <= R * M:
                S1_idxs.append(idx)
                
        byze = 0
        for c in inputs:
            if c.is_byzantine():
                byze = byze + 1
        self.byze_encount.append(byze)

        features = []
        num_para = len(updates[0])
        for update in updates:
            feature0 = (update > 0).sum().item() / num_para
            feature1 = (update < 0).sum().item() / num_para
            feature2 = (update == 0).sum().item() / num_para

            features.append([feature0, feature1, feature2])

        # kmeans test for cnn_bn model 
        kmeans = KMeans(n_clusters=2, random_state=0).fit(features)
        # print(kmeans)
        flag = 1 if np.sum(kmeans.labels_) > num // 2 else 0
        S2_idxs = list(
            [idx for idx, label in enumerate(kmeans.labels_) if label == flag]
        )

        inter = list(set(S1_idxs) & set(S2_idxs))

        chose_id = []
        byze_count = 0
        for index in inter:
            chose_id.append(int(inputs[index]._id))
            if inputs[index].is_byzantine():
                byze_count = byze_count + 1
        chose_id.sort()
        self.byze_notIdent.append(byze_count)
        if len(self.byze_notIdent)> 5 and np.sum(self.byze_encount[-5:]) > 0:
            fnr = np.sum(self.byze_notIdent[-5:]) / np.sum(self.byze_encount[-5:])
            self.debug_loger.info(f"Recent 5 round FNR is {fnr:.4f}")
        if np.sum(self.byze_encount) > 0:
            fnr = np.sum(self.byze_notIdent) / np.sum(self.byze_encount)
            self.debug_loger.info(f"Total FNR is {fnr:.4f}") 
        self.debug_loger.info(f"Signguard result size is {len(chose_id)},ID is {chose_id}")

        benign_updates = []
        for idx in inter:
            if l2norms[idx] > M:
                updates[idx] = torch_utils.clip_tensor_norm_(updates[idx], M)
            benign_updates.append(updates[idx])

        values = self.agg(torch.vstack(benign_updates))
        return values
