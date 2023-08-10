# add Defense scheme's detection results for malicious clients， fixed the implementation by OPPO
from typing import List, Union

import numpy as np
import torch

from blades.clients.client import BladesClient
from .mean import _BaseAggregator


class Dnc(_BaseAggregator):
    r"""A robust aggregator from paper `Manipulating the Byzantine: Optimizing
    Model Poisoning Attacks and Defenses for Federated Learning.
    NDSS 2021, CCF A
    <https://par.nsf.gov/servlets/purl/10286354>`_.
    """

    def __init__(
        self, debug_loger,
        num_byzantine, *, 
        sub_dim=10000, 
        num_iters=1, 
        filter_frac=1.0
    ) -> None:
        super(Dnc, self).__init__()

        self.num_byzantine = num_byzantine
        self.sub_dim = sub_dim
        self.num_iters = num_iters
        self.fliter_frac = filter_frac
        self.debug_loger = debug_loger
        self.byze_encount = []
        self.byze_notIdent = []

    def __call__(
        self, inputs: Union[List[BladesClient], List[torch.Tensor], torch.Tensor]
    ):
        updates = self._get_updates(inputs)
        d = len(updates[0])
        byze = 0
        for c in inputs:
            if c.is_byzantine():
                byze = byze + 1
        self.byze_encount.append(byze)
        benign_ids = []
        for i in range(self.num_iters):
            indices = torch.randperm(d)[: self.sub_dim]
            sub_updates = updates[:, indices]
            mu = sub_updates.mean(dim=0)
            centered_update = sub_updates - mu
            v = torch.svd(centered_update)[2]

            sub_updates = torch.vstack(list([update - mu for update in sub_updates]))
            vv = torch.transpose(v,0,1)
            s = []
            for i in range(len(sub_updates)):
                s.append((torch.dot(sub_updates[i], vv[i]) ** 2).item())
            s = np.array(s)

            good = s.argsort()[
                : len(updates) - int(self.fliter_frac * self.num_byzantine)
            ]
            benign_ids.extend(good)

        benign_ids = list(set(benign_ids))
        benign_updates = updates[benign_ids, :].mean(dim=0)

        chose_id = []
        byze_count = 0
        for index in benign_ids:
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
        self.debug_loger.info(f"DNC result size is {len(chose_id)},ID is {chose_id}")

        return benign_updates
