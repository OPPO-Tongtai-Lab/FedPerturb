# Re-implementation based on thesis code and add Defense scheme's detection results for malicious clients by OPPO

from typing import List, Union

import torch
import numpy as np
from blades.clients.client import BladesClient
from .mean import _BaseAggregator


class Multikrum(_BaseAggregator):
    r"""A robust aggregator from paper `Machine Learning with Adversaries:
    Byzantine Tolerant Gradient Descent.

    <https://dl.acm.org/doi/abs/10.5555/3294771.3294783>`_.

    Given a collection of vectors, ``Krum`` strives to find one of the vector that is
    closest to another :math:`K-M-2` ones with respect to squared Euclidean distance,
    which can be expressed by:

      .. math::
         Krum := \{{\Delta}_i | i = \arg\min_{i \in [K]} \sum_{i \rightarrow j}  \lVert
         {\Delta}_i - {\Delta}_j \rVert^2 \}

    where :math:`i \rightarrow j` is the indices of the :math:`K-M-2` nearest neighbours
    of :math:`{\Delta}_i` measured by squared ``Euclidean distance``,  :math:`K` is the
    number of input in total, and :math:`M` is the number of Byzantine input.

    Args:
        num_excluded {int} -- Total number of Byzantine workers.
        m {int} -- Number of workers for aggregation.
    """

    def __init__(self, debug_loger = None, num_excluded=5, multk=True):
        self.debug_loger = debug_loger
        self.num_byz = num_excluded
        self.multk = multk
        self.byze_encount = []
        self.byze_notIdent = []
        super(Multikrum, self).__init__()

    def __call__(self, inputs: Union[List[BladesClient], List[torch.Tensor]]):
        updates = self._get_updates(inputs)

        byze = 0
        for c in inputs:
            if c.is_byzantine():
                byze = byze + 1
        self.byze_encount.append(byze)

        candidates = []
        candidate_indices = []
        remaining_updates = updates
        all_indices = np.arange(len(updates))

        while len(remaining_updates) > 2 * self.num_byz + 2:
            torch.cuda.empty_cache()
            distances = []
            for update in remaining_updates:
                distance = []
                for update_ in remaining_updates:
                    distance.append(torch.norm((update - update_)) ** 2)
                distance = torch.Tensor(distance).float()
                distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

            distances = torch.sort(distances, dim=1)[0]
            scores = torch.sum(distances[:, :len(remaining_updates) - 2 - self.num_byz], dim=1)
            indices = torch.argsort(scores)[:len(remaining_updates) - 2 - self.num_byz]

            candidate_indices.append(all_indices[indices[0].cpu().numpy()])
            all_indices = np.delete(all_indices, indices[0].cpu().numpy())
            candidates = remaining_updates[indices[0]][None, :] if not len(candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
            remaining_updates = torch.cat((remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
            if not self.multk:
                break
        
        values = torch.mean(candidates, dim=0)

        chose_id = []
        byze_count = 0
        for index in candidate_indices:
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
        self.debug_loger.info(f"Multikrum  result ID is {chose_id}")

        # values = torch.stack([updates[i] for i in top_m_indices], dim=0).mean(dim=0)
        return values

    def __str__(self):
        return "Krum (m={})".format(self.m)
