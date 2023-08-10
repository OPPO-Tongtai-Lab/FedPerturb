# Copyright 2023, OPPO.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# added implementation of the FLAME

from typing import List

import torch
import numpy as np
from blades.clients.client import BladesClient
import hdbscan
from typing import  Optional

class Flame(object):
    r"""``Flame`` it a cluster-based aggregator from paper `FLAME: 
    Taming Backdoors in Federated Learning.

    <https://www.usenix.org/conference/usenixsecurity22/presentation/nguyen>`_.
    """
    def __init__(
        self,
        debug_loger,
        part_size: Optional[int]= 10,
        noise_level:Optional[float] = 0.001,
    ):
        self.debug_loger = debug_loger
        self.min_cluster_size = int(part_size // 2 + 1)
        self.noise_level = noise_level
        self.eopch = 0
        self.byze_encount = []
        self.byze_notIdent = []

    def __call__(self, clients: List[BladesClient]):
        # record current epoch
        self.eopch = self.eopch + 1
        updates = torch.vstack(list(map(lambda w: w.get_update(), clients)))
        byze = 0
        for c in clients:
            if c.is_byzantine():
                byze = byze + 1
        self.byze_encount.append(byze)
        updates_array = np.array(list(map(lambda w: np.array(w.get_update().to(torch.device("cpu"))), clients))).astype(float)
        
        clusterer = hdbscan.HDBSCAN(min_cluster_size = self.min_cluster_size, min_samples=1, allow_single_cluster=True, metric="cosine", algorithm="generic")
        clusterer.fit_predict(updates_array)
        cluster_result = clusterer.labels_
    
        cluster_num = clusterer.labels_.max() + 1
        cluster_size = []
        for i in range(0, cluster_num):
            cluster_size.append(np.sum(cluster_result == i))
        # find the index of the largest cluster
        true_cluster_index = np.where(cluster_result == np.array(cluster_size).argmax())
        cluster_id = []
        byze_count = 0
        for index in true_cluster_index[0]:
            cluster_id.append(int(clients[index]._id))
            if clients[index].is_byzantine():
                byze_count = byze_count + 1
        cluster_id.sort()
        self.byze_notIdent.append(byze_count)
        if len(self.byze_notIdent)> 5 and np.sum(self.byze_encount[-5:]) > 0:
            fnr = np.sum(self.byze_notIdent[-5:]) / np.sum(self.byze_encount[-5:])
            self.debug_loger.info(f"Recent 5 round FNR is {fnr:.4f}")
        if np.sum(self.byze_encount) > 0:
            fnr = np.sum(self.byze_notIdent) / np.sum(self.byze_encount)
            self.debug_loger.info(f"Total FNR is {fnr:.4f}") 
           
        self.debug_loger.info(f"FLAME cluster result size is {len(cluster_id)},ID is {cluster_id}")
        l2norms = [torch.norm(update).item() for update in updates]
        norm_midean = np.median(l2norms)
        if self.eopch == 1:
            self.noise = torch.normal(0, pow(0.001, 2) , size=updates[0].shape).to(updates[0].device)
        pseudo_gradients = torch.vstack(
            list(
                map(
                    lambda update: (update - self.noise) * min(1, norm_midean/ torch.norm(update).item()),
                    updates[true_cluster_index[0]],
                    # chose_up,
                )
            )
        )
        update_mean = torch.mean(pseudo_gradients, dim=0)
        
        # bound norm for the first few epoch in case gradient explosion, 
        # the running_mean and running_var will greater than 1, and the norm will greater than 20
        if self.eopch< 10:# disable noise for the first few epoch
            norm_midean = 1
        
        noise = torch.normal(0, pow(self.noise_level * norm_midean, 2) , size=update_mean.shape).to(update_mean.device)
        self.noise = noise
        result = update_mean + noise
        return result
