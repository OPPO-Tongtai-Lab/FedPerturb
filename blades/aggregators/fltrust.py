# add Defense scheme's detection results for malicious clients by OPPO

from typing import List

import torch
import numpy as np
from blades.clients.client import BladesClient


class Fltrust(object):
    r"""``Fltrust`` it a trusted-based aggregator from paper `FLTrust:
    Byzantine-robust Federated Learning via Trust Bootstrapping.

    <https://arxiv.org/abs/2012.13995>`_.
    """
    def __init__(
        self,
        debug_loger,
    ):
        self.debug_loger = debug_loger

    def __call__(self, clients: List[BladesClient]):
        trusted_clients = [client for client in clients if client.is_trusted()]
        assert len(trusted_clients) == 1
        trusted_client = trusted_clients[0]

        untrusted_clients = [client for client in clients if client.is_byzantine()]
        beny_clients = [client for client in clients if (not client.is_trusted() and not client.is_byzantine())]
        by_size = len(untrusted_clients)
        untrusted_clients.extend(beny_clients)

        trusted_update = trusted_client.get_update()
        trusted_norm = torch.norm(trusted_update).item()
        untrusted_updates = list(map(lambda w: w.get_update(), untrusted_clients))
        cosine_similarity = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        ts = torch.Tensor(
            list(
                map(
                    lambda update: torch.nn.functional.relu(
                        cosine_similarity(trusted_update, update)
                    ),
                    untrusted_updates,
                )
            )
        )
        pseudo_gradients = torch.vstack(
            list(
                map(
                    lambda update: update * trusted_norm / torch.norm(update).item(),
                    untrusted_updates,
                )
            )
        )
        true_update = (pseudo_gradients.T @ ts.to(pseudo_gradients.device) ) / ts.sum()
        tslist = list(np.round(np.array(ts),4))
        self.debug_loger.info(f"FLTrust sim score is {tslist}, byzantine score is{tslist[0:by_size]}")
        return true_update
