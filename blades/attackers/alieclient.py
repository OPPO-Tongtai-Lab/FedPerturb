from typing import Generator, List
from blades.clients import BladesClient

import numpy as np
import torch
from scipy.stats import norm

from blades.clients.client import ByzantineClient


class AlieClient(ByzantineClient):
    """
    :param num_clients: Total number of input
    :param num_byzantine: Number of Byzantine input
    """

    def __init__(
        self,
        num_clients: int,
        num_byzantine: int,
        z=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Number of supporters
        if z is not None:
            self.z_max = z
        else:
            s = np.floor(num_clients / 2 + 1) - num_byzantine
            cdf_value = (num_clients - num_byzantine - s) / (
                num_clients - num_byzantine
            )
            self.z_max = norm.ppf(cdf_value)
        self.n_good = num_clients - num_byzantine



class AlieAdversary:
    def __init__(self):
        pass

    def omniscient_callback(self, clients: List[BladesClient]):
        updates = []
        for client in clients:
            if client.is_byzantine():
                updates.append(client.get_update())

        stacked_updates = torch.stack(updates, 1)
        mu = torch.mean(stacked_updates, 1)
        std = torch.std(stacked_updates, 1)
        update = None
        for client in clients:
            if client.is_byzantine():
                if update is None:
                    update = mu - std * client.z_max
                client.save_update(update)
        update = None
