from typing import Generator, List
from blades.clients import BladesClient
import numpy as np
import torch

from blades.aggregators.multikrum import Multikrum
from blades.clients.client import ByzantineClient

# Local Model Poisoning Attacks to Byzantine-Robust Federated Learning
# untarget poison attack ,directed deviation goal is to deviate a global model
# parameter the most towards the inverse of the direction along
# which the global model parameter would change without attacks.
class FangattackClient(ByzantineClient):
    def omniscient_callback(self, simulator):
        pass

    def train_global_model(self, train_set: Generator, global_round: int , local_steps: int, opt: torch.optim.Optimizer) -> None:
        pass
        pass


class FangattackAdversary:
    r""""""

    def __init__(
        self,
        num_byz: int,
        agg: str,
        dev_type="std",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dev_type = dev_type
        self.agg = agg
        self.num_byzantine = num_byz

    def attack_median_and_trimmedmean(self, clients: List[BladesClient]):
        benign_update = []
        num_byzantine = 0
        for w in clients:
            if not w.is_byzantine():
                benign_update.append(w.get_update())
            else:
                num_byzantine += 1
        if num_byzantine == 0:
            return 
        benign_update = torch.stack(benign_update, 0)
        agg_grads = torch.mean(benign_update, 0)
        deviation = torch.sign(agg_grads)
        device = benign_update.device
        b = 2
        max_vector = torch.max(benign_update, 0)[0]
        min_vector = torch.min(benign_update, 0)[0]

        max_ = (max_vector > 0).type(torch.FloatTensor).to(device)
        min_ = (min_vector < 0).type(torch.FloatTensor).to(device)

        max_[max_ == 1] = b
        max_[max_ == 0] = 1 / b
        min_[min_ == 1] = b
        min_[min_ == 0] = 1 / b

        max_range = torch.cat(
            (max_vector[:, None], (max_vector * max_)[:, None]), dim=1
        )
        min_range = torch.cat(
            ((min_vector * min_)[:, None], min_vector[:, None]), dim=1
        )

        rand = (
            torch.from_numpy(
                np.random.uniform(0, 1, [len(deviation), num_byzantine])
            )
            .type(torch.FloatTensor)
            .to(benign_update.device)
        )

        max_rand = (
            torch.stack([max_range[:, 0]] * rand.shape[1]).T
            + rand * torch.stack([max_range[:, 1] - max_range[:, 0]] * rand.shape[1]).T
        )
        min_rand = (
            torch.stack([min_range[:, 0]] * rand.shape[1]).T
            + rand * torch.stack([min_range[:, 1] - min_range[:, 0]] * rand.shape[1]).T
        )

        mal_vec = (
            torch.stack(
                [(deviation < 0).type(torch.FloatTensor)] * max_rand.shape[1]
            ).T.to(device)
            * max_rand
            + torch.stack(
                [(deviation > 0).type(torch.FloatTensor)] * min_rand.shape[1]
            ).T.to(device)
            * min_rand
        ).T
        i = 0
        for client in (clients):
            if client.is_byzantine():
                client.save_update(mal_vec[i])
                i = i + 1


    def attack_multikrum(self, clients: List[BladesClient]):
        multi_krum = Multikrum(num_excluded=self.num_byzantine, k=1)
        benign_update = []
        for w in clients:
            if not w.is_byzantine():
                benign_update.append(w.get_update())
        benign_update = torch.stack(benign_update, 0)
        agg_updates = torch.mean(benign_update, 0)
        deviation = torch.sign(agg_updates)

        def compute_lambda(all_updates, model_re, n_attackers):

            distances = []
            n_benign, d = all_updates.shape
            for update in all_updates:
                distance = torch.norm((all_updates - update), dim=1)
                distances = (
                    distance[None, :]
                    if not len(distances)
                    else torch.cat((distances, distance[None, :]), 0)
                )

            distances[distances == 0] = 10000
            distances = torch.sort(distances, dim=1)[0]
            scores = torch.sum(
                distances[:, : n_benign - 2 - n_attackers],
                dim=1,
            )
            min_score = torch.min(scores)
            term_1 = min_score / (
                (n_benign - n_attackers - 1) * torch.sqrt(torch.Tensor([d]))[0]
            )
            max_wre_dist = torch.max(torch.norm((all_updates - model_re), dim=1)) / (
                torch.sqrt(torch.Tensor([d]))[0]
            )

            return term_1 + max_wre_dist

        # all_updates = torch.stack(
        #     list(map(lambda w: w.get_update(), clients))
        # )
        lambda_ = compute_lambda(benign_update, agg_updates, self.num_byzantine)

        threshold = 1e-5
        mal_update = []

        while lambda_ > threshold:
            mal_update = -lambda_ * deviation
            mal_updates = torch.stack([mal_update] * self.num_byzantine)
            mal_updates = torch.cat((mal_updates, benign_update), 0)

            # print(mal_updates.shape, n_attackers)
            value = multi_krum(mal_updates)
            krum_candidate = multi_krum.top_m_indices
            if krum_candidate[0] < self.num_byzantine:
                return mal_update
            else:
                mal_update = []

            lambda_ *= 0.5

        if not len(mal_update):
            mal_update = agg_updates - lambda_ * deviation

        # i = 0
        for client in (clients):
            if client.is_byzantine():
                client.save_update(mal_update)
                # i = i + 1
        # return mal_update

    def omniscient_callback(self, clients: List[BladesClient]):
        if self.agg in ["median", "trimmedmean"]:
            self.attack_median_and_trimmedmean(clients)
        elif self.agg in ["multikrum"]:
            self.attack_multikrum(clients)
        else:
            self.attack_multikrum(clients)
            # raise NotImplementedError(
            #     f"Adaptive attacks to {self.agg} " f"is not supported yet."
            # )
