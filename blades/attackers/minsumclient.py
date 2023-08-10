# added implementation of the Minsum by OPPO
import torch
from typing import Generator, List

from blades.clients.client import ByzantineClient, BladesClient
from typing import  Optional


class MinsumClient(ByzantineClient):
    def omniscient_callback(self, simulator):
        pass

    # def train_global_model(self, train_set: Generator, global_round: int , local_steps: int, opt: torch.optim.Optimizer,):
    #     pass


class MinsumAdversary:
    r""""""

    def __init__(
        self,
        dev_type="unit_vec",
        threshold=100.0,
        threshold_diff=1e-5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dev_type = dev_type
        self.threshold = threshold
        self.threshold_diff = threshold_diff

    def omniscient_callback(self, clients: List[BladesClient]):
        byzantine = []
        for client in clients:
            if client.is_byzantine():
                byzantine.append(client)
        all_updates = torch.stack(
            list(map(lambda w: w.get_update(), byzantine))
        )
        model_re = torch.mean(all_updates, 0)

        if self.dev_type == 'unit_vec':
            deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
        elif self.dev_type == 'sign':
            deviation = torch.sign(model_re)
        elif self.dev_type == 'std':
            deviation = torch.std(all_updates, 0)
        
        lamda = torch.Tensor([self.threshold]).to(all_updates.device)

        # print(lamda)
        # threshold_diff = 1e-5
        lamda_fail = lamda
        lamda_succ = 0
        
        distances = []
        # calcaulate the distance between any two updates
        for update in all_updates:
            distance = torch.norm((all_updates - update), dim=1) ** 2
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
        
        scores = torch.sum(distances, dim=1)
        min_score = torch.min(scores)
        del distances

        while torch.abs(lamda_succ - lamda) > self.threshold_diff:
            mal_update = (model_re - lamda * deviation)
            distance = torch.norm((all_updates - mal_update), dim=1) ** 2
            score = torch.sum(distance)
            
            if score <= min_score:
                # print('successful lamda is ', lamda)
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2

        # print(lamda_succ)
        mal_update = (model_re - lamda_succ * deviation)
        for client in byzantine:
            client.save_update(mal_update)
        # return mal_update