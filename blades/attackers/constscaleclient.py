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

# added implementation of the Constrain-and-scale
from typing import Generator
import random
import numpy as np
import torch
import copy
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from scipy.stats import norm
import torchvision.transforms as transforms
import torchvision

from blades.clients.client import ByzantineClient
from typing import  Optional
from blades.utils.utils import model_dist_norm

class ConstscaleClient(ByzantineClient):
    r"""``Constrain-and-scale`` it a model replacement backdoor attack scheme 
    from paper `How To Backdoor Federated Learning`

    <https://arxiv.org/abs/1807.00459>`_.
    """
    """
    :param num_clients: Total number of input
    :param num_byzantine: Number of Byzantine input
    """

    def __init__(
        self,
        dataset: Optional[str] = "cifar10",
        batch_size: Optional[int] = 64,
        size_of_secret_dataset: Optional[int] = 200,
        poison_eopch: Optional[int] = 400,
        scale:Optional[int] = 10,
        num_byz:Optional[int] = 1,
        alpha_loss :Optional[float] = 1.0,
        target: Optional[int]= 2,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.size_of_secret_dataset = size_of_secret_dataset
        self.poison_eopch = poison_eopch # start poison epoch
        self.scale = scale
        self.target = target
        self.num_byz = num_byz
        self.alpha_loss = alpha_loss
        self.baseline = False # Adversary wants to scale his weights. Baseline model doesn't do this
        self.poison_images_test = [330, 568, 3934, 12336, 30560]
        self.poison_images = [30696, 33105, 33615, 33907, 36848, 40713, 41706]
        self.get_poison_dataset()


    def get_poison_dataset(self):
        data_root = "./data"
        stats = {"mean": (0.4914, 0.4822, 0.4465), "std": (0.2023, 0.1994, 0.2010), }
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(mean=stats["mean"], std=stats["std"]),
            ]
        )
        self.train_dataset = torchvision.datasets.CIFAR10(data_root, train=True, download=False,
                                             transform=train_transform)

        indices = list()
        # create array that starts with poisoned images
        #create candidates:
        range_no_id = list(range(50000))
        for image in self.poison_images + self.poison_images_test:
            if image in range_no_id:
                range_no_id.remove(image)

        # add random images to other parts of the batch
        for batches in range(0, self.size_of_secret_dataset):
            range_iter = random.sample(range_no_id, self.batch_size)
            indices.extend(range_iter)
        self.poison_dataset = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size,
                                sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))

        self.poison_test_dataset= torch.utils.data.DataLoader(self.train_dataset,batch_size=128,
                                    sampler=torch.utils.data.sampler.SubsetRandomSampler(range(1000)) )


    def on_train_round_end(self) -> None:
        if self.global_round < self.poison_eopch:
            pass
        else:
            # reset model
            self.resume_original_model()
            self.save_original_model()
            _, acc_p = self.test_poison(self.global_model)
            poison_lr = 0.05
            if not self.baseline:
                if acc_p > 20.0:
                    poison_lr /=50
                if acc_p > 60.0:
                    poison_lr /=100

            retrain_no_times = 15
            step_lr = True
            poison_optimizer = torch.optim.SGD(self.global_model.parameters(), lr = poison_lr,
                                               momentum = 0.9, weight_decay = 0.005)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(poison_optimizer,
                                                             milestones=[0.2 * retrain_no_times,
                                                                         0.8 * retrain_no_times],
                                                             gamma=0.1)
            # is_stepped = False
            # is_stepped_15 = False
            # saved_batch = None
            # try:
            for internal_epoch in range(1, retrain_no_times + 1):
                
                data_iterator = self.poison_dataset
                for batch_id, batch in enumerate(data_iterator):
                    for pos, image in enumerate(self.poison_images):
                        batch[0][pos] = self.train_dataset[image][0]
                        batch[0][pos].add_(torch.FloatTensor(batch[0][pos].shape).normal_(0, 0.01))
                        batch[1][pos] = self.target

                    data, targets = batch
                    data = data.to(self.device)
                    targets = targets.to(self.device)

                    poison_optimizer.zero_grad()
                    output = self.global_model(data)
                    class_loss = nn.functional.cross_entropy(output, targets)
                    update = self._get_para(current=True) - self._get_para(current=False)
                    distance_loss = torch.norm(update)
                    loss = self.alpha_loss * class_loss + (1 - self.alpha_loss) * distance_loss
                    loss.backward()

                    poison_optimizer.step()
                loss_p, acc_p = self.test_poison(self.global_model)
                if step_lr:
                    scheduler.step()
            if not self.baseline:
                ### We scale data according to formula: L = 100*X-99*G = G + (100*X- 100*G).
                clip_rate = self.scale / self.num_byz
                self.update_buffer = (self._get_para(current=True) - self._get_para(current=False)) * clip_rate


    def test_poison(self, model):
        model.eval()
        total_loss = 0.0
        correct = 0.0
        data_iterator = self.poison_test_dataset
        dataset_size = 0

        for batch_id, batch in enumerate(data_iterator):
            for pos in range(len(batch[0])):
                batch[0][pos] = self.train_dataset[random.choice(self.poison_images_test)][0]
                batch[1][pos] = self.target

            data, targets = batch
            data = data.to(self.device).requires_grad_(False)
            targets = targets.to(self.device).requires_grad_(False)
            
            output = model(data)
            total_loss += nn.functional.cross_entropy(output, targets,
                                            reduction='sum').data.item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().to(dtype=torch.float)
            dataset_size += targets.shape[0]

        acc = float(100.0 * (correct / dataset_size))
        total_l = total_loss / dataset_size
        model.train()
        return total_l, acc

    def get_update_benign(self) -> torch.Tensor:
        return torch.nan_to_num(self.updata_benign)

    def get_update_byzantineNoPrune(self) -> torch.Tensor:
        return torch.nan_to_num(self.update_byzantineNoPrune)
        
class ConstscaleAdversary:
    def __init__(self):
        pass

    def omniscient_callback(self, simulator):
        pass
