# add some helper functions, modified the implementation of some functions by OPPO
import logging
import os
import pickle
from abc import ABC
from functools import partial
from typing import Optional, Generator

import numpy as np
import torch
from sklearn.utils import shuffle

from blades.utils.utils import set_random_seed
from .customdataset import CustomTensorDataset
from torchvision import datasets, transforms
from collections import defaultdict
import random

# logger = logging.getLogger(__name__)


class FLDataset(ABC):
    def __init__(
        self,
        train_set=None,
        test_set=None,
        data_root: str = "./data",
        cache_name: str = "",
        iid: Optional[bool] = True,
        alpha: Optional[float] = 0.9,
        num_clients: Optional[int] = 20,
        num_classes: Optional[int] = 10,
        seed=1,
        train_loaders=None,
        test_loaders=None,
        train_bs: Optional[int] = 32,
        train_transform=None,
        test_transform=None,
        # model: Optional[str] = 'resnet'
    ):
        self.train_transform = train_transform
        self.test_transform = test_transform
        if train_loaders:
            self.train_loaders = train_loaders
            self.test_loaders = test_loaders
            self.train_bs = train_bs
            return

        self.num_classes = num_classes
        self.train_bs = train_bs
        if cache_name == "":
            cache_name = self.__class__.__name__ + ".obj"
        self._data_path = os.path.join(data_root, cache_name)
        self.num_clients = num_clients
        # Meta parameters for data partitioning, for comparison with cache.
        # Regenerate dataset if those parameters are different.
        meta_info = {
            "num_clients": num_clients,
            "data_root": data_root,
            "train_bs": train_bs,
            "iid": iid,
            "alpha": alpha,
            "seed": seed,
        }

        regenerate = True
        # load train_data and test_data if it exit
        if os.path.exists(self._data_path):
            with open(self._data_path, "rb") as f:
                loaded_meta_info = pickle.load(f)
                if loaded_meta_info == meta_info:
                    regenerate = False
                else:
                    print(
                        "arguments for data partitioning didn't match the cache, datasets will be regenerated using the new setting."
                    )
        else:
            print("datasets will be generated using the new setting")

        if regenerate:
            returns = self._generate_datasets(
                train_set,
                test_set,
                iid=iid,
                alpha=alpha,
                num_clients=num_clients,
                seed=seed,
            )
            with open(self._data_path, "wb") as f:
                pickle.dump(meta_info, f)
                for obj in returns:
                    pickle.dump(obj, f)

        assert os.path.isfile(self._data_path)
        with open(self._data_path, "rb") as f:
            (_, train_clients, self.train_loaders, self.test_loaders) = [
                pickle.load(f) for _ in range(4)
            ]

        # assert sorted(train_clients) == sorted(test_clients)
        # self._preprocess()

    def __reduce__(self):
        deserializer = FLDataset
        return (
            partial(
                deserializer,
                train_loaders=self.train_loaders,
                test_loaders=self.test_loaders,
                train_bs=self.train_bs,
                train_transform=self.train_transform,
                test_transform=self.test_transform,
            ),
            (),
        )

    def _generate_datasets(
        self, train_set, test_set, *, iid=True, alpha=0.9, num_clients=20, seed=1
    ):
        train_user_ids = [id for id in range(num_clients)]
        self.train_dataset = train_set
        self.test_dataset = test_set
        indices_per_participant = self.sample_dirichlet_train_data(num_clients, alpha = alpha, iid = iid)

        train_loaders = [self.get_train(indices) for pos, indices in
                         indices_per_participant.items()]

        # train_loaders = train_loaders
        test_loaders = self.get_test()

        return train_user_ids, train_loaders, test_loaders


    def sample_dirichlet_train_data(self, no_participants, iid, alpha=0.9):
        """
            Input: Number of participants and alpha (param for distribution)
            Output: A list of indices denoting data in CIFAR training set.
            Requires: cifar_classes, a preprocessed class-indice dictionary.
            Sample Method: take a uniformly sampled 10-dimension vector as parameters for
            dirichlet distribution to sample number of images in each class.
        """

        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            # if ind in self.params['poison_images'] or ind in self.params['poison_images_test']:
            #     continue
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]
        class_size = len(cifar_classes[0])
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())

        for n in range(no_classes):
            random.shuffle(cifar_classes[n])
            if iid == True:
            # if alpha == 1.0:
                sampled_probabilities = np.array(no_participants * [class_size // no_participants])
            else:
                sampled_probabilities = class_size * np.random.dirichlet(
                    np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                per_participant_list[user].extend(sampled_list)
                cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]

        return per_participant_list

    def get_train(self, indices):
        """
        This method is used along with Dirichlet distribution
        :param params:
        :param indices:
        :return:
        """
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                            batch_size=self.train_bs,
                                            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
                                            # num_workers=16,
                                            pin_memory=True)
        return train_loader

    def get_test(self):

        test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                  batch_size=256,
                                                  shuffle=False,
                                                #   num_workers=8,
                                                #   pin_memory=True
                                                  )

        return test_loader

    def get_clients(self):
        train_user_ids = [str(id) for id in range(self.num_clients)]
        return train_user_ids

    def get_train_loader(self, u_id: str) -> Generator:
        """
        Get the local dataset of given user `id`.
        Args:
            u_id (str): user id.

        Returns: the `generator` of dataset for the given `u_id`.
        """
        return self.train_loaders[int(u_id)]

    def get_train_data(self, u_id, num_batches):
        data = [next(self._train_dls[u_id]) for _ in range(num_batches)]
        return data

    def get_all_test_data(self, u_id):
        return self.test_data
        # return self._test_dls[u_id]
    
    def get_test_loader(self):
        return self.test_loaders


