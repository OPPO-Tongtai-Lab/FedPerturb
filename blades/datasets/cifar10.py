# modified the implementation by OPPO
from typing import Optional

import numpy as np
import torchvision
import torchvision.transforms as transforms

from .fldataset import FLDataset


class CIFAR10(FLDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        path (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to
            True.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(
        self,
        data_root: str = "./data",
        cache_name: str = "",
        iid: Optional[bool] = True,
        alpha: Optional[float] = 0.9,
        num_clients: Optional[int] = 20,
        seed=1,
        train_data=None,
        test_data=None,
        train_bs: Optional[int] = 32,
    ):
        stats = {
            "mean": (0.4914, 0.4822, 0.4465),
            "std": (0.2023, 0.1994, 0.2010),
        }
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=stats["mean"], std=stats["std"]),
            ]
        )
        train_transform = transforms.Compose(
            [
                # transforms.RandomResizedCrop(32, scale=(0.75, 1.0), ratio=(1.0, 1.0)),
                transforms.ToTensor(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(mean=stats["mean"], std=stats["std"]),
                # transforms.RandomErasing(p=0.25),
            ]
        )

        train_dataset = torchvision.datasets.CIFAR10(data_root, train=True, download=True,
                                             transform=train_transform)

        test_dataset = torchvision.datasets.CIFAR10(data_root, train=False, transform=test_transform)
        super(CIFAR10, self).__init__(
            train_dataset,
            test_dataset,
            data_root=data_root,
            cache_name=cache_name,
            iid=iid,
            alpha=alpha,
            num_clients=num_clients,
            seed=seed,
            train_loaders=train_data,
            test_loaders=test_data,
            train_bs=train_bs,
            train_transform=train_transform,
            test_transform=test_transform,
            # model = model
        )
