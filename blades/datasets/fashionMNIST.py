# add the implementation by OPPO
from typing import Optional

import torchvision

from .fldataset import FLDataset
import torchvision.transforms as transforms

class FashionMNIST(FLDataset):
    """"""

    def __init__(
        self,
        data_root: str = "./data",
        cache_name: str = "",
        iid: Optional[bool] = True,
        alpha: Optional[float] = 0.1,
        num_clients: Optional[int] = 20,
        seed=1,
        train_data=None,
        test_data=None,
        train_bs: Optional[int] = 32,
    ):
        
        train_dataset = torchvision.datasets.FashionMNIST(
            train=True, download=True, transform=transforms.ToTensor(), root=data_root
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            train=False, download=True, transform=transforms.ToTensor(), root=data_root
        )
        super(FashionMNIST, self).__init__(
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
        )
