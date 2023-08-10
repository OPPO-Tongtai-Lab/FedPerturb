# modified the implementation by OPPO
import math
import os

import torch
import argparse


# from args import options
from blades.core.simulator import Simulator
from blades.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST
from blades.models import CCTNet10, CCTNet100, MLP, ResNet18_L, ResNet18, CNN, AlexNet
from blades.utils.utils import set_random_seed
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_mode", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--global_round",  type=int, default=100)
    parser.add_argument("--local_round",   type=int, default=2, help="local train batchsize num")
    parser.add_argument("--save_interval", type=int, default=0)
    parser.add_argument("--start_rounds",  type=int, default=0)

    parser.add_argument("--batch_size",        type=int, default=64)
    parser.add_argument("--test_batch_size",   type=int, default=256)
    parser.add_argument("--validate_interval", type=int, default=2)

    parser.add_argument("--num_clients",   type=int, default=50)
    parser.add_argument("--part_size",     type=int, default=50, help="Number of participants per round")
    parser.add_argument("--num_byzantine", type=int, default=10)
    parser.add_argument("--lr",            type=float, default=0.01, help="learning rate")
    parser.add_argument("--optimizer",     type=str,   default="SGD", help="SGD for cifar10, Adam for famnist")

    parser.add_argument("--trusted_id", type=str, default="11")
    parser.add_argument("--client", type=str, default=None)
    parser.add_argument("--server", type=str, default=None)

    parser.add_argument("--attack",  type=str,   default="fedperturb",      help="Select attack types.")
    parser.add_argument("--attrate", type=float, default=0.02,         help="Attack parameter selection ratio.")
    parser.add_argument("--attscal", type=float, default=1.5,         help="FedPerturb attack scale rate.")
    parser.add_argument("--agg",     type=str,   default="mean",   help="Aggregator.")
    parser.add_argument("--dataset", type=str,   default="famnist",   help="Dataset")
    parser.add_argument("--model",   type=str,   default="cnn",  help="model")
    
    parser.add_argument("--algorithm",type=str,default="fedavg",
        help="Optimization algorithm, either 'fedavg' or 'fedsgd'.",
    )
    
    parser.add_argument("--non_iid", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--serv_momentum", type=float, default=0.0)
    parser.add_argument("--gpu_per_actor", type=float, default=1)
    parser.add_argument("--server_lr", type=float, default=1)

    parser.add_argument("--dp", action="store_true", default=False)
    # Parameters for DP. They will take effect only if `dp`
    # is  `True`.
    parser.add_argument("--dp_privacy_delta", type=float, default=1e-6)
    parser.add_argument("--dp_privacy_epsilon", type=float, default=1.0)
    parser.add_argument("--dp_clip_threshold", type=float, default=0.5)

    parser.add_argument(
        "--config_path", type=str, default=None, help="Path to config file."
    )

    options = parser.parse_args()
    options.agg = options.agg.lower()
    options.attack = options.attack.lower()
    if options.algorithm == "fedsgd":
        options.local_round = 1
    options.dp_privacy_sensitivity = 2 * options.dp_clip_threshold / options.batch_size
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    if options.config_path:
        options.aggregator_kws = {}
        over_write_args_from_file(options, options.config_path)
        options.agg = options.agg.lower()
        options.attack = options.attack.lower()

    EXP_DIR = os.path.join(ROOT_DIR, f"outputs/{options.dataset}")

    # todo target attack
    options.target = None
    options.source = None

    attack_args = {
        "constscale": {},
        "signflipping": {},
        "noise": {"std": 0.1},
        "labelflipping": {},
        "permutation": {},
        "attackclippedclustering": {},
        "fangattack": { },
        "minmax": {},
        "minsum": {},
        "ipm": {"epsilon": 0.5},
        "alie": {
            # "z": 1.5, 
            "num_clients": options.num_clients,
            "num_byzantine": options.num_byzantine,
        },
        "fedperturb":{
            "prune_ration": options.attrate, "scale": options.attscal,
              "poison_range": "lf_conv", # bias, lf_conv, last few or first few
             "model": options.model, "fix_range": True,
        }
    }
    agg_args = {
        "trimmedmean": {"num_excluded": 10 + 1},
        "median": {},
        "mean": {},
        "signguard": {"agg": "mean"},
        "geomed": {},
        "dnc": {"num_byzantine": 10},
        "autogm": {"lamb": 2.0},
        "clippedclustering": {"max_tau": 2.0, "signguard": True, "linkage": "average"},
        "clustering": {},
        "centeredclipping": {},
        "multikrum": {"num_excluded": 10, "multk": True},
        "fltrust":{},
        "flame":{"part_size": options.part_size, "noise_level": 0.001},
    }

    options.attack_kws = attack_args[options.attack]
    options.aggregator_kws = agg_args[options.agg]

    options.adversary_args = {
        "fangattack": {"num_byz": options.num_byzantine, "agg": options.agg},
        "distancemaximization": {
            "num_byzantine": options.num_byzantine,
            "agg": "trimmedmean",
        },
    }
    
    options.log_dir = (
        EXP_DIR
        + f"/{options.attack}"+ f"/{options.agg}" 
        + f"/b{options.num_byzantine}" + f"_{options.model}"
        + (f"_stRound{options.start_rounds}"
            if options.start_rounds != 0
            else ""
        )
        + ("_attkws_" + "_".join([k +"_" + str(v) for k, v in options.attack_kws.items()])
            if options.attack_kws
            else ""
        )
        + ("_aggkws_" + "_".join([k + str(v) for k, v in options.aggregator_kws.items()])
            if options.aggregator_kws
            else ""
        )
        + f"_c{options.num_clients}" + f"_cPer{options.part_size}"
        + (f"_lr{options.lr * options.server_lr}") + (f"_bs{options.batch_size}")
        + (f"_localround{options.local_round}") + (f"_noniid_{options.alpha}" if options.non_iid else "_iid")
        + (
            f"_privacy_epsilon{options.dp_privacy_epsilon}_clip_threshold"
            f"{options.dp_clip_threshold}"
            if options.dp
            else ""
        )
        + f"_seed{options.seed}"
    )

    options.pret_dir = (  # pretrain checkpoint dir
        f"backup/{options.dataset}"+ f"/{options.agg}"  
        + f"/b0" + f"_{options.model}"
        + ("_aggkws_" + "_".join([k + str(v) for k, v in options.aggregator_kws.items()])
            if options.aggregator_kws
            else ""
        )
        + f"_c{options.num_clients}"+ f"_cPer{options.part_size}"
        + (f"_lr{options.lr * options.server_lr}") + (f"_bs{options.batch_size}")
        + (f"_localround{options.local_round}") + (f"_noniid_{options.alpha}" if options.non_iid else "_iid")
        + (
            f"_privacy_epsilon{options.dp_privacy_epsilon}_clip_threshold"
            f"{options.dp_clip_threshold}"
            if options.dp
            else ""
        )
        + f"_seed{options.seed}"
    )

    if not torch.cuda.is_available():
        print("We currently do not have any GPU on your machine. ")
        options.num_gpus = 0
        options.gpu_per_actor = 0
        set_random_seed(options.seed, torch.device("cpu"))
    else:
        set_random_seed(options.seed, torch.device("cuda"))

    privacy_factor = (
        options.dp_privacy_sensitivity
        * math.sqrt(2 * math.log(1.25 / options.dp_privacy_delta))
        / options.dp_privacy_epsilon
    )
    
    
    print_result = False
    if print_result:
        # Test E: 0, loss: 2.3106, top1: 10.00, BA: 0.00
        E = []
        loss = []
        top1 = []
        BA = []
        import re
        lineRE = re.compile(r'\d+\.\d+|\d+')
        filename = os.path.join(options.log_dir, "stats")
        for line in open(filename, 'r'):
            if "{" in line:
                continue
            m = lineRE.findall(line)
            if not m:
                continue
            E.append(int(m[0]))
            loss.append(float(m[1]))
            top1.append(float(m[3]))
            BA.append(float(m[4]))
        jpgname = os.path.join(options.log_dir, "result.jpg")
        import matplotlib.pyplot as plt
        plt.plot(E, loss, label='loss')
        plt.plot(E, top1, label='top1')
        plt.plot(E, BA, label='BA')
        plt.legend()
        plt.savefig(jpgname)

    

    if not os.path.exists(options.log_dir):
        os.makedirs(options.log_dir)
        
    # Use absolute path name, 
    data_root = os.path.abspath("./data")

    cache_name = (
        options.dataset
        + "_"
        + options.algorithm
        + ("_noniid" if options.non_iid else "")
        + f"_{str(options.num_clients)}_{str(options.seed)}"
        + F"_bs{str(options.batch_size)}"
        + ".obj"
    )
    if options.dataset == "cifar10":
        dataset = CIFAR10(
            data_root   = data_root,
            cache_name  = cache_name,
            train_bs    = options.batch_size,
            num_clients = options.num_clients,
            iid   = not options.non_iid,
            alpha = options.alpha, 
            seed  = options.seed,
        )  # built-in federated cifar10 dataset

    elif options.dataset == "mnist":
        dataset = MNIST(
            data_root=data_root,
            cache_name=cache_name,
            train_bs=options.batch_size,
            num_clients=options.num_clients,
            iid=not options.non_iid,
            alpha = options.alpha, 
            seed  = options.seed,
        )  # built-in federated MNIST dataset
        # model = MLP()
    elif options.dataset == "cifar100":
        dataset = CIFAR100(
            data_root=data_root,
            cache_name=cache_name,
            train_bs=options.batch_size,
            num_clients=options.num_clients,
            iid=not options.non_iid,
            alpha = options.alpha, 
            seed  = options.seed,
        )  # built-in federated cifar100 dataset
        # model = CCTNet100()
    elif options.dataset == "famnist":
        dataset = FashionMNIST(
            data_root=data_root,
            cache_name=cache_name,
            train_bs=options.batch_size,
            num_clients=options.num_clients,
            iid=not options.non_iid,
            alpha = options.alpha, 
            seed  = options.seed,
        )
    else:
        raise NotImplementedError

    # configuration parameters
    conf_args = {
        # "global_model": model,  # global global_model
        "model_name": options.model,
        "dataset_name": options.dataset,
        "is_iid": not options.non_iid,
        "dataset": dataset,
        "aggregator": options.agg,  # defense: robust aggregation
        "aggregator_kws": options.aggregator_kws,
        "num_byzantine": options.num_byzantine,  # number of byzantine input
        "use_cuda": options.gpu_per_actor > 0.0,
        "attack": options.attack,  # attack strategy
        "attack_kws": options.attack_kws,
        "adversary_kws": options.adversary_args[options.attack]
            if options.attack in options.adversary_args
            else {},
        "target":options.target, # poison target , none or class number
        "source":options.source,
        # "num_actors": options.num_actors,  # number of training actors
        # "gpu_per_actor": options.gpu_per_actor,
        "log_path": options.log_dir,
        "seed": options.seed,  # reproducibility
        "configs": options,
    }

    simulator = Simulator(**conf_args)

    if options.trusted_id:
        simulator.set_trusted_clients([options.trusted_id])

    if options.algorithm == "fedsgd":
        # opt = torch.optim.SGD(
        #     model.parameters(), lr=options.server_lr, momentum=options.serv_momentum
        # )
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     opt, milestones=[2000, 3000, 5000], gamma=0.5
        # )
        assert options.local_round == 1, "fedsgd requires that only one SGD is taken."

        # runtime parameters
        run_args = {
            "client_optimizer": "SGD",  # server_opt, server optimizer
            # "server_optimizer": opt,  # client optimizer
            "loss": "cross_entropy",  # loss function
            "global_rounds": options.global_round,  # number of global rounds
            "local_steps": options.local_round,
            "client_lr": 1.0,
            "validate_interval": options.validate_interval,
            # "server_lr_scheduler": lr_scheduler,
            "dp_kws": {
                "clip_threshold": options.dp_clip_threshold,
                "noise_factor": privacy_factor,
            }
            if options.dp
            else {},
        }

    elif options.algorithm == "fedavg":

        # runtime parameters
        run_args = {
            "server_optimizer": options.optimizer,  # server_opt, server optimizer
            "client_optimizer": options.optimizer,  # client optimizer
            "loss_func": "cross_entropy",  # loss function
            "global_rounds": options.global_round,  # number of global rounds
            "start_rounds": options.start_rounds,
            "save_interval": options.save_interval,
            "part_size": options.part_size,
            "local_steps": options.local_round,  # number of seps
            "client_lr": options.lr,  # learning rateteps per round
            "server_lr": options.server_lr,
            "validate_interval": options.validate_interval,
            "pretrain_dir": options.pret_dir,
            # "client_lr_scheduler": lr_scheduler,
        }

    else:
        raise NotImplementedError

    simulator.run(**run_args)
