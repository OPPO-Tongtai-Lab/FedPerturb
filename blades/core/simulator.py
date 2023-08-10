# add some helper functions, modified the implementation of some functions, remove ray by OPPO
import copy
import importlib
import logging
from time import time
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange
import datetime

from blades.clients import RSAClient
from blades.clients.client import BladesClient, ByzantineClient
from blades.datasets.fldataset import FLDataset
from blades.servers import BladesServer, RSAServer
from blades.utils.utils import (
    initialize_logger,
    set_random_seed,
    top1_accuracy,
    get_backdoor_pattern,
)
import os
import random
from blades.models import CCTNet10, CCTNet100, MLP, ResNet18_L, ResNet18, AlexNet,CNN, VGG11, VGG11_bn, CNN_bn

def save_model(file_name=None, model=None, epoch=None, new_folder_name='saved_models_update'):
    if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    filename = "%s/%s_epoch_%s.pth" %(new_folder_name, file_name, epoch)
    torch.save(model.state_dict(), filename)

class Simulator(object):
    """Synchronous and parallel training with specified aggregators.

    :param dataset: FLDataset that consists local data of all input
    :param aggregator: String (name of built-in aggregation scheme) or
                       a callable which takes a list of tensors and returns an
                       aggregated tensor.
    :param num_byzantine: Number of Byzantine input under built-in attack.
                          It should be `0` if you have custom attack strategy.
    :type num_byzantine: int, optional
    :param attack: ``None`` by default. One of the built-in attacks, i.e.,
                    ``None``, ``noise``, ``labelflipping``,
                    ``signflipping``, ``alie``,
                    ``ipm``.
                    It should be ``None`` if you have custom attack strategy.
    :type attack: str
    :param num_actors: Number of ``Ray actors`` that will be created for local
                training.
    :type num_actors: int
    :param mode: Training mode, either ``actor`` or ``trainer``. For large
            scale client population, ``actor mode`` is favorable
    :type mode: str
    :param log_path: The path of logging
    :type log_path: str
    """

    def __init__(
        self,
        dataset: FLDataset,
        model_name: Optional[str] = None,
        configs=None,
        dataset_name: Optional[str] = "cifar10",
        is_iid: Optional[bool] = True,
        num_byzantine: Optional[int] = 0,
        attack: Optional[str] = None,
        attack_kws: Optional[Dict[str, float]] = None,
        adversary_kws: Optional[Dict[str, float]] = None,
        target: Optional[int] = None,
        source: Optional[int] = None,
        aggregator: Union[Callable[[list], torch.Tensor], str] = "mean",
        aggregator_kws: Optional[Dict[str, float]] = None,
        # num_actors: Optional[int] = 1,
        # gpu_per_actor: Optional[float] = 0,
        log_path: str = "./outputs",
        metrics: Optional[dict] = None,
        use_cuda: Optional[bool] = False,
        seed: Optional[int] = None,
        **kwargs,
    ):

        if configs is None:
            configs = {}
        self.configs = configs

        if use_cuda or ("gpu_per_actor" in kwargs and kwargs["gpu_per_actor"] > 0.0):
            self.device = torch.device("cuda")
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            # os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # for debug ,but trainning is 5 times slower
        else:
            self.device = torch.device("cpu")

        # Setup logger
        initialize_logger(log_path)
        if model_name == "resnet18_L": #light
            self.global_model = ResNet18_L().to(self.device)
            self.local_model = ResNet18_L().to(self.device)
        elif model_name == "resnet18":
            self.global_model = ResNet18().to(self.device)
            self.local_model = ResNet18().to(self.device)
        elif model_name == "resnet18_B":
            from torchvision import models
            self.global_model = models.resnet18().to(self.device)
            n_features = self.global_model.fc.in_features
            self.global_model.fc = torch.nn.Linear(n_features, 10)
            self.local_model = copy.deepcopy(self.global_model).to(self.device)
        elif model_name == "alexnet":
            self.global_model = AlexNet().to(self.device)
            self.local_model = AlexNet().to(self.device)
        elif  model_name == "cnn":
            self.global_model = CNN().to(self.device)
            self.local_model = CNN().to(self.device)
        elif  model_name == "vgg11":
            self.global_model = VGG11().to(self.device)
            self.local_model = VGG11().to(self.device)
        elif  model_name == "vgg11_bn":
            self.global_model = VGG11_bn().to(self.device)
            self.local_model = VGG11_bn().to(self.device)
        elif model_name == "cnn_bn":
            self.global_model = CNN_bn().to(self.device)
            self.local_model = CNN_bn().to(self.device)
            
        self.dataset_name = dataset_name
        self.metrics = {"top1": top1_accuracy} if metrics is None else metrics
        self.json_logger = logging.getLogger("stats")
        self.debug_logger = logging.getLogger("debug")
        # self.debug_logger.info(self.__str__())

        self.random_states = {}
        self.omniscient_callbacks = []
        self.checkpoint = os.path.join(log_path, "checkpoint")
        if not os.path.exists(self.checkpoint):
            os.makedirs(self.checkpoint)

        if aggregator_kws is None:
            aggregator_kws = {}
        self._init_aggregator(aggregator=aggregator, aggregator_kws=aggregator_kws)

        if kwargs:
            # User passed in extra keyword arguments but isn't connecting
            # through the simulator. Raise an error, since most likely a typo
            # in keyword
            unknown = ", ".join(kwargs)
            raise RuntimeError(f"Unknown keyword argument(s): {unknown}")

        self.dataset = dataset

        self.target = target
        self.source = source
        if attack_kws is None:
            attack_kws = {}
        
        # attacker's client use poison function overload the function in BladesClient
        # act before or between train step
        self.num_byzantine = num_byzantine
        self.attack = attack
        self._setup_clients(
            attack,
            num_byzantine=num_byzantine,
            attack_kws=attack_kws,
        )

        if adversary_kws is None:
            adversary_kws = {}
        # adversary method act after train and before aggregator
        self._setup_adversary(attack, adversary_kws=adversary_kws)
        # now = datetime.datetime.now()
        # current_time = now.strftime("%D,%H:%M:%S")
        r = {
            "num_clie": len(self.dataset.get_clients()),
            "is_iid": is_iid,
            "num_byzant": num_byzantine,
            "attack": attack,
            "aggregator": aggregator,
            "seed": seed,
            "time": (datetime.datetime.now()+datetime.timedelta(hours=8)).strftime("%F,%H:%M:%S")
        }

        # Output to file
        self.json_logger.info(r)
        self.debug_logger.info(r)
        # set_random_seed(seed, self.device)

    def _init_aggregator(self, aggregator, aggregator_kws):
        if type(aggregator) == str:
            agg_path = importlib.import_module("blades.aggregators.%s" % aggregator)
            agg_scheme = getattr(agg_path, aggregator.capitalize())
            self.aggregator = agg_scheme(self.debug_logger, **aggregator_kws)
        else:
            self.aggregator = aggregator

    def _setup_adversary(self, attack: str, adversary_kws):
        module_path = importlib.import_module("blades.attackers.%sclient" % attack)
        adversary_cls = getattr(
            module_path, "%sAdversary" % attack.capitalize(), lambda: None
        )
        self.adversary = adversary_cls(**adversary_kws) if adversary_cls else None

    def _setup_clients(self, attack: str, num_byzantine, attack_kws):
        import importlib

        if attack is None:
            num_byzantine = 0
        users = self.dataset.get_clients()
        self._clients = {}
        for i, u in enumerate(users):
            u = str(u)
            if i < num_byzantine:
                module_path = importlib.import_module(
                    "blades.attackers.%sclient" % attack
                )
                attack_scheme = getattr(module_path, "%sClient" % attack.capitalize())
                client = attack_scheme(id=u, debug_loger = self.debug_logger, device=self.device,
                                         dataset_name= self.dataset_name, **attack_kws)
                self._register_omniscient_callback(client.omniscient_callback)
            else:
                if self.configs.client == "RSA":
                    per_model = copy.deepcopy(self.global_model)
                    per_opt = torch.optim.SGD(per_model.parameters(), lr=1.0)
                    client = RSAClient(per_model, per_opt, lambda_=0.1, id=u, device=self.device,)
                else:
                    client = BladesClient(id=u, debug_loger = self.debug_logger, 
                                            device=self.device, dataset_name= self.dataset_name)
            self._clients[u] = client

    def _register_omniscient_callback(self, callback):
        self.omniscient_callbacks.append(callback)

    def get_clients(self) -> List:
        r"""Return all clients as a list."""
        return list(self._clients.values())

    def set_trusted_clients(self, ids: List[str]) -> None:
        r"""Set a list of input as trusted. This is usable for trusted-based
        algorithms that assume some clients are known as not Byzantine.

        :param ids: a list of client ids that are trusted
        :type ids: list
        """
        for id in ids:
            self._clients[id].trust()

    def cache_random_state(self) -> None:
        # This function should be used for reproducibility
        if self.device != torch.device("cpu"):
            self.random_states["torch_cuda"] = torch.cuda.get_rng_state()
        self.random_states["torch"] = torch.get_rng_state()
        self.random_states["numpy"] = np.random.get_state()

    def restore_random_state(self) -> None:
        # This function should be used for reproducibility
        if self.device != torch.device("cpu"):
            torch.cuda.set_rng_state(self.random_states["torch_cuda"])
        torch.set_rng_state(self.random_states["torch"])
        np.random.set_state(self.random_states["numpy"])

    def register_attackers(
        self, clients: List[ByzantineClient], replace_indices=None
    ) -> None:
        r"""Register a list of clients as attackers. Those malicious clients
        replace the first few clients.

        Args:
            clients:  a list of Byzantine clients that will replace some of
                        honest ones.
            replace_indices:  a list of indices of clients to be replaced by
                                the Byzantine clients. The length of this
                                list should be equal to that of ``clients``
                                parameter. If it remains ``None``, the first
                                ``n`` clients will be replaced, where ``n`` is
                                 the length of ``clients``.
        """
        if replace_indices:
            assert len(clients) < len(replace_indices)
        else:
            replace_indices = list(range(len(clients)))
        assert len(clients) < len(self._clients)

        client_li = self.get_clients()
        for i in replace_indices:
            id = client_li[i].id()
            clients[i].set_id(id)
            self._clients[id] = clients[i]
            self._register_omniscient_callback(clients[i].omniscient_callback)

    def parallel_call(self, clients, f: Callable[[BladesClient], None]) -> None:
        self.cache_random_state()
        _ = [f(worker) for worker in clients]
        self.restore_random_state()

    def parallel_get(self, clients, f: Callable[[BladesClient], Any]) -> list:
        results = []
        for w in clients:
            self.cache_random_state()
            results.append(f(w))
            self.restore_random_state()
        return results

    def copy_params(self, model, global_model):
        for name, param in global_model.state_dict().items():
            model.state_dict()[name].copy_(param.clone())

    def check_cossim(self, clients: List[BladesClient]):
        byzantine = []
        benign = []

        for client in clients:
            if not client.is_byzantine():
                benign.append(client)
            else:
                byzantine.append(client)
        be_norm = [torch.norm(be.get_update()).to(torch.device("cpu")) for be in benign]
        by_norm = [torch.norm(by.get_update()).to(torch.device("cpu")) for by in byzantine]
        if self.attack == "fedperturb":
            # by_norm_noPrune = [torch.norm(by.get_update_byzantineNoPrune()).to(torch.device("cpu")) for by in byzantine]
            # by_norm_noPrune = list(np.round(np.array(by_norm_noPrune),4))
            by_norm_be = [torch.norm(by.get_update_benign()).to(torch.device("cpu")) for by in byzantine]
            by_norm_be = list(np.round(np.array(by_norm_be),4))
        else:
            by_norm_be = None

        be_norm = list(np.round(np.array(be_norm),4))
        by_norm = list(np.round(np.array(by_norm),4))
        
        self.debug_logger.info(f"Benign norm {be_norm},mean {np.mean(be_norm):.4f},byz {by_norm},by_beni {by_norm_be}")
        if self.attack == "fedperturb":
            scale = []
            for client in byzantine:
                scale.append(client.dy_scale.to(torch.device("cpu")))
            scale = list(np.round(np.array(scale),4))
            self.debug_logger.info(f"Dynamic scaling is {scale}")
        cosine_similarity = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        pudiff = 0
        diff = 0
        if self.attack == "fedperturb":
            for by in byzantine:
                be_sim = []
                by_sim = []
                for be in benign[:10]:
                    be_sim.append(cosine_similarity(by.get_update_benign(), be.get_update()).item())
                    by_sim.append(cosine_similarity(by.get_update(), be.get_update()).item())

                be_sim = list(np.round(np.array(be_sim),4))
                by_sim = list(np.round(np.array(by_sim),4))
                self.debug_logger.info(f"Beni cosSim of {by._id} is {be_sim}, mean {np.mean(be_sim):.4f}")
                self.debug_logger.info(f"byza cosSim of {by._id} is {by_sim}, mean {np.mean(by_sim):.4f}")
                
                diff = (np.mean(by_sim) - np.mean(be_sim)) / np.mean(be_sim)
        else:
            for by in byzantine:
                by_sim = []
                for be in benign[:10]:
                    by_sim.append(cosine_similarity(by.get_update(), be.get_update()).item())
                by_sim = list(np.round(np.array(by_sim),4))
                self.debug_logger.info(f"byza cosSim of {by._id} is {by_sim}, mean {np.mean(by_sim):.4f}")
        
        bee = []
        rd = random.sample(benign, 2)
        for be1 in rd:
            for be2 in benign[:10]: 
                bee.append(cosine_similarity(be1.get_update(), be2.get_update()).item())  
        bee = list(np.round(np.array(bee),4))
        ll = len(rd)
        l2 = len(benign[:10])
        for i in range(ll):
            self.debug_logger.info(f"beny cosSim {rd[i]._id}:  {bee[i*l2: (i+1)*l2]}")    
        #         
        # self.debug_logger.info(f"Diff ratio is {diff:.4f}, prune diff is {pudiff:.4f}")
        self.debug_logger.info(f"Diff ratio is {diff:.4f}")

    def train_actor(
        self, global_round: int, local_steps: int, clients: List[BladesClient],
        lr: float, *args,  **kwargs,
    ) -> list:
        r"""Run local training using ``ray`` actors.

        Args:
            global_round (int): The current global round.
            local_steps (int): The number of local update steps.
            clients (list): A list of clients that perform local training.
            lr (float): Learning rate for client optimizer.
        """
        if self.attack == "constscale":
            client_per_round = random.sample(clients[1:], self.part_size)
            client_per_round[0] = self._clients['0']
        else:
            client_per_round = random.sample(clients, self.part_size)
        
        id_list = []
        global_model = self.server.get_model()
        # The parameter to be copied should be .state_dict() not .named_parameters()!!!!
        # we need  aggtrgate running_mean and running_var expect   !!num_batches_tracked!!  , 
        # num_batches_tracked will lead norm of updata to be very large

        for client in client_per_round:
            model = self.local_model
            self.copy_params(model, global_model)
            if self.client_optimizer == "SGD":
                opt = torch.optim.SGD(model.parameters(), lr=lr,
                                        momentum=0.9, weight_decay=5e-4)
            elif self.client_optimizer == "Adam":
                opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

            client.set_global_model_ref(model)
            local_dataset = self.dataset.get_train_loader(client.id())

            client.train_global_model(
                train_set=local_dataset, global_round=global_round ,local_steps = local_steps, opt=opt)
        torch.cuda.empty_cache()    
        for client in client_per_round:
            self._clients[client._id] = client
            id_list.append(int(client._id))

        # Collusion algorithm, all malicious clients jointly calculate
        if self.adversary and self.num_byzantine:
            self.adversary.omniscient_callback(client_per_round)

        # If there are Byzantine workers, ask them to craft attackers based on
        # the updated settings. Each malicious client computes its own
        for omniscient_attacker_callback in self.omniscient_callbacks:
            omniscient_attacker_callback(client_per_round)

        self.check_cossim(client_per_round)

        self.server.global_update(client_per_round)
        # self.log_variance(global_round, updates)
        return id_list

    def test_actor(self, global_round, batch_size, loss_func):
        """Evaluates the global global_model using test set.

        Args:
            global_round: the current global round number
            batch_size: test batch size

        Returns:
        """
        loader = self.dataset.get_test_loader()
        model = self.server.get_model()
        model.to(self.device)
        model.eval()
        r = {"type": "client_val","E": global_round,"Loss": 0,"Length":0
        }
        for name in self.metrics:
            r[name] = 0
        # if loss_func == "cross_entropy":
        #     lfunc = nn.modules.loss.CrossEntropyLoss()
        for batch_id, batch in enumerate(loader):
            data, target = batch
            data = data.to(self.device)
            target = target.to(self.device)
            output = model(data)
            r["Loss"] += nn.functional.cross_entropy(output, target).item() * len(target)
            r["Length"] += len(target)

            for name, metric in self.metrics.items():
                r[name] += metric(output, target) * len(target)
        for name in self.metrics:
            r[name] /= r["Length"]
        r["Loss"] /= r["Length"]
        backdoor_acc = 0 

        if self.target != None and self.attack == "prune":
            backdoor_pattern = get_backdoor_pattern(self.dataset_name)
            backdoor_acc = self.test_backdoor(loader, backdoor_pattern, self.source, self.target)
        elif self.attack == "constscale":
            _, backdoor_acc = self._clients['0'].test_poison(model)
        
        loss, top1 = self.log_validate(r, backdoor_acc)
        self.debug_logger.info(
            f"Test global round {global_round}, loss: {loss:.4f}, top1: {top1:.4f}, BA: {backdoor_acc:.4f}"
        )
        
        return loss, top1, backdoor_acc

    def test_backdoor(self, test_loader, backdoor_pattern, source_class, target_class):
        model = self.server.get_model()
        model.to(self.device)
        model.eval()
        correct = 0
        n = 0
        x_offset, y_offset = backdoor_pattern.shape[0], backdoor_pattern.shape[1]
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(self.device), target.to(self.device)
            keep_idxs = (target == source_class)
            bk_data = copy.deepcopy(data[keep_idxs])
            bk_target = copy.deepcopy(target[keep_idxs])
            bk_data[:, :, -x_offset:, -y_offset:] = backdoor_pattern
            bk_target[:] = target_class
            output = model(bk_data)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct+= pred.eq(bk_target.view_as(pred)).sum().item()
            n+= bk_target.shape[0]
        return  np.round(100.0*(float(correct) / n), 2)

    def log_variance(self, cur_round, update):
        updates = []
        for client in self._clients.values():
            if not client.is_byzantine():
                updates.append(client.get_update())
        mean_update = torch.mean(torch.vstack(updates), dim=0)
        var_avg = torch.mean(
            torch.var(torch.vstack(updates), dim=0, unbiased=False)
        ).item()
        norm = torch.norm(
            torch.var(torch.vstack(updates), dim=0, unbiased=False)
        ).item()
        avg_norm = torch.norm(mean_update)
        var_norm = torch.sqrt(
            torch.mean(
                torch.tensor(
                    [
                        torch.norm(model_update - mean_update) ** 2
                        for model_update in updates
                    ]
                )
            )
        )
        r = {
            "type": "variance", "Round":   cur_round, "avg": var_avg,
            "norm": norm,       "avg_norm": avg_norm, "VN_ratio": var_norm / avg_norm,
        }

        # Output to file
        self.json_logger.info(r)

    def log_validate(self, metrics, backdoor_acc):
        top1 = metrics["top1"] 
        loss = metrics["Loss"] 
        self.json_logger.info(f"Test E: {metrics['E']}, loss: {loss:.4f}, top1: {top1:.2f}, BA: {backdoor_acc:.2f}")
        return loss, top1


    def run(
        self,
        server_optimizer: Union[torch.optim.Optimizer, str] = "SGD",
        client_optimizer: Union[torch.optim.Optimizer, str] = "SGD",
        loss_func: Optional[str] = "cross_entropy",
        global_rounds: Optional[int] = 1,
        start_rounds: Optional[int] = 0,
        save_interval: Optional[int] = 0,
        local_steps: Optional[int] = 1,
        part_size: Optional[int] = 10,
        validate_interval: Optional[int] = 1,
        test_batch_size: Optional[int] = 64,
        server_lr: Optional[float] = 0.1,
        client_lr: Optional[float] = 0.1,
        server_lr_scheduler: Optional[torch.optim.lr_scheduler.MultiStepLR] = None,
        client_lr_scheduler: Optional[torch.optim.lr_scheduler.MultiStepLR] = None,
        pretrain_dir: str = "./backup",
        dp_kws: Optional[Dict[str, float]] = None,
    ):
        """Run the adversarial training.

        :param global_model: The global global_model for training.
        :type global_model: torch.nn.Module
        :param server_optimizer: The optimizer for server-side optimization.
                                urrently, the `str` type only supports ``SGD``
        :type server_optimizer: torch.optim.Optimizer or str
        :param client_optimizer: Optimizer for client-side optimization.
                            Currently, the ``str`` type only supports ``SGD``
        :type client_optimizer: torch.optim.Optimizer or str
        :param loss: A Pytorch Loss function. See `Python documentation
        <https://pytorch.org/docs/stable/nn.html#loss-functions>`_.
                     Currently, the `str` type only supports ``crossentropy``
        :type loss: str
        :param global_rounds: Number of communication rounds in total.
        :type global_rounds: int, optional
        :param local_steps: Number of local steps of each global round.
        :type local_steps: int, optional
        :param validate_interval: Interval of evaluating the global global_model using
        test set.
        :type validate_interval: int, optional
        :return: None
        """
        if dp_kws:
            dp_kws.update({"dp": True})
        else:
            dp_kws = {}

        global_model = self.global_model
        if self.device != torch.device("cpu"):
            global_model = global_model.to("cuda")

        self.part_size = part_size
        if start_rounds > 1:
            start_rounds_N = start_rounds - 1
            pret_dir = pretrain_dir
            # if self.dataset_name == 'cifar10':
            loaded_params = torch.load(f"{pret_dir}/checkpoint/epoch_{start_rounds_N}.pth", map_location=self.device)

            global_model.load_state_dict(loaded_params)
        else:
            start_rounds = 0  

        # reset_model_weights(global_model)
        self.client_optimizer = client_optimizer
        if server_optimizer == "SGD":
            self.server_opt = torch.optim.SGD(
                global_model.parameters(), lr=server_lr, **dp_kws
            )
        elif server_optimizer == "Adam":
            self.server_opt = torch.optim.Adam(
                global_model.parameters(), lr=server_lr, **dp_kws
            )
        else:
            self.server_opt = server_optimizer

        if save_interval > 0:
            # save_list = list(range(start_rounds, global_rounds, save_interval))
             save_list = list([10,20,50,100,150,200,250,300,400,500,600,700,800,1000,1200,1400,1500])
        else:
            save_list = list()

        # self.client_opt = client_optimizer
        if self.configs.server == "RSA":
            self.server = RSAServer(
                optimizer=self.server_opt,
                model=global_model,
                aggregator=self.aggregator,
            )
        else:
            self.server = BladesServer(
                optimizer=self.server_opt,
                model=global_model,
                aggregator=self.aggregator,
                lr = server_lr,
            )
        for client in self.get_clients():
            client.set_loss(loss_func)
        global_start = time()
        ret = []
        traget_lr = 0.2
        lr = client_lr
        max_top = 0
        with trange(start_rounds, global_rounds + 1) as rounds:
            for global_rounds in rounds:
                round_start = time()

                if global_rounds <= 1000:
                    lr = client_lr
                else:
                    lr = client_lr/10
                
                idlist = self.train_actor(
                    global_rounds, local_steps, self.get_clients(), lr, **dp_kws
                )

                ret.append(time() - round_start)
                
                if global_rounds % validate_interval == 0:
                    loss, top1, BA = self.test_actor(
                        global_round=global_rounds, batch_size=test_batch_size, loss_func = loss_func
                    )
                    max_top = max(max_top, top1)
                    rounds.set_postfix(loss=loss, top1=max_top, BA = BA)
                    if loss > 1000 or np.isnan(loss):
                        raise RuntimeError(f"Loss too big: {loss}")

                if global_rounds + 1 in save_list:
                    filename = f"{self.checkpoint}/epoch_{global_rounds + 1}.pth"
                    torch.save(self.server.get_model().state_dict(), filename)
                self.debug_logger.info(
                    f"E={global_rounds}; Selr = {server_lr:}; Client lr = {lr:5f}; Time cost = {time() - global_start:.2f}, client{idlist}"
                )
                
            return ret

       