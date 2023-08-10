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

from typing import Generator, List
import random
import numpy as np
import torch
import copy
from scipy.stats import norm

from blades.clients.client import ByzantineClient, BladesClient
from typing import  Optional
from blades.utils.utils import get_backdoor_pattern


class FedperturbClient(ByzantineClient):
    """
    :param num_clients: Total number of input
    :param num_byzantine: Number of Byzantine input
    """

    def __init__(
        self,
        prune_ration: Optional[float] = 0.1,
        method: Optional[str] = None,
        scale:Optional[float] = 1.2,
        conspire: Optional[bool] = True,
        poison_range: Optional[str] = "lf_conv",
        model: Optional[str] = "resnet18",
        fix_range: Optional[bool] = True,
        ind: Optional[int] = 0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prune_ration = prune_ration
        self.method = method
        # Dynamic scaling, set the upper bound on the change in l2 norm
        # let R be the rest parameter , T be the target parameter
        # then the scale for target parameter is sqrt( (L2_scale^2 * (R^2 + T^2) - R^2) / T^2 )
        self.L2_scale = scale
        self.conspire = conspire
        self.poison_range = poison_range
        self.ind = ind
        self.model = model 
        self.fix_range = fix_range # poison range, fix or random
        
        # self.dev_type = dev_type
        # lf_conv: last few conv
        # ff_conv: first few conv
    def on_train_round_end(self) -> None:
        pass
        

    def get_update_benign(self) -> torch.Tensor:
        return torch.nan_to_num(self.updata_benign)


    def set_conspire(self, poison_data, indices):
        self.updata_benign = self.update_buffer.clone()
        l2_beni = torch.norm(self.updata_benign)
        l2_target = torch.norm(poison_data[indices])
        res_pow = pow(l2_beni,2) - pow(l2_target,2)
        self.dy_scale = pow( (pow(self.L2_scale * l2_beni, 2) - res_pow) / pow(l2_target, 2) , 0.5)
        # target_updata =  poison_data.clone().detach()
        poison_updata =  self.update_buffer[self.beg:self.end].clone().detach()
        if np.isinf(self.dy_scale.to(torch.device("cpu"))) or np.isnan(self.dy_scale.to(torch.device("cpu"))):
            self.dy_scale = torch.tensor(1e8)
        poison_updata[indices] = -self.dy_scale * poison_data[indices].clone().detach()

        self.update_buffer[self.beg:self.end] = poison_updata.clone().detach()
        
        


    def get_target_data(self, target_name, poison_range):
        beg = 0
        # if self.model == "resnet18":
        if (poison_range == "lf_conv" or poison_range == "ff_conv"):
            for name, data in self.global_model.state_dict().items():
                if "num_batches_tracked" in name:
                    continue
                if target_name not in name:
                    end = beg + len(data.view(-1))
                    beg = end
                    continue
                end = beg + len(data.view(-1))
                self.beg = beg
                self.end = end
                self.target_data = self.update_buffer[beg:end].clone().detach()
                self.shape_out = data.shape[0]
                self.shape_in = data.shape[1]
                self.datasize = data.shape[2] * data.shape[3]
                beg = end
        elif poison_range == "bias":
            for name, data in self.global_model.state_dict().items():
                if "num_batches_tracked" in name:
                    continue
                if  target_name not in name:
                    end = beg + len(data.view(-1))
                    beg = end
                    continue
                end = beg + len(data.view(-1))
                self.beg = beg
                self.end = end
                self.target_data = self.update_buffer[beg:end].clone().detach()
                beg = end

class FedperturbAdversary:
    def __init__(self):
        self.fix_ind_flag = False
    

    def omniscient_callback(self, clients: List[BladesClient]):
        byzantine = []
        for client in clients:
            if client.is_byzantine():
                byzantine.append(client)
        byz = byzantine[0]
        if byz.model == "resnet18":
            if byz.conspire and byz.poison_range == "bias":
                chose = "linear.bias"
                # ll = [w.target_data for w in byzantine]
                # cc = torch.stack(ll, 1)
                mean = torch.mean( torch.stack([w.target_data for w in byzantine], 1), 1)
                _, indices = torch.topk(mean, int(len(mean) * byz.prune_ration))
                for client in byzantine:
                    client.set_conspire(mean, indices)
            elif byz.poison_range == "lf_conv" or byz.poison_range == "ff_conv":
                if byz.poison_range == "lf_conv":
                    conv = ["layer4.0.conv1", "layer4.0.conv2", "layer4.1.conv1", "layer4.1.conv2"]
                elif byz.poison_range == "ff_conv":
                    conv = [ "layer1.0.conv1", "layer1.0.conv2", "layer1.1.conv1", "layer1.1.conv2"]
                chose = conv[random.randint(0, len(conv) - 1)]
                # chose = conv[0]
                self.scale_conv(chose, byzantine)
            else:    
                pass
        elif byz.model == "cnn" :
            bias = ["fc2.bias"]
            chose = bias[0]
            self.scale_bias(chose, byzantine)

        elif byz.model == "alexnet" :
            if byz.conspire and byz.poison_range == "bias":
                bias = ["classifier.bias"]
                chose = bias[0]
                self.scale_bias(chose, byzantine)

            elif byz.poison_range == "lf_conv":
                conv = ["features.10.weight"]
                chose = conv[0]
                self.scale_conv(chose, byzantine)
                
        elif byz.model == "vgg11":
            chose = "classifier.6.bias"
            self.scale_bias(chose, byzantine)
        elif byz.model == "vgg11_bn":
            conv = ["features.22.weight", "features.25.weight"]
            chose = conv[random.randint(0, len(conv) - 1)]
            self.scale_conv(chose, byzantine)
        elif byz.model == "cnn_bn":
            conv = ["layer2.0.weight", "layer3.0.weight"]
            chose = conv[random.randint(0, len(conv) - 1)]
            self.scale_conv(chose, byzantine)
        else:
            pass

    def scale_conv(self, chose, byzantine: List[BladesClient]):
        byz = byzantine[0]
        for byza in byzantine:
            byza.get_target_data(chose, byz.poison_range)
        
        mean = torch.mean(torch.stack([w.target_data for w in byzantine], 1), 1)
        norm = torch.norm(mean.view([byz.shape_out, byz.shape_in, -1]), dim = 2).view(-1)
        _, indices = torch.topk(norm, int(len(norm) * byz.prune_ration))
        indicess = list(list(range(ind * byz.datasize, (ind+1) * byz.datasize)) for ind in indices)
        indices_t = list()
        for ind in indicess:
            indices_t = indices_t + ind
        for client in byzantine:
            client.set_conspire(mean, indices_t)
    
    def scale_bias(self, chose, byzantine: List[BladesClient]):
        for byza in byzantine:
            byza.get_target_data(chose, "bias")
        byz = byzantine[0]
        mean = torch.mean(torch.stack([w.target_data for w in byzantine], 1), 1)
        if byz.fix_range and (not self.fix_ind_flag):
            self.fix_ind_flag = True
            _, indices = torch.topk(mean, int(len(mean) * byz.prune_ration))
            self.store_ind = indices
        if self.fix_ind_flag:
            indices = self.store_ind
        else:
            _, indices = torch.topk(mean, int(len(mean) * byz.prune_ration))
        
        for client in byzantine:
            client.set_conspire(mean, indices)
