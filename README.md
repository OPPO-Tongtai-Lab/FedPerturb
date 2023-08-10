## FedPerturb: Covert Poisoning Attack on Federated Learning via Partial Perturbation

### Overview
Federated learning breaks through the barrier of data owners by allowing them to collaboratively train a federated machine learning model without compromising the privacy of their own data. However, Federation Learning also faces the threat of poisoning attacks, especially from the client model updates, which may impair the accuracy of the global model. To defend against the poisoning attacks, previous work aims to identify the malicious updates in high dimensional spaces. However, we find that the distances in high dimensional spaces cannot identify the changes in a small subset of dimensions, and the small changes may affect the global models severely. Based on this finding, we propose an untargeted poisoning attack under the federated learning setting via the partial perturbations on a small subset of the carefully selected model parameters, and present two attack object selection strategies. We experimentally demonstrate that the proposed attack scheme achieves high attack success rate on five state-of-the-art defense schemes. Furthermore, the proposed attack scheme remains effective at low malicious client ratios and still circumvents three defense schemes with a malicious client ratio as low as 2%.

The implementation code of FedPerturb is blades/attackers/fedperturbclient.py

### Depdendencies 
+ hdbscan==0.8.29   
+ matplotlib==3.3.3   
+ numpy==1.23.5   
+ ruamel.yaml>=0.17.21  
+ ruamel.yaml.clib>=0.2.6   
+ scikit_learn==0.23.2   
+ scipy==1.5.2   
+ timm==0.9.2   
+ torch==1.10.2   
+ torchvision==0.11.3   
+ tqdm==4.53.0   
+ python==3.8   

### Get start

We use a simulator adapted from [Blades: A simulator and benchmark for Byzantine-robust federated Learning with Attacks and Defenses Experimental Simulation](https://github.com/lishenghui/blades) to simulate federated Learning and test FedPerturb.

You can use the following command to test the effectiveness of FedPerturb's attack on the CIFAR10 dataset at iid versus different levels of non-iid with different malicious party ratios for multiple defense schemes.

    sh experiments.sh

For fashionMNIST, you can use command:

    sh famnist_exp.sh

The result of the run will be saved in outputs/

### License

    Copyright 2023, OPPO.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

### Acknowledgement
[Blades](https://github.com/lishenghui/blades)