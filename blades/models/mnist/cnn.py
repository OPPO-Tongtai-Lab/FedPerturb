import torch.nn.functional as F
from torch import nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 30, kernel_size=5)
        self.conv2 = nn.Conv2d(30, 50, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(800, 512)
        self.fc2 = nn.Linear(512, 10) # total 453472

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def cnn(**kwargs):
    model = CNN(**kwargs)
    return model

def create_model():
    return MLP(), nn.modules.loss.CrossEntropyLoss()