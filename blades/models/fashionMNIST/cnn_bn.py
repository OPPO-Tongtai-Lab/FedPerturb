# add the implementation by OPPO
import math
import torch.nn as nn
import torch.nn.init as init

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(   
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16), 
            nn.ReLU()) #16, 28, 28
        self.pool1=nn.MaxPool2d(2) #16, 14, 14
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU())#32, 12, 12
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU()) #64, 10, 10
        self.pool2=nn.MaxPool2d(2)  #64, 5, 5
        self.fc = nn.Linear(5*5*64, 10)
    def forward(self, x):
        out = self.layer1(x)
        out=self.pool1(out)
        out = self.layer2(out)
        out=self.layer3(out)
        out=self.pool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def CNN_bn():
    return CNN()