import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):
    def __init__(self, n_feature, h1_units, h2_units):
        super(Net, self).__init__()
        self.h1 = nn.Linear(n_feature, h1_units)
        #self.bn1 = nn.BatchNorm1d(h1_units)
        self.h2 = nn.Linear(h1_units, h2_units)
        #self.bn2 = nn.BatchNorm1d(h2_units)
        self.out = nn.Linear(h2_units, 1)

    def forward(self, x):
        x = self.h1(x)
        x = F.relu(x)
        x = self.h2(x)
        x = F.relu(x)
        x = self.out(x)
        return x
