import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 30)
        self.layer2 = nn.Linear(30, 1)
        # this line breaks the network
        # self.layer2 = nn.Linear(29, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x
