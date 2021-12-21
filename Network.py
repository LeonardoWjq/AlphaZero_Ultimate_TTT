import torch
from torch import nn
from torch.nn import functional as F

# Dual-headed network
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # Layer Definitions
        self.conv1 = nn.Conv2d(in_channels=1, 
                               out_channels=128, 
                               kernel_size=3, 
                               stride=1)
        self.conv2 = nn.Conv2d(in_channels=128, 
                               out_channels=64, 
                               kernel_size=3, 
                               stride=1)
        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=16, 
                               kernel_size=3, 
                               stride=1)
        # Then ReLU
        self.fc_pi = nn.Linear(16*3*3, 81)
        # Then Softmax
        self.fc_v = nn.Linear(16*3*3, 1)
        # Then tanh

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)

        pi = x.view(-1, 16*3*3)
        pi = self.fc_pi(pi)
        pi = F.softmax(pi, dim=1)

        v = x.view(-1, 16*3*3)
        v = self.fc_v(v)
        v = torch.tanh(v)

        return pi, v

