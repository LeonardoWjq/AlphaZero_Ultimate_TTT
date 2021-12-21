import torch
from torch import nn
from torch.nn import functional as F


# Dual-headed network
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        torch.set_default_dtype(torch.float64)

        # Layer Definitions
        self.conv1 = nn.Conv2d(in_channels=1, 
                               out_channels=128, 
                               kernel_size=3, 
                               stride=3)
        self.conv2 = nn.Conv2d(in_channels=128, 
                               out_channels=128, 
                               kernel_size=1, 
                               stride=1)
        self.conv3 = nn.Conv2d(in_channels=128,
                               out_channels=128, 
                               kernel_size=1, 
                               stride=1)

        self.fc_1 = nn.Linear(128*3*3, 512)
        # Then ReLU
        self.fc_2 = nn.Linear(512, 256)
        # Then ReLU
        self.fc_pi = nn.Linear(256, 81)
        # Then Softmax
        

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.view(-1, 128*3*3)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        x = F.relu(x)

        pi = self.fc_pi(x)
        pi = F.softmax(pi, dim=1)


        return pi

