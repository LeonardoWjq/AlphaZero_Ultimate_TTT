import torch
from torch import nn
from torch.nn import functional as F


class Network(nn.Module):
    def __init__(self, is_regression=True):
        super(Network, self).__init__()
        torch.set_default_dtype(torch.float64)
        self.is_regression = is_regression

        # Layer Definitions
        self.conv1 = nn.Conv2d(in_channels=1, 
                               out_channels=32, 
                               kernel_size=3, 
                               stride=1)

        self.conv2 = nn.Conv2d(in_channels=32, 
                               out_channels=32, 
                               kernel_size=3, 
                               stride=1)
        
        

        self.fc_1 = nn.Linear(32*5*5, 512)
        self.fc_2 = nn.Linear(512,256)
        self.fc_3 = nn.Linear(256, 119)
        self.fc_4 = nn.Linear(128, 64)
        self.scalar = nn.Linear(64, 1)
        self.categorical = nn.Linear(64, 5)
        self.softmax = nn.Softmax(dim=1)
        

        

    def forward(self, x:torch.tensor, y:torch.tensor):
        x = self.conv1(x) # 7*7*32
        x = F.relu(x)

        x = self.conv2(x) # 5*5*32
        x = F.relu(x)


        x = x.view(-1, 32*5*5) # 800
        x = self.fc_1(x) # 256
        x = F.relu(x)

        x = self.fc_2(x) # 256
        x = F.relu(x)

        x = self.fc_3(x) # 119
        x = F.relu(x)

        vec = torch.cat((x,y), dim=1) # 128
        vec = self.fc_4(vec) # 64
        vec = F.relu(vec)

        if self.is_regression:
            out = self.scalar(vec) # 1
        else:
            logits = self.categorical(vec)
            out = self.softmax(logits)
        
        return out