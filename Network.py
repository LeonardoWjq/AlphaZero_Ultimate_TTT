import torch
from torch import nn
from torch.nn import functional as F


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        torch.set_default_dtype(torch.float64)

        # Layer Definitions
        self.conv1 = nn.Conv2d(in_channels=1, 
                               out_channels=32, 
                               kernel_size=3, 
                               stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, 
                               out_channels=32, 
                               kernel_size=3, 
                               stride=1)

        self.fc_1 = nn.Linear(32*5*5, 64)
        self.fc_2 = nn.Linear(64, 64)
        self.fc_3 = nn.Linear(64, 81)
        self.fc_4 = nn.Linear(90,32)
        self.fc_5 = nn.Linear(32,90)
        self.fc_6 = nn.Linear(90,1)

        

    def forward(self, x:torch.tensor, y:torch.tensor):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.view(-1, 32*5*5) # 800
        x = self.fc_1(x) # 64
        g_x = F.relu(x) # 64
        g_x = self.fc_2(g_x) # 64
        x = F.relu(x + g_x) # 64
        x = self.fc_3(x) # 81
        x = F.relu(x) # 81
        vec = torch.cat((x,y), dim=1) # 90
        g_vec = self.fc_4(vec) # 32
        g_vec = F.relu(g_vec) # 32
        g_vec = self.fc_5(g_vec) # 90
        vec = F.relu(vec + g_vec) # 90
        out = self.fc_6(vec) # 1
        return out



    

# class main():
#     test()

# if __name__ == '__main__':
#     main()