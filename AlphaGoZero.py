import torch
from torch import nn

# https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf
class ConvolutionalBlock(nn.Module):
    def __init__(self):
        super(ConvolutionalBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bnorm = nn.BatchNorm2d(256)

    def forward(self, x):
        x = self.conv(x)
        x = self.bnorm(x)
        x = nn.ReLU(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, inplanes=128, planes=128, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bnorm1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bnorm2 = nn.BatchNorm2d(256)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bnorm1(x)
        x = nn.ReLU(x)

        x = self.conv2(x)
        x = self.bnorm2(x)
        
        x = x + residual
        x = nn.ReLU(x)

        return x

class DualHeaded(nn.Module):
    def __init__(self):
        super(DualHeaded, self).__init__()

        # Policy Head
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1, stride=1) 
        self.bnorm1 = nn.BatchNorm2d(2)
        # Then ReLU
        self.fc1 = nn.Linear(TODO, TODO)
        self.log_probs = nn.LogSoftmax(dim=1)
        
        # Value Head
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1) 
        self.bnorm2 = nn.BatchNorm2d(1)
        # Then ReLU
        self.fc2 = nn.Linear(TODO, 256)
        # Then ReLU
        self.fc3 = nn.Linear(256, 1)
        # Then tanh
    
    def forward(self, x):

        # Policy
        pi = self.conv1(x)
        pi = self.bnorm1(pi)
        pi = nn.ReLU(pi)
        pi = self.fc1(pi)
        pi = self.log_probs(pi).exp()

        # Value
        v = self.conv2(x)
        v = self.bnorm2(v)
        v = nn.ReLU(v)
        v = self.fc2(v)
        v = nn.ReLU(v)
        v = self.fc3(v)
        v = torch.tanh(v)

        return pi, v

class AlphaGoZero(nn.Module):
    def __init__(self):
        super(AlphaGoZero, self).__init__()
        self.conv_block = ConvolutionalBlock()

        for i in range(39):
            setattr(self, "rb%i" % i, ResidualBlock())

        self.dual_head = DualHeaded()
    
    def forward(self, x):
        x = self.conv_block(x)

        for i in range(39):
            s = getattr(self, "rb%i" % i)(x)

        x = self.dual_head(x)
        return x