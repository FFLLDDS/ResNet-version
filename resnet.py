import torch
import torch.nn as nn


class BlockNet(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.iteration = i
        self.num_channels = 16*(2**self.iteration)
        
        self.conv = nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)
        self.bnorm = nn.BatchNorm2d(self.num_channels)
        self.relu = nn.ReLU()
   
    def forward(self, input):
        x = self.conv(input)
        x = self.bnorm(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.bnorm(x)
        x = self.relu(x)
        return x + input


class ShrinkNet(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.iteration = i

        self.in_channels_shrink = 16*(2**(self.iteration-1))
        self.out_channels_shrink = 16*(2**self.iteration)
        self.conv_shrink = nn.Conv2d(in_channels=self.in_channels_shrink, out_channels=self.out_channels_shrink, kernel_size=3, stride=2, padding=1)
        self.bnorm_shrink = nn.BatchNorm2d(self.out_channels_shrink)
        
        self.num_channels = 16*(2**self.iteration)
        self.conv = nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)
        self.bnorm_conv = nn.BatchNorm2d(self.num_channels)

        self.lin_input = int(16*(2**(self.iteration-1)) * (32/(2**(self.iteration-1)))**2)
        self.lin_out1 = int(16*(2**(self.iteration)))
        self.lin_out2 = int(32/(2**(self.iteration)))
        self.lin_output = self.lin_out1 * self.lin_out2**2
        
        self.linear = nn.Linear(self.lin_input, self.lin_output) # Projecting the input of the skip connection to the size of the output after change of dimensions
        self.relu = nn.ReLU()
        
    def forward(self, input):
        x = self.conv_shrink(input)
        x = self.bnorm_shrink(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.bnorm_conv(x)
        x = self.relu(x)
        
        input = input.view(-1, self.lin_input)
        input = self.linear(input)
        input = input.view(-1, self.lin_out1, self.lin_out2, self.lin_out2)
        input = self.bnorm_conv(input)
       
        return x + input


class ResNet(nn.Module):
    def __init__(self, num_blocks=1):
        super().__init__()
        self.num_blocks = num_blocks
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(64*8*8, 10)
        
        self.block0 = BlockNet(0)
        self.block1 = BlockNet(1)
        self.block2 = BlockNet(2)
        
        self.shrink1 = ShrinkNet(1)
        self.shrink2 = ShrinkNet(2)

    def forward(self, x):
        x = self.conv1(x)
        
        for k in range(self.num_blocks):
            x = self.block0(x)
           
        x = self.shrink1(x)
        for j in range(self.num_blocks):
            x = self.block1(x)
            
        x = self.shrink2(x)
        for j in range(self.num_blocks):
            x = self.block2(x)   
            
        x = x.view(-1, 64*8*8)
        x = self.fc(x)
        return x