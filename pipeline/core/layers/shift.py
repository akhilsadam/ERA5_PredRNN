__author__ = 'asadam'

import torch
import torch.nn as nn
import torch.nn.functional as F

class Shift(nn.Module):    
    def __init__(self, in_channels, mid_channels=32):
        super().__init__()
    
        self.in_channels = in_channels
        self.fc_loc = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(True),
            nn.Linear(mid_channels, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def forward(self, x, localization_info):
        xs = localization_info.view(-1, self.in_channels)
        
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x