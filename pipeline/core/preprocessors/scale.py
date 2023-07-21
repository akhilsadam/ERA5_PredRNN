from dataclasses import dataclass
import torch
import gc, os, jpcm
import numpy as np
import torch.nn as nn
import logging
import imageio.v3 as imageio
from tqdm import tqdm

from .base import PreprocessorBase

logger = logging.getLogger('scale-preprocessor')

class Preprocessor(PreprocessorBase):
    def __init__(self, config):
        cdict = {
            'datadir': 'data', # directory where data is stored
        }
        cdict.update(config)
        for k,v in cdict.items():
            setattr(self, k, v)
            
        super().__init__(config)
        
        self.wp = 'WP_' if self.weather_prediction else ''                    

        #precompute
        self.precompute()
    
    def precompute(self):
        shape, _ = super().precompute_scale(use_datasets=False)
        self.shape = shape
        
    def load(self, device):
        '''
        [B, T, C, H, W] -> [B, T, latent, H, W]
        '''
        self.scale, self.shift = super().load_scale(device)
                    
        self.patch_x = self.shape[2]
        self.patch_y = self.shape[3]
        self.latent_dims = list(range(self.shape[1]+1))
        # self.input_transform = lambda x: torch.stack([in_tf(data[v]['method'])(data_torch[v],x[:,v,:,:]) for v in range(self.n_var)],dim=1)
        self.batched_input_transform = lambda x: x*self.scale + self.shift
        # self.output_transform = lambda a: torch.stack([out_tf(data[v]['method'])(data_torch[v],a[:,latent_dims[v]:latent_dims[v+1]]).reshape(-1, self.shapex, self.shapey) for v in range(self.n_var)],dim=1)
        self.batched_output_transform = lambda a: (a-self.shift)/self.scale