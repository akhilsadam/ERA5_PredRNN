from dataclasses import dataclass
import torch
import gc, os, jpcm
import numpy as np
import torch.nn as nn
import logging
import imageio.v3 as imageio
from tqdm import tqdm
logger = logging.getLogger('control-preprocessor')

class Preprocessor:
    def __init__(self, config):
        cdict = {
            'datadir': 'data', # directory where data is stored
        }
        cdict.update(config)
        for k,v in cdict.items():
            setattr(self, k, v)
        
        assert self.train_data_paths not in [None,[]], "train_data_paths (training datasets) must be specified and not empty"
        assert self.valid_data_paths not in [None,[]], "valid_data_paths (validation datasets) must be specified and not empty"
        assert self.n_var > 0, "n_var (number of variables) must be specified and greater than 0"
        assert self.shapex > 0, "shapex (x dimension of data) must be specified and greater than 0"
        assert self.shapey > 0, "shapey (y dimension of data) must be specified and greater than 0"

        
    def load(self, device):
        '''
        [B, T, C, H, W] -> [B, T, latent]
        '''        
        def in_tf(x):
            return x.reshape(x.size(0),x.size(1),self.shapex*self.shapey)#torch.matmul(eigen.T, x.reshape(x.size(0),x.size(1),rows))
                
        def out_tf(a):
            return a.reshape(a.size(0), a.size(1), self.shapex, self.shapey)
            
        latent_dims = np.cumsum([self.shapey*self.shapex,]*self.n_var).tolist()
        latent_dims.insert(0,0)
        
        self.latent_dims = latent_dims
        # self.input_transform = lambda x: torch.stack([in_tf(data[v]['method'])(data_torch[v],x[:,v,:,:]) for v in range(self.n_var)],dim=1)
        self.batched_input_transform = lambda x: torch.cat([in_tf(x[:,:,v,:,:]) for v in range(self.n_var)],dim=2)
        # self.output_transform = lambda a: torch.stack([out_tf(data[v]['method'])(data_torch[v],a[:,latent_dims[v]:latent_dims[v+1]]).reshape(-1, self.shapex, self.shapey) for v in range(self.n_var)],dim=1)
        self.batched_output_transform = lambda a: torch.stack([out_tf(a[:,:,latent_dims[v]:latent_dims[v+1]]) for v in range(self.n_var)],dim=2)