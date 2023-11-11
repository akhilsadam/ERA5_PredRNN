from dataclasses import dataclass
import torch
import gc, os, jpcm
import numpy as np
import torch.nn as nn
import logging
import imageio.v3 as imageio
from tqdm import tqdm

from .base import PreprocessorBase

logger = logging.getLogger('SHM-preprocessor')

class Preprocessor(PreprocessorBase):
    def __init__(self, config):
        cdict = {
            'datadir': 'data', # directory where data is stored
            'n_modes': 10,
        }
        cdict.update(config)
        for k,v in cdict.items():
            setattr(self, k, v)
            
        super().__init__(config)
        
        self.wp = 'WP_' if self.weather_prediction else ''                    
        self.flags = []#'spatial',# if this flag is present, then we expect a spatial dataset NTCHW with self.reduced_shape (CHW)
                      
        #precompute
        # self.precompute()
    
    def precompute(self):
        shape, _ = super().precompute_scale(use_datasets=False)
        self.shape = shape
        self.reduced_shape = shape[1:]
        
        
    def load(self, device):
        '''
        [B, T, C, H, W] -> [B, T, latent, H, W]
        '''
        self.scale, self.shift = super().load_scale(device)
        self.shape=(1,7,720,1440)  # TODO fix pipeline
        self.inshape = (7,self.n_modes,self.n_modes)
        
        import torch_harmonics as th
        
        sht = th.RealSHT(self.shape[2], self.shape[3], grid="equiangular", lmax=self.n_modes, mmax=self.n_modes).to(device)
        isht = th.InverseRealSHT(self.shape[2], self.shape[3], grid="equiangular", lmax=self.n_modes, mmax=self.n_modes).to(device)
                    
        self.patch_x = 1 #self.shape[2]
        self.patch_y = 1 #self.shape[3]
        self.latent_dims = [0,self.shape[1]*self.n_modes**2]#list(range(self.shape[1]+1))
        # self.input_transform = lambda x: torch.stack([in_tf(data[v]['method'])(data_torch[v],x[:,v,:,:]) for v in range(self.n_var)],dim=1)
        self.batched_input_transform = lambda x: sht(x*self.scale + self.shift)
        # self.output_transform = lambda a: torch.stack([out_tf(data[v]['method'])(data_torch[v],a[:,latent_dims[v]:latent_dims[v+1]]).reshape(-1, self.shapex, self.shapey) for v in range(self.n_var)],dim=1)
        self.batched_output_transform = lambda a: (isht(a) \
            .reshape((a.size(0),a.size(1),self.shape[1],self.shape[2],self.shape[3]))-self.shift)/self.scale