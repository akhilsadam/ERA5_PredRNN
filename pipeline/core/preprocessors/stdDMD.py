from dataclasses import dataclass
import torch
import gc, os, jpcm
import numpy as np
import torch.nn as nn
import logging
import imageio.v3 as imageio
from tqdm import tqdm
from pydmd import DMD
# Patch-based DMD

from .base import PreprocessorBase

logger = logging.getLogger('DMD-preprocessor')

class Preprocessor(PreprocessorBase):
    def __init__(self, config):
        cdict = {
            'datadir': 'data', # directory where data is stored
            'eigenvector': lambda var: f'DMD_eigenvector_{var}.npz', # place to store precomputed eigenvectors
            'make_eigenvector': True, # whether to compute eigenvectors or not
            'max_n_eigenvectors': 100, # maximum number of eigenvectors (otherwise uses PVE to determine)
            'PVE_threshold': 0.99, # PVE threshold to determine number of eigenvectors
            # 'randomized_svd_k': 10, # number of eigenvectors to compute using randomized SVD
            'n_patch': 1,
        }
        cdict.update(config)
        for k,v in cdict.items():
            setattr(self, k, v)
            
        self.randomized_svd_k = self.max_n_eigenvectors // 2 # 2 for real and imaginary parts
            
        super().__init__(config)
            
        self.wp = 'WP_' if self.weather_prediction else ''    
        self.eigenvector_path = lambda var: f"{self.datadir}/{self.wp}{self.eigenvector(var)}"
        self.eigenvector_vis_path =  f"{self.datadir}/{self.wp}DMD_eigen_vis/"
        
        self.cmap = jpcm.get('desert')        
        with torch.no_grad():
            if self.make_eigenvector:
                self.precompute()
                
        self.state = {}

    
    def precompute(self):
        datasets, shape, _ = super().precompute_scale(use_datasets=True)

        assert shape[-2] // self.n_patch == shape[-2] / self.n_patch, f"Patch size {self.n_patch} does not divide evenly into shape {shape[-2]}"
        assert shape[-1] // self.n_patch == shape[-1] / self.n_patch, f"Patch size {self.n_patch} does not divide evenly into shape {shape[-1]}"

        rows = shape[-2]*shape[-1] // self.n_patch**2
        cols = sum(d.shape[0] for d in datasets) * self.n_patch**2
        
        datasets = [d.reshape(d.shape[0],d.shape[1],-1,self.n_patch,self.n_patch) for d in datasets]
        
        approx_mem_req = (8/1024**3) * (rows*cols + cols**2 + rows**2 + rows)
        if approx_mem_req > 2:
            print(f"Warning: Approximate memory requirement is {approx_mem_req:.2f} GB.")

        for v in tqdm(range(shape[1])):
            logger.info(f'Computing eigenvectors for variable {v}...')
            # Make data matrix
            dataset = np.concatenate([d[:,v,:,:] for d in datasets],axis=0)
            dataset = dataset.reshape(cols,rows).T

            # Make DMD
            dmd = DMD(svd_rank=self.randomized_svd_k)
            dmd.fit(dataset)

            # Get eigenmodes
            eigenvalues = dmd.eigs
            eigenvectors = dmd.modes
            print(eigenvectors)

            # input transformation is a = U.T @ x, output transformation is y = U @ a
            latent_dimension = self.randomized_svd_k
            
            # save eigenvectors
            logger.info(f'Saving eigenvectors for variable {v}...')
            vdict = {
                'eigenvalues': eigenvalues,
                'eigenvectors': eigenvectors,
                'latent_dimension': latent_dimension,
                'method': [rows,cols]
            }
            np.savez(self.eigenvector_path(v), **vdict)
            
            del dataset, dmd, eigenvectors, latent_dimension, vdict
            gc.collect()
            
    def load(self, device):
        '''
        [B, T, C, H, W] -> [B, T, latent, H, W]
        '''
        self.scale, self.shift = super().load_scale(device)
        
        # 1-channel implementation
        data = np.load(self.eigenvector_path(0)) # for v in range(self.n_var) ] # DMD dataset for each channel
        
        # for (i,d) in enumerate(data):
        modes = torch.from_numpy(data['eigenvectors'])
        eigs = torch.from_numpy(data['eigenvalues'])
        
        self.transform = torch.linalg.multi_dot(
                [modes, torch.diag(eigs), torch.linalg.pinv(modes)]
            ).to(device).to(torch.cfloat)

        print(self.transform.size())

        
  
        # self.batched_input_transform =
        # self.batched_output_transform = lambda x: A * x