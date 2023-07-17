from dataclasses import dataclass
import torch
import gc, os, jpcm
import numpy as np
import torch.nn as nn
import logging
import imageio.v3 as imageio
from tqdm import tqdm

from .base import PreprocessorBase

logger = logging.getLogger('POD_v4-preprocessor')

# POD, but all variables at once!

def simple_randomized_torch_svd(M, k=10):
    # citation: https://github.com/smortezavi/Randomized_SVD_GPU/blob/master/pytorch_randomized_svd.ipynb
    B = M.clone().detach()
    m, n = B.size()
    transpose = False
    if m < n:
        transpose = True
        B = B.transpose(0, 1)
        m, n = B.size()
    rand_matrix = torch.rand((n,k), dtype=torch.float, device=M.device)  # short side by k
    Q, _ = torch.linalg.qr(B @ rand_matrix)                              # long side by k
    smaller_matrix = (Q.transpose(0, 1) @ B)            # k by short side
    U_hat, s, V = torch.svd(smaller_matrix,False)
    U = (Q @ U_hat)
    return (V.transpose(0, 1), s, U.transpose(0, 1)) if transpose else (U, s, V)

class Preprocessor(PreprocessorBase):
    def __init__(self, config):
        cdict = {
            'datadir': 'data', # directory where data is stored
            'eigenvector': lambda var: f'POD_4_eigenvector_{var}.npz', # place to store precomputed eigenvectors
            'make_eigenvector': True, # whether to compute eigenvectors or not
            'max_n_eigenvectors': 100, # maximum number of eigenvectors (otherwise uses PVE to determine)
            'PVE_threshold': 0.99, # PVE threshold to determine number of eigenvectors
            # 'randomized_svd_k': 10, # number of eigenvectors to compute using randomized SVD
        }
        cdict.update(config)
        for k,v in cdict.items():
            setattr(self, k, v)
        
        self.randomized_svd_k = self.max_n_eigenvectors 
            
        super().__init__(config)
        
        self.wp = 'WP_' if self.weather_prediction else ''    
        self.eigenvector_path = lambda var: f"{self.datadir}/{self.wp}{self.eigenvector(var)}"
        self.eigenvector_vis_path =  f"{self.datadir}/{self.wp}POD_4_eigen_vis/"
        
        self.cmap = jpcm.get('desert')        
        with torch.no_grad():
            if self.make_eigenvector:
                self.precompute()
                

    
    def precompute(self):
        datasets, shape, _ = super().precompute_scale(use_datasets=True)

        rows = shape[1]*shape[-2]*shape[-1]
        cols = sum(d.shape[0] for d in datasets)
        
        # approx_mem_req = (8/1024**3) * (rows*cols + cols**2 + rows**2 + rows)
        # if approx_mem_req > 2:
        #     print(f"Warning: Approximate memory requirement is {approx_mem_req:.2f} GB. Will use CPU for eigenvector computation.")
        device = torch.device('cpu')
        # else:
        #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            

        # for v in tqdm(range(shape[1])):
        logger.info(f'Computing eigenvectors for all variables...')
        # Make data matrix
        dataset = torch.cat([torch.tensor(d, dtype=torch.float, device=device) for d in datasets],dim=0)
        dataset = dataset.reshape(cols,rows).T
        # print(dataset.shape)
        # Make SVD
        U, s, V = simple_randomized_torch_svd(dataset, k=self.randomized_svd_k)
        # Get PVE and truncate
        PVE = torch.cumsum(s**2, dim=0) / torch.sum(s**2)
        loc = torch.where(PVE > self.PVE_threshold)[0][0]
        if loc > self.max_n_eigenvectors:
            logger.warn(f'PVE threshold {self.PVE_threshold} not reached! Using {self.max_n_eigenvectors} eigenvectors instead of {loc} eigenvectors.')
            loc = self.max_n_eigenvectors
        else:
            logger.info(f'PVE threshold {self.PVE_threshold} reached at {loc} eigenvectors.')
        # truncate
        eigenvectors = U[:,:loc].cpu().numpy() # input transformation is a = U.T @ x, output transformation is y = U @ a
        latent_dimension = loc.item()
        
        # save eigenvectors
        logger.info(f'Saving eigenvectors ...')
        vdict = {
            'eigenvectors': eigenvectors,
            'latent_dimension': latent_dimension,
            'method': [rows,cols]
        }
        np.savez(self.eigenvector_path(""), **vdict)
        
     
        for i in tqdm(range(latent_dimension)):
            # convert colormap
            fev = eigenvectors[:,i]
            qev = fev.reshape(shape[1],shape[-2],shape[-1])
            for v in tqdm(range(shape[1])):
                logger.info(f'Plotting eigenvectors for variable {v}...')
                os.makedirs(f"{self.eigenvector_vis_path}/{v}/", exist_ok=True)
                
                ev = qev[v]
                
                nev = (ev - np.min(ev)) / (np.max(ev) - np.min(ev))
                imc = self.cmap(nev).reshape(shape[-2],shape[-1],4)[:,:,:3]
                imc /= np.max(imc)
                imc = (imc*255).astype(np.uint8)                
                imageio.imwrite(f"{self.eigenvector_vis_path}/{v}/_{i}.png", imc, format='JPEG')
            
        del dataset, U, s, V, PVE, loc, eigenvectors, latent_dimension, vdict
        gc.collect()
        
    def load(self, device):
        '''
        [B, T, C, H, W] -> [B, T, latent, H, W]
        '''
        self.scale, self.shift = super().load_scale(device)
        
        data = [np.load(self.eigenvector_path(v)) for v in range(self.n_var)]
        self.data_torch = [torch.from_numpy(d['eigenvectors']).float().to(device) for d in data]
        latent_dims = np.cumsum([data[v]['latent_dimension'] for v in range(self.n_var)]).tolist()
        latent_dims.insert(0,0)
        
        def in_tf(method):
            rows = method[0] 
            return lambda eigen,x: torch.einsum('sl,bts->btl',eigen, (x*self.scale + self.shift).reshape(x.size(0),x.size(1),rows))# torch.matmul(eigen.T, x.reshape(x.size(0),x.size(1),rows))
                
        def out_tf(eigen,a):
            out = torch.einsum('sl,btl->bts',eigen, a)
            return out.reshape(out.size(0), out.size(1), self.shapex, self.shapey) / self.scale - self.shift
            
        self.patch_x = 1
        self.patch_y = 1
        self.latent_dims = latent_dims
        # self.input_transform = lambda x: torch.stack([in_tf(data[v]['method'])(data_torch[v],x[:,v,:,:]) for v in range(self.n_var)],dim=1)
        self.batched_input_transform = lambda x: torch.cat([in_tf(data[v]['method'])(self.data_torch[v],x[:,:,v,:,:]) for v in range(self.n_var)],dim=2).unsqueeze(-1).unsqueeze(-1)
        # self.output_transform = lambda a: torch.stack([out_tf(data[v]['method'])(data_torch[v],a[:,latent_dims[v]:latent_dims[v+1]]).reshape(-1, self.shapex, self.shapey) for v in range(self.n_var)],dim=1)
        self.batched_output_transform = lambda a: torch.stack([out_tf(self.data_torch[v],a[:,:,latent_dims[v]:latent_dims[v+1],0,0]) for v in range(self.n_var)],dim=2)