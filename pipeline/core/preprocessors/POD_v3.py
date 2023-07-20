from dataclasses import dataclass
import torch
import gc, os, jpcm
import numpy as np
import torch.nn as nn
import logging
import imageio.v3 as imageio
from tqdm import tqdm

# Patch-based POD, but applying on delta-frame

from .base import PreprocessorBase

logger = logging.getLogger('POD_v3-preprocessor')

def simple_randomized_torch_svd(M, k=10):
    # citation: https://github.com/smortezavi/Randomized_SVD_GPU/blob/master/pytorch_randomized_svd.ipynb
    B = M.clone().detach()
    m, n = B.size()
    transpose = False
    if m < n:
        transpose = True
        B = B.transpose(0, 1)
        m, n = B.size()
    rand_matrix = torch.randn((n,k), dtype=torch.double, device=M.device)  # short side by k
    Q, _ = torch.linalg.qr(B @ rand_matrix)                              # long side by k
    smaller_matrix = (Q.transpose(0, 1) @ B)            # k by short side
    U_hat, s, V = torch.svd(smaller_matrix,False)
    U = (Q @ U_hat)
    return (V.transpose(0, 1), s, U.transpose(0, 1)) if transpose else (U, s, V)

class Preprocessor(PreprocessorBase):
    def __init__(self, config):
        cdict = {
            'datadir': 'data', # directory where data is stored
            'eigenvector': lambda var: f'POD_v3_eigenvector_{var}.npz', # place to store precomputed eigenvectors
            'make_eigenvector': True, # whether to compute eigenvectors or not
            'max_n_eigenvectors': 100, # maximum number of eigenvectors (otherwise uses PVE to determine)
            'PVE_threshold': 0.99, # PVE threshold to determine number of eigenvectors
            # 'randomized_svd_k': 10, # number of eigenvectors to compute using randomized SVD
            'n_patch': 1,
        }
        cdict.update(config)
        for k,v in cdict.items():
            setattr(self, k, v)
            
        self.randomized_svd_k = self.max_n_eigenvectors 
        
        super().__init__(config)
            
        self.wp = 'WP_' if self.weather_prediction else ''    
        self.eigenvector_path = lambda var: f"{self.datadir}/{self.wp}{self.eigenvector(var)}"
        self.eigenvector_vis_path =  f"{self.datadir}/{self.wp}POD_v3_eigen_vis/"
        
        self.cmap = jpcm.get('desert')        
        with torch.no_grad():
            if self.make_eigenvector:
                self.precompute()
                

    
    def precompute(self):
        datasets, shape, _ = super().precompute_scale(use_datasets=True)

        assert shape[-2] // self.n_patch == shape[-2] / self.n_patch, f"Patch size {self.n_patch} does not divide evenly into shape {shape[-2]}"
        assert shape[-1] // self.n_patch == shape[-1] / self.n_patch, f"Patch size {self.n_patch} does not divide evenly into shape {shape[-1]}"

        rows = shape[-2]*shape[-1] // self.n_patch**2
        patch_x = shape[-2] // self.n_patch
        patch_y = shape[-1] // self.n_patch
        cols = sum(d.shape[0] for d in datasets) * self.n_patch**2
        qcols = sum(d.shape[0]-1 for d in datasets) * self.n_patch**2
        
        datasets = [d.reshape(d.shape[0],d.shape[1],-1,self.n_patch,self.n_patch) for d in datasets]
        
        approx_mem_req = (8/1024**3) * (rows*cols + cols**2 + rows**2 + rows)
        if approx_mem_req > 2:
            print(f"Warning: Approximate memory requirement is {approx_mem_req:.2f} GB. Will use CPU for eigenvector computation.")
            device = torch.device('cpu')
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            

        for v in tqdm(range(shape[1])):
            logger.info(f'Computing eigenvectors for variable {v}...')
            # Make data matrix
            dataset = torch.cat([torch.diff(torch.tensor(d[:,v,:,:], dtype=torch.double, device=device),dim=0) for d in datasets],dim=0)
            dataset = dataset.reshape(qcols,rows).T
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
            logger.info(f'Saving eigenvectors for variable {v}...')
            vdict = {
                'eigenvectors': eigenvectors,
                'latent_dimension': latent_dimension,
                'method': [rows,cols]
            }
            np.savez(self.eigenvector_path(v), **vdict)
            
            logger.info(f'Plotting eigenvectors for variable {v}...')
            for i in tqdm(range(latent_dimension)):
                os.makedirs(f"{self.eigenvector_vis_path}/{v}/", exist_ok=True)
                # convert colormap
                imc = self.cmap(eigenvectors[:,i]).reshape(patch_x,patch_y,4)[:,:,:3]
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
            return lambda eigen,x: torch.einsum('sl,btsxy->btlxy',eigen, (x*self.scale + self.shift).reshape(x.size(0),x.size(1),rows, self.n_patch, self.n_patch))# torch.matmul(eigen.T, x.reshape(x.size(0),x.size(1),rows))
                
        def out_tf(eigen,a):
            out = torch.einsum('sl,btlxy->btsxy',eigen, a)
            return out.reshape(out.size(0), out.size(1), self.shapex, self.shapey) / self.scale - self.shift
            
        self.patch_x = self.n_patch
        self.patch_y = self.n_patch
        self.latent_dims = latent_dims
        # self.input_transform = lambda x: torch.stack([in_tf(data[v]['method'])(data_torch[v],x[:,v,:,:]) for v in range(self.n_var)],dim=1)
        def batched_input_transform(x):
            x_diff = torch.diff(x,dim=1) # B T C H W
            # add back zeros in first frame shape
            x_diff = torch.cat((torch.zeros(x_diff.shape[0],1,x_diff.shape[2],x_diff.shape[3],x_diff.shape[4],device=x_diff.device),x_diff),dim=1)
            
            # save first frame
            self.first_frame = x[:,0,:,:,:] #TODO make this thread-safe!!
            
            return torch.cat([in_tf(data[v]['method'])(self.data_torch[v],x_diff[:,:,v,:,:]) for v in range(self.n_var)],dim=2)
        # self.output_transform = lambda a: torch.stack([out_tf(data[v]['method'])(data_torch[v],a[:,latent_dims[v]:latent_dims[v+1]]).reshape(-1, self.shapex, self.shapey) for v in range(self.n_var)],dim=1)
        def batched_output_transform(x):
            a = torch.cumsum(x,dim=1)
            return torch.stack([out_tf(self.data_torch[v],a[:,:,latent_dims[v]:latent_dims[v+1],:,:]) for v in range(self.n_var)],dim=2) + self.first_frame.unsqueeze(1)
        
        self.batched_input_transform = batched_input_transform
        self.batched_output_transform = batched_output_transform