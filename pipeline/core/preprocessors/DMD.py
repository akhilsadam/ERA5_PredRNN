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

# def simple_randomized_torch_svd(M, k=10):
#     # citation: https://github.com/smortezavi/Randomized_SVD_GPU/blob/master/pytorch_randomized_svd.ipynb
#     B = M.clone().detach()
#     m, n = B.size()
#     transpose = False
#     if m < n:
#         transpose = True
#         B = B.transpose(0, 1)
#         m, n = B.size()
#     rand_matrix = torch.rand((n,k), dtype=torch.double, device=M.device)  # short side by k
#     Q, _ = torch.linalg.qr(B @ rand_matrix)                              # long side by k
#     smaller_matrix = (Q.transpose(0, 1) @ B)            # k by short side
#     U_hat, s, V = torch.svd(smaller_matrix,False)
#     U = (Q @ U_hat)
#     return (V.transpose(0, 1), s, U.transpose(0, 1)) if transpose else (U, s, V)

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
        patch_x = shape[-2] // self.n_patch
        patch_y = shape[-1] // self.n_patch
        cols = sum(d.shape[0] for d in datasets) * self.n_patch**2
        
        datasets = [d.reshape(d.shape[0],d.shape[1],-1,self.n_patch,self.n_patch) for d in datasets]
        
        approx_mem_req = (8/1024**3) * (rows*cols + cols**2 + rows**2 + rows)
        if approx_mem_req > 2:
            print(f"Warning: Approximate memory requirement is {approx_mem_req:.2f} GB.")
        #     device = torch.device('cpu')
        # else:
        #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            

        for v in tqdm(range(shape[1])):
            logger.info(f'Computing eigenvectors for variable {v}...')
            # Make data matrix
            dataset = np.concatenate([d[:,v,:,:] for d in datasets],axis=0)
            dataset = dataset.reshape(cols,rows).T
            # Make DMD
            dmd = DMD(svd_rank=self.randomized_svd_k)
            dmd.fit(dataset)
            # Get PVE and truncate
            # s = np.array([edmd.eigs
            # PVE = np.cumsum(s**2, dim=0) / np.sum(s**2)
            # loc = np.where(PVE > self.PVE_threshold)[0][0]
            # if loc > self.max_n_eigenvectors:
            #     logger.warn(f'PVE threshold {self.PVE_threshold} not reached! Using {self.max_n_eigenvectors} eigenvectors instead of {loc} eigenvectors.')
            #     loc = self.max_n_eigenvectors
            # else:
            #     logger.info(f'PVE threshold {self.PVE_threshold} reached at {loc} eigenvectors.')
            # # truncate
            eigenvectors = dmd.modes
            # input transformation is a = U.T @ x, output transformation is y = U @ a
            latent_dimension = self.randomized_svd_k
            
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
                ev0 = eigenvectors[:,i]
                for ev,rl in zip([ev0.real, ev0.imag],['real','imag']):
                    nev = (ev - np.min(ev)) / (np.max(ev) - np.min(ev))
                    imc = self.cmap(nev).reshape(shape[-2],shape[-1],4)[:,:,:3]                
                    imc /= np.max(imc)
                    imc = (imc*255).astype(np.uint8)                
                    imageio.imwrite(f"{self.eigenvector_vis_path}/{v}/_{i}_{rl}.png", imc, format='JPEG')
            
            del dataset, dmd, eigenvectors, latent_dimension, vdict
            gc.collect()
        
    def load(self, device):
        '''
        [B, T, C, H, W] -> [B, T, latent, H, W]
        '''
        self.scale, self.shift = super().load_scale(device)
        
        data = [np.load(self.eigenvector_path(v)) for v in range(self.n_var)]
        self.data_torch = [torch.from_numpy(d['eigenvectors']).cfloat().to(device) for d in data]
        latent_dims = np.cumsum([data[v]['latent_dimension'] * 2 for v in range(self.n_var)]).tolist()
        latent_dims.insert(0,0)
        self.latent_dims = latent_dims
        
        def in_tf(v,method, e, x):
            rows = method[0] 
            
            patch = lambda x : (x*self.scale + self.shift).reshape(x.size(0), x.size(1), rows, self.n_patch, self.n_patch).permute(0,1,3,4,2).unsqueeze(-1).cfloat() # last dims are rows (spatial), 1
            # etf = lambda e : torch.transpose(e,-2,-1).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            # phx = torch.matmul(etf(e), patch(x)).squeeze(-1).permute(0,1,4,2,3) # X
            # change transpose for pinv
            
            phx = torch.linalg.lstsq(e.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0), patch(x)).solution.squeeze(-1).permute(0,1,4,2,3)
            
            self.state[v] = phx[:,0]
            # self.state[v] = phx
            
            # print(phx.size(),type(e[0]))
            
            ev = lambda pre,post: post / pre # get eigenvalues
            # add missing seq element to beginning (ones)
            ev_add = lambda ev: torch.cat([torch.ones_like(ev[:,0:1,:]), ev], dim=1)
            
            # combine all
            combine = ev_add(ev(phx[:,:-1],phx[:,1:]))  # pre is 
            out = torch.concat([combine.real, combine.imag], dim=-1)
            return out
            
            # least squares performs worse...
            
            # patch = lambda x : (x*self.scale + self.shift).reshape(x.size(0), x.size(1), rows, self.n_patch, self.n_patch).permute(0,1,3,4,2).unsqueeze(-1) # B for lstsq
            # return lambda eigen, x: torch.linalg.lstsq(eigen.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0), \
            #     patch(x)).solution.squeeze(-1).permute(0,1,4,2,3)
            
            # dot product performs badly since not linear
            
            # return lambda eigen,x: torch.einsum('sl,btsxy->btlxy',eigen, (x*self.scale + self.shift).reshape(x.size(0),x.size(1),rows, self.n_patch, self.n_patch))# torch.matmul(eigen.T, x.reshape(x.size(0),x.size(1),rows))
                
        def out_tf(v,eigen,a):
            phx = self.state[v].unsqueeze(1).permute(0,1,3,4,2)
            q = torch.transpose(eigen,-2,-1).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            k = a.size(-1) // 2
            ac = torch.complex(a[...,:k],a[...,k:]).permute(0,1,3,4,2)
            evs = torch.cumprod(ac,dim=1)
            # evs = ac
            
            aqx = torch.mul(evs,phx)
            # print(aqx.shape, q.shape)
            out = torch.matmul(aqx,q).real.permute(0,1,4,2,3)
            return out.reshape(out.size(0), out.size(1), self.shapex, self.shapey) / self.scale - self.shift
            
            # out = torch.einsum('sl,btlxy->btsxy',eigen, a)
            # return out.reshape(out.size(0), out.size(1), self.shapex, self.shapey) / self.scale - self.shift
            
        self.patch_x = self.n_patch
        self.patch_y = self.n_patch
        
        # self.input_transform = lambda x: torch.stack([in_tf(data[v]['method'])(data_torch[v],x[:,v,:,:]) for v in range(self.n_var)],dim=1)
        self.batched_input_transform = lambda x: torch.cat([in_tf(v,data[v]['method'],self.data_torch[v],x[:,:,v,:,:]) for v in range(self.n_var)],dim=2)
        # self.output_transform = lambda a: torch.stack([out_tf(data[v]['method'])(data_torch[v],a[:,latent_dims[v]:latent_dims[v+1]]).reshape(-1, self.shapex, self.shapey) for v in range(self.n_var)],dim=1)
        self.batched_output_transform = lambda a: torch.stack([out_tf(v,self.data_torch[v],a[:,:,latent_dims[v]:latent_dims[v+1],:,:]) for v in range(self.n_var)],dim=2)