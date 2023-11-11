from dataclasses import dataclass
import torch
import gc, os, jpcm
import numpy as np
import math
import threading
from threading import Lock
import torch.nn as nn
import logging
import imageio.v3 as imageio
from tqdm import tqdm
from core.viz.viz_salient import pilsaven as pilsave

from .base import PreprocessorBase

logger = logging.getLogger('POD_by_SHM_snapshot-preprocessor')

def make_US(D, D_t, PVE_threshold, max_v, us=True):
    with torch.no_grad():        
        # lower dim in time
        covr = (D_t.real @ D.real)  # [T, T]
        covi = (D_t.imag @ D.imag)
        cov = covr+covi
        del covr,covi
        # print(cov.shape)
        # get eigenvectors and eigenvalues
        eig, Q = torch.linalg.eigh(cov) # [T, in ascending order], [T, T] 
        # flip for descending order
        eig = eig.flip(0)
        Q = Q.flip(1)
        # Q @ torch.diag(eig) @ Q^T = cov
        # D = USV^T -> D_t @ D = V @ S^2 @ V^T, so US = DV = D @ Q
        
        # reduce Q
        PVE = torch.cumsum(eig[1:], dim=0) / torch.sum(eig[1:]) # make sure to skip the mean
        n = min(torch.where(PVE > PVE_threshold)[0][0] + 1, len(eig))
        print(n, PVE[n])
        
        if n > max_v:
            logger.warning(f"Warning: PVE threshold of {PVE_threshold} is too high. Limiting to {max_v} eigenvectors instead.")
            n = max_v
        
        Q_red = Q[:,:n] # [T, n]
        
        if us:
            out = torch.complex(D.real @ Q_red, D.imag @ Q_red) # [S, n]  
        
        else:
        # return U = D @ Q_red @ S^-1
            # out = D @ Q_red @ torch.diag(1/torch.sqrt(eig[:n])) # [S, n]
            Q2 = Q_red @ torch.diag(1/torch.sqrt(eig[:n]))
            out = torch.complex(D.real @ Q2, D.imag @ Q2) , eig[:n] # [S, n] 
            
        
        del D, D_t, cov, eig, Q, Q_red
        gc.collect()
        return out
              
    


class Preprocessor(PreprocessorBase):
    def __init__(self, config):
        cdict = {
            'datadir': 'data', # directory where data is stored
            'eigenvector': lambda var: f'SHM_POD_snap_eigenvector_{var}.npz', # place to store precomputed eigenvectors
            'make_eigenvector': True, # whether to compute eigenvectors or not
            'max_set_eigenvectors': 30, # maximum number of eigenvectors (otherwise uses PVE to determine)
            'max_eigenvectors': 300,
            'PVE_threshold': 0.999, # PVE threshold to determine number of eigenvectors
            'PVE_threshold_2': 0.9995,
            'n_modes': 720, #720 is maximum possible => accurate
            'n_sets': 1, # number of datasets to use, -1 for all
            # 'randomized_svd_k': 10, # number of eigenvectors to compute using randomized SVD
        }
        cdict.update(config)
        for k,v in cdict.items():
            setattr(self, k, v)
        
        super().__init__(config)
        
        self.wp = 'WP_' if self.weather_prediction else ''    
        self.eigenvector_path = lambda var: f"{self.datadir}/{self.wp}{self.eigenvector(var)}"
        self.eigenvector_vis_path =  f"{self.datadir}/{self.wp}SHM_POD_snap_eigen_vis/"
        
        self.cmap = jpcm.get('desert')        
        with torch.no_grad():
            if self.make_eigenvector:
                self.precompute()
                torch.cuda.empty_cache()
                

    
    def precompute(self):
        datasets, shape, _ = super().precompute_scale(use_datasets=True, lazy=True )#self.weather_prediction) # TODO change back

        rows = shape[1]*shape[-2]*shape[-1]
        cols = sum(d.shape[0] for d in datasets)
        
        # approx_mem_req = (8/1024**3) * (rows*cols + cols**2 + rows**2 + rows)
        # if approx_mem_req > 2:
        #     print(f"Warning: Approximate memory requirement is {approx_mem_req:.2f} GB. Will use CPU for eigenvector computation.")
        # device = torch.device('cpu')
        # else:
        devices = range(torch.cuda.device_count())
        
        import torch_harmonics as th
        sht = th.RealSHT(shape[2], shape[3], grid="equiangular", lmax=self.n_modes, mmax=self.n_modes)
        isht = th.InverseRealSHT(shape[2], shape[3], grid="equiangular", lmax=self.n_modes, mmax=self.n_modes)
            

        # for v in tqdm(range(shape[1])):
        logger.info(f'Computing eigenvectors for all variables (each dataset)...')
        # print(dataset.shape)
        # Make SVD
        
        with torch.no_grad():
        
            USs = []
            dsets = datasets if self.n_sets == -1 else datasets[:self.n_sets]
            if len(dsets) > 1:
                for dset in tqdm(dsets):
                    D_t = sht(torch.from_numpy(dset.load()).float()).reshape(dset.shape[0],-1) # [T, S=C*H*W]
                    D = D_t.T # [S, T]
                    USs.append(make_US(D, D_t, self.PVE_threshold, max_v=self.max_set_eigenvectors))
                    gc.collect()
                    
                M = torch.cat(USs, dim=1) # [S, n*n_sets]
            else: 
                M = sht(torch.from_numpy(dsets[0].load()).float()).reshape(dsets[0].shape[0],-1).T # [S, T]
                
            M_t = M.T # [T, S]
            logger.info(f'Computing eigenvectors for all variables (together)...')
            U, eig = make_US(M, M_t, self.PVE_threshold_2, max_v = self.max_eigenvectors,us=False) # [S, n]    #  do we want it downscaled by importance?


        revert = isht(U.T.reshape((-1,self.n_modes,self.n_modes))).reshape((-1,shape[1],shape[2]*shape[3])).reshape((-1, rows)).T # [true S, n]

        eigenvectors = revert.numpy() # input transformation is a = U.T @ x, output transformation is y = U @ a
        evals = eig.numpy()
        print(eigenvectors.shape)
        latent_dimension = eigenvectors.shape[1]
        
        # save eigenvectors
        logger.info(f'Saving eigenvectors ...')
        vdict = {
            'eigenvectors': eigenvectors,
            'eigenvalues' : evals,
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
                imageio.imwrite(f"{self.eigenvector_vis_path}/{v}/{i}.png", imc, format='JPEG')
            
        torch.cuda.synchronize()            
        del datasets, U, eigenvectors, evals, latent_dimension, vdict
        gc.collect()
        torch.cuda.empty_cache()
        
    def load(self, device):
        '''
        [B, T, C, H, W] -> [B, T, latent, H, W]
        '''
        self.scale, self.shift = super().load_scale(device)
        
        data = np.load(self.eigenvector_path(""))
        
        m = min(int(data['latent_dimension']), 150) # artificial limiter for memory
        
        self.eigenvectors = torch.from_numpy(data['eigenvectors'][:,:m]).float()
        self.sing_values = torch.sqrt(torch.from_numpy(data['eigenvalues'][:m])).float()
        latent_dims = [0,m,] # number of latent dimensions
        # latent_dims.insert(0,0)
        rows = data['method'][0] 
        
        eigen = self.eigenvectors.to(device) # unit length axes (loss starts at 300)
        
        ete = self.eigenvectors.T @ self.eigenvectors
        
        ete_inv = torch.linalg.inv(ete)
        del ete
        
        eigenvs_pinv = ete_inv @ self.eigenvectors.T
        del ete_inv
        
        eigen_pinv = eigenvs_pinv.to(device)
        del eigenvs_pinv
        
        # eigen = torch.einsum('sl,l->sl',self.eigenvectors,self.sing_values).to(device) # importance-scaled axes (Mahalanobis, isotropic gaussian result) - (loss starts at 1e26?!)
        # eigen = torch.einsum('sl,l->sl',self.eigenvectors,1/self.sing_values).to(device) # rev-importance-scaled axes (really shrinks small axes) - (loss also starts at 300, but has difficulty decreasing..
        
        def in_tf(eigen, x):
            return torch.einsum('ls,bts->btl',eigen_pinv, (x*self.scale + self.shift).reshape(x.size(0),x.size(1),rows)) 
            
        def out_tf(eigen, a):
            out = torch.einsum('sl,btl->bts',eigen, a) 
            return (out.reshape(out.size(0), out.size(1), -1, self.shapex, self.shapey) - self.shift) / self.scale
            
        self.patch_x = 1
        self.patch_y = 1
        self.latent_dims = latent_dims
        # self.input_transform = lambda x: torch.stack([in_tf(data[v]['method'])(data_torch[v],x[:,v,:,:]) for v in range(self.n_var)],dim=1)
        self.batched_input_transform = lambda x: in_tf(eigen, x).unsqueeze(-1).unsqueeze(-1)
        # self.output_transform = lambda a: torch.stack([out_tf(data[v]['method'])(data_torch[v],a[:,latent_dims[v]:latent_dims[v+1]]).reshape(-1, self.shapex, self.shapey) for v in range(self.n_var)],dim=1)
        self.batched_output_transform = lambda a: out_tf(eigen, a[:,:,:,0,0])