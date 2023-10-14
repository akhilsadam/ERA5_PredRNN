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

logger = logging.getLogger('POD3D_by_snapshot-preprocessor') 

# modified with compression

def make_US(D, D_t, PVE_threshold, max_v, us=True, full_decomp=False):
    with torch.no_grad():        
        # lower dim in time
        cov = D_t @ D # [T, T]
        # print(cov.shape)
        # get eigenvectors and eigenvalues
        eig, Q = torch.linalg.eigh(cov) # [T, in ascending order], [T, T] 
        # flip for descending order
        eig = eig.flip(0)
        Q = Q.flip(1)
        # Q @ torch.diag(eig) @ Q^T = cov
        # D = USV^T -> D_t @ D = V @ S^2 @ V^T, so US = DV = D @ Q
        
        # reduce Q
        PVE = torch.cumsum(eig[1:]**2, dim=0) / torch.sum(eig[1:]**2) # make sure to skip the mean
        n = min(torch.where(PVE > PVE_threshold)[0][0] + 1, len(eig))
        print(n, PVE[n-1])
        
        if n > max_v:
            logger.warning(f"Warning: PVE threshold of {PVE_threshold} is too high. Limiting to {max_v} eigenvectors instead.")
            n = max_v
        
        Q_red = Q[:,:n] # [T, n]
        
        if us:
            out = D @ Q_red # [S, n] 
            
        
        # return U = D @ Q_red @ S^-1
        out = D @ Q_red @ torch.diag(1/torch.sqrt(eig[:n])) # [S, n]
        
        if full_decomp:
            # make complete reduced SVD
            U = out
            S = torch.diag(torch.sqrt(eig[:n]))
            Vt = Q_red.T
            return U, S, Vt
        
        
        del D, D_t, cov, eig, Q, Q_red
        gc.collect()
        return out
              
    


class Preprocessor(PreprocessorBase):
    def __init__(self, config):
        cdict = {
            'datadir': 'data', # directory where data is stored
            'eigenvector': lambda var: f'POD_snap3d2_eigenvector_{var}.npz', # place to store precomputed eigenvectors
            'make_eigenvector': True, # whether to compute eigenvectors or not
            'max_set_eigenvectors': 100, # maximum number of eigenvectors (otherwise uses PVE to determine)
            'max_eigenvectors': 400,
            'PVE_threshold': 0.999, # PVE threshold to determine number of eigenvectors
            'PVE_threshold_2': 0.9999,
            'PVE_threshold_3': 0.99,
            'n_sets': 1, # number of datasets to use, -1 for all
            'sampling_rate': 1, # sampling rate for new time batches
            # 'randomized_svd_k': 10, # number of eigenvectors to compute using randomized SVD
        }
        cdict.update(config)
        for k,v in cdict.items():
            setattr(self, k, v)
        
        super().__init__(config)
        
        self.wp = 'WP_' if self.weather_prediction else ''    
        self.eigenvector_path = lambda var: f"{self.datadir}/{self.wp}{self.eigenvector(var)}"
        self.eigenvector_vis_path =  f"{self.datadir}/{self.wp}POD_snap3d2_eigen_vis/"
        
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
            

        # for v in tqdm(range(shape[1])):
        logger.info(f'Computing eigenvectors for all variables (each dataset)...')
        # print(dataset.shape)
        # Make SVD
        
        with torch.no_grad():
        
            USs = []
            dsets = datasets if self.n_sets == -1 else datasets[:self.n_sets]
            if len(dsets) > 1:
                for dset in tqdm(dsets):
                    D_t_raw = torch.from_numpy(dset.load()).float().reshape(dset.shape[0],-1) # [T, S=C*H*W]
                    
                    # chopping into new T-sets:
                    D_t = D_t_raw.unfold(0, self.total_length, self.sampling_rate).permute((0,2,1)) # [# sets (new T), total_length, C*H*W]
                    D_t = D_t.reshape((D_t.size(0), -1)) # [# sets (new T), total_length*C*H*W]
                    
                    D = D_t.T # [S, T]
                    USs.append(make_US(D, D_t, self.PVE_threshold, max_v=self.max_set_eigenvectors))
                    gc.collect()
                    
                M = torch.cat(USs, dim=1) # [S, n*n_sets]
            else: 
                D_t_raw = torch.from_numpy(dsets[0].load()).float().reshape(dsets[0].shape[0],-1) # [S, T]
                
                # chopping into new T-sets:
                D_t = D_t_raw.unfold(0, self.total_length, self.sampling_rate).permute((0,2,1)) # [# sets (new T), total_length, C*H*W]
                M = D_t.reshape((D_t.size(0), -1)).T # [# sets (new T), total_length*C*H*W]
                
                
            M_t = M.T # [T, S]
            logger.info(f'Computing eigenvectors for all variables (together)...')
            U = make_US(M, M_t, self.PVE_threshold_2, max_v = self.max_eigenvectors, us=False) # [S, n]   


            logger.info("Compressing eigenvectors by SVD...")
            eU, eS, eVt = make_US(U, U.T, self.PVE_threshold_3, max_v = self.max_eigenvectors, us=False, full_decomp=True) # [S, n]

        eigenvectors = U.numpy() # input transformation is a = U.T @ x, output transformation is y = U @ a
        enU = eU.numpy()
        enS = eS.numpy()
        enVt = eVt.numpy()
        print(eigenvectors.shape)
        latent_dimension = eigenvectors.shape[1]
        
        # save eigenvectors
        logger.info(f'Saving eigenvectors ...')
        vdict = {
            'eU': enU,
            'eS': enS,
            'eVt': enVt,
            'latent_dimension': latent_dimension,
            'shape': shape, # [T, C, H, W]
        }
        np.savez(self.eigenvector_path(""), **vdict)
        
     
        for i in tqdm(range(latent_dimension)):
            # convert colormap
            fev = eigenvectors[:,i]
            qev = fev.reshape(self.total_length, shape[1], shape[2], shape[3]) # [total_length, C, H, W]
            for v in tqdm(range(shape[1])):
                logger.info(f'Plotting eigenvectors for variable {v}...')
                os.makedirs(f"{self.eigenvector_vis_path}/{v}/", exist_ok=True)
                
                sa = self.total_length*shape[1]*shape[2]
                
                ev = qev[:,v,:,:].reshape(sa,-1) # [T*H, W]
                
                nev = (ev - np.min(ev)) / (np.max(ev) - np.min(ev))
                imc = self.cmap(nev).reshape(sa,shape[-1],4)[:,:,:3]
                imc /= np.max(imc)
                imc = (imc*255).astype(np.uint8)                
                imageio.imwrite(f"{self.eigenvector_vis_path}/{v}/{i}.png", imc, format='JPEG')
            
        torch.cuda.synchronize()            
        del datasets, U, eigenvectors, latent_dimension, vdict
        gc.collect()
        torch.cuda.empty_cache()
        
    def load(self, device):
        '''
        [B, T, C, H, W] -> [B, T, latent, H, W]
        '''
        self.scale, self.shift = super().load_scale(device)
        
        data = np.load(self.eigenvector_path(""))
        self.eU = torch.from_numpy(data['eU']).float().to(device)
        self.eS = torch.from_numpy(data['eS']).float().to(device)
        self.eVt = torch.from_numpy(data['eVt']).float().to(device)        
        
        latent_dims = [0,int(data['latent_dimension']),] # number of latent dimensions
        # latent_dims.insert(0,0)
        
        input_length = self.input_length
        total_length = self.total_length
        # predict_length = self.total_length - self.input_length
        shp = data['shape']
        
        def in_tf(x):
            _,c,h,w = shp # [T, C, H, W]
            # make two pairs of indices (input, output)
            
            rescale = (x*self.scale + self.shift)
            
            rescale[:,input_length:] *= 0.0 # zero out output data
            
            flatx = rescale.reshape(x.size(0),x.size(1)*c*h*w)

            return torch.einsum('ma,ac,cl,bm->bl',self.eU, self.eS, self.eVt, flatx).unsqueeze(1) # torch.matmul(eigen.T, x.reshape(x.size(0),x.size(1),rows))
                
        def out_tf(a):
            _,c,h,w = shp
            out = torch.einsum('ma,ac,cl,btl->btm',self.eU, self.eS, self.eVt, a).squeeze(1) # [B, S]
            return (out.reshape(out.size(0), total_length, c, h, w) - self.shift) / self.scale 
            
        self.patch_x = 1
        self.patch_y = 1
        self.latent_dims = latent_dims
        # self.input_transform = lambda x: torch.stack([in_tf(data[v]['method'])(data_torch[v],x[:,v,:,:]) for v in range(self.n_var)],dim=1)
        self.batched_input_transform = lambda x: (in_tf(x)).unsqueeze(-1).unsqueeze(-1)
        # self.output_transform = lambda a: torch.stack([out_tf(data[v]['method'])(data_torch[v],a[:,latent_dims[v]:latent_dims[v+1]]).reshape(-1, self.shapex, self.shapey) for v in range(self.n_var)],dim=1)
        self.batched_output_transform = lambda a: (out_tf(a[:,:,:,0,0]))