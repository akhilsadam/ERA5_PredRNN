from dataclasses import dataclass
import torch
import gc
import numpy as np
import torch.nn as nn
import logging
from tqdm import tqdm
logger = logging.getLogger('POD-preprocessor')

def simple_randomized_torch_svd(M, k=10):
    # citation: https://github.com/smortezavi/Randomized_SVD_GPU/blob/master/pytorch_randomized_svd.ipynb
    B = torch.tensor(M)
    m, n = B.size()
    transpose = False
    if m < n:
        transpose = True
        B = B.transpose(0, 1)
        m, n = B.size()
    rand_matrix = torch.rand((n,k), dtype=torch.double, device=M.device)  # short side by k
    Q, _ = torch.qr(B @ rand_matrix)                              # long side by k
    smaller_matrix = (Q.transpose(0, 1) @ B)            # k by short side
    U_hat, s, V = torch.svd(smaller_matrix,False)
    U = (Q @ U_hat)
    return (V.transpose(0, 1), s, U.transpose(0, 1)) if transpose else (U, s, V)

class Preprocessor:
    def __init__(self, config):
        cdict = {
            'datadir': 'data', # directory where data is stored
            'eigenvector': lambda var: f'POD_eigenvector_{var}.npy', # place to store precomputed eigenvectors
            'make_eigenvector': True, # whether to compute eigenvectors or not
            'max_n_eigenvectors': 100, # maximum number of eigenvectors (otherwise uses PVE to determine)
            'PVE_threshold': 0.99, # PVE threshold to determine number of eigenvectors
            'randomized_svd_k': 10, # number of eigenvectors to compute using randomized SVD
        }
        cdict.update(config)
        for k,v in cdict.items():
            setattr(self, k, v)
            
        self.eigenvector_path = lambda var: f"{self.datadir}/{self.eigenvector(var)}"
        
        assert self.train_data_paths not in [None,[]], "train_data_paths (training datasets) must be specified and not empty"
        assert self.valid_data_paths not in [None,[]], "valid_data_paths (validation datasets) must be specified and not empty"
        assert self.n_var > 0, "n_var (number of variables) must be specified and greater than 0"
        assert self.shapex > 0, "shapex (x dimension of data) must be specified and greater than 0"
        assert self.shapey > 0, "shapey (y dimension of data) must be specified and greater than 0"
        
        if self.make_eigenvector:
            self.precompute()
        self.preload()
    
    def precompute(self):
        shape = None # (time, var, shapex, shapey)
        datasets = []
        logger.info('Loading training datasets for precomputation (eigenvectors)...')
        for i,trainset in tqdm(enumerate(self.train_data_paths)):
            data = np.load(trainset, mmap_mode='r')
            try:
                raw = data['input_raw_data']
                shape = raw.shape
                assert len(raw.shape)==4, f"Raw data in {trainset} is not 4D!"
                assert shape[1] == self.n_var, f"Number of variables in {trainset} does not match n_var!"
                assert shape[2] == self.shapex, f"Shape of raw data in {trainset} does not match shapex!"
                assert shape[3] == self.shapey, f"Shape of raw data in {trainset} does not match shapey!"
            except Exception as e:
                print(f'Warning: Failed to load dataset {i}! Skipping... (Exception "{e}" was thrown.)')
            else:
                datasets.append(raw)

        rows = shape[-2]*shape[-1]
        cols = sum(d.shape[0] for d in datasets)
        
        approx_mem_req = (8/1024**3) * (rows*cols + cols**2 + rows**2 + rows)
        if approx_mem_req > 2:
            print(f"Warning: Approximate memory requirement is {approx_mem_req:.2f} GB. Will use CPU for eigenvector computation.")
            device = torch.device('cpu')
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            

        for v in tqdm(range(shape[1])):
            logger.info(f'Computing eigenvectors for variable {v}...')
            # Make data matrix
            dataset = torch.cat([torch.tensor(d[:,v,:,:], dtype=torch.double, device=device) for d in datasets],dim=0)
            dataset = dataset.reshape(cols,rows).T
            # Make SVD
            U, s, V = simple_randomized_torch_svd(dataset, k=self.randomized_svd_k)
            # Get PVE and truncate
            PVE = torch.cumsum(s**2, dim=0) / torch.sum(s**2)
            loc = torch.where(PVE > self.PVE_threshold)[0][0]
            if loc > self.max_n_eigenvectors:
                logger.warn(f'PVE threshold {self.PVE_threshold} not reached! Using {self.max_n_eigenvectors} eigenvectors instead of {loc} eigenvectors.')
                loc = self.max_n_eigenvectors
            # truncate
            if rows < cols:
                eigenvectors = U[:,:loc].cpu().numpy() # input transformation is a = U.T @ x, output transformation is y = U @ a
                input_transform = lambda eigen,x: torch.matmul(eigen.T, x.reshape(-1,rows))
                output_transform = lambda eigen,a: torch.matmul(eigen, a).reshape(-1,shape[-2],shape[-1])
            else:
                eigenvectors = V[:loc,:].cpu().numpy() # input transformation is a = V @ x, output transformation is y = V.T @ a
                input_transform = lambda eigen,x: torch.matmul(eigen, x.reshape(-1,cols))
                output_transform = lambda eigen,a: torch.matmul(eigen.T, a).reshape(-1,shape[-2],shape[-1])
            latent_dimension = loc    
            
            # save eigenvectors
            logger.info(f'Saving eigenvectors for variable {v}...')
            vdict = {
                'eigenvectors': eigenvectors,
                'latent_dimension': latent_dimension,
                'input_transform': input_transform,
                'output_transform': output_transform,
            }
            np.save(self.eigenvector_path(v), **vdict)
            
            del dataset, U, s, V, PVE, loc, eigenvectors, input_transform, output_transform, latent_dimension, vdict
            gc.collect()
        
    def preload(self):
        data = [np.load(self.eigenvector_path(v)) for v in range(self.n_var)]
        latent_dims = np.cumsum([data[v]['latent_dimension'] for v in range(self.n_var)]).tolist()
        latent_dims.insert(0,0)
        self.latent_dims = latent_dims
        self.input_transform = lambda x: torch.cat([data[v]['input_transform'](data[v]['eigenvectors'],x[:,v,:,:]) for v in range(self.n_var)],dim=1)
        self.output_transform = lambda a: torch.stack([data[v]['output_transform'](data[v]['eigenvectors'],a[:,latent_dims[v]:latent_dims[v+1]]).reshape(-1, self.shapex, self.shapey) for v in range(self.n_var)],dim=1)
        