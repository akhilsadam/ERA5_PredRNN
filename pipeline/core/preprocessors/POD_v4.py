from dataclasses import dataclass
import torch
import gc, os, jpcm
import numpy as np
import math
import torch.nn as nn
import logging
import imageio.v3 as imageio
from tqdm import tqdm
from core.viz.viz_salient import pilsaven as pilsave

from .base import PreprocessorBase

logger = logging.getLogger('POD_v4-preprocessor')

# POD, but all variables at once!
import inspect
def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]
def print_trace():
    print('--- start GC collect ---')
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(obj.element_size() * obj.nelement() if len(obj.size()) > 0 else 0, type(obj), obj.size(), retrieve_name(obj))
        except:
            pass
    print('--- end GC collect ---')

def simple_randomized_torch_svd(M, k=10):
    # citation: https://github.com/smortezavi/Randomized_SVD_GPU/blob/master/pytorch_randomized_svd.ipynb
    B = M.clone().detach()
    m, n = B.size()
    transpose = False
    if m < n:
        transpose = True
        B = B.transpose(0, 1)
        m, n = B.size()
    rand_matrix = torch.randn((n,k), dtype=torch.float, device=M.device)  # short side by k
    Q, _ = torch.linalg.qr(B @ rand_matrix)                              # long side by k
    smaller_matrix = (Q.transpose(0, 1) @ B)            # k by short side
    U_hat, s, Vt = torch.linalg.svd(smaller_matrix,True)
    U = (Q @ U_hat)
    return (Vt.transpose(0, 1), s, U.transpose(0, 1)) if transpose else (U, s, Vt)

def randomized_torch_svd(dataset, devices, m, n, k=100, skip=0, savepath=""):
    # only works on multiple GPU machines
    # citation: https://github.com/smortezavi/Randomized_SVD_GPU/blob/master/pytorch_randomized_svd.ipynb
    # https://discuss.pytorch.org/t/matmul-on-multiple-gpus/33122/3
    device0 = torch.device('cuda:0')
    
    # transpose = m < n ### TODO
    if True: #m < n:
        transpose = True
        dlong = n
        dshort = m
    else:
        transpose = False
        dshort = n
        dlong = m
    n_patches = len(dataset)
    dims = [d.shape[0] for d in dataset]
    
    with torch.no_grad():
        def loader(i, dev, tp):
            dataseti = dataset[i].load() if type(dataset[i]) is not torch.Tensor else dataset[i] # lazy check
            if tp:
                tensor = torch.from_numpy(dataseti.reshape(dataseti.shape[0],m)).float().to(dev)
            else:
                tensor = torch.from_numpy(dataseti.reshape(dataseti.shape[0],m).T).float().to(dev)
            del dataseti
            gc.collect()
            return tensor
                   
        
        load = lambda i, dev, tp : loader(i,dev, tp)
          
        def make_Q():  
            def make_Y():
                rand_matrix = torch.randn((dshort,k), dtype=torch.float, device=device0)         # short side by k
                # pilsave(f"{savepath}rand_matrix.png", jpcm.get('desert'), rand_matrix.cpu().numpy())
                Y = split_mult(dlong, k, n_patches, rand_matrix, dims, load, transpose, devices, skip=skip, mult_order=0)             # long side by k  # parallelize
                
                del rand_matrix
                gc.collect()
                torch.cuda.empty_cache()
                return Y
            
            Y = make_Y()
            torch.cuda.empty_cache()
            print_trace()
            
            # pilsave(f"{savepath}Y.png", jpcm.get('desert'), Y.cpu().numpy())

        
            Q, _ = torch.linalg.qr(Y)                                                       # long side by k  
            del Y, _
            gc.collect()
            torch.cuda.empty_cache()
            # pilsave(f"{savepath}Q.png", jpcm.get('desert'), Q.cpu().numpy())
            return Q
        
        Q = make_Q()
        torch.cuda.empty_cache()     
        # print_trace()
        
        smaller_matrix = split_mult(k, dshort, n_patches, Q.transpose(0, 1), dims, load, transpose, devices, skip=skip, mult_order=1)  # k by short side # parallelize
        # print_trace()
        # pilsave(f"{savepath}smaller_matrix.png", jpcm.get('desert'), smaller_matrix.cpu().numpy())
        U_hat, s, Vt = torch.linalg.svd(smaller_matrix,True)
        
        if transpose:
            U = Vt.transpose(0, 1)
            # pilsave(f"{savepath}U.png", jpcm.get('desert'), U.cpu().numpy())
            # pilsave(f"{savepath}U_hat.png", jpcm.get('desert'), U_hat.cpu().numpy())
            del smaller_matrix, Q, U_hat
            gc.collect()
            torch.cuda.empty_cache()
            return U, s
        else:
            U = (Q @ U_hat)
            # pilsave(f"{savepath}U.png", jpcm.get('desert'), U.cpu().numpy())
            # pilsave(f"{savepath}U_hat.png", jpcm.get('desert'), U_hat.cpu().numpy())
            del smaller_matrix, Q, U_hat, Vt
            gc.collect()
            torch.cuda.empty_cache()
            return U, s
        
    #     U = (Q @ U_hat)
    # return (Vt, s, U.transpose(0, 1)) if transpose else (U, s, Vt.transpose(0, 1))

def split_mult(N, K, n_patches, B, dims, load, transpose, devices, skip=0, mult_order=0):
    with torch.no_grad():
        device0 = torch.device('cuda:0')
        ngpu = len(devices)
        n_batch = math.ceil(n_patches / (ngpu-skip))
        sdims = np.cumsum(dims).tolist()
        sdims.insert(0,0)
        
        C = torch.empty(N, K, device=device0)
        x = 0
        for p3 in tqdm(range(n_batch)): # assume n_patches >> ngpu, and each patch maximally fills a GPU 
            shift = p3 * (ngpu-skip)
            A = []
            B_ = []   
            C_ = []

            # print_trace()
            
            for i in range(0, ngpu-skip):
                p = (shift + i)
                if p >= n_patches:
                    break
                torch.cuda.set_device(i+skip)
                device = torch.device('cuda:' + str(i+skip))
                # each GPU has a slice of A
                ai = load(p, device, transpose)
                A.append(ai)

            # now let's matmul
            for i in range(0, ngpu-skip):
                p = (shift + i)
                if p >= n_patches:
                    break
                torch.cuda.set_device(i+skip)
                device = torch.device('cuda:' + str(i+skip))
                
                if not transpose:
                    B_slice = torch.empty(dims[p], B.size(1), device=device)
                    B_slice.copy_(B[sdims[p]:sdims[p]+dims[p],:])
                    B_.append(B_slice)
                else:
                    db = B.to(device)
                    B_.append(db)

            # Step 2: issue the matmul on each GPU
            for i in range(0, ngpu-skip):
                p = (shift + i)
                if p >= n_patches:
                    break
                torch.cuda.set_device(i+skip)
                device = torch.device('cuda:' + str(i+skip))
                
                if mult_order == 1:
                    C_.append(torch.matmul(B_[i][:,sdims[p]:sdims[p]+dims[p]], A[i]))
                else:
                    C_.append(torch.matmul(A[i], B_[i]))
                
            D_ = []
            for i in range(0, ngpu-skip):
                p = (shift + i)
                if p >= n_patches:
                    break
                torch.cuda.set_device(0)
                
                if not transpose or mult_order == 1:
                    add = C_[i].to(device0)
                    D_.append(add)
                    C += add
                    del add
                else:
                    dx = C_[i].shape[0]
                    # print(x,dx)
                    C[x:x+dx,:].copy_(C_[i])
                    x += dx
                    # dx = C_[i].shape[1]
                    # # print(x,dx)
                    # C[:,x:x+dx].copy_(C_[i])
                    # x += dx
            for i in A:
                del i
            for i in B_:
                del i
            for i in C_:
                del i  
            for i in D_:
                del i
            del A, C_, B_, D_, ai
            if transpose:
                del db
            gc.collect()
            for i in range(skip, ngpu):
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            # pilsave(f"{savepath}mult_{p3}_{mult_order}.png", jpcm.get('desert'), C.cpu().numpy())
        
    return C

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
                torch.cuda.empty_cache()
                

    
    def precompute(self):
        datasets, shape, _ = super().precompute_scale(use_datasets=True, lazy=self.weather_prediction)

        rows = shape[1]*shape[-2]*shape[-1]
        cols = sum(d.shape[0] for d in datasets)
        
        # approx_mem_req = (8/1024**3) * (rows*cols + cols**2 + rows**2 + rows)
        # if approx_mem_req > 2:
        #     print(f"Warning: Approximate memory requirement is {approx_mem_req:.2f} GB. Will use CPU for eigenvector computation.")
        # device = torch.device('cpu')
        # else:
        devices = range(torch.cuda.device_count())
            

        # for v in tqdm(range(shape[1])):
        logger.info(f'Computing eigenvectors for all variables...')
        # print(dataset.shape)
        # Make SVD
        skip = 1 if self.weather_prediction else 0
        U, s = randomized_torch_svd(datasets, devices, rows, cols, k=self.randomized_svd_k, skip=skip, savepath=self.eigenvector_path(""))
        gc.collect()
        torch.cuda.empty_cache()
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
                imageio.imwrite(f"{self.eigenvector_vis_path}/{v}/{i}.png", imc, format='JPEG')
            
        torch.cuda.synchronize()            
        del datasets, U, s, PVE, loc, eigenvectors, latent_dimension, vdict
        gc.collect()
        torch.cuda.empty_cache()
        
    def load(self, device):
        '''
        [B, T, C, H, W] -> [B, T, latent, H, W]
        '''
        self.scale, self.shift = super().load_scale(device)
        
        data = np.load(self.eigenvector_path(""))
        self.data_torch = torch.from_numpy(data['eigenvectors']).float().to(device)
        latent_dims = [0,int(data['latent_dimension']),] # number of latent dimensions
        # latent_dims.insert(0,0)
        
        def in_tf(method):
            rows = method[0] 
            return lambda eigen,x: torch.einsum('sl,bts->btl',eigen, (x*self.scale + self.shift).reshape(x.size(0),x.size(1),rows))# torch.matmul(eigen.T, x.reshape(x.size(0),x.size(1),rows))
                
        def out_tf(eigen,a):
            out = torch.einsum('sl,btl->bts',eigen, a)
            return (out.reshape(out.size(0), out.size(1), -1, self.shapex, self.shapey) - self.shift) / self.scale 
            
        self.patch_x = 1
        self.patch_y = 1
        self.latent_dims = latent_dims
        # self.input_transform = lambda x: torch.stack([in_tf(data[v]['method'])(data_torch[v],x[:,v,:,:]) for v in range(self.n_var)],dim=1)
        self.batched_input_transform = lambda x: (in_tf(data['method'])(self.data_torch,x)).unsqueeze(-1).unsqueeze(-1)
        # self.output_transform = lambda a: torch.stack([out_tf(data[v]['method'])(data_torch[v],a[:,latent_dims[v]:latent_dims[v+1]]).reshape(-1, self.shapex, self.shapey) for v in range(self.n_var)],dim=1)
        self.batched_output_transform = lambda a: (out_tf(self.data_torch,a[:,:,:,0,0]))