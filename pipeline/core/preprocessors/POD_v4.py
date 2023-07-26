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

logger = logging.getLogger('POD_v4-preprocessor')


class ThreadSafeList():
    # constructor
    def __init__(self):
        # initialize the list
        self._list = list()
        # initialize the lock
        self._lock = Lock()
 
    # add a value to the list
    def append(self, value):
        # acquire the lock
        with self._lock:
            # append the value
            self._list.append(value)
 
    # remove and return the last value from the list
    def pop(self):
        # acquire the lock
        with self._lock:
            # pop a value from the list
            return self._list.pop()
 
    # read a value from the list at an index
    def get(self, index):
        # acquire the lock
        with self._lock:
            # read a value at the index
            return self._list[index]
 
    # return the number of items in the list
    def length(self):
        # acquire the lock
        with self._lock:
            return len(self._list)

# POD, but all variables at once!
import inspect
def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]
def print_trace(objects = None, uname=""):
    gc.collect()
    if objects is None:
        # print_trace(gc.garbage, "unreachable")
        objects = gc.get_objects()
    print(f'--- start GC collect {uname} ---')
    items = {}
    for obj in objects:
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                size = obj.element_size() * obj.nelement() if len(obj.size()) > 0 else 0
                name = retrieve_name(obj)[0]
                print(size, type(obj), obj.size(), name)
                items[name] = size
        except:
            pass
    print(f'--- end GC collect {uname} ---')
    return items

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

def randomized_torch_svd(dataset, devices, m, n, k=100, skip=0, savepath="", nbatch=1):
    # only works on multiple GPU machines
    # citation: https://github.com/smortezavi/Randomized_SVD_GPU/blob/master/pytorch_randomized_svd.ipynb
    # https://discuss.pytorch.org/t/matmul-on-multiple-gpus/33122/3
    device0 = torch.device('cuda:0')
    max_gb = 0.8 * torch.cuda.get_device_properties(device0).total_memory / 1024**3
    
    # transpose = m < n ### TODO
    if m > n:
        transpose = True
        short_dim = n
        long_dim = m
    else:
        transpose = False
        long_dim = n
        short_dim = m
        
    print(f"Sizes are {short_dim} by {long_dim}, with k={k}.")
    total = len(dataset)
    n_patches = math.ceil(total / nbatch)
    dims0 = [d.shape[0] for d in dataset]
    # sum batches together to get correct dim size
    dims = []
    for i in range(n_patches):
        x = 0
        for j in range(nbatch):
            p = (i*nbatch + j)
            if p >= total:
                break
            x += dims0[p]
        dims.append(x)
    
    with torch.no_grad():
        def loader(i, dev, tp, nbatch):            
            tensors = ThreadSafeList()
            ordering = ThreadSafeList()
            
            def tf(j, ts, ord):
                dataseti = dataset[j].load() if type(dataset[j]) is not torch.Tensor else dataset[j] # lazy check
                if tp:
                    dataseti = dataseti.reshape(dataseti.shape[0],m)
                else:
                    dataseti = dataseti.reshape(dataseti.shape[0],m).T
                ts.append(torch.from_numpy(dataseti).float())
                ord.append(j)
                # print(f"Loaded {j}.")
            
            threads = []
            for j in range(i*nbatch, min((i+1)*nbatch, total)):
                t = threading.Thread(target=tf, args=(j, tensors, ordering))
                t.start()
                threads.append(t)
            
            for t in threads:
                t.join()
                    
            data = [tensors.get(i) for i in np.argsort(ordering._list)]
                    
            if tp:
                out0 = torch.cat(data, dim=0)
            else:
                out0 = torch.cat(data, dim=1)
            out = out0.to(dev)
                
            torch.cuda.synchronize()
            del tensors, ordering, data, out0
            gc.collect()
            torch.cuda.empty_cache()
            return out
        
        load = lambda i, dev, tp, batch : loader(i,dev, tp, batch)
          
        def make_Q():  
            def make_Y():
                rand_matrix = torch.randn((long_dim,k), dtype=torch.float, device=device0)         # short side by k
                # pilsave(f"{savepath}rand_matrix.png", jpcm.get('desert'), rand_matrix.cpu().numpy())
                Y = split_mult(short_dim, k, nbatch, n_patches, rand_matrix, dims, load, transpose, devices, skip=skip, mult_order=0, max_gb=max_gb)             # long side by k  # parallelize
                
                del rand_matrix
                gc.collect()
                torch.cuda.empty_cache()
                return Y
            
            Y = make_Y()
            gc.collect()
            torch.cuda.empty_cache()
            try:
                gby = print_trace()['Y'] / 1024**3
                print(f"Y is {gby:.2f} GB ({gby/max_gb*100:.2f}% of GPU memory).")
            except Exception as e:
                print(f"Error: {e}")

            
            # pilsave(f"{savepath}Y.png", jpcm.get('desert'), Y.cpu().numpy())

        
            Q, _ = torch.linalg.qr(Y)                                                       # long side by k  
            del Y, _
            gc.collect()
            torch.cuda.empty_cache()
            # pilsave(f"{savepath}Q.png", jpcm.get('desert'), Q.cpu().numpy())
            return Q
        
        Q = make_Q()
        torch.cuda.empty_cache()     
        print_trace()
        
        smaller_matrix = split_mult(k, long_dim, nbatch, n_patches, Q.transpose(0, 1), dims, load, transpose, devices, skip=skip, mult_order=1, max_gb=max_gb)  # k by short side # parallelize
        print_trace()
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

def split_mult(N, K, nbatch, n_patches, B, dims, load, transpose, devices, skip=0, mult_order=0, max_gb=12):
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

            print_trace()
            
            for i in range(0, ngpu-skip):
                p = (shift + i)
                if p >= n_patches:
                    break
                torch.cuda.set_device(i+skip)
                device = torch.device('cuda:' + str(i+skip))
                # each GPU has a slice of A
                ai = load(p, device, transpose, nbatch)
                A.append(ai)
                del ai

            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            print_trace()
            
            # now let's matmul
            for i in range(0, ngpu-skip):
                p = (shift + i)
                if p >= n_patches:
                    break
                torch.cuda.set_device(i+skip)
                device = torch.device('cuda:' + str(i+skip))
                
                if not transpose and mult_order == 0:
                    B_slice = torch.empty(dims[p], B.size(1), device=device)
                    # print(B_slice.shape, B[sdims[p]:sdims[p]+dims[p],:].shape)
                    B_slice.copy_(B[sdims[p]:sdims[p]+dims[p],:])
                    B_.append(B_slice)
                    del B_slice
                else:
                    db = B.to(device)
                    B_.append(db)
                    del db

            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            print_trace()

            # Step 2: issue the matmul on each GPU
            for i in range(0, ngpu-skip):
                p = (shift + i)
                if p >= n_patches:
                    break
                torch.cuda.set_device(i+skip)
                device = torch.device('cuda:' + str(i+skip))
                
                if transpose and mult_order == 1:
                    C_.append(torch.matmul(B_[i][:,sdims[p]:sdims[p]+dims[p]], A[i]))
                elif not transpose and mult_order == 1:
                    C_.append(torch.matmul(B_[i], A[i]))
                    print(C_[i].shape, B_[i].shape, A[i].shape)
                else:
                    C_.append(torch.matmul(A[i], B_[i]))
                    
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()               
            print_trace()
            
            D_ = []
            for i in range(0, ngpu-skip):
                p = (shift + i)
                if p >= n_patches:
                    break
                torch.cuda.set_device(0)
                
                if (not transpose and mult_order==0) or (transpose and mult_order == 1):
                    torch.cuda.empty_cache()
                    gb = torch.cuda.max_memory_allocated() / 1024**3
                    frac = gb / max_gb
                    add_cycles = math.ceil(frac / (1-frac))
                    
                    print_trace()
                    
                    if add_cycles <= 1:
                        add = C_[i].to(device0)
                        D_.append(add)
                        C.add_(add)
                        torch.cuda.synchronize()
                        del add
                        gc.collect()
                    else:
                        print(f"GPU0 has {gb:.2f} GB ({frac*100:.2f}% of GPU memory) allocated. Will sum using {add_cycles} chunks.")
                        
                        l = C_[i].size(0)
                        step = math.ceil(l/add_cycles)
                        T = torch.empty(step, C_[i].size(1), device=device0)
                        for z in range(add_cycles):
                            a = z * step
                            b = min((z+1) * step, l)
                            
                            if b == l:
                                T = T[:b-a,:]
                
                            T.copy_(C_[i][a:b,:])
                            gb = torch.cuda.max_memory_allocated() / 1024**3
                            print(f"x1 GPU0 has {gb:.2f} GB allocated.")
                            T.add_(C[a:b,:])
                            C[a:b,:].copy_(T)
                            
                            torch.cuda.synchronize()
                            gc.collect()
                            torch.cuda.empty_cache()
                            
                            gb = torch.cuda.max_memory_allocated() / 1024**3
                            print(f"x2 GPU0 has {gb:.2f} GB allocated.")
                            
                elif not transpose and mult_order == 1:
                    dx = C_[i].shape[1]
                    # print(x,dx)
                    C[:,x:x+dx].copy_(C_[i])
                    x += dx
                else:
                    dx = C_[i].shape[0]
                    # print(x,dx)
                    C[x:x+dx,:].copy_(C_[i])
                    x += dx
            for i in A:
                del i
            for i in B_:
                del i
            for i in C_:
                del i  
            for i in D_:
                del i
            del A, C_, B_, D_
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