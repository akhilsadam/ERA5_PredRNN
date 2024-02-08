__author__ = 'asadam'

import torch
import torch.nn as nn
import torch.nn.functional as F


# problem with all these approaches is that the mode matrix (U, V^t for USV^T) or (Q for QAQ^-1) can be arbitrarily scrambled
# this scrambling leads to useless forward stepping... we need to enforce some kind of ordering on the modes
# Currently the ordering is enforced by eigenvalue magnitude,
# but this keeps changing (unstable), and is not guaranteed to be consistent across mode-sets.

# using QR helps with unscrambling, but magnitude is stil -> inf...

# even just modulating the s-matrix causes a magnitude blowup...



# HANKEL POD

class Operator(nn.Module):
    def __init__(self, nlatent, ntime, h, w, device, n_embd=100, nlayers=2, **kwargs):
        super(Operator, self).__init__()
        
        self.nlatent = nlatent
        self.ntime = ntime
        self.nlayers = nlayers
        self.n_embd = n_embd
        self.h = h
        self.w = w
       
        self.m = ntime // 2

        self.device = device
        
        self.A = nn.Parameter(torch.empty((self.m, self.m, self.m, self.m),device=device))
        nn.init.xavier_uniform_(self.A)
        
    def POD(self,x):
        with torch.no_grad():            
            vssv = torch.einsum("Bt..., BT... -> BtT", x, x) # X^T X, shape BtT, results in vs^t s v^t
            # e, v = torch.linalg.eigh(vssv) # shapes are Bt, BtT #+ self.reg[None,:,:]
            _, e, vt = torch.linalg.svd(vssv)
            # print(e[0])
            s = torch.sqrt(e)
            
            a2 = torch.where(s > 1, 0, 1)
            
            d = s / (s**2 + a2)
            sinv = torch.diag_embed(d,offset=0,dim1=-2,dim2=-1)
            ####
            # want s^-1 v^t v s^t s v^t = s v^t
            # svt = torch.einsum("Bm, Bmn -> Bmn", s, vt)
            ####
            # s = torch.sqrt(e) # note should be real since vssv is positive semi-definite and symmetric
            # assert torch.all(torch.isfinite(sinv)) ,"38"
            u = torch.einsum("Bt..., BTt, BTm -> Bm...", x, vt, sinv)
            # assert torch.all(torch.isfinite(u)), "40"
        return u, vt, sinv
    
    def hankel(self, x):
        with torch.no_grad(): 
            # shape Btchw
            xl = x.reshape(x.shape[0], self.ntime, -1) # Btl
            
            x_chunks = [xl[:,i:i+self.m,:].reshape(x.shape[0], -1) for i in range(self.ntime - self.m)] # Br
            
            hankel_ = torch.stack(x_chunks, dim=1) # Bcr
            
            u, _, _= self.POD(hankel_) # unweighted, see the PLOS ONE paper to fix for weather
            
            # u shape is Bmr, so
            
            u_res = u.reshape(x.shape[0], self.m, self.m, -1) # BMr -> BM t' l
            
            uf = u_res.reshape(x.shape[0], self.m, self.m, self.nlatent, self.h, self.w) # BM t' l -> BMT'chw
        
        return uf
        
    def en(self, u, x):
        return torch.einsum("BMtchw, Bchw -> BMt", u, x)
    
    def step(self, x):
        return torch.einsum("mtMT, BMT -> Bmt", self.A, x)
    
    def de(self, u, x):
        return torch.einsum("BMt, BMtchw -> Bchw", x, u)

    def forward(self,x0): # just a single step
        
        u = self.hankel(x0)
        
        z = self.en(u, x0[:,-1])
        
        z2 = self.step(z)
        
        return self.de(u, z2)[:,None,:,:,:]
    
    
    