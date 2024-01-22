__author__ = 'asadam'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchgeometry.core import warp_perspective

class Operator(nn.Module):
    def __init__(self, nlatent, ntime, h, w, device, n_embd=100, nlayers=1, **kwargs):
        super(Operator, self).__init__()
        
        self.nlatent = nlatent
        self.ntime = ntime
        # self.nlayers = nlayers
        self.n_embd = n_embd
        self.h = h
        self.w = w
        # self.reg = torch.eye(ntime, device=device)  # regularization for POD
  
        self.DMD = nn.Parameter((1/(5*ntime)) * torch.eye(5*ntime, device=device)[None,:,:].repeat(ntime, 1, 1)) # ntime, nmodes x nmodes (note nmodes = ntime)
        # self.weight = nn.Parameter(torch.tensor(0.0)) # center selection vector
        self.LR_CNN = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.UD_CNN = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.LR2_CNN = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.UD2_CNN = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.LR_CNN.weight.data = nn.Parameter((1/3) * torch.tensor([[[[1,0,-1],[1,0,-1],[1,0,-1]]]], device=device, dtype=torch.float32))
        self.UD_CNN.weight.data = nn.Parameter((1/3) * torch.tensor([[[[1,1,1],[0,0,0],[-1,-1,-1]]]], device=device, dtype=torch.float32))
        self.LR2_CNN.weight.data = nn.Parameter((1/6) * torch.tensor([[[[1,0,1],[1,0,1],[1,0,1]]]], device=device, dtype=torch.float32))
        self.UD2_CNN.weight.data = nn.Parameter((1/6) * torch.tensor([[[[1,1,1],[0,0,0],[1,1,1]]]], device=device, dtype=torch.float32))
          
        self.filters = [self.LR_CNN, self.UD_CNN, self.LR2_CNN, self.UD2_CNN]   # no need to register since they are already registered as parameters
        self.warps = nn.ParameterList([nn.Parameter(torch.tensor([[[1,0,0],[0,1,0],[0,0,1]]], device=device, dtype=torch.float32)) for _ in range(5)])        
        
    def CNN(self, filter, x):
        xs = x.view(-1, 1, self.h, self.w)
        xs = filter(xs)
        # print(filter.weight.data.requires_grad)
        return xs.view(x.shape)         
        
    def POD(self,x):
        # with torch.no_grad():            
        vssv = torch.einsum("Btchw, BTchw -> BtT", x, x) # X^T X, shape BtT, results in vs^t s v^t
        # e, v = torch.linalg.eigh(vssv) # shapes are Bt, BtT #+ self.reg[None,:,:]
        _, e, vt = torch.linalg.svd(vssv)
        # print(e[0])
        e = torch.where(e > 1, e, 1)
        s = torch.sqrt(e)
        # einv = torch.diag_embed(torch.reciprocal(e),offset=0,dim1=-2,dim2=-1)
        sinv = torch.diag_embed(torch.reciprocal(s),offset=0,dim1=-2,dim2=-1)
        # s = torch.sqrt(e) # note should be real since vssv is positive semi-definite and symmetric
        u = torch.einsum("Btchw, BTt, BTm -> Bmchw", x, vt, sinv)
        return u
    
    def down(self,u,x):
        return torch.einsum("BMchw, BTchw -> BTM", u, x)

    def up(self,z,u):
        return torch.einsum("BM, BMchw -> Bchw", z, u)[:,None,:,:,:] # skip time dimension
    
    def morph(self,x,warp):
        xs = x.view(1, -1, self.h, self.w)
        xs = warp_perspective(xs,warp,dsize=(self.h,self.w))
        # print(filter.weight.data.requires_grad)
        return xs.view(x.shape)  
        
          
    def forward(self,x0): # just a single step
        # x has shape BTcHW, and TC = [T0-20]c are to be updated to [T21]c.
        
        mu = x0[:,-1:,:,:,:]
        # std = (3 * torch.std(x, dim=1, keepdim=True))
        # print(std.shape)
        # to local
        x = x0 - mu
        # x2 = x2 / std
        # this is okay iff no fixed dimensions remain; i.e. we nondimensionalize every part of the equation!
        # but scales are linear and differ per situation
        # so this is really a local projection only; we have lost global information with the batch norm...
        # which is why the equation must be nondimensionalized completely..
        
        # First POD
        u = self.POD(x)
        dus = [self.POD(self.CNN(f, x)) for f in self.filters]
        all_us = [u] + dus
        
        

        # linalg solve for vector in SV-space (note M,m are just mode dimensions of size same as ntime)
        zs = [self.down(u,x) for u in all_us]
        
        # assert torch.isnan(x3).sum() == 0
        
        # DMD
        zcat = torch.cat(zs, dim=2)
        
        z2cat = torch.einsum("TMm, BTM -> Bm", self.DMD, zcat)
        # assert torch.isnan(x4).sum() == 0
        
        # inverse POD
        x1 = mu
        for a in range(5):
            x1 = x1 + self.morph(self.up(z2cat[:,a:a+self.ntime], all_us[a]), self.warps[a])
        # assert torch.isnan(x5).sum() == 0
        
        return x1 # B_chw, or BTchw since we want to keep time dimension (but it's just length 1)