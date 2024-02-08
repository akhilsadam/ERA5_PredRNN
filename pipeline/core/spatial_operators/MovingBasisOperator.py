__author__ = 'asadam'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
from natten import NeighborhoodAttention2D
from core.layers.shift import Shift

class Operator(nn.Module):
    def __init__(self, nlatent, ntime, h, w, device, n_embd=100, nlayers=5, **kwargs):
        super(Operator, self).__init__()
        
        self.nlatent = nlatent
        self.ntime = ntime
        self.nlayers = nlayers
        self.n_embd = n_embd
        self.h = h
        self.w = w
        # self.reg = torch.eye(ntime, device=device)  # regularization for POD
  
        self.mixers = nn.ModuleList([nn.Linear(ntime**2, ntime**2) for _ in range(nlayers)])
        self.select = nn.Linear(ntime**2, ntime)
       
        self.ncnn = 2
        self.k=3
        self.widen_range = 3
        self.CNNs = nn.ModuleList([DeformConv2d(self.nlatent, self.nlatent, kernel_size=self.k, stride=1, padding=1, bias=True) for _ in range(self.ncnn*nlayers)])
        self.offsets = nn.ParameterList([nn.Parameter(torch.empty(2*self.k**2,ntime,nlatent)) for _ in range(self.ncnn*nlayers)])
        
        for filter in self.CNNs:
            nn.init.xavier_uniform_(filter.weight.data,0.001)
            nn.init.constant_(filter.bias.data,0.0)
        for offset in self.offsets:
            nn.init.constant_(offset.data,0.0)
        # self.ATTs = nn.ModuleList([NeighborhoodAttention2D(dim=nlatent, kernel_size=self.k, dilation=1, num_heads=1) for _ in range(self.ncnn*nlayers)])
        # self.shifts = nn.ModuleList([Shift(ntime) for _ in range(self.ncnn*nlayers)])
        # self.ws = nn.ParameterList([nn.Parameter(torch.tensor([1.0],device=device)) for _ in range(nlayers)])
        # self.w2s = nn.ParameterList([nn.Parameter(torch.tensor([1.0],device=device)) for _ in range(nlayers)])
        
    def Mix(self, x, i):
        xs = x.reshape(-1, self.ntime**2)
        return self.mixers[i](xs).reshape(x.shape)
    
    def Select(self, x):
        xs = x.reshape(-1, self.ntime**2)
        return self.select(xs).reshape(x.shape[:-1])
    
    def offset(self,x,i):
        return torch.einsum("Btchw, qtc -> Btqhw", x, self.offsets[i]).reshape(-1,2*self.k**2,self.h,self.w)
        
    def Left(self, x, u, i):
        xs = x.view(-1, self.nlatent, self.h, self.w)
        for filter in self.CNNs[i:i+self.ncnn]:
            act = xs
            
            # for _ in range(self.widen_range):
            act = filter(act, offset=self.offset(x,i))
            
            # pool = torch.nn.functional.avg_pool2d(act, kernel_size=3, stride=1, padding=1) # downsample gradient to avoid edge artifacts
            #TODO add upsample, preferably using POD or such
            xs = act + xs
        # print(filter.weight.data.requires_grad)
        return xs.view(x.shape)
    
    # def Left(self, x, u, i):
    #     xs = x.view(-1, self.nlatent, self.h, self.w)
    #     # for filter in self.CNNs[i:i+self.ncnn]:

    #     xq = self.CNNs[i](xs, offset=self.offset(x,i))

    #     key = self.CNNs[i+1](act, offset=self.offset(x,i+1))
    #     value = self.CNNs[i+2](act, offset=self.offset(x,i+2))
    #     query = self.CNNs[i+3](act, offset=self.offset(x,i+3))
        
    #     att = torch.einsum("Btchw, BTChw -> BtcTC", query, key)
    #     act = self.w2s[i] * torch.cos(self.ws[i] * att) 
    #     xa = torch.einsum("BtcTC, Btchw -> BTChw", act, value)

    #     xs = xa + xq
    #     # print(filter.weight.data.requires_grad)
    #     return xs.view(x.shape)
    
    
    # def Left(self, x, u, i):
    #     xs = x.permute(0,1,3,4,2).reshape(-1, self.h, self.w, self.nlatent)
    #     us = u.reshape(-1, self.ntime) # the modes
        
    #     # for i in range(j*self.ncnn,(j+1)*self.ncnn):
    #     for j in range(self.ncnn):
    #         q = i*self.ncnn+j
    #         xs = self.ATTs[q](self.shifts[q](xs, us))
    #     # q = i*self.ncnn
    #     # xs = self.shifts[q+1](self.ATTs[i](self.shifts[q](xs, us)), us)
    #     # xs = act + self.resids[i]*xs
    #     # print(filter.weight.data.requires_grad)
    #     return xs.reshape(-1,self.ntime,self.h,self.w,self.nlatent).permute(0,1,4,2,3)
        
    def POD(self,x):
        with torch.no_grad():            
            vssv = torch.einsum("Btchw, BTchw -> BtT", x, x) # X^T X, shape BtT, results in vs^t s v^t
            # e, v = torch.linalg.eigh(vssv) # shapes are Bt, BtT #+ self.reg[None,:,:]
            _, e, vt = torch.linalg.svd(vssv)
            # print(e[0])
            e = torch.where(e > 1, e, 1)
            s = torch.sqrt(e)
            # einv = torch.diag_embed(torch.reciprocal(e),offset=0,dim1=-2,dim2=-1)
            sinv = torch.diag_embed(torch.reciprocal(s),offset=0,dim1=-2,dim2=-1)
            ####
            # want s^-1 v^t v s^t s v^t = s v^t
            # svt = torch.einsum("Bm, Bmn -> Bmn", s, vt)
            ####
            # s = torch.sqrt(e) # note should be real since vssv is positive semi-definite and symmetric
            u = torch.einsum("Btchw, BTt, BTm -> Bmchw", x, vt, sinv)
        
        return u
    
    def down(self,u,x):
        return torch.einsum("BMchw, BTchw -> BTM", u, x)
    
    def up(self,z,u):
        return torch.einsum("BTM, BMchw -> BTchw", z, u) 

    def upf(self,z,u):
        return torch.einsum("BM, BMchw -> Bchw", z, u) # skip time dimension  
    
    def layer(self,x,i):
        u = self.POD(x)
        
        svt = self.down(u,x)
        a = self.Mix(svt,i)
        u2 = self.up(a,u)
        
        return self.Left(u2, a ,i)   
    
    def outlayer(self,x):
        u = self.POD(x)
        svt = self.down(u,x)
        a = self.Select(svt)
        return self.upf(a,u)[:,None,:,:,:]
          
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
        
        for i in range(self.nlayers):
            x = self.layer(x,i)

        x = self.outlayer(x)
        
        return x + mu
        
        # return x1 # B_chw, or BTchw since we want to keep time dimension (but it's just length 1)