__author__ = 'asadam'

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.layers._att import sqrt_act

estkqv = lambda i: "tTc, Bp...Tcx -> Bp...tcx" if i==0 else "tT, Bp...Tcx -> Bp...tcx"

esta = "Bptcx, BpTcx, etT -> Bpecx"

# moveaxis B..tcxp
# reshape B..tcxp -> B..tcxhw
# conv B..tcxhw -> B..tcxhw
# reshape B..tcxhw -> B..tcxp
# moveaxis B..tcxp -> B..tcx

# skip # estp = "tx, Bp...tcx -> Bp...tcx"

estb = "Bptcx, Bpecx, etT -> BpTcx"

estc ="tx, Bptcx -> Bpc"
    
class Operator(nn.Module):
    def __init__(self, nlatent, nspatial, ntime, device, h, w, n_embd=100, nlayers=2):
        super(Operator, self).__init__()
        
        self.nlatent = nlatent
        self.nspatial = nspatial
        self.ntime = ntime
        self.nlayers = nlayers
        # self.n_embd = n_embd
        self.h = h
        self.w = w
  
        self.Ks = torch.nn.ParameterList([nn.Parameter(torch.empty((ntime,ntime,nlatent),device=device)),
            *[nn.Parameter(torch.empty((ntime,ntime),device=device)) for _ in range(nlayers-1)]])
        self.Qs = torch.nn.ParameterList([nn.Parameter(torch.empty((ntime,ntime,nlatent),device=device)),
            *[nn.Parameter(torch.empty((ntime,ntime),device=device)) for _ in range(nlayers-1)]])
        self.Vs = torch.nn.ParameterList([nn.Parameter(torch.empty((ntime,ntime,nlatent),device=device)),
            *[nn.Parameter(torch.empty((ntime,ntime),device=device)) for _ in range(nlayers-1)]])
  
        [nn.init.xavier_uniform_(self.Ks[i]) for i in range(nlayers)]
        [nn.init.xavier_uniform_(self.Qs[i]) for i in range(nlayers)]
        [nn.init.xavier_uniform_(self.Vs[i]) for i in range(nlayers)]
        
        self.enc = nn.Parameter(torch.empty((ntime,ntime,ntime),device=device))
        self.dec = nn.Parameter(torch.empty((ntime,ntime,ntime),device=device))
        
        nn.init.orthogonal_(self.enc)
        nn.init.orthogonal_(self.dec)
        
        # self.a = nn.Parameter(torch.ones((ntime,nspatial),device=device)) # scaling vector / propagator
        cchan = ntime * nspatial * nlatent
        self.k = 5
        self.a = nn.Conv2d(cchan,cchan,(self.k,self.k),padding="same",bias=False, groups=nspatial * nlatent) 
        # init a 
        self.a.weight.data.fill_(1/(self.k**2))
        # self.a.weight.data[self.k//2,self.k//2] = 1
        
        # self.res = nn.Parameter(torch.zeros((ntime), device=device))
        self.weight = nn.Parameter(torch.ones((ntime,nspatial),device=device) / (nspatial*ntime)) # center selection vector
          
    def forward(self,x): # just a single step
        # x has shape BpTcx, and TCx = [T0-20]cx are to be updated to [T21]cx and reduced to [T21]c at center, with operator depending on p
        # this will be repeated multiple times with the ConvExt framework.
        
        # square lifting
        qs = []
        
        for i in range(self.nlayers):

            k       = torch.einsum(estkqv(i), self.Ks[i], x)
            qs.append(torch.einsum(estkqv(i), self.Qs[i], x))
            v       = torch.einsum(estkqv(i), self.Vs[i], x)
            
            x = sqrt_act(torch.einsum(esta, k, v, self.enc)) 
            
        # propagate
        # xm = torch.moveaxis(x, 2, -1) # Bptcx -> Btcxp
        # xp = xm
        xm = x.reshape(x.shape[0],self.h, self.w, *x.shape[2:])
        xp = xm.permute(0,3,4,5,1,2).reshape(xm.shape[0], -1 ,self.w, self.h)
        # CONV2D, same padding
        xc = self.a(xp)
        # reshape
        xq = xc.reshape(x.shape[0],*x.shape[2:],self.w, self.h).permute(0,4,5,1,2,3)
        b = xq.reshape(x.shape)        
        # b = torch.einsum(estp, self.a, x) # scale time and position together (finite difference, without sums)
        
        # projection
        for i in range(self.nlayers-1,-1,-1):
            
            b = torch.einsum(estb, qs[i], b, self.dec)
        
        y = torch.einsum(estc, self.weight, b) # select a time and position together (finite difference with sums)
        
        return y[:,:,None,:] # Bp_c, select a position, from center selection vector
    
    
