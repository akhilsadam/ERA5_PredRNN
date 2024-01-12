__author__ = 'asadam'

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.layers._att import sqrt_act

estkqv = lambda i: "tTc, Bp...Tcx -> Bp...tcx" if i==0 else "tT, Bp...Tcx -> Bp...tcx"

esta = "Bp...tcx, Bp...Tcx, etT -> Bp...ecx"

estp = "tx, Bp...tcx -> Bp...tcx"

estb = "Bp...tcx, Bp...ecx, etT -> Bp...Tcx"

estc ="tx, Bp...tcx -> Bp...c"
    
class Operator(nn.Module):
    def __init__(self, nlatent, nspatial, ntime, device, n_embd=100, nlayers=1):
        super(Operator, self).__init__()
        
        self.nlatent = nlatent
        self.nspatial = nspatial
        self.ntime = ntime
        self.nlayers = nlayers
        # self.n_embd = n_embd
  
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
        
        self.a = nn.Parameter(torch.ones((ntime,nspatial),device=device)) # scaling vector / propagator
        self.w = nn.Parameter(torch.ones((ntime,nspatial),device=device) / (nspatial*ntime)) # center selection vector
          
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
        b = torch.einsum(estp, self.a, x) # scale time and position together (finite difference, without sums)
        
        # projection
        for i in range(self.nlayers-1,-1,-1):
            
            b = torch.einsum(estb, qs[i], b, self.dec)
        
        y = torch.einsum(estc, self.w, b) # select a time and position together (finite difference with sums)
        
        return y[:,:,None,:] # Bp_c, select a position, from center selection vector
    
    
