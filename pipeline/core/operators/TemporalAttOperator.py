__author__ = 'asadam'

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.layers._att import sqrt_act

# TODO make matrices complex? - we just need complex eigenvalues, not really matrices...

# def cmult(ar,ai,br,bi):
#     return ar @ br - ai @ bi, ar @ bi + ai @ br
# def c3mult(ar,ai,br,bi,cr,ci):
#     return cmult(*cmult(ar,ai,br,bi),cr,ci)

# def make_powers(ar,ai,j):
#     res = [ar,]
#     ies = [ai,]
#     cr = ar
#     ci = ai
#     for i in range(j):
#         cr, ci = cmult(cr,ci,ar,ai)
#         res.append(cr)
#         ies.append(ci)
#     return zip(res,ies)

class Operator(nn.Module):
    def __init__(self, nlatent, nspatial, ntime, device, n_embd=100, nlayers=1):
        super(Operator, self).__init__()
        
        self.nlatent = nlatent
        self.nspatial = nspatial
        self.ntime = ntime
        # self.nlayers = nlayers
        self.n_embd = n_embd
  
        self.K = nn.Parameter(torch.empty((ntime,ntime,nlatent),device=device))
        self.Q = nn.Parameter(torch.empty((ntime,ntime,nlatent),device=device))
        self.V = nn.Parameter(torch.empty((ntime,ntime,nlatent),device=device))
  
        nn.init.xavier_uniform_(self.K)
        nn.init.xavier_uniform_(self.Q)
        nn.init.xavier_uniform_(self.V)
  
        self.w = nn.Parameter(torch.ones(ntime,nspatial) / (nspatial*ntime)) # center selection vector
          
    def forward(self,x): # just a single step
        # x has shape BpTcx, and TCx = [T0-20]cx are to be updated to [T21]cx and reduced to [T21]c at center, with operator depending on p
        # this will be repeated multiple times with the ConvExt framework.

        k = torch.einsum("tTc, BpTcx -> Bptcx", self.K, x)
        q = torch.einsum("tTc, BpTcx -> Bptcx", self.Q, x)
        v = torch.einsum("tTc, BpTcx -> Bptcx", self.V, x)
        
        a = sqrt_act(torch.einsum("Bptcx, BpTcx -> BptTx", k, v))
        b = torch.einsum("Bptcx, BptTx -> BpTcx", q, a)
        
        y = torch.einsum("tx, Bptcx -> Bpc", self.w, b) # select a time and position together (finite difference)
        
        return y[:,:,None,:] # Bp_c, select a position, from center selection vector