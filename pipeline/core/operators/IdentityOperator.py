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
  
        self.w = nn.Parameter(torch.ones(nspatial) / nspatial) # center selection vector
          
    def forward(self,x): # just a single step
        # x has shape BpTcx, and TCx = [T0-20]cx are to be updated to [T21]cx and reduced to [T21]c at center, with operator depending on p
        # this will be repeated multiple times with the ConvExt framework.
        
        y = torch.einsum("x, Bpcx -> Bpc", self.w, x[:,:,-1,:,:])[:,:,None,:] # BpTc
        
        return y