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
        # self.nlayers = nlayers
        self.n_embd = n_embd
        
        # spatial att. w modal att in one set
        self.K = nn.Parameter(0.01*torch.rand((nspatial,nspatial))[:,None,:,None].repeat(1,nlatent,1,nlatent).to(device))
        self.Q = nn.Parameter(0.01*torch.rand((nspatial,nspatial))[:,None,:,None].repeat(1,nlatent,1,nlatent).to(device))
        self.V = nn.Parameter(0.01*torch.rand((nspatial,nspatial))[:,None,:,None].repeat(1,nlatent,1,nlatent).to(device))
        
        # # modal / channel / latent att.
        # self.Km = nn.Parameter(torch.eye(nlatent).to(device))
        # self.Qm = nn.Parameter(torch.eye(nlatent).to(device))
        # self.Vm = nn.Parameter(torch.eye(nlatent).to(device))
        
        self.d = (nlatent,nlatent,nspatial,nspatial)
        
        # state matrix
        # assert self.n_embd >= self.nspatial * self.n_latent, "Need more variables!" # Ignore since such a large A is impossible
        self.Aoo = nn.Parameter(torch.ones(*self.d))
        # self.Aooi = nn.Parameter(torch.zeros(*self.d))
        # self.Ann = nn.Parameter(torch.zeros(self.n_embd)) # TODO ADD THIS BACK IN, turned off for NaN loss
        # self.Anni = nn.Parameter(torch.eye(self.n_embd))
        
        # projection matrix
        # does double duty as a linear compression since A is still too large...
        self.Aon = nn.Parameter(torch.zeros(*self.d,self.n_embd))
        self.Ano = nn.Parameter(torch.zeros(*self.d,self.n_embd))
        # self.Aoni = nn.Parameter(torch.zeros(*self.d,self.n_embd))
        # self.Anoi = nn.Parameter(torch.zeros(*self.d,self.n_embd))   
          
    def forward(self,x): # just a single step
        # x has shape BpTcx, and TCx = [T0-20]cx are to be updated to [T21]cx and reduced to [T21]c at center, with operator depending on p
        # this will be repeated multiple times with the ConvExt framework.
        
        # # kernel in ####      
        # k = torch.einsum("XCxc, BpTcx -> BpTCX", self.K, x)
        # v = torch.einsum("XCxc, BpTcx -> BpTCX", self.V, x)
        # # outer product lift 
        # kvt = torch.einsum("BpTcx, BpTCX -> BpTcCxX", k, v)
        # # TODO test with square root activation on KVT
        
        # optimized einsum for above
        kvt = sqrt_act(torch.einsum("YKxc, XCxc, BpTcx -> BpTCKXY", self.V, self.K, x))
        
        #nan check
        # assert torch.isnan(kvt).sum() == 0, "NaN in kvt"
        
        # DMD-memory ####
        h = torch.einsum("cCxX, BpTcCxX -> BpTcCxX", self.Aoo, kvt[:,:,-1:]) \
            + torch.einsum("ckxyn, CKXYn, BpTCKXY -> BpTckxy", self.Aon, self.Ano, kvt[:,:,-2:-1]) \

        # TODO add this, removing for memory savings
        # Ann = self.Ann
        # for j in range(2,x.shape[2]):    
        #     h = h + torch.einsum("ckxyn, n, CKXYn, BpTCKXY -> BpTckxy", self.Aon, Ann, self.Ano, kvt[:,:,-j-1:-j])
        #     Ann = Ann * self.Ann
        
                
        #nan check
        # assert torch.isnan(h).sum() == 0, "NaN in h"
        
        # select vector  ####
        q = torch.einsum("XCxc, Bpcx -> BpCX", self.Q, x[:,:,-1,:,:]) # last vector query
        # project
        y = torch.einsum("BpCX, BpTCcXx -> BpTc", q, h)
        # print(h.shape, y.shape)
        
        return y