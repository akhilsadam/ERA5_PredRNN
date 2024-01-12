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
        self.K = nn.Parameter(torch.empty((nspatial,nspatial,nlatent),device=device))
        self.Q = nn.Parameter(torch.empty((nspatial,nspatial,nlatent),device=device))
        self.V = nn.Parameter(torch.empty((nspatial,nspatial,nlatent),device=device))
        
        nn.init.xavier_uniform_(self.K)
        nn.init.xavier_uniform_(self.Q)
        nn.init.xavier_uniform_(self.V)
        
        # # modal / channel / latent att.
        # self.Km = nn.Parameter(torch.eye(nlatent).to(device))
        # self.Qm = nn.Parameter(torch.eye(nlatent).to(device))
        # self.Vm = nn.Parameter(torch.eye(nlatent).to(device))
        
        self.d = (nspatial,nspatial)
        
        # state matrix
        # assert self.n_embd >= self.nspatial * self.n_latent, "Need more variables!" # Ignore since such a large A is impossible
        self.Aoo = nn.Parameter(torch.ones(*self.d))
        # self.Aooi = nn.Parameter(torch.zeros(*self.d))
        self.Ann = nn.Parameter(torch.zeros(self.n_embd))
        # self.Anni = nn.Parameter(torch.eye(self.n_embd))
        
        # projection matrix
        # does double duty as a linear compression since A is still too large...
        self.Aon = nn.Parameter(torch.empty(*self.d,self.n_embd))
        self.Ano = nn.Parameter(torch.empty(*self.d,self.n_embd))
        
        nn.init.xavier_uniform_(self.Aon)
        nn.init.xavier_uniform_(self.Ano)
        
        # self.Aoni = nn.Parameter(torch.zeros(*self.d,self.n_embd))
        # self.Anoi = nn.Parameter(torch.zeros(*self.d,self.n_embd))   
        
        self.w = nn.Parameter(torch.ones(nspatial) / nspatial) # center selection vector
          
    def forward(self,x): # just a single step
        # x has shape BpTcx, and TCx = [T0-20]cx are to be updated to [T21]cx and reduced to [T21]c at center, with operator depending on p
        # this will be repeated multiple times with the ConvExt framework.
        
        # # kernel in ####      
        # k = torch.einsum("Xxc, BpTcx -> BpTcX", self.K, x)
        # v = torch.einsum("Xxc, BpTcx -> BpTcX", self.V, x)
        # # outer product lift 
        # kvt = torch.einsum("BpTcx, BpTcX -> BpTxX", k, v)
        # # TODO test with square root activation on KVT
        
        # optimized einsum for above
        kvt = torch.einsum("Yxc, Xxc, BpTcx -> BpTXY", self.Q, self.K, x) # sqrt_act(
        
        #nan check
        # assert torch.isnan(kvt).sum() == 0, "NaN in kvt"
        
        # DMD-memory ####
        Aon = torch.clamp(self.Aon, min=-1.0, max=1.0)
        Ano = torch.clamp(self.Ano, min=-1.0, max=1.0)        
        
        h = torch.einsum("xX, BpTxX -> BpTxX", self.Aoo, kvt[:,:,-1:]) \
            + torch.einsum("xyn, XYn, BpTXY -> BpTxy", Aon, Ano, kvt[:,:,-2:-1]) \

        Ann = self.Ann
        for j in range(2,x.shape[2]):    
            h = h + torch.einsum("xyn, n, XYn, BpTXY -> BpTxy", Aon, Ann, Ano, kvt[:,:,-j-1:-j])
            Ann = torch.clamp(Ann * self.Ann, min=-1.0, max=1.0)
        
                
        #nan check
        # assert torch.isnan(h).sum() == 0, "NaN in h"
        
        # select vector  ####
        q = torch.einsum("Xxc, Bpcx -> BpcX", self.V, x[:,:,-1,:,:]) # last vector query
        # project
        y = torch.einsum("x, BpcX, BpTXx -> BpTc", self.w, q, h)
        # print(h.shape, y.shape)
        
        return y