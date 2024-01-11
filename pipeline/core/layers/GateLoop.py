__author__ = 'asadam'

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.layers._att import sqrt_att_a, att_b

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

class GateLoopOperator(nn.Module):
    def __init__(self, nlatent, nspatial, device, n_embd=100, nlayers=1):
        super(GateLoopOperator, self).__init__()
        
        self.nlatent = nlatent
        self.nspatial = nspatial
        # self.nlayers = nlayers
        self.n_embd = n_embd
        
        # spatial att. w modal att in one set
        self.K = torch.Parameter(nn.eye(nspatial)[:,None,:,None].repeat(1,nlatent,1,nlatent).to(device))
        self.Q = torch.Parameter(nn.eye(nspatial)[:,None,:,None].repeat(1,nlatent,1,nlatent).to(device))
        self.V = torch.Parameter(nn.eye(nspatial)[:,None,:,None].repeat(1,nlatent,1,nlatent).to(device))
        
        # # modal / channel / latent att.
        # self.Km = torch.Parameter(nn.eye(nlatent).to(device))
        # self.Qm = torch.Parameter(nn.eye(nlatent).to(device))
        # self.Vm = torch.Parameter(nn.eye(nlatent).to(device))
        
        self.d = (nlatent,nlatent,nspatial,nspatial)
        
        # state matrix
        # assert self.n_embd >= self.nspatial * self.n_latent, "Need more variables!" # Ignore since such a large A is impossible
        self.Aoo = torch.Parameter(nn.ones(*self.d))
        # self.Aooi = torch.Parameter(nn.zeros(*self.d))
        self.Ann = torch.Parameter(nn.eye(self.n_embd))
        # self.Anni = torch.Parameter(nn.eye(self.n_embd))
        
        # projection matrix
        # does double duty as a linear compression since A is still too large...
        self.Aon = torch.Parameter(nn.zeros(*self.d,self.n_embd))
        self.Ano = torch.Parameter(nn.zeros(*self.d,self.n_embd))
        # self.Aoni = torch.Parameter(nn.zeros(*self.d,self.n_embd))
        # self.Anoi = torch.Parameter(nn.zeros(*self.d,self.n_embd))   
          
    def step(self,x):
        # x has shape BpTcx, and TCx = [T0-20]cx are to be updated to [T21]cx and reduced to [T21]c at center, with operator depending on p
        # this will be repeated multiple times with the ConvExt framework.
        
        # kernel in ####      
        k = torch.einsum("XCxc, BpTcx -> BpTCX", self.K, x)
        v = torch.einsum("XCxc, BpTcx -> BpTCX", self.V, x)
        # outer product lift 
        kvt = torch.einsum("BpTcx, BpTCX -> BpTcCxX", k, v)
        # TODO test with square root activation on KVT
        
        # DMD-memory ####
        h = torch.einsum("cCxX, BpTcCxX -> BpTcCxX", self.Aoo, kvt[:,:,-1:]) \
            + torch.einsum("ckxyn, CKXYn, BpTCKXY -> BpTckxy", self.Aon, self.Ano, kvt[:,:,-2:-1]) \
            
        Ann = self.Ann
        for j in range(2,x.shape[2]):    
            h = h + torch.einsum("ckxyn, nm, CKXYm, BpTCKXY -> BpTckxy", self.Aon, Ann, self.Ano, kvt[:,:,-j-1:-j])
            Ann = Ann @ Ann
        del Ann
        
        
        # select vector  ####
        q = torch.einsum("XCxc, BpTcx -> BpTCX", self.Q, x)
        # project
        y = torch.einsum("BpTCX, BpTCcXx -> BpTcx", q, h)
        
        return y