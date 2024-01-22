__author__ = 'asadam'

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.layers._att import sqrt_act

class Operator(nn.Module):
    def __init__(self, nlatent, nspatial, ntime, device, n_embd=100, nlayers=1, **kwargs):
        super(Operator, self).__init__()
        
        self.nlatent = nlatent
        self.nspatial = nspatial
        self.ntime = ntime
        # self.nlayers = nlayers
        self.n_embd = n_embd
  
        # self.reg = torch.eye(ntime, device=device)  # regularization for POD
  
        self.K_DMD = nn.Parameter(torch.empty((ntime,ntime,ntime),device=device)) # ntime, nmodes x nmodes (note nmodes = ntime)
        self.V_DMD = nn.Parameter(torch.empty((ntime,ntime,ntime),device=device)) # ntime, nmodes x nmodes (note nmodes = ntime)
        self.Q_DMD = nn.Parameter(torch.empty((ntime,ntime,ntime),device=device)) # ntime, nmodes x nmodes (note nmodes = ntime)
        self.DMD = nn.Parameter((1/ntime) * torch.eye(ntime, device=device)[None,:,:].repeat(ntime, 1, 1)) # ntime, nmodes x nmodes (note nmodes = ntime)
        
        nn.init.xavier_uniform_(self.K_DMD,gain=0.01)
        nn.init.xavier_uniform_(self.V_DMD,gain=0.01)
        nn.init.xavier_uniform_(self.Q_DMD,gain=0.01)
        # nn.init.xavier_uniform_(self.Q2_DMD)
        
# self.w = nn.Parameter(torch.tensor(0.0)) # center selection vector
          
    def forward(self,x): # just a single step
        # x has shape BpTcx, and TCx = [T0-20]cx are to be updated to [T21]cx and reduced to [T21]c at center, with operator depending on p
        # this will be repeated multiple times with the ConvExt framework.
        
        mu = x[:,:,-1:,:,:]
        # std = (3 * torch.std(x, dim=2, keepdim=True))
        # print(std.shape)
        # to local
        x2 = x - mu
        # x2 = x2 / std
        # this is okay iff no fixed dimensions remain; i.e. we nondimensionalize every part of the equation!
        # but scales are linear and differ per situation
        # so this is really a local projection only; we have lost global information with the batch norm...
        # which is why the equation must be nondimensionalized completely..
        
        # First POD
        with torch.no_grad():
                
            vssv = torch.einsum("Bptcx, BpTcx -> BtT", x2, x2) # X^T X, shape BtT, results in vs^t s v^t
            # e, v = torch.linalg.eigh(vssv) # shapes are Bt, BtT #+ self.reg[None,:,:]
            
            _, e, vt = torch.linalg.svd(vssv)
            
            # print(e[0])
                    
            e = torch.where(e > 1, e, 1)
            s = torch.sqrt(e)
            # einv = torch.diag_embed(torch.reciprocal(e),offset=0,dim1=-2,dim2=-1)
            sinv = torch.diag_embed(torch.reciprocal(s),offset=0,dim1=-2,dim2=-1)
            # s = torch.sqrt(e) # note should be real since vssv is positive semi-definite and symmetric
            
            u = torch.einsum("Bptcx, BTt, BTm -> Bpmcx", x2, vt, sinv)

        # linalg solve for vector in SV-space (note M,m are just mode dimensions of size same as ntime)
        x3 = torch.einsum("BpMcx, BpTcx -> BMT", u, x2)
        
        # assert torch.isnan(x3).sum() == 0
        
        # DMD
        k = torch.einsum("TMm, BTM -> Bm", self.K_DMD, x3)
        v = torch.einsum("TMm, BTM -> Bm", self.V_DMD, x3)
        
        ktv = torch.einsum("Bm, BM -> BmM", k, v)
        
        q = torch.einsum("TMm, BTM -> Bm", self.Q_DMD, x3)
        
        x4 = torch.einsum("Bm, BmM -> BM", q, ktv) \
            + torch.einsum("TMm, BTM -> Bm", self.DMD, x3)
        
        # assert torch.isnan(x4).sum() == 0
        
        # inverse POD
        x5 = torch.einsum("Bm, Bpmcx -> Bpcx", x4, u)[:,:,None,:,:]
        
        # assert torch.isnan(x5).sum() == 0
        
        x6 = x5 + mu # x5 * std + mu
        
        return x6[:,:,:,:,0] # assuming x dim is length 1 for now...so no position selection
    # Bp_c, or BpTc since we want to keep time dimension (but it's just length 1)