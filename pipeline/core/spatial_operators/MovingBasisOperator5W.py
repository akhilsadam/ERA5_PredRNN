__author__ = 'asadam'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_harmonics as th




# piDMD (using SFNO SHT), no POD at all

class Operator(nn.Module):
    def __init__(self, nlatent, ntime, h, w, device, n_embd=400, nlayers=1, activation=torch.nn.ReLU(), **kwargs):
        super(Operator, self).__init__()
        
        self.nlatent = nlatent
        self.ntime = ntime
        self.nlayers = nlayers
        self.n_embd = n_embd
        self.h = h
        self.w = w
    
        self.device = device
        
        self.activation = activation        
        
        self.n_modes = 360 # 720 is default for full resolution        
        self.sht = th.RealSHT(h, w, lmax=self.n_modes, mmax=self.n_modes+1, grid="equiangular").to(self.device)
        self.isht = th.InverseRealSHT(h, w, lmax=self.n_modes, mmax=self.n_modes+1, grid="equiangular").to(self.device)

    
        self.A = nn.Parameter(torch.empty((self.nlatent,self.nlatent,self.n_modes,self.n_modes+1),device=device)) # L, M, channels(out), channels(in)
        self.A_i = nn.Parameter(torch.empty((self.nlatent,self.nlatent,self.n_modes,self.n_modes+1),device=device)) # L, M, channels(out), channels(in)
        nn.init.xavier_uniform_(self.A.data,gain=0.1)        
        nn.init.xavier_uniform_(self.A_i.data,gain=0.1)  
        
    # @torch.compile(fullgraph=False)    
    def step(self,x):
        with torch.no_grad():
            x_sht = self.sht(x) # BTchw -> BTclm    
        
        cA = torch.complex(self.A,self.A_i)
        y_sht = torch.einsum("Cclm,BTclm->BTClm",cA,x_sht) + x_sht
        
        y = self.isht(y_sht)
        return y
    
    def forward(self, x):
        xu = torch.mean(x,dim=1,keepdim=True)
        x = x - xu
       
        y = self.step(x)
        
        # add cohesion loss
        if self.training:
            loss = torch.nn.functional.mse_loss(y[:,:-1],x[:,1:])
            loss.backward(retain_graph=True) # only if training!
        
        return y[:,-1:] + xu

    
    
    