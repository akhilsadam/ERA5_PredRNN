import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter
from core.models.model_base import BaseModel
from core.loss import loss_mixed

class DMDNet(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super().__init__()
        
        self.preprocessor = configs.preprocessor
        
        self.m = self.preprocessor.latent_dims[-1] # number of modes
        self.A = nn.ModuleList([Parameter(torch.eye(self.m).repeat(self.input_length,1,1).to(torch.cfloat)) for _ in range(self.num_layers)])

    def core_forward(self, seq_total, istrain=True, **kwargs):
        total = self.preprocessor.batched_input_transform(seq_total)  # needs dimension shrinking like POD
        inpt = total[:,:self.input_length,:]
        nc, sx, sy = inpt.shape[-3:]
        x = inpt.reshape(inpt.shape[0],inpt.shape[1],-1)
        
        x2 = self.flat_forward(x)
        
        if not istrain:
            outpt = x2        
            outpt = outpt.reshape(outpt.shape[0],outpt.shape[1],nc,sx,sy)
            out = torch.cat((total[:,:self.input_length,:],outpt),dim=1)  
            out = self.preprocessor.batched_output_transform(out)
        else:
            out = inpt
                    
        loss_pred = loss_mixed(out, seq_total, self.input_length) #self.weight) # not weighted, coefficient loss
        loss_decouple = torch.tensor(0.0)

        return loss_pred, loss_decouple, out

    def flat_forward(self,x, i):
        # x is sequence (L,N)
        # u_pre = (self.modes.T@x).to(torch.cfloat) # (m,L) @ (L,N) = (m,N)
        u_pre = x.to(torch.cfloat)
        
        u = u_pre # (m,m) @ (m,N) = (m,N)
        # u = torch.linalg.multi_dot([self.rot.T,self.modes.T,x]) # (m,m) @ (m,L) @ (L,N) = (m,N)

              
        uc = u
        for _ in range(self.predict_length): # prediction loop
            u2 = torch.zeros_like(uc[:,0])
            for j in range(self.input_length):
                u2 += self.A[i][j] @ uc[:,-j] # (m) # A[i] is (N,m,m), so A[i][j] is (m,m)
            uc = torch.cat([uc[:,1:],u2.unsqueeze(1)],dim=1)
        
        # x2 = torch.linalg.multi_dot([self.modes, self.rot, uc]) # (L,N)
        x2_pre = uc.to(torch.double) #(self.rot@uc).to(torch.double) # (m,N)
        # x2 = self.modes@x2_pre # (L,m) @ (m,N) = (L,N)
        x2 = x2_pre # (m,N)
        
        return x2