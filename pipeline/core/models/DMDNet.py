import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter
from core.models.model_base import BaseModel
from core.loss import loss_mixed

class DMDNet(BaseModel):
    def __init__(self, num_layers, num_hidden, configs):
        super().__init__(num_layers, num_hidden, configs)
        
        self.preprocessor = configs.preprocessor
        self.preprocessor.load(device=configs.device)
        self.device = configs.device
        self.input_length = configs.input_length
        self.predict_length = configs.total_length - configs.input_length
        self.total_length = configs.total_length
        
        # self.reweight = torch.zeros_like(self.weight).T.to(self.device)
        
        self.m = self.preprocessor.latent_dims[-1] # number of modes
        self.A = nn.ParameterList([Parameter(torch.eye(self.m).repeat(self.input_length,1,1).to(torch.cfloat)) for _ in range(self.num_layers)])

    def core_forward(self, seq_total, istrain=True, **kwargs):
        total = self.preprocessor.batched_input_transform(seq_total)  # needs dimension shrinking like POD
        nc, sx, sy = total.shape[-3:]
        total_flat = total.reshape(total.shape[0],total.shape[1],-1)
        inpt = total_flat[:,:self.input_length,:]
        
        x_in = inpt
        x2 = inpt * 0.0
        loss_decouple = torch.tensor(0.0)
        for i in range(self.num_layers):
            step, decouple_step = self.flat_forward(x_in,i) # single step forward
            loss_decouple = loss_decouple + decouple_step
            x2 = x2 + step
            # TODO want to change x_in to be x_in - self.flat_backward(x2,i) (predictor and corrector) for next layer
        
        if not istrain:
            outpt = x2        
            outpt = outpt.reshape(outpt.shape[0],outpt.shape[1],nc,sx,sy)
            out = torch.cat((total[:,:self.input_length,:],outpt),dim=1)  
            out = self.preprocessor.batched_output_transform(out)
        else:
            out = total_flat
        
        # print(x2.size(),total.size())
        
        loss_pred = loss_mixed(x2, total_flat[:,self.input_length:,], 0, weight=1.0) # not weighted, coefficient loss

        return loss_pred, loss_decouple, out

    def flat_forward(self,x, i):
        # x is sequence (L,N)
        # u_pre = (self.modes.T@x).to(torch.cfloat) # (m,L) @ (L,N) = (m,N)
        u_pre = x.to(torch.cfloat)
        
        u = u_pre # (m,m) @ (m,N) = (m,N)
        # u = torch.linalg.multi_dot([self.rot.T,self.modes.T,x]) # (m,m) @ (m,L) @ (L,N) = (m,N)

        # nbatch = u.shape[0]
        # latent = u.shape[-1]      
        decouple = torch.tensor(0.0)
        uc = u
        for _ in range(self.predict_length): # prediction loop
            u2 = torch.zeros_like(uc[:,0])
            for j in range(self.input_length):
                adj =  uc[:,-j] @ self.A[i][j] # (m) # A[i] is (N,m,m), so A[i][j] is (m,m)
                
                # decouple = decouple - torch.sum((torch.bmm(u2.real.view(nbatch,1,latent),adj.real.view(nbatch,latent,1)))**2) # decouple loss, note batched vector dot
                # this decouple loss is flat-out wrong
                
                u2 = u2 + adj
            uc = torch.cat([uc[:,1:],u2.unsqueeze(1)],dim=1)
        
        # x2 = torch.linalg.multi_dot([self.modes, self.rot, uc]) # (L,N)
        x2_pre = uc.to(torch.float) #(self.rot@uc).to(torch.double) # (m,N)
        # x2 = self.modes@x2_pre # (L,m) @ (m,N) = (L,N)
        x2 = x2_pre # (m,N)
        
        return x2, decouple       
    
    def edit_config(self,configs):
        if configs.patch_size != 1:
            print(f"Warning: patch_size is {configs.patch_size} but should be 1 for TF model. Setting to 1.")
            configs.patch_size = 1
        return configs