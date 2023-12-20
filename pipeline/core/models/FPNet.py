import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter
from core.models.model_base import BaseModel
from core.layers.Siren import MLP
from core.loss import loss_mixed

class FPNet(BaseModel):
    def __init__(self, num_layers, num_hidden, configs):
        super().__init__(num_layers, num_hidden, configs)
        
        self.preprocessor = configs.preprocessor
        self.preprocessor.load(device=configs.device)
        self.device = configs.device
        self.input_length = configs.input_length
        self.predict_length = configs.total_length - configs.input_length
        self.total_length = configs.total_length
        
        # self.reweight = torch.zeros_like(self.weight).T.to(self.device)
        
        self.m = self.preprocessor.latent_dims[-1]*self.preprocessor.patch_x*self.preprocessor.patch_y # number of modes
        sz = self.m * (self.input_length + 1)
        osz = self.m
        self.net = MLP(sz,sz,osz,configs.model_args['n_layers'],"sin",omega_0=(self.m/math.pi))


    def core_forward(self, seq_total, istrain=True, **kwargs):
        total = self.preprocessor.batched_input_transform(seq_total)  # needs dimension shrinking like POD
        nc, sx, sy = total.shape[-3:]
        total_flat = total.reshape(total.shape[0],total.shape[1],-1)
        inpt = total_flat[:,:self.input_length,:]
        
        x_in = inpt
        x2, decouple_loss = self.flat_forward(x_in) # single step forward
        
        if not istrain:
            outpt = x2        
            outpt = outpt.reshape(outpt.shape[0],outpt.shape[1],nc,sx,sy)
            out = torch.cat((total[:,:self.input_length,:],outpt),dim=1)  
            out = self.preprocessor.batched_output_transform(out)
        else:
            out = total_flat
        
        loss_pred = loss_mixed(x2[:,-self.predict_length:,], total_flat[:,self.input_length:,], 0, weight=1.0, a=0.1, b=0.01) # not weighted, coefficient loss

        return loss_pred, decouple_loss*1e-3, out


    def R(self,x,t):
        # t  :  B, N, m
        # x  :  B, 1, m - the solution at next step
        
        # # normalize against the last known value
        # n = t[:,-1:,:]
        
        # t = t / n
        # x = x / n
        
        # network
        flat = torch.flatten(torch.cat([x,t],dim=1),start_dim=1)
        flat = self.net(flat)
        x2 = flat.reshape(x.shape)
        
        # skip rescale
        
        return x2
        
        
        

    def fixed_point(self, t, it=6):
        f = lambda x,t : x - self.R(x, t)
        
        # initial condition
        x = t[:,-1:,:]

        # generate noise
        ep = torch.rand_like(x) * x *  1e-3
        
        # initial condition offset
        x = x + ep

        # solver
        for _ in range(it):
            x = f(x,t)
            
        # regularization
        xp = x + ep
        xn = x - ep
        
        bwd = self.R(xp,t) - x # > ep
        fwd = x - self.R(xn,t) # > 0, < ep
        
        decouple = torch.tensor(0.0) #torch.mean(torch.nn.functional.relu(ep - bwd)**2) \
            #+ torch.mean(torch.nn.functional.relu(-fwd)**2) \
            #+ torch.mean(torch.nn.functional.relu(fwd-ep)**2)

        return x, decouple

    def flat_forward(self,x):
        decouple = 0.0
        
        uc = x #(B,N,m)
        for _ in range(self.predict_length): # prediction loop
            u2, dc = self.fixed_point(uc)
            decouple += dc
            uc = torch.cat([uc[:,1:],u2],dim=1)
        x2 = uc
    
        return x2, decouple       
               
    
    def edit_config(self,configs):
        if configs.patch_size != 1:
            print(f"Warning: patch_size is {configs.patch_size} but should be 1 for TF model. Setting to 1.")
            configs.patch_size = 1
        return configs