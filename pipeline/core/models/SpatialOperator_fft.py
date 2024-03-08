import torch
import math
import torch.nn as nn
# from torch.nn.parameter import Parameter
from core.models.model_base import BaseModel
# from core.layers.ConvExt import ConvX
from core.spatial_operators.MovingBasisOperator4W import Operator
# from core.spatial_operators.MovingBasisOperator6 import Operator # CNNs are high memory cost
# from core.layers.Siren import MLP
from core.loss import loss_mixed

import torch_harmonics as th

class SpatialOperator(BaseModel):
    def __init__(self, num_layers, num_hidden, configs):
        super().__init__(num_layers, num_hidden, configs)
        
        self.preprocessor = configs.preprocessor
        self.preprocessor.load(device=configs.device)
        self.device = configs.device
        self.input_length = configs.input_length
        self.predict_length = configs.total_length - configs.input_length
        self.total_length = configs.total_length
        
        self.in_channel = self.preprocessor.latent_dims[-1]
        self.height = self.preprocessor.patch_x
        self.width = self.preprocessor.patch_y
        
        # self.area_weight = configs.area_weight
        
        acts = {'relu':nn.ReLU(),'tanh':nn.Tanh(),'sigmoid':nn.Sigmoid(),'sin': lambda x: torch.sin(x)}
        
        self.activation = acts[configs.model_args['activation']] if 'activation' in configs.model_args else nn.ReLU()
        
        h = self.preprocessor.patch_x
        w = self.preprocessor.patch_y
        
        
        self.sht = th.RealSHT(h, w, grid="equiangular").to(self.device)
        self.n_modes = 720 # seems to be default?
        self.isht = th.InverseRealSHT(h, w, lmax=self.n_modes, mmax=self.n_modes+1, grid="equiangular").to(self.device)
        
        self.operator = Operator(self.in_channel, self.input_length, self.n_modes, self.n_modes+1, device=self.device, nlayers=1, activation = self.activation)

        # torch.backends.cuda.preferred_linalg_library('magma')

    def core_forward(self, seq_total, istrain=True, **kwargs):
        total_pre = self.preprocessor.batched_input_transform(seq_total)  # a scale preprocessor is expected
        with torch.no_grad():
            total = self.sht(total_pre)
        inpt = total[:,:self.input_length,:,:,:]
        
        if istrain:
            predict_length = 2
            lshift = 2 + predict_length
            rshift = self.input_length - 2
        else:
            predict_length = self.predict_length
            lshift = self.predict_length
            rshift = self.input_length
        
        x_in = inpt
        x2 = self.flat_forward(x_in, predict_length) # single step forward
        
        if not istrain:
            outpt = x2        
            out = torch.cat((total[:,:self.input_length,:,:,:],outpt),dim=1)
            with torch.no_grad():
                out = self.isht(out)
            out = self.preprocessor.batched_output_transform(out)
            loss_pred = torch.tensor(0.0)
        else:
            out = total
            
        q = torch.norm(loss_mixed(x2[:,-lshift:,], total[:,rshift:self.input_length+predict_length,], 0, weight=1.0, a=0.2, b=0.01)) # not weighted, coefficient loss w derivatives
        
        loss_pred = (self.predict_length / predict_length) * q
        decouple_loss = torch.tensor(0.0)
        # decouple_loss = r
        
        return loss_pred, decouple_loss, out

    def flat_forward(self,x, predict_length):        
        uc = x # BTCHW
        for _ in range(predict_length): # prediction loop
            u2 = self.operator(uc)
            # print(u2.shape) # TODO turn on for debugging
            uc = torch.cat([uc[:,1:],u2],dim=1)
        x2 = uc
    
        return x2      
               
    # def unfold_operator(self,x):
    #     # x has shape BTCHW, want BpTCx, where p = HxW and x = _ (empty)
    #     y = x.permute((0,3,4,1,2)).reshape((x.shape[0],self.height*self.width,self.input_length,self.in_channel,1))
    #     z = self.operator(y) # BpTc
    #     return z.permute((0,2,3,1)).reshape((x.shape[0],1,self.in_channel,self.height,self.width))# + x[:,-1:,:,:,:] # BTCHW
    
    
    def edit_config(self,configs):
        if configs.patch_size != 1:
            print(f"Warning: patch_size is {configs.patch_size} but should be 1 for TF model. Setting to 1.")
            configs.patch_size = 1
        return configs