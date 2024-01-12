import torch
import math
import torch.nn as nn
# from torch.nn.parameter import Parameter
from core.models.model_base import BaseModel
from core.layers.ConvExt import ConvX
from core.operators.StackTemporalAttOperator import Operator
# from core.layers.Siren import MLP
from core.loss import loss_mixed

class GateLoop(BaseModel):
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
        
        # assuming a 100mph wind/data speed, over 6 hours this is 600mi, or 20px on either side = 40px
        # if we use a 6px filter, or only 3px per side, only 15mph winds can be captured...which may not be sufficient!
        fsize = 3      
        
        self.conv = ConvX(in_time=self.input_length, in_channel=self.in_channel, height=self.height, width=self.width,
                          filter_size=fsize, slices=1, operator_class=Operator, device=self.device)


    def core_forward(self, seq_total, istrain=True, **kwargs):
        total = self.preprocessor.batched_input_transform(seq_total)  # a scale preprocessor is expected
        inpt = total[:,:self.input_length,:,:,:]
        
        x_in = inpt
        x2 = self.flat_forward(x_in) # single step forward
        
        if not istrain:
            outpt = x2        
            out = torch.cat((total[:,:self.input_length,:,:,:],outpt),dim=1)  
            out = self.preprocessor.batched_output_transform(out)
        else:
            out = total
        
        loss_pred = loss_mixed(x2[:,-self.predict_length:,], total[:,self.input_length:,], 0, weight=1.0, a=0.2, b=0.01) # not weighted, coefficient loss

        return loss_pred, torch.tensor(0.0), out

    def flat_forward(self,x):        
        uc = x # BTCHW
        for _ in range(self.predict_length): # prediction loop
            u2 = self.conv.forward(uc)
            # print(u2.shape) # TODO turn on for debugging
            uc = torch.cat([uc[:,1:],u2],dim=1)
        x2 = uc
    
        return x2      
               
    
    def edit_config(self,configs):
        if configs.patch_size != 1:
            print(f"Warning: patch_size is {configs.patch_size} but should be 1 for TF model. Setting to 1.")
            configs.patch_size = 1
        return configs