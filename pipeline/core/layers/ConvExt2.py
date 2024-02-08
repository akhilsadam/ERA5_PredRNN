__author__ = 'asadam'

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvX(nn.Module):
    def __init__(self, in_time, in_channel, height, width, filter_size, slices, operator_class, device, operator_args=[], operator_kwargs={}):
        super(ConvX, self).__init__()

        assert height % 2*slices == 0, "Cannot slice, height is not evenly divisible into desired slices" 
        assert width % 2*slices == 0, "Cannot slice, width is not evenly divisible into desired slices" 
        assert filter_size % 2 == 1, "Filter size must be odd"
        
        self.latent = in_channel
        self.t = in_time
        self.h = height # wrapped BC here by def'n # should be zero or ignored, though
        self.w = width # wrapped BC here by def'n
        self.filter_size = filter_size
        self.s = filter_size
        self.n = slices
        self.uh = height // slices
        self.uw = width // slices
        self.patches = self.uh*self.uw
        self.operator = operator_class(self.latent, self.filter_size**2, in_time, device=device, *operator_args, **operator_kwargs)
        self.device = device
        self.pad = (filter_size - 1) // 2
        self.puh = self.uh + 2 * self.pad
        self.puw = self.uw + 2 * self.pad
        
        # self.buffer = None
        
        # self.traced_operator = torch.jit.trace(self.operator,torch.rand((1,self.patches,self.t,self.latent,self.filter_size**2),device=self.device))
        
    def ind(self, j, i):
        
        jc = j * self.uh
        jl = jc #- self.pad
        jh = jc + self.puh
        
        ic = i * self.uw 
        il = ic #- self.pad
        ih = ic + self.puw
        
        return jl, jh, il, ih
    
    def unfold_operator(self,in_data,shape):
        # data has shape uw+2xpad, uh+2xpad:
        flat_in_shape = (*in_data.shape[:-2],1,self.puh,self.puw) # (BxTxC)_HW 

        full_uf_shape = (*shape,self.s**2,self.patches) # BTCLp

        
        small = in_data.reshape(flat_in_shape) # BTCHW  -> (BxTxC)_HW 
        large = F.unfold(small, (self.filter_size,self.filter_size)).reshape(full_uf_shape) # (BxTxC)_HW, hw -> (BxTxC)Lp -> BTCLp : L:=hxw

        windows = large.permute((0,4,1,2,3))# BTCLp -> BpTCL
        out = self.operator(windows).permute((0,2,3,1)) # BpTCL -> BpTC -> BTCp
        
        return out.reshape((out.shape[0],out.shape[1],out.shape[2],self.uh,self.uw))
    
    def forward(self, x):
        
        # size is BTCHW, or BTCyx
        shape = x.shape[:-2]        
        x = x.reshape((x.shape[0]*x.shape[1]*x.shape[2], self.h, self.w)) # BTCHW  -> (BxTxC)HW
        
        # pad
        x = F.pad(x, (self.pad,self.pad), mode='circular')
        # print("PADDED")
        x = F.pad(x, (0,0,self.pad,self.pad), mode='constant', value=0) # size is (BxTxC)hw, where hw = H+2p, W+2p
        
        # print(x.shape)
        
        # double for loop version of unfold
        patches = []
        for i in range(self.n):
            
            ypatches = []
            
            for j in range(self.n):
                
                jl, jh, il, ih = self.ind(j,i)
                
                ypatches.append(self.unfold_operator(x[:,jl:jh,il:ih],shape))
            
            patches.append(torch.cat(ypatches,dim=-2))
        
        y = torch.cat(patches,dim=-1)                          
    
        return y
        