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
        self.operator = operator_class(self.latent, self.filter_size**2, device=device, *operator_args, **operator_kwargs)
        self.device = device
        self.pad = (filter_size - 1) // 2
        self.puh = self.uh + 2 * self.pad
        self.puw = self.uw + 2 * self.pad
        
        self.buffer = None
        
    def ind(self, i, j):
        
        il = i * self.uh + self.uh // 2
        ilp = il - self.pad
        ih = il + self.uh
        ihp = ih + self.pad
        jl = j * self.uw + self.uw // 2
        jlp = jl - self.pad
        jh = jl + self.uw 
        jhp = jh + self.pad
        
        return il, ih, jl, jh, ilp, ihp, jlp, jhp
    
    def unfold_operator(self,in_data):
        # data has shape uw+2xpad, uh+2xpad:
        flat_in_shape = (in_data.shape[0]*in_data.shape[1]*in_data.shape[2], 1, self.puh, self.puw) # (BxTxC)_HW 

        full_uf_shape = (*in_data.shape[:-2],self.s**2,self.patches) # BTCLp

        
        small = in_data.reshape(flat_in_shape) # BTCHW  -> (BxTxC)_HW 
        large = F.unfold(small, (self.filter_size,self.filter_size)).reshape(full_uf_shape) # (BxTxC)_HW, hw -> (BxTxC)Lp -> BTCLp : L:=hxw

        windows = large.permute((0,4,1,2,3))# BTCLp -> BpTCL
        out = self.operator(windows).permute((0,2,3,1)) # BpTCL -> BpTC -> BTCp
        
        return out.reshape((out.shape[0],out.shape[1],out.shape[2],self.uh,self.uw))
    
    def forward(self, x):
        y = torch.empty((x.shape[0],1,*x.shape[2:]), device=self.device) # blanked timestep
        
        if (self.buffer is None) or (self.buffer.shape[0] != x.shape[0]):
            del self.buffer
            self.batch = x.shape[0]
            self.buffer = torch.empty((self.batch, self.t, self.latent, self.uw, self.uh), device=self.device)
            
        
        for i in range(self.n-1): # height / first dim
            for j in range(self.n-1): # width / second dim
                
                # centerface tiles
                il, ih, jl, jh, ilp, ihp, jlp, jhp = self.ind(i,j)
                           
                y[:,:,:,il:ih,jl:jh] = self.unfold_operator(x[:,:,:,ilp:ihp,jlp:jhp])
                
                
        # TODO fix side and corner faces!
        # i = self.n
        # for j in range(self.n-1):
            
        #     il, ih, jl, jh, ilp, ihp, jlp, jhp = self.ind(i,j)
            
        #     im = self.h
            
        #     self.buffer[:,:,:,:im-ilp,:] = x[:,:,:,ilp:,jlp:jhp]
        #     self.buffer[:,:,:,im-ilp:,:] = x[:,:,:,:ihp,jlp:jhp]
            
        #     out = self.unfold_operator(self.buffer)
            
        #     y[:,:,:,il:,jl:jh] = out[:,:,:,:im-il,:]
        #     y[:,:,:,:ih,jl:jh] = out[:,:,:,im-il:,:]
            
            
        # j = self.n
        # for i in range(self.n-1):
            
        #     il, ih, jl, jh, ilp, ihp, jlp, jhp = self.ind(i,j)
            
        #     jm = self.w
            
        #     self.buffer[:,:,:,:,:jm-jlp] = x[:,:,:,ilp:ihp,jlp:]
        #     self.buffer[:,:,:,:,jm-jlp:] = x[:,:,:,ilp:ihp,:jhp]
            
        #     out = self.unfold_operator(self.buffer)
            
        #     y[:,:,:,il:ih,jl:] = out[:,:,:,:,:jm-jl]
        #     y[:,:,:,il:ih,:jh] = out[:,:,:,:,jm-jl:]

        # i = self.n
        # j = self.n
            
        # il, ih, jl, jh, ilp, ihp, jlp, jhp = self.ind(i,j)
        
        # im = self.h
        # jm = self.w
        
        # self.buffer[:,:,:,:im-ilp,:jm-jlp] = x[:,:,:,ilp:,jlp:] # bottom right of x
        # self.buffer[:,:,:,im-ilp:,:jm-jlp] = x[:,:,:,:ihp,jlp:]
        # self.buffer[:,:,:,:im-ilp,jm-jlp:] = x[:,:,:,ilp:,:jhp]
        # self.buffer[:,:,:,im-ilp:,jm-jlp:] = x[:,:,:,:ihp,:jhp] # top left of x
        
        # out = self.unfold_operator(self.buffer)
        
        # y[:,:,:,il:,jl:] = out[:,:,:,:im-il,:jm-jl]
        # y[:,:,:,il:,:jh] = out[:,:,:,:im-il,jm-jl:]               
        # y[:,:,:,:ih,jl:] = out[:,:,:,im-il:,:jm-jl]
        # y[:,:,:,:ih,:jh] = out[:,:,:,im-il:,jm-jl:]                  
    
        return y
        