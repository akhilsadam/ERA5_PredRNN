from dataclasses import dataclass
import torch
import gc, os, jpcm
import numpy as np
import torch.nn as nn
import logging
import imageio.v3 as imageio
from tqdm import tqdm

from .base import PreprocessorBase

logger = logging.getLogger('CNN-preprocessor')

class CNNAutoencoder(nn.Module):
    def __init__(self, k1, k2, channels,wp=False):
        super().__init__()
        assert k1%2==1 and k2%2==1, "Kernel sizes must be odd!"
        # standard conv layer, then several atrous conv layer 
      
        
        
        
        # 720x1440 -> 45x90 requires ~16x reduction (about 4000 latent dim!)
        # 64x64 -> 4x4
        
        self.conv2 = nn.Conv2d(channels, channels, (k2, k2), dilation=(2,2)) 
        self.conv3 = nn.Conv2d(channels, channels, (k2, k2), dilation=(2,2)) # 
        self.conv4 = nn.Conv2d(channels, channels, (k2, k2), dilation=(2,2)) 
        self.conv5 = nn.Conv2d(channels, channels, (k2, k2), dilation=(2,2))
        
        if wp:
            self.conv1 = nn.Conv2d(channels, channels, (k1, k1)) # local receptive field
            self.pad_1x = lambda x : nn.functional.pad(x, (k1//2,k1//2,0,0), mode='circular')
            self.pad_1y = nn.ReflectionPad2d((0,0,k1//2,k1//2))
            self.encoder_cd = nn.Sequential(self.conv1, nn.ReLU(), self.conv2, nn.ReLU(), self.conv3, nn.ReLU(), self.conv4, nn.ReLU(), self.conv5)
            self.encoder_c = lambda x: self.encoder_cd(self.pad_1y(self.pad_1x(x)))
        else:
            self.conv1 = nn.Conv2d(channels, channels, (k1, k1), padding='same') # local receptive field
            self.encoder_c = nn.Sequential(self.conv1, nn.ReLU(), self.conv2, nn.ReLU(), self.conv3, nn.ReLU(), self.conv4, nn.ReLU(), self.conv5)
            
        # decoder
        self.rconv5 = nn.ConvTranspose2d(channels, channels, (k2, k2), dilation=(2,2)) #
        self.rconv4 = nn.ConvTranspose2d(channels, channels, (k2, k2), dilation=(2,2)) #
        self.rconv3 = nn.ConvTranspose2d(channels, channels, (k2, k2), dilation=(2,2)) #
        self.rconv2 = nn.ConvTranspose2d(channels, channels, (k2, k2), dilation=(2,2)) #
        self.rconv1 = nn.Conv2d(channels, channels, (k1, k1), padding='same') # local receptive field
 
        self.decoder_c = nn.Sequential(self.rconv5, nn.ReLU(), self.rconv4, nn.ReLU(), self.rconv3, nn.ReLU(), self.rconv2, nn.ReLU(), self.rconv1)
    
        self.encoder = self.encoder2D
        self.decoder = self.decoder2D
        
    def encoder2D(self, inpt):
        shape = inpt.shape
        inpt = inpt.reshape(-1, shape[-3], shape[-2], shape[-1]) # NCHW
        cnnpt = self.encoder_c(inpt)
        self.cnnl = cnnpt.shape[-3]
        self.cnnx = cnnpt.shape[-2]
        self.cnny = cnnpt.shape[-1]
        self.final_shape = shape
        return cnnpt.reshape(shape[0], shape[1], self.cnnl,self.cnnx,self.cnny) # NCL
    
    def decoder2D(self, outpt):
        outpt = outpt.reshape(-1, self.cnnl, self.cnnx, self.cnny) # NCHW
        cnnpto = self.decoder_c(outpt)
        return cnnpto.reshape(self.final_shape) # NTCHW 


class Preprocessor(PreprocessorBase, nn.Module): # inherit from both, so we can use the preprocessor as a module
    def __init__(self, config):
        
        cdict = {
            'datadir': 'data', # directory where data is stored
        }
        cdict.update(config)
        for k,v in cdict.items():
            setattr(self, k, v)
            
        super(Preprocessor,self).__init__(config=config) # call PreprocessorBase init
        nn.Module.__init__(self) # call nn.Module init
        
        self.wp = 'WP_' if self.weather_prediction else ''                    

        self.flags = ['spatial',# if this flag is present, then we expect a spatial dataset NTCHW with self.reduced_shape (CHW)
                      ]


        shape, _ = self.precompute_scale(use_datasets=False)
        self.shape = shape
        
        self.CNNAuto = CNNAutoencoder(5,7,self.shape[1], self.weather_prediction).to(self.device)
        
        self.CNNAuto.encoder(torch.zeros((1,*self.shape[1:])).to(self.device))
        self.reduced_shape = (self.CNNAuto.cnnl, self.CNNAuto.cnnx, self.CNNAuto.cnny) # not bothering to calculate by hand
        

        
    def load(self, device):
        '''
        [B, T, C, H, W] -> [B, T, latent, H, W]
        '''
        self.scale, self.shift = super().load_scale(device)
                    
        self.patch_x = self.reduced_shape[1]
        self.patch_y = self.reduced_shape[2]
        self.latent_dims = list(range(self.reduced_shape[0]+1))
        # self.input_transform = lambda x: torch.stack([in_tf(data[v]['method'])(data_torch[v],x[:,v,:,:]) for v in range(self.n_var)],dim=1)
        self.batched_input_transform = lambda x: self.CNNAuto.encoder(x*self.scale + self.shift)
        # self.output_transform = lambda a: torch.stack([out_tf(data[v]['method'])(data_torch[v],a[:,latent_dims[v]:latent_dims[v+1]]).reshape(-1, self.shapex, self.shapey) for v in range(self.n_var)],dim=1)
        self.batched_output_transform = lambda a: self.CNNAuto.decoder((a-self.shift)/self.scale)