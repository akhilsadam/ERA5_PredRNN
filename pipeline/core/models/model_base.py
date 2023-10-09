import logging
import abc
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from core.utils import preprocess
from core.utils.tsne import visualization

logging.basicConfig(level = logging.INFO,format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%y-%m-%d %H:%M',)
logger = logging.getLogger(__name__)

def abstractmethod(f):
    logger.warn(f"Using default method for {f.__name__}. This should be overridden in the subclass.")
    return f

class BaseModel(nn.Module):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, num_layers, num_hidden, configs):
        super(BaseModel, self).__init__()

        self.configs = self.edit_config(configs) 
        
        self.visual = self.configs.visual
        self.visual_path = self.configs.visual_path
        self.skip_time = self.configs.skip_time
        # self.wavelet = self.configs.wavelet
        
        self.num_layers = num_layers
        self.num_hidden = num_hidden
       
        self.patch_size = self.configs.patch_size
        height = self.configs.img_height // self.patch_size
        width = self.configs.img_width // self.patch_size
        # print(self.frame_channel)
        self.frame_channel = self.configs.img_channel*self.patch_size**2
        self.img_channel = self.configs.img_channel
        # print(f"self.configs.img_channel:{self.configs.img_channel}, self.frame_channel: {self.frame_channel}")
        self.cur_height = height
        self.cur_width = width
        self.img_height = self.configs.img_height
        self.img_width = self.configs.img_width
    
    @abc.abstractmethod
    def edit_config(self,configs):
        '''Modify configs to match the model. This method should be overridden in the subclass.'''
        abstractmethod(__name__)
        return configs
    
    @abc.abstractmethod
    def core_forward(self, seq_in, istrain, **kwargs):
        """This method should contain the actual forward pass of your model. Runs on each timestep in the forward() loop"""
        abstractmethod(__name__)

        
    def forward(self, frames_tensor, mask_true=None, istrain=True):
        '''
        frames_tensor shape: [batch, length, channel, height, width]
        '''
        loss_pred, decouple_loss, next_frames = self.core_forward(frames_tensor, istrain, mask_true=mask_true)
            
        print(f"loss_pred:{loss_pred}, decouple_loss:{decouple_loss}")
        loss = loss_pred + self.configs.decouple_beta*decouple_loss

        if istrain:
            next_frames = None
            return loss, loss_pred.detach(), decouple_loss.detach()
        else:
            next_frames = self.reshape_back_tensor(next_frames, self.patch_size)
            return next_frames, loss, loss_pred, decouple_loss
    

    def reshape_back_tensor(self, img_tensor, patch_size):
        assert 5 == img_tensor.ndim, f"img_tensor.ndim: {img_tensor.ndim} != 5 in reshape_back_tensor"
        batch_size = img_tensor.shape[0]
        seq_length = img_tensor.shape[1]
        channels = img_tensor.shape[2]
        cur_height = img_tensor.shape[3]
        cur_width = img_tensor.shape[4]
        img_channels = channels // (patch_size*patch_size)
        img_tensor = torch.reshape(img_tensor, [batch_size, seq_length, img_channels, patch_size, patch_size,
                                        cur_height, cur_width])
        img_tensor = img_tensor.permute(0,1,2,5,3,6,4)
        return torch.reshape(img_tensor, [batch_size, seq_length, img_channels,
                                          cur_height*patch_size, cur_width*patch_size])

    def reshape_tensor(self, img_tensor, patch_size):
        assert 5 == img_tensor.ndim
        batch_size = img_tensor.shape[0]
        seq_length = img_tensor.shape[1]
        channels = img_tensor.shape[2]
        cur_height = img_tensor.shape[3]
        cur_width = img_tensor.shape[4]

        img_tensor = torch.reshape(img_tensor, [batch_size, seq_length, channels, 
                                                cur_height//patch_size, patch_size,
                                                cur_width//patch_size, patch_size])
        img_tensor = img_tensor.permute(0,1,2,4,6,3,5)
        return torch.reshape(img_tensor, [batch_size, seq_length, channels*patch_size*patch_size,
                                          cur_height//patch_size, cur_width//patch_size])


    def get_weighted_loss(self, pred_tensor, true_tensor):
        pred_tensor = self.reshape_back_tensor(pred_tensor, self.patch_size)
        true_tensor = self.reshape_back_tensor(true_tensor, self.patch_size)
        # print(f"self.area_weight shape: {self.area_weight.shape}, pred_tensor shape: {pred_tensor.shape}, true_tensor shape:{true_tensor.shape}")
        return torch.mean((pred_tensor-true_tensor)**2*self.area_weight)

    def wv_to_img(self, frames_wv):
        '''
        frames_wv: [B, T, C, H, W]
        '''
        frames_wv = self.reshape_back_tensor(frames_wv, self.patch_size//2)
        B, T, C, H, W = frames_wv.shape
        frames_wv = torch.reshape(frames_wv, (B*T, C, H, W))

        Yl = frames_wv[:,0:self.img_channel,:,:]
        Yh1 = frames_wv[:,self.img_channel:2*self.img_channel,:,:]
        Yh2 = frames_wv[:,2*self.img_channel:3*self.img_channel,:,:]
        Yh3 = frames_wv[:,3*self.img_channel:4*self.img_channel,:,:]
        Yh = [torch.stack((Yh1, Yh2, Yh3), dim=2)]

        img_tensor = self.ifm((Yl, Yh))
        C, H, W = img_tensor.shape[1:]
        img_tensor = torch.reshape(img_tensor, (B, T, C, H, W))
        if self.configs.center_enhance:
            img_tensor = self.de_enhance(img_tensor)
        return img_tensor

    def img_to_wv(self, frames_tensor):
        '''
        img_tensor: [B, T, C, H, W]
        '''
        if self.configs.center_enhance:
            frames_tensor = self.enhance(frames_tensor)
        batch = frames_tensor.shape[0]
        seq_length = frames_tensor.shape[1]
        frames_tensor = torch.reshape(frames_tensor, (batch*seq_length, self.img_channel, self.configs.img_height, self.configs.img_width))
        Yl, Yh = self.xfm(frames_tensor)
        frames_tensor = torch.cat(((Yl, Yh[0][:,:,0], Yh[0][:,:,1], Yh[0][:,:,2])), dim=1)
        frames_tensor = torch.reshape(frames_tensor, (batch, seq_length, self.img_channel*4, self.img_height//2, self.img_width//2))
        frames_tensor = self.reshape_tensor(frames_tensor, self.patch_size//2)
        return frames_tensor