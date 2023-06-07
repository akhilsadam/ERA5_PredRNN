import logging
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
    def __init__(self, num_layers, num_hidden, configs):
        super(BaseModel, self).__init__()

        self.configs = configs
        self.visual = self.configs.visual
        self.visual_path = self.configs.visual_path
        self.skip_time = self.configs.skip_time
        # self.wavelet = self.configs.wavelet
        
        self.num_layers = num_layers
        self.num_hidden = num_hidden
       
        self.patch_size = configs.patch_size
        height = configs.img_height // self.patch_size
        width = configs.img_width // self.patch_size
        # print(self.frame_channel)
        self.frame_channel = self.configs.img_channel*self.patch_size**2
        self.img_channel = self.configs.img_channel
        # print(f"self.configs.img_channel:{self.configs.img_channel}, self.frame_channel: {self.frame_channel}")
        self.cur_height = height
        self.cur_width = width
        self.img_height = configs.img_height
        self.img_width = configs.img_width
    
    
    @abstractmethod
    def init_memory(self):
        """Override this method in your model if you use memory units. Runs on start of each call to forward()"""
        return None
    
    @abstractmethod
    def core_forward(self, last_frame, memory):
        """This method should contain the actual forward pass of your model. Runs on each timestep in the forward() loop"""
        return last_frame

    @abstractmethod
    def get_decouple_loss(self, layer):
        """This method should return the decoupling loss for a given layer. Decoupling loss is defined as a dot product of spatial and temporal states for ConvLSTM frameworks"""
        return torch.tensor(0.0).to(self.configs.device)
        
    def forward(self, frames_tensor, mask_true, istrain=True):
        '''
        frames_tensor shape: [batch, length, channel, height, width]
        '''
        # print(f"Inside, frames_tensor shape:{frames_tensor.shape}, frames_tensor device: {frames_tensor.get_device()}")
        # print(f"Inside, self.area_weight device: {self.area_weight.get_device()}")

        tensor_device = frames_tensor.get_device()
        batch = frames_tensor.shape[0]
        mask_true = mask_true.contiguous()

        decouple_loss = []

        loss = 0
        memory = self.init_memory()
        next_frames = torch.empty(batch, self.configs.total_length-1, self.frame_channel, self.cur_height, self.cur_width).to(tensor_device)
        # print(f"in the begining, next_frames shape:{next_frames.shape}")

        for t in range(self.configs.total_length-1):
            if (
                self.configs.reverse_scheduled_sampling == 1
                and t == 0
                or self.configs.reverse_scheduled_sampling != 1
                and t < self.configs.input_length
            ):
                net =  frames_tensor[:, t]
            elif self.configs.reverse_scheduled_sampling == 1:
                # print(f"t: {t}, mask_true[:, t - 1]: {np.sum(mask_true[:, t - 1].detach().cpu().numpy())}")
                net = mask_true[:, t-1] * frames_tensor[:, t] + (1 - mask_true[:, t-1]) * x_gen
            else:
                net = mask_true[:, t - self.configs.input_length] * frames_tensor[:, t] + \
                              (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            x_gen = self.core_forward(net, memory)
            next_frames[:,t] = x_gen

            # decoupling loss
            decouple_loss.extend(
                self.get_decouple_loss(i)
                for i in range(self.num_layers)
            )
            
        decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))

        if self.configs.weighted_loss and not self.configs.is_WV:
            loss_pred = self.get_weighted_loss(next_frames, frames_tensor[:,1:,:,:,:])
        else:
            loss_pred = self.MSE_criterion(next_frames, frames_tensor[:,1:,:,:,:])
        print(f"loss_pred:{loss_pred}, decouple_loss:{decouple_loss}")
        loss = loss_pred + self.configs.decouple_beta*decouple_loss

        if istrain:
            next_frames = None
            return loss, loss_pred.detach(), decouple_loss.detach()
        else:
            if self.configs.is_WV:
                next_frames = self.wv_to_img(next_frames)
            else:
                next_frames = self.reshape_back_tensor(next_frames, self.patch_size)
            return next_frames, loss, loss_pred, decouple_loss
    

    def reshape_back_tensor(self, img_tensor, patch_size):
        assert 5 == img_tensor.ndim
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