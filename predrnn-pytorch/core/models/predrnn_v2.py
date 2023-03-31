import torch
import numpy as np
import torch.nn as nn
from core.layers.SpatioTemporalLSTMCell_v2 import SpatioTemporalLSTMCell
import torch.nn.functional as F
from core.utils import preprocess
from core.utils.tsne import visualization
import sys
import pywt as pw
from pytorch_wavelets import DWTForward, DWTInverse # (or import DWT, IDWT)\


class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(RNN, self).__init__()

        self.configs = configs
        self.visual = self.configs.visual
        self.visual_path = self.configs.visual_path
        self.skip_time = self.configs.skip_time
        self.wavelet = self.configs.wavelet
        if configs.is_WV:
            self.configs.img_channel = self.configs.img_channel * 10
        
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        # a magic number: the prescribed value for mean pressure.
        self.mean_press = 0.4475
        # wavelet transformation function
        self.xfm = DWTForward(J=1, mode='zero', wave='db1')
        self.ifm = DWTInverse(mode='zero', wave='db1')

        cell_list = []

        self.patch_size = configs.patch_size
        # print(self.frame_channel)
        if configs.is_WV:
            self.last_patch = 20
            height = int(configs.img_height/2) // self.last_patch
            width = int(configs.img_width/2) // self.last_patch
            self.frame_channel = 4*int(self.configs.img_channel/10)*self.last_patch**2
            self.img_channel = int(self.configs.img_channel/10)
            
        else:
            height = configs.img_height // self.patch_size
            width = configs.img_width // self.patch_size
            self.frame_channel = self.patch_size * self.patch_size * self.configs.img_channel
            self.img_channel = self.configs.img_channel
            # print(f"self.configs.img_channel:{self.configs.img_channel}, self.frame_channel: {self.frame_channel}")
        self.cur_height = height
        self.cur_width = width
        self.img_height = configs.img_height
        self.img_width = configs.img_width
        
        if configs.use_weight ==1 :
            self.layer_weights = np.array([float(xi) for xi in configs.layer_weight.split(',')])
            if configs.is_WV ==0:
                if self.layer_weights.shape[0] != self.configs.img_channel:
                    print('error! number of channels and weigth should be the same')
                    print('weight length: '+str(self.layer_weights.shape[0]) +', number of channel: '+str(self.configs.img_channel))
                    sys.exit()
                self.layer_weights = np.repeat(self.layer_weights, self.patch_size * self.patch_size)[np.newaxis,...]
            else:
                self.layer_weights = np.repeat(self.layer_weights, 4*self.last_patch**2)[np.newaxis,...]
        else:
            self.layer_weights = np.ones((1))
        self.layer_weights = torch.FloatTensor(self.layer_weights).to(self.configs.device)

        lat = torch.abs(torch.linspace(-np.pi/2, np.pi/2, self.img_height))
        cos_lat = torch.reshape(torch.cos(lat), (-1,1))
        self.area_weight = (cos_lat*720/torch.sum(cos_lat)).to(self.configs.device)
        # self.area_weight = (torch.ones(self.img_height)/self.img_height).to(self.configs.device)

        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(SpatioTemporalLSTMCell(in_channel, num_hidden[i], height, width, configs.filter_size,
                                                    configs.stride, configs.layer_norm).to(self.configs.device))
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel, kernel_size=1, stride=1, padding=0,
                                   bias=False).to(self.configs.device)
        # shared adapter
        adapter_num_hidden = num_hidden[0]
        self.adapter = nn.Conv2d(adapter_num_hidden, adapter_num_hidden, 1, stride=1, padding=0, bias=False)
    
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
        pred_tensor = self.reshape_back_tensor(pred_tensor)
        true_tensor = self.reshape_back_tensor(true_tensor)
        return torch.dot(torch.mean((pred_tensor*self.layer_weights-true_tensor*self.layer_weights)**2, dim=(0,1,3,4)), self.area_weight)
        
    def forward(self, frames_tensor, mask_true, istrain=True):
        '''
        frames_tensor shape: [batch, length, channel, height, width]
        '''
        batch = frames_tensor.shape[0]
        mask_true = mask_true.contiguous().to(self.configs.device)
        # print(f"in the beginning, mask_true shape: {mask_true.shape}")

        frames_tensor = self.enhance(frames_tensor)
        # print(f"ehance err:{torch.max(torch.abs(self.de_enhance(frames_tensor1) - frames_tensor))}")


        if self.configs.is_WV:
            frames_tensor = self.img_to_wv(frames_tensor)
        else:
            frames_tensor = self.reshape_tensor(frames_tensor, self.patch_size).to(self.configs.device)
        # print(f"framesTensor shape: {frames_tensor.shape}, mask_true shape: {mask_true.shape}")

        h_t = []
        c_t = []
        delta_c_list = []
        delta_m_list = []
        if self.visual:
            delta_c_visual = []
            delta_m_visual = []

        decouple_loss = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], self.cur_height, self.cur_width]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)
        
        loss = 0
        memory = torch.zeros([batch, self.num_hidden[0], self.cur_height, self.cur_width]).to(self.configs.device)
        next_frames = torch.empty(batch, self.configs.total_length-1, self.frame_channel, self.cur_height, self.cur_width).to(self.configs.device)
        # print(f"in the begining, next_frames shape:{next_frames.shape}")
       
        for t in range(0, self.configs.total_length-1):
            if self.configs.reverse_scheduled_sampling == 1:
                # reverse schedule sampling
                if t == 0:
                    net =  frames_tensor[:, t].to(self.configs.device)
                else:
                    # print(f"t: {t}, mask_true[:, t - 1]: {np.sum(mask_true[:, t - 1].detach().cpu().numpy())}")
                    net = mask_true[:, t-1] * frames_tensor[:, t] + (1 - mask_true[:, t-1]) * x_gen.to(self.configs.device)
            else:
                # schedule sampling
                if t < self.configs.input_length:
                    net = frames_tensor[:, t]
                else:
                    net = mask_true[:, t - self.configs.input_length] * frames_tensor[:, t] + \
                          (1 - mask_true[:, t - self.configs.input_length]) * x_gen
            h_t[0], c_t[0], memory, delta_c, delta_m = self.cell_list[0](net, h_t[0], c_t[0], memory)
            delta_c_list[0] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
            delta_m_list[0] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)
            if self.visual:
                delta_c_visual.append(delta_c.view(delta_c.shape[0], delta_c.shape[1], -1))
                delta_m_visual.append(delta_m.view(delta_m.shape[0], delta_m.shape[1], -1))

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory, delta_c, delta_m = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)
                delta_c_list[i] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
                delta_m_list[i] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)
                if self.visual:
                    delta_c_visual.append(delta_c.view(delta_c.shape[0], delta_c.shape[1], -1))
                    delta_m_visual.append(delta_m.view(delta_m.shape[0], delta_m.shape[1], -1))

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            # print(f"x_gen shape: {x_gen.shape}")
            next_frames[:,t] = x_gen
            # decoupling loss
            for i in range(0, self.num_layers):
                decouple_loss.append(
                    torch.mean(torch.abs(torch.cosine_similarity(delta_c_list[i], delta_m_list[i], dim=2))))

        if self.configs.press_constraint and self.configs.is_WV:
            next_frames = self.reshape_back_tensor(next_frames, self.last_patch)
            # print(f"before trans back to img, next_frames shape:{next_frames.shape}")
            next_frames = self.wv_to_img(next_frames, self.img_channel)
            # print(f"after trans back to img, next_frames shape:{next_frames.shape}")
            next_frames = self.de_enhance(next_frames)
            diff = self.mean_press - torch.mean(self.area_weight*next_frames[:,:,self.configs.layer_need_enhance])
            next_frames[:,:,self.configs.layer_need_enhance,:,:] += diff
            # print(f"frames_tensor weighted mean press: {torch.mean(self.area_weight*next_frames[:,:,self.configs.layer_need_enhance])}")
            next_frames = self.enhance(next_frames)
            next_frames = self.img_to_wv(next_frames)
            # print(f"after trans back to img, next_frames shape:{next_frames.shape}")

        if self.visual:
            # visualization of delta_c and delta_m
            delta_c_visual = torch.stack(delta_c_visual, dim=0)
            delta_m_visual = torch.stack(delta_m_visual, dim=0)
            visualization(self.configs.total_length, self.num_layers, delta_c_visual, delta_m_visual, self.visual_path)
            self.visual = 0
        decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))
        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        #next_frames = next_frames[:,np.arange(0,self.configs.total_length),:,:,:]
        #frames_tensor = frames_tensor[:,np.arange(0,self.configs.total_length),:,:,:]

        # print(f"Before loss calculation: next_frames shape:{next_frames.shape}, frames_tensor shape:{frames_tensor.shape}")
        if istrain:
            if self.configs.weighted_loss and not self.configs.is_WV:
                loss1 = self.get_weighted_loss(next_frames, frames_tensor[:,1:,:,:,:])
            else:
                loss1 = self.MSE_criterion(next_frames*self.layer_weights, frames_tensor[:,1:,:,:,:]*self.layer_weights)
            loss = loss1 + self.configs.decouple_beta*decouple_loss
            next_frames = None
        else:
            if self.configs.is_WV:
                next_frames = self.reshape_back_tensor(next_frames, self.last_patch)
                next_frames = self.wv_to_img(next_frames, self.img_channel)
        return next_frames, loss



    def wv_to_img(self, frames_wv, channel):
        '''
        frames_wv: [B, T, C, H, W]
        '''
        B, T, C, H, W = frames_wv.shape
        frames_wv = torch.reshape(frames_wv, (B*T, C, H, W))

        Yl = frames_wv[:,0:channel,:,:]
        Yh1 = frames_wv[:,channel:2*channel,:,:]
        Yh2 = frames_wv[:,2*channel:3*channel,:,:]
        Yh3 = frames_wv[:,3*channel:4*channel,:,:]
        Yh = [torch.stack((Yh1, Yh2, Yh3), dim=2)]
        img_tensor = self.ifm((Yl, Yh))
        
        C, H, W = img_tensor.shape[1:]
        img_tensor = torch.reshape(img_tensor, (B, T, C, H, W))
        return img_tensor

    def img_to_wv(self, frames_tensor):
        '''
        img_tensor: [B, T, C, H, W]
        '''
        batch = frames_tensor.shape[0]
        seq_length = frames_tensor.shape[1]
        frames_tensor = torch.reshape(frames_tensor, (batch*seq_length, self.img_channel, self.configs.img_height, self.configs.img_width))
        Yl, Yh = self.xfm(frames_tensor)

        Yl = torch.reshape(Yl, (batch, seq_length, self.img_channel, self.img_height//2, self.img_width//2))
        Yh1 = torch.reshape(Yh[0][:,:,0], (batch, seq_length, self.img_channel, self.img_height//2, self.img_width//2))
        Yh2 = torch.reshape(Yh[0][:,:,1], (batch, seq_length, self.img_channel, self.img_height//2, self.img_width//2))
        Yh3 = torch.reshape(Yh[0][:,:,2], (batch, seq_length, self.img_channel, self.img_height//2, self.img_width//2))
        # print(f"Yl: {Yl.shape}, Yh1: {Yh1.shape}, Yh2: {Yh2.shape}, Yh3: {Yh3.shape}")
                
        Yl = self.reshape_tensor(Yl, self.last_patch)
        Yh1 = self.reshape_tensor(Yh1, self.last_patch)
        Yh2 = self.reshape_tensor(Yh2, self.last_patch)
        Yh3 = self.reshape_tensor(Yh3, self.last_patch)
        # print(f"Yl: {Yl.shape}, Yh1: {Yh1.shape}, Yh2: {Yh2.shape}, Yh3: {Yh3.shape}")

        frames_tensor = torch.cat(((Yl, Yh1, Yh2, Yh3)), dim=2)
        return frames_tensor
    
    def enhance(self, img_tensor):
        #center enhance
        layer_en = img_tensor[0,:,self.configs.layer_need_enhance,:,:].clone()
        #unnormalize
        layer_en = layer_en *(105000 - 98000) + 98000
        self.zonal_mean = torch.mean(1/(layer_en[0,:,:]), dim=1) #get lattitude mean of the first time step
        layer_en = (1/layer_en) - self.zonal_mean[None,:,None]
        #re-normalize
        layer_en = (layer_en + 3e-7) / 7.7e-7
        img_tensor[0,:,self.configs.layer_need_enhance,:,:] = layer_en
        return img_tensor

    def de_enhance(self, img_tensor):
        layer_en = img_tensor[0,:,self.configs.layer_need_enhance,:,:].clone()
        #unnormalize
        layer_en = layer_en * 7.7e-7 - 3e-7
        anomaly_zonal = 1/(layer_en + self.zonal_mean[None,:,None])
        #re-normalize
        layer_en = (anomaly_zonal - 98000) / (105000 - 98000)
        img_tensor[0,:,self.configs.layer_need_enhance,:,:] = layer_en
        return img_tensor