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

from core.models.model_base import BaseModel

class RNN(BaseModel):
    def __init__(self, num_layers, num_hidden, configs):
        super(RNN, self).__init__(num_layers, num_hidden, configs)
        self.preprocessor = configs.preprocessor
        self.preprocessor.load(device=configs.device)
        self.device = configs.device
        self.input_length = configs.input_length
        self.predict_length = configs.total_length - configs.input_length
        self.total_length = configs.total_length
        self.img_channel = self.preprocessor.latent_dims[-1]
        
        shapex = self.preprocessor.patch_x
        shapey = self.preprocessor.patch_y
        configs.img_height = shapex
        configs.img_width = shapey
        
        self.wavelet = self.configs.wavelet
        if configs.is_WV:
            self.configs.img_channel = self.configs.img_channel * 10
        
        # a magic number: the prescribed value for mean pressure.
        self.mean_press = 0.4480
        self.mean_press_diff = torch.tensor([0, 1.39538188e-04,  1.99499756e-04,  1.47195304e-04,
                                            3.02930121e-05, -1.19205533e-04, -3.21148090e-04, -5.80101568e-04,
                                            -8.30103159e-04, -1.01917370e-03, -9.90518653e-04, -1.02008264e-03,
                                            -9.83945028e-04, -9.25706082e-04, -8.45840217e-04, -7.55326795e-04,
                                            -6.86238213e-04, -6.25248607e-04, -5.60617360e-04, -4.71305105e-04,
                                            -3.53708443e-04, -2.18092631e-04, -3.34181112e-04, -1.63298334e-04]).to(configs.device)
        # wavelet transformation function
        self.xfm = DWTForward(J=1, mode='zero', wave='db1')
        self.ifm = DWTInverse(mode='zero', wave='db1')

        cell_list = []

        # print(self.frame_channel)
        if configs.is_WV:           
            self.frame_channel = int(self.configs.img_channel/10)*self.patch_size**2
            self.img_channel = int(self.img_channel/10)
            
        else:
            self.frame_channel = self.configs.img_channel*self.patch_size**2
            # self.img_channel = self.configs.img_channel
            # print(f"self.configs.img_channel:{self.configs.img_channel}, self.frame_channel: {self.frame_channel}")
        
        if configs.use_weight ==1 :
            self.layer_weights = np.array([float(xi) for xi in configs.layer_weight.split(',')])
            if configs.is_WV ==0:
                if self.layer_weights.shape[0] != self.img_channel:
                    print('error! number of channels and weigth should be the same')
                    print('weight length: '+str(self.layer_weights.shape[0]) +', number of channel: '+str(self.img_channel))
                    sys.exit()
                self.layer_weights = np.repeat(self.layer_weights, self.patch_size**2)[np.newaxis,...]
            else:
                self.layer_weights = np.repeat(self.layer_weights, self.patch_size**2)[np.newaxis,...]
        else:
            self.layer_weights = np.ones((1))
        self.layer_weights = torch.FloatTensor(self.layer_weights).to(self.configs.device)

        # self.area_weight = nn.Parameter(configs.area_weight, requires_grad=False)
        self.area_weight = configs.area_weight
        self.MSE_criterion = nn.MSELoss()
        height = shapex // self.patch_size
        width = shapey // self.patch_size
        
        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i-1]
            cell_list.append(SpatioTemporalLSTMCell(in_channel, num_hidden[i], height, width, configs.filter_size,
                                                    configs.stride, configs.layer_norm))
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        # shared adapter
        adapter_num_hidden = num_hidden[0]
        self.adapter = nn.Conv2d(adapter_num_hidden, adapter_num_hidden, 1, stride=1, padding=0, bias=False)
        
    def edit_config(self,configs):
        configs.visual = 0
        configs.visual_path = ''
        configs.patch_size = configs.model_args['patch_size']
        return configs

        
    def core_forward(self, frames_tensor, istrain=True, mask_true=None):
        '''
        frames_tensor shape: [batch, length, channel, height, width]
        '''
        # print(f"Inside, frames_tensor shape:{frames_tensor.shape}, frames_tensor device: {frames_tensor.get_device()}")
        # print(f"Inside, self.area_weight device: {self.area_weight.get_device()}")
        seq_in = frames_tensor[:,:self.input_length,:]
        inpt = self.preprocessor.batched_input_transform(seq_in)
        
        nc, sx, sy = inpt.shape[-3:]
        frames_tensor = inpt.reshape(inpt.shape[0],inpt.shape[1],-1,1,1)
        

        tensor_device = frames_tensor.get_device()
        self.mean_press = torch.mean(self.area_weight*frames_tensor[:,0,self.configs.layer_need_enhance])
        batch = frames_tensor.shape[0]
        mask_true = mask_true.contiguous()
        
        if self.configs.center_enhance:
            chan_en = frames_tensor[0,:,self.configs.layer_need_enhance,:,:]*(105000-98000)+98000
            self.zonal_mean = torch.mean(1/(chan_en[0,:,:]), dim=1) #get lattitude mean of the first time step

        if self.configs.is_WV:
            frames_tensor = self.img_to_wv(frames_tensor)
        else:
            if self.configs.center_enhance:
                frames_tensor = self.enhance(frames_tensor)
            frames_tensor = self.reshape_tensor(frames_tensor, self.patch_size)
        # print(f"frames_tensor shape: {frames_tensor.shape}, frames_tensor device:{frames_tensor.device}, mask_true shape: {mask_true.shape}")

        h_t = []
        c_t = []
        delta_c_list = []
        delta_m_list = []
        if self.visual:
            delta_c_visual = []
            delta_m_visual = []

        decouple_loss = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], self.cur_height, self.cur_width]).to(tensor_device)
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)
        
        loss = 0
        memory = torch.zeros([batch, self.num_hidden[0], self.cur_height, self.cur_width]).to(tensor_device)
        next_frames = torch.empty(batch, self.configs.total_length-1, self.frame_channel, self.cur_height, self.cur_width).to(tensor_device)
        # print(f"in the begining, next_frames shape:{next_frames.shape}")
        gen_pressure_mean = 0
        for t in range(0, self.configs.total_length-1):
            if self.configs.reverse_scheduled_sampling == 1:
                # reverse schedule sampling
                if t == 0:
                    net =  frames_tensor[:, t]
                else:
                    # print(f"t: {t}, mask_true[:, t - 1]: {np.sum(mask_true[:, t - 1].detach().cpu().numpy())}")
                    net = mask_true[:, t-1] * frames_tensor[:, t] + (1 - mask_true[:, t-1]) * x_gen
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
            x_gen = torch.unsqueeze(x_gen, dim=1)
            # print(f"x_gen shape: {x_gen.shape}")
            if self.configs.press_constraint:
                if self.configs.is_WV:
                    # print(f"wv_to_img+de_enhance err:{torch.max(torch.abs(self.img_to_wv(self.enhance(self.de_enhance(self.wv_to_img(x_gen)))) - x_gen))}")
                    x_gen = self.wv_to_img(x_gen)
                    # print(f"img_to_wv err:{torch.max(torch.abs(self.wv_to_img(self.img_to_wv(x_gen)) - x_gen))}")
                    # print(f"after trans back to img, x_gen shape:{x_gen.shape}")
                    gen_mean = torch.mean(self.area_weight*x_gen[:,:,self.configs.layer_need_enhance])
                    x_gen[:,:,self.configs.layer_need_enhance,:,:] = x_gen[:,:,self.configs.layer_need_enhance,:,:] + self.mean_press - gen_mean
                    # print(f"frames_tensor weighted mean press: {torch.mean(self.area_weight*x_gen[:,:,self.configs.layer_need_enhance])}")
                    x_gen= self.img_to_wv(x_gen)
                    # print(f"wv_to_img+de_enhance err:{torch.max(torch.abs(self.img_to_wv(self.wv_to_img(x_gen)) - x_gen))}")
                else:
                    x_gen = self.reshape_back_tensor(x_gen, self.patch_size)
                    if self.configs.center_enhance:
                        x_gen = self.de_enhance(x_gen)
                    gen_mean = torch.mean(self.area_weight*x_gen[:,:,self.configs.layer_need_enhance])
                    x_gen[:,:,self.configs.layer_need_enhance,:,:] = x_gen[:,:,self.configs.layer_need_enhance,:,:]+self.mean_press+self.mean_press_diff[(t+1)%24]-gen_mean
                    if self.configs.center_enhance:
                        x_gen = self.enhance(x_gen)
                    x_gen = self.reshape_tensor(x_gen, self.patch_size)

            x_gen = torch.squeeze(x_gen, dim=1)
            next_frames[:,t] = x_gen

            # decoupling loss
            for i in range(0, self.num_layers):
                decouple_loss.append(
                    torch.mean(torch.abs(torch.cosine_similarity(delta_c_list[i], delta_m_list[i], dim=2))))

        
        if self.visual:
            # visualization of delta_c and delta_m
            delta_c_visual = torch.stack(delta_c_visual, dim=0)
            delta_m_visual = torch.stack(delta_m_visual, dim=0)
            visualization(self.configs.total_length, self.num_layers, delta_c_visual, delta_m_visual, self.visual_path)
            self.visual = 0
        
        decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))

        if self.configs.display_press_mean:
            if self.configs.is_WV:
                next_frames1 = self.wv_to_img(next_frames)
                gen_mean = torch.mean(self.area_weight*next_frames1[:,:,self.configs.layer_need_enhance])
            else:
                next_frames1 = self.reshape_back_tensor(next_frames, self.patch_size)
                if self.configs.center_enhance:
                    next_frames1 = self.de_enhance(next_frames1)
                gen_mean = torch.mean(self.area_weight*next_frames1[:,:,self.configs.layer_need_enhance])
            print(f"The generated pressure mean: {gen_mean}")

        if self.configs.weighted_loss and not self.configs.is_WV:
            loss_pred = self.get_weighted_loss(next_frames, frames_tensor[:,1:,:,:,:])
        else:
            loss_pred = self.MSE_criterion(next_frames, frames_tensor[:,1:,:,:,:])
        
        if not istrain and self.configs.is_WV:
                next_frames = self.wv_to_img(next_frames)
                
        outpt = next_frames
        outpt = outpt.reshape(outpt.shape[0],outpt.shape[1],nc,sx,sy)       
        out = self.preprocessor.batched_output_transform(outpt)
       
        all_frames = torch.cat([frames_tensor[:, 0].unsqueeze(1), out], dim=1)
        return loss_pred, decouple_loss, all_frames
    
    def enhance(self, img_tensor):
        #center enhance
        chan_en = img_tensor[0,:,self.configs.layer_need_enhance,:,:].clone()
        #unnormalize
        chan_en = chan_en *(105000 - 98000) + 98000
        chan_en = (1/chan_en) - self.zonal_mean[None,:,None]
        #re-normalize
        chan_en = (chan_en + 3e-7) / 7.7e-7
        img_tensor[0,:,self.configs.layer_need_enhance,:,:] = chan_en
        return img_tensor

    def de_enhance(self, img_tensor):
        chan_en = img_tensor[0,:,self.configs.layer_need_enhance,:,:].clone()
        #unnormalize
        chan_en = chan_en * 7.7e-7 - 3e-7
        anomaly_zonal = 1/(chan_en + self.zonal_mean[None,:,None])
        #re-normalize
        chan_en = (anomaly_zonal - 98000) / (105000 - 98000)
        img_tensor[0,:,self.configs.layer_need_enhance,:,:] = chan_en
        return img_tensor