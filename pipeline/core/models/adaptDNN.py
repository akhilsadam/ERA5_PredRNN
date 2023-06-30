import torch
import math
import torch.nn as nn
from core.models.model_base import BaseModel
from core.loss import loss_mixed

class adaptDNN(BaseModel):
    # copies a lot of code from https://github.com/pytorch/examples/blob/main/word_language_model/model.py
    
    def __init__(self, num_layers, num_hidden, configs):
        super(adaptDNN, self).__init__(num_layers, num_hidden, configs)
        # torch.autograd.set_detect_anomaly(True)
        self.preprocessor = configs.preprocessor
        self.model_args = configs.model_args
        assert self.preprocessor is not None, "Preprocessor is None, please check config! Cannot operate on raw data."
        assert self.model_args is not None, "Model args is None, please check config!"
        # assert configs.input_length == configs.total_length//2, "TF model requires input_length == total_length//2"
        assert configs.input_length > 0, "Model requires input_length"
        assert configs.total_length > configs.input_length, "Model requires total_length"
        
        self.steps = self.model_args['nstep'] if 'nstep' in self.model_args else 1
        
        # transformer
        # B S E: batch, sequence, embedding (latent)
        self.preprocessor.load(device=configs.device)
        self.device = configs.device
        self.input_length = configs.input_length
        self.predict_length = configs.total_length - configs.input_length
        self.total_length = configs.total_length
        
        self.initialization = self.model_args['initialization'] if 'initialization' in self.model_args else None

        self.z = self.model_args['hidden']
        self.latent = self.preprocessor.latent_dims[-1] * self.preprocessor.patch_x * self.preprocessor.patch_y
        self.z.insert(0, self.latent)
        self.z.append(self.latent)
        # self.channels = 1
        
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(
                self.z[i], self.z[i+1],
                bias=True,
                device = self.device,
                )
            for i in range(len(self.z)-1)])
        
        act = torch.nn.ReLU() if self.model_args['activation'] == 'relu' else torch.sin()
        self.act = act if isinstance(act, list) else [act,]*(len(self.z)-1)
        
        # self.model = TF_base( \
        #                  ninp0=configs.preprocessor.latent_dims[-1],
        #                  ninp=self.model_args['n_embd'] if 'n_embd' in self.model_args else configs.preprocessor.latent_dims[-1],
        #                  nhead=self.model_args['n_head'],
        #                  nhid=self.model_args['n_ffn_embd'],
        #                  nlayers=self.model_args['n_layers'],
        #                  dropout=self.model_args['dropout'],
        #                  initialization=self.model_args['initialization'],
        #                  activation=self.model_args['activation'])
        
        nlayers = len(self.layers)
        for i,layer in enumerate(self.layers):
            self.init_FFN_weights(layer, layer_num=i, nlayers=nlayers)
            
        # self.grad_filter = nn.Conv1d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='circular')
        # weights = torch.tensor([0.5, 0., -0.5])
        # weights = weights.view(1, 1, 3).repeat(self.channels, self.channels,1)
        # with torch.no_grad():
        #     self.grad_filter.weight = nn.Parameter(weights)
            
        self.grad_filter = nn.Linear(self.latent, self.latent, bias=False)
        initrange = math.sqrt(3/self.latent)
        nn.init.uniform_(self.grad_filter.weight, -initrange, initrange)
        # nn.init.ones_(self.grad_filter.weight)
        self.resweight = nn.Parameter(torch.ones(self.steps).to(self.device)*1.0)
        self.resweight.data[1::2] = -1.0
            
    def edit_config(self,configs):
        if configs.patch_size != 1:
            print(f"Warning: patch_size is {configs.patch_size} but should be 1 for TF model. Setting to 1.")
            configs.patch_size = 1
        return configs
    
    #https://openreview.net/forum?id=qpeAhwxTopw     
    #https://openreview.net/pdf?id=qpeAhwxTopw
    #https://arxiv.org/pdf/2110.12661.pdf
    def init_FFN_weights(self,tf_encoder_layer, layer_num=0, nlayers=3):
        # initialize the weights of the feed-forward network (assuming RELU)
        # TODO need to add option if using sine activation
        fan_in = self.z[layer_num]
        fan_out = self.z[layer_num+1]
        if self.initialization not in [None,[]]:
            self.initialization(tf_encoder_layer.weight, layer_num)
            nn.init.zeros_(tf_encoder_layer.bias)
            return
        elif layer_num == 0:
            initrange = 0.0 #math.sqrt(3 / fan_in)
            nn.init.uniform_(tf_encoder_layer.weight, -initrange, initrange)
        else:
            initrange = math.sqrt(6 / (fan_in + fan_out)) # GLOROT for ReLU
            nn.init.uniform_(tf_encoder_layer.weight, -initrange, initrange)
        nn.init.zeros_(tf_encoder_layer.bias)
        
    def get_cfl(self, a,b):
        grad_mean = self.grad_filter(a).abs().mean()
        delta = b - a
        delta_mean = delta.abs().mean()
        return delta_mean / grad_mean
        
    def core_forward(self, seq_total, istrain=True):
        inl = self.configs.input_length - 1
        test = self.preprocessor.batched_input_transform(seq_total)
        nc, sx, sy = test.shape[-3:]
        test = test.reshape(test.shape[0],test.shape[1],-1)
        
        # loss_pred = 0.0
            
        # print("INPUTSIZE", inpt.size())
            
        # outpt = self.model(inpt)
        # outpt = torch.cat((inpt,outpt),dim=1)
        
        predicted = []
        last_predicted_value = test[:,inl,:].unsqueeze(1)
        
        # cfls = []
        # get CFL estimate:
        # cfl = self.cfl #self.get_cfl(test[:,inl-1,:].unsqueeze(1),last_predicted_value).item()
        # cmax = 0.9
        
        for i in range(self.predict_length):

            # Make prediction
            x0 = last_predicted_value.detach().requires_grad_(True)
            
            # cflit = int(cfl / cmax + 0.5)
            
            for j in range(self.steps):
                x = x0
                for i,layer in enumerate(self.layers):
                    y = layer(x)
                    if i < len(self.layers)-1:
                        x = self.act[i](y)
                    
                # y2 = y + 1.0
                        
                grad = self.grad_filter(x0)
                # print(f"grad: {grad.size()}")
                x2 = x0 + self.resweight[j] * y * grad / self.steps # residual connection, 
                x0 = x2

            # cfl = self.get_cfl(last_predicted_value,x0).item()
            # cfls.append(cfl)
       
            last_predicted_value = x0

            # print(f"last_predicted_value shape: {last_predicted_value.size()}")
            # Reshape from [batch_size, nlatent] --> [batch_size, 1, nlatent]

            # Detach the predicted element from the graph and concatenate with 
            # tgt in dimension 1 or 0
            # print(f"tgt shape: {tgt.size()}")
            
            tc = test[:,inl+i+1,:].unsqueeze(1)            
            
            # loss_Test = torch.nn.functional.mse_loss(last_predicted_value[2], tc[2]) # loss on batch 2
            # loss_Test.backward()
            # assert x0.grad[0].abs().sum().item() == 0, "Gradient for incorrect loss should be zero"
            
            # loss_pred += torch.nn.functional.mse_loss(last_predicted_value, tc)
            
            predicted.append(last_predicted_value)
            
            if istrain:
                # teacher forcing
                last_predicted_value = tc
        
        # print("OUTPUTSIZE", outpt.size())

        out = self.preprocessor.batched_output_transform(torch.cat(predicted,dim=1).reshape(test.shape[0], self.predict_length, nc, sx, sy))
            
        
        # print(f"AVG CFL: {cfls}")
        print(f'Resweight: {self.resweight}')
        out = torch.concat([seq_total[:,:self.configs.input_length,:],out],dim=1)
        loss_pred = loss_mixed(out, seq_total, self.input_length)
        
        loss_decouple = 100 * loss_pred * (self.resweight.sum() - self.resweight.abs().sum())**2
        return loss_pred, loss_decouple, out




   