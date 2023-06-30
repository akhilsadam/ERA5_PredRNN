import torch
import math
import torch.nn as nn
from core.models.model_base import BaseModel
from core.loss import loss_mixed
  
class LSTM(BaseModel):
    # copies a lot of code from https://github.com/pytorch/examples/blob/main/word_language_model/model.py
    
    def __init__(self, num_layers, num_hidden, configs):
        super(LSTM, self).__init__(num_layers, num_hidden, configs)
        self.preprocessor = configs.preprocessor
        self.model_args = configs.model_args
        assert self.preprocessor is not None, "Preprocessor is None, please check config! Cannot operate on raw data."
        assert self.model_args is not None, "Model args is None, please check config!"
        # assert configs.input_length == configs.total_length//2, "TF model requires input_length == total_length//2"
        assert configs.input_length > 0, "Model requires input_length"
        assert configs.total_length > configs.input_length, "Model requires total_length"
        
        # transformer
        # B S E: batch, sequence, embedding (latent)
        self.preprocessor.load(device=configs.device)
        self.device = configs.device
        self.input_length = configs.input_length
        self.predict_length = configs.total_length - configs.input_length
        self.total_length = configs.total_length
        
        in_dim = self.preprocessor.latent_dims[-1] * self.preprocessor.patch_x * self.preprocessor.patch_y

        
        self.encoder = nn.Linear(in_dim, self.model_args['n_embd'])
        self.decoder = nn.Linear(self.model_args['n_embd'], in_dim)
        nlayers=self.model_args['n_layers']
        dropout=self.model_args['dropout']
        self.lstm = nn.LSTM(self.model_args['n_embd'], self.model_args['n_embd'], nlayers, batch_first=True, dropout=dropout, bidirectional=False)  
        
                        #  nhead=self.model_args['n_head'],
                        #  nhid=self.model_args['n_ffn_embd'],
                        
        self.H0 = nn.Parameter(torch.zeros(nlayers, 1, self.model_args['n_embd']))
        self.C0 = nn.Parameter(torch.zeros(nlayers, 1, self.model_args['n_embd']))
                         
        #  
        # initialization=self.model_args['initialization'],
                        #  activation=self.model_args['activation'])
        
        # transformer
        # B S E: batch, sequence, embedding (latent)
        
    def edit_config(self,configs):
        if configs.patch_size != 1:
            print(f"Warning: patch_size is {configs.patch_size} but should be 1 for TF model. Setting to 1.")
            configs.patch_size = 1
        return configs
        
    def core_forward(self, seq_total, istrain=True):
        seq_in = seq_total[:,:self.input_length,:]
        inpt = self.preprocessor.batched_input_transform(seq_in)
            
        nc, sx, sy = inpt.shape[-3:]
        inpt = inpt.reshape(inpt.shape[0],inpt.shape[1],-1)
            
        inpt_encoded = self.encoder(inpt)
            
        h0 = self.H0.expand(-1, inpt.size(0), -1).contiguous()
        c0 = self.C0.expand(-1, inpt.size(0), -1).contiguous()    
            
        outpt_encoded, _ = self.lstm(inpt_encoded,(h0,c0))
                
        outpt = self.decoder(outpt_encoded)
        outpt = outpt.reshape(outpt.shape[0],outpt.shape[1],nc,sx,sy)    
        out = self.preprocessor.batched_output_transform(outpt)
        out = torch.cat((seq_total[:,:self.input_length,:],out),dim=1)

            
        loss_pred = loss_mixed(out, seq_total, self.input_length)
        loss_decouple = torch.tensor(0.0)
        return loss_pred, loss_decouple, out

    def init_weights(self):
        initrange = math.sqrt(3 / self.ninp) #0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        initrange = math.sqrt(6 / (self.ninp + self.ntoken))
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def init_FFN_weights(self,tf_encoder_layer, layer_num=0):
        # initialize the weights of the feed-forward network (assuming RELU)
        # TODO need to add option if using sine activation
        if self.initialization not in [None,[]]:
            self.initialization(tf_encoder_layer.linear1.weight, layer_num)
            self.initialization(tf_encoder_layer.linear2.weight, layer_num)
        else:
            initrange = math.sqrt(3 / self.ninp)
            nn.init.uniform_(tf_encoder_layer.linear1.weight, -initrange, initrange)
            nn.init.uniform_(tf_encoder_layer.linear2.weight, -initrange, initrange)
        nn.init.zeros_(tf_encoder_layer.linear1.bias)
        nn.init.zeros_(tf_encoder_layer.linear2.bias)


    