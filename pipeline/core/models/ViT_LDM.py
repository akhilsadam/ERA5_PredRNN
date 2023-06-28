import torch
import math
import torch.nn as nn
import numpy as np
from core.models.model_base import BaseModel
from core.loss import loss_mixed

 #TODO UPDATE THIS to work with new input/output transforms
 
# also check why loss is nan

class ViT_LDM(BaseModel):
    # copies a lot of code from https://github.com/pytorch/examples/blob/main/word_language_model/model.py
    # https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
    
    def __init__(self, num_layers, num_hidden, configs):
        super(ViT_LDM, self).__init__(num_layers, num_hidden, configs)
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
        
        self.n_embd = self.model_args['n_embd'] # latent embedding
        self.att_embd = self.model_args['att_embd'] # attention embedding - determines size of attention matrix (should be divisible by n_embd)
        self.ntoken = self.preprocessor.latent_dims[-1]
        self.encode = self.model_args['n_latent_encode'] # latent encoding size (should be large enough to accomodate transfer to attention embedding)
        
        self.steps = self.model_args['diffusion_steps'] # number of diffusion steps
        i = torch.arange(self.steps, dtype=torch.float32)
        self.betas = (math.sqrt(0.00085) * (1-i) + math.sqrt(0.012) * i) ** 2 # stable diffusion schedule
        self.betas, self.alphas, self.alphas_bar = self.enforce_zero_terminal_snr(self.betas)
        self.rng = np.random.default_rng()
        # 
        self.encoder0 = nn.Linear(self.ntoken, self.encode)
        self.encoder1 = nn.Linear(self.encode, self.n_embd * self.att_embd)
        self.decoder1 = nn.Linear(self.n_embd * self.att_embd, self.encode)
        self.decoder0 = nn.Linear(self.encode, self.ntoken)
        self.init_weights()
        
        self.model = UNet( \
                         ninp=self.n_embd,
                         nhead=self.model_args['n_head'],
                         nhid=self.model_args['n_ffn_embd'],
                         nlayers=self.model_args['n_layers'],
                         dropout=self.model_args['dropout'],
                         initialization=self.model_args['initialization'],
                         activation=self.model_args['activation'])
        
        # transformer
        # B S E: batch, sequence, embedding (latent)
        
    def edit_config(self,configs):
        if configs.patch_size != 1:
            print(f"Warning: patch_size is {configs.patch_size} but should be 1 for TF model. Setting to 1.")
            configs.patch_size = 1
        return configs
        
        
    # https://arxiv.org/pdf/2305.08891.pdf
    def enforce_zero_terminal_snr(self,betas):
        # Convert betas to alphas_bar_sqrt
        alphas=1-betas
        alphas_bar=alphas.cumprod(0)
        alphas_bar_sqrt=alphas_bar.sqrt()
        # Store old values.
        alphas_bar_sqrt_0=alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T=alphas_bar_sqrt[-1].clone()
        # Shift so last timestep is zero.
        alphas_bar_sqrt-=alphas_bar_sqrt_T
        # Scale so first time step is back to old value.
        alphas_bar_sqrt*=alphas_bar_sqrt_0/(alphas_bar_sqrt_0-alphas_bar_sqrt_T)
        # Convert alphas_bar_sqrt to betas
        alphas_bar=alphas_bar_sqrt**2
        alphas=alphas_bar[1:]/alphas_bar[:-1]
        alphas=torch.cat([alphas_bar[0:1],alphas])
        betas=1-alphas
        return betas, alphas, alphas_bar
        
        
    def ddpm_train(self, noise, query, target):
        # select a_bar
        index = self.rng.integers(low=0, high=self.steps)
        a_bar = self.alphas_bar[index]
        # calculate loss
        src = torch.sqrt(a_bar) * target + torch.sqrt(1-a_bar) * noise
        noise_pred = self.model.forward(src, query)
        loss = torch.nn.functional.mse_loss(noise, noise_pred)
        
        return loss
        
    # def ddpm_eval(query, src):
    #     outpt = self.decoder(outpt_encoded)
        
    #     outpt = torch.cat((inpt,outpt),dim=1)
        
    #     out = self.preprocessor.batched_output_transform(outpt)
            
    #     loss_pred = loss_mixed(out, seq_total, self.input_length)
    #     loss_decouple = torch.tensor(0.0)
        
    def core_forward(self, seq_total, istrain=True):
        total = self.preprocessor.batched_input_transform(seq_total) # batch, seq, latent_input        
       
        del_frames_flat = total[:,1:,:] - total[:,:-1,:] # now batch, seq-1, latent_input
        query_frames_flat = total[:,:-1,:] # now batch, seq-1, latent_input
               
        # encode it 
        del_frames_flat = self.encoder(del_frames_flat) # now batch, seq-1, att*embd
        query_frames_flat = self.encoder(query_frames_flat) # now batch, seq-1, att*embd
        del_frames = del_frames_flat.reshape(del_frames_flat.size(0),del_frames_flat.size(1), self.att_embd, self.n_embd) # now batch, seq-1, att, embd
        query_frames = query_frames_flat.reshape(query_frames_flat.size(0),query_frames_flat.size(1), self.att_embd, self.n_embd) # now batch, seq-1, att, embd

        if istrain:
            loss_pred = torch.tensor(0.0, device=self.device)
            for i in range(self.total_length-1):   
                
                query_frame = query_frames[:,i]
                
                e_frame = torch.randn_like(query_frame, device=self.device) # X_T (the final noise frame)
                
                loss = self.ddpm_train(noise=e_frame, query=query_frame, target=del_frames[:,i])    
                loss_pred += loss / self.total_length-1
            
            loss_pred += torch.nn.functional.mse_loss(self.decoder(query_frames_flat), total[:,:-1,:]) # autoencoder loss
            
            out = self.preprocessor.batched_output_transform(total)

        else:
            loss_pred = torch.tensor(0.0, device=self.device)
            e_frame = torch.randn_like(query_frame, device=self.device) # X_T (the final noise frame)
            current_frame = query_frames[:,self.input_length-1]
            predictions = []
            for i in range(self.input_length-1, self.total_length):
                
                out_frame = self.ddpm_eval(last_frame=e_frame, query=current_frame)
                current_frame = out_frame + query_frame
                predictions.append(current_frame)
                
            out = torch.stack(predictions, dim=1)
            out = out.reshape(out.size(0), out.size(1), -1)
            out = self.decoder(out)
            out = self.preprocessor.batched_output_transform(out)
            out = torch.cat((seq_total[:,:self.input_length,:],out),dim=1)
            
        loss_decouple = torch.tensor(0.0)
        return loss_pred, loss_decouple, out

    def init_weights(self):
        initrange = math.sqrt(6 / self.n_embd + self.encode) #0.1
        nn.init.uniform_(self.encoder1.weight, -initrange, initrange)
        nn.init.uniform_(self.decoder1.weight, -initrange, initrange)
        initrange = math.sqrt(6 / (self.encode + self.ntoken))
        nn.init.uniform_(self.encoder0.weight, -initrange, initrange)
        nn.init.uniform_(self.decoder0.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder0.bias)
        nn.init.zeros_(self.encoder0.bias)
        nn.init.zeros_(self.decoder1.bias)
        nn.init.zeros_(self.encoder1.bias)
        
    def encoder(self, x):
        x = self.encoder0(x)
        x = torch.nn.functional.relu(x)
        x = self.encoder1(x)
        return x
        
    def decoder(self, x):
        x = self.decoder1(x)
        x = torch.nn.functional.relu(x)
        x = self.decoder0(x)
        return x
    
    
# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :,:]
        return self.dropout(x)
    
class UNet(nn.Module):
    """Unet model based on the Transformer architecture."""

    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.5, initialization=None, activation='relu'):
        super(UNet, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except BaseException as e:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or '
                              'lower.') from e
        self.model_type = 'Transformer'
        # self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.en_layers = nn.ModuleList([RZTXEncoderLayer(ninp, nhead, nhid, dropout, activation) for _ in range(nlayers)])
        self.de_layers = nn.ModuleList([RZTXEncoderLayer(ninp, nhead, nhid, dropout, activation, kdim=ninp*2, vdim=ninp*2) for _ in range(nlayers)])
        
        self.ninp = ninp

        self.initialization = initialization

        for i,tf_encoder_layer in enumerate(self.en_layers):
            self.init_FFN_weights(tf_encoder_layer,i)
        for i,tf_encoder_layer in enumerate(self.de_layers):
            self.init_FFN_weights(tf_encoder_layer,i)

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

    def forward(self, src, query):
    

        # src = self.encoder(src) * math.sqrt(self.ninp)
        # src = self.pos_encoder(src)

        en_stack = []

        for layer in self.en_layers:
            en_out = layer(query, src)
            en_stack.append(en_out)
            src = en_out
            
        de_out = query    
        for layer in self.de_layers:
            stacked_src = torch.cat([en_stack.pop(),de_out], dim=-1) # concat along embedding dim
            de_out = layer(query, stacked_src)
                
        return de_out
    

import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn import TransformerEncoder

class RZTXEncoderLayer(Module):
    r"""RZTXEncoderLayer is made up of self-attn and feedforward network with
    residual weights for faster convergece.
    This encoder layer is based on the paper "ReZero is All You Need:
    Fast Convergence at Large Depth".
    Thomas Bachlechner∗, Bodhisattwa Prasad Majumder∗, Huanru Henry Mao∗,
    Garrison W. Cottrell, Julian McAuley. 2020.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        use_res_init: Use residual initialization
    Examples::
        >>> encoder_layer = LDMEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', kdim=None, vdim=None):
        super().__init__()
        
        if kdim is None:
            kdim = d_model
        if vdim is None:
            vdim = d_model

        self.cross_attn = MultiheadAttention(d_model, nhead, dropout=dropout, kdim=kdim, vdim=vdim)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.resweight = nn.Parameter(torch.Tensor([0]))

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, query, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        ## type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in PyTorch Transformer class.
        """
        # cross-attention layer
        src2 = src
        src2 = self.cross_attn(query, src2, src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src2 = src2[0] # no attention weights
        src2 = src2 * self.resweight
        att_out = src[...,:src2.size(-1)] + self.dropout1(src2) # workaround for decoder




        # Pointiwse FF Layer
        src = att_out
        src2 = att_out       
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src2 = src2 * self.resweight
        src = src + self.dropout2(src2)
        return src