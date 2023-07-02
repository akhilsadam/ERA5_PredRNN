import torch
import math
import torch.nn as nn
from core.models.model_base import BaseModel
from core.loss import loss_mixed

# same as BERT, but removed positional encoding, and linear encoder/decoder layers (note none of these changes made a major difference)
# Now remaking without mean pooling - just directly learning all 20 next steps (since changing that did not make a difference).

# We probably need some spatial attention to make this work.
# So I'm adding a spatial self-attention layer directly prior to temporal self-attention.


class DAT(BaseModel):
    # copies a lot of code from https://github.com/pytorch/examples/blob/main/word_language_model/model.py
    
    def __init__(self, num_layers, num_hidden, configs):
        super(DAT, self).__init__(num_layers, num_hidden, configs)
        self.preprocessor = configs.preprocessor
        self.model_args = configs.model_args
        assert self.preprocessor is not None, "Preprocessor is None, please check config! Cannot operate on raw data."
        assert self.model_args is not None, "Model args is None, please check config!"
        assert configs.input_length == configs.total_length//2, "TF model requires input_length == total_length//2"
        assert configs.input_length > 0, "Model requires input_length"
        assert configs.total_length > configs.input_length, "Model requires total_length"
        
        # transformer
        # B S E: batch, sequence, embedding (latent)                
        self.preprocessor.load(device=configs.device)
        self.device = configs.device
        self.input_length = configs.input_length
        self.predict_length = configs.total_length - configs.input_length
        self.total_length = configs.total_length
        
        patch_x = self.preprocessor.patch_x
        patch_y = self.preprocessor.patch_y
        
        d_space_original = self.preprocessor.latent_dims[-1] * patch_x * patch_y
        d_space = self.model_args['n_embd']
        d_time_original = configs.input_length
        
        if d_space != d_space_original:
            print (f"Warning: n_embd is {d_space} but should be {d_space_original} for TF model. Setting to {d_space_original}.")
            d_space = d_space_original

        self.model = DualAttentionTransformer( \
                         d_time_original=d_time_original,
                         d_space_original=d_space_original,
                         d_space=d_space,
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
        
    def core_forward(self, seq_total, istrain=True, **kwargs):
        seq_in = seq_total[:,:self.input_length,:]
        inpt = self.preprocessor.batched_input_transform(seq_in)
        
        nc, sx, sy = inpt.shape[-3:]
        inpt = inpt.reshape(inpt.shape[0],inpt.shape[1],-1)
        
        predictions = []
        for i in range(self.predict_length):
            outpt = self.model(inpt)
            out = outpt.mean(dim=1)
            predictions.append(out.unsqueeze(1))
            inpt = torch.cat((inpt,out.unsqueeze(1)),dim=1)[:,-self.input_length:,:]
        
        outpt = torch.cat(predictions,dim=1)        
        outpt = outpt.reshape(outpt.shape[0],outpt.shape[1],nc,sx,sy)    
        out = self.preprocessor.batched_output_transform(outpt)
        out = torch.cat((seq_total[:,:self.input_length,:],out),dim=1)
                    
        loss_pred = loss_mixed(out, seq_total, self.input_length)
        loss_decouple = torch.tensor(0.0)
        return loss_pred, loss_decouple, out

    
        
# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_space))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_space))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_space: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_space)
    """

    def __init__(self, d_space, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_space)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_space, 2).float() * (-math.log(10000.0) / d_space))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)#.transpose(0, 1)
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
        # above doc is incorrect, x is [batch size, sequence length, embed dim]

        x = x + self.pe[:, :x.size(1), :] # batch is first dimension, sequence is second dimension
        return self.dropout(x)
    
class DualAttentionTransformer(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a fully-connected output layer."""

    def __init__(self, d_time_original, d_space_original, d_space, nhead, nhid, nlayers, dropout=0.5, initialization=None, activation='relu'):
        super(DualAttentionTransformer, self).__init__()

        self.model_type = 'Transformer'
        self.src_mask = None
        # self.pos_encoder = PositionalEncoding(d_space, dropout)
        self.layers = ModuleList([DualAttentionTransformerLayer(d_time_original, d_space, nhead, nhid, dropout, activation, batch_first=True)]*nlayers)
        # self.encoder = nn.Linear(d_space_original, d_space)
        self.d_space = d_space
        self.d_space_original = d_space_original
        # self.decoder = nn.Linear(d_space, d_space_original)
        self.initialization = initialization

        # self.init_weights()
        for i,tf_encoder_layer in enumerate(self.layers):
            self.init_FFN_weights(tf_encoder_layer,i)


    # def init_weights(self):
    #     initrange = math.sqrt(3 / self.d_space) #0.1
    #     nn.init.uniform_(self.encoder.weight, -initrange, initrange)
    #     nn.init.zeros_(self.decoder.bias)
    #     initrange = math.sqrt(6 / (self.d_space + self.d_space_original))
    #     nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def init_FFN_weights(self,tf_encoder_layer, layer_num=0):
        # initialize the weights of the feed-forward network (assuming RELU)
        # TODO need to add option if using sine activation
        if self.initialization not in [None,[]]:
            self.initialization(tf_encoder_layer.linear1.weight, layer_num)
            self.initialization(tf_encoder_layer.linear2.weight, layer_num)
        else:
            initrange = math.sqrt(3 / self.d_space)
            nn.init.uniform_(tf_encoder_layer.linear1.weight, -initrange, initrange)
            nn.init.uniform_(tf_encoder_layer.linear2.weight, -initrange, initrange)
        nn.init.zeros_(tf_encoder_layer.linear1.bias)
        nn.init.zeros_(tf_encoder_layer.linear2.bias)

    def forward(self, src, has_mask=True):
        # if has_mask:
        #     device = src.device
        #     if self.src_mask is None or self.src_mask.size(0) != len(src):
        #         mask = self._generate_square_subsequent_mask(len(src)).to(device)
        #         self.src_mask = mask
        # else:
        #     self.src_mask = None

        # src = self.encoder(src) * math.sqrt(self.d_space)
        # src = self.pos_encoder(src)
        # output = self.transformer_encoder(src)
        output = src
        for i,layer in enumerate(self.layers):
            output = output + layer(output)
        # output = self.decoder(output)
        return output
    
    
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
import torch.nn.functional as F
from torch.nn import LayerNorm

class DualAttentionTransformerLayer(Module):

    def __init__(self, d_time, d_space, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', kdim_time=None, vdim_time=None, kdim_space=None, vdim_space=None, batch_first=True, **factory_kwargs):
        super().__init__()
        
        if kdim_time is None:
            kdim_time = d_time
        if vdim_time is None:
            vdim_time = d_time
        if kdim_space is None:
            kdim_space = d_space
        if vdim_space is None:
            vdim_space = d_space
            
        self.batch_first = batch_first
            
        #    batch first ->  B S E: batch, sequence, spatial embedding (latent)
        #not batch first -> S B E: sequence, batch, spatial embedding (latent)
        # - either way spatial embedding is last dimension

        layer_norm_eps = 1e-5 
        self.attn_space = MultiheadAttention(d_time, nhead, dropout=dropout, kdim=kdim_time, vdim=vdim_time, batch_first=False, **factory_kwargs)
        self.attn_time = MultiheadAttention(d_space, nhead, dropout=dropout, kdim=kdim_space, vdim=vdim_space, batch_first=batch_first, **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_space, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_space)
        self.norm0 = LayerNorm(d_time, eps=layer_norm_eps, **factory_kwargs)
        self.norm1 = LayerNorm(d_space, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_space, eps=layer_norm_eps, **factory_kwargs)
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

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        ## type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in PyTorch Transformer class.
        """
        ##########
        # self-attention layer in space
        ##########
        # permute
        if self.batch_first:
            src2 = src.permute(2,0,1) # get spatial dimension first, so we have Spatial, Batch, Temporal
        else:
            src2 = src.permute(2,1,0) # get spatial dimension first, so we have Spatial, Batch, Temporal
        att_out = src2 + self.dropout1(self.attn_space(src2, src2, src2)[0])
        # att_out = src[...,:src2.size(-1)] + self.dropout1(src2) # workaround for decoder
        # normalization
        att_out = self.norm0(att_out)
        # permute back
        if self.batch_first:
            spatial_out = att_out.permute(1,2,0) # get batch dimension first, so we have Batch, Temporal, Spatial
        else:   
            spatial_out = att_out.permute(2,1,0) # get temporal dimension first, so we have Temporal, Batch, Spatial
        ##########
        # self-attention layer in time
        ##########
        src2 = spatial_out
        src2 = self.attn_time(src2, src2, src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src2 = src2[0] # no attention weights
        att_out = spatial_out + self.dropout1(src2)
        # temporal_out = src[...,:src2.size(-1)] + self.dropout1(src2) # workaround for decode
        # normalization
        temporal_out = self.norm1(att_out)
        ##########
        # Pointwise FF Layer
        ##########
        src2 = temporal_out       
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        # src2 = src2 * self.resweight
        src = temporal_out + self.dropout2(src2)
        
        # normalization again
        norm_out_2 = self.norm2(src) * self.resweight
        ##########
        return norm_out_2