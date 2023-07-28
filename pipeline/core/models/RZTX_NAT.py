import torch
import math
import torch.nn as nn
from core.models.model_base import BaseModel
from core.loss import loss_mixed

class RZTX_NAT(BaseModel):
    # copies a lot of code from https://github.com/pytorch/examples/blob/main/word_language_model/model.py
    
    def __init__(self, num_layers, num_hidden, configs):
        super(RZTX_NAT, self).__init__(num_layers, num_hidden, configs)
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
        
        shapex = self.preprocessor.patch_x
        shapey = self.preprocessor.patch_y
        
        ntoken = self.preprocessor.latent_dims[-1] * shapex * shapey
        ninp = self.model_args['n_embd'] 
        
        spatial_preprocessor = 'flags' in self.preprocessor.__dict__ and 'spatial' in self.preprocessor.flags
        if spatial_preprocessor:
            reduced_shape = self.preprocessor.reduced_shape
            channels = ninp // (reduced_shape[1]*reduced_shape[2])
            assert ninp % (reduced_shape[1]*reduced_shape[2]) == 0, "ninp must be divisible by reduced_shape[1]*reduced_shape[2]"
        else:
            reduced_shape = None
            channels = ninp
                
        self.model = ReZero_base( \
                         seq_len = self.input_length,
                         ntoken=ntoken,
                         channels = channels,
                         ninp=ninp,
                         nhead=self.model_args['n_head'],
                         nhid=self.model_args['n_ffn_embd'],
                         nlayers=self.model_args['n_layers'],
                         dropout=self.model_args['dropout'],
                         initialization=self.model_args['initialization'],
                         activation=self.model_args['activation'],
                         reduced_shape = reduced_shape, device=self.device).to(self.device)
        
        # transformer
        # B S E: batch, sequence, embedding (latent)
        
    def edit_config(self,configs):
        if configs.patch_size != 1:
            print(f"Warning: patch_size is {configs.patch_size} but should be 1 for TF model. Setting to 1.")
            configs.patch_size = 1
        return configs
        
    def core_forward(self, seq_total, istrain=True, **kwargs):
        total = self.preprocessor.batched_input_transform(seq_total)
        inpt = total[:,:self.input_length,:]
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
        out = torch.cat((total[:,:self.input_length,:],outpt),dim=1)  
        out = self.preprocessor.batched_output_transform(out)
                    
        loss_pred = loss_mixed(out, seq_total, self.input_length)
        loss_decouple = torch.tensor(0.0)

        return loss_pred, loss_decouple, out

    
        
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

        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
class NAT(nn.Module):
    def __init__(self, d, num_heads, k1, k2, channels, spatial, device) -> None:
        super().__init__()
        assert k1%2==1 and k2%2==1, "Kernel sizes must be odd!"
        # standard conv layer, then atrous conv layer such that the entire region is covered
        ###
        # Modules expect inputs of shape [batch_size, *, dim]:
        # NA1D: [batch_size, sequence_length, dim]
        # NA2D: [batch_size, height, width, dim]
        # NA3D: [batch_size, depth, height, width, dim]
        if spatial and channels > 1:
            from natten import NeighborhoodAttention3D
            # self.conv1 = nn.Conv2d(channels, channels, (k1, k1))
            # self.conv2 = nn.Conv2d(channels, channels, (k2, k2), dilation=(2,2))

            # in is [batch_size, dim, channels, height, width]
            
            self.rperm1 = lambda x : x.permute(0,4,1,2,3)
            self.perm1 = lambda x : x.permute(0,2,3,4,1)

            self.conv1 = NeighborhoodAttention3D(dim=d, kernel_size=k1,dilation=1, kernel_size_d=channels, dilation_d=1, num_heads=num_heads).to(device) 
            self.conv2 = NeighborhoodAttention3D(dim=d, kernel_size=k2, dilation=2, kernel_size_d=channels, dilation_d=1, num_heads=num_heads).to(device)
            
            self.combineA = lambda x : self.conv1(self.perm1(x))
            self.combineB = lambda x : self.rperm1(self.conv2(x))
        elif spatial and channels == 1:
            from natten import NeighborhoodAttention2D
            # self.conv1 = nn.Conv2d(channels, channels, (k1, k1))
            # self.conv2 = nn.Conv2d(channels, channels, (k2, k2), dilation=(2,2))

            # in is [batch_size, height, width, dim]
            
            self.perm1 = lambda x : x.squeeze(2).permute(0,2,3,1)
            self.rperm1 = lambda x : x.permute(0,3,1,2).unsqueeze(2)

            self.conv1 = NeighborhoodAttention2D(dim=d, kernel_size=k1, dilation=1, num_heads=num_heads).to(device) 
            self.conv2 = NeighborhoodAttention2D(dim=d, kernel_size=k2, dilation=2, num_heads=num_heads).to(device)
            
            self.combineA = lambda x : self.conv1(self.perm1(x))
            self.combineB = lambda x : self.rperm1(self.conv2(x))
            
        else:
            from natten import NeighborhoodAttention1D
            # self.conv1 = nn.Conv1d(channels, channels, k1, padding=k1//2, padding_mode='zeros')
            # self.conv2 = nn.Conv1d(channels, channels, k2, padding='same', padding_mode='circular', dilation=8)
            self.perm1 = lambda x : x.permute(0,2,1)
            self.rperm1 = lambda x : x.permute(0,2,1)
            self.conv1 = NeighborhoodAttention1D(dim=d, kernel_size=k1, dilation=1, num_heads=num_heads).to(device)
            self.conv2 = NeighborhoodAttention1D(dim=d, kernel_size=k2, dilation=2, num_heads=num_heads).to(device)   
            
            self.combineA = lambda x : self.conv1(self.perm1(x))
            self.combineB = lambda x : self.rperm1(self.conv2(x))
        self.seq = lambda x : self.combineB(self.combineA(x)) + x # residual connection     
            
    
class ReZero_base(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a fully-connected output layer."""

    def __init__(self, seq_len, ntoken, channels, ninp, nhead, nhid, nlayers, dropout=0.5, initialization=None, activation='relu', reduced_shape=None, device=None):
        super(ReZero_base, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except BaseException as e:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or '
                              'lower.') from e
        self.model_type = 'Transformer'
        self.src_mask = None
        # self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = RZTXEncoderLayer(seq_len, ninp, nhead, nhid, dropout, activation, batch_first=True, channels=channels, reduced_shape=reduced_shape, device=device) #batch_first=False 
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp
        self.ntoken = ntoken
        self.initialization = initialization
        
        if reduced_shape is None:
            self.encoder_c = nn.Linear(ntoken, ninp)
            self.decoder_c = nn.Linear(ninp, ntoken)
            self.init_weights(self.encoder_c,ntoken,ninp)
            self.init_weights(self.decoder_c,ninp,ntoken)
            self.encoder = self.encoder1D
            self.decoder = self.decoder1D
        else:
            self.encoder = lambda x: x
            self.decoder = lambda x: x

        # self.init_weights()
        # for i,tf_encoder_layer in enumerate(self.transformer_encoder.layers):
        #     self.init_FFN_weights(tf_encoder_layer,i)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float('-inf'))
            .masked_fill(mask == 1, 0.0)
        )
        return mask

    def init_weights(self,a,b,c):
        initrange = math.sqrt(6 / (b+c)) #0.1
        nn.init.uniform_(a.weight, -initrange, initrange)
        nn.init.zeros_(a.bias)

    # def init_FFN_weights(self,tf_encoder_layer, layer_num=0):
    #     # initialize the weights of the feed-forward network (assuming RELU)
    #     # TODO need to add option if using sine activation
    #     if self.initialization not in [None,[]]:
    #         self.initialization(tf_encoder_layer.linear1.weight, layer_num)
    #         self.initialization(tf_encoder_layer.linear2.weight, layer_num)
    #     else:
    #         initrange = math.sqrt(3 / self.ninp)
    #         nn.init.uniform_(tf_encoder_layer.linear1.weight, -initrange, initrange)
    #         nn.init.uniform_(tf_encoder_layer.linear2.weight, -initrange, initrange)
    #     nn.init.zeros_(tf_encoder_layer.linear1.bias)
    #     nn.init.zeros_(tf_encoder_layer.linear2.bias)
        
    def encoder1D(self, inpt):
        # self.batch, self.seqlen, self.nc  = inpt.shape
        # inpt = inpt.reshape(inpt.shape[0],inpt.shape[1],-1)
        return self.encoder_c(inpt)
        
    def decoder1D(self, outpt):
        outpt = self.decoder_c(outpt)
        # outpt = outpt.reshape(self.batch, self.seqlen,self.nc )
        return outpt    

    def forward(self, src, has_mask=True):
        # if has_mask:
        #     device = src.device
        #     if self.src_mask is None or self.src_mask.size(0) != len(src):
        #         mask = self._generate_square_subsequent_mask(len(src)).to(device)
        #         self.src_mask = mask
        # else:
        #     self.src_mask = None

        src = self.encoder(src)# * math.sqrt(self.ninp)
        output = self.transformer_encoder(src)#, self.src_mask)
        output = self.decoder(output)
        return output
    

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
        >>> encoder_layer = RZTXEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """
    def __init__(self, seq_len, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', batch_first=True, channels=1, reduced_shape=None, device=None):
        super().__init__()

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        # Implementation of Feedforward model
        # self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        # self.linear2 = Linear(dim_feedforward, d_model)
        
        spatial = reduced_shape is not None        
        self.reduced_shape = reduced_shape
        if not spatial: channels = 1
        self.conv = NAT(seq_len, nhead, 9,9,channels,spatial=spatial, device=device)
        
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
        # Self attention layer
        src2 = src
        src2 = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src2 = src2[0] # no attention weights
        src2 = src2 * self.resweight
        src = src + self.dropout1(src2)

        # Pointiwse FF Layer
        src2 = src            
        
        if self.reduced_shape is not None:
            shape = (src2.shape[0], src2.shape[1], self.reduced_shape[0], self.reduced_shape[1], self.reduced_shape[2])
            # print(shape)
            src3 = src2.reshape(shape)
            src4 = self.conv.seq(src3)
            src5 = src4.reshape(src2.shape)
        else:
            shape = (src2.shape[0], src2.shape[1], src2.shape[2])
            src3 = src2.reshape(shape)
            src4 = self.conv.seq(src3)
            src5 = src4.reshape(src2.shape)
            
        src2 = self.dropout(src2) + src5
        
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src2 = src2 * self.resweight
        src = src + self.dropout2(src2)
        return src