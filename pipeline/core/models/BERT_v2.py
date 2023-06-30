import torch
import math
import torch.nn as nn
from core.models.model_base import BaseModel
from core.loss import loss_mixed

# same as BERT, but removed positional encoding.

class BERT(BaseModel):
    # copies a lot of code from https://github.com/pytorch/examples/blob/main/word_language_model/model.py
    
    def __init__(self, num_layers, num_hidden, configs):
        super(BERT, self).__init__(num_layers, num_hidden, configs)
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
        
        self.model = BERT_base( \
                         ntoken=self.preprocessor.latent_dims[-1] * shapex * shapey,
                         ninp=self.model_args['n_embd'],
                         nhead=self.model_args['n_head'],
                         nhid=self.model_args['n_ffn_embd'],
                         nlayers=self.model_args['n_layers'],
                         dropout=self.model_args['dropout'],
                         initialization=self.model_args['initialization'],
                         activation=self.model_args['activation'], device=self.device).to(self.device)
        
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
    
class BERT_base(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a fully-connected output layer."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5, initialization=None, activation='relu', device=None):
        super(BERT_base, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except BaseException as e:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or '
                              'lower.') from e
            
        self.device = device
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout, activation, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Linear(ntoken, ninp).to(self.device)
        self.ninp = ninp
        self.ntoken = ntoken
        self.decoder = nn.Linear(ninp, ntoken).to(self.device)
        self.initialization = initialization

        self.init_weights()
        for i,tf_encoder_layer in enumerate(self.transformer_encoder.layers):
            self.init_FFN_weights(tf_encoder_layer,i)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float('-inf'))
            .masked_fill(mask == 1, 0.0)
        )
        return mask

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

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.encoder(src) * math.sqrt(self.ninp)
        # src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output