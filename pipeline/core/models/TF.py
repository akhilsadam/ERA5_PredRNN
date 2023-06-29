import torch
import math
import torch.nn as nn
from core.models.model_base import BaseModel
from core.loss import loss_mixed

 #TODO UPDATE THIS to work with new input/output transforms
 # obsolete code below - also not a good model overall.

class TF(BaseModel):
    # copies a lot of code from https://github.com/pytorch/examples/blob/main/word_language_model/model.py
    
    def __init__(self, num_layers, num_hidden, configs):
        super(TF, self).__init__(num_layers, num_hidden, configs)
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
        self.model = TimeSeriesTransformer(
                    input_size=configs.preprocessor.latent_dims[-1],
                    batch_first=True,
                    dim_val=self.model_args['n_embd'] if 'n_embd' in self.model_args else configs.preprocessor.latent_dims[-1],  
                    n_encoder_layers=self.model_args['n_encoder_layers'],
                    n_decoder_layers=self.model_args['n_decoder_layers'],
                    n_heads=self.model_args['n_head'],
                    dropout_encoder=self.model_args['dropout'], 
                    dropout_decoder=self.model_args['dropout'],
                    dropout_pos_enc=self.model_args['dropout_pos_enc'],
                    dim_feedforward_encoder=self.model_args['n_ffn_embd'],
                    dim_feedforward_decoder=self.model_args['n_ffn_embd'],
                    num_predicted_features=configs.preprocessor.latent_dims[-1]
        )
        
        # self.model = TF_base( \
        #                  ninp0=configs.preprocessor.latent_dims[-1],
        #                  ninp=self.model_args['n_embd'] if 'n_embd' in self.model_args else configs.preprocessor.latent_dims[-1],
        #                  nhead=self.model_args['n_head'],
        #                  nhid=self.model_args['n_ffn_embd'],
        #                  nlayers=self.model_args['n_layers'],
        #                  dropout=self.model_args['dropout'],
        #                  initialization=self.model_args['initialization'],
        #                  activation=self.model_args['activation'])
        
    def edit_config(self,configs):
        if configs.patch_size != 1:
            print(f"Warning: patch_size is {configs.patch_size} but should be 1 for TF model. Setting to 1.")
            configs.patch_size = 1
        return configs
        
    def core_forward(self, seq_total, istrain=True):
        
        inl = self.configs.input_length
        test = self.preprocessor.batched_input_transform(seq_total)
        inpt = test[:,:inl,:]
        # loss_pred = 0.0
            
        # print("INPUTSIZE", inpt.size())
            
        # outpt = self.model(inpt)
        # outpt = torch.cat((inpt,outpt),dim=1)
        
        src = inpt
        tgt = inpt[:, -1, :].unsqueeze(1)
        
        for i in range(self.predict_length):
            
            
            
            
            # Create masks
            dim_a = tgt.shape[1]

            dim_b = src.shape[1]

            tgt_mask = generate_square_subsequent_mask(
                dim1=dim_a,
                dim2=dim_a,
                device=self.device
                )

            src_mask = generate_square_subsequent_mask(
                dim1=dim_a,
                dim2=dim_b,
                device=self.device
                )

            # Make prediction
            prediction = self.model(src, tgt, src_mask, tgt_mask) 
            if i == self.predict_length -1:
                break            
            
            last_predicted_value = prediction[:, -1, :].unsqueeze(1)
            # print(f"last_predicted_value shape: {last_predicted_value.size()}")
            # Reshape from [batch_size, nlatent] --> [batch_size, 1, nlatent]

            # Detach the predicted element from the graph and concatenate with 
            # tgt in dimension 1 or 0
            
            # loss_pred += torch.nn.functional.mse_loss(last_predicted_value, test[:,inl+i,:].unsqueeze(1))
            # print(f"tgt shape: {tgt.size()}")
            if istrain:
                # teacher forcing
                tgt = torch.cat((tgt, test[:,inl+i+1,:].unsqueeze(1).requires_grad_(True)), 1)
            else:
                # self-generated reasoning chain
                tgt = torch.cat((tgt, last_predicted_value.detach().requires_grad_(True)), 1)
        
        # print("OUTPUTSIZE", outpt.size())

        out = self.preprocessor.batched_output_transform(prediction)
            
        # loss_pred = torch.nn.functional.mse_loss(out[:,self.configs.input_length:], seq_total[:,self.configs.input_length:])
        loss_decouple = torch.tensor(0.0)
        out = torch.concat([seq_total[:,:self.configs.input_length,:],out],dim=1)
        
        loss_pred = loss_mixed(out, seq_total, self.input_length)

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
        # print(f"position: {position.size()}")
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # print(f"pe: {pe.size()}")
        pe[:, 0::2] = torch.sin(position * div_term)
        # print(f"cos term :{torch.cos(position * div_term).size()}")
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) #.transpose(0, 1)
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
    
# fom KasperGroesLudvigsen's github

def generate_square_subsequent_mask(dim1: int, dim2: int, device) -> torch.Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    Modified from: 
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    Args:

        dim1: int, for both src and tgt masking, this must be target sequence
              length

        dim2: int, for src masking this must be encoder sequence length (i.e. 
              the length of the input sequence to the model), 
              and for tgt masking, this must be target sequence length 


    Return:

        A Tensor of shape [dim1, dim2]
    """
    return torch.triu(torch.ones(dim1, dim2, device=device) * float('-inf'), diagonal=1)


class TimeSeriesTransformer(nn.Module):

    """
    This class implements a transformer model that can be used for times series
    forecasting. This time series transformer model is based on the paper by
    Wu et al (2020) [1]. The paper will be referred to as "the paper".

    A detailed description of the code can be found in my article here:

    https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e

    In cases where the paper does not specify what value was used for a specific
    configuration/hyperparameter, this class uses the values from Vaswani et al
    (2017) [2] or from PyTorch source code.

    Unlike the paper, this class assumes that input layers, positional encoding 
    layers and linear mapping layers are separate from the encoder and decoder, 
    i.e. the encoder and decoder only do what is depicted as their sub-layers 
    in the paper. For practical purposes, this assumption does not make a 
    difference - it merely means that the linear and positional encoding layers
    are implemented inside the present class and not inside the 
    Encoder() and Decoder() classes.

    [1] Wu, N., Green, B., Ben, X., O'banion, S. (2020). 
    'Deep Transformer Models for Time Series Forecasting: 
    The Influenza Prevalence Case'. 
    arXiv:2001.08317 [cs, stat] [Preprint]. 
    Available at: http://arxiv.org/abs/2001.08317 (Accessed: 9 March 2022).

    [2] Vaswani, A. et al. (2017) 
    'Attention Is All You Need'.
    arXiv:1706.03762 [cs] [Preprint]. 
    Available at: http://arxiv.org/abs/1706.03762 (Accessed: 9 March 2022).

    """

    def __init__(self, 
        input_size: int,
        batch_first: bool,
        dim_val: int=512,  
        n_encoder_layers: int=4,
        n_decoder_layers: int=4,
        n_heads: int=8,
        dropout_encoder: float=0.2, 
        dropout_decoder: float=0.2,
        dropout_pos_enc: float=0.1,
        dim_feedforward_encoder: int=2048,
        dim_feedforward_decoder: int=2048,
        num_predicted_features: int=1
        ): 

        """
        Args:

            input_size: int, number of input variables. 1 if univariate.

            dec_seq_len: int, the length of the input sequence fed to the decoder

            dim_val: int, aka d_model. All sub-layers in the model produce 
                     outputs of dimension dim_val

            n_encoder_layers: int, number of stacked encoder layers in the encoder

            n_decoder_layers: int, number of stacked encoder layers in the decoder

            n_heads: int, the number of attention heads (aka parallel attention layers)

            dropout_encoder: float, the dropout rate of the encoder

            dropout_decoder: float, the dropout rate of the decoder

            dropout_pos_enc: float, the dropout rate of the positional encoder

            dim_feedforward_encoder: int, number of neurons in the linear layer 
                                     of the encoder

            dim_feedforward_decoder: int, number of neurons in the linear layer 
                                     of the decoder

            num_predicted_features: int, the number of features you want to predict.
                                    Most of the time, this will be 1 because we're
                                    only forecasting FCR-N prices in DK2, but in
                                    we wanted to also predict FCR-D with the same
                                    model, num_predicted_features should be 2.
        """

        super().__init__() 

        #print("input_size is: {}".format(input_size))
        #print("dim_val is: {}".format(dim_val))

        # Creating the three linear layers needed for the model
        self.encoder_input_layer = nn.Linear(
            in_features=input_size, 
            out_features=dim_val 
            )

        self.decoder_input_layer = nn.Linear(
            in_features=num_predicted_features,
            out_features=dim_val
            )  
        
        self.linear_mapping = nn.Linear(
            in_features=dim_val, 
            out_features=num_predicted_features
            )
        
        initrange = math.sqrt(3 / (input_size)) # GLOROT for ReLU
        nn.init.uniform_(self.encoder_input_layer.weight, -initrange, initrange)
        nn.init.uniform_(self.decoder_input_layer.weight, -initrange, initrange)
        initrange = math.sqrt(6 / (dim_val + num_predicted_features)) # GLOROT for ReLU
        nn.init.uniform_(self.linear_mapping.weight, -initrange, initrange)


        # Create positional encoder
        self.positional_encoding_layer = PositionalEncoding(
            d_model=dim_val,
            dropout=dropout_pos_enc
            )
 
        # The encoder layer used in the paper is identical to the one used by
        # Vaswani et al (2017) on which the PyTorch module is based.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val, 
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            batch_first=batch_first
            )

        # Stack the encoder layers in nn.TransformerDecoder
        # It seems the option of passing a normalization instance is redundant
        # in my case, because nn.TransformerEncoderLayer per default normalizes
        # after each sub-layer
        # (https://github.com/pytorch/pytorch/issues/24930).
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers, 
            norm=None
            )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_decoder,
            dropout=dropout_decoder,
            batch_first=batch_first
            )

        # Stack the decoder layers in nn.TransformerDecoder
        # It seems the option of passing a normalization instance is redundant
        # in my case, because nn.TransformerDecoderLayer per default normalizes
        # after each sub-layer
        # (https://github.com/pytorch/pytorch/issues/24930).
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_decoder_layers, 
            norm=None
            )
        
    def init_FFN_weights(self,tf_encoder_layer, layer_num=0):
        # initialize the weights of the feed-forward network (assuming RELU)
        # TODO need to add option if using sine activation
        if self.initialization not in [None,[]]:
            self.initialization(tf_encoder_layer.linear1.weight, layer_num)
            self.initialization(tf_encoder_layer.linear2.weight, layer_num)
        else:
            initrange = math.sqrt(3 / (self.dim_val)) # GLOROT for ReLU
            nn.init.uniform_(tf_encoder_layer.linear1.weight, -initrange, initrange)
            nn.init.uniform_(tf_encoder_layer.linear2.weight, -initrange, initrange)
        nn.init.zeros_(tf_encoder_layer.linear1.bias)
        nn.init.zeros_(tf_encoder_layer.linear2.bias)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor=None, 
                tgt_mask: torch.Tensor=None) -> torch.Tensor:
        """
        Returns a tensor of shape:

        [target_sequence_length, batch_size, num_predicted_features]
        
        Args:

            src: the encoder's output sequence. Shape: (S,E) for unbatched input, 
                 (S, N, E) if batch_first=False or (N, S, E) if 
                 batch_first=True, where S is the source sequence length, 
                 N is the batch size, and E is the number of features (1 if univariate)

            tgt: the sequence to the decoder. Shape: (T,E) for unbatched input, 
                 (T, N, E)(T,N,E) if batch_first=False or (N, T, E) if 
                 batch_first=True, where T is the target sequence length, 
                 N is the batch size, and E is the number of features (1 if univariate)

            src_mask: the mask for the src sequence to prevent the model from 
                      using data points from the target sequence

            tgt_mask: the mask for the tgt sequence to prevent the model from
                      using data points from the target sequence


        """

        #print("From model.forward(): Size of src as given to forward(): {}".format(src.size()))
        #print("From model.forward(): tgt size = {}".format(tgt.size()))

        # Pass throguh the input layer right before the encoder
        src = self.encoder_input_layer(src) # src shape: [batch_size, src length, dim_val] regardless of number of input features
        #print("From model.forward(): Size of src after input layer: {}".format(src.size()))

        # Pass through the positional encoding layer
        src = self.positional_encoding_layer(src) # src shape: [batch_size, src length, dim_val] regardless of number of input features
        #print("From model.forward(): Size of src after pos_enc layer: {}".format(src.size()))

        # Pass through all the stacked encoder layers in the encoder
        # Masking is only needed in the encoder if input sequences are padded
        # which they are not in this time series use case, because all my
        # input sequences are naturally of the same length. 
        # (https://github.com/huggingface/transformers/issues/4083)
        src = self.encoder( # src shape: [batch_size, enc_seq_len, dim_val]
            src=src
            )
        #print("From model.forward(): Size of src after encoder: {}".format(src.size()))

        # Pass decoder input through decoder input layer
        decoder_output = self.decoder_input_layer(tgt) # src shape: [target sequence length, batch_size, dim_val] regardless of number of input features
        #print("From model.forward(): Size of decoder_output after linear decoder layer: {}".format(decoder_output.size()))

        #if src_mask is not None:
            #print("From model.forward(): Size of src_mask: {}".format(src_mask.size()))
        #if tgt_mask is not None:
            #print("From model.forward(): Size of tgt_mask: {}".format(tgt_mask.size()))

        # Pass throguh decoder - output shape: [batch_size, target seq len, dim_val]
        decoder_output = self.decoder(
            tgt=decoder_output,
            memory=src,
            tgt_mask=tgt_mask,
            memory_mask=src_mask
            )

        #print("From model.forward(): decoder_output shape after decoder: {}".format(decoder_output.shape))

        # Pass through linear mapping
        decoder_output = self.linear_mapping(decoder_output) # shape [batch_size, target seq len]
        #print("From model.forward(): decoder_output size after linear_mapping = {}".format(decoder_output.size()))

        return decoder_output
    
# class TF_base(nn.Module):
#     """Container module with an encoder, a recurrent or transformer module, and a fully-connected output layer."""

#     def __init__(self, ninp0, ninp, nhead, nhid, nlayers, dropout=0.5, initialization=None, activation='relu'):
#         super(TF_base, self).__init__()
#         try:
#             from torch.nn import TransformerEncoder, TransformerEncoderLayer
#         except BaseException as e:
#             raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or '
#                               'lower.') from e
#         self.model_type = 'Transformer'
#         self.src_mask = None
#         self.pos_encoder = PositionalEncoding(ninp, dropout)
#         encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout, activation, batch_first=True)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
#         self.encoder = nn.Linear(ninp0,ninp)#nn.Embedding(ninp0, ninp)
#         self.ninp = ninp
#         self.decoder = nn.Linear(ninp, ninp0)
#         self.initialization = initialization

#         self.init_weights()
#         for i,tf_encoder_layer in enumerate(self.transformer_encoder.layers):
#             self.init_FFN_weights(tf_encoder_layer,i)

#     def _generate_square_subsequent_mask(self, sz):
#         mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
#         mask = (
#             mask.float()
#             .masked_fill(mask == 0, float('-inf'))
#             .masked_fill(mask == 1, 0.0)
#         )
#         return mask

#     def init_weights(self):
#         initrange = 0.1
#         nn.init.uniform_(self.encoder.weight, -initrange, initrange)
#         nn.init.zeros_(self.decoder.bias)
#         nn.init.uniform_(self.decoder.weight, -initrange, initrange)

#     def init_FFN_weights(self,tf_encoder_layer, layer_num=0):
#         # initialize the weights of the feed-forward network (assuming RELU)
#         # TODO need to add option if using sine activation
#         if self.initialization not in [None,[]]:
#             self.initialization(tf_encoder_layer.linear1.weight, layer_num)
#             self.initialization(tf_encoder_layer.linear2.weight, layer_num)
#         else:
#             initrange = math.sqrt(3 / self.ninp)
#             nn.init.uniform_(tf_encoder_layer.linear1.weight, -initrange, initrange)
#             nn.init.uniform_(tf_encoder_layer.linear2.weight, -initrange, initrange)
#         nn.init.zeros_(tf_encoder_layer.linear1.bias)
#         nn.init.zeros_(tf_encoder_layer.linear2.bias)

#     def forward(self, src, has_mask=True):
#         if has_mask:
#             device = src.device
#             if self.src_mask is None or self.src_mask.size(0) != len(src):
#                 mask = self._generate_square_subsequent_mask(len(src)).to(device)
#                 self.src_mask = mask
#         else:
#             self.src_mask = None

#         # b, t, l = src.size()

#         src = self.encoder(src) * math.sqrt(self.ninp)
#         src = self.pos_encoder(src)
#         output = self.transformer_encoder(src, self.src_mask)
#         output = self.decoder(output)
#         return output
    