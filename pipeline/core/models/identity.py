import math
import torch
import torch.nn as nn
from core.models.model_base import BaseModel
from core.loss import loss_mixed
  
class Identity(BaseModel):
    # copies a lot of code from https://github.com/pytorch/examples/blob/main/word_language_model/model.py
    
    def __init__(self, num_layers, num_hidden, configs):
        # Inheritance
        super(Identity, self).__init__(num_layers, num_hidden, configs)
        self.preprocessor = configs.preprocessor
        self.model_args = configs.model_args

        # Error handling
        assert self.preprocessor is not None, "Preprocessor is None, please check config! Cannot operate on raw data."
        assert self.model_args is not None, "Model args is None, please check config!"
        assert configs.input_length > 0, "Model requires input_length"
        assert configs.total_length > configs.input_length, "Model requires total_length"
        
        # transformer
        # B S E: batch, sequence, embedding (latent)
        self.preprocessor.load(device=configs.device)
        self.device = configs.device
        self.input_length = configs.input_length
        self.predict_length = configs.total_length - configs.input_length
        self.total_length = configs.total_length

        self.resweight = nn.Parameter(torch.Tensor([0]))

    def edit_config(self,configs):
        if configs.patch_size != 1:
            print(f"Warning: patch_size is {configs.patch_size} but should be 1 for TF model. Setting to 1.")
            configs.patch_size = 1
        return configs
        
        
    def core_forward(self, seq_total, istrain=True, **kwargs):
        # bs, ts, nc, sx, sy = seq_total.shape
        inpt = self.preprocessor.batched_input_transform(seq_total)
        out = self.preprocessor.batched_output_transform(inpt)

        loss_pred = self.resweight
        loss_decouple = torch.tensor(0.0)

        return loss_pred, loss_decouple, out
