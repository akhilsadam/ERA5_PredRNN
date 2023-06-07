import torch
import numpy as np
import torch.nn as nn
import sys
from core.models.model_base import BaseModel
class TF(BaseModel):
    # Transformer     
    
    def __init__(self, num_layers, num_hidden, configs):
        super(POD, self).__init__(num_layers, num_hidden, configs)
        
