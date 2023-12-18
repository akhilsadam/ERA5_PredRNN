__author__ = 'asadam'

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.layers._att import sqrt_att_a, att_b

class GateLoop(nn.Module):
    def __init__(self, nlatent, nspatial, device, nlayers=1):
        super(GateLoop, self).__init__()
        
        self.nlatent = nlatent
        self.nspatial = nspatial
        self.nlayers = nlayers
        
        # spatial att.
        self.Ks = nn.ParameterList([ \
            torch.Parameter(nn.eye(nspatial).to(device)) \
            for _ in nlayers])
        self.Qs = nn.ParameterList([ \
            torch.Parameter(nn.eye(nspatial).to(device)) \
            for _ in nlayers])
        self.Vs = nn.ParameterList([ \
            torch.Parameter(nn.eye(nspatial).to(device)) \
            for _ in nlayers])
        
        # modal / latent att.
        self.Km = nn.ParameterList([ \
            torch.Parameter(nn.eye(nlatent).to(device)) \
            for _ in nlayers])
        self.Qm = nn.ParameterList([ \
            torch.Parameter(nn.eye(nlatent).to(device)) \
            for _ in nlayers])
        self.Vm = nn.ParameterList([ \
            torch.Parameter(nn.eye(nlatent).to(device)) \
            for _ in nlayers])    
        
        # state matrix
        # self.A = torch.Parameter(nn.eye(self.nlatent*self.nspatial)) - piped in each step...
        
    def forward(self,x):
        
        # kernel in
        # TODO
        