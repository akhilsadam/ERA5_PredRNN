import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.init import zeros_, ones_, constant_
from functools import partial
import numpy as np

acts = {
    "relu": partial(F.relu, inplace=True),
    "selu": partial(F.selu, inplace=True),
    "elu": partial(F.elu, inplace=True),
    "tanh": F.tanh,
    "mish": partial(F.mish, inplace=True),
    "sig": partial(F.logsigmoid, inplace=True),
}

torch.pi = torch.acos(torch.zeros(1)).item() * 2

# define model
class ModLayer(nn.Module):
  def __init__(self, in_size, out_size, omega_0: float = 1., is_first: bool = False, train_omega: bool = False, sign: int = 0):
    super(ModLayer, self).__init__()
    
    self.sign = 1 if sign == 0 else -1
    
    self.in_size = in_size
    self.out_size = out_size

    self.train_omega = train_omega
    if train_omega:
      self.omega_cache = omega_0
      self.omega = Parameter(torch.tensor(omega_0).cuda())
    else:
      self.omega = omega_0
    self.is_first = is_first

    self.linear = nn.Linear(in_size, out_size).cuda()

    self.init_weights()

  def init_weights(self, m=6):
      with torch.no_grad():
          if self.is_first:
              self.linear.weight.uniform_(-1 / self.in_size,
                                            1 / self.in_size)
          else:
              self.linear.weight.uniform_(-np.sqrt(m / self.in_size) / self.omega,
                                            np.sqrt(m / self.in_size) / self.omega)

  def reset_parameters(self):
    if self.train_omega:
      constant_(self.omega)
    self.init_weights()

class SineLayer(ModLayer):
  def forward(self, x):
    x = self.linear(x)
    return torch.sin(self.omega * x)

class SignedSineLayer(ModLayer):
  def forward(self, x):
    x = self.linear(x)
    return torch.sin(self.sign * self.omega * x)

class SawLayer(ModLayer):
  def forward(self, x):
    x = self.linear(x)
    wx = torch.pi*self.omega*x
    eps = 0.3
    # sawtooth (modulus)
    adj = torch.cos(wx)*torch.sin(wx) / (torch.sin(wx)**2 + eps**2)
    return 0.5 - (torch.arctan(adj)) / torch.pi

class SharpTriLayer(ModLayer):
  def forward(self,x):
    x = self.linear(x)
    wx = torch.pi*self.omega*x
    eps = 0.3
    # not really a triangle wave (sharpened)
    adj = torch.sin(wx)
    return adj + (1/3 * adj**3) + (1/5 * adj**5)

class TriLayer(ModLayer):
  def forward(self, x):
    x = self.linear(x)
    wx = torch.pi*self.omega*x
    # really a triangle wave
    adj = lambda n : (-1**n) * torch.sin((2*n + 1)*2*wx) * (2*n+1)**(-2)
    return (adj(0) + adj(1) + adj(2)) * 8 * torch.pi**(-2)

class TrTriLayer(ModLayer):
  def __init__(self,*args, **kwargs):
    super().__init__(*args,**kwargs)

    self.a = Parameter(torch.tensor(1.0).cuda())
    self.b = Parameter(torch.tensor(1.0).cuda())
    self.c = Parameter(torch.tensor(1.0).cuda())

  def reset_parameters(self):
    ones_(self.a)
    ones_(self.b)
    ones_(self.c)

  def forward(self, x):
    x = self.linear(x)
    wx = torch.pi*self.omega*x
    # really a triangle wave
    adj = lambda n : (-1**n) * torch.sin((2*n + 1)*2*wx) * (2*n+1)**(-2)
    return (self.a * adj(0) + self.b * adj(1) + self.c * adj(2)) * 8 * torch.pi**(-2)

class GaussMixLayer(ModLayer):

  def init_weights(self):
    super().init_weights(m=6/5)

  def forward(self, x):
    x = self.linear(x)
    wx = torch.pi*self.omega*x
    #
    a = torch.exp(-5*wx**2)
    b = torch.exp(-5*(wx+1)**2)
    c = torch.exp(-5*(wx-1)**2)
    d = -torch.exp(-5*(wx-0.5)**2)
    e = -torch.exp(-5*(x+0.5)**2)
    #
    return (1/0.72) * (a + b + c + d + e)

class AddFMLayer(ModLayer):

  def __init__(self,*args, **kwargs):
    self.starter = 0.5
    super().__init__(*args,**kwargs)

    self.af = Parameter(torch.tensor(self.starter).cuda())
    self.sf = Parameter(torch.tensor(0.0).cuda())

    # if not self.is_first:
    #   self.fm = Parameter(torch.tensor(1.0).cuda())
    #   ## when FM modulates between two waves:
    #   # it might make more sense to
    #   # start fm at 0.0? tested and that is not the case

  def init_weights(self):
    super().init_weights(m=2*(6+self.starter) / (1+self.starter))

  def reset_parameters(self):
    zeros_(self.af)
    zeros_(self.sf)
    # if not self.is_first:
    #   ones_(self.fm)

  def forward(self, x):
    x2 = self.linear(x)

    wx2 = self.omega*x2



    # a triangle wave (HF)
    adj = lambda n, y : (-1**n) * torch.sin((2*n + 1)*2*torch.pi*y) * (2*n+1)**(-2)
    tri = lambda y : (adj(0,y) + adj(1,y) + adj(2,y)) * 8 * torch.pi**(-2)
    # a sin wave (MF)
    sin = lambda y : torch.sin(y)
    # additive/subtractive synth including constant (LF)
    wave = lambda y : (sin(y) + self.af * tri(y)) / (1.0 + self.af)
    # # FM synth
    # if not self.is_first:
    #   wx = torch.pi*self.omega*x
    #   return self.fm * wave(wx2) + (1.0-self.fm) * sin(wx2)
    return wave(wx2)

class AddFMLayer2(ModLayer):

  def __init__(self,*args, **kwargs):
    self.starter = 1.0
    super().__init__(*args,**kwargs)

    self.af = Parameter(torch.tensor(self.starter).cuda())
    self.af2 = Parameter(torch.tensor(self.starter).cuda())

  def init_weights(self):
    super().init_weights(m=(6*(1-self.starter) + self.starter))

  def reset_parameters(self):
    zeros_(self.af)
    zeros_(self.af2)

  def forward(self, x):
    x2 = self.linear(x)
    wx2 = torch.pi*self.omega*x2

    # a wave broken into frequency components (additive synthesis only)
    adj = lambda n, y : (torch.sin(y)**n) / n
    return adj(1,wx2) + self.af * adj(3,wx2) + self.af2 * adj(5,wx2)
  
class AddFMLayer3(ModLayer):

  def __init__(self,*args, **kwargs):
    self.starter = 1.0
    
    super().__init__(*args,**kwargs)
    self.limit = 1.0/self.omega
    self.af = Parameter(torch.tensor(self.starter).cuda())
    self.af2 = Parameter(torch.tensor(self.starter).cuda())

  def init_weights(self):
    super().init_weights(m=(6*(1-self.starter) + self.starter))

  def reset_parameters(self):
    zeros_(self.af)
    zeros_(self.af2)

  def forward(self, x):
    x2 = self.linear(x)
    wx2 = torch.pi*self.omega*x2
    
    # bound paf
    paf = torch.sigmoid(self.af) * self.limit
    paf2 = torch.sigmoid(self.af2) * self.limit

    # a wave broken into frequency components (additive synthesis only)
    adj = lambda n, y : (torch.sin(y)**n) / n
    return (adj(1,wx2) - paf * adj(3,wx2) + paf2 * adj(5,wx2)) # * torch.exp(-0.5*(torch.pi*x2)**2) # exponential window makes accuracy worse

sin_acts = \
 {
    "sin" : SineLayer,
    "ssin" : SignedSineLayer,
    "tri" : TriLayer,
    "trtri": TrTriLayer,
    "addfm": AddFMLayer2,
    "gm" : GaussMixLayer
 }


class MLP(nn.Module):
  def __init__(self, in_size, hidden_size, out_size, hidden_layers: int = 3, act: str = "relu", last_act: str = "None", omega_0: float = 1., train_omega: bool = False):
    super(MLP, self).__init__()
    self.in_size = in_size
    self.hidden_size = hidden_size
    self.out_size = out_size
    self.hidden_layers = hidden_layers

    self.lin_list = nn.ModuleList([]).cuda()
    self.use_sinlike = act in sin_acts.keys()
    if self.use_sinlike:
      MLayer = sin_acts[act]
      self.omega_0 = omega_0
      self.train_omega = train_omega
      for l in range(hidden_layers+1):
        if l==0:
          self.lin_list.append(MLayer(in_size, hidden_size, omega_0, True, train_omega, sign=l%2).cuda())
        else:
          self.lin_list.append(MLayer(hidden_size, hidden_size, omega_0, False, train_omega, sign=l%2).cuda())
    else:
      self.act = acts[act]
      for l in range(hidden_layers+1):
        if l==0:
          self.lin_list.append(nn.Linear(in_size, hidden_size).cuda())
        else:
          self.lin_list.append(nn.Linear(hidden_size, hidden_size).cuda())
     
    if self.use_sinlike:
      if last_act in sin_acts.keys():
        MLayer = sin_acts[last_act]     
      self.lin_list.append(MLayer(hidden_size, hidden_size, omega_0, False, train_omega).cuda())
    else:
      self.act = acts[act]
      self.lin_list.append(nn.Linear(hidden_size, hidden_size).cuda())
          
    self.lin_list.append(nn.Linear(hidden_size, out_size).cuda())

  def forward(self, x):
    if self.use_sinlike:
      for i,f in enumerate(self.lin_list):
        x = f(x)
      return x
    for i,f in enumerate(self.lin_list[:-1]):
      x = self.act(f(x))

    out = self.lin_list[-1](x)
    return out