import torch
import torch.nn as nn
import torch.nn.functional as F


# we do equivalent Q f(K V^T)
# instead of f(Q K^T) V

def sqrt_act(a):
    return torch.sqrt(torch.nn.functional.relu(a))

def sqrt_att_a(K,V,x, dim):
    # act(Kxx^tV) # no need to transpose V
    return sqrt_act(
        nn.multi_dot(K, x.view(-1,dim,1), x.view(-1,1,dim), V) # basically identify singular value coefficients.
    )
    
def lin_att_a(K,V,x, dim):
    # act(Kxx^tV) # no need to transpose V
    return nn.multi_dot(K, x.view(-1,dim,1), x.view(-1,1,dim), V)
    
def att_b(KV,Q,x):
    return nn.multi_dot(KV.T, Q, x)