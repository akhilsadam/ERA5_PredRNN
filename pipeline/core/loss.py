# batch, sequence, latent
# input_length 
import torch

def loss_mse(out, seq_total, input_length):
    return torch.nn.functional.mse_loss(out[:,input_length:,:], seq_total[:,input_length:,:])

def loss_grad(out, seq_total, input_length):
    return torch.nn.functional.mse_loss(out[:,input_length+1:,:] - out[:,input_length:-1,:], seq_total[:,input_length+1:,:] - seq_total[:,input_length:-1,:])

def loss_laplace(out, seq_total, input_length):
    return torch.nn.functional.mse_loss(out[:,input_length+2:,:] - 2*out[:,input_length+1:-1,:] + out[:,input_length:-2,:], seq_total[:,input_length+2:,:] - 2*seq_total[:,input_length+1:-1,:] + seq_total[:,input_length:-2,:])

def loss_mixed(out, seq_total, input_length):
    return loss_mse(out, seq_total, input_length) + loss_grad(out, seq_total, input_length) + 0.1 * loss_laplace(out, seq_total, input_length)

def loss_mixed_additional(out, seq_total, input_length):
    return loss_grad(out, seq_total, input_length) + 0.1 * loss_laplace(out, seq_total, input_length)