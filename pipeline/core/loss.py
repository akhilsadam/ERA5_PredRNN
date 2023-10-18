# batch, sequence, latent
# input_length 
import torch
# import lpips
# perceptual = lpips.LPIPS(net='vgg').cuda()

def loss_mse(out, seq_total, input_length, weight):
    return torch.mean( \ 
        weight * (out[:,input_length:,:] - seq_total[:,input_length:,:])**2)

def loss_grad(out, seq_total, input_length, weight):
    return torch.mean( \
        weight * ((out[:,input_length+1:,:] - out[:,input_length:-1,:]) - (seq_total[:,input_length+1:,:] - seq_total[:,input_length:-1,:]))**2)

def loss_laplace(out, seq_total, input_length, weight):
    return torch.mean( \
        weight * ((out[:,input_length+2:,:] - 2*out[:,input_length+1:-1,:] + out[:,input_length:-2,:]) \
            -  (seq_total[:,input_length+2:,:] - 2*seq_total[:,input_length+1:-1,:] + seq_total[:,input_length:-2,:]))**2)

# def loss_perceptual(out, seq_total, input_length):
#     pl = (seq_total.size(1) - input_length) * seq_total.size(2)
#     o2 = out[:,input_length:,:].reshape(out.size(0), 1, pl*out.size(3), out.size(4)).expand(-1,3,-1,-1)
#     sq2 = seq_total[:,input_length:,:].reshape(out.size(0), 1, pl*out.size(3), out.size(4)).expand(-1,3,-1,-1)
#     # normalize to [-1,1]
#     o3 = (o2 - o2.min()) / (o2.max() - o2.min()) * 2 - 1
#     sq3 = (sq2 - sq2.min()) / (sq2.max() - sq2.min()) * 2 - 1
#     # print(f"o2 shape: {o2.shape}, sq2 shape: {sq2.shape}")
#     return perceptual(o3,sq3).mean()

def loss_mixed(out, seq_total, input_length, weight, a=0.1, b=0.01):
    return loss_mse(out, seq_total, input_length, weight) + a*loss_grad(out, seq_total, input_length, weight) + b * loss_laplace(out, seq_total, input_length, weight)

def loss_mixed_additional(out, seq_total, input_length, weight, a=0.1):
    return loss_grad(out, seq_total, input_length, weight) + a * loss_laplace(out, seq_total, input_length, weight)