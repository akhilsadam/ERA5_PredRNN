import torch
import torch.nn as nn

# https://www.coderskitchen.com/guided-backpropagation-with-pytorch-and-tensorflow/#

def relu_hook_function(module, grad_in, grad_out):
    if isinstance(module, torch.nn.ReLU):
        return (torch.clamp(grad_in[0], min=0.),)
    
def modify_model(test_model):
    handles = []
    for i, module in enumerate(test_model.modules()):
        if isinstance(module, torch.nn.ReLU):
            print(test_model.named_modules())
            handles.append(module.register_backward_hook(relu_hook_function))
    return handles

def revert_model(handles):
    for handle in handles:
        handle.remove()