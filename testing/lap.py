import numpy as np
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Process

interactive = True

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(28, 50),nn.Linear(50, 50),nn.Linear(50, 28)]
        self.stack = nn.Sequential(
            self.layers[0],
            nn.ReLU(),
            self.layers[1],
            nn.ReLU(),
            self.layers[2]
        )

        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=1.0)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.stack(x)

n=10000
batch_size = 100
epochs = 5
lr = 5e-1
xdata = torch.rand(n, 28, device=device)
ydata = 0.5*torch.sin(xdata) + (xdata)**4 + torch.sigmoid(xdata) + 0.1*torch.randn_like(xdata)

def train(animate, config):
    
    model = NeuralNetwork().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    losses = []
    prev_gradient = None
    gmag = []
    # hessiansamples = []
    moveamt = [0]*len(model.layers)
    hessianmag = []
    
    radius = config['radius']
    cav = config['concavity_adaptation'] if 'concavity_adaptation' in config else 0.0
    label = config['label']
    color = config['color'] if 'color' in config else 'blue'
    imgs = None
    size = n//batch_size
    model.train()
    with tqdm(total=epochs*size) as pbar:
        i = 0
        for epoch in range(epochs):
            for batch in range(size):
                X, y = xdata[batch*batch_size:(batch+1)*batch_size], ydata[batch*batch_size:(batch+1)*batch_size]

                # Compute prediction error
                pred = model(X)
                loss = loss_fn(pred, y)

                # Backpropagation
                loss.backward()
                loss = loss.item()
                losses.append(loss)
                
                # calculate neighborhood features
                with torch.no_grad():
                    grad = [layer.weight.grad.flatten() for layer in model.layers]
                    
                    gmag.append(sum([g.abs().mean().item() for g in grad]))
                    if prev_gradient is not None and radius > 0:
                        # print(prev_gradient)
                        hessiansample = [(prev_gradient_i - grad_i) for randvec_i, prev_gradient_i, grad_i in zip(randvecs, prev_gradient, grad)]
                        # note hessiansample is not a true hessian, but a sample of the hessian along a random direction,
                        # and here has units in 1/randvec since we divide by randvec_i twice
                        hessianmag.append(sum([(torch.dot(randvec_i,hessiansample_i)).item() for randvec_i, hessiansample_i in zip(randvecs,hessiansample)]) / (lr * radius))
                        # print(f"grad: {gmag[-1]:.3f}, hessian: {hessiansample}")
                

                        # apply neighborhood adaptations
                        moveamt = [(- cav * torch.dot(grad_i,hessiansample_i).item() * randvec_i/ (lr * radius) ).reshape(layer.weight.data.shape) for randvec_i, grad_i, hessiansample_i, layer in zip(randvecs, grad,hessiansample, model.layers)] # amount to move along randvec
                        # note now we have proper units w.r.t randvec 
                        
                        # # smooth hessianmag
                        # if len(hessianmag) > 5:
                        #     hessianmag[-1] = sum(hessianmag[-5:])/5
                
                    prev_gradient = [g.clone() for g in grad]
                    
                # step
                optimizer.step()
                
                # measure neighborhood
                with torch.no_grad():
                    
                    
                    randvecs = []
                    for k, layer in enumerate(model.layers):
                        randveci = torch.nn.functional.normalize(torch.randn_like(layer.weight.data) + moveamt[k])
                        randvecs.append(randveci.flatten())
                        layer.weight.data = layer.weight.data - lr * radius * randveci
                    
                optimizer.zero_grad()
                i += 1

                if i == 50:
                    i-=50
                    pbar.set_description(f"loss: {loss:>7f}")
                    pbar.update(50)
                    data = [losses, gmag, hessianmag]
                    imgs = animate(data, imgs, label, color)
                    

# matplotlib.use('TkAgg') # install gui backend like PyQt5 or Tkinter, don't use this command

# with plt.ion():
fig,axs = plt.subplots(1,3, figsize=(12,4))
axs[0].set_title("Loss")
axs[1].set_title("Gradient magnitude")
axs[2].set_title("Concavity magnitude") #  (estimate requires small r>0 and small lr)
axs[0].set_yscale('log')
axs[1].set_yscale('log')

def animate(data, imgs, label, color):
    if imgs is not None:
        for img in imgs:
            img.pop(0).remove()
            
    imgs = [ \
        axs[0].plot(data[0], label=label, color=color),
        axs[1].plot(data[1], label=label, color=color),
        axs[2].plot(data[2], label=label, color=color),
    ]

    
    [ax.legend() for ax in axs]
    fig.canvas.draw()
    if interactive: 
        plt.pause(0.01)
    return imgs

# anishow = ani.FuncAnimation(fig, animate, interval=100, cache_frame_data=False)

print("Starting training...")

cs = ['blue', 'red', 'green']
cv = [0, 0, .5]
for i, radius in enumerate([0, 0.01, 0.01]):
    cva = cv[i]
    config = {'radius': radius, 'concavity_adaptation':cva, 'label': f"r{radius}_c{cva}", 'color' : cs[i]}
    train(animate, config)

plt.show()

