import os,sys, importlib, numpy as np
import matplotlib.pyplot as plt
import jpcm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as mticker
import traceback
from PIL import Image

cm = jpcm.get('sunburst')

def pilsave(name, cmap, data):
    img = Image.fromarray((cmap(data)[:, :, :3] * 255).astype(np.uint8))
    img.save(name)

def viz(data, frame, gt, config):
    # data shape : (length, channels, height, width)    
    checkpoint_dir = config.gen_frm_dir
    result_path = f"{checkpoint_dir}saliency/"
    
    os.makedirs(result_path, exist_ok=True)
    
    fig, axs = plt.subplots(ncols=3,figsize=(15,15))
    datas = [data, frame, gt]
    names = ['salient', 'predicted', 'ground_truth']
    for i in range(3):
        ax = axs[i]
        data = datas[i]
        in_length, channels, height, width = data.shape
        scale = np.nanstd(data)
        data = data / (3*scale)
        data.swapaxes(1,2)
        datas[i] = data.reshape(in_length * height, channels * width)
        
        pos = ax.imshow(datas[i], interpolation='nearest', cmap=cm)
        fig.colorbar(pos, ax=ax)  
        
        pilsave(result_path + f'{names[i]}.png', cm, datas[i])
        
    plt.ylabel('Time [height]')
    plt.xlabel('Salient/Gradient, Predicted, Ground Truth [width]')
    plt.savefig(result_path + f'all_mpl.png')
    plt.close()
    