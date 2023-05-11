import numpy as np
from matplotlib import pyplot as plt
import pygrib
import tqdm
import os, sys
import param, normalize


def convert(path, output_path):
    # load data
    print('Loading data...')
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f'{output_path}/snapshots', exist_ok=True)

    data = pygrib.open(path)
    n_step = len(data)
    mid = n_step // 2
    varnames = [normalize.short(x) for x in param.data['variable']]
    n_var = len(varnames)

    var_arrs = [[] for _ in range(n_var)]

    # collect data
    print('Collecting data...')
    for i, slice in tqdm(enumerate(data)):
        r = i % n_var
        var_arrs[r].append(slice.values[np.newaxis,...])

    # normalize data
    print('Normalizing data...')
    for i in range(n_var):
        var_arrs[i] = normalize.norm_func[varnames[i]](var_arrs[i]) 

    # plot distribution
    print(f'Making distribution snapshots (at timestamp {mid})...')
    for i,s in tqdm(enumerate(varnames)):
        plt.figure(figsize=(5,5))
        plt.hist(var_arrs[i][mid,...])
        plt.title(f'{normalize.short_inv[s]} distribution at timestamp {mid}')
        plt.savefig(f'{output_path}/snapshots/distribution_{normalize.short_inv[s]}.png')
        plt.close()

    # collect data
    print('Stacking data...')
    final_data = np.vstack([var_arrs[i][np.newaxis,...] for i in range(n_var)])
    final_data = np.swapaxes(final_data,0,1)
    q = (final_data.shape[2] // 2) * 2 # trim to even size for clipping
    final_data = final_data[:,:,:q,:]
    print('\tData Shape:', final_data.shape)

    # plot snapshots
    print(f'Making variable snapshots (at timestamp {mid})...')
    for i,s in tqdm(enumerate(varnames)):
        plt.figure(figsize=(20,10))
        plt.imshow(final_data[mid,i,:,:])
        plt.colorbar()
        plt.title(f'{normalize.short_inv[s]} at timestamp {mid}')
        plt.savefig(f'{output_path}/snapshots/{s}.png')
        plt.close()


    # make clips
    print('Making clips...')
    in_step = 2
    out_step = 2

    final_clips = np.ones((2,int(np.ceil(n_step/(in_step))),2))*in_step
    final_clips[0,:,0] = np.arange(0,n_step,in_step)
    final_clips[1,:,0] = np.arange(in_step,n_step+1,out_step)
    final_clips[1,:,1] = out_step
    print('\tClip Shape:', final_clips.shape)

    dim_shape = final_data.shape[1:]
    final_ds = {
        'input_raw_data': final_data,
        'dims': np.array(dim_shape).astype(np.int32),
        'clips': final_clips.astype(np.int32),
    }

    # save data
    print('Saving data...')
    np.savez(f'{output_path}/data.npz', **final_ds)

    print('Done!')
    
if __name__ == '__main__':
    assert len(sys.argv) == 3, 'Usage: python convert.py <input_path> <output_path>'
    convert(*sys.argv[1:3])