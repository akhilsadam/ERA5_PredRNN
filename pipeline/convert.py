import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import os, sys
import param, normalize


def convert(path, directory, logger=None, pygrib_fmt=True, final_data=None, input_length=20, total_length=40, **kwargs):
    if logger is None:
        import logging
        logger = logging.getLogger()
    
    # load data
    logger.info('Loading data...')
    os.makedirs(directory, exist_ok=True)
    os.makedirs(f'{directory}/snapshots', exist_ok=True)

    varnames = [normalize.short[x.split(' ')[0]] for x in param.data['variable']]
    n_var = len(varnames)
    
    if pygrib_fmt:
        import pygrib
        # collect data
        logger.info('Collecting data...')
        var_arrs = [[] for _ in range(n_var)]
        
        for i, pathi in enumerate(path):
            data = pygrib.open(pathi)
            for j,slice in tqdm(enumerate(data)):
                var_arrs[i].append(slice.values[np.newaxis,...])
            
            n_step = j+1
        mid = n_step // 2
        # normalize data
        logger.info('Normalizing data...')
        for i in range(n_var):
            var_arrs[i] = normalize.norm_func[varnames[i]](var_arrs[i]) 

        # plot distribution
        logger.info(f'Making distribution snapshots (at timestamp {mid})...')
        for i,s in tqdm(enumerate(varnames)):
            plt.figure(figsize=(5,5))
            plt.hist(var_arrs[i][mid,...])
            plt.title(f'{normalize.short_inv[s]} distribution at timestamp {mid}')
            plt.savefig(f'{directory}/snapshots/distribution_{normalize.short_inv[s]}.png')
            plt.close()

        # collect data
        logger.info('Stacking data...')
        final_data = np.vstack([var_arrs[i][np.newaxis,...] for i in range(n_var)])
        final_data = np.swapaxes(final_data,0,1)
        q = (final_data.shape[2] // 2) * 2 # trim to even size for clipping
        final_data = final_data[:,:,:q,:]
    
    else:
        assert final_data is not None, logger.critical('final_data must be provided if pygrib_fmt is False')
        assert len(final_data.shape) == 4, logger.critical('final_data must be 4D: (time, var, lat, lon)')
        n_step = final_data.shape[0]
        mid = n_step // 2
    
    logger.info('\tData Shape:', final_data.shape)

    # plot snapshots
    logger.info(f'Making variable snapshots (at timestamp {mid})...')
    for i,s in tqdm(enumerate(varnames)):
        plt.figure(figsize=(20,10))
        plt.imshow(final_data[mid,i,:,:])
        plt.colorbar()
        plt.title(f'{normalize.short_inv[s]} at timestamp {mid}')
        plt.savefig(f'{directory}/snapshots/{s}.png')
        plt.close()


    # make clips
    logger.info('Making clips...')
    in_step = input_length
    out_step = input_length

    nc = int(n_step//in_step)
    final_clips = np.ones((2,nc,2))*in_step
    final_clips[0,:,0] = np.arange(0,n_step,in_step)[:nc]
    final_clips[1,:,0] = np.arange(in_step,n_step+1,out_step)[:nc]
    final_clips[1,:,1] = out_step
    logger.info('\tClip Shape:', list(final_clips.shape))

    dim_shape = final_data.shape[1:]
    final_ds = {
        'input_raw_data': final_data,
        'dims': np.array([dim_shape,]).astype(np.int32),
        'clips': final_clips.astype(np.int32),
    }

    # save data
    logger.info('Saving data...')
    np.savez(f'{directory}/data.npz', **final_ds)
    # copy param.py to directory
    os.system(f'cp {os.path.dirname(__file__)}/param.py {directory}/param.py')

    logger.info('Done!')
    
if __name__ == '__main__':
    assert len(sys.argv) == 3, 'Usage: python convert.py <input_path> <directory>'
    convert(*sys.argv[1:3])