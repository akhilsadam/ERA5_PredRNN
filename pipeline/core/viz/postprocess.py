import os,sys, importlib, numpy as np
import matplotlib.pyplot as plt
import jpcm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as mticker
import traceback
import zipfile

import normalize

custom = True


def get_shape(path):
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if name.endswith('input_raw_data.npy'):
                npy = archive.open(name)
                version = np.lib.format.read_magic(npy)
                shape, fortran, dtype = np.lib.format._read_array_header(npy, version)
                break
    return shape

def visualize(hyp):
    user=os.popen('whoami').read().replace('\n','')
    print(f'User: {user}')
    try:
        modelname = hyp.model_name
        preprocessor = hyp.preprocessor_name
        input_length = int(hyp.input_length)
        total_length = int(hyp.total_length)
        assert input_length > 0
        assert total_length > input_length


        spec = importlib.util.spec_from_file_location("module.name", f'./user/{user}_param.py')
        userparam = importlib.util.module_from_spec(spec)
        sys.modules["module.name"] = userparam
        spec.loader.exec_module(userparam)
        
        datadir = userparam.param['data_dir']
        folders = os.listdir(datadir)
        
        if hyp.weather_prediction:        

            header = 'CDS'
            # get first folder with header
            path = [f for f in folders if header in f][0]
            fullpath = f'{datadir}/{path}'
            
            spec = importlib.util.spec_from_file_location("module.name", f'{fullpath}/param.py')
            genparam = importlib.util.module_from_spec(spec)
            sys.modules["module.name"] = genparam
            spec.loader.exec_module(genparam)
            varnames = [normalize.short[x.split(" ")[0]] for x in genparam.data['variable']]
        else:
            varnames = [f'var {x}' for x in range(5)]
            
            header = 'PDE'
            # get first folder with header
            path = [f for f in folders if header in f][0]
            fullpath = f'{datadir}/{path}'
            
        # data_shape = get_shape(f'{fullpath}/data.npz')
        
        # # shape is (time, var, x, y)
        # n_in_batch = data_shape[0] - total_length + 1
        # n_batch = hyp.n_valid
        
        
        options=hyp.opt_str
        checkpoint_dir = f"{userparam.param['model_dir']}/{hyp.model_name}/{hyp.preprocessor_name}{options}/"
                
        try: 
            n = int(hyp.pretrain_name.split('_')[1].split('.')[0])
            model = f"{modelname}-{preprocessor}-{options}-{n}it"
        except:
            n = max([int(i.split('_')[1].split('.')[0]) for i in os.listdir(checkpoint_dir) if ('.ckpt' in i and 'best' not in i)])
            model = f"{modelname}-{preprocessor}-{options}~={n}it" #WV_0_PC_0_EH_0_PS_1
        
        result_path = f"{checkpoint_dir}test_result/"
        
        gt = np.load(f'{result_path}true_data.npy')#.replace('/mnt/c','C:'))
        pd = np.load(f'{result_path}pred_data.npy')#.replace('/mnt/c','C:'))
               
        # unnormalize data
        if hyp.weather_prediction:
            invfunc = lambda x: normalize.norm_inv[varnames[x]]

            for i in range(gt.shape[3]):
                gt[:,:,:,i,:,:] = invfunc(i)(gt[:,:,:,i,:,:])
                pd[:,:,:,i,:,:] = invfunc(i)(pd[:,:,:,i,:,:])
        
        def make_plots(gt, pd):
            # stepi = 5
            # var_dicts = {0:'u wind', 1:'v wind', 2:'Sea Surface Temperature',
            #             3:'Surface Pressure', 4:'Precipitation'}
            # var_dicts = {0:'-',}
            shift = gt.shape[2] - pd.shape[2]
            # var = 0
            # b = 4 # BATCH
            # a = 3 # MINIBATCH
            # stepi =34

            bs = gt.shape[0] * gt.shape[1]
            # bts = [0]
            # bts = np.arange(0,bs-0.9,1).astype(int)
            # linspace instead at 5 points
            bts = np.linspace(0,bs-0.9,5).astype(int)
            sps = [0,5,25,30,38]
            # sps = list(range(39))
            
            cmw = jpcm.get('desert')

            variables = gt.shape[3]
            for var in range(variables):
                for bq in bts:
                    b = bq // gt.shape[1]
                    if b == gt.shape[0]:
                        continue # check to make sure
                    a = bq % gt.shape[1]
                    for stepi in sps:

                        stepd = input_length - 1 # last true frame

                        rm = np.min(gt[b,a,shift+stepi,var,:,:])
                        rx = np.max(gt[b,a,shift+stepi,var,:,:])

                        fig, axs = plt.subplots(2,3, figsize=(15,10))
                        ax0 = axs[0,0]
                        im0 = ax0.imshow(gt[b,a,shift+stepi,var,:,:], vmin=rm, vmax=rx, cmap=cmw)
                        divider = make_axes_locatable(ax0)
                        cax = divider.append_axes('right', size='5%', pad=0.05)
                        fig.colorbar(im0, cax=cax, orientation='vertical')
                        ax0.set_title(f'GT (ERA5) {varnames[var]}')
                        ax1 = axs[0,1]
                        im1 = ax1.imshow(pd[b,a,stepi,var,:,:], vmin=rm, vmax=rx, cmap=cmw)
                        divider = make_axes_locatable(ax1)
                        cax = divider.append_axes('right', size='5%', pad=0.05)
                        fig.colorbar(im1, cax=cax, orientation='vertical')
                        ax1.set_title(f'{modelname} {varnames[var]}')
                        
                        
                        rm2 = np.min(gt[b,a,shift+stepi+1,var,:,:]-gt[b,a,shift+stepi,var,:,:])
                        rx2 = np.max(gt[b,a,shift+stepi+1,var,:,:]-gt[b,a,shift+stepi,var,:,:])

                        ax0 = axs[1,0]
                        im0 = ax0.imshow(gt[b,a,shift+stepi,var,:,:]-gt[b,a,shift+stepd,var,:,:], vmin=rm2, vmax=rx2, cmap=cmw)
                        divider = make_axes_locatable(ax0)
                        cax = divider.append_axes('right', size='5%', pad=0.05)
                        fig.colorbar(im0, cax=cax, orientation='vertical')
                        ax0.set_title(f'GT (ERA5) (change from last true) {varnames[var]}')

                        ax1 = axs[1,1]
                        im1 = ax1.imshow((pd[b,a,stepi,var,:,:]-pd[b,a,stepd,var,:,:]), vmin=rm2, vmax=rx2, cmap=cmw)
                        divider = make_axes_locatable(ax1)
                        cax = divider.append_axes('right', size='5%', pad=0.05)
                        fig.colorbar(im1, cax=cax, orientation='vertical')
                        ax1.set_title(f'{modelname} (change from last true) {varnames[var]}')

                        ax2 = axs[0,2]
                        d = gt[b,a,shift+stepi,var,:,:]-pd[b,a,stepi,var,:,:]
                        d0 = gt[b,a,shift+stepi+1,var,:,:]-gt[b,a,shift+stepi,var,:,:]
                        im2 = ax2.imshow(np.abs(d), cmap=cmw)
                        divider = make_axes_locatable(ax2)
                        cax = divider.append_axes('right', size='5%', pad=0.05)
                        fig.colorbar(im2, cax=cax, orientation='vertical')
                        mse = np.mean(d**2)
                        relmse = mse / np.mean(gt[b,a,shift+stepi,var,:,:]**2)
                        meanMse = mse*100 / np.mean(d0**2)
                        # medianMse = np.median(d**2)*100 / np.median(d0**2)
                        ax2.set_title('Actual MSE')
                        ax3 = axs[1,2]
                        ax3.set_title(f'Actual mse={mse:.4f}\nRelative mse={relmse:.4f}', \
                            y=0.5)
                        nm = f'frame_{stepi}_from_batch_{b*gt.shape[1] + a}_var_{varnames[var]}'
                        plt.suptitle(f'{nm}-{model}')
                        plt.savefig(f'{result_path}{nm}.png')#.replace('/mnt/c','C:'))
                        plt.close()
                        # plt.show()

                    # fig,axs = plt.subplots(1,6, figsize=(15,6))

            total_length = gt.shape[2]
            fig = plt.figure(figsize=(24,6), constrained_layout=True)
            gs = fig.add_gridspec(1, 6,  width_ratios=(8,2,8,2,8,3))
            axs = []
            for i in range(3):
                axs.append(fig.add_subplot(gs[0, 2*i]))
                axs.append(fig.add_subplot(gs[0, 2*i+1], sharey=axs[-1]))
            # axs.append(fig.add_subplot(gs[0, 6]))
            cmap = jpcm.get('desert')
            cs = cmap.resampled(gt.shape[0]).colors
            avg = 0.0
            ravg = 0.0
            k = 0
            allmeans = []
            allrelmeans = []
            allspmeans = []
            for ba in range(gt.shape[0]*gt.shape[1]): # for each batch (i.e. each general section of time)
                b = ba // gt.shape[1]
                a = ba % gt.shape[1]
                means = []
                relMeans = []
                spMeans = []
                for f in range(gt.shape[2]-shift-1): # for each frame (i.e. each time step in the batch)
                    d = gt[b,a,shift+f,var,:,:]-pd[b,a,f,var,:,:]
                    g = gt[b,a,shift+f,var,:,:]
                    g0 = gt[b,a,shift+f+1,var,:,:]-gt[b,a,shift+f,var,:,:]
                    d0 = pd[b,a,f+1,var,:,:]-pd[b,a,f,var,:,:]
                    meanMse = np.mean(d**2)
                    # meanShiftMse = np.mean(d0**2)
                    meang = np.mean(g**2)
                    means.append(meanMse)
                    relMeans.append(meanMse/meang)
                    spMeans.append(np.mean(d0**2) / np.mean(g0**2))

                axs[0].plot(means, color=cs[b], label=f'Timestep Batch {b}')
                axs[2].plot(relMeans, color=cs[b], label=f'Timestep Batch {b}')
                spMeans[:stepd] = [1.0]*stepd
                spMeans[stepd] = np.nan
                axs[4].plot(spMeans, color=cs[b], label=f'Timestep Batch {b}')
                avg += np.mean(means[stepd+1:])
                ravg += np.mean(relMeans[stepd+1:])
                k+=1
                allmeans.extend(means[stepd+1:])
                allrelmeans.extend(relMeans[stepd+1:])
                allspmeans.extend(spMeans[stepd+1:])
            avg /= k
            # avg *= total_length/(total_length-input_length)
            ravg /= k
            # ravg *= total_length/(total_length-input_length)
            # plt.plot(medians, label='Median')
            axs[0].set_xlabel('        Lead Time (in frames, first half is known data, second half is testing)')
            axs[0].set_title(f'MSE (absolute) = {avg:.4f}')
            axs[0].set_ylabel('MSE')
            axs[0].set_yscale('log')
            axs[2].set_yscale('log')
            axs[4].set_yscale('log')
            axs[2].set_title(f'Relative MSE = {ravg:.4f} :\n(MSE / frame magnitude)')
            axs[4].set_title('Scale (rel.MSE of frame delta) :\n (pred.frame-to-frame MSE / frame-to-frame MSE)')
            allmeansa = np.array(allmeans)
            # allmeansa = allmeansa[np.where(allmeansa > 0)]
            allrelmeansa = np.array(allrelmeans)
            # allrelmeansa = allrelmeansa[np.where(allrelmeansa > 0)]
            allspmeansa = np.array(allspmeans)
            # allspmeansa = allspmeansa[np.where(allspmeansa1)]
            axs[1].hist(allmeansa, bins=100, color='k', orientation='horizontal')
            axs[1].set_title(f'med={np.median(allmeansa):.4f}\niqr={np.quantile(allmeansa,0.75)-np.quantile(allmeansa,0.25):.4f}')
            axs[3].hist(allrelmeansa, bins=100, color='k', orientation='horizontal')
            axs[3].set_title(f'med={np.median(allrelmeansa):.4f}\niqr={np.quantile(allrelmeansa,0.75)-np.quantile(allrelmeansa,0.25):.4f}')
            axs[5].hist(allspmeansa, bins=100, color='k', orientation='horizontal')
            axs[5].set_title(f'med={np.median(allspmeansa):.4f}\niqr={np.quantile(allspmeansa,0.75)-np.quantile(allspmeansa,0.25):.4f}')
            minsy = [np.min(allmeansa),np.min(allrelmeansa),np.min(allspmeansa)]
            maxsy = [np.max(allmeansa),np.max(allrelmeansa),np.max(allspmeansa)]
            
            
            divider = make_axes_locatable(axs[5])
            cax = divider.append_axes('right', size='13%', pad=0.20)
                       
            fig.colorbar(
                mpl.cm.ScalarMappable(
                    norm=mpl.colors.Normalize(0, gt.shape[0] * gt.shape[1] * gt.shape[2]), cmap=cmap # * gt.shape[2] if  not using custom dataloader
                ),
                cax=cax,
                orientation='vertical',
                label=f'Timestep Start Time',
                # ticks=np.arange(0, gt.shape[0] * gt.shape[1] * gt.shape[2], gt.shape[2]).tolist(),
            )
            # plt.legend()/
            if preprocessor in ['raw','scale','control']:
                # skip input_length frames
                for maxy,miny, ax in zip(maxsy, minsy,axs[::2]):
                    ax.set_xlim([input_length, total_length-1])
                    ax.set_ylim([miny, maxy])
            
            [ax.yaxis.set_major_locator(plt.MaxNLocator(10)) for ax in axs[::2]]
            formatter = mticker.FuncFormatter(lambda y, _: '{:g}'.format(y))
            [ax.yaxis.set_major_formatter(formatter) for ax in axs[::2]]
            [ax.yaxis.set_minor_formatter(formatter) for ax in axs[::2]]
            [plt.setp(ax.get_yminorticklabels(), visible=False) for ax in axs[::2]]
            [ax.get_yaxis().set_visible(False) for ax in axs[1::2]]

            plt.suptitle(f'{model} MSE for output frames (all frames shown)')
            # plt.tight_layout()
            plt.savefig(f'{result_path}mse.png',bbox_inches='tight')#.replace('/mnt/c','C:'))
            plt.show()
            
        make_plots(gt,pd)
        
    except Exception as e:
        print(f'Error: {e} for {model}, skipping...')
        print(traceback.format_exc())