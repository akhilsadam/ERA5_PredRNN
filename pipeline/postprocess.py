import os,sys, importlib, numpy as np
import matplotlib.pyplot as plt
import jpcm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

def visualize(hyp):
    user=os.popen('whoami').read().replace('\n','')
    print(f'User: {user}')
    try:
        modelname = hyp.model_name
        preprocessor = hyp.preprocessor_name
        model = f"{modelname}/{preprocessor}/" #WV_0_PC_0_EH_0_PS_1

        spec = importlib.util.spec_from_file_location("module.name", f'./user/{user}_param.py')
        userparam = importlib.util.module_from_spec(spec)
        sys.modules["module.name"] = userparam
        spec.loader.exec_module(userparam)

        checkpoint_dir = userparam.param['model_dir']

        result_path = f"{checkpoint_dir}/{model}/test_result/"
        
        gt = np.load(f'{result_path}true_data.npy'.replace('/mnt/c','C:'))
        pd = np.load(f'{result_path}pred_data.npy'.replace('/mnt/c','C:'))
        
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
            bts = np.arange(0,bs-0.9,3).astype(int)
            sps = [5,25,30,38]

            variables = gt.shape[3]
            for var in range(variables):
                for bq in bts:
                    b = bq // gt.shape[1]
                    a = bq % gt.shape[1]
                    for stepi in sps:

                        stepd = 19 # last true frame

                        fig, axs = plt.subplots(2,3, figsize=(15,10))
                        ax0 = axs[0,0]
                        im0 = ax0.imshow(gt[b,a,shift+stepi,var,:,:])
                        divider = make_axes_locatable(ax0)
                        cax = divider.append_axes('right', size='5%', pad=0.05)
                        fig.colorbar(im0, cax=cax, orientation='vertical')
                        ax0.set_title(f'GT (ERA5) {var}')
                        ax1 = axs[0,1]
                        im1 = ax1.imshow(pd[b,a,stepi,var,:,:])
                        divider = make_axes_locatable(ax1)
                        cax = divider.append_axes('right', size='5%', pad=0.05)
                        fig.colorbar(im1, cax=cax, orientation='vertical')
                        ax1.set_title(f'{modelname} {var}')


                        ax0 = axs[1,0]
                        im0 = ax0.imshow(gt[b,a,shift+stepi,var,:,:]-gt[b,a,shift+stepd,var,:,:])
                        divider = make_axes_locatable(ax0)
                        cax = divider.append_axes('right', size='5%', pad=0.05)
                        fig.colorbar(im0, cax=cax, orientation='vertical')
                        ax0.set_title(f'GT (ERA5) (change from last true) {var}')

                        ax1 = axs[1,1]
                        im1 = ax1.imshow((pd[b,a,stepi,var,:,:]-pd[b,a,stepd,var,:,:]))
                        divider = make_axes_locatable(ax1)
                        cax = divider.append_axes('right', size='5%', pad=0.05)
                        fig.colorbar(im1, cax=cax, orientation='vertical')
                        ax1.set_title(f'{modelname} (change from last true) {var}')

                        ax2 = axs[0,2]
                        d = gt[b,a,shift+stepi,var,:,:]-pd[b,a,stepi,var,:,:]
                        d0 = gt[b,a,shift+stepi+1,var,:,:]-gt[b,a,shift+stepi,var,:,:]
                        im2 = ax2.imshow(np.abs(d))
                        divider = make_axes_locatable(ax2)
                        cax = divider.append_axes('right', size='5%', pad=0.05)
                        fig.colorbar(im2, cax=cax, orientation='vertical')
                        mse = np.mean(d**2)
                        relmse = mse / np.mean(gt[b,a,shift+stepi,var,:,:]**2)
                        meanMse = mse*100 / np.mean(d0**2)
                        # medianMse = np.median(d**2)*100 / np.median(d0**2)
                        ax2.set_title('Actual MSE')
                        ax3 = axs[1,2]
                        ax3.set_title(f'(as compared to off-by-one frame):\n\trelative mse={meanMse:.4f}%.\nActual mse={mse:.4f}\nScaled mse={relmse:.4f}', \
                            y=0.5)
                        nm = f'frame_{stepi}_from_batch_{b*gt.shape[1] + a}_var_{var}'
                        plt.suptitle(nm)
                        plt.savefig(f'{result_path}{nm}.png'.replace('/mnt/c','C:'))
                        plt.close()
                        # plt.show()

            fig,axs = plt.subplots(1,2, figsize=(10,5))
            cmap = jpcm.get('desert')
            cs = cmap.resampled(gt.shape[0]).colors
            avg = 0.0
            for ba in range(gt.shape[0]*gt.shape[1]): # for each batch (i.e. each general section of time)
                b = ba // gt.shape[1]
                a = ba % gt.shape[1]
                means = []
                relMeans = []
                for f in range(gt.shape[2]-shift-1): # for each frame (i.e. each time step in the batch)
                    d = gt[b,a,shift+f,var,:,:]-pd[b,a,f,var,:,:]
                    d0 = gt[b,a,shift+f+1,var,:,:]-gt[b,a,shift+f,var,:,:]
                    meanMse = np.mean(d**2)
                    meanShiftMse = np.mean(d0**2)
                    means.append(meanMse) # average over minibatch
                    relMeans.append(100*meanMse/meanShiftMse) # average over minibatch

                axs[0].plot(means, color=cs[b], label=f'Batch {b}')
                axs[1].plot(relMeans, color=cs[b], label=f'Batch {b}')
                avg += np.mean(means)
            avg /= gt.shape[0]*gt.shape[1]
            avg *= 2 # because we only have half the frames
            # plt.plot(medians, label='Median')
            axs[0].set_xlabel('        Lead Time (in frames, first half is known data, second half is testing)')
            axs[0].set_title(f'MSE (absolute) = {avg:.4f}')
            axs[0].set_yscale('log')
            axs[1].set_yscale('log')
            axs[1].set_title('Relative MSE (as compared to frame stepping MSE) %')
            divider = make_axes_locatable(axs[1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(
                mpl.cm.ScalarMappable(
                    norm=mpl.colors.Normalize(0, gt.shape[0] * gt.shape[1] * gt.shape[2]), cmap=cmap
                ),
                cax=cax,
                orientation='vertical',
                label=f'Batch Start Time (every {gt.shape[2]} frames)',
                # ticks=np.arange(0, gt.shape[0] * gt.shape[1] * gt.shape[2], gt.shape[2]).tolist(),
            )
            # plt.legend()/
            plt.suptitle(f'{modelname} MSE')
            plt.savefig(f'{result_path}mse.png'.replace('/mnt/c','C:'))
            plt.close()
            
        make_plots(gt,pd)
        
    except Exception as e:
        print(f'Error: {e} for {model}, skipping...')