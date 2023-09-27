import os.path
import datetime
import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from core.utils import preprocess, metrics
# import lpips
import torch
import wandb

from scipy import ndimage


# modified from trainer.py - added save_output option

def center_enhance(img, min_distance = 100, sigma=4, radii=np.arange(0, 20, 2),find_max=True,enhance=True,multiply=2):
    if enhance:
        filter_blurred = ndimage.gaussian_filter(img,1)
        res_img = img + 30*(img - filter_blurred)
    else:
        res_img = ndimage.gaussian_filter(img,3)
    return res_img

# loss_fn_alex = lpips.LPIPS(net='alex')


def train(model, ims, real_input_flag, configs, itr):
    c1, c2, c3 = model.train(ims, real_input_flag)
    if configs.reverse_input:
        ims_rev = np.flip(ims, axis=1).copy()
        c12, c22, c32= model.train(ims_rev, real_input_flag)
        c1 += (c1+c12) / 2
        c2 += (c2+c22) / 2
        c3 += (c3+c32) / 2
    return c1, c2, c3
            
def update(model, cost,c2,c3, configs, itr):
    model.step(cost, c2, c3)
    
    if itr % configs.display_interval == 0:
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'itr: ' + str(itr))
        print('training loss: ' + str(cost))


def test(model, test_input_handle, configs, itr, last_test=False):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'testing...')
    
    memory_saving = True # configs.weather_prediction
    
    # reverse schedule sampling
    if configs.reverse_scheduled_sampling == 1:
        mask_input = 1
    else:
        mask_input = configs.input_length

    real_input_flag = torch.zeros((configs.test_batch_size, configs.total_length-mask_input-1, 1, 1, 1))
    # print(f"real_input_flag: {real_input_flag.shape}")
    if configs.reverse_scheduled_sampling == 1:
        real_input_flag[:, :configs.input_length - 1, :, :] = 1.0

    # test_ims = test_input_handle.get_batch()
    real_input_flag = torch.FloatTensor(real_input_flag).to(configs.device)
    test_ims_ALL = []
    img_out_ALL = []
    avg_mse = 0
    n = 0
    
    test_iterations = configs.test_iterations
    
    if last_test:
        try:
            mab = test_input_handle.get_max_batches()
            test_input_handle.reset_for_full_test()
        except:
            pass
        else:
            test_iterations = mab
    
    for i in range(test_iterations):
        try:
            test_ims = test_input_handle.get_batch()
            # print(f"test_ims shape: {test_ims.shape}")
            
            test_ims = torch.FloatTensor(test_ims).to(configs.device)
            output_length = configs.total_length - configs.input_length
            torch.cuda.empty_cache()
            img_out, loss, loss_pred, decouple_loss = model.test(test_ims, real_input_flag)
            img_out = img_out.detach()

            avg_mse += torch.mean((img_out[:,-output_length:]-test_ims[:,-output_length:])**2*configs.area_weight).cpu().numpy()
            n+=1
            print(f"{configs.save_file}, loss: {loss.mean()}, avg_mse: {avg_mse}")

            if memory_saving and configs.save_output:
                test_ims_ALL.append(test_ims.cpu().numpy())
                img_out_ALL.append(img_out.cpu().numpy())
                del test_ims
                del img_out
                torch.cuda.empty_cache()
            else:
                test_ims_ALL.append(test_ims)
                img_out_ALL.append(img_out)
            
            test_input_handle.next()

        except Exception as e:
            print(f"Error: {e}")
            break
    avg_mse /= n


    # real_input_flag = torch.FloatTensor(real_input_flag).to(configs.device)
    # test_ims = torch.FloatTensor(test_ims).to(configs.device)
    # output_length = configs.total_length - configs.input_length
    # torch.cuda.empty_cache()
    # img_out, loss, loss_pred, decouple_loss = model.test(test_ims, real_input_flag)
    # img_out = img_out.detach()
    # # print(f"test_ims shape: {test_ims.shape}, img_out shape: {img_out.shape}")

    # # for i in range(configs.total_length-1):
    # #     mse_i = np.mean((img_out[:,i]-test_ims[:,i+1])**2*configs.area_weight)
    # #     img_mse.append(mse_i)
    # #     print(i, mse_i)
    # avg_mse = torch.mean((img_out[:,-output_length:]-test_ims[:,-output_length:])**2*configs.area_weight).cpu().numpy()
    # print(f"{configs.save_file}, loss: {loss.mean()}, avg_mse: {avg_mse}")
    # test_input_handle.next()

    if last_test and configs.save_output:
        res_path = os.path.join(configs.gen_frm_dir, str(itr))
        if not os.path.exists(res_path):
            os.makedirs(res_path)
            print('trainer.test function created path:', res_path)
        else:
            print('trainer.test function found path:', res_path)
            
        if memory_saving:
            # print("saving test results to:", res_path)
            # import gc
            # gc.collect()
            A = np.stack(test_ims_ALL)
            B = np.stack(img_out_ALL)
            
        else:
            A =  torch.stack(test_ims_ALL).cpu().numpy()
            B =  torch.stack(img_out_ALL).cpu().numpy()
            
        print(f"A shape: {A.shape}, B shape: {B.shape}")
        tdp = os.path.join(res_path,'true_data.npy')
        pdp = os.path.join(res_path,'pred_data.npy')
        # # touch the files to make sure they exist
        # os.system(f"touch {tdp}")
        # os.system(f"touch {pdp}")
        np.save(tdp, A)
        np.save(pdp, B)
        print('saved test results to:', res_path)
    if configs.upload_run:
        wandb.log({"Test mse": float(avg_mse)})
    return avg_mse


  