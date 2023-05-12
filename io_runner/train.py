import os, importlib, numpy as np, subprocess, sys

user=os.popen('whoami').read().replace('\n','')
userparam=importlib.import_module(f'user.{user}_param')
datadir = userparam.param['data_dir']
checkpoint_dir = userparam.param['model_dir']
os.makedirs(checkpoint_dir, exist_ok=True)

train = ['PDE_47c1f778-4bf1-46bc-a27a-e60a268ba2fb',]
valid = ['PDE_cb9898ef-3e5a-4271-a856-eb73704f1986',]

train_data_paths = ','.join([f"{datadir}/{tr}/data.npz" for tr in train])
valid_data_paths = ','.join([f"{datadir}/{vd}/data.npz" for vd in valid])

dat = np.load(f"{datadir}/{train[0]}/data.npz")
shp = dat['dims'][0]
l = dat['input_raw_data'].shape[0]
param = importlib.import_module('param',f"{datadir}/{train[0]}")

if 'year' in param.data:
    img_channel = '3'
    img_layers = '0,1,2'
    input_length = '24'
    total_length = '48'
else:
    img_channel = '1' # we only have one channel for PDE data
    img_layers = '0'
    input_length = '2' # should depend on n of timesteps - exactly how?
    total_length = '4' # twice the input length (why?)

print('Data Dims:',shp)

cmd = f"python3 -u ../predrnn-pytorch/run2.py \
--is_training 1 \
--device cuda:0 \
--dataset_name mnist \
--train_data_paths {train_data_paths} \
--valid_data_paths {valid_data_paths} \
--save_dir {checkpoint_dir} \
--gen_frm_dir {checkpoint_dir} \
--model_name predrnn_v2 \
--reverse_input 0 \
--is_WV 0 \
--press_constraint 0 \
--center_enhance 0 \
--patch_size 40 \
--weighted_loss 1 \
--upload_run 1 \
--layer_need_enhance 1 \
--find_max False \
--multiply 2 \
--img_height {shp[1]} \
--img_width {shp[2]} \
--use_weight 0 \
--layer_weight 20 \
--img_channel {img_channel} \
--img_layers {img_layers} \
--input_length {input_length} \
--total_length {total_length} \
--num_hidden 480,480,480,480,480,480 \
--skip_time 1 \
--wavelet db1 \
--filter_size 5 \
--stride 1 \
--layer_norm 1 \
--decouple_beta 0.05 \
--reverse_scheduled_sampling 1 \
--r_sampling_step_1 25000 \
--r_sampling_step_2 50000 \
--r_exp_alpha 2500 \
--lr 1e-4 \
--batch_size 3 \
--test_batch_size 9 \
--max_iterations 20000 \
--display_interval 1000 \
--test_interval 10 \
--snapshot_interval 2000 \
--conv_on_input 0 \
--res_on_conv 0 \
--curr_best_mse 0.03"
    
process = subprocess.run(cmd, shell=True)