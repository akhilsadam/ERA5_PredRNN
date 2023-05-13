import os, importlib, numpy as np, subprocess, sys
# change these params
training=False
train = ['PDE_5af2c49b-0070-459b-845e-568f699c24e5',]
valid = ['PDE_f7c4a4f2-8320-44d0-9262-0e09bc50eae7',]
pretrain_name='model_16000.ckpt'
###############################################


user=os.popen('whoami').read().replace('\n','')
userparam=importlib.import_module(f'user.{user}_param')

if userparam.param['WSL']:
    os.environ['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib'

datadir = userparam.param['data_dir']
checkpoint_dir = userparam.param['model_dir']
os.makedirs(checkpoint_dir, exist_ok=True)

train_data_paths = ','.join([f"{datadir}/{tr}/data.npz" for tr in train])
valid_data_paths = ','.join([f"{datadir}/{vd}/data.npz" for vd in valid])
test_path = f"{datadir}/test_{valid[0]}/"

dat = np.load(f"{datadir}/{train[0]}/data.npz")
shp = dat['dims'][0]
l = dat['input_raw_data'].shape[0]
param = importlib.import_module('param',f"{datadir}/{train[0]}")

if 'year' in param.data:
    img_channel = '3'
    img_layers = '0,1,2'
    input_length = '24'
    total_length = '48'
    layer_need_enhance = '1'
    patch_size = '40'
    num_hidden = '480,480,480,480,480,480'
    lr = '1e-4'
else:
    img_channel = '1' # we only have one channel for PDE data
    img_layers = '0'
    input_length = '2' # should depend on n of timesteps - exactly how?
    total_length = '4' # twice the input length (why?)
    layer_need_enhance = '0' # not sure what the enhancement is on that variable - some sort of renormalization..
    patch_size = '64' # divides the image l,w - breaks it into patches.
    num_hidden = '64,64,64,64,64,64' # number of hidden units in each layer
    lr = '1e-3' # learning rate

if training:
    save = f"--save_dir {checkpoint_dir}"
    concurrency = '--save_dir {checkpoint_dir}'
    train_int = '1'
    batch = '3'
    test_batch = '9'
else:
    concurrency = '--concurent_step 1' # not sure what this does - seems to step and update tensors at the same time (unsure if this works given comment)
    train_int = '0'
    batch = '3'
    test_batch = '3'

if not pretrain_name:
    pretrained = ''
else:
    print('Using pretrained model')
    pretrained =f'--pretrained_model {checkpoint_dir} ' + \
    f'--pretrained_model_name {pretrain_name} '
    
print('Data Dims:',shp)

cmd = f"python3 -u ../predrnn-pytorch/run2.py \
--is_training {train_int} \
{concurrency} \
--device cuda:0 \
--dataset_name mnist \
--train_data_paths {train_data_paths} \
--valid_data_paths {valid_data_paths} \
{save} \
--gen_frm_dir {checkpoint_dir} \
--model_name predrnn_v2 \
--reverse_input 0 \
--is_WV 0 \
--press_constraint 0 \
--center_enhance 0 \
--patch_size {patch_size} \
--weighted_loss 1 \
--upload_run 1 \
--layer_need_enhance {layer_need_enhance} \
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
--num_hidden {num_hidden} \
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
--lr {lr} \
--batch_size {batch} \
--test_batch_size {test_batch} \
--max_iterations 20000 \
--display_interval 1000 \
--test_interval 10 \
--snapshot_interval 2000 \
--conv_on_input 0 \
--res_on_conv 0 \
--curr_best_mse 0.03 \
{pretrained}"

process = subprocess.run(cmd, shell=True)