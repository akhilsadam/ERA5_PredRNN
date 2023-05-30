import os, importlib, numpy as np, subprocess, sys
# change these params
training=True
train = ['PDE_1a6ffcea-c787-454b-b5e4-936ddffaca5c','PDE_7e6be5b4-4b72-46a0-9a5b-2b8ef47ba7af',
         'PDE_8fd4a2c2-4024-41ea-b042-c5a9d5a7b4a4','PDE_6514bad1-7bb8-4e8d-ae26-591672875882',
         'PDE_c5684c0f-60c5-4b1c-85bf-f7d9a44c5f4d','PDE_c5403523-1954-49d1-947f-b1ca9c60096a']
valid = ['PDE_fd2f7f76-f3c5-48f6-bdeb-9f2b676d5d49',]
pretrain_name=None #'model_20000.ckpt'
save_test_output=False #True
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
    rss='1'
    test_iterations = 0; # number of test iterations to run
else:
    img_channel = '1' # we only have one channel for PDE data
    img_layers = '0'
    input_length = 20 #'2' # (length of input sequence?)
    total_length = 40 #'4' # (complete sequence length?)
    layer_need_enhance = '0' # not sure what the enhancement is on that variable - some sort of renormalization..
    patch_size = '1' # divides the image l,w - breaks it into patches that are FCN into the hidden layers (so each patch_size x patch_size -> # of hidden units).
    num_hidden = '16,16,16,16,16,16' # number of hidden units in each layer per patch (so 64**2 * 16 = 65536 parameters per layer, or 393216 parameters total) 
    # (use 64 if you want 1.5M parameters-this is similar to 1.8M on the full problem)
    lr = '1e-3' # learning rate
    rss = '0' # reverse scheduled sampling - turning it off for now
    test_iterations = 1; # number of test iterations to run

if training:
    save = f"--save_dir {checkpoint_dir}"
    concurrency = f'--save_dir {checkpoint_dir}'
    train_int = '1'
    batch = '3'
    test_batch = '9'
    test_iterations = 0; # number of test iterations to run
else:
    save = ''
    concurrency = '--concurent_step 1' # not sure what this does - seems to step and update tensors at the same time (unsure if this works given comment)
    train_int = '0'
    batch = '3'
    test_batch = '3'
    test_iterations = 100; # number of test iterations to run

if pretrain_name is None:
    pretrained = ''
else:
    print('Using pretrained model')
    pretrained =f'--pretrained_model {checkpoint_dir} ' + \
    f'--pretrained_model_name {pretrain_name} '

save_test_output_arg = 'True' if save_test_output and not training else 'False'

print('Data Dims:',shp)

cmd = f"python3 -u ../predrnn-pytorch/run2.py \
--is_training {train_int} \
--test_iterations {test_iterations} \
{concurrency} \
--device cuda:0 \
--dataset_name mnist \
--train_data_paths {train_data_paths} \
--valid_data_paths {valid_data_paths} \
{save} \
--save_output {save_test_output_arg} \
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
--reverse_scheduled_sampling {rss} \
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