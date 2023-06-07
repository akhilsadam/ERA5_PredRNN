import os, importlib, numpy as np, subprocess, sys
# change these params
training=False
train = ['PDE_1b0ed7df-0651-4f9a-85a5-4c6a6b534898','PDE_55fbd98b-cd2f-4045-ab3f-3af6ddae531b',
         'PDE_14a60c9a-6f59-43cc-9519-c45bd8720cc2','PDE_ccfa2cb4-6622-4a57-9b02-68bb00696a4c',
         'PDE_b6893217-e82f-4ef1-8dfd-5531a19f7522','PDE_fcd3a352-00a1-4a8f-b3ee-c79846497236']
valid = ['PDE_063698fc-15d0-43d2-9098-8da521dd6b4c',]
pretrain_name='model_best_mse.ckpt'
save_test_output=True
###############################################
model_name = 'BERT' # [BERT, predrnn_v2]
model_config = \
    {
        'BERT':{
            'n_layers': 6, # number of layers in the transformer
            'n_head': 8, # number of heads in the transformer
            'n_embd': 512, # number of hidden units in the transformer
            'dropout': 0.1, # dropout rate
            'initialization': None, # initialization method as list of functions (None uses default for FFN RELU)
            'activation': 'relu', # activation function
        },
        # https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
        
    }
model_config_toy = \
    {
        'BERT':{
            'n_layers': 6, # number of layers in the transformer
            'n_head': 2, # number of heads in the transformer
            'n_embd': 8, # number of hidden units in the transformer
            'dropout': 0.0, # dropout rate
            'initialization': None, # initialization method as list of functions
            'activation': 'relu', # activation function
        },
        # https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
        
    }
# note predrnn_v2 does not work with any preprocessing or other options
###############################################
preprocessor_name = 'POD' # [raw, POD]
preprocessor_config = \
    {
        'POD':{
            'eigenvector': lambda var: f'POD_eigenvector_{var}.npy', # place to store precomputed eigenvectors in the data directory
            # (var is the variable name)
            'make_eigenvector': True, # whether to compute eigenvectors or not
            'max_n_eigenvectors': 100, # maximum number of eigenvectors (otherwise uses PVE to determine)
            'PVE_threshold': 0.99, # PVE threshold to determine number of eigenvectors
        },
    }
###############################################
########## DO NOT EDIT BELOW THIS LINE ########
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

weather_prediction = 'year' in param.data

if weather_prediction:
    
    img_channel = 3
    img_layers = '0,1,2'
    input_length = '24'
    total_length = '48'
    layer_need_enhance = '1'
    patch_size = '40'
    num_hidden = '480,480,480,480,480,480'
    lr = '1e-4'
    rss='1'
    test_iterations = 1; # number of test iterations to run
else:
    img_channel = 1 # we only have one channel for PDE data
    img_layers = '0'
    input_length = 20 #'2' # (length of input sequence?)
    total_length = 40 #'4' # (complete sequence length?)
    layer_need_enhance = '0' # not sure what the enhancement is on that variable - some sort of renormalization..
    patch_size = '1' # divides the image l,w - breaks it into patches that are FCN into the hidden layers (so each patch_size x patch_size -> # of hidden units).
    num_hidden = '8,8,8,8,8,8' # number of hidden units in each layer per patch (so 64**2 * 16 = 65536 parameters per layer, or 393216 parameters total) 
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
    test_iterations = 1; # number of test iterations to run
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

run2=importlib.import_module('../predrnn-pytorch/run2.py')

cmdargs = f"--is_training {train_int} \
--test_iterations {test_iterations} \
{concurrency} \
--device cuda:0 \
--dataset_name mnist \
--train_data_paths {train_data_paths} \
--valid_data_paths {valid_data_paths} \
{save} \
--save_output {save_test_output_arg} \
--gen_frm_dir {checkpoint_dir} \
--model_name {model_name} \
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

args = run2.parser.parse_args(cmdargs.split(' '))
if preprocessor_name is not 'raw':
    preprocessor_args = preprocessor_config[preprocessor_name]
    preprocessor_args['data_dir'] = datadir
    preprocessor_args['train_data_paths'] = train_data_paths
    preprocessor_args['train_data_paths'] = train_data_paths
    preprocessor_args['n_var'] = img_channel
    preprocessor_args['shapex'] = shp[1]
    preprocessor_args['shapey'] = shp[2]
    args.preprocessor = \
        importlib.import_module(f'../predrnn-pytorch/core/preprocessors/{preprocessor_name}.py') \
        .Preprocessor(preprocessor_args)
if model_name is not 'predrnn_v2':
    if weather_prediction:
        model_args = model_config[model_name]
    else:
        model_args = model_config_toy[model_name]
    args.model = model_args
run2.main(args)