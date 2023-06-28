import copy
import os, importlib, numpy as np, subprocess, sys, logging
from adabound import AdaBound as adb
logger = logging.getLogger(__name__)
from torch.optim import ASGD, Adam, SGD
from torch.optim.lr_scheduler import CyclicLR
###############################################
model_config = \
    {
        'TF':{
            'n_encoder_layers': 8, # number of layers in the encoder
            'n_decoder_layers': 8, # number of layers in the decoder
            'n_head': 8, # number of heads in the transformer
            'n_embd': 512, # number of hidden units in the transformer
            'n_ffn_embd': 2048, # number of hidden units in the FFN
            'dropout': 0.1, # dropout rate
            'dropout_pos_enc': 0.05, # dropout rate for positional encoding
            'initialization': None, # initialization method as list of functions (None uses default for FFN RELU)
            'activation': 'relu', # activation function
            'optimizer' :  lambda x,y : ASGD(x,lr=y) # [None, Adam, ASGD,...]'
        },
        # https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
        'DNN':{
            'hidden': [], # number of hidden units for all layers in sequence
            'initialization': None, # initialization method as list of functions
            'activation': 'relu', # activation function
            'optimizer' :  lambda x,y : ASGD(x,lr=y) # [None, Adam, ASGD,...]'
        },

    }
model_config_toy = \
    {# note base learning rate is 1e-3 for all models (denoted by y in the optimizer)
        'TF':{
            'n_encoder_layers': 2, # number of layers in the encoder
            'n_decoder_layers': 2, # number of layers in the decoder
            'n_head': 2, # number of heads in the transformer
            'n_embd': 32, # number of hidden units in the transformer
            'n_ffn_embd': 100, # number of hidden units in the FFN
            'dropout': 0.05, # dropout rate
            'dropout_pos_enc': 0.05, # dropout rate for positional encoding
            'initialization': None, # initialization method as list of functions
            'activation': 'relu', # activation function
            'optimizer' :  lambda x,y : ASGD(x,lr=100*y) # [None, Adam, ASGD,...]'
        },
        'TF_POD':{
            'n_encoder_layers': 2, # number of layers in the encoder
            'n_decoder_layers': 2, # number of layers in the decoder
            'n_head': 2, # number of heads in the transformer
            'n_embd': 32, # number of hidden units in the transformer
            'n_ffn_embd': 100, # number of hidden units in the FFN
            'dropout': 0.05, # dropout rate
            'dropout_pos_enc': 0.05, # dropout rate for positional encoding
            'initialization': None, # initialization method as list of functions
            'activation': 'relu', # activation function
            'optimizer' :  lambda x,y : ASGD(x,lr=50*y) # [None, Adam, ASGD,...]'
        },
        'DualAttentionTransformer':{
            'windows': [[16,16],[2,2],[2,2]], # list of window sizes for the shifted attention
            'shifts': [[0,0],[0,0],[1,1]], # list of shifts for the shifted attention
            'n_head': 1, # number of heads in the transformer
            'n_embd': 4096, # number of hidden units in the transformer
            'n_ffn_embd': 4096, # number of hidden units in the FFN
            'dropout': 0.1, # dropout rate
            'initialization': None, # initialization method as list of functions
            'activation': 'relu', # activation function
            'optimizer' :  lambda x,y : Adam(x, lr=5e-5), # final_lr=0.1), #SGD(x, lr=0.4),#, momentum=0.1, nesterov=True), #ASGD(x,lr=100*y), # [None, Adam, ASGD,...]'
            'scheduler' : lambda x : CyclicLR(x, base_lr=5e-6, max_lr=5e-4, cycle_momentum=False, step_size_up=20),
            'batch_size': 16, # batch size
        },
        'BERT':{
            'n_layers': 4, # number of layers in the transformer
            'n_head': 1, # number of heads in the transformer
            'n_embd': 4096, # number of hidden units in the transformer
            'n_ffn_embd': 4096, # number of hidden units in the FFN
            'dropout': 0.1, # dropout rate
            'initialization': None, # initialization method as list of functions
            'activation': 'relu', # activation function
            'optimizer' :  lambda x,y : Adam(x, lr=1e-5), # final_lr=0.1), #SGD(x, lr=0.4),#, momentum=0.1, nesterov=True), #ASGD(x,lr=100*y), # [None, Adam, ASGD,...]'
            'scheduler' : lambda x : CyclicLR(x, base_lr=5e-6, max_lr=2e-5, cycle_momentum=False, step_size_up=20),
            'batch_size': 16, # batch size
        },
        'BERT_POD':{
            'n_layers': 6, # number of layers in the transformer
            'n_head': 2, # number of heads in the transformer
            'n_embd': 8, # number of hidden units in the transformer
            'n_ffn_embd': 8, # number of hidden units in the FFN
            'dropout': 0.01, # dropout rate
            'initialization': None, # initialization method as list of functions
            'activation': 'relu', # activation function
            'optimizer' :  lambda x,y : ASGD(x,lr=100*y), # [None, Adam, ASGD,...]'
            'batch_size': 9, # batch size
        },
        'rBERT':{
            'n_layers': 6, # number of layers in the transformer
            'n_head': 4, # number of heads in the transformer
            'n_embd': 8, # number of hidden units in the transformer
            'n_ffn_embd': 8, # number of hidden units in the FFN
            'dropout': 0.01, # dropout rate
            'initialization': None, # initialization method as list of functions
            'activation': 'relu', # activation function
            'optimizer' :  lambda x,y : Adam(x, lr=1e-5), # [None, Adam, ASGD,...]'
            'scheduler' : lambda x : CyclicLR(x, base_lr=5e-6, max_lr=2e-5, cycle_momentum=False, step_size_up=20),
            'batch_size': 3, # batch size
            'nstep': 8,
        },
        # https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
        'DNN':{
            'hidden': [320], # number of hidden units for all layers in sequence
            'initialization': None, # initialization method as list of functions
            'activation': 'relu', # activation function
            'optimizer' :  lambda x,y : ASGD(x,lr=100*y) # [None, Adam, ASGD,...]'
        },
        'adaptDNN':{
            'hidden': [], # number of hidden units for all layers in sequence
            'initialization': None, # initialization method as list of functions
            'activation': 'relu', # activation function
            'optimizer' :  lambda x,y : ASGD(x,lr=500*y), # [None, Adam, ASGD,...]'
            'nstep': 8,
        },
        'reZeroTF':{
            'n_layers': 6, # number of layers in the transformer
            'n_head': 2, # number of heads in the transformer
            'n_embd': 8, # number of hidden units in the transformer
            'n_ffn_embd': 8, # number of hidden units in the FFN
            'dropout': 0.0, # dropout rate
            'initialization': None, # initialization method as list of functions
            'activation': 'relu', # activation function
            'optimizer' :  lambda x,y : ASGD(x,lr=500*y), # [None, Adam, ASGD,...]'
            'batch_size': 9, # batch size
        },
        'LSTM':{
            'n_layers': 8, # number of layers 
            'n_embd': 8, # number of hidden units
            'dropout': 0.0, # dropout rate
            'optimizer' :  lambda x,y : Adam(x,lr=1e-3), # [None, Adam, ASGD,...]'
            # 'scheduler' : lambda x : CyclicLR(x, base_lr=1e-6, max_lr=5e-5, cycle_momentum=False, step_size_up=20),
            'batch_size': 9, # batch size
        },
        'rLSTM':{
            'n_layers': 6, # number of layers 
            'n_embd': 8, # number of hidden units
            'dropout': 0.0, # dropout rate
            'optimizer' :  lambda x,y : Adam(x,lr=y), # [None, Adam, ASGD,...]'
            'batch_size': 9, # batch size
        },
        'ViT_LDM':{
            'n_layers': 6, # number of layers
            'n_head': 2,
            'n_ffn_embd': 2000,
            'att_embd': 32, # matrix attention dim
            'n_latent_encode': 640, # latent dim for encoder
            'n_embd': 8, # number of hidden units
            'diffusion_steps': 1000,
            'dropout': 0.0, # dropout rate
            'activation': 'relu',
            'initialization': None,
            'optimizer' : lambda x,y : Adam(x, lr=5e-5), # [None, Adam, ASGD,...]'
            # 'scheduler' : lambda x : CyclicLR(x, base_lr=1e-5, max_lr=5e-4, cycle_momentum=False, step_size_up=20),
            'batch_size': 9, # batch size
        },
    }
# note predrnn_v2 does not work with any preprocessing or other options
###############################################
preprocessor_config = \
    {
        'POD':{
            'eigenvector': lambda var: f'POD_eigenvector_{var}.npz', # place to store precomputed eigenvectors in the data directory
            # (var is the variable name)
            'make_eigenvector': False, # whether to compute eigenvectors or not (only needs to be done once)
            'max_n_eigenvectors': 1000, # maximum number of eigenvectors (otherwise uses PVE to determine)
            'PVE_threshold': 0.99, # PVE threshold to determine number of eigenvectors
        },
        'control':{
        },
    }
    
###############################################


def operate_loop(hyp):
    ########## DO NOT EDIT BELOW THIS LINE ########
    ###############################################
    user=os.popen('whoami').read().replace('\n','')
    userparam=importlib.import_module(f'user.{user}_param')

    if userparam.param['WSL']:
        os.environ['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib'

    datadir = os.path.abspath(userparam.param['data_dir'])
    checkpoint_dir = f"{userparam.param['model_dir']}/{hyp.model_name}/{hyp.preprocessor_name}/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    data_key = 'PDE' if not hyp.weather_prediction else 'CDS'
    ###############################################
    datasets = [
        i
        for i in os.listdir(datadir)
        if not os.path.isfile(os.path.join(datadir, i)) and data_key in i
    ]
    logger.info(f'Found {len(datasets)} datasets: {datasets}')
    assert len(datasets) > hyp.n_valid, logger.critical('Insufficient number of datasets found (cannot train)')
    datasets.sort()
    train = datasets[:-hyp.n_valid]
    valid = datasets[-hyp.n_valid:]
    ###############################################
    train_data_paths = ','.join([f"{datadir}/{tr}/data.npz" for tr in train])
    valid_data_paths = ','.join([f"{datadir}/{vd}/data.npz" for vd in valid])
    # test_path = f"{datadir}/test_{valid[0]}/"

    dat = np.load(f"{datadir}/{train[0]}/data.npz")
    shp = dat['dims'][0]
    l = dat['input_raw_data'].shape[0]
    param = importlib.import_module('param',f"{datadir}/{train[0]}")

    if hyp.weather_prediction:

        img_channel = 3
        img_layers = '0,1,2'
        input_length = '24'
        total_length = '48'
        layer_need_enhance = '1'
        patch_size = '40'
        num_hidden = '480,480,480,480,480,480'
        lr = '1e-4'
        rss='1'; # number of test iterations to run
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
        rss = '0' # reverse scheduled sampling - turning it off for now; # number of test iterations to run

    test_iterations = 1
    if hyp.training:
        save = f"--save_dir {checkpoint_dir}"
        concurrency = f'--save_dir {checkpoint_dir}'
        train_int = '1'
        test_batch = '9'
        test_iterations = 100; # number of test iterations to run
    else:
        save = ''
        concurrency = '--concurent_step 1' # not sure what this does - keep it off for now
        train_int = '0'
        test_batch = '9'
        test_iterations = 200; # number of test iterations to run

    batch = '3'
    if hyp.pretrain_name is None:
        pretrained = ''
    else:
        print('Using pretrained model')
        pretrained =f'--pretrained_model {checkpoint_dir} ' + \
        f'--pretrained_model_name {hyp.pretrain_name} '

    save_test_output_arg = 'True' if hyp.save_test_output and not hyp.training else 'False'

    print('Data Dims:',shp)

    from core.run2 import run2


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
    --model_name {hyp.model_name} \
    --reverse_input 0 \
    --is_WV 0 \
    --press_constraint 0 \
    --center_enhance 0 \
    --patch_size {patch_size} \
    --weighted_loss 1 \
    --upload_run 1 \
    --project {hyp.project_name} \
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
    --max_iterations {hyp.max_iterations} \
    --display_interval 1000 \
    --test_interval 10 \
    --snapshot_interval 500 \
    --conv_on_input 0 \
    --res_on_conv 0 \
    --curr_best_mse 0.03 \
    {pretrained}"

    parser = run2.make_parser()
    args = parser.parse_args(cmdargs.split())
    if hyp.preprocessor_name != 'raw':
        preprocessor_args = copy.deepcopy(args.__dict__)
        preprocessor_args.update(preprocessor_config[hyp.preprocessor_name],allow_override=True)
        preprocessor_args['datadir'] = datadir
        preprocessor_args['train_data_paths'] = train_data_paths.split(',')
        preprocessor_args['valid_data_paths'] = valid_data_paths.split(',')
        preprocessor_args['n_var'] = img_channel
        preprocessor_args['shapex'] = shp[1]
        preprocessor_args['shapey'] = shp[2]
        args.preprocessor = \
            importlib.import_module(f'core.preprocessors.{hyp.preprocessor_name}') \
            .Preprocessor(preprocessor_args)
    args.preprocessor_name = hyp.preprocessor_name
    if hyp.model_name != 'predrnn_v2':
        cmodel_config = copy.deepcopy(args.__dict__)
        if hyp.weather_prediction:
            cmodel_config.update(model_config,allow_override=True)
        else:
            cmodel_config.update(model_config_toy,allow_override=True)
            
        nm = hyp.model_name + '_' + hyp.preprocessor_name
        # model_args = cmodel_config[nm] if nm in cmodel_config else cmodel_config[hyp.model_name] 
        if nm in cmodel_config:
            key = nm
        elif hyp.model_name in cmodel_config:
            key = hyp.model_name
        elif '_v' in hyp.model_name:
            hnm = hyp.model_name.split('_v')[0]
            nm = hnm + '_' + hyp.preprocessor_name
            if nm in cmodel_config:
                key = nm
            else:
                key = hnm
        
        model_args = cmodel_config[key]
            
        for k,v in hyp.overrides.items():
            if k in model_args:
                model_args[k] = v
        args.model_args = model_args
        args.optim_lm = model_args['optimizer']
        args.scheduler = model_args['scheduler'] if 'scheduler' in model_args else None
        args.batch_size = model_args['batch_size'] if 'batch_size' in model_args else args.batch_size
    args.weather_prediction = hyp.weather_prediction
    run2.main(args)