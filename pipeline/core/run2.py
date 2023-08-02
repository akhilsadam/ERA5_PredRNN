# import tensorflow as tf
# tf.config.list_physical_devices('GPU')
import os,sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import shutil
import argparse
import numpy as np
import math
from core.data_provider import datasets_factory
from core.utils import preprocess
import core.trainer2 as trainer
import pywt as pw
import torch
import torch.nn as nn
import random
import wandb

from scipy import ndimage

def center_enhance(img, min_distance = 100, sigma=4, radii=np.arange(0, 20, 2),find_max=True,enhance=True,multiply=2):
    if enhance:
        filter_blurred = ndimage.gaussian_filter(img,1)
        res_img = img + 30*(img - filter_blurred)
    else:
        res_img = ndimage.gaussian_filter(img,3)
    return res_img

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# -----------------------------------------------------------------------------
class run2:
    def make_parser():
        parser = argparse.ArgumentParser(description='PyTorch video prediction model - PredRNN')

        # more params on model selection (added by Akhil)
        parser.add_argument('--multigpu', type=bool, default=False)

        # option to save test output
        parser.add_argument('--save_output', type=str2bool, default=False)
        # option to save test output (number of frames)
        parser.add_argument('--test_iterations', type=int, default=1)

        ##
        # training/test
        parser.add_argument('--is_training', type=int, default=1)
        parser.add_argument('--device', type=str, default='cpu:0')
        parser.add_argument('--test_batch_size', type=int, default=15)

        # data
        parser.add_argument('--dataset_name', type=str, default='mnist')
        parser.add_argument('--train_data_paths', type=str, default='data/moving-mnist-example/moving-mnist-train.npz')
        parser.add_argument('--valid_data_paths', type=str, default='data/moving-mnist-example/moving-mnist-valid.npz')
        parser.add_argument('--save_dir', type=str, default='')
        parser.add_argument('--gen_frm_dir', type=str, default='results/mnist_predrnn')
        parser.add_argument('--input_length', type=int, default=10)
        parser.add_argument('--total_length', type=int, default=20)
        parser.add_argument('--img_width', type=int, default=64)
        parser.add_argument('--img_height', type=int, default=64)
        parser.add_argument('--img_channel', type=int, default=1)
        parser.add_argument('--img_layers', type=str, default='1')
        parser.add_argument('--concurent_step', type=int, default=1)
        parser.add_argument('--use_weight', type=int, default=0)
        parser.add_argument('--layer_weight', type=str, default='1,1,1')
        parser.add_argument('--skip_time', type=int, default=1)
        parser.add_argument('--wavelet', type=str, default='db1')

        #center enhancement
        parser.add_argument('--center_enhance', type=int, default=0)
        parser.add_argument('--layer_need_enhance', type=int, default=0)
        parser.add_argument('--find_max', type=str2bool, default=True)
        parser.add_argument('--multiply', type=float, default=1.0)

        # model
        parser.add_argument('--model_name', type=str, default='predrnn')
        parser.add_argument('--pretrained_model', type=str, default='')
        parser.add_argument('--pretrained_model_name', type=str, default='model_best.ckpt')
        parser.add_argument('--num_hidden', type=str, default='64,64,64,64')
        parser.add_argument('--filter_size', type=int, default=5)
        parser.add_argument('--stride', type=int, default=1)
        parser.add_argument('--patch_size', type=int, default=4)
        parser.add_argument('--patch_size1', type=int, default=4)
        parser.add_argument('--layer_norm', type=int, default=1)
        parser.add_argument('--decouple_beta', type=float, default=0.1)

        # reverse scheduled sampling
        parser.add_argument('--reverse_scheduled_sampling', type=int, default=0)
        parser.add_argument('--r_sampling_step_1', type=float, default=25000)
        parser.add_argument('--r_sampling_step_2', type=int, default=50000)
        parser.add_argument('--r_exp_alpha', type=int, default=5000)
        # scheduled sampling
        parser.add_argument('--scheduled_sampling', type=int, default=1)
        parser.add_argument('--sampling_stop_iter', type=int, default=50000)
        parser.add_argument('--sampling_start_value', type=float, default=1.0)
        parser.add_argument('--sampling_changing_rate', type=float, default=0.00002)

        # optimization
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--reverse_input', type=int, default=1)
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--max_iterations', type=int, default=80000)
        parser.add_argument('--display_interval', type=int, default=100)
        parser.add_argument('--test_interval', type=int, default=5000)
        parser.add_argument('--snapshot_interval', type=int, default=5000)
        parser.add_argument('--num_save_samples', type=int, default=10)
        parser.add_argument('--n_gpu', type=int, default=1)

        # visualization of memory decoupling
        parser.add_argument('--visual', type=int, default=0)
        parser.add_argument('--visual_path', type=str, default='./decoupling_visual')

        # action-based predrnn
        parser.add_argument('--injection_action', type=str, default='concat')
        parser.add_argument('--conv_on_input', type=int, default=0, help='conv on input')
        parser.add_argument('--res_on_conv', type=int, default=0, help='res on conv')
        parser.add_argument('--num_action_ch', type=int, default=4, help='num action ch')

        # new era5 module
        parser.add_argument('--is_static', type=int, default=0)
        parser.add_argument('--is_scale', type=int, default=0)
        parser.add_argument('--out_scale1', type=str, default='')
        parser.add_argument('--out_scale2', type=str, default='')
        parser.add_argument('--in_scale1', type=str, default='')
        parser.add_argument('--in_scale2', type=str, default='')
        parser.add_argument('--noise_val', type=float, default=0)
        parser.add_argument('--out_channel', type=int, default=5)
        parser.add_argument('--stat_layers', type=int, default=8)
        parser.add_argument('--stat_layers2', type=int, default=5)
        parser.add_argument('--out_weights', type=str, default='')
        parser.add_argument('--curr_best_mse', type=float, default=1e5)
        parser.add_argument('--isloss', type=int, default=1)
        parser.add_argument('--is_logscale', type=int, default=0)
        parser.add_argument('--is_WV', type=int, default=0)
        parser.add_argument('--press_constraint', type=int, default=1)
        parser.add_argument('--weighted_loss', type=int, default=1)
        parser.add_argument('--display_press_mean', type=int, default=1)
        parser.add_argument('--upload_run', type=int, default=1)
        parser.add_argument('--project', type=str, default='PC_PredRNN')
        parser.add_argument('--opt', type=str, default='Adam')
        parser.add_argument('--save_best_name', type=str, default='best_mse')
        parser.add_argument('--gpu_num', type=int, default=3)
        parser.add_argument('--time_step', type=str, default='')
        
        return parser


    def main(args):
        print(args)
        if args.multigpu:
            from core.models.model_factory_multiGPU import Model
        else:
            args.gpu_num = int(args.device.split(':')[1])
            torch.cuda.set_device(args.gpu_num)
            from core.models.model_factory import Model


        def reserve_schedule_sampling_exp(itr):
            if itr < args.r_sampling_step_1:
                r_eta = 0.5
            elif itr < args.r_sampling_step_2:
                r_eta = 1.0 - 0.5 * math.exp(-float(itr - args.r_sampling_step_1) / args.r_exp_alpha)
            else:
                r_eta = 1.0

            if itr < args.r_sampling_step_1:
                eta = 0.5
            elif itr < args.r_sampling_step_2:
                eta = 0.5 - (0.5 / (args.r_sampling_step_2 - args.r_sampling_step_1)) * (itr - args.r_sampling_step_1)
            else:
                eta = 0.0

            r_random_flip = np.random.random_sample(
                (args.batch_size, args.input_length - 1))
            r_true_token = (r_random_flip < r_eta)

            random_flip = np.random.random_sample(
                (args.batch_size, args.total_length - args.input_length - 1))
            true_token = (random_flip < eta)

            real_input_flag = []
            for i in range(args.batch_size):
                for j in range(args.total_length - 2):
                    if j < args.input_length - 1:
                        if r_true_token[i, j]:
                            real_input_flag.append(1)
                        else:
                            real_input_flag.append(0)
                    else:
                        if true_token[i, j - (args.input_length - 1)]:
                            real_input_flag.append(1)
                        else:
                            real_input_flag.append(0)
            
            real_input_flag = np.array(real_input_flag)
            real_input_flag = np.reshape(real_input_flag, (args.batch_size, args.total_length - 2, 1, 1, 1))
            return real_input_flag


        def schedule_sampling(eta, itr):
            zeros = np.zeros((args.batch_size, args.total_length - args.input_length - 1, 1, 1, 1))
            if not args.scheduled_sampling:
                return 0.0, zeros

            if itr < args.sampling_stop_iter:
                eta -= args.sampling_changing_rate
            else:
                eta = 0.0
            random_flip = np.random.random_sample(
                (args.batch_size, args.total_length - args.input_length - 1))
            true_token = (random_flip < eta)
        
            real_input_flag = []
            for i in range(args.batch_size):
                for j in range(args.total_length - args.input_length - 1):
                    if true_token[i, j]:
                        real_input_flag.append(1)
                    else:
                        real_input_flag.append(0)
            real_input_flag = np.array(real_input_flag)
            real_input_flag = np.reshape(real_input_flag, (args.batch_size, args.total_length - args.input_length - 1, 1, 1, 1))
            return eta, real_input_flag



        def train_wrapper(model):
            torch.cuda.empty_cache()
            if args.pretrained_model and os.path.exists(args.pretrained_model):
                model.load(args.pretrained_model)
            model.network.train()
            real_input_flag = np.zeros((args.batch_size, args.total_length-1-1, 1, 1, 1))
            real_input_flag[:, :args.input_length - 1, :, :] = 1.0
            train_data_files = args.train_data_paths.split(',')
            for file in train_data_files:
                print(file)
            if len(train_data_files) > 0:
                print(f"train_data_files nums: {len(train_data_files)}")
                eta = args.sampling_start_value
                random.shuffle(train_data_files)
                curr_pos = 0
                curr_train_path = train_data_files[curr_pos]
                test_input_handle = datasets_factory.data_provider(
                            args.dataset_name, curr_train_path, 
                            args.valid_data_paths, 
                            args.test_batch_size, args.img_height, 
                            args.img_width,
                            seq_length=args.total_length, 
                            injection_action=args.injection_action, 
                            concurent_step=args.concurent_step,
                            img_channel=args.img_channel, img_layers=args.img_layers,
                            is_testing=True, is_training=False, is_WV=args.is_WV)
                train_input_handle = datasets_factory.data_provider(
                            args.dataset_name, curr_train_path, 
                            args.valid_data_paths, 
                            args.batch_size, args.img_height, 
                            args.img_width,
                            seq_length=args.total_length, 
                            injection_action=args.injection_action, 
                            concurent_step=args.concurent_step,
                            img_channel=args.img_channel, img_layers=args.img_layers,
                            is_testing=False, is_training=True, is_WV=args.is_WV)
                for itr in range(1, args.max_iterations + 1):
                    if train_input_handle.no_batch_left():
                        if curr_pos < len(train_data_files)-1:
                            curr_pos += 1
                        else:
                            curr_pos = 0
                        curr_train_path = train_data_files[curr_pos]
                        print(curr_train_path)
                        #curr_train_path = ','.join(listi)
                        train_input_handle = datasets_factory.data_provider(
                            args.dataset_name, curr_train_path, 
                            args.valid_data_paths, 
                            args.batch_size, args.img_height, 
                            args.img_width,
                            seq_length=args.total_length, 
                            injection_action=args.injection_action, 
                            concurent_step=args.concurent_step,
                            img_channel = args.img_channel,img_layers = args.img_layers,
                            is_testing=False,is_training=True,is_WV=args.is_WV)
                    
                    for i in range(model.accumulate_batch):
                        ims = train_input_handle.get_batch()
                        ims = ims[:,:,:args.img_channel,:,:]
                        
                        # if args.reverse_scheduled_sampling == 1:
                        #     real_input_flag = reserve_schedule_sampling_exp(itr)
                        # else:
                        #     eta, real_input_flag = schedule_sampling(eta, itr)
                        lossargs = trainer.train(model, ims, real_input_flag, args, itr)
                    trainer.update(model, *lossargs)
                    print(f"Iteration: {itr}, ims.shape: {ims.shape}")    
                    
                    if itr % args.snapshot_interval == 0:
                        model.save(itr)

                    if itr % args.test_interval == 0:
                        test_input_handle.begin(do_shuffle=False)
                        test_err = trainer.test(model, test_input_handle, args, 'test_result')
                        print('current test mse: '+str(np.round(test_err,6)))
                        if test_err < args.curr_best_mse:
                            print(f'At step {itr}, Best test: '+str(np.round(test_err,6)))
                            args.curr_best_mse = test_err
                            model.save(args.save_best_name)
                    train_input_handle.next()
                        

        def test_wrapper(model):
            model.load(args.pretrained_model)
            test_input_handle = datasets_factory.data_provider(
                args.dataset_name, args.train_data_paths, args.valid_data_paths, args.batch_size, args.img_height, args.img_width,
                seq_length=args.total_length, injection_action=args.injection_action, concurent_step=args.concurent_step,
                img_channel = args.img_channel,img_layers = args.img_layers,
                is_testing=True,is_training=False,is_WV=args.is_WV)
            test_err = trainer.test(model, test_input_handle, args, 'test_result')
            print(f"The test mse is {test_err}")


        lat = torch.linspace(-np.pi/2, np.pi/2, args.img_height+1)
        lat = (lat[1:] + lat[:-1])*0.5
        cos_lat = torch.reshape(torch.cos(lat), (-1,1))
        args.area_weight = (cos_lat*720/torch.sum(cos_lat)).to(args.device)
        
        if not args.weather_prediction:
            args.area_weight = torch.ones_like(args.area_weight).to(args.device)

        # save_file = ['WV', str(args.is_WV), 'PC', str(args.press_constraint), 'EH', str(args.center_enhance), 'PS', str(args.patch_size)]
        # args.save_file = '_'.join(save_file)
        if 'save_file' not in args:
            args.save_file = ''
        if args.time_step:
            args.save_file = '_'.join((args.save_file, args.time_step))
        
        # run_name = ['bs', str(1), 'opt', args.opt, 'lr', str(args.lr), 'lr_sch', 'no', args.time_step]
        # run_name = [args.optim_lm.__name__, 'lr', str(args.lr), 'lr_sch', 'no', args.time_step]
        args.run_name = '' #'_'.join(run_name)

        len_nh = len([int(x) for x in args.num_hidden.split(',')])
        print(f"model.num_hidden length: {len_nh}")
        # if len_nh != 4:
        #     args.save_file = '_'.join((args.save_file, str(len_nh)))
        #     args.run_name = '_'.join((args.run_name, str(len_nh)))
        print(f"args.run_name: {args.run_name}")
        if args.save_dir:
            args.save_dir = os.path.join(args.save_dir, args.save_file)
            if not os.path.exists(args.save_dir):
                # shutil.rmtree(args.save_dir)
                os.makedirs(args.save_dir)
                print('Created:', args.save_dir)

        if args.pretrained_model and os.path.exists(args.pretrained_model):
            args.pretrained_model = os.path.join(args.pretrained_model, args.save_file, args.pretrained_model_name)
            print(f"pretrained_model: {args.pretrained_model}")

        args.gen_frm_dir = os.path.join(args.gen_frm_dir, args.save_file)
        if not os.path.exists(args.gen_frm_dir):
            # shutil.rmtree(args.gen_frm_dir)
            os.makedirs(args.gen_frm_dir)
            print('Created:', args.gen_frm_dir)

        print('Initializing models')
        model = Model(args)
        # model= nn.DataParallel(model, device_ids=[0, 1, 2])
        #model.to(args.device)







        if args.is_training:
            train_wrapper(model)
        else:
            test_wrapper(model)
        
        model.finish()


if __name__ == '__main__':
    parser = run2.make_parser()
    args = parser.parse_args()
    run2.main(args)