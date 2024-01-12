import os
import threading
import numpy as np
import torch
from torch.optim import Adam
from core.models import predrnn, predrnn_v2_adj, action_cond_predrnn, action_cond_predrnn_v2, \
    TF, DNN, adaptDNN, BERT, BERT_v2, BERT_v3, rBERT, RZTX, RZTX_CNN, RZTX_NAT, RZTX_SROM, RZTX_NAT_LG, RZTX_CNN_LG, LSTM, rLSTM, ViT_LDM, \
    DAT_v2, linint, identity, DMDNet, ComplexDMDNet, FPNet, GateLoop
from core.utils.ext import prefixprint
from torchview import draw_graph
import traceback, sys
import wandb, json
import gc
from accelerate import Accelerator, load_checkpoint_in_model

from core.utils import saliency
from core.viz.viz_salient import viz

os.environ["WANDB_SILENT"] = "true"

class Model(object):
    def __init__(self, configs):
        self.configs = configs
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        # self.num_layers = len(self.num_hidden)
        self.num_layers = configs.num_layers
        networks_map = {
            # 'predrnn': predrnn.RNN,
            'predrnn_v2': predrnn_v2_adj.RNN,
            # 'action_cond_predrnn': action_cond_predrnn.RNN,
            # 'action_cond_predrnn_v2': action_cond_predrnn_v2.RNN,
            'TF': TF.TF,
            'DNN': DNN.DNN,
            'adaptDNN': adaptDNN.adaptDNN,
            'BERT': BERT.BERT,
            'BERT_v2': BERT_v2.BERT,
            "BERT_v3": BERT_v3.BERT,
            "DualAttentionTransformer": DAT_v2.DAT,
            'rBERT': rBERT.rBERT,
            'reZeroTF': RZTX.RZTX,
            'reZeroCNN': RZTX_CNN.RZTX_CNN,
            'reZeroCNN_LG': RZTX_CNN_LG.RZTX_CNN,
            'reZeroNAT': RZTX_NAT.RZTX_NAT,
            'reZeroNAT_LG': RZTX_NAT_LG.RZTX_NAT_LG,
            'reZeroSROM': RZTX_SROM.RZTX,
            'LSTM': LSTM.LSTM,
            'rLSTM': rLSTM.rLSTM,
            'ViT_LDM': ViT_LDM.ViT_LDM,
            'linint': linint.LinearIntegrator, 
            'identity': identity.Identity,
            'DMDNet': DMDNet.DMDNet,
            'ComplexDMDNet':ComplexDMDNet.DMDNet,
            'FPNet':FPNet.FPNet,
            'GateLoop':GateLoop.GateLoop,
        }
        torch.backends.cuda.matmul.allow_tf32 = True
        # device = configs.device # this is plural if Accelerate is used
        # self.device = device
        
        self.start_itr = 0

        self.accelerator = Accelerator()
        device = self.accelerator.device
        self.device = self.accelerator.device
        self.configs.device = self.accelerator.device
        self.configs.preprocessor.device = self.accelerator.device
        self.configs.area_weight = self.configs.area_weight.to(device, non_blocking=True)
        
        thread = threading.current_thread().name
        name = configs.model_name
        self.print = prefixprint(level=1,n=80,tag=f"{device}:{thread}:{name}:").printfunction
    
        self.saliency = configs.interpret if hasattr(configs, 'interpret') else False
        self.accumulate_batch = configs.accumulate_batch if hasattr(configs, 'accumulate_batch') else 1

        if configs.model_name in networks_map:
            self.network_handle = networks_map[configs.model_name]
            self.init_net()
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)
        
        if 'predrnn' not in configs.model_name and self.saliency:#and 'pretrained_model' not in configs.__dict__:
            try:
                self.modelvis()
            except Exception as e:
                ex_type, ex_value, ex_traceback = sys.exc_info()
                trace_back = traceback.extract_tb(ex_traceback)
                self.print(f"Is graphviz installed? Could not visualize model: {e}")
                    # Format stacktrace
                stack_trace = []
                for trace in trace_back:
                    stack_trace.append("File : %s , Line : %d, Func.Name : %s, Message : %s" % (trace[0], trace[1], trace[2], trace[3]))
                self.print("Exception type : %s " % ex_type.__name__)
                self.print("Exception message : %s" %ex_value)
                self.print("Stack trace : %s" %stack_trace)
        
        
        self.optimizer = configs.optim_lm(self.network.parameters(), configs.lr) \
            if configs.optim_lm is not None else Adam(self.network.parameters(), lr=configs.lr)
        self.scheduler = configs.scheduler(self.optimizer) if configs.scheduler is not None else None
        self.network, self.optimizer, self.scheduler = self.accelerator.prepare(self.network, self.optimizer, self.scheduler)
        
        if self.configs.upload_run:
            self.upload_wandb()
    
    def init_net(self):                
        self.network = self.network_handle(self.num_layers, self.num_hidden, self.configs)
        # self.network = torch.nn.DataParallel(self.network,device_ids=[self.device,], output_device=self.device)
        # self.network.to(self.device)


    def modelvis(self):
        draw_graph(self.network, input_size= \
            (self.configs.batch_size,self.configs.total_length, \
            self.configs.img_channel,self.configs.img_height,self.configs.img_width), expand_nested=False, roll=True, save_graph=True, filename=self.configs.model_name, directory=self.configs.save_dir)  
        self.init_net()
        # model_graph.visual_graph.render(format='svg')
    
    def upload_wandb(self):
        # Uploading to wandb
        run_name = (
            f'{self.configs.model_name}_{self.configs.preprocessor_name}_'
            + ('train' if self.configs.is_training else 'test')
            + '_il_' + str(self.configs.input_length)
        
            # + '_'.join((self.configs.save_file, self.configs.run_name))
        )
        
        self.conf_dict = {
            'model_name': self.configs.model_name,
            'opt' : self.optimizer.__class__.__name__,
            'lr' : self.optimizer.param_groups[0]["lr"],
            'preprocessor' : self.configs.preprocessor_name,
        }
        
        self.conf_dict.update(self.configs.model_args)
        
        self.wrun = wandb.init(project=self.configs.project, name=run_name, allow_val_change=True, \
        config=self.conf_dict)
        
    def finish(self):
        self.wrun.finish()

    def save(self, itr):
        citr = itr + self.start_itr
        # stats = {}
        # stats['net_param'] = self.network.state_dict()
        checkpoint_path = os.path.join(self.configs.save_dir, 'model'+'_'+str(citr)+'.ckpt')
                
        # torch.save(stats, checkpoint_path)
        self.accelerator.wait_for_everyone()
        self.accelerator.save_model(self.network, checkpoint_path, max_shard_size="1GB")
        self.print("saved model to %s" % checkpoint_path)
        
        confile = checkpoint_path+'/config.json'
        def dumper(obj):
            try:
                return obj.toJSON()
            except:
                return obj.__dict__
        
        if not os.path.exists(confile):
            with open(confile, 'w') as f:
                json.dump(self.conf_dict, f, default=dumper, indent=4)
                
    def load(self, checkpoint_path):
        self.print('loading model from', checkpoint_path)
        
        try:
            chk_itr = int(checkpoint_path.split('_')[-1].split('.')[0])
        except Exception as e:
            self.start_itr = 0
            self.print(f"Could not get iteration from checkpoint path: {e}")
        else:
            self.start_itr = chk_itr
        # stats = torch.load(checkpoint_path, map_location=torch.device(self.device))
        #### self.print('model.transformer_encoder.layers.0.self_attn.in_proj_weight', stats['net_param']['model.transformer_encoder.layers.0.self_attn.in_proj_weight'])
        # self.network.load_state_dict(stats['net_param'])
        # unwrapped_model = self.accelerator.unwrap_model(self.network)
        # unwrapped_model.load_state_dict(torch.load(checkpoint_path))
        load_checkpoint_in_model(self.network, checkpoint_path)

    def train(self, frames, mask, istrain=True):
        # gc.collect()
        # torch.cuda.empty_cache()
        frames_tensor_cpu = torch.FloatTensor(frames)
        frames_tensor = frames_tensor_cpu.to(self.device, non_blocking=True)
        del frames_tensor_cpu
        
        # mask_tensor_cpu = torch.FloatTensor(mask)
        # mask_tensor = mask_tensor_cpu.to(self.device, non_blocking=True)
        # del mask_tensor_cpu
        mask_tensor = None
        
        with self.accelerator.accumulate(self.network):
            self.optimizer.zero_grad()
            loss, loss_pred, decouple_loss = self.network(frames_tensor, mask_tensor,istrain=istrain)
            # torch.cuda.empty_cache()
            # loss.backward()
            self.accelerator.backward(loss)
        
        try:
            if loss.dtype == torch.cfloat:
                nploss = np.sqrt(loss.imag.item()**2 + loss.real.item()**2)
            else:
                nploss = loss.item() #.detach().cpu().numpy()
        except Exception as e:
            nploss = 0.0
            self.print (f"Could not convert loss to numpy: {e}")
        return nploss, loss_pred, decouple_loss
        
    def step(self, nploss, loss_pred, decouple_loss):
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        if self.configs.upload_run:
            try:
                self.wrun.log({"Total Loss": nploss, "Pred Loss": loss_pred, 'Decop Loss': decouple_loss})
            except Exception as e:
                self.print (f"Could not log to wandb: {e}")
        

    def test(self, frames_tensor, mask_tensor):
        input_length = self.configs.input_length
        total_length = self.configs.total_length
        output_length = total_length - input_length
        final_next_frames = []
        # if self.configs.concurent_step > 1:
        #     # I have not modified it to make it work.
        #     for i in range(self.configs.concurent_step):
        #         self.print(i)
        #         with torch.no_grad():
        #             next_frames, loss, loss_pred, decouple_loss= self.network(frames_tensor[:,input_length*i:input_length*i+total_length,:,:,:], mask_tensor, istrain=False)
        #         self.print(f"next_frames shape:{next_frames.shape}, frames_tensor shape:{frames_tensor.shape}")
        #         frames_tensor[:,input_length*i+input_length:input_length*i+total_length,:,:,:] = next_frames[:,-output_length:,:,:,:]
        #         final_next_frames.append(next_frames[:,-output_length:,:,:,:].detach().cpu().numpy())
        #         del next_frames
        #         torch.cuda.empty_cache()
        if self.saliency and not self.configs.is_training:
            # add handles
            handles = saliency.modify_model(self.network)
            # backprop
            frames_tensor.requires_grad = True
            next_frames, loss, loss_pred, decouple_loss = self.network(frames_tensor, mask_tensor, istrain=False)
            loss.backward()
            # get saliency maps
            batch = frames_tensor.shape[0]
            idx = batch//2
            salient = frames_tensor.grad[idx,:input_length :, :, :].detach().cpu().numpy()
            frame = frames_tensor[idx,:input_length, :, :, :].detach().cpu().numpy()
            gt = frames_tensor[idx,input_length:, :, :, :].detach().cpu().numpy()
            print("Gradient shape:", salient.shape)
            viz(salient, frame, gt, self.configs)
            # remove handles
            saliency.revert_model(handles)
            with torch.no_grad():
                next_frames = next_frames.detach()
        else:
            with torch.no_grad():
                next_frames, loss, loss_pred, decouple_loss = self.network(frames_tensor, mask_tensor, istrain=False)


        return next_frames, loss, loss_pred, decouple_loss
