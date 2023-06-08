import os
import numpy as np
import torch
from torch.optim import Adam
from core.models import predrnn, predrnn_v2, action_cond_predrnn, action_cond_predrnn_v2, TF, DNN
from torchview import draw_graph
import traceback, sys
import wandb
import gc

class Model(object):
    def __init__(self, configs):
        self.configs = configs
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        networks_map = {
            'predrnn': predrnn.RNN,
            'predrnn_v2': predrnn_v2.RNN,
            'action_cond_predrnn': action_cond_predrnn.RNN,
            'action_cond_predrnn_v2': action_cond_predrnn_v2.RNN,
            'TF': TF.TF,
            'DNN': DNN.DNN,
        }

        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            self.network = Network(self.num_layers, self.num_hidden, configs).to(self.configs.device)
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)
        
        if 'predrnn' not in configs.model_name and 'pretrained_model' not in configs.__dict__:
            try:
                self.modelvis()
            except Exception as e:
                ex_type, ex_value, ex_traceback = sys.exc_info()
                trace_back = traceback.extract_tb(ex_traceback)
                print(f"Is graphviz installed? Could not visualize model: {e}")
                    # Format stacktrace
                stack_trace = []
                for trace in trace_back:
                    stack_trace.append("File : %s , Line : %d, Func.Name : %s, Message : %s" % (trace[0], trace[1], trace[2], trace[3]))
                print("Exception type : %s " % ex_type.__name__)
                print("Exception message : %s" %ex_value)
                print("Stack trace : %s" %stack_trace)
                

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=configs.lr)
        if self.configs.upload_run:
            self.upload_wandb()
    
    def modelvis(self):
        draw_graph(self.network, input_size= \
            (self.configs.batch_size,self.configs.total_length, \
            self.configs.img_channel,self.configs.img_height,self.configs.img_width), expand_nested=False, roll=True, save_graph=True, filename=self.configs.model_name, directory=self.configs.save_dir)  
        # model_graph.visual_graph.render(format='svg')
    
    def upload_wandb(self):
        # Uploading to wandb
        run_name = (
            f'{self.configs.model_name}_{self.configs.preprocessor_name}_'
            + '_'.join((self.configs.save_file, self.configs.run_name))
        )
        wandb.init(project=self.configs.project, name=run_name)
        wandb.config.model_name = self.configs.model_name
        wandb.config.opt = self.configs.opt
        wandb.config.lr = self.configs.lr
        wandb.config.batch_size = 1
        wandb.config.preprocessor = self.configs.preprocessor_name

    def save(self, itr):
        stats = {}
        stats['net_param'] = self.network.state_dict()
        checkpoint_path = os.path.join(self.configs.save_dir, 'model'+'_'+str(itr)+'.ckpt')
        torch.save(stats, checkpoint_path)
        print("save model to %s" % checkpoint_path)

    def load(self, checkpoint_path):
        print('load model:', checkpoint_path)
        stats = torch.load(checkpoint_path, map_location=torch.device(self.configs.device))
        self.network.load_state_dict(stats['net_param'])

    def train(self, frames, mask, istrain=True):
        gc.collect()
        torch.cuda.empty_cache()
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        self.optimizer.zero_grad()
        loss, loss_pred, decouple_loss = self.network(frames_tensor, mask_tensor,istrain=istrain)
        torch.cuda.empty_cache()
        loss.backward()
        self.optimizer.step()
        if self.configs.upload_run:
            wandb.log({"Total Loss": float(loss), "Pred Loss": loss_pred, 'Decop Loss': decouple_loss})
        return loss.detach().cpu().numpy()

    def test(self, frames_tensor, mask_tensor):
        input_length = self.configs.input_length
        total_length = self.configs.total_length
        output_length = total_length - input_length
        final_next_frames = []
        if self.configs.concurent_step > 1:
            # I have not modified it to make it work.
            for i in range(self.configs.concurent_step):
                print(i)
                with torch.no_grad():
                    next_frames, loss, loss_pred, decouple_loss= self.network(frames_tensor[:,input_length*i:input_length*i+total_length,:,:,:], mask_tensor, istrain=False)
                print(f"next_frames shape:{next_frames.shape}, frames_tensor shape:{frames_tensor.shape}")
                frames_tensor[:,input_length*i+input_length:input_length*i+total_length,:,:,:] = next_frames[:,-output_length:,:,:,:]
                final_next_frames.append(next_frames[:,-output_length:,:,:,:].detach().cpu().numpy())
                del next_frames
                torch.cuda.empty_cache()
        else:
            with torch.no_grad():
                next_frames, loss, loss_pred, decouple_loss = self.network(frames_tensor, mask_tensor, istrain=False)

        return next_frames, loss, loss_pred, decouple_loss
