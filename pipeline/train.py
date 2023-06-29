from tqdm import tqdm
import traceback
import torch
import subprocess,argparse,sys,os,numpy as np, threading
import signal
###############################################
parser=argparse.ArgumentParser()
parser.add_argument("--hyperthreading", help="Run # threads per GPU", type=int, default=1)
args = parser.parse_args()
hyt = args.hyperthreading
###############################################
from config import operate_loop
###############################################
# change these params
visualize=True

if visualize:
    from postprocess import visualize as viz

class hyperparam:
    training=True #False # train or test
    max_iterations = 10025
    pretrain_name=None #'model_3000.ckpt' #'model_best_mse.ckpt' # None if no pretrained model
    ##
    model_name = 'rLSTM' # [adaptDNN,DNN,TF,BERT,rBERT,reZeroTF, predrnn_v2]
    preprocessor_name = 'control' # [raw, control, POD] # raw is no preprocessing for predrnn_v2, else use control
    project_name = 'LS6_toy1_control_v3' # name of wandb project
    ##
    save_test_output=True # save test output to file
    weather_prediction=False # use PDE_* data or CDS_* data
    n_valid = 1 # number of validation datasets to use
    ## 
    overrides = {}


hyp = hyperparam()
# hyp.overrides.update({'n_embd': 4096//4}) #64
# hyp.overrides.update({'n_ffn_embd': 4096}) #128
hyp.max_iterations = 2005

tr = [True, False]
# tr=[False]
ptn = [None, 'model_2000.ckpt']
# ptn = ['model_1500.ckpt']
names = ['reZeroTF', 'DualAttentionTransformer']#['ViT_LDM','BERT','rBERT','reZeroTF','LSTM','rLSTM']



if __name__ == '__main__':
    
    queue = names.copy()
    running = True    
    devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
    n_gpus = len(devices)
    busy_threads = np.zeros((n_gpus,hyt))
        
    def run_job(gpu_id, thread_id, name):
        global busy_threads
        busy_threads[gpu_id,thread_id] = 1
        
        device = devices[gpu_id]
        print(f"Running {name} on {device} thread {thread_id}")
        run(name, device)
    
        busy_threads[gpu_id,thread_id] = 0

    def worker(gpu_id, thread_id):
        global queue, busy_threads, running
        while running:
            if busy_threads[gpu_id,thread_id] == 0: # need to check if busy, so not using queue.consume
                job = queue.pop(0) # wait for 1 second
                if job is not None:
                    run_job(gpu_id,thread_id, job)
                else:
                    break    
        
    def run(name, device):
        skip = False
        for t,p in zip(tr,ptn):
            hyp.training = t
            hyp.pretrain_name = p
            hyp.model_name = name
            try:
                operate_loop(hyp, device)
            except Exception as e:
                print(f'Error: {e} for {name} {"Training" if t else "Test"} generated from {p}')
                print(traceback.format_exc())
                print('Skipping...')
                skip = True
                break
        if visualize and not skip:
            viz(hyp)
            
    # start workers
    threads = []
    
    def signal_handler(sig, frame):
        print('\nResting workers...')     
        global threads, running
        running = False
        for t in threads:
            t.join()
        print('Done.')
        
    signal.signal(signal.SIGINT, signal_handler)
    
    for gpu_id in range(n_gpus):
        for thread in range(hyt):
            t = threading.Thread(target=worker, args=(gpu_id,thread))
            threads.append(t)
            t.start()

    print("Started workers.\nPlease kill with Ctrl-C if necessary.")