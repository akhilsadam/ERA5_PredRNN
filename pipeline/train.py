from tqdm import tqdm
import traceback
import torch
import subprocess,argparse,sys,os,numpy as np
import multiprocessing
multiprocessing.set_start_method('spawn', True)
from multiprocessing import Process, Value, Array, Lock
import signal, time
###############################################
parser=argparse.ArgumentParser()
parser.add_argument("--hyperthreading", help="Run # processes per GPU", type=int, default=1)
parser.add_argument('-m','--models', nargs='+', help='<Required> Model Names', required=True)
parser.add_argument('-il','--input_lengths', nargs='+', help='Input length list', required=False)
args = parser.parse_args()
hyt = args.hyperthreading
names = args.models
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
    input_length = 20 # number of input frames (must be <= total_length)
    total_length = 40 # total number of frames (must be equal to frames slices as given by dataset)
    ## 
    overrides = {}
    ##
    opt_str=""


hyp = hyperparam()
# hyp.overrides.update({'n_embd': 400}) #64
# hyp.overrides.update({'n_ffn_embd': 400}) #128
hyp.max_iterations = 2005

hyp.overrides.update({'n_embd': 400}) #64

tr = [True, False]
# tr = [True]
# tr=[False]
ptn = [None, 'model_2000.ckpt']
# ptn = [None]
# ptn = ['model_1500.ckpt']
# names = ['BERT','BERT_v2','rBERT','LSTM','rLSTM', 'DNN', 'adaptDNN']#['ViT_LDM','BERT','rBERT','reZeroTF','LSTM','rLSTM']
if args.input_lengths is None:
    input_lengths = [hyp.input_length]*len(names)
else:
    assert len(args.input_lengths) == len(names), "Must provide input_lengths for each model"
    input_lengths = args.input_lengths
    makestr = True
########################

queue = names # thread immutable
running = Value('i', 1)   
n_gpus = torch.cuda.device_count()
gpus = range(n_gpus) # also immutable
busy_processes = Array('i', np.zeros((n_gpus*hyt), dtype='int32'), lock=False) # mutable
arr_lock = Lock()

def run_job(gpu_id, thread_id,i):    
    device = f'cuda:0'
    print(f"Running {queue[i]} on {device} thread {thread_id}")
    run(i, device)

def worker(gpu_id, thread_id, value, env, queue, busy_processes_buffer, lock, running):
    os.environ.update(env)
    busy_processes = np.frombuffer(busy_processes_buffer, dtype='int32').reshape((n_gpus,hyt))
    
    while running.value == 1:
        with lock:
            busy = busy_processes[gpu_id,thread_id]
        if busy == 0: # need to check if busy, so not using queue.consume
            cvalue = value.value
            if len(queue) > cvalue:
                value.value += 1
                job = cvalue # wait for 1 second
                if job is not None:
                    with lock:
                        busy_processes[gpu_id,thread_id] = 1
                    # print(gpu_id, thread_id, queue[job], input_lengths[job])
                    run_job(gpu_id, thread_id, job)
                    #time.sleep(6)
                    with lock:
                        busy_processes[gpu_id,thread_id] = 0
                else:
                    break    
            else:
                break
    
def run(i, device):
    skip = False
    for t,p in zip(tr,ptn):
        hyp.training = t
        hyp.pretrain_name = p
        hyp.model_name = queue[i]
        il = input_lengths[i]
        hyp.input_length = il
        if makestr:
            hyp.opt_str = f"_il_{il}"
        try:
            operate_loop(hyp, device)
        except Exception as e:
            print(f'Error: {e} for {hyp.model_name} {"Training" if t else "Test"} generated from {p}')
            print(traceback.format_exc())
            print('Skipping...')
            skip = True
            break
    if visualize and not skip:
        viz(hyp)

def signal_handler(sig, frame):
    print('\nResting workers...')     
    global processes, running
    running.value = 0
    for t in processes:
        t.join()
    print('Done.')
                


if __name__ == '__main__':
    
    
    # start workers
    processes = []

    signal.signal(signal.SIGINT, signal_handler)
    value = Value('i', 0)
    for gpu_id in gpus:
        for thread in range(hyt):
            env = os.environ.copy()
            env.update({'CUDA_VISIBLE_DEVICES': str(gpu_id)})
            t = Process(target=worker, args=(gpu_id,thread,value, env, queue, busy_processes, arr_lock, running))
            processes.append(t)
            t.start()
            # time.sleep(0.01)

    print("Started workers.\nPlease kill with Ctrl-C if necessary.")
    
    for t in processes:
        t.join()