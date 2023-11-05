from tqdm import tqdm
import traceback
import torch
import subprocess,argparse,sys,os,numpy as np
import multiprocessing
multiprocessing.set_start_method('spawn', True)
from multiprocessing import Process, Value, Array, Lock
import logging
import signal, time
###############################################
parser=argparse.ArgumentParser()
parser.add_argument("--hyperthreading", help="Run # processes per GPU", type=int, default=1)
parser.add_argument('-m','--models', nargs='+', help='<Required> Model Names', required=True)
parser.add_argument('-il','--input_lengths', nargs='+', help='Input length list', required=False)
parser.add_argument('-pn','--project_names', nargs='+', help='Wandb project name', required=False)
parser.add_argument('-a','--mode', help='Mode [salient, train & test, train, test]', required=False, type=int, default=0)
parser.add_argument('-p', '--preload', help='Preload data',type=int ,required=False, default=0)
parser.add_argument('-mds', '--max_datasets', help='Max datasets',type=int ,required=False, default=-1)
parser.add_argument('-pre', '--preprocessor', help='Preprocessor',type=str ,required=False, default='POD_v4')
parser.add_argument('-gpu', '--gpu', help='GPU',type=int ,required=False, default=-1)
args = parser.parse_args()
hyt = args.hyperthreading
names = args.models
mode = args.mode
###############################################
from config import operate_loop
###############################################
# change these params
visualize=True

if visualize:
    from core.viz.postprocess import visualize as viz

class hyperparam:
    training=True #False # train or test
    max_iterations = 10025
    pretrain_name=None #'model_1000.ckpt' #'model_best_mse.ckpt' # None if no pretrained model
    snapshot_interval = 400 # save model every n iterations
    ##
    model_name = 'rLSTM' # [adaptDNN,DNN,TF,BERT,rBERT,reZeroTF, predrnn_v2]
    preprocessor_name = args.preprocessor # [raw, control, POD, DMD] # raw is no preprocessing for predrnn_v2, else use control
    project_name = 'DMDNet' # name of wandb project
    interpret = False # interpret model
    ##
    save_test_output=True # save test output to file
    weather_prediction=True # use PDE_* data or CDS_* data
    n_valid = 1 # number of validation datasets to use
    max_datasets = args.max_datasets # maximum number of datasets to use (0 for all)
    ##
    input_length = 20 # number of input frames (must be <= total_length)
    total_length = 40 # total number of frames (must be equal to frames slices as given by dataset)
    ## 
    overrides = {}
    ##
    opt_str="_WP" if weather_prediction else ""


hyp = hyperparam()
# hyp.overrides.update({'n_embd': 200}) #64
# hyp.overrides.update({'n_ffn_embd': 200}) #128
# hyp.overrides.update({'n_head': 4})
hyp.n_valid = 12 if hyp.weather_prediction else 1
hyp.max_iterations = 4000
# hyp.overrides.update({'n_embd': 400}) #64
if mode == -1:
    tr = [False]
    ptn = ['last']
    sal = [True]
    if args.preload != 0:
        ptn = [f'model_{args.preload}.ckpt']
elif mode == 0:
    tr = [True, False]
    ptn = [hyp.pretrain_name, 'last']
    sal = [False, False]
elif mode == 1:
    tr = [True]
    ptn = [None]
    sal = [False]
else:
    tr = [False]
    ptn = ['last']
    sal = [False]
    if args.preload != 0:
        ptn = [f'model_{args.preload}.ckpt']
# ptn = ['model_1500.ckpt']
# names = ['BERT','BERT_v2','rBERT','LSTM','rLSTM', 'DNN', 'adaptDNN']#['ViT_LDM','BERT','rBERT','reZeroTF','LSTM','rLSTM']
if args.input_lengths is None:
    input_lengths = [hyp.input_length]*len(names)
    ilstrs = ["" for _ in names]
else:
    assert len(args.input_lengths) == len(names), "Must provide input_lengths for each model"
    input_lengths = args.input_lengths
    ilstrs = [f"_il_{il}" for il in input_lengths]
if args.project_names is None:
    project_names = [hyp.project_name]*len(names)
    pstrs = ["" for _ in names]
else:
    assert len(args.project_names) == len(names), "Must provide project_name for each model"
    project_names = args.project_names
    pstrs = [f"_pn_{pn}" for pn in project_names]
########################

queue = names # thread immutable
running = Value('i', 1)   
n_gpus = torch.cuda.device_count()
gpus = range(n_gpus) if args.gpu==-1 else [args.gpu,] # also immutable
busy_processes = Array('i', np.zeros((n_gpus*hyt), dtype='int32'), lock=False) # mutable
arr_lock = Lock()

def run_job(gpu_id, thread_id,i):    
    device = f'cuda:{gpu_id}'
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
    hyp.model_name = queue[i]
    il = input_lengths[i]
    hyp.input_length = il
    hyp.project_name = project_names[i]
    hyp.opt_str = f"{hyp.opt_str}{ilstrs[i]}"
    
    logging.basicConfig(level = logging.INFO,format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%y-%m-%d %H:%M', handlers = [logging.FileHandler(f'run_{i}.log'), logging.StreamHandler()])
    
    if mode < 3:
        for t,p,s in zip(tr,ptn,sal):
            hyp.training = t
            hyp.pretrain_name = p
            hyp.interpret = s
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

# def signal_handler(sig, frame):
#     print('\nResting workers...')     
#     global processes, running
#     running.value = 0
#     for t in processes:
#         t.join()
#     print('Done.')
                


if __name__ == '__main__':
    run(0,"cuda:0")
    
#     # start workers
#     processes = []

#     signal.signal(signal.SIGINT, signal_handler)
#     value = Value('i', 0)
#     for gpu_id in gpus:
#         for thread in range(hyt):
#             env = os.environ.copy()
#             # env.update({'CUDA_VISIBLE_DEVICES': str(gpu_id)})
#             t = Process(target=worker, args=(gpu_id,thread,value, env, queue, busy_processes, arr_lock, running))
#             processes.append(t)
#             t.start()
#             # time.sleep(0.01)

#     print("Started workers.\nPlease kill with Ctrl-C if necessary.")
    
#     for t in processes:
#         t.join()