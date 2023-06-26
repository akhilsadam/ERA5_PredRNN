from tqdm import tqdm
import traceback
###############################################
from config import operate_loop
###############################################
# change these params
class hyperparam:
    training=True #False # train or test
    max_iterations = 10025
    pretrain_name=None #'model_3000.ckpt' #'model_best_mse.ckpt' # None if no pretrained model
    ##
    model_name = 'rLSTM' # [adaptDNN,DNN,TF,BERT,rBERT,reZeroTF, predrnn_v2]
    preprocessor_name = 'POD' # [raw, control, POD] # raw is no preprocessing for predrnn_v2, else use control
    project_name = 'toy1_control_v2' # name of wandb project
    ##
    save_test_output=True # save test output to file
    weather_prediction=False # use PDE_* data or CDS_* data
    n_valid = 1 # number of validation datasets to use
    ## 
    overrides = {}


hyp = hyperparam()
hyp.overrides.update({'n_embd': 64})
hyp.overrides.update({'n_ffn_embd': 128})
hyp.max_iterations = 1025

tr = [True, False]
ptn = [None, 'model_1000.ckpt']
names = ['rBERT']#['BERT','rBERT','reZeroTF','LSTM','rLSTM']
for n in tqdm(names):
    for t,p in zip(tr,ptn):
        hyp.training = t
        hyp.pretrain_name = p
        hyp.model_name = n
        try:
            operate_loop(hyp)
        except Exception as e:
            print(f'Error: {e} for {n} {"Training" if t else "Test"} generated from {p}')
            print(traceback.format_exc())
            print('Skipping...')
            break