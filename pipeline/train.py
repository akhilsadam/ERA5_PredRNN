from tqdm import tqdm
import traceback
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
names = ['BERT_v3, DualAttentionTransformer']#['ViT_LDM','BERT','rBERT','reZeroTF','LSTM','rLSTM']

if __name__ == '__main__':
    for n in tqdm(names):
        skip = False
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
                skip = True
                break
        if visualize and not skip:
            viz(hyp)