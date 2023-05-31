if __name__ == '__main__' and not __package__:
    import sys
    sys.path.insert(0, "/home/prsood/projects/def-whkchun/prsood/multi-modal-emotion/")
    __package__ = 'DoubleModels'

from .train_model.audio_training import train_audio_network
from .models.text_audio import  BertAudioClassifier , LSTMAudioClassifier
import wandb
from utils.data_loaders import BertAudioDataset , LstmAudioDataset
from utils.global_functions import arg_parse 
import torch
import pandas as pd
from transformers import AutoConfig
import numpy as np
import cProfile
import io
import pstats

def runModel(df,param_dict , model_param):

    df = df[df['num_channels'] == 6] # we need to pre-process to see the number of channels on each video then set this number accordingly

    df_train = df[df['split'] == "train"]#.head(8) # 9989 rows of training data
    df_test = df[df['split'] == "test"]#.head(8) # 2610 rows of testing data
    df_val = df[df['split'] == "val"]#.head(8) # 1106 rows of validation data 


    max_len = 92 # just max number of tokens from LSTM   
        
    model_path = "facebook/wav2vec2-large-960h"
    # model = AudioMLP(model_param)
    config = AutoConfig.from_pretrained(model_path)
    pooling_mode = 'mean'
    problem_type = 'multi_label_classification'
    num_labels = model_param['output_dim']
    input_dim = model_param['input_dim']
    lstm_layers = model_param['lstm_layers']
    hidden_layers = model_param['hidden_layers']




    setattr(config, 'pooling_mode', pooling_mode)
    setattr(config, 'problem_type', problem_type)
    setattr(config, 'num_labels', num_labels)
    setattr(config, 'input_dim', input_dim)
    setattr(config, 'lstm_layers', lstm_layers)
    setattr(config, 'hidden_layers', hidden_layers)


    if  "Bert" in param_dict['model']:
        model = BertAudioClassifier(config)  
        train, val , test = BertAudioDataset(df_train , max_len=max_len), BertAudioDataset(df_val , max_len=max_len) , BertAudioDataset(df_test , max_len=max_len)
    else:
        train, val , test = LstmAudioDataset(df_train , max_len=max_len), LstmAudioDataset(df_val , max_len=max_len) , LstmAudioDataset(df_test , max_len=max_len)
        glove_vec = train.get_glove_vocab()
        model = LSTMAudioClassifier(glove_vec , config)
    
    # model.freeze_feature_extractor()

    # Due to memory issues i am going to be doing this inside of the training file, the dataloader of everything
    
    wandb.watch(model, log = "all")
    
    epoch = param_dict['epoch']
    lr = param_dict['lr']
    patience = param_dict['patience']
    clip = param_dict['clip']
    T_max = param_dict['T_max']
    batch_size = param_dict['batch_size']
    weight_decay = param_dict['weight_decay']
    

    
    model =  train_audio_network(model, df_train, df_val, lr, epoch , batch_size , weight_decay,T_max , patience , clip )
    # evaluate_audio(model, df_test)

    

def main():
    project_name = "MLP_test_audio"
    args =  arg_parse(project_name)
    # wandb.init(project="Emotion_" + args.dataset + "_" + args.model, entity="ddi" , config = args)
    wandb.init(project=project_name, entity="ddi" , config = args)
    config = wandb.config
    # print(config)
    # Set random seeds
    np.random.seed(config.seed)
    torch.random.manual_seed(config.seed)

    
    param_dict = {
        'epoch':config.epoch ,
        'patience':config.patience ,
        'lr': config.learning_rate ,
        'clip': config.clip ,
        'batch_size':config.batch_size ,
        'weight_decay':config.weight_decay ,
        'model': config.model,
        'T_max':config.T_max ,
    }

    model_param = {
        'input_dim':config.input_dim ,
        'output_dim':config.output_dim ,
        'lstm_layers':config.lstm_layers ,
        'hidden_layers':config.hidden_layers ,
    }

    
    # df = pd.read_pickle(f"{args.dataset}.pkl")
    
    print(f" in main \n param_dict = {param_dict} \n model_param = {model_param} \n df {args.dataset}")
    runModel(   pd.read_pickle(f"{args.dataset}.pkl") , param_dict , model_param)

if __name__ == '__main__':
    main()
    # pr = cProfile.Profile()
    # pr.enable()

    # my_result = main()

    # pr.disable()
    # s = io.StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    # ps.print_stats()

    # with open('test.txt', 'w+') as f:
    #     f.write(s.getvalue())
    
 
