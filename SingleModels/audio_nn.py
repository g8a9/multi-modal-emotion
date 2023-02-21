
import sys 
sys.path.insert(0, "/home/prsood/projects/def-whkchun/prsood/multi-modal-emotion/")
__package__ = 'SingleModels'


import os
from .train_model.audio_training import train_audio_network , evaluate_audio
from .models.audio import  Wav2Vec2ForSpeechClassification
import wandb
from utils.data_loaders import Wav2VecAudioDataset
from utils.global_functions import arg_parse , hidden_layer_count , Metrics
import torch
import pandas as pd
from transformers import AutoConfig
import numpy as np
from sklearn.model_selection import train_test_split

from transformers import AutoModelForAudioClassification

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from .models.audio import collate_batch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler




def prepare_dataloader(df , batch_size, pin_memory=True, num_workers=4):
    """
    we load in our dataset, and we just make a random distributed sampler to evenly partition our 
    dataset on each GPU
    say we have 32 data points, if batch size = 8 then it will make 4 dataloaders of size 8 each 
    """
    dataset = Wav2VecAudioDataset(df.head(264))
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, 
            num_workers=num_workers, drop_last=False, shuffle=True, 
            collate_fn=collate_batch
        )
    return dataloader

def runModel( rank , df_train , df_val , df_test ,param_dict , model_param , wandb):
    """
    Start by getting all the required values from our dictionary
    Then when all the stuff is done, we start to apply our multi-processing to our model and start it up
    """    
    epoch = param_dict['epoch']
    lr = param_dict['lr']
    patience = param_dict['patience']
    clip = param_dict['clip']
    T_max = param_dict['T_max']
    batch_size = param_dict['batch_size']
    weight_decay = param_dict['weight_decay']
    weights = param_dict['weights']
    label2id = param_dict['label2id']
    id2label = param_dict['id2label']

    num_labels = model_param['output_dim']
    input_dim = model_param['input_dim']
    lstm_layers = model_param['lstm_layers']
    hidden_layers = model_param['hidden_layers']

    # from here

    # model_path = "facebook/wav2vec2-large-960h"
    # model_path = "facebook/wav2vec2-large-960h-lv60-self"
    model_path = "superb/wav2vec2-base-superb-er"
    # model_path = "facebook/wav2vec2-base-960h"
    
    config = AutoConfig.from_pretrained(model_path)
    pooling_mode = 'mean'
    problem_type = 'multi_classification'

    setattr(config, 'pooling_mode', pooling_mode)
    setattr(config, 'label2id', label2id)
    setattr(config, 'id2label', id2label)
    setattr(config, 'problem_type', problem_type)
    setattr(config, 'num_labels', num_labels)
    setattr(config, 'input_dim', input_dim)
    setattr(config, 'lstm_layers', lstm_layers)
    setattr(config, 'hidden_layers', hidden_layers)

    # to here is audio specific

    
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(rank)).to(rank) #TODO: maybe error here?
    Metric = Metrics(num_classes = num_labels, id2label = id2label , rank = rank)
    df_train = prepare_dataloader(df_train  , batch_size = batch_size ) #TODO: maybe prepare dataloader?
    df_val = prepare_dataloader(df_val  , batch_size = batch_size )
    df_test = prepare_dataloader(df_test , batch_size = batch_size )

    if param_dict['model'] == "TransformersWav2Vec2":
        # this is our baseline model from Transformers
        model = AutoModelForAudioClassification.from_pretrained(model_path, config = config).to(rank)# num_labels=num_labels,label2id=label2id,id2label=id2label).to(rank)
        
    else:
        model = Wav2Vec2ForSpeechClassification.from_pretrained( model_path , config=config , rank = rank).to(rank)
        model.freeze_feature_extractor()

    print(f"config = \n {config} \n" , flush = True)
    wandb.watch(model, log = "all")
    
    model = train_audio_network(model, df_train, df_val, criterion , lr, epoch ,  weight_decay,T_max, Metric , patience , clip )
    evaluate_audio(model , df_test ,  Metric)

    

def main():
    project_name = "MLP_test_audio"

    # here is where we are going to get the arguments from our yaml/argparse file 
    args =  arg_parse(project_name)
    
    run = wandb.init(project=project_name, entity="ddi" , config = args)
    config = wandb.config
    np.random.seed(config.seed)
    torch.random.manual_seed(config.seed)
    param_dict = {
        'epoch':config.epoch ,
        'patience':config.patience ,
        'lr': config.learning_rate ,
        'clip': config.clip ,
        'batch_size': 8,#config.batch_size,
        'weight_decay':config.weight_decay ,
        'model': config.model,
        'T_max':config.T_max ,
        'seed': config.seed,
    }

    model_param = {
        'input_dim':config.input_dim ,
        'output_dim':config.output_dim ,
        'lstm_layers':config.lstm_layers ,
        # Need to add the hidden layer count for each modality for their hidden layers 
        'hidden_layers':hidden_layer_count(config.hidden_layers) ,
    }

    df = pd.read_pickle(f"{args.dataset}.pkl")
    # df = df[df['num_channels'] == 6] # we need to pre-process to see the number of channels on each video then set this number accordingly
    
    # they are just making sure that our audio files are from 0 to 20 seconds 
    df = df[df["size_padding"] < 1000000] # so we dont break our space constraints

    df = df[df["audio_shape"] > 3280] # so we dont break our space constraints
    
    """
    Due to data imbalance we are going to reweigh our CrossEntropyLoss
    To do this we calculate 1 - (num_class/len(df)) the rest of the functions are just to order them properly and then convert to a tensor
    """
    weights = torch.Tensor(list(dict(sorted((dict(1 - (df['emotion'].value_counts()/len(df))).items()))).values()))

    df_train, df_test, _, __ = train_test_split(df, df["emotion"], test_size = 0.25, random_state = param_dict['seed'] ,)# stratify=df["emotion"])
    df_train, df_val, _, __ = train_test_split(df_train, df_train["emotion"], test_size = 0.25, random_state = param_dict['seed'] ,)# stratify=df_train["emotion"])

    label2id = df.drop_duplicates('label').set_index('label').to_dict()['emotion']
    id2label = {v: k for k, v in label2id.items()}
    
    param_dict['weights'] = weights
    param_dict['label2id'] = label2id
    param_dict['id2label'] = id2label

    print(f" in main \n param_dict = {param_dict} \n model_param = {model_param} \n df {args.dataset}")

    runModel("cuda",df_train , df_val , df_test ,param_dict , model_param , run )

if __name__ == '__main__':
    main()
    
    
