
import os
import sys
sys.path.insert(0,"/home/prsood/projects/def-whkchun/prsood/multi-modal-emotion/") 
__package__ = 'TripleModels'
from transformers import logging
logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore")

from .train_model.tav_train import train_tav_network, evaluate_tav
from .models.tav import TAVForMAE , TAVFormer , collate_batch
import wandb

from utils.data_loaders import TextAudioVideoDataset
from utils.global_functions import arg_parse 
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils.global_functions import arg_parse , hidden_layer_count , Metrics , F1_Loss
from sklearn.model_selection import train_test_split

class BatchCollation:
    def __init__(self,  check , mask) -> None:
        self.check = check
        self.mask = mask
    
    def __call__(self, batch):
        return collate_batch(batch , self.check , self.mask)


def prepare_dataloader(df , batch_size, label_task, mask , pin_memory=True, num_workers=4 , check = "train" ): # num_W = 8 kills it 
    """
    we load in our dataset, and we just make a random distributed sampler to evenly partition our 
    dataset on each GPU
    say we have 32 data points, if batch size = 8 then it will make 4 dataloaders of size 8 each 
    """
    if batch_size > 8:
        num_workers = 2
    dataset = TextAudioVideoDataset(df , max_len=70 , feature_col1="audio_path", feature_col2="video_path" , feature_col3="text" , label_col=label_task , timings="timings" , speaker="speaker")
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, 
            num_workers=num_workers ,drop_last=False, shuffle=True , 
            collate_fn = BatchCollation(check , mask)
            )
    return dataloader

def runModel( accelerator, df_train , df_val, df_test  ,param_dict , model_param):
    """
    Start by getting all the required values from our dictionary
    Then when all the stuff is done, we start to apply our multi-processing to our model and start it up
    """    
    # setup(rank , world_size)
    device = accelerator 
    # print(f"rank at beginning is {rank}" , flush = True)
    max_len = 70 # just max number of tokens from LSTM    keep this line in here somewhere
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
    label_task = param_dict['label_task']
    model_name = param_dict['model']
    mask = param_dict['mask']

    num_labels = model_param['output_dim']
    dropout = model_param['dropout']
    early_div = model_param['early_div']


    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device)).to(device)
    # criterion = F1_Loss(num_classes = num_labels,weights=weights.to(device) , beta=0 ).to(device)
    Metric = Metrics(num_classes = num_labels, id2label = id2label , rank = device)
    df_train = prepare_dataloader(df_train,  batch_size ,  label_task, mask , check = "train")
    df_val = prepare_dataloader(df_val,  batch_size ,  label_task, mask , check = "val")
    df_test = prepare_dataloader(df_test ,  batch_size,  label_task, mask , check = "val")
    if model_name == "TAVFormer":
        model = TAVFormer(model_param).to(device)
    else:
        model = TAVForMAE(model_param).to(device) # Make them TAVFormer because the other one isnt memory efficient
    wandb.watch(model, log = "all")

    model = train_tav_network(model, df_train, df_val, criterion , lr, epoch ,  weight_decay,T_max, Metric , patience , clip )
    evaluate_tav(model, df_test, Metric)
    # cleanup()


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    project_name = "MLP_test_text"
    args = arg_parse(project_name)

    wandb.init(entity="ddi" , config = args)
    config = wandb.config
    
    # config = args
    np.random.seed(config.seed)
    torch.random.manual_seed(config.seed)
    param_dict = {
        'epoch':config.epoch ,
        'patience':config.patience ,
        'lr': config.learning_rate ,
        'clip': config.clip ,
        'batch_size':config.batch_size ,
        'weight_decay':config.weight_decay ,
        'model':config.model,
        'T_max':config.T_max ,
        'seed':config.seed,
        'label_task':config.label_task,
        'mask':config.mask,
    }

    df = pd.read_pickle(f"{args.dataset}.pkl")
    if param_dict['label_task'] == "sentiment":
        number_index = "sentiment"
        label_index = "sentiment_label"
    else:
        number_index = "emotion"
        label_index = "emotion_label"

    if "IEMOCAP" in args.dataset:
        # df = df[ (df['emotion_label'] != "surprised")  & (df['emotion_label'] != "fearful")  & (df['emotion_label'] != "other")  & (df['emotion_label'] != "disgusted")   ]
        # df_train, df_test, _, __ = train_test_split(df, df[number_index], test_size = 0.25, random_state = param_dict['seed'] , stratify=df[number_index])
        # df_train, df_val, _, __ = train_test_split(df_train, df_train[number_index], test_size = 0.25, random_state = param_dict['seed'] , stratify=df_train[number_index])
        df_train = df[df['split'] == "train"]
        df_test = df[df['split'] == "test"]
        df_val = df[df['split'] == "val"]
    else:
        # df = df[df["size_padding"] < 999426]
        df = df[df["audio_shape"] > 10000] # Still seeing what the best configuration is for these
        # df_train, df_test, _, __ = train_test_split(df, df[number_index], test_size = 0.25, random_state = param_dict['seed'] , stratify=df[number_index])
        # df_train, df_val, _, __ = train_test_split(df_train, df_train[number_index], test_size = 0.25, random_state = param_dict['seed'] , stratify=df_train[number_index])
        df = df[ (df['emotion_label'] != "fear")  & (df['emotion_label'] != "disgust")]
        df_train = df[df['split'] == "train"] 
        df_test = df[df['split'] == "test"] 
        df_val = df[df['split'] == "val"] 


        
    df = df[~df['timings'].isna()] # Still seeing what the best configuration is for these

    """
    Due to data imbalance we are going to reweigh our CrossEntropyLoss
    To do this we calculate 1 - (num_class/len(df)) the rest of the functions are just to order them properly and then convert to a tensor
    """
    
    
    weights = torch.Tensor(list(dict(sorted((dict(1 - (df[number_index].value_counts()/len(df))).items()))).values()))
    # Now normalize the weights
    # weights /= weights.sum() Do this inside loss function
    label2id = df.drop_duplicates(label_index).set_index(label_index).to_dict()[number_index]
    id2label = {v: k for k, v in label2id.items()}

    model_param = {
        'output_dim':len(weights) ,
        'dropout' : config.dropout,
        'early_div' : config.early_div,
        'num_layers' : config.num_layers,
    }
    
    param_dict['weights'] = weights
    param_dict['label2id'] = label2id
    param_dict['id2label'] = id2label

    print(f" in main \n param_dict = {param_dict} \n model_param = {model_param} \n df {args.dataset} , with df = {len(df)} \n ")

    runModel("cuda" ,df_train , df_val , df_test ,param_dict , model_param  )
    
    
if __name__ == '__main__':
    main()





