import sys
sys.path.insert(0,"/home/prsood/projects/def-whkchun/prsood/multi-modal-emotion/") 
__package__ = 'DoubleModels'

from .train_model.visual_bert_train import train_vbert_network, evaluate_vbert
from .models.visualBert import VBertClassifier , collate_batch
import wandb

from utils.data_loaders import VBertDataset
from utils.global_functions import arg_parse 
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils.global_functions import arg_parse , hidden_layer_count , Metrics
from sklearn.model_selection import train_test_split


def prepare_dataloader(df , batch_size, pin_memory=True, num_workers=0):
    """
    we load in our dataset, and we just make a random distributed sampler to evenly partition our 
    dataset on each GPU
    say we have 32 data points, if batch size = 8 then it will make 4 dataloaders of size 8 each 
    """
    max_len = 70 # just max number of tokens from LSTM    keep this line in here somewhere

    dataset = VBertDataset(df , max_len)
    # dataset.img_path.to("cuda")
    # dataset.texts.to("cuda")
    # dataset.labels.to("cuda")
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, 
            num_workers=num_workers, drop_last=True, shuffle=True , collate_fn=collate_batch)
    return dataloader

def runModel( rank , df_train , df_val, df_test  ,param_dict , model_param , wandb):
    """
    Start by getting all the required values from our dictionary
    Then when all the stuff is done, we start to apply our multi-processing to our model and start it up
    """    
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

    num_labels = model_param['output_dim']
    input_dim = model_param['input_dim']
    lstm_layers = model_param['lstm_layers']
    hidden_layers = model_param['hidden_layers']

    criterion = torch.nn.CrossEntropyLoss().to(rank)
    Metric = Metrics(num_classes = num_labels, id2label = id2label , rank = rank)
    df_train = prepare_dataloader(df_train, batch_size = batch_size )
    df_val = prepare_dataloader(df_val, batch_size = batch_size )
    df_test = prepare_dataloader(df_test , batch_size = batch_size)

    model = VBertClassifier(model_param).to(rank)
    
    wandb.watch(model, log = "all")
    model = train_vbert_network(model, df_train, df_val, criterion , lr, epoch ,  weight_decay,T_max, Metric , patience , clip )
    evaluate_vbert(model, df_test, Metric)


def main():
    project_name = "MLP_test_text"
    args = arg_parse(project_name)
    run = wandb.init(project=project_name, entity="ddi" , config = args)
    config = wandb.config
    np.random.seed(config.seed)
    torch.random.manual_seed(config.seed)

    
    param_dict = {
        'epoch':config.epoch ,
        'patience':config.patience ,
        'lr':config.learning_rate ,
        'clip': config.clip ,
        'batch_size':config.batch_size ,
        'weight_decay':config.weight_decay ,
        'model': config.model,
        'T_max':config.T_max ,
        'seed':config.seed,
    }

    model_param = {
        'input_dim':config.input_dim ,
        'output_dim':config.output_dim ,
        'lstm_layers':config.lstm_layers ,
        # Need to add the hidden layer count for each modality for their hidden layers 
        'hidden_layers':hidden_layer_count(config.hidden_layers) ,
    }

    df = pd.read_pickle(f"{args.dataset}.pkl")

    """
    Due to data imbalance we are going to reweigh our CrossEntropyLoss
    To do this we calculate 1 - (num_class/len(df)) the rest of the functions are just to order them properly and then convert to a tensor
    """
    try:
        weights = torch.Tensor(list(dict(sorted((dict(1 - (df['emotion'].value_counts()/len(df))).items()))).values()))
        # weights = torch.Tensor([1,1,1,1,1,1,1])
        df_train, df_test, _, __ = train_test_split(df, df["emotion"], test_size = 0.25, random_state = param_dict['seed'] , stratify=df["emotion"])
        df_train, df_val, _, __ = train_test_split(df_train, df_train["emotion"], test_size = 0.25, random_state = param_dict['seed'] , stratify=df_train["emotion"])
        label2id = df.drop_duplicates('label').set_index('label').to_dict()['emotion']
    except:
        weights = torch.Tensor(list(dict(sorted((dict(1 - (df['label'].value_counts()/len(df))).items()))).values()))
        # weights = torch.Tensor([0,1])
        df_train, df_test, _, __ = train_test_split(df, df["label"], test_size = 0.25, random_state = param_dict['seed'] , stratify=df["label"])
        df_train, df_val, _, __ = train_test_split(df_train, df_train["label"], test_size = 0.25, random_state = param_dict['seed'] , stratify=df_train["label"])
        label2id = df.drop_duplicates('label').set_index('label').to_dict()['sentiment']
        label2id = {v: k for k, v in label2id.items()}

    id2label = {v: k for k, v in label2id.items()}
    
    param_dict['weights'] = weights
    param_dict['label2id'] = label2id
    param_dict['id2label'] = id2label

    print(f" in main \n param_dict = {param_dict} \n model_param = {model_param} \n df {args.dataset}")

    runModel("cuda",df_train , df_val , df_test ,param_dict , model_param , run )
    
if __name__ == '__main__':
    main()





