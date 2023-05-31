import sys
sys.path.insert(0,"/home/prsood/projects/def-whkchun/prsood/multi-modal-emotion/") 
__package__ = 'DoubleModels'


from .train_model.text_video_train import train_bert_video_network, evaluate_bert_video
from .models.text_video import BertVideoMAE_MTL1Shared_Classifier , BertVideoMAE_LateFusion_Classifier , collate_batch
import wandb

from utils.data_loaders import TextVideoDataset
from utils.global_functions import arg_parse 
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils.global_functions import arg_parse , hidden_layer_count , Metrics
from sklearn.model_selection import train_test_split


def prepare_dataloader(df , batch_size, label_task , pin_memory=True, num_workers=4 , check = "train"):
    """
    we load in our dataset, and we just make a random distributed sampler to evenly partition our 
    dataset on each GPU
    say we have 32 data points, if batch size = 8 then it will make 4 dataloaders of size 8 each 
    """

    dataset = TextVideoDataset(df , max_len=70 , feature_col1="video_path" , feature_col2="text" , label_col=label_task , timings="timings" , speaker="speaker")
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, 
            num_workers=num_workers, drop_last=True, shuffle=True , collate_fn = lambda batch: collate_batch(batch, check))
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
    label_task = param_dict['label_task']

    num_labels = model_param['output_dim']
    input_dim = model_param['input_dim']
    lstm_layers = model_param['lstm_layers']
    hidden_layers = model_param['hidden_layers']

    criterion = torch.nn.CrossEntropyLoss(weight=weights).to(rank)
    Metric = Metrics(num_classes = num_labels, id2label = id2label , rank = rank)
    df_train = prepare_dataloader(df_train,  batch_size ,  label_task , check = "train")
    df_val = prepare_dataloader(df_val,  batch_size ,  label_task , check = "val")
    df_test = prepare_dataloader(df_test ,  batch_size,  label_task , check = "val")

    if param_dict['model'] == "1MTL":
        model = BertVideoMAE_MTL1Shared_Classifier(model_param).to(rank)
    elif param_dict['model'] == "LateFusion":
        model = BertVideoMAE_LateFusion_Classifier(model_param).to(rank)

    
    wandb.watch(model, log = "all")
    model = train_bert_video_network(model, df_train, df_val, criterion , lr, epoch ,  weight_decay,T_max, Metric , patience , clip )
    evaluate_bert_video(model, df_test, Metric)


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
        'lr': config.learning_rate ,
        'clip': config.clip ,
        'batch_size':config.batch_size ,
        'weight_decay':config.weight_decay ,
        'model': config.model,
        'T_max':config.T_max ,
        'seed':config.seed,
        'label_task':config.label_task,
    }

    df = pd.read_pickle(f"{args.dataset}.pkl")
    if param_dict['label_task'] == "sentiment":
        number_index = "sentiment"
        label_index = "sentiment_label"
    else:
        number_index = "emotion"
        label_index = "emotion_label"

    if "IEMOCAP" in args.dataset:
        df = df[ (df['emotion_label'] != "surprised")  & (df['emotion_label'] != "fearful")  & (df['emotion_label'] != "other")  & (df['emotion_label'] != "disgusted")   ]
        df_train, df_test, _, __ = train_test_split(df, df[number_index], test_size = 0.25, random_state = param_dict['seed'] , stratify=df[number_index])
        df_train, df_val, _, __ = train_test_split(df_train, df_train[number_index], test_size = 0.25, random_state = param_dict['seed'] , stratify=df_train[number_index])
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
    label2id = df.drop_duplicates(label_index).set_index(label_index).to_dict()[number_index]
    id2label = {v: k for k, v in label2id.items()}

    model_param = {
        'input_dim':config.input_dim ,
        'output_dim':len(weights) ,
        'lstm_layers':config.lstm_layers ,
        'hidden_layers':hidden_layer_count(config.hidden_layers) ,
    }
    
    param_dict['weights'] = weights
    param_dict['label2id'] = label2id
    param_dict['id2label'] = id2label

    print(f" in main \n param_dict = {param_dict} \n model_param = {model_param} \n df {args.dataset} , with df_size = {len(df)} \n ")

    runModel("cuda",df_train , df_val , df_test ,param_dict , model_param , run )
    
if __name__ == '__main__':
    main()





