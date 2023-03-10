import os
import sys
sys.path.insert(0,"/home/prsood/projects/def-whkchun/prsood/multi-modal-emotion/") 
__package__ = 'TripleModels'
from transformers import logging
logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore")

from .train_model.tav_train import train_tav_network, evaluate_tav
from .models.tav import PreFormer , TAVForMAE  , collate_batch , MySampler , NewCrossEntropyLoss
import wandb

from utils.data_loaders import TextAudioVideoDataset
import pandas as pd
import torch
import numpy as np
from utils.global_functions import arg_parse , Metrics , FBetaLoss , PrecisionLoss 
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
class BatchCollation:
    def __init__(self,  check  ) -> None:
        self.check = check
    
    def __call__(self, batch):
        return collate_batch(batch , self.check)


def prepare_dataloader(df , batch_size, label_task , pin_memory=True, num_workers=4 , check = "train" ): # num_W = 8 kills it 
    """
    we load in our dataset, and we just make a random distributed sampler to evenly partition our 
    dataset on each GPU
    say we have 32 data points, if batch size = 8 then it will make 4 dataloaders of size 8 each 
    """
    if batch_size > 8:
        num_workers = 2
  

    
    # TODO: DATASET SPECIFIC
    # df = df.head(350)
    dataset = TextAudioVideoDataset(df , max_len=70 , feature_col1="audio_path", feature_col2="video_path" , feature_col3="text" , label_col=label_task , timings="timings" , speaker="speaker")
    if check == "train":
        labels = df[label_task].value_counts()
        class_counts = torch.Tensor(list(dict(sorted((dict((labels)).items()))).values())).to(int)

        samples_weight = torch.tensor([1/class_counts[t] for t in dataset.labels])
        print(samples_weight, "\n\n" , len(samples_weight))
        sampler = WeightedRandomSampler(list(samples_weight), int(1*len(samples_weight)))
        # sampler = MySampler(list(samples_weight), len(samples_weight) , replacement=True , epoch=0)
        dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, 
                num_workers=num_workers ,drop_last=False, shuffle=False, sampler = sampler,
                collate_fn = BatchCollation(check))
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, 
                num_workers=num_workers ,drop_last=False, shuffle=True,
                collate_fn = BatchCollation(check))

    return dataloader

def runModel( accelerator, df_train , df_val, df_test  ,param_dict , model_param):
    """
    Start by getting all the required values from our dictionary
    Then when all the stuff is done, we start to apply our multi-processing to our model and start it up
    """    
    device = accelerator 
    epoch = param_dict['epoch']
    lr = param_dict['lr']
    patience = param_dict['patience']
    clip = param_dict['clip']
    T_max = param_dict['T_max']
    batch_size = param_dict['batch_size']
    loss = param_dict['loss']
    beta = param_dict['beta']
    weight_decay = param_dict['weight_decay']
    weights = param_dict['weights']
    id2label = param_dict['id2label']
    label_task = param_dict['label_task']
    model_name = param_dict['model']
    mask = param_dict['mask']

    num_labels = model_param['output_dim']
    
    if loss == "CrossEntropy":
        # criterion = NewCrossEntropyLoss(class_weights=weights.to(device)).to(device) # TODO: have to call epoch in forward function
        criterion = torch.nn.CrossEntropyLoss().to(device)
       
    elif loss == "NewCrossEntropy":
        # criterion = PrecisionLoss(num_classes = num_labels,weights=weights.to(device)).to(device)
        criterion = NewCrossEntropyLoss(class_weights = weights.to(device)).to(device)
    elif loss == "Precision":
        # criterion = PrecisionLoss(num_classes = num_labels,weights=weights.to(device)).to(device)
        criterion = PrecisionLoss(num_classes = num_labels).to(device)
       
    elif loss == "FBeta":
        # criterion = FBetaLoss(num_classes = num_labels,weights=weights.to(device) , beta=beta ).to(device)
        criterion = FBetaLoss(num_classes = num_labels, beta=beta ).to(device)

    print(loss , flush = True)
    Metric = Metrics(num_classes = num_labels, id2label = id2label , rank = device)
    df_train = prepare_dataloader(df_train,  batch_size ,  label_task , check = "train")
    df_val = prepare_dataloader(df_val,  batch_size ,  label_task , check = "val")
    df_test = prepare_dataloader(df_test ,  batch_size,  label_task , check = "val")
    
    model = TAVForMAE(model_param).to(device)
  

    PREFormer = PreFormer().to(f"cpu")
    
    
    wandb.watch(model, log = "all")
    wandb.watch(PREFormer, log = "all")
    checkpoint = None#torch.load(f"/home/prsood/projects/ctb-whkchun/prsood/TAV_Train/MAEncoder/aht69be1/lively-sweep-11/7.pt")
    # model.load_state_dict(checkpoint['model_state_dict'])
    # criterion = checkpoint['loss']
    # PREFormer = checkpoint['PREFormer']

    model , PREFormer= train_tav_network(model , PREFormer , df_train, df_val, criterion , lr, epoch ,  weight_decay,T_max, Metric , patience , clip , checkpoint )
    evaluate_tav(model , PREFormer , df_test, Metric)
    # cleanup()


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
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
        'lr': config.learning_rate,
        'clip': config.clip ,
        'batch_size':1,#config.batch_size ,
        'weight_decay':config.weight_decay ,
        'model':config.model,
        'T_max':config.T_max ,
        'seed':config.seed,
        'label_task':config.label_task,
        'mask':config.mask,
        'loss':"NewCrossEntropy",#config.loss,
        'beta':config.beta,
    }

    
    df = pd.read_pickle(f"{args.dataset}.pkl")
    if param_dict['label_task'] == "sentiment":
        number_index = "sentiment"
        label_index = "sentiment_label"
    else:
        number_index = "emotion"
        label_index = "emotion_label"

    # TODO: DATASET SPECIFIC
    if "IEMOCAP" in args.dataset:
        df_train = df[df['split'] == "train"]
        df_test = df[df['split'] == "test"]
        df_val = df[df['split'] == "val"]
    else:
        df = df[df["audio_shape"] > 10000] # Still seeing what the best configuration is for these
        # df = df[ (df['emotion_label'] != "fear")  & (df['emotion_label'] != "disgust")]
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
        'output_dim':len(weights) ,
        'dropout':config.dropout,
        'early_div':config.early_div,
        'num_layers':config.num_layers,
        'learn_PosEmbeddings':config.learn_PosEmbeddings,
    }
    
    param_dict['weights'] = weights
    param_dict['label2id'] = label2id
    param_dict['id2label'] = id2label

    print(f" in main \n param_dict = {param_dict} \n model_param = {model_param} \n df {args.dataset} , with df = {len(df)} \n ")
    runModel("cuda" ,df_train , df_val , df_test ,param_dict , model_param  )
    
    
if __name__ == '__main__':
    main()


# [14, 7, 13, 9, 11, 3, 19, 2, 6, 14, 6, 9, 12, 4, 5, 22, 6, 10, 10, 1, 12, 13, 4, 8, 9, 14, 14, 2, 6, 13, 15, 4, 15, 11, 14, 5, 6, 2]
# [14, 21, 34, 43, 54, 57, 76, 78, 84, 98, 104, 113, 125, 129, 134, 156, 162, 172, 182, 183, 195, 208, 212, 220, 229, 243, 257, 259, 265, 278, 293, 297, 312, 323, 337, 342, 348, 350]

