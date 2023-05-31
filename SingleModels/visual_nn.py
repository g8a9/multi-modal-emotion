if __name__ == '__main__' and not __package__:
    import sys
    sys.path.insert(0, "/home/prsood/projects/def-whkchun/prsood/multi-modal-emotion/")
    __package__ = 'SingleModels'
import torch

import os
from .models.visual import VisualClassification,ResNet50Classification
from utils.data_loaders import VisualDataset
from utils.global_functions import arg_parse , Metrics, MySampler , NewCrossEntropyLoss
from sklearn.model_selection import train_test_split
import wandb
import numpy as np
import pandas as pd


from .train_model.visual_training import visual_train, evaluate




# NEW IMPORTS
from torch.utils.data import DataLoader
from .models.visual import collate_batch
##


class BatchCollation:
    def __init__(self,  check  ) -> None:
        self.check = check
    
    def __call__(self, batch):
        return collate_batch(batch , self.check)

def prepare_dataloader(df , batch_size, label_task , epoch_switch , pin_memory=True, num_workers=8 , check = "train" ): # num_W = 8 kills it 
    """
    we load in our dataset, and we just make a random distributed sampler to evenly partition our 
    dataset on each GPU
    say we have 32 data points, if batch size = 8 then it will make 4 dataloaders of size 8 each 
    """    
    # TODO: DATASET SPECIFIC
    dataset = VisualDataset(df  , feature_col="video_path" , label_col=label_task ,  timings="timings" , speaker="speaker")
    if check == "train":
        labels = df[label_task].value_counts()
        class_counts = torch.Tensor(list(dict(sorted((dict((labels)).items()))).values())).to(int)

        samples_weight = torch.tensor([1/class_counts[t] for t in dataset.labels])
        print(samples_weight, "\n\n" , len(samples_weight))
        # sampler = WeightedRandomSampler(list(samples_weight), int(1*len(samples_weight)))
        sampler = MySampler(list(samples_weight), len(samples_weight) , replacement=True , epoch=0 , epoch_switch = epoch_switch)
        dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, 
                num_workers=num_workers ,drop_last=False, shuffle=False, sampler = sampler , collate_fn=BatchCollation(check=check))
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, 
                num_workers=num_workers ,drop_last=False, shuffle=False , collate_fn=BatchCollation(check=check))

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
    epoch_switch = param_dict['epoch_switch']

    num_labels = model_param['output_dim']
    
    if loss == "CrossEntropy":
        # criterion = NewCrossEntropyLoss(class_weights=weights.to(device)).to(device) # TODO: have to call epoch in forward function
        criterion = torch.nn.CrossEntropyLoss().to(device)
       
    elif loss == "NewCrossEntropy":
        # criterion = PrecisionLoss(num_classes = num_labels,weights=weights.to(device)).to(device)
        criterion = NewCrossEntropyLoss(class_weights = weights.to(device) , epoch_switch = epoch_switch).to(device)


    print(loss , flush = True)
    Metric = Metrics(num_classes = num_labels, id2label = id2label , rank = device)
    df_train = prepare_dataloader(df_train,  batch_size ,  label_task , epoch_switch , check = "train")
    df_val = prepare_dataloader(df_val,  batch_size ,  label_task , epoch_switch , check = "val")
    df_test = prepare_dataloader(df_test ,  batch_size,  label_task , epoch_switch , check = "val")

    
    model = VisualClassification(model_param).to(device)


    wandb.watch(model, log = "all")
    model = visual_train(model, df_train, df_val, criterion , lr, epoch ,  weight_decay,T_max, Metric , patience , clip )
    evaluate(model, df_test, Metric)


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
        'epoch':config.epoch,
        'patience': config.patience ,
        'lr': config.learning_rate,
        'clip': config.clip ,
        'batch_size': 8,#config.batch_size ,
        'weight_decay':config.weight_decay ,
        'model':config.model,
        'T_max':config.T_max ,
        'seed':config.seed,
        'label_task':config.label_task,
        'mask':config.mask,
        'loss':config.loss,
        'beta':config.beta,
        'epoch_switch':config.epoch_switch,

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
        # df = df[df['audio_shape'] > 3280] # Still seeing what the best configuration is for these
        df_train = df[df['split'] == "train"] 
        df_test = df[df['split'] == "test"] 
        df_val = df[df['split'] == "val"] 

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



