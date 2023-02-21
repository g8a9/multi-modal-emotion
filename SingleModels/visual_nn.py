if __name__ == '__main__' and not __package__:
    import sys
    sys.path.insert(0, "/home/prsood/projects/def-whkchun/prsood/multi-modal-emotion/")
    __package__ = 'SingleModels'
import torch


from .models.visual import VisualClassification
from utils.data_loaders import VisualDataset
from sklearn.model_selection import train_test_split
import wandb
import numpy as np
import pandas as pd


from .train_model.visual_training import visual_train, evaluate_visual
from utils.global_functions import arg_parse,hidden_layer_count , Metrics




# NEW IMPORTS
from torch.utils.data import DataLoader
from .models.visual import collate_batch
##



def prepare_dataloader(df , batch_size, pin_memory=True, num_workers=4):

    # CHANGE DATASET
    dataset = VisualDataset(df)
    #
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, 
            num_workers=num_workers, drop_last=False, shuffle=True, 
            collate_fn=collate_batch
        )
    return dataloader



def runModel(rank , df_train , df_val , df_test , param_dict , model_param , wandb):
    # CHANGE
    # max_len = 92

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


    
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(rank)).to(rank) #TODO: maybe error here?
    Metric = Metrics(num_classes = num_labels, id2label = id2label , rank = rank)
    df_train = prepare_dataloader(df_train  , batch_size = batch_size ) #TODO: maybe prepare dataloader?
    df_val = prepare_dataloader(df_val  , batch_size = batch_size )
    df_test = prepare_dataloader(df_test , batch_size = batch_size )

    # COMPLETE
    # Dummy
    if param_dict['model'] == "Video":
        # this is our baseline model from Transformers
        model = VisualClassification(model_param).to(rank)
        
    # Else if Resnet
     

    # Else Transformer    
    else:
        model = VisualClassification(model_param).to(rank)

    # print(f"config = \n {config} \n" , flush = True)
   
    wandb.watch(model, log = "all")  
    model = visual_train(model, df_train, df_val, criterion , lr, epoch ,  weight_decay,T_max, Metric , patience , clip )
     
    evaluate_visual(model , df_test ,  Metric)

    
def main():
    project_name = "MLP_test_visual"
    
    # print(f"RIGHT AFTER START: {args} , {type(args.hidden_layers)}")
    # wandb.init(project="Emotion_" + args.dataset + "_" + args.model, entity="ddi" , config = args)
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
    }

    model_param = {
        'input_dim':config.input_dim ,
        'output_dim':config.output_dim ,
        'lstm_layers':config.lstm_layers ,
        'hidden_layers': hidden_layer_count(config.hidden_layers) ,
    }

    
    df = pd.read_pickle(f"{args.dataset}.pkl")

    weights = torch.Tensor(list(dict(sorted((dict(1 - (df['emotion'].value_counts()/len(df))).items()))).values()))

    df_train, df_test, _, __ = train_test_split(df, df["emotion"], test_size = 0.25, random_state = param_dict['seed'] ,)# stratify=df["emotion"])
    df_train, df_val, _, __ = train_test_split(df_train, df_train["emotion"], test_size = 0.25, random_state = param_dict['seed'] ,)# stratify=df_train["emotion"])

    label2id = df.drop_duplicates('label').set_index('label').to_dict()['emotion']
    id2label = {v: k for k, v in label2id.items()}
    
    param_dict['weights'] = weights
    param_dict['label2id'] = label2id
    param_dict['id2label'] = id2label


    print(f" in main \n param_dict = {param_dict} \n model_param = {model_param}")

    runModel("cuda" , df_train , df_val ,df_test, param_dict , model_param , run)
  
    
if __name__ == '__main__':
    main()



