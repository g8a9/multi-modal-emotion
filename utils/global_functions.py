from transformers import logging
logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore")
from re import X
import torch
from argparse import ArgumentParser
import io
import numpy as np
from torchmetrics.classification import MulticlassF1Score , MulticlassRecall , MulticlassPrecision , MulticlassAccuracy , MulticlassConfusionMatrix 
from torchvision.transforms import functional as F
import wandb
import os
import collections
from torch.optim import AdamW
from torch.utils.data.sampler import Sampler
from torch import nn


class MySampler(Sampler):
    
    def __init__(self, weights, num_samples, replacement = True , epoch = 0 , epoch_switch = 2):
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement
        self.epoch = epoch
        self.epoch_switch = epoch_switch

    def __iter__(self):
        if self.epoch % self.epoch_switch == 0:
            print("we are in multinomial dataloader" , flush = True)
            rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement)
            yield from iter(rand_tensor.tolist())
        else:
            print("we are in iterative dataloader" , flush = True)
            # TODO: DATASET SPECIFIC
            yield from iter(range(self.num_samples))

    def __len__(self) -> int:
        return self.num_samples
    
class NewCrossEntropyLoss(nn.Module):
    """
    This criterion (`CrossEntropyLoss`) combines `LogSoftMax` and `NLLLoss` in one single class.
    
    NOTE: Computes per-element losses for a mini-batch (instead of the average loss over the entire mini-batch).
    """
    

    def __init__(self, class_weights , epoch_switch = 2):
        super().__init__()
        self.class_weights = class_weights
        self.epoch_switch = epoch_switch
        self.weightedCEL = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.normalCEL = torch.nn.CrossEntropyLoss()
        self.iter1 = 1
        self.iter2 = 1
        self.iter3 = 1

    def forward(self, logits, target , epoch ):
        if epoch % self.epoch_switch == 0:
            if self.iter1 == 1:
                print("We are in normal unweighted CEL" , flush = True)
                self.iter1 += 1
                self.iter2 = 1
                self.iter3 = 1
            return self.normalCEL(logits, target)
        else:
            if self.iter2 == 1 or self.iter3 == 1200:
                print("We are in weighted CEL" , flush = True)
                self.iter2 += 1
                self.iter1 = 1
            self.iter3 += 1
            return self.weightedCEL(logits, target)



def pool(input : np.array , mode : str) -> np.array:
    """
    Supported modes are 'mean', 'max' and 'median'
    Given an array with one dimension, we take the mean max or
    median of it and return it
    """
    if mode == 'mean':
        return torch.Tensor(input.mean(0))
    elif mode == 'max':
        return torch.Tensor(input.max(0))
    elif mode == 'median':
        return torch.Tensor(np.median(input,0))
    else:
        raise NotImplementedError("The supported modes are 'mean', 'max' and 'median'")

class Crop:

  def __init__(self , params):
    self.params = params

  def __call__(self , frames):
      a,b,c,d = self.params
      new_vid = torch.rand((3 , 16 , c , d))
      #Now we are only focusing just the last two dimensions which are width and height of an image, thus we crop each frame in a video one by one
      for idx , frame in enumerate(frames):
          new_vid[idx] = F.crop(frame, *self.params)
      return new_vid

class Metrics:
    """
    Here is where we compute the scores using torch metric 
    """
    def __init__(self, num_classes : int , id2label : dict ,rank , top_k = 1 , average = 'none' , multidim_average='global', ignore_index=None, validate_args=False) -> None:
        self.num_classes = num_classes
        self.top_k = top_k
        self.average = average
        self.multidim_average = multidim_average
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self.lol = None
        self.rank = rank
        if self.rank == "cpu":
            self.lol = "cpu"
        elif self.rank == "cuda":
            self.lol = "cuda"       
        else:
            self.lol = rank
        
        self.confusionMatrix = MulticlassConfusionMatrix( num_classes = self.num_classes , ignore_index=self.ignore_index , normalize='none' , validate_args=self.validate_args).to(device=self.lol)
        self.multiF1 = MulticlassF1Score(self.num_classes, self.top_k, self.average, self.multidim_average, self.ignore_index, self.validate_args).to(device=self.lol)
        self.multiRec = MulticlassRecall(self.num_classes, self.top_k, self.average, self.multidim_average, self.ignore_index, self.validate_args).to(device=self.lol)
        self.multiPrec = MulticlassPrecision(self.num_classes, self.top_k, self.average, self.multidim_average, self.ignore_index, self.validate_args).to(device=self.lol)
        self.multiAcc = MulticlassAccuracy(self.num_classes, self.top_k, self.average, self.multidim_average, self.ignore_index, self.validate_args).to(device=self.lol)
        self.scalarMacroF1 = MulticlassF1Score(self.num_classes, self.top_k, 'macro', self.multidim_average, self.ignore_index, self.validate_args).to(device=self.lol)
        self.scalarWeightedF1 = MulticlassF1Score(self.num_classes, self.top_k, 'weighted', self.multidim_average, self.ignore_index, self.validate_args).to(device=self.lol)
        self.scalarPrec = MulticlassPrecision(self.num_classes, self.top_k, 'macro', self.multidim_average, self.ignore_index, self.validate_args).to(device=self.lol)
        self.scalarRec = MulticlassRecall(self.num_classes, self.top_k, 'macro', self.multidim_average, self.ignore_index, self.validate_args).to(device=self.lol)
        self.scalarAcc = MulticlassAccuracy(self.num_classes, self.top_k, 'weighted', self.multidim_average, self.ignore_index, self.validate_args).to(device=self.lol)

        # self.MetricList = [self.confusionMatrix,self.multiF1,self.multiRec,self.multiPrec,self.multiAcc,self.scalarMacroF1,self.scalarWeightedF1,self.scalarRec,self.scalarPrec,self.scalarAcc]

        self.id2label = id2label

        
    
    def update_metrics(self, preds , target ):
        self.confusionMatrix.update(preds,target)
        self.multiF1.update(preds,target)
        self.multiRec.update(preds,target)
        self.multiPrec.update(preds,target)
        self.multiAcc.update(preds,target)
        self.scalarMacroF1.update(preds,target)
        self.scalarWeightedF1.update(preds,target)
        self.scalarRec.update(preds,target)
        self.scalarPrec.update(preds,target)
        self.scalarAcc.update(preds,target)
        

    def reset_metrics(self):
        self.scalarMacroF1.reset()
        self.scalarWeightedF1.reset()
        self.scalarRec.reset()
        self.scalarAcc.reset()
        self.scalarPrec.reset()
        self.multiF1.reset()
        self.multiRec.reset()
        self.multiPrec.reset()
        self.multiAcc.reset()
        self.confusionMatrix.reset()

    
    def compute_scores(self, name): # name is train test or val
        scalarMacroF1 = self.scalarMacroF1.compute()
        scalarWeightedF1 = self.scalarWeightedF1.compute()
        scalarRec = self.scalarRec.compute()
        scalarAccuracy = self.scalarAcc.compute()
        scalarPrec = self.scalarPrec.compute()
        multiF1 = self.multiF1.compute()
        multiRec = self.multiRec.compute()
        multiPrec = self.multiPrec.compute()
        multiAccuracy = self.multiAcc.compute()
        confusionMatrix = self.confusionMatrix.compute()
        return { name +  "/" + "multiAcc/" + self.id2label[v]: k.item() for v, k in enumerate(multiAccuracy)}, { name +  "/" + "multiF1/" + self.id2label[v]: k.item() for v, k in enumerate(multiF1)} , { name +  "/" + "multiRec/" + self.id2label[v]: k.item() for v, k in enumerate(multiRec)} , { name +  "/" + "multiPrec/" + self.id2label[v]: k.item() for v, k in enumerate(multiPrec)}, scalarAccuracy,scalarMacroF1,scalarWeightedF1, scalarRec, scalarPrec , confusionMatrix

def hidden_layer_count(string):
    """
    checks that dimensions of hidden layers are consistent
    """
    x = string.split(',')
    if len(x) == 1 or len(x)%2 == 0:
        return list(map(int, x))
    raise ArgumentParser.ArgumentTypeError(f'Missing a dimension in hidden layers, Need to input an even amount of dimensions, that is greater then 1 : {string}')

def save_model(model , optimizer , criterion , scheduler , epoch , step , path , log_val):
    """
    path: no trailing backslash
    """
    if epoch == 0 and step < log_val + 1:
        try:
            os.makedirs(f"{path}/{wandb.run.project}/")
        except:
            pass            
        try:
            os.makedirs(f"{path}/{wandb.run.project}/{wandb.run.sweep_id}/")
        except:
            pass            
        try:
            os.makedirs(f"{path}/{wandb.run.project}/{wandb.run.sweep_id}/{wandb.run.name}/")
        except:
            os.makedirs(f"{path}/{wandb.run.project}/{wandb.run.sweep_id}/copy_{wandb.run.name}/")
    try:
        torch.save({
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion.state_dict(),
            'scheduler' : scheduler.state_dict(),
            }, f"{path}/{wandb.run.project}/{wandb.run.sweep_id}/{wandb.run.name}/best.pt")
    except:
        torch.save({
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion.state_dict(),
            'scheduler' : scheduler.state_dict(),
            }, f"{path}/{wandb.run.project}/{wandb.run.sweep_id}/copy_{wandb.run.name}/best.pt")

def load_model(model , optimizer , criterion , path):
    """
    path: no trailing backslash
    """
    p = ""
    try:
        checkpoint = torch.load(f"{path}/{wandb.run.project}/{wandb.run.sweep_id}/{wandb.run.name}/best.pt")
        p = f"{path}/{wandb.run.project}/{wandb.run.sweep_id}/{wandb.run.name}/best.pt"
    except:
        checkpoint = torch.load(f"{path}/{wandb.run.project}/{wandb.run.sweep_id}/copy_{wandb.run.name}/best.pt")
        p = f"{path}/{wandb.run.project}/{wandb.run.sweep_id}/copy_{wandb.run.name}/best.pt"

    print(f"Current best model is on epoch {checkpoint['epoch']}, and step {checkpoint['step']} on path {p}" , flush = True)

    model.load_state_dict(checkpoint['model_state_dict']) 
    model_params = [ param for param in model.parameters() if param.requires_grad == True]
    optimizer = AdamW(model_params)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
    criterion.load_state_dict(checkpoint['loss'])

        
    return model , optimizer , criterion


def arg_parse(description):
    """
    description : str , is the name you want to give to the parser usually the model_modality used
    """
   # pdb.set_trace()
    parser = ArgumentParser(description= f" Run experiments on {description} ")

    # parser.add_argument("--")
    parser.add_argument("--learning_rate" , "-l" , help="Set the learning rate"  , default=0.000001, type=float)
    parser.add_argument("--epoch"  , "-e", help="Set the number of epochs"  , default=3, type = int)
    parser.add_argument("--batch_size", "-b", help="Set the batch_size"  , default=1, type=int)
    parser.add_argument("--weight_decay", "-w", help="Set the weight_decay" , default=0.0001, type=float)
    parser.add_argument("--clip", "-c", help="Set the gradient clip" , default=1.0, type=float)
    parser.add_argument("--epoch_switch", "-es", help="Set the epoch to switch from iterative to weightedRandomSampler" , default=2, type=int)
    parser.add_argument("--patience", "-p", help="Set the patience" , default=10.0, type=float)
    parser.add_argument("--T_max", "-t", help="Set the gradient T_max" , default=2, type=int)
    parser.add_argument("--mask", "-ma", help="True/False on if we want to use masking in model" , default=False, type=bool)
    parser.add_argument("--loss", "-ls", help="Which Loss function we are going to use" , default="NewCrossEntropy", type=str)
    parser.add_argument("--beta", "-beta", help="For FBeta loss, what beta to pick" , default=1, type=float)

    # Set the seed
    parser.add_argument("--seed", "-s", help="Set the random seed" , default=32, type=int)
    
    # These are values in the yaml that we set
    parser.add_argument("--dataset"  , "-d", help="The dataset we are using currently, or the folder the dataset is inside", default = "../data/text_audio_video_emotion_data") 
    parser.add_argument("--model"  , "-m", help="The model we are using currently", default = "MAE_encoder") 
    parser.add_argument("--label_task"  , "-lt", help="What specific classification label we are using: Emotion or Sentiment", default = 'emotion') 

    # These are args that we use as input to the model
    parser.add_argument("--input_dim", "-z", help="Set the input dimension", default=2 ,type=int)
    parser.add_argument("--output_dim", "-y", help="Set the output dimension" , default=7, type=int)
    parser.add_argument("--lstm_layers"  , "-ll", help="set number of LSTM layers" , default=1  ,  type=int) 
    parser.add_argument("--hidden_layers"  , "-o", help="values corresponding to each hidden layer" , default="32,32" , type = str)
    parser.add_argument("--early_div"  , "-ed", help="If we should do division earlier in the transformer" , default=False , type = bool)
    parser.add_argument("--dropout"  , "-dr", help="the rate for each dropout layer" , default=0.5 , type = float)
    parser.add_argument("--num_layers"  , "-nl", help="the number of layers in our transformers model" , default=12 , type = int)
    parser.add_argument("--learn_PosEmbeddings"  , "-lpe", help="If we should learn our positional embeddings" , default=True , type = bool)
    return parser.parse_args()

     