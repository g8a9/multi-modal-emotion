from transformers import logging
logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore")
from re import X
import torch
from argparse import ArgumentParser
import dill as pickle
import io
import numpy as np
from torchmetrics.classification import MulticlassF1Score , MulticlassRecall , MulticlassPrecision , MulticlassAccuracy , MulticlassConfusionMatrix 
from torchvision.transforms import functional as F
from torch import nn
from torch.nn import functional as Fnn


        

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
      for idx , frame in enumerate(frames):
          new_vid[idx] = F.crop(frame, *self.params)
      return new_vid

class F1_Loss(nn.Module):
    '''Calculate F1 score. Can work with gpu tensors
  
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''

    def __init__(self, num_classes = -1 , weights = torch.Tensor([1]), beta = 1, epsilon=1e-7):
        super().__init__()
        self.weights = weights 
        # Normalize thew weights
        if self.weights.sum() != 1:
            self.weights /= self.weights.sum()
        self.epsilon = epsilon
        self.beta = beta
        self.num_classes = num_classes
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, y_pred, y_true,):
        """
        preds first , truth/target second and must be int64
        """
        pred_sm = self.softmax(y_pred)
        y_pred = torch.argmax( pred_sm , dim = 1)

        y_true = Fnn.one_hot(y_true, self.num_classes).to(torch.float32)
        y_pred = Fnn.one_hot(y_pred, self.num_classes).to(torch.float32)

        cnf_matrix = y_true.T @ (pred_sm*y_pred)
        tp = torch.diag(cnf_matrix)
        fp = cnf_matrix.sum(axis=0) - tp  
        precision = tp / (tp + fp + self.epsilon)

        # fn = cnf_matrix.sum(axis=1) - tp
        # recall = tp / (tp + fn + self.epsilon)
        # # F0.5-Measure = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
        # f1 = ((1+self.beta**2)*precision*recall) / (((self.beta**2)*precision) + recall + self.epsilon)
        # f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        
        return 1 - (precision*self.weights).sum()

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
        self.scalarRec = MulticlassRecall(self.num_classes, self.top_k, 'macro', self.multidim_average, self.ignore_index, self.validate_args).to(device=self.lol)
        self.scalarPrec = MulticlassPrecision(self.num_classes, self.top_k, 'macro', self.multidim_average, self.ignore_index, self.validate_args).to(device=self.lol)
        self.scalarAcc = MulticlassAccuracy(self.num_classes, self.top_k, 'macro', self.multidim_average, self.ignore_index, self.validate_args).to(device=self.lol)

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


def arg_parse(description):
    """
    description : str , is the name you want to give to the parser usually the model_modality used
    """
   # pdb.set_trace()
    parser = ArgumentParser(description= f" Run experiments on {description} ")

    # parser.add_argument("--")
    parser.add_argument("--learning_rate" , "-l" , help="Set the learning rate"  , default=0.001, type=float)
    parser.add_argument("--epoch"  , "-e", help="Set the number of epochs"  , default=3, type = int)
    parser.add_argument("--batch_size", "-b", help="Set the batch_size"  , default=16, type=int)
    parser.add_argument("--weight_decay", "-w", help="Set the weight_decay" , default=0.000001, type=float)
    parser.add_argument("--clip", "-c", help="Set the gradient clip" , default=1.0, type=float)
    parser.add_argument("--patience", "-p", help="Set the patience" , default=10.0, type=float)
    parser.add_argument("--T_max", "-t", help="Set the gradient T_max" , default=10, type=int)
    parser.add_argument("--mask", "-ma", help="True/False on if we want to use masking in model" , default=False, type=bool)

    # Set the seed
    parser.add_argument("--seed", "-s", help="Set the random seed" , default=32, type=int)
    
    # These are values in the yaml that we set
    parser.add_argument("--dataset"  , "-d", help="The dataset we are using currently, or the folder the dataset is inside") 
    parser.add_argument("--model"  , "-m", help="The model we are using currently") 
    parser.add_argument("--label_task"  , "-lt", default='emotion' , help="What specific classification label we are using: Emotion or Sentiment") 

    # These are args that we use as input to the model
    parser.add_argument("--input_dim", "-z", help="Set the input dimension", default=2 ,type=int)
    parser.add_argument("--output_dim", "-y", help="Set the output dimension" , default=7, type=int)
    parser.add_argument("--lstm_layers"  , "-ll", help="set number of LSTM layers" , default=1  ,  type=int) 
    parser.add_argument("--hidden_layers"  , "-o", help="values corresponding to each hidden layer" , default="32,32" , type = str)
    parser.add_argument("--early_div"  , "-ed", help="If we should do division earlier in the transformer" , default=False , type = bool)
    parser.add_argument("--dropout"  , "-dr", help="the rate for each dropout layer" , default=0.1 , type = float)
    parser.add_argument("--num_layers"  , "-nl", help="the number of layers in our transformers model" , default=12 , type = int)
    return parser.parse_args()

 

