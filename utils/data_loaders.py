import warnings
warnings.filterwarnings("ignore") 
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer , AutoTokenizer
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe, vocab
import dill as pickle

#------------------------------------------------------------TRIPLE MODELS BELOW--------------------------------------------------------------------
class TextAudioVideoDataset(Dataset):
    """
    feature_col1 : audio paths 
    feature_col2 : video paths 
    feature_col3 : text
    """

    def __init__(self, df , max_len , feature_col1 , feature_col2 , feature_col3  , label_col , timings = None , speaker = None):
        
        tokenizer = AutoTokenizer.from_pretrained('j-hartmann/emotion-english-distilroberta-base')

        self.grad = df['dialog'].value_counts().sort_index().tolist()
        self.grad_sum = [sum(self.grad[:i+1]) for i,x in enumerate(self.grad)]
        self.ctr = 0

        self.labels = df[label_col].values.tolist()

        self.audio_path = df[feature_col1].values
        self.video_path = df[feature_col2].values


        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = max_len, truncation=True,
                                return_tensors="pt") for text in df[feature_col3]]

        if timings != None:
            self.timings = df[timings].values.tolist()
        else:
            self.timings = [None]*len(self.labels)
        
        try: # if speaker is in there, then we are doing IEMOCAP otherwise it will fail and go onto MELD
            self.speaker = df[speaker].values.tolist()
        except:
            self.speaker = [None]*len(self.labels)

    def retGradAccum(self , i: int) -> int:
        RETgrad = self.grad[self.ctr]
        RETgrad_sum = self.grad_sum[self.ctr]
        if i + 1 == self.grad_sum[self.ctr]:
            self.ctr += 1
        if self.ctr == len(self.grad):
            self.resetCtr()
        return RETgrad , RETgrad_sum

    def resetCtr(self):
        self.ctr = 0
        
    def __len__(self): return len(self.labels)


    def __getitem__(self, idx): 
        # d = {"text" : , "video_path" : self.video_path[idx] , "labels" : np.array(self.labels[idx])}
        
        return [self.texts[idx] , self.audio_path[idx] , {"vid_path" : self.video_path[idx] , "timings":self.timings[idx] , "speaker" :self.speaker[idx]}] , np.array(self.labels[idx])

#------------------------------------------------------------DOUBLE MODELS BELOW--------------------------------------------------------------------

class AudioVideoDataset(Dataset):
    """
    feature_col1 : audio paths 
    feature_col2 : video paths 
    """

    def __init__(self, df , feature_col1 , feature_col2  , label_col , timings = None , speaker = None):
        

        self.labels = df[label_col].values.tolist()

        self.audio_path = df[feature_col1].values
        self.video_path = df[feature_col2].values


        if timings != None:
            self.timings = df[timings].values.tolist()
        else:
            self.timings = [None]*len(self.labels)
        
        try: # if speaker is in there, then we are doing IEMOCAP otherwise it will fail and go onto MELD
            self.speaker = df[speaker].values.tolist()
        except:
            self.speaker = [None]*len(self.labels)
        


    def __len__(self): return len(self.labels)


    def __getitem__(self, idx): 
        # d = {"text" : , "video_path" : self.video_path[idx] , "labels" : np.array(self.labels[idx])}
        
        return [self.audio_path[idx] , {"vid_path" : self.video_path[idx] , "timings":self.timings[idx] , "speaker" :self.speaker[idx]}] , np.array(self.labels[idx])

class TextAudioDataset(Dataset):
    """
    feature_col1 : audio paths 
    feature_col2 : text
    """

    def __init__(self, df , max_len , feature_col1 , feature_col2  , label_col , timings = None , speaker = None):
        
        tokenizer = AutoTokenizer.from_pretrained('j-hartmann/emotion-english-distilroberta-base')

        self.labels = df[label_col].values.tolist()

        self.audio_path = df[feature_col1].values


        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = max_len, truncation=True,
                                return_tensors="pt") for text in df[feature_col2]]
        


    def __len__(self): return len(self.labels)


    def __getitem__(self, idx): 
        # d = {"text" : , "video_path" : self.video_path[idx] , "labels" : np.array(self.labels[idx])}
        
        return [self.texts[idx] , self.audio_path[idx]] , np.array(self.labels[idx])

class TextVideoDataset(Dataset):
    """
    feature_col1 : video paths 
    feature_col2 : text
    """

    def __init__(self, df , max_len , feature_col1 , feature_col2  , label_col , timings = None , speaker = None):
        
        tokenizer = AutoTokenizer.from_pretrained('j-hartmann/emotion-english-distilroberta-base')

        self.labels = df[label_col].values.tolist()
        self.video_path = df[feature_col1].values


        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = max_len, truncation=True,
                                return_tensors="pt") for text in df[feature_col2]]

        if timings != None:
            self.timings = df[timings].values.tolist()
        else:
            self.timings = [None]*len(self.labels)
        
        try: # if speaker is in there, then we are doing IEMOCAP otherwise it will fail and go onto MELD
            self.speaker = df[speaker].values.tolist()
        except:
            self.speaker = [None]*len(self.labels)
        


    def __len__(self): return len(self.labels)


    def __getitem__(self, idx): 
        # d = {"text" : , "video_path" : self.video_path[idx] , "labels" : np.array(self.labels[idx])}
        
        return [self.texts[idx] , {"vid_path" : self.video_path[idx] , "timings":self.timings[idx] , "speaker" :self.speaker[idx]}] , np.array(self.labels[idx])
class VBertDataset(Dataset):
    """
    feature_col1 : img paths 
    feature_col2 : text
    """

    def __init__(self, df , max_len , feature_col1 , feature_col2 , label_col):
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        self.labels = df[label_col].values.tolist()
        self.img_path = df[feature_col1].values


        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = max_len, truncation=True,
                                return_tensors="pt") for text in df[feature_col2]]
        


    def __len__(self): return len(self.labels)


    def __getitem__(self, idx): 
        # d = {"text" : , "img_path" : self.img_path[idx] , "labels" : np.array(self.labels[idx])}
        
        return [self.texts[idx] , self.img_path[idx]] , np.array(self.labels[idx])


#------------------------------------------------------------SINGLE MODELS BELOW--------------------------------------------------------------------


class VisualDataset(Dataset):
    """A basic dataset where the underlying data is a list of (x,y) tuples. Data
    returned from the dataset should be a (transform(x), y) tuple.
    Args:
    source      -- a list of (x,y) data samples
    transform   -- a torchvision.transforms transform
    """
    def __init__(self, df , feature_col , label_col):
        self.video_path = df[feature_col].values
        self.label = df[label_col].values

    def __len__(self): return len(self.label)
    
    def __getitem__(self, idx): return self.video_path[idx] , np.array(self.label[idx])

class ImageDataset(Dataset):
    """A basic dataset where the underlying data is a list of (x,y) tuples. Data
    returned from the dataset should be a (transform(x), y) tuple.
    Args:
    source      -- a list of (x,y) data samples
    transform   -- a torchvision.transforms transform
    """
    def __init__(self, df , feature_col , label_col):

        
        self.img_path = df[feature_col].values
        self.labels = df[label_col].values.tolist()
        

    def __len__(self): return len(self.labels)
    
    def __getitem__(self, idx): return self.img_path[idx] , np.array(self.labels[idx])

class Wav2VecAudioDataset(Dataset):
    def __init__(self, df , feature_col , label_col):
        """
        Initialize the dataset loader.

        :data: The dataset to be loaded.
        :labels: The labels for the dataset."""

        self.labels = df[label_col].values 
        # Want a tensor of all the features d
        self.audio_features = df[feature_col].values

       
        assert self.audio_features.shape == self.labels.shape  # TODO Double check that this asserts the right dims.
        self.length = self.audio_features.shape[0]


    def __getitem__(self, idx: int): return self.audio_features[idx], np.array(self.labels[idx])

    def __len__(self): return self.length
    

class BertDataset(Dataset):
    """
    Load text dataset for BERT processing.
    """

    def __init__(self, df , max_len , feature_col , label_col):
        tokenizer = AutoTokenizer.from_pretrained('j-hartmann/emotion-english-distilroberta-base')

        self.labels = df[label_col].values.tolist()
        
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = max_len, truncation=True,
                                return_tensors="pt") for text in df[feature_col]]

    def __len__(self): return len(self.labels)


    def __getitem__(self, idx): return self.texts[idx], self.labels[idx]
