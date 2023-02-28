import pdb
import numpy as np
import torch
import torch.nn as nn
from torch import nn
import sys
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model

from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor
import torchaudio
from torch.nn.utils.rnn import pad_sequence

def speech_file_to_array_fn(path , target_sampling_rate):
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze()
    return torch.mean(speech, dim=0) if len(speech.shape) > 1 else speech # over the channel dimension


def collate_batch(batch): # batch is a pseudo pandas array of two columns
    """
    Here we are going to take some raw-input and pre-process them into what we need
    So we can see all the steps in the ML as they go along
    """

    target_sampling_rate = 16000 # Wav2Vec2 specific

    speech_list = []
    label_list = []
   
    for (input_path , label) in batch: # change the input path to values that the model can use
        # for audios case we are changing the path to a like a matrix of size 100K
        speech_list.append(speech_file_to_array_fn(input_path , target_sampling_rate))
        label_list.append(label)

    speech_list = pad_sequence(speech_list , batch_first = True)

    return speech_list, torch.Tensor(np.array(label_list))
   

class Wav2Vec2ForSpeechClassification(nn.Module):
    def __init__(self, args ,  dropout = .5):
        super(Wav2Vec2ForSpeechClassification , self).__init__()
        # model_path = "facebook/wav2vec2-large-960h"
        # model_path = "facebook/wav2vec2-large-960h-lv60-self"
        model_path = "superb/wav2vec2-base-superb-er"
        # model_path = "facebook/wav2vec2-base-960h"
        self.output_dim = args['output_dim']
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_path) 
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(768, self.output_dim)
        

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
    
        hidden_states = torch.mean(outputs[0], dim=1)

        x = self.dropout(hidden_states)
        x = self.out_proj(x)
        # we are NOT running a softmax or sigmoid as our last layer, we dont need it 
        return x

