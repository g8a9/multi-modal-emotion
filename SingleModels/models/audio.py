import pdb
import numpy as np
import torch
import torch.nn as nn
from torch import nn
import sys
import random

from transformers import AutoProcessor , AutoModel
from utils.global_functions import pool

from torch.nn.utils.rnn import pad_sequence

import h5py
try:
    VIDEOS =  h5py.File('../../data/videos_context.hdf5','r', libver='latest' , swmr=True)
    AUDIOS =  h5py.File('../../data/audio.hdf5','r', libver='latest' , swmr=True)
except:
    VIDEOS =  h5py.File('data/videos_context.hdf5','r', libver='latest' , swmr=True)
    AUDIOS =  h5py.File('data/audio.hdf5','r', libver='latest' , swmr=True)
    
    
def get_white_noise(signal:torch.Tensor,SNR) -> torch.Tensor :
  # @author: sleek_eagle
    shap = signal.shape
    signal = torch.flatten(signal)
    #RMS value of signal
    RMS_s=torch.sqrt(torch.mean(signal**2))
    #RMS values of noise
    RMS_n=torch.sqrt(RMS_s**2/(pow(10,SNR/100)))
    noise=torch.normal(0.0, RMS_n.item() , signal.shape)
    return noise.reshape(shap)

def ret0(signal , SNR) -> int:
    return 0


def speech_file_to_array_fn(path , check = "train"):
    func_ = [ret0, get_white_noise]
    singular_func_ = random.choices(population=func_, weights=[.5 , .5], k=1)[0]
    
    speech_array = torch.Tensor(AUDIOS[f"{check}_{path.split('/')[-1][:-4]}"][()])
    if check == "train":
        speech_array += singular_func_(speech_array,SNR=10)
    return  speech_array


# TODO: DATASET SPECIFIC
PROC = AutoProcessor.from_pretrained("audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim")

def collate_batch(batch , check): # batch is a pseudo pandas array of two columns
    """
    Here we are going to take some raw-input and pre-process them into what we need
    So we can see all the steps in the ML as they go along
    """
    speech_list = []
    label_list = []
    speech_list_mask = None


    for (input , label) in batch:
        speech_list.append(speech_file_to_array_fn(input , check))
        

        # TODO: DATASET SPECIFIC
        label_list.append(label)

    numpy_speech_list = [item.numpy() for item in speech_list] 
    # speech_list_input_values = torch.Tensor(np.array(PROC( numpy_speech_list , sampling_rate = 16000 , padding = True)['input_values']))


    """ Audio works as intended as per test_audio_mask.ipynb"""
    speech_list_mask = 0#torch.Tensor(np.array(PROC( numpy_speech_list, sampling_rate = 16000 , padding = True)['attention_mask']))
    del numpy_speech_list
    
    speech_list_input_values = pad_sequence(speech_list , batch_first = True, padding_value=0)
    del speech_list
    # speech_list_input_values = (speech_list_input_values1 + speech_list_input_values2)/2

    # Batch_size , 16 , 224 , 224 

    audio_features = {'audio_features':speech_list_input_values , 
            'attention_mask':speech_list_mask, 
        }

    return audio_features , torch.Tensor(np.array(label_list))


class Wav2Vec2ForSpeechClassification(nn.Module):
    def __init__(self, args):
        super(Wav2Vec2ForSpeechClassification, self).__init__()
        self.output_dim = args['output_dim']
        self.dropout = args['dropout']
        self.learn_PosEmbeddings = args['learn_PosEmbeddings']
        self.num_layers = args['num_layers']

        self.test_ctr = 1
        self.train_ctr = 1

        self.wav2vec2 = AutoModel.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
        
        self.aud_norm = nn.LayerNorm(1024)

        self.dropout = nn.Dropout(self.dropout)
        self.linear1 = nn.Linear(1024, self.output_dim)

   
    def forward(self, audio_features , check):
        aud_outputs = self.wav2vec2(audio_features)[0]
        aud_outputs = torch.mean(aud_outputs, dim=1)
        
        aud_outputs = self.aud_norm(aud_outputs)
        
        if check == "train":
            aud_outputs = self.dropout(aud_outputs)
        aud_outputs = self.linear1(aud_outputs)

        return aud_outputs # returns [batch_size,output_dim]