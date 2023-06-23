import torch
from torch import nn
from transformers import BertModel
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoProcessor , AutoModel
import torchaudio
import numpy as np
import random
from torch.nn.utils.rnn import pad_sequence

import h5py
try:
    AUDIOS =  h5py.File('../../data/audio.hdf5','r', libver='latest' , swmr=True)
except:
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
    text_mask = []

    input_list = []
    speech_list = []
    label_list = []

    
    text_list_mask = None
    speech_list_mask = None


    for (input , label) in batch:
        text = input[0]
        input_list.append(text['input_ids'].tolist()[0])
        text_mask.append(text['attention_mask'].tolist()[0])
        audio_path = input[1]
        speech_list.append(speech_file_to_array_fn(audio_path , check))
        
        label_list.append(label)
    
    text_list_mask = torch.Tensor(np.array(text_mask))
    
    del text_mask    

    numpy_speech_list = [item.numpy() for item in speech_list] 
    # speech_list_input_values = torch.Tensor(np.array(PROC( numpy_speech_list , sampling_rate = 16000 , padding = True)['input_values']))


    """ Audio works as intended as per test_audio_mask.ipynb"""
    speech_list_mask = 0#torch.Tensor(np.array(PROC( numpy_speech_list, sampling_rate = 16000 , padding = True)['attention_mask']))
    del numpy_speech_list
    
    speech_list_input_values = pad_sequence(speech_list , batch_first = True, padding_value=0)
    del speech_list
    # speech_list_input_values = (speech_list_input_values1 + speech_list_input_values2)/2

    # Batch_size , 16 , 224 , 224 
    
    
    
    text = {'input_ids':torch.Tensor(np.array(input_list)).type(torch.LongTensor) , 
            'attention_mask':text_list_mask,
        }

    audio_features = {'audio_features':speech_list_input_values , 
            'attention_mask':speech_list_mask, 
        }

    return [text , audio_features ] , torch.Tensor(np.array(label_list))



class BertAudioClassifier(nn.Module): 
    def __init__(self, args):
        super(BertAudioClassifier, self).__init__()
        self.output_dim = args['output_dim']
        self.dropout = args['dropout']
        self.learn_PosEmbeddings = args['learn_PosEmbeddings']
        self.num_layers = args['num_layers']

        self.test_ctr = 1
        self.train_ctr = 1

        self.bert = AutoModel.from_pretrained('j-hartmann/emotion-english-distilroberta-base')
        self.wav2vec2 = AutoModel.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
        self.wav_2_768_2 = nn.Linear(1024 , 768)
        self.wav_2_768_2.weight = torch.nn.init.xavier_normal_(self.wav_2_768_2.weight)
        
        self.bert_norm = nn.LayerNorm(768)
        self.aud_norm = nn.LayerNorm(768)

        self.dropout = nn.Dropout(self.dropout)
        self.linear1 = nn.Linear(768*2, self.output_dim)

    def forward(self, input_ids , text_attention_mask , audio_features , check):
        _, text_outputs = self.bert(input_ids= input_ids, attention_mask=text_attention_mask,return_dict=False)
        
        del _
        del input_ids
        del text_attention_mask
        
        
        aud_outputs = self.wav2vec2(audio_features)[0]
        
        del audio_features
        
        # aud_output = torch.rand((1 , 64 , 1024))
        aud_outputs = torch.mean(self.wav_2_768_2(aud_outputs), dim=1)
        
        text_outputs = self.bert_norm(text_outputs) # 
        aud_outputs = self.aud_norm(aud_outputs)
        
        ta = torch.cat([text_outputs, aud_outputs],dim=1)
        
        del text_outputs
        del aud_outputs 

        #Classifier Head
        if check == "train":
            ta = self.dropout(ta)
        ta = self.linear1(ta)

        return ta # returns [batch_size,output_dim]