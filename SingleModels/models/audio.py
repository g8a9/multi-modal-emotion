import pdb
import numpy as np
import torch
import torch.nn as nn
from torch import nn
import sys
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)

from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor
import torchaudio
from utils.global_functions import pool


def speech_file_to_array_fn(path , target_sampling_rate):

    # path = path[6:]
    
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return pool(speech , "mean")

def pad_and_process_values(input_val, processor , target_sampling_rate):
    # m = 166571 # takes 8 minutes to get this value on a pre-processing step with 10K data points
    m = max(map(np.shape , input_val))[0]
    inp = []
    for matrix in input_val:
        n = matrix.shape[0]
        mat = np.pad(matrix, (0, m-n), 'constant')
        inp.append(mat)
    

    result = processor(raw_speech = inp, sampling_rate=target_sampling_rate , return_attention_mask=True)

    input_values = result['input_values']

    attention_mask = result['attention_mask']

    return input_values  , attention_mask


def collate_batch(batch): # batch is a pseudo pandas array of two columns
    """
    Here we are going to take some raw-input and pre-process them into what we need
    So we can see all the steps in the ML as they go along
    """
    # print("in collate batch broski")
    # model_path = "facebook/wav2vec2-large-960h"
    # model_path = "facebook/wav2vec2-base-960h"

    # model_path = "facebook/wav2vec2-large-960h-lv60-self"
    model_path = "superb/wav2vec2-base-superb-er"


    feature_extractor =  Wav2Vec2FeatureExtractor.from_pretrained(model_path) 
    target_sampling_rate = feature_extractor.sampling_rate

    # processor = Wav2Vec2Processor.from_pretrained(model_path , return_attention_mask=True)
    # target_sampling_rate = processor.feature_extractor.sampling_rate

    speech_list = []
    label_list = []
   
    for (input_path , label) in batch: # change the input path to values that the model can use
        # for audios case we are changing the path to a like a matrix of size 100K
        speech_list.append(speech_file_to_array_fn(input_path , target_sampling_rate))
        label_list.append(label)

    speech_list , att_mask = pad_and_process_values(speech_list , feature_extractor , target_sampling_rate )
    # result = processor(speech_list, sampling_rate=target_sampling_rate , padding = True , max_length=int(target_sampling_rate * 20.0), truncation=True)
    # speech_list = result['input_values']
    

    return [torch.Tensor(np.array(speech_list)) , torch.Tensor(np.array(att_mask))], torch.Tensor(np.array(label_list))
   

class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.hidden_layers = config.hidden_layers[0]

        #batch size(2,4,8) , hidden_size (768) -> hidden_size (768) , hidden_layers (300)

        self.dense = nn.LSTM( input_size = config.hidden_size , hidden_size = self.hidden_layers, batch_first = True)

        self.dense1 = nn.Linear( 300 ,  200)

        self.dense2 = nn.Linear( 200 ,  100)

        self.dropout = nn.Dropout(config.final_dropout)

        self.tanh = nn.Tanh()

        self.out_proj = nn.Linear(100, config.num_labels)



    def forward(self, features, **kwargs):
        x = features
        
        x,_ = self.dense(x)

        x = self.tanh(x)
        x = self.dropout(x)

        x = self.dense1(x)
        x = self.tanh(x)
        x = self.dropout(x)

        x = self.dense2(x)
        x = self.tanh(x)
        x = self.dropout(x)

        x = self.out_proj(x)

        x = torch.mean(x, dim=1)

        return x

class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config , rank):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config
        self.rank = rank
        self.lol = None
        if self.rank == None:
            self.lol =  "cpu"
        else:
            self.lol = "cuda"



        self.wav2vec2 = Wav2Vec2Model(config).to(device=self.lol)
        self.classifier = Wav2Vec2ClassificationHead(config).to(device=self.lol)
        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(self, hidden_states, mode="mean"):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(self, input_values, attention_mask=None, output_attentions=None, output_hidden_states=None, return_dict=None, labels=None,):
        # print(f"We are on GPU {self.rank} with input shape {input_values.shape}")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print("before wav2vec2" , flush = True)
        outputs = self.wav2vec2(input_values.to(device=self.lol),attention_mask=attention_mask.to(device=self.lol),output_attentions=output_attentions,output_hidden_states=output_hidden_states,return_dict=return_dict)
        print(f"so our hidden states value is \n {outputs[0]} \n")
        
        # print("after wav2vec2" , flush = True)
        # outputs is made up of 2 elements hidden states at pos 0 and extracted features at pos 1     
        hidden_states = outputs[0]
        # print(f"dimension of hidden states for classification is {hidden_states.shape}")
        # print("after hidden states" , flush = True)
        # hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode) # 3 dim matrix and we want to make it 2 dim
        # print("before classifier" , flush = True)
        logits = self.classifier(hidden_states)
        # we are NOT running a softmax or sigmoid as our last layer, we dont need it 
        return logits
