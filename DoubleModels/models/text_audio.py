import torch
from torch import nn
from transformers import BertModel
import torch.nn as nn
import torch.nn.functional as F
from torchtext.vocab import GloVe

from transformers import Wav2Vec2Processor
import torchaudio
import numpy as np


def speech_file_to_array_fn(path , target_sampling_rate):

    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

def pad_and_process_values(input_val , num_channels , processor , target_sampling_rate):
    
    
    m = max(map(np.shape , input_val))[1]
    inp = []
    for matrix in input_val:
        n = matrix.shape[1]
        mat = np.resize(matrix , (1,m*num_channels))[0]
        mat[n*num_channels : m*num_channels] = 0
        inp.append(mat)
    # input_val = [   np.resize(matrix , (1,m*num_channels))[0] for matrix in input_val  ]

    

    result = processor(inp, sampling_rate=target_sampling_rate , padding = True)

    result = result['input_values']

    return result , m*num_channels


def collate_batch(batch): # batch is a pseudo pandas array of two columns
    """
    Here we are going to take some raw-input and pre-process them into what we need
    So we can see all the steps in the ML as they go along
    """
    model_path = "facebook/wav2vec2-large-960h"
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    target_sampling_rate = processor.feature_extractor.sampling_rate
    speech_list = []
    label_list = []
    number_of_channels = 6

    for (input_path , label) in batch:
        speech_list.append(speech_file_to_array_fn(input_path , target_sampling_rate))
        label_list.append(label)


    
    
    # label_list = torch.tensor(label_list, dtype=torch.int64)
    speech_list , pad_val = pad_and_process_values(speech_list , number_of_channels , processor , target_sampling_rate )

    # speech_list = np.array(speech_list)

    
    return [speech_list,pad_val] , torch.Tensor(np.array(label_list))
   


class BertAudioClassifier: 
    def __init__(self, args ,dropout=0.5):
        super(BertAudioClassifier, self).__init__()

        self.output_dim = args['output_dim']

        self.bert = BertModel.from_pretrained('bert-base-cased')

        self.dropout = nn.Dropout(dropout)

        self.linear = nn.Linear(768, self.output_dim)

        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        final_layer = self.relu(linear_output)
        linear_output = self.linear(final_layer)

        return linear_output

    def __init__(self, config):
        super().__init__()

        # hidden layers is a list of numbers, that isnt applicable here, so we will just take the first value
        self.config = config
        self.hidden_layers = config.hidden_layers[0]
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(self.hidden_layers, config.num_labels)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, features, size ,  **kwargs):
        self.dense = nn.Linear( size ,  self.hidden_layers)
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        x = torch.mean(x, dim = 1)
        x = self.sigmoid(x) 
        return x

    

class LSTMAudioClassifier:
