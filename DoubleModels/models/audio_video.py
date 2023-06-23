from glob import glob
from transformers import logging
logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore") 
import torch
from torch import nn
from transformers import  VideoMAEModel  , AutoModel , AutoProcessor , AutoConfig
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices

import numpy as np
from numpy.random import choice
import random

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    Resize,
    PILToTensor,
    ToPILImage,
    Normalize,
    RandomErasing,
    # RandomShortSideScale,
)
from utils.global_functions import Crop
from torch.nn.utils.rnn import pad_sequence
from torch import nn

import h5py
try:
    VIDEOS =  h5py.File('../../data/videos_context.hdf5','r', libver='latest' , swmr=True)
    AUDIOS =  h5py.File('../../data/audio.hdf5','r', libver='latest' , swmr=True)
except:
    VIDEOS =  h5py.File('data/videos_context.hdf5','r', libver='latest' , swmr=True)
    AUDIOS =  h5py.File('data/audio.hdf5','r', libver='latest' , swmr=True)

def videoMAE_features(path , timings , speaker , check):

    if check == "train":
        transform = Compose(
                        [
                            # TODO: DATASET SPECIFIC                            
                            RandomHorizontalFlip(p=0) if speaker == None else Crop((120,2,245,355)) if speaker else Crop((120,362,245,355)), # if not IEMOCAP then do nothing, else 
                            # Hone in on either the left_speaker or right_speaker in the video
                            # Lambda(lambda x: body_face(x , bbox)), # cropped bodies only
                            RandomHorizontalFlip(p=0.5), # 
                            RandomVerticalFlip(p=0.5), # 
                        ]
                    )
    else:
        transform = Compose(
                        [
                            # TODO: DATASET SPECIFIC    
                            # Lambda(lambda x: body_face(x , bbox)), # cropped bodies only,                        
                            RandomHorizontalFlip(p=0) if speaker == None else Crop((120,2,245,355)) if speaker else Crop((120,362,245,355)),
                        ]
                    )
   
    
    video = torch.Tensor(   VIDEOS[f"{check}_{path.split('/')[-1][:-4]}"][()]  ) # H5PY, how to remove data after loading it into memory
    video = transform(video)
    
    return video



    
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

    video_list = []
    input_list = []
    speech_list = []
    label_list = []
    
    speech_list_mask = None
    vid_mask = None


    for (input , label) in batch:
        audio_path = input[0]
        speech_list.append(speech_file_to_array_fn(audio_path , check))
        
        vid_features = input[1]#[6:] for debug

        # TODO: DATASET SPECIFIC
        video_list.append(videoMAE_features(vid_features['vid_path'] , vid_features['timings'] , vid_features['speaker'] , check))
        label_list.append(label)
    batch_size = len(label_list)
    
    
    vid_mask = torch.randint(-13, 2, (batch_size, 1568)) # 8*14*14 = 1568 is just the sequence length of every video with VideoMAE, so i just hardcoded it, 
    vid_mask[vid_mask > 0] = 0
    vid_mask = vid_mask.bool()
    # now we have a random mask over values so it can generalize better   
    x = torch.count_nonzero(vid_mask.int()).item()
    rem = (1568*batch_size - x)%batch_size
    if rem != 0:
        idx = torch.where(vid_mask.view(-1) == 0)[0]  # get all indicies of 0 in flat tensor
        num_to_change =  rem# as follows from example abow
        idx_to_change = choice(idx, size=num_to_change, replace=False)
        vid_mask.view(-1)[idx_to_change] = 1
    

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

    visual_embeds = {'visual_embeds':torch.stack(video_list).permute(0,2,1,3,4) , 
            'attention_mask':vid_mask , 
        }
    return [ audio_features , visual_embeds] , torch.Tensor(np.array(label_list))


class AudioVideoClassifier(nn.Module):
    """
    Model for Bert and VideoMAE classifier
    """

    def __init__(self, args):
        super(AudioVideoClassifier, self).__init__()
        self.output_dim = args['output_dim']
        self.dropout = args['dropout']
        self.learn_PosEmbeddings = args['learn_PosEmbeddings']
        self.num_layers = args['num_layers']

        self.test_ctr = 1
        self.train_ctr = 1

        self.wav2vec2 = AutoModel.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
        self.wav_2_768_2 = nn.Linear(1024 , 768)
        self.wav_2_768_2.weight = torch.nn.init.xavier_normal_(self.wav_2_768_2.weight)
        self.videomae = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        
        self.vid_norm = nn.LayerNorm(768)
        self.aud_norm = nn.LayerNorm(768)


        self.dropout = nn.Dropout(self.dropout)
        self.linear1 = nn.Linear(768*2, self.output_dim)
        

    
    def forward(self, audio_features , video_embeds , visual_mask  , check = "train"):        
        #Transformer Time
        
        
        aud_outputs = self.wav2vec2(audio_features)[0]
        
        del audio_features
        
        # aud_output = torch.rand((1 , 64 , 1024))
        aud_outputs = torch.mean(self.wav_2_768_2(aud_outputs), dim=1)
        

        vid_outputs = self.videomae(video_embeds , visual_mask)[0] # Now it has 2 dimensions 
        
        del video_embeds
        del visual_mask
        
        vid_outputs = torch.mean(vid_outputs, dim=1) # Now it has 2 dimensions 

        
        aud_outputs = self.aud_norm(aud_outputs)
        vid_outputs = self.vid_norm(vid_outputs) 

        # All of these ouputs are 4 x [batch_size , 768]

        #Concatenate Outputs
        tav = torch.cat([ aud_outputs , vid_outputs],dim=1)
        
        del aud_outputs 
        del vid_outputs

        #Classifier Head
        if check == "train":
            tav = self.dropout(tav)
        tav = self.linear1(tav)

        return tav # returns [batch_size,output_dim]