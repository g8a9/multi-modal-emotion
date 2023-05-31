from glob import glob
from transformers import logging
logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore") 
import torch
from torch import nn
from torch.nn import functional as F
import math
from transformers import  VideoMAEModel  , AutoModel , AutoProcessor , AutoConfig
from transformers import  VideoMAEModel  , AutoModel , AutoProcessor , AutoConfig
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices
from utils.TAVFormer import VideoMAEEncoder , Head

import numpy as np
from pytorchvideo.data.encoded_video import EncodedVideo
from numpy.random import choice

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    UniformTemporalSubsample,
)

from torchvision.transforms._transforms_video import NormalizeVideo

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    Resize,
    PILToTensor,
)
from utils.global_functions import Crop
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint

from PIL import Image

import h5py
VIDEOS =  h5py.File('../../data/videos_with_context.hdf5','r', libver='latest' , swmr=True)
AUDIOS =  h5py.File('../../data/audio.hdf5','r', libver='latest' , swmr=True)

NONE = 0
FUNC = 0
import random


def draw(img , bbox):
    black_img = np.zeros(img.shape)
    for i in range(len(bbox)):
        x1 , y1 , x2 , y2 = bbox[i][0],bbox[i][1],bbox[i][2],bbox[i][3]
        roi = img[y1:y2 , x1:x2]
        black_img[y1:y2, x1:x2] = roi
    del img

    return black_img

def body_face(test_vid : torch.Tensor , bbox):
  test_vid = test_vid.permute(1,0,2,3)
  for i , img in enumerate(test_vid):
    img = img.permute(1 , 2 , 0).numpy()
    output_image =  torch.Tensor(draw(img , bbox[i])).permute(2 , 0 , 1)
    del img
    test_vid[i , ...] = output_image
  return test_vid.permute(1 , 0 , 2 , 3)

def videoMAE_features(path , bbox  , timings , speaker , check):
    if timings == None:
        beg = 0
        end = 500
    else:
        beg = timings[0]
        end = timings[1]
        if end - beg < .1:
            beg = 0
            end = 500
    
    # func_ = [None, body_face]
    # singular_func_ = random.choices(population=func_, weights=[.25 , .75], k=1)[0]

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
   
    
    video = torch.Tensor(   VIDEOS[f"{check}_{path.split('/')[-1][:-4]}"][()]  )
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
    # print(f"path is {path}\nshape of ret is {ret.shape}\n" , flush = True)
    return  speech_array

# TODO: DATASET SPECIFIC
PROC = AutoProcessor.from_pretrained("audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim")

def collate_batch(batch , check): # batch is a pseudo pandas array of two columns
def collate_batch(batch , check): # batch is a pseudo pandas array of two columns
    """
    Here we are going to take some raw-input and pre-process them into what we need
    So we can see all the steps in the ML as they go along
    """
    text_mask = []

    video_list = []
    input_list = []
    speech_list = []
    label_list = []

    
    text_list_mask = None
    speech_list_mask = None
    vid_mask = None


    for (input , label) in batch:
        text = input[0]
        audio_path = input[1]
        vid_features = input[2]#[6:] for debug
        
        input_list.append(text['input_ids'].tolist()[0])
        text_mask.append(text['attention_mask'].tolist()[0])
        speech_list.append(speech_file_to_array_fn(audio_path , check = check))
        # TODO: DATASET SPECIFIC
        video_list.append(videoMAE_features(vid_features['vid_path'] , vid_features['bbox'] , vid_features['timings'] , vid_features['speaker'] , check))
        label_list.append(label)
    batch_size = len(label_list)
    
    text_list_mask = torch.Tensor(np.array(text_mask))
    #Just realized videoMAE does masking with the opposite way, with 1s being mask and 0's not being mask, anyways implementation still works because
    # We just need there to be around 100 points to look at. We technically are doing it the opposite way
    vid_mask = torch.randint(-2, 5, (batch_size, 1568//2)) # 8*14*14 = 1568 is just the sequence length of every video with VideoMAE, so i just hardcoded it, 
    vid_mask[vid_mask > 0] = 1
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
    speech_list_mask = torch.Tensor(np.array(PROC( numpy_speech_list, sampling_rate = 16000 , padding = True)['attention_mask']))
    del numpy_speech_list
    torch.cuda.empty_cache()
    speech_list_input_values = pad_sequence(speech_list , batch_first = True, padding_value=0)
    # speech_list_input_values = (speech_list_input_values1 + speech_list_input_values2)/2

    # Batch_size , 16 , 224 , 224  
    text = {'input_ids':torch.Tensor(np.array(input_list)).type(torch.LongTensor) , 
            'attention_mask':text_list_mask,
        }

    audio_features = {'audio_features':speech_list_input_values , 
            'attention_mask':speech_list_mask, 
        }

    visual_embeds = {'visual_embeds':torch.stack(video_list).permute(0,2,1,3,4) , 
            'attention_mask':vid_mask , 
        }
    return [text , audio_features , visual_embeds] , torch.Tensor(np.array(label_list))

class TAVForMAE(nn.Module):
    """
    Model for Bert and VideoMAE classifier
    """

    def __init__(self, args):
        super(TAVForMAE, self).__init__()
        self.output_dim = args['output_dim']
        self.dropout = args['dropout']
        self.learn_PosEmbeddings = args['learn_PosEmbeddings']
        self.num_layers = args['num_layers']

        self.test_ctr = 1
        self.train_ctr = 1
        
        # self.videomae = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        self.videomae = VideoMAEModel.from_pretrained("thegigasurgeon/mopping_224_32_frames_resampling_1_large")

        self.bert = AutoModel.from_pretrained('j-hartmann/emotion-english-distilroberta-base')
        self.bert_norm = nn.LayerNorm(768)
        
        # self.head = Head( self.videomae.config , 12 , self.learn_PosEmbeddings)
        
        self.vid_norm = nn.LayerNorm(768)
        
        self.aud_norm = nn.LayerNorm(768)

        self.dropout = nn.Dropout(self.dropout)
        self.linear1 = nn.Linear(768*3, self.output_dim)

        self.wav2vec2 = AutoModel.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
        self.wav_2_768_2 = nn.Linear(1024 , 768)
        self.wav_2_768_2.weight = torch.nn.init.xavier_normal_(self.wav_2_768_2.weight)
        
        self.vid_2_768_2 = nn.Linear(1024 , 768)
        self.vid_2_768_2.weight = torch.nn.init.xavier_normal_(self.vid_2_768_2.weight)
        
        self.masked_spec_embed = nn.Parameter(torch.FloatTensor(1024).uniform_())#specific wav2vec2 hidden size
        
        self.wav_2_768 = nn.Linear(1024 , 768)
        self.wav_2_768.weight = torch.nn.init.xavier_normal_(self.wav_2_768.weight)
        
        # self.weights = nn.Parameter(torch.ones(4) / 4)
        
        self.iter = 1
        
    def forward(self, input_ids , text_attention_mask, audio_features , audio_mask , video_embeds , visual_mask , check = "train"):
        # Get embeddings from text/audio
        embedded_bert = self.get_text(input_ids)
        embedded_audio , audio_mask = self.get_audio(audio_features, audio_mask , True if check == "train" else False)
        

        
        #Combine Features Over Sequence Length
        
        ta = self.concat_([embedded_bert , embedded_audio] , dim = 1)
    
        # Get attention mask and positions
        text_pos , text_mask = self.masking(embedded_bert , text_attention_mask)
        
        
        audio_pos , audio_mask = self.masking(embedded_audio , audio_mask.to(torch.int))
        
        ta_embed = self.concat_([text_pos , audio_pos+1] , dim = 1)
        del text_pos
        del audio_pos
        #Concatenate Masks
        ta_mask = self.concat_([text_mask , audio_mask] , dim = -1)
        del audio_mask
        del text_mask
        # Get our early fusion on embeddings
        # tav = checkpoint(self.head , video_embeds , visual_mask ,  ta , ta_mask , ta_embed)
        
        del ta 
        del ta_mask 
        del ta_embed
        
        #Transformer Time
        aud_outputs = checkpoint(self.wav2vec2 , audio_features)[0]
        del audio_features
        # aud_output = torch.rand((1 , 64 , 1024))
        aud_outputs = torch.mean(self.wav_2_768_2(aud_outputs), dim=1)

        vid_outputs = checkpoint(self.videomae , video_embeds , visual_mask)[0] # Now it has 2 dimensions 
        del video_embeds
        del visual_mask
        vid_outputs = torch.mean(self.vid_2_768_2(vid_outputs), dim=1) # Now it has 2 dimensions 

        text_outputs = checkpoint(self.bert , input_ids, text_attention_mask).pooler_output 
        # text_outputs = torch.mean(text_outputs, dim=1) # Now it has 2 dimensions 
        
        del input_ids
        del text_attention_mask
        
        text_outputs = self.bert_norm(text_outputs) # 
        aud_outputs = self.aud_norm(aud_outputs)
        vid_outputs = self.vid_norm(vid_outputs) 

        # All of these ouputs are 4 x [batch_size , 768]
        # Take weighted mean
        # tav = (0.4 * tav + 0.4 * text_outputs + 0.1 * aud_outputs + 0.1 * vid_outputs) / 4
        # tav = torch.stack([tav,text_outputs, aud_outputs , vid_outputs],dim=0)
        # tav = torch.sum(tav * self.weights.view(-1, 1, 1), dim=0)
        # outputs = torch.stack([output1, output2, output3, output4], dim=0)
        
        # Compute the weighted average along the new dimension to get the ensemble prediction
        #Concatenate Outputs
        # tav = torch.cat([tav , text_outputs, aud_outputs , vid_outputs],dim=1)
        tav = torch.cat([ text_outputs, aud_outputs , vid_outputs],dim=1)
        if self.iter == 2:
            print(f"shape of tav is {tav.shape}\n")
            self.iter += 1
        del text_outputs
        del aud_outputs
        del vid_outputs
        # Should i add another positional embedding here?
        
        #Classifier Head
        if check == "train":
            tav = self.dropout(tav)
        tav = self.linear1(tav)
        return tav 

        
    def _mask_hidden_states(self, hidden_states, attention_mask, training = False):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """

        batch_size, sequence_length, hidden_size = hidden_states.size()
        # `wav2vec2.config.apply_spec_augment` can set masking to False
        if not getattr(self.wav2vec2.config, "apply_spec_augment", True) or sequence_length < self.wav2vec2.config.mask_time_length:
            return hidden_states

        # generate indices & apply SpecAugment along time axis


        if self.wav2vec2.config.mask_time_prob > 0 and training:
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.wav2vec2.config.mask_time_prob,
                mask_length=self.wav2vec2.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.wav2vec2.config.mask_time_min_masks,#ERROR HERE
                min_masks=self.wav2vec2.config.mask_time_min_masks,#ERROR HERE
            )
            mask_time_indices = torch.tensor(mask_time_indices, device=hidden_states.device, dtype=torch.bool)
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)

        if self.wav2vec2.config.mask_feature_prob > 0 and training:
            # generate indices & apply SpecAugment along feature axis
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.wav2vec2.config.mask_feature_prob,
                mask_length=self.wav2vec2.config.mask_feature_length,
                min_masks=self.wav2vec2.config.mask_feature_min_masks,
            )
            mask_feature_indices = torch.tensor(mask_feature_indices, device=hidden_states.device, dtype=torch.bool)
            mask_feature_indices = mask_feature_indices[:, None].expand(-1, sequence_length, -1)
            hidden_states[mask_feature_indices] = 0

        return hidden_states 
        
    def _get_feat_extract_output_lengths(self , input_lengths, add_adapter = None):
        """
        Computes the output length of the convolutional layers
        """

        # add_adapter = wav2vec2.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        for kernel_size, stride in zip(self.wav2vec2.config.conv_kernel, self.wav2vec2.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        
        return input_lengths

    def _get_feature_vector_attention_mask(self, feature_vector_length: int, attention_mask: torch.LongTensor, add_adapter=None):
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        output_lengths = output_lengths.to(torch.long)

        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask
    
    def get_text(self , input_ids ):
        if input_ids is not None:
            embedded_bert = self.bert.embeddings(input_ids = input_ids)
        return embedded_bert
    
    def get_audio(self , audio_features , audio_mask , train = False):
        extract_features = self.wav2vec2.feature_extractor(audio_features)
        del audio_features
        if audio_mask is not None:
            audio_mask = self._get_feature_vector_attention_mask(extract_features.shape[2], audio_mask, add_adapter=False)
        embedded_audio, extract_features = self.wav2vec2.feature_projection(extract_features.transpose(1, 2))
        
        embedded_audio, extract_features = self.wav2vec2.feature_projection(extract_features.transpose(1, 2))
        
        # Multiple CheckPointing like this produces an error if we are also trying to find unused params in Accelerate/DDP
        embedded_audio = self._mask_hidden_states(embedded_audio, audio_mask, train)
        embedded_audio = embedded_audio + self.wav2vec2.encoder.pos_conv_embed(embedded_audio)
        embedded_audio = self.wav2vec2.encoder.layer_norm(embedded_audio)
        embedded_audio = self.wav2vec2.encoder.dropout(embedded_audio)
        embedded_audio = self.wav_2_768(embedded_audio)
        return embedded_audio , audio_mask
    
    def get_video(self , video_embeds , visual_mask):
        embedded_video = self.videomae.embeddings(video_embeds , ~visual_mask) # this is a different visual mask then the one we input into the encoder 27 lines from here on if statement 
        del video_embeds
        return embedded_video
    
    
    def concat_(self , ls_o_tensor, dim):
        tensors = [tensor for tensor in ls_o_tensor if tensor is not None]
        
        if len(tensors) == 0:
            return None
        elif len(tensors) == 1:
            return tensors[0]
        else:
            return torch.cat(tensors, dim=dim)
        
    
    def masking(self , m , mask):
        batch_size , seq_len , _ = m.shape 
        
        pos = torch.zeros((batch_size , seq_len ) , device=torch.device('cuda') , dtype=torch.int)            
        if mask is not None:
            mask = (1.0 - mask[:, None, None, :]) * torch.finfo(torch.float16).min
        return pos , mask