from transformers import logging
logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore") 
import torch
from torch import nn
from torch.nn import functional as F
import math
from transformers import  VideoMAEModel  , AutoModel , AutoProcessor
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices
from utils.TAVFormer import TransformerEncoder

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
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)
from utils.global_functions import Crop
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
import dill as pickle

from torch.utils.checkpoint import checkpoint



def videoMAE_features(path  , clip_duration , speaker , check):
    if clip_duration == None:
        beg = 0
        end = 500
    else:
        beg = clip_duration[0]
        end = clip_duration[1]
        if end - beg < .1:
            beg = 0
            end = 500

    # feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base")
    # mean = feature_extractor.image_mean
    # std = feature_extractor.image_std
    # resize_to = feature_extractor.size['shortest_edge']
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225] 
    resize_to = {'shortest_edge': 224}
    resize_to = resize_to['shortest_edge']
    num_frames_to_sample = 16 # SET by MODEL CANT CHANGE

    # i hardcoded the feature extractor stuff just because it saves a couple milliseconds every run, so hopefully makes 
    # it a tad faster overall
    if check == "train":
        transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(16),
                            Lambda(lambda x: x / 255.0),
                            NormalizeVideo(mean, std),
                            RandomHorizontalFlip(p=0) if speaker == None else Crop((120,2,245,355)) if speaker else Crop((120,362,245,355)), # if not IEMOCAP then do nothing, else 
                            # Hone in on either the left_speaker or right_speaker in the video
                            RandomShortSideScale(min_size=256, max_size=320),
                            RandomCrop((224, 224)), # Need to be at 224,224 as our height and width for VideoMAE
                            RandomHorizontalFlip(p=0.5), # 
                        ]
                    ),
                ),
            ]
        )
    else:
        transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(num_frames_to_sample),
                            Lambda(lambda x: x / 255.0),
                            Normalize(mean, std),
                            RandomHorizontalFlip(p=0) if speaker == None else Crop((120,2,245,355)) if speaker else Crop((120,362,245,355)),
                            Resize((resize_to, resize_to)),
                        ]
                    ),
                ),
            ]
        )
    video = EncodedVideo.from_path(path)
    
    video_data = video.get_clip(start_sec=beg, end_sec=end)
    video_data = transform(video_data)
    
    return video_data['video']
    


def speech_file_to_array_fn(path , target_sampling_rate = 16000):
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze()
    return torch.mean(speech, dim=0) if len(speech.shape) > 1 else speech # over the channel dimension


PROC = AutoProcessor.from_pretrained("audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim")

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
        input_list.append(text['input_ids'].tolist()[0])
        text_mask.append(text['attention_mask'].tolist()[0])
        audio_path = input[1]
        speech_list.append(speech_file_to_array_fn(audio_path))
        
        vid_features = input[2]#[6:] for debug
        video_list.append(videoMAE_features(vid_features['vid_path'] , vid_features['timings'] , vid_features['speaker'] , check))
        label_list.append(label)
    batch_size = len(label_list)
    
    text_list_mask = torch.Tensor(np.array(text_mask))
    vid_mask = torch.randint(0, 2, (batch_size, 1568)).bool() # 1568 is just the sequence length of every video with VideoMAE, so i just hardcoded it, 
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
    speech_list_mask = torch.Tensor(np.array(PROC( numpy_speech_list, sampling_rate = 16000 , padding = True)['attention_mask']))
        
    speech_list_input_values = pad_sequence(speech_list , batch_first = True, padding_value=0)
    # speech_list_input_values = (speech_list_input_values1 + speech_list_input_values2)/2
    
    
    
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


class PreFormer(nn.Module):
    """
    Model for Bert and VideoMAE classifier
    """

    def __init__(self):
        super(PreFormer, self).__init__()
        
        self.wav2vec2 = AutoModel.from_pretrained("audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim")
        self.wav2vec2.feature_projection.projection.out_features = 768
        self.wav2vec2.encoder.pos_conv_embed.conv.in_channels = 768
        self.wav2vec2.encoder.pos_conv_embed.conv.out_channels = 768
        
        self.masked_spec_embed = nn.Parameter(torch.FloatTensor(1024).uniform_())#specific wav2vec2 hidden size
        # self.wav2vec2.encoder.pos_conv_embed.padding = nn.Identity()
        # self.wav2vec2.encoder.layer_norm.normalized_shape = (768,)
        self.videomae = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")

        self.wav_2_768 = nn.Linear(1024 , 768)
        self.wav_2_768.weight = torch.nn.init.xavier_normal_(self.wav_2_768.weight)

    def _mask_hidden_states(self, hidden_states, attention_mask, training = False):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """

        # `wav2vec2.config.apply_spec_augment` can set masking to False
        if not getattr(self.wav2vec2.config, "apply_spec_augment", True):
            return hidden_states

        # generate indices & apply SpecAugment along time axis
        batch_size, sequence_length, hidden_size = hidden_states.size()


        if self.wav2vec2.config.mask_time_prob > 0 and training:
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.wav2vec2.config.mask_time_prob,
                mask_length=self.wav2vec2.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.wav2vec2.config.mask_time_min_masks,
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

    def forward(self, audio_features , video_embeds , text_mask = None , audio_mask = None , visual_mask = None, device = "cpu" , train = False):
        
        self.wav2vec2 = self.wav2vec2.to(device)
        self.wav_2_768 = self.wav_2_768.to(device)
        #Audio Embeds
        # torch.cuda.empty_cache()
        extract_features = checkpoint(self.wav2vec2.feature_extractor,audio_features.to(device))
        if audio_mask is not None:
            audio_mask = self._get_feature_vector_attention_mask(extract_features.shape[2], audio_mask, add_adapter=False)
        embedded_audio, extract_features = checkpoint(self.wav2vec2.feature_projection,extract_features.transpose(1, 2))
        del extract_features
        # Multiple CheckPointing like this produces an error if we are also trying to find unused params in Accelerate/DDP
        embedded_audio = self._mask_hidden_states(embedded_audio.to("cpu"), audio_mask, train).to(device)
        embedded_audio = embedded_audio + checkpoint(self.wav2vec2.encoder.pos_conv_embed,embedded_audio)
        embedded_audio = checkpoint(self.wav2vec2.encoder.layer_norm,embedded_audio)
        embedded_audio = checkpoint(self.wav2vec2.encoder.dropout,embedded_audio)
        embedded_audio = self.wav_2_768(embedded_audio).to("cpu")
        # torch.cuda.empty_cache()
        #Video Embeds
        embedded_video = self.videomae.embeddings(video_embeds , visual_mask)
        #Combine Features Over Sequence Length
        tav = torch.concat((embedded_audio , embedded_video) , dim = 1)

        #Create Static Embedding And Masks

        #1's , Audio
        batch_size , seq_len , _ = embedded_audio.shape
        audio_pos = torch.zeros((batch_size , seq_len ))             
        if audio_mask is not None:
            audio_mask = 1.0 - audio_mask[:, None, None, :] * torch.finfo(audio_features.dtype).min
        del embedded_audio

        #2's , Video
        batch_size , seq_len , _ = embedded_video.shape
        visual_pos = torch.ones((batch_size , seq_len ))         
        if visual_mask is not None:
            visual_mask = torch.zeros((batch_size , 1 , 1 , seq_len)).type(torch.float)
        del embedded_video

        tav_embed = torch.concat((audio_pos , visual_pos) , dim = 1).type(torch.LongTensor)
        #Concatenate Masks
        attention_mask = None
        if text_mask is not None and audio_mask is not None and visual_mask is not None:
            attention_mask = torch.concat((text_mask , audio_mask , visual_mask),  dim = -1)

        return tav, tav_embed , attention_mask
        

class TAVFormer(nn.Module):
    """
    Modelfor Bert and VideoMAE classifier
    """

    def __init__(self, args):
        super(TAVFormer, self).__init__()
        self.output_dim = args['output_dim']
        self.dropout = args['dropout']
        self.early_div = args['early_div']
        self.num_layers = args['num_layers']

        self.embedding = nn.Embedding(3 , 768)
        self.embedding.weight.requires_grad = False # Now the embedding is static

        self.transformer = TransformerEncoder(768 ,  num_layers = self.num_layers , dropout = self.dropout  , early_div =  self.early_div).apply(self.randomize_model)

        self.dropout = nn.Dropout(self.dropout)
        self.fc_norm = nn.LayerNorm(768)
        self.linear = nn.Linear(768, self.output_dim)


    def randomize_model(self,model):
        for module_ in model.named_modules(): 
            if isinstance(module_[1],(torch.nn.Linear, torch.nn.Embedding)):
                module_[1].weight = torch.nn.init.xavier_uniform_(module_[1].weight)
            elif isinstance(module_[1], torch.nn.LayerNorm):
                module_[1].bias.data.zero_()
                module_[1].weight = torch.nn.init.constant_(nn.Parameter(torch.empty(768)) , 1)
                # module_[1].weight.data.fill_(1.0)
            if isinstance(module_[1], torch.nn.Linear) and module_[1].bias is not None:
                module_[1].bias.data.zero_()
        return model


    def forward(self, hidden_states , pos_embed , attention_mask = None):
        #Transformer
        tav = checkpoint(self.transformer , hidden_states + self.embedding(pos_embed) , attention_mask)
        del pos_embed
        del attention_mask
        #Classifier Head
        tav = self.fc_norm(torch.mean(tav, dim=1))
        tav = self.dropout(tav)
        tav = self.linear(tav)
        torch.cuda.empty_cache()
        return tav # returns [batch_size,output_dim]

class TAVForMAE(nn.Module):
    """
    Modelfor Bert and VideoMAE classifier
    """

    def __init__(self, args):
        super(TAVForMAE, self).__init__()
        self.output_dim = args['output_dim']
        self.dropout = args['dropout']
        self.learn_PosEmbeddings = args['learn_PosEmbeddings']

        self.embedding = nn.Embedding(3 , 768)
        self.embedding.weight.requires_grad = self.learn_PosEmbeddings # Now the embedding is static

        self.random_mae = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").apply(self.randomize_model)

        self.bert = AutoModel.from_pretrained('j-hartmann/emotion-english-distilroberta-base')

    
        self.dropout = nn.Dropout(self.dropout)
        self.fc_norm = nn.LayerNorm(768)
        self.linear = nn.Linear(768*2, self.output_dim)


    def randomize_model(self,model):
        for module_ in model.named_modules(): 
            if isinstance(module_[1],(torch.nn.Linear, torch.nn.Embedding)):
                module_[1].weight = torch.nn.init.xavier_uniform_(module_[1].weight)
            elif isinstance(module_[1], torch.nn.LayerNorm):
                module_[1].bias.data.zero_()
                module_[1].weight = torch.nn.init.constant_(nn.Parameter(torch.empty(768)) , 1)
                # module_[1].weight.data.fill_(1.0)
            if isinstance(module_[1], torch.nn.Linear) and module_[1].bias is not None:
                module_[1].bias.data.zero_()
        return model
    
    def forward(self, input_ids , hidden_states, pos_embed , attention_mask = None , batch_size = 2):
        av = hidden_states + self.embedding(pos_embed)
        del hidden_states
        del pos_embed
        #Transformer Time
        av = self.random_mae.encoder(av)
        av = self.fc_norm(torch.mean(av[0], dim=1)) # tav[0] is the hidden state output
        _, t = self.bert(input_ids= input_ids, attention_mask=attention_mask,return_dict=False)
        tav = torch.cat([t,av],dim=1)
        #Classifier Head
        tav = self.dropout(tav)
        tav = self.linear(tav)
        return tav # returns [batch_size,output_dim]
class TAVForW2V2(nn.Module):
    """
    Modelfor Bert and VideoMAE classifier
    """

    def __init__(self, args):
        super(TAVForW2V2, self).__init__()
        self.output_dim = args['output_dim']
        self.dropout = args['dropout']
        self.learn_PosEmbeddings = args['learn_PosEmbeddings']

        self.embedding = nn.Embedding(3 , 768)
        self.embedding.weight.requires_grad = self.learn_PosEmbeddings # Now the embedding is static

        self.random_wav2vec2 = AutoModel.from_pretrained("superb/wav2vec2-base-superb-er").apply(self.randomize_model)
        # Just using their architecture since the other wav2vec2 one is about 1024 hidden size and this is 768
        # Still randomizing the weights so dont worry about that 
        self.dropout = nn.Dropout(self.dropout)
        self.fc_norm = nn.LayerNorm(768)
        self.linear = nn.Linear(768, self.output_dim)


    def randomize_model(self,model):
        for module_ in model.named_modules(): 
            if isinstance(module_[1],(torch.nn.Linear, torch.nn.Embedding)):
                module_[1].weight = torch.nn.init.xavier_uniform_(module_[1].weight)
            elif isinstance(module_[1], torch.nn.LayerNorm):
                module_[1].bias.data.zero_()
                module_[1].weight = torch.nn.init.constant_(nn.Parameter(torch.empty(768)) , 1)
                # module_[1].weight.data.fill_(1.0)
            if isinstance(module_[1], torch.nn.Linear) and module_[1].bias is not None:
                module_[1].bias.data.zero_()
        return model
    
    def forward(self, hidden_states: torch.Tensor, pos_embed: torch.Tensor , attention_mask = None , batch_size = 2):
        tav = hidden_states + self.embedding(pos_embed)
        del hidden_states
        del pos_embed
        #Transformer Time
        if batch_size >= 8:
            # tav = checkpoint(self.random_mae.encoder,tav)
            for layer in self.random_wav2vec2.encoder.layers:
                tav = checkpoint(layer,tav)
                tav = tav[0]
        else:
            for layer in self.random_wav2vec2.encoder.layers:
                tav = layer(tav)
                tav = tav[0]
        #Classifier Head
        tav = self.fc_norm(torch.mean(tav, dim=1)) # tav[0] is the hidden state output
        tav = self.dropout(tav)
        tav = self.linear(tav)
        return tav # returns [batch_size,output_dim]


