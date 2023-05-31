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
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices
from utils.TAVFormer import TransformerEncoder , VideoMAEEncoder

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
    RandomVerticalFlip,
    Resize,
    PILToTensor,
    ToPILImage,
    Normalize,
    RandomErasing,
    # RandomShortSideScale,
)
from utils.global_functions import Crop
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
import dill as pickle
from torch import nn

from torch.utils.checkpoint import checkpoint

from PIL import Image

def videoMAE_features(path  , clip_duration , speaker , check):
    # TODO: DATASET SPECIFIC
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
                            UniformTemporalSubsample(num_frames_to_sample),
                            Lambda(lambda x: x / 255.0),
                            NormalizeVideo(mean, std),                            
                            # TODO: DATASET SPECIFIC                            
                            RandomHorizontalFlip(p=0) if speaker == None else Crop((120,2,245,355)) if speaker else Crop((120,362,245,355)), # if not IEMOCAP then do nothing, else 
                            # Hone in on either the left_speaker or right_speaker in the video
                            RandomShortSideScale(min_size=256, max_size=320),
                            #Removed the random crop to 224 and am just using a resize now
                            Resize((224, 224)), # Need to be at 224,224 as our height and width for VideoMAE
                            RandomHorizontalFlip(p=0.5), # 
                            RandomVerticalFlip(p=0.5), # 
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
                            NormalizeVideo(mean, std),                            
                            # TODO: DATASET SPECIFIC                            
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

def getKatna(path  , clip_duration , speaker , check):
    # TODO: DATASET SPECIFIC
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225] 
    resize_to = {'shortest_edge': 224}
    resize_to = resize_to['shortest_edge']

    if check == "train":
                transform = Compose(
                          [
                                # PILToTensor(),
                                Lambda(lambda x: x / 255.0),
                                Normalize(mean, std),                            
                                # TODO: DATASET SPECIFIC    
                                RandomHorizontalFlip(p=0) if speaker == None else Crop((120,2,245,355)) if speaker else Crop((120,362,245,355)), # if not IEMOCAP then do nothing, else,
                                Resize((224, 224)),
                                RandomHorizontalFlip(p=0.5),
                                RandomVerticalFlip(p=0.5),
                        ]
        )
    else:
        transform = Compose(
                          [
                                # PILToTensor(),
                                Lambda(lambda x: x / 255.0),
                                Normalize(mean, std),                            
                                # TODO: DATASET SPECIFIC    
                                RandomHorizontalFlip(p=0) if speaker == None else Crop((120,2,245,355)) if speaker else Crop((120,362,245,355)), # if not IEMOCAP then do nothing, else,
                                Resize((224, 224)),
                        ]
        )
    lst = []
    for file in glob(path):
        lst.append( PILToTensor()(Image.open(file)))
    img_batch = torch.stack(lst)
    transformed_img_batch = transform(img_batch)
    padded_transformed_img_batch = F.pad(transformed_img_batch, (0,0,0,0,0,0,0,16-transformed_img_batch.shape[0]), mode='constant', value=0)
    return padded_transformed_img_batch.unsqueeze(0)

    


def speech_file_to_array_fn(path , target_sampling_rate = 16000):
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze()
    return torch.mean(speech, dim=0) if len(speech.shape) > 1 else speech # over the channel dimension

# TODO: DATASET SPECIFIC
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

        # TODO: DATASET SPECIFIC
        video_list.append(videoMAE_features(vid_features['vid_path'] , vid_features['timings'] , vid_features['speaker'] , check))
        label_list.append(label)
    batch_size = len(label_list)
    
    text_list_mask = torch.Tensor(np.array(text_mask))
    vid_mask = torch.randint(-13, 2, (batch_size, 1568)) # 8*14*14 = 1568 is just the sequence length of every video with VideoMAE, so i just hardcoded it, 
    vid_mask[vid_mask < 0] = 0
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


class PreFormer(nn.Module):
    """
    Model for Bert and VideoMAE classifier
    """

    def __init__(self):
        super(PreFormer, self).__init__()
        
        self.bert = AutoModel.from_pretrained('j-hartmann/emotion-english-distilroberta-base')

        self.wav2vec2 = AutoModel.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
        
        self.masked_spec_embed = nn.Parameter(torch.FloatTensor(1024).uniform_())#specific wav2vec2 hidden size
        
        self.videomae = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        self.wav_2_768 = nn.Linear(1024 , 768)
        self.wav_2_768.weight = torch.nn.init.xavier_normal_(self.wav_2_768.weight)

        self.check_shapes = 1

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

    def forward(self, input_ids = None , audio_features = None , video_embeds = None , text_mask = None , audio_mask = None , visual_mask = None, device = "cpu" , train = False):
        self.wav2vec2 = self.wav2vec2.to(device)
        self.wav_2_768 = self.wav_2_768.to(device)
        #Text Embeds
        if input_ids is not None:
            embedded_bert = self.bert.embeddings(input_ids = input_ids)
        #Audio Embeds
        # torch.cuda.empty_cache()
        extract_features = self.wav2vec2.feature_extractor(audio_features.to(device))
        del audio_features
        if audio_mask is not None:
            audio_mask = self._get_feature_vector_attention_mask(extract_features.shape[2], audio_mask, add_adapter=False)
        embedded_audio, extract_features = self.wav2vec2.feature_projection(extract_features.transpose(1, 2))
        
        # Multiple CheckPointing like this produces an error if we are also trying to find unused params in Accelerate/DDP
        embedded_audio = self._mask_hidden_states(embedded_audio.to("cpu"), audio_mask, train).to(device)
        embedded_audio = embedded_audio + self.wav2vec2.encoder.pos_conv_embed(embedded_audio)
        embedded_audio = self.wav2vec2.encoder.layer_norm(embedded_audio)
        embedded_audio = self.wav2vec2.encoder.dropout(embedded_audio)
        embedded_audio = self.wav_2_768(embedded_audio).to("cpu")
        # torch.cuda.empty_cache()

        #Video Embeds

        embedded_video = self.videomae.embeddings(video_embeds , ~visual_mask) # this is a different visual mask then the one we input into the encoder 27 lines from here on if statement 
        del video_embeds
        #Combine Features Over Sequence Length
        if input_ids is not None:
            tav = torch.concat((embedded_bert , embedded_audio , embedded_video) , dim = 1)
        else:
            tav = torch.concat((embedded_audio , embedded_video) , dim = 1)


        #Create Static Embedding And Masks
        #0's , Text
        if input_ids is not None:
            batch_size , seq_len , _ = embedded_bert.shape 
            text_pos = torch.zeros((batch_size , seq_len ))            
            if text_mask is not None:
                text_mask = (1.0 - text_mask[:, None, None, :]) * torch.finfo(torch.float16).min
            

        #1's , Audio
        batch_size , seq_len , _ = embedded_audio.shape
        audio_pos = torch.ones((batch_size , seq_len ))             
        if audio_mask is not None:
            audio_mask = 1.0 - audio_mask[:, None, None, :] * torch.finfo(torch.float16).min # float32 was causing underflow 
        

        #2's , Video
        batch_size , seq_len , _ = embedded_video.shape
        visual_pos = torch.ones((batch_size , seq_len )) + 1         
        if visual_mask is not None:
            visual_mask = torch.zeros((batch_size , 1 , 1 , seq_len)).type(torch.float) # Thus everything gets attention as  torch.finfo(audio_features.dtype).min is the smallest we can get 
        


        if input_ids is not None:
            tav_embed = torch.concat((text_pos , audio_pos , visual_pos) , dim = 1).type(torch.LongTensor)
        else:
            tav_embed = torch.concat((audio_pos , visual_pos) , dim = 1).type(torch.LongTensor)

        #Concatenate Masks
        attention_mask = None
        if text_mask is not None and audio_mask is not None and visual_mask is not None:
            attention_mask = torch.concat((text_mask , audio_mask , visual_mask),  dim = -1)
        else:
            attention_mask = torch.concat((audio_mask , visual_mask),  dim = -1)

        if self.check_shapes == 1:
            self.check_shapes += 1
            print(f"Text shape is {embedded_bert.shape}\nAudio shape is {embedded_audio.shape}\nVideo shape is {embedded_video.shape}\n" , flush = True)

        return tav, tav_embed , attention_mask 
        

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

        self.embedding = nn.Embedding(3 , 768)
        self.embedding.weight.requires_grad = self.learn_PosEmbeddings # Now the embedding is static

        self.bert = AutoModel.from_pretrained('j-hartmann/emotion-english-distilroberta-base')
        self.bert_norm = nn.LayerNorm(768)

        self.random_mae_config = AutoConfig.from_pretrained("MCG-NJU/videomae-base")
        self.random_mae_encoder = VideoMAEEncoder(self.random_mae_config, 12).apply(self.randomize_model)
        self.rand_norm = nn.LayerNorm(768)

        self.vid_norm = nn.LayerNorm(768)
        
        self.aud_norm = nn.LayerNorm(768)


        self.dropout = nn.Dropout(self.dropout)
        self.linear1 = nn.Linear(768*4, self.output_dim)
        # self.gelu = nn.GELU()
        # self.linear2 = nn.Linear(768*2, self.output_dim)

        self.wav2vec2 = AutoModel.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
        self.videomae = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        self.wav_2_768_2 = nn.Linear(1024 , 768)
        self.wav_2_768_2.weight = torch.nn.init.xavier_normal_(self.wav_2_768_2.weight)


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
    
    def forward(self, input_ids , text_attention_mask, audio_features , video_embeds , visual_mask , hidden_states, pos_embed , attention_mask , batch_size = 2 , check = "train"):
        av = hidden_states + self.embedding(pos_embed)
        #Transformer Time
        aud_outputs = self.wav2vec2(audio_features)[0]
        # aud_output = torch.rand((1 , 64 , 1024))
        aud_outputs = torch.mean(self.wav_2_768_2(aud_outputs), dim=1)

        vid_outputs = self.videomae(video_embeds , visual_mask)[0] # Now it has 2 dimensions 
        vid_outputs = torch.mean(vid_outputs, dim=1) # Now it has 2 dimensions 



        _, t = self.bert(input_ids= input_ids, attention_mask=text_attention_mask,return_dict=False)
        t = self.bert_norm(t) # 
        av = self.random_mae_encoder(av , attention_mask)
        av = self.rand_norm(torch.mean(av, dim=1)) # av[0] is the hidden state output
        aud_outputs = self.aud_norm(aud_outputs)
        vid_outputs = self.vid_norm(vid_outputs) 

        # All of these ouputs are 4 x [batch_size , 768]

        #Concatenate Outputs
        tav = torch.cat([av,t, aud_outputs , vid_outputs],dim=1)
        #Classifier Head
        if check == "train":
            tav = self.dropout(tav)
        tav = self.linear1(tav)
        # tav = self.gelu(tav)
        # if check == "train":
        #     tav = self.dropout(tav)
        # tav = self.linear2(tav)
        return tav # returns [batch_size,output_dim]



