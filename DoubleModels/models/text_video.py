import torch
from torch import nn
from transformers import  VideoMAEModel  , AutoModel
import numpy as np
from pytorchvideo.data.encoded_video import EncodedVideo

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
    


def collate_batch(batch , check): # batch is a pseudo pandas array of two columns
    """
    Here we are going to take some raw-input and pre-process them into what we need
    So we can see all the steps in the ML as they go along
    """
    video_list = []
    input_list = []
    att_list = []
    # tok_list = []
    label_list = []
   
    for (input , label) in batch:
        text = input[0]
        input_list.append(text['input_ids'].tolist()[0])
        att_list.append(text['attention_mask'].tolist()[0])
        # tok_list.append(text['token_type_ids'].tolist()[0])
        
        vid_features = input[1]#[6:] for debug
        video_list.append(videoMAE_features(vid_features['vid_path'] , vid_features['timings'] , vid_features['speaker'] , check))
        label_list.append(label)
    
    text = {'input_ids':torch.Tensor(np.array(input_list)).type(torch.LongTensor) , 
            'attention_mask':torch.Tensor(np.array(att_list)).type(torch.LongTensor) , 
            # 'token_type_ids':torch.Tensor(np.array(tok_list)).type(torch.LongTensor)
        }

    return [text , torch.stack(video_list).permute(0,2,1,3,4)] , torch.Tensor(np.array(label_list))


class CustomRobLayer(nn.Module):
    def __init__(self, original_layer, shared_layer):
        super().__init__()
        self.original_layer = original_layer
        self.shared_layer = shared_layer

    def forward(self, **x):
        original_out = self.original_layer(**x)
        return self.shared_layer(original_out)


class CustomOriginalLayer(nn.Module):
    def __init__(self, original_layer, shared_layer):
        super().__init__()
        self.original_layer = original_layer
        self.shared_layer = shared_layer

    def forward(self, *x):
        original_out = self.original_layer(*x)
        return self.shared_layer(original_out)

class BertVideoMAE_MTL1Shared_Classifier(nn.Module):
    """
    Model for Bert and VideoMAE3d classifier
    """

    def __init__(self, args ,dropout=0.5):
        super(BertVideoMAE_MTL1Shared_Classifier, self).__init__()

        self.output_dim = args['output_dim']

        self.shared_layer = nn.Linear(768,768)
        torch.nn.init.xavier_normal_(self.shared_layer.weight)
        
        self.bert = AutoModel.from_pretrained('j-hartmann/emotion-english-distilroberta-base')
        self.bert.embeddings = CustomRobLayer(self.bert.embeddings , self.shared_layer)

        self.videomae = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
        self.videomae.embeddings = CustomOriginalLayer(self.videomae.embeddings,self.shared_layer)

        self.dropout = nn.Dropout(dropout)
        self.fc_norm = nn.LayerNorm(768)
        self.linear_test = nn.Linear(768, self.output_dim)

    def forward(self, input_ids , video_embeds , task_id ,attention_mask = None , token_type_ids = None , check = "train"):
        output = 0
        if task_id == 0:
            output = self.bert(input_ids = input_ids , attention_mask = attention_mask)['pooler_output']
        else:
            vid_features = self.videomae(video_embeds)
            output = self.fc_norm(torch.mean(vid_features[0], dim=1))
        # Batch_size x 768
        dropout_output = self.dropout(output)
        linear_output =  self.linear_test(dropout_output)
        
        return linear_output # returns [batch_size,output_dim]

class BertVideoMAE_LateFusion_Classifier(nn.Module):
    """
    Model for Bert and VideoMAE classifier
    """

    def __init__(self, args ,dropout=0.5):
        super(BertVideoMAE_LateFusion_Classifier, self).__init__()

        self.output_dim = args['output_dim']
        self.bert = AutoModel.from_pretrained('j-hartmann/emotion-english-distilroberta-base')
        self.videomae = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
        
        self.dropout = nn.Dropout(dropout)
        self.fc_norm = nn.LayerNorm(768)
        self.linear = nn.Linear(768*2, self.output_dim)


    def forward(self, input_ids , video_embeds, task_id = None , attention_mask = None , token_type_ids = None , check = "train"):
        # print("here \n \n inside late fusion" , flush=True)
        _, pooled_output = self.bert(input_ids= input_ids, attention_mask=attention_mask,return_dict=False)

        vid_features = self.videomae(video_embeds)
        video_embeds = self.fc_norm(torch.mean(vid_features[0], dim=1))

        comb_features = torch.cat([pooled_output,video_embeds],dim=1)
        # Batch_size x 768*2

        dropout_output = self.dropout(comb_features)
        linear_output =  self.linear(dropout_output)
        
        return linear_output # returns [batch_size,output_dim]
