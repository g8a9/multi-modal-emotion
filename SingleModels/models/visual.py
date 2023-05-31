import torch
from torch import nn
import numpy as np
import torchvision.transforms as T
from PIL import Image
from decord import VideoReader
from decord import cpu
from pytorchvideo.data.encoded_video import EncodedVideo
from transformers import  VideoMAEModel 

from pytorchvideo.transforms import (
    ApplyTransformToKey,
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
)
from utils.global_functions import Crop
from numpy.random import choice


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




def collate_batch(batch , check): # batch is a pseudo pandas array of two columns
    """
    Here we are going to take some raw-input and pre-process them into what we need
    So we can see all the steps in the ML as they go along
    """
    video_list = []
    label_list = []

    
    for (input_ , label) in batch:
        # print(input_path,"\n", flush = True)
        # val = resnet3d_features(input_path)
        # print(f"val  is {val}", flush = True)
        video_list.append(videoMAE_features(input_['vid_path'] , input_['timings'] , input_['speaker'] , check))
        label_list.append(label)
        
    batch_size = len(label_list)
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
    

    return torch.stack(video_list).permute(0,2,1,3,4) , vid_mask  , torch.Tensor(np.array(label_list))




class ResNet50Classification(nn.Module):
    """A simple ConvNet for binary classification."""
    def __init__(self , args):
        super(ResNet50Classification, self).__init__()

        self.input_dim = args["input_dim"]
        self.output_dim = args["output_dim"]
        # self.output_dim = 7

        self.hidden_dim = args['hidden_layers']

        self.dropout = nn.Dropout(.5)

        self.linear = nn.Linear(768, 300)

        self.linear1 = nn.Linear(300, self.output_dim)

        self.sigmoid = nn.Sigmoid()

        self.resnet = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True).to("cuda")
        self.resnet.blocks[5].proj = torch.nn.Linear(in_features=2048,out_features=768).to("cuda")



    def forward(self,x):
        # self.resnet.eval()

        # print(f"x = {x}\n x.shape = {x.shape}", flush=True)
        x = self.resnet(x)
        # print(f"AFTER RESNET MODEL x = {x}\n x.shape = {x.shape}", flush=True)
        x = self.dropout(x)
        x = self.linear(x)
        # print(f"AFTER LINEAR x = {x}\n x.shape = {x.shape}", flush=True)

        x = self.sigmoid(x)

        x = self.dropout(x)

        x = self.linear1(x)
        # print(f"OUTPUT x = {x}\n x.shape = {x.shape}", flush=True)

        return x

class VisualClassification(nn.Module):
    """A simple ConvNet for binary classification."""
    def __init__(self , args):
        super(VisualClassification, self).__init__()

        self.dropout = args['dropout']
        self.output_dim = args['output_dim']

        self.videomae = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

        self.dropout = nn.Dropout(self.dropout)

        self.linear = nn.Linear(768, self.output_dim)

    def forward(self, video_embeds, visual_mask , check):


        # print(f"input_id = {input_id}\n input_id.shape = {input_id.shape}")

        x = self.videomae(video_embeds , visual_mask)[0] # Now it has 2 dimensions 
        # print(f"pooled_output = {pooled_output}\n pooled_output.shape = {pooled_output.shape}")
        x = torch.mean(x, dim=1)
        if check == "train":
            x = self.dropout(x)
        # print(f"dropout_output = {dropout_output}\n dropout_output.shape = {dropout_output.shape}")

        x = self.linear(x)  
        # print(f"linear_output = {linear_output}\n linear_output.shape = {linear_output.shape}")

        return x
