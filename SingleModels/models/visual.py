
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
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
import h5py
from transformers import  VideoMAEModel  , AutoModel 
from utils.global_functions import Crop

from numpy.random import choice

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




def collate_batch(batch , check): # batch is a pseudo pandas array of two columns
    """
    Here we are going to take some raw-input and pre-process them into what we need
    So we can see all the steps in the ML as they go along
    """
    video_list = []
    label_list = []

    
    for (input , label) in batch:
        video_list.append(videoMAE_features(input['vid_path'] , input['timings'] , input['speaker'] , check))
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
        
    visual_embeds = {'visual_embeds':torch.stack(video_list).permute(0,2,1,3,4) , 
            'attention_mask':vid_mask , 
        }
    return visual_embeds , torch.Tensor(np.array(label_list))



class VisualClassification(nn.Module):
    """A simple ConvNet for binary classification."""
    def __init__(self , args):
        super(VisualClassification, self).__init__()

        self.output_dim = args['output_dim']
        self.dropout = args['dropout']
        self.learn_PosEmbeddings = args['learn_PosEmbeddings']
        self.num_layers = args['num_layers']

        self.videomae = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        self.vid_norm = nn.LayerNorm(768)
        self.dropout = nn.Dropout(self.dropout)
        self.linear1 = nn.Linear(768, self.output_dim)

    def forward(self, video_embeds , visual_mask , check = "train"):
        vid_outputs = self.videomae(video_embeds , visual_mask)[0] # Now it has 2 dimensions 
        
        del video_embeds
        del visual_mask
        
        vid_outputs = torch.mean(vid_outputs, dim=1) # Now it has 2 dimensions 
        vid_outputs = self.vid_norm(vid_outputs) 
        if check == "train":
            vid_outputs = self.dropout(vid_outputs)
        vid_outputs = self.linear1(vid_outputs)

        return vid_outputs # returns [batch_size,output_dim]
