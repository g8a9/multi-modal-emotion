import torch
from torch import nn
from transformers import BertModel , VideoMAEModel , VideoMAEFeatureExtractor , AutoModel
import numpy as np
from pytorchvideo.data.encoded_video import EncodedVideo


from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)

from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from numpy.random import choice
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    Resize,
)
from torchvision.transforms import functional as F
import h5py
try:
    VIDEOS =  h5py.File('../../data/videos_context.hdf5','r', libver='latest' , swmr=True)
except:
    VIDEOS =  h5py.File('data/videos_context.hdf5','r', libver='latest' , swmr=True)

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
    text_mask = []

    video_list = []
    input_list = []
    label_list = []

    
    text_list_mask = None
    vid_mask = None


    for (input , label) in batch:
        text = input[0]
        input_list.append(text['input_ids'].tolist()[0])
        text_mask.append(text['attention_mask'].tolist()[0])
        
        vid_features = input[1]#[6:] for debug

        # TODO: DATASET SPECIFIC
        video_list.append(videoMAE_features(vid_features['vid_path'] , vid_features['timings'] , vid_features['speaker'] , check))
        label_list.append(label)
    batch_size = len(label_list)
    
    text_list_mask = torch.Tensor(np.array(text_mask))
    
    del text_mask
    
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
    
    text = {'input_ids':torch.Tensor(np.array(input_list)).type(torch.LongTensor) , 
            'attention_mask':text_list_mask,
        }

    visual_embeds = {'visual_embeds':torch.stack(video_list).permute(0,2,1,3,4) , 
            'attention_mask':vid_mask , 
        }
    return [text  , visual_embeds] , torch.Tensor(np.array(label_list))



class BertVideoMAE(nn.Module):
    """
    Model for Bert and VideoMAE classifier
    """

    def __init__(self, args ,dropout=0.5):
        super(BertVideoMAE, self).__init__()

        self.output_dim = args['output_dim']
        self.dropout = args['dropout']
        self.learn_PosEmbeddings = args['learn_PosEmbeddings']
        self.num_layers = args['num_layers']

        self.test_ctr = 1
        self.train_ctr = 1

        self.bert = AutoModel.from_pretrained('j-hartmann/emotion-english-distilroberta-base')
        self.videomae = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        
        self.bert_norm = nn.LayerNorm(768)
        self.vid_norm = nn.LayerNorm(768)

        self.dropout = nn.Dropout(self.dropout)
        self.linear1 = nn.Linear(768*2, self.output_dim)
        

    
    def forward(self, input_ids , text_attention_mask , video_embeds , visual_mask  , check = "train"):        
        #Transformer Time
        _, text_outputs = self.bert(input_ids= input_ids, attention_mask=text_attention_mask,return_dict=False)
        
        del _
        del input_ids
        del text_attention_mask
        

        vid_outputs = self.videomae(video_embeds , visual_mask)[0] # Now it has 2 dimensions 
        
        del video_embeds
        del visual_mask
        
        vid_outputs = torch.mean(vid_outputs, dim=1) # Now it has 2 dimensions 

        
        text_outputs = self.bert_norm(text_outputs) # 
        vid_outputs = self.vid_norm(vid_outputs) 

        # All of these ouputs are 4 x [batch_size , 768]

        #Concatenate Outputs
        tav = torch.cat([text_outputs , vid_outputs],dim=1)
        
        del text_outputs
        del vid_outputs

        #Classifier Head
        if check == "train":
            tav = self.dropout(tav)
        tav = self.linear1(tav)

        return tav # returns [batch_size,output_dim]