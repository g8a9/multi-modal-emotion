import h5py
import torch
import numpy as np
import pandas as pd
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
from tqdm import tqdm

import gc

df = pd.read_pickle("../data/TAV_MELD_bounding_box.pkl")

def videoMAE_features(path  , timings  , check):
    # TODO: DATASET SPECIFIC
    if timings == None:
        beg = 0
        end = 500
    else:
        beg = timings[0]
        end = timings[1]
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
    
    # print(f"We have path {path}\ntimings are {timings}\nspeaker is {speaker}\ncheck is {check}\nsingular_func_ is {singular_func_.__name__ if singular_func_ is not None else None}" , flush= True)

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
                            # TODO: DATASET SPECIFIC                            
                            Lambda(lambda x: x / 255.0),
                            NormalizeVideo(mean, std),                            
                            # Hone in on either the left_speaker or right_speaker in the video
                            Resize((resize_to, resize_to)), # Need to be at 224,224 as our height and width for VideoMAE, bbox is for 224 , 224
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
                            Resize((resize_to, resize_to)),
                            # TODO: DATASET SPECIFIC                            
                        ]
                    ),
                ),
            ]
        )
    video = EncodedVideo.from_path(path)
    
    video_data = video.get_clip(start_sec=beg, end_sec=end)
    video_data = transform(video_data)
    return video_data['video'].numpy()


def write2File(writefile , path , timings, check):
    
    filename = f"{check}_{path.split('/')[-1][:-4]}"
    # generate some data for this piece of data
    data = videoMAE_features(path[3:]  , timings  , check)
    writefile.create_dataset(filename, data=data)
    gc.collect()
    
    
def fun(df , f , i): 
    sub_df = df.iloc[i*1000:(i+1)*1000]
    sub_df.progress_apply(lambda x: write2File(f , x['video_path'] , x['timings'] , x['split']  ) , axis = 1 )
    
f = h5py.File('../data/videos.hdf5','a', libver='latest' , swmr=True)
f.swmr_mode = True
tqdm.pandas()
for i in tqdm(range(13,14)): # change first arg in range, to the next applicable one in case it crashes
    fun(df , f, i)
    gc.collect()
read_file = h5py.File('../data/videos.hdf5','r', libver='latest' , swmr=True)
print(len(list(read_file.keys()))) 
# print(list(read_file.keys()))