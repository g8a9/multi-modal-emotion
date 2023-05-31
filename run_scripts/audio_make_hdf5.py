import h5py
import torch
import numpy as np
import pandas as pd
from pytorchvideo.data.encoded_video import EncodedVideo
from numpy.random import choice
import torchaudio
import math
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

def speech_file_to_array_fn(path , timings , speaker , target_sampling_rate = 16000 , check = "train"):
    
    speech_array, sampling_rate = torchaudio.load(path)
    if speaker is None and timings is not None: # MELD
        # print(f"timings are {timings}\nsampling_rate is {sampling_rate}\nlen of speech_array is {len(speech_array[0])}\nsingular_func_ is {singular_func_.__name__}" , flush  = True)
        start = timings[0]
        end = timings[1]
        if end - start > .2:
            start_sample = math.floor(start * sampling_rate)
            end_sample = math.ceil(end * sampling_rate)
            # extract the desired segment
            if start_sample - end_sample > 3280:
                speech_array = speech_array[:, start_sample:end_sample]
            # print(f"len of speech_array is {len(speech_array[0])} after timings" , flush  = True)
    
    
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    del sampling_rate
    speech_array = resampler(speech_array).squeeze()
    speech_array = torch.mean(speech_array, dim=0) if len(speech_array.shape) > 1 else speech_array # over the channel dimension
    return speech_array.numpy()


def write2File(writefile , path , timings , speaker, check):
    
    filename = f"{check}_{path.split('/')[-1][:-4]}"
    # generate some data for this piece of data
    data = speech_file_to_array_fn(path[3:]  , timings , speaker)
    writefile.create_dataset(filename, data=data)
    gc.collect()
    
    
def fun(df , f , i): 
    sub_df = df.iloc[i*1000:(i+1)*1000]
    sub_df.progress_apply(lambda x: write2File(f , x['audio_path'] , x['timings'] , None , x['split']  ) , axis = 1 )
    
f = h5py.File('../data/audio.hdf5','a', libver='latest' , swmr=True)
f.swmr_mode = True
tqdm.pandas()
for i in tqdm(range(0,14)): # change first arg in range, to the next applicable one in case it crashes
    fun(df , f, i)
    gc.collect()
read_file = h5py.File('../data/audio.hdf5','r', libver='latest' , swmr=True)
len(list(read_file.keys())) , list(read_file.keys())
x = torch.Tensor(read_file['train_dia0_utt0'][()])
torch.save(x , 'train_dia0_utt0_numpy_audio.pt')
