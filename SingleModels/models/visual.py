import torch
from torch import nn
import numpy as np
import torchvision.transforms as T
from PIL import Image
from decord import VideoReader
from decord import cpu
from pytorchvideo.data.encoded_video import EncodedVideo


from utils.transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)

from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)



def get_index(num_frames, num_segments=16, dense_sample_rate=8):
    sample_range = num_segments * dense_sample_rate
    sample_pos = max(1, 1 + num_frames - sample_range)
    t_stride = dense_sample_rate
    start_idx = 0 if sample_pos == 1 else sample_pos // 2
    offsets = np.array([
        (idx * t_stride + start_idx) %
        num_frames for idx in range(num_segments)
    ])
    return offsets # + 1  i rm the +1 because i would hit an out of bounds error sometimes


def load_video(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    num_frames = len(vr)
    frame_indices = get_index(num_frames, 16, 16)

    # transform
    crop_size = 224
    scale_size = 256
    input_mean = [0.485, 0.456, 0.406]
    input_std = [0.229, 0.224, 0.225]

    transform = T.Compose([
        GroupScale(int(scale_size)),
        GroupCenterCrop(crop_size),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(input_mean, input_std) 
    ])

    images_group = list()
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy())
        images_group.append(img)
    torch_imgs = transform(images_group)
    
    # The model expects inputs of shape: B x C x T x H x W
    TC, H, W = torch_imgs.shape
    torch_imgs = torch_imgs.reshape(1, TC//3, 3, H, W).permute(0, 2, 1, 3, 4)

    return torch_imgs


def resnet3d_features(path):
    side_size = 256
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    crop_size = 256
    num_frames = 8
    sampling_rate = 8
    frames_per_second = 30

# Note that this transform is specific to the slow_R50 model.
    transform =  ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x/255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(
                    size=side_size
                ),
                CenterCropVideo(crop_size=(crop_size, crop_size))
            ]
        ),
    )

# The duration of the input clip is also specific to the model.
    clip_duration = (num_frames * sampling_rate)/frames_per_second

    video = EncodedVideo.from_path(path)
    video_data = video.get_clip(start_sec=0, end_sec=clip_duration)
    video_data = transform(video_data)
    
    return video_data['video']




def collate_batch(batch): # batch is a pseudo pandas array of two columns
    """
    Here we are going to take some raw-input and pre-process them into what we need
    So we can see all the steps in the ML as they go along
    """
    video_list = []
    label_list = []

    
    for (input_path , label) in batch:
        # print(input_path,"\n", flush = True)
        # val = resnet3d_features(input_path)
        # print(f"val  is {val}", flush = True)
        video_list.append(resnet3d_features(input_path))
        label_list.append(label)

    

    return torch.stack(video_list)  , torch.Tensor(np.array(label_list))




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

        self.input_dim = args["input_dim"]
        self.output_dim = args["output_dim"]

        self.hidden_dim = args['hidden_layers']

        self.hidden_layers = [] 

        self.in_layer = nn.Conv3d(self.input_dim , self.hidden_dim[0] , kernel_size=3) 

        for i in range(0,len(self.hidden_dim) - 1): 
            self.hidden_layers.append(
                nn.Conv3d(self.hidden_dim[i], self.hidden_dim[i+1] , kernel_size=3)
            )
        self.act = nn.GELU()
        self.out_layer = nn.Linear(18585600, self.output_dim)
        self.sigmoid = nn.Sigmoid()
        # self.drop = nn.Dropout(0.5)

    def forward(self, x):
        # x shape ([16, 1, 3, 16, 224, 224])
        bs = len(x)
        x = x.squeeze(1)
        # x after conv3D : ([16, 32, 14, 222, 222])
        # because we turn the 3 into a 32

        # after we run the second hidden dim
        # x after second conv3D [16, 32, 11, 219, 219]

        x = self.in_layer(x)
        x = self.act(x)


        for layer in self.hidden_layers:
            x = layer(x)
            x = self.act(x)
        x = x.view(bs, -1)
        x = self.out_layer(x)
        x = self.sigmoid(x)
        return x.view(-1) # since the second dim should be 1
