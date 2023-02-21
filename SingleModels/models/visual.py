
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from decord import VideoReader
from decord import cpu


from utils.transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
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

def collate_batch(batch): # batch is a pseudo pandas array of two columns
    """
    Here we are going to take some raw-input and pre-process them into what we need
    So we can see all the steps in the ML as they go along
    """
    video_list = []
    label_list = []

    
    for (input_path , label) in batch:
        video_list.append(load_video(input_path))
        label_list.append(label)
    
    return torch.stack(video_list)  , torch.Tensor(np.array(label_list))



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
