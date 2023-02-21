import torch.nn as nn
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50 

def collate_batch(batch): # batch is a pseudo pandas array of two columns
    """
    Here we are going to take some raw-input and pre-process them into what we need
    So we can see all the steps in the ML as they go along
    """
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(32, scale=(0.75, 1.0)),
        transforms.ToTensor()])

    img_list = []
    label_list = []
    # pdb.set_trace()
    
   
    for (input_path , label) in batch:
        img_list.append(transform(Image.open(input_path).convert("RGB")))
        label_list.append(label)

    return torch.stack(img_list) , torch.Tensor(np.array(label_list))


class ResnetClassification(nn.Module):
    """ ResNet50 for classification"""
    def __init__(self , args):

        super(ResnetClassification, self).__init__()
        model = resnet50(pretrained=True) 
        for param in model.parameters():
            param.requires_grad = False

        num_ftrs = model.fc.in_features

        model.fc = nn.Linear(num_ftrs , args["output_dim"])

        # nNLayers = []

        # nNLayers.append(nn.Conv2d(num_ftrs, args['hidden_layers'][0] , kernel_size=3))

        # for i in range(0,len(args['hidden_layers']) - 1): # so we get to the second last value, since last value corresponds to the out layer
        #     nNLayers.append(nn.Conv2d(args['hidden_layers'][i], args['hidden_layers'][i+1] , kernel_size=3))

        # nNLayers.append(nn.Linear(args['hidden_layers'][-1], args["output_dim"]))

        # model.fc = nn.Sequential(*nNLayers)

        self.model = model

    def forward(self, inputs):
        # print(f"input = {inputs}\n input.shape = {inputs.shape}")
        out = self.model(inputs)
        # print(f"Output = {out}\n Output.shape = {out.shape}")
        return out



class ImageClassification(nn.Module):
    """A simple ConvNet for binary classification."""
    def __init__(self , args):
        super(ImageClassification, self).__init__()

        self.input_dim = args["input_dim"]
        self.output_dim = args["output_dim"]

        self.hidden_dim = args['hidden_layers']

        self.hidden_layers = [] # array of all hidden layers

        self.in_layer = nn.Conv2d(self.input_dim , self.hidden_dim[0] , kernel_size=3) # 32 , 64 , 42 , 20

        for i in range(0,len(self.hidden_dim) - 1): # so we get to the second last value, since last value corresponds to the out layer
            self.hidden_layers.append(
                nn.Conv2d(self.hidden_dim[i], self.hidden_dim[i+1] , kernel_size=3)
            )

        self.fc = nn.Linear(25088, args['output_dim'])
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        bs = len(x)

        fx = self.in_layer(x)
        fx = self.relu(fx)

        for layer in self.hidden_layers:
            fx = layer(fx)
            fx = self.relu(fx)

        fx = fx.view(bs, -1)
        fx = self.fc(fx)
        fx = self.sigmoid(fx)
        return fx.view(-1)

