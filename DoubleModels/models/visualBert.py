import torch
from torch import nn
from transformers import VisualBertForPreTraining

import torch, torchvision
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image



def collate_batch(batch): # batch is a pseudo pandas array of two columns
    """
    Here we are going to take some raw-input and pre-process them into what we need
    So we can see all the steps in the ML as they go along
    """


	# pdb.set_trace()
    
    img_list = []
    input_list = []
    att_list = []
    tok_list = []
    image_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(size=(64, 64)),
        torchvision.transforms.ToTensor()
    ]
)

    

    label_list = []
   
    for (input , label) in batch:
        text = input[0]
        img_path = input[1]#[6:] for debug
        input_list.append(text['input_ids'].tolist()[0])
        att_list.append(text['attention_mask'].tolist()[0])
        tok_list.append(text['token_type_ids'].tolist()[0])
        
        img = Image.open(img_path).convert("RGB")
        img = image_transform(img)

        img_list.append(img)
        label_list.append(label)


    tensor_img = torch.stack(img_list).to("cuda")

    vision_module = torchvision.models.resnet50(pretrained=True).to("cuda")
    vision_module.fc = torch.nn.Linear(in_features=2048,out_features=1024).to("cuda")
    image_features = torch.nn.functional.relu(vision_module(tensor_img))
    image_features = image_features.reshape(len(label_list) , 1 , 1024)
    text = {'input_ids':torch.Tensor(np.array(input_list)).type(torch.LongTensor) , 
            'attention_mask':torch.Tensor(np.array(att_list)).type(torch.LongTensor) , 
            'token_type_ids':torch.Tensor(np.array(tok_list)).type(torch.LongTensor)
        }


    # return [text , img_list] , torch.Tensor(np.array(label_list))
    return [text , image_features.cpu()] , torch.Tensor(np.array(label_list))

class VBertClassifier(nn.Module):
    """
    Model for VisualBERT
    """

    def __init__(self, args ,dropout=0.5):
        super(VBertClassifier, self).__init__()

        self.output_dim = args['output_dim']

        self.vbert = VisualBertForPreTraining.from_pretrained('uclanlp/visualbert-nlvr2-coco-pre')

        self.dropout = nn.Dropout(dropout)

        self.linear = nn.Linear(30522, 256)

        self.final = nn.Linear(256, self.output_dim)

        self.linear1 = nn.Linear(256, 128)

        self.linear2 = nn.Linear(128, self.output_dim)

        self.sigmoid = nn.Sigmoid()

        self.relu = nn.LeakyReLU()

    def forward(self, input_ids , attention_mask = None , token_type_ids = None , visual_embeds = None , visual_attention_mask = None , visual_token_type_ids = None , check = "train"):

        # print(f"input_id = {input_ids}\n input_id.shape = {input_ids.shape}")

        pooled_output = self.vbert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, visual_embeds=visual_embeds, visual_attention_mask=visual_attention_mask, visual_token_type_ids=visual_token_type_ids)
        # print(f"pooled_output = {pooled_output}\n")  
        dropout_output = self.dropout(torch.mean(pooled_output['prediction_logits'] , dim = 1))
        

        linear_output = self.linear(dropout_output)
        # print(f"linear_output = {linear_output}\n linear_output.shape = {linear_output.shape}")

        x = self.sigmoid(linear_output)

        x = self.dropout(x)

        x = self.final(x)

        # x = self.sigmoid(x)

        # x = self.dropout(x)

        # x = self.linear2(x)

        return x # returns [batch_size,output_dim]
