from transformers import logging
logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore") 
import torch
from tqdm import tqdm
import wandb
from utils.early_stopping import EarlyStopping
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers.optimization import AdamW
import os
import torch, torchvision
import matplotlib.pyplot as plt
from copy import deepcopy
import detectron2
import cv2
import pandas as pd
import numpy as np
import torch.distributed as dist
from ..models.tav import PreFormer
import dill as pickle

PRE = PreFormer().to(f"cpu")

def get_statistics(input,label,model,criterion,total_loss,Metric,check="train",location = "/home/prsood/projects/def-whkchun/prsood/multi-modal-emotion/Inference/visbertTest.txt"):
    batch_loss = None 
    device = "cuda"
    text = input[0]
    text_input_ids = text["input_ids"]
    text_attention_mask = text["attention_mask"]

    audio_features = input[1]
    audio_input_ids = audio_features["audio_features"]
    audio_attention_mask = audio_features["attention_mask"]

    video_embeds = input[2]
    video_input_ids = video_embeds["visual_embeds"]
    video_attention_mask = video_embeds["attention_mask"]   
    if text_attention_mask != None: # Then all of them are
        tav, tav_embed , attention_mask = PRE(input_ids = text_input_ids.to(f"cpu") , audio_features = audio_input_ids.to(f"cpu") , video_embeds = video_input_ids.to(f"cpu") , text_mask = text_attention_mask.to(f"cpu") , audio_mask = audio_attention_mask.to(f"cpu") , visual_mask = video_attention_mask.to(f"cpu"), device = device, train = True if check == "train" else False)    
        output = model(tav.to(device), tav_embed.to(device), attention_mask.to(device))    
    else:
        tav, tav_embed , attention_mask = PRE(input_ids = text_input_ids.to(f"cpu") , audio_features = audio_input_ids.to(f"cpu") , video_embeds = video_input_ids.to(f"cpu") , text_mask = None , audio_mask = None , visual_mask = None , device = device, train = True if check == "train" else False)    
        output = model(tav.to(device), tav_embed.to(device), None)    

    del tav_embed
    label = label.type(torch.LongTensor).to(device)
    Metric.update_metrics(torch.argmax(output , dim = 1) , label.long())
    if criterion is not None:
        batch_loss = criterion(output, label)
        total_loss += batch_loss.item()
    # print(f"truth = {label}\npreds = {torch.argmax(output , dim = 1)}\n\n" , flush = True)
    return batch_loss , total_loss


def one_epoch(train_dataloader , model , criterion , optimizer , scheduler, clip , Metric):
    total_loss_train = 0   
    # batch accumulation parameter
    accum_iter = 1
    for batch_idx , (train_input, train_label) in enumerate(tqdm(train_dataloader , desc="training")):
        
        train_batch_loss , total_loss_train = get_statistics(train_input , train_label , model , criterion , total_loss_train , Metric)
        
        train_batch_loss = train_batch_loss / accum_iter 

        # backward pass
        train_batch_loss.backward()

        # weights update
        if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_dataloader)):
            optimizer.step()
            for param in model.parameters():
                param.grad = None
        # # if clip is not None:
        # #     torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        # train_batch_loss.backward()# Back propagate
        # optimizer.step()  # Run optimization step
            torch.cuda.empty_cache()
        # torch.cuda.empty_cache()
          
    
    return model , optimizer , train_batch_loss,total_loss_train/len(train_dataloader)

def validate(val_dataloader , model , criterion, Metric , name="val" , location = "val"):
    total_loss_val = 0
    with torch.no_grad():
        for val_input, val_label in tqdm(val_dataloader, desc="validate" if name == "val" else "testing"):
            val_batch_loss , total_loss_val = get_statistics(val_input , val_label , model , criterion , total_loss_val , Metric , name , location)
    
    return val_batch_loss,total_loss_val/len(val_dataloader)

    
   
def train_tav_network(model, train_dataloader, val_dataloader, criterion,learning_rate, epochs , weight_decay , T_max ,Metric, patience=None , clip=None):
    optimizer = AdamW([ param for param in model.parameters() if param.requires_grad == True], lr= learning_rate, weight_decay=weight_decay)
    earlystop = EarlyStopping("",model,patience,model_name=model.__class__.__name__)
    scheduler = CosineAnnealingLR(optimizer , T_max=T_max)  # To prevent fitting to local minima != global minima
    for epoch_num in tqdm(range(epochs), desc="epochs"):
        optimizer.zero_grad()  # Zero out gradients before each epoch.
        model , optimizer , train_batch_loss,train_loss = one_epoch(train_dataloader ,  model , criterion , optimizer, scheduler , clip , Metric ) 
        # print(f"We are here on rank = {rank} after train" , flush = True)

        multiAcc , multiF1, multiRec, multiPrec , Acc, F1Macro, F1Weighted, Rec, Prec , _ = Metric.compute_scores("train")
        d1 = {
                "epoch": epoch_num,
                # "train/batch_loss": train_batch_loss,
                "train/loss": train_loss,
                "train/acc": Acc,
                "train/precision": Prec,
                "train/recall" : Rec,
                "train/weighted-f1-score": F1Weighted,
                "train/macro-f1-score": F1Macro,
            }
        print(f"\n in train \n Confusion Matrix = {_} \n" , flush = True)
        wandb.log({**d1 , **multiF1, **multiRec, **multiPrec, **multiAcc}) 
        Metric.reset_metrics()

        scheduler.step() # this is for CosineAnnealing

        val_batch_loss,val_loss = validate(val_dataloader  , model , criterion , Metric)
        multiAcc , multiF1, multiRec, multiPrec , Acc, F1Macro, F1Weighted, Rec, Prec , _ = Metric.compute_scores("val")
        d1 = {
                # "val/batch_loss": val_batch_loss,
                "val/loss": val_loss,
                "val/acc": Acc,
                "val/precision": Prec,
                "val/recall": Rec,
                "val/weighted-f1-score": F1Weighted,
                "val/macro-f1-score": F1Macro,
                }
        print(f"\n in val \n Confusion Matrix = {_} \n" , flush = True)
        # print("We have reached val logging" , flush = True)
        wandb.log({**d1 , **multiF1, **multiRec, **multiPrec, **multiAcc})
        Metric.reset_metrics()

        if patience is not None: # this is to make sure that gradietn descent isnt stuck in a local minima 
            if earlystop(model, val_loss):
                model = earlystop.best_state
        if epoch_num == 0:
            os.mkdir(f"/home/prsood/projects/ctb-whkchun/prsood/TAV_Train/{wandb.run.name}/")
        torch.save(model.state_dict(), f"/home/prsood/projects/ctb-whkchun/prsood/TAV_Train/{wandb.run.name}/{epoch_num}.pt")
    return model

def evaluate_tav(model, test_dataloader , Metric , location = "/home/prsood/projects/def-whkchun/prsood/multi-modal-emotion/Inference/visbertTest.txt"):
    validate(test_dataloader  , model , None , Metric , name = "test" , location=location)
    multiAcc , multiF1, multiRec, multiPrec , Acc, F1Macro, F1Weighted, Rec, Prec , ConfusionMatrix = Metric.compute_scores("test")
    
    d1 = {
            "test/acc": Acc,
            "test/precision": Prec,
            "test/recall": Rec,
            "test/weighted-f1-score": F1Weighted,
            "test/macro-f1-score": F1Macro,
            "test/ConfusionMatrix": ConfusionMatrix,
            }
    
    wandb.log({**d1 , **multiF1, **multiRec, **multiPrec, **multiAcc})   
    print(f"\n in TEST \n Confusion Matrix = {ConfusionMatrix} \n", flush = True)
