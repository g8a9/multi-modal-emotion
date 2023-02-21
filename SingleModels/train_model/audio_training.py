import torch
from tqdm import tqdm
import wandb
from utils.early_stopping import EarlyStopping
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers.optimization import AdamW
import numpy as np
import os



def get_statistics(input,label,model,criterion,total_loss, Metric):
    batch_loss = None
    # print("inside Stats" , flush = True) 
    label = label.to(f"cuda")


    print(f"inputs before model is \n input = {input} \n labels = {label}")

    
    if model.__class__.__name__ == "AutoModelForAudioClassification":
        output = model(input.to(f"cuda")) # our output should be of size : (batch_size x num_classes) so 4x7 -> argmax goes to 4x1
        output = output.logits
    else:
        input_values = input[0]
        att_mask = input[1]
        output = model(input_values = input_values , attention_mask = att_mask) # our output should be of size : (batch_size x num_classes) so 4x7 -> argmax goes to 4x1


    # print("before calc loss" , flush = True)
    if criterion is not None:
        batch_loss = criterion(output, label.long())
        total_loss += batch_loss.item()
    
    print(f"outputs of the model = \n {output} \n {torch.argmax(output , dim = 1)}")
    
    # print("before update metrics" , flush = True)
    Metric.update_metrics(torch.argmax(output , dim = 1) , label.long())

    return batch_loss , total_loss 


def one_epoch(train_dataloader , model , criterion , optimizer, clip , Metric):
    total_loss_train = 0


    for train_input, train_label in tqdm(train_dataloader , desc="training"):
        # print("before getting Stats" , flush = True)


        train_batch_loss , total_loss_train  = get_statistics(train_input , train_label , model , criterion , total_loss_train , Metric)

        train_batch_loss.backward()  # Back propagate
        if clip is not None: # so gradients can beccome super big and super small, this clip is like a little flag that says we are too big or too small, 
            # lets cut it off here eg. clip is 50, if our gradients are 200 itll jsut make it 50 
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()  # Run optimization step
    
    return  model , optimizer , train_batch_loss,total_loss_train/len(train_dataloader.dataset)




def validate(val_dataloader , model , criterion, Metric , name = "val"):
    total_loss_val = 0
    with torch.no_grad():

        for val_input, val_label in tqdm(val_dataloader, desc = "validate" if name == "val" else "testing" ):

            # print(f"Validation input and labels \n input = {val_input} \n label = {val_label} \n")
            print("VALIDATION")
            val_batch_loss , total_loss_val = get_statistics(val_input , val_label , model , criterion , total_loss_val  , Metric )

    return  val_batch_loss,total_loss_val/len(val_dataloader.dataset) 


def train_audio_network(model, train_dataloader, val_dataloader, criterion , learning_rate, epochs , weight_decay,T_max, Metric , patience = None, clip = None  ) :  # TODO Fill in the information necessary as variables here.
    """
    Trains a single-task audio model.
    :train_data: folder containing 10 pickle files of train
    :val_data: just the val data
    :model: The model function that you want to htrain.
    :epochs (int): The number of Epochs to train for.
    :patience (int): Set the patience for early stopping.
    :clip (float): Set value for gradient clipping.
    """    
    
    optimizer = AdamW(model.parameters(), lr = learning_rate , weight_decay=weight_decay)

    earlystop = EarlyStopping("",model,patience , model_name=model.__class__.__name__)

    scheduler = CosineAnnealingLR(optimizer , T_max=T_max)

    for epoch_num in tqdm(range(epochs), desc="epochs"):
        model.train() # we want the model flag for training to be set true, this way the model knows its about to change
        optimizer.zero_grad()  # Zero out gradients before each epoch.
        
        # this is where we start to train our model
        model , optimizer , train_batch_loss,train_loss = one_epoch(train_dataloader ,  model , criterion , optimizer , clip , Metric ) 
        multiAcc , multiF1, multiRec, multiPrec , Acc, F1, Rec, Prec , _ = Metric.compute_scores("train")
        d1 = {
                "epoch": epoch_num,
                "train/batch_loss": train_batch_loss,
                "train/train_loss": train_loss,
                "train/acc": Acc,
                "train/precision": Prec,
                "train/recall" : Rec,
                "train/f1-score": F1,
            }
        print(f"\n in train \n Confusion Matrix = {_} \n")
        wandb.log({**d1 , **multiF1, **multiRec, **multiPrec, **multiAcc}) 
        Metric.reset_metrics()
        scheduler.step() # this is for CosineAnnealing

        model.eval() # this is where we put the flag for evaluating on so we dont change anything by accident
        optimizer.zero_grad()  # Zero out gradients before each epoch.

        val_batch_loss,val_loss = validate(val_dataloader  , model , criterion , Metric)

        multiAcc , multiF1, multiRec, multiPrec , Acc, F1, Rec, Prec , _ = Metric.compute_scores("val")

        d1 = {
                "val/batch_loss": val_batch_loss,
                "val/total_loss_val": val_loss,
                "val/total_acc_val": Acc,
                "val/precision": Prec,
                "val/recall": Rec,
                "val/f1-score": F1,
                }
        print(f"\n in val \n Confusion Matrix = {_} \n")
        wandb.log({**d1 , **multiF1, **multiRec, **multiPrec, **multiAcc})
        Metric.reset_metrics()

        if patience is not None: # this is to make sure that gradietn descent isnt stuck in a local minima 
            if earlystop(model, val_loss):
                model = earlystop.best_state
                
    return model

def evaluate_audio(model, test_dataloader , Metric):
    model.eval() # this is where we put the flag for evaluating on so we dont change anything by accident
    validate(test_dataloader  , model , None , Metric , name = "test")
    multiAcc , multiF1, multiRec, multiPrec , Acc, F1, Rec, Prec , ConfusionMatrix = Metric.compute_scores("test")
    d1 = {
            "test/total_acc_test": Acc,
            "test/precision": Prec,
            "test/recall": Rec,
            "test/f1-score": F1,
            "test/ConfusionMatrix": ConfusionMatrix,
            }
    wandb.log({**d1 , **multiF1, **multiRec, **multiPrec, **multiAcc})
    print(f"\n in TEST \n Confusion Matrix = {ConfusionMatrix} \n")
           

