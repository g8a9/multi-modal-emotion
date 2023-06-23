import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim import AdamW
from utils.global_functions import save_model , load_model
import os
# from transformers.optimization import AdamW

import warnings
import wandb
from torch.utils.checkpoint import checkpoint

PATIENCE_ITER = 0
F1_ITER = 0
def get_statistics(input,label,model,criterion, Metric , check = "train",epoch = None):
    batch_loss = None
    label = label.to(f"cuda")

    
    mask = input['attention_mask'].to(f"cuda")
    input_id = input['input_ids'].squeeze(1).to(f"cuda")
    del input
    output = model(input_id, mask , check)
    del input_id
    del mask
    Metric.update_metrics(torch.argmax(output , dim = 1) , label.long())
    if criterion is not None:
        # batch_loss = criterion(output, label)
        batch_loss = criterion(output, label , epoch = epoch if epoch is not None else 0) # TODO: Turn this on with Sampler
    del output
    del label
    return batch_loss 

def get_statistics_big_batch(input,label,model,criterion, Metric , check = "train",epoch = None):
    batch_loss = None
    label = label.to(f"cuda")

    
    mask = input['attention_mask'].to(f"cuda")
    input_id = input['input_ids'].squeeze(1).to(f"cuda")
    del input
    output = checkpoint(model,input_id, mask , check, use_reentrant=False)
    del input_id
    del mask
    Metric.update_metrics(torch.argmax(output , dim = 1) , label.long())
    if criterion is not None:
        # batch_loss = criterion(output, label)
        batch_loss = criterion(output, label , epoch = epoch if epoch is not None else 0) # TODO: Turn this on with Sampler
    del output
    del label
    return batch_loss 


def grad_accum(epoch , train_dataloader , val_dataloader , model , criterion , optimizer , scheduler, clip , patience , Metric , prev_val_loss , prev_f1 , total_loss_train , iters , log_val , path):
    global PATIENCE_ITER, F1_ITER
    gen = iter(train_dataloader)
    batch_size = train_dataloader.batch_size
    fn = get_statistics_big_batch if batch_size > 32 else get_statistics
    for i in range( (iters // log_val) + 1):
        for j in range(log_val):
            try:
                batch_idx = i*log_val + j
                train_input , train_label = next(gen)
                accum_iter, accum_sum = train_dataloader.dataset.retGradAccum(i = batch_idx)
                train_batch_loss = fn(train_input , train_label , model , criterion , Metric ,  check="train" , epoch = epoch) / accum_iter 
                del train_input
                del train_label
                total_loss_train += train_batch_loss.item()

                # backward pass
                train_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_([ param for param in model.parameters() if param.requires_grad == True], clip)
                # do gradient accumulation here for dialogues
                if ((batch_idx + 1) % accum_sum == 0) or (batch_idx + 1 == iters): 
                    optimizer.step()
                    scheduler.step(epoch + batch_idx / iters)
                    model.zero_grad()
            except StopIteration:
                break
            
        prev_val_loss , prev_f1 = run_validation(epoch  , val_dataloader , model , criterion , optimizer , scheduler , Metric , prev_val_loss , prev_f1 , total_loss_train/iters , batch_idx  , log_val , path )
        if PATIENCE_ITER == patience:
            break
        #Do logging every log_val steps 
        
    return model , optimizer , criterion , prev_val_loss , prev_f1

def not_grad_accum(epoch , train_dataloader , val_dataloader , model , criterion , optimizer , scheduler, clip , patience , Metric , prev_val_loss , prev_f1 , total_loss_train , iters , log_val , path):
    global PATIENCE_ITER, F1_ITER
    gen = iter(train_dataloader)
    batch_size = train_dataloader.batch_size
    fn = get_statistics_big_batch if batch_size > 32 else get_statistics
    for i in range( (len(train_dataloader) // log_val) + 1):
        for j in range(log_val):
            try:
                batch_idx = i*log_val + j
                train_input , train_label = next(gen)
                train_batch_loss = fn(train_input , train_label , model , criterion , Metric ,  check="train" , epoch = epoch) 
                del train_input
                del train_label
                total_loss_train += train_batch_loss.item()

                # backward pass
                train_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_([ param for param in model.parameters() if param.requires_grad == True], clip)
                optimizer.step()
                scheduler.step(epoch + batch_idx / iters)
                model.zero_grad()
            except StopIteration:
                break
        prev_val_loss , prev_f1 = run_validation(epoch  , val_dataloader , model , criterion , optimizer , scheduler  , Metric , prev_val_loss , prev_f1 , total_loss_train/iters , batch_idx  , log_val , path )
        if PATIENCE_ITER == patience:
            break
    return model , optimizer , criterion , prev_val_loss , prev_f1


def run_validation(epoch  , val_dataloader , model ,  criterion , optimizer , scheduler , Metric , prev_val_loss , prev_f1 , curr_loss , step , log_val , path):
    global PATIENCE_ITER, F1_ITER
    log(Metric , curr_loss , "train")
    val_loss , weightedF1 = validate(val_dataloader , model ,  criterion, Metric , name="val")
    if weightedF1 > prev_f1:
        print(f"we have seen weightedF1 increase the previous best and we are updating our best f1 score to {weightedF1}")
        prev_f1 = weightedF1
        save_model(model ,  optimizer , criterion , scheduler , epoch , step, path , log_val)
    if val_loss < prev_val_loss:
        PATIENCE_ITER = 0
        prev_val_loss = val_loss
    else:
        PATIENCE_ITER += 1
        print(f"we have seen loss increase for {PATIENCE_ITER} steps and validation loss is {val_loss}, and previous best validation loss is {prev_val_loss}")
    return prev_val_loss , prev_f1



def validate(val_dataloader , model , criterion, Metric , name="val"):
    total_loss_val = 0
    with torch.no_grad():
        for val_input, val_label in val_dataloader:
            val_batch_loss = get_statistics(val_input , val_label , model , criterion , Metric , name , epoch = None )
            if criterion is not None:
                total_loss_val += val_batch_loss.item()
            del val_input
            del val_label
        weightedF1 = log(Metric , total_loss_val/len(val_dataloader) if criterion is not None else 0 , name)
    return total_loss_val/len(val_dataloader) , weightedF1


def one_epoch(epoch , train_dataloader , val_dataloader , model , criterion , optimizer , scheduler, clip , epoch_switch , patience , Metric , prev_val_loss , prev_f1):
    total_loss_train = 0   
    iters1 = len(train_dataloader[0])
    iters2 = len(train_dataloader[1])
    log_val1 = iters1 // 5
    log_val2 = iters2 // 5
    wandb.log({"log_val_multinomial" : log_val1 , "log_val_iterative" : log_val2})

    path = '/'.join(os.getcwd().split('/')[:-3]) + "/TAV_Text"
    if epoch % epoch_switch == 0:
        model , optimizer , criterion , prev_val_loss , prev_f1 = not_grad_accum(epoch , train_dataloader[0] , val_dataloader , model , criterion , optimizer , scheduler, clip , patience , Metric , prev_val_loss , prev_f1 , total_loss_train , iters1 , log_val1 , path)
    else:
        model , optimizer , criterion , prev_val_loss , prev_f1 = grad_accum(epoch , train_dataloader[1] , val_dataloader , model , criterion , optimizer , scheduler, clip , patience , Metric , prev_val_loss , prev_f1 , total_loss_train , iters2 , log_val2 , path)

    return model , optimizer , criterion , scheduler , prev_val_loss , prev_f1

   
def train_text_network(model , train_dataloader, val_dataloader, criterion,learning_rate, epochs , weight_decay , T_max ,Metric, patience , clip , epoch_switch , checkpoint = None):
    optimizer = AdamW([ param for param in model.parameters() if param.requires_grad == True], lr= learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer , T_0=T_max)  # To prevent fitting to local minima != global minima
    prev_val_loss = 100
    prev_f1 = 0
    path = '/'.join(os.getcwd().split('/')[:-3]) + "/TAV_Text"
    if checkpoint is not None:
        # epoch_num = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    for epoch_num in tqdm(range(epochs), desc="epochs"):
        wandb.log({"epoch": epoch_num,"learning_rate": scheduler.get_last_lr()[0]})
        optimizer.zero_grad()  # Zero out gradients before each epoch.
        
        model , optimizer , criterion , scheduler , prev_val_loss , prev_f1 = one_epoch(epoch_num , train_dataloader, val_dataloader ,  model , criterion , optimizer, scheduler , clip , epoch_switch , patience , Metric , prev_val_loss , prev_f1 ) 
        if PATIENCE_ITER == patience:
            break
    model , _ , _ = load_model(model , optimizer , criterion, path)
    return model

def evaluate_text(model , test_dataloader , Metric):
    validate(test_dataloader  , model , None , Metric , name = "test")


def log(Metric , loss , check = "train"):
    multiAcc , multiF1, multiRec, multiPrec , Acc, F1Macro, F1Weighted, Rec, Prec , _ = Metric.compute_scores(f"{check}")
    d1 = {
            f"{check}/loss": loss,
            f"{check}/acc": Acc,
            f"{check}/precision": Prec,
            f"{check}/recall" : Rec,
            f"{check}/weighted-f1-score": F1Weighted,
            f"{check}/macro-f1-score": F1Macro,
        }
    print(f"\n in {check} \n Confusion Matrix = \n{_} \n" , flush = True)
    wandb.log({**d1 , **multiF1, **multiRec, **multiPrec, **multiAcc}) 
    Metric.reset_metrics()
    return F1Weighted

