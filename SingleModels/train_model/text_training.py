import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim import AdamW
from utils.global_functions import save_model , load_model

# from transformers.optimization import AdamW

import warnings
import wandb

PATIENCE_ITER = 0

def get_statistics(input,label,model,criterion, Metric , check = "train",epoch = None):
    batch_loss = None
    label = label.to(f"cuda")

    
    mask = input['attention_mask'].to(f"cuda")
    input_id = input['input_ids'].squeeze(1).to(f"cuda")
    # print(f"input_id = {input_id}")
    output = model(input_id, mask , check)
    
    Metric.update_metrics(torch.argmax(output , dim = 1) , label.long())
    if criterion is not None:
        # batch_loss = criterion(output, label)
        batch_loss = criterion(output, label , epoch = epoch if epoch is not None else 1) # TODO: Turn this on with Sampler
        
    return batch_loss 

def not_grad_accum(epoch , train_dataloader , val_dataloader , model  , criterion , optimizer , scheduler, clip , patience , Metric , prev_val_loss , total_loss_train , iters , log_val , path):
    global PATIENCE_ITER
    # print(iters)
    for batch_idx , (train_input, train_label) in enumerate(train_dataloader):
        train_batch_loss = get_statistics(train_input , train_label , model  , criterion , Metric, check="train" , epoch = epoch) 
        total_loss_train += train_batch_loss.item()

        # backward pass
        train_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_([ param for param in model.parameters() if param.requires_grad == True] , clip)
        optimizer.step()
        scheduler.step(epoch + batch_idx / iters)
        model.zero_grad()
        if ((batch_idx + 1) % log_val == 0) or (batch_idx + 1 == iters): 
            log(Metric , total_loss_train/iters , "train")
            val_loss = validate(val_dataloader , model  , criterion, Metric , name="val")
            if val_loss < prev_val_loss:
                PATIENCE_ITER = 0
                print(f"we have seen loss decrease the previous best and we are updating our best loss val to {val_loss}")
                prev_val_loss = val_loss
                save_model(model  , optimizer , criterion , scheduler , epoch , batch_idx, path , log_val)
            else:
                PATIENCE_ITER += 1
                print(f"we have seen loss increase for {PATIENCE_ITER} steps and validation loss is {val_loss}, and previous best validtion loss is {prev_val_loss}")
                if PATIENCE_ITER == patience:
                    break
    return prev_val_loss


    
def grad_accum(epoch , train_dataloader , val_dataloader , model  , criterion , optimizer , scheduler, clip , patience , Metric , prev_val_loss , total_loss_train , iters , log_val , path):
    global PATIENCE_ITER
    for batch_idx , (train_input, train_label) in enumerate(train_dataloader):
        accum_iter, accum_sum = train_dataloader.dataset.retGradAccum(i = batch_idx)
        train_batch_loss = get_statistics(train_input , train_label , model  , criterion , Metric, check="train" , epoch = epoch) / accum_iter 
        
        total_loss_train += train_batch_loss.item()

        # backward pass
        train_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_([ param for param in model.parameters() if param.requires_grad == True] , clip)
        optimizer.step()
        scheduler.step(epoch + batch_idx / iters)
        model.zero_grad()
        # do gradient accumulation here for dialogues
        if ((batch_idx + 1) % accum_sum == 0) or (batch_idx + 1 == iters): 
            optimizer.step()
            scheduler.step(epoch + batch_idx / iters)
            model.zero_grad()
        #Do logging every log_val steps 
        if ((batch_idx + 1) % log_val == 0) or (batch_idx + 1 == iters): 
            log(Metric , total_loss_train/iters , "train")
            val_loss = validate(val_dataloader , model  , criterion, Metric , name="val")
            if val_loss < prev_val_loss:
                PATIENCE_ITER = 0
                print(f"we have seen loss decrease the previous best and we are updating our best loss val to {val_loss}")
                prev_val_loss = val_loss
                save_model(model  , optimizer , criterion , scheduler , epoch , batch_idx, path , log_val)
            else:
                PATIENCE_ITER += 1
                print(f"we have seen loss increase for {PATIENCE_ITER} steps and validation loss is {val_loss}, and previous best validtion loss is {prev_val_loss}")
                if PATIENCE_ITER == patience:
                    break
    return prev_val_loss

def validate(val_dataloader , model , criterion, Metric , name="val"):
    total_loss_val = 0
    with torch.no_grad():
        for val_input, val_label in val_dataloader:
            val_batch_loss = get_statistics(val_input , val_label , model , criterion , Metric , name , epoch = None )
            if criterion is not None:
                total_loss_val += val_batch_loss.item()

        log(Metric , total_loss_val/len(val_dataloader) if criterion is not None else 0 , name)
    return total_loss_val/len(val_dataloader)

def one_epoch(epoch , train_dataloader , val_dataloader , model , criterion , optimizer , scheduler, clip , epoch_switch , patience , Metric , prev_val_loss):
    total_loss_train = 0   
    iters = len(train_dataloader)
    log_val = 2400
    path = "/home/prsood/projects/def-whkchun/prsood/Text_Train"
    if epoch % epoch_switch == 0:
        prev_val_loss = not_grad_accum(epoch , train_dataloader , val_dataloader , model , criterion , optimizer , scheduler, clip , patience , Metric , prev_val_loss , total_loss_train , iters , log_val , path)
    else:
        prev_val_loss = grad_accum(epoch , train_dataloader , val_dataloader , model , criterion , optimizer , scheduler, clip , patience , Metric , prev_val_loss , total_loss_train , iters , log_val , path)
    model , optimizer , criterion = load_model(model , optimizer , criterion, path)
    return model , optimizer , criterion , scheduler , prev_val_loss

def train_text_network(model , train_dataloader, val_dataloader, criterion,learning_rate, epochs , weight_decay , T_max ,Metric, patience , clip , epoch_switch , checkpoint = None):
    optimizer = AdamW([ param for param in model.parameters() if param.requires_grad == True], lr= learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer , T_0=T_max)  # To prevent fitting to local minima != global minima
    prev_val_loss = 100
    
    if checkpoint is not None:
        # epoch_num = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    for epoch_num in tqdm(range(epochs), desc="epochs"):
        wandb.log({"epoch": epoch_num,"learning_rate": scheduler.get_last_lr()[0]})
        optimizer.zero_grad()  # Zero out gradients before each epoch.
        
        model , optimizer , criterion , scheduler , prev_val_loss = one_epoch(epoch_num , train_dataloader, val_dataloader ,  model , criterion , optimizer, scheduler , clip , epoch_switch , patience , Metric , prev_val_loss ) 
                
        if PATIENCE_ITER == patience:
            return model
    return model


def evaluate_text(model  , test_dataloader , Metric):
    validate(test_dataloader  , model  , None , Metric , name = "test")



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
    print(f"\n in {check} \n Confusion Matrix = {_} \n" , flush = True)
    wandb.log({**d1 , **multiF1, **multiRec, **multiPrec, **multiAcc}) 
    Metric.reset_metrics()