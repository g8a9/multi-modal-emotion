from transformers import logging
logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore") 
import torch
from tqdm import tqdm
import wandb
from utils.early_stopping import EarlyStopping
from utils.global_functions import save_model , load_model
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim import AdamW
import torch
import os
from torchmetrics.classification import MulticlassF1Score

scalarWeightedF1 = MulticlassF1Score(7, 1, 'weighted', 'global', None, False).to("cuda")

def get_statistics(input , label , model , PREFormer , criterion , Metric , check="train" , epoch = None):
    global scalarWeightedF1
    batch_loss = None 
    device = "cuda"
    batch_size = len(label)
    text = input[0]
    text_input_ids = text["input_ids"].to(device)
    text_attention_mask = text["attention_mask"].to(device)

    audio_features = input[1]
    audio_input_ids = audio_features["audio_features"].to(device)
    audio_attention_mask = audio_features["attention_mask"].to(device)

    video_embeds = input[2]
    video_input_ids = video_embeds["visual_embeds"].to(device)
    video_attention_mask = video_embeds["attention_mask"].to(device)


    # assert video_input_ids.shape == (len(label), 16, 3, 224, 224) , f"Shape of video is {video_input_ids.shape}"
 
    output = model(input_ids = text_input_ids , text_attention_mask = text_attention_mask , audio_features = audio_input_ids  , audio_mask = audio_attention_mask
                   , video_embeds = video_input_ids , visual_mask = video_attention_mask  , check = check)
    
    del text_input_ids
    del text_attention_mask
    del audio_input_ids
    del audio_attention_mask
    del video_input_ids
    del video_attention_mask
    
    preds = torch.argmax(output , dim = 1)
    label = label.type(torch.LongTensor).to(device)
    
    f1 = None # scalarWeightedF1(preds , label.long())
    
    Metric.update_metrics(preds , label.long())
    del preds
    if criterion is not None:
        # batch_loss = criterion(output, label)
        batch_loss = criterion(output, label , epoch = epoch if epoch is not None else 1) # TODO: Turn this on with Sampler
    del label
    del output
    return batch_loss , f1

PATIENCE_ITER = 0
F1_ITER = 0

def grad_accum(epoch , train_dataloader , val_dataloader , model , PREFormer , criterion , optimizer , scheduler, clip , patience , Metric , prev_val_loss , prev_f1  , total_loss_train , iters , log_val , path):
    global PATIENCE_ITER
    global F1_ITER
    gen = iter(train_dataloader)
    for i in range( (iters // log_val) + 1):
        for j in range(log_val):
            try:
                batch_idx = i*log_val + j
                train_input , train_label = next(gen)
                accum_iter, accum_sum = train_dataloader.dataset.retGradAccum(i = batch_idx)
                train_batch_loss , f1 = get_statistics(train_input , train_label , model , PREFormer , criterion , Metric, check="train" , epoch = epoch) 
                total_loss_train +=  (train_batch_loss.item() / accum_iter )


                # loss_factor = 0.5
                # f1_factor = 1.0 - loss_factor
                # lr_factor = (loss_factor * train_batch_loss.item() / iters + f1_factor * (1 - f1))
                
                # backward pass
                train_batch_loss.backward()
                # torch.nn.utils.clip_grad_norm_([ param for param in model.parameters() if param.requires_grad == True] + [ param for param in PREFormer.parameters() if param.requires_grad == True], clip)
                torch.nn.utils.clip_grad_norm_([ param for param in model.parameters() if param.requires_grad == True] , clip)
                # do gradient accumulation here for dialogues
                if ((batch_idx + 1) % accum_sum == 0) or (batch_idx + 1 == iters): 
                    optimizer.step()
                    scheduler.step(epoch + batch_idx / iters )
                    model.zero_grad()
                    # PREFormer.zero_grad()
            except StopIteration:
                print(f" batch index is {batch_idx}" , flush = True)
                break
        
        prev_val_loss , prev_f1 = run_validation(epoch  , val_dataloader , model , PREFormer , criterion , optimizer , scheduler , Metric , prev_val_loss , prev_f1 , total_loss_train/iters , batch_idx  , log_val , path )
        if PATIENCE_ITER == patience or F1_ITER == patience:
            break
        #Do logging every log_val steps 
        
    return prev_val_loss , prev_f1

def not_grad_accum(epoch , train_dataloader , val_dataloader , model , PREFormer , criterion , optimizer , scheduler, clip , patience , Metric , prev_val_loss , prev_f1 , total_loss_train , iters , log_val , path):
    global PATIENCE_ITER
    global F1_ITER
    gen = iter(train_dataloader)
    
    for i in range( (iters // log_val) + 1):
        for j in range(log_val):
            try:
                batch_idx = i*log_val + j
                train_input , train_label = next(gen)
                train_batch_loss , f1 = get_statistics(train_input , train_label , model , PREFormer , criterion , Metric, check="train" , epoch = epoch) 
                total_loss_train += train_batch_loss.item()
                
                # loss_factor = 0.5
                # f1_factor = 1.0 - loss_factor
                # lr_factor = (loss_factor * train_batch_loss.item() / iters + f1_factor * (1 - f1))

                # backward pass
                train_batch_loss.backward()
                # torch.nn.utils.clip_grad_norm_([ param for param in model.parameters() if param.requires_grad == True] + [ param for param in PREFormer.parameters() if param.requires_grad == True], clip)
                torch.nn.utils.clip_grad_norm_([ param for param in model.parameters() if param.requires_grad == True] , clip)
                optimizer.step()
                scheduler.step(epoch + batch_idx / iters )
                model.zero_grad()
                # PREFormer.zero_grad()
            except StopIteration:
                print(f" batch index is {batch_idx}" , flush = True)
                break
        prev_val_loss , prev_f1 = run_validation(epoch  , val_dataloader , model , PREFormer , criterion , optimizer , scheduler  , Metric , prev_val_loss , prev_f1 , total_loss_train/iters , batch_idx  , log_val , path )
        if PATIENCE_ITER == patience or F1_ITER == patience:
            break
    return prev_val_loss , prev_f1


def run_validation(epoch  , val_dataloader , model , PREFormer , criterion , optimizer , scheduler , Metric , prev_val_loss , prev_f1 , curr_loss , step , log_val , path):
    global PATIENCE_ITER
    global F1_ITER
    log(Metric , curr_loss , "train")
    val_loss , weightedF1 = validate(val_dataloader , model , PREFormer , criterion, Metric , name="val")
    if weightedF1 > prev_f1:
        print(f"we have seen weightedF1 increase the previous best and we are updating our best f1 score to {weightedF1}")
        prev_f1 = weightedF1
        save_model(model , PREFormer , optimizer , criterion , scheduler , epoch , step, path , log_val)
    elif prev_f1 < .4:
        F1_ITER += 1
    if val_loss < prev_val_loss:
        PATIENCE_ITER = 0
        prev_val_loss = val_loss
    else:
        PATIENCE_ITER += 1
        print(f"we have seen loss increase for {PATIENCE_ITER} steps and validation loss is {val_loss}, and previous best validation loss is {prev_val_loss}")
    return prev_val_loss , prev_f1


def validate(val_dataloader , model , PREFormer , criterion, Metric , name="val"):
def validate(val_dataloader , model , PREFormer , criterion, Metric , name="val"):
    total_loss_val = 0
    with torch.no_grad():
        for val_input, val_label in val_dataloader:
            val_batch_loss , _ = get_statistics(val_input , val_label , model , PREFormer , criterion , Metric , name , epoch = None )
            if criterion is not None:
                total_loss_val += val_batch_loss.item()

    weightedF1 = log(Metric , total_loss_val/len(val_dataloader) if criterion is not None else 0 , name)
    return total_loss_val/len(val_dataloader) , weightedF1


def one_epoch(epoch , train_dataloader , val_dataloader , model , PREFormer , criterion , optimizer , scheduler, clip , epoch_switch , patience , Metric , prev_val_loss , prev_f1):
    total_loss_train = 0   
    iters1 = len(train_dataloader[0])
    iters2 = len(train_dataloader[1])
    log_val1 = 1998 // train_dataloader[0].batch_size
    log_val2 = 1998
    # path = "/home/prsood/projects/def-whkchun/prsood/TAV_Train" Supposed to be jsut one
    path = '/'.join(os.getcwd().split('/')[:-3]) + "/TAV_Train" # this tav train lives one folder above multi-modal-emotisons
    if epoch % epoch_switch == 0:
        print(f"We have log val {log_val1} for multi-nomial dataloader" , flush = True)
        prev_val_loss , prev_f1 = not_grad_accum(epoch , train_dataloader[0] , val_dataloader , model , PREFormer , criterion , optimizer , scheduler, clip , patience , Metric , prev_val_loss , prev_f1 , total_loss_train , iters1 , log_val1 , path)
    else:
        print(f"We have log val {log_val2} for iterative dataloader" , flush = True)
        prev_val_loss , prev_f1 = grad_accum(epoch , train_dataloader[1] , val_dataloader , model , PREFormer , criterion , optimizer , scheduler, clip , patience , Metric , prev_val_loss , prev_f1 , total_loss_train , iters2 , log_val2 , path)
    model , PREFormer , optimizer , criterion = load_model(model , PREFormer , optimizer , criterion, path)
    return model , PREFormer , optimizer , criterion , scheduler , prev_val_loss , prev_f1

   
def train_tav_network(model , PREFormer , train_dataloader, val_dataloader, criterion,learning_rate, epochs , weight_decay , T_max ,Metric, patience , clip , epoch_switch , checkpoint = None):
    # optimizer = AdamW([ param for param in model.parameters() if param.requires_grad == True] + [ param for param in PREFormer.parameters() if param.requires_grad == True], lr= learning_rate, weight_decay=weight_decay)
    optimizer = AdamW([ param for param in model.parameters() if param.requires_grad == True], lr= learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer , T_0=T_max)  # To prevent fitting to local minima != global minima
    prev_val_loss = 100
    prev_f1 = 0
    
    if checkpoint is not None:
        # epoch_num = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    for epoch_num in tqdm(range(epochs), desc="epochs"):
        wandb.log({"epoch": epoch_num,"learning_rate": scheduler.get_last_lr()[0]})
        optimizer.zero_grad()  # Zero out gradients before each epoch.
        
        model , PREFormer , optimizer , criterion , scheduler , prev_val_loss , prev_f1 = one_epoch(epoch_num , train_dataloader, val_dataloader ,  model , PREFormer , criterion , optimizer, scheduler , clip , epoch_switch , patience , Metric , prev_val_loss , prev_f1 ) 
                
        if PATIENCE_ITER == patience or F1_ITER == patience:
            return model , PREFormer
    return model , PREFormer

def evaluate_tav(model , PREFormer , test_dataloader , Metric):
    validate(test_dataloader  , model , PREFormer , None , Metric , name = "test")


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
    return F1Weighted

