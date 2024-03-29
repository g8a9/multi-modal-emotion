from transformers import logging
logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore") 
import torch
from tqdm import tqdm
import wandb
from utils.early_stopping import EarlyStopping
from utils.global_functions import save_model
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers.optimization import AdamW
import torch



def get_statistics(input , label , model , PREFormer , criterion , Metric , check="train" , epoch = None):
    batch_loss = None 
    device = "cuda"
    batch_size = len(label)
    text = input[0]
    text_input_ids = text["input_ids"]
    text_attention_mask = text["attention_mask"]

    audio_features = input[1]
    audio_input_ids = audio_features["audio_features"]
    audio_attention_mask = audio_features["attention_mask"]

    video_embeds = input[2]
    video_input_ids = video_embeds["visual_embeds"]
    video_attention_mask = video_embeds["attention_mask"]   
    
    tav, tav_embed , attention_mask = PREFormer(input_ids = None , audio_features = audio_input_ids.to(f"cpu") , video_embeds = video_input_ids.to(f"cpu") , text_mask = None , 
                                                audio_mask = audio_attention_mask.to(f"cpu") , visual_mask = video_attention_mask.to(f"cpu"), device = device, train = True if check == "train" else False)    
    #TODO:PREFORMER NEEDS VISUAL_MASK FOR THE RANDOMNESS PART
 
    
    output = model(input_ids = text_input_ids.to(device) , text_attention_mask = text_attention_mask.to(device) , hidden_states = tav.to(device), pos_embed = tav_embed.to(device),
                        attention_mask = attention_mask.to(device) , batch_size = batch_size , check = check)    


    label = label.type(torch.LongTensor).to(device)
    Metric.update_metrics(torch.argmax(output , dim = 1) , label.long())
    if criterion is not None:
        # batch_loss = criterion(output, label)
        batch_loss = criterion(output, label , epoch = epoch if epoch is not None else 1) # TODO: Turn this on with Sampler
    return batch_loss 


def one_epoch(epoch , train_dataloader , model , PREFormer , criterion , optimizer , scheduler, clip , Metric):
    total_loss_train = 0   
    iters = len(train_dataloader)
    for batch_idx , (train_input, train_label) in enumerate(tqdm(train_dataloader , desc="training")): # batch idx starts off at 0
        accum_iter, accum_sum = (1 , 1) if epoch % 2 == 0 else train_dataloader.dataset.retGradAccum(i = batch_idx)
        # accum_iter, accum_sum = (1 , 1) For REgular cross Entropy
        
        train_batch_loss = get_statistics(train_input , train_label , model , PREFormer , criterion , Metric, check="train" , epoch = epoch) / accum_iter 
        
        total_loss_train += train_batch_loss.item()

        # backward pass
        train_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_([ param for param in model.parameters() if param.requires_grad == True] + [ param for param in PREFormer.parameters() if param.requires_grad == True], clip)
        if ((batch_idx + 1) % accum_sum == 0) or (batch_idx + 1 == iters): 
            if epoch % 2 == 1:
                print(f"\n we are on batch index {batch_idx} and our accum_iter = {accum_iter} \n" , flush = True)
            optimizer.step()
            scheduler.step(epoch + batch_idx / iters)
            model.zero_grad()
            PREFormer.zero_grad()
            
        # torch.cuda.empty_cache()
          
    
    return model , PREFormer , optimizer , total_loss_train/iters

def validate(val_dataloader , model , PREFormer , criterion, Metric , name="val"):
    total_loss_val = 0
    with torch.no_grad():
        for val_input, val_label in tqdm(val_dataloader, desc="validate" if name == "val" else "testing"):
            val_batch_loss = get_statistics(val_input , val_label , model , PREFormer , criterion , Metric , name , epoch = None )
            if criterion is not None:
                total_loss_val += val_batch_loss.item()
    
    return total_loss_val/len(val_dataloader)


   
def train_tav_network(model , PREFormer , train_dataloader, val_dataloader, criterion,learning_rate, epochs , weight_decay , T_max ,Metric, patience=None , clip=None , checkpoint = None):
    optimizer = AdamW([ param for param in model.parameters() if param.requires_grad == True] + [ param for param in PREFormer.parameters() if param.requires_grad == True], lr= learning_rate, weight_decay=weight_decay)
    earlystop = EarlyStopping("",model,patience,model_name=model.__class__.__name__)
    scheduler = CosineAnnealingWarmRestarts(optimizer , T_0=T_max)  # To prevent fitting to local minima != global minima
    if checkpoint is not None:
        # epoch_num = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    for epoch_num in tqdm(range(epochs), desc="epochs"):
        optimizer.zero_grad()  # Zero out gradients before each epoch.
        model , PREFormer , optimizer ,train_loss = one_epoch(epoch_num , train_dataloader ,  model , PREFormer , criterion , optimizer, scheduler , clip , Metric ) 

        multiAcc , multiF1, multiRec, multiPrec , Acc, F1Macro, F1Weighted, Rec, Prec , _ = Metric.compute_scores("train")
        d1 = {
                "epoch": epoch_num,
                "learning_rate": scheduler.get_last_lr()[0],
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
        save_model(model , PREFormer , optimizer , criterion , scheduler , epoch_num)

        # scheduler.step() # this is for CosineAnnealing

        val_loss = validate(val_dataloader  , model , PREFormer , criterion , Metric)
        multiAcc , multiF1, multiRec, multiPrec , Acc, F1Macro, F1Weighted, Rec, Prec , _ = Metric.compute_scores("val")
        d1 = {
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
                print(f"Val loss has increased so we are going back to best state on epoch {epoch_num}" , flush = True)
                model = earlystop.best_state
    return model , PREFormer

def evaluate_tav(model , PREFormer , test_dataloader , Metric):
    validate(test_dataloader  , model , PREFormer , None , Metric , name = "test")
    multiAcc , multiF1, multiRec, multiPrec , Acc, F1Macro, F1Weighted, Rec, Prec , _ = Metric.compute_scores("test")
    
    d1 = {
            "test/acc": Acc,
            "test/precision": Prec,
            "test/recall": Rec,
            "test/weighted-f1-score": F1Weighted,
            "test/macro-f1-score": F1Macro,
            }
    
    wandb.log({**d1 , **multiF1, **multiRec, **multiPrec, **multiAcc})   
    print(f"\n in TEST \n Confusion Matrix = {_} \n", flush = True)
