import torch
from tqdm import tqdm
import wandb
from utils.early_stopping import EarlyStopping
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers.optimization import AdamW

def get_statistics(input,label,model,criterion,total_loss,Metric, check = "train",location = "/home/prsood/projects/def-whkchun/prsood/multi-modal-emotion/Inference/imageTest.txt"):
    batch_loss = None
    label = label.type(torch.LongTensor).to(f"cuda")

    # print(f"inputs before model is \n input = {input} \n labels = {label}")
    output = model(input.to(f"cuda"))
   

    if criterion is not None:
        # print("before crit")
        batch_loss = criterion(output, label)
        total_loss += batch_loss.item()
        # print("after crit")

    Metric.update_metrics(torch.argmax(output , dim = 1) , label.long())
    if check == "test":
        y = torch.argmax(output , dim = 1)
        with open(location,'a') as f:
            for i in range(len(label)):
                f.write(f"label: {label[i]} \tpred: {y[i]} \n")
            f.write("\n")
    return batch_loss , total_loss 


def one_epoch(train_dataloader , model , criterion , optimizer, clip , Metric):
    total_loss_train = 0
   
    for train_input, train_label in tqdm(train_dataloader , desc="training"):
        
        train_batch_loss , total_loss_train = get_statistics(train_input , train_label , model , criterion , total_loss_train , Metric)
        
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        model.zero_grad()
        train_batch_loss.backward()  # Back propagate
        optimizer.step()  # Run optimization step
          
    
    return model , optimizer , train_batch_loss,total_loss_train/len(train_dataloader.dataset)


def validate(val_dataloader , model , criterion, Metric , name="val" , location = "val"):

    total_loss_val = 0
    with torch.no_grad():

        for val_input, val_label in tqdm(val_dataloader, desc="validate" if name == "val" else "testing"):

            val_batch_loss , total_loss_val = get_statistics(val_input , val_label , model , criterion , total_loss_val , Metric , name , location)
                     
  
    return val_batch_loss,total_loss_val/len(val_dataloader.dataset)

    
   


def img_train(model, train_dataloader, val_dataloader, criterion,learning_rate, epochs , weight_decay , T_max ,Metric, patience=None , clip=None):
    

    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")
    # if model.__class__.__name__ == "BertClassifier":
    #     from transformers.optimization import AdamW
    # else:

    optimizer = AdamW(model.parameters(), lr= learning_rate, weight_decay=weight_decay)
    
    earlystop = EarlyStopping("",model,patience,model_name=model.__class__.__name__)

    scheduler = CosineAnnealingLR(optimizer , T_max=T_max)  # To prevent fitting to local minima != global minima

    
    for epoch_num in tqdm(range(epochs), desc="epochs"):
        # model.train()

        # print("before zero grad")
        optimizer.zero_grad()  # Zero out gradients before each epoch.

        # this is where we start to train our model
        # print("before one epoch")
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

        # model.eval() # this is where we put the flag for evaluating on so we dont change anything by accident
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
    torch.save(model.state_dict(), "/home/prsood/projects/def-whkchun/prsood/multi-modal-emotion/Inference/Image.pt")     
    return model

def evaluate_img(model, test_dataloader , Metric, location = "/home/prsood/projects/def-whkchun/prsood/multi-modal-emotion/Inference/imageTest.txt"):
    # model.eval() # this is where we put the flag for evaluating on so we dont change anything by accident
    name = "test"
    validate(test_dataloader  , model , None , Metric , name, location)
    multiAcc , multiF1, multiRec, multiPrec , Acc, F1, Rec, Prec , ConfusionMatrix = Metric.compute_scores("test")
    d1 = {
            "test/total_acc_test": Acc,
            "test/precision": Prec,
            "test/recall": Rec,
            "test/f1-score": F1,
            "test/ConfusionMatrix": ConfusionMatrix,
            }
    wandb.log({**d1 , **multiF1, **multiRec, **multiPrec, **multiAcc})
    with open(location,'a') as f:
        f.write(f"\n in TEST \n Confusion Matrix = {ConfusionMatrix} \n")    
    print(f"\n in TEST \n Confusion Matrix = {ConfusionMatrix} \n") 