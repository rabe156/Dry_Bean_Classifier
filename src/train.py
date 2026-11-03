import torch 
from torchmetrics.functional.classification import multiclass_f1_score
from torch import nn
from pathlib import Path

class EarlyStopping ():
    def __init__(self,patience=10,min_delta=0) :
        self.patience=patience
        self.min_delta=min_delta
        self.counter=0
        self.best_F1=None
        self.early_stop=False

    def __call__(self,val_F1):
        if self.best_F1==None :
            self.best_F1=val_F1
        elif val_F1 <= self.best_F1+self.min_delta :
            self.counter=self.counter+1
            if self.counter>=self.patience :
                self.early_stop=True
        else :
            self.best_F1 = val_F1
            self.counter=0



def Loss():
    return nn.CrossEntropyLoss()

def Optimizer(model,lr=0.001):
    return torch.optim.Adam(model.parameters() , lr=lr)

def train(model , TrainLoader , ValLoader , device,epochs=100) :
    loss=Loss().to(device)
    opt=Optimizer(model)
    early_stopping=EarlyStopping()
    History={
           "train_loss": [],
            "val_loss": [],
            "train_f1": [],
            "val_f1": []
    }
    
    for epoch in range(epochs) :
        model.train()
        TrainTotalLoss=0.0
        AllPredictions=[]
        Alltargets=[]
        AllValP=[]
        AllValT=[]
        for X,y in TrainLoader :
            X=X.to(device)
            y=y.to(device)
            opt.zero_grad()

            logits=model(X)

            trainY=torch.argmax(logits,dim=1)
            AllPredictions.append(trainY.detach().cpu())
            Alltargets.append(y.detach().cpu())

            TrainLoss=loss(logits,y)
            
            TrainLoss.backward()

            opt.step()

            TrainTotalLoss+=TrainLoss.item()*X.size(0)
           

        model.eval()
        ValTotalLoss=0.0
      
        with torch.inference_mode():
            for Xval,yval in ValLoader :

                Xval=Xval.to(device)
                yval=yval.to(device)    

                ValLogits=model(Xval)
                valY=torch.argmax(ValLogits,dim=1)
               
                
                AllValP.append(valY.detach().cpu())
                AllValT.append(yval.detach().cpu())

                ValLoss=loss(ValLogits , yval)
                

                ValTotalLoss+=ValLoss.item()*yval.size(0)


        EpochTrainLoss=TrainTotalLoss/len(TrainLoader.dataset)
        EpochTrainF1=multiclass_f1_score(torch.cat(AllPredictions),torch.cat(Alltargets),num_classes=7,average="macro")
        EpochValLoss=ValTotalLoss/len(ValLoader.dataset)
        EpochValF1=multiclass_f1_score(torch.cat(AllValP),torch.cat(AllValT),num_classes=7,average="macro")

        History["train_loss"].append(EpochTrainLoss)
        History["val_loss"].append(EpochValLoss)
        History["train_f1"].append(EpochTrainF1.item())
        History["val_f1"].append(EpochValF1.item())

        if (epoch+1) % 5 == 0 :
            print(f"epoch number{epoch+1} | Trainloss={EpochTrainLoss:.2f}  | valloss={EpochValLoss:.2f} | train f1 ={EpochTrainF1 : .2f} | val f1 ={EpochValF1 : .2f} ")

        early_stopping(EpochValF1.item())
        if early_stopping.counter==0 :
            checkpoint={
                "epoch" : epoch+1,
                "model_stat_dict" :model.state_dict(),
                "optimizer_stat_dict" :opt.state_dict(),
                "best_val_F1":EpochValF1.item(),
                "input_dim":model.last[0].in_features,
                "hidden_dim": [
                                layer.out_features
                                for layer in model.last
                                if isinstance(layer, nn.Linear)
                                ],
                "output_dim":model.classifier.out_features,
                "lr":opt.param_groups[0]["lr"],
                "batch_size": TrainLoader.batch_size,
                "Early_patience":early_stopping.patience
            }
            Path("models").mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint , "models/best_model.pth")
        if early_stopping.early_stop :
            print("early stopinch triggered on epoch" , epoch,"with a score of ",early_stopping.best_F1)
            break
    
    CPLoading=torch.load("models/best_model.pth" ,map_location=device, weights_only=True)
    model.load_state_dict(CPLoading["model_stat_dict"])
    return History,checkpoint 
