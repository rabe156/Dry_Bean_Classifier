import torch 
from torch import nn

class BeanClassifier(nn.Module) :
    def __init__(self,In,Hidden,Out) :
        super().__init__ ()
        layers=[]
        current=In
        for dim in Hidden :
            layers.append(nn.Linear(in_features=current , out_features=dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.2))
            current=dim

        self.last=nn.Sequential(*layers)
        self.classifier=nn.Linear(in_features=current , out_features=Out) 

    def forward(self , x):
        issingle=(x.ndim == 1)
        if issingle :
            x=x.unsqueeze(0)

        x=self.last(x)
        logits=self.classifier(x)

        if issingle :
            logits=logits.squeeze(0)

        return logits