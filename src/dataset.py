import torch 
from torch.utils.data import Dataset , DataLoader

class BeanDataset(Dataset):
    def __init__(self,X,y) :
        self.X=torch.tensor(X,dtype=torch.float32)
        self.y=torch.tensor(y,dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self , idx):
        return self.X[idx] , self.y[idx]


def BeanLoader(data, Shuf=False ,batch_size=32):
    DataLoad=DataLoader(
        dataset=data ,
        batch_size=batch_size,
        shuffle=Shuf,
        num_workers=0,
        drop_last=False
    )
    return DataLoad