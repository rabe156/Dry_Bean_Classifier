import torch 
from torchmetrics.functional.classification import multiclass_f1_score,multiclass_precision,multiclass_recall,multiclass_confusion_matrix
from torch import nn

def evaluate(model ,TestLoader,device ) :

    model.eval()
    TestTotalLoss=0.0
    Allpredictions=[]
    AllTrue=[]
    loss=nn.CrossEntropyLoss()
    with torch.inference_mode():
        for Xtest ,ytest in TestLoader :
            Xtest=Xtest.to(device)
            ytest=ytest.to(device)

            TestLogits=model(Xtest)
            TestY = torch.argmax(TestLogits, dim=1)

            testloss=loss(TestLogits , ytest)
            TestTotalLoss+=testloss.item()*ytest.size(0)
            
            Allpredictions.append(TestY.detach().cpu())
            AllTrue.append(ytest.detach().cpu())

        TestF1macro=multiclass_f1_score(torch.cat(Allpredictions) , torch.cat(AllTrue),num_classes=7,average="macro")
        TestLoss=TestTotalLoss/len(TestLoader.dataset)
        TestPrecison=multiclass_precision(torch.cat(Allpredictions) , torch.cat(AllTrue),num_classes=7,average=None)
        TotalRecall=multiclass_recall(torch.cat(Allpredictions) , torch.cat(AllTrue),num_classes=7,average=None)
        Testf1=multiclass_f1_score(torch.cat(Allpredictions) , torch.cat(AllTrue),num_classes=7,average=None)
        TestConfusion=multiclass_confusion_matrix(torch.cat(Allpredictions) , torch.cat(AllTrue),num_classes=7)

    return TestF1macro ,TestLoss,TestPrecison,TotalRecall,Testf1,TestConfusion