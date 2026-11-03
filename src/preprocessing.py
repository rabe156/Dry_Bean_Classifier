import pandas as pd 
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.model_selection import train_test_split

def Clean(X,y,TargetC):
    dataset=pd.concat([X,y] , axis=1)
    dataset=dataset.drop_duplicates()
    y=dataset[TargetC]
    X=dataset.drop(TargetC , axis=1)
    return X,y

def Encode(y):
    LE=LabelEncoder()
    y=LE.fit_transform(y)
    return  y,LE

def Split(X,y):
    Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,y,train_size=0.7,random_state=42 ,stratify = y )
    Xval,Xtest,Yval,Ytest=train_test_split(Xtest,Ytest,train_size=0.5,random_state=42,stratify=Ytest)
    return Xtrain ,Ytrain ,Xval,Yval,Xtest,Ytest

def Stand(Xtrain,Xval,Xtest):
    Scaler=StandardScaler()
    Xtrain=Scaler.fit_transform(Xtrain)
    print("Training means:")
    print(Xtrain.mean(axis=0))
    print("Training standard deviations:")
    print(Xtrain.std(axis=0))
    Xval=Scaler.transform(Xval)
    Xtest=Scaler.transform(Xtest)
    return Xtrain , Xval ,Xtest,Scaler

def PrepareData(X,y,TargetC) : 
    X_clean, y_clean = Clean(X, y,TargetC)
    y_encoded, target_encoder = Encode(y_clean)
    X_train,y_train, X_val, y_val,X_test,  y_test = Split(X_clean, y_encoded)
    X_train_scaled, X_val_scaled, X_test_scaled,Scaler = Stand(X_train, X_val, X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled , y_train , y_val ,y_test ,target_encoder,Scaler


if __name__ == "__main__":
    print("Running preprocessing module directly for testing...")