import torch 
from torch import nn
from .model import BeanClassifier
import joblib
import numpy as np
import pandas as pd

def predict(X,scaler,encoder,checkpoint,features,device) :
    Scaler=joblib.load(scaler)
    Encoder=joblib.load(encoder)
    Names=joblib.load(features)
    checkpoint=torch.load(checkpoint,map_location=device,weights_only=True)
    In=checkpoint["input_dim"]
    Hidden=checkpoint["hidden_dim"]
    Out=checkpoint["output_dim"]

    model=BeanClassifier(In,Hidden,Out)
    ModelDict=checkpoint["model_stat_dict"]
    model.load_state_dict(ModelDict)
    model=model.to(device)

    
    if isinstance(X, pd.DataFrame):
        
        if X.columns.has_duplicates:
            raise ValueError("Duplicate feature names detected")
        
        missing = set(Names) - set(X.columns)
        extra = set(X.columns) - set(Names)
        if missing or extra:
            raise ValueError(
                f"Missing features: {sorted(missing)}; "
                f"Unexpected features: {sorted(extra)}"
            )
        # Reorder features according to training
        X = X[Names]

    else:
        # Convert lists, arrays and tensors to NumPy
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()

        X = np.asarray(X, dtype=np.float32)

        # Handle one-dimensional input
        if X.ndim == 1:
            X = X.reshape(1, -1)

    # Validate shape for both input types
    if X.shape != (1, In):
        raise ValueError(
            f"Expected one observation with {In} features, "
            f"received shape {X.shape}"
        )
    
    Xscaled=Scaler.transform(X)
    Xscaled=torch.tensor(Xscaled ,dtype=torch.float32,device=device)
    

    model.eval()
    with torch.inference_mode() :
        logit=model(Xscaled)
        prob=torch.softmax(logit,dim=1)
        y=torch.argmax(logit, dim=1)

        prediction=Encoder.inverse_transform(y.cpu().numpy())

    return prediction[0] , y.item() , prob.squeeze(0).cpu().numpy()

