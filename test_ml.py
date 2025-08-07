import pytest
import numpy as np 
from sklearn.ensemble import RandomForestClassifier 
from ml.model import train_model, inference
from train_model import X_train, y_train, X_test, y_test




def test_model():
    """
    Test that model is a instance of Random Forest Classifier
    """
    m = train_model(X_train, y_train)
    assert isinstance(m,RandomForestClassifier), (
        f"Expected RandomForestClassifier, but returned {type(m). __name__}"
    )
    
    

def test_expected_type():
    """
    Test if the ML Function returns predictions with the same shape as the input lables.
    """
    X = np.random.rand(100,5)
    y = np.random.randint(2, size=100)

    model = train_model(X,y)

    y_preds = inference(model,X)
    
    
    assert y.shape == y_preds.shape, (
        f" Expected  to be {y.shape}, but returned {y_preds.shape}"
    )
    


def test_consistent_feature():
    """
    Ensure that the training and test datasets have the same number of columns.
    """
    train_col = X_train.shape[1]
    test_col = X_test.shape[1]

    assert train_col == test_col, (
        f" Feature is not matching: X_train has {train_col} features, but X_test has {test_col}."
    )
        
    
