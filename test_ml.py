import pytest
import numpy as np 
from sklearn.ensemble import RandomForestClassifier 
from ml.model import train_model, compute_model_metrics, inference, predict
from train_model import X_train, y_train, X_test, y_test
from ml.data import apply_label


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
    If any ML functions return the expected type of result.
    """
    final_result = predict(X_test)
    assert isinstance(final_result, np.ndarray), (
        f" Expected np.ndarray, but returned {type(result).__name__}"
    )
    


def test_expected_size():
    """
    If the training and test datasets have the expected size.
    """
    expected = {
        "X_train" : (1000, 20),
        "y_train" : (1000, ),
        "X_test" : (200, 20),
        "y_test" : (200,),

        for i, expected in expected.items():
            data = locals()[i]
        assert data.shape == expected, (
            f"{i} should have shape {expected}, but returned {data.shape}."
        )
        
    }
