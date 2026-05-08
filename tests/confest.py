=# tests/conftest.py
import pytest
import numpy as np
import pandas as pd

@pytest.fixture(autouse=True)
def set_random_seed():
    """Фиксирует random seed для всех тестов"""
    np.random.seed(42)
    return

@pytest.fixture
def sample_time_series():
    """Общий фикстура для тестов временных рядов"""
    length = 100
    anomaly_start = 40
    anomaly_end = 60
    
    data = pd.DataFrame({
        'feature1': np.random.randn(length),
        'feature2': np.random.randn(length),
    })
    labels = pd.Series(np.zeros(length, dtype=bool))
    labels[anomaly_start:anomaly_end] = True
    
    return data, labels

@pytest.fixture
def sample_scores_and_gt():
    """Фикстура с оценками и ground truth"""
    ground_truth = np.array([0,0,0,0,0,1,1,1,1,1,0,0,0])
    scores = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.4,0.3,0.2])
    return scores, ground_truth
