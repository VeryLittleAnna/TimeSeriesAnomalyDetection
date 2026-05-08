# tests/test_detectors.py

import sys
import os

# Добавляем родительскую директорию (где лежит v2_utils.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

from v2_utils import BaseAnomalyDetector


class TestBaseAnomalyDetector:
    
    @pytest.fixture
    def sample_data(self):
        X = np.random.randn(100, 5)
        return X
    
    def test_isolation_forest_fit_predict(self, sample_data):
        """Тест IsolationForest"""
        detector = BaseAnomalyDetector(IsolationForest, contamination=0.1, random_state=42)
        detector.fit(sample_data)
        scores = detector.predict_scores(sample_data)
        
        assert len(scores) == len(sample_data)
        assert scores.dtype == float
    
    def test_one_class_svm_fit_predict(self, sample_data):
        """Тест OneClassSVM"""
        detector = BaseAnomalyDetector(OneClassSVM, nu=0.1)
        detector.fit(sample_data)
        scores = detector.predict_scores(sample_data)
        
        assert len(scores) == len(sample_data)
    
    def test_lof_fit_predict(self, sample_data):
        """Тест LocalOutlierFactor"""
        detector = BaseAnomalyDetector(LocalOutlierFactor, contamination=0.1, novelty=True)
        detector.fit(sample_data)
        scores = detector.predict_scores(sample_data)
        
        assert len(scores) == len(sample_data)
    
    def test_predict_without_fit_raises_error(self, sample_data):
        """Тест: вызов predict без fit вызывает ошибку"""
        detector = BaseAnomalyDetector(IsolationForest)
        
        with pytest.raises(ValueError):
            detector.predict_scores(sample_data)
    
    def test_get_set_params(self):
        """Тест get_params и set_params"""
        detector = BaseAnomalyDetector(IsolationForest, contamination=0.1, random_state=42)
        
        params = detector.get_params()
        assert params['contamination'] == 0.1
        
        detector.set_params(contamination=0.2)
        assert detector.params['contamination'] == 0.2