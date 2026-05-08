# tests/test_evaluators.py

import sys
import os

# Добавляем родительскую директорию (где лежит v2_utils.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd

from v2_utils import AdvancedDetectionEvaluator, DetectionEvaluator


class TestAdvancedDetectionEvaluator:
    
    @pytest.fixture
    def evaluator(self):
        return AdvancedDetectionEvaluator()
    
    def test_detection_latency_basic(self, evaluator):
        """Тест базовой латенси без simulation_id"""
        ground_truth = np.array([0,0,0,0,0,1,1,1,1,1,0,0,0])
        predictions = np.array([0,0,0,0,0,0,0,1,1,1,0,0,0])
        sim_ids = pd.Series(np.zeros_like(ground_truth))
        
        latency = evaluator.calculate_detection_latency(predictions, ground_truth, sim_ids)
        assert latency == 2.0
    
    def test_detection_latency_no_anomaly(self, evaluator):
        """Тест: нет аномалий"""
        ground_truth = np.zeros(100)
        predictions = np.zeros(100)
        sim_ids = pd.Series(np.zeros_like(ground_truth))
        
        latency = evaluator.calculate_detection_latency(predictions, ground_truth, sim_ids)
        assert latency == float('inf')
    
    def test_detection_latency_immediate(self, evaluator):
        """Тест: мгновенное обнаружение"""
        ground_truth = np.array([0,0,1,1,1,0,0])
        predictions = np.array([0,0,1,1,1,0,0])
        sim_ids = pd.Series(np.zeros_like(ground_truth))
        
        latency = evaluator.calculate_detection_latency(predictions, ground_truth, sim_ids)
        assert latency == 0.0
    
    def test_detection_latency_with_sim_ids(self, evaluator):
        """Тест латенси с учетом simulation_id"""
        ground_truth = np.array([0,0,1,1,0,0,   0,0,1,1,0,0])
        predictions = np.array([0,0,0,1,0,0,   0,0,1,0,0,0])
        sim_id = np.array([0,0,0,0,0,0,   1,1,1,1,1,1])
        
        latency = evaluator.calculate_detection_latency(predictions, ground_truth, sim_id)
        # Симуляция 0: latency = 1, Симуляция 1: latency = 0, средняя = 0.5
        assert latency == 0.5
    
    def test_roc_auc_perfect(self, evaluator):
        """Тест идеального разделения"""
        scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])
        ground_truth = np.array([0, 0, 0, 1, 1, 1])
        
        results = evaluator.calculate_roc_analysis(scores, ground_truth)
        assert results['roc_auc'] == 1.0
    
    def test_roc_auc_random(self, evaluator):
        """Тест случайного классификатора"""
        np.random.seed(42)
        scores = np.random.random(1000)
        ground_truth = np.random.randint(0, 2, 1000)
        
        results = evaluator.calculate_roc_analysis(scores, ground_truth)
        # ROC-AUC должен быть около 0.5
        assert 0.45 < results['roc_auc'] < 0.55
    
    def test_calculate_metrics_with_sim_ids(self, evaluator):
        """Тест calculate_metrics с simulation_id"""
        scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95, 0.2, 0.3, 0.4])
        ground_truth = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0])
        sim_id = np.array([0,0,0,0,0,0, 1,1,1])
        
        results = evaluator.calculate_metrics(scores, ground_truth, simulation_id=sim_id)
        
        assert 'roc_auc' in results
        assert 'detection_latency_75' in results
        assert 'detection_latency_90' in results

    def test_survival_metrics_simple(self, evaluator):
        """Тест базовых метрик выживаемости: все аномалии обнаружены"""
        predictions = np.array([0,0,0,1,1,1, 0,0,0,1,1,1])
        ground_truth = np.array([0,0,1,1,1,1, 0,0,1,1,1,1])
        sim_id = np.array([0,0,0,0,0,0, 1,1,1,1,1,1])
        
        surv_metrics = evaluator.calculate_survival_metrics(predictions, ground_truth, sim_id)
        
        assert surv_metrics['mttd'] == 1.0  # среднее латенси = 1
        assert surv_metrics['rmst'] > 0
        assert surv_metrics['km_estimator'] is not None
    
    def test_survival_metrics_censored(self, evaluator):
        """Тест с цензурированием: аномалии не обнаружены"""
        predictions = np.zeros(20)
        ground_truth = np.array([0,0,1,1,1,1,1,0,0,0, 0,0,1,1,1,1,1,0,0,0])
        sim_id = np.array([0]*10 + [1]*10)
        
        surv_metrics = evaluator.calculate_survival_metrics(predictions, ground_truth, sim_id, max_time=5)
        
        assert surv_metrics['mttd'] == 5.0
        assert surv_metrics['rmst'] == 5.0
    
    def test_survival_metrics_no_anomalies(self, evaluator):
        """Тест: нет аномалий в ground_truth"""
        predictions = np.random.randint(0, 2, 100)
        ground_truth = np.zeros(100)
        
        surv_metrics = evaluator.calculate_survival_metrics(predictions, ground_truth, None)
        
        assert surv_metrics['km_estimator'] is None
        assert surv_metrics['mttd'] == float('inf')
        assert surv_metrics['rmst'] is None  # или 0, зависит от реализации
    
    def test_survival_metrics_single_segment(self, evaluator):
        """Тест: одна непрерывная аномалия"""
        predictions = np.array([0,0,0,1,0,0,0])
        ground_truth = np.array([0,1,1,1,1,1,0])
        
        surv_metrics = evaluator.calculate_survival_metrics(predictions, ground_truth, None)
        
        # Аномалия с индекса 1, обнаружение на индексе 3 -> latency = 2
        assert surv_metrics['mttd'] == 2.0
    
    def test_calculate_metrics_includes_survival(self, evaluator):
        """Тест: calculate_metrics корректно добавляет survival метрики"""
        scores = np.array([0.1, 0.2, 0.8, 0.85, 0.9, 0.95])
        ground_truth = np.array([0, 0, 1, 1, 1, 1])
        predictions = (scores > 0.75).astype(int)
        
        # Мокаем calculate_roc_analysis и calculate_detection_latency
        # Но проверим, что ключи есть
        results = evaluator.calculate_metrics(scores, ground_truth)
        
        assert 'survival_mttd' in results
        assert 'survival_rmst' in results
        assert 'detection_latency_75' in results
        assert 'detection_latency_90' in results


class TestDetectionEvaluator:
    
    @pytest.fixture
    def evaluator(self):
        return DetectionEvaluator(delay_window=5)
    
    def test_add_detection_delay(self, evaluator):
        """Тест добавления окна задержки"""
        ground_truth = np.array([0,0,1,0,0,0,0])
        delayed = evaluator.add_detection_delay(ground_truth)
        
        expected = np.array([0,0,1,1,1,1,1])
        # assert np.array_equal(delayed, expected)
        np.testing.assert_array_equal(delayed, expected, 
                                   err_msg=f"\n delayed: {delayed}\n expected: {expected}")
    
    def test_add_detection_delay_boundary(self, evaluator):
        """Тест на границе массива"""
        ground_truth = np.array([0,0,1,0,0])
        delayed = evaluator.add_detection_delay(ground_truth)
        # delay_window=5, но массив короче -> должно быть True до конца
        expected = np.array([0,0,1,1,1])
        assert np.array_equal(delayed, expected)
    
    def test_calculate_binary_metrics(self, evaluator):
        """Тест бинарных метрик"""
        predictions = np.array([1,0,1,0,1,0,1])
        ground_truth = np.array([1,1,0,0,1,1,0])
        
        metrics = evaluator.calculate_binary_metrics(predictions, ground_truth)
        
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'accuracy' in metrics
        
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
    
    def test_calculate_metrics(self, evaluator):
        """Тест calculate_metrics для ROC-AUC"""
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        ground_truth = np.array([0, 0, 1, 1])
        
        metrics = evaluator.calculate_metrics(scores, ground_truth)
        
        assert 'roc_auc' in metrics
        assert 0 <= metrics['roc_auc'] <= 1


# Параметризованные тесты
@pytest.mark.parametrize("threshold,expected_latency", [
    (0.5, 0.0),
    (0.75, 0.0),
    (0.9, 1.0),
])
def test_latency_different_thresholds(threshold, expected_latency):
    """Тест латенси для разных порогов"""
    evaluator = AdvancedDetectionEvaluator()
    scores = np.array([0.4, 0.6, 0.8, 0.95, 0.95, 0.95])
    ground_truth = np.array([0, 0, 1, 1, 1, 1])
    sim_ids = pd.Series(np.zeros_like(scores))
    
    predictions = (scores > threshold).astype(int)
    latency = evaluator.calculate_detection_latency(predictions, ground_truth, sim_ids)
    
    # assert latency == expected_latency
    np.testing.assert_array_equal(latency, expected_latency, 
                                   err_msg=f"\n latency: {latency}\n expected: {expected_latency}\n {ground_truth=}\n {predictions=}")