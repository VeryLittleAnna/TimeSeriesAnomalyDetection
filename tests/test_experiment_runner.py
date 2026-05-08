import sys
import os

# Добавляем родительскую директорию (где лежит v2_utils.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd

from v2_utils import ExperimentRunner, DetectionEvaluator


class MockDataGenerator:
    """Мок-генератор данных для тестирования ExperimentRunner"""
    
    def __init__(self, X=None, y=None, sim_ids=None):
        self.X = X if X is not None else np.random.randn(100, 5)
        self.y = y if y is not None else np.random.randint(0, 2, 100)
        self.sim_ids = sim_ids if sim_ids is not None else np.zeros(100)
    
    def get_full_data(self, get_simulation_ids=False):
        if get_simulation_ids:
            return self.X, self.y, self.sim_ids
        return self.X, self.y


class MockDetector:
    """Мок-детектор аномалий"""
    
    def __init__(self, **kwargs):
        self.params = kwargs
    
    def fit(self, X):
        pass
    
    def predict_scores(self, X):
        return np.random.random(len(X))


class TestExperimentRunner:
    
    @pytest.fixture
    def mock_generator(self):
        X = np.random.randn(100, 3)
        y = np.array([0]*80 + [1]*20)
        sim_ids = np.array([0]*50 + [1]*50)
        return MockDataGenerator(X, y, sim_ids)
    
    @pytest.fixture
    def runner(self, mock_generator):
        return ExperimentRunner(mock_generator, evaluator=DetectionEvaluator)
    
    def test_register_detector(self, runner):
        """Тест регистрации детектора"""
        runner.register_detector('test_detector', MockDetector, {'param1': 42})
        
        assert 'test_detector' in runner.detectors
        assert runner.detectors['test_detector']['class'] == MockDetector
        assert runner.detectors['test_detector']['default_params'] == {'param1': 42}
    
    def test_register_detector_without_params(self, runner):
        """Тест регистрации детектора без параметров"""
        runner.register_detector('simple_detector', MockDetector)
        
        assert runner.detectors['simple_detector']['default_params'] == {}
    
    def test_run_single_experiment_basic(self, runner, mock_generator):
        """Тест базового запуска одного эксперимента"""
        runner.register_detector('mock', MockDetector)
        X, y, sim_ids = mock_generator.get_full_data(get_simulation_ids=True)
        
        results = runner.run_single_experiment(
            'mock', X, y, simulation_ids=sim_ids,
            detector_params={'test': 1}
        )
        
        assert len(results) == 1
        assert results[0]['detector'] == 'mock'
        assert 'roc_auc' in results[0]
        assert 'precision' in results[0]
        assert 'recall' in results[0]
    
    def test_run_single_experiment_with_default_params(self, runner, mock_generator):
        """Тест использования параметров по умолчанию"""
        runner.register_detector('mock', MockDetector, {'default_param': 100})
        X, y, sim_ids = mock_generator.get_full_data(get_simulation_ids=True)
        
        results = runner.run_single_experiment(
            'mock', X, y, simulation_ids=sim_ids,
            detector_params=None  # Должны использоваться default_params
        )
        
        assert results[0]['default_param'] == 100
    
    def test_run_single_experiment_nonexistent_detector(self, runner):
        """Тест: запуск с несуществующим детектором"""
        X = np.random.randn(50, 3)
        y = np.random.randint(0, 2, 50)
        
        with pytest.raises(KeyError):
            runner.run_single_experiment('unknown', X, y)
    
    def test_comprehensive_experiments(self, runner):
        """Тест комплексного эксперимента с разными параметрами"""
        runner.register_detector('mock1', MockDetector)
        runner.register_detector('mock2', MockDetector)
        
        model_params_list = {
            'mock1': [{'p1': 1}, {'p1': 2}],
            'mock2': [{'p2': 'a'}]
        }
        
        results_df = runner.run_comprehensive_experiments(
            model_params_list=model_params_list,
            train_size=0.7
        )
        
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == 3  # 2 + 1 комбинации параметров
        assert 'detector' in results_df.columns
        assert 'model_params' in results_df.columns
        assert 'roc_auc' in results_df.columns
    
    def test_get_best_models(self, runner):
        """Тест получения лучших моделей"""
        runner.register_detector('detector_a', MockDetector)
        runner.register_detector('detector_b', MockDetector)
        
        # Запускаем эксперименты
        model_params_list = {
            'detector_a': [{}],
            'detector_b': [{}]
        }
        
        runner.run_comprehensive_experiments(model_params_list)
        
        # Получаем топ-1 по roc_auc
        best = runner.get_best_models(metric='roc_auc', top_k=1)
        
        assert len(best) == 1
        assert best.iloc[0]['roc_auc'] == runner.results['roc_auc'].max()
    
    def test_results_stored_in_dataframe(self, runner):
        """Тест: результаты сохраняются в DataFrame"""
        runner.register_detector('test', MockDetector)
        
        model_params_list = {'test': [{'param': i} for i in range(3)]}
        runner.run_comprehensive_experiments(model_params_list)
        
        assert isinstance(runner.results, pd.DataFrame)
        assert len(runner.results) == 3
        assert all(col in runner.results.columns for col in 
                  ['detector', 'roc_auc', 'model_params', 'anomaly_ratio'])
    
    def test_empty_detectors(self, runner):
        """Тест: запуск без зарегистрированных детекторов"""
        model_params_list = {}
        
        results_df = runner.run_comprehensive_experiments(model_params_list)
        
        assert len(results_df) == 0
        assert isinstance(results_df, pd.DataFrame)