# tests/test_data_loader.py
import sys
import os

# Добавляем родительскую директорию (где лежит v2_utils.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import tempfile
import pandas as pd
import numpy as np

from v2_utils import CSVDataLoader


@pytest.fixture
def create_test_csv_files():
    """Создает временные CSV файлы для тестирования"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Создаем директории
        anomaly_dir = os.path.join(tmpdir, 'anomaly')
        normal_dir = os.path.join(tmpdir, 'normal')
        os.makedirs(anomaly_dir)
        os.makedirs(normal_dir)
        
        # Создаем файл с аномалиями
        df_anomaly = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'y': np.concatenate([np.zeros(80), np.ones(20)])
        })
        anomaly_path = os.path.join(anomaly_dir, 'anomaly_1.csv')
        df_anomaly.to_csv(anomaly_path)
        
        # Создаем нормальный файл
        df_normal = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'y': np.zeros(100)
        })
        normal_path = os.path.join(normal_dir, 'normal_1.csv')
        df_normal.to_csv(normal_path)
        
        yield tmpdir, anomaly_dir, normal_dir


class TestCSVDataLoader:
    
    def test_initialization(self, create_test_csv_files):
        """Тест инициализации загрузчика"""
        _, anomaly_dir, normal_dir = create_test_csv_files
        
        loader = CSVDataLoader(
            data_dir=anomaly_dir,
            normal_dir=normal_dir,
            file_fraction=1.0,
            shuffle=False
        )
        
        assert len(loader.files) == 1
        assert len(loader.normal_files) == 1
    
    def test_get_full_data_returns_three_values(self, create_test_csv_files):
        """Тест: get_full_data возвращает 3 значения"""
        _, anomaly_dir, normal_dir = create_test_csv_files
        
        loader = CSVDataLoader(
            data_dir=anomaly_dir,
            normal_dir=normal_dir
        )
        
        result = loader.get_full_data(get_simulation_ids=True)
        assert len(result) == 3
        X, y, sim_ids = result
        
        assert X is not None
        assert y is not None
        assert sim_ids is not None
    
    def test_simulation_ids_are_unique_per_file(self, create_test_csv_files):
        """Тест: simulation_id уникален для каждого файла"""
        _, anomaly_dir, normal_dir = create_test_csv_files
        
        loader = CSVDataLoader(
            data_dir=anomaly_dir,
            normal_dir=normal_dir
        )
        
        X, y, sim_ids = loader.get_full_data(get_simulation_ids=True)
        
        # Должно быть 2 уникальных sim_id (anomaly + normal)
        assert sim_ids.nunique() == 2
    
    def test_simulation_ids_length_matches(self, create_test_csv_files):
        """Тест: длина sim_ids совпадает с длиной данных"""
        _, anomaly_dir, normal_dir = create_test_csv_files
        
        loader = CSVDataLoader(
            data_dir=anomaly_dir,
            normal_dir=normal_dir
        )
        
        X, y, sim_ids = loader.get_full_data(get_simulation_ids=True)
        
        assert len(sim_ids) == len(y)
        assert len(sim_ids) == len(X) if hasattr(X, '__len__') else X.shape[0]
    
    def test_max_samples_limit(self, create_test_csv_files):
        """Тест: ограничение max_samples"""
        _, anomaly_dir, normal_dir = create_test_csv_files
        
        loader = CSVDataLoader(
            data_dir=anomaly_dir,
            normal_dir=normal_dir,
            max_samples=50
        )
        
        X, y, sim_ids = loader.get_full_data(get_simulation_ids=True)
        
        assert len(y) <= 50
        assert len(sim_ids) <= 50