# tests/test_data_loader.py
import sys
import os

# Добавляем родительскую директорию (где лежит v2_utils.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import tempfile
import pandas as pd
import numpy as np

from dataset_utils import CSVDataLoader


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


    def test_stride_via_init(self, create_test_csv_files):
        """Тест stride через параметр __init__ (только с window)"""
        _, anomaly_dir, normal_dir = create_test_csv_files
        
        loader1 = CSVDataLoader(
            data_dir=anomaly_dir,
            normal_dir=normal_dir,
            stride=1
        )
        
        loader3 = CSVDataLoader(
            data_dir=anomaly_dir,
            normal_dir=normal_dir,
            stride=3
        )
        
        # Без window stride не работает
        X1, y1 = loader1.get_full_data(window=10)
        X3, y3 = loader3.get_full_data(window=10)
        
        # С stride=3 данных должно быть меньше
        assert len(y3) < len(y1)
        assert abs(len(y3) - len(y1) // 3) <= 2
    
    def test_stride_via_get_full_data(self, create_test_csv_files):
        """Тест stride через параметр get_full_data (только с window)"""
        _, anomaly_dir, normal_dir = create_test_csv_files
        
        loader = CSVDataLoader(
            data_dir=anomaly_dir,
            normal_dir=normal_dir
        )
        
        X1, y1 = loader.get_full_data(window=10, stride=1)
        X2, y2 = loader.get_full_data(window=10, stride=2)
        
        # С stride=2 данных должно быть меньше
        assert len(y2) < len(y1)
        assert abs(len(y2) - len(y1) // 2) <= 1

    def test_window_statistics(self, create_test_csv_files):
        """Тест: проверка корректности вычисления статистик для окон"""
        _, anomaly_dir, normal_dir = create_test_csv_files
        
        loader = CSVDataLoader(
            data_dir=anomaly_dir,
            normal_dir=normal_dir
        )
        
        window_size = 10
        
        windows, y_windows = loader.get_full_data(window=window_size, extract_stats=False)
        features, y_stats = loader.get_full_data(window=window_size, extract_stats=True)
        
        assert windows.shape[0] == features.shape[0]
        assert features.shape[1] == windows.shape[2] * 23  # n_features * n_stats
        
        # Проверка mean для первого окна и первой фичи
        manual_mean = np.mean(windows[0, :, 0])
        computed_mean = features[0, 0]  # первый признак первой фичи
        assert np.allclose(manual_mean, computed_mean)

        manual_std = np.std(windows[0, :, 0])  # первое окно, все временные точки, первая фича
        computed_std = features[0, 2 * windows.shape[2]]
        assert np.allclose(manual_std, computed_std)
        # # Проверка std для первого окна и первой фичи
        # manual_std = np.std(windows[0, :, 0])
        # computed_std = features[0, 1]  # второй признак первой фичи
        # assert np.allclose(manual_std, computed_std)
