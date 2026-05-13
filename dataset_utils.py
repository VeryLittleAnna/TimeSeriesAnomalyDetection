import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split
import seaborn as sns
from torch.utils.data import Subset, Dataset
from typing import List, Tuple, Optional, Dict, Any, Union



class ChunkedWindowDataset(Dataset):
    def __init__(self, chunks_dir, preload_to_ram=False):
        # self.chunk_files = sorted(glob.glob(f"{chunks_dir}/chunk_w*.npz"))
        self.chunk_files = []
        assert os.path.exists(chunks_dir), f"Path {chunks_dir} does not exist"
        for filename in os.listdir(chunks_dir):
            if filename.endswith('.npz') and os.path.isfile(os.path.join(chunks_dir, filename)):
                self.chunk_files.append(os.path.join(chunks_dir, filename))
        self.chunk_sizes = []
        self.cumsum = [0]
        
        for cf in self.chunk_files:
            data = np.load(cf)
            size = data['X'].shape[0]
            self.chunk_sizes.append(size)
            self.cumsum.append(self.cumsum[-1] + size)
        
        self.total_size = self.cumsum[-1]

        if preload_to_ram:
            print("Preloading all chunks to RAM...")
            self.data = []
            for cf in self.chunk_files:
                data = np.load(cf)
                self.data.append(data['X'].astype(np.float32))
            self.chunk_files = None
            print(self.data[0].shape)
        else:
            self.data = None
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        if self.data is not None:
            chunk_idx = np.searchsorted(self.cumsum, idx, side='right') - 1
            offset = idx - self.cumsum[chunk_idx]
            X = self.data[chunk_idx][offset]
        else:
            chunk_idx = np.searchsorted(self.cumsum, idx, side='right') - 1
            offset = idx - self.cumsum[chunk_idx]
            data = np.load(self.chunk_files[chunk_idx])
            X = data['X'][offset].astype(np.float32)
        return torch.FloatTensor(X)

def create_dataloaders(dir_path, batch_size=32, preload_to_ram=False):
    dataset = ChunkedWindowDataset(dir_path, preload_to_ram=preload_to_ram)

    total_size = len(dataset)
    indices = list(range(total_size))
    # np.random.seed(42)
    # np.random.shuffle(indices)
    # не смешиваем симуляции np.random.shuffle(indices)
    train_size = int(0.7 * total_size)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    print(f"Train: {len(train_dataset)}, test: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader


class CSVDataLoader:
    def __init__(self,
            data_dir: str,
            normal_dir: Optional[str] = None, 
            file_fraction: float = 1.0, # будет влиять на долю симуляций с аномалиями
            shuffle: bool = False, 
            random_seed: int = 42,
            verbose: bool = False,
            max_samples: Optional[int] = None,
            anomaly_ratio = 0.07, # доля аномальных cэмплов (если влезает по max_samples)
            normalize: bool = False,
            scaler: Any = None,
            stride: int = 1,
            extract_stats: bool = False
        ):
        self.data_dir = data_dir
        self.normal_dir = normal_dir
        self.file_fraction = file_fraction
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.max_samples = max_samples
        self.anomaly_ratio = anomaly_ratio
        self.normalize = normalize
        self.scaler = scaler
        self.stride = stride
        self.extract_stats = extract_stats
        self.scaler_fitted = False  # флаг, что scaler уже обучен
        
        self.files = self._load_file_paths(data_dir)
        
        self.normal_files = []
        
        if not self.files and not self.normal_files:
            raise ValueError(f"No CSV files found in {data_dir} and {normal_dir}")
        # anomaly files
        self.files = self._apply_file_fraction(self.files)
        self.sizes = [sum(1 for line in open(file, "r")) - 1 for file in self.files]
        
        # files with normal data
        if normal_dir:
            self.normal_files = self._load_file_paths(normal_dir)
        self.normal_sizes = [sum(1 for line in open(file, "r")) - 1 for file in self.normal_files]
        verbose and print(f"0: {sum(self.normal_sizes)=}")
        if max_samples is not None and self.normal_files:
            current_total = sum(self.normal_sizes) + sum(self.sizes)
            if current_total > max_samples:
                target_normal = max(0, max_samples - sum(self.sizes))
                ratio = min(1.0, target_normal / sum(self.normal_sizes))
                self.normal_files = self._apply_file_fraction(self.normal_files, ratio)
                self.normal_sizes = [sum(1 for line in open(file, "r")) - 1 for file in self.normal_files]
        verbose and print(f"1: {sum(self.normal_sizes)=}")
        if anomaly_ratio < 1.0 and self.normal_files:
            target_normal = np.ceil(sum(self.sizes) * (1 - anomaly_ratio) / anomaly_ratio) # in samples
            ratio = min(1.0, target_normal / sum(self.normal_sizes))
            self.normal_files = self._apply_file_fraction(self.normal_files, min(1, ratio))
            self.normal_sizes = [sum(1 for line in open(file, "r")) - 1 for file in self.normal_files]
        verbose and print(f"2: {sum(self.normal_sizes)=}")
        # self.normal_files = self._apply_file_fraction(self.normal_files)
        # verbose and print(len(self.files), "files:", *self.files)
        # verbose and print(len(self.normal_files), "Normal files:", *self.normal_files)
        
        verbose and print("files sizes:", sum(self.sizes))
        verbose and print("normal sizes:", sum(self.normal_sizes))

    def _normalize_data(self, X):
        if not self.normalize:
            return X
        
        # в 2D если нужно
        original_shape = X.shape
        if len(original_shape) > 2:
            X_reshaped = X.reshape(-1, X.shape[-1])
        else:
            X_reshaped = X.values if hasattr(X, 'values') else X
        
        if self.scaler is not None:
            if not self.scaler_fitted:
                X_norm = self.scaler.fit_transform(X_reshaped)
                self.scaler_fitted = True
            else:
                X_norm = self.scaler.transform(X_reshaped)
        else:
            self.scaler = StandardScaler()
            X_norm = self.scaler.fit_transform(X_reshaped)
            self.scaler_fitted = True
            
        # в исходную форму
        if len(original_shape) > 2:
            X_norm = X_norm.reshape(original_shape)
        elif hasattr(X, 'iloc'):
            X_norm = pd.DataFrame(X_norm, columns=X.columns, index=X.index)
        
        return X_norm
    
    def _load_file_paths(self, directory: str) -> List[str]:
        files = []
        for filename in os.listdir(directory):
            if filename.endswith('.csv') and os.path.isfile(os.path.join(directory, filename)):
                files.append(os.path.join(directory, filename))
        return files
    
    def _apply_file_fraction(self, files: List[str], fraction: Optional[int] = None) -> List[str]:
        if not files:
            return []
        if fraction is None:
            fraction = self.file_fraction
        n_files = max(1, int(np.ceil(len(files) * fraction)))
        if self.shuffle:
            np.random.seed(self.random_seed)
            indices = np.random.permutation(len(files))
            selected_files = [files[i] for i in indices[:n_files]]
        else:
            selected_files = files[:n_files]
        return selected_files
    
    def __getitem__(self, index: int):
        main_total = sum(self.sizes)
        
        if index < main_total:
            cumsizes = np.cumsum(self.sizes)
            file_ind = np.searchsorted(cumsizes, index, side="right")
            ind_in_file = (index if file_ind == 0 else index - cumsizes[file_ind - 1])
            x, y = self.load_single_file(self.files[file_ind], ind_in_file)
        else:
            normal_index = index - main_total
            cumsizes = np.cumsum(self.normal_sizes)
            file_ind = np.searchsorted(cumsizes, normal_index, side="right")
            ind_in_file = (normal_index if file_ind == 0 else normal_index - cumsizes[file_ind - 1])
            x, y = self.load_single_file(self.normal_files[file_ind], ind_in_file, is_normal=True)
        
        return x, y
    
    def __len__(self):
        return sum(self.sizes) + sum(self.normal_sizes)
    
    def load_single_file(self, file_path: str, index: Optional[int] = None, is_normal: bool = False) -> Tuple:
        df = pd.read_csv(file_path, index_col=0)
        cols = [x for x in df.columns if x[0] != "_" and x != "index" and x != "level_0" and x != 'Unnamed: 0']  # служебные
        df = df[cols]
        
        if 'y' not in df.columns:
            raise ValueError(f"Column 'y' not found in {file_path}")
        
        X = df.drop('y', axis=1)
        y = df['y']

        if self.normalize:
            X = self._normalize_data(X)
        
        if is_normal:
            y = pd.Series(np.zeros(len(y), dtype=bool), index=y.index)
        
        if index is not None:
            return X.iloc[index], y.iloc[index]
        else:
            return X, y
    
    def __iter__(self):
        self.current_file_index = 0
        self.current_row_index = 0
        self.current_file_data = None
        self.in_normal_files = False
        self._load_current_file()
        return self
    
    def _load_current_file(self):
        if not self.in_normal_files:
            if self.current_file_index < len(self.files):
                file_path = self.files[self.current_file_index]
                self.current_file_data = self.load_single_file(file_path)
                self.current_file_rows = len(self.current_file_data[0])
                self.current_row_index = 0
            else:
                self.in_normal_files = True
                self.current_file_index = 0
                self._load_current_file()
        else:
            if self.current_file_index < len(self.normal_files):
                file_path = self.normal_files[self.current_file_index]
                self.current_file_data = self.load_single_file(file_path, is_normal=True)
                self.current_file_rows = len(self.current_file_data[0])
                self.current_row_index = 0
            else:
                self.current_file_data = None
    
    def __next__(self):
        if self.current_file_data is None:
            raise StopIteration
        
        X, y = self.current_file_data
        x_row = X.iloc[self.current_row_index]
        y_row = y.iloc[self.current_row_index]
        self.current_row_index += 1
        
        if self.current_row_index >= self.current_file_rows:
            self.current_file_index += 1
            self._load_current_file()
        
        return x_row, y_row

    def _extract_window_stats_vectorized(self, windows):
        """
        Векторизованная версия: windows shape (n_windows, window_size, n_features)
        """
        n_windows, window_size, n_features = windows.shape
        
        means = np.mean(windows, axis=1)
        stds = np.std(windows, axis=1)
        mins = np.min(windows, axis=1)
        maxs = np.max(windows, axis=1)
        medians = np.median(windows, axis=1)
        
        skews = skew(windows, axis=1)
        kurtoses = kurtosis(windows, axis=1, fisher=True)
        
        q25 = np.percentile(windows, 25, axis=1)
        q50 = np.percentile(windows, 50, axis=1)
        q75 = np.percentile(windows, 75, axis=1)
        # trend
        x = np.arange(window_size).reshape(-1, 1)
        X_design = np.column_stack([x, np.ones(window_size)])  # (window_size, 2)
        windows_2d = windows.transpose(1, 0, 2).reshape(window_size, -1)  # (window_size, n_windows * n_features)
        coeffs_2d = np.linalg.lstsq(X_design, windows_2d, rcond=None)[0]  # (2, n_windows * n_features)
        slopes = coeffs_2d[0].reshape(n_windows, n_features)  # (n_windows, n_features)
        
        energy = np.sum(windows ** 2, axis=1)
        
        windows_centered = windows - means[:, np.newaxis, :]
        autocorr_num = np.sum(windows_centered[:, :-1, :] * windows_centered[:, 1:, :], axis=1)
        autocorr_den = np.sum(windows_centered ** 2, axis=1)
        autocorr = np.divide(autocorr_num, autocorr_den, out=np.zeros_like(autocorr_num), where=autocorr_den!=0)
        autocorr = np.nan_to_num(autocorr)
        
        mean_centered = windows - means[:, np.newaxis, :]
        crossings = np.sum((mean_centered[:, :-1, :] * mean_centered[:, 1:, :]) < 0, axis=1)
        
        first_vals = windows[:, 0, :]
        last_vals = windows[:, -1, :]
        
        # FFT first 5 harmonics
        fft_vals = np.abs(np.fft.fft(windows, axis=1))[:, :5, :]
        cv = np.divide(stds, means, out=np.zeros_like(stds), where=means!=0)
        rms = np.sqrt(np.mean(windows ** 2, axis=1))
        
        features = np.concatenate([
            means, mins, stds, maxs, medians,  # вот так правильно
            skews, kurtoses, q25, q50, q75,
            slopes, energy, autocorr, crossings,
            first_vals, last_vals,
            fft_vals[:, 0, :], fft_vals[:, 1, :], fft_vals[:, 2, :], 
            fft_vals[:, 3, :], fft_vals[:, 4, :],
            cv, rms
        ], axis=1)
        return features

    
    def _generate_windows(self, X, y, window_size, stride=None, extract_stats=None):
        if X.shape[0] < window_size:
            return [], []
        if stride is None:
            stride = self.stride
        if extract_stats is None:
            extract_stats = self.extract_stats
        y = y[window_size - 1:][::stride]
        windows = np.lib.stride_tricks.sliding_window_view(X, window_size, axis=0).transpose(0, 2, 1) # (n, w, f)
        # windows = np.lib.stride_tricks.sliding_window_view(X, window_size, axis=0)
        windows = windows[::stride]
        if extract_stats:
            windows = self._extract_window_stats_vectorized(windows)
        # windows = windows.transpose(0, 2, 1)
        assert(windows.shape[0] == y.shape[0])
        windows = np.nan_to_num(windows, nan=0)
        return windows, y

    def save_windows_chunked(self, data_gen, output_dir, window=None, stride=None, chunk_size=5000):
        """Сохраняет окна в несколько .npz файлов по chunk_size штук"""
        os.makedirs(output_dir, exist_ok=True)
        if stride is None:
            stride = self.stride
        
        chunk_idx = 0
        total_saved = 0
        X_chunk = []
        
        for file in data_gen.files:
            tmp_x, tmp_y = data_gen.load_single_file(file)
            if window is not None:
                tmp_x, _ = data_gen._generate_windows(tmp_x, tmp_y, window, stride=stride)
            
            X_chunk.append(tmp_x)
            
            if sum(len(x) for x in X_chunk) >= chunk_size:
                X_combined = np.concatenate(X_chunk, axis=0)[:chunk_size].astype(np.float32)
                if self.max_samples and total_saved + len(X_combined) > self.max_samples:
                    X_combined = X_combined[:-(self.max_samples - total_saved)]
                np.savez_compressed(f"{output_dir}/chunk_{chunk_idx:05d}.npz", X=X_combined)
                total_saved += len(X_combined)
                chunk_idx += 1
                X_chunk = []
            if self.max_samples and total_saved >= self.max_samples:
                break
        
        if X_chunk:
            X_combined = np.concatenate(X_chunk, axis=0).astype(np.float32)
            np.savez_compressed(f"{output_dir}/chunk_{chunk_idx:04d}.npz", X=X_combined)
        
        print(f"Saved {chunk_idx+1} chunks to {output_dir}, total: {total_saved}")
    
    def get_full_data(self, max_samples: Optional[int] = None,  window=None, get_simulation_ids=False, stride=None, extract_stats=None) -> Tuple[pd.DataFrame, pd.Series]:
        X_list, y_list, sim_id_list = [], [], []
        sim_id_cnt = 0

        if max_samples is None and self.max_samples is not None:
            max_samples = self.max_samples
        

        for file in self.files:
            tmp_x, tmp_y = self.load_single_file(file)
            if window is not None:
                tmp_x, tmp_y = self._generate_windows(tmp_x, tmp_y, window, stride=stride, extract_stats=extract_stats)
            X_list.append(tmp_x)
            y_list.append(tmp_y)
            sim_id_list.append(pd.Series([sim_id_cnt] * len(tmp_x)))
            sim_id_cnt += 1
        
        # Нормальные данные
        for file in self.normal_files:
            tmp_x, tmp_y = self.load_single_file(file, is_normal=True)
            if window is not None:
                tmp_x, tmp_y = self._generate_windows(tmp_x, tmp_y, window, stride=stride, extract_stats=extract_stats)
            X_list.append(tmp_x)
            y_list.append(tmp_y)
            sim_id_list.append(pd.Series([sim_id_cnt] * len(tmp_x)))
            sim_id_cnt += 1

        
        # Объединение
        if not isinstance(X_list[0], pd.DataFrame):
            X = np.concatenate(X_list)
        else:
            X = pd.concat(X_list, ignore_index=True)
        
        if y_list and hasattr(y_list[0], 'iloc'):
            y = pd.concat(y_list, ignore_index=True).astype(bool)
        else:
            y = np.concatenate(y_list).astype(bool)

        simulation_ids = pd.concat(sim_id_list, ignore_index=True)

        if self.normalize and not self.scaler_fitted:
            X = self._normalize_data(X)
        
        # Ограничение по количеству samples
        if max_samples is not None and max_samples < len(X):
            X = X.iloc[:max_samples] if hasattr(X, 'iloc') else X[:max_samples]
            y = y.iloc[:max_samples] if hasattr(y, 'iloc') else y[:max_samples]
            simulation_ids = simulation_ids.iloc[:max_samples]
        
        # print(f"Loaded {len(X)} samples from {len(self.files)} main files and {len(self.normal_files)} normal files")
        print(f"   Итого: X.shape={X.shape}, y.shape={y.shape}")
        print(f"   Доля аномалий: {y.mean():.4f}")
        if get_simulation_ids:
            return X, y, simulation_ids
        return X, y