import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import os
import traceback
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve, auc

from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional, Dict, Any, Union

from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split, GroupShuffleSplit

import warnings
warnings.filterwarnings('ignore')


import torch

from lifelines import KaplanMeierFitter


N_DV = 20

# Визуализация
def plot_with_anomalies(
    df: pd.DataFrame,
    value_col: str,
    anomaly_col: Union[pd.Series, List[bool]],
    title: str = "Временной ряд с аномалиями",
    trunc: Optional[int] = None
) -> None:
    """
    Визуализация временного ряда с выделением аномальных участков
    """
    df_plot = df.copy()
    if trunc is not None:
        df_plot = df_plot.iloc[-trunc:].reset_index(drop=True)
        anomaly_series = anomaly_col.iloc[-trunc:].reset_index(drop=True) if hasattr(anomaly_col, 'iloc') else anomaly_col[-trunc:]
    else:
        anomaly_series = pd.Series(anomaly_col)
    fig, ax = plt.subplots(figsize=(15, 6))

    ax.plot(df_plot.index, df_plot[value_col], linewidth=1, color='blue', label=value_col)

    anomaly_regions = []
    in_anomaly = False
    start_idx = None

    for i, (idx, is_anomaly) in enumerate(zip(df_plot.index, anomaly_series)):
        if is_anomaly and not in_anomaly:
            in_anomaly = True
            start_idx = idx
        elif not is_anomaly and in_anomaly:
            in_anomaly = False
            anomaly_regions.append((start_idx, df_plot.index[i-1]))
            start_idx = None

    if in_anomaly and start_idx is not None:
        anomaly_regions.append((start_idx, df_plot.index[-1]))

    for i, (start, end) in enumerate(anomaly_regions):
        ax.axvspan(start, end, alpha=0.3, color='red',
                  label='Аномальные участки' if i == 0 else "")

    ax.set_title(title, fontsize=14)
    ax.set_ylabel(value_col, fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if title is not None:
        plt.title(title)
    plt.show()


class AdvancedDetectionEvaluator:
    def __init__(self, **kwargs):
        pass

    def _normalize_scores(self, scores, method="sigmoid"):
        if method == "sigmoid":
            return 1 / (1 + np.exp(-(scores - scores.mean()) / scores.std()))
        else:
            raise NotImplementedError(f"_normalize_scores in AdvancedDetectionEvaluator with {method=}")
    
    def calculate_binary_metrics(self, predictions, ground_truth, simulation_ids):
        """Расчет бинарных метрик с правильными формулами"""
        
        tn, fp, fn, tp = confusion_matrix(ground_truth, predictions).ravel()

        precision = precision_score(ground_truth, predictions, zero_division=0)
        recall = recall_score(ground_truth, predictions, zero_division=0)  # TDR = Recall
        f1 = f1_score(ground_truth, predictions, zero_division=0)
        accuracy = accuracy_score(ground_truth, predictions)
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        fdr = tp / (tp + fn) if (tp + fn) > 0 else 0  # False Detection Rate 

        detection_latency = self.calculate_detection_latency(predictions, ground_truth, simulation_ids)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'fdr': fdr,
            'far' : far,
            'detection_latency': detection_latency,
            # 'confusion_matrix': [tn, fp, fn, tp]
        }
    
    def calculate_detection_latency(self, predictions, ground_truth, simulation_ids):
        y = pd.Series(ground_truth).reset_index(drop=True)
        pred = pd.Series(predictions).reset_index(drop=True)
        
        if simulation_ids is not None:
            sim_id = pd.Series(simulation_ids).reset_index(drop=True)
            y_segments = (y != y.shift()).cumsum()
            segment_key = y_segments.astype(str) + "_" + sim_id.astype(str)
        else:
            segment_key = (y != y.shift()).cumsum()
        
        position_in_segment = y.groupby(segment_key).cumcount()
        anomaly_mask = (y == 1)
        
        if not anomaly_mask.any():
          return float('inf')
        anomaly_data = pd.DataFrame({
          'segment_key': segment_key[anomaly_mask].values,
          'position': position_in_segment[anomaly_mask],
          'prediction': pred[anomaly_mask]
        })
        # Находим первое обнаружение в каждом сегменте
        first_detections = anomaly_data[anomaly_data['prediction'] == 1].groupby('segment_key')['position'].first()
        
        return first_detections.mean() if len(first_detections) > 0 else float('inf')
    
    
    def calculate_roc_analysis(self, scores, ground_truth):
        fpr, tpr, thresholds = roc_curve(ground_truth, scores)
        roc_auc = auc(fpr, tpr)
        diff = (1 - tpr - fpr)
        diff = np.where(np.isnan(diff), np.inf, diff)
        eer_ind = np.nanargmin(np.absolute(diff))
        eer_threshold = thresholds[eer_ind]
        eer = (fpr[eer_ind] + (1 - tpr[eer_ind])) / 2
        
        precision, recall, pr_thresholds = precision_recall_curve(ground_truth, scores)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_thr = pr_thresholds[np.argmax(f1)]
        pr_auc = auc(recall, precision)
        
        return {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'optimal_thr': optimal_thr,
            'eer': eer,
        }
    
    def calculate_metrics(self, scores, ground_truth, optimal_thr=None, *args, simulation_ids=None, **kwargs):
        results = self.calculate_roc_analysis(scores, ground_truth)
        if optimal_thr is None:
            optimal_thr = results["optimal_thr"]
        else:
            results["optimal_thr"] = optimal_thr
        results.update(self.calculate_binary_metrics((scores > optimal_thr), ground_truth, simulation_ids))
        # results["detection_latency"] = self.calculate_detection_latency((scores > optimal_thr), ground_truth, simulation_ids)
        # results["detection_latency_90"] = self.calculate_detection_latency((scores > 0.90), ground_truth, simulation_ids)
        surv_metrics = self.calculate_survival_metrics((scores > optimal_thr), ground_truth, simulation_ids)
        results["survival_mttd"] = surv_metrics["mttd"]
        results["survival_km"] = surv_metrics["km_estimator"]
        return results

    def calculate_3d_roc_surface(self, scores, ground_truth, latency_windows=None):
        """Расчет 3D ROC поверхности (FPR, TPR, Latency)"""
        if latency_windows is None:
            latency_windows = [5, 10, 20, 50]  # Различные временные окна
        
        fpr_3d = []
        tpr_3d = []
        latency_3d = []
        
        thresholds = np.unique(scores)
        
        for window in latency_windows:
            adapted_gt = self.adapt_ground_truth_with_latency(ground_truth, window)
            
            for threshold in thresholds:
                predictions = (scores >= threshold).astype(int)
                
                fpr, tpr, _ = roc_curve(adapted_gt, predictions)
                if len(fpr) > 1 and len(tpr) > 1:
                    fpr_3d.append(fpr[1])
                    tpr_3d.append(tpr[1])
                    latency_3d.append(window)
        
        return {
            'fpr_3d': np.array(fpr_3d),
            'tpr_3d': np.array(tpr_3d),
            'latency_3d': np.array(latency_3d)
        }
    
    def adapt_ground_truth_with_latency(self, ground_truth, latency_window):
        """Адаптация ground truth с учетом допустимой задержки"""
        adapted_gt = ground_truth.copy()
        
        for i in range(len(ground_truth)):
            if ground_truth[i] == 1 and i + latency_window < len(ground_truth):
                adapted_gt[i:i+latency_window] = 1
                
        return adapted_gt

    def calculate_survival_metrics(self, predictions, ground_truth, simulation_ids=None, max_time=100, thr=None): #надо ли здесь тоже параметризовать
        """
        Оценивает функцию выживаемости S(t) и среднее время до обнаружения (MTTD)
        с учётом цензурированных аномалий.

        S(t)=P(T>t) - вероятность того, что аномалия остается необнаруженной дольше времени t
        Возвращает словарь с
        - 'km_estimator': обученный объект KaplanMeierFitter (содержит S(t))
        - 'mttd': среднее время до обнаружения (float)
        - 'survival_table': DataFrame с временами и вероятностями выживаемости
        - 'rmst' - площадь под кривой выживаемости
        """
        y = pd.Series(ground_truth)
        pred = pd.Series(predictions)

        if simulation_ids is not None:
            sim_id = pd.Series(simulation_ids)
            y_segments = (y != y.shift()).cumsum()
            segment_key = y_segments.astype(str) + "_" + sim_id.astype(str)
        else:
            segment_key = (y != y.shift()).cumsum()


        times = []
        events = []
        for key, group in y.groupby(segment_key):
            if group.iloc[0] == 0:
                continue

            segment_length = len(group)
            pred_seg = pred.loc[group.index]

            detection_positions = np.where(pred_seg == 1)[0]
            if len(detection_positions) > 0:
                time = detection_positions[0] + 1
                event = 1
            else:
                # Цензурирование: аномалия не обнаружена до конца сегмента
                time = segment_length
                event = 0
            times.append(time)
            events.append(event)

        if not times:
            return {'km_estimator': None, 'mttd': float('inf'), 'rmst': None}

        kmf = KaplanMeierFitter()
        kmf.fit(times, event_observed=events)

        # Вычисление MTTD как интеграла S(t) dt (сумма для дискретных времён)
        # max_time = max(times)
        t_grid = np.arange(1, max_time + 1)
        survival_probs = kmf.survival_function_at_times(t_grid).fillna(0).values
        mttd = np.sum(survival_probs)

        # Таблица выживаемости для возврата
        survival_table = kmf.survival_function_.reset_index()
        survival_table.columns = ['time', 'survival_probability']

        times = kmf.survival_function_.index.values
        surv = kmf.survival_function_.values.flatten()
        max_time = min(max_time, len(predictions))
        full_times = np.arange(1, max_time + 1)
        surv_interp = np.interp(full_times, times, surv, left=1, right=0)
        rmst = np.sum(surv_interp) # restricted_mean_survival_time - площадь под кривой выживаемости

        return {
            'km_estimator': kmf,
            'mttd': mttd,
            'rmst': rmst,
            # 'survival_table': survival_table
        }


class BaseAnomalyDetector:
    def __init__(self, model_class, **params):
        self.model_class = model_class
        self.params = params
        self.model = None
    
    def fit(self, X):
        self.model = self.model_class(**self.params)
        if hasattr(self.model, 'fit'):
            self.model.fit(X)
        else:
            raise NotImplementedError()
        return self
    
    def predict_scores(self, X):
        if self.model is None:
            raise ValueError()
        if isinstance(self.model, LocalOutlierFactor):
            if self.model.novelty:  # ← проверяем режим
                scores = -self.model.decision_function(X)
            else:
                scores = -self.model.negative_outlier_factor_
        elif hasattr(self.model, "decision_function"):
            scores = -self.model.decision_function(X)
        elif hasattr(self.model, "score_samples"): 
            scores = -self.model.score_samples(X)
        else:
            raise NotImplementedError()

        if hasattr(self.model, '_raw_scores_fitted'):
            min_score = self.model._min_score
            max_score = self.model._max_score
            normalized = (scores - min_score) / (max_score - min_score)
            normalized = np.clip(normalized, 0, 1)
        else:
            normalized = 1 / (1 + np.exp(-scores / np.std(scores)))
        
        return normalized
    
    def get_params(self):
        return self.params
    
    def set_params(self, **params):
        self.params.update(params)
        return self


class ExperimentRunner:    
    def __init__(self, data_generator, evaluator=AdvancedDetectionEvaluator):
        self.data_generator = data_generator
        self.detectors = {}
        self.results = []
        self.evaluator = evaluator
    
    def register_detector(self, name, model_class, default_params=None):
        if default_params is None:
            default_params = {}
        self.detectors[name] = {'class': model_class, 'default_params': default_params}
    
    
    def run_single_experiment(self, detector_name, X_train, X_test, y_train, y_test, sim_ids_train, sim_ids_test,
                            detector_params=None, delays=None, thresholds=None):
        if hasattr(y_train, 'values'):
            y_train = y_train.values
        if hasattr(y_test, 'values'):
            y_test = y_test.values
        # if delays is None:
        #     delays = [0, 1, 2, 5, 10]
        if detector_params is None:
            detector_params = self.detectors[detector_name]['default_params']
        
        detector = BaseAnomalyDetector(
            self.detectors[detector_name]['class'], 
            **detector_params
        )
        
        try:
            
            detector.model = detector.model_class(**detector_params)
            if detector_name == 'GradientBoosting':
                detector.model.fit(X_train, y_train)
            else:
                detector.model.fit(X_train)
            scores = detector.predict_scores(X_test)
            train_scores = detector.predict_scores(X_train)
            # print(f"{detector_name=}: range=[{scores.min():.3f}, {scores.max():.3f}], "
            #       f"mean={scores.mean():.3f}, std={scores.std():.3f}, "
            #       f"p50={np.median(scores):.3f}, p75={np.percentile(scores, 75):.3f}, p90={np.percentile(scores, 90):.3f}")            
            experiment_results = []
            evaluator = self.evaluator()
            scores = (pd.Series(scores.values if hasattr(scores, 'values') else scores)).reset_index(drop=True)
            train_scores = (pd.Series(train_scores.values if hasattr(train_scores, 'values') else train_scores)).reset_index(drop=True)

            train_metrics = evaluator.calculate_metrics(train_scores, y_train, apply_delay=False, simulation_ids=sim_ids_train)
            
            metrics = evaluator.calculate_metrics(scores, y_test, optimal_thr=train_metrics["optimal_thr"], apply_delay=False, simulation_ids=sim_ids_test)
            experiment_results.append({
                'detector': detector_name,
                # 'delay': delay,
                **metrics,
                **detector_params,
                'test_scores': np.array(scores.values),
                'test_target': np.array(y_test)
            })
            
            return experiment_results
            
        except Exception as e:
            print(f"Ошибка в эксперименте {detector_name}: {e}")
            traceback.print_exc()
            return []
    
    def run_comprehensive_experiments(self, model_params_list, window=30,
                                    train_size=0.7, test_delays=None, custom_thresholds=None, autoencoder=None, random_state=42):
        if test_delays is None:
            test_delays = [0, 1, 2, 5, 10, 15, 20]
        
        all_results = []

        dataset, target, simulation_ids = self.data_generator.get_full_data(get_simulation_ids=True, window=window)

        if autoencoder is not None:
            print(f"Применяем автоэнкодер к данным. Исходная форма: {dataset.shape}")
            dataset_tensor = torch.FloatTensor(dataset.values if hasattr(dataset, 'values') else dataset)
            dataset = autoencoder.encode(dataset_tensor).detach().cpu().numpy()
            print(f"После автоэнкодера: {dataset.shape}")

        gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(dataset, target, groups=simulation_ids))
        
        X_train, X_test = (dataset.iloc[train_idx], dataset.iloc[test_idx]) if hasattr(dataset, 'iloc') else (dataset[train_idx], dataset[test_idx])
        y_train, y_test = (target.iloc[train_idx], target.iloc[test_idx]) if hasattr(target, 'iloc') else (target[train_idx], target[test_idx])
        sim_ids_train, sim_ids_test = (simulation_ids.iloc[train_idx], simulation_ids.iloc[test_idx]) if hasattr(simulation_ids, "iloc") else (simulation_ids[train_idx], simulation_ids[test_idx])
        # print(f"{np.isnan(X_train).mean()=}, {np.isnan(X_test).mean()=}")

        # X_train, X_test, y_train, y_test, sim_ids_train, sim_ids_test = train_test_split(
        #     dataset, target, simulation_ids,
        #     train_size=train_size, 
        #     random_state=random_state,
        #     shuffle=False, # по-хорошему не надо перемешивать
        #     # stratify=target
        # )
        y_test = y_test.reset_index(drop=True) if hasattr(y_test, 'reset_index') else y_test
        sim_ids_test = sim_ids_test.reset_index(drop=True) if hasattr(sim_ids_test, 'reset_index') else sim_ids_test
        print(f"{X_train.shape=}, {y_train.shape=}, {X_test.shape=}, {y_test.shape=}, {y_train.mean()=}, {y_test.mean()=}")
            
        
        for detector_name in self.detectors.keys():                
            if detector_name in model_params_list:
                param_combinations = model_params_list[detector_name]
            else:
                param_combinations = [{}]
            
            for params in param_combinations:
                results = self.run_single_experiment(
                    detector_name, 
                    X_train, X_test, y_train, y_test, sim_ids_train, sim_ids_test,
                    # dataset, target,
                    detector_params=params, delays=test_delays,
                    thresholds=custom_thresholds,
                    # autoencoder=autoencoder
                )
                # print(results)
                
                for result in results:
                    result.update({
                        'freq': "3min",
                        'size': len(dataset),
                        'anomaly_ratio': np.mean(target),
                        'model_params': params
                    })
                    
                all_results.extend(results)
        
        self.results = pd.DataFrame(all_results)
        return self.results
    
    
    
    def get_best_models(self, metric='roc_auc', top_k=5):
        return (self.results.nlargest(top_k, metric))

    def get_best_per_detector(self, metric='pr_auc'):
        """
        Возвращает таблицу с лучшими параметрами для каждого детектора по указанной метрике
        """
        if self.results is None or self.results.empty:
            return pd.DataFrame()
        
        best_per_detector = self.results.loc[
            self.results.groupby('detector')[metric].idxmax()
        ].reset_index(drop=True)
        
        return best_per_detector[['detector', metric, 'model_params'] + 
                                  [col for col in self.results.columns if col not in ['detector', metric, 'model_params']]]
    

