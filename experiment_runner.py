# experiment_runner.py
import yaml
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import argparse
import os
import shutil
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataset_utils import CSVDataLoader
from v2_utils import ExperimentRunner, plot_with_anomalies, AdvancedDetectionEvaluator, BaseAnomalyDetector, ExperimentRunner
import sys

from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from autoencoders import RecurrentAutoencoder
from vae import VariationalRecurrentAutoencoder




def setup_experiment_logging(results_path: Path, experiment_name: str):
    log_file = results_path / f"{experiment_name}.log"

    logger = logging.getLogger(f"experiment.{experiment_name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # удалить старые handler-ы именно у этого logger-а
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8", delay=False)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info(f"Log file: {log_file}")

    return logger


def run_single_experiment(config_path: str, save_plots: bool = True):
    """Запуск одного эксперимента из конфиг-файла"""
    
    # Загрузка конфигурации
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Создание папки с результатами
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = config['experiment']['name']
    results_dir = Path(config['output']['results_path']) / f"{timestamp}_{exp_name}"
    results_dir.mkdir(parents=True, exist_ok=True)
    # out_file = results_dir / f"{exp_name}_out.log"
    # sys.stdout = open(Path(out_file), 'w', buffering=1)
    # sys.stdout = sys.__stdout__

    
    # Сохранение копии конфига
    with open(results_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Настройка логирования
    logger = setup_experiment_logging(results_dir, exp_name)
    logger.info(f"Starting experiment: {exp_name}")
    logger.info(f"Results will be saved to: {results_dir}")
    
    # Сохранение графиков в подпапку
    plots_dir = results_dir / "plots"
    if save_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Загрузка модели автокодировщика
    autoencoder = None
    if 'autoencoder' in config and config['autoencoder'].get('model_path'):
        model_path = config['autoencoder']['model_path']
        window = config['data']['window']
        
        # Предполагаем, что есть метод from_pretrained
        autoencoder = RecurrentAutoencoder.from_pretrained(model_path)
        autoencoder.eval()
        logger.info(f"Loaded autoencoder from {model_path}")
    for handler in logger.handlers:
        handler.flush()
    # Модифицируем функцию run_experiments_by_anomaly для сохранения графиков
    def run_with_logging(*args, **kwargs):
        return run_experiments_by_anomaly(
            *args,
            results_path=str(results_dir),
            save_plots_dir=str(plots_dir) if save_plots else None,
            logger=logger,
            **kwargs
        )
    
    # Запуск эксперимента
    try:
        summary, pivot = run_with_logging(
            anomaly_numbers=config['anomaly_numbers'],
            data_dir_template=config['data']['data_dir_template'],
            model_params_list=config['detectors'],
            file_fraction=config['data'].get('file_fraction', 0.01),
            anomaly_ratio=config['data'].get('anomaly_ratio', 0.11),
            max_samples=config['data'].get('max_samples'),
            result_title=exp_name,
            window=config['data']['window'],
            autoencoder=autoencoder,
            extract_stats=config['data'].get('extract_stats', 0)
        )
        
        logger.info(f"Experiment completed successfully")
        logger.info(f"Results saved to {results_dir}")
        for handler in logger.handlers:
            handler.flush()
        
        # Сохранение summary
        if summary is not None:
            summary.to_csv(results_dir / "summary.csv", index=False)
            pivot.to_csv(results_dir / "pivot_table.csv")
        
        return results_dir, summary, pivot
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}", exc_info=True)
        raise

# def recreate_dir(path):
#     if os.path.exists(path):
#         shutil.rmtree(path)
#     os.makedirs(path)



def run_experiments_by_anomaly(
        anomaly_numbers: List[int],
        data_dir_template: str = "./gen_data/gen_data_{}",
        model_params_list=None,
        file_fraction=0.01,
        anomaly_ratio=0.11,
        max_samples=None,
        result_title="",
        results_path="./results/my",
        window=30,
        autoencoder=None,
        save_plots_dir: str = None, logger=None,
        extract_stats=False,
        **kwargs
    ):
    # timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    # results_path = f"{results_path}/{timestamp}_{result_title if result_title else ""}"
    # recreate_dir(results_path)
    # timestamp = "-"
    normal_dir = data_dir_template.format(0)
    all_best_models = []
    all_best_per_detector = []
    
    for i in anomaly_numbers:
        logger.info(f"\n{'='*50}")
        logger.info(f"Запуск экспериментов для аномалии {i}")
        logger.info(f"{'='*50}")
        start_time = time.time()
        
        data_dir = data_dir_template.format(i)
        
        
        data_gen = CSVDataLoader(data_dir, 
                                 file_fraction=file_fraction,
                                 normal_dir=normal_dir, 
                                 max_samples=max_samples,
                                 anomaly_ratio=anomaly_ratio,
                                 shuffle=True,
                                 normalize=True,
                                 extract_stats=extract_stats
                                )
        # print(f"{data_gen.extract_stats=}")
        
        runner = ExperimentRunner(data_gen, evaluator=AdvancedDetectionEvaluator)
        
        runner.register_detector(
            'IsolationForest', 
            IsolationForest,
            {'random_state': 42, 'contamination': 0.1}
        )

        runner.register_detector(
            'LocalOutlierFactor', 
            LocalOutlierFactor,
            {'contamination': 0.1, "novelty": True}
        )

        # runner.register_detector(
        #     'OneClassSVM', 
        #     OneClassSVM,
        #     {'kernel': 'linear', 'nu': 0.1} # {'kernel': 'rbf', 'gamma': 'auto', 'nu': 0.1}
        # )
        
        results = runner.run_comprehensive_experiments(
            model_params_list=model_params_list,
            test_delays=None,
            autoencoder=autoencoder,
            window=window,
        )
        
        # filename = f"{results_path}/experiments_results_{i}.csv"
        # results.to_csv(filename, index=False)
        # print(f"Результаты сохранены в {filename}")
        
        best_pr = runner.get_best_models('pr_auc', top_k=1)
        best_pr['anomaly_number'] = i
        best_pr['best_by_metric'] = 'pr_auc'
        all_best_models.append(best_pr)
        scores_filename = f"{results_path}/scores_ANOM{i}.npy"
        np.save(scores_filename, np.stack([best_pr["test_scores"], best_pr["test_target"]]))
        logger.info(f"Скоры и метки для test сохранены в {scores_filename}")
        
        best_roc = runner.get_best_models('roc_auc', top_k=1)
        best_roc['anomaly_number'] = i
        best_roc['best_by_metric'] = 'roc_auc'
        all_best_models.append(best_roc)
        # f1 ?
        
        best_per_detector = runner.get_best_per_detector('pr_auc')
        if not best_per_detector.empty:
            best_per_detector['anomaly_number'] = i
            all_best_per_detector.append(best_per_detector)
        if save_plots_dir and best_pr.iloc[0]["survival_km"]:
            plt.figure(figsize=(10, 6))
            best_pr.iloc[0]["survival_km"].plot_survival_function()
            plt.title(f'Функции выживаемости - Аномалия {i}')
            plt.ylabel('Вероятность необнаружения аномалии')
            plt.xlabel('Время (шаги)')
            plt.grid(True)
            
            # Сохраняем график
            plot_path = Path(save_plots_dir) / f"survival_anomaly_{i}.png"
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved survival plot to {plot_path}")
        
        logger.info(f"Аномалия {i}:")
        logger.info(f"  - Лучший PR AUC: {best_pr['pr_auc'].iloc[0]:.4f} ({best_pr['detector'].iloc[0]})")
        logger.info(f"  - Лучший ROC AUC: {best_roc['roc_auc'].iloc[0]:.4f} ({best_roc['detector'].iloc[0]})")
        # logger.info(f"  - Лучший F1: {best_f1['f1_score'].iloc[0]:.4f} ({best_f1['detector'].iloc[0]})")
        cur = time.time()
        logger.info(f"\n-------------------\nВремя выполнения: {cur - start_time:.4f} секунд\n")
        for handler in logger.handlers:
            handler.flush()
    
    if all_best_models and all_best_per_detector:
        summary_df = pd.concat(all_best_models, ignore_index=True)
        numeric_cols = summary_df.select_dtypes(include=['number', 'bool', 'int64', 'float64', 'string']).columns.tolist()
        numeric_cols += [col for col in summary_df.columns if summary_df[col].dtype == 'object' and all(isinstance(x, str) for x in summary_df[col].dropna().head(10))]
        summary_df = summary_df[numeric_cols]
        summary_filename = f"{results_path}/best_models_summary.csv"
        summary_df.to_csv(summary_filename, index=False)
        logger.info(f"\nСводная таблица сохранена в {summary_filename}")
        
        per_detector_df = pd.concat(all_best_per_detector, ignore_index=True)
        pivot_df = per_detector_df.pivot_table(
            index='anomaly_number', 
            columns='detector', 
            values='pr_auc',
            aggfunc='first'
        )
        pivot_filename = f"{results_path}/pr_auc_pivot.csv"
        pivot_df.to_csv(pivot_filename)
        logger.info(f"Сводная таблица PR AUC по детекторам сохранена в {pivot_filename}")
        return summary_df, pivot_df
    return None, None

# experiment_runner.py - добавить в конец файла

def main():
    """Точка входа для запуска из командной строки"""
    parser = argparse.ArgumentParser(description='Запуск эксперимента с детекторами аномалий')
    parser.add_argument('config', type=str, help='Путь к YAML конфиг файлу')
    parser.add_argument('--save-plots', action='store_true', default=True,
                       help='Сохранять графики')
    parser.add_argument('--no-save-plots', action='store_false', dest='save_plots',
                       help='Не сохранять графики')
    
    args = parser.parse_args()
    
    # Проверяем существование конфига
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Ошибка: Конфиг файл не найден: {config_path}")
        return 1
    
    print(f"📖 Загрузка конфига: {config_path}")
    
    # Запускаем эксперимент
    try:
        results_dir, summary, pivot = run_single_experiment(
            str(config_path), 
            save_plots=args.save_plots
        )
        print(f"\n✅ Эксперимент успешно завершен!")
        print(f"📁 Результаты сохранены в: {results_dir}")
        return 0
    except Exception as e:
        print(f"\n❌ Ошибка при выполнении эксперимента: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())