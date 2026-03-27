import os
import pandas as pd
import numpy as np
import multiprocessing as mp
from typing import List, Dict, Any, Optional
from temain import TemainProcess
import traceback
from multiprocessing import get_context
import logging
logging.basicConfig(level=logging.INFO)

N_DV = 20


class SingleProcessGenerator:
    def __init__(self,
                 simulation_id: int,
                 output_dir: str,
                 total_steps: int = 10000,
                 warmup_steps: int = 0,
                 min_start = 0,
                 steps_after_fault: int = None,
                 fault_id: int = 0, # пока никак
                 freq: str = "60s",
                 reduce: bool = True,
                 random_seed: int = 0,
                 verbose: int = 1, # 0, 1, 2
            ):

        self.simulation_id = simulation_id
        self.output_dir = output_dir
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        if steps_after_fault is None and fault_id != 0:
            self.steps_after_fault = max(200, total_steps)
        elif fault_id == 0:
            self.steps_after_fault = 0
        else:
            self.steps_after_fault = steps_after_fault
        self.freq = freq
        self.random_seed = random_seed
        self.fault_id = fault_id
        #fix freq
        self.reduce = reduce
        if self.reduce:
            self.stride = (int(self.freq[:-1]) if self.freq[-1] == "s" else int(self.freq[:-3]) * 60) # min
        else:
            self.stride = 1
        self.verbose = verbose
        

        if self.total_steps <= self.steps_after_fault and fault_id != 0:
            raise ValueError(f"total_steps=({total_steps}) must be greater than steps_after_fault=({steps_after_fault})")

        if fault_id == 0:
            # L, R = self.warmup_steps, self.warmup_steps + self.total_steps
            self.fault_step = self.warmup_steps + self.total_steps
        else:
            np.random.seed(self.random_seed)
            available_start_steps = self.total_steps - self.steps_after_fault
            min_steps_before_anomaly = max(available_start_steps // 5, min_start) # 20%
            L = self.warmup_steps + min_steps_before_anomaly
            R = self.warmup_steps + available_start_steps
            if L > R:
                L = R
            self.fault_step = np.random.randint(L, R + 1)
        if self.verbose > 1:
            if fault_id != 0:
                print(f"{L=}, {R=}, {self.steps_after_fault=}")
            print(self.fault_step)
        self.process = None
        self.data_buffer = []
        self.labels_buffer = []
        os.makedirs(output_dir, exist_ok=True)


    def initialize_process(self) -> None:
        np.random.seed(self.random_seed)
        if self.reduce: # генерация с частотой 1 секунда, а дальше усреднение
            self.process = TemainProcess(freq="1s", seed=self.random_seed) # fix freq
        else:
            self.process = TemainProcess(freq=self.freq, seed=self.random_seed) # fix freq

    def create_dv_array_for_fault(self, n_steps) -> tuple:
        DV_array = np.zeros((n_steps, N_DV))  #N_DV
        if self.fault_id != 0:
            DV_array[self.fault_step:, self.fault_id - 1] = 1
        # тут надо бы разные типы тогда учесть, пока только ступенька для переменной 1
        return DV_array

    def run_simulation(self):
        try:
            self.initialize_process()
            self.n = min(self.warmup_steps + self.total_steps, self.fault_step + self.steps_after_fault)
            #тут бы еще замечать совсем расхождение процесса
            DVS = self.create_dv_array_for_fault(self.n)

            if self.verbose > 1:
                print(f"DEBUG: self.n={self.n}, self.fault_step={self.fault_step}, self.stride={self.stride}")
                print(f"DEBUG: (self.n - self.fault_step)={self.n - self.fault_step}")
                print(f"DEBUG: (self.stride)={self.stride}")
                print(f"DEBUG: (self.n - self.fault_step) * self.stride={(self.n - self.fault_step) * self.stride}")
                print(f"DEBUG: DVS shape={DVS.shape}, DVS[self.fault_step:] shape={DVS[self.fault_step:].shape}, {DVS.mean()=}")

            normal = self.process.simulate(self.fault_step * self.stride)
            anomaly = self.process.simulate((self.n - self.fault_step) * self.stride, DV=np.repeat(DVS[self.fault_step:], self.stride, axis=0))
            
            normal = normal.iloc[::self.stride].reset_index()
            anomaly = anomaly.iloc[::self.stride].reset_index()

            result = pd.concat([normal, anomaly], ignore_index=True).reset_index()
            labels = (DVS != 0).any(axis = 1)
            if self.verbose > 1:
                print(f"{labels.mean()=}")
            steps_from_anomaly_start = np.cumsum(labels) - 1


            data_df = pd.DataFrame(result)
            data_df["y"] = labels
            data_df["_steps_from_anomaly_start"] = steps_from_anomaly_start
            output_file = os.path.join(self.output_dir, str(self.simulation_id) + ".csv")
            data_df = data_df.iloc[self.warmup_steps:]
            data_df.to_csv(output_file, index=True)
            if self.verbose > 1:
                print(data_df.shape)
                print(data_df["y"].sum())
            if self.verbose > 0:
                print(f"Process {self.simulation_id} with {self.n} samples: Results saved to {output_file}")

        except Exception as e:
            print(f"Process {self.simulation_id}: Failed with error: {e}")
            traceback.print_exc()
            raise ValueError()


class ParallelDataGenerator:
    def __init__(self, output_dir, num_processes=None):
        self.output_dir = output_dir
        self.num_processes = num_processes or max(1, mp.cpu_count() - 2)
        os.makedirs(output_dir, exist_ok=True)
        for file_name in os.listdir(self.output_dir):
            file_path = os.path.join(self.output_dir, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

    @staticmethod
    def _run_single_simulation(config: Dict[str, Any]) -> Dict[str, Any]:
        try:
            generator = SingleProcessGenerator(**config)
            generator.run_simulation()
            return {
                'simulation_id': config['simulation_id'],
                'status': 'success'
            }
        except Exception as e:
            return {
                'simulation_id': config.get('simulation_id', -1),
                'status': 'error',
                'error': str(e)
            }

    def generate_simulation_configs(self, num_simulations, base_config = None):
        if base_config is None:
            base_config = {}
        
        configs = []
        for i in range(num_simulations):
            config = {
                'simulation_id': i,
                'output_dir': self.output_dir,
                'total_steps': base_config.get('total_steps', 10000),
                'min_start': base_config.get('min_start', 20),
                'warmup_steps': base_config.get('warmup_steps', 20),
                'steps_after_fault': base_config.get('steps_after_fault', None),
                'fault_id': base_config.get('fault_id', 0),
                'freq': base_config.get('freq', '60s'),
                'random_seed': base_config.get('random_seed', 0) + i,
                'reduce': base_config.get('reduce', False),
            }
            configs.append(config)
        return configs

    def generate(self,num_simulations, base_config=None):
        configs = self.generate_simulation_configs(num_simulations, base_config)
        # with mp.Pool(processes=self.num_processes) as pool:
        #     results = pool.map(self._run_single_simulation, configs)
        
        with get_context("spawn").Pool(processes=self.num_processes) as pool:
            results = pool.map(self._run_single_simulation, configs)
        success_count = sum(1 for r in results if r.get('status') == 'success')
        print(f"Generated {success_count}/{num_simulations} successful simulations")
        return results