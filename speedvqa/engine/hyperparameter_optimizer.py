"""
超参数优化系统

支持网格搜索和贝叶斯优化，实现参数空间定义和采样，支持多试验并行执行。
"""

import os
import json
import time
import logging
import multiprocessing as mp
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from itertools import product

import numpy as np
import torch
import yaml
from sklearn.model_selection import ParameterGrid

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

try:
    import skopt
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    skopt = None


@dataclass
class TrialResult:
    """单次试验结果"""
    trial_id: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    status: str  # 'completed', 'failed', 'pruned'
    duration: float
    timestamp: str
    error_message: Optional[str] = None


@dataclass
class OptimizationConfig:
    """优化配置"""
    method: str  # 'grid_search', 'bayesian_optuna', 'bayesian_skopt'
    n_trials: int = 100
    timeout: Optional[int] = None  # 总超时时间（秒）
    n_jobs: int = 1  # 并行作业数
    random_state: int = 42
    direction: str = 'maximize'  # 'maximize' or 'minimize'
    metric_name: str = 'val_accuracy'
    
    # 早停配置
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 0.01
    
    # 剪枝配置（仅Optuna）
    pruning: bool = False
    pruning_warmup_steps: int = 5


class ParameterSpace:
    """参数空间定义"""
    
    def __init__(self):
        self.parameters = {}
    
    def add_float(self, name: str, low: float, high: float, log: bool = False):
        """添加浮点参数"""
        self.parameters[name] = {
            'type': 'float',
            'low': low,
            'high': high,
            'log': log
        }
        return self
    
    def add_int(self, name: str, low: int, high: int, log: bool = False):
        """添加整数参数"""
        self.parameters[name] = {
            'type': 'int',
            'low': low,
            'high': high,
            'log': log
        }
        return self
    
    def add_categorical(self, name: str, choices: List[Any]):
        """添加分类参数"""
        self.parameters[name] = {
            'type': 'categorical',
            'choices': choices
        }
        return self
    
    def add_fixed(self, name: str, value: Any):
        """添加固定参数"""
        self.parameters[name] = {
            'type': 'fixed',
            'value': value
        }
        return self
    
    def to_grid_params(self) -> Dict[str, List[Any]]:
        """转换为网格搜索参数格式"""
        grid_params = {}
        for name, param in self.parameters.items():
            if param['type'] == 'float':
                # 为浮点参数生成网格点
                if param.get('log', False):
                    grid_params[name] = np.logspace(
                        np.log10(param['low']), 
                        np.log10(param['high']), 
                        num=5
                    ).tolist()
                else:
                    grid_params[name] = np.linspace(
                        param['low'], 
                        param['high'], 
                        num=5
                    ).tolist()
            elif param['type'] == 'int':
                if param.get('log', False):
                    grid_params[name] = np.logspace(
                        np.log10(param['low']), 
                        np.log10(param['high']), 
                        num=min(5, param['high'] - param['low'] + 1),
                        dtype=int
                    ).tolist()
                else:
                    grid_params[name] = list(range(
                        param['low'], 
                        min(param['high'] + 1, param['low'] + 5)
                    ))
            elif param['type'] == 'categorical':
                grid_params[name] = param['choices']
            elif param['type'] == 'fixed':
                grid_params[name] = [param['value']]
        
        return grid_params
    
    def to_optuna_space(self, trial):
        """转换为Optuna参数空间"""
        params = {}
        for name, param in self.parameters.items():
            if param['type'] == 'float':
                if param.get('log', False):
                    params[name] = trial.suggest_loguniform(
                        name, param['low'], param['high']
                    )
                else:
                    params[name] = trial.suggest_uniform(
                        name, param['low'], param['high']
                    )
            elif param['type'] == 'int':
                if param.get('log', False):
                    params[name] = trial.suggest_int(
                        name, param['low'], param['high'], log=True
                    )
                else:
                    params[name] = trial.suggest_int(
                        name, param['low'], param['high']
                    )
            elif param['type'] == 'categorical':
                params[name] = trial.suggest_categorical(name, param['choices'])
            elif param['type'] == 'fixed':
                params[name] = param['value']
        
        return params
    
    def to_skopt_space(self):
        """转换为scikit-optimize参数空间"""
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize is required for Bayesian optimization")
        
        dimensions = []
        param_names = []
        
        for name, param in self.parameters.items():
            if param['type'] == 'float':
                if param.get('log', False):
                    dimensions.append(Real(param['low'], param['high'], prior='log-uniform'))
                else:
                    dimensions.append(Real(param['low'], param['high']))
                param_names.append(name)
            elif param['type'] == 'int':
                dimensions.append(Integer(param['low'], param['high']))
                param_names.append(name)
            elif param['type'] == 'categorical':
                dimensions.append(Categorical(param['choices']))
                param_names.append(name)
            # 固定参数不加入优化空间
        
        return dimensions, param_names


class BaseOptimizer(ABC):
    """优化器基类"""
    
    def __init__(self, parameter_space: ParameterSpace, 
                 config: OptimizationConfig,
                 objective_function: Callable[[Dict[str, Any]], float]):
        self.parameter_space = parameter_space
        self.config = config
        self.objective_function = objective_function
        self.results: List[TrialResult] = []
        self.best_result: Optional[TrialResult] = None
        
        # 设置日志
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 创建结果保存目录
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.save_dir = Path(f"runs/hyperopt/{timestamp}")
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def optimize(self) -> TrialResult:
        """执行优化"""
        pass
    
    def _evaluate_trial(self, trial_id: str, parameters: Dict[str, Any]) -> TrialResult:
        """评估单次试验"""
        start_time = time.time()
        
        try:
            # 执行目标函数
            metric_value = self.objective_function(parameters)
            
            duration = time.time() - start_time
            
            result = TrialResult(
                trial_id=trial_id,
                parameters=parameters,
                metrics={self.config.metric_name: metric_value},
                status='completed',
                duration=duration,
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
            )
            
            self.logger.info(f"Trial {trial_id} completed: {metric_value:.4f}")
            
        except Exception as e:
            duration = time.time() - start_time
            result = TrialResult(
                trial_id=trial_id,
                parameters=parameters,
                metrics={},
                status='failed',
                duration=duration,
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                error_message=str(e)
            )
            
            self.logger.error(f"Trial {trial_id} failed: {e}")
        
        # 保存结果
        self.results.append(result)
        self._update_best_result(result)
        self._save_result(result)
        
        return result
    
    def _update_best_result(self, result: TrialResult):
        """更新最佳结果"""
        if result.status != 'completed':
            return
        
        metric_value = result.metrics.get(self.config.metric_name)
        if metric_value is None:
            return
        
        if self.best_result is None:
            self.best_result = result
        else:
            best_metric = self.best_result.metrics.get(self.config.metric_name)
            if best_metric is None:
                self.best_result = result
            else:
                if self.config.direction == 'maximize':
                    if metric_value > best_metric:
                        self.best_result = result
                else:  # minimize
                    if metric_value < best_metric:
                        self.best_result = result
    
    def _save_result(self, result: TrialResult):
        """保存单次试验结果"""
        result_file = self.save_dir / f"trial_{result.trial_id}.json"
        with open(result_file, 'w') as f:
            json.dump(asdict(result), f, indent=2)
    
    def save_summary(self):
        """保存优化总结"""
        summary = {
            'config': asdict(self.config),
            'parameter_space': self.parameter_space.parameters,
            'total_trials': len(self.results),
            'completed_trials': len([r for r in self.results if r.status == 'completed']),
            'failed_trials': len([r for r in self.results if r.status == 'failed']),
            'best_result': asdict(self.best_result) if self.best_result else None,
            'all_results': [asdict(r) for r in self.results]
        }
        
        summary_file = self.save_dir / "optimization_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Optimization summary saved to {summary_file}")


class GridSearchOptimizer(BaseOptimizer):
    """网格搜索优化器"""
    
    def optimize(self) -> TrialResult:
        """执行网格搜索优化"""
        self.logger.info("Starting grid search optimization")
        
        # 生成参数网格
        grid_params = self.parameter_space.to_grid_params()
        param_grid = ParameterGrid(grid_params)
        
        total_combinations = len(param_grid)
        self.logger.info(f"Total parameter combinations: {total_combinations}")
        
        if self.config.n_jobs == 1:
            # 串行执行
            for i, params in enumerate(param_grid):
                if i >= self.config.n_trials:
                    break
                
                trial_id = f"grid_{i:04d}"
                self._evaluate_trial(trial_id, params)
        else:
            # 并行执行
            self._parallel_grid_search(param_grid)
        
        self.save_summary()
        return self.best_result
    
    def _parallel_grid_search(self, param_grid):
        """并行网格搜索"""
        # 使用线程池而不是进程池，避免序列化问题
        from concurrent.futures import ThreadPoolExecutor
        
        with ThreadPoolExecutor(max_workers=self.config.n_jobs) as executor:
            # 提交任务
            future_to_trial = {}
            for i, params in enumerate(param_grid):
                if i >= self.config.n_trials:
                    break
                
                trial_id = f"grid_{i:04d}"
                future = executor.submit(self._evaluate_single_trial, trial_id, params)
                future_to_trial[future] = trial_id
            
            # 收集结果
            for future in as_completed(future_to_trial):
                trial_id = future_to_trial[future]
                try:
                    result = future.result()
                    self.results.append(result)
                    self._update_best_result(result)
                    self._save_result(result)
                    self.logger.info(f"Completed trial {trial_id}")
                except Exception as e:
                    self.logger.error(f"Trial {trial_id} failed with exception: {e}")
    
    def _evaluate_single_trial(self, trial_id: str, parameters: Dict[str, Any]) -> TrialResult:
        """评估单次试验（用于并行执行）"""
        start_time = time.time()
        
        try:
            # 执行目标函数
            metric_value = self.objective_function(parameters)
            
            duration = time.time() - start_time
            
            result = TrialResult(
                trial_id=trial_id,
                parameters=parameters,
                metrics={self.config.metric_name: metric_value},
                status='completed',
                duration=duration,
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
            )
            
            self.logger.info(f"Trial {trial_id} completed: {metric_value:.4f}")
            
        except Exception as e:
            duration = time.time() - start_time
            result = TrialResult(
                trial_id=trial_id,
                parameters=parameters,
                metrics={},
                status='failed',
                duration=duration,
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                error_message=str(e)
            )
            
            self.logger.error(f"Trial {trial_id} failed: {e}")
        
        return result


class BayesianOptunaOptimizer(BaseOptimizer):
    """基于Optuna的贝叶斯优化器"""
    
    def __init__(self, *args, **kwargs):
        if not OPTUNA_AVAILABLE:
            raise ImportError("optuna is required for Bayesian optimization")
        super().__init__(*args, **kwargs)
    
    def optimize(self) -> TrialResult:
        """执行贝叶斯优化"""
        self.logger.info("Starting Bayesian optimization with Optuna")
        
        # 创建study
        direction = 'maximize' if self.config.direction == 'maximize' else 'minimize'
        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=self.config.random_state)
        )
        
        # 设置剪枝器
        if self.config.pruning:
            study.sampler = optuna.samplers.TPESampler(seed=self.config.random_state)
            study.pruner = optuna.pruners.MedianPruner(
                n_startup_trials=self.config.pruning_warmup_steps
            )
        
        # 定义目标函数
        def objective(trial):
            trial_id = f"optuna_{trial.number:04d}"
            params = self.parameter_space.to_optuna_space(trial)
            
            # 添加固定参数
            for name, param in self.parameter_space.parameters.items():
                if param['type'] == 'fixed':
                    params[name] = param['value']
            
            result = self._evaluate_trial(trial_id, params)
            
            if result.status == 'failed':
                raise optuna.TrialPruned()
            
            return result.metrics[self.config.metric_name]
        
        # 执行优化
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs
        )
        
        # 保存Optuna特定的结果
        self._save_optuna_results(study)
        self.save_summary()
        
        return self.best_result
    
    def _save_optuna_results(self, study):
        """保存Optuna特定结果"""
        optuna_results = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'trials': [
                {
                    'number': trial.number,
                    'params': trial.params,
                    'value': trial.value,
                    'state': trial.state.name
                }
                for trial in study.trials
            ]
        }
        
        optuna_file = self.save_dir / "optuna_results.json"
        with open(optuna_file, 'w') as f:
            json.dump(optuna_results, f, indent=2)


class BayesianSkoptOptimizer(BaseOptimizer):
    """基于scikit-optimize的贝叶斯优化器"""
    
    def __init__(self, *args, **kwargs):
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize is required for Bayesian optimization")
        super().__init__(*args, **kwargs)
    
    def optimize(self) -> TrialResult:
        """执行贝叶斯优化"""
        self.logger.info("Starting Bayesian optimization with scikit-optimize")
        
        # 获取参数空间
        dimensions, param_names = self.parameter_space.to_skopt_space()
        
        # 定义目标函数
        def objective(x):
            # 构建参数字典
            params = dict(zip(param_names, x))
            
            # 添加固定参数
            for name, param in self.parameter_space.parameters.items():
                if param['type'] == 'fixed':
                    params[name] = param['value']
            
            trial_id = f"skopt_{len(self.results):04d}"
            result = self._evaluate_trial(trial_id, params)
            
            if result.status == 'failed':
                return float('inf') if self.config.direction == 'minimize' else float('-inf')
            
            metric_value = result.metrics[self.config.metric_name]
            # scikit-optimize总是最小化，所以需要转换
            return -metric_value if self.config.direction == 'maximize' else metric_value
        
        # 执行优化
        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=self.config.n_trials,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs if self.config.n_jobs > 1 else 1
        )
        
        # 保存scikit-optimize特定结果
        self._save_skopt_results(result, param_names)
        self.save_summary()
        
        return self.best_result
    
    def _save_skopt_results(self, result, param_names):
        """保存scikit-optimize特定结果"""
        skopt_results = {
            'best_params': dict(zip(param_names, result.x)),
            'best_value': result.fun,
            'n_calls': len(result.func_vals),
            'func_vals': result.func_vals.tolist(),
            'x_iters': [dict(zip(param_names, x)) for x in result.x_iters]
        }
        
        skopt_file = self.save_dir / "skopt_results.json"
        with open(skopt_file, 'w') as f:
            json.dump(skopt_results, f, indent=2)


class HyperparameterOptimizer:
    """超参数优化器主类"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 设置日志格式
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def optimize(self, parameter_space: ParameterSpace, 
                 objective_function: Callable[[Dict[str, Any]], float]) -> TrialResult:
        """执行超参数优化"""
        
        # 选择优化器
        if self.config.method == 'grid_search':
            optimizer = GridSearchOptimizer(parameter_space, self.config, objective_function)
        elif self.config.method == 'bayesian_optuna':
            optimizer = BayesianOptunaOptimizer(parameter_space, self.config, objective_function)
        elif self.config.method == 'bayesian_skopt':
            optimizer = BayesianSkoptOptimizer(parameter_space, self.config, objective_function)
        else:
            raise ValueError(f"Unsupported optimization method: {self.config.method}")
        
        # 执行优化
        self.logger.info(f"Starting hyperparameter optimization with {self.config.method}")
        best_result = optimizer.optimize()
        
        if best_result:
            self.logger.info(f"Optimization completed. Best result:")
            self.logger.info(f"  Parameters: {best_result.parameters}")
            self.logger.info(f"  Metric: {best_result.metrics}")
        else:
            self.logger.warning("No successful trials found")
        
        return best_result
    
    @classmethod
    def from_config_file(cls, config_path: str) -> 'HyperparameterOptimizer':
        """从配置文件创建优化器"""
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config_dict = yaml.safe_load(f)
            else:
                config_dict = json.load(f)
        
        config = OptimizationConfig(**config_dict)
        return cls(config)


def create_parameter_space_from_config(space_config: Dict[str, Any]) -> ParameterSpace:
    """从配置创建参数空间"""
    space = ParameterSpace()
    
    for name, param_config in space_config.items():
        param_type = param_config['type']
        
        if param_type == 'float':
            space.add_float(
                name,
                param_config['low'],
                param_config['high'],
                param_config.get('log', False)
            )
        elif param_type == 'int':
            space.add_int(
                name,
                param_config['low'],
                param_config['high'],
                param_config.get('log', False)
            )
        elif param_type == 'categorical':
            space.add_categorical(name, param_config['choices'])
        elif param_type == 'fixed':
            space.add_fixed(name, param_config['value'])
        else:
            raise ValueError(f"Unsupported parameter type: {param_type}")
    
    return space


# 示例使用函数
def example_objective_function(params: Dict[str, Any]) -> float:
    """示例目标函数"""
    # 这里应该是实际的模型训练和验证过程
    # 返回验证指标（如准确率）
    
    # 模拟训练过程
    time.sleep(0.1)  # 模拟训练时间
    
    # 模拟基于参数的性能
    lr = params.get('learning_rate', 0.001)
    batch_size = params.get('batch_size', 32)
    
    # 简单的模拟函数：学习率和批次大小的组合效果
    score = 0.8 + 0.1 * np.sin(lr * 1000) + 0.05 * np.cos(batch_size / 10)
    score += np.random.normal(0, 0.02)  # 添加噪声
    
    return max(0, min(1, score))  # 限制在[0,1]范围内


if __name__ == "__main__":
    # 示例用法
    
    # 创建参数空间
    space = ParameterSpace()
    space.add_float('learning_rate', 1e-5, 1e-1, log=True)
    space.add_int('batch_size', 16, 128)
    space.add_categorical('optimizer', ['adam', 'sgd', 'adamw'])
    space.add_fixed('epochs', 10)
    
    # 创建优化配置
    config = OptimizationConfig(
        method='grid_search',
        n_trials=20,
        n_jobs=2,
        direction='maximize',
        metric_name='val_accuracy'
    )
    
    # 创建优化器
    optimizer = HyperparameterOptimizer(config)
    
    # 执行优化
    best_result = optimizer.optimize(space, example_objective_function)
    
    print(f"Best parameters: {best_result.parameters}")
    print(f"Best score: {best_result.metrics}")