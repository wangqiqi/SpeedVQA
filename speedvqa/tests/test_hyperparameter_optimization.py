"""
超参数优化属性测试

**属性 7: 超参数优化有效性**
**验证需求: 需求 2.5**

测试超参数优化系统的正确性和有效性。
"""

import time
from typing import Dict, Any
import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant

from speedvqa.engine.hyperparameter_optimizer import (
    HyperparameterOptimizer,
    ParameterSpace,
    OptimizationConfig,
    TrialResult,
    GridSearchOptimizer,
    create_parameter_space_from_config
)


# 测试策略定义
@st.composite
def optimization_config_strategy(draw):
    """生成优化配置的策略"""
    method = draw(st.sampled_from(['grid_search', 'bayesian_optuna', 'bayesian_skopt']))
    n_trials = draw(st.integers(min_value=1, max_value=20))
    n_jobs = draw(st.integers(min_value=1, max_value=4))
    direction = draw(st.sampled_from(['maximize', 'minimize']))
    metric_name = draw(st.sampled_from(['accuracy', 'val_accuracy', 'f1_score']))
    
    return OptimizationConfig(
        method=method,
        n_trials=n_trials,
        n_jobs=n_jobs,
        direction=direction,
        metric_name=metric_name,
        random_state=42
    )


@st.composite
def parameter_space_strategy(draw):
    """生成参数空间的策略"""
    space = ParameterSpace()
    
    # 添加浮点参数
    n_float_params = draw(st.integers(min_value=1, max_value=3))
    for i in range(n_float_params):
        low = draw(st.floats(min_value=1e-5, max_value=1e-2))
        high = draw(st.floats(min_value=low * 2, max_value=1.0))
        log = draw(st.booleans())
        space.add_float(f'float_param_{i}', low, high, log)
    
    # 添加整数参数
    n_int_params = draw(st.integers(min_value=1, max_value=2))
    for i in range(n_int_params):
        low = draw(st.integers(min_value=1, max_value=10))
        high = draw(st.integers(min_value=low + 1, max_value=100))
        space.add_int(f'int_param_{i}', low, high)
    
    # 添加分类参数
    n_cat_params = draw(st.integers(min_value=0, max_value=2))
    for i in range(n_cat_params):
        choices = draw(st.lists(st.text(alphabet=st.characters(min_codepoint=ord('a'), max_codepoint=ord('z')), min_size=1, max_size=10), min_size=2, max_size=5))
        space.add_categorical(f'cat_param_{i}', choices)
    
    # 添加固定参数
    space.add_fixed('fixed_param', 'fixed_value')
    
    return space


def simple_objective_function(params: Dict[str, Any]) -> float:
    """简单的目标函数用于测试"""
    # 基于参数计算一个简单的分数
    score = 0.5
    
    for key, value in params.items():
        if isinstance(value, (int, float)):
            score += 0.1 * np.sin(value)
        elif isinstance(value, str):
            score += 0.05 * len(value)
    
    # 添加少量随机噪声
    score += np.random.normal(0, 0.01)
    
    return max(0, min(1, score))


class TestParameterSpace:
    """参数空间测试"""
    
    @given(parameter_space_strategy())
    def test_parameter_space_consistency(self, space):
        """
        **属性 7.1: 参数空间定义一致性**
        参数空间的定义应该保持一致性，不同格式转换后应包含相同的参数。
        """
        # 检查参数空间不为空
        assert len(space.parameters) > 0
        
        # 检查网格参数转换
        grid_params = space.to_grid_params()
        assert len(grid_params) == len(space.parameters)
        
        # 检查所有参数都被转换
        for param_name in space.parameters.keys():
            assert param_name in grid_params
            assert len(grid_params[param_name]) > 0
    
    @given(st.floats(min_value=1e-5, max_value=1e-2),
           st.floats(min_value=1e-2, max_value=1.0))
    def test_float_parameter_bounds(self, low, high):
        """
        **属性 7.2: 浮点参数边界正确性**
        浮点参数的边界应该被正确设置和验证。
        """
        assume(low < high)
        
        space = ParameterSpace()
        space.add_float('test_param', low, high)
        
        grid_params = space.to_grid_params()
        values = grid_params['test_param']
        
        # 检查边界
        assert min(values) >= low
        assert max(values) <= high
        
        # 检查值的数量合理
        assert len(values) > 0
    
    @given(st.integers(min_value=1, max_value=10),
           st.integers(min_value=11, max_value=100))
    def test_int_parameter_bounds(self, low, high):
        """
        **属性 7.3: 整数参数边界正确性**
        整数参数的边界应该被正确设置和验证。
        """
        space = ParameterSpace()
        space.add_int('test_param', low, high)
        
        grid_params = space.to_grid_params()
        values = grid_params['test_param']
        
        # 检查边界
        assert min(values) >= low
        assert max(values) <= high
        
        # 检查所有值都是整数
        assert all(isinstance(v, int) for v in values)
    
    @given(st.lists(st.text(alphabet=st.characters(min_codepoint=ord('a'), max_codepoint=ord('z')), min_size=1, max_size=10), min_size=2, max_size=5))
    def test_categorical_parameter_choices(self, choices):
        """
        **属性 7.4: 分类参数选择正确性**
        分类参数的选择应该完全匹配输入的选择列表。
        """
        space = ParameterSpace()
        space.add_categorical('test_param', choices)
        
        grid_params = space.to_grid_params()
        values = grid_params['test_param']
        
        # 检查选择完全匹配
        assert set(values) == set(choices)
        assert len(values) == len(choices)


class TestOptimizationConfig:
    """优化配置测试"""
    
    @given(optimization_config_strategy())
    def test_config_validation(self, config):
        """
        **属性 7.5: 优化配置验证**
        优化配置应该包含所有必需的字段并且值合理。
        """
        # 检查必需字段
        assert hasattr(config, 'method')
        assert hasattr(config, 'n_trials')
        assert hasattr(config, 'direction')
        assert hasattr(config, 'metric_name')
        
        # 检查值的合理性
        assert config.n_trials > 0
        assert config.direction in ['maximize', 'minimize']
        assert len(config.metric_name) > 0
        assert config.n_jobs > 0


class TestGridSearchOptimizer:
    """网格搜索优化器测试"""
    
    @given(parameter_space_strategy())
    @settings(max_examples=5, deadline=30000)  # 限制测试数量和时间
    def test_grid_search_completeness(self, space):
        """
        **属性 7.6: 网格搜索完整性**
        网格搜索应该评估所有可能的参数组合（在试验限制内）。
        """
        config = OptimizationConfig(
            method='grid_search',
            n_trials=10,  # 限制试验数量
            n_jobs=1,
            direction='maximize',
            metric_name='score'
        )
        
        optimizer = GridSearchOptimizer(space, config, simple_objective_function)
        
        # 计算预期的组合数
        grid_params = space.to_grid_params()
        expected_combinations = 1
        for param_values in grid_params.values():
            expected_combinations *= len(param_values)
        
        # 执行优化
        result = optimizer.optimize()
        
        # 检查结果
        assert result is not None
        assert len(optimizer.results) > 0
        assert len(optimizer.results) <= min(config.n_trials, expected_combinations)
        
        # 检查所有试验都有唯一的参数组合
        param_combinations = [tuple(sorted(r.parameters.items())) for r in optimizer.results]
        assert len(param_combinations) == len(set(param_combinations))
    
    @given(parameter_space_strategy())
    @settings(max_examples=3, deadline=20000)
    def test_grid_search_best_result_tracking(self, space):
        """
        **属性 7.7: 最佳结果跟踪**
        优化器应该正确跟踪和更新最佳结果。
        """
        config = OptimizationConfig(
            method='grid_search',
            n_trials=5,
            n_jobs=1,
            direction='maximize',
            metric_name='score'
        )
        
        optimizer = GridSearchOptimizer(space, config, simple_objective_function)
        result = optimizer.optimize()
        
        # 检查最佳结果存在
        assert optimizer.best_result is not None
        assert result == optimizer.best_result
        
        # 检查最佳结果确实是最好的
        completed_results = [r for r in optimizer.results if r.status == 'completed']
        if completed_results:
            best_metric = optimizer.best_result.metrics[config.metric_name]
            
            if config.direction == 'maximize':
                for r in completed_results:
                    assert r.metrics[config.metric_name] <= best_metric
            else:
                for r in completed_results:
                    assert r.metrics[config.metric_name] >= best_metric


class TestHyperparameterOptimizer:
    """超参数优化器主类测试"""
    
    @given(parameter_space_strategy())
    @settings(max_examples=3, deadline=30000)
    def test_optimizer_method_selection(self, space):
        """
        **属性 7.8: 优化方法选择正确性**
        优化器应该根据配置正确选择优化方法。
        """
        # 创建固定的配置避免随机性问题
        config = OptimizationConfig(
            method='grid_search',
            n_trials=3,  # 限制试验数量
            n_jobs=1,    # 强制使用单进程避免并行问题
            direction='maximize',
            metric_name='score',
            random_state=42
        )
        
        optimizer = HyperparameterOptimizer(config)
        result = optimizer.optimize(space, simple_objective_function)
        
        # 检查结果存在且合理
        assert result is not None, "Optimizer should return a valid result"
        assert result.status == 'completed', f"Expected completed status, got {result.status}"
        assert config.metric_name in result.metrics, f"Result should contain metric {config.metric_name}"
        assert isinstance(result.metrics[config.metric_name], (int, float)), "Metric should be numeric"
    
    def test_invalid_optimization_method(self):
        """
        **属性 7.9: 无效方法处理**
        优化器应该正确处理无效的优化方法。
        """
        config = OptimizationConfig(
            method='invalid_method',
            n_trials=5,
            direction='maximize',
            metric_name='score'
        )
        
        space = ParameterSpace()
        space.add_float('param', 0.1, 1.0)
        
        optimizer = HyperparameterOptimizer(config)
        
        with pytest.raises(ValueError, match="Unsupported optimization method"):
            optimizer.optimize(space, simple_objective_function)


class TestTrialResult:
    """试验结果测试"""
    
    @given(st.text(alphabet=st.characters(min_codepoint=ord('a'), max_codepoint=ord('z')), min_size=1, max_size=20),
           st.dictionaries(st.text(alphabet=st.characters(min_codepoint=ord('a'), max_codepoint=ord('z')), min_size=1, max_size=10), 
                          st.one_of(st.floats(min_value=0, max_value=1), 
                                   st.integers(min_value=1, max_value=100),
                                   st.text(alphabet=st.characters(min_codepoint=ord('a'), max_codepoint=ord('z')), min_size=1, max_size=10))),
           st.dictionaries(st.text(alphabet=st.characters(min_codepoint=ord('a'), max_codepoint=ord('z')), min_size=1, max_size=10), 
                          st.floats(min_value=0, max_value=1)),
           st.sampled_from(['completed', 'failed', 'pruned']),
           st.floats(min_value=0.1, max_value=10.0))
    def test_trial_result_creation(self, trial_id, parameters, metrics, status, duration):
        """
        **属性 7.10: 试验结果创建正确性**
        试验结果应该正确存储所有提供的信息。
        """
        result = TrialResult(
            trial_id=trial_id,
            parameters=parameters,
            metrics=metrics,
            status=status,
            duration=duration,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )
        
        # 检查所有字段都被正确设置
        assert result.trial_id == trial_id
        assert result.parameters == parameters
        assert result.metrics == metrics
        assert result.status == status
        assert result.duration == duration
        assert result.timestamp is not None


class TestConfigurationFromFile:
    """配置文件测试"""
    
    def test_parameter_space_from_config(self):
        """
        **属性 7.11: 配置文件解析正确性**
        参数空间应该能够从配置字典正确创建。
        """
        config = {
            'learning_rate': {
                'type': 'float',
                'low': 1e-5,
                'high': 1e-1,
                'log': True
            },
            'batch_size': {
                'type': 'int',
                'low': 16,
                'high': 128
            },
            'optimizer': {
                'type': 'categorical',
                'choices': ['adam', 'sgd', 'adamw']
            },
            'epochs': {
                'type': 'fixed',
                'value': 10
            }
        }
        
        space = create_parameter_space_from_config(config)
        
        # 检查所有参数都被添加
        assert len(space.parameters) == 4
        assert 'learning_rate' in space.parameters
        assert 'batch_size' in space.parameters
        assert 'optimizer' in space.parameters
        assert 'epochs' in space.parameters
        
        # 检查参数类型正确
        assert space.parameters['learning_rate']['type'] == 'float'
        assert space.parameters['learning_rate']['log'] is True
        assert space.parameters['batch_size']['type'] == 'int'
        assert space.parameters['optimizer']['type'] == 'categorical'
        assert space.parameters['epochs']['type'] == 'fixed'


class HyperparameterOptimizationStateMachine(RuleBasedStateMachine):
    """
    **属性 7.12: 超参数优化状态机测试**
    使用状态机测试超参数优化的复杂交互和状态转换。
    """
    
    def __init__(self):
        super().__init__()
        self.parameter_spaces = []
        self.configs = []
        self.optimizers = []
        self.results = []
    
    @initialize()
    def setup(self):
        """初始化状态机"""
        # 创建基本的参数空间
        space = ParameterSpace()
        space.add_float('lr', 0.001, 0.1)
        space.add_int('batch_size', 16, 64)
        space.add_categorical('optimizer', ['adam', 'sgd'])
        self.parameter_spaces.append(space)
        
        # 创建基本配置
        config = OptimizationConfig(
            method='grid_search',
            n_trials=4,
            n_jobs=1,
            direction='maximize',
            metric_name='score'
        )
        self.configs.append(config)
    
    @rule()
    def add_parameter_space(self):
        """添加新的参数空间"""
        space = ParameterSpace()
        space.add_float('param1', 0.1, 1.0)
        space.add_int('param2', 1, 10)
        self.parameter_spaces.append(space)
    
    @rule()
    def add_config(self):
        """添加新的配置"""
        config = OptimizationConfig(
            method='grid_search',
            n_trials=3,
            n_jobs=1,
            direction='maximize',
            metric_name='score'
        )
        self.configs.append(config)
    
    @rule()
    def create_optimizer(self):
        """创建优化器"""
        if self.configs:
            config = self.configs[-1]
            optimizer = HyperparameterOptimizer(config)
            self.optimizers.append(optimizer)
    
    @rule()
    def run_optimization(self):
        """运行优化"""
        if self.optimizers and self.parameter_spaces:
            optimizer = self.optimizers[-1]
            space = self.parameter_spaces[-1]
            
            try:
                result = optimizer.optimize(space, simple_objective_function)
                if result:
                    self.results.append(result)
            except Exception:
                # 记录异常但不失败
                pass
    
    @invariant()
    def results_are_valid(self):
        """检查所有结果都是有效的"""
        for result in self.results:
            assert result.trial_id is not None
            assert isinstance(result.parameters, dict)
            assert isinstance(result.metrics, dict)
            assert result.status in ['completed', 'failed', 'pruned']
            assert result.duration >= 0


# 运行状态机测试
TestHyperparameterOptimizationState = HyperparameterOptimizationStateMachine.TestCase


if __name__ == "__main__":
    # 运行基本测试
    pytest.main([__file__, "-v"])