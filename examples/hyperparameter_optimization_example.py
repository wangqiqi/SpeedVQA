#!/usr/bin/env python3
"""
超参数优化使用示例

演示如何使用SpeedVQA的超参数优化系统进行模型调优。
"""

import os
import sys
import yaml
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from speedvqa.engine.hyperparameter_optimizer import (
    HyperparameterOptimizer,
    ParameterSpace,
    OptimizationConfig,
    create_parameter_space_from_config
)


def mock_training_objective(params):
    """
    模拟训练目标函数
    
    在实际使用中，这个函数应该：
    1. 使用给定的超参数配置训练模型
    2. 在验证集上评估模型性能
    3. 返回验证指标（如准确率）
    """
    
    # 模拟训练过程
    print(f"Training with parameters: {params}")
    
    # 基于参数计算模拟性能
    lr = params.get('learning_rate', 0.001)
    batch_size = params.get('batch_size', 32)
    optimizer_type = params.get('optimizer_type', 'adam')
    weight_decay = params.get('weight_decay', 1e-4)
    
    # 简单的性能模拟函数
    base_score = 0.75
    
    # 学习率影响
    lr_factor = 1.0 - abs(np.log10(lr) + 3) * 0.05  # 最优在1e-3附近
    
    # 批次大小影响
    batch_factor = 1.0 - abs(batch_size - 64) * 0.001  # 最优在64附近
    
    # 优化器影响
    optimizer_factors = {'adam': 1.0, 'adamw': 1.02, 'sgd': 0.95}
    opt_factor = optimizer_factors.get(optimizer_type, 1.0)
    
    # 权重衰减影响
    wd_factor = 1.0 - abs(np.log10(weight_decay) + 4) * 0.02  # 最优在1e-4附近
    
    # 计算最终分数
    score = base_score * lr_factor * batch_factor * opt_factor * wd_factor
    
    # 添加随机噪声
    score += np.random.normal(0, 0.01)
    
    # 限制在合理范围内
    score = max(0.5, min(0.95, score))
    
    print(f"Validation accuracy: {score:.4f}")
    return score


def example_1_manual_parameter_space():
    """示例1: 手动定义参数空间"""
    
    print("=== 示例1: 手动定义参数空间 ===")
    
    # 创建参数空间
    space = ParameterSpace()
    space.add_float('learning_rate', 1e-5, 1e-1, log=True)
    space.add_int('batch_size', 16, 128)
    space.add_categorical('optimizer_type', ['adam', 'adamw', 'sgd'])
    space.add_float('weight_decay', 1e-6, 1e-2, log=True)
    space.add_fixed('epochs', 10)  # 固定参数
    
    # 创建优化配置
    config = OptimizationConfig(
        method='grid_search',
        n_trials=20,
        n_jobs=1,  # 使用单进程避免并行问题
        direction='maximize',
        metric_name='val_accuracy',
        random_state=42
    )
    
    # 创建优化器
    optimizer = HyperparameterOptimizer(config)
    
    # 执行优化
    best_result = optimizer.optimize(space, mock_training_objective)
    
    if best_result:
        print(f"\n最佳参数: {best_result.parameters}")
        print(f"最佳性能: {best_result.metrics}")
        print(f"训练时间: {best_result.duration:.2f}秒")
    else:
        print("优化失败，没有找到有效结果")


def example_2_config_file_parameter_space():
    """示例2: 从配置文件定义参数空间"""
    
    print("\n=== 示例2: 从配置文件定义参数空间 ===")
    
    # 读取配置文件
    config_path = Path(__file__).parent.parent / 'configs' / 'hyperopt_example.yaml'
    
    if not config_path.exists():
        print(f"配置文件不存在: {config_path}")
        return
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # 从配置创建参数空间
    space = create_parameter_space_from_config(config_dict['parameter_space'])
    
    # 从配置创建优化配置
    opt_config = OptimizationConfig(**config_dict['optimization'])
    opt_config.n_trials = 15  # 减少试验数量用于演示
    opt_config.n_jobs = 1     # 使用单进程
    
    # 如果配置使用贝叶斯优化但optuna不可用，则改用网格搜索
    if opt_config.method in ['bayesian_optuna', 'bayesian_skopt']:
        try:
            import optuna
        except ImportError:
            print("Optuna not available, switching to grid search")
            opt_config.method = 'grid_search'
            opt_config.n_trials = 10
    
    # 创建优化器
    optimizer = HyperparameterOptimizer(opt_config)
    
    # 执行优化
    best_result = optimizer.optimize(space, mock_training_objective)
    
    if best_result:
        print(f"\n最佳参数: {best_result.parameters}")
        print(f"最佳性能: {best_result.metrics}")
        print(f"训练时间: {best_result.duration:.2f}秒")
    else:
        print("优化失败，没有找到有效结果")


def example_3_bayesian_optimization():
    """示例3: 贝叶斯优化（需要安装optuna）"""
    
    print("\n=== 示例3: 贝叶斯优化 ===")
    
    try:
        import optuna
        
        # 创建参数空间
        space = ParameterSpace()
        space.add_float('learning_rate', 1e-4, 1e-2, log=True)
        space.add_int('batch_size', 32, 96)
        space.add_categorical('optimizer_type', ['adam', 'adamw'])
        space.add_float('weight_decay', 1e-5, 1e-3, log=True)
        
        # 创建贝叶斯优化配置
        config = OptimizationConfig(
            method='bayesian_optuna',
            n_trials=25,
            n_jobs=1,
            direction='maximize',
            metric_name='val_accuracy',
            random_state=42,
            pruning=True,
            pruning_warmup_steps=3
        )
        
        # 创建优化器
        optimizer = HyperparameterOptimizer(config)
        
        # 执行优化
        best_result = optimizer.optimize(space, mock_training_objective)
        
        if best_result:
            print(f"\n最佳参数: {best_result.parameters}")
            print(f"最佳性能: {best_result.metrics}")
            print(f"训练时间: {best_result.duration:.2f}秒")
        else:
            print("优化失败，没有找到有效结果")
            
    except ImportError:
        print("需要安装optuna才能运行贝叶斯优化示例:")
        print("pip install optuna")


def example_4_compare_methods():
    """示例4: 比较不同优化方法"""
    
    print("\n=== 示例4: 比较不同优化方法 ===")
    
    # 定义相同的参数空间
    space = ParameterSpace()
    space.add_float('learning_rate', 1e-4, 1e-2, log=True)
    space.add_int('batch_size', 32, 64)
    space.add_categorical('optimizer_type', ['adam', 'adamw'])
    
    methods = ['grid_search']
    
    # 检查是否可以使用贝叶斯优化
    try:
        import optuna
        methods.append('bayesian_optuna')
    except ImportError:
        pass
    
    results = {}
    
    for method in methods:
        print(f"\n--- 使用 {method} ---")
        
        config = OptimizationConfig(
            method=method,
            n_trials=12,
            n_jobs=1,
            direction='maximize',
            metric_name='val_accuracy',
            random_state=42
        )
        
        optimizer = HyperparameterOptimizer(config)
        best_result = optimizer.optimize(space, mock_training_objective)
        
        if best_result:
            results[method] = best_result
            print(f"最佳性能: {best_result.metrics['val_accuracy']:.4f}")
        else:
            print("优化失败")
    
    # 比较结果
    if results:
        print("\n=== 方法比较 ===")
        for method, result in results.items():
            print(f"{method}: {result.metrics['val_accuracy']:.4f}")
        
        best_method = max(results.keys(), 
                         key=lambda m: results[m].metrics['val_accuracy'])
        print(f"\n最佳方法: {best_method}")


if __name__ == "__main__":
    # 设置随机种子确保可重现性
    np.random.seed(42)
    
    print("SpeedVQA 超参数优化示例")
    print("=" * 50)
    
    # 运行示例
    example_1_manual_parameter_space()
    example_2_config_file_parameter_space()
    example_3_bayesian_optimization()
    example_4_compare_methods()
    
    print("\n" + "=" * 50)
    print("示例完成！")
    
    print("\n使用提示:")
    print("1. 在实际使用中，将mock_training_objective替换为真实的训练函数")
    print("2. 根据计算资源调整n_trials和n_jobs参数")
    print("3. 使用贝叶斯优化可以更高效地搜索参数空间")
    print("4. 保存优化结果到文件以便后续分析")