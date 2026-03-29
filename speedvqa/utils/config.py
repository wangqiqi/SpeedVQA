"""
SpeedVQA配置管理系统

提供YAML配置文件加载、验证、合并等功能。
支持配置继承和命令行参数覆盖。
"""

import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import argparse


def get_package_root() -> Path:
    """`speedvqa` 包根目录（内含 `configs/`）。"""
    return Path(__file__).resolve().parent.parent


def get_builtin_default_config_path() -> str:
    return str(get_package_root() / "configs" / "default.yaml")


def resolve_config_file(config_path: str) -> Path:
    """解析配置文件路径：支持 cwd 相对路径，否则回退到包内 `speedvqa/configs/`。"""
    p = Path(config_path)
    if p.is_file():
        return p.resolve()
    fallback = get_package_root() / config_path
    if fallback.is_file():
        return fallback.resolve()
    raise FileNotFoundError(f"Config file not found: {config_path}")


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = None
    
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """加载配置文件"""
        if config_path is None:
            config_path = self.config_path or get_builtin_default_config_path()
        
        config_path = resolve_config_file(str(config_path))
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 处理配置继承（如果有defaults字段）
        if 'defaults' in self.config:
            base_configs = []
            for default in self.config['defaults']:
                base_config_path = config_path.parent / f'{default}.yaml'
                if base_config_path.exists():
                    with open(base_config_path, 'r', encoding='utf-8') as f:
                        base_config = yaml.safe_load(f)
                        base_configs.append(base_config)
            
            # 合并配置（后面的配置覆盖前面的）
            merged_config = {}
            for base_config in base_configs:
                merged_config = self._deep_merge(merged_config, base_config)
            
            # 当前配置覆盖基础配置
            self.config = self._deep_merge(merged_config, self.config)
            
            # 移除defaults字段
            if 'defaults' in self.config:
                del self.config['defaults']
        
        return self.config
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """深度合并字典"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def update_config(self, updates: Dict[str, Any]):
        """更新配置"""
        if self.config is None:
            raise ValueError("Config not loaded. Call load_config() first.")
        
        self.config = self._deep_merge(self.config, updates)
    
    def save_config(self, save_path: str):
        """保存配置"""
        if self.config is None:
            raise ValueError("Config not loaded. Call load_config() first.")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
    
    def get(self, key: str, default=None):
        """获取配置值（支持点号分隔的嵌套键）"""
        if self.config is None:
            raise ValueError("Config not loaded. Call load_config() first.")
        
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """设置配置值（支持点号分隔的嵌套键）"""
        if self.config is None:
            raise ValueError("Config not loaded. Call load_config() first.")
        
        keys = key.split('.')
        config = self.config
        
        # 导航到最后一级的父字典
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # 设置值
        config[keys[-1]] = value
    
    def validate_config(self) -> bool:
        """验证配置完整性"""
        if self.config is None:
            raise ValueError("Config not loaded. Call load_config() first.")
        
        required_keys = [
            'model.vision.backbone',
            'model.text.encoder',
            'data.dataset_path',
            'train.epochs',
            'train.optimizer.lr'
        ]
        
        missing_keys = []
        for key in required_keys:
            if self.get(key) is None:
                missing_keys.append(key)
        
        if missing_keys:
            print(f"Missing required config keys: {missing_keys}")
            return False
        
        return True
    
    def print_config(self, indent: int = 0):
        """打印配置（用于调试）"""
        if self.config is None:
            print("Config not loaded")
            return
        
        self._print_dict(self.config, indent)
    
    def _print_dict(self, d: Dict, indent: int):
        """递归打印字典"""
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                self._print_dict(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")


def load_config(config_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """加载配置文件（支持命令行参数覆盖）。默认使用包内 `speedvqa/configs/default.yaml`。"""
    if config_path is None:
        config_path = get_builtin_default_config_path()
    resolved = str(resolve_config_file(config_path))
    config_manager = ConfigManager()
    config = config_manager.load_config(resolved)
    
    # 命令行参数覆盖
    if kwargs:
        config_manager.config = config
        config_manager.update_config(kwargs)
        config = config_manager.config
    
    # 验证配置
    config_manager.config = config
    if not config_manager.validate_config():
        raise ValueError("Invalid configuration")
    
    return config


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='SpeedVQA Training and Inference')
    
    # 基本参数
    parser.add_argument('--config', type=str, default=get_builtin_default_config_path(),
                       help='Path to config file')
    parser.add_argument('--data', type=str, 
                       help='Path to dataset directory')
    parser.add_argument('--output', type=str, default='./runs',
                       help='Output directory')
    
    # 训练参数
    parser.add_argument('--epochs', type=int,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int,
                       help='Batch size')
    parser.add_argument('--lr', type=float,
                       help='Learning rate')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'auto'],
                       help='Device to use')
    
    # 模型参数
    parser.add_argument('--model', type=str,
                       help='Model architecture')
    parser.add_argument('--pretrained', action='store_true',
                       help='Use pretrained weights')
    
    # 其他参数
    parser.add_argument('--resume', type=str,
                       help='Resume training from checkpoint')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    return parser.parse_args()


def args_to_config_updates(args: argparse.Namespace) -> Dict[str, Any]:
    """将命令行参数转换为配置更新"""
    updates = {}
    
    # 映射命令行参数到配置键
    arg_mapping = {
        'data': 'data.dataset_path',
        'epochs': 'train.epochs',
        'batch_size': 'data.dataloader.batch_size',
        'lr': 'train.optimizer.lr',
        'device': 'hardware.gpu.device_ids',
        'model': 'model.name',
        'resume': 'train.resume'
    }
    
    for arg_name, config_key in arg_mapping.items():
        arg_value = getattr(args, arg_name, None)
        if arg_value is not None:
            # 处理嵌套键
            keys = config_key.split('.')
            nested_dict = updates
            for key in keys[:-1]:
                if key not in nested_dict:
                    nested_dict[key] = {}
                nested_dict = nested_dict[key]
            nested_dict[keys[-1]] = arg_value
    
    return updates


def create_experiment_config(base_config_path: str, experiment_name: str, 
                           updates: Dict[str, Any]) -> str:
    """创建实验配置文件"""
    config_manager = ConfigManager()
    config = config_manager.load_config(str(resolve_config_file(base_config_path)))
    config_manager.config = config
    config_manager.update_config(updates)
    
    # 设置实验名称
    config_manager.set('train.experiment_name', experiment_name)
    
    experiments_dir = get_package_root() / "configs" / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    experiment_config_path = experiments_dir / f'{experiment_name}.yaml'
    config_manager.save_config(str(experiment_config_path))
    
    return str(experiment_config_path)


def validate_paths_in_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """验证配置中的路径是否存在"""
    validation_result = {
        'valid': True,
        'missing_paths': [],
        'warnings': []
    }
    
    # 检查数据路径
    dataset_path = config.get('data', {}).get('dataset_path')
    if dataset_path and not Path(dataset_path).exists():
        validation_result['missing_paths'].append(f"Dataset path: {dataset_path}")
        validation_result['valid'] = False
    
    # 检查恢复训练路径
    resume_path = config.get('train', {}).get('resume')
    if resume_path and not Path(resume_path).exists():
        validation_result['missing_paths'].append(f"Resume checkpoint: {resume_path}")
        validation_result['valid'] = False
    
    # 检查缓存目录
    cache_dir = config.get('data', {}).get('cache_dir')
    if cache_dir:
        cache_path = Path(cache_dir)
        if not cache_path.exists():
            validation_result['warnings'].append(f"Cache directory will be created: {cache_dir}")
    
    return validation_result


# 便捷函数
def get_default_config() -> Dict[str, Any]:
    """获取默认配置"""
    return load_config()


def create_minimal_config(dataset_path: str, output_dir: str = './runs') -> Dict[str, Any]:
    """创建最小配置"""
    return {
        'model': {
            'name': 'speedvqa',
            'vision': {'backbone': 'mobilenet_v3_small', 'pretrained': True, 'feature_dim': 1024},
            'text': {'encoder': 'distilbert-base-uncased', 'max_length': 128, 'feature_dim': 768},
            'fusion': {'method': 'concat', 'hidden_dim': 1792},
            'classifier': {'hidden_dims': [512, 256], 'num_classes': 2}
        },
        'data': {
            'dataset_path': dataset_path,
            'dataloader': {'batch_size': 32, 'num_workers': 4}
        },
        'train': {
            'epochs': 50,
            'save_dir': output_dir,
            'optimizer': {'type': 'adamw', 'lr': 0.001, 'weight_decay': 0.0005}
        }
    }