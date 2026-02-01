"""
SpeedVQA日志系统

集成TensorBoard和Weights & Biases，实现检查点保存和恢复。
"""

import os
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
import torch


class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(self, log_dir: str, experiment_name: str, 
                 use_tensorboard: bool = True, use_wandb: bool = False,
                 wandb_config: Optional[Dict[str, Any]] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        
        # 设置基础日志
        self._setup_basic_logging()
        
        # TensorBoard
        self.use_tensorboard = use_tensorboard
        self.tb_writer = None
        if use_tensorboard:
            self._setup_tensorboard()
        
        # Weights & Biases
        self.use_wandb = use_wandb
        self.wandb_run = None
        if use_wandb:
            self._setup_wandb(wandb_config or {})
        
        # 指标历史
        self.metrics_history = []
        self.current_step = 0
        
        self.logger.info(f"Training logger initialized for experiment: {experiment_name}")
        self.logger.info(f"Log directory: {self.log_dir}")
        self.logger.info(f"TensorBoard: {'enabled' if use_tensorboard else 'disabled'}")
        self.logger.info(f"Weights & Biases: {'enabled' if use_wandb else 'disabled'}")
    
    def _setup_basic_logging(self):
        """设置基础日志"""
        log_file = self.log_dir / f'{self.experiment_name}.log'
        
        # 创建logger
        self.logger = logging.getLogger(f'SpeedVQA-{self.experiment_name}')
        self.logger.setLevel(logging.INFO)
        
        # 避免重复添加handler
        if not self.logger.handlers:
            # 文件handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            
            # 控制台handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # 格式化
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def _setup_tensorboard(self):
        """设置TensorBoard"""
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            tb_log_dir = self.log_dir / 'tensorboard' / self.experiment_name
            self.tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
            self.logger.info(f"TensorBoard logging to: {tb_log_dir}")
            
        except ImportError:
            self.logger.warning("TensorBoard not available. Install with: pip install tensorboard")
            self.use_tensorboard = False
    
    def _setup_wandb(self, wandb_config: Dict[str, Any]):
        """设置Weights & Biases"""
        try:
            import wandb
            
            # 默认配置
            config = {
                'project': wandb_config.get('project', 'speedvqa'),
                'entity': wandb_config.get('entity', None),
                'name': self.experiment_name,
                'dir': str(self.log_dir),
                'reinit': True
            }
            
            # 添加用户配置
            config.update(wandb_config)
            
            self.wandb_run = wandb.init(**config)
            self.logger.info(f"Weights & Biases initialized: {self.wandb_run.url}")
            
        except ImportError:
            self.logger.warning("Weights & Biases not available. Install with: pip install wandb")
            self.use_wandb = False
        except Exception as e:
            self.logger.warning(f"Failed to initialize Weights & Biases: {e}")
            self.use_wandb = False
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], 
                   step: Optional[int] = None, prefix: str = ''):
        """
        记录指标
        
        Args:
            metrics: 指标字典
            step: 步数（可选，默认使用内部计数器）
            prefix: 指标前缀（如 'train/', 'val/'）
        """
        if step is None:
            step = self.current_step
            self.current_step += 1
        
        # 添加前缀
        if prefix and not prefix.endswith('/'):
            prefix += '/'
        
        prefixed_metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
        
        # 记录到历史
        self.metrics_history.append({
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'metrics': prefixed_metrics
        })
        
        # TensorBoard记录
        if self.use_tensorboard and self.tb_writer:
            for key, value in prefixed_metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, step)
        
        # Weights & Biases记录
        if self.use_wandb and self.wandb_run:
            try:
                self.wandb_run.log(prefixed_metrics, step=step)
            except Exception as e:
                self.logger.warning(f"Failed to log to wandb: {e}")
        
        # 基础日志记录
        metrics_str = ', '.join([f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}' 
                                for k, v in prefixed_metrics.items()])
        self.logger.info(f"Step {step}: {metrics_str}")
    
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """记录超参数"""
        # TensorBoard
        if self.use_tensorboard and self.tb_writer:
            # 只记录标量超参数
            scalar_hparams = {k: v for k, v in hparams.items() 
                            if isinstance(v, (int, float, str, bool))}
            try:
                self.tb_writer.add_hparams(scalar_hparams, {})
            except Exception as e:
                self.logger.warning(f"Failed to log hyperparameters to TensorBoard: {e}")
        
        # Weights & Biases
        if self.use_wandb and self.wandb_run:
            try:
                self.wandb_run.config.update(hparams)
            except Exception as e:
                self.logger.warning(f"Failed to log hyperparameters to wandb: {e}")
        
        # 保存到文件
        hparams_file = self.log_dir / f'{self.experiment_name}_hparams.json'
        with open(hparams_file, 'w') as f:
            json.dump(hparams, f, indent=2, default=str)
        
        self.logger.info(f"Hyperparameters saved to: {hparams_file}")
    
    def log_model_graph(self, model, input_sample):
        """记录模型图"""
        if self.use_tensorboard and self.tb_writer:
            try:
                self.tb_writer.add_graph(model, input_sample)
                self.logger.info("Model graph logged to TensorBoard")
            except Exception as e:
                self.logger.warning(f"Failed to log model graph: {e}")
    
    def log_images(self, images: Dict[str, torch.Tensor], step: Optional[int] = None):
        """记录图像"""
        if step is None:
            step = self.current_step
        
        # TensorBoard
        if self.use_tensorboard and self.tb_writer:
            for tag, image in images.items():
                try:
                    self.tb_writer.add_image(tag, image, step)
                except Exception as e:
                    self.logger.warning(f"Failed to log image {tag} to TensorBoard: {e}")
        
        # Weights & Biases
        if self.use_wandb and self.wandb_run:
            try:
                import wandb
                wandb_images = {}
                for tag, image in images.items():
                    if isinstance(image, torch.Tensor):
                        image = image.detach().cpu().numpy()
                    wandb_images[tag] = wandb.Image(image)
                self.wandb_run.log(wandb_images, step=step)
            except Exception as e:
                self.logger.warning(f"Failed to log images to wandb: {e}")
    
    def log_text(self, text_dict: Dict[str, str], step: Optional[int] = None):
        """记录文本"""
        if step is None:
            step = self.current_step
        
        # TensorBoard
        if self.use_tensorboard and self.tb_writer:
            for tag, text in text_dict.items():
                try:
                    self.tb_writer.add_text(tag, text, step)
                except Exception as e:
                    self.logger.warning(f"Failed to log text {tag} to TensorBoard: {e}")
        
        # 基础日志
        for tag, text in text_dict.items():
            self.logger.info(f"{tag}: {text}")
    
    def save_checkpoint(self, checkpoint_data: Dict[str, Any], 
                       checkpoint_path: str, is_best: bool = False):
        """保存检查点"""
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 添加日志信息到检查点
        checkpoint_data['logging_info'] = {
            'experiment_name': self.experiment_name,
            'log_dir': str(self.log_dir),
            'current_step': self.current_step,
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存检查点
        torch.save(checkpoint_data, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        if is_best:
            self.logger.info("✓ New best model saved!")
        
        # 记录到wandb
        if self.use_wandb and self.wandb_run:
            try:
                artifact = self.wandb_run.use_artifact(f"model-{self.experiment_name}:latest", type="model")
                artifact.add_file(str(checkpoint_path))
                self.wandb_run.log_artifact(artifact)
            except Exception as e:
                self.logger.warning(f"Failed to log checkpoint to wandb: {e}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 恢复日志状态
        if 'logging_info' in checkpoint:
            logging_info = checkpoint['logging_info']
            self.current_step = logging_info.get('current_step', 0)
            self.logger.info(f"Checkpoint loaded from: {checkpoint_path}")
            self.logger.info(f"Restored to step: {self.current_step}")
        
        return checkpoint
    
    def save_metrics_history(self):
        """保存指标历史"""
        history_file = self.log_dir / f'{self.experiment_name}_metrics_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        self.logger.info(f"Metrics history saved: {history_file}")
    
    def close(self):
        """关闭日志记录器"""
        # 保存指标历史
        self.save_metrics_history()
        
        # 关闭TensorBoard
        if self.tb_writer:
            self.tb_writer.close()
            self.logger.info("TensorBoard writer closed")
        
        # 关闭Weights & Biases
        if self.use_wandb and self.wandb_run:
            try:
                self.wandb_run.finish()
                self.logger.info("Weights & Biases run finished")
            except Exception as e:
                self.logger.warning(f"Error closing wandb: {e}")
        
        self.logger.info("Training logger closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def setup_logging(log_dir: str, experiment_name: str, 
                 config: Optional[Dict[str, Any]] = None) -> TrainingLogger:
    """
    设置训练日志（便捷函数）
    
    Args:
        log_dir: 日志目录
        experiment_name: 实验名称
        config: 日志配置
        
    Returns:
        logger: 训练日志记录器
    """
    config = config or {}
    
    return TrainingLogger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        use_tensorboard=config.get('tensorboard', True),
        use_wandb=config.get('wandb', {}).get('enabled', False),
        wandb_config=config.get('wandb', {})
    )


if __name__ == '__main__':
    # 测试日志系统
    print("=== SpeedVQA Logging System Test ===")
    
    import tempfile
    import torch
    
    temp_dir = tempfile.mkdtemp()
    
    # 创建日志记录器
    logger = TrainingLogger(
        log_dir=temp_dir,
        experiment_name='test_experiment',
        use_tensorboard=True,
        use_wandb=False  # 避免需要wandb账户
    )
    
    # 测试超参数记录
    hparams = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'model_type': 'speedvqa',
        'optimizer': 'adamw'
    }
    logger.log_hyperparameters(hparams)
    print("✓ Hyperparameters logged")
    
    # 测试指标记录
    for step in range(5):
        train_metrics = {
            'loss': 1.0 - step * 0.1,
            'accuracy': 0.5 + step * 0.1
        }
        val_metrics = {
            'loss': 0.9 - step * 0.08,
            'accuracy': 0.6 + step * 0.08,
            'f1_score': 0.55 + step * 0.09
        }
        
        logger.log_metrics(train_metrics, step, 'train')
        logger.log_metrics(val_metrics, step, 'val')
    
    print("✓ Metrics logged")
    
    # 测试检查点保存
    checkpoint_data = {
        'epoch': 5,
        'model_state_dict': {'dummy': 'data'},
        'optimizer_state_dict': {'dummy': 'data'},
        'metrics': {'accuracy': 0.95}
    }
    
    checkpoint_path = Path(temp_dir) / 'test_checkpoint.pth'
    logger.save_checkpoint(checkpoint_data, str(checkpoint_path), is_best=True)
    print("✓ Checkpoint saved")
    
    # 测试检查点加载
    loaded_checkpoint = logger.load_checkpoint(str(checkpoint_path))
    assert 'logging_info' in loaded_checkpoint
    print("✓ Checkpoint loaded")
    
    # 关闭日志记录器
    logger.close()
    print("✓ Logger closed")
    
    # 清理
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("\n✓ Logging system test completed!")