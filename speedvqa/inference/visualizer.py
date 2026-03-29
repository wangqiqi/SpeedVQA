"""
推理结果可视化器 (ResultVisualizer)

支持推理结果的可视化展示，包括：
- 答案和置信度显示
- 关键区域标注可视化
- 推理时间统计显示
- 批量可视化支持
- 摘要报告生成
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import logging
from dataclasses import dataclass, field
import json
from datetime import datetime


@dataclass
class VisualizationResult:
    """可视化结果数据结构"""
    image_path: str  # 输出图像路径
    success: bool  # 是否成功
    error_message: Optional[str] = None  # 错误信息
    annotations: Dict[str, Any] = field(default_factory=dict)  # 标注信息
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据


class ResultVisualizer:
    """
    推理结果可视化器
    
    支持将推理结果可视化为带有标注的图像，包括：
    - 答案和置信度显示
    - 关键区域标注
    - 推理时间统计
    - 批量可视化
    - 摘要报告生成
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化结果可视化器
        
        Args:
            config: 配置字典，支持以下参数：
                - font_size: 字体大小 (默认: 20)
                - text_color: 文本颜色RGB元组 (默认: (0, 255, 0))
                - box_color: 边界框颜色RGB元组 (默认: (0, 255, 0))
                - box_thickness: 边界框厚度 (默认: 2)
                - confidence_threshold: 置信度阈值 (默认: 0.5)
                - region_color: 关键区域颜色RGB元组 (默认: (255, 0, 0))
                - region_thickness: 关键区域边界厚度 (默认: 3)
                - low_confidence_color: 低置信度警告颜色 (默认: (0, 0, 255))
        """
        self.config = config or {}
        self.logger = self._setup_logger()
        
        # 可视化配置
        self.font_size = self.config.get('font_size', 20)
        self.text_color = self.config.get('text_color', (0, 255, 0))  # 绿色
        self.box_color = self.config.get('box_color', (0, 255, 0))  # 绿色
        self.box_thickness = self.config.get('box_thickness', 2)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.region_color = self.config.get('region_color', (255, 0, 0))  # 红色
        self.region_thickness = self.config.get('region_thickness', 3)
        self.low_confidence_color = self.config.get('low_confidence_color', (0, 0, 255))  # 蓝色
        
        self.logger.info("ResultVisualizer initialized")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('ResultVisualizer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def visualize_inference_result(self, 
                                   image: Union[np.ndarray, Image.Image, str],
                                   answer: str,
                                   confidence: float,
                                   question: str,
                                   inference_time_ms: float,
                                   region_coords: Optional[Tuple[float, float, float, float]] = None,
                                   output_path: Optional[str] = None) -> VisualizationResult:
        """
        可视化单个推理结果
        
        Args:
            image: 输入图像 (numpy array、PIL Image或图像路径)
            answer: 推理答案 (YES/NO)
            confidence: 置信度 [0, 1]
            question: 问题文本
            inference_time_ms: 推理时间(毫秒)
            region_coords: 关键区域坐标 (x1, y1, x2, y2)，可选
            output_path: 输出图像路径
            
        Returns:
            VisualizationResult: 可视化结果
        """
        try:
            # 验证输入参数
            if answer not in ['YES', 'NO']:
                raise ValueError(f"Answer must be 'YES' or 'NO', got {answer}")
            
            if not (0 <= confidence <= 1):
                raise ValueError(f"Confidence must be in [0, 1], got {confidence}")
            
            if not isinstance(question, str) or len(question) == 0:
                raise ValueError(f"Question must be non-empty string, got {question}")
            
            if inference_time_ms <= 0:
                raise ValueError(f"Inference time must be positive, got {inference_time_ms}")
            
            # 加载图像
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image.astype(np.uint8))
            
            # 确保是PIL Image
            if not isinstance(image, Image.Image):
                raise ValueError(f"Unsupported image type: {type(image)}")
            
            # 创建副本以避免修改原始图像
            image_copy = image.copy()
            draw = ImageDraw.Draw(image_copy)
            
            # 尝试加载字体，如果失败则使用默认字体
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", self.font_size)
                font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", self.font_size - 4)
            except OSError:
                font = ImageFont.load_default()
                font_small = font
            
            # 绘制关键区域标注（如果提供）
            if region_coords is not None:
                self._draw_region_annotation(draw, region_coords, image_copy.size)
            
            # 准备文本信息
            text_lines = [
                f"Q: {question}",
                f"A: {answer}",
                f"Conf: {confidence:.1%}",
                f"Time: {inference_time_ms:.1f}ms"
            ]
            
            # 绘制文本背景和文本
            y_offset = 10
            for line in text_lines:
                # 获取文本大小
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # 选择背景颜色
                if confidence < self.confidence_threshold and "Conf" in line:
                    bg_color = (50, 50, 50)  # 深灰色背景用于低置信度
                else:
                    bg_color = (0, 0, 0)  # 黑色背景
                
                # 绘制背景矩形
                draw.rectangle(
                    [(5, y_offset), (text_width + 15, y_offset + text_height + 10)],
                    fill=bg_color,
                    outline=self.text_color,
                    width=1
                )
                
                # 选择文本颜色
                text_color = self.low_confidence_color if (confidence < self.confidence_threshold and "Conf" in line) else self.text_color
                
                # 绘制文本
                draw.text((10, y_offset + 5), line, font=font, fill=text_color)
                
                y_offset += text_height + 15
            
            # 如果置信度低于阈值，添加警告
            if confidence < self.confidence_threshold:
                warning_text = "⚠ Low Confidence"
                bbox = draw.textbbox((0, 0), warning_text, font=font_small)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                draw.rectangle(
                    [(5, y_offset), (text_width + 15, y_offset + text_height + 10)],
                    fill=(0, 0, 0),
                    outline=self.low_confidence_color,
                    width=2
                )
                draw.text((10, y_offset + 5), warning_text, font=font_small, fill=self.low_confidence_color)
            
            # 保存输出图像
            if output_path is None:
                output_path = "visualization_result.jpg"
            
            # 确保输出目录存在
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            image_copy.save(output_path, quality=95)
            
            # 创建结果对象
            result = VisualizationResult(
                image_path=output_path,
                success=True,
                error_message=None,
                annotations={
                    'answer': answer,
                    'confidence': confidence,
                    'question': question,
                    'inference_time_ms': inference_time_ms,
                    'region_coords': region_coords
                },
                metadata={
                    'image_size': image_copy.size,
                    'format': 'JPEG',
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            self.logger.info(f"Visualization saved to {output_path}")
            return result
        
        except Exception as e:
            self.logger.error(f"Visualization failed: {str(e)}")
            return VisualizationResult(
                image_path="",
                success=False,
                error_message=str(e),
                annotations={},
                metadata={}
            )
    
    def _draw_region_annotation(self, draw: ImageDraw.ImageDraw, 
                               region_coords: Tuple[float, float, float, float],
                               image_size: Tuple[int, int]) -> None:
        """
        绘制关键区域标注
        
        Args:
            draw: PIL ImageDraw对象
            region_coords: 关键区域坐标 (x1, y1, x2, y2)，范围[0, 1]
            image_size: 图像大小 (width, height)
        """
        try:
            x1, y1, x2, y2 = region_coords
            
            # 验证坐标范围
            if not all(0 <= c <= 1 for c in region_coords):
                self.logger.warning(f"Region coordinates out of range [0, 1]: {region_coords}")
                return
            
            # 转换为像素坐标
            width, height = image_size
            px1 = int(x1 * width)
            py1 = int(y1 * height)
            px2 = int(x2 * width)
            py2 = int(y2 * height)
            
            # 绘制关键区域边界框
            draw.rectangle(
                [(px1, py1), (px2, py2)],
                outline=self.region_color,
                width=self.region_thickness
            )
            
            # 绘制关键区域标签
            label_text = "Key Region"
            bbox = draw.textbbox((0, 0), label_text, font=ImageFont.load_default())
            label_width = bbox[2] - bbox[0]
            label_height = bbox[3] - bbox[1]
            
            # 在关键区域上方绘制标签
            label_y = max(0, py1 - label_height - 5)
            draw.rectangle(
                [(px1, label_y), (px1 + label_width + 10, label_y + label_height + 5)],
                fill=(0, 0, 0),
                outline=self.region_color,
                width=1
            )
            draw.text((px1 + 5, label_y + 2), label_text, font=ImageFont.load_default(), fill=self.region_color)
        
        except Exception as e:
            self.logger.warning(f"Failed to draw region annotation: {str(e)}")
    
    def visualize_batch_results(self,
                               images: List[Union[np.ndarray, Image.Image, str]],
                               answers: List[str],
                               confidences: List[float],
                               questions: List[str],
                               inference_times_ms: List[float],
                               region_coords_list: Optional[List[Optional[Tuple[float, float, float, float]]]] = None,
                               output_dir: Optional[str] = None) -> List[VisualizationResult]:
        """
        可视化批量推理结果
        
        Args:
            images: 输入图像列表
            answers: 推理答案列表
            confidences: 置信度列表
            questions: 问题文本列表
            inference_times_ms: 推理时间列表
            region_coords_list: 关键区域坐标列表，可选
            output_dir: 输出目录
            
        Returns:
            List[VisualizationResult]: 可视化结果列表
        """
        if not (len(images) == len(answers) == len(confidences) == len(questions) == len(inference_times_ms)):
            raise ValueError("All input lists must have the same length")
        
        if output_dir is None:
            output_dir = "visualizations"
        
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 如果没有提供关键区域坐标，使用None列表
        if region_coords_list is None:
            region_coords_list = [None] * len(images)
        
        if len(region_coords_list) != len(images):
            raise ValueError("region_coords_list must have the same length as other lists")
        
        results = []
        for i, (image, answer, confidence, question, inference_time, region_coords) in enumerate(
            zip(images, answers, confidences, questions, inference_times_ms, region_coords_list)
        ):
            output_path = Path(output_dir) / f"result_{i:04d}.jpg"
            result = self.visualize_inference_result(
                image, answer, confidence, question, inference_time,
                region_coords=region_coords,
                output_path=str(output_path)
            )
            results.append(result)
        
        self.logger.info(f"Batch visualization completed: {len(results)} results saved to {output_dir}")
        return results
    
    def create_summary_report(self,
                             results: List[VisualizationResult],
                             output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        创建可视化结果摘要报告
        
        Args:
            results: 可视化结果列表
            output_path: 输出报告路径
            
        Returns:
            Dict[str, Any]: 摘要报告
        """
        total_results = len(results)
        successful_results = sum(1 for r in results if r.success)
        failed_results = total_results - successful_results
        
        # 收集统计信息
        confidences = []
        inference_times = []
        answers_count = {'YES': 0, 'NO': 0}
        low_confidence_count = 0
        
        for result in results:
            if result.success and result.annotations:
                confidence = result.annotations.get('confidence', 0)
                confidences.append(confidence)
                inference_times.append(result.annotations.get('inference_time_ms', 0))
                
                answer = result.annotations.get('answer', 'UNKNOWN')
                if answer in answers_count:
                    answers_count[answer] += 1
                
                if confidence < self.confidence_threshold:
                    low_confidence_count += 1
        
        # 计算统计指标
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_results': total_results,
            'successful_results': successful_results,
            'failed_results': failed_results,
            'success_rate': successful_results / total_results if total_results > 0 else 0,
            'answer_distribution': answers_count,
            'low_confidence_count': low_confidence_count,
            'low_confidence_rate': low_confidence_count / successful_results if successful_results > 0 else 0,
            'statistics': {
                'avg_confidence': float(np.mean(confidences)) if confidences else 0,
                'min_confidence': float(np.min(confidences)) if confidences else 0,
                'max_confidence': float(np.max(confidences)) if confidences else 0,
                'std_confidence': float(np.std(confidences)) if confidences else 0,
                'avg_inference_time_ms': float(np.mean(inference_times)) if inference_times else 0,
                'min_inference_time_ms': float(np.min(inference_times)) if inference_times else 0,
                'max_inference_time_ms': float(np.max(inference_times)) if inference_times else 0,
                'std_inference_time_ms': float(np.std(inference_times)) if inference_times else 0,
                'total_inference_time_ms': float(np.sum(inference_times)) if inference_times else 0,
            }
        }
        
        # 保存报告
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            self.logger.info(f"Summary report saved to {output_path}")
        
        return report
    
    def get_visualization_info(self) -> Dict[str, Any]:
        """获取可视化器信息"""
        return {
            'font_size': self.font_size,
            'text_color': self.text_color,
            'box_color': self.box_color,
            'box_thickness': self.box_thickness,
            'confidence_threshold': self.confidence_threshold,
            'region_color': self.region_color,
            'region_thickness': self.region_thickness,
            'low_confidence_color': self.low_confidence_color
        }
    
    def visualize_performance_statistics(self,
                                        results: List[VisualizationResult],
                                        output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        生成性能统计可视化
        
        Args:
            results: 可视化结果列表
            output_path: 输出统计文件路径
            
        Returns:
            Dict[str, Any]: 性能统计信息
        """
        report = self.create_summary_report(results, output_path)
        
        # 生成可读的统计摘要
        summary_text = f"""
=== Visualization Performance Report ===
Generated: {report['timestamp']}

Results Summary:
  Total Results: {report['total_results']}
  Successful: {report['successful_results']}
  Failed: {report['failed_results']}
  Success Rate: {report['success_rate']:.1%}

Answer Distribution:
  YES: {report['answer_distribution']['YES']}
  NO: {report['answer_distribution']['NO']}

Confidence Statistics:
  Average: {report['statistics']['avg_confidence']:.3f}
  Min: {report['statistics']['min_confidence']:.3f}
  Max: {report['statistics']['max_confidence']:.3f}
  Std Dev: {report['statistics']['std_confidence']:.3f}
  Low Confidence (<{self.confidence_threshold}): {report['low_confidence_count']} ({report['low_confidence_rate']:.1%})

Inference Time Statistics (ms):
  Average: {report['statistics']['avg_inference_time_ms']:.2f}
  Min: {report['statistics']['min_inference_time_ms']:.2f}
  Max: {report['statistics']['max_inference_time_ms']:.2f}
  Std Dev: {report['statistics']['std_inference_time_ms']:.2f}
  Total: {report['statistics']['total_inference_time_ms']:.2f}
"""
        
        self.logger.info(summary_text)
        
        return report
    
    def export_results_to_csv(self,
                             results: List[VisualizationResult],
                             output_path: str) -> None:
        """
        将可视化结果导出为CSV格式
        
        Args:
            results: 可视化结果列表
            output_path: 输出CSV文件路径
        """
        try:
            import csv
            
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # 写入表头
                writer.writerow([
                    'Index', 'Success', 'Image Path', 'Answer', 'Confidence',
                    'Question', 'Inference Time (ms)', 'Region Coords', 'Error Message'
                ])
                
                # 写入数据
                for i, result in enumerate(results):
                    writer.writerow([
                        i,
                        result.success,
                        result.image_path,
                        result.annotations.get('answer', 'N/A') if result.annotations else 'N/A',
                        result.annotations.get('confidence', 'N/A') if result.annotations else 'N/A',
                        result.annotations.get('question', 'N/A') if result.annotations else 'N/A',
                        result.annotations.get('inference_time_ms', 'N/A') if result.annotations else 'N/A',
                        result.annotations.get('region_coords', 'N/A') if result.annotations else 'N/A',
                        result.error_message or ''
                    ])
            
            self.logger.info(f"Results exported to CSV: {output_path}")
        
        except Exception as e:
            self.logger.error(f"Failed to export results to CSV: {str(e)}")


def build_result_visualizer(config: Optional[Dict[str, Any]] = None) -> ResultVisualizer:
    """
    构建结果可视化器（YOLO风格的简单接口）
    
    Args:
        config: 配置字典
        
    Returns:
        ResultVisualizer: 结果可视化器实例
    """
    visualizer = ResultVisualizer(config)
    
    # 打印可视化器信息
    info = visualizer.get_visualization_info()
    print("\n=== Result Visualizer Info ===")
    print(f"Font Size: {info['font_size']}")
    print(f"Text Color: {info['text_color']}")
    print(f"Box Color: {info['box_color']}")
    print(f"Confidence Threshold: {info['confidence_threshold']}")
    
    return visualizer


if __name__ == '__main__':
    # 测试结果可视化器
    print("Result Visualizer Module")
    print("=" * 50)
    print("\nUsage Example:")
    print("  from speedvqa.inference.visualizer import ResultVisualizer")
    print("  visualizer = ResultVisualizer()")
    print("  result = visualizer.visualize_inference_result(")
    print("      'roi_image.jpg', 'YES', 0.95, 'Is there a person?', 25.5")
    print("  )")
