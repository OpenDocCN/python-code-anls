# `.\yolov8\ultralytics\models\rtdetr\model.py`

```
# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Interface for Baidu's RT-DETR, a Vision Transformer-based real-time object detector. RT-DETR offers real-time
performance and high accuracy, excelling in accelerated backends like CUDA with TensorRT. It features an efficient
hybrid encoder and IoU-aware query selection for enhanced detection accuracy.

For more information on RT-DETR, visit: https://arxiv.org/pdf/2304.08069.pdf
"""

from ultralytics.engine.model import Model  # 导入 Model 类
from ultralytics.nn.tasks import RTDETRDetectionModel  # 导入 RTDETRDetectionModel 类

from .predict import RTDETRPredictor  # 导入 RTDETRPredictor 类
from .train import RTDETRTrainer  # 导入 RTDETRTrainer 类
from .val import RTDETRValidator  # 导入 RTDETRValidator 类


class RTDETR(Model):
    """
    Interface for Baidu's RT-DETR model. This Vision Transformer-based object detector provides real-time performance
    with high accuracy. It supports efficient hybrid encoding, IoU-aware query selection, and adaptable inference speed.

    Attributes:
        model (str): Path to the pre-trained model. Defaults to 'rtdetr-l.pt'.
    """

    def __init__(self, model="rtdetr-l.pt") -> None:
        """
        Initializes the RT-DETR model with the given pre-trained model file. Supports .pt and .yaml formats.

        Args:
            model (str): Path to the pre-trained model. Defaults to 'rtdetr-l.pt'.

        Raises:
            NotImplementedError: If the model file extension is not 'pt', 'yaml', or 'yml'.
        """
        # 调用父类 Model 的构造函数，初始化模型和任务为 'detect'
        super().__init__(model=model, task="detect")

    @property
    def task_map(self) -> dict:
        """
        Returns a task map for RT-DETR, associating tasks with corresponding Ultralytics classes.

        Returns:
            dict: A dictionary mapping task names to Ultralytics task classes for the RT-DETR model.
        """
        # 返回一个任务映射字典，将任务名称映射到相应的 Ultralytics 类
        return {
            "detect": {
                "predictor": RTDETRPredictor,  # 预测器类
                "validator": RTDETRValidator,  # 验证器类
                "trainer": RTDETRTrainer,      # 训练器类
                "model": RTDETRDetectionModel,  # 检测模型类
            }
        }
```