# `.\yolov8\ultralytics\models\sam\model.py`

```py
# Ultralytics YOLO , AGPL-3.0 license
"""
SAM model interface.

This module provides an interface to the Segment Anything Model (SAM) from Ultralytics, designed for real-time image
segmentation tasks. The SAM model allows for promptable segmentation with unparalleled versatility in image analysis,
and has been trained on the SA-1B dataset. It features zero-shot performance capabilities, enabling it to adapt to new
image distributions and tasks without prior knowledge.

Key Features:
    - Promptable segmentation
    - Real-time performance
    - Zero-shot transfer capabilities
    - Trained on SA-1B dataset
"""

from pathlib import Path

from ultralytics.engine.model import Model  # 导入Ultralytics的Model类
from ultralytics.utils.torch_utils import model_info  # 导入模型信息工具函数

from .build import build_sam  # 导入SAM模型构建函数
from .predict import Predictor  # 导入预测器类


class SAM(Model):
    """
    SAM (Segment Anything Model) interface class.

    SAM is designed for promptable real-time image segmentation. It can be used with a variety of prompts such as
    bounding boxes, points, or labels. The model has capabilities for zero-shot performance and is trained on the SA-1B
    dataset.
    """

    def __init__(self, model="sam_b.pt") -> None:
        """
        Initializes the SAM model with a pre-trained model file.

        Args:
            model (str): Path to the pre-trained SAM model file. File should have a .pt or .pth extension.

        Raises:
            NotImplementedError: If the model file extension is not .pt or .pth.
        """
        # 检查模型文件是否是以.pt或.pth结尾，如果不是则抛出异常
        if model and Path(model).suffix not in {".pt", ".pth"}:
            raise NotImplementedError("SAM prediction requires pre-trained *.pt or *.pth model.")
        # 调用父类构造函数初始化模型
        super().__init__(model=model, task="segment")

    def _load(self, weights: str, task=None):
        """
        Loads the specified weights into the SAM model.

        Args:
            weights (str): Path to the weights file.
            task (str, optional): Task name. Defaults to None.
        """
        # 使用指定的权重文件构建SAM模型
        self.model = build_sam(weights)
    def predict(self, source, stream=False, bboxes=None, points=None, labels=None, **kwargs):
        """
        执行给定图像或视频源的分割预测。

        Args:
            source (str): 图像或视频文件的路径，或者是一个 PIL.Image 对象，或者是一个 numpy.ndarray 对象。
            stream (bool, optional): 如果为 True，则启用实时流处理。默认为 False。
            bboxes (list, optional): 提示的分割边界框坐标列表。默认为 None。
            points (list, optional): 提示的分割点列表。默认为 None。
            labels (list, optional): 提示的分割标签列表。默认为 None。

        Returns:
            (list): 模型的预测结果。
        """
        # 设置默认的参数覆盖值
        overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024)
        kwargs.update(overrides)
        # 组装提示信息的字典
        prompts = dict(bboxes=bboxes, points=points, labels=labels)
        # 调用父类的预测方法，并传递参数
        return super().predict(source, stream, prompts=prompts, **kwargs)

    def __call__(self, source=None, stream=False, bboxes=None, points=None, labels=None, **kwargs):
        """
        'predict' 方法的别名。

        Args:
            source (str): 图像或视频文件的路径，或者是一个 PIL.Image 对象，或者是一个 numpy.ndarray 对象。
            stream (bool, optional): 如果为 True，则启用实时流处理。默认为 False。
            bboxes (list, optional): 提示的分割边界框坐标列表。默认为 None。
            points (list, optional): 提示的分割点列表。默认为 None。
            labels (list, optional): 提示的分割标签列表。默认为 None。

        Returns:
            (list): 模型的预测结果。
        """
        # 调用 'predict' 方法进行预测
        return self.predict(source, stream, bboxes, points, labels, **kwargs)

    def info(self, detailed=False, verbose=True):
        """
        记录有关 SAM 模型的信息。

        Args:
            detailed (bool, optional): 如果为 True，则显示关于模型的详细信息。默认为 False。
            verbose (bool, optional): 如果为 True，则在控制台上显示信息。默认为 True。

        Returns:
            (tuple): 包含模型信息的元组。
        """
        # 调用 model_info 函数获取模型信息
        return model_info(self.model, detailed=detailed, verbose=verbose)

    @property
    def task_map(self):
        """
        提供从 'segment' 任务到其相应的 'Predictor' 的映射。

        Returns:
            (dict): 将 'segment' 任务映射到其相应 'Predictor' 的字典。
        """
        # 返回 'segment' 任务到 'Predictor' 类的映射字典
        return {"segment": {"predictor": Predictor}}
```