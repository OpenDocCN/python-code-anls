# `.\yolov8\ultralytics\models\nas\model.py`

```py
# 从 pathlib 模块导入 Path 类，用于处理文件路径
from pathlib import Path

# 导入 PyTorch 库
import torch

# 从 Ultralytics 引擎的 model 模块中导入 Model 类
from ultralytics.engine.model import Model

# 从 Ultralytics 的 utils 模块中导入下载相关的函数
from ultralytics.utils.downloads import attempt_download_asset

# 从 Ultralytics 的 utils 模块中导入与 PyTorch 相关的工具函数
from ultralytics.utils.torch_utils import model_info

# 导入当前目录下的 predict.py 文件中的 NASPredictor 类
from .predict import NASPredictor

# 导入当前目录下的 val.py 文件中的 NASValidator 类
from .val import NASValidator


class NAS(Model):
    """
    YOLO NAS model for object detection.

    This class provides an interface for the YOLO-NAS models and extends the `Model` class from Ultralytics engine.
    It is designed to facilitate the task of object detection using pre-trained or custom-trained YOLO-NAS models.

    Example:
        ```python
        from ultralytics import NAS

        model = NAS('yolo_nas_s')
        results = model.predict('ultralytics/assets/bus.jpg')
        ```py

    Attributes:
        model (str): Path to the pre-trained model or model name. Defaults to 'yolo_nas_s.pt'.

    Note:
        YOLO-NAS models only support pre-trained models. Do not provide YAML configuration files.
    """

    def __init__(self, model="yolo_nas_s.pt") -> None:
        """Initializes the NAS model with the provided or default 'yolo_nas_s.pt' model."""
        # 断言所提供的模型文件不是 YAML 配置文件，因为 YOLO-NAS 模型仅支持预训练模型
        assert Path(model).suffix not in {".yaml", ".yml"}, "YOLO-NAS models only support pre-trained models."
        # 调用父类 Model 的初始化方法，传入模型路径和任务类型为 "detect"
        super().__init__(model, task="detect")

    def _load(self, weights: str, task=None) -> None:
        """Loads an existing NAS model weights or creates a new NAS model with pretrained weights if not provided."""
        # 动态导入 super_gradients 模块，用于加载模型权重
        import super_gradients

        # 获取权重文件的后缀名
        suffix = Path(weights).suffix
        # 如果后缀为 ".pt"，则加载模型权重
        if suffix == ".pt":
            self.model = torch.load(attempt_download_asset(weights))
        # 如果后缀为空字符串，则根据权重名称获取预训练的 COCO 权重
        elif suffix == "":
            self.model = super_gradients.training.models.get(weights, pretrained_weights="coco")

        # 重写模型的 forward 方法，忽略额外的参数
        def new_forward(x, *args, **kwargs):
            """Ignore additional __call__ arguments."""
            return self.model._original_forward(x)

        # 保存原始的 forward 方法，并将新的 forward 方法赋值给模型
        self.model._original_forward = self.model.forward
        self.model.forward = new_forward

        # 标准化模型的属性
        self.model.fuse = lambda verbose=True: self.model
        self.model.stride = torch.tensor([32])
        self.model.names = dict(enumerate(self.model._class_names))
        self.model.is_fused = lambda: False  # for info()
        self.model.yaml = {}  # for info()
        self.model.pt_path = weights  # for export()
        self.model.task = "detect"  # for export()
    # 定义一个方法用于记录模型信息
    def info(self, detailed=False, verbose=True):
        """
        Logs model info.

        Args:
            detailed (bool): Show detailed information about model.
            verbose (bool): Controls verbosity.
        """
        # 调用 model_info 函数，传入模型对象和其他参数，并返回结果
        return model_info(self.model, detailed=detailed, verbose=verbose, imgsz=640)

    @property
    # 定义一个属性，返回一个字典，将任务映射到相应的预测器和验证器类
    def task_map(self):
        """Returns a dictionary mapping tasks to respective predictor and validator classes."""
        # 返回包含映射关系的字典
        return {"detect": {"predictor": NASPredictor, "validator": NASValidator}}
```