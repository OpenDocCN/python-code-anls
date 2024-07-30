# `.\yolov8\ultralytics\models\yolo\segment\__init__.py`

```py
# 导入模块：从当前包中导入 SegmentationPredictor、SegmentationTrainer 和 SegmentationValidator 类
from .predict import SegmentationPredictor
from .train import SegmentationTrainer
from .val import SegmentationValidator

# __all__ 变量定义：指定在使用 `from package import *` 时应导入的公共接口
__all__ = "SegmentationPredictor", "SegmentationTrainer", "SegmentationValidator"
```