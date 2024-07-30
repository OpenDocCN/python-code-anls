# `.\yolov8\ultralytics\models\yolo\pose\__init__.py`

```py
# 导入模块 predict 中的 PosePredictor 类
# 导入模块 train 中的 PoseTrainer 类
# 导入模块 val 中的 PoseValidator 类
from .predict import PosePredictor
from .train import PoseTrainer
from .val import PoseValidator

# 定义 __all__ 变量，包含需要在该模块中公开的类名字符串
__all__ = "PoseTrainer", "PoseValidator", "PosePredictor"
```