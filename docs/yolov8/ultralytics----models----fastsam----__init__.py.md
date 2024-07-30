# `.\yolov8\ultralytics\models\fastsam\__init__.py`

```py
# 导入模块中的特定成员：FastSAM 模型、FastSAMPredictor 预测器、FastSAMValidator 验证器
from .model import FastSAM
from .predict import FastSAMPredictor
from .val import FastSAMValidator

# 定义一个列表 __all__，包含需要在模块外部可见的符号名称
__all__ = "FastSAMPredictor", "FastSAM", "FastSAMValidator"
```