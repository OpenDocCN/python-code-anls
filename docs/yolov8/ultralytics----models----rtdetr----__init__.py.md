# `.\yolov8\ultralytics\models\rtdetr\__init__.py`

```py
# 导入模块，使用相对路径从当前包中导入相关模块和类
from .model import RTDETR
from .predict import RTDETRPredictor
from .val import RTDETRValidator

# 定义 __all__ 变量，指定当前包的公共接口，以便通过 `from package import *` 导入指定符号
__all__ = "RTDETRPredictor", "RTDETRValidator", "RTDETR"
```