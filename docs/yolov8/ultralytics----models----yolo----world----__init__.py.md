# `.\yolov8\ultralytics\models\yolo\world\__init__.py`

```py
# 导入WorldTrainer类从.train模块
from .train import WorldTrainer

# 将WorldTrainer添加到当前模块的__all__列表中，使其在使用import *时被导入
__all__ = ["WorldTrainer"]
```