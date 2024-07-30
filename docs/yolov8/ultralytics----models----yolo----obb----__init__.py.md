# `.\yolov8\ultralytics\models\yolo\obb\__init__.py`

```py
# 导入本地模块中的OBBPredictor类、OBBTrainer类和OBBValidator类
from .predict import OBBPredictor
from .train import OBBTrainer
from .val import OBBValidator

# 将OBBPredictor、OBBTrainer和OBBValidator这三个符号添加到__all__列表中，使它们可以被 from module import * 导入
__all__ = "OBBPredictor", "OBBTrainer", "OBBValidator"
```