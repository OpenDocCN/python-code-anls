# `.\yolov8\ultralytics\models\sam\__init__.py`

```py
# 导入模块 `SAM` 和 `Predictor`，它们来自当前目录下的 `model.py` 和 `predict.py`
from .model import SAM
from .predict import Predictor

# 定义公开接口 `__all__`，包含字符串 "SAM" 和 "Predictor" 的元组或列表
__all__ = "SAM", "Predictor"  # tuple or list
```