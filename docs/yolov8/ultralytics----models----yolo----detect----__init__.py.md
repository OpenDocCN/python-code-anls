# `.\yolov8\ultralytics\models\yolo\detect\__init__.py`

```py
# 导入 DetectionPredictor、DetectionTrainer 和 DetectionValidator 模块
# 这些模块来自当前目录下的 predict.py、train.py 和 val.py 文件
from .predict import DetectionPredictor
from .train import DetectionTrainer
from .val import DetectionValidator

# 将 DetectionPredictor、DetectionTrainer 和 DetectionValidator 导出到外部模块使用
__all__ = "DetectionPredictor", "DetectionTrainer", "DetectionValidator"
```