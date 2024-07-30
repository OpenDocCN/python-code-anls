# `.\yolov8\ultralytics\models\yolo\classify\__init__.py`

```py
# 导入分类任务相关模块和类
from ultralytics.models.yolo.classify.predict import ClassificationPredictor
from ultralytics.models.yolo.classify.train import ClassificationTrainer
from ultralytics.models.yolo.classify.val import ClassificationValidator

# 将这些类添加到模块的公开接口中，方便其他模块导入时使用
__all__ = "ClassificationPredictor", "ClassificationTrainer", "ClassificationValidator"
```