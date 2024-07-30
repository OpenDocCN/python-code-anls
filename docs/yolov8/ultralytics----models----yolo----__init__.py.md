# `.\yolov8\ultralytics\models\yolo\__init__.py`

```py
# 导入Ultralytics YOLO相关模块和函数
from ultralytics.models.yolo import classify, detect, obb, pose, segment, world

# 从当前目录中导入YOLO和YOLOWorld类
from .model import YOLO, YOLOWorld

# 定义__all__变量，包含需要公开的所有符号名称
__all__ = "classify", "segment", "detect", "pose", "obb", "world", "YOLO", "YOLOWorld"
```