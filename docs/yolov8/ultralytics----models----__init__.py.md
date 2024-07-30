# `.\yolov8\ultralytics\models\__init__.py`

```py
# 导入Ultralytics YOLO相关模块，这些模块都遵循AGPL-3.0许可证

# 从当前目录下的fastsam模块中导入FastSAM类
from .fastsam import FastSAM

# 从当前目录下的nas模块中导入NAS类
from .nas import NAS

# 从当前目录下的rtdetr模块中导入RTDETR类
from .rtdetr import RTDETR

# 从当前目录下的sam模块中导入SAM类
from .sam import SAM

# 从当前目录下的yolo模块中导入YOLO类和YOLOWorld类
from .yolo import YOLO, YOLOWorld

# 定义__all__变量，使得YOLO、RTDETR、SAM、FastSAM、NAS、YOLOWorld可以直接通过简单导入方式使用
__all__ = "YOLO", "RTDETR", "SAM", "FastSAM", "NAS", "YOLOWorld"  # allow simpler import
```