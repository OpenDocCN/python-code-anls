# `.\yolov8\ultralytics\solutions\__init__.py`

```py
# 从当前包导入以下模块和函数
from .ai_gym import AIGym
from .analytics import Analytics
from .distance_calculation import DistanceCalculation
from .heatmap import Heatmap
from .object_counter import ObjectCounter
from .parking_management import ParkingManagement, ParkingPtsSelection
from .queue_management import QueueManager
from .speed_estimation import SpeedEstimator
from .streamlit_inference import inference

# 定义 __all__ 列表，用于指定当前模块导出的公共接口
__all__ = (
    "AIGym",                   # 导出 AIGym 类
    "DistanceCalculation",     # 导出 DistanceCalculation 类
    "Heatmap",                 # 导出 Heatmap 类
    "ObjectCounter",           # 导出 ObjectCounter 类
    "ParkingManagement",       # 导出 ParkingManagement 类
    "ParkingPtsSelection",     # 导出 ParkingPtsSelection 类
    "QueueManager",            # 导出 QueueManager 类
    "SpeedEstimator",          # 导出 SpeedEstimator 类
    "Analytics",               # 导出 Analytics 类
)
```