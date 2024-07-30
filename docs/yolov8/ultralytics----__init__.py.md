# `.\yolov8\ultralytics\__init__.py`

```
# Ultralytics YOLO 🚀, AGPL-3.0 license

# 定义模块版本号
__version__ = "8.2.69"

# 导入操作系统模块
import os

# 设置环境变量（放置在导入语句之前）
# 设置 OpenMP 线程数为 1，以减少训练过程中的 CPU 使用率
os.environ["OMP_NUM_THREADS"] = "1"

# 从 ultralytics.data.explorer.explorer 模块中导入 Explorer 类
from ultralytics.data.explorer.explorer import Explorer
# 从 ultralytics.models 模块中导入 NAS、RTDETR、SAM、YOLO、FastSAM、YOLOWorld 类
from ultralytics.models import NAS, RTDETR, SAM, YOLO, FastSAM, YOLOWorld
# 从 ultralytics.utils 模块中导入 ASSETS 和 SETTINGS
from ultralytics.utils import ASSETS, SETTINGS
# 从 ultralytics.utils.checks 模块中导入 check_yolo 函数，并将其命名为 checks
from ultralytics.utils.checks import check_yolo as checks
# 从 ultralytics.utils.downloads 模块中导入 download 函数
from ultralytics.utils.downloads import download

# 将 SETTINGS 赋值给 settings 变量
settings = SETTINGS

# 定义 __all__ 变量，包含了可以通过 `from package import *` 导入的名字
__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "YOLOWorld",
    "NAS",
    "SAM",
    "FastSAM",
    "RTDETR",
    "checks",
    "download",
    "settings",
    "Explorer",
)
```