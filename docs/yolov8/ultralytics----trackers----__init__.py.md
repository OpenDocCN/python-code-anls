# `.\yolov8\ultralytics\trackers\__init__.py`

```py
# 导入依赖模块以支持Ultralytics YOLO，这里使用AGPL-3.0许可证

# 从当前包中导入BOTSORT类
from .bot_sort import BOTSORT
# 从当前包中导入BYTETracker类
from .byte_tracker import BYTETracker
# 从当前包中导入register_tracker函数
from .track import register_tracker

# 将register_tracker, BOTSORT, BYTETracker三个名称加入到模块的__all__列表中，
# 这样在使用from package import *时，只会导入这三个对象，使导入更加简洁
__all__ = "register_tracker", "BOTSORT", "BYTETracker"  # 允许更简单的导入方式
```