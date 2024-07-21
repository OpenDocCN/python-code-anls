# `.\pytorch\torch\_C\_monitor.pyi`

```
# Defined in torch/csrc/monitor/python_init.cpp

# 导入 datetime 模块，用于处理日期和时间
import datetime
# 导入 Enum 类型，用于定义枚举类型 Aggregation
from enum import Enum
# 导入 Callable 类型，用于定义可调用对象类型
from typing import Callable

# 定义 Aggregation 枚举，包含不同的聚合方式
class Aggregation(Enum):
    VALUE = ...
    MEAN = ...
    COUNT = ...
    SUM = ...
    MAX = ...
    MIN = ...

# 定义 Stat 类，表示统计信息
class Stat:
    name: str  # 统计项名称
    count: int  # 统计次数

    # 初始化 Stat 对象
    def __init__(
        self,
        name: str,  # 统计项名称
        aggregations: list[Aggregation],  # 支持的聚合方式列表
        window_size: int,  # 窗口大小
        max_samples: int = -1,  # 最大样本数，默认为无限制
    ) -> None: ...

    # 向 Stat 对象添加数据
    def add(self, v: float) -> None: ...

    # 获取 Stat 对象的统计数据，返回聚合方式到对应值的字典
    def get(self) -> dict[Aggregation, float]: ...

# 定义 Event 类，表示事件
class Event:
    name: str  # 事件名称
    timestamp: datetime.datetime  # 时间戳
    data: dict[str, int | float | bool | str]  # 事件数据字典

    # 初始化 Event 对象
    def __init__(
        self,
        name: str,  # 事件名称
        timestamp: datetime.datetime,  # 时间戳
        data: dict[str, int | float | bool | str],  # 事件数据字典
    ) -> None: ...

# 定义 log_event 函数，用于记录事件
def log_event(e: Event) -> None: ...

# 定义 EventHandlerHandle 类型
class EventHandlerHandle: ...

# 定义 register_event_handler 函数，注册事件处理函数
def register_event_handler(handler: Callable[[Event], None]) -> EventHandlerHandle: ...

# 定义 unregister_event_handler 函数，取消注册事件处理函数
def unregister_event_handler(handle: EventHandlerHandle) -> None: ...
```