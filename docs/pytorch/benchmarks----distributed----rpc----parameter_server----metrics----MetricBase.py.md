# `.\pytorch\benchmarks\distributed\rpc\parameter_server\metrics\MetricBase.py`

```
from abc import ABC, abstractmethod

# 定义抽象基类 MetricBase，继承自 ABC（Abstract Base Class）
class MetricBase(ABC):
    # 初始化方法，接收参数 name，初始化实例变量 name, start, end
    def __init__(self, name):
        self.name = name  # 设置实例变量 name，表示指标的名称
        self.start = None  # 初始化开始时间为 None
        self.end = None  # 初始化结束时间为 None

    # 抽象方法：记录指标开始时间，子类需要实现具体逻辑
    @abstractmethod
    def record_start(self):
        return

    # 抽象方法：记录指标结束时间，子类需要实现具体逻辑
    @abstractmethod
    def record_end(self):
        return

    # 抽象方法：计算并返回指标的运行时间，子类需要实现具体逻辑
    @abstractmethod
    def elapsed_time(self):
        return

    # 返回指标的名称
    def get_name(self):
        return self.name

    # 返回指标的结束时间
    def get_end(self):
        return self.end
```