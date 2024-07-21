# `.\pytorch\benchmarks\distributed\rpc\parameter_server\metrics\CPUMetric.py`

```
import time  # 导入时间模块

from .MetricBase import MetricBase  # 导入自定义的MetricBase类


class CPUMetric(MetricBase):
    def __init__(self, name: str):
        self.name = name  # 初始化实例变量name，用于存储指标名称
        self.start = None  # 初始化实例变量start，用于记录起始时间
        self.end = None  # 初始化实例变量end，用于记录结束时间

    def record_start(self):
        self.start = time.time()  # 记录当前时间作为起始时间

    def record_end(self):
        self.end = time.time()  # 记录当前时间作为结束时间

    def elapsed_time(self):
        if self.start is None:  # 如果起始时间为None，则抛出运行时错误
            raise RuntimeError("start is None")
        if self.end is None:  # 如果结束时间为None，则抛出运行时错误
            raise RuntimeError("end is None")
        return self.end - self.start  # 返回记录的时间差，即经过的时间
```