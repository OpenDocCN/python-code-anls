# `.\pytorch\benchmarks\distributed\rpc\parameter_server\metrics\CUDAMetric.py`

```py
import torch  # 导入PyTorch库

from .MetricBase import MetricBase  # 导入自定义的MetricBase类


class CUDAMetric(MetricBase):
    def __init__(self, rank: int, name: str):
        self.rank = rank  # 初始化CUDA设备的排名
        self.name = name  # 初始化指标的名称
        self.start = None  # 初始化开始事件为None
        self.end = None  # 初始化结束事件为None

    def record_start(self):
        self.start = torch.cuda.Event(enable_timing=True)  # 创建一个CUDA事件对象用于记录开始时间
        with torch.cuda.device(self.rank):  # 设置当前CUDA设备为指定的rank
            self.start.record()  # 记录开始时间

    def record_end(self):
        self.end = torch.cuda.Event(enable_timing=True)  # 创建一个CUDA事件对象用于记录结束时间
        with torch.cuda.device(self.rank):  # 设置当前CUDA设备为指定的rank
            self.end.record()  # 记录结束时间

    def elapsed_time(self):
        if not self.start.query():  # 如果开始事件尚未完成
            raise RuntimeError("start event did not complete")  # 抛出运行时错误
        if not self.end.query():  # 如果结束事件尚未完成
            raise RuntimeError("end event did not complete")  # 抛出运行时错误
        return self.start.elapsed_time(self.end)  # 返回开始和结束事件之间的经过时间

    def synchronize(self):
        self.start.synchronize()  # 同步开始事件
        self.end.synchronize()  # 同步结束事件
```