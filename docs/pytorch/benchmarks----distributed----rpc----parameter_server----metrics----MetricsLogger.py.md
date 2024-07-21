# `.\pytorch\benchmarks\distributed\rpc\parameter_server\metrics\MetricsLogger.py`

```
from .CPUMetric import CPUMetric  # 导入CPUMetric类，用于处理CPU相关的性能指标
from .CUDAMetric import CUDAMetric  # 导入CUDAMetric类，用于处理CUDA相关的性能指标

class MetricsLogger:
    def __init__(self, rank=None):
        self.rank = rank  # 初始化MetricsLogger对象的rank属性

        # 初始化MetricsLogger对象的metrics属性为空字典
        self.metrics = {}

    def record_start(self, type, key, name, cuda):
        # 如果类型(type)已存在于metrics中，并且具有相同的key，则引发RuntimeError
        if type in self.metrics and key in self.metrics[type]:
            raise RuntimeError(f"metric_type={type} with key={key} already exists")

        # 如果cuda为True，且rank为None，则引发RuntimeError
        if cuda:
            if self.rank is None:
                raise RuntimeError("rank is required for cuda")
            metric = CUDAMetric(self.rank, name)  # 创建CUDAMetric对象
        else:
            metric = CPUMetric(name)  # 创建CPUMetric对象

        # 如果类型(type)不在metrics中，则在metrics中创建一个新的空字典
        if type not in self.metrics:
            self.metrics[type] = {}

        self.metrics[type][key] = metric  # 将metric添加到metrics[type][key]中
        metric.record_start()  # 记录metric的开始时间

    def record_end(self, type, key):
        # 如果类型(type)不在metrics中，或者key不在metrics[type]中，则引发RuntimeError
        if type not in self.metrics or key not in self.metrics[type]:
            raise RuntimeError(f"metric_type={type} with key={key} not found")

        # 如果metrics[type][key]的结束时间已存在，则引发RuntimeError
        if self.metrics[type][key].get_end() is not None:
            raise RuntimeError(
                f"end for metric_type={type} with key={key} already exists"
            )

        self.metrics[type][key].record_end()  # 记录metric的结束时间

    def clear_metrics(self):
        self.metrics.clear()  # 清空metrics字典

    def get_metrics(self):
        return self.metrics  # 返回metrics字典

    def get_processed_metrics(self):
        r"""
        处理在基准测试期间记录的性能指标。

        Returns::
            返回一个字典，其键为性能指标，值为经过的时间列表。

        Examples::

            >>> instance = MetricsLogger(rank)
            >>> instance.cuda_record_start("forward_metric_type", "1", "forward_pass")
            >>> instance.cuda_record_end("forward_metric_type", "1")
            >>> instance.cuda_record_start("forward_metric_type", "2", "forward_pass")
            >>> instance.cuda_record_end("forward_metric_type", "2")
            >>> print(instance.metrics)
            {
                "forward_metric_type": {
                    "1": metric1,
                    "2": metric2
                }
            }

            >>> print(instance.get_processed_metrics())
            {
                "forward_metric_type,forward_pass" : [.0429, .0888]
            }
        """
        processed_metrics = {}  # 初始化一个空字典，用于存储处理后的性能指标

        for metric_type in self.metrics.keys():  # 遍历metrics字典的键
            for metric_key in self.metrics[metric_type].keys():  # 遍历metrics字典中每个类型的键
                metric = self.metrics[metric_type][metric_key]  # 获取具体的metric对象

                if isinstance(metric, CUDAMetric):  # 如果metric是CUDAMetric类型的实例
                    metric.synchronize()  # 同步CUDA操作

                metric_name = metric.get_name()  # 获取metric的名称
                elapsed_time = metric.elapsed_time()  # 获取metric的经过时间
                processed_metric_name = f"{metric_type},{metric_name}"  # 构建处理后的指标名称

                if processed_metric_name not in processed_metrics:
                    processed_metrics[processed_metric_name] = []  # 如果处理后的指标名称不存在，则创建空列表

                processed_metrics[processed_metric_name].append(elapsed_time)  # 将经过时间添加到处理后的指标名称对应的列表中

        return processed_metrics  # 返回处理后的性能指标字典
```