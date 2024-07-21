# `.\pytorch\torch\distributed\elastic\metrics\api.py`

```
#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# 导入必要的库
import abc              # 导入抽象基类模块
import time             # 导入时间模块
from collections import namedtuple   # 导入命名元组模块
from functools import wraps          # 导入装饰器模块
from typing import Dict, Optional    # 导入类型提示模块中的字典和可选类型
from typing_extensions import deprecated  # 导入已弃用标记模块


__all__ = [
    "MetricsConfig",
    "MetricHandler",
    "ConsoleMetricHandler",
    "NullMetricHandler",
    "MetricStream",
    "configure",
    "getStream",
    "prof",
    "profile",
    "put_metric",
    "publish_metric",
    "get_elapsed_time_ms",
    "MetricData",
]

# 定义命名元组 MetricData，用于存储指标数据
MetricData = namedtuple("MetricData", ["timestamp", "group_name", "name", "value"])


class MetricsConfig:
    __slots__ = ["params"]

    def __init__(self, params: Optional[Dict[str, str]] = None):
        self.params = params
        if self.params is None:
            self.params = {}


class MetricHandler(abc.ABC):
    @abc.abstractmethod
    def emit(self, metric_data: MetricData):
        # 抽象方法，用于发射（处理）指标数据
        pass


class ConsoleMetricHandler(MetricHandler):
    def emit(self, metric_data: MetricData):
        # 实现 MetricHandler 的 emit 方法，打印指标数据到控制台
        print(
            f"[{metric_data.timestamp}][{metric_data.group_name}]: {metric_data.name}={metric_data.value}"
        )


class NullMetricHandler(MetricHandler):
    def emit(self, metric_data: MetricData):
        # 实现 MetricHandler 的 emit 方法，但不执行任何操作（空实现）
        pass


class MetricStream:
    def __init__(self, group_name: str, handler: MetricHandler):
        # 初始化 MetricStream 对象，指定组名和指标处理器
        self.group_name = group_name
        self.handler = handler

    def add_value(self, metric_name: str, metric_value: int):
        # 向指标流中添加指标名称和值
        self.handler.emit(
            MetricData(time.time(), self.group_name, metric_name, metric_value)
        )


# 存储不同组名的指标处理器映射
_metrics_map: Dict[str, MetricHandler] = {}
# 默认的指标处理器为 NullMetricHandler
_default_metrics_handler: MetricHandler = NullMetricHandler()


# 配置指标处理器，可以指定组名，如果组名为 None，则配置默认指标处理器
# pyre-fixme[9]: group has type `str`; used as `None`.
def configure(handler: MetricHandler, group: Optional[str] = None):
    if group is None:
        global _default_metrics_handler
        # 将默认指标处理器设置为给定的处理器
        # pyre-fixme[9]: _default_metrics_handler has type `NullMetricHandler`; used
        #  as `MetricHandler`.
        _default_metrics_handler = handler
    else:
        # 将指定组名的指标处理器设置为给定的处理器
        _metrics_map[group] = handler


# 获取指定组名的指标流对象
def getStream(group: str):
    if group in _metrics_map:
        handler = _metrics_map[group]
    else:
        handler = _default_metrics_handler
    return MetricStream(group, handler)


# 获取函数的完全限定名作为指标名称
def _get_metric_name(fn):
    qualname = fn.__qualname__
    split = qualname.split(".")
    if len(split) == 1:
        module = fn.__module__
        if module:
            return module.split(".")[-1] + "." + split[0]
        else:
            return split[0]
    else:
        return qualname


# 用于性能分析的装饰器函数，发布函数执行的持续时间、调用次数、成功次数和失败次数的指标
def prof(fn=None, group: str = "torchelastic"):
    r"""
    @profile decorator publishes duration.ms, count, success, failure metrics for the function that it decorates.
    """
    The metric name defaults to the qualified name (``class_name.def_name``) of the function.
    If the function does not belong to a class, it uses the leaf module name instead.

    Usage

    ::

     @metrics.prof
     def x():
         pass

     @metrics.prof(group="agent")
     def y():
         pass
    """

    # 定义一个装饰器函数 `prof`
    def wrap(f):
        # 使用 functools.wraps 来保留原始函数的元数据
        @wraps(f)
        def wrapper(*args, **kwargs):
            # 获取函数的度量名
            key = _get_metric_name(f)
            try:
                # 记录函数执行开始时间
                start = time.time()
                # 执行原始函数
                result = f(*args, **kwargs)
                # 记录成功的度量信息
                put_metric(f"{key}.success", 1, group)
            except Exception:
                # 记录失败的度量信息
                put_metric(f"{key}.failure", 1, group)
                raise
            finally:
                # 记录函数执行时间
                put_metric(f"{key}.duration.ms", get_elapsed_time_ms(start), group)  # type: ignore[possibly-undefined]
            return result

        return wrapper

    # 如果传入了函数 `fn`，则返回其被 `wrap` 装饰后的函数
    if fn:
        return wrap(fn)
    else:
        return wrap
# 使用装饰器声明一个已废弃的函数，提醒使用者改用 `@prof` 装饰器，可能会在未来发出警告
@deprecated("Deprecated, use `@prof` instead", category=FutureWarning)
# 定义一个名为 `profile` 的函数，该函数用来添加延迟和成功/失败指标到任意给定的函数中
def profile(group=None):
    """
    @profile decorator adds latency and success/failure metrics to any given function.

    Usage

    ::

     @metrics.profile("my_metric_group")
     def some_function(<arguments>):
    """

    # 内部函数 `wrap`，用来包装被装饰的函数
    def wrap(func):
        # 使用被装饰函数的名称和文档字符串来装饰 `wrapper` 函数
        @wraps(func)
        # `wrapper` 函数接受任意参数并捕获执行过程中的异常
        def wrapper(*args, **kwargs):
            try:
                # 记录函数执行开始时间
                start_time = time.time()
                # 执行被装饰的函数，并获取其返回值
                result = func(*args, **kwargs)
                # 发布成功指标，记录函数执行成功的次数
                publish_metric(group, f"{func.__name__}.success", 1)
            except Exception:
                # 发布失败指标，记录函数执行失败的次数
                publish_metric(group, f"{func.__name__}.failure", 1)
                # 如果有异常则重新抛出
                raise
            finally:
                # 无论是否有异常，都发布函数执行时长的指标
                publish_metric(
                    group,
                    f"{func.__name__}.duration.ms",
                    get_elapsed_time_ms(start_time),  # type: ignore[possibly-undefined]
                )
            # 返回被装饰函数的返回值
            return result

        # 返回装饰后的 `wrapper` 函数
        return wrapper

    # 返回装饰器函数 `wrap`
    return wrap


# 发布一个指标数据点的函数，指定指标名称、值和指标组名，默认组名为 "torchelastic"
def put_metric(metric_name: str, metric_value: int, metric_group: str = "torchelastic"):
    """
    Publish a metric data point.

    Usage

    ::

     put_metric("metric_name", 1)
     put_metric("metric_name", 1, "metric_group_name")
    """
    # 获取指定组名的数据流对象，并添加指定的指标名称和值
    getStream(metric_group).add_value(metric_name, metric_value)


# 已废弃的函数，提醒使用者改用 `put_metric(metric_group)(metric_name, metric_value)`，可能会在未来发出警告
@deprecated(
    "Deprecated, use `put_metric(metric_group)(metric_name, metric_value)` instead",
    category=FutureWarning,
)
# 发布一个指标数据点的函数，指定指标组名、指标名称和指标值
def publish_metric(metric_group: str, metric_name: str, metric_value: int):
    # 获取指定组名的数据流对象，并添加指定的指标名称和值
    metric_stream = getStream(metric_group)
    metric_stream.add_value(metric_name, metric_value)


# 计算从给定开始时间到当前时间的经过时间（毫秒），并返回整数类型的结果
def get_elapsed_time_ms(start_time_in_seconds: float):
    """Return the elapsed time in millis from the given start time."""
    # 获取当前时间
    end_time = time.time()
    # 计算时间差并转换为毫秒
    return int((end_time - start_time_in_seconds) * 1000)
```