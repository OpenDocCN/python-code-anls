# `.\pytorch\torch\distributed\elastic\metrics\__init__.py`

```py
#!/usr/bin/env/python3
# mypy: allow-untyped-defs

# 导入所需模块
import time
import torch.distributed.elastic.metrics as metrics

# 配置全局的度量处理器
metrics.configure(metrics.NullMetricsHandler())
metrics.configure(metrics.ConsoleMetricsHandler(), "my_module")

# 定义一个函数，用于示例
def my_method():
    # 记录开始时间
    start = time.time()
    # 调用 calculate() 函数
    calculate()
    # 记录结束时间
    end = time.time()
    # 记录计算时延度量，使用 my_module 作为度量组名
    metrics.put_metric("calculate_latency", int(end-start), "my_module")

# 配置度量处理器，用于另一个示例模块 foobar
metrics.configure(metrics.ConsoleMetricsHandler(), "foobar")
metrics.configure(metrics.ConsoleMetricsHandler(), "Bar")

# 定义一个被度量的函数 foo()
@metrics.prof
def foo():
    pass

# 定义一个类 Bar，其中的方法 baz() 也会被度量
class Bar():

    @metrics.prof
    def baz():
        pass

# @metrics.prof 装饰器会发布以下度量
# <leaf_module or classname>.success - 如果函数成功完成，则为 1
# <leaf_module or classname>.failure - 如果函数抛出异常，则为 1
# <leaf_module or classname>.duration.ms - 函数执行时长（毫秒）

# torch.distributed.elastic.metrics.MetricHandler 负责发出度量
# 导入必要的模块和函数，包括配置、控制台指标处理器等
from .api import (
    configure,                 # 导入配置函数
    ConsoleMetricHandler,      # 导入控制台指标处理器类
    get_elapsed_time_ms,       # 导入获取毫秒级流逝时间函数
    getStream,                 # 导入获取流函数
    MetricData,                # 导入指标数据类
    MetricHandler,             # 导入指标处理器基类
    MetricsConfig,             # 导入指标配置类
    NullMetricHandler,         # 导入空指标处理器类
    prof,                      # 导入 prof 函数
    profile,                   # 导入 profile 函数
    publish_metric,            # 导入发布指标函数
    put_metric,                # 导入记录指标函数
)

# 定义初始化指标函数，可选参数为指标配置对象
def initialize_metrics(cfg: Optional[MetricsConfig] = None):
    # 函数体暂时为空
    pass

# 尝试导入静态初始化模块，用于弹性分布式的指标
try:
    from torch.distributed.elastic.metrics.static_init import *  # type: ignore[import] # noqa: F401 F403
except ModuleNotFoundError:
    # 如果模块未找到，忽略异常
    pass
```