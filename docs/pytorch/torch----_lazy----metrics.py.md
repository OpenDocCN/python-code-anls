# `.\pytorch\torch\_lazy\metrics.py`

```py
# 使用 mypy: allow-untyped-defs 来允许未类型化的定义
import torch._C._lazy

# 定义函数 reset，用于重置所有的度量计数器
def reset():
    """Resets all metric counters."""
    # 调用 torch._C._lazy._reset_metrics() 方法重置度量计数器
    torch._C._lazy._reset_metrics()

# 定义函数 counter_names，用于获取当前所有活跃的计数器名称
def counter_names():
    """Retrieves all the currently active counter names."""
    # 返回 torch._C._lazy._counter_names() 方法的结果，即当前活跃计数器的名称列表
    return torch._C._lazy._counter_names()

# 定义函数 counter_value，用于返回具有指定名称的计数器的值
def counter_value(name: str):
    """Return the value of the counter with the specified name"""
    # 返回 torch._C._lazy._counter_value(name) 方法的结果，即指定名称计数器的值
    return torch._C._lazy._counter_value(name)

# 定义函数 metrics_report，用于返回组合的（lazy 核心和后端）度量报告
def metrics_report():
    """Return the combined (lazy core and backend) metric report"""
    # 返回 torch._C._lazy._metrics_report() 方法的结果，即组合的度量报告
    return torch._C._lazy._metrics_report()
```