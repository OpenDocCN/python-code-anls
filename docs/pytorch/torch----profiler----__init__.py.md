# `.\pytorch\torch\profiler\__init__.py`

```
# 声明一个文档字符串，介绍了 PyTorch Profiler 工具的功能和用途
# 允许未类型化定义，用于 mypy 检查
r"""
PyTorch Profiler is a tool that allows the collection of performance metrics during training and inference.
Profiler's context manager API can be used to better understand what model operators are the most expensive,
examine their input shapes and stack traces, study device kernel activity and visualize the execution trace.

.. note::
    An earlier version of the API in :mod:`torch.autograd` module is considered legacy and will be deprecated.

"""

# 导入操作系统相关的功能
import os

# 导入需要的 C++ 扩展模块和函数
from torch._C._autograd import _supported_activities, DeviceType, kineto_available
from torch._C._profiler import _ExperimentalConfig, ProfilerActivity, RecordScope
# 导入记录函数调用的相关功能
from torch.autograd.profiler import KinetoStepTracker, record_function
# 导入优化器相关的钩子函数
from torch.optim.optimizer import register_optimizer_step_post_hook

# 导入自定义的性能分析器相关模块和函数
from .profiler import (
    _KinetoProfile,
    ExecutionTraceObserver,
    profile,
    ProfilerAction,
    schedule,
    supported_activities,
    tensorboard_trace_handler,
)

# 导出的模块和函数列表
__all__ = [
    "profile",
    "schedule",
    "supported_activities",
    "tensorboard_trace_handler",
    "ProfilerAction",
    "ProfilerActivity",
    "kineto_available",
    "DeviceType",
    "record_function",
    "ExecutionTraceObserver",
]

# 导入 Intel VTune 标记工具
from . import itt

# 定义一个优化器后处理钩子函数，用于注册到优化器中，跟踪优化步骤
def _optimizer_post_hook(optimizer, args, kwargs):
    # 使用 KinetoStepTracker 类增加优化器步骤计数
    KinetoStepTracker.increment_step("Optimizer")

# 如果环境变量 KINETO_USE_DAEMON 存在且非空，注册优化器后处理钩子函数
if os.environ.get("KINETO_USE_DAEMON", None):
    _ = register_optimizer_step_post_hook(_optimizer_post_hook)
```