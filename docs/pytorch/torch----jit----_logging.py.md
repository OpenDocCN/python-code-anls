# `.\pytorch\torch\jit\_logging.py`

```
# 导入 torch 库，用于后续操作
import torch

# 从 torch 库中获取 prim 模块中的 AddStatValue 函数或操作
add_stat_value = torch.ops.prim.AddStatValue

# 从 torch 库中获取 _C 模块中的 logging_set_logger 函数，并赋值给 set_logger 变量
set_logger = torch._C._logging_set_logger

# 从 torch 库中获取 _C 模块中的 LockingLogger 类，并赋值给 LockingLogger 变量
LockingLogger = torch._C.LockingLogger

# 从 torch 库中获取 _C 模块中的 AggregationType 类，并赋值给 AggregationType 变量
AggregationType = torch._C.AggregationType

# 从 torch 库中获取 _C 模块中的 NoopLogger 类，并赋值给 NoopLogger 变量
NoopLogger = torch._C.NoopLogger

# 从 torch 库中获取 prim 模块中的 TimePoint 函数或操作
time_point = torch.ops.prim.TimePoint
```