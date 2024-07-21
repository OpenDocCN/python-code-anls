# `.\pytorch\torch\_lazy\config.py`

```py
# 引入 mypy 模块的配置：允许未类型化的定义
import torch._C._lazy

# 获取用于强制 LTC 回退的配置
def get_force_fallback():
    """Get the config used to force LTC fallback"""
    # 调用 torch._C._lazy 模块的 _get_force_fallback 函数，返回其结果
    return torch._C._lazy._get_force_fallback()

# 设置用于强制 LTC 回退的配置值
def set_force_fallback(configval):
    """Set the config used to force LTC fallback"""
    # 调用 torch._C._lazy 模块的 _set_force_fallback 函数，设置配置值为 configval
    torch._C._lazy._set_force_fallback(configval)

# 设置是否重用 IR 节点以加快追踪速度的配置
def set_reuse_ir(val: bool):
    """Set the config to reuse IR nodes for faster tracing"""
    # 调用 torch._C._lazy 模块的 _set_reuse_ir 函数，设置是否重用 IR 节点为 val
    torch._C._lazy._set_reuse_ir(val)
```