# `.\pytorch\torch\_lazy\device_context.py`

```py
# mypy: allow-untyped-defs
# 导入线程库和类型定义相关模块
import threading
from typing import Any, Dict

# 导入 Torch C++ 扩展模块
import torch._C._lazy

# 设备上下文管理类
class DeviceContext:
    # 类变量，存储不同设备的上下文对象字典
    _CONTEXTS: Dict[str, Any] = dict()
    # 线程锁，用于保护上下文字典的并发访问
    _CONTEXTS_LOCK = threading.Lock()

    # 初始化方法，设定设备
    def __init__(self, device):
        self.device = device


# 获取设备上下文的函数
def get_device_context(device=None):
    # 如果设备参数为 None，则获取默认设备类型
    if device is None:
        device = torch._C._lazy._get_default_device_type()
    else:
        device = str(device)
    
    # 使用线程锁保护以下操作
    with DeviceContext._CONTEXTS_LOCK:
        # 尝试从上下文字典中获取对应设备的上下文对象
        devctx = DeviceContext._CONTEXTS.get(device, None)
        # 如果该设备上下文不存在，则创建新的上下文对象并添加到字典中
        if devctx is None:
            devctx = DeviceContext(device)
            DeviceContext._CONTEXTS[device] = devctx
        # 返回获取或新创建的设备上下文对象
        return devctx
```