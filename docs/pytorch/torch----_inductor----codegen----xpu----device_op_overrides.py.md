# `.\pytorch\torch\_inductor\codegen\xpu\device_op_overrides.py`

```
# mypy: allow-untyped-defs
# 从 common 模块中导入 DeviceOpOverrides 和 register_device_op_overrides 函数
from ..common import DeviceOpOverrides, register_device_op_overrides

# 定义 XPUDeviceOpOverrides 类，继承自 DeviceOpOverrides 类
class XPUDeviceOpOverrides(DeviceOpOverrides):
    
    # 覆盖 import_get_raw_stream_as 方法，返回一个字符串格式化的导入语句
    def import_get_raw_stream_as(self, name):
        return f"from torch._C import _xpu_getCurrentRawStream as {name}"

    # 覆盖 set_device 方法，返回设置 XPU 设备的调用语句
    def set_device(self, device_idx):
        return f"torch.xpu.set_device({device_idx})"

    # 覆盖 synchronize 方法，返回 XPU 同步操作的调用语句
    def synchronize(self):
        return "torch.xpu.synchronize()"

    # 覆盖 device_guard 方法，返回创建 XPU 设备上下文管理器的调用语句
    def device_guard(self, device_idx):
        return f"torch.xpu._DeviceGuard({device_idx})"

# 将 XPUDeviceOpOverrides 实例注册为 xpu 设备的操作覆盖
register_device_op_overrides("xpu", XPUDeviceOpOverrides())
```