# `.\pytorch\torch\_inductor\codegen\cuda\device_op_overrides.py`

```py
# mypy: allow-untyped-defs
# 导入未类型化定义的函数和类
from ..common import DeviceOpOverrides, register_device_op_overrides

# 定义 CUDADeviceOpOverrides 类，继承自 DeviceOpOverrides 类
class CUDADeviceOpOverrides(DeviceOpOverrides):

    # 定义 import_get_raw_stream_as 方法，返回一个字符串格式化的导入语句
    def import_get_raw_stream_as(self, name):
        return f"from torch._C import _cuda_getCurrentRawStream as {name}"

    # 定义 set_device 方法，返回一个字符串格式化的设置当前 CUDA 设备的语句
    def set_device(self, device_idx):
        return f"torch.cuda.set_device({device_idx})"

    # 定义 synchronize 方法，返回一个字符串格式化的 CUDA 设备同步语句
    def synchronize(self):
        return "torch.cuda.synchronize()"

    # 定义 device_guard 方法，返回一个字符串格式化的 CUDA 设备保护语句
    def device_guard(self, device_idx):
        return f"torch.cuda._DeviceGuard({device_idx})"

# 调用 register_device_op_overrides 函数，注册 CUDA 设备操作的覆盖方法
register_device_op_overrides("cuda", CUDADeviceOpOverrides())
```