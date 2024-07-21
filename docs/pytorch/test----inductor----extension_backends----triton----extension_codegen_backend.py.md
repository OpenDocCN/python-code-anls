# `.\pytorch\test\inductor\extension_backends\triton\extension_codegen_backend.py`

```
# 导入所需模块和类
from torch._inductor.codegen import triton, wrapper
from torch._inductor.codegen.common import DeviceOpOverrides
from torch._inductor.scheduler import BaseScheduling

# 定义一个继承自wrapper.WrapperCodeGen的类ExtensionWrapperCodegen
class ExtensionWrapperCodegen(wrapper.WrapperCodeGen):
    def __init__(self):
        # 调用父类的初始化方法
        super().__init__()

# 定义一个继承自BaseScheduling的类ExtensionScheduling
class ExtensionScheduling(BaseScheduling):
    def __init__(self, scheduler):
        # 初始化时将scheduler保存为实例变量
        self.scheduler = scheduler
        # 创建一个triton.TritonScheduling的实例并保存为实例变量
        self._triton_scheduling = triton.TritonScheduling(scheduler)

    # 判断两个节点是否可以垂直融合，始终返回True
    def can_fuse_vertical(self, node1, node2):
        return True

    # 判断两个节点是否可以水平融合，始终返回True
    def can_fuse_horizontal(self, node1, node2):
        return True

    # 根据sizes调用_triton_scheduling的group_fn方法进行分组
    def group_fn(self, sizes):
        return self._triton_scheduling.group_fn(sizes)

    # 生成template_node节点的代码模板，epilogue_nodes未使用
    def codegen_template(self, template_node, epilogue_nodes):
        pass

    # 根据节点node调用_triton_scheduling的codegen_node方法生成代码
    def codegen_node(self, node):
        self._triton_scheduling.codegen_node(node)

    # 同步代码生成，未实现具体功能
    def codegen_sync(self):
        pass

    # 刷新_triton_scheduling的状态
    def flush(self):
        self._triton_scheduling.flush()

# 定义一个继承自DeviceOpOverrides的类CPUDeviceOpOverrides
class CPUDeviceOpOverrides(DeviceOpOverrides):
    # 根据name生成一个占位符函数的字符串
    def import_get_raw_stream_as(self, name: str) -> str:
        return f"def {name}(name): None\n"

    # 设置设备，但实现中未使用device_idx参数，返回空字符串
    def set_device(self, device_idx: int) -> str:  # noqa: ARG002 unused-argument
        return ""

    # 同步设备操作，未实现具体功能
    def synchronize(self) -> None:
        pass

    # 设备保护，但实现中未使用device_idx参数，返回空字符串
    def device_guard(self, device_idx: int) -> str:  # noqa: ARG002 unused-argument
        return ""
```