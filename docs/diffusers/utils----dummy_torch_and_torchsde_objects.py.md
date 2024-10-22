# `.\diffusers\utils\dummy_torch_and_torchsde_objects.py`

```py
# 该文件由命令 `make fix-copies` 自动生成，请勿编辑。
from ..utils import DummyObject, requires_backends  # 从上级模块导入 DummyObject 和 requires_backends 函数


class CosineDPMSolverMultistepScheduler(metaclass=DummyObject):  # 定义 CosineDPMSolverMultistepScheduler 类，使用 DummyObject 作为元类
    _backends = ["torch", "torchsde"]  # 定义支持的后端列表

    def __init__(self, *args, **kwargs):  # 初始化方法，接受任意位置和关键字参数
        requires_backends(self, ["torch", "torchsde"])  # 检查是否满足后端要求

    @classmethod
    def from_config(cls, *args, **kwargs):  # 类方法，从配置创建实例
        requires_backends(cls, ["torch", "torchsde"])  # 检查类是否满足后端要求

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # 类方法，从预训练模型创建实例
        requires_backends(cls, ["torch", "torchsde"])  # 检查类是否满足后端要求


class DPMSolverSDEScheduler(metaclass=DummyObject):  # 定义 DPMSolverSDEScheduler 类，使用 DummyObject 作为元类
    _backends = ["torch", "torchsde"]  # 定义支持的后端列表

    def __init__(self, *args, **kwargs):  # 初始化方法，接受任意位置和关键字参数
        requires_backends(self, ["torch", "torchsde"])  # 检查是否满足后端要求

    @classmethod
    def from_config(cls, *args, **kwargs):  # 类方法，从配置创建实例
        requires_backends(cls, ["torch", "torchsde"])  # 检查类是否满足后端要求

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # 类方法，从预训练模型创建实例
        requires_backends(cls, ["torch", "torchsde"])  # 检查类是否满足后端要求
```