# `.\diffusers\utils\dummy_torch_and_scipy_objects.py`

```py
# 该文件由命令 `make fix-copies` 自动生成，请勿编辑。
from ..utils import DummyObject, requires_backends  # 从上级模块导入 DummyObject 和 requires_backends 函数


class LMSDiscreteScheduler(metaclass=DummyObject):  # 定义一个使用 DummyObject 作为元类的类 LMSDiscreteScheduler
    _backends = ["torch", "scipy"]  # 定义支持的后端列表，包括 torch 和 scipy

    def __init__(self, *args, **kwargs):  # 构造函数，接受任意数量的位置和关键字参数
        requires_backends(self, ["torch", "scipy"])  # 调用 requires_backends 验证当前对象是否支持指定后端

    @classmethod
    def from_config(cls, *args, **kwargs):  # 定义一个类方法，从配置创建类实例
        requires_backends(cls, ["torch", "scipy"])  # 验证当前类是否支持指定后端

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # 定义一个类方法，从预训练模型创建类实例
        requires_backends(cls, ["torch", "scipy"])  # 验证当前类是否支持指定后端
```