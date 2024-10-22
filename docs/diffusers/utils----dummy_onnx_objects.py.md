# `.\diffusers\utils\dummy_onnx_objects.py`

```py
# 该文件由命令 `make fix-copies` 自动生成，请勿编辑。
from ..utils import DummyObject, requires_backends  # 从 utils 模块导入 DummyObject 和 requires_backends 函数

class OnnxRuntimeModel(metaclass=DummyObject):  # 定义 OnnxRuntimeModel 类，使用 DummyObject 作为其元类
    _backends = ["onnx"]  # 定义一个类属性，表示支持的后端列表

    def __init__(self, *args, **kwargs):  # 初始化方法，接受任意数量的位置和关键字参数
        requires_backends(self, ["onnx"])  # 调用 requires_backends 函数，检查是否支持 'onnx' 后端

    @classmethod
    def from_config(cls, *args, **kwargs):  # 类方法，接受任意数量的位置和关键字参数
        requires_backends(cls, ["onnx"])  # 调用 requires_backends 函数，检查类是否支持 'onnx' 后端

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # 类方法，接受任意数量的位置和关键字参数
        requires_backends(cls, ["onnx"])  # 调用 requires_backends 函数，检查类是否支持 'onnx' 后端
```