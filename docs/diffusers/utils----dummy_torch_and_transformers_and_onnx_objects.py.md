# `.\diffusers\utils\dummy_torch_and_transformers_and_onnx_objects.py`

```py
# 此文件由命令 `make fix-copies` 自动生成，勿编辑
from ..utils import DummyObject, requires_backends  # 从 utils 模块导入 DummyObject 类和 requires_backends 函数


class OnnxStableDiffusionImg2ImgPipeline(metaclass=DummyObject):  # 定义使用 DummyObject 作为元类的类
    _backends = ["torch", "transformers", "onnx"]  # 定义支持的后端列表

    def __init__(self, *args, **kwargs):  # 构造函数，接收任意参数
        requires_backends(self, ["torch", "transformers", "onnx"])  # 验证所需后端是否可用

    @classmethod
    def from_config(cls, *args, **kwargs):  # 类方法，根据配置创建实例
        requires_backends(cls, ["torch", "transformers", "onnx"])  # 验证所需后端是否可用

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # 类方法，从预训练模型加载
        requires_backends(cls, ["torch", "transformers", "onnx"])  # 验证所需后端是否可用


class OnnxStableDiffusionInpaintPipeline(metaclass=DummyObject):  # 定义使用 DummyObject 作为元类的类
    _backends = ["torch", "transformers", "onnx"]  # 定义支持的后端列表

    def __init__(self, *args, **kwargs):  # 构造函数，接收任意参数
        requires_backends(self, ["torch", "transformers", "onnx"])  # 验证所需后端是否可用

    @classmethod
    def from_config(cls, *args, **kwargs):  # 类方法，根据配置创建实例
        requires_backends(cls, ["torch", "transformers", "onnx"])  # 验证所需后端是否可用

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # 类方法，从预训练模型加载
        requires_backends(cls, ["torch", "transformers", "onnx"])  # 验证所需后端是否可用


class OnnxStableDiffusionInpaintPipelineLegacy(metaclass=DummyObject):  # 定义使用 DummyObject 作为元类的类
    _backends = ["torch", "transformers", "onnx"]  # 定义支持的后端列表

    def __init__(self, *args, **kwargs):  # 构造函数，接收任意参数
        requires_backends(self, ["torch", "transformers", "onnx"])  # 验证所需后端是否可用

    @classmethod
    def from_config(cls, *args, **kwargs):  # 类方法，根据配置创建实例
        requires_backends(cls, ["torch", "transformers", "onnx"])  # 验证所需后端是否可用

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # 类方法，从预训练模型加载
        requires_backends(cls, ["torch", "transformers", "onnx"])  # 验证所需后端是否可用


class OnnxStableDiffusionPipeline(metaclass=DummyObject):  # 定义使用 DummyObject 作为元类的类
    _backends = ["torch", "transformers", "onnx"]  # 定义支持的后端列表

    def __init__(self, *args, **kwargs):  # 构造函数，接收任意参数
        requires_backends(self, ["torch", "transformers", "onnx"])  # 验证所需后端是否可用

    @classmethod
    def from_config(cls, *args, **kwargs):  # 类方法，根据配置创建实例
        requires_backends(cls, ["torch", "transformers", "onnx"])  # 验证所需后端是否可用

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # 类方法，从预训练模型加载
        requires_backends(cls, ["torch", "transformers", "onnx"])  # 验证所需后端是否可用


class OnnxStableDiffusionUpscalePipeline(metaclass=DummyObject):  # 定义使用 DummyObject 作为元类的类
    _backends = ["torch", "transformers", "onnx"]  # 定义支持的后端列表

    def __init__(self, *args, **kwargs):  # 构造函数，接收任意参数
        requires_backends(self, ["torch", "transformers", "onnx"])  # 验证所需后端是否可用

    @classmethod
    def from_config(cls, *args, **kwargs):  # 类方法，根据配置创建实例
        requires_backends(cls, ["torch", "transformers", "onnx"])  # 验证所需后端是否可用

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # 类方法，从预训练模型加载
        requires_backends(cls, ["torch", "transformers", "onnx"])  # 验证所需后端是否可用


class StableDiffusionOnnxPipeline(metaclass=DummyObject):  # 定义使用 DummyObject 作为元类的类
    _backends = ["torch", "transformers", "onnx"]  # 定义支持的后端列表

    def __init__(self, *args, **kwargs):  # 构造函数，接收任意参数
        requires_backends(self, ["torch", "transformers", "onnx"])  # 验证所需后端是否可用

    @classmethod
    def from_config(cls, *args, **kwargs):  # 类方法，根据配置创建实例
        requires_backends(cls, ["torch", "transformers", "onnx"])  # 验证所需后端是否可用

    @classmethod  # 类方法的开始
    # 定义一个类方法，从预训练模型加载
        def from_pretrained(cls, *args, **kwargs):
            # 检查是否满足所需的后端库要求，确保 'torch'、'transformers' 和 'onnx' 可用
            requires_backends(cls, ["torch", "transformers", "onnx"])
```