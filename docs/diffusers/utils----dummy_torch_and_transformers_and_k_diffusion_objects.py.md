# `.\diffusers\utils\dummy_torch_and_transformers_and_k_diffusion_objects.py`

```py
# 该文件由命令 `make fix-copies` 自动生成，请勿编辑。
from ..utils import DummyObject, requires_backends  # 从 utils 模块导入 DummyObject 和 requires_backends 函数


class StableDiffusionKDiffusionPipeline(metaclass=DummyObject):  # 定义一个名为 StableDiffusionKDiffusionPipeline 的类，使用 DummyObject 作为其元类
    _backends = ["torch", "transformers", "k_diffusion"]  # 定义一个类属性 _backends，包含可用的后端框架

    def __init__(self, *args, **kwargs):  # 定义初始化方法，接受任意数量的位置和关键字参数
        requires_backends(self, ["torch", "transformers", "k_diffusion"])  # 检查当前对象是否具备必要的后端依赖

    @classmethod
    def from_config(cls, *args, **kwargs):  # 定义一个类方法 from_config，接受任意数量的位置和关键字参数
        requires_backends(cls, ["torch", "transformers", "k_diffusion"])  # 检查类是否具备必要的后端依赖

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # 定义一个类方法 from_pretrained，接受任意数量的位置和关键字参数
        requires_backends(cls, ["torch", "transformers", "k_diffusion"])  # 检查类是否具备必要的后端依赖


class StableDiffusionXLKDiffusionPipeline(metaclass=DummyObject):  # 定义一个名为 StableDiffusionXLKDiffusionPipeline 的类，使用 DummyObject 作为其元类
    _backends = ["torch", "transformers", "k_diffusion"]  # 定义一个类属性 _backends，包含可用的后端框架

    def __init__(self, *args, **kwargs):  # 定义初始化方法，接受任意数量的位置和关键字参数
        requires_backends(self, ["torch", "transformers", "k_diffusion"])  # 检查当前对象是否具备必要的后端依赖

    @classmethod
    def from_config(cls, *args, **kwargs):  # 定义一个类方法 from_config，接受任意数量的位置和关键字参数
        requires_backends(cls, ["torch", "transformers", "k_diffusion"])  # 检查类是否具备必要的后端依赖

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # 定义一个类方法 from_pretrained，接受任意数量的位置和关键字参数
        requires_backends(cls, ["torch", "transformers", "k_diffusion"])  # 检查类是否具备必要的后端依赖
```