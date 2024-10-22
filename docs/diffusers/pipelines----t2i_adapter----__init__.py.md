# `.\diffusers\pipelines\t2i_adapter\__init__.py`

```py
# 从 typing 模块导入 TYPE_CHECKING，用于类型检查的特殊常量
from typing import TYPE_CHECKING

# 从 utils 模块导入一系列工具函数和常量
from ...utils import (
    DIFFUSERS_SLOW_IMPORT,  # 慢速导入的标志
    OptionalDependencyNotAvailable,  # 可选依赖项不可用的异常
    _LazyModule,  # 懒加载模块的类
    get_objects_from_module,  # 从模块中获取对象的函数
    is_torch_available,  # 检查 PyTorch 是否可用的函数
    is_transformers_available,  # 检查 Transformers 是否可用的函数
)

# 初始化一个空字典，用于存储虚拟对象
_dummy_objects = {}
# 初始化一个空字典，用于存储导入结构
_import_structure = {}

# 尝试检查是否可用必要的依赖项
try:
    if not (is_transformers_available() and is_torch_available()):  # 如果 Transformers 和 PyTorch 不可用
        raise OptionalDependencyNotAvailable()  # 抛出可选依赖项不可用异常
# 捕获可选依赖项不可用的异常
except OptionalDependencyNotAvailable:
    # 从 utils 模块导入虚拟对象，避免抛出错误
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403

    # 更新虚拟对象字典，获取虚拟对象
    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    # 如果依赖项可用，更新导入结构，添加 StableDiffusionAdapterPipeline
    _import_structure["pipeline_stable_diffusion_adapter"] = ["StableDiffusionAdapterPipeline"]
    # 更新导入结构，添加 StableDiffusionXLAdapterPipeline
    _import_structure["pipeline_stable_diffusion_xl_adapter"] = ["StableDiffusionXLAdapterPipeline"]

# 如果正在进行类型检查或慢速导入
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        # 检查是否可用必要的依赖项
        if not (is_transformers_available() and is_torch_available()):  # 如果 Transformers 和 PyTorch 不可用
            raise OptionalDependencyNotAvailable()  # 抛出可选依赖项不可用异常
    # 捕获可选依赖项不可用的异常
    except OptionalDependencyNotAvailable:
        # 从虚拟对象模块导入所有内容，避免抛出错误
        from ...utils.dummy_torch_and_transformers_objects import *  # noqa F403
    else:
        # 如果依赖项可用，导入 StableDiffusionAdapterPipeline
        from .pipeline_stable_diffusion_adapter import StableDiffusionAdapterPipeline
        # 导入 StableDiffusionXLAdapterPipeline
        from .pipeline_stable_diffusion_xl_adapter import StableDiffusionXLAdapterPipeline
else:
    # 如果不是类型检查或慢速导入，使用懒加载模块
    import sys

    # 将当前模块替换为懒加载模块
    sys.modules[__name__] = _LazyModule(
        __name__,  # 模块名称
        globals()["__file__"],  # 模块文件路径
        _import_structure,  # 模块导入结构
        module_spec=__spec__,  # 模块规格
    )
    # 遍历虚拟对象字典，设置当前模块的属性
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)  # 设置属性
```