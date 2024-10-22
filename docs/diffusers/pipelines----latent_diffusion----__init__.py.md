# `.\diffusers\pipelines\latent_diffusion\__init__.py`

```py
# 从 typing 模块导入 TYPE_CHECKING，用于类型检查
from typing import TYPE_CHECKING

# 从相对路径导入一些工具函数和常量
from ...utils import (
    DIFFUSERS_SLOW_IMPORT,  # 导入用于指示慢导入的常量
    OptionalDependencyNotAvailable,  # 导入用于处理缺少可选依赖的异常
    _LazyModule,  # 导入延迟加载模块的类
    get_objects_from_module,  # 导入从模块获取对象的函数
    is_torch_available,  # 导入检查 PyTorch 是否可用的函数
    is_transformers_available,  # 导入检查 Transformers 是否可用的函数
)

# 初始化一个空字典，用于存放虚拟对象
_dummy_objects = {}
# 初始化一个空字典，用于存放模块的导入结构
_import_structure = {}

# 尝试执行以下代码块
try:
    # 检查 Transformers 和 PyTorch 是否可用，如果任意一个不可用，则抛出异常
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
# 捕获缺少可选依赖的异常
except OptionalDependencyNotAvailable:
    # 从工具模块导入虚拟的 Torch 和 Transformers 对象，忽略某些检查
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403

    # 更新虚拟对象字典，将导入的对象添加进去
    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    # 如果依赖都可用，更新导入结构字典，记录需要导入的类
    _import_structure["pipeline_latent_diffusion"] = ["LDMBertModel", "LDMTextToImagePipeline"]
    _import_structure["pipeline_latent_diffusion_superresolution"] = ["LDMSuperResolutionPipeline"]

# 如果在类型检查状态或需要慢导入的状态下执行
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    # 尝试执行以下代码块
    try:
        # 检查 Transformers 和 PyTorch 是否可用
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()

    # 捕获缺少可选依赖的异常
    except OptionalDependencyNotAvailable:
        # 从工具模块导入虚拟的 Torch 和 Transformers 对象
        from ...utils.dummy_torch_and_transformers_objects import *
    else:
        # 如果依赖可用，导入相关的类
        from .pipeline_latent_diffusion import LDMBertModel, LDMTextToImagePipeline
        from .pipeline_latent_diffusion_superresolution import LDMSuperResolutionPipeline

# 否则（不是在类型检查状态下）
else:
    # 导入 sys 模块
    import sys

    # 将当前模块的 sys.modules 条目替换为延迟加载的模块
    sys.modules[__name__] = _LazyModule(
        __name__,  # 当前模块名
        globals()["__file__"],  # 当前模块文件路径
        _import_structure,  # 模块的导入结构
        module_spec=__spec__,  # 模块的规范
    )

    # 遍历虚拟对象字典，将其中的对象设置为当前模块的属性
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
```