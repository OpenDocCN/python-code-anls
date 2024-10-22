# `.\diffusers\pipelines\stable_diffusion_k_diffusion\__init__.py`

```py
# 导入类型检查支持
from typing import TYPE_CHECKING

# 从工具模块中导入必要的工具和依赖
from ...utils import (
    DIFFUSERS_SLOW_IMPORT,  # 慢导入标志
    OptionalDependencyNotAvailable,  # 可选依赖不可用异常
    _LazyModule,  # 懒加载模块
    get_objects_from_module,  # 从模块获取对象的函数
    is_k_diffusion_available,  # 检查 k_diffusion 是否可用
    is_k_diffusion_version,  # 检查 k_diffusion 的版本
    is_torch_available,  # 检查 PyTorch 是否可用
    is_transformers_available,  # 检查 transformers 是否可用
)

# 初始化一个空字典以存储虚拟对象
_dummy_objects = {}
# 初始化一个空字典以定义导入结构
_import_structure = {}

# 尝试块，检查依赖是否可用
try:
    if not (
        is_transformers_available()  # 检查 transformers 可用性
        and is_torch_available()  # 检查 PyTorch 可用性
        and is_k_diffusion_available()  # 检查 k_diffusion 可用性
        and is_k_diffusion_version(">=", "0.0.12")  # 检查 k_diffusion 版本
    ):
        raise OptionalDependencyNotAvailable()  # 抛出可选依赖不可用异常
except OptionalDependencyNotAvailable:
    # 如果依赖不可用，导入虚拟对象以避免错误
    from ...utils import dummy_torch_and_transformers_and_k_diffusion_objects  # noqa F403

    # 更新虚拟对象字典
    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_and_k_diffusion_objects))
else:
    # 如果依赖可用，定义导入结构
    _import_structure["pipeline_stable_diffusion_k_diffusion"] = ["StableDiffusionKDiffusionPipeline"]
    _import_structure["pipeline_stable_diffusion_xl_k_diffusion"] = ["StableDiffusionXLKDiffusionPipeline"]

# 根据类型检查或慢导入标志，进行进一步处理
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (
            is_transformers_available()  # 检查 transformers 可用性
            and is_torch_available()  # 检查 PyTorch 可用性
            and is_k_diffusion_available()  # 检查 k_diffusion 可用性
            and is_k_diffusion_version(">=", "0.0.12")  # 检查 k_diffusion 版本
        ):
            raise OptionalDependencyNotAvailable()  # 抛出可选依赖不可用异常

    except OptionalDependencyNotAvailable:
        # 如果依赖不可用，导入虚拟对象以避免错误
        from ...utils.dummy_torch_and_transformers_and_k_diffusion_objects import *
    else:
        # 如果依赖可用，导入相关管道
        from .pipeline_stable_diffusion_k_diffusion import StableDiffusionKDiffusionPipeline
        from .pipeline_stable_diffusion_xl_k_diffusion import StableDiffusionXLKDiffusionPipeline

else:
    # 如果不是类型检查，设置懒加载模块
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,  # 当前模块名
        globals()["__file__"],  # 当前文件路径
        _import_structure,  # 导入结构
        module_spec=__spec__,  # 模块规范
    )

    # 将虚拟对象设置到当前模块
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
```