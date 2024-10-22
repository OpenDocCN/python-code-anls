# `.\diffusers\pipelines\semantic_stable_diffusion\__init__.py`

```py
# 导入类型检查相关常量
from typing import TYPE_CHECKING

# 从工具模块导入必要的组件
from ...utils import (
    DIFFUSERS_SLOW_IMPORT,  # 导入慢加载标志
    OptionalDependencyNotAvailable,  # 导入可选依赖不可用异常
    _LazyModule,  # 导入懒加载模块
    get_objects_from_module,  # 导入从模块获取对象的函数
    is_torch_available,  # 导入检查 PyTorch 是否可用的函数
    is_transformers_available,  # 导入检查 Transformers 是否可用的函数
)

# 初始化一个空字典用于存储虚拟对象
_dummy_objects = {}
# 初始化一个空字典用于存储导入结构
_import_structure = {}

# 尝试执行依赖检查
try:
    # 检查 Transformers 和 PyTorch 是否都可用
    if not (is_transformers_available() and is_torch_available()):
        # 如果不可用，抛出异常
        raise OptionalDependencyNotAvailable()
# 捕获可选依赖不可用的异常
except OptionalDependencyNotAvailable:
    # 导入虚拟对象以避免实际依赖
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403

    # 更新虚拟对象字典
    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    # 如果依赖可用，更新导入结构
    _import_structure["pipeline_output"] = ["SemanticStableDiffusionPipelineOutput"]
    _import_structure["pipeline_semantic_stable_diffusion"] = ["SemanticStableDiffusionPipeline"]

# 根据类型检查或慢加载标志执行进一步的检查
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        # 检查 Transformers 和 PyTorch 是否都可用
        if not (is_transformers_available() and is_torch_available()):
            # 如果不可用，抛出异常
            raise OptionalDependencyNotAvailable()
    # 捕获可选依赖不可用的异常
    except OptionalDependencyNotAvailable:
        # 从虚拟对象模块导入所有内容
        from ...utils.dummy_torch_and_transformers_objects import *
    else:
        # 如果依赖可用，导入语义稳定扩散管道
        from .pipeline_semantic_stable_diffusion import SemanticStableDiffusionPipeline

# 如果不是类型检查或慢加载
else:
    import sys

    # 用懒加载模块替代当前模块
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )

    # 将虚拟对象字典中的每个对象设置到当前模块
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
```