# `.\diffusers\pipelines\stable_cascade\__init__.py`

```py
# 从类型检查模块导入常量
from typing import TYPE_CHECKING

# 从上级目录导入实用工具函数和常量
from ...utils import (
    DIFFUSERS_SLOW_IMPORT,  # 慢导入标志
    OptionalDependencyNotAvailable,  # 可选依赖不可用异常
    _LazyModule,  # 懒加载模块工具
    get_objects_from_module,  # 从模块获取对象的函数
    is_torch_available,  # 检查 PyTorch 是否可用
    is_transformers_available,  # 检查 Transformers 是否可用
)

# 初始化一个空字典，用于存储占位符对象
_dummy_objects = {}
# 初始化一个空字典，用于存储导入结构
_import_structure = {}

# 尝试检查依赖项是否可用
try:
    # 如果 Transformers 和 PyTorch 不可用，抛出异常
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
# 捕获可选依赖不可用异常
except OptionalDependencyNotAvailable:
    # 从实用工具导入占位符对象
    from ...utils import dummy_torch_and_transformers_objects

    # 更新占位符对象字典
    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    # 如果依赖项可用，更新导入结构字典
    _import_structure["pipeline_stable_cascade"] = ["StableCascadeDecoderPipeline"]
    _import_structure["pipeline_stable_cascade_combined"] = ["StableCascadeCombinedPipeline"]
    _import_structure["pipeline_stable_cascade_prior"] = ["StableCascadePriorPipeline"]

# 根据类型检查或慢导入标志执行以下操作
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    # 尝试检查依赖项是否可用
    try:
        # 如果 Transformers 和 PyTorch 不可用，抛出异常
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()
    # 捕获可选依赖不可用异常
    except OptionalDependencyNotAvailable:
        # 导入占位符对象以避免错误
        from ...utils.dummy_torch_and_transformers_objects import *  # noqa F403
    else:
        # 导入可用的管道类
        from .pipeline_stable_cascade import StableCascadeDecoderPipeline
        from .pipeline_stable_cascade_combined import StableCascadeCombinedPipeline
        from .pipeline_stable_cascade_prior import StableCascadePriorPipeline
else:
    # 如果不进行类型检查和慢导入，执行懒加载
    import sys

    # 用懒加载模块替换当前模块
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )

    # 设置当前模块的占位符对象
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
```