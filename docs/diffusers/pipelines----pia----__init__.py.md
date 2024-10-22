# `.\diffusers\pipelines\pia\__init__.py`

```py
# 导入类型检查相关的常量
from typing import TYPE_CHECKING

# 从 utils 模块导入所需的常量和函数
from ...utils import (
    DIFFUSERS_SLOW_IMPORT,  # 慢导入的标志
    OptionalDependencyNotAvailable,  # 可选依赖不可用的异常
    _LazyModule,  # 懒加载模块的类
    get_objects_from_module,  # 从模块中获取对象的函数
    is_torch_available,  # 检查是否可用 PyTorch 的函数
    is_transformers_available,  # 检查是否可用 Transformers 的函数
)

# 初始化一个空字典用于存储假对象
_dummy_objects = {}
# 初始化一个空字典用于存储导入结构
_import_structure = {}

try:
    # 检查 Transformers 和 PyTorch 是否可用
    if not (is_transformers_available() and is_torch_available()):
        # 如果不可用，则抛出异常
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 如果捕获到可选依赖不可用的异常，则导入假对象
    from ...utils import dummy_torch_and_transformers_objects

    # 将假对象更新到 _dummy_objects 字典
    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    # 如果依赖可用，则更新导入结构
    _import_structure["pipeline_pia"] = ["PIAPipeline", "PIAPipelineOutput"]

# 检查类型检查或慢导入的标志
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        # 再次检查 Transformers 和 PyTorch 是否可用
        if not (is_transformers_available() and is_torch_available()):
            # 如果不可用，则抛出异常
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果捕获到可选依赖不可用的异常，则导入假对象
        from ...utils.dummy_torch_and_transformers_objects import *

    else:
        # 如果依赖可用，则从 pipeline_pia 导入相关类
        from .pipeline_pia import PIAPipeline, PIAPipelineOutput

else:
    # 如果不是类型检查或慢导入，则使用懒加载模块
    import sys

    # 将当前模块替换为懒加载模块实例
    sys.modules[__name__] = _LazyModule(
        __name__,  # 模块名称
        globals()["__file__"],  # 模块文件路径
        _import_structure,  # 导入结构
        module_spec=__spec__,  # 模块规格
    )
    # 将假对象设置到当前模块中
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
```