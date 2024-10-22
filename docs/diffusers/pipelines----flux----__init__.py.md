# `.\diffusers\pipelines\flux\__init__.py`

```py
# 从 typing 模块导入 TYPE_CHECKING，用于类型检查
from typing import TYPE_CHECKING

# 从 utils 模块导入所需的工具函数和常量
from ...utils import (
    DIFFUSERS_SLOW_IMPORT,  # 导入慢加载的标志
    OptionalDependencyNotAvailable,  # 导入可选依赖未找到的异常
    _LazyModule,  # 导入延迟模块加载的类
    get_objects_from_module,  # 导入从模块获取对象的函数
    is_torch_available,  # 导入检查 PyTorch 是否可用的函数
    is_transformers_available,  # 导入检查 Transformers 是否可用的函数
)

# 定义一个空字典，用于存储虚拟对象
_dummy_objects = {}
# 定义一个空字典，用于存储额外导入的对象
_additional_imports = {}
# 定义模块的导入结构，初始化 pipeline_output
_import_structure = {"pipeline_output": ["FluxPipelineOutput"]}

# 尝试检查 Transformers 和 PyTorch 是否可用
try:
    if not (is_transformers_available() and is_torch_available()):  # 如果不可用
        raise OptionalDependencyNotAvailable()  # 抛出异常
# 捕获可选依赖未找到的异常
except OptionalDependencyNotAvailable:
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403  # 导入虚拟对象模块

    # 更新 _dummy_objects 字典，获取虚拟对象
    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    # 如果可用，更新导入结构以包含 FluxPipeline
    _import_structure["pipeline_flux"] = ["FluxPipeline"]
# 如果进行类型检查或慢加载标志为真
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        # 检查 Transformers 和 PyTorch 是否可用
        if not (is_transformers_available() and is_torch_available()):  # 如果不可用
            raise OptionalDependencyNotAvailable()  # 抛出异常
    # 捕获可选依赖未找到的异常
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import *  # noqa F403  # 导入虚拟对象
    else:
        # 如果可用，从 pipeline_flux 导入 FluxPipeline
        from .pipeline_flux import FluxPipeline
else:
    import sys  # 导入 sys 模块

    # 将当前模块替换为一个延迟加载的模块
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )

    # 将 _dummy_objects 中的对象设置为当前模块的属性
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
    # 将 _additional_imports 中的对象设置为当前模块的属性
    for name, value in _additional_imports.items():
        setattr(sys.modules[__name__], name, value)
```