# `.\diffusers\pipelines\stable_audio\__init__.py`

```py
# 从 typing 模块导入 TYPE_CHECKING，用于类型检查
from typing import TYPE_CHECKING

# 从上级模块的 utils 导入多个对象和函数
from ...utils import (
    # 导入 DIFFUSERS_SLOW_IMPORT 变量
    DIFFUSERS_SLOW_IMPORT,
    # 导入 OptionalDependencyNotAvailable 异常类
    OptionalDependencyNotAvailable,
    # 导入 LazyModule 类
    _LazyModule,
    # 导入获取模块对象的函数
    get_objects_from_module,
    # 导入判断是否可用的函数
    is_torch_available,
    is_transformers_available,
    is_transformers_version,
)

# 定义一个空字典，用于存储占位对象
_dummy_objects = {}
# 定义一个空字典，用于存储模块导入结构
_import_structure = {}

# 尝试检测可用性
try:
    # 检查 transformers 和 torch 是否可用，并且 transformers 版本是否满足要求
    if not (is_transformers_available() and is_torch_available() and is_transformers_version(">=", "4.27.0")):
        # 如果不满足条件，则抛出异常
        raise OptionalDependencyNotAvailable()
# 捕获 OptionalDependencyNotAvailable 异常
except OptionalDependencyNotAvailable:
    # 从 utils 导入 dummy_torch_and_transformers_objects 模块
    from ...utils import dummy_torch_and_transformers_objects

    # 更新 _dummy_objects 字典，填充占位对象
    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
# 如果没有异常，执行以下代码
else:
    # 将模型导入结构添加到 _import_structure 字典中
    _import_structure["modeling_stable_audio"] = ["StableAudioProjectionModel"]
    _import_structure["pipeline_stable_audio"] = ["StableAudioPipeline"]

# 检查类型或慢导入标志
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    # 尝试检测可用性
    try:
        # 再次检查 transformers 和 torch 是否可用，并且 transformers 版本是否满足要求
        if not (is_transformers_available() and is_torch_available() and is_transformers_version(">=", "4.27.0")):
            # 如果不满足条件，则抛出异常
            raise OptionalDependencyNotAvailable()
    # 捕获 OptionalDependencyNotAvailable 异常
    except OptionalDependencyNotAvailable:
        # 从 dummy_torch_and_transformers_objects 模块导入所有内容
        from ...utils.dummy_torch_and_transformers_objects import *

    # 如果没有异常，执行以下代码
    else:
        # 从 modeling_stable_audio 模块导入 StableAudioProjectionModel 类
        from .modeling_stable_audio import StableAudioProjectionModel
        # 从 pipeline_stable_audio 模块导入 StableAudioPipeline 类
        from .pipeline_stable_audio import StableAudioPipeline

# 如果不是类型检查或慢导入
else:
    # 导入 sys 模块
    import sys

    # 使用 _LazyModule 创建一个懒加载模块
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
    # 遍历 _dummy_objects 字典，将每个占位对象添加到当前模块
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
```