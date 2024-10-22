# `.\diffusers\pipelines\aura_flow\__init__.py`

```py
# 从 typing 模块导入 TYPE_CHECKING，以支持类型检查
from typing import TYPE_CHECKING

# 从 utils 模块导入所需的工具和常量
from ...utils import (
    DIFFUSERS_SLOW_IMPORT,  # 导入标识符，用于判断慢导入
    OptionalDependencyNotAvailable,  # 导入自定义异常，用于处理可选依赖未找到的情况
    _LazyModule,  # 导入懒加载模块的类
    get_objects_from_module,  # 导入从模块获取对象的函数
    is_torch_available,  # 导入检查 PyTorch 是否可用的函数
    is_transformers_available,  # 导入检查 Transformers 是否可用的函数
)

# 创建一个空字典用于存储虚拟对象
_dummy_objects = {}
# 创建一个字典用于存储模块的导入结构
_import_structure = {}

# 尝试检查依赖关系
try:
    # 如果 Transformers 和 PyTorch 都不可用，抛出异常
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
# 捕获可选依赖未找到的异常
except OptionalDependencyNotAvailable:
    # 从 utils 导入虚拟对象模块，避免依赖缺失时的错误
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403

    # 更新 _dummy_objects 字典，填充虚拟对象
    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
# 如果依赖可用，更新导入结构字典
else:
    # 将 "pipeline_aura_flow" 添加到导入结构字典
    _import_structure["pipeline_aura_flow"] = ["AuraFlowPipeline"]

# 检查是否处于类型检查模式或是否为慢导入
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        # 如果 Transformers 和 PyTorch 都不可用，抛出异常
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()

    # 捕获可选依赖未找到的异常
    except OptionalDependencyNotAvailable:
        # 从虚拟对象模块导入所有对象
        from ...utils.dummy_torch_and_transformers_objects import *
    # 如果依赖可用，从指定模块导入 AuraFlowPipeline
    else:
        from .pipeline_aura_flow import AuraFlowPipeline

# 如果不在类型检查模式或慢导入
else:
    # 导入 sys 模块以进行模块操作
    import sys

    # 将当前模块的 sys.modules 条目替换为 _LazyModule 的实例
    sys.modules[__name__] = _LazyModule(
        __name__,  # 当前模块名称
        globals()["__file__"],  # 当前模块的文件路径
        _import_structure,  # 导入结构字典
        module_spec=__spec__,  # 模块规范
    )

    # 遍历 _dummy_objects 字典，将每个虚拟对象设置到当前模块
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
```