# `.\diffusers\pipelines\latte\__init__.py`

```py
# 导入类型检查模块
from typing import TYPE_CHECKING

# 从相对路径导入多个实用工具函数和常量
from ...utils import (
    DIFFUSERS_SLOW_IMPORT,  # 用于判断是否为慢导入
    OptionalDependencyNotAvailable,  # 表示可选依赖不可用的异常
    _LazyModule,  # 懒加载模块的类
    get_objects_from_module,  # 从模块中获取对象的函数
    is_torch_available,  # 检查 PyTorch 是否可用的函数
    is_transformers_available,  # 检查 Transformers 是否可用的函数
)

# 存放空对象的字典
_dummy_objects = {}
# 存放导入结构的字典
_import_structure = {}

# 尝试检查是否可以使用 transformers 和 torch
try:
    if not (is_transformers_available() and is_torch_available()):  # 检查两个库的可用性
        raise OptionalDependencyNotAvailable()  # 如果不可用，抛出异常
except OptionalDependencyNotAvailable:  # 捕获可选依赖不可用的异常
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403  # 导入虚拟对象以避免错误

    # 更新 _dummy_objects 字典，填充虚拟对象
    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    # 如果可用，将 "pipeline_latte" 加入导入结构
    _import_structure["pipeline_latte"] = ["LattePipeline"]

# 检查是否进行类型检查或慢导入
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):  # 再次检查库的可用性
            raise OptionalDependencyNotAvailable()  # 如果不可用，抛出异常

    except OptionalDependencyNotAvailable:  # 捕获可选依赖不可用的异常
        from ...utils.dummy_torch_and_transformers_objects import *  # 导入虚拟对象以避免错误
    else:
        # 如果可用，导入 LattePipeline
        from .pipeline_latte import LattePipeline

else:
    import sys  # 导入系统模块

    # 用懒加载模块替换当前模块
    sys.modules[__name__] = _LazyModule(
        __name__,  # 当前模块名称
        globals()["__file__"],  # 当前模块文件路径
        _import_structure,  # 导入结构
        module_spec=__spec__,  # 模块规范
    )

    # 将 _dummy_objects 中的虚拟对象添加到当前模块中
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)  # 设置模块属性
```