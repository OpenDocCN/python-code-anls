# `.\diffusers\pipelines\lumina\__init__.py`

```py
# 导入类型检查模块
from typing import TYPE_CHECKING

# 从相对路径的 utils 模块中导入多个工具和常量
from ...utils import (
    DIFFUSERS_SLOW_IMPORT,  # 表示是否慢加载
    OptionalDependencyNotAvailable,  # 可选依赖未找到异常
    _LazyModule,  # 延迟加载模块的工具
    get_objects_from_module,  # 从模块中获取对象的工具
    is_torch_available,  # 检查 PyTorch 是否可用的工具
    is_transformers_available,  # 检查 Transformers 是否可用的工具
)

# 存储虚拟对象的字典
_dummy_objects = {}
# 存储模块导入结构的字典
_import_structure = {}

# 尝试执行以下代码块
try:
    # 检查 Transformers 和 PyTorch 是否可用，若都不可用则抛出异常
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
# 捕获可选依赖未找到异常
except OptionalDependencyNotAvailable:
    # 从 utils 模块中导入虚拟对象（占位符）
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403

    # 更新 _dummy_objects 字典，加入虚拟对象
    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
# 如果没有异常，执行以下代码
else:
    # 在导入结构字典中添加键值对，表示可用的管道
    _import_structure["pipeline_lumina"] = ["LuminaText2ImgPipeline"]

# 如果是类型检查或者慢加载标志为真，执行以下代码块
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        # 再次检查 Transformers 和 PyTorch 是否可用，若都不可用则抛出异常
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()

    # 捕获可选依赖未找到异常
    except OptionalDependencyNotAvailable:
        # 从 utils 中导入虚拟对象（占位符）
        from ...utils.dummy_torch_and_transformers_objects import *
    else:
        # 从 pipeline_lumina 模块导入 LuminaText2ImgPipeline
        from .pipeline_lumina import LuminaText2ImgPipeline

# 否则，执行以下代码块
else:
    # 导入 sys 模块
    import sys

    # 将当前模块替换为延迟加载模块
    sys.modules[__name__] = _LazyModule(
        __name__,  # 模块名称
        globals()["__file__"],  # 当前文件路径
        _import_structure,  # 导入结构
        module_spec=__spec__,  # 模块规格
    )

    # 遍历 _dummy_objects 字典，将每个虚拟对象设置到当前模块
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
```