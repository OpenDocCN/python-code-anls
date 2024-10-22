# `.\diffusers\pipelines\pixart_alpha\__init__.py`

```py
# 导入类型检查相关的类型
from typing import TYPE_CHECKING

# 从父目录导入所需的工具和依赖
from ...utils import (
    DIFFUSERS_SLOW_IMPORT,  # 慢导入的标志
    OptionalDependencyNotAvailable,  # 可选依赖不可用的异常
    _LazyModule,  # 懒加载模块的工具
    get_objects_from_module,  # 从模块中获取对象的工具
    is_torch_available,  # 检查 PyTorch 是否可用
    is_transformers_available,  # 检查 Transformers 是否可用
)

# 初始化一个空字典以存储虚拟对象
_dummy_objects = {}
# 初始化一个字典以存储模块的导入结构
_import_structure = {}

# 尝试检查依赖是否可用
try:
    # 如果 Transformers 和 Torch 都不可用，抛出异常
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
# 捕获可选依赖不可用的异常
except OptionalDependencyNotAvailable:
    # 从工具中导入虚拟对象以避免错误
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403

    # 更新虚拟对象字典
    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
# 如果依赖可用，更新导入结构
else:
    # 在导入结构中添加 PixArtAlphaPipeline
    _import_structure["pipeline_pixart_alpha"] = ["PixArtAlphaPipeline"]
    # 在导入结构中添加 PixArtSigmaPipeline
    _import_structure["pipeline_pixart_sigma"] = ["PixArtSigmaPipeline"]

# 如果在类型检查或慢导入模式下，进行以下操作
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        # 检查 Transformers 和 Torch 是否可用
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()

    # 捕获可选依赖不可用的异常
    except OptionalDependencyNotAvailable:
        # 从工具中导入虚拟对象以避免错误
        from ...utils.dummy_torch_and_transformers_objects import *
    else:
        # 从 PixArtAlphaPipeline 模块中导入所需的对象
        from .pipeline_pixart_alpha import (
            ASPECT_RATIO_256_BIN,  # 256 比例的常量
            ASPECT_RATIO_512_BIN,  # 512 比例的常量
            ASPECT_RATIO_1024_BIN,  # 1024 比例的常量
            PixArtAlphaPipeline,  # PixArtAlphaPipeline 类
        )
        # 从 PixArtSigmaPipeline 模块中导入所需的对象
        from .pipeline_pixart_sigma import ASPECT_RATIO_2048_BIN, PixArtSigmaPipeline

# 如果不是类型检查或慢导入模式，执行以下操作
else:
    # 导入 sys 模块以进行模块操作
    import sys

    # 用懒加载模块替换当前模块
    sys.modules[__name__] = _LazyModule(
        __name__,  # 当前模块的名称
        globals()["__file__"],  # 当前模块的文件路径
        _import_structure,  # 模块的导入结构
        module_spec=__spec__,  # 模块的规范
    )

    # 遍历虚拟对象字典并设置当前模块的属性
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)  # 设置模块的属性
```