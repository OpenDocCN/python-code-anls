# `.\diffusers\schedulers\deprecated\__init__.py`

```py
# 从 typing 模块导入 TYPE_CHECKING，用于类型检查时的导入
from typing import TYPE_CHECKING

# 从上级模块的 utils 导入多个功能
from ...utils import (
    # 导入慢速导入的标志
    DIFFUSERS_SLOW_IMPORT,
    # 导入处理可选依赖项不可用的异常
    OptionalDependencyNotAvailable,
    # 导入延迟加载模块的工具
    _LazyModule,
    # 导入从模块获取对象的函数
    get_objects_from_module,
    # 导入检查 PyTorch 是否可用的函数
    is_torch_available,
    # 导入检查 transformers 是否可用的函数
    is_transformers_available,
)

# 初始化一个空字典，用于存储占位对象
_dummy_objects = {}
# 初始化一个空字典，用于存储导入结构
_import_structure = {}

# 尝试执行以下代码块
try:
    # 检查 transformers 和 torch 是否都可用
    if not (is_transformers_available() and is_torch_available()):
        # 如果不可用，则引发可选依赖项不可用异常
        raise OptionalDependencyNotAvailable()
# 捕获可选依赖项不可用的异常
except OptionalDependencyNotAvailable:
    # 从 utils 导入占位对象以避免错误
    from ...utils import dummy_pt_objects  # noqa F403

    # 更新占位对象字典，获取 dummy_pt_objects 中的对象
    _dummy_objects.update(get_objects_from_module(dummy_pt_objects))
# 如果没有异常，执行以下代码
else:
    # 定义可用模块的导入结构
    _import_structure["scheduling_karras_ve"] = ["KarrasVeScheduler"]
    _import_structure["scheduling_sde_vp"] = ["ScoreSdeVpScheduler"]

# 如果处于类型检查或启用了慢速导入，执行以下代码
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    # 尝试执行以下代码块
    try:
        # 检查 torch 是否可用
        if not is_torch_available():
            # 如果不可用，则引发可选依赖项不可用异常
            raise OptionalDependencyNotAvailable()
    # 捕获可选依赖项不可用的异常
    except OptionalDependencyNotAvailable:
        # 从 dummy_pt_objects 导入所有对象
        from ...utils.dummy_pt_objects import *  # noqa F403
    else:
        # 从调度模块导入 KarrasVeScheduler
        from .scheduling_karras_ve import KarrasVeScheduler
        # 从调度模块导入 ScoreSdeVpScheduler
        from .scheduling_sde_vp import ScoreSdeVpScheduler

# 否则，执行以下代码
else:
    # 导入 sys 模块以进行模块管理
    import sys

    # 使用 LazyModule 封装当前模块，支持延迟加载
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )

    # 遍历占位对象字典，将每个对象设置到当前模块中
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
```