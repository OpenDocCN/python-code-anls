# `.\diffusers\pipelines\controlnet_xs\__init__.py`

```py
# 从 typing 模块导入 TYPE_CHECKING，用于类型检查
from typing import TYPE_CHECKING

# 从相对路径导入多个工具和依赖项
from ...utils import (
    DIFFUSERS_SLOW_IMPORT,  # 导入 DIFFUSERS_SLOW_IMPORT 常量
    OptionalDependencyNotAvailable,  # 导入可选依赖不可用异常
    _LazyModule,  # 导入延迟加载模块
    get_objects_from_module,  # 导入从模块获取对象的函数
    is_flax_available,  # 导入检查 Flax 是否可用的函数
    is_torch_available,  # 导入检查 PyTorch 是否可用的函数
    is_transformers_available,  # 导入检查 Transformers 是否可用的函数
)

# 初始化空字典，用于存放虚拟对象
_dummy_objects = {}
# 初始化空字典，用于存放导入结构
_import_structure = {}

# 尝试检查 Transformers 和 Torch 是否可用
try:
    if not (is_transformers_available() and is_torch_available()):
        # 如果不可用，则抛出异常
        raise OptionalDependencyNotAvailable()
# 捕获可选依赖不可用的异常
except OptionalDependencyNotAvailable:
    # 从工具中导入虚拟对象以避免错误
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403

    # 更新虚拟对象字典
    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    # 如果可用，更新导入结构以包含相关管道
    _import_structure["pipeline_controlnet_xs"] = ["StableDiffusionControlNetXSPipeline"]
    _import_structure["pipeline_controlnet_xs_sd_xl"] = ["StableDiffusionXLControlNetXSPipeline"]
# 尝试检查 Transformers 和 Flax 是否可用
try:
    if not (is_transformers_available() and is_flax_available()):
        # 如果不可用，则抛出异常
        raise OptionalDependencyNotAvailable()
# 捕获可选依赖不可用的异常
except OptionalDependencyNotAvailable:
    # 从工具中导入虚拟对象以避免错误
    from ...utils import dummy_flax_and_transformers_objects  # noqa F403

    # 更新虚拟对象字典
    _dummy_objects.update(get_objects_from_module(dummy_flax_and_transformers_objects))
else:
    # 如果可用，不执行任何操作（留空）
    pass  # _import_structure["pipeline_flax_controlnet"] = ["FlaxStableDiffusionControlNetPipeline"]

# 如果正在进行类型检查或慢导入，则执行以下代码
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    # 尝试检查 Transformers 和 Torch 是否可用
    try:
        if not (is_transformers_available() and is_torch_available()):
            # 如果不可用，则抛出异常
            raise OptionalDependencyNotAvailable()

    # 捕获可选依赖不可用的异常
    except OptionalDependencyNotAvailable:
        # 从工具中导入虚拟对象以避免错误
        from ...utils.dummy_torch_and_transformers_objects import *
    else:
        # 如果可用，导入相关的管道
        from .pipeline_controlnet_xs import StableDiffusionControlNetXSPipeline
        from .pipeline_controlnet_xs_sd_xl import StableDiffusionXLControlNetXSPipeline

    # 尝试检查 Transformers 和 Flax 是否可用
    try:
        if not (is_transformers_available() and is_flax_available()):
            # 如果不可用，则抛出异常
            raise OptionalDependencyNotAvailable()
    # 捕获可选依赖不可用的异常
    except OptionalDependencyNotAvailable:
        # 从工具中导入虚拟对象以避免错误
        from ...utils.dummy_flax_and_transformers_objects import *  # noqa F403
    else:
        # 如果可用，不执行任何操作（留空）
        pass  # from .pipeline_flax_controlnet import FlaxStableDiffusionControlNetPipeline

# 如果不进行类型检查或慢导入，执行以下代码
else:
    import sys  # 导入 sys 模块

    # 将当前模块替换为延迟加载模块
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],  # 当前文件的全局变量
        _import_structure,  # 导入结构
        module_spec=__spec__,  # 模块规格
    )
    # 遍历虚拟对象字典，设置模块属性
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
```