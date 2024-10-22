# `.\diffusers\pipelines\animatediff\__init__.py`

```py
# 从 typing 模块导入 TYPE_CHECKING，用于静态类型检查
from typing import TYPE_CHECKING

# 从上级目录的 utils 模块导入多个工具和常量
from ...utils import (
    DIFFUSERS_SLOW_IMPORT,  # 用于判断是否需要慢速导入
    OptionalDependencyNotAvailable,  # 可选依赖未满足时的异常
    _LazyModule,  # 用于懒加载模块的工具
    get_objects_from_module,  # 从模块中获取对象的工具
    is_torch_available,  # 检查 PyTorch 是否可用
    is_transformers_available,  # 检查 Transformers 库是否可用
)

# 初始化一个空字典用于存储占位对象
_dummy_objects = {}
# 定义模块的导入结构，初始化 pipeline_output 的导入
_import_structure = {"pipeline_output": ["AnimateDiffPipelineOutput"]}

# 尝试检查依赖库是否可用
try:
    # 如果 Transformers 和 Torch 库都不可用，则抛出异常
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
# 捕获可选依赖未满足的异常
except OptionalDependencyNotAvailable:
    # 从 utils 模块导入占位对象
    from ...utils import dummy_torch_and_transformers_objects

    # 更新占位对象字典，获取占位对象
    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    # 如果依赖可用，更新导入结构以包含动画相关的管道
    _import_structure["pipeline_animatediff"] = ["AnimateDiffPipeline"]
    _import_structure["pipeline_animatediff_controlnet"] = ["AnimateDiffControlNetPipeline"]
    _import_structure["pipeline_animatediff_sdxl"] = ["AnimateDiffSDXLPipeline"]
    _import_structure["pipeline_animatediff_sparsectrl"] = ["AnimateDiffSparseControlNetPipeline"]
    _import_structure["pipeline_animatediff_video2video"] = ["AnimateDiffVideoToVideoPipeline"]

# 如果是类型检查或慢速导入，则执行以下代码
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        # 检查依赖库是否可用
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()
    # 捕获可选依赖未满足的异常
    except OptionalDependencyNotAvailable:
        # 从 utils 模块导入所有占位对象
        from ...utils.dummy_torch_and_transformers_objects import *

    else:
        # 如果依赖可用，导入动画管道的具体实现
        from .pipeline_animatediff import AnimateDiffPipeline
        from .pipeline_animatediff_controlnet import AnimateDiffControlNetPipeline
        from .pipeline_animatediff_sdxl import AnimateDiffSDXLPipeline
        from .pipeline_animatediff_sparsectrl import AnimateDiffSparseControlNetPipeline
        from .pipeline_animatediff_video2video import AnimateDiffVideoToVideoPipeline
        from .pipeline_output import AnimateDiffPipelineOutput

# 否则，执行懒加载
else:
    import sys

    # 用懒加载模块初始化当前模块
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,  # 使用之前定义的导入结构
        module_spec=__spec__,  # 模块的规范信息
    )
    # 将占位对象添加到当前模块
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
```