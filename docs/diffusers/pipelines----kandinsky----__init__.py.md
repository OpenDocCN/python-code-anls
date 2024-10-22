# `.\diffusers\pipelines\kandinsky\__init__.py`

```py
# 从 typing 模块导入 TYPE_CHECKING，用于类型检查
from typing import TYPE_CHECKING

# 从 utils 模块导入相关的工具和常量
from ...utils import (
    DIFFUSERS_SLOW_IMPORT,  # 导入用于标识慢速导入的常量
    OptionalDependencyNotAvailable,  # 导入表示可选依赖不可用的异常
    _LazyModule,  # 导入懒加载模块的类
    get_objects_from_module,  # 导入从模块获取对象的函数
    is_torch_available,  # 导入检查 Torch 库可用性的函数
    is_transformers_available,  # 导入检查 Transformers 库可用性的函数
)

# 用于存储虚拟对象的字典
_dummy_objects = {}
# 用于存储模块导入结构的字典
_import_structure = {}

try:
    # 检查 Transformers 和 Torch 是否都可用
    if not (is_transformers_available() and is_torch_available()):
        # 如果不可用，则抛出可选依赖不可用异常
        raise OptionalDependencyNotAvailable()
# 捕获可选依赖不可用异常
except OptionalDependencyNotAvailable:
    # 从 utils 模块导入虚拟的 Torch 和 Transformers 对象，忽略 F403 警告
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403

    # 更新 _dummy_objects 字典，获取虚拟对象
    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    # 如果可用，则更新导入结构字典，添加各种管道的导入
    _import_structure["pipeline_kandinsky"] = ["KandinskyPipeline"]
    _import_structure["pipeline_kandinsky_combined"] = [
        "KandinskyCombinedPipeline",
        "KandinskyImg2ImgCombinedPipeline",
        "KandinskyInpaintCombinedPipeline",
    ]
    _import_structure["pipeline_kandinsky_img2img"] = ["KandinskyImg2ImgPipeline"]
    _import_structure["pipeline_kandinsky_inpaint"] = ["KandinskyInpaintPipeline"]
    _import_structure["pipeline_kandinsky_prior"] = ["KandinskyPriorPipeline", "KandinskyPriorPipelineOutput"]
    _import_structure["text_encoder"] = ["MultilingualCLIP"]

# 如果在类型检查模式或慢速导入模式下执行
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        # 再次检查 Transformers 和 Torch 是否都可用
        if not (is_transformers_available() and is_torch_available()):
            # 如果不可用，则抛出可选依赖不可用异常
            raise OptionalDependencyNotAvailable()
    # 捕获可选依赖不可用异常
    except OptionalDependencyNotAvailable:
        # 从虚拟对象模块导入所有内容
        from ...utils.dummy_torch_and_transformers_objects import *

    else:
        # 从各个管道模块导入实际的类
        from .pipeline_kandinsky import KandinskyPipeline
        from .pipeline_kandinsky_combined import (
            KandinskyCombinedPipeline,
            KandinskyImg2ImgCombinedPipeline,
            KandinskyInpaintCombinedPipeline,
        )
        from .pipeline_kandinsky_img2img import KandinskyImg2ImgPipeline
        from .pipeline_kandinsky_inpaint import KandinskyInpaintPipeline
        from .pipeline_kandinsky_prior import KandinskyPriorPipeline, KandinskyPriorPipelineOutput
        from .text_encoder import MultilingualCLIP

else:
    # 如果不是在类型检查模式或慢速导入模式，则使用懒加载模块
    import sys

    # 将当前模块的引用替换为懒加载模块
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,  # 传入导入结构
        module_spec=__spec__,  # 传入模块规范
    )

    # 将虚拟对象添加到当前模块
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
```