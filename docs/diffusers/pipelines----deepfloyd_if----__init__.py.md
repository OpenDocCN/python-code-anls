# `.\diffusers\pipelines\deepfloyd_if\__init__.py`

```py
# 从 typing 模块导入 TYPE_CHECKING，用于类型检查时的条件导入
from typing import TYPE_CHECKING

# 从上级目录的 utils 模块导入所需的工具函数和常量
from ...utils import (
    DIFFUSERS_SLOW_IMPORT,  # 导入一个标志，用于慢速导入的检查
    OptionalDependencyNotAvailable,  # 导入自定义异常，用于处理可选依赖不可用的情况
    _LazyModule,  # 导入懒加载模块的类
    get_objects_from_module,  # 导入函数，从模块获取对象
    is_torch_available,  # 导入函数，检查 PyTorch 是否可用
    is_transformers_available,  # 导入函数，检查 Transformers 是否可用
)

# 初始化一个空字典，用于存储虚拟对象
_dummy_objects = {}
# 定义一个导入结构字典，列出不同模块的导入项
_import_structure = {
    "timesteps": [  # timesteps 模块的导入项列表
        "fast27_timesteps",
        "smart100_timesteps",
        "smart185_timesteps",
        "smart27_timesteps",
        "smart50_timesteps",
        "super100_timesteps",
        "super27_timesteps",
        "super40_timesteps",
    ]
}

# 尝试块，用于检查所需库的可用性
try:
    # 检查 Transformers 和 PyTorch 是否都可用，如果不可用则抛出异常
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()  # 抛出可选依赖不可用的异常
# 捕获可选依赖不可用的异常
except OptionalDependencyNotAvailable:
    # 从 utils 模块导入虚拟对象以处理依赖不可用的情况
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403

    # 更新虚拟对象字典，填充从 dummy 对象模块获取的对象
    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
# 如果没有异常，继续执行以下代码
else:
    # 向导入结构字典添加与管道相关的项
    _import_structure["pipeline_if"] = ["IFPipeline"]
    _import_structure["pipeline_if_img2img"] = ["IFImg2ImgPipeline"]
    _import_structure["pipeline_if_img2img_superresolution"] = ["IFImg2ImgSuperResolutionPipeline"]
    _import_structure["pipeline_if_inpainting"] = ["IFInpaintingPipeline"]
    _import_structure["pipeline_if_inpainting_superresolution"] = ["IFInpaintingSuperResolutionPipeline"]
    _import_structure["pipeline_if_superresolution"] = ["IFSuperResolutionPipeline"]
    _import_structure["pipeline_output"] = ["IFPipelineOutput"]
    _import_structure["safety_checker"] = ["IFSafetyChecker"]
    _import_structure["watermark"] = ["IFWatermarker"]

# 检查类型检查或慢速导入标志
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    # 尝试块，检查所需库的可用性
    try:
        # 检查 Transformers 和 PyTorch 是否都可用
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()  # 抛出可选依赖不可用的异常

    # 捕获可选依赖不可用的异常
    except OptionalDependencyNotAvailable:
        # 从虚拟对象模块导入所有内容
        from ...utils.dummy_torch_and_transformers_objects import *
    # 如果没有异常，继续执行以下代码
    else:
        # 从各个管道模块导入所需的类
        from .pipeline_if import IFPipeline
        from .pipeline_if_img2img import IFImg2ImgPipeline
        from .pipeline_if_img2img_superresolution import IFImg2ImgSuperResolutionPipeline
        from .pipeline_if_inpainting import IFInpaintingPipeline
        from .pipeline_if_inpainting_superresolution import IFInpaintingSuperResolutionPipeline
        from .pipeline_if_superresolution import IFSuperResolutionPipeline
        from .pipeline_output import IFPipelineOutput
        from .safety_checker import IFSafetyChecker
        # 从 timesteps 模块导入所需的时间步长对象
        from .timesteps import (
            fast27_timesteps,
            smart27_timesteps,
            smart50_timesteps,
            smart100_timesteps,
            smart185_timesteps,
            super27_timesteps,
            super40_timesteps,
            super100_timesteps,
        )
        # 从 watermark 模块导入水印对象
        from .watermark import IFWatermarker

# 如果不进行类型检查且没有慢速导入标志
else:
    # 导入 sys 模块
    import sys

    # 将当前模块替换为懒加载模块
    sys.modules[__name__] = _LazyModule(
        __name__,  # 模块名称
        globals()["__file__"],  # 当前文件路径
        _import_structure,  # 导入结构
        module_spec=__spec__,  # 模块的规格
    )

    # 遍历虚拟对象字典，将每个对象添加到当前模块
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)  # 设置模块属性
```