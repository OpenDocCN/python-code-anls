# `.\diffusers\pipelines\deprecated\versatile_diffusion\__init__.py`

```py
# 从类型检查模块导入 TYPE_CHECKING，用于静态类型检查
from typing import TYPE_CHECKING

# 从上级模块的 utils 导入所需的功能和常量
from ....utils import (
    DIFFUSERS_SLOW_IMPORT,  # 表示是否进行慢速导入
    OptionalDependencyNotAvailable,  # 表示可选依赖项不可用的异常
    _LazyModule,  # 用于延迟加载模块的工具
    is_torch_available,  # 检查 PyTorch 是否可用的函数
    is_transformers_available,  # 检查 Transformers 是否可用的函数
    is_transformers_version,  # 检查 Transformers 版本的函数
)

# 初始化一个空字典，用于存放假对象
_dummy_objects = {}
# 初始化一个空字典，用于存放模块的导入结构
_import_structure = {}

# 尝试检查必要的依赖项是否可用
try:
    # 如果 Transformers 和 PyTorch 不可用，或者 Transformers 版本不符合要求，抛出异常
    if not (is_transformers_available() and is_torch_available() and is_transformers_version(">=", "4.25.0")):
        raise OptionalDependencyNotAvailable()
# 捕获可选依赖项不可用的异常
except OptionalDependencyNotAvailable:
    # 从 utils 中导入假对象以避免缺失依赖时的错误
    from ....utils.dummy_torch_and_transformers_objects import (
        VersatileDiffusionDualGuidedPipeline,  # 导入双重引导管道
        VersatileDiffusionImageVariationPipeline,  # 导入图像变化管道
        VersatileDiffusionPipeline,  # 导入通用扩散管道
        VersatileDiffusionTextToImagePipeline,  # 导入文本到图像管道
    )

    # 更新假对象字典
    _dummy_objects.update(
        {
            "VersatileDiffusionDualGuidedPipeline": VersatileDiffusionDualGuidedPipeline,
            "VersatileDiffusionImageVariationPipeline": VersatileDiffusionImageVariationPipeline,
            "VersatileDiffusionPipeline": VersatileDiffusionPipeline,
            "VersatileDiffusionTextToImagePipeline": VersatileDiffusionTextToImagePipeline,
        }
    )
# 如果依赖项可用，则设置导入结构
else:
    _import_structure["modeling_text_unet"] = ["UNetFlatConditionModel"]  # 设置文本 UNet 模型的导入结构
    _import_structure["pipeline_versatile_diffusion"] = ["VersatileDiffusionPipeline"]  # 设置通用扩散管道的导入结构
    _import_structure["pipeline_versatile_diffusion_dual_guided"] = ["VersatileDiffusionDualGuidedPipeline"]  # 设置双重引导管道的导入结构
    _import_structure["pipeline_versatile_diffusion_image_variation"] = ["VersatileDiffusionImageVariationPipeline"]  # 设置图像变化管道的导入结构
    _import_structure["pipeline_versatile_diffusion_text_to_image"] = ["VersatileDiffusionTextToImagePipeline"]  # 设置文本到图像管道的导入结构

# 检查是否为类型检查或是否需要慢速导入
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        # 如果 Transformers 和 PyTorch 不可用，或者 Transformers 版本不符合要求，抛出异常
        if not (is_transformers_available() and is_torch_available() and is_transformers_version(">=", "4.25.0")):
            raise OptionalDependencyNotAvailable()
    # 捕获可选依赖项不可用的异常
    except OptionalDependencyNotAvailable:
        # 从 utils 中导入假对象以避免缺失依赖时的错误
        from ....utils.dummy_torch_and_transformers_objects import (
            VersatileDiffusionDualGuidedPipeline,  # 导入双重引导管道
            VersatileDiffusionImageVariationPipeline,  # 导入图像变化管道
            VersatileDiffusionPipeline,  # 导入通用扩散管道
            VersatileDiffusionTextToImagePipeline,  # 导入文本到图像管道
        )
    # 如果依赖项可用，则导入实际的管道实现
    else:
        from .pipeline_versatile_diffusion import VersatileDiffusionPipeline  # 导入通用扩散管道
        from .pipeline_versatile_diffusion_dual_guided import VersatileDiffusionDualGuidedPipeline  # 导入双重引导管道
        from .pipeline_versatile_diffusion_image_variation import VersatileDiffusionImageVariationPipeline  # 导入图像变化管道
        from .pipeline_versatile_diffusion_text_to_image import VersatileDiffusionTextToImagePipeline  # 导入文本到图像管道

# 如果不是类型检查或慢速导入
else:
    import sys  # 导入系统模块

    # 使用延迟加载模块的方式替换当前模块
    sys.modules[__name__] = _LazyModule(
        __name__,  # 模块名称
        globals()["__file__"],  # 当前文件路径
        _import_structure,  # 导入结构
        module_spec=__spec__,  # 模块的规范
    )

    # 将假对象的属性设置到当前模块
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)  # 设置假对象的属性
```