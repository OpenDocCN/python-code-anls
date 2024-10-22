# `.\diffusers\pipelines\controlnet\__init__.py`

```py
# 导入类型检查工具
from typing import TYPE_CHECKING

# 从 utils 模块导入必要的工具和常量
from ...utils import (
    DIFFUSERS_SLOW_IMPORT,  # 导入慢导入标志
    OptionalDependencyNotAvailable,  # 导入可选依赖不可用异常
    _LazyModule,  # 导入延迟模块工具
    get_objects_from_module,  # 导入从模块获取对象的函数
    is_flax_available,  # 导入检查 Flax 可用性的函数
    is_torch_available,  # 导入检查 Torch 可用性的函数
    is_transformers_available,  # 导入检查 Transformers 可用性的函数
)

# 初始化一个空字典用于存储虚拟对象
_dummy_objects = {}
# 初始化一个空字典用于存储导入结构
_import_structure = {}

try:
    # 检查 Transformers 和 Torch 是否可用
    if not (is_transformers_available() and is_torch_available()):
        # 如果不可用，抛出异常
        raise OptionalDependencyNotAvailable()
# 捕获可选依赖不可用的异常
except OptionalDependencyNotAvailable:
    # 从 utils 导入虚拟对象
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403

    # 更新虚拟对象字典
    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    # 如果依赖可用，更新导入结构字典
    _import_structure["multicontrolnet"] = ["MultiControlNetModel"]
    _import_structure["pipeline_controlnet"] = ["StableDiffusionControlNetPipeline"]
    _import_structure["pipeline_controlnet_blip_diffusion"] = ["BlipDiffusionControlNetPipeline"]
    _import_structure["pipeline_controlnet_img2img"] = ["StableDiffusionControlNetImg2ImgPipeline"]
    _import_structure["pipeline_controlnet_inpaint"] = ["StableDiffusionControlNetInpaintPipeline"]
    _import_structure["pipeline_controlnet_inpaint_sd_xl"] = ["StableDiffusionXLControlNetInpaintPipeline"]
    _import_structure["pipeline_controlnet_sd_xl"] = ["StableDiffusionXLControlNetPipeline"]
    _import_structure["pipeline_controlnet_sd_xl_img2img"] = ["StableDiffusionXLControlNetImg2ImgPipeline"]

try:
    # 检查 Transformers 和 Flax 是否可用
    if not (is_transformers_available() and is_flax_available()):
        # 如果不可用，抛出异常
        raise OptionalDependencyNotAvailable()
# 捕获可选依赖不可用的异常
except OptionalDependencyNotAvailable:
    # 从 utils 导入虚拟 Flax 和 Transformers 对象
    from ...utils import dummy_flax_and_transformers_objects  # noqa F403

    # 更新虚拟对象字典
    _dummy_objects.update(get_objects_from_module(dummy_flax_and_transformers_objects))
else:
    # 如果依赖可用，更新导入结构字典
    _import_structure["pipeline_flax_controlnet"] = ["FlaxStableDiffusionControlNetPipeline"]

# 如果类型检查或慢导入标志被设置
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        # 检查 Transformers 和 Torch 是否可用
        if not (is_transformers_available() and is_torch_available()):
            # 如果不可用，抛出异常
            raise OptionalDependencyNotAvailable()

    # 捕获可选依赖不可用的异常
    except OptionalDependencyNotAvailable:
        # 导入虚拟的 Torch 和 Transformers 对象
        from ...utils.dummy_torch_and_transformers_objects import *
    else:
        # 如果依赖可用，导入相应模块
        from .multicontrolnet import MultiControlNetModel
        from .pipeline_controlnet import StableDiffusionControlNetPipeline
        from .pipeline_controlnet_blip_diffusion import BlipDiffusionControlNetPipeline
        from .pipeline_controlnet_img2img import StableDiffusionControlNetImg2ImgPipeline
        from .pipeline_controlnet_inpaint import StableDiffusionControlNetInpaintPipeline
        from .pipeline_controlnet_inpaint_sd_xl import StableDiffusionXLControlNetInpaintPipeline
        from .pipeline_controlnet_sd_xl import StableDiffusionXLControlNetPipeline
        from .pipeline_controlnet_sd_xl_img2img import StableDiffusionXLControlNetImg2ImgPipeline

    try:
        # 检查 Transformers 和 Flax 是否可用
        if not (is_transformers_available() and is_flax_available()):
            # 如果不可用，抛出异常
            raise OptionalDependencyNotAvailable()
    # 捕获可选依赖项不可用的异常
        except OptionalDependencyNotAvailable:
            # 从 dummy 模块导入所有内容，忽略 F403 警告
            from ...utils.dummy_flax_and_transformers_objects import *  # noqa F403
        else:
            # 从 pipeline_flax_controlnet 模块导入 FlaxStableDiffusionControlNetPipeline
            from .pipeline_flax_controlnet import FlaxStableDiffusionControlNetPipeline
# 如果之前的条件不满足，执行以下代码
else:
    # 导入 sys 模块，用于访问和操作 Python 解释器的运行时环境
    import sys

    # 将当前模块替换为一个延迟加载的模块对象
    sys.modules[__name__] = _LazyModule(
        __name__,  # 模块名称
        globals()["__file__"],  # 当前文件的路径
        _import_structure,  # 模块的导入结构
        module_spec=__spec__,  # 模块的规格对象
    )
    # 遍历假对象字典，将每个对象属性设置到当前模块
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)  # 设置模块属性
```