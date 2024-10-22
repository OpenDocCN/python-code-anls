# `.\diffusers\pipelines\stable_diffusion\__init__.py`

```py
# 导入类型检查相关的模块
from typing import TYPE_CHECKING

# 从工具模块导入多个依赖项和函数
from ...utils import (
    DIFFUSERS_SLOW_IMPORT,  # 慢速导入标志
    OptionalDependencyNotAvailable,  # 可选依赖未找到的异常
    _LazyModule,  # 懒加载模块的工具
    get_objects_from_module,  # 从模块获取对象的函数
    is_flax_available,  # 检查 Flax 库是否可用
    is_k_diffusion_available,  # 检查 K-Diffusion 库是否可用
    is_k_diffusion_version,  # 检查 K-Diffusion 版本
    is_onnx_available,  # 检查 ONNX 库是否可用
    is_torch_available,  # 检查 PyTorch 库是否可用
    is_transformers_available,  # 检查 Transformers 库是否可用
    is_transformers_version,  # 检查 Transformers 版本
)

# 创建空字典以存储虚拟对象
_dummy_objects = {}
# 创建空字典以存储额外导入
_additional_imports = {}
# 定义初始导入结构，包含管道输出
_import_structure = {"pipeline_output": ["StableDiffusionPipelineOutput"]}

# 如果 Transformers 和 Flax 可用，则扩展管道输出
if is_transformers_available() and is_flax_available():
    _import_structure["pipeline_output"].extend(["FlaxStableDiffusionPipelineOutput"])
# 尝试检查 PyTorch 和 Transformers 的可用性
try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()  # 如果不可用则引发异常
except OptionalDependencyNotAvailable:
    # 导入虚拟对象以处理缺少的依赖项
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403

    # 更新虚拟对象字典
    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    # 如果可用，更新导入结构，添加多个管道
    _import_structure["clip_image_project_model"] = ["CLIPImageProjection"]
    _import_structure["pipeline_cycle_diffusion"] = ["CycleDiffusionPipeline"]
    _import_structure["pipeline_stable_diffusion"] = ["StableDiffusionPipeline"]
    _import_structure["pipeline_stable_diffusion_attend_and_excite"] = ["StableDiffusionAttendAndExcitePipeline"]
    _import_structure["pipeline_stable_diffusion_gligen"] = ["StableDiffusionGLIGENPipeline"]
    _import_structure["pipeline_stable_diffusion_gligen_text_image"] = ["StableDiffusionGLIGENTextImagePipeline"]
    _import_structure["pipeline_stable_diffusion_img2img"] = ["StableDiffusionImg2ImgPipeline"]
    _import_structure["pipeline_stable_diffusion_inpaint"] = ["StableDiffusionInpaintPipeline"]
    _import_structure["pipeline_stable_diffusion_inpaint_legacy"] = ["StableDiffusionInpaintPipelineLegacy"]
    _import_structure["pipeline_stable_diffusion_instruct_pix2pix"] = ["StableDiffusionInstructPix2PixPipeline"]
    _import_structure["pipeline_stable_diffusion_latent_upscale"] = ["StableDiffusionLatentUpscalePipeline"]
    _import_structure["pipeline_stable_diffusion_model_editing"] = ["StableDiffusionModelEditingPipeline"]
    _import_structure["pipeline_stable_diffusion_paradigms"] = ["StableDiffusionParadigmsPipeline"]
    _import_structure["pipeline_stable_diffusion_upscale"] = ["StableDiffusionUpscalePipeline"]
    _import_structure["pipeline_stable_unclip"] = ["StableUnCLIPPipeline"]
    _import_structure["pipeline_stable_unclip_img2img"] = ["StableUnCLIPImg2ImgPipeline"]
    _import_structure["safety_checker"] = ["StableDiffusionSafetyChecker"]
    _import_structure["stable_unclip_image_normalizer"] = ["StableUnCLIPImageNormalizer"]
# 尝试检查更严格的 Transformers 和 PyTorch 依赖条件
try:
    if not (is_transformers_available() and is_torch_available() and is_transformers_version(">=", "4.25.0")):
        raise OptionalDependencyNotAvailable()  # 如果条件不满足则引发异常
except OptionalDependencyNotAvailable:
    # 导入缺失的管道以处理可选依赖
    from ...utils.dummy_torch_and_transformers_objects import (
        StableDiffusionImageVariationPipeline,  # 导入图像变体管道
    )
    # 更新 _dummy_objects 字典，将键 "StableDiffusionImageVariationPipeline" 映射到 StableDiffusionImageVariationPipeline 对象
    _dummy_objects.update({"StableDiffusionImageVariationPipeline": StableDiffusionImageVariationPipeline})
# 如果没有可选依赖，则添加 StableDiffusionImageVariationPipeline 到导入结构
else:
    _import_structure["pipeline_stable_diffusion_image_variation"] = ["StableDiffusionImageVariationPipeline"]

# 尝试检查依赖条件是否满足
try:
    # 判断 transformers 和 torch 是否可用，且 transformers 版本是否满足条件
    if not (is_transformers_available() and is_torch_available() and is_transformers_version(">=", "4.26.0")):
        # 如果条件不满足，抛出依赖不可用异常
        raise OptionalDependencyNotAvailable()
# 捕获依赖不可用的异常
except OptionalDependencyNotAvailable:
    # 从虚拟对象模块导入 StableDiffusionDepth2ImgPipeline
    from ...utils.dummy_torch_and_transformers_objects import (
        StableDiffusionDepth2ImgPipeline,
    )

    # 更新虚拟对象字典，添加 StableDiffusionDepth2ImgPipeline
    _dummy_objects.update(
        {
            "StableDiffusionDepth2ImgPipeline": StableDiffusionDepth2ImgPipeline,
        }
    )
# 如果没有抛出异常，则添加 StableDiffusionDepth2ImgPipeline 到导入结构
else:
    _import_structure["pipeline_stable_diffusion_depth2img"] = ["StableDiffusionDepth2ImgPipeline"]

# 尝试检查其他依赖条件是否满足
try:
    # 判断 transformers 和 onnx 是否可用
    if not (is_transformers_available() and is_onnx_available()):
        # 如果条件不满足，抛出依赖不可用异常
        raise OptionalDependencyNotAvailable()
# 捕获依赖不可用的异常
except OptionalDependencyNotAvailable:
    # 从虚拟 ONNX 对象模块导入
    from ...utils import dummy_onnx_objects  # noqa F403

    # 更新虚拟对象字典，添加 ONNX 对象
    _dummy_objects.update(get_objects_from_module(dummy_onnx_objects))
# 如果没有抛出异常，则添加 ONNX 相关的导入结构
else:
    _import_structure["pipeline_onnx_stable_diffusion"] = [
        "OnnxStableDiffusionPipeline",
        "StableDiffusionOnnxPipeline",
    ]
    _import_structure["pipeline_onnx_stable_diffusion_img2img"] = ["OnnxStableDiffusionImg2ImgPipeline"]
    _import_structure["pipeline_onnx_stable_diffusion_inpaint"] = ["OnnxStableDiffusionInpaintPipeline"]
    _import_structure["pipeline_onnx_stable_diffusion_inpaint_legacy"] = ["OnnxStableDiffusionInpaintPipelineLegacy"]
    _import_structure["pipeline_onnx_stable_diffusion_upscale"] = ["OnnxStableDiffusionUpscalePipeline"]

# 检查 transformers 和 flax 是否可用
if is_transformers_available() and is_flax_available():
    # 从调度器模块导入 PNDMSchedulerState
    from ...schedulers.scheduling_pndm_flax import PNDMSchedulerState

    # 更新额外导入字典，添加 PNDMSchedulerState
    _additional_imports.update({"PNDMSchedulerState": PNDMSchedulerState})
    # 添加 flax 相关的导入结构
    _import_structure["pipeline_flax_stable_diffusion"] = ["FlaxStableDiffusionPipeline"]
    _import_structure["pipeline_flax_stable_diffusion_img2img"] = ["FlaxStableDiffusionImg2ImgPipeline"]
    _import_structure["pipeline_flax_stable_diffusion_inpaint"] = ["FlaxStableDiffusionInpaintPipeline"]
    _import_structure["safety_checker_flax"] = ["FlaxStableDiffusionSafetyChecker"]

# 检查类型检查或慢导入条件
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    # 尝试检查依赖条件
    try:
        # 判断 transformers 和 torch 是否可用
        if not (is_transformers_available() and is_torch_available()):
            # 如果条件不满足，抛出依赖不可用异常
            raise OptionalDependencyNotAvailable()

    # 捕获依赖不可用的异常
    except OptionalDependencyNotAvailable:
        # 从虚拟对象模块导入所有内容
        from ...utils.dummy_torch_and_transformers_objects import *
    else:
        # 从模块中导入 CLIPImageProjection 类
        from .clip_image_project_model import CLIPImageProjection
        # 从模块中导入 StableDiffusionPipeline 和 StableDiffusionPipelineOutput 类
        from .pipeline_stable_diffusion import (
            StableDiffusionPipeline,
            StableDiffusionPipelineOutput,
        )
        # 从模块中导入 StableDiffusionImg2ImgPipeline 类
        from .pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipeline
        # 从模块中导入 StableDiffusionInpaintPipeline 类
        from .pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline
        # 从模块中导入 StableDiffusionInstructPix2PixPipeline 类
        from .pipeline_stable_diffusion_instruct_pix2pix import (
            StableDiffusionInstructPix2PixPipeline,
        )
        # 从模块中导入 StableDiffusionLatentUpscalePipeline 类
        from .pipeline_stable_diffusion_latent_upscale import (
            StableDiffusionLatentUpscalePipeline,
        )
        # 从模块中导入 StableDiffusionUpscalePipeline 类
        from .pipeline_stable_diffusion_upscale import StableDiffusionUpscalePipeline
        # 从模块中导入 StableUnCLIPPipeline 类
        from .pipeline_stable_unclip import StableUnCLIPPipeline
        # 从模块中导入 StableUnCLIPImg2ImgPipeline 类
        from .pipeline_stable_unclip_img2img import StableUnCLIPImg2ImgPipeline
        # 从模块中导入 StableDiffusionSafetyChecker 类
        from .safety_checker import StableDiffusionSafetyChecker
        # 从模块中导入 StableUnCLIPImageNormalizer 类
        from .stable_unclip_image_normalizer import StableUnCLIPImageNormalizer

    try:
        # 检查是否可用所需的 transformers 和 torch 库及其版本
        if not (is_transformers_available() and is_torch_available() and is_transformers_version(">=", "4.25.0")):
            # 抛出可选依赖不可用异常
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 导入 dummy_torch_and_transformers_objects 中的 StableDiffusionImageVariationPipeline 类
        from ...utils.dummy_torch_and_transformers_objects import (
            StableDiffusionImageVariationPipeline,
        )
    else:
        # 从模块中导入 StableDiffusionImageVariationPipeline 类
        from .pipeline_stable_diffusion_image_variation import (
            StableDiffusionImageVariationPipeline,
        )

    try:
        # 检查是否可用所需的 transformers 和 torch 库及其版本
        if not (is_transformers_available() and is_torch_available() and is_transformers_version(">=", "4.26.0")):
            # 抛出可选依赖不可用异常
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 导入 dummy_torch_and_transformers_objects 中的 StableDiffusionDepth2ImgPipeline 类
        from ...utils.dummy_torch_and_transformers_objects import StableDiffusionDepth2ImgPipeline
    else:
        # 从模块中导入 StableDiffusionDepth2ImgPipeline 类
        from .pipeline_stable_diffusion_depth2img import (
            StableDiffusionDepth2ImgPipeline,
        )

    try:
        # 检查是否可用所需的 transformers 和 onnx 库
        if not (is_transformers_available() and is_onnx_available()):
            # 抛出可选依赖不可用异常
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 从 dummy_onnx_objects 中导入所有对象
        from ...utils.dummy_onnx_objects import *
    else:
        # 从模块中导入 OnnxStableDiffusionPipeline 和 StableDiffusionOnnxPipeline 类
        from .pipeline_onnx_stable_diffusion import (
            OnnxStableDiffusionPipeline,
            StableDiffusionOnnxPipeline,
        )
        # 从模块中导入 OnnxStableDiffusionImg2ImgPipeline 类
        from .pipeline_onnx_stable_diffusion_img2img import (
            OnnxStableDiffusionImg2ImgPipeline,
        )
        # 从模块中导入 OnnxStableDiffusionInpaintPipeline 类
        from .pipeline_onnx_stable_diffusion_inpaint import (
            OnnxStableDiffusionInpaintPipeline,
        )
        # 从模块中导入 OnnxStableDiffusionUpscalePipeline 类
        from .pipeline_onnx_stable_diffusion_upscale import (
            OnnxStableDiffusionUpscalePipeline,
        )

    try:
        # 检查是否可用所需的 transformers 和 flax 库
        if not (is_transformers_available() and is_flax_available()):
            # 抛出可选依赖不可用异常
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 从 dummy_flax_objects 中导入所有对象
        from ...utils.dummy_flax_objects import *
    # 如果条件不满足，则执行以下导入操作
        else:
            # 从模块中导入 FlaxStableDiffusionPipeline 类
            from .pipeline_flax_stable_diffusion import FlaxStableDiffusionPipeline
            # 从模块中导入 FlaxStableDiffusionImg2ImgPipeline 类
            from .pipeline_flax_stable_diffusion_img2img import (
                FlaxStableDiffusionImg2ImgPipeline,
            )
            # 从模块中导入 FlaxStableDiffusionInpaintPipeline 类
            from .pipeline_flax_stable_diffusion_inpaint import (
                FlaxStableDiffusionInpaintPipeline,
            )
            # 从模块中导入 FlaxStableDiffusionPipelineOutput 类
            from .pipeline_output import FlaxStableDiffusionPipelineOutput
            # 从模块中导入 FlaxStableDiffusionSafetyChecker 类
            from .safety_checker_flax import FlaxStableDiffusionSafetyChecker
# 否则分支，处理模块的懒加载
else:
    # 导入 sys 模块以便操作模块相关功能
    import sys

    # 将当前模块名映射到一个懒加载模块实例
    sys.modules[__name__] = _LazyModule(
        __name__,  # 当前模块的名称
        globals()["__file__"],  # 当前模块的文件路径
        _import_structure,  # 定义的导入结构
        module_spec=__spec__,  # 当前模块的规格
    )

    # 遍历虚拟对象字典，将每个对象设置到当前模块中
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
    # 遍历附加导入字典，将每个对象设置到当前模块中
    for name, value in _additional_imports.items():
        setattr(sys.modules[__name__], name, value)
```