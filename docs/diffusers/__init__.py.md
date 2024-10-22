# `.\diffusers\__init__.py`

```py
# 定义当前版本号
__version__ = "0.30.3"

# 导入类型检查功能
from typing import TYPE_CHECKING

# 从 utils 模块中导入多个对象
from .utils import (
    DIFFUSERS_SLOW_IMPORT,  # 导入慢速导入标志
    OptionalDependencyNotAvailable,  # 导入可选依赖项不可用异常
    _LazyModule,  # 导入懒加载模块工具
    is_flax_available,  # 检查 Flax 是否可用
    is_k_diffusion_available,  # 检查 K Diffusion 是否可用
    is_librosa_available,  # 检查 Librosa 是否可用
    is_note_seq_available,  # 检查 NoteSeq 是否可用
    is_onnx_available,  # 检查 ONNX 是否可用
    is_scipy_available,  # 检查 SciPy 是否可用
    is_sentencepiece_available,  # 检查 SentencePiece 是否可用
    is_torch_available,  # 检查 PyTorch 是否可用
    is_torchsde_available,  # 检查 Torchsde 是否可用
    is_transformers_available,  # 检查 Transformers 是否可用
)

# 懒加载导入的基础
# 懒加载是为了提高导入效率

# 添加新对象到初始化时，请将其添加到 `_import_structure` 中
# `_import_structure` 是一个字典，列出对象名称，用于延迟实际导入
# 这样 `import diffusers` 提供命名空间中的名称，而不实际导入任何内容

_import_structure = {
    "configuration_utils": ["ConfigMixin"],  # 配置工具的混合类
    "loaders": ["FromOriginalModelMixin"],  # 加载器的混合类
    "models": [],  # 模型列表初始化为空
    "pipelines": [],  # 流水线列表初始化为空
    "schedulers": [],  # 调度器列表初始化为空
    "utils": [  # 工具列表
        "OptionalDependencyNotAvailable",  # 可选依赖项不可用异常
        "is_flax_available",  # Flax 可用性检查
        "is_inflect_available",  # Inflect 可用性检查
        "is_invisible_watermark_available",  # 隐形水印可用性检查
        "is_k_diffusion_available",  # K Diffusion 可用性检查
        "is_k_diffusion_version",  # K Diffusion 版本检查
        "is_librosa_available",  # Librosa 可用性检查
        "is_note_seq_available",  # NoteSeq 可用性检查
        "is_onnx_available",  # ONNX 可用性检查
        "is_scipy_available",  # SciPy 可用性检查
        "is_torch_available",  # PyTorch 可用性检查
        "is_torchsde_available",  # Torchsde 可用性检查
        "is_transformers_available",  # Transformers 可用性检查
        "is_transformers_version",  # Transformers 版本检查
        "is_unidecode_available",  # Unidecode 可用性检查
        "logging",  # 日志记录工具
    ],
}

# 尝试检查 ONNX 是否可用
try:
    if not is_onnx_available():  # 如果 ONNX 不可用
        raise OptionalDependencyNotAvailable()  # 抛出可选依赖项不可用异常
except OptionalDependencyNotAvailable:  # 捕获异常
    from .utils import dummy_onnx_objects  # 导入假 ONNX 对象

    # 将假对象添加到导入结构中
    _import_structure["utils.dummy_onnx_objects"] = [
        name for name in dir(dummy_onnx_objects) if not name.startswith("_")  # 排除以 "_" 开头的名称
    ]

else:  # 如果没有抛出异常
    # 将 ONNX 模型添加到流水线列表中
    _import_structure["pipelines"].extend(["OnnxRuntimeModel"])

# 尝试检查 PyTorch 是否可用
try:
    if not is_torch_available():  # 如果 PyTorch 不可用
        raise OptionalDependencyNotAvailable()  # 抛出可选依赖项不可用异常
except OptionalDependencyNotAvailable:  # 捕获异常
    from .utils import dummy_pt_objects  # 导入假 PyTorch 对象

    # 将假对象添加到导入结构中
    _import_structure["utils.dummy_pt_objects"] = [name for name in dir(dummy_pt_objects) if not name.startswith("_")]

else:  # 如果没有抛出异常
    # 扩展模型导入结构，将新模型名称添加到“models”列表中
        _import_structure["models"].extend(
            [   
                # 添加不同类型的模型名称
                "AsymmetricAutoencoderKL",
                "AuraFlowTransformer2DModel",
                "AutoencoderKL",
                "AutoencoderKLCogVideoX",
                "AutoencoderKLTemporalDecoder",
                "AutoencoderOobleck",
                "AutoencoderTiny",
                "CogVideoXTransformer3DModel",
                "ConsistencyDecoderVAE",
                "ControlNetModel",
                "ControlNetXSAdapter",
                "DiTTransformer2DModel",
                "FluxTransformer2DModel",
                "HunyuanDiT2DControlNetModel",
                "HunyuanDiT2DModel",
                "HunyuanDiT2DMultiControlNetModel",
                "I2VGenXLUNet",
                "Kandinsky3UNet",
                "LatteTransformer3DModel",
                "LuminaNextDiT2DModel",
                "ModelMixin",
                "MotionAdapter",
                "MultiAdapter",
                "PixArtTransformer2DModel",
                "PriorTransformer",
                "SD3ControlNetModel",
                "SD3MultiControlNetModel",
                "SD3Transformer2DModel",
                "SparseControlNetModel",
                "StableAudioDiTModel",
                "StableCascadeUNet",
                "T2IAdapter",
                "T5FilmDecoder",
                "Transformer2DModel",
                "UNet1DModel",
                "UNet2DConditionModel",
                "UNet2DModel",
                "UNet3DConditionModel",
                "UNetControlNetXSModel",
                "UNetMotionModel",
                "UNetSpatioTemporalConditionModel",
                "UVit2DModel",
                "VQModel",
            ]
        )
    
        # 设置优化相关的导入结构，包括不同的调度函数
        _import_structure["optimization"] = [
            "get_constant_schedule",  # 获取常数调度
            "get_constant_schedule_with_warmup",  # 获取带预热的常数调度
            "get_cosine_schedule_with_warmup",  # 获取带预热的余弦调度
            "get_cosine_with_hard_restarts_schedule_with_warmup",  # 获取带硬重启的余弦调度
            "get_linear_schedule_with_warmup",  # 获取带预热的线性调度
            "get_polynomial_decay_schedule_with_warmup",  # 获取带预热的多项式衰减调度
            "get_scheduler",  # 获取调度器
        ]
        # 扩展管道导入结构，将新管道名称添加到“pipelines”列表中
        _import_structure["pipelines"].extend(
            [   
                # 添加不同类型的管道名称
                "AudioPipelineOutput",
                "AutoPipelineForImage2Image",
                "AutoPipelineForInpainting",
                "AutoPipelineForText2Image",
                "ConsistencyModelPipeline",
                "DanceDiffusionPipeline",
                "DDIMPipeline",
                "DDPMPipeline",
                "DiffusionPipeline",
                "DiTPipeline",
                "ImagePipelineOutput",
                "KarrasVePipeline",
                "LDMPipeline",
                "LDMSuperResolutionPipeline",
                "PNDMPipeline",
                "RePaintPipeline",
                "ScoreSdeVePipeline",
                "StableDiffusionMixin",
            ]
        )
    # 扩展调度器（schedulers）的导入结构列表
        _import_structure["schedulers"].extend(
            [  # 添加多个调度器的名称到列表中
                "AmusedScheduler",  # 添加 AmusedScheduler
                "CMStochasticIterativeScheduler",  # 添加 CMStochasticIterativeScheduler
                "CogVideoXDDIMScheduler",  # 添加 CogVideoXDDIMScheduler
                "CogVideoXDPMScheduler",  # 添加 CogVideoXDPMScheduler
                "DDIMInverseScheduler",  # 添加 DDIMInverseScheduler
                "DDIMParallelScheduler",  # 添加 DDIMParallelScheduler
                "DDIMScheduler",  # 添加 DDIMScheduler
                "DDPMParallelScheduler",  # 添加 DDPMParallelScheduler
                "DDPMScheduler",  # 添加 DDPMScheduler
                "DDPMWuerstchenScheduler",  # 添加 DDPMWuerstchenScheduler
                "DEISMultistepScheduler",  # 添加 DEISMultistepScheduler
                "DPMSolverMultistepInverseScheduler",  # 添加 DPMSolverMultistepInverseScheduler
                "DPMSolverMultistepScheduler",  # 添加 DPMSolverMultistepScheduler
                "DPMSolverSinglestepScheduler",  # 添加 DPMSolverSinglestepScheduler
                "EDMDPMSolverMultistepScheduler",  # 添加 EDMDPMSolverMultistepScheduler
                "EDMEulerScheduler",  # 添加 EDMEulerScheduler
                "EulerAncestralDiscreteScheduler",  # 添加 EulerAncestralDiscreteScheduler
                "EulerDiscreteScheduler",  # 添加 EulerDiscreteScheduler
                "FlowMatchEulerDiscreteScheduler",  # 添加 FlowMatchEulerDiscreteScheduler
                "FlowMatchHeunDiscreteScheduler",  # 添加 FlowMatchHeunDiscreteScheduler
                "HeunDiscreteScheduler",  # 添加 HeunDiscreteScheduler
                "IPNDMScheduler",  # 添加 IPNDMScheduler
                "KarrasVeScheduler",  # 添加 KarrasVeScheduler
                "KDPM2AncestralDiscreteScheduler",  # 添加 KDPM2AncestralDiscreteScheduler
                "KDPM2DiscreteScheduler",  # 添加 KDPM2DiscreteScheduler
                "LCMScheduler",  # 添加 LCMScheduler
                "PNDMScheduler",  # 添加 PNDMScheduler
                "RePaintScheduler",  # 添加 RePaintScheduler
                "SASolverScheduler",  # 添加 SASolverScheduler
                "SchedulerMixin",  # 添加 SchedulerMixin
                "ScoreSdeVeScheduler",  # 添加 ScoreSdeVeScheduler
                "TCDScheduler",  # 添加 TCDScheduler
                "UnCLIPScheduler",  # 添加 UnCLIPScheduler
                "UniPCMultistepScheduler",  # 添加 UniPCMultistepScheduler
                "VQDiffusionScheduler",  # 添加 VQDiffusionScheduler
            ]
        )  # 结束扩展调度器的导入结构
        # 将训练工具（training_utils）中的 EMAModel 添加到导入结构
        _import_structure["training_utils"] = ["EMAModel"]
# 尝试检查 PyTorch 和 SciPy 是否可用
try:
    # 如果两者都不可用，则抛出可选依赖不可用异常
    if not (is_torch_available() and is_scipy_available()):
        raise OptionalDependencyNotAvailable()
# 捕获可选依赖不可用异常
except OptionalDependencyNotAvailable:
    # 从 utils 模块导入假对象，以避免缺失依赖的问题  # noqa F403
    from .utils import dummy_torch_and_scipy_objects  # noqa F403

    # 将假对象的名称添加到导入结构中，过滤掉以 "_" 开头的名称
    _import_structure["utils.dummy_torch_and_scipy_objects"] = [
        name for name in dir(dummy_torch_and_scipy_objects) if not name.startswith("_")
    ]
# 如果没有异常，则扩展调度器的导入结构
else:
    _import_structure["schedulers"].extend(["LMSDiscreteScheduler"])

# 尝试检查 PyTorch 和 torchsde 是否可用
try:
    # 如果两者都不可用，则抛出可选依赖不可用异常
    if not (is_torch_available() and is_torchsde_available()):
        raise OptionalDependencyNotAvailable()
# 捕获可选依赖不可用异常
except OptionalDependencyNotAvailable:
    # 从 utils 模块导入假对象，以避免缺失依赖的问题  # noqa F403
    from .utils import dummy_torch_and_torchsde_objects  # noqa F403

    # 将假对象的名称添加到导入结构中，过滤掉以 "_" 开头的名称
    _import_structure["utils.dummy_torch_and_torchsde_objects"] = [
        name for name in dir(dummy_torch_and_torchsde_objects) if not name.startswith("_")
    ]
# 如果没有异常，则扩展调度器的导入结构
else:
    _import_structure["schedulers"].extend(["CosineDPMSolverMultistepScheduler", "DPMSolverSDEScheduler"])

# 尝试检查 PyTorch 和 transformers 是否可用
try:
    # 如果两者都不可用，则抛出可选依赖不可用异常
    if not (is_torch_available() and is_transformers_available()):
        raise OptionalDependencyNotAvailable()
# 捕获可选依赖不可用异常
except OptionalDependencyNotAvailable:
    # 从 utils 模块导入假对象，以避免缺失依赖的问题  # noqa F403
    from .utils import dummy_torch_and_transformers_objects  # noqa F403

    # 将假对象的名称添加到导入结构中，过滤掉以 "_" 开头的名称
    _import_structure["utils.dummy_torch_and_transformers_objects"] = [
        name for name in dir(dummy_torch_and_transformers_objects) if not name.startswith("_")
    ]
# else 语句缺失

# 尝试检查 PyTorch、transformers 和 k-diffusion 是否可用
try:
    # 如果三者都不可用，则抛出可选依赖不可用异常
    if not (is_torch_available() and is_transformers_available() and is_k_diffusion_available()):
        raise OptionalDependencyNotAvailable()
# 捕获可选依赖不可用异常
except OptionalDependencyNotAvailable:
    # 从 utils 模块导入假对象，以避免缺失依赖的问题  # noqa F403
    from .utils import dummy_torch_and_transformers_and_k_diffusion_objects  # noqa F403

    # 将假对象的名称添加到导入结构中，过滤掉以 "_" 开头的名称
    _import_structure["utils.dummy_torch_and_transformers_and_k_diffusion_objects"] = [
        name for name in dir(dummy_torch_and_transformers_and_k_diffusion_objects) if not name.startswith("_")
    ]
# 如果没有异常，则扩展管道的导入结构
else:
    _import_structure["pipelines"].extend(["StableDiffusionKDiffusionPipeline", "StableDiffusionXLKDiffusionPipeline"])

# 尝试检查 PyTorch、transformers 和 sentencepiece 是否可用
try:
    # 如果三者都不可用，则抛出可选依赖不可用异常
    if not (is_torch_available() and is_transformers_available() and is_sentencepiece_available()):
        raise OptionalDependencyNotAvailable()
# 捕获可选依赖不可用异常
except OptionalDependencyNotAvailable:
    # 从 utils 模块导入假对象，以避免缺失依赖的问题  # noqa F403
    from .utils import dummy_torch_and_transformers_and_sentencepiece_objects  # noqa F403

    # 将假对象的名称添加到导入结构中，过滤掉以 "_" 开头的名称
    _import_structure["utils.dummy_torch_and_transformers_and_sentencepiece_objects"] = [
        name for name in dir(dummy_torch_and_transformers_and_sentencepiece_objects) if not name.startswith("_")
    ]
# 如果没有异常，则扩展管道的导入结构
else:
    _import_structure["pipelines"].extend(["KolorsImg2ImgPipeline", "KolorsPAGPipeline", "KolorsPipeline"])

# 尝试检查 PyTorch、transformers 和 onnx 是否可用
try:
    # 如果三者都不可用，则抛出可选依赖不可用异常
    if not (is_torch_available() and is_transformers_available() and is_onnx_available()):
        raise OptionalDependencyNotAvailable()
# 捕获可选依赖不可用异常
except OptionalDependencyNotAvailable:
    # 从 utils 模块导入假对象，以避免缺失依赖的问题  # noqa F403
    from .utils import dummy_torch_and_transformers_and_onnx_objects  # noqa F403
    # 将 "utils.dummy_torch_and_transformers_and_onnx_objects" 的导入结构存储为一个列表
        _import_structure["utils.dummy_torch_and_transformers_and_onnx_objects"] = [
            # 遍历 dummy_torch_and_transformers_and_onnx_objects 的属性名，排除以 "_" 开头的私有属性
            name for name in dir(dummy_torch_and_transformers_and_onnx_objects) if not name.startswith("_")
        ]
else:  # 如果前面的条件不满足
    # 将各个管道名称添加到 _import_structure["pipelines"] 列表中
    _import_structure["pipelines"].extend(
        [
            "OnnxStableDiffusionImg2ImgPipeline",  # ONNX 图像到图像管道
            "OnnxStableDiffusionInpaintPipeline",   # ONNX 图像修复管道
            "OnnxStableDiffusionInpaintPipelineLegacy",  # ONNX 旧版图像修复管道
            "OnnxStableDiffusionPipeline",  # ONNX 生成管道
            "OnnxStableDiffusionUpscalePipeline",  # ONNX 图像放大管道
            "StableDiffusionOnnxPipeline",  # 稳定扩散 ONNX 管道
        ]
    )

try:  # 尝试检查是否可用
    # 检查是否可用 PyTorch 和 librosa
    if not (is_torch_available() and is_librosa_available()): 
        raise OptionalDependencyNotAvailable()  # 如果不可用，则抛出异常
except OptionalDependencyNotAvailable:  # 捕获异常
    # 从工具模块导入虚拟对象以避免错误
    from .utils import dummy_torch_and_librosa_objects  # noqa F403

    # 获取虚拟对象中所有公共属性并添加到 _import_structure
    _import_structure["utils.dummy_torch_and_librosa_objects"] = [
        name for name in dir(dummy_torch_and_librosa_objects) if not name.startswith("_")  # 遍历所有属性
    ]

else:  # 如果没有抛出异常
    # 将音频扩散管道名称添加到 _import_structure["pipelines"] 列表中
    _import_structure["pipelines"].extend(["AudioDiffusionPipeline", "Mel"])

try:  # 尝试检查依赖项
    # 检查 transformers、PyTorch 和 note_seq 是否可用
    if not (is_transformers_available() and is_torch_available() and is_note_seq_available()):
        raise OptionalDependencyNotAvailable()  # 抛出异常
except OptionalDependencyNotAvailable:  # 捕获异常
    # 导入虚拟对象
    from .utils import dummy_transformers_and_torch_and_note_seq_objects  # noqa F403

    # 将虚拟对象的公共属性添加到 _import_structure
    _import_structure["utils.dummy_transformers_and_torch_and_note_seq_objects"] = [
        name for name in dir(dummy_transformers_and_torch_and_note_seq_objects) if not name.startswith("_")  # 遍历所有属性
    ]

else:  # 如果没有异常
    # 将频谱扩散管道名称添加到 _import_structure["pipelines"] 列表中
    _import_structure["pipelines"].extend(["SpectrogramDiffusionPipeline"])

try:  # 尝试检查 Flax 是否可用
    if not is_flax_available():  # 如果 Flax 不可用
        raise OptionalDependencyNotAvailable()  # 抛出异常
except OptionalDependencyNotAvailable:  # 捕获异常
    # 导入虚拟 Flax 对象
    from .utils import dummy_flax_objects  # noqa F403

    # 将虚拟对象的公共属性添加到 _import_structure
    _import_structure["utils.dummy_flax_objects"] = [
        name for name in dir(dummy_flax_objects) if not name.startswith("_")  # 遍历所有属性
    ]

else:  # 如果没有异常
    # 将多个 Flax 模型名称添加到 _import_structure
    _import_structure["models.controlnet_flax"] = ["FlaxControlNetModel"]  # 控制网络模型
    _import_structure["models.modeling_flax_utils"] = ["FlaxModelMixin"]  # Flax 模型混合类
    _import_structure["models.unets.unet_2d_condition_flax"] = ["FlaxUNet2DConditionModel"]  # Flax 2D 条件 U-Net 模型
    _import_structure["models.vae_flax"] = ["FlaxAutoencoderKL"]  # Flax 自编码器
    # 将 Flax 扩散管道名称添加到 _import_structure["pipelines"] 列表中
    _import_structure["pipelines"].extend(["FlaxDiffusionPipeline"])  
    # 将 Flax 调度器名称添加到 _import_structure["schedulers"] 列表中
    _import_structure["schedulers"].extend(
        [
            "FlaxDDIMScheduler",  # Flax DDIM 调度器
            "FlaxDDPMScheduler",  # Flax DDPMS 调度器
            "FlaxDPMSolverMultistepScheduler",  # Flax 多步 DPM 求解器调度器
            "FlaxEulerDiscreteScheduler",  # Flax Euler 离散调度器
            "FlaxKarrasVeScheduler",  # Flax Karras VE 调度器
            "FlaxLMSDiscreteScheduler",  # Flax LMS 离散调度器
            "FlaxPNDMScheduler",  # Flax PNDM 调度器
            "FlaxSchedulerMixin",  # Flax 调度器混合类
            "FlaxScoreSdeVeScheduler",  # Flax SDE VE 调度器
        ]
    )

try:  # 尝试检查 Flax 和 transformers 是否可用
    if not (is_flax_available() and is_transformers_available()):  # 如果其中一个不可用
        raise OptionalDependencyNotAvailable()  # 抛出异常
except OptionalDependencyNotAvailable:  # 捕获异常
    # 导入虚拟对象
    from .utils import dummy_flax_and_transformers_objects  # noqa F403

    # 将虚拟对象的公共属性添加到 _import_structure
    _import_structure["utils.dummy_flax_and_transformers_objects"] = [
        name for name in dir(dummy_flax_and_transformers_objects) if not name.startswith("_")  # 遍历所有属性
    ]

else:  # 如果没有异常
    # 扩展 _import_structure 字典中 "pipelines" 键对应的列表
        _import_structure["pipelines"].extend(
            # 添加多个管道类的名称到列表中
            [
                "FlaxStableDiffusionControlNetPipeline",  # 控制网络管道类
                "FlaxStableDiffusionImg2ImgPipeline",     # 图像到图像转换管道类
                "FlaxStableDiffusionInpaintPipeline",      # 图像修复管道类
                "FlaxStableDiffusionPipeline",             # 标准稳定扩散管道类
                "FlaxStableDiffusionXLPipeline",           # 扩展的稳定扩散管道类
            ]
        )
try:
    # 检查 note_seq 是否可用
    if not (is_note_seq_available()):
        # 如果不可用，抛出可选依赖项不可用异常
        raise OptionalDependencyNotAvailable()
# 捕获可选依赖项不可用异常
except OptionalDependencyNotAvailable:
    # 从 utils 模块导入虚拟的 note_seq 对象，避免未导入的错误
    from .utils import dummy_note_seq_objects  # noqa F403

    # 将虚拟对象的名称添加到导入结构中（只包含非私有属性）
    _import_structure["utils.dummy_note_seq_objects"] = [
        name for name in dir(dummy_note_seq_objects) if not name.startswith("_")
    ]

# 如果没有异常，则继续执行以下代码
else:
    # 向导入结构中添加 MidiProcessor
    _import_structure["pipelines"].extend(["MidiProcessor"])

# 如果在类型检查或慢速导入模式下
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    # 从配置工具模块导入 ConfigMixin
    from .configuration_utils import ConfigMixin

    try:
        # 检查 onnx 是否可用
        if not is_onnx_available():
            # 如果不可用，抛出可选依赖项不可用异常
            raise OptionalDependencyNotAvailable()
    # 捕获可选依赖项不可用异常
    except OptionalDependencyNotAvailable:
        # 从 utils 模块导入虚拟的 onnx 对象，避免未导入的错误
        from .utils.dummy_onnx_objects import *  # noqa F403
    else:
        # 从管道模块导入 OnnxRuntimeModel
        from .pipelines import OnnxRuntimeModel

    try:
        # 检查 torch 是否可用
        if not is_torch_available():
            # 如果不可用，抛出可选依赖项不可用异常
            raise OptionalDependencyNotAvailable()
    # 捕获可选依赖项不可用异常
    except OptionalDependencyNotAvailable:
        # 从 utils 模块导入虚拟的 pytorch 对象，避免未导入的错误
        from .utils.dummy_pt_objects import *  # noqa F403
        
    try:
        # 检查 torch 和 scipy 是否都可用
        if not (is_torch_available() and is_scipy_available()):
            # 如果不满足条件，抛出可选依赖项不可用异常
            raise OptionalDependencyNotAvailable()
    # 捕获可选依赖项不可用异常
    except OptionalDependencyNotAvailable:
        # 从 utils 模块导入虚拟的 torch 和 scipy 对象，避免未导入的错误
        from .utils.dummy_torch_and_scipy_objects import *  # noqa F403
    else:
        # 从调度器模块导入 LMSDiscreteScheduler
        from .schedulers import LMSDiscreteScheduler

    try:
        # 检查 torch 和 torchsde 是否都可用
        if not (is_torch_available() and is_torchsde_available()):
            # 如果不满足条件，抛出可选依赖项不可用异常
            raise OptionalDependencyNotAvailable()
    # 捕获可选依赖项不可用异常
    except OptionalDependencyNotAvailable:
        # 从 utils 模块导入虚拟的 torch 和 torchsde 对象，避免未导入的错误
        from .utils.dummy_torch_and_torchsde_objects import *  # noqa F403
    else:
        # 从调度器模块导入 CosineDPMSolverMultistepScheduler 和 DPMSolverSDEScheduler
        from .schedulers import CosineDPMSolverMultistepScheduler, DPMSolverSDEScheduler

    try:
        # 检查 torch 和 transformers 是否都可用
        if not (is_torch_available() and is_transformers_available()):
            # 如果不满足条件，抛出可选依赖项不可用异常
            raise OptionalDependencyNotAvailable()
    # 捕获可选依赖项不可用异常
    except OptionalDependencyNotAvailable:
        # 从 utils 模块导入虚拟的 torch 和 transformers 对象，避免未导入的错误
        from .utils.dummy_torch_and_transformers_objects import *  # noqa F403
        
    try:
        # 检查 torch、transformers 和 k_diffusion 是否都可用
        if not (is_torch_available() and is_transformers_available() and is_k_diffusion_available()):
            # 如果不满足条件，抛出可选依赖项不可用异常
            raise OptionalDependencyNotAvailable()
    # 捕获可选依赖项不可用异常
    except OptionalDependencyNotAvailable:
        # 从 utils 模块导入虚拟的 torch、transformers 和 k_diffusion 对象，避免未导入的错误
        from .utils.dummy_torch_and_transformers_and_k_diffusion_objects import *  # noqa F403
    else:
        # 从管道模块导入 StableDiffusionKDiffusionPipeline 和 StableDiffusionXLKDiffusionPipeline
        from .pipelines import StableDiffusionKDiffusionPipeline, StableDiffusionXLKDiffusionPipeline

    try:
        # 检查 torch、transformers 和 sentencepiece 是否都可用
        if not (is_torch_available() and is_transformers_available() and is_sentencepiece_available()):
            # 如果不满足条件，抛出可选依赖项不可用异常
            raise OptionalDependencyNotAvailable()
    # 捕获可选依赖项不可用异常
    except OptionalDependencyNotAvailable:
        # 从 utils 模块导入虚拟的 torch、transformers 和 sentencepiece 对象，避免未导入的错误
        from .utils.dummy_torch_and_transformers_and_sentencepiece_objects import *  # noqa F403
    else:
        # 从管道模块导入 KolorsImg2ImgPipeline、KolorsPAGPipeline 和 KolorsPipeline
        from .pipelines import KolorsImg2ImgPipeline, KolorsPAGPipeline, KolorsPipeline
        
    try:
        # 检查 torch、transformers 和 onnx 是否都可用
        if not (is_torch_available() and is_transformers_available() and is_onnx_available()):
            # 如果不满足条件，抛出可选依赖项不可用异常
            raise OptionalDependencyNotAvailable()
    # 捕获可选依赖项不可用异常
    except OptionalDependencyNotAvailable:
        # 从 utils 模块导入虚拟的 torch、transformers 和 onnx 对象，避免未导入的错误
        from .utils.dummy_torch_and_transformers_and_onnx_objects import *  # noqa F403
    else:  # 如果没有满足上面的条件，则执行以下导入
        from .pipelines import (  # 从当前包的 pipelines 模块中导入以下类
            OnnxStableDiffusionImg2ImgPipeline,  # 导入图像到图像转换的 ONNX 管道
            OnnxStableDiffusionInpaintPipeline,  # 导入图像修复的 ONNX 管道
            OnnxStableDiffusionInpaintPipelineLegacy,  # 导入旧版图像修复的 ONNX 管道
            OnnxStableDiffusionPipeline,  # 导入基础的 ONNX 扩散管道
            OnnxStableDiffusionUpscalePipeline,  # 导入图像放大的 ONNX 管道
            StableDiffusionOnnxPipeline,  # 导入稳定扩散的 ONNX 管道
        )

    try:  # 尝试执行以下代码
        if not (is_torch_available() and is_librosa_available()):  # 检查 PyTorch 和 librosa 是否可用
            raise OptionalDependencyNotAvailable()  # 如果其中一个不可用，则抛出异常
    except OptionalDependencyNotAvailable:  # 捕获可选依赖不可用的异常
        from .utils.dummy_torch_and_librosa_objects import *  # noqa F403  # 导入虚拟的 torch 和 librosa 对象以避免错误
    else:  # 如果没有抛出异常，执行以下导入
        from .pipelines import AudioDiffusionPipeline, Mel  # 从 pipelines 导入音频扩散管道和 Mel 类

    try:  # 尝试执行以下代码
        if not (is_transformers_available() and is_torch_available() and is_note_seq_available()):  # 检查 transformers, PyTorch 和 note_seq 是否可用
            raise OptionalDependencyNotAvailable()  # 如果其中一个不可用，则抛出异常
    except OptionalDependencyNotAvailable:  # 捕获可选依赖不可用的异常
        from .utils.dummy_transformers_and_torch_and_note_seq_objects import *  # noqa F403  # 导入虚拟的 transformers, torch 和 note_seq 对象以避免错误
    else:  # 如果没有抛出异常，执行以下导入
        from .pipelines import SpectrogramDiffusionPipeline  # 从 pipelines 导入声谱扩散管道

    try:  # 尝试执行以下代码
        if not is_flax_available():  # 检查 Flax 是否可用
            raise OptionalDependencyNotAvailable()  # 如果不可用，则抛出异常
    except OptionalDependencyNotAvailable:  # 捕获可选依赖不可用的异常
        from .utils.dummy_flax_objects import *  # noqa F403  # 导入虚拟的 Flax 对象以避免错误
    else:  # 如果没有抛出异常，执行以下导入
        from .models.controlnet_flax import FlaxControlNetModel  # 从模型导入 Flax 控制网络模型
        from .models.modeling_flax_utils import FlaxModelMixin  # 导入 Flax 模型混合工具
        from .models.unets.unet_2d_condition_flax import FlaxUNet2DConditionModel  # 导入 2D 条件 UNet 模型
        from .models.vae_flax import FlaxAutoencoderKL  # 导入 Flax 的自编码器模型
        from .pipelines import FlaxDiffusionPipeline  # 从 pipelines 导入 Flax 扩散管道
        from .schedulers import (  # 从调度器模块导入以下类
            FlaxDDIMScheduler,  # 导入 Flax 的 DDIM 调度器
            FlaxDDPMScheduler,  # 导入 Flax 的 DDPMS 调度器
            FlaxDPMSolverMultistepScheduler,  # 导入 Flax 的多步 DPM 求解器调度器
            FlaxEulerDiscreteScheduler,  # 导入 Flax 的 Euler 离散调度器
            FlaxKarrasVeScheduler,  # 导入 Flax 的 Karras VE 调度器
            FlaxLMSDiscreteScheduler,  # 导入 Flax 的 LMS 离散调度器
            FlaxPNDMScheduler,  # 导入 Flax 的 PNDM 调度器
            FlaxSchedulerMixin,  # 导入 Flax 调度器混合工具
            FlaxScoreSdeVeScheduler,  # 导入 Flax 的 SDE VE 调度器
        )

    try:  # 尝试执行以下代码
        if not (is_flax_available() and is_transformers_available()):  # 检查 Flax 和 transformers 是否可用
            raise OptionalDependencyNotAvailable()  # 如果其中一个不可用，则抛出异常
    except OptionalDependencyNotAvailable:  # 捕获可选依赖不可用的异常
        from .utils.dummy_flax_and_transformers_objects import *  # noqa F403  # 导入虚拟的 Flax 和 transformers 对象以避免错误
    else:  # 如果没有抛出异常，执行以下导入
        from .pipelines import (  # 从 pipelines 导入以下类
            FlaxStableDiffusionControlNetPipeline,  # 导入 Flax 稳定扩散控制网络管道
            FlaxStableDiffusionImg2ImgPipeline,  # 导入 Flax 图像到图像的稳定扩散管道
            FlaxStableDiffusionInpaintPipeline,  # 导入 Flax 图像修复的稳定扩散管道
            FlaxStableDiffusionPipeline,  # 导入 Flax 稳定扩散管道
            FlaxStableDiffusionXLPipeline,  # 导入 Flax 稳定扩散 XL 管道
        )

    try:  # 尝试执行以下代码
        if not (is_note_seq_available()):  # 检查 note_seq 是否可用
            raise OptionalDependencyNotAvailable()  # 如果不可用，则抛出异常
    except OptionalDependencyNotAvailable:  # 捕获可选依赖不可用的异常
        from .utils.dummy_note_seq_objects import *  # noqa F403  # 导入虚拟的 note_seq 对象以避免错误
    else:  # 如果没有抛出异常，执行以下导入
        from .pipelines import MidiProcessor  # 从 pipelines 导入 MIDI 处理器
# 如果不是主模块执行，则进行模块的延迟加载
else:
    # 导入系统模块，以便对模块进行操作
    import sys

    # 将当前模块的名称映射到一个懒加载模块实例
    sys.modules[__name__] = _LazyModule(
        # 当前模块名称
        __name__,
        # 当前模块的文件路径
        globals()["__file__"],
        # 导入结构的定义
        _import_structure,
        # 模块的规格
        module_spec=__spec__,
        # 额外对象，包括版本信息
        extra_objects={"__version__": __version__},
    )
```