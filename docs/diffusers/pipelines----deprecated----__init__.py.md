# `.\diffusers\pipelines\deprecated\__init__.py`

```py
# 从 typing 模块导入 TYPE_CHECKING，用于类型检查
from typing import TYPE_CHECKING

# 从上层模块的 utils 中导入多个工具函数和常量
from ...utils import (
    DIFFUSERS_SLOW_IMPORT,  # 慢速导入标识
    OptionalDependencyNotAvailable,  # 可选依赖未找到异常
    _LazyModule,  # 懒加载模块
    get_objects_from_module,  # 从模块获取对象的工具函数
    is_librosa_available,  # 检查 librosa 库是否可用
    is_note_seq_available,  # 检查 note_seq 库是否可用
    is_torch_available,  # 检查 torch 库是否可用
    is_transformers_available,  # 检查 transformers 库是否可用
)

# 用于存储虚拟对象的字典
_dummy_objects = {}
# 用于存储导入结构的字典
_import_structure = {}

try:
    # 检查 torch 是否可用，若不可用则引发异常
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
# 捕获可选依赖未找到的异常
except OptionalDependencyNotAvailable:
    # 从 utils 中导入虚拟 PyTorch 对象
    from ...utils import dummy_pt_objects

    # 更新虚拟对象字典，以包含从虚拟对象模块获取的对象
    _dummy_objects.update(get_objects_from_module(dummy_pt_objects))
# 如果 torch 可用，则定义导入结构
else:
    _import_structure["latent_diffusion_uncond"] = ["LDMPipeline"]  # 添加 LDM 管道
    _import_structure["pndm"] = ["PNDMPipeline"]  # 添加 PNDM 管道
    _import_structure["repaint"] = ["RePaintPipeline"]  # 添加 RePaint 管道
    _import_structure["score_sde_ve"] = ["ScoreSdeVePipeline"]  # 添加 ScoreSDE VE 管道
    _import_structure["stochastic_karras_ve"] = ["KarrasVePipeline"]  # 添加 Karras VE 管道

try:
    # 检查 transformers 和 torch 是否都可用，若不可用则引发异常
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
# 捕获可选依赖未找到的异常
except OptionalDependencyNotAvailable:
    # 从 utils 中导入虚拟 torch 和 transformers 对象
    from ...utils import dummy_torch_and_transformers_objects

    # 更新虚拟对象字典，以包含从虚拟对象模块获取的对象
    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
# 如果两者都可用，则定义导入结构
else:
    _import_structure["alt_diffusion"] = [  # 添加替代扩散相关的管道
        "AltDiffusionImg2ImgPipeline",
        "AltDiffusionPipeline",
        "AltDiffusionPipelineOutput",
    ]
    _import_structure["versatile_diffusion"] = [  # 添加多功能扩散相关的管道
        "VersatileDiffusionDualGuidedPipeline",
        "VersatileDiffusionImageVariationPipeline",
        "VersatileDiffusionPipeline",
        "VersatileDiffusionTextToImagePipeline",
    ]
    _import_structure["vq_diffusion"] = ["VQDiffusionPipeline"]  # 添加 VQ 扩散管道
    _import_structure["stable_diffusion_variants"] = [  # 添加稳定扩散变体相关的管道
        "CycleDiffusionPipeline",
        "StableDiffusionInpaintPipelineLegacy",
        "StableDiffusionPix2PixZeroPipeline",
        "StableDiffusionParadigmsPipeline",
        "StableDiffusionModelEditingPipeline",
    ]

try:
    # 检查 torch 和 librosa 是否都可用，若不可用则引发异常
    if not (is_torch_available() and is_librosa_available()):
        raise OptionalDependencyNotAvailable()
# 捕获可选依赖未找到的异常
except OptionalDependencyNotAvailable:
    # 从 utils 中导入虚拟 torch 和 librosa 对象
    from ...utils import dummy_torch_and_librosa_objects  # noqa F403

    # 更新虚拟对象字典，以包含从虚拟对象模块获取的对象
    _dummy_objects.update(get_objects_from_module(dummy_torch_and_librosa_objects))

else:
    _import_structure["audio_diffusion"] = ["AudioDiffusionPipeline", "Mel"]  # 添加音频扩散相关的管道

try:
    # 检查 transformers、torch 和 note_seq 是否都可用，若不可用则引发异常
    if not (is_transformers_available() and is_torch_available() and is_note_seq_available()):
        raise OptionalDependencyNotAvailable()
# 捕获可选依赖未找到的异常
except OptionalDependencyNotAvailable:
    # 从 utils 中导入虚拟 transformers、torch 和 note_seq 对象
    from ...utils import dummy_transformers_and_torch_and_note_seq_objects  # noqa F403

    # 更新虚拟对象字典，以包含从虚拟对象模块获取的对象
    _dummy_objects.update(get_objects_from_module(dummy_transformers_and_torch_and_note_seq_objects))

else:
    _import_structure["spectrogram_diffusion"] = ["MidiProcessor", "SpectrogramDiffusionPipeline"]  # 添加频谱扩散相关的管道

# 如果正在进行类型检查或设置为慢速导入，则执行检查
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        # 检查 torch 是否可用，若不可用则引发异常
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    # 捕获可选依赖项未找到的异常
    except OptionalDependencyNotAvailable:
        # 从虚拟模块导入占位符对象
        from ...utils.dummy_pt_objects import *

    # 如果没有抛出异常，导入以下管道类
    else:
        from .latent_diffusion_uncond import LDMPipeline
        from .pndm import PNDMPipeline
        from .repaint import RePaintPipeline
        from .score_sde_ve import ScoreSdeVePipeline
        from .stochastic_karras_ve import KarrasVePipeline

    # 尝试检查是否可用所需的库
    try:
        if not (is_transformers_available() and is_torch_available()):
            # 如果库不可用，抛出异常
            raise OptionalDependencyNotAvailable()
    # 捕获可选依赖项未找到的异常
    except OptionalDependencyNotAvailable:
        # 从虚拟模块导入占位符对象
        from ...utils.dummy_torch_and_transformers_objects import *

    # 如果没有抛出异常，导入以下管道和类
    else:
        from .alt_diffusion import AltDiffusionImg2ImgPipeline, AltDiffusionPipeline, AltDiffusionPipelineOutput
        from .audio_diffusion import AudioDiffusionPipeline, Mel
        from .spectrogram_diffusion import SpectrogramDiffusionPipeline
        from .stable_diffusion_variants import (
            CycleDiffusionPipeline,
            StableDiffusionInpaintPipelineLegacy,
            StableDiffusionModelEditingPipeline,
            StableDiffusionParadigmsPipeline,
            StableDiffusionPix2PixZeroPipeline,
        )
        from .stochastic_karras_ve import KarrasVePipeline
        from .versatile_diffusion import (
            VersatileDiffusionDualGuidedPipeline,
            VersatileDiffusionImageVariationPipeline,
            VersatileDiffusionPipeline,
            VersatileDiffusionTextToImagePipeline,
        )
        from .vq_diffusion import VQDiffusionPipeline

    # 尝试检查是否可用所需的音频库
    try:
        if not (is_torch_available() and is_librosa_available()):
            # 如果库不可用，抛出异常
            raise OptionalDependencyNotAvailable()
    # 捕获可选依赖项未找到的异常
    except OptionalDependencyNotAvailable:
        # 从虚拟模块导入占位符对象
        from ...utils.dummy_torch_and_librosa_objects import *
    # 如果没有抛出异常，导入音频相关类
    else:
        from .audio_diffusion import AudioDiffusionPipeline, Mel

    # 尝试检查是否可用所有必要的库
    try:
        if not (is_transformers_available() and is_torch_available() and is_note_seq_available()):
            # 如果库不可用，抛出异常
            raise OptionalDependencyNotAvailable()
    # 捕获可选依赖项未找到的异常
    except OptionalDependencyNotAvailable:
        # 从虚拟模块导入占位符对象
        from ...utils.dummy_transformers_and_torch_and_note_seq_objects import *  # noqa F403
    # 如果没有抛出异常，导入音频频谱相关类
    else:
        from .spectrogram_diffusion import (
            MidiProcessor,
            SpectrogramDiffusionPipeline,
        )
else:
    # 导入 sys 模块以便访问 Python 的模块系统
    import sys

    # 将当前模块名（__name__）的 sys.modules 条目替换为一个懒加载模块对象
    sys.modules[__name__] = _LazyModule(
        __name__,  # 当前模块的名称
        globals()["__file__"],  # 当前模块的文件路径
        _import_structure,  # 用于指定模块的导入结构
        module_spec=__spec__,  # 模块的规格，提供模块的元数据
    )
    # 遍历 _dummy_objects 字典，将每个对象的名称和值设置到当前模块中
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)  # 为当前模块设置属性
```