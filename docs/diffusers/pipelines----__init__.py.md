# `.\diffusers\pipelines\__init__.py`

```py
# 导入类型检查的模块
from typing import TYPE_CHECKING

# 从父级目录的 utils 模块中导入多个对象和函数
from ..utils import (
    DIFFUSERS_SLOW_IMPORT,  # 导入一个慢加载的功能
    OptionalDependencyNotAvailable,  # 导入可选依赖不可用的异常
    _LazyModule,  # 导入懒加载模块的工具
    get_objects_from_module,  # 导入从模块中获取对象的函数
    is_flax_available,  # 导入检查 Flax 库是否可用的函数
    is_k_diffusion_available,  # 导入检查 K-Diffusion 库是否可用的函数
    is_librosa_available,  # 导入检查 Librosa 库是否可用的函数
    is_note_seq_available,  # 导入检查 NoteSeq 库是否可用的函数
    is_onnx_available,  # 导入检查 ONNX 库是否可用的函数
    is_sentencepiece_available,  # 导入检查 SentencePiece 库是否可用的函数
    is_torch_available,  # 导入检查 PyTorch 库是否可用的函数
    is_torch_npu_available,  # 导入检查 NPU 版 PyTorch 是否可用的函数
    is_transformers_available,  # 导入检查 Transformers 库是否可用的函数
)

# 初始化一个空字典以存储假对象
_dummy_objects = {}
# 定义一个字典以组织导入的模块结构
_import_structure = {
    "controlnet": [],  # 控制网模块
    "controlnet_hunyuandit": [],  # 控制网相关模块
    "controlnet_sd3": [],  # 控制网 SD3 模块
    "controlnet_xs": [],  # 控制网 XS 模块
    "deprecated": [],  # 存放弃用模块
    "latent_diffusion": [],  # 潜在扩散模块
    "ledits_pp": [],  # LEDITS PP 模块
    "marigold": [],  # 万寿菊模块
    "pag": [],  # PAG 模块
    "stable_diffusion": [],  # 稳定扩散模块
    "stable_diffusion_xl": [],  # 稳定扩散 XL 模块
}

try:
    # 检查 PyTorch 是否可用
    if not is_torch_available():
        # 如果不可用，抛出可选依赖不可用的异常
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 如果捕获到异常，从 utils 模块导入假对象（PyTorch 相关）
    from ..utils import dummy_pt_objects  # noqa F403

    # 将获取的假对象更新到字典中
    _dummy_objects.update(get_objects_from_module(dummy_pt_objects))
else:
    # 如果 PyTorch 可用，更新导入结构，添加自动管道类
    _import_structure["auto_pipeline"] = [
        "AutoPipelineForImage2Image",  # 图像到图像的自动管道
        "AutoPipelineForInpainting",  # 图像修复的自动管道
        "AutoPipelineForText2Image",  # 文本到图像的自动管道
    ]
    # 添加一致性模型管道
    _import_structure["consistency_models"] = ["ConsistencyModelPipeline"]
    # 添加舞蹈扩散管道
    _import_structure["dance_diffusion"] = ["DanceDiffusionPipeline"]
    # 添加 DDIM 管道
    _import_structure["ddim"] = ["DDIMPipeline"]
    # 添加 DDPM 管道
    _import_structure["ddpm"] = ["DDPMPipeline"]
    # 添加 DiT 管道
    _import_structure["dit"] = ["DiTPipeline"]
    # 扩展潜在扩散模块，添加超分辨率管道
    _import_structure["latent_diffusion"].extend(["LDMSuperResolutionPipeline"])
    # 添加管道工具的输出类型
    _import_structure["pipeline_utils"] = [
        "AudioPipelineOutput",  # 音频管道输出
        "DiffusionPipeline",  # 扩散管道
        "StableDiffusionMixin",  # 稳定扩散混合类
        "ImagePipelineOutput",  # 图像管道输出
    ]
    # 扩展弃用模块，添加弃用的管道
    _import_structure["deprecated"].extend(
        [
            "PNDMPipeline",  # PNDM 管道
            "LDMPipeline",  # LDM 管道
            "RePaintPipeline",  # 重绘管道
            "ScoreSdeVePipeline",  # Score SDE VE 管道
            "KarrasVePipeline",  # Karras VE 管道
        ]
    )
try:
    # 检查 PyTorch 和 Librosa 是否都可用
    if not (is_torch_available() and is_librosa_available()):
        # 如果其中一个不可用，抛出可选依赖不可用的异常
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 捕获异常，从 utils 模块导入假对象（PyTorch 和 Librosa 相关）
    from ..utils import dummy_torch_and_librosa_objects  # noqa F403

    # 更新假对象字典
    _dummy_objects.update(get_objects_from_module(dummy_torch_and_librosa_objects))
else:
    # 如果两个库都可用，扩展弃用模块，添加音频扩散管道和 Mel 类
    _import_structure["deprecated"].extend(["AudioDiffusionPipeline", "Mel"])

try:
    # 检查 Transformers、PyTorch 和 NoteSeq 是否都可用
    if not (is_transformers_available() and is_torch_available() and is_note_seq_available()):
        # 如果其中一个不可用，抛出可选依赖不可用的异常
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 捕获异常，从 utils 模块导入假对象（Transformers、PyTorch 和 NoteSeq 相关）
    from ..utils import dummy_transformers_and_torch_and_note_seq_objects  # noqa F403

    # 更新假对象字典
    _dummy_objects.update(get_objects_from_module(dummy_transformers_and_torch_and_note_seq_objects))
else:
    # 如果三个库都可用，扩展弃用模块，添加 MIDI 处理器和谱图扩散管道
    _import_structure["deprecated"].extend(
        [
            "MidiProcessor",  # MIDI 处理器
            "SpectrogramDiffusionPipeline",  # 谱图扩散管道
        ]
    )

try:
    # 检查 PyTorch 和 Transformers 库是否可用
        if not (is_torch_available() and is_transformers_available()):
            # 如果任一库不可用，抛出异常表示可选依赖不可用
            raise OptionalDependencyNotAvailable()
# 捕获未满足可选依赖项的异常
except OptionalDependencyNotAvailable:
    # 从上层模块导入虚拟的 Torch 和 Transformers 对象
    from ..utils import dummy_torch_and_transformers_objects  # noqa F403

    # 更新虚拟对象的字典，以获取导入的虚拟对象
    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    # 将过时的导入结构中添加一组管道名称
    _import_structure["deprecated"].extend(
        [
            "VQDiffusionPipeline",
            "AltDiffusionPipeline",
            "AltDiffusionImg2ImgPipeline",
            "CycleDiffusionPipeline",
            "StableDiffusionInpaintPipelineLegacy",
            "StableDiffusionPix2PixZeroPipeline",
            "StableDiffusionParadigmsPipeline",
            "StableDiffusionModelEditingPipeline",
            "VersatileDiffusionDualGuidedPipeline",
            "VersatileDiffusionImageVariationPipeline",
            "VersatileDiffusionPipeline",
            "VersatileDiffusionTextToImagePipeline",
        ]
    )
    # 为“amused”添加相关管道名称
    _import_structure["amused"] = ["AmusedImg2ImgPipeline", "AmusedInpaintPipeline", "AmusedPipeline"]
    # 为“animatediff”添加相关管道名称
    _import_structure["animatediff"] = [
        "AnimateDiffPipeline",
        "AnimateDiffControlNetPipeline",
        "AnimateDiffSDXLPipeline",
        "AnimateDiffSparseControlNetPipeline",
        "AnimateDiffVideoToVideoPipeline",
    ]
    # 为“flux”添加相关管道名称
    _import_structure["flux"] = ["FluxPipeline"]
    # 为“audioldm”添加相关管道名称
    _import_structure["audioldm"] = ["AudioLDMPipeline"]
    # 为“audioldm2”添加相关管道名称
    _import_structure["audioldm2"] = [
        "AudioLDM2Pipeline",
        "AudioLDM2ProjectionModel",
        "AudioLDM2UNet2DConditionModel",
    ]
    # 为“blip_diffusion”添加相关管道名称
    _import_structure["blip_diffusion"] = ["BlipDiffusionPipeline"]
    # 为“cogvideo”添加相关管道名称
    _import_structure["cogvideo"] = [
        "CogVideoXPipeline",
        "CogVideoXImageToVideoPipeline",
        "CogVideoXVideoToVideoPipeline",
    ]
    # 为“controlnet”扩展相关管道名称
    _import_structure["controlnet"].extend(
        [
            "BlipDiffusionControlNetPipeline",
            "StableDiffusionControlNetImg2ImgPipeline",
            "StableDiffusionControlNetInpaintPipeline",
            "StableDiffusionControlNetPipeline",
            "StableDiffusionXLControlNetImg2ImgPipeline",
            "StableDiffusionXLControlNetInpaintPipeline",
            "StableDiffusionXLControlNetPipeline",
        ]
    )
    # 为“pag”扩展相关管道名称
    _import_structure["pag"].extend(
        [
            "AnimateDiffPAGPipeline",
            "KolorsPAGPipeline",
            "HunyuanDiTPAGPipeline",
            "StableDiffusion3PAGPipeline",
            "StableDiffusionPAGPipeline",
            "StableDiffusionControlNetPAGPipeline",
            "StableDiffusionXLPAGPipeline",
            "StableDiffusionXLPAGInpaintPipeline",
            "StableDiffusionXLControlNetPAGPipeline",
            "StableDiffusionXLPAGImg2ImgPipeline",
            "PixArtSigmaPAGPipeline",
        ]
    )
    # 为“controlnet_xs”扩展相关管道名称
    _import_structure["controlnet_xs"].extend(
        [
            "StableDiffusionControlNetXSPipeline",
            "StableDiffusionXLControlNetXSPipeline",
        ]
    )
    # 为“controlnet_hunyuandit”扩展相关管道名称
    _import_structure["controlnet_hunyuandit"].extend(
        [
            "HunyuanDiTControlNetPipeline",
        ]
    )
    # 将 "StableDiffusion3ControlNetPipeline" 添加到 "controlnet_sd3" 的导入结构中
    _import_structure["controlnet_sd3"].extend(
        [
            "StableDiffusion3ControlNetPipeline",
        ]
    )
    # 定义 "deepfloyd_if" 的导入结构，包含多个管道
    _import_structure["deepfloyd_if"] = [
        "IFImg2ImgPipeline",
        "IFImg2ImgSuperResolutionPipeline",
        "IFInpaintingPipeline",
        "IFInpaintingSuperResolutionPipeline",
        "IFPipeline",
        "IFSuperResolutionPipeline",
    ]
    # 设置 "hunyuandit" 的导入结构，仅包含一个管道
    _import_structure["hunyuandit"] = ["HunyuanDiTPipeline"]
    # 定义 "kandinsky" 的导入结构，包含多个相关管道
    _import_structure["kandinsky"] = [
        "KandinskyCombinedPipeline",
        "KandinskyImg2ImgCombinedPipeline",
        "KandinskyImg2ImgPipeline",
        "KandinskyInpaintCombinedPipeline",
        "KandinskyInpaintPipeline",
        "KandinskyPipeline",
        "KandinskyPriorPipeline",
    ]
    # 定义 "kandinsky2_2" 的导入结构，包含多个管道
    _import_structure["kandinsky2_2"] = [
        "KandinskyV22CombinedPipeline",
        "KandinskyV22ControlnetImg2ImgPipeline",
        "KandinskyV22ControlnetPipeline",
        "KandinskyV22Img2ImgCombinedPipeline",
        "KandinskyV22Img2ImgPipeline",
        "KandinskyV22InpaintCombinedPipeline",
        "KandinskyV22InpaintPipeline",
        "KandinskyV22Pipeline",
        "KandinskyV22PriorEmb2EmbPipeline",
        "KandinskyV22PriorPipeline",
    ]
    # 定义 "kandinsky3" 的导入结构，包含两个管道
    _import_structure["kandinsky3"] = [
        "Kandinsky3Img2ImgPipeline",
        "Kandinsky3Pipeline",
    ]
    # 定义 "latent_consistency_models" 的导入结构，包含两个管道
    _import_structure["latent_consistency_models"] = [
        "LatentConsistencyModelImg2ImgPipeline",
        "LatentConsistencyModelPipeline",
    ]
    # 将 "LDMTextToImagePipeline" 添加到 "latent_diffusion" 的导入结构中
    _import_structure["latent_diffusion"].extend(["LDMTextToImagePipeline"])
    # 将稳定扩散相关的管道添加到 "ledits_pp" 的导入结构中
    _import_structure["ledits_pp"].extend(
        [
            "LEditsPPPipelineStableDiffusion",
            "LEditsPPPipelineStableDiffusionXL",
        ]
    )
    # 设置 "latte" 的导入结构，仅包含一个管道
    _import_structure["latte"] = ["LattePipeline"]
    # 设置 "lumina" 的导入结构，仅包含一个管道
    _import_structure["lumina"] = ["LuminaText2ImgPipeline"]
    # 将 "MarigoldDepthPipeline" 和 "MarigoldNormalsPipeline" 添加到 "marigold" 的导入结构中
    _import_structure["marigold"].extend(
        [
            "MarigoldDepthPipeline",
            "MarigoldNormalsPipeline",
        ]
    )
    # 设置 "musicldm" 的导入结构，仅包含一个管道
    _import_structure["musicldm"] = ["MusicLDMPipeline"]
    # 设置 "paint_by_example" 的导入结构，仅包含一个管道
    _import_structure["paint_by_example"] = ["PaintByExamplePipeline"]
    # 设置 "pia" 的导入结构，仅包含一个管道
    _import_structure["pia"] = ["PIAPipeline"]
    # 设置 "pixart_alpha" 的导入结构，包含两个管道
    _import_structure["pixart_alpha"] = ["PixArtAlphaPipeline", "PixArtSigmaPipeline"]
    # 设置 "semantic_stable_diffusion" 的导入结构，仅包含一个管道
    _import_structure["semantic_stable_diffusion"] = ["SemanticStableDiffusionPipeline"]
    # 设置 "shap_e" 的导入结构，包含两个管道
    _import_structure["shap_e"] = ["ShapEImg2ImgPipeline", "ShapEPipeline"]
    # 定义 "stable_audio" 的导入结构，包含两个管道
    _import_structure["stable_audio"] = [
        "StableAudioProjectionModel",
        "StableAudioPipeline",
    ]
    # 定义 "stable_cascade" 的导入结构，包含多个管道
    _import_structure["stable_cascade"] = [
        "StableCascadeCombinedPipeline",
        "StableCascadeDecoderPipeline",
        "StableCascadePriorPipeline",
    ]
    # 向 stable_diffusion 的导入结构中添加多个相关的管道名称
    _import_structure["stable_diffusion"].extend(
        [
            # 添加 CLIP 图像投影管道
            "CLIPImageProjection",
            # 添加稳定扩散深度到图像管道
            "StableDiffusionDepth2ImgPipeline",
            # 添加稳定扩散图像变体管道
            "StableDiffusionImageVariationPipeline",
            # 添加稳定扩散图像到图像管道
            "StableDiffusionImg2ImgPipeline",
            # 添加稳定扩散图像修复管道
            "StableDiffusionInpaintPipeline",
            # 添加稳定扩散指令图像到图像管道
            "StableDiffusionInstructPix2PixPipeline",
            # 添加稳定扩散潜在上采样管道
            "StableDiffusionLatentUpscalePipeline",
            # 添加稳定扩散主管道
            "StableDiffusionPipeline",
            # 添加稳定扩散上采样管道
            "StableDiffusionUpscalePipeline",
            # 添加稳定 UnCLIP 图像到图像管道
            "StableUnCLIPImg2ImgPipeline",
            # 添加稳定 UnCLIP 管道
            "StableUnCLIPPipeline",
            # 添加稳定扩散 LDM 3D 管道
            "StableDiffusionLDM3DPipeline",
        ]
    )
    # 为 aura_flow 设置导入结构，包括其管道
    _import_structure["aura_flow"] = ["AuraFlowPipeline"]
    # 为 stable_diffusion_3 设置相关管道
    _import_structure["stable_diffusion_3"] = [
        # 添加稳定扩散 3 管道
        "StableDiffusion3Pipeline",
        # 添加稳定扩散 3 图像到图像管道
        "StableDiffusion3Img2ImgPipeline",
        # 添加稳定扩散 3 图像修复管道
        "StableDiffusion3InpaintPipeline",
    ]
    # 为 stable_diffusion_attend_and_excite 设置导入结构
    _import_structure["stable_diffusion_attend_and_excite"] = ["StableDiffusionAttendAndExcitePipeline"]
    # 为 stable_diffusion_safe 设置安全管道
    _import_structure["stable_diffusion_safe"] = ["StableDiffusionPipelineSafe"]
    # 为 stable_diffusion_sag 设置导入结构
    _import_structure["stable_diffusion_sag"] = ["StableDiffusionSAGPipeline"]
    # 为 stable_diffusion_gligen 设置导入结构
    _import_structure["stable_diffusion_gligen"] = [
        # 添加稳定扩散 GLIGEN 管道
        "StableDiffusionGLIGENPipeline",
        # 添加稳定扩散 GLIGEN 文本图像管道
        "StableDiffusionGLIGENTextImagePipeline",
    ]
    # 为 stable_video_diffusion 设置导入结构
    _import_structure["stable_video_diffusion"] = ["StableVideoDiffusionPipeline"]
    # 向 stable_diffusion_xl 的导入结构中添加多个管道
    _import_structure["stable_diffusion_xl"].extend(
        [
            # 添加稳定扩散 XL 图像到图像管道
            "StableDiffusionXLImg2ImgPipeline",
            # 添加稳定扩散 XL 图像修复管道
            "StableDiffusionXLInpaintPipeline",
            # 添加稳定扩散 XL 指令图像到图像管道
            "StableDiffusionXLInstructPix2PixPipeline",
            # 添加稳定扩散 XL 主管道
            "StableDiffusionXLPipeline",
        ]
    )
    # 为 stable_diffusion_diffedit 设置导入结构
    _import_structure["stable_diffusion_diffedit"] = ["StableDiffusionDiffEditPipeline"]
    # 为 stable_diffusion_ldm3d 设置导入结构
    _import_structure["stable_diffusion_ldm3d"] = ["StableDiffusionLDM3DPipeline"]
    # 为 stable_diffusion_panorama 设置导入结构
    _import_structure["stable_diffusion_panorama"] = ["StableDiffusionPanoramaPipeline"]
    # 为 t2i_adapter 设置导入结构，包括适配管道
    _import_structure["t2i_adapter"] = [
        # 添加稳定扩散适配器管道
        "StableDiffusionAdapterPipeline",
        # 添加稳定扩散 XL 适配器管道
        "StableDiffusionXLAdapterPipeline",
    ]
    # 为 text_to_video_synthesis 设置多个视频合成相关管道
    _import_structure["text_to_video_synthesis"] = [
        # 添加文本到视频稳定扩散管道
        "TextToVideoSDPipeline",
        # 添加文本到视频零管道
        "TextToVideoZeroPipeline",
        # 添加文本到视频零稳定扩散 XL 管道
        "TextToVideoZeroSDXLPipeline",
        # 添加视频到视频稳定扩散管道
        "VideoToVideoSDPipeline",
    ]
    # 为 i2vgen_xl 设置导入结构
    _import_structure["i2vgen_xl"] = ["I2VGenXLPipeline"]
    # 为 unclip 设置相关管道
    _import_structure["unclip"] = ["UnCLIPImageVariationPipeline", "UnCLIPPipeline"]
    # 为 unidiffuser 设置多个管道
    _import_structure["unidiffuser"] = [
        # 添加图像文本管道输出
        "ImageTextPipelineOutput",
        # 添加 UniDiffuser 模型
        "UniDiffuserModel",
        # 添加 UniDiffuser 管道
        "UniDiffuserPipeline",
        # 添加 UniDiffuser 文本解码器
        "UniDiffuserTextDecoder",
    ]
    # 为 wuerstchen 设置多个管道
    _import_structure["wuerstchen"] = [
        # 添加 Wuerstchen 组合管道
        "WuerstchenCombinedPipeline",
        # 添加 Wuerstchen 解码器管道
        "WuerstchenDecoderPipeline",
        # 添加 Wuerstchen 先验管道
        "WuerstchenPriorPipeline",
    ]
# 尝试检查 ONNX 是否可用
try:
    # 如果 ONNX 不可用，抛出异常
    if not is_onnx_available():
        raise OptionalDependencyNotAvailable()
# 捕获可选依赖不可用的异常
except OptionalDependencyNotAvailable:
    # 从工具模块导入假 ONNX 对象，防止导入错误
    from ..utils import dummy_onnx_objects  # noqa F403

    # 更新虚拟对象字典，添加假 ONNX 对象
    _dummy_objects.update(get_objects_from_module(dummy_onnx_objects))
# 如果 ONNX 可用，更新导入结构
else:
    _import_structure["onnx_utils"] = ["OnnxRuntimeModel"]

# 尝试检查 PyTorch、Transformers 和 ONNX 是否都可用
try:
    # 如果任何一个不可用，抛出异常
    if not (is_torch_available() and is_transformers_available() and is_onnx_available()):
        raise OptionalDependencyNotAvailable()
# 捕获可选依赖不可用的异常
except OptionalDependencyNotAvailable:
    # 从工具模块导入假 PyTorch、Transformers 和 ONNX 对象
    from ..utils import dummy_torch_and_transformers_and_onnx_objects  # noqa F403

    # 更新虚拟对象字典，添加假对象
    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_and_onnx_objects))
# 如果都可用，扩展导入结构
else:
    _import_structure["stable_diffusion"].extend(
        [
            "OnnxStableDiffusionImg2ImgPipeline",
            "OnnxStableDiffusionInpaintPipeline",
            "OnnxStableDiffusionPipeline",
            "OnnxStableDiffusionUpscalePipeline",
            "StableDiffusionOnnxPipeline",
        ]
    )

# 尝试检查 PyTorch、Transformers 和 K-Diffusion 是否都可用
try:
    # 如果任何一个不可用，抛出异常
    if not (is_torch_available() and is_transformers_available() and is_k_diffusion_available()):
        raise OptionalDependencyNotAvailable()
# 捕获可选依赖不可用的异常
except OptionalDependencyNotAvailable:
    # 从工具模块导入假 PyTorch、Transformers 和 K-Diffusion 对象
    from ..utils import (
        dummy_torch_and_transformers_and_k_diffusion_objects,
    )

    # 更新虚拟对象字典，添加假对象
    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_and_k_diffusion_objects))
# 如果都可用，更新导入结构
else:
    _import_structure["stable_diffusion_k_diffusion"] = [
        "StableDiffusionKDiffusionPipeline",
        "StableDiffusionXLKDiffusionPipeline",
    ]

# 尝试检查 PyTorch、Transformers 和 SentencePiece 是否都可用
try:
    # 如果任何一个不可用，抛出异常
    if not (is_torch_available() and is_transformers_available() and is_sentencepiece_available()):
        raise OptionalDependencyNotAvailable()
# 捕获可选依赖不可用的异常
except OptionalDependencyNotAvailable:
    # 从工具模块导入假 PyTorch、Transformers 和 SentencePiece 对象
    from ..utils import (
        dummy_torch_and_transformers_and_sentencepiece_objects,
    )

    # 更新虚拟对象字典，添加假对象
    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_and_sentencepiece_objects))
# 如果都可用，更新导入结构
else:
    _import_structure["kolors"] = [
        "KolorsPipeline",
        "KolorsImg2ImgPipeline",
    ]

# 尝试检查 Flax 是否可用
try:
    # 如果 Flax 不可用，抛出异常
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
# 捕获可选依赖不可用的异常
except OptionalDependencyNotAvailable:
    # 从工具模块导入假 Flax 对象，防止导入错误
    from ..utils import dummy_flax_objects  # noqa F403

    # 更新虚拟对象字典，添加假 Flax 对象
    _dummy_objects.update(get_objects_from_module(dummy_flax_objects))
# 如果 Flax 可用，更新导入结构
else:
    _import_structure["pipeline_flax_utils"] = ["FlaxDiffusionPipeline"]

# 尝试检查 Flax 和 Transformers 是否都可用
try:
    # 如果任何一个不可用，抛出异常
    if not (is_flax_available() and is_transformers_available()):
        raise OptionalDependencyNotAvailable()
# 捕获可选依赖不可用的异常
except OptionalDependencyNotAvailable:
    # 从工具模块导入假 Flax 和 Transformers 对象
    from ..utils import dummy_flax_and_transformers_objects  # noqa F403

    # 更新虚拟对象字典，添加假对象
    _dummy_objects.update(get_objects_from_module(dummy_flax_and_transformers_objects))
# 如果都可用，扩展导入结构
else:
    _import_structure["controlnet"].extend(["FlaxStableDiffusionControlNetPipeline"])
    # 将稳定扩散模型相关的类名添加到导入结构中
        _import_structure["stable_diffusion"].extend(
            [
                # 添加图像到图像转换管道类名
                "FlaxStableDiffusionImg2ImgPipeline",
                # 添加图像修复管道类名
                "FlaxStableDiffusionInpaintPipeline",
                # 添加基础稳定扩散管道类名
                "FlaxStableDiffusionPipeline",
            ]
        )
    # 将稳定扩散 XL 模型相关的类名添加到导入结构中
        _import_structure["stable_diffusion_xl"].extend(
            [
                # 添加稳定扩散 XL 管道类名
                "FlaxStableDiffusionXLPipeline",
            ]
        )
# 检查是否为类型检查或慢导入条件
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        # 检查是否可用 PyTorch
        if not is_torch_available():
            # 如果不可用，则引发可选依赖项不可用异常
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 导入占位符对象以避免运行时错误
        from ..utils.dummy_pt_objects import *  # noqa F403

    else:
        # 导入自动图像到图像管道相关类
        from .auto_pipeline import (
            AutoPipelineForImage2Image,
            AutoPipelineForInpainting,
            AutoPipelineForText2Image,
        )
        # 导入一致性模型管道
        from .consistency_models import ConsistencyModelPipeline
        # 导入舞蹈扩散管道
        from .dance_diffusion import DanceDiffusionPipeline
        # 导入 DDIM 管道
        from .ddim import DDIMPipeline
        # 导入 DDPM 管道
        from .ddpm import DDPMPipeline
        # 导入已弃用的管道
        from .deprecated import KarrasVePipeline, LDMPipeline, PNDMPipeline, RePaintPipeline, ScoreSdeVePipeline
        # 导入 DIT 管道
        from .dit import DiTPipeline
        # 导入潜在扩散超分辨率管道
        from .latent_diffusion import LDMSuperResolutionPipeline
        # 导入管道工具类
        from .pipeline_utils import (
            AudioPipelineOutput,
            DiffusionPipeline,
            ImagePipelineOutput,
            StableDiffusionMixin,
        )

    try:
        # 检查是否可用 PyTorch 和 librosa
        if not (is_torch_available() and is_librosa_available()):
            # 如果不可用，则引发可选依赖项不可用异常
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 导入占位符对象以避免运行时错误
        from ..utils.dummy_torch_and_librosa_objects import *
    else:
        # 导入已弃用的音频扩散管道和 Mel 类
        from .deprecated import AudioDiffusionPipeline, Mel

    try:
        # 检查是否可用 PyTorch 和 transformers
        if not (is_torch_available() and is_transformers_available()):
            # 如果不可用，则引发可选依赖项不可用异常
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 导入占位符对象以避免运行时错误
        from ..utils.dummy_torch_and_transformers_objects import *
else:
    # 导入 sys 模块
    import sys

    # 创建懒加载模块实例
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
    # 设置占位符对象到当前模块
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
```