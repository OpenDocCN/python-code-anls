# `.\diffusers\models\__init__.py`

```py
# 版权声明，表明该代码属于 HuggingFace 团队，保留所有权利。
# 根据 Apache 2.0 许可证进行许可，使用该文件需遵循许可证的条款。
# 提供许可证的获取链接。
#
# 许可证条款的摘要，软件在“现状”基础上分发，没有任何明示或暗示的保证。
# 参考许可证以获取关于权限和限制的具体信息。

# 导入类型检查模块
from typing import TYPE_CHECKING

# 从上层目录的 utils 模块导入所需的工具和常量
from ..utils import (
    DIFFUSERS_SLOW_IMPORT,  # 慢导入的常量
    _LazyModule,            # 延迟加载模块的工具
    is_flax_available,      # 检查 Flax 库是否可用的函数
    is_torch_available,     # 检查 PyTorch 库是否可用的函数
)

# 初始化一个空的字典，用于存储导入结构
_import_structure = {}

# 如果 PyTorch 可用，添加相关模块到导入结构字典
if is_torch_available():
    _import_structure["adapter"] = ["MultiAdapter", "T2IAdapter"]  # 适配器模块
    _import_structure["autoencoders.autoencoder_asym_kl"] = ["AsymmetricAutoencoderKL"]  # 非对称自动编码器
    _import_structure["autoencoders.autoencoder_kl"] = ["AutoencoderKL"]  # 自动编码器
    _import_structure["autoencoders.autoencoder_kl_cogvideox"] = ["AutoencoderKLCogVideoX"]  # CogVideoX 自动编码器
    _import_structure["autoencoders.autoencoder_kl_temporal_decoder"] = ["AutoencoderKLTemporalDecoder"]  # 时间解码器
    _import_structure["autoencoders.autoencoder_oobleck"] = ["AutoencoderOobleck"]  # Oobleck 自动编码器
    _import_structure["autoencoders.autoencoder_tiny"] = ["AutoencoderTiny"]  # Tiny 自动编码器
    _import_structure["autoencoders.consistency_decoder_vae"] = ["ConsistencyDecoderVAE"]  # 一致性解码器 VAE
    _import_structure["autoencoders.vq_model"] = ["VQModel"]  # VQ 模型
    _import_structure["controlnet"] = ["ControlNetModel"]  # 控制网络模型
    _import_structure["controlnet_hunyuan"] = ["HunyuanDiT2DControlNetModel", "HunyuanDiT2DMultiControlNetModel"]  # Hunyuan 控制网络模型
    _import_structure["controlnet_sd3"] = ["SD3ControlNetModel", "SD3MultiControlNetModel"]  # SD3 控制网络模型
    _import_structure["controlnet_sparsectrl"] = ["SparseControlNetModel"]  # 稀疏控制网络模型
    _import_structure["controlnet_xs"] = ["ControlNetXSAdapter", "UNetControlNetXSModel"]  # XS 适配器和 U-Net 模型
    _import_structure["embeddings"] = ["ImageProjection"]  # 图像投影模块
    _import_structure["modeling_utils"] = ["ModelMixin"]  # 模型混合工具
    _import_structure["transformers.auraflow_transformer_2d"] = ["AuraFlowTransformer2DModel"]  # AuraFlow 2D 模型
    _import_structure["transformers.cogvideox_transformer_3d"] = ["CogVideoXTransformer3DModel"]  # CogVideoX 3D 模型
    _import_structure["transformers.dit_transformer_2d"] = ["DiTTransformer2DModel"]  # DiT 2D 模型
    _import_structure["transformers.dual_transformer_2d"] = ["DualTransformer2DModel"]  # 双重 2D 模型
    _import_structure["transformers.hunyuan_transformer_2d"] = ["HunyuanDiT2DModel"]  # Hunyuan 2D 模型
    _import_structure["transformers.latte_transformer_3d"] = ["LatteTransformer3DModel"]  # Latte 3D 模型
    _import_structure["transformers.lumina_nextdit2d"] = ["LuminaNextDiT2DModel"]  # Lumina Next DiT 2D 模型
    _import_structure["transformers.pixart_transformer_2d"] = ["PixArtTransformer2DModel"]  # PixArt 2D 模型
    _import_structure["transformers.prior_transformer"] = ["PriorTransformer"]  # 优先变换模型
    _import_structure["transformers.stable_audio_transformer"] = ["StableAudioDiTModel"]  # 稳定音频模型
    # 将 T5FilmDecoder 类添加到 transformers.t5_film_transformer 的导入结构中
        _import_structure["transformers.t5_film_transformer"] = ["T5FilmDecoder"]
        # 将 Transformer2DModel 类添加到 transformers.transformer_2d 的导入结构中
        _import_structure["transformers.transformer_2d"] = ["Transformer2DModel"]
        # 将 FluxTransformer2DModel 类添加到 transformers.transformer_flux 的导入结构中
        _import_structure["transformers.transformer_flux"] = ["FluxTransformer2DModel"]
        # 将 SD3Transformer2DModel 类添加到 transformers.transformer_sd3 的导入结构中
        _import_structure["transformers.transformer_sd3"] = ["SD3Transformer2DModel"]
        # 将 TransformerTemporalModel 类添加到 transformers.transformer_temporal 的导入结构中
        _import_structure["transformers.transformer_temporal"] = ["TransformerTemporalModel"]
        # 将 UNet1DModel 类添加到 unets.unet_1d 的导入结构中
        _import_structure["unets.unet_1d"] = ["UNet1DModel"]
        # 将 UNet2DModel 类添加到 unets.unet_2d 的导入结构中
        _import_structure["unets.unet_2d"] = ["UNet2DModel"]
        # 将 UNet2DConditionModel 类添加到 unets.unet_2d_condition 的导入结构中
        _import_structure["unets.unet_2d_condition"] = ["UNet2DConditionModel"]
        # 将 UNet3DConditionModel 类添加到 unets.unet_3d_condition 的导入结构中
        _import_structure["unets.unet_3d_condition"] = ["UNet3DConditionModel"]
        # 将 I2VGenXLUNet 类添加到 unets.unet_i2vgen_xl 的导入结构中
        _import_structure["unets.unet_i2vgen_xl"] = ["I2VGenXLUNet"]
        # 将 Kandinsky3UNet 类添加到 unets.unet_kandinsky3 的导入结构中
        _import_structure["unets.unet_kandinsky3"] = ["Kandinsky3UNet"]
        # 将 MotionAdapter 和 UNetMotionModel 类添加到 unets.unet_motion_model 的导入结构中
        _import_structure["unets.unet_motion_model"] = ["MotionAdapter", "UNetMotionModel"]
        # 将 UNetSpatioTemporalConditionModel 类添加到 unets.unet_spatio_temporal_condition 的导入结构中
        _import_structure["unets.unet_spatio_temporal_condition"] = ["UNetSpatioTemporalConditionModel"]
        # 将 StableCascadeUNet 类添加到 unets.unet_stable_cascade 的导入结构中
        _import_structure["unets.unet_stable_cascade"] = ["StableCascadeUNet"]
        # 将 UVit2DModel 类添加到 unets.uvit_2d 的导入结构中
        _import_structure["unets.uvit_2d"] = ["UVit2DModel"]
# 检查 Flax 库是否可用
if is_flax_available():
    # 在导入结构中添加 ControlNet 的 Flax 模型
    _import_structure["controlnet_flax"] = ["FlaxControlNetModel"]
    # 在导入结构中添加 2D 条件 UNet 的 Flax 模型
    _import_structure["unets.unet_2d_condition_flax"] = ["FlaxUNet2DConditionModel"]
    # 在导入结构中添加 VAE 的 Flax 模型
    _import_structure["vae_flax"] = ["FlaxAutoencoderKL"]

# 检查类型检查或慢导入条件
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    # 检查 PyTorch 库是否可用
    if is_torch_available():
        # 从适配器模块导入多个适配器类
        from .adapter import MultiAdapter, T2IAdapter
        # 从自动编码器模块导入多个自动编码器类
        from .autoencoders import (
            AsymmetricAutoencoderKL,
            AutoencoderKL,
            AutoencoderKLCogVideoX,
            AutoencoderKLTemporalDecoder,
            AutoencoderOobleck,
            AutoencoderTiny,
            ConsistencyDecoderVAE,
            VQModel,
        )
        # 从 ControlNet 模块导入 ControlNet 模型
        from .controlnet import ControlNetModel
        # 从 Hunyuan ControlNet 模块导入模型
        from .controlnet_hunyuan import HunyuanDiT2DControlNetModel, HunyuanDiT2DMultiControlNetModel
        # 从 SD3 ControlNet 模块导入模型
        from .controlnet_sd3 import SD3ControlNetModel, SD3MultiControlNetModel
        # 从 SparseControlNet 模块导入模型
        from .controlnet_sparsectrl import SparseControlNetModel
        # 从 XS ControlNet 模块导入适配器和模型
        from .controlnet_xs import ControlNetXSAdapter, UNetControlNetXSModel
        # 从嵌入模块导入图像投影类
        from .embeddings import ImageProjection
        # 从建模工具模块导入模型混合类
        from .modeling_utils import ModelMixin
        # 从转换器模块导入多个转换器类
        from .transformers import (
            AuraFlowTransformer2DModel,
            CogVideoXTransformer3DModel,
            DiTTransformer2DModel,
            DualTransformer2DModel,
            FluxTransformer2DModel,
            HunyuanDiT2DModel,
            LatteTransformer3DModel,
            LuminaNextDiT2DModel,
            PixArtTransformer2DModel,
            PriorTransformer,
            SD3Transformer2DModel,
            StableAudioDiTModel,
            T5FilmDecoder,
            Transformer2DModel,
            TransformerTemporalModel,
        )
        # 从 UNet 模块导入多个 UNet 类
        from .unets import (
            I2VGenXLUNet,
            Kandinsky3UNet,
            MotionAdapter,
            StableCascadeUNet,
            UNet1DModel,
            UNet2DConditionModel,
            UNet2DModel,
            UNet3DConditionModel,
            UNetMotionModel,
            UNetSpatioTemporalConditionModel,
            UVit2DModel,
        )

    # 检查 Flax 库是否可用
    if is_flax_available():
        # 从 Flax ControlNet 模块导入模型
        from .controlnet_flax import FlaxControlNetModel
        # 从 UNet 模块导入 Flax 2D 条件模型
        from .unets import FlaxUNet2DConditionModel
        # 从 Flax VAE 模块导入模型
        from .vae_flax import FlaxAutoencoderKL

# 如果以上条件都不满足
else:
    # 导入系统模块
    import sys

    # 用懒加载模块替代当前模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```