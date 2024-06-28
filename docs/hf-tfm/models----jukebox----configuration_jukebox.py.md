# `.\models\jukebox\configuration_jukebox.py`

```
# coding=utf-8
# 版权 2022 年 OpenAI 团队和 HuggingFace Inc. 团队。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可；
# 除非符合许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件按“原样”分发，
# 不附带任何明示或暗示的保证或条件。
# 有关详细信息，请参阅许可证。
""" Jukebox 配置 """

import os
from typing import List, Union

from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# Jukebox 预训练配置文件映射
JUKEBOX_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "openai/jukebox-5b-lyrics": "https://huggingface.co/openai/jukebox-5b-lyrics/blob/main/config.json",
    "openai/jukebox-1b-lyrics": "https://huggingface.co/openai/jukebox-1b-lyrics/blob/main/config.json",
}

# 大型注意力列表
_LARGE_ATTENTION = [
    "block_attn",  # 块注意力
    "transpose_block_attn",  # 转置块注意力
    "prev_block_attn",  # 前一块注意力
    "block_attn",  # 块注意力
    "transpose_block_attn",  # 转置块注意力
    "prev_block_attn",  # 前一块注意力
    "block_attn",  # 块注意力
    "transpose_block_attn",  # 转置块注意力
    "prev_block_attn",  # 前一块注意力
    "block_attn",  # 块注意力
    "transpose_block_attn",  # 转置块注意力
    "prev_block_attn",  # 前一块注意力
    "block_attn",  # 块注意力
    "transpose_block_attn",  # 转置块注意力
    "prev_block_attn",  # 前一块注意力
    "block_attn",  # 块注意力
    "transpose_block_attn",  # 转置块注意力
    "prev_block_attn",  # 前一块注意力
    "cross_attention",  # 交叉注意力
    "block_attn",  # 块注意力
    "transpose_block_attn",  # 转置块注意力
    "prev_block_attn",  # 前一块注意力
    "block_attn",  # 块注意力
    "transpose_block_attn",  # 转置块注意力
    "prev_block_attn",  # 前一块注意力
    "block_attn",  # 块注意力
    "transpose_block_attn",  # 转置块注意力
    "prev_block_attn",  # 前一块注意力
    "cross_attention",  # 交叉注意力
    "block_attn",  # 块注意力
    "transpose_block_attn",  # 转置块注意力
    "prev_block_attn",  # 前一块注意力
    "block_attn",  # 块注意力
    "transpose_block_attn",  # 转置块注意力
    "prev_block_attn",  # 前一块注意力
    "block_attn",  # 块注意力
    "transpose_block_attn",  # 转置块注意力
    "prev_block_attn",  # 前一块注意力
    "cross_attention",  # 交叉注意力
    "block_attn",  # 块注意力
    "transpose_block_attn",  # 转置块注意力
    "prev_block_attn",  # 前一块注意力
    "block_attn",  # 块注意力
    "transpose_block_attn",  # 转置块注意力
    "prev_block_attn",  # 前一块注意力
    "block_attn",  # 块注意力
    "transpose_block_attn",  # 转置块注意力
    "prev_block_attn",  # 前一块注意力
    "cross_attention",  # 交叉注意力
    "block_attn",  # 块注意力
    "transpose_block_attn",  # 转置块注意力
    "prev_block_attn",  # 前一块注意力
    "block_attn",  # 块注意力
    "transpose_block_attn",  # 转置块注意力
    "prev_block_attn",  # 前一块注意力
    "block_attn",  # 块注意力
    "transpose_block_attn",  # 转置块注意力
    "prev_block_attn",  # 前一块注意力
    "cross_attention",  # 交叉注意力
]
# 定义三个注意力模式的名称列表
_RawColumnPreviousRowAttention = ["block_attn", "transpose_block_attn", "prev_block_attn"]
# 定义全连接密集注意力模式的名称列表
_FullDenseAttention = ["dense_attention"]
# 定义Prime-Prime-Dense注意力模式的名称列表
_PrimePrimeDenseAttention = ["prime_attn", "prime_attn", "dense_attn"]


# 定义函数，返回全连接密集注意力模式的名称
def full_dense_attention(layer):
    return _FullDenseAttention[0]


# 定义函数，根据层索引返回RawColumnPreviousRowAttention模式的名称
def raw_column_previous_row_attention(layer):
    return _RawColumnPreviousRowAttention[layer % 3]


# 定义函数，根据层索引返回large separated enc dec w lyrics模式的名称
def large_separated_enc_dec_w_lyrics(layer):
    return _LARGE_ATTENTION[layer % 79]  # _LARGE_ATTENTION未定义，可能存在错误


# 定义函数，根据层索引返回enc dec with lyrics模式的名称
def enc_dec_with_lyrics(layer):
    if layer % 16 == 15:
        return _PrimePrimeDenseAttention[layer % 3]
    return _RawColumnPreviousRowAttention[layer % 3]


# 定义全局变量，包含不同注意力模式的名称及其对应的函数引用
ATTENTION_PATTERNS = {
    "full_dense_attention": full_dense_attention,
    "raw_column_previous_row_attention": raw_column_previous_row_attention,  # 用于替换行、列和上一行注意力
    "large_separated_enc_dec_w_lyrics": large_separated_enc_dec_w_lyrics,  # 用于带歌词的大型分离enc dec模型
    "enc_dec_with_lyrics": enc_dec_with_lyrics,  # 用于带歌词的编码器-解码器模型
}


# 定义配置类，存储JukeboxPrior模型的配置信息
class JukeboxPriorConfig(PretrainedConfig):
    """
        This is the configuration class to store the configuration of a [`JukeboxPrior`]. It is used to instantiate a
        `JukeboxPrior` according to the specified arguments, defining the model architecture. Instantiating a
        configuration with the defaults will yield a similar configuration to that of the top level prior from the
        [openai/jukebox-1b-lyrics](https://huggingface.co/openai/jukebox
    -1b-lyrics) architecture.

        Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
        documentation from [`PretrainedConfig`] for more information.



    """

    # 模型类型
    model_type = "jukebox_prior"
    # 属性映射字典，映射配置项到模型的实际参数名
    attribute_map = {
        "max_position_embeddings": "n_positions",  # 最大位置嵌入数对应的模型参数名
        "num_attention_heads": "n_head",  # 注意力头数对应的模型参数名
    }
    # 定义初始化函数，用于创建一个对象实例
    def __init__(
        self,
        act_fn="quick_gelu",  # 激活函数名称，默认为 "quick_gelu"
        level=0,  # 模型的层级，默认为 0
        alignment_head=2,  # 对齐头部参数，默认为 2
        alignment_layer=68,  # 对齐层参数，默认为 68
        attention_multiplier=0.25,  # 注意力乘子，默认为 0.25
        attention_pattern="enc_dec_with_lyrics",  # 注意力模式，默认为 "enc_dec_with_lyrics"
        attn_dropout=0,  # 注意力部分的 dropout 概率，默认为 0
        attn_res_scale=False,  # 注意力残差比例，默认为 False
        blocks=64,  # 块数，默认为 64
        conv_res_scale=None,  # 卷积残差比例，默认为 None
        num_layers=72,  # 层数，默认为 72
        emb_dropout=0,  # 嵌入部分的 dropout 概率，默认为 0
        encoder_config=None,  # 编码器配置信息，默认为 None
        encoder_loss_fraction=0.4,  # 编码器损失分数，默认为 0.4
        hidden_size=2048,  # 隐藏层大小，默认为 2048
        init_scale=0.2,  # 初始化比例，默认为 0.2
        is_encoder_decoder=True,  # 是否是编码解码模型，默认为 True
        lyric_vocab_size=80,  # 歌词词汇量大小，默认为 80
        mask=False,  # 是否使用掩码，默认为 False
        max_duration=600,  # 最大持续时间，默认为 600
        max_nb_genres=1,  # 最大音乐类型数，默认为 1
        merged_decoder=True,  # 是否合并解码器，默认为 True
        metadata_conditioning=True,  # 是否使用元数据条件，默认为 True
        metadata_dims=[604, 7898],  # 元数据维度，默认为 [604, 7898]
        min_duration=0,  # 最小持续时间，默认为 0
        mlp_multiplier=1.0,  # 多层感知机乘数，默认为 1.0
        music_vocab_size=2048,  # 音乐词汇量大小，默认为 2048
        n_ctx=6144,  # 上下文大小，默认为 6144
        n_heads=2,  # 多头注意力的头数，默认为 2
        nb_relevant_lyric_tokens=384,  # 相关歌词标记数，默认为 384
        res_conv_depth=3,  # 残余卷积深度，默认为 3
        res_conv_width=128,  # 残余卷积宽度，默认为 128
        res_convolution_multiplier=1,  # 残余卷积乘数，默认为 1
        res_dilation_cycle=None,  # 残余扩张周期，默认为 None
        res_dilation_growth_rate=1,  # 残余扩张增长率，默认为 1
        res_downs_t=[3, 2, 2],  # 残余下采样时序，默认为 [3, 2, 2]
        res_strides_t=[2, 2, 2],  # 残余步长时序，默认为 [2, 2, 2]
        resid_dropout=0,  # 残余 dropout 概率，默认为 0
        sampling_rate=44100,  # 采样率，默认为 44100
        spread=None,  # 传播参数，默认为 None
        timing_dims=64,  # 时间维度，默认为 64
        zero_out=False,  # 是否清零，默认为 False
        **kwargs,  # 其他关键字参数，用于捕获未指定的关键字参数
    ):
        ):
        # 初始化函数，接受多个参数并将它们赋值给对象的属性
        self.act_fn = act_fn
        self.alignment_head = alignment_head
        self.alignment_layer = alignment_layer
        self.attention_multiplier = attention_multiplier
        self.attention_pattern = attention_pattern
        self.attn_dropout = attn_dropout
        self.attn_res_scale = attn_res_scale
        self.blocks = blocks
        self.conv_res_scale = conv_res_scale
        self.num_layers = num_layers
        self.emb_dropout = emb_dropout
        self.music_vocab_size = music_vocab_size
        # 如果提供了编码器配置，将其转换为 JukeboxPriorConfig 对象
        if encoder_config is not None:
            self.encoder_config = JukeboxPriorConfig(**encoder_config)
        else:
            self.encoder_config = None
        self.encoder_loss_fraction = encoder_loss_fraction
        self.init_scale = init_scale
        self.is_encoder_decoder = is_encoder_decoder
        self.lyric_vocab_size = lyric_vocab_size
        self.level = level
        self.mask = mask
        self.max_duration = max_duration
        self.max_nb_genres = max_nb_genres
        self.merged_decoder = merged_decoder
        self.metadata_conditioning = metadata_conditioning
        self.metadata_dims = metadata_dims
        self.min_duration = min_duration
        self.mlp_multiplier = mlp_multiplier
        self.n_ctx = n_ctx
        self.n_heads = n_heads
        self.nb_relevant_lyric_tokens = nb_relevant_lyric_tokens
        self.res_conv_depth = res_conv_depth
        self.res_conv_width = res_conv_width
        self.res_convolution_multiplier = res_convolution_multiplier
        self.res_dilation_cycle = res_dilation_cycle
        self.res_dilation_growth_rate = res_dilation_growth_rate
        self.res_downs_t = res_downs_t
        self.res_strides_t = res_strides_t
        self.resid_dropout = resid_dropout
        self.sampling_rate = sampling_rate
        self.spread = spread
        self.timing_dims = timing_dims
        self.hidden_size = hidden_size
        self.zero_out = zero_out
        # 设置对象的初始化完成标志

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], level=0, **kwargs
    ) -> "PretrainedConfig":
        # 设置传递给类方法的特殊标记位
        cls._set_token_in_kwargs(kwargs)

        # 获取配置字典和更新后的 kwargs
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果配置字典指定模型类型为 "jukebox"，则使用对应级别的先验配置
        if config_dict.get("model_type") == "jukebox":
            config_dict = config_dict[f"prior_{level}"]

        # 检查配置字典中的模型类型是否与类的模型类型匹配，如果不匹配则发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 从配置字典创建并返回一个新的 PretrainedConfig 对象实例
        return cls.from_dict(config_dict, **kwargs)
# 定义 JukeboxVQVAEConfig 类，继承自 PretrainedConfig 类，用于存储 JukeboxVQVAE 模型的配置信息
class JukeboxVQVAEConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`JukeboxVQVAE`]. It is used to instantiate a
    `JukeboxVQVAE` according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the VQVAE from
    [openai/jukebox-1b-lyrics](https://huggingface.co/openai/jukebox-1b-lyrics) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    # 定义一个函数，用于构建 VQVAE 模型
    def build_model(
        act_fn: str = "relu",  # 激活函数，默认为 ReLU
        nb_discrete_codes: int = 2048,  # VQVAE 的离散码数量，默认为 2048
        commit: float = 0.02,  # Commit loss 的乘数，默认为 0.02
        conv_input_shape: int = 1,  # 音频通道数，默认为 1
        conv_res_scale: bool = False,  # 是否缩放 JukeboxResConv1DBlock 的残差，默认为 False
        embed_dim: int = 64,  # Codebook 向量的嵌入维度，默认为 64
        hop_fraction: List[int] = [0.125, 0.5, 0.5],  # 进行采样过程时使用的非交叠窗口的分数列表，默认为 [0.125, 0.5, 0.5]
        levels: int = 3,  # 在 VQVAE 中使用的层级数，默认为 3
        lmu: float = 0.99,  # 用于代码本更新的指数移动平均系数，默认为 0.99
        multipliers: List[int] = [2, 1, 1],  # 每个层级使用的深度和宽度乘数列表，默认为 [2, 1, 1]
        res_conv_depth: int = 4,  # 编码器和解码器块的深度，默认为 4
        res_conv_width: int = 32,  # 编码器和解码器块的宽度，默认为 32
        res_convolution_multiplier: int = 1,  # JukeboxResConv1DBlock 中隐藏维度的缩放因子，默认为 1
        res_dilation_cycle: int = None,  # JukeboxResnet 中使用的扩张周期值，默认为 None
        res_dilation_growth_rate: int = 3,  # VQVAE 中使用的 ResNet 扩张增长率，默认为 3
        res_downs_t: List[int] = [3, 2, 2],  # 分层 VQ-VAE 中每个层级的下采样率列表，默认为 [3, 2, 2]
        res_strides_t: List[int] = [2, 2, 2],  # 分层 VQ-VAE 中每个层级的步长列表，默认为 [2, 2, 2]
        sample_length: int = 1058304,  # VQVAE 的最大输入形状，默认为 1058304
        init_scale: float = 0.2,  # 初始化尺度，默认为 0.2
        zero_out: bool = False,  # 初始化时是否将卷积权重置零，默认为 False
    ):
        """
        构建 VQVAE 模型，根据给定的参数设置各种配置和参数。
        """
        # 函数体为空，用于声明函数的开始
        pass
    # 设定模型类型为 "jukebox_vqvae"
    model_type = "jukebox_vqvae"
    
    # 定义类的初始化方法，接受多个参数
    def __init__(
        self,
        act_fn="relu",  # 激活函数，默认为 relu
        nb_discrete_codes=2048,  # 离散代码数量，默认为 2048
        commit=0.02,  # commit 参数，默认为 0.02
        conv_input_shape=1,  # 卷积输入形状，默认为 1
        conv_res_scale=False,  # 是否使用卷积残差缩放，默认为 False
        embed_dim=64,  # 嵌入维度，默认为 64
        hop_fraction=[0.125, 0.5, 0.5],  # hop fraction 列表，默认值为 [0.125, 0.5, 0.5]
        levels=3,  # 级别数量，默认为 3
        lmu=0.99,  # lmu 参数，默认为 0.99
        multipliers=[2, 1, 1],  # 多重因子列表，默认为 [2, 1, 1]
        res_conv_depth=4,  # 卷积深度，默认为 4
        res_conv_width=32,  # 卷积宽度，默认为 32
        res_convolution_multiplier=1,  # 卷积乘数，默认为 1
        res_dilation_cycle=None,  # 膨胀周期，默认为 None
        res_dilation_growth_rate=3,  # 膨胀增长率，默认为 3
        res_downs_t=[3, 2, 2],  # 下采样 t 列表，默认为 [3, 2, 2]
        res_strides_t=[2, 2, 2],  # 步幅 t 列表，默认为 [2, 2, 2]
        sample_length=1058304,  # 样本长度，默认为 1058304
        init_scale=0.2,  # 初始化规模，默认为 0.2
        zero_out=False,  # 是否置零，默认为 False
        **kwargs,  # 其他关键字参数
    ):
        self.hop_fraction = hop_fraction  # 设置类属性 hop_fraction
        self.conv_input_shape = conv_input_shape  # 设置类属性 conv_input_shape
        self.sample_length = sample_length  # 设置类属性 sample_length
    
        # 设置 VQVAE 参数（全部使用）
        self.levels = levels
        self.embed_dim = embed_dim
        self.nb_discrete_codes = nb_discrete_codes
        self.res_conv_width = res_conv_width
        self.res_conv_depth = res_conv_depth
        self.res_convolution_multiplier = res_convolution_multiplier
        self.res_dilation_growth_rate = res_dilation_growth_rate
        self.res_dilation_cycle = res_dilation_cycle
        self.multipliers = multipliers
        self.res_downs_t = res_downs_t
        self.res_strides_t = res_strides_t
        self.lmu = lmu
        self.commit = commit
        self.conv_res_scale = conv_res_scale
        self.act_fn = act_fn
        self.init_scale = init_scale
        self.zero_out = zero_out
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)  # 在关键字参数中设置令牌
    
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)  # 获取配置字典和更新后的关键字参数
    
        # 如果加载的是 CLIPConfig，获取文本配置字典
        if config_dict.get("model_type") == "jukebox":
            config_dict = config_dict["vqvae_config"]
    
        # 检查配置字典中的模型类型是否与类中定义的模型类型一致
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )
    
        # 从配置字典和关键字参数实例化类并返回
        return cls.from_dict(config_dict, **kwargs)
class JukeboxConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`JukeboxModel`].

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information. Instantiating a configuration with the defaults will
    yield a similar configuration to that of
    [openai/jukebox-1b-lyrics](https://huggingface.co/openai/jukebox-1b-lyrics) architecture.


    The downsampling and stride are used to determine downsampling of the input sequence. For example, downsampling =
    (5,3), and strides = (2, 2) will downsample the audio by 2^5 = 32 to get the first level of codes, and 2**8 = 256
    to get the second level codes. This is mostly true for training the top level prior and the upsamplers.

    Args:
        vqvae_config (`JukeboxVQVAEConfig`, *optional*):
            Configuration for the `JukeboxVQVAE` model.
        prior_config_list (`List[JukeboxPriorConfig]`, *optional*):
            List of the configs for each of the `JukeboxPrior` of the model. The original architecture uses 3 priors.
        nb_priors (`int`, *optional*, defaults to 3):
            Number of prior models that will sequentially sample tokens. Each prior is conditional auto regressive
            (decoder) model, apart from the top prior, which can include a lyric encoder. The available models were
            trained using a top prior and 2 upsampler priors.
        sampling_rate (`int`, *optional*, defaults to 44100):
            Sampling rate of the raw audio.
        timing_dims (`int`, *optional*, defaults to 64):
            Dimensions of the JukeboxRangeEmbedding layer which is equivalent to traditional positional embedding
            layer. The timing embedding layer converts the absolute and relative position in the currently sampled
            audio to a tensor of length `timing_dims` that will be added to the music tokens.
        min_duration (`int`, *optional*, defaults to 0):
            Minimum duration of the audios to generate
        max_duration (`float`, *optional*, defaults to 600.0):
            Maximum duration of the audios to generate
        max_nb_genres (`int`, *optional*, defaults to 5):
            Maximum number of genres that can be used to condition a single sample.
        metadata_conditioning (`bool`, *optional*, defaults to `True`):
            Whether or not to use metadata conditioning, corresponding to the artist, the genre and the min/maximum
            duration.

    Example:

    ```python
    >>> from transformers import JukeboxModel, JukeboxConfig

    >>> # Initializing a Jukebox configuration
    >>> configuration = JukeboxConfig()

    >>> # Initializing a model from the configuration
    >>> model = JukeboxModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    # 类型标识符，用于标识该配置类是`jukebox`类型的配置
    model_type = "jukebox"
    # 初始化方法，用于实例化 JukeboxConfig 对象
    def __init__(
        self,
        vqvae_config=None,
        prior_config_list=None,
        nb_priors=3,
        sampling_rate=44100,
        timing_dims=64,
        min_duration=0,
        max_duration=600.0,
        max_nb_genres=5,
        metadata_conditioning=True,
        **kwargs,
    ):
        # 如果 vqvae_config 为 None，则用空字典初始化
        if vqvae_config is None:
            vqvae_config = {}
            # 记录日志，说明 vqvae_config 是 None，使用默认值初始化 JukeboxVQVAE
            logger.info("vqvae_config is None. initializing the JukeboxVQVAE with default values.")

        # 使用给定的 vqvae_config 字典创建 JukeboxVQVAEConfig 对象，并赋值给 self.vqvae_config
        self.vqvae_config = JukeboxVQVAEConfig(**vqvae_config)

        # 如果 prior_config_list 不为 None，则依次用 JukeboxPriorConfig 类实例化列表中的每个配置
        if prior_config_list is not None:
            self.prior_configs = [JukeboxPriorConfig(**prior_config) for prior_config in prior_config_list]
        else:
            # 否则初始化为空列表
            self.prior_configs = []
            # 对于每个 prior_idx 在 nb_priors 范围内，尝试从 kwargs 中获取配置信息，如果没有则使用空字典初始化
            for prior_idx in range(nb_priors):
                prior_config = kwargs.pop(f"prior_{prior_idx}", None)
                if prior_config is None:
                    prior_config = {}
                    # 记录日志，说明该 prior_idx 的配置是 None，使用默认值初始化 JukeboxPriorConfig 列表
                    logger.info(
                        f"prior_{prior_idx}'s  config is None. Initializing the JukeboxPriorConfig list with default"
                        " values."
                    )
                # 使用 prior_config 字典创建 JukeboxPriorConfig 对象，并添加到 prior_configs 列表中
                self.prior_configs.append(JukeboxPriorConfig(**prior_config))

        # 将 vqvae_config 中的 hop_fraction 属性赋值给当前对象的 hop_fraction 属性
        self.hop_fraction = self.vqvae_config.hop_fraction

        # 将传入的各种元数据配置参数赋值给对象的相应属性
        self.nb_priors = nb_priors
        self.max_nb_genres = max_nb_genres
        self.sampling_rate = sampling_rate
        self.timing_dims = timing_dims
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.metadata_conditioning = metadata_conditioning

        # 调用父类的初始化方法，传入剩余的 kwargs 参数
        super().__init__(**kwargs)

    @classmethod
    def from_configs(cls, prior_configs: List[JukeboxPriorConfig], vqvae_config: JukeboxVQVAEConfig, **kwargs):
        r"""
        Instantiate a [`JukeboxConfig`] (or a derived class) from clip text model configuration and clip vision model
        configuration.

        Returns:
            [`JukeboxConfig`]: An instance of a configuration object
        """
        # 将 prior_configs 列表中每个配置对象转换为字典形式，存入 prior_config_list
        prior_config_list = [config.to_dict() for config in prior_configs]
        # 调用当前类的初始化方法，传入 prior_config_list 和 vqvae_config 的字典形式，以及 kwargs 参数
        return cls(prior_config_list=prior_config_list, vqvae_config_dict=vqvae_config.to_dict(), **kwargs)

    def to_dict(self):
        # 重写父类的 to_dict 方法，将对象转换为字典形式
        result = super().to_dict()
        # 将 prior_configs 列表中每个配置对象转换为字典形式，存入 result 字典的 "prior_config_list" 键下
        result["prior_config_list"] = [config.to_dict() for config in result.pop("prior_configs")]
        return result
```