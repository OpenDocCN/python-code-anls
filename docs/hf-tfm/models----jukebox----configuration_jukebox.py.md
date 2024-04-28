# `.\models\jukebox\configuration_jukebox.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，声明代码作者和许可方
# 根据 Apache 许可证 2.0 版本，使用此文件需要遵守许可证规定
# 可以在以下链接获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制
""" Jukebox 配置"""

# 导入所需的库
import os
from typing import List, Union

# 导入预训练配置
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# Jukebox 预训练配置文件映射
JUKEBOX_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "openai/jukebox-5b-lyrics": "https://huggingface.co/openai/jukebox-5b-lyrics/blob/main/config.json",
    "openai/jukebox-1b-lyrics": "https://huggingface.co/openai/jukebox-1b-lyrics/blob/main/config.json",
}

# 大型注意力机制列表
_LARGE_ATTENTION = [
    "block_attn",
    "transpose_block_attn",
    "prev_block_attn",
    # 省略部分内容，此处只展示部分内容
    "cross_attention",
    "block_attn",
    "transpose_block_attn",
    "prev_block_attn",
    # 省略部分内容，此处只展示部分内容
    "cross_attention",
]
# 定义三个不同类型的注意力模式
_RawColumnPreviousRowAttention = ["block_attn", "transpose_block_attn", "prev_block_attn"]
_FullDenseAttention = ["dense_attention"]
_PrimePrimeDenseAttention = ["prime_attn", "prime_attn", "dense_attn"]

# 返回完全密集的注意力模式
def full_dense_attention(layer):
    return _FullDenseAttention[0]

# 返回原始列和前一行的注意力模式
def raw_column_previous_row_attention(layer):
    return _RawColumnPreviousRowAttention[layer % 3]

# 返回大型分离的编码器-解码器模型与歌词的注意力模式
def large_separated_enc_dec_w_lyrics(layer):
    return _LARGE_ATTENTION[layer % 79]

# 返回带有歌词的编码器-解码器模型的注意力模式
def enc_dec_with_lyrics(layer):
    if layer % 16 == 15:
        return _PrimePrimeDenseAttention[layer % 3]
    return _RawColumnPreviousRowAttention[layer % 3]

# 定义不同注意力模式的映射关系
ATTENTION_PATTERNS = {
    "full_dense_attention": full_dense_attention,
    "raw_column_previous_row_attention": raw_column_previous_row_attention,  # Alternate row, column and previous row attn
    "large_separated_enc_dec_w_lyrics": large_separated_enc_dec_w_lyrics,  # Used by large separated_enc_dec model with lyrics
    "enc_dec_with_lyrics": enc_dec_with_lyrics,  # Used by encoder_decoder model with lyrics
}

# 定义JukeboxPriorConfig类，用于存储JukeboxPrior模型的配置信息
class JukeboxPriorConfig(PretrainedConfig):
    """
        This is the configuration class to store the configuration of a [`JukeboxPrior`]. It is used to instantiate a
        `JukeboxPrior` according to the specified arguments, defining the model architecture. Instantiating a
        configuration with the defaults will yield a similar configuration to that of the top level prior from the
        [openai/jukebox-1b-lyrics](https://huggingface.co/openai/jukebox-1b-lyrics) architecture.

        Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
        documentation from [`PretrainedConfig`] for more information.
    """

    model_type = "jukebox_prior"
    attribute_map = {
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
    }
    # 初始化函数，设置模型的各种参数
    def __init__(
        self,
        act_fn="quick_gelu",  # 激活函数，默认为 quick_gelu
        level=0,  # 模型级别，默认为 0
        alignment_head=2,  # 对齐头部，默认为 2
        alignment_layer=68,  # 对齐层，默认为 68
        attention_multiplier=0.25,  # 注意力乘数，默认为 0.25
        attention_pattern="enc_dec_with_lyrics",  # 注意力模式，默认为 enc_dec_with_lyrics
        attn_dropout=0,  # 注意力丢弃率，默认为 0
        attn_res_scale=False,  # 注意力残差缩放，默认为 False
        blocks=64,  # 块数，默认为 64
        conv_res_scale=None,  # 卷积残差缩放，默认为 None
        num_layers=72,  # 层数，默认为 72
        emb_dropout=0,  # 嵌入丢弃率，默认为 0
        encoder_config=None,  # 编码器配置，默认为 None
        encoder_loss_fraction=0.4,  # 编码器损失分数，默认为 0.4
        hidden_size=2048,  # 隐藏层大小，默认为 2048
        init_scale=0.2,  # 初始化缩放，默认为 0.2
        is_encoder_decoder=True,  # 是否为编码器-解码器模型，默认为 True
        lyric_vocab_size=80,  # 歌词词汇量大小，默认为 80
        mask=False,  # 是否掩码，默认为 False
        max_duration=600,  # 最大持续时间，默认为 600
        max_nb_genres=1,  # 最大音乐类型数，默认为 1
        merged_decoder=True,  # 合并解码器，默认为 True
        metadata_conditioning=True,  # 元数据条件，默认为 True
        metadata_dims=[604, 7898],  # 元数据维度，默认为 [604, 7898]
        min_duration=0,  # 最小持续时间，默认为 0
        mlp_multiplier=1.0,  # 多层感知机乘数，默认为 1.0
        music_vocab_size=2048,  # 音乐词汇量大小，默认为 2048
        n_ctx=6144,  # 上下文大小，默认为 6144
        n_heads=2,  # 头数，默认为 2
        nb_relevant_lyric_tokens=384,  # 相关歌词标记数，默认为 384
        res_conv_depth=3,  # 残差卷积深度，默认为 3
        res_conv_width=128,  # 残差卷积宽度，默认为 128
        res_convolution_multiplier=1,  # 残差卷积乘数，默认为 1
        res_dilation_cycle=None,  # 残差扩张周期，默认为 None
        res_dilation_growth_rate=1,  # 残差扩张增长率，默认为 1
        res_downs_t=[3, 2, 2],  # 残差下采样 t，默认为 [3, 2, 2]
        res_strides_t=[2, 2, 2],  # 残差步长 t，默认为 [2, 2, 2]
        resid_dropout=0,  # 残差丢弃率，默认为 0
        sampling_rate=44100,  # 采样率，默认为 44100
        spread=None,  # 传播，默认为 None
        timing_dims=64,  # 时间维度，默认为 64
        zero_out=False,  # 是否清零，默认为 False
        **kwargs,  # 其他关键字参数
        self.act_fn = act_fn
        # 激活函数
        self.alignment_head = alignment_head
        # 对齐头
        self.alignment_layer = alignment_layer
        # 对齐层
        self.attention_multiplier = attention_multiplier
        # 注意力乘数
        self.attention_pattern = attention_pattern
        # 注意力模式
        self.attn_dropout = attn_dropout
        # 注意力丢弃
        self.attn_res_scale = attn_res_scale
        # 注意力残差比例
        self.blocks = blocks
        # 块数
        self.conv_res_scale = conv_res_scale
        # 卷积残差比例
        self.num_layers = num_layers
        # 层数
        self.emb_dropout = emb_dropout
        # 嵌入丢弃
        self.music_vocab_size = music_vocab_size
        # 音乐词汇量大小
        if encoder_config is not None:
            self.encoder_config = JukeboxPriorConfig(**encoder_config)
        else:
            self.encoder_config = None
        # 编码器配置
        self.encoder_loss_fraction = encoder_loss_fraction
        # 编码器损失分数
        self.init_scale = init_scale
        # 初始化比例
        self.is_encoder_decoder = is_encoder_decoder
        # 是否为编码器解码器
        self.lyric_vocab_size = lyric_vocab_size
        # 歌词词汇量大小
        self.level = level
        # 等级
        self.mask = mask
        # 掩码
        self.max_duration = max_duration
        # 最大持续时间
        self.max_nb_genres = max_nb_genres
        # 最大流派数
        self.merged_decoder = merged_decoder
        # 合并解码器
        self.metadata_conditioning = metadata_conditioning
        # 元数据调节
        self.metadata_dims = metadata_dims
        # 元数据维度
        self.min_duration = min_duration
        # 最小持续时间
        self.mlp_multiplier = mlp_multiplier
        # 多层感知机乘数
        self.n_ctx = n_ctx
        # 上下文数
        self.n_heads = n_heads
        # 头数
        self.nb_relevant_lyric_tokens = nb_relevant_lyric_tokens
        # 相关歌词标记数
        self.res_conv_depth = res_conv_depth
        # 卷积深度
        self.res_conv_width = res_conv_width
        # 卷积宽度
        self.res_convolution_multiplier = res_convolution_multiplier
        # 卷积乘数
        self.res_dilation_cycle = res_dilation_cycle
        # 膨胀周期
        self.res_dilation_growth_rate = res_dilation_growth_rate
        # 膨胀增长率
        self.res_downs_t = res_downs_t
        # 下采样
        self.res_strides_t = res_strides_t
        # 步幅
        self.resid_dropout = resid_dropout
        # 残差丢弃
        self.sampling_rate = sampling_rate
        # 采样率
        self.spread = spread
        # 传播
        self.timing_dims = timing_dims
        # 时间维度
        self.hidden_size = hidden_size
        # 隐藏层大小
        self.zero_out = zero_out
        # 零输出

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], level=0, **kwargs
    ) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果从 JukeboxConfig 加载，则获取先前配置字典
        if config_dict.get("model_type") == "jukebox":
            config_dict = config_dict[f"prior_{level}"]

        # 如果配置字典中存在 "model_type" 并且类中有 "model_type" 属性，且二者不相等，则发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 从配置字典和参数中创建实例
        return cls.from_dict(config_dict, **kwargs)
class JukeboxVQVAEConfig(PretrainedConfig):
    """
    这是用于存储 [`JukeboxVQVAE`] 配置的类。它用于根据指定的参数实例化一个 `JukeboxVQVAE`，定义模型架构。使用默认值实例化配置将产生类似于来自 [openai/jukebox-1b-lyrics](https://huggingface.co/openai/jukebox-1b-lyrics) 架构的 VQVAE 的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。
    """
    Args:
        act_fn (`str`, *optional*, defaults to `"relu"`):
            # 模型的激活函数
            Activation function of the model.
        nb_discrete_codes (`int`, *optional*, defaults to 2048):
            # VQVAE 的离散码数量
            Number of codes of the VQVAE.
        commit (`float`, *optional*, defaults to 0.02):
            # Commit loss 的乘数
            Commit loss multiplier.
        conv_input_shape (`int`, *optional*, defaults to 1):
            # 音频通道数
            Number of audio channels.
        conv_res_scale (`bool`, *optional*, defaults to `False`):
            # 是否对 `JukeboxResConv1DBlock` 的残差进行缩放
            Whether or not to scale the residuals of the `JukeboxResConv1DBlock`.
        embed_dim (`int`, *optional*, defaults to 64):
            # 代码本向量的嵌入维度
            Embedding dimension of the codebook vectors.
        hop_fraction (`List[int]`, *optional*, defaults to `[0.125, 0.5, 0.5]`):
            # 在继续采样过程时使用的非交叉窗口的比例
            Fraction of non-intersecting window used when continuing the sampling process.
        levels (`int`, *optional*, defaults to 3):
            # VQVAE 中使用的层级数
            Number of hierarchical levels that used in the VQVAE.
        lmu (`float`, *optional*, defaults to 0.99):
            # 用于代码本更新的指数移动平均系数
            Used in the codebook update, exponential moving average coefficient. For more detail refer to Appendix A.1
            of the original [VQVAE paper](https://arxiv.org/pdf/1711.00937v2.pdf)
        multipliers (`List[int]`, *optional*, defaults to `[2, 1, 1]`):
            # 每个层级使用的深度和宽度乘数
            Depth and width multipliers used for each level. Used on the `res_conv_width` and `res_conv_depth`
        res_conv_depth (`int`, *optional*, defaults to 4):
            # 编码器和解码器块的深度
            Depth of the encoder and decoder block. If no `multipliers` are used, this is the same for each level.
        res_conv_width (`int`, *optional*, defaults to 32):
            # 编码器和解码器块的宽度
            Width of the encoder and decoder block. If no `multipliers` are used, this is the same for each level.
        res_convolution_multiplier (`int`, *optional*, defaults to 1):
            # `JukeboxResConv1DBlock` 中隐藏维度的缩放因子
            Scaling factor of the hidden dimension used in the `JukeboxResConv1DBlock`.
        res_dilation_cycle (`int`, *optional*):
            # `JukeboxResnet` 中使用的扩张周期值
            Dilation cycle value used in the `JukeboxResnet`. If an int is used, each new Conv1 block will have a depth
            reduced by a power of `res_dilation_cycle`.
        res_dilation_growth_rate (`int`, *optional*, defaults to 3):
            # VQVAE 中使用的 Resnet 扩张增长率
            Resnet dilation growth rate used in the VQVAE (dilation_growth_rate ** depth)
        res_downs_t (`List[int]`, *optional*, defaults to `[3, 2, 2]`):
            # 分层 VQ-VAE 的每个层级的下采样率
            Downsampling rate for each level of the hierarchical VQ-VAE.
        res_strides_t (`List[int]`, *optional*, defaults to `[2, 2, 2]`):
            # 分层 VQ-VAE 的每个层级使用的步幅
            Stride used for each level of the hierarchical VQ-VAE.
        sample_length (`int`, *optional*, defaults to 1058304):
            # VQVAE 的最大输入形状
            Provides the max input shape of the VQVAE. Is used to compute the input shape of each level.
        init_scale (`float`, *optional*, defaults to 0.2):
            # 初始化比例
            Initialization scale.
        zero_out (`bool`, *optional*, defaults to `False`):
            # 初始化时是否将卷积权重归零
            Whether or not to zero out convolution weights when initializing.
    """
    # 设置模型类型为"jukebox_vqvae"
    model_type = "jukebox_vqvae"

    # 初始化函数，接受多个参数
    def __init__(
        self,
        act_fn="relu",  # 激活函数，默认为"relu"
        nb_discrete_codes=2048,  # 离散码的数量，默认为2048
        commit=0.02,  # commit参数，默认为0.02
        conv_input_shape=1,  # 卷积输入形状，默认为1
        conv_res_scale=False,  # 卷积残差缩放，默认为False
        embed_dim=64,  # 嵌入维度，默认为64
        hop_fraction=[0.125, 0.5, 0.5],  # 跳跃分数列表，默认为[0.125, 0.5, 0.5]
        levels=3,  # 等级数，默认为3
        lmu=0.99,  # lmu参数，默认为0.99
        multipliers=[2, 1, 1],  # 乘数列表，默认为[2, 1, 1]
        res_conv_depth=4,  # 残差卷积深度，默认为4
        res_conv_width=32,  # 残差卷积宽度，默认为32
        res_convolution_multiplier=1,  # 残差卷积乘数，默认为1
        res_dilation_cycle=None,  # 残差扩张周期，默认为None
        res_dilation_growth_rate=3,  # 残差扩张增长率，默认为3
        res_downs_t=[3, 2, 2],  # 残差下采样列表，默认为[3, 2, 2]
        res_strides_t=[2, 2, 2],  # 残差步长列表，默认为[2, 2, 2]
        sample_length=1058304,  # 样本长度，默认为1058304
        init_scale=0.2,  # 初始化规模，默认为0.2
        zero_out=False,  # 是否置零，默认为False
        **kwargs,  # 其他关键字参数
    ):
        # 将参数赋值给对象属性
        self.hop_fraction = hop_fraction
        self.conv_input_shape = conv_input_shape
        self.sample_length = sample_length

        # VQVAE参数（全部使用）
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

    # 类方法，从预训练模型中加载配置
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 设置token在kwargs中
        cls._set_token_in_kwargs(kwargs)

        # 获取配置字典和kwargs
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果从CLIPConfig加载，则获取文本配置字典
        if config_dict.get("model_type") == "jukebox":
            config_dict = config_dict["vqvae_config"]

        # 如果配置字典中存在"model_type"并且类中有"model_type"属性，并且它们不相等，则发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 从字典中创建实例并返回
        return cls.from_dict(config_dict, **kwargs)
# 定义 JukeboxConfig 类，用于存储 JukeboxModel 的配置信息
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
    ```py
    """

    # 模型类型为 "jukebox"
    model_type = "jukebox"
    def __init__(
        self,
        vqvae_config=None,  # 初始化方法，接受一个 VQVAE 配置和一个列表 of prior 配置等参数
        prior_config_list=None,
        nb_priors=3,
        sampling_rate=44100,
        timing_dims=64,
        min_duration=0,
        max_duration=600.0,
        max_nb_genres=5,
        metadata_conditioning=True,
        **kwargs,  # 接受额外的关键字参数
    ):
        if vqvae_config is None:
            vqvae_config = {}  # 如果没有传入 vqvae_config，则初始化为空字典
            logger.info("vqvae_config is None. initializing the JukeboxVQVAE with default values.")

        self.vqvae_config = JukeboxVQVAEConfig(**vqvae_config)  # 使用传入的 vqvae_config 创建 JukeboxVQVAEConfig 对象
        if prior_config_list is not None:
            self.prior_configs = [JukeboxPriorConfig(**prior_config) for prior_config in prior_config_list]
        else:
            self.prior_configs = []  # 如果 prior_config_list 为空，则初始化为空列表
            for prior_idx in range(nb_priors):
                prior_config = kwargs.pop(f"prior_{prior_idx}", None)  # 从 kwargs 中弹出一个 prior 配置
                if prior_config is None:
                    prior_config = {}  # 如果配置不存在，则使用默认值
                    logger.info(
                        f"prior_{prior_idx}'s  config is None. Initializing the JukeboxPriorConfig list with default"
                        " values."
                    )
                self.prior_configs.append(JukeboxPriorConfig(**prior_config))  # 使用配置创建 JukeboxPriorConfig 对象

        self.hop_fraction = self.vqvae_config.hop_fraction

        self.nb_priors = nb_priors

        # Metadata conditioning
        self.max_nb_genres = max_nb_genres
        self.sampling_rate = sampling_rate
        self.timing_dims = timing_dims
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.metadata_conditioning = metadata_conditioning

        super().__init__(**kwargs)  # 调用父类的初始化方法，传入额外的关键字参数

    @classmethod
    def from_configs(cls, prior_configs: List[JukeboxPriorConfig], vqvae_config: JukeboxVQVAEConfig, **kwargs):
        r"""
        Instantiate a [`JukeboxConfig`] (or a derived class) from clip text model configuration and clip vision model
        configuration.

        Returns:
            [`JukeboxConfig`]: An instance of a configuration object
        """
        prior_config_list = [config.to_dict() for config in prior_configs]  # 转换 prior 配置为字典列表
        return cls(prior_config_list=prior_config_list, vqvae_config_dict=vqvae_config.to_dict(), **kwargs)  # 根据配置实例化对象

    def to_dict(self):
        # Override the default to_dict to apply to_dict to the list of prior configs.
        result = super().to_dict()  # 调用父类的 to_dict 方法获取对象的字典表示
        result["prior_config_list"] = [config.to_dict() for config in result.pop("prior_configs")]  # 转换 prior 配置为字典列表
        return result  # 返回对象的字典表示
```