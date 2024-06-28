# `.\models\pix2struct\configuration_pix2struct.py`

```
# 设置代码文件的编码格式为 UTF-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有，保留所有权利
#
# 根据 Apache 许可证版本 2.0 进行许可
# 除非符合许可证规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件是基于“按原样”分发的
# 没有任何明示或暗示的担保或条件，包括但不限于
# 适销性、特定用途的适用性或非侵权性的保证
# 有关更多信息，请参阅许可证

""" Pix2Struct 模型配置 """

# 导入操作系统模块
import os
# 导入 Union 类型
from typing import Union

# 导入预训练配置类 PretrainedConfig
from ...configuration_utils import PretrainedConfig
# 导入日志工具
from ...utils import logging

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# Pix2Struct 预训练配置存档映射表
PIX2STRUCT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/pix2struct-textcaps-base": (
        "https://huggingface.co/google/pix2struct-textcaps-base/resolve/main/config.json"
    ),
}

# Pix2StructTextConfig 类，继承自 PretrainedConfig 类
class Pix2StructTextConfig(PretrainedConfig):
    r"""
    这是用于存储 [`Pix2StructTextModel`] 配置的配置类。它用于根据指定的参数实例化
    Pix2Struct 文本模型，定义模型架构。使用默认值实例化配置将产生类似于
    [google/pix2struct-base](https://huggingface.co/google/pix2struct-base) 架构使用的 Pix2Struct 文本解码器的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。有关更多信息，请阅读
    [`PretrainedConfig`] 的文档。
    # 定义模型类型为 "pix2struct_text_model"
    model_type = "pix2struct_text_model"
    python
        # 在推断时要忽略的键列表
        keys_to_ignore_at_inference = ["past_key_values"]
        # 属性映射字典，将类参数名映射到配置文件中的属性名
        attribute_map = {
            "hidden_size": "hidden_size",
            "num_attention_heads": "num_heads",
            "num_hidden_layers": "num_layers",
        }
    
        # 类的初始化方法，定义了模型配置的默认参数
        def __init__(
            self,
            vocab_size=50244,
            hidden_size=768,
            d_kv=64,
            d_ff=2048,
            num_layers=12,
            num_heads=12,
            relative_attention_num_buckets=32,
            relative_attention_max_distance=128,
            dropout_rate=0.1,
            layer_norm_epsilon=1e-6,
            initializer_factor=1.0,
            dense_act_fn="gelu_new",
            decoder_start_token_id=0,
            use_cache=False,
            pad_token_id=0,
            eos_token_id=1,
            tie_word_embeddings=False,
            is_decoder=True,
            **kwargs,
        ):
            # 初始化类的各个参数
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.d_kv = d_kv
            self.d_ff = d_ff
            self.num_layers = num_layers
            self.num_heads = num_heads
            self.relative_attention_num_buckets = relative_attention_num_buckets
            self.relative_attention_max_distance = relative_attention_max_distance
            self.dropout_rate = dropout_rate
            self.layer_norm_epsilon = layer_norm_epsilon
            self.initializer_factor = initializer_factor
            self.use_cache = use_cache
    
            self.eos_token_id = eos_token_id
            self.decoder_start_token_id = decoder_start_token_id
    
            # 为了向后兼容，设置密集层激活函数
            self.dense_act_fn = dense_act_fn
    
            # 调用父类的初始化方法，传入参数
            super().__init__(
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                decoder_start_token_id=decoder_start_token_id,
                tie_word_embeddings=tie_word_embeddings,
                is_decoder=is_decoder,
                **kwargs,
            )
    
        # 类方法，从预训练模型加载配置
        @classmethod
        def from_pretrained(
            cls, pretrainehidden_size_name_or_path: Union[str, os.PathLike], **kwargs
        ) -> "PretrainedConfig":
            # 设置 token 的参数到 kwargs 中
            cls._set_token_in_kwargs(kwargs)
    
            # 获取预训练模型的配置字典和额外的 kwargs
            config_dict, kwargs = cls.get_config_dict(pretrainehidden_size_name_or_path, **kwargs)
    
            # 如果配置字典中的模型类型是 "pix2struct"，则获取其中的文本配置字典
            if config_dict.get("model_type") == "pix2struct":
                config_dict = config_dict["text_config"]
    
            # 如果配置字典中指定的模型类型与类中定义的模型类型不匹配，发出警告
            if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
                logger.warning(
                    f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                    f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
                )
    
            # 从配置字典和 kwargs 创建并返回一个类实例
            return cls.from_dict(config_dict, **kwargs)
# Pix2StructVisionConfig 类，继承自 PretrainedConfig 类
class Pix2StructVisionConfig(PretrainedConfig):
    r"""
    这是一个配置类，用于存储 [`Pix2StructVisionModel`] 的配置。它被用来实例化一个 Pix2Struct 视觉模型，根据指定的参数定义模型架构。
    默认情况下实例化一个配置将产生类似于 Pix2Struct-base [google/pix2struct-base](https://huggingface.co/google/pix2struct-base) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可以用来控制模型的输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。
    ```
    # 模型类型字符串，指定为"pix2struct_vision_model"
    model_type = "pix2struct_vision_model"
    # 初始化函数，设置 Transformer 模型的各种参数
    def __init__(
        self,
        hidden_size=768,  # 隐藏层大小，默认为768
        patch_embed_hidden_size=768,  # 补丁嵌入的隐藏层大小，默认为768
        d_ff=2048,  # Feedforward 层的大小，默认为2048
        d_kv=64,  # 键值映射的维度，默认为64
        num_hidden_layers=12,  # Transformer 模型的隐藏层数，默认为12
        num_attention_heads=12,  # 注意力头的数量，默认为12
        dense_act_fn="gelu_new",  # 密集层激活函数，默认为"gelu_new"
        layer_norm_eps=1e-6,  # 层归一化的 epsilon 值，默认为1e-6
        dropout_rate=0.0,  # 总体的 dropout 率，默认为0.0
        attention_dropout=0.0,  # 注意力层的 dropout 率，默认为0.0
        initializer_range=1e-10,  # 参数初始化的范围，默认为1e-10
        initializer_factor=1.0,  # 参数初始化的因子，默认为1.0
        seq_len=4096,  # 序列的长度，默认为4096
        relative_attention_num_buckets=32,  # 相对注意力的桶数量，默认为32
        relative_attention_max_distance=128,  # 相对注意力的最大距离，默认为128
        **kwargs,  # 其他参数，以字典形式接收
    ):
        # 调用父类的初始化方法，传递其他参数
        super().__init__(**kwargs)

        # 初始化 Transformer 模型的各种参数
        self.hidden_size = hidden_size
        self.patch_embed_hidden_size = patch_embed_hidden_size
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.dense_act_fn = dense_act_fn
        self.seq_len = seq_len
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.d_kv = d_kv

    @classmethod
    # 从预训练模型中加载配置信息，并返回一个预训练配置对象
    def from_pretrained(
        cls, pretrainehidden_size_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> "PretrainedConfig":
        # 设置 kwargs 中的 token 相关参数
        cls._set_token_in_kwargs(kwargs)

        # 获取配置字典和更新后的 kwargs
        config_dict, kwargs = cls.get_config_dict(pretrainehidden_size_name_or_path, **kwargs)

        # 如果加载的模型类型为 "pix2struct"，则使用视觉配置字典
        if config_dict.get("model_type") == "pix2struct":
            config_dict = config_dict["vision_config"]

        # 如果配置字典中包含模型类型，并且该类型与当前类的模型类型不同，发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 根据配置字典和 kwargs 创建一个新的类实例
        return cls.from_dict(config_dict, **kwargs)
class Pix2StructConfig(PretrainedConfig):
    r"""
    [`Pix2StructConfig`] is the configuration class to store the configuration of a
    [`Pix2StructForConditionalGeneration`]. It is used to instantiate a Pix2Struct model according to the specified
    arguments, defining the text model and vision model configs. Instantiating a configuration with the defaults will
    yield a similar configuration to that of the Pix2Struct-base
    [google/pix2struct-base](https://huggingface.co/google/pix2struct-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`Pix2StructTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`Pix2StructVisionConfig`].
        initializer_factor (`float`, *optional*, defaults to 1.0):
            Factor to multiply the initialization range with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        is_vqa (`bool`, *optional*, defaults to `False`):
            Whether the model has been fine-tuned for VQA or not.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie the word embeddings between the text and vision models.
        is_encoder_decoder (`bool`, *optional*, defaults to `True`):
            Whether the model follows an encoder-decoder architecture.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import Pix2StructConfig, Pix2StructForConditionalGeneration

    >>> # Initializing a Pix2StructConfig with google/pix2struct-base style configuration
    >>> configuration = Pix2StructConfig()

    >>> # Initializing a Pix2StructForConditionalGeneration (with random weights) from the google/pix2struct-base style configuration
    >>> model = Pix2StructForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a Pix2StructConfig from a Pix2StructTextConfig and a Pix2StructVisionConfig

    >>> # Initializing a Pix2Struct text and Pix2Struct vision configuration
    >>> config_text = Pix2StructTextConfig()
    >>> config_vision = Pix2StructVisionConfig()

    >>> config = Pix2StructConfig.from_text_vision_configs(config_text, config_vision)
    ```"""

    # 定义模型类型为 "pix2struct"
    model_type = "pix2struct"

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        initializer_factor=1.0,
        initializer_range=0.02,
        is_vqa=False,
        tie_word_embeddings=False,
        is_encoder_decoder=True,
        **kwargs,
    ):
        # 调用父类的初始化方法，初始化基本配置
        super().__init__(**kwargs)
        # 如果传入了文本配置和视觉配置，则保存在相应属性中
        self.text_config = text_config
        self.vision_config = vision_config
        # 初始化器相关参数
        self.initializer_factor = initializer_factor
        self.initializer_range = initializer_range
        # 是否是 VQA 模型的标志位
        self.is_vqa = is_vqa
        # 是否绑定词嵌入的标志位
        self.tie_word_embeddings = tie_word_embeddings
        # 是否是编码器-解码器结构的标志位
        self.is_encoder_decoder = is_encoder_decoder
    ):
        # 调用父类的初始化方法，传递参数以绑定词嵌入、是否编码解码器等属性
        super().__init__(tie_word_embeddings=tie_word_embeddings, is_encoder_decoder=is_encoder_decoder, **kwargs)

        # 如果文本配置为空，则初始化为空字典，并记录日志
        if text_config is None:
            text_config = {}
            logger.info("text_config is None. Initializing the Pix2StructTextConfig with default values.")

        # 如果视觉配置为空，则初始化为空字典，并记录日志
        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. Initializing the Pix2StructVisionConfig with default values.")

        # 根据给定的文本配置创建 Pix2StructTextConfig 实例
        self.text_config = Pix2StructTextConfig(**text_config)
        # 根据给定的视觉配置创建 Pix2StructVisionConfig 实例
        self.vision_config = Pix2StructVisionConfig(**vision_config)

        # 从文本配置中获取解码器起始标记的 ID
        self.decoder_start_token_id = self.text_config.decoder_start_token_id
        # 从文本配置中获取填充标记的 ID
        self.pad_token_id = self.text_config.pad_token_id
        # 从文本配置中获取结束标记的 ID
        self.eos_token_id = self.text_config.eos_token_id

        # 初始化因子乘数
        self.initializer_factor = initializer_factor
        # 初始化范围值
        self.initializer_range = initializer_range

        # 将初始化范围值分别赋给文本配置和视觉配置
        self.text_config.initializer_range = self.initializer_range
        self.vision_config.initializer_range = self.initializer_range

        # 设定是否是视觉问答模型的标志
        self.is_vqa = is_vqa

    @classmethod
    def from_text_vision_configs(
        cls, text_config: Pix2StructTextConfig, vision_config: Pix2StructVisionConfig, **kwargs
    ):
        r"""
        Instantiate a [`Pix2StructConfig`] (or a derived class) from pix2struct text model configuration and pix2struct
        vision model configuration.

        Returns:
            [`Pix2StructConfig`]: An instance of a configuration object
        """

        # 从文本配置和视觉配置实例创建一个新的 Pix2StructConfig 实例，并返回
        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)
```