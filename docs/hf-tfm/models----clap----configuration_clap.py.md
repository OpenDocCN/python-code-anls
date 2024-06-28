# `.\models\clap\configuration_clap.py`

```
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" CLAP model configuration"""

# 导入必要的模块
import os
from typing import Union

# 导入配置工具类和日志记录工具
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取logger对象用于日志记录
logger = logging.get_logger(__name__)

# 预训练模型的配置信息，包含模型名称和对应的配置文件URL
CLAP_PRETRAINED_MODEL_ARCHIVE_LIST = {
    "laion/clap-htsat-fused": "https://huggingface.co/laion/clap-htsat-fused/resolve/main/config.json",
    "laion/clap-htsat-unfused": "https://huggingface.co/laion/clap-htsat-unfused/resolve/main/config.json",
}

# CLAP模型的配置类，继承自PretrainedConfig
class ClapTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ClapTextModel`]. It is used to instantiate a CLAP
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the CLAP
    [calp-hsat-fused](https://huggingface.co/laion/clap-hsat-fused) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Examples:

    ```python
    >>> from transformers import ClapTextConfig, ClapTextModel

    >>> # Initializing a CLAP text configuration
    >>> configuration = ClapTextConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = ClapTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    # 模型类型的字符串标识
    model_type = "clap_text_model"

    # 初始化方法，定义了模型的各种参数配置
    def __init__(
        self,
        vocab_size=50265,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=514,
        type_vocab_size=1,
        initializer_factor=1.0,
        layer_norm_eps=1e-12,
        projection_dim=512,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        position_embedding_type="absolute",
        use_cache=True,
        projection_hidden_act="relu",
        **kwargs,
    ):
        # 调用父类的初始化方法，设置特殊的令牌 ID 和其他关键字参数
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        # 设置配置类的属性
        self.vocab_size = vocab_size  # 词汇表大小
        self.hidden_size = hidden_size  # 隐藏层大小
        self.num_hidden_layers = num_hidden_layers  # 隐藏层数量
        self.num_attention_heads = num_attention_heads  # 注意力头的数量
        self.hidden_act = hidden_act  # 隐藏层激活函数类型
        self.intermediate_size = intermediate_size  # 中间层大小
        self.hidden_dropout_prob = hidden_dropout_prob  # 隐藏层的dropout概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob  # 注意力概率dropout概率
        self.max_position_embeddings = max_position_embeddings  # 最大位置嵌入长度
        self.type_vocab_size = type_vocab_size  # 类型词汇表大小
        self.initializer_factor = initializer_factor  # 初始化因子
        self.layer_norm_eps = layer_norm_eps  # 层归一化 epsilon 值
        self.position_embedding_type = position_embedding_type  # 位置嵌入类型
        self.use_cache = use_cache  # 是否使用缓存
        self.projection_hidden_act = projection_hidden_act  # 投影隐藏层激活函数类型
        self.projection_dim = projection_dim  # 投影维度

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 在 kwargs 中设置令牌相关参数
        cls._set_token_in_kwargs(kwargs)

        # 获取配置字典和可能更新后的 kwargs
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果从 ClapConfig 加载，获取文本配置字典
        if config_dict.get("model_type") == "clap":
            config_dict = config_dict["text_config"]

        # 检查模型类型是否与当前类匹配，如果不匹配则发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 根据配置字典和 kwargs 创建配置实例
        return cls.from_dict(config_dict, **kwargs)
# 定义一个配置类 `ClapAudioConfig`，用于存储 `ClapAudioModel` 的配置信息。
# 继承自 `PretrainedConfig` 类，可以用来控制模型的输出。
class ClapAudioConfig(PretrainedConfig):
    # 模型类型标识为 "clap_audio_model"
    model_type = "clap_audio_model"

    # 初始化方法，用于设置配置类的各项参数
    def __init__(
        self,
        window_size=8,  # 滑动窗口大小，默认为 8
        num_mel_bins=64,  # Mel 频谱的 bin 数量，默认为 64
        spec_size=256,  # 音频谱图的尺寸，默认为 256
        hidden_act="gelu",  # 隐藏层激活函数，默认为 "gelu"
        patch_size=4,  # Patch 的大小，默认为 4
        patch_stride=[4, 4],  # Patch 的步幅，默认为 [4, 4]
        num_classes=527,  # 分类的类别数量，默认为 527
        hidden_size=768,  # 隐藏层大小，默认为 768
        projection_dim=512,  # 投影维度，默认为 512
        depths=[2, 2, 6, 2],  # 不同 Transformer 层的堆叠深度，默认为 [2, 2, 6, 2]
        num_attention_heads=[4, 8, 16, 32],  # 不同 Transformer 层的注意力头数，默认为 [4, 8, 16, 32]
        enable_fusion=False,  # 是否启用融合，默认为 False
        hidden_dropout_prob=0.1,  # 隐藏层的 dropout 概率，默认为 0.1
        fusion_type=None,  # 融合类型，默认为 None
        patch_embed_input_channels=1,  # Patch 嵌入的输入通道数，默认为 1
        flatten_patch_embeds=True,  # 是否展平 Patch 嵌入，默认为 True
        patch_embeds_hidden_size=96,  # Patch 嵌入的隐藏层大小，默认为 96
        enable_patch_layer_norm=True,  # 是否启用 Patch 层归一化，默认为 True
        drop_path_rate=0.0,  # DropPath 的比率，默认为 0.0
        attention_probs_dropout_prob=0.0,  # 注意力矩阵的 dropout 概率，默认为 0.0
        qkv_bias=True,  # 是否使用 QKV 的偏置，默认为 True
        mlp_ratio=4.0,  # MLP 层中隐藏层和输入层的维度比率，默认为 4.0
        aff_block_r=4,  # 仿射块的参数 r，默认为 4
        num_hidden_layers=4,  # 隐藏层的数量，默认为 4
        projection_hidden_act="relu",  # 投影层的激活函数，默认为 "relu"
        layer_norm_eps=1e-5,  # LayerNorm 的 epsilon，默认为 1e-5
        initializer_factor=1.0,  # 初始化因子，默认为 1.0
        **kwargs,  # 其余未命名的参数
    ):
        # 调用父类的初始化方法，传入所有关键字参数
        super().__init__(**kwargs)
        # 设置模型的窗口大小
        self.window_size = window_size
        # 设置梅尔频谱的频道数量
        self.num_mel_bins = num_mel_bins
        # 设置规范化后的频谱大小
        self.spec_size = spec_size
        # 设置每个补丁的大小
        self.patch_size = patch_size
        # 设置补丁的步长
        self.patch_stride = patch_stride
        # 设置类别数量
        self.num_classes = num_classes
        # 设置隐藏层的大小
        self.hidden_size = hidden_size
        # 设置层级列表
        self.depths = depths
        # 设置隐藏层数量
        self.num_hidden_layers = num_hidden_layers
        # 设置注意力头的数量
        self.num_attention_heads = num_attention_heads
        # 重新设置窗口大小（这里可能是冗余的，因为之前已经设置过）
        self.window_size = window_size
        # 启用融合
        self.enable_fusion = enable_fusion
        # 设置融合类型
        self.fusion_type = fusion_type
        # 设置隐藏层激活函数
        self.hidden_act = hidden_act
        # 设置隐藏层的dropout概率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 设置投影维度
        self.projection_dim = projection_dim
        # 是否展平补丁嵌入
        self.flatten_patch_embeds = flatten_patch_embeds
        # 补丁嵌入的隐藏层大小
        self.patch_embeds_hidden_size = patch_embeds_hidden_size
        # 是否启用补丁层的规范化
        self.enable_patch_layer_norm = enable_patch_layer_norm
        # 设置丢弃路径的率
        self.drop_path_rate = drop_path_rate
        # 注意力概率的dropout概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 是否在QKV上使用偏置
        self.qkv_bias = qkv_bias
        # 多层感知机的比率
        self.mlp_ratio = mlp_ratio
        # 补丁嵌入的输入通道数
        self.patch_embed_input_channels = patch_embed_input_channels
        # AffineBlock的半径
        self.aff_block_r = aff_block_r
        # 层归一化的epsilon值
        self.layer_norm_eps = layer_norm_eps
        # 初始化因子
        self.initializer_factor = initializer_factor
        # 投影的隐藏层激活函数
        self.projection_hidden_act = projection_hidden_act

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 设置kwargs中的token
        cls._set_token_in_kwargs(kwargs)

        # 获取预训练模型的配置字典和剩余的kwargs
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果配置字典中的模型类型为"clap"，则从中获取音频配置字典
        if config_dict.get("model_type") == "clap":
            config_dict = config_dict["audio_config"]

        # 如果配置字典中包含模型类型，并且类本身有model_type属性，并且配置的模型类型不是类本身的模型类型，则发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 根据配置字典和kwargs创建配置对象
        return cls.from_dict(config_dict, **kwargs)
# `ClapConfig` 类，继承自 `PretrainedConfig`，用于存储 `ClapModel` 的配置信息。
# 该类用于实例化一个 CLAP 模型，根据指定的参数定义文本模型和音频模型的配置。
class ClapConfig(PretrainedConfig):
    r"""
    [`ClapConfig`] is the configuration class to store the configuration of a [`ClapModel`]. It is used to instantiate
    a CLAP model according to the specified arguments, defining the text model and audio model configs. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the CLAP
    [laion/clap-htsat-fused](https://huggingface.co/laion/clap-htsat-fused) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`ClapTextConfig`].
        audio_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`ClapAudioConfig`].
        logit_scale_init_value (`float`, *optional*, defaults to 14.29):
            The inital value of the *logit_scale* paramter. Default is used as per the original CLAP implementation.
        projection_dim (`int`, *optional*, defaults to 512):
            Dimentionality of text and audio projection layers.
        projection_hidden_act (`str`, *optional*, defaults to `"relu"`):
            Activation function for the projection layers.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            Factor to scale the initialization of the model weights.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import ClapConfig, ClapModel

    >>> # Initializing a ClapConfig with laion-ai/base style configuration
    >>> configuration = ClapConfig()

    >>> # Initializing a ClapModel (with random weights) from the laion-ai/base style configuration
    >>> model = ClapModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a ClapConfig from a ClapTextConfig and a ClapAudioConfig
    >>> from transformers import ClapTextConfig, ClapAudioConfig

    >>> # Initializing a ClapText and ClapAudioConfig configuration
    >>> config_text = ClapTextConfig()
    >>> config_audio = ClapAudioConfig()

    >>> config = ClapConfig.from_text_audio_configs(config_text, config_audio)
    ```"""

    # 类属性 `model_type`，指定为 "clap"，用于标识模型类型
    model_type = "clap"

    # 初始化方法，用于创建 `ClapConfig` 的实例对象
    def __init__(
        self,
        text_config=None,  # 文本配置的字典，用于初始化 `ClapTextConfig`
        audio_config=None,  # 音频配置的字典，用于初始化 `ClapAudioConfig`
        logit_scale_init_value=(1 / 0.07),  # `logit_scale` 参数的初始值，默认为 CLAP 实现的原始值
        projection_dim=512,  # 文本和音频投影层的维度
        projection_hidden_act="relu",  # 投影层的激活函数，默认为 ReLU
        initializer_factor=1.0,  # 模型权重初始化的缩放因子，默认为 1.0
        **kwargs,  # 其他可选的关键字参数
    ):
    ):
        super().__init__(**kwargs)
        
        # 如果 text_config 参数为 None，则初始化为空字典，并记录日志信息
        if text_config is None:
            text_config = {}
            logger.info("text_config is None. Initializing the ClapTextConfig with default values.")
        
        # 如果 audio_config 参数为 None，则初始化为空字典，并记录日志信息
        if audio_config is None:
            audio_config = {}
            logger.info("audio_config is None. initializing the ClapAudioConfig with default values.")
        
        # 使用给定的 text_config 和 audio_config 创建 ClapTextConfig 和 ClapAudioConfig 实例
        self.text_config = ClapTextConfig(**text_config)
        self.audio_config = ClapAudioConfig(**audio_config)
        
        # 设置投影维度（projection_dim）到 text_config 和 audio_config 的实例中
        self.text_config.projection_dim = projection_dim
        self.audio_config.projection_dim = projection_dim
        
        # 设置投影隐藏层激活函数（projection_hidden_act）到 text_config 和 audio_config 的实例中
        self.text_config.projection_hidden_act = projection_hidden_act
        self.audio_config.projection_hidden_act = projection_hidden_act
        
        # 设置对象自身的投影维度和投影隐藏层激活函数
        self.projection_dim = projection_dim
        self.projection_hidden_act = projection_hidden_act
        
        # 设置隐藏层大小（hidden_size）为 text_config 的隐藏层大小
        self.hidden_size = self.text_config.hidden_size
        
        # 设置 logit_scale_init_value 和 initializer_factor
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = initializer_factor
        
        # 计算总的隐藏层数，由 text_config 的隐藏层数和 audio_config 的深度之和得到
        self.num_hidden_layers = self.text_config.num_hidden_layers + len(self.audio_config.depths)

    @classmethod
    def from_text_audio_configs(cls, text_config: ClapTextConfig, audio_config: ClapAudioConfig, **kwargs):
        r"""
        Instantiate a [`ClapConfig`] (or a derived class) from clap text model configuration and clap audio model
        configuration.

        Returns:
            [`ClapConfig`]: An instance of a configuration object
        """

        # 从给定的 text_config 和 audio_config 创建一个 ClapConfig 类的实例，并返回
        return cls(text_config=text_config.to_dict(), audio_config=audio_config.to_dict(), **kwargs)
```