# `.\models\mpt\configuration_mpt.py`

```
# coding=utf-8
# Copyright 2023 HuggingFace Inc. team and MosaicML NLP team.
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
"""
Mpt configuration
"""
from typing import TYPE_CHECKING, Optional, Union

# 检查是否在类型检查环境下
if TYPE_CHECKING:
    pass

# 导入预训练配置类
from ...configuration_utils import PretrainedConfig
# 导入日志工具
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# MPT预训练模型配置文件映射字典
MPT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "mosaicml/mpt-7b": "https://huggingface.co/mosaicml/mpt-7b/resolve/main/config.json",
}


class MptAttentionConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`MptAttention`] class. It is used to instantiate
    attention layers according to the specified arguments, defining the layers architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the MPT
    [mosaicml/mpt-7b](https://huggingface.co/mosaicml/mpt-7b) architecture. Most of the arguments are kept for backward
    compatibility with previous MPT models that are hosted on the Hub (previously with `trust_remote_code=True`).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    Args:
        attn_type (`str`, *optional*, defaults to `"multihead_attention"`):
            type of attention to use. Options: `"multihead_attention"`, `"multiquery_attention"`.
        attn_pdrop (`float`, *optional*, defaults to 0.0):
            The dropout probability for the attention layers.
        attn_impl (`str`, *optional*, defaults to `"torch"`):
            The attention implementation to use. One of `"torch"`, `"flash"`, or `"triton"`.
        clip_qkv (`float`, *optional*):
            If not `None`, clip the queries, keys, and values in the attention layer to this value.
        softmax_scale (`float`, *optional*, defaults to `None`):
            If not `None`, scale the softmax in the attention layer by this value. If `None`, will default to
            `1/sqrt(hidden_size)`.
        prefix_lm (`bool`, *optional*, defaults to `False`)):
            Whether the model should operate as a Prefix LM. This requires passing an extra `prefix_mask` argument
            which indicates which tokens belong to the prefix. Tokens in the prefix can attend to one another
            bi-directionally. Tokens outside the prefix use causal attention.
        qk_ln (`bool`, *optional*, defaults to `False`):
            Whether to apply layer normalization to the queries and keys in the attention layer.
        attn_uses_sequence_id (`bool`, *optional*, defaults to `False`)):
            Whether to restrict attention to tokens that have the same token_type_ids. When the model is in `train`
            mode, this requires passing an extra *token_type_ids* argument which indicates which sub-sequence each
            token belongs to. Defaults to `False` meaning any provided *token_type_ids* will be ignored.
        alibi (`bool`, *optional*, defaults to `True`):
            Whether or not to use the alibi bias instead of positional embedding.
        alibi_bias_max (`int`, *optional*, defaults to 8):
            The maximum value of the alibi bias.
    """

    def __init__(
        self,
        attn_type="multihead_attention",
        attn_pdrop=0,
        attn_impl="torch",
        clip_qkv=None,
        softmax_scale=None,
        prefix_lm=False,
        qk_ln=False,
        attn_uses_sequence_id=False,
        alibi=True,
        alibi_bias_max=8,
        **kwargs,
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化注意力机制类型
        self.attn_type = attn_type
        # 初始化注意力机制中的 dropout 概率
        self.attn_pdrop = attn_pdrop
        # 初始化注意力机制的实现方式
        self.attn_impl = attn_impl
        # 如果设置了 clip_qkv 参数，则用其值来剪裁注意力层中的 queries、keys 和 values
        self.clip_qkv = clip_qkv
        # 如果设置了 softmax_scale 参数，则用其值来缩放注意力层中的 softmax 操作
        self.softmax_scale = softmax_scale
        # 是否将模型设置为 Prefix LM，这要求传入额外的 prefix_mask 参数
        self.prefix_lm = prefix_lm
        # 是否对注意力层中的 queries 和 keys 应用层归一化
        self.qk_ln = qk_ln
        # 是否限制注意力仅应用于具有相同 token_type_ids 的 tokens
        self.attn_uses_sequence_id = attn_uses_sequence_id
        # 是否使用 alibi 偏置而不是位置嵌入
        self.alibi = alibi
        # 初始化 alibi 偏置的最大值
        self.alibi_bias_max = alibi_bias_max

        # 检查 attn_type 是否为支持的类型，否则抛出 ValueError 异常
        if attn_type not in ["multihead_attention", "multiquery_attention"]:
            raise ValueError(
                f"`attn_type` has to be either `multihead_attention` or `multiquery_attention`. Received: {attn_type}"
            )

    @classmethod
    # 从预训练模型名称或路径加载配置，并返回预训练配置对象
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs) -> "PretrainedConfig":
        # 在 kwargs 中设置 token
        cls._set_token_in_kwargs(kwargs)

        # 调用 get_config_dict 方法获取配置字典和更新后的 kwargs
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果配置字典中的模型类型是 "mpt"，则使用其对应的注意力配置
        if config_dict.get("model_type") == "mpt":
            config_dict = config_dict["attn_config"]

        # 如果配置字典中包含 "model_type" 并且类有 "model_type" 属性，并且两者不相等，发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 使用 from_dict 方法根据配置字典创建预训练配置对象
        return cls.from_dict(config_dict, **kwargs)
class MptConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`MptModel`]. It is used to instantiate a Mpt model
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to the Mpt-7b architecture
    [mosaicml/mpt-7b](https://huggingface.co/mosaicml/mpt-7b).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    # 定义 Transformer 编码器的配置参数

    Args:
        d_model (`int`, *optional*, defaults to 2048):
            # 嵌入和隐藏状态的维度。
            Dimensionality of the embeddings and hidden states.
        n_heads (`int`, *optional*, defaults to 16):
            # 每个注意力层中的注意力头数量。
            Number of attention heads for each attention layer in the Transformer encoder.
        n_layers (`int`, *optional*, defaults to 24):
            # Transformer 编码器中隐藏层的数量。
            Number of hidden layers in the Transformer encoder.
        expansion_ratio (`int`, *optional*, defaults to 4):
            # MLP 中上/下扩展比率。
            The ratio of the up/down scale in the MLP.
        max_seq_len (`int`, *optional*, defaults to 2048):
            # 模型的最大序列长度。
            The maximum sequence length of the model.
        vocab_size (`int`, *optional*, defaults to 50368):
            # Mpt 模型的词汇量大小。定义了在调用 `MptModel` 时可以表示的不同标记的最大数量。
            Vocabulary size of the Mpt model. Defines the maximum number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`MptModel`]. Check [this
            discussion](https://huggingface.co/bigscience/mpt/discussions/120#633d28389addb8530b406c2a) on how the
            `vocab_size` has been defined.
        resid_pdrop (`float`, *optional*, defaults to 0.0):
            # 在与残差结合之前应用于注意力输出的 dropout 概率。
            The dropout probability applied to the attention output before combining with residual.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
            # 层归一化层中使用的 epsilon。
            The epsilon to use in the layer normalization layers.
        emb_pdrop (`float`, *optional*, defaults to 0.0):
            # 嵌入层的 dropout 概率。
            The dropout probability for the embedding layer.
        learned_pos_emb (`bool`, *optional*, defaults to `True`):
            # 是否使用学习的位置编码。
            Whether to use learned positional embeddings.
        attn_config (`dict`, *optional*):
            # 用于配置模型注意力模块的字典。
            A dictionary used to configure the model's attention module.
        init_device (`str`, *optional*, defaults to `"cpu"`):
            # 用于参数初始化的设备。为了向后兼容而定义。
            The device to use for parameter initialization. Defined for backward compatibility
        logit_scale (`float`, *optional*):
            # 如果不为 None，则缩放 logits 的值。
            If not None, scale the logits by this value.
        no_bias (`bool`, *optional*, defaults to `True`):
            # 是否在所有线性层中使用偏置。
            Whether to use bias in all linear layers.
        verbose (`int`, *optional*, defaults to 0):
            # 用于日志记录的详细级别。在先前版本的 MPT 模型中用于日志记录。此参数已弃用。
            The verbosity level to use for logging. Used in the previous versions of MPT models for logging. This
            argument is deprecated.
        embedding_fraction (`float`, *optional*, defaults to 1.0):
            # 缩放嵌入层梯度的比例。
            The fraction to scale the gradients of the embedding layer by.
        norm_type (`str`, *optional*, defaults to `"low_precision_layernorm"`):
            # 要使用的层归一化类型。所有 MPT 模型使用相同的层归一化实现。为了向后兼容而定义。
            Type of layer norm to use. All MPT models uses the same layer norm implementation. Defined for backward
            compatibility.
        use_cache (`bool`, *optional*, defaults to `False`):
            # 模型是否应返回最后的 key/values 注意力（并非所有模型都使用）。
            Whether or not the model should return the last key/values attentions (not used by all models).
        initializer_range (`float`, *optional*, defaults to 0.02):
            # 用于初始化所有权重矩阵的截断正态初始化器的标准差。
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    Example:

    ```python
    # 导入 transformers 库中的 MptConfig 和 MptModel 类
    >>> from transformers import MptConfig, MptModel

    # 初始化一个 MptConfig 实例
    >>> # Initializing a Mpt configuration
    >>> configuration = MptConfig()

    # 使用配置初始化一个模型实例（权重随机生成）
    >>> # Initializing a model (with random weights) from the configuration
    >>> model = MptModel(configuration)

    # 获取模型的配置信息
    >>> # Accessing the model configuration
    >>> configuration = model.config

    # 设定模型类型为 "mpt"
    model_type = "mpt"

    # 定义一个属性映射字典，将 MptConfig 中的部分属性名映射到另一种命名方式
    attribute_map = {
        "num_attention_heads": "n_heads",  # 注意力头数量映射为 n_heads
        "hidden_size": "d_model",          # 隐藏层大小映射为 d_model
        "num_hidden_layers": "n_layers",   # 隐藏层层数映射为 n_layers
    }

    # 定义 MptConfig 类
    def __init__(
        self,
        d_model: int = 2048,
        n_heads: int = 16,
        n_layers: int = 24,
        expansion_ratio: int = 4,
        max_seq_len: int = 2048,
        vocab_size: int = 50368,
        resid_pdrop: float = 0.0,
        layer_norm_epsilon: float = 1e-5,
        emb_pdrop: float = 0.0,
        learned_pos_emb: bool = True,
        attn_config: MptAttentionConfig = None,
        init_device: str = "cpu",
        logit_scale: Optional[Union[float, str]] = None,
        no_bias: bool = True,
        verbose: int = 0,
        embedding_fraction: float = 1.0,
        norm_type: str = "low_precision_layernorm",
        use_cache: bool = False,
        initializer_range=0.02,
        **kwargs,
    ):
        # 如果没有给定 attn_config，则初始化一个空的 MptAttentionConfig 对象
        if attn_config is None:
            self.attn_config = MptAttentionConfig()
        # 如果 attn_config 是字典类型，则使用这些参数初始化一个 MptAttentionConfig 对象
        elif isinstance(attn_config, dict):
            self.attn_config = MptAttentionConfig(**attn_config)
        else:
            self.attn_config = attn_config

        # 初始化各个属性值
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.expansion_ratio = expansion_ratio
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.resid_pdrop = resid_pdrop
        self.emb_pdrop = emb_pdrop
        self.learned_pos_emb = learned_pos_emb
        self.init_device = init_device
        self.logit_scale = logit_scale
        self.no_bias = no_bias
        self.verbose = verbose
        self.embedding_fraction = embedding_fraction
        self.norm_type = norm_type
        self.layer_norm_epsilon = layer_norm_epsilon
        self.use_cache = use_cache
        self.initializer_range = initializer_range

        # 调用父类的构造函数，传递其他未命名的关键字参数
        super().__init__(**kwargs)
```