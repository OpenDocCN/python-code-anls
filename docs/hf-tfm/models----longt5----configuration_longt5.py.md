# `.\transformers\models\longt5\configuration_longt5.py`

```
# coding=utf-8
# Copyright 2022, The LongT5 Authors and HuggingFace Inc.
# 版权声明，指定版权所有者及版权信息
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache License, Version 2.0 发布，表示遵循 Apache 许可证 2.0 版本
# you may not use this file except in compliance with the License.
# 除非符合许可证的规定，否则不得使用此文件
# You may obtain a copy of the License at
# 您可以在以下网址获得许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# 在适用法律要求或书面同意的情况下，软件
# distributed under the License is distributed on an "AS IS" BASIS,
# 按“原样”分发在许可证的基础上，
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 不附带任何形式的明示或暗示担保或条件
# See the License for the specific language governing permissions and
# limitations under the License.
# 请参阅许可证以了解特定语言的权限和限制
""" LongT5 model configuration"""
# LongT5 模型的配置

from typing import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxSeq2SeqConfigWithPast
from ...utils import logging

# 导入必要的库和模块

logger = logging.get_logger(__name__)

# 获取预训练模型配置的映射表
LONGT5_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/long-t5-local-base": "https://huggingface.co/google/long-t5-local-base/blob/main/config.json",
    "google/long-t5-local-large": "https://huggingface.co/google/long-t5-local-large/blob/main/config.json",
    "google/long-t5-tglobal-base": "https://huggingface.co/google/long-t5-tglobal-base/blob/main/config.json",
    "google/long-t5-tglobal-large": "https://huggingface.co/google/long-t5-tglobal-large/blob/main/config.json",
}

# 定义 LongT5Config 类，继承自 PretrainedConfig
class LongT5Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LongT5Model`] or a [`FlaxLongT5Model`]. It is
    used to instantiate a LongT5 model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the LongT5
    [google/long-t5-local-base](https://huggingface.co/google/long-t5-local-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    # 这是用于存储 [`LongT5Model`] 或 [`FlaxLongT5Model`] 的配置信息的配置类。根据指定的参数实例化 LongT5 模型，定义模型架构。使用默认值实例化配置将产生类似于 LongT5 [google/long-t5-local-base](https://huggingface.co/google/long-t5-local-base) 架构的配置。

    # 配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。
    Arguments:
        vocab_size (`int`, *optional*, defaults to 32128):
            Vocabulary size of the LongT5 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`LongT5Model`].
        d_model (`int`, *optional*, defaults to 512):
            Size of the encoder layers and the pooler layer.
        d_kv (`int`, *optional*, defaults to 64):
            Size of the key, query, value projections per attention head. `d_kv` has to be equal to `d_model //
            num_heads`.
        d_ff (`int`, *optional*, defaults to 2048):
            Size of the intermediate feed forward layer in each `LongT5Block`.
        num_layers (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer encoder.
        num_decoder_layers (`int`, *optional*):
            Number of hidden layers in the Transformer decoder. Will use the same value as `num_layers` if not set.
        num_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        local_radius (`int`, *optional*, defaults to 127)
            Number of tokens to the left/right for each token to locally self-attend in a local attention mechanism.
        global_block_size (`int`, *optional*, defaults to 16)
            Lenght of blocks an input sequence is divided into for a global token representation. Used only for
            `encoder_attention_type = "transient-global"`.
        relative_attention_num_buckets (`int`, *optional*, defaults to 32):
            The number of buckets to use for each attention layer.
        relative_attention_max_distance (`int`, *optional*, defaults to 128):
            The maximum distance of the longer sequences for the bucket separation.
        dropout_rate (`float`, *optional*, defaults to 0.1):
            The ratio for all dropout layers.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        initializer_factor (`float`, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        feed_forward_proj (`string`, *optional*, defaults to `"relu"`):
            Type of feed forward layer to be used. Should be one of `"relu"` or `"gated-gelu"`. LongT5v1.1 uses the
            `"gated-gelu"` feed forward projection. Original LongT5 implementation uses `"gated-gelu"`.
        encoder_attention_type (`string`, *optional*, defaults to `"local"`):
            Type of encoder attention to be used. Should be one of `"local"` or `"transient-global"`, which are
            supported by LongT5 implementation.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
    """

    model_type = "longt5"
    # 在推断时需要忽略的键值
    keys_to_ignore_at_inference = ["past_key_values"]
    # 属性映射，用于转换参数名称
    attribute_map = {"hidden_size": "d_model", "num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}

    # 初始化函数，设置模型参数和属性
    def __init__(
        self,
        vocab_size=32128,  # 词汇表大小，默认为32128
        d_model=512,  # 模型维度，默认为512
        d_kv=64,  # 键值的维度，默认为64
        d_ff=2048,  # 前馈神经网络的隐藏层维度，默认为2048
        num_layers=6,  # 层数，默认为6
        num_decoder_layers=None,  # 解码器层数，默认为None
        num_heads=8,  # 注意力头数，默认为8
        local_radius=127,  # 局部注意力的半径，默认为127
        global_block_size=16,  # 全局块的大小，默认为16
        relative_attention_num_buckets=32,  # 相对注意力的桶数，默认为32
        relative_attention_max_distance=128,  # 相对注意力的最大距离，默认为128
        dropout_rate=0.1,  # 丢弃率，默认为0.1
        layer_norm_epsilon=1e-6,  # 层归一化的epsilon，默认为1e-6
        initializer_factor=1.0,  # 初始化因子，默认为1.0
        feed_forward_proj="relu",  # 前馈神经网络的激活函数，默认为"relu"
        is_encoder_decoder=True,  # 是否为编码器-解码器模型，默认为True
        encoder_attention_type="local",  # 编码器的注意力类型，默认为"local"
        use_cache=True,  # 是否使用缓存，默认为True
        pad_token_id=0,  # 填充标记ID，默认为0
        eos_token_id=1,  # 终止标记ID，默认为1
        **kwargs,  # 其他关键字参数
    ):
        # 设置模型参数和属性
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        # 默认为对称
        self.num_decoder_layers = num_decoder_layers if num_decoder_layers is not None else self.num_layers
        self.num_heads = num_heads
        self.local_radius = local_radius
        self.global_block_size = global_block_size
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.encoder_attention_type = encoder_attention_type
        self.use_cache = use_cache

        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = act_info[0] == "gated"

        if len(act_info) > 1 and act_info[0] != "gated" or len(act_info) > 2:
            # 检查前馈神经网络的激活函数格式是否正确
            raise ValueError(
                f"`feed_forward_proj`: {feed_forward_proj} is not a valid activation function of the dense layer. "
                "Please make sure `feed_forward_proj` is of the format `gated-{ACT_FN}` or `{ACT_FN}`, e.g. "
                "'gated-gelu' or 'relu'"
            )

        # 为了向后兼容性
        if feed_forward_proj == "gated-gelu":
            self.dense_act_fn = "gelu_new"

        # 调用父类的初始化方法
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )
# 定义一个名为 LongT5OnnxConfig 的类，继承自 OnnxSeq2SeqConfigWithPast 类
class LongT5OnnxConfig(OnnxSeq2SeqConfigWithPast):
    
    # inputs 属性，返回一个字符串到整数到字符串的映射
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 公共输入，包含 "input_ids" 和 "attention_mask"，用于编码器输入
        common_inputs = {
            "input_ids": {0: "batch", 1: "encoder_sequence"},
            "attention_mask": {0: "batch", 1: "encoder_sequence"},
        }
        # 如果使用过去的信息
        if self.use_past:
            # 调整输入的 attention_mask
            common_inputs["attention_mask"][1] = "past_encoder_sequence + sequence"
            # 添加解码器的输入
            common_inputs["decoder_input_ids"] = {0: "batch"}
            # 调整解码器的 attention_mask
            common_inputs["decoder_attention_mask"] = {0: "batch", 1: "past_decoder_sequence + sequence"}
        else:
            # 添加解码器的输入
            common_inputs["decoder_input_ids"] = {0: "batch", 1: "decoder_sequence"}
            # 设置解码器的 attention_mask
            common_inputs["decoder_attention_mask"] = {0: "batch", 1: "decoder_sequence"}

        # 如果使用过去的信息，则填充输入
        if self.use_past:
            self.fill_with_past_key_values_(common_inputs, direction="inputs")

        # 返回输入集合
        return common_inputs

    # default_onnx_opset 属性，返回整数
    @property
    def default_onnx_opset(self) -> int:
        # 返回默认的 ONNX 操作集版本号
        return 13
```