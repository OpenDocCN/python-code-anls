# `.\models\udop\configuration_udop.py`

```py
# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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
""" UDOP model configuration"""


from ...configuration_utils import PretrainedConfig  # 导入预训练配置类
from ...utils import logging  # 导入日志工具


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

UDOP_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/udop-large": "https://huggingface.co/microsoft/udop-large/resolve/main/config.json",
}


class UdopConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`UdopForConditionalGeneration`]. It is used to
    instantiate a UDOP model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the UDOP
    [microsoft/udop-large](https://huggingface.co/microsoft/udop-large) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    """

    model_type = "udop"  # 模型类型为 UDOP
    keys_to_ignore_at_inference = ["past_key_values"]  # 推断时忽略的键列表
    attribute_map = {"hidden_size": "d_model", "num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}  # 属性映射表

    def __init__(
        self,
        vocab_size=33201,  # 词汇表大小
        d_model=1024,  # 模型的隐藏层大小
        d_kv=64,  # key 和 value 向量的大小
        d_ff=4096,  # 前向传播网络中间层的大小
        num_layers=24,  # 模型的层数
        num_decoder_layers=None,  # 解码器层数
        num_heads=16,  # 注意力头的数量
        relative_attention_num_buckets=32,  # 相对注意力的桶数
        relative_attention_max_distance=128,  # 相对注意力的最大距离
        relative_bias_args=[{"type": "1d"}, {"type": "horizontal"}, {"type": "vertical"}],  # 相对偏置参数
        dropout_rate=0.1,  # dropout 率
        layer_norm_epsilon=1e-6,  # 层归一化的 epsilon 参数
        initializer_factor=1.0,  # 初始化因子
        feed_forward_proj="relu",  # 前向传播网络的激活函数
        is_encoder_decoder=True,  # 是否为编码器-解码器结构
        use_cache=True,  # 是否使用缓存
        pad_token_id=0,  # 填充 token 的 id
        eos_token_id=1,  # 终止 token 的 id
        max_2d_position_embeddings=1024,  # 最大的二维位置嵌入数
        image_size=224,  # 图像尺寸
        patch_size=16,  # 图像分块的大小
        num_channels=3,  # 图像通道数
        **kwargs,  # 其他参数
    ):
        super().__init__(**kwargs)  # 调用父类的初始化方法，传递其他参数
        ):
        # 初始化 Transformer 模型的各种参数
        self.vocab_size = vocab_size  # 词汇表大小
        self.d_model = d_model  # 模型的隐藏层维度
        self.d_kv = d_kv  # 键值对的维度
        self.d_ff = d_ff  # 前向传播层的维度
        self.num_layers = num_layers  # 总层数
        self.num_decoder_layers = (
            num_decoder_layers if num_decoder_layers is not None else self.num_layers
        )  # 解码器层数，默认与编码器对称
        self.num_heads = num_heads  # 头的数量
        self.relative_attention_num_buckets = relative_attention_num_buckets  # 相对注意力的桶数
        self.relative_attention_max_distance = relative_attention_max_distance  # 相对注意力的最大距离
        self.dropout_rate = dropout_rate  # Dropout 比率
        self.layer_norm_epsilon = layer_norm_epsilon  # Layer normalization 的 epsilon 值
        self.initializer_factor = initializer_factor  # 初始化因子
        self.feed_forward_proj = feed_forward_proj  # 前向传播层的激活函数
        self.use_cache = use_cache  # 是否使用缓存

        # UDOP 属性
        self.max_2d_position_embeddings = max_2d_position_embeddings  # 二维位置嵌入的最大值
        self.image_size = image_size  # 图像尺寸
        self.patch_size = patch_size  # 补丁尺寸
        self.num_channels = num_channels  # 通道数
        if not isinstance(relative_bias_args, list):
            raise ValueError("`relative_bias_args` should be a list of dictionaries.")
        self.relative_bias_args = relative_bias_args  # 相对偏置参数列表

        # 解析前向传播激活函数
        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]  # 密集层的激活函数
        self.is_gated_act = act_info[0] == "gated"  # 是否是门控激活函数

        # 检查前向传播激活函数格式是否正确
        if len(act_info) > 1 and act_info[0] != "gated" or len(act_info) > 2:
            raise ValueError(
                f"`feed_forward_proj`: {feed_forward_proj} is not a valid activation function of the dense layer."
                "Please make sure `feed_forward_proj` is of the format `gated-{ACT_FN}` or `{ACT_FN}`, e.g. "
                "'gated-gelu' or 'relu'"
            )

        # 调用父类构造函数，初始化基本参数
        super().__init__(
            pad_token_id=pad_token_id,  # 填充符号的 ID
            eos_token_id=eos_token_id,  # 终止符号的 ID
            is_encoder_decoder=is_encoder_decoder,  # 是否是编码器-解码器模型
            **kwargs,  # 其它参数
        )
```