# `.\models\funnel\configuration_funnel.py`

```py
# coding=utf-8
# Copyright 2020, Hugging Face
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
""" Funnel Transformer model configuration"""

# 导入预训练配置类和日志工具
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取全局日志记录器
logger = logging.get_logger(__name__)

# 预训练模型配置文件映射表，映射模型名称到其配置文件的 URL
FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "funnel-transformer/small": "https://huggingface.co/funnel-transformer/small/resolve/main/config.json",
    "funnel-transformer/small-base": "https://huggingface.co/funnel-transformer/small-base/resolve/main/config.json",
    "funnel-transformer/medium": "https://huggingface.co/funnel-transformer/medium/resolve/main/config.json",
    "funnel-transformer/medium-base": "https://huggingface.co/funnel-transformer/medium-base/resolve/main/config.json",
    "funnel-transformer/intermediate": (
        "https://huggingface.co/funnel-transformer/intermediate/resolve/main/config.json"
    ),
    "funnel-transformer/intermediate-base": (
        "https://huggingface.co/funnel-transformer/intermediate-base/resolve/main/config.json"
    ),
    "funnel-transformer/large": "https://huggingface.co/funnel-transformer/large/resolve/main/config.json",
    "funnel-transformer/large-base": "https://huggingface.co/funnel-transformer/large-base/resolve/main/config.json",
    "funnel-transformer/xlarge": "https://huggingface.co/funnel-transformer/xlarge/resolve/main/config.json",
    "funnel-transformer/xlarge-base": "https://huggingface.co/funnel-transformer/xlarge-base/resolve/main/config.json",
}

# 定义 FunnelConfig 类，继承自 PretrainedConfig 类
class FunnelConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FunnelModel`] or a [`TFBertModel`]. It is used to
    instantiate a Funnel Transformer model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the Funnel
    Transformer [funnel-transformer/small](https://huggingface.co/funnel-transformer/small) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """

    # 指定模型类型为 "funnel"
    model_type = "funnel"
    # 定义属性映射，将配置中的参数名映射到模型参数名
    attribute_map = {
        "hidden_size": "d_model",               # hidden_size 映射到 d_model
        "num_attention_heads": "n_head",        # num_attention_heads 映射到 n_head
    }
    # 初始化函数，用于创建一个新的模型对象
    def __init__(
        self,
        vocab_size=30522,                         # 设置词汇表大小，默认为30522
        block_sizes=[4, 4, 4],                     # 每个块的大小列表，默认为[4, 4, 4]
        block_repeats=None,                       # 每个块的重复次数列表，默认为None
        num_decoder_layers=2,                     # 解码器层数，默认为2
        d_model=768,                              # 模型的维度，默认为768
        n_head=12,                                # 注意力头的数量，默认为12
        d_head=64,                                # 每个注意力头的维度，默认为64
        d_inner=3072,                             # 内部隐藏层的维度，默认为3072
        hidden_act="gelu_new",                    # 隐藏层激活函数，默认为"gelu_new"
        hidden_dropout=0.1,                       # 隐藏层的Dropout比率，默认为0.1
        attention_dropout=0.1,                    # 注意力层的Dropout比率，默认为0.1
        activation_dropout=0.0,                   # 激活函数的Dropout比率，默认为0.0
        initializer_range=0.1,                    # 初始化范围，默认为0.1
        initializer_std=None,                     # 初始化标准差，默认为None
        layer_norm_eps=1e-9,                      # Layer Norm的epsilon，默认为1e-9
        pooling_type="mean",                      # 汇聚类型，默认为"mean"
        attention_type="relative_shift",          # 注意力类型，默认为"relative_shift"
        separate_cls=True,                        # 是否分开处理CLS，默认为True
        truncate_seq=True,                        # 是否截断序列，默认为True
        pool_q_only=True,                         # 是否仅对query池化，默认为True
        **kwargs,                                 # 其他关键字参数
    ):
        self.vocab_size = vocab_size               # 设置词汇表大小属性
        self.block_sizes = block_sizes             # 设置块大小列表属性
        self.block_repeats = [1] * len(block_sizes) if block_repeats is None else block_repeats
                                                  # 设置块重复次数列表属性，若未提供则为每个块设置为1次
        assert len(block_sizes) == len(
            self.block_repeats
        ), "`block_sizes` and `block_repeats` should have the same length."  # 检查块大小列表和重复次数列表长度是否相同

        self.num_decoder_layers = num_decoder_layers  # 设置解码器层数属性
        self.d_model = d_model                      # 设置模型维度属性
        self.n_head = n_head                        # 设置注意力头数量属性
        self.d_head = d_head                        # 设置每个注意力头维度属性
        self.d_inner = d_inner                      # 设置内部隐藏层维度属性
        self.hidden_act = hidden_act                # 设置隐藏层激活函数属性
        self.hidden_dropout = hidden_dropout        # 设置隐藏层Dropout比率属性
        self.attention_dropout = attention_dropout  # 设置注意力层Dropout比率属性
        self.activation_dropout = activation_dropout  # 设置激活函数Dropout比率属性
        self.initializer_range = initializer_range  # 设置初始化范围属性
        self.initializer_std = initializer_std      # 设置初始化标准差属性
        self.layer_norm_eps = layer_norm_eps        # 设置Layer Norm的epsilon属性

        assert pooling_type in [
            "mean",
            "max",
        ], f"Got {pooling_type} for `pooling_type` but only 'mean' and 'max' are supported."
                                                  # 检查汇聚类型是否支持，只支持'mean'和'max'
        self.pooling_type = pooling_type            # 设置汇聚类型属性

        assert attention_type in [
            "relative_shift",
            "factorized",
        ], f"Got {attention_type} for `attention_type` but only 'relative_shift' and 'factorized' are supported."
                                                  # 检查注意力类型是否支持，只支持'relative_shift'和'factorized'
        self.attention_type = attention_type        # 设置注意力类型属性
        self.separate_cls = separate_cls            # 设置是否分开处理CLS属性
        self.truncate_seq = truncate_seq            # 设置是否截断序列属性
        self.pool_q_only = pool_q_only              # 设置是否仅对query池化属性

        super().__init__(**kwargs)                  # 调用父类初始化函数，并传递其他关键字参数

    @property
    def num_hidden_layers(self):
        return sum(self.block_sizes)                # 返回总隐藏层数，即所有块大小之和

    @num_hidden_layers.setter
    def num_hidden_layers(self, value):
        raise NotImplementedError(
            "This model does not support the setting of `num_hidden_layers`. Please set `block_sizes`."
        )                                           # 设置num_hidden_layers属性的setter方法，不支持设置，提出错误提示

    @property
    def num_blocks(self):
        return len(self.block_sizes)                # 返回块数量，即块大小列表的长度

    @num_blocks.setter
    def num_blocks(self, value):
        raise NotImplementedError("This model does not support the setting of `num_blocks`. Please set `block_sizes`.")
                                                  # 设置num_blocks属性的setter方法，不支持设置，提出错误提示
```