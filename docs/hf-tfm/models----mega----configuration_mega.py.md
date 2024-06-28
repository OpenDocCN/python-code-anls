# `.\models\mega\configuration_mega.py`

```py
# coding=utf-8
# 上面的行指定了文件的编码格式为 UTF-8

# Copyright 2023 The Mega Authors and The HuggingFace Inc. team.
# 以下是版权声明，说明了代码的版权归属

# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache License, Version 2.0 授权许可

# you may not use this file except in compliance with the License.
# 除非遵循许可证，否则不能使用该文件

# You may obtain a copy of the License at
# 可以在下面链接获取许可证副本

#     http://www.apache.org/licenses/LICENSE-2.0
# 许可证的获取链接

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 根据许可证分发的软件以“原样”分发，没有任何明示或暗示的保证或条件

# See the License for the specific language governing permissions and
# limitations under the License.
# 查看许可证以获取具体的语言控制权限和限制

""" MEGA configuration"""
# 注释说明这是 MEGA 模型的配置文件

from collections import OrderedDict
from typing import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging

# 引入必要的模块和类

logger = logging.get_logger(__name__)

# 获取logger对象，用于记录日志

MEGA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "mnaylor/mega-base-wikitext": "https://huggingface.co/mnaylor/mega-base-wikitext/resolve/main/config.json",
}

# 定义一个映射表，将模型名称映射到预训练模型配置文件的下载链接

class MegaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MegaModel`]. It is used to instantiate a Mega
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Mega
    [mnaylor/mega-base-wikitext](https://huggingface.co/mnaylor/mega-base-wikitext) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Examples:

    ```
    >>> from transformers import MegaConfig, MegaModel

    >>> # Initializing a Mega configuration
    >>> configuration = MegaConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = MegaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """
    # MegaConfig 类的文档字符串，描述了配置一个 MegaModel 所需的参数和用法示例

    model_type = "mega"
    # 模型类型为 "mega"
    # 初始化函数，用于初始化一个类的实例
    def __init__(
        self,
        vocab_size=30522,  # 词汇表大小，默认为30522
        hidden_size=128,  # 隐藏层大小，默认为128
        num_hidden_layers=4,  # 隐藏层数量，默认为4
        intermediate_size=256,  # 中间层大小，默认为256
        ema_projection_size=16,  # EMA（指数移动平均）投影大小，默认为16
        bidirectional=True,  # 是否双向，默认为True
        shared_representation_size=64,  # 共享表示大小，默认为64
        use_chunking=False,  # 是否使用分块，默认为False
        chunk_size=-1,  # 分块大小，默认为-1
        truncation=None,  # 截断类型，默认为None
        normalize_before_mega=True,  # 在巨型模块之前是否进行归一化，默认为True
        normalization_type="scalenorm",  # 归一化类型，默认为"scalenorm"
        norm_affine=True,  # 归一化是否包含仿射变换，默认为True
        activation="silu",  # 激活函数类型，默认为"silu"
        attention_activation="softmax",  # 注意力机制的激活函数类型，默认为"softmax"
        dropout_prob=0.1,  # 一般的dropout概率，默认为0.1
        hidden_dropout_prob=0.1,  # 隐藏层dropout概率，默认为0.1
        attention_probs_dropout_prob=0.1,  # 注意力机制的dropout概率，默认为0.1
        use_feature_dropout=False,  # 是否使用特征dropout，默认为False
        use_normalized_ffn=True,  # 是否使用归一化的前馈网络，默认为True
        nffn_hidden_size=256,  # 归一化前馈网络的隐藏层大小，默认为256
        normalize_before_ffn=True,  # 在前馈网络之前是否进行归一化，默认为True
        nffn_activation_dropout_prob=0.1,  # 前馈网络的激活函数dropout概率，默认为0.1
        max_positions=2048,  # 最大位置编码数，默认为2048
        add_token_type_embeddings=False,  # 是否添加token类型的嵌入，默认为False
        type_vocab_size=2,  # token类型的词汇表大小，默认为2
        initializer_range=0.02,  # 初始化范围，默认为0.02
        ema_delta_alpha_range=0.2,  # EMA增量α范围，默认为0.2
        ema_beta_range=0.02,  # EMA β范围，默认为0.02
        ema_gamma_omega_range=1.0,  # EMA γ和ω范围，默认为1.0
        pad_token_id=1,  # 填充token的ID，默认为1
        bos_token_id=0,  # 开始token的ID，默认为0
        eos_token_id=2,  # 结束token的ID，默认为2
        relative_positional_bias="rotary",  # 相对位置偏置类型，默认为"rotary"
        classifier_dropout=None,  # 分类器的dropout概率，默认为None
        use_cache=True,  # 是否使用缓存，默认为True
        add_lm_hidden_dense_layer=True,  # 是否添加语言模型隐藏层密集层，默认为True
        **kwargs,  # 其他关键字参数
    ):
        ):
        # 调用父类的初始化方法，设置特定的参数
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        # 初始化模型的各种参数
        self.vocab_size = vocab_size                    # 词汇表大小
        self.hidden_size = hidden_size                  # 隐藏层大小
        self.num_hidden_layers = num_hidden_layers      # 隐藏层数量
        self.activation = activation                    # 激活函数类型
        self.attention_activation = attention_activation  # 注意力机制激活函数类型
        self.intermediate_size = intermediate_size      # 中间层大小
        self.ema_projection_size = ema_projection_size  # 指数移动平均投影大小
        self.bidirectional = bidirectional              # 是否使用双向模型
        self.shared_representation_size = shared_representation_size  # 共享表示大小
        self.use_chunking = use_chunking                # 是否使用分块处理
        self.chunk_size = chunk_size                    # 分块大小
        self.truncation = truncation                    # 截断长度
        self.normalize_before_mega = normalize_before_mega  # 在大模型之前进行归一化
        self.normalization_type = normalization_type    # 归一化类型
        self.norm_affine = norm_affine                  # 归一化的仿射变换
        self.dropout_prob = dropout_prob                # 通用丢弃概率
        self.hidden_dropout_prob = hidden_dropout_prob  # 隐藏层丢弃概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob  # 注意力概率丢弃概率
        self.use_feature_dropout = use_feature_dropout  # 是否使用特征丢弃
        self.use_normalized_ffn = use_normalized_ffn    # 是否使用归一化的前馈网络
        self.nffn_hidden_size = nffn_hidden_size        # 归一化前馈网络的隐藏层大小
        self.normalize_before_ffn = normalize_before_ffn  # 在前馈网络之前进行归一化
        self.nffn_activation_dropout_prob = nffn_activation_dropout_prob  # 前馈网络激活丢弃概率
        self.max_positions = max_positions              # 最大位置编码
        self.add_token_type_embeddings = add_token_type_embeddings  # 是否添加类型嵌入
        self.type_vocab_size = type_vocab_size          # 类型词汇表大小
        self.initializer_range = initializer_range      # 初始化范围
        self.ema_delta_alpha_range = ema_delta_alpha_range  # 指数移动平均增量 alpha 范围
        self.ema_beta_range = ema_beta_range            # 指数移动平均 beta 范围
        self.ema_gamma_omega_range = ema_gamma_omega_range  # 指数移动平均 gamma 和 omega 范围
        self.relative_positional_bias = relative_positional_bias  # 相对位置偏差
        self.use_cache = use_cache                      # 是否使用缓存
        self.classifier_dropout = classifier_dropout    # 分类器丢弃概率
        self.add_lm_hidden_dense_layer = add_lm_hidden_dense_layer  # 是否添加语言模型的隐藏密集层
        self.num_attention_heads = 1  # not used but required by Hugging Face  # 注意力头数量，Hugging Face 要求但未使用
class MegaOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务是多选，则定义动态轴包括批次、选项和序列
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        # 否则，定义动态轴包括批次和序列
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        # 返回有序字典，定义输入名称和对应的动态轴
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),         # 定义输入名称 input_ids 和动态轴
                ("attention_mask", dynamic_axis),    # 定义输入名称 attention_mask 和动态轴
            ]
        )
```