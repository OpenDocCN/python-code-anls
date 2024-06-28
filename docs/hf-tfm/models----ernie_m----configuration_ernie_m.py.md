# `.\models\ernie_m\configuration_ernie_m.py`

```
# coding=utf-8
# 上面是声明文件编码格式为 UTF-8，确保支持中文等特殊字符的正确显示
# Copyright 2023 Xuan Ouyang, Shuohuan Wang, Chao Pang, Yu Sun, Hao Tian, Hua Wu, Haifeng Wang and The HuggingFace Inc. team. All rights reserved.
# 版权声明，保留所有权利给 Xuan Ouyang, Shuohuan Wang, Chao Pang, Yu Sun, Hao Tian, Hua Wu, Haifeng Wang 和 HuggingFace Inc. 团队
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache License, Version 2.0 许可证授权，除非遵守许可证，否则不得使用此文件
# You may obtain a copy of the License at
# 可以在上述网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 除非适用法律要求或书面同意，否则根据许可证分发的软件基于“按原样”分发，没有任何明示或暗示的担保或条件
""" ErnieM model configuration"""
# ErnieM 模型配置信息
# Adapted from original paddlenlp repository.(https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/transformers/ernie_m/configuration.py)
# 改编自原始 PaddleNLP 仓库中的代码，此处为 ErnieM 模型的配置文件位置

from __future__ import annotations

from typing import Dict

from ...configuration_utils import PretrainedConfig

# 定义预训练模型及其对应的配置文件链接映射
ERNIE_M_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "susnato/ernie-m-base_pytorch": "https://huggingface.co/susnato/ernie-m-base_pytorch/blob/main/config.json",
    "susnato/ernie-m-large_pytorch": "https://huggingface.co/susnato/ernie-m-large_pytorch/blob/main/config.json",
}


class ErnieMConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ErnieMModel`]. It is used to instantiate a
    Ernie-M model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the `Ernie-M`
    [susnato/ernie-m-base_pytorch](https://huggingface.co/susnato/ernie-m-base_pytorch) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    pass  # 此处为占位符，表示该类暂时不需要添加额外的属性或方法
    # 定义函数的默认参数和说明文档，指定了ErnieMModel的输入参数
    Args:
        vocab_size (`int`, *optional*, defaults to 250002):
            `inputs_ids`的词汇表大小。也是令牌嵌入矩阵的词汇大小。
            定义了在调用`ErnieMModel`时`inputs_ids`可以表示的不同令牌数量。
        hidden_size (`int`, *optional*, defaults to 768):
            嵌入层、编码器层和池化层的维度。
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Transformer编码器中隐藏层的数量。
        num_attention_heads (`int`, *optional*, defaults to 12):
            Transformer编码器每个注意力层的注意力头数。
        intermediate_size (`int`, *optional*, defaults to 3072):
            编码器中前馈（ff）层的维度。输入张量首先从hidden_size投影到intermediate_size，
            然后再从intermediate_size投影回hidden_size。通常，intermediate_size大于hidden_size。
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            前馈层中的非线性激活函数。支持`"gelu"`、`"relu"`和其他torch支持的激活函数。
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            嵌入层和编码器中所有全连接层的dropout概率。
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            所有编码器层中`MultiHeadAttention`中使用的dropout概率，用于丢弃部分注意力目标。
        max_position_embeddings (`int`, *optional*, defaults to 514):
            位置编码维度的最大值，决定了输入序列的最大支持长度。
        initializer_range (`float`, *optional*, defaults to 0.02):
            用于初始化所有权重矩阵的正态分布的标准差。
            令牌词汇表中填充令牌的索引。
        pad_token_id (`int`, *optional*, defaults to 1):
            填充令牌的ID。
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            层归一化层使用的epsilon。
        classifier_dropout (`float`, *optional*):
            分类头部的dropout比率。
        act_dropout (`float`, *optional*, defaults to 0.0):
            在激活函数后使用的dropout概率，用于`ErnieMEncoderLayer`。
    
    # 定义了模型类型为"ernie_m"，并创建了属性映射字典
    model_type = "ernie_m"
    attribute_map: Dict[str, str] = {"dropout": "classifier_dropout", "num_classes": "num_labels"}
    # 初始化函数，用于初始化一个 Transformer 模型的配置参数
    def __init__(
        self,
        vocab_size: int = 250002,                     # 词汇表大小，默认为 250002
        hidden_size: int = 768,                       # 隐藏层大小，默认为 768
        num_hidden_layers: int = 12,                  # 隐藏层数，默认为 12
        num_attention_heads: int = 12,                # 注意力头数，默认为 12
        intermediate_size: int = 3072,                # 中间层大小，默认为 3072
        hidden_act: str = "gelu",                     # 隐藏层激活函数，默认为 "gelu"
        hidden_dropout_prob: float = 0.1,             # 隐藏层 dropout 概率，默认为 0.1
        attention_probs_dropout_prob: float = 0.1,    # 注意力矩阵 dropout 概率，默认为 0.1
        max_position_embeddings: int = 514,           # 最大位置编码数，默认为 514
        initializer_range: float = 0.02,              # 参数初始化范围，默认为 0.02
        pad_token_id: int = 1,                        # 填充 token 的 id，默认为 1
        layer_norm_eps: float = 1e-05,                # Layer Normalization 的 epsilon，默认为 1e-05
        classifier_dropout=None,                      # 分类器 dropout 概率，默认为 None
        act_dropout=0.0,                              # 激活函数 dropout 概率，默认为 0.0
        **kwargs,                                     # 其余未命名参数
    ):
        # 调用父类的初始化函数，设置填充 token 的 id 和其它未命名参数
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        # 设置模型的各项参数
        self.vocab_size = vocab_size                   # 设置词汇表大小
        self.hidden_size = hidden_size                 # 设置隐藏层大小
        self.num_hidden_layers = num_hidden_layers     # 设置隐藏层数
        self.num_attention_heads = num_attention_heads # 设置注意力头数
        self.intermediate_size = intermediate_size     # 设置中间层大小
        self.hidden_act = hidden_act                   # 设置隐藏层激活函数
        self.hidden_dropout_prob = hidden_dropout_prob # 设置隐藏层 dropout 概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob # 设置注意力矩阵 dropout 概率
        self.max_position_embeddings = max_position_embeddings  # 设置最大位置编码数
        self.initializer_range = initializer_range    # 设置参数初始化范围
        self.layer_norm_eps = layer_norm_eps          # 设置 Layer Normalization 的 epsilon
        self.classifier_dropout = classifier_dropout  # 设置分类器 dropout 概率
        self.act_dropout = act_dropout                # 设置激活函数 dropout 概率
```