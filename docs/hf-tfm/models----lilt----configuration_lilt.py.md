# `.\models\lilt\configuration_lilt.py`

```
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" LiLT configuration"""

from ...configuration_utils import PretrainedConfig  # 导入PretrainedConfig类，用于处理预训练模型配置
from ...utils import logging  # 导入logging模块，用于日志记录

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

LILT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "SCUT-DLVCLab/lilt-roberta-en-base": (
        "https://huggingface.co/SCUT-DLVCLab/lilt-roberta-en-base/resolve/main/config.json"
    ),
}

class LiltConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LiltModel`]. It is used to instantiate a LiLT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LiLT
    [SCUT-DLVCLab/lilt-roberta-en-base](https://huggingface.co/SCUT-DLVCLab/lilt-roberta-en-base) architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Examples:

    ```python
    >>> from transformers import LiltConfig, LiltModel

    >>> # Initializing a LiLT SCUT-DLVCLab/lilt-roberta-en-base style configuration
    >>> configuration = LiltConfig()
    >>> # Randomly initializing a model from the SCUT-DLVCLab/lilt-roberta-en-base style configuration
    >>> model = LiltModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "lilt"  # 定义模型类型为"lilt"

    def __init__(
        self,
        vocab_size=30522,  # 词汇表大小，默认为30522
        hidden_size=768,  # 隐藏层大小，默认为768
        num_hidden_layers=12,  # 隐藏层数，默认为12
        num_attention_heads=12,  # 注意力头数，默认为12
        intermediate_size=3072,  # 中间层大小，默认为3072
        hidden_act="gelu",  # 隐藏层激活函数，默认为GELU
        hidden_dropout_prob=0.1,  # 隐藏层Dropout概率，默认为0.1
        attention_probs_dropout_prob=0.1,  # 注意力概率Dropout概率，默认为0.1
        max_position_embeddings=512,  # 最大位置嵌入长度，默认为512
        type_vocab_size=2,  # 类型词汇表大小，默认为2
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-12,  # LayerNorm的epsilon，默认为1e-12
        pad_token_id=0,  # 填充token的ID，默认为0
        position_embedding_type="absolute",  # 位置嵌入类型，默认为绝对位置编码
        classifier_dropout=None,  # 分类器的Dropout，默认为None
        channel_shrink_ratio=4,  # 通道缩小比例，默认为4
        max_2d_position_embeddings=1024,  # 最大二维位置嵌入长度，默认为1024
        **kwargs,  # 其他关键字参数
    ):
        """
        Initializes a new instance of LiltConfig with optional parameters to define the model architecture.

        Parameters:
        - vocab_size: The size of the vocabulary.
        - hidden_size: The size of the hidden layers.
        - num_hidden_layers: The number of hidden layers.
        - num_attention_heads: The number of attention heads in the multi-head attention setups.
        - intermediate_size: The size of the intermediate (i.e., feed-forward) layer in the transformer blocks.
        - hidden_act: The activation function (e.g., "gelu").
        - hidden_dropout_prob: The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        - attention_probs_dropout_prob: The dropout ratio for the attention probabilities.
        - max_position_embeddings: The maximum length of the input sequences.
        - type_vocab_size: The size of the token type vocab.
        - initializer_range: The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        - layer_norm_eps: The epsilon used by LayerNorm layers.
        - pad_token_id: The ID of the padding token.
        - position_embedding_type: The type of position embeddings.
        - classifier_dropout: The dropout ratio for classifier.
        - channel_shrink_ratio: The shrink ratio of channel.
        - max_2d_position_embeddings: The maximum length of the 2D position embeddings.
        - **kwargs: Additional keyword arguments.

        """
        super().__init__(**kwargs)  # 调用父类的初始化方法，传入所有关键字参数
        ):
            # 调用父类的初始化方法，设定填充标记的 ID 和其他可选参数
            super().__init__(pad_token_id=pad_token_id, **kwargs)

            # 设置模型的词汇表大小
            self.vocab_size = vocab_size
            # 设置隐藏层的大小
            self.hidden_size = hidden_size
            # 设置隐藏层的数量
            self.num_hidden_layers = num_hidden_layers
            # 设置注意力头的数量
            self.num_attention_heads = num_attention_heads
            # 设置隐藏层激活函数的类型
            self.hidden_act = hidden_act
            # 设置中间层大小
            self.intermediate_size = intermediate_size
            # 设置隐藏层的 dropout 概率
            self.hidden_dropout_prob = hidden_dropout_prob
            # 设置注意力概率 dropout 概率
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            # 设置最大位置嵌入的大小
            self.max_position_embeddings = max_position_embeddings
            # 设置类型词汇表的大小
            self.type_vocab_size = type_vocab_size
            # 设置初始化范围
            self.initializer_range = initializer_range
            # 设置层归一化的 epsilon 值
            self.layer_norm_eps = layer_norm_eps
            # 设置位置嵌入的类型
            self.position_embedding_type = position_embedding_type
            # 设置分类器 dropout 概率
            self.classifier_dropout = classifier_dropout
            # 设置通道收缩比率
            self.channel_shrink_ratio = channel_shrink_ratio
            # 设置最大二维位置嵌入的大小
            self.max_2d_position_embeddings = max_2d_position_embeddings
```