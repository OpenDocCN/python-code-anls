# `.\models\luke\configuration_luke.py`

```py
# coding=utf-8
# Copyright Studio Ousia and The HuggingFace Inc. team.
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

""" LUKE configuration"""

from ...configuration_utils import PretrainedConfig  # 导入预训练配置类
from ...utils import logging  # 导入日志工具

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# LUKE 预训练配置与其配置文件的映射字典
LUKE_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "studio-ousia/luke-base": "https://huggingface.co/studio-ousia/luke-base/resolve/main/config.json",
    "studio-ousia/luke-large": "https://huggingface.co/studio-ousia/luke-large/resolve/main/config.json",
}


class LukeConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LukeModel`]. It is used to instantiate a LUKE
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LUKE
    [studio-ousia/luke-base](https://huggingface.co/studio-ousia/luke-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Examples:

    ```
    >>> from transformers import LukeConfig, LukeModel

    >>> # Initializing a LUKE configuration
    >>> configuration = LukeConfig()

    >>> # Initializing a model from the configuration
    >>> model = LukeModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

    """

    # 模型类型标识为 LUKE
    model_type = "luke"

    def __init__(
        self,
        vocab_size=50267,
        entity_vocab_size=500000,
        hidden_size=768,
        entity_emb_size=256,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_entity_aware_attention=True,
        classifier_dropout=None,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        **kwargs,
    ):
        """
        Initialize a LUKE configuration with default values.

        Args:
            vocab_size (int): Size of the token vocabulary.
            entity_vocab_size (int): Size of the entity vocabulary.
            hidden_size (int): Size of the encoder layers and the pooler layer.
            entity_emb_size (int): Dimensionality of the entity embeddings.
            num_hidden_layers (int): Number of hidden layers in the Transformer encoder.
            num_attention_heads (int): Number of attention heads for each attention layer in the Transformer encoder.
            intermediate_size (int): Size of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
            hidden_act (str): The non-linear activation function (function or string) in the encoder and pooler.
            hidden_dropout_prob (float): The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob (float): The dropout ratio for the attention probabilities.
            max_position_embeddings (int): The maximum sequence length that this model might ever be used with.
            type_vocab_size (int): The vocabulary size of the "type" (i.e., token type IDs) embeddings.
            initializer_range (float): The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            layer_norm_eps (float): The epsilon used by the layer normalization layers.
            use_entity_aware_attention (bool): Whether to use entity-aware attention in the model.
            classifier_dropout (float or None): The dropout probability for the classifier layer (None means no dropout).
            pad_token_id (int): The ID of the padding token in the token vocabulary.
            bos_token_id (int): The ID of the beginning-of-sequence token in the token vocabulary.
            eos_token_id (int): The ID of the end-of-sequence token in the token vocabulary.
            **kwargs: Additional configuration arguments.

        """
        # 调用父类 PretrainedConfig 的初始化方法，传递所有参数
        super().__init__(
            vocab_size=vocab_size,
            entity_vocab_size=entity_vocab_size,
            hidden_size=hidden_size,
            entity_emb_size=entity_emb_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        # 是否使用实体感知注意力机制
        self.use_entity_aware_attention = use_entity_aware_attention

        # 分类器层的 dropout 概率
        self.classifier_dropout = classifier_dropout
        """
        Constructs LukeConfig.
        """
        # 调用父类的初始化方法，设置特定的配置参数
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        # 设置配置对象的词汇表大小
        self.vocab_size = vocab_size
        # 设置配置对象的实体词汇表大小
        self.entity_vocab_size = entity_vocab_size
        # 设置配置对象的隐藏层大小
        self.hidden_size = hidden_size
        # 设置配置对象的实体嵌入大小
        self.entity_emb_size = entity_emb_size
        # 设置配置对象的隐藏层数量
        self.num_hidden_layers = num_hidden_layers
        # 设置配置对象的注意力头数量
        self.num_attention_heads = num_attention_heads
        # 设置配置对象的隐藏层激活函数类型
        self.hidden_act = hidden_act
        # 设置配置对象的中间层大小
        self.intermediate_size = intermediate_size
        # 设置配置对象的隐藏层丢弃率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 设置配置对象的注意力概率丢弃率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 设置配置对象的最大位置嵌入长度
        self.max_position_embeddings = max_position_embeddings
        # 设置配置对象的类型词汇表大小
        self.type_vocab_size = type_vocab_size
        # 设置配置对象的初始化范围
        self.initializer_range = initializer_range
        # 设置配置对象的层归一化 epsilon 值
        self.layer_norm_eps = layer_norm_eps
        # 设置是否使用实体感知注意力机制的标志
        self.use_entity_aware_attention = use_entity_aware_attention
        # 设置分类器层的丢弃率
        self.classifier_dropout = classifier_dropout
```