# `.\models\markuplm\configuration_markuplm.py`

```
# coding=utf-8
# Copyright 2021, The Microsoft Research Asia MarkupLM Team authors
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
MarkupLM model configuration
"""

# 从配置工具中导入预训练配置类 PretrainedConfig
from ...configuration_utils import PretrainedConfig
# 从工具中导入日志记录功能
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义预训练模型与其配置文件的映射字典
MARKUPLM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/markuplm-base": "https://huggingface.co/microsoft/markuplm-base/resolve/main/config.json",
    "microsoft/markuplm-large": "https://huggingface.co/microsoft/markuplm-large/resolve/main/config.json",
}

# 定义 MarkupLMConfig 类，继承自 PretrainedConfig 类
class MarkupLMConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MarkupLMModel`]. It is used to instantiate a
    MarkupLM model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the MarkupLM
    [microsoft/markuplm-base](https://huggingface.co/microsoft/markuplm-base) architecture.

    Configuration objects inherit from [`BertConfig`] and can be used to control the model outputs. Read the
    documentation from [`BertConfig`] for more information.

    Examples:

    ```python
    >>> from transformers import MarkupLMModel, MarkupLMConfig

    >>> # Initializing a MarkupLM microsoft/markuplm-base style configuration
    >>> configuration = MarkupLMConfig()

    >>> # Initializing a model from the microsoft/markuplm-base style configuration
    >>> model = MarkupLMModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    # 模型类型为 "markuplm"
    model_type = "markuplm"

    # 初始化函数，设置了多个模型配置参数
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
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
        pad_token_id=0,
        bos_token_id=0,
        eos_token_id=2,
        max_xpath_tag_unit_embeddings=256,
        max_xpath_subs_unit_embeddings=1024,
        tag_pad_id=216,
        subs_pad_id=1001,
        xpath_unit_hidden_size=32,
        max_depth=50,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        **kwargs,
        ):
        # 调用父类的初始化方法，传入相关参数和关键字参数
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        # 初始化模型的词汇表大小
        self.vocab_size = vocab_size
        # 初始化模型的隐藏层大小
        self.hidden_size = hidden_size
        # 初始化模型的隐藏层数量
        self.num_hidden_layers = num_hidden_layers
        # 初始化模型的注意力头数量
        self.num_attention_heads = num_attention_heads
        # 初始化模型的隐藏层激活函数
        self.hidden_act = hidden_act
        # 初始化模型的中间层大小
        self.intermediate_size = intermediate_size
        # 初始化模型的隐藏层丢弃率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 初始化模型的注意力丢弃率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 初始化模型的最大位置嵌入长度
        self.max_position_embeddings = max_position_embeddings
        # 初始化模型的类型词汇表大小
        self.type_vocab_size = type_vocab_size
        # 初始化模型的初始化范围
        self.initializer_range = initializer_range
        # 初始化模型的层归一化 epsilon
        self.layer_norm_eps = layer_norm_eps
        # 初始化模型的位置嵌入类型
        self.position_embedding_type = position_embedding_type
        # 初始化模型是否使用缓存
        self.use_cache = use_cache
        # 初始化模型分类器的丢弃率
        self.classifier_dropout = classifier_dropout
        # 额外的属性
        # 初始化模型的最大深度
        self.max_depth = max_depth
        # 初始化模型的最大XPath标签单元嵌入
        self.max_xpath_tag_unit_embeddings = max_xpath_tag_unit_embeddings
        # 初始化模型的最大XPath子项单元嵌入
        self.max_xpath_subs_unit_embeddings = max_xpath_subs_unit_embeddings
        # 初始化模型标签填充符的ID
        self.tag_pad_id = tag_pad_id
        # 初始化模型子项填充符的ID
        self.subs_pad_id = subs_pad_id
        # 初始化模型XPath单元隐藏层大小
        self.xpath_unit_hidden_size = xpath_unit_hidden_size
```