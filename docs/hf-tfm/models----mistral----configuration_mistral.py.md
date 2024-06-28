# `.\models\mistral\configuration_mistral.py`

```
# coding=utf-8
# Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.
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
""" Mistral model configuration"""

from ...configuration_utils import PretrainedConfig  # 导入预训练配置基类
from ...utils import logging  # 导入日志模块


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器对象

MISTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "mistralai/Mistral-7B-v0.1": "https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/config.json",
    "mistralai/Mistral-7B-Instruct-v0.1": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/resolve/main/config.json",
}

class MistralConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MistralModel`]. It is used to instantiate an
    Mistral model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Mistral-7B-v0.1 or Mistral-7B-Instruct-v0.1.

    [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
    [mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    ```python
    >>> from transformers import MistralModel, MistralConfig

    >>> # Initializing a Mistral 7B style configuration
    >>> configuration = MistralConfig()

    >>> # Initializing a model from the Mistral 7B style configuration
    >>> model = MistralModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "mistral"  # 模型类型为 mistral
    keys_to_ignore_at_inference = ["past_key_values"]  # 推断阶段要忽略的键列表

    def __init__(
        self,
        vocab_size=32000,  # 词汇表大小，默认为 32000
        hidden_size=4096,  # 隐藏层大小，默认为 4096
        intermediate_size=14336,  # 中间层大小，默认为 14336
        num_hidden_layers=32,  # 隐藏层层数，默认为 32
        num_attention_heads=32,  # 注意力头数，默认为 32
        num_key_value_heads=8,  # 键值头数，默认为 8
        hidden_act="silu",  # 隐藏层激活函数，默认为 "silu"
        max_position_embeddings=4096 * 32,  # 最大位置嵌入数，默认为 4096 * 32
        initializer_range=0.02,  # 初始化范围，默认为 0.02
        rms_norm_eps=1e-6,  # RMS 归一化的 epsilon，默认为 1e-6
        use_cache=True,  # 是否使用缓存，默认为 True
        pad_token_id=None,  # 填充标记的 id，默认为 None
        bos_token_id=1,  # 起始标记的 id，默认为 1
        eos_token_id=2,  # 终止标记的 id，默认为 2
        tie_word_embeddings=False,  # 是否绑定词嵌入，默认为 False
        rope_theta=10000.0,  # ROPE 参数，默认为 10000.0
        sliding_window=4096,  # 滑动窗口大小，默认为 4096
        attention_dropout=0.0,  # 注意力层的 dropout 比率，默认为 0.0
        **kwargs,  # 其他关键字参数
    ):
        super().__init__(**kwargs)  # 调用父类的初始化方法
        ):
        # 设置模型的词汇表大小
        self.vocab_size = vocab_size
        # 设置模型的最大位置嵌入数量
        self.max_position_embeddings = max_position_embeddings
        # 设置模型的隐藏层大小
        self.hidden_size = hidden_size
        # 设置模型的中间层大小
        self.intermediate_size = intermediate_size
        # 设置模型的隐藏层数量
        self.num_hidden_layers = num_hidden_layers
        # 设置模型的注意力头数量
        self.num_attention_heads = num_attention_heads
        # 设置模型的滑动窗口大小
        self.sliding_window = sliding_window

        # 为了向后兼容性
        # 如果未提供键值头数量，则使用注意力头数量
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        # 设置模型的键值头数量
        self.num_key_value_heads = num_key_value_heads
        # 设置模型的隐藏层激活函数
        self.hidden_act = hidden_act
        # 设置模型的初始化范围
        self.initializer_range = initializer_range
        # 设置模型的RMS归一化的epsilon值
        self.rms_norm_eps = rms_norm_eps
        # 设置模型是否使用缓存
        self.use_cache = use_cache
        # 设置模型的ROPE theta值
        self.rope_theta = rope_theta
        # 设置模型的注意力dropout率
        self.attention_dropout = attention_dropout

        # 调用父类初始化方法，设置模型的特殊标记ID，并传递额外参数
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
```