# `.\models\qwen2\configuration_qwen2.py`

```
# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
""" Qwen2 model configuration"""

from ...configuration_utils import PretrainedConfig  # 导入预训练配置类
from ...utils import logging  # 导入日志工具


logger = logging.get_logger(__name__)  # 获取模块的日志记录器

QWEN2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "Qwen/Qwen2-7B-beta": "https://huggingface.co/Qwen/Qwen2-7B-beta/resolve/main/config.json",
}


class Qwen2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen2Model`]. It is used to instantiate a
    Qwen2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Qwen2-7B-beta [Qwen/Qwen2-7B-beta](https://huggingface.co/Qwen/Qwen2-7B-beta).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    ```python
    >>> from transformers import Qwen2Model, Qwen2Config

    >>> # Initializing a Qwen2 style configuration
    >>> configuration = Qwen2Config()

    >>> # Initializing a model from the Qwen2-7B style configuration
    >>> model = Qwen2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

    model_type = "qwen2"  # 模型类型为 Qwen2
    keys_to_ignore_at_inference = ["past_key_values"]  # 推断时忽略的键列表

    def __init__(
        self,
        vocab_size=151936,  # 词汇表大小，默认为 151936
        hidden_size=4096,  # 隐藏层大小，默认为 4096
        intermediate_size=22016,  # 中间层大小，默认为 22016
        num_hidden_layers=32,  # 隐藏层层数，默认为 32
        num_attention_heads=32,  # 注意力头数，默认为 32
        num_key_value_heads=32,  # 键值头数，默认为 32
        hidden_act="silu",  # 隐藏层激活函数，默认为 silu
        max_position_embeddings=32768,  # 最大位置嵌入数，默认为 32768
        initializer_range=0.02,  # 初始化范围，默认为 0.02
        rms_norm_eps=1e-6,  # RMS 归一化参数，默认为 1e-6
        use_cache=True,  # 是否使用缓存，默认为 True
        tie_word_embeddings=False,  # 是否绑定词嵌入，默认为 False
        rope_theta=10000.0,  # ROPE 参数，默认为 10000.0
        use_sliding_window=False,  # 是否使用滑动窗口，默认为 False
        sliding_window=4096,  # 滑动窗口大小，默认为 4096
        max_window_layers=28,  # 最大窗口层数，默认为 28
        attention_dropout=0.0,  # 注意力机制的 dropout，默认为 0.0
        **kwargs,  # 其他关键字参数
        ):
            # 设置模型的超参数
            self.vocab_size = vocab_size
            self.max_position_embeddings = max_position_embeddings
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.use_sliding_window = use_sliding_window
            self.sliding_window = sliding_window
            self.max_window_layers = max_window_layers

            # 为了向后兼容性
            if num_key_value_heads is None:
                num_key_value_heads = num_attention_heads

            # 设置键值头的数量
            self.num_key_value_heads = num_key_value_heads
            self.hidden_act = hidden_act
            self.initializer_range = initializer_range
            self.rms_norm_eps = rms_norm_eps
            self.use_cache = use_cache
            self.rope_theta = rope_theta
            self.attention_dropout = attention_dropout

            # 调用父类的初始化方法，传入参数和关键字参数
            super().__init__(
                tie_word_embeddings=tie_word_embeddings,
                **kwargs,
            )
```