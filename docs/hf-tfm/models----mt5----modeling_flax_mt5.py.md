# `.\models\mt5\modeling_flax_mt5.py`

```
# coding=utf-8
# Copyright 2021 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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
""" Flax mT5 model."""

import jax.numpy as jnp  # 导入 JAX 库的 NumPy 接口

from ...utils import logging  # 导入相对路径下的 logging 模块
from ..t5.modeling_flax_t5 import FlaxT5EncoderModel, FlaxT5ForConditionalGeneration, FlaxT5Model  # 导入 FlaxT5 相关模块
from .configuration_mt5 import MT5Config  # 导入 MT5 模型配置

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

_CONFIG_FOR_DOC = "T5Config"  # 用于文档的配置信息

# Copied from transformers.models.bart.modeling_flax_bart.shift_tokens_right
def shift_tokens_right(input_ids: jnp.ndarray, pad_token_id: int, decoder_start_token_id: int) -> jnp.ndarray:
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = jnp.zeros_like(input_ids)  # 创建一个和 input_ids 形状相同的全零数组
    shifted_input_ids = shifted_input_ids.at[:, 1:].set(input_ids[:, :-1])  # 将 input_ids 向右移动一个位置
    shifted_input_ids = shifted_input_ids.at[:, 0].set(decoder_start_token_id)  # 设置起始位置的 token id 为 decoder_start_token_id

    shifted_input_ids = jnp.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)  # 将 -100 的位置替换为 pad_token_id
    return shifted_input_ids


class FlaxMT5Model(FlaxT5Model):
    r"""
    This class overrides [`FlaxT5Model`]. Please check the superclass for the appropriate documentation alongside usage
    examples.

    Examples:

    ```python
    >>> from transformers import FlaxMT5Model, AutoTokenizer

    >>> model = FlaxMT5Model.from_pretrained("google/mt5-small")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

    >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    >>> summary = "Weiter Verhandlung in Syrien."
    >>> inputs = tokenizer(article, return_tensors="np")

    >>> decoder_input_ids = tokenizer(text_target=summary, return_tensors="np").input_ids

    >>> outputs = model(input_ids=inputs["input_ids"], decoder_input_ids=decoder_input_ids)
    >>> hidden_states = outputs.last_hidden_state
    ```"""

    model_type = "mt5"  # 模型类型为 mt5
    config_class = MT5Config  # 使用 MT5Config 类配置


class FlaxMT5EncoderModel(FlaxT5EncoderModel):
    r"""
    This class overrides [`FlaxT5EncoderModel`]. Please check the superclass for the appropriate documentation
    alongside usage examples.

    Examples:

    ```python
    >>> from transformers import FlaxT5EncoderModel, AutoTokenizer

    >>> model = FlaxT5EncoderModel.from_pretrained("google/mt5-small")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

    >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    >>> summary = "Weiter Verhandlung in Syrien."
    ```
    # 定义一个字符串变量，表示模型类型为 "mt5"
    model_type = "mt5"
    # 定义一个变量，表示配置类为 MT5Config，但未使用该变量
    config_class = MT5Config
# 定义一个用于条件生成的FlaxMT5ForConditionalGeneration类，它继承自FlaxT5ForConditionalGeneration类。
# 请查看超类以获取适当的文档和用法示例。

class FlaxMT5ForConditionalGeneration(FlaxT5ForConditionalGeneration):
    # 类型标识为"mt5"
    model_type = "mt5"
    # 配置类为MT5Config
    config_class = MT5Config
```