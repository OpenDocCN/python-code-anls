# `.\models\openai\configuration_openai.py`

```py
# coding=utf-8
# 设置代码文件的编码格式为 UTF-8

# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# 版权声明，指明代码的版权归属于 OpenAI Team 和 HuggingFace Inc. 团队

# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# 版权声明，指明代码的版权归属于 NVIDIA 公司，保留所有权利

# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache License, Version 2.0 许可协议授权，详见链接

# you may not use this file except in compliance with the License.
# 除非遵循许可协议，否则不得使用此文件

# You may obtain a copy of the License at
# 可以在上述链接获取许可协议的副本

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 除非适用法律要求或书面同意，按"原样"分发软件，无论明示或默示的任何保证或条件

""" OpenAI GPT configuration"""
# 模块说明：OpenAI GPT 的配置信息

from ...configuration_utils import PretrainedConfig
# 导入预训练配置类 PretrainedConfig
from ...utils import logging
# 导入日志工具模块

logger = logging.get_logger(__name__)
# 获取当前模块的日志记录器对象

OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "openai-community/openai-gpt": "https://huggingface.co/openai-community/openai-gpt/resolve/main/config.json"
}
# 预训练配置映射表，指定预训练模型的配置文件下载链接

class OpenAIGPTConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`OpenAIGPTModel`] or a [`TFOpenAIGPTModel`]. It is
    used to instantiate a GPT model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the GPT
    [openai-community/openai-gpt](https://huggingface.co/openai-community/openai-gpt) architecture from OpenAI.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Examples:

    ```
    >>> from transformers import OpenAIGPTConfig, OpenAIGPTModel

    >>> # Initializing a GPT configuration
    >>> configuration = OpenAIGPTConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = OpenAIGPTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """
    # OpenAIGPTConfig 类，用于存储 OpenAI GPT 模型的配置信息

    model_type = "openai-gpt"
    # 模型类型为 openai-gpt

    attribute_map = {
        "max_position_embeddings": "n_positions",
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }
    # 属性映射表，将配置中的属性名映射到模型的相应参数名称

    def __init__(
        self,
        vocab_size=40478,
        n_positions=512,
        n_embd=768,
        n_layer=12,
        n_head=12,
        afn="gelu",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        **kwargs,
    ):
        # 初始化方法，设置模型配置的各个参数

        super().__init__(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            afn=afn,
            resid_pdrop=resid_pdrop,
            embd_pdrop=embd_pdrop,
            attn_pdrop=attn_pdrop,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_range=initializer_range,
            summary_type=summary_type,
            summary_use_proj=summary_use_proj,
            summary_activation=summary_activation,
            summary_proj_to_labels=summary_proj_to_labels,
            summary_first_dropout=summary_first_dropout,
            **kwargs,
        )
        # 调用父类 PretrainedConfig 的初始化方法，设置配置参数
        ):
        # 初始化Transformer模型的参数
        self.vocab_size = vocab_size  # 词汇表大小
        self.n_positions = n_positions  # 序列位置编码的最大长度
        self.n_embd = n_embd  # 嵌入层的维度
        self.n_layer = n_layer  # Transformer模型的层数
        self.n_head = n_head  # 注意力头的数量
        self.afn = afn  # 激活函数
        self.resid_pdrop = resid_pdrop  # 残差连接的dropout率
        self.embd_pdrop = embd_pdrop  # 嵌入层dropout率
        self.attn_pdrop = attn_pdrop  # 注意力层dropout率
        self.layer_norm_epsilon = layer_norm_epsilon  # Layer Norm层的epsilon值
        self.initializer_range = initializer_range  # 参数初始化范围
        self.summary_type = summary_type  # 摘要类型
        self.summary_use_proj = summary_use_proj  # 是否在摘要时使用投影
        self.summary_activation = summary_activation  # 摘要层的激活函数
        self.summary_first_dropout = summary_first_dropout  # 摘要层的第一个dropout率
        self.summary_proj_to_labels = summary_proj_to_labels  # 摘要层是否投影到标签空间
        super().__init__(**kwargs)  # 调用父类初始化方法
```