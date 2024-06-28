# `.\models\deprecated\open_llama\configuration_open_llama.py`

```py
# coding=utf-8
# Copyright 2023 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" Open-Llama model configuration"""

# 导入预训练配置类 PretrainedConfig
from ....configuration_utils import PretrainedConfig
# 导入日志工具
from ....utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义 Open-Llama 预训练模型配置文件映射字典，指定模型名称及其配置文件的 URL
OPEN_LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "s-JoL/Open-Llama-V1": "https://huggingface.co/s-JoL/Open-Llama-V1/blob/main/config.json",
}

# OpenLlamaConfig 类，继承自 PretrainedConfig，用于存储 Open-Llama 模型的配置信息
class OpenLlamaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`OpenLlamaModel`]. It is used to instantiate an
    Open-Llama model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the
    [s-JoL/Open-Llama-V1](https://huggingface.co/s-JoL/Open-Llama-V1).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    # 定义函数的参数及其默认取值
    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Open-Llama 模型的词汇表大小。定义可以表示的不同标记数量，当调用 [`OpenLlamaModel`] 时传递 `inputs_ids`。
        hidden_size (`int`, *optional*, defaults to 4096):
            隐藏表示的维度。
        intermediate_size (`int`, *optional*, defaults to 11008):
            MLP 表示的维度。
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Transformer 编码器中的隐藏层数量。
        num_attention_heads (`int`, *optional*, defaults to 32):
            Transformer 编码器中每个 attention 层的注意力头数量。
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            解码器中的非线性激活函数（函数或字符串）。
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            此模型可能使用的最大序列长度。通常设置一个较大的值（例如 512、1024 或 2048）以防万一。
        initializer_range (`float`, *optional*, defaults to 0.02):
            用于初始化所有权重矩阵的截断正态初始化器的标准偏差。
        rms_norm_eps (`float`, *optional*, defaults to 1e-12):
            rms 归一化层使用的 epsilon。
        use_cache (`bool`, *optional*, defaults to `True`):
            模型是否应返回最后的关键/值注意力（并非所有模型都使用）。只在 `config.is_decoder=True` 时相关。
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
            是否绑定权重嵌入
        rope_scaling (`Dict`, *optional*):
            包含 RoPE 嵌入的缩放配置的字典。目前支持两种缩放策略：线性和动态。它们的缩放因子必须是大于1的浮点数。
            期望格式是 `{"type": 策略名称, "factor": 缩放因子}`。在使用此标志时，不更新 `max_position_embeddings` 到预期的新最大值。
            更多关于这些缩放策略行为的信息，请参阅以下主题：
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/。这是一个实验性功能，可能在未来版本中发生重大 API 更改。
    
    # 示例
    Example:
    
    ```
    >>> from transformers import OpenLlamaModel, OpenLlamaConfig

    >>> # Initializing a Open-Llama open_llama-7b style configuration
    >>> configuration = OpenLlamaConfig()

    >>> # Initializing a model from the open_llama-7b style configuration
    >>> model = OpenLlamaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    # 设定模型类型为 "open-llama"
    model_type = "open-llama"
    
    # 定义初始化方法，接受多个配置参数
    def __init__(
        self,
        vocab_size=100000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        use_memory_efficient_attention=True,
        hidden_dropout_prob=0.1,
        attention_dropout_prob=0.1,
        use_stable_embedding=True,
        shared_input_output_embedding=True,
        rope_scaling=None,
        **kwargs,
    ):
        # 初始化实例变量
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
    
        # 设置注意事项中的 "use_memory_efficient_attention" 参数，若未提供则使用默认值
        self.use_memory_efficient_attention = kwargs.pop(
            "use_memorry_efficient_attention", use_memory_efficient_attention
        )
        
        # 设置隐藏层和注意力层的 dropout 概率
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        
        # 稳定嵌入使用标志
        self.use_stable_embedding = use_stable_embedding
        
        # 共享输入输出嵌入标志
        self.shared_input_output_embedding = shared_input_output_embedding
        
        # "rope_scaling" 是可选参数，用于某些定制操作
        self.rope_scaling = rope_scaling
        
        # 调用内部方法，验证 "rope_scaling" 参数的有效性
        self._rope_scaling_validation()
    
        # 调用父类的初始化方法，传递必要的参数和可能的其他关键字参数
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
    
    # 以下注释标识来自于 transformers.models.llama.configuration_llama.LlamaConfig._rope_scaling_validation
    # (此处应该有 _rope_scaling_validation 方法的具体实现，但在这段代码中未提供)
    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        # 检查是否设置了 `rope_scaling`，如果没有设置则直接返回
        if self.rope_scaling is None:
            return

        # 检查 `rope_scaling` 是否为字典类型且包含两个字段
        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            # 如果不是符合要求的字典类型，则抛出数值错误异常
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        
        # 获取 `rope_scaling` 字典中的 `type` 和 `factor` 字段
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        
        # 检查 `type` 字段是否为 `linear` 或 `dynamic`
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            # 如果不是预期的类型，则抛出数值错误异常
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        
        # 检查 `factor` 字段是否为大于 1 的浮点数
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            # 如果不是预期的浮点数或者小于等于 1，则抛出数值错误异常
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")
```