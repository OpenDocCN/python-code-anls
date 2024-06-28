# `.\models\fuyu\configuration_fuyu.py`

```
# coding=utf-8
# Copyright 2023 Adept AI and the HuggingFace Inc. team. All rights reserved.
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
""" Fuyu model configuration"""

# 导入所需的模块和类
from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 预训练模型配置文件映射，指定模型名称和其对应的配置文件链接
FUYU_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "adept/fuyu-8b": "https://huggingface.co/adept/fuyu-8b/resolve/main/config.json",
}

# 定义 FuyuConfig 类，继承自 PretrainedConfig 类
class FuyuConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FuyuForCausalLM`]. It is used to instantiate an
    Fuyu model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the
    [adept/fuyu-8b](https://huggingface.co/adept/fuyu-8b).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    ```python
    >>> from transformers import FuyuConfig

    >>> # Initializing a Fuyu fuyu-7b style configuration
    >>> configuration = FuyuConfig()
    ```
    """

    # 指定模型类型为 "fuyu"
    model_type = "fuyu"
    # 推理阶段忽略的键列表，这些键对于推断阶段不起作用
    keys_to_ignore_at_inference = ["past_key_values"]

    # 初始化方法，定义了 FuyuConfig 类的各种配置参数
    def __init__(
        self,
        vocab_size=262144,
        hidden_size=4096,
        intermediate_size=16384,
        num_hidden_layers=36,
        num_attention_heads=64,
        hidden_act="relu2",
        max_position_embeddings=16384,
        image_size=300,
        patch_size=30,
        num_channels=3,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=25000.0,
        rope_scaling=None,
        qk_layernorm=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        partial_rotary_factor=0.5,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        text_config=None,
        **kwargs,
    ):
        # 调用父类的初始化方法，设置模型配置的基本参数
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        # 额外的模型配置参数
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.qk_layernorm = qk_layernorm
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.partial_rotary_factor = partial_rotary_factor
        self.text_config = text_config
        ):
            # 如果 text_config 参数为 None，则使用默认数值初始化文本模型配置字典
            if text_config is None:
                text_config = {
                    "vocab_size": vocab_size,
                    "max_position_embeddings": max_position_embeddings,
                    "hidden_size": hidden_size,
                    "intermediate_size": intermediate_size,
                    "num_hidden_layers": num_hidden_layers,
                    "num_attention_heads": num_attention_heads,
                    "hidden_act": hidden_act,
                    "initializer_range": initializer_range,
                    "layer_norm_eps": layer_norm_eps,
                    "use_cache": use_cache,
                    "rope_theta": rope_theta,
                    "rope_scaling": rope_scaling,
                    "qk_layernorm": qk_layernorm,
                    "hidden_dropout": hidden_dropout,
                    "attention_dropout": attention_dropout,
                    "partial_rotary_factor": partial_rotary_factor,
                    "pad_token_id": pad_token_id,
                    "bos_token_id": bos_token_id,
                    "eos_token_id": eos_token_id,
                    "tie_word_embeddings": tie_word_embeddings,
                }
                # 记录日志，指示 text_config 为 None，使用默认值初始化文本模型
                logger.info("text_config is None. initializing the text model with default values.")
            
            # 如果 text_config 中包含 "model_type" 键，将其值赋给 text_model_type；否则使用默认值 "persimmon"
            text_model_type = text_config["model_type"] if "model_type" in text_config else "persimmon"
            # 使用 CONFIG_MAPPING 中相应的类初始化 self.text_config，传入 text_config 的全部参数
            self.text_config = CONFIG_MAPPING[text_model_type](**text_config)

            # 将参数赋给当前对象的属性
            self.vocab_size = vocab_size
            self.max_position_embeddings = max_position_embeddings
            self.image_size = image_size
            self.patch_size = patch_size
            self.num_channels = num_channels
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
            self.use_cache = use_cache
            self.rope_theta = rope_theta
            self.rope_scaling = rope_scaling
            self.qk_layernorm = qk_layernorm
            self.hidden_dropout = hidden_dropout
            self.attention_dropout = attention_dropout
            self.partial_rotary_factor = partial_rotary_factor

            # 调用 _rope_scaling_validation 方法，确保 rope_scaling 参数有效性
            self._rope_scaling_validation()

            # 调用父类的初始化方法，传入部分参数，完成对象初始化
            super().__init__(
                pad_token_id=pad_token_id,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                tie_word_embeddings=tie_word_embeddings,
                **kwargs,
            )

        # 从 transformers.models.llama.configuration_llama.LlamaConfig._rope_scaling_validation 复制
    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        # 检查是否设置了 `rope_scaling`，如果未设置则直接返回
        if self.rope_scaling is None:
            return

        # 检查 `rope_scaling` 是否为字典类型，并且包含两个字段
        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            # 如果不符合要求，抛出 ValueError 异常
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        
        # 获取 `rope_scaling` 中的 `type` 和 `factor` 字段的值
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        
        # 检查 `type` 字段是否为有效的值（'linear' 或 'dynamic'）
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            # 如果不是有效的值，抛出 ValueError 异常
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        
        # 检查 `factor` 字段是否为浮点数且大于 1
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            # 如果不是符合要求的值，抛出 ValueError 异常
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")
```