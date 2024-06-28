# `.\models\falcon\configuration_falcon.py`

```
# coding=utf-8
# Copyright 2023 the Falcon authors and HuggingFace Inc. team.  All rights reserved.
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
Falcon configuration
"""
# 从 Transformers 库导入基类 PretrainedConfig
from ...configuration_utils import PretrainedConfig
# 从 Transformers 库导入日志工具
from ...utils import logging

# 获取 logger 对象，用于记录日志
logger = logging.get_logger(__name__)

# Falcon 模型预训练配置文件的映射字典，指定了模型名称与配置文件的 URL 地址
FALCON_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "tiiuae/falcon-40b": "https://huggingface.co/tiiuae/falcon-40b/resolve/main/config.json",
    "tiiuae/falcon-7b": "https://huggingface.co/tiiuae/falcon-7b/resolve/main/config.json",
}

# FalconConfig 类，继承自 PretrainedConfig 类，用于存储 Falcon 模型的配置信息
class FalconConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FalconModel`]. It is used to instantiate a Falcon
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the
    [tiiuae/falcon-7b](https://huggingface.co/tiiuae/falcon-7b) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Example:

    ```python
    >>> from transformers import FalconModel, FalconConfig

    >>> # Initializing a small (2-layer) Falcon configuration
    >>> configuration = FalconConfig(num_hidden_layers=2)

    >>> # Initializing a model from the small configuration
    >>> model = FalconModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    # 模型类型标识为 "falcon"
    model_type = "falcon"
    # 推断时忽略的关键字列表
    keys_to_ignore_at_inference = ["past_key_values"]

    # FalconConfig 类的初始化方法，定义了模型的各种配置参数
    def __init__(
        self,
        vocab_size=65024,
        hidden_size=4544,
        num_hidden_layers=32,
        num_attention_heads=71,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        num_kv_heads=None,
        alibi=False,
        new_decoder_architecture=False,
        multi_query=True,
        parallel_attn=True,
        bias=False,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        rope_scaling=None,
        bos_token_id=11,
        eos_token_id=11,
        **kwargs,
    ):
        # 调用父类 PretrainedConfig 的初始化方法，设置模型配置的基本参数
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_range=initializer_range,
            use_cache=use_cache,
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
            num_kv_heads=num_kv_heads,
            alibi=alibi,
            new_decoder_architecture=new_decoder_architecture,
            multi_query=multi_query,
            parallel_attn=parallel_attn,
            bias=bias,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        ):
        self.vocab_size = vocab_size
        # Backward compatibility with n_embed kwarg
        n_embed = kwargs.pop("n_embed", None)
        # 设置隐藏层大小，如果未指定 n_embed 则使用 hidden_size
        self.hidden_size = hidden_size if n_embed is None else n_embed
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        # 如果 num_kv_heads 未指定，则使用 num_attention_heads
        self.num_kv_heads = num_attention_heads if num_kv_heads is None else num_kv_heads
        self.alibi = alibi
        self.new_decoder_architecture = new_decoder_architecture
        # 当 new_decoder_architecture 为 True 时，忽略 multi_query
        self.multi_query = multi_query  # Ignored when new_decoder_architecture is True
        self.parallel_attn = parallel_attn
        self.bias = bias
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        # 运行 _rope_scaling_validation 方法验证 rope_scaling 的设置
        self._rope_scaling_validation()

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

    @property
    def head_dim(self):
        # 返回每个注意力头的维度
        return self.hidden_size // self.num_attention_heads

    @property
    def rotary(self):
        # 如果 alibi 为 False，则返回 True，表示支持旋转注意力
        return not self.alibi

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if self.alibi:
            # 当 alibi 为 True 时，不支持 rope_scaling，抛出异常
            raise ValueError("`rope_scaling` is not supported when `alibi` is `True`.")

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            # rope_scaling 必须是包含 type 和 factor 两个字段的字典
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            # type 字段必须是 ['linear', 'dynamic'] 中的一个
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            # factor 字段必须是大于 1 的浮点数
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")
```