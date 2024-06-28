# `.\models\decision_transformer\configuration_decision_transformer.py`

```py
# coding=utf-8
# Copyright 2022 The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
""" Decision Transformer model configuration"""

# 导入必要的库和模块
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 预训练模型及其配置文件的映射字典
DECISION_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "edbeeching/decision-transformer-gym-hopper-medium": (
        "https://huggingface.co/edbeeching/decision-transformer-gym-hopper-medium/resolve/main/config.json"
    ),
    # 查看所有 DecisionTransformer 模型，请访问 https://huggingface.co/models?filter=decision_transformer
}

# DecisionTransformerConfig 类，用于存储 DecisionTransformer 模型的配置信息
class DecisionTransformerConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`DecisionTransformerModel`]. It is used to
    instantiate a Decision Transformer model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the standard
    DecisionTransformer architecture. Many of the config options are used to instatiate the GPT2 model that is used as
    part of the architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Example:

    ```
    >>> from transformers import DecisionTransformerConfig, DecisionTransformerModel

    >>> # Initializing a DecisionTransformer configuration
    >>> configuration = DecisionTransformerConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = DecisionTransformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    # 模型类型
    model_type = "decision_transformer"
    # 推理时忽略的键列表
    keys_to_ignore_at_inference = ["past_key_values"]
    # 属性映射字典，用于调整配置
    attribute_map = {
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }
    # 初始化函数，用于设置模型的各种参数和配置
    def __init__(
        self,
        state_dim=17,  # 状态维度，默认为17
        act_dim=4,  # 动作维度，默认为4
        hidden_size=128,  # 隐藏层大小，默认为128
        max_ep_len=4096,  # 最大的 episode 长度，默认为4096
        action_tanh=True,  # 是否对动作进行 tanh 处理，默认为True
        vocab_size=1,  # 词汇表大小，默认为1
        n_positions=1024,  # 序列位置编码的最大长度，默认为1024
        n_layer=3,  # Transformer 层数，默认为3
        n_head=1,  # 自注意力机制中的头数，默认为1
        n_inner=None,  # FeedForward 层中间层的维度，默认为None
        activation_function="relu",  # 激活函数，默认为 relu
        resid_pdrop=0.1,  # 残差连接中的 dropout 概率，默认为0.1
        embd_pdrop=0.1,  # Embedding 层的 dropout 概率，默认为0.1
        attn_pdrop=0.1,  # 注意力机制中的 dropout 概率，默认为0.1
        layer_norm_epsilon=1e-5,  # Layer Normalization 中的 epsilon，默认为1e-5
        initializer_range=0.02,  # 参数初始化范围，默认为0.02
        scale_attn_weights=True,  # 是否对注意力权重进行缩放，默认为True
        use_cache=True,  # 是否使用缓存，默认为True
        bos_token_id=50256,  # 起始 token 的 id，默认为50256
        eos_token_id=50256,  # 结束 token 的 id，默认为50256
        scale_attn_by_inverse_layer_idx=False,  # 是否根据逆层索引缩放注意力，默认为False
        reorder_and_upcast_attn=False,  # 是否重新排序并提升注意力，默认为False
        **kwargs,
    ):
        self.state_dim = state_dim  # 初始化模型的状态维度
        self.act_dim = act_dim  # 初始化模型的动作维度
        self.hidden_size = hidden_size  # 初始化模型的隐藏层大小
        self.max_ep_len = max_ep_len  # 初始化模型的最大 episode 长度
        self.action_tanh = action_tanh  # 初始化模型的动作是否经过 tanh 处理
        self.vocab_size = vocab_size  # 初始化模型的词汇表大小
        self.n_positions = n_positions  # 初始化模型的序列位置编码的最大长度
        self.n_layer = n_layer  # 初始化模型的 Transformer 层数
        self.n_head = n_head  # 初始化模型的自注意力机制中的头数
        self.n_inner = n_inner  # 初始化模型的 FeedForward 层中间层的维度
        self.activation_function = activation_function  # 初始化模型的激活函数
        self.resid_pdrop = resid_pdrop  # 初始化模型的残差连接中的 dropout 概率
        self.embd_pdrop = embd_pdrop  # 初始化模型的 Embedding 层的 dropout 概率
        self.attn_pdrop = attn_pdrop  # 初始化模型的注意力机制中的 dropout 概率
        self.layer_norm_epsilon = layer_norm_epsilon  # 初始化模型的 Layer Normalization 中的 epsilon
        self.initializer_range = initializer_range  # 初始化模型的参数初始化范围
        self.scale_attn_weights = scale_attn_weights  # 初始化模型的是否对注意力权重进行缩放
        self.use_cache = use_cache  # 初始化模型的是否使用缓存
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx  # 初始化模型的是否根据逆层索引缩放注意力
        self.reorder_and_upcast_attn = reorder_and_upcast_attn  # 初始化模型的是否重新排序并提升注意力

        self.bos_token_id = bos_token_id  # 初始化模型的起始 token 的 id
        self.eos_token_id = eos_token_id  # 初始化模型的结束 token 的 id

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)  # 调用父类的初始化函数，并传递参数
```