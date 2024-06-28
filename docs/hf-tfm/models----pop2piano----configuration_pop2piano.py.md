# `.\models\pop2piano\configuration_pop2piano.py`

```
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
""" Pop2Piano model configuration"""


from ...configuration_utils import PretrainedConfig  # 导入预训练配置的工具类
from ...utils import logging  # 导入日志工具类


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

# 预训练配置模型的映射字典，将模型名称映射到预训练配置文件的 URL
POP2PIANO_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "sweetcocoa/pop2piano": "https://huggingface.co/sweetcocoa/pop2piano/blob/main/config.json"
}


class Pop2PianoConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Pop2PianoForConditionalGeneration`]. It is used
    to instantiate a Pop2PianoForConditionalGeneration model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the
    Pop2Piano [sweetcocoa/pop2piano](https://huggingface.co/sweetcocoa/pop2piano) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    # 定义 `Pop2PianoForConditionalGeneration` 模型的词汇表大小，默认为 2400
    # `inputs_ids` 调用时传入的不同令牌数量，用于 `Pop2PianoForConditionalGeneration`
    vocab_size = 2400
    
    # 定义作曲家的数量，默认为 21
    composer_vocab_size = 21
    
    # 定义编码器层和池化层的大小，默认为 512
    d_model = 512
    
    # 定义每个注意力头中键、查询、值投影的大小，默认为 64
    # 投影层的 `inner_dim` 将被定义为 `num_heads * d_kv`
    d_kv = 64
    
    # 定义每个 `Pop2PianoBlock` 中中间前馈层的大小，默认为 2048
    d_ff = 2048
    
    # 定义Transformer编码器中隐藏层的数量，默认为 6
    num_layers = 6
    
    # 定义Transformer解码器中隐藏层的数量，默认与 `num_layers` 相同
    # 若未设置，将与 `num_layers` 使用相同的值
    num_decoder_layers = None
    
    # 定义Transformer编码器中每个注意力层的注意力头数量，默认为 8
    num_heads = 8
    
    # 定义每个注意力层使用的桶数量，默认为 32
    relative_attention_num_buckets = 32
    
    # 定义用于桶分离的较长序列的最大距离，默认为 128
    relative_attention_max_distance = 128
    
    # 定义所有dropout层的比率，默认为 0.1
    dropout_rate = 0.1
    
    # 定义层归一化层使用的 epsilon 值，默认为 1e-6
    layer_norm_epsilon = 1e-6
    
    # 初始化所有权重矩阵的因子，默认为 1.0
    # 用于初始化测试内部使用，通常应保持为 1.0
    initializer_factor = 1.0
    
    # 定义要使用的前馈层类型，默认为 `"gated-gelu"`
    # 应为 `"relu"` 或 `"gated-gelu"` 之一
    feed_forward_proj = "gated-gelu"
    
    # 模型是否应返回最后的键/值注意力，默认为 `True`
    # 并非所有模型都使用此选项
    use_cache = True
    
    # 定义在 `Pop2PianoDenseActDense` 和 `Pop2PianoDenseGatedActDense` 中使用的激活函数类型，默认为 `"relu"`
    dense_act_fn = "relu"
    
    # 模型类型设置为 `"pop2piano"`
    model_type = "pop2piano"
    
    # 在推断时忽略的键列表，默认包含 `"past_key_values"`
    keys_to_ignore_at_inference = ["past_key_values"]
    # 初始化函数，用于初始化一个自定义的Transformer模型配置
    def __init__(
        self,
        vocab_size=2400,  # 词汇表大小，默认为2400
        composer_vocab_size=21,  # 作曲家词汇表大小，默认为21
        d_model=512,  # Transformer模型的隐藏层维度，默认为512
        d_kv=64,  # 注意力机制中key和value的维度，默认为64
        d_ff=2048,  # Feed Forward网络中间层的维度，默认为2048
        num_layers=6,  # Transformer模型中的层数，默认为6
        num_decoder_layers=None,  # 解码器层数，如果为None则与num_layers相同
        num_heads=8,  # 多头注意力机制中的头数，默认为8
        relative_attention_num_buckets=32,  # 相对位置编码中的桶数，默认为32
        relative_attention_max_distance=128,  # 相对位置编码的最大距离，默认为128
        dropout_rate=0.1,  # Dropout的比率，默认为0.1
        layer_norm_epsilon=1e-6,  # Layer Normalization中的epsilon，默认为1e-6
        initializer_factor=1.0,  # 初始化因子，默认为1.0
        feed_forward_proj="gated-gelu",  # 前向传播的激活函数，默认为"gated-gelu"
        is_encoder_decoder=True,  # 是否是编码器-解码器模型，默认为True
        use_cache=True,  # 是否使用缓存，默认为True
        pad_token_id=0,  # 填充token的ID，默认为0
        eos_token_id=1,  # 结束token的ID，默认为1
        dense_act_fn="relu",  # Dense层的激活函数，默认为"relu"
        **kwargs,  # 其他参数
    ):
        self.vocab_size = vocab_size  # 初始化词汇表大小
        self.composer_vocab_size = composer_vocab_size  # 初始化作曲家词汇表大小
        self.d_model = d_model  # 初始化隐藏层维度
        self.d_kv = d_kv  # 初始化key和value的维度
        self.d_ff = d_ff  # 初始化Feed Forward网络中间层的维度
        self.num_layers = num_layers  # 初始化Transformer模型中的层数
        self.num_decoder_layers = num_decoder_layers if num_decoder_layers is not None else self.num_layers  # 初始化解码器层数
        self.num_heads = num_heads  # 初始化多头注意力机制中的头数
        self.relative_attention_num_buckets = relative_attention_num_buckets  # 初始化相对位置编码中的桶数
        self.relative_attention_max_distance = relative_attention_max_distance  # 初始化相对位置编码的最大距离
        self.dropout_rate = dropout_rate  # 初始化Dropout的比率
        self.layer_norm_epsilon = layer_norm_epsilon  # 初始化Layer Normalization中的epsilon
        self.initializer_factor = initializer_factor  # 初始化初始化因子
        self.feed_forward_proj = feed_forward_proj  # 初始化前向传播的激活函数
        self.use_cache = use_cache  # 初始化是否使用缓存
        self.dense_act_fn = dense_act_fn  # 初始化Dense层的激活函数
        self.is_gated_act = self.feed_forward_proj.split("-")[0] == "gated"  # 检查是否是gated激活函数
        self.hidden_size = self.d_model  # 初始化隐藏层大小为模型维度
        self.num_attention_heads = num_heads  # 初始化注意力头数
        self.num_hidden_layers = num_layers  # 初始化隐藏层的数量

        # 调用父类的初始化方法，设置pad_token_id、eos_token_id、is_encoder_decoder等参数
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )
```