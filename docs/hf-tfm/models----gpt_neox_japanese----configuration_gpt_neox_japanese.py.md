# `.\models\gpt_neox_japanese\configuration_gpt_neox_japanese.py`

```
# coding=utf-8
# Copyright 2022 ABEJA, Inc. and The HuggingFace Inc. team. All rights reserved.
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
""" GPTNeoX Japanese model configuration"""

# 从相应的库中导入所需的类和函数
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取与当前模块相关联的日志记录器
logger = logging.get_logger(__name__)

# 定义一个字典，映射预训练模型名称到其配置文件的 URL
GPT_NEOX_JAPANESE_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "abeja/gpt-neox-japanese-2.7b": "https://huggingface.co/abeja/gpt-neox-japanese-2.7b/resolve/main/config.json",
}

# 定义一个配置类，用于存储 GPTNeoXJapanese 模型的配置信息
class GPTNeoXJapaneseConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GPTNeoXModelJapanese`]. It is used to instantiate
    a GPTNeoX model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the GPTNeoXJapanese
    [abeja/gpt-neox-japanese-2.7b](https://huggingface.co/abeja/gpt-neox-japanese-2.7b) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information. Default configs is set as 2.7B model
    """
    # 模型类型字符串常量，表示这是一个 GPTNeoXJapanese 模型
    model_type = "gpt_neox_japanese"
    # 初始化函数，用于创建一个新的实例对象
    def __init__(
        self,
        vocab_size=32000,  # 设置词汇表大小，默认为32000
        hidden_size=2560,  # 设置隐藏层大小，默认为2560
        num_hidden_layers=32,  # 设置隐藏层数，默认为32
        num_attention_heads=32,  # 设置注意力头数，默认为32
        intermediate_multiple_size=4,  # 设置中间层大小的倍数，默认为4
        hidden_act="gelu",  # 设置隐藏层激活函数，默认为GELU
        rotary_pct=1.00,  # 设置使用rotary位置嵌入的百分比，默认为100%
        rotary_emb_base=10000,  # 设置rotary位置嵌入的基础值，默认为10000
        max_position_embeddings=2048,  # 设置最大位置嵌入数，默认为2048
        initializer_range=0.02,  # 设置参数初始化范围，默认为0.02
        layer_norm_eps=1e-5,  # 设置层归一化的 epsilon，默认为1e-5
        use_cache=True,  # 设置是否使用缓存，默认为True
        bos_token_id=31996,  # 设置起始标记的 token id，默认为31996
        eos_token_id=31999,  # 设置结束标记的 token id，默认为31999
        attention_dropout=0.1,  # 设置注意力层的 dropout 比例，默认为0.1
        hidden_dropout=0.0,  # 设置隐藏层的 dropout 比例，默认为0.0
        **kwargs,  # 允许传入额外的关键字参数
    ):
        # 调用父类的初始化方法，传递起始标记和结束标记的 token id
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        # 设置实例对象的各个属性
        self.vocab_size = vocab_size  # 初始化词汇表大小
        self.max_position_embeddings = max_position_embeddings  # 初始化最大位置嵌入数
        self.hidden_size = hidden_size  # 初始化隐藏层大小
        self.num_hidden_layers = num_hidden_layers  # 初始化隐藏层数
        self.num_attention_heads = num_attention_heads  # 初始化注意力头数
        self.intermediate_multiple_size = intermediate_multiple_size  # 初始化中间层大小的倍数
        self.hidden_act = hidden_act  # 初始化隐藏层激活函数
        self.rotary_pct = rotary_pct  # 初始化使用rotary位置嵌入的百分比
        self.rotary_emb_base = rotary_emb_base  # 初始化rotary位置嵌入的基础值
        self.initializer_range = initializer_range  # 初始化参数初始化范围
        self.layer_norm_eps = layer_norm_eps  # 初始化层归一化的 epsilon
        self.use_cache = use_cache  # 初始化是否使用缓存
        self.attention_dropout = attention_dropout  # 初始化注意力层的 dropout 比例
        self.hidden_dropout = hidden_dropout  # 初始化隐藏层的 dropout 比例
```