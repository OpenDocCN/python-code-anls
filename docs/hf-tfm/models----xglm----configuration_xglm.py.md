# `.\models\xglm\configuration_xglm.py`

```
# coding=utf-8
# Copyright The HuggingFace Inc. team. All rights reserved.
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
""" XGLM model configuration"""

# 导入预训练配置基类和日志工具
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 预训练配置文件映射，指定模型名称到配置文件的映射
XGLM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/xglm-564M": "https://huggingface.co/facebook/xglm-564M/resolve/main/config.json",
    # 查看所有 XGLM 模型信息，请访问 https://huggingface.co/models?filter=xglm
}

# XGLM 模型配置类，继承自预训练配置基类
class XGLMConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`XGLMModel`]. It is used to instantiate an XGLM
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the XGLM
    [facebook/xglm-564M](https://huggingface.co/facebook/xglm-564M) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    # 模型类型设定为 "xglm"
    model_type = "xglm"
    # 在推断时忽略的键列表，这些键不会在推断时使用
    keys_to_ignore_at_inference = ["past_key_values"]
    # 定义一个映射，将模型参数的外部名称映射到内部名称
    attribute_map = {
        "num_attention_heads": "attention_heads",  # 将外部参数 "num_attention_heads" 映射为内部参数 "attention_heads"
        "hidden_size": "d_model",  # 将外部参数 "hidden_size" 映射为内部参数 "d_model"
        "num_hidden_layers": "num_layers",  # 将外部参数 "num_hidden_layers" 映射为内部参数 "num_layers"
    }
    
    # 初始化方法，设置模型的各种参数
    def __init__(
        self,
        vocab_size=256008,  # 词汇表大小，默认为 256008
        max_position_embeddings=2048,  # 最大位置编码，默认为 2048
        d_model=1024,  # 隐藏层大小，默认为 1024
        ffn_dim=4096,  # Feedforward 层的维度，默认为 4096
        num_layers=24,  # 网络层数，默认为 24
        attention_heads=16,  # 注意力头的数量，默认为 16
        activation_function="gelu",  # 激活函数，默认为 "gelu"
        dropout=0.1,  # 普通 dropout 的比例，默认为 0.1
        attention_dropout=0.1,  # 注意力层的 dropout 比例，默认为 0.1
        activation_dropout=0.0,  # 激活函数的 dropout 比例，默认为 0.0
        layerdrop=0.0,  # 层级 dropout 的比例，默认为 0.0
        init_std=0.02,  # 参数初始化的标准差，默认为 0.02
        scale_embedding=True,  # 是否对嵌入进行缩放，默认为 True，如果是，则缩放因子为 sqrt(d_model)
        use_cache=True,  # 是否使用缓存，默认为 True
        decoder_start_token_id=2,  # 解码器起始标记的 id，默认为 2
        pad_token_id=1,  # 填充标记的 id，默认为 1
        bos_token_id=0,  # 起始标记的 id，默认为 0
        eos_token_id=2,  # 结束标记的 id，默认为 2
        **kwargs,  # 其他可变关键字参数
    ):
        self.vocab_size = vocab_size  # 设置词汇表大小
        self.max_position_embeddings = max_position_embeddings  # 设置最大位置编码
        self.d_model = d_model  # 设置隐藏层大小
        self.ffn_dim = ffn_dim  # 设置 Feedforward 层的维度
        self.num_layers = num_layers  # 设置网络层数
        self.attention_heads = attention_heads  # 设置注意力头的数量
        self.activation_function = activation_function  # 设置激活函数
        self.dropout = dropout  # 设置普通 dropout 比例
        self.attention_dropout = attention_dropout  # 设置注意力层的 dropout 比例
        self.activation_dropout = activation_dropout  # 设置激活函数的 dropout 比例
        self.layerdrop = layerdrop  # 设置层级 dropout 比例
        self.init_std = init_std  # 设置参数初始化的标准差
        self.scale_embedding = scale_embedding  # 设置是否缩放嵌入
        self.use_cache = use_cache  # 设置是否使用缓存
    
        # 调用父类的初始化方法，传入特殊的 token id 参数和其他可变关键字参数
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            **kwargs,
        )
```