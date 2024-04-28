# `.\transformers\models\seamless_m4t\configuration_seamless_m4t.py`

```py
# 这是一个 Python 脚本，包含了 SeamlessM4T 模型配置的定义。
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
# 这个注释块包含了脚本的版权信息和许可证协议。

# 导入必要的模块
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义 SeamlessM4T 预训练模型配置的映射
SEAMLESS_M4T_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/hf-seamless-m4t-medium": "https://huggingface.co/facebook/hf-seamless-m4t-medium/resolve/main/config.json",
    # See all SeamlessM4T models at https://huggingface.co/models?filter=seamless_m4t
}

# 定义 SeamlessM4TConfig 类
class SeamlessM4TConfig(PretrainedConfig):
    r"""
    这是一个存储 SeamlessM4T 模型配置的类。它用于根据指定的参数实例化一个 SeamlessM4T 模型，定义模型的架构。
    使用默认值实例化配置将产生与 "facebook/hf-seamless-m4t-medium" 架构类似的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。请参阅 [`PretrainedConfig`] 文档以了解更多信息。

    示例用法:

    ```python
    >>> from transformers import SeamlessM4TModel, SeamlessM4TConfig

    >>> # 初始化一个 "facebook/hf-seamless-m4t-medium" 风格的配置
    >>> configuration = SeamlessM4TConfig()

    >>> # 从 "facebook/hf-seamless-m4t-medium" 风格的配置实例化一个模型
    >>> model = SeamlessM4TModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```py
    """

    # 模型类型为 "seamless_m4t"
    model_type = "seamless_m4t"
    # 初始化函数，用于创建一个新的实例
    def __init__(
        self,
        vocab_size=256102,  # 词汇表大小，默认为256102
        t2u_vocab_size=10082,  # T2U词汇表大小，默认为10082
        # 共享配置
        hidden_size=1024,  # 隐藏层大小，默认为1024
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-5,  # 层归一化的epsilon值，默认为1e-5
        use_cache=True,  # 是否使用缓存，默认为True
        max_position_embeddings=1024,  # 最大位置编码长度，默认为1024
        is_encoder_decoder=True,  # 是否是编码器-解码器模型，默认为True
        encoder_layerdrop=0.05,  # 编码器层dropout率，默认为0.05
        decoder_layerdrop=0.05,  # 解码器层dropout率，默认为0.05
        activation_function="relu",  # 激活函数，默认为relu
        dropout=0.1,  # dropout率，默认为0.1
        attention_dropout=0.1,  # 注意力dropout率，默认为0.1
        activation_dropout=0.0,  # 激活函数dropout率，默认为0.0
        scale_embedding=True,  # 是否缩放嵌入，默认为True
        # 文本编码器|解码器
        encoder_layers=24,  # 编码器层数，默认为24
        encoder_ffn_dim=8192,  # 编码器中FFN层的维度，默认为8192
        encoder_attention_heads=16,  # 编码器中注意力头的数量，默认为16
        decoder_layers=24,  # 解码器层数，默认为24
        decoder_ffn_dim=8192,  # 解码器中FFN层的维度，默认为8192
        decoder_attention_heads=16,  # 解码器中注意力头的数量，默认为16
        decoder_start_token_id=3,  # 解码器起始标记的ID，默认为3
        max_new_tokens=256,  # 最大新标记数量，默认为256
        pad_token_id=0,  # 填充标记的ID，默认为0
        bos_token_id=2,  # 起始标记的ID，默认为2
        eos_token_id=3,  # 终止标记的ID，默认为3
        # 语音编码器
        speech_encoder_layers=24,  # 语音编码器层数，默认为24
        speech_encoder_attention_heads=16,  # 语音编码器中注意力头的数量，默认为16
        speech_encoder_intermediate_size=4096,  # 语音编码器中间层的大小，默认为4096
        speech_encoder_hidden_act="swish",  # 语音编码器中隐藏层激活函数，默认为swish
        speech_encoder_dropout=0.0,  # 语音编码器dropout率，默认为0.0
        add_adapter=True,  # 是否添加适配器，默认为True
        speech_encoder_layerdrop=0.1,  # 语音编码器层dropout率，默认为0.1
        feature_projection_input_dim=160,  # 特征投影的输入维度，默认为160
        num_conv_pos_embeddings=128,  # 卷积位置嵌入的数量，默认为128
        num_conv_pos_embedding_groups=16,  # 卷积位置嵌入的组数，默认为16
        adaptor_kernel_size=8,  # 适配器卷积核大小，默认为8
        adaptor_stride=8,  # 适配器卷积步长，默认为8
        adaptor_dropout=0.1,  # 适配器dropout率，默认为0.1
        num_adapter_layers=1,  # 适配器层数，默认为1
        position_embeddings_type="relative",  # 位置嵌入类型，默认为relative
        rotary_embedding_base=10000,  # 旋转嵌入的基础值，默认为10000
        max_source_positions=4096,  # 最大源位置，默认为4096
        conv_depthwise_kernel_size=31,  # 深度卷积核大小，默认为31
        # T2U配置
        t2u_bos_token_id=0,  # T2U起始标记的ID，默认为0
        t2u_pad_token_id=1,  # T2U填充标记的ID，默认为1
        t2u_eos_token_id=2,  # T2U终止标记的ID，默认为2
        t2u_decoder_start_token_id=2,  # T2U解码器起始标记的ID，默认为2
        t2u_max_new_tokens=1024,  # T2U最大新标记数量，默认为1024
        t2u_encoder_layers=6,  # T2U编码器层数，默认为6
        t2u_encoder_ffn_dim=8192,  # T2U编码器中FFN层的维度，默认为8192
        t2u_encoder_attention_heads=16,  # T2U编码器中注意力头的数量，默认为16
        t2u_decoder_layers=6,  # T2U解码器层数，默认为6
        t2u_decoder_ffn_dim=8192,  # T2U解码器中FFN层的维度，默认为8192
        t2u_decoder_attention_heads=16,  # T2U解码器中注意力头的数量，默认为16
        t2u_max_position_embeddings=2048,  # T2U最大位置编码长度，默认为2048
        # Hifi-GAN声码器配置
        sampling_rate=16000,  # 采样率，默认为16000
        upsample_initial_channel=512,  # 上采样初始通道数，默认为512
        upsample_rates=[5, 4, 4, 2, 2],  # 上采样率列表，默认为[5, 4, 4, 2, 2]
        upsample_kernel_sizes=[11, 8, 8, 4, 4],  # 上采样卷积核大小列表，默认为[11, 8, 8, 4,
```