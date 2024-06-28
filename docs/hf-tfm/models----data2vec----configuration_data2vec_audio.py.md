# `.\models\data2vec\configuration_data2vec_audio.py`

```
# coding=utf-8
# 设置文件编码为 UTF-8，确保支持中文等多种字符集
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
# 版权声明，版权归 HuggingFace Inc. 团队所有
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache License 2.0 版本授权许可，允许复制、修改、发布、分发本软件
# you may not use this file except in compliance with the License.
# 除非遵守许可证，否则不得使用本文件
# You may obtain a copy of the License at
# 可以从以下链接获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# 除非适用法律要求或书面同意，否则不得使用本软件
# distributed under the License is distributed on an "AS IS" BASIS,
# 本软件按“原样”分发，不提供任何形式的担保或条件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 没有明示或暗示的担保或条件
# See the License for the specific language governing permissions and
# 请参阅许可证以获取详细的许可条款
# limitations under the License.
# 许可下的限制
""" Data2VecText configuration"""
# Data2VecText 配置模块说明

import math  # 导入 math 模块

from ...configuration_utils import PretrainedConfig  # 导入预训练配置类
from ...utils import logging  # 导入日志记录工具


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

DATA2VEC_AUDIO_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/data2vec-base-960h": "https://huggingface.co/facebook/data2vec-audio-base-960h/resolve/main/config.json",
    # 预训练模型映射字典，指定模型名称和配置文件的 URL
    # See all Data2VecAudio models at https://huggingface.co/models?filter=data2vec-audio
    # 查看所有 Data2VecAudio 模型的链接
}


class Data2VecAudioConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Data2VecAudioModel`]. It is used to instantiate
    an Data2VecAudio model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Data2VecAudio
    [facebook/data2vec-audio-base-960h](https://huggingface.co/facebook/data2vec-audio-base-960h) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Example:

    ```python
    >>> from transformers import Data2VecAudioConfig, Data2VecAudioModel

    >>> # Initializing a Data2VecAudio facebook/data2vec-audio-base-960h style configuration
    >>> configuration = Data2VecAudioConfig()

    >>> # Initializing a model (with random weights) from the facebook/data2vec-audio-base-960h style configuration
    >>> model = Data2VecAudioModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    # Data2VecAudioConfig 类的说明文档和示例用法

    model_type = "data2vec-audio"  # 模型类型为 data2vec-audio
    # 初始化函数，用于创建一个 Transformer 模型的实例
    def __init__(
        self,
        vocab_size=32,  # 词汇表大小，默认为32
        hidden_size=768,  # 隐藏层大小，默认为768
        num_hidden_layers=12,  # Transformer 模型的隐藏层层数，默认为12
        num_attention_heads=12,  # 注意力头的数量，默认为12
        intermediate_size=3072,  # 中间层的大小，默认为3072
        hidden_act="gelu",  # 隐藏层激活函数，默认为 GELU
        hidden_dropout=0.1,  # 隐藏层的 dropout 比率，默认为0.1
        activation_dropout=0.1,  # 激活函数的 dropout 比率，默认为0.1
        attention_dropout=0.1,  # 注意力层的 dropout 比率，默认为0.1
        feat_proj_dropout=0.0,  # 特征投影层的 dropout 比率，默认为0.0
        final_dropout=0.1,  # 最终输出层的 dropout 比率，默认为0.1
        layerdrop=0.1,  # 层间 dropout 比率，默认为0.1
        initializer_range=0.02,  # 初始化权重范围，默认为0.02
        layer_norm_eps=1e-5,  # Layer normalization 的 epsilon 参数，默认为1e-5
        feat_extract_activation="gelu",  # 特征提取层的激活函数，默认为 GELU
        conv_dim=(512, 512, 512, 512, 512, 512, 512),  # 卷积层的维度列表，默认为指定维度
        conv_stride=(5, 2, 2, 2, 2, 2, 2),  # 卷积层的步幅列表，默认为指定步幅
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),  # 卷积层的核大小列表，默认为指定核大小
        conv_bias=False,  # 是否在卷积层使用偏置，默认为False
        num_conv_pos_embedding_groups=16,  # 卷积位置嵌入的组数，默认为16
        conv_pos_kernel_size=19,  # 卷积位置嵌入的核大小，默认为19
        num_conv_pos_embeddings=5,  # 卷积位置嵌入的数量，默认为5
        mask_time_prob=0.05,  # 时间掩码的概率，默认为0.05
        mask_time_length=10,  # 时间掩码的长度，默认为10
        mask_time_min_masks=2,  # 最小时间掩码数，默认为2
        mask_feature_prob=0.0,  # 特征掩码的概率，默认为0.0
        mask_feature_length=10,  # 特征掩码的长度，默认为10
        mask_feature_min_masks=0,  # 最小特征掩码数，默认为0
        ctc_loss_reduction="sum",  # CTC 损失函数的减少方式，默认为"sum"
        ctc_zero_infinity=False,  # CTC 损失函数是否允许无穷大，默认为False
        use_weighted_layer_sum=False,  # 是否使用加权层求和，默认为False
        classifier_proj_size=256,  # 分类器投影层的大小，默认为256
        tdnn_dim=(512, 512, 512, 512, 1500),  # TDNN 层的维度列表，默认为指定维度
        tdnn_kernel=(5, 3, 3, 1, 1),  # TDNN 层的核大小列表，默认为指定核大小
        tdnn_dilation=(1, 2, 3, 1, 1),  # TDNN 层的膨胀率列表，默认为指定膨胀率
        xvector_output_dim=512,  # x-vector 输出的维度，默认为512
        pad_token_id=0,  # 填充符的 token id，默认为0
        bos_token_id=1,  # 开始符的 token id，默认为1
        eos_token_id=2,  # 结束符的 token id，默认为2
        add_adapter=False,  # 是否添加适配器层，默认为False
        adapter_kernel_size=3,  # 适配器层的核大小，默认为3
        adapter_stride=2,  # 适配器层的步幅，默认为2
        num_adapter_layers=3,  # 适配器层的数量，默认为3
        output_hidden_size=None,  # 输出隐藏层的大小，默认为None
        **kwargs,  # 其它关键字参数
    ):
        # 计算卷积层步幅列表中所有元素的乘积，并返回结果
        @property
        def inputs_to_logits_ratio(self):
            return math.prod(self.conv_stride)
```