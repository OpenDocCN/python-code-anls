# `.\models\wav2vec2\configuration_wav2vec2.py`

```
# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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
Wav2Vec2 model configuration
"""

import functools  # 导入functools模块，提供高阶函数操作工具
import operator   # 导入operator模块，提供函数形式的操作符接口

from ...configuration_utils import PretrainedConfig  # 从全局导入PretrainedConfig类
from ...utils import logging  # 从全局导入logging工具


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器对象

# 定义Wav2Vec2预训练模型配置文件映射字典，包含模型名称和对应的配置文件下载链接
WAV_2_VEC_2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/wav2vec2-base-960h": "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/config.json",
    # 查看所有Wav2Vec2模型，链接见https://huggingface.co/models?filter=wav2vec2
}


class Wav2Vec2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Wav2Vec2Model`]. It is used to instantiate an
    Wav2Vec2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Wav2Vec2
    [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Example:

    ```python
    >>> from transformers import Wav2Vec2Config, Wav2Vec2Model

    >>> # Initializing a Wav2Vec2 facebook/wav2vec2-base-960h style configuration
    >>> configuration = Wav2Vec2Config()

    >>> # Initializing a model (with random weights) from the facebook/wav2vec2-base-960h style configuration
    >>> model = Wav2Vec2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "wav2vec2"  # 指定模型类型为"wav2vec2"
    # 初始化函数，用于创建一个 Transformer 模型的实例
    def __init__(
        self,
        vocab_size=32,  # 词汇表大小，默认为32
        hidden_size=768,  # 隐藏层大小，默认为768
        num_hidden_layers=12,  # Transformer 的隐藏层层数，默认为12
        num_attention_heads=12,  # 注意力头的数量，默认为12
        intermediate_size=3072,  # Feedforward 层的中间层大小，默认为3072
        hidden_act="gelu",  # 隐藏层激活函数，默认为GELU
        hidden_dropout=0.1,  # 隐藏层的Dropout率，默认为0.1
        activation_dropout=0.1,  # 激活函数的Dropout率，默认为0.1
        attention_dropout=0.1,  # 注意力层的Dropout率，默认为0.1
        feat_proj_dropout=0.0,  # 特征投影层的Dropout率，默认为0.0
        feat_quantizer_dropout=0.0,  # 特征量化器的Dropout率，默认为0.0
        final_dropout=0.1,  # 最终输出层的Dropout率，默认为0.1
        layerdrop=0.1,  # 层级Dropout率，默认为0.1
        initializer_range=0.02,  # 参数初始化范围，默认为0.02
        layer_norm_eps=1e-5,  # Layer Norm 的 epsilon 值，默认为1e-5
        feat_extract_norm="group",  # 特征提取的归一化方式，默认为"group"
        feat_extract_activation="gelu",  # 特征提取的激活函数，默认为GELU
        conv_dim=(512, 512, 512, 512, 512, 512, 512),  # 卷积层的维度设置，默认为一个元组
        conv_stride=(5, 2, 2, 2, 2, 2, 2),  # 卷积层的步幅设置，默认为一个元组
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),  # 卷积层的核大小设置，默认为一个元组
        conv_bias=False,  # 是否使用卷积层的偏置，默认为False
        num_conv_pos_embeddings=128,  # 卷积位置嵌入的数量，默认为128
        num_conv_pos_embedding_groups=16,  # 卷积位置嵌入的分组数量，默认为16
        do_stable_layer_norm=False,  # 是否使用稳定的Layer Norm，默认为False
        apply_spec_augment=True,  # 是否应用频谱增强，默认为True
        mask_time_prob=0.05,  # 时间掩码的概率，默认为0.05
        mask_time_length=10,  # 时间掩码的长度，默认为10
        mask_time_min_masks=2,  # 时间掩码的最小数量，默认为2
        mask_feature_prob=0.0,  # 特征掩码的概率，默认为0.0
        mask_feature_length=10,  # 特征掩码的长度，默认为10
        mask_feature_min_masks=0,  # 特征掩码的最小数量，默认为0
        num_codevectors_per_group=320,  # 每组编码向量的数量，默认为320
        num_codevector_groups=2,  # 编码向量组的数量，默认为2
        contrastive_logits_temperature=0.1,  # 对比日志的温度参数，默认为0.1
        num_negatives=100,  # 负样本的数量，默认为100
        codevector_dim=256,  # 编码向量的维度，默认为256
        proj_codevector_dim=256,  # 投影编码向量的维度，默认为256
        diversity_loss_weight=0.1,  # 多样性损失的权重，默认为0.1
        ctc_loss_reduction="sum",  # CTC损失的减少方式，默认为"sum"
        ctc_zero_infinity=False,  # CTC损失中是否使用无穷大，默认为False
        use_weighted_layer_sum=False,  # 是否使用加权层求和，默认为False
        classifier_proj_size=256,  # 分类器投影的大小，默认为256
        tdnn_dim=(512, 512, 512, 512, 1500),  # TDNN 层的维度设置，默认为一个元组
        tdnn_kernel=(5, 3, 3, 1, 1),  # TDNN 层的核大小设置，默认为一个元组
        tdnn_dilation=(1, 2, 3, 1, 1),  # TDNN 层的膨胀率设置，默认为一个元组
        xvector_output_dim=512,  # x-vector 的输出维度，默认为512
        pad_token_id=0,  # 填充标记的ID，默认为0
        bos_token_id=1,  # 起始标记的ID，默认为1
        eos_token_id=2,  # 结束标记的ID，默认为2
        add_adapter=False,  # 是否添加适配器，默认为False
        adapter_kernel_size=3,  # 适配器的核大小，默认为3
        adapter_stride=2,  # 适配器的步幅，默认为2
        num_adapter_layers=3,  # 适配器的层数，默认为3
        output_hidden_size=None,  # 输出隐藏层的大小，默认为None
        adapter_attn_dim=None,  # 适配器的注意力维度，默认为None
        **kwargs,  # 其他关键字参数
    ):
        # 计算输入到 Logits 的比例，使用 functools.reduce 和 operator.mul 函数
        @property
        def inputs_to_logits_ratio(self):
            return functools.reduce(operator.mul, self.conv_stride, 1)
```