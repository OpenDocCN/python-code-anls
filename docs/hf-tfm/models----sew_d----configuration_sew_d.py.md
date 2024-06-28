# `.\models\sew_d\configuration_sew_d.py`

```py
# coding=utf-8
# 上面是指定源代码文件的编码格式为 UTF-8

# 版权声明及许可协议，指明此代码的版权信息和使用许可
# Copyright 2021 ASAPP Inc. and The HuggingFace Inc. team. All rights reserved.
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

""" SEW-D model configuration"""
# 以上是模块的简短描述，指明本文件用于SEW-D模型的配置

# 导入 functools 和 operator 模块
import functools
import operator

# 从相关的模块导入 PretrainedConfig 类
from ...configuration_utils import PretrainedConfig
# 导入日志记录工具
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# SEW-D 预训练模型配置文件映射表，将模型名称映射到其配置文件的 URL
SEW_D_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "asapp/sew-d-tiny-100k": "https://huggingface.co/asapp/sew-d-tiny-100k/resolve/main/config.json",
    # 查看所有 SEW-D 模型的列表可访问 https://huggingface.co/models?filter=sew-d
}

# SEWDConfig 类，继承自 PretrainedConfig 类
class SEWDConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SEWDModel`]. It is used to instantiate a SEW-D
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the SEW-D
    [asapp/sew-d-tiny-100k](https://huggingface.co/asapp/sew-d-tiny-100k) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Example:

    ```
    >>> from transformers import SEWDConfig, SEWDModel

    >>> # Initializing a SEW-D asapp/sew-d-tiny-100k style configuration
    >>> configuration = SEWDConfig()

    >>> # Initializing a model (with random weights) from the asapp/sew-d-tiny-100k style configuration
    >>> model = SEWDModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    # SEWDConfig 类的文档字符串，描述了 SEWDConfig 的配置信息和使用方法

    # 模型类型为 "sew-d"
    model_type = "sew-d"
    # 初始化函数，设置模型的各种参数和默认值
    def __init__(
        self,
        vocab_size=32,  # 词汇表大小，默认为32
        hidden_size=768,  # 隐藏层大小，默认为768
        num_hidden_layers=12,  # Transformer中隐藏层的数量，默认为12
        num_attention_heads=12,  # 注意力头的数量，默认为12
        intermediate_size=3072,  # 中间层大小，默认为3072
        squeeze_factor=2,  # 压缩因子，默认为2
        max_position_embeddings=512,  # 最大位置嵌入数，默认为512
        position_buckets=256,  # 位置桶的数量，默认为256
        share_att_key=True,  # 是否共享注意力的键，默认为True
        relative_attention=True,  # 是否使用相对注意力，默认为True
        pos_att_type=("p2c", "c2p"),  # 位置注意力类型，默认为("p2c", "c2p")
        norm_rel_ebd="layer_norm",  # 相对位置编码的规范化方法，默认为"layer_norm"
        hidden_act="gelu_python",  # 隐藏层激活函数，默认为"gelu_python"
        hidden_dropout=0.1,  # 隐藏层的dropout率，默认为0.1
        activation_dropout=0.1,  # 激活函数的dropout率，默认为0.1
        attention_dropout=0.1,  # 注意力机制的dropout率，默认为0.1
        feat_proj_dropout=0.0,  # 特征投影的dropout率，默认为0.0
        final_dropout=0.1,  # 最终输出的dropout率，默认为0.1
        initializer_range=0.02,  # 参数初始化的范围，默认为0.02
        layer_norm_eps=1e-7,  # 层归一化的epsilon值，默认为1e-7
        feature_layer_norm_eps=1e-5,  # 特征层归一化的epsilon值，默认为1e-5
        feat_extract_norm="group",  # 特征提取的归一化方法，默认为"group"
        feat_extract_activation="gelu",  # 特征提取的激活函数，默认为"gelu"
        conv_dim=(64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512),  # 卷积层的通道数，默认为指定的元组
        conv_stride=(5, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1),  # 卷积层的步长，默认为指定的元组
        conv_kernel=(10, 3, 1, 3, 1, 3, 1, 3, 1, 2, 1, 2, 1),  # 卷积层的核大小，默认为指定的元组
        conv_bias=False,  # 卷积层是否使用偏置，默认为False
        num_conv_pos_embeddings=128,  # 卷积位置嵌入的数量，默认为128
        num_conv_pos_embedding_groups=16,  # 卷积位置嵌入的分组数量，默认为16
        apply_spec_augment=True,  # 是否应用特定的数据增强方法，默认为True
        mask_time_prob=0.05,  # 时间掩码的概率，默认为0.05
        mask_time_length=10,  # 时间掩码的长度，默认为10
        mask_time_min_masks=2,  # 时间掩码的最小数量，默认为2
        mask_feature_prob=0.0,  # 特征掩码的概率，默认为0.0
        mask_feature_length=10,  # 特征掩码的长度，默认为10
        mask_feature_min_masks=0,  # 特征掩码的最小数量，默认为0
        ctc_loss_reduction="mean",  # CTC损失的减少方法，默认为"mean"
        ctc_zero_infinity=False,  # CTC损失是否将无穷值设置为零，默认为False
        use_weighted_layer_sum=False,  # 是否使用加权层的总和，默认为False
        classifier_proj_size=256,  # 分类器投影大小，默认为256
        pad_token_id=0,  # 填充token的ID，默认为0
        bos_token_id=1,  # 开始token的ID，默认为1
        eos_token_id=2,  # 结束token的ID，默认为2
        **kwargs,
    ):
    
    # 计算输入到logits的比率，通过减少conv_stride的乘积来实现
    @property
    def inputs_to_logits_ratio(self):
        return functools.reduce(operator.mul, self.conv_stride, 1)
    
    # 获取隐藏层dropout的值，并发出警告
    @property
    def hidden_dropout(self):
        logger.warning_once("hidden_dropout is not used by the model and will be removed as config attribute in v4.35")
        return self._hidden_dropout
    
    # 将实例序列化为Python字典的方法
    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.
        """
        # 调用父类的to_dict方法，获取基类的字典表示
        output = super().to_dict()
        # 将_hidden_dropout键改为hidden_dropout，并将其值存入字典中
        output["hidden_dropout"] = output.pop("_hidden_dropout")
        return output
```