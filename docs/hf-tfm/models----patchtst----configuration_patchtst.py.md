# `.\models\patchtst\configuration_patchtst.py`

```py
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
"""PatchTST model configuration"""

from typing import List, Optional, Union

# 导入所需的类和函数
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 预训练模型的配置映射表，指定了模型名称及其对应的配置文件链接
PATCHTST_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "ibm/patchtst-base": "https://huggingface.co/ibm/patchtst-base/resolve/main/config.json",
    # 查看所有 PatchTST 模型的链接地址：https://huggingface.co/ibm/models?filter=patchtst
}


class PatchTSTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`PatchTSTModel`]. It is used to instantiate an
    PatchTST model according to the specified arguments, defining the model architecture.
    [ibm/patchtst](https://huggingface.co/ibm/patchtst) architecture.

    Configuration objects inherit from [`PretrainedConfig`] can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    ```
    >>> from transformers import PatchTSTConfig, PatchTSTModel

    >>> # Initializing an PatchTST configuration with 12 time steps for prediction
    >>> configuration = PatchTSTConfig(prediction_length=12)

    >>> # Randomly initializing a model (with random weights) from the configuration
    >>> model = PatchTSTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    # 指定模型类型
    model_type = "patchtst"
    # 属性映射字典，将模型配置属性名映射到实际使用的名称
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "num_attention_heads",
        "num_hidden_layers": "num_hidden_layers",
    }
    # 初始化函数，用于初始化时间序列特定配置和Transformer模型参数
    def __init__(
        self,
        # 输入时间序列的通道数，默认为1
        num_input_channels: int = 1,
        # 上下文长度，默认为32，表示模型每次处理的时间步数
        context_length: int = 32,
        # 分布输出类型，默认为"student_t"，指定模型输出的概率分布类型
        distribution_output: str = "student_t",
        # 损失函数类型，默认为"mse"，表示模型训练过程中使用的损失函数类型
        loss: str = "mse",
        # PatchTST模型参数
        patch_length: int = 1,
        patch_stride: int = 1,
        # Transformer模型架构配置
        num_hidden_layers: int = 3,
        d_model: int = 128,
        num_attention_heads: int = 4,
        share_embedding: bool = True,
        channel_attention: bool = False,
        ffn_dim: int = 512,
        norm_type: str = "batchnorm",
        norm_eps: float = 1e-05,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
        positional_dropout: float = 0.0,
        path_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        bias: bool = True,
        activation_function: str = "gelu",
        pre_norm: bool = True,
        positional_encoding_type: str = "sincos",
        use_cls_token: bool = False,
        init_std: float = 0.02,
        share_projection: bool = True,
        scaling: Optional[Union[str, bool]] = "std",
        # 掩码预训练相关参数
        do_mask_input: Optional[bool] = None,
        mask_type: str = "random",
        random_mask_ratio: float = 0.5,
        num_forecast_mask_patches: Optional[Union[List[int], int]] = [2],
        channel_consistent_masking: Optional[bool] = False,
        unmasked_channel_indices: Optional[List[int]] = None,
        mask_value: int = 0,
        # 头部相关参数
        pooling_type: str = "mean",
        head_dropout: float = 0.0,
        prediction_length: int = 24,
        num_targets: int = 1,
        output_range: Optional[List] = None,
        # 分布头部相关参数
        num_parallel_samples: int = 100,
        **kwargs,
    ):
        # time series specific configuration
        # 设置上下文长度
        self.context_length = context_length
        # 设置输入通道数量
        self.num_input_channels = num_input_channels  # n_vars
        # 损失函数
        self.loss = loss
        # 输出分布类型
        self.distribution_output = distribution_output
        # 并行采样数量
        self.num_parallel_samples = num_parallel_samples

        # Transformer 架构配置
        # 模型维度
        self.d_model = d_model
        # 注意力头数
        self.num_attention_heads = num_attention_heads
        # 前馈神经网络维度
        self.ffn_dim = ffn_dim
        # 隐藏层数量
        self.num_hidden_layers = num_hidden_layers
        # 全连接层的 dropout
        self.dropout = dropout
        # 注意力机制的 dropout
        self.attention_dropout = attention_dropout
        # 是否共享嵌入层
        self.share_embedding = share_embedding
        # 通道注意力
        self.channel_attention = channel_attention
        # 规范化类型
        self.norm_type = norm_type
        # 规范化的 epsilon 值
        self.norm_eps = norm_eps
        # 位置编码的 dropout
        self.positional_dropout = positional_dropout
        # 路径的 dropout
        self.path_dropout = path_dropout
        # 前馈网络的 dropout
        self.ff_dropout = ff_dropout
        # 是否添加偏置
        self.bias = bias
        # 激活函数类型
        self.activation_function = activation_function
        # 是否在规范化前应用激活函数
        self.pre_norm = pre_norm
        # 位置编码类型
        self.positional_encoding_type = positional_encoding_type
        # 是否使用 CLS token
        self.use_cls_token = use_cls_token
        # 初始化标准差
        self.init_std = init_std
        # 缩放倍率
        self.scaling = scaling

        # PatchTST 参数
        # 补丁长度
        self.patch_length = patch_length
        # 补丁步长
        self.patch_stride = patch_stride

        # Mask 预训练
        # 是否进行输入遮罩
        self.do_mask_input = do_mask_input
        # 遮罩类型
        self.mask_type = mask_type
        # 随机遮罩比例
        self.random_mask_ratio = random_mask_ratio  # for random masking
        # 预测遮罩的数量
        self.num_forecast_mask_patches = num_forecast_mask_patches  # for forecast masking
        # 通道一致性遮罩
        self.channel_consistent_masking = channel_consistent_masking
        # 未遮罩通道的索引
        self.unmasked_channel_indices = unmasked_channel_indices
        # 遮罩值
        self.mask_value = mask_value

        # 通用头参数
        # 汇聚类型
        self.pooling_type = pooling_type
        # 头部的 dropout
        self.head_dropout = head_dropout

        # 用于预测头
        # 是否共享投影
        self.share_projection = share_projection
        # 预测长度
        self.prediction_length = prediction_length

        # 用于预测和回归头
        # 并行采样数量
        self.num_parallel_samples = num_parallel_samples

        # 回归
        # 目标数量
        self.num_targets = num_targets
        # 输出范围
        self.output_range = output_range

        super().__init__(**kwargs)
```