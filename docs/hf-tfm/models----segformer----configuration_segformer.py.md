# `.\models\segformer\configuration_segformer.py`

```
# coding=utf-8
# Copyright 2021 NVIDIA and The HuggingFace Inc. team. All rights reserved.
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
""" SegFormer model configuration"""

# 引入警告模块，用于处理警告信息
import warnings
# 引入有序字典模块，用于维护字典插入顺序
from collections import OrderedDict
# 引入 Mapping 类型，用于类型提示
from typing import Mapping

# 引入版本比较模块
from packaging import version

# 引入配置工具模块中的 PretrainedConfig 类
from ...configuration_utils import PretrainedConfig
# 引入 OnnxConfig 类，用于处理 ONNX 模型配置
from ...onnx import OnnxConfig
# 引入日志工具模块
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 预训练模型配置文件的映射字典，包含预训练模型名称及其配置文件的 URL
SEGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "nvidia/segformer-b0-finetuned-ade-512-512": (
        "https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512/resolve/main/config.json"
    ),
    # 查看所有 SegFormer 模型的列表，请访问 https://huggingface.co/models?filter=segformer
}


class SegformerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SegformerModel`]. It is used to instantiate an
    SegFormer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the SegFormer
    [nvidia/segformer-b0-finetuned-ade-512-512](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    # 定义 SegFormerConfig 类，用于配置 SegFormer 模型的参数
    class SegformerConfig:
        # 初始化方法，设定各种参数的默认值
        def __init__(
            self,
            num_channels: int = 3,
            num_encoder_blocks: int = 4,
            depths: List[int] = [2, 2, 2, 2],
            sr_ratios: List[int] = [8, 4, 2, 1],
            hidden_sizes: List[int] = [32, 64, 160, 256],
            patch_sizes: List[int] = [7, 3, 3, 3],
            strides: List[int] = [4, 2, 2, 2],
            num_attention_heads: List[int] = [1, 2, 5, 8],
            mlp_ratios: List[int] = [4, 4, 4, 4],
            hidden_act: str or function = "gelu",
            hidden_dropout_prob: float = 0.0,
            attention_probs_dropout_prob: float = 0.0,
            classifier_dropout_prob: float = 0.1,
            initializer_range: float = 0.02,
            drop_path_rate: float = 0.1,
            layer_norm_eps: float = 1e-06,
            decoder_hidden_size: int = 256,
            semantic_loss_ignore_index: int = 255
        ):
            pass
    >>> configuration = SegformerConfig()

    >>> # 从 nvidia/segformer-b0-finetuned-ade-512-512 风格的配置初始化模型
    >>> model = SegformerModel(configuration)

    >>> # 访问模型配置信息
    >>> configuration = model.config
# 定义 SegformerOnnxConfig 类，继承自 OnnxConfig 类
class SegformerOnnxConfig(OnnxConfig):
    # 定义最低的 Torch ONNX 版本要求为 1.11
    torch_onnx_minimum_version = version.parse("1.11")

    # 定义 inputs 属性，返回一个有序字典，表示模型的输入信息
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
            ]
        )

    # 定义 atol_for_validation 属性，返回浮点数，表示验证过程中的绝对误差容差
    @property
    def atol_for_validation(self) -> float:
        return 1e-4

    # 定义 default_onnx_opset 属性，返回整数，表示默认的 ONNX 操作集版本
    @property
    def default_onnx_opset(self) -> int:
        return 12
```