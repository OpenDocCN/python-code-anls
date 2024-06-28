# `.\models\layoutlmv3\configuration_layoutlmv3.py`

```
# coding=utf-8
# Copyright 2022 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
""" LayoutLMv3 model configuration"""

# 引入 OrderedDict 类和一些类型定义
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Mapping, Optional

# 引入 version 模块
from packaging import version

# 引入配置工具函数
from ...configuration_utils import PretrainedConfig

# 引入 OnnxConfig 类和相关工具函数
from ...onnx import OnnxConfig
from ...onnx.utils import compute_effective_axis_dimension

# 引入日志记录工具
from ...utils import logging

# 如果是类型检查阶段，引入额外的类型和工具
if TYPE_CHECKING:
    from ...processing_utils import ProcessorMixin
    from ...utils import TensorType

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# LayoutLMv3 预训练模型配置文件映射
LAYOUTLMV3_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/layoutlmv3-base": "https://huggingface.co/microsoft/layoutlmv3-base/resolve/main/config.json",
}


# LayoutLMv3 配置类，继承自 PretrainedConfig
class LayoutLMv3Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LayoutLMv3Model`]. It is used to instantiate an
    LayoutLMv3 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the LayoutLMv3
    [microsoft/layoutlmv3-base](https://huggingface.co/microsoft/layoutlmv3-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```python
    >>> from transformers import LayoutLMv3Config, LayoutLMv3Model

    >>> # Initializing a LayoutLMv3 microsoft/layoutlmv3-base style configuration
    >>> configuration = LayoutLMv3Config()

    >>> # Initializing a model (with random weights) from the microsoft/layoutlmv3-base style configuration
    >>> model = LayoutLMv3Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    # 模型类型标识为 "layoutlmv3"
    model_type = "layoutlmv3"
    # 初始化函数，用于创建一个新的对象
    def __init__(
        self,
        vocab_size=50265,  # 词汇表大小，默认为50265
        hidden_size=768,  # 隐藏层大小，默认为768
        num_hidden_layers=12,  # 隐藏层层数，默认为12
        num_attention_heads=12,  # 注意力头数，默认为12
        intermediate_size=3072,  # 中间层大小，默认为3072
        hidden_act="gelu",  # 隐藏层激活函数，默认为GELU
        hidden_dropout_prob=0.1,  # 隐藏层的dropout概率，默认为0.1
        attention_probs_dropout_prob=0.1,  # 注意力概率的dropout概率，默认为0.1
        max_position_embeddings=512,  # 最大位置嵌入数，默认为512
        type_vocab_size=2,  # 类型词汇表大小，默认为2
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-5,  # 层归一化的epsilon值，默认为1e-5
        pad_token_id=1,  # 填充token的ID，默认为1
        bos_token_id=0,  # 开始token的ID，默认为0
        eos_token_id=2,  # 结束token的ID，默认为2
        max_2d_position_embeddings=1024,  # 最大二维位置嵌入数，默认为1024
        coordinate_size=128,  # 坐标大小，默认为128
        shape_size=128,  # 形状大小，默认为128
        has_relative_attention_bias=True,  # 是否有相对注意力偏置，默认为True
        rel_pos_bins=32,  # 相对位置bin数，默认为32
        max_rel_pos=128,  # 最大相对位置，默认为128
        rel_2d_pos_bins=64,  # 二维相对位置bin数，默认为64
        max_rel_2d_pos=256,  # 最大二维相对位置，默认为256
        has_spatial_attention_bias=True,  # 是否有空间注意力偏置，默认为True
        text_embed=True,  # 是否包含文本嵌入，默认为True
        visual_embed=True,  # 是否包含视觉嵌入，默认为True
        input_size=224,  # 输入大小，默认为224
        num_channels=3,  # 通道数，默认为3
        patch_size=16,  # 补丁大小，默认为16
        classifier_dropout=None,  # 分类器的dropout，默认为None
        **kwargs,  # 其他关键字参数
    ):
        # 调用父类（Transformer）的初始化函数，传递参数
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        # 设置对象的其他属性
        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.coordinate_size = coordinate_size
        self.shape_size = shape_size
        self.has_relative_attention_bias = has_relative_attention_bias
        self.rel_pos_bins = rel_pos_bins
        self.max_rel_pos = max_rel_pos
        self.has_spatial_attention_bias = has_spatial_attention_bias
        self.rel_2d_pos_bins = rel_2d_pos_bins
        self.max_rel_2d_pos = max_rel_2d_pos
        self.text_embed = text_embed
        self.visual_embed = visual_embed
        self.input_size = input_size
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.classifier_dropout = classifier_dropout
class LayoutLMv3OnnxConfig(OnnxConfig):
    # 定义 torch ONNX 要求的最低版本为 1.12
    torch_onnx_minimum_version = version.parse("1.12")

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 输入的顺序在问答和序列分类任务中有所不同
        if self.task in ["question-answering", "sequence-classification"]:
            # 返回有序字典，包含不同输入名称及其维度信息
            return OrderedDict(
                [
                    ("input_ids", {0: "batch", 1: "sequence"}),
                    ("attention_mask", {0: "batch", 1: "sequence"}),
                    ("bbox", {0: "batch", 1: "sequence"}),
                    ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
                ]
            )
        else:
            # 返回有序字典，包含不同输入名称及其维度信息
            return OrderedDict(
                [
                    ("input_ids", {0: "batch", 1: "sequence"}),
                    ("bbox", {0: "batch", 1: "sequence"}),
                    ("attention_mask", {0: "batch", 1: "sequence"}),
                    ("pixel_values", {0: "batch", 1: "num_channels"}),
                ]
            )

    @property
    def atol_for_validation(self) -> float:
        # 设置用于验证的绝对误差容限
        return 1e-5

    @property
    def default_onnx_opset(self) -> int:
        # 默认的 ONNX 运算集版本
        return 12

    def generate_dummy_inputs(
        self,
        processor: "ProcessorMixin",
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional["TensorType"] = None,
        num_channels: int = 3,
        image_width: int = 40,
        image_height: int = 40,
        """
        Generate inputs to provide to the ONNX exporter for the specific framework

        Args:
            processor ([`ProcessorMixin`]):
                The processor associated with this model configuration.
            batch_size (`int`, *optional*, defaults to -1):
                The batch size to export the model for (-1 means dynamic axis).
            seq_length (`int`, *optional*, defaults to -1):
                The sequence length to export the model for (-1 means dynamic axis).
            is_pair (`bool`, *optional*, defaults to `False`):
                Indicate if the input is a pair (sentence 1, sentence 2).
            framework (`TensorType`, *optional*, defaults to `None`):
                The framework (PyTorch or TensorFlow) that the processor will generate tensors for.
            num_channels (`int`, *optional*, defaults to 3):
                The number of channels of the generated images.
            image_width (`int`, *optional*, defaults to 40):
                The width of the generated images.
            image_height (`int`, *optional*, defaults to 40):
                The height of the generated images.

        Returns:
            Mapping[str, Any]: holding the kwargs to provide to the model's forward function
        """

        # A dummy image is used so OCR should not be applied
        setattr(processor.image_processor, "apply_ocr", False)

        # If dynamic axis (-1) we forward with a fixed dimension of 2 samples to avoid optimizations made by ONNX
        # 计算有效的批量维度，如果为动态轴（-1），则使用默认的固定批量维度，避免ONNX的优化
        batch_size = compute_effective_axis_dimension(
            batch_size, fixed_dimension=OnnxConfig.default_fixed_batch, num_token_to_add=0
        )
        # If dynamic axis (-1) we forward with a fixed dimension of 8 tokens to avoid optimizations made by ONNX
        # 根据是否成对生成的token数量，计算有效的序列长度维度，避免ONNX的优化
        token_to_add = processor.tokenizer.num_special_tokens_to_add(is_pair)
        seq_length = compute_effective_axis_dimension(
            seq_length, fixed_dimension=OnnxConfig.default_fixed_sequence, num_token_to_add=token_to_add
        )
        # Generate dummy inputs according to compute batch and sequence
        # 根据计算得到的批次和序列长度生成虚拟文本输入
        dummy_text = [[" ".join([processor.tokenizer.unk_token]) * seq_length]] * batch_size

        # Generate dummy bounding boxes
        # 生成虚拟的边界框输入
        dummy_bboxes = [[[48, 84, 73, 128]]] * batch_size

        # If dynamic axis (-1) we forward with a fixed dimension of 2 samples to avoid optimizations made by ONNX
        # 根据是否成对生成的token数量，计算有效的批量维度，避免ONNX的优化
        dummy_image = self._generate_dummy_images(batch_size, num_channels, image_height, image_width)

        # 将生成的虚拟输入传递给处理器，生成模型前向函数所需的kwargs字典
        inputs = dict(
            processor(
                dummy_image,
                text=dummy_text,
                boxes=dummy_bboxes,
                return_tensors=framework,
            )
        )

        return inputs
```