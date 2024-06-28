# `.\models\mobilenet_v2\modeling_mobilenet_v2.py`

```py
# coding=utf-8
# Copyright 2022 Apple Inc. and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch MobileNetV2 model."""


from typing import Optional, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
    SemanticSegmenterOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_mobilenet_v2 import MobileNetV2Config


logger = logging.get_logger(__name__)


# General docstring
_CONFIG_FOR_DOC = "MobileNetV2Config"

# Base docstring
_CHECKPOINT_FOR_DOC = "google/mobilenet_v2_1.0_224"
_EXPECTED_OUTPUT_SHAPE = [1, 1280, 7, 7]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "google/mobilenet_v2_1.0_224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"


MOBILENET_V2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/mobilenet_v2_1.4_224",
    "google/mobilenet_v2_1.0_224",
    "google/mobilenet_v2_0.37_160",
    "google/mobilenet_v2_0.35_96",
    # See all MobileNetV2 models at https://huggingface.co/models?filter=mobilenet_v2
]


def _build_tf_to_pytorch_map(model, config, tf_weights=None):
    """
    A map of modules from TF to PyTorch.
    """

    # Initialize an empty map to store TF to PyTorch module mappings
    tf_to_pt_map = {}

    # Check if the model is an instance of MobileNetV2ForImageClassification or MobileNetV2ForSemanticSegmentation
    if isinstance(model, (MobileNetV2ForImageClassification, MobileNetV2ForSemanticSegmentation)):
        backbone = model.mobilenet_v2  # Get the MobileNetV2 backbone from the model
    else:
        backbone = model  # Otherwise, use the model directly

    # Function to handle Exponential Moving Average (EMA) weights in TF
    def ema(x):
        return x + "/ExponentialMovingAverage" if x + "/ExponentialMovingAverage" in tf_weights else x

    # Map TF weights to PyTorch model components for the convolutional stem
    prefix = "MobilenetV2/Conv/"
    tf_to_pt_map[ema(prefix + "weights")] = backbone.conv_stem.first_conv.convolution.weight
    tf_to_pt_map[ema(prefix + "BatchNorm/beta")] = backbone.conv_stem.first_conv.normalization.bias
    tf_to_pt_map[ema(prefix + "BatchNorm/gamma")] = backbone.conv_stem.first_conv.normalization.weight
    tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = backbone.conv_stem.first_conv.normalization.running_mean
    tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = backbone.conv_stem.first_conv.normalization.running_var

    prefix = "MobilenetV2/expanded_conv/depthwise/"
    tf_to_pt_map[ema(prefix + "depthwise_weights")] = backbone.conv_stem.conv_3x3.convolution.weight
    tf_to_pt_map[ema(prefix + "BatchNorm/beta")] = backbone.conv_stem.conv_3x3.normalization.bias
    tf_to_pt_map[ema(prefix + "BatchNorm/gamma")] = backbone.conv_stem.conv_3x3.normalization.weight
    tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = backbone.conv_stem.conv_3x3.normalization.running_mean
    tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = backbone.conv_stem.conv_3x3.normalization.running_var


# 将 TensorFlow 模型参数映射到 PyTorch 模型参数，处理卷积层的权重和规范化参数
tf_to_pt_map[ema(prefix + "depthwise_weights")] = backbone.conv_stem.conv_3x3.convolution.weight
tf_to_pt_map[ema(prefix + "BatchNorm/beta")] = backbone.conv_stem.conv_3x3.normalization.bias
tf_to_pt_map[ema(prefix + "BatchNorm/gamma")] = backbone.conv_stem.conv_3x3.normalization.weight
tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = backbone.conv_stem.conv_3x3.normalization.running_mean
tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = backbone.conv_stem.conv_3x3.normalization.running_var



    prefix = "MobilenetV2/expanded_conv/project/"
    tf_to_pt_map[ema(prefix + "weights")] = backbone.conv_stem.reduce_1x1.convolution.weight
    tf_to_pt_map[ema(prefix + "BatchNorm/beta")] = backbone.conv_stem.reduce_1x1.normalization.bias
    tf_to_pt_map[ema(prefix + "BatchNorm/gamma")] = backbone.conv_stem.reduce_1x1.normalization.weight
    tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = backbone.conv_stem.reduce_1x1.normalization.running_mean
    tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = backbone.conv_stem.reduce_1x1.normalization.running_var


# 将 TensorFlow 模型参数映射到 PyTorch 模型参数，处理扩展卷积块的投影部分的权重和规范化参数
prefix = "MobilenetV2/expanded_conv/project/"
tf_to_pt_map[ema(prefix + "weights")] = backbone.conv_stem.reduce_1x1.convolution.weight
tf_to_pt_map[ema(prefix + "BatchNorm/beta")] = backbone.conv_stem.reduce_1x1.normalization.bias
tf_to_pt_map[ema(prefix + "BatchNorm/gamma")] = backbone.conv_stem.reduce_1x1.normalization.weight
tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = backbone.conv_stem.reduce_1x1.normalization.running_mean
tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = backbone.conv_stem.reduce_1x1.normalization.running_var



    for i in range(16):
        tf_index = i + 1
        pt_index = i
        pointer = backbone.layer[pt_index]

        prefix = f"MobilenetV2/expanded_conv_{tf_index}/expand/"
        tf_to_pt_map[ema(prefix + "weights")] = pointer.expand_1x1.convolution.weight
        tf_to_pt_map[ema(prefix + "BatchNorm/beta")] = pointer.expand_1x1.normalization.bias
        tf_to_pt_map[ema(prefix + "BatchNorm/gamma")] = pointer.expand_1x1.normalization.weight
        tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = pointer.expand_1x1.normalization.running_mean
        tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = pointer.expand_1x1.normalization.running_var

        prefix = f"MobilenetV2/expanded_conv_{tf_index}/depthwise/"
        tf_to_pt_map[ema(prefix + "depthwise_weights")] = pointer.conv_3x3.convolution.weight
        tf_to_pt_map[ema(prefix + "BatchNorm/beta")] = pointer.conv_3x3.normalization.bias
        tf_to_pt_map[ema(prefix + "BatchNorm/gamma")] = pointer.conv_3x3.normalization.weight
        tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = pointer.conv_3x3.normalization.running_mean
        tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = pointer.conv_3x3.normalization.running_var

        prefix = f"MobilenetV2/expanded_conv_{tf_index}/project/"
        tf_to_pt_map[ema(prefix + "weights")] = pointer.reduce_1x1.convolution.weight
        tf_to_pt_map[ema(prefix + "BatchNorm/beta")] = pointer.reduce_1x1.normalization.bias
        tf_to_pt_map[ema(prefix + "BatchNorm/gamma")] = pointer.reduce_1x1.normalization.weight
        tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = pointer.reduce_1x1.normalization.running_mean
        tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = pointer.reduce_1x1.normalization.running_var


# 遍历每个 MobileNetV2 扩展卷积块的索引，映射 TensorFlow 模型参数到 PyTorch 模型参数
for i in range(16):
    tf_index = i + 1
    pt_index = i
    pointer = backbone.layer[pt_index]

    # 处理当前扩展卷积块的扩展部分权重和规范化参数
    prefix = f"MobilenetV2/expanded_conv_{tf_index}/expand/"
    tf_to_pt_map[ema(prefix + "weights")] = pointer.expand_1x1.convolution.weight
    tf_to_pt_map[ema(prefix + "BatchNorm/beta")] = pointer.expand_1x1.normalization.bias
    tf_to_pt_map[ema(prefix + "BatchNorm/gamma")] = pointer.expand_1x1.normalization.weight
    tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = pointer.expand_1x1.normalization.running_mean
    tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = pointer.expand_1x1.normalization.running_var

    # 处理当前扩展卷积块的深度可分离卷积部分的权重和规范化参数
    prefix = f"MobilenetV2/expanded_conv_{tf_index}/depthwise/"
    tf_to_pt_map[ema(prefix + "depthwise_weights")] = pointer.conv_3x3.convolution.weight
    tf_to_pt_map[ema(prefix + "BatchNorm/beta")] = pointer.conv_3x3.normalization.bias
    tf_to_pt_map[ema(prefix + "BatchNorm/gamma")] = pointer.conv_3x3.normalization.weight
    tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = pointer.conv_3x3.normalization.running_mean
    tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = pointer.conv_3x3.normalization.running_var

    # 处理当前扩展卷积块的投影部分的权重和规范化参数
    prefix = f"MobilenetV2/expanded_conv_{tf_index}/project/"
    tf_to_pt_map[ema(prefix + "weights")] = pointer.reduce_1x1.convolution.weight
    tf_to_pt_map[ema(prefix + "BatchNorm/beta")] = pointer.reduce_1x1.normalization.bias
    tf_to_pt_map[ema(prefix + "BatchNorm/gamma")] = pointer.reduce_1x1.normalization.weight
    tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = pointer.reduce_1x1.normalization.running_mean
    tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = pointer.reduce_1x1.normalization.running_var



    prefix = "MobilenetV2/Conv_1/"
    tf_to_pt_map[ema(prefix + "weights")] = backbone.conv_1x1.convolution.weight


# 将 TensorFlow 模型参数映射到 PyTorch 模型参数，处理 MobileNetV2 的第一个卷积层的权重
prefix = "MobilenetV2/Conv_1/"
tf_to_pt_map[ema(prefix + "weights")] = backbone.conv_1x1.convolution.weight
    # 将 TensorFlow 中的指定层参数映射到 PyTorch 模型中对应的权重和偏置
    tf_to_pt_map[ema(prefix + "BatchNorm/beta")] = backbone.conv_1x1.normalization.bias
    tf_to_pt_map[ema(prefix + "BatchNorm/gamma")] = backbone.conv_1x1.normalization.weight
    tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = backbone.conv_1x1.normalization.running_mean
    tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = backbone.conv_1x1.normalization.running_var

    # 如果模型是 MobileNetV2 图像分类器，则映射额外的层参数
    if isinstance(model, MobileNetV2ForImageClassification):
        prefix = "MobilenetV2/Logits/Conv2d_1c_1x1/"
        tf_to_pt_map[ema(prefix + "weights")] = model.classifier.weight
        tf_to_pt_map[ema(prefix + "biases")] = model.classifier.bias

    # 如果模型是 MobileNetV2 语义分割模型，则映射额外的层参数
    if isinstance(model, MobileNetV2ForSemanticSegmentation):
        prefix = "image_pooling/"
        tf_to_pt_map[prefix + "weights"] = model.segmentation_head.conv_pool.convolution.weight
        tf_to_pt_map[prefix + "BatchNorm/beta"] = model.segmentation_head.conv_pool.normalization.bias
        tf_to_pt_map[prefix + "BatchNorm/gamma"] = model.segmentation_head.conv_pool.normalization.weight
        tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = model.segmentation_head.conv_pool.normalization.running_mean
        tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = model.segmentation_head.conv_pool.normalization.running_var

        prefix = "aspp0/"
        tf_to_pt_map[prefix + "weights"] = model.segmentation_head.conv_aspp.convolution.weight
        tf_to_pt_map[prefix + "BatchNorm/beta"] = model.segmentation_head.conv_aspp.normalization.bias
        tf_to_pt_map[prefix + "BatchNorm/gamma"] = model.segmentation_head.conv_aspp.normalization.weight
        tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = model.segmentation_head.conv_aspp.normalization.running_mean
        tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = model.segmentation_head.conv_aspp.normalization.running_var

        prefix = "concat_projection/"
        tf_to_pt_map[prefix + "weights"] = model.segmentation_head.conv_projection.convolution.weight
        tf_to_pt_map[prefix + "BatchNorm/beta"] = model.segmentation_head.conv_projection.normalization.bias
        tf_to_pt_map[prefix + "BatchNorm/gamma"] = model.segmentation_head.conv_projection.normalization.weight
        tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = model.segmentation_head.conv_projection.normalization.running_mean
        tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = model.segmentation_head.conv_projection.normalization.running_var

        prefix = "logits/semantic/"
        tf_to_pt_map[ema(prefix + "weights")] = model.segmentation_head.classifier.convolution.weight
        tf_to_pt_map[ema(prefix + "biases")] = model.segmentation_head.classifier.convolution.bias

    # 返回 TensorFlow 到 PyTorch 参数映射的字典
    return tf_to_pt_map
# 将 TensorFlow 模型的权重加载到 PyTorch 模型中
def load_tf_weights_in_mobilenet_v2(model, config, tf_checkpoint_path):
    try:
        import numpy as np  # 导入 NumPy 库
        import tensorflow as tf  # 导入 TensorFlow 库
    except ImportError:
        logger.error(
            "Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise

    # 从 TensorFlow 模型中加载权重
    init_vars = tf.train.list_variables(tf_checkpoint_path)  # 获取 TensorFlow 检查点文件中的变量列表
    tf_weights = {}  # 创建一个空字典，用于存储 TensorFlow 权重

    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")  # 记录日志，显示正在加载的 TensorFlow 权重名称和形状
        array = tf.train.load_variable(tf_checkpoint_path, name)  # 加载 TensorFlow 权重变量
        tf_weights[name] = array  # 将加载的 TensorFlow 权重存入字典中

    # 构建 TensorFlow 到 PyTorch 的权重映射
    tf_to_pt_map = _build_tf_to_pytorch_map(model, config, tf_weights)

    for name, pointer in tf_to_pt_map.items():
        logger.info(f"Importing {name}")  # 记录日志，显示正在导入的权重名称

        if name not in tf_weights:
            logger.info(f"{name} not in tf pre-trained weights, skipping")  # 如果权重名称不在 TensorFlow 预训练权重中，则跳过
            continue

        array = tf_weights[name]  # 获取 TensorFlow 权重数组

        if "depthwise_weights" in name:
            logger.info("Transposing depthwise")  # 记录日志，显示正在转置深度可分离卷积权重
            array = np.transpose(array, (2, 3, 0, 1))  # 对深度可分离卷积的权重进行转置操作
        elif "weights" in name:
            logger.info("Transposing")  # 记录日志，显示正在转置权重
            if len(pointer.shape) == 2:  # 如果指针的形状是二维（即复制到线性层）
                array = array.squeeze().transpose()  # 对数组进行压缩并转置
            else:
                array = np.transpose(array, (3, 2, 0, 1))  # 对权重数组进行转置操作

        if pointer.shape != array.shape:
            raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")  # 抛出数值错误，如果指针形状与数组形状不匹配

        logger.info(f"Initialize PyTorch weight {name} {array.shape}")  # 记录日志，显示正在初始化 PyTorch 权重的名称和形状
        pointer.data = torch.from_numpy(array)  # 将 NumPy 数组转换为 PyTorch 张量，并赋值给指针的数据属性

        # 从 TensorFlow 权重字典中移除不需要的条目
        tf_weights.pop(name, None)
        tf_weights.pop(name + "/RMSProp", None)
        tf_weights.pop(name + "/RMSProp_1", None)
        tf_weights.pop(name + "/ExponentialMovingAverage", None)
        tf_weights.pop(name + "/Momentum", None)

    logger.info(f"Weights not copied to PyTorch model: {', '.join(tf_weights.keys())}")  # 记录日志，显示未复制到 PyTorch 模型的权重名称列表
    return model  # 返回加载了 TensorFlow 权重的 PyTorch 模型


def make_divisible(value: int, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """
    Ensure that all layers have a channel count that is divisible by `divisor`. This function is taken from the
    original TensorFlow repo. It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)  # 确保通道数可被 `divisor` 整除
    # 确保向下取整不会减少超过 10%
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)  # 返回确保可被 `divisor` 整除的通道数


def apply_depth_multiplier(config: MobileNetV2Config, channels: int) -> int:
    # 根据给定的参数计算可被指定整除的深度，以满足网络深度的要求
    return make_divisible(int(round(channels * config.depth_multiplier)), config.depth_divisible_by, config.min_depth)
    """
    Apply TensorFlow-style "SAME" padding to a convolution layer. See the notes at:
    https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2
    """
    # 获取输入特征的高度和宽度
    in_height = int(features.shape[-2])
    in_width = int(features.shape[-1])
    # 获取卷积层的步幅、卷积核大小和膨胀率
    stride_height, stride_width = conv_layer.stride
    kernel_height, kernel_width = conv_layer.kernel_size
    dilation_height, dilation_width = conv_layer.dilation

    # 计算沿高度和宽度方向的填充量
    if in_height % stride_height == 0:
        pad_along_height = max(kernel_height - stride_height, 0)
    else:
        pad_along_height = max(kernel_height - (in_height % stride_height), 0)

    if in_width % stride_width == 0:
        pad_along_width = max(kernel_width - stride_width, 0)
    else:
        pad_along_width = max(kernel_width - (in_width % stride_width), 0)

    # 计算左右和上下填充的具体值
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top

    # 构建填充元组，考虑膨胀率对填充量的影响
    padding = (
        pad_left * dilation_width,
        pad_right * dilation_width,
        pad_top * dilation_height,
        pad_bottom * dilation_height,
    )
    # 使用 PyTorch 提供的函数对特征进行填充
    return nn.functional.pad(features, padding, "constant", 0.0)
    ) -> None:
        super().__init__()
        self.config = config

        # 检查输入通道数是否能被分组数整除，如果不能则抛出错误
        if in_channels % groups != 0:
            raise ValueError(f"Input channels ({in_channels}) are not divisible by {groups} groups.")
        # 检查输出通道数是否能被分组数整除，如果不能则抛出错误
        if out_channels % groups != 0:
            raise ValueError(f"Output channels ({out_channels}) are not divisible by {groups} groups.")

        # 根据配置计算填充数，如果不使用 TensorFlow 填充则按照公式计算
        padding = 0 if config.tf_padding else int((kernel_size - 1) / 2) * dilation

        # 创建卷积层对象
        self.convolution = nn.Conv2d(
            in_channels=in_channels,        # 输入通道数
            out_channels=out_channels,      # 输出通道数
            kernel_size=kernel_size,        # 卷积核大小
            stride=stride,                  # 步长
            padding=padding,                # 填充数
            dilation=dilation,              # 空洞卷积率
            groups=groups,                  # 分组数
            bias=bias,                      # 是否使用偏置
            padding_mode="zeros",           # 填充模式
        )

        # 如果需要使用归一化
        if use_normalization:
            # 创建批归一化层对象
            self.normalization = nn.BatchNorm2d(
                num_features=out_channels,                                      # 输入特征数
                eps=config.layer_norm_eps if layer_norm_eps is None else layer_norm_eps,  # 归一化的 epsilon
                momentum=0.997,                                                 # 动量
                affine=True,                                                    # 是否使用仿射变换
                track_running_stats=True,                                       # 是否追踪运行时统计信息
            )
        else:
            self.normalization = None

        # 如果需要使用激活函数
        if use_activation:
            # 根据配置或者传入的激活函数名称选择激活函数
            if isinstance(use_activation, str):
                self.activation = ACT2FN[use_activation]
            elif isinstance(config.hidden_act, str):
                self.activation = ACT2FN[config.hidden_act]
            else:
                self.activation = config.hidden_act
        else:
            self.activation = None

    # 前向传播函数定义
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # 如果配置要求使用 TensorFlow 填充，则应用 TensorFlow 填充函数
        if self.config.tf_padding:
            features = apply_tf_padding(features, self.convolution)
        # 对特征进行卷积操作
        features = self.convolution(features)
        # 如果定义了归一化层，则对特征进行归一化处理
        if self.normalization is not None:
            features = self.normalization(features)
        # 如果定义了激活函数，则对特征进行激活函数处理
        if self.activation is not None:
            features = self.activation(features)
        # 返回处理后的特征
        return features
# 定义 MobileNetV2InvertedResidual 类，继承自 nn.Module
class MobileNetV2InvertedResidual(nn.Module):
    # 初始化函数，接受配置对象 config，输入通道数 in_channels，输出通道数 out_channels，步长 stride 和膨胀率 dilation
    def __init__(
        self, config: MobileNetV2Config, in_channels: int, out_channels: int, stride: int, dilation: int = 1
    ) -> None:
        super().__init__()

        # 根据配置计算扩展后的通道数，确保可被 config.depth_divisible_by 整除且不低于 config.min_depth
        expanded_channels = make_divisible(
            int(round(in_channels * config.expand_ratio)), config.depth_divisible_by, config.min_depth
        )

        # 如果步长不是 1 或 2，抛出 ValueError 异常
        if stride not in [1, 2]:
            raise ValueError(f"Invalid stride {stride}.")

        # 判断是否使用残差连接，条件是步长为 1 并且输入通道数等于输出通道数
        self.use_residual = (stride == 1) and (in_channels == out_channels)

        # 定义扩展 1x1 卷积层，将输入通道数扩展到 expanded_channels
        self.expand_1x1 = MobileNetV2ConvLayer(
            config, in_channels=in_channels, out_channels=expanded_channels, kernel_size=1
        )

        # 定义 3x3 卷积层，处理扩展后的通道数数据，支持指定步长、组卷积和空洞卷积
        self.conv_3x3 = MobileNetV2ConvLayer(
            config,
            in_channels=expanded_channels,
            out_channels=expanded_channels,
            kernel_size=3,
            stride=stride,
            groups=expanded_channels,
            dilation=dilation,
        )

        # 定义降维 1x1 卷积层，将通道数降至 out_channels，不使用激活函数
        self.reduce_1x1 = MobileNetV2ConvLayer(
            config,
            in_channels=expanded_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_activation=False,
        )

    # 前向传播函数，接受特征张量 features，返回处理后的特征张量
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # 将输入特征作为残差备份
        residual = features

        # 依次经过扩展 1x1 卷积、3x3 卷积和降维 1x1 卷积
        features = self.expand_1x1(features)
        features = self.conv_3x3(features)
        features = self.reduce_1x1(features)

        # 如果使用残差连接，将残差张量和处理后的特征张量相加；否则直接返回处理后的特征张量
        return residual + features if self.use_residual else features


# 定义 MobileNetV2Stem 类，继承自 nn.Module
class MobileNetV2Stem(nn.Module):
    # 初始化函数，接受配置对象 config，输入通道数 in_channels，扩展通道数 expanded_channels 和输出通道数 out_channels
    def __init__(self, config: MobileNetV2Config, in_channels: int, expanded_channels: int, out_channels: int) -> None:
        super().__init__()

        # 第一层是普通的 3x3 卷积层，步长为 2，将通道数扩展到 expanded_channels
        self.first_conv = MobileNetV2ConvLayer(
            config,
            in_channels=in_channels,
            out_channels=expanded_channels,
            kernel_size=3,
            stride=2,
        )

        # 如果配置要求首层是扩展层，则将扩展 1x1 卷积层设置为 None；否则定义扩展 1x1 卷积层
        if config.first_layer_is_expansion:
            self.expand_1x1 = None
        else:
            self.expand_1x1 = MobileNetV2ConvLayer(
                config, in_channels=expanded_channels, out_channels=expanded_channels, kernel_size=1
            )

        # 定义 3x3 卷积层，处理扩展后的通道数数据，步长为 1，组卷积使用 expanded_channels 组
        self.conv_3x3 = MobileNetV2ConvLayer(
            config,
            in_channels=expanded_channels,
            out_channels=expanded_channels,
            kernel_size=3,
            stride=1,
            groups=expanded_channels,
        )

        # 定义降维 1x1 卷积层，将通道数降至 out_channels，不使用激活函数
        self.reduce_1x1 = MobileNetV2ConvLayer(
            config,
            in_channels=expanded_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_activation=False,
        )
    # 定义一个前向传播方法，接受一个特征张量作为输入，返回处理后的特征张量
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # 使用self.first_conv对输入特征进行卷积操作，并更新features
        features = self.first_conv(features)
        # 如果存在self.expand_1x1模块，则对features进行1x1扩展卷积操作，并更新features
        if self.expand_1x1 is not None:
            features = self.expand_1x1(features)
        # 使用self.conv_3x3对features进行3x3卷积操作，并更新features
        features = self.conv_3x3(features)
        # 使用self.reduce_1x1对features进行1x1降维卷积操作，并更新features
        features = self.reduce_1x1(features)
        # 返回处理后的特征张量作为输出
        return features
# 定义一个继承自 `PreTrainedModel` 的抽象类，用于处理权重初始化和预训练模型的下载与加载接口。
class MobileNetV2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定配置类为 `MobileNetV2Config`
    config_class = MobileNetV2Config
    # 指定加载 TensorFlow 权重的函数为 `load_tf_weights_in_mobilenet_v2`
    load_tf_weights = load_tf_weights_in_mobilenet_v2
    # 指定基础模型的前缀为 "mobilenet_v2"
    base_model_prefix = "mobilenet_v2"
    # 主输入的名称为 "pixel_values"
    main_input_name = "pixel_values"
    # 不支持梯度检查点
    supports_gradient_checkpointing = False

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d]) -> None:
        """Initialize the weights"""
        # 如果模块是线性层或卷积层，使用正态分布初始化权重，均值为 0，标准差为配置文件中的 `initializer_range`
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置，将偏置初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是批归一化层，将偏置初始化为零，权重初始化为 1
        elif isinstance(module, nn.BatchNorm2d):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


# 定义一个字符串常量，描述 MobileNetV2 模型的起始文档字符串
MOBILENET_V2_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`MobileNetV2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义一个字符串常量，描述 MobileNetV2 模型的输入文档字符串
MOBILENET_V2_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`MobileNetV2ImageProcessor.__call__`] for details.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# 使用 `add_start_docstrings` 装饰器，为 `MobileNetV2Model` 类添加文档字符串，包含模型输出原始隐藏状态的描述和 `MOBILENET_V2_START_DOCSTRING` 的内容
@add_start_docstrings(
    "The bare MobileNetV2 model outputting raw hidden-states without any specific head on top.",
    MOBILENET_V2_START_DOCSTRING,
)
class MobileNetV2Model(MobileNetV2PreTrainedModel):
    pass  # 该类目前未添加额外的方法或属性，继承自 `MobileNetV2PreTrainedModel`
    def __init__(self, config: MobileNetV2Config, add_pooling_layer: bool = True):
        super().__init__(config)
        self.config = config

        # Output channels for the projection layers
        channels = [16, 24, 24, 32, 32, 32, 64, 64, 64, 64, 96, 96, 96, 160, 160, 160, 320]
        channels = [apply_depth_multiplier(config, x) for x in channels]

        # Strides for the depthwise layers
        strides = [2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1]

        self.conv_stem = MobileNetV2Stem(
            config,
            in_channels=config.num_channels,
            expanded_channels=apply_depth_multiplier(config, 32),
            out_channels=channels[0],
        )

        current_stride = 2  # first conv layer has stride 2
        dilation = 1

        self.layer = nn.ModuleList()
        for i in range(16):
            # Keep making the feature maps smaller or use dilated convolution?
            if current_stride == config.output_stride:
                layer_stride = 1
                layer_dilation = dilation
                dilation *= strides[i]  # larger dilation starts in next block
            else:
                layer_stride = strides[i]
                layer_dilation = 1
                current_stride *= layer_stride

            self.layer.append(
                MobileNetV2InvertedResidual(
                    config,
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    stride=layer_stride,
                    dilation=layer_dilation,
                )
            )

        if config.finegrained_output and config.depth_multiplier < 1.0:
            output_channels = 1280
        else:
            output_channels = apply_depth_multiplier(config, 1280)

        self.conv_1x1 = MobileNetV2ConvLayer(
            config,
            in_channels=channels[-1],
            out_channels=output_channels,
            kernel_size=1,
        )

        self.pooler = nn.AdaptiveAvgPool2d((1, 1)) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError

    @add_start_docstrings_to_model_forward(MOBILENET_V2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Forward pass of the MobileNetV2 model.

        Args:
            pixel_values (Optional[torch.Tensor]): Input tensor of shape (batch_size, channels, height, width).
            output_hidden_states (Optional[bool]): Whether to return hidden states.
            return_dict (Optional[bool]): Whether to return as a dictionary.

        Returns:
            BaseModelOutputWithPoolingAndNoAttention: A namedtuple with the model outputs.
        """
        # Implementation of the forward pass is provided by the library
        pass
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定 output_hidden_states，则使用模型配置中的默认值

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果未指定 return_dict，则使用模型配置中的默认值

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        # 如果未提供 pixel_values，抛出数值错误异常

        hidden_states = self.conv_stem(pixel_values)
        # 将输入的像素值通过卷积层 self.conv_stem 进行处理，得到隐藏状态

        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出所有隐藏状态，则初始化空元组 all_hidden_states，否则设为 None

        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states)
            # 逐层将隐藏状态通过 self.layer 中的每个层处理

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                # 如果需要输出所有隐藏状态，则将当前层的隐藏状态添加到 all_hidden_states 中

        last_hidden_state = self.conv_1x1(hidden_states)
        # 将最终的隐藏状态通过卷积层 self.conv_1x1 进行最后的处理，得到最终隐藏状态

        if self.pooler is not None:
            pooled_output = torch.flatten(self.pooler(last_hidden_state), start_dim=1)
            # 如果定义了池化器 self.pooler，则对最终隐藏状态进行池化处理，然后展平成一维张量
        else:
            pooled_output = None
            # 如果未定义池化器，则池化输出设为 None

        if not return_dict:
            return tuple(v for v in [last_hidden_state, pooled_output, all_hidden_states] if v is not None)
        # 如果不需要返回字典形式的结果，则返回包含非 None 值的元组

        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=all_hidden_states,
        )
        # 否则，返回包含各隐藏状态的 BaseModelOutputWithPoolingAndNoAttention 对象
``
    ) -> Union[tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss). If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 如果 return_dict 不为 None，则使用指定的 return_dict，否则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 MobileNetV2 模型处理像素值，返回输出结果，可以包含隐藏状态
        outputs = self.mobilenet_v2(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        # 如果 return_dict 为 True，则从 outputs 中获取 pooler_output；否则从 outputs 的第二个元素获取
        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        # 对池化后的输出进行 dropout 和分类器处理，得到 logits
        logits = self.classifier(self.dropout(pooled_output))

        # 初始化损失为 None
        loss = None
        # 如果 labels 不为 None，则计算损失
        if labels is not None:
            # 根据配置确定问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型计算相应的损失
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    # 如果只有一个标签，使用 squeeze 处理 logits 和 labels 后计算损失
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # 单标签分类问题，使用交叉熵损失函数
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # 多标签分类问题，使用带 logits 的二元交叉熵损失函数
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果 return_dict 为 False，则按照指定格式返回结果
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则返回 ImageClassifierOutputWithNoAttention 对象
        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )
class MobileNetV2DeepLabV3Plus(nn.Module):
    """
    The neural network from the paper "Encoder-Decoder with Atrous Separable Convolution for Semantic Image
    Segmentation" https://arxiv.org/abs/1802.02611
    """

    def __init__(self, config: MobileNetV2Config) -> None:
        super().__init__()

        # 定义平均池化层，输出大小为1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        # 定义池化后的卷积层，将通道数应用深度乘数后，输出通道数为256
        self.conv_pool = MobileNetV2ConvLayer(
            config,
            in_channels=apply_depth_multiplier(config, 320),
            out_channels=256,
            kernel_size=1,
            stride=1,
            use_normalization=True,
            use_activation="relu",
            layer_norm_eps=1e-5,
        )

        # 定义ASPP模块中的卷积层，输出通道数为256
        self.conv_aspp = MobileNetV2ConvLayer(
            config,
            in_channels=apply_depth_multiplier(config, 320),
            out_channels=256,
            kernel_size=1,
            stride=1,
            use_normalization=True,
            use_activation="relu",
            layer_norm_eps=1e-5,
        )

        # 定义投影卷积层，输入通道数为512，输出通道数为256
        self.conv_projection = MobileNetV2ConvLayer(
            config,
            in_channels=512,
            out_channels=256,
            kernel_size=1,
            stride=1,
            use_normalization=True,
            use_activation="relu",
            layer_norm_eps=1e-5,
        )

        # 定义二维Dropout层，按照指定的概率进行丢弃
        self.dropout = nn.Dropout2d(config.classifier_dropout_prob)

        # 定义分类器卷积层，输入通道数为256，输出通道数为类别数
        self.classifier = MobileNetV2ConvLayer(
            config,
            in_channels=256,
            out_channels=config.num_labels,
            kernel_size=1,
            use_normalization=False,
            use_activation=False,
            bias=True,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        spatial_size = features.shape[-2:]

        # 对输入特征进行全局平均池化
        features_pool = self.avg_pool(features)
        # 应用池化后的卷积层
        features_pool = self.conv_pool(features_pool)
        # 进行双线性插值，调整特征大小与原始大小相同
        features_pool = nn.functional.interpolate(
            features_pool, size=spatial_size, mode="bilinear", align_corners=True
        )

        # 应用ASPP模块中的卷积层
        features_aspp = self.conv_aspp(features)

        # 将池化后的特征与ASPP模块的特征拼接起来
        features = torch.cat([features_pool, features_aspp], dim=1)

        # 应用投影卷积层
        features = self.conv_projection(features)
        # 应用Dropout层
        features = self.dropout(features)
        # 应用分类器卷积层，得到最终的特征映射
        features = self.classifier(features)
        return features


@add_start_docstrings(
    """
    MobileNetV2 model with a semantic segmentation head on top, e.g. for Pascal VOC.
    """,
    MOBILENET_V2_START_DOCSTRING,
)
class MobileNetV2ForSemanticSegmentation(MobileNetV2PreTrainedModel):
    def __init__(self, config: MobileNetV2Config) -> None:
        super().__init__(config)

        # 设置类别数
        self.num_labels = config.num_labels
        # 创建MobileNetV2基础模型，不包括池化层
        self.mobilenet_v2 = MobileNetV2Model(config, add_pooling_layer=False)
        # 创建深度可分离卷积模型用于语义分割
        self.segmentation_head = MobileNetV2DeepLabV3Plus(config)

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(MOBILENET_V2_INPUTS_DOCSTRING)
    # 使用装饰器替换返回文档字符串，指定输出类型为SemanticSegmenterOutput，配置类为_CONFIG_FOR_DOC
    @replace_return_docstrings(output_type=SemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    # 前向传播方法，接受多个参数
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,  # 输入像素值张量，可选
        labels: Optional[torch.Tensor] = None,         # 标签张量，可选
        output_hidden_states: Optional[bool] = None,   # 是否输出隐藏状态张量，可选
        return_dict: Optional[bool] = None,            # 是否返回字典格式结果，可选
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 MobileNetV2 处理输入图像，获取输出的隐藏状态
        outputs = self.mobilenet_v2(
            pixel_values,
            output_hidden_states=True,  # 我们需要中间的隐藏状态作为输出
            return_dict=return_dict,
        )

        # 如果配置要求返回字典，则从输出中获取编码器的隐藏状态
        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]

        # 使用分割头部处理编码器的最后一个隐藏状态，得到预测的 logits
        logits = self.segmentation_head(encoder_hidden_states[-1])

        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                raise ValueError("标签数量应大于1")
            else:
                # 将 logits 上采样到原始图像大小
                upsampled_logits = nn.functional.interpolate(
                    logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
                # 计算交叉熵损失
                loss_fct = CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
                loss = loss_fct(upsampled_logits, labels)

        # 如果不要求返回字典，则根据设置决定输出内容
        if not return_dict:
            if output_hidden_states:
                # 如果需要隐藏状态，则包含 logits 和隐藏状态
                output = (logits,) + outputs[1:]
            else:
                # 否则只包含 logits 和额外的输出
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回带有损失、logits、隐藏状态和注意力的 SemanticSegmenterOutput 对象
        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=None,
        )
```