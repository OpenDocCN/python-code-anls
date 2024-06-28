# `.\models\mobilenet_v1\modeling_mobilenet_v1.py`

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
""" PyTorch MobileNetV1 model."""

from typing import Optional, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPoolingAndNoAttention, ImageClassifierOutputWithNoAttention
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_mobilenet_v1 import MobileNetV1Config

logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "MobileNetV1Config"

# Base docstring
_CHECKPOINT_FOR_DOC = "google/mobilenet_v1_1.0_224"
_EXPECTED_OUTPUT_SHAPE = [1, 1024, 7, 7]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "google/mobilenet_v1_1.0_224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

MOBILENET_V1_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/mobilenet_v1_1.0_224",
    "google/mobilenet_v1_0.75_192",
    # See all MobileNetV1 models at https://huggingface.co/models?filter=mobilenet_v1
]

def _build_tf_to_pytorch_map(model, config, tf_weights=None):
    """
    A map of modules from TF to PyTorch.
    """
    # 初始化一个空的 TF 到 PyTorch 的映射字典
    tf_to_pt_map = {}

    if isinstance(model, MobileNetV1ForImageClassification):
        # 如果模型是 MobileNetV1ForImageClassification 的实例，则获取其 mobilenet_v1 属性
        backbone = model.mobilenet_v1
    else:
        # 否则，直接使用整个模型作为 backbone
        backbone = model

    # TF 模型中的前缀
    prefix = "MobilenetV1/Conv2d_0/"
    # 将 TF 中的权重映射到 PyTorch 模型的对应位置
    tf_to_pt_map[prefix + "weights"] = backbone.conv_stem.convolution.weight
    tf_to_pt_map[prefix + "BatchNorm/beta"] = backbone.conv_stem.normalization.bias
    tf_to_pt_map[prefix + "BatchNorm/gamma"] = backbone.conv_stem.normalization.weight
    tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = backbone.conv_stem.normalization.running_mean
    tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = backbone.conv_stem.normalization.running_var
    # 循环遍历范围为 0 到 12
    for i in range(13):
        # 计算 TensorFlow 中的索引（从 1 开始）
        tf_index = i + 1
        # 计算 PyTorch 中的索引（每个 i 对应两个）
        pt_index = i * 2

        # 获取指定索引的 backbone 层
        pointer = backbone.layer[pt_index]
        # 创建 MobileNetV1/Conv2d_{tf_index}_depthwise/ 前缀
        prefix = f"MobilenetV1/Conv2d_{tf_index}_depthwise/"
        # 将 TensorFlow 参数映射到 PyTorch 参数：深度卷积层权重
        tf_to_pt_map[prefix + "depthwise_weights"] = pointer.convolution.weight
        # 将 TensorFlow 参数映射到 PyTorch 参数：批归一化层偏置
        tf_to_pt_map[prefix + "BatchNorm/beta"] = pointer.normalization.bias
        # 将 TensorFlow 参数映射到 PyTorch 参数：批归一化层权重
        tf_to_pt_map[prefix + "BatchNorm/gamma"] = pointer.normalization.weight
        # 将 TensorFlow 参数映射到 PyTorch 参数：批归一化层移动均值
        tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = pointer.normalization.running_mean
        # 将 TensorFlow 参数映射到 PyTorch 参数：批归一化层移动方差
        tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = pointer.normalization.running_var

        # 获取指定索引的 backbone 层
        pointer = backbone.layer[pt_index + 1]
        # 创建 MobileNetV1/Conv2d_{tf_index}_pointwise/ 前缀
        prefix = f"MobilenetV1/Conv2d_{tf_index}_pointwise/"
        # 将 TensorFlow 参数映射到 PyTorch 参数：逐点卷积层权重
        tf_to_pt_map[prefix + "weights"] = pointer.convolution.weight
        # 将 TensorFlow 参数映射到 PyTorch 参数：批归一化层偏置
        tf_to_pt_map[prefix + "BatchNorm/beta"] = pointer.normalization.bias
        # 将 TensorFlow 参数映射到 PyTorch 参数：批归一化层权重
        tf_to_pt_map[prefix + "BatchNorm/gamma"] = pointer.normalization.weight
        # 将 TensorFlow 参数映射到 PyTorch 参数：批归一化层移动均值
        tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = pointer.normalization.running_mean
        # 将 TensorFlow 参数映射到 PyTorch 参数：批归一化层移动方差
        tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = pointer.normalization.running_var

    # 如果模型是 MobileNetV1ForImageClassification 类型
    if isinstance(model, MobileNetV1ForImageClassification):
        # 创建 MobilenetV1/Logits/Conv2d_1c_1x1/ 前缀
        prefix = "MobilenetV1/Logits/Conv2d_1c_1x1/"
        # 将 TensorFlow 参数映射到 PyTorch 参数：分类器权重
        tf_to_pt_map[prefix + "weights"] = model.classifier.weight
        # 将 TensorFlow 参数映射到 PyTorch 参数：分类器偏置
        tf_to_pt_map[prefix + "biases"] = model.classifier.bias

    # 返回 TensorFlow 到 PyTorch 参数映射字典
    return tf_to_pt_map
# 将 TensorFlow 模型的权重加载到 PyTorch 模型中
def load_tf_weights_in_mobilenet_v1(model, config, tf_checkpoint_path):
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise

    # 从 TensorFlow 模型加载权重变量列表
    init_vars = tf.train.list_variables(tf_checkpoint_path)
    tf_weights = {}
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        # 加载 TensorFlow 模型的变量数据
        array = tf.train.load_variable(tf_checkpoint_path, name)
        tf_weights[name] = array

    # 构建 TensorFlow 到 PyTorch 权重映射
    tf_to_pt_map = _build_tf_to_pytorch_map(model, config, tf_weights)

    for name, pointer in tf_to_pt_map.items():
        logger.info(f"Importing {name}")
        if name not in tf_weights:
            logger.info(f"{name} not in tf pre-trained weights, skipping")
            continue

        array = tf_weights[name]

        # 根据权重名字中的特定标识进行转置操作
        if "depthwise_weights" in name:
            logger.info("Transposing depthwise")
            array = np.transpose(array, (2, 3, 0, 1))
        elif "weights" in name:
            logger.info("Transposing")
            if len(pointer.shape) == 2:  # 复制到线性层
                array = array.squeeze().transpose()
            else:
                array = np.transpose(array, (3, 2, 0, 1))

        # 检查指针和数组的形状是否匹配
        if pointer.shape != array.shape:
            raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")

        logger.info(f"Initialize PyTorch weight {name} {array.shape}")
        # 将 NumPy 数组转换为 PyTorch 张量并赋值给指针
        pointer.data = torch.from_numpy(array)

        # 从字典中移除已处理的权重名字及其特定变体
        tf_weights.pop(name, None)
        tf_weights.pop(name + "/RMSProp", None)
        tf_weights.pop(name + "/RMSProp_1", None)
        tf_weights.pop(name + "/ExponentialMovingAverage", None)

    # 打印未复制到 PyTorch 模型中的权重名字列表
    logger.info(f"Weights not copied to PyTorch model: {', '.join(tf_weights.keys())}")
    # 返回加载了 TensorFlow 权重的 PyTorch 模型
    return model


# 将 TensorFlow 风格的 "SAME" 填充应用到卷积层
def apply_tf_padding(features: torch.Tensor, conv_layer: nn.Conv2d) -> torch.Tensor:
    """
    Apply TensorFlow-style "SAME" padding to a convolution layer. See the notes at:
    https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2
    """
    in_height, in_width = features.shape[-2:]
    stride_height, stride_width = conv_layer.stride
    kernel_height, kernel_width = conv_layer.kernel_size

    # 计算垂直方向和水平方向的填充量
    if in_height % stride_height == 0:
        pad_along_height = max(kernel_height - stride_height, 0)
    else:
        pad_along_height = max(kernel_height - (in_height % stride_height), 0)

    if in_width % stride_width == 0:
        pad_along_width = max(kernel_width - stride_width, 0)
    else:
        pad_along_width = max(kernel_width - (in_width % stride_width), 0)

    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    # 计算垂直方向上的顶部填充量，使用整数除法向下取整
    pad_top = pad_along_height // 2
    # 计算垂直方向上的底部填充量，保证总的填充量为 pad_along_height
    pad_bottom = pad_along_height - pad_top
    
    # 定义填充的元组，顺序为 (左, 右, 上, 下)
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    # 使用 PyTorch 的 nn.functional.pad 函数对 features 进行填充，采用常数填充方式，填充值为 0.0
    return nn.functional.pad(features, padding, "constant", 0.0)
class MobileNetV1ConvLayer(nn.Module):
    # 定义 MobileNetV1 模型的卷积层模块
    def __init__(
        self,
        config: MobileNetV1Config,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: Optional[int] = 1,
        groups: Optional[int] = 1,
        bias: bool = False,
        use_normalization: Optional[bool] = True,
        use_activation: Optional[bool or str] = True,
    ) -> None:
        # 初始化函数，设置各种参数和层

        super().__init__()
        self.config = config

        # 检查输入和输出通道数是否能被分组数整除
        if in_channels % groups != 0:
            raise ValueError(f"Input channels ({in_channels}) are not divisible by {groups} groups.")
        if out_channels % groups != 0:
            raise ValueError(f"Output channels ({out_channels}) are not divisible by {groups} groups.")

        # 计算填充大小，根据配置是否进行 TensorFlow 风格的填充
        padding = 0 if config.tf_padding else int((kernel_size - 1) / 2)

        # 创建卷积层对象
        self.convolution = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
            padding_mode="zeros",
        )

        # 如果使用归一化层，则创建 Batch Normalization 层
        if use_normalization:
            self.normalization = nn.BatchNorm2d(
                num_features=out_channels,
                eps=config.layer_norm_eps,
                momentum=0.9997,
                affine=True,
                track_running_stats=True,
            )
        else:
            self.normalization = None

        # 根据配置选择是否使用激活函数，并设置激活函数对象
        if use_activation:
            if isinstance(use_activation, str):
                self.activation = ACT2FN[use_activation]
            elif isinstance(config.hidden_act, str):
                self.activation = ACT2FN[config.hidden_act]
            else:
                self.activation = config.hidden_act
        else:
            self.activation = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # 前向传播函数，定义模型的数据流向

        # 如果配置为 TensorFlow 风格的填充，则应用填充函数
        if self.config.tf_padding:
            features = apply_tf_padding(features, self.convolution)
        
        # 经过卷积层处理
        features = self.convolution(features)
        
        # 如果有归一化层，则应用归一化
        if self.normalization is not None:
            features = self.normalization(features)
        
        # 如果有激活函数，则应用激活函数
        if self.activation is not None:
            features = self.activation(features)
        
        # 返回处理后的特征
        return features


class MobileNetV1PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # MobileNetV1 预训练模型的抽象类，用于初始化权重和简单的预训练模型下载和加载接口。

    config_class = MobileNetV1Config
    load_tf_weights = load_tf_weights_in_mobilenet_v1
    base_model_prefix = "mobilenet_v1"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = False
    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d]) -> None:
        """Initialize the weights"""
        # 检查当前模块是否为线性层或二维卷积层
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 如果是线性层或二维卷积层，使用正态分布初始化权重，均值为0，标准差为self.config.initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置项，将偏置项初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果当前模块是二维批标准化层
        elif isinstance(module, nn.BatchNorm2d):
            # 将批标准化层的偏置项初始化为零
            module.bias.data.zero_()
            # 将批标准化层的权重初始化为1
            module.weight.data.fill_(1.0)
# MOBILENET_V1_START_DOCSTRING 的值是一个原始字符串，用于描述 MobileNetV1Model 的模型信息和参数说明。
MOBILENET_V1_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`MobileNetV1Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# MOBILENET_V1_INPUTS_DOCSTRING 的值是一个原始字符串，用于描述 MobileNetV1Model 的输入参数说明。
MOBILENET_V1_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`MobileNetV1ImageProcessor.__call__`] for details.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# 使用 @add_start_docstrings 装饰器为 MobileNetV1Model 添加文档字符串，描述了模型输出和模型参数的说明。
@add_start_docstrings(
    "The bare MobileNetV1 model outputting raw hidden-states without any specific head on top.",
    MOBILENET_V1_START_DOCSTRING,
)
class MobileNetV1Model(MobileNetV1PreTrainedModel):
    # MobileNetV1Model 类的定义，继承自 MobileNetV1PreTrainedModel 类。
    # 初始化函数，接受 MobileNetV1Config 实例和一个布尔类型参数 add_pooling_layer
    def __init__(self, config: MobileNetV1Config, add_pooling_layer: bool = True):
        # 调用父类的初始化方法
        super().__init__(config)
        # 将传入的配置信息保存到 self.config 属性中
        self.config = config

        # 设定初始深度为 32
        depth = 32
        # 根据深度乘数和最小深度计算出初始输出通道数
        out_channels = max(int(depth * config.depth_multiplier), config.min_depth)

        # 创建 MobileNetV1ConvLayer 实例作为卷积的初始层 conv_stem
        self.conv_stem = MobileNetV1ConvLayer(
            config,
            in_channels=config.num_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
        )

        # 预设每个卷积层的步幅
        strides = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1]

        # 创建一个空的 nn.ModuleList，用于存储所有的卷积层
        self.layer = nn.ModuleList()
        # 循环创建 13 层卷积层
        for i in range(13):
            # 每一层卷积层的输入通道数等于上一层的输出通道数
            in_channels = out_channels

            # 如果当前层的步幅为 2 或者是第一层（i == 0），则需要更新深度和输出通道数
            if strides[i] == 2 or i == 0:
                depth *= 2
                out_channels = max(int(depth * config.depth_multiplier), config.min_depth)

            # 添加一个深度卷积层
            self.layer.append(
                MobileNetV1ConvLayer(
                    config,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=strides[i],
                    groups=in_channels,
                )
            )

            # 添加一个 1x1 的卷积层
            self.layer.append(
                MobileNetV1ConvLayer(
                    config,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                )
            )

        # 如果 add_pooling_layer 为 True，则创建一个自适应平均池化层
        self.pooler = nn.AdaptiveAvgPool2d((1, 1)) if add_pooling_layer else None

        # 调用内部方法完成权重初始化和最终处理
        self.post_init()

    # 用于剪枝不需要的注意力头，但目前未实现具体功能
    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError

    # 前向传播函数，接受像素值、是否返回隐藏状态、是否返回字典等参数
    @add_start_docstrings_to_model_forward(MOBILENET_V1_INPUTS_DOCSTRING)
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
        ) -> Union[tuple, BaseModelOutputWithPoolingAndNoAttention]:
        # 设置是否输出所有隐藏状态，默认为模型配置中的设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置是否返回字典形式的输出，默认为模型配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果未指定像素值，抛出数值错误异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 使用卷积层处理输入的像素值
        hidden_states = self.conv_stem(pixel_values)

        # 如果需要输出所有隐藏状态，则初始化一个空元组
        all_hidden_states = () if output_hidden_states else None

        # 遍历每个层次的模块
        for i, layer_module in enumerate(self.layer):
            # 依次将输入的隐藏状态传递给每个层次的模块进行处理
            hidden_states = layer_module(hidden_states)

            # 如果需要输出所有隐藏状态，则将当前层的隐藏状态添加到列表中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        # 获取最终的隐藏状态作为最后一个隐藏层的输出
        last_hidden_state = hidden_states

        # 如果定义了池化器，则对最终的隐藏状态进行池化处理并展平
        if self.pooler is not None:
            pooled_output = torch.flatten(self.pooler(last_hidden_state), start_dim=1)
        else:
            pooled_output = None

        # 如果不需要返回字典形式的输出，则返回一个元组，包含非空的结果
        if not return_dict:
            return tuple(v for v in [last_hidden_state, pooled_output, all_hidden_states] if v is not None)

        # 如果需要返回字典形式的输出，则构建一个相应的输出对象
        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=all_hidden_states,
        )
# 使用装饰器添加文档字符串到类的起始部分，描述了该类是基于 MobileNetV1 模型的图像分类模型
@add_start_docstrings(
    """
    MobileNetV1 model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    MOBILENET_V1_START_DOCSTRING,  # 添加了来自 MOBILENET_V1_START_DOCSTRING 的文档字符串
)
class MobileNetV1ForImageClassification(MobileNetV1PreTrainedModel):
    def __init__(self, config: MobileNetV1Config) -> None:
        super().__init__(config)

        # 设置分类器的类别数目
        self.num_labels = config.num_labels
        # 创建 MobileNetV1 模型
        self.mobilenet_v1 = MobileNetV1Model(config)

        # 获取 MobileNetV1 最后一层卷积的输出通道数
        last_hidden_size = self.mobilenet_v1.layer[-1].convolution.out_channels

        # 分类器头部
        # 使用给定的 dropout 概率创建 Dropout 层
        self.dropout = nn.Dropout(config.classifier_dropout_prob, inplace=True)
        # 创建线性层作为分类器，输出维度为最后一层卷积的输出通道数到类别数目的映射
        self.classifier = nn.Linear(last_hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # 初始化权重并应用最终处理
        self.post_init()

    # 使用装饰器添加模型前向方法的文档字符串，描述了输入参数和期望的输出
    @add_start_docstrings_to_model_forward(MOBILENET_V1_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,  # 提供了模型使用的检查点信息
        output_type=ImageClassifierOutputWithNoAttention,  # 指定了输出类型
        config_class=_CONFIG_FOR_DOC,  # 提供了用于文档的配置类信息
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,  # 描述了期望的输出
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        # 方法参数：pixel_values 接收图像的像素值张量，可以为空
        # output_hidden_states 控制是否输出隐藏状态的标志，可以为空
        # labels 接收标签张量，可以为空
        # return_dict 控制是否返回字典类型的输出，可以为空
    # 返回类型注解，可以返回元组或者带有无注意力输出的图像分类器输出
    ) -> Union[tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            用于计算图像分类/回归损失的标签。索引应在 `[0, ..., config.num_labels - 1]` 范围内。
            如果 `config.num_labels == 1`，则计算回归损失（均方误差损失）。
            如果 `config.num_labels > 1`，则计算分类损失（交叉熵损失）。
        """
        # 如果 return_dict 不为 None，则使用该值；否则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 调用 MobileNetV1 模型进行前向传播，返回输出
        outputs = self.mobilenet_v1(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)
    
        # 如果 return_dict 为 True，则使用 outputs.pooler_output 作为汇聚输出；否则使用 outputs 的第二个元素
        pooled_output = outputs.pooler_output if return_dict else outputs[1]
    
        # 对汇聚输出应用 dropout 和分类器，得到 logits
        logits = self.classifier(self.dropout(pooled_output))
    
        # 初始化损失为 None
        loss = None
        # 如果 labels 不为 None，则计算损失
        if labels is not None:
            # 如果问题类型未定义，则根据情况设置问题类型
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
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
    
        # 如果 return_dict 为 False，则返回 logits 和 outputs 的其余部分作为输出元组
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
    
        # 如果 return_dict 为 True，则返回 ImageClassifierOutputWithNoAttention 类的实例
        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )
```