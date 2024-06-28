# `.\models\convnextv2\modeling_tf_convnextv2.py`

```
# coding=utf-8
# Copyright 2023 Meta Platforms Inc. and The HuggingFace Inc. team. All rights reserved.
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
""" TF 2.0 ConvNextV2 model."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithNoAttention,
    TFBaseModelOutputWithPooling,
    TFBaseModelOutputWithPoolingAndNoAttention,
    TFImageClassifierOutputWithNoAttention,
)
from ...modeling_tf_utils import (
    TFModelInputType,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    get_initializer,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import shape_list
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from .configuration_convnextv2 import ConvNextV2Config

logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "ConvNextV2Config"

# Base docstring
_CHECKPOINT_FOR_DOC = "facebook/convnextv2-tiny-1k-224"
_EXPECTED_OUTPUT_SHAPE = [1, 768, 7, 7]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "facebook/convnextv2-tiny-1k-224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

CONVNEXTV2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/convnextv2-tiny-1k-224",
    # See all ConvNextV2 models at https://huggingface.co/models?filter=convnextv2
]

# Copied from transformers.models.convnext.modeling_tf_convnext.TFConvNextDropPath with ConvNext->ConvNextV2
class TFConvNextV2DropPath(keras.layers.Layer):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    References:
        (1) github.com:rwightman/pytorch-image-models
    """

    def __init__(self, drop_path: float, **kwargs):
        super().__init__(**kwargs)
        self.drop_path = drop_path

    def call(self, x: tf.Tensor, training=None):
        if training:
            # 计算保留概率
            keep_prob = 1 - self.drop_path
            # 创建与输入张量相同形状的随机张量
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            # 应用随机深度操作
            return (x / keep_prob) * random_tensor
        # 若不训练状态，则直接返回输入张量
        return x


class TFConvNextV2GRN(keras.layers.Layer):
    """GRN (Global Response Normalization) layer"""
    # 初始化函数，接受一个配置对象和一个整数维度作为参数
    def __init__(self, config: ConvNextV2Config, dim: int, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 将输入的维度参数存储在对象的属性中
        self.dim = dim

    # 构建模型的方法，用于创建模型的权重
    def build(self, input_shape: tf.TensorShape = None):
        # 创建名为 "weight" 的模型权重，形状为 (1, 1, 1, self.dim)，使用零初始化器
        self.weight = self.add_weight(
            name="weight",
            shape=(1, 1, 1, self.dim),
            initializer=keras.initializers.Zeros(),
        )
        # 创建名为 "bias" 的模型偏置，形状同样为 (1, 1, 1, self.dim)，使用零初始化器
        self.bias = self.add_weight(
            name="bias",
            shape=(1, 1, 1, self.dim),
            initializer=keras.initializers.Zeros(),
        )
        # 调用父类的构建方法，传递输入形状参数
        return super().build(input_shape)

    # 模型的调用方法，用于执行前向传播
    def call(self, hidden_states: tf.Tensor):
        # 计算每个样本的全局特征向量的欧几里得范数
        global_features = tf.norm(hidden_states, ord="euclidean", axis=(1, 2), keepdims=True)
        # 对全局特征向量进行归一化，确保分母不为零
        norm_features = global_features / (tf.reduce_mean(global_features, axis=-1, keepdims=True) + 1e-6)
        # 计算加权后的隐藏状态并加上偏置项
        hidden_states = self.weight * (hidden_states * norm_features) + self.bias + hidden_states
        # 返回加权后的隐藏状态作为输出
        return hidden_states
# 从transformers.models.convnext.modeling_tf_convnext.TFConvNextEmbeddings复制并改为ConvNextV2
class TFConvNextV2Embeddings(keras.layers.Layer):
    """这个类与src/transformers/models/swin/modeling_swin.py中的SwinEmbeddings类类似（并受其启发）。"""

    def __init__(self, config: ConvNextV2Config, **kwargs):
        super().__init__(**kwargs)
        # 定义用于提取补丁嵌入的卷积层
        self.patch_embeddings = keras.layers.Conv2D(
            filters=config.hidden_sizes[0],    # 输出特征的数量
            kernel_size=config.patch_size,     # 补丁大小
            strides=config.patch_size,         # 步幅大小
            name="patch_embeddings",           # 层名称
            kernel_initializer=get_initializer(config.initializer_range),  # 卷积核初始化器
            bias_initializer=keras.initializers.Zeros(),  # 偏置项初始化器
        )
        # LayerNormalization 层，用于标准化输入数据
        self.layernorm = keras.layers.LayerNormalization(epsilon=1e-6, name="layernorm")
        self.num_channels = config.num_channels  # 通道数
        self.config = config  # 配置参数

    def call(self, pixel_values):
        if isinstance(pixel_values, dict):
            pixel_values = pixel_values["pixel_values"]

        # 检查像素值张量的通道维度是否与配置中设置的一致
        tf.debugging.assert_equal(
            shape_list(pixel_values)[1],
            self.num_channels,
            message="确保像素值的通道维度与配置中设置的一致。",
        )

        # 当在CPU上运行时，`keras.layers.Conv2D`不支持`NCHW`格式，需要将输入格式从`NCHW`转换为`NHWC`
        # shape = (batch_size, in_height, in_width, in_channels)
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))

        # 提取补丁嵌入特征
        embeddings = self.patch_embeddings(pixel_values)
        # 应用层标准化
        embeddings = self.layernorm(embeddings)
        return embeddings

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经构建，则直接返回
        if getattr(self, "patch_embeddings", None) is not None:
            with tf.name_scope(self.patch_embeddings.name):
                # 根据配置构建补丁嵌入层
                self.patch_embeddings.build([None, None, None, self.config.num_channels])
        if getattr(self, "layernorm", None) is not None:
            with tf.name_scope(self.layernorm.name):
                # 根据配置构建层标准化层
                self.layernorm.build([None, None, None, self.config.hidden_sizes[0]])


class TFConvNextV2Layer(keras.layers.Layer):
    """这对应于原始实现中的`Block`类。

    有两个等效的实现方式：
    [DwConv, LayerNorm (channels_first), Conv, GELU, 1x1 Conv]; 全部在(N, C, H, W)中
    [DwConv, 转换到(N, H, W, C), LayerNorm (channels_last), Linear, GELU, Linear]; 再转换回来

    作者在PyTorch中发现第二种方式略快。由于我们已经将输入调整为遵循NHWC顺序，因此可以直接应用操作而无需排列。
    """
    # 初始化函数，用于初始化类的实例
    def __init__(self, config: ConvNextV2Config, dim: int, drop_path: float = 0.0, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 设置输入通道数
        self.dim = dim
        # 设置模型配置
        self.config = config
        # 深度可分离卷积层，使用7x7的卷积核
        self.dwconv = keras.layers.Conv2D(
            filters=dim,
            kernel_size=7,
            padding="same",
            groups=dim,  # 分组数与输入通道数相同，实现深度可分离卷积
            kernel_initializer=get_initializer(config.initializer_range),  # 设置卷积核的初始化器
            bias_initializer=keras.initializers.Zeros(),  # 设置偏置的初始化器为零
            name="dwconv",  # 层的名称为dwconv
        )  # depthwise conv，深度可分离卷积
        # 层归一化层，使用默认的epsilon值为1e-6
        self.layernorm = keras.layers.LayerNormalization(
            epsilon=1e-6,
            name="layernorm",
        )
        # 点卷积层，输出单元数为4倍的输入通道数dim
        self.pwconv1 = keras.layers.Dense(
            units=4 * dim,
            kernel_initializer=get_initializer(config.initializer_range),  # 设置全连接层的权重初始化器
            bias_initializer=keras.initializers.Zeros(),  # 设置偏置的初始化器为零
            name="pwconv1",  # 层的名称为pwconv1
        )  # pointwise/1x1 convs，使用线性层实现的1x1卷积
        # 获取激活函数
        self.act = get_tf_activation(config.hidden_act)
        # ConvNextV2GRN模块，使用4倍dim的输出单元数
        self.grn = TFConvNextV2GRN(config, 4 * dim, dtype=tf.float32, name="grn")
        # 点卷积层2，输出单元数为dim
        self.pwconv2 = keras.layers.Dense(
            units=dim,
            kernel_initializer=get_initializer(config.initializer_range),  # 设置全连接层的权重初始化器
            bias_initializer=keras.initializers.Zeros(),  # 设置偏置的初始化器为零
            name="pwconv2",  # 层的名称为pwconv2
        )
        # 使用`layers.Activation`代替`tf.identity`，以更好地控制训练行为
        # 如果drop_path大于0.0，则使用TFConvNextV2DropPath进行随机深度跳连，否则使用线性激活层
        self.drop_path = (
            TFConvNextV2DropPath(drop_path, name="drop_path")
            if drop_path > 0.0
            else keras.layers.Activation("linear", name="drop_path")
        )

    # 定义call方法，实现类的函数调用功能
    def call(self, hidden_states, training=False):
        # 将输入数据赋值给input变量
        input = hidden_states
        # 深度可分离卷积层的前向传播
        x = self.dwconv(hidden_states)
        # 层归一化层的前向传播
        x = self.layernorm(x)
        # 点卷积层1的前向传播
        x = self.pwconv1(x)
        # 使用激活函数激活输出
        x = self.act(x)
        # ConvNextV2GRN模块的前向传播
        x = self.grn(x)
        # 点卷积层2的前向传播
        x = self.pwconv2(x)
        # 使用drop_path函数进行深度跳连，控制是否训练过程
        x = self.drop_path(x, training=training)
        # 返回原始输入和处理后的结果的和，实现残差连接
        x = input + x
        # 返回最终的输出结果
        return x
    # 定义 build 方法，用于构建模型结构
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回，避免重复构建
        if self.built:
            return
        # 标记模型已构建
        self.built = True
        
        # 如果存在 dwconv 属性，则构建 depthwise convolution 层
        if getattr(self, "dwconv", None) is not None:
            # 在命名空间内构建 dwconv 层
            with tf.name_scope(self.dwconv.name):
                self.dwconv.build([None, None, None, self.dim])
        
        # 如果存在 layernorm 属性，则构建 Layer Normalization 层
        if getattr(self, "layernorm", None) is not None:
            # 在命名空间内构建 layernorm 层
            with tf.name_scope(self.layernorm.name):
                self.layernorm.build([None, None, None, self.dim])
        
        # 如果存在 pwconv1 属性，则构建 pointwise convolution 层
        if getattr(self, "pwconv1", None) is not None:
            # 在命名空间内构建 pwconv1 层
            with tf.name_scope(self.pwconv1.name):
                self.pwconv1.build([None, None, self.dim])
        
        # 如果存在 grn 属性，则构建 global reduction network 层
        if getattr(self, "grn", None) is not None:
            # 在命名空间内构建 grn 层
            with tf.name_scope(self.grn.name):
                self.grn.build(None)
        
        # 如果存在 pwconv2 属性，则构建第二个 pointwise convolution 层
        if getattr(self, "pwconv2", None) is not None:
            # 在命名空间内构建 pwconv2 层
            with tf.name_scope(self.pwconv2.name):
                self.pwconv2.build([None, None, 4 * self.dim])
        
        # 如果存在 drop_path 属性，则构建 drop path 层
        if getattr(self, "drop_path", None) is not None:
            # 在命名空间内构建 drop path 层
            with tf.name_scope(self.drop_path.name):
                self.drop_path.build(None)
# Copied from transformers.models.convnext.modeling_tf_convnext.TFConvNextStage with ConvNext->ConvNextV2
class TFConvNextV2Stage(keras.layers.Layer):
    """ConvNextV2 stage, consisting of an optional downsampling layer + multiple residual blocks.

    Args:
        config (`ConvNextV2Config`):
            Model configuration class.
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`):
            Number of output channels.
        depth (`int`):
            Number of residual blocks.
        drop_path_rates(`List[float]`):
            Stochastic depth rates for each layer.
    """

    def __init__(
        self,
        config: ConvNextV2Config,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        stride: int = 2,
        depth: int = 2,
        drop_path_rates: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Check if downsampling is needed based on input and output channels, or if stride > 1
        if in_channels != out_channels or stride > 1:
            # Define a downsampling layer if conditions are met
            self.downsampling_layer = [
                keras.layers.LayerNormalization(
                    epsilon=1e-6,
                    name="downsampling_layer.0",
                ),
                # Additional comment on the Conv2D layer
                # This layer expects NHWC input format due to a previous format transformation.
                # Outputs are in NHWC format throughout until the format is changed back to NCHW.
                keras.layers.Conv2D(
                    filters=out_channels,
                    kernel_size=kernel_size,
                    strides=stride,
                    kernel_initializer=get_initializer(config.initializer_range),
                    bias_initializer=keras.initializers.Zeros(),
                    name="downsampling_layer.1",
                ),
            ]
        else:
            # If no downsampling is needed, use an identity function
            self.downsampling_layer = [tf.identity]

        # Initialize stochastic depth rates or set them to 0.0 if not provided
        drop_path_rates = drop_path_rates or [0.0] * depth
        # Create a list of TFConvNextV2Layer instances based on depth
        self.layers = [
            TFConvNextV2Layer(
                config,
                dim=out_channels,
                drop_path=drop_path_rates[j],
                name=f"layers.{j}",
            )
            for j in range(depth)
        ]
        # Store input and output channel counts and the stride value
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def call(self, hidden_states):
        # Apply the downsampling layers to the input hidden_states
        for layer in self.downsampling_layer:
            hidden_states = layer(hidden_states)
        # Apply each residual layer in self.layers sequentially to hidden_states
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        # Return the processed hidden_states after all layers
        return hidden_states
    # 如果已经建立过网络结构，则直接返回，不做重复建立
    if self.built:
        return
    # 设置标志位表示网络已经建立
    self.built = True

    # 检查是否存在子层，若存在则逐层建立网络结构
    if getattr(self, "layers", None) is not None:
        for layer in self.layers:
            # 使用 TensorFlow 的命名空间，将每个子层的建立过程包裹起来
            with tf.name_scope(layer.name):
                layer.build(None)

    # 如果输入通道数不等于输出通道数，或者步幅大于1，需要建立降采样层
    if self.in_channels != self.out_channels or self.stride > 1:
        # 使用 TensorFlow 的命名空间，建立第一个降采样层
        with tf.name_scope(self.downsampling_layer[0].name):
            self.downsampling_layer[0].build([None, None, None, self.in_channels])
        # 使用 TensorFlow 的命名空间，建立第二个降采样层
        with tf.name_scope(self.downsampling_layer[1].name):
            self.downsampling_layer[1].build([None, None, None, self.in_channels])
# 定义 TFConvNextV2Encoder 类，继承自 keras.layers.Layer
class TFConvNextV2Encoder(keras.layers.Layer):
    
    # 初始化方法，接受一个 ConvNextV2Config 类型的配置对象和其他关键字参数
    def __init__(self, config: ConvNextV2Config, **kwargs):
        super().__init__(**kwargs)
        
        # 初始化空列表 stages 用于存储 TFConvNextV2Stage 对象
        self.stages = []
        
        # 生成一个线性空间的张量，作为各阶段的丢弃路径率，根据配置对象中的深度计算
        drop_path_rates = tf.linspace(0.0, config.drop_path_rate, sum(config.depths))
        drop_path_rates = tf.split(drop_path_rates, config.depths)
        drop_path_rates = [x.numpy().tolist() for x in drop_path_rates]
        
        # 设置初始通道数为配置对象中隐藏层大小列表的第一个元素
        prev_chs = config.hidden_sizes[0]
        
        # 遍历每个阶段的数量，并创建 TFConvNextV2Stage 实例，加入到 stages 列表中
        for i in range(config.num_stages):
            out_chs = config.hidden_sizes[i]
            stage = TFConvNextV2Stage(
                config,
                in_channels=prev_chs,
                out_channels=out_chs,
                stride=2 if i > 0 else 1,
                depth=config.depths[i],
                drop_path_rates=drop_path_rates[i],
                name=f"stages.{i}",
            )
            self.stages.append(stage)
            prev_chs = out_chs

    # 定义 call 方法，处理输入的隐藏状态张量及一些可选的参数，返回模型输出
    def call(
        self,
        hidden_states: tf.Tensor,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, TFBaseModelOutputWithNoAttention]:
        # 初始化 all_hidden_states 为空元组或 None，根据输出隐藏状态的参数设置
        all_hidden_states = () if output_hidden_states else None
        
        # 遍历 self.stages 中的每个层模块，对隐藏状态进行处理
        for i, layer_module in enumerate(self.stages):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            hidden_states = layer_module(hidden_states)
        
        # 如果输出隐藏状态，将最终隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # 如果 return_dict 为 False，返回一个元组，包括隐藏状态和所有隐藏状态
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)
        
        # 如果 return_dict 为 True，返回 TFBaseModelOutputWithNoAttention 类型的对象
        return TFBaseModelOutputWithNoAttention(last_hidden_state=hidden_states, hidden_states=all_hidden_states)

    # 定义 build 方法，用于构建层次结构，对每个阶段应用命名空间
    def build(self, input_shape=None):
        for stage in self.stages:
            with tf.name_scope(stage.name):
                stage.build(None)


# 使用 keras_serializable 装饰器声明 TFConvNextV2MainLayer 类
@keras_serializable
class TFConvNextV2MainLayer(keras.layers.Layer):
    # 指定配置类为 ConvNextV2Config
    config_class = ConvNextV2Config

    # 初始化方法，接受一个 ConvNextV2Config 类型的配置对象和其他关键字参数
    def __init__(self, config: ConvNextV2Config, **kwargs):
        super().__init__(**kwargs)
        
        # 将配置对象保存到 self.config 属性中
        self.config = config
        
        # 创建 TFConvNextV2Embeddings 实例，用于处理嵌入层
        self.embeddings = TFConvNextV2Embeddings(config, name="embeddings")
        
        # 创建 TFConvNextV2Encoder 实例，用于处理编码器层
        self.encoder = TFConvNextV2Encoder(config, name="encoder")
        
        # 创建 LayerNormalization 层，用于规范化层次结构
        self.layernorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")
        
        # 创建 GlobalAvgPool2D 层，用于全局平均池化，设置 data_format 参数为 "channels_last"
        self.pooler = keras.layers.GlobalAvgPool2D(data_format="channels_last")

    # 使用 unpack_inputs 装饰器声明 call 方法，处理输入像素值、输出隐藏状态、返回字典等参数
    @unpack_inputs
    def call(
        self,
        pixel_values: TFModelInputType | None = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        # 其他输入参数省略，这里未列出的参数将由装饰器处理
        ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        # 设置是否输出隐藏状态，如果未指定则使用模型配置中的默认设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置是否返回字典形式的输出，如果未指定则使用模型配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果未提供像素值，则抛出数值错误异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 将像素值进行嵌入处理，获取嵌入输出
        embedding_output = self.embeddings(pixel_values, training=training)

        # 将嵌入输出传递给编码器进行编码
        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 获取编码器的最后隐藏状态作为编码器的第一个输出
        last_hidden_state = encoder_outputs[0]

        # 使用池化器处理最后隐藏状态，生成池化输出
        pooled_output = self.pooler(last_hidden_state)
        
        # 调整最后隐藏状态的维度顺序为NCHW格式
        last_hidden_state = tf.transpose(last_hidden_state, perm=(0, 3, 1, 2))
        
        # 对池化输出进行层归一化处理
        pooled_output = self.layernorm(pooled_output)

        # 如果设置了输出隐藏状态，则将其他隐藏状态输出也转换为NCHW格式
        if output_hidden_states:
            hidden_states = tuple([tf.transpose(h, perm=(0, 3, 1, 2)) for h in encoder_outputs[1]])

        # 如果不返回字典形式的输出，则根据设置返回相应的输出元组
        if not return_dict:
            hidden_states = hidden_states if output_hidden_states else ()
            return (last_hidden_state, pooled_output) + hidden_states

        # 返回带有池化输出和其他隐藏状态的TFBaseModelOutputWithPoolingAndNoAttention对象
        return TFBaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=hidden_states if output_hidden_states else encoder_outputs.hidden_states,
        )

    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回
        if self.built:
            return
        # 标记模型已构建
        self.built = True

        # 如果嵌入层存在，则构建嵌入层
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        
        # 如果编码器存在，则构建编码器
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        
        # 如果层归一化存在，则构建层归一化
        if getattr(self, "layernorm", None) is not None:
            with tf.name_scope(self.layernorm.name):
                self.layernorm.build([None, self.config.hidden_sizes[-1]])
"""
    <Tip>

    TensorFlow models and layers in `transformers` accept two formats as input:

    - having all inputs as keyword arguments (like PyTorch models), or
    - having all inputs as a list, tuple or dict in the first positional argument.

    The reason the second format is supported is that Keras methods prefer this format when passing inputs to models
    and layers. Because of this support, when using methods like `model.fit()` things should "just work" for you - just
    pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second
    format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with
    the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first
    positional argument:

    - a single Tensor with `pixel_values` only and nothing else: `model(pixel_values)`
    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
      `model([pixel_values, attention_mask])` or `model([pixel_values, attention_mask, token_type_ids])`
    - a dictionary with one or several input Tensors associated to the input names given in the docstring:
      `model({"pixel_values": pixel_values, "token_type_ids": token_type_ids})`

    Note that when creating models and layers with
    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don't need to worry
    about any of this, as you can just pass inputs like you would to any other Python function!

    </Tip>
"""
    Args:
        pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]`, `Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`ConvNextImageProcessor.__call__`] for details.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in
            eager mode, in graph mode the value will always be set to `True`.
"""
@add_start_docstrings(
    "The bare ConvNextV2 model outputting raw features without any specific head on top.",
    CONVNEXTV2_START_DOCSTRING,
)
"""
class TFConvNextV2Model(TFConvNextV2PreTrainedModel):
    """
    @add_start_docstrings_to_model_forward(CONVNEXTV2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    """
    def call(
        self,
        pixel_values: TFModelInputType | None = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPoolingAndNoAttention, Tuple[tf.Tensor]]:
        """
        Process inputs through the TFConvNextV2 model and return outputs.

        Args:
            pixel_values: Input pixel values (image tensors).
            output_hidden_states: Whether to output hidden states.
            return_dict: Whether to return outputs as a dictionary.
            training: Whether the model is in training mode.

        Returns:
            Either a TFBaseModelOutputWithPoolingAndNoAttention object or a tuple with tensors.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Forward pass through TFConvNextV2MainLayer
        outputs = self.convnextv2(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        if not return_dict:
            return outputs[:]  # Return all outputs as a tuple

        # Return structured outputs as TFBaseModelOutputWithPoolingAndNoAttention
        return TFBaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=outputs.last_hidden_state,
            pooler_output=outputs.pooler_output,
            hidden_states=outputs.hidden_states,
        )

    def build(self, input_shape=None):
        """
        Build method for TFConvNextV2Model. Checks if model is already built before constructing layers.

        Args:
            input_shape: Shape of the input tensors (not used in this implementation).
        """
        if self.built:
            return
        self.built = True
        if getattr(self, "convnextv2", None) is not None:
            with tf.name_scope(self.convnextv2.name):
                self.convnextv2.build(None)


"""
@add_start_docstrings(
    """
    ConvNextV2 Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    CONVNEXTV2_START_DOCSTRING,
)
"""
class TFConvNextV2ForImageClassification(TFConvNextV2PreTrainedModel, TFSequenceClassificationLoss):
    """
    Initialize TFConvNextV2ForImageClassification model.

    Args:
        config: ConvNextV2 configuration object.
        *inputs: Variable length argument list.
        **kwargs: Additional keyword arguments.

    Attributes:
        num_labels: Number of output labels for classification.
        convnextv2: TFConvNextV2MainLayer instance for feature extraction.
        classifier: Dense layer for classification head.
    """
    def __init__(self, config: ConvNextV2Config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels
        self.convnextv2 = TFConvNextV2MainLayer(config, name="convnextv2")

        # Classifier head
        self.classifier = keras.layers.Dense(
            units=config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            bias_initializer=keras.initializers.Zeros(),
            name="classifier",
        )

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CONVNEXTV2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=TFImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    # 定义模型的调用方法，用于推理过程
    def call(
        self,
        pixel_values: TFModelInputType | None = None,  # 输入像素值，可以是TensorFlow模型输入类型或者None
        output_hidden_states: Optional[bool] = None,   # 是否输出隐藏状态，默认为None，如果为True则输出隐藏状态
        return_dict: Optional[bool] = None,             # 是否返回字典格式的输出，默认为None，如果为True则返回字典
        labels: np.ndarray | tf.Tensor | None = None,   # 图像分类/回归的标签，可以是numpy数组或TensorFlow张量类型，可选
        training: Optional[bool] = False,               # 是否处于训练模式，默认为False
    ) -> Union[TFImageClassifierOutputWithNoAttention, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 根据输入或默认配置确定是否输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 根据输入或默认配置确定是否使用字典格式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果未提供像素值，则抛出数值错误
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 调用ConvNextV2模型进行前向传播
        outputs = self.convnextv2(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 根据返回格式选择池化输出
        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        # 使用分类器模型进行分类或回归预测
        logits = self.classifier(pooled_output)

        # 如果提供了标签，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        # 如果不要求字典格式的输出，则返回相应的元组
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果要求字典格式的输出，则返回TFImageClassifierOutputWithNoAttention对象
        return TFImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )

    # 构建模型的方法，用于设置模型的构建过程
    def build(self, input_shape=None):
        # 如果模型已经构建过则直接返回
        if self.built:
            return
        self.built = True

        # 如果定义了ConvNextV2模型，则构建ConvNextV2模型
        if getattr(self, "convnextv2", None) is not None:
            with tf.name_scope(self.convnextv2.name):
                self.convnextv2.build(None)

        # 如果定义了分类器模型，则根据配置构建分类器模型
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_sizes[-1]])
```