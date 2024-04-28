# `.\models\convnext\modeling_tf_convnext.py`

```py
# coding=utf-8
# 版权 2022 年 Meta Platforms 公司和 The HuggingFace 公司保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可;
# 除非符合许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据“原样”分发，不附带任何形式的担保或
# 按任何方式的条件，无论是明示的还是暗示的。
# 有关许可证的详细信息，请参见许可证。
""" TF 2.0 ConvNext 模型。"""


from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling, TFSequenceClassifierOutput
from ...modeling_tf_utils import (
    TFModelInputType,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    get_initializer,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import shape_list
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_convnext import ConvNextConfig


logger = logging.get_logger(__name__)


_CONFIG_FOR_DOC = "ConvNextConfig"
_CHECKPOINT_FOR_DOC = "facebook/convnext-tiny-224"


class TFConvNextDropPath(tf.keras.layers.Layer):
    """每个样本的 Drop path（随机深度），用于残差块的主路径。
    参考：
        （1）github.com:rwightman/pytorch-image-models
    """

    def __init__(self, drop_path: float, **kwargs):
        super().__init__(**kwargs)
        # 初始化 Drop path 的比率
        self.drop_path = drop_path

    def call(self, x: tf.Tensor, training=None):
        # 如果处于训练模式
        if training:
            # 计算保留的概率
            keep_prob = 1 - self.drop_path
            # 创建与输入张量相同形状的随机张量
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            # 应用 Drop path 并返回结果
            return (x / keep_prob) * random_tensor
        # 如果处于推断模式，直接返回输入张量
        return x


class TFConvNextEmbeddings(tf.keras.layers.Layer):
    """这个类类似于（并受到）src/transformers/models/swin/modeling_swin.py 中的 SwinEmbeddings 类。"""
    # 初始化函数，接受一个 ConvNextConfig 对象和其他关键字参数
    def __init__(self, config: ConvNextConfig, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 创建一个卷积层，用于将像素值转换为嵌入向量
        self.patch_embeddings = tf.keras.layers.Conv2D(
            filters=config.hidden_sizes[0],  # 卷积核数量等于隐藏尺寸列表的第一个元素
            kernel_size=config.patch_size,  # 卷积核大小等于补丁尺寸
            strides=config.patch_size,  # 步幅等于补丁尺寸
            name="patch_embeddings",  # 设置卷积层的名称
            kernel_initializer=get_initializer(config.initializer_range),  # 设置卷积核的初始化方式
            bias_initializer=tf.keras.initializers.Zeros(),  # 设置偏置项的初始化方式
        )
        # 创建一个 LayerNormalization 层，用于规范化嵌入向量
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="layernorm")
        # 记录输入像素值的通道数
        self.num_channels = config.num_channels
        # 记录配置信息
        self.config = config

    # 调用函数，将像素值转换为嵌入向量
    def call(self, pixel_values):
        # 如果像素值是字典，则从中获取"pixel_values"键对应的值
        if isinstance(pixel_values, dict):
            pixel_values = pixel_values["pixel_values"]

        # 断言像素值的通道维度与配置中设置的通道数相匹配
        tf.debugging.assert_equal(
            shape_list(pixel_values)[1],
            self.num_channels,
            message="Make sure that the channel dimension of the pixel values match with the one set in the configuration.",
        )

        # 将输入的像素值的维度格式从NCHW转换为NHWC，因为在CPU上`tf.keras.layers.Conv2D`不支持`NCHW`格式
        # shape = (batch_size, in_height, in_width, in_channels)
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))

        # 将像素值转换为嵌入向量
        embeddings = self.patch_embeddings(pixel_values)
        # 对嵌入向量进行规范化
        embeddings = self.layernorm(embeddings)
        # 返回嵌入向量
        return embeddings

    # 构建函数，用于构建网络层
    def build(self, input_shape=None):
        # 如果已经构建过网络层，则直接返回
        if self.built:
            return
        # 将构建标志置为True
        self.built = True
        # 如果卷积层存在，则构建卷积层
        if getattr(self, "patch_embeddings", None) is not None:
            with tf.name_scope(self.patch_embeddings.name):
                self.patch_embeddings.build([None, None, None, self.config.num_channels])
        # 如果 LayerNormalization 层存在，则构建 LayerNormalization 层
        if getattr(self, "layernorm", None) is not None:
            with tf.name_scope(self.layernorm.name):
                self.layernorm.build([None, None, None, self.config.hidden_sizes[0]])
```  
# 定义一个名为 TFConvNextLayer 的类，继承自 tf.keras.layers.Layer
"""This corresponds to the `Block` class in the original implementation.
这对应于原始实现中的 `Block` 类。

There are two equivalent implementations: [DwConv, LayerNorm (channels_first), Conv, GELU,1x1 Conv]; all in (N, C, H, W) (2) [DwConv, Permute to (N, H, W, C), LayerNorm (channels_last), Linear, GELU, Linear]; Permute back
有两种等效的实现：[DwConv, LayerNorm (channels_first), Conv, GELU,1x1 Conv]; all in (N, C, H, W) (2) [DwConv, Permute to (N, H, W, C), LayerNorm (channels_last), Linear, GELU, Linear]; Permute back

The authors used (2) as they find it slightly faster in PyTorch. Since we already permuted the inputs to follow NHWC ordering, we can just apply the operations straight-away without the permutation.
作者使用了 (2)，因为他们发现在 PyTorch 中稍微更快。由于我们已经排列了输入以遵循 NHWC 排列顺序，所以我们可以直接应用操作而无需排列。

Args:
    config ([`ConvNextConfig`]): Model configuration class.
    dim (`int`): Number of input channels.
    drop_path (`float`): Stochastic depth rate. Default: 0.0.
"""
# 初始化方法，接受 config、dim 和 drop_path 作为参数
def __init__(self, config, dim, drop_path=0.0, **kwargs):
    # 调用父类的初始化方法
    super().__init__(**kwargs)
    # 初始化属性 dim 和 config
    self.dim = dim
    self.config = config
    # 创建深度可分离卷积对象，设置参数
    self.dwconv = tf.keras.layers.Conv2D(
        filters=dim,
        kernel_size=7,
        padding="same",
        groups=dim,
        kernel_initializer=get_initializer(config.initializer_range),
        bias_initializer="zeros",
        name="dwconv",
    )  # depthwise conv
    # 创建层归一化对象，设置参数
    self.layernorm = tf.keras.layers.LayerNormalization(
        epsilon=1e-6,
        name="layernorm",
    )
    # 创建第一个点卷积对象，设置参数
    self.pwconv1 = tf.keras.layers.Dense(
        units=4 * dim,
        kernel_initializer=get_initializer(config.initializer_range),
        bias_initializer="zeros",
        name="pwconv1",
    )  # pointwise/1x1 convs, implemented with linear layers
    # 获取激活函数并设置为对象的属性
    self.act = get_tf_activation(config.hidden_act)
    # 创建第二个点卷积对象，设置参数
    self.pwconv2 = tf.keras.layers.Dense(
        units=dim,
        kernel_initializer=get_initializer(config.initializer_range),
        bias_initializer="zeros",
        name="pwconv2",
    )
    # 使用 `layers.Activation` 替代 `tf.identity` 来更好地控制 `training` 行为。
    # 根据 drop_path 的值实例化 TFConvNextDropPath 或 tf.keras.layers.Activation 对象
    self.drop_path = (
        TFConvNextDropPath(drop_path, name="drop_path")
        if drop_path > 0.0
        else tf.keras.layers.Activation("linear", name="drop_path")
    )
    # 建立该层的权重，用于缩放层的输出
    def build(self, input_shape: tf.TensorShape = None):
        # 如果层的缩放初始值大于0，则创建一个名为"layer_scale_parameter"的可训练权重
        self.layer_scale_parameter = (
            self.add_weight(
                shape=(self.dim,),
                initializer=tf.keras.initializers.Constant(value=self.config.layer_scale_init_value),
                trainable=True,
                name="layer_scale_parameter",
            )
            if self.config.layer_scale_init_value > 0
            else None
        )

        # 如果已经建立，直接返回
        if self.built:
            return
        self.built = True
        # 如果存在dwconv，则使用其命名空间构建
        if getattr(self, "dwconv", None) is not None:
            with tf.name_scope(self.dwconv.name):
                self.dwconv.build([None, None, None, self.dim])
        # 如果存在layernorm，则使用其命名空间构建
        if getattr(self, "layernorm", None) is not None:
            with tf.name_scope(self.layernorm.name):
                self.layernorm.build([None, None, None, self.dim])
        # 如果存在pwconv1，则使用其命名空间构建
        if getattr(self, "pwconv1", None) is not None:
            with tf.name_scope(self.pwconv1.name):
                self.pwconv1.build([None, None, self.dim])
        # 如果存在pwconv2，则使用其命名空间构建
        if getattr(self, "pwconv2", None) is not None:
            with tf.name_scope(self.pwconv2.name):
                self.pwconv2.build([None, None, 4 * self.dim])
        # 如果存在drop_path，则使用其命名空间构建
        if getattr(self, "drop_path", None) is not None:
            with tf.name_scope(self.drop_path.name):
                self.drop_path.build(None)

    # 调用该层进行前向传播
    def call(self, hidden_states, training=False):
        # 保存输入
        input = hidden_states
        # 使用dwconv进行深度卷积
        x = self.dwconv(hidden_states)
        # 对x进行layernorm
        x = self.layernorm(x)
        # 对x进行pwconv1
        x = self.pwconv1(x)
        # 对x进行激活函数操作
        x = self.act(x)
        # 对x进行pwconv2
        x = self.pwconv2(x)

        # 如果存在layer_scale_parameter，则将x乘以该参数
        if self.layer_scale_parameter is not None:
            x = self.layer_scale_parameter * x

        # 对输入进行加权，然后加上dropout path的输出
        x = input + self.drop_path(x, training=training)
        # 返回结果
        return x
class TFConvNextStage(tf.keras.layers.Layer):
    """ConvNext stage, consisting of an optional downsampling layer + multiple residual blocks.

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
        config: ConvNextConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        stride: int = 2,
        depth: int = 2,
        drop_path_rates: Optional[List[float]] = None,
        **kwargs,
    ):
        # 调用父类构造函数
        super().__init__(**kwargs)
        # 如果输入通道数不等于输出通道数 或者 步幅大于1，则定义下采样层
        if in_channels != out_channels or stride > 1:
            self.downsampling_layer = [
                tf.keras.layers.LayerNormalization(
                    epsilon=1e-6,
                    name="downsampling_layer.0",
                ),
                # 此层的输入将遵循NHWC格式，因为我们在`TFConvNextEmbeddings`层中将输入从NCHW转置为NHWC。
                # 从此刻开始，模型中所有的输出都将是NHWC格式，直到最终输出，再次转换为NCHW格式。
                tf.keras.layers.Conv2D(
                    filters=out_channels,
                    kernel_size=kernel_size,
                    strides=stride,
                    kernel_initializer=get_initializer(config.initializer_range),
                    bias_initializer=tf.keras.initializers.Zeros(),
                    name="downsampling_layer.1",
                ),
            ]
        else:
            self.downsampling_layer = [tf.identity]

        drop_path_rates = drop_path_rates or [0.0] * depth
        # 创建多个TFConvNextLayer，作为多个残差块
        self.layers = [
            TFConvNextLayer(
                config,
                dim=out_channels,
                drop_path=drop_path_rates[j],
                name=f"layers.{j}",
            )
            for j in range(depth)
        ]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def call(self, hidden_states):
        # 应用下采样层
        for layer in self.downsampling_layer:
            hidden_states = layer(hidden_states)
        # 应用多个残差块
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states
    def build(self, input_shape=None):
        # 如果已经构建过网络，则直接返回，不进行重复构建
        if self.built:
            return
        # 设置构建完成标志为 True
        self.built = True
        # 如果存在子层，则依次构建子层
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                # 使用 TensorFlow 的 name_scope 为每个子层创建独立的命名空间
                with tf.name_scope(layer.name):
                    # 构建子层
                    layer.build(None)
        # 如果输入通道数不等于输出通道数或步长大于1
        if self.in_channels != self.out_channels or self.stride > 1:
            # 使用 TensorFlow 的 name_scope 为下采样层1创建独立的命名空间
            with tf.name_scope(self.downsampling_layer[0].name):
                # 构建下采样层1
                self.downsampling_layer[0].build([None, None, None, self.in_channels])
            # 使用 TensorFlow 的 name_scope 为下采样层2创建独立的命名空间
            with tf.name_scope(self.downsampling_layer[1].name):
                # 构建下采样层2
                self.downsampling_layer[1].build([None, None, None, self.in_channels])
# 定义 TFConvNextEncoder 类，用于构建 ConvNext 模型的编码器部分
class TFConvNextEncoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 初始化空列表，用于存储编码器的各个阶段
        self.stages = []
        # 计算每个阶段的 drop_path_rates
        drop_path_rates = tf.linspace(0.0, config.drop_path_rate, sum(config.depths))
        drop_path_rates = tf.split(drop_path_rates, config.depths)
        drop_path_rates = [x.numpy().tolist() for x in drop_path_rates]
        # 初始化 prev_chs 为输入特征的通道数
        prev_chs = config.hidden_sizes[0]
        # 遍历每个阶段
        for i in range(config.num_stages):
            # 获取当前阶段的输出通道数
            out_chs = config.hidden_sizes[i]
            # 创建 TFConvNextStage 实例，并加入到 self.stages 列表中
            stage = TFConvNextStage(
                config,
                in_channels=prev_chs,
                out_channels=out_chs,
                stride=2 if i > 0 else 1,
                depth=config.depths[i],
                drop_path_rates=drop_path_rates[i],
                name=f"stages.{i}",
            )
            self.stages.append(stage)
            # 更新 prev_chs 为当前阶段的输出通道数，供下一个阶段使用

    # 定义 call 方法，用于调用编码器
    def call(self, hidden_states, output_hidden_states=False, return_dict=True):
        # 初始化 all_hidden_states 为空元组或 None，根据 output_hidden_states 参数决定是否保存隐藏状态
        all_hidden_states = () if output_hidden_states else None

        # 遍历每个阶段，对隐藏状态进行编码
        for i, layer_module in enumerate(self.stages):
            # 如果需要保存隐藏状态，则将隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 调用当前阶段的 call 方法，对隐藏状态进行编码
            hidden_states = layer_module(hidden_states)

        # 如果需要保存隐藏状态，则将最后一个隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典形式的输出，则将隐藏状态和所有隐藏状态返回为元组形式
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        # 返回字典形式的输出，包括最后一个隐藏状态和所有隐藏状态
        return TFBaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states)

    # 定义 build 方法，用于构建编码器
    def build(self, input_shape=None):
        # 遍历每个阶段，构建阶段的网络层
        for stage in self.stages:
            with tf.name_scope(stage.name):
                stage.build(None)


# 定义 TFConvNextMainLayer 类，用于构建 ConvNext 模型的主体部分
@keras_serializable
class TFConvNextMainLayer(tf.keras.layers.Layer):
    # 指定配置类为 ConvNextConfig
    config_class = ConvNextConfig

    # 初始化方法
    def __init__(self, config: ConvNextConfig, add_pooling_layer: bool = True, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 存储配置信息
        self.config = config
        # 创建 TFConvNextEmbeddings 实例，用于处理输入数据的嵌入层
        self.embeddings = TFConvNextEmbeddings(config, name="embeddings")
        # 创建 TFConvNextEncoder 实例，用于处理嵌入层的编码器
        self.encoder = TFConvNextEncoder(config, name="encoder")
        # 创建 LayerNormalization 层，用于归一化编码器输出
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")
        # 如果需要添加池化层，则创建 GlobalAvgPool2D 层
        self.pooler = tf.keras.layers.GlobalAvgPool2D(data_format="channels_first") if add_pooling_layer else None

    # 定义 call 方法，用于调用主体部分的网络
    @unpack_inputs
    def call(
        self,
        pixel_values: TFModelInputType | None = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    # 定义模型的前向传播方法，接受像素值作为输入，并返回模型输出或元组，包含模型输出和隐藏状态
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        # 如果输出隐藏状态为None，则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果返回字典为None，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果像素值为None，则引发值错误
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 使用嵌入层将像素值转换为嵌入向量
        embedding_output = self.embeddings(pixel_values, training=training)

        # 使用编码器对嵌入向量进行编码
        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 获取编码器的最后隐藏状态
        last_hidden_state = encoder_outputs[0]
        # 将输出格式转换为NCHW，以在模块中保持一致性
        last_hidden_state = tf.transpose(last_hidden_state, perm=(0, 3, 1, 2))
        # 对最后隐藏状态应用池化和层归一化
        pooled_output = self.layernorm(self.pooler(last_hidden_state))

        # 如果输出隐藏状态为True，则将其他隐藏状态输出也转换为NCHW格式
        if output_hidden_states:
            hidden_states = tuple([tf.transpose(h, perm=(0, 3, 1, 2)) for h in encoder_outputs[1]])

        # 如果返回字典为False，则根据输出隐藏状态返回相应的输出
        if not return_dict:
            hidden_states = hidden_states if output_hidden_states else ()
            return (last_hidden_state, pooled_output) + hidden_states

        # 否则，返回带池化的基本模型输出，包括最后隐藏状态、池化输出和其他隐藏状态
        return TFBaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=hidden_states if output_hidden_states else encoder_outputs.hidden_states,
        )

    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回
        if self.built:
            return
        # 将模型标记为已构建
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
class TFConvNextPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置该类的配置类为ConvNextConfig
    config_class = ConvNextConfig
    # 设置基础模型前缀为"convnext"
    base_model_prefix = "convnext"
    # 设置主要输入名称为"pixel_values"
    main_input_name = "pixel_values"


CONVNEXT_START_DOCSTRING = r"""
    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.

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

    Parameters:
        config ([`ConvNextConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

CONVNEXT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`):
            传入的像素值。像素值可以使用 [`AutoImageProcessor`] 获得。具体请参考 [`ConvNextImageProcessor.__call__`]。
            值的类型可以是 `np.ndarray`, `tf.Tensor`, `List[tf.Tensor]`, `Dict[str, tf.Tensor]` 或 `Dict[str, np.ndarray]`。
            每个例子的形状必须为 `(batch_size, num_channels, height, width)`。

        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。详见返回的张量中的 `hidden_states` 部分。该参数只能在 eager 模式下使用，在 graph 模式下会使用配置中的值。

        return_dict (`bool`, *optional*):
            是否返回一个 [`~utils.ModelOutput`] 而不是一个普通的元组。该参数只能在 eager 模式下使用，在 graph 模式下该值总是会设置为 True。
"""
定义一个 TFConvNextModel 类，继承自 TFConvNextPreTrainedModel 类，表示 ConvNext 模型。

该模型不带特定头部，直接输出原始特征。

参数：
    config: ConvNext 配置对象。
    *inputs: 可变位置参数，传入给父类构造函数的参数。
    add_pooling_layer: 是否添加池化层，默认为 True。
    **kwargs: 关键字参数，传入给父类构造函数的参数。
"""
@add_start_docstrings(
    "The bare ConvNext model outputting raw features without any specific head on top.",
    CONVNEXT_START_DOCSTRING,
)
class TFConvNextModel(TFConvNextPreTrainedModel):
    def __init__(self, config, *inputs, add_pooling_layer=True, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 初始化 ConvNextMainLayer 类，传入 ConvNext 配置对象和是否添加池化层的参数
        self.convnext = TFConvNextMainLayer(config, add_pooling_layer=add_pooling_layer, name="convnext")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CONVNEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        pixel_values: TFModelInputType | None = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        r"""
        Returns:

        Examples:

        ```py
        >>> from transformers import AutoImageProcessor, TFConvNextModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/convnext-tiny-224")
        >>> model = TFConvNextModel.from_pretrained("facebook/convnext-tiny-224")

        >>> inputs = image_processor(images=image, return_tensors="tf")
        >>> outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        # 如果未指定像素值，则引发 ValueError 异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 根据是否返回字典和是否输出隐藏状态来确定输出设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 ConvNextMainLayer 的 call 方法
        outputs = self.convnext(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 如果不返回字典，则返回元组形式的输出
        if not return_dict:
            return (outputs[0],) + outputs[1:]

        # 返回 TFBaseModelOutputWithPooling 对象，包含最后的隐藏状态、池化输出和所有隐藏状态
        return TFBaseModelOutputWithPooling(
            last_hidden_state=outputs.last_hidden_state,
            pooler_output=outputs.pooler_output,
            hidden_states=outputs.hidden_states,
        )

    def build(self, input_shape=None):
        # 如果已构建，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果 convnext 属性存在，则构建 convnext 层
        if getattr(self, "convnext", None) is not None:
            with tf.name_scope(self.convnext.name):
                self.convnext.build(None)


@add_start_docstrings(
    """
    ConvNext Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    CONVNEXT_START_DOCSTRING,
)
"""
# 定义 TFConvNextForImageClassification 类，继承自 TFConvNextPreTrainedModel 和 TFSequenceClassificationLoss
class TFConvNextForImageClassification(TFConvNextPreTrainedModel, TFSequenceClassificationLoss):
    # 初始化方法，接受配置参数和其他输入
    def __init__(self, config: ConvNextConfig, *inputs, **kwargs):
        # 调用父类初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 设置标签数量
        self.num_labels = config.num_labels
        # 创建 TFConvNextMainLayer 对象
        self.convnext = TFConvNextMainLayer(config, name="convnext")

        # 分类器头部
        self.classifier = tf.keras.layers.Dense(
            units=config.num_labels,  # 设置输出单元数量
            kernel_initializer=get_initializer(config.initializer_range),  # 设置权重初始化器
            bias_initializer="zeros",  # 设置偏置初始化器
            name="classifier",  # 设置层名称
        )
        self.config = config

    # 注解，解包输入参数
    @unpack_inputs
    # 添加模型正向传播的文档字符串
    @add_start_docstrings_to_model_forward(CONVNEXT_INPUTS_DOCSTRING)
    # 替换返回值的文档字符串
    @replace_return_docstrings(output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    # 定义模型正向传播方法
    def call(
        # 像素值输入，可能为 None
        pixel_values: TFModelInputType | None = None,
        # 是否输出隐藏状态，默认为 None
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典，默认为 None
        return_dict: Optional[bool] = None,
        # 标签数组或张量，可能为 None
        labels: np.ndarray | tf.Tensor | None = None,
        # 是否训练，默认为 False
        training: Optional[bool] = False,
    ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```py
        >>> from transformers import AutoImageProcessor, TFConvNextForImageClassification
        >>> import tensorflow as tf
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/convnext-tiny-224")
        >>> model = TFConvNextForImageClassification.from_pretrained("facebook/convnext-tiny-224")

        >>> inputs = image_processor(images=image, return_tensors="tf")
        >>> outputs = model(**inputs)
        >>> logits = outputs.logits
        >>> # model predicts one of the 1000 ImageNet classes
        >>> predicted_class_idx = tf.math.argmax(logits, axis=-1)[0]
        >>> print("Predicted class:", model.config.id2label[int(predicted_class_idx)])
        ```"""
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果像素值为None，则抛出错误
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 使用ConvNext模型进行前向传播
        outputs = self.convnext(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 如果return_dict为True，则使用pooler_output作为pooled_output，否则使用outputs[1]
        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        # 将pooled_output传入分类器以获取logits
        logits = self.classifier(pooled_output)
        # 如果labels为None，则loss为None，否则使用hf_compute_loss计算loss
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        # 如果return_dict为False，则输出不包含loss，否则输出包含loss
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回TFSequenceClassifierOutput对象，包含loss、logits和hidden_states
        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )

    def build(self, input_shape=None):
        # 如果已经构建，则直接返回
        if self.built:
            return
        self.built = True
        # 如果convnext模型不为None，则构建convnext模型
        if getattr(self, "convnext", None) is not None:
            with tf.name_scope(self.convnext.name):
                self.convnext.build(None)
        # 如果classifier模型不为None，则构建classifier模型
        if getattr(self, "classifier", None) is not None:
            if hasattr(self.classifier, "name"):
                with tf.name_scope(self.classifier.name):
                    self.classifier.build([None, None, self.config.hidden_sizes[-1]])
```py  
```