# `.\models\convnextv2\modeling_tf_convnextv2.py`

```py
# 设定文件编码
# 版权声明
#
# 引入未来版本的特性
# 引入类型提示
import numpy as np
import tensorflow as tf

# 引入其他模块
# 引入自定义激活函数
# 引入输出模型
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithNoAttention,
    TFBaseModelOutputWithPooling,
    TFBaseModelOutputWithPoolingAndNoAttention,
    TFImageClassifierOutputWithNoAttention,
)
# 引入工具模块
from ...modeling_tf_utils import (
    TFModelInputType,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    get_initializer,
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
# 引入配置信息
from .configuration_convnextv2 import ConvNextV2Config

# 获取日志记录器
# 通用文档字符串
_CONFIG_FOR_DOC = "ConvNextV2Config"
# 基础文档字符串
_CHECKPOINT_FOR_DOC = "facebook/convnextv2-tiny-1k-224"
_EXPECTED_OUTPUT_SHAPE = [1, 768, 7, 7]
# 图像分类文档字符串
_IMAGE_CLASS_CHECKPOINT = "facebook/convnextv2-tiny-1k-224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"
# 预训练模型列表
CONVNEXTV2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/convnextv2-tiny-1k-224",
    # 查看所有ConvNextV2模型：https://huggingface.co/models?filter=convnextv2
]

# 从transformers.models.convnext.modeling_tf_convnext.TFConvNextDropPath中复制代码，并将ConvNext改为ConvNextV2
# 实现Drop paths (Stochastic Depth)
class TFConvNextV2DropPath(tf.keras.layers.Layer):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    References:
        (1) github.com:rwightman/pytorch-image-models
    """

    def __init__(self, drop_path: float, **kwargs):
        super().__init__(**kwargs)
        self.drop_path = drop_path

    def call(self, x: tf.Tensor, training=None):
        if training:
            keep_prob = 1 - self.drop_path
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x

# ConvNextV2GRN（全局响应归一化）层
class TFConvNextV2GRN(tf.keras.layers.Layer):
    """GRN (Global Response Normalization) layer"""
    # 初始化函数，接受配置和维度参数
    def __init__(self, config: ConvNextV2Config, dim: int, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 保存传入的维度参数
        self.dim = dim

    # 构建函数，用于构建层的权重
    def build(self, input_shape: tf.TensorShape = None):
        # PT的`nn.Parameters`必须映射到TF层权重，以继承相同的名称层次结构（反之亦然）
        # 添加权重参数，名称为"weight"，初始化为全零，形状为(1, 1, 1, self.dim)
        self.weight = self.add_weight(
            name="weight",
            shape=(1, 1, 1, self.dim),
            initializer=tf.keras.initializers.Zeros(),
        )
        # 添加偏置参数，名称为"bias"，初始化为全零，形状为(1, 1, 1, self.dim)
        self.bias = self.add_weight(
            name="bias",
            shape=(1, 1, 1, self.dim),
            initializer=tf.keras.initializers.Zeros(),
        )
        # 调用父类的构建函数
        return super().build(input_shape)

    # 调用函数，处理隐藏状态的特征转换
    def call(self, hidden_states: tf.Tensor):
        # 计算全局特征的范数，使用欧氏距离，沿着指定轴(axis=(1, 2))，保持维度
        global_features = tf.norm(hidden_states, ord="euclidean", axis=(1, 2), keepdims=True)
        # 对全局特征进行归一化处理，除以平均值并加上一个极小值以避免除零错误
        norm_features = global_features / (tf.reduce_mean(global_features, axis=-1, keepdims=True) + 1e-6)
        # 使用权重和偏置对隐藏状态进行特征转换
        hidden_states = self.weight * (hidden_states * norm_features) + self.bias + hidden_states
        # 返回转换后的隐藏状态
        return hidden_states
# 从transformers.models.convnext.modeling_tf_convnext.TFConvNextEmbeddings中复制类，并将ConvNext改为ConvNextV2
class TFConvNextV2Embeddings(tf.keras.layers.Layer):
    """这个类与 src/transformers/models/swin/modeling_swin.py 中的 SwinEmbeddings 类相似（并受其启发）。
    """

    def __init__(self, config: ConvNextV2Config, **kwargs):
        super().__init__(**kwargs)
        
        # 创建一个 tf.keras.layers.Conv2D 层，用于对图像进行卷积并生成嵌入特征
        self.patch_embeddings = tf.keras.layers.Conv2D(
            filters=config.hidden_sizes[0],
            kernel_size=config.patch_size,
            strides=config.patch_size,
            name="patch_embeddings",
            kernel_initializer=get_initializer(config.initializer_range),
            bias_initializer=tf.keras.initializers.Zeros(),
        )
        
        # 创建一个 LayerNormalization 层，用于对嵌入特征进行标准化处理
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="layernorm")
        
        # 记录通道数和配置信息
        self.num_channels = config.num_channels
        self.config = config

    def call(self, pixel_values):
        # 如果 pixel_values 是字典，则获取其键为 "pixel_values" 的值作为输入数据
        if isinstance(pixel_values, dict):
            pixel_values = pixel_values["pixel_values"]

        # 检查 pixel_values 的通道维度是否与配置中设置的一致
        tf.debugging.assert_equal(
            shape_list(pixel_values)[1],
            self.num_channels,
            message="Make sure that the channel dimension of the pixel values match with the one set in the configuration.",
        )

        # 在 CPU 上运行时，`tf.keras.layers.Conv2D` 不支持 `NCHW` 格式，因此将输入格式从 `NCHW` 改为 `NHWC`
        # shape = (batch_size, in_height, in_width, in_channels)
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))

        # 对像素值进行嵌入处理
        embeddings = self.patch_embeddings(pixel_values)
        
        # 对嵌入特征进行标准化处理
        embeddings = self.layernorm(embeddings)
        
        # 返回嵌入特征
        return embeddings

    def build(self, input_shape=None):
        # 如果已经构建，则直接返回
        if self.built:
            return
        self.built = True
        
        # 构建 patch_embeddings 层，并设置其输入形状
        if getattr(self, "patch_embeddings", None) is not None:
            with tf.name_scope(self.patch_embeddings.name):
                self.patch_embeddings.build([None, None, None, self.config.num_channels])
        
        # 构建 layernorm 层，并设置其输入形状
        if getattr(self, "layernorm", None) is not None:
            with tf.name_scope(self.layernorm.name):
                self.layernorm.build([None, None, None, self.config.hidden_sizes[0]])


class TFConvNextV2Layer(tf.keras.layers.Layer):
    """这对应于原始实现中的 "Block" 类。

    存在两个等价的实现方式：[DwConv, LayerNorm (channels_first), Conv, GELU,1x1 Conv]; 全部在 (N, C,
    H, W) 维度下处理 (2) [DwConv, Permute to (N, H, W, C), LayerNorm (channels_last), Linear, GELU, Linear]; Permute back

    作者在 PyTorch 中使用了 (2)，因为他们发现它稍微更快一些。由于我们已经将输入调整为遵循 NHWC 排序，因此可以直接应用这些操作，而不需要重新排列。
    Args:
        config (`ConvNextV2Config`):
            Model configuration class.  # 模型配置类
        dim (`int`):
            Number of input channels.  # 输入通道数
        drop_path (`float`, defaults to 0.0):
            Stochastic depth rate.  # 随机深度率
    """

    def __init__(self, config: ConvNextV2Config, dim: int, drop_path: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.config = config
        self.dwconv = tf.keras.layers.Conv2D(
            filters=dim,
            kernel_size=7,
            padding="same",
            groups=dim,
            kernel_initializer=get_initializer(config.initializer_range),
            bias_initializer=tf.keras.initializers.Zeros(),
            name="dwconv",
        )  # depthwise conv  # 深度可分离卷积
        self.layernorm = tf.keras.layers.LayerNormalization(
            epsilon=1e-6,
            name="layernorm",
        )  # 层归一化
        self.pwconv1 = tf.keras.layers.Dense(
            units=4 * dim,
            kernel_initializer=get_initializer(config.initializer_range),
            bias_initializer=tf.keras.initializers.Zeros(),
            name="pwconv1",
        )  # pointwise/1x1 convs, implemented with linear layers  # 点卷积/1x1卷积，用线性层实现
        self.act = get_tf_activation(config.hidden_act)  # 获取激活函数
        self.grn = TFConvNextV2GRN(config, 4 * dim, dtype=tf.float32, name="grn")  # ConvNextV2GRN实例化
        self.pwconv2 = tf.keras.layers.Dense(
            units=dim,
            kernel_initializer=get_initializer(config.initializer_range),
            bias_initializer=tf.keras.initializers.Zeros(),
            name="pwconv2",
        )  # 点卷积层
        # Using `layers.Activation` instead of `tf.identity` to better control `training`
        # behaviour.
        self.drop_path = (
            TFConvNextV2DropPath(drop_path, name="drop_path")
            if drop_path > 0.0
            else tf.keras.layers.Activation("linear", name="drop_path")
        )  # 使用`layers.Activation`而不是`tf.identity`来更好地控制`training`行为

    def call(self, hidden_states, training=False):
        input = hidden_states
        x = self.dwconv(hidden_states)  # 深度可分离卷积
        x = self.layernorm(x)  # 层归一化
        x = self.pwconv1(x)  # 点卷积层1
        x = self.act(x)  # 激活函数
        x = self.grn(x)  # ConvNextV2GRN实例化的调用
        x = self.pwconv2(x)  # 点卷积层2
        x = self.drop_path(x, training=training)  # 随机深度率
        x = input + x  # 残差连接
        return x
    # 定义函数用于构建神经网络模型的层，如果已经构建过，则直接返回
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在深度可分离卷积层，构建它
        if getattr(self, "dwconv", None) is not None:
            # 使用 tf.name_scope 设置命名空间
            with tf.name_scope(self.dwconv.name):
                # 根据给定的输入形状进行构建
                self.dwconv.build([None, None, None, self.dim])
        # 如果存在层归一化层，构建它
        if getattr(self, "layernorm", None) is not None:
            with tf.name_scope(self.layernorm.name):
                self.layernorm.build([None, None, None, self.dim])
        # 如果存在逐点卷积层 pwconv1，构建它
        if getattr(self, "pwconv1", None) is not None:
            with tf.name_scope(self.pwconv1.name):
                self.pwconv1.build([None, None, self.dim])
        # 如果存在全局归一化层 grn，构建它
        if getattr(self, "grn", None) is not None:
            with tf.name_scope(self.grn.name):
                self.grn.build(None)
        # 如果存在第二个逐点卷积层 pwconv2，构建它
        if getattr(self, "pwconv2", None) is not None:
            with tf.name_scope(self.pwconv2.name):
                self.pwconv2.build([None, None, 4 * self.dim])
        # 如果存在 drop_path 层，构建它
        if getattr(self, "drop_path", None) is not None:
            with tf.name_scope(self.drop_path.name):
                self.drop_path.build(None)
# 从transformers.models.convnext.modeling_tf_convnext.TFConvNextStage复制并修改为ConvNextV2
class TFConvNextV2Stage(tf.keras.layers.Layer):
    """ConvNextV2阶段，包括可选的下采样层+多个残差块。

    Args:
        config (`ConvNextV2V2Config`):
            模型配置类。
        in_channels (`int`):
            输入通道数。
        out_channels (`int`):
            输出通道数。
        depth (`int`):
            残差块数量。
        drop_path_rates(`List[float]`):
            每层的随机深度率。
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
        # 如果输入通道数不等于输出通道数或步长大于1
        if in_channels != out_channels or stride > 1:
            # 下采样层包含层归一化和卷积层
            self.downsampling_layer = [
                tf.keras.layers.LayerNormalization(
                    epsilon=1e-6,
                    name="downsampling_layer.0",
                ),
                # 输入到此层将遵循NHWC格式，因为我们在`TFConvNextV2Embeddings`层中将输入从NCHW转置为NHWC。
                # 从此点开始，模型中所有的输出都将是NHWC，直到我们再次更改为NCHW的输出。
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
            # 如果输入通道数等于输出通道数且步长为1，则使用恒等函数作为下采样层
            self.downsampling_layer = [tf.identity]

        # 如果未提供drop_path_rates，则初始化为与深度相同数量的0.0
        drop_path_rates = drop_path_rates or [0.0] * depth
        # 创建多个TFConvNextV2Layer层组成的列表，每个层都有相应的drop_path_rate
        self.layers = [
            TFConvNextV2Layer(
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

    # 前向传播函数
    def call(self, hidden_states):
        # 应用下采样层
        for layer in self.downsampling_layer:
            hidden_states = layer(hidden_states)
        # 应用多个残差块层
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        # 返回最终的隐藏状态
        return hidden_states
    # 定义一个方法，用于构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建完成，则直接返回
        if self.built:
            return
        # 标记模型已经构建完成
        self.built = True
        # 检查是否存在子层
        if getattr(self, "layers", None) is not None:
            # 遍历子层
            for layer in self.layers:
                # 使用子层的名称作为作用域，构建子层
                with tf.name_scope(layer.name):
                    layer.build(None)
        # 如果输入通道数和输出通道数不相等，或者步长大于1
        if self.in_channels != self.out_channels or self.stride > 1:
            # 使用降采样层的名称作为作用域，构建降采样层1
            with tf.name_scope(self.downsampling_layer[0].name):
                self.downsampling_layer[0].build([None, None, None, self.in_channels])
            # 使用降采样层的名称作为作用域，构建降采样层2
            with tf.name_scope(self.downsampling_layer[1].name):
                self.downsampling_layer[1].build([None, None, None, self.in_channels])
# 定义 TFConvNextV2Encoder 类，继承自 tf.keras.layers.Layer
class TFConvNextV2Encoder(tf.keras.layers.Layer):
    # 初始化方法，接受 config 参数并调用父类初始化方法
    def __init__(self, config: ConvNextV2Config, **kwargs):
        super().__init__(**kwargs)
        # 创建存储阶段对象的列表
        self.stages = []
        # 根据 config 中的参数创建 drop_path_rates 列表
        drop_path_rates = tf.linspace(0.0, config.drop_path_rate, sum(config.depths))
        drop_path_rates = tf.split(drop_path_rates, config.depths)
        drop_path_rates = [x.numpy().tolist() for x in drop_path_rates]
        prev_chs = config.hidden_sizes[0]
        # 遍历阶段数量，创建 TFConvNextV2Stage 对象并加入到 self.stages 列表中
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

    # 调用方法，接受 hidden_states、output_hidden_states 和 return_dict 作为参数
    def call(
        self,
        hidden_states: tf.Tensor,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, TFBaseModelOutputWithNoAttention]:
        # 初始化 all_hidden_states 变量
        all_hidden_states = () if output_hidden_states else None
        # 遍历 self.stages 列表，调用每个阶段的 call 方法
        for i, layer_module in enumerate(self.stages):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states = layer_module(hidden_states)
        # 根据 output_hidden_states 的值决定是否将 hidden_states 加入到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        # 根据 return_dict 的值决定返回结果的形式
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)
        # 返回 TFBaseModelOutputWithNoAttention 对象
        return TFBaseModelOutputWithNoAttention(last_hidden_state=hidden_states, hidden_states=all_hidden_states)

    # 构建方法，接受 input_shape 作为参数
    def build(self, input_shape=None):
        # 遍历 self.stages 列表，为每个阶段设置命名空间并调用其 build 方法
        for stage in self.stages:
            with tf.name_scope(stage.name):
                stage.build(None)


# 使用 keras_serializable 装饰器声明 TFConvNextV2MainLayer 类可序列化
@keras_serializable
# 定义 TFConvNextV2MainLayer 类，继承自 tf.keras.layers.Layer
class TFConvNextV2MainLayer(tf.keras.layers.Layer):
    # 设置 config_class 属性为 ConvNextV2Config 类
    config_class = ConvNextV2Config
    # 初始化方法，接受 config 参数并调用父类初始化方法
    def __init__(self, config: ConvNextV2Config, **kwargs):
        super().__init__(**kwargs)
        # 存储 config 参数
        self.config = config
        # 创建 TFConvNextV2Embeddings 对象并命名为 embeddings
        self.embeddings = TFConvNextV2Embeddings(config, name="embeddings")
        # 创建 TFConvNextV2Encoder 对象并命名为 encoder
        self.encoder = TFConvNextV2Encoder(config, name="encoder")
        # 创建 LayerNormalization 层并命名为 layernorm
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")
        # 创建 GlobalAvgPool2D 层并命名为 pooler
        # 设置 data_format 为 "channels_last"
        self.pooler = tf.keras.layers.GlobalAvgPool2D(data_format="channels_last")

    # 调用方法，接受 pixel_values、output_hidden_states、return_dict 和 training 作为参数
    def call(
        self,
        pixel_values: TFModelInputType | None = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    # 定义函数，返回值类型为 TFBaseModelOutputWithPooling 或者包含了 tf.Tensor 的元组
    def call(
        self,
        pixel_values: tf.Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = None,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:

        # 如果 output_hidden_states 为 None，则使用模型配置中的值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果 return_dict 为 None，则使用模型配置中的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果未提供 pixel_values，则抛出 ValueError 异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 使用输入的像素值进行嵌入
        embedding_output = self.embeddings(pixel_values, training=training)

        # 使用嵌入的输出进行编码
        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 获取编码器的最后一个隐藏状态
        last_hidden_state = encoder_outputs[0]

        # 使用池化器对最后一个隐藏状态进行池化
        pooled_output = self.pooler(last_hidden_state)
        # 调整最后一个隐藏状态的输出格式为 NCHW，保持模块间的一致性
        last_hidden_state = tf.transpose(last_hidden_state, perm=(0, 3, 1, 2))
        # 对池化输出进行 Layernorm 处理
        pooled_output = self.layernorm(pooled_output)

        # 如果需要输出所有隐藏状态
        if output_hidden_states:
            # 将所有隐藏状态转换为 NCHW 格式
            hidden_states = tuple([tf.transpose(h, perm=(0, 3, 1, 2)) for h in encoder_outputs[1]])

        # 如果不需要返回字典，则根据需要返回隐藏状态
        if not return_dict:
            hidden_states = hidden_states if output_hidden_states else ()
            return (last_hidden_state, pooled_output) + hidden_states

        # 如果需要返回字典，则创建一个包含所有输出的字典
        return TFBaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=hidden_states if output_hidden_states else encoder_outputs.hidden_states,
        )

    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回
        if self.built:
            return
        # 标记模型为已构建状态
        self.built = True
        # 如果模型中包含嵌入层，则构建嵌入层
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        # 如果模型中包含编码器，则构建编码器
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果模型中包含 Layernorm 层，则构建 Layernorm 层
        if getattr(self, "layernorm", None) is not None:
            with tf.name_scope(self.layernorm.name):
                self.layernorm.build([None, self.config.hidden_sizes[-1]])
class TFConvNextV2PreTrainedModel(TFPreTrainedModel):
    """
    一个抽象类，用于处理权重初始化和一个简单的接口，用于下载和加载预训练模型。
    """

    config_class = ConvNextV2Config
    base_model_prefix = "convnextv2"
    main_input_name = "pixel_values"


CONVNEXTV2_START_DOCSTRING = r"""
    这个模型继承自[`TFPreTrainedModel`]。查看超类文档以了解库实现的所有模型的通用方法（如下载或保存、调整输入嵌入、修剪头部等）。

    这个模型也是一个 [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) 子类。将它用作一个常规的 TF 2.0 Keras 模型，并参考 TF 2.0 文档以获取与一般用法和行为相关的所有内容。

    <Tip>

    `transformers` 中的 TensorFlow 模型和层接受两种格式的输入：

    - 将所有输入作为关键字参数（与 PyTorch 模型类似），或者
    - 将所有输入作为列表、元组或字典传递给第一个位置参数。

    支持第二种格式的原因是，当将输入传递给模型和层时，Keras 方法更喜欢这种格式。因为有了这种支持，当使用 `model.fit()` 等方法时，事情应该 "正常工作" - 只需以 `model.fit()` 支持的任何格式传递输入和标签即可！
    不过，如果要在 Keras 方法之外使用第二种格式，例如在使用 Keras `Functional` API 创建自己的层或模型时，有三种可能性可用于在第一个位置参数中收集所有输入张量：

    - 只有一个具有 `pixel_values` 的张量，没有其他内容：`model(pixel_values)`
    - 长度不同的列表，其中包含一个或多个输入张量，按文档字符串中给出的顺序：`model([pixel_values, attention_mask])` 或 `model([pixel_values, attention_mask, token_type_ids])`
    - 一个字典，其中包含一个或多个与文档字符串中给出的输入名称相关联的输入张量：`model({"pixel_values": pixel_values, "token_type_ids": token_type_ids})`

    请注意，当使用 [子类化](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) 创建模型和层时，您不需要担心这些，因为您可以像对待任何其他 Python 函数一样传递输入！

    </Tip>

    参数:
        config ([`ConvNextV2Config`]): 包含模型所有参数的模型配置类。
            使用配置文件初始化不会加载与模型相关的权重，只会加载配置。查看 [`~TFPreTrainedModel.from_pretrained`] 方法以加载模型权重。
"""

CONVNEXTV2_INPUTS_DOCSTRING = r"""
    # 函数参数说明：
    # pixel_values：像素值，可以是 numpy 数组、TensorFlow 张量、TensorFlow 张量列表、字符串到 TensorFlow 张量字典或字符串到 numpy 数组字典，每个示例的形状应为 (batch_size, num_channels, height, width)
    # 这些像素值可以通过 AutoImageProcessor 获取，详见 ConvNextImageProcessor.__call__ 函数的说明

    # output_hidden_states（可选）：
    # 是否返回所有层的隐藏状态。请参阅返回张量中的 hidden_states 获取更多详细信息。该参数仅在急切模式下可用，在图模式下将使用配置中的值

    # return_dict（可选）：
    # 是否返回 ModelOutput 而不是普通元组。此参数可在急切模式下使用，在图模式下该值将始终设置为 True
"""
# 给 ConvNextV2 模型添加文档字符串，描述模型输出原始特征而没有具体的顶部头部
@add_start_docstrings(
    "The bare ConvNextV2 model outputting raw features without any specific head on top.",
    CONVNEXTV2_START_DOCSTRING,
)
# 定义 TFConvNextV2Model 类，继承自 TFConvNextV2PreTrainedModel 类
class TFConvNextV2Model(TFConvNextV2PreTrainedModel):
    # 初始化方法
    def __init__(self, config: ConvNextV2Config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 创建 TFConvNextV2MainLayer 实例，用于构建 ConvNextV2 主层
        self.convnextv2 = TFConvNextV2MainLayer(config, name="convnextv2")

    # 定义 call 方法，用于模型前向传播
    @unpack_inputs
    @add_start_docstrings_to_model_forward(CONVNEXTV2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def call(
        self,
        pixel_values: TFModelInputType | None = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPoolingAndNoAttention, Tuple[tf.Tensor]]:
        # 检查是否需要返回字典类型的结果
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 检查输入像素值是否为空
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 使用 ConvNextV2 主层进行前向传播
        outputs = self.convnextv2(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 如果不需要返回字典类型的结果，则直接返回输出结果
        if not return_dict:
            return outputs[:]

        # 否则，构建 TFBaseModelOutputWithPoolingAndNoAttention 实例，并返回字典类型的结果
        return TFBaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=outputs.last_hidden_state,
            pooler_output=outputs.pooler_output,
            hidden_states=outputs.hidden_states,
        )

    # 定义 build 方法，用于构建模型
    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回
        if self.built:
            return
        # 标记模型已构建
        self.built = True
        # 如果 ConvNextV2 主层已经存在，则构建主层
        if getattr(self, "convnextv2", None) is not None:
            with tf.name_scope(self.convnextv2.name):
                self.convnextv2.build(None)


# 给 ConvNextV2 模型添加文档字符串，描述模型具有一个顶部的图像分类头部（在池化特征之上的线性层），例如用于 ImageNet
@add_start_docstrings(
    """
    ConvNextV2 Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    CONVNEXTV2_START_DOCSTRING,
)
# 定义 TFConvNextV2ForImageClassification 类，继承自 TFConvNextV2PreTrainedModel 类和 TFSequenceClassificationLoss 类
class TFConvNextV2ForImageClassification(TFConvNextV2PreTrainedModel, TFSequenceClassificationLoss):
    # 初始化方法
    def __init__(self, config: ConvNextV2Config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 设置标签数量
        self.num_labels = config.num_labels
        # 创建 TFConvNextV2MainLayer 实例，用于构建 ConvNextV2 主层
        self.convnextv2 = TFConvNextV2MainLayer(config, name="convnextv2")

        # 分类器头部
        self.classifier = tf.keras.layers.Dense(
            units=config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            bias_initializer=tf.keras.initializers.Zeros(),
            name="classifier",
        )

    # 定义 call 方法，用于模型前向传播
    @unpack_inputs
"""
    # 添加模型前向传播的文档字符串
    @add_start_docstrings_to_model_forward(CONVNEXTV2_INPUTS_DOCSTRING)
    # 添加代码示例的文档字符串
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=TFImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    # 模型的调用方法，接受输入像素值、是否输出隐藏状态、是否返回字典形式的输出、标签、训练标志
    def call(
        self,
        pixel_values: TFModelInputType | None = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[TFImageClassifierOutputWithNoAttention, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 如果未指定output_hidden_states，则使用self.config.output_hidden_states
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定return_dict，则使用self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果像素值为空，则引发值错误异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 调用convnextv2模型，并传递相应参数
        outputs = self.convnextv2(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 如果return_dict为False，则使用outputs[1]作为池化输出；否则使用outputs.pooler_output
        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        # 使用分类器对池化输出进行分类，得到logits
        logits = self.classifier(pooled_output)
        # 如果标签不为空，则计算损失；否则损失为None
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        # 如果return_dict为False，则组装输出结果；否则返回字典形式的输出结果
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )

    # 构建模型
    def build(self, input_shape=None):
        # 如果已构建，则直接返回
        if self.built:
            return
        self.built = True
        # 如果convnextv2模型存在，则构建convnextv2模型
        if getattr(self, "convnextv2", None) is not None:
            with tf.name_scope(self.convnextv2.name):
                self.convnextv2.build(None)
        # 如果classifier模型存在，则构建classifier模型
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_sizes[-1]])
```