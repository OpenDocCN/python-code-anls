# `.\models\deit\modeling_tf_deit.py`

```py
# 这是一个 TensorFlow 实现的 DeiT 模型
# 它是由 Facebook AI Research (FAIR) 和 Hugging Face 团队开发的
# 这个文件包含了 DeiT 模型的定义和相关的实用函数

# 导入所需的 Python 库和模块
import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPooling,
    TFImageClassifierOutput,
    TFMaskedImageModelingOutput,
)
from ...modeling_tf_utils import (
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    get_initializer,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_deit import DeiTConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义文档中使用的常量
_CONFIG_FOR_DOC = "DeiTConfig"
_CHECKPOINT_FOR_DOC = "facebook/deit-base-distilled-patch16-224"
_EXPECTED_OUTPUT_SHAPE = [1, 198, 768]
_IMAGE_CLASS_CHECKPOINT = "facebook/deit-base-distilled-patch16-224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

# 定义预训练模型的列表
TF_DEIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/deit-base-distilled-patch16-224",
    # See all DeiT models at https://huggingface.co/models?filter=deit
]

# 定义一个输出类型
@dataclass
class TFDeiTForImageClassificationWithTeacherOutput(ModelOutput):
    """
    Output type of [`DeiTForImageClassificationWithTeacher`].
    """
    # 代码省略
    # 定义函数参数，包括预测分数、分类头部和蒸馏头部的预测分数，隐藏状态和注意力权重
    Args:
        logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`):
            预测分数，作为分类头部和蒸馏头部预测分数的平均值。
        cls_logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`):
            分类头部的预测分数（即在类标记的最终隐藏状态之上的线性层）。
        distillation_logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`):
            蒸馏头部的预测分数（即在蒸馏标记的最终隐藏状态之上的线性层）。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            包含 tf.Tensor（嵌入输出和每一层输出的一个）的元组，形状为`(batch_size, sequence_length, hidden_size)`。模型在每一层输出的隐藏状态加上初始嵌入输出。
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            包含 tf.Tensor（每一层一个）的元组，形状为`(batch_size, num_heads, sequence_length, sequence_length)`。用于计算自注意力头中的加权平均值的注意力权重。
    
    logits: tf.Tensor = None
    cls_logits: tf.Tensor = None
    distillation_logits: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None
class TFDeiTEmbeddings(tf.keras.layers.Layer):
    """
    Construct the CLS token, distillation token, position and patch embeddings. Optionally, also the mask token.
    """
    # 定义 TFDeiTEmbeddings 类，用于构建 CLS token、distillation token、position 和 patch embeddings。可选择构建 mask token。

    def __init__(self, config: DeiTConfig, use_mask_token: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        # 初始化方法
        self.config = config
        self.use_mask_token = use_mask_token
        self.patch_embeddings = TFDeiTPatchEmbeddings(config=config, name="patch_embeddings")
        # 初始化 patch_embeddings
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob, name="dropout")
        # 初始化 dropout 层

    def build(self, input_shape=None):
        # 构建方法
        self.cls_token = self.add_weight(
            shape=(1, 1, self.config.hidden_size),
            initializer=tf.keras.initializers.zeros(),
            trainable=True,
            name="cls_token",
        )
        # 添加 CLS token 的权重
        self.distillation_token = self.add_weight(
            shape=(1, 1, self.config.hidden_size),
            initializer=tf.keras.initializers.zeros(),
            trainable=True,
            name="distillation_token",
        )
        # 添加 distillation token 的权重
        self.mask_token = None
        # 初始化 mask token 为 None
        if self.use_mask_token:
            self.mask_token = self.add_weight(
                shape=(1, 1, self.config.hidden_size),
                initializer=tf.keras.initializers.zeros(),
                trainable=True,
                name="mask_token",
            )
        # 如果使用 mask token，则添加 mask token 的权重
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = self.add_weight(
            shape=(1, num_patches + 2, self.config.hidden_size),
            initializer=tf.keras.initializers.zeros(),
            trainable=True,
            name="position_embeddings",
        )
        # 添加 position embeddings 的权重

        if self.built:
            return
        # 如果已经构建过，则直接返回
        self.built = True
        # 标记为已构建
        if getattr(self, "patch_embeddings", None) is not None:
            with tf.name_scope(self.patch_embeddings.name):
                self.patch_embeddings.build(None)
        # 构建 patch_embeddings
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)
        # 构建 dropout
    
    def call(
        self, pixel_values: tf.Tensor, bool_masked_pos: tf.Tensor | None = None, training: bool = False
        # 调用方法，接受像素值、bool_masked_pos 和训练标志参数
    # 定义一个方法，输入为当前类实例，像素数值，返回值为张量
    def forward(self, pixel_values: tf.Tensor) -> tf.Tensor:
        # 使用patch_embeddings方法获取嵌入向量
        embeddings = self.patch_embeddings(pixel_values)
        # 获取嵌入向量的维度信息
        batch_size, seq_length, _ = shape_list(embeddings)

        # 如果存在掩码位置，执行以下代码
        if bool_masked_pos is not None:
            # 使用self.mask_token创建与嵌入向量相同形状的掩码标记
            mask_tokens = tf.tile(self.mask_token, [batch_size, seq_length, 1])
            # 用mask_tokens替换掩码的视觉标记
            mask = tf.expand_dims(bool_masked_pos, axis=-1)
            mask = tf.cast(mask, dtype=mask_tokens.dtype)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # 使用self.cls_token重复batch_size次，沿着axis=0轴叠加
        cls_tokens = tf.repeat(self.cls_token, repeats=batch_size, axis=0)
        # 使用self.distillation_token重复batch_size次，沿着axis=0轴叠加
        distillation_tokens = tf.repeat(self.distillation_token, repeats=batch_size, axis=0)
        # 沿着axis=1轴将cls_tokens、distillation_tokens、embeddings连接
        embeddings = tf.concat((cls_tokens, distillation_tokens, embeddings), axis=1)
        # 将位置嵌入加到嵌入向量中
        embeddings = embeddings + self.position_embeddings
        # 使用dropout方法对嵌入向量进行处理，training参数为训练标志
        embeddings = self.dropout(embeddings, training=training)
        # 返回嵌入向量
        return embeddings
class TFDeiTPatchEmbeddings(tf.keras.layers.Layer):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config: DeiTConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        # 从配置中提取图像大小、补丁大小、通道数和隐藏层大小
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        # 将图像大小和补丁大小转换为元组形式
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        # 计算图像中的补丁数量
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        # 创建投影层，将图像中的每个补丁映射到隐藏状态空间
        self.projection = tf.keras.layers.Conv2D(
            hidden_size, kernel_size=patch_size, strides=patch_size, name="projection"
        )

    def call(self, pixel_values: tf.Tensor) -> tf.Tensor:
        # 获取像素值张量的形状信息
        batch_size, height, width, num_channels = shape_list(pixel_values)
        # 在动态执行模式下，验证像素值张量的通道数是否与配置中设置的一致
        if tf.executing_eagerly() and num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 在动态执行模式下，验证输入图像的大小是否与配置中设置的一致
        if tf.executing_eagerly() and (height != self.image_size[0] or width != self.image_size[1]):
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        # 将像素值通过投影层映射为隐藏状态
        x = self.projection(pixel_values)
        # 获取映射后张量的形状信息
        batch_size, height, width, num_channels = shape_list(x)
        # 将张量形状调整为(batch_size, seq_length, hidden_size)
        x = tf.reshape(x, (batch_size, height * width, num_channels))
        return x

    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True
        # 如果投影层已经存在，则构建投影层
        if getattr(self, "projection", None) is not None:
            with tf.name_scope(self.projection.name):
                self.projection.build([None, None, None, self.num_channels])


# Copied from transformers.models.vit.modeling_tf_vit.TFViTSelfAttention with ViT->DeiT
class TFDeiTSelfAttention(tf.keras.layers.Layer):
    # 初始化函数，用于初始化一个 DeiT 层
    def __init__(self, config: DeiTConfig, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 检查隐藏大小是否是注意力头数的倍数
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        # 设置注意力头数和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        # 初始化查询、键、值以及 Dropout 层
        self.query = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        self.key = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        self.value = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )
        self.dropout = tf.keras.layers.Dropout(rate=config.attention_probs_dropout_prob)
        self.config = config

    # 重新排列张量的维度以适应注意力得分计算的需求
    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # 从 [batch_size, seq_length, all_head_size] 重塑为 [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

        # 将张量从 [batch_size, seq_length, num_attention_heads, attention_head_size] 转置为 [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    # 实现 DeiT 层的调用逻辑
    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        # 获取隐藏状态张量的批量大小
        batch_size = shape_list(hidden_states)[0]
        # 使用 self.query() 函数处理隐藏状态张量，生成混合查询层
        mixed_query_layer = self.query(inputs=hidden_states)
        # 使用 self.key() 函数处理隐藏状态张量，生成混合键层
        mixed_key_layer = self.key(inputs=hidden_states)
        # 使用 self.value() 函数处理隐藏状态张量，生成混合值层
        mixed_value_layer = self.value(inputs=hidden_states)
        # 将混合查询层转置以进行得分计算，得到查询层
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        # 将混合键层转置以进行得分计算，得到键层
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        # 将混合值层转置以进行得分计算，得到值层
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        # 对 "query" 和 "key" 进行点积运算以获得原始注意力分数
        # (batch size, num_heads, seq_len_q, seq_len_k)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        # 计算缩放因子 dk，并将注意力分数除以 dk
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, dk)

        # 将注意力分数归一化为概率
        attention_probs = stable_softmax(logits=attention_scores, axis=-1)

        # 实际上这里是随机地将整个标记丢弃以进行关注，这可能有些不寻常，但取自原始Transformer论文。
        attention_probs = self.dropout(inputs=attention_probs, training=training)

        # 如果需要，对头部进行掩码
        if head_mask is not None:
            attention_probs = tf.multiply(attention_probs, head_mask)

        # 计算注意力输出
        attention_output = tf.matmul(attention_probs, value_layer)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, all_head_size)
        attention_output = tf.reshape(tensor=attention_output, shape=(batch_size, -1, self.all_head_size))
        # 如果需要输出注意力，将注意力概率包含在输出中
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)

        return outputs

    def build(self, input_shape=None):
        # 如果已经构建过网络，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在查询函数，则构建查询函数
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.hidden_size])
        # 如果存在键函数，则构建键函数
        if getattr(self, "key", None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.config.hidden_size])
        # 如果存在值函数，则构建值函数
        if getattr(self, "value", None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.config.hidden_size])
# 从transformers.models.vit.modeling_tf_vit.TFViTSelfOutput复制并将ViT->DeiT
# 创建TFDeiTSelfOutput类，继承自tf.keras.layers.Layer类
class TFDeiTSelfOutput(tf.keras.layers.Layer):
    """
    The residual connection is defined in TFDeiTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """
    # 初始化函数
    def __init__(self, config: DeiTConfig, **kwargs):
        super().__init__(**kwargs)
        # 创建一个全连接层，使用DeiT模型配置中的隐藏层大小和初始化值。层名为“dense”
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个丢弃层，使用DeiT模型配置中的丢弃概率
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        # 保存配置信息
        self.config = config

    # 前向传播函数
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 全连接层前向传播
        hidden_states = self.dense(inputs=hidden_states)
        # 丢弃层前向传播
        hidden_states = self.dropout(inputs=hidden_states, training=training)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# 从transformers.models.vit.modeling_tf_vit.TFViTAttention复制并将ViT->DeiT
# 创建TFDeiTAttention类，继承自tf.keras.layers.Layer类
class TFDeiTAttention(tf.keras.layers.Layer):
    # 初始化函数
    def __init__(self, config: DeiTConfig, **kwargs):
        super().__init__(**kwargs)
        # 创建TFDeiTSelfAttention对象
        self.self_attention = TFDeiTSelfAttention(config, name="attention")
        # 创建TFDeiTSelfOutput对象
        self.dense_output = TFDeiTSelfOutput(config, name="output")

    def prune_heads(self, heads):
        raise NotImplementedError

    # 前向传播函数
    def call(
        self,
        input_tensor: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        # 调用self_attention对象的前向传播
        self_outputs = self.self_attention(
            hidden_states=input_tensor, head_mask=head_mask, output_attentions=output_attentions, training=training
        )
        # 通过dense_output对象得到attention_output
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=input_tensor, training=training
        )
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "self_attention", None) is not None:
            with tf.name_scope(self.self_attention.name):
                self.self_attention.build(None)
        if getattr(self, "dense_output", None) is not None:
            with tf.name_scope(self.dense_output.name):
                self.dense_output.build(None)


# 从transformers.models.vit.modeling_tf_vit.TFViTIntermediate复制并将ViT->DeiT
# 创建TFDeiTIntermediate类，继承自tf.keras.layers.Layer类
class TFDeiTIntermediate(tf.keras.layers.Layer):
    # 初始化函数，接受一个 DeiTConfig 配置对象和可变关键字参数
    def __init__(self, config: DeiTConfig, **kwargs):
        # 调用父类初始化方法
        super().__init__(**kwargs)

        # 创建一个 Dense 层，其单元数等于配置中的 intermediate_size，初始化器为配置中的 initializer_range
        self.dense = tf.keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # 如果 hidden_act 是字符串类型，那么使用对应的 TensorFlow 激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:  # 如果 hidden_act 不是字符串，直接使用配置中提供的激活函数
            self.intermediate_act_fn = config.hidden_act
        # 将配置对象保存为类的属性
        self.config = config

    # call 方法定义模型在数据上的前向传播行为
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 将输入通过 Dense 层处理
        hidden_states = self.dense(inputs=hidden_states)
        # 应用激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)

        # 返回处理后的数据
        return hidden_states

    # build 方法用于确保层被构建
    def build(self, input_shape=None):
        # 如果层已构建，则不再重复构建
        if self.built:
            return
        # 设置构建标志为 True
        self.built = True
        # 如果 dense 层已定义，构建 dense 层
        if getattr(self, "dense", None) is not None:
            # 使用 dense 层的名称创建一个命名空间
            with tf.name_scope(self.dense.name):
                # 构建 dense 层，输入形状的最后一维为配置中的 hidden_size
                self.dense.build([None, None, self.config.hidden_size])
# 从transformers.models.vit.modeling_tf_vit.TFViTOutput复制并修改为ViT->DeiT
class TFDeiTOutput(tf.keras.layers.Layer):
    def __init__(self, config: DeiTConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，用于将隐藏状态映射到DeiT的隐藏尺寸
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个dropout层，用于在训练时随机置零隐藏状态中的一些元素
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 将隐藏状态通过全连接层进行映射
        hidden_states = self.dense(inputs=hidden_states)
        # 在训练时对隐藏状态进行dropout
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 将dropout后的隐藏状态与输入张量相加
        hidden_states = hidden_states + input_tensor

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # 构建全连接层，输入形状为[None, None, config.intermediate_size]
                self.dense.build([None, None, self.config.intermediate_size])


class TFDeiTLayer(tf.keras.layers.Layer):
    """对应于timm实现中的Block类"""

    def __init__(self, config: DeiTConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建DeiT自注意力层
        self.attention = TFDeiTAttention(config, name="attention")
        # 创建DeiT中间层
        self.intermediate = TFDeiTIntermediate(config, name="intermediate")
        # 创建DeiT输出层
        self.deit_output = TFDeiTOutput(config, name="output")

        # 创建层标准化层，用于在输入和输出之间进行标准化
        self.layernorm_before = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="layernorm_before"
        )
        self.layernorm_after = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="layernorm_after"
        )
        self.config = config

    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    # 定义模型的call方法，接受输入hidden_states并返回attention机制的输出
    ) -> Tuple[tf.Tensor]:
        # 在self-attention之前应用layernorm
        attention_outputs = self.attention(
            input_tensor=self.layernorm_before(inputs=hidden_states, training=training),  # 在训练时应用layernorm
            head_mask=head_mask,  # 头部掩码
            output_attentions=output_attentions,  # 是否输出attention权重
            training=training,  # 是否处于训练模式
        )
        attention_output = attention_outputs[0]  # 获取attention输出

        # 第一个残差连接，将attention输出与输入相加
        hidden_states = attention_output + hidden_states

        # 在self-attention之后再次应用layernorm
        layer_output = self.layernorm_after(inputs=hidden_states, training=training)  # 在训练时应用layernorm

        # 对层输出进行中间处理
        intermediate_output = self.intermediate(hidden_states=layer_output, training=training)

        # 在这里执行第二个残差连接
        layer_output = self.deit_output(
            hidden_states=intermediate_output, input_tensor=hidden_states, training=training  # 在训练时使用输入张量
        )
        outputs = (layer_output,) + attention_outputs[1:]  # 如果有需要，添加attention权重

        return outputs  # 返回模型输出

    # 构建模型
    def build(self, input_shape=None):
        if self.built:  # 如果已经构建，则直接返回
            return
        self.built = True  # 标记为已构建
        if getattr(self, "attention", None) is not None:  # 如果存在attention模块
            with tf.name_scope(self.attention.name):  # 使用attention模块的名称范围
                self.attention.build(None)  # 构建attention模块
        if getattr(self, "intermediate", None) is not None:  # 如果存在中间处理模块
            with tf.name_scope(self.intermediate.name):  # 使用中间处理模块的名称范围
                self.intermediate.build(None)  # 构建中间处理模块
        if getattr(self, "deit_output", None) is not None:  # 如果存在输出模块
            with tf.name_scope(self.deit_output.name):  # 使用输出模块的名称范围
                self.deit_output.build(None)  # 构建输出模块
        if getattr(self, "layernorm_before", None) is not None:  # 如果存在layernorm模块（在self-attention之前）
            with tf.name_scope(self.layernorm_before.name):  # 使用layernorm模块的名称范围
                self.layernorm_before.build([None, None, self.config.hidden_size])  # 构建layernorm模块
        if getattr(self, "layernorm_after", None) is not None:  # 如果存在layernorm模块（在self-attention之后）
            with tf.name_scope(self.layernorm_after.name):  # 使用layernorm模块的名称范围
                self.layernorm_after.build([None, None, self.config.hidden_size])  # 构建layernorm模块
# 从transformers.models.vit.modeling_tf_vit.TFViTEncoder中复制而来，并将ViT改为DeiT
class TFDeiTEncoder(tf.keras.layers.Layer):
    def __init__(self, config: DeiTConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建DeiTLayer的列表，用于组成编码器的层
        self.layer = [TFDeiTLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

    # 调用编码器，对输入进行编码处理
    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        # 初始化存储隐藏状态和注意力权重的空元组
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # 遍历编码器的每一层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态加入到all_hidden_states中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 调用DeiTLayer，进行编码处理
            layer_outputs = layer_module(
                hidden_states=hidden_states,
                head_mask=head_mask[i],
                output_attentions=output_attentions,
                training=training,
            )
            hidden_states = layer_outputs[0]  # 更新隐藏状态为当前层的输出

            # 如果需要输出注意力权重，则将当前层的注意力权重加入到all_attentions中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # 添加最后一层的隐藏状态到all_hidden_states中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要返回字典，则返回元组形式的结果
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        # 返回字典形式的结果
        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )

    # 构建编码器
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "layer", None) is not None:
            # 遍历每一层并构建
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)


# 定义DeiT的主层
@keras_serializable
class TFDeiTMainLayer(tf.keras.layers.Layer):
    config_class = DeiTConfig

    def __init__(
        self, config: DeiTConfig, add_pooling_layer: bool = True, use_mask_token: bool = False, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.config = config

        # 初始化DeiT的嵌入层和编码器
        self.embeddings = TFDeiTEmbeddings(config, use_mask_token=use_mask_token, name="embeddings")
        self.encoder = TFDeiTEncoder(config, name="encoder")

        # 初始化层归一化层和池化层（可选）
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")
        self.pooler = TFDeiTPooler(config, name="pooler") if add_pooling_layer else None

    # 获取输入嵌入层
    def get_input_embeddings(self) -> TFDeiTPatchEmbeddings:
        return self.embeddings.patch_embeddings

    # 剪枝模型中的头部
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError
    # 获取头部掩码，如果已经存在头部掩码则抛出异常
    def get_head_mask(self, head_mask):
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.config.num_hidden_layers

        return head_mask

    # 模型调用函数，接收输入参数并返回模型输出
    @unpack_inputs
    def call(
        self,
        pixel_values: tf.Tensor | None = None,
        bool_masked_pos: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor, ...]]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 若像素值为None，则抛出数值错误异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 调整像素值张量的维度顺序，将通道维度置于最后一个维度
        pixel_values = tf.transpose(pixel_values, (0, 2, 3, 1))

        # 如果需要，则准备头部掩码
        # head_mask中的1.0表示保留该头部注意力
        # attention_probs形状为bsz x n_heads x N x N
        # 输入的head_mask形状为[num_heads]或[num_hidden_layers x num_heads]
        # 将head_mask转换为形状[num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask)

        # 将像素值输入到嵌入层中得到嵌入输出
        embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos, training=training)

        # 将嵌入输出传入编码器中，得到编码器输出
        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        
        # 获取序列输出
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output, training=training)
        
        # 如果存在池化器，则对序列输出进行池化操作得到池化输出
        pooled_output = self.pooler(sequence_output, training=training) if self.pooler is not None else None

        # 如果不返回字典，则将头部输出和编码器的额外输出合并返回
        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        # 返回带有池化结果的模型输出
        return TFBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
```  
    # 构建模型的方法，用于构建模型的层和组件
    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回，不再重复构建
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果存在嵌入层，则构建嵌入层
        if getattr(self, "embeddings", None) is not None:
            # 在命名空间下构建嵌入层
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        # 如果存在编码器，则构建编码器
        if getattr(self, "encoder", None) is not None:
            # 在命名空间下构建编码器
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果存在层归一化，则构建层归一化
        if getattr(self, "layernorm", None) is not None:
            # 在命名空间下构建层归一化
            with tf.name_scope(self.layernorm.name):
                # 构建层归一化，指定输入形状为 [None, None, self.config.hidden_size]
                self.layernorm.build([None, None, self.config.hidden_size])
        # 如果存在池化器，则构建池化器
        if getattr(self, "pooler", None) is not None:
            # 在命名空间下构建池化器
            with tf.name_scope(self.pooler.name):
                self.pooler.build(None)
# 从 transformers.models.vit.modeling_tf_vit.TFViTPreTrainedModel 复制代码到这里
class TFDeiTPreTrainedModel(TFPreTrainedModel):
    """
    处理权重初始化的抽象类，并提供一个简单接口来下载和加载预训练模型。
    """

    # 默认的配置类
    config_class = DeiTConfig
    # 模型的前缀
    base_model_prefix = "deit"
    # 主要输入名称
    main_input_name = "pixel_values"


# DeiT 模型的文档字符串
DEIT_START_DOCSTRING = r"""
    该模型是 TensorFlow 的 [tf.keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer)。
    可以像普通的 TensorFlow 模块一样使用，并参考 TensorFlow 文档了解有关常规用法和行为的所有内容。

    参数：
        config ([`DeiTConfig`]): 包含模型所有参数的模型配置类。
            使用配置文件进行初始化不会加载与模型相关的权重，只会加载配置。请查看 [`~PreTrainedModel.from_pretrained`] 方法加载模型权重。
"""

# DeiT 模型的输入文档字符串
DEIT_INPUTS_DOCSTRING = r"""
    参数：
        pixel_values (`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):
            像素值。可以使用 [`AutoImageProcessor`] 获取像素值。详细信息请参阅 [`DeiTImageProcessor.__call__`]。

        head_mask (`tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            用于屏蔽自注意力模块中所选头部的遮罩。掩码值取值范围 [0, 1]：

            - 1 表示头部**未屏蔽**，
            - 0 表示头部**已屏蔽**。

        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。有关详细信息，请参见返回张量中的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关详细信息，请参见返回张量中的 `hidden_states`。
        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是普通的元组。
"""


@add_start_docstrings(
    "不带特定头部的裸 DeiT 模型变换器，输出原始隐藏状态。",
    DEIT_START_DOCSTRING,
)
class TFDeiTModel(TFDeiTPreTrainedModel):
    def __init__(
        self, config: DeiTConfig, add_pooling_layer: bool = True, use_mask_token: bool = False, **kwargs
    ) -> None:
        super().__init__(config, **kwargs)

        self.deit = TFDeiTMainLayer(
            config, add_pooling_layer=add_pooling_layer, use_mask_token=use_mask_token, name="deit"
        )

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DEIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    # 定义一个调用方法，接受多个参数，并返回模型输出
    def call(
        self,
        pixel_values: tf.Tensor | None = None,  # 像素值张量
        bool_masked_pos: tf.Tensor | None = None,  # 布尔掩码位置张量
        head_mask: tf.Tensor | None = None,  # 头部掩码张量
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典类型
        training: bool = False,  # 是否为训练模式
    ) -> Union[Tuple, TFBaseModelOutputWithPooling]:  # 返回值类型注释
        # 调用deit模型进行推断，得到模型输出
        outputs = self.deit(
            pixel_values=pixel_values,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )  # 模型推断
        return outputs  # 返回模型输出

    # 构建该类，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:  # 如果已经构建
            return  # 直接返回
        self.built = True  # 设置构建标志为True
        if getattr(self, "deit", None) is not None:  # 如果存在deit模型
            with tf.name_scope(self.deit.name):  # 使用deit模型的名称创建命名空间
                self.deit.build(None)  # 构建deit模型
# 从transformers.models.vit.modeling_tf_vit.TFViTPooler中复制代码，并将ViT改为DeiT
class TFDeiTPooler(tf.keras.layers.Layer):
    def __init__(self, config: DeiTConfig, **kwargs):
        super().__init__(**kwargs)

        # 定义一个全连接层，用于池化模型，将隐藏状态投影到config.hidden_size维度，激活函数为tanh
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 通过简单地取第一个标记对应的隐藏状态来“池化”模型
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(inputs=first_token_tensor)

        return pooled_output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # 构建全连接层
                self.dense.build([None, None, self.config.hidden_size])


class TFDeitPixelShuffle(tf.keras.layers.Layer):
    """TF layer implementation of torch.nn.PixelShuffle"""

    def __init__(self, upscale_factor: int, **kwargs) -> None:
        super().__init__(**kwargs)
        if not isinstance(upscale_factor, int) or upscale_factor < 2:
            raise ValueError(f"upscale_factor must be an integer value >= 2 got {upscale_factor}")
        self.upscale_factor = upscale_factor

    def call(self, x: tf.Tensor) -> tf.Tensor:
        hidden_states = x
        batch_size, _, _, num_input_channels = shape_list(hidden_states)
        block_size_squared = self.upscale_factor**2
        output_depth = int(num_input_channels / block_size_squared)
        # 当输出通道数>=2时，PyTorch的PixelShuffle和TF的depth_to_space在输出上有所不同，因为用于组合的通道顺序是另一个的排列
        # https://stackoverflow.com/questions/68272502/tf-depth-to-space-not-same-as-torchs-pixelshuffle-when-output-channels-1
        permutation = tf.constant(
            [[i + j * block_size_squared for i in range(block_size_squared) for j in range(output_depth)]]
        )
        # 使用tf.gather函数按照指定的索引顺序重新排列隐藏状态
        hidden_states = tf.gather(params=hidden_states, indices=tf.tile(permutation, [batch_size, 1]), batch_dims=-1)
        # 使用tf.nn.depth_to_space函数将深度数据转换为空间数据
        hidden_states = tf.nn.depth_to_space(hidden_states, block_size=self.upscale_factor, data_format="NHWC")
        return hidden_states


class TFDeitDecoder(tf.keras.layers.Layer):
    def __init__(self, config: DeiTConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        # 定义一个卷积层，用于解码器，输出通道数为config.encoder_stride**2 * config.num_channels，核大小为1
        self.conv2d = tf.keras.layers.Conv2D(
            filters=config.encoder_stride**2 * config.num_channels, kernel_size=1, name="0"
        )
        # 创建一个TFDeitPixelShuffle层，用于将特征图上采样
        self.pixel_shuffle = TFDeitPixelShuffle(config.encoder_stride, name="1")
        self.config = config
```py  
    # 定义一个函数，接受输入张量和一个训练标志，并返回一个张量
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 初始化隐藏层状态为输入张量
        hidden_states = inputs
        # 对隐藏层状态进行二维卷积操作
        hidden_states = self.conv2d(hidden_states)
        # 对隐藏层状态进行像素重组操作
        hidden_states = self.pixel_shuffle(hidden_states)
        # 返回处理后的隐藏层状态
        return hidden_states

    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建好，直接返回
        if self.built:
            return
        # 将模型标记为已构建
        self.built = True
        # 如果模型中存在conv2d属性
        if getattr(self, "conv2d", None) is not None:
            # 在命名域中构建conv2d模块
            with tf.name_scope(self.conv2d.name):
                # 构建conv2d模块
                self.conv2d.build([None, None, None, self.config.hidden_size])
        # 如果模型中存在pixel_shuffle属性
        if getattr(self, "pixel_shuffle", None) is not None:
            # 在命名域中构建pixel_shuffle模块
            with tf.name_scope(self.pixel_shuffle.name):
                # 构建pixel_shuffle模块
                self.pixel_shuffle.build(None)
@add_start_docstrings(
    "DeiT Model with a decoder on top for masked image modeling, as proposed in"
    " [SimMIM](https://arxiv.org/abs/2111.09886).",
    DEIT_START_DOCSTRING,
)
class TFDeiTForMaskedImageModeling(TFDeiTPreTrainedModel):
    def __init__(self, config: DeiTConfig) -> None:
        super().__init__(config)

        # Initialize the DeiT model with a decoder on top for masked image modeling
        self.deit = TFDeiTMainLayer(config, add_pooling_layer=False, use_mask_token=True, name="deit")
        # Initialize the decoder for the model
        self.decoder = TFDeitDecoder(config, name="decoder")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DEIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFMaskedImageModelingOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        pixel_values: tf.Tensor | None = None,
        bool_masked_pos: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # Build the DeiT model if it exists
        if getattr(self, "deit", None) is not None:
            with tf.name_scope(self.deit.name):
                self.deit.build(None)
        # Build the decoder if it exists
        if getattr(self, "decoder", None) is not None:
            with tf.name_scope(self.decoder.name):
                self.decoder.build(None)


@add_start_docstrings(
    """
    DeiT Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.
    """,
    DEIT_START_DOCSTRING,
)
class TFDeiTForImageClassification(TFDeiTPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: DeiTConfig):
        super().__init__(config)

        # Number of labels for classification
        self.num_labels = config.num_labels
        # Initialize the DeiT model transformer
        self.deit = TFDeiTMainLayer(config, add_pooling_layer=False, name="deit")

        # Classifier head
        self.classifier = (
            tf.keras.layers.Dense(config.num_labels, name="classifier")
            if config.num_labels > 0
            else tf.keras.layers.Activation("linear", name="classifier")
        )
        # Store the configuration
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DEIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFImageClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        pixel_values: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        labels: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[tf.Tensor, TFImageClassifierOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, TFDeiTForImageClassification
        >>> import tensorflow as tf
        >>> from PIL import Image
        >>> import requests

        >>> tf.keras.utils.set_random_seed(3)  # doctest: +IGNORE_RESULT
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> # note: we are loading a TFDeiTForImageClassificationWithTeacher from the hub here,
        >>> # so the head will be randomly initialized, hence the predictions will be random
        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
        >>> model = TFDeiTForImageClassification.from_pretrained("facebook/deit-base-distilled-patch16-224")

        >>> inputs = image_processor(images=image, return_tensors="tf")
        >>> outputs = model(**inputs)
        >>> logits = outputs.logits
        >>> # model predicts one of the 1000 ImageNet classes
        >>> predicted_class_idx = tf.math.argmax(logits, axis=-1)[0]
        >>> print("Predicted class:", model.config.id2label[int(predicted_class_idx)])
        Predicted class: little blue heron, Egretta caerulea
        ```py"""
        // 检查是否需要返回字典
          return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        // 运行DEIT模型，获取输出
        outputs = self.deit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        // 获取序列输出
        sequence_output = outputs[0]

        // 通过分类器获取logits
        logits = self.classifier(sequence_output[:, 0, :])
        // 我们不使用蒸馏令牌

        // 计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        // 如果不需要返回字典，则返回输出元组
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        // 返回TFImageClassifierOutput对象
        return TFImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回
        if self.built:
            return
        # 设置模型已经构建的标志为 True
        self.built = True
        # 如果存在 self.deit 属性，则设置其名称作为命名空间，并构建该层
        if getattr(self, "deit", None) is not None:
            with tf.name_scope(self.deit.name):
                self.deit.build(None)
        # 如果存在 self.classifier 属性，则设置其名称作为命名空间，并根据输入形状构建该层
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])

-   检查模型是否已经构建，如果是，则直接返回，不再执行后续操作。
-   将模型已经构建的标志设置为 True，表示模型已经构建完成。
-   检查存在属性 self.deit，如果存在，则为其设置命名空间为 self.deit.name，并构建该层。
-   检查存在属性 self.classifier，如果存在，则为其设置命名空间为 self.classifier.name，并根据输入形状构建该层。
# 使用装饰器为类添加文档字符串，描述DeiT模型的作用以及警告信息
@add_start_docstrings(
    """
    DeiT Model transformer with image classification heads on top (a linear layer on top of the final hidden state of
    the [CLS] token and a linear layer on top of the final hidden state of the distillation token) e.g. for ImageNet.

    .. warning::

            This model supports inference-only. Fine-tuning with distillation (i.e. with a teacher) is not yet
            supported.
    """,
    DEIT_START_DOCSTRING,
)
# 定义类TFDeiTForImageClassificationWithTeacher，继承自TFDeiTPreTrainedModel
class TFDeiTForImageClassificationWithTeacher(TFDeiTPreTrainedModel):
    # 定义初始化方法，接受一个DeiTConfig类型的config参数
    def __init__(self, config: DeiTConfig) -> None:
        # 调用父类的初始化方法
        super().__init__(config)

        # 获取config中的标签数量，并保存为成员变量num_labels
        self.num_labels = config.num_labels
        # 创建TFDeiTMainLayer对象，并保存为成员变量deit
        self.deit = TFDeiTMainLayer(config, add_pooling_layer=False, name="deit")

        # 分类器头部
        # 创建一个Dense层，输出维度为config.num_labels，并保存为成员变量cls_classifier；
        # 如果config.num_labels小于等于0，则创建一个线性激活函数层，作为cls_classifier
        self.cls_classifier = (
            tf.keras.layers.Dense(config.num_labels, name="cls_classifier")
            if config.num_labels > 0
            else tf.keras.layers.Activation("linear", name="cls_classifier")
        )
        # 创建一个Dense层，输出维度为config.num_labels，并保存为成员变量distillation_classifier；
        # 如果config.num_labels小于等于0，则创建一个线性激活函数层，作为distillation_classifier
        self.distillation_classifier = (
            tf.keras.layers.Dense(config.num_labels, name="distillation_classifier")
            if config.num_labels > 0
            else tf.keras.layers.Activation("linear", name="distillation_classifier")
        )
        # 保存config为成员变量
        self.config = config

    # 使用装饰器为call方法添加文档字符串，描述输入、输出等信息
    @unpack_inputs
    @add_start_docstrings_to_model_forward(DEIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=TFDeiTForImageClassificationWithTeacherOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    # 定义call方法，接受多个参数
    def call(
        self,
        pixel_values: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    # 定义一个方法，用于获取模型输出
    ) -> Union[tuple, TFDeiTForImageClassificationWithTeacherOutput]:
        # 如果 return_dict 为空，则设为配置中的 use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 DEiT 模型进行推理
        outputs = self.deit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 获取序列输出
        sequence_output = outputs[0]

        # 获取分类器的输出
        cls_logits = self.cls_classifier(sequence_output[:, 0, :])
        distillation_logits = self.distillation_classifier(sequence_output[:, 1, :])

        # 在推理过程中，返回两个分类器预测结果的平均值
        logits = (cls_logits + distillation_logits) / 2

        # 如果不需要返回字典格式的输出
        if not return_dict:
            # 构建输出结果
            output = (logits, cls_logits, distillation_logits) + outputs[1:]
            return output

        # 返回 TFDeiTForImageClassificationWithTeacherOutput 类型的输出
        return TFDeiTForImageClassificationWithTeacherOutput(
            logits=logits,
            cls_logits=cls_logits,
            distillation_logits=distillation_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 构建方法
    def build(self, input_shape=None):
        # 如果已经构建完成，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 self.deit，则构建 DEiT 模型
        if getattr(self, "deit", None) is not None:
            with tf.name_scope(self.deit.name):
                self.deit.build(None)
        # 如果存在 self.cls_classifier，则构建分类器
        if getattr(self, "cls_classifier", None) is not None:
            with tf.name_scope(self.cls_classifier.name):
                self.cls_classifier.build([None, None, self.config.hidden_size])
        # 如果存在 self.distillation_classifier，则构建蒸馏分类器
        if getattr(self, "distillation_classifier", None) is not None:
            with tf.name_scope(self.distillation_classifier.name):
                self.distillation_classifier.build([None, None, self.config.hidden_size])
```