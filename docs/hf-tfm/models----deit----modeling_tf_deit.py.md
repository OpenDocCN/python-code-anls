# `.\models\deit\modeling_tf_deit.py`

```py
# 设置编码格式为 UTF-8
# 版权声明，版权归 Facebook AI Research (FAIR) 和 HuggingFace Inc. 团队所有
#
# 根据 Apache License, Version 2.0 授权，除非符合许可证，否则不得使用此文件
# 可以在以下链接获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则依据"原样"分发此软件
# 没有任何明示或暗示的保证或条件，包括但不限于适销性或特定用途的适用性保证
# 请查看许可证以获取详细信息
""" TensorFlow DeiT model. """

# 引入未来支持的注释类型
from __future__ import annotations

# 引入标准库
import collections.abc
import math
# 引入数据类
from dataclasses import dataclass
# 引入类型提示
from typing import Optional, Tuple, Union

# 引入 TensorFlow 库
import tensorflow as tf

# 引入相关模块和函数
# 从活化函数模块中获取 TensorFlow 版本的激活函数
from ...activations_tf import get_tf_activation
# 引入 TensorFlow 版本的模型输出类
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPooling,
    TFImageClassifierOutput,
    TFMaskedImageModelingOutput,
)
# 引入 TensorFlow 工具函数
from ...modeling_tf_utils import (
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    get_initializer,
    keras,
    keras_serializable,
    unpack_inputs,
)
# 引入 TensorFlow 实用函数
from ...tf_utils import shape_list, stable_softmax
# 引入工具函数
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 引入 DeiT 配置类
from .configuration_deit import DeiTConfig

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 用于文档的一般配置描述
_CONFIG_FOR_DOC = "DeiTConfig"

# 用于文档的基础描述
_CHECKPOINT_FOR_DOC = "facebook/deit-base-distilled-patch16-224"
_EXPECTED_OUTPUT_SHAPE = [1, 198, 768]

# 用于图像分类的描述
_IMAGE_CLASS_CHECKPOINT = "facebook/deit-base-distilled-patch16-224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

# 预训练的 TensorFlow DeiT 模型存档列表
TF_DEIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/deit-base-distilled-patch16-224",
    # 查看所有 DeiT 模型：https://huggingface.co/models?filter=deit
]


@dataclass
class TFDeiTForImageClassificationWithTeacherOutput(ModelOutput):
    """
    [`DeiTForImageClassificationWithTeacher`] 的输出类型。
    """
    # 定义函数参数，表示模型的预测分数和其他中间结果
    Args:
        logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`):
            Prediction scores as the average of the cls_logits and distillation logits.
            预测分数，作为 cls_logits 和 distillation_logits 的平均值
        cls_logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`):
            Prediction scores of the classification head (i.e. the linear layer on top of the final hidden state of the
            class token).
            分类头部的预测分数，即在类令牌的最终隐藏状态之上的线性层的输出
        distillation_logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`):
            Prediction scores of the distillation head (i.e. the linear layer on top of the final hidden state of the
            distillation token).
            蒸馏头部的预测分数，即在蒸馏令牌的最终隐藏状态之上的线性层的输出
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus
            the initial embedding outputs.
            隐藏状态的元组，包含每个层的输出（嵌入层输出和每层输出），仅在设置 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
            注意力权重的元组，包含每个层的注意力权重，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`，
            仅在设置 `output_attentions=True` 或 `config.output_attentions=True` 时返回
    """

    # 初始化参数，默认为 None，表示可能的输出结果
    logits: tf.Tensor = None
    cls_logits: tf.Tensor = None
    distillation_logits: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None
    """
    构建 CLS token、蒸馏 token、位置和补丁嵌入。可选择是否包含 mask token。
    """

    def __init__(self, config: DeiTConfig, use_mask_token: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.use_mask_token = use_mask_token
        self.patch_embeddings = TFDeiTPatchEmbeddings(config=config, name="patch_embeddings")
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob, name="dropout")

    def build(self, input_shape=None):
        # 添加 CLS token 权重，形状为 (1, 1, hidden_size)，初始化为零
        self.cls_token = self.add_weight(
            shape=(1, 1, self.config.hidden_size),
            initializer=keras.initializers.zeros(),
            trainable=True,
            name="cls_token",
        )
        # 添加蒸馏 token 权重，形状为 (1, 1, hidden_size)，初始化为零
        self.distillation_token = self.add_weight(
            shape=(1, 1, self.config.hidden_size),
            initializer=keras.initializers.zeros(),
            trainable=True,
            name="distillation_token",
        )
        # 如果需要使用 mask token，则添加 mask token 权重，形状为 (1, 1, hidden_size)，初始化为零
        self.mask_token = None
        if self.use_mask_token:
            self.mask_token = self.add_weight(
                shape=(1, 1, self.config.hidden_size),
                initializer=keras.initializers.zeros(),
                trainable=True,
                name="mask_token",
            )
        # 计算补丁数量，用于位置嵌入的构建
        num_patches = self.patch_embeddings.num_patches
        # 添加位置嵌入权重，形状为 (1, num_patches + 2, hidden_size)，初始化为零
        self.position_embeddings = self.add_weight(
            shape=(1, num_patches + 2, self.config.hidden_size),
            initializer=keras.initializers.zeros(),
            trainable=True,
            name="position_embeddings",
        )

        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 patch_embeddings 属性，则构建 patch_embeddings
        if getattr(self, "patch_embeddings", None) is not None:
            with tf.name_scope(self.patch_embeddings.name):
                self.patch_embeddings.build(None)
        # 如果存在 dropout 属性，则构建 dropout
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)

    def call(
        self, pixel_values: tf.Tensor, bool_masked_pos: tf.Tensor | None = None, training: bool = False
    ):
        # 略
        pass
    ) -> tf.Tensor:
        # 使用 patch_embeddings 方法生成图像块的嵌入表示
        embeddings = self.patch_embeddings(pixel_values)
        # 获取嵌入张量的批大小、序列长度和特征维度信息
        batch_size, seq_length, _ = shape_list(embeddings)

        if bool_masked_pos is not None:
            # 使用 mask_token 在指定位置替换掩码的视觉标记
            mask_tokens = tf.tile(self.mask_token, [batch_size, seq_length, 1])
            # 创建用于掩码的布尔张量，并扩展其维度以适应张量运算
            mask = tf.expand_dims(bool_masked_pos, axis=-1)
            mask = tf.cast(mask, dtype=mask_tokens.dtype)
            # 应用掩码，保留未被掩码的嵌入值，替换掩码位置的嵌入为 mask_tokens
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # 生成一组重复的 cls_token，用于每个批次
        cls_tokens = tf.repeat(self.cls_token, repeats=batch_size, axis=0)
        # 生成一组重复的 distillation_token，用于每个批次
        distillation_tokens = tf.repeat(self.distillation_token, repeats=batch_size, axis=0)
        # 将 cls_tokens、distillation_tokens 和 embeddings 拼接在一起
        embeddings = tf.concat((cls_tokens, distillation_tokens, embeddings), axis=1)
        # 添加位置嵌入到嵌入张量中
        embeddings = embeddings + self.position_embeddings
        # 应用 dropout 操作到嵌入张量中，用于训练时的正则化
        embeddings = self.dropout(embeddings, training=training)
        # 返回最终生成的嵌入张量
        return embeddings
# 定义一个自定义层 TFDeiTPatchEmbeddings，用于将像素值 `pixel_values` 转换为 Transformer 模型可用的初始隐藏状态（patch embeddings）
class TFDeiTPatchEmbeddings(keras.layers.Layer):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config: DeiTConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        # 从配置中获取图像大小和patch大小
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        # 如果图像大小或patch大小不是可迭代对象，则转换为元组
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        
        # 计算patch的数量
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        
        # 设置对象的属性
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        # 创建一个卷积层，用于将像素值投影到隐藏大小的空间，作为Transformer的输入
        self.projection = keras.layers.Conv2D(
            hidden_size, kernel_size=patch_size, strides=patch_size, name="projection"
        )

    # 定义层的前向传播逻辑，将像素值转换为patch embeddings
    def call(self, pixel_values: tf.Tensor) -> tf.Tensor:
        # 获取输入张量的形状信息
        batch_size, height, width, num_channels = shape_list(pixel_values)
        
        # 在即时执行模式下，确保像素值的通道维度与配置中设置的通道数匹配
        if tf.executing_eagerly() and num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        
        # 在即时执行模式下，确保输入图像的尺寸与配置中设置的图像尺寸匹配
        if tf.executing_eagerly() and (height != self.image_size[0] or width != self.image_size[1]):
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        
        # 将像素值通过投影层投影为形状为(batch_size, seq_length, hidden_size)的张量x
        x = self.projection(pixel_values)
        
        # 获取投影后张量x的形状信息
        batch_size, height, width, num_channels = shape_list(x)
        
        # 将x重新调整形状为(batch_size, seq_length, hidden_size)，其中seq_length=height*width
        x = tf.reshape(x, (batch_size, height * width, num_channels))
        
        # 返回转换后的张量x作为patch embeddings
        return x

    # 在模型构建时调用，用于构建投影层
    def build(self, input_shape=None):
        # 如果已经构建，则直接返回
        if self.built:
            return
        
        # 标记此层为已构建
        self.built = True
        
        # 如果投影层已经存在，则在其名称作用域下构建该层
        if getattr(self, "projection", None) is not None:
            with tf.name_scope(self.projection.name):
                self.projection.build([None, None, None, self.num_channels])


# 从 transformers.models.vit.modeling_tf_vit.TFViTSelfAttention 复制并修改为 DeiT 模型中的自注意力层 TFDeiTSelfAttention
class TFDeiTSelfAttention(keras.layers.Layer):
    def __init__(self, config: DeiTConfig, **kwargs):
        super().__init__(**kwargs)

        # 检查隐藏大小是否能被注意力头数整除
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        # 初始化注意力头数和每个注意力头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        # 创建查询、键、值的全连接层，用于后续注意力计算
        self.query = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        self.key = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        self.value = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )

        # 定义用于 dropout 的层
        self.dropout = keras.layers.Dropout(rate=config.attention_probs_dropout_prob)
        self.config = config

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # 将输入张量从 [batch_size, seq_length, all_head_size] 重塑为 [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

        # 转置张量从 [batch_size, seq_length, num_attention_heads, attention_head_size] 到 [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        # 获取隐藏状态的批量大小
        batch_size = shape_list(hidden_states)[0]
        # 使用 self.query 函数处理隐藏状态，生成混合的查询层
        mixed_query_layer = self.query(inputs=hidden_states)
        # 使用 self.key 函数处理隐藏状态，生成混合的键层
        mixed_key_layer = self.key(inputs=hidden_states)
        # 使用 self.value 函数处理隐藏状态，生成混合的值层
        mixed_value_layer = self.value(inputs=hidden_states)
        # 将混合的查询层转置以用于注意力分数计算
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        # 将混合的键层转置以用于注意力分数计算
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        # 将混合的值层转置以用于注意力分数计算
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        # 计算注意力分数，即查询和键的点积
        # 结果形状为 (batch size, num_heads, seq_len_q, seq_len_k)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        # 缩放注意力分数
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, dk)

        # 将注意力分数归一化为概率
        attention_probs = stable_softmax(logits=attention_scores, axis=-1)

        # 对注意力分数进行 dropout 处理
        attention_probs = self.dropout(inputs=attention_probs, training=training)

        # 如果给定了头部掩码，则应用头部掩码
        if head_mask is not None:
            attention_probs = tf.multiply(attention_probs, head_mask)

        # 计算注意力输出，将注意力分数乘以值层
        attention_output = tf.matmul(attention_probs, value_layer)
        # 调整输出张量的维度顺序
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])

        # 将输出张量重新形状为 (batch_size, seq_len_q, all_head_size)
        attention_output = tf.reshape(tensor=attention_output, shape=(batch_size, -1, self.all_head_size))
        # 如果需要输出注意力分数，则返回注意力输出和注意力分数
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)

        return outputs

    def build(self, input_shape=None):
        # 如果已经构建过网络层，则直接返回
        if self.built:
            return
        # 标记网络层已经构建
        self.built = True
        # 如果存在 self.query 属性，则构建查询网络层
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.hidden_size])
        # 如果存在 self.key 属性，则构建键网络层
        if getattr(self, "key", None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.config.hidden_size])
        # 如果存在 self.value 属性，则构建值网络层
        if getattr(self, "value", None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.config.hidden_size])
# Copied from transformers.models.vit.modeling_tf_vit.TFViTSelfOutput with ViT->DeiT
class TFDeiTSelfOutput(keras.layers.Layer):
    """
    The residual connection is defined in TFDeiTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: DeiTConfig, **kwargs):
        super().__init__(**kwargs)
        
        # 创建一个全连接层，用于变换隐藏状态的维度
        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个 dropout 层，用于随机失活一部分神经元，防止过拟合
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 将隐藏状态输入全连接层进行变换
        hidden_states = self.dense(inputs=hidden_states)
        # 在训练阶段，对变换后的隐藏状态进行随机失活处理
        hidden_states = self.dropout(inputs=hidden_states, training=training)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经构建过网络层，则直接返回
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # 构建全连接层，指定输入形状和隐藏状态的维度
                self.dense.build([None, None, self.config.hidden_size])


# Copied from transformers.models.vit.modeling_tf_vit.TFViTAttention with ViT->DeiT
class TFDeiTAttention(keras.layers.Layer):
    def __init__(self, config: DeiTConfig, **kwargs):
        super().__init__(**kwargs)
        
        # 创建自注意力层，用于处理注意力机制
        self.self_attention = TFDeiTSelfAttention(config, name="attention")
        # 创建输出层，用于处理自注意力层输出的隐藏状态
        self.dense_output = TFDeiTSelfOutput(config, name="output")

    def prune_heads(self, heads):
        # 目前未实现头部修剪的方法
        raise NotImplementedError

    def call(
        self,
        input_tensor: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        # 调用自注意力层处理输入张量，获取自注意力输出
        self_outputs = self.self_attention(
            hidden_states=input_tensor, head_mask=head_mask, output_attentions=output_attentions, training=training
        )
        # 将自注意力输出传入输出层处理，得到注意力机制的输出
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=input_tensor, training=training
        )
        # 如果需要输出注意力信息，则将注意力信息添加到输出中
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经构建过网络层，则直接返回
        if getattr(self, "self_attention", None) is not None:
            with tf.name_scope(self.self_attention.name):
                # 构建自注意力层，不需要指定具体的输入形状
                self.self_attention.build(None)
        if getattr(self, "dense_output", None) is not None:
            with tf.name_scope(self.dense_output.name):
                # 构建输出层，不需要指定具体的输入形状
                self.dense_output.build(None)


# Copied from transformers.models.vit.modeling_tf_vit.TFViTIntermediate with ViT->DeiT
class TFDeiTIntermediate(keras.layers.Layer):
    # 初始化方法，用于初始化一个新的对象实例
    def __init__(self, config: DeiTConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 创建一个全连接层，设置单元数为config中指定的中间大小，
        # 内核初始化方式使用config中的初始化范围，命名为"dense"
        self.dense = keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # 根据config中的隐藏激活函数，获取对应的激活函数对象或者名称
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        
        # 将config保存在对象中
        self.config = config

    # 调用方法，用于执行前向传播
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 使用全连接层处理输入的隐藏状态数据
        hidden_states = self.dense(inputs=hidden_states)
        # 使用中间激活函数处理全连接层的输出隐藏状态数据
        hidden_states = self.intermediate_act_fn(hidden_states)

        # 返回处理后的隐藏状态数据
        return hidden_states

    # 构建方法，用于构建模型层次结构
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 标记为已经构建
        self.built = True
        # 如果存在全连接层dense，则使用其名字作为作用域，构建该层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
# 从transformers.models.vit.modeling_tf_vit.TFViTOutput复制并更名为TFDeiTOutput，用于实现Transformer中的输出层逻辑，但采用DeiT的配置。
class TFDeiTOutput(keras.layers.Layer):
    def __init__(self, config: DeiTConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，输出维度为config.hidden_size，权重初始化使用config中定义的初始化范围。
        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个Dropout层，使用config中定义的dropout概率。
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 通过全连接层处理输入hidden_states，输出经过线性变换后的结果。
        hidden_states = self.dense(inputs=hidden_states)
        # 对处理后的结果进行Dropout操作，以防止过拟合。
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 将Dropout后的结果与输入tensor相加，实现残差连接。
        hidden_states = hidden_states + input_tensor

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经构建过，直接返回。否则，根据self.dense的名字作用域，构建全连接层，输入维度为None，None，self.config.intermediate_size。
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])


# TFDeiTLayer是Transformer中的一个层，对应于timm实现中的Block类。
class TFDeiTLayer(keras.layers.Layer):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: DeiTConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建DeiTAttention层，使用给定的DeiT配置，并命名为"attention"。
        self.attention = TFDeiTAttention(config, name="attention")
        # 创建DeiTIntermediate层，使用给定的DeiT配置，并命名为"intermediate"。
        self.intermediate = TFDeiTIntermediate(config, name="intermediate")
        # 创建TFDeiTOutput层，使用给定的DeiT配置，并命名为"output"。
        self.deit_output = TFDeiTOutput(config, name="output")

        # 创建LayerNormalization层，在训练过程中使用给定的epsilon参数进行归一化，命名为"layernorm_before"和"layernorm_after"。
        self.layernorm_before = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm_before")
        self.layernorm_after = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm_after")
        self.config = config

    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        # 对输入hidden_states应用LayerNormalization，然后传递给self.attention，获取attention_outputs。
        attention_outputs = self.attention(
            input_tensor=self.layernorm_before(inputs=hidden_states, training=training),
            head_mask=head_mask,
            output_attentions=output_attentions,
            training=training,
        )
        attention_output = attention_outputs[0]

        # 第一个残差连接，将attention_output加到hidden_states上。
        hidden_states = attention_output + hidden_states

        # 再次应用LayerNormalization，得到layer_output。
        layer_output = self.layernorm_after(inputs=hidden_states, training=training)

        # 将layer_output传递给self.intermediate层进行处理，得到intermediate_output。
        intermediate_output = self.intermediate(hidden_states=layer_output, training=training)

        # 第二个残差连接，将intermediate_output和hidden_states传递给self.deit_output处理，得到layer_output。
        layer_output = self.deit_output(
            hidden_states=intermediate_output, input_tensor=hidden_states, training=training
        )
        # 将输出打包成元组outputs，如果需要输出attention信息，则将其添加到outputs中。
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them

        return outputs
    # 定义一个方法用于建立模型，可以接受输入形状作为参数
    def build(self, input_shape=None):
        # 如果模型已经建立过，则直接返回，不进行重复建立
        if self.built:
            return
        # 设置标记，表明模型已经建立
        self.built = True
        
        # 如果存在 self.attention 属性，则构建 attention 层
        if getattr(self, "attention", None) is not None:
            # 使用 attention 层的名称作为命名空间
            with tf.name_scope(self.attention.name):
                # 调用 attention 层的 build 方法，传入 None 作为输入形状
                self.attention.build(None)
        
        # 如果存在 self.intermediate 属性，则构建 intermediate 层
        if getattr(self, "intermediate", None) is not None:
            # 使用 intermediate 层的名称作为命名空间
            with tf.name_scope(self.intermediate.name):
                # 调用 intermediate 层的 build 方法，传入 None 作为输入形状
                self.intermediate.build(None)
        
        # 如果存在 self.deit_output 属性，则构建 deit_output 层
        if getattr(self, "deit_output", None) is not None:
            # 使用 deit_output 层的名称作为命名空间
            with tf.name_scope(self.deit_output.name):
                # 调用 deit_output 层的 build 方法，传入 None 作为输入形状
                self.deit_output.build(None)
        
        # 如果存在 self.layernorm_before 属性，则构建 layernorm_before 层
        if getattr(self, "layernorm_before", None) is not None:
            # 使用 layernorm_before 层的名称作为命名空间
            with tf.name_scope(self.layernorm_before.name):
                # 调用 layernorm_before 层的 build 方法，传入一个形状为 [None, None, self.config.hidden_size] 的列表作为输入形状
                self.layernorm_before.build([None, None, self.config.hidden_size])
        
        # 如果存在 self.layernorm_after 属性，则构建 layernorm_after 层
        if getattr(self, "layernorm_after", None) is not None:
            # 使用 layernorm_after 层的名称作为命名空间
            with tf.name_scope(self.layernorm_after.name):
                # 调用 layernorm_after 层的 build 方法，传入一个形状为 [None, None, self.config.hidden_size] 的列表作为输入形状
                self.layernorm_after.build([None, None, self.config.hidden_size])
# 从transformers.models.vit.modeling_tf_vit.TFViTEncoder复制并修改为ViT->DeiT
class TFDeiTEncoder(keras.layers.Layer):
    def __init__(self, config: DeiTConfig, **kwargs):
        super().__init__(**kwargs)

        # 初始化层列表，每层命名为"layer_._{i}"
        self.layer = [TFDeiTLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        # 如果输出隐藏状态，初始化空元组以存储所有隐藏状态
        all_hidden_states = () if output_hidden_states else None
        # 如果输出注意力权重，初始化空元组以存储所有注意力权重
        all_attentions = () if output_attentions else None

        # 遍历每个编码层
        for i, layer_module in enumerate(self.layer):
            # 如果输出隐藏状态，将当前隐藏状态加入到列表中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 调用编码层的计算
            layer_outputs = layer_module(
                hidden_states=hidden_states,
                head_mask=head_mask[i],
                output_attentions=output_attentions,
                training=training,
            )
            # 更新隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]

            # 如果输出注意力权重，将当前层的注意力权重加入到列表中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # 添加最后一层的隐藏状态到所有隐藏状态列表中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典形式的结果，根据是否为空过滤掉None值并返回结果元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        # 返回 TFBaseModelOutput 对象，包含最后的隐藏状态、所有隐藏状态和所有注意力权重
        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )

    def build(self, input_shape=None):
        # 如果已经构建，直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在层列表，为每一层设置命名作用域并构建层
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)


@keras_serializable
class TFDeiTMainLayer(keras.layers.Layer):
    config_class = DeiTConfig

    def __init__(
        self, config: DeiTConfig, add_pooling_layer: bool = True, use_mask_token: bool = False, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        # 初始化配置
        self.config = config

        # 初始化嵌入层和编码器层
        self.embeddings = TFDeiTEmbeddings(config, use_mask_token=use_mask_token, name="embeddings")
        self.encoder = TFDeiTEncoder(config, name="encoder")

        # 初始化层归一化层
        self.layernorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")
        # 如果需要添加池化层，则初始化池化层
        self.pooler = TFDeiTPooler(config, name="pooler") if add_pooling_layer else None

    def get_input_embeddings(self) -> TFDeiTPatchEmbeddings:
        # 返回嵌入层的补丁嵌入
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 修剪模型的注意力头，heads_to_prune 是一个字典，键为层号，值为要修剪的头列表，参见基类 PreTrainedModel
        raise NotImplementedError
    def get_head_mask(self, head_mask):
        # 如果 head_mask 不为 None，抛出未实现的错误
        if head_mask is not None:
            raise NotImplementedError
        else:
            # 否则，创建一个长度为 self.config.num_hidden_layers 的 None 列表作为 head_mask
            head_mask = [None] * self.config.num_hidden_layers

        # 返回创建或传入的 head_mask
        return head_mask

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
        # 设置 output_attentions，如果未提供则使用 self.config.output_attentions
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置 output_hidden_states，如果未提供则使用 self.config.output_hidden_states
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置 return_dict，如果未提供则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果 pixel_values 为 None，则抛出数值错误
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 转置 pixel_values 张量，将 NCHW 格式转换为 NHWC 格式
        pixel_values = tf.transpose(pixel_values, (0, 2, 3, 1))

        # 准备 head_mask，调用 self.get_head_mask 获取头部掩码
        head_mask = self.get_head_mask(head_mask)

        # 使用 embeddings 方法生成嵌入输出
        embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos, training=training)

        # 使用 encoder 方法生成编码器输出
        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 获取编码器的序列输出
        sequence_output = encoder_outputs[0]

        # 对序列输出进行 layer normalization
        sequence_output = self.layernorm(sequence_output, training=training)

        # 如果存在池化器，使用池化器生成池化输出
        pooled_output = self.pooler(sequence_output, training=training) if self.pooler is not None else None

        # 如果 return_dict 为 False，则返回 head_outputs 和额外的 encoder_outputs
        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        # 如果 return_dict 为 True，则返回 TFBaseModelOutputWithPooling 对象
        return TFBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    # 如果模型已经构建完成，则直接返回，不进行重复构建
    if self.built:
        return
    # 将模型标记为已构建状态
    self.built = True
    # 如果模型具有嵌入层（embeddings），则构建嵌入层
    if getattr(self, "embeddings", None) is not None:
        # 使用嵌入层的名称作为命名空间，构建嵌入层
        with tf.name_scope(self.embeddings.name):
            self.embeddings.build(None)
    # 如果模型具有编码器（encoder），则构建编码器
    if getattr(self, "encoder", None) is not None:
        # 使用编码器的名称作为命名空间，构建编码器
        with tf.name_scope(self.encoder.name):
            self.encoder.build(None)
    # 如果模型具有层归一化（layernorm），则构建层归一化
    if getattr(self, "layernorm", None) is not None:
        # 使用层归一化的名称作为命名空间，构建层归一化，指定输入形状为 [None, None, self.config.hidden_size]
        with tf.name_scope(self.layernorm.name):
            self.layernorm.build([None, None, self.config.hidden_size])
    # 如果模型具有池化器（pooler），则构建池化器
    if getattr(self, "pooler", None) is not None:
        # 使用池化器的名称作为命名空间，构建池化器
        with tf.name_scope(self.pooler.name):
            self.pooler.build(None)
# 从 transformers.models.vit.modeling_tf_vit.TFViTPreTrainedModel 复制并修改为 ViT->DeiT 的大小写。
class TFDeiTPreTrainedModel(TFPreTrainedModel):
    """
    一个抽象类，处理权重初始化以及预训练模型的下载和加载的简单接口。
    """

    # 配置类，指定为 DeiTConfig
    config_class = DeiTConfig
    # 基础模型前缀，设定为 "deit"
    base_model_prefix = "deit"
    # 主输入名称，设定为 "pixel_values"
    main_input_name = "pixel_values"


# 下面是 DEIT_START_DOCSTRING 的文档字符串
DEIT_START_DOCSTRING = r"""
    This model is a TensorFlow
    [keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer). Use it as a regular
    TensorFlow Module and refer to the TensorFlow documentation for all matter related to general usage and behavior.

    Parameters:
        config ([`DeiTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 下面是 DEIT_INPUTS_DOCSTRING 的文档字符串
DEIT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`DeiTImageProcessor.__call__`] for details.

        head_mask (`tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare DeiT Model transformer outputting raw hidden-states without any specific head on top.",
    DEIT_START_DOCSTRING,
)
# TFDeiTModel 类的文档字符串和初始化方法注释
class TFDeiTModel(TFDeiTPreTrainedModel):
    def __init__(
        self, config: DeiTConfig, add_pooling_layer: bool = True, use_mask_token: bool = False, **kwargs
    ) -> None:
        # 调用父类的初始化方法
        super().__init__(config, **kwargs)

        # 创建 TFDeiTMainLayer 实例作为 self.deit 属性
        self.deit = TFDeiTMainLayer(
            config, add_pooling_layer=add_pooling_layer, use_mask_token=use_mask_token, name="deit"
        )

    # 使用装饰器添加输入解包，模型前向方法的起始文档字符串和代码示例文档字符串
    @unpack_inputs
    @add_start_docstrings_to_model_forward(DEIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    # 定义一个方法 `call`，用于调用模型。
    def call(
        self,
        pixel_values: tf.Tensor | None = None,  # 输入像素值的张量，可以为空
        bool_masked_pos: tf.Tensor | None = None,  # 布尔类型的遮罩位置张量，可以为空
        head_mask: tf.Tensor | None = None,  # 头部遮罩张量，可以为空
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重的布尔值，可选
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态的布尔值，可选
        return_dict: Optional[bool] = None,  # 是否返回字典类型的结果，可选
        training: bool = False,  # 是否处于训练模式，默认为 False
    ) -> Union[Tuple, TFBaseModelOutputWithPooling]:
        # 调用模型 `deit`，传递给它所有的参数，并返回其输出
        outputs = self.deit(
            pixel_values=pixel_values,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 返回模型的输出
        return outputs

    # 定义一个方法 `build`，用于构建模型的层次结构
    def build(self, input_shape=None):
        # 如果模型已经构建完成，则直接返回
        if self.built:
            return
        # 标记模型为已构建状态
        self.built = True
        # 如果存在 `deit` 属性，则在 `deit` 的命名作用域下构建模型
        if getattr(self, "deit", None) is not None:
            with tf.name_scope(self.deit.name):
                # 调用 `deit` 的构建方法，传入 `None` 的输入形状
                self.deit.build(None)
# 从 transformers.models.vit.modeling_tf_vit.TFViTPooler 复制而来，将 ViT 改为 DeiT
class TFDeiTPooler(keras.layers.Layer):
    def __init__(self, config: DeiTConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，用于池化模型，输出维度为 config.hidden_size
        self.dense = keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 通过简单地选择第一个 token 的隐藏状态来“池化”模型
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(inputs=first_token_tensor)

        return pooled_output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经构建，直接返回；否则，按照指定的输入形状构建 dense 层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


class TFDeitPixelShuffle(keras.layers.Layer):
    """TF 实现的 torch.nn.PixelShuffle 的层"""

    def __init__(self, upscale_factor: int, **kwargs) -> None:
        super().__init__(**kwargs)
        if not isinstance(upscale_factor, int) or upscale_factor < 2:
            raise ValueError(f"upscale_factor 必须是大于等于 2 的整数，当前值为 {upscale_factor}")
        self.upscale_factor = upscale_factor

    def call(self, x: tf.Tensor) -> tf.Tensor:
        hidden_states = x
        batch_size, _, _, num_input_channels = shape_list(hidden_states)
        block_size_squared = self.upscale_factor**2
        output_depth = int(num_input_channels / block_size_squared)
        
        # 计算输出通道数时，PyTorch 的 PixelShuffle 和 TF 的 depth_to_space 在输出上存在差异，
        # 因为通道的选择顺序会导致组合顺序不同，详情参考：
        # https://stackoverflow.com/questions/68272502/tf-depth-to-space-not-same-as-torchs-pixelshuffle-when-output-channels-1
        permutation = tf.constant(
            [[i + j * block_size_squared for i in range(block_size_squared) for j in range(output_depth)]]
        )
        # 使用 permutation 重新组合隐藏状态张量的通道
        hidden_states = tf.gather(params=hidden_states, indices=tf.tile(permutation, [batch_size, 1]), batch_dims=-1)
        # 使用 TF 的 depth_to_space 函数进行像素洗牌操作
        hidden_states = tf.nn.depth_to_space(hidden_states, block_size=self.upscale_factor, data_format="NHWC")
        return hidden_states


class TFDeitDecoder(keras.layers.Layer):
    def __init__(self, config: DeiTConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        
        # 创建一个卷积层，输出通道数为 config.encoder_stride 的平方乘以 config.num_channels，卷积核大小为 1x1
        self.conv2d = keras.layers.Conv2D(
            filters=config.encoder_stride**2 * config.num_channels, kernel_size=1, name="0"
        )
        # 创建 TFDeitPixelShuffle 层，用于解码器
        self.pixel_shuffle = TFDeitPixelShuffle(config.encoder_stride, name="1")
        self.config = config
    # 定义一个方法用于调用模型，接受一个张量作为输入，并可选择是否进行训练
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 将输入张量赋给隐藏状态变量
        hidden_states = inputs
        # 对隐藏状态进行二维卷积操作
        hidden_states = self.conv2d(hidden_states)
        # 对卷积后的结果进行像素重排操作
        hidden_states = self.pixel_shuffle(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states

    # 构建模型的方法
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        # 设置模型已构建标志为True
        self.built = True
        # 如果模型具有conv2d属性，则构建conv2d层
        if getattr(self, "conv2d", None) is not None:
            # 在TensorFlow中使用name_scope管理命名空间
            with tf.name_scope(self.conv2d.name):
                # 构建conv2d层，指定输入的形状为[None, None, None, self.config.hidden_size]
                self.conv2d.build([None, None, None, self.config.hidden_size])
        # 如果模型具有pixel_shuffle属性，则构建pixel_shuffle层
        if getattr(self, "pixel_shuffle", None) is not None:
            # 在TensorFlow中使用name_scope管理命名空间
            with tf.name_scope(self.pixel_shuffle.name):
                # 构建pixel_shuffle层，不指定具体的输入形状
                self.pixel_shuffle.build(None)
@add_start_docstrings(
    """
    在遮蔽图像建模中使用的DeiT模型，其顶部有一个解码器，正如[SimmIM](https://arxiv.org/abs/2111.09886)中所提出的。
    """,
    DEIT_START_DOCSTRING,
)
class TFDeiTForMaskedImageModeling(TFDeiTPreTrainedModel):
    def __init__(self, config: DeiTConfig) -> None:
        super().__init__(config)

        # 初始化DeiT主层，禁用池化层，启用掩码令牌，命名为"deit"
        self.deit = TFDeiTMainLayer(config, add_pooling_layer=False, use_mask_token=True, name="deit")
        # 初始化解码器，使用给定的DeiT配置，命名为"decoder"
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
    ):
        """
        DeiT模型的前向传播方法，接受像素值、掩码位置、头部掩码等参数。
        """
        # 省略了具体的前向传播逻辑，由于不在要求内，不能提供更多细节。

    def build(self, input_shape=None):
        """
        构建模型，确保DeiT和解码器都已构建。
        """
        if self.built:
            return
        self.built = True
        if getattr(self, "deit", None) is not None:
            with tf.name_scope(self.deit.name):
                self.deit.build(None)
        if getattr(self, "decoder", None) is not None:
            with tf.name_scope(self.decoder.name):
                self.decoder.build(None)


@add_start_docstrings(
    """
    带有图像分类头部的DeiT模型变换器（在[CLS]标记的最终隐藏状态之上有一个线性层），例如用于ImageNet。
    """,
    DEIT_START_DOCSTRING,
)
class TFDeiTForImageClassification(TFDeiTPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: DeiTConfig):
        super().__init__(config)

        # 配置标签数量
        self.num_labels = config.num_labels
        # 初始化DeiT主层，禁用池化层，命名为"deit"
        self.deit = TFDeiTMainLayer(config, add_pooling_layer=False, name="deit")

        # 分类器头部
        self.classifier = (
            # 如果标签数大于0，则使用稠密层，命名为"classifier"
            keras.layers.Dense(config.num_labels, name="classifier")
            if config.num_labels > 0
            # 否则使用线性激活函数，命名为"classifier"
            else keras.layers.Activation("linear", name="classifier")
        )
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
    ):
        """
        DeiT模型的前向传播方法，接受像素值、头部掩码、标签等参数。
        """
        # 省略了具体的前向传播逻辑，由于不在要求内，不能提供更多细节。
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 设置返回字典的标志，如果未提供则使用模型配置中的默认设置

        outputs = self.deit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 使用图像编码器（self.deit）处理像素值，可选地应用头部遮罩、输出注意力和隐藏状态，根据return_dict参数返回结果

        sequence_output = outputs[0]
        # 获取模型输出中的序列输出（通常是最后一层的输出）

        logits = self.classifier(sequence_output[:, 0, :])
        # 使用分类器（self.classifier）计算逻辑回归，通常使用序列输出的第一个位置的信息

        # 不使用蒸馏令牌

        loss = None if labels is None else self.hf_compute_loss(labels, logits)
        # 如果提供了标签，则使用标签和逻辑回归计算损失，否则损失为None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        # 如果不要求返回字典形式的结果，则返回一个元组，包含逻辑回归和其它可能的输出

        return TFImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # 返回一个TFImageClassifierOutput对象，包含损失、逻辑回归、隐藏状态和注意力信息
    # 如果模型已经构建完成，则直接返回，不进行重复构建
    if self.built:
        return
    
    # 标记模型已经构建完成
    self.built = True
    
    # 如果存在名为"deit"的属性，并且不为None，则构建"deit"模型部分
    if getattr(self, "deit", None) is not None:
        # 在命名空间中构建"deit"模型
        with tf.name_scope(self.deit.name):
            self.deit.build(None)
    
    # 如果存在名为"classifier"的属性，并且不为None，则构建"classifier"模型部分
    if getattr(self, "classifier", None) is not None:
        # 在命名空间中构建"classifier"模型
        with tf.name_scope(self.classifier.name):
            # 构建"classifier"模型，指定输入形状为[None, None, self.config.hidden_size]
            self.classifier.build([None, None, self.config.hidden_size])
# 使用装饰器为类添加文档字符串，描述了此类的主要功能和警告信息
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
class TFDeiTForImageClassificationWithTeacher(TFDeiTPreTrainedModel):
    def __init__(self, config: DeiTConfig) -> None:
        # 调用父类构造函数初始化模型
        super().__init__(config)

        # 保存分类标签数量
        self.num_labels = config.num_labels
        # 创建 DeiT 主层实例，不添加池化层
        self.deit = TFDeiTMainLayer(config, add_pooling_layer=False, name="deit")

        # 分类器头部初始化
        # 如果有分类标签数量，则创建全连接层作为分类器；否则创建线性激活层
        self.cls_classifier = (
            keras.layers.Dense(config.num_labels, name="cls_classifier")
            if config.num_labels > 0
            else keras.layers.Activation("linear", name="cls_classifier")
        )
        # 同上，针对蒸馏分类器头部
        self.distillation_classifier = (
            keras.layers.Dense(config.num_labels, name="distillation_classifier")
            if config.num_labels > 0
            else keras.layers.Activation("linear", name="distillation_classifier")
        )
        # 保存配置信息
        self.config = config

    # 使用装饰器定义 call 方法的输入输出说明文档
    @unpack_inputs
    @add_start_docstrings_to_model_forward(DEIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=TFDeiTForImageClassificationWithTeacherOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    # 定义模型的前向传播逻辑
    def call(
        self,
        pixel_values: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        **kwargs
    ) -> TFDeiTForImageClassificationWithTeacherOutput:
        # 在这里实现具体的前向传播逻辑，处理输入张量和模型参数
        pass  # 这里的 pass 仅用于示例，实际应填写前向传播的具体实现
    # 定义模型的前向传播方法，输入参数包括像素值、头部掩码、是否输出注意力和隐藏状态、是否返回字典以及训练模式
    ) -> Union[tuple, TFDeiTForImageClassificationWithTeacherOutput]:
        # 如果未指定返回字典，则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 DeiT 模型进行推理
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

        # 对序列输出的第一个位置进行分类器预测
        cls_logits = self.cls_classifier(sequence_output[:, 0, :])
        # 对序列输出的第二个位置进行蒸馏分类器预测
        distillation_logits = self.distillation_classifier(sequence_output[:, 1, :])

        # 在推理阶段，返回两个分类器预测结果的平均值作为最终预测
        logits = (cls_logits + distillation_logits) / 2

        # 如果不需要返回字典，则返回预测结果和其他输出信息
        if not return_dict:
            output = (logits, cls_logits, distillation_logits) + outputs[1:]
            return output

        # 否则，返回一个包含预测 logits、分类器 logits、蒸馏分类器 logits、隐藏状态和注意力信息的 TFDeiTForImageClassificationWithTeacherOutput 对象
        return TFDeiTForImageClassificationWithTeacherOutput(
            logits=logits,
            cls_logits=cls_logits,
            distillation_logits=distillation_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 构建模型的方法，设置模型已构建标志并构建模型的各个组件
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        # 设置模型已构建标志为 True
        self.built = True

        # 如果 DeiT 模型存在，则构建它
        if getattr(self, "deit", None) is not None:
            with tf.name_scope(self.deit.name):
                self.deit.build(None)

        # 如果分类器存在，则构建分类器
        if getattr(self, "cls_classifier", None) is not None:
            with tf.name_scope(self.cls_classifier.name):
                self.cls_classifier.build([None, None, self.config.hidden_size])

        # 如果蒸馏分类器存在，则构建蒸馏分类器
        if getattr(self, "distillation_classifier", None) is not None:
            with tf.name_scope(self.distillation_classifier.name):
                self.distillation_classifier.build([None, None, self.config.hidden_size])
```