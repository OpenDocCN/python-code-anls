# `.\models\clip\modeling_tf_clip.py`

```
# 设置编码为 UTF-8
# 版权声明和许可证信息
# 该代码遵循 Apache License, Version 2.0
# 可以在符合许可证条件的情况下使用和分发
"""
TF 2.0 CLIP 模型。
"""

# 导入必要的模块和库
from __future__ import annotations  # 使得类型注解支持延迟引用

import math  # 导入数学模块
from dataclasses import dataclass  # 导入 dataclass 用于定义数据类
from typing import Any, Optional, Tuple, Union  # 导入类型提示相关的模块

import numpy as np  # 导入 NumPy 库
import tensorflow as tf  # 导入 TensorFlow 库

# 导入所需的自定义模块和函数
from ...activations_tf import get_tf_activation  # 导入获取 TensorFlow 激活函数的函数
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling  # 导入 TF 模型输出相关类

# 公共 API
from ...modeling_tf_utils import (
    TFModelInputType,  # 导入 TF 模型输入类型
    TFPreTrainedModel,  # 导入 TF 预训练模型基类
    get_initializer,  # 导入获取初始化器函数
    keras,  # 导入 Keras 相关功能
    keras_serializable,  # 导入 Keras 序列化函数
    unpack_inputs,  # 导入解包输入函数
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax  # 导入 TensorFlow 工具函数
from ...utils import (
    ModelOutput,  # 导入模型输出类
    add_start_docstrings,  # 导入添加文档字符串函数
    add_start_docstrings_to_model_forward,  # 导入添加前向模型文档字符串函数
    logging,  # 导入日志记录函数
    replace_return_docstrings,  # 导入替换返回文档字符串函数
)
from .configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig  # 导入 CLIP 模型配置类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器对象

_CHECKPOINT_FOR_DOC = "openai/clip-vit-base-patch32"  # 预训练模型的检查点名称

TF_CLIP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "openai/clip-vit-base-patch32",  # 支持的预训练模型列表，包括一个 CLIP 模型
    # 可以在 https://huggingface.co/models?filter=clip 查看所有 CLIP 模型
]

LARGE_NEGATIVE = -1e8  # 定义一个大负数，用于在计算中作为标记

# 从 transformers.models.bart.modeling_tf_bart._expand_mask 复制而来的函数
def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None):
    """
    将注意力掩码从 `[bsz, seq_len]` 扩展到 `[bsz, 1, tgt_seq_len, src_seq_len]`。
    """
    src_len = shape_list(mask)[1]  # 获取掩码的序列长度
    tgt_len = tgt_len if tgt_len is not None else src_len  # 如果未指定目标长度，则使用源长度
    one_cst = tf.constant(1.0)  # 创建一个 TensorFlow 常量张量，值为 1.0
    mask = tf.cast(mask, dtype=one_cst.dtype)  # 将掩码转换为指定的数据类型的张量
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))  # 使用 tf.tile 扩展掩码张量

    return (one_cst - expanded_mask) * LARGE_NEGATIVE  # 返回扩展后的掩码张量，同时使用大负数进行标记


# 对比损失函数，改编自 https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits: tf.Tensor) -> tf.Tensor:
    """
    计算对比损失，使用稀疏分类交叉熵作为损失函数。
    """
    return tf.math.reduce_mean(
        keras.metrics.sparse_categorical_crossentropy(
            y_true=tf.range(shape_list(logits)[0]), y_pred=logits, from_logits=True
        )
    )


def clip_loss(similarity: tf.Tensor) -> tf.Tensor:
    """
    计算 CLIP 损失，结合文本和图像的对比损失。
    """
    caption_loss = contrastive_loss(similarity)  # 计算文本对比损失
    image_loss = contrastive_loss(tf.transpose(similarity))  # 计算图像对比损失
    return (caption_loss + image_loss) / 2.0  # 返回文本和图像损失的平均值


@dataclass
class TFCLIPOutput(ModelOutput):
    """
    TFCLIP 模型的输出类，继承自 ModelOutput。
    """
    # 这里是数据类的定义，包含了 ModelOutput 类的所有属性和方法
    """
    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image:(`tf.Tensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text:(`tf.Tensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds(`tf.Tensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`TFCLIPTextModel`].
        image_embeds(`tf.Tensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of
            [`TFCLIPVisionModel`].
        text_model_output([`~modeling_tf_utils.TFBaseModelOutputWithPooling`]):
            The output of the [`TFCLIPTextModel`].
        vision_model_output([`~modeling_tf_utils.TFBaseModelOutputWithPooling`]):
            The output of the [`TFCLIPVisionModel`].
    """

    # 定义类属性，用于存储各种输入和输出的张量或模型输出
    loss: tf.Tensor | None = None
    logits_per_image: tf.Tensor = None
    logits_per_text: tf.Tensor = None
    text_embeds: tf.Tensor = None
    image_embeds: tf.Tensor = None
    text_model_output: TFBaseModelOutputWithPooling = None
    vision_model_output: TFBaseModelOutputWithPooling = None

    # 将对象转换为元组的方法，忽略文本和视觉模型输出，直接使用它们的 `to_tuple` 方法获取元组表示
    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            # 如果键不是 "text_model_output" 或 "vision_model_output"，则直接取值
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )
class TFCLIPVisionEmbeddings(keras.layers.Layer):
    # TFCLIPVisionEmbeddings 类，用于实现 CLIP 视觉嵌入层

    def __init__(self, config: CLIPVisionConfig, **kwargs):
        # 初始化函数，接受 CLIPVisionConfig 类型的配置参数和其他关键字参数

        super().__init__(**kwargs)

        # 设置嵌入维度
        self.embed_dim = config.hidden_size
        # 图像尺寸
        self.image_size = config.image_size
        # 图像分块大小
        self.patch_size = config.patch_size

        # 计算图像中的分块数量
        self.num_patches = (self.image_size // self.patch_size) ** 2
        # 计算位置编码数量（分块数量 + [CLS] token）
        self.num_positions = self.num_patches + 1

        # 保存配置参数
        self.config = config

        # 定义图像分块嵌入层，使用 2D 卷积实现
        self.patch_embedding = keras.layers.Conv2D(
            filters=self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="valid",
            data_format="channels_last",
            use_bias=False,
            kernel_initializer=get_initializer(self.config.initializer_range * self.config.initializer_factor),
            name="patch_embedding",
        )

    def build(self, input_shape: tf.TensorShape = None):
        # 构建函数，用于构建层的权重和其他状态

        factor = self.config.initializer_factor

        # 添加类别嵌入向量权重
        self.class_embedding = self.add_weight(
            shape=(self.embed_dim,),
            initializer=get_initializer(self.embed_dim**-0.5 * factor),
            trainable=True,
            name="class_embedding",
        )

        # 添加位置编码向量权重
        with tf.name_scope("position_embedding"):
            self.position_embedding = self.add_weight(
                shape=(self.num_positions, self.embed_dim),
                initializer=get_initializer(self.config.initializer_range * factor),
                trainable=True,
                name="embeddings",
            )

        if self.built:
            return
        self.built = True

        # 如果已构建，直接返回；否则构建图像分块嵌入层
        if getattr(self, "patch_embedding", None) is not None:
            with tf.name_scope(self.patch_embedding.name):
                self.patch_embedding.build([None, None, None, self.config.num_channels])

    def call(self, pixel_values: tf.Tensor) -> tf.Tensor:
        """`pixel_values` is expected to be of NCHW format."""
        # 调用函数，对输入的像素值进行处理并返回嵌入向量

        batch_size, num_channels, height, width = shape_list(pixel_values)

        # 当在 CPU 上运行时，`tf.nn.conv2d` 不支持 `NCHW` 格式，
        # 因此将输入格式从 `NCHW` 转换为 `NHWC`
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))

        # 使用图像分块嵌入层处理像素值
        patch_embeds = self.patch_embedding(pixel_values)

        # 将2D空间维度转换为单个时间维度
        patch_embeds = tf.reshape(tensor=patch_embeds, shape=(batch_size, self.num_patches, -1))

        # 添加 [CLS] token 到嵌入的分块 token 中
        class_embeds = tf.broadcast_to(self.class_embedding, shape=(batch_size, 1, self.embed_dim))
        embeddings = tf.concat((class_embeds, patch_embeds), axis=1)

        # 添加位置编码到嵌入向量中
        embeddings = embeddings + self.position_embedding

        return embeddings
    def __init__(self, config: CLIPTextConfig, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = config.hidden_size  # 从配置中获取嵌入维度大小

        self.config = config  # 将配置对象存储在实例中以供后续使用

    def build(self, input_shape: tf.TensorShape = None):
        with tf.name_scope("token_embedding"):
            # 创建并添加嵌入权重变量，形状为 (词汇大小, 嵌入维度)
            self.weight = self.add_weight(
                shape=(self.config.vocab_size, self.embed_dim),
                initializer=get_initializer(self.config.initializer_factor * self.config.initializer_range),
                trainable=True,
                name="weight",
            )

        with tf.name_scope("position_embedding"):
            # 创建并添加位置嵌入权重变量，形状为 (最大位置嵌入数, 嵌入维度)
            self.position_embedding = self.add_weight(
                shape=(self.config.max_position_embeddings, self.embed_dim),
                initializer=get_initializer(self.config.initializer_factor * self.config.initializer_range),
                trainable=True,
                name="embeddings",
            )

        super().build(input_shape)

    def call(
        self,
        input_ids: tf.Tensor = None,
        position_ids: tf.Tensor = None,
        inputs_embeds: tf.Tensor = None,
    ) -> tf.Tensor:
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            # 如果未提供预先计算好的嵌入向量，则根据输入的 ids 获取对应的嵌入向量
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        input_shape = shape_list(inputs_embeds)[:-1]

        if position_ids is None:
            # 如果未提供位置 ids，则生成默认位置 ids
            position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)

        # 根据位置 ids 获取位置嵌入向量，并在批次维度上进行复制以匹配输入嵌入的形状
        position_embeds = tf.gather(params=self.position_embedding, indices=position_ids)
        position_embeds = tf.tile(input=position_embeds, multiples=(input_shape[0], 1, 1))
        
        # 最终的嵌入向量是输入嵌入向量和位置嵌入向量的和
        final_embeddings = inputs_embeds + position_embeds

        return final_embeddings
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: CLIPConfig, **kwargs):
        super().__init__(**kwargs)

        # 初始化层参数
        self.embed_dim = config.hidden_size  # 获取隐藏层大小
        self.num_attention_heads = config.num_attention_heads  # 获取注意力头的数量
        self.attention_head_size = self.embed_dim // self.num_attention_heads  # 计算每个注意力头的大小
        if self.attention_head_size * self.num_attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_attention_heads})."
            )

        # 初始化投影矩阵的标准差
        factor = config.initializer_factor
        in_proj_std = (self.embed_dim**-0.5) * ((2 * config.num_hidden_layers) ** -0.5) * factor
        out_proj_std = (self.embed_dim**-0.5) * factor

        # 设置平方根注意力头大小
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        # 定义查询、键、值的投影层
        self.q_proj = keras.layers.Dense(
            units=self.embed_dim, kernel_initializer=get_initializer(in_proj_std), name="q_proj"
        )
        self.k_proj = keras.layers.Dense(
            units=self.embed_dim, kernel_initializer=get_initializer(in_proj_std), name="k_proj"
        )
        self.v_proj = keras.layers.Dense(
            units=self.embed_dim, kernel_initializer=get_initializer(in_proj_std), name="v_proj"
        )

        # 定义 dropout 层
        self.dropout = keras.layers.Dropout(rate=config.attention_dropout)

        # 定义输出投影层
        self.out_proj = keras.layers.Dense(
            units=self.embed_dim, kernel_initializer=get_initializer(out_proj_std), name="out_proj"
        )

    # 从 transformers.models.bert.modeling_tf_bert.TFBertSelfAttention.transpose_for_scores 复制而来
    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # 将张量从 [batch_size, seq_length, all_head_size] 重塑为 [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

        # 将张量从 [batch_size, seq_length, num_attention_heads, attention_head_size] 转置为 [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        causal_attention_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        """Input shape: Batch x Time x Channel"""
        
        # 获取隐藏状态张量的批量大小
        batch_size = shape_list(hidden_states)[0]
        
        # 通过线性投影计算混合的查询、键和值张量
        mixed_query_layer = self.q_proj(inputs=hidden_states)
        mixed_key_layer = self.k_proj(inputs=hidden_states)
        mixed_value_layer = self.v_proj(inputs=hidden_states)
        
        # 将混合的查询、键和值张量转置以便于计算注意力分数
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        # 计算注意力分数，使用query和key的点积
        # 结果形状为(batch size, num_heads, seq_len_q, seq_len_k)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        
        # 缩放注意力分数
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, dk)

        # 首先应用因果注意力掩码
        if causal_attention_mask is not None:
            # 使用预先计算的因果注意力掩码，添加到注意力分数中
            attention_scores = tf.add(attention_scores, causal_attention_mask)

        # 如果存在普通注意力掩码，则也应用它
        if attention_mask is not None:
            # 使用预先计算的注意力掩码，添加到注意力分数中
            attention_scores = tf.add(attention_scores, attention_mask)

        # 将注意力分数归一化为概率值
        _attention_probs = stable_softmax(logits=attention_scores, axis=-1)

        # 使用dropout对注意力概率进行处理
        attention_probs = self.dropout(inputs=_attention_probs, training=training)

        # 计算注意力输出值
        attention_output = tf.matmul(attention_probs, value_layer)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])

        # 重新整形注意力输出的张量形状
        attention_output = tf.reshape(tensor=attention_output, shape=(batch_size, -1, self.embed_dim))

        # 将注意力输出通过输出投影层
        attention_output = self.out_proj(attention_output, training=training)
        
        # 在TFBert中，注意力权重在dropout之后返回
        # 但是在CLIP中，它们在dropout之前返回
        outputs = (attention_output, _attention_probs) if output_attentions else (attention_output,)

        return outputs
    # 构建方法用于初始化层，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 将标志位设置为已构建
        self.built = True
        
        # 如果存在查询投影层，则构建该层
        if getattr(self, "q_proj", None) is not None:
            with tf.name_scope(self.q_proj.name):
                # 构建查询投影层，期望输入形状为 [None, None, self.embed_dim]
                self.q_proj.build([None, None, self.embed_dim])
        
        # 如果存在键投影层，则构建该层
        if getattr(self, "k_proj", None) is not None:
            with tf.name_scope(self.k_proj.name):
                # 构建键投影层，期望输入形状为 [None, None, self.embed_dim]
                self.k_proj.build([None, None, self.embed_dim])
        
        # 如果存在值投影层，则构建该层
        if getattr(self, "v_proj", None) is not None:
            with tf.name_scope(self.v_proj.name):
                # 构建值投影层，期望输入形状为 [None, None, self.embed_dim]
                self.v_proj.build([None, None, self.embed_dim])
        
        # 如果存在输出投影层，则构建该层
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                # 构建输出投影层，期望输入形状为 [None, None, self.embed_dim]
                self.out_proj.build([None, None, self.embed_dim])
# 定义 TFCLIPMLP 类，继承自 keras.layers.Layer
class TFCLIPMLP(keras.layers.Layer):
    # 初始化方法，接受一个 CLIPConfig 类型的配置对象和其他关键字参数
    def __init__(self, config: CLIPConfig, **kwargs):
        super().__init__(**kwargs)

        # 获取激活函数，根据配置中的隐藏层激活函数类型
        self.activation_fn = get_tf_activation(config.hidden_act)

        # 计算初始化因子
        factor = config.initializer_factor
        # 计算输入投影的标准差
        in_proj_std = (config.hidden_size**-0.5) * ((2 * config.num_hidden_layers) ** -0.5) * factor
        # 计算全连接层的标准差
        fc_std = (2 * config.hidden_size) ** -0.5 * factor

        # 创建全连接层 fc1，units 表示输出单元数，kernel_initializer 设置初始化方式，名称为 "fc1"
        self.fc1 = keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(fc_std), name="fc1"
        )
        # 创建全连接层 fc2，units 表示输出单元数，kernel_initializer 设置初始化方式，名称为 "fc2"
        self.fc2 = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(in_proj_std), name="fc2"
        )
        # 保存配置对象
        self.config = config

    # 调用方法，接受隐藏状态输入，返回全连接层的输出
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 使用全连接层 fc1 对隐藏状态进行投影
        hidden_states = self.fc1(inputs=hidden_states)
        # 使用激活函数对投影后的隐藏状态进行激活
        hidden_states = self.activation_fn(hidden_states)
        # 使用全连接层 fc2 对激活后的隐藏状态再次投影
        hidden_states = self.fc2(inputs=hidden_states)
        # 返回投影后的隐藏状态
        return hidden_states

    # 构建方法，用于构建层的结构
    def build(self, input_shape=None):
        # 如果层已经构建，则直接返回
        if self.built:
            return
        # 设置层已构建的标志为 True
        self.built = True
        # 如果存在 fc1 层，则构建 fc1 层，设置输入形状为 [None, None, self.config.hidden_size]
        if getattr(self, "fc1", None) is not None:
            with tf.name_scope(self.fc1.name):
                self.fc1.build([None, None, self.config.hidden_size])
        # 如果存在 fc2 层，则构建 fc2 层，设置输入形状为 [None, None, self.config.intermediate_size]
        if getattr(self, "fc2", None) is not None:
            with tf.name_scope(self.fc2.name):
                self.fc2.build([None, None, self.config.intermediate_size])


# 定义 TFCLIPEncoderLayer 类，继承自 keras.layers.Layer
class TFCLIPEncoderLayer(keras.layers.Layer):
    # 初始化方法，接受一个 CLIPConfig 类型的配置对象和其他关键字参数
    def __init__(self, config: CLIPConfig, **kwargs):
        super().__init__(**kwargs)

        # 设置嵌入维度为配置中的隐藏大小
        self.embed_dim = config.hidden_size
        # 创建自注意力层 self_attn
        self.self_attn = TFCLIPAttention(config, name="self_attn")
        # 创建第一个层规范化层 layer_norm1
        self.layer_norm1 = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm1")
        # 创建 MLP 层 mlp
        self.mlp = TFCLIPMLP(config, name="mlp")
        # 创建第二个层规范化层 layer_norm2
        self.layer_norm2 = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm2")

    # 调用方法，接受隐藏状态、注意力掩码、因果注意力掩码、输出注意力的标志和训练的标志，返回处理后的隐藏状态
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        causal_attention_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        """
        Args:
            hidden_states (`tf.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`tf.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            causal_attention_mask (`tf.Tensor`): causal attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`):
                Whether or not to return the attentions tensors of all attention layers. See `outputs` under returned
                tensors for more detail.
        """
        residual = hidden_states  # 存储输入的隐藏状态作为残差连接的起点

        hidden_states = self.layer_norm1(inputs=hidden_states)  # 对输入的隐藏状态进行 layer normalization
        attention_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            training=training,
        )  # 使用自注意力机制处理隐藏状态，可能返回注意力输出
        hidden_states = attention_outputs[0]  # 更新隐藏状态为自注意力机制的输出
        hidden_states = residual + hidden_states  # 残差连接：加上原始输入的隐藏状态

        residual = hidden_states  # 存储当前状态作为残差连接的起点
        hidden_states = self.layer_norm2(inputs=hidden_states)  # 对当前状态进行第二次 layer normalization
        hidden_states = self.mlp(hidden_states=hidden_states)  # 使用多层感知机处理当前状态
        hidden_states = residual + hidden_states  # 残差连接：加上前一步的输出

        outputs = (hidden_states,) + attention_outputs[1:]  # 如果需要返回注意力张量，则添加到输出中

        return outputs  # 返回处理后的输出结果，可能包含注意力张量和隐藏状态

    def build(self, input_shape=None):
        if self.built:
            return  # 如果已经构建过，直接返回

        self.built = True  # 标记模型已经构建

        if getattr(self, "self_attn", None) is not None:
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)  # 构建自注意力层

        if getattr(self, "layer_norm1", None) is not None:
            with tf.name_scope(self.layer_norm1.name):
                self.layer_norm1.build([None, None, self.embed_dim])  # 构建第一个 layer normalization 层

        if getattr(self, "mlp", None) is not None:
            with tf.name_scope(self.mlp.name):
                self.mlp.build(None)  # 构建多层感知机

        if getattr(self, "layer_norm2", None) is not None:
            with tf.name_scope(self.layer_norm2.name):
                self.layer_norm2.build([None, None, self.embed_dim])  # 构建第二个 layer normalization 层
class TFCLIPEncoder(keras.layers.Layer):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`TFCLIPEncoderLayer`].

    Args:
        config: CLIPConfig
    """

    def __init__(self, config: CLIPConfig, **kwargs):
        super().__init__(**kwargs)

        # 初始化 Transformer 编码器的多层自注意力层
        self.layers = [TFCLIPEncoderLayer(config, name=f"layers_._{i}") for i in range(config.num_hidden_layers)]

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        causal_attention_mask: tf.Tensor,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        # 初始化用于保存所有隐藏状态的元组，如果需要输出隐藏状态的话
        all_hidden_states = () if output_hidden_states else None
        # 初始化用于保存所有注意力权重的元组，如果需要输出注意力权重的话
        all_attentions = () if output_attentions else None

        # 遍历所有编码器层
        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                # 将当前隐藏状态添加到所有隐藏状态的元组中
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 调用当前编码器层进行前向传播
            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
                output_attentions=output_attentions,
                training=training,
            )
            hidden_states = layer_outputs[0]  # 更新隐藏状态为当前层的输出

            if output_attentions:
                # 将当前层的注意力权重添加到所有注意力权重的元组中
                all_attentions = all_attentions + (layer_outputs[1],)

        # 添加最后一层的隐藏状态（如果需要输出隐藏状态）
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 根据 return_dict 标志返回不同的输出格式
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        # 返回 TFBaseModelOutput 对象，包含最终的隐藏状态、所有隐藏状态和所有注意力权重
        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )

    def build(self, input_shape=None):
        # 如果已经构建，则直接返回
        if self.built:
            return
        self.built = True
        # 对每一层进行构建
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)


class TFCLIPTextTransformer(keras.layers.Layer):
    def __init__(self, config: CLIPTextConfig, **kwargs):
        super().__init__(**kwargs)

        # 初始化文本 Transformer 的嵌入层
        self.embeddings = TFCLIPTextEmbeddings(config, name="embeddings")
        # 初始化文本 Transformer 的编码器层
        self.encoder = TFCLIPEncoder(config, name="encoder")
        # 初始化最终的层归一化层
        self.final_layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="final_layer_norm")

        # 用于计算 `pooled_output`
        self.eos_token_id = config.eos_token_id  # EOS（End of Sentence）符号的标识符
        self.embed_dim = config.hidden_size  # 嵌入维度大小

    def call(
        self,
        input_ids: TFModelInputType,
        attention_mask: tf.Tensor,
        position_ids: tf.Tensor,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        # 这里略去对 TFCLIPTextTransformer 的 call 方法的注释，因为上面已经详细解释了 TFCLIPEncoder 的 call 方法。
        ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        # 获取输入 `input_ids` 的形状信息
        input_shape = shape_list(input_ids)

        # 使用 `self.embeddings` 对象处理输入 `input_ids` 和 `position_ids`，生成嵌入输出
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # 从输入形状信息中获取批大小和序列长度
        batch_size, seq_length = input_shape
        # CLIP 的文本模型使用因果遮蔽，此处准备因果注意力遮罩
        # 参考链接：https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self._build_causal_attention_mask(batch_size, seq_length, dtype=embedding_output.dtype)

        # 检查注意力遮罩并扩展其维度
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        attention_mask = _expand_mask(attention_mask)

        # 调用编码器 `self.encoder` 进行编码器的前向传播
        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 获取编码器输出的序列输出
        sequence_output = encoder_outputs[0]
        # 应用最终的层归一化到序列输出
        sequence_output = self.final_layer_norm(inputs=sequence_output)

        # 如果 `self.eos_token_id` 为 2，则执行以下操作
        if self.eos_token_id == 2:
            # `eos_token_id` 在 PR #24773 之前是错误的：我们保持这里的操作。
            # 当配置中的 `eos_token_id` 无法正确处理额外的新标记时，具有此类 `eos_token_id` 的 CLIP 模型不能正常工作
            # ------------------------------------------------------------
            # text_embeds.shape = [batch_size, n_ctx, transformer.width]
            # 从 eot 嵌入中获取特征（eot_token 是每个序列中的最高编号）
            pooled_output = tf.gather_nd(
                params=sequence_output,
                indices=tf.stack(
                    values=(tf.range(input_shape[0], dtype=tf.int64), tf.math.argmax(input_ids, axis=-1)), axis=1
                ),
            )
        else:
            # 配置从 PR #24773 中更新了 `eos_token_id`（因此可以使用额外的新标记）
            pooled_output = tf.gather_nd(
                params=sequence_output,
                indices=tf.stack(
                    values=(
                        tf.range(input_shape[0], dtype=tf.int64),
                        tf.math.argmax(tf.cast(input_ids == self.eos_token_id, dtype=tf.int8), axis=-1),
                    ),
                    axis=1,
                ),
            )

        # 如果 `return_dict` 为 False，则返回非字典形式的输出
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        # 返回 TFBaseModelOutputWithPooling 类型的字典形式输出
        return TFBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    # 构建因果注意力掩码
    def _build_causal_attention_mask(self, batch_size, seq_length, dtype=tf.float32):
        # 如果 seq_length 是运行时值，则 tf.constant 不支持，根据 TensorFlow 文档，
        # tf.fill 可以处理运行时动态形状：https://www.tensorflow.org/api_docs/python/tf/fill
        # 创建一个长度为 seq_length 的零对角线张量，并转换为指定的数据类型
        diag = tf.cast(tf.fill((seq_length,), 0.0), dtype)

        # 创建一个形状为 (seq_length, seq_length) 的二维注意力掩码，所有位置初始化为 -10000.0
        to_mask = tf.cast(tf.fill((seq_length, seq_length), -10000.0), dtype)

        # 设置二维注意力掩码的对角线和下三角部分为 0（即不需要掩码的位置）
        to_mask = tf.linalg.band_part(to_mask, 0, -1)
        # to_mask = tf.linalg.band_part(to_mask, -1, 0)  # 如果需要上三角部分也不被掩码，可以取消注释此行
        to_mask = tf.linalg.set_diag(to_mask, diagonal=diag)

        # 将二维注意力掩码扩展成形状为 (batch_size, 1, seq_length, seq_length) 的四维张量并返回
        return tf.broadcast_to(input=to_mask, shape=(batch_size, 1, seq_length, seq_length))

    # 构建模型
    def build(self, input_shape=None):
        if self.built:
            return  # 如果模型已经构建，则直接返回

        self.built = True  # 标记模型已构建

        # 如果存在嵌入层，则构建嵌入层
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)

        # 如果存在编码器，则构建编码器
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)

        # 如果存在最终层归一化，则构建最终层归一化
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                # 构建最终层归一化，输入形状为 [None, None, self.embed_dim]
                self.final_layer_norm.build([None, None, self.embed_dim])
@keras_serializable
class TFCLIPTextMainLayer(keras.layers.Layer):
    # 设置配置类为CLIPTextConfig
    config_class = CLIPTextConfig

    def __init__(self, config: CLIPTextConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        # 初始化文本模型为TFCLIPTextTransformer对象
        self.text_model = TFCLIPTextTransformer(config, name="text_model")

    def get_input_embeddings(self) -> keras.layers.Layer:
        # 返回文本模型的嵌入层
        return self.text_model.embeddings

    def set_input_embeddings(self, value: tf.Variable):
        # 设置文本模型的嵌入层权重和词汇表大小
        self.text_model.embeddings.weight = value
        self.text_model.embeddings.vocab_size = shape_list(value)[0]

    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        if input_ids is None:
            # 如果input_ids为None，则抛出数值错误异常
            raise ValueError("You have to specify input_ids")

        input_shape = shape_list(input_ids)

        if attention_mask is None:
            # 如果attention_mask为None，则创建一个形状与input_ids相同的全1张量
            attention_mask = tf.fill(dims=input_shape, value=1)

        # 调用文本模型进行前向传播
        text_model_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        return text_model_outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "text_model", None) is not None:
            with tf.name_scope(self.text_model.name):
                # 构建文本模型
                self.text_model.build(None)


class TFCLIPVisionTransformer(keras.layers.Layer):
    def __init__(self, config: CLIPVisionConfig, **kwargs):
        super().__init__(**kwargs)

        # 初始化视觉嵌入层、预层归一化、编码器、后层归一化和嵌入维度
        self.embeddings = TFCLIPVisionEmbeddings(config, name="embeddings")
        self.pre_layernorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="pre_layrnorm")
        self.encoder = TFCLIPEncoder(config, name="encoder")
        self.post_layernorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="post_layernorm")
        self.embed_dim = config.hidden_size

    def call(
        self,
        pixel_values: TFModelInputType,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    ):
        # 在视觉Transformer中调用嵌入层、编码器和归一化层，进行前向传播
        pass  # 这里应该有更多的代码，但已经被省略
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        # 输入：像素值作为输入，通过嵌入层获取嵌入输出
        embedding_output = self.embeddings(pixel_values=pixel_values)
        # 对嵌入输出进行预层归一化处理
        embedding_output = self.pre_layernorm(inputs=embedding_output)

        # 使用编码器处理嵌入输出，生成编码器的输出
        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=None,
            causal_attention_mask=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 从编码器输出中提取序列输出
        sequence_output = encoder_outputs[0]
        # 提取池化后的输出，仅保留每个序列的第一个向量
        pooled_output = sequence_output[:, 0, :]
        # 对池化输出进行后层归一化处理
        pooled_output = self.post_layernorm(inputs=pooled_output)

        # 如果不要求返回字典形式的结果，则返回序列输出和池化输出以及可能的其他编码器输出
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        # 如果要求返回字典形式的结果，则构建 TFBaseModelOutputWithPooling 对象
        return TFBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果定义了嵌入层，则构建嵌入层
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        # 如果定义了预层归一化，则构建预层归一化层
        if getattr(self, "pre_layernorm", None) is not None:
            with tf.name_scope(self.pre_layernorm.name):
                self.pre_layernorm.build([None, None, self.embed_dim])
        # 如果定义了编码器，则构建编码器
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果定义了后层归一化，则构建后层归一化层
        if getattr(self, "post_layernorm", None) is not None:
            with tf.name_scope(self.post_layernorm.name):
                self.post_layernorm.build([None, self.embed_dim])
# 使用 keras_serializable 装饰器声明 TFCLIPVisionMainLayer 类，使其可序列化
@keras_serializable
class TFCLIPVisionMainLayer(keras.layers.Layer):
    # 设定配置类为 CLIPVisionConfig
    config_class = CLIPVisionConfig

    # 初始化方法，接受配置对象 config 和其他关键字参数 kwargs
    def __init__(self, config: CLIPVisionConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 将传入的配置对象保存在 self.config 中
        self.config = config
        # 创建 TFCLIPVisionTransformer 对象作为视觉模型，命名为 "vision_model"
        self.vision_model = TFCLIPVisionTransformer(config, name="vision_model")

    # 返回视觉模型的嵌入层
    def get_input_embeddings(self) -> keras.layers.Layer:
        return self.vision_model.embeddings

    # 使用 unpack_inputs 装饰器定义 call 方法，处理输入参数，并返回视觉模型的输出
    @unpack_inputs
    def call(
        self,
        pixel_values: TFModelInputType | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        # 如果 pixel_values 为 None，抛出数值错误异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 调用视觉模型，传入参数并获取其输出
        vision_model_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 返回视觉模型的输出
        return vision_model_outputs

    # 构建方法，用于构建层，确保视觉模型已经构建
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 设置标志位表明已构建
        self.built = True
        # 如果视觉模型存在，则在名称作用域内构建它
        if getattr(self, "vision_model", None) is not None:
            with tf.name_scope(self.vision_model.name):
                self.vision_model.build(None)


# 使用 keras_serializable 装饰器声明 TFCLIPMainLayer 类，使其可序列化
@keras_serializable
class TFCLIPMainLayer(keras.layers.Layer):
    # 设定配置类为 CLIPConfig
    config_class = CLIPConfig
    # 初始化方法，接受一个CLIPConfig对象和其他关键字参数
    def __init__(self, config: CLIPConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 检查config.text_config是否为CLIPTextConfig类型，如果不是则抛出数值错误异常
        if not isinstance(config.text_config, CLIPTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type CLIPTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        # 检查config.vision_config是否为CLIPVisionConfig类型，如果不是则抛出数值错误异常
        if not isinstance(config.vision_config, CLIPVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type CLIPVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        # 将传入的config对象保存到实例变量self.config中
        self.config = config

        # 分别获取text_config和vision_config对象，保存到局部变量中
        text_config = config.text_config
        vision_config = config.vision_config

        # 设置实例变量projection_dim为config.projection_dim
        self.projection_dim = config.projection_dim

        # 创建TFCLIPTextTransformer实例并保存到实例变量self.text_model中
        self.text_model = TFCLIPTextTransformer(text_config, name="text_model")
        
        # 创建TFCLIPVisionTransformer实例并保存到实例变量self.vision_model中
        self.vision_model = TFCLIPVisionTransformer(vision_config, name="vision_model")

        # 创建一个Dense层用于视觉特征的投影，设置单元数为projection_dim，
        # 使用指定的初始化器初始化权重，不使用偏置，命名为visual_projection
        self.visual_projection = keras.layers.Dense(
            units=self.projection_dim,
            kernel_initializer=get_initializer(vision_config.hidden_size**-0.5 * self.config.initializer_factor),
            use_bias=False,
            name="visual_projection",
        )

        # 创建一个Dense层用于文本特征的投影，设置单元数为projection_dim，
        # 使用指定的初始化器初始化权重，不使用偏置，命名为text_projection
        self.text_projection = keras.layers.Dense(
            units=self.projection_dim,
            kernel_initializer=get_initializer(text_config.hidden_size**-0.5 * self.config.initializer_factor),
            use_bias=False,
            name="text_projection",
        )

        # 设置实例变量text_embed_dim为text_config.hidden_size
        self.text_embed_dim = text_config.hidden_size
        
        # 设置实例变量vision_embed_dim为vision_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

    # 构建方法，用于构建模型的组件
    def build(self, input_shape: tf.TensorShape = None):
        # 添加一个名为logit_scale的可训练权重，形状为(1,)，初始值为config.logit_scale_init_value
        self.logit_scale = self.add_weight(
            shape=(1,),
            initializer=keras.initializers.Constant(self.config.logit_scale_init_value),
            trainable=True,
            name="logit_scale",
        )

        # 如果已经构建过模型，则直接返回，避免重复构建
        if self.built:
            return
        self.built = True

        # 如果存在text_model实例，则在名为text_model的作用域内构建text_model
        if getattr(self, "text_model", None) is not None:
            with tf.name_scope(self.text_model.name):
                self.text_model.build(None)

        # 如果存在vision_model实例，则在名为vision_model的作用域内构建vision_model
        if getattr(self, "vision_model", None) is not None:
            with tf.name_scope(self.vision_model.name):
                self.vision_model.build(None)

        # 如果存在visual_projection实例，则在名为visual_projection的作用域内构建visual_projection
        if getattr(self, "visual_projection", None) is not None:
            with tf.name_scope(self.visual_projection.name):
                self.visual_projection.build([None, None, self.vision_embed_dim])

        # 如果存在text_projection实例，则在名为text_projection的作用域内构建text_projection
        if getattr(self, "text_projection", None) is not None:
            with tf.name_scope(self.text_projection.name):
                self.text_projection.build([None, None, self.text_embed_dim])

    # 使用装饰器unpack_inputs修饰的方法，用于解包输入数据
    @unpack_inputs
    @unpack_inputs
    # 使用装饰器 unpack_inputs 解包输入参数，使得方法可以接收多种输入形式
    def get_text_features(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> tf.Tensor:
        # 如果没有提供 input_ids，则抛出数值错误异常
        if input_ids is None:
            raise ValueError("You have to specify either input_ids")

        # 获取 input_ids 的形状信息
        input_shape = shape_list(input_ids)

        # 如果 attention_mask 未提供，则使用维度为 input_shape 的全 1 张量填充
        if attention_mask is None:
            attention_mask = tf.fill(dims=input_shape, value=1)

        # 使用 text_model 处理输入数据，获取文本模型的输出
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 从 text_outputs 中获取池化输出（一般是第二个输出）
        pooled_output = text_outputs[1]
        # 使用 text_projection 对池化输出进行投影，得到文本特征表示
        text_features = self.text_projection(inputs=pooled_output)

        # 返回文本特征表示
        return text_features

    @unpack_inputs
    # 使用装饰器 unpack_inputs 解包输入参数，使得方法可以接收多种输入形式
    def get_image_features(
        self,
        pixel_values: TFModelInputType | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> tf.Tensor:
        # 如果没有提供 pixel_values，则抛出数值错误异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 使用 vision_model 处理输入数据，获取视觉模型的输出
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 从 vision_outputs 中获取池化输出（一般是第二个输出）
        pooled_output = vision_outputs[1]
        # 使用 visual_projection 对池化输出进行投影，得到图像特征表示
        image_features = self.visual_projection(inputs=pooled_output)

        # 返回图像特征表示
        return image_features

    @unpack_inputs
    # 使用装饰器 unpack_inputs 解包输入参数，使得方法可以接收多种输入形式
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        pixel_values: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        # 如果既没有提供 input_ids 也没有提供 pixel_values，则抛出数值错误异常
        if input_ids is None and pixel_values is None:
            raise ValueError("You have to specify either input_ids or pixel_values")

        # 如果提供了 input_ids，则处理文本特征
        if input_ids is not None:
            input_shape = shape_list(input_ids)
            if attention_mask is None:
                attention_mask = tf.fill(dims=input_shape, value=1)
            text_outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                training=training,
            )
            pooled_output = text_outputs[1]
            text_features = self.text_projection(inputs=pooled_output)
            return text_features

        # 如果提供了 pixel_values，则处理图像特征
        if pixel_values is not None:
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                training=training,
            )
            pooled_output = vision_outputs[1]
            image_features = self.visual_projection(inputs=pooled_output)
            return image_features

        # 如果设置了 return_loss 为 True，则返回损失值和相应特征
        if return_loss:
            return text_features, image_features

        # 否则根据 return_dict 的设置返回相应的特征表示
        if return_dict:
            return {'text_features': text_features, 'image_features': image_features}
        else:
            return text_features, image_features
    ) -> Union[TFCLIPOutput, Tuple[tf.Tensor]]:
        # 如果未提供 input_ids 参数，则抛出数值错误
        if input_ids is None:
            raise ValueError("You have to specify either input_ids")
        # 如果未提供 pixel_values 参数，则抛出数值错误
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 获取输入张量的形状信息
        input_shape = shape_list(input_ids)

        # 如果未提供 attention_mask 参数，则用值为 1 填充并赋给 attention_mask
        if attention_mask is None:
            attention_mask = tf.fill(dims=input_shape, value=1)

        # 调用 vision_model 处理图像输入，并返回相应的输出
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 调用 text_model 处理文本输入，并返回相应的输出
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 从 vision_outputs 中提取图像嵌入向量
        image_embeds = vision_outputs[1]
        # 通过 visual_projection 对图像嵌入向量进行投影
        image_embeds = self.visual_projection(inputs=image_embeds)

        # 从 text_outputs 中提取文本嵌入向量
        text_embeds = text_outputs[1]
        # 通过 text_projection 对文本嵌入向量进行投影
        text_embeds = self.text_projection(inputs=text_embeds)

        # 对图像嵌入向量进行标准化处理
        image_embeds = image_embeds / tf.norm(tensor=image_embeds, ord="euclidean", axis=-1, keepdims=True)
        # 对文本嵌入向量进行标准化处理
        text_embeds = text_embeds / tf.norm(tensor=text_embeds, ord="euclidean", axis=-1, keepdims=True)

        # 计算余弦相似度作为 logits
        logit_scale = tf.math.exp(self.logit_scale)
        logits_per_text = tf.matmul(text_embeds, image_embeds, transpose_b=True) * logit_scale
        logits_per_image = tf.transpose(logits_per_text)

        # 初始化 loss 为 None
        loss = None
        # 如果需要返回 loss，则计算 clip_loss 并将其重塑为形状为 (1,) 的张量
        if return_loss:
            loss = clip_loss(logits_per_text)
            loss = tf.reshape(loss, (1,))

        # 如果不返回字典格式的输出，以元组形式返回各种输出
        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return (loss,) + output if loss is not None else output

        # 如果需要返回字典格式的输出，则构建 TFCLIPOutput 对象并返回
        return TFCLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )
# CLIP 文本输入的文档字符串，用于说明如何传递文本输入给模型
CLIP_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.__call__`] and
            [`PreTrainedTokenizer.encode`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
            config will be used instead.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in
            eager mode, in graph mode the value will always be set to True.
        training (`bool`, *optional*, defaults to `False``):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).


注释：

# input_ids: 输入序列token在词汇表中的索引，可以是np.ndarray、tf.Tensor、List[tf.Tensor]、Dict[str, tf.Tensor]或Dict[str, np.ndarray]类型，每个示例必须具有形状为({0})。
# attention_mask: 可选参数，用于避免在填充token索引上执行注意力操作的掩码。掩码值在[0, 1]之间选择：
#   - 1表示不被掩盖的token，
#   - 0表示被掩盖的token。
# position_ids: 可选参数，输入序列中每个token在位置嵌入中的位置索引。选择范围为[0, config.max_position_embeddings - 1]。
# output_attentions: 可选参数，是否返回所有注意力层的注意力张量。详细信息请参见返回的张量中的`attentions`。此参数仅在动态图模式下有效，在静态图模式下将使用配置中的值。
# output_hidden_states: 可选参数，是否返回所有层的隐藏状态。详细信息请参见返回的张量中的`hidden_states`。此参数仅在动态图模式下有效，在静态图模式下将使用配置中的值。
# return_dict: 可选参数，是否返回[`~utils.ModelOutput`]而不是普通元组。此参数可以在动态图模式下使用，在静态图模式下将始终设置为True。
# training: 可选参数，默认为`False`，指示模型是否处于训练模式（例如，某些模块如dropout在训练和评估之间有不同的行为）。
# CLIP_VISION_INPUTS_DOCSTRING 是一个原始字符串（raw string），用于描述 CLIP 模型的输入参数。
CLIP_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`CLIPImageProcessor.__call__`] for details. output_attentions (`bool`, *optional*): Whether or not to
            return the attentions tensors of all attention layers. See `attentions` under returned tensors for more
            detail. This argument can be used only in eager mode, in graph mode the value in the config will be used
            instead.
        output_hidden_states (`bool`, *optional`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        return_dict (`bool`, *optional`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in
            eager mode, in graph mode the value will always be set to True.
        training (`bool`, *optional*, defaults to `False``):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""

# CLIP_INPUTS_DOCSTRING 是一个原始字符串（raw string），用于描述 CLIP 模型的输入参数，不同于 CLIP_VISION_INPUTS_DOCSTRING。
CLIP_INPUTS_DOCSTRING = r"""
"""
    # 定义一个函数，接受多种类型的输入数据作为参数，这些数据用于描述输入序列的特征和掩码
    Args:
        # 输入序列的标记索引，可以是多种数据类型，如 np.ndarray, tf.Tensor, List[tf.Tensor], Dict[str, tf.Tensor] 或 Dict[str, np.ndarray]
        # 每个样本都必须具有形状为 ({0}) 的索引
        input_ids (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]`, `Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `({0})`):
            Indices of input sequence tokens in the vocabulary.
            
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.__call__`] and
            [`PreTrainedTokenizer.encode`] for details.
            
            [What are input IDs?](../glossary#input-ids)
        
        # 像素值，可以是多种数据类型，如 np.ndarray, tf.Tensor, List[tf.Tensor], Dict[str, tf.Tensor] 或 Dict[str, np.ndarray]
        # 每个样本必须具有形状为 (batch_size, num_channels, height, width) 的像素值
        pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]`, `Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`CLIPImageProcessor.__call__`] for details.
        
        # 可选参数，用于避免在填充标记索引上执行注意力操作的掩码
        attention_mask (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            
            [What are attention masks?](../glossary#attention-mask)
        
        # 可选参数，指定每个输入序列标记在位置嵌入中的位置索引
        position_ids (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.
            
            [What are position IDs?](../glossary#position-ids)
        
        # 可选参数，指定是否返回对比损失
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.
        
        # 可选参数，在 eager 模式下是否返回所有注意力层的注意力张量
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
            config will be used instead.
        
        # 可选参数，在 eager 模式下是否返回所有层的隐藏状态
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        
        # 可选参数，指定是否返回一个 `~utils.ModelOutput` 而不是普通的元组。在 eager 模式下可以使用，图模式下始终为 True。
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in
            eager mode, in graph mode the value will always be set to True.
        
        # 可选参数，指定是否以训练模式运行模型（某些模块如 dropout 在训练和评估中有不同的行为）
        training (`bool`, *optional*, defaults to `False`):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""
Define TFCLIPTextModel class inheriting from TFCLIPPreTrainedModel.
"""
class TFCLIPTextModel(TFCLIPPreTrainedModel):
    # Specify the configuration class for text CLIP
    config_class = CLIPTextConfig

    def __init__(self, config: CLIPTextConfig, *inputs, **kwargs):
        """
        Initialize TFCLIPTextModel.

        Args:
            config (CLIPTextConfig): Model configuration object.
            *inputs: Variable length input arguments.
            **kwargs: Keyword arguments for additional configuration.
        """
        # Call superclass initialization
        super().__init__(config, *inputs, **kwargs)

        # Initialize the main CLIP text layer
        self.clip = TFCLIPTextMainLayer(config, name="clip")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CLIP_TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFBaseModelOutputWithPooling, config_class=CLIPTextConfig)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        """
        Perform the forward pass of the model.

        Args:
            input_ids (TFModelInputType, optional): Input tensor of token ids.
            attention_mask (np.ndarray or tf.Tensor, optional): Attention mask for masking padded tokens.
            position_ids (np.ndarray or tf.Tensor, optional): Position indices for the input tokens.
            output_attentions (bool, optional): Whether to output attentions.
            output_hidden_states (bool, optional): Whether to output hidden states.
            return_dict (bool, optional): Whether to return a dictionary.
            training (bool, optional): Whether the model is in training mode.

        Returns:
            TFBaseModelOutputWithPooling or Tuple[tf.Tensor]: Model outputs.

        Examples:
            Example usage of the model:
            ```python
            >>> from transformers import AutoTokenizer, TFCLIPTextModel

            >>> model = TFCLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
            >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

            >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="tf")

            >>> outputs = model(**inputs)
            >>> last_hidden_state = outputs.last_hidden_state
            >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
            ```
        """
        # Forward pass through the CLIP model
        outputs = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        return outputs

    def build(self, input_shape=None):
        """
        Build method for constructing the model.

        Args:
            input_shape: Shape of the input tensor (not used here).
        """
        if self.built:
            return
        self.built = True
        # Build the main CLIP layer if defined
        if getattr(self, "clip", None) is not None:
            with tf.name_scope(self.clip.name):
                self.clip.build(None)


"""
Define TFCLIPVisionModel class inheriting from TFCLIPPreTrainedModel.
"""
class TFCLIPVisionModel(TFCLIPPreTrainedModel):
    # Specify the configuration class for vision CLIP
    config_class = CLIPVisionConfig
    # Define the main input name for vision model
    main_input_name = "pixel_values"

    def __init__(self, config: CLIPVisionConfig, *inputs, **kwargs):
        """
        Initialize TFCLIPVisionModel.

        Args:
            config (CLIPVisionConfig): Model configuration object.
            *inputs: Variable length input arguments.
            **kwargs: Keyword arguments for additional configuration.
        """
        # Call superclass initialization
        super().__init__(config, *inputs, **kwargs)

        # Initialize the main CLIP vision layer
        self.clip = TFCLIPVisionMainLayer(config, name="clip")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBaseModelOutputWithPooling, config_class=CLIPVisionConfig)
    def call(
        self,
        pixel_values: TFModelInputType | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,

    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        """
        Perform the forward pass of the model.

        Args:
            pixel_values (TFModelInputType, optional): Input tensor of pixel values.
            output_attentions (bool, optional): Whether to output attentions.
            output_hidden_states (bool, optional): Whether to output hidden states.
            return_dict (bool, optional): Whether to return a dictionary.
            training (bool, optional): Whether the model is in training mode.

        Returns:
            TFBaseModelOutputWithPooling or Tuple[tf.Tensor]: Model outputs.
        """
        # Forward pass through the CLIP model
        outputs = self.clip(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        return outputs
        ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFCLIPVisionModel

        >>> model = TFCLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="tf")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""

        # 调用 self.clip 方法进行模型推断
        outputs = self.clip(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 返回推断的输出结果
        return outputs

    def build(self, input_shape=None):
        # 如果模型已经构建完成，直接返回
        if self.built:
            return
        # 设置模型已构建标志为 True
        self.built = True
        # 如果 self.clip 存在，则在命名空间下构建 clip 模型
        if getattr(self, "clip", None) is not None:
            with tf.name_scope(self.clip.name):
                self.clip.build(None)
# 使用装饰器为类添加文档字符串，使用CLIP_START_DOCSTRING作为模板
@add_start_docstrings(CLIP_START_DOCSTRING)
class TFCLIPModel(TFCLIPPreTrainedModel):
    # 设置配置类为CLIPConfig
    config_class = CLIPConfig

    # 初始化方法，接受配置对象config和任意额外输入
    def __init__(self, config: CLIPConfig, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 创建TFCLIPMainLayer实例并赋给self.clip
        self.clip = TFCLIPMainLayer(config, name="clip")

    # 使用装饰器解包输入并添加文档字符串，使用CLIP_TEXT_INPUTS_DOCSTRING作为模板
    @unpack_inputs
    @add_start_docstrings_to_model_forward(CLIP_TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def get_text_features(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> tf.Tensor:
        r"""
        返回文本特征张量(`tf.Tensor`，形状为`(batch_size, output_dim)`):
        通过将投影层应用于[`TFCLIPTextModel`]的汇总输出获得的文本嵌入。

        Examples:
        
        ```python
        >>> from transformers import AutoTokenizer, TFCLIPModel
        
        >>> model = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="tf")
        >>> text_features = model.get_text_features(**inputs)
        ```"""

        # 调用self.clip的get_text_features方法，传入各种输入参数
        text_features = self.clip.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 返回文本特征张量
        return text_features

    # 使用装饰器解包输入并添加文档字符串，使用CLIP_VISION_INPUTS_DOCSTRING作为模板
    @unpack_inputs
    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values: TFModelInputType | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> tf.Tensor:
        r"""
        Returns:
            image_features (`tf.Tensor` of shape `(batch_size, output_dim`): The image embeddings obtained by applying
            the projection layer to the pooled output of [`TFCLIPVisionModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFCLIPModel

        >>> model = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="tf")

        >>> image_features = model.get_image_features(**inputs)
        ```"""

        # 调用 CLIP 模型获取图像特征
        image_features = self.clip.get_image_features(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 返回获取的图像特征张量
        return image_features

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CLIP_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFCLIPOutput, config_class=CLIPConfig)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        pixel_values: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    def serving_output(self, output: TFCLIPOutput) -> TFCLIPOutput:
        """
        返回经过服务输出处理后的 TFCLIPOutput 对象。

        Parameters:
        output (TFCLIPOutput): 待处理的 TFCLIPOutput 对象。

        Returns:
        TFCLIPOutput: 经过服务输出处理后的 TFCLIPOutput 对象。
        """
        # TODO: 目前在 saved_model=True 模式下存在问题，因为 TensorFlow 无法追踪嵌套的 dataclass 结构。
        # 参考链接: https://github.com/huggingface/transformers/pull/16886
        return output

    def build(self, input_shape=None):
        """
        构建模型的方法。如果已经构建过，则直接返回，否则进行构建。

        Parameters:
        input_shape: 输入张量的形状，默认为 None。
        """
        if self.built:
            return
        self.built = True
        # 如果模型已经包含了 CLIP 模型实例，则在命名空间下构建该模型。
        if getattr(self, "clip", None) is not None:
            with tf.name_scope(self.clip.name):
                self.clip.build(None)
```