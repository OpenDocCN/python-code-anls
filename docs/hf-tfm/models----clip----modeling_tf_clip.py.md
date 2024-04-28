# `.\transformers\models\clip\modeling_tf_clip.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，禁止未经许可使用此文件
# 可以在以下链接获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”基础分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制
""" TF 2.0 CLIP model."""

# 导入必要的库
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import tensorflow as tf

# 导入相关模块
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling
from ...modeling_tf_utils import (
    TFModelInputType,
    TFPreTrainedModel,
    get_initializer,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点
_CHECKPOINT_FOR_DOC = "openai/clip-vit-base-patch32"

# 预训练模型存档列表
TF_CLIP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "openai/clip-vit-base-patch32",
    # 查看所有 CLIP 模型 https://huggingface.co/models?filter=clip
]

# 定义一个大负数常量
LARGE_NEGATIVE = -1e8

# 从 transformers.models.bart.modeling_tf_bart._expand_mask 复制的函数
def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None):
    """
    将注意力掩码从 `[bsz, seq_len]` 扩展到 `[bsz, 1, tgt_seq_len, src_seq_len]`。
    """
    src_len = shape_list(mask)[1]
    tgt_len = tgt_len if tgt_len is not None else src_len
    one_cst = tf.constant(1.0)
    mask = tf.cast(mask, dtype=one_cst.dtype)
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))

    return (one_cst - expanded_mask) * LARGE_NEGATIVE

# 对比损失函数，改编自 https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits: tf.Tensor) -> tf.Tensor:
    return tf.math.reduce_mean(
        tf.keras.metrics.sparse_categorical_crossentropy(
            y_true=tf.range(shape_list(logits)[0]), y_pred=logits, from_logits=True
        )
    )

# CLIP 损失函数
def clip_loss(similarity: tf.Tensor) -> tf.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(tf.transpose(similarity))
    return (caption_loss + image_loss) / 2.0

# TFCLIPOutput 数据类
@dataclass
class TFCLIPOutput(ModelOutput):
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
    
    # 定义属性：损失张量，表示图像-文本相似性的对比损失，可选，当`return_loss`为`True`时返回
    loss: tf.Tensor | None = None
    # 定义属性：每个图像的文本对应的 logits，表示`image_embeds`和`text_embeds`之间的点积得分，表示图像-文本相似性得分
    logits_per_image: tf.Tensor = None
    # 定义属性：每个文本的图像对应的 logits，表示`text_embeds`和`image_embeds`之间的点积得分，表示文本-图像相似性得分
    logits_per_text: tf.Tensor = None
    # 定义属性：文本嵌入张量，通过将`TFCLIPTextModel`的池化输出应用到投影层得到
    text_embeds: tf.Tensor = None
    # 定义属性：图像嵌入张量，通过将`TFCLIPVisionModel`的池化输出应用到投影层得到
    image_embeds: tf.Tensor = None
    # 定义属性：文本模型输出，是[`TFCLIPTextModel`]的输出
    text_model_output: TFBaseModelOutputWithPooling = None
    # 定义属性：视觉模型输出，是[`TFCLIPVisionModel`]的输出
    vision_model_output: TFBaseModelOutputWithPooling = None

    # 定义方法：将对象转换为元组
    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            # 如果键不是"text_model_output"或"vision_model_output"，则返回对应的属性值，否则返回属性值的元组形式
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )
class TFCLIPVisionEmbeddings(tf.keras.layers.Layer):
    # 定义 TFCLIPVisionEmbeddings 类，继承自 tf.keras.layers.Layer
    def __init__(self, config: CLIPVisionConfig, **kwargs):
        # 初始化方法，接受一个 CLIPVisionConfig 类型的参数和其他关键字参数
        super().__init__(**kwargs)
        # 调用父类的初始化方法

        self.embed_dim = config.hidden_size
        # 从 config 中获取隐藏层大小作为嵌入维度
        self.image_size = config.image_size
        # 从 config 中获取图像大小
        self.patch_size = config.patch_size
        # 从 config 中获取patch大小

        self.num_patches = (self.image_size // self.patch_size) ** 2
        # 计算图像中的patch数量
        self.num_positions = self.num_patches + 1
        # 计算位置编码的数量

        self.config = config
        # 保存 config 对象

        self.patch_embedding = tf.keras.layers.Conv2D(
            filters=self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="valid",
            data_format="channels_last",
            use_bias=False,
            kernel_initializer=get_initializer(self.config.initializer_range * self.config.initializer_factor),
            name="patch_embedding",
        )
        # 创建卷积层用于patch嵌入

    def build(self, input_shape: tf.TensorShape = None):
        # 构建方法，接受输入形状参数，默认为 None
        factor = self.config.initializer_factor
        # 获取初始化因子

        self.class_embedding = self.add_weight(
            shape=(self.embed_dim,),
            initializer=get_initializer(self.embed_dim**-0.5 * factor),
            trainable=True,
            name="class_embedding",
        )
        # 添加类别嵌入权重

        with tf.name_scope("position_embedding"):
            self.position_embedding = self.add_weight(
                shape=(self.num_positions, self.embed_dim),
                initializer=get_initializer(self.config.initializer_range * factor),
                trainable=True,
                name="embeddings",
            )
        # 添加位置嵌入权重

        if self.built:
            return
        # 如果已经构建过，则直接返回
        self.built = True
        # 标记为已构建
        if getattr(self, "patch_embedding", None) is not None:
            with tf.name_scope(self.patch_embedding.name):
                self.patch_embedding.build([None, None, None, self.config.num_channels])
        # 构建patch嵌入层

    def call(self, pixel_values: tf.Tensor) -> tf.Tensor:
        """`pixel_values` is expected to be of NCHW format."""
        # 定义 call 方法，接受像素值张量，返回张量，期望像素值为NCHW格式

        batch_size, num_channels, height, width = shape_list(pixel_values)
        # 获取像素值张量的形状信息

        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))
        # 将像素值张量转置为NHWC格式

        patch_embeds = self.patch_embedding(pixel_values)
        # 使用patch嵌入层处理像素值张量

        patch_embeds = tf.reshape(tensor=patch_embeds, shape=(batch_size, self.num_patches, -1))
        # 将嵌入结果reshape为(batch_size, num_patches, embed_dim)

        class_embeds = tf.broadcast_to(self.class_embedding, shape=(batch_size, 1, self.embed_dim))
        # 将类别嵌入广播到与patch嵌入相同的形状

        embeddings = tf.concat((class_embeds, patch_embeds), axis=1)
        # 拼接类别嵌入和patch嵌入

        embeddings = embeddings + self.position_embedding
        # 添加位置嵌入

        return embeddings
        # 返回嵌入结果
    # 初始化函数，初始化一个 CLIPTextEmbedding 实例
    def __init__(self, config: CLIPTextConfig, **kwargs):
        # 调用父类初始化方法
        super().__init__(**kwargs)

        # 设定嵌入维度为配置中的隐藏层大小
        self.embed_dim = config.hidden_size

        # 保存配置信息
        self.config = config

    # 构建模型
    def build(self, input_shape: tf.TensorShape = None):
        # 在名为 "token_embedding" 的命名空间下
        with tf.name_scope("token_embedding"):
            # 添加权重矩阵，形状为 (词汇大小, 嵌入维度)，可训练
            self.weight = self.add_weight(
                shape=(self.config.vocab_size, self.embed_dim),
                # 使用给定的初始化器初始化权重
                initializer=get_initializer(self.config.initializer_factor * self.config.initializer_range),
                trainable=True,
                name="weight",
            )

        # 在名为 "position_embedding" 的命名空间下
        with tf.name_scope("position_embedding"):
            # 添加位置嵌入矩阵，形状为 (最大位置嵌入数量, 嵌入维度)，可训练
            self.position_embedding = self.add_weight(
                shape=(self.config.max_position_embeddings, self.embed_dim),
                # 使用给定的初始化器初始化权重
                initializer=get_initializer(self.config.initializer_factor * self.config.initializer_range),
                trainable=True,
                name="embeddings",
            )

        # 调用父类的 build 方法
        super().build(input_shape)

    # 调用模型
    def call(
        self,
        input_ids: tf.Tensor = None,
        position_ids: tf.Tensor = None,
        inputs_embeds: tf.Tensor = None,
    ) -> tf.Tensor:
        """
        根据输入张量应用嵌入。

        Returns:
            final_embeddings (`tf.Tensor`): 输出嵌入张量。
        """
        # 若未提供 input_ids 或 inputs_embeds，则抛出异常
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 若未提供 inputs_embeds，则根据 input_ids 检查嵌入是否在合理范围内，并从权重中取出对应的嵌入
        if inputs_embeds is None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        # 获取输入嵌入张量的形状
        input_shape = shape_list(inputs_embeds)[:-1]

        # 若未提供 position_ids，则创建默认的位置编码
        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)

        # 根据位置编码获取位置嵌入
        position_embeds = tf.gather(params=self.position_embedding, indices=position_ids)
        # 将位置嵌入扩展到与输入嵌入相同的形状
        position_embeds = tf.tile(input=position_embeds, multiples=(input_shape[0], 1, 1))
        # 将输入嵌入与位置嵌入相加得到最终嵌入张量
        final_embeddings = inputs_embeds + position_embeds

        # 返回最终嵌入张量
        return final_embeddings
class TFCLIPAttention(tf.keras.layers.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: CLIPConfig, **kwargs):
        super().__init__(**kwargs)

        # 获取配置中的隐藏层大小
        self.embed_dim = config.hidden_size
        # 获取配置中的注意力头数
        self.num_attention_heads = config.num_attention_heads
        # 计算每个注意力头的大小
        self.attention_head_size = self.embed_dim // self.num_attention_heads
        # 如果 embed_dim 不能被 num_heads 整除，则抛出异常
        if self.attention_head_size * self.num_attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_attention_heads})."
            )

        # 初始化标准差因子
        factor = config.initializer_factor
        # 计算输入投影矩阵的标准差
        in_proj_std = (self.embed_dim**-0.5) * ((2 * config.num_hidden_layers) ** -0.5) * factor
        # 计算输出投影矩阵的标准差
        out_proj_std = (self.embed_dim**-0.5) * factor

        # 计算注意力头大小的平方根
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        # 初始化查询投影层
        self.q_proj = tf.keras.layers.Dense(
            units=self.embed_dim, kernel_initializer=get_initializer(in_proj_std), name="q_proj"
        )
        # 初始化键投影层
        self.k_proj = tf.keras.layers.Dense(
            units=self.embed_dim, kernel_initializer=get_initializer(in_proj_std), name="k_proj"
        )
        # 初始化值投影层
        self.v_proj = tf.keras.layers.Dense(
            units=self.embed_dim, kernel_initializer=get_initializer(in_proj_std), name="v_proj"
        )

        # 初始化丢弃层
        self.dropout = tf.keras.layers.Dropout(rate=config.attention_dropout)

        # 初始化输出投影层
        self.out_proj = tf.keras.layers.Dense(
            units=self.embed_dim, kernel_initializer=get_initializer(out_proj_std), name="out_proj"
        )

    # 从 Transformers 中的 TFBertSelfAttention 中复制 transpose_for_scores 方法
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
        """定义一个方法，接受隐藏状态作为输入，返回注意力输出和注意力权重

        输入形状: Batch x Time x Channel
        """
        
        # 获取隐藏状态的批量大小
        batch_size = shape_list(hidden_states)[0]
        
        # 通过线性变换得到查询、键、值
        mixed_query_layer = self.q_proj(inputs=hidden_states)
        mixed_key_layer = self.k_proj(inputs=hidden_states)
        mixed_value_layer = self.v_proj(inputs=hidden_states)
        
        # 调整形状以便进行注意力计算
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        # 计算注意力分数
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, dk)

        # 应用因果注意力掩码
        if causal_attention_mask is not None:
            attention_scores = tf.add(attention_scores, causal_attention_mask)

        # 应用注意力掩码
        if attention_mask is not None:
            attention_scores = tf.add(attention_scores, attention_mask)

        # 将注意力分数归一化为概率
        _attention_probs = stable_softmax(logits=attention_scores, axis=-1)

        # 使用dropout进行注意力权重的处理
        attention_probs = self.dropout(inputs=_attention_probs, training=training)

        # 计算注意力输出
        attention_output = tf.matmul(attention_probs, value_layer)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])

        # 重塑注意力输出的形状
        attention_output = tf.reshape(tensor=attention_output, shape=(batch_size, -1, self.embed_dim))

        # 通过输出投影层处理注意力输出
        attention_output = self.out_proj(attention_output, training=training)

        # 根据需要返回注意力输出和注意力权重
        outputs = (attention_output, _attention_probs) if output_attentions else (attention_output,)

        return outputs
    # 检查是否已经构建了层，如果已经构建则直接返回，避免重复构建
    def build(self, input_shape=None):
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在查询投影（q_proj）属性，则构建查询投影层
        if getattr(self, "q_proj", None) is not None:
            # 使用查询投影层的名称作为命名空间，构建查询投影层
            with tf.name_scope(self.q_proj.name):
                # 构建查询投影层，输入形状为 [None, None, self.embed_dim]
                self.q_proj.build([None, None, self.embed_dim])
        # 如果存在键投影（k_proj）属性，则构建键投影层
        if getattr(self, "k_proj", None) is not None:
            # 使用键投影层的名称作为命名空间，构建键投影层
            with tf.name_scope(self.k_proj.name):
                # 构建键投影层，输入形状为 [None, None, self.embed_dim]
                self.k_proj.build([None, None, self.embed_dim])
        # 如果存在值投影（v_proj）属性，则构建值投影层
        if getattr(self, "v_proj", None) is not None:
            # 使用值投影层的名称作为命名空间，构建值投影层
            with tf.name_scope(self.v_proj.name):
                # 构建值投影层，输入形状为 [None, None, self.embed_dim]
                self.v_proj.build([None, None, self.embed_dim])
        # 如果存在输出投影（out_proj）属性，则构建输出投影层
        if getattr(self, "out_proj", None) is not None:
            # 使用输出投影层的名称作为命名空间，构建输出投影层
            with tf.name_scope(self.out_proj.name):
                # 构建输出投影层，输入形状为 [None, None, self.embed_dim]
                self.out_proj.build([None, None, self.embed_dim])
class TFCLIPMLP(tf.keras.layers.Layer):
    # 初始化方法，接受一个 CLIPConfig 对象和其他关键字参数
    def __init__(self, config: CLIPConfig, **kwargs):
        super().__init__(**kwargs)

        # 根据配置获取激活函数
        self.activation_fn = get_tf_activation(config.hidden_act)

        # 计算初始化因子
        factor = config.initializer_factor
        in_proj_std = (config.hidden_size**-0.5) * ((2 * config.num_hidden_layers) ** -0.5) * factor
        fc_std = (2 * config.hidden_size) ** -0.5 * factor

        # 创建第一个全连接层
        self.fc1 = tf.keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(fc_std), name="fc1"
        )
        # 创建第二个全连接层
        self.fc2 = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(in_proj_std), name="fc2"
        )
        # 保存配置
        self.config = config

    # 前向传播方法，接受隐藏状态张量，返回变换后的隐藏状态张量
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 全连接层变换
        hidden_states = self.fc1(inputs=hidden_states)
        # 激活函数
        hidden_states = self.activation_fn(hidden_states)
        # 全连接层变换
        hidden_states = self.fc2(inputs=hidden_states)
        # 返回变换后的隐藏状态张量
        return hidden_states

    # 构建方法，在第一次调用 call 方法时被调用，用于构建层的参数
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经创建了 fc1 层，则构建 fc1 层的参数
        if getattr(self, "fc1", None) is not None:
            with tf.name_scope(self.fc1.name):
                self.fc1.build([None, None, self.config.hidden_size])
        # 如果已经创建了 fc2 层，则构建 fc2 层的参数
        if getattr(self, "fc2", None) is not None:
            with tf.name_scope(self.fc2.name):
                self.fc2.build([None, None, self.config.intermediate_size])


class TFCLIPEncoderLayer(tf.keras.layers.Layer):
    # 初始化方法，接受一个 CLIPConfig 对象和其他关键字参数
    def __init__(self, config: CLIPConfig, **kwargs):
        super().__init__(**kwargs)

        # 获取嵌入维度
        self.embed_dim = config.hidden_size
        # 创建自注意力层
        self.self_attn = TFCLIPAttention(config, name="self_attn")
        # 创建第一个层规范化层
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm1")
        # 创建多层感知机
        self.mlp = TFCLIPMLP(config, name="mlp")
        # 创建第二个层规范化层
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm2")

    # 前向传播方法，接受隐藏状态张量、注意力掩码、因果注意力掩码、是否输出注意力矩阵、是否处于训练模式，返回变换后的隐藏状态张量
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
            hidden_states (`tf.Tensor`): 输入到层的形状为 `(batch, seq_len, embed_dim)` 的张量
            attention_mask (`tf.Tensor`): 大小为 `(batch, 1, tgt_len, src_len)` 的注意力掩码，
                其中填充元素由非常大的负值指示。
            causal_attention_mask (`tf.Tensor`): 大小为 `(batch, 1, tgt_len, src_len)` 的因果注意力掩码，
                其中填充元素由非常大的负值指示。
            output_attentions (`bool`):
                是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回的张量下的 `outputs`。
        """
        residual = hidden_states  # 保存输入的残差连接

        hidden_states = self.layer_norm1(inputs=hidden_states)  # LayerNorm 层
        attention_outputs = self.self_attn(  # 自注意力层
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            training=training,
        )
        hidden_states = attention_outputs[0]  # 自注意力输出

        hidden_states = residual + hidden_states  # 残差连接

        residual = hidden_states  # 保存自注意力输出的残差连接

        hidden_states = self.layer_norm2(inputs=hidden_states)  # LayerNorm 层
        hidden_states = self.mlp(hidden_states=hidden_states)  # 多层感知机
        hidden_states = residual + hidden_states  # 残差连接

        outputs = (hidden_states,) + attention_outputs[1:]  # 如果输出注意力，则添加注意力张量

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "self_attn", None) is not None:
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)  # 构建自注意力层
        if getattr(self, "layer_norm1", None) is not None:
            with tf.name_scope(self.layer_norm1.name):
                self.layer_norm1.build([None, None, self.embed_dim])  # 构建第一个 LayerNorm 层
        if getattr(self, "mlp", None) is not None:
            with tf.name_scope(self.mlp.name):
                self.mlp.build(None)  # 构建多层感知机
        if getattr(self, "layer_norm2", None) is not None:
            with tf.name_scope(self.layer_norm2.name):
                self.layer_norm2.build([None, None, self.embed_dim])  # 构建第二个 LayerNorm 层
class TFCLIPEncoder(tf.keras.layers.Layer):
    """
    Transformer 编码器，由 `config.num_hidden_layers` 个自注意力层组成。每个层都是一个 [`TFCLIPEncoderLayer`]。

    Args:
        config: CLIPConfig
    """

    def __init__(self, config: CLIPConfig, **kwargs):
        super().__init__(**kwargs)

        # 初始化编码器层列表
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
        # 如果需要输出隐藏状态，则初始化全部隐藏状态
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力权重，则初始化全部注意力权重
        all_attentions = () if output_attentions else None

        # 遍历每个编码器层
        for i, layer_module in enumerate(self.layers):
            # 如果需要输出隐藏状态，则保存当前隐藏状态
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 调用编码器层，计算输出
            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
                output_attentions=output_attentions,
                training=training,
            )
            # 更新隐藏状态为编码器层的输出
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力权重，则保存当前层的注意力权重
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # 添加最后一层的隐藏状态到全部隐藏状态中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典，则将结果组装成元组返回
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        # 返回 TFBaseModelOutput 类的实例，包含最后的隐藏状态、全部隐藏状态和全部注意力权重
        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )

    def build(self, input_shape=None):
        # 如果已经构建过了，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 构建每个编码器层
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)


class TFCLIPTextTransformer(tf.keras.layers.Layer):
    def __init__(self, config: CLIPTextConfig, **kwargs):
        super().__init__(**kwargs)

        # CLIP 文本嵌入层
        self.embeddings = TFCLIPTextEmbeddings(config, name="embeddings")
        # CLIP 编码器
        self.encoder = TFCLIPEncoder(config, name="encoder")
        # 最终的层归一化
        self.final_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="final_layer_norm"
        )

        # 用于计算 `pooled_output`
        self.eos_token_id = config.eos_token_id
        self.embed_dim = config.hidden_size
    # 定义一个方法，接受以下参数：
    # - input_ids: TFModelInputType 类型的参数，表示输入的标识符
    # - attention_mask: tf.Tensor 类型的参数，表示注意力掩码
    # - position_ids: tf.Tensor 类型的参数，表示位置标识符
    # - output_attentions: bool 类型的参数，表示是否输出注意力权重
    # - output_hidden_states: bool 类型的参数，表示是否输出隐藏状态
    # - return_dict: bool 类型的参数，表示是否返回字典
    # - training: bool 类型的参数，默认为 False，表示是否处于训练模式
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        # 获取输入张量的形状
        input_shape = shape_list(input_ids)

        # 使用嵌入层处理输入的 token ids 和位置 ids
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        batch_size, seq_length = input_shape
        # CLIP 的文本模型使用因果注意力掩码，这里准备它
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self._build_causal_attention_mask(batch_size, seq_length, dtype=embedding_output.dtype)

        # 检查注意力掩码并反转
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        attention_mask = _expand_mask(attention_mask)

        # 将嵌入输出传入编码器
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
        # 应用最终的层归一化
        sequence_output = self.final_layer_norm(inputs=sequence_output)

        if self.eos_token_id == 2:
            # 在 PR #24773 之前，`eos_token_id` 是不正确的：让我们保留这里的更改
            # 具有这种 `eos_token_id` 的 CLIP 模型在配置中无法正确处理额外添加的新标记
            # ------------------------------------------------------------
            # text_embeds.shape = [batch_size, n_ctx, transformer.width]
            # 从 eot 嵌入中获取特征（eot_token 是每个序列中的最高数字）
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

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return TFBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    # 构建因果注意力掩码，用于在自注意力机制中屏蔽未来信息
    def _build_causal_attention_mask(self, batch_size, seq_length, dtype=tf.float32):
        # 如果 seq_length 是运行时值，tf.constant 不支持，使用 tf.fill 处理动态形状
        diag = tf.cast(tf.fill((seq_length,), 0.0), dtype)

        # 创建一个二维加性注意力掩码，所有位置都被屏蔽
        to_mask = tf.cast(tf.fill((seq_length, seq_length), -10000.0), dtype)

        # 将对角线和下三角部分设置为0（即不屏蔽的位置）
        # 提示：将二维矩阵视为（查询序列，键序列）的空间
        to_mask = tf.linalg.band_part(to_mask, 0, -1)
        # to_mask = tf.linalg.band_part(to_mask, -1, 0)
        to_mask = tf.linalg.set_diag(to_mask, diagonal=diag)

        return tf.broadcast_to(input=to_mask, shape=(batch_size, 1, seq_length, seq_length))

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
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
                self.final_layer_norm.build([None, None, self.embed_dim])
# 使用 keras_serializable 装饰器将类声明为 Keras 序列化的类
@keras_serializable
class TFCLIPTextMainLayer(tf.keras.layers.Layer):
    # 指定配置类为 CLIPTextConfig
    config_class = CLIPTextConfig

    # 初始化方法，接受 CLIPTextConfig 类型的配置参数
    def __init__(self, config: CLIPTextConfig, **kwargs):
        super().__init__(**kwargs)
        # 存储配置参数
        self.config = config
        # 创建文本模型对象
        self.text_model = TFCLIPTextTransformer(config, name="text_model")

    # 获取输入嵌入的方法
    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        # 返回文本模型的嵌入层
        return self.text_model.embeddings

    # 设置输入嵌入的方法
    def set_input_embeddings(self, value: tf.Variable):
        # 设置文本模型的嵌入权重
        self.text_model.embeddings.weight = value
        # 设置文本模型的词汇量大小
        self.text_model.embeddings.vocab_size = shape_list(value)[0]

    # 前向传播方法
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
        # 如果没有传入 input_ids，则抛出异常
        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        # 获取 input_ids 的形状
        input_shape = shape_list(input_ids)

        # 如果没有传入 attention_mask，则使用全 1 的 mask
        if attention_mask is None:
            attention_mask = tf.fill(dims=input_shape, value=1)

        # 调用文本模型的前向传播方法，获取模型输出
        text_model_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 返回模型输出
        return text_model_outputs

    # 构建方法，用于构建模型
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果存在文本模型，则构建文本模型
        if getattr(self, "text_model", None) is not None:
            with tf.name_scope(self.text_model.name):
                self.text_model.build(None)


# 定义 TFCLIPVisionTransformer 类
class TFCLIPVisionTransformer(tf.keras.layers.Layer):
    # 初始化方法，接受 CLIPVisionConfig 类型的配置参数
    def __init__(self, config: CLIPVisionConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建视觉嵌入对象
        self.embeddings = TFCLIPVisionEmbeddings(config, name="embeddings")
        # 创建预层归一化层
        self.pre_layernorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="pre_layrnorm")
        # 创建编码器对象
        self.encoder = TFCLIPEncoder(config, name="encoder")
        # 创建后层归一化层
        self.post_layernorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="post_layernorm")
        # 存储嵌入维度大小
        self.embed_dim = config.hidden_size

    # 前向传播方法
    def call(
        self,
        pixel_values: TFModelInputType,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    # 定义一个方法，该方法返回 TFBaseModelOutputWithPooling 类型或者元组（包含 TFBaseModelOutputWithPooling 类型的元素和 tf.Tensor 类型的元素）
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        # 使用输入像素值进行嵌入操作，得到嵌入输出
        embedding_output = self.embeddings(pixel_values=pixel_values)
        # 对嵌入输出进行预层归一化
        embedding_output = self.pre_layernorm(inputs=embedding_output)

        # 使用编码器对嵌入输出进行编码
        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=None,
            causal_attention_mask=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 获取编码器输出的序列输出
        sequence_output = encoder_outputs[0]
        # 从序列输出中提取池化输出
        pooled_output = sequence_output[:, 0, :]
        # 对池化输出进行后层归一化
        pooled_output = self.post_layernorm(inputs=pooled_output)

        # 如果不返回字典，则返回编码器输出的序列输出、池化输出以及编码器的其他隐藏状态
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        # 返回 TFBaseModelOutputWithPooling 对象，其中包含序列输出、池化输出、编码器的隐藏状态以及注意力权重
        return TFBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    # 构建方法，用于构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 构建嵌入层
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        # 构建预层归一化层
        if getattr(self, "pre_layernorm", None) is not None:
            with tf.name_scope(self.pre_layernorm.name):
                self.pre_layernorm.build([None, None, self.embed_dim])
        # 构建编码器
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 构建后层归一化层
        if getattr(self, "post_layernorm", None) is not None:
            with tf.name_scope(self.post_layernorm.name):
                self.post_layernorm.build([None, self.embed_dim])
# 声明一个 Keras 可序列化的类装饰器，用于将类装饰为可序列化的类
@keras_serializable
# 定义一个 TFCLIP 视觉主层，继承自 Keras 层
class TFCLIPVisionMainLayer(tf.keras.layers.Layer):
    # 配置类为 CLIP 视觉配置类
    config_class = CLIPVisionConfig

    # 初始化方法，接受一个 CLIP 视觉配置对象和其他关键字参数
    def __init__(self, config: CLIPVisionConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 将传入的配置对象保存为属性
        self.config = config
        # 创建一个 CLIP 视觉变换器模型，命名为 "vision_model"
        self.vision_model = TFCLIPVisionTransformer(config, name="vision_model")

    # 获取输入嵌入的方法，返回视觉模型的嵌入层
    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        return self.vision_model.embeddings

    # 调用方法，接受各种输入参数，返回 TFBaseModelOutputWithPooling 对象或者包含张量的元组
    @unpack_inputs
    def call(
        self,
        pixel_values: TFModelInputType | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        # 如果像素值为 None，则引发 ValueError 异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 调用视觉模型的 call 方法，传入像素值和其他参数，获取视觉模型的输出
        vision_model_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 返回视觉模型的输出
        return vision_model_outputs

    # 构建方法，用于构建层，根据输入形状构建视觉模型
    def build(self, input_shape=None):
        # 如果已经构建，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在视觉模型，则使用其名称作为命名空间构建视觉模型
        if getattr(self, "vision_model", None) is not None:
            with tf.name_scope(self.vision_model.name):
                self.vision_model.build(None)


# 声明一个 Keras 可序列化的类装饰器，用于将类装饰为可序列化的类
@keras_serializable
# 定义一个 TFCLIP 主层，继承自 Keras 层
class TFCLIPMainLayer(tf.keras.layers.Layer):
    # 配置类为 CLIP 配置类
    config_class = CLIPConfig
    # 初始化方法，接受配置参数和其他关键字参数
    def __init__(self, config: CLIPConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 检查文本配置是否为 CLIPTextConfig 类型，如果不是则抛出数值错误
        if not isinstance(config.text_config, CLIPTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type CLIPTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        # 检查视觉配置是否为 CLIPVisionConfig 类型，如果不是则抛出数值错误
        if not isinstance(config.vision_config, CLIPVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type CLIPVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        # 将配置参数保存到实例变量中
        self.config = config

        # 获取文本配置和视觉配置
        text_config = config.text_config
        vision_config = config.vision_config

        # 设置投影维度
        self.projection_dim = config.projection_dim

        # 创建文本模型和视觉模型
        self.text_model = TFCLIPTextTransformer(text_config, name="text_model")
        self.vision_model = TFCLIPVisionTransformer(vision_config, name="vision_model")

        # 创建视觉投影层
        self.visual_projection = tf.keras.layers.Dense(
            units=self.projection_dim,
            kernel_initializer=get_initializer(vision_config.hidden_size**-0.5 * self.config.initializer_factor),
            use_bias=False,
            name="visual_projection",
        )

        # 创建文本投影层
        self.text_projection = tf.keras.layers.Dense(
            units=self.projection_dim,
            kernel_initializer=get_initializer(text_config.hidden_size**-0.5 * self.config.initializer_factor),
            use_bias=False,
            name="text_projection",
        )
        # 设置文本嵌入维度和视觉嵌入维度
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

    # 构建方法，接受输入形状参数
    def build(self, input_shape: tf.TensorShape = None):
        # 添加权重 logit_scale
        self.logit_scale = self.add_weight(
            shape=(1,),
            initializer=tf.keras.initializers.Constant(self.config.logit_scale_init_value),
            trainable=True,
            name="logit_scale",
        )

        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果���在文本模型，则构建文本模型
        if getattr(self, "text_model", None) is not None:
            with tf.name_scope(self.text_model.name):
                self.text_model.build(None)
        # 如果存在视觉模型，则构建视觉模型
        if getattr(self, "vision_model", None) is not None:
            with tf.name_scope(self.vision_model.name):
                self.vision_model.build(None)
        # 如果存在视觉投影层，则构建视觉投影层
        if getattr(self, "visual_projection", None) is not None:
            with tf.name_scope(self.visual_projection.name):
                self.visual_projection.build([None, None, self.vision_embed_dim])
        # 如果存在文本投影层，则构建文本投影层
        if getattr(self, "text_projection", None) is not None:
            with tf.name_scope(self.text_projection.name):
                self.text_projection.build([None, None, self.text_embed_dim])

    # 解包输入
    @unpack_inputs
    # 获取文本特征的方法
    def get_text_features(
        self,
        input_ids: TFModelInputType | None = None,  # 输入的 token ID
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力掩码
        position_ids: np.ndarray | tf.Tensor | None = None,  # 位置 ID
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典形式结果
        training: bool = False,  # 是否处于训练模式
    ) -> tf.Tensor:  # 返回值为张量
        # 如果输入的 token ID 为空，则抛出 ValueError 异常
        if input_ids is None:
            raise ValueError("You have to specify either input_ids")

        # 获取输入 token ID 的形状
        input_shape = shape_list(input_ids)

        # 如果注意力掩码为空，则创建一个形状与输入 token ID 相同的掩码，用值 1 填充
        if attention_mask is None:
            attention_mask = tf.fill(dims=input_shape, value=1)

        # 调用文本模型，传入输入 token ID、注意力掩码等参数，获取文本输出
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 从文本输出中获取池化后的输出
        pooled_output = text_outputs[1]
        # 将池化输出传入文本投影层，得到文本特征
        text_features = self.text_projection(inputs=pooled_output)

        # 返回文本特征
        return text_features

    # 获取图像特征的方法，使用装饰器解包输入参数
    @unpack_inputs
    def get_image_features(
        self,
        pixel_values: TFModelInputType | None = None,  # 像素值
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典形式结果
        training: bool = False,  # 是否处于训练模式
    ) -> tf.Tensor:  # 返回值为张量
        # 如果像素值为空，则抛出 ValueError 异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 调用图像模型，传入像素值等参数，获取图像输出
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 从图像输出中获取池化后的输出
        pooled_output = vision_outputs[1]
        # 将池化输出传入视觉投影层，得到图像特征
        image_features = self.visual_projection(inputs=pooled_output)

        # 返回图像特征
        return image_features

    # 调用方法，同时接收文本和图像输入，返回结果
    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 文本输入的 token ID
        pixel_values: TFModelInputType | None = None,  # 图像输入的像素值
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力掩码
        position_ids: np.ndarray | tf.Tensor | None = None,  # 位置 ID
        return_loss: Optional[bool] = None,  # 是否返回损失
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典形式结果
        training: bool = False,  # 是否处于训练模式
    # 定义函数，接受输入参数并返回 TFCLIPOutput 或 Tuple[tf.Tensor] 类型的结果
    def call(
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=None,
        return_loss=None,
    ):
        # 如果 input_ids 为 None，则抛出数值错误
        if input_ids is None:
            raise ValueError("You have to specify either input_ids")
        # 如果 pixel_values 为 None，则抛出数值错误
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 获取 input_ids 的形状
        input_shape = shape_list(input_ids)

        # 如果 attention_mask 为 None，则用值为 1 填充形状为 input_shape 的张量
        if attention_mask is None:
            attention_mask = tf.fill(dims=input_shape, value=1)

        # 使用 vision_model 处理像素值，返回视觉输出
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 使用 text_model 处理输入文本，返回文本输出
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 获取图像嵌入
        image_embeds = vision_outputs[1]
        # 使用 visual_projection 处理图像嵌入
        image_embeds = self.visual_projection(inputs=image_embeds)

        # 获取文本嵌入
        text_embeds = text_outputs[1]
        # 使用 text_projection 处理文本嵌入
        text_embeds = self.text_projection(inputs=text_embeds)

        # 对图像嵌入进行归一化处理
        image_embeds = image_embeds / tf.norm(tensor=image_embeds, ord="euclidean", axis=-1, keepdims=True)
        # 对文本嵌入进行归一化处理
        text_embeds = text_embeds / tf.norm(tensor=text_embeds, ord="euclidean", axis=-1, keepdims=True)

        # 计算余弦相似度作为 logits
        logit_scale = tf.math.exp(self.logit_scale)
        logits_per_text = tf.matmul(text_embeds, image_embeds, transpose_b=True) * logit_scale
        logits_per_image = tf.transpose(logits_per_text)

        # 初始化损失为 None
        loss = None
        # 如果需要返回损失
        if return_loss:
            # 计算 clip_loss
            loss = clip_loss(logits_per_text)
            # 将损失重塑为形状为 (1,) 的张量
            loss = tf.reshape(loss, (1,))

        # 如果不需要返回字典
        if not return_dict:
            # 构建输出元组
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            # 如果损失不为 None，则将损失添加到输出元组中
            return (loss,) + output if loss is not None else output

        # 返回 TFCLIPOutput 对象
        return TFCLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )
class TFCLIPPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 定义 TFCLIPPreTrainedModel 类，用于处理权重初始化和预训练模型的下载和加载

    config_class = CLIPConfig
    # 设置 config_class 属性为 CLIPConfig 类

    base_model_prefix = "clip"
    # 设置 base_model_prefix 属性为 "clip"

    _keys_to_ignore_on_load_missing = [r"position_ids"]
    # 设置 _keys_to_ignore_on_load_missing 属性为 ["position_ids"]

    _keys_to_ignore_on_load_unexpected = [r"position_ids"]
    # 设置 _keys_to_ignore_on_load_unexpected 属性为 ["position_ids"]


CLIP_START_DOCSTRING = r"""

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

    - a single Tensor with `input_ids` only and nothing else: `model(input_ids)`
    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
    `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
    - a dictionary with one or several input Tensors associated to the input names given in the docstring:
    `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    Note that when creating models and layers with
    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don't need to worry
    about any of this, as you can just pass inputs like you would to any other Python function!

    </Tip>

    Args:
        config ([`CLIPConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义 CLIP_START_DOCSTRING 字符串，包含模型的继承信息、使用提示和参数说明

CLIP_TEXT_INPUTS_DOCSTRING = r"""

# 定义 CLIP_TEXT_INPUTS_DOCSTRING 字符串，用于文档说明
    # 定义函数参数说明
    Args:
        # 输入序列标记的索引，可以是 np.ndarray、tf.Tensor、List[tf.Tensor]、Dict[str, tf.Tensor] 或 Dict[str, np.ndarray] 类型，
        # 每个示例必须具有形状为({0})的形状
        input_ids (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `({0})`):
            Indices of input sequence tokens in the vocabulary.
            
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.__call__`] and
            [`PreTrainedTokenizer.encode`] for details.
            
            [What are input IDs?](../glossary#input-ids)
        # 避免在填充标记索引上执行注意力的掩码。掩码值选择在[0, 1]之间：
        # - 1表示**未屏蔽**的标记，
        # - 0表示**已屏蔽**的标记。
        attention_mask (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            
            [What are attention masks?](../glossary#attention-mask)
        # 输入序列标记在位置嵌入中的每个位置的索引。选择范围为[0, config.max_position_embeddings - 1]。
        position_ids (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.
            
            [What are position IDs?](../glossary#position-ids)
        # 是否返回所有注意力层的注意力张量
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
            config will be used instead.
        # 是否返回所有层的隐藏状态
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        # 是否返回一个`~utils.ModelOutput`而不是一个普通元组
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in
            eager mode, in graph mode the value will always be set to True.
        # 是否在训练模式下使用模型
        training (`bool`, *optional*, defaults to `False``):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
CLIP_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`CLIPImageProcessor.__call__`] for details. output_attentions (`bool`, *optional*): Whether or not to
            return the attentions tensors of all attention layers. See `attentions` under returned tensors for more
            detail. This argument can be used only in eager mode, in graph mode the value in the config will be used
            instead.
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
"""
    Args:
        input_ids (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `({0})`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.__call__`] and
            [`PreTrainedTokenizer.encode`] for details.
            [What are input IDs?](../glossary#input-ids)
        
        pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` `Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`CLIPImageProcessor.__call__`] for details.
        
        attention_mask (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        
        position_ids (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
            [What are position IDs?](../glossary#position-ids)
        
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.
        
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
        
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
        
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in eager mode, in graph mode the value will always be set to True.
        
        training (`bool`, *optional*, defaults to `False``):
            Whether or not to use the model in training mode (some modules like dropout modules have different behaviors between training and evaluation).
# 导入所需模块或库
from transformers import AutoTokenizer, TFCLIPTextModel

# 定义一个 TFCLIPTextModel 类，继承自 TFCLIPPreTrainedModel 类
class TFCLIPTextModel(TFCLIPPreTrainedModel):
    # 设定配置类为 CLIPTextConfig
    config_class = CLIPTextConfig

    # 初始化方法
    def __init__(self, config: CLIPTextConfig, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 实例化 TFCLIPTextMainLayer 类，并赋值给 self.clip 属性
        self.clip = TFCLIPTextMainLayer(config, name="clip")

    # 调用方法
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
        r"""
        返回模型输出结果
        """
        # 调用 self.clip 的 call 方法，传入参数，并将结果赋值给 outputs
        outputs = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 返回 outputs
        return outputs

    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建完毕，则直接返回
        if self.built:
            return
        # 标记模型已构建
        self.built = True
        # 如果 self.clip 存在
        if getattr(self, "clip", None) is not None:
            # 在 self.clip 的命名作用域下，构建 self.clip
            with tf.name_scope(self.clip.name):
                self.clip.build(None)


# 定义一个 TFCLIPVisionModel 类，继承自 TFCLIPPreTrainedModel 类
class TFCLIPVisionModel(TFCLIPPreTrainedModel):
    # 设定配置类为 CLIPVisionConfig
    config_class = CLIPVisionConfig
    # 设置主要输入名称为 "pixel_values"
    main_input_name = "pixel_values"

    # 初始化方法
    def __init__(self, config: CLIPVisionConfig, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 实例化 TFCLIPVisionMainLayer 类，并赋值给 self.clip 属性
        self.clip = TFCLIPVisionMainLayer(config, name="clip")

    # 调用方法
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
        ```py"""

        # 调用 CLIP 模型进行推理
        outputs = self.clip(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 返回 CLIP 模型的输出
        return outputs

    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果模型中包含 CLIP 模型
        if getattr(self, "clip", None) is not None:
            # 在命名空间下构建 CLIP 模型
            with tf.name_scope(self.clip.name):
                self.clip.build(None)
# 使用装饰器为模型类添加文档字符串，起始部分
@add_start_docstrings(CLIP_START_DOCSTRING)
class TFCLIPModel(TFCLIPPreTrainedModel):
    # 配置类为 CLIPConfig
    config_class = CLIPConfig

    # 初始化方法，接收配置和输入参数
    def __init__(self, config: CLIPConfig, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 初始化 CLIP 主层
        self.clip = TFCLIPMainLayer(config, name="clip")

    # 使用装饰器对文档字符串添加前缀，并注解输入和输出
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
        Returns:
            text_features (`tf.Tensor` of shape `(batch_size, output_dim`): The text embeddings obtained by applying
            the projection layer to the pooled output of [`TFCLIPTextModel`].

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, TFCLIPModel

        >>> model = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="tf")
        >>> text_features = model.get_text_features(**inputs)
        ```py"""

        # 调用 CLIP 主层的获取文本特征方法
        text_features = self.clip.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 返回文本特征
        return text_features

    # 使用装饰器对文档字符串添加前缀，并注解输入和输出
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
        ```py"""

        # 使用CLIP模型获取图像特征
        image_features = self.clip.get_image_features(
            pixel_values=pixel_values,  # 图像的像素值
            output_attentions=output_attentions,  # 是否返回注意力权重
            output_hidden_states=output_hidden_states,  # 是否返回隐藏状态
            return_dict=return_dict,  # 是否以字典形式返回结果
        )

        # 返回图像特征
        return image_features

    # 覆盖`call`方法，将其用作模型的前向传播
    @unpack_inputs
    @add_start_docstrings_to_model_forward(CLIP_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFCLIPOutput, config_class=CLIPConfig)
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 输入图像的ID
        pixel_values: TFModelInputType | None = None,  # 图像的像素值
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力掩码
        position_ids: np.ndarray | tf.Tensor | None = None,  # 位置ID
        return_loss: Optional[bool] = None,  # 是否返回损失值
        output_attentions: Optional[bool] = None,  # 是否返回注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否返回隐藏状态
        return_dict: Optional[bool] = None,  # 是否以字典形式返回结果
        training: bool = False,  # 是否处于训练模式
    ) -> Union[TFCLIPOutput, Tuple[tf.Tensor]]:
        r"""
        返回类型为 TFCLIPOutput 或 tf.Tensor 的元组

        示例：

        ```python
        >>> import tensorflow as tf
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFCLIPModel

        >>> model = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="tf", padding=True
        ... )

        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = tf.nn.softmax(logits_per_image, axis=1)  # we can take the softmax to get the label probabilities
        ```py"""

        # 调用 CLIP 模型进行推理
        outputs = self.clip(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_loss=return_loss,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return outputs

    # 用于在 serving 时输出结果
    def serving_output(self, output: TFCLIPOutput) -> TFCLIPOutput:
        # TODO: 目前由于 TensorFlow 无法跟踪嵌套的 dataclass，使用 saved_model=True 时会失败。
        # 参考：https://github.com/huggingface/transformers/pull/16886
        return output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果 clip 属性存在，则构建 clip 模型
        if getattr(self, "clip", None) is not None:
            with tf.name_scope(self.clip.name):
                self.clip.build(None)
```