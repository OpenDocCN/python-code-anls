# `.\transformers\models\blip\modeling_tf_blip.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 Salesforce Team 作者和 HuggingFace Team 所有
# 根据 Apache 许可证 2.0 版本授权使用本文件
# 只有在遵守许可证的情况下才能使用本文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以了解特定语言的权限和限制
""" TensorFlow BLIP 模型。"""

# 导入必要的库和模块
from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import tensorflow as tf

# 导入相关的模型输出和工具函数
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling
from ...modeling_tf_utils import (
    TFPreTrainedModel,
    get_initializer,
    get_tf_activation,
    keras_serializable,
    shape_list,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, stable_softmax
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 导入 BLIP 模型的配置类
from .configuration_blip import BlipConfig, BlipTextConfig, BlipVisionConfig
# 导入 BLIP 文本模型相关的类和函数
from .modeling_tf_blip_text import BLIP_TEXT_INPUTS_DOCSTRING, TFBlipTextLMHeadModel, TFBlipTextModel

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点
_CHECKPOINT_FOR_DOC = "Salesforce/blip-vqa-base"

# 预训练的 BLIP 模型列表
TF_BLIP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Salesforce/blip-vqa-base",
    "Salesforce/blip-vqa-capfilt-large",
    "Salesforce/blip-image-captioning-base",
    "Salesforce/blip-image-captioning-large",
    "Salesforce/blip-itm-base-coco",
    "Salesforce/blip-itm-large-coco",
    "Salesforce/blip-itm-base-flickr",
    "Salesforce/blip-itm-large-flickr",
    # 查看所有 BLIP 模型：https://huggingface.co/models?filter=blip
]

# 对比损失函数
def contrastive_loss(logits: tf.Tensor) -> tf.Tensor:
    return tf.math.reduce_mean(
        tf.keras.metrics.sparse_categorical_crossentropy(
            y_true=tf.range(shape_list(logits)[0]), y_pred=logits, from_logits=True
        )
    )

# BLIP 损失函数
def blip_loss(similarity: tf.Tensor) -> tf.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(tf.transpose(similarity))
    return (caption_loss + image_loss) / 2.0

# BLIP 生成模型的输出类
@dataclass
class TFBlipForConditionalGenerationModelOutput(ModelOutput):
    """
    从包含图像嵌入的最后隐藏状态的池化的视觉模型输出的基类适应的类。此类还添加了来自文本解码器的损失项。
    Args:
        loss (`tf.Tensor`, *optional*, returned when `labels` is provided, `tf.Tensor` of shape `(1,)`):
            Languge modeling loss from the text decoder. # 文本解码器的语言建模损失
        logits (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`, *optional*):
            Prediction scores of the language modeling head of the text decoder model. # 文本解码器模型的语言建模头部的预测分数
        image_embeds (`tf.Tensor` of shape `(batch_size, output_dim)`, *optional*):
            The image embeddings obtained after applying the Vision Transformer model to the input image. # 将 Vision Transformer 模型应用于输入图像后得到的图像嵌入
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the model. # 模型最后一层输出的隐藏状态序列
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings, if the model has an embedding layer, + one for
            the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs. # 每层模型输出的隐藏状态元组，还包括可选的初始嵌入输出
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads. # 在经过注意力 softmax 后的注意力权重，用于计算自注意力头部的加权平均值
    """

    loss: Tuple[tf.Tensor] | None = None  # 损失值，默认为 None
    logits: Tuple[tf.Tensor] | None = None  # 预测分数，默认为 None
    image_embeds: tf.Tensor | None = None  # 图像嵌入，默认为 None
    last_hidden_state: tf.Tensor = None  # 最后隐藏状态，默认为 None
    hidden_states: Tuple[tf.Tensor] | None = None  # 隐藏状态元组，默认为 None
    attentions: Tuple[tf.Tensor] | None = None  # 注意力权重元组，默认为 None

    @property
    def decoder_logits(self):
        warnings.warn(
            "`decoder_logits` attribute is deprecated and will be removed in version 5 of Transformers."
            " Please use the `logits` attribute to retrieve the final output instead.",
            FutureWarning,
        )
        return self.logits  # 返回 logits 属性作为最终输出，警告该属性已经过时并将在 Transformers 的第五个版本中移除
# TFBlipTextVisionModelOutput 类，继承自 ModelOutput 类，用于表示文本视觉模型的输出结果
@dataclass
class TFBlipTextVisionModelOutput(ModelOutput):
    """
    从包含图像嵌入的池化最后隐藏状态的基类适应的视觉模型输出类。此类还添加了来自文本解码器的损失项。

    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            文本解码器的语言建模损失。
        image_embeds (`tf.Tensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            通过将投影层应用于 pooler_output 获得的图像嵌入。
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的隐藏状态的序列。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            `tf.Tensor` 元组（如果模型具有嵌入层，则为一个 + 每个层的输出一个）的形状为 `(batch_size, sequence_length, hidden_size)`。

            模型每一层的隐藏状态，以及可选的初始嵌入输出。
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            `tf.Tensor` 元组（每个层一个）的形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    # 损失项，默认为 None
    loss: tf.Tensor | None = None
    # 图像嵌入，默认为 None
    image_embeds: tf.Tensor | None = None
    # 最后隐藏状态，不可选
    last_hidden_state: tf.Tensor = None
    # 隐藏状态，默认为 None
    hidden_states: Tuple[tf.Tensor] | None = None
    # 注意力，默认为 None
    attentions: Tuple[tf.Tensor] | None = None


@dataclass
class TFBlipImageTextMatchingModelOutput(ModelOutput):
    """
    Adapted from the base class for vision model's outputs that also contains image embeddings of the pooling of the
    last hidden states. This class also adds the loss term from the text decoder as well as the image-text similarity
    scores.
    """
    Args:
        itm_score (`tf.Tensor`):
            The image-text similarity scores.
            图像-文本相似度分数。
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Languge modeling loss from the text decoder.
            文本解码器产生的语言建模损失。
        image_embeds (`tf.Tensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
            通过将投影层应用于pooler_output获得的图像嵌入。
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
            模型最后一层输出的隐藏状态序列。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings, if the model has an embedding layer, + one for
            the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
            模型在每一层输出的隐藏状态，加上可选的初始嵌入输出。
        vision_pooler_output (`tf.Tensor` of shape `(batch_size, hidden_size)`, *optional*):
            Last layer hidden-state of the vision of the vision-only branch of the model.
            模型视觉-仅分支的最后一层隐藏状态。
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
            注意力softmax后的注意力权重，用于计算自注意力头中的加权平均值。
        question_embeds (`tf.Tensor`):
            The question embeddings obtained by the text projection layer.
            通过文本投影层获得的问题嵌入。
    """

    # 初始化为None
    itm_score: tf.Tensor | None = None
    loss: tf.Tensor | None = None
    image_embeds: tf.Tensor | None = None
    last_hidden_state: tf.Tensor | None = None
    hidden_states: Tuple[tf.Tensor] | None = None
    vision_pooler_output: tf.Tensor | None = None
    attentions: Tuple[tf.Tensor] | None = None
    question_embeds: Tuple[tf.Tensor] | None = None
from dataclasses import dataclass
from typing import Any, Tuple
import tensorflow as tf

# 定义一个名为TFBlipOutput的数据类，继承自ModelOutput
@dataclass
class TFBlipOutput(ModelOutput):
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
            The text embeddings obtained by applying the projection layer to the pooled output of [`BlipTextModel`].
        image_embeds(`tf.Tensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of [`BlipVisionModel`].
        text_model_output(`BaseModelOutputWithPooling`):
            The output of the [`BlipTextModel`].
        vision_model_output(`BaseModelOutputWithPooling`):
            The output of the [`BlipVisionModel`].
    """

    # 定义数据类的属性
    loss: tf.Tensor | None = None  # 图像-文本相似性的对比损失
    logits_per_image: tf.Tensor = None  # 图像嵌入和文本嵌入之间的点积得分，代表图像-文本相似性得分
    logits_per_text: tf.Tensor = None  # 文本嵌入和图像嵌入之间的点积得分，代表文本-图像相似性得分
    text_embeds: tf.Tensor = None  # 通过对`BlipTextModel`的池化输出应用投影层获得的文本嵌入
    image_embeds: tf.Tensor = None  # 通过对`BlipVisionModel`的池化输出应用投影层获得的图像嵌入
    text_model_output: TFBaseModelOutputWithPooling = None  # `BlipTextModel`的输出
    vision_model_output: TFBaseModelOutputWithPooling = None  # `BlipVisionModel`的输出

    # 将对象转换为元组的方法
    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            # 如果属性不是"text_model_output"或"vision_model_output"，则直接返回属性值；否则返回该属性的元组形式
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


# 定义名为TFBlipVisionEmbeddings的类，继承自tf.keras.layers.Layer
class TFBlipVisionEmbeddings(tf.keras.layers.Layer):
    # 初始化方法
    def __init__(self, config: BlipVisionConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config  # 存储BlipVisionConfig实例
        self.embed_dim = config.hidden_size  # 嵌入维度为BlipVisionConfig的隐藏大小
        self.image_size = config.image_size  # 图像大小为BlipVisionConfig的图像大小
        self.patch_size = config.patch_size  # 补丁大小为BlipVisionConfig的补丁大小

        # 定义补丁嵌入层
        self.patch_embedding = tf.keras.layers.Conv2D(
            filters=self.embed_dim,  # 卷积滤波器数等于嵌入维度
            kernel_size=self.patch_size,  # 卷积核大小等于补丁大小
            strides=self.patch_size,  # 卷积步幅等于补丁大小
            kernel_initializer=get_initializer(self.config.initializer_range),  # 使用BlipVisionConfig的初始化范围初始化卷积核
            data_format="channels_last",  # 数据格式为通道在后
            name="patch_embedding",  # 层的名称为patch_embedding
        )

        # 计算图像中的补丁数量
        self.num_patches = (self.image_size // self.patch_size) ** 2
        # 计算位置编码的数量
        self.num_positions = self.num_patches + 1
    # 构建模型，在给定输入形状的情况下
    def build(self, input_shape=None):
        # 添加类别嵌入权重，形状为 (1, 1, 嵌入维度)，可训练
        self.class_embedding = self.add_weight(
            shape=(1, 1, self.embed_dim),
            initializer=get_initializer(self.config.initializer_range),
            trainable=True,
            name="class_embedding",
        )

        # 添加位置嵌入权重，形状为 (1, 位置数目, 嵌入维度)，可训练
        self.position_embedding = self.add_weight(
            shape=(1, self.num_positions, self.embed_dim),
            initializer=get_initializer(self.config.initializer_range),
            trainable=True,
            name="position_embedding",
        )

        # 如果模型已构建，则直接返回
        if self.built:
            return
        # 标记模型已构建
        self.built = True
        # 如果存在 patch_embedding 属性
        if getattr(self, "patch_embedding", None) is not None:
            # 使用 patch_embedding 的命名空间
            with tf.name_scope(self.patch_embedding.name):
                # 构建 patch_embedding
                self.patch_embedding.build([None, None, None, 3])

    # 调用模型，输入像素值的张量，返回嵌入后的张量
    def call(self, pixel_values: tf.Tensor) -> tf.Tensor:
        # 输入通道为第一维，进行转置以匹配模型的要求
        batch_size = tf.shape(pixel_values)[0]
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))
        # 使用 patch_embedding 嵌入像素值
        patch_embeds = self.patch_embedding(pixel_values)
        # 将嵌入后的张量 reshape 为 (batch_size, num_patches, -1)
        patch_embeds = tf.reshape(patch_embeds, (batch_size, self.num_patches, -1))

        # 广播类别嵌入以匹配 batch_size
        class_embeds = tf.broadcast_to(self.class_embedding, (batch_size, 1, self.embed_dim))
        # 将类别嵌入和 patch 嵌入连接起来
        embeddings = tf.concat([class_embeds, patch_embeds], axis=1)
        # 将位置嵌入添加到嵌入张量中
        embeddings = embeddings + self.position_embedding[:, : tf.shape(embeddings)[1], :]
        # 返回嵌入后的张量
        return embeddings
# 从transformers.models.clip.modeling_tf_clip.TFCLIPTextEmbeddings复制并将CLIP->Blip
class TFBlipTextEmbeddings(tf.keras.layers.Layer):
    def __init__(self, config: BlipTextConfig, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = config.hidden_size

        self.config = config

    def build(self, input_shape: tf.TensorShape = None):
        # 在"token_embedding"作用域下构建权重矩阵
        self.weight = self.add_weight(
            shape=(self.config.vocab_size, self.embed_dim),
            initializer=get_initializer(self.config.initializer_factor * self.config.initializer_range),
            trainable=True,
            name="weight",
        )

        # 在"position_embedding"作用域下构建位置编码矩阵
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
        根据输入张量应用嵌入。

        返回:
            final_embeddings (`tf.Tensor`): 输出嵌入张量。
        """
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        input_shape = shape_list(inputs_embeds)[:-1]

        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)

        position_embeds = tf.gather(params=self.position_embedding, indices=position_ids)
        position_embeds = tf.tile(input=position_embeds, multiples=(input_shape[0], 1, 1))
        final_embeddings = inputs_embeds + position_embeds

        return final_embeddings


class TFBlipAttention(tf.keras.layers.Layer):
    """来自'Attention Is All You Need'论文的多头注意力"""
    def __init__(self, config, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 保存传入的配置信息
        self.config = config
        # 设置嵌入维度为配置中的隐藏层大小
        self.embed_dim = config.hidden_size
        # 设置注意力头的数量为配置中的注意力头数量
        self.num_heads = config.num_attention_heads
        # 计算每个头的维度
        self.head_dim = self.embed_dim // self.num_heads
        # 检查嵌入维度是否能被注意力头的数量整除
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        # 设置缩放因子为头维度的负半数
        self.scale = self.head_dim**-0.5
        # 初始化一个丢弃层
        self.dropout = tf.keras.layers.Dropout(config.attention_dropout, name="dropout")

        # 初始化一个全连接层，用于计算查询、键和值
        self.qkv = tf.keras.layers.Dense(
            3 * self.embed_dim, kernel_initializer=get_initializer(config.initializer_range), name="qkv"
        )

        # 初始化一个全连接层，用于投影
        self.projection = tf.keras.layers.Dense(
            self.embed_dim, kernel_initializer=get_initializer(config.initializer_range), name="projection"
        )

    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor | None = None,
        output_attentions: Optional[bool] = False,
        training: Optional[bool] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor | None, Tuple[tf.Tensor] | None]:
        """Input shape: Batch x Time x Channel"""

        # 获取输入张量的形状信息
        bsz, tgt_len, embed_dim = shape_list(hidden_states)

        # 使用全连接层计算混合的查询、键和值
        mixed_qkv = self.qkv(hidden_states)
        # 重新整形混合的查询、键和值张量
        mixed_qkv = tf.reshape(mixed_qkv, (bsz, tgt_len, 3, self.num_heads, self.head_dim))
        # 转置张量维度
        mixed_qkv = tf.transpose(mixed_qkv, perm=(2, 0, 3, 1, 4))

        # 分别提取查询、键和值
        query_states, key_states, value_states = mixed_qkv[0], mixed_qkv[1], mixed_qkv[2]

        # 计算原始注意力分数，即查询和键的点积
        attention_scores = query_states @ tf.transpose(key_states, (0, 1, 3, 2))

        # 缩放注意力分数
        attention_scores = attention_scores * self.scale

        # 将注意力分数归一化为概率
        attention_probs = stable_softmax(attention_scores, axis=-1)

        # 使用丢弃层进行注意力概率的随机丢弃
        attention_probs = self.dropout(attention_probs, training=training)

        # 如果存在头部遮罩，则将注意力概率与头部遮罩相乘
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文向量，即注意力概率与值的乘积
        context_layer = tf.transpose(attention_probs @ value_states, perm=(0, 2, 1, 3))

        # 重新整形上下文向量
        new_context_layer_shape = shape_list(context_layer)[:-2] + [self.embed_dim]
        context_layer = tf.reshape(context_layer, new_context_layer_shape)

        # 使用投影层进行最终的线性映射
        output = self.projection(context_layer)

        # 如果需要输出注意力分数，则返回输出和注意力分数，否则只返回输出
        outputs = (output, attention_probs) if output_attentions else (output, None)

        return outputs
    # 如果模型已经构建，则直接返回，不进行重复构建
    def build(self, input_shape=None):
        if self.built:
            return
        # 设置模型已构建的标志为 True
        self.built = True
        # 如果存在 dropout 层，则构建该层
        if getattr(self, "dropout", None) is not None:
            # 在命名空间中构建 dropout 层
            with tf.name_scope(self.dropout.name):
                # 构建 dropout 层
                self.dropout.build(None)
        # 如果存在 qkv 层，则构建该层
        if getattr(self, "qkv", None) is not None:
            # 在命名空间中构建 qkv 层
            with tf.name_scope(self.qkv.name):
                # 构建 qkv 层，输入形状为 [None, None, self.embed_dim]
                self.qkv.build([None, None, self.embed_dim])
        # 如果存在 projection 层，则构建该层
        if getattr(self, "projection", None) is not None:
            # 在命名空间中构建 projection 层
            with tf.name_scope(self.projection.name):
                # 构建 projection 层，输入形状为 [None, None, self.embed_dim]
                self.projection.build([None, None, self.embed_dim])
class TFBlipMLP(tf.keras.layers.Layer):
    # 初始化方法，接受 BlipConfig 对象作为配置参数
    def __init__(self, config: BlipConfig, **kwargs):
        super().__init__(**kwargs)

        # 根据配置参数获取激活函数
        self.activation_fn = get_tf_activation(config.hidden_act)

        # 计算输入投影标准差
        in_proj_std = (config.hidden_size**-0.5) * ((2 * config.num_hidden_layers) ** -0.5)
        # 计算全连接层标准差
        fc_std = (2 * config.hidden_size) ** -0.5

        # 第一个全连接层，设置隐藏单元数和初始化方法
        self.fc1 = tf.keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(fc_std), name="fc1"
        )
        # 第二个全连接层，设置隐藏单元数和初始化方法
        self.fc2 = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(in_proj_std), name="fc2"
        )
        # 保存配置参数
        self.config = config

    # 前向传播方法，接受隐藏状态作为输入，返回经过前馈神经网络处理后的隐藏状态
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 第一个全连接层，将隐藏状态作为输入
        hidden_states = self.fc1(inputs=hidden_states)
        # 应用激活函数
        hidden_states = self.activation_fn(hidden_states)
        # 第二个全连接层，将前一层的输出作为输入
        hidden_states = self.fc2(inputs=hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states

    # 构建方法，用于构建层的参数
    def build(self, input_shape=None):
        # 如果已经构建，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在第一个全连接层，则构建其参数
        if getattr(self, "fc1", None) is not None:
            with tf.name_scope(self.fc1.name):
                self.fc1.build([None, None, self.config.hidden_size])
        # 如果存在第二个全连接层，则构建其参数
        if getattr(self, "fc2", None) is not None:
            with tf.name_scope(self.fc2.name):
                self.fc2.build([None, None, self.config.intermediate_size])


class TFBlipEncoderLayer(tf.keras.layers.Layer):
    # 初始化方法，接受 BlipConfig 对象作为配置参数
    def __init__(self, config: BlipConfig, **kwargs):
        super().__init__(**kwargs)
        # 获取嵌入维度
        self.embed_dim = config.hidden_size
        # 自注意力机制层
        self.self_attn = TFBlipAttention(config, name="self_attn")
        # 第一个层规范化层
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm1")
        # 多层感知机层
        self.mlp = TFBlipMLP(config, name="mlp")
        # 第二个层规范化层
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm2")

    # 前向传播方法
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        output_attentions: Optional[bool] = False,
        training: Optional[bool] = None,
    ) -> Tuple[tf.Tensor]:
        """
        Args:
            hidden_states (`tf.Tensor`): 输入到层的张量，形状为 `(batch, seq_len, embed_dim)`
            attention_mask (`tf.Tensor`): 注意力掩码，大小为 `(batch, 1, tgt_len, src_len)`，其中填充元素由非常大的负值指示。
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *可选*):
                是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回的张量中的 `attentions`。
        """
        residual = hidden_states

        # 应用第一个层的层归一化
        hidden_states = self.layer_norm1(hidden_states)
        # 应用自注意力机制
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            head_mask=attention_mask,
            output_attentions=output_attentions,
            training=training,
        )
        # 添加残差连接
        hidden_states = hidden_states + residual
        residual = hidden_states
        # 应用第二个层的层归一化
        hidden_states = self.layer_norm2(hidden_states)
        # 应用前向网络
        hidden_states = self.mlp(hidden_states)

        hidden_states = hidden_states + residual

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建自注意力层
        if getattr(self, "self_attn", None) is not None:
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)
        # 构建第一个层的层归一化
        if getattr(self, "layer_norm1", None) is not None:
            with tf.name_scope(self.layer_norm1.name):
                self.layer_norm1.build([None, None, self.embed_dim])
        # 构建前向网络
        if getattr(self, "mlp", None) is not None:
            with tf.name_scope(self.mlp.name):
                self.mlp.build(None)
        # 构建第二个层的层归一化
        if getattr(self, "layer_norm2", None) is not None:
            with tf.name_scope(self.layer_norm2.name):
                self.layer_norm2.build([None, None, self.embed_dim])
class TFBlipPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置配置类为BlipConfig
    config_class = BlipConfig
    # 设置基础模型前缀为"blip"
    base_model_prefix = "blip"
    # 在加载模型时忽略的键列表
    _keys_to_ignore_on_load_missing = [r"position_ids"]


BLIP_START_DOCSTRING = r"""
    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`BlipConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

BLIP_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`BlipImageProcessor`]. See [`BlipImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

BLIP_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            # 输入序列标记在词汇表中的索引。默认情况下，会忽略填充。
            # 可以使用 [`AutoProcessor`] 获取索引。详情请参阅 [`BlipProcessor.__call__`]。
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 避免在填充的标记索引上执行注意力的掩码。
            # 掩码值选择在 `[0, 1]` 之间：
            # - 1 表示**未掩码**的标记，
            # - 0 表示**已掩码**的标记。
            [What are attention masks?](../glossary#attention-mask)
        position_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 每个输入序列标记在位置嵌入中的位置索引。在范围 `[0, config.max_position_embeddings - 1]` 中选择。
            [What are position IDs?](../glossary#position-ids)
        pixel_values (`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):
            # 像素值。默认情况下，会忽略填充。可以使用 [`BlipImageProcessor`] 获取像素值。详情请参阅 [`BlipImageProcessor.__call__`]。
        return_loss (`bool`, *optional*):
            # 是否返回对比损失。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多详细信息，请参阅返回张量下的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多详细信息，请参阅返回张量下的 `hidden_states`。
        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通元组。
```  
"""
Transformer 编码器，由 `config.num_hidden_layers` 个自注意力层组成。每个层都是一个 [`BlipEncoderLayer`]。

参数:
    config (`BlipConfig`):
        `TFBlipEncoder` 的对应视觉配置。

"""
@keras_serializable
class TFBlipEncoder(tf.keras.layers.Layer):
    # 设置配置类为 BlipConfig
    config_class = BlipConfig

    def __init__(self, config: BlipConfig, **kwargs):
        super().__init__(**kwargs)
        # 保存配置
        self.config = config
        # 创建多个 BlipEncoderLayer 层
        self.layers = [TFBlipEncoderLayer(config, name=f"layers_._{i}") for i in range(config.num_hidden_layers)]

    @unpack_inputs
    def call(
        self,
        inputs_embeds,
        attention_mask: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = None,
"""
    ) -> Union[Tuple, TFBaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embedded representation of the inputs. Should be float, not int tokens.
            attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                output_attentions=output_attentions,
                training=training,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)
class TFBlipVisionModel(TFBlipPreTrainedModel):
    # 主输入名称为 "pixel_values"
    main_input_name = "pixel_values"
    # 配置类为 BlipVisionConfig
    config_class = BlipVisionConfig

    def __init__(self, config: BlipVisionConfig, *args, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *args, **kwargs)
        # 保存配置对象
        self.config = config

        # 创建嵌入层对象
        self.embeddings = TFBlipVisionEmbeddings(config, name="embeddings")
        # 创建编码器对象
        self.encoder = TFBlipEncoder(config, name="encoder")
        # 创建后层归一化层对象
        self.post_layernorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="post_layernorm")
        # 隐藏层维度等于隐藏大小
        self.embed_dim = config.hidden_size

    def serving_output(self, output: TFBaseModelOutputWithPooling) -> TFBaseModelOutputWithPooling:
        # 如果配置中输出隐藏状态，则转换隐藏状态为张量
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        # 如果配置中输出注意力，则转换注意力为张量
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        # 返回带池化的模型输出对象
        return TFBaseModelOutputWithPooling(
            last_hidden_state=output.last_hidden_state,
            pooler_output=output.pooler_output,
            hidden_states=hs,
            attentions=attns,
        )

    # 标记函数参数为输入解包
    @unpack_inputs
    # 添加模型前向传播的文档字符串
    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    # 替换返回类型的文档字符串
    @replace_return_docstrings(output_type=TFBaseModelOutputWithPooling, config_class=BlipVisionConfig)
    def call(
        self,
        pixel_values: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = None,
    ) -> Union[Tuple, TFBaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        # 设置是否输出注意力权重，默认与配置一致
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置是否输出隐藏状态，默认与配置一致
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置是否返回字典形式的输出，默认与配置一致
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果未提供像素值，则抛出异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 对像素值进行嵌入
        hidden_states = self.embeddings(pixel_values)

        # 编码器的前向传播
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 获取最后一个隐藏状态
        last_hidden_state = encoder_outputs[0]
        # 对最后一个隐藏状态进行后层归一化
        last_hidden_state = self.post_layernorm(last_hidden_state)

        # 对最后一个隐藏状态进行池化，取第一个位置的值
        pooled_output = last_hidden_state[:, 0, :]
        # 如果输入的秩不同，TensorFlow 会出错，因此插入一个单例维度
        pooled_output = self.post_layernorm(tf.expand_dims(pooled_output, 1))
        pooled_output = tf.squeeze(pooled_output, 1)

        # 如果不返回字典，则返回一个元组
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 返回字典形式的输出
        return TFBaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.embeddings

    # 模型构建
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建嵌入层
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        # 构建编码器
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 构建后层归一化层
        if getattr(self, "post_layernorm", None) is not None:
            with tf.name_scope(self.post_layernorm.name):
                self.post_layernorm.build([None, None, self.embed_dim])
```py  
class TFBlipMainLayer(tf.keras.layers.Layer):
    # 配置类，用于定义Blip的配置参数
    config_class = BlipConfig

    # 初始化函数，接受Blip的配置参数，以及其他参数
    def __init__(self, config: BlipConfig, *args, **kwargs):
        # 调用父类初始化函数
        super().__init__(*args, **kwargs)

        # 检查config.text_config是否为BlipTextConfig类型，如果不是则抛出ValueError
        if not isinstance(config.text_config, BlipTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type BlipTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        # 检查config.vision_config是否为BlipVisionConfig类型，如果不是则抛出ValueError
        if not isinstance(config.vision_config, BlipVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type BlipVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        # 获取text_config和vision_config
        text_config = config.text_config
        vision_config = config.vision_config

        # 设置投影维度，文本嵌入维度和视觉嵌入维度
        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        # 创建文本模型和视觉模型
        self.text_model = TFBlipTextModel(text_config, name="text_model")
        self.vision_model = TFBlipVisionModel(vision_config, name="vision_model")

        # 创建视觉投影层和文本投影层
        self.visual_projection = tf.keras.layers.Dense(
            self.projection_dim,
            use_bias=False,
            kernel_initializer=get_initializer(config.initializer_range),
            name="visual_projection",
        )
        self.text_projection = tf.keras.layers.Dense(
            self.projection_dim,
            use_bias=False,
            kernel_initializer=get_initializer(config.initializer_range),
            name="text_projection",
        )

        # 设置配置参数
        self.config = config

    # 构建函数，用于构建模型
    def build(self, input_shape=None):
        # 添加logit_scale参数，用于对预测值进行缩放
        self.logit_scale = self.add_weight(
            name="logit_scale",
            shape=[],
            initializer=tf.keras.initializers.Constant(self.config.logit_scale_init_value),
            trainable=True,
        )

        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True
        # 构建文本模型、视觉模型、视觉投影层和文本投影层
        if getattr(self, "text_model", None) is not None:
            with tf.name_scope(self.text_model.name):
                self.text_model.build(None)
        if getattr(self, "vision_model", None) is not None:
            with tf.name_scope(self.vision_model.name):
                self.vision_model.build(None)
        if getattr(self, "visual_projection", None) is not None:
            with tf.name_scope(self.visual_projection.name):
                self.visual_projection.build([None, None, self.vision_embed_dim])
        if getattr(self, "text_projection", None) is not None:
            with tf.name_scope(self.text_projection.name):
                self.text_projection.build([None, None, self.text_embed_dim])

    # 解包输入
    @unpack_inputs
    # 定义一个方法，用于调用 BLIP 模型
    def call(
        self,
        input_ids: tf.Tensor | None = None,  # 输入的文本 ID 张量，默认为 None
        pixel_values: tf.Tensor | None = None,  # 输入的像素值张量，默认为 None
        attention_mask: tf.Tensor | None = None,  # 注意力掩码张量，默认为 None
        position_ids: tf.Tensor | None = None,  # 位置 ID 张量，默认为 None
        return_loss: Optional[bool] = None,  # 是否返回损失值，默认为 None
        output_attentions: Optional[bool] = None,  # 是否输出注意力，默认为 None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，默认为 None
        return_dict: Optional[bool] = None,  # 是否返回字典，默认为 None
        training: Optional[bool] = None,  # 是否训练，默认为 None
    ) -> Union[Tuple, TFBlipOutput]:  # 返回值类型为元组或 TFBlipOutput 类型

        # 如果未指定输出注意力，则使用 BLIP 模型的配置中的值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用视觉模型处理像素值
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 使用文本模型处理文本 ID、注意力掩码和位置 ID
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 获取视觉模型的嵌入
        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        # 获取文本模型的嵌入
        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        # 对特征进行归一化处理
        image_embeds = image_embeds / tf.norm(image_embeds, ord=2, axis=-1, keepdims=True)
        text_embeds = text_embeds / tf.norm(text_embeds, ord=2, axis=-1, keepdims=True)

        # 计���余弦相似度作为 logits
        logit_scale = tf.exp(self.logit_scale)
        logits_per_text = tf.matmul(text_embeds, image_embeds, transpose_b=True) * logit_scale
        logits_per_image = tf.transpose(logits_per_text)

        loss = None
        # 如果需要返回损失值
        if return_loss:
            loss = blip_loss(logits_per_text)
            loss = tf.reshape(loss, (1,))

        # 如果不需要返回字典
        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        # 返回 TFBlipOutput 对象
        return TFBlipOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )
#     input_ids: tf.Tensor | None = None, 模型输入的 token IDs 张量或 None
#     pixel_values: tf.Tensor | None = None, 图像的像素值张量或 None
#     attention_mask: tf.Tensor | None = None, 注意力遮罩张量或 None
#     position_ids: tf.Tensor | None = None, 位置 ID 张量或 None
#     return_loss: Optional[bool] = None, 是否返回损失值的可选布尔值，默认为 None
#     output_attentions: Optional[bool] = None, 是否返回注意力权重的可选布尔值，默认为 None
#     output_hidden_states: Optional[bool] = None, 是否返回隐藏状态的可选布尔值，默认为 None
#     return_dict: Optional[bool] = None, 是否返回字典的可选布尔值，默认为 None
#     training: Optional[bool] = None, 是否处于训练模式的可选布尔值，默认为 None
    ) -> tf.Tensor:
        r"""
        返回:
            text_features (`tf.Tensor` of shape `(batch_size, output_dim`): 通过将[`TFBlipTextModel`]的汇总输出应用于投影层获得的文本嵌入。

        示例:

        ```python
        >>> from transformers import AutoProcessor, TFBlipModel

        >>> model = TFBlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

        >>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="tf")
        >>> text_features = model.get_text_features(**inputs)
        ```py"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 BLIP 文本模型获取文本输出
        text_outputs = self.blip.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=return_dict,
        )

        # 获取文本输出的汇总向量
        pooled_output = text_outputs[1]
        # 将汇总输出应用于文本投影层，得到文本特征
        text_features = self.blip.text_projection(pooled_output)

        return text_features

    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values: tf.Tensor | None = None,
        return_dict: Optional[bool] = None,
    ) -> tf.Tensor:
        r"""
        返回:
            image_features (`tf.Tensor` of shape `(batch_size, output_dim`): 通过将[`TFBlipVisionModel`]的汇总输出应用于投影层获得的图像嵌入。

        示例:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFBlipModel

        >>> model = TFBlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="tf")

        >>> image_features = model.get_image_features(**inputs)
        ```py"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 BLIP 视觉模型获取图像输出
        vision_outputs = self.blip.vision_model(pixel_values=pixel_values, return_dict=return_dict)

        # 获取图像输出的汇总向量
        pooled_output = vision_outputs[1]  # pooled_output
        # 将汇总输出应用于视觉投影层，得到图像特征
        image_features = self.blip.visual_projection(pooled_output)

        return image_features

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "blip", None) is not None:
            with tf.name_scope(self.blip.name):
                # 构建 BLIP 模型
                self.blip.build(None)
```  
# 使用装饰器为 BLIP 模型添加文档字符串，描述了模型的结构和用途，以及可选的输入参数和行为
@add_start_docstrings(
    """
    BLIP Model for image captioning. The model consists of a vision encoder and a text decoder. One can optionally pass
    `input_ids` to the model, which serve as a text prompt, to make the text decoder continue the prompt. Otherwise,
    the decoder starts generating text from the [BOS] (beginning-of-sequence) token. will start generating the caption
    from the text input. If no text input is provided, the decoder will start with the [BOS] token only.
    """,
    BLIP_START_DOCSTRING,  # 添加 BLIP 模型的通用文档字符串
)
# 定义 TFBlipForConditionalGeneration 类，继承自 TFBlipPreTrainedModel，用于条件生成任务
class TFBlipForConditionalGeneration(TFBlipPreTrainedModel):
    config_class = BlipConfig  # 设置配置类为 BlipConfig
    _keys_to_ignore_on_load_missing = [r"text_decoder.cls.predictions.decoder.bias"]  # 加载模型时忽略缺失的键
    main_input_name = "pixel_values"  # 主输入的名称为 "pixel_values"

    def __init__(self, config: BlipConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)  # 调用父类的构造函数

        # 创建 BLIP 视觉模型，使用 TFBlipVisionModel 类，传入视觉配置和名称
        self.vision_model = TFBlipVisionModel(config.vision_config, name="vision_model")

        # 创建 BLIP 文本解码器，使用 TFBlipTextLMHeadModel 类，传入文本配置和名称
        self.text_decoder = TFBlipTextLMHeadModel(config.text_config, name="text_decoder")

        # 设置文本解码器的初始输入 ID 为 BOS（开始序列）标记 ID
        self.decoder_input_ids = config.text_config.bos_token_id
        # 设置文本解码器的填充标记 ID
        self.decoder_pad_token_id = config.text_config.pad_token_id

    # 获取输入嵌入层
    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        # 返回视觉模型的嵌入补丁层
        return self.vision_model.embeddings.patch_embedding

    # 定义调用方法，接收一系列输入参数，执行 BLIP 模型的前向传播
    @unpack_inputs
    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)  # 添加视觉输入文档字符串
    @replace_return_docstrings(output_type=TFBlipForConditionalGenerationModelOutput, config_class=BlipConfig)  # 替换返回的文档字符串
    def call(
        self,
        pixel_values: tf.Tensor,  # 视觉输入张量
        input_ids: tf.Tensor | None = None,  # 文本输入张量，可选
        attention_mask: tf.Tensor | None = None,  # 注意力掩码张量，可选
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选
        labels: tf.Tensor | None = None,  # 标签张量，可选
        return_dict: Optional[bool] = None,  # 是否返回字典，可选
        training: Optional[bool] = None,  # 是否处于训练模式，可选
    ) -> Union[Tuple, TFBlipForConditionalGenerationModelOutput]:
        r"""
        返回生成的文本和相关输出。

        Examples:

        ```py
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFBlipForConditionalGeneration

        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        >>> model = TFBlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "A picture of"

        >>> inputs = processor(images=image, text=text, return_tensors="tf")

        >>> outputs = model(**inputs)
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 通过视觉模型获取视觉输出
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 获取图像嵌入向量
        image_embeds = vision_outputs[0]

        # 使用文本解码器生成输出
        outputs = self.text_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            labels=labels,
            return_dict=return_dict,
            training=training,
        )

        # 如果不返回字典，则返回多个输出
        if not return_dict:
            outputs = (outputs[0], outputs[1], image_embeds, vision_outputs[0]) + vision_outputs[2:]
            return tuple(output for output in outputs if output is not None)

        # 如果输出中包含损失且其维度为标量，则重新整形
        if outputs.loss is not None and outputs.loss.shape.rank == 0:
            outputs.loss = tf.reshape(outputs.loss, (1,))

        # 返回 TFBlipForConditionalGenerationModelOutput 类型的输出
        return TFBlipForConditionalGenerationModelOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            image_embeds=image_embeds,
            last_hidden_state=vision_outputs.last_hidden_state,
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
        )

    def generate(
        self,
        pixel_values: tf.Tensor,
        input_ids: tf.Tensor | None = None,
        attention_mask: tf.Tensor | None = None,
        **generate_kwargs,
    ) -> tf.Tensor:
        r"""
        Overrides *generate* function to be able to use the model as a conditional generator

        Parameters:
            pixel_values (`tf.Tensor` of shape `(batch_size, num_channels, image_height, image_width)`:
                Input image to be processed
            input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:


        Examples:
        ```py
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFBlipForConditionalGeneration

        >>> model = TFBlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="tf")

        >>> outputs = model.generate(**inputs)
        >>> print(processor.decode(outputs[0], skip_special_tokens=True))
        two cats sleeping on a couch
        ```
        """

        batch_size = pixel_values.shape[0]  # 获取批次大小
        vision_outputs = self.vision_model(pixel_values=pixel_values)  # 使用视觉模型处理像素值，返回视觉模型的输出

        image_embeds = vision_outputs[0]  # 提取视觉模型输出的图像嵌入向量

        image_attention_mask = tf.ones(shape_list(image_embeds)[:-1], dtype=tf.int32)  # 生成图像的注意力掩码，全为1，形状与图像嵌入向量的形状一致

        if isinstance(input_ids, list):  # 如果输入的input_ids是列表形式
            input_ids = tf.convert_to_tensor(input_ids, dtype=tf.int32)  # 转换为张量
        elif input_ids is None:  # 如果没有提供input_ids
            input_ids = tf.convert_to_tensor(  # 使用默认的input_ids，即decoder输入的起始标记和结束标记
                [[self.decoder_input_ids, self.config.text_config.eos_token_id]], dtype=tf.int32
            )

            input_ids = tf.tile(input_ids, (batch_size, 1))  # 在批次维度上复制，形成与批次大小相同的input_ids

        # PyTorch: input_ids[:, 0] = self.config.text_config.bos_token_id
        input_ids = tf.concat(  # 在每个序列的开头添加开始标记
            [tf.ones((batch_size, 1), dtype=tf.int32) * self.config.text_config.bos_token_id, input_ids[:, 1:]], axis=1
        )
        attention_mask = attention_mask[:, :-1] if attention_mask is not None else None  # 截断attention_mask以匹配输入长度

        # 生成文本描述
        outputs = self.text_decoder.generate(
            input_ids=input_ids[:, :-1],  # 截断input_ids以匹配输出长度
            eos_token_id=self.config.text_config.sep_token_id,  # 结束标记的ID
            pad_token_id=self.config.text_config.pad_token_id,  # 填充标记的ID
            attention_mask=attention_mask,  # 注意力掩码
            encoder_hidden_states=image_embeds,  # 编码器的隐藏状态，即图像嵌入向量
            encoder_attention_mask=image_attention_mask,  # 编码器的注意力掩码
            **generate_kwargs,  # 传递其他生成参数
        )

        return outputs  # 返回生成的文本描述结果的张量
```py  
    # 构建模型，如果已经构建过则直接返回
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在视觉模型，则构建视觉模型
        if getattr(self, "vision_model", None) is not None:
            # 使用视觉模型的名称创建命名空间
            with tf.name_scope(self.vision_model.name):
                # 构建视觉模型
                self.vision_model.build(None)
        # 如果存在文本解码器，则构建文本解码器
        if getattr(self, "text_decoder", None) is not None:
            # 使用文本解码器的名称创建命名空间
            with tf.name_scope(self.text_decoder.name):
                # 构建文本解码器
                self.text_decoder.build(None)
# 导入模块，包括添加文档字符串的辅助函数
@add_start_docstrings(
    """
    BLIP 模型用于视觉问答。该模型包括一个视觉编码器、一个文本编码器以及一个文本解码器。
    视觉编码器将编码输入图像，文本编码器将编码输入问题以及图像的编码，
    而文本解码器将输出问题的答案。
    """,
    BLIP_START_DOCSTRING,
)
# 定义 TFBlipForQuestionAnswering 类，继承自 TFBlipPreTrainedModel
class TFBlipForQuestionAnswering(TFBlipPreTrainedModel):
    # 指定配置类
    config_class = BlipConfig
    # 加载时忽略的键列表
    _keys_to_ignore_on_load_missing = [r"text_decoder.cls.predictions.decoder.bias"]

    # 初始化方法
    def __init__(self, config: BlipConfig, *args, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *args, **kwargs)

        # 创建视觉模型，使用 TFBlipVisionModel 类
        self.vision_model = TFBlipVisionModel(config.vision_config, name="vision_model")

        # 创建文本编码器，使用 TFBlipTextModel 类，不添加池化层
        self.text_encoder = TFBlipTextModel(config.text_config, name="text_encoder", add_pooling_layer=False)

        # 创建文本解码器，使用 TFBlipTextLMHeadModel 类
        self.text_decoder = TFBlipTextLMHeadModel(config.text_config, name="text_decoder")

        # 设置解码器的填充标记和起始标记
        self.decoder_pad_token_id = config.text_config.pad_token_id
        self.decoder_start_token_id = config.text_config.bos_token_id

    # 获取输入嵌入的方法
    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        # 返回视觉模型的补丁嵌入
        return self.vision_model.embeddings.patch_embedding

    # 从 transformers.models.t5.modeling_tf_t5.TFT5PreTrainedModel._shift_right 方法调整而来
    def _shift_right(self, input_ids):
        # 获取解码器的起始标记和填充标记
        decoder_start_token_id = self.decoder_start_token_id
        pad_token_id = self.decoder_pad_token_id

        # 如果起始标记或填充标记未定义，则引发错误
        if decoder_start_token_id is None or pad_token_id is None:
            raise ValueError("decoder_start_token_id and pad_token_id must be defined!")

        # 创建起始标记张量，并转换为与输入标识相容的数据类型
        start_tokens = tf.fill((shape_list(input_ids)[0], 1), decoder_start_token_id)
        start_tokens = tf.cast(start_tokens, input_ids.dtype)  # Ensure compatible dtypes for concatenation
        # 将起始标记与输入标识的前面部分连接起来，形成右移后的输入标识
        shifted_input_ids = tf.concat([start_tokens, input_ids[:, :-1]], -1)

        # 替换标签中可能存在的 -100 值为填充标记
        shifted_input_ids = tf.where(
            shifted_input_ids == -100,
            tf.cast(tf.fill(shape_list(shifted_input_ids), pad_token_id), shifted_input_ids.dtype),
            shifted_input_ids,
        )

        # "Verify that `labels` has only positive values and -100"
        # 确保 shifted_input_ids 只包含正值和 -100
        tf.debugging.assert_greater_equal(shifted_input_ids, tf.constant(0, dtype=shifted_input_ids.dtype))

        return shifted_input_ids

    # 将输入解压并解包的装饰器
    @unpack_inputs
    # 将开始文档字符串添加到模型向前传播方法
    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    # 替换返回值文档字符串
    @replace_return_docstrings(output_type=TFBlipTextVisionModelOutput, config_class=BlipVisionConfig)
    # 定义一个方法用于调用模型
    def call(
        self,
        # 输入的 token IDs 张量
        input_ids: tf.Tensor,
        # 像素值张量，默认为 None
        pixel_values: tf.Tensor | None = None,
        # 解码器输入的 token IDs 张量，默认为 None
        decoder_input_ids: tf.Tensor | None = None,
        # 解码器注意力掩码张量，默认为 None
        decoder_attention_mask: tf.Tensor | None = None,
        # 注意力掩码张量，默认为 None
        attention_mask: tf.Tensor | None = None,
        # 是否输出注意力权重，默认为 None
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，默认为 None
        output_hidden_states: Optional[bool] = None,
        # 标签张量，默认为 None
        labels: tf.Tensor | None = None,
        # 是否返回字典，默认为 None
        return_dict: Optional[bool] = None,
        # 是否训练，默认为 None
        training: Optional[bool] = None,
    # 定义一个方法用于生成模型输出
    def generate(
        self,
        # 输入的 token IDs 张量
        input_ids: tf.Tensor,
        # 像素值张量
        pixel_values: tf.Tensor,
        # 注意力掩码张量，默认为 None
        attention_mask: tf.Tensor | None = None,
        # 其他生成参数
        **generate_kwargs,
    ) -> tf.Tensor:
        r"""
        Overrides *generate* function to be able to use the model as a conditional generator

        Parameters:
            input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            pixel_values (`tf.Tensor` of shape `(batch_size, num_channels, image_height, image_width)`:
                Input image to be processed
            attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`. `1` for
                tokens that are NOT MASKED, `0` for MASKED tokens.
            generate_kwargs (dict, *optional*):
                Additional arguments passed to the `generate` function of the decoder


        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFBlipForQuestionAnswering

        >>> model = TFBlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "How many cats are in the picture?"

        >>> inputs = processor(images=image, text=text, return_tensors="tf")

        >>> outputs = model.generate(**inputs)
        >>> print(processor.decode(outputs[0], skip_special_tokens=True))
        2
        ```py
        """
        vision_outputs = self.vision_model(pixel_values=pixel_values)

        image_embeds = vision_outputs[0]

        image_attention_mask = tf.ones(shape_list(image_embeds)[:-1], dtype=tf.int32)

        if isinstance(input_ids, list):
            input_ids = tf.Tensor(input_ids)  # 将输入的列表转换为张量

        question_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=False,
        )

        question_embeds = question_outputs[0]

        question_attention_mask = tf.ones(shape_list(question_embeds)[:-1], dtype=tf.int32)

        bos_ids = tf.fill(
            (tf.shape(question_embeds)[0], 1), value=tf.cast(self.decoder_start_token_id, input_ids.dtype)
        )

        outputs = self.text_decoder.generate(
            input_ids=bos_ids,
            eos_token_id=self.config.text_config.sep_token_id,
            pad_token_id=self.config.text_config.pad_token_id,
            encoder_hidden_states=question_embeds,
            encoder_attention_mask=question_attention_mask,
            **generate_kwargs,
        )

        return outputs
    # 构建模型方法，用于构建模型结构
    def build(self, input_shape=None):
        # 如果模型已经构建过，直接返回，避免重复构建
        if self.built:
            return
        # 设置模型构建状态为已构建
        self.built = True
        # 如果存在视觉模型，则构建视觉模型
        if getattr(self, "vision_model", None) is not None:
            # 使用 TensorFlow 的命名空间，将视觉模型的构建置于同一作用域下
            with tf.name_scope(self.vision_model.name):
                # 构建视觉模型，输入形状为 None，表示未指定输入形状
                self.vision_model.build(None)
        # 如果存在文本编码器，则构建文本编码器
        if getattr(self, "text_encoder", None) is not None:
            # 使用 TensorFlow 的命名空间，将文本编码器的构建置于同一作用域下
            with tf.name_scope(self.text_encoder.name):
                # 构建文本编码器，输入形状为 None，表示未指定输入形状
                self.text_encoder.build(None)
        # 如果存在文本解码器，则构建文本解码器
        if getattr(self, "text_decoder", None) is not None:
            # 使用 TensorFlow 的命名空间，将文本解码器的构建置于同一作用域下
            with tf.name_scope(self.text_decoder.name):
                # 构建文本解码器，输入形状为 None，表示未指定输入形状
                self.text_decoder.build(None)
# 添加模型文档字符串，描述 BLIP 模型的用途和功能
@add_start_docstrings(
    """
    BLIP Model with a vision and text projector, and a classification head on top. The model is used in the context of
    image-text retrieval. Given an image and a text, the model returns the probability of the text being relevant to
    the image.
    """,
    BLIP_START_DOCSTRING,
)
# 定义 TFBlipForImageTextRetrieval 类，继承自 TFBlipPreTrainedModel
class TFBlipForImageTextRetrieval(TFBlipPreTrainedModel):
    # 指定配置类为 BlipConfig
    config_class = BlipConfig

    # 初始化方法
    def __init__(self, config: BlipConfig, *args, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *args, **kwargs)

        # 创建视觉模型对象
        self.vision_model = TFBlipVisionModel(config.vision_config, name="vision_model")

        # 创建文本编码器对象
        self.text_encoder = TFBlipTextModel(config.text_config, name="text_encoder", add_pooling_layer=False)

        # 视觉投影层
        self.vision_proj = tf.keras.layers.Dense(
            config.image_text_hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="vision_proj",
        )

        # 文本投影层
        self.text_proj = tf.keras.layers.Dense(
            config.image_text_hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="text_proj",
        )

        # 图像文本匹配头
        self.itm_head = tf.keras.layers.Dense(
            2, kernel_initializer=get_initializer(config.initializer_range), name="itm_head"
        )

        # 获取解码器填充标记 ID
        self.decoder_pad_token_id = (
            config.text_config.pad_token_id
            if not hasattr(config, "decoder_pad_token_id")
            else config.decoder_pad_token_id
        )
        # 获取解码器起始标记 ID
        self.decoder_start_token_id = (
            config.text_config.bos_token_id
            if not hasattr(config, "decoder_start_token_id")
            else config.decoder_start_token_id
        )
        # 保存配置信息
        self.config = config

    # 获取输入嵌入层
    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        return self.vision_model.embeddings.patch_embedding

    # 模型调用方法，处理输入数据并返回输出
    @unpack_inputs
    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBlipImageTextMatchingModelOutput, config_class=BlipVisionConfig)
    def call(
        self,
        input_ids: tf.Tensor,
        pixel_values: tf.Tensor | None = None,
        use_itm_head: Optional[bool] = True,
        attention_mask: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = None,
    # 构建模型的方法，用于构建模型的各个部分
    def build(self, input_shape=None):
        # 如果模型已经构建完成，则直接返回
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        # 如果模型具有视觉模型属性
        if getattr(self, "vision_model", None) is not None:
            # 在命名空间中构建视觉模型
            with tf.name_scope(self.vision_model.name):
                # 构建视觉模型
                self.vision_model.build(None)
        # 如果模型具有文本编码器属性
        if getattr(self, "text_encoder", None) is not None:
            # 在命名空间中构建文本编码器
            with tf.name_scope(self.text_encoder.name):
                # 构建文本编码器
                self.text_encoder.build(None)
        # 如果模型具有视觉投影属性
        if getattr(self, "vision_proj", None) is not None:
            # 在命名空间中构建视觉投影
            with tf.name_scope(self.vision_proj.name):
                # 构建视觉投影，输入形状为[None, None, self.config.vision_config.hidden_size]
                self.vision_proj.build([None, None, self.config.vision_config.hidden_size])
        # 如果模型具有文本投影属性
        if getattr(self, "text_proj", None) is not None:
            # 在命名空间中构建文本投影
            with tf.name_scope(self.text_proj.name):
                # 构建文本投影，输入形状为[None, None, self.config.text_config.hidden_size]
                self.text_proj.build([None, None, self.config.text_config.hidden_size])
        # 如果模型具有多任务头属性
        if getattr(self, "itm_head", None) is not None:
            # 在命名空间中构建多任务头
            with tf.name_scope(self.itm_head.name):
                # 构建多任务头，输入形状为[None, None, self.config.text_config.hidden_size]
                self.itm_head.build([None, None, self.config.text_config.hidden_size])
```