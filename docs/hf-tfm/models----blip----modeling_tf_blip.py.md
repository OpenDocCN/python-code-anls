# `.\models\blip\modeling_tf_blip.py`

```
# coding=utf-8
# Copyright 2023 The Salesforce Team Authors and The HuggingFace Team. All rights reserved.
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
""" TensorFlow BLIP model."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import tensorflow as tf

# Importing specific modules and functions from custom TensorFlow utility files
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling
from ...modeling_tf_utils import (
    TFPreTrainedModel,
    get_initializer,
    get_tf_activation,
    keras,
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
# Importing configuration specific to BLIP models
from .configuration_blip import BlipConfig, BlipTextConfig, BlipVisionConfig
# Importing components from text modeling module for BLIP
from .modeling_tf_blip_text import BLIP_TEXT_INPUTS_DOCSTRING, TFBlipTextLMHeadModel, TFBlipTextModel


# Setting up logging specific to the current module
logger = logging.get_logger(__name__)

# Specifying a checkpoint reference for documentation purposes
_CHECKPOINT_FOR_DOC = "Salesforce/blip-vqa-base"

# List of pre-trained model archives for BLIP models
TF_BLIP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Salesforce/blip-vqa-base",
    "Salesforce/blip-vqa-capfilt-large",
    "Salesforce/blip-image-captioning-base",
    "Salesforce/blip-image-captioning-large",
    "Salesforce/blip-itm-base-coco",
    "Salesforce/blip-itm-large-coco",
    "Salesforce/blip-itm-base-flickr",
    "Salesforce/blip-itm-large-flickr",
    # See all BLIP models at https://huggingface.co/models?filter=blip
]


# Function for contrastive loss computation, adapted from transformers.models.clip.modeling_tf_clip.contrastive_loss
def contrastive_loss(logits: tf.Tensor) -> tf.Tensor:
    """
    Computes the contrastive loss based on the sparse categorical crossentropy.

    Args:
        logits (tf.Tensor): Logits tensor representing predictions.

    Returns:
        tf.Tensor: Mean contrastive loss value.
    """
    return tf.math.reduce_mean(
        keras.metrics.sparse_categorical_crossentropy(
            y_true=tf.range(shape_list(logits)[0]), y_pred=logits, from_logits=True
        )
    )


# Function for BLIP-specific loss computation, adapted from transformers.models.clip.modeling_tf_clip.clip_loss
def blip_loss(similarity: tf.Tensor) -> tf.Tensor:
    """
    Computes the BLIP loss, which is an average of contrastive losses calculated for captions and images.

    Args:
        similarity (tf.Tensor): Tensor representing similarity between captions and images.

    Returns:
        tf.Tensor: Computed BLIP loss value.
    """
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(tf.transpose(similarity))
    return (caption_loss + image_loss) / 2.0


@dataclass
class TFBlipForConditionalGenerationModelOutput(ModelOutput):
    """
    Output data structure for TFBlipForConditionalGenerationModel, inheriting from ModelOutput.

    Attributes:
        This class inherits attributes and methods from ModelOutput and adds none in this implementation.
    """
    pass
    Args:
        loss (`tf.Tensor`, *optional*, returned when `labels` is provided, `tf.Tensor` of shape `(1,)`):
            Languge modeling loss from the text decoder.
        logits (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`, *optional*):
            Prediction scores of the language modeling head of the text decoder model.
        image_embeds (`tf.Tensor` of shape `(batch_size, output_dim)`, *optional*):
            The image embeddings obtained after applying the Vision Transformer model to the input image.
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings, if the model has an embedding layer, + one for
            the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    # 初始化各个属性为 None，这些属性用于存储模型推断的结果
    loss: Tuple[tf.Tensor] | None = None
    logits: Tuple[tf.Tensor] | None = None
    image_embeds: tf.Tensor | None = None
    last_hidden_state: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor, ...] | None = None
    attentions: Tuple[tf.Tensor, ...] | None = None

    @property
    def decoder_logits(self):
        # 发出警告，提醒用户 `decoder_logits` 属性即将被移除，建议使用 `logits` 属性来获取最终输出
        warnings.warn(
            "`decoder_logits` attribute is deprecated and will be removed in version 5 of Transformers."
            " Please use the `logits` attribute to retrieve the final output instead.",
            FutureWarning,
        )
        # 返回 `logits` 属性的值作为输出
        return self.logits
# 定义一个用于 TFBlip 文本视觉模型输出的数据类，继承自 ModelOutput 基类
@dataclass
class TFBlipTextVisionModelOutput(ModelOutput):
    """
    从基类适配的视觉模型输出的扩展，还包含了最后隐藏状态的图像嵌入。该类还添加了文本解码器的损失项。

    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            文本解码器的语言建模损失。
        image_embeds (`tf.Tensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            图像嵌入，通过将投影层应用于池化器输出获得。
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的隐藏状态序列。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            元组 `tf.Tensor` 的隐藏状态（如果模型具有嵌入层，则为输出的初始嵌入输出 + 每一层的输出），
            形状为 `(batch_size, sequence_length, hidden_size)`。
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            元组 `tf.Tensor` 的注意力权重（每层一个），形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    # 损失项，类型为 tf.Tensor 或 None
    loss: tf.Tensor | None = None
    # 图像嵌入，类型为 tf.Tensor 或 None
    image_embeds: tf.Tensor | None = None
    # 最后一层隐藏状态，类型为 tf.Tensor 或 None
    last_hidden_state: tf.Tensor = None
    # 隐藏状态元组，包含模型每层的隐藏状态，类型为 Tuple[tf.Tensor] 或 None
    hidden_states: Tuple[tf.Tensor, ...] | None = None
    # 注意力权重元组，包含每层的注意力权重，类型为 Tuple[tf.Tensor] 或 None
    attentions: Tuple[tf.Tensor, ...] | None = None


# 定义一个用于 TFBlip 图像文本匹配模型输出的数据类，继承自 ModelOutput 基类
@dataclass
class TFBlipImageTextMatchingModelOutput(ModelOutput):
    """
    从基类适配的视觉模型输出的扩展，还包含了最后隐藏状态的图像嵌入。该类还添加了文本解码器的损失项以及图像文本相似度分数。

    (此处省略了进一步的文档内容，未提供完整的注释)
    """
    Args:
        itm_score (`tf.Tensor`):
            图像和文本的相似度分数。
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            文本解码器的语言建模损失。
        image_embeds (`tf.Tensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            通过投影层应用到池化输出得到的图像嵌入。
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的隐藏状态序列输出。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            模型每一层的隐藏状态元组，包括可能的初始嵌入层输出。
        vision_pooler_output (`tf.Tensor` of shape `(batch_size, hidden_size)`, *optional*):
            模型视觉分支中视觉池化层的最后一层隐藏状态。
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            注意力权重的元组，用于计算自注意力头中的加权平均值。
        question_embeds (`tf.Tensor`):
            文本投影层得到的问题嵌入。
    """

    itm_score: tf.Tensor | None = None  # 初始化图像和文本的相似度分数，默认为 None
    loss: tf.Tensor | None = None  # 初始化语言建模损失，默认为 None；在提供 `labels` 时返回
    image_embeds: tf.Tensor | None = None  # 初始化图像嵌入，默认为 None；当 `with_projection=True` 时返回
    last_hidden_state: tf.Tensor = None  # 初始化最后一层隐藏状态，默认为 None
    hidden_states: Tuple[tf.Tensor, ...] | None = None  # 初始化隐藏状态元组，默认为 None；在 `output_hidden_states=True` 时返回
    vision_pooler_output: tf.Tensor | None = None  # 初始化视觉池化层的最后一层隐藏状态，默认为 None
    attentions: Tuple[tf.Tensor, ...] | None = None  # 初始化注意力权重元组，默认为 None；在 `output_attentions=True` 时返回
    question_embeds: Tuple[tf.Tensor] | None = None  # 初始化问题嵌入，默认为 None
@dataclass
class TFBlipOutput(ModelOutput):
    """
    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image: (`tf.Tensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text: (`tf.Tensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds: (`tf.Tensor` of shape `(batch_size, output_dim)`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`BlipTextModel`].
        image_embeds: (`tf.Tensor` of shape `(batch_size, output_dim)`):
            The image embeddings obtained by applying the projection layer to the pooled output of [`BlipVisionModel`].
        text_model_output: (`BaseModelOutputWithPooling`):
            The output of the [`BlipTextModel`].
        vision_model_output: (`BaseModelOutputWithPooling`):
            The output of the [`BlipVisionModel`].
    """

    loss: tf.Tensor | None = None
    logits_per_image: tf.Tensor = None
    logits_per_text: tf.Tensor = None
    text_embeds: tf.Tensor = None
    image_embeds: tf.Tensor = None
    text_model_output: TFBaseModelOutputWithPooling = None
    vision_model_output: TFBaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert TFBlipOutput object to a tuple, excluding `text_model_output` and `vision_model_output` which are 
        converted to tuples separately.

        Returns:
            Tuple[Any]: A tuple representation of the object.
        """
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class TFBlipVisionEmbeddings(keras.layers.Layer):
    def __init__(self, config: BlipVisionConfig, **kwargs):
        """
        Initialize the TFBlipVisionEmbeddings layer.

        Args:
            config (BlipVisionConfig): Configuration object for BlipVisionModel.
            **kwargs: Additional keyword arguments passed to the Layer constructor.
        """
        super().__init__(**kwargs)
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        # Define patch embedding layer
        self.patch_embedding = keras.layers.Conv2D(
            filters=self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            kernel_initializer=get_initializer(self.config.initializer_range),
            data_format="channels_last",
            name="patch_embedding",
        )

        # Calculate number of patches and positions
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
    # 构建模型的方法，在输入形状为 `input_shape` 的情况下进行构建
    def build(self, input_shape=None):
        # 添加类别嵌入权重，形状为 (1, 1, embed_dim)，使用给定范围的初始化器进行初始化
        self.class_embedding = self.add_weight(
            shape=(1, 1, self.embed_dim),
            initializer=get_initializer(self.config.initializer_range),
            trainable=True,
            name="class_embedding",
        )

        # 添加位置嵌入权重，形状为 (1, num_positions, embed_dim)，使用给定范围的初始化器进行初始化
        self.position_embedding = self.add_weight(
            shape=(1, self.num_positions, self.embed_dim),
            initializer=get_initializer(self.config.initializer_range),
            trainable=True,
            name="position_embedding",
        )

        # 如果模型已经构建，则直接返回
        if self.built:
            return
        self.built = True

        # 如果存在 `patch_embedding` 属性，则对其进行构建
        if getattr(self, "patch_embedding", None) is not None:
            with tf.name_scope(self.patch_embedding.name):
                self.patch_embedding.build([None, None, None, 3])

    # 模型的调用方法，接受像素值张量作为输入，返回嵌入张量作为输出
    def call(self, pixel_values: tf.Tensor) -> tf.Tensor:
        # 输入张量是通道优先的，进行转置以适应模型的通道次序（通道在最后的顺序）
        batch_size = tf.shape(pixel_values)[0]
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))
        
        # 使用 `patch_embedding` 对转置后的像素值进行嵌入
        patch_embeds = self.patch_embedding(pixel_values)
        # 将嵌入张量重新调整形状为 (batch_size, num_patches, -1)
        patch_embeds = tf.reshape(patch_embeds, (batch_size, self.num_patches, -1))

        # 扩展类别嵌入以匹配批次大小，并与 patch 嵌入连接起来
        class_embeds = tf.broadcast_to(self.class_embedding, (batch_size, 1, self.embed_dim))
        embeddings = tf.concat([class_embeds, patch_embeds], axis=1)
        
        # 将位置嵌入加到嵌入张量中（仅限于嵌入张量的长度部分）
        embeddings = embeddings + self.position_embedding[:, : tf.shape(embeddings)[1], :]
        
        # 返回最终的嵌入张量作为模型的输出
        return embeddings
# Copied from transformers.models.clip.modeling_tf_clip.TFCLIPTextEmbeddings with CLIP->Blip
class TFBlipTextEmbeddings(keras.layers.Layer):
    def __init__(self, config: BlipTextConfig, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = config.hidden_size  # 设置嵌入维度为配置中的隐藏层大小
        self.config = config  # 保存配置信息

    def build(self, input_shape: tf.TensorShape = None):
        with tf.name_scope("token_embedding"):
            # 创建 token 嵌入权重，形状为 (词汇大小, 嵌入维度)，使用指定初始化方法
            self.weight = self.add_weight(
                shape=(self.config.vocab_size, self.embed_dim),
                initializer=get_initializer(self.config.initializer_factor * self.config.initializer_range),
                trainable=True,
                name="weight",
            )

        with tf.name_scope("position_embedding"):
            # 创建位置嵌入权重，形状为 (最大位置嵌入数, 嵌入维度)，使用指定初始化方法
            self.position_embedding = self.add_weight(
                shape=(self.config.max_position_embeddings, self.embed_dim),
                initializer=get_initializer(self.config.initializer_factor * self.config.initializer_range),
                trainable=True,
                name="embeddings",
            )

        super().build(input_shape)  # 调用父类的 build 方法

    def call(
        self,
        input_ids: tf.Tensor = None,
        position_ids: tf.Tensor = None,
        inputs_embeds: tf.Tensor = None,
    ) -> tf.Tensor:
        """
        Applies embedding based on inputs tensor.

        Args:
            input_ids (tf.Tensor, optional): 输入的 token ID 张量
            position_ids (tf.Tensor, optional): 输入的位置 ID 张量
            inputs_embeds (tf.Tensor, optional): 输入的嵌入张量

        Returns:
            final_embeddings (`tf.Tensor`): 输出的嵌入张量.
        """
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)  # 检查嵌入是否在合理范围内
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)  # 根据 input_ids 获取嵌入向量

        input_shape = shape_list(inputs_embeds)[:-1]  # 获取输入嵌入张量的形状，去除最后一个维度

        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)  # 自动生成位置 ID

        position_embeds = tf.gather(params=self.position_embedding, indices=position_ids)  # 根据 position_ids 获取位置嵌入
        position_embeds = tf.tile(input=position_embeds, multiples=(input_shape[0], 1, 1))  # 复制位置嵌入以匹配输入的形状
        final_embeddings = inputs_embeds + position_embeds  # 计算最终的嵌入张量

        return final_embeddings


class TFBlipAttention(keras.layers.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config  # 将传入的配置保存为实例变量
        self.embed_dim = config.hidden_size  # 设置嵌入维度为配置中的隐藏大小
        self.num_heads = config.num_attention_heads  # 设置注意力头的数量为配置中的值
        self.head_dim = self.embed_dim // self.num_heads  # 计算每个注意力头的维度
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5  # 计算缩放因子，用于注意力分数的缩放
        self.dropout = keras.layers.Dropout(config.attention_dropout, name="dropout")  # 初始化丢弃层

        self.qkv = keras.layers.Dense(
            3 * self.embed_dim, kernel_initializer=get_initializer(config.initializer_range), name="qkv"
        )  # 创建用于查询、键、值的全连接层

        self.projection = keras.layers.Dense(
            self.embed_dim, kernel_initializer=get_initializer(config.initializer_range), name="projection"
        )  # 创建用于投影的全连接层

    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor | None = None,
        output_attentions: Optional[bool] = False,
        training: Optional[bool] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor | None, Tuple[tf.Tensor] | None]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = shape_list(hidden_states)  # 获取隐藏状态张量的形状信息

        mixed_qkv = self.qkv(hidden_states)  # 将隐藏状态张量映射到查询、键、值空间
        mixed_qkv = tf.reshape(mixed_qkv, (bsz, tgt_len, 3, self.num_heads, self.head_dim))  # 重塑成多头查询、键、值张量
        mixed_qkv = tf.transpose(mixed_qkv, perm=(2, 0, 3, 1, 4))  # 调整张量顺序以便后续操作

        query_states, key_states, value_states = mixed_qkv[0], mixed_qkv[1], mixed_qkv[2]  # 分离查询、键、值张量

        # 计算注意力分数，即查询和键的点积
        attention_scores = query_states @ tf.transpose(key_states, (0, 1, 3, 2))

        attention_scores = attention_scores * self.scale  # 缩放注意力分数

        # 将注意力分数归一化为注意力概率
        attention_probs = stable_softmax(attention_scores, axis=-1)

        # 使用丢弃层在训练时随机丢弃注意力概率中的值
        attention_probs = self.dropout(attention_probs, training=training)

        # 如果存在头部掩码，则应用头部掩码
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = tf.transpose(attention_probs @ value_states, perm=(0, 2, 1, 3))  # 计算加权值张量

        new_context_layer_shape = shape_list(context_layer)[:-2] + [self.embed_dim]  # 调整上下文层的形状
        context_layer = tf.reshape(context_layer, new_context_layer_shape)

        output = self.projection(context_layer)  # 对上下文层进行投影

        outputs = (output, attention_probs) if output_attentions else (output, None)  # 根据需求返回输出

        return outputs
    # 构建函数，用于构建模型的层
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回，避免重复构建
        if self.built:
            return
        # 设置标志为已构建
        self.built = True
        
        # 如果存在 dropout 层，构建 dropout 层
        if getattr(self, "dropout", None) is not None:
            # 使用 dropout 层的名称作为命名空间
            with tf.name_scope(self.dropout.name):
                # 调用 dropout 层的 build 方法，传入空的输入形状
                self.dropout.build(None)
        
        # 如果存在 qkv 层，构建 qkv 层
        if getattr(self, "qkv", None) is not None:
            # 使用 qkv 层的名称作为命名空间
            with tf.name_scope(self.qkv.name):
                # 调用 qkv 层的 build 方法，传入输入形状为 [None, None, self.embed_dim]
                self.qkv.build([None, None, self.embed_dim])
        
        # 如果存在 projection 层，构建 projection 层
        if getattr(self, "projection", None) is not None:
            # 使用 projection 层的名称作为命名空间
            with tf.name_scope(self.projection.name):
                # 调用 projection 层的 build 方法，传入输入形状为 [None, None, self.embed_dim]
                self.projection.build([None, None, self.embed_dim])
class TFBlipMLP(keras.layers.Layer):
    # TFBlipMLP 类，用于定义一个多层感知机（MLP）的自定义层
    def __init__(self, config: BlipConfig, **kwargs):
        super().__init__(**kwargs)

        # 获取激活函数
        self.activation_fn = get_tf_activation(config.hidden_act)

        # 计算输入投影的标准差
        in_proj_std = (config.hidden_size**-0.5) * ((2 * config.num_hidden_layers) ** -0.5)
        # 计算全连接层的初始化标准差
        fc_std = (2 * config.hidden_size) ** -0.5

        # 创建全连接层 fc1，用于中间层，初始化权重
        self.fc1 = keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(fc_std), name="fc1"
        )
        # 创建全连接层 fc2，用于输入投影，初始化权重
        self.fc2 = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(in_proj_std), name="fc2"
        )
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 前向传播函数 call，接收隐藏状态张量并返回处理后的张量

        # 使用 fc1 进行全连接操作
        hidden_states = self.fc1(inputs=hidden_states)
        # 应用激活函数
        hidden_states = self.activation_fn(hidden_states)
        # 使用 fc2 进行全连接操作
        hidden_states = self.fc2(inputs=hidden_states)
        return hidden_states

    def build(self, input_shape=None):
        # 构建函数 build，在第一次调用时构建层的权重

        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 设置为已构建状态
        self.built = True

        # 如果存在 fc1 层，则在 tf 的名称作用域下构建 fc1 层
        if getattr(self, "fc1", None) is not None:
            with tf.name_scope(self.fc1.name):
                self.fc1.build([None, None, self.config.hidden_size])

        # 如果存在 fc2 层，则在 tf 的名称作用域下构建 fc2 层
        if getattr(self, "fc2", None) is not None:
            with tf.name_scope(self.fc2.name):
                self.fc2.build([None, None, self.config.intermediate_size])


class TFBlipEncoderLayer(keras.layers.Layer):
    # TFBlipEncoderLayer 类，用于定义一个编码器层
    def __init__(self, config: BlipConfig, **kwargs):
        super().__init__(**kwargs)
        
        # 设置嵌入维度
        self.embed_dim = config.hidden_size
        # 初始化自注意力层 self_attn
        self.self_attn = TFBlipAttention(config, name="self_attn")
        # 初始化第一层规范化层 layer_norm1
        self.layer_norm1 = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm1")
        # 初始化多层感知机 MLP
        self.mlp = TFBlipMLP(config, name="mlp")
        # 初始化第二层规范化层 layer_norm2
        self.layer_norm2 = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm2")

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        output_attentions: Optional[bool] = False,
        training: Optional[bool] = None,
    ) -> Tuple[tf.Tensor]:
        """
        Args:
            hidden_states (`tf.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`tf.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states  # 保存输入的隐藏状态作为残差连接的起点

        hidden_states = self.layer_norm1(hidden_states)  # 执行第一个层归一化操作

        # 使用自注意力机制处理隐藏状态，并获取注意力权重
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            head_mask=attention_mask,
            output_attentions=output_attentions,
            training=training,
        )

        hidden_states = hidden_states + residual  # 将残差连接添加到处理后的隐藏状态中
        residual = hidden_states  # 更新残差连接的起点为当前的隐藏状态

        hidden_states = self.layer_norm2(hidden_states)  # 执行第二个层归一化操作

        hidden_states = self.mlp(hidden_states)  # 使用多层感知机处理隐藏状态

        hidden_states = hidden_states + residual  # 再次将残差连接添加到处理后的隐藏状态中

        outputs = (hidden_states,)  # 准备输出为一个元组，包含处理后的隐藏状态

        if output_attentions:
            outputs += (attn_weights,)  # 如果需要输出注意力权重，则添加到输出中

        return outputs  # 返回最终的输出结果

    def build(self, input_shape=None):
        if self.built:
            return  # 如果已经构建过，则直接返回

        self.built = True  # 标记模型已经构建完成

        if getattr(self, "self_attn", None) is not None:
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)  # 构建自注意力层

        if getattr(self, "layer_norm1", None) is not None:
            with tf.name_scope(self.layer_norm1.name):
                self.layer_norm1.build([None, None, self.embed_dim])  # 构建第一个层归一化层

        if getattr(self, "mlp", None) is not None:
            with tf.name_scope(self.mlp.name):
                self.mlp.build(None)  # 构建多层感知机层

        if getattr(self, "layer_norm2", None) is not None:
            with tf.name_scope(self.layer_norm2.name):
                self.layer_norm2.build([None, None, self.embed_dim])  # 构建第二个层归一化层
BLIP_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
        attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Mask to avoid performing attention on padding tokens.
        token_type_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Segment token indices to indicate first and second portions of the inputs. Only used by some models like BERT.
        position_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Indices of positions of each input token in the position embeddings.
        inputs_embeds (`tf.Tensor`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated embeddings.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""
    Args:
        input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            # 输入序列中每个token在词汇表中的索引。默认情况下会忽略填充。
            # 可以使用[`AutoProcessor`]获取这些索引。详见[`BlipProcessor.__call__`]。
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 用于避免在填充的token索引上执行注意力计算的掩码。掩码取值在`[0, 1]`之间：
            # - 1表示**未被掩码**的token，
            # - 0表示**被掩码**的token。
            [What are attention masks?](../glossary#attention-mask)
        position_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 输入序列中每个token在位置嵌入中的位置索引。取值范围是`[0, config.max_position_embeddings - 1]`。
            [What are position IDs?](../glossary#position-ids)
        pixel_values (`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):
            # 像素值。默认情况下会忽略填充。
            # 可以使用[`BlipImageProcessor`]获取这些像素值。详见[`BlipImageProcessor.__call__`]。
        return_loss (`bool`, *optional*):
            # 是否返回对比损失。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。返回结果中的`attentions`字段会有更详细的说明。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。返回结果中的`hidden_states`字段会有更详细的说明。
        return_dict (`bool`, *optional*):
            # 是否返回[`~utils.ModelOutput`]而不是普通的元组。
"""
@keras_serializable
class TFBlipEncoder(keras.layers.Layer):
    config_class = BlipConfig
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`BlipEncoderLayer`].

    Args:
        config (`BlipConfig`):
            The corresponding vision configuration for the `BlipEncoder`.
    """

    def __init__(self, config: BlipConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        # 创建一个由多个 `TFBlipEncoderLayer` 组成的列表，每个层使用配置参数并命名
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
        # Determine if `output_attentions` should be overridden by `self.config.output_attentions`
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # Determine if `output_hidden_states` should be overridden by `self.config.output_hidden_states`
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # Determine if `return_dict` should be overridden by `self.config.use_return_dict`
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Initialize empty tuple for encoder states if `output_hidden_states` is False
        encoder_states = () if output_hidden_states else None
        # Initialize empty tuple for all attentions if `output_attentions` is False
        all_attentions = () if output_attentions else None

        # Start with the embedded inputs as the initial hidden states
        hidden_states = inputs_embeds

        # Iterate through each encoder layer
        for idx, encoder_layer in enumerate(self.layers):
            # Append current hidden states to encoder states if `output_hidden_states` is True
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            
            # Pass the current hidden states through the encoder layer
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                output_attentions=output_attentions,
                training=training,
            )

            # Update hidden states with the output of the encoder layer
            hidden_states = layer_outputs[0]

            # Append attention weights of the current layer if `output_attentions` is True
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Append final hidden states to encoder states if `output_hidden_states` is True
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        # Return outputs based on `return_dict` flag
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

    def build(self, input_shape=None):
        # Check if the model is already built; if yes, return immediately
        if self.built:
            return
        
        # Mark the model as built
        self.built = True
        
        # If `self.layers` attribute exists, iterate through each layer and build it
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    # Build each layer with `None` input shape
                    layer.build(None)
    class TFBlipVisionModel(TFBlipPreTrainedModel):
        # 主要输入名称为 "pixel_values"
        main_input_name = "pixel_values"
        # 配置类为 BlipVisionConfig
        config_class = BlipVisionConfig

        def __init__(self, config: BlipVisionConfig, *args, **kwargs):
            # 调用父类的初始化方法
            super().__init__(config, *args, **kwargs)
            # 保存配置对象
            self.config = config

            # 创建嵌入层对象，使用 TFBlipVisionEmbeddings 类
            self.embeddings = TFBlipVisionEmbeddings(config, name="embeddings")
            # 创建编码器对象，使用 TFBlipEncoder 类
            self.encoder = TFBlipEncoder(config, name="encoder")
            # 创建后层归一化层对象，使用给定的 epsilon 参数
            self.post_layernorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="post_layernorm")
            # 设置嵌入维度为配置中的隐藏大小
            self.embed_dim = config.hidden_size

        def serving_output(self, output: TFBaseModelOutputWithPooling) -> TFBaseModelOutputWithPooling:
            # 如果配置要求输出隐藏状态，则将隐藏状态转换为张量
            hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
            # 如果配置要求输出注意力，则将注意力转换为张量
            attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

            # 返回包含指定属性的 TFBaseModelOutputWithPooling 对象
            return TFBaseModelOutputWithPooling(
                last_hidden_state=output.last_hidden_state,
                pooler_output=output.pooler_output,
                hidden_states=hs,
                attentions=attns,
            )

        @unpack_inputs
        @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
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
        返回类型提示：可能是元组或 TFBaseModelOutputWithPooling 类的对象

        """
        # 如果未指定 output_attentions 参数，则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定 output_hidden_states 参数，则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定 return_dict 参数，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果 pixel_values 参数为 None，则抛出数值错误异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 将像素值通过嵌入层处理，得到隐藏状态
        hidden_states = self.embeddings(pixel_values)

        # 使用编码器处理隐藏状态，获取编码器的输出
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 获取编码器的最后隐藏状态，并通过后层归一化处理
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        # 提取汇聚输出，即编码器输出的第一个位置
        pooled_output = last_hidden_state[:, 0, :]
        # TensorFlow 对输入的秩（rank）不一致时可能会出错，因此插入一个单维度来确保一致性
        pooled_output = self.post_layernorm(tf.expand_dims(pooled_output, 1))
        pooled_output = tf.squeeze(pooled_output, 1)

        # 如果不要求返回字典形式，则返回一个元组
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 否则，返回 TFBaseModelOutputWithPooling 对象，其中包含编码器输出的各项属性
        return TFBaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def get_input_embeddings(self):
        return self.embeddings

    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回
        if self.built:
            return
        self.built = True
        # 如果 embeddings 属性存在，则构建 embeddings 层
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        # 如果 encoder 属性存在，则构建 encoder 层
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果 post_layernorm 属性存在，则构建 post_layernorm 层，输入形状为 [None, None, self.embed_dim]
        if getattr(self, "post_layernorm", None) is not None:
            with tf.name_scope(self.post_layernorm.name):
                self.post_layernorm.build([None, None, self.embed_dim])
# 定义 TFBlipMainLayer 类，继承自 keras.layers.Layer，用于实现主层逻辑
class TFBlipMainLayer(keras.layers.Layer):
    # 设置类属性 config_class 为 BlipConfig 类型
    config_class = BlipConfig

    # 初始化方法，接受 BlipConfig 类型的 config 参数及其他位置和关键字参数
    def __init__(self, config: BlipConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 检查 config.text_config 是否为 BlipTextConfig 类型，若不是则抛出 ValueError 异常
        if not isinstance(config.text_config, BlipTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type BlipTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        # 检查 config.vision_config 是否为 BlipVisionConfig 类型，若不是则抛出 ValueError 异常
        if not isinstance(config.vision_config, BlipVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type BlipVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        # 从 config 中获取 text_config 和 vision_config 对象
        text_config = config.text_config
        vision_config = config.vision_config

        # 设置实例变量，分别表示投影维度、文本嵌入维度和视觉嵌入维度
        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        # 创建 TFBlipTextModel 实例并赋给 self.text_model，命名为 "text_model"
        self.text_model = TFBlipTextModel(text_config, name="text_model")
        
        # 创建 TFBlipVisionModel 实例并赋给 self.vision_model，命名为 "vision_model"
        self.vision_model = TFBlipVisionModel(vision_config, name="vision_model")

        # 创建 Dense 层实例 self.visual_projection，用于视觉投影，设置投影维度、不使用偏置、使用指定初始化器
        self.visual_projection = keras.layers.Dense(
            self.projection_dim,
            use_bias=False,
            kernel_initializer=get_initializer(config.initializer_range),
            name="visual_projection",
        )

        # 创建 Dense 层实例 self.text_projection，用于文本投影，设置投影维度、不使用偏置、使用指定初始化器
        self.text_projection = keras.layers.Dense(
            self.projection_dim,
            use_bias=False,
            kernel_initializer=get_initializer(config.initializer_range),
            name="text_projection",
        )

        # 将 config 参数赋给实例变量 self.config
        self.config = config

    # build 方法，用于构建层，接受 input_shape 参数
    def build(self, input_shape=None):
        # 创建并添加名为 logit_scale 的可训练权重，初始化为 config.logit_scale_init_value
        self.logit_scale = self.add_weight(
            name="logit_scale",
            shape=[],
            initializer=keras.initializers.Constant(self.config.logit_scale_init_value),
            trainable=True,
        )

        # 如果已经构建过，则直接返回
        if self.built:
            return
        
        # 标记为已构建
        self.built = True
        
        # 如果存在 self.text_model，则构建 self.text_model
        if getattr(self, "text_model", None) is not None:
            with tf.name_scope(self.text_model.name):
                self.text_model.build(None)
        
        # 如果存在 self.vision_model，则构建 self.vision_model
        if getattr(self, "vision_model", None) is not None:
            with tf.name_scope(self.vision_model.name):
                self.vision_model.build(None)
        
        # 如果存在 self.visual_projection，则构建 self.visual_projection
        if getattr(self, "visual_projection", None) is not None:
            with tf.name_scope(self.visual_projection.name):
                self.visual_projection.build([None, None, self.vision_embed_dim])
        
        # 如果存在 self.text_projection，则构建 self.text_projection
        if getattr(self, "text_projection", None) is not None:
            with tf.name_scope(self.text_projection.name):
                self.text_projection.build([None, None, self.text_embed_dim])

    # unpack_inputs 装饰器用于处理输入参数的解包操作
    @unpack_inputs
    # 定义 BLIP 模型的调用方法，接受多个输入参数和可选的输出参数，并返回 TFBlipOutput 或元组
    def call(
        self,
        input_ids: tf.Tensor | None = None,  # 输入的文本序列的张量，可选
        pixel_values: tf.Tensor | None = None,  # 输入的图像像素值的张量，可选
        attention_mask: tf.Tensor | None = None,  # 文本的注意力遮罩张量，可选
        position_ids: tf.Tensor | None = None,  # 文本的位置编码张量，可选
        return_loss: Optional[bool] = None,  # 是否返回损失值，可选
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选
        return_dict: Optional[bool] = None,  # 是否返回字典形式的输出，可选
        training: Optional[bool] = None,  # 是否处于训练模式，可选
    ) -> Union[Tuple, TFBlipOutput]:  # 返回值可以是元组或 TFBlipOutput 对象

        # 如果没有显式指定，使用 BLIP 模型配置中的设定值来填充相应的输出参数
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用视觉模型处理图像输入，并根据指定参数输出相应的结果
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 使用文本模型处理文本输入，并根据指定参数输出相应的结果
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 从视觉模型的输出中获取图像嵌入表示，并应用视觉投影层
        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        # 从文本模型的输出中获取文本嵌入表示，并应用文本投影层
        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        # 对图像嵌入进行 L2 范数归一化
        image_embeds = image_embeds / tf.norm(image_embeds, ord=2, axis=-1, keepdims=True)
        # 对文本嵌入进行 L2 范数归一化
        text_embeds = text_embeds / tf.norm(text_embeds, ord=2, axis=-1, keepdims=True)

        # 使用余弦相似度计算作为对数概率（logits）
        logit_scale = tf.exp(self.logit_scale)
        logits_per_text = tf.matmul(text_embeds, image_embeds, transpose_b=True) * logit_scale
        logits_per_image = tf.transpose(logits_per_text)

        # 如果需要返回损失值，则计算 BLIP 损失
        loss = None
        if return_loss:
            loss = blip_loss(logits_per_text)
            loss = tf.reshape(loss, (1,))

        # 如果不需要返回字典形式的输出，则返回一个包含多个输出的元组
        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典形式的输出，则创建 TFBlipOutput 对象并返回
        return TFBlipOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )
class TFBlipModel(TFBlipPreTrainedModel):
    # 指定配置类为BlipConfig
    config_class = BlipConfig
    # 在加载模型时忽略的键列表
    _keys_to_ignore_on_load_missing = [r"text_decoder.cls.predictions.decoder.bias"]
    # 主输入名称为"input_ids"
    main_input_name = "input_ids"

    def __init__(self, config: BlipConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 初始化TFBlipMainLayer作为模型的主要层，使用给定的配置
        self.blip = TFBlipMainLayer(config, name="blip")

    def serving_output(self, output: TFBlipOutput) -> TFBlipOutput:
        # 用于模型服务输出，直接返回给定的TFBlipOutput对象
        return TFBlipOutput(
            logits_per_image=output.logits_per_image,
            logits_per_text=output.logits_per_text,
            text_embeds=output.text_embeds,
            image_embeds=output.image_embeds,
        )

    @unpack_inputs
    @add_start_docstrings_to_model_forward(BLIP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBlipOutput, config_class=BlipConfig)
    def call(
        self,
        input_ids: tf.Tensor | None = None,
        pixel_values: tf.Tensor | None = None,
        attention_mask: tf.Tensor | None = None,
        position_ids: tf.Tensor | None = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = None,
    ) -> Union[Tuple, TFBlipOutput]:
        r"""
        模型的前向传播方法，接受多种输入参数并返回输出。

        Args:
            input_ids: 输入的token IDs张量，可以为None。
            pixel_values: 图像像素值张量，可以为None。
            attention_mask: 注意力遮罩张量，可以为None。
            position_ids: 位置IDs张量，可以为None。
            return_loss: 是否返回损失值，可选布尔值。
            output_attentions: 是否输出注意力张量，可选布尔值。
            output_hidden_states: 是否输出隐藏状态张量，可选布尔值。
            return_dict: 是否返回字典格式输出，可选布尔值。
            training: 是否处于训练模式，可选布尔值。

        Returns:
            模型的输出结果，类型为TFBlipOutput或一个元组。

        Examples:
        
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFBlipModel

        >>> model = TFBlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="tf", padding=True
        ... )

        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # 图像文本相似度得分
        >>> probs = tf.nn.softmax(logits_per_image, axis=1)  # 可以使用softmax获取标签概率
        ```
        """
        # 调用self.blip对象的call方法，传递所有参数，并返回其输出
        outputs = self.blip(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_loss=return_loss,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        return outputs

    @add_start_docstrings_to_model_forward(BLIP_TEXT_INPUTS_DOCSTRING)
    def get_text_features(
        self,
        input_ids: tf.Tensor | None = None,
        attention_mask: tf.Tensor | None = None,
        position_ids: tf.Tensor | None = None,
        return_dict: Optional[bool] = None,
        # 方法用于获取文本特征，接受文本相关的输入参数并返回对应的特征
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果 return_dict 不为 None，则使用它；否则使用配置中的 use_return_dict

        vision_outputs = self.blip.vision_model(pixel_values=pixel_values, return_dict=return_dict)
        # 使用 BLIP 视觉模型处理像素值，获取视觉输出，根据 return_dict 决定是否返回字典形式的结果

        pooled_output = vision_outputs[1]  # pooled_output
        # 从视觉输出中取出第二个元素作为汇聚输出，通常用于特征投影

        image_features = self.blip.visual_projection(pooled_output)
        # 使用 BLIP 视觉投影层对汇聚输出进行特征投影，得到图像特征

        return image_features
        # 返回经过特征投影后的图像特征张量
@add_start_docstrings(
    """
    BLIP Model for image captioning. The model consists of a vision encoder and a text decoder. One can optionally pass
    `input_ids` to the model, which serve as a text prompt, to make the text decoder continue the prompt. Otherwise,
    the decoder starts generating text from the [BOS] (beginning-of-sequence) token. will start generating the caption
    from the text input. If no text input is provided, the decoder will start with the [BOS] token only.
    """,
    BLIP_START_DOCSTRING,
)
class TFBlipForConditionalGeneration(TFBlipPreTrainedModel):
    """
    TFBlipForConditionalGeneration 类，继承自 TFBlipPreTrainedModel，用于图像字幕生成任务。

    Attributes:
        config_class (BlipConfig): 配置类为 BlipConfig。
        _keys_to_ignore_on_load_missing (list): 在加载时忽略的缺失键列表。
        main_input_name (str): 主要输入名称为 "pixel_values"。
    """

    config_class = BlipConfig
    _keys_to_ignore_on_load_missing = [r"text_decoder.cls.predictions.decoder.bias"]
    main_input_name = "pixel_values"

    def __init__(self, config: BlipConfig, *args, **kwargs):
        """
        初始化方法，接受 BlipConfig 类型的配置参数。

        Args:
            config (BlipConfig): BLIP 模型的配置参数。
            *args: 位置参数。
            **kwargs: 关键字参数。
        """
        super().__init__(config, *args, **kwargs)

        self.vision_model = TFBlipVisionModel(config.vision_config, name="vision_model")
        """
        vision_model 属性，TFBlipVisionModel 类型，使用 vision_config 初始化的视觉模型。

        Args:
            config.vision_config: 视觉配置参数。
            name (str): 模型名称为 "vision_model"。
        """

        self.text_decoder = TFBlipTextLMHeadModel(config.text_config, name="text_decoder")
        """
        text_decoder 属性，TFBlipTextLMHeadModel 类型，使用 text_config 初始化的文本解码器。

        Args:
            config.text_config: 文本配置参数。
            name (str): 模型名称为 "text_decoder"。
        """

        self.decoder_input_ids = config.text_config.bos_token_id
        """
        decoder_input_ids 属性，int 类型，表示文本解码器的起始标记 ID。

        Args:
            config.text_config.bos_token_id: 开始序列的标记 ID。
        """

        self.decoder_pad_token_id = config.text_config.pad_token_id
        """
        decoder_pad_token_id 属性，int 类型，表示文本解码器的填充标记 ID。

        Args:
            config.text_config.pad_token_id: 填充标记的 ID。
        """

    def get_input_embeddings(self) -> keras.layers.Layer:
        """
        获取输入嵌入层的方法。

        Returns:
            keras.layers.Layer: 返回视觉模型的 patch_embedding 层作为输入嵌入层。
        """
        return self.vision_model.embeddings.patch_embedding

    @unpack_inputs
    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBlipForConditionalGenerationModelOutput, config_class=BlipConfig)
    def call(
        self,
        pixel_values: tf.Tensor,
        input_ids: tf.Tensor | None = None,
        attention_mask: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: tf.Tensor | None = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = None,
        **kwargs
    ) -> TFBlipForConditionalGenerationModelOutput:
        """
        模型的前向传播方法，用于推断和训练。

        Args:
            pixel_values (tf.Tensor): 输入的像素值张量。
            input_ids (tf.Tensor, optional): 输入的文本 ID 张量。默认为 None。
            attention_mask (tf.Tensor, optional): 注意力掩码张量。默认为 None。
            output_attentions (bool, optional): 是否输出注意力。默认为 None。
            output_hidden_states (bool, optional): 是否输出隐藏状态。默认为 None。
            labels (tf.Tensor, optional): 标签张量。默认为 None。
            return_dict (bool, optional): 是否返回字典格式结果。默认为 None。
            training (bool, optional): 是否为训练模式。默认为 None。

        Returns:
            TFBlipForConditionalGenerationModelOutput: BLIP 条件生成模型的输出结果。
        """
        ) -> Union[Tuple, TFBlipForConditionalGenerationModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
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

        # 检查是否需要返回字典形式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 使用视觉模型处理输入的像素值，返回视觉特征
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 提取视觉特征的第一个输出，通常是图像嵌入
        image_embeds = vision_outputs[0]

        # 使用文本解码器生成文本输出
        outputs = self.text_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            labels=labels,
            return_dict=False,  # 强制不返回字典
            training=training,
        )

        # 如果不需要返回字典，则按预期输出格式返回结果元组
        if not return_dict:
            outputs = (outputs[0], outputs[1], image_embeds, vision_outputs[0]) + vision_outputs[2:]
            return tuple(output for output in outputs if output is not None)

        # 如果有标签，提取损失和逻辑回归结果
        if labels is not None:
            loss = outputs[0]
            logits = outputs[1]
        else:
            loss = None
            logits = outputs[0]

        # 如果存在损失并且其维度为0，则进行形状调整以保证一致性
        if loss is not None and loss.shape.rank == 0:
            loss = tf.reshape(loss, (1,))

        # 返回模型输出的命名元组，包括损失、逻辑回归结果、图像嵌入和视觉模型的隐藏状态等
        return TFBlipForConditionalGenerationModelOutput(
            loss=loss,
            logits=logits,
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
        ```python
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

        # 获取批次大小
        batch_size = pixel_values.shape[0]
        
        # 使用视觉模型处理输入图像，返回视觉输出
        vision_outputs = self.vision_model(pixel_values=pixel_values)

        # 从视觉输出中获取图像嵌入
        image_embeds = vision_outputs[0]

        # 创建图像注意力掩码，默认全为1，形状与图像嵌入维度相同
        image_attention_mask = tf.ones(shape_list(image_embeds)[:-1], dtype=tf.int32)

        # 如果输入的input_ids是列表，则转换为张量
        if isinstance(input_ids, list):
            input_ids = tf.convert_to_tensor(input_ids, dtype=tf.int32)
        # 如果input_ids为None，则使用默认的decoder输入ID和结束标记创建张量
        elif input_ids is None:
            input_ids = tf.convert_to_tensor(
                [[self.decoder_input_ids, self.config.text_config.eos_token_id]], dtype=tf.int32
            )
            # 扩展为与批次大小匹配的形状
            input_ids = tf.tile(input_ids, (batch_size, 1))

        # 添加起始标记到input_ids的开头，与PyTorch中的操作等效
        input_ids = tf.concat(
            [tf.ones((batch_size, 1), dtype=tf.int32) * self.config.text_config.bos_token_id, input_ids[:, 1:]], axis=1
        )
        
        # 调整attention_mask的长度，与输入序列长度相匹配
        attention_mask = attention_mask[:, :-1] if attention_mask is not None else None

        # 调用文本解码器的generate方法生成文本序列
        outputs = self.text_decoder.generate(
            input_ids=input_ids[:, :-1],
            eos_token_id=self.config.text_config.sep_token_id,
            pad_token_id=self.config.text_config.pad_token_id,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            **generate_kwargs,
        )

        # 返回生成的输出序列
        return outputs
    # 定义模型构建方法，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 设置标志位，表示模型已经构建
        self.built = True
        # 如果存在视觉模型，则构建视觉模型
        if getattr(self, "vision_model", None) is not None:
            # 使用视觉模型的名称作为命名空间
            with tf.name_scope(self.vision_model.name):
                # 构建视觉模型，传入空的输入形状
                self.vision_model.build(None)
        # 如果存在文本解码器，则构建文本解码器
        if getattr(self, "text_decoder", None) is not None:
            # 使用文本解码器的名称作为命名空间
            with tf.name_scope(self.text_decoder.name):
                # 构建文本解码器，传入空的输入形状
                self.text_decoder.build(None)
"""
BLIP Model for visual question answering. The model consists of a vision encoder, a text encoder as well as a text
decoder. The vision encoder will encode the input image, the text encoder will encode the input question together
with the encoding of the image, and the text decoder will output the answer to the question.
"""
# 导入所需的模块和函数装饰器
@add_start_docstrings(
    """
    BLIP Model for visual question answering. The model consists of a vision encoder, a text encoder as well as a text
    decoder. The vision encoder will encode the input image, the text encoder will encode the input question together
    with the encoding of the image, and the text decoder will output the answer to the question.
    """,
    BLIP_START_DOCSTRING,
)
# 继承自 TFBlipPreTrainedModel 类
class TFBlipForQuestionAnswering(TFBlipPreTrainedModel):
    # 使用 BlipConfig 类来配置模型
    config_class = BlipConfig
    # 在加载时忽略的关键字列表
    _keys_to_ignore_on_load_missing = [r"text_decoder.cls.predictions.decoder.bias"]

    # 模型初始化方法
    def __init__(self, config: BlipConfig, *args, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *args, **kwargs)

        # 创建视觉模型，使用 TFBlipVisionModel 类
        self.vision_model = TFBlipVisionModel(config.vision_config, name="vision_model")

        # 创建文本编码器，使用 TFBlipTextModel 类
        self.text_encoder = TFBlipTextModel(config.text_config, name="text_encoder", add_pooling_layer=False)

        # 创建文本解码器，使用 TFBlipTextLMHeadModel 类
        self.text_decoder = TFBlipTextLMHeadModel(config.text_config, name="text_decoder")

        # 解码器的填充标记 ID
        self.decoder_pad_token_id = config.text_config.pad_token_id
        # 解码器的起始标记 ID
        self.decoder_start_token_id = config.text_config.bos_token_id

    # 获取输入嵌入的方法
    def get_input_embeddings(self) -> keras.layers.Layer:
        # 返回视觉模型的补丁嵌入层
        return self.vision_model.embeddings.patch_embedding

    # 定义的方法来实现标记右移，类似于 transformers.models.t5.modeling_tf_t5.TFT5PreTrainedModel._shift_right 方法
    def _shift_right(self, input_ids):
        # 获取解码器的起始标记 ID 和填充标记 ID
        decoder_start_token_id = self.decoder_start_token_id
        pad_token_id = self.decoder_pad_token_id

        # 如果起始标记 ID 或填充标记 ID 未定义，则抛出 ValueError
        if decoder_start_token_id is None or pad_token_id is None:
            raise ValueError("decoder_start_token_id and pad_token_id must be defined!")

        # 创建起始标记序列，并确保与输入标记兼容的数据类型
        start_tokens = tf.fill((shape_list(input_ids)[0], 1), decoder_start_token_id)
        start_tokens = tf.cast(start_tokens, input_ids.dtype)  # 确保拼接时数据类型兼容
        # 将起始标记序列与输入标记序列右移一位进行拼接
        shifted_input_ids = tf.concat([start_tokens, input_ids[:, :-1]], -1)

        # 将标签中可能存在的 -100 值替换为填充标记 ID
        shifted_input_ids = tf.where(
            shifted_input_ids == -100,
            tf.cast(tf.fill(shape_list(shifted_input_ids), pad_token_id), shifted_input_ids.dtype),
            shifted_input_ids,
        )

        # 断言确保 `labels` 只包含正值和 -100
        tf.debugging.assert_greater_equal(shifted_input_ids, tf.constant(0, dtype=shifted_input_ids.dtype))

        return shifted_input_ids

    # 装饰器函数，用于将输入拆包并添加模型前向传播的文档字符串
    @unpack_inputs
    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    # 替换返回值文档字符串的装饰器函数
    @replace_return_docstrings(output_type=TFBlipTextVisionModelOutput, config_class=BlipVisionConfig)
    # 定义一个方法 `call`，用于执行模型推理或训练过程
    def call(
        self,
        input_ids: tf.Tensor,  # 输入文本的 token IDs，作为模型的输入
        pixel_values: tf.Tensor | None = None,  # 图像像素值，可选，用于图像输入模型
        decoder_input_ids: tf.Tensor | None = None,  # 解码器的输入 token IDs，可选
        decoder_attention_mask: tf.Tensor | None = None,  # 解码器的注意力遮罩，可选
        attention_mask: tf.Tensor | None = None,  # 注意力遮罩，控制模型哪些部分需要关注，可选
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选
        labels: tf.Tensor | None = None,  # 标签，用于模型的监督学习，可选
        return_dict: Optional[bool] = None,  # 是否以字典形式返回输出，可选
        training: Optional[bool] = None,  # 是否处于训练模式，可选
    ):
        # 定义一个方法 `generate`，用于生成模型输出（如文本生成）
        def generate(
            self,
            input_ids: tf.Tensor,  # 输入文本的 token IDs，作为生成器的输入
            pixel_values: tf.Tensor,  # 图像像素值，用于图像输入模型
            attention_mask: tf.Tensor | None = None,  # 注意力遮罩，控制模型哪些部分需要关注，可选
            **generate_kwargs,  # 其他生成参数，以字典形式传递
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
        ```
        """
        # 使用视觉模型处理输入图像，获取视觉输出
        vision_outputs = self.vision_model(pixel_values=pixel_values)

        # 提取图像嵌入表示
        image_embeds = vision_outputs[0]

        # 生成图像注意力掩码，形状与图像嵌入表示的前几维相同，最后一维是整数类型
        image_attention_mask = tf.ones(shape_list(image_embeds)[:-1], dtype=tf.int32)

        # 如果输入的input_ids是列表，则转换为Tensor类型
        if isinstance(input_ids, list):
            input_ids = tf.Tensor(input_ids)

        # 使用文本编码器处理输入文本序列，得到文本输出
        question_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=False,
        )

        # 提取问题嵌入表示
        question_embeds = question_outputs[0]

        # 生成问题的注意力掩码，形状与问题嵌入表示的前几维相同，最后一维是整数类型
        question_attention_mask = tf.ones(shape_list(question_embeds)[:-1], dtype=tf.int32)

        # 构造起始标记的Tensor，形状为(batch_size, 1)，值为self.decoder_start_token_id
        bos_ids = tf.fill(
            (tf.shape(question_embeds)[0], 1), value=tf.cast(self.decoder_start_token_id, input_ids.dtype)
        )

        # 使用文本解码器生成输出序列
        outputs = self.text_decoder.generate(
            input_ids=bos_ids,
            eos_token_id=self.config.text_config.sep_token_id,
            pad_token_id=self.config.text_config.pad_token_id,
            encoder_hidden_states=question_embeds,
            encoder_attention_mask=question_attention_mask,
            **generate_kwargs,
        )

        # 返回生成的输出序列
        return outputs
    # 定义神经网络层的构建方法，用于建立模型的输入形状
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回，避免重复构建
        if self.built:
            return
        # 设置标志位，表示模型已经构建
        self.built = True
        
        # 如果存在视觉模型，使用 TensorFlow 的命名空间来构建视觉模型
        if getattr(self, "vision_model", None) is not None:
            with tf.name_scope(self.vision_model.name):
                # 调用视觉模型的build方法，传入空输入形状
                self.vision_model.build(None)
        
        # 如果存在文本编码器，使用 TensorFlow 的命名空间来构建文本编码器
        if getattr(self, "text_encoder", None) is not None:
            with tf.name_scope(self.text_encoder.name):
                # 调用文本编码器的build方法，传入空输入形状
                self.text_encoder.build(None)
        
        # 如果存在文本解码器，使用 TensorFlow 的命名空间来构建文本解码器
        if getattr(self, "text_decoder", None) is not None:
            with tf.name_scope(self.text_decoder.name):
                # 调用文本解码器的build方法，传入空输入形状
                self.text_decoder.build(None)
"""
BLIP Model with a vision and text projector, and a classification head on top. The model is used in the context of
image-text retrieval. Given an image and a text, the model returns the probability of the text being relevant to
the image.
"""
# 继承自 TFBlipPreTrainedModel 的 BLIP 图像文本检索模型类
class TFBlipForImageTextRetrieval(TFBlipPreTrainedModel):
    # 使用 BlipConfig 类作为配置类
    config_class = BlipConfig

    def __init__(self, config: BlipConfig, *args, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *args, **kwargs)

        # 创建 BLIP 视觉模型，使用配置中的视觉配置
        self.vision_model = TFBlipVisionModel(config.vision_config, name="vision_model")

        # 创建 BLIP 文本编码器，使用配置中的文本配置，并禁用池化层
        self.text_encoder = TFBlipTextModel(config.text_config, name="text_encoder", add_pooling_layer=False)

        # 视觉投影层，用于将视觉特征投影到共享空间
        self.vision_proj = keras.layers.Dense(
            config.image_text_hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="vision_proj",
        )

        # 文本投影层，用于将文本特征投影到共享空间
        self.text_proj = keras.layers.Dense(
            config.image_text_hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="text_proj",
        )

        # 图像文本匹配头部，用于预测文本与图像相关性的概率
        self.itm_head = keras.layers.Dense(
            2, kernel_initializer=get_initializer(config.initializer_range), name="itm_head"
        )

        # 解码器的填充标记 ID，根据配置中的文本填充标记 ID 或解码器的开始标记 ID
        self.decoder_pad_token_id = (
            config.text_config.pad_token_id
            if not hasattr(config, "decoder_pad_token_id")
            else config.decoder_pad_token_id
        )
        self.decoder_start_token_id = (
            config.text_config.bos_token_id
            if not hasattr(config, "decoder_start_token_id")
            else config.decoder_start_token_id
        )
        self.config = config

    # 获取输入嵌入的方法，返回视觉模型的补丁嵌入层
    def get_input_embeddings(self) -> keras.layers.Layer:
        return self.vision_model.embeddings.patch_embedding

    # 调用方法，对输入数据进行前向传播
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
        # 其他参数用于模型前向传播，如像素值、注意力掩码、是否返回字典等
    ):
    # 构建方法，用于构造模型结构。如果已经构建过，直接返回。
    def build(self, input_shape=None):
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        
        # 如果存在视觉模型，使用视觉模型的名称作为命名空间，构建视觉模型
        if getattr(self, "vision_model", None) is not None:
            with tf.name_scope(self.vision_model.name):
                self.vision_model.build(None)
        
        # 如果存在文本编码器，使用文本编码器的名称作为命名空间，构建文本编码器
        if getattr(self, "text_encoder", None) is not None:
            with tf.name_scope(self.text_encoder.name):
                self.text_encoder.build(None)
        
        # 如果存在视觉投影层，使用视觉投影层的名称作为命名空间，构建视觉投影层
        if getattr(self, "vision_proj", None) is not None:
            with tf.name_scope(self.vision_proj.name):
                self.vision_proj.build([None, None, self.config.vision_config.hidden_size])
        
        # 如果存在文本投影层，使用文本投影层的名称作为命名空间，构建文本投影层
        if getattr(self, "text_proj", None) is not None:
            with tf.name_scope(self.text_proj.name):
                self.text_proj.build([None, None, self.config.text_config.hidden_size])
        
        # 如果存在itm_head，使用itm_head的名称作为命名空间，构建itm_head
        if getattr(self, "itm_head", None) is not None:
            with tf.name_scope(self.itm_head.name):
                self.itm_head.build([None, None, self.config.text_config.hidden_size])
```