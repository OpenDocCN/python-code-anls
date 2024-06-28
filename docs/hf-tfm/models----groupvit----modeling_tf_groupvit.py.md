# `.\models\groupvit\modeling_tf_groupvit.py`

```py
# coding=utf-8
# Copyright 2022 NVIDIA and The HuggingFace Team. All rights reserved.
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
""" TF 2.0 GroupViT model."""


from __future__ import annotations

import collections.abc
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling
from ...modeling_tf_utils import (
    TFModelInputType,
    TFPreTrainedModel,
    get_initializer,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_tensorflow_probability_available,
    logging,
    replace_return_docstrings,
)
from .configuration_groupvit import GroupViTConfig, GroupViTTextConfig, GroupViTVisionConfig


logger = logging.get_logger(__name__)

# soft dependency
if is_tensorflow_probability_available():
    try:
        import tensorflow_probability as tfp

        # On the first call, check whether a compatible version of TensorFlow is installed
        # TensorFlow Probability depends on a recent stable release of TensorFlow
        _ = tfp.distributions.Normal(loc=0.0, scale=1.0)
    except ImportError:
        logger.error(
            "GroupViT models are not usable since `tensorflow_probability` can't be loaded. "
            "It seems you have `tensorflow_probability` installed with the wrong tensorflow version."
            "Please try to reinstall it following the instructions here: https://github.com/tensorflow/probability."
        )

_CHECKPOINT_FOR_DOC = "nvidia/groupvit-gcc-yfcc"

TF_GROUPVIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "nvidia/groupvit-gcc-yfcc",
    # See all GroupViT models at https://huggingface.co/models?filter=groupvit
]


LARGE_NEGATIVE = -1e8


# Copied from transformers.models.bart.modeling_tf_bart._expand_mask
def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    # 获取输入 mask 的序列长度
    src_len = shape_list(mask)[1]
    # 如果未指定目标长度，则使用输入 mask 的序列长度
    tgt_len = tgt_len if tgt_len is not None else src_len
    # 创建一个常数张量，值为 1.0
    one_cst = tf.constant(1.0)
    # 将 mask 转换为常数张量类型
    mask = tf.cast(mask, dtype=one_cst.dtype)
    # 在第二个维度上扩展 mask，扩展后的形状为 `[bsz, 1, tgt_seq_len, src_seq_len]`
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))
    # 返回计算结果，其中 `one_cst - expanded_mask` 表示两个数的差
    # `LARGE_NEGATIVE` 与该差相乘，结果为一个较大的负数
    return (one_cst - expanded_mask) * LARGE_NEGATIVE
# 对比损失函数，从 https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html 适配而来
def contrastive_loss(logits: tf.Tensor) -> tf.Tensor:
    # 计算稀疏分类交叉熵损失的平均值，用于对比损失
    return tf.math.reduce_mean(
        # 使用稀疏分类交叉熵损失函数，计算 logits 的损失
        keras.metrics.sparse_categorical_crossentropy(
            y_true=tf.range(shape_list(logits)[0]), y_pred=logits, from_logits=True
        )
    )


# 从 transformers.models.clip.modeling_tf_clip.clip_loss 复制过来，将 clip 替换为 groupvit
def groupvit_loss(similarity: tf.Tensor) -> tf.Tensor:
    # 计算对比损失函数用于 caption 和 image
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(tf.transpose(similarity))
    return (caption_loss + image_loss) / 2.0


def hard_softmax(logits: tf.Tensor, dim: int) -> tf.Tensor:
    y_soft = stable_softmax(logits, dim)
    # 直通算法。
    index = tf.argmax(y_soft, dim)
    y_hard = tf.one_hot(
        index,
        depth=shape_list(logits)[dim],
        # TensorFlow 期望轴在 -1 或 [0, 3) 范围内，但接收到的是 -2
        # 因此使用以下代码片段。
        axis=range(len(shape_list(logits)))[dim],
        dtype=y_soft.dtype,
    )
    ret = y_hard - tf.stop_gradient(y_soft) + y_soft

    return ret


def gumbel_softmax(logits: tf.Tensor, tau: float = 1, hard: bool = False, dim: int = -1) -> tf.Tensor:
    gumbel_dist = tfp.distributions.Gumbel(0.0, 1.0)
    gumbels = gumbel_dist.sample(tf.shape(logits), dtype=logits.dtype)

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = stable_softmax(gumbels, dim)

    if hard:
        # 直通算法。
        index = tf.argmax(y_soft, dim)
        y_hard = tf.one_hot(
            index,
            depth=shape_list(logits)[dim],
            # TensorFlow 期望轴在 -1 或 [0, 3) 范围内，但接收到的是 -2
            # 因此使用以下代码片段。
            axis=range(len(shape_list(logits)))[dim],
            dtype=y_soft.dtype,
        )
        ret = y_hard - tf.stop_gradient(y_soft) + y_soft
    else:
        # 重参数化技巧。
        ret = y_soft
    return ret


def resize_attention_map(attentions: tf.Tensor, height: int, width: int, align_corners: bool = False) -> tf.Tensor:
    """
    Args:
        attentions (`tf.Tensor`): shape 为 [batch_size, groups, feat_height*feat_width] 的注意力图
        height (`int`): 输出注意力图的高度
        width (`int`): 输出注意力图的宽度
        align_corners (`bool`, *optional*): `nn.functional.interpolate` 的 `align_corner` 参数。

    Returns:
        `tf.Tensor`: shape 为 [batch_size, groups, height, width] 的调整大小后的注意力图
    """

    scale = (height * width // attentions.shape[2]) ** 0.5
    if height > width:
        feat_width = int(np.round(width / scale))
        feat_height = shape_list(attentions)[2] // feat_width
    else:
        feat_height = int(np.round(height / scale))
        feat_width = shape_list(attentions)[2] // feat_height
    # 获取注意力张量的批量大小
    batch_size = shape_list(attentions)[0]
    # 获取注意力张量中的群组数，这里指代群组标记的数量
    groups = shape_list(attentions)[1]  # number of group token
    # 将注意力张量重新形状为 [batch_size, groups, feat_height, feat_width]
    attentions = tf.reshape(attentions, (batch_size, groups, feat_height, feat_width))
    # 调整注意力张量的维度顺序，将维度重新排列为 [batch_size, groups, feat_height, feat_width]
    attentions = tf.transpose(attentions, perm=(0, 2, 3, 1))
    # 如果指定了align_corners为True，则使用双线性插值方式调整注意力张量大小
    if align_corners:
        attentions = tf.compat.v1.image.resize(
            attentions,
            size=(height, width),
            method="bilinear",
            align_corners=align_corners,
        )
    # 如果align_corners为False，则使用普通的双线性插值方式调整注意力张量大小
    else:
        attentions = tf.image.resize(attentions, size=(height, width), method="bilinear")
    # 再次调整注意力张量的维度顺序，将维度重新排列为 [batch_size, groups, height, width]
    attentions = tf.transpose(attentions, perm=(0, 3, 1, 2))
    # 返回调整后的注意力张量
    return attentions
# 从注意力图中获取分组信息的函数
def get_grouping_from_attentions(attentions: Tuple[tf.Tensor], hw_shape: Tuple[int]) -> tf.Tensor:
    """
    Args:
        attentions (`tuple(tf.Tensor)`): TFGroupViTVisionTransformer 返回的注意力图元组
        hw_shape (`tuple(int)`): 输出注意力图的高度和宽度
    Returns:
        `tf.Tensor`: 形状为 [batch_size, groups, height, width] 的注意力图
    """

    # 存储所有注意力图的列表
    attn_maps = []
    # 上一个注意力掩码的初始值设为 None
    prev_attn_masks = None
    # 遍历每一个注意力图
    for attn_masks in attentions:
        # 调整注意力掩码的维度顺序，[batch_size, num_groups, height x width] -> [batch_size, height x width, num_groups]
        attn_masks = tf.transpose(attn_masks, perm=(0, 2, 1))
        # 如果是第一个注意力图，则直接赋值给 prev_attn_masks
        if prev_attn_masks is None:
            prev_attn_masks = attn_masks
        else:
            # 否则，进行矩阵乘法操作
            prev_attn_masks = tf.matmul(prev_attn_masks, attn_masks)
        # 调整得到的注意力图的维度顺序，[batch_size, height x width, num_groups] -> [batch_size, num_groups, height x width]
        # 然后调整为 [batch_size, num_groups, height, width] 的形状
        cur_attn_map = resize_attention_map(tf.transpose(prev_attn_masks, perm=(0, 2, 1)), *hw_shape)
        # 将当前注意力图添加到列表中
        attn_maps.append(cur_attn_map)

    # 获取最终的分组注意力图，即最后一个注意力图
    final_grouping = attn_maps[-1]

    # 返回最终的分组注意力图，并停止梯度传播
    return tf.stop_gradient(final_grouping)
    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image (`tf.Tensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text (`tf.Tensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        segmentation_logits (`tf.Tensor` of shape `(batch_size, config.num_labels, logits_height, logits_width)`):
            Classification scores for each pixel.

            <Tip warning={true}>

            The logits returned do not necessarily have the same size as the `pixel_values` passed as inputs. This is
            to avoid doing two interpolations and lose some quality when a user needs to resize the logits to the
            original image size as post-processing. You should always check your logits shape and resize as needed.

            </Tip>

        text_embeds (`tf.Tensor` of shape `(batch_size, output_dim)`):
            The text embeddings obtained by applying the projection layer to the pooled output of
            [`TFGroupViTTextModel`].
        image_embeds (`tf.Tensor` of shape `(batch_size, output_dim)`):
            The image embeddings obtained by applying the projection layer to the pooled output of
            [`TFGroupViTVisionModel`].
        text_model_output (`TFBaseModelOutputWithPooling`):
            The output of the [`TFGroupViTTextModel`].
        vision_model_output (`TFBaseModelOutputWithPooling`):
            The output of the [`TFGroupViTVisionModel`].
    """

    # Initialize optional attributes with None
    loss: tf.Tensor | None = None
    logits_per_image: tf.Tensor = None
    logits_per_text: tf.Tensor = None
    segmentation_logits: tf.Tensor = None
    text_embeds: tf.Tensor = None
    image_embeds: tf.Tensor = None
    text_model_output: TFBaseModelOutputWithPooling = None
    vision_model_output: TFBaseModelOutputWithPooling = None

    # Define method to convert attributes to a tuple, excluding specific complex types
    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            # If key is not in exclusion list, return the attribute value; otherwise, convert complex type to tuple
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )
# 定义 TFGroupViTCrossAttentionLayer 类，继承自 keras.layers.Layer
class TFGroupViTCrossAttentionLayer(keras.layers.Layer):
    # 初始化方法，接受 GroupViTVisionConfig 类型的 config 参数和其他关键字参数
    def __init__(self, config: GroupViTVisionConfig, **kwargs):
        super().__init__(**kwargs)
        # 创建 TFGroupViTAttention 实例并赋给 self.attn 属性，名称为 "attn"
        self.attn = TFGroupViTAttention(config, name="attn")
        # 创建 LayerNormalization 实例并赋给 self.norm2 属性，使用 config 中的参数
        self.norm2 = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="norm2")
        # 创建 TFGroupViTMLP 实例并赋给 self.mlp 属性，名称为 "mlp"
        self.mlp = TFGroupViTMLP(config, name="mlp")
        # 创建 LayerNormalization 实例并赋给 self.norm_post 属性，使用 config 中的参数
        self.norm_post = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="norm_post")
        # 将 config 参数赋给 self.config 属性
        self.config = config

    # call 方法，定义层的正向传播逻辑，接受 query, key 和 training 参数，返回一个 tf.Tensor
    def call(self, query: tf.Tensor, key: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 将 query 赋给变量 x
        x = query
        # 使用 self.attn 对象处理 query 和 encoder_hidden_states=key，并加到 x 上
        x = x + self.attn(query, encoder_hidden_states=key)[0]
        # 使用 self.norm2 对 x 进行 LayerNormalization，然后传入 self.mlp 处理并加到 x 上
        x = x + self.mlp(self.norm2(x))
        # 对 x 使用 self.norm_post 进行 LayerNormalization
        x = self.norm_post(x)
        # 返回处理后的 x
        return x

    # build 方法，用于构建层，设置内部变量和子层的建立过程
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 检查并建立 self.attn 对象
        if getattr(self, "attn", None) is not None:
            with tf.name_scope(self.attn.name):
                self.attn.build(None)
        # 检查并建立 self.norm2 对象
        if getattr(self, "norm2", None) is not None:
            with tf.name_scope(self.norm2.name):
                self.norm2.build([None, None, self.config.hidden_size])
        # 检查并建立 self.mlp 对象
        if getattr(self, "mlp", None) is not None:
            with tf.name_scope(self.mlp.name):
                self.mlp.build(None)
        # 检查并建立 self.norm_post 对象
        if getattr(self, "norm_post", None) is not None:
            with tf.name_scope(self.norm_post.name):
                self.norm_post.build([None, None, self.config.hidden_size])


# 定义 TFGroupViTAssignAttention 类，继承自 keras.layers.Layer
class TFGroupViTAssignAttention(keras.layers.Layer):
    # 初始化方法，接受 GroupViTVisionConfig 类型的 config 参数和其他关键字参数
    def __init__(self, config: GroupViTVisionConfig, **kwargs):
        super().__init__(**kwargs)
        # 计算缩放因子，赋给 self.scale 属性
        self.scale = config.hidden_size**-0.5
        # 创建 Dense 层实例 q_proj，并赋给 self.q_proj 属性，名称为 "q_proj"
        self.q_proj = keras.layers.Dense(config.hidden_size, name="q_proj")
        # 创建 Dense 层实例 k_proj，并赋给 self.k_proj 属性，名称为 "k_proj"
        self.k_proj = keras.layers.Dense(config.hidden_size, name="k_proj")
        # 创建 Dense 层实例 v_proj，并赋给 self.v_proj 属性，名称为 "v_proj"
        self.v_proj = keras.layers.Dense(config.hidden_size, name="v_proj")
        # 创建 Dense 层实例 proj，并赋给 self.proj 属性，名称为 "proj"
        self.proj = keras.layers.Dense(config.hidden_size, name="proj")
        # 从 config 参数获取 assign_eps，并赋给 self.assign_eps 属性
        self.assign_eps = config.assign_eps
        # 将 config 参数赋给 self.config 属性
        self.config = config

    # get_attn 方法，接受 attn, gumbel, hard, training 参数，返回处理后的 attn tf.Tensor
    def get_attn(self, attn: tf.Tensor, gumbel: bool = True, hard: bool = True, training: bool = False) -> tf.Tensor:
        # 如果 gumbel 为真且在训练状态下，使用 gumbel_softmax 处理 attn
        if gumbel and training:
            attn = gumbel_softmax(attn, dim=-2, hard=hard)
        else:
            # 否则根据 hard 的值选择合适的 softmax 函数处理 attn
            if hard:
                attn = hard_softmax(attn, dim=-2)
            else:
                attn = stable_softmax(attn, axis=-2)

        # 返回处理后的 attn
        return attn
    # 实现注意力机制的调用函数，计算并返回注意力加权后的输出及软注意力分布

    def call(self, query: tf.Tensor, key: tf.Tensor, training: bool = False):
        # 将key作为value备份
        value = key
        # 对query进行投影操作，将其映射到指定维度空间
        query = self.q_proj(query)

        # 对key进行投影操作，将其映射到指定维度空间
        key = self.k_proj(key)

        # 对value进行投影操作，将其映射到指定维度空间
        value = self.v_proj(value)

        # 计算原始注意力分布，query与key的转置矩阵相乘，并乘以缩放因子
        raw_attn = tf.matmul(query, key, transpose_b=True) * self.scale

        # 根据原始注意力分布获取注意力权重，可选择性使用Gumbel-Softmax进行采样
        attn = self.get_attn(raw_attn, training=training)
        soft_attn = self.get_attn(raw_attn, training=training, gumbel=False, hard=False)

        # 归一化注意力权重，防止数值不稳定，加上一个小的常数eps
        attn = attn / (tf.math.reduce_sum(attn, axis=-1, keepdims=True) + self.assign_eps)

        # 根据注意力权重对value进行加权求和，得到最终输出
        out = tf.matmul(attn, value)

        # 对输出结果进行最终投影，映射到指定的维度空间
        out = self.proj(out)

        # 返回最终输出结果及soft_attn，即软注意力分布
        return out, soft_attn

    # 构建注意力层，设置各投影操作的维度
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        
        # 如果q_proj存在，则设置其输入形状
        if getattr(self, "q_proj", None) is not None:
            with tf.name_scope(self.q_proj.name):
                self.q_proj.build([None, None, self.config.hidden_size])
        
        # 如果k_proj存在，则设置其输入形状
        if getattr(self, "k_proj", None) is not None:
            with tf.name_scope(self.k_proj.name):
                self.k_proj.build([None, None, self.config.hidden_size])
        
        # 如果v_proj存在，则设置其输入形状
        if getattr(self, "v_proj", None) is not None:
            with tf.name_scope(self.v_proj.name):
                self.v_proj.build([None, None, self.config.hidden_size])
        
        # 如果proj存在，则设置其输入形状
        if getattr(self, "proj", None) is not None:
            with tf.name_scope(self.proj.name):
                self.proj.build([None, None, self.config.hidden_size])
class TFGroupViTTokenAssign(keras.layers.Layer):
    def __init__(self, config: GroupViTVisionConfig, num_group_token: int, num_output_group: int, **kwargs):
        super().__init__(**kwargs)
        self.num_output_group = num_output_group
        # 对 group_tokens 进行层归一化
        self.norm_tokens = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="norm_tokens")
        # 根据配置计算 MLP 的维度
        assign_mlp_ratio = (
            config.assign_mlp_ratio
            if isinstance(config.assign_mlp_ratio, collections.abc.Iterable)
            else (config.assign_mlp_ratio, config.assign_mlp_ratio)
        )
        tokens_dim, channels_dim = [int(x * config.hidden_size) for x in assign_mlp_ratio]
        # 创建用于中间 MLP 的层
        self.mlp_inter = TFGroupViTMixerMLP(config, num_group_token, tokens_dim, num_output_group, name="mlp_inter")
        # 对 post_tokens 进行层归一化
        self.norm_post_tokens = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="norm_post_tokens")
        # 对输入 x 进行层归一化
        self.norm_x = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="norm_x")
        # 创建用于前分配注意力的层
        self.pre_assign_attn = TFGroupViTCrossAttentionLayer(config, name="pre_assign_attn")
        # 创建分配注意力的层
        self.assign = TFGroupViTAssignAttention(config, name="assign")
        # 对新的 x 进行层归一化
        self.norm_new_x = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="norm_new_x")
        # 创建用于通道的 MLP 层
        self.mlp_channels = TFGroupViTMLP(
            config, config.hidden_size, channels_dim, config.hidden_size, name="mlp_channels"
        )
        self.config = config

    def project_group_token(self, group_tokens: tf.Tensor) -> tf.Tensor:
        """
        Args:
            group_tokens (tf.Tensor): group tokens, [batch_size, num_group_tokens, channels]

        Returns:
            projected_group_tokens (tf.Tensor): [batch_size, num_output_groups, channels]
        """
        # 使用中间 MLP 层对 group_tokens 进行投影
        projected_group_tokens = self.mlp_inter(group_tokens)
        # 对投影后的 group_tokens 进行层归一化
        projected_group_tokens = self.norm_post_tokens(projected_group_tokens)
        return projected_group_tokens

    def call(self, image_tokens: tf.Tensor, group_tokens: tf.Tensor, training: bool = False):
        """
        Args:
            image_tokens (`tf.Tensor`): image tokens, of shape [batch_size, input_length, channels]
            group_tokens (`tf.Tensor`): group tokens, [batch_size, num_group_tokens, channels]
        """

        # 对 group_tokens 进行层归一化
        group_tokens = self.norm_tokens(group_tokens)
        # 对 image_tokens 进行层归一化
        image_tokens = self.norm_x(image_tokens)
        # 使用中间 MLP 层对 group_tokens 进行投影，得到投影后的 group_tokens
        projected_group_tokens = self.project_group_token(group_tokens)
        # 使用前分配注意力层对投影后的 group_tokens 和 image_tokens 进行处理
        projected_group_tokens = self.pre_assign_attn(projected_group_tokens, image_tokens)
        # 使用分配注意力层对投影后的 group_tokens 和 image_tokens 进行分配
        new_image_tokens, attention = self.assign(projected_group_tokens, image_tokens)
        # 将投影后的 group_tokens 添加到新的 image_tokens 上
        new_image_tokens += projected_group_tokens

        # 对新的 image_tokens 进行层归一化和 MLP 通道操作
        new_image_tokens = new_image_tokens + self.mlp_channels(self.norm_new_x(new_image_tokens))

        return new_image_tokens, attention
    # 定义 build 方法，用于构建模型结构
    def build(self, input_shape=None):
        # 如果模型已经构建完成，直接返回，避免重复构建
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        
        # 如果存在 norm_tokens 属性，则构建与其相关的操作
        if getattr(self, "norm_tokens", None) is not None:
            # 使用 tf.name_scope 为 norm_tokens 创建命名空间
            with tf.name_scope(self.norm_tokens.name):
                # 使用 norm_tokens 属性构建操作，输入形状为 [None, None, self.config.hidden_size]
                self.norm_tokens.build([None, None, self.config.hidden_size])
        
        # 如果存在 mlp_inter 属性，则构建与其相关的操作
        if getattr(self, "mlp_inter", None) is not None:
            # 使用 tf.name_scope 为 mlp_inter 创建命名空间
            with tf.name_scope(self.mlp_inter.name):
                # 使用 mlp_inter 属性构建操作，输入形状为 None
                self.mlp_inter.build(None)
        
        # 如果存在 norm_post_tokens 属性，则构建与其相关的操作
        if getattr(self, "norm_post_tokens", None) is not None:
            # 使用 tf.name_scope 为 norm_post_tokens 创建命名空间
            with tf.name_scope(self.norm_post_tokens.name):
                # 使用 norm_post_tokens 属性构建操作，输入形状为 [None, None, self.config.hidden_size]
                self.norm_post_tokens.build([None, None, self.config.hidden_size])
        
        # 如果存在 norm_x 属性，则构建与其相关的操作
        if getattr(self, "norm_x", None) is not None:
            # 使用 tf.name_scope 为 norm_x 创建命名空间
            with tf.name_scope(self.norm_x.name):
                # 使用 norm_x 属性构建操作，输入形状为 [None, None, self.config.hidden_size]
                self.norm_x.build([None, None, self.config.hidden_size])
        
        # 如果存在 pre_assign_attn 属性，则构建与其相关的操作
        if getattr(self, "pre_assign_attn", None) is not None:
            # 使用 tf.name_scope 为 pre_assign_attn 创建命名空间
            with tf.name_scope(self.pre_assign_attn.name):
                # 使用 pre_assign_attn 属性构建操作，输入形状为 None
                self.pre_assign_attn.build(None)
        
        # 如果存在 assign 属性，则构建与其相关的操作
        if getattr(self, "assign", None) is not None:
            # 使用 tf.name_scope 为 assign 创建命名空间
            with tf.name_scope(self.assign.name):
                # 使用 assign 属性构建操作，输入形状为 None
                self.assign.build(None)
        
        # 如果存在 norm_new_x 属性，则构建与其相关的操作
        if getattr(self, "norm_new_x", None) is not None:
            # 使用 tf.name_scope 为 norm_new_x 创建命名空间
            with tf.name_scope(self.norm_new_x.name):
                # 使用 norm_new_x 属性构建操作，输入形状为 [None, None, self.config.hidden_size]
                self.norm_new_x.build([None, None, self.config.hidden_size])
        
        # 如果存在 mlp_channels 属性，则构建与其相关的操作
        if getattr(self, "mlp_channels", None) is not None:
            # 使用 tf.name_scope 为 mlp_channels 创建命名空间
            with tf.name_scope(self.mlp_channels.name):
                # 使用 mlp_channels 属性构建操作，输入形状为 None
                self.mlp_channels.build(None)
# Adapted from transformers.models.vit.modeling_tf_vit.TFViTPatchEmbeddings with ViT->GroupViT
# 这个类从 TFViTPatchEmbeddings 修改而来，用于 GroupViT 模型
class TFGroupViTPatchEmbeddings(keras.layers.Layer):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """
    
    def __init__(self, config: GroupViTConfig, **kwargs):
        super().__init__(**kwargs)
        # 从配置中获取图像大小和补丁大小
        image_size, patch_size = config.image_size, config.patch_size
        num_channels = config.num_channels
        # hidden_size 作为成员变量保存，因为在调用方法中会用到
        self.hidden_size = config.hidden_size
        
        # 如果图像大小和补丁大小不是可迭代对象，则转换为元组
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        # 计算图像中的补丁数量
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        
        # 保存图像大小、补丁大小、补丁数量、通道数和配置
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_channels = num_channels
        self.config = config
        
        # 创建投影层，用于将像素值投影到隐藏状态的大小
        self.projection = keras.layers.Conv2D(
            filters=self.hidden_size,                 # 输出通道数即隐藏状态的维度
            kernel_size=patch_size,                   # 卷积核大小设置为补丁大小
            strides=patch_size,                       # 步长设置为补丁大小，用于不重叠地提取补丁
            padding="valid",                          # 使用有效填充方式
            data_format="channels_last",              # 数据格式为通道在最后
            use_bias=True,                            # 使用偏置
            kernel_initializer=get_initializer(self.config.initializer_range),  # 使用指定初始化器初始化权重
            bias_initializer="zeros",                 # 偏置初始化为零
            name="projection",                        # 层的名称为 projection
        )

    def call(
        self, pixel_values: tf.Tensor, interpolate_pos_encoding: bool = False, training: bool = False
        # call 方法接收像素值张量、是否插值位置编码的标志和训练模式的标志
        # 未完整注释，仅包含方法签名部分，后续应继续添加注释
    ) -> tf.Tensor:
        # 从输入的像素值张量中获取批量大小、通道数、高度和宽度
        batch_size, num_channels, height, width = shape_list(pixel_values)
        # 如果在执行即时模式并且像素值的通道数与配置中设置的通道数不匹配，则引发值错误
        if tf.executing_eagerly() and num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 如果不插值位置编码且在执行即时模式下，输入图像的高度或宽度与模型配置的不匹配，则引发值错误
        if (
            not interpolate_pos_encoding
            and tf.executing_eagerly()
            and (height != self.image_size[0] or width != self.image_size[1])
        ):
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )

        # 当在 CPU 上运行时，`keras.layers.Conv2D` 不支持 `NCHW` 格式，因此将输入格式从 `NCHW` 转换为 `NHWC`
        # shape = (batch_size, in_height, in_width, in_channels=num_channels)
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))

        # 使用投影层对像素值进行投影
        projection = self.projection(pixel_values)

        # 将2D空间维度改为单个时间维度
        # shape = (batch_size, num_patches, out_channels=embed_dim)
        num_patches = (width // self.patch_size[1]) * (height // self.patch_size[0])

        # 在 TFGroupViTVisionEmbeddings 中，此层的嵌入将被层归一化处理
        # LayerNormalization 层需要具有静态的最后一个维度（否则在使用符号张量时会导致 test_keras_save_load 失败）
        # 这就是为什么在 reshape 方法中使用了 hidden_size
        embeddings = tf.reshape(tensor=projection, shape=(batch_size, num_patches, self.hidden_size))

        return embeddings

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果 projection 属性已存在，则构建 projection 层
        if getattr(self, "projection", None) is not None:
            with tf.name_scope(self.projection.name):
                self.projection.build([None, None, None, self.num_channels])
# Adapted from transformers.vit.modeling_tf_vit.TFViTEmbeddings
class TFGroupViTVisionEmbeddings(keras.layers.Layer):
    """
    Construct the position and patch embeddings.

    """

    def __init__(self, config: GroupViTVisionConfig, **kwargs):
        super().__init__(**kwargs)

        # 初始化补丁嵌入层对象
        self.patch_embeddings = TFGroupViTPatchEmbeddings(config, name="patch_embeddings")
        # 添加 dropout 层，使用配置中的 dropout 率
        self.dropout = keras.layers.Dropout(rate=config.dropout, name="dropout")
        # 添加 LayerNormalization 层，使用配置中的 epsilon 值
        self.layernorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")
        # 保存配置对象
        self.config = config

    def build(self, input_shape=None):
        # 获取补丁数量
        num_patches = self.patch_embeddings.num_patches
        # 添加位置嵌入权重，形状为 (1, num_patches, hidden_size)，使用零初始化
        self.position_embeddings = self.add_weight(
            shape=(1, num_patches, self.config.hidden_size),
            initializer="zeros",
            trainable=True,
            name="position_embeddings",
        )

        if self.built:
            return
        self.built = True
        # 如果已经构建，直接返回
        if getattr(self, "patch_embeddings", None) is not None:
            with tf.name_scope(self.patch_embeddings.name):
                self.patch_embeddings.build(None)
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)
        if getattr(self, "layernorm", None) is not None:
            with tf.name_scope(self.layernorm.name):
                # 构建 LayerNormalization 层，输入形状为 [None, None, hidden_size]
                self.layernorm.build([None, None, self.config.hidden_size])

    def interpolate_pos_encoding(self, embeddings, height, width) -> tf.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        # 获取 embeddings 的形状信息
        batch_size, num_patches, dim = shape_list(embeddings)
        # 获取位置编码的数量
        num_positions = shape_list(self.position_embeddings)[1]

        # 如果补丁数量与位置编码数量相等，并且高度与宽度也相等，则直接返回位置编码
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        # 否则，进行插值处理
        patch_pos_embed = self.position_embeddings
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        # 使用双三次插值方法调整位置编码的大小
        patch_pos_embed = tf.image.resize(
            images=tf.reshape(
                patch_pos_embed, shape=(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
            ),
            size=(h0, w0),
            method="bicubic",
        )
        patch_pos_embed = tf.reshape(tensor=patch_pos_embed, shape=(1, -1, dim))
        return patch_pos_embed

    def call(
        self, pixel_values: tf.Tensor, interpolate_pos_encoding: bool = False, training: bool = False
    ):
        # 神经网络层的调用方法，根据像素值计算输出
    # 定义函数的返回类型为 TensorFlow 张量
    ) -> tf.Tensor:
        # 从 pixel_values 的形状列表中获取高度和宽度信息
        _, _, height, width = shape_list(pixel_values)
        # 将像素值转换为补丁的嵌入向量，并根据需要插值位置编码
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        # 对嵌入向量进行层归一化处理
        embeddings = self.layernorm(embeddings)

        # 如果需要插值位置编码，则将其添加到每个 token 的嵌入向量中
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            # 否则，直接添加预定义的位置编码到嵌入向量中
            embeddings = embeddings + self.position_embeddings

        # 对嵌入向量应用 dropout 操作
        embeddings = self.dropout(embeddings)

        # 返回处理后的嵌入向量
        return embeddings
# 从 transformers.models.clip.modeling_tf_clip.TFCLIPTextEmbeddings 复制代码，修改为 GroupViT
class TFGroupViTTextEmbeddings(keras.layers.Layer):
    def __init__(self, config: GroupViTTextConfig, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = config.hidden_size  # 设置嵌入维度为配置文件中的隐藏大小

        self.config = config  # 保存配置信息

    def build(self, input_shape: tf.TensorShape = None):
        with tf.name_scope("token_embedding"):
            # 添加权重矩阵，形状为 (词汇表大小, 嵌入维度)，根据配置的初始化因子和范围进行初始化
            self.weight = self.add_weight(
                shape=(self.config.vocab_size, self.embed_dim),
                initializer=get_initializer(self.config.initializer_factor * self.config.initializer_range),
                trainable=True,
                name="weight",
            )

        with tf.name_scope("position_embedding"):
            # 添加位置嵌入矩阵，形状为 (最大位置嵌入数, 嵌入维度)，根据配置的初始化因子和范围进行初始化
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
            final_embeddings (`tf.Tensor`): 输出的嵌入张量。
        """
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")  # 抛出数值错误，要求指定 input_ids 或 inputs_embeds

        if inputs_embeds is None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)  # 检查嵌入索引是否在范围内
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)  # 根据输入的 input_ids 从权重中获取嵌入向量

        input_shape = shape_list(inputs_embeds)[:-1]  # 获取输入嵌入张量的形状

        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)  # 如果位置嵌入为空，则创建一个位置张量

        position_embeds = tf.gather(params=self.position_embedding, indices=position_ids)  # 根据位置索引获取位置嵌入向量
        position_embeds = tf.tile(input=position_embeds, multiples=(input_shape[0], 1, 1))  # 按指定倍数复制位置嵌入向量
        final_embeddings = inputs_embeds + position_embeds  # 最终的嵌入向量为输入嵌入向量加上位置嵌入向量

        return final_embeddings


class TFGroupViTStage(keras.layers.Layer):
    """这对应于 GroupViT 实现中的 `GroupingLayer` 类。"""

    def __init__(
        self,
        config: GroupViTVisionConfig,
        depth: int,
        num_prev_group_token: int,
        num_group_token: int,
        num_output_group: int,
        **kwargs,
    ):
        super().__init__(**kwargs)  # 调用父类的构造方法，传递任意关键字参数
        self.config = config  # 设置当前对象的config属性为传入的config参数
        self.depth = depth  # 设置当前对象的depth属性为传入的depth参数
        self.num_group_token = num_group_token  # 设置当前对象的num_group_token属性为传入的num_group_token参数
        self.layers = [TFGroupViTEncoderLayer(config, name=f"layers_._{i}") for i in range(depth)]  # 根据depth参数创建TFGroupViTEncoderLayer对象的列表，每个对象命名为layers_._{i}

        if num_group_token > 0:
            self.downsample = TFGroupViTTokenAssign(
                config=config,
                num_group_token=num_group_token,
                num_output_group=num_output_group,
                name="downsample",
            )  # 如果num_group_token大于0，则创建TFGroupViTTokenAssign对象赋值给self.downsample属性，使用传入的config、num_group_token、num_output_group参数
        else:
            self.downsample = None  # 否则将self.downsample属性设为None

        if num_prev_group_token > 0 and num_group_token > 0:
            self.group_projector = [
                keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="group_projector.0"),
                TFGroupViTMixerMLP(
                    config, num_prev_group_token, config.hidden_size // 2, num_group_token, name="group_projector.1"
                ),
            ]  # 如果num_prev_group_token和num_group_token均大于0，则创建包含LayerNormalization和TFGroupViTMixerMLP对象的列表，赋值给self.group_projector属性，使用传入的config、num_prev_group_token、config.hidden_size和num_group_token参数
        else:
            self.group_projector = None  # 否则将self.group_projector属性设为None

    def build(self, input_shape=None):
        if self.num_group_token > 0:
            self.group_token = self.add_weight(
                shape=(1, self.num_group_token, self.config.hidden_size),
                initializer="zeros",
                trainable=True,
                name="group_token",
            )  # 如果num_group_token大于0，则创建形状为(1, num_group_token, config.hidden_size)的可训练权重，赋值给self.group_token属性
        else:
            self.group_token = None  # 否则将self.group_token属性设为None

        if self.built:
            return  # 如果已经构建过，则直接返回
        self.built = True  # 标记已经构建过

        if getattr(self, "downsample", None) is not None:
            with tf.name_scope(self.downsample.name):
                self.downsample.build(None)  # 如果self.downsample不为None，则使用其名称作为作用域，在作用域内调用其build方法

        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)  # 遍历self.layers列表中的每个层对象，使用其名称作为作用域，在作用域内调用其build方法

        if getattr(self, "group_projector", None) is not None:
            with tf.name_scope(self.group_projector[0].name):
                self.group_projector[0].build([None, None, self.config.hidden_size])  # 如果self.group_projector不为None，则使用其第一个元素的名称作为作用域，在作用域内调用其build方法，传入形状为[None, None, config.hidden_size]的参数
            with tf.name_scope(self.group_projector[1].name):
                self.group_projector[1].build(None)  # 使用self.group_projector的第二个元素的名称作为作用域，在作用域内调用其build方法，不传入任何参数

    @property
    def with_group_token(self):
        return self.group_token is not None  # 返回self.group_token是否不为None的布尔值

    def split_x(self, x: tf.Tensor) -> tf.Tensor:
        if self.with_group_token:
            return x[:, : -self.num_group_token], x[:, -self.num_group_token :]  # 如果self.with_group_token为True，则返回x张量的前部分（去掉最后self.num_group_token列）和后部分（最后self.num_group_token列）
        else:
            return x, None  # 否则返回x张量和None

    def concat_x(self, x: tf.Tensor, group_token: tf.Tensor | None = None) -> tf.Tensor:
        if group_token is None:
            return x  # 如果group_token为None，则直接返回x张量
        return tf.concat([x, group_token], axis=1)  # 否则在axis=1的维度上连接x张量和group_token张量，并返回结果张量

    def call(
        self,
        hidden_states: tf.Tensor,
        prev_group_token: tf.Tensor | None = None,
        output_attentions: bool = False,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        """
        Args:
            hidden_states (`tf.Tensor`): 输入层的张量，形状为 `(batch, seq_len, embed_dim)`
            attention_mask (`tf.Tensor`): 注意力掩码，形状为 `(batch, 1, tgt_len, src_len)`，其中填充元素由极大负值指示。
            output_attentions (`bool`, *可选*):
                是否返回 Grouping block 的分组张量。
        """
        # 如果开启了 group token 功能
        if self.with_group_token:
            # 复制并展开 group token 到与 hidden_states 相同的形状
            group_token = tf.tile(self.group_token, multiples=(shape_list(hidden_states)[0], 1, 1))
            # 如果存在 group_projector，对 group token 应用每一层的 projector
            if self.group_projector is not None:
                for layer in self.group_projector:
                    prev_group_token = layer(prev_group_token)
                group_token = group_token + prev_group_token  # 将前一个 group token 添加到当前 group token
        else:
            group_token = None

        x = hidden_states

        # 将 hidden_states 和 group_token 进行连接
        cat_x = self.concat_x(x, group_token)
        # 遍历每一层并应用
        for layer in self.layers:
            layer_out = layer(
                cat_x,
                attention_mask=None,
                causal_attention_mask=None,
                output_attentions=None,
            )
            cat_x = layer_out[0]  # 更新 cat_x 到当前层的输出

        # 将 cat_x 拆分回原始的 hidden_states 和 group_token
        x, group_token = self.split_x(cat_x)

        attention = None
        # 如果存在 downsample 层，进行降采样操作
        if self.downsample is not None:
            x, attention = self.downsample(x, group_token)

        # 输出结果为 (x, group_token)，如果需要返回 attentions，则加入 attention
        outputs = (x, group_token)
        if output_attentions:
            outputs = outputs + (attention,)

        return outputs
class TFGroupViTMLP(keras.layers.Layer):
    # TFGroupViTMLP 类，继承自 keras.layers.Layer
    def __init__(
        self,
        config: GroupViTVisionConfig,
        hidden_size: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        output_size: Optional[int] = None,
        **kwargs,
    ):
        # 初始化函数，接受配置 config 和可选的隐藏大小、中间大小、输出大小等参数
        super().__init__(**kwargs)
        self.config = config
        # 获取激活函数
        self.activation_fn = get_tf_activation(config.hidden_act)
        # 设置隐藏大小，默认从配置中获取
        hidden_size = hidden_size if hidden_size is not None else config.hidden_size
        # 设置中间大小，默认从配置中获取
        intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        # 设置输出大小，默认为隐藏大小
        output_size = output_size if output_size is not None else hidden_size
        # 创建 Dense 层 fc1，用于中间层
        self.fc1 = keras.layers.Dense(intermediate_size, name="fc1")
        # 创建 Dense 层 fc2，用于输出层
        self.fc2 = keras.layers.Dense(output_size, name="fc2")
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 调用函数，传入隐藏状态 hidden_states 和训练标志 training
        # 将 hidden_states 输入到 fc1 中
        hidden_states = self.fc1(hidden_states)
        # 使用激活函数处理 fc1 输出
        hidden_states = self.activation_fn(hidden_states)
        # 将处理后的 hidden_states 输入到 fc2 中
        hidden_states = self.fc2(hidden_states)
        return hidden_states

    def build(self, input_shape=None):
        # 构建函数，在第一次调用时构建层的内部变量
        if self.built:
            return
        self.built = True
        # 构建 fc1 层
        if getattr(self, "fc1", None) is not None:
            with tf.name_scope(self.fc1.name):
                self.fc1.build([None, None, self.hidden_size])
        # 构建 fc2 层
        if getattr(self, "fc2", None) is not None:
            with tf.name_scope(self.fc2.name):
                self.fc2.build([None, None, self.intermediate_size])


class TFGroupViTMixerMLP(TFGroupViTMLP):
    # TFGroupViTMixerMLP 类，继承自 TFGroupViTMLP
    def call(self, x, training: bool = False):
        # 调用函数，传入输入 x 和训练标志 training
        # 调用父类 TFGroupViTMLP 的 call 方法，将 x 转置后输入
        x = super().call(hidden_states=tf.transpose(x, perm=(0, 2, 1)))
        # 返回转置后的结果
        return tf.transpose(x, perm=(0, 2, 1))


# Adapted from transformers.models.clip.modeling_tf_clip.TFCLIPAttention
class TFGroupViTAttention(keras.layers.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    # TFGroupViTAttention 类，继承自 keras.layers.Layer
    """来自《Attention Is All You Need》论文的多头注意力"""
    def __init__(self, config: GroupViTConfig, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = config.hidden_size  # 设置嵌入维度为配置中的隐藏大小
        self.num_attention_heads = config.num_attention_heads  # 设置注意力头的数量为配置中的注意力头数
        self.attention_head_size = self.embed_dim // self.num_attention_heads  # 计算每个注意力头的大小
        if self.attention_head_size * self.num_attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: "
                f"{self.num_attention_heads})."
            )

        factor = config.initializer_factor  # 从配置中获取初始化因子
        # 计算输入投影的标准差
        in_proj_std = (self.embed_dim**-0.5) * ((2 * config.num_hidden_layers) ** -0.5) * factor
        # 计算输出投影的标准差
        out_proj_std = (self.embed_dim**-0.5) * factor

        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)  # 计算注意力头大小的平方根

        # 初始化查询投影层，使用自定义的初始化器
        self.q_proj = keras.layers.Dense(
            units=self.embed_dim, kernel_initializer=get_initializer(in_proj_std), name="q_proj"
        )
        # 初始化键投影层，使用自定义的初始化器
        self.k_proj = keras.layers.Dense(
            units=self.embed_dim, kernel_initializer=get_initializer(in_proj_std), name="k_proj"
        )
        # 初始化值投影层，使用自定义的初始化器
        self.v_proj = keras.layers.Dense(
            units=self.embed_dim, kernel_initializer=get_initializer(in_proj_std), name="v_proj"
        )

        # 初始化 dropout 层，设定丢弃率为配置中的注意力丢弃率
        self.dropout = keras.layers.Dropout(rate=config.attention_dropout)

        # 初始化输出投影层，使用自定义的初始化器
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
        attention_mask: tf.Tensor = None,
        causal_attention_mask: tf.Tensor = None,
        output_attentions: bool = None,
        encoder_hidden_states: tf.Tensor = None,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        """Input shape: Batch x Time x Channel"""

        # 获取隐藏状态的批次大小
        batch_size = shape_list(hidden_states)[0]
        # 判断是否为跨注意力机制
        is_cross_attention = encoder_hidden_states is not None

        # 计算混合后的查询向量
        mixed_query_layer = self.q_proj(inputs=hidden_states)
        if is_cross_attention:
            # 若为跨注意力，计算混合后的键和值向量
            mixed_key_layer = self.k_proj(inputs=encoder_hidden_states)
            mixed_value_layer = self.v_proj(inputs=encoder_hidden_states)
        else:
            # 否则，计算混合后的键和值向量
            mixed_key_layer = self.k_proj(inputs=hidden_states)
            mixed_value_layer = self.v_proj(inputs=hidden_states)

        # 调整张量形状以进行注意力得分计算
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        # 计算注意力分数，即查询向量和键向量的点积
        # 结果维度为(batch size, num_heads, seq_len_q, seq_len_k)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        # 将注意力分数除以 sqrt(注意力头大小)
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, dk)

        # 先应用因果注意力掩码
        if causal_attention_mask is not None:
            # 加上因果注意力掩码（在 TFCLIPModel call() 函数中预先计算）
            attention_scores = tf.add(attention_scores, causal_attention_mask)

        # 若存在普通注意力掩码，则也应用
        if attention_mask is not None:
            # 加上普通注意力掩码（在 TFCLIPModel call() 函数中预先计算）
            attention_scores = tf.add(attention_scores, attention_mask)

        # 将注意力分数归一化为概率
        _attention_probs = stable_softmax(logits=attention_scores, axis=-1)

        # 对注意力概率进行 dropout 处理
        attention_probs = self.dropout(inputs=_attention_probs)

        # 计算注意力输出值
        attention_output = tf.matmul(attention_probs, value_layer)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])

        # 重新整形注意力输出张量的形状
        attention_output = tf.reshape(tensor=attention_output, shape=(batch_size, -1, self.embed_dim))

        # 通过输出投影层处理注意力输出
        attention_output = self.out_proj(attention_output)

        # 根据模型不同的输出设置，返回注意力输出和可能的注意力权重
        outputs = (attention_output, _attention_probs) if output_attentions else (attention_output,)

        return outputs
    # 构建方法用于构造模型，根据输入形状初始化模型的各个组件
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回，避免重复构建
        if self.built:
            return
        # 设置标志位为已构建
        self.built = True
        
        # 如果存在查询投影层，则构建查询投影层
        if getattr(self, "q_proj", None) is not None:
            # 在命名空间内构建查询投影层，输入形状为 [None, None, self.embed_dim]
            with tf.name_scope(self.q_proj.name):
                self.q_proj.build([None, None, self.embed_dim])
        
        # 如果存在键投影层，则构建键投影层
        if getattr(self, "k_proj", None) is not None:
            # 在命名空间内构建键投影层，输入形状为 [None, None, self.embed_dim]
            with tf.name_scope(self.k_proj.name):
                self.k_proj.build([None, None, self.embed_dim])
        
        # 如果存在值投影层，则构建值投影层
        if getattr(self, "v_proj", None) is not None:
            # 在命名空间内构建值投影层，输入形状为 [None, None, self.embed_dim]
            with tf.name_scope(self.v_proj.name):
                self.v_proj.build([None, None, self.embed_dim])
        
        # 如果存在输出投影层，则构建输出投影层
        if getattr(self, "out_proj", None) is not None:
            # 在命名空间内构建输出投影层，输入形状为 [None, None, self.embed_dim]
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.embed_dim])
# 从 transformers.models.clip.modeling_tf_clip.TFCLIPEncoderLayer 复制代码并改名为 TFGroupViTEncoderLayer，用于 GroupViT 模型
class TFGroupViTEncoderLayer(keras.layers.Layer):
    # 初始化函数，接受 GroupViTConfig 对象作为配置参数
    def __init__(self, config: GroupViTConfig, **kwargs):
        super().__init__(**kwargs)

        # 设定嵌入维度为隐藏大小
        self.embed_dim = config.hidden_size
        # 创建自注意力层，使用 TFGroupViTAttention 类，并命名为 "self_attn"
        self.self_attn = TFGroupViTAttention(config, name="self_attn")
        # 创建第一个层规范化层，使用 LayerNormalization，设定 epsilon 为 config.layer_norm_eps，命名为 "layer_norm1"
        self.layer_norm1 = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm1")
        # 创建 MLP 层，使用 TFGroupViTMLP 类，命名为 "mlp"
        self.mlp = TFGroupViTMLP(config, name="mlp")
        # 创建第二个层规范化层，使用 LayerNormalization，设定 epsilon 为 config.layer_norm_eps，命名为 "layer_norm2"
        self.layer_norm2 = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm2")

    # 调用函数，实现层的前向传播逻辑
    def call(
        self,
        hidden_states: tf.Tensor,               # 输入的隐藏状态张量，形状为 `(batch, seq_len, embed_dim)`
        attention_mask: tf.Tensor,              # 注意力掩码张量，形状为 `(batch, 1, tgt_len, src_len)`
        causal_attention_mask: tf.Tensor,       # 因果注意力掩码张量，形状为 `(batch, 1, tgt_len, src_len)`
        output_attentions: bool,                # 是否输出所有注意力层的注意力张量
        training: bool = False,                 # 是否处于训练模式
    ) -> Tuple[tf.Tensor]:                     # 返回类型为包含一个张量的元组
        """
        Args:
            hidden_states (`tf.Tensor`): 输入层的隐藏状态，形状为 `(batch, seq_len, embed_dim)`
            attention_mask (`tf.Tensor`): 注意力掩码，形状为 `(batch, 1, tgt_len, src_len)`
                其中填充元素由非常大的负值表示。
            causal_attention_mask (`tf.Tensor`): 因果注意力掩码，形状为 `(batch, 1, tgt_len, src_len)`
                其中填充元素由非常大的负值表示。
            output_attentions (`bool`):
                是否返回所有注意力层的注意力张量。查看返回的 `outputs` 中的详细信息。
        """
        residual = hidden_states

        # 应用第一个层规范化层
        hidden_states = self.layer_norm1(inputs=hidden_states)
        # 使用 self_attn 进行自注意力计算
        attention_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            training=training,
        )
        # 取自注意力输出的第一个张量作为新的隐藏状态
        hidden_states = attention_outputs[0]
        # 添加残差连接
        hidden_states = residual + hidden_states

        residual = hidden_states
        # 应用第二个层规范化层
        hidden_states = self.layer_norm2(inputs=hidden_states)
        # 应用 MLP 层
        hidden_states = self.mlp(hidden_states=hidden_states)
        # 添加残差连接
        hidden_states = residual + hidden_states

        # 如果需要输出注意力张量，则将其添加到输出中
        outputs = (hidden_states,) + attention_outputs[1:]  # 如果输出注意力张量，则添加它们

        return outputs
    # 构建神经网络层次结构。如果已经构建过，则直接返回。
    def build(self, input_shape=None):
        if self.built:
            return
        # 标记该层次已经构建
        self.built = True
        # 如果存在 self_attn 属性，则构建 self_attn 层
        if getattr(self, "self_attn", None) is not None:
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)
        # 如果存在 layer_norm1 属性，则构建 layer_norm1 层
        if getattr(self, "layer_norm1", None) is not None:
            with tf.name_scope(self.layer_norm1.name):
                self.layer_norm1.build([None, None, self.embed_dim])
        # 如果存在 mlp 属性，则构建 mlp 层
        if getattr(self, "mlp", None) is not None:
            with tf.name_scope(self.mlp.name):
                self.mlp.build(None)
        # 如果存在 layer_norm2 属性，则构建 layer_norm2 层
        if getattr(self, "layer_norm2", None) is not None:
            with tf.name_scope(self.layer_norm2.name):
                self.layer_norm2.build([None, None, self.embed_dim])
# 从 transformers.models.clip.modeling_tf_clip.TFGroupViTTextEncoder 适配而来的自定义层
class TFGroupViTTextEncoder(keras.layers.Layer):
    def __init__(self, config: GroupViTTextConfig, **kwargs):
        super().__init__(**kwargs)

        # 初始化多层 TFGroupViTEncoderLayer，根据配置创建指定数量的编码器层
        self.layers = [TFGroupViTEncoderLayer(config, name=f"layers_._{i}") for i in range(config.num_hidden_layers)]

    def call(
        self,
        hidden_states,               # 输入的隐藏状态张量
        attention_mask: tf.Tensor,   # 注意力掩码张量
        causal_attention_mask: tf.Tensor,  # 因果注意力掩码张量
        output_attentions: bool,     # 是否输出注意力矩阵
        output_hidden_states: bool,  # 是否输出隐藏状态
        return_dict: bool,           # 是否返回字典格式的输出
        training: bool = False,      # 是否处于训练模式
    ) -> Union[Tuple, TFBaseModelOutput]:
        # 如果需要输出隐藏状态，则初始化空元组存储编码器状态
        encoder_states = () if output_hidden_states else None
        # 如果需要输出注意力矩阵，则初始化空元组存储注意力矩阵
        all_attentions = () if output_attentions else None

        # 遍历每个编码器层进行前向传播
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                # 如果需要输出隐藏状态，将当前隐藏状态添加到状态元组中
                encoder_states = encoder_states + (hidden_states,)

            # 调用编码器层的前向传播，获取层的输出
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                causal_attention_mask,
                output_attentions=output_attentions,
            )
            # 更新隐藏状态为编码器层的输出的第一个元素
            hidden_states = layer_outputs[0]

            if output_attentions:
                # 如果需要输出注意力矩阵，将当前层的注意力矩阵添加到 all_attentions 元组中
                all_attentions = all_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，将最终的隐藏状态添加到状态元组中
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        # 如果不需要返回字典格式的输出，根据需要返回隐藏状态、编码器状态和注意力矩阵
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        # 否则，返回 TFBaseModelOutput 对象，包含最终的隐藏状态、编码器状态和注意力矩阵
        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 self.layers，则遍历每个层并构建它们
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)


# TFGroupViTVisionEncoder 类
class TFGroupViTVisionEncoder(keras.layers.Layer):
    def __init__(self, config: GroupViTVisionConfig, **kwargs) -> None:
        super().__init__(**kwargs)

        # 初始化多个 TFGroupViTStage，根据配置创建多个视觉编码阶段
        self.stages = [
            TFGroupViTStage(
                config=config,
                depth=config.depths[i],
                num_group_token=config.num_group_tokens[i],
                num_output_group=config.num_output_groups[i],
                num_prev_group_token=config.num_output_groups[i - 1] if i > 0 else 0,
                name=f"stages_._{i}",
            )
            for i in range(len(config.depths))
        ]

    def call(
        self,
        hidden_states: tf.Tensor,    # 输入的隐藏状态张量
        output_hidden_states: bool,  # 是否输出隐藏状态
        output_attentions: bool,     # 是否输出注意力矩阵
        return_dict: bool,           # 是否返回字典格式的输出
        training: bool = False,      # 是否处于训练模式

    ) -> None:
        # 遍历每个视觉编码阶段进行前向传播
        for stage in self.stages:
            # 调用每个阶段的前向传播函数
            hidden_states = stage(
                hidden_states,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                return_dict=return_dict,
            )

        # 返回最终的隐藏状态张量作为视觉编码器的输出
        return hidden_states
    ) -> Union[tuple, TFBaseModelOutput]:
        # 如果输出隐藏状态，则初始化一个空元组，否则设为 None
        all_hidden_states = () if output_hidden_states else None
        # 如果输出注意力权重，则初始化一个空元组，否则设为 None
        all_groupings = () if output_attentions else None

        # 初始化 group_tokens 为 None
        group_tokens = None

        # 遍历 self.stages 中的每个阶段
        for stage in self.stages:
            # 如果输出隐藏状态，则将当前隐藏状态加入到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 调用当前阶段的处理函数，获取当前层的输出
            layer_outputs = stage(hidden_states, group_tokens, output_attentions)

            # 更新隐藏状态为当前层输出的第一个元素
            hidden_states = layer_outputs[0]
            # 更新 group_tokens 为当前层输出的第二个元素
            group_tokens = layer_outputs[1]

            # 如果输出注意力权重且当前层有有效的注意力权重输出，则将其加入 all_groupings 中
            if output_attentions and layer_outputs[2] is not None:
                all_groupings = all_groupings + (layer_outputs[2],)

        # 如果输出隐藏状态，则将最终隐藏状态加入 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不要求返回字典形式的输出，则按顺序返回非空的结果元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_groupings] if v is not None)
        
        # 如果需要返回字典形式的输出，则构造 TFBaseModelOutput 对象返回
        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_groupings
        )

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        
        # 标记当前模型已经构建
        self.built = True
        
        # 如果模型已经定义了 stages 属性，则对每个层进行构建
        if getattr(self, "stages", None) is not None:
            for layer in self.stages:
                # 使用当前层的名称为其创建命名空间，并进行构建
                with tf.name_scope(layer.name):
                    layer.build(None)
# 从 transformers.models.clip.modeling_tf_clip.TFCLIPTextTransformer 复制的代码，将 CLIPText 改为 GroupViTText，将 CLIPEncoder 改为 GroupViTTextEncoder
class TFGroupViTTextTransformer(keras.layers.Layer):
    def __init__(self, config: GroupViTTextConfig, **kwargs):
        super().__init__(**kwargs)

        # 初始化 GroupViTTextEmbeddings 层，用于处理输入文本的嵌入表示
        self.embeddings = TFGroupViTTextEmbeddings(config, name="embeddings")
        
        # 初始化 GroupViTTextEncoder 层，用于对嵌入表示进行编码得到输出特征
        self.encoder = TFGroupViTTextEncoder(config, name="encoder")
        
        # 最终的层归一化，用于规范化最终输出的特征向量
        self.final_layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="final_layer_norm")

        # 用于计算 `pooled_output` 的相关属性
        self.eos_token_id = config.eos_token_id  # EOS（结束符）的 token ID
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
        # 函数的参数：input_ids 输入的模型输入，attention_mask 注意力掩码，position_ids 位置编码，output_attentions 是否输出注意力权重，output_hidden_states 是否输出隐藏状态，return_dict 是否返回字典格式的输出
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        # 获取输入 `input_ids` 的形状信息
        input_shape = shape_list(input_ids)

        # 使用输入的 `input_ids` 和 `position_ids` 作为参数，调用嵌入层对象 `self.embeddings` 进行嵌入操作
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # 从输入形状信息中提取批大小和序列长度
        batch_size, seq_length = input_shape
        
        # CLIP 文本模型使用因果掩码，在这里准备它
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self._build_causal_attention_mask(batch_size, seq_length, dtype=embedding_output.dtype)

        # 检查注意力掩码并扩展其维度
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        attention_mask = _expand_mask(attention_mask)

        # 调用编码器 `self.encoder`，传入嵌入输出、注意力掩码等参数，并接收编码器的输出
        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 从编码器输出中提取序列输出
        sequence_output = encoder_outputs[0]

        # 对序列输出进行最终层归一化处理
        sequence_output = self.final_layer_norm(inputs=sequence_output)

        # 如果 `self.eos_token_id` 等于 2
        if self.eos_token_id == 2:
            # 如果 `eos_token_id` 在 PR #24773 之前是错误的，保持之前的操作
            # 对 `sequence_output` 进行聚合操作，选择每个序列中最高数值的位置作为池化输出
            pooled_output = tf.gather_nd(
                params=sequence_output,
                indices=tf.stack(
                    values=(tf.range(input_shape[0], dtype=tf.int64), tf.math.argmax(input_ids, axis=-1)), axis=1
                ),
            )
        else:
            # 如果 `eos_token_id` 在 PR #24773 中被更新了，允许额外新标记的使用
            # 对 `sequence_output` 进行聚合操作，选择每个序列中 `self.eos_token_id` 的位置作为池化输出
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

        # 如果不返回字典形式的结果，则返回元组形式的输出
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        # 返回 TFBaseModelOutputWithPooling 对象，包含序列输出、池化输出、编码器的隐藏状态和注意力权重
        return TFBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    # 构建因果注意力掩码，用于自注意力机制
    def _build_causal_attention_mask(self, batch_size, seq_length, dtype=tf.float32):
        # 如果 seq_length 是运行时值，不能用 tf.constant 处理。根据 TensorFlow 文档，tf.fill 可处理动态形状：
        # https://www.tensorflow.org/api_docs/python/tf/fill
        diag = tf.cast(tf.fill((seq_length,), 0.0), dtype)

        # 创建一个二维加性注意力掩码，所有位置均被掩盖
        to_mask = tf.cast(tf.fill((seq_length, seq_length), -10000.0), dtype)

        # 将二维矩阵的对角线及其以下三角部分设置为 0，即不被掩盖的位置
        # 提示：将二维矩阵视为 (query_seq, key_seq) 的空间
        to_mask = tf.linalg.band_part(to_mask, 0, -1)
        # to_mask = tf.linalg.band_part(to_mask, -1, 0)  # 这行代码是注释掉的备选方案
        to_mask = tf.linalg.set_diag(to_mask, diagonal=diag)

        return tf.broadcast_to(input=to_mask, shape=(batch_size, 1, seq_length, seq_length))

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.embed_dim])
# Adapted from transformers.models.clip.modeling_tf_clip.TFCLIPVisionTransformer
# 自transformers库中TFCLIPVisionTransformer模块改编而来，用于处理视觉信息的Transformer模型

class TFGroupViTVisionTransformer(keras.layers.Layer):
    def __init__(self, config: GroupViTVisionConfig, **kwargs):
        super().__init__(**kwargs)

        # 初始化视觉嵌入层对象，使用GroupViTVisionEmbeddings处理视觉嵌入
        self.embeddings = TFGroupViTVisionEmbeddings(config, name="embeddings")
        
        # 初始化视觉编码器对象，使用GroupViTVisionEncoder处理视觉编码
        self.encoder = TFGroupViTVisionEncoder(config, name="encoder")
        
        # 初始化层归一化对象，设置epsilon值为config中定义的层归一化epsilon值
        self.layernorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")
        
        # 设置嵌入维度为config中定义的隐藏层大小
        self.embed_dim = config.hidden_size

    # 定义模型调用方法，接收像素值和其他配置参数，并返回模型输出
    def call(
        self,
        pixel_values: TFModelInputType,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    ) -> Union[Tuple, TFBaseModelOutputWithPooling]:
        # 获取嵌入层的输出，即像素值的嵌入表示
        embedding_output = self.embeddings(pixel_values)

        # 将嵌入输出传入编码器，获取编码器的输出
        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        # 获取编码器输出的最后隐藏状态
        last_hidden_state = encoder_outputs[0]

        # 对最后隐藏状态进行层归一化处理
        last_hidden_state = self.layernorm(last_hidden_state)

        # 计算池化输出，通过对最后隐藏状态在第1维度上求均值得到
        pooled_output = tf.math.reduce_mean(last_hidden_state, axis=1)

        # 如果不需要返回字典形式的结果，则返回元组形式的输出
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 如果需要返回字典形式的结果，则构建TFBaseModelOutputWithPooling对象返回
        return TFBaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    # 构建方法，在首次调用时构建嵌入层、编码器和层归一化对象
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        
        # 如果已定义嵌入层，则使用tf.name_scope构建嵌入层
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        
        # 如果已定义编码器，则使用tf.name_scope构建编码器
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        
        # 如果已定义层归一化对象，则使用tf.name_scope构建层归一化对象
        if getattr(self, "layernorm", None) is not None:
            with tf.name_scope(self.layernorm.name):
                self.layernorm.build([None, None, self.embed_dim])


@keras_serializable
# 从transformers.models.clip.modeling_tf_clip.TFCLIPTextMainLayer复制而来，将CLIP替换为GroupViT
# 从transformers库中TFCLIPTextMainLayer模块复制而来，修改为处理GroupViT的文本主层

class TFGroupViTTextMainLayer(keras.layers.Layer):
    config_class = GroupViTTextConfig

    def __init__(self, config: GroupViTTextConfig, **kwargs):
        super().__init__(**kwargs)
        
        # 初始化配置对象
        self.config = config
        
        # 初始化文本模型对象，使用TFGroupViTTextTransformer处理文本信息
        self.text_model = TFGroupViTTextTransformer(config, name="text_model")

    # 获取输入嵌入层对象的方法，返回文本模型的嵌入层对象
    def get_input_embeddings(self) -> keras.layers.Layer:
        return self.text_model.embeddings

    # 设置输入嵌入层对象的方法，设置文本模型的嵌入层权重和词汇大小
    def set_input_embeddings(self, value: tf.Variable):
        self.text_model.embeddings.weight = value
        self.text_model.embeddings.vocab_size = shape_list(value)[0]

    # 对输入参数进行解包处理的装饰器
    @unpack_inputs
    # 定义一个方法 `call`，用于执行模型的前向传播
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
        # 如果 `input_ids` 为空，则抛出数值错误
        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        # 获取 `input_ids` 的形状
        input_shape = shape_list(input_ids)

        # 如果 `attention_mask` 为空，则创建一个形状与 `input_shape` 相同的张量，填充值为 1
        if attention_mask is None:
            attention_mask = tf.fill(dims=input_shape, value=1)

        # 调用 `text_model` 进行文本模型的前向传播，并传递相应的参数
        text_model_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 返回文本模型的输出结果
        return text_model_outputs

    # 定义一个方法 `build`，用于构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        # 标记模型已构建
        self.built = True
        # 如果存在 `text_model` 属性，则在其命名空间内构建模型
        if getattr(self, "text_model", None) is not None:
            with tf.name_scope(self.text_model.name):
                self.text_model.build(None)
# 使用 keras_serializable 装饰器使该类可以序列化为 Keras 模型
@keras_serializable
# 从 transformers.models.clip.modeling_tf_clip.TFCLIPVisionMainLayer 复制而来，并将 CLIP 改为 GroupViT
class TFGroupViTVisionMainLayer(keras.layers.Layer):
    # 指定配置类为 GroupViTVisionConfig
    config_class = GroupViTVisionConfig

    def __init__(self, config: GroupViTVisionConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        # 创建 TFGroupViTVisionTransformer 模型，并命名为 vision_model
        self.vision_model = TFGroupViTVisionTransformer(config, name="vision_model")

    # 返回 vision_model 的 embeddings 层作为输入嵌入
    def get_input_embeddings(self) -> keras.layers.Layer:
        return self.vision_model.embeddings

    # 对输入进行解包，并调用 vision_model 进行前向传播
    @unpack_inputs
    def call(
        self,
        pixel_values: TFModelInputType | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        # 如果 pixel_values 为 None，则抛出数值错误
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 调用 vision_model 进行前向传播，并返回其输出
        vision_model_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        return vision_model_outputs

    # 构建层次结构，如果已经构建过，则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果 vision_model 存在，则在其命名空间下构建模型
        if getattr(self, "vision_model", None) is not None:
            with tf.name_scope(self.vision_model.name):
                self.vision_model.build(None)


# 使用 keras_serializable 装饰器使该类可以序列化为 Keras 模型
@keras_serializable
# 从 transformers.models.clip.modeling_tf_clip.TFCLIPMainLayer 改编而来
class TFGroupViTMainLayer(keras.layers.Layer):
    # 指定配置类为 GroupViTConfig
    config_class = GroupViTConfig
    # 初始化方法，接受一个配置对象 config 和其他关键字参数
    def __init__(self, config: GroupViTConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 检查 config.text_config 是否为 GroupViTTextConfig 类型，否则引发 ValueError 异常
        if not isinstance(config.text_config, GroupViTTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type GroupViTTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        # 检查 config.vision_config 是否为 GroupViTVisionConfig 类型，否则引发 ValueError 异常
        if not isinstance(config.vision_config, GroupViTVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type GroupViTVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        # 将传入的 config 对象赋值给实例变量 self.config
        self.config = config

        # 从 config 对象中获取 text_config 和 vision_config 对象，并分别赋值给 text_config 和 vision_config 变量
        text_config = config.text_config
        vision_config = config.vision_config

        # 设置实例变量，分别为投影维度和投影中间维度
        self.projection_dim = config.projection_dim
        self.projection_intermediate_dim = config.projection_intermediate_dim
        # 设置文本嵌入维度和视觉嵌入维度
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        # 创建 TFGroupViTTextTransformer 对象，用于文本模型的转换，命名为 "text_model"
        self.text_model = TFGroupViTTextTransformer(text_config, name="text_model")
        # 创建 TFGroupViTVisionTransformer 对象，用于视觉模型的转换，命名为 "vision_model"
        self.vision_model = TFGroupViTVisionTransformer(vision_config, name="vision_model")

        # 定义视觉投影层，包括 Dense 层、BatchNormalization 层和 ReLU 激活函数层
        self.visual_projection = [
            keras.layers.Dense(self.projection_intermediate_dim, name="visual_projection.0"),
            keras.layers.BatchNormalization(name="visual_projection.1", momentum=0.9, epsilon=1e-5),
            keras.layers.ReLU(name="visual_projection.2"),
            keras.layers.Dense(self.projection_dim, name="visual_projection.3"),
        ]
        # 定义文本投影层，包括 Dense 层、BatchNormalization 层和 ReLU 激活函数层
        self.text_projection = [
            keras.layers.Dense(self.projection_intermediate_dim, name="text_projection.0"),
            keras.layers.BatchNormalization(name="text_projection.1", momentum=0.9, epsilon=1e-5),
            keras.layers.ReLU(name="text_projection.2"),
            keras.layers.Dense(self.projection_dim, name="text_projection.3"),
        ]
    def build(self, input_shape=None):
        # 添加一个可训练的名为logit_scale的权重，初始值为config中的logit_scale_init_value
        self.logit_scale = self.add_weight(
            shape=(1,),
            initializer=keras.initializers.Constant(self.config.logit_scale_init_value),
            trainable=True,
            name="logit_scale",
        )

        # 如果模型已经建立，则直接返回
        if self.built:
            return
        # 标记模型已经建立
        self.built = True
        
        # 如果存在text_model，则构建text_model
        if getattr(self, "text_model", None) is not None:
            with tf.name_scope(self.text_model.name):
                self.text_model.build(None)
        
        # 如果存在vision_model，则构建vision_model
        if getattr(self, "vision_model", None) is not None:
            with tf.name_scope(self.vision_model.name):
                self.vision_model.build(None)
        
        # 如果存在visual_projection，则分别构建其各个层
        if getattr(self, "visual_projection", None) is not None:
            with tf.name_scope(self.visual_projection[0].name):
                self.visual_projection[0].build([None, None, None, self.vision_embed_dim])
            with tf.name_scope(self.visual_projection[1].name):
                self.visual_projection[1].build((None, self.projection_intermediate_dim))
            with tf.name_scope(self.visual_projection[3].name):
                self.visual_projection[3].build([None, None, None, self.projection_intermediate_dim])
        
        # 如果存在text_projection，则分别构建其各个层
        if getattr(self, "text_projection", None) is not None:
            with tf.name_scope(self.text_projection[0].name):
                self.text_projection[0].build([None, None, None, self.text_embed_dim])
            with tf.name_scope(self.text_projection[1].name):
                self.text_projection[1].build((None, self.projection_intermediate_dim))
            with tf.name_scope(self.text_projection[3].name):
                self.text_projection[3].build([None, None, None, self.projection_intermediate_dim])

    @unpack_inputs
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
        # 如果未提供input_ids，则抛出数值错误异常
        if input_ids is None:
            raise ValueError("You have to specify either input_ids")

        # 获取input_ids的形状
        input_shape = shape_list(input_ids)

        # 如果未提供attention_mask，则使用全1填充
        if attention_mask is None:
            attention_mask = tf.fill(dims=input_shape, value=1)

        # 使用text_model处理输入，获取文本输出
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 从文本输出中获取汇总的输出
        pooled_output = text_outputs[1]
        
        # 将汇总的输出依次经过text_projection的每一层处理
        for layer in self.text_projection:
            pooled_output = layer(pooled_output)

        # 返回文本特征
        text_features = pooled_output
        return text_features

    @unpack_inputs
    # 定义一个方法，用于获取图像特征
    def get_image_features(
        self,
        pixel_values: TFModelInputType | None = None,  # 像素值输入，可以为 None
        output_attentions: Optional[bool] = None,      # 是否输出注意力权重，默认为 None
        output_hidden_states: Optional[bool] = None,    # 是否输出隐藏状态，默认为 None
        return_dict: Optional[bool] = None,             # 是否返回字典形式的输出，默认为 None
        training: bool = False,                         # 是否处于训练模式，默认为 False
    ) -> tf.Tensor:                                    # 返回类型为 TensorFlow 的张量

        # 如果像素值为 None，则抛出 ValueError 异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 使用视觉模型处理像素值，根据参数选择是否返回注意力权重和隐藏状态
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 获取视觉模型输出的汇总特征（一般是第二个输出）
        pooled_output = vision_outputs[1]

        # 通过每层的可调用层对汇总特征进行变换
        for layer in self.visual_projection:
            pooled_output = layer(pooled_output)

        # 将处理后的图像特征赋给变量 image_features
        image_features = pooled_output

        # 返回图像特征
        return image_features

    # 使用装饰器 unpack_inputs 定义一个方法，用于调用模型
    def call(
        self,
        input_ids: TFModelInputType | None = None,            # 输入的 token IDs，可以为 None
        pixel_values: TFModelInputType | None = None,         # 像素值输入，可以为 None
        attention_mask: np.ndarray | tf.Tensor | None = None, # 注意力掩码，可以为 None
        position_ids: np.ndarray | tf.Tensor | None = None,   # 位置 IDs，可以为 None
        return_loss: Optional[bool] = None,                   # 是否返回损失，默认为 None
        output_attentions: Optional[bool] = None,             # 是否输出注意力权重，默认为 None
        output_hidden_states: Optional[bool] = None,          # 是否输出隐藏状态，默认为 None
        output_segmentation: Optional[bool] = None,           # 是否输出分割结果，默认为 None
        return_dict: Optional[bool] = None,                   # 是否返回字典形式的输出，默认为 None
        training: bool = False,                               # 是否处于训练模式，默认为 False
# GROUPVIT_TEXT_INPUTS_DOCSTRING 变量，包含了关于输入格式的文档字符串，用于说明 TF 2.0 模型接受的输入格式。
GROUPVIT_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `({0})`):
            # 输入序列的标记索引在词汇表中的位置。
            # 可以使用 [`AutoTokenizer`] 获取这些索引。详情请参见 [`PreTrainedTokenizer.__call__`] 和 [`PreTrainedTokenizer.encode`]。
            # [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            # 遮罩，用于在填充标记索引上避免进行注意力计算。
            # 遮罩值为 `[0, 1]`：
            # - 1 表示**未被遮罩**的标记，
            # - 0 表示**被遮罩**的标记。
            # [什么是注意力遮罩？](../glossary#attention-mask)
        position_ids (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            # 每个输入序列标记在位置嵌入中的位置索引。
            # 选取范围为 `[0, config.max_position_embeddings - 1]`。
            # [什么是位置 ID？](../glossary#position-ids)
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。
            # 返回的张量中详细说明了 `attentions`。此参数仅在 eager 模式下可用，在图模式下将使用配置中的值。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。
            # 返回的张量中详细说明了 `hidden_states`。此参数仅在 eager 模式下可用，在图模式下将使用配置中的值。
        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通的元组。此参数仅在 eager 模式下可用，在图模式下值始终为 True。
        training (`bool`, *optional*, defaults to `False``):
            # 是否将模型设置为训练模式（某些模块如 dropout 模块在训练和评估中行为不同）。
"""

GROUPVIT_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]`, `Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`CLIPImageProcessor.__call__`] for details.
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
"""

GROUPVIT_INPUTS_DOCSTRING = r"""
    A docstring describing the inputs expected by a function or method in the GROUPVIT module.

"""
    Args:
        input_ids (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]`, `Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `({0})`):
            # 输入序列标记在词汇表中的索引。
            # 可以使用 [`AutoTokenizer`] 获取索引。详见 [`PreTrainedTokenizer.__call__`] 和 [`PreTrainedTokenizer.encode`]。
            # [什么是输入 ID?](../glossary#input-ids)

        pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]`, `Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`):
            # 像素值。可以使用 [`AutoImageProcessor`] 获取像素值。详见 [`CLIPImageProcessor.__call__`]。
        
        attention_mask (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            # 遮罩，避免在填充的标记索引上执行注意力操作。
            # 遮罩值在 `[0, 1]` 之间：
            # - 1 表示**未遮罩**的标记，
            # - 0 表示**已遮罩**的标记。
            # [什么是注意力遮罩?](../glossary#attention-mask)

        position_ids (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            # 每个输入序列标记在位置嵌入中的位置索引。
            # 选择范围为 `[0, config.max_position_embeddings - 1]`。
            # [什么是位置 ID?](../glossary#position-ids)

        return_loss (`bool`, *optional*):
            # 是否返回对比损失。

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。详见返回的张量中的 `attentions`。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。详见返回的张量中的 `hidden_states`。

        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通元组。

        training (`bool`, *optional*, defaults to `False`):
            # 是否在训练模式中使用模型（某些模块在训练和评估之间具有不同的行为）。
"""
TFGroupViTTextModel 类定义了一个基于 TFGroupViTPreTrainedModel 的文本模型。
"""
class TFGroupViTTextModel(TFGroupViTPreTrainedModel):
    # 设置配置类
    config_class = GroupViTTextConfig
    # 主输入名称
    main_input_name = "input_ids"

    def __init__(self, config: GroupViTTextConfig, *inputs, **kwargs):
        # 调用父类构造函数
        super().__init__(config, *inputs, **kwargs)

        # 初始化 TFGroupViTTextMainLayer 实例作为模型的主要组件
        self.groupvit = TFGroupViTTextMainLayer(config, name="groupvit")

    @unpack_inputs
    # 将输入参数解包后，添加文档字符串到模型的前向传播方法
    @add_start_docstrings_to_model_forward(GROUPVIT_TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 替换模型前向传播方法的返回文档字符串
    @replace_return_docstrings(output_type=TFBaseModelOutputWithPooling, config_class=GroupViTTextConfig)
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
        r"""
        模型的前向传播函数，接受输入参数并返回模型的输出。

        Returns:
            TFBaseModelOutputWithPooling 或者包含 tf.Tensor 的元组

        Examples:
        示例用法，展示了如何使用模型进行推理。

        ```
        >>> from transformers import CLIPTokenizer, TFGroupViTTextModel

        >>> tokenizer = CLIPTokenizer.from_pretrained("nvidia/groupvit-gcc-yfcc")
        >>> model = TFGroupViTTextModel.from_pretrained("nvidia/groupvit-gcc-yfcc")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="tf")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```
        """

        # 调用 self.groupvit 的前向传播方法并返回结果
        outputs = self.groupvit(
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
        # 如果已经构建完成，则直接返回
        if self.built:
            return
        # 设置构建完成标志
        self.built = True
        # 如果 self.groupvit 存在，则在 TensorFlow 的命名空间内构建组件
        if getattr(self, "groupvit", None) is not None:
            with tf.name_scope(self.groupvit.name):
                self.groupvit.build(None)


class TFGroupViTVisionModel(TFGroupViTPreTrainedModel):
    # 设置配置类
    config_class = GroupViTVisionConfig
    # 主输入名称
    main_input_name = "pixel_values"

    def __init__(self, config: GroupViTVisionConfig, *inputs, **kwargs):
        # 调用父类构造函数
        super().__init__(config, *inputs, **kwargs)

        # 初始化 TFGroupViTVisionMainLayer 实例作为模型的主要组件
        self.groupvit = TFGroupViTVisionMainLayer(config, name="groupvit")

    @unpack_inputs
    # 添加文档字符串到模型的前向传播方法
    @add_start_docstrings_to_model_forward(GROUPVIT_VISION_INPUTS_DOCSTRING)
    # 替换模型前向传播方法的返回文档字符串
    @replace_return_docstrings(output_type=TFBaseModelOutputWithPooling, config_class=GroupViTVisionConfig)
    def call(
        self,
        pixel_values: TFModelInputType | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        r"""
        返回模型的输出结果。

        Examples:

        ```
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFGroupViTVisionModel

        >>> processor = AutoProcessor.from_pretrained("nvidia/groupvit-gcc-yfcc")
        >>> model = TFGroupViTVisionModel.from_pretrained("nvidia/groupvit-gcc-yfcc")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="tf")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```
        """

        outputs = self.groupvit(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "groupvit", None) is not None:
            with tf.name_scope(self.groupvit.name):
                self.groupvit.build(None)
# 使用装饰器添加文档字符串，指定类的起始文档字符串
@add_start_docstrings(GROUPVIT_START_DOCSTRING)
# 定义 TFGroupViTModel 类，继承自 TFGroupViTPreTrainedModel 类
class TFGroupViTModel(TFGroupViTPreTrainedModel):
    # 指定配置类为 GroupViTConfig
    config_class = GroupViTConfig

    # 初始化方法，接受 GroupViTConfig 类型的配置对象和其他参数
    def __init__(self, config: GroupViTConfig, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        
        # 创建 TFGroupViTMainLayer 实例，命名为 groupvit
        self.groupvit = TFGroupViTMainLayer(config, name="groupvit")

    # 使用装饰器添加文档字符串到模型前向传播方法
    @unpack_inputs
    @add_start_docstrings_to_model_forward(GROUPVIT_TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        返回文本特征张量 (`tf.Tensor` of shape `(batch_size, output_dim`): 
        通过将投影层应用于 [`TFGroupViTTextModel`] 的汇总输出所得到的文本嵌入。

        示例:

        ```
        >>> from transformers import CLIPTokenizer, TFGroupViTModel

        >>> model = TFGroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc")
        >>> tokenizer = CLIPTokenizer.from_pretrained("nvidia/groupvit-gcc-yfcc")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="tf")
        >>> text_features = model.get_text_features(**inputs)
        ```"""

        # 调用 TFGroupViTMainLayer 的 get_text_features 方法，返回文本特征张量
        text_features = self.groupvit.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 返回文本特征张量
        return text_features

    # 使用装饰器添加文档字符串到模型前向传播方法
    @unpack_inputs
    @add_start_docstrings_to_model_forward(GROUPVIT_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values: TFModelInputType | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> tf.Tensor:
        r"""
        """
    ) -> tf.Tensor:
        r"""
        Returns:
            image_features (`tf.Tensor` of shape `(batch_size, output_dim`): The image embeddings obtained by applying
            the projection layer to the pooled output of [`TFGroupViTVisionModel`].

        Examples:

        ```
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFGroupViTModel

        >>> model = TFGroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc")
        >>> processor = AutoProcessor.from_pretrained("nvidia/groupvit-gcc-yfcc")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="tf")

        >>> image_features = model.get_image_features(**inputs)
        ```"""

        # 调用 TFGroupViTVisionModel 的方法获取图像特征
        image_features = self.groupvit.get_image_features(
            pixel_values=pixel_values,  # 图像像素值
            output_attentions=output_attentions,  # 是否输出注意力权重
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
            return_dict=return_dict,  # 是否以字典形式返回结果
            training=training,  # 是否处于训练模式
        )

        return image_features

    @unpack_inputs
    @add_start_docstrings_to_model_forward(GROUPVIT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFGroupViTModelOutput, config_class=GroupViTConfig)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        pixel_values: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_segmentation: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        ) -> Union[TFGroupViTModelOutput, Tuple[tf.Tensor]]:
        r"""
        Returns:

        Examples:

        ```
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFGroupViTModel
        >>> import tensorflow as tf

        >>> model = TFGroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc")
        >>> processor = AutoProcessor.from_pretrained("nvidia/groupvit-gcc-yfcc")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="tf", padding=True
        ... )

        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = tf.math.softmax(logits_per_image, axis=1)  # we can take the softmax to get the label probabilities
        ```"""

        # 调用模型的 forward 方法，传递输入参数进行推理
        outputs = self.groupvit(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_loss=return_loss,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_segmentation=output_segmentation,
            return_dict=return_dict,
            training=training,
        )

        return outputs

    def serving_output(self, output: TFGroupViTModelOutput) -> TFGroupViTModelOutput:
        # TODO: As is this currently fails with saved_model=True, because
        # TensorFlow cannot trace through nested dataclasses. Reference:
        # https://github.com/huggingface/transformers/pull/16886
        # 返回模型输出作为服务端输出
        return output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果模型已经构建，则直接返回
        if getattr(self, "groupvit", None) is not None:
            # 使用 TensorFlow 的命名空间来构建模型组件
            with tf.name_scope(self.groupvit.name):
                self.groupvit.build(None)
```