# `.\models\funnel\modeling_tf_funnel.py`

```
# coding=utf-8
# Copyright 2020-present Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
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
""" TF 2.0 Funnel model."""


from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFMaskedLMOutput,
    TFMultipleChoiceModelOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (
    TFMaskedLanguageModelingLoss,
    TFModelInputType,
    TFMultipleChoiceLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    get_initializer,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_funnel import FunnelConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "FunnelConfig"

TF_FUNNEL_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "funnel-transformer/small",  # B4-4-4H768
    "funnel-transformer/small-base",  # B4-4-4H768, no decoder
    "funnel-transformer/medium",  # B6-3x2-3x2H768
    "funnel-transformer/medium-base",  # B6-3x2-3x2H768, no decoder
    "funnel-transformer/intermediate",  # B6-6-6H768
    "funnel-transformer/intermediate-base",  # B6-6-6H768, no decoder
    "funnel-transformer/large",  # B8-8-8H1024
    "funnel-transformer/large-base",  # B8-8-8H1024, no decoder
    "funnel-transformer/xlarge-base",  # B10-10-10H1024
    "funnel-transformer/xlarge",  # B10-10-10H1024, no decoder
]

INF = 1e6


class TFFunnelEmbeddings(keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)  # 调用父类的初始化方法

        self.config = config  # 保存配置对象的引用
        self.hidden_size = config.hidden_size  # 从配置对象中获取隐藏层大小
        self.initializer_std = 1.0 if config.initializer_std is None else config.initializer_std  # 设置初始化器的标准差

        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")  # 创建 LayerNormalization 层
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout)  # 创建 Dropout 层

    def build(self, input_shape=None):
        with tf.name_scope("word_embeddings"):  # 定义名称域为 "word_embeddings"
            self.weight = self.add_weight(
                name="weight",  # 参数名为 "weight"
                shape=[self.config.vocab_size, self.hidden_size],  # 权重张量的形状
                initializer=get_initializer(initializer_range=self.initializer_std),  # 使用给定的初始化器初始化权重
            )

        if self.built:  # 如果已经构建过，直接返回
            return
        self.built = True  # 标记为已构建
        if getattr(self, "LayerNorm", None) is not None:  # 如果存在 LayerNorm 层
            with tf.name_scope(self.LayerNorm.name):  # 使用 LayerNorm 层的名称作为名称域
                self.LayerNorm.build([None, None, self.config.d_model])  # 构建 LayerNorm 层

    def call(self, input_ids=None, inputs_embeds=None, training=False):
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        assert not (input_ids is None and inputs_embeds is None)  # 断言输入张量不能同时为空
        assert not (input_ids is not None and inputs_embeds is not None)  # 断言输入张量不能同时不为空

        if input_ids is not None:  # 如果输入张量 input_ids 不为空
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)  # 检查 input_ids 是否在有效范围内
            inputs_embeds = tf.gather(self.weight, input_ids)  # 使用权重张量 self.weight 获取对应的嵌入向量

        final_embeddings = self.LayerNorm(inputs=inputs_embeds)  # 应用 LayerNorm 层
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)  # 应用 Dropout 层

        return final_embeddings  # 返回最终嵌入张量
    """
    Contains helpers for `TFFunnelRelMultiheadAttention`.
    """

    # 类属性，代表 <cls> token 的类型 ID，默认为 2
    cls_token_type_id: int = 2

    def __init__(self, config):
        # 初始化函数，根据传入的配置对象 config 初始化各个实例变量
        self.d_model = config.d_model  # 模型的维度
        self.attention_type = config.attention_type  # 注意力类型
        self.num_blocks = config.num_blocks  # 块的数量
        self.separate_cls = config.separate_cls  # 是否分离 <cls> token
        self.truncate_seq = config.truncate_seq  # 是否截断序列
        self.pool_q_only = config.pool_q_only  # 是否只池化查询（query）
        self.pooling_type = config.pooling_type  # 池化的类型

        self.sin_dropout = keras.layers.Dropout(config.hidden_dropout)  # Sinusoidal dropout
        self.cos_dropout = keras.layers.Dropout(config.hidden_dropout)  # Cosinusoidal dropout
        self.pooling_mult = None  # 池化倍数，初始为 None

    def init_attention_inputs(self, inputs_embeds, attention_mask=None, token_type_ids=None, training=False):
        """Returns the attention inputs associated to the inputs of the model."""
        # 初始化注意力输入，根据模型的输入返回相应的注意力输入
        # inputs_embeds 的形状为 batch_size x seq_len x d_model
        # attention_mask 和 token_type_ids 的形状为 batch_size x seq_len
        self.pooling_mult = 1  # 设置池化倍数为 1
        self.seq_len = seq_len = shape_list(inputs_embeds)[1]  # 记录序列的长度
        position_embeds = self.get_position_embeds(seq_len, training=training)  # 获取位置嵌入
        token_type_mat = self.token_type_ids_to_mat(token_type_ids) if token_type_ids is not None else None  # 将 token_type_ids 转换为 token_type_mat
        # 根据配置是否分离 <cls> token，创建对应的 mask
        cls_mask = (
            tf.pad(tf.ones([seq_len - 1, seq_len - 1], dtype=inputs_embeds.dtype), [[1, 0], [1, 0]])
            if self.separate_cls
            else None
        )
        return (position_embeds, token_type_mat, attention_mask, cls_mask)

    def token_type_ids_to_mat(self, token_type_ids):
        """Convert `token_type_ids` to `token_type_mat`."""
        # 将 token_type_ids 转换为 token_type_mat，用于区分不同类型的 token
        token_type_mat = tf.equal(tf.expand_dims(token_type_ids, -1), tf.expand_dims(token_type_ids, -2))
        # 将 <cls> token 视为与 A 和 B 都在同一段中
        cls_ids = tf.equal(token_type_ids, tf.constant([self.cls_token_type_id], dtype=token_type_ids.dtype))
        cls_mat = tf.logical_or(tf.expand_dims(cls_ids, -1), tf.expand_dims(cls_ids, -2))
        return tf.logical_or(cls_mat, token_type_mat)

    def stride_pool_pos(self, pos_id, block_index):
        """
        Pool `pos_id` while keeping the cls token separate (if `self.separate_cls=True`).
        """
        # 对 pos_id 进行池化，如果 self.separate_cls=True，则保持 <cls> token 分开处理
        if self.separate_cls:
            # 在分离 <cls> token 的情况下，将 <cls> token 视为前一个块的第一个 token
            # 第一个实际块的位置始终为 1，前一个块的位置将为 `1 - 2 ** block_index`
            cls_pos = tf.constant([-(2**block_index) + 1], dtype=pos_id.dtype)
            # 如果截断序列，则从 pos_id 的第二个位置开始池化
            # 否则从 pos_id 的第一个位置开始池化
            pooled_pos_id = pos_id[1:-1] if self.truncate_seq else pos_id[1:]
            return tf.concat([cls_pos, pooled_pos_id[::2]], 0)
        else:
            # 如果不分离 <cls> token，则直接每隔一个位置进行池化
            return pos_id[::2]
    def relative_pos(self, pos, stride, pooled_pos=None, shift=1):
        """
        Build the relative positional vector between `pos` and `pooled_pos`.
        构建 `pos` 和 `pooled_pos` 之间的相对位置向量。
        """
        if pooled_pos is None:
            pooled_pos = pos

        # Calculate the reference point based on pooled_pos and pos
        ref_point = pooled_pos[0] - pos[0]
        # Calculate the number of elements to remove
        num_remove = shift * shape_list(pooled_pos)[0]
        # Calculate the maximum distance based on the reference point, stride, and number of elements to remove
        max_dist = ref_point + num_remove * stride
        # Calculate the minimum distance based on pooled_pos and pos
        min_dist = pooled_pos[0] - pos[-1]

        # Generate a range tensor from max_dist to min_dist-1 with step -stride
        return tf.range(max_dist, min_dist - 1, -stride)

    def stride_pool(self, tensor, axis):
        """
        Perform pooling by stride slicing the tensor along the given axis.
        在给定的轴上通过步进切片对张量进行池化。
        """
        if tensor is None:
            return None

        # If axis is a list or tuple of ints, recursively perform stride pool for each axis
        if isinstance(axis, (list, tuple)):
            for ax in axis:
                tensor = self.stride_pool(tensor, ax)
            return tensor

        # If tensor is a list or tuple of tensors, recursively perform stride pool for each tensor
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(self.stride_pool(x, axis) for x in tensor)

        # Handle negative axis values
        axis %= len(shape_list(tensor))

        # Determine the axis_slice based on conditions
        axis_slice = slice(None, -1, 2) if self.separate_cls and self.truncate_seq else slice(None, None, 2)
        enc_slice = [slice(None)] * axis + [axis_slice]

        # If separate_cls is True, concatenate the first slice of tensor with tensor along the specified axis
        if self.separate_cls:
            cls_slice = [slice(None)] * axis + [slice(None, 1)]
            tensor = tf.concat([tensor[cls_slice], tensor], axis)
        
        # Return the sliced tensor
        return tensor[enc_slice]

    def pool_tensor(self, tensor, mode="mean", stride=2):
        """Apply 1D pooling to a tensor of size [B x T (x H)]."""
        if tensor is None:
            return None

        # If tensor is a list or tuple of tensors, recursively apply pool_tensor to each tensor
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(self.pool_tensor(x, mode=mode, stride=stride) for x in tensor)

        # Adjust tensor based on separate_cls and truncate_seq conditions
        if self.separate_cls:
            suffix = tensor[:, :-1] if self.truncate_seq else tensor
            tensor = tf.concat([tensor[:, :1], suffix], axis=1)

        ndim = len(shape_list(tensor))
        # Expand tensor dimensions if ndim equals 2
        if ndim == 2:
            tensor = tensor[:, :, None]

        # Perform 1D pooling based on mode (mean, max, min)
        if mode == "mean":
            tensor = tf.nn.avg_pool1d(tensor, stride, strides=stride, data_format="NWC", padding="SAME")
        elif mode == "max":
            tensor = tf.nn.max_pool1d(tensor, stride, strides=stride, data_format="NWC", padding="SAME")
        elif mode == "min":
            tensor = -tf.nn.max_pool1d(-tensor, stride, strides=stride, data_format="NWC", padding="SAME")
        else:
            raise NotImplementedError("The supported modes are 'mean', 'max' and 'min'.")

        # Squeeze the tensor if ndim equals 2
        return tf.squeeze(tensor, 2) if ndim == 2 else tensor
    def pre_attention_pooling(self, output, attention_inputs):
        """Pool `output` and the proper parts of `attention_inputs` before the attention layer."""
        # 解包 attention_inputs 中的各个部分
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs
        
        # 如果仅对查询进行池化
        if self.pool_q_only:
            # 如果使用因式化注意力类型
            if self.attention_type == "factorized":
                # 对位置嵌入的前两部分进行池化操作，然后将其余部分保持不变
                position_embeds = self.stride_pool(position_embeds[:2], 0) + position_embeds[2:]
            # 对 token 类型矩阵进行池化
            token_type_mat = self.stride_pool(token_type_mat, 1)
            # 对类别掩码进行池化
            cls_mask = self.stride_pool(cls_mask, 0)
            
            # 对输出进行张量池化操作
            output = self.pool_tensor(output, mode=self.pooling_type)
        else:
            # 池化倍数乘以2
            self.pooling_mult *= 2
            # 如果使用因式化注意力类型
            if self.attention_type == "factorized":
                # 对位置嵌入进行池化操作
                position_embeds = self.stride_pool(position_embeds, 0)
            # 对 token 类型矩阵进行池化，使用步长为 [1, 2]
            token_type_mat = self.stride_pool(token_type_mat, [1, 2])
            # 对类别掩码进行池化，使用步长为 [1, 2]
            cls_mask = self.stride_pool(cls_mask, [1, 2])
            # 对注意力掩码进行张量池化操作，使用模式为 "min"
            attention_mask = self.pool_tensor(attention_mask, mode="min")
            # 对输出进行张量池化操作
            output = self.pool_tensor(output, mode=self.pooling_type)
        
        # 更新 attention_inputs
        attention_inputs = (position_embeds, token_type_mat, attention_mask, cls_mask)
        # 返回池化后的输出和更新后的 attention_inputs
        return output, attention_inputs

    def post_attention_pooling(self, attention_inputs):
        """Pool the proper parts of `attention_inputs` after the attention layer."""
        # 解包 attention_inputs 中的各个部分
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs
        
        # 如果仅对查询进行池化
        if self.pool_q_only:
            # 池化倍数乘以2
            self.pooling_mult *= 2
            # 如果使用因式化注意力类型
            if self.attention_type == "factorized":
                # 将位置嵌入的前两部分保持不变，对剩余部分进行池化操作
                position_embeds = position_embeds[:2] + self.stride_pool(position_embeds[2:], 0)
            # 对 token 类型矩阵进行池化，使用步长为 2
            token_type_mat = self.stride_pool(token_type_mat, 2)
            # 对类别掩码进行池化，使用步长为 1
            cls_mask = self.stride_pool(cls_mask, 1)
            # 对注意力掩码进行张量池化操作，使用模式为 "min"
            attention_mask = self.pool_tensor(attention_mask, mode="min")
        
        # 更新 attention_inputs
        attention_inputs = (position_embeds, token_type_mat, attention_mask, cls_mask)
        # 返回更新后的 attention_inputs
        return attention_inputs
def _relative_shift_gather(positional_attn, context_len, shift):
    batch_size, n_head, seq_len, max_rel_len = shape_list(positional_attn)
    # 获取 positional_attn 的形状信息，分别为 batch_size, n_head, seq_len, max_rel_len

    # 对 positional_attn 进行形状重塑，将其变为 [batch_size, n_head, max_rel_len, seq_len]
    positional_attn = tf.reshape(positional_attn, [batch_size, n_head, max_rel_len, seq_len])
    # 从第三个维度开始截取，即将 shift 后的部分保留，得到 [batch_size, n_head, max_rel_len - shift, seq_len]
    positional_attn = positional_attn[:, :, shift:, :]
    # 再次进行形状重塑，得到 [batch_size, n_head, seq_len, max_rel_len - shift]
    positional_attn = tf.reshape(positional_attn, [batch_size, n_head, seq_len, max_rel_len - shift])
    # 从最后一个维度截取 context_len 长度的部分，得到 [batch_size, n_head, seq_len, context_len]
    positional_attn = positional_attn[..., :context_len]
    # 返回处理后的 positional_attn
    return positional_attn


class TFFunnelRelMultiheadAttention(keras.layers.Layer):
    def __init__(self, config, block_index, **kwargs):
        super().__init__(**kwargs)
        # 初始化 TFFunnelRelMultiheadAttention 层，使用 config 中的参数和 block_index

        self.attention_type = config.attention_type
        self.n_head = n_head = config.n_head
        self.d_head = d_head = config.d_head
        self.d_model = d_model = config.d_model
        self.initializer_range = config.initializer_range
        self.block_index = block_index

        # 定义不同的 Dropout 层，分别用于隐藏层和注意力层
        self.hidden_dropout = keras.layers.Dropout(config.hidden_dropout)
        self.attention_dropout = keras.layers.Dropout(config.attention_dropout)

        # 获取初始化器，用于后面的层的权重初始化
        initializer = get_initializer(config.initializer_range)

        # 定义查询、键、值头部的全连接层
        self.q_head = keras.layers.Dense(
            n_head * d_head, use_bias=False, kernel_initializer=initializer, name="q_head"
        )
        self.k_head = keras.layers.Dense(n_head * d_head, kernel_initializer=initializer, name="k_head")
        self.v_head = keras.layers.Dense(n_head * d_head, kernel_initializer=initializer, name="v_head")

        # 定义后处理层和 LayerNormalization 层
        self.post_proj = keras.layers.Dense(d_model, kernel_initializer=initializer, name="post_proj")
        self.layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        # 定义缩放因子，用于缩放注意力得分
        self.scale = 1.0 / (d_head**0.5)
    # 在神经网络层的构建函数中，用于构建模型的输入形状
    def build(self, input_shape=None):
        # 从对象属性中获取头数、头的维度和模型的维度
        n_head, d_head, d_model = self.n_head, self.d_head, self.d_model
        # 根据指定的初始化范围获取初始化器
        initializer = get_initializer(self.initializer_range)

        # 添加权重变量 r_w_bias，形状为 (n_head, d_head)，用指定初始化器初始化
        self.r_w_bias = self.add_weight(
            shape=(n_head, d_head), initializer=initializer, trainable=True, name="r_w_bias"
        )
        # 添加权重变量 r_r_bias，形状为 (n_head, d_head)，用指定初始化器初始化
        self.r_r_bias = self.add_weight(
            shape=(n_head, d_head), initializer=initializer, trainable=True, name="r_r_bias"
        )
        # 添加权重变量 r_kernel，形状为 (d_model, n_head, d_head)，用指定初始化器初始化
        self.r_kernel = self.add_weight(
            shape=(d_model, n_head, d_head), initializer=initializer, trainable=True, name="r_kernel"
        )
        # 添加权重变量 r_s_bias，形状为 (n_head, d_head)，用指定初始化器初始化
        self.r_s_bias = self.add_weight(
            shape=(n_head, d_head), initializer=initializer, trainable=True, name="r_s_bias"
        )
        # 添加权重变量 seg_embed，形状为 (2, n_head, d_head)，用指定初始化器初始化
        self.seg_embed = self.add_weight(
            shape=(2, n_head, d_head), initializer=initializer, trainable=True, name="seg_embed"
        )

        # 如果已经构建过，直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True

        # 如果存在 q_head 属性，构建其模型结构
        if getattr(self, "q_head", None) is not None:
            with tf.name_scope(self.q_head.name):
                self.q_head.build([None, None, d_model])
        
        # 如果存在 k_head 属性，构建其模型结构
        if getattr(self, "k_head", None) is not None:
            with tf.name_scope(self.k_head.name):
                self.k_head.build([None, None, d_model])
        
        # 如果存在 v_head 属性，构建其模型结构
        if getattr(self, "v_head", None) is not None:
            with tf.name_scope(self.v_head.name):
                self.v_head.build([None, None, d_model])
        
        # 如果存在 post_proj 属性，构建其模型结构
        if getattr(self, "post_proj", None) is not None:
            with tf.name_scope(self.post_proj.name):
                self.post_proj.build([None, None, n_head * d_head])
        
        # 如果存在 layer_norm 属性，构建其模型结构
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, d_model])
    def relative_positional_attention(self, position_embeds, q_head, context_len, cls_mask=None):
        """Relative attention score for the positional encodings"""
        # q_head has shape batch_size x sea_len x n_head x d_head
        
        if self.attention_type == "factorized":
            # Notations from the paper, appending A.2.2, final formula (https://arxiv.org/abs/2006.03236)
            # phi and pi have shape seq_len x d_model, psi and omega have shape context_len x d_model
            
            phi, pi, psi, omega = position_embeds
            
            # Shape n_head x d_head
            u = self.r_r_bias * self.scale
            
            # Shape d_model x n_head x d_head
            w_r = self.r_kernel
            
            # Shape batch_size x sea_len x n_head x d_model
            q_r_attention = tf.einsum("binh,dnh->bind", q_head + u, w_r)
            q_r_attention_1 = q_r_attention * phi[:, None]
            q_r_attention_2 = q_r_attention * pi[:, None]
            
            # Shape batch_size x n_head x seq_len x context_len
            positional_attn = tf.einsum("bind,jd->bnij", q_r_attention_1, psi) + tf.einsum(
                "bind,jd->bnij", q_r_attention_2, omega
            )
        
        else:
            # Notations from the paper, appending A.2.1, final formula (https://arxiv.org/abs/2006.03236)
            # Grab the proper positional encoding, shape max_rel_len x d_model
            
            if shape_list(q_head)[1] != context_len:
                shift = 2
                r = position_embeds[self.block_index][1]
            else:
                shift = 1
                r = position_embeds[self.block_index][0]
            
            # Shape n_head x d_head
            v = self.r_r_bias * self.scale
            
            # Shape d_model x n_head x d_head
            w_r = self.r_kernel
            
            # Shape max_rel_len x n_head x d_model
            r_head = tf.einsum("td,dnh->tnh", r, w_r)
            
            # Shape batch_size x n_head x seq_len x max_rel_len
            positional_attn = tf.einsum("binh,tnh->bnit", q_head + v, r_head)
            
            # Shape batch_size x n_head x seq_len x context_len
            positional_attn = _relative_shift_gather(positional_attn, context_len, shift)
        
        if cls_mask is not None:
            positional_attn *= cls_mask
        
        return positional_attn
    def relative_token_type_attention(self, token_type_mat, q_head, cls_mask=None):
        """Relative attention score for the token_type_ids"""
        # 如果token_type_mat为None，则返回0
        if token_type_mat is None:
            return 0
        # 获取token_type_mat的形状信息
        batch_size, seq_len, context_len = shape_list(token_type_mat)
        
        # q_head的形状为 batch_size x seq_len x n_head x d_head
        # Shape n_head x d_head
        r_s_bias = self.r_s_bias * self.scale
        
        # Shape batch_size x n_head x seq_len x 2
        # 计算相对注意力偏置 token_type_bias
        token_type_bias = tf.einsum("bind,snd->bnis", q_head + r_s_bias, self.seg_embed)
        
        # Shape batch_size x n_head x seq_len x context_len
        # 将token_type_mat扩展为与token_type_bias相同的形状
        token_type_mat = tf.tile(token_type_mat[:, None], [1, shape_list(q_head)[2], 1, 1])
        
        # Shapes batch_size x n_head x seq_len
        # 将token_type_bias分为两部分：diff_token_type 和 same_token_type
        diff_token_type, same_token_type = tf.split(token_type_bias, 2, axis=-1)
        
        # Shape batch_size x n_head x seq_len x context_len
        # 根据token_type_mat的值选择不同的token_type_attn
        token_type_attn = tf.where(
            token_type_mat,
            tf.tile(same_token_type, [1, 1, 1, context_len]),
            tf.tile(diff_token_type, [1, 1, 1, context_len]),
        )

        # 如果存在cls_mask，则将token_type_attn与cls_mask相乘
        if cls_mask is not None:
            token_type_attn *= cls_mask
        
        # 返回计算得到的token_type_attn
        return token_type_attn
    def call(self, query, key, value, attention_inputs, output_attentions=False, training=False):
        # query has shape batch_size x seq_len x d_model
        # key and value have shapes batch_size x context_len x d_model
        # position_embeds, token_type_mat, attention_mask, cls_mask are unpacked from attention_inputs

        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs

        batch_size, seq_len, _ = shape_list(query)
        context_len = shape_list(key)[1]
        n_head, d_head = self.n_head, self.d_head

        # Shape batch_size x seq_len x n_head x d_head
        q_head = tf.reshape(self.q_head(query), [batch_size, seq_len, n_head, d_head])
        # Shapes batch_size x context_len x n_head x d_head
        k_head = tf.reshape(self.k_head(key), [batch_size, context_len, n_head, d_head])
        v_head = tf.reshape(self.v_head(value), [batch_size, context_len, n_head, d_head])

        q_head = q_head * self.scale
        # Shape n_head x d_head
        r_w_bias = self.r_w_bias * self.scale
        # Shapes batch_size x n_head x seq_len x context_len
        content_score = tf.einsum("bind,bjnd->bnij", q_head + r_w_bias, k_head)
        positional_attn = self.relative_positional_attention(position_embeds, q_head, context_len, cls_mask)
        token_type_attn = self.relative_token_type_attention(token_type_mat, q_head, cls_mask)

        # merge attention scores
        attn_score = content_score + positional_attn + token_type_attn

        # perform masking
        if attention_mask is not None:
            attention_mask = tf.cast(attention_mask, dtype=attn_score.dtype)
            attn_score = attn_score - (INF * (1 - attention_mask[:, None, None]))

        # attention probability
        attn_prob = stable_softmax(attn_score, axis=-1)
        attn_prob = self.attention_dropout(attn_prob, training=training)

        # attention output, shape batch_size x seq_len x n_head x d_head
        attn_vec = tf.einsum("bnij,bjnd->bind", attn_prob, v_head)

        # Shape shape batch_size x seq_len x d_model
        attn_out = self.post_proj(tf.reshape(attn_vec, [batch_size, seq_len, n_head * d_head]))
        attn_out = self.hidden_dropout(attn_out, training=training)

        output = self.layer_norm(query + attn_out)
        return (output, attn_prob) if output_attentions else (output,)
# 定义一个名为TFFunnelPositionwiseFFN的自定义层，继承自keras.layers.Layer
class TFFunnelPositionwiseFFN(keras.layers.Layer):

    # 初始化函数，接收config和kwargs参数
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 根据config中的initializer_range获取初始化器
        initializer = get_initializer(config.initializer_range)
        # 创建一个全连接层，输入维度为config.d_model，输出维度为config.d_inner，使用刚初始化的initializer，命名为linear_1
        self.linear_1 = keras.layers.Dense(config.d_inner, kernel_initializer=initializer, name="linear_1")
        # 根据config中的hidden_act获取激活函数
        self.activation_function = get_tf_activation(config.hidden_act)
        # 创建一个dropout层，使用config中的activation_dropout参数
        self.activation_dropout = keras.layers.Dropout(config.activation_dropout)
        # 创建一个全连接层，输入维度为config.d_inner，输出维度为config.d_model，使用刚初始化的initializer，命名为linear_2
        self.linear_2 = keras.layers.Dense(config.d_model, kernel_initializer=initializer, name="linear_2")
        # 创建一个dropout层，使用config中的hidden_dropout参数
        self.dropout = keras.layers.Dropout(config.hidden_dropout)
        # 创建一个LayerNormalization层，输入维度为config.d_model，使用config中的layer_norm_eps参数，命名为layer_norm
        self.layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        # 保存config参数
        self.config = config

    # 定义调用函数，接收hidden和training两个参数
    def call(self, hidden, training=False):
        # 使用linear_1层处理hidden
        h = self.linear_1(hidden)
        # 使用激活函数处理h
        h = self.activation_function(h)
        # 使用activation_dropout层处理h，根据training参数决定是否启用训练模式
        h = self.activation_dropout(h, training=training)
        # 使用linear_2层处理h
        h = self.linear_2(h)
        # 使用dropout层处理h，根据training参数决定是否启用训练模式
        h = self.dropout(h, training=training)
        # 返回LayerNormalization层处理后的结果
        return self.layer_norm(hidden + h)

    # 构建函数，接收input_shape参数，默认为None
    def build(self, input_shape=None):
        # 如果已经构建好了，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 检查linear_1是否存在
        if getattr(self, "linear_1", None) is not None:
            # 在linear_1的作用域内构建该层，指定输入维度为[None, None, self.config.d_model]
            with tf.name_scope(self.linear_1.name):
                self.linear_1.build([None, None, self.config.d_model])
        # 检查linear_2是否存在
        if getattr(self, "linear_2", None) is not None:
            # 在linear_2的作用域内构建该层，指定输入维度为[None, None, self.config.d_inner]
            with tf.name_scope(self.linear_2.name):
                self.linear_2.build([None, None, self.config.d_inner])
        # 检查layer_norm是否存在
        if getattr(self, "layer_norm", None) is not None:
            # 在layer_norm的作用域内构建该层，指定输入维度为[None, None, self.config.d_model]
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.d_model])


# 定义一个名为TFFunnelLayer的自定义层，继承自keras.layers.Layer
class TFFunnelLayer(keras.layers.Layer):
    
    # 初始化函数，接收config、block_index和kwargs参数
    def __init__(self, config, block_index, **kwargs):
        super().__init__(**kwargs)
        # 创建一个TFFunnelRelMultiheadAttention层，使用config和block_index参数，命名为attention
        self.attention = TFFunnelRelMultiheadAttention(config, block_index, name="attention")
        # 创建一个TFFunnelPositionwiseFFN层，使用config参数，命名为ffn
        self.ffn = TFFunnelPositionwiseFFN(config, name="ffn")

    # 定义调用函数，接收query、key、value、attention_inputs、output_attentions和training参数
    def call(self, query, key, value, attention_inputs, output_attentions=False, training=False):
        # 使用attention层处理输入数据
        attn = self.attention(
            query, key, value, attention_inputs, output_attentions=output_attentions, training=training
        )
        # 使用ffn层处理attention的结果，根据training参数决定是否启用训练模式
        output = self.ffn(attn[0], training=training)
        # 返回output和attn[1]的元组（如果output_attentions为True），否则返回只含有output的元组
        return (output, attn[1]) if output_attentions else (output,)

    # 构建函数，接收input_shape参数，默认为None
    def build(self, input_shape=None):
        # 如果已经构建好了，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 检查attention是否存在
        if getattr(self, "attention", None) is not None:
            # 在attention的作用域内构建该层，输入形状为None
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        # 检查ffn是否存在
        if getattr(self, "ffn", None) is not None:
            # 在ffn的作用域内构建该层，输入形状为None
            with tf.name_scope(self.ffn.name):
                self.ffn.build(None)


# 定义一个名为TFFunnelEncoder的自定义层，继承自keras.layers.Layer
class TFFunnelEncoder(keras.layers.Layer):
    # 初始化函数，用于创建一个新的 TFFunnel 模型实例
    def __init__(self, config, **kwargs):
        # 调用父类（可能是超类或基类）的初始化方法，传递关键字参数
        super().__init__(**kwargs)
        # 从配置对象中获取是否分离类别信息的标志
        self.separate_cls = config.separate_cls
        # 从配置对象中获取是否仅使用问题（Query）的池化结果的标志
        self.pool_q_only = config.pool_q_only
        # 从配置对象中获取每个块重复次数的列表
        self.block_repeats = config.block_repeats
        # 使用配置对象创建 TFFunnelAttentionStructure 对象，处理注意力结构
        self.attention_structure = TFFunnelAttentionStructure(config)
        # 创建一个包含多个块的列表，每个块包含多个 TFFunnelLayer 层
        self.blocks = [
            [TFFunnelLayer(config, block_index, name=f"blocks_._{block_index}_._{i}") for i in range(block_size)]
            for block_index, block_size in enumerate(config.block_sizes)
        ]

    # 调用函数，实现 TFFunnel 模型的前向传播
    def call(
        self,
        inputs_embeds,
        attention_mask=None,
        token_type_ids=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        training=False,
    ):
        # 如果输入的注意力掩码是长张量，则需要进行类型转换，因为池化操作不适用于长张量。
        # attention_mask = tf.cast(attention_mask, inputs_embeds.dtype)
        
        # 初始化注意力输入，使用输入的嵌入向量和可能的注意力掩码、标记类型 ID 和训练标志
        attention_inputs = self.attention_structure.init_attention_inputs(
            inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            training=training,
        )
        
        # 将输入的嵌入向量赋值给隐藏状态
        hidden = inputs_embeds

        # 如果需要输出所有隐藏状态，则初始化一个列表，并将当前的输入嵌入向量添加进去
        all_hidden_states = (inputs_embeds,) if output_hidden_states else None
        
        # 如果需要输出所有注意力权重，则初始化一个空元组
        all_attentions = () if output_attentions else None

        # 遍历所有的 Transformer block
        for block_index, block in enumerate(self.blocks):
            # 判断是否需要进行池化操作，条件是隐藏状态的第二维度大于1（有多个 token），并且不是第一个 block
            pooling_flag = shape_list(hidden)[1] > (2 if self.separate_cls else 1)
            pooling_flag = pooling_flag and block_index > 0
            pooled_hidden = tf.zeros(shape_list(hidden))

            # 如果满足池化条件，则调用注意力结构的预池化函数
            if pooling_flag:
                pooled_hidden, attention_inputs = self.attention_structure.pre_attention_pooling(
                    hidden, attention_inputs
                )

            # 遍历当前 block 中的每一层
            for layer_index, layer in enumerate(block):
                # 根据 block_index 获取该 block 的重复次数
                for repeat_index in range(self.block_repeats[block_index]):
                    # 判断当前是否需要池化操作
                    do_pooling = (repeat_index == 0) and (layer_index == 0) and pooling_flag
                    if do_pooling:
                        query = pooled_hidden
                        key = value = hidden if self.pool_q_only else pooled_hidden
                    else:
                        query = key = value = hidden

                    # 调用 Transformer 层进行前向传播
                    layer_output = layer(
                        query, key, value, attention_inputs, output_attentions=output_attentions, training=training
                    )
                    hidden = layer_output[0]

                    # 如果需要池化，则调用注意力结构的后池化函数
                    if do_pooling:
                        attention_inputs = self.attention_structure.post_attention_pooling(attention_inputs)

                    # 如果需要输出注意力权重，则将当前层的注意力权重添加到 all_attentions 中
                    if output_attentions:
                        all_attentions = all_attentions + layer_output[1:]

                    # 如果需要输出所有隐藏状态，则将当前层的隐藏状态添加到 all_hidden_states 中
                    if output_hidden_states:
                        all_hidden_states = all_hidden_states + (hidden,)

        # 如果不需要以字典形式返回结果，则将结果组合成元组返回
        if not return_dict:
            return tuple(v for v in [hidden, all_hidden_states, all_attentions] if v is not None)
        
        # 如果需要以字典形式返回结果，则构建 TFBaseModelOutput 并返回
        return TFBaseModelOutput(last_hidden_state=hidden, hidden_states=all_hidden_states, attentions=all_attentions)

    # 构建方法，用于构建整个模型
    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回
        if self.built:
            return
        
        # 将模型标记为已构建状态
        self.built = True
        
        # 遍历每个 block 中的每个层，并调用其 build 方法构建层
        for block in self.blocks:
            for layer in block:
                with tf.name_scope(layer.name):
                    layer.build(None)
def upsample(x, stride, target_len, separate_cls=True, truncate_seq=False):
    """
    Upsample tensor `x` to match `target_len` by repeating the tokens `stride` time on the sequence length dimension.
    将张量 `x` 进行上采样，使其在序列长度维度上重复 `stride` 次，以匹配 `target_len` 的长度。
    """
    if stride == 1:
        return x
    if separate_cls:
        cls = x[:, :1]  # 提取张量 `x` 的第一个元素作为 cls
        x = x[:, 1:]  # 去除张量 `x` 的第一个元素后的部分
    output = tf.repeat(x, repeats=stride, axis=1)  # 在序列长度维度上重复张量 `x`，重复次数为 `stride`
    if separate_cls:
        if truncate_seq:
            output = tf.pad(output, [[0, 0], [0, stride - 1], [0, 0]])  # 如果需要截断序列，则在最后一维上进行填充
        output = output[:, : target_len - 1]  # 截取输出张量 `output` 的前 `target_len - 1` 个元素
        output = tf.concat([cls, output], axis=1)  # 将 cls 与处理后的 output 进行连接
    else:
        output = output[:, :target_len]  # 截取输出张量 `output` 的前 `target_len` 个元素
    return output  # 返回处理后的张量 `output`


class TFFunnelDecoder(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.separate_cls = config.separate_cls  # 初始化是否分离 cls 标记
        self.truncate_seq = config.truncate_seq  # 初始化是否截断序列标记
        self.stride = 2 ** (len(config.block_sizes) - 1)  # 初始化上采样步长 `stride`
        self.attention_structure = TFFunnelAttentionStructure(config)  # 初始化注意力结构
        self.layers = [TFFunnelLayer(config, 0, name=f"layers_._{i}") for i in range(config.num_decoder_layers)]  # 初始化解码器层列表

    def call(
        self,
        final_hidden,
        first_block_hidden,
        attention_mask=None,
        token_type_ids=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        training=False,
    ):
        upsampled_hidden = upsample(
            final_hidden,
            stride=self.stride,
            target_len=shape_list(first_block_hidden)[1],
            separate_cls=self.separate_cls,
            truncate_seq=self.truncate_seq,
        )  # 调用上采样函数对 final_hidden 进行处理

        hidden = upsampled_hidden + first_block_hidden  # 将上采样后的 hidden 与第一个块的 hidden 相加
        all_hidden_states = (hidden,) if output_hidden_states else None  # 如果需要输出隐藏状态，则将 hidden 存入元组
        all_attentions = () if output_attentions else None  # 如果需要输出注意力，则初始化空元组

        attention_inputs = self.attention_structure.init_attention_inputs(
            hidden,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            training=training,
        )  # 初始化注意力输入结构

        for layer in self.layers:
            layer_output = layer(
                hidden, hidden, hidden, attention_inputs, output_attentions=output_attentions, training=training
            )  # 对每一层进行处理
            hidden = layer_output[0]  # 获取每一层的输出作为下一层的输入

            if output_attentions:
                all_attentions = all_attentions + layer_output[1:]  # 如果需要输出注意力，则将每一层的注意力加入到 all_attentions 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden,)  # 如果需要输出隐藏状态，则将每一层的隐藏状态加入到 all_hidden_states 中

        if not return_dict:
            return tuple(v for v in [hidden, all_hidden_states, all_attentions] if v is not None)  # 如果不返回字典，则返回元组形式的结果
        return TFBaseModelOutput(last_hidden_state=hidden, hidden_states=all_hidden_states, attentions=all_attentions)  # 返回 TFBaseModelOutput 类的实例作为字典形式的结果

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True  # 设置标记已构建
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)  # 构建每一层
@keras_serializable
class TFFunnelBaseLayer(keras.layers.Layer):
    """Base model without decoder"""

    # 使用 FunnelConfig 类来配置模型
    config_class = FunnelConfig

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        # 将传入的配置保存到实例中
        self.config = config
        # 根据配置设置是否输出注意力权重
        self.output_attentions = config.output_attentions
        # 根据配置设置是否输出隐藏状态
        self.output_hidden_states = config.output_hidden_states
        # 根据配置设置是否返回字典形式的输出
        self.return_dict = config.use_return_dict

        # 创建嵌入层对象，并命名为 "embeddings"
        self.embeddings = TFFunnelEmbeddings(config, name="embeddings")
        # 创建编码器层对象，并命名为 "encoder"
        self.encoder = TFFunnelEncoder(config, name="encoder")

    def get_input_embeddings(self):
        # 返回嵌入层对象，用于获取输入的嵌入表示
        return self.embeddings

    def set_input_embeddings(self, value):
        # 设置嵌入层的权重为给定的值，并更新词汇表大小
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    def _prune_heads(self, heads_to_prune):
        # 当前未实现的方法，用于在 TF 2.0 模型中修剪注意力头
        raise NotImplementedError  # Not implemented yet in the library fr TF 2.0 models

    @unpack_inputs
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
    ):
        if input_ids is not None and inputs_embeds is not None:
            # 如果同时指定了 input_ids 和 inputs_embeds，则抛出错误
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            # 获取输入的形状
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            # 获取输入嵌入的形状（去除最后一个维度）
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            # 如果既没有指定 input_ids 也没有指定 inputs_embeds，则抛出错误
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            # 如果没有提供注意力掩码，则默认为全 1 的张量，形状与输入相同
            attention_mask = tf.fill(input_shape, 1)

        if token_type_ids is None:
            # 如果没有提供 token_type_ids，则默认为全 0 的张量，形状与输入相同
            token_type_ids = tf.fill(input_shape, 0)

        if inputs_embeds is None:
            # 如果未提供 inputs_embeds，则通过嵌入层获取 input_ids 的嵌入表示
            inputs_embeds = self.embeddings(input_ids, training=training)

        # 将输入嵌入传递给编码器层进行编码
        encoder_outputs = self.encoder(
            inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        return encoder_outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "embeddings", None) is not None:
            # 如果存在嵌入层对象，则构建嵌入层
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        if getattr(self, "encoder", None) is not None:
            # 如果存在编码器对象，则构建编码器
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)


@keras_serializable
class TFFunnelMainLayer(keras.layers.Layer):
    """Base model with decoder"""

    # 使用 FunnelConfig 类来配置模型
    config_class = FunnelConfig
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)  # 调用父类的初始化方法，传递任意额外的关键字参数

        self.config = config  # 将配置对象保存到实例变量中
        self.block_sizes = config.block_sizes  # 从配置中获取块大小并保存到实例变量中
        self.output_attentions = config.output_attentions  # 从配置中获取是否输出注意力权重，并保存到实例变量中
        self.output_hidden_states = config.output_hidden_states  # 从配置中获取是否输出隐藏状态，并保存到实例变量中
        self.return_dict = config.use_return_dict  # 从配置中获取是否使用返回字典，并保存到实例变量中

        self.embeddings = TFFunnelEmbeddings(config, name="embeddings")  # 使用配置创建嵌入层对象并保存到实例变量中
        self.encoder = TFFunnelEncoder(config, name="encoder")  # 使用配置创建编码器对象并保存到实例变量中
        self.decoder = TFFunnelDecoder(config, name="decoder")  # 使用配置创建解码器对象并保存到实例变量中

    def get_input_embeddings(self):
        return self.embeddings  # 返回保存的嵌入层对象

    def set_input_embeddings(self, value):
        self.embeddings.weight = value  # 设置嵌入层的权重为给定值
        self.embeddings.vocab_size = shape_list(value)[0]  # 设置嵌入层的词汇大小为给定值的形状的第一个维度大小

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError  # 抛出未实现错误，表示此方法在TF 2.0模型库中尚未实现

    @unpack_inputs
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
        ):
            # 如果同时指定了 input_ids 和 inputs_embeds，则抛出数值错误
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
            # 如果指定了 input_ids，则获取其形状
            elif input_ids is not None:
                input_shape = shape_list(input_ids)
            # 如果指定了 inputs_embeds，则获取其形状但不包括最后一维
            elif inputs_embeds is not None:
                input_shape = shape_list(inputs_embeds)[:-1]
            else:
                # 如果既没有指定 input_ids 也没有指定 inputs_embeds，则抛出数值错误
                raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            # 如果未指定 attention_mask，则用 1 填充，形状为 input_shape
            attention_mask = tf.fill(input_shape, 1)

        if token_type_ids is None:
            # 如果未指定 token_type_ids，则用 0 填充，形状为 input_shape
            token_type_ids = tf.fill(input_shape, 0)

        if inputs_embeds is None:
            # 如果未指定 inputs_embeds，则调用 self.embeddings 构建 embeddings
            inputs_embeds = self.embeddings(input_ids, training=training)

        # 使用 self.encoder 处理 inputs_embeds
        encoder_outputs = self.encoder(
            inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            training=training,
        )

        # 使用 self.decoder 处理 encoder 的输出以生成 decoder 的输出
        decoder_outputs = self.decoder(
            final_hidden=encoder_outputs[0],
            first_block_hidden=encoder_outputs[1][self.block_sizes[0]],
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        if not return_dict:
            # 如果不返回字典，则根据需要构建输出元组
            idx = 0
            outputs = (decoder_outputs[0],)
            if output_hidden_states:
                idx += 1
                outputs = outputs + (encoder_outputs[1] + decoder_outputs[idx],)
            if output_attentions:
                idx += 1
                outputs = outputs + (encoder_outputs[2] + decoder_outputs[idx],)
            return outputs

        # 如果返回字典，则构建 TFBaseModelOutput 对象作为返回值
        return TFBaseModelOutput(
            last_hidden_state=decoder_outputs[0],
            hidden_states=(encoder_outputs.hidden_states + decoder_outputs.hidden_states)
            if output_hidden_states
            else None,
            attentions=(encoder_outputs.attentions + decoder_outputs.attentions) if output_attentions else None,
        )

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记模型已构建
        self.built = True
        # 如果存在 self.embeddings 属性，则构建 embeddings
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        # 如果存在 self.encoder 属性，则构建 encoder
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果存在 self.decoder 属性，则构建 decoder
        if getattr(self, "decoder", None) is not None:
            with tf.name_scope(self.decoder.name):
                self.decoder.build(None)
class TFFunnelDiscriminatorPredictions(keras.layers.Layer):
    """Prediction module for the discriminator, made up of two dense layers."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 初始化第一个全连接层，输出维度为 config.d_model
        initializer = get_initializer(config.initializer_range)
        self.dense = keras.layers.Dense(config.d_model, kernel_initializer=initializer, name="dense")
        # 获取激活函数并存储
        self.activation_function = get_tf_activation(config.hidden_act)
        # 初始化第二个全连接层，输出维度为 1，用于预测
        self.dense_prediction = keras.layers.Dense(1, kernel_initializer=initializer, name="dense_prediction")
        # 存储配置
        self.config = config

    def call(self, discriminator_hidden_states):
        # 前向传播过程
        # 全连接层操作
        hidden_states = self.dense(discriminator_hidden_states)
        # 应用激活函数
        hidden_states = self.activation_function(hidden_states)
        # 对输出进行压缩成一维，用于预测
        logits = tf.squeeze(self.dense_prediction(hidden_states))
        return logits

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经构建，则直接返回
        # 构建第一个全连接层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.d_model])
        # 构建第二个全连接层
        if getattr(self, "dense_prediction", None) is not None:
            with tf.name_scope(self.dense_prediction.name):
                self.dense_prediction.build([None, None, self.config.d_model])


class TFFunnelMaskedLMHead(keras.layers.Layer):
    """Masked Language Model (MLM) head for TFFunnel model."""

    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)
        # 存储配置和嵌入层
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embeddings = input_embeddings

    def build(self, input_shape):
        # 创建偏置项，并且允许其训练
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")
        # 调用父类的 build 方法
        super().build(input_shape)

    def get_output_embeddings(self):
        # 返回输入嵌入层
        return self.input_embeddings

    def set_output_embeddings(self, value):
        # 设置输出嵌入层的权重和词汇大小
        self.input_embeddings.weight = value
        self.input_embeddings.vocab_size = shape_list(value)[0]

    def get_bias(self):
        # 返回偏置项
        return {"bias": self.bias}

    def set_bias(self, value):
        # 设置偏置项的值，并更新配置中的词汇大小
        self.bias = value["bias"]
        self.config.vocab_size = shape_list(value["bias"])[0]

    def call(self, hidden_states, training=False):
        # 前向传播过程
        # 获取序列长度
        seq_length = shape_list(tensor=hidden_states)[1]
        # 将隐藏状态重塑为二维张量
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.hidden_size])
        # 计算权重与输入嵌入层的乘积
        hidden_states = tf.matmul(a=hidden_states, b=self.input_embeddings.weight, transpose_b=True)
        # 将结果重新形状为三维张量
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
        # 添加偏置项
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)

        return hidden_states


class TFFunnelClassificationHead(keras.layers.Layer):
    """Classification head for TFFunnel model."""
    # 初始化方法，接收配置信息、标签数和额外的关键字参数
    def __init__(self, config, n_labels, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 根据配置中的初始化范围获取初始化器
        initializer = get_initializer(config.initializer_range)
        # 创建一个全连接层，用于隐藏层，设置输出维度为config.d_model，使用指定的初始化器
        self.linear_hidden = keras.layers.Dense(config.d_model, kernel_initializer=initializer, name="linear_hidden")
        # 创建一个Dropout层，用于隐藏层，设置丢弃率为config.hidden_dropout
        self.dropout = keras.layers.Dropout(config.hidden_dropout)
        # 创建一个全连接层，用于输出层，设置输出维度为n_labels，使用相同的初始化器
        self.linear_out = keras.layers.Dense(n_labels, kernel_initializer=initializer, name="linear_out")
        # 保存配置信息
        self.config = config

    # 前向传播方法，接收隐藏层的输入和是否处于训练模式
    def call(self, hidden, training=False):
        # 经过隐藏层的全连接操作
        hidden = self.linear_hidden(hidden)
        # 使用双曲正切激活函数处理隐藏层输出
        hidden = keras.activations.tanh(hidden)
        # 在训练时对隐藏层输出进行丢弃操作
        hidden = self.dropout(hidden, training=training)
        # 经过输出层的全连接操作，得到最终输出
        return self.linear_out(hidden)

    # 构建方法，用于构建模型的各层
    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回
        if self.built:
            return
        # 将模型标记为已构建
        self.built = True
        # 如果存在隐藏层，按照指定的形状构建全连接层
        if getattr(self, "linear_hidden", None) is not None:
            with tf.name_scope(self.linear_hidden.name):
                self.linear_hidden.build([None, None, self.config.d_model])
        # 如果存在输出层，按照指定的形状构建全连接层
        if getattr(self, "linear_out", None) is not None:
            with tf.name_scope(self.linear_out.name):
                self.linear_out.build([None, None, self.config.d_model])
    @staticmethod
    def convert_attention_mask(attention_mask: tf.Tensor, dtype: tf.DType = tf.float32) -> tf.Tensor:
        """
        Converts a 2D Tensor to a boolean mask with shape [batch_size, 1, 1, sequence_length].

        Args:
            attention_mask (:obj:`tf.Tensor`): The attention mask.
            dtype (:obj:`tf.DType`, `optional`, defaults to :obj:`tf.float32`):
                The datatype of the resulting mask tensor.

        Returns:
            :obj:`tf.Tensor`: The boolean mask tensor.
        """
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


    Parameters:
        config ([`XxxConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
这是一个长字符串，用于文档化函数参数说明。
详细说明了模型输入的各个参数及其形状和含义。
"""

@add_start_docstrings(
    """
    基础的Funnel Transformer模型，输出原始隐藏状态，没有上采样头（也称为解码器）或任何特定任务的头部。
    """,
    FUNNEL_START_DOCSTRING,
)
class TFFunnelBaseModel(TFFunnelPreTrainedModel):
    """
    Funnel Transformer模型的基类，继承自TFFunnelPreTrainedModel。

    继承自TFFunnelPreTrainedModel的功能和属性将被此基类继承和使用。
    """
    # 初始化函数，用于创建一个新的Funnel模型实例
    def __init__(self, config: FunnelConfig, *inputs, **kwargs) -> None:
        # 调用父类的初始化函数，传入配置和其他可变参数
        super().__init__(config, *inputs, **kwargs)
        # 创建一个TFFunnelBaseLayer的实例作为该模型的核心组件，命名为"funnel"
        self.funnel = TFFunnelBaseLayer(config, name="funnel")

    # 调用函数，将输入传递给funnel模型的前向方法，返回模型输出
    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="funnel-transformer/small-base",
        output_type=TFBaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[Tuple[tf.Tensor], TFBaseModelOutput]:
        # 调用self.funnel的call方法，将各种输入参数传递给Funnel模型
        return self.funnel(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

    # serving_output函数，用于生成模型服务的输出
    def serving_output(self, output):
        # 创建TFBaseModelOutput实例作为输出，包含last_hidden_state、hidden_states和attentions
        # 注意：hidden_states和attentions未使用tf.convert_to_tensor转换，因为它们维度不同
        return TFBaseModelOutput(
            last_hidden_state=output.last_hidden_state,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )

    # build函数，用于构建模型，设置各个组件的连接和初始化
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        # 将模型标记为已构建
        self.built = True
        # 如果存在self.funnel属性，则在tf的命名作用域内构建funnel模型
        if getattr(self, "funnel", None) is not None:
            with tf.name_scope(self.funnel.name):
                self.funnel.build(None)
@add_start_docstrings(
    """
    The bare Funnel Transformer Model transformer outputting raw hidden-states without any specific head on top.
    """,
    FUNNEL_START_DOCSTRING,
)
class TFFunnelModel(TFFunnelPreTrainedModel):
    """
    Funnel Transformer model for processing raw hidden-states without additional heads.

    Args:
        config (FunnelConfig): The model configuration class instance.

    Attributes:
        funnel (TFFunnelMainLayer): The main layer of the Funnel Transformer.

    """
    def __init__(self, config: FunnelConfig, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        # Initialize Funnel main layer
        self.funnel = TFFunnelMainLayer(config, name="funnel")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="funnel-transformer/small",
        output_type=TFBaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[Tuple[tf.Tensor], TFBaseModelOutput]:
        """
        Perform the forward pass of the Funnel model.

        Args:
            input_ids (TFModelInputType | None): Input token IDs.
            attention_mask (np.ndarray | tf.Tensor | None): Mask for attention scores.
            token_type_ids (np.ndarray | tf.Tensor | None): Segment token indices.
            inputs_embeds (np.ndarray | tf.Tensor | None): Embedded inputs.
            output_attentions (Optional[bool]): Whether to output attentions.
            output_hidden_states (Optional[bool]): Whether to output hidden states.
            return_dict (Optional[bool]): Whether to return as dictionary.
            training (bool): Whether in training mode.

        Returns:
            Union[Tuple[tf.Tensor], TFBaseModelOutput]: The model outputs.

        """
        return self.funnel(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

    def serving_output(self, output):
        """
        Format the model output for serving.

        Args:
            output: Output from the model.

        Returns:
            TFBaseModelOutput: Formatted output for serving.

        """
        # Ensure compatibility for non-tensor outputs
        return TFBaseModelOutput(
            last_hidden_state=output.last_hidden_state,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )

    def build(self, input_shape=None):
        """
        Build the model layers.

        Args:
            input_shape: Shape of the input tensor.

        """
        if self.built:
            return
        self.built = True
        if getattr(self, "funnel", None) is not None:
            with tf.name_scope(self.funnel.name):
                self.funnel.build(None)


@add_start_docstrings(
    """
    Funnel model with a binary classification head on top as used during pretraining for identifying generated tokens.
    """,
    FUNNEL_START_DOCSTRING,
)
class TFFunnelForPreTraining(TFFunnelPreTrainedModel):
    """
    Funnel Transformer model for pretraining with a binary classification head.

    Args:
        config (FunnelConfig): The model configuration class instance.

    Attributes:
        funnel (TFFunnelMainLayer): The main layer of the Funnel Transformer.
        discriminator_predictions (TFFunnelDiscriminatorPredictions): Predictions layer for discriminator.

    """
    def __init__(self, config: FunnelConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)

        # Initialize Funnel main layer and discriminator predictions layer
        self.funnel = TFFunnelMainLayer(config, name="funnel")
        self.discriminator_predictions = TFFunnelDiscriminatorPredictions(config, name="discriminator_predictions")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFFunnelForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        **kwargs,
    ) -> Union[Tuple[tf.Tensor], TFFunnelForPreTrainingOutput]:
        r"""
        模型调用方法，接收多个输入参数，生成预测输出或模型状态。

        Returns:
            返回一个元组或 TFFunnelForPreTrainingOutput 对象，包含模型的输出 logits 和可能的状态信息。

        Examples:
        
        ```python
        >>> from transformers import AutoTokenizer, TFFunnelForPreTraining
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("funnel-transformer/small")
        >>> model = TFFunnelForPreTraining.from_pretrained("funnel-transformer/small")

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
        >>> logits = model(inputs).logits
        ```"""
        # 使用输入调用模型的主干网络（如 Funnel），生成鉴别器的隐藏状态
        discriminator_hidden_states = self.funnel(
            input_ids,
            attention_mask,
            token_type_ids,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 获取鉴别器的序列输出（通常是隐藏状态的第一个元素）
        discriminator_sequence_output = discriminator_hidden_states[0]
        # 将鉴别器序列输出传递给鉴别器预测模块，生成最终的预测 logits
        logits = self.discriminator_predictions(discriminator_sequence_output)

        # 如果不要求返回字典形式的输出，则返回 logits 和其它鉴别器隐藏状态
        if not return_dict:
            return (logits,) + discriminator_hidden_states[1:]

        # 否则，返回包含 logits、隐藏状态和注意力权重的 TFFunnelForPreTrainingOutput 对象
        return TFFunnelForPreTrainingOutput(
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )

    def serving_output(self, output):
        # 输出服务化接口，不将 hidden_states 和 attentions 转换为 Tensor，因为它们具有不同的维度
        return TFFunnelForPreTrainingOutput(
            logits=output.logits, hidden_states=output.hidden_states, attentions=output.attentions
        )

    def build(self, input_shape=None):
        # 模型构建方法，如果已经构建过则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在主干网络 (funnel)，则在命名空间下构建它
        if getattr(self, "funnel", None) is not None:
            with tf.name_scope(self.funnel.name):
                self.funnel.build(None)
        # 如果存在鉴别器预测模块，则在命名空间下构建它
        if getattr(self, "discriminator_predictions", None) is not None:
            with tf.name_scope(self.discriminator_predictions.name):
                self.discriminator_predictions.build(None)
@add_start_docstrings("""Funnel Model with a `language modeling` head on top.""", FUNNEL_START_DOCSTRING)
class TFFunnelForMaskedLM(TFFunnelPreTrainedModel, TFMaskedLanguageModelingLoss):
    def __init__(self, config: FunnelConfig, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)

        # 初始化 Funnel 主层，并命名为 "funnel"
        self.funnel = TFFunnelMainLayer(config, name="funnel")
        # 初始化 Funnel Masked LM Head，并关联到 Funnel embeddings，命名为 "lm_head"
        self.lm_head = TFFunnelMaskedLMHead(config, self.funnel.embeddings, name="lm_head")

    def get_lm_head(self) -> TFFunnelMaskedLMHead:
        # 返回 Funnel Masked LM Head 对象
        return self.lm_head

    def get_prefix_bias_name(self) -> str:
        # 发出警告，指出方法 get_prefix_bias_name 已被弃用，建议使用 `get_bias`
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        # 返回 lm_head 对象的名称前缀，与当前对象名称组合而成的字符串
        return self.name + "/" + self.lm_head.name

    @unpack_inputs
    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="funnel-transformer/small",
        output_type=TFMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: bool = False,
    ) -> Union[Tuple[tf.Tensor], TFMaskedLMOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        # 调用 Funnel 主层进行模型前向传播
        outputs = self.funnel(
            input_ids,
            attention_mask,
            token_type_ids,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 获取序列输出（即模型输出的第一个元素）
        sequence_output = outputs[0]
        # 使用 lm_head 处理序列输出，得到预测分数
        prediction_scores = self.lm_head(sequence_output, training=training)

        # 如果没有传入 labels，则损失为 None；否则计算 masked language modeling 损失
        loss = None if labels is None else self.hf_compute_loss(labels, prediction_scores)

        # 如果不需要返回字典，则返回 tuple 格式的输出
        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典格式的输出，则构建 TFMaskedLMOutput 对象
        return TFMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    # 定义一个方法用于处理输出的 TFMaskedLMOutput 对象，输入和输出都是 TFMaskedLMOutput 类型
    def serving_output(self, output: TFMaskedLMOutput) -> TFMaskedLMOutput:
        # 不将 hidden_states 和 attentions 转换为 Tensor，因为它们的维度各不相同
        # output.logits 是输出的对数概率
        return TFMaskedLMOutput(logits=output.logits, hidden_states=output.hidden_states, attentions=output.attentions)

    # 构建方法，用于构建模型
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True  # 标记模型已经构建

        # 如果有 funnel 属性，构建 funnel
        if getattr(self, "funnel", None) is not None:
            with tf.name_scope(self.funnel.name):
                self.funnel.build(None)  # 调用 funnel 的 build 方法

        # 如果有 lm_head 属性，构建 lm_head
        if getattr(self, "lm_head", None) is not None:
            with tf.name_scope(self.lm_head.name):
                self.lm_head.build(None)  # 调用 lm_head 的 build 方法
@add_start_docstrings(
    """
    Funnel Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    FUNNEL_START_DOCSTRING,
)
class TFFunnelForSequenceClassification(TFFunnelPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: FunnelConfig, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        # 初始化Funnel模型的基础层
        self.funnel = TFFunnelBaseLayer(config, name="funnel")
        # 初始化Funnel模型的分类头部
        self.classifier = TFFunnelClassificationHead(config, config.num_labels, name="classifier")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="funnel-transformer/small-base",
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: bool = False,
    ) -> Union[Tuple[tf.Tensor], TFSequenceClassifierOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 调用Funnel模型的前向传播
        outputs = self.funnel(
            input_ids,
            attention_mask,
            token_type_ids,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 获取最后一层隐藏状态
        last_hidden_state = outputs[0]
        # 获取汇聚的输出
        pooled_output = last_hidden_state[:, 0]
        # 通过分类器预测logits
        logits = self.classifier(pooled_output, training=training)

        # 计算损失，如果提供了标签
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果不要求返回字典，则返回元组形式的输出
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回TFSequenceClassifierOutput对象，包括损失、logits、隐藏状态和注意力分布
        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    # 处理模型输出，不对 hidden_states 和 attentions 使用 tf.convert_to_tensor 转换，
    # 因为它们的维度不同
    def serving_output(self, output: TFSequenceClassifierOutput) -> TFSequenceClassifierOutput:
        # 返回一个新的 TFSequenceClassifierOutput 对象，保留 logits、hidden_states 和 attentions
        return TFSequenceClassifierOutput(
            logits=output.logits, hidden_states=output.hidden_states, attentions=output.attentions
        )

    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建，直接返回
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        # 如果存在 self.funnel 属性，则构建 self.funnel
        if getattr(self, "funnel", None) is not None:
            # 使用 self.funnel 的名称作为命名空间
            with tf.name_scope(self.funnel.name):
                # 调用 self.funnel 的 build 方法
                self.funnel.build(None)
        # 如果存在 self.classifier 属性，则构建 self.classifier
        if getattr(self, "classifier", None) is not None:
            # 使用 self.classifier 的名称作为命名空间
            with tf.name_scope(self.classifier.name):
                # 调用 self.classifier 的 build 方法
                self.classifier.build(None)
@add_start_docstrings(
    """
    Funnel Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    FUNNEL_START_DOCSTRING,
)
class TFFunnelForMultipleChoice(TFFunnelPreTrainedModel, TFMultipleChoiceLoss):
    """
    使用 Funnel 模型，并在其顶部添加一个多选分类头部（一个线性层位于汇总输出之上，并带有 softmax），例如用于 RocStories/SWAG 任务。
    继承自 TFFunnelPreTrainedModel 和 TFMultipleChoiceLoss。
    """

    def __init__(self, config: FunnelConfig, *inputs, **kwargs) -> None:
        """
        初始化方法，设置模型的配置参数和输入。

        Args:
            config (FunnelConfig): Funnel 模型的配置对象。
            *inputs: 可变位置参数，传递给父类构造函数。
            **kwargs: 关键字参数，传递给父类构造函数。
        """
        super().__init__(config, *inputs, **kwargs)

        # 创建 Funnel 的基础层对象，命名为 "funnel"
        self.funnel = TFFunnelBaseLayer(config, name="funnel")
        # 创建 Funnel 分类头部对象，用于多选分类，输出维度为 1，命名为 "classifier"
        self.classifier = TFFunnelClassificationHead(config, 1, name="classifier")

    @property
    def dummy_inputs(self):
        """
        返回一个字典，包含用于模型前向传播的虚拟输入数据。

        Returns:
            dict: 包含虚拟输入数据的字典，键为 "input_ids"，值为形状为 (3, 3, 4) 的 tf.Tensor。
        """
        return {"input_ids": tf.ones((3, 3, 4), dtype=tf.int32)}

    @unpack_inputs
    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="funnel-transformer/small-base",
        output_type=TFMultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: bool = False,
        **kwargs
    ) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
        """
        模型前向传播方法，接受多种输入和控制参数。

        Args:
            input_ids (TFModelInputType, optional): 输入的 token IDs，形状为 (batch_size, num_choices, sequence_length)。
            attention_mask (np.ndarray or tf.Tensor, optional): 注意力掩码，形状与 input_ids 相同。
            token_type_ids (np.ndarray or tf.Tensor, optional): token 类型 IDs，形状与 input_ids 相同。
            inputs_embeds (np.ndarray or tf.Tensor, optional): 嵌入输入，形状为 (batch_size, num_choices, sequence_length, embedding_dim)。
            output_attentions (bool, optional): 是否返回注意力权重。
            output_hidden_states (bool, optional): 是否返回隐藏状态。
            return_dict (bool, optional): 是否返回字典形式的输出。
            labels (np.ndarray or tf.Tensor, optional): 分类标签，形状为 (batch_size, num_choices)。
            training (bool, optional): 是否为训练模式。

        Returns:
            Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]: 返回模型的输出结果。
        """
        # 函数实现由装饰器 add_start_docstrings_to_model_forward 和 add_code_sample_docstrings 添加的文档字符串提供详细信息。
        pass  # 实际上的前向传播逻辑在具体的调用中执行，这里暂时不做任何操作，保留 pass 语句。
    ) -> Union[Tuple[tf.Tensor], TFMultipleChoiceModelOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)
        """
        # 如果提供了 `input_ids`，则获取其第二个维度的大小作为选择数量，第三个维度的大小作为序列长度
        if input_ids is not None:
            num_choices = shape_list(input_ids)[1]
            seq_length = shape_list(input_ids)[2]
        else:
            # 如果未提供 `input_ids`，则使用 `inputs_embeds` 的第二个和第三个维度作为选择数量和序列长度
            num_choices = shape_list(inputs_embeds)[1]
            seq_length = shape_list(inputs_embeds)[2]

        # 将输入张量展平成二维张量，以便与模型处理的期望形状匹配
        flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
        flat_attention_mask = tf.reshape(attention_mask, (-1, seq_length)) if attention_mask is not None else None
        flat_token_type_ids = tf.reshape(token_type_ids, (-1, seq_length)) if token_type_ids is not None else None
        flat_inputs_embeds = (
            tf.reshape(inputs_embeds, (-1, seq_length, shape_list(inputs_embeds)[3]))
            if inputs_embeds is not None
            else None
        )

        # 调用模型的前向传播函数 `funnel`，传递展平后的输入张量和其他相关参数
        outputs = self.funnel(
            flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 从模型输出中获取最后一层隐藏状态和池化输出
        last_hidden_state = outputs[0]
        pooled_output = last_hidden_state[:, 0]

        # 使用分类器模型 `classifier` 对池化输出进行分类预测
        logits = self.classifier(pooled_output, training=training)

        # 将 logits 重新形状为二维张量，以匹配多选选择的期望形状
        reshaped_logits = tf.reshape(logits, (-1, num_choices))

        # 如果未提供标签 `labels`，则损失值为 None；否则使用 `hf_compute_loss` 函数计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, reshaped_logits)

        # 如果 `return_dict` 为 False，则返回一个元组，包含损失值和模型输出的其他部分
        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果 `return_dict` 为 True，则返回一个 `TFMultipleChoiceModelOutput` 对象，包含损失值、logits、隐藏状态和注意力权重
        return TFMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def serving_output(self, output: TFMultipleChoiceModelOutput) -> TFMultipleChoiceModelOutput:
        # 作为服务输出，直接将给定的输出对象中的 logits、hidden_states 和 attentions 作为输出返回
        return TFMultipleChoiceModelOutput(
            logits=output.logits, hidden_states=output.hidden_states, attentions=output.attentions
        )
    # 定义 build 方法，用于构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        # 设置模型状态为已构建
        self.built = True
        
        # 如果模型中有名为 "funnel" 的子模型存在
        if getattr(self, "funnel", None) is not None:
            # 在命名空间下构建 "funnel" 子模型
            with tf.name_scope(self.funnel.name):
                self.funnel.build(None)
        
        # 如果模型中有名为 "classifier" 的子模型存在
        if getattr(self, "classifier", None) is not None:
            # 在命名空间下构建 "classifier" 子模型
            with tf.name_scope(self.classifier.name):
                self.classifier.build(None)
@add_start_docstrings(
    """
    Funnel Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    FUNNEL_START_DOCSTRING,
)
class TFFunnelForTokenClassification(TFFunnelPreTrainedModel, TFTokenClassificationLoss):
    def __init__(self, config: FunnelConfig, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels  # 初始化模型的标签数量

        self.funnel = TFFunnelMainLayer(config, name="funnel")  # 创建主要的Funnel层
        self.dropout = keras.layers.Dropout(config.hidden_dropout)  # 设置dropout层
        self.classifier = keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )  # 设置分类器，用于将隐藏状态输出映射到标签空间
        self.config = config  # 存储配置信息

    @unpack_inputs
    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="funnel-transformer/small",
        output_type=TFTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: bool = False,
    ) -> Union[Tuple[tf.Tensor], TFTokenClassifierOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        outputs = self.funnel(
            input_ids,
            attention_mask,
            token_type_ids,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict=return_dict,
            training=training,
        )  # 调用Funnel模型的前向传播

        sequence_output = outputs[0]  # 获取模型输出的序列隐藏状态

        sequence_output = self.dropout(sequence_output, training=training)  # 在训练时应用dropout

        logits = self.classifier(sequence_output)  # 将序列隐藏状态映射到标签空间的logits

        loss = None if labels is None else self.hf_compute_loss(labels, logits)  # 如果有标签，则计算损失

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output  # 如果不返回字典，则返回元组形式的输出

        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )  # 如果返回字典，则使用TFTokenClassifierOutput包装输出
    # 定义一个方法，用于处理模型的输出，将其转换为 TFTokenClassifierOutput 类型
    def serving_output(self, output: TFTokenClassifierOutput) -> TFTokenClassifierOutput:
        # 由于 hidden_states 和 attentions 的维度不同，并非所有都可以通过 tf.convert_to_tensor 转换为张量
        # 所以这里不对它们进行转换
        return TFTokenClassifierOutput(
            logits=output.logits, hidden_states=output.hidden_states, attentions=output.attentions
        )

    # 定义一个方法，用于构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建过，直接返回
        if self.built:
            return
        # 标记模型为已构建状态
        self.built = True
        # 如果存在名为 "funnel" 的属性，构建 funnel 模型
        if getattr(self, "funnel", None) is not None:
            with tf.name_scope(self.funnel.name):
                self.funnel.build(None)
        # 如果存在名为 "classifier" 的属性，构建 classifier 模型
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                # 构建 classifier 模型，输入形状为 [None, None, self.config.hidden_size]
                self.classifier.build([None, None, self.config.hidden_size])
# 使用装饰器添加模型文档字符串，描述 Funnel 模型在提取式问答任务（如 SQuAD）上的用途
@add_start_docstrings(
    """
    Funnel Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    FUNNEL_START_DOCSTRING,  # 引用已定义的 FUNNEL_START_DOCSTRING
)
# 定义 TFFunnelForQuestionAnswering 类，继承自 TFFunnelPreTrainedModel 和 TFQuestionAnsweringLoss
class TFFunnelForQuestionAnswering(TFFunnelPreTrainedModel, TFQuestionAnsweringLoss):
    # 初始化方法
    def __init__(self, config: FunnelConfig, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        # 设置模型的标签数目
        self.num_labels = config.num_labels

        # 创建 Funnel 主层，并命名为 "funnel"
        self.funnel = TFFunnelMainLayer(config, name="funnel")
        
        # 创建用于问答输出的 Dense 层，输出维度为 config.num_labels，使用指定的初始化器
        self.qa_outputs = keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )
        
        # 保存配置信息
        self.config = config

    # 使用装饰器定义 call 方法，用于模型的前向传播
    @unpack_inputs
    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="funnel-transformer/small",
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        start_positions: np.ndarray | tf.Tensor | None = None,
        end_positions: np.ndarray | tf.Tensor | None = None,
        training: bool = False,
    ) -> Union[Tuple[tf.Tensor], TFQuestionAnsweringModelOutput]:
        r"""
        start_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """

        outputs = self.funnel(
            input_ids,
            attention_mask,
            token_type_ids,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 获取模型的输出序列表示
        sequence_output = outputs[0]

        # 通过全连接层获取起始位置和结束位置的预测分数
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        loss = None
        # 如果提供了起始位置和结束位置的标签，则计算损失
        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions, "end_position": end_positions}
            loss = self.hf_compute_loss(labels, (start_logits, end_logits))

        # 如果不要求返回字典，则返回起始位置和结束位置的预测分数以及额外的输出
        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TFQuestionAnsweringModelOutput 对象，包含损失、起始位置预测分数、结束位置预测分数、隐藏状态和注意力权重
        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def serving_output(self, output: TFQuestionAnsweringModelOutput) -> TFQuestionAnsweringModelOutput:
        # 针对服务输出，直接复制输入的 TFQuestionAnsweringModelOutput 对象
        # 不转换 hidden_states 和 attentions 到 Tensor，因为它们具有不同的维度
        return TFQuestionAnsweringModelOutput(
            start_logits=output.start_logits,
            end_logits=output.end_logits,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果模型具有 "funnel" 属性，则构建 "funnel" 模型
        if getattr(self, "funnel", None) is not None:
            with tf.name_scope(self.funnel.name):
                self.funnel.build(None)
        # 如果模型具有 "qa_outputs" 属性，则构建 "qa_outputs" 层
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])
```