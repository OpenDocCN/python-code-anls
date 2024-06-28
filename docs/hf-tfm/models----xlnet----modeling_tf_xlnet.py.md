# `.\models\xlnet\modeling_tf_xlnet.py`

```
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""
 TF 2.0 XLNet model.
"""


from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFModelInputType,
    TFMultipleChoiceLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFSequenceSummary,
    TFSharedEmbeddings,
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
from .configuration_xlnet import XLNetConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "xlnet/xlnet-base-cased"
_CONFIG_FOR_DOC = "XLNetConfig"

TF_XLNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "xlnet/xlnet-base-cased",
    "xlnet/xlnet-large-cased",
    # See all XLNet models at https://huggingface.co/models?filter=xlnet
]


class TFXLNetRelativeAttention(keras.layers.Layer):
    """
    相对注意力层的 TensorFlow 2.0 实现。

    Args:
        config (XLNetConfig): XLNet 模型的配置对象。

    Raises:
        ValueError: 如果配置中的隐藏大小不是注意力头数的倍数。

    Attributes:
        n_head (int): 注意力头的数量。
        d_head (int): 每个注意力头的隐藏大小。
        d_model (int): 模型的隐藏大小。
        scale (float): 缩放因子，用于注意力计算。
        initializer_range (float): 初始化范围。
        output_attentions (bool): 是否输出注意力权重。
        layer_norm (keras.layers.LayerNormalization): 应用在每个子层输出上的层归一化层。
        dropout (keras.layers.Dropout): 用于应用 dropout 的层。
        config (XLNetConfig): XLNet 模型的配置对象。
    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        if config.d_model % config.n_head != 0:
            raise ValueError(
                f"The hidden size ({config.d_model}) is not a multiple of the number of attention "
                f"heads ({config.n_head}"
            )

        self.n_head = config.n_head
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.scale = 1 / (config.d_head**0.5)
        self.initializer_range = config.initializer_range
        self.output_attentions = config.output_attentions

        self.layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        self.dropout = keras.layers.Dropout(config.dropout)
        self.config = config
    # 在神经网络模型中建立权重参数，用于自注意力机制的构建
    def build(self, input_shape=None):
        # 根据指定的初始化范围获取初始化器
        initializer = get_initializer(self.initializer_range)
        # 添加查询向量权重矩阵 q
        self.q = self.add_weight(
            shape=(self.d_model, self.n_head, self.d_head), initializer=initializer, trainable=True, name="q"
        )
        # 添加键向量权重矩阵 k
        self.k = self.add_weight(
            shape=(self.d_model, self.n_head, self.d_head), initializer=initializer, trainable=True, name="k"
        )
        # 添加值向量权重矩阵 v
        self.v = self.add_weight(
            shape=(self.d_model, self.n_head, self.d_head), initializer=initializer, trainable=True, name="v"
        )
        # 添加输出向量权重矩阵 o
        self.o = self.add_weight(
            shape=(self.d_model, self.n_head, self.d_head), initializer=initializer, trainable=True, name="o"
        )
        # 添加相对位置编码向量权重矩阵 r
        self.r = self.add_weight(
            shape=(self.d_model, self.n_head, self.d_head), initializer=initializer, trainable=True, name="r"
        )
        # 添加相对位置编码的尾部-尾部偏置矩阵 r_r_bias
        self.r_r_bias = self.add_weight(
            shape=(self.n_head, self.d_head), initializer="zeros", trainable=True, name="r_r_bias"
        )
        # 添加相对位置编码的尾部-序列偏置矩阵 r_s_bias
        self.r_s_bias = self.add_weight(
            shape=(self.n_head, self.d_head), initializer="zeros", trainable=True, name="r_s_bias"
        )
        # 添加相对位置编码的尾部-权重偏置矩阵 r_w_bias
        self.r_w_bias = self.add_weight(
            shape=(self.n_head, self.d_head), initializer="zeros", trainable=True, name="r_w_bias"
        )
        # 添加分段嵌入向量权重矩阵 seg_embed
        self.seg_embed = self.add_weight(
            shape=(2, self.n_head, self.d_head), initializer=initializer, trainable=True, name="seg_embed"
        )

        # 如果已经构建过网络，直接返回
        if self.built:
            return
        # 标记网络已构建
        self.built = True
        # 如果存在层归一化，则对其进行构建
        if getattr(self, "layer_norm", None) is not None:
            # 在指定的命名域下构建层归一化，设置输入形状为 [None, None, self.config.d_model]
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.d_model])

    # 剪枝指定的注意力头，但未实现具体功能
    def prune_heads(self, heads):
        raise NotImplementedError

    # 执行相对偏移以形成相对注意力分数
    def rel_shift(self, x, klen=-1):
        """perform relative shift to form the relative attention score."""
        # 获取张量 x 的形状列表
        x_size = shape_list(x)

        # 将张量 x 重塑为新的形状
        x = tf.reshape(x, (x_size[1], x_size[0], x_size[2], x_size[3]))
        # 从第二个元素开始切片，实现相对偏移
        x = x[1:, ...]
        # 再次重塑张量 x
        x = tf.reshape(x, (x_size[0], x_size[1] - 1, x_size[2], x_size[3]))
        # 切片以控制长度为 klen
        x = x[:, 0:klen, :, :]
        # 返回处理后的张量 x
        return x

    # 执行相对注意力的核心计算
    def rel_attn_core(
        self, q_head, k_head_h, v_head_h, k_head_r, seg_mat, attn_mask, head_mask, output_attentions, training=False
    ):
        # 此方法的具体实现逻辑未提供，用于执行相对注意力的核心计算
    ):
        """Core relative positional attention operations."""
        # 计算基于内容的注意力分数
        ac = tf.einsum("ibnd,jbnd->ijbn", q_head + self.r_w_bias, k_head_h)

        # 计算基于位置的注意力分数
        bd = tf.einsum("ibnd,jbnd->ijbn", q_head + self.r_r_bias, k_head_r)
        bd = self.rel_shift(bd, klen=shape_list(ac)[1])

        # 计算基于段落的注意力分数
        if seg_mat is None:
            ef = 0
        else:
            ef = tf.einsum("ibnd,snd->ibns", q_head + self.r_s_bias, self.seg_embed)
            ef = tf.einsum("ijbs,ibns->ijbn", seg_mat, ef)

        # 合并注意力分数并执行掩码处理
        attn_score = (ac + bd + ef) * self.scale
        if attn_mask is not None:
            # 根据掩码类型进行不同的处理
            if attn_mask.dtype == tf.float16 or attn_mask.dtype == tf.bfloat16:
                attn_score = attn_score - 65500 * attn_mask
            else:
                attn_score = attn_score - 1e30 * attn_mask

        # 计算注意力概率
        attn_prob = stable_softmax(attn_score, axis=1)

        attn_prob = self.dropout(attn_prob, training=training)

        # 如果需要，对注意力头进行掩码处理
        if head_mask is not None:
            attn_prob = attn_prob * head_mask

        # 计算注意力输出向量
        attn_vec = tf.einsum("ijbn,jbnd->ibnd", attn_prob, v_head_h)

        if output_attentions:
            return attn_vec, attn_prob

        # 返回注意力输出向量
        return attn_vec

    def post_attention(self, h, attn_vec, residual=True, training=False):
        """Post-attention processing."""
        # 后处理注意力向量，投影回 `d_model` 空间
        attn_out = tf.einsum("ibnd,hnd->ibh", attn_vec, self.o)

        attn_out = self.dropout(attn_out, training=training)

        if residual:
            attn_out = attn_out + h
        output = self.layer_norm(attn_out)

        return output

    def call(
        self,
        h,
        g,
        attn_mask_h,
        attn_mask_g,
        r,
        seg_mat,
        mems: np.ndarray | tf.Tensor | None = None,
        target_mapping: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = False,
        training: bool = False,
# 定义一个自定义的XLNet层，继承自Keras的Layer类
class TFXLNetFeedForward(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # LayerNormalization层，用于归一化输入数据
        self.layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        # 第一个全连接层，用于特征提取
        self.layer_1 = keras.layers.Dense(
            config.d_inner, kernel_initializer=get_initializer(config.initializer_range), name="layer_1"
        )
        # 第二个全连接层，用于映射到输出维度
        self.layer_2 = keras.layers.Dense(
            config.d_model, kernel_initializer=get_initializer(config.initializer_range), name="layer_2"
        )
        # Dropout层，用于防止过拟合
        self.dropout = keras.layers.Dropout(config.dropout)
        # 根据配置选择激活函数
        if isinstance(config.ff_activation, str):
            self.activation_function = get_tf_activation(config.ff_activation)
        else:
            self.activation_function = config.ff_activation
        # 保存配置信息
        self.config = config

    # 定义层的前向传播过程
    def call(self, inp, training=False):
        output = inp
        output = self.layer_1(output)  # 第一个全连接层
        output = self.activation_function(output)  # 激活函数
        output = self.dropout(output, training=training)  # Dropout层
        output = self.layer_2(output)  # 第二个全连接层
        output = self.dropout(output, training=training)  # 再次应用Dropout层
        output = self.layer_norm(output + inp)  # 残差连接和LayerNormalization
        return output

    # 构建层，初始化各子层
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建LayerNormalization层
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.d_model])
        # 构建第一个全连接层
        if getattr(self, "layer_1", None) is not None:
            with tf.name_scope(self.layer_1.name):
                self.layer_1.build([None, None, self.config.d_model])
        # 构建第二个全连接层
        if getattr(self, "layer_2", None) is not None:
            with tf.name_scope(self.layer_2.name):
                self.layer_2.build([None, None, self.config.d_inner])


# 定义一个XLNet层，继承自Keras的Layer类
class TFXLNetLayer(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 相对注意力机制层
        self.rel_attn = TFXLNetRelativeAttention(config, name="rel_attn")
        # 前馈神经网络层
        self.ff = TFXLNetFeedForward(config, name="ff")
        # Dropout层
        self.dropout = keras.layers.Dropout(config.dropout)

    # 定义层的前向传播过程
    def call(
        self,
        output_h,
        output_g,
        non_tgt_mask,
        attn_mask,
        pos_emb,
        seg_mat,
        mems: np.ndarray | tf.Tensor | None = None,
        target_mapping: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = False,
        training: bool = False,
    ):
        # 调用相对注意力模块计算输出
        outputs = self.rel_attn(
            output_h,
            output_g,
            non_tgt_mask,
            attn_mask,
            pos_emb,
            seg_mat,
            mems,
            target_mapping,
            head_mask,
            output_attentions,
            training=training,
        )
        # 分离输出中的 h 和 g
        output_h, output_g = outputs[:2]

        # 如果存在 output_g，则通过前馈网络处理
        if output_g is not None:
            output_g = self.ff(output_g, training=training)
        
        # 通过前馈网络处理 output_h
        output_h = self.ff(output_h, training=training)

        # 如果 outputs 还包含额外的注意力信息，则重新加入输出中
        outputs = (output_h, output_g) + outputs[2:]  # 如果有额外的注意力信息，再次添加到输出中
        return outputs

    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        
        # 设置为已构建状态
        self.built = True
        
        # 如果存在相对注意力模块，构建其子模块
        if getattr(self, "rel_attn", None) is not None:
            with tf.name_scope(self.rel_attn.name):
                self.rel_attn.build(None)
        
        # 如果存在前馈网络模块，构建其子模块
        if getattr(self, "ff", None) is not None:
            with tf.name_scope(self.ff.name):
                self.ff.build(None)
class TFXLNetLMHead(keras.layers.Layer):
    # TFXLNetLMHead 类定义，继承自 keras.layers.Layer

    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)
        # 初始化方法，接受 config 和 input_embeddings 参数
        self.config = config
        # 将 config 参数赋值给实例变量 self.config
        # 输出权重与输入嵌入相同，但每个标记都有一个仅输出的偏置
        self.input_embeddings = input_embeddings
        # 将 input_embeddings 参数赋值给实例变量 self.input_embeddings

    def build(self, input_shape):
        # build 方法，用于构建层，在此处添加权重
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")
        # 添加一个名为 bias 的权重，形状为 (config.vocab_size,)
        super().build(input_shape)
        # 调用父类的 build 方法，传入 input_shape 参数

    def get_output_embeddings(self):
        # 返回输入嵌入 self.input_embeddings
        return self.input_embeddings

    def set_output_embeddings(self, value):
        # 设置输出嵌入的值，并更新 vocab_size
        self.input_embeddings.weight = value
        self.input_embeddings.vocab_size = shape_list(value)[0]

    def get_bias(self):
        # 返回偏置 self.bias
        return {"bias": self.bias}

    def set_bias(self, value):
        # 设置偏置 self.bias 的值，并更新 config.vocab_size
        self.bias = value["bias"]
        self.config.vocab_size = shape_list(value["bias"])[0]

    def call(self, hidden_states):
        # 定义层的前向传播逻辑
        hidden_states = self.input_embeddings(hidden_states, mode="linear")
        # 使用输入嵌入层处理隐藏状态，模式为 "linear"
        hidden_states = hidden_states + self.bias
        # 添加偏置到隐藏状态中
        return hidden_states
        # 返回处理后的隐藏状态


@keras_serializable
class TFXLNetMainLayer(keras.layers.Layer):
    # TFXLNetMainLayer 类定义，继承自 keras.layers.Layer

    config_class = XLNetConfig
    # 类变量 config_class，指定为 XLNetConfig 类

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 初始化方法，接受 config 参数
        self.config = config
        # 将 config 参数赋值给实例变量 self.config
        self.output_hidden_states = config.output_hidden_states
        # 设置输出隐藏状态的配置
        self.output_attentions = config.output_attentions
        # 设置输出注意力权重的配置
        self.return_dict = config.return_dict
        # 设置返回字典的配置

        self.mem_len = config.mem_len
        # 设置记忆长度的配置
        self.reuse_len = config.reuse_len
        # 设置重用长度的配置
        self.d_model = config.d_model
        # 设置模型维度的配置
        self.same_length = config.same_length
        # 设置是否长度相同的配置
        self.attn_type = config.attn_type
        # 设置注意力类型的配置
        self.bi_data = config.bi_data
        # 设置是否双向数据的配置
        self.clamp_len = config.clamp_len
        # 设置长度截断的配置
        self.n_layer = config.n_layer
        # 设置层数的配置
        self.use_bfloat16 = config.use_bfloat16
        # 设置是否使用 bfloat16 的配置
        self.initializer_range = config.initializer_range
        # 设置初始化范围的配置

        self.word_embedding = TFSharedEmbeddings(
            config.vocab_size, config.d_model, initializer_range=config.initializer_range, name="word_embedding"
        )
        # 创建 TFSharedEmbeddings 实例 word_embedding，共享嵌入
        self.layer = [TFXLNetLayer(config, name=f"layer_._{i}") for i in range(config.n_layer)]
        # 创建 TFXLNetLayer 实例的列表 layer，根据配置的层数
        self.dropout = keras.layers.Dropout(config.dropout)
        # 创建 Dropout 层，使用配置的 dropout 概率

        self.use_mems_eval = config.use_mems_eval
        # 设置评估时是否使用记忆的配置
        self.use_mems_train = config.use_mems_train
        # 设置训练时是否使用记忆的配置

    def get_input_embeddings(self):
        # 返回输入嵌入层 self.word_embedding
        return self.word_embedding

    def set_input_embeddings(self, value):
        # 设置输入嵌入层的值，并更新 vocab_size
        self.word_embedding.weight = value
        self.word_embedding.vocab_size = shape_list(value)[0]
    # 构建函数，用于初始化模型的权重和层结构
    def build(self, input_shape=None):
        # 获取指定初始化范围的初始化器
        initializer = get_initializer(self.initializer_range)
        # 添加名为 mask_emb 的可训练权重，形状为 (1, 1, self.d_model)
        self.mask_emb = self.add_weight(
            shape=(1, 1, self.d_model), initializer=initializer, trainable=True, name="mask_emb"
        )

        # 如果模型已经建立，则直接返回，避免重复构建
        if self.built:
            return
        self.built = True  # 设置模型为已建立状态

        # 如果存在 word_embedding 属性，则构建它
        if getattr(self, "word_embedding", None) is not None:
            with tf.name_scope(self.word_embedding.name):
                self.word_embedding.build(None)

        # 如果存在 layer 属性，则逐层构建每一层
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)

    # 剪枝注意力头的方法，抛出未实现错误
    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError

    # 创建自注意力掩码的方法，返回一个浮点数掩码，用于指示哪些位置需要被屏蔽
    def create_mask(self, qlen, mlen):
        """
        Creates causal attention mask. Float mask where 1.0 indicates masked, 0.0 indicates not-masked.

        Args:
            qlen: Query 的长度
            mlen: Memory 的长度
        """
        attn_mask = tf.ones([qlen, qlen])  # 创建全为 1 的注意力掩码矩阵
        mask_u = tf.linalg.band_part(attn_mask, 0, -1)  # 上三角矩阵
        mask_dia = tf.linalg.band_part(attn_mask, 0, 0)  # 对角线矩阵
        attn_mask_pad = tf.zeros([qlen, mlen])  # 创建全为 0 的填充掩码矩阵
        ret = tf.concat([attn_mask_pad, mask_u - mask_dia], 1)  # 拼接得到最终的掩码矩阵

        # 如果设置了 same_length 标志，则生成长度相同的掩码矩阵
        if self.same_length:
            mask_l = tf.linalg.band_part(attn_mask, -1, 0)  # 下三角矩阵
            ret = tf.concat([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], 1)  # 拼接得到长度相同的掩码矩阵

        return ret  # 返回生成的注意力掩码矩阵

    # 缓存当前输出到内存中的方法，用于在模型推理或训练时存储隐藏状态
    def cache_mem(self, curr_out, prev_mem):
        # 如果设置了 reuse_len 并且大于 0，则截取当前输出的前部分作为有效输出
        if self.reuse_len is not None and self.reuse_len > 0:
            curr_out = curr_out[: self.reuse_len]

        # 如果未定义 mem_len 或 mem_len 为 0，则设定截断点为 0
        if self.mem_len is None or self.mem_len == 0:
            cutoff = 0
        else:
            # 否则，根据 mem_len 设定截断点
            cutoff = -self.mem_len

        # 如果之前的记忆 prev_mem 为空，则直接使用当前输出的截断部分
        if prev_mem is None:
            new_mem = curr_out[cutoff:]
        else:
            # 否则，将当前输出与之前的记忆连接，并根据截断点进行截取
            new_mem = tf.concat([prev_mem, curr_out], 0)[cutoff:]

        return tf.stop_gradient(new_mem)  # 返回新的内存状态，并停止梯度传播
    def positional_embedding(pos_seq, inv_freq, bsz=None):
        # 使用 tf.einsum 计算正弦和余弦函数输入的乘积
        sinusoid_inp = tf.einsum("i,d->id", pos_seq, inv_freq)
        # 将正弦和余弦函数结果连接起来，形成位置编码
        pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], axis=-1)
        # 在第二维度增加一个维度，用于后续的扩展操作
        pos_emb = pos_emb[:, None, :]

        if bsz is not None:
            # 如果指定了 batch size，使用 tf.tile 扩展 pos_emb 的第二维度
            pos_emb = tf.tile(pos_emb, [1, bsz, 1])

        return pos_emb

    def relative_positional_encoding(self, qlen, klen, bsz=None):
        """create relative positional encoding."""
        # 创建频率序列，用于计算位置编码
        freq_seq = tf.range(0, self.d_model, 2.0)
        inv_freq = 1 / (10000 ** (freq_seq / self.d_model))

        if self.attn_type == "bi":
            # 如果是双向注意力，设置起始和结束位置
            beg, end = klen, -qlen
        elif self.attn_type == "uni":
            # 如果是单向注意力，设置起始和结束位置
            beg, end = klen, -1
        else:
            # 抛出异常，表示未知的注意力类型
            raise ValueError(f"Unknown `attn_type` {self.attn_type}.")

        if self.bi_data:
            # 如果使用双向数据，生成正向和反向的位置序列
            fwd_pos_seq = tf.range(beg, end, -1.0)
            bwd_pos_seq = tf.range(-beg, -end, 1.0)

            if self.clamp_len > 0:
                # 如果设置了 clamp_len，则对序列进行截断
                fwd_pos_seq = tf.clip_by_value(fwd_pos_seq, -self.clamp_len, self.clamp_len)
                bwd_pos_seq = tf.clip_by_value(bwd_pos_seq, -self.clamp_len, self.clamp_len)

            if bsz is not None:
                # 如果指定了 batch size，按照 batch size 的一半创建正向和反向的位置编码
                if bsz % 2 != 0:
                    raise ValueError(f"With bi_data, the batch size {bsz} should be divisible by 2")
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz // 2)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq, bsz // 2)
            else:
                # 否则创建不带 batch size 的正向和反向位置编码
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq)

            # 拼接正向和反向的位置编码
            pos_emb = tf.concat([fwd_pos_emb, bwd_pos_emb], axis=1)
        else:
            # 如果不使用双向数据，只生成正向的位置序列
            fwd_pos_seq = tf.range(beg, end, -1.0)
            if self.clamp_len > 0:
                # 如果设置了 clamp_len，则对序列进行截断
                fwd_pos_seq = tf.clip_by_value(fwd_pos_seq, -self.clamp_len, self.clamp_len)
            # 创建正向的位置编码
            pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz)

        return pos_emb

    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        mems: np.ndarray | tf.Tensor | None = None,
        perm_mask: np.ndarray | tf.Tensor | None = None,
        target_mapping: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        input_mask: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        use_mems: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
# TFXLNetPreTrainedModel 类，继承自 TFPreTrainedModel 类，用于处理权重初始化和预训练模型下载及加载的抽象类。
class TFXLNetPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 配置类变量，指定为 XLNetConfig 类型，用于配置模型的参数和结构
    config_class = XLNetConfig
    # 基础模型前缀，指定为 "transformer"，用于模型命名空间管理
    base_model_prefix = "transformer"


# dataclass 装饰器标记 TFXLNetModelOutput 类，定义了 TFXLNetModel 的输出类型
@dataclass
class TFXLNetModelOutput(ModelOutput):
    """
    Output type of [`TFXLNetModel`].

    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, num_predict, hidden_size)`):
            Sequence of hidden-states at the last layer of the model.

            `num_predict` corresponds to `target_mapping.shape[1]`. If `target_mapping` is `None`, then `num_predict`
            corresponds to `sequence_length`.
        mems (`List[tf.Tensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
            token ids which have their past given to this model should not be passed as `input_ids` as they have
            already been computed.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 最后一层模型的隐藏状态，形状为 `(batch_size, num_predict, hidden_size)` 的张量
    last_hidden_state: tf.Tensor = None
    # 预先计算的隐藏状态列表，长度为 `config.n_layers` 的张量列表
    mems: List[tf.Tensor] | None = None
    # 可选项，当 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回，包含每层模型输出的元组
    hidden_states: Tuple[tf.Tensor, ...] | None = None
    # 可选项，当 `output_attentions=True` 或 `config.output_attentions=True` 时返回，包含每层注意力权重的元组
    attentions: Tuple[tf.Tensor, ...] | None = None


@dataclass
class TFXLNetLMHeadModelOutput(ModelOutput):
    """
    Output type of [`TFXLNetLMHeadModel`].
    """

    # 此处未定义具体的输出结构或参数，但作为 TFXLNetLMHeadModel 的输出类型声明
    pass
    # 定义函数的参数列表，包含多个可选参数，用于语言建模任务
    Args:
        loss (`tf.Tensor` of shape *(1,)*, *optional*, returned when `labels` is provided):
            如果提供了 `labels`，则返回的语言建模损失（用于下一个标记预测）。
        logits (`tf.Tensor` of shape `(batch_size, num_predict, config.vocab_size)`):
            语言建模头部的预测分数（在应用 SoftMax 之前的每个词汇标记的分数）。
    
            `num_predict` 对应于 `target_mapping.shape[1]`。如果 `target_mapping` 是 `None`，则 `num_predict`
            对应于 `sequence_length`。
        mems (`List[tf.Tensor]` of length `config.n_layers`):
            包含预计算的隐藏状态。可以用于加速顺序解码。已经计算过其过去的令牌 id 不应该作为 `input_ids` 传递，
            因为它们已经被计算过。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            一个元组，包含 `tf.Tensor`（一个用于嵌入输出 + 每个层的输出）的形状为 `(batch_size, sequence_length, hidden_size)`。
    
            模型每个层的输出以及初始嵌入输出的隐藏状态。
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            一个元组，包含 `tf.Tensor`（每个层的一个）的形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
    
            注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
# 使用 `dataclass` 装饰器定义一个数据类，表示XLNet用于序列分类任务的输出。
@dataclass
class TFXLNetForSequenceClassificationOutput(ModelOutput):
    """
    [`TFXLNetForSequenceClassification`] 的输出类型。

    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, 当 `label` 被提供时返回):
            分类（如果 `config.num_labels==1` 则为回归）损失。
        logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`):
            分类（或回归，如果 `config.num_labels==1`）得分（SoftMax 之前）。
        mems (`List[tf.Tensor]` of length `config.n_layers`):
            包含预计算隐藏状态。可以用于加速序列解码。将已经计算过其过去的令牌 id 传递给该模型不应作为 `input_ids` 传递。
        hidden_states (`tuple(tf.Tensor)`, *optional*, 当 `output_hidden_states=True` 传递或者 `config.output_hidden_states=True` 时返回):
            形状为 `(batch_size, sequence_length, hidden_size)` 的 `tf.Tensor` 元组（一个用于嵌入输出，一个用于每一层的输出）。

            模型每一层输出的隐藏状态加上初始嵌入输出。
        attentions (`tuple(tf.Tensor)`, *optional*, 当 `output_attentions=True` 传递或者 `config.output_attentions=True` 时返回):
            形状为 `(batch_size, num_heads, sequence_length, sequence_length)` 的 `tf.Tensor` 元组（每层一个）。

            在注意力 softmax 后的注意力权重，用于在自注意力头中计算加权平均值。
    """

    # 损失值，类型为 `tf.Tensor` 或 `None`
    loss: tf.Tensor | None = None
    # 预测的 logits，类型为 `tf.Tensor` 或 `None`
    logits: tf.Tensor = None
    # 隐藏状态的记忆列表，类型为 `List[tf.Tensor]` 或 `None`
    mems: List[tf.Tensor] | None = None
    # 每一层的隐藏状态，类型为 `Tuple[tf.Tensor, ...]` 或 `None`
    hidden_states: Tuple[tf.Tensor, ...] | None = None
    # 注意力权重，类型为 `Tuple[tf.Tensor, ...]` 或 `None`
    attentions: Tuple[tf.Tensor, ...] | None = None


@dataclass
class TFXLNetForTokenClassificationOutput(ModelOutput):
    """
    [`TFXLNetForTokenClassificationOutput`] 的输出类型。
    """
    # loss: `tf.Tensor`类型，形状为`(1,)`，可选参数，当提供`labels`时返回。
    #       分类损失。
    loss: tf.Tensor | None = None
    
    # logits: `tf.Tensor`类型，形状为`(batch_size, sequence_length, config.num_labels)`。
    #         分类分数（SoftMax之前的）。
    logits: tf.Tensor = None
    
    # mems: 长度为`config.n_layers`的`List[tf.Tensor]`类型，可选参数。
    #       包含预先计算的隐藏状态。可以用于加速顺序解码。
    #       此模型已经计算了过去的令牌id，不应作为`input_ids`传递。
    mems: List[tf.Tensor] | None = None
    
    # hidden_states: 可选参数，当传递`output_hidden_states=True`或`config.output_hidden_states=True`时返回。
    #                是一个包含两个`tf.Tensor`的元组。
    #                第一个`tf.Tensor`为嵌入层的输出，第二个为每一层的输出。
    #                形状为`(batch_size, sequence_length, hidden_size)`。
    #                模型每一层的隐藏状态加上初始嵌入层的输出。
    hidden_states: Tuple[tf.Tensor, ...] | None = None
    
    # attentions: 可选参数，当传递`output_attentions=True`或`config.output_attentions=True`时返回。
    #             是一个包含多个`tf.Tensor`的元组，每个`tf.Tensor`对应一个层的注意力权重。
    #             形状为`(batch_size, num_heads, sequence_length, sequence_length)`。
    #             注意力softmax之后的注意力权重，用于在自注意力头中计算加权平均值。
    attentions: Tuple[tf.Tensor, ...] | None = None
# 定义一个数据类，用于存储 `TFXLNetForMultipleChoice` 模型的输出结果
@dataclass
class TFXLNetForMultipleChoiceOutput(ModelOutput):
    """
    Output type of [`TFXLNetForMultipleChoice`].

    Args:
        loss (`tf.Tensor` of shape *(1,)*, *optional*, returned when `labels` is provided):
            分类损失。
        logits (`tf.Tensor` of shape `(batch_size, num_choices)`):
            `num_choices` 是输入张量的第二维度。参见上文中的 `input_ids`。

            分类得分（SoftMax 之前）。
        mems (`List[tf.Tensor]` of length `config.n_layers`):
            包含预先计算的隐藏状态。可用于加速顺序解码。这个模型接收到的 token id 不应作为 `input_ids` 传递，因为它们已经被计算过。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            元组，包含 `tf.Tensor`（一个用于嵌入输出，每一层一个用于层的输出）的形状为 `(batch_size, sequence_length, hidden_size)`。

            每层模型的隐藏状态，以及初始嵌入的输出。
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            元组，包含每一层的 `tf.Tensor` 的形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    loss: tf.Tensor | None = None
    logits: tf.Tensor = None
    mems: List[tf.Tensor] | None = None
    hidden_states: Tuple[tf.Tensor, ...] | None = None
    attentions: Tuple[tf.Tensor, ...] | None = None


# 定义一个数据类，用于存储 `TFXLNetForQuestionAnsweringSimple` 模型的输出结果
@dataclass
class TFXLNetForQuestionAnsweringSimpleOutput(ModelOutput):
    """
    Output type of [`TFXLNetForQuestionAnsweringSimple`].
    """
    # 定义函数参数和返回值的注释文档字符串，描述了函数可能接收的参数和返回的值的类型和含义
    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            总的跨度抽取损失，由开始和结束位置的交叉熵之和组成。
        start_logits (`tf.Tensor` of shape `(batch_size, sequence_length,)`):
            跨度开始位置的分数（SoftMax 之前）。
        end_logits (`tf.Tensor` of shape `(batch_size, sequence_length,)`):
            跨度结束位置的分数（SoftMax 之前）。
        mems (`List[tf.Tensor]` of length `config.n_layers`):
            包含预先计算的隐藏状态的列表。可以用于加速序列解码。
            给定到该模型的过去标记 id 不应作为 `input_ids` 传递，因为它们已经计算过。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            包含模型每一层输出的隐藏状态的元组。
            第一个张量是嵌入层的输出，后续的张量是每一层输出的隐藏状态。
            形状为 `(batch_size, sequence_length, hidden_size)`。
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            包含自注意力机制 softmax 后的注意力权重的元组。
            用于计算自注意力头部的加权平均值。
            形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

    """
    
    # 初始化变量，表示损失、开始位置分数、结束位置分数、预先计算的隐藏状态、每层的隐藏状态、每层的注意力权重
    loss: tf.Tensor | None = None
    start_logits: tf.Tensor = None
    end_logits: tf.Tensor = None
    mems: List[tf.Tensor] | None = None
    hidden_states: Tuple[tf.Tensor, ...] | None = None
    attentions: Tuple[tf.Tensor, ...] | None = None
"""
    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
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

    Parameters:
        config ([`XLNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 用于存储 XLNet 模型输入说明文档的常量字符串
XLNET_INPUTS_DOCSTRING = r"""
"""


# 使用装饰器 `add_start_docstrings` 给 `TFXLNetModel` 类添加描述文档
@add_start_docstrings(
    "The bare XLNet Model transformer outputting raw hidden-states without any specific head on top.",
    XLNET_START_DOCSTRING,  # 引用之前定义的模型文档字符串
)
# 定义 TFXLNetModel 类，继承自 TFXLNetPreTrainedModel
class TFXLNetModel(TFXLNetPreTrainedModel):
    # 初始化方法，接收一个 config 对象和任意数量的位置参数和关键字参数
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 创建一个名为 transformer 的 TFXLNetMainLayer 实例
        self.transformer = TFXLNetMainLayer(config, name="transformer")

    # 使用装饰器 `unpack_inputs` 给该方法添加描述文档
    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFXLNetModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 添加代码示例的文档字符串，指定检查点、输出类型和配置类
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 输入的 token IDs，可以为空
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力掩码，可以为空，可以是 NumPy 数组或 TensorFlow 张量
        mems: np.ndarray | tf.Tensor | None = None,  # 循环记忆，可以为空，可以是 NumPy 数组或 TensorFlow 张量
        perm_mask: np.ndarray | tf.Tensor | None = None,  # 排列掩码，可以为空，可以是 NumPy 数组或 TensorFlow 张量
        target_mapping: np.ndarray | tf.Tensor | None = None,  # 目标映射，可以为空，可以是 NumPy 数组或 TensorFlow 张量
        token_type_ids: np.ndarray | tf.Tensor | None = None,  # token 类型 IDs，可以为空，可以是 NumPy 数组或 TensorFlow 张量
        input_mask: np.ndarray | tf.Tensor | None = None,  # 输入掩码，可以为空，可以是 NumPy 数组或 TensorFlow 张量
        head_mask: np.ndarray | tf.Tensor | None = None,  # 头部掩码，可以为空，可以是 NumPy 数组或 TensorFlow 张量
        inputs_embeds: np.ndarray | tf.Tensor | None = None,  # 嵌入输入，可以为空，可以是 NumPy 数组或 TensorFlow 张量
        use_mems: Optional[bool] = None,  # 是否使用循环记忆，可选布尔类型
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选布尔类型
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选布尔类型
        return_dict: Optional[bool] = None,  # 是否返回字典形式的输出，可选布尔类型
        training: bool = False,  # 是否在训练模式下，布尔类型，默认为 False
    ) -> Union[TFXLNetModelOutput, Tuple[tf.Tensor]]:
        # 调用 Transformer 模型的前向传播
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_mems=use_mems,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        return outputs

    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 标记为已经构建
        self.built = True
        # 如果存在 Transformer 模型，则在命名空间下构建
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
@add_start_docstrings(
    """
    XLNet Model with a language modeling head on top (linear layer with weights tied to the input embeddings).
    """,
    XLNET_START_DOCSTRING,
)
class TFXLNetLMHeadModel(TFXLNetPreTrainedModel, TFCausalLanguageModelingLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 初始化 XLNet 主体层，使用给定的配置参数
        self.transformer = TFXLNetMainLayer(config, name="transformer")
        # 初始化语言建模头部，与词嵌入权重相连
        self.lm_loss = TFXLNetLMHead(config, self.transformer.word_embedding, name="lm_loss")
        # 不支持 XLA 生成
        self.supports_xla_generation = False

    def get_lm_head(self):
        # 返回语言建模头部对象
        return self.lm_loss

    def get_prefix_bias_name(self):
        # 警告，方法已弃用，请使用 `get_bias` 替代
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        # 返回头部名称和语言建模头部名称的组合
        return self.name + "/" + self.lm_loss.name

    def prepare_inputs_for_generation(self, inputs, past_key_values=None, use_mems=None, **kwargs):
        # 在输入序列末尾添加虚拟标记（不被注意力机制使用）
        effective_batch_size = inputs.shape[0]
        dummy_token = tf.zeros((effective_batch_size, 1), dtype=inputs.dtype)

        # 计算新标记和最后两个生成标记的注意力值，其余从 `past` 缓存重新加载。
        # 完全自回归模型应该有 offset = 1；offset = 2 似乎计算稍好。
        offset = 2

        if past_key_values:
            # 如果过去键值存在，则在末尾添加虚拟标记
            input_ids = tf.concat([inputs[:, -offset:], dummy_token], axis=1)
        else:
            # 否则，在末尾添加虚拟标记
            input_ids = tf.concat([inputs, dummy_token], axis=1)

        # 构建排列掩码，使之前的标记不看到最后一个标记
        sequence_length = input_ids.shape[1]
        perm_mask = tf.zeros((effective_batch_size, sequence_length, sequence_length - 1))
        perm_mask_seq_end = tf.ones((effective_batch_size, sequence_length, 1))
        perm_mask = tf.concat([perm_mask, perm_mask_seq_end], axis=-1)

        # 仅预测最后一个标记
        target_mapping = tf.zeros((effective_batch_size, 1, sequence_length - 1))
        target_mapping_seq_end = tf.ones((effective_batch_size, 1, 1))
        target_mapping = tf.concat([target_mapping, target_mapping_seq_end], axis=-1)

        inputs = {
            "input_ids": input_ids,
            "perm_mask": perm_mask,
            "target_mapping": target_mapping,
            "use_mems": use_mems,
        }

        # 如果模型参数中定义了过去键值，则用于更快速的解码
        if past_key_values:
            inputs["mems"] = tuple(layer_past[:-offset, :, :] for layer_past in past_key_values)

        return inputs

    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFXLNetLMHeadModelOutput, config_class=_CONFIG_FOR_DOC)
    # 定义一个方法用于调用模型，接受多个输入参数，每个参数都有指定的类型或者可以为空
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 输入的模型的输入 IDs，可以是指定的类型或者空
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力掩码，可以是 NumPy 数组、张量或者空
        mems: np.ndarray | tf.Tensor | None = None,  # 记忆项，可以是 NumPy 数组、张量或者空
        perm_mask: np.ndarray | tf.Tensor | None = None,  # 排列掩码，可以是 NumPy 数组、张量或者空
        target_mapping: np.ndarray | tf.Tensor | None = None,  # 目标映射，可以是 NumPy 数组、张量或者空
        token_type_ids: np.ndarray | tf.Tensor | None = None,  # 标记类型 IDs，可以是 NumPy 数组、张量或者空
        input_mask: np.ndarray | tf.Tensor | None = None,  # 输入掩码，可以是 NumPy 数组、张量或者空
        head_mask: np.ndarray | tf.Tensor | None = None,  # 头部掩码，可以是 NumPy 数组、张量或者空
        inputs_embeds: np.ndarray | tf.Tensor | None = None,  # 输入的嵌入表示，可以是 NumPy 数组、张量或者空
        use_mems: Optional[bool] = None,  # 是否使用记忆项，可选布尔值，默认为 None
        output_attentions: Optional[bool] = None,  # 是否输出注意力，可选布尔值，默认为 None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选布尔值，默认为 None
        return_dict: Optional[bool] = None,  # 是否返回字典形式的输出，可选布尔值，默认为 None
        labels: np.ndarray | tf.Tensor | None = None,  # 标签，可以是 NumPy 数组、张量或者空
        training: bool = False,  # 是否处于训练模式，默认为 False
    ):
        # 定义模型调用方法，实现模型的前向传播等功能
        pass  # 此处实际上只是定义了方法的结构，具体的实现需要在此基础上完成

    # 构建方法，用于构建模型的结构
    def build(self, input_shape=None):
        # 如果模型已经构建完毕，则直接返回
        if self.built:
            return
        # 设置模型为已构建状态
        self.built = True
        # 如果存在 transformer 属性，则在其命名空间内构建 transformer
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        # 如果存在 lm_loss 属性，则在其命名空间内构建 lm_loss
        if getattr(self, "lm_loss", None) is not None:
            with tf.name_scope(self.lm_loss.name):
                self.lm_loss.build(None)
# 使用装饰器添加模型文档字符串，描述 XLNet 模型用于序列分类/回归任务的顶层结构
@add_start_docstrings(
    """
    XLNet Model with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g.
    for GLUE tasks.
    """,
    XLNET_START_DOCSTRING,
)
# 定义 TFXLNetForSequenceClassification 类，继承自 TFXLNetPreTrainedModel 和 TFSequenceClassificationLoss
class TFXLNetForSequenceClassification(TFXLNetPreTrainedModel, TFSequenceClassificationLoss):
    
    # 初始化方法
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 设置类别数目
        self.num_labels = config.num_labels
        
        # 创建 XLNet 主层，命名为 'transformer'
        self.transformer = TFXLNetMainLayer(config, name="transformer")
        
        # 创建序列摘要层，用于生成序列摘要，初始化范围为 config.initializer_range，命名为 'sequence_summary'
        self.sequence_summary = TFSequenceSummary(
            config, initializer_range=config.initializer_range, name="sequence_summary"
        )
        
        # 创建输出层，用于生成 logits，层的输出尺寸为 config.num_labels，权重初始化方法为 config.initializer_range 中的 initializer，命名为 'logits_proj'
        self.logits_proj = keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="logits_proj"
        )
        
        # 保存模型配置信息
        self.config = config
    
    # 使用装饰器添加模型前向传播的文档字符串，描述输入参数的含义和形状要求
    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFXLNetForSequenceClassificationOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义模型的前向传播方法
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        mems: np.ndarray | tf.Tensor | None = None,
        perm_mask: np.ndarray | tf.Tensor | None = None,
        target_mapping: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        input_mask: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        use_mems: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: bool = False,
        # 省略部分输入参数说明
    ) -> Union[TFXLNetForSequenceClassificationOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 调用transformer模型进行前向传播，返回输出结果
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_mems=use_mems,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 从transformer模型的输出中取第一个元素作为模型输出
        output = transformer_outputs[0]

        # 对模型输出进行序列摘要（summary）
        output = self.sequence_summary(output)
        # 将摘要后的结果投影到logits空间
        logits = self.logits_proj(output)

        # 如果labels不为空，计算损失值；否则损失为None
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果return_dict为False，则返回元组形式的输出
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果return_dict为True，则返回TFXLNetForSequenceClassificationOutput对象
        return TFXLNetForSequenceClassificationOutput(
            loss=loss,
            logits=logits,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果模型已经构建，直接返回
        if self.built:
            return
        # 标记模型为已构建状态
        self.built = True
        # 如果self.transformer存在，则在其命名作用域下构建transformer模型
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        # 如果self.sequence_summary存在，则在其命名作用域下构建序列摘要模型
        if getattr(self, "sequence_summary", None) is not None:
            with tf.name_scope(self.sequence_summary.name):
                self.sequence_summary.build(None)
        # 如果self.logits_proj存在，则在其命名作用域下构建logits投影模型
        if getattr(self, "logits_proj", None) is not None:
            with tf.name_scope(self.logits_proj.name):
                self.logits_proj.build([None, None, self.config.d_model])
# 使用装饰器添加文档字符串，描述了这个类的作用是在XLNET模型基础上添加一个多选分类的头部，例如用于RocStories/SWAG任务。
@add_start_docstrings(
    """
    XLNET Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    XLNET_START_DOCSTRING,  # 引用XLNET模型的起始文档字符串
)
class TFXLNetForMultipleChoice(TFXLNetPreTrainedModel, TFMultipleChoiceLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 初始化XLNET主层，命名为'transformer'
        self.transformer = TFXLNetMainLayer(config, name="transformer")
        # 初始化序列摘要层，用于生成序列汇总
        self.sequence_summary = TFSequenceSummary(
            config, initializer_range=config.initializer_range, name="sequence_summary"
        )
        # 初始化逻辑回归投影层，用于多选分类，输出维度为1
        self.logits_proj = keras.layers.Dense(
            1, kernel_initializer=get_initializer(config.initializer_range), name="logits_proj"
        )
        # 保存模型配置
        self.config = config

    # 使用装饰器添加文档字符串，描述这个方法的作用是处理XLNET模型的前向传播，支持多种输入
    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,  # 引用文档中的检查点示例
        output_type=TFXLNetForMultipleChoiceOutput,  # 引用XLNET多选输出的类型
        config_class=_CONFIG_FOR_DOC,  # 引用文档中的配置类示例
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        input_mask: np.ndarray | tf.Tensor | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        mems: np.ndarray | tf.Tensor | None = None,
        perm_mask: np.ndarray | tf.Tensor | None = None,
        target_mapping: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        use_mems: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: bool = False,
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)
        """
        # 如果传入了 input_ids，则获取其第二维度的大小作为 num_choices
        if input_ids is not None:
            num_choices = shape_list(input_ids)[1]
            # 获取 input_ids 的第三维度大小作为 seq_length
            seq_length = shape_list(input_ids)[2]
        else:
            # 否则，从 inputs_embeds 中获取第二维度大小作为 num_choices
            num_choices = shape_list(inputs_embeds)[1]
            # 获取 inputs_embeds 的第三维度大小作为 seq_length
            seq_length = shape_list(inputs_embeds)[2]

        # 根据是否为 None，将各输入张量展平成二维张量
        flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
        flat_attention_mask = tf.reshape(attention_mask, (-1, seq_length)) if attention_mask is not None else None
        flat_token_type_ids = tf.reshape(token_type_ids, (-1, seq_length)) if token_type_ids is not None else None
        flat_input_mask = tf.reshape(input_mask, (-1, seq_length)) if input_mask is not None else None
        flat_inputs_embeds = (
            tf.reshape(inputs_embeds, (-1, seq_length, shape_list(inputs_embeds)[3]))
            if inputs_embeds is not None
            else None
        )

        # 调用 Transformer 模型，传入展平后的输入张量和其他参数
        transformer_outputs = self.transformer(
            flat_input_ids,
            flat_attention_mask,
            mems,
            perm_mask,
            target_mapping,
            flat_token_type_ids,
            flat_input_mask,
            head_mask,
            flat_inputs_embeds,
            use_mems,
            output_attentions,
            output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 从 Transformer 输出中取出第一个元素作为输出
        output = transformer_outputs[0]

        # 对输出进行序列摘要
        logits = self.sequence_summary(output)

        # 对序列摘要后的结果进行投影，得到最终的 logits
        logits = self.logits_proj(logits)

        # 将 logits 重新 reshape 成二维张量
        reshaped_logits = tf.reshape(logits, (-1, num_choices))

        # 如果提供了 labels，则计算损失，否则损失设为 None
        loss = None if labels is None else self.hf_compute_loss(labels, reshaped_logits)

        # 如果 return_dict 为 False，则返回扁平化后的 logits 和可能的其他输出
        if not return_dict:
            output = (reshaped_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则返回一个 TFXLNetForMultipleChoiceOutput 对象
        return TFXLNetForMultipleChoiceOutput(
            loss=loss,
            logits=reshaped_logits,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
    # 构建模型的方法，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在变换器（transformer），则在其命名空间内构建变换器
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        # 如果存在序列摘要（sequence_summary），则在其命名空间内构建序列摘要
        if getattr(self, "sequence_summary", None) is not None:
            with tf.name_scope(self.sequence_summary.name):
                self.sequence_summary.build(None)
        # 如果存在 logits 项目（logits_proj），则在其命名空间内构建 logits 项目
        if getattr(self, "logits_proj", None) is not None:
            with tf.name_scope(self.logits_proj.name):
                # 构建 logits 项目，输入形状为 [None, None, self.config.d_model]
                self.logits_proj.build([None, None, self.config.d_model])
"""
XLNet Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
Named-Entity-Recognition (NER) tasks.
"""
# 继承自TFXLNetPreTrainedModel和TFTokenClassificationLoss的XLNet模型，用于标记分类任务（如命名实体识别NER）。

class TFXLNetForTokenClassification(TFXLNetPreTrainedModel, TFTokenClassificationLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 设置分类的标签数目
        self.num_labels = config.num_labels

        # 初始化XLNet的主要层，命名为'transformer'
        self.transformer = TFXLNetMainLayer(config, name="transformer")
        
        # 设置分类器，为一个全连接层，输出大小为config.num_labels，使用config中定义的初始化范围初始化权重
        self.classifier = keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        
        # 保存配置参数
        self.config = config

    # 将多个输入解包并传递给模型的前向传播函数，同时添加了额外的文档字符串说明
    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFXLNetForTokenClassificationOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        mems: np.ndarray | tf.Tensor | None = None,
        perm_mask: np.ndarray | tf.Tensor | None = None,
        target_mapping: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        input_mask: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        use_mems: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: bool = False,
        """
        进行模型的前向传播，接受多种输入参数，包括input_ids、attention_mask等，以及用于训练的labels和是否处于训练模式的training标志位。
        """
    ) -> Union`
    ) -> Union[TFXLNetForTokenClassificationOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """

        # 使用 transformer 处理输入数据，返回 transformer 的输出
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_mems=use_mems,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 获取 transformer 的输出中的第一个元素，即 logits
        output = transformer_outputs[0]
        # 使用 classifier 处理 logits，得到最终的分类结果
        logits = self.classifier(output)
        # 如果提供了 labels，则计算损失；否则损失为 None
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果 return_dict 为 False，则返回一个包含 logits 和其他输出的元组
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则构建 TFXLNetForTokenClassificationOutput 对象并返回
        return TFXLNetForTokenClassificationOutput(
            loss=loss,
            logits=logits,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 设置标记为已构建
        self.built = True

        # 如果存在 transformer 属性，则构建 transformer
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)

        # 如果存在 classifier 属性，则构建 classifier
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
# 使用自定义的文档字符串描述这个类，说明它是在XLNet模型基础上构建的用于抽取式问答任务的模型，
# 通过在隐藏状态输出的基础上添加线性层来计算'起始位置logits'和'结束位置logits'。
@add_start_docstrings(
    """
    XLNet Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    XLNET_START_DOCSTRING,
)
class TFXLNetForQuestionAnsweringSimple(TFXLNetPreTrainedModel, TFQuestionAnsweringLoss):
    
    # 初始化方法，接收配置和其他输入参数
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 创建XLNet的主要层，并命名为"transformer"
        self.transformer = TFXLNetMainLayer(config, name="transformer")
        # 创建一个全连接层用于问答输出，其输出维度为config.num_labels，初始化方式为指定范围内的初始化器
        self.qa_outputs = keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )
        # 将配置保存为对象的属性
        self.config = config

    # 调用方法，实现模型的前向传播
    @unpack_inputs
    # 添加模型前向传播的文档字符串，描述输入参数的形状和含义
    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 添加示例代码的文档字符串，描述模型的检查点、输出类型和配置类
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFXLNetForQuestionAnsweringSimpleOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        mems: np.ndarray | tf.Tensor | None = None,
        perm_mask: np.ndarray | tf.Tensor | None = None,
        target_mapping: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        input_mask: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        use_mems: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        start_positions: np.ndarray | tf.Tensor | None = None,
        end_positions: np.ndarray | tf.Tensor | None = None,
        training: bool = False,
    ) -> Union[TFXLNetForQuestionAnsweringSimpleOutput, Tuple[tf.Tensor]]:
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
        # 调用 Transformer 模型进行前向传播，获取输出
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_mems=use_mems,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 从 Transformer 输出中提取序列输出
        sequence_output = transformer_outputs[0]

        # 将序列输出传递给 QA 输出层，得到起始位置和结束位置的预测 logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        # 计算损失，如果提供了起始位置和结束位置的标签
        loss = None
        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            loss = self.hf_compute_loss(labels, (start_logits, end_logits))

        # 如果不需要返回字典，则构建输出元组
        if not return_dict:
            output = (start_logits, end_logits) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典，则构建 TFXLNetForQuestionAnsweringSimpleOutput 对象
        return TFXLNetForQuestionAnsweringSimpleOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经定义了 Transformer 模型，则构建它
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        # 如果已经定义了 QA 输出层，则构建它
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])
```