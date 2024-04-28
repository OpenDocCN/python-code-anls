# `.\transformers\models\xlnet\modeling_tf_xlnet.py`

```
# 设置文件编码为 utf-8
# 版权声明和许可协议信息
# 导入必要的库和模块

"""
TF 2.0 XLNet 模型。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

# 导入相关函数和类
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
# 导入 XLNet 配置类
from .configuration_xlnet import XLNetConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 以下是一些常量和列表
_CHECKPOINT_FOR_DOC = "xlnet-base-cased"
_CONFIG_FOR_DOC = "XLNetConfig"

# 预训练模型列表
TF_XLNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "xlnet-base-cased",
    "xlnet-large-cased",
    # 查看所有 XLNet 模型：https://huggingface.co/models?filter=xlnet
]

class TFXLNetRelativeAttention(tf.keras.layers.Layer):
    # 定义 XLNet 相对注意力机制类
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        
        # 如果隐藏单元数不能被注意力头数整除，抛出异常
        if config.d_model % config.n_head != 0:
            raise ValueError(
                f"The hidden size ({config.d_model}) is not a multiple of the number of attention "
                f"heads ({config.n_head}"
            )
        
        # 初始化相对注意力机制相关参数
        self.n_head = config.n_head
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.scale = 1 / (config.d_head**0.5)
        self.initializer_range = config.initializer_range
        self.output_attentions = config.output_attentions

        # 定义层标准化和丢弃层
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.config = config
    # 构建 self 对象，用于初始化，输入形状为 None 或者指定形状
    def build(self, input_shape=None):
        # 初始化权重矩阵 q，形状为 (d_model, n_head, d_head)，使用指定范围的初始化器
        initializer = get_initializer(self.initializer_range)
        self.q = self.add_weight(
            shape=(self.d_model, self.n_head, self.d_head), initializer=initializer, trainable=True, name="q"
        )
        # 初始化权重矩阵 k，形状同上
        self.k = self.add_weight(
            shape=(self.d_model, self.n_head, self.d_head), initializer=initializer, trainable=True, name="k"
        )
        # 初始化权重矩阵 v，形状同上
        self.v = self.add_weight(
            shape=(self.d_model, self.n_head, self.d_head), initializer=initializer, trainable=True, name="v"
        )
        # 初始化权重矩阵 o，形状同上
        self.o = self.add_weight(
            shape=(self.d_model, self.n_head, self.d_head), initializer=initializer, trainable=True, name="o"
        )
        # 初始化权重矩阵 r，形状同上
        self.r = self.add_weight(
            shape=(self.d_model, self.n_head, self.d_head), initializer=initializer, trainable=True, name="r"
        )
        # 初始化 r_r_bias，形状为 (n_head, d_head)，使用零初始化器
        self.r_r_bias = self.add_weight(
            shape=(self.n_head, self.d_head), initializer="zeros", trainable=True, name="r_r_bias"
        )
        # 初始化 r_s_bias，形状同上
        self.r_s_bias = self.add_weight(
            shape=(self.n_head, self.d_head), initializer="zeros", trainable=True, name="r_s_bias"
        )
        # 初始化 r_w_bias，形状同上
        self.r_w_bias = self.add_weight(
            shape=(self.n_head, self.d_head), initializer="zeros", trainable=True, name="r_w_bias"
        )
        # 初始化 seg_embed，形状为 (2, n_head, d_head)，使用指定范围的初始化器
        self.seg_embed = self.add_weight(
            shape=(2, self.n_head, self.d_head), initializer=initializer, trainable=True, name="seg_embed"
        )

        # 如果已经构建过网络，则直接返回
        if self.built:
            return
        # 标记网络已经构建
        self.built = True
        # 如果存在 layer_norm 属性，执行以下代码
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                # 构建 layer_norm 层，输入形状为 [None, None, self.config.d_model]
                self.layer_norm.build([None, None, self.config.d_model])

    # 剪枝指定 attention 头
    def prune_heads(self, heads):
        # 抛出未实现异常
        raise NotImplementedError

    # 执行相对位移，用于形成相对注意力分数
    def rel_shift(self, x, klen=-1):
        """perform relative shift to form the relative attention score."""
        # 获取 x 的大小
        x_size = shape_list(x)

        # 将 x 转换成 (x_size[1], x_size[0], x_size[2], x_size[3]) 的形状
        x = tf.reshape(x, (x_size[1], x_size[0], x_size[2], x_size[3]))
        # 去除第一行，得到 x 的新形状
        x = x[1:, ...]
        # 将 x 转换成 (x_size[0], x_size[1] - 1, x_size[2], x_size[3]) 的形状
        x = tf.reshape(x, (x_size[0], x_size[1] - 1, x_size[2], x_size[3]))
        # 对 x 进行切片，保留前 klen 列的数据
        x = x[:, 0:klen, :, :]

        return x

    # 执行相对注意力核心操作
    def rel_attn_core(
        self, q_head, k_head_h, v_head_h, k_head_r, seg_mat, attn_mask, head_mask, output_attentions, training=False
        """Core relative positional attention operations."""
        # 定义核心的相对位置注意力操作

        # content based attention score
        ac = tf.einsum("ibnd,jbnd->ijbn", q_head + self.r_w_bias, k_head_h)
        # 计算基于内容的注意力分数

        # position based attention score
        bd = tf.einsum("ibnd,jbnd->ijbn", q_head + self.r_r_bias, k_head_r)
        bd = self.rel_shift(bd, klen=shape_list(ac)[1])
        # 计算基于位置的注意力分数，并进行位置偏移

        # segment based attention score
        if seg_mat is None:
            ef = 0
        else:
            ef = tf.einsum("ibnd,snd->ibns", q_head + self.r_s_bias, self.seg_embed)
            ef = tf.einsum("ijbs,ibns->ijbn", seg_mat, ef)
        # 计算基于段的注意力分数，若无则为0

        # merge attention scores and perform masking
        attn_score = (ac + bd + ef) * self.scale
        if attn_mask is not None:
            if attn_mask.dtype == tf.float16 or attn_mask.dtype == tf.bfloat16:
                attn_score = attn_score - 65500 * attn_mask
            else:
                attn_score = attn_score - 1e30 * attn_mask
        # 合并注意力分数并执行掩码处理

        # attention probability
        attn_prob = stable_softmax(attn_score, axis=1)
        attn_prob = self.dropout(attn_prob, training=training)
        # 注意力概率计算及dropout操作

        # Mask heads if we want to
        if head_mask is not None:
            attn_prob = attn_prob * head_mask
        # 如果需要，则对注意力头进行掩码处理

        # attention output
        attn_vec = tf.einsum("ijbn,jbnd->ibnd", attn_prob, v_head_h)
        # 注意力输出计算

        if output_attentions:
            return attn_vec, attn_prob
        # 若需要输出注意力，即返回注意力向量和注意力概率

        return attn_vec


    def post_attention(self, h, attn_vec, residual=True, training=False):
        """Post-attention processing."""
         # 后注意处理部分

        # post-attention projection (back to `d_model`)
        attn_out = tf.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        # 后注意处理投影（返回到 'd_model'）

        attn_out = self.dropout(attn_out, training=training)
        # dropout操作

        if residual:
            attn_out = attn_out + h
        output = self.layer_norm(attn_out)
        # 如果要保留残差连接，则与输入加和，然后进行layer normalization

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
        ):
        # 定义call方法的输入参数和默认值
class TFXLNetFeedForward(tf.keras.layers.Layer):
    # 初始化方法，用于创建层对象
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # LayerNormalization层，用于规范化输入数据
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        # 第一个全连接层
        self.layer_1 = tf.keras.layers.Dense(
            config.d_inner, kernel_initializer=get_initializer(config.initializer_range), name="layer_1"
        )
        # 第二个全连接层
        self.layer_2 = tf.keras.layers.Dense(
            config.d_model, kernel_initializer=get_initializer(config.initializer_range), name="layer_2"
        )
        # Dropout层，用于防止过拟合
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        # 激活函数
        if isinstance(config.ff_activation, str):
            self.activation_function = get_tf_activation(config.ff_activation)
        else:
            self.activation_function = config.ff_activation
        # 配置信息
        self.config = config

    # 调用方法，定义层的前向传播逻辑
    def call(self, inp, training=False):
        # 将输入赋给输出变量
        output = inp
        # 第一个全连接层
        output = self.layer_1(output)
        # 激活函数
        output = self.activation_function(output)
        # Dropout层
        output = self.dropout(output, training=training)
        # 第二个全连接层
        output = self.layer_2(output)
        # Dropout层
        output = self.dropout(output, training=training)
        # LayerNormalization层，将原始输入和输出相加并规范化
        output = self.layer_norm(output + inp)
        return output

    # 构建方法，用于构建层的权重
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
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


class TFXLNetLayer(tf.keras.layers.Layer):
    # 初始化方法，用于创建层对象
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 相对注意力层
        self.rel_attn = TFXLNetRelativeAttention(config, name="rel_attn")
        # 前馈网络层
        self.ff = TFXLNetFeedForward(config, name="ff")
        # Dropout层，用于防止过拟合
        self.dropout = tf.keras.layers.Dropout(config.dropout)

    # 调用方法，定义层的前向传播逻辑
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
    # 定义类的一个方法，用于计算相对注意力
    def __call__(
        self,
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
    ):
        # 调用相对注意力方法计算输出
        outputs = self.rel_attn(
            output_h,  # 输出 query
            output_g,  # 输出 key
            non_tgt_mask,  # 掩码
            attn_mask,  # 注意力掩码
            pos_emb,  # 位置编码
            seg_mat,  # 分段信息
            mems,  # 记忆
            target_mapping,  # 目标映射
            head_mask,  # 注意力头部掩码
            output_attentions,  # 是否输出注意力
            training=training,  # 是否训练
        )
        # 从输出中获取输出 query 和输出 key
        output_h, output_g = outputs[:2]

        if output_g is not None:
            # 如果输出 key 不为空，则调用前馈神经网络计算输出 key
            output_g = self.ff(output_g, training=training)
        # 调用前馈神经网络计算输出 query
        output_h = self.ff(output_h, training=training)

        # 将再次计算的注意力添加到输出中（如果存在的话）
        outputs = (output_h, output_g) + outputs[2:]  # Add again attentions if there are there
        # 返回输出
        return outputs

    # 定义类的一个方法，用于构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建过了，直接返回
        if self.built:
            return
        # 设置模型为已构建状态
        self.built = True
        # 如果相对注意力方法存在，则进行构建
        if getattr(self, "rel_attn", None) is not None:
            with tf.name_scope(self.rel_attn.name):
                self.rel_attn.build(None)
        # 如果前馈神经网络方法存在，则进行构建
        if getattr(self, "ff", None) is not None:
            with tf.name_scope(self.ff.name):
                self.ff.build(None)
class TFXLNetLMHead(tf.keras.layers.Layer):
    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.input_embeddings = input_embeddings
    # 在build方法中创建偏置参数
    def build(self, input_shape):
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")
        super().build(input_shape)
    # 返回输入嵌入层
    def get_output_embeddings(self):
        return self.input_embeddings
    # 设置输出嵌入层
    def set_output_embeddings(self, value):
        self.input_embeddings.weight = value
        self.input_embeddings.vocab_size = shape_list(value)[0]
    # 返回偏置参数
    def get_bias(self):
        return {"bias": self.bias}
    # 设置偏置参数
    def set_bias(self, value):
        self.bias = value["bias"]
        self.config.vocab_size = shape_list(value["bias"])[0]
    # 对隐藏状态进行处理，使用input_embeddings进行线性处理后加上偏置参数
    def call(self, hidden_states):
        hidden_states = self.input_embeddings(hidden_states, mode="linear")
        hidden_states = hidden_states + self.bias
        return hidden_states


@keras_serializable
class TFXLNetMainLayer(tf.keras.layers.Layer):
    config_class = XLNetConfig

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 初始化各种配置参数
        self.config = config
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.return_dict = config.return_dict
        self.mem_len = config.mem_len
        self.reuse_len = config.reuse_len
        self.d_model = config.d_model
        self.same_length = config.same_length
        self.attn_type = config.attn_type
        self.bi_data = config.bi_data
        self.clamp_len = config.clamp_len
        self.n_layer = config.n_layer
        self.use_bfloat16 = config.use_bfloat16
        self.initializer_range = config.initializer_range
        # 创建共享的词嵌入层
        self.word_embedding = TFSharedEmbeddings(
            config.vocab_size, config.d_model, initializer_range=config.initializer_range, name="word_embedding"
        )
        # 创建XLNetLayer列表
        self.layer = [TFXLNetLayer(config, name=f"layer_._{i}") for i in range(config.n_layer)]
        # 创建dropout层
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.use_mems_eval = config.use_mems_eval
        self.use_mems_train = config.use_mems_train
    # 返回输入嵌入层
    def get_input_embeddings(self):
        return self.word_embedding
    # 设置输入嵌入层
    def set_input_embeddings(self, value):
        self.word_embedding.weight = value
        self.word_embedding.vocab_size = shape_list(value)[0]
    def build(self, input_shape=None):
        # 获取初始化器
        initializer = get_initializer(self.initializer_range)
        # 创建一个可训练的权重变量，用于mask（掩码）
        self.mask_emb = self.add_weight(
            shape=(1, 1, self.d_model), initializer=initializer, trainable=True, name="mask_emb"
        )
        # 如果已经构建过了，则直接返回
        if self.built:
            return
        self.built = True
        # 构建词嵌入
        if getattr(self, "word_embedding", None) is not None:
            with tf.name_scope(self.word_embedding.name):
                self.word_embedding.build(None)
        # 构建图层
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)

    def _prune_heads(self, heads_to_prune):
        # 剪枝头部注意力

    def create_mask(self, qlen, mlen):
        """
        创建自回归注意力的掩码，浮点掩码，其中1.0表示掩盖，0.0表示未掩盖。

        Args:
            qlen: 多长的查询序列？
            mlen: 多长的序列？

        ```

                  same_length=False:      same_length=True:
                  <mlen > <  qlen >       <mlen > <  qlen >
               ^ [0 0 0 0 0 1 1 1 1]     [0 0 0 0 0 1 1 1 1]
                 [0 0 0 0 0 0 1 1 1]     [1 0 0 0 0 0 1 1 1]
            qlen [0 0 0 0 0 0 0 1 1]     [1 1 0 0 0 0 0 1 1]
                 [0 0 0 0 0 0 0 0 1]     [1 1 1 0 0 0 0 0 1]
               v [0 0 0 0 0 0 0 0 0]     [1 1 1 1 0 0 0 0 0]
        ```
        """
        # 创建一个全1的掩码矩阵
        attn_mask = tf.ones([qlen, qlen])
        # 创建上三角掩码矩阵
        mask_u = tf.linalg.band_part(attn_mask, 0, -1)
        # 创建对角线掩码矩阵
        mask_dia = tf.linalg.band_part(attn_mask, 0, 0)
        # 创建形状为（qlen, mlen）的全零矩阵
        attn_mask_pad = tf.zeros([qlen, mlen])
        # 将mask_u - mask_dia和attn_mask_pad连接起来
        ret = tf.concat([attn_mask_pad, mask_u - mask_dia], 1)
        # 如果设置了same_length，则再创建一个下三角掩码矩阵
        if self.same_length:
            mask_l = tf.linalg.band_part(attn_mask, -1, 0)
            # 将ret[:, :qlen] + mask_l - mask_dia和ret[:, qlen:]连接起来
            ret = tf.concat([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], 1)
        return ret

    def cache_mem(self, curr_out, prev_mem):
        # 缓存隐藏状态到内存
        # 如果设置了reuse_len，则截断当前输出
        if self.reuse_len is not None and self.reuse_len > 0:
            curr_out = curr_out[: self.reuse_len]

        # 如果mem_len为None或为0，则返回过去��当前的所有隐藏状态
        if self.mem_len is None or self.mem_len == 0:
            cutoff = 0
        else:
            # 如果mem_len已定义，则返回最后mem_len个隐藏状态
            cutoff = -self.mem_len
        if prev_mem is None:
            # 如果prev_mem为None，则返回从cutoff开始的所有隐藏状态
            new_mem = curr_out[cutoff:]
        else:
            # 如果prev_mem不为None，则将prev_mem和curr_out连接起来，并截取从cutoff开始的所有隐藏状态
            new_mem = tf.concat([prev_mem, curr_out], 0)[cutoff:]

        return tf.stop_gradient(new_mem)

    @staticmethod
    ...
    # 生成位置编码
    def positional_embedding(pos_seq, inv_freq, bsz=None):
        # 计算正弦和余弦函数的输入
        sinusoid_inp = tf.einsum("i,d->id", pos_seq, inv_freq)
        # 拼接正弦和余弦函数的结果，按最后一个维度拼接
        pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], axis=-1)
        # 在0维度上增加一个维度
        pos_emb = pos_emb[:, None, :]

        # 如果有指定bsz，则复制pos_emb，将其扩展成[1, bsz, 某个维度]
        if bsz is not None:
            pos_emb = tf.tile(pos_emb, [1, bsz, 1])

        return pos_emb

    # 生成相对位置编码
    def relative_positional_encoding(self, qlen, klen, bsz=None):
        """create relative positional encoding."""
        freq_seq = tf.range(0, self.d_model, 2.0)
        inv_freq = 1 / (10000 ** (freq_seq / self.d_model))

        if self.attn_type == "bi":
            # 如果是双向注意力，设置开始和结束的位置
            beg, end = klen, -qlen
        elif self.attn_type == "uni":
            # 如果是单向注意力，设置开始和结束的位置
            beg, end = klen, -1
        else:
            raise ValueError(f"Unknown `attn_type` {self.attn_type}.")

        # 如果使用双向数据
        if self.bi_data:
            fwd_pos_seq = tf.range(beg, end, -1.0)
            bwd_pos_seq = tf.range(-beg, -end, 1.0)

            # 如果设定了截断长度，则限制位置编码的范围
            if self.clamp_len > 0:
                fwd_pos_seq = tf.clip_by_value(fwd_pos_seq, -self.clamp_len, self.clamp_len)
                bwd_pos_seq = tf.clip_by_value(bwd_pos_seq, -self.clamp_len, self.clamp_len)

            if bsz is not None:
                # 如果有指定bsz，则将双向位置编码分别传入位置编码函数并进行拼接
                if bsz % 2 != 0:
                    raise ValueError(f"With bi_data, the batch size {bsz} should be divisible by 2")
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz // 2)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq, bsz // 2)
            else:
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq)

            # 将双向位置编码拼接在一起
            pos_emb = tf.concat([fwd_pos_emb, bwd_pos_emb], axis=1)
        else:
            fwd_pos_seq = tf.range(beg, end, -1.0)
            if self.clamp_len > 0:
                fwd_pos_seq = tf.clip_by_value(fwd_pos_seq, -self.clamp_len, self.clamp_len)
            # 若没有指定bsz，则直接生成单向位置编码
            pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz)

        return pos_emb

    # 模型的调用函数
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
class TFXLNetPreTrainedModel(TFPreTrainedModel):
    """
    一个处理权重初始化和简单接口用于下载和加载预训练模型的抽象类。
    """

    # XLNet 模型的配置类
    config_class = XLNetConfig
    # 模型名称前缀
    base_model_prefix = "transformer"


@dataclass
class TFXLNetModelOutput(ModelOutput):
    """
    [`TFXLNetModel`] 的输出类型。

    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, num_predict, hidden_size)`):
            模型最后一层的隐藏状态序列。

            `num_predict` 对应于 `target_mapping.shape[1]`。如果 `target_mapping` 是 `None`，则 `num_predict` 对应于 `sequence_length`。
        mems (`List[tf.Tensor]` of length `config.n_layers`):
            包含预先计算的隐藏态。可以用来加速顺序解码。将这个模型的过去传递给这些模型的 Token ID 不应该作为 `input_ids` 传递，因为它们已经计算过。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            形状为 `(batch_size, sequence_length, hidden_size)` 的 `tf.Tensor` 元组。

            每一层的模型的隐藏状态加上初始嵌入输出。
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            形状为 `(batch_size, num_heads, sequence_length, sequence_length)` 的 `tf.Tensor` 元组。

            注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    # 最后一层的隐藏状态
    last_hidden_state: tf.Tensor = None
    # 计算过的隐藏态列表
    mems: List[tf.Tensor] | None = None
    # 隐藏态的元组
    hidden_states: Tuple[tf.Tensor] | None = None
    # 注意力权重的元组
    attentions: Tuple[tf.Tensor] | None = None


@dataclass
class TFXLNetLMHeadModelOutput(ModelOutput):
    """
    [`TFXLNetLMHeadModel`] 的输出类型。
    """
    # loss参数: `tf.Tensor`类型，形状为*(1,)*，可选参数，labels参数存在时返回。语言建模损失（用于下一个标记的预测）。
    # logits参数: `tf.Tensor`类型，形状为`(batch_size, num_predict, config.vocab_size)`。语言建模头的预测分数（SoftMax之前每个词汇标记的分数）。
    # mems参数: `List[tf.Tensor]`类型，长度为`config.n_layers`。包含预先计算的隐藏状态。可用于加速顺序解码。将过去的标记id传递给模型时，不应将其作为`input_ids`传递，因为它们已经计算过了。
    # hidden_states参数: `tuple(tf.Tensor)`类型，可选参数，当传递了`output_hidden_states=True`或`config.output_hidden_states=True`时返回。形状为`(batch_size, sequence_length, hidden_size)`的`tf.Tensor`元组。
    # attentions参数: `tuple(tf.Tensor)`类型，可选参数，当传递了`output_attentions=True`或`config.output_attentions=True`时返回。形状为`(batch_size, num_heads, sequence_length, sequence_length)`的`tf.Tensor`元组。
# 定义一个数据类，用于存储 XLNet 用于序列分类的输出
@dataclass
class TFXLNetForSequenceClassificationOutput(ModelOutput):
    """
    Output type of [`TFXLNetForSequenceClassification`].

    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `label` is provided):
            分类（如果配置文件中 num_labels==1 则为回归）的损失。
        logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`):
            分类（如果配置文件中 num_labels==1 则为回归）的分数（SoftMax 之前）。
        mems (`List[tf.Tensor]` of length `config.n_layers`):
            包含预先计算的隐藏状态。可用于加速顺序解码。已经在此模型中给定其过去的令牌 id 的输入不应该作为 input_ids 传递，因为它们已经计算过。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            形状为 `(batch_size, sequence_length, hidden_size)` 的 `tf.Tensor` 元组（一个用于嵌入的输出 + 一个用于每个层的输出）。

            每一层模型在每层的输出加上初始嵌入输出。
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            形状为 `(batch_size, num_heads, sequence_length, sequence_length)` 的 `tf.Tensor` 元组（每个层一个）。

            注意力 softmax 之后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    loss: tf.Tensor | None = None
    logits: tf.Tensor = None
    mems: List[tf.Tensor] | None = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None


@dataclass
class TFXLNetForTokenClassificationOutput(ModelOutput):
    """
    Output type of [`TFXLNetForTokenClassificationOutput`].
    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            分类损失。
        logits (`tf.Tensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            分类分数（SoftMax之前）。
        mems (`List[tf.Tensor]` of length `config.n_layers`):
            包含预先计算的隐藏状态。可以用于加速顺序解码。已经计算过其过去的token id不应该作为`input_ids`传递给该模型。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            形状为`(batch_size, sequence_length, hidden_size)`的`tf.Tensor`元组（一个用于嵌入的输出 + 一个用于每一层的输出）。

            模型在每一层输出的隐藏状态加上初始嵌入输出。
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            形状为`(batch_size, num_heads, sequence_length, sequence_length)`的`tf.Tensor`元组（每一层一个）。

            注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    loss: tf.Tensor | None = None
    logits: tf.Tensor = None
    mems: List[tf.Tensor] | None = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None
# 定义了一个数据类 TFXLNetForMultipleChoiceOutput，用于存储 TF-XLNet 多选题模型的输出
@dataclass
class TFXLNetForMultipleChoiceOutput(ModelOutput):
    """
    Output type of [`TFXLNetForMultipleChoice`].

    Args:
        loss (`tf.Tensor` of shape *(1,)*, *optional*, returned when `labels` is provided):
            分类损失。
        logits (`tf.Tensor` of shape `(batch_size, num_choices)`):
            *num_choices* 是输入张量的第二个维度。(参见上面的 *input_ids*)。

            分类分数（SoftMax 之前）。
        mems (`List[tf.Tensor]` of length `config.n_layers`):
            包含预计算的隐藏状态。可用于加快顺序解码的过程。已经计算过过去给出给定模型的 token id 不应作为 `input_ids` 传递，因为它们已经被计算过了。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            `tf.Tensor` 元组（一个用于嵌入的输出 + 一个用于每个层的输出）的形状为 `(batch_size, sequence_length, hidden_size)`。

            模型在每一层输出的隐藏状态加上初始嵌入输出。
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            `tf.Tensor` 元组（每层一个）的形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    # 分类损失，默认为 None
    loss: tf.Tensor | None = None
    # 分类分数
    logits: tf.Tensor = None
    # 隐藏状态
    mems: List[tf.Tensor] | None = None
    # 每层的隐藏状态
    hidden_states: Tuple[tf.Tensor] | None = None
    # 注意力权重
    attentions: Tuple[tf.Tensor] | None = None



@dataclass
class TFXLNetForQuestionAnsweringSimpleOutput(ModelOutput):
    """
    Output type of [`TFXLNetForQuestionAnsweringSimple`].
    """
    # 参数:
    loss (`tf.Tensor` of shape `(1,)`, *optional*, 当`labels`被提供时返回):
        总的跨度抽取损失是起始位置和结束位置的交叉熵的总和。
    start_logits (`tf.Tensor` of shape `(batch_size, sequence_length,)`):
        跨度起始得分（SoftMax之前）。
    end_logits (`tf.Tensor` of shape `(batch_size, sequence_length,)`):
        跨度结束得分（SoftMax之前）。
    mems (`List[tf.Tensor]` of length `config.n_layers`):
        包含预先计算的隐藏状态。可以用于加速顺序解码。将已计算过去给这个模型的标记 ID 不应作为`input_ids`传递，因为它们已经被计算过。
    hidden_states (`tuple(tf.Tensor)`, *optional*, 当`output_hidden_states=True`被传递或当`config.output_hidden_states=True`时返回):
        形状为`(batch_size, sequence_length, hidden_size)`的 `tf.Tensor` 元组（一个用于嵌入输出，一个用于每个层的输出）。

        每层模型的隐藏状态加上初始嵌入输出。
    attentions (`tuple(tf.Tensor)`, *optional*, 当`output_attentions=True`被传递或当`config.output_attentions=True`时返回):
        形状为`(batch_size, num_heads, sequence_length, sequence_length)`的 `tf.Tensor` 元组（每层一个）。

        在注意力 softmax 之后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    loss: tf.Tensor | None = None
    start_logits: tf.Tensor = None
    end_logits: tf.Tensor = None
    mems: List[tf.Tensor] | None = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None
# XLNet 模型的文档字符串，包含了模型的继承信息、使用方法以及参数说明
XLNET_START_DOCSTRING = r"""

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

    Parameters:
        config ([`XLNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# XLNet 模型的输入说明文档字符串，暂时为空
XLNET_INPUTS_DOCSTRING = r"""
"""


# 添加文档字符串到模型上，包括模型的基本信息以及参数说明
@add_start_docstrings(
    "The bare XLNet Model transformer outputting raw hidden-states without any specific head on top.",
    XLNET_START_DOCSTRING,
)
class TFXLNetModel(TFXLNetPreTrainedModel):
    # 初始化方法
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 创建 XLNet 主层
        self.transformer = TFXLNetMainLayer(config, name="transformer")

    # 定义模型的前向传播方法
    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 使用装饰器为该函数添加代码示例和文档字符串，用于自动生成 API 文档
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,  # 示例代码检查点
        output_type=TFXLNetModelOutput,  # 输出类型为TFXLNetModelOutput
        config_class=_CONFIG_FOR_DOC,  # 用于示例文档的配置类
    )
    # 定义一个call方法，用于执行XLNet模型的前向传播
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 输入的token ID，可以为空
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力掩码，可以为numpy数组或张量，可以为空
        mems: np.ndarray | tf.Tensor | None = None,  # 记忆内容，可以为numpy数组或张量，可以为空
        perm_mask: np.ndarray | tf.Tensor | None = None,  # 排列掩码，可以为numpy数组或张量，可以为空
        target_mapping: np.ndarray | tf.Tensor | None = None,  # 目标映射，可以为numpy数组或张量，可以为空
        token_type_ids: np.ndarray | tf.Tensor | None = None,  # token类型ID，可以为numpy数组或张量，可以为空
        input_mask: np.ndarray | tf.Tensor | None = None,  # 输入掩码，可以为numpy数组或张量，可以为空
        head_mask: np.ndarray | tf.Tensor | None = None,  # 头掩码，可以为numpy数组或张量，可以为空
        inputs_embeds: np.ndarray | tf.Tensor | None = None,  # 输入嵌入层，可以为numpy数组或张量，可以为空
        use_mems: Optional[bool] = None,  # 是否使用记忆，可选的布尔值，默认为空
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选的布尔值，默认为空
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏层状态，可选的布尔值，默认为空
        return_dict: Optional[bool] = None,  # 是否返回字典形式的输出，可选的布尔值，默认为空
        training: bool = False,  # 是否在训练模式中，布尔值，默认为False
    ) -> Union[TFXLNetModelOutput, Tuple[tf.Tensor]]:  # 返回值的类型为TFXLNetModelOutput或tf.Tensor的元组
        # 调用transformer方法执行XLNet模型的前向传播，并返回输出结果
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
        # 返回模型的输出结果
        return outputs
    
    # 构建模型，初始化权重
    def build(self, input_shape=None):
        # 如果已经构建过，则���接返回
        if self.built:
            return
        # 设置模型已构建标志为True
        self.built = True
        # 检查transformer属性是否存在
        if getattr(self, "transformer", None) is not None:
            # 在transformer.name的命名空间内构建transformer模型
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
# 使用 add_start_docstrings 方法为模型添加文档字符串
@add_start_docstrings(
    """
    XLNet Model with a language modeling head on top (linear layer with weights tied to the input embeddings).
    """,
    XLNET_START_DOCSTRING,
)
# TFXLNetLMHeadModel 继承自 TFXLNetPreTrainedModel 和 TFCausalLanguageModelingLoss
class TFXLNetLMHeadModel(TFXLNetPreTrainedModel, TFCausalLanguageModelingLoss):
    # 初始化方法
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 创建 transformer 层
        self.transformer = TFXLNetMainLayer(config, name="transformer")
        # 创建 lm_loss 层
        self.lm_loss = TFXLNetLMHead(config, self.transformer.word_embedding, name="lm_loss")
        # 不能将模型转换为图形
        self.supports_xla_generation = False

    # 获取 lm_head
    def get_lm_head(self):
        return self.lm_loss

    # 获取前缀偏差名称
    def get_prefix_bias_name(self):
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        return self.name + "/" + self.lm_loss.name

    # 为生成准备输入
    def prepare_inputs_for_generation(self, inputs, past_key_values=None, use_mems=None, **kwargs):
        # 在末尾添加虚拟标记（在此标记上没有注意力）
        effective_batch_size = inputs.shape[0]
        dummy_token = tf.zeros((effective_batch_size, 1), dtype=inputs.dtype)

        # 在每次解码通行证中，计算新标记和最后两个生成标记的注意力值，
        # 其余内容从“过去”缓存中重新加载。纯自回归模型将具有偏移 = 1；偏移 = 2 似乎计算稍微更好。
        offset = 2

        if past_key_values:
            input_ids = tf.concat([inputs[:, -offset:], dummy_token], axis=1)
        else:
            input_ids = tf.concat([inputs, dummy_token], axis=1)

        # 构建排列掩码，使前一个标记不看到最后一个标记
        sequence_length = input_ids.shape[1]
        perm_mask = tf.zeros((effective_batch_size, sequence_length, sequence_length - 1))
        perm_mask_seq_end = tf.ones((effective_batch_size, sequence_length, 1))
        perm_mask = tf.concat([perm_mask, perm_mask_seq_end], axis=-1)

        # 我们只会预测最后一个标记
        target_mapping = tf.zeros((effective_batch_size, 1, sequence_length - 1))
        target_mapping_seq_end = tf.ones((effective_batch_size, 1, 1))
        target_mapping = tf.concat([target_mapping, target_mapping_seq_end], axis=-1)

        inputs = {
            "input_ids": input_ids,
            "perm_mask": perm_mask,
            "target_mapping": target_mapping,
            "use_mems": use_mems,
        }

        # 如果在模��kwargs中定义了过去，则使用它来加快解码
        if past_key_values:
            inputs["mems"] = tuple(layer_past[:-offset, :, :] for layer_past in past_key_values)

        return inputs

    # 为 model_forward 添加文档字符串
    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFXLNetLMHeadModelOutput, config_class=_CONFIG_FOR_DOC)
    # 定义一个方法，用于调用模型
    def call(
        # 输入标识符，可以是TensorFlow模型输入类型或None
        self,
        input_ids: TFModelInputType | None = None,
        # 注意力掩码，可以是NumPy数组、TensorFlow张量或None
        attention_mask: np.ndarray | tf.Tensor | None = None,
        # 记忆，可以是NumPy数组、TensorFlow张量或None
        mems: np.ndarray | tf.Tensor | None = None,
        # 排列掩码，可以是NumPy数组、TensorFlow张量或None
        perm_mask: np.ndarray | tf.Tensor | None = None,
        # 目标映射，可以是NumPy数组、TensorFlow张量或None
        target_mapping: np.ndarray | tf.Tensor | None = None,
        # 标记类型标识符，可以是NumPy数组、TensorFlow张量或None
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        # 输入掩码，可以是NumPy数组、TensorFlow张量或None
        input_mask: np.ndarray | tf.Tensor | None = None,
        # 头部掩码，可以是NumPy数组、TensorFlow张量或None
        head_mask: np.ndarray | tf.Tensor | None = None,
        # 输入嵌入，可以是NumPy数组、TensorFlow张量或None
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        # 是否使用记忆
        use_mems: Optional[bool] = None,
        # 是否输出注意力权重
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典
        return_dict: Optional[bool] = None,
        # 标签，可以是NumPy数组、TensorFlow张量或None
        labels: np.ndarray | tf.Tensor | None = None,
        # 是否处于训练模式
        training: bool = False,
    # 构建方法，用于构建模型
    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回
        if self.built:
            return
        # 设置模型已构建标志为True
        self.built = True
        # 如果存在transformer属性
        if getattr(self, "transformer", None) is not None:
            # 在transformer的命名范围内构建transformer
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        # 如果存在lm_loss属性
        if getattr(self, "lm_loss", None) is not None:
            # 在lm_loss的命名范围内构建lm_loss
            with tf.name_scope(self.lm_loss.name):
                self.lm_loss.build(None)
```  
# 使用给定的 XLNet 配置和输入构建一个带有序列分类/回归头部的 XLNet 模型，例如用于 GLUE 任务
class TFXLNetForSequenceClassification(TFXLNetPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的构造函数
        super().__init__(config, *inputs, **kwargs)
        # 设置标签的数量
        self.num_labels = config.num_labels

        # 构建 XLNet 主体层
        self.transformer = TFXLNetMainLayer(config, name="transformer")
        # 构建序列摘要层
        self.sequence_summary = TFSequenceSummary(
            config, initializer_range=config.initializer_range, name="sequence_summary"
        )
        # 构建逻辑回归层
        self.logits_proj = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="logits_proj"
        )
        # 保存配置
        self.config = config

    # 装饰 call 方法，用于模型的前向传播
    @unpack_inputs
    # 添加模型前向传播的文档字符串
    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 添加代码示例的文档字符串
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFXLNetForSequenceClassificationOutput,
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
    ) -> Union[TFXLNetForSequenceClassificationOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional`):
            用于计算序列分类/回归损失的标签。指数应在`[0, ..., config.num_labels - 1]`范围内。如果`config.num_labels == 1`，则计算回归损失（均方损失）；如果`config.num_labels > 1`，则计算分类损失（交叉熵损失）。
        """
        # 将输入传递给transformer模型，并获取transformer模型的输出
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
        # 从transformer输出中获取第一个元素
        output = transformer_outputs[0]

        # 对输出进行序列摘要
        output = self.sequence_summary(output)
        # 将序列摘要过的输出传递给logits_proj层
        logits = self.logits_proj(output)

        # 如果labels不为None，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果return_dict为False，则返回输出和损失；否则以TFXLNetForSequenceClassificationOutput的形式返回输出、损失、mems、隐藏状态和注意力权重
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFXLNetForSequenceClassificationOutput(
            loss=loss,
            logits=logits,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    # 构建模型
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果transformer存在，则构建它
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        # 如果sequence_summary存在，则构建它
        if getattr(self, "sequence_summary", None) is not None:
            with tf.name_scope(self.sequence_summary.name):
                self.sequence_summary.build(None)
        # 如果logits_proj存在，则构建它
        if getattr(self, "logits_proj", None) is not None:
            with tf.name_scope(self.logits_proj.name):
                self.logits_proj.build([None, None, self.config.d_model])
    # 添加起始文档字符串
    @add_start_docstrings(
        """
        XLNET Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
        softmax) e.g. for RocStories/SWAG tasks.
        """,
        XLNET_START_DOCSTRING,
    )
    # 定义 TFXLNetForMultipleChoice 类，继承自 TFXLNetPreTrainedModel 和 TFMultipleChoiceLoss
    class TFXLNetForMultipleChoice(TFXLNetPreTrainedModel, TFMultipleChoiceLoss):
        
        # 初始化方法
        def __init__(self, config, *inputs, **kwargs):
            # 调用父类的初始化方法
            super().__init__(config, *inputs, **kwargs)

            # 创建 transformer 层
            self.transformer = TFXLNetMainLayer(config, name="transformer")
            # 创建 sequence_summary 层
            self.sequence_summary = TFSequenceSummary(
                config, initializer_range=config.initializer_range, name="sequence_summary"
            )
            # 创建 logits_proj 层
            self.logits_proj = tf.keras.layers.Dense(
                1, kernel_initializer=get_initializer(config.initializer_range), name="logits_proj"
            )
            # 保存配置信息
            self.config = config

        # 定义 call 方法
        @unpack_inputs
        @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
        @add_code_sample_docstrings(
            checkpoint=_CHECKPOINT_FOR_DOC,
            output_type=TFXLNetForMultipleChoiceOutput,
            config_class=_CONFIG_FOR_DOC,
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
        # 定义函数，用于处理多选题的XLNet模型输出结果
        # labels: 具有形状`(batch_size,)`的`tf.Tensor`类型，用于计算多选题分类损失。索引应该在`[0, ..., num_choices]`之间，其中`num_choices`是输入张量的第二维的大小 (见上面的`input_ids`）
        def call(
            self,
            input_ids: Optional[tf.Tensor] = None,
            attention_mask: Optional[tf.Tensor] = None,
            mems: Optional[List[tf.Tensor]] = None,
            perm_mask: Optional[tf.Tensor] = None,
            target_mapping: Optional[tf.Tensor] = None,
            token_type_ids: Optional[tf.Tensor] = None,
            input_mask: Optional[tf.Tensor] = None,
            inputs_embeds: Optional[tf.Tensor] = None,
            head_mask: Optional[tf.Tensor] = None,
            use_mems: bool = True,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            training: bool = False,
            labels: Optional[tf.Tensor] = None,
        ) -> Union[TFXLNetForMultipleChoiceOutput, Tuple[tf.Tensor]]:
        
        # 如果input_ids不为None，确定多选题的选择数和序列长度
            if input_ids is not None:
                num_choices = shape_list(input_ids)[1]
                seq_length = shape_list(input_ids)[2]
            else:
                num_choices = shape_list(inputs_embeds)[1]
                seq_length = shape_list(inputs_embeds)[2]

            # 对输入进行扁平化处理
            flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
            flat_attention_mask = tf.reshape(attention_mask, (-1, seq_length)) if attention_mask is not None else None
            flat_token_type_ids = tf.reshape(token_type_ids, (-1, seq_length)) if token_type_ids is not None else None
            flat_input_mask = tf.reshape(input_mask, (-1, seq_length)) if input_mask is not None else None
            flat_inputs_embeds = (
                tf.reshape(inputs_embeds, (-1, seq_length, shape_list(inputs_embeds)[3]))
                if inputs_embeds is not None
                else None
            )
            # 使用XLNet模型进行前向传播
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
            # 获取XLNet模型的输出
            output = transformer_outputs[0]
            # 对输出进行序列汇总
            logits = self.sequence_summary(output)
            # 对logits进行投影
            logits = self.logits_proj(logits)
            # 重塑logits的形状
            reshaped_logits = tf.reshape(logits, (-1, num_choices))
            # 如果labels不为None，计算损失
            loss = None if labels is None else self.hf_compute_loss(labels, reshaped_logits)
            
            # 如果return_dict为False，返回输出结果
            if not return_dict:
                output = (reshaped_logits,) + transformer_outputs[1:]
                return ((loss,) + output) if loss is not None else output
            
            # 返回包含损失、logits、mems、hidden_states和attentions的XLNet输出结果
            return TFXLNetForMultipleChoiceOutput(
                loss=loss,
                logits=reshaped_logits,
                mems=transformer_outputs.mems,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
            )
    # 构建神经网络模型，如果已经构建过，则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 标记该模型已经被构建
        self.built = True
        # 如果模型中包含 transformer 属性，则构建 transformer
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        # 如果模型中包含 sequence_summary 属性，则构建 sequence_summary
        if getattr(self, "sequence_summary", None) is not None:
            with tf.name_scope(self.sequence_summary.name):
                self.sequence_summary.build(None)
        # 如果模型中包含 logits_proj 属性，则构建 logits_proj
        if getattr(self, "logits_proj", None) is not None:
            with tf.name_scope(self.logits_proj.name):
                self.logits_proj.build([None, None, self.config.d_model])
# 使用装饰器添加模型文档字符串和XLNet的文档字符串，说明了这是在XLNet模型基础上加上了一个标记分类头
class TFXLNetForTokenClassification(TFXLNetPreTrainedModel, TFTokenClassificationLoss):
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 设置标签数量
        self.num_labels = config.num_labels

        # 初始化XLNet的主层，命名为"transformer"
        self.transformer = TFXLNetMainLayer(config, name="transformer")
        # 初始化分类器，使用线性层，权重初始化为给定配置的初始化范围内的值，命名为"classifier"
        self.classifier = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        # 存储配置信息
        self.config = config

    # 使用装饰器unpack_inputs解包输入参数，为模型的前向传播方法添加模型文档字符串、模型输入文档字符串和代码示例文档字符串
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
    ):
``` 
    ) -> Union[TFXLNetForTokenClassificationOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional`):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        
        # 使用 Transformer 模型对输入进行处理
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
        
        # 获取 Transformer 处理后的输出
        output = transformer_outputs[0]
        # 使用分类器获取 logits
        logits = self.classifier(output)
        # 计算损失（loss），如果 labels 不为空
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        if not return_dict:
            # 如果不返回字典，则组合输出
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回 Token 分类的输出结果对象
        return TFXLNetForTokenClassificationOutput(
            loss=loss,
            logits=logits,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经构建了模型，则直接返回
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                # 使用 Transformer 构建模型
                self.transformer.build(None)
        # 如果存在分类器，则构建分类器
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
# 在 XLNet 模型基础上增加了一个用于抽取性问题回答任务的跨度分类头部
class TFXLNetForQuestionAnsweringSimple(TFXLNetPreTrainedModel, TFQuestionAnsweringLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 初始化 XLNet 主层
        self.transformer = TFXLNetMainLayer(config, name="transformer")
        # 输出层，用于计算跨度开始 logit 和跨度结束 logit
        self.qa_outputs = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )
        self.config = config

    # 定义模型的前向传播函数
    @unpack_inputs
    # 增加输入的文档字符串
    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 增加代码示例的文档字符串
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
    # 定义一个函数，用于回答问题并返回开始和结束位置的逻辑输出
    def call(
        self,
        input_ids: tf.Tensor,
        attention_mask: tf.Tensor,
        mems: Optional[List[tf.Tensor]] = None,
        perm_mask: Optional[tf.Tensor] = None,
        target_mapping: Optional[tf.Tensor] = None,
        token_type_ids: Optional[tf.Tensor] = None,
        input_mask: Optional[tf.Tensor] = None,
        head_mask: Optional[tf.Tensor] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        use_mems: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        start_positions: Optional[tf.Tensor] = None,
        end_positions: Optional[tf.Tensor] = None,
    ) -> Union[TFXLNetForQuestionAnsweringSimpleOutput, Tuple[tf.Tensor]]:
        # 接收问题相关的输入和标签
        # 输入包括 input_ids、attention_mask、mems 等
        # 标签包括 start_positions 和 end_positions
    
        # 将输入传递给 transformer 模型，获取序列输出
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
        sequence_output = transformer_outputs[0]
    
        # 将序列输出传递给问答输出层，获取开始和结束位置的逻辑输出
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)
    
        # 如果提供了开始和结束位置的标签，则计算损失函数
        loss = None
        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            loss = self.hf_compute_loss(labels, (start_logits, end_logits))
    
        # 根据是否需要返回字典，返回相应的输出
        if not return_dict:
            output = (start_logits, end_logits) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
    
        return TFXLNetForQuestionAnsweringSimpleOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
    
    # 定义模型构建函数
    def build(self, input_shape=None):
        # 如果模型已经构建，直接返回
        if self.built:
            return
        self.built = True
    
        # 构建 transformer 子模型
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
    
        # 构建问答输出子模型
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])
```