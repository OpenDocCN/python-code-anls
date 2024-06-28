# `.\models\gpt2\modeling_tf_gpt2.py`

```py
# coding=utf-8
# 文件编码声明，指明使用 UTF-8 编码
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# 版权声明，版权归属于 OpenAI 团队和 HuggingFace Inc. 团队
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# 版权声明，版权归属于 NVIDIA 公司，保留所有权利
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache License, Version 2.0 进行许可，可以免费使用该文件
# you may not use this file except in compliance with the License.
# 除非符合 Apache License，Version 2.0 的规定，否则不能使用此文件
# You may obtain a copy of the License at
# 可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 除非法律要求或书面同意，否则本文件中分发的软件按"原样"分发，
# 没有任何明示或暗示的担保或条件
# See the License for the specific language governing permissions and
# limitations under the License.
# 请参阅许可证，了解具体的语言管理权限和限制

""" TF 2.0 OpenAI GPT-2 model."""
# 此文件是 TF 2.0 版本的 OpenAI GPT-2 模型的实现

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

# 从外部模块导入必要的函数和类
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFCausalLMOutputWithCrossAttentions,
    TFSequenceClassifierOutputWithPast,
)
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFConv1D,
    TFModelInputType,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    TFSequenceSummary,
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
from .configuration_gpt2 import GPT2Config

# 获取全局日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置
_CHECKPOINT_FOR_DOC = "openai-community/gpt2"
_CONFIG_FOR_DOC = "GPT2Config"

# GPT-2 的预训练模型存档列表
TF_GPT2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "openai-community/gpt2",
    "openai-community/gpt2-medium",
    "openai-community/gpt2-large",
    "openai-community/gpt2-xl",
    "distilbert/distilgpt2",
    # See all GPT-2 models at https://huggingface.co/models?filter=openai-community/gpt2
]

# 定义 TFAttention 类，继承自 keras.layers.Layer
class TFAttention(keras.layers.Layer):
    def __init__(self, nx, config, scale=False, is_cross_attention=False, **kwargs):
        super().__init__(**kwargs)

        n_state = nx  # 在注意力机制中，n_state=768 (nx=n_embd)
        # [将 nx 替换为 n_state，以保持与 TF 实现的一致性]
        assert n_state % config.n_head == 0
        self.n_head = config.n_head  # 设置注意头的数量
        self.split_size = n_state  # 分割大小设置为 n_state
        self.scale = scale  # 是否进行缩放
        self.output_attentions = config.output_attentions  # 是否输出注意力权重

        self.is_cross_attention = is_cross_attention  # 是否为交叉注意力

        if self.is_cross_attention:
            self.c_attn = TFConv1D(n_state * 2, nx, initializer_range=config.initializer_range, name="c_attn")
            self.q_attn = TFConv1D(n_state, nx, initializer_range=config.initializer_range, name="q_attn")
        else:
            self.c_attn = TFConv1D(n_state * 3, nx, initializer_range=config.initializer_range, name="c_attn")
        # 根据是否为交叉注意力设置不同的卷积层

        self.c_proj = TFConv1D(n_state, nx, initializer_range=config.initializer_range, name="c_proj")
        # c_proj 卷积层设置

        self.attn_dropout = keras.layers.Dropout(config.attn_pdrop)  # 注意力分数的 dropout
        self.resid_dropout = keras.layers.Dropout(config.resid_pdrop)  # 残差的 dropout
        self.pruned_heads = set()  # 初始化修剪的注意力头集合
        self.embed_dim = n_state  # 嵌入维度设置为 n_state

    def prune_heads(self, heads):
        pass  # 修剪注意力头的方法，当前为空

    @staticmethod
    def causal_attention_mask(nd, ns, dtype):
        """
        生成因果注意力掩码，下三角矩阵，从右下角开始计算。与 tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd) 相同，
        但在 TPUs 上不会产生垃圾数据。
        """
        i = tf.range(nd)[:, None]
        j = tf.range(ns)
        m = i >= j - ns + nd
        return tf.cast(m, dtype)

    def _attn(self, q, k, v, attention_mask, head_mask, output_attentions, training=False):
        # q, k, v 的形状为 [batch, heads, sequence, features]

        w = tf.matmul(q, k, transpose_b=True)  # 计算注意力分数

        if self.scale:
            dk = tf.cast(shape_list(k)[-1], dtype=w.dtype)  # 缩放注意力分数
            w = w / tf.math.sqrt(dk)

        if not self.is_cross_attention:
            # 如果不是交叉注意力，实现因果掩码
            _, _, nd, ns = shape_list(w)
            b = self.causal_attention_mask(nd, ns, dtype=w.dtype)
            b = tf.reshape(b, [1, 1, nd, ns])
            w = w * b - 1e4 * (1 - b)

        if attention_mask is not None:
            # 应用给定的注意力掩码
            attention_mask = tf.cast(attention_mask, dtype=w.dtype)
            w = w + attention_mask

        w = stable_softmax(w, axis=-1)  # 对注意力分数进行 softmax
        w = self.attn_dropout(w, training=training)  # 应用注意力 dropout

        if head_mask is not None:
            w = w * head_mask  # 如果有头部掩码，应用头部掩码

        outputs = [tf.matmul(w, v)]  # 计算加权和
        if output_attentions:
            outputs.append(w)  # 如果需要输出注意力权重，添加到输出中
        return outputs  # 返回输出结果
    def merge_heads(self, x):
        # 将输入张量 x 的维度进行转置，[0, 2, 1, 3] 表示将第二维和第三维进行交换
        x = tf.transpose(x, [0, 2, 1, 3])
        # 获取输入张量 x 的形状
        x_shape = shape_list(x)
        # 根据 x 的形状，将其最后两个维度合并成一个新的维度
        new_x_shape = x_shape[:-2] + [x_shape[-2] * x_shape[-1]]
        # 将输入张量 x 重新按照新的形状进行重塑
        return tf.reshape(x, new_x_shape)

    def split_heads(self, x):
        # 获取输入张量 x 的形状
        x_shape = shape_list(x)
        # 根据 x 的形状，将其最后一个维度分割成两个维度，用于实现多头注意力机制
        new_x_shape = x_shape[:-1] + [self.n_head, x_shape[-1] // self.n_head]
        # 将输入张量 x 按照新的形状进行重塑
        x = tf.reshape(x, new_x_shape)
        # 将输入张量 x 的维度进行转置，(0, 2, 1, 3) 表示维度的调整顺序
        return tf.transpose(x, (0, 2, 1, 3))  # (batch, head, seq_length, head_features)

    def call(
        self,
        x,
        layer_past,
        attention_mask,
        head_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        use_cache,
        output_attentions,
        training=False,
    ):
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            # 对输入张量 x 进行 q_attn 权重的操作，得到查询张量 query
            query = self.q_attn(x)
            # 对编码器隐藏状态进行 c_attn 权重的操作，得到键值对张量 kv_out
            kv_out = self.c_attn(encoder_hidden_states)
            # 将键值对张量 kv_out 沿着最后一个维度分割成两个张量，分别表示键和值
            key, value = tf.split(kv_out, 2, axis=2)
            # 注意力遮罩掩码为编码器的注意力掩码
            attention_mask = encoder_attention_mask
        else:
            # 对输入张量 x 进行 c_attn 权重的操作
            x = self.c_attn(x)
            # 将处理后的张量 x 分割成查询、键、值三个张量
            query, key, value = tf.split(x, 3, axis=2)

        # 将查询张量 query 进行多头分割处理
        query = self.split_heads(query)
        # 将键张量 key 进行多头分割处理
        key = self.split_heads(key)
        # 将值张量 value 进行多头分割处理
        value = self.split_heads(value)

        if layer_past is not None:
            # 如果过去的层存在，则进行未来信息的拼接
            past_key, past_value = tf.unstack(layer_past, axis=0, num=2)
            key = tf.concat([past_key, key], axis=-2)
            value = tf.concat([past_value, value], axis=-2)

        # 用于处理 keras 序列化问题，根据 use_cache 决定返回的张量
        if use_cache:
            present = tf.stack([key, value], axis=0)
        else:
            present = (None,)

        # 对查询、键、值张量进行自注意力计算，包括非归一化的抑制机制、掩码、多头注意力输出等
        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions, training=training)
        a = attn_outputs[0]

        # 将多头注意力输出张量进行头合并操作
        a = self.merge_heads(a)
        # 对合并后的张量进行 c_proj 权重的操作
        a = self.c_proj(a)
        # 对 c_proj 结果进行残差连接和 dropout 处理
        a = self.resid_dropout(a, training=training)

        # 输出结果包括 a（处理后的张量）、present（用于处理 keras 序列化问题）、attentions（注意力结果）
        outputs = [a, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions)

    def build(self, input_shape=None):
        # 如果已经构建过则直接返回
        if self.built:
            return
        self.built = True
        # 如果是交叉注意力，则 c_attn_shape 为 2 倍的 embed_dim，否则为 3 倍
        if self.is_cross_attention:
            c_attn_shape = 2 * self.embed_dim
        else:
            c_attn_shape = 3 * self.embed_dim
        # 对 c_proj 层、c_attn 层、q_attn 层进行构建
        if getattr(self, "c_proj", None) is not None:
            with tf.name_scope(self.c_proj.name):
                # 构建 c_proj 层
                self.c_proj.build([None, None, self.embed_dim])
        if getattr(self, "c_attn", None) is not None:
            with tf.name_scope(self.c_attn.name):
                # 构建 c_attn 层
                self.c_attn.build([None, None, c_attn_shape])
        if getattr(self, "q_attn", None) is not None:
            with tf.name_scope(self.q_attn.name):
                # 构建 q_attn 层
                self.q_attn.build([None, None, self.embed_dim])
class TFMLP(keras.layers.Layer):
    def __init__(self, n_state, config, **kwargs):
        super().__init__(**kwargs)
        nx = config.n_embd  # 从配置中获取嵌入维度大小
        # 创建第一个一维卷积层，用于处理状态和嵌入维度之间的转换
        self.c_fc = TFConv1D(n_state, nx, initializer_range=config.initializer_range, name="c_fc")
        # 创建第二个一维卷积层，用于嵌入维度和状态之间的转换
        self.c_proj = TFConv1D(nx, n_state, initializer_range=config.initializer_range, name="c_proj")
        # 获取激活函数
        self.act = get_tf_activation(config.activation_function)
        # 创建一个丢弃层，用于在训练时随机丢弃部分数据，以减少过拟合
        self.dropout = keras.layers.Dropout(config.resid_pdrop)
        self.intermediate_size = n_state  # 中间层的大小
        self.embed_dim = nx  # 嵌入维度大小

    def call(self, x, training=False):
        # 应用激活函数到第一个卷积层的输出
        h = self.act(self.c_fc(x))
        # 将第一个卷积层的输出输入到第二个卷积层中
        h2 = self.c_proj(h)
        # 在训练时对第二个卷积层的输出进行随机丢弃
        h2 = self.dropout(h2, training=training)
        return h2

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "c_fc", None) is not None:
            with tf.name_scope(self.c_fc.name):
                # 构建第一个卷积层
                self.c_fc.build([None, None, self.intermediate_size])
        if getattr(self, "c_proj", None) is not None:
            with tf.name_scope(self.c_proj.name):
                # 构建第二个卷积层
                self.c_proj.build([None, None, self.embed_dim])


class TFBlock(keras.layers.Layer):
    def __init__(self, config, scale=False, **kwargs):
        super().__init__(**kwargs)
        nx = config.n_embd  # 从配置中获取嵌入维度大小
        # 内部维度大小为 n_inner 或者默认为 4 * nx
        inner_dim = config.n_inner if config.n_inner is not None else 4 * nx
        # 第一个层归一化层，用于归一化输入数据
        self.ln_1 = keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="ln_1")
        # 自注意力层，用于学习输入序列内部的依赖关系
        self.attn = TFAttention(nx, config, scale, name="attn")
        # 第二个层归一化层，用于归一化自注意力层的输出
        self.ln_2 = keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="ln_2")

        if config.add_cross_attention:
            # 如果配置中需要加入跨注意力机制，则创建跨注意力层
            self.crossattention = TFAttention(nx, config, scale, name="crossattention", is_cross_attention=True)
            # 对跨注意力层的输出进行归一化
            self.ln_cross_attn = keras.layers.LayerNormalization(
                epsilon=config.layer_norm_epsilon, name="ln_cross_attn"
            )

        # 多层感知机，用于处理每个注意力块的输出
        self.mlp = TFMLP(inner_dim, config, name="mlp")
        self.hidden_size = config.hidden_size  # 隐藏层大小

    def call(
        self,
        x,
        layer_past,
        attention_mask,
        head_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        use_cache,
        output_attentions,
        training=False,
    ):
        # 使用 self.ln_1 对输入 x 进行层归一化处理
        a = self.ln_1(x)
        # 使用 self.attn 进行自注意力机制操作
        output_attn = self.attn(
            a,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            training=training,
        )
        # 从 output_attn 中提取注意力权重 a
        a = output_attn[0]  # output_attn: a, present, (attentions)
        # 提取除注意力权重外的其他输出
        outputs = output_attn[1:]
        # 更新 x，加上注意力权重 a
        x = x + a

        # Cross-Attention Block
        # 如果存在编码器隐藏状态，则添加交叉注意力块
        if encoder_hidden_states is not None:
            # 检查是否已实例化 self.crossattention
            if not hasattr(self, "crossattention"):
                # 抛出异常，要求在实例化时设置 config.add_cross_attention=True
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )

            # 使用 self.ln_cross_attn 对输入 x 进行层归一化处理
            ca = self.ln_cross_attn(x)
            # 使用 self.crossattention 进行交叉注意力机制操作
            output_cross_attn = self.crossattention(
                ca,
                layer_past=None,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=False,
                output_attentions=output_attentions,
                training=training,
            )
            # 从 output_cross_attn 中提取交叉注意力权重 ca
            ca = output_cross_attn[0]  # output_attn: a, present, (cross_attentions)
            # 更新 x，加上交叉注意力权重 ca
            x = x + ca
            # 添加交叉注意力权重到输出
            outputs = outputs + output_cross_attn[2:]  # add cross attentions if we output attention weights

        # 使用 self.ln_2 对更新后的 x 进行层归一化处理
        m = self.ln_2(x)
        # 使用 self.mlp 进行多层感知机操作
        m = self.mlp(m, training=training)
        # 更新 x，加上多层感知机输出 m
        x = x + m

        # 将更新后的 x 和输出列表组合成最终输出
        outputs = [x] + outputs
        # 返回最终输出，包括 x、present、(attentions, cross_attentions)
        return outputs  # x, present, (attentions, cross_attentions)

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 检查并构建 self.ln_1
        if getattr(self, "ln_1", None) is not None:
            with tf.name_scope(self.ln_1.name):
                self.ln_1.build([None, None, self.hidden_size])
        # 检查并构建 self.attn
        if getattr(self, "attn", None) is not None:
            with tf.name_scope(self.attn.name):
                self.attn.build(None)
        # 检查并构建 self.ln_2
        if getattr(self, "ln_2", None) is not None:
            with tf.name_scope(self.ln_2.name):
                self.ln_2.build([None, None, self.hidden_size])
        # 检查并构建 self.mlp
        if getattr(self, "mlp", None) is not None:
            with tf.name_scope(self.mlp.name):
                self.mlp.build(None)
        # 检查并构建 self.crossattention
        if getattr(self, "crossattention", None) is not None:
            with tf.name_scope(self.crossattention.name):
                self.crossattention.build(None)
        # 检查并构建 self.ln_cross_attn
        if getattr(self, "ln_cross_attn", None) is not None:
            with tf.name_scope(self.ln_cross_attn.name):
                self.ln_cross_attn.build([None, None, self.hidden_size])
# 定义一个自定义的 Keras 层 TFGPT2MainLayer，用于实现 GPT-2 主层的功能
@keras_serializable
class TFGPT2MainLayer(keras.layers.Layer):
    # 配置类，指定为 GPT-2 的配置类 GPT2Config
    config_class = GPT2Config

    # 初始化方法，接受配置对象 config 和其他输入参数
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(*inputs, **kwargs)

        # 将配置对象保存到实例变量 self.config 中
        self.config = config
        # 是否输出注意力权重
        self.output_attentions = config.output_attentions
        # 是否输出隐藏状态
        self.output_hidden_states = config.output_hidden_states
        # 是否使用缓存
        self.use_cache = config.use_cache
        # 是否返回字典形式的输出
        self.return_dict = config.use_return_dict

        # 隐藏层的数量
        self.num_hidden_layers = config.n_layer
        # 嵌入向量的维度
        self.n_embd = config.n_embd
        # 位置编码的长度
        self.n_positions = config.n_positions
        # 初始化范围
        self.initializer_range = config.initializer_range

        # 词嵌入层，用于将输入的词索引转换为词向量
        self.wte = keras.layers.Embedding(
            input_dim=config.vocab_size,
            output_dim=config.hidden_size,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="wte",
        )
        # 位置嵌入层，用于将位置索引转换为位置向量
        self.wpe = keras.layers.Embedding(
            input_dim=config.n_positions,
            output_dim=config.n_embd,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="wpe",
        )
        # Dropout 层，用于在训练过程中随机置零部分输入向量，防止过拟合
        self.drop = keras.layers.Dropout(config.embd_pdrop)
        
        # 多头注意力层列表，使用 TFBlock 类创建，共 config.n_layer 个
        self.h = [TFBlock(config, scale=True, name=f"h_._{i}") for i in range(config.n_layer)]
        
        # 最后的 LayerNormalization 层，用于归一化隐藏层的输出
        self.ln_f = keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="ln_f")
        
        # 嵌入向量的维度，即隐藏层的维度
        self.embed_dim = config.hidden_size

    # 返回词嵌入层对象
    def get_input_embeddings(self):
        return self.wte

    # 设置新的输入词嵌入层
    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    # 未实现的方法，用于剪枝模型的注意力头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        raise NotImplementedError

    # 解包输入的装饰器，用于处理输入参数并调用模型
    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    # 构建方法，用于构造模型的各个组件的计算图
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回，避免重复构建
        if self.built:
            return
        # 设置标志位，表示模型已经构建
        self.built = True
        
        # 如果存在 self.wte 属性，则构建 wte 组件
        if getattr(self, "wte", None) is not None:
            # 在命名作用域内构建 wte 组件
            with tf.name_scope(self.wte.name):
                self.wte.build(None)
        
        # 如果存在 self.wpe 属性，则构建 wpe 组件
        if getattr(self, "wpe", None) is not None:
            # 在命名作用域内构建 wpe 组件
            with tf.name_scope(self.wpe.name):
                self.wpe.build(None)
        
        # 如果存在 self.ln_f 属性，则构建 ln_f 组件
        if getattr(self, "ln_f", None) is not None:
            # 在命名作用域内构建 ln_f 组件
            with tf.name_scope(self.ln_f.name):
                self.ln_f.build([None, None, self.embed_dim])
        
        # 如果存在 self.h 属性，则依次构建其中的每个 layer 组件
        if getattr(self, "h", None) is not None:
            for layer in self.h:
                # 在命名作用域内构建当前 layer 组件
                with tf.name_scope(layer.name):
                    layer.build(None)
# 定义一个名为 TFGPT2PreTrainedModel 的类，继承自 TFPreTrainedModel，用于处理权重初始化和预训练模型的下载和加载接口
class TFGPT2PreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定配置类为 GPT2Config
    config_class = GPT2Config
    # 基础模型的前缀为 "transformer"
    base_model_prefix = "transformer"
    
    # 在从 PyTorch 模型加载到 TensorFlow 模型时，忽略掉指定的层名中含有 'h.\d+.attn.bias' 或 'h.\d+.crossattention.bias' 的层
    # 这些层在加载过程中被视为授权的意外/丢失层
    _keys_to_ignore_on_load_unexpected = [r"h.\d+.attn.bias", r"h.\d+.crossattention.bias"]

    @property
    def input_signature(self):
        # 返回输入签名，指定了输入张量的规格
        # GPT-2 理论上支持 token_type_ids，但在实践中很少使用，而且其实现意味着传递 token_type_ids=0 会产生与 token_type_ids=None 不同的输出
        # 因此，默认情况下移除 token_type_ids 参数，即使通常应该包含它
        return {
            "input_ids": tf.TensorSpec((None, None), tf.int32, name="input_ids"),
            "attention_mask": tf.TensorSpec((None, None), tf.int32, name="attention_mask"),
        }


@dataclass
class TFGPT2DoubleHeadsModelOutput(ModelOutput):
    """
    Base class for outputs of models predicting if two sentences are consecutive or not.

    Args:
        logits (`tf.Tensor` of shape `(batch_size, num_choices, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        mc_logits (`tf.Tensor` of shape `(batch_size, num_choices)`):
            Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
        past_key_values (`List[tf.Tensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            List of `tf.Tensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_heads,
            sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
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

    # 定义 TFGPT2DoubleHeadsModelOutput 类，用于存储模型输出，包括语言建模头部的预测分数、多选分类头部的预测分数以及可选的额外信息如过去的键值、隐藏状态和注意力权重
    logits: tf.Tensor = None
    # 定义一个变量 mc_logits，类型为 tf.Tensor，默认为 None
    mc_logits: tf.Tensor = None
    # 定义一个变量 past_key_values，类型为 List[tf.Tensor] 或 None，默认为 None
    past_key_values: List[tf.Tensor] | None = None
    # 定义一个变量 hidden_states，类型为 Tuple[tf.Tensor] 或 None，默认为 None
    hidden_states: Tuple[tf.Tensor] | None = None
    # 定义一个变量 attentions，类型为 Tuple[tf.Tensor] 或 None，默认为 None
    attentions: Tuple[tf.Tensor] | None = None
# 定义 GPT2_START_DOCSTRING 为多行字符串，包含模型的继承关系、使用说明和参数说明
GPT2_START_DOCSTRING = r"""

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
        config ([`GPT2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义 GPT2_INPUTS_DOCSTRING 为空字符串
GPT2_INPUTS_DOCSTRING = r"""
"""

# 在类文档字符串中添加该类的说明并引用 GPT2_START_DOCSTRING
@add_start_docstrings(
    "The bare GPT2 Model transformer outputting raw hidden-states without any specific head on top.",
    GPT2_START_DOCSTRING,
)
# 定义 TFGPT2Model 类继承自 TFGPT2PreTrainedModel
class TFGPT2Model(TFGPT2PreTrainedModel):
    # 初始化函数，接受模型配置和输入参数
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类构造函数初始化模型配置和输入参数
        super().__init__(config, *inputs, **kwargs)
        # 使用TFGPT2MainLayer类创建transformer关键字参数
        self.transformer = TFGPT2MainLayer(config, name="transformer")

    # 使用装饰器函数unpack_inputs和add_start_docstrings_to_model_forward添加函数说明
    @unpack_inputs
    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    定义一个装饰器，用于为代码示例添加文档字符串，指定了文档检查点、输出类型和配置类

    def call(
        self,
        input_ids: TFModelInputType | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutputWithPastAndCrossAttentions, Tuple[tf.Tensor]]:
        r"""
        encoder_hidden_states  (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

        past_key_values (`Tuple[Tuple[tf.Tensor]]` of length `config.n_layers`)
            contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past` are used, the user can optionally input only the last `decoder_input_ids` (those that don't have
            their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*, defaults to `True`):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past`). Set to `False` during training, `True` during generation
        """

        outputs = self.transformer(
            input_ids=input_ids,  # 输入的token序列的ID
            past_key_values=past_key_values,  # 预先计算的注意力机制的键值对状态，用于加速解码
            attention_mask=attention_mask,  # 注意力掩码，避免对编码器输入的填充token进行注意力计算
            token_type_ids=token_type_ids,  # token类型ID，用于区分不同的句子或段落
            position_ids=position_ids,  # token在序列中的位置ID
            head_mask=head_mask,  # 头部掩码，控制哪些注意力头部被保留或屏蔽
            inputs_embeds=inputs_embeds,  # 输入的嵌入表示，用于提供预先计算的嵌入向量
            encoder_hidden_states=encoder_hidden_states,  # 编码器的隐藏状态序列，用于解码器的交叉注意力
            encoder_attention_mask=encoder_attention_mask,  # 编码器的注意力掩码，用于解码器的交叉注意力
            use_cache=use_cache,  # 是否使用缓存来加速解码
            output_attentions=output_attentions,  # 是否返回注意力权重
            output_hidden_states=output_hidden_states,  # 是否返回隐藏状态
            return_dict=return_dict,  # 是否以字典形式返回结果
            training=training,  # 是否在训练模式下运行
        )

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
"""
The GPT2 Model transformer with a language modeling head on top (linear layer with weights tied to the input
embeddings).
"""
# 声明一个 TF 模型类，继承自 TFGPT2PreTrainedModel 和 TFCausalLanguageModelingLoss
class TFGPT2LMHeadModel(TFGPT2PreTrainedModel, TFCausalLanguageModelingLoss):
    
    # 初始化方法，接收配置参数和其他输入
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 创建 GPT2 的主要层，即 transformer 层
        self.transformer = TFGPT2MainLayer(config, name="transformer")

    # 获取输出嵌入的方法，返回输入嵌入
    def get_output_embeddings(self):
        return self.get_input_embeddings()

    # 设置输出嵌入的方法，设置输入嵌入
    def set_output_embeddings(self, value):
        self.set_input_embeddings(value)

    # 为生成准备输入的方法，根据输入参数组织生成所需的输入数据
    def prepare_inputs_for_generation(self, inputs, past_key_values=None, use_cache=None, **kwargs):
        # 获取 token_type_ids，如果在 kwargs 中定义了，则获取最后一个 token 的 token_type_ids
        token_type_ids = kwargs.get("token_type_ids", None)
        
        # 如果 past_key_values 存在，则仅使用 inputs 的最后一个 token
        if past_key_values:
            inputs = tf.expand_dims(inputs[:, -1], -1)
            if token_type_ids is not None:
                token_type_ids = tf.expand_dims(token_type_ids[:, -1], -1)

        # 获取 position_ids、attention_mask 等参数
        position_ids = kwargs.get("position_ids", None)
        attention_mask = kwargs.get("attention_mask", None)

        # 如果 attention_mask 存在且 position_ids 不存在，则根据 attention_mask 计算 position_ids
        if attention_mask is not None and position_ids is None:
            position_ids = tf.math.cumsum(attention_mask, axis=-1, exclusive=True)
            if past_key_values:
                position_ids = tf.expand_dims(position_ids[:, -1], -1)

        # 返回生成模型所需的输入数据字典
        return {
            "input_ids": inputs,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "token_type_ids": token_type_ids,
        }

    # 调用方法，实现模型的前向传播
    @unpack_inputs
    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFCausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
        # 这里省略了函数体的部分
    # 定义神经网络层的构建方法，参数input_shape为输入形状，默认为None
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回，避免重复构建
        if self.built:
            return
        # 将标记设为已构建状态
        self.built = True
        # 检查是否存在transformer属性，并且不为None
        if getattr(self, "transformer", None) is not None:
            # 使用transformer的名称作为命名空间
            with tf.name_scope(self.transformer.name):
                # 调用transformer对象的build方法，传入None作为输入形状
                self.transformer.build(None)
"""
    通过在顶部添加多选分类头来扩展 GPT2 模型变换器，例如用于 RocStories/SWAG 任务。这两个头部都是线性层。语言建模头部将其权重绑定到输入嵌入，分类头部将输入分类令牌索引的输入序列。
""",
    GPT2_START_DOCSTRING,
)
class TFGPT2DoubleHeadsModel(TFGPT2PreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        config.num_labels = 1  # 设置分类数量为1
        self.transformer = TFGPT2MainLayer(config, name="transformer")  # 初始化 GPT2 主层
        self.multiple_choice_head = TFSequenceSummary(
            config, initializer_range=config.initializer_range, name="multiple_choice_head"  # 初始化多选头部
        )

    @unpack_inputs
    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)  # 添加模型前向传播的文档字符串
    @replace_return_docstrings(output_type=TFGPT2DoubleHeadsModelOutput, config_class=_CONFIG_FOR_DOC)  # 替换返回文档字符串
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 输入的模型 ID
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,  # 过去的键值对
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力遮罩
        token_type_ids: np.ndarray | tf.Tensor | None = None,  # 令牌类型 ID
        position_ids: np.ndarray | tf.Tensor | None = None,  # 位置 ID
        head_mask: np.ndarray | tf.Tensor | None = None,  # 头部遮罩
        inputs_embeds: np.ndarray | tf.Tensor | None = None,  # 输入的嵌入
        mc_token_ids: np.ndarray | tf.Tensor | None = None,  # 多选令牌 ID
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 输出注意力
        output_hidden_states: Optional[bool] = None,  # 输出隐藏状态
        return_dict: Optional[bool] = None,  # 返回字典
        training: Optional[bool] = False,  # 训练模式
    ):
        # 实现模型的前向传播逻辑，详细见 GPT2 输入文档字符串

    @property
    def input_signature(self):
        return {
            "input_ids": tf.TensorSpec((None, None, None), tf.int32, name="input_ids"),  # 输入的模型 ID 规范
            "attention_mask": tf.TensorSpec((None, None, None), tf.int32, name="attention_mask"),  # 注意力遮罩规范
            "mc_token_ids": tf.TensorSpec((None, None), tf.int32, name="mc_token_ids"),  # 多选令牌 ID 规范
        }

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)  # 构建 GPT2 主层
        if getattr(self, "multiple_choice_head", None) is not None:
            with tf.name_scope(self.multiple_choice_head.name):
                self.multiple_choice_head.build(None)  # 构建多选头部
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    GPT2_START_DOCSTRING,
    )
    # 定义一个名为 TFGPT2ForSequenceClassification 的类，继承自 TFGPT2PreTrainedModel 和 TFSequenceClassificationLoss
    class TFGPT2ForSequenceClassification(TFGPT2PreTrainedModel, TFSequenceClassificationLoss):
        def __init__(self, config, *inputs, **kwargs):
            # 调用父类的初始化方法
            super().__init__(config, *inputs, **kwargs)
            # 设置类属性 num_labels，表示分类的标签数量
            self.num_labels = config.num_labels
            # 创建一个名为 score 的全连接层 Dense，用于分类得分计算
            self.score = keras.layers.Dense(
                config.num_labels,
                kernel_initializer=get_initializer(config.initializer_range),
                name="score",
                use_bias=False,
            )
            # 创建一个 GPT2 主体层，用于序列转换
            self.transformer = TFGPT2MainLayer(config, name="transformer")
            # 存储配置信息
            self.config = config

        @unpack_inputs
        @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
        @add_code_sample_docstrings(
            checkpoint="microsoft/DialogRPT-updown",
            output_type=TFSequenceClassifierOutputWithPast,
            config_class=_CONFIG_FOR_DOC,
        )
        # 定义 call 方法，接收输入并进行模型前向传播
        def call(
            self,
            input_ids: TFModelInputType | None = None,
            past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
            attention_mask: np.ndarray | tf.Tensor | None = None,
            token_type_ids: np.ndarray | tf.Tensor | None = None,
            position_ids: np.ndarray | tf.Tensor | None = None,
            head_mask: np.ndarray | tf.Tensor | None = None,
            inputs_embeds: np.ndarray | tf.Tensor | None = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            labels: np.ndarray | tf.Tensor | None = None,
            training: Optional[bool] = False,
    ) -> Union[TFSequenceClassifierOutputWithPast, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the cross entropy classification loss. Indices should be in `[0, ...,
            config.vocab_size - 1]`.
        """
        # 调用Transformer模型处理输入数据，获取Transformer的输出结果
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 从Transformer的输出中获取隐藏状态（hidden_states）
        hidden_states = transformer_outputs[0]
        
        # 将隐藏状态通过分类器得到预测的logits
        logits = self.score(hidden_states)
        
        # 获取logits的形状信息
        logits_shape = shape_list(logits)
        
        # 初始化in_logits变量
        in_logits = None
        
        # 如果模型配置中没有定义pad_token_id
        if self.config.pad_token_id is None:
            # 将序列长度设置为-1
            sequence_lengths = -1
        else:
            # 如果输入中包含input_ids
            if input_ids is not None:
                # 计算每个序列的有效长度
                sequence_lengths = (
                    tf.argmax(tf.cast(tf.math.equal(input_ids, self.config.pad_token_id), input_ids.dtype), axis=-1)
                    - 1
                )
                # 将小于0的长度替换为默认序列长度-1
                sequence_lengths = tf.where(sequence_lengths >= 0, sequence_lengths, input_ids.shape[-1] - 1)
                # 根据有效长度从logits中抽取相应的部分
                in_logits = tf.gather(logits, sequence_lengths, batch_dims=1, axis=1)
            else:
                # 如果没有输入input_ids，则将序列长度设置为-1
                sequence_lengths = -1
                # 记录警告日志，说明在使用inputs_embeds时无法检测到填充标记
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )
        
        # 初始化损失值
        loss = None
        
        # 如果提供了标签数据
        if labels is not None:
            # 断言条件，确保模型能处理批次大小大于1的情况，或者至少有一个填充标记被定义
            assert (
                self.config.pad_token_id is not None or logits_shape[0] == 1
            ), "Cannot handle batch sizes > 1 if no padding token is defined."

            # 如果sequence_lengths不是Tensor，说明在计算中已经从logits中获取了相应的部分
            if not tf.is_tensor(sequence_lengths):
                in_logits = logits[0 : logits_shape[0], sequence_lengths]

            # 计算交叉熵损失
            loss = self.hf_compute_loss(tf.reshape(labels, [-1]), tf.reshape(in_logits, [-1, self.num_labels]))
        
        # 如果没有in_logits，则使用原始logits作为池化后的logits
        pooled_logits = in_logits if in_logits is not None else logits
        
        # 如果return_dict为False，则返回一个元组
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        # 如果return_dict为True，则返回TFSequenceClassifierOutputWithPast对象
        return TFSequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
    # 定义一个方法 `build`，用于构建模型，可以接受输入形状参数 `input_shape`
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回，不再重复构建
        if self.built:
            return
        # 设置标志位，表示模型已经构建
        self.built = True
        # 如果模型具有 `score` 属性且不为 `None`
        if getattr(self, "score", None) is not None:
            # 使用 `tf.name_scope` 建立命名空间，命名为 `self.score.name`
            with tf.name_scope(self.score.name):
                # 调用 `self.score` 的 `build` 方法，传入形状参数 `[None, None, self.config.n_embd]`
                self.score.build([None, None, self.config.n_embd])
        # 如果模型具有 `transformer` 属性且不为 `None`
        if getattr(self, "transformer", None) is not None:
            # 使用 `tf.name_scope` 建立命名空间，命名为 `self.transformer.name`
            with tf.name_scope(self.transformer.name):
                # 调用 `self.transformer` 的 `build` 方法，传入 `None` 作为参数
                self.transformer.build(None)
```