# `.\models\ctrl\modeling_tf_ctrl.py`

```
# coding=utf-8
# Copyright 2018 Salesforce and HuggingFace Inc. team.
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
""" TF 2.0 CTRL model."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...modeling_tf_outputs import TFBaseModelOutputWithPast, TFCausalLMOutputWithPast, TFSequenceClassifierOutput
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    get_initializer,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_ctrl import CTRLConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "Salesforce/ctrl"
_CONFIG_FOR_DOC = "CTRLConfig"

TF_CTRL_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Salesforce/ctrl"
    # See all CTRL models at https://huggingface.co/models?filter=Salesforce/ctrl
]


def angle_defn(pos, i, d_model_size):
    # Calculate the rates of angles for positional encoding
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model_size)
    return pos * angle_rates


def positional_encoding(position, d_model_size):
    # Create positional encodings using sinusoidal patterns
    angle_rads = angle_defn(np.arange(position)[:, np.newaxis], np.arange(d_model_size)[np.newaxis, :], d_model_size)

    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    pos_encoding = tf.convert_to_tensor(np.concatenate([sines, cosines], axis=-1))

    return pos_encoding


def scaled_dot_product_attention(q, k, v, mask, attention_mask=None, head_mask=None):
    # Calculate scaled dot-product attention
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(shape_list(k)[-1], dtype=matmul_qk.dtype)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += tf.cast(mask * -1e4, dtype=scaled_attention_logits.dtype)

    if attention_mask is not None:
        # Apply the attention mask
        attention_mask = tf.cast(attention_mask, dtype=scaled_attention_logits.dtype)
        scaled_attention_logits = scaled_attention_logits + attention_mask

    attention_weights = stable_softmax(scaled_attention_logits, axis=-1)

    # Mask heads if we want to
    # 如果给定了头部掩码（head_mask），则将注意力权重（attention_weights）与头部掩码逐元素相乘
    if head_mask is not None:
        attention_weights = attention_weights * head_mask
    
    # 将注意力权重（已经经过处理的，如果有头部掩码的话）与值（v）相乘，得到注意力机制的输出
    output = tf.matmul(attention_weights, v)
    
    # 返回注意力机制的输出和注意力权重
    return output, attention_weights
class TFMultiHeadAttention(keras.layers.Layer):
    # 初始化多头注意力层
    def __init__(self, d_model_size, num_heads, output_attentions=False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model_size = d_model_size
        self.output_attentions = output_attentions

        # 计算每个头部的深度
        self.depth = int(d_model_size / self.num_heads)

        # 定义权重矩阵，用于查询（q）、键（k）、值（v）的线性映射
        self.Wq = keras.layers.Dense(d_model_size, name="Wq")
        self.Wk = keras.layers.Dense(d_model_size, name="Wk")
        self.Wv = keras.layers.Dense(d_model_size, name="Wv")

        # 最终输出的全连接层
        self.dense = keras.layers.Dense(d_model_size, name="dense")

    # 将输入张量分割成多个头部
    def split_into_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    # 多头注意力层的调用方法
    def call(self, v, k, q, mask, layer_past, attention_mask, head_mask, use_cache, output_attentions, training=False):
        batch_size = shape_list(q)[0]

        # 线性映射到查询、键、值空间
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        # 将查询、键、值分割成多个头部
        q = self.split_into_heads(q, batch_size)
        k = self.split_into_heads(k, batch_size)
        v = self.split_into_heads(v, batch_size)

        # 如果存在过去的键值对，将当前的键值对与过去的连接起来
        if layer_past is not None:
            past_key, past_value = tf.unstack(layer_past, axis=0)
            k = tf.concat((past_key, k), axis=-2)
            v = tf.concat((past_value, v), axis=-2)

        # 如果使用缓存，存储当前的键值对
        if use_cache:
            present = tf.stack((k, v), axis=0)
        else:
            present = (None,)

        # 进行缩放点积注意力计算
        output = scaled_dot_product_attention(q, k, v, mask, attention_mask, head_mask)
        scaled_attention = tf.transpose(output[0], perm=[0, 2, 1, 3])
        attn = output[1]

        # 将多头注意力的输出重塑为原始形状
        original_size_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model_size))

        # 通过全连接层处理重塑后的注意力表示
        output = self.dense(original_size_attention)
        outputs = (output, present)

        # 如果需要输出注意力权重，添加到输出中
        if output_attentions:
            outputs = outputs + (attn,)

        return outputs

    # 构建多头注意力层
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True

        # 构建权重矩阵
        if getattr(self, "Wq", None) is not None:
            with tf.name_scope(self.Wq.name):
                self.Wq.build([None, None, self.d_model_size])
        if getattr(self, "Wk", None) is not None:
            with tf.name_scope(self.Wk.name):
                self.Wk.build([None, None, self.d_model_size])
        if getattr(self, "Wv", None) is not None:
            with tf.name_scope(self.Wv.name):
                self.Wv.build([None, None, self.d_model_size])
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.d_model_size])
    # 初始化方法，设置模型的大小和隐藏层大小
    def __init__(self, d_model_size, dff, **kwargs):
        # 调用父类初始化方法
        super().__init__(**kwargs)

        # 创建第一个全连接层，使用ReLU激活函数，命名为"0"
        self.dense_0 = keras.layers.Dense(dff, activation="relu", name="0")
        
        # 创建第二个全连接层，输出维度为d_model_size，命名为"2"
        self.dense_2 = keras.layers.Dense(d_model_size, name="2")
        
        # 设置模型大小和隐藏层大小
        self.d_model_size = d_model_size
        self.dff = dff

    # 模型调用方法，接受输入并返回第二个全连接层的输出
    def call(self, inputs, trainable=False):
        # 第一个全连接层的输出
        dense_0_output = self.dense_0(inputs)
        
        # 第二个全连接层的输出
        dense_2_output = self.dense_2(dense_0_output)

        # 返回第二个全连接层的输出作为模型的输出
        return dense_2_output

    # 构建方法，用于构建模型结构
    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回
        if self.built:
            return
        
        # 标记模型已经构建
        self.built = True
        
        # 如果dense_0层存在，则构建dense_0层
        if getattr(self, "dense_0", None) is not None:
            with tf.name_scope(self.dense_0.name):
                self.dense_0.build([None, None, self.d_model_size])
        
        # 如果dense_2层存在，则构建dense_2层
        if getattr(self, "dense_2", None) is not None:
            with tf.name_scope(self.dense_2.name):
                self.dense_2.build([None, None, self.dff])
class TFEncoderLayer(keras.layers.Layer):
    # 定义 Transformer 编码器层的 Keras 自定义层

    def __init__(
        self, d_model_size, num_heads, dff, rate=0.1, layer_norm_epsilon=1e-6, output_attentions=False, **kwargs
    ):
        super().__init__(**kwargs)

        self.output_attentions = output_attentions

        # 创建多头注意力机制层，用于编码器层
        self.multi_head_attention = TFMultiHeadAttention(
            d_model_size, num_heads, output_attentions=self.output_attentions, name="multi_head_attention"
        )
        
        # 创建点式前馈网络层，用于编码器层
        self.ffn = TFPointWiseFeedForwardLayer(d_model_size, dff, name="ffn")

        # 创建第一个层归一化层
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=layer_norm_epsilon, name="layernorm1")
        
        # 创建第二个层归一化层
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=layer_norm_epsilon, name="layernorm2")

        # 创建第一个 dropout 层
        self.dropout1 = keras.layers.Dropout(rate)
        
        # 创建第二个 dropout 层
        self.dropout2 = keras.layers.Dropout(rate)
        
        # 保存模型的尺寸
        self.d_model_size = d_model_size

    def call(self, x, mask, layer_past, attention_mask, head_mask, use_cache, output_attentions, training=False):
        # 对输入进行第一个归一化
        normed = self.layernorm1(x)
        
        # 使用多头注意力机制进行计算
        attn_outputs = self.multi_head_attention(
            normed,
            normed,
            normed,
            mask,
            layer_past,
            attention_mask,
            head_mask,
            use_cache,
            output_attentions,
            training=training,
        )
        
        # 从多头注意力机制的输出中获取注意力机制的结果
        attn_output = attn_outputs[0]
        
        # 对注意力机制的输出应用第一个 dropout 层
        attn_output = self.dropout1(attn_output, training=training)
        
        # 计算第一步的输出
        out1 = x + attn_output

        # 对第一步的输出进行第二个归一化
        out2 = self.layernorm2(out1)
        
        # 使用点式前馈网络进行计算
        ffn_output = self.ffn(out2)
        
        # 对点式前馈网络的输出应用第二个 dropout 层
        ffn_output = self.dropout2(ffn_output, training=training)
        
        # 计算第二步的输出
        out2 = out1 + ffn_output

        # 将所有输出整合到一个元组中返回
        outputs = (out2,) + attn_outputs[1:]
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        
        # 构建多头注意力机制层
        if getattr(self, "multi_head_attention", None) is not None:
            with tf.name_scope(self.multi_head_attention.name):
                self.multi_head_attention.build(None)
        
        # 构建点式前馈网络层
        if getattr(self, "ffn", None) is not None:
            with tf.name_scope(self.ffn.name):
                self.ffn.build(None)
        
        # 构建第一个归一化层
        if getattr(self, "layernorm1", None) is not None:
            with tf.name_scope(self.layernorm1.name):
                self.layernorm1.build([None, None, self.d_model_size])
        
        # 构建第二个归一化层
        if getattr(self, "layernorm2", None) is not None:
            with tf.name_scope(self.layernorm2.name):
                self.layernorm2.build([None, None, self.d_model_size])


@keras_serializable
class TFCTRLMainLayer(keras.layers.Layer):
    # 基于 Keras 的 TFCTRL 主要层，用于 CTRL 模型

    # 配置类为 CTRLConfig
    config_class = CTRLConfig
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.config = config  # 设置模型的配置参数
        self.output_hidden_states = config.output_hidden_states  # 是否输出隐藏状态的配置
        self.output_attentions = config.output_attentions  # 是否输出注意力权重的配置
        self.use_cache = config.use_cache  # 是否使用缓存的配置
        self.return_dict = config.use_return_dict  # 是否返回字典的配置

        self.d_model_size = config.n_embd  # 获取模型的嵌入维度大小
        self.num_layers = config.n_layer  # 获取模型的层数

        self.pos_encoding = positional_encoding(config.n_positions, self.d_model_size)  # 计算位置编码

        self.w = keras.layers.Embedding(
            input_dim=config.vocab_size,  # 输入词汇表大小
            output_dim=config.n_embd,  # 输出嵌入维度
            embeddings_initializer=get_initializer(config.initializer_range),  # 获取初始化器
            name="w",  # 设置层名称
        )

        self.dropout = keras.layers.Dropout(config.embd_pdrop)  # 设置dropout层
        self.h = [
            TFEncoderLayer(
                config.n_embd,  # 嵌入维度大小
                config.n_head,  # 头数
                config.dff,  # 前馈网络的大小
                config.resid_pdrop,  # 残差dropout率
                config.layer_norm_epsilon,  # 层归一化的epsilon值
                self.output_attentions,  # 是否输出注意力权重
                name=f"h_._{i}",  # 设置层名称
            )
            for i in range(config.n_layer)  # 循环创建编码层
        ]
        self.layernorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="layernorm")  # 设置层归一化操作

    def get_input_embeddings(self):
        return self.w  # 返回输入嵌入层

    def set_input_embeddings(self, new_embeddings):
        self.w = new_embeddings  # 设置新的输入嵌入层

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        raise NotImplementedError  # 抛出未实现错误

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
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ):
        # 模型的前向传播方法，使用unpack_inputs装饰器解压输入参数
        ...

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True  # 标记模型已构建
        if getattr(self, "w", None) is not None:
            with tf.name_scope(self.w.name):
                self.w.build(None)  # 构建输入嵌入层
        if getattr(self, "layernorm", None) is not None:
            with tf.name_scope(self.layernorm.name):
                self.layernorm.build([None, None, self.config.n_embd])  # 构建层归一化层
        if getattr(self, "h", None) is not None:
            for layer in self.h:
                with tf.name_scope(layer.name):
                    layer.build(None)  # 构建每个编码层
# TFCTRLPreTrainedModel 类的定义，继承自 TFPreTrainedModel，用于处理权重初始化以及下载和加载预训练模型的简单接口。
class TFCTRLPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定配置类为 CTRLConfig
    config_class = CTRLConfig
    # 基础模型前缀为 "transformer"
    base_model_prefix = "transformer"


# 下面是对 CTRLModel 的文档字符串和注释
CTRL_START_DOCSTRING = r"""

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
        config ([`CTRLConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 空白，等待后续完善输入文档字符串的部分
CTRL_INPUTS_DOCSTRING = r"""
"""

# 添加文档字符串说明到 TFCTRLModel 类，描述其作为 CTRL 模型的裸变压器输出原始隐藏状态的特性。
@add_start_docstrings(
    "The bare CTRL Model transformer outputting raw hidden-states without any specific head on top.",
    CTRL_START_DOCSTRING,
)
class TFCTRLModel(TFCTRLPreTrainedModel):
    # 初始化方法，接受配置参数和输入，调用父类的初始化方法
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 创建一个 TFCTRLMainLayer 类的实例作为 self.transformer，并命名为 "transformer"
        self.transformer = TFCTRLMainLayer(config, name="transformer")

    # 装饰器：解包输入参数，并添加文档字符串到模型前向传播方法
    @unpack_inputs
    @add_start_docstrings_to_model_forward(CTRL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        # 添加样例代码的文档字符串，指定检查点、输出类型、配置类
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义模型的前向传播方法
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
        training: Optional[bool] = False,
    ) -> Union[Tuple, TFBaseModelOutputWithPast]:
        # 调用 self.transformer 的前向传播方法，传入所有指定的参数
        outputs = self.transformer(
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
        # 返回前向传播的输出结果
        return outputs

    # 构建方法，用于构建模型的层次结构
    def build(self, input_shape=None):
        # 如果模型已经构建，直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果 self.transformer 存在，则在其命名空间下构建模型
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                # 调用 self.transformer 的构建方法，传入 None 参数
                self.transformer.build(None)
class TFCTRLBiasLayer(keras.layers.Layer):
    """
    Bias as a layer. It is used for serialization purposes: `keras.Model.save_weights` stores on a per-layer basis,
    so all weights have to be registered in a layer.
    """

    def __init__(self, shape, initializer, trainable, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.shape = shape  # 初始化bias的形状
        self.initializer = initializer  # 初始化bias的方式
        self.trainable = trainable  # 是否可以训练

    def build(self, input_shape):
        self.bias = self.add_weight(
            name="bias", shape=self.shape, initializer=self.initializer, trainable=self.trainable
        )  # 添加bias作为权重到层中
        super().build(input_shape)

    def call(self, x):
        return x + self.bias  # 在输入张量x上添加bias



@add_start_docstrings(
    """
    The CTRL Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    CTRL_START_DOCSTRING,
)
class TFCTRLLMHeadModel(TFCTRLPreTrainedModel, TFCausalLanguageModelingLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFCTRLMainLayer(config, name="transformer")  # 初始化transformer层
        self.bias_layer = TFCTRLBiasLayer(
            name="lm_head", shape=[1, config.vocab_size], initializer="zeros", trainable=True
        )  # 初始化bias层，用于LM头部

    def get_output_embeddings(self):
        return self.get_input_embeddings()  # 获取输出的嵌入

    def set_output_embeddings(self, value):
        self.set_input_embeddings(value)  # 设置输出的嵌入

    def get_bias(self):
        return {"lm_head.bias": self.bias_layer.bias}  # 获取当前bias的值

    def set_bias(self, value):
        # Replaces the existing layers containing bias for correct (de)serialization.
        vocab_size = value["lm_head.bias"].shape[-1]  # 获取vocab_size
        self.bias_layer = TFCTRLBiasLayer(
            name="final_logits_bias", shape=[1, vocab_size], initializer="zeros", trainable=True
        )  # 初始化一个新的bias层
        self.bias_layer.build(None)  # 构建新的bias层
        self.bias_layer.bias.assign(value["lm_head.bias"])  # 分配给新bias层的值
    def prepare_inputs_for_generation(self, inputs, past_key_values=None, use_cache=None, **kwargs):
        # 从 kwargs 中获取 token_type_ids，默认为 None
        token_type_ids = kwargs.get("token_type_ids", None)
        # 如果 past_key_values 不为 None，则只使用 inputs 的最后一个 token
        if past_key_values:
            # 将 inputs 的最后一个 token 扩展为单独的维度
            inputs = tf.expand_dims(inputs[:, -1], -1)
            # 如果 token_type_ids 不为 None，则也将其最后一个 token 扩展为单独的维度
            if token_type_ids is not None:
                token_type_ids = tf.expand_dims(token_type_ids[:, -1], -1)

        # 从 kwargs 中获取 position_ids、attention_mask，默认为 None
        position_ids = kwargs.get("position_ids", None)
        attention_mask = kwargs.get("attention_mask", None)

        # 如果 attention_mask 不为 None 而 position_ids 为 None，则根据 attention_mask 计算 position_ids
        if attention_mask is not None and position_ids is None:
            position_ids = tf.math.cumsum(attention_mask, axis=-1, exclusive=True)
            # 如果 past_key_values 不为 None，则将 position_ids 的最后一个 token 扩展为单独的维度
            if past_key_values:
                position_ids = tf.expand_dims(position_ids[:, -1], -1)

        # 返回一个包含准备好的输入的字典
        return {
            "input_ids": inputs,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "token_type_ids": token_type_ids,
        }

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CTRL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFCausalLMOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义 call 方法，包含多个参数用于模型推断和训练
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
    ) -> Union[Tuple, TFCausalLMOutputWithPast]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the cross entropy classification loss. Indices should be in `[0, ...,
            config.vocab_size - 1]`.
        """
        # 调用 Transformer 模型进行前向传播，获取变换器的输出
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
        # 从变换器的输出中获取隐藏状态
        hidden_states = transformer_outputs[0]
        # 计算逻辑回归层的输出，使用权重转置
        logits = tf.matmul(hidden_states, self.transformer.w.weights, transpose_b=True)
        # 对逻辑回归输出应用偏置层
        logits = self.bias_layer(logits)

        loss = None
        if labels is not None:
            # 将标签向左移动一个位置，并且截取最后一个逻辑回归标记
            shifted_logits = logits[:, :-1]
            labels = labels[:, 1:]
            # 计算损失函数，使用标签和移动后的逻辑回归输出
            loss = self.hf_compute_loss(labels, shifted_logits)

        if not return_dict:
            # 如果不返回字典，则输出元组形式
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回带有过去键值的 TFCausalLMOutputWithPast 对象，包括损失、逻辑回归输出和变换器的中间状态
        return TFCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        # 设置构建状态为已完成
        self.built = True
        if getattr(self, "transformer", None) is not None:
            # 构建变换器模型
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        if getattr(self, "bias_layer", None) is not None:
            # 构建偏置层
            with tf.name_scope(self.bias_layer.name):
                self.bias_layer.build(None)
@add_start_docstrings(
    """
    The CTRL Model transformer with a sequence classification head on top (linear layer).

    [`TFCTRLForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-1, GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    CTRL_START_DOCSTRING,
)
class TFCTRLForSequenceClassification(TFCTRLPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.classifier = keras.layers.Dense(
            config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="classifier",
            use_bias=False,
        )
        self.transformer = TFCTRLMainLayer(config, name="transformer")
        self.config = config

    def get_output_embeddings(self):
        # Remove after transformers v4.32. Fix this model's `test_model_common_attributes` test too.
        logger.warning(
            "Sequence classification models do not have output embeddings. `.get_output_embeddings` will be removed "
            "in transformers v4.32."
        )
        # 返回当前模型的权重矩阵 w 作为输出嵌入
        return self.transformer.w

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CTRL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSequenceClassifierOutput,
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
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ):
        # 调用模型的前向传播方法，根据输入参数计算模型输出
        # unpack_inputs 解包输入参数，以便使用它们进行计算
        # add_start_docstrings_to_model_forward 添加模型前向传播文档字符串
        # add_code_sample_docstrings 添加代码示例的文档字符串
        pass
    ) -> Union[Tuple, TFSequenceClassifierOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the cross entropy classification loss. Indices should be in `[0, ...,
            config.vocab_size - 1]`.
        """

        # 使用transformer处理输入，返回transformer的输出
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

        # 从transformer的输出中获取隐藏状态
        hidden_states = transformer_outputs[0]
        
        # 使用分类器获取logits
        logits = self.classifier(hidden_states)
        in_logits = None
        
        # 如果没有定义pad_token_id，则将sequence_lengths设为-1
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            # 如果有输入input_ids，则计算每个样本的序列长度
            if input_ids is not None:
                # 计算序列中最后一个非pad_token_id的位置
                sequence_lengths = (
                    tf.argmax(tf.cast(tf.math.equal(input_ids, self.config.pad_token_id), input_ids.dtype), axis=-1)
                    - 1
                )
                # 如果序列长度大于等于0，则保留该长度；否则使用默认的序列长度
                sequence_lengths = tf.where(sequence_lengths >= 0, sequence_lengths, input_ids.shape[-1] - 1)
                # 从logits中提取对应序列长度位置的值
                in_logits = tf.gather(logits, sequence_lengths, batch_dims=1, axis=1)
            else:
                # 如果没有input_ids，则警告并设定sequence_lengths为-1
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )
        
        loss = None

        # 如果提供了labels，则计算损失
        if labels is not None:
            # 根据输入类型获取batch_size和sequence_length
            if input_ids is not None:
                batch_size, sequence_length = shape_list(input_ids)[:2]
            else:
                batch_size, sequence_length = shape_list(inputs_embeds)[:2]
            # 如果未定义pad_token_id且batch_size不等于1，则引发错误
            if self.config.pad_token_id is None and batch_size != 1:
                raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")

            # 如果sequence_lengths不是tensor，则从logits中提取对应的值
            if not tf.is_tensor(sequence_lengths):
                in_logits = logits[0:batch_size, sequence_lengths]

            # 计算损失
            loss = self.hf_compute_loss(tf.reshape(labels, [-1, 1]), tf.reshape(in_logits, [-1, self.num_labels]))

        # 如果in_logits不为None，则使用它作为pooled_logits；否则使用logits
        pooled_logits = in_logits if in_logits is not None else logits

        # 如果不返回dict，则返回输出元组
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果返回dict，则返回TFSequenceClassifierOutput对象
        return TFSequenceClassifierOutput(
            loss=loss,
            logits=pooled_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
    # 定义 build 方法，用于构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建完成，则直接返回
        if self.built:
            return
        # 将模型标记为已构建
        self.built = True
        
        # 如果模型具有分类器属性，执行以下代码块
        if getattr(self, "classifier", None) is not None:
            # 使用分类器的名称空间来构建分类器模型
            with tf.name_scope(self.classifier.name):
                # 调用分类器对象的 build 方法来构建模型，输入形状为 [None, None, self.config.n_embd]
                self.classifier.build([None, None, self.config.n_embd])
        
        # 如果模型具有 transformer 属性，执行以下代码块
        if getattr(self, "transformer", None) is not None:
            # 使用 transformer 的名称空间来构建 transformer 模型
            with tf.name_scope(self.transformer.name):
                # 调用 transformer 对象的 build 方法来构建模型，输入形状为 None（即没有明确的输入形状要求）
                self.transformer.build(None)
```