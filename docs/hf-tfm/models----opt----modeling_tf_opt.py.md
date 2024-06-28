# `.\models\opt\modeling_tf_opt.py`

```
# coding=utf-8
# Copyright 2022 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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
""" TF 2.0 OPT model."""

# Future annotations are imported from the __future__ module to ensure compatibility with future versions of Python.
from __future__ import annotations

# Necessary imports from standard library and third-party packages
from typing import Optional, Tuple, Union

import numpy as np  # NumPy library for numerical operations
import tensorflow as tf  # TensorFlow library for deep learning

# Importing specific functions and classes from sibling modules
from ...activations_tf import get_tf_activation  # Function to get TensorFlow activation functions
from ...modeling_tf_outputs import TFBaseModelOutputWithPast, TFCausalLMOutputWithPast  # Output classes for TF models

# Public API imports from sibling modules
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    TFSharedEmbeddings,
    keras,
    keras_serializable,
    unpack_inputs,
)
# Utilities for TensorFlow operations and models
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
# Various utility functions and decorators
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# Importing OPT model configuration
from .configuration_opt import OPTConfig  # Configuration class specific to OPT model


logger = logging.get_logger(__name__)  # Logger instance for logging messages


_CHECKPOINT_FOR_DOC = "facebook/opt-350m"  # Pretrained model checkpoint identifier
_CONFIG_FOR_DOC = "OPTConfig"  # Configuration class identifier for documentation purposes

# Base model docstring
_EXPECTED_OUTPUT_SHAPE = [1, 8, 1024]  # Expected output shape for the base model

# Causal LM output example
_CAUSAL_LM_EXPECTED_OUTPUT = (
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious. I'm just a little bit of a weirdo."
)

LARGE_NEGATIVE = -1e8  # Constant representing a large negative value


def _make_causal_mask(input_ids_shape: tf.TensorShape, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.

    Args:
        input_ids_shape (tf.TensorShape): Shape of input tensor representing input ids.
        past_key_values_length (int): Length of past key values for attention mechanism.

    Returns:
        tf.Tensor: Causal mask tensor for bi-directional self-attention.
    """
    bsz = input_ids_shape[0]  # Batch size extracted from input_ids_shape
    tgt_len = input_ids_shape[1]  # Target sequence length extracted from input_ids_shape

    # Initialize a mask with large negative values
    mask = tf.fill((tgt_len, tgt_len), tf.cast(LARGE_NEGATIVE, tf.float32))
    # Apply upper triangular part of the mask
    mask = tf.linalg.band_part(mask, 0, -1) - tf.linalg.band_part(mask, 0, 0)

    # Concatenate zeros for past key values if present
    if past_key_values_length > 0:
        mask = tf.concat([tf.zeros((tgt_len, past_key_values_length)), mask], axis=-1)

    return tf.tile(mask[None, None, :, :], (bsz, 1, 1, 1))


# Function copied from BART model implementation to expand attention mask
def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.

    Args:
        mask (tf.Tensor): Tensor representing attention mask.
        tgt_len (Optional[int]): Target sequence length (default: None).

    Returns:
        tf.Tensor: Expanded attention mask tensor.
    """
    src_len = shape_list(mask)[1]  # Source sequence length extracted from mask tensor
    tgt_len = tgt_len if tgt_len is not None else src_len  # Use provided tgt_len or src_len if None
    one_cst = tf.constant(1.0)  # Constant tensor with value 1.0
    mask = tf.cast(mask, dtype=one_cst.dtype)  # Cast mask tensor to the same dtype as one_cst
    # Expand the mask tensor
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))

    return expanded_mask
    # 返回一个数值，计算方式为 (one_cst - expanded_mask) * LARGE_NEGATIVE
    return (one_cst - expanded_mask) * LARGE_NEGATIVE
class TFOPTLearnedPositionalEmbedding(keras.layers.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        # 设置偏移量为2，以便在指定padding_idx时，将embedding ids偏移2，并相应调整num_embeddings
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim, **kwargs)

    def call(self, attention_mask, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        attention_mask = tf.cast(attention_mask, tf.int64)

        # create positions depending on attention_mask
        # 根据attention_mask创建位置张量
        positions = tf.math.cumsum(attention_mask, axis=1) * attention_mask - 1

        # cut positions if `past_key_values_length` is > 0
        # 如果past_key_values_length > 0，则截取positions张量的后部分
        positions = positions[:, past_key_values_length:]

        return super().call(positions + self.offset)


# Copied from transformers.models.bart.modeling_tf_bart.TFBartAttention with Bart->OPT
class TFOPTAttention(keras.layers.Layer):
    """Multi-headed attention from "Attention Is All You Need"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = keras.layers.Dropout(dropout)
        self.head_dim = embed_dim // num_heads
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        # Linear projections for keys, queries, values, and output
        # 用于键（k_proj）、查询（q_proj）、数值（v_proj）、输出（out_proj）的线性投影层
        self.k_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="k_proj")
        self.q_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="q_proj")
        self.v_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="v_proj")
        self.out_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="out_proj")

    def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
        # Reshape tensor into [bsz, num_heads, seq_len, head_dim] format
        # 将张量重塑为[bsz, num_heads, seq_len, head_dim]格式
        return tf.transpose(tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))

    def call(
        self,
        hidden_states: tf.Tensor,
        key_value_states: tf.Tensor | None = None,
        past_key_value: Tuple[Tuple[tf.Tensor]] | None = None,
        attention_mask: tf.Tensor | None = None,
        layer_head_mask: tf.Tensor | None = None,
        training: Optional[bool] = False,
        **kwargs
    ):
    # 定义一个方法用于构建网络模型，可以接受输入形状参数
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回，避免重复构建
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        
        # 如果存在 k_proj 属性，则构建 k_proj 层
        if getattr(self, "k_proj", None) is not None:
            # 在命名作用域下构建 k_proj 层，并指定输入形状为 [None, None, self.embed_dim]
            with tf.name_scope(self.k_proj.name):
                self.k_proj.build([None, None, self.embed_dim])
        
        # 如果存在 q_proj 属性，则构建 q_proj 层
        if getattr(self, "q_proj", None) is not None:
            # 在命名作用域下构建 q_proj 层，并指定输入形状为 [None, None, self.embed_dim]
            with tf.name_scope(self.q_proj.name):
                self.q_proj.build([None, None, self.embed_dim])
        
        # 如果存在 v_proj 属性，则构建 v_proj 层
        if getattr(self, "v_proj", None) is not None:
            # 在命名作用域下构建 v_proj 层，并指定输入形状为 [None, None, self.embed_dim]
            with tf.name_scope(self.v_proj.name):
                self.v_proj.build([None, None, self.embed_dim])
        
        # 如果存在 out_proj 属性，则构建 out_proj 层
        if getattr(self, "out_proj", None) is not None:
            # 在命名作用域下构建 out_proj 层，并指定输入形状为 [None, None, self.embed_dim]
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.embed_dim])
# 定义 TFOPTDecoderLayer 类，继承自 keras.layers.Layer
class TFOPTDecoderLayer(keras.layers.Layer):
    # 初始化方法，接受一个 config 参数和其他关键字参数
    def __init__(self, config: OPTConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        
        # 从 config 中获取是否在层归一化之前执行操作的标志
        self.do_layer_norm_before = config.do_layer_norm_before
        # 设置嵌入维度为隐藏大小
        self.embed_dim = config.hidden_size
        
        # 创建自注意力机制的实例 self_attn，传入嵌入维度、注意力头数、注意力层的 dropout、名称和是否为解码器标志
        self.self_attn = TFOPTAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            name="self_attn",
            is_decoder=True,
        )
        
        # 创建 dropout 层，使用给定的 dropout 率
        self.dropout = keras.layers.Dropout(config.dropout)
        
        # 获取激活函数并赋值给 activation_fn
        self.activation_fn = get_tf_activation(config.activation_function)

        # 创建自注意力层后的层归一化层 self_attn_layer_norm，使用给定的 epsilon 值和名称
        self.self_attn_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        
        # 创建第一个全连接层 fc1，使用给定的维度 config.ffn_dim 和名称
        self.fc1 = keras.layers.Dense(config.ffn_dim, name="fc1")
        
        # 创建第二个全连接层 fc2，使用前一个全连接层输出的维度作为输入维度，输出维度为嵌入维度，设置名称为 fc2
        self.fc2 = keras.layers.Dense(self.embed_dim, name="fc2")
        
        # 创建最终的层归一化层 final_layer_norm，使用给定的 epsilon 值和名称
        self.final_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        
        # 将 config 参数存储在对象中
        self.config = config

    # 定义 call 方法，实现层的正向传播
    def call(
        self,
        hidden_states: tf.Tensor,  # 输入的隐藏状态张量
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力掩码，可以是 NumPy 数组、Tensor 或 None
        layer_head_mask: tf.Tensor | None = None,  # 层头掩码，可以是 Tensor 或 None
        past_key_value: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,  # 过去的键值对，可选的元组类型
        training: Optional[bool] = False,  # 训练模式标志，默认为 False
        output_attentions: Optional[bool] = False,  # 输出注意力权重的标志，默认为 False
        use_cache: Optional[bool] = False,  # 使用缓存的标志，默认为 False

= None,
        training: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        """
        Args:
            hidden_states (`tf.Tensor`): 输入到层的张量，形状为 `(batch, seq_len, embed_dim)`
            attention_mask (`tf.Tensor`, *可选*): 注意力掩码，形状为 `(batch, 1, tgt_len, src_len)`，
                其中填充元素由非常大的负值表示。
            layer_head_mask (`tf.Tensor`, *可选*): 给定层中注意力头的掩码，形状为 `(decoder_attention_heads,)`
            past_key_value (`Tuple(tf.Tensor)`, *可选*): 缓存的过去键和值投影状态
            training (`bool`, *可选*, 默认为 `False`):
                是否在训练模式下使用模型（某些模块如 dropout 在训练和评估中的行为不同）。
        """
        residual = hidden_states

        # 125m, 1.7B, ..., 175B 在进行自注意力之前应用层归一化
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # 自注意力机制
        # 解码器单向自注意力缓存的键/值对在位置 1 和 2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None

        # 将当前的自注意力缓存添加到 present_key_value 元组的位置 1 和 2
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
        )
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = residual + hidden_states

        # 350m 在进行自注意力之后应用层归一化
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # 全连接层
        residual = hidden_states
        # 125m, 1.7B, ..., 175B 在进行自注意力之前应用层归一化
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = residual + hidden_states

        # 350m 在进行自注意力之后应用层归一化
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        return (hidden_states, self_attn_weights, present_key_value)
    # 构建模型的方法，用于在给定输入形状的情况下构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建完成，直接返回
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        
        # 如果存在 self_attn 属性，构建 self_attn 层
        if getattr(self, "self_attn", None) is not None:
            # 在命名空间下构建 self_attn 层
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)
        
        # 如果存在 self_attn_layer_norm 属性，构建 self_attn_layer_norm 层
        if getattr(self, "self_attn_layer_norm", None) is not None:
            # 在命名空间下构建 self_attn_layer_norm 层
            with tf.name_scope(self.self_attn_layer_norm.name):
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        
        # 如果存在 fc1 属性，构建 fc1 层
        if getattr(self, "fc1", None) is not None:
            # 在命名空间下构建 fc1 层
            with tf.name_scope(self.fc1.name):
                self.fc1.build([None, None, self.embed_dim])
        
        # 如果存在 fc2 属性，构建 fc2 层
        if getattr(self, "fc2", None) is not None:
            # 在命名空间下构建 fc2 层
            with tf.name_scope(self.fc2.name):
                self.fc2.build([None, None, self.config.ffn_dim])
        
        # 如果存在 final_layer_norm 属性，构建 final_layer_norm 层
        if getattr(self, "final_layer_norm", None) is not None:
            # 在命名空间下构建 final_layer_norm 层
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.embed_dim])
OPT_START_DOCSTRING = r"""
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

    Args:
        config ([`OPTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 添加了一个文档字符串，详细描述了模型的继承关系和输入格式的支持情况

@add_start_docstrings(
    "The bare OPT Model outputting raw hidden-states without any specific head on top.",
    OPT_START_DOCSTRING,
)
# 使用装饰器 @add_start_docstrings 添加了模型类的说明，包括输出的类型和继承的文档信息

class TFOPTPreTrainedModel(TFPreTrainedModel):
    """
    TFOPT Pretrained Model that inheritates from transformers.TFPreTrainedModel

    Args:
        config: OPTConfig
    """
    
    # 定义一个 TF 优化模型的子类，继承自 TFPreTrainedModel
    # 参数 config 为 OPTConfig 类的实例，用于配置模型

    config_class = OPTConfig
    # 指定配置类为 OPTConfig，用于设置模型的参数

    base_model_prefix = "model"
    # 设置基础模型前缀为 "model"

OPT_INPUTS_DOCSTRING = r"""
"""

# 添加一个空的文档字符串 OPT_INPUTS_DOCSTRING，等待后续补充说明
    Args:
        input_ids (`tf.Tensor` of shape `({0})`):
            输入序列中词汇表中的输入序列标记的索引。

            可以使用 [`AutoTokenizer`] 获得这些索引。有关详细信息，请参见 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`]。

            [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`tf.Tensor` of shape `({0})`, *optional*):
            遮罩，用于在填充标记索引上避免执行注意力操作。遮罩值选择在 `[0, 1]`：

            - 1 表示**未遮罩**的标记，
            - 0 表示**遮罩**的标记。

            [什么是注意力遮罩？](../glossary#attention-mask)
        head_mask (`tf.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
            用于在编码器中将选定的注意力模块头部置零的遮罩。遮罩值选择在 `[0, 1]`：

            - 1 表示**未遮罩**的头部，
            - 0 表示**遮罩**的头部。

        past_key_values (`Tuple[Tuple[tf.Tensor]]` of length `config.n_layers`)
            包含注意力块预计算的键和值隐藏状态。可用于加速解码过程。
            如果使用 `past_key_values`，用户可以选择只输入最后的 `decoder_input_ids`（这些没有给出其过去键值状态的模型）的形状为 `(batch_size, 1)`，而不是所有 `decoder_input_ids` 的形状为 `(batch_size, sequence_length)`。
        use_cache (`bool`, *optional*, defaults to `True`):
            如果设置为 `True`，则返回 `past_key_values` 键值状态，可用于加速解码（参见 `past_key_values`）。在训练期间设置为 `False`，在生成期间设置为 `True`。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回张量下的 `attentions`。此参数仅在即时模式下可用，在图模式下将使用配置中的值。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关更多详细信息，请参见返回张量下的 `hidden_states`。此参数仅在即时模式下可用，在图模式下将使用配置中的值。
        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是普通元组。此参数可以在即时模式下使用，在图模式下将始终设置为 True。
        training (`bool`, *optional*, defaults to `False`):
            是否在训练模式下使用模型（某些模块如 dropout 模块在训练和评估中有不同的行为）。
"""
@keras_serializable
class TFOPTDecoder(keras.layers.Layer):
    config_class = OPTConfig

    def __init__(self, config: OPTConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config  # 初始化配置对象，包含解码器的各种配置参数
        self.padding_idx = config.pad_token_id  # 设置填充标记的索引
        self.layerdrop = config.layerdrop  # 设置层跳跃的概率
        num_embeddings = config.max_position_embeddings  # 获取最大位置编码的数量
        self.embed_tokens = TFSharedEmbeddings(
            config.vocab_size, config.word_embed_proj_dim, config.pad_token_id, name="embed_tokens"
        )  # 初始化共享的词嵌入对象
        self.embed_positions = TFOPTLearnedPositionalEmbedding(
            num_embeddings,
            config.hidden_size,
            name="embed_positions",
        )  # 初始化位置编码对象

        # 注意：`config._remove_final_layer_norm` 仅用于保持与旧版本的兼容性，
        # 在 transformers v4.20.1 之前微调过的检查点需要使用，详见 https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        else:
            self.final_layer_norm = None  # 如果不需要最终的层归一化，则为 None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = keras.layers.Dense(config.word_embed_proj_dim, name="project_out", use_bias=False)
            self.project_in = keras.layers.Dense(config.hidden_size, name="project_in", use_bias=False)
        else:
            self.project_in = None
            self.project_out = None  # 如果词嵌入投影维度与隐藏层维度相同，则为 None

        self.layers = [TFOPTDecoderLayer(config, name=f"layers.{i}") for i in range(config.num_hidden_layers)]
        self.dropout = keras.layers.Dropout(config.dropout)  # 初始化 dropout 层

    def get_embed_tokens(self):
        return self.embed_tokens  # 返回词嵌入对象

    def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens  # 设置新的词嵌入对象

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens.vocab_size = new_embeddings.shape[0]  # 更新词汇表大小
        self.embed_tokens.weight = new_embeddings  # 更新词嵌入权重矩阵

    def get_input_embeddings(self):
        return self.embed_tokens  # 返回当前词嵌入对象
    # 如果模型已经构建，则直接返回，不进行重复构建
    if self.built:
        return

    # 将模型标记为已构建状态
    self.built = True

    # 如果存在嵌入标记，构建嵌入标记模块
    if getattr(self, "embed_tokens", None) is not None:
        with tf.name_scope(self.embed_tokens.name):
            self.embed_tokens.build(None)

    # 如果存在位置嵌入，构建位置嵌入模块
    if getattr(self, "embed_positions", None) is not None:
        with tf.name_scope(self.embed_positions.name):
            self.embed_positions.build(None)

    # 如果存在最终层归一化，构建最终层归一化模块
    if getattr(self, "final_layer_norm", None) is not None:
        with tf.name_scope(self.final_layer_norm.name):
            self.final_layer_norm.build([None, None, self.config.hidden_size])

    # 如果存在输出投影层，构建输出投影层模块
    if getattr(self, "project_out", None) is not None:
        with tf.name_scope(self.project_out.name):
            self.project_out.build([None, None, self.config.hidden_size])

    # 如果存在输入投影层，构建输入投影层模块
    if getattr(self, "project_in", None) is not None:
        with tf.name_scope(self.project_in.name):
            self.project_in.build([None, None, self.config.word_embed_proj_dim])

    # 如果存在多层结构，逐层构建每一层
    if getattr(self, "layers", None) is not None:
        for layer in self.layers:
            with tf.name_scope(layer.name):
                layer.build(None)
# 使用 keras_serializable 装饰器将类声明为可序列化的 Keras 模型
@keras_serializable
class TFOPTMainLayer(keras.layers.Layer):
    # 设置配置类为 OPTConfig
    config_class = OPTConfig

    # 初始化方法，接受配置对象 config 和其他关键字参数
    def __init__(self, config: OPTConfig, **kwargs):
        super().__init__(**kwargs)
        # 将配置对象 config 存储在实例中
        self.config = config
        # 创建 TFOPTDecoder 对象，并命名为 "decoder"
        self.decoder = TFOPTDecoder(config, name="decoder")

    # 获取输入嵌入的方法，返回解码器的嵌入标记
    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    # 设置输入嵌入的方法，用新的嵌入替换解码器的嵌入标记
    def set_input_embeddings(self, new_embeddings):
        self.decoder.set_input_embeddings(new_embeddings)

    # 使用 unpack_inputs 装饰器定义的调用方法，接受多个输入参数，返回 TFBaseModelOutputWithPast 或者 Tensor 元组
    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
        **kwargs,
    ) -> Union[TFBaseModelOutputWithPast, Tuple[tf.Tensor]]:
        # 根据传入的参数或者配置对象设置输出注意力和隐藏状态
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用解码器对象进行处理，返回结果存储在 outputs 变量中
        outputs = self.decoder(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 如果 return_dict 为 False，则直接返回 outputs
        if not return_dict:
            return outputs

        # 否则，构造 TFBaseModelOutputWithPast 对象，返回其中的属性作为输出
        return TFBaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 构建方法，用于构建模型结构，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在解码器对象，则在解码器的名称空间内构建其结构
        if getattr(self, "decoder", None) is not None:
            with tf.name_scope(self.decoder.name):
                self.decoder.build(None)


# 使用 add_start_docstrings 装饰器添加模型的文档字符串说明和 OPT_START_DOCSTRING
@add_start_docstrings(
    "The bare TF OPT Model outputting raw hidden-states without any specific head on top.",
    OPT_START_DOCSTRING,
)
# 使用 keras_serializable 装饰器将类声明为可序列化的 Keras 模型
@keras_serializable
class TFOPTModel(TFOPTPreTrainedModel):
    # 设置配置类为 OPTConfig
    config_class = OPTConfig

    # 初始化方法，接受配置对象 config 和其他关键字参数
    def __init__(self, config: OPTConfig, **kwargs):
        super().__init__(config, **kwargs)
        # 将配置对象 config 存储在实例中
        self.config = config
        # 创建 TFOPTMainLayer 对象，并命名为 "model"
        self.model = TFOPTMainLayer(config, name="model")
    # 获取输入嵌入层，即模型解码器的嵌入标记
    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    # 设置输入嵌入层，用新的嵌入进行替换
    def set_input_embeddings(self, new_embeddings):
        self.model.set_input_embeddings(new_embeddings)

    # 使用装饰器 unpack_inputs 解包输入参数，并为模型的 call 方法添加文档字符串
    # 该方法用于模型调用，接收多个输入参数，返回模型输出或包含过去键值的对象
    @unpack_inputs
    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
        **kwargs,
    ) -> Union[TFBaseModelOutputWithPast, Tuple[tf.Tensor]]:
        # 根据传入的参数或者配置决定是否使用输出注意力、隐藏状态及缓存
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用模型的前向传播方法，传递给模型的参数包括输入数据、注意力掩码、头部掩码等
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 如果不要求返回字典形式的输出，则直接返回模型的原始输出
        if not return_dict:
            return outputs

        # 构造 TFBaseModelOutputWithPast 对象，包含最后隐藏状态、过去键值、隐藏状态、注意力等信息
        return TFBaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 用于服务输出的方法，根据配置决定是否返回过去键值、隐藏状态和注意力
    def serving_output(self, output):
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        # 返回 TFBaseModelOutputWithPast 对象，包含最后隐藏状态、过去键值、隐藏状态、注意力
        return TFBaseModelOutputWithPast(
            last_hidden_state=output.last_hidden_state,
            past_key_values=pkv,
            hidden_states=hs,
            attentions=attns,
        )
    # 定义模型的构建方法，用于构建模型的层次结构
    def build(self, input_shape=None):
        # 如果模型已经构建完成，则直接返回，避免重复构建
        if self.built:
            return
        # 将模型的构建状态标记为已构建
        self.built = True
        # 检查模型是否已经实例化，如果是，则在命名空间下构建模型
        if getattr(self, "model", None) is not None:
            with tf.name_scope(self.model.name):
                # 使用 None 的输入形状构建模型
                self.model.build(None)
@add_start_docstrings(
    """
    The OPT Model transformer with a language modeling head on top.
    """,
    OPT_START_DOCSTRING,
)
@keras_serializable
class TFOPTForCausalLM(TFOPTPreTrainedModel, TFCausalLanguageModelingLoss):
    # 使用 OPTConfig 作为配置类
    config_class = OPTConfig

    def __init__(self, config: OPTConfig, **kwargs):
        # 调用父类构造函数，初始化配置
        super().__init__(config, **kwargs)
        self.config = config
        # 创建 TFOPTMainLayer 模型，命名为 "model"
        self.model = TFOPTMainLayer(config, name="model")

    def get_output_embeddings(self):
        # 获取模型的输入嵌入
        return self.model.get_input_embeddings()

    def prepare_inputs_for_generation(self, inputs, past_key_values=None, use_cache=None, **kwargs):
        # 获取 kwargs 中的注意力遮罩
        attention_mask = kwargs.get("attention_mask", None)

        # 如果 past_key_values 存在，则只使用输入的最后一个标记
        if past_key_values:
            inputs = tf.expand_dims(inputs[:, -1], -1)

        # 返回准备好的输入字典
        return {
            "input_ids": inputs,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    @unpack_inputs
    @replace_return_docstrings(output_type=TFCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFCausalLMOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_CAUSAL_LM_EXPECTED_OUTPUT,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        labels: np.ndarray | tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
        **kwargs,
    ):
        # 实现模型的前向传播逻辑，详细说明参考函数装饰器
        def serving_output(self, output):
        # 根据配置决定是否使用缓存来处理输出中的过去键值
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        # 根据配置决定是否输出隐藏状态
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        # 根据配置决定是否输出注意力权重
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        # 返回带有过去键值的语言模型输出对象
        return TFCausalLMOutputWithPast(
            past_key_values=pkv,
            hidden_states=hs,
            attentions=attns,
            loss=output.loss,
            logits=output.logits,
        )

    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 标记已经构建
        self.built = True
        # 如果存在模型属性，则在名称作用域内构建模型
        if getattr(self, "model", None) is not None:
            with tf.name_scope(self.model.name):
                self.model.build(None)
```