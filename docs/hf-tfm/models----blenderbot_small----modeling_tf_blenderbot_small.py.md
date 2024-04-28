# `.\transformers\models\blenderbot_small\modeling_tf_blenderbot_small.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 2021 年版权归 Facebook 公司和 HuggingFace 公司团队所有。保留所有权利。
# 根据 Apache 许可证 2.0 版本（“许可证”）授权；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获得许可证副本：
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按“原样”分发软件，
# 没有任何明示或暗示的保证或条件。
# 请参阅许可证以获取特定语言的许可证权限。

# 导入必要的库
from __future__ import annotations

import random
from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

# 导入模型相关的输出
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFSeq2SeqLMOutput,
    TFSeq2SeqModelOutput,
)

# 导入模型相关的配置
from .configuration_blenderbot_small import BlenderbotSmallConfig

# 导入公共 API
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFPreTrainedModel,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

# 设置日志记录器
logger = logging.get_logger(__name__)

# 文档中使用的模型检查点名称
_CHECKPOINT_FOR_DOC = "facebook/blenderbot_small-90M"
_CONFIG_FOR_DOC = "BlenderbotSmallConfig"

# 定义一个较大的负值，用于在 softmax 计算中避免溢出
LARGE_NEGATIVE = -1e8


# 从 transformers.models.bart.modeling_tf_bart.shift_tokens_right 复制的函数
def shift_tokens_right(input_ids: tf.Tensor, pad_token_id: int, decoder_start_token_id: int):
    # 将 pad_token_id 和 decoder_start_token_id 转换为输入张量的数据类型
    pad_token_id = tf.cast(pad_token_id, input_ids.dtype)
    decoder_start_token_id = tf.cast(decoder_start_token_id, input_ids.dtype)
    # 创建起始 token 张量，形状与输入张量的行数相同，列数为 1
    start_tokens = tf.fill(
        (shape_list(input_ids)[0], 1), tf.convert_to_tensor(decoder_start_token_id, input_ids.dtype)
    )
    # 将输入张量向右移动一位，起始位置用起始 token 填充
    shifted_input_ids = tf.concat([start_tokens, input_ids[:, :-1]], -1)
    # 将标签中可能的 -100 值替换为 pad_token_id
    shifted_input_ids = tf.where(
        shifted_input_ids == -100,
        tf.fill(shape_list(shifted_input_ids), tf.convert_to_tensor(pad_token_id, input_ids.dtype)),
        shifted_input_ids,
    )

    # 断言保证标签张量中的值都为正数或 -100
    assert_gte0 = tf.debugging.assert_greater_equal(shifted_input_ids, tf.constant(0, dtype=input_ids.dtype))

    # 通过包装结果来确保断言操作被调用
    with tf.control_dependencies([assert_gte0]):
        shifted_input_ids = tf.identity(shifted_input_ids)

    return shifted_input_ids


# 从 transformers.models.bart.modeling_tf_bart._make_causal_mask 复制的代码
# 创建用于双向自注意力的因果遮罩
def _make_causal_mask(input_ids_shape: tf.TensorShape, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    # 获取批大小
    bsz = input_ids_shape[0]
    # 获取目标长度
    tgt_len = input_ids_shape[1]
    # 创建全为大负数的因果遮罩
    mask = tf.ones((tgt_len, tgt_len)) * LARGE_NEGATIVE
    # 创建用于生成遮罩的条件
    mask_cond = tf.range(shape_list(mask)[-1])

    # 根据条件生成因果遮罩
    mask = tf.where(mask_cond < tf.reshape(mask_cond + 1, (shape_list(mask)[-1], 1)), 0.0, mask)

    # 如果过去的键值长度大于0，则在遮罩前添加过去的键值长度个零
    if past_key_values_length > 0:
        mask = tf.concat([tf.zeros((tgt_len, past_key_values_length)), mask], axis=-1)

    return tf.tile(mask[None, None, :, :], (bsz, 1, 1, 1))


# 从`[bsz, seq_len]`扩展注意力遮罩到`[bsz, 1, tgt_seq_len, src_seq_len]`
def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    # 获取源序列长度
    src_len = shape_list(mask)[1]
    # 如果目标序列长度未指定，则使用源序列长度
    tgt_len = tgt_len if tgt_len is not None else src_len
    one_cst = tf.constant(1.0)
    mask = tf.cast(mask, dtype=one_cst.dtype)
    # 在维度上进行复制，从`[bsz, seq_len]`扩展到`[bsz, 1, tgt_seq_len, src_seq_len]`
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))

    return (one_cst - expanded_mask) * LARGE_NEGATIVE


# 学习位置嵌入的模块，上限为固定的最大大小
class TFBlenderbotSmallLearnedPositionalEmbedding(tf.keras.layers.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs):
        super().__init__(num_embeddings, embedding_dim, **kwargs)

    def call(
        self, input_shape: tf.TensorShape, past_key_values_length: int = 0, position_ids: tf.Tensor | None = None
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        # 如果未提供位置 id，则创建位置 id
        if position_ids is None:
            seq_len = input_shape[1]
            position_ids = tf.range(seq_len, delta=1, name="range")
            position_ids += past_key_values_length

        return super().call(tf.cast(position_ids, dtype=tf.int32))


# 多头注意力机制，来自 "Attention Is All You Need"
class TFBlenderbotSmallAttention(tf.keras.layers.Layer):
    """Multi-headed attention from "Attention Is All You Need"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        **kwargs,
    # 初始化函数，继承父类的初始化方法，并设置一些参数
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.head_dim = embed_dim // num_heads
        # 检查 embed_dim 是否可以被 num_heads 整除
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        # 初始化一些 Dense 层
        self.k_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="k_proj")
        self.q_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="q_proj")
        self.v_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="v_proj")
        self.out_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="out_proj")

    # 将输入张量重塑为指定形状
    def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
        return tf.transpose(tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))

    # 模型调用函数，接受一些输入参数
    def call(
        self,
        hidden_states: tf.Tensor,
        key_value_states: tf.Tensor | None = None,
        past_key_value: Tuple[Tuple[tf.Tensor]] | None = None,
        attention_mask: tf.Tensor | None = None,
        layer_head_mask: tf.Tensor | None = None,
        training: Optional[bool] = False,
    # 构建模型，设置输入形状
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建 k_proj 层
        if getattr(self, "k_proj", None) is not None:
            with tf.name_scope(self.k_proj.name):
                self.k_proj.build([None, None, self.embed_dim])
        # 构建 q_proj 层
        if getattr(self, "q_proj", None) is not None:
            with tf.name_scope(self.q_proj.name):
                self.q_proj.build([None, None, self.embed_dim])
        # 构建 v_proj 层
        if getattr(self, "v_proj", None) is not None:
            with tf.name_scope(self.v_proj.name):
                self.v_proj.build([None, None, self.embed_dim])
        # 构建 out_proj 层
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.embed_dim])
# 从transformers.models.bart.modeling_tf_bart.TFBartEncoderLayer复制并修改为BlenderbotSmall
class TFBlenderbotSmallEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: BlenderbotSmallConfig, **kwargs):
        super().__init__(**kwargs)
        # 设置嵌入维度为配置中的d_model
        self.embed_dim = config.d_model
        # 初始化自注意力层
        self.self_attn = TFBlenderbotSmallAttention(
            self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout, name="self_attn"
        )
        # 初始化自注意力层的层归一化
        self.self_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        # 初始化dropout层
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        # 获取激活函数
        self.activation_fn = get_tf_activation(config.activation_function)
        # 初始化激活函数的dropout层
        self.activation_dropout = tf.keras.layers.Dropout(config.activation_dropout)
        # 初始化全连接层1
        self.fc1 = tf.keras.layers.Dense(config.encoder_ffn_dim, name="fc1")
        # 初始化全连接层2
        self.fc2 = tf.keras.layers.Dense(self.embed_dim, name="fc2")
        # 初始化最终的层归一化
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        # 保存配置
        self.config = config

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: np.ndarray | tf.Tensor | None,
        layer_head_mask: tf.Tensor | None,
        training: Optional[bool] = False,
    ) -> tf.Tensor:
        """
        Args:
            hidden_states (`tf.Tensor`): 输入到层的形状为`(batch, seq_len, embed_dim)`
            attention_mask (`tf.Tensor`): 大小为`(batch, 1, tgt_len, src_len)`的注意力掩码，
                其中填充元素由非常大的负值指示。
            layer_head_mask (`tf.Tensor`): 给定层中注意力头的掩码大小为`(encoder_attention_heads,)`
        """
        # 保存残差连接
        residual = hidden_states
        # 使用自注意力层处理输入
        hidden_states, self_attn_weights, _ = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, layer_head_mask=layer_head_mask
        )

        # 断言自注意力层的输出形状与残差形状相同
        tf.debugging.assert_equal(
            shape_list(hidden_states),
            shape_list(residual),
            message=f"Self attn modified the shape of query {shape_list(residual)} to {shape_list(hidden_states)}",
        )

        # 应用dropout
        hidden_states = self.dropout(hidden_states, training=training)
        # 加上残差连接
        hidden_states = residual + hidden_states
        # 应用自注意力层的层归一化
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 保存残差连接
        residual = hidden_states
        # 应用激活函数和全连接层1
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 应用激活函数的dropout
        hidden_states = self.activation_dropout(hidden_states, training=training)
        # 应用全连接层2
        hidden_states = self.fc2(hidden_states)
        # 应用dropout
        hidden_states = self.dropout(hidden_states, training=training)
        # 加上残差连接
        hidden_states = residual + hidden_states
        # 应用最终的层归一化
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states, self_attn_weights
    # 构建模型，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在 self_attn 属性，则构建 self_attn 层
        if getattr(self, "self_attn", None) is not None:
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)
        # 如果存在 self_attn_layer_norm 属性，则构建 self_attn_layer_norm 层
        if getattr(self, "self_attn_layer_norm", None) is not None:
            with tf.name_scope(self.self_attn_layer_norm.name):
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        # 如果存在 fc1 属性，则构建 fc1 层
        if getattr(self, "fc1", None) is not None:
            with tf.name_scope(self.fc1.name):
                self.fc1.build([None, None, self.embed_dim])
        # 如果存在 fc2 属性，则构建 fc2 层
        if getattr(self, "fc2", None) is not None:
            with tf.name_scope(self.fc2.name):
                self.fc2.build([None, None, self.config.encoder_ffn_dim])
        # 如果存在 final_layer_norm 属性，则构建 final_layer_norm 层
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.embed_dim])
# 从transformers.models.bart.modeling_tf_bart.TFBartDecoderLayer中复制代码并将Bart改为BlenderbotSmall
class TFBlenderbotSmallDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: BlenderbotSmallConfig, **kwargs):
        super().__init__(**kwargs)
        # 初始化层，设置嵌入维度为模型配置中的d_model
        self.embed_dim = config.d_model
        # 创建BlenderbotSmallAttention层用于自注意力机制
        self.self_attn = TFBlenderbotSmallAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="self_attn",
            is_decoder=True,
        )
        # Dropout层，用于自注意力机制输出
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        # 激活函数，根据模型配置获取相应激活函数
        self.activation_fn = get_tf_activation(config.activation_function)
        # 激活函数的Dropout层
        self.activation_dropout = tf.keras.layers.Dropout(config.activation_dropout)

        # LayerNormalization层，用于自注意力机制输出
        self.self_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        # 创建BlenderbotSmallAttention层用于编码器注意力机制
        self.encoder_attn = TFBlenderbotSmallAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="encoder_attn",
            is_decoder=True,
        )
        # LayerNormalization层，用于编码器注意力机制输出
        self.encoder_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="encoder_attn_layer_norm")
        # 第一个全连接层，前馈神经网络的第一层
        self.fc1 = tf.keras.layers.Dense(config.decoder_ffn_dim, name="fc1")
        # 第二个全连接层，前馈神经网络的第二层
        self.fc2 = tf.keras.layers.Dense(self.embed_dim, name="fc2")
        # 最终的LayerNormalization层
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        # 记录模型配置
        self.config = config

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        layer_head_mask: tf.Tensor | None = None,
        cross_attn_layer_head_mask: tf.Tensor | None = None,
        past_key_value: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        training: Optional[bool] = False,
    # 构建模型，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在 self_attn 属性，则构建 self_attn 层
        if getattr(self, "self_attn", None) is not None:
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)
        # 如果存在 self_attn_layer_norm 属性，则构建 self_attn_layer_norm 层
        if getattr(self, "self_attn_layer_norm", None) is not None:
            with tf.name_scope(self.self_attn_layer_norm.name):
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        # 如果存在 encoder_attn 属性，则构建 encoder_attn 层
        if getattr(self, "encoder_attn", None) is not None:
            with tf.name_scope(self.encoder_attn.name):
                self.encoder_attn.build(None)
        # 如果存在 encoder_attn_layer_norm 属性，则构建 encoder_attn_layer_norm 层
        if getattr(self, "encoder_attn_layer_norm", None) is not None:
            with tf.name_scope(self.encoder_attn_layer_norm.name):
                self.encoder_attn_layer_norm.build([None, None, self.embed_dim])
        # 如果存在 fc1 属性，则构建 fc1 层
        if getattr(self, "fc1", None) is not None:
            with tf.name_scope(self.fc1.name):
                self.fc1.build([None, None, self.embed_dim])
        # 如果存在 fc2 属性，则构建 fc2 层
        if getattr(self, "fc2", None) is not None:
            with tf.name_scope(self.fc2.name):
                self.fc2.build([None, None, self.config.decoder_ffn_dim])
        # 如果存在 final_layer_norm 属性，则构建 final_layer_norm 层
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.embed_dim])
class TFBlenderbotSmallPreTrainedModel(TFPreTrainedModel):
    # 定义 TFBlenderbotSmallPreTrainedModel 类，继承自 TFPreTrainedModel
    config_class = BlenderbotSmallConfig
    # 设置 config_class 为 BlenderbotSmallConfig
    base_model_prefix = "model"
    # 设置 base_model_prefix 为 "model"


BLENDERBOT_SMALL_START_DOCSTRING = r"""
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
        config ([`BlenderbotSmallConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""
# 定义 BLENDERBOT_SMALL_START_DOCSTRING 字符串，包含模型的继承信息、使用提示和参数说明


BLENDERBOT_SMALL_GENERATION_EXAMPLE = r"""
    Conversation example::

    ```py
    >>> from transformers import AutoTokenizer, TFBlenderbotSmallForConditionalGeneration

    >>> mname = "facebook/blenderbot_small-90M"
    >>> model = BlenderbotSmallForConditionalGeneration.from_pretrained(mname)
    >>> tokenizer = AutoTokenizer.from_pretrained(mname)



# 定义 BLENDERBOT_SMALL_GENERATION_EXAMPLE 字符串，包含对话示例的代码块
    # 定义人类的话语内容
    UTTERANCE = "My friends are cool but they eat too many carbs."
    # 打印人类的话语
    print("Human: ", UTTERANCE)
    # 使用分词器将人类话语转换为模型输入的张量
    inputs = tokenizer([UTTERANCE], return_tensors="tf")
    
    # 使用模型生成回复
    reply_ids = model.generate(**inputs)
    # 打印模型生成的回复
    print("Bot: ", tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0])
    # 模型回复的内容：他们吃什么样的碳水化合物？我对碳水化合物了解不多。
    
    # 定义人类的回复
    REPLY = "I'm not sure"
    # 打印人类的回复
    print("Human: ", REPLY)
    # 下一个话语的内容，包括前一话语和人类的回复
    NEXT_UTTERANCE = (
        "My friends are cool but they eat too many carbs.</s> "
        "<s>what kind of carbs do they eat? i don't know much about carbs.</s> "
        "<s>I'm not sure."
    )
    
    # 使用分词器将下一个话语转换为模型输入的张量
    inputs = tokenizer([NEXT_UTTERANCE], return_tensors="tf")
    # 移除张量中的token_type_ids，因为在此情境中不需要
    inputs.pop("token_type_ids")
    # 使用模型生成下一个话语的回复
    next_reply_ids = model.generate(**inputs)
    # 打印模型生成的下一个话语的回复
    print("Bot: ", tokenizer.batch_decode(next_reply_ids, skip_special_tokens=True)[0])
    ```py  
"""
BLENDERBOT_SMALL_INPUTS_DOCSTRING = r"""
"""


@keras_serializable
class TFBlenderbotSmallEncoder(tf.keras.layers.Layer):
    config_class = BlenderbotSmallConfig
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`TFBlenderbotSmallEncoderLayer`].

    Args:
        config: BlenderbotSmallConfig
    """

    def __init__(
        self, config: BlenderbotSmallConfig, embed_tokens: Optional[tf.keras.layers.Embedding] = None, **kwargs
    ):
        super().__init__(**kwargs)
        self.config = config
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.layerdrop = config.encoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = tf.math.sqrt(float(config.d_model)) if config.scale_embedding else 1.0

        self.embed_tokens = embed_tokens
        self.embed_positions = TFBlenderbotSmallLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            name="embed_positions",
        )
        self.layers = [TFBlenderbotSmallEncoderLayer(config, name=f"layers.{i}") for i in range(config.encoder_layers)]
        self.layernorm_embedding = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm_embedding")
        self.embed_dim = config.d_model

    def get_embed_tokens(self):
        return self.embed_tokens

    def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens

    @unpack_inputs
    def call(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
    ):
        """
        Perform the forward pass of the encoder.

        Args:
            input_ids: Optional, input token IDs.
            inputs_embeds: Optional, input embeddings.
            attention_mask: Optional, mask to avoid attending to padding tokens.
            head_mask: Optional, mask to nullify some heads of the attention modules.
            output_attentions: Optional, whether to return attentions.
            output_hidden_states: Optional, whether to return hidden states.
            return_dict: Optional, whether to return a dictionary.
            training: Optional, whether the model is in training mode.

        Returns:
            Depending on the configuration, returns a dictionary with different elements.
        """
        ...

    def build(self, input_shape=None):
        """
        Builds the model.

        Args:
            input_shape: Optional, input shape.

        Returns:
            None
        """
        if self.built:
            return
        self.built = True
        if getattr(self, "embed_positions", None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)
        if getattr(self, "layernorm_embedding", None) is not None:
            with tf.name_scope(self.layernorm_embedding.name):
                self.layernorm_embedding.build([None, None, self.embed_dim])
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)


@keras_serializable
class TFBlenderbotSmallDecoder(tf.keras.layers.Layer):
    config_class = BlenderbotSmallConfig
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`TFBlenderbotSmallDecoderLayer`]

    Args:
        config: BlenderbotSmallConfig
        embed_tokens: output embedding
    """

    def __init__(
        self, config: BlenderbotSmallConfig, embed_tokens: Optional[tf.keras.layers.Embedding] = None, **kwargs
    ):
        """
        Initialize the decoder layer.

        Args:
            config: The decoder configuration.
            embed_tokens: Optional, output embedding.

        Returns:
            None
        """
        super().__init__(**kwargs)
        self.config = config
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = tf.math.sqrt(float(config.d_model)) if config.scale_embedding else 1.0

        self.embed_tokens = embed_tokens
        self.embed_positions = TFBlenderbotSmallLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            name="embed_positions",
        )
        self.layers = [TFBlenderbotSmallDecoderLayer(config, name=f"layers.{i}") for i in range(config.decoder_layers)]
        self.layernorm_embedding = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm_embedding")
        self.embed_dim = config.d_model
        # 调用父类的初始化方法，传入关键字参数
        super().__init__(**kwargs)
        # 将配置信息保存在实例中
        self.config = config
        # 将填充索引保存在实例中
        self.padding_idx = config.pad_token_id
        # 将嵌入词向量保存在实例中
        self.embed_tokens = embed_tokens
        # 保存解码器层的丢弃率
        self.layerdrop = config.decoder_layerdrop
        # 创建一个可学习的位置嵌入层
        self.embed_positions = TFBlenderbotSmallLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            name="embed_positions",
        )
        # 如果配置中指定了对嵌入进行缩放，则设置缩放因子为根号下 d_model，否则为 1.0
        self.embed_scale = tf.math.sqrt(float(config.d_model)) if config.scale_embedding else 1.0
        # 创建解码器层列表，根据配置中的解码器层数进行创建
        self.layers = [TFBlenderbotSmallDecoderLayer(config, name=f"layers.{i}") for i in range(config.decoder_layers)]
        # 创建嵌入层的 LayerNormalization 层
        self.layernorm_embedding = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm_embedding")

        # 创建一个 dropout 层，用于在训练过程中进行随机丢弃
        self.dropout = tf.keras.layers.Dropout(config.dropout)

    # 获取嵌入词向量
    def get_embed_tokens(self):
        return self.embed_tokens

    # 设置嵌入词向量
    def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens

    # 定义解码器的前向传播逻辑
    @unpack_inputs
    def call(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
    ):
        # 省略部分代码，具体逻辑在后续注释中解释

    # 构建解码器网络结构
    def build(self, input_shape=None):
        # 如果已经构建过网络结构，则直接返回，不再重复构建
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果已经定义了嵌入位置层，则构建嵌入位置层
        if getattr(self, "embed_positions", None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)
        # 如果已经定义了 LayerNormalization 层，则构建 LayerNormalization 层
        if getattr(self, "layernorm_embedding", None) is not None:
            with tf.name_scope(self.layernorm_embedding.name):
                self.layernorm_embedding.build([None, None, self.config.d_model])
        # 遍历解码器的每一层，并构建每一层
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)
# 使用 keras_serializable 装饰器将类 TFBlenderbotSmallMainLayer 序列化为 Keras 模型
@keras_serializable
class TFBlenderbotSmallMainLayer(tf.keras.layers.Layer):
    # 指定配置类为 BlenderbotSmallConfig
    config_class = BlenderbotSmallConfig

    # 初始化方法，接受 BlenderbotSmallConfig 类型的 config 参数和其他关键字参数
    def __init__(self, config: BlenderbotSmallConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 将传入的 config 参数赋值给 self.config
        self.config = config
        # 创建一个共享的 Embedding 层，用于共享输入的词嵌入
        self.shared = tf.keras.layers.Embedding(
            input_dim=config.vocab_size,
            output_dim=config.d_model,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=self.config.init_std),
            name="model.shared",
        )
        # 添加一个属性用于指定层的名称范围（用于加载/存储权重）
        self.shared.load_weight_prefix = "model.shared"

        # 创建编码器对象，传入 config 和共享的 Embedding 层
        self.encoder = TFBlenderbotSmallEncoder(config, self.shared, name="encoder")
        # 创建解码器对象，传入 config 和共享的 Embedding 层
        self.decoder = TFBlenderbotSmallDecoder(config, self.shared, name="decoder")

    # 获取输入词嵌入层
    def get_input_embeddings(self):
        return self.shared

    # 设置输入词嵌入层
    def set_input_embeddings(self, new_embeddings):
        # 将新的词嵌入层赋值给 self.shared
        self.shared = new_embeddings
        # 更新编码器和解码器的 embed_tokens 属性为新的词嵌入层
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    # 定义 call 方法，接受多个输入参数和关键字参数
    @unpack_inputs
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_position_ids=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs: Optional[Union[Tuple, TFBaseModelOutput]] = None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
        **kwargs,
        ):
            # 如果用户指定了是否输出隐藏状态，则使用用户提供的设置，否则使用模型配置中的默认设置
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )

        if encoder_outputs is None:
            # 如果没有提供编码器输出，则通过编码器处理输入数据来获取编码器输出
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                training=training,
            )
        # 如果用户传入的编码器输出是一个元组，并且设置了return_dict=True，则将其包装在TFBaseModelOutput中
        elif return_dict and not isinstance(encoder_outputs, TFBaseModelOutput):
            encoder_outputs = TFBaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        # 如果用户传入的编码器输出是TFBaseModelOutput，并且设置了return_dict=False，则将其包装在元组中
        elif not return_dict and not isinstance(encoder_outputs, tuple):
            encoder_outputs = encoder_outputs.to_tuple()

        # 通过解码器处理解码器输入，生成解码器输出
        decoder_outputs = self.decoder(
            decoder_input_ids,
            attention_mask=decoder_attention_mask,
            position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        if not return_dict:
            # 如果不返回字典形式的结果，则将解码器输出和编码器输出连接在一起返回
            return decoder_outputs + encoder_outputs

        # 返回TFSeq2SeqModelOutput类型的字典形式的结果
        return TFSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
    # 如果模型已经构建，则直接返回，不进行重复构建
    if self.built:
        return
    # 设置模型已构建标志位为 True
    self.built = True
    # 共享/绑定的权重应在模型基本命名空间中
    # 在 tf.name_scope 后面添加 "/"（不是在开头！）将其放置在根命名空间而不是当前命名空间。
    with tf.name_scope(self.shared.load_weight_prefix + "/" + self.shared.name + "/"):
        # 构建共享层
        self.shared.build(None)
    # 如果存在编码器，则构建编码器
    if getattr(self, "encoder", None) is not None:
        with tf.name_scope(self.encoder.name):
            # 构建编码器
            self.encoder.build(None)
    # 如果存在解码器，则构建解码器
    if getattr(self, "decoder", None) is not None:
        with tf.name_scope(self.decoder.name):
            # 构建解码器
            self.decoder.build(None)
# 添加模型文档字符串，描述该模型输出原始隐藏状态，没有特定的头部
# 并引用 BLENDERBOT_SMALL_START_DOCSTRING
@add_start_docstrings(
    "The bare BLENDERBOT_SMALL Model outputting raw hidden-states without any specific head on top.",
    BLENDERBOT_SMALL_START_DOCSTRING,
)
# 定义 TFBlenderbotSmallModel 类，继承自 TFBlenderbotSmallPreTrainedModel
class TFBlenderbotSmallModel(TFBlenderbotSmallPreTrainedModel):
    # 初始化方法，接受 BlenderbotSmallConfig 类型的配置参数
    def __init__(self, config: BlenderbotSmallConfig, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 创建 TFBlenderbotSmallMainLayer 对象，命名为 "model"
        self.model = TFBlenderbotSmallMainLayer(config, name="model")

    # 获取编码器
    def get_encoder(self):
        return self.model.encoder

    # 获取解码器
    def get_decoder(self):
        return self.model.decoder

    # 定义 call 方法，接受多个输入参数，并返回 Union[Tuple[tf.Tensor], TFSeq2SeqModelOutput] 类型的输出
    @unpack_inputs
    @add_start_docstrings_to_model_forward(BLENDERBOT_SMALL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSeq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: tf.Tensor | None = None,
        attention_mask: tf.Tensor | None = None,
        decoder_input_ids: tf.Tensor | None = None,
        decoder_attention_mask: tf.Tensor | None = None,
        decoder_position_ids: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        decoder_head_mask: tf.Tensor | None = None,
        cross_attn_head_mask: tf.Tensor | None = None,
        encoder_outputs: Optional[Union[Tuple, TFBaseModelOutput]] = None,
        past_key_values: List[tf.Tensor] | None = None,
        inputs_embeds: tf.Tensor | None = None,
        decoder_inputs_embeds: tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
        **kwargs,
    ) -> Union[Tuple[tf.Tensor], TFSeq2SeqModelOutput]:
        # 调用 self.model 的方法，传入各种输入参数
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 返回模型输出
        return outputs

    # 从 transformers.models.bart.modeling_tf_bart.TFBartModel.serving_output 复制而来
    # 定义一个方法用于处理模型的输出
    def serving_output(self, output):
        # 如果配置了使用缓存，则提取出模型输出中的过去键值（past_key_values）中的第一个值
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        # 如果配置了输出隐藏状态，则将模型输出中的解码器隐藏状态转换为张量
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置了输出注意力权重，则将模型输出中的解码器注意力权重转换为张量
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        # 如果配置了输出注意力权重，则将模型输出中的交叉注意力权重转换为张量
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        # 如果配置了输出隐藏状态，则将模型输出中的编码器隐藏状态转换为张量
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置了输出注意力权重，则将模型输出中的编码器注意力权重转换为张量
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None

        # 返回一个 TFSeq2SeqModelOutput 对象，其中包括了模型的输出信息
        return TFSeq2SeqModelOutput(
            last_hidden_state=output.last_hidden_state,  # 最后隐藏状态
            past_key_values=pkv,  # 过去键值
            decoder_hidden_states=dec_hs,  # 解码器隐藏状态
            decoder_attentions=dec_attns,  # 解码器注意力权重
            cross_attentions=cross_attns,  # 交叉注意力权重
            encoder_last_hidden_state=output.encoder_last_hidden_state,  # 编码器最后隐藏状态
            encoder_hidden_states=enc_hs,  # 编码器隐藏状态
            encoder_attentions=enc_attns,  # 编码器注意力权重
        )

    # 构建方法
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记已构建
        self.built = True
        # 如果模型已经存在，则在模型名称作用域下构建模型
        if getattr(self, "model", None) is not None:
            with tf.name_scope(self.model.name):
                self.model.build(None)
# 定义 BiasLayer 类，用于表示偏置，用于序列化目的
class BiasLayer(tf.keras.layers.Layer):
    """
    Bias as a layer. It is used for serialization purposes: `tf.keras.Model.save_weights` stores on a per-layer basis,
    so all weights have to be registered in a layer.
    """

    def __init__(self, shape, initializer, trainable, name, **kwargs):
        super().__init__(name=name, **kwargs)
        # 添加偏置权重变量，用于模型训练
        self.bias = self.add_weight(name=name, shape=shape, initializer=initializer, trainable=trainable)

    def call(self, x):
        return x + self.bias


# 定义 TFBlenderbotSmallForConditionalGeneration 类，继承自 TFBlenderbotSmallPreTrainedModel 和 TFCausalLanguageModelingLoss
@add_start_docstrings(
    "The BLENDERBOT_SMALL Model with a language modeling head. Can be used for summarization.",
    BLENDERBOT_SMALL_START_DOCSTRING,
)
class TFBlenderbotSmallForConditionalGeneration(TFBlenderbotSmallPreTrainedModel, TFCausalLanguageModelingLoss):
    _keys_to_ignore_on_load_unexpected = [
        r"model.encoder.embed_tokens.weight",
        r"model.decoder.embed_tokens.weight",
    ]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 创建 TFBlenderbotSmallMainLayer 模型实例
        self.model = TFBlenderbotSmallMainLayer(config, name="model")
        self.use_cache = config.use_cache
        # 创建 BiasLayer 实例，用于添加偏置到最终输出 logits
        # final_bias_logits 在 pytorch 中被注册为缓冲区，为了一致性，不可训练
        self.bias_layer = BiasLayer(
            name="final_logits_bias", shape=[1, config.vocab_size], initializer="zeros", trainable=False
        )

    def get_decoder(self):
        return self.model.decoder

    def get_encoder(self):
        return self.model.encoder

    def get_output_embeddings(self):
        return self.get_input_embeddings()

    def set_output_embeddings(self, value):
        self.set_input_embeddings(value)

    def get_bias(self):
        return {"final_logits_bias": self.bias_layer.bias}

    def set_bias(self, value):
        # 替换现有包含偏置的层，以进行正确的（反）序列化
        vocab_size = value["final_logits_bias"].shape[-1]
        self.bias_layer = BiasLayer(
            name="final_logits_bias", shape=[1, vocab_size], initializer="zeros", trainable=False
        )
        self.bias_layer.bias.assign(value["final_logits_bias"])

    @unpack_inputs
    @add_start_docstrings_to_model_forward(BLENDERBOT_SMALL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BLENDERBOT_SMALL_GENERATION_EXAMPLE)
    # 定义一个方法，用于调用模型
    def call(
        # 输入序列的 token IDs 张量，可以为空
        input_ids: tf.Tensor | None = None,
        # 注意力掩码张量，可以为空
        attention_mask: tf.Tensor | None = None,
        # 解码器输入序列的 token IDs 张量，可以为空
        decoder_input_ids: tf.Tensor | None = None,
        # 解码器注意力掩码张量，可以为空
        decoder_attention_mask: tf.Tensor | None = None,
        # 解码器位置 IDs 张量，可以为空
        decoder_position_ids: tf.Tensor | None = None,
        # 头部掩码张量，可以为空
        head_mask: tf.Tensor | None = None,
        # 解码器头部掩码张量，可以为空
        decoder_head_mask: tf.Tensor | None = None,
        # 交叉注意力头部掩码张量，可以为空
        cross_attn_head_mask: tf.Tensor | None = None,
        # 编码器输出的可选模型输出，可以为空
        encoder_outputs: Optional[TFBaseModelOutput] = None,
        # 过去的键值列表，可以为空
        past_key_values: List[tf.Tensor] | None = None,
        # 输入嵌入张量，可以为空
        inputs_embeds: tf.Tensor | None = None,
        # 解码器输入嵌入张量，可以为空
        decoder_inputs_embeds: tf.Tensor | None = None,
        # 是否使用缓存，可以为空
        use_cache: Optional[bool] = None,
        # 是否输出注意力权重，可以为空
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，可以为空
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典格式的输出，可以为空
        return_dict: Optional[bool] = None,
        # 标签张量，可以为空
        labels: tf.Tensor | None = None,
        # 是否处于训练模式，缺省为 False
        training: Optional[bool] = False,
    ) -> Union[Tuple[tf.Tensor], TFSeq2SeqLMOutput]:
        r"""
        labels (`tf.tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        """

        # 如果提供了标签，则根据配置处理标签
        if labels is not None:
            # 将标签中的填充标记（pad_token_id）替换为-100
            labels = tf.where(
                labels == self.config.pad_token_id,
                tf.cast(tf.fill(shape_list(labels), -100), labels.dtype),
                labels,
            )
            # 设置 use_cache 为 False，用于控制是否使用缓存
            use_cache = False
            # 如果未提供 decoder_input_ids 和 decoder_inputs_embeds，则根据标签生成 decoder_input_ids
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        # 调用模型生成输出结果
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 计算语言模型的 logits
        lm_logits = tf.matmul(outputs[0], self.model.shared.weights, transpose_b=True)
        # 添加偏置层
        lm_logits = self.bias_layer(lm_logits)
        # 如果提供了标签，则计算 masked_lm_loss
        masked_lm_loss = None if labels is None else self.hf_compute_loss(labels, lm_logits)

        # 如果不返回字典形式的结果，则将输出格式化为元组
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        # 返回 TFSeq2SeqLMOutput 对象
        return TFSeq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,  # d 输出的索引 1
            decoder_hidden_states=outputs.decoder_hidden_states,  # d 输出的索引 2
            decoder_attentions=outputs.decoder_attentions,  # d 输出的索引 3
            cross_attentions=outputs.cross_attentions,  # d 输出的索引 4
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,  # encoder 输出的索引 0
            encoder_hidden_states=outputs.encoder_hidden_states,  # e 输出的索引 1
            encoder_attentions=outputs.encoder_attentions,  # e 输出的索引 2
        )

    # 以下是从 transformers.models.bart.modeling_tf_bart.TFBartForConditionalGeneration.serving_output 复制的代码
    # 定义一个方法，用于处理模型的输出结果
    def serving_output(self, output):
        # 如果配置要求使用缓存，则获取过去的键值对，否则设置为 None
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        # 如果配置要求输出隐藏状态，则将 decoder_hidden_states 转换为张量，否则设置为 None
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置要求输出注意力权重，则将 decoder_attentions 转换为张量，否则设置为 None
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        # 如果配置要求输出注意力权重，则将 cross_attentions 转换为张量，否则设置为 None
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        # 如果配置要求输出隐藏状态，则将 encoder_hidden_states 转换为张量，否则设置为 None
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置要求输出注意力权重，则将 encoder_attentions 转换为张量，否则设置为 None
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None

        # 返回 TFSeq2SeqLMOutput 对象，包含模型的输出信息
        return TFSeq2SeqLMOutput(
            logits=output.logits,
            past_key_values=pkv,
            decoder_hidden_states=dec_hs,
            decoder_attentions=dec_attns,
            cross_attentions=cross_attns,
            encoder_last_hidden_state=output.encoder_last_hidden_state,
            encoder_hidden_states=enc_hs,
            encoder_attentions=enc_attns,
        )

    # 从 transformers.models.bart.modeling_tf_bart.TFBartForConditionalGeneration.prepare_inputs_for_generation 复制而来的方法
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # 如果使用了 past_key_values，则只保留 decoder_input_ids 的最后一个 token
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # 根据不同的情况计算 decoder_position_ids
        if decoder_attention_mask is not None:  # xla
            # 使用累积求和函数计算 decoder_position_ids
            decoder_position_ids = tf.math.cumsum(decoder_attention_mask, axis=-1, exclusive=True)[:, -1:]
        elif past_key_values is not None:  # no xla + past_key_values
            # 如果没有使用 xla 且使用了 past_key_values，则使用 past_key_values 的 shape[2] 作为 decoder_position_ids
            decoder_position_ids = past_key_values[0][0].shape[2]
        else:  # no xla + no past_key_values
            # 如果既没有使用 xla 也没有使用 past_key_values，则使用 decoder_input_ids 的长度范围作为 decoder_position_ids
            decoder_position_ids = tf.range(decoder_input_ids.shape[1])

        # 返回一个字典，包含模型生成所需的输入信息
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_position_ids": decoder_position_ids,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }
    # 构建模型，如果已经构建过则直接返回
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在模型属性，则构建模型
        if getattr(self, "model", None) is not None:
            # 使用模型的名称创建命名空间
            with tf.name_scope(self.model.name):
                # 构建模型
                self.model.build(None)
        # 如果存在偏置层属性，则构建偏置层
        if getattr(self, "bias_layer", None) is not None:
            # 使用偏置层的名称创建命名空间
            with tf.name_scope(self.bias_layer.name):
                # 构建偏置层
                self.bias_layer.build(None)
```