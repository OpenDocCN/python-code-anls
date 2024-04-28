# `.\transformers\models\pegasus\modeling_tf_pegasus.py`

```py
# 定义一个 TensorFlow 2.0 Pegasus 模型，该模型用于生成文本摘要
from __future__ import annotations

import random
from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf

# 从 Hugging Face 的自定义包中导入 TF 激活函数
from ...activations_tf import get_tf_activation
# 从 Hugging Face 的自定义包中导入 TF 模型输出相关的类
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFSeq2SeqLMOutput,
    TFSeq2SeqModelOutput,
)

# Public API
# 从 Hugging Face 的工具包中导入 TF 语言建模损失类、TF 模型输入类型、TF 预训练模型类等
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    keras_serializable,
    unpack_inputs,
)
# 从 Hugging Face 的 TensorFlow 工具包中导入辅助函数，用于检查嵌入范围、处理形状等
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
# 从 Hugging Face 的工具包中导入辅助函数，用于添加文档字符串、记录日志等
from ...utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

# 从 Pegasus 配置中导入 PegasusConfig 类
from .configuration_pegasus import PegasusConfig

# 获取全局日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置信息
_CHECKPOINT_FOR_DOC = "google/pegasus-large"
_CONFIG_FOR_DOC = "PegasusConfig"

# 定义一个较大的负数常量
LARGE_NEGATIVE = -1e8


# 从 transformers.models.bart.modeling_tf_bart 中复制的函数，用于将输入的 token_ids 向右移动一位
def shift_tokens_right(input_ids: tf.Tensor, pad_token_id: int, decoder_start_token_id: int):
    pad_token_id = tf.cast(pad_token_id, input_ids.dtype)
    decoder_start_token_id = tf.cast(decoder_start_token_id, input_ids.dtype)
    # 创建与输入 token_ids 相同形状的起始 token_ids
    start_tokens = tf.fill(
        (shape_list(input_ids)[0], 1), tf.convert_to_tensor(decoder_start_token_id, input_ids.dtype)
    )
    # 将原始 token_ids 右移一位，并将填充的 token 替换为 pad_token_id
    shifted_input_ids = tf.concat([start_tokens, input_ids[:, :-1]], -1)
    # 将可能存在的 -100 值替换为 pad_token_id
    shifted_input_ids = tf.where(
        shifted_input_ids == -100,
        tf.fill(shape_list(shifted_input_ids), tf.convert_to_tensor(pad_token_id, input_ids.dtype)),
        shifted_input_ids,
    )

    # 确保 shifted_input_ids 中的值都大于等于 0
    assert_gte0 = tf.debugging.assert_greater_equal(shifted_input_ids, tf.constant(0, dtype=input_ids.dtype))

    # 使用控制依赖项确保断言操作的调用
    with tf.control_dependencies([assert_gte0]):
        shifted_input_ids = tf.identity(shifted_input_ids)

    return shifted_input_ids

# 从 transformers.models.bart.modeling_tf_bart 中复制的函数，用于创建自回归模型的蒙版
# 创建用于双向自注意力的因果掩码
def _make_causal_mask(input_ids_shape: tf.TensorShape, past_key_values_length: int = 0):
    bsz = input_ids_shape[0]
    tgt_len = input_ids_shape[1]
    创建全为负数的掩码
    mask = tf.ones((tgt_len, tgt_len)) * LARGE_NEGATIVE
    mask_cond = tf.range(shape_list(mask)[-1])

    # 根据掩码条件生成掩码矩阵
    mask = tf.where(mask_cond < tf.reshape(mask_cond + 1, (shape_list(mask)[-1], 1)), 0.0, mask)

    # 如果过去的键值长度大于0，则在掩码前添加额外的零
    if past_key_values_length > 0:
        mask = tf.concat([tf.zeros((tgt_len, past_key_values_length)), mask], axis=-1)

    return tf.tile(mask[None, None, :, :], (bsz, 1, 1, 1))


# 从`[bsz, seq_len]`扩展到`[bsz, 1, tgt_seq_len, src_seq_len]`的注意力掩码
def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None):
    src_len = shape_list(mask)[1]
    tgt_len = tgt_len if tgt_len is not None else src_len
    one_cst = tf.constant(1.0)
    mask = tf.cast(mask, dtype=one_cst.dtype)
    扩展掩码矩阵
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))

    return (one_cst - expanded_mask) * LARGE_NEGATIVE


# 生成正弦位置嵌入层，用于产生任意长度的正弦位置嵌入
class TFPegasusSinusoidalPositionalEmbedding(tf.keras.layers.Layer):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, **kwargs):
        super().__init__(**kwargs)

        if embedding_dim % 2 != 0:
            raise NotImplementedError(f"odd embedding_dim {embedding_dim} not supported")

        self.embedding_dim = embedding_dim
        self.num_positions = num_positions

    def build(self, input_shape: tf.TensorShape):
        """
        构建共享令牌嵌入层，Shared weights logic adapted from
        https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24
        """

        weight = self._init_weight(self.num_positions, self.embedding_dim)

        self.weight = self.add_weight(
            name="embeddings",
            shape=[self.num_positions, self.embedding_dim],
        )
        weight = tf.cast(weight, dtype=self.weight.dtype)

        self.weight.assign(weight)

        super().build(input_shape)

    @staticmethod
    # 初始化位置编码权重，用于产生位置编码向量
    def _init_weight(n_pos: int, dim: int):
        # 根据公式计算位置编码值
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        # 创建一个全零的位置编码权重表
        table = np.zeros_like(position_enc)
        # 为权重表的前半部分赋值为 sin 函数的值
        table[:, 0 : dim // 2] = np.sin(position_enc[:, 0::2])
        # 为权重表的后半部分赋值为 cos 函数的值
        table[:, dim // 2 :] = np.cos(position_enc[:, 1::2])
        # 将 NumPy 数组转换为 TensorFlow 张量
        table = tf.convert_to_tensor(table)
        # 停止对 table 张量的梯度计算
        tf.stop_gradient(table)
        # 返回位置编码权重表
        return table
    
    # 根据输入序列长度以及可选的 past_key_values_length 计算位置 ID，并从位置编码权重表中获取对应的位置编码向量
    def call(
        self, input_shape: tf.TensorShape, past_key_values_length: int = 0, position_ids: tf.Tensor | None = None
    ):
        # 如果未传入 position_ids，则根据输入序列长度计算位置 ID
        if position_ids is None:
            seq_len = input_shape[1]
            position_ids = tf.range(past_key_values_length, seq_len + past_key_values_length, delta=1, name="range")
        # 从位置编码权重表中获取对应的位置编码向量
        return tf.gather(self.weight, position_ids)
# 从transformers.models.bart.modeling_tf_bart.TFBartAttention复制得到TFPegasusAttention类，将Bart改为Pegasus
class TFPegasusAttention(tf.keras.layers.Layer):
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
        # 初始化函数，设定参数和属性
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.head_dim = embed_dim // num_heads
        if (self.head_dim * num_heads) != self.embed_dim:
            # 如果embed_dim不能整除num_heads，抛出异常
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="k_proj")
        self.q_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="q_proj")
        self.v_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="v_proj")
        self.out_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="out_proj")

    # 定义形状函数，用于调整数据张量的形状
    def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
        return tf.transpose(tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))

    # 实现调用函数，处理注意力操作的前向传播
    def call(
        self,
        hidden_states: tf.Tensor,
        key_value_states: tf.Tensor | None = None,
        past_key_value: Tuple[Tuple[tf.Tensor]] | None = None,
        attention_mask: tf.Tensor | None = None,
        layer_head_mask: tf.Tensor | None = None,
        training: Optional[bool] = False,
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建网络层
        if getattr(self, "k_proj", None) is not None:
            with tf.name_scope(self.k_proj.name):
                self.k_proj.build([None, None, self.embed_dim])
        if getattr(self, "q_proj", None) is not None:
            with tf.name_scope(self.q_proj.name):
                self.q_proj.build([None, None, self.embed_dim])
        if getattr(self, "v_proj", None) is not None:
            with tf.name_scope(self.v_proj.name):
                self.v_proj.build([None, None, self.embed_dim])
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.embed_dim])


# 从transformers.models.mbart.modeling_tf_mbart.TFMBartEncoderLayer复制得到TFPegasusEncoderLayer类，将MBart改为Pegasus
class TFPegasusEncoderLayer(tf.keras.layers.Layer):
    # 初始化函数，接受 PegasusConfig 类型的参数和其他关键字参数
    def __init__(self, config: PegasusConfig, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 设置嵌入维度为配置中的 d_model
        self.embed_dim = config.d_model
        # 创建 self_attention 层对象
        self.self_attn = TFPegasusAttention(
            self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout, name="self_attn"
        )
        # 创建 self_attention 层后的层归一化对象
        self.self_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        # 创建 dropout 层
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        # 获取激活函数
        self.activation_fn = get_tf_activation(config.activation_function)
        # 创建激活函数后的 dropout 层
        self.activation_dropout = tf.keras.layers.Dropout(config.activation_dropout)
        # 创建全连接层 fc1
        self.fc1 = tf.keras.layers.Dense(config.encoder_ffn_dim, name="fc1")
        # 创建全连接层 fc2
        self.fc2 = tf.keras.layers.Dense(self.embed_dim, name="fc2")
        # 创建最终层归一化对象
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        # 保存配置
        self.config = config

    # 调用函数，接受隐藏状态、注意力掩码、层头掩码和训练参数
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        layer_head_mask: tf.Tensor,
        training: Optional[bool] = False,
    ):
        """
        Args:
            hidden_states (`tf.Tensor`): 输入到该层的张量，形状为 *(batch, seq_len, embed_dim)*
            attention_mask (`tf.Tensor`): 大小为*(batch, 1, tgt_len, src_len)*的注意力掩码，
                其中填充元素由非常大的负值表示。
            layer_head_mask (`tf.Tensor`): 给定层中的注意头的掩码，大小为 *(encoder_attention_heads,)*
        """
        # 保存隐藏状态以便残差连接
        residual = hidden_states
        # 对隐藏状态进行层归一化
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 进行 self-attention 操作
        hidden_states, self_attn_weights, _ = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, layer_head_mask=layer_head_mask
        )

        # 检查 self attention 操作是否修改了隐藏状态的形状
        tf.debugging.assert_equal(
            shape_list(hidden_states),
            shape_list(residual),
            message=f"Self attn modified the shape of query {shape_list(residual)} to {shape_list(hidden_states)}",
        )

        # 对隐藏状态进行 dropout 操作
        hidden_states = self.dropout(hidden_states, training=training)
        # 残差连接
        hidden_states = residual + hidden_states

        # 保存隐藏状态以便残差连接
        residual = hidden_states
        # 对隐藏状态进行最终层归一化
        hidden_states = self.final_layer_norm(hidden_states)
        # 使用激活函数处理全连接层 fc1
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 对全连接层 fc1 的输出进行 dropout
        hidden_states = self.activation_dropout(hidden_states, training=training)
        # 使用全连接层 fc2 处理隐藏状态
        hidden_states = self.fc2(hidden_states)
        # 对全连接层 fc2 的输出进行 dropout
        hidden_states = self.dropout(hidden_states, training=training)
        # 残差连接
        hidden_states = residual + hidden_states

        # 返回隐藏状态和 self attention 权重
        return hidden_states, self_attn_weights
    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已经被构建过, 则直接返回
        if self.built:
            return
        # 设置模型已被构建的标志
        self.built = True
        
        # 构建自注意力层
        if getattr(self, "self_attn", None) is not None:
            # 设置自注意力层的命名空间
            with tf.name_scope(self.self_attn.name):
                # 构建自注意力层
                self.self_attn.build(None)
        
        # 构建自注意力层归一化层
        if getattr(self, "self_attn_layer_norm", None) is not None:
            # 设置自注意力层归一化层的命名空间
            with tf.name_scope(self.self_attn_layer_norm.name):
                # 构建自注意力层归一化层, 输入形状为 [None, None, self.embed_dim]
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        
        # 构建全连接层 1
        if getattr(self, "fc1", None) is not None:
            # 设置全连接层 1 的命名空间
            with tf.name_scope(self.fc1.name):
                # 构建全连接层 1, 输入形状为 [None, None, self.embed_dim]
                self.fc1.build([None, None, self.embed_dim])
        
        # 构建全连接层 2
        if getattr(self, "fc2", None) is not None:
            # 设置全连接层 2 的命名空间
            with tf.name_scope(self.fc2.name):
                # 构建全连接层 2, 输入形状为 [None, None, self.config.encoder_ffn_dim]
                self.fc2.build([None, None, self.config.encoder_ffn_dim])
        
        # 构建最终层归一化层
        if getattr(self, "final_layer_norm", None) is not None:
            # 设置最终层归一化层的命名空间
            with tf.name_scope(self.final_layer_norm.name):
                # 构建最终层归一化层, 输入形状为 [None, None, self.embed_dim]
                self.final_layer_norm.build([None, None, self.embed_dim])
# 从transformers.models.mbart.modeling_tf_mbart.TFMBartDecoderLayer复制并修改为Pegasus
class TFPegasusDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: PegasusConfig, **kwargs):
        # 调用父类构造函数
        super().__init__(**kwargs)
        # 获取嵌入维度
        self.embed_dim = config.d_model
        # 创建Pegasus自注意力层对象
        self.self_attn = TFPegasusAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="self_attn",
            is_decoder=True,
        )
        # 添加丢弃层，用于随机丢弃一部分神经元
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        # 获取激活函数
        self.activation_fn = get_tf_activation(config.activation_function)
        # 添加激活函数的丢弃层
        self.activation_dropout = tf.keras.layers.Dropout(config.activation_dropout)

        # 添加自注意力层规范化层
        self.self_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        # 创建Pegasus编码器注意力层对象
        self.encoder_attn = TFPegasusAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="encoder_attn",
            is_decoder=True,
        )
        # 添加编码器注意力层规范化层
        self.encoder_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="encoder_attn_layer_norm")
        # 添加全连接层1
        self.fc1 = tf.keras.layers.Dense(config.decoder_ffn_dim, name="fc1")
        # 添加全连接层2
        self.fc2 = tf.keras.layers.Dense(self.embed_dim, name="fc2")
        # 添加最终规范化层
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        # 保存配置
        self.config = config

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None = None,
        encoder_hidden_states: tf.Tensor | None = None,
        encoder_attention_mask: tf.Tensor | None = None,
        layer_head_mask: tf.Tensor | None = None,
        cross_attn_layer_head_mask: tf.Tensor | None = None,
        past_key_value: Tuple[tf.Tensor] | None = None,
        training: Optional[bool] = False,
    # 构建模型，如果已经构建过了则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 标记模型已构建
        self.built = True
        # 如果存在自注意力层，则构建自注意力层
        if getattr(self, "self_attn", None) is not None:
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)
        # 如果存在自注意力层的层归一化，则构建该层归一化
        if getattr(self, "self_attn_layer_norm", None) is not None:
            with tf.name_scope(self.self_attn_layer_norm.name):
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        # 如果存在编码器注意力层，则构建编码器注意力层
        if getattr(self, "encoder_attn", None) is not None:
            with tf.name_scope(self.encoder_attn.name):
                self.encoder_attn.build(None)
        # 如果存在编码器注意力层的层归一化，则构建该层归一化
        if getattr(self, "encoder_attn_layer_norm", None) is not None:
            with tf.name_scope(self.encoder_attn_layer_norm.name):
                self.encoder_attn_layer_norm.build([None, None, self.embed_dim])
        # 如果存在第一个全连接层，则构建该层
        if getattr(self, "fc1", None) is not None:
            with tf.name_scope(self.fc1.name):
                self.fc1.build([None, None, self.embed_dim])
        # 如果存在第二个全连接层，则构建该层
        if getattr(self, "fc2", None) is not None:
            with tf.name_scope(self.fc2.name):
                self.fc2.build([None, None, self.config.decoder_ffn_dim])
        # 如果存在最终的层归一化层，则构建该层归一化
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.embed_dim])
class TFPegasusPreTrainedModel(TFPreTrainedModel):
    # 定义一个 Pegasus 预训练模型类，继承自 TFPreTrainedModel
    config_class = PegasusConfig
    # 将 config 类设置为 PegasusConfig
    base_model_prefix = "model"
    # 设置基础模型前缀为 "model"


PEGASUS_START_DOCSTRING = r"""
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
        config ([`PegasusConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""
# Pegasus 模型文档字符串的起始部分，提供了模型继承、使用说明以及输入格式的提示，以及配置参数说明

PEGASUS_GENERATION_EXAMPLE = r"""
    Summarization example:

    ```python
    >>> from transformers import AutoTokenizer, TFPegasusForConditionalGeneration

    >>> model = TFPegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")

    >>> ARTICLE_TO_SUMMARIZE = (


# Pegasus 模型生成示例的字符串，提供了摘要生成的示例代码
    ...     "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
    ...     "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
    ...     "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
    ... )
    >>> inputs = tokenizer(ARTICLE_TO_SUMMARIZE, max_length=1024, return_tensors="tf")

    # 生成摘要
    >>> summary_ids = model.generate(input_ids)
    # 解码生成的摘要，跳过特殊标记并清除标记化空格
    >>> print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    ```py
# 这是 PEGASUS 模型的输入文档字符串
PEGASUS_INPUTS_DOCSTRING = r"""
"""

# 这是 PEGASUS Encoder 的 Keras 序列化类
@keras_serializable
class TFPegasusEncoder(tf.keras.layers.Layer):
    # 设置配置类为 PegasusConfig
    config_class = PegasusConfig
    """
    PEGASUS 编码器由 config.encoder_layers 个自注意力层组成。每个层都是 TFPegasusEncoderLayer。

    参数:
        config: PegasusConfig
    """

    def __init__(self, config: PegasusConfig, embed_tokens: Optional[tf.keras.layers.Embedding] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        # 创建 Dropout 层
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        # 设置 encoder 层的 layer drop 概率
        self.layerdrop = config.encoder_layerdrop
        # 设置填充 token ID
        self.padding_idx = config.pad_token_id
        # 设置最大输入长度
        self.max_source_positions = config.max_position_embeddings
        # 如果 config.scale_embedding 为 True，则缩放输入 embedding 
        self.embed_scale = tf.math.sqrt(float(config.d_model)) if config.scale_embedding else 1.0

        # 设置 token embedding 层
        self.embed_tokens = embed_tokens
        # 创建正弦位置 embedding 层
        self.embed_positions = TFPegasusSinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            name="embed_positions",
        )
        # 创建 encoder 层列表
        self.layers = [TFPegasusEncoderLayer(config, name=f"layers.{i}") for i in range(config.encoder_layers)]
        # 创建最终的 layer norm 层
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layer_norm")

    # 获取 token embedding 层
    def get_embed_tokens(self):
        return self.embed_tokens

    # 设置 token embedding 层
    def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens

    # 编码器前向传播
    @unpack_inputs
    def call(
        self,
        input_ids: tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        attention_mask: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ):
        pass

    # 构建编码器层
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "embed_positions", None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.d_model])
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)

# PEGASUS 解码器的 Keras 序列化类
@keras_serializable
class TFPegasusDecoder(tf.keras.layers.Layer):
    # 设置配置类为 PegasusConfig
    config_class = PegasusConfig
    """
    PEGASUS 解码器由 config.decoder_layers 个解码层组成。每个层都是 TFPegasusDecoderLayer。

    参数:
        config: PegasusConfig
        embed_tokens: 输出 embedding
    """
    def __init__(self, config: PegasusConfig, embed_tokens: Optional[tf.keras.layers.Embedding] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.embed_tokens = embed_tokens
        self.layerdrop = config.decoder_layerdrop
        self.embed_positions = TFPegasusSinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            name="embed_positions",
        )
        self.embed_scale = tf.math.sqrt(float(config.d_model)) if config.scale_embedding else 1.0
        self.layers = [TFPegasusDecoderLayer(config, name=f"layers.{i}") for i in range(config.decoder_layers)]
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layer_norm")
        self.dropout = tf.keras.layers.Dropout(config.dropout)

根据 PegasusConfig 和可选的 tf.keras.layers.Embedding 对象初始化。
- config : PegasusConfig 对象，包含模型的配置信息。
- embed_tokens : 可选的 tf.keras.layers.Embedding 对象，用于编码输入的标记。
- padding_idx : 配置信息中的 pad_token_id。
- layerdrop : 配置信息中的 decoder_layerdrop。
- embed_positions : TFPegasusSinusoidalPositionalEmbedding 对象，用于位置编码。
- embed_scale : 根据 d_model 计算得出的缩放因子，如果 scale_embedding 为真，则是 d_model 的平方根，否则为 1.0。
- layers : TFPegasusDecoderLayer 对象列表，存储 decoder 层。
- layer_norm : LayerNormalization 对象，将输入进行标准化处理。
- dropout : Dropout 对象，用于进行 dropout 操作。


    def get_embed_tokens(self):
        return self.embed_tokens

获取 embed_tokens 对象的方法。


   def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens

设置 embed_tokens 对象的方法。


    @unpack_inputs
    def call(
        self,
        input_ids: tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        attention_mask: tf.Tensor | None = None,
        position_ids: tf.Tensor | None = None,
        encoder_hidden_states: tf.Tensor | None = None,
        encoder_attention_mask: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        cross_attn_head_mask: tf.Tensor | None = None,
        past_key_values: Tuple[Tuple[tf.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,

定义 call 方法，用于进行前向传播。
- input_ids : 输入标记的张量。
- inputs_embeds : 输入的嵌入张量。
- attention_mask : 注意力掩码张量。
- position_ids : 位置标记的张量。
- encoder_hidden_states : 编码器隐藏状态的张量。
- encoder_attention_mask : 编码器的注意力掩码张量。
- head_mask : 注意力头的掩码张量。
- cross_attn_head_mask : 跨注意力头的掩码张量。
- past_key_values : 用于缓存的键值对的元组。
- use_cache : 是否使用缓存。
- output_attentions : 是否输出注意力权重。
- output_hidden_states : 是否输出隐藏状态。
- return_dict : 是否以字典的形式返回结果。
- training : 是否在训练模式下。


    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "embed_positions", None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.d_model])
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)

构建模型。
- 如果 embed_positions 对象存在，则构建 embed_positions。
- 如果 layer_norm 对象存在，则构建 layer_norm。
- 如果 layers 列表存在，则依次构建每个 layer。
# 使用 keras_serializable 装饰器将该类标记为 Keras 序列化对象，以便于模型保存和加载
@keras_serializable
# 定义 TFPegasusMainLayer 类，继承自 tf.keras.layers.Layer 类
class TFPegasusMainLayer(tf.keras.layers.Layer):
    # 配置类属性指向 PegasusConfig 类
    config_class = PegasusConfig

    # 初始化方法
    def __init__(self, config: PegasusConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 将传入的配置参数保存到对象属性中
        self.config = config
        # 创建一个共享的 Embedding 层，用于输入和输出的 token 表示
        self.shared = tf.keras.layers.Embedding(
            # 设置输入维度为词汇表大小
            input_dim=config.vocab_size,
            # 设置输出维度为模型维度大小
            output_dim=config.d_model,
            # 设置初始化方式为 TruncatedNormal 初始化，标准差为配置中的 init_std
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=self.config.init_std),
            # 设置层的名称为 "model.shared"
            name="model.shared",
        )
        # 添加一个额外的属性来指定层的预期名称作用域（用于加载/存储权重）
        self.shared.load_weight_prefix = "model.shared"

        # 创建 Pegasus 编码器层
        self.encoder = TFPegasusEncoder(config, self.shared, name="encoder")
        # 创建 Pegasus 解码器层
        self.decoder = TFPegasusDecoder(config, self.shared, name="decoder")

    # 获取输入 Embedding 层的方法
    def get_input_embeddings(self):
        return self.shared

    # 设置输入 Embedding 层的方法
    def set_input_embeddings(self, new_embeddings):
        # 更新共享的 Embedding 层
        self.shared = new_embeddings
        # 更新编码器的 Embedding 层
        self.encoder.embed_tokens = self.shared
        # 更新解码器的 Embedding 层
        self.decoder.embed_tokens = self.shared

    # 定义调用方法
    @unpack_inputs
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
        past_key_values: Tuple[Tuple[tf.Tensor]] = None,
        inputs_embeds: tf.Tensor | None = None,
        decoder_inputs_embeds: tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
        **kwargs,
):
        # 如果解码器的输入 ID 和嵌入向量都为 None，则不使用缓存
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            use_cache = False

        # 如果用户未指定输出隐藏状态，则使用模型配置中的默认设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # 如果编码器输出为 None，则使用编码器进行前向传播
        if encoder_outputs is None:
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
        # 如果用户传递的是编码器输出的元组，并且设置了 return_dict=True，则将其包装在 TFBaseModelOutput 中
        elif return_dict and not isinstance(encoder_outputs, TFBaseModelOutput):
            encoder_outputs = TFBaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        # 如果用户传递了 TFBaseModelOutput 作为编码器输出，并且设置了 return_dict=False，则将其包装在元组中
        elif not return_dict and not isinstance(encoder_outputs, tuple):
            encoder_outputs = encoder_outputs.to_tuple()

        # 使用解码器进行前向传播
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

        # 如果 return_dict 为 False，则将解码器输出和编码器输出组合返回
        if not return_dict:
            return decoder_outputs + encoder_outputs

        # 返回 TFSeq2SeqModelOutput 类型的对象，包括解码器和编码器的相关隐藏状态和注意力权重
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
```  
    def build(self, input_shape=None):
        # 如果模型已经构建完成，则直接返回，不做任何操作
        if self.built:
            return
        # 标记模型为已构建状态
        self.built = True
        # 共享/绑定的权重需要在模型基本命名空间中
        # 将"/"添加到名称范围的末尾（而不是开头！）可以将其放在根命名空间而不是当前命名空间中。
        with tf.name_scope(self.shared.load_weight_prefix + "/" + self.shared.name + "/"):
            # 构建共享层
            self.shared.build(None)
        # 如果存在编码器，则构建编码器
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果存在解码器，则构建解码器
        if getattr(self, "decoder", None) is not None:
            with tf.name_scope(self.decoder.name):
                self.decoder.build(None)
# 使用给定的文档字符串添加起始文档字符串
@add_start_docstrings(
    "The bare PEGASUS Model outputting raw hidden-states without any specific head on top.",
    PEGASUS_START_DOCSTRING,
)
# 定义一个 TF Pegasus 模型类，继承自 TF Pegasus 预训练模型类
class TFPegasusModel(TFPegasusPreTrainedModel):
    # 初始化方法
    def __init__(self, config: PegasusConfig, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 创建 Pegasus 主层，并命名为 "model"
        self.model = TFPegasusMainLayer(config, name="model")

    # 获取编码器的方法
    def get_encoder(self):
        return self.model.encoder

    # 获取解码器的方法
    def get_decoder(self):
        return self.model.decoder

    # 定义模型调用方法
    @unpack_inputs
    # 使用给定的文档字符串添加模型前向传播的起始文档字符串
    @add_start_docstrings_to_model_forward(PEGASUS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 使用给定的代码示例文档字符串添加模型前向传播的代码示例文档字符串
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSeq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        decoder_input_ids: np.ndarray | tf.Tensor | None = None,
        decoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        decoder_position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        decoder_head_mask: np.ndarray | tf.Tensor | None = None,
        cross_attn_head_mask: np.ndarray | tf.Tensor | None = None,
        encoder_outputs: Optional[Union[Tuple, TFBaseModelOutput]] = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        decoder_inputs_embeds: np.ndarray | tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        **kwargs,
    ) -> Union[TFSeq2SeqModelOutput, Tuple[tf.Tensor]]:
        # 调用 Pegasus 主层的前向传播方法
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
        # 如果配置中使用了缓存，则获取过去的键值对
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        # 如果配置中输出了隐藏状态，则转换输出的解码器隐藏状态为张量
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置中输出了注意力权重，则转换输出的解码器注意力权重为张量
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        # 如果配置中输出了注意力权重，则转换输出的交叉注意力权重为张量
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        # 如果配置中输出了隐藏状态，则转换输出的编码器隐藏状态为张量
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置中输出了注意力权重，则转换输出的编码器注意力权重为张量
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None

        # 返回一个 TFSeq2SeqModelOutput 对象，包含输出的各项内容
        return TFSeq2SeqModelOutput(
            last_hidden_state=output.last_hidden_state,
            past_key_values=pkv,
            decoder_hidden_states=dec_hs,
            decoder_attentions=dec_attns,
            cross_attentions=cross_attns,
            encoder_last_hidden_state=output.encoder_last_hidden_state,
            encoder_hidden_states=enc_hs,
            encoder_attentions=enc_attns,
        )

    # 构建模型
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 设置为已构建状态
        self.built = True
        # 如果模型已存在，则在模型的名称范围内构建模型
        if getattr(self, "model", None) is not None:
            with tf.name_scope(self.model.name):
                self.model.build(None)
# 该类是 BiasLayer 类，用于序列化目的。它被用作一个层，用于存储偏差权重。
class BiasLayer(tf.keras.layers.Layer):
    """
    Bias as a layer. It is used for serialization purposes: `tf.keras.Model.save_weights` stores on a per-layer basis,
    so all weights have to be registered in a layer.
    """

    def __init__(self, shape, initializer, trainable, name, **kwargs):
        # 初始化该层，设置偏差权重的形状、初始化方式、是否可训练以及层的名称
        super().__init__(name=name, **kwargs)
        # 创建偏差权重，注意这个权重在序列化时不会被作用域限制
        self.bias = self.add_weight(name=name, shape=shape, initializer=initializer, trainable=trainable)

    def call(self, x):
        # 在输入 x 上应用偏差并返回
        return x + self.bias


# 该类是 TFPegasusForConditionalGeneration 类，它是 PEGASUS 模型的实现，可用于文本摘要任务。
@add_start_docstrings(
    "The PEGASUS Model with a language modeling head. Can be used for summarization.",
    PEGASUS_START_DOCSTRING,
)
class TFPegasusForConditionalGeneration(TFPegasusPreTrainedModel, TFCausalLanguageModelingLoss):
    # 在加载模型时需要忽略的一些意外键
    _keys_to_ignore_on_load_unexpected = [
        r"model.encoder.embed_tokens.weight",
        r"model.decoder.embed_tokens.weight",
    ]

    def __init__(self, config, *inputs, **kwargs):
        # 初始化该模型，包括创建 TFPegasusMainLayer 实例和 BiasLayer 实例
        super().__init__(config, *inputs, **kwargs)
        self.model = TFPegasusMainLayer(config, name="model")
        self.use_cache = config.use_cache
        self.bias_layer = BiasLayer(
            name="final_logits_bias", shape=[1, config.vocab_size], initializer="zeros", trainable=False
        )

    def get_decoder(self):
        # 返回模型的解码器
        return self.model.decoder

    def get_encoder(self):
        # 返回模型的编码器
        return self.model.encoder

    def get_output_embeddings(self):
        # 返回模型的输出嵌入层
        return self.get_input_embeddings()

    def set_output_embeddings(self, value):
        # 设置模型的输出嵌入层
        self.set_input_embeddings(value)

    def get_bias(self):
        # 返回模型的偏差层
        return {"final_logits_bias": self.bias_layer.bias}

    def set_bias(self, value):
        # 设置模型的偏差层
        vocab_size = value["final_logits_bias"].shape[-1]
        self.bias_layer = BiasLayer(
            name="final_logits_bias", shape=[1, vocab_size], initializer="zeros", trainable=False
        )
        self.bias_layer.bias.assign(value["final_logits_bias"])
    # 定义一个方法，用于调用Transformer模型，接受各种输入参数
    def call(
        self,
        # 输入的token IDs，可以是TensorFlow模型的输入类型或None
        input_ids: TFModelInputType | None = None,
        # 注意力掩码，可以是NumPy数组、TensorFlow张量或None
        attention_mask: np.ndarray | tf.Tensor | None = None,
        # 解码器的输入token IDs，可以是NumPy数组、TensorFlow张量或None
        decoder_input_ids: np.ndarray | tf.Tensor | None = None,
        # 解码器的注意力掩码，可以是NumPy数组、TensorFlow张量或None
        decoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        # 解码器的位置IDs，可以是NumPy数组、TensorFlow张量或None
        decoder_position_ids: np.ndarray | tf.Tensor | None = None,
        # 头部掩码，可以是NumPy数组、TensorFlow张量或None
        head_mask: np.ndarray | tf.Tensor | None = None,
        # 解码器头部掩码，可以是NumPy数组、TensorFlow张量或None
        decoder_head_mask: np.ndarray | tf.Tensor | None = None,
        # 跨注意力头部掩码，可以是NumPy数组、TensorFlow张量或None
        cross_attn_head_mask: np.ndarray | tf.Tensor | None = None,
        # 编码器输出，可选的Transformer模型输出类型
        encoder_outputs: Optional[TFBaseModelOutput] = None,
        # 过去的键值对，可选的元组类型，包含NumPy数组或TensorFlow张量
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        # 输入的嵌入向量，可以是NumPy数组、TensorFlow张量或None
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        # 解码器的输入嵌入向量，可以是NumPy数组、TensorFlow张量或None
        decoder_inputs_embeds: np.ndarray | tf.Tensor | None = None,
        # 是否使用缓存，可选的布尔类型，默认为None
        use_cache: Optional[bool] = None,
        # 是否输出注意力权重，可选的布尔类型，默认为None
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，可选的布尔类型，默认为None
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典形式的输出，可选的布尔类型，默认为None
        return_dict: Optional[bool] = None,
        # 标签，可以是NumPy数组或TensorFlow张量，用于训练时的标签
        labels: np.ndarray | tf.Tensor | None = None,
        # 是否处于训练模式，布尔类型，默认为False
        training: bool = False,
    ) -> Union[TFSeq2SeqLMOutput, Tuple[tf.Tensor]]:
        """
        定义函数的输入参数和返回值的类型注解

        labels (`tf.tensor` of shape `(batch_size, sequence_length)`, *optional*):
            用于计算掩码语言建模损失的标签。索引应该在 `[0, ..., config.vocab_size]` 或 -100 之间（参见 `input_ids` 的文档字符串）。索引设置为 `-100` 的标记将被忽略（掩盖），损失仅计算具有标签在 `[0, ..., config.vocab_size]` 中的标记。

        Returns:
            函数的返回值说明

        """

        if labels is not None:
            labels = tf.where(
                labels == self.config.pad_token_id,
                tf.cast(tf.fill(shape_list(labels), -100), labels.dtype),
                labels,
            )
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        lm_logits = tf.matmul(outputs[0], self.model.shared.weights, transpose_b=True)
        lm_logits = self.bias_layer(lm_logits)
        masked_lm_loss = None if labels is None else self.hf_compute_loss(labels, lm_logits)

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        return TFSeq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,  # index 1 of d outputs
            decoder_hidden_states=outputs.decoder_hidden_states,  # index 2 of d outputs
            decoder_attentions=outputs.decoder_attentions,  # index 3 of d outputs
            cross_attentions=outputs.cross_attentions,  # index 4 of d outputs
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,  # index 0 of encoder outputs
            encoder_hidden_states=outputs.encoder_hidden_states,  # 1 of e out
            encoder_attentions=outputs.encoder_attentions,  # 2 of e out
        )

    # 从transformers.models.bart.modeling_tf_bart.TFBartForConditionalGeneration.serving_output复制而来
    # 该方法用于对模型的输出进行后处理，生成最终的序列生成输出
    def serving_output(self, output):
        # 如果配置了使用缓存，则从输出中提取过去的关键值
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        # 如果配置了输出隐藏状态，则从输出中提取解码器隐藏状态
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置了输出注意力权重，则从输出中提取解码器注意力权重
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        # 如果配置了输出注意力权重，则从输出中提取交叉注意力权重
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        # 如果配置了输出隐藏状态，则从输出中提取编码器隐藏状态
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置了输出注意力权重，则从输出中提取编码器注意力权重
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None
    
        # 返回一个 TFSeq2SeqLMOutput 对象，包含模型的输出结果
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
    
    # 该方法用于准备用于生成输出的输入数据
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
        # 如果使用了过去的关键值，则只需要最新的decoder_input_ids
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
    
        # 如果提供了decoder_attention_mask，则计算decoder_position_ids
        if decoder_attention_mask is not None:
            decoder_position_ids = tf.math.cumsum(decoder_attention_mask, axis=-1, exclusive=True)[:, -1:]
        # 如果使用了过去的关键值，则decoder_position_ids为过去关键值的shape[2]
        elif past_key_values is not None:
            decoder_position_ids = past_key_values[0][0].shape[2]
        # 否则，decoder_position_ids为decoder_input_ids的序列长度
        else:
            decoder_position_ids = tf.range(decoder_input_ids.shape[1])
    
        # 返回一个字典，包含用于生成输出的所有必要输入
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
    
    # 该方法用于从标签中准备decoder_input_ids
    def prepare_decoder_input_ids_from_labels(self, labels: tf.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
    # 检查是否已经构建过该层，若已构建则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 标记该层为已构建状态
        self.built = True
        # 如果该层包含一个子模型
        if getattr(self, "model", None) is not None:
            # 在 TensorFlow 中使用指定名称空间创建该子模型的构建过程
            with tf.name_scope(self.model.name):
                # 构建子模型
                self.model.build(None)
        # 如果该层包含一个偏置层
        if getattr(self, "bias_layer", None) is not None:
            # 在 TensorFlow 中使用指定名称空间创建该偏置层的构建过程
            with tf.name_scope(self.bias_layer.name):
                # 构建偏置层
                self.bias_layer.build(None)
```