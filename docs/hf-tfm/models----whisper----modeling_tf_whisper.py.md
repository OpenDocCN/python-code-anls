# `.\transformers\models\whisper\modeling_tf_whisper.py`

```
# 设置脚本编码为 UTF-8
# 版权声明
# 版权所有 2022 年 OpenAI 作者和 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；
# 除非符合许可证，否则您不得使用此文件。
# 您可以在以下网址获得许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 按“原样”分发，不提供任何明示或暗示的保证
# 或适用于特定用途的条件。
# 有关详细信息，请参阅许可证。
""" TensorFlow Whisper 模型。"""

# 引入必要的库
from __future__ import annotations
import math
import random
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
# 导入相关模块和类
from ...activations_tf import get_tf_activation
from ...generation.configuration_utils import GenerationConfig
from ...generation.tf_logits_process import TFLogitsProcessorList
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFSeq2SeqLMOutput,
    TFSeq2SeqModelOutput,
)
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_whisper import WhisperConfig
from .tokenization_whisper import TASK_IDS, TO_LANGUAGE_CODE

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 用于文档的配置字符串
_CONFIG_FOR_DOC = "WhisperConfig"

# 预训练模型存档列表
TF_WHISPER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "openai/whisper-base",
    # 查看所有 Whisper 模型 https://huggingface.co/models?filter=whisper
]

# 用于生成大负数的常量
LARGE_NEGATIVE = -1e8

# 用于初始化位置嵌入的正弦函数
def sinusoidal_embedding_init(shape, dtype=tf.float32) -> tf.Tensor:
    """返回用于位置嵌入的正弦函数"""
    length, channels = shape
    if channels % 2 != 0:
        raise ValueError(
            f"正弦位置嵌入的通道数必须为 2 的倍数，当前通道数为 {channels}。"
        )
    log_timescale_increment = math.log(10000) / (channels // 2 - 1)
    inv_timescales = tf.exp(-log_timescale_increment * tf.range(channels // 2, dtype=tf.float32))
    scaled_time = tf.reshape(tf.range(length, dtype=tf.float32), (-1, 1)) * tf.reshape(inv_timescales, (1, -1))
    return tf.cast(tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1), dtype)

# 从 transformers.models.bart.modeling_tf_bart.shift_tokens_right 复制的函数
def shift_tokens_right(input_ids: tf.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """将输入的 tokens 向右移动一位"""
    pad_token_id = tf.cast(pad_token_id, input_ids.dtype)
    decoder_start_token_id = tf.cast(decoder_start_token_id, input_ids.dtype)
``` 
    # 创建一个形状为(input_ids的行数, 1)的张量，其中填充值为decoder_start_token_id
    start_tokens = tf.fill(
        (shape_list(input_ids)[0], 1), tf.convert_to_tensor(decoder_start_token_id, input_ids.dtype)
    )
    # 将start_tokens与input_ids的前n-1列拼接起来，形成shifted_input_ids
    shifted_input_ids = tf.concat([start_tokens, input_ids[:, :-1]], -1)
    # 将shifted_input_ids中可能的-100值替换为pad_token_id
    shifted_input_ids = tf.where(
        shifted_input_ids == -100,
        tf.fill(shape_list(shifted_input_ids), tf.convert_to_tensor(pad_token_id, input_ids.dtype)),
        shifted_input_ids,
    )

    # 断言shifted_input_ids里的值必须为正数或-100
    assert_gte0 = tf.debugging.assert_greater_equal(shifted_input_ids, tf.constant(0, dtype=input_ids.dtype))

    # 确保断言操作被调用，通过将结果包装在一个无操作的身份操作中
    with tf.control_dependencies([assert_gte0]):
        shifted_input_ids = tf.identity(shifted_input_ids)

    return shifted_input_ids
# 从transformers.models.bart.modeling_tf_bart._make_causal_mask中复制而来的函数，用于生成用于双向自注意力的因果掩码
def _make_causal_mask(input_ids_shape: tf.TensorShape, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    生成用于双向自注意力的因果掩码。
    """
    # 获取批量大小
    bsz = input_ids_shape[0]
    # 获取目标长度
    tgt_len = input_ids_shape[1]
    # 创建形状为(tgt_len, tgt_len)的由-LARGE_NEGATIVE填充的全1矩阵
    mask = tf.ones((tgt_len, tgt_len)) * LARGE_NEGATIVE
    # 创建掩码条件，形状为(mask的最后一个维度的长度)
    mask_cond = tf.range(shape_list(mask)[-1])

    # 将掩码的下三角部分置零
    mask = tf.where(mask_cond < tf.reshape(mask_cond + 1, (shape_list(mask)[-1], 1)), 0.0, mask)

    # 如果有过去的键值对长度大于0，则将全零部分拼接到掩码左侧
    if past_key_values_length > 0:
        mask = tf.concat([tf.zeros((tgt_len, past_key_values_length)), mask], axis=-1)

    return tf.tile(mask[None, None, :, :], (bsz, 1, 1, 1))


# 从transformers.models.bart.modeling_tf_bart._expand_mask中复制而来的函数，用于扩展掩码
def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    将注意力掩码从`[bsz, seq_len]`扩展到`[bsz, 1, tgt_seq_len, src_seq_len]`。
    """
    # 获取源序列长度
    src_len = shape_list(mask)[1]
    # 如果未指定目标长度，则将目标长度设为与源序列长度相同
    tgt_len = tgt_len if tgt_len is not None else src_len
    # 创建常数张量1
    one_cst = tf.constant(1.0)
    # 将掩码转换为与one_cst相同的数据类型
    mask = tf.cast(mask, dtype=one_cst.dtype)
    # 在第2和第3维度上复制掩码，使其形状为`[bsz, 1, tgt_seq_len, src_seq_len]`
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))

    return (one_cst - expanded_mask) * LARGE_NEGATIVE


class TFWhisperPositionalEmbedding(tf.keras.layers.Layer):
    def __init__(
        self,
        num_positions: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        embedding_initializer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # 初始化位置嵌入层
        self.num_positions = num_positions
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.embedding_initializer = tf.keras.initializers.get(embedding_initializer)

    def build(self, input_shape):
        # 添加权重
        self.weight = self.add_weight(
            name="weight",
            shape=[self.num_positions, self.embedding_dim],
            initializer=self.embedding_initializer,
            trainable=True,
        )
        super().build(input_shape)

    def call(self, input_ids, past_key_values_length=0):
        # 将过去键值对的长度转换为tf.int32类型
        past_key_values_length = tf.cast(past_key_values_length, tf.int32)
        # 生成gather的索引
        gather_indices = tf.range(tf.shape(input_ids)[1], delta=1) + past_key_values_length
        # 通过gather获取位置嵌入张量
        return tf.gather(self.weight, gather_indices)


class TFWhisperAttention(tf.keras.layers.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        **kwargs,
        # 调用父类构造方法初始化对象
        super().__init__(**kwargs)
        # 初始化嵌入维度
        self.embed_dim = embed_dim
        # 初始化头数
        self.num_heads = num_heads
        # 初始化丢弃层
        self.dropout = tf.keras.layers.Dropout(dropout)
        # 初始化头维度
        self.head_dim = embed_dim // num_heads

        # 检查头维度乘以头数是否等于嵌入维度，若不等则抛出 ValueError
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        # 初始化缩放因子
        self.scaling = self.head_dim**-0.5
        # 初始化是否为解码器标志
        self.is_decoder = is_decoder

        # 初始化 k 投影层
        self.k_proj = tf.keras.layers.Dense(embed_dim, use_bias=False, name="k_proj")
        # 初始化 v 投影层
        self.v_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="v_proj")
        # 初始化 q 投影层
        self.q_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="q_proj")
        # 初始化输出投影层
        self.out_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="out_proj")

    # 从transformers.models.bart.modeling_tf_bart.TFBartAttention._shape中复制而来，将tensor重塑为指定形状
    def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
        return tf.transpose(tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))

    # 从transformers.models.bart.modeling_tf_bart.TFBartAttention.call中复制而来，定义了注意力机制的计算过程
    def call(
        self,
        hidden_states: tf.Tensor,
        key_value_states: tf.Tensor | None = None,
        past_key_value: Tuple[Tuple[tf.Tensor]] | None = None,
        attention_mask: tf.Tensor | None = None,
        layer_head_mask: tf.Tensor | None = None,
        training: Optional[bool] = False,
    # 构建注意力层的方法，用来确认是否已经构建完成，并对 k 投影层、v 投影层、q 投影层和输出投影层进行构建
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "k_proj", None) is not None:
            with tf.name_scope(self.k_proj.name):
                self.k_proj.build([None, None, self.embed_dim])
        if getattr(self, "v_proj", None) is not None:
            with tf.name_scope(self.v_proj.name):
                self.v_proj.build([None, None, self.embed_dim])
        if getattr(self, "q_proj", None) is not None:
            with tf.name_scope(self.q_proj.name):
                self.q_proj.build([None, None, self.embed_dim])
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.embed_dim])
# 从transformers.models.speech_to_text.modeling_tf_speech_to_text.TFSpeech2TextEncoderLayer复制了代码，并将Speech2Text->Whisper
class TFWhisperEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: WhisperConfig, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = config.d_model  # 从config中获取嵌入维度
        self.self_attn = TFWhisperAttention(  # 创建自注意力层
            self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout, name="self_attn"
        )
        self.self_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")  # 创建自注意力层后的LayerNormalization
        self.dropout = tf.keras.layers.Dropout(config.dropout)  # 设置dropout层
        self.activation_fn = get_tf_activation(config.activation_function)  # 获取激活函数
        self.activation_dropout = tf.keras.layers.Dropout(config.activation_dropout)  # 设置激活函数的dropout层
        self.fc1 = tf.keras.layers.Dense(config.encoder_ffn_dim, name="fc1")  # 创建第一个全连接层
        self.fc2 = tf.keras.layers.Dense(self.embed_dim, name="fc2")  # 创建第二个全连接层
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")  # 创建最终的LayerNormalization
        self.config = config  # 存储配置信息

    def call(
        self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, layer_head_mask: tf.Tensor, training: bool = False
    ):
        """
        Args:
            hidden_states (`tf.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`tf.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`tf.Tensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`
        """
        residual = hidden_states  # 保存输入hidden_states的副本
        hidden_states = self.self_attn_layer_norm(hidden_states)  # 使用LayerNormalization处理输入hidden_states
        hidden_states, self_attn_weights, _ = self.self_attn(  # 通过自注意力层处理hidden_states，同时获取自注意力权重
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            training=training,
        )

        tf.debugging.assert_equal(  # 断言保证hidden_states和residual的形状相同
            shape_list(hidden_states),
            shape_list(residual),
            message=f"Self attn modified the shape of query {shape_list(residual)} to {shape_list(hidden_states)}",
        )

        hidden_states = self.dropout(hidden_states, training=training)  # 使用dropout层处理hidden_states
        hidden_states = residual + hidden_states  # 将处理后的hidden_states和residual相加

        residual = hidden_states  # 保存上一步操作后的hidden_states副本
        hidden_states = self.final_layer_norm(hidden_states)  # 使用LayerNormalization处理hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))  # 使用激活函数和第一个全连接层处理hidden_states
        hidden_states = self.activation_dropout(hidden_states, training=training)  # 使用dropout层处理激活函数后的hidden_states
        hidden_states = self.fc2(hidden_states)  # 使用第二个全连接层处理hidden_states
        hidden_states = self.dropout(hidden_states, training=training)  # 使用dropout层处理hidden_states
        hidden_states = residual + hidden_states  # 将处理后的hidden_states和residual相加

        return hidden_states, self_attn_weights  # 返回处理后的hidden_states和自注意力权重
    # 构建模型，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 设置标记为已构建
        self.built = True
        # 如果存在 self_attn，则构建 self_attn，并设置名字空间
        if getattr(self, "self_attn", None) is not None:
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)
        # 如果存在 self_attn_layer_norm，则构建 self_attn_layer_norm，并设置名字空间
        if getattr(self, "self_attn_layer_norm", None) is not None:
            with tf.name_scope(self.self_attn_layer_norm.name):
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        # 如果存在 fc1，则构建 fc1，并设置名字空间
        if getattr(self, "fc1", None) is not None:
            with tf.name_scope(self.fc1.name):
                self.fc1.build([None, None, self.embed_dim])
        # 如果存在 fc2，则构建 fc2，并设置名字空间
        if getattr(self, "fc2", None) is not None:
            with tf.name_scope(self.fc2.name):
                self.fc2.build([None, None, self.config.encoder_ffn_dim])
        # 如果存在 final_layer_norm，则构建 final_layer_norm，并设置名字空间
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.embed_dim])
# 从transformers.models.speech_to_text.modeling_tf_speech_to_text.TFSpeech2TextDecoderLayer复制并修改为Whisper
class TFWhisperDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: WhisperConfig, **kwargs):
        super().__init__(**kwargs)
        # 设置嵌入维度为config中的d_model
        self.embed_dim = config.d_model

        # 初始化self_attn为TFWhisperAttention对象，用于自注意力机制
        self.self_attn = TFWhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="self_attn",
            is_decoder=True,
        )
        # 添加丢弃层，根据config中的dropout设置
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        # 获取激活函数
        self.activation_fn = get_tf_activation(config.activation_function)
        # 添加激活函数的丢弃层，根据config中的activation_dropout设置
        self.activation_dropout = tf.keras.layers.Dropout(config.activation_dropout)

        # 初始化self_attn_layer_norm为LayerNormalization层，用于归一化自注意力机制的输出
        self.self_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        # 初始化encoder_attn为TFWhisperAttention对象，用于编码解码注意力机制
        self.encoder_attn = TFWhisperAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="encoder_attn",
            is_decoder=True,
        )
        # 初始化encoder_attn_layer_norm为LayerNormalization层，用于归一化编码解码注意力机制的输出
        self.encoder_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="encoder_attn_layer_norm")
        # 初始化fc1为全连接层，设置输出维度为config中的decoder_ffn_dim
        self.fc1 = tf.keras.layers.Dense(config.decoder_ffn_dim, name="fc1")
        # 初始化fc2为全连接层，设置输出维度为self.embed_dim
        self.fc2 = tf.keras.layers.Dense(self.embed_dim, name="fc2")
        # 初始化final_layer_norm为LayerNormalization层，用于归一化最终的输出
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        # 保存config
        self.config = config

    def call(
        self,
        hidden_states,
        attention_mask: tf.Tensor | None = None,
        encoder_hidden_states: tf.Tensor | None = None,
        encoder_attention_mask: tf.Tensor | None = None,
        layer_head_mask: tf.Tensor | None = None,
        cross_attn_layer_head_mask: tf.Tensor | None = None,
        past_key_value: Tuple[tf.Tensor] | None = None,
        training=False,
    # 构建神经网络模型，如果已经构建则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在注意力机制，构建自注意力层并指定名字空间
        if getattr(self, "self_attn", None) is not None:
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)
        # 如果存在自注意力层归一化，构建自注意力层归一化并指定名字空间
        if getattr(self, "self_attn_layer_norm", None) is not None:
            with tf.name_scope(self.self_attn_layer_norm.name):
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        # 如果存在编码器注意力机制，构建编码器注意力层并指定名字空间
        if getattr(self, "encoder_attn", None) is not None:
            with tf.name_scope(self.encoder_attn.name):
                self.encoder_attn.build(None)
        # 如果存在编码器注意力层归一化，构建编码器注意力层归一化并指定名字空间
        if getattr(self, "encoder_attn_layer_norm", None) is not None:
            with tf.name_scope(self.encoder_attn_layer_norm.name):
                self.encoder_attn_layer_norm.build([None, None, self.embed_dim])
        # 如果存在全连接层1，构建全连接层1并指定名字空间
        if getattr(self, "fc1", None) is not None:
            with tf.name_scope(self.fc1.name):
                self.fc1.build([None, None, self.embed_dim])
        # 如果存在全连接层2，构建全连接层2并指定名字空间
        if getattr(self, "fc2", None) is not None:
            with tf.name_scope(self.fc2.name):
                self.fc2.build([None, None, self.config.decoder_ffn_dim])
        # 如果存在最终的层归一化，构建最终的层归一化并指定名字空间
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.embed_dim])
class TFWhisperPreTrainedModel(TFPreTrainedModel):
    # 设置配置类为 WhisperConfig
    config_class = WhisperConfig
    # 模型基类的前缀为 "model"
    base_model_prefix = "model"
    # 主输入名称为 "input_features"
    main_input_name = "input_features"

    def _get_feat_extract_output_lengths(self, input_lengths: tf.Tensor) -> int:
        """
        Computes the output length of the convolutional layers
        计算卷积层的输出长度
        """
        # 计算输入长度
        input_lengths = (input_lengths - 1) // 2 + 1

        # 返回输出长度
        return input_lengths

    @property
    def dummy_inputs(self) -> Dict[str, tf.Tensor]:
        """
        Dummy inputs to build the network.

        Returns:
            `Dict[str, tf.Tensor]`: The dummy inputs.
        """
        # 返回虚拟输入，包括主输入和解码器输入
        return {
            self.main_input_name: tf.random.uniform(
                [1, self.config.num_mel_bins, self.config.max_source_positions * 2 - 1], dtype=tf.float32
            ),
            "decoder_input_ids": tf.constant([[1, 3]], dtype=tf.int32),
        }

    @property
    def input_signature(self):
        # 返回输入签名，指定了输入的形状和类型
        return {
            "input_features": tf.TensorSpec((None, self.config.num_mel_bins, None), tf.float32, name="input_features"),
            "decoder_input_ids": tf.TensorSpec((None, None), tf.int32, name="decoder_input_ids"),
            "decoder_attention_mask": tf.TensorSpec((None, None), tf.int32, name="decoder_attention_mask"),
        }


WHISPER_START_DOCSTRING = r"""
    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`WhisperConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

WHISPER_INPUTS_DOCSTRING = r"""
"""


@keras_serializable
class TFWhisperEncoder(tf.keras.layers.Layer):
    # 设置配置类为 WhisperConfig
    config_class = WhisperConfig
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`TFWhisperEncoderLayer`].

    Args:
        config: WhisperConfig
        embed_tokens (TFWhisperEmbedding): output embedding
    """
    # 初始化 Whisper 编码器类
    def __init__(self, config: WhisperConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 保存配置信息
        self.config = config
        # 获取编码器的 dropout 率
        self.layerdrop = config.encoder_layerdrop
    
        # 设置一些模型参数
        self.embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(self.embed_dim) if config.scale_embedding else 1.0
    
        # 创建第一个 1D 卷积层
        self.conv1 = tf.keras.layers.Conv1D(self.embed_dim, kernel_size=3, strides=1, padding="valid", name="conv1")
        # 创建第二个 1D 卷积层
        self.conv2 = tf.keras.layers.Conv1D(self.embed_dim, kernel_size=3, strides=2, padding="valid", name="conv2")
    
        # 创建位置编码层
        self.embed_positions = TFWhisperPositionalEmbedding(
            num_positions=self.max_source_positions,
            embedding_dim=self.embed_dim,
            embedding_initializer=sinusoidal_embedding_init,
            name="embed_positions",
        )
        self.embed_positions.trainable = False
    
        # 创建编码器层
        self.encoder_layers = [TFWhisperEncoderLayer(config, name=f"layers.{i}") for i in range(config.encoder_layers)]
        # 创建层归一化层
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layer_norm")
    
        # 创建 dropout 层
        self.dropout = tf.keras.layers.Dropout(config.dropout)
    
    # 实现编码器的前向传播
    @unpack_inputs
    def call(
        self,
        input_features=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
    ):
        # 根据输入的 shape 进行构建
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "conv1", None) is not None:
            with tf.name_scope(self.conv1.name):
                self.conv1.build([None, None, self.num_mel_bins])
        if getattr(self, "conv2", None) is not None:
            with tf.name_scope(self.conv2.name):
                self.conv2.build([None, None, self.embed_dim])
        if getattr(self, "embed_positions", None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.d_model])
        if getattr(self, "encoder_layers", None) is not None:
            for layer in self.encoder_layers:
                with tf.name_scope(layer.name):
                    layer.build(None)
@keras_serializable
class TFWhisperDecoder(tf.keras.layers.Layer):
    config_class = WhisperConfig
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`TFWhisperDecoderLayer`]

    Args:
        config: WhisperConfig
    """

    def __init__(self, config: WhisperConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.dropout = tf.keras.layers.Dropout(config.dropout)  # 初始化一个丢弃层，用于在训练时进行随机丢弃
        self.layerdrop = config.decoder_layerdrop  # 记录层级丢弃率
        self.padding_idx = config.pad_token_id  # 记录填充 token 的索引
        self.max_target_positions = config.max_target_positions  # 记录目标序列的最大长度
        self.max_source_positions = config.max_source_positions  # 记录源序列的最大长度
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0  # 计算嵌入尺度，用于缩放嵌入向量

        self.embed_tokens = tf.keras.layers.Embedding(
            input_dim=config.vocab_size,
            output_dim=config.d_model,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=self.config.init_std),  # 使用截断正态分布初始化嵌入层的参数
            name="embed_tokens",
        )
        self.embed_positions = TFWhisperPositionalEmbedding(
            self.max_target_positions, config.d_model, name="embed_positions"  # 初始化位置编码层

        self.decoder_layers = [TFWhisperDecoderLayer(config, name=f"layers.{i}") for i in range(config.decoder_layers)]  # 初始化一系列 Transformer 解码层

        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layer_norm")  # 初始化层归一化层

    def get_input_embeddings(self):
        return self.embed_tokens  # 获取输入嵌入层

    def set_input_embeddings(self, value):
        self.embed_tokens = value  # 设置输入嵌入层

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        batch_size, seq_len = input_shape[0], input_shape[1]  # 获取输入形状

        combined_attention_mask = tf.cond(
            tf.math.greater(seq_len, 1),
            lambda: _make_causal_mask(input_shape, past_key_values_length=past_key_values_length),  # 创建因果掩码
            lambda: _expand_mask(tf.ones((batch_size, seq_len + past_key_values_length)), tgt_len=seq_len),  # 扩展掩码
        )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, tgt_len=input_shape[-1])  # 扩展注意力掩码
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask  # 合并掩码
            )
        return combined_attention_mask  # 返回合并后的注意力掩码

    @unpack_inputs
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        encoder_hidden_states=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
    # 构建自定义层的方法，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 标记已经构建过
        self.built = True
        # 如果存在嵌入标记，则构建嵌入层
        if getattr(self, "embed_tokens", None) is not None:
            with tf.name_scope(self.embed_tokens.name):
                self.embed_tokens.build(None)
        # 如果存在嵌入位置标记，则构建嵌入位置层
        if getattr(self, "embed_positions", None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)
        # 如果存在层标准化标记，则构建层标准化层
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.d_model])
        # 如果存在解码层，则依次构建解码层
        if getattr(self, "decoder_layers", None) is not None:
            for layer in self.decoder_layers:
                with tf.name_scope(layer.name):
                    layer.build(None)
# 为 TFWhisperMainLayer 类添加文档字符串和序列化支持装饰器
@add_start_docstrings(
    "The bare Whisper Model outputting raw hidden-states without any specific head on top.",
    WHISPER_START_DOCSTRING,
)
@keras_serializable
class TFWhisperMainLayer(tf.keras.layers.Layer):
    # 使用 WhisperConfig 作为配置类
    config_class = WhisperConfig

    # 初始化方法，接受 WhisperConfig 类型的配置参数
    def __init__(self, config: WhisperConfig, **kwargs):
        # 调用父类初始化方法
        super().__init__(**kwargs)
        # 将传入的配置参数赋值给实例变量 config
        self.config = config
        # 创建 TFWhisperEncoder 实例并命名为 encoder
        self.encoder = TFWhisperEncoder(config, name="encoder")
        # 创建 TFWhisperDecoder 实例并命名为 decoder
        self.decoder = TFWhisperDecoder(config, name="decoder")

    # 获取输入嵌入层对象的方法
    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    # 设置输入嵌入层对象的方法
    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    # 获取编码器对象的方法
    def get_encoder(self):
        return self.encoder

    # 获取解码器对象的方法
    def get_decoder(self):
        return self.decoder

    # call 方法，用于模型前向传播
    @add_start_docstrings_to_model_forward(WHISPER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @unpack_inputs
    def call(
        self,
        input_features=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_position_ids=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
    ):
        # 方法主体部分略，用于模型前向传播


# 为 TFWhisperModel 类添加文档字符串
@add_start_docstrings(
    "The bare Whisper Model outputting raw hidden-states without any specific head on top.",
    WHISPER_START_DOCSTRING,
)
class TFWhisperModel(TFWhisperPreTrainedModel):
    # 初始化方法，接受 WhisperConfig 类型的配置参数
    def __init__(self, config: WhisperConfig, **kwargs):
        # 调用父类初始化方法
        super().__init__(config, **kwargs)
        # 创建 TFWhisperMainLayer 实例并命名为 model
        self.model = TFWhisperMainLayer(config, name="model")

    # 获取输入嵌入层对象的方法
    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    # 设置输入嵌入层对象的方法
    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    # 获取编码器对象的方法
    def get_encoder(self):
        return self.model.encoder

    # 获取解码器对象的方法
    def get_decoder(self):
        return self.model.decoder

    # 获取解码器对象的方法
    def decoder(self):
        return self.model.decoder

    # 获取编码器对象的方法
    def encoder(self):
        return self.model.encoder

    # call 方法，用于模型前向传播
    @add_start_docstrings_to_model_forward(WHISPER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    @unpack_inputs
    # 定义了一个方法，用于调用模型进行推理或训练
    def call(
        # 输入特征，可以是 TFModelInputType 类型或 None
        self,
        input_features: TFModelInputType | None = None,
        # 解码器输入的 token IDs，可以是 numpy 数组、tf.Tensor 或 None
        decoder_input_ids: np.ndarray | tf.Tensor | None = None,
        # 解码器注意力掩码，可以是 numpy 数组、tf.Tensor 或 None
        decoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        # 解码器位置 IDs，可以是 numpy 数组、tf.Tensor 或 None
        decoder_position_ids: np.ndarray | tf.Tensor | None = None,
        # 头部掩码，可以是 numpy 数组、tf.Tensor 或 None
        head_mask: np.ndarray | tf.Tensor | None = None,
        # 解码器头部掩码，可以是 numpy 数组、tf.Tensor 或 None
        decoder_head_mask: np.ndarray | tf.Tensor | None = None,
        # 跨注意力头部掩码，可以是 numpy 数组、tf.Tensor 或 None
        cross_attn_head_mask: np.ndarray | tf.Tensor | None = None,
        # 编码器输出，可以是包含一组 numpy 数组或 tf.Tensor 的元组，或 None
        encoder_outputs: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        # 过去的键值对，可以是包含一组 numpy 数组或 tf.Tensor 的元组，或 None
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        # 解码器输入的嵌入，可以是包含一组 numpy 数组或 tf.Tensor 的元组，或 None
        decoder_inputs_embeds: Optional[Tuple[Union[np.ndarray, tf.Tensor]]] = None,
        # 是否使用缓存，可以是布尔值或 None
        use_cache: Optional[bool] = None,
        # 是否输出注意力权重，可以是布尔值或 None
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，可以是布尔值或 None
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典形式的输出，可以是布尔值或 None
        return_dict: Optional[bool] = None,
        # 是否处于训练状态，布尔值，默认为 False
        training: bool = False,
    ) -> Union[Tuple[tf.Tensor], TFSeq2SeqModelOutput]:
        r"""
        返回值:

        示例:

         ```python
         >>> import tensorflow as tf
         >>> from transformers import TFWhisperModel, AutoFeatureExtractor
         >>> from datasets import load_dataset

         >>> model = TFWhisperModel.from_pretrained("openai/whisper-base")
         >>> feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
         >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
         >>> inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="tf")
         >>> input_features = inputs.input_features
         >>> decoder_input_ids = tf.convert_to_tensor([[1, 1]]) * model.config.decoder_start_token_id
         >>> last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
         >>> list(last_hidden_state.shape)
         [1, 2, 512]
         ```"""
        # 调用内部模型进行处理，传递所有参数
        outputs = self.model(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 返回模型输出
        return outputs
    ``` 
        # 为服务输出设置函数，接受输出为参数
        def serving_output(self, output):
            # 如果配置使用缓存，则从输出的过去键值对元组中获取第二个元素作为pkv；否则设置为None
            pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
            # 如果配置输出隐藏层状态，则将输出的解码器隐藏状态转换为张量；否则设置为None
            dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
            # 如果配置输出注意力权重，则将输出的解码器注意力权重转换为张量；否则设置为None
            dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
            # 如果配置输出注意力权重，则将输出的交叉注意力权重转换为张量；否则设置为None
            cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
            # 如果配置输出隐藏层状态，则将输出的编码器隐藏状态转换为张量；否则设置为None
            enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
            # 如果配置输出注意力权重，则将输出的编码器注意力权重转换为张量；否则设置为None
            enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None
            # 返回TFSeq2SeqModelOutput对象
            return TFSeq2SeqModelOutput(
                last_hidden_state=output.last_hidden_state,  # 最后隐藏层状态
                past_key_values=pkv,  # 过去键值对
                decoder_hidden_states=dec_hs,  # 解码器隐藏层状态
                decoder_attentions=dec_attns,  # 解码器注意力权重
                cross_attentions=cross_attns,  # 交叉注意力权重
                encoder_last_hidden_state=output.encoder_last_hidden_state,  # 编码器最后隐藏层状态
                encoder_hidden_states=enc_hs,  # 编码器隐藏层状态
                encoder_attentions=enc_attns,  # 编码器注意力权重
            )
    
        # 构建模型
        def build(self, input_shape=None):
            # 如果已经构建过，则直接返回
            if self.built:
                return
            # 设置构建标志为True
            self.built = True
            # 如果已经存在模型对象
            if getattr(self, "model", None) is not None:
                # 在模型的名称作用域下构建模型
                with tf.name_scope(self.model.name):
                    self.model.build(None)
# 使用装饰器添加模型的文档字符串，描述该模型的作用和用途
@add_start_docstrings(
    "The Whisper Model with a language modeling head. Can be used for automatic speech recognition.",
    WHISPER_START_DOCSTRING,
)
# 定义 TFWhisperForConditionalGeneration 类，继承自 TFWhisperPreTrainedModel 和 TFCausalLanguageModelingLoss
class TFWhisperForConditionalGeneration(TFWhisperPreTrainedModel, TFCausalLanguageModelingLoss):
    # 定义模型的前缀
    base_model_prefix = "model"
    # 定义加载时需要忽略的键列表
    _keys_to_ignore_on_load_missing = [
        r"encoder.version",
        r"decoder.version",
        r"proj_out.weight",
    ]
    # 定义保存时需要忽略的键列表
    _keys_to_ignore_on_save = [
        r"proj_out.weight",
    ]

    # 定义初始化方法，接受 WhisperConfig 类型的 config 参数
    def __init__(self, config: WhisperConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, **kwargs)
        # 初始化 model 属性，使用 TFWhisperMainLayer 类来创建对象
        self.model = TFWhisperMainLayer(config, name="model")

    # 定义获取编码器的方法
    def get_encoder(self):
        return self.model.get_encoder()

    # 定义获取解码器的方法
    def get_decoder(self):
        return self.model.get_decoder()

    # 定义获取输出嵌入的方法
    def get_output_embeddings(self):
        return self.get_input_embeddings()

    # 定义设置输出嵌入的方法
    def set_output_embeddings(self, value):
        self.set_input_embeddings(value)

    # 定义调整 token 嵌入大小的方法
    def resize_token_embeddings(self, new_num_tokens: int) -> tf.keras.layers.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        return new_embeddings

    # 定义 call 方法，用于模型的前向传播
    @add_start_docstrings_to_model_forward(WHISPER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @unpack_inputs
    def call(
        self,
        # 输入特征
        input_features: TFModelInputType | None = None,
        # 解码器输入的 token IDs
        decoder_input_ids: np.ndarray | tf.Tensor | None = None,
        # 解码器的注意力遮罩
        decoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        # 解码器的位置 IDs
        decoder_position_ids: np.ndarray | tf.Tensor | None = None,
        # 头部遮罩
        head_mask: np.ndarray | tf.Tensor | None = None,
        # 解码器头部遮罩
        decoder_head_mask: np.ndarray | tf.Tensor | None = None,
        # 跨 attention 头部遮罩
        cross_attn_head_mask: np.ndarray | tf.Tensor | None = None,
        # 编码器输出
        encoder_outputs: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        # 过去的 key-values 对
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        # 解码器输入嵌入
        decoder_inputs_embeds: Optional[Tuple[Union[np.ndarray, tf.Tensor]]] = None,
        # 标签
        labels: np.ndarray | tf.Tensor | None = None,
        # 是否使用缓存
        use_cache: Optional[bool] = None,
        # 是否输出注意力
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典
        return_dict: Optional[bool] = None,
        # 是否训练
        training: bool = False,
    # 定义 generate 方法，用于生成文本
    def generate(
        self,
        inputs: Optional[tf.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[TFLogitsProcessorList] = None,
        seed: Optional[List[int]] = None,
        return_timestamps: Optional[bool] = None,
        task: Optional[str] = None,
        language: Optional[str] = None,
        is_multilingual: Optional[bool] = None,
        prompt_ids: Optional[tf.Tensor] = None,
        return_token_timestamps=None,
        **kwargs,
    # 定义一个方法，用于处理模型的输出
    def serving_output(self, output):
        # 如果使用缓存，则获取过去的关键数值
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        # 如果配置要求输出隐藏状态，则将输出的解码器隐藏状态转换为张量
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置要求输出注意力权重，则将输出的解码器注意力权重转换为张量
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        # 如果配置要求输出交叉注意力权重，则将输出的交叉注意力权重转换为张量
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        # 如果配置要求输出隐藏状态，则将输出的编码器隐藏状态转换为张量
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置要求输出注意力权重，则将输出的编码器注意力权重转换为张量
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None

        # 返回 Seq2Seq 模型的输出
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

    # 准备生成的输入
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        use_cache=None,
        encoder_outputs=None,
        attention_mask=None,
        decoder_attention_mask=None,
        **kwargs,
    ):
        # 如果使用过去的关键数值，则截取解码器输入的标识符
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # 如果存在解码器的注意力遮罩，则计算解码器位置标识符
        if decoder_attention_mask is not None:  # xla
            decoder_position_ids = tf.math.cumsum(decoder_attention_mask, axis=-1, exclusive=True)[:, -1:]
        # 如果没有使用 xla 并且使用过去的关键数值，则获取过去关键数值的长度作为解码器位置标识符
        elif past_key_values is not None:  # no xla + past
            decoder_position_ids = past_key_values[0][0].shape[2]
        # 如果没有使用 xla 且没有使用过去的关键数值，则使用解码器输入的长度作为解码器位置标识符
        else:  # no xla + no past
            decoder_position_ids = tf.range(decoder_input_ids.shape[1])
        # 将解码器位置标识符广播到解码器输入的形状
        decoder_position_ids = tf.broadcast_to(decoder_position_ids, decoder_input_ids.shape)

        # 返回包含模型输入信息的字典
        return {
            "input_features": None,  # 为了让 Keras.layer.__call__ 正常运行，需要传入该参数
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "use_cache": use_cache,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_position_ids": decoder_position_ids,
        }

    # 构建模型
    def build(self, input_shape=None):
        # 如果已构建，则直接返回
        if self.built:
            return
        # 设置为已构建
        self.built = True
        # 如果模型存在，则为模型建立名称
        if getattr(self, "model", None) is not None:
            with tf.name_scope(self.model.name):
                self.model.build(None)
```