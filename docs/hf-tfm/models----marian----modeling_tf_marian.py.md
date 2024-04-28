# `.\transformers\models\marian\modeling_tf_marian.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 版权所有 2021 年 Marian 团队的作者和 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版 ("许可证") 授权；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用于强制性法律或书面同意，软件按原样基础分发，
# 没有任何保证或条件，无论是明示的还是暗示的。
# 请参阅许可证，以了解具体语言之特定权限和限制。
""" TF 2.0 Marian 模型 """


from __future__ import annotations

import random
from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFSeq2SeqLMOutput,
    TFSeq2SeqModelOutput,
)

# 公共 API
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
from .configuration_marian import MarianConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "Helsinki-NLP/opus-mt-en-de"
_CONFIG_FOR_DOC = "MarianConfig"

LARGE_NEGATIVE = -1e8

# 从 transformers.models.bart.modeling_tf_bart.shift_tokens_right 复制而来
def shift_tokens_right(input_ids: tf.Tensor, pad_token_id: int, decoder_start_token_id: int):
    # 将 pad_token_id 和 decoder_start_token_id 转换为与 input_ids 相同的数据类型
    pad_token_id = tf.cast(pad_token_id, input_ids.dtype)
    decoder_start_token_id = tf.cast(decoder_start_token_id, input_ids.dtype)
    # 创建开始 token
    start_tokens = tf.fill(
        (shape_list(input_ids)[0], 1), tf.convert_to_tensor(decoder_start_token_id, input_ids.dtype)
    )
    # 向右位移输入 token
    shifted_input_ids = tf.concat([start_tokens, input_ids[:, :-1]], -1)
    # 将标签中可能存在的 -100 值替换为 pad_token_id
    shifted_input_ids = tf.where(
        shifted_input_ids == -100,
        tf.fill(shape_list(shifted_input_ids), tf.convert_to_tensor(pad_token_id, input_ids.dtype)),
        shifted_input_ids,
    )

    # "验证 `labels` 中只包含正值和 -100"
    assert_gte0 = tf.debugging.assert_greater_equal(shifted_input_ids, tf.constant(0, dtype=input_ids.dtype))

    # 确保通过将结果包装在 identity 函数中调用断言操作
    with tf.control_dependencies([assert_gte0]):
        shifted_input_ids = tf.identity(shifted_input_ids)

    return shifted_input_ids

# 从 transformers.models.bart.modeling_tf_bart._make_causal_mask 复制而来
def _make_causal_mask(input_ids_shape: tf.TensorShape, past_key_values_length: int = 0):
    """
    创建用于双向自注意力的因果掩码。
    """
    # 获取输入张量的批量大小和目标长度
    bsz = input_ids_shape[0]
    tgt_len = input_ids_shape[1]
    # 创建全为大负数的二维张量作为初始掩码
    mask = tf.ones((tgt_len, tgt_len)) * LARGE_NEGATIVE
    # 创建一个张量，表示掩码条件
    mask_cond = tf.range(shape_list(mask)[-1])

    # 根据掩码条件，将掩码中小于当前索引的位置置为0
    mask = tf.where(mask_cond < tf.reshape(mask_cond + 1, (shape_list(mask)[-1], 1)), 0.0, mask)

    # 如果过去键值长度大于0，则在掩码的左侧添加全零部分，以对齐过去的键值
    if past_key_values_length > 0:
        mask = tf.concat([tf.zeros((tgt_len, past_key_values_length)), mask], axis=-1)

    # 返回扩展后的掩码，扩展维度为[bsz, 1, tgt_len, tgt_len]
    return tf.tile(mask[None, None, :, :], (bsz, 1, 1, 1))
# 扩展注意力掩码，使其从 [bsz, seq_len] 形状变为 [bsz, 1, tgt_seq_len, src_seq_len] 形状
def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None):
    # 获取输入张量的第二维大小，即 src_seq_len
    src_len = shape_list(mask)[1]
    # 如果提供了 tgt_len，则使用它；否则使用 src_len 作为 tgt_len
    tgt_len = tgt_len if tgt_len is not None else src_len
    # 创建一个全 1 的张量
    one_cst = tf.constant(1.0)
    # 将输入掩码张量转换为 one_cst 的数据类型
    mask = tf.cast(mask, dtype=one_cst.dtype)
    # 将掩码张量沿第二维和第三维复制扩展到 [bsz, 1, tgt_len, src_len] 的形状
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))
    # 返回由 1 减去扩展掩码乘以一个很大的负数得到的张量
    return (one_cst - expanded_mask) * LARGE_NEGATIVE

# 正弦位置编码层
class TFMarianSinusoidalPositionalEmbedding(tf.keras.layers.Layer):
    # 初始化函数，设置位置编码维度和最大位置数量
    def __init__(self, num_positions: int, embedding_dim: int, **kwargs):
        super().__init__(**kwargs)
        # 检查 embedding_dim 是否为偶数
        if embedding_dim % 2 != 0:
            raise NotImplementedError(f"odd embedding_dim {embedding_dim} not supported")
        self.embedding_dim = embedding_dim
        self.num_positions = num_positions

    # 创建位置编码权重
    def build(self, input_shape: tf.TensorShape):
        # 初始化位置编码权重
        weight = self._init_weight(self.num_positions, self.embedding_dim)
        # 添加可训练的位置编码权重
        self.weight = self.add_weight(
            name="embeddings",
            shape=[self.num_positions, self.embedding_dim],
        )
        # 将初始化的权重赋值给可训练的权重
        weight = tf.cast(weight, dtype=self.weight.dtype)
        self.weight.assign(weight)
        super().build(input_shape)

    # 初始化位置编码权重
    @staticmethod
    def _init_weight(n_pos: int, dim: int):
        # 计算位置编码矩阵
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        # 创建初始化的权重矩阵
        table = np.zeros_like(position_enc)
        # 前半部分为 sin 编码
        table[:, 0 : dim // 2] = np.sin(position_enc[:, 0::2])
        # 后半部分为 cos 编码
        table[:, dim // 2 :] = np.cos(position_enc[:, 1::2])
        # 转换为 tensor 并停止梯度
        table = tf.convert_to_tensor(table)
        tf.stop_gradient(table)
        return table

    # 前向传播，根据输入 shape 和 past_key_values_length 生成位置编码
    def call(
        self, input_shape: tf.TensorShape, past_key_values_length: int = 0, position_ids: tf.Tensor | None = None
    ):
        # 如果没有提供 position_ids，根据输入序列长度生成
        if position_ids is None:
            seq_len = input_shape[1]
            position_ids = tf.range(past_key_values_length, seq_len + past_key_values_length, delta=1, name="range")
        # 根据 position_ids 从权重中获取对应的位置编码
        return tf.gather(self.weight, position_ids)

# 注意力层
class TFMarianAttention(tf.keras.layers.Layer):
    # 多头注意力机制来自 "Attention Is All You Need"

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
        # 初始化参数
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.head_dim = embed_dim // num_heads
        # 检查 head_dim 是 embed_dim 的整数倍
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        # 初始化线性变换层
        self.k_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="k_proj")
        self.q_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="q_proj")
        self.v_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="v_proj")
        self.out_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="out_proj")

    # 将张量重新形状成需要的格式
    def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
        return tf.transpose(tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))

    # 实现多头注意力机制的前向传播
    def call(
        self,
        hidden_states: tf.Tensor,
        key_value_states: tf.Tensor | None = None,
        past_key_value: Tuple[Tuple[tf.Tensor]] | None = None,
        attention_mask: tf.Tensor | None = None,
        layer_head_mask: tf.Tensor | None = None,
        training: Optional[bool] = False,
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 构建线性变换层
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
# 从transformers.models.bart.modeling_tf_bart.TFBartEncoderLayer复制代码，并将Bart->Marian
class TFMarianEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: MarianConfig, **kwargs):
        super().__init__(**kwargs)
        # 初始化编码器层的参数
        self.embed_dim = config.d_model
        self.self_attn = TFMarianAttention(
            self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout, name="self_attn"
        )
        self.self_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.activation_fn = get_tf_activation(config.activation_function)
        self.activation_dropout = tf.keras.layers.Dropout(config.activation_dropout)
        self.fc1 = tf.keras.layers.Dense(config.encoder_ffn_dim, name="fc1")
        self.fc2 = tf.keras.layers.Dense(self.embed_dim, name="fc2")
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        self.config = config

    # 定义编码器层的前向传播方法
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: np.ndarray | tf.Tensor | None,
        layer_head_mask: tf.Tensor | None,
        training: Optional[bool] = False,
    ) -> tf.Tensor:
        """
        Args:
            hidden_states (`tf.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`tf.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`tf.Tensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`
        """
        # 保存残差连接
        residual = hidden_states
        # 执行自注意力机制
        hidden_states, self_attn_weights, _ = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, layer_head_mask=layer_head_mask
        )

        # 断言自注意力机制没有改变隐藏状态的形状
        tf.debugging.assert_equal(
            shape_list(hidden_states),
            shape_list(residual),
            message=f"Self attn modified the shape of query {shape_list(residual)} to {shape_list(hidden_states)}",
        )

        # 应用dropout层
        hidden_states = self.dropout(hidden_states, training=training)
        # 添加残差连接
        hidden_states = residual + hidden_states
        # 执行层归一化
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 保存残差连接
        residual = hidden_states
        # 执行激活函数和全连接层1
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.activation_dropout(hidden_states, training=training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        # 添加残差连接
        hidden_states = residual + hidden_states
        # 执行最终层归一化
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states, self_attn_weights
    # 构建模型，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建 self attention 层
        if getattr(self, "self_attn", None) is not None:
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)
        # 构建 self attention 层的 layer normalization
        if getattr(self, "self_attn_layer_norm", None) is not None:
            with tf.name_scope(self.self_attn_layer_norm.name):
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        # 构建第一个全连接层
        if getattr(self, "fc1", None) is not None:
            with tf.name_scope(self.fc1.name):
                self.fc1.build([None, None, self.embed_dim])
        # 构建第二个全连接层
        if getattr(self, "fc2", None) is not None:
            with tf.name_scope(self.fc2.name):
                self.fc2.build([None, None, self.config.encoder_ffn_dim])
        # 构建最终的layer normalization
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.embed_dim])
# 从transformers.models.bart.modeling_tf_bart.TFBartDecoderLayer复制过来，只修改了Bart->Marian
class TFMarianDecoderLayer(tf.keras.layers.Layer):
    # 初始化函数，接受MarianConfig类型的config参数
    def __init__(self, config: MarianConfig, **kwargs):
        # 父类初始化
        super().__init__(**kwargs)
        # 设置embed_dim为config中的d_model值
        self.embed_dim = config.d_model
        # 创建self_attn层，是TFMarianAttention类型的注意力层
        self.self_attn = TFMarianAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="self_attn",
            is_decoder=True,
        )
        # 创建dropout层
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        # 获取激活函数
        self.activation_fn = get_tf_activation(config.activation_function)
        # 创建激活函数的dropout层
        self.activation_dropout = tf.keras.layers.Dropout(config.activation_dropout)

        # 创建LayerNormalization层用于self_attention部分
        self.self_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        # 创建encoder_attn层，是TFMarianAttention类型的注意力层
        self.encoder_attn = TFMarianAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="encoder_attn",
            is_decoder=True,
        )
        # 创建LayerNormalization层用于encoder_attention部分
        self.encoder_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="encoder_attn_layer_norm")
        # 创建全连接层1
        self.fc1 = tf.keras.layers.Dense(config.decoder_ffn_dim, name="fc1")
        # 创建全连接层2
        self.fc2 = tf.keras.layers.Dense(self.embed_dim, name="fc2")
        # 创建最终的LayerNormalization层
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        # 保存config参数
        self.config = config

    # 定义调用函数
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
    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        # 将 built 属性设置为 True，表示模型已经构建完成
        self.built = True
        
        # 如果存在 self_attn 层，则构建它
        if getattr(self, "self_attn", None) is not None:
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)
        
        # 如果存在 self_attn_layer_norm 层，则构建它
        if getattr(self, "self_attn_layer_norm", None) is not None:
            with tf.name_scope(self.self_attn_layer_norm.name):
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        
        # 如果存在 encoder_attn 层，则构建它
        if getattr(self, "encoder_attn", None) is not None:
            with tf.name_scope(self.encoder_attn.name):
                self.encoder_attn.build(None)
        
        # 如果存在 encoder_attn_layer_norm 层，则构建它
        if getattr(self, "encoder_attn_layer_norm", None) is not None:
            with tf.name_scope(self.encoder_attn_layer_norm.name):
                self.encoder_attn_layer_norm.build([None, None, self.embed_dim])
        
        # 如果存在 fc1 层，则构建它
        if getattr(self, "fc1", None) is not None:
            with tf.name_scope(self.fc1.name):
                self.fc1.build([None, None, self.embed_dim])
        
        # 如果存在 fc2 层，则构建它
        if getattr(self, "fc2", None) is not None:
            with tf.name_scope(self.fc2.name):
                self.fc2.build([None, None, self.config.decoder_ffn_dim])
        
        # 如果存在 final_layer_norm 层，则构建它
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.embed_dim])
# 继承自 TFPreTrainedModel 类，这个类提供了一些通用的模型方法
class TFMarianPreTrainedModel(TFPreTrainedModel):
    # 设置配置类为 MarianConfig
    config_class = MarianConfig
    # 设置模型前缀为 "model"
    base_model_prefix = "model"


# 这是 Marian 模型的文档字符串，包含了一些使用提示
MARIAN_START_DOCSTRING = r"""
    # 这个模型继承自 TFPreTrainedModel 类
    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    # 这个模型也是一个 tf.keras.Model 的子类
    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.

    # 关于输入格式的提示
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

    # 模型参数说明
    Args:
        config ([`MarianConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""
MARIAN_GENERATION_EXAMPLE = r"""
        TF version of marian-nmt's transformer.h (c++). Designed for the OPUS-NMT translation checkpoints. Available
        models are listed [here](https://huggingface.co/models?search=Helsinki-NLP).

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, TFMarianMTModel
        >>> from typing import List

        >>> src = "fr"  # source language
        >>> trg = "en"  # target language
        >>> sample_text = "où est l'arrêt de bus ?"
        >>> model_name = f"Helsinki-NLP/opus-mt-{src}-{trg}"

        >>> model = TFMarianMTModel.from_pretrained(model_name)
        >>> tokenizer = AutoTokenizer.from_pretrained(model_name)
        >>> batch = tokenizer([sample_text], return_tensors="tf")
        >>> gen = model.generate(**batch)
        >>> tokenizer.batch_decode(gen, skip_special_tokens=True)
        "Where is the bus stop ?"
        ```
"""

MARIAN_INPUTS_DOCSTRING = r"""
"""


@keras_serializable
class TFMarianEncoder(tf.keras.layers.Layer):
    config_class = MarianConfig
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`TFMarianEncoderLayer`].

    Args:
        config: MarianConfig
    """

    def __init__(self, config: MarianConfig, embed_tokens: Optional[tf.keras.layers.Embedding] = None, **kwargs):
        super().__init__(**kwargs)
        # 初始化层
        self.config = config
        # 丢弃概率
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        # 层丢弃概率
        self.layerdrop = config.encoder_layerdrop
        # 填充索引
        self.padding_idx = config.pad_token_id
        # 最大源序列长度
        self.max_source_positions = config.max_position_embeddings
        # 嵌入缩放
        self.embed_scale = tf.math.sqrt(float(config.d_model)) if config.scale_embedding else 1.0

        # 嵌入标记
        self.embed_tokens = embed_tokens
        # 嵌入位置
        self.embed_positions = TFMarianSinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            name="embed_positions",
        )
        # 编码器层列表
        self.layers = [TFMarianEncoderLayer(config, name=f"layers.{i}") for i in range(config.encoder_layers)]

    def get_embed_tokens(self):
        return self.embed_tokens

    def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens

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
        training: bool = False,
    # 定义一个方法用于构建模型，接受输入形状参数（默认为None）
    def build(self, input_shape=None):
        # 检查模型是否已经构建，若已构建则直接返回
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        # 检查是否存在embed_positions属性，并且不为None
        if getattr(self, "embed_positions", None) is not None:
            # 进入embed_positions的命名空间，构建embed_positions模型
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)
        # 检查是否存在layers属性，并且不为None
        if getattr(self, "layers", None) is not None:
            # 遍历模型中的每一层
            for layer in self.layers:
                # 进入当前层的命名空间，构建当前层模型
                with tf.name_scope(layer.name):
                    layer.build(None)
# 使用 keras_serializable 装饰器将类声明为可序列化的 Keras 层
@keras_serializable
class TFMarianDecoder(tf.keras.layers.Layer):
    # 使用 MarianConfig 类作为类的配置
    config_class = MarianConfig
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`TFMarianDecoderLayer`]

    Args:
        config: MarianConfig
        embed_tokens: output embedding
    """

    def __init__(self, config: MarianConfig, embed_tokens: Optional[tf.keras.layers.Embedding] = None, **kwargs):
        super().__init__(**kwargs)
        # 初始化类的属性
        self.config = config
        self.padding_idx = config.pad_token_id
        self.embed_tokens = embed_tokens
        self.layerdrop = config.decoder_layerdrop
        # 创建位置编码对象
        self.embed_positions = TFMarianSinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            name="embed_positions",
        )
        # 根据配置使用不同的标量乘法，初始化 embed_scale
        self.embed_scale = tf.math.sqrt(float(config.d_model)) if config.scale_embedding else 1.0
        # 初始化一组 Transformer 解码层
        self.layers = [TFMarianDecoderLayer(config, name=f"layers.{i}") for i in range(config.decoder_layers)]

        # 初始化 Dropout 层
        self.dropout = tf.keras.layers.Dropout(config.dropout)

    def get_embed_tokens(self):
        return self.embed_tokens

    def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens

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
        past_key_values: Tuple[Tuple[tf.Tensor]] | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建嵌入位置对象
        if getattr(self, "embed_positions", None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)
        # 构建 Transformer 解码层
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)


@keras_serializable
class TFMarianMainLayer(tf.keras.layers.Layer):
    # 使用 MarianConfig 类作为类的配置
    config_class = MarianConfig
```  
    # 初始化方法，接收一个MarianConfig对象和其他关键字参数
    def __init__(self, config: MarianConfig, **kwargs):
        # 调用父类初始化方法
        super().__init__(**kwargs)

        # 设置self.config为传入的config参数
        self.config = config
        # 创建一个共享的Embedding层，根据config中的参数初始化
        self.shared = tf.keras.layers.Embedding(
            input_dim=config.vocab_size,
            output_dim=config.d_model,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=self.config.init_std),
            name="model.shared",
        )
        # 设置加载权重时的前缀名
        self.shared.load_weight_prefix = "model.shared"

        # 创建编码器对象，传入config和共享的Embedding层
        self.encoder = TFMarianEncoder(config, self.shared, name="encoder")
        # 创建解码器对象，传入config和共享的Embedding层
        self.decoder = TFMarianDecoder(config, self.shared, name="decoder")

    # 获取输入Embedding层的方法
    def get_input_embeddings(self):
        return self.shared

    # 设置新的输入Embedding层的方法
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        # 设置编码器的词嵌入为新的输入Embedding层
        self.encoder.embed_tokens = self.shared
        # 设置解码器的词嵌入为新的输入Embedding层
        self.decoder.embed_tokens = self.shared

    # 定义模型调用的方法，支持解包输入
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
        training: bool = False,
        **kwargs,
    # 这是一个 Transformer 的前向传播函数
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
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
    ):
        # 如果没有传入解码器输入 ID 和解码器输入嵌入,则 use_cache 设为 False
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            use_cache = False
    
        # 如果没有传入 output_hidden_states,则使用配置文件中的值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
    
        # 如果没有传入编码器输出,则调用编码器前向传播函数
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
        # 如果返回字典且编码器输出不是 TFBaseModelOutput 类型,则将其封装为 TFBaseModelOutput
        elif return_dict and not isinstance(encoder_outputs, TFBaseModelOutput):
            encoder_outputs = TFBaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        # 如果不返回字典且编码器输出不是元组类型,则将其转换为元组
        elif not return_dict and not isinstance(encoder_outputs, tuple):
            encoder_outputs = encoder_outputs.to_tuple()
    
        # 调用解码器前向传播函数
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
    
        # 如果不返回字典,则返回解码器输出和编码器输出
        if not return_dict:
            return decoder_outputs + encoder_outputs
    
        # 否则返回 TFSeq2SeqModelOutput 类型的输出
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
    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建过了就直接返回
        if self.built:
            return
        # 设置 built 标志为 True 表示模型已构建完成
        self.built = True
        # 将共享/绑定的权重放到模型基命名空间下
        # 在 tf.name_scope 的末尾加上 "/" 可以将其放到根命名空间而不是当前命名空间
        with tf.name_scope(self.shared.load_weight_prefix + "/" + self.shared.name + "/"):
            # 构建共享的模型部分
            self.shared.build(None)
        # 如果有编码器部分，在编码器的命名空间下构建它
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果有解码器部分，在解码器的命名空间下构建它
        if getattr(self, "decoder", None) is not None:
            with tf.name_scope(self.decoder.name):
                self.decoder.build(None)
# 添加文档字符串，说明这是一个基本的 MARIAN 模型，输出原始的隐藏状态，没有特定的头部
# 继承 TFMarianPreTrainedModel 类
class TFMarianModel(TFMarianPreTrainedModel):
    # 初始化方法，接受一个 MarianConfig 对象和一些可选输入参数
    def __init__(self, config: MarianConfig, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 创建一个 TFMarianMainLayer 对象
        self.model = TFMarianMainLayer(config, name="model")

    # 获取编码器部分
    def get_encoder(self):
        return self.model.encoder

    # 获取解码器部分
    def get_decoder(self):
        return self.model.decoder

    # 使用装饰器处理输入
    # 添加模型正向传播的文档字符串
    # 添加模型正向传播的代码示例文档字符串
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
        encoder_outputs: tf.Tensor | None = None,
        past_key_values: Tuple[Tuple[tf.Tensor]] | None = None,
        inputs_embeds: tf.Tensor | None = None,
        decoder_inputs_embeds: tf.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        training: bool = False,
        **kwargs,
    ) -> Tuple[tf.Tensor] | TFSeq2SeqModelOutput:
        # 调用模型的正向传播功能
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

        # 返回模型正向传播的输出
        return outputs

    # 从 transformers.models.bart.modeling_tf_bart.TFBartModel.serving_output 复制而来
    # 定义一个方法用于处理模型的输出
    def serving_output(self, output):
        # 如果配置了使用缓存，则提取输出中的过去键值，否则为 None
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        # 如果配置了输出隐藏状态，则将输出的解码器隐藏状态转换为张量，否则为 None
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置了输出注意力权重，则将输出的解码器注意力权重转换为张量，否则为 None
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        # 如果配置了输出注意力权重，则将输出的交叉注意力权重转换为张量，否则为 None
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        # 如果配置了输出隐藏状态，则将输出的编码器隐藏状态转换为张量，否则为 None
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置了输出注意力权重，则将输出的编码器注意力权重转换为张量，否则为 None
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None

        # 返回一个 TFSeq2SeqModelOutput 对象，包含了模型输出的相关信息
        return TFSeq2SeqModelOutput(
            last_hidden_state=output.last_hidden_state,  # 最后一个隐藏状态
            past_key_values=pkv,  # 过去键值
            decoder_hidden_states=dec_hs,  # 解码器隐藏状态
            decoder_attentions=dec_attns,  # 解码器注意力权重
            cross_attentions=cross_attns,  # 交叉注意力权重
            encoder_last_hidden_state=output.encoder_last_hidden_state,  # 编码器最后一个隐藏状态
            encoder_hidden_states=enc_hs,  # 编码器隐藏状态
            encoder_attentions=enc_attns,  # 编码器注意力权重
        )

    # 定义一个方法用于构建模型
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果已经存在模型，则在模型的命名空间内构建模型
        if getattr(self, "model", None) is not None:
            with tf.name_scope(self.model.name):
                self.model.build(None)
# 定义 BiasLayer 类，继承自 tf.keras.layers.Layer
class BiasLayer(tf.keras.layers.Layer):
    """
    Bias as a layer. It is used for serialization purposes: `tf.keras.Model.save_weights` stores on a per-layer basis,
    so all weights have to be registered in a layer.
    """

    # 初始化 BiasLayer 类
    def __init__(self, shape, initializer, trainable, name, **kwargs):
        # 调用父类构造函数
        super().__init__(name=name, **kwargs)
        # 添加偏置权重
        self.bias = self.add_weight(name=name, shape=shape, initializer=initializer, trainable=trainable)

    # 对输入进行运算
    def call(self, x):
        return x + self.bias


# 定义 TFMarianMTModel 类，继承自 TFMarianPreTrainedModel 和 TFCausalLanguageModelingLoss
@add_start_docstrings(
    "The MARIAN Model with a language modeling head. Can be used for summarization.",
    MARIAN_START_DOCSTRING,
)
class TFMarianMTModel(TFMarianPreTrainedModel, TFCausalLanguageModelingLoss):
    # 在加载时忽略的键列表
    _keys_to_ignore_on_load_unexpected = [
        r"model.encoder.embed_tokens.weight",
        r"model.decoder.embed_tokens.weight",
    ]

    # 初始化 TFMarianMTModel 类
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类构造函数
        super().__init__(config, *inputs, **kwargs)
        # 创建 TFMarianMainLayer 对象
        self.model = TFMarianMainLayer(config, name="model")
        # 是否使用缓存
        self.use_cache = config.use_cache
        # final_bias_logits 在 pytorch 中作为缓冲区注册，为了一致性不可训练
        self.bias_layer = BiasLayer(
            name="final_logits_bias", shape=[1, config.vocab_size], initializer="zeros", trainable=False
        )

    # 获取解码器
    def get_decoder(self):
        return self.model.decoder

    # 获取编码器
    def get_encoder(self):
        return self.model.encoder

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.get_input_embeddings()

    # 设置输出嵌入
    def set_output_embeddings(self, value):
        self.set_input_embeddings(value)

    # 获取偏置
    def get_bias(self):
        return {"final_logits_bias": self.bias_layer.bias}

    # 设置偏置
    def set_bias(self, value):
        # 替换包含偏置的现有层，以便正确 (de)serialization
        vocab_size = value["final_logits_bias"].shape[-1]
        self.bias_layer = BiasLayer(
            name="final_logits_bias", shape=[1, vocab_size], initializer="zeros", trainable=False
        )
        self.bias_layer.bias.assign(value["final_logits_bias"])

    # 拆包输入
    # 添加相关文档字符串到模型前向传递
    # 替换返回文档字符串
    # 添加 MARIAN 生成示例文档字符串
    @unpack_inputs
    @add_start_docstrings_to_model_forward(MARIAN_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(MARIAN_GENERATION_EXAMPLE)
    # 定义一个方法，用于执行模型的前向传播
    def call(
        # 输入序列的标识符张量，默认为 None
        input_ids: tf.Tensor | None = None,
        # 注意力掩码张量，默认为 None
        attention_mask: tf.Tensor | None = None,
        # 解码器输入序列的标识符张量，默认为 None
        decoder_input_ids: tf.Tensor | None = None,
        # 解码器的注意力掩码张量，默认为 None
        decoder_attention_mask: tf.Tensor | None = None,
        # 解码器位置序列的标识符张量，默认为 None
        decoder_position_ids: tf.Tensor | None = None,
        # 编码器的头掩码张量，默认为 None
        head_mask: tf.Tensor | None = None,
        # 解码器的头掩码张量，默认为 None
        decoder_head_mask: tf.Tensor | None = None,
        # 交叉注意力的头掩码张量，默认为 None
        cross_attn_head_mask: tf.Tensor | None = None,
        # 编码器的输出，默认为 None
        encoder_outputs: TFBaseModelOutput | None = None,
        # 过去键值元组的元组，默认为 None
        past_key_values: Tuple[Tuple[tf.Tensor]] | None = None,
        # 输入嵌入张量，默认为 None
        inputs_embeds: tf.Tensor | None = None,
        # 解码器输入嵌入张量，默认为 None
        decoder_inputs_embeds: tf.Tensor | None = None,
        # 是否使用缓存，默认为 None
        use_cache: bool | None = None,
        # 是否输出注意力，默认为 None
        output_attentions: bool | None = None,
        # 是否输出隐藏状态，默认为 None
        output_hidden_states: bool | None = None,
        # 是否返回字典，默认为 None
        return_dict: bool | None = None,
        # 标签张量，默认为 None
        labels: tf.Tensor | None = None,
        # 是否训练，默认为 False
        training: bool = False,
    ) -> Tuple[tf.Tensor] | TFSeq2SeqLMOutput:
        r"""
        labels (`tf.tensor` of shape `(batch_size, sequence_length)`, *optional`):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        """

        # 如果存在标签数据
        if labels is not None:
            # 将标签中的填充 token 替换为 -100（忽略这些标签），其他的标签保持不变
            labels = tf.where(
                labels == self.config.pad_token_id,
                tf.fill(shape_list(labels), tf.cast(-100, labels.dtype)),
                labels,
            )
            use_cache = False
            # 如果解码器输入 ID 和嵌入都为 None，则根据标签数据生成解码器输入 ID
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        # 使用模型进行前向传播
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
        # 计算语言模型 logits
        lm_logits = tf.matmul(outputs[0], self.model.shared.weights, transpose_b=True)
        lm_logits = self.bias_layer(lm_logits)
        # 计算 masked language modeling loss，若无标签数据则返回 None
        masked_lm_loss = None if labels is None else self.hf_compute_loss(labels, lm_logits)

        # 如果不要返回字典，则组装输出为元组
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        # 返回 TFSeq2SeqLMOutput 类型的输出
        return TFSeq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,  # d outputs 的第一个索引
            decoder_hidden_states=outputs.decoder_hidden_states,  # d outputs 的第二个索引
            decoder_attentions=outputs.decoder_attentions,  # d outputs 的第三个索引
            cross_attentions=outputs.cross_attentions,  # d outputs 的第四个索引
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,  # encoder outputs 的第一个索引
            encoder_hidden_states=outputs.encoder_hidden_states,  # encoder outputs 的第二个索引
            encoder_attentions=outputs.encoder_attentions,  # encoder outputs 的第三个索引
        )

    # 从 transformers.models.bart.modeling_tf_bart.TFBartForConditionalGeneration.serving_output 复制
    # 定义一个方法用于处理输出，在生成阶段被调用
    def serving_output(self, output):
        # 如果使用缓存，则获取过去的关键值，否则设为None
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        # 如果要输出隐藏状态，则将decoder_hidden_states转换为张量
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        # 如果要输出注意力权重，则将decoder_attentions转换为张量
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        # 如果要输出交叉注意力权重，则将cross_attentions转换为张量
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        # 如果要输出编码器隐藏状态，则将encoder_hidden_states转换为张量
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        # 如果要输出编码器注意力权重，则将encoder_attentions转换为张量
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None

        # 返回相关输出作为TFSeq2SeqLMOutput对象
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

    # 从transformers库中拷贝的方法，用于为生成准备输入
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
        # 如果使用过去的关键值，则截取decoder_input_ids的最后一位
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # 如果存在decoder_attention_mask，则计算decoder_position_ids
        if decoder_attention_mask is not None:  
            decoder_position_ids = tf.math.cumsum(decoder_attention_mask, axis=-1, exclusive=True)[:, -1:]
        # 如果没有xla和过去的关键值存在，则直接使用decoder_input_ids的shape
        elif past_key_values is not None:  
            decoder_position_ids = past_key_values[0][0].shape[2]
        # 如果既没有xla也没有过去的关键值，则使用decoder_input_ids的range作为decoder_position_ids
        else:  
            decoder_position_ids = tf.range(decoder_input_ids.shape[1])

        return {
            "input_ids": None,  # encoder_outputs已经定义，不需要input_ids
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_position_ids": decoder_position_ids,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # 改变此项以避免缓存（可能用于调试）
        }

    # 从标签中准备解码器输入
    def prepare_decoder_input_ids_from_labels(self, labels: tf.Tensor):
        # 右移标签并填充解码起始令牌
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
    # 构建模型，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 将构建标记置为 True
        self.built = True
        # 检查模型是否存在，如果存在则在指定名称范围内构建
        if getattr(self, "model", None) is not None:
            with tf.name_scope(self.model.name):
                self.model.build(None)
        # 检查偏置层是否存在，如果存在则在指定名称范围内构建
        if getattr(self, "bias_layer", None) is not None:
            with tf.name_scope(self.bias_layer.name):
                self.bias_layer.build(None)
```