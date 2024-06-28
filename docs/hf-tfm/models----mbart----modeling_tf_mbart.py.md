# `.\models\mbart\modeling_tf_mbart.py`

```
# 设置文件编码为UTF-8

# 版权声明，声明此代码的版权归The Fairseq Authors和The HuggingFace Inc.团队所有
# 根据Apache许可证2.0版授权，除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，本软件是基于"按现状"提供的，不提供任何明示或暗示的保证或条件
# 有关特定语言的权限，请参阅许可证

""" TF 2.0 MBart model."""

from __future__ import annotations  # 使用未来的注释类型

import random  # 导入随机模块
from typing import Optional, Tuple, Union  # 导入类型提示相关模块

import tensorflow as tf  # 导入TensorFlow模块

from ...activations_tf import get_tf_activation  # 导入激活函数获取函数
from ...modeling_tf_outputs import (  # 导入TensorFlow模型输出相关模块
    TFBaseModelOutput,
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFSeq2SeqLMOutput,
    TFSeq2SeqModelOutput,
)

# 公共API
from ...modeling_tf_utils import (  # 导入TensorFlow模型实用工具函数
    TFCausalLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax  # 导入TensorFlow实用工具函数
from ...utils import (  # 导入通用实用函数
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_mbart import MBartConfig  # 导入MBart配置文件

logger = logging.get_logger(__name__)  # 获取模块专用的日志记录器

_CHECKPOINT_FOR_DOC = "facebook/mbart-large-cc25"  # 用于文档的预训练模型检查点
_CONFIG_FOR_DOC = "MBartConfig"  # 用于文档的MBart配置信息

LARGE_NEGATIVE = -1e8  # 设定一个大负数常量，值为-1e8

def shift_tokens_right(input_ids: tf.Tensor, pad_token_id: int):
    """
    将输入的token向右移动一个位置，并用最后一个非pad token（即<LID> token）进行包装。需要注意的是，与其他类似Bart的模型不同，MBart没有单一的`decoder_start_token_id`。
    """
    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")  # 如果pad_token_id为None，则抛出数值错误异常
    # 将标签中可能的-100值替换为`pad_token_id`
    input_ids = tf.where(
        input_ids == -100, tf.fill(shape_list(input_ids), tf.cast(pad_token_id, input_ids.dtype)), input_ids
    )
    language_id_index = (
        tf.reduce_sum(tf.cast(tf.math.not_equal(input_ids, pad_token_id), dtype=input_ids.dtype), axis=-1) - 1
    )
    language_id_index = tf.stack(
        [tf.range(shape_list(input_ids)[0], dtype=input_ids.dtype), language_id_index], axis=-1
    )
    languages_ids = tf.gather_nd(input_ids, language_id_index)

    shifted_input_ids = tf.concat([tf.expand_dims(languages_ids, axis=-1), input_ids[:, :-1]], axis=-1)

    return shifted_input_ids

# 从transformers.models.bart.modeling_tf_bart._make_causal_mask复制过来的函数
def _make_causal_mask(input_ids_shape: tf.TensorShape, past_key_values_length: int = 0):
    """
    创建用于双向自注意力的因果掩码。
    """
    # 获取输入张量的第一维大小，通常表示批量大小
    bsz = input_ids_shape[0]
    
    # 获取输入张量的第二维大小，通常表示序列长度
    tgt_len = input_ids_shape[1]
    
    # 创建一个形状为 (tgt_len, tgt_len) 的张量，并用大负数填充
    mask = tf.ones((tgt_len, tgt_len)) * LARGE_NEGATIVE
    
    # 创建一个形状为 (tgt_len,) 的张量，包含从 0 到 tgt_len-1 的整数
    mask_cond = tf.range(shape_list(mask)[-1])
    
    # 根据条件重新设定 mask 张量的部分值为 0
    mask = tf.where(mask_cond < tf.reshape(mask_cond + 1, (shape_list(mask)[-1], 1)), 0.0, mask)
    
    # 如果过去的键值对长度大于 0，则在 mask 的左侧连接一个形状为 (tgt_len, past_key_values_length) 的全零张量
    if past_key_values_length > 0:
        mask = tf.concat([tf.zeros((tgt_len, past_key_values_length)), mask], axis=-1)
    
    # 将 mask 张量在批量维度和其他维度上进行复制，以匹配输出形状
    return tf.tile(mask[None, None, :, :], (bsz, 1, 1, 1))
# Copied from transformers.models.bart.modeling_tf_bart._expand_mask
def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    # 获取输入 mask 的序列长度
    src_len = shape_list(mask)[1]
    # 如果未指定 tgt_len，则使用 src_len
    tgt_len = tgt_len if tgt_len is not None else src_len
    # 创建一个常数张量，数值为 1.0
    one_cst = tf.constant(1.0)
    # 将 mask 转换为与 one_cst 相同的数据类型
    mask = tf.cast(mask, dtype=one_cst.dtype)
    # 在第二维和第三维上复制 mask，形成新的扩展 mask 张量
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))

    # 返回经过扩展的 mask 与一个大负数相乘的结果
    return (one_cst - expanded_mask) * LARGE_NEGATIVE


# Copied from transformers.models.bart.modeling_tf_bart.TFBartLearnedPositionalEmbedding with Bart->MBart
class TFMBartLearnedPositionalEmbedding(keras.layers.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs):
        # MBart 设定如果指定了 padding_idx，则通过偏移 2 调整 embedding ids，并相应调整 num_embeddings
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim, **kwargs)

    def call(
        self,
        input_shape: Optional[tf.TensorShape] = None,
        past_key_values_length: int = 0,
        position_ids: tf.Tensor | None = None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        # 如果 position_ids 未指定，则根据 input_shape 创建默认位置 ids
        if position_ids is None:
            seq_len = input_shape[1]
            position_ids = tf.range(seq_len, delta=1, name="range")
            position_ids += past_key_values_length

        # 根据 position_ids 的类型设置偏移的数据类型
        offset_dtype = position_ids.dtype if isinstance(position_ids, tf.Tensor) else tf.int32
        # 调用父类 Embedding 的 call 方法，加上偏移值 self.offset
        return super().call(position_ids + tf.constant(self.offset, dtype=offset_dtype))


# Copied from transformers.models.bart.modeling_tf_bart.TFBartAttention with Bart->MBart
class TFMBartAttention(keras.layers.Layer):
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
        # 初始化多头注意力层的参数
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.is_decoder = is_decoder
        self.bias = bias
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = keras.layers.Dropout(dropout)
        self.head_dim = embed_dim // num_heads
        # 检查 embed_dim 是否能被 num_heads 整除，否则抛出异常
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        # 计算缩放因子，用于缩放注意力分数
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        # 创建用于投影的 Dense 层，每个都带有偏置
        self.k_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="k_proj")
        self.q_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="q_proj")
        self.v_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="v_proj")
        self.out_proj = keras.layers.Dense(embed_dim, use_bias=bias, name="out_proj")

    # 重塑张量的形状，用于多头注意力的计算
    def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
        return tf.transpose(tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))

    # 定义模型的调用方法，实现注意力机制的计算
    def call(
        self,
        hidden_states: tf.Tensor,
        key_value_states: tf.Tensor | None = None,
        past_key_value: Tuple[Tuple[tf.Tensor]] | None = None,
        attention_mask: tf.Tensor | None = None,
        layer_head_mask: tf.Tensor | None = None,
        training: Optional[bool] = False,
    ):
        # 这里会实现具体的注意力计算逻辑，但在当前代码段中并未展示完整的实现细节
        pass

    # 构建模型的方法，用于构建每个投影层的 Dense 层
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 检查并构建 k_proj 投影层
        if getattr(self, "k_proj", None) is not None:
            with tf.name_scope(self.k_proj.name):
                self.k_proj.build([None, None, self.embed_dim])
        # 检查并构建 q_proj 投影层
        if getattr(self, "q_proj", None) is not None:
            with tf.name_scope(self.q_proj.name):
                self.q_proj.build([None, None, self.embed_dim])
        # 检查并构建 v_proj 投影层
        if getattr(self, "v_proj", None) is not None:
            with tf.name_scope(self.v_proj.name):
                self.v_proj.build([None, None, self.embed_dim])
        # 检查并构建 out_proj 投影层
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.embed_dim])
class TFMBartEncoderLayer(keras.layers.Layer):
    def __init__(self, config: MBartConfig, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = config.d_model
        # 初始化自注意力层，使用配置中的参数
        self.self_attn = TFMBartAttention(
            self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout, name="self_attn"
        )
        # 自注意力层的 LayerNormalization
        self.self_attn_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        self.dropout = keras.layers.Dropout(config.dropout)
        # 激活函数使用配置中的激活函数类型
        self.activation_fn = get_tf_activation(config.activation_function)
        self.activation_dropout = keras.layers.Dropout(config.activation_dropout)
        # 第一个全连接层
        self.fc1 = keras.layers.Dense(config.encoder_ffn_dim, name="fc1")
        # 第二个全连接层
        self.fc2 = keras.layers.Dense(self.embed_dim, name="fc2")
        # 最终的 LayerNormalization
        self.final_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        self.config = config

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        layer_head_mask: tf.Tensor,
        training: Optional[bool] = False,
    ):
        """
        Args:
            hidden_states (`tf.Tensor`): 输入到层的张量，形状为 *(batch, seq_len, embed_dim)*
            attention_mask (`tf.Tensor`): 注意力掩码张量，形状为 *(batch, 1, tgt_len, src_len)*，
                其中填充元素由非常大的负值表示。
            layer_head_mask (`tf.Tensor`): 给定层中注意力头的掩码张量，形状为 *(encoder_attention_heads,)*
        """
        # 保留残差连接
        residual = hidden_states
        # 使用 LayerNormalization 对输入进行归一化
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 调用自注意力层进行计算
        hidden_states, self_attn_weights, _ = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, layer_head_mask=layer_head_mask
        )

        # 断言保证自注意力操作没有改变张量的形状
        tf.debugging.assert_equal(
            shape_list(hidden_states),
            shape_list(residual),
            message=f"Self attn modified the shape of query {shape_list(residual)} to {shape_list(hidden_states)}",
        )

        # 应用 dropout
        hidden_states = self.dropout(hidden_states, training=training)
        # 残差连接
        hidden_states = residual + hidden_states

        # 保留残差连接
        residual = hidden_states
        # 最终的 LayerNormalization
        hidden_states = self.final_layer_norm(hidden_states)
        # 使用激活函数和第一个全连接层进行前向传播
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 应用激活函数的 dropout
        hidden_states = self.activation_dropout(hidden_states, training=training)
        # 第二个全连接层
        hidden_states = self.fc2(hidden_states)
        # 应用 dropout
        hidden_states = self.dropout(hidden_states, training=training)
        # 残差连接
        hidden_states = residual + hidden_states

        return hidden_states, self_attn_weights
    # 构建模型结构，如果已经构建过，则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        
        # 将标志置为已构建
        self.built = True
        
        # 如果存在 self_attn 属性，则构建 self attention 层
        if getattr(self, "self_attn", None) is not None:
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)
        
        # 如果存在 self_attn_layer_norm 属性，则构建 self attention 层的 layer normalization 层
        if getattr(self, "self_attn_layer_norm", None) is not None:
            with tf.name_scope(self.self_attn_layer_norm.name):
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        
        # 如果存在 fc1 属性，则构建第一个全连接层
        if getattr(self, "fc1", None) is not None:
            with tf.name_scope(self.fc1.name):
                self.fc1.build([None, None, self.embed_dim])
        
        # 如果存在 fc2 属性，则构建第二个全连接层
        if getattr(self, "fc2", None) is not None:
            with tf.name_scope(self.fc2.name):
                self.fc2.build([None, None, self.config.encoder_ffn_dim])
        
        # 如果存在 final_layer_norm 属性，则构建最终的 layer normalization 层
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.embed_dim])
class TFMBartDecoderLayer(keras.layers.Layer):
    # TFMBartDecoderLayer 类，继承自 keras.layers.Layer
    def __init__(self, config: MBartConfig, **kwargs):
        # 初始化方法
        super().__init__(**kwargs)
        # 设置嵌入维度为 config.d_model
        self.embed_dim = config.d_model
        # 初始化 self_attn 层，使用 TFMBartAttention
        self.self_attn = TFMBartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="self_attn",
            is_decoder=True,
        )
        # 设置 dropout 层
        self.dropout = keras.layers.Dropout(config.dropout)
        # 获取激活函数并设置 activation_fn
        self.activation_fn = get_tf_activation(config.activation_function)
        # 设置激活函数的 dropout 层
        self.activation_dropout = keras.layers.Dropout(config.activation_dropout)

        # 初始化 self_attn_layer_norm，LayerNormalization 层
        self.self_attn_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        # 初始化 encoder_attn 层，使用 TFMBartAttention
        self.encoder_attn = TFMBartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="encoder_attn",
            is_decoder=True,
        )
        # 初始化 encoder_attn_layer_norm，LayerNormalization 层
        self.encoder_attn_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="encoder_attn_layer_norm")
        # 初始化全连接层 fc1
        self.fc1 = keras.layers.Dense(config.decoder_ffn_dim, name="fc1")
        # 初始化全连接层 fc2
        self.fc2 = keras.layers.Dense(self.embed_dim, name="fc2")
        # 初始化 final_layer_norm，LayerNormalization 层
        self.final_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        # 设置配置信息
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
        # call 方法，定义了层的正向传播逻辑和参数
    # 构建模型的方法，用于在第一次调用时构建模型结构
    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回，避免重复构建
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        
        # 如果存在 self_attn 属性，则构建 self attention 层
        if getattr(self, "self_attn", None) is not None:
            # 在命名空间下构建 self attention 层
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)
        
        # 如果存在 self_attn_layer_norm 属性，则构建 self attention 层的 layer normalization
        if getattr(self, "self_attn_layer_norm", None) is not None:
            # 在命名空间下构建 self attention 层的 layer normalization
            with tf.name_scope(self.self_attn_layer_norm.name):
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        
        # 如果存在 encoder_attn 属性，则构建 encoder-decoder attention 层
        if getattr(self, "encoder_attn", None) is not None:
            # 在命名空间下构建 encoder-decoder attention 层
            with tf.name_scope(self.encoder_attn.name):
                self.encoder_attn.build(None)
        
        # 如果存在 encoder_attn_layer_norm 属性，则构建 encoder-decoder attention 层的 layer normalization
        if getattr(self, "encoder_attn_layer_norm", None) is not None:
            # 在命名空间下构建 encoder-decoder attention 层的 layer normalization
            with tf.name_scope(self.encoder_attn_layer_norm.name):
                self.encoder_attn_layer_norm.build([None, None, self.embed_dim])
        
        # 如果存在 fc1 属性，则构建第一个全连接层
        if getattr(self, "fc1", None) is not None:
            # 在命名空间下构建第一个全连接层
            with tf.name_scope(self.fc1.name):
                self.fc1.build([None, None, self.embed_dim])
        
        # 如果存在 fc2 属性，则构建第二个全连接层
        if getattr(self, "fc2", None) is not None:
            # 在命名空间下构建第二个全连接层
            with tf.name_scope(self.fc2.name):
                self.fc2.build([None, None, self.config.decoder_ffn_dim])
        
        # 如果存在 final_layer_norm 属性，则构建最终的 layer normalization 层
        if getattr(self, "final_layer_norm", None) is not None:
            # 在命名空间下构建最终的 layer normalization 层
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.embed_dim])
class TFMBartPreTrainedModel(TFPreTrainedModel):
    # 指定该模型所使用的配置类
    config_class = MBartConfig
    # 模型的基础名称前缀
    base_model_prefix = "model"



MBART_START_DOCSTRING = r"""
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
        config ([`MBartConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""



MBART_INPUTS_DOCSTRING = r"""
"""



MBART_GENERATION_EXAMPLE = r"""
    Translation example:

    ```python
    >>> from transformers import AutoTokenizer, TFMBartForConditionalGeneration

    >>> model = TFMBartForConditionalGeneration.from_pretrained("facebook/mbart-large-en-ro")
    >>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-en-ro")

    >>> example_english_phrase = "42 is the answer"
    ```
    >>> inputs = tokenizer(example_english_phrase, return_tensors="tf")
    # 使用预训练的tokenizer将输入的英文短语编码成模型可以处理的张量形式

    >>> # Translate
    >>> generated_ids = model.generate(**inputs, num_beams=4, max_length=5)
    # 使用预训练的翻译模型生成翻译结果，采用4束搜索，最大长度限制为5个token

    >>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    # 解码生成的token IDs，跳过特殊token并保留tokenization空格，然后返回第一个翻译结果的文本形式
    '42 este răspuns'
    ```

    Mask filling example:

    ```python
    >>> from transformers import AutoTokenizer, TFMBartForConditionalGeneration
    >>> import tensorflow as tf

    >>> model = TFMBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")
    # 从预训练的MBart模型加载条件生成器模型

    >>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")
    # 从预训练的MBart模型加载tokenizer

    >>> # de_DE is the language symbol id <LID> for German
    >>> TXT = "</s> Meine Freunde sind <mask> nett aber sie essen zu viel Kuchen. </s> de_DE"
    # 定义一个包含掩码的文本字符串，用于在德语中填充掩码位置的词语

    >>> input_ids = tokenizer([TXT], add_special_tokens=False, return_tensors="tf")["input_ids"]
    # 使用tokenizer将文本转换为模型输入的token IDs张量，不添加特殊token

    >>> logits = model(input_ids).logits
    # 通过模型获取预测的logits

    >>> masked_index = tf.where(input_ids[0] == tokenizer.mask_token_id)[0, 0]
    # 找到输入中掩码token的索引位置

    >>> probs = tf.nn.softmax(logits[0, masked_index], axis=0)
    # 对模型预测的掩码位置的logits进行softmax操作，得到概率分布

    >>> values, predictions = tf.math.top_k(probs, 5)
    # 获取最高的五个概率值及其对应的索引作为预测结果

    >>> tokenizer.decode(predictions).split()
    # 解码预测的token IDs，并以列表形式返回词语预测结果
    ['nett', 'sehr', 'ganz', 'nicht', 'so']
    ```
"""


@keras_serializable
class TFMBartEncoder(keras.layers.Layer):
    # MBart 配置类，用于配置编码器层
    config_class = MBartConfig
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`TFMBartEncoderLayer`].

    Args:
        config: MBartConfig
    """

    def __init__(self, config: MBartConfig, embed_tokens: Optional[keras.layers.Embedding] = None, **kwargs):
        super().__init__(**kwargs)
        # 从配置中初始化各种参数和层
        self.config = config
        self.dropout = keras.layers.Dropout(config.dropout)
        self.layerdrop = config.encoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = tf.math.sqrt(float(config.d_model)) if config.scale_embedding else 1.0

        self.embed_tokens = embed_tokens
        # 学习得到的位置嵌入
        self.embed_positions = TFMBartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            name="embed_positions",
        )
        # 编码器层的列表
        self.layers = [TFMBartEncoderLayer(config, name=f"layers.{i}") for i in range(config.encoder_layers)]
        # 嵌入层的 LayerNormalization
        self.layernorm_embedding = keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm_embedding")
        # 编码器层的 LayerNormalization
        self.layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="layer_norm")
        # 嵌入维度
        self.embed_dim = config.d_model

    def get_embed_tokens(self):
        return self.embed_tokens

    def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens

    @unpack_inputs
    # 编码器的调用方法，包括输入的解包，注意力掩码等
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        inputs_embeds: tf.Tensor | None = None,
        attention_mask: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ):
        # 省略部分代码，用于处理编码器的输入和返回值

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建位置嵌入层
        if getattr(self, "embed_positions", None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)
        # 构建嵌入层的 LayerNormalization
        if getattr(self, "layernorm_embedding", None) is not None:
            with tf.name_scope(self.layernorm_embedding.name):
                self.layernorm_embedding.build([None, None, self.embed_dim])
        # 构建编码器层的 LayerNormalization
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.d_model])
        # 为每个编码器层构建
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)


@keras_serializable
class TFMBartDecoder(keras.layers.Layer):
    config_class = MBartConfig
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`TFMBartDecoderLayer`]

    Args:
        config: MBartConfig
            MBart模型的配置对象，包含模型的各种设置和超参数
        embed_tokens: output embedding
            可选的嵌入层对象，用于将输入的token转换为向量表示
    """

    def __init__(self, config: MBartConfig, embed_tokens: Optional[keras.layers.Embedding] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.embed_tokens = embed_tokens
        self.layerdrop = config.decoder_layerdrop
        self.embed_positions = TFMBartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            name="embed_positions",
        )
        self.embed_scale = tf.math.sqrt(float(config.d_model)) if config.scale_embedding else 1.0
        self.layers = [TFMBartDecoderLayer(config, name=f"layers.{i}") for i in range(config.decoder_layers)]
        self.layernorm_embedding = keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm_embedding")
        self.layer_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="layer_norm")

        self.dropout = keras.layers.Dropout(config.dropout)

    def get_embed_tokens(self):
        return self.embed_tokens

    def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens

    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType = None,
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
        training: Optional[bool] = False,
    ) -> Union[
        TFBaseModelOutputWithPastAndCrossAttentions, Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]
            # 定义了调用该方法时的输入和输出类型及结构，返回模型输出或特定的元组
    # 定义一个方法用于构建模型结构，支持输入形状参数，默认为 None
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回，避免重复构建
        if self.built:
            return
        # 标记模型已构建
        self.built = True
        
        # 如果存在 embed_positions 属性，则构建 embed_positions
        if getattr(self, "embed_positions", None) is not None:
            # 使用 embed_positions 的命名空间来构建，传入 None 作为输入形状
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)
        
        # 如果存在 layernorm_embedding 属性，则构建 layernorm_embedding
        if getattr(self, "layernorm_embedding", None) is not None:
            # 使用 layernorm_embedding 的命名空间来构建，输入形状为 [None, None, self.config.d_model]
            with tf.name_scope(self.layernorm_embedding.name):
                self.layernorm_embedding.build([None, None, self.config.d_model])
        
        # 如果存在 layer_norm 属性，则构建 layer_norm
        if getattr(self, "layer_norm", None) is not None:
            # 使用 layer_norm 的命名空间来构建，输入形状为 [None, None, self.config.d_model]
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.d_model])
        
        # 如果存在 layers 属性（通常是一个层的列表），逐层构建每一层
        if getattr(self, "layers", None) is not None:
            # 遍历每一层
            for layer in self.layers:
                # 使用每一层的命名空间来构建，传入 None 作为输入形状
                with tf.name_scope(layer.name):
                    layer.build(None)
@keras_serializable
class TFMBartMainLayer(keras.layers.Layer):
    # 指定配置类
    config_class = MBartConfig

    def __init__(self, config: MBartConfig, **kwargs):
        super().__init__(**kwargs)

        # 初始化配置
        self.config = config
        # 创建共享的嵌入层，用于共享模型中的词汇表和嵌入向量
        self.shared = keras.layers.Embedding(
            input_dim=config.vocab_size,
            output_dim=config.d_model,
            embeddings_initializer=keras.initializers.TruncatedNormal(stddev=self.config.init_std),
            name="model.shared",
        )
        # 添加额外的属性以指定层的名称前缀（用于加载/存储权重）
        self.shared.load_weight_prefix = "model.shared"

        # 创建编码器和解码器对象
        self.encoder = TFMBartEncoder(config, self.shared, name="encoder")
        self.decoder = TFMBartDecoder(config, self.shared, name="decoder")

    def get_input_embeddings(self):
        # 返回输入嵌入层对象
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        # 设置新的输入嵌入层
        self.shared = new_embeddings
        # 更新编码器和解码器的嵌入层
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType = None,
        attention_mask: tf.Tensor | None = None,
        decoder_input_ids: tf.Tensor | None = None,
        decoder_attention_mask: tf.Tensor | None = None,
        decoder_position_ids: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        decoder_head_mask: tf.Tensor | None = None,
        cross_attn_head_mask: tf.Tensor | None = None,
        encoder_outputs: Optional[Union[Tuple, TFBaseModelOutput]] = None,
        past_key_values: Tuple[Tuple[tf.Tensor]] | None = None,
        inputs_embeds: tf.Tensor | None = None,
        decoder_inputs_embeds: tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
        **kwargs,
    ):
        # 模型调用方法，解包输入参数并进行处理
        # （具体处理逻辑在未显示的代码中实现）
        ) -> Union[TFSeq2SeqModelOutput, tf.Tensor]:
        # 如果没有提供解码器的输入 ID 和嵌入，禁用缓存
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            use_cache = False

        # 如果未显式提供隐藏状态的输出，则使用模型配置中的默认设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # 如果没有提供解码器输入 ID 但提供了输入 ID，则通过将输入 ID 右移来生成解码器输入 ID
        if decoder_input_ids is None and input_ids is not None:
            decoder_input_ids = shift_tokens_right(input_ids, self.config.pad_token_id)

        # 如果没有提供编码器输出，则调用编码器模型进行前向传播
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
        # 如果用户传递了一个元组作为编码器输出，并且设置了 return_dict=True，则将其包装在 TFBaseModelOutput 中
        elif return_dict and not isinstance(encoder_outputs, TFBaseModelOutput):
            encoder_outputs = TFBaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        # 如果用户传递了 TFBaseModelOutput 作为编码器输出，并且设置了 return_dict=False，则将其包装在元组中
        elif not return_dict and not isinstance(encoder_outputs, tuple):
            encoder_outputs = encoder_outputs.to_tuple()

        # 调用解码器模型进行解码操作，生成解码器的输出
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

        # 如果 return_dict=False，则返回解码器和编码器的输出作为元组
        if not return_dict:
            return decoder_outputs + encoder_outputs

        # 如果 return_dict=True，则返回一个 TFSeq2SeqModelOutput 对象，包含解码器和编码器的相关输出
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
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回，避免重复构建
        if self.built:
            return
        # 设置标志为已构建
        self.built = True
        
        # 共享/共同权重期望位于模型基本命名空间中
        # 在 tf.name_scope 的末尾添加 "/"（但不是开头！）将其放入根命名空间，而不是当前命名空间。
        with tf.name_scope(self.shared.load_weight_prefix + "/" + self.shared.name + "/"):
            # 构建共享模型部分
            self.shared.build(None)
        
        # 如果存在编码器，则在其命名空间下构建
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        
        # 如果存在解码器，则在其命名空间下构建
        if getattr(self, "decoder", None) is not None:
            with tf.name_scope(self.decoder.name):
                self.decoder.build(None)
# 使用装饰器添加文档字符串，描述该类为一个输出原始隐藏状态的MBART模型，没有特定头部的处理方式
@add_start_docstrings(
    "The bare MBART Model outputting raw hidden-states without any specific head on top.",
    MBART_START_DOCSTRING,
)
# 继承TFMBartPreTrainedModel类，初始化时接受一个MBartConfig类型的配置对象和其他可选参数
class TFMBartModel(TFMBartPreTrainedModel):
    def __init__(self, config: MBartConfig, *inputs, **kwargs):
        # 调用父类的初始化方法，传入配置和其他参数
        super().__init__(config, *inputs, **kwargs)

        # 创建TFMBartMainLayer对象，用给定名称初始化为"model"
        self.model = TFMBartMainLayer(config, name="model")

    # 返回self.model的encoder部分
    def get_encoder(self):
        return self.model.encoder

    # 返回self.model的decoder部分
    def get_decoder(self):
        return self.model.decoder

    # 使用装饰器unpack_inputs和add_start_docstrings_to_model_forward，以及add_code_sample_docstrings
    # 定义call方法，接受多种输入参数，并返回Union[TFSeq2SeqModelOutput, Tuple[tf.Tensor]]
    @unpack_inputs
    @add_start_docstrings_to_model_forward(MBART_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSeq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType = None,
        attention_mask: tf.Tensor | None = None,
        decoder_input_ids: tf.Tensor | None = None,
        decoder_attention_mask: tf.Tensor | None = None,
        decoder_position_ids: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        decoder_head_mask: tf.Tensor | None = None,
        cross_attn_head_mask: tf.Tensor | None = None,
        encoder_outputs: Optional[Union[Tuple, TFBaseModelOutput]] = None,
        past_key_values: Tuple[Tuple[tf.Tensor]] | None = None,
        inputs_embeds: tf.Tensor | None = None,
        decoder_inputs_embeds: tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
        **kwargs,
    ) -> Union[TFSeq2SeqModelOutput, Tuple[tf.Tensor]]:
        # 调用self.model的forward方法，传入所有参数，获取输出结果
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

        # 返回self.model的输出结果
        return outputs

    # 从transformers.models.bart.modeling_tf_bart.TFBartModel.serving_output复制而来
    # 定义一个方法用于处理模型的输出
    def serving_output(self, output):
        # 如果配置中启用缓存，则获取过去的关键值，否则为 None
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        # 如果配置中启用输出隐藏状态，则将输出的解码器隐藏状态转换为张量，否则为 None
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置中启用输出注意力权重，则将输出的解码器注意力权重转换为张量，否则为 None
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        # 如果配置中启用输出注意力权重，则将输出的交叉注意力权重转换为张量，否则为 None
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        # 如果配置中启用输出隐藏状态，则将输出的编码器隐藏状态转换为张量，否则为 None
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置中启用输出注意力权重，则将输出的编码器注意力权重转换为张量，否则为 None
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None

        # 返回一个 TFSeq2SeqModelOutput 对象，包含各种处理后的模型输出
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

    # 构建方法
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记该实例已经构建
        self.built = True
        # 如果存在模型对象，则在其名称域中进行构建
        if getattr(self, "model", None) is not None:
            with tf.name_scope(self.model.name):
                # 使用模型对象构建
                self.model.build(None)
# Copied from transformers.models.bart.modeling_tf_bart.BiasLayer
class BiasLayer(keras.layers.Layer):
    """
    Bias as a layer. It is used for serialization purposes: `keras.Model.save_weights` stores on a per-layer basis,
    so all weights have to be registered in a layer.
    """

    def __init__(self, shape, initializer, trainable, name, **kwargs):
        super().__init__(name=name, **kwargs)
        # Note: the name of this variable will NOT be scoped when serialized, i.e. it will not be in the format of
        # "outer_layer/inner_layer/.../name:0". Instead, it will be "name:0". For further details, see:
        # https://github.com/huggingface/transformers/pull/18833#issuecomment-1233090214
        # 添加一个偏置项作为层的权重，用于模型的序列化，保证所有权重在保存时都能被注册在一个层中
        self.bias = self.add_weight(name=name, shape=shape, initializer=initializer, trainable=trainable)

    def call(self, x):
        # 在调用时，返回输入张量 x 加上偏置项 self.bias
        return x + self.bias


@add_start_docstrings(
    "The MBART Model with a language modeling head. Can be used for summarization, after fine-tuning the pretrained models.",
    MBART_START_DOCSTRING,
)
class TFMBartForConditionalGeneration(TFMBartPreTrainedModel, TFCausalLanguageModelingLoss):
    _keys_to_ignore_on_load_unexpected = [
        r"model.encoder.embed_tokens.weight",
        r"model.decoder.embed_tokens.weight",
    ]

    def __init__(self, config, *inputs, **kwargs):
        # 使用给定的配置初始化模型
        super().__init__(config, *inputs, **kwargs)
        # 创建 MBART 主模型层，命名为 "model"
        self.model = TFMBartMainLayer(config, name="model")
        # 是否使用缓存的标志
        self.use_cache = config.use_cache
        # 创建一个偏置层用于最终的 logits，注册为缓冲区，不可训练以保持一致性
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
        # 返回输入嵌入
        return self.get_input_embeddings()

    def set_output_embeddings(self, value):
        # 设置输出嵌入
        self.set_input_embeddings(value)

    def get_bias(self):
        # 返回偏置层的偏置项
        return {"final_logits_bias": self.bias_layer.bias}

    def set_bias(self, value):
        # 替换现有的包含偏置项的层，以保证正确的（反）序列化
        vocab_size = value["final_logits_bias"].shape[-1]
        self.bias_layer = BiasLayer(
            name="final_logits_bias", shape=[1, vocab_size], initializer="zeros", trainable=False
        )
        self.bias_layer.bias.assign(value["final_logits_bias"])

    @unpack_inputs
    @add_start_docstrings_to_model_forward(MBART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(MBART_GENERATION_EXAMPLE)
    # 定义一个方法 `call`，用于调用模型进行推理或训练
    def call(
        self,
        input_ids: TFModelInputType = None,  # 输入模型的输入 ID，类型为 TFModelInputType
        attention_mask: tf.Tensor | None = None,  # 注意力掩码，用于指示模型应关注哪些部分
        decoder_input_ids: tf.Tensor | None = None,  # 解码器的输入 ID
        decoder_attention_mask: tf.Tensor | None = None,  # 解码器的注意力掩码
        decoder_position_ids: tf.Tensor | None = None,  # 解码器的位置 ID
        head_mask: tf.Tensor | None = None,  # 头部掩码，用于指示哪些头部应该被屏蔽
        decoder_head_mask: tf.Tensor | None = None,  # 解码器头部的掩码
        cross_attn_head_mask: tf.Tensor | None = None,  # 交叉注意力头部的掩码
        encoder_outputs: Optional[TFBaseModelOutput] = None,  # 编码器的输出
        past_key_values: Tuple[Tuple[tf.Tensor]] = None,  # 过去的键值对，用于存储解码器的历史状态
        inputs_embeds: tf.Tensor | None = None,  # 输入的嵌入表示
        decoder_inputs_embeds: tf.Tensor | None = None,  # 解码器输入的嵌入表示
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典形式的输出
        labels: tf.Tensor | None = None,  # 标签，用于训练时的监督
        training: Optional[bool] = False,  # 是否处于训练模式，默认为 False
    ) -> Union[TFSeq2SeqLMOutput, Tuple[tf.Tensor]]:
        """
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
            Returns either a TFSeq2SeqLMOutput object or a tuple of tf.Tensor depending on `return_dict`.

        """

        # Adjust labels: replace pad_token_id with -100 and keep others unchanged
        if labels is not None:
            labels = tf.where(
                labels == self.config.pad_token_id,
                tf.cast(tf.fill(shape_list(labels), -100), labels.dtype),
                labels,
            )
            use_cache = False
            
            # If decoder_input_ids and decoder_inputs_embeds are not provided, shift labels to the right
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)

        # Pass inputs to the model and get outputs
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

        # Calculate language modeling logits
        lm_logits = tf.matmul(outputs[0], self.model.shared.weights, transpose_b=True)
        lm_logits = self.bias_layer(lm_logits)
        
        # Calculate masked language modeling loss if labels are provided
        masked_lm_loss = None if labels is None else self.hf_compute_loss(labels, lm_logits)

        # Prepare output based on return_dict flag
        if not return_dict:
            # Return tuple of outputs
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        else:
            # Return TFSeq2SeqLMOutput object with specific attributes assigned
            return TFSeq2SeqLMOutput(
                loss=masked_lm_loss,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,  # index 1 of d outputs
                decoder_hidden_states=outputs.decoder_hidden_states,  # index 2 of d outputs
                decoder_attentions=outputs.decoder_attentions,  # index 3 of d outputs
                cross_attentions=outputs.cross_attentions,  # index 4 of d outputs
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,  # index 0 of encoder outputs
                encoder_hidden_states=outputs.encoder_hidden_states,  # index 1 of encoder outputs
                encoder_attentions=outputs.encoder_attentions,  # index 2 of encoder outputs
            )
    # 定义一个方法用于处理模型输出，生成用于序列到序列任务的输出对象
    def serving_output(self, output):
        # 如果配置要求使用缓存，则获取过去的关键值，否则置为None
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        # 如果配置要求输出隐藏状态，则将输出的解码器隐藏状态转换为张量，否则置为None
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置要求输出注意力分布，则将输出的解码器注意力分布转换为张量，否则置为None
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        # 如果配置要求输出交叉注意力分布，则将输出的交叉注意力分布转换为张量，否则置为None
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        # 如果配置要求输出隐藏状态，则将输出的编码器隐藏状态转换为张量，否则置为None
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置要求输出注意力分布，则将输出的编码器注意力分布转换为张量，否则置为None
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None

        # 返回一个 TFSeq2SeqLMOutput 对象，包含模型输出的日志概率、过去关键值、各种隐藏状态和注意力分布
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

    # 从 transformers 库中 TF 版本的 BART 模型中复制的方法，用于为生成准备输入
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
        # 如果有过去的关键值存在，则截取 decoder_input_ids 的最后一个标记
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # 如果存在 decoder_attention_mask，则计算累积位置 ID
        if decoder_attention_mask is not None:  # xla
            decoder_position_ids = tf.math.cumsum(decoder_attention_mask, axis=-1, exclusive=True)[:, -1:]
        # 如果没有 XLA，但存在过去的关键值，则获取过去关键值的第一个批次的位置数
        elif past_key_values is not None:  # no xla + past_key_values
            decoder_position_ids = past_key_values[0][0].shape[2]
        # 否则，生成标准的位置 ID 序列
        else:  # no xla + no past_key_values
            decoder_position_ids = tf.range(decoder_input_ids.shape[1])

        # 返回一个字典，包含生成所需的输入信息，例如输入标识符、编码器输出、过去关键值等
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

    # 定义一个方法，从标签生成解码器输入标识符，用于模型训练
    def prepare_decoder_input_ids_from_labels(self, labels: tf.Tensor):
        # 将标签向右移动一个位置，返回作为解码器的输入
        return shift_tokens_right(labels, self.config.pad_token_id)
    # 定义模型构建方法，参数为输入形状 input_shape，默认为 None
    def build(self, input_shape=None):
        # 如果模型已经构建过，直接返回，不进行重复构建
        if self.built:
            return
        # 设置标志位表明模型已构建
        self.built = True
        # 如果存在模型属性，进行模型构建
        if getattr(self, "model", None) is not None:
            # 使用模型的名称创建命名空间，并调用模型的构建方法
            with tf.name_scope(self.model.name):
                self.model.build(None)
        # 如果存在偏置层属性，进行偏置层的构建
        if getattr(self, "bias_layer", None) is not None:
            # 使用偏置层的名称创建命名空间，并调用偏置层的构建方法
            with tf.name_scope(self.bias_layer.name):
                self.bias_layer.build(None)
```