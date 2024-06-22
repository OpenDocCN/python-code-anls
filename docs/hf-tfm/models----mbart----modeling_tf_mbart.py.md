# `.\transformers\models\mbart\modeling_tf_mbart.py`

```py
# 设置文件编码为 utf-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，本文件受版权保护
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的详细信息
""" TF 2.0 MBart model."""
# 导入必要的库
from __future__ import annotations
import random
from typing import Optional, Tuple, Union
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
    TFModelInputType,
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
from .configuration_mbart import MBartConfig
# 获取日志记录器
logger = logging.get_logger(__name__)
# 用于文档的检查点和配置
_CHECKPOINT_FOR_DOC = "facebook/mbart-large-cc25"
_CONFIG_FOR_DOC = "MBartConfig"
# 定义一个大负数
LARGE_NEGATIVE = -1e8

# 将输入的 token 向右移动一个位置，并包装最后一个非填充 token（<LID> token）
def shift_tokens_right(input_ids: tf.Tensor, pad_token_id: int):
    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # 将标签中可能存在的 -100 值替换为 `pad_token_id`
    input_ids = tf.where(
        input_ids == -100, tf.fill(shape_list(input_ids), tf.cast(pad_token_id, input_ids.dtype)), input_ids
    )
    # 找到最后一个非填充 token 的索引
    language_id_index = (
        tf.reduce_sum(tf.cast(tf.math.not_equal(input_ids, pad_token_id), dtype=input_ids.dtype), axis=-1) - 1
    )
    language_id_index = tf.stack(
        [tf.range(shape_list(input_ids)[0], dtype=input_ids.dtype), language_id_index], axis=-1
    )
    languages_ids = tf.gather_nd(input_ids, language_id_index)
    # 将输入的 token 向右移动一个位置
    shifted_input_ids = tf.concat([tf.expand_dims(languages_ids, axis=-1), input_ids[:, :-1]], axis=-1)
    return shifted_input_ids

# 生成用于双向自注意力的因果掩码
def _make_causal_mask(input_ids_shape: tf.TensorShape, past_key_values_length: int = 0):
    bsz = input_ids_shape[0]
    # 获取目标序列的长度
    tgt_len = input_ids_shape[1]
    # 创建一个形状为(tgt_len, tgt_len)的全为LARGE_NEGATIVE的张量作为初始mask
    mask = tf.ones((tgt_len, tgt_len)) * LARGE_NEGATIVE
    # 创建一个与mask长度相同的张量，值为0到mask长度-1
    mask_cond = tf.range(shape_list(mask)[-1])

    # 根据条件将mask中小于当前位置的值设为0
    mask = tf.where(mask_cond < tf.reshape(mask_cond + 1, (shape_list(mask)[-1], 1)), 0.0, mask)

    # 如果过去的键值对长度大于0，则在mask左侧添加一列全为0的张量
    if past_key_values_length > 0:
        mask = tf.concat([tf.zeros((tgt_len, past_key_values_length)), mask], axis=-1)

    # 将mask张量在第0维和第1维上分别复制bsz和1次，扩展为(bsz, 1, tgt_len, tgt_len)的张量
    return tf.tile(mask[None, None, :, :], (bsz, 1, 1, 1))
# 从transformers.models.bart.modeling_tf_bart._expand_mask中复制过来的函数，将注意力掩码从`[bsz, seq_len]`扩展到`[bsz, 1, tgt_seq_len, src_seq_len]`
def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None):
    # 获取掩码的源序列长度
    src_len = shape_list(mask)[1]
    # 如果未指定目标序列长度，则使用源序列长度
    tgt_len = tgt_len if tgt_len is not None else src_len
    # 创建一个常数张量，值为1.0
    one_cst = tf.constant(1.0)
    # 将掩码转换为与one_cst相同数据类型的张量
    mask = tf.cast(mask, dtype=one_cst.dtype)
    # 在第二维和第三维上复制掩码，扩展为`[bsz, 1, tgt_seq_len, src_seq_len]`
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))

    return (one_cst - expanded_mask) * LARGE_NEGATIVE


# 从transformers.models.bart.modeling_tf_bart.TFBartLearnedPositionalEmbedding中复制过来的类，用MBart替换Bart
class TFMBartLearnedPositionalEmbedding(tf.keras.layers.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs):
        # MBart设置了如果指定了padding_idx，则将嵌入id偏移2，并相应调整num_embeddings。其他模型没有这个hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim, **kwargs)

    def call(
        self,
        input_shape: Optional[tf.TensorShape] = None,
        past_key_values_length: int = 0,
        position_ids: tf.Tensor | None = None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        if position_ids is None:
            seq_len = input_shape[1]
            position_ids = tf.range(seq_len, delta=1, name="range")
            position_ids += past_key_values_length

        offset_dtype = position_ids.dtype if isinstance(position_ids, tf.Tensor) else tf.int32
        return super().call(position_ids + tf.constant(self.offset, dtype=offset_dtype))


# 从transformers.models.bart.modeling_tf_bart.TFBartAttention中复制过来的类，用MBart替换Bart
class TFMBartAttention(tf.keras.layers.Layer):
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
class TFMBartEncoderLayer(tf.keras.layers.Layer):
    # 定义 TFMBartEncoderLayer 类，继承自 tf.keras.layers.Layer
    def __init__(self, config: MBartConfig, **kwargs):
        # 初始化函数，接受 MBartConfig 类型的 config 参数和其他关键字参数
        super().__init__(**kwargs)
        # 调用父类的初始化函数
        self.embed_dim = config.d_model
        # 设置 embed_dim 属性为 config 的 d_model 属性
        self.self_attn = TFMBartAttention(
            self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout, name="self_attn"
        )
        # 创建 TFMBartAttention 实例 self_attn
        self.self_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        # 创建 LayerNormalization 实例 self_attn_layer_norm
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        # 创建 Dropout 实例 self.dropout
        self.activation_fn = get_tf_activation(config.activation_function)
        # 获取激活函数
        self.activation_dropout = tf.keras.layers.Dropout(config.activation_dropout)
        # 创建 Dropout 实例 self.activation_dropout
        self.fc1 = tf.keras.layers.Dense(config.encoder_ffn_dim, name="fc1")
        # 创建 Dense 层实例 self.fc1
        self.fc2 = tf.keras.layers.Dense(self.embed_dim, name="fc2")
        # 创建 Dense 层实例 self.fc2
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        # 创建 LayerNormalization 实例 self.final_layer_norm
        self.config = config
        # 设置 config 属性为传入的 config 参数

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        layer_head_mask: tf.Tensor,
        training: Optional[bool] = False,
    ):
        """
        Args:
            hidden_states (`tf.Tensor`): input to the layer of shape *(batch, seq_len, embed_dim)*
            attention_mask (`tf.Tensor`): attention mask of size
                *(batch, 1, tgt_len, src_len)* where padding elements are indicated by very large negative values.
            layer_head_mask (`tf.Tensor`): mask for attention heads in a given layer of size
                *(encoder_attention_heads,)*
        """
        # 定义 call 方法，接受 hidden_states, attention_mask, layer_head_mask 和 training 参数
        residual = hidden_states
        # 将 hidden_states 赋值给 residual
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 对 hidden_states 进行 LayerNormalization
        hidden_states, self_attn_weights, _ = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, layer_head_mask=layer_head_mask
        )
        # 调用 self_attn 进行自注意力计算

        tf.debugging.assert_equal(
            shape_list(hidden_states),
            shape_list(residual),
            message=f"Self attn modified the shape of query {shape_list(residual)} to {shape_list(hidden_states)}",
        )
        # 断言 hidden_states 和 residual 的形状是否相同

        hidden_states = self.dropout(hidden_states, training=training)
        # 对 hidden_states 进行 Dropout
        hidden_states = residual + hidden_states
        # 将 residual 和 hidden_states 相加

        residual = hidden_states
        # 将 hidden_states 赋值给 residual
        hidden_states = self.final_layer_norm(hidden_states)
        # 对 hidden_states 进行 LayerNormalization
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 对 hidden_states 进行激活函数和全连接层操作
        hidden_states = self.activation_dropout(hidden_states, training=training)
        # 对 hidden_states 进行 Dropout
        hidden_states = self.fc2(hidden_states)
        # 对 hidden_states 进行全连接层操作
        hidden_states = self.dropout(hidden_states, training=training)
        # 对 hidden_states 进行 Dropout
        hidden_states = residual + hidden_states
        # 将 residual 和 hidden_states 相加

        return hidden_states, self_attn_weights
        # 返回 hidden_states 和 self_attn_weights
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
class TFMBartDecoderLayer(tf.keras.layers.Layer):
    # 初始化方法，接受 MBartConfig 对象和其他参数
    def __init__(self, config: MBartConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 设置嵌入维度为配置中的模型维度
        self.embed_dim = config.d_model
        # 创建自注意力层对象，传入嵌入维度、注意力头数、dropout等参数
        self.self_attn = TFMBartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="self_attn",
            is_decoder=True,
        )
        # 创建 Dropout 层，传入配置中的 dropout 参数
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        # 获取激活函数对象
        self.activation_fn = get_tf_activation(config.activation_function)
        # 创建激活函数的 Dropout 层，传入配置中的激活函数dropout参数
        self.activation_dropout = tf.keras.layers.Dropout(config.activation_dropout)

        # 创建自注意力层的 LayerNormalization 层
        self.self_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        # 创建编码器注意力层对象，传入嵌入维度、注意力头数、dropout等参数
        self.encoder_attn = TFMBartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="encoder_attn",
            is_decoder=True,
        )
        # 创建编码器注意力层的 LayerNormalization 层
        self.encoder_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="encoder_attn_layer_norm")
        # 创建全连接层1，传入配置中的解码器前馈网络维度
        self.fc1 = tf.keras.layers.Dense(config.decoder_ffn_dim, name="fc1")
        # 创建全连接层2，传入嵌入维度
        self.fc2 = tf.keras.layers.Dense(self.embed_dim, name="fc2")
        # 创建最终的 LayerNormalization 层
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        # 保存配置对象
        self.config = config

    # 调用方法，接受隐藏状态、注意力掩码、编码器隐藏状态等参数
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
class TFMBartPreTrainedModel(TFPreTrainedModel):
    # 设置配置类为 MBartConfig
    config_class = MBartConfig
    # 设置基础模型前缀为 "model"
    base_model_prefix = "model"


MBART_START_DOCSTRING = r"""
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
    >>> inputs = tokenizer(example_english_phrase, return_tensors="tf")
    # 使用tokenizer对英文短语进行编码，返回TensorFlow格式的输入

    >>> # Translate
    >>> generated_ids = model.generate(**inputs, num_beams=4, max_length=5)
    # 使用模型生成翻译结果，设置beam搜索数量为4，最大长度为5
    >>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    # 对生成的结果进行解码，跳过特殊标记并保留分词空格，返回第一个结果

    Mask filling example:

    ```py
    >>> from transformers import AutoTokenizer, TFMBartForConditionalGeneration
    >>> import tensorflow as tf

    >>> model = TFMBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")
    # 从预训练模型中加载TFMBartForConditionalGeneration模型
    >>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")
    # 从预训练模型中加载AutoTokenizer

    >>> # de_DE is the language symbol id <LID> for German
    >>> TXT = "</s> Meine Freunde sind <mask> nett aber sie essen zu viel Kuchen. </s> de_DE"
    # 定义待填充的文本，包含德语标识符

    >>> input_ids = tokenizer([TXT], add_special_tokens=False, return_tensors="tf")["input_ids"]
    # 使用tokenizer对文本进行编码，不添加特殊标记，返回TensorFlow格式的输入
    >>> logits = model(input_ids).logits
    # 使用模型获取logits

    >>> masked_index = tf.where(input_ids[0] == tokenizer.mask_token_id)[0, 0]
    # 找到文本中的掩码位置
    >>> probs = tf.nn.softmax(logits[0, masked_index], axis=0)
    # 对logits进行softmax操作
    >>> values, predictions = tf.math.top_k(probs, 5)
    # 获取概率最高的前5个预测值

    >>> tokenizer.decode(predictions).split()
    # 解码预测结果并分词
    ['nett', 'sehr', 'ganz', 'nicht', 'so']
    # 返回预测结果的分词列表
    ```
# 定义一个继承自 tf.keras.layers.Layer 的 TFMBartEncoder 类，用于实现 MBart 模型的编码器部分
@keras_serializable
class TFMBartEncoder(tf.keras.layers.Layer):
    # 设置配置类为 MBartConfig
    config_class = MBartConfig
    """
    Transformer 编码器，由 config.encoder_layers 个自注意力层组成。每个层都是一个 TFMBartEncoderLayer。

    Args:
        config: MBartConfig
    """

    def __init__(self, config: MBartConfig, embed_tokens: Optional[tf.keras.layers.Embedding] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.layerdrop = config.encoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = tf.math.sqrt(float(config.d_model)) if config.scale_embedding else 1.0

        self.embed_tokens = embed_tokens
        self.embed_positions = TFMBartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            name="embed_positions",
        )
        self.layers = [TFMBartEncoderLayer(config, name=f"layers.{i}") for i in range(config.encoder_layers)]
        self.layernorm_embedding = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm_embedding")
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layer_norm")
        self.embed_dim = config.d_model

    def get_embed_tokens(self):
        return self.embed_tokens

    def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens

    @unpack_inputs
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
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "embed_positions", None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)
        if getattr(self, "layernorm_embedding", None) is not None:
            with tf.name_scope(self.layernorm_embedding.name):
                self.layernorm_embedding.build([None, None, self.embed_dim])
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.d_model])
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)


@keras_serializable
class TFMBartDecoder(tf.keras.layers.Layer):
    config_class = MBartConfig
    """
    # MBart 解码器，由 config.decoder_layers 层组成。每一层都是一个 TFMBartDecoderLayer 对象
    """
    Args:
        config: MBartConfig，MBart 模型的配置
        embed_tokens: 输出的嵌入层
    """
    def __init__(self, config: MBartConfig, embed_tokens: Optional[tf.keras.layers.Embedding] = None, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 保存配置信息
        self.config = config
        # 保存填充标记的索引
        self.padding_idx = config.pad_token_id
        # 保存嵌入层
        self.embed_tokens = embed_tokens
        # 保存层丢弃率
        self.layerdrop = config.decoder_layerdrop
        # 创建位置嵌入层
        self.embed_positions = TFMBartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            name="embed_positions",
        )
        # 设置嵌入层缩放因子
        self.embed_scale = tf.math.sqrt(float(config.d_model)) if config.scale_embedding else 1.0
        # 创建多个解码层
        self.layers = [TFMBartDecoderLayer(config, name=f"layers.{i}") for i in range(config.decoder_layers)]
        # 创建嵌入层的 LayerNormalization 层
        self.layernorm_embedding = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm_embedding")
        # 创建 LayerNormalization 层
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layer_norm")

        # 创建 Dropout 层
        self.dropout = tf.keras.layers.Dropout(config.dropout)

    # 获取嵌入层
    def get_embed_tokens(self):
        return self.embed_tokens

    # 设置嵌入层
    def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens

    # 模型调用方法
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
    # 构建模型，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在嵌入位置信息，则构建嵌入位置信息
        if getattr(self, "embed_positions", None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)
        # 如果存在层归一化嵌入信息，则构建层归一化嵌入信息
        if getattr(self, "layernorm_embedding", None) is not None:
            with tf.name_scope(self.layernorm_embedding.name):
                self.layernorm_embedding.build([None, None, self.config.d_model])
        # 如果存在层归一化信息，则构建层归一化信息
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.d_model])
        # 如果存在多层，则逐层构建
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)
# 使用 keras_serializable 装饰器将类标记为可序列化的
@keras_serializable
class TFMBartMainLayer(tf.keras.layers.Layer):
    # 设置配置类为 MBartConfig
    config_class = MBartConfig

    # 初始化方法，接受 MBartConfig 类型的配置参数
    def __init__(self, config: MBartConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 保存传入的配置参数
        self.config = config
        # 创建共享的嵌入层，用于共享输入和输出的嵌入矩阵
        self.shared = tf.keras.layers.Embedding(
            input_dim=config.vocab_size,
            output_dim=config.d_model,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=self.config.init_std),
            name="model.shared",
        )
        # 设置加载权重时的前缀
        self.shared.load_weight_prefix = "model.shared"

        # 创建编码器和解码器
        self.encoder = TFMBartEncoder(config, self.shared, name="encoder")
        self.decoder = TFMBartDecoder(config, self.shared, name="decoder")

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.shared

    # 设置输入嵌入层
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    # 定义调用方法，接受多个输入参数
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
        # 定义函数的返回类型为 TFSeq2SeqModelOutput 或 tf.Tensor
        ) -> Union[TFSeq2SeqModelOutput, tf.Tensor]:
        # 如果没有提供解码器输入，也没有提供解码器输入的嵌入，则不使用缓存
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            use_cache = False

        # 如果输出隐藏状态不为空，则使用传入的值，否则使用配置中的值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # 如果没有提供解码器输入，并且提供了输入，则将输入向右移动一个位置作为解码器输入
        if decoder_input_ids is None and input_ids is not None:
            decoder_input_ids = shift_tokens_right(input_ids, self.config.pad_token_id)

        # 如果没有提供编码器输出，则使用编码器对输入进行编码
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
        # 如果用户传递了一个元组作为编码器输出，并且 return_dict=True，则将其包装在 TFBaseModelOutput 中
        elif return_dict and not isinstance(encoder_outputs, TFBaseModelOutput):
            encoder_outputs = TFBaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        # 如果用户传递了 TFBaseModelOutput 作为编码器输出，并且 return_dict=False，则将其包装在元组中
        elif not return_dict and not isinstance(encoder_outputs, tuple):
            encoder_outputs = encoder_outputs.to_tuple()

        # 使用解码器对输入进行解码
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

        # 如果 return_dict=False，则返回解码器输出和编码器输出的组合
        if not return_dict:
            return decoder_outputs + encoder_outputs

        # 如果 return_dict=True，则返回 TFSeq2SeqModelOutput 对象
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
    # 构建模型，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 共享/绑定的权重应该在模型基本命名空间中
        # 在 tf.name_scope 后面添加 "/"（不是在开头！）将其放在根命名空间而不是当前命名空间
        with tf.name_scope(self.shared.load_weight_prefix + "/" + self.shared.name + "/"):
            # 构建共享的模型部分
            self.shared.build(None)
        # 如果存在编码器部分
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                # 构建编码器部分
                self.encoder.build(None)
        # 如果存在解码器部分
        if getattr(self, "decoder", None) is not None:
            with tf.name_scope(self.decoder.name):
                # 构建解码器部分
                self.decoder.build(None)
# 添加起始文档字符串，描述该类是一个输出原始隐藏状态而没有特定头部的 MBART 模型
# 继承自 TFMBartPreTrainedModel 类
class TFMBartModel(TFMBartPreTrainedModel):
    # 初始化函数，接受 MBartConfig 类型的配置参数
    def __init__(self, config: MBartConfig, *inputs, **kwargs):
        # 调用父类的初始化函数
        super().__init__(config, *inputs, **kwargs)

        # 创建 TFMBartMainLayer 对象，命名为 "model"
        self.model = TFMBartMainLayer(config, name="model")

    # 获取编码器
    def get_encoder(self):
        return self.model.encoder

    # 获取解码器
    def get_decoder(self):
        return self.model.decoder

    # 调用函数，接受多个输入参数，返回 TFSeq2SeqModelOutput 或 Tuple[tf.Tensor] 类型的输出
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
        # 调用 self.model 的 call 方法，传入各种参数
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
        # 如果配置中输出了隐藏状态，则将decoder_hidden_states转换为张量
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置中输出了注意力权重，则将decoder_attentions转换为张量
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        # 如果配置中输出了注意力权重，则将cross_attentions转换为张量
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        # 如果配置中输出了隐藏状态，则将encoder_hidden_states转换为张量
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置中输出了注意力权重，则将encoder_attentions转换为张量
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None

        # 返回一个TFSeq2SeqModelOutput对象，包含各种输出信息
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
        # 如果已经构建过了，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果模型已经存在
        if getattr(self, "model", None) is not None:
            # 在模型的命名空间下构建模型
            with tf.name_scope(self.model.name):
                self.model.build(None)
# 定义 BiasLayer 类，用于添加偏置作为一个层，用于序列化目的
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


# 定义 TFMBartForConditionalGeneration 类，继承自 TFMBartPreTrainedModel 和 TFCausalLanguageModelingLoss
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
        super().__init__(config, *inputs, **kwargs)
        # 创建 TFMBartMainLayer 模型对象
        self.model = TFMBartMainLayer(config, name="model")
        self.use_cache = config.use_cache
        # 创建 BiasLayer 对象，用于添加最终输出的偏置
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
    @add_start_docstrings_to_model_forward(MBART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(MBART_GENERATION_EXAMPLE)
    # 定义一个方法，用于调用模型
    def call(
        # 输入的 token IDs，默认为 None
        input_ids: TFModelInputType = None,
        # 注意力掩码，默认为 None
        attention_mask: tf.Tensor | None = None,
        # 解码器的输入 token IDs，默认为 None
        decoder_input_ids: tf.Tensor | None = None,
        # 解码器的注意力掩码，默认为 None
        decoder_attention_mask: tf.Tensor | None = None,
        # 解码器的位置 IDs，默认为 None
        decoder_position_ids: tf.Tensor | None = None,
        # 头部掩码，默认为 None
        head_mask: tf.Tensor | None = None,
        # 解码器头部掩码，默认为 None
        decoder_head_mask: tf.Tensor | None = None,
        # 交叉注意力头部掩码，默认为 None
        cross_attn_head_mask: tf.Tensor | None = None,
        # 编码器输出，默认为 None
        encoder_outputs: Optional[TFBaseModelOutput] = None,
        # 过去的键值对，默认为 None
        past_key_values: Tuple[Tuple[tf.Tensor]] = None,
        # 输入的嵌入，默认为 None
        inputs_embeds: tf.Tensor | None = None,
        # 解码器输入的嵌入，默认为 None
        decoder_inputs_embeds: tf.Tensor | None = None,
        # 是否使用缓存，默认为 None
        use_cache: Optional[bool] = None,
        # 输出注意力权重，默认为 None
        output_attentions: Optional[bool] = None,
        # 输出隐藏状态，默认为 None
        output_hidden_states: Optional[bool] = None,
        # 返回字典，默认为 None
        return_dict: Optional[bool] = None,
        # 标签，默认为 None
        labels: tf.Tensor | None = None,
        # 是否训练，默认为 False
        training: Optional[bool] = False,
        ) -> Union[TFSeq2SeqLMOutput, Tuple[tf.Tensor]]:
        """
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        """

        # 如果存在标签，则将标签中的 pad_token_id 替换为 -100，其余保持不变
        if labels is not None:
            labels = tf.where(
                labels == self.config.pad_token_id,
                tf.cast(tf.fill(shape_list(labels), -100), labels.dtype),
                labels,
            )
            use_cache = False
            # 如果 decoder_input_ids 和 decoder_inputs_embeds 都为空，则将标签右移一个位置作为 decoder_input_ids
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)

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
        # 计算 LM logits
        lm_logits = tf.matmul(outputs[0], self.model.shared.weights, transpose_b=True)
        lm_logits = self.bias_layer(lm_logits)
        # 计算 masked_lm_loss
        masked_lm_loss = None if labels is None else self.hf_compute_loss(labels, lm_logits)

        # 如果不返回字典，则返回 lm_logits 和其他输出
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        # 返回 TFSeq2SeqLMOutput 对象
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

    # Copied from transformers.models.bart.modeling_tf_bart.TFBartForConditionalGeneration.serving_output
    # 定义一个方法，用于处理模型输出
    def serving_output(self, output):
        # 如果配置中使用缓存，则获取过去的键值对
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        # 如果配置中输出隐藏状态，则将解码器隐藏状态转换为张量
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置中输出注意力权重，则将解码器注意力权重转换为张量
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        # 如果配置中输出注意力权重，则将交叉注意力权重转换为张量
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        # 如果配置中输出隐藏状态，则将编码器隐藏状态转换为张量
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置中输出注意力权重，则将编码器注意力权重转换为张量
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None

        # 返回处理后的输出对象
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

    # 从transformers库中复制的方法，用于为生成准备输入
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
        # 如果使用了过去的键值对，则截取解码器输入的最后一个标记
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # 如果存在解码器注意力掩码，则计算解码器位置ID
        if decoder_attention_mask is not None:  # xla
            decoder_position_ids = tf.math.cumsum(decoder_attention_mask, axis=-1, exclusive=True)[:, -1:]
        # 如果存在过去的键值对，则获取解码器位置ID
        elif past_key_values is not None:  # no xla + past_key_values
            decoder_position_ids = past_key_values[0][0].shape[2]
        # 如果不存在解码器注意力掩码和过去的键值对，则生成解码器位置ID
        else:  # no xla + no past_key_values
            decoder_position_ids = tf.range(decoder_input_ids.shape[1])

        # 返回生成所需的输入参数
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

    # 从标签中准备解码器输入标记的方法
    def prepare_decoder_input_ids_from_labels(self, labels: tf.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id)
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