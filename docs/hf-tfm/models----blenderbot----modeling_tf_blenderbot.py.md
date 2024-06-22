# `.\transformers\models\blenderbot\modeling_tf_blenderbot.py`

```py
# 设置文件编码为 UTF-8
# 版权声明及许可信息
from __future__ import annotations

# 导入所需模块和库
import os
import random
import warnings
from typing import List, Optional, Tuple, Union

# 导入 TensorFlow 库
import tensorflow as tf

# 导入模型输出相关类
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFSeq2SeqLMOutput,
    TFSeq2SeqModelOutput,
)

# 公共 API
# 导入模型相关的实用函数和类
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFPreTrainedModel,
    keras_serializable,
    unpack_inputs,
)
# 导入常用函数和类
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 导入 Blenderbot 相关配置类
from .configuration_blenderbot import BlenderbotConfig

# 设置日志记录器
logger = logging.get_logger(__name__)

# 用于文档的模型检查点
_CHECKPOINT_FOR_DOC = "facebook/blenderbot-400M-distill"
# 用于文档的模型配置
_CONFIG_FOR_DOC = "BlenderbotConfig"

# 定义一个大的负数常量
LARGE_NEGATIVE = -1e8

# 从 transformers.models.bart.modeling_tf_bart 中复制的函数
# 将输入的 token 向右移动一位
def shift_tokens_right(input_ids: tf.Tensor, pad_token_id: int, decoder_start_token_id: int):
    # 转换 pad_token_id 和 decoder_start_token_id 的数据类型为输入 token 的数据类型
    pad_token_id = tf.cast(pad_token_id, input_ids.dtype)
    decoder_start_token_id = tf.cast(decoder_start_token_id, input_ids.dtype)
    # 创建一个形状为 (batch_size, 1) 的张量，填充值为 decoder_start_token_id
    start_tokens = tf.fill(
        (shape_list(input_ids)[0], 1), tf.convert_to_tensor(decoder_start_token_id, input_ids.dtype)
    )
    # 将 start_tokens 与 input_ids 向右合并
    shifted_input_ids = tf.concat([start_tokens, input_ids[:, :-1]], -1)
    # 将 labels 中可能存在的 -100 值替换为 pad_token_id
    shifted_input_ids = tf.where(
        shifted_input_ids == -100,
        tf.fill(shape_list(shifted_input_ids), tf.convert_to_tensor(pad_token_id, input_ids.dtype)),
        shifted_input_ids,
    )

    # 断言 shifted_input_ids 中的值均大于等于 0
    assert_gte0 = tf.debugging.assert_greater_equal(shifted_input_ids, tf.constant(0, dtype=input_ids.dtype))

    # 确保断言操作被调用，通过在结果外部添加一个身份操作
    with tf.control_dependencies([assert_gte0]):
        shifted_input_ids = tf.identity(shifted_input_ids)

    return shifted_input_ids

# 从 transformers.models.bart.modeling_tf_bart 中复制的函数
# 创建一个因果注意力掩码
# 创建用于双向自注意力的因果掩码
def _make_causal_mask(input_ids_shape: tf.TensorShape, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    创建用于双向自注意力的因果掩码。
    """
    # 获取批量大小和目标序列长度
    bsz = input_ids_shape[0]
    tgt_len = input_ids_shape[1]
    # 创建一个初始值为大负数的掩码
    mask = tf.ones((tgt_len, tgt_len)) * LARGE_NEGATIVE
    # 创建掩码条件
    mask_cond = tf.range(shape_list(mask)[-1])

    # 将对角线以上的元素设置为0，实现因果掩码
    mask = tf.where(mask_cond < tf.reshape(mask_cond + 1, (shape_list(mask)[-1], 1)), 0.0, mask)

    # 如果存在过去的键值对长度，则在掩码前面增加相应的0掩码
    if past_key_values_length > 0:
        mask = tf.concat([tf.zeros((tgt_len, past_key_values_length)), mask], axis=-1)

    return tf.tile(mask[None, None, :, :], (bsz, 1, 1, 1))


# 从transformers.models.bart.modeling_tf_bart._expand_mask复制而来
def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    将注意力掩码从 `[bsz, seq_len]` 扩展到 `[bsz, 1, tgt_seq_len, src_seq_len]`。
    """
    # 获取源序列长度和目标序列长度
    src_len = shape_list(mask)[1]
    tgt_len = tgt_len if tgt_len is not None else src_len
    one_cst = tf.constant(1.0)
    mask = tf.cast(mask, dtype=one_cst.dtype)
    # 在第2和第3个维度上进行广播，扩展掩码
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))

    return (one_cst - expanded_mask) * LARGE_NEGATIVE


class TFBlenderbotLearnedPositionalEmbedding(tf.keras.layers.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    此模块学习位置嵌入，最大尺寸为固定值。
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs):
        super().__init__(num_embeddings, embedding_dim, **kwargs)

    def call(
        self, input_shape: tf.TensorShape, past_key_values_length: int = 0, position_ids: tf.Tensor | None = None
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        # 如果没有提供位置ID，则根据输入形状创建位置ID
        if position_ids is None:
            seq_len = input_shape[1]
            position_ids = tf.range(seq_len, delta=1, name="range")
            position_ids += past_key_values_length

        return super().call(tf.cast(position_ids, dtype=tf.int32))


# 从transformers.models.bart.modeling_tf_bart.TFBartAttention复制而来，将Bart替换为Blenderbot
class TFBlenderbotAttention(tf.keras.layers.Layer):
    """Multi-headed attention from "Attention Is All You Need"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        **kwargs,
    # 初始化 Transformer 层
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True, **kwargs):
        # 调用父类初始化方法
        super().__init__(**kwargs)
        # 设置嵌入维度
        self.embed_dim = embed_dim
        # 设置注意力头的数量
        self.num_heads = num_heads
        # 初始化 Dropout 层
        self.dropout = tf.keras.layers.Dropout(dropout)
        # 计算每个注意力头的维度
        self.head_dim = embed_dim // num_heads
        # 检查是否能够完全分割 embed_dim 到 num_heads 个头
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                # 抛出错误，要求 embed_dim 必须能够被 num_heads 整除
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        # 计算缩放因子
        self.scaling = self.head_dim**-0.5
        # 是否为解码器层
        self.is_decoder = is_decoder

        # 初始化密钥、查询和数值投影层
        self.k_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="k_proj")
        self.q_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="q_proj")
        self.v_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="v_proj")
        # 初始化输出投影层
        self.out_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="out_proj")

    # 将输入张量形状变换为 (batch_size, num_heads, seq_len, head_dim)
    def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
        return tf.transpose(tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))

    # Transformer 层的前向传播
    def call(
        self,
        hidden_states: tf.Tensor,
        key_value_states: tf.Tensor | None = None,
        past_key_value: Tuple[Tuple[tf.Tensor]] | None = None,
        attention_mask: tf.Tensor | None = None,
        layer_head_mask: tf.Tensor | None = None,
        training: Optional[bool] = False,
    ):
        # 该方法将在每次前向传播时被调用
        pass

    # 构建 Transformer 层
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 设置为已构建状态
        self.built = True
        # 构建密钥投影层
        if getattr(self, "k_proj", None) is not None:
            with tf.name_scope(self.k_proj.name):
                self.k_proj.build([None, None, self.embed_dim])
        # 构建查询投影层
        if getattr(self, "q_proj", None) is not None:
            with tf.name_scope(self.q_proj.name):
                self.q_proj.build([None, None, self.embed_dim])
        # 构建数值投影层
        if getattr(self, "v_proj", None) is not None:
            with tf.name_scope(self.v_proj.name):
                self.v_proj.build([None, None, self.embed_dim])
        # 构建输出投影层
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.embed_dim])
# 从transformers.models.mbart.modeling_tf_mbart.TFMBartEncoderLayer复制代码，并将MBart->Blenderbot
class TFBlenderbotEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: BlenderbotConfig, **kwargs):
        super().__init__(**kwargs)
        # 设置嵌入维度为配置文件中的d_model
        self.embed_dim = config.d_model
        # 初始化自注意力层，使用BlenderbotAttention
        self.self_attn = TFBlenderbotAttention(
            self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout, name="self_attn"
        )
        # 初始化自注意力层的LayerNormalization
        self.self_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        # 初始化dropout层
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        # 设置激活函数
        self.activation_fn = get_tf_activation(config.activation_function)
        # 初始化激活函数的dropout层
        self.activation_dropout = tf.keras.layers.Dropout(config.activation_dropout)
        # 初始化全连接层1
        self.fc1 = tf.keras.layers.Dense(config.encoder_ffn_dim, name="fc1")
        # 初始化全连接层2
        self.fc2 = tf.keras.layers.Dense(self.embed_dim, name="fc2")
        # 初始化最终的LayerNormalization
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        # 保存配置
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
            hidden_states (`tf.Tensor`): input to the layer of shape *(batch, seq_len, embed_dim)*
            attention_mask (`tf.Tensor`): attention mask of size
                *(batch, 1, tgt_len, src_len)* where padding elements are indicated by very large negative values.
            layer_head_mask (`tf.Tensor`): mask for attention heads in a given layer of size
                *(encoder_attention_heads,)*
        """
        # 保留输入的残差连接
        residual = hidden_states
        # 对输入进行自注意力层的LayerNormalization
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 经过自注意力层的计算
        hidden_states, self_attn_weights, _ = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, layer_head_mask=layer_head_mask
        )

        # 断言自注意力层的输出和残差连接的形状一致
        tf.debugging.assert_equal(
            shape_list(hidden_states),
            shape_list(residual),
            message=f"Self attn modified the shape of query {shape_list(residual)} to {shape_list(hidden_states)}",
        )

        # 使用dropout层进行正则化
        hidden_states = self.dropout(hidden_states, training=training)
        # 添加残差连接
        hidden_states = residual + hidden_states

        # 保留输入的残差连接
        residual = hidden_states
        # 对输出进行最终的LayerNormalization
        hidden_states = self.final_layer_norm(hidden_states)
        # 使用激活函数和全连接层1进行计算
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 使用激活函数的dropout层进行正则化
        hidden_states = self.activation_dropout(hidden_states, training=training)
        # 使用全连接层2进行计算
        hidden_states = self.fc2(hidden_states)
        # 使用dropout层进行正则化
        hidden_states = self.dropout(hidden_states, training=training)
        # 添加残差连接
        hidden_states = residual + hidden_states

        # 返回结果和自注意力权重
        return hidden_states, self_attn_weights
    # 构建函数，用于构建模型的层
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回，不重复构建
        if self.built:
            return
        # 将标记设置为已构建
        self.built = True
        # 如果存在自注意力机制，则构建自注意力层
        if getattr(self, "self_attn", None) is not None:
            with tf.name_scope(self.self_attn.name):
                # 构建自注意力层
                self.self_attn.build(None)
        # 如果存在自注意力层的层归一化，则构建层归一化层
        if getattr(self, "self_attn_layer_norm", None) is not None:
            with tf.name_scope(self.self_attn_layer_norm.name):
                # 构建层归一化层
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        # 如果存在第一个全连接层，则构建第一个全连接层
        if getattr(self, "fc1", None) is not None:
            with tf.name_scope(self.fc1.name):
                # 构建第一个全连接层
                self.fc1.build([None, None, self.embed_dim])
        # 如果存在第二个全连接层，则构建第二个全连接层
        if getattr(self, "fc2", None) is not None:
            with tf.name_scope(self.fc2.name):
                # 构建第二个全连接层
                self.fc2.build([None, None, self.config.encoder_ffn_dim])
        # 如果存在最终的层归一化层，则构建最终的层归一化层
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                # 构建最终的层归一化层
                self.final_layer_norm.build([None, None, self.embed_dim])
# 从 transformers.models.mbart.modeling_tf_mbart.TFMBartDecoderLayer 复制代码，并将 MBart 替换为 Blenderbot
class TFBlenderbotDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: BlenderbotConfig, **kwargs):
        super().__init__(**kwargs)
        # 设置嵌入维度为配置中的 d_model
        self.embed_dim = config.d_model
        # 创建自注意力机制，用于处理自注意力
        self.self_attn = TFBlenderbotAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="self_attn",
            is_decoder=True,
        )
        # 添加 dropout 层，用于防止过拟合
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        # 获取激活函数，并初始化激活函数的 dropout 层
        self.activation_fn = get_tf_activation(config.activation_function)
        self.activation_dropout = tf.keras.layers.Dropout(config.activation_dropout)

        # 添加自注意力层的 LayerNormalization 层
        self.self_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        # 创建编码器注意力机制，用于处理编码器-解码器之间的注意力
        self.encoder_attn = TFBlenderbotAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="encoder_attn",
            is_decoder=True,
        )
        # 添加编码器注意力层的 LayerNormalization 层
        self.encoder_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="encoder_attn_layer_norm")
        # 添加全连接层 1，用于解码器的前馈网络
        self.fc1 = tf.keras.layers.Dense(config.decoder_ffn_dim, name="fc1")
        # 添加全连接层 2，用于解码器的前馈网络
        self.fc2 = tf.keras.layers.Dense(self.embed_dim, name="fc2")
        # 添加最终的 LayerNormalization 层
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        # 保存配置信息
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
    # 构建模型的方法，用于构建自注意力层、层归一化层和全连接层等组件
    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回，不重复构建
        if self.built:
            return
        # 将标志设置为已构建
        self.built = True
        # 如果存在自注意力层，则构建自注意力层
        if getattr(self, "self_attn", None) is not None:
            # 使用自注意力层的名称作为命名空间
            with tf.name_scope(self.self_attn.name):
                # 构建自注意力层
                self.self_attn.build(None)
        # 如果存在自注意力层归一化层，则构建该层
        if getattr(self, "self_attn_layer_norm", None) is not None:
            # 使用自注意力层归一化层的名称作为命名空间
            with tf.name_scope(self.self_attn_layer_norm.name):
                # 构建自注意力层归一化层
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        # 如果存在编码器注意力层，则构建编码器注意力层
        if getattr(self, "encoder_attn", None) is not None:
            # 使用编码器注意力层的名称作为命名空间
            with tf.name_scope(self.encoder_attn.name):
                # 构建编码器注意力层
                self.encoder_attn.build(None)
        # 如果存在编码器注意力层归一化层，则构建该层
        if getattr(self, "encoder_attn_layer_norm", None) is not None:
            # 使用编码器注意力层归一化层的名称作为命名空间
            with tf.name_scope(self.encoder_attn_layer_norm.name):
                # 构建编码器注意力层归一化层
                self.encoder_attn_layer_norm.build([None, None, self.embed_dim])
        # 如果存在第一个全连接层，则构建该层
        if getattr(self, "fc1", None) is not None:
            # 使用第一个全连接层的名称作为命名空间
            with tf.name_scope(self.fc1.name):
                # 构建第一个全连接层
                self.fc1.build([None, None, self.embed_dim])
        # 如果存在第二个全连接层，则构建该层
        if getattr(self, "fc2", None) is not None:
            # 使用第二个全连接层的名称作为命名空间
            with tf.name_scope(self.fc2.name):
                # 构建第二个全连接层
                self.fc2.build([None, None, self.config.decoder_ffn_dim])
        # 如果存在最终层归一化层，则构建该层
        if getattr(self, "final_layer_norm", None) is not None:
            # 使用最终层归一化层的名称作为命名空间
            with tf.name_scope(self.final_layer_norm.name):
                # 构建最终层归一化层
                self.final_layer_norm.build([None, None, self.embed_dim])
class TFBlenderbotPreTrainedModel(TFPreTrainedModel):
    # 该类继承自 TFPreTrainedModel，用于 Blenderbot 预训练模型
    config_class = BlenderbotConfig
    # 基础模型前缀
    base_model_prefix = "model"


BLENDERBOT_START_DOCSTRING = r"""
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
        config ([`BlenderbotConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

BLENDERBOT_GENERATION_EXAMPLE = r"""
    Conversation example::

    ```py
    >>> from transformers import AutoTokenizer, TFBlenderbotForConditionalGeneration

    >>> mname = "facebook/blenderbot-400M-distill"
    >>> model = TFBlenderbotForConditionalGeneration.from_pretrained(mname)
    >>> tokenizer = AutoTokenizer.from_pretrained(mname)
    # 定义一个人类的发言内容
    UTTERANCE = "My friends are cool but they eat too many carbs."
    # 打印人类的发言内容
    print("Human: ", UTTERANCE)
    
    # 使用分词器将人类的发言内容转换为模型的输入张量
    inputs = tokenizer([UTTERANCE], return_tensors="tf")
    # 使用模型生成回复的标识符
    reply_ids = model.generate(**inputs)
    # 打印生成的机器人回复
    print("Bot: ", tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0])
    
    # 定义另一个人类的发言内容
    REPLY = "I'm not sure"
    # 打印另一个人类的发言内容
    print("Human: ", REPLY)
    
    # 定义下一个人类的发言内容
    NEXT_UTTERANCE = (
        "My friends are cool but they eat too many carbs.</s> <s>That's unfortunate. "
        "Are they trying to lose weight or are they just trying to be healthier?</s> "
        "<s> I'm not sure."
    )
    # 使用分词器将下一个人类的发言内容转换为模型的输入张量
    inputs = tokenizer([NEXT_UTTERANCE], return_tensors="tf")
    # 使用模型生成下一个回复的标识符
    next_reply_ids = model.generate(**inputs)
    # 打印生成的下一个机器人回复
    print("Bot: ", tokenizer.batch_decode(next_reply_ids, skip_special_tokens=True)[0])
"""

BLENDERBOT_INPUTS_DOCSTRING = r"""
"""

# 使用 keras_serializable 装饰器标记这是一个可以序列化的 Keras 层
@keras_serializable
class TFBlenderbotEncoder(tf.keras.layers.Layer):
    # 该层对应的配置类为 BlenderbotConfig
    config_class = BlenderbotConfig
    """
    Transformer 编码器，由 *config.encoder_layers* 个自注意力层组成。每一层都是一个 `TFBlenderbotEncoderLayer`。

    Args:
        config: BlenderbotConfig，模型配置
    """

    def __init__(self, config: BlenderbotConfig, embed_tokens: Optional[tf.keras.layers.Embedding] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.dropout = tf.keras.layers.Dropout(config.dropout)  # Dropout 层，用于在训练过程中随机丢弃部分神经元
        self.layerdrop = config.encoder_layerdrop  # 每一层的丢弃概率
        self.padding_idx = config.pad_token_id  # 填充标记的索引
        self.max_source_positions = config.max_position_embeddings  # 输入序列的最大长度
        self.embed_scale = tf.math.sqrt(float(config.d_model)) if config.scale_embedding else 1.0  # 嵌入缩放因子

        self.embed_tokens = embed_tokens  # 嵌入层的嵌入权重
        self.embed_positions = TFBlenderbotLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            name="embed_positions",
        )  # 学习位置嵌入
        self.layers = [TFBlenderbotEncoderLayer(config, name=f"layers.{i}") for i in range(config.encoder_layers)]  # 编码器层列表
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layer_norm")  # 层归一化

    # 获取嵌入层的嵌入权重
    def get_embed_tokens(self):
        return self.embed_tokens

    # 设置嵌入层的嵌入权重
    def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens

    # 模型调用方法，执行编码器的前向传播
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
        # 省略了具体的前向传播逻辑

    # 构建编码器层，主要用于构建层的参数
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建位置嵌入
        if getattr(self, "embed_positions", None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)
        # 构建层归一化
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.d_model])
        # 构建编码器层
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)


@keras_serializable
class TFBlenderbotDecoder(tf.keras.layers.Layer):
    config_class = BlenderbotConfig
    """
    Transformer 解码器，由 *config.decoder_layers* 个层组成。每一层都是一个 `TFBlenderbotDecoderLayer`

    Args:
        config: BlenderbotConfig，模型配置
        embed_tokens: 输出嵌入
    """
    # 初始化方法，接受 BlenderbotConfig 和可选的嵌入标记作为参数
    def __init__(self, config: BlenderbotConfig, embed_tokens: Optional[tf.keras.layers.Embedding] = None, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 保存配置信息
        self.config = config
        # 保存填充索引
        self.padding_idx = config.pad_token_id
        # 保存嵌入标记
        self.embed_tokens = embed_tokens
        # 保存解码器层丢弃率
        self.layerdrop = config.decoder_layerdrop
        # 创建学习位置嵌入对象
        self.embed_positions = TFBlenderbotLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            name="embed_positions",
        )
        # 设置嵌入缩放因子
        self.embed_scale = tf.math.sqrt(float(config.d_model)) if config.scale_embedding else 1.0
        # 创建解码器层列表
        self.layers = [TFBlenderbotDecoderLayer(config, name=f"layers.{i}") for i in range(config.decoder_layers)]
        # 创建层归一化层
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layer_norm")

        # 创建丢弃层
        self.dropout = tf.keras.layers.Dropout(config.dropout)

    # 获取嵌入标记
    def get_embed_tokens(self):
        return self.embed_tokens

    # 设置嵌入标记
    def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens

    # 调用方法，接受多个参数
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
    
    # 构建方法，接受输入形状作为参数
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 构建嵌入位置
        if getattr(self, "embed_positions", None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)
        # 构建层归一化
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.d_model])
        # 构建每个解码器层
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)
# 使用 keras_serializable 装饰器，将类 TFBlenderbotMainLayer 标记为可序列化的 Keras 层
@keras_serializable
class TFBlenderbotMainLayer(tf.keras.layers.Layer):
    # 指定配置类为 BlenderbotConfig
    config_class = BlenderbotConfig

    # 初始化方法
    def __init__(self, config: BlenderbotConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 保存传入的配置参数
        self.config = config
        # 创建共享的嵌入层，用于共享输入和输出的词嵌入
        self.shared = tf.keras.layers.Embedding(
            input_dim=config.vocab_size,
            output_dim=config.d_model,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=self.config.init_std),
            name="model.shared",
        )
        # 为了加载/存储权重时指定图层的预期名称范围，添加了额外的属性
        self.shared.load_weight_prefix = "model.shared"

        # 创建编码器对象
        self.encoder = TFBlenderbotEncoder(config, self.shared, name="encoder")
        # 创建解码器对象
        self.decoder = TFBlenderbotDecoder(config, self.shared, name="decoder")

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.shared

    # 设置输入嵌入层
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        # 更新编码器和解码器的嵌入层
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    # 调用方法，执行模型的前向传播
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
        # 检查是否需要输出隐藏状态，默认为模型配置中的值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # 如果没有提供编码器输出，则调用编码器进行前向传播
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
        # 如果用户传递了元组作为编码器输出，并且设置了return_dict=True，则将其包装在TFBaseModelOutput中
        elif return_dict and not isinstance(encoder_outputs, TFBaseModelOutput):
            encoder_outputs = TFBaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        # 如果用户传递了TFBaseModelOutput作为编码器输出，并且设置了return_dict=False，则将其包装在元组中
        elif not return_dict and not isinstance(encoder_outputs, tuple):
            encoder_outputs = encoder_outputs.to_tuple()

        # 调用解码器进行前向传播
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

        # 如果不需要返回字典，则将解码器输出和编码器输出合并返回
        if not return_dict:
            return decoder_outputs + encoder_outputs

        # 返回TFSeq2SeqModelOutput对象，包含解码器和编码器的相关输出
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
    # 如果模型已经构建完成，则直接返回，不进行重复构建
    if self.built:
        return
    # 将模型标记为已构建
    self.built = True
    # 共享/绑定的权重预期在模型的基本命名空间中
    # 在 tf.name_scope 后面添加 "/"（不是在前面！）将其放置在根命名空间而不是当前命名空间
    with tf.name_scope(self.shared.load_weight_prefix + "/" + self.shared.name + "/"):
        # 构建共享模型
        self.shared.build(None)
    # 如果存在编码器，则构建编码器
    if getattr(self, "encoder", None) is not None:
        with tf.name_scope(self.encoder.name):
            self.encoder.build(None)
    # 如果存在解码器，则构建解码器
    if getattr(self, "decoder", None) is not None:
        with tf.name_scope(self.decoder.name):
            self.decoder.build(None)
# 使用装饰器添加模型文档字符串，说明这是一个不带特定头部的原始隐藏状态的 BLENDERBOT 模型
@add_start_docstrings(
    "The bare BLENDERBOT Model outputting raw hidden-states without any specific head on top.",
    BLENDERBOT_START_DOCSTRING,
)
# 定义 TFBlenderbotModel 类，继承自 TFBlenderbotPreTrainedModel
class TFBlenderbotModel(TFBlenderbotPreTrainedModel):
    # 初始化方法
    def __init__(self, config: BlenderbotConfig, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 创建 TFBlenderbotMainLayer 实例，并将其保存为模型的属性
        self.model = TFBlenderbotMainLayer(config, name="model")

    # 获取编码器的方法
    def get_encoder(self):
        return self.model.encoder

    # 获取解码器的方法
    def get_decoder(self):
        return self.model.decoder

    # 从预训练模型加载模型
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        # 如果预训练模型是 "facebook/blenderbot-90M"，则警告用户使用 "facebook/small_blenderbot-90M" 替代
        if pretrained_model_name_or_path == "facebook/blenderbot-90M":
            from ..blenderbot_small import TFBlenderbotSmallModel
            warnings.warn(
                "The checkpoint `facebook/blenderbot-90M` is deprecated. In the future, please use the identical"
                " checkpoint `facebook/small_blenderbot-90M` with"
                " `TFBlenderbotSmallForConditionalGeneration.from_pretrained('facebook/small_blenderbot-90M')`"
                " instead.",
                FutureWarning,
            )
            # 返回 TFBlenderbotSmallModel 实例
            return TFBlenderbotSmallModel.from_pretrained(pretrained_model_name_or_path)
        
        # 如果预训练模型不是 "facebook/blenderbot-90M"，则调用父类的 from_pretrained 方法加载模型
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    # 定义 call 方法，模型前向传播的具体实现
    @unpack_inputs
    @add_start_docstrings_to_model_forward(BLENDERBOT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
    # 定义一个方法，接受一系列输入并返回模型输出或元组
    def __call__(
        self,
        input_ids=None,  # 输入的token ID
        attention_mask=None,  # 注意力掩码，指示哪些token需要被关注
        decoder_input_ids=None,  # 解码器的token ID
        decoder_attention_mask=None,  # 解码器的注意力掩码
        decoder_position_ids=None,  # 解码器的位置 ID
        head_mask=None,  # 注意力头的掩码
        decoder_head_mask=None,  # 解码器注意力头的掩码
        cross_attn_head_mask=None,  # 交叉注意力头的掩码
        encoder_outputs=None,  # 编码器的输出
        past_key_values=None,  # 过去的键值对
        inputs_embeds=None,  # 输入的嵌入向量
        decoder_inputs_embeds=None,  # 解码器输入的嵌入向量
        use_cache=None,  # 是否使用缓存
        output_attentions=None,  # 是否输出注意力权重
        output_hidden_states=None,  # 是否输出隐藏状态
        return_dict=None,  # 是否以字典形式返回结果
        training=None,  # 是否处于训练模式
    ) -> Union[Tuple[tf.Tensor], TFSeq2SeqModelOutput]:
        # 调用模型进行前向传播，获取输出
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

    # 从transformers.models.bart.modeling_tf_bart.TFBartModel.serving_output中复制过来的方法
    def serving_output(self, output):
        # 如果配置中使用了缓存，则获取过去的键值对
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        # 如果配置中输出了隐藏状态，则将解码器的隐藏状态转换为张量
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置中输出了注意力权重，则将解码器的注意力权重转换为张量
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        # 如果配置中输出了注意力权重，则将交叉注意力权重转换为张量
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        # 如果配置中输出了隐藏状态，则将编码器的隐藏状态转换为张量
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置中输出了注意力权重，则将编码器的注意力权重转换为张量
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None

        # 返回模型输出
        return TFSeq2SeqModelOutput(
            last_hidden_state=output.last_hidden_state,  # 最后一个隐藏状态
            past_key_values=pkv,  # 过去的键值对
            decoder_hidden_states=dec_hs,  # 解码器的隐藏状态
            decoder_attentions=dec_attns,  # 解码器的注意力权重
            cross_attentions=cross_attns,  # 交叉注意力权重
            encoder_last_hidden_state=output.encoder_last_hidden_state,  # 编码器的最后一个隐藏状态
            encoder_hidden_states=enc_hs,  # 编码器的隐藏状态
            encoder_attentions=enc_attns,  # 编码器的注意力权重
        )

    # 构建方法
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 将构建状态设置为True
        self.built = True
        # 如果存在模型，则在其名称域下构建模型
        if getattr(self, "model", None) is not None:
            with tf.name_scope(self.model.name):
                # 构建模型，传入空的输入形状
                self.model.build(None)
# 从transformers.models.bart.modeling_tf_bart.BiasLayer中复制的BiasLayer类
class BiasLayer(tf.keras.layers.Layer):
    """
    Bias作为一个层。它用于序列化目的：`tf.keras.Model.save_weights` 按层保存，所以所有权重必须在一个层中注册。
    """

    def __init__(self, shape, initializer, trainable, name, **kwargs):
        super().__init__(name=name, **kwargs)
        # 注意：当序列化时，此变量的名称不会被作用域化，即不会是格式为"outer_layer/inner_layer/.../name:0"的格式。
        # 而是"名称:0"。更多详情请参见：
        # https://github.com/huggingface/transformers/pull/18833#issuecomment-1233090214
        # 添加偏置权重，作为该层的权重
        self.bias = self.add_weight(name=name, shape=shape, initializer=initializer, trainable=trainable)

    def call(self, x):
        # 返回输入张量和偏置张量的和
        return x + self.bias


@add_start_docstrings(
    "带有语言建模头部的BLENDERBOT模型。可用于摘要。",
    BLENDERBOT_START_DOCSTRING,
)
class TFBlenderbotForConditionalGeneration(TFBlenderbotPreTrainedModel, TFCausalLanguageModelingLoss):
    # 在加载时忽略的键
    _keys_to_ignore_on_load_unexpected = [
        r"model.encoder.embed_tokens.weight",
        r"model.decoder.embed_tokens.weight",
    ]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 初始化BLENDERBOT主层
        self.model = TFBlenderbotMainLayer(config, name="model")
        # 是否使用缓存
        self.use_cache = config.use_cache
        # final_bias_logits在pytorch中被注册为缓冲区，为了一致性不可训练。
        # 创建BiasLayer实例作为最终偏置
        self.bias_layer = BiasLayer(
            name="final_logits_bias", shape=[1, config.vocab_size], initializer="zeros", trainable=False
        )

    def get_decoder(self):
        # 获取解码器
        return self.model.decoder

    def get_encoder(self):
        # 获取编码器
        return self.model.encoder

    def get_output_embeddings(self):
        # 获取输出嵌入层
        return self.get_input_embeddings()

    def set_output_embeddings(self, value):
        # 设置输出嵌入层
        self.set_input_embeddings(value)

    def get_bias(self):
        # 获取偏置
        return {"final_logits_bias": self.bias_layer.bias}

    def set_bias(self, value):
        # 替换包含偏置的现有层，以便进行正确的（反）序列化
        vocab_size = value["final_logits_bias"].shape[-1]
        # 创建新的BiasLayer实例
        self.bias_layer = BiasLayer(
            name="final_logits_bias", shape=[1, vocab_size], initializer="zeros", trainable=False
        )
        # 将给定的偏置值分配给新的偏置层
        self.bias_layer.bias.assign(value["final_logits_bias"])

    @classmethod
    # 类方法：从预训练模型中加载模型实例
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        # 如果预训练模型名称或路径为 "facebook/blenderbot-90M"
        if pretrained_model_name_or_path == "facebook/blenderbot-90M":
            # 导入小型 Blenderbot 模型类
            from ..blenderbot_small import TFBlenderbotSmallForConditionalGeneration

            # 发出警告，说明该检查点已被弃用
            warnings.warn(
                "The checkpoint `facebook/blenderbot-90M` is deprecated. In the future, please use the identical"
                " checkpoint `facebook/small_blenderbot-90M` with"
                " `TFBlenderbotSmallForConditionalGeneration.from_pretrained('facebook/small_blenderbot-90M')`"
                " instead.",
                FutureWarning,
            )
            # 返回预训练模型实例
            return TFBlenderbotSmallForConditionalGeneration.from_pretrained(pretrained_model_name_or_path)

        # 调用父类的 from_pretrained 方法
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    # 装饰器：解包输入
    @unpack_inputs
    # 装饰器：向模型前向方法添加文档字符串
    @add_start_docstrings_to_model_forward(BLENDERBOT_INPUTS_DOCSTRING)
    # 装饰器：替换返回值文档字符串
    @replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # 装饰器：向模型后向方法添加文档字符串
    @add_end_docstrings(BLENDERBOT_GENERATION_EXAMPLE)
    # 模型前向方法定义
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
        labels: tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[Tuple[tf.Tensor], TFSeq2SeqLMOutput]:
        r"""
        labels (`tf.tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        """
        # 如果传入了标签，则根据条件重新设置标签，将 pad_token_id 对应的标签设为 -100
        if labels is not None:
            labels = tf.where(
                labels == self.config.pad_token_id,
                tf.cast(tf.fill(shape_list(labels), -100), labels.dtype),
                labels,
            )
            # 禁用缓存，因为标签已经改变
            use_cache = False
            # 如果 decoder_input_ids 和 decoder_inputs_embeds 都未提供，则根据标签生成 decoder_input_ids
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
        # 计算 LM logits
        lm_logits = tf.matmul(outputs[0], self.model.shared.weights, transpose_b=True)
        # 添加偏置
        lm_logits = self.bias_layer(lm_logits)
        # 如果传入了标签，则计算 masked language modeling loss
        masked_lm_loss = None if labels is None else self.hf_compute_loss(labels, lm_logits)

        # 如果不要求返回字典，则返回输出元组
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        # 如果要求返回字典，则返回 TFSeq2SeqLMOutput 对象
        return TFSeq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,  # 模型输出的 past key values，对应于 d outputs 的索引 1
            decoder_hidden_states=outputs.decoder_hidden_states,  # 模型输出的 decoder hidden states，对应于 d outputs 的索引 2
            decoder_attentions=outputs.decoder_attentions,  # 模型输出的 decoder attentions，对应于 d outputs 的索引 3
            cross_attentions=outputs.cross_attentions,  # 模型输出的 cross attentions，对应于 d outputs 的索引 4
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,  # 模型输出的 encoder last hidden state，对应于 encoder outputs 的索引 0
            encoder_hidden_states=outputs.encoder_hidden_states,  # 模型输出的 encoder hidden states，对应于 encoder outputs 的索引 1
            encoder_attentions=outputs.encoder_attentions,  # 模型输出的 encoder attentions，对应于 encoder outputs 的索引 2
        )

    # 从 transformers.models.bart.modeling_tf_bart.TFBartForConditionalGeneration.serving_output 复制而来
```py  
    # 定义一个方法，用于处理模型的输出并返回适合服务的格式
    def serving_output(self, output):
        # 如果配置中使用了缓存，则获取解码器的过去键值对中的值，否则设为 None
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        # 如果配置中输出了隐藏状态，则将解码器的隐藏状态转换为张量，否则设为 None
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置中输出了注意力权重，则将解码器的注意力权重转换为张量，否则设为 None
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        # 如果配置中输出了注意力权重，则将交叉注意力的权重转换为张量，否则设为 None
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        # 如果配置中输出了隐藏状态，则将编码器的隐藏状态转换为张量，否则设为 None
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置中输出了注意力权重，则将编码器的注意力权重转换为张量，否则设为 None
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None

        # 返回转换后的输出
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

    # 从transformers.models.bart.modeling_tf_bart.TFBartForConditionalGeneration.prepare_inputs_for_generation中复制而来
    # 准备用于生成的输入，根据给定的参数进行适当的调整
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

        # 如果有解码器的注意力掩码，则计算解码器位置标识
        if decoder_attention_mask is not None:  # xla
            decoder_position_ids = tf.math.cumsum(decoder_attention_mask, axis=-1, exclusive=True)[:, -1:]
        # 如果使用了过去的键值对，则解码器位置标识为过去键值对的第一个键的形状
        elif past_key_values is not None:  # no xla + past_key_values
            decoder_position_ids = past_key_values[0][0].shape[2]
        # 否则，解码器位置标识为解码器输入的标记数的范围
        else:  # no xla + no past_key_values
            decoder_position_ids = tf.range(decoder_input_ids.shape[1])

        # 返回准备好的输入字典
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
            "use_cache": use_cache,  # 更改此参数以避免缓存（可能用于调试）
        }
    # 根据给定的输入形状构建模型，如果已经构建过则直接返回
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回，不做任何操作
        if self.built:
            return
        # 将构建标志设置为 True，表示模型已经构建
        self.built = True
        # 如果模型存在，则为模型设置命名空间并构建模型
        if getattr(self, "model", None) is not None:
            # 使用模型的名称创建命名空间
            with tf.name_scope(self.model.name):
                # 构建模型，传入空的输入形状
                self.model.build(None)
        # 如果存在偏置层，则为偏置层设置命名空间并构建偏置层
        if getattr(self, "bias_layer", None) is not None:
            # 使用偏置层的名称创建命名空间
            with tf.name_scope(self.bias_layer.name):
                # 构建偏置层，传入空的输入形状
                self.bias_layer.build(None)
```