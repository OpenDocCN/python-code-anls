# `.\transformers\models\opt\modeling_tf_opt.py`

```py
# 设置文件编码为utf-8
# 版权声明
#
# 引入必要的模块和函数
from __future__ import annotations
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutputWithPast, TFCausalLMOutputWithPast
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    TFSharedEmbeddings,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_opt import OPTConfig
# 设置日志
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = "facebook/opt-350m"  # 用于文档的检查点名称
_CONFIG_FOR_DOC = "OPTConfig"  # 用于文档的配置名称
# 预期的输出形状
_EXPECTED_OUTPUT_SHAPE = [1, 8, 1024]
# Causal LM output的预期输出
_CAUSAL_LM_EXPECTED_OUTPUT = "Hey, are you conscious? Can you talk to me?\nI'm not conscious. I'm just a little bit of a weirdo."

# 定义函数_make_causal_mask，用于创建用于双向自注意力的因果掩码
def _make_causal_mask(input_ids_shape: tf.TensorShape, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz = input_ids_shape[0]
    tgt_len = input_ids_shape[1]
    # 生成一个和输入长度相同的掩码矩阵，上三角部分设为-LARGE_NEGATIVE，然后将其中的0值替换为-LARGE_NEGATIVE
    mask = tf.fill((tgt_len, tgt_len), tf.cast(LARGE_NEGATIVE, tf.float32))
    mask = tf.linalg.band_part(mask, 0, -1) - tf.linalg.band_part(mask, 0, 0)
    # 如果有历史键值对的长度大于0，则在掩码矩阵的左侧填充0
    if past_key_values_length > 0:
        mask = tf.concat([tf.zeros((tgt_len, past_key_values_length)), mask], axis=-1)
    return tf.tile(mask[None, None, :, :], (bsz, 1, 1, 1))

# 定义函数_expand_mask，用于扩展注意力掩码
def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    # 获取输入掩码的长度
    src_len = shape_list(mask)[1]
    # 如果未指定目标长度，则设为源长度
    tgt_len = tgt_len if tgt_len is not None else src_len
    one_cst = tf.constant(1.0)
    mask = tf.cast(mask, dtype=one_cst.dtype)
    # 将掩码扩展为指定的目标长度
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))
    return (one_cst - expanded_mask) * LARGE_NEGATIVE
# 这个类继承自 tf.keras.layers.Embedding，用于学习可训练的位置编码
class TFOPTLearnedPositionalEmbedding(tf.keras.layers.Embedding):
    """
    这个模块学习的位置编码最大长度是固定的。
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs):
        # OPT 设置中，如果指定了 padding_idx，则位置编码 id 要偏移 2，并相应地调整 num_embeddings
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim, **kwargs)

    # 该方法根据 attention_mask 计算位置编码
    def call(self, attention_mask, past_key_values_length: int = 0):
        # attention_mask 的 shape 应该是 [bsz x seqlen]
        attention_mask = tf.cast(attention_mask, tf.int64)

        # 根据 attention_mask 创建位置编码
        positions = tf.math.cumsum(attention_mask, axis=1) * attention_mask - 1

        # 如果有 past_key_values_length，则截断位置编码
        positions = positions[:, past_key_values_length:]

        # 返回位置编码，并加上偏移量 offset
        return super().call(positions + self.offset)


# 该类复制自 transformers.models.bart.modeling_tf_bart.TFBartAttention，将 Bart 替换成 OPT
class TFOPTAttention(tf.keras.layers.Layer):
    """来自"Attention Is All You Need"的多头注意力机制"""

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
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.head_dim = embed_dim // num_heads
        # 检查 embed_dim 是否能被 num_heads 整除
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        # 创建用于计算 key、query 和 value 的全连接层
        self.k_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="k_proj")
        self.q_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="q_proj")
        self.v_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="v_proj")
        self.out_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="out_proj")

    # 该方法用于调整张量的形状
    def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
        return tf.transpose(tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))

    # 该方法实现了注意力机制的前向传播
    def call(
        self,
        hidden_states: tf.Tensor,
        key_value_states: tf.Tensor | None = None,
        past_key_value: Tuple[Tuple[tf.Tensor]] | None = None,
        attention_mask: tf.Tensor | None = None,
        layer_head_mask: tf.Tensor | None = None,
        training: Optional[bool] = False,
    # 定义 build 方法，input_shape 参数为 None
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 将 built 属性设为 True，表示已构建
        self.built = True
        # 如果 k_proj 属性存在，构建 k_proj
        if getattr(self, "k_proj", None) is not None:
            # 使用 k_proj 的名字创建一个名称域
            with tf.name_scope(self.k_proj.name):
                # 构建 k_proj，输入形状为 [None, None, embed_dim]
                self.k_proj.build([None, None, self.embed_dim])
        # 如果 q_proj 属性存在，构建 q_proj
        if getattr(self, "q_proj", None) is not None:
            # 使用 q_proj 的名字创建一个名称域
            with tf.name_scope(self.q_proj.name):
                # 构建 q_proj，输入形状为 [None, None, embed_dim]
                self.q_proj.build([None, None, self.embed_dim])
        # 如果 v_proj 属性存在，构建 v_proj
        if getattr(self, "v_proj", None) is not None:
            # 使用 v_proj 的名字创建一个名称域
            with tf.name_scope(self.v_proj.name):
                # 构建 v_proj，输入形状为 [None, None, embed_dim]
                self.v_proj.build([None, None, self.embed_dim])
        # 如果 out_proj 属性存在，构建 out_proj
        if getattr(self, "out_proj", None) is not None:
            # 使用 out_proj 的名字创建一个名称域
            with tf.name_scope(self.out_proj.name):
                # 构建 out_proj，输入形状为 [None, None, embed_dim]
                self.out_proj.build([None, None, self.embed_dim])
# 定义一个 TensorFlow 的 OPT (Open-ended Transformer) 解码器层类
class TFOPTDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: OPTConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 是否在注意力和前馈神经网络之前进行层归一化
        self.do_layer_norm_before = config.do_layer_norm_before
        # 嵌入维度大小
        self.embed_dim = config.hidden_size
        # 创建自注意力层
        self.self_attn = TFOPTAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            name="self_attn",
            is_decoder=True,
        )
        # 创建dropout层
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        # 获取激活函数
        self.activation_fn = get_tf_activation(config.activation_function)

        # 创建层归一化层
        self.self_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        # 创建两个全连接层
        self.fc1 = tf.keras.layers.Dense(config.ffn_dim, name="fc1")
        self.fc2 = tf.keras.layers.Dense(self.embed_dim, name="fc2")
        # 创建最终的层归一化层
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        # 保存配置信息
        self.config = config

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        layer_head_mask: tf.Tensor | None = None,
        past_key_value: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        training: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ):


I am an AI language model known as GPT-3.5, trained by OpenAI. I have been designed to assist with a wide range of tasks, including providing detailed explanations and commenting on code. The above code represents a TensorFlow implementation of an OPT (Open-ended Transformer) decoder layer. Each line of the code has been annotated to explain its purpose and functionality.
    ) -> Tuple[tf.Tensor, tf.Tensor, Tuple[Tuple[tf.Tensor]]]:
        """
        Args:
            hidden_states (`tf.Tensor`): 输入到层的张量，形状为 `(batch, seq_len, embed_dim)`
            attention_mask (`tf.Tensor`, *可选*): 大小为 `(batch, 1, tgt_len, src_len)` 的注意力掩码，其中填充元素用非常大的负值表示。
            layer_head_mask (`tf.Tensor`, *可选*): 给定层中注意力头的掩码，大小为 `(decoder_attention_heads,)`
            past_key_value (`Tuple(tf.Tensor)`, *可选*): 缓存的过去键和值投影状态
            training (`bool`, *可选*, 默认为 `False`):
                是否在训练模式下使用模型（某些模块如 dropout 在训练和评估时有不同的行为）。
        """
        # 保留残差连接
        residual = hidden_states

        # 如果在注意力之前进行层归一化
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # 自注意力机制
        # 缓存的解码器单向自注意力键/值元组位于位置 1、2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None

        # 将当前自注意力缓存添加到 present_key_value 元组的位置 1、2
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
        )
        # 应用 dropout
        hidden_states = self.dropout(hidden_states, training=training)
        # 残差连接
        hidden_states = residual + hidden_states

        # 如果不在注意力之前进行层归一化
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # 全连接层
        residual = hidden_states
        # 如果在注意力之前进行层归一化
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = residual + hidden_states

        # 如果不在注意力之前进行层归一化
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        return (hidden_states, self_attn_weights, present_key_value)
    # 构建神经网络模型，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在 self_attn 层，则构建 self_attn
        if getattr(self, "self_attn", None) is not None:
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)
        # 如果存在 self_attn_layer_norm 层，则构建 self_attn_layer_norm
        if getattr(self, "self_attn_layer_norm", None) is not None:
            with tf.name_scope(self.self_attn_layer_norm.name):
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        # 如果存在 fc1 层，则构建 fc1
        if getattr(self, "fc1", None) is not None:
            with tf.name_scope(self.fc1.name):
                self.fc1.build([None, None, self.embed_dim])
        # 如果存在 fc2 层，则构建 fc2
        if getattr(self, "fc2", None) is not None:
            with tf.name_scope(self.fc2.name):
                self.fc2.build([None, None, self.config.ffn_dim])
        # 如果存在 final_layer_norm 层，则构建 final_layer_norm
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.embed_dim])
# 输入文档字符串模板
OPT_START_DOCSTRING = r"""
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
        config ([`OPTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare OPT Model outputting raw hidden-states without any specific head on top.",  # 添加模型输出说明文档
    OPT_START_DOCSTRING,
)
class TFOPTPreTrainedModel(TFPreTrainedModel):
    """
    TFOPT Pretrained Model that inheritates from transformers.TFPreTrainedModel

    Args:
        config: OPTConfig
    """

    config_class = OPTConfig  # 模型配置类
    base_model_prefix = "model"  # 基础模型前缀


OPT_INPUTS_DOCSTRING = r"""
"""
    # 输入序列的索引值，是从词汇表中取得的
    Args:
        input_ids (`tf.Tensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.
    
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
    
            [What are input IDs?](../glossary#input-ids)
    # 注意力掩码，用于避免对填充标记进行注意力计算
        attention_mask (`tf.Tensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
    
            [What are attention masks?](../glossary#attention-mask)
    # 编码器注意力头的掩码，用于屏蔽某些注意力头
        head_mask (`tf.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:
    
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
    
        past_key_values (`Tuple[Tuple[tf.Tensor]]` of length `config.n_layers`)
            contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
    # 是否使用缓存的key/value状态来加速解码
        use_cache (`bool`, *optional*, defaults to `True`):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`). Set to `False` during training, `True` during generation
    # 是否返回注意力张量
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
            config will be used instead.
    # 是否返回所有层的隐藏状态
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
    # 是否返回一个`~utils.ModelOutput`对象而不是元组
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in
            eager mode, in graph mode the value will always be set to True.
    # 是否为训练模式(一些模块如dropout会有不同行为)
        training (`bool`, *optional*, defaults to `False`):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
# 导入 keras_serializable 装饰器
@keras_serializable
# 定义 TFOPTDecoder 类，继承自 tf.keras.layers.Layer
class TFOPTDecoder(tf.keras.layers.Layer):
    # 设置配置类为 OPTConfig
    config_class = OPTConfig

    # 初始化方法，接受配置参数 config 和其他关键字参数
    def __init__(self, config: OPTConfig, **kwargs):
        # 调用父类初始化方法
        super().__init__(**kwargs)
        # 保存配置参数
        self.config = config
        # 设置填充索引为配置中的 pad_token_id
        self.padding_idx = config.pad_token_id
        # 设置可选层的丢弃比例为配置中的 layerdrop
        self.layerdrop = config.layerdrop
        # 设置词嵌入矩阵的维度
        num_embeddings = config.max_position_embeddings
        # 创建共享词嵌入层对象 embed_tokens
        self.embed_tokens = TFSharedEmbeddings(
            config.vocab_size, config.word_embed_proj_dim, config.pad_token_id, name="embed_tokens"
        )
        # 创建位置嵌入层对象 embed_positions
        self.embed_positions = TFOPTLearnedPositionalEmbedding(
            num_embeddings,
            config.hidden_size,
            name="embed_positions",
        )

        # _remove_final_layer_norm 仅用于保持与 transformers v4.20.1 之前微调过的检查点的向后兼容性
        # 参考 https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            # 创建最终层归一化层对象 final_layer_norm
            self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        else:
            self.final_layer_norm = None

        # 如果词嵌入投影维度不等于隐藏层维度
        if config.word_embed_proj_dim != config.hidden_size:
            # 创建输出投影层对象 project_out 以及输入投影层对象 project_in
            self.project_out = tf.keras.layers.Dense(config.word_embed_proj_dim, name="project_out", use_bias=False)
            self.project_in = tf.keras.layers.Dense(config.hidden_size, name="project_in", use_bias=False)

        else:
            self.project_in = None
            self.project_out = None

        # 创建多个 OPT 解码器层对象，并存储在列表中
        self.layers = [TFOPTDecoderLayer(config, name=f"layers.{i}") for i in range(config.num_hidden_layers)]
        # 创建丢弃层对象 dropout
        self.dropout = tf.keras.layers.Dropout(config.dropout)

    # 获取词嵌入矩阵
    def get_embed_tokens(self):
        return self.embed_tokens

    # 设置词嵌入矩阵
    def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens

    # 设置输入词嵌入矩阵
    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens.vocab_size = new_embeddings.shape[0]
        self.embed_tokens.weight = new_embeddings

    # 获取输入词嵌入矩阵
    def get_input_embeddings(self):
        return self.embed_tokens
    # 定义_prepare_decoder_attention_mask()函数，用于准备解码器的注意力掩码
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, past_key_values_length):
        # 创建因果掩码
        # [batch_size, seq_len] -> [batch_size, 1, tgt_seq_len, src_seq_len]
        # 确保注意力掩码的形状正确
        _, seq_length = input_shape
        tf.debugging.assert_equal(
            seq_length + past_key_values_length,
            shape_list(attention_mask)[1],
            message="Attention mask shape should be (batch_size, seq_length + past_key_values_length)"
            f" but is {shape_list(attention_mask)[1]} with input_ids shape {input_shape} and past length"
            f" {past_key_values_length}.",
        )
    
        # 将注意力掩码沿着 tgt_seq_len 方向扩展
        expanded_attn_mask = _expand_mask(attention_mask, tgt_len=input_shape[-1])
        # 如果 seq_length 大于1，则将因果掩码和扩展的注意力掩码进行合并
        if seq_length > 1:
            combined_attention_mask = (
                _make_causal_mask(input_shape, past_key_values_length=past_key_values_length) + expanded_attn_mask
            )
        else:
            # 否则，只使用扩展的注意力掩码
            combined_attention_mask = expanded_attn_mask
    
        return combined_attention_mask
    
    # 定义call()函数，用于执行模型的前向传播
    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
        
    # 定义build()函数，用于构建模型的层
    def build(self, input_shape=None):
        # 如果模型已经构建完毕，则直接返回
        if self.built:
            return
        self.built = True
        # 构建词嵌入层
        if getattr(self, "embed_tokens", None) is not None:
            with tf.name_scope(self.embed_tokens.name):
                self.embed_tokens.build(None)
        # 构建位置编码层
        if getattr(self, "embed_positions", None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)
        # 构建最终层归一化层
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.config.hidden_size])
        # 构建输出层
        if getattr(self, "project_out", None) is not None:
            with tf.name_scope(self.project_out.name):
                self.project_out.build([None, None, self.config.hidden_size])
        # 构建输入层
        if getattr(self, "project_in", None) is not None:
            with tf.name_scope(self.project_in.name):
                self.project_in.build([None, None, self.config.word_embed_proj_dim])
        # 逐层构建编码器层
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)
@keras_serializable
# 声明 TFOPTMainLayer 类，该类是可序列化的
class TFOPTMainLayer(tf.keras.layers.Layer):
    # 使用 OPTConfig 作为配置类
    config_class = OPTConfig

    def __init__(self, config: OPTConfig, **kwargs):
        # 调用父类构造函数
        super().__init__(**kwargs)
        # 将传入的 config 参数赋值给 self.config
        self.config = config
        # 创建 TFOPTDecoder 对象，使用传入的 config 参数作为配置
        self.decoder = TFOPTDecoder(config, name="decoder")

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    # 设置输入嵌入
    def set_input_embeddings(self, new_embeddings):
        self.decoder.set_input_embeddings(new_embeddings)

    # 定义 call 方法
    # *kwargs 表示接收任意数量的关键字参数
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
        # 判断 output_attentions 参数是否为 None，如果是则使用 self.config.output_attentions
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 判断 output_hidden_states 参数是否为 None，如果是则使用 self.config.output_hidden_states
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 判断 use_cache 参数是否为 None，如果是则使用 self.config.use_cache
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # 判断 return_dict 参数是否为 None，如果是则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 self.decoder 的 call 方法进行前向传播，得到输出结果
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

        # 如果 return_dict 为 False，则直接返回输出结果
        if not return_dict:
            return outputs

        # 否则，将输出结果包装成 TFBaseModelOutputWithPast 类型并返回
        return TFBaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 设置为已构建状态
        self.built = True
        # 如果 self.decoder 存在，则在命名空间下构建 self.decoder
        if getattr(self, "decoder", None) is not None:
            with tf.name_scope(self.decoder.name):
                self.decoder.build(None)


@add_start_docstrings(
    "The bare TF OPT Model outputting raw hidden-states without any specific head on top.",
    OPT_START_DOCSTRING,
)
@keras_serializable
# 声明 TFOPTModel 类，该类是可序列化的
class TFOPTModel(TFOPTPreTrainedModel):
    # 使用 OPTConfig 作为配置类
    config_class = OPTConfig

    def __init__(self, config: OPTConfig, **kwargs):
        # 调用父类构造函数
        super().__init__(config, **kwargs)
        # 将传入的 config 参数赋值给 self.config
        self.config = config
        # 创建 TFOPTMainLayer 对象，使用传入的 config 参数作为配置
        self.model = TFOPTMainLayer(config, name="model")
    # 获取输入嵌入层，即模型解码器的嵌入层
    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    # 设置输入嵌入层，即用新的嵌入层替换当前模型解码器的嵌入层
    def set_input_embeddings(self, new_embeddings):
        self.model.set_input_embeddings(new_embeddings)

    # 将模型方法进行装饰，添加文档字符串和代码示例的注释，用于模型调用
    @unpack_inputs
    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    # 定义模型的调用方法
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
    ) -> Union[TFBaseModelOutputWithPast,
    # 构建神经网络模型
    def build(self, input_shape=None):
        # 如果模型已被构建，则直接返回
        if self.built:
            return
        # 将 built 属性设置为 True，表示模型已被构建
        self.built = True
        # 如果 self.model 属性不为 None
        if getattr(self, "model", None) is not None:
            # 使用 TensorFlow 的名称域 scope 来构建模型
            with tf.name_scope(self.model.name):
                # 调用 self.model 的 build 方法构建模型，输入形状为 None
                self.model.build(None)
# 导入必要的模块和类
@add_start_docstrings(
    """
    The OPT Model transformer with a language modeling head on top.
    """,
    OPT_START_DOCSTRING,
)
@keras_serializable
class TFOPTForCausalLM(TFOPTPreTrainedModel, TFCausalLanguageModelingLoss):
    config_class = OPTConfig

    def __init__(self, config: OPTConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.model = TFOPTMainLayer(config, name="model")

    # 获取输出的嵌入层
    def get_output_embeddings(self):
        return self.model.get_input_embeddings()

    # 为生成准备输入数据
    def prepare_inputs_for_generation(self, inputs, past_key_values=None, use_cache=None, **kwargs):
        attention_mask = kwargs.get("attention_mask", None)

        # 如果过去的键值存在，则只使用最后一个标记作为输入
        if past_key_values:
            inputs = tf.expand_dims(inputs[:, -1], -1)

        return {
            "input_ids": inputs,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    # 调用 OPT 模型进行前向传播
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
    def serving_output(self, output):
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFCausalLMOutputWithPast(
            past_key_values=pkv,
            hidden_states=hs,
            attentions=attns,
            loss=output.loss,
            logits=output.logits,
        )

    # 构建模型
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "model", None) is not None:
            with tf.name_scope(self.model.name):
                self.model.build(None)
```