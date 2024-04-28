# `.\transformers\models\speech_to_text\modeling_tf_speech_to_text.py`

```
# 设置文件编码格式为 utf-8
# 版权声明
#
# 此处为 Apache License, Version 2.0 的授权许可链接
#
# 预定义变量，用于表示大的负数
LARGE_NEGATIVE = -1e8

from __future__ import annotations
import random
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf 模块引入 get_tf_activation 和 glu 函数
from ...modeling_tf_outputs 模块引入各种 TFBaseModelOutput* 类
from ...modeling_tf_utils 模块引入各种工具函数
from ...tf_utils 模块引入 check_embeddings_within_bounds、shape_list、stable_softmax 函数
from ...utils 模块引入各种工具函数
from .configuration_speech_to_text 模块引入 Speech2TextConfig 类
# 查看是否超出边界
check_embeddings_within_bounds(input_embeddings: tf.Tensor, 
                                bounds: Tuple[float, float], 
                                message: str) -> None:

# 定义了一个函数 shift_tokens_right，作用是将输入的 input_ids 右移一位
def shift_tokens_right(input_ids: tf.Tensor, pad_token_id: int, decoder_start_token_id: int):
    pad_token_id = tf.cast(pad_token_id, input_ids.dtype)
    decoder_start_token_id = tf.cast(decoder_start_token_id, input_ids.dtype)
    start_tokens = tf.fill(
        (shape_list(input_ids)[0], 1), tf.convert_to_tensor(decoder_start_token_id, input_ids.dtype)
    )
    shifted_input_ids = tf.concat([start_tokens, input_ids[:, :-1]], -1)
    # 将 labels 中可能存在的-100值替换为 pad_token_id
    shifted_input_ids = tf.where(
        shifted_input_ids == -100,
        tf.fill(shape_list(shifted_input_ids), tf.convert_to_tensor(pad_token_id, input_ids.dtype)),
        shifted_input_ids,
    )

    # 验证 labels 只包含正值和-100
    assert_gte0 = tf.debugging.assert_greater_equal(shifted_input_ids, tf.constant(0, dtype=input_ids.dtype))

    # 通过添加 identity 操作确保执行了断言操作
    # 使用assert_gte0作为控制依赖，确保shifted_input_ids大于等于0
    with tf.control_dependencies([assert_gte0]):
        # 创建shifted_input_ids的副本
        shifted_input_ids = tf.identity(shifted_input_ids)
    
    # 返回shifted_input_ids
    return shifted_input_ids
# 从transformers.models.bart.modeling_tf_bart中复制过来的函数，用于生成自注意力的时序掩码
def _make_causal_mask(input_ids_shape: tf.TensorShape, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    生成用于双向自注意力的时序掩码。
    """
    # 获取输入id的形状
    bsz = input_ids_shape[0]
    tgt_len = input_ids_shape[1]
    # 生成掩码，全部设置为LARGE_NEGATIVE
    mask = tf.ones((tgt_len, tgt_len)) * LARGE_NEGATIVE
    # 判断mask_cond是否小于mask_cond + 1，是则赋值为0.0
    mask_cond = tf.range(shape_list(mask)[-1])
    mask = tf.where(mask_cond < tf.reshape(mask_cond + 1, (shape_list(mask)[-1], 1)), 0.0, mask)
    # 如果past_key_values_length大于0，则在mask左侧补充0
    if past_key_values_length > 0:
        mask = tf.concat([tf.zeros((tgt_len, past_key_values_length)), mask], axis=-1)
    # 在batch维度上复制mask
    return tf.tile(mask[None, None, :, :], (bsz, 1, 1, 1))


# 从transformers.models.bart.modeling_tf_bart中复制过来的函数，将注意力掩码从`[bsz, seq_len]`扩展到`[bsz, 1, tgt_seq_len, src_seq_len]`
def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    将注意力掩码从`[bsz, seq_len]`扩展到`[bsz, 1, tgt_seq_len, src_seq_len]`。
    """
    # 获取输入掩码的长度
    src_len = shape_list(mask)[1]
    tgt_len = tgt_len if tgt_len is not None else src_len
    # 创建常量张量one_cst，并转换mask的数据类型为one_cst的数据类型
    one_cst = tf.constant(1.0)
    mask = tf.cast(mask, dtype=one_cst.dtype)
    # 在第2维度上复制mask
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))
    # 返回扩展后的掩码
    return (one_cst - expanded_mask) * LARGE_NEGATIVE


class TFConv1dSubsampler(tf.keras.layers.Layer):
    """
    Convolutional subsampler: a stack of 1D convolution (along temporal dimension) followed by non-linear activation
    via gated linear units (https://arxiv.org/abs/1911.08460)
    一维卷积下采样器：由一维卷积（沿时序维度）和非线性激活函数通过门控线性单元组成。
    """

    def __init__(self, config: Speech2TextConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.num_layers = config.num_conv_layers
        self.in_channels = config.input_feat_per_channel * config.input_channels
        self.mid_channels = config.conv_channels
        self.out_channels = config.d_model
        self.kernel_sizes = config.conv_kernel_sizes

        # 创建一维卷积层的列表，根据config中的参数设置
        self.conv_layers = [
            tf.keras.layers.Conv1D(
                filters=self.mid_channels if i < self.num_layers - 1 else self.out_channels * 2,
                kernel_size=k,
                strides=2,
                name=f"conv_layers.{i}",
            )
            for i, k in enumerate(self.kernel_sizes)
        ]
    def call(self, input_features: tf.Tensor) -> tf.Tensor:
        # TF Conv1D假设输入为Batch x Time x Channels，与输入相同
        hidden_states = tf.cast(input_features, tf.float32)
        for i, conv in enumerate(self.conv_layers):
            # 相当于PT中nn.Conv1d的`padding=k // 2`
            pad_len = self.kernel_sizes[i] // 2
            hidden_shapes = shape_list(hidden_states)
            hidden_states = tf.concat(
                (
                    tf.zeros((hidden_shapes[0], pad_len, hidden_shapes[2])),
                    hidden_states,
                    tf.zeros((hidden_shapes[0], pad_len, hidden_shapes[2])),
                ),
                axis=1,
            )

            hidden_states = conv(hidden_states)
            hidden_states = glu(hidden_states, axis=2)  # 在通道维度上进行GLU操作
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "conv_layers", None) is not None:
            for i, layer in enumerate(self.conv_layers):
                with tf.name_scope(layer.name):
                    layer.build([None, None, self.in_channels] if i == 0 else [None, None, self.mid_channels // 2])
class TFSpeech2TextSinusoidalPositionalEmbedding(tf.keras.layers.Layer):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.offset = 2  # 设置偏移量，用于生成位置嵌入
        self.embedding_dim = embedding_dim  # 嵌入维度
        self.padding_idx = padding_idx  # 填充索引
        self.embedding_weights = self._get_embedding(num_positions + self.offset, embedding_dim, padding_idx)  # 获取嵌入权重

    @staticmethod
    def _get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None) -> tf.Tensor:
        """
        Build sinusoidal embeddings. This matches the implementation in tensor2tensor, but differs slightly from the
        description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2  # 嵌入维度的一半
        emb = tf.math.log(10000.0) / (half_dim - 1)  # 计算频率参数
        emb = tf.math.exp(tf.range(half_dim, dtype=tf.float32) * -emb)  # 计算正弦和余弦的参数
        emb = tf.expand_dims(tf.range(num_embeddings, dtype=tf.float32), axis=1) * tf.expand_dims(emb, axis=0)  # 计算位置嵌入
        emb = tf.reshape(tf.concat([tf.math.sin(emb), tf.math.cos(emb)], axis=1), shape=[num_embeddings, -1])  # 组合正弦和余弦
        if embedding_dim % 2 == 1:
            # zero pad
            emb = tf.concat([emb, tf.zeros((num_embeddings, 1))], axis=1)  # 若嵌入维度为奇数，进行零填充
        if padding_idx is not None:
            emb = tf.concat([emb[:padding_idx, :], tf.zeros((1, tf.shape(emb)[1])), emb[padding_idx + 1 :, :]], axis=0)  # 处理填充索引
        return emb

    def call(self, input_ids: tf.Tensor, past_key_values_length: int = 0) -> tf.Tensor:
        bsz, seq_len = shape_list(input_ids)  # 获取输入张量的形状
        # Create the position ids from the input token ids. Any padded tokens remain padded.
        position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)  # 根据输入标记 ID 创建位置 ID

        # Matt: The PyTorch code does a lot of work to cache the embeddings, setting the cached values as a
        # model attribute in the forward pass. This is extremely forbidden in TF, which wants forward calls to be
        # idempotent. TF doesn't need that caching anyway, since it can just store constants during compilation,
        # so we just remove all of that code.
        embeddings = self._get_embedding(
            self.padding_idx + 1 + seq_len + self.offset + past_key_values_length, self.embedding_dim, self.padding_idx
        )  # 获取嵌入张量
        return tf.reshape(tf.gather(embeddings, tf.reshape(position_ids, (-1,)), axis=0), (bsz, seq_len, -1))  # 获取嵌入张量中对应位置的嵌入向量

    @staticmethod
    def create_position_ids_from_input_ids(
        input_ids: tf.Tensor, padding_idx: int, past_key_values_length: Optional[int] = 0
        ) -> tf.Tensor:
        # 定义一个函数，接受一个输入参数并返回一个张量
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
        symbols are ignored. This is modified from fairseq's `utils.make_positions`.

        Args:
            x: tf.Tensor x:
        Returns: tf.Tensor
        """
        # 创建一个遮罩，标记出非填充符号的位置
        mask = tf.cast(tf.math.not_equal(input_ids, padding_idx), dtype=tf.int32)
        # 计算增量索引，从padding_idx+1开始
        incremental_indices = (tf.math.cumsum(mask, axis=1) + past_key_values_length) * mask
        # 返回增量索引并添加padding_idx，转换数据类型为int64
        return tf.cast(incremental_indices, dtype=tf.int64) + padding_idx
# 从transformers.models.bart.modeling_tf_bart.TFBartAttention复制并修改成Speech2Text
class TFSpeech2TextAttention(tf.keras.layers.Layer):
    """从“Attention Is All You Need”中的多头注意力机制创建"""

    def __init__(
        self,
        embed_dim: int,  # 嵌入维度
        num_heads: int,  # 注意力头数
        dropout: float = 0.0,  # dropout率
        is_decoder: bool = False,  # 是否为解码器
        bias: bool = True,  # 是否使用偏置
        **kwargs,  # 其他参数
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim  # 初始化嵌入维度

        self.num_heads = num_heads  # 初始化注意力头数
        self.dropout = tf.keras.layers.Dropout(dropout)  # 定义Dropout层
        self.head_dim = embed_dim // num_heads  # 每个头的维度
        if (self.head_dim * num_heads) != self.embed_dim:  # 检查嵌入维度是否能整除头数
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5  # 缩放因子
        self.is_decoder = is_decoder  # 是否为解码器

        self.k_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="k_proj")  # 线性变换层，用于映射k
        self.q_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="q_proj")  # 线性变换层，用于映射q
        self.v_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="v_proj")  # 线性变换层，用于映射v
        self.out_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="out_proj")  # 线性变换层，用于最终输出

    def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
        return tf.transpose(tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))

    def call(
        self,
        hidden_states: tf.Tensor,  # 输入的隐藏状态
        key_value_states: tf.Tensor | None = None,  # 键值对的状态，可选
        past_key_value: Tuple[Tuple[tf.Tensor]] | None = None,  # 过去的键值对，可选
        attention_mask: tf.Tensor | None = None,  # 注意力掩码，可选
        layer_head_mask: tf.Tensor | None = None,  # 层头掩码，可选
        training: Optional[bool] = False,  # 是否训练模式
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
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


class TFSpeech2TextEncoderLayer(tf.keras.layers.Layer):
    # 初始化函数，接收配置参数和其他关键字参数
    def __init__(self, config: Speech2TextConfig, **kwargs):
        # 调用父类初始化函数
        super().__init__(**kwargs)
        # 设置嵌入维度
        self.embed_dim = config.d_model
        # 创建自注意力层对象
        self.self_attn = TFSpeech2TextAttention(
            self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout, name="self_attn"
        )
        # 创建自注意力层标准化层对象
        self.self_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        # 创建 Dropout 层对象
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        # 获取激活函数对象
        self.activation_fn = get_tf_activation(config.activation_function)
        # 创建激活函数 Dropout 层对象
        self.activation_dropout = tf.keras.layers.Dropout(config.activation_dropout)
        # 创建全连接层对象 fc1
        self.fc1 = tf.keras.layers.Dense(config.encoder_ffn_dim, name="fc1")
        # 创建全连接层对象 fc2
        self.fc2 = tf.keras.layers.Dense(self.embed_dim, name="fc2")
        # 创建最终输出层标准化层对象
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        # 保存配置参数
        self.config = config

    # 调用函数，执行前向传播
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
        # 保存残差连接
        residual = hidden_states
        # 对输入进行自注意力层标准化
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 调用自注意力层进行处理
        hidden_states, self_attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            training=training,
        )

        # 断言自注意力层的输出与输入的形状相同
        tf.debugging.assert_equal(
            shape_list(hidden_states),
            shape_list(residual),
            message=f"Self attn modified the shape of query {shape_list(residual)} to {shape_list(hidden_states)}",
        )

        # 对当前输出进行 Dropout 处理
        hidden_states = self.dropout(hidden_states, training=training)
        # 残差连接
        hidden_states = residual + hidden_states

        # 保存残差连接
        residual = hidden_states
        # 最终输出进行标准化
        hidden_states = self.final_layer_norm(hidden_states)
        # 使用激活函数处理全连接层 fc1 输出
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 全连接层输出进行 Dropout 处理
        hidden_states = self.activation_dropout(hidden_states, training=training)
        # 使用全连接层 fc2 处理数据
        hidden_states = self.fc2(hidden_states)
        # 对输出进行 Dropout 处理
        hidden_states = self.dropout(hidden_states, training=training)
        # 残差连接
        hidden_states = residual + hidden_states

        # 返回处理后的 hidden_states 和 self_attn_weights
        return hidden_states, self_attn_weights
    # 构建模型，如果已经构建过了则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在 self_attn 属性，就构建 self_attn
        if getattr(self, "self_attn", None) is not None:
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)
        # 如果存在 self_attn_layer_norm 属性，就构建 self_attn_layer_norm
        if getattr(self, "self_attn_layer_norm", None) is not None:
            with tf.name_scope(self.self_attn_layer_norm.name):
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        # 如果存在 fc1 属性，就构建 fc1
        if getattr(self, "fc1", None) is not None:
            with tf.name_scope(self.fc1.name):
                self.fc1.build([None, None, self.embed_dim])
        # 如果存在 fc2 属性，就构建 fc2
        if getattr(self, "fc2", None) is not None:
            with tf.name_scope(self.fc2.name):
                self.fc2.build([None, None, self.config.encoder_ffn_dim])
        # 如果存在 final_layer_norm 属性，就构建 final_layer_norm
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.embed_dim])
class TFSpeech2TextDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: Speech2TextConfig, **kwargs):
        super().__init__(**kwargs)  # 调用父类的构造函数
        self.embed_dim = config.d_model  # 设置嵌入维度为配置中的模型维度

        # 初始化自注意力层
        self.self_attn = TFSpeech2TextAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="self_attn",
            is_decoder=True,
        )

        # 初始化dropout层
        self.dropout = tf.keras.layers.Dropout(config.dropout)

        # 获取激活函数
        self.activation_fn = get_tf_activation(config.activation_function)

        # 初始化激活函数的dropout层
        self.activation_dropout = tf.keras.layers.Dropout(config.activation_dropout)

        # 初始化自注意力层的LayerNormalization层
        self.self_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")

        # 初始化编码器注意力层
        self.encoder_attn = TFSpeech2TextAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            name="encoder_attn",
            is_decoder=True,
        )

        # 初始化编码器注意力层的LayerNormalization层
        self.encoder_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="encoder_attn_layer_norm")

        # 初始化全连接层1
        self.fc1 = tf.keras.layers.Dense(config.decoder_ffn_dim, name="fc1")

        # 初始化全连接层2
        self.fc2 = tf.keras.layers.Dense(self.embed_dim, name="fc2")

        # 初始化最终的LayerNormalization层
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")

        # 保存配置
        self.config = config

    # 定义调用方法
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
    # 构建模型，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 设置标志为已构建
        self.built = True
        # 如果存在自注意力机制，则构建自注意力层
        if getattr(self, "self_attn", None) is not None:
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)
        # 如果存在自注意力层归一化，则构建自注意力层归一化层
        if getattr(self, "self_attn_layer_norm", None) is not None:
            with tf.name_scope(self.self_attn_layer_norm.name):
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        # 如果存在编码器注意力机制，则构建编码器注意力层
        if getattr(self, "encoder_attn", None) is not None:
            with tf.name_scope(self.encoder_attn.name):
                self.encoder_attn.build(None)
        # 如果存在编码器注意力层归一化，则构建编码器注意力层归一化层
        if getattr(self, "encoder_attn_layer_norm", None) is not None:
            with tf.name_scope(self.encoder_attn_layer_norm.name):
                self.encoder_attn_layer_norm.build([None, None, self.embed_dim])
        # 如果存在第一个全连接层，则构建第一个全连接层
        if getattr(self, "fc1", None) is not None:
            with tf.name_scope(self.fc1.name):
                self.fc1.build([None, None, self.embed_dim])
        # 如果存在第二个全连接层，则构建第二个全连接层
        if getattr(self, "fc2", None) is not None:
            with tf.name_scope(self.fc2.name):
                self.fc2.build([None, None, self.config.decoder_ffn_dim])
        # 如果存在最终归一化层，则构建最终归一化层
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.embed_dim])
class TFSpeech2TextPreTrainedModel(TFPreTrainedModel):
    # 定义一个继承自TFPreTrainedModel的类，用于演示语音到文本的预训练模型
    config_class = Speech2TextConfig
    # 指定配置类为Speech2TextConfig
    base_model_prefix = "model"
    # 设置基础模型前缀为"model"
    main_input_name = "input_features"
    # 设置主输入名称为"input_features"
    _keys_to_ignore_on_load_unexpected = [r"encoder.embed_positions.weights"]
    # 指定在加载时忽略的键名

    def _get_feat_extract_output_lengths(self, input_lengths: tf.Tensor):
        """
        Computes the output length of the convolutional layers
        """
        # 计算卷积层的输出长度
        for _ in range(self.config.num_conv_layers):
            input_lengths = (input_lengths - 1) // 2 + 1
            # 通过循环迭代计算输出长度

        return input_lengths
        # 返回计算结果

    @property
    def input_signature(self):
        # 定义输入签名方法
        return {
            "input_features": tf.TensorSpec(
                (None, None, self.config.input_feat_per_channel * self.config.input_channels),
                tf.float32,
                name="input_features",
            ),
            "attention_mask": tf.TensorSpec((None, None), tf.int32, name="attention_mask"),
            "decoder_input_ids": tf.TensorSpec((None, None), tf.int32, name="decoder_input_ids"),
            "decoder_attention_mask": tf.TensorSpec((None, None), tf.int32, name="decoder_attention_mask"),
        }
        # 返回包含输入特征和注意力掩码等信息的字典

SPEECH_TO_TEXT_START_DOCSTRING = r"""
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
"""
    Note that when creating models and layers with
    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don't need to worry
    about any of this, as you can just pass inputs like you would to any other Python function!

    </Tip>
    # 注意：当使用子类化创建模型和层时，您无需担心这些，因为您可以像对待任何其他Python函数一样传递输入！

    Parameters:
        config ([`Speech2TextConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
    # 参数:
    #     config ([`Speech2TextConfig`]):
    #         包含模型所有参数的模型配置类。使用配置文件初始化不会加载与模型关联的权重，只会加载配置。查看
    #         [`~TFPreTrainedModel.from_pretrained`] 方法以加载模型权重。
# 定义一个文档字符串，用于指定模块Speech to Text输入的说明
SPEECH_TO_TEXT_INPUTS_DOCSTRING = r"""
"""

# 定义一个基于Transformer的编码器层，由多个自注意力层组成
# 每一层都是一个TFSpeech2TextEncoderLayer
class TFSpeech2TextEncoder(tf.keras.layers.Layer):
    # 配置类为Speech2TextConfig
    config_class = Speech2TextConfig
    """
    Transformer编码器，由config.encoder_layers个自注意力层组成，每层是一个TFSpeech2TextEncoderLayer

    Args:
        config: Speech2TextConfig
    """

    def __init__(self, config: Speech2TextConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = tf.math.sqrt(float(embed_dim)) if config.scale_embedding else 1.0

        # 基于config配置创建TFConv1dSubsampler层，用于卷积操作
        self.conv = TFConv1dSubsampler(config, name="conv")

        # 创建位置编码层，用于将序列位置信息嵌入到特征中
        self.embed_positions = TFSpeech2TextSinusoidalPositionalEmbedding(
            num_positions=config.max_source_positions,
            embedding_dim=embed_dim,
            padding_idx=self.padding_idx,
            name="embed_positions",
        )
        # 创建多个TFSpeech2TextEncoderLayer层，作为Transformer编码器的主要层
        self.layers = [TFSpeech2TextEncoderLayer(config, name=f"layers.{i}") for i in range(config.encoder_layers)]
        # 创建LayerNormalization层，用于归一化每个Transformer编码器层的输出
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layer_norm")

    def _get_feat_extract_output_lengths(self, input_lengths: tf.Tensor):
        """
        计算卷积层的输出长度
        """
        for _ in range(self.config.num_conv_layers):
            input_lengths = (input_lengths - 1) // 2 + 1

        return input_lengths

    def _get_feature_vector_attention_mask(self, feature_vector_length, attention_mask):
        # 生成一个3D的注意力掩码，用于输入特征向量
        # 如果情况特殊，将其转换为2D
        if len(attention_mask.shape) > 2:
            attention_mask = attention_mask[:, :, -1]

        # 计算特征提取输出长度
        subsampled_lengths = self._get_feat_extract_output_lengths(tf.math.reduce_sum(attention_mask, -1))
        bsz = shape_list(attention_mask)[0]
        # 构建索引，用于生成2D的注意力掩码
        indices = tf.concat(
            (
                tf.expand_dims(tf.range(bsz, dtype=attention_mask.dtype), -1),
                tf.expand_dims(subsampled_lengths - 1, -1),
            ),
            axis=-1,
        )
        attention_mask = tf.scatter_nd(indices=indices, # scatter_nd用指定的索引位置更新给定的数值，生成新的张量
                                       updates=tf.ones(bsz), 
                                       shape=[bsz, feature_vector_length])
        # 反转并累加注意力掩码
        attention_mask = tf.cast(tf.reverse(tf.math.cumsum(tf.reverse(attention_mask, [-1]), -1), [-1]), tf.int64)
        return attention_mask

    # 定义call方法，用于前向传播
    @unpack_inputs
    def call(
        self,
        input_features=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
    # 构建神经网络模型
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 设置标志为已构建
        self.built = True
        # 如果存在卷积层
        if getattr(self, "conv", None) is not None:
            # 在 tensorflow 中设置作用域名称为卷积层的名称
            with tf.name_scope(self.conv.name):
                # 构建卷积层
                self.conv.build(None)
        # 如果存在位置嵌入
        if getattr(self, "embed_positions", None) is not None:
            # 在 tensorflow 中设置作用域名称为位置嵌入的名称
            with tf.name_scope(self.embed_positions.name):
                # 构建位置嵌入
                self.embed_positions.build(None)
        # 如果存在层归一化
        if getattr(self, "layer_norm", None) is not None:
            # 在 tensorflow 中设置作用域名称为层归一化的名称
            with tf.name_scope(self.layer_norm.name):
                # 构建层归一化，要求输入形状为 [None, None, self.config.d_model]
                self.layer_norm.build([None, None, self.config.d_model])
        # 如果存在多个层
        if getattr(self, "layers", None) is not None:
            # 对每一层进行构建
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    # 在 tensorflow 中设置作用域名称为层的名称
                    layer.build(None)
# 使用keras_serializable装饰器将TFSpeech2TextDecoder类标记为序列化对象
@keras_serializable
class TFSpeech2TextDecoder(tf.keras.layers.Layer):
    # 将config_class属性设置为Speech2TextConfig类
    config_class = Speech2TextConfig
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`TFSpeech2TextDecoderLayer`]
    描述Transformer解码器由config.decoder_layers层组成，每一层都是一个TFSpeech2TextDecoderLayer

    Args:
        config: Speech2TextConfig
        参数：config是Speech2TextConfig的实例
    """

    # 初始化方法
    def __init__(self, config: Speech2TextConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_target_positions
        self.embed_scale = tf.math.sqrt(float(config.d_model)) if config.scale_embedding else 1.0

        # 创建TFSharedEmbeddings实例，并赋值给embed_tokens属性
        self.embed_tokens = TFSharedEmbeddings(config.vocab_size, config.d_model, name="embed_tokens")

        # 创建TFSpeech2TextSinusoidalPositionalEmbedding实例，并赋值给embed_positions属性
        self.embed_positions = TFSpeech2TextSinusoidalPositionalEmbedding(
            num_positions=config.max_target_positions,
            embedding_dim=config.d_model,
            padding_idx=self.padding_idx,
            name="embed_positions",
        )

        # 创建多个TFSpeech2TextDecoderLayer实例，并赋值给layers属性
        self.layers = [TFSpeech2TextDecoderLayer(config, name=f"layers.{i}") for i in range(config.decoder_layers)]
        # 创建LayerNormalization层，并赋值给layer_norm属性
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layer_norm")

        # 创建Dropout层，并赋值给dropout属性
        self.dropout = tf.keras.layers.Dropout(config.dropout)

    # 获取embed_tokens属性的方法
    def get_embed_tokens(self):
        return self.embed_tokens

    # 设置embed_tokens属性的方法
    def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens

    # unpack_inputs装饰器修饰call方法
    @unpack_inputs
    def call(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
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
    # build方法
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果embed_tokens存在，则调用其build方法
        if getattr(self, "embed_tokens", None) is not None:
            with tf.name_scope(self.embed_tokens.name):
                self.embed_tokens.build(None)
        # 如果embed_positions存在，则调用其build方法
        if getattr(self, "embed_positions", None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)
        # 如果layer_norm存在，则调用其build方法
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.d_model])
        # 如果layers存在，则遍历其中的每一层，并调用其build方法
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)


@keras_serializable
class TFSpeech2TextMainLayer(tf.keras.layers.Layer):
    config_class = Speech2TextConfig
    # 初始化函数，接受配置参数并调用父类的初始化方法，保存配置信息
    def __init__(self, config: Speech2TextConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
    
        # 初始化编码器和解码器对象
        self.encoder = TFSpeech2TextEncoder(config, name="encoder")
        self.decoder = TFSpeech2TextDecoder(config, name="decoder")
    
    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.decoder.embed_tokens
    
    # 设置输入嵌入
    def set_input_embeddings(self, new_embeddings):
        self.decoder.embed_tokens = new_embeddings
    
    # call方法，接受多个输入参数，使用@unpack_inputs注解解包输入
    @unpack_inputs
    def call(
        self,
        input_features=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
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
        **kwargs,
          
    # 构建方法，接受输入形状作为参数，如果已构建则直接返回，否则构建编码器和解码器对象
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建编码器对象
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 构建解码器对象
        if getattr(self, "decoder", None) is not None:
            with tf.name_scope(self.decoder.name):
                self.decoder.build(None)
# 使用装饰器添加文档字符串，描述模型功能以及输入输出
@add_start_docstrings(
    "The bare Speech2Text Model outputting raw hidden-states without any specific head on top.",
    SPEECH_TO_TEXT_START_DOCSTRING,
)
# 定义 TFSpeech2TextModel 类，继承自 TFSpeech2TextPreTrainedModel 类
class TFSpeech2TextModel(TFSpeech2TextPreTrainedModel):
    # 初始化方法，接受配置对象 config 和其他参数
    def __init__(self, config: Speech2TextConfig, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 创建 TFSpeech2TextMainLayer 对象，传入配置对象和名称参数
        self.model = TFSpeech2TextMainLayer(config, name="model")

    # 获取编码器的方法
    def get_encoder(self):
        # 返回模型的编码器
        return self.model.encoder

    # 获取解码器的方法
    def get_decoder(self):
        # 返回模型的解码器
        return self.model.decoder

    # 定义 call 方法，接受多个输入参数，并返回模型输出
    @unpack_inputs
    @add_start_docstrings_to_model_forward(SPEECH_TO_TEXT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSeq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_features: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        decoder_input_ids: np.ndarray | tf.Tensor | None = None,
        decoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        decoder_head_mask: np.ndarray | tf.Tensor | None = None,
        cross_attn_head_mask: np.ndarray | tf.Tensor | None = None,
        encoder_outputs: np.ndarray | tf.Tensor | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        decoder_inputs_embeds: np.ndarray | tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        **kwargs,
    ) -> Union[Tuple, TFSeq2SeqModelOutput]:
        # 调用模型的前向传播方法，传入各种输入参数
        outputs = self.model(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
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
    # 定义一个方法用于返回模型输出
    def serving_output(self, output):
        # 如果配置中使用缓存，则从output的过去key value中获取pkv，否则为None
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        # 如果配置中输出隐藏状态，则将output的decoder hidden states转换为张量，否则为None
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置中输出注意力分布，则将output的decoder attentions转换为张量，否则为None
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        # 如果配置中输出注意力分布，则将output的cross attentions转换为张量，否则为None
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        # 如果配置中输出隐藏状态，则将output的encoder hidden states转换为张量，否则为None
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置中输出注意力分布，则将output的encoder attentions转换为张量，否则为None
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None
        
        # 返回转换后的TFSeq2SeqModelOutput对象
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
    
    # 定义一个方法用于构建模型，如果已经构建过，则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 检查self中是否存在model属性
        if getattr(self, "model", None) is not None:
            # 在命名空间中构建模型
            with tf.name_scope(self.model.name):
                self.model.build(None)
# 使用装饰器添加模型介绍到文档字符串起始部分，说明该模型可以用于摘要生成
@add_start_docstrings(
    "The Speech2Text Model with a language modeling head. Can be used for summarization.",
    SPEECH_TO_TEXT_START_DOCSTRING,
)
# 声明一个类 TFSpeech2TextForConditionalGeneration，继承自 TFSpeech2TextPreTrainedModel 和 TFCausalLanguageModelingLoss
class TFSpeech2TextForConditionalGeneration(TFSpeech2TextPreTrainedModel, TFCausalLanguageModelingLoss):
    # 初始化方法，接收一个 Speech2TextConfig 类型的 config 参数
    def __init__(self, config: Speech2TextConfig):
        # 调用父类初始化方法
        super().__init__(config)
        # 创建 TFSpeech2TextMainLayer 对象，命名为 model
        self.model = TFSpeech2TextMainLayer(config, name="model")
        # 创建一个 Dense 层用于语言模型头，输出大小为 config.vocab_size，不使用偏置，命名为 lm_head
        self.lm_head = tf.keras.layers.Dense(self.config.vocab_size, use_bias=False, name="lm_head")
        # TODO (Joao): 常量输出调查为什么 Speech2Text 在 XLA 生成中存在数值问题
        # 设置是否支持 XLA 生成为 False
        self.supports_xla_generation = False
        # 保存传入的 config 参数
        self.config = config

    # 获取编码器
    def get_encoder(self):
        return self.model.encoder

    # 获取解码器
    def get_decoder(self):
        return self.model.decoder

    # 调整 token embeddings 的大小
    def resize_token_embeddings(self, new_num_tokens: int) -> tf.Variable:
        # 调用父类的 resize_token_embeddings 方法，返回新的 embeddings
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        return new_embeddings

    # 获取输出 embeddings
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出 embeddings
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 实现 call 方法，接收多个输入参数和标志位
    @unpack_inputs
    @add_start_docstrings_to_model_forward(SPEECH_TO_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_features: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        decoder_input_ids: np.ndarray | tf.Tensor | None = None,
        decoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        decoder_head_mask: np.ndarray | tf.Tensor | None = None,
        cross_attn_head_mask: np.ndarray | tf.Tensor | None = None,
        encoder_outputs: np.ndarray | tf.Tensor | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        decoder_inputs_embeds: np.ndarray | tf.Tensor | None = None,
        labels: np.ndarray | tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
        **kwargs,
    # 定义用于输出的方法，接受输出对象作为参数
    def serving_output(self, output):
        # 如果配置中使用了缓存，则从输出对象的过去键值中获取键值，否则为 None
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        # 如果配置中输出隐藏状态，则将输出对象的解码器隐藏状态转换为张量，否则为 None
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置中输出注意力权重，则将输出对象的解码器注意力权重转换为张量，否则为 None
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        # 如果配置中输出注意力权重，则将输出对象的交叉注意力权重转换为张量，否则为 None
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        # 如果配置中输出隐藏状态，则将输出对象的编码器隐藏状态转换为张量，否则为 None
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        # 如果配置中输出注意力权重，则将输出对象的编码器注意力权重转换为张量，否则为 None
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None

        # 返回 TFSeq2SeqLMOutput 对象，包含输出对象的不同属性
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

    # 准备用于生成的输入，接受解码器输入 ID、过去键值等参数
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # 如果存在过去键值，则截取解码器输入 ID 的最后一个 token
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # 返回包含不同输入参数的字典
        return {
            "input_features": None,  # 需要传递以使 Keras.layer.__call__ 快乐
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # 修改此处以避免缓存（可能用于调试）
        }

    # 构建模型，接受输入形状作为参数
    def build(self, input_shape=None):
        # 如果已经构建过了，则直接返回
        if self.built:
            return
        # 设置为已构建
        self.built = True
        # 如果存在模型，则构建模型
        if getattr(self, "model", None) is not None:
            with tf.name_scope(self.model.name):
                self.model.build(None)
        # 如果存在语言模型头部，则构建语言模型头部
        if getattr(self, "lm_head", None) is not None:
            with tf.name_scope(self.lm_head.name):
                self.lm_head.build([None, None, self.config.d_model])

    # 将 TensorFlow 权重重命名为 PyTorch 权重
    def tf_to_pt_weight_rename(self, tf_weight):
        # 如果是语言模型头部权重，则将其重命名为模型的嵌入令牌权重
        if tf_weight == "lm_head.weight":
            return tf_weight, "model.decoder.embed_tokens.weight"
        # 否则返回原始权重
        else:
            return (tf_weight,)
```