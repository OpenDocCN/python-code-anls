# `.\models\esm\modeling_tf_esm.py`

```py
# 设置编码为 UTF-8
# 版权声明，该代码由 Meta 和 HuggingFace Inc. 团队拥有
# 根据 Apache 许可证 2.0 版本进行许可
# 除非符合许可证的规定，否则您不得使用此文件
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发本软件
# 无任何担保或条件，明示或暗示
# 有关软件的详细信息，请参见许可证
""" PyTorch ESM model."""


# 导入必要的库和模块
from __future__ import annotations
import os
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from tensorflow.keras.activations import gelu
from tensorflow.keras.layers import Dense, Dropout, Embedding, Layer, LayerNormalization

# 导入相关函数和类
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFBaseModelOutputWithPoolingAndCrossAttentions,
    TFMaskedLMOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (
    TFMaskedLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    get_initializer,
    shape_list,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, stable_softmax
from ...utils import logging
from .configuration_esm import EsmConfig

# 获取 logger
logger = logging.get_logger(__name__)

# 用于文档的模型检查点和配置信息
_CHECKPOINT_FOR_DOC = "facebook/esm2_t6_8M_UR50D"
_CONFIG_FOR_DOC = "EsmConfig"

# 预训练模型的存档列表
TF_ESM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/esm2_t6_8M_UR50D",
    "facebook/esm2_t12_35M_UR50D",
    # This is not a complete list of all ESM models!
    # See all ESM models at https://huggingface.co/models?filter=esm
]


# 旋转张量的一半
def rotate_half(x):
    x1, x2 = tf.split(x, 2, axis=-1)
    return tf.concat((-x2, x1), axis=-1)


# 应用旋转位置嵌入
def apply_rotary_pos_emb(x, cos, sin):
    cos = cos[:, :, : tf.shape(x)[-2], :]
    sin = sin[:, :, : tf.shape(x)[-2], :]
    return (x * cos) + (rotate_half(x) * sin)


# 使层对称于最后两个维度，用于接触预测
def symmetrize(x):
    return x + tf.linalg.matrix_transpose(x)  # Transposes last two dimensions only


# 执行平均乘积校正，用于接触预测
def average_product_correct(x):
    a1 = tf.reduce_sum(x, -1, keepdims=True)
    a2 = tf.reduce_sum(x, -2, keepdims=True)
    a12 = tf.reduce_sum(x, (-1, -2), keepdims=True)
    avg = a1 * a2
    avg = avg / a12
    normalized = x - avg
    return normalized


# 旋转嵌入层
class TFRotaryEmbedding(Layer):
    """
    Rotary position embeddings based on those in
    [RoFormer](https://huggingface.co/docs/transformers/model_doc/roformer). Query and keys are transformed by rotation
    matrices which depend on their relative positions.
    """

    # 初始化方法
    def __init__(self, dim: int, name=None):
        # 调用父类的初始化方法
        super().__init__(name=name)
        # 设置维度
        self.dim = dim

    # 构建方法
    def build(self, input_shape):
        # 调用父类的构建方法
        super().build(input_shape)
        # 添加权重 inv_freq 并初始化
        self.inv_freq = self.add_weight(
            "inv_freq", shape=(self.dim // 2,), dtype=tf.float32, initializer=get_initializer(1.0), trainable=False
        )
        # 计算并赋值 inv_freq 的值
        self.inv_freq.assign(
            1.0 / (10000 ** (tf.range(start=0, limit=self.dim, delta=2, dtype=tf.float32) / self.dim))
        )

    # 计算余弦和正弦值
    def _compute_cos_sin(self, x, seq_dimension=2):
        # 获取序列长度
        seq_len = tf.shape(x)[seq_dimension]

        # 生成频率
        t = tf.range(seq_len, dtype=self.inv_freq.dtype)
        freqs = tf.einsum("i, j -> ij", t, self.inv_freq)  # Outer multiplication
        emb = tf.concat((freqs, freqs), axis=-1)[None, None, :, :]

        return tf.cos(emb), tf.sin(emb)

    # 调用方法
    def call(self, q: tf.Tensor, k: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # 计算余弦和正弦值
        cos_emb, sin_emb = self._compute_cos_sin(k, seq_dimension=-2)

        # 应用旋转位置嵌入
        return (
            apply_rotary_pos_emb(q, cos_emb, sin_emb),
            apply_rotary_pos_emb(k, cos_emb, sin_emb),
        )
class TFEsmContactPredictionHead(Layer):
    """Performs symmetrization, apc, and computes a logistic regression on the output features"""

    def __init__(
        self,
        in_features: int,
        bias=True,
        eos_idx: int = 2,
        name=None,
    ):
        # 初始化函数，设置类的参数和属性
        super().__init__(name=name)
        self.eos_idx = eos_idx  # 设置 EOS（End of Sequence）的索引
        self.in_features = in_features  # 输入特征的数量
        # 创建用于逻辑回归的全连接层
        self.regression = Dense(1, use_bias=bias, activation="sigmoid", name="regression")

    def build(self, input_shape=None):
        # 构建模型层，用于在运行时构建层的权重
        if self.built:
            return
        self.built = True
        if getattr(self, "regression", None) is not None:
            with tf.name_scope(self.regression.name):
                # 构建逻辑回归层
                self.regression.build((None, self.in_features))

    def call(self, tokens, attentions):
        # 对输入的注意力进行处理
        # 移除 EOS 令牌的注意力
        eos_mask = tf.cast(tokens != self.eos_idx, attentions.dtype)
        eos_mask = tf.expand_dims(eos_mask, 1) * tf.expand_dims(eos_mask, 2)
        attentions = attentions * eos_mask[:, None, None, :, :]
        attentions = attentions[..., :-1, :-1]
        # 移除 CLS 令牌的注意力
        attentions = attentions[..., 1:, 1:]
        batch_size, layers, heads, seqlen, _ = shape_list(attentions)
        # 重新调整注意力矩阵的形状
        attentions = tf.reshape(attentions, (batch_size, layers * heads, seqlen, seqlen))

        # 对称化并修正平均乘积注意力
        attentions = average_product_correct(symmetrize(attentions))
        attentions = tf.transpose(attentions, perm=(0, 2, 3, 1))
        # 使用逻辑回归层对注意力进行回归预测
        return tf.squeeze(self.regression(attentions), 3)


class TFEsmEmbeddings(Layer):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config, name=None):
        # 初始化函数，设置类的参数和属性
        super().__init__(name=name)
        # 初始化词嵌入层和位置嵌入层
        self.word_embeddings = Embedding(
            config.vocab_size,
            config.hidden_size,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="word_embeddings",
        )
        self.position_embeddings = Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="position_embeddings",
        )

        if config.emb_layer_norm_before:
            # 是否在嵌入层之前使用层归一化
            self.layer_norm = LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        else:
            self.layer_norm = None
        # 确定位置编码类型
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        self.position_ids = tf.range(config.max_position_embeddings)[None, :]
        # 填充标记的索引
        self.padding_idx = config.pad_token_id
        self.token_dropout = config.token_dropout  # 令牌丢弃概率
        self.mask_token_id = config.mask_token_id  # 掩码标记的索引
        self.config = config  # 嵌入层的配置信息
    def call(
        self, input_ids=None, attention_mask=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        # 如果位置 ids 未提供
        if position_ids is None:
            # 如果输入 ids 已提供
            if input_ids is not None:
                # 从输入的标记 ids 中创建位置 ids，未填充的标记保持填充状态
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                # 从输入的嵌入向量中创建位置 ids
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        # 如果未提供输入的嵌入向量
        if inputs_embeds is None:
            # 检查输入的标记 ids 是否在词汇表大小范围内
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            # 使用输入的标记 ids 获取嵌入向量
            inputs_embeds = self.word_embeddings(input_ids)

        # 嵌入向量即为预训练模型输入
        embeddings = inputs_embeds

        # Matt: ESM 可以以略微不同的方式处理 MLM 中的遮罩。如果 token_dropout 标志为 False，则处理方式与 BERT/RoBERTa 相同。
        # 如果设置为 True，则将遮罩标记的标记视为选择输入的标记并将其归零。当遮罩标记不存在时，通过使嵌入向量乘以一个因子来补偿
        # （训练中的未遮罩标记的比例）/（样本中未遮罩标记的比例）。这类似于在评估时放大输出的方式，当实际上没有去除任何值时，
        # 丢弃层会减小其传递的输出，或者在训练中扩大它们的未丢弃输出（或等效地，在评估时放大输出，当实际上没有丢弃任何值）。
        if self.token_dropout:
            # 将遮罩标记的标记所在位置的嵌入向量值设置为 0.0
            embeddings = tf.where((input_ids == self.mask_token_id)[:, :, None], 0.0, embeddings)
            # 计算在训练中未遮罩的标记比例
            mask_ratio_train = 0.15 * 0.8  # 在所有 ESM 模型的训练运行中都是硬编码的比例
            # 计算样本中观察到的未遮罩标记比例
            src_lengths = tf.cast(tf.reduce_sum(attention_mask, axis=-1), tf.float32)
            masked_tokens = input_ids == self.mask_token_id
            mask_ratio_observed = tf.math.count_nonzero(masked_tokens, dtype=tf.float32, axis=-1) / src_lengths
            # 对嵌入向量进行调整，使其乘以 (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]
            embeddings = embeddings * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        # 如果位置嵌入类型为 "absolute"
        if self.position_embedding_type == "absolute":
            # 获取位置嵌入向��
            position_embeddings = self.position_embeddings(position_ids)
            # 将位置嵌入向量与嵌入向量相加
            embeddings += position_embeddings

        # 如果层标准化存在
        if self.layer_norm is not None:
            # 对嵌入向量进行层标准化
            embeddings = self.layer_norm(embeddings)
        # 如果注意力遮罩存在
        if attention_mask is not None:
            # 将嵌入向量乘以注意力遮罩
            embeddings = embeddings * tf.cast(tf.expand_dims(attention_mask, -1), embeddings.dtype)
        # Matt: 我认为这行代码从 BERT 错误地复制过来了，暂时禁用它。
        # embeddings = self.dropout(embeddings)
        # 返回嵌入向量
        return embeddings
    # 创建位置编码，从输入的嵌入向量中生成序列位置标识符
    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: tf.Tensor

        Returns: tf.Tensor
        """
        # 获取输入嵌入向量的形状
        input_shape = shape_list(inputs_embeds)[:-1]
        # 获取序列长度
        sequence_length = input_shape[1]

        # 生成序列位置标识符，从padding_idx开始，到序列长度加上padding_idx
        position_ids = tf.range(
            start=self.padding_idx + 1, limit=sequence_length + self.padding_idx + 1, dtype=tf.int64
        )
        # 将位置标识符广播到输入形状
        return tf.broadcast_to(tf.expand_dims(position_ids, 0), input_shape)

    # 构建位置编码层
    def build(self, input_shape=None):
        # 如果已经构建，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在词嵌入，则构建词嵌入
        if getattr(self, "word_embeddings", None) is not None:
            with tf.name_scope(self.word_embeddings.name):
                self.word_embeddings.build(None)
        # 如果存在位置嵌入，则构建位置嵌入
        if getattr(self, "position_embeddings", None) is not None:
            with tf.name_scope(self.position_embeddings.name):
                self.position_embeddings.build(None)
        # 如果存在层归一化，则构建层归一化
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.hidden_size])
class TFEsmSelfAttention(Layer):
    def __init__(self, config, position_embedding_type=None, name=None):
        # 初始化自注意力层对象，设置属性
        super().__init__(name=name)
        # 如果隐藏层大小不是注意力头数的倍数且没有嵌入大小属性，则引发异常
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 创建查询、键、值网络层
        self.query = Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        self.key = Dense(self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key")
        self.value = Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )

        # 创建dropout层
        self.dropout = Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        
        # 如果采用相对位置编码或旋转编码，则初始化相关属性
        self.rotary_embeddings = None
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = Embedding(
                2 * config.max_position_embeddings - 1,
                self.attention_head_size,
                embeddings_initializer=get_initializer(config.initializer_range),
            )
        elif self.position_embedding_type == "rotary":
            self.rotary_embeddings = TFRotaryEmbedding(dim=self.attention_head_size, name="rotary_embeddings")

        # 判断是否是解码器
        self.is_decoder = config.is_decoder
        self.config = config

    def transpose_for_scores(self, x: tf.Tensor) -> tf.Tensor:
        # 将张量重新形状为[batch_size, num_heads, seq_length, head_size]并转置
        new_x_shape = shape_list(x)[:-1] + [self.num_attention_heads, self.attention_head_size]
        x = tf.reshape(x, new_x_shape)
        return tf.transpose(x, perm=(0, 2, 1, 3))

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        encoder_hidden_states: tf.Tensor | None = None,
        encoder_attention_mask: tf.Tensor | None = None,
        past_key_value: Tuple[Tuple[tf.Tensor]] | None = None,
        output_attentions: Optional[bool] = False,
        training: bool = False,
    # 构建神经网络模型
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        # 标记模型为已构建状态
        self.built = True
        # 如果存在查询(query)属性，则使用 TensorFlow 的命名空间为其构建
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.hidden_size])
        # 如果存在键(key)属性，则使用 TensorFlow 的命名空间为其构建
        if getattr(self, "key", None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.config.hidden_size])
        # 如果存在值(value)属性，则使用 TensorFlow 的命名空间为其构建
        if getattr(self, "value", None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.config.hidden_size])
        # 如果存在旋转嵌入(rotary_embeddings)属性，则使用 TensorFlow 的命名空间为其构建
        if getattr(self, "rotary_embeddings", None) is not None:
            with tf.name_scope(self.rotary_embeddings.name):
                self.rotary_embeddings.build(None)
# 创建名为 TFEsmSelfOutput 的自定义层，继承自 Layer
class TFEsmSelfOutput(Layer):
    # 初始化函数
    def __init__(self, config, name=None):
        # 调用父类的初始化函数
        super().__init__(name=name)
        # 创建一个全连接层 Dense，用于变换隐藏状态的维度
        self.dense = Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个 Dropout 层，用于在训练时随机断开一定比例的神经元连接，防止过拟合
        self.dropout = Dropout(config.hidden_dropout_prob)
        # 保存模型配置
        self.config = config

    # 前向传播函数
    def call(self, hidden_states, input_tensor, training=False):
        # 对隐藏状态进行全连接变换
        hidden_states = self.dense(hidden_states)
        # 在训练时对变换后的隐藏状态进行随机断开
        hidden_states = self.dropout(hidden_states, training=training)
        # 将变换后的隐藏状态和输入张量相加
        hidden_states += input_tensor
        # 返回相加后的结果
        return hidden_states

    # 构建层的函数
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 构建全连接层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# 创建名为 TFEsmAttention 的自定义层，继承自 Layer
class TFEsmAttention(Layer):
    # 初始化函数
    def __init__(self, config, name=None):
        # 调用父类的初始化函数
        super().__init__(name=name)
        # 创建 self 层，实例化 TFEsmSelfAttention 层
        self.self = TFEsmSelfAttention(config, name="self")
        # 创建 output_layer 层，实例化 TFEsmSelfOutput 层
        self.output_layer = TFEsmSelfOutput(config, name="output")
        # 初始化剪枝头信息的集合
        self.pruned_heads = set()
        # 创建 LayerNorm 层，用于进行层归一化操作
        self.LayerNorm = LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 保存模型配置
        self.config = config

    # 剪枝头信息的函数
    def prune_heads(self, heads):
        # 抛出未实现的异常
        raise NotImplementedError

    # 前向传播函数
    def call(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        training=False,
    ):
        # 对隐藏状态进行层归一化
        hidden_states_ln = self.LayerNorm(hidden_states)
        # 通过 self 层处理隐藏状态
        self_outputs = self.self(
            hidden_states_ln,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            training,
        )
        # 通过 output_layer 层处理 self 层输出
        attention_output = self.output_layer(self_outputs[0], hidden_states)
        # 输出包括 attention_output 和 self_outputs 的其余部分
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        # 返回输出结果
        return outputs

    # 构建层的函数
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 构建 self 层
        if getattr(self, "self", None) is not None:
            with tf.name_scope(self.self.name):
                self.self.build(None)
        # 构建 output_layer 层
        if getattr(self, "output_layer", None) is not None:
            with tf.name_scope(self.output_layer.name):
                self.output_layer.build(None)
        # 构建 LayerNorm 层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
    # 初始化函数，接受一个 EsmConfig 对象和其他关键字参数
    def __init__(self, config: EsmConfig, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 创建一个全连接层，设置单元数为 config.intermediate_size，使用 config.initializer_range 来初始化权重，命名为"dense"
        self.dense = tf.keras.layers.Dense(
            units=config.intermediate_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )
        # 保存配置对象
        self.config = config

    # 调用函数，接受一个 tf.Tensor 类型的 hidden_states 参数，返回一个 tf.Tensor 类型的结果
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 将 hidden_states 输入到全连接层中得到输出
        hidden_states = self.dense(inputs=hidden_states)
        # 使用 gelu 激活函数处理输出
        hidden_states = tf.nn.gelu(hidden_states)
        # 返回处理后的结果
        return hidden_states

    # 构建函数，用于构建层的变量，当已经构建过时直接返回，否则构建并设置构建标志为 True
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在全连接层，则使用其名字创建命名空间，并构建全连接层的权重变量
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
# 定义 TF 模型类 TFEsmOutput，继承自 Layer 类
class TFEsmOutput(Layer):
    # 初始化方法，接受 config 和 name 两个参数
    def __init__(self, config, name=None):
        # 调用父类的初始化方法
        super().__init__(name=name)
        # 创建一个全连接层，输出维度为 config.hidden_size，初始化方法为 config.initializer_range，名称为 "dense"
        self.dense = Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个 Dropout 层，dropout rate 为 config.hidden_dropout_prob
        self.dropout = Dropout(config.hidden_dropout_prob)
        # 将 config 参数保存到实例中
        self.config = config

    # 定义 call 方法，接受 hidden_states, input_tensor 和 training 三个参数
    def call(self, hidden_states, input_tensor, training=False):
        # 将 hidden_states 输入到全连接层中
        hidden_states = self.dense(hidden_states)
        # 在全连接层的输出上应用 dropout，如果 training 为 True
        hidden_states = self.dropout(hidden_states, training=training)
        # 将输入的 input_tensor 加到 hidden_states 上
        hidden_states += input_tensor
        # 返回计算结果
        return hidden_states

    # 定义 build 方法，接受 input_shape 参数
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 将 built 属性设置为 True
        self.built = True
        # 如果实例中存在 dense 层
        if getattr(self, "dense", None) is not None:
            # 使用 dense 层的名称创建一个命名空间
            with tf.name_scope(self.dense.name):
                # 构建 dense 层，输入形状为 [None, None, self.config.intermediate_size]
                self.dense.build([None, None, self.config.intermediate_size])

# 定义 TF 模型类 TFEsmLayer，继承自 Layer 类
class TFEsmLayer(Layer):
    # 初始化方法，接受 config 和 name 两个参数
    def __init__(self, config, name=None):
        # 调用父类的初始化方法
        super().__init__(name=name)
        # 保存一些参数到实例中
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        # 创建一个 TFEsmAttention 实例，名称为 "attention"
        self.attention = TFEsmAttention(config, name="attention")
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        # 如果 add_cross_attention 为 True 且不是 decoder 模型，则抛出异常
        if self.add_cross_attention:
            if not self.is_decoder:
                raise RuntimeError(f"{self} should be used as a decoder model if cross attention is added")
            # 创建一个 TFEsmAttention 实例
            self.crossattention = TFEsmAttention(config)
        # 创建一个 TFEsmIntermediate 实例，名称为 "intermediate"
        self.intermediate = TFEsmIntermediate(config, name="intermediate")
        # 创建一个 TFEsmOutput 实例，名称为 "output"
        self.output_layer = TFEsmOutput(config, name="output")
        # 创建一个 LayerNormalization 层，epsilon 为 config.layer_norm_eps，名称为 "LayerNorm"
        self.LayerNorm = LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 将 config 参数保存到实例中
        self.config = config

    # 定义 call 方法，接受一系列参数
    def call(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        training=False,
    # 如果过去的键/值不为None，则decoder uni方向的self-attention缓存键/值元组位于位置1,2
    self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
    # 进行self-attention计算
    self_attention_outputs = self.attention(
        hidden_states,
        attention_mask,
        head_mask,
        output_attentions=output_attentions,
        past_key_value=self_attn_past_key_value,
        training=training,
    )
    attention_output = self_attention_outputs[0]
    
    # 如果是decoder模型，最后一个输出为self-attn缓存的元组
    if self.is_decoder:
        outputs = self_attention_outputs[1:-1]
        present_key_value = self_attention_outputs[-1]
    else:
        outputs = self_attention_outputs[1:]  # 如果我们输出注意力权重，添加self注意力
    
    cross_attn_present_key_value = None
    # 如果是decoder模型并且传入了encoder隐藏状态
    if self.is_decoder and encoder_hidden_states is not None:
        if not hasattr(self, "crossattention"):
            raise AttributeError(
                f"If `encoder_hidden_states` are passed, {self} has to be instantiated"
                " with cross-attention layers by setting `config.add_cross_attention=True`"
            )
    
        # cross_attn缓存的键/值元组位于过去键/值元组的位置3,4
        cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
        # 进行cross-attention计算
        cross_attention_outputs = self.crossattention(
            attention_output,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            cross_attn_past_key_value,
            output_attentions,
            training=training,
        )
        attention_output = cross_attention_outputs[0]
        outputs = outputs + cross_attention_outputs[1:-1]  # 如果我们输出注意力权重，添加cross-attention
    
        # 在present_key_value元组的位置3,4上添加cross-attn缓存
        cross_attn_present_key_value = cross_attention_outputs[-1]
        present_key_value = present_key_value + cross_attn_present_key_value
    
    layernorm_output = self.LayerNorm(attention_output)
    intermediate_output = self.intermediate(hidden_states=layernorm_output)
    layer_output = self.output_layer(
        hidden_states=intermediate_output, input_tensor=attention_output, training=training
    )
    outputs = (layer_output,) + outputs  # 如果我们输出的话，添加注意力
    
    # 如果是decoder，返回注意力键/值作为最后的输出
    if self.is_decoder:
        outputs = outputs + (present_key_value,)
    
    return outputs
    # 构建神经网络模型，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 设置标记为已构建
        self.built = True
        # 如果定义了注意力机制，则构建注意力层
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        # 如果定义了中间层，则构建中间层
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        # 如果定义了输出层，则构建输出层
        if getattr(self, "output_layer", None) is not None:
            with tf.name_scope(self.output_layer.name):
                self.output_layer.build(None)
        # 如果定义了层归一化，则构建层归一化
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
# 创建一个名为 TFEsmEncoder 的类，继承自 Layer 类
class TFEsmEncoder(Layer):
    # 初始化方法，接受 config 和 name 两个参数
    def __init__(self, config, name=None):
        # 调用父类的初始化方法
        super().__init__(name=name)
        # 将参数 config 赋值给实例变量 self.config
        self.config = config
        # 创建一个由 TFEsmLayer 对象组成的列表，列表长度为 config.num_hidden_layers
        self.layer = [TFEsmLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]
        # 创建一个 LayerNormalization 层的实例，命名为 "emb_layer_norm_after"
        self.emb_layer_norm_after = LayerNormalization(epsilon=config.layer_norm_eps, name="emb_layer_norm_after")

    # call 方法，接受多个参数
    def call(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        training=False,
    ):
        # 根据 output_hidden_states 的值确定是否创建 all_hidden_states 变量
        all_hidden_states = () if output_hidden_states else None
        # 根据 output_attentions 的值确定是否创建 all_self_attentions 变量
        all_self_attentions = () if output_attentions else None
        # 根据 output_attentions 的值和 self.config.add_cross_attention 的值确定是否创建 all_cross_attentions 变量
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        # 根据 use_cache 的值确定是否创建 next_decoder_cache 变量
        next_decoder_cache = () if use_cache else None
        
        # 遍历 self.layer 列表
        for i, layer_module in enumerate(self.layer):
            # 根据 output_hidden_states 的值确定是否将 hidden_states 加入到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            # 根据 head_mask 是否为 None 决定是否创建 layer_head_mask 变量
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 根据 past_key_values 是否为 None 决定是否创建 past_key_value 变量
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            # 调用 layer_module 的 __call__ 方法，并传入相应参数，得到 layer_outputs
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
                training,
            )
            
            # 将 layer_outputs 中的第一个元素（hidden_states）赋值给 hidden_states
            hidden_states = layer_outputs[0]
            # 根据 use_cache 是否为 True 决定是否将 layer_outputs[-1] 加入到 next_decoder_cache 中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 根据 output_attentions 是否为 True 决定是否将 layer_outputs[1] 加入到 all_self_attentions 中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果 self.config.add_cross_attention 为 True，将 layer_outputs[2] 加入到 all_cross_attentions 中
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
        
        # 根据 self.emb_layer_norm_after 的真假判断是否调用 emb_layer_norm_after 方法
        if self.emb_layer_norm_after:
            hidden_states = self.emb_layer_norm_after(hidden_states)
        
        # 根据 output_hidden_states 的值判断是否将 hidden_states 加入到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # 根据 return_dict 的真假判断返回不同的结果
        if not return_dict:
            # 如果 return_dict 为 False，则返回指定的几个变量中非 None 的部分
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        else:
            # 如果 return_dict 为 True，则返回 TFBaseModelOutputWithPastAndCrossAttentions 类的实例
            return TFBaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=next_decoder_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
                cross_attentions=all_cross_attentions,
            )
    # 构建神经网络模型
    def build(self, input_shape=None):
        # 如果模型已经构建过，直接返回
        if self.built:
            return
        # 将模型标记为已构建
        self.built = True
        # 如果存在emb_layer_norm_after属性
        if getattr(self, "emb_layer_norm_after", None) is not None:
            # 在 TensorFlow 中设置名称范围
            with tf.name_scope(self.emb_layer_norm_after.name):
                # 构建emb_layer_norm_after属性
                self.emb_layer_norm_after.build([None, None, self.config.hidden_size])
        # 如果存在layer属性
        if getattr(self, "layer", None) is not None:
            # 遍历每个层
            for layer in self.layer:
                # 在 TensorFlow 中设置名称范围
                with tf.name_scope(layer.name):
                    # 构建该层
                    layer.build(None)
# 从transformers.models.bert.modeling_tf_bert.TFBertPooler复制并把Bert改成Esm
class TFEsmPooler(tf.keras.layers.Layer):
    # 初始化方法，接收EsmConfig类型的config参数
    def __init__(self, config: EsmConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 创建一个全连接层，设置units为config.hidden_size，kernel_initializer为config.initializer_range，激活函数为"tanh"
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        # 存储config参数
        self.config = config

    # 调用方法，接收tf.Tensor类型的hidden_states参数，返回tf.Tensor类型的pooled_output
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # "池化"模型，直接取第一个标记对应的隐藏状态
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(inputs=first_token_tensor)

        return pooled_output

    # 构建方法，接收input_shape参数，默认为None
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 设置为已构建状态
        self.built = True
        # 如果存在dense属性
        if getattr(self, "dense", None) is not None:
            # 在名称空间self.dense.name下构建dense层
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# Esm预训练模型类，继承自TFPreTrainedModel
class TFEsmPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 配置类为EsmConfig
    config_class = EsmConfig
    # 基础模型前缀为"esm"

# ESM_START_DOCSTRING常量
ESM_START_DOCSTRING = r"""

    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a Keras [Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a
    regular Keras model and refer to the TF/Keras documentation for all matters related to general usage and behavior.

    Parameters:
        config ([`EsmConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

# ESM_INPUTS_DOCSTRING常量
ESM_INPUTS_DOCSTRING = r"""
        Args:
            input_ids (`tf.Tensor` of shape `({0})`):
                输入序列标记在词汇表中的索引。

                可以使用 [`AutoTokenizer`] 获取索引。有关详细信息，请参阅 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`]。

                [什么是输入 ID？](../glossary#input-ids)
            attention_mask (`tf.Tensor` of shape `({0})`, *optional*):
                避免对填充标记索引执行注意力操作的掩码。掩码值为 `[0, 1]` 中的一个：

                - 对于 **未掩码** 的标记，为 1，
                - 对于 **掩码** 的标记，为 0。

                [什么是注意力掩码？](../glossary#attention-mask)
            position_ids (`tf.Tensor` of shape `({0})`, *optional*):
                输入序列标记在位置嵌入中的位置索引。在范围 `[0, config.max_position_embeddings - 1]` 中选择。

                [什么是位置 ID？](../glossary#position-ids)
            head_mask (`tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                用于将自注意力模块中的特定头部置零的掩码。掩码值为 `[0, 1]` 中的一个：

                - 表示 **未掩码** 的头部，为 1，
                - 表示 **掩码** 的头部，为 0。

            inputs_embeds (`tf.Tensor` of shape `({0}, hidden_size)`, *optional*):
                可选地，您可以选择直接传递嵌入表示，而不是传递 `input_ids`。如果您希望对 `input_ids` 索引转换为关联向量具有更多控制权，则这是有用的，而不是使用模型的内部嵌入查找矩阵。
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回的张量中的 `attentions`。
            output_hidden_states (`bool`, *optional*):
                是否返回所有层的隐藏状态。有关更多详细信息，请参见返回的张量中的 `hidden_states`。
            return_dict (`bool`, *optional*):
                是否返回 [`~file_utils.ModelOutput`] 而不是普通元组。
"""
定义了一个名为TFEsmMainLayer的类，继承自Layer类
该类用于输出ESM模型的原始隐藏状态，没有特定的头部
TFEsmMainLayer可以作为编码器或解码器，如果它作为解码器，则在自注意力层之间添加了一个交叉注意力层，遵循Attention is all you need中描述的架构
要使其成为解码器，需要使用配置的is_decoder参数设置为True进行初始化
要在Seq2Seq模型中使用该类，需要将is_decoder参数和add_cross_attention参数都设置为True; 在前向传递中，需要输入encoder_hidden_states。
"""

@add_start_docstrings(
    "The bare ESM Model transformer outputting raw hidden-states without any specific head on top.",
    ESM_START_DOCSTRING,
)
class TFEsmMainLayer(Layer):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, add_pooling_layer=True, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.config = config
        self.is_decoder = config.is_decoder

        # 创建一个TFEsmEmbeddings对象，用于处理嵌入层
        self.embeddings = TFEsmEmbeddings(config, name="embeddings")
        # 创建一个TFEsmEncoder对象，用于处理编码层
        self.encoder = TFEsmEncoder(config, name="encoder")
        # 如果add_pooling_layer为True，则创建一个TFEsmPooler对象，用于在编码层之后添加池化层，否则为None
        self.pooler = TFEsmPooler(config, name="pooler") if add_pooling_layer else None
        # 创建一个TFEsmContactPredictionHead对象，用于处理接触预测头部，接触预测头部的输入特征数为self.config.num_hidden_layers * self.config.num_attention_heads，并且有偏置项，名称为"contact_head"
        self.contact_head = TFEsmContactPredictionHead(
            in_features=self.config.num_hidden_layers * self.config.num_attention_heads, bias=True, name="contact_head"
        )

    def build(self, input_shape=None):
        # 如果已经构建，则返回
        if self.built:
            return
        self.built = True
        # 构建TFEsmEmbeddings对象
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        # 构建TFEsmEncoder对象
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 构建TFEsmPooler对象
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build(None)
        # 构建TFEsmContactPredictionHead对象
        if getattr(self, "contact_head", None) is not None:
            with tf.name_scope(self.contact_head.name):
                self.contact_head.build(None)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value: tf.Variable):
        self.embeddings.word_embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError
    # 定义一个类方法call，接受多个参数，其中input_ids为TFModelInputType类型或None
    # attention_mask为np.ndarray或tf.Tensor类型或None
    # position_ids为np.ndarray或tf.Tensor类型或None
    # head_mask为np.ndarray或tf.Tensor类型或None
    # inputs_embeds为np.ndarray或tf.Tensor类型或None
    # encoder_hidden_states为np.ndarray或tf.Tensor类型或None
    # encoder_attention_mask为np.ndarray或tf.Tensor类型或None
    # past_key_values为一个可选的Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]类型参数
    # use_cache为一个可选的bool类型参数
    # output_attentions为一个可选的bool类型参数
    # output_hidden_states为一个可选的bool类型参数
    # return_dict为一个可选的bool类型参数
    # training为一个bool类型参数，默认为False
    # 定义一个类方法predict_contacts，接受tokens和attention_mask两个参数
    def predict_contacts(self, tokens, attention_mask):
        # 调用self对象的方法，并传入tokens和attention_mask参数，设置return_dict和output_attentions参数为True
        attns = self(tokens, attention_mask=attention_mask, return_dict=True, output_attentions=True).attentions
        # 将attentions列表堆叠为一个张量，axis=1表示沿着列方向堆叠
        attns = tf.stack(attns, axis=1)  # Matches the original model layout
        # 在原始模型中，填充标记的注意力完全被置零
        # 这在大多数情况下都没关系，因为其他标记不会关注它们，
        # 但是对于需要以注意力为输入的接触预测任务来说很重要
        # 因此在这里我们必须模仿原始模型的行为
        # 将attention_mask转换为和attns相同的数据类型
        attention_mask = tf.cast(attention_mask, attns.dtype)
        # 将attns和attention_mask相乘，实现对填充token的注意力置零
        attns *= attention_mask[:, None, None, None]
        attns *= attention_mask[:, None, None, :, None]
        # 调用self对象的contact_head方法，传入tokens和attns参数
        return self.contact_head(tokens, attns)
# 添加起始文档字符串，描述该模型是一个裸的 ESM 模型转换器，输出没有特定顶部的原始隐藏状态。
# 引用 ESM_START_DOCSTRING 中的描述
# 创建 TFEsmModel 类，继承自 TFEsmPreTrainedModel
class TFEsmModel(TFEsmPreTrainedModel):
    # 初始化函数，接受一个 EsmConfig 类型的配置参数和其他可选参数
    def __init__(self, config: EsmConfig, add_pooling_layer=True, *inputs, **kwargs):
        # 调用父类的初始化函数
        super().__init__(config, *inputs, **kwargs)

        # 给对象添加一个名为 esm 的 TFEsmMainLayer 类型的实例
        self.esm = TFEsmMainLayer(config, add_pooling_layer=add_pooling_layer, name="esm")

    # 定义一个 call 方法，接受一系列输入参数 
    # 使用 @unpack_inputs 装饰器
    # 在模型前向传播过程中添加起始文档字符串，引用 ESM_INPUTS_DOCSTRING 中的描述
    # 添加代码示例文档字符串，引用 _CHECKPOINT_FOR_DOC、TFBaseModelOutputWithPoolingAndCrossAttentions、_CONFIG_FOR_DOC 中的描述
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,


        # 此处省略了一些参数，不进行详细解释
        #  实际代码中应包含完整参数列表
    def forward(
        self, 
        input_ids: tf.Tensor, 
        attention_mask: tf.Tensor = None, 
        position_ids: tf.Tensor = None, 
        head_mask: tf.Tensor = None, 
        inputs_embeds: tf.Tensor = None, 
        encoder_hidden_states: tf.Tensor = None, 
        encoder_attention_mask: tf.Tensor = None, 
        past_key_values: Tuple[Tuple[tf.Tensor]] = None, 
        use_cache: bool = True, 
        output_attentions: bool = False, 
        output_hidden_states: bool = False, 
        return_dict: bool = True, 
        training: bool = False
    ) -> Union[TFBaseModelOutputWithPoolingAndCrossAttentions, Tuple[tf.Tensor]]:
        r"""
        encoder_hidden_states  (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

        past_key_values (`Tuple[Tuple[tf.Tensor]]` of length `config.n_layers`)
            contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*, defaults to `True`):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`). Set to `False` during training, `True` during generation
        """
        # 使用ESM模型进行前向传播
        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 返回输出结���
        return outputs

    # 预测接触点的方法
    def predict_contacts(self, tokens, attention_mask):
        return self.esm.predict_contacts(tokens, attention_mask)

    # 构建方法
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果esm属性存在
        if getattr(self, "esm", None) is not None:
            # 使用ESM模型的名称进行命名
            with tf.name_scope(self.esm.name):
                # 构建ESM模型
                self.esm.build(None)
# 为EsmForMaskedLM模型添加文档字符串和ESM_START_DOCSTRING中定义的语言模型头
@add_start_docstrings("""ESM Model with a `language modeling` head on top.""", ESM_START_DOCSTRING)
class TFEsmForMaskedLM(TFEsmPreTrainedModel, TFMaskedLanguageModelingLoss):
    # 在加载丢失的键时要忽略的键列表
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    # 在加载意外的键时要忽略的键列表
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    # 初始化函数，传入参数为配置config
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)

        # 如果配置中指定为解码器，则发出警告
        if config.is_decoder:
            logger.warning(
                "If you want to use `EsmForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 初始化ESM模型层，不添加池化层，命名为"esm"
        self.esm = TFEsmMainLayer(config, add_pooling_layer=False, name="esm")
        # 初始化ESM语言建模头，命名为"lm_head"
        self.lm_head = TFEsmLMHead(config, name="lm_head")
        # 如果配置指定词嵌入层要绑定，则确保构建了词嵌入层
        if config.tie_word_embeddings:
            # 确保word embeddings被构建，以便我们有实际可绑定的东西
            with tf.name_scope(os.path.join(self._name_scope(), "esm", "embeddings", "word_embeddings")):
                self.esm.embeddings.word_embeddings.build((None, None))
            self.lm_head.decoder = self.esm.embeddings.word_embeddings.weights[0]

    # 获取输出嵌入层
    def get_output_embeddings(self):
        return self.lm_head.decoder

    # 设置输出嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    # 获取语言建模头
    def get_lm_head(self):
        return self.lm_head

    # 对模型进行前向传播
    @unpack_inputs
    @add_start_docstrings_to_model_forward(ESM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="<mask>",
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        labels: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        ):
    ) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional`):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        # 设置返回字典，如果未指定则使用配置文件中的返回字典设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 ESM 模型进行处理
        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        # 如果存在标签，则计算 masked language modeling 损失
        if labels is not None:
            masked_lm_loss = self.hf_compute_loss(labels=labels, logits=prediction_scores)

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return TFMaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 预测接触点
    def predict_contacts(self, tokens, attention_mask):
        return self.esm.predict_contacts(tokens, attention_mask)

    # 构建模型
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果存在 ESM 模型，则构建 ESM 模型
        if getattr(self, "esm", None) is not None:
            with tf.name_scope(self.esm.name):
                self.esm.build(None)
        # 如果存在 lm_head 模型，则构建 lm_head 模型
        if getattr(self, "lm_head", None) is not None:
            with tf.name_scope(self.lm_head.name):
                self.lm_head.build(None)
class TFEsmLMHead(Layer):
    """ESM Head for masked language modeling."""

    # 初始化方法，设置神经网络层
    def __init__(self, config, name=None):
        super().__init__(name=name)
        # 创建一个全连接层，隐藏层大小为config.hidden_size
        self.dense = Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # 创建一个 LayerNormalization 层
        self.layer_norm = LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        
        # 根据配置参数判断是否共享词嵌入层
        if config.tie_word_embeddings:
            self.decoder = None
        else:
            # 创建一个全连接层，输出大小为config.vocab_size，不使用偏置
            self.decoder = Dense(
                config.vocab_size,
                kernel_initializer=get_initializer(config.initializer_range),
                name="decoder",
                use_bias=False,
            )
        self.config = config

    # 构建网络结构
    def build(self, input_shape=None):
        # 在构建之前做一些处理，如设置 bias、建立层的权重等
        if self.built:
            return
        self.built = True
        # 添加 bias 权重
        self.bias = self.add_weight("bias", shape=(self.config.vocab_size,), initializer="zeros", trainable=True)
        
        # 如果存在 dense 层，建立权重
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果存在 layer_norm 层，建立权重
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.hidden_size])
        # 如果存在 decoder 层且不共享词嵌入，建立权重
        if getattr(self, "decoder", None) is not None and not self.config.tie_word_embeddings:
            with tf.name_scope(self.decoder.name):
                self.decoder.build([None, None, self.config.hidden_size])

    # 获取 bias 权重
    def get_bias(self):
        return {"bias": self.bias}

    # 神经网络前向传播
    def call(self, features):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # 投影到词汇表大小，添加偏置
        if self.config.tie_word_embeddings:
            x = tf.matmul(x, self.decoder, transpose_b=True) + self.bias
        else:
            x = self.decoder(x) + self.bias
        return x


@add_start_docstrings(
    """
    ESM Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    ESM_START_DOCSTRING,
)
class TFEsmForSequenceClassification(TFEsmPreTrainedModel, TFSequenceClassificationLoss):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # 初始化方法，创建 ESM 模型和分类器
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        # 创建 ESM 模型
        self.esm = TFEsmMainLayer(config, add_pooling_layer=False, name="esm")
        # 创建分类器
        self.classifier = TFEsmClassificationHead(config, name="classifier")

    # 解压输入参数，为模型正向传播添加文档字符串
    @unpack_inputs
    @add_start_docstrings_to_model_forward(ESM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        labels: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
    r"""
    labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
        Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
        config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
        `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
    # 调用 ESMM 及相关操作，处理模型的输入和输出
    outputs = self.esm(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        training=training,
    )
    # 获取 ESMM 的输出序列
    sequence_output = outputs[0]
    # 通过分类器得到预测的 logits
    logits = self.classifier(sequence_output)
    
    # 计算损失值，如果 labels 不为空
    loss = None if labels is None else self.hf_compute_loss(labels, logits)
    
    # 如果不需要返回字典，则构建输出并返回
    if not return_dict:
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    
    # 返回 TFSequenceClassifierOutput 类型的输出字典
    return TFSequenceClassifierOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
    
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True
        # 构建 ESMM 模型
        if getattr(self, "esm", None) is not None:
            with tf.name_scope(self.esm.name):
                self.esm.build(None)
        # 构建分类器
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build(None)
# 添加起始文档字符串，描述该模型在顶部的标记分类头部分的作用，例如用于命名实体识别（NER）任务
@add_start_docstrings(
    """
    ESM Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    ESM_START_DOCSTRING,
)
# 定义 TFEsmForTokenClassification 类，继承自 TFEsmPreTrainedModel 和 TFTokenClassificationLoss
class TFEsmForTokenClassification(TFEsmPreTrainedModel, TFTokenClassificationLoss):
    # 在加载时忽略的键列表，对于未预期的键
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    # 在加载时忽略的键列表，对于缺失的键
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # 初始化函数，接受配置参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置标签数目
        self.num_labels = config.num_labels

        # 实例化 ESM 主层，不添加池化层
        self.esm = TFEsmMainLayer(config, add_pooling_layer=False, name="esm")
        # 设置 dropout 层
        self.dropout = Dropout(config.hidden_dropout_prob)
        # 设置分类器层
        self.classifier = Dense(config.num_labels, name="classifier")
        # 保存配置参数
        self.config = config

    # 调用方法，用于模型的前向传播
    @unpack_inputs
    # 添加起始文档字符串到模型的前向传播，描述输入参数的作用和形状
    @add_start_docstrings_to_model_forward(ESM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 添加代码示例文档字符串到模型的前向传播，描述其用法、参数、返回类型等
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义模型的前向传播方法
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        labels: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        # 如果标签存在，则返回字典，否则返回 None
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 ESM 主层的前向传播方法
        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 获取序列输出
        sequence_output = outputs[0]

        # 对序列输出应用 dropout，用于训练时防止过拟合
        sequence_output = self.dropout(sequence_output, training=training)
        # 应用分类器层，得到预测的标签 logits
        logits = self.classifier(sequence_output)

        # 如果标签不存在，则损失为 None，否则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果不返回字典，则返回元组
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TFTokenClassifierOutput 对象，包含损失、logits、隐藏状态和注意力权重
        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    # 构建神经网络模型
    def build(self, input_shape=None):
        # 如果模型已经构建好，则直接返回
        if self.built:
            return
        # 标记模型已构建
        self.built = True
        # 如果存在 esm 属性，按照 esm 的名字创建命名空间，并构建 esm
        if getattr(self, "esm", None) is not None:
            with tf.name_scope(self.esm.name):
                self.esm.build(None)
        # 如果存在 classifier 属性，按照 classifier 的名字创建命名空间，并构建 classifier
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
class TFEsmClassificationHead(Layer):
    """Head for sentence-level classification tasks."""
    # 定义一个用于句子级分类任务的头部

    def __init__(self, config, name=None):
        super().__init__(name=name)
        # 调用父类的构造函数
        self.dense = Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        # 创建一个全连接层，用于变换特征向量
        self.dropout = Dropout(config.hidden_dropout_prob)
        # 创建一个dropout层，用于防止过拟合
        self.out_proj = Dense(
            config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="linear",
            name="out_proj",
        )
        # 创建一个全连接层，用于输出分类结果
        self.config = config
        # 存储配置信息

    def call(self, features, training=False):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        # 取出每个样本的第一个token作为特征
        x = self.dropout(x, training=training)
        # 对特征进行dropout操作
        x = self.dense(x)
        # 对特征进行全连接层变换
        x = self.dropout(x, training=training)
        # 再次进行dropout操作
        x = self.out_proj(x)
        # 对变换后的特征进行全连接层变换，得到最终输出
        return x

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经构建过了，则直接返回
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果dense层存在，则构建dense层
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.config.hidden_size])
        # 如果out_proj层存在，则构建out_proj层


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: tf.Tensor x:

    Returns: tf.Tensor
    """
    # 用输入的token序列生成对应的位置id序列
    # 非填充符号用其位置数字代替。位置数字从padding_idx+1开始。忽略填充符号。
    # 这是从fairseq的`utils.make_positions`修改的。

    # 计算非填充符号的掩码
    mask = tf.cast(input_ids != padding_idx, tf.int64)
    # 计算递增的位置索引
    incremental_indices = (tf.cumsum(mask, axis=1) + past_key_values_length) * mask
    return incremental_indices + padding_idx
    # 返回填充后的位置id序列
```