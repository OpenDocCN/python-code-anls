# `.\models\esm\modeling_tf_esm.py`

```
# 设定编码格式为 UTF-8

# 版权声明和许可证信息

# 导入所需的库和模块
from __future__ import annotations  # 使用未来版本的 annotations 特性

import os  # 导入操作系统相关的功能
from typing import Optional, Tuple, Union  # 引入类型提示需要的数据结构

import numpy as np  # 导入 NumPy 库，用于科学计算
import tensorflow as tf  # 导入 TensorFlow 深度学习框架

# 导入 HuggingFace Transformers 相关的文件操作和模型输出等
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
    keras,
    shape_list,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, stable_softmax
from ...utils import logging  # 导入日志记录工具
from .configuration_esm import EsmConfig  # 导入 ESM 模型的配置文件

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

# 用于文档的模型检查点和配置信息
_CHECKPOINT_FOR_DOC = "facebook/esm2_t6_8M_UR50D"
_CONFIG_FOR_DOC = "EsmConfig"

# 预训练模型存档列表
TF_ESM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/esm2_t6_8M_UR50D",
    "facebook/esm2_t12_35M_UR50D",
    # 这里没有列出所有 ESM 模型，可以在 https://huggingface.co/models?filter=esm 查看完整列表
]


def rotate_half(x):
    """
    将张量沿最后一个维度分割成两半，然后进行旋转操作。
    Args:
        x: 输入的张量

    Returns:
        tf.Tensor: 旋转后的张量
    """
    x1, x2 = tf.split(x, 2, axis=-1)
    return tf.concat((-x2, x1), axis=-1)


def apply_rotary_pos_emb(x, cos, sin):
    """
    应用旋转位置嵌入到输入张量 x 中。
    Args:
        x: 输入的张量
        cos: 余弦值张量
        sin: 正弦值张量

    Returns:
        tf.Tensor: 应用旋转位置嵌入后的张量
    """
    cos = cos[:, :, : tf.shape(x)[-2], :]
    sin = sin[:, :, : tf.shape(x)[-2], :]

    return (x * cos) + (rotate_half(x) * sin)


def symmetrize(x):
    """
    对最后两个维度进行转置操作，使层对称化，用于接触预测。
    Args:
        x: 输入张量

    Returns:
        tf.Tensor: 对称化后的张量
    """
    return x + tf.linalg.matrix_transpose(x)  # 仅转置最后两个维度


def average_product_correct(x):
    """
    执行平均产品校正，用于接触预测。
    Args:
        x: 输入张量

    Returns:
        tf.Tensor: 校正后的张量
    """
    a1 = tf.reduce_sum(x, -1, keepdims=True)
    a2 = tf.reduce_sum(x, -2, keepdims=True)
    a12 = tf.reduce_sum(x, (-1, -2), keepdims=True)

    avg = a1 * a2
    avg = avg / a12
    normalized = x - avg
    return normalized


class TFRotaryEmbedding(keras.layers.Layer):
    """
    基于 RoFormer 中的旋转位置嵌入，对查询和键进行旋转矩阵变换，依赖它们的相对位置。
    """

    # 在此类中定义相关的方法和初始化操作
    def __init__(self, dim: int, name=None):
        super().__init__(name=name)
        # Matt: The PyTorch version of this layer does a lot of work to cache values, but we just rely on TF compilation
        # and/or XLA to sort out constants like that. It actually may not seem like this layer needs to be stateful at
        # all when we benefit from TF compilation, but it does. The reason is that self.inv_freq is a buffer in the
        # original implementation, but all the shared ESM checkpoints were trained with fp16 params. This means that
        # the inv_freq tensor was stored as a float16, and we need to replicate those lower-precision values or our
        # models give different outputs from the original.
        self.dim = dim

    def build(self, input_shape):
        super().build(input_shape)
        # 创建一个名为 "inv_freq" 的权重变量，其形状为 (self.dim // 2,)，数据类型为 tf.float32，初始化为 1.0，不可训练
        self.inv_freq = self.add_weight(
            "inv_freq", shape=(self.dim // 2,), dtype=tf.float32, initializer=get_initializer(1.0), trainable=False
        )
        # 计算 inv_freq 的值，这是一个与序列长度相关的正弦余弦嵌入频率
        self.inv_freq.assign(
            1.0 / (10000 ** (tf.range(start=0, limit=self.dim, delta=2, dtype=tf.float32) / self.dim))
        )

    def _compute_cos_sin(self, x, seq_dimension=2):
        # 获取输入张量 x 的序列长度
        seq_len = tf.shape(x)[seq_dimension]

        # 创建一个序列 t，数据类型与 self.inv_freq 相同，长度为 seq_len
        t = tf.range(seq_len, dtype=self.inv_freq.dtype)
        # 计算频率矩阵 freqs，是 t 和 self.inv_freq 的外积
        freqs = tf.einsum("i, j -> ij", t, self.inv_freq)  # Outer multiplication
        # 创建正弦和余弦嵌入矩阵 emb，通过连接 freqs 和其自身的拷贝，axis=-1 表示在最后一个维度上连接
        emb = tf.concat((freqs, freqs), axis=-1)[None, None, :, :]

        return tf.cos(emb), tf.sin(emb)

    def call(self, q: tf.Tensor, k: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # 计算正弦和余弦嵌入矩阵 cos_emb 和 sin_emb，针对张量 k 在序列维度上进行计算
        cos_emb, sin_emb = self._compute_cos_sin(k, seq_dimension=-2)

        # 应用旋转位置嵌入到输入张量 q 和 k 上，并返回结果
        return (
            apply_rotary_pos_emb(q, cos_emb, sin_emb),
            apply_rotary_pos_emb(k, cos_emb, sin_emb),
        )
class TFEsmContactPredictionHead(keras.layers.Layer):
    """Performs symmetrization, apc, and computes a logistic regression on the output features"""

    def __init__(
        self,
        in_features: int,
        bias=True,
        eos_idx: int = 2,
        name=None,
    ):
        super().__init__(name=name)
        self.eos_idx = eos_idx  # 设置 eos 标记的索引值
        self.in_features = in_features  # 输入特征的维度
        self.regression = keras.layers.Dense(1, use_bias=bias, activation="sigmoid", name="regression")  # 定义逻辑回归层

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True  # 标记层已经构建
        if getattr(self, "regression", None) is not None:
            with tf.name_scope(self.regression.name):
                self.regression.build((None, self.in_features))  # 构建逻辑回归层的计算图

    def call(self, tokens, attentions):
        # remove eos token attentions
        eos_mask = tf.cast(tokens != self.eos_idx, attentions.dtype)  # 创建一个用于屏蔽 eos 标记的掩码
        eos_mask = tf.expand_dims(eos_mask, 1) * tf.expand_dims(eos_mask, 2)  # 将掩码扩展到适当的维度
        attentions = attentions * eos_mask[:, None, None, :, :]  # 使用掩码屏蔽 eos 标记的注意力值
        attentions = attentions[..., :-1, :-1]  # 移除最后一个维度中的 eos 标记的注意力值

        # remove cls token attentions
        attentions = attentions[..., 1:, 1:]  # 移除第一个维度中的 cls 标记的注意力值
        batch_size, layers, heads, seqlen, _ = shape_list(attentions)  # 获取注意力张量的形状信息
        attentions = tf.reshape(attentions, (batch_size, layers * heads, seqlen, seqlen))  # 重新整形注意力张量的维度

        # features: batch x channels x tokens x tokens (symmetric)
        attentions = average_product_correct(symmetrize(attentions))  # 对注意力张量进行对称化和平均产品校正
        attentions = tf.transpose(attentions, perm=(0, 2, 3, 1))  # 转置注意力张量的维度顺序
        return tf.squeeze(self.regression(attentions), 3)  # 使用逻辑回归层进行预测，并压缩维度以匹配输出形状


class TFEsmEmbeddings(keras.layers.Layer):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    # 初始化函数，用于创建一个新的对象实例
    def __init__(self, config, name=None):
        # 调用父类的初始化函数
        super().__init__(name=name)
        # 创建词嵌入层，用于将词汇索引映射到向量表示
        self.word_embeddings = keras.layers.Embedding(
            config.vocab_size,
            config.hidden_size,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="word_embeddings",
        )
        # 创建位置嵌入层，用于表示输入序列中每个位置的信息
        self.position_embeddings = keras.layers.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="position_embeddings",
        )

        # 根据配置选择是否添加层归一化操作
        if config.emb_layer_norm_before:
            self.layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        else:
            self.layer_norm = None
        # 定义位置嵌入类型，默认为绝对位置编码
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        # 创建位置 ID，用于表示序列中每个位置的索引
        self.position_ids = tf.range(config.max_position_embeddings)[None, :]

        # 定义填充符的索引
        self.padding_idx = config.pad_token_id
        # 定义是否对 token 进行 dropout 的配置
        self.token_dropout = config.token_dropout
        # 定义 mask token 的索引
        self.mask_token_id = config.mask_token_id
        # 保存配置对象的引用
        self.config = config
        ):
            if position_ids is None:
                if input_ids is not None:
                    # 从输入的标记 IDs 创建位置 IDs。任何填充的标记保持填充状态。
                    position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
                else:
                    position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

            if inputs_embeds is None:
                # 检查输入的标记 IDs 是否在词汇表大小范围内
                check_embeddings_within_bounds(input_ids, self.config.vocab_size)
                inputs_embeds = self.word_embeddings(input_ids)

            # 注意：如果未来要支持 ESM-1（而不是1b！），则需要在此处支持嵌入比例因子。
            embeddings = inputs_embeds

            # Matt: ESM 有处理 MLM 掩码的选项，稍微不同于通常。如果 token_dropout 标志为 False，
            # 则与 BERT/RoBERTa 处理方式相同。如果设置为 True，则屏蔽的标记被视为选择输入丢失并清零。
            # 当屏蔽的标记不存在时，通过缩放嵌入来补偿 (训练期间未屏蔽标记的比例) / (样本中未屏蔽标记的比例)。
            # 这类似于评估期间丢弃层缩小输出的方式（或者等价地，在训练期间缩放未丢弃的输出）。
            if self.token_dropout:
                # 将屏蔽标记的嵌入清零
                embeddings = tf.where((input_ids == self.mask_token_id)[:, :, None], 0.0, embeddings)
                # 训练时的屏蔽比率，硬编码为所有 ESM 模型训练运行中使用的比率
                mask_ratio_train = 0.15 * 0.8
                # 计算源长度
                src_lengths = tf.cast(tf.reduce_sum(attention_mask, axis=-1), tf.float32)
                # 检查是否有屏蔽的标记
                masked_tokens = input_ids == self.mask_token_id
                # 观察到的屏蔽比率
                mask_ratio_observed = tf.math.count_nonzero(masked_tokens, dtype=tf.float32, axis=-1) / src_lengths
                # 缩放嵌入以补偿 mask-dropout
                embeddings = embeddings * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

            if self.position_embedding_type == "absolute":
                # 如果位置嵌入类型为绝对位置，则添加位置嵌入到嵌入中
                position_embeddings = self.position_embeddings(position_ids)
                embeddings += position_embeddings

            if self.layer_norm is not None:
                # 如果有层归一化，则对嵌入进行归一化
                embeddings = self.layer_norm(embeddings)
            if attention_mask is not None:
                # 如果存在注意力掩码，则将其应用于嵌入
                embeddings = embeddings * tf.cast(tf.expand_dims(attention_mask, -1), embeddings.dtype)
            # Matt: 我认为这行代码从 BERT 复制过来时出错了，暂时禁用它。
            # embeddings = self.dropout(embeddings)
            return embeddings
    # 如果已经构建过模型则直接返回，避免重复构建
    if self.built:
        return
    
    # 标记模型已经构建
    self.built = True
    
    # 如果存在词嵌入，则构建词嵌入层
    if getattr(self, "word_embeddings", None) is not None:
        # 在词嵌入的命名空间下，构建词嵌入层
        with tf.name_scope(self.word_embeddings.name):
            self.word_embeddings.build(None)
    
    # 如果存在位置嵌入，则构建位置嵌入层
    if getattr(self, "position_embeddings", None) is not None:
        # 在位置嵌入的命名空间下，构建位置嵌入层
        with tf.name_scope(self.position_embeddings.name):
            self.position_embeddings.build(None)
    
    # 如果存在层归一化，则构建层归一化层
    if getattr(self, "layer_norm", None) is not None:
        # 在层归一化的命名空间下，构建层归一化层，输入形状为 [None, None, self.config.hidden_size]
        with tf.name_scope(self.layer_norm.name):
            self.layer_norm.build([None, None, self.config.hidden_size])
class TFEsmSelfAttention(keras.layers.Layer):
    # 定义一个自注意力层的 TensorFlow 扩展类
    def __init__(self, config, position_embedding_type=None, name=None):
        # 初始化函数，设置参数并配置层的名称
        super().__init__(name=name)
        # 检查隐藏大小是否可以被注意力头数整除，若不能则抛出错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 设置注意力头数和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 创建查询、键、值的全连接层
        self.query = keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        self.key = keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        self.value = keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )

        # 设置注意力概率的 dropout 层
        self.dropout = keras.layers.Dropout(config.attention_probs_dropout_prob)
        # 设置位置嵌入类型，默认为绝对位置
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        self.rotary_embeddings = None
        # 如果位置嵌入类型是相对键或者相对键-查询，则创建距离嵌入层
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = keras.layers.Embedding(
                2 * config.max_position_embeddings - 1,
                self.attention_head_size,
                embeddings_initializer=get_initializer(config.initializer_range),
            )
        # 如果位置嵌入类型是旋转，则创建旋转嵌入对象
        elif self.position_embedding_type == "rotary":
            self.rotary_embeddings = TFRotaryEmbedding(dim=self.attention_head_size, name="rotary_embeddings")

        # 设置是否为解码器的标志和存储配置信息
        self.is_decoder = config.is_decoder
        self.config = config

    def transpose_for_scores(self, x: tf.Tensor) -> tf.Tensor:
        # 重新排列张量的维度以便进行注意力计算
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
        **kwargs
    ) -> Tuple[tf.Tensor, Optional[Tuple[tf.Tensor]]]:
        # 定义自注意力层的调用方法，处理输入张量并返回处理后的张量和可选的注意力张量
    # 定义 build 方法，用于构建模型结构
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回，避免重复构建
        if self.built:
            return
        # 标记模型为已构建状态
        self.built = True
        
        # 如果存在查询（query）属性，则构建查询的结构
        if getattr(self, "query", None) is not None:
            # 使用查询的名称作为命名空间
            with tf.name_scope(self.query.name):
                # 构建查询的形状为 [None, None, self.config.hidden_size]
                self.query.build([None, None, self.config.hidden_size])
        
        # 如果存在键（key）属性，则构建键的结构
        if getattr(self, "key", None) is not None:
            # 使用键的名称作为命名空间
            with tf.name_scope(self.key.name):
                # 构建键的形状为 [None, None, self.config.hidden_size]
                self.key.build([None, None, self.config.hidden_size])
        
        # 如果存在值（value）属性，则构建值的结构
        if getattr(self, "value", None) is not None:
            # 使用值的名称作为命名空间
            with tf.name_scope(self.value.name):
                # 构建值的形状为 [None, None, self.config.hidden_size]
                self.value.build([None, None, self.config.hidden_size])
        
        # 如果存在旋转嵌入（rotary_embeddings）属性，则构建其结构
        if getattr(self, "rotary_embeddings", None) is not None:
            # 使用旋转嵌入的名称作为命名空间
            with tf.name_scope(self.rotary_embeddings.name):
                # 构建旋转嵌入，传入 None 作为输入形状参数
                self.rotary_embeddings.build(None)
# 自定义 Keras 层，实现自注意力机制的输出层
class TFEsmSelfOutput(keras.layers.Layer):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        # 创建一个全连接层，用于映射隐藏状态到指定大小的输出空间
        self.dense = keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个 Dropout 层，用于在训练时随机丢弃部分神经元，防止过拟合
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        # 保存配置信息
        self.config = config

    def call(self, hidden_states, input_tensor, training=False):
        # 将隐藏状态通过全连接层映射，并应用激活函数
        hidden_states = self.dense(hidden_states)
        # 在训练模式下，对映射后的输出应用 Dropout
        hidden_states = self.dropout(hidden_states, training=training)
        # 将映射后的输出与输入张量相加，实现残差连接
        hidden_states += input_tensor
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果层已构建，则直接返回；否则，构建全连接层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # 构建全连接层，输入形状为 [None, None, hidden_size]
                self.dense.build([None, None, self.config.hidden_size])


# 自定义 Keras 层，实现注意力机制的中间层
class TFEsmAttention(keras.layers.Layer):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        # 创建自注意力层
        self.self = TFEsmSelfAttention(config, name="self")
        # 创建自注意力层的输出层
        self.output_layer = TFEsmSelfOutput(config, name="output")
        # 初始化一个空集合，用于存储要剪枝的注意力头
        self.pruned_heads = set()
        # 创建 LayerNormalization 层，用于对输入进行归一化
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 保存配置信息
        self.config = config

    def prune_heads(self, heads):
        # 剪枝方法暂未实现
        raise NotImplementedError

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
        # 对输入的隐藏状态进行 LayerNormalization
        hidden_states_ln = self.LayerNorm(hidden_states)
        # 调用自注意力层进行计算，传入各种参数
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
        # 将自注意力层的输出传递给输出层，同时传入原始的隐藏状态
        attention_output = self.output_layer(self_outputs[0], hidden_states)
        # 组装最终的输出，包括注意力输出和可能的其他信息
        outputs = (attention_output,) + self_outputs[1:]  # 如果需要输出注意力，将其添加到输出中
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果层已构建，则直接返回；否则，构建自注意力层和输出层
        if getattr(self, "self", None) is not None:
            with tf.name_scope(self.self.name):
                # 构建自注意力层
                self.self.build(None)
        if getattr(self, "output_layer", None) is not None:
            with tf.name_scope(self.output_layer.name):
                # 构建自注意力层的输出层
                self.output_layer.build(None)
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                # 构建 LayerNormalization 层，输入形状为 [None, None, hidden_size]
                self.LayerNorm.build([None, None, self.config.hidden_size])
    # 初始化函数，用于创建一个新的实例
    def __init__(self, config: EsmConfig, **kwargs):
        # 调用父类的初始化方法，传入额外的关键字参数
        super().__init__(**kwargs)

        # 创建一个全连接层，用于处理输入数据
        self.dense = keras.layers.Dense(
            units=config.intermediate_size,  # 设置全连接层的输出单元数
            kernel_initializer=get_initializer(config.initializer_range),  # 初始化权重的方式
            name="dense",  # 设置层的名称
        )
        self.config = config  # 保存配置信息到实例中

    # 调用函数，用于定义模型的前向传播逻辑
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)  # 将输入数据传入全连接层处理
        hidden_states = tf.nn.gelu(hidden_states)  # 使用GELU激活函数处理全连接层输出
        return hidden_states  # 返回处理后的数据

    # 构建函数，用于构建模型的层次结构
    def build(self, input_shape=None):
        if self.built:  # 如果模型已经构建过，直接返回
            return
        self.built = True  # 标记模型已构建

        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):  # 使用全连接层的名称作为命名空间
                self.dense.build([None, None, self.config.hidden_size])  # 构建全连接层的结构
# 自定义的 Transformer Encoder 层，继承自 keras.layers.Layer
class TFEsmLayer(keras.layers.Layer):
    # 初始化方法，接收配置参数 config 和可选的层名字 name
    def __init__(self, config, name=None):
        # 调用父类的初始化方法
        super().__init__(name=name)
        # 设定前馈传播时的块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度维度，设定为1
        self.seq_len_dim = 1
        # 创建自注意力层 TFEsmAttention 对象
        self.attention = TFEsmAttention(config, name="attention")
        # 是否作为解码器使用的标志
        self.is_decoder = config.is_decoder
        # 是否添加交叉注意力机制的标志
        self.add_cross_attention = config.add_cross_attention
        # 如果添加了交叉注意力且不是解码器，则引发运行时错误
        if self.add_cross_attention:
            if not self.is_decoder:
                raise RuntimeError(f"{self} should be used as a decoder model if cross attention is added")
            # 创建交叉注意力层 TFEsmAttention 对象
            self.crossattention = TFEsmAttention(config)
        # 创建中间层对象 TFEsmIntermediate
        self.intermediate = TFEsmIntermediate(config, name="intermediate")
        # 创建输出层对象 TFEsmOutput
        self.output_layer = TFEsmOutput(config, name="output")
        # 创建层归一化对象 LayerNorm
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 保存配置对象到 self.config
        self.config = config

    # 调用方法，实现层的前向传播逻辑
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
        # 如果过去的键/值对存在，则提取自注意力的缓存键/值对，位置在1和2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 使用自注意力模块处理隐藏状态，应用注意力掩码和头掩码，输出注意力信息
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            training=training,
        )
        # 提取自注意力模块的输出结果
        attention_output = self_attention_outputs[0]

        # 如果是解码器模型，则最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]  # 提取除了最后一个元素外的所有元素
            present_key_value = self_attention_outputs[-1]  # 提取最后一个元素作为当前键/值对
        else:
            outputs = self_attention_outputs[1:]  # 如果输出注意力权重，则添加自注意力信息

        cross_attn_present_key_value = None
        # 如果是解码器且有编码器的隐藏状态
        if self.is_decoder and encoder_hidden_states is not None:
            # 如果模型没有交叉注意力层，则抛出异常
            if not hasattr(self, "crossattention"):
                raise AttributeError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated"
                    " with cross-attention layers by setting `config.add_cross_attention=True`"
                )

            # 提取交叉注意力模块的缓存键/值对，位置在过去键/值对元组的倒数第二和倒数第一位置
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 使用交叉注意力模块处理自注意力输出、注意力掩码、头掩码、编码器隐藏状态等信息
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
            # 提取交叉注意力模块的输出结果
            attention_output = cross_attention_outputs[0]
            # 添加交叉注意力信息到输出中
            outputs = outputs + cross_attention_outputs[1:-1]

            # 将交叉注意力的键/值对添加到当前键/值对元组的第三和第四位置
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 对注意力输出进行 LayerNorm 处理
        layernorm_output = self.LayerNorm(attention_output)
        # 使用中间层处理 LayerNorm 后的输出
        intermediate_output = self.intermediate(hidden_states=layernorm_output)
        # 使用输出层处理中间层的输出和注意力输出
        layer_output = self.output_layer(
            hidden_states=intermediate_output, input_tensor=attention_output, training=training
        )
        # 将处理后的输出添加到总输出中
        outputs = (layer_output,) + outputs

        # 如果是解码器模型，将注意力的键/值对作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs
    # 构建函数，用于构建模型的各个部分
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回，避免重复构建
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        
        # 如果存在注意力层，构建注意力层并设置名称作用域
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        
        # 如果存在中间层，构建中间层并设置名称作用域
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        
        # 如果存在输出层，构建输出层并设置名称作用域
        if getattr(self, "output_layer", None) is not None:
            with tf.name_scope(self.output_layer.name):
                self.output_layer.build(None)
        
        # 如果存在 LayerNorm 层，构建 LayerNorm 层并设置名称作用域，
        # 输入形状为 [None, None, self.config.hidden_size]
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
# 定义自定义的 Transformer 编码器层，继承自 Keras 的 Layer 类
class TFEsmEncoder(keras.layers.Layer):
    # 初始化方法，接收配置参数和可选的名称
    def __init__(self, config, name=None):
        super().__init__(name=name)
        # 保存配置参数
        self.config = config
        # 创建多个 Transformer 编码层，根据配置中的隐藏层数量
        self.layer = [TFEsmLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]
        # 创建一个 LayerNormalization 层，用于对嵌入层之后的结果进行归一化处理
        self.emb_layer_norm_after = keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="emb_layer_norm_after"
        )

    # 定义调用方法，处理输入数据和各种选项
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
        # 初始化存储所有隐藏状态、自注意力和交叉注意力的元组，如果需要输出的话
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 初始化存储下一个解码器缓存的元组，如果需要使用缓存的话
        next_decoder_cache = () if use_cache else None

        # 遍历所有 Transformer 编码层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前层的隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 获取当前层的过去键值对（如果有的话）
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 调用当前层的处理方法，获取该层的输出结果
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

            # 更新隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]
            # 如果需要使用缓存，则将当前层的输出缓存加入到 next_decoder_cache 中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果需要输出注意力分布，则将当前层的自注意力加入到 all_self_attentions 中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果配置中包含交叉注意力，则将当前层的交叉注意力加入到 all_cross_attentions 中
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果存在嵌入层之后的归一化层，则对最终的隐藏状态进行归一化处理
        if self.emb_layer_norm_after:
            hidden_states = self.emb_layer_norm_after(hidden_states)

        # 如果需要输出所有隐藏状态，则将最终的隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要以字典形式返回结果，则以元组形式返回多个结果
        if not return_dict:
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
        
        # 如果需要以字典形式返回结果，则创建 TFBaseModelOutputWithPastAndCrossAttentions 对象返回
        return TFBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
    # 如果已经构建过网络，则直接返回，避免重复构建
    if self.built:
        return

    # 标记网络已经构建
    self.built = True

    # 如果存在额外的嵌入层归一化操作，构建该层
    if getattr(self, "emb_layer_norm_after", None) is not None:
        # 在 TensorFlow 的命名空间下构建嵌入层归一化操作
        with tf.name_scope(self.emb_layer_norm_after.name):
            # 构建嵌入层归一化操作，指定输入形状为 [None, None, self.config.hidden_size]
            self.emb_layer_norm_after.build([None, None, self.config.hidden_size])

    # 如果存在多层网络结构，逐层构建网络
    if getattr(self, "layer", None) is not None:
        # 遍历每一层网络
        for layer in self.layer:
            # 在 TensorFlow 的命名空间下构建当前层网络
            with tf.name_scope(layer.name):
                # 构建当前层网络，输入形状暂时为 None，表示动态适配
                layer.build(None)
"""
定义一个自定义的 Keras 层 TFEsmPooler，用于 ESM 模型的池化操作。
从 transformers.models.bert.modeling_tf_bert.TFBertPooler 复制并修改为 ESM。

Parameters:
    config (EsmConfig): ESM 模型的配置对象，包含模型的各种参数。

Attributes:
    dense (Dense): 密集连接层，用于处理隐藏状态向量。
    config (EsmConfig): ESM 模型的配置对象。

Methods:
    call(hidden_states: tf.Tensor) -> tf.Tensor:
        对隐藏状态进行池化操作，只使用第一个 token 对应的隐藏状态。
    build(input_shape=None):
        构建层，初始化密集连接层。
"""

"""
ESM 模型的预训练模型基类 TFEsmPreTrainedModel。

An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
models.

Attributes:
    config_class (EsmConfig): 模型配置类，指定为 EsmConfig。
    base_model_prefix (str): 基础模型名称前缀，设为 "esm"。

Notes:
    该类提供了预训练模型的通用方法，如初始化权重、下载和加载预训练模型等。
"""

"""
ESM 模型的输入文档字符串，描述模型的基本信息和使用方法。

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

"""
ESM 模型的输入文档字符串，描述输入参数的详细信息和用法示例。
"""
        Args:
            input_ids (`tf.Tensor` of shape `({0})`):
                # 输入序列中的词汇索引。可以使用 `AutoTokenizer` 获取这些索引。参见 `PreTrainedTokenizer.encode` 和 `PreTrainedTokenizer.__call__`。
                # [什么是输入 ID？](../glossary#input-ids)
                Indices of input sequence tokens in the vocabulary.

            attention_mask (`tf.Tensor` of shape `({0})`, *optional*):
                # 遮罩，用于避免在填充令牌的索引上执行注意力操作。
                # 遮罩值选在 `[0, 1]`：
                # - 1 表示 **不遮罩** 的标记，
                # - 0 表示 **遮罩** 的标记。
                Mask to avoid performing attention on padding token indices.

            position_ids (`tf.Tensor` of shape `({0})`, *optional*):
                # 输入序列标记在位置嵌入中的位置索引。选在 `[0, config.max_position_embeddings - 1]` 范围内。
                # [什么是位置 ID？](../glossary#position-ids)
                Indices of positions of each input sequence tokens in the position embeddings.

            head_mask (`tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                # 用于置空自注意力模块中的选择头部的遮罩。
                # 遮罩值选在 `[0, 1]`：
                # - 1 表示头部 **不被遮罩**，
                # - 0 表示头部 **被遮罩**。
                Mask to nullify selected heads of the self-attention modules.

            inputs_embeds (`tf.Tensor` of shape `({0}, hidden_size)`, *optional*):
                # 可选，直接传递嵌入表示而不是 `input_ids`。如果想要更精确地控制如何将 `input_ids` 索引转换为相关联的向量，这很有用。
                # 比模型内部嵌入查找矩阵更有控制力。
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.

            output_attentions (`bool`, *optional*):
                # 是否返回所有注意力层的注意力张量。查看返回张量下的 `attentions` 获取更多细节。
                Whether or not to return the attentions tensors of all attention layers.

            output_hidden_states (`bool`, *optional*):
                # 是否返回所有层的隐藏状态。查看返回张量下的 `hidden_states` 获取更多细节。
                Whether or not to return the hidden states of all layers.

            return_dict (`bool`, *optional*):
                # 是否返回 [`~file_utils.ModelOutput`] 而不是普通的元组。
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare ESM Model transformer outputting raw hidden-states without any specific head on top.",
    ESM_START_DOCSTRING,
)
class TFEsmMainLayer(keras.layers.Layer):
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
        self.is_decoder = config.is_decoder  # 初始化解码器标志位

        self.embeddings = TFEsmEmbeddings(config, name="embeddings")  # 初始化嵌入层
        self.encoder = TFEsmEncoder(config, name="encoder")  # 初始化编码器
        self.pooler = TFEsmPooler(config, name="pooler") if add_pooling_layer else None  # 初始化池化层（如果需要）

        self.contact_head = TFEsmContactPredictionHead(
            in_features=self.config.num_hidden_layers * self.config.num_attention_heads, bias=True, name="contact_head"
        )  # 初始化接触预测头部

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)  # 构建嵌入层
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)  # 构建编码器
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build(None)  # 构建池化层
        if getattr(self, "contact_head", None) is not None:
            with tf.name_scope(self.contact_head.name):
                self.contact_head.build(None)  # 构建接触预测头部

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings  # 获取输入嵌入层的词嵌入

    def set_input_embeddings(self, value: tf.Variable):
        self.embeddings.word_embeddings.weight = value  # 设置输入嵌入层的权重
        self.embeddings.vocab_size = shape_list(value)[0]  # 设置词汇表大小

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError  # 剪枝头部的方法，未实现
    # 定义一个方法，用于调用模型，接收多种输入参数并返回预测结果
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
        training: bool = False,
    ):
        # 定义一个方法，用于预测模型的接触点（contacts）
        def predict_contacts(self, tokens, attention_mask):
            # 调用当前对象（self）的call方法，传入tokens和attention_mask作为输入，
            # 并设定return_dict和output_attentions参数为True，以获取注意力权重信息。
            attns = self(tokens, attention_mask=attention_mask, return_dict=True, output_attentions=True).attentions
            # 将得到的注意力权重列表堆叠成一个张量，维度顺序与原始模型一致
            attns = tf.stack(attns, axis=1)
            
            # 在原始模型中，对于填充标记的注意力权重被完全置零。
            # 这通常不会有太大影响，因为其他标记不会关注它们，
            # 但是在接触点预测任务中，它们作为输入需要被模仿。
            # 因此，这里要做的是将填充标记对应位置的注意力权重置零。
            attention_mask = tf.cast(attention_mask, attns.dtype)
            attns *= attention_mask[:, None, None, None]  # 扩展维度匹配注意力权重张量
            attns *= attention_mask[:, None, None, :, None]  # 扩展维度匹配注意力权重张量
            
            # 调用模型的contact_head方法，传入tokens和处理后的注意力权重attns作为参数，
            # 返回接触点预测的结果。
            return self.contact_head(tokens, attns)
# 给 TFEsmModel 类添加文档字符串，描述其作为没有特定顶部头的原始隐藏状态输出的 ES 模型转换器
@add_start_docstrings(
    "The bare ESM Model transformer outputting raw hidden-states without any specific head on top.",
    ESM_START_DOCSTRING,
)
class TFEsmModel(TFEsmPreTrainedModel):
    def __init__(self, config: EsmConfig, add_pooling_layer=True, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 初始化 ES 模型的主层，根据给定的配置和是否添加池化层
        self.esm = TFEsmMainLayer(config, add_pooling_layer=add_pooling_layer, name="esm")

    # 对 call 方法进行装饰，添加文档字符串以描述模型前向传播的输入
    @unpack_inputs
    @add_start_docstrings_to_model_forward(ESM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
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
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
        # 这里继续列出所有的参数，描述它们的作用和可选性
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
        return outputs

    def predict_contacts(self, tokens, attention_mask):
        # 调用模型的方法来预测接触点
        return self.esm.predict_contacts(tokens, attention_mask)

    def build(self, input_shape=None):
        if self.built:
            return
        # 标记模型已构建
        self.built = True
        if getattr(self, "esm", None) is not None:
            with tf.name_scope(self.esm.name):
                # 构建模型的子模块
                self.esm.build(None)
# 为模型添加文档字符串，描述其为带有顶部语言建模头的ESM模型
@add_start_docstrings("""ESM Model with a `language modeling` head on top.""", ESM_START_DOCSTRING)
class TFEsmForMaskedLM(TFEsmPreTrainedModel, TFMaskedLanguageModelingLoss):
    # 在加载过程中忽略缺失的关键字列表
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    # 在加载过程中忽略意外的关键字列表
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        # 如果配置指示为decoder，则发出警告
        if config.is_decoder:
            logger.warning(
                "If you want to use `EsmForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 初始化ESM主层，不添加池化层，并命名为"esm"
        self.esm = TFEsmMainLayer(config, add_pooling_layer=False, name="esm")
        # 初始化ESM语言建模头，并命名为"lm_head"
        self.lm_head = TFEsmLMHead(config, name="lm_head")
        
        # 如果需要绑定词嵌入
        if config.tie_word_embeddings:
            # 确保词嵌入已构建，以便进行绑定
            with tf.name_scope(os.path.join(self._name_scope(), "esm", "embeddings", "word_embeddings")):
                self.esm.embeddings.word_embeddings.build((None, None))
            # 将lm_head的解码器设置为与ESM的词嵌入权重相同
            self.lm_head.decoder = self.esm.embeddings.word_embeddings.weights[0]

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.lm_head.decoder

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    # 获取语言建模头
    def get_lm_head(self):
        return self.lm_head

    # 模型调用函数，解包输入并添加模型前向传播的文档字符串
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
        # 模型前向传播逻辑在此实现
    ) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        # 设置是否返回字典格式的输出，如果未提供，则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 ESM 模型进行前向传播
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
        # 获取模型输出的序列特征
        sequence_output = outputs[0]
        # 使用语言模型头部生成预测分数
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        # 如果提供了标签，则计算掩码语言建模损失
        if labels is not None:
            masked_lm_loss = self.hf_compute_loss(labels=labels, logits=prediction_scores)

        # 如果不要求返回字典格式的输出
        if not return_dict:
            # 构造输出元组，包含预测分数及可能的额外输出
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 返回 TFMaskedLMOutput 对象，包含损失、预测分数、隐藏状态和注意力权重
        return TFMaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def predict_contacts(self, tokens, attention_mask):
        # 调用 ESM 模型的预测接口，用于生成联系
        return self.esm.predict_contacts(tokens, attention_mask)

    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回
        if self.built:
            return
        # 设置模型为已构建状态
        self.built = True
        # 如果存在 ESM 模型，则在命名空间下构建它
        if getattr(self, "esm", None) is not None:
            with tf.name_scope(self.esm.name):
                self.esm.build(None)
        # 如果存在语言模型头部，则在命名空间下构建它
        if getattr(self, "lm_head", None) is not None:
            with tf.name_scope(self.lm_head.name):
                self.lm_head.build(None)
class TFEsmLMHead(keras.layers.Layer):
    """ESM Head for masked language modeling."""

    def __init__(self, config, name=None):
        super().__init__(name=name)
        # 创建一个全连接层，用于将输入特征映射到隐藏层大小的输出空间
        self.dense = keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # 添加一个 LayerNormalization 层，用于标准化输入向量
        self.layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")

        # 如果设置了 tie_word_embeddings，decoder 为 None；否则创建一个全连接层，用于解码到词汇表大小
        if config.tie_word_embeddings:
            self.decoder = None
        else:
            self.decoder = keras.layers.Dense(
                config.vocab_size,
                kernel_initializer=get_initializer(config.initializer_range),
                name="decoder",
                use_bias=False,
            )
        self.config = config

    def build(self, input_shape=None):
        # 分离偏置项以匹配 PT 模型，并允许权重交叉加载工作
        # 将其放在 build 方法中，以便在将其添加为权重时获得正确的名称
        if self.built:
            return
        self.built = True
        # 添加一个名为 "bias" 的权重，形状为 (config.vocab_size,)，并初始化为零，可训练
        self.bias = self.add_weight("bias", shape=(self.config.vocab_size,), initializer="zeros", trainable=True)
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # 构建 dense 层，输入形状为 [None, None, config.hidden_size]
                self.dense.build([None, None, self.config.hidden_size])
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                # 构建 layer_norm 层，输入形状为 [None, None, config.hidden_size]
                self.layer_norm.build([None, None, self.config.hidden_size])
        if getattr(self, "decoder", None) is not None and not self.config.tie_word_embeddings:
            with tf.name_scope(self.decoder.name):
                # 构建 decoder 层，输入形状为 [None, None, config.hidden_size]
                self.decoder.build([None, None, self.config.hidden_size])

    def get_bias(self):
        return {"bias": self.bias}

    def call(self, features):
        # 经过 dense 层映射特征
        x = self.dense(features)
        # 使用 gelu 激活函数
        x = tf.nn.gelu(x)
        # 使用 layer_norm 层标准化输出
        x = self.layer_norm(x)

        # 根据 tie_word_embeddings 决定如何将 x 投影回词汇表大小，同时加上偏置
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

    def __init__(self, config):
        super().__init__(config)
        # 设置分类或回归任务的标签数量
        self.num_labels = config.num_labels
        self.config = config

        # 创建 ESM 主层，不添加池化层，命名为 "esm"
        self.esm = TFEsmMainLayer(config, add_pooling_layer=False, name="esm")
        # 创建分类头部，命名为 "classifier"
        self.classifier = TFEsmClassificationHead(config, name="classifier")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ESM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 将当前函数用作代码示例的文档字符串，指定了一些参数和返回类型的信息
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
        # 设置 return_dict 变量，若未提供则使用 self.config.use_return_dict 中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 self.esm 方法，执行序列编码模型的前向传播
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
        # 从模型输出中获取序列输出
        sequence_output = outputs[0]
        # 将序列输出传递给分类器，生成分类任务的 logits
        logits = self.classifier(sequence_output)

        # 计算损失，如果 labels 不为 None，则使用 labels 和 logits 计算损失值
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果 return_dict 为 False，则构建输出元组
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则构建 TFSequenceClassifierOutput 对象作为输出
        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 构建模型，设置输入形状并初始化模型的各个组件
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在 self.esm，则在命名空间 self.esm.name 下构建它
        if getattr(self, "esm", None) is not None:
            with tf.name_scope(self.esm.name):
                self.esm.build(None)
        # 如果存在 self.classifier，则在命名空间 self.classifier.name 下构建它
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build(None)
@add_start_docstrings(
    """
    ESM Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    ESM_START_DOCSTRING,
)
class TFEsmForTokenClassification(TFEsmPreTrainedModel, TFTokenClassificationLoss):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        # 初始化时设置分类标签数量
        self.num_labels = config.num_labels

        # 创建 ESM 主模型层，不包含池化层
        self.esm = TFEsmMainLayer(config, add_pooling_layer=False, name="esm")
        # Dropout 层，用于防止过拟合
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        # 分类器层，将隐藏状态输出转化为分类预测
        self.classifier = keras.layers.Dense(config.num_labels, name="classifier")
        # 保存配置信息
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ESM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFTokenClassifierOutput,
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
    ) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 确定是否返回字典格式的输出结果
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 ESM 主模型进行前向传播
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

        # 在训练时使用 Dropout 层防止过拟合
        sequence_output = self.dropout(sequence_output, training=training)
        # 使用分类器层生成分类预测 logits
        logits = self.classifier(sequence_output)

        # 如果没有提供标签，则不计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 根据是否返回字典格式来组织输出
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TFTokenClassifierOutput 格式的结果
        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    # 如果模型已经构建好，直接返回，不做重复构建
    if self.built:
        return
    # 将模型标记为已构建状态
    self.built = True
    
    # 如果存在名为"esm"的属性，并且不为None，执行以下操作
    if getattr(self, "esm", None) is not None:
        # 在命名空间下以"esm"的名称构建模型
        with tf.name_scope(self.esm.name):
            # 调用esm对象的build方法，传入None作为输入形状
            self.esm.build(None)
    
    # 如果存在名为"classifier"的属性，并且不为None，执行以下操作
    if getattr(self, "classifier", None) is not None:
        # 在命名空间下以"classifier"的名称构建模型
        with tf.name_scope(self.classifier.name):
            # 调用classifier对象的build方法，传入[None, None, self.config.hidden_size]作为输入形状
            self.classifier.build([None, None, self.config.hidden_size])
class TFEsmClassificationHead(keras.layers.Layer):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, name=None):
        super().__init__(name=name)
        # 定义一个全连接层，用于生成隐藏层大小的输出，激活函数为tanh
        self.dense = keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        # 定义一个Dropout层，用于在训练时随机丢弃部分输入，以防止过拟合
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        # 定义一个全连接层，用于生成类别数目大小的输出，激活函数为线性（即无激活函数）
        self.out_proj = keras.layers.Dense(
            config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="linear",
            name="out_proj",
        )
        self.config = config

    def call(self, features, training=False):
        # 提取features中的第一个位置的向量（对应于<s> token，即[CLS]），作为输入x
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        # 在训练阶段使用dropout随机丢弃部分输入向量，防止过拟合
        x = self.dropout(x, training=training)
        # 将输入向量x通过全连接层dense进行线性变换，并应用tanh激活函数
        x = self.dense(x)
        # 再次在训练阶段使用dropout随机丢弃部分输出向量，防止过拟合
        x = self.dropout(x, training=training)
        # 将处理后的向量x通过全连接层out_proj进行线性变换，生成最终的分类输出
        x = self.out_proj(x)
        return x

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果dense层已定义，则建立其内部权重
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # 建立dense层的权重，输入形状为[None, None, hidden_size]
                self.dense.build([None, None, self.config.hidden_size])
        # 如果out_proj层已定义，则建立其内部权重
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                # 建立out_proj层的权重，输入形状为[None, None, hidden_size]
                self.out_proj.build([None, None, self.config.hidden_size])


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        input_ids: 输入的整数张量，表示输入的符号序列
        padding_idx: 表示填充符号的索引
        past_key_values_length: 过去键值长度，用于计算增量索引

    Returns:
        tf.Tensor: 包含位置ID的张量，替换非填充符号为其位置数字
    """
    # 创建一个掩码，标记出不是填充符号的位置
    mask = tf.cast(input_ids != padding_idx, tf.int64)
    # 计算每个位置的增量索引，跳过填充符号，位置编号从padding_idx+1开始
    incremental_indices = (tf.cumsum(mask, axis=1) + past_key_values_length) * mask
    # 将增量索引加上padding_idx，得到最终的位置ID张量
    return incremental_indices + padding_idx
```