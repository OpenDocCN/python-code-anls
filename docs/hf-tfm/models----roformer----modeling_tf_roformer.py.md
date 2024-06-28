# `.\models\roformer\modeling_tf_roformer.py`

```
# 导入所需模块和库
import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

# 从内部模块导入函数和类
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPooling,
    TFCausalLMOutput,
    TFMaskedLMOutput,
    TFMultipleChoiceModelOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFMaskedLanguageModelingLoss,
    TFModelInputType,
    TFMultipleChoiceLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFSequenceSummary,
    TFTokenClassificationLoss,
    get_initializer,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from .configuration_roformer import RoFormerConfig

# 获取 logger 对象用于记录日志
logger = logging.get_logger(__name__)

# 文档中使用的模型检查点和配置信息
_CHECKPOINT_FOR_DOC = "junnyu/roformer_chinese_base"
_CONFIG_FOR_DOC = "RoFormerConfig"

# RoFormer 的预训练模型归档列表
TF_ROFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "junnyu/roformer_chinese_small",
    "junnyu/roformer_chinese_base",
    "junnyu/roformer_chinese_char_small",
    "junnyu/roformer_chinese_char_base",
    "junnyu/roformer_small_discriminator",
    "junnyu/roformer_small_generator",
    # 更多 RoFormer 模型详见 https://huggingface.co/models?filter=roformer
]

class TFRoFormerSinusoidalPositionalEmbedding(keras.layers.Layer):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, **kwargs):
        super().__init__(**kwargs)

        # 初始化函数，确保嵌入维度是偶数，否则抛出异常
        if embedding_dim % 2 != 0:
            raise NotImplementedError(f"odd embedding_dim {embedding_dim} not supported")

        # 嵌入维度和位置数量属性
        self.embedding_dim = embedding_dim
        self.num_positions = num_positions
    def build(self, input_shape: tf.TensorShape):
        """
        Build shared token embedding layer Shared weights logic adapted from
        https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24
        """
        # 初始化权重矩阵
        weight = self._init_weight(self.num_positions, self.embedding_dim)

        # 添加权重作为层的一个参数
        self.weight = self.add_weight(
            name="embeddings",
            shape=[self.num_positions, self.embedding_dim],
        )
        # 将初始权重转换为与self.weight相同的数据类型
        weight = tf.cast(weight, dtype=self.weight.dtype)

        # 将初始权重赋值给self.weight
        self.weight.assign(weight)

        # 调用父类的build方法，传入输入形状
        super().build(input_shape)

    @staticmethod
    def _init_weight(n_pos: int, dim: int):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        # 创建位置编码矩阵，基于论文中的公式，使用sin和cos函数
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        table = np.zeros_like(position_enc)
        # 第一列全为零
        table[:, 0 : dim // 2] = np.sin(position_enc[:, 0::2])
        # 第二列开始使用cos函数
        table[:, dim // 2 :] = np.cos(position_enc[:, 1::2])
        # 转换为TensorFlow的张量
        table = tf.convert_to_tensor(table)
        # 停止梯度计算
        tf.stop_gradient(table)
        return table

    def call(self, input_shape: tf.TensorShape, past_key_values_length: int = 0):
        """Input is expected to be of size [bsz x seqlen]."""
        # 获取输入张量的形状，bsz为批量大小，seq_len为序列长度
        bsz, seq_len = input_shape[:2]

        # 生成位置索引，从past_key_values_length开始，到seq_len + past_key_values_length结束，步长为1
        positions = tf.range(past_key_values_length, seq_len + past_key_values_length, delta=1, name="range")
        # 使用gather操作从self.weight中获取指定位置的embedding向量
        return tf.gather(self.weight, positions)
class TFRoFormerEmbeddings(keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config: RoFormerConfig, **kwargs):
        super().__init__(**kwargs)
        
        # 初始化层，从配置中获取参数
        self.config = config
        self.embedding_size = config.embedding_size
        self.initializer_range = config.initializer_range
        
        # LayerNormalization 层，用于标准化输入数据
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        
        # Dropout 层，用于在训练过程中随机断开神经元连接，防止过拟合
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def build(self, input_shape=None):
        with tf.name_scope("word_embeddings"):
            # 添加词嵌入权重矩阵，形状为 [词汇量大小, 嵌入维度]
            self.weight = self.add_weight(
                name="weight",
                shape=[self.config.vocab_size, self.embedding_size],
                initializer=get_initializer(self.initializer_range),
            )

        with tf.name_scope("token_type_embeddings"):
            # 添加类型嵌入权重矩阵，形状为 [类型词汇量大小, 嵌入维度]
            self.token_type_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.config.type_vocab_size, self.embedding_size],
                initializer=get_initializer(self.initializer_range),
            )

        if self.built:
            return
        self.built = True
        
        # 如果 LayerNorm 层已存在，则构建该层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.embedding_size])

    def call(
        self,
        input_ids: tf.Tensor = None,
        token_type_ids: tf.Tensor = None,
        inputs_embeds: tf.Tensor = None,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Applies embedding based on inputs tensor.

        Args:
            input_ids (tf.Tensor): 输入的词汇 ID 张量
            token_type_ids (tf.Tensor): 输入的类型 ID 张量
            inputs_embeds (tf.Tensor): 输入的嵌入张量
            training (bool): 是否在训练模式中使用 Dropout

        Returns:
            final_embeddings (`tf.Tensor`): 输出的嵌入张量.
        """
        assert not (input_ids is None and inputs_embeds is None)

        if input_ids is not None:
            # 检查输入的词汇 ID 是否在有效范围内
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            # 根据词汇 ID 从权重矩阵中获取对应的词嵌入张量
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        input_shape = shape_list(inputs_embeds)[:-1]

        if token_type_ids is None:
            # 如果未提供类型 ID，则使用默认值 0
            token_type_ids = tf.fill(dims=input_shape, value=0)

        # 根据类型 ID 从类型嵌入权重矩阵中获取类型嵌入张量
        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)
        
        # 将词嵌入张量和类型嵌入张量相加得到最终的嵌入张量
        final_embeddings = inputs_embeds + token_type_embeds
        
        # 对最终嵌入张量进行 LayerNormalization 处理
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        
        # 在训练模式中，对最终嵌入张量应用 Dropout
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)

        return final_embeddings
    def __init__(self, config: RoFormerConfig, **kwargs):
        super().__init__(**kwargs)

        # 检查隐藏大小是否是注意力头数的整数倍，如果不是则引发错误
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        # 初始化变量
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        # 创建用于查询、键、值的全连接层，初始化器使用配置中的范围
        self.query = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        self.key = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        self.value = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )
        # Dropout 层，丢弃率为配置中的注意力概率丢弃概率
        self.dropout = keras.layers.Dropout(rate=config.attention_probs_dropout_prob)
        # 是否使用旋转值机制的标志
        self.rotary_value = config.rotary_value
        # 保存配置对象
        self.config = config

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # 将张量从 [batch_size, seq_length, all_head_size] 重塑为 [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

        # 转置张量，从 [batch_size, seq_length, num_attention_heads, attention_head_size] 到 [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        sinusoidal_pos: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        batch_size = shape_list(hidden_states)[0]  # 获取隐藏状态的批量大小
        mixed_query_layer = self.query(inputs=hidden_states)  # 使用查询函数处理隐藏状态
        mixed_key_layer = self.key(inputs=hidden_states)  # 使用键函数处理隐藏状态
        mixed_value_layer = self.value(inputs=hidden_states)  # 使用值函数处理隐藏状态
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)  # 调整查询层的形状以进行注意力计算
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)  # 调整键层的形状以进行注意力计算
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)  # 调整值层的形状以进行注意力计算

        if sinusoidal_pos is not None:
            if self.rotary_value:
                # 如果启用旋转值，应用旋转位置嵌入到查询、键和值层
                query_layer, key_layer, value_layer = self.apply_rotary_position_embeddings(
                    sinusoidal_pos, query_layer, key_layer, value_layer
                )
            else:
                # 否则，只应用旋转位置嵌入到查询和键层
                query_layer, key_layer = self.apply_rotary_position_embeddings(sinusoidal_pos, query_layer, key_layer)

        # 计算"查询"和"键"之间的点积，得到原始注意力分数
        # 结果形状为(batch size, num_heads, seq_len_q, seq_len_k)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, dk)  # 缩放注意力分数

        if attention_mask is not None:
            # 应用注意力掩码（在TFRoFormerModel的call()函数中预先计算）
            attention_scores = tf.add(attention_scores, attention_mask)

        # 将注意力分数归一化为概率
        attention_probs = stable_softmax(logits=attention_scores, axis=-1)

        # 对注意力概率进行dropout，这一步在原始Transformer论文中提到过
        attention_probs = self.dropout(inputs=attention_probs, training=training)

        # 如果需要，对注意力头进行掩码处理
        if head_mask is not None:
            attention_probs = tf.multiply(attention_probs, head_mask)

        # 计算注意力输出值
        attention_output = tf.matmul(attention_probs, value_layer)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])

        # 重新整形得到最终输出
        # 形状为(batch_size, seq_len_q, all_head_size)
        attention_output = tf.reshape(tensor=attention_output, shape=(batch_size, -1, self.all_head_size))

        # 构建输出元组，可能包含注意力输出和注意力概率，取决于output_attentions标志位
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)

        return outputs
    # 应用旋转位置嵌入到查询、键、值的层中
    def apply_rotary_position_embeddings(sinusoidal_pos, query_layer, key_layer, value_layer=None):
        # 将输入的正弦和余弦位置编码张量按照最后一个维度切分为两部分
        sin, cos = tf.split(sinusoidal_pos, num_or_size_splits=2, axis=-1)
        # 将每个位置的正弦值重复两次，构成新的正弦位置编码张量
        sin_pos = tf.repeat(sin, 2, axis=-1)
        # 将每个位置的余弦值重复两次，构成新的余弦位置编码张量
        cos_pos = tf.repeat(cos, 2, axis=-1)
        
        # 将查询层中每隔一个位置的向量进行旋转处理，形成旋转后的查询层
        rotate_half_query_layer = tf.stack([-query_layer[..., 1::2], query_layer[..., ::2]], axis=-1)
        rotate_half_query_layer = tf.reshape(rotate_half_query_layer, shape_list(query_layer))
        # 对查询层应用旋转位置嵌入公式
        query_layer = query_layer * cos_pos + rotate_half_query_layer * sin_pos
        
        # 将键层中每隔一个位置的向量进行旋转处理，形成旋转后的键层
        rotate_half_key_layer = tf.stack([-key_layer[..., 1::2], key_layer[..., ::2]], axis=-1)
        rotate_half_key_layer = tf.reshape(rotate_half_key_layer, shape_list(key_layer))
        # 对键层应用旋转位置嵌入公式
        key_layer = key_layer * cos_pos + rotate_half_key_layer * sin_pos
        
        # 如果值层不为空，则对值层中每隔一个位置的向量进行旋转处理，形成旋转后的值层
        if value_layer is not None:
            rotate_half_value_layer = tf.stack([-value_layer[..., 1::2], value_layer[..., ::2]], axis=-1)
            rotate_half_value_layer = tf.reshape(rotate_half_value_layer, shape_list(value_layer))
            # 对值层应用旋转位置嵌入公式
            value_layer = value_layer * cos_pos + rotate_half_value_layer * sin_pos
            # 返回旋转后的查询、键、值层
            return query_layer, key_layer, value_layer
        
        # 如果值层为空，则只返回旋转后的查询、键层
        return query_layer, key_layer

    # 构建模型的方法
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        
        # 标记模型已经构建
        self.built = True
        
        # 如果存在查询张量，则构建查询张量的形状
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.hidden_size])
        
        # 如果存在键张量，则构建键张量的形状
        if getattr(self, "key", None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.config.hidden_size])
        
        # 如果存在值张量，则构建值张量的形状
        if getattr(self, "value", None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.config.hidden_size])
# Copied from transformers.models.bert.modeling_tf_bert.TFBertSelfOutput with Bert->RoFormer
# 定义了一个名为 TFRoFormerSelfOutput 的自定义层，用于 RoFormer 模型的自我输出处理

class TFRoFormerSelfOutput(keras.layers.Layer):
    def __init__(self, config: RoFormerConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，输出单元数为 config.hidden_size，使用指定初始化器初始化权重矩阵
        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        
        # 创建一个 LayerNormalization 层，设置 epsilon 参数为 config.layer_norm_eps
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        
        # 创建一个 Dropout 层，设置 dropout 比率为 config.hidden_dropout_prob
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        
        # 保存配置参数
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 将输入 hidden_states 通过全连接层 self.dense 进行线性变换
        hidden_states = self.dense(inputs=hidden_states)
        
        # 在训练过程中，对输出 hidden_states 应用 dropout
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        
        # 将 dropout 处理后的 hidden_states 与输入 input_tensor 相加，并进行 LayerNormalization
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        # 返回处理后的 hidden_states
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        
        # 如果存在 self.dense 层，则使用输入形状构建该层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        
        # 如果存在 self.LayerNorm 层，则使用输入形状构建该层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


class TFRoFormerAttention(keras.layers.Layer):
    def __init__(self, config: RoFormerConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个 TFRoFormerSelfAttention 层，命名为 self_attention，用于处理 RoFormer 的自注意力机制
        self.self_attention = TFRoFormerSelfAttention(config, name="self")
        
        # 创建一个 TFRoFormerSelfOutput 层，命名为 dense_output，用于处理自我输出
        self.dense_output = TFRoFormerSelfOutput(config, name="output")

    def prune_heads(self, heads):
        # 未实现的方法，用于剪枝多头注意力机制的头部
        raise NotImplementedError

    def call(
        self,
        input_tensor: tf.Tensor,
        attention_mask: tf.Tensor,
        sinusoidal_pos: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        # 使用 self_attention 层处理输入的 input_tensor，获取自注意力机制的输出
        self_outputs = self.self_attention(
            hidden_states=input_tensor,
            attention_mask=attention_mask,
            sinusoidal_pos=sinusoidal_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            training=training,
        )
        
        # 使用 dense_output 层处理 self_attention 的输出，并与原始输入 input_tensor 相加，处理自我输出
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=input_tensor, training=training
        )
        
        # 将处理后的 attention_output 作为主要输出，如果需要输出 attentions，则将其附加在输出元组中
        outputs = (attention_output,) + self_outputs[1:]  # 如果需要输出 attentions，则添加它们

        # 返回最终输出元组
        return outputs
    # 定义神经网络层的构建方法，用于在给定输入形状的情况下构建网络层
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回，避免重复构建
        if self.built:
            return
        # 标记为已构建
        self.built = True
        
        # 如果存在 self_attention 属性，则构建 self_attention
        if getattr(self, "self_attention", None) is not None:
            # 在命名空间下构建 self_attention
            with tf.name_scope(self.self_attention.name):
                # 调用 self_attention 的 build 方法，传入 None 作为输入形状
                self.self_attention.build(None)
        
        # 如果存在 dense_output 属性，则构建 dense_output
        if getattr(self, "dense_output", None) is not None:
            # 在命名空间下构建 dense_output
            with tf.name_scope(self.dense_output.name):
                # 调用 dense_output 的 build 方法，传入 None 作为输入形状
                self.dense_output.build(None)
# 从 transformers.models.bert.modeling_tf_bert.TFBertIntermediate 复制而来，将 Bert 替换为 RoFormer
class TFRoFormerIntermediate(keras.layers.Layer):
    def __init__(self, config: RoFormerConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，用于处理中间层的输出
        self.dense = keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # 根据配置获取中间激活函数，可以是字符串或者函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 将输入的隐藏状态经过全连接层处理
        hidden_states = self.dense(inputs=hidden_states)
        # 使用中间激活函数处理全连接层的输出
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经建立，则不做任何操作
        if getattr(self, "dense", None) is not None:
            # 如果存在全连接层，建立其结构
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# 从 transformers.models.bert.modeling_tf_bert.TFBertOutput 复制而来，将 Bert 替换为 RoFormer
class TFRoFormerOutput(keras.layers.Layer):
    def __init__(self, config: RoFormerConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，用于处理输出层的输出
        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个层归一化层，用于处理输出的归一化
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建一个 dropout 层，用于输出的随机失活
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 将输入的隐藏状态经过全连接层处理
        hidden_states = self.dense(inputs=hidden_states)
        # 使用 dropout 层对全连接层输出进行处理
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 使用层归一化层对全连接层输出和输入进行处理
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经建立，则不做任何操作
        if getattr(self, "dense", None) is not None:
            # 如果存在全连接层，建立其结构
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])
        if getattr(self, "LayerNorm", None) is not None:
            # 如果存在层归一化层，建立其结构
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# 定义 RoFormer 层，包含注意力层、中间层和输出层
class TFRoFormerLayer(keras.layers.Layer):
    def __init__(self, config: RoFormerConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建 RoFormer 注意力层
        self.attention = TFRoFormerAttention(config, name="attention")
        # 创建 RoFormer 中间层
        self.intermediate = TFRoFormerIntermediate(config, name="intermediate")
        # 创建 RoFormer 输出层
        self.roformer_output = TFRoFormerOutput(config, name="output")
    # 定义一个方法，用于 RoFormer 模型的前向传播计算
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        sinusoidal_pos: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        # 调用注意力层的计算，得到注意力层的输出
        attention_outputs = self.attention(
            input_tensor=hidden_states,
            attention_mask=attention_mask,
            sinusoidal_pos=sinusoidal_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            training=training,
        )
        # 取注意力层输出的第一个元素作为注意力输出
        attention_output = attention_outputs[0]
        # 经过中间层处理，得到中间层的输出
        intermediate_output = self.intermediate(hidden_states=attention_output)
        # 经过 RoFormer 输出层的处理，得到最终层的输出
        layer_output = self.roformer_output(
            hidden_states=intermediate_output, input_tensor=attention_output, training=training
        )
        # 如果有需要，将注意力输出一并返回
        outputs = (layer_output,) + attention_outputs[1:]  # 如果有输出注意力信息，则添加进去

        return outputs

    # 构建方法，用于在 TensorFlow 中构建网络层
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 设置为已构建状态
        self.built = True
        # 如果存在 attention 层，则在 TensorFlow 的名称空间下构建
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        # 如果存在 intermediate 层，则在 TensorFlow 的名称空间下构建
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        # 如果存在 roformer_output 层，则在 TensorFlow 的名称空间下构建
        if getattr(self, "roformer_output", None) is not None:
            with tf.name_scope(self.roformer_output.name):
                self.roformer_output.build(None)
# 定义 TFRoFormerEncoder 类，继承自 keras.layers.Layer
class TFRoFormerEncoder(keras.layers.Layer):
    
    # 初始化方法，接受 RoFormerConfig 对象和其他关键字参数
    def __init__(self, config: RoFormerConfig, **kwargs):
        super().__init__(**kwargs)
        
        # 创建 embed_positions 属性，使用 TFRoFormerSinusoidalPositionalEmbedding 类
        self.embed_positions = TFRoFormerSinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            config.hidden_size // config.num_attention_heads,
            name="embed_positions",
        )
        
        # 创建 layer 属性，是 TFRoFormerLayer 对象组成的列表
        self.layer = [TFRoFormerLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

    # call 方法，定义了层的正向传播逻辑
    def call(
        self,
        hidden_states: tf.Tensor,            # 输入的隐藏状态张量
        attention_mask: tf.Tensor,           # 注意力掩码张量
        head_mask: tf.Tensor,                # 头部掩码张量
        output_attentions: bool,             # 是否输出注意力权重
        output_hidden_states: bool,          # 是否输出隐藏状态
        return_dict: bool,                   # 是否返回字典
        training: bool = False,              # 是否处于训练模式
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        # 如果输出隐藏状态，初始化空元组 all_hidden_states，否则设为 None
        all_hidden_states = () if output_hidden_states else None
        # 如果输出注意力权重，初始化空元组 all_attentions，否则设为 None
        all_attentions = () if output_attentions else None

        # 生成正弦位置编码，形状为 [1, 1, sequence_length, embed_size_per_head]
        sinusoidal_pos = self.embed_positions(shape_list(hidden_states)[:-1])[None, None, :, :]

        # 遍历每个层模块
        for i, layer_module in enumerate(self.layer):
            # 如果输出隐藏状态，将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 调用层模块的正向传播方法，计算层的输出
            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                sinusoidal_pos=sinusoidal_pos,
                head_mask=head_mask[i],
                output_attentions=output_attentions,
                training=training,
            )
            # 更新隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]

            # 如果输出注意力权重，将当前层的注意力权重添加到 all_attentions 中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # 如果输出隐藏状态，将最后一个隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果 return_dict 为 False，则返回不为 None 的元组项
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        # 如果 return_dict 为 True，则返回 TFBaseModelOutput 对象
        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )

    # build 方法，用于构建层，初始化 embed_positions 和 layer
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        
        # 如果 embed_positions 存在，则构建它
        if getattr(self, "embed_positions", None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)
        
        # 遍历每个层，构建每个层
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)
    # 初始化方法，用于创建一个新的 RoFormer 模型实例
    def __init__(self, config: RoFormerConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 创建一个全连接层，用于将输入向量转换到指定维度
        self.dense = keras.layers.Dense(
            units=config.embedding_size,                      # 设置全连接层的输出维度
            kernel_initializer=get_initializer(config.initializer_range),  # 设置权重初始化器
            name="dense",                                     # 设置层名称
        )

        # 根据配置参数选择或者创建激活函数转换器
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_tf_activation(config.hidden_act)  # 根据字符串获取激活函数
        else:
            self.transform_act_fn = config.hidden_act  # 直接使用配置中的激活函数

        # 创建 LayerNormalization 层，用于归一化输入数据
        self.LayerNorm = keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps,  # 设置归一化层的 epsilon 参数
            name="LayerNorm"                 # 设置层名称
        )
        
        # 保存配置参数供模型使用
        self.config = config

    # 模型调用方法，用于定义模型的前向传播逻辑
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 输入数据通过全连接层进行线性变换
        hidden_states = self.dense(inputs=hidden_states)
        # 应用激活函数转换非线性特征
        hidden_states = self.transform_act_fn(hidden_states)
        # 输入数据通过归一化层进行归一化处理
        hidden_states = self.LayerNorm(inputs=hidden_states)

        # 返回处理后的数据作为模型输出
        return hidden_states

    # 构建方法，用于构建模型的各层并初始化权重
    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回
        if self.built:
            return
        self.built = True
        
        # 如果存在全连接层，则构建全连接层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])  # 构建全连接层的权重

        # 如果存在归一化层，则构建归一化层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.embedding_size])  # 构建归一化层的权重
class TFRoFormerLMPredictionHead(keras.layers.Layer):
    def __init__(self, config: RoFormerConfig, input_embeddings: keras.layers.Layer, **kwargs):
        super().__init__(**kwargs)

        self.config = config  # 存储 RoFormer 的配置信息
        self.embedding_size = config.embedding_size  # 提取配置中的嵌入大小

        self.transform = TFRoFormerPredictionHeadTransform(config, name="transform")  # 初始化预测头的转换层

        # 输出权重与输入嵌入相同，但每个标记都有一个只输出的偏置项
        self.input_embeddings = input_embeddings  # 存储输入的嵌入层对象

    def build(self, input_shape=None):
        # 添加一个形状为 (vocab_size,) 的可训练偏置项，初始化为零向量
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")

        if self.built:
            return
        self.built = True
        if getattr(self, "transform", None) is not None:
            with tf.name_scope(self.transform.name):
                self.transform.build(None)  # 构建转换层的内部结构

    def get_output_embeddings(self) -> keras.layers.Layer:
        return self.input_embeddings  # 返回当前的输入嵌入层对象

    def set_output_embeddings(self, value: tf.Variable):
        self.input_embeddings.weight = value  # 设置输入嵌入的权重为给定值
        self.input_embeddings.vocab_size = shape_list(value)[0]  # 更新嵌入的词汇表大小

    def get_bias(self) -> Dict[str, tf.Variable]:
        return {"bias": self.bias}  # 返回当前的偏置项作为字典

    def set_bias(self, value: tf.Variable):
        self.bias = value["bias"]  # 设置偏置项为给定的值中的 "bias" 键
        self.config.vocab_size = shape_list(value["bias"])[0]  # 更新配置中的词汇表大小信息

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.transform(hidden_states=hidden_states)  # 应用预测头的转换层
        seq_length = shape_list(hidden_states)[1]  # 获取隐藏状态张量的序列长度
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.embedding_size])  # 将隐藏状态重塑为二维张量
        hidden_states = tf.matmul(a=hidden_states, b=self.input_embeddings.weight, transpose_b=True)  # 执行矩阵乘法
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])  # 重塑为三维张量
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)  # 添加偏置项到张量

        return hidden_states


# 从 transformers.models.bert.modeling_tf_bert.TFBertMLMHead 复制并修改为 RoFormer
class TFRoFormerMLMHead(keras.layers.Layer):
    def __init__(self, config: RoFormerConfig, input_embeddings: keras.layers.Layer, **kwargs):
        super().__init__(**kwargs)

        self.predictions = TFRoFormerLMPredictionHead(config, input_embeddings, name="predictions")  # 初始化预测头

    def call(self, sequence_output: tf.Tensor) -> tf.Tensor:
        prediction_scores = self.predictions(hidden_states=sequence_output)  # 执行预测头的前向传播

        return prediction_scores

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "predictions", None) is not None:
            with tf.name_scope(self.predictions.name):
                self.predictions.build(None)  # 构建预测头的内部结构


@keras_serializable
class TFRoFormerMainLayer(keras.layers.Layer):
    config_class = RoFormerConfig  # 设置 RoFormer 的配置类
    def __init__(self, config: RoFormerConfig, add_pooling_layer: bool = True, **kwargs):
        super().__init__(**kwargs)  # 调用父类的初始化方法

        self.config = config  # 保存配置对象到实例变量

        self.embeddings = TFRoFormerEmbeddings(config, name="embeddings")  # 创建 RoFormer 的 embeddings 层对象
        if config.embedding_size != config.hidden_size:
            self.embeddings_project = keras.layers.Dense(config.hidden_size, name="embeddings_project")  # 如果 embedding_size 不等于 hidden_size，则创建 Dense 层

        self.encoder = TFRoFormerEncoder(config, name="encoder")  # 创建 RoFormer 的 encoder 层对象

    def get_input_embeddings(self) -> keras.layers.Layer:
        return self.embeddings  # 返回 embeddings 层对象作为输入 embeddings

    def set_input_embeddings(self, value: tf.Variable):
        self.embeddings.weight = value  # 设置 embeddings 层的权重为给定的 value
        self.embeddings.vocab_size = shape_list(value)[0]  # 设置 embeddings 层的词汇量大小为 value 的第一个维度大小

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError  # 抛出未实现异常，表明子类应该实现这个方法

    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ):
        """
        RoFormer 模型的前向传播方法，接收多个输入参数，并返回相应的输出。

        这里的装饰器 @unpack_inputs 用于解包输入参数，详见其定义。
        """
        # 具体的前向传播逻辑在这里实现，但代码中没有具体展示

    def build(self, input_shape=None):
        if self.built:
            return  # 如果已经构建过，直接返回

        self.built = True  # 设置标志位表示模型已构建

        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)  # 构建 embeddings 层

        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)  # 构建 encoder 层

        if getattr(self, "embeddings_project", None) is not None:
            with tf.name_scope(self.embeddings_project.name):
                self.embeddings_project.build([None, None, self.config.embedding_size])  # 构建 embeddings_project 层
# 导入 `TFPreTrainedModel` 的子类 `TFRoFormerPreTrainedModel`
class TFRoFormerPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 配置类是 `RoFormerConfig`
    config_class = RoFormerConfig
    # 基础模型前缀是 "roformer"
    base_model_prefix = "roformer"


# RoFormer 模型文档字符串的起始部分
ROFORMER_START_DOCSTRING = r"""

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
        config ([`RoFormerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# RoFormer 模型输入文档字符串的起始部分
ROFORMER_INPUTS_DOCSTRING = r"""
"""


# 添加文档字符串说明到 `TFRoFormerModel` 类
@add_start_docstrings(
    "The bare RoFormer Model transformer outputing raw hidden-states without any specific head on top.",
    ROFORMER_START_DOCSTRING,
)
class TFRoFormerModel(TFRoFormerPreTrainedModel):
    # 初始化函数，接受一个RoFormer配置对象和其他输入参数，并调用父类的初始化方法
    def __init__(self, config: RoFormerConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 创建一个TFRoFormerMainLayer对象，命名为"roformer"
        self.roformer = TFRoFormerMainLayer(config, name="roformer")

    # 装饰器：解压输入参数，将模型前向传播的文档字符串添加到方法上
    # 添加代码示例的文档字符串，指定检查点、输出类型和配置类
    @unpack_inputs
    @add_start_docstrings_to_model_forward(ROFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义模型的前向传播方法
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        # 调用self.roformer对象进行前向传播，传递所有输入参数
        outputs = self.roformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 返回模型的输出
        return outputs

    # 构建函数，用于构建模型，如果已经构建则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 检查self.roformer属性是否存在，并在名称作用域内构建self.roformer对象
        if getattr(self, "roformer", None) is not None:
            with tf.name_scope(self.roformer.name):
                self.roformer.build(None)
@add_start_docstrings("""RoFormer Model with a `language modeling` head on top.""", ROFORMER_START_DOCSTRING)
# 使用给定的文档字符串为 RoFormer 模型添加头部语言建模功能的说明文档

class TFRoFormerForMaskedLM(TFRoFormerPreTrainedModel, TFMaskedLanguageModelingLoss):
    # TFRoFormerForMaskedLM 类继承自 TFRoFormerPreTrainedModel 和 TFMaskedLanguageModelingLoss

    def __init__(self, config: RoFormerConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 调用父类的构造函数，初始化模型配置和其他输入参数

        if config.is_decoder:
            logger.warning(
                "If you want to use `TFRoFormerForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )
            # 如果配置要求是解码器，则发出警告，建议在使用时确保 config.is_decoder=False，以便实现双向自注意力

        self.roformer = TFRoFormerMainLayer(config, name="roformer")
        # 初始化 RoFormer 的主层，使用给定的配置和名称"roformer"

        self.mlm = TFRoFormerMLMHead(config, input_embeddings=self.roformer.embeddings, name="mlm___cls")
        # 初始化 RoFormer 的 MLM 头部，使用给定的配置、输入嵌入和名称"mlm___cls"

    def get_lm_head(self) -> keras.layers.Layer:
        return self.mlm.predictions
        # 返回 MLM 头部的预测结果作为语言建模的输出层

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ROFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
        # 定义模型的调用方法，接受一系列输入参数，执行前向传播操作
    ) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        # 调用 RoFormer 模型进行推理，返回结果包括 MLM 相关的输出和额外信息
        outputs = self.roformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 获取 RoFormer 的输出序列
        sequence_output = outputs[0]
        # 使用 MLM 层对序列进行预测得分计算
        prediction_scores = self.mlm(sequence_output=sequence_output, training=training)
        # 如果有标签数据，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=prediction_scores)

        # 如果不要求返回字典，则输出结果包括预测分数和额外的输出状态
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TFMaskedLMOutput 对象，包括损失、预测分数、隐藏状态和注意力信息
        return TFMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建过网络结构，则直接返回
        if self.built:
            return
        # 标记该模型已经构建
        self.built = True
        # 如果 RoFormer 模型存在，则建立 RoFormer
        if getattr(self, "roformer", None) is not None:
            with tf.name_scope(self.roformer.name):
                self.roformer.build(None)
        # 如果 MLM 模型存在，则建立 MLM
        if getattr(self, "mlm", None) is not None:
            with tf.name_scope(self.mlm.name):
                self.mlm.build(None)
@add_start_docstrings(
    """RoFormer Model with a `language modeling` head on top for CLM fine-tuning.""", ROFORMER_START_DOCSTRING
)
class TFRoFormerForCausalLM(TFRoFormerPreTrainedModel, TFCausalLanguageModelingLoss):
    def __init__(self, config: RoFormerConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        if not config.is_decoder:
            # 如果要单独使用 `TFRoFormerForCausalLM`，需要设置 `is_decoder=True`
            logger.warning("If you want to use `TFRoFormerForCausalLM` as a standalone, add `is_decoder=True.`")

        # 初始化 RoFormer 主层
        self.roformer = TFRoFormerMainLayer(config, name="roformer")
        # 初始化 RoFormer 的 MLM 头部
        self.mlm = TFRoFormerMLMHead(config, input_embeddings=self.roformer.embeddings, name="mlm___cls")

    def get_lm_head(self) -> keras.layers.Layer:
        # 返回 MLM 头部的预测层
        return self.mlm.predictions

    @unpack_inputs
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFCausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[TFCausalLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the cross entropy classification loss. Indices should be in `[0, ...,
            config.vocab_size - 1]`.
        """
        # 调用 RoFormer 主层进行前向传播
        outputs = self.roformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 从 RoFormer 输出中获取序列输出
        sequence_output = outputs[0]
        # 使用 MLM 头部对序列输出进行预测
        logits = self.mlm(sequence_output=sequence_output, training=training)
        loss = None

        if labels is not None:
            # 将标签向左移动一个位置并去掉最后一个预测标记
            shifted_logits = logits[:, :-1]
            labels = labels[:, 1:]
            # 计算交叉熵损失
            loss = self.hf_compute_loss(labels=labels, logits=shifted_logits)

        if not return_dict:
            # 如果不要求返回字典，则返回元组形式的输出
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TFCausalLMOutput 格式的输出
        return TFCausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    # 定义一个方法用于构建模型，在没有指定输入形状的情况下
    def build(self, input_shape=None):
        # 如果模型已经构建完成，则直接返回，避免重复构建
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        
        # 检查是否存在名为 "roformer" 的属性，如果存在则构建相关部分
        if getattr(self, "roformer", None) is not None:
            # 在 TensorFlow 中使用命名空间来管理作用域，这里创建 roformer 的命名空间
            with tf.name_scope(self.roformer.name):
                # 调用 roformer 对象的 build 方法，参数为 None 表示使用默认输入形状
                self.roformer.build(None)
        
        # 检查是否存在名为 "mlm" 的属性，如果存在则构建相关部分
        if getattr(self, "mlm", None) is not None:
            # 在 TensorFlow 中使用命名空间来管理作用域，这里创建 mlm 的命名空间
            with tf.name_scope(self.mlm.name):
                # 调用 mlm 对象的 build 方法，参数为 None 表示使用默认输入形状
                self.mlm.build(None)
class TFRoFormerClassificationHead(keras.layers.Layer):
    """Head for sentence-level classification tasks."""

    def __init__(self, config: RoFormerConfig, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

        # 创建一个全连接层，用于将输入特征映射到隐藏层的维度上
        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 添加一个dropout层，用于在训练过程中随机丢弃部分神经元，防止过拟合
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        # 创建输出层，将隐藏层映射到标签数量的维度上
        self.out_proj = keras.layers.Dense(
            units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="out_proj"
        )

        # 根据配置初始化分类器的激活函数
        if isinstance(config.hidden_act, str):
            self.classifier_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.classifier_act_fn = config.hidden_act
        self.config = config

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 从输入的隐藏状态中仅保留第一个特征向量，通常代表[CLS] token
        hidden_states = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS])
        # 应用dropout操作到隐藏状态，根据training参数决定是否启用dropout
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 将dropout后的隐藏状态输入到全连接层中进行线性变换
        hidden_states = self.dense(inputs=hidden_states)
        # 应用分类器的激活函数到全连接层输出的隐藏状态上
        hidden_states = self.classifier_act_fn(hidden_states)
        # 再次应用dropout操作到激活函数后的隐藏状态上
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 将dropout后的隐藏状态输入到输出层中进行线性变换，得到最终的分类输出
        hidden_states = self.out_proj(hidden_states)

        # 返回最终的分类输出张量
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果dense层已经初始化，则构建dense层的计算图
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果out_proj层已经初始化，则构建out_proj层的计算图
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.config.hidden_size])


@add_start_docstrings(
    """
    RoFormer Model transformer with a sequence classification/regression head on top e.g., for GLUE tasks.
    """,
    ROFORMER_START_DOCSTRING,
)
# 创建用于序列分类任务的RoFormer模型，继承自TFRoFormerPreTrainedModel和TFSequenceClassificationLoss类
class TFRoFormerForSequenceClassification(TFRoFormerPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: RoFormerConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 设置分类任务的类别数目
        self.num_labels = config.num_labels

        # 初始化RoFormer主层和分类头部
        self.roformer = TFRoFormerMainLayer(config, name="roformer")
        self.classifier = TFRoFormerClassificationHead(config, name="classifier")

    # 以下是装饰器和注释，用于说明模型的输入和输出格式，以及示例代码等
    @unpack_inputs
    @add_start_docstrings_to_model_forward(ROFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 定义输入的token ids，类型为TFModelInputType或None，默认为None
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 定义注意力掩码，类型为np.ndarray或tf.Tensor或None，默认为None
        token_type_ids: np.ndarray | tf.Tensor | None = None,  # 定义token类型 ids，类型为np.ndarray或tf.Tensor或None，默认为None
        head_mask: np.ndarray | tf.Tensor | None = None,  # 定义头部掩码，类型为np.ndarray或tf.Tensor或None，默认为None
        inputs_embeds: np.ndarray | tf.Tensor | None = None,  # 定义嵌入输入，类型为np.ndarray或tf.Tensor或None，默认为None
        output_attentions: Optional[bool] = None,  # 是否输出注意力信息，类型为可选的布尔值，默认为None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，类型为可选的布尔值，默认为None
        return_dict: Optional[bool] = None,  # 是否返回字典形式的输出，类型为可选的布尔值，默认为None
        labels: np.ndarray | tf.Tensor | None = None,  # 定义标签，类型为np.ndarray或tf.Tensor或None，默认为None
        training: Optional[bool] = False,  # 是否处于训练模式，类型为可选的布尔值，默认为False
    ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 使用 RoFormer 模型进行前向传播
        outputs = self.roformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 使用分类器对 RoFormer 的隐藏状态进行分类
        logits = self.classifier(hidden_states=outputs[0], training=training)
        # 计算损失，如果标签为None则损失为None
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        # 如果不返回字典形式的输出，则按元组形式返回结果
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TFSequenceClassifierOutput 对象，包含损失、logits、隐藏状态和注意力信息
        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回
        if self.built:
            return
        self.built = True
        # 构建 RoFormer 模型
        if getattr(self, "roformer", None) is not None:
            with tf.name_scope(self.roformer.name):
                self.roformer.build(None)
        # 构建分类器
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build(None)
    """
    RoFormer Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """
    # 继承自TFRoFormerPreTrainedModel和TFMultipleChoiceLoss，实现RoFormer模型用于多项选择分类任务
    @add_start_docstrings(
        ROFORMER_START_DOCSTRING,
    )
    class TFRoFormerForMultipleChoice(TFRoFormerPreTrainedModel, TFMultipleChoiceLoss):
        def __init__(self, config: RoFormerConfig, *inputs, **kwargs):
            super().__init__(config, *inputs, **kwargs)
    
            # 初始化RoFormer主层
            self.roformer = TFRoFormerMainLayer(config, name="roformer")
            # 序列摘要层，用于生成序列摘要特征
            self.sequence_summary = TFSequenceSummary(config, config.initializer_range, name="sequence_summary")
            # 分类器，使用Dense层实现，用于多项选择任务的分类
            self.classifier = keras.layers.Dense(
                units=1, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
            )
            self.config = config
    
        # 根据输入参数解包输入，添加模型正向传播的文档字符串
        @unpack_inputs
        @add_start_docstrings_to_model_forward(
            ROFORMER_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
        )
        @add_code_sample_docstrings(
            checkpoint=_CHECKPOINT_FOR_DOC,
            output_type=TFMultipleChoiceModelOutput,
            config_class=_CONFIG_FOR_DOC,
        )
        def call(
            self,
            input_ids: TFModelInputType | None = None,
            attention_mask: np.ndarray | tf.Tensor | None = None,
            token_type_ids: np.ndarray | tf.Tensor | None = None,
            head_mask: np.ndarray | tf.Tensor | None = None,
            inputs_embeds: np.ndarray | tf.Tensor | None = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            labels: np.ndarray | tf.Tensor | None = None,
            training: Optional[bool] = False,
    ) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)
        """
        # 如果 `input_ids` 不为空，则确定 `num_choices` 和 `seq_length`
        if input_ids is not None:
            num_choices = shape_list(input_ids)[1]  # 获取 `input_ids` 的第二维度大小
            seq_length = shape_list(input_ids)[2]   # 获取 `input_ids` 的第三维度大小
        else:
            num_choices = shape_list(inputs_embeds)[1]  # 获取 `inputs_embeds` 的第二维度大小
            seq_length = shape_list(inputs_embeds)[2]   # 获取 `inputs_embeds` 的第三维度大小

        # 将输入张量展平为二维，如果对应输入张量不为空
        flat_input_ids = tf.reshape(tensor=input_ids, shape=(-1, seq_length)) if input_ids is not None else None
        flat_attention_mask = (
            tf.reshape(tensor=attention_mask, shape=(-1, seq_length)) if attention_mask is not None else None
        )
        flat_token_type_ids = (
            tf.reshape(tensor=token_type_ids, shape=(-1, seq_length)) if token_type_ids is not None else None
        )
        flat_inputs_embeds = (
            tf.reshape(tensor=inputs_embeds, shape=(-1, seq_length, shape_list(inputs_embeds)[3]))
            if inputs_embeds is not None
            else None
        )

        # 调用 `roformer` 模型进行前向传播
        outputs = self.roformer(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 对输出进行序列摘要处理
        logits = self.sequence_summary(inputs=outputs[0], training=training)
        logits = self.classifier(inputs=logits)  # 将序列摘要后的结果送入分类器

        # 将 logits 重新整形为二维形状
        reshaped_logits = tf.reshape(tensor=logits, shape=(-1, num_choices))

        # 如果提供了标签 `labels`，计算损失
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=reshaped_logits)

        # 如果 `return_dict` 为 False，则返回输出元组
        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果 `return_dict` 为 True，则返回多项选择模型的输出对象
        return TFMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        
        # 设置已构建标志为 True
        self.built = True

        # 如果 `roformer` 模型存在，则构建其网络结构
        if getattr(self, "roformer", None) is not None:
            with tf.name_scope(self.roformer.name):
                self.roformer.build(None)

        # 如果 `sequence_summary` 模型存在，则构建其网络结构
        if getattr(self, "sequence_summary", None) is not None:
            with tf.name_scope(self.sequence_summary.name):
                self.sequence_summary.build(None)

        # 如果 `classifier` 模型存在，则构建其网络结构
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
# 定义 TFRoFormerForTokenClassification 类，用于在 RoFormer 模型的基础上增加一个标记分类头部，例如用于命名实体识别（NER）任务
@add_start_docstrings(
    """
    RoFormer Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    ROFORMER_START_DOCSTRING,
)
class TFRoFormerForTokenClassification(TFRoFormerPreTrainedModel, TFTokenClassificationLoss):
    def __init__(self, config: RoFormerConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 初始化类别数量
        self.num_labels = config.num_labels

        # 初始化 RoFormer 主层
        self.roformer = TFRoFormerMainLayer(config, name="roformer")
        # 初始化 Dropout 层
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        # 初始化分类器 Dense 层，用于分类器的线性变换
        self.classifier = keras.layers.Dense(
            units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        # 存储模型配置
        self.config = config

    # 定义模型的前向传播方法，接受多个输入参数，并返回 TFTokenClassifierOutput 类型的输出
    @unpack_inputs
    @add_start_docstrings_to_model_forward(ROFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
        # 更多参数用于控制模型行为
    ) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        定义方法的返回类型，可以是 TFTokenClassifierOutput 或者包含 tf.Tensor 的元组。
        如果方法没有返回值，应该返回 None。
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            用于计算标记分类损失的标签。索引应在 `[0, ..., config.num_labels - 1]` 范围内。
        """
        # 调用 RoFormer 模型进行前向传播
        outputs = self.roformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 获取模型输出的序列输出
        sequence_output = outputs[0]
        # 对序列输出应用 dropout 操作，用于防止过拟合
        sequence_output = self.dropout(inputs=sequence_output, training=training)
        # 将处理后的序列输出送入分类器中得到 logits
        logits = self.classifier(inputs=sequence_output)
        # 计算损失值，如果没有提供标签，则损失值为 None
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        # 如果 return_dict 为 False，则返回元组形式的输出
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则返回 TFTokenClassifierOutput 对象
        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建过网络，则直接返回
        if self.built:
            return
        # 标记网络已经构建
        self.built = True
        # 如果存在 RoFormer 模型，则构建 RoFormer
        if getattr(self, "roformer", None) is not None:
            with tf.name_scope(self.roformer.name):
                self.roformer.build(None)
        # 如果存在分类器模型，则构建分类器
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
# 使用装饰器添加模型的起始文档字符串，描述了 RoFormer 模型及其用途，特别是在抽取式问答任务（如 SQuAD）中的应用
@add_start_docstrings(
    """
    RoFormer Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    ROFORMER_START_DOCSTRING,  # 引用了 RoFormer 模型的起始文档字符串
)
class TFRoFormerForQuestionAnswering(TFRoFormerPreTrainedModel, TFQuestionAnsweringLoss):
    def __init__(self, config: RoFormerConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels  # 从配置中获取标签数目

        self.roformer = TFRoFormerMainLayer(config, name="roformer")  # 初始化 RoFormer 主层
        self.qa_outputs = keras.layers.Dense(
            units=config.num_labels,  # 输出单元数为配置中的标签数目
            kernel_initializer=get_initializer(config.initializer_range),  # 使用配置中的初始化范围初始化权重
            name="qa_outputs"  # 输出层的名称为 "qa_outputs"
        )
        self.config = config  # 保存配置参数

    # 使用装饰器为 call 方法添加输入参数的起始文档字符串，描述了模型的输入格式及其用途
    @unpack_inputs
    @add_start_docstrings_to_model_forward(ROFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,  # 提供了示例代码的检查点位置
        output_type=TFQuestionAnsweringModelOutput,  # 输出类型为 TFQuestionAnsweringModelOutput
        config_class=_CONFIG_FOR_DOC,  # 使用的配置类
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 输入的 token IDs
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力遮罩
        token_type_ids: np.ndarray | tf.Tensor | None = None,  # token 类型 IDs
        head_mask: np.ndarray | tf.Tensor | None = None,  # 头部遮罩
        inputs_embeds: np.ndarray | tf.Tensor | None = None,  # 输入的嵌入表示
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典格式的输出
        start_positions: np.ndarray | tf.Tensor | None = None,  # 起始位置
        end_positions: np.ndarray | tf.Tensor | None = None,  # 结束位置
        training: Optional[bool] = False,  # 是否处于训练模式

        ...
        # 方法未完全显示，继续注释剩余部分
        ...
        ):
        # 方法主体未显示完全，继续注释其余部分
        ...


由于代码块过长，无法在此显示完整的 `call` 方法主体部分。
    ) -> Union[TFQuestionAnsweringModelOutput, Tuple[tf.Tensor]]:
        r"""
        start_positions (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        # 调用 RoFormer 模型进行预测，返回输出结果
        outputs = self.roformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 获取序列输出
        sequence_output = outputs[0]
        # 将序列输出传入 QA 输出层，得到 logits
        logits = self.qa_outputs(inputs=sequence_output)
        # 将 logits 按照最后一个维度分割成 start_logits 和 end_logits
        start_logits, end_logits = tf.split(value=logits, num_or_size_splits=2, axis=-1)
        # 移除 start_logits 和 end_logits 的最后一个维度，使其变为一维张量
        start_logits = tf.squeeze(input=start_logits, axis=-1)
        end_logits = tf.squeeze(input=end_logits, axis=-1)
        # 初始化损失为 None
        loss = None

        # 如果提供了 start_positions 和 end_positions，则计算损失
        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions, "end_position": end_positions}
            # 使用 labels 和 logits 计算损失
            loss = self.hf_compute_loss(labels=labels, logits=(start_logits, end_logits))

        # 如果不需要返回字典形式的输出，则根据是否有损失返回不同的输出
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典形式的输出，则构造 TFQuestionAnsweringModelOutput 对象返回
        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        # 设置模型已构建标志为 True
        self.built = True
        # 如果模型中包含 RoFormer 层，则构建 RoFormer 层
        if getattr(self, "roformer", None) is not None:
            with tf.name_scope(self.roformer.name):
                self.roformer.build(None)
        # 如果模型中包含 QA 输出层，则构建 QA 输出层
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])
```