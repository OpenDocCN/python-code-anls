# `.\transformers\models\roformer\modeling_tf_roformer.py`

```py
# 设置文件编码为 UTF-8
# 版权声明及许可信息
#
# 版权所有 2021 年 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（"许可证"）获得许可；
# 除非符合许可证的要求，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则依照许可证分发的软件
# 是按"原样"的基础分发，不附带任何明示或暗示的保证或条件。
# 请参阅许可证了解特定语言的许可信息和限制。
""" TF 2.0 RoFormer 模型。"""

# 导入必要的模块和库
from __future__ import annotations
import math
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf

# 导入相关的模块和函数
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

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置信息
_CHECKPOINT_FOR_DOC = "junnyu/roformer_chinese_base"
_CONFIG_FOR_DOC = "RoFormerConfig"

# RoFormer 预训练模型的存档列表
TF_ROFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "junnyu/roformer_chinese_small",
    "junnyu/roformer_chinese_base",
    "junnyu/roformer_chinese_char_small",
    "junnyu/roformer_chinese_char_base",
    "junnyu/roformer_small_discriminator",
    "junnyu/roformer_small_generator",
    # 查看所有 RoFormer 模型的列表 https://huggingface.co/models?filter=roformer
]


class TFRoFormerSinusoidalPositionalEmbedding(tf.keras.layers.Layer):
    """这个模块生成任意长度的正弦位置嵌入。"""

    def __init__(self, num_positions: int, embedding_dim: int, **kwargs):
        super().__init__(**kwargs)

        # 如果嵌入维度为奇数，则抛出异常
        if embedding_dim % 2 != 0:
            raise NotImplementedError(f"odd embedding_dim {embedding_dim} not supported")

        # 初始化嵌入维度和位置数量
        self.embedding_dim = embedding_dim
        self.num_positions = num_positions
    # 构建共享的 token 嵌入层
    def build(self, input_shape: tf.TensorShape):
        # 根据参数初始化权重
        weight = self._init_weight(self.num_positions, self.embedding_dim)
        
        # 创建可训练的权重变量
        self.weight = self.add_weight(
            name="embeddings",
            shape=[self.num_positions, self.embedding_dim],
        )
        
        # 将初始化的权重值赋给可训练的权重变量
        weight = tf.cast(weight, dtype=self.weight.dtype)
        self.weight.assign(weight)
        
        # 调用父类的 build 方法
        super().build(input_shape)
    
    # 初始化位置编码权重
    @staticmethod
    def _init_weight(n_pos: int, dim: int):
        # 计算位置编码值
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        table = np.zeros_like(position_enc)
        
        # 计算 sin 和 cos 值并填充到 table 中
        table[:, 0 : dim // 2] = np.sin(position_enc[:, 0::2])
        table[:, dim // 2 :] = np.cos(position_enc[:, 1::2])
        
        # 将 numpy 数组转换为 Tensor，并停止梯度
        table = tf.convert_to_tensor(table)
        tf.stop_gradient(table)
        return table
    
    # 获取位置编码
    def call(self, input_shape: tf.TensorShape, past_key_values_length: int = 0):
        # 获取 batch 大小和序列长度
        bsz, seq_len = input_shape[:2]
        
        # 根据过去关键值的长度计算当前位置
        positions = tf.range(past_key_values_length, seq_len + past_key_values_length, delta=1, name="range")
        
        # 从权重矩阵中获取对应位置的值
        return tf.gather(self.weight, positions)
class TFRoFormerEmbeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config: RoFormerConfig, **kwargs):
        super().__init__(**kwargs)

        # 初始化 RoFormerEmbeddings 类
        self.config = config
        self.embedding_size = config.embedding_size
        self.initializer_range = config.initializer_range
        # LayerNormalization 层，用于归一化
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # Dropout 层，用于随机失活
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def build(self, input_shape=None):
        # 创建词嵌入矩阵
        with tf.name_scope("word_embeddings"):
            self.weight = self.add_weight(
                name="weight",
                shape=[self.config.vocab_size, self.embedding_size],
                # 从指定范围内初始化权重
                initializer=get_initializer(self.initializer_range),
            )

        # 创建 token_type_embeddings 矩阵
        with tf.name_scope("token_type_embeddings"):
            self.token_type_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.config.type_vocab_size, self.embedding_size],
                # 从指定范围内初始化权重
                initializer=get_initializer(self.initializer_range),
            )

        # 如果已经构建则返回
        if self.built:
            return
        self.built = True
        # 构建 LayerNorm 层
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


        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        # 确保 input_ids 和 inputs_embeds 至少有一个不为空
        assert not (input_ids is None and inputs_embeds is None)

        # 如果 input_ids 不为空，根据 input_ids 获取对应的词嵌入
        if input_ids is not None:
            # 确保 input_ids 的值在词表范围内
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        input_shape = shape_list(inputs_embeds)[:-1]

        # 如果 token_type_ids 为空，创建一个与 inputs_embeds 形状相同的填充为 0 的张量
        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)

        # 根据 token_type_ids 获取 token_type_embeddings
        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)
        # 将词嵌入与 token_type_embeddings 相加得到最终的 embeddings
        final_embeddings = inputs_embeds + token_type_embeds
        # 对最终的 embeddings 进行 LayerNormalization
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        # 对 embeddings 进行 Dropout 处理
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)

        return final_embeddings


class TFRoFormerSelfAttention(tf.keras.layers.Layer):
    # 初始化函数，接收配置文件和额外参数
    def __init__(self, config: RoFormerConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 检查隐藏层大小是否能被注意力头数整除
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        # 初始化注意力头数、每个注意力头的大小、所有头大小以及注意力头大小的平方根
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        # 初始化查询、键、值的全连接层以及 dropout 层
        self.query = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        self.key = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        self.value = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )
        self.dropout = tf.keras.layers.Dropout(rate=config.attention_probs_dropout_prob)
        self.rotary_value = config.rotary_value
        self.config = config

    # 为得分矩阵转置，将 [batch_size, seq_length, all_head_size] 转换为 [batch_size, seq_length, num_attention_heads, attention_head_size]
    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

        # 将张量从 [batch_size, seq_length, num_attention_heads, attention_head_size] 转置为 [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    # 执行函数，传入隐藏状态、注意力掩码、正弦位置编码、头掩码、是否输出注意力矩阵、训练标识符
    ) -> Tuple[tf.Tensor]:
        # 获取隐藏层状态的批次大小
        batch_size = shape_list(hidden_states)[0]
        # 对隐藏层状态进行查询操作
        mixed_query_layer = self.query(inputs=hidden_states)
        # 对隐藏层状态进行键操作
        mixed_key_layer = self.key(inputs=hidden_states)
        # 对隐藏层状态进行值操作
        mixed_value_layer = self.value(inputs=hidden_states)
        # 将查询层进行转置以便用于计算注意力分数
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        # 将键层进行转置以便用于计算注意力分数
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        # 将值层进行转置以便用于计算注意力分数
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        # 如果存在正弦位置编码
        if sinusoidal_pos is not None:
            # 如果启用了旋转值，则应用正弦位置编码到查询层、键层和值层
            if self.rotary_value:
                query_layer, key_layer, value_layer = self.apply_rotary_position_embeddings(
                    sinusoidal_pos, query_layer, key_layer, value_layer
                )
            # 否则只应用正弦位置编码到查询层和键层
            else:
                query_layer, key_layer = self.apply_rotary_position_embeddings(sinusoidal_pos, query_layer, key_layer)

        # 获取原始注意力分数，即"查询"和"键"的点积
        # (batch size, num_heads, seq_len_q, seq_len_k)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        # 将缩放因子转换为与注意力分数数据类型相同的类型
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        # 将注意力分数除以缩放因子
        attention_scores = tf.divide(attention_scores, dk)

        # 如果存在注意力屏蔽
        if attention_mask is not None:
            # 应用注意力屏蔽（在TFRoFormerModel的call()函数中为所有层进行预计算）
            attention_scores = tf.add(attention_scores, attention_mask)

        # 将注意力分数归一化为概率
        attention_probs = stable_softmax(logits=attention_scores, axis=-1)

        # 进行Dropout操作以防止过拟合
        attention_probs = self.dropout(inputs=attention_probs, training=training)

        # 如果存在注意力头部屏蔽
        if head_mask is not None:
            # 将注意力概率与注意力头部屏蔽相乘
            attention_probs = tf.multiply(attention_probs, head_mask)

        # 计算最终的注意力输出
        attention_output = tf.matmul(attention_probs, value_layer)
        # 将注意力输出进行转置
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])

        # 将注意力输出进行重塑，以便与模型的其余部分兼容
        # (batch_size, seq_len_q, all_head_size)
        attention_output = tf.reshape(tensor=attention_output, shape=(batch_size, -1, self.all_head_size))
        # 返回注意力输出
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)

        # 返回输出结果
        return outputs

    @staticmethod
    # 应用旋转位置嵌入到查询、键、值层
    def apply_rotary_position_embeddings(sinusoidal_pos, query_layer, key_layer, value_layer=None):
        # 将正弦和余弦部分拆分开来
        sin, cos = tf.split(sinusoidal_pos, num_or_size_splits=2, axis=-1)
        # 将正弦部分进行重复，使每个角度对应两个连续的位置编码
        sin_pos = tf.repeat(sin, 2, axis=-1)
        # 将余弦部分进行重复，使每个角度对应两个连续的位置编码
        cos_pos = tf.repeat(cos, 2, axis=-1)
        # 旋转查询层的一半，相邻位置编码进行交换
        rotate_half_query_layer = tf.stack([-query_layer[..., 1::2], query_layer[..., ::2]], axis=-1)
        rotate_half_query_layer = tf.reshape(rotate_half_query_layer, shape_list(query_layer))
        # 根据位置编码旋转查询层
        query_layer = query_layer * cos_pos + rotate_half_query_layer * sin_pos
        # 旋转键层的一半，相邻位置编码进行交换
        rotate_half_key_layer = tf.stack([-key_layer[..., 1::2], key_layer[..., ::2]], axis=-1)
        rotate_half_key_layer = tf.reshape(rotate_half_key_layer, shape_list(key_layer))
        # 根据位置编码旋转键层
        key_layer = key_layer * cos_pos + rotate_half_key_layer * sin_pos
        if value_layer is not None:
            # 旋转值层的一半，相邻位置编码进行交换
            rotate_half_value_layer = tf.stack([-value_layer[..., 1::2], value_layer[..., ::2]], axis=-1)
            rotate_half_value_layer = tf.reshape(rotate_half_value_layer, shape_list(value_layer))
            # 根据位置编码旋转值层
            value_layer = value_layer * cos_pos + rotate_half_value_layer * sin_pos
            return query_layer, key_layer, value_layer
        return query_layer, key_layer

    # 构建自注意力层
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在查询层，则构建查询层
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.hidden_size])
        # 如果存在键层，则构建键层
        if getattr(self, "key", None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.config.hidden_size])
        # 如果存在值层，则构建值层
        if getattr(self, "value", None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.config.hidden_size])
# 自定义Keras层类，用于RoFormer模型中的SelfOutput处理
class TFRoFormerSelfOutput(tf.keras.layers.Layer):
    # 初始化层，配置参数来源于RoFormerConfig
    def __init__(self, config: RoFormerConfig, **kwargs):
        super().__init__(**kwargs)  # 调用父类初始化方法

        # 定义一个全连接层，用于变换隐藏状态的大小
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 添加层归一化，提高训练稳定性
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 添加dropout层，用于训练中的随机失活
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        # 保存配置到层内部
        self.config = config

    # 调用方法定义层的前向传播逻辑
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)  # 通过全连接层处理隐藏状态
        hidden_states = self.dropout(inputs=hidden_states, training=training)  # 应用dropout
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)  # 应用层归一化并加上输入张量

        return hidden_states  # 返回处理后的隐藏状态

    # 构建层的结构，可选定义
    def build(self, input_shape=None):
        if self.built:  # 如果已经构建，则不再重复构建
            return
        self.built = True  # 标记层为已构建
        if getattr(self, "dense", None) is not None:  # 如果dense层存在
            with tf.name_scope(self.dense.name):  # 使用命名空间包装
                self.dense.build([None, None, self.config.hidden_size])  # 构建dense层
        if getattr(self, "LayerNorm", None) is not None:  # 如果LayerNorm层存在
            with tf.name_scope(self.LayerNorm.name):  # 使用命名空间包装
                self.LayerNorm.build([None, None, self.config.hidden_size])  # 构建LayerNorm层

# 自定义Keras层类，用于RoFormer模型中的Attention处理
class TFRoFormerAttention(tf.keras.layers.Layer):
    # 初始化层，配置参数来源于RoFormerConfig
    def __init__(self, config: RoFormerConfig, **kwargs):
        super().__init__(**kwargs)  # 调用父类初始化方法

        # 定义自注意力层
        self.self_attention = TFRoFormerSelfAttention(config, name="self")
        # 定义输出层
        self.dense_output = TFRoFormerSelfOutput(config, name="output")

    # 用于修剪注意力头的方法（未实现）
    def prune_heads(self, heads):
        raise NotImplementedError

    # 调用方法定义层的前向传播逻辑
    def call(
        self,
        input_tensor: tf.Tensor,
        attention_mask: tf.Tensor,
        sinusoidal_pos: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        # 调用自注意力层进行处理
        self_outputs = self.self_attention(
            hidden_states=input_tensor,
            attention_mask=attention_mask,
            sinusoidal_pos=sinusoidal_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            training=training,
        )
        # 将自注意力层的输出传递给输出层进行进一步处理
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=input_tensor, training=training
        )
        # 结合原始输出和处理后的输出
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them

        return outputs  # 返回最终的输出结果
    # 定义 build 函数，接收可选参数 input_shape，用于构建对象的内部组件
    def build(self, input_shape=None):
        # 如果对象已经构建过，则直接返回
        if self.built:
            return
        # 标记对象已构建
        self.built = True
        # 如果对象具有 self_attention 属性且不为 None，则在指定命名空间中构建 self_attention
        if getattr(self, "self_attention", None) is not None:
            with tf.name_scope(self.self_attention.name):
                self.self_attention.build(None)
        # 如果对象具有 dense_output 属性且不为 None，则在指定命名空间中构建 dense_output
        if getattr(self, "dense_output", None) is not None:
            with tf.name_scope(self.dense_output.name):
                self.dense_output.build(None)
# 以Bert为模板，创建RoFormer中间层的类
class TFRoFormerIntermediate(tf.keras.layers.Layer):
    # 初始化函数，接受RoFormer配置参数
    def __init__(self, config: RoFormerConfig, **kwargs):
        # 调用父类初始化函数
        super().__init__(**kwargs)

        # 创建一个全连接层，单元数为配置中的中间层大小，权重初始化采用配置中的初始化范围
        self.dense = tf.keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # 判断配置中的隐藏层激活函数是否为字符串类型，若是则根据字符串名获取对应的TensorFlow激活函数，若不是则直接使用配置中的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        # 保存配置参数
        self.config = config

    # 前向传播函数，接受隐藏状态张量，返回经过中间层与激活函数处理后的结果张量
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 使用全连接层处理隐藏状态张量
        hidden_states = self.dense(inputs=hidden_states)
        # 使用中间层的激活函数处理结果张量
        hidden_states = self.intermediate_act_fn(hidden_states)

        # 返回处理后的结果张量
        return hidden_states

    # 构建函数，用于构建层的结构
    def build(self, input_shape=None):
        # 如果层已经构建过，直接返回
        if self.built:
            return
        # 设置层为已构建状态
        self.built = True
        # 如果存在全连接层，根据配置参数构建全连接层的结构
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# 以Bert为模板，创建RoFormer输出层的类
class TFRoFormerOutput(tf.keras.layers.Layer):
    # 初始化函数，接受RoFormer配置参数
    def __init__(self, config: RoFormerConfig, **kwargs):
        # 调用父类初始化函数
        super().__init__(**kwargs)

        # 创建一个全连接层，单元数为配置中的隐藏层大小，权重初始化采用配置中的初始化范围
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个LayerNormalization层，设置epsilon为配置中的LayerNorm系数，用于对全连接层的输出做BN处理
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建一个Dropout层，设置丢弃率为配置中的丢弃率，用于对全连接层的输出做随机丢弃
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        # 保存配置参数
        self.config = config

    # 前向传播函数，接受隐藏状态张量、输入张量和训练标志，返回经过全连接层、随机丢弃、LayerNormalization处理后的结果张量
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 使用全连接层处理隐藏状态张量
        hidden_states = self.dense(inputs=hidden_states)
        # 使用Dropout层对处理后的结果张量进行随机丢弃，在训练状态下使用
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 使用LayerNormalization层对丢弃后的结果张量与输入张量做残差连接及规范化处理
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        # 返回处理后的结果张量
        return hidden_states

    # 构建函数，用于构建层的结构
    def build(self, input_shape=None):
        # 如果层已经构建过，直接返回
        if self.built:
            return
        # 设置层为已构建状态
        self.built = True
        # 如果存在全连接层，根据配置参数构建全连接层的结构
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])
        # 如果存在LayerNormalization层，根据配置参数构建LayerNormalization层的结构
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# 创建RoFormer层的类，用于组合RoFormer的注意力、中间、输出层
class TFRoFormerLayer(tf.keras.layers.Layer):
    # 初始化函数，接受RoFormer配置参数
    def __init__(self, config: RoFormerConfig, **kwargs):
        # 调用父类初始化函数
        super().__init__(**kwargs)

        # 创建RoFormer的注意力层
        self.attention = TFRoFormerAttention(config, name="attention")
        # 创建RoFormer的中间层
        self.intermediate = TFRoFormerIntermediate(config, name="intermediate")
        # 创建RoFormer的输出层
        self.roformer_output = TFRoFormerOutput(config, name="output")
    # 定义 RoFormer 模型的 call 方法，用于进行前向传播
    def call(
        self,
        hidden_states: tf.Tensor,  # 输入的隐藏状态张量
        attention_mask: tf.Tensor,  # 注意力掩码张量，用于指示哪些位置需要被屏蔽
        sinusoidal_pos: tf.Tensor,  # 正弦位置编码张量，用于表示位置信息
        head_mask: tf.Tensor,  # 头部掩码张量，用于指定哪些注意力头应该被屏蔽
        output_attentions: bool,  # 是否输出注意力权重
        training: bool = False,  # 是否处于训练模式
    ) -> Tuple[tf.Tensor]:  # 返回一个元组，包含 RoFormer 模型的输出张量
        # 调用注意力机制层，获取注意力输出
        attention_outputs = self.attention(
            input_tensor=hidden_states,
            attention_mask=attention_mask,
            sinusoidal_pos=sinusoidal_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            training=training,
        )
        # 获取注意力输出张量
        attention_output = attention_outputs[0]
        # 通过中间层进行特征变换
        intermediate_output = self.intermediate(hidden_states=attention_output)
        # RoFormer 输出层对特征进行处理
        layer_output = self.roformer_output(
            hidden_states=intermediate_output, input_tensor=attention_output, training=training
        )
        # 构建输出元组，包含 RoFormer 输出和可能的注意力权重
        outputs = (layer_output,) + attention_outputs[1:]  # 如果需要输出注意力权重，则添加到输出元组中

        return outputs

    # 构建 RoFormer 模型
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 构建注意力机制层
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        # 构建中间层
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        # 构建 RoFormer 输出层
        if getattr(self, "roformer_output", None) is not None:
            with tf.name_scope(self.roformer_output.name):
                self.roformer_output.build(None)
# TFRoFormerEncoder 是一个 Tensorflow 的 Keras 层，实现了 RoFormer 编码器的功能
class TFRoFormerEncoder(tf.keras.layers.Layer):
    def __init__(self, config: RoFormerConfig, **kwargs):
        super().__init__(**kwargs)
        # 创建位置编码层，用于给输入序列添加位置信息
        self.embed_positions = TFRoFormerSinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            config.hidden_size // config.num_attention_heads,
            name="embed_positions",
        )
        # 创建多个 TFRoFormerLayer 层，堆叠用于构建编码器
        self.layer = [TFRoFormerLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

    # 编码器的前向传播过程
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        # 初始化保存中间隐藏状态和注意力权重的元组
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # 计算位置编码
        sinusoidal_pos = self.embed_positions(shape_list(hidden_states)[:-1])[None, None, :, :]

        # 遍历每个 TFRoFormerLayer 层，进行前向传播
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 调用 TFRoFormerLayer 层进行前向传播
            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                sinusoidal_pos=sinusoidal_pos,
                head_mask=head_mask[i],
                output_attentions=output_attentions,
                training=training,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # 添加最后一层的隐藏状态
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 根据参数决定返回格式
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )

    # 构建层的权重
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "embed_positions", None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)


# TFRoFormerPredictionHeadTransform 是一个 Tensorflow 的 Keras 层，用于预测头的转换
class TFRoFormerPredictionHeadTransform(tf.keras.layers.Layer):
    # 定义一个类，继承自tf.keras.Model，并接收RoFormerConfig类型的config参数以及其他参数
    def __init__(self, config: RoFormerConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
    
        # 创建一个全连接层，输入单元数为config.embedding_size，权重初始化通过config.initializer_range获取
        self.dense = tf.keras.layers.Dense(
            units=config.embedding_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )
    
        # 判断config.hidden_act的类型，如果是字符串则通过get_tf_activation函数获取对应的激活函数，否则直接使用config.hidden_act作为激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.transform_act_fn = config.hidden_act
    
        # 创建一个LayerNormalization层，epsilon通过config.layer_norm_eps获取
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 将config保存到self.config中
        self.config = config
    
    # 定义call方法，输入一个tf.Tensor类型的hidden_states，返回一个tf.Tensor类型的结果
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 将hidden_states传入全连接层
        hidden_states = self.dense(inputs=hidden_states)
        # 将全连接层的输出经过transform_act_fn函数处理
        hidden_states = self.transform_act_fn(hidden_states)
        # 将处理后的hidden_states传入LayerNorm层
        hidden_states = self.LayerNorm(inputs=hidden_states)
    
        # 返回处理后的hidden_states
        return hidden_states
    
    # 定义build方法，用于构建模型
    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回
        if self.built:
            return
        # 将self.built标记为已构建
        self.built = True
        # 如果存在dense属性，则构建dense层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果存在LayerNorm属性，则构建LayerNorm层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.embedding_size])
# 定义 RoFormer 语言模型预测头部类，继承自 tf.keras.layers.Layer
class TFRoFormerLMPredictionHead(tf.keras.layers.Layer):
    # 初始化方法，接受 RoFormer 配置、输入 embeddings 和其他参数
    def __init__(self, config: RoFormerConfig, input_embeddings: tf.keras.layers.Layer, **kwargs):
        # 调用父类初始化方法
        super().__init__(**kwargs)

        # 保存配置信息和 embedding 大小
        self.config = config
        self.embedding_size = config.embedding_size

        # 初始化预测头部的转换层
        self.transform = TFRoFormerPredictionHeadTransform(config, name="transform")

        # 输入 embeddings 与输出权重相同，但每个 token 有一个单独的输出偏置
        self.input_embeddings = input_embeddings

    # 构建方法，构建输出权重偏置并添加到权重中
    def build(self, input_shape=None):
        # 添加输出偏置权重并初始化为零向量
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")

        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在转换层，则构建转换层
        if getattr(self, "transform", None) is not None:
            with tf.name_scope(self.transform.name):
                self.transform.build(None)

    # 获取输出 embeddings 方法
    def get_output_embeddings(self) -> tf.keras.layers.Layer:
        return self.input_embeddings

    # 设置输出 embeddings 方法
    def set_output_embeddings(self, value: tf.Variable):
        self.input_embeddings.weight = value
        self.input_embeddings.vocab_size = shape_list(value)[0]

    # 获取偏置方法
    def get_bias(self) -> Dict[str, tf.Variable]:
        return {"bias": self.bias}

    # 设置偏置方法
    def set_bias(self, value: tf.Variable):
        self.bias = value["bias"]
        self.config.vocab_size = shape_list(value["bias"])[0]

    # 调用方法，进行转换和计算输出结果
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 对隐藏状态进行转换
        hidden_states = self.transform(hidden_states=hidden_states)
        seq_length = shape_list(hidden_states)[1]
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.embedding_size])
        # 矩阵乘法得到输出结果
        hidden_states = tf.matmul(a=hidden_states, b=self.input_embeddings.weight, transpose_b=True)
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
        # 添加偏置计算最终输出
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)

        return hidden_states


# ���贝自 transformers.models.bert.modeling_tf_bert.TFBertMLMHead，将 Bert 替换为 RoFormer
# 定义 RoFormer 语言模型头部类，继承自 tf.keras.layers.Layer
class TFRoFormerMLMHead(tf.keras.layers.Layer):
    # 初始化方法，接受 RoFormer 配置、输入 embeddings 和其他参数
    def __init__(self, config: RoFormerConfig, input_embeddings: tf.keras.layers.Layer, **kwargs):
        # 调用父类初始化方法
        super().__init__(**kwargs)

        # 创建 RoFormer 语言模型预测头部对象
        self.predictions = TFRoFormerLMPredictionHead(config, input_embeddings, name="predictions")

    # 调用方法，对输入进行预测
    def call(self, sequence_output: tf.Tensor) -> tf.Tensor:
        prediction_scores = self.predictions(hidden_states=sequence_output)

        return prediction_scores

    # 构建方法，如果已经构建则直接返回，否则构建预测头部
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在预测头部，则构建预测头部
        if getattr(self, "predictions", None) is not None:
            with tf.name_scope(self.predictions.name):
                self.predictions.build(None)

# 定义 RoFormer 主层类，继承自 tf.keras.layers.Layer
@keras_serializable
class TFRoFormerMainLayer(tf.keras.layers.Layer):
    # RoFormer 主层配置类为 RoFormerConfig
    config_class = RoFormerConfig
    def __init__(self, config: RoFormerConfig, add_pooling_layer: bool = True, **kwargs):
        # 调用父类的初始化方法，传入额外的参数
        super().__init__(**kwargs)

        # 保存传入的 RoFormerConfig 对象
        self.config = config

        # 创建 TFRoFormerEmbeddings 对象，命名为 "embeddings"
        self.embeddings = TFRoFormerEmbeddings(config, name="embeddings")
        # 如果 embedding_size 不等于 hidden_size，则创建一个 Dense 层，将 embedding_size 转换为 hidden_size
        if config.embedding_size != config.hidden_size:
            self.embeddings_project = tf.keras.layers.Dense(config.hidden_size, name="embeddings_project")

        # 创建 TFRoFormerEncoder 对象，命名为 "encoder"
        self.encoder = TFRoFormerEncoder(config, name="encoder")

    # 获取输入 embeddings 的方法
    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        return self.embeddings

    # 设置输入 embeddings 的方法
    def set_input_embeddings(self, value: tf.Variable):
        # 设置 embeddings 的权重为传入的 value
        self.embeddings.weight = value
        # 设置 embeddings 的词汇量大小为 value 的第一个维度大小
        self.embeddings.vocab_size = shape_list(value)[0]

    # 剪枝模型头部的方法
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 抛出未实现的错误，子类需要重写该方法
        raise NotImplementedError

    # 封装输入的装饰器
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
    # 构建模型的方法
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True
        # 如果 embeddings 对象存在，构建 embeddings
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        # 如果 encoder 对象存在，构建 encoder
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果 embeddings_project 对象存在，构建 embeddings_project
        if getattr(self, "embeddings_project", None) is not None:
            with tf.name_scope(self.embeddings_project.name):
                self.embeddings_project.build([None, None, self.config.embedding_size])
class TFRoFormerPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置模型的配置类
    config_class = RoFormerConfig
    # 设置模型的基础前缀
    base_model_prefix = "roformer"


ROFORMER_START_DOCSTRING = r"""

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
        config ([`RoFormerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

ROFORMER_INPUTS_DOCSTRING = r"""
"""


@add_start_docstrings(
    "The bare RoFormer Model transformer outputing raw hidden-states without any specific head on top.",
    ROFORMER_START_DOCSTRING,
)
# 继承自 TFRoFormerPreTrainedModel 的 RoFormer 模型
class TFRoFormerModel(TFRoFormerPreTrainedModel):
    # 初始化方法，接收 RoFormerConfig 对象以及其他可变位置和关键字参数
    def __init__(self, config: RoFormerConfig, *inputs, **kwargs):
        # 调用父类的初始化方法，传递配置对象以及其他参数
        super().__init__(config, *inputs, **kwargs)

        # 创建 RoFormer 主层，使用给定的配置对象
        self.roformer = TFRoFormerMainLayer(config, name="roformer")

    # 使用装饰器 unpack_inputs，将输入参数展开
    # 使用装饰器 add_start_docstrings_to_model_forward，添加模型前向传播的文档字符串
    # 使用装饰器 add_code_sample_docstrings，添加代码示例的文档字符串
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
        # 调用 RoFormer 主层，传递各种输入参数
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

        # 返回 RoFormer 主层的输出
        return outputs

    # 构建方法，在第一次调用时构建模型
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 设置标志表示已经构建过
        self.built = True
        # 如果 RoFormer 主层存在
        if getattr(self, "roformer", None) is not None:
            # 在命名作用域内构建 RoFormer 主层
            with tf.name_scope(self.roformer.name):
                self.roformer.build(None)
# 使用装饰器为 TFRoFormerForMaskedLM 添加文档字符串，描述其作为带有语言建模头的 RoFormer 模型
@add_start_docstrings("""RoFormer Model with a `language modeling` head on top.""", ROFORMER_START_DOCSTRING)
# 定义 TFRoFormerForMaskedLM 类，继承自 TFRoFormerPreTrainedModel 和 TFMaskedLanguageModelingLoss
class TFRoFormerForMaskedLM(TFRoFormerPreTrainedModel, TFMaskedLanguageModelingLoss):
    # 初始化函数，接受 RoFormerConfig 对象和其他参数
    def __init__(self, config: RoFormerConfig, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 如果配置为解码器，则发出警告
        if config.is_decoder:
            logger.warning(
                "If you want to use `TFRoFormerForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 创建 RoFormer 主层对象
        self.roformer = TFRoFormerMainLayer(config, name="roformer")
        # 创建 RoFormer 的 MLM 头部对象
        self.mlm = TFRoFormerMLMHead(config, input_embeddings=self.roformer.embeddings, name="mlm___cls")

    # 获取语言建模头部
    def get_lm_head(self) -> tf.keras.layers.Layer:
        return self.mlm.predictions

    # 定义 call 方法，处理输入并执行前向传播
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
    ):
        # 进行模型前向传播，处理输入数据
    ) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional`):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        # 调用 roformer 模型进行预测
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
        # 使用 MLM 模型进行预测
        prediction_scores = self.mlm(sequence_output=sequence_output, training=training)
        # 如果没有标签，损失设为 None，否则计算损失函数
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=prediction_scores)

        # 如果不需要返回字典，则返回预测分数和其他输出信息
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 MaskedLM 模型输出
        return TFMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建，则直接返回
        if self.built:
            return
        self.built = True
        # 如果 roformer 模型存在，构建 roformer
        if getattr(self, "roformer", None) is not None:
            with tf.name_scope(self.roformer.name):
                self.roformer.build(None)
        # 如果 MLM 模型存在，构建 MLM
        if getattr(self, "mlm", None) is not None:
            with tf.name_scope(self.mlm.name):
                self.mlm.build(None)
# 这个类定义了一个基于 RoFormer 模型的因果语言模型(Causal Language Model, CLM)
@add_start_docstrings(
    """RoFormer Model with a `language modeling` head on top for CLM fine-tuning.""", ROFORMER_START_DOCSTRING
)
class TFRoFormerForCausalLM(TFRoFormerPreTrainedModel, TFCausalLanguageModelingLoss):
    def __init__(self, config: RoFormerConfig, *inputs, **kwargs):
        # 调用父类的构造方法，并传入 RoFormerConfig 对象
        super().__init__(config, *inputs, **kwargs)

        # 如果模型不是解码器模型，则记录警告信息
        if not config.is_decoder:
            logger.warning("If you want to use `TFRoFormerForCausalLM` as a standalone, add `is_decoder=True.`")

        # 初始化 RoFormer 主干模型和 Masked Language Model (MLM) 头部
        self.roformer = TFRoFormerMainLayer(config, name="roformer")
        self.mlm = TFRoFormerMLMHead(config, input_embeddings=self.roformer.embeddings, name="mlm___cls")

    # 获取语言模型预测头部
    def get_lm_head(self) -> tf.keras.layers.Layer:
        return self.mlm.predictions

    # 定义模型的前向传播过程
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
        # 通过 RoFormer 主干模型获得序列输出
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
        sequence_output = outputs[0]

        # 将序列输出传入 MLM 头部获得预测logits
        logits = self.mlm(sequence_output=sequence_output, training=training)
        loss = None

        # 如果提供了标签数据，则计算交叉熵损失
        if labels is not None:
            # 将标签向左移动一位，以匹配logits形状
            shifted_logits = logits[:, :-1]
            labels = labels[:, 1:]
            loss = self.hf_compute_loss(labels=labels, logits=shifted_logits)

        # 根据 return_dict 参数返回不同形式的输出
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFCausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    # 定义一个方法用于构建神经网络模型，输入参数为输入数据的形状，默认为None
    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 检查是否存在roformer属性，如果存在则进行下一步操作
        if getattr(self, "roformer", None) is not None:
            # 在TensorFlow中创建一个命名空间，名称为roformer.name
            with tf.name_scope(self.roformer.name):
                # 调用roformer对象的build方法，传入None作为输入形状
                self.roformer.build(None)
        # 检查是否存在mlm属性，如果存在则进行下一步操作
        if getattr(self, "mlm", None) is not None:
            # 在TensorFlow中创建一个命名空间，名称为mlm.name
            with tf.name_scope(self.mlm.name):
                # 调用mlm对象的build方法，传入None作为输入形状
                self.mlm.build(None)
class TFRoFormerClassificationHead(tf.keras.layers.Layer):
    """Head for sentence-level classification tasks."""

    # 初始化分类头对象
    def __init__(self, config: RoFormerConfig, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

        # 定义全连接层，用于特征映射
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        
        # 定义Dropout层，用于防止过拟合
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        
        # 定义输出层，用于最终分类预测
        self.out_proj = tf.keras.layers.Dense(
            units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="out_proj"
        )

        # 根据配置初始化激活函数
        if isinstance(config.hidden_act, str):
            self.classifier_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.classifier_act_fn = config.hidden_act
        self.config = config

    # 定义模型的前向传播逻辑
    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 取第一个位置的隐藏状态，一般用于分类任务
        hidden_states = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS])
        
        # 在隐藏状态上进行Dropout操作
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        
        # 全连接层特征映射
        hidden_states = self.dense(inputs=hidden_states)
        
        # 经过激活函数
        hidden_states = self.classifier_act_fn(hidden_states)
        
        # 再次进行Dropout操作
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        
        # 最终分类预测
        hidden_states = self.out_proj(hidden_states)

        return hidden_states

    # 模型构建
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.config.hidden_size])


@add_start_docstrings(
    """
    RoFormer Model transformer with a sequence classification/regression head on top e.g., for GLUE tasks.
    """,
    ROFORMER_START_DOCSTRING,
)
# RoFormer模型，用于序列分类或回归任务
class TFRoFormerForSequenceClassification(TFRoFormerPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: RoFormerConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 获取分类标签数目
        self.num_labels = config.num_labels

        # RoFormer主模型
        self.roformer = TFRoFormerMainLayer(config, name="roformer")
        
        # 分类头
        self.classifier = TFRoFormerClassificationHead(config, name="classifier")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ROFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义一个 call 方法，用于执行模型推理
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 输入数据的 token id
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力掩码
        token_type_ids: np.ndarray | tf.Tensor | None = None,  # token 类型 id
        head_mask: np.ndarray | tf.Tensor | None = None,  # 注意力头掩码
        inputs_embeds: np.ndarray | tf.Tensor | None = None,  # 输入的嵌入向量
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出中间隐藏状态
        return_dict: Optional[bool] = None,  # 是否以字典形式返回输出
        labels: np.ndarray | tf.Tensor | None = None,  # 标签数据
        training: Optional[bool] = False,  # 是否为训练模式
    ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        # 调用 roformer 模型获取输出
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
        # 将 roformer 输出通过分类器获得最终预测结果
        logits = self.classifier(hidden_states=outputs[0], training=training)
        # 如果提供了标签数据，则计算损失函数
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)
    
        # 根据 return_dict 参数返回不同格式的输出
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
    
        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    # 定义 build 方法，用于构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建好了，直接返回
        if self.built:
            return
        self.built = True
        # 构建 roformer 模型
        if getattr(self, "roformer", None) is not None:
            with tf.name_scope(self.roformer.name):
                self.roformer.build(None)
        # 构建分类器
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build(None)
# 该类继承自 TFRoFormerPreTrainedModel 和 TFMultipleChoiceLoss，用于多项选择分类任务
@add_start_docstrings(
    """
    RoFormer Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    ROFORMER_START_DOCSTRING,
)
class TFRoFormerForMultipleChoice(TFRoFormerPreTrainedModel, TFMultipleChoiceLoss):
    def __init__(self, config: RoFormerConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 初始化 RoFormer 主体层
        self.roformer = TFRoFormerMainLayer(config, name="roformer")
        # 初始化序列总结层
        self.sequence_summary = TFSequenceSummary(config, config.initializer_range, name="sequence_summary")
        # 初始化分类层
        self.classifier = tf.keras.layers.Dense(
            units=1, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        self.config = config

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
    # 定义一个多选模型的前向传播函数
    def call(
        self,
        input_ids: Optional[TF_TENSOR_LIKE] = None,
        attention_mask: Optional[TF_TENSOR_LIKE] = None,
        token_type_ids: Optional[TF_TENSOR_LIKE] = None,
        head_mask: Optional[TF_TENSOR_LIKE] = None,
        inputs_embeds: Optional[TF_TENSOR_LIKE] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[TF_TENSOR_LIKE] = None,
        training: Optional[bool] = False,
    ) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
        # 根据输入 tensor 的形状计算出选项个数和序列长度
        if input_ids is not None:
            num_choices = shape_list(input_ids)[1]
            seq_length = shape_list(input_ids)[2]
        else:
            num_choices = shape_list(inputs_embeds)[1]
            seq_length = shape_list(inputs_embeds)[2]
        
        # 将输入 tensor 拉平成二维
        flat_input_ids = tf.reshape(tensor=input_ids, shape=(-1, seq_length)) if input_ids is not None else None
        flat_attention_mask = tf.reshape(tensor=attention_mask, shape=(-1, seq_length)) if attention_mask is not None else None
        flat_token_type_ids = tf.reshape(tensor=token_type_ids, shape=(-1, seq_length)) if token_type_ids is not None else None
        flat_inputs_embeds = tf.reshape(tensor=inputs_embeds, shape=(-1, seq_length, shape_list(inputs_embeds)[3])) if inputs_embeds is not None else None
        
        # 将拉平后的输入传给 RoFormer 模型
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
        
        # 对输出的序列特征进行汇总
        logits = self.sequence_summary(inputs=outputs[0], training=training)
        
        # 将汇总后的特征送入分类器得到最终的多选分类logits
        logits = self.classifier(inputs=logits)
        reshaped_logits = tf.reshape(tensor=logits, shape=(-1, num_choices))
        
        # 计算损失函数
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=reshaped_logits)
        
        # 根据返回设置，组织输出结果
        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return TFMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建好了就直接返回
        if self.built:
            return
        self.built = True
        
        # 分别构建 RoFormer、序列汇总和分类器子模块
        if getattr(self, "roformer", None) is not None:
            with tf.name_scope(self.roformer.name):
                self.roformer.build(None)
        if getattr(self, "sequence_summary", None) is not None:
            with tf.name_scope(self.sequence_summary.name):
                self.sequence_summary.build(None)
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
# 添加文档字符串，描述 RoFormer 模型及其在标记分类任务（如命名实体识别）上的用途
@add_start_docstrings(
    """
    RoFormer Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    ROFORMER_START_DOCSTRING,
)
class TFRoFormerForTokenClassification(TFRoFormerPreTrainedModel, TFTokenClassificationLoss):
    def __init__(self, config: RoFormerConfig, *inputs, **kwargs):
        # 调用父类的构造函数
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels

        # 初始化 RoFormer 主层
        self.roformer = TFRoFormerMainLayer(config, name="roformer")
        # 添加 dropout 层
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        # 添加分类器层
        self.classifier = tf.keras.layers.Dense(
            units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        self.config = config

    # 将各种输入参数解包，并为模型的前向传播方法添加文档字符串
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
    ) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 使用 RoFormer 进行前向传播，获取模型输出
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
        # 从模型输出中获取序列输出
        sequence_output = outputs[0]
        # 对序列输出进行 Dropout 操作，用于防止过拟合
        sequence_output = self.dropout(inputs=sequence_output, training=training)
        # 将 Dropout 后的序列输出传递给分类器，得到分类 logits
        logits = self.classifier(inputs=sequence_output)
        # 如果提供了标签，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        # 如果不返回字典形式的结果，则将结果组装成元组返回
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果返回字典形式的结果，则将损失、logits、隐藏状态和注意力矩阵组装成 TFTokenClassifierOutput 对象返回
        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果 RoFormer 子模型存在，则构建 RoFormer
        if getattr(self, "roformer", None) is not None:
            with tf.name_scope(self.roformer.name):
                self.roformer.build(None)
        # 如果分类器存在，则构建分类器
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                # 构建分类器，指定输入形状
                self.classifier.build([None, None, self.config.hidden_size])
# 定义 RoFormerForQuestionAnswering 类，继承 TFRoFormerPreTrainedModel 和 TFQuestionAnsweringLoss
@add_start_docstrings(
    """
    RoFormer Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    ROFORMER_START_DOCSTRING,
)
class TFRoFormerForQuestionAnswering(TFRoFormerPreTrainedModel, TFQuestionAnsweringLoss):
    def __init__(self, config: RoFormerConfig, *inputs, **kwargs):
        # 调用父类构造函数
        super().__init__(config, *inputs, **kwargs)
        # 设置标签数量
        self.num_labels = config.num_labels
        # 创建 TFRoFormerMainLayer 实例
        self.roformer = TFRoFormerMainLayer(config, name="roformer")
        # 创建用于计算起始和结束位置的密集层
        self.qa_outputs = tf.keras.layers.Dense(
            units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ROFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        # 输入 ID 序列
        input_ids: TFModelInputType | None = None,
        # 注意力掩码
        attention_mask: np.ndarray | tf.Tensor | None = None,
        # 输入序列的类型 ID
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        # 注意力头的掩码
        head_mask: np.ndarray | tf.Tensor | None = None,
        # 输入嵌入
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        # 是否输出注意力权重
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典格式的输出
        return_dict: Optional[bool] = None,
        # 问题答案的起始位置
        start_positions: np.ndarray | tf.Tensor | None = None,
        # 问题答案的结束位置
        end_positions: np.ndarray | tf.Tensor | None = None,
        # 是否进行训练
        training: Optional[bool] = False,
    # 定义一个函数，输入参数为输入和输出，返回一个 TFQuestionAnsweringModelOutput 对象或元组
    def call(self, input_ids: tf.Tensor, attention_mask: tf.Tensor, token_type_ids: tf.Tensor = None, head_mask: tf.Tensor = None, inputs_embeds: tf.Tensor = None, start_positions: tf.Tensor = None, end_positions: tf.Tensor = None, output_attentions: bool = None, output_hidden_states: bool = None, return_dict: bool = None, training: bool = None) -> Union[TFQuestionAnsweringModelOutput, Tuple[tf.Tensor]]:
        # 调用 RoFormer 模型进行计算
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
        # 获取 RoFormer 输出的序列
        sequence_output = outputs[0]
        # 通过 QA 输出层得到 logits
        logits = self.qa_outputs(inputs=sequence_output)
        # 将 logits 分解为起始位置和结束位置的 logits
        start_logits, end_logits = tf.split(value=logits, num_or_size_splits=2, axis=-1)
        # 移除 logits 的最后一个维度
        start_logits = tf.squeeze(input=start_logits, axis=-1)
        end_logits = tf.squeeze(input=end_logits, axis=-1)
        loss = None
        
        # 计算损失，如果存在起始和结束位置
        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions, "end_position": end_positions}
            loss = self.hf_compute_loss(labels=labels, logits=(start_logits, end_logits))
        
        # 如果不返回字典，则组合输出
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        # 返回 TFQuestionAnsweringModelOutput 对象
        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已经被构建，则直接返回
        if self.built:
            return
        self.built = True
        # 如果 RoFormer 模型存在，构建 RoFormer
        if getattr(self, "roformer", None) is not None:
            with tf.name_scope(self.roformer.name):
                self.roformer.build(None)
        # 如果 QA 输出层存在，构建 QA 输出层
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])
```