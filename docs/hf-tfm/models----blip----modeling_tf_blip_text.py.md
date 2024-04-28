# `.\transformers\models\blip\modeling_tf_blip_text.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 Salesforce Team 作者和 HuggingFace Team 所有
# 根据 BSD-3-Clause 许可证授权使用此文件
# 可以在遵守许可证的情况下使用此文件
# 可以在以下链接获取许可证的副本：https://opensource.org/licenses/BSD-3-Clause
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础，没有任何明示或暗示的担保或条件
# 请查看许可证以了解特定语言的权限和限制

# 导入必要的库
from __future__ import annotations
import math
from typing import Optional, Tuple
import tensorflow as tf

# 导入模型输出相关的类
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFBaseModelOutputWithPoolingAndCrossAttentions,
    TFCausalLMOutputWithCrossAttentions,
)
# 导入模型工具相关的函数和类
from ...modeling_tf_utils import (
    TFModelInputType,
    TFPreTrainedModel,
    get_initializer,
    get_tf_activation,
    keras_serializable,
    shape_list,
    unpack_inputs,
)
# 导入 TensorFlow 工具函数
from ...tf_utils import check_embeddings_within_bounds, invert_attention_mask, stable_softmax
# 导入通用工具函数
from ...utils import add_start_docstrings_to_model_forward, logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# BLIP 模型文本输入的文档字符串
BLIP_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            # 输入序列标记在词汇表中的索引。默认情况下，将忽略填充。
            # 可以使用 AutoProcessor 获取索引。有关详细信息，请参阅 BlipProcessor.__call__。

            # 什么是输入 ID？
        attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 避免在填充标记索引上执行注意力的掩码。掩码值选择在 [0, 1] 范围内：

            # - 对于**未屏蔽**的标记，值为 1，
            # - 对于**屏蔽**的标记，值为 0。

            # 什么是注意力掩码？
        position_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 每个输入序列标记在位置嵌入中的位置索引。选择范围为 [0, config.max_position_embeddings - 1]。

            # 什么是位置 ID？
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多详细信息，请参阅返回张量中的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多详细信息，请参阅返回张量中的 `hidden_states`。
        return_dict (`bool`, *optional*):
            # 是否返回一个 `~utils.ModelOutput` 而不是一个普通元组。
# 从 TensorFlow 中导入必要的模块
import tensorflow as tf

# 从 BLIP 项目中导入 TFBlipTextEmbeddings 类，用于构建文本嵌入层
# 参考自 https://github.com/salesforce/BLIP/blob/main/models/med.py#L52
class TFBlipTextEmbeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word and position embeddings."""

    # 定义 TFBlipTextEmbeddings 类的初始化方法
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 创建词嵌入层，用于将词索引转换为词嵌入向量
        self.word_embeddings = tf.keras.layers.Embedding(
            config.vocab_size,
            config.hidden_size,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="word_embeddings",
        )
        # 创建位置嵌入层，用于将位置索引转换为位置嵌入向量
        self.position_embeddings = tf.keras.layers.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="position_embeddings",
        )

        # 创建 LayerNormalization 层，用于对嵌入向量进行归一化处理
        # 保持与 PyTorch 模型变量名称一致，以便加载任何 TensorFlow 检查点文件
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建 Dropout 层，用于在训练过程中进行随机失活
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob, name="dropout")

        # 创建位置索引
        self.position_ids = tf.expand_dims(tf.range(config.max_position_embeddings), 0)
        # 获取位置嵌入类型，默认为"absolute"
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        # 存储配置信息
        self.config = config

    # 定义 TFBlipTextEmbeddings 类的调用方法
    def call(self, input_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0, training=None):
        # 计算输入的形状
        if input_ids is not None:
            input_shape = tf.shape(input_ids)
        else:
            input_shape = tf.shape(inputs_embeds)[:-1]

        seq_length = input_shape[1]

        # 如果未提供位置索引，则根据过去的关键值长度和序列长度生成位置索引
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # 如果未提供嵌入向量，则使用词嵌入层将输入词索引转换为词嵌入向量
        if inputs_embeds is None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = self.word_embeddings(input_ids)

        # 将词嵌入向量赋值给 embeddings 变量
        embeddings = inputs_embeds

        # 如果位置嵌入类型为"absolute"，则将位置嵌入向量加到词嵌入向量上
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        
        # 对嵌入向量进行 LayerNormalization 归一化处理
        embeddings = self.LayerNorm(embeddings)
        # 在训练过程中对嵌入向量进行随机失活
        embeddings = self.dropout(embeddings, training=training)
        # 返回处理后的嵌入向量
        return embeddings
    # 如果模型已经构建，则直接返回，不进行重复构建
    def build(self, input_shape=None):
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        # 如果存在单词嵌入（词向量）层，则构建该层
        if getattr(self, "word_embeddings", None) is not None:
            # 在 TensorFlow 中创建一个命名作用域，用于组织模型中的相关操作
            with tf.name_scope(self.word_embeddings.name):
                # 构建单词嵌入层
                self.word_embeddings.build(None)
        # 如果存在位置嵌入层，则构建该层
        if getattr(self, "position_embeddings", None) is not None:
            with tf.name_scope(self.position_embeddings.name):
                # 构建位置嵌入层
                self.position_embeddings.build(None)
        # 如果存在 LayerNorm（归一化）层，则构建该层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                # 构建 LayerNorm 层，需要指定输入的形状为 [None, None, self.config.hidden_size]
                self.LayerNorm.build([None, None, self.config.hidden_size])
        # 如果存在 dropout 层，则构建该层
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):
                # 构建 dropout 层，输入形状为 None（即未指定）
                self.dropout.build(None)
# 从给定链接的代码中适配而来，定义了一个自注意力层 TFBlipTextSelfAttention
class TFBlipTextSelfAttention(tf.keras.layers.Layer):
    # 初始化函数，接受配置和是否跨注意力的参数
    def __init__(self, config, is_cross_attention, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 保存配置信息
        self.config = config
        # 检查隐藏层大小是否是注意力头数的整数倍，如果不是则抛出异常
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                % (config.hidden_size, config.num_attention_heads)
            )

        # 保存注意力头数和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 创建查询、键、值的全连接层
        self.query = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        self.key = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        self.value = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )

        # 创建 dropout 层
        self.dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_prob)
        # 获取位置嵌入类型，默认为绝对位置
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 如果位置嵌入类型为相对位置，则创建距离嵌入层
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = tf.keras.layers.Embedding(
                2 * config.max_position_embeddings - 1, self.attention_head_size
            )
        # 保存是否为跨注意力
        self.is_cross_attention = is_cross_attention

    # 将输入张量转换为注意力分数的形状
    def transpose_for_scores(self, x):
        new_x_shape = tf.concat(
            [tf.shape(x)[:-1], tf.constant([self.num_attention_heads, self.attention_head_size], dtype=tf.int32)],
            axis=0,
        )
        x = tf.reshape(x, new_x_shape)
        return tf.transpose(x, perm=(0, 2, 1, 3))

    # 调用函数，接受多个参数
    def call(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        training=None,
    # 如果已经构建过，则直接返回，避免重复构建
    def build(self, input_shape=None):
        if self.built:
            return
        # 标记已经构建
        self.built = True
        # 如果有查询向量，则构建查询向量
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.hidden_size])
        # 如果是交叉注意力机制
        if self.is_cross_attention:
            # 如果有键向量，则构建键向量
            if getattr(self, "key", None) is not None:
                with tf.name_scope(self.key.name):
                    self.key.build([None, None, self.config.encoder_hidden_size])
            # 如果有值向量，则构建值向量
            if getattr(self, "value", None) is not None:
                with tf.name_scope(self.value.name):
                    self.value.build([None, None, self.config.encoder_hidden_size])
        else:
            # 如果有键向量，则构建键向量
            if getattr(self, "key", None) is not None:
                with tf.name_scope(self.key.name):
                    self.key.build([None, None, self.config.hidden_size])
            # 如果有值向量，则构建值向量
            if getattr(self, "value", None) is not None:
                with tf.name_scope(self.value.name):
                    self.value.build([None, None, self.config.hidden_size])
# 定义了一个基于 TF 的 BLIP 文本自注意力层
class TFBlipTextSelfOutput(tf.keras.layers.Layer):
    def __init__(self, config: BlipTextConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 创建一个全连接层，用于转换输入特征的维度
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个 LayerNormalization 层，用于归一化输入特征
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建一个 Dropout 层，用于随机置零输入特征
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        # 存储 BLIP 文本配置
        self.config = config

    # 定义该层的前向传播过程
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        # 使用全连接层转换输入特征的维度
        hidden_states = self.dense(inputs=hidden_states)
        # 使用 Dropout 层随机置零转换后的特征
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 使用 LayerNormalization 层对转换后的特征进行归一化，并与输入特征相加
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        # 返回处理后的特征
        return hidden_states

    # 构建层结构
    def build(self, input_shape=None):
        # 如果层已经构建完成，则直接返回
        if self.built:
            return
        # 标记该层已经构建
        self.built = True
        # 如果存在全连接层，则构建全连接层结构
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果存在 LayerNormalization 层，则构建 LayerNormalization 层结构
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# 从 https://github.com/salesforce/BLIP/blob/main/models/med.py#242 改编
# 定义了一个基于 TF 的 BLIP 文本注意力层
class TFBlipTextAttention(tf.keras.layers.Layer):
    def __init__(self, config, is_cross_attention=False, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 创建一个 BLIP 文本自注意力层
        self.self = TFBlipTextSelfAttention(config, is_cross_attention, name="self")
        # 创建一个 BLIP 文本自注意力层的输出层
        self.self_output = TFBlipTextSelfOutput(config, name="output")

    # 定义该层的前向传播过程
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        encoder_hidden_states: tf.Tensor | None = None,
        encoder_attention_mask: tf.Tensor | None = None,
        past_key_value: Tuple[Tuple[tf.Tensor]] | None = None,
        output_attentions: Optional[bool] = False,
        training: Optional[bool] = None,
    ):
        # 对输入进行自注意力计算，并获取注意力矩阵
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            training=training,
        )
        # 使用 BLIP 文本自注意力层的输出层处理自注意力计算的结果
        attention_output = self.self_output(self_outputs[0], hidden_states, training=training)
        # 将处理后的结果作为输出
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs
    # 构建神经网络模型
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果存在self属性
        if getattr(self, "self", None) is not None:
            # 使用tf的命名空间为self.name创建一个作用域
            with tf.name_scope(self.self.name):
                # 构建self属性
                self.self.build(None)
        # 如果存在self_output属性
        if getattr(self, "self_output", None) is not None:
            # 使用tf的命名空间为self_output.name创建一个作用域
            with tf.name_scope(self.self_output.name):
                # 构建self_output属性
                self.self_output.build(None)
# 从transformers.models.bert.modeling_tf_bert.TFBertIntermediate复制代码，并将Bert->BlipText
class TFBlipTextIntermediate(tf.keras.layers.Layer):
    def __init__(self, config: BlipTextConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，units为config中的intermediate_size，kernel_initializer为config中的initializer_range，名称为"dense"
        self.dense = tf.keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # 如果config中的hidden_act是字符串类型，则使用get_tf_activation函数获取对应的激活函数，否则直接使用config中的hidden_act
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 将输入的hidden_states通过全连接层dense进行计算
        hidden_states = self.dense(inputs=hidden_states)
        # 使用中间激活函数对计算结果进行激活
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在dense层，则构建dense层，输入形状为[None, None, self.config.hidden_size]
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


class TFBlipTextOutput(tf.keras.layers.Layer):
    def __init__(self, config: BlipTextConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，units为config中的hidden_size，kernel_initializer为config中的initializer_range，名称为"dense"
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建LayerNormalization层，epsilon为config中的layer_norm_eps，名称为"LayerNorm"
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建Dropout层，丢弃率为config中的hidden_dropout_prob
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 将输入的hidden_states通过全连接层dense进行计算
        hidden_states = self.dense(inputs=hidden_states)
        # 在训练时使用dropout对计算结果进行处理
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 将处理后的结果与输入的input_tensor相加后通过LayerNormalization层处理
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在dense层，则构建dense层，输入形状为[None, None, self.config.intermediate_size]
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])
        # 如果存在LayerNorm层，则构建LayerNorm层，输入形状为[None, None, self.config.hidden_size]


class TFBlipTextLayer(tf.keras.layers.Layer):
    # 初始化方法，用于创建一个新的TFBlipTextSelfAttention层实例
    def __init__(self, config, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 将传入的配置参数保存在实例中
        self.config = config
        # 创建TFBlipTextAttention层实例，命名为"attention"
        self.attention = TFBlipTextAttention(config, name="attention")
        # 如果当前实例是解码器，则创建TFBlipTextAttention层实例，命名为"crossattention"
        if self.config.is_decoder:
            self.crossattention = TFBlipTextAttention(
                config, is_cross_attention=self.config.is_decoder, name="crossattention"
            )
        # 创建TFBlipTextIntermediate层实例，命名为"intermediate"
        self.intermediate = TFBlipTextIntermediate(config, name="intermediate")
        # 创建TFBlipTextOutput层实例，命名为"output"
        self.self_output = TFBlipTextOutput(config, name="output")

    # 调用方法，实现了自定义层的前向传播逻辑
    def call(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        training=None,
    ):
        # 如果过去的键值对不为空，则获取解码器单向自注意力的缓存键值对
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 使用自注意力层处理隐藏状态
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            training=training,
        )
        # 获取自注意力的输出
        attention_output = self_attention_outputs[0]

        # 将注意力输出之外的其他输出保存在outputs中
        outputs = self_attention_outputs[1:-1]
        # 获取当前键值对
        present_key_value = self_attention_outputs[-1]

        # 如果存在编码器的隐藏状态，则进行交叉注意力计算
        if encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions=output_attentions,
                training=training,
            )
            # 获取交叉注意力的输出
            attention_output = cross_attention_outputs[0]
            # 如果输出注意力权重，则将交叉注意力的其他输出添加到outputs中
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights
        # 将注意力输出传递给中间层
        intermediate_output = self.intermediate(attention_output)
        # 将中间层的输出传递给输出层
        layer_output = self.self_output(intermediate_output, attention_output, training=training)
        # 将输出层的结果保存在outputs中
        outputs = (layer_output,) + outputs

        # 将当前键值对添加到outputs中
        outputs = outputs + (present_key_value,)

        # 返回outputs
        return outputs

    # 构建方法，用于构建自定义层
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 设置已构建标志位为True
        self.built = True
        # 构建注意力层
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        # 构建中间层
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        # 构建输出层
        if getattr(self, "self_output", None) is not None:
            with tf.name_scope(self.self_output.name):
                self.self_output.build(None)
        # 构建交叉注意力层
        if getattr(self, "crossattention", None) is not None:
            with tf.name_scope(self.crossattention.name):
                self.crossattention.build(None)
# 从给定链接的代码库中导入的 TensorFlow 序列化装饰器
@keras_serializable
# 定义 TFBlipTextEncoder 类，继承自 tf.keras.layers.Layer
class TFBlipTextEncoder(tf.keras.layers.Layer):
    # 指定该类所需的配置类为 BlipTextConfig
    config_class = BlipTextConfig

    # 初始化方法
    def __init__(self, config, name=None, **kwargs):
        # 调用父类的初始化方法
        super().__init__(name=name, **kwargs)
        # 保存配置对象
        self.config = config
        # 创建多个 TFBlipTextLayer 实例，存储在列表中
        self.layer = [TFBlipTextLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

    # call 方法用于前向传播
    @unpack_inputs
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
        training=None,
    ):
        # 如果要输出隐藏状态，则初始化存储所有隐藏状态的元组
        all_hidden_states = () if output_hidden_states else None
        # 如果要输出注意力，则初始化存储所有自注意力权重的元组
        all_self_attentions = () if output_attentions else None
        # 如果是解码器并且要输出注意力，则初始化存储所有交叉注意力权重的元组
        all_cross_attentions = () if output_attentions and self.config.is_decoder else None
        # 如果要使用缓存，则初始化存储下一个解码器缓存的元组
        next_decoder_cache = () if use_cache else None

        # 循环遍历所有 Transformer 层
        for i in range(self.config.num_hidden_layers):
            # 获取当前层模块
            layer_module = self.layer[i]
            # 如果要输出隐藏状态，则将当前隐藏状态添加到存储所有隐藏状态的元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 获取过去的键值对（如果存在）
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 调用当前层模块的前向传播方法
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
                training=training,
            )

            # 更新隐藏状态为当前层模块的输出
            hidden_states = layer_outputs[0]
            # 如果要使用缓存，则将当前层的输出添加到存储下一个解码器缓存的元组中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果要输出注意力，则将当前层的注意力权重添加到相应的元组中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果要输出隐藏状态，则将最终的隐藏状态添加到存储所有隐藏状态的元组中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典，则返回元组
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
        # 返回 TFBaseModelOutputWithPastAndCrossAttentions 类的实例，封装了模型的输出
        return TFBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
    # 构建神经网络模型
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 检查是否存在子层
        if getattr(self, "layer", None) is not None:
            # 遍历每个子层
            for layer in self.layer:
                # 使用子层的名称创建命名空间
                with tf.name_scope(layer.name):
                    # 构建子层
                    layer.build(None)
# 从transformers.models.bert.modeling_tf_bert.TFBertPooler复制代码，并将Bert->BlipText
class TFBlipTextPooler(tf.keras.layers.Layer):
    def __init__(self, config: BlipTextConfig, **kwargs):
        super().__init__(**kwargs)

        # 定义一个全连接层，用于池化模型，其中包括了隐藏层单元数、初始化方法和激活函数
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 我们通过简单地取第一个标记对应的隐藏状态来“池化”模型。
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(inputs=first_token_tensor)

        return pooled_output

    def build(self, input_shape=None):
        # 如果已经构建，则返回
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# 从transformers.models.bert.modeling_tf_bert.TFBertPredictionHeadTransform复制代码，并将Bert->BlipText
class TFBlipTextPredictionHeadTransform(tf.keras.layers.Layer):
    def __init__(self, config: BlipTextConfig, **kwargs):
        super().__init__(**kwargs)

        # 定义一个全连接层，用于转换预测头，其中包括了隐藏层单元数、初始化方法和激活函数
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )

        # 如果隐藏激活函数是字符串类型，则将其转换为相应的激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.transform_act_fn = config.hidden_act

        # 应用层归一化
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 将隐藏状态通过全连接层进行变换
        hidden_states = self.dense(inputs=hidden_states)
        # 应用激活函数
        hidden_states = self.transform_act_fn(hidden_states)
        # 应用层归一化
        hidden_states = self.LayerNorm(inputs=hidden_states)

        return hidden_states

    def build(self, input_shape=None):
        # 如果已经构建，则返回
        if self.built:
            return
        self.built = True
        # 构建全连接层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 构建层归一化
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


class TFBlipTextLMPredictionHead(tf.keras.layers.Layer):
    # 初始化方法，用于创建新的实例
    def __init__(self, config, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 创建文本预测头变换实例
        self.transform = TFBlipTextPredictionHeadTransform(config, name="transform")

        # 输出权重与输入嵌入相同，但每个标记都有一个仅输出的偏置
        self.decoder = tf.keras.layers.Dense(
            config.vocab_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="decoder",
            use_bias=False,
        )
        # 保存配置
        self.config = config

    # 构建方法，用于构建层的内部逻辑
    def build(self, input_shape=None):
        # 添加偏置权重
        self.bias = self.add_weight(name="bias", shape=(self.config.vocab_size,), initializer="zeros", trainable=True)

        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 构建变换层
        if getattr(self, "transform", None) is not None:
            with tf.name_scope(self.transform.name):
                self.transform.build(None)
        # 构建解码器层
        if getattr(self, "decoder", None) is not None:
            with tf.name_scope(self.decoder.name):
                self.decoder.build([None, None, self.config.hidden_size])

    # 调用方法，用于调用层以执行其逻辑
    def call(self, hidden_states):
        # 对隐藏状态进行变换
        hidden_states = self.transform(hidden_states)
        # 使用解码器进行解码，并添加偏置
        hidden_states = self.decoder(hidden_states) + self.bias
        # 返回处理后的隐藏状态
        return hidden_states
# 定义一个名为TFBlipTextOnlyMLMHead的TensorFlow层
class TFBlipTextOnlyMLMHead(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 实例化TFBlipTextLMPredictionHead对象
        self.predictions = TFBlipTextLMPredictionHead(config, name="predictions")

    # 对输入的序列输出进行处理
    def call(self, sequence_output: tf.Tensor) -> tf.Tensor:
        # 调用predictions对象，产生预测分数
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

    # 构建层
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True
        # 如果存在predictions属性
        if getattr(self, "predictions", None) is not None:
            with tf.name_scope(self.predictions.name):
                # 构建predictions对象
                self.predictions.build(None)


# 从给定链接处调整而来
class TFBlipTextPreTrainedModel(TFPreTrainedModel):
    """
    一个处理权重初始化和一个简单的接口以下载和加载预训练模型的抽象类。
    """

    # 使用BlipTextConfig作为配置类
    config_class = BlipTextConfig
    # 模型的基础名前缀
    base_model_prefix = "bert"
    # 在加载过程中忽略的键列表
    _keys_to_ignore_on_load_missing = [r"position_ids"]


# 从给定链接处调整而来
class TFBlipTextModel(TFBlipTextPreTrainedModel):
    """
    该模型可以作为编码器（只有自注意力）或解码器，此时在自注意力层之间添加了一个交叉注意力层，遵循[Attention is
    all you need](https://arxiv.org/abs/1706.03762)中描述的架构 by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser 和 Illia Polosukhin. 参数和`is_decoder`设为`True`; 随后预计在前向传递中输入`encoder_hidden_states`。
    """

    # 初始化模型
    def __init__(self, config, add_pooling_layer=True, name=None, **kwargs):
        super().__init__(config, name=name, **kwargs)
        self.config = config

        # 实例化嵌入层
        self.embeddings = TFBlipTextEmbeddings(config, name="embeddings")
        # 实例化编码器层
        self.encoder = TFBlipTextEncoder(config, name="encoder")
        # 如果需要添加汇聚层，则实例化汇聚层
        self.pooler = TFBlipTextPooler(config, name="pooler") if add_pooling_layer else None

    # 获取输入的嵌入层
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入的嵌入层
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 获取扩展的注意力掩码
    @tf.function
    def get_extended_attention_mask(
        self, attention_mask: tf.Tensor, input_shape: Tuple[int], is_decoder: bool
    @add_start_docstrings_to_model_forward(BLIP_TEXT_INPUTS_DOCSTRING)
    @unpack_inputs
    # 定义一个方法，用于调用模型
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 输入的 token IDs
        attention_mask: tf.Tensor | None = None,  # 注意力掩码
        position_ids: tf.Tensor | None = None,  # 位置 IDs
        head_mask: tf.Tensor | None = None,  # 头部掩码
        inputs_embeds: tf.Tensor | None = None,  # 输入的嵌入向量
        encoder_embeds: tf.Tensor | None = None,  # 编码器嵌入向量对应的张量
        encoder_hidden_states: tf.Tensor | None = None,  # 编码器隐藏状态
        encoder_attention_mask: tf.Tensor | None = None,  # 编码器注意力掩码
        past_key_values: Tuple[Tuple[tf.Tensor]] | None = None,  # 过去的键值对
        use_cache: bool | None = None,  # 是否使用缓存
        output_attentions: bool | None = None,  # 是否输出注意力
        output_hidden_states: bool | None = None,  # 是否输出隐藏状态
        return_dict: bool | None = None,  # 是否返回字典
        is_decoder: bool = False,  # 是否为解码器
        training: bool = False,  # 是否为训练模式
    # 构建模型
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在嵌入层，则构建嵌入层
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        # 如果存在编码器，则构建编码器
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果存在池化层，则构建池化层
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build(None)
# 从指定链接处的代码库中适配而来，定义了一个基于BLIP的文本语言模型的TensorFlow版本
class TFBlipTextLMHeadModel(TFBlipTextPreTrainedModel):
    # 加载时要忽略的意外键列表
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    # 加载时要忽略的缺失键列表
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    # 初始化模型
    def __init__(self, config, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, **kwargs)

        # 创建 BLIP 文本模型，不添加池化层
        self.bert = TFBlipTextModel(config, add_pooling_layer=False, name="bert")
        # 创建 BLIP 文本模型的 MLM 头部
        self.cls = TFBlipTextOnlyMLMHead(config, name="cls")

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 模型前向传播方法，包含模型的输入说明和解包输入
    @add_start_docstrings_to_model_forward(BLIP_TEXT_INPUTS_DOCSTRING)
    @unpack_inputs
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_logits=False,
        is_decoder=True,
        training=None,
    ):
        # 准备生成的输入
        def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        # 获取输入形状
        input_shape = input_ids.shape
        # 如果模型作为编码器-解码器模型中的解码器使用，则动态创建解码器注意力掩码
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # 如果使用过去的键值对，则截取解码器输入的 input_ids
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "encoder_hidden_states": model_kwargs.get("encoder_hidden_states", None),
            "encoder_attention_mask": model_kwargs.get("encoder_attention_mask", None),
            "is_decoder": True,
        }

    # 重新排序缓存
    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

    # 构建模型
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在 self.bert，则构建 self.bert
        if getattr(self, "bert", None) is not None:
            with tf.name_scope(self.bert.name):
                self.bert.build(None)
        # 如果存在 self.cls，则构建 self.cls
        if getattr(self, "cls", None) is not None:
            with tf.name_scope(self.cls.name):
                self.cls.build(None)
```