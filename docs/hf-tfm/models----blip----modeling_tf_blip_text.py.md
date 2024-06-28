# `.\models\blip\modeling_tf_blip_text.py`

```py
# 导入所需的库和模块
from __future__ import annotations

import math  # 导入数学库，用于数学运算
from typing import Optional, Tuple  # 导入类型提示相关模块

import tensorflow as tf  # 导入 TensorFlow 库

# 导入模型输出相关的类和函数
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFBaseModelOutputWithPoolingAndCrossAttentions,
    TFCausalLMOutputWithCrossAttentions,
)
# 导入 TensorFlow 下的实用工具函数和类
from ...modeling_tf_utils import (
    TFModelInputType,
    TFPreTrainedModel,
    get_initializer,
    get_tf_activation,
    keras,
    keras_serializable,
    shape_list,
    unpack_inputs,
)
# 导入一些 TensorFlow 下的实用函数，用于注意力机制和序列处理
from ...tf_utils import check_embeddings_within_bounds, invert_attention_mask, stable_softmax
# 导入通用工具函数和 logging 工具
from ...utils import add_start_docstrings_to_model_forward, logging
# 导入模型配置文件相关的类
from .configuration_blip import BlipTextConfig

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 用于文档字符串的 BLIP 文本输入说明
BLIP_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            # 输入序列的标记索引在词汇表中的位置。默认情况下，将忽略填充部分。

            # 可以使用 [`AutoProcessor`] 获得这些索引。有关详情，请参见 [`BlipProcessor.__call__`]。

            # [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 遮罩，用于在填充的标记索引上避免执行注意力操作。遮罩的取值范围为 `[0, 1]`：

            # - 1 表示**未被遮罩**的标记，
            # - 0 表示**被遮罩**的标记。

            # [什么是注意力遮罩？](../glossary#attention-mask)
        position_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 每个输入序列标记在位置嵌入中的位置索引。取值范围为 `[0, config.max_position_embeddings - 1]`。

            # [什么是位置 ID？](../glossary#position-ids)
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多细节，请参见返回的张量中的 `attentions`。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多细节，请参见返回的张量中的 `hidden_states`。

        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通的元组。
# 从 https://github.com/salesforce/BLIP/blob/main/models/med.py#L52 适配而来的代码，该类定义了一个 TFBlipTextEmbeddings 类
class TFBlipTextEmbeddings(keras.layers.Layer):
    """Construct the embeddings from word and position embeddings."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 初始化词嵌入层，根据配置文件指定的词汇大小和隐藏大小
        self.word_embeddings = keras.layers.Embedding(
            config.vocab_size,
            config.hidden_size,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="word_embeddings",
        )
        # 初始化位置嵌入层，根据配置文件指定的最大位置嵌入和隐藏大小
        self.position_embeddings = keras.layers.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="position_embeddings",
        )

        # 使用 PyTorch 模型的变量命名风格，因此未使用蛇形命名法来定义 LayerNormalization 层，
        # 以便能够加载任何 TensorFlow 检查点文件
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 定义丢弃层，根据配置文件指定的隐藏层丢弃概率
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob, name="dropout")

        # 创建位置ID张量，用于表示绝对位置嵌入的位置
        self.position_ids = tf.expand_dims(tf.range(config.max_position_embeddings), 0)
        # 获取配置文件中的位置嵌入类型，默认为"absolute"
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        # 保存配置对象
        self.config = config

    def call(self, input_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0, training=None):
        # 如果传入了 input_ids，则获取其形状
        if input_ids is not None:
            input_shape = tf.shape(input_ids)
        else:
            # 否则获取 inputs_embeds 的形状，但不包括最后一个维度
            input_shape = tf.shape(inputs_embeds)[:-1]

        # 获取序列长度
        seq_length = input_shape[1]

        # 如果未提供 position_ids，则从预测键值长度到序列长度获取位置ID
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # 如果未提供 inputs_embeds，则根据 input_ids 检查嵌入是否在有效范围内，并获取词嵌入
        if inputs_embeds is None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = self.word_embeddings(input_ids)

        # 获取嵌入
        embeddings = inputs_embeds

        # 如果位置嵌入类型为"absolute"，则获取位置嵌入并加到嵌入中
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        # 对嵌入进行 LayerNorm 处理
        embeddings = self.LayerNorm(embeddings)
        # 使用丢弃层进行丢弃处理，根据训练状态决定是否启用丢弃
        embeddings = self.dropout(embeddings, training=training)
        
        # 返回最终的嵌入表示
        return embeddings
    # 如果模型已经构建完成，则直接返回，不进行重复构建
    if self.built:
        return
    # 设置标志位，表示模型已经构建完成
    self.built = True
    
    # 如果存在词嵌入层对象，构建该层
    if getattr(self, "word_embeddings", None) is not None:
        # 在 TensorFlow 中使用命名空间为词嵌入层命名，并构建该层
        with tf.name_scope(self.word_embeddings.name):
            self.word_embeddings.build(None)
    
    # 如果存在位置嵌入层对象，构建该层
    if getattr(self, "position_embeddings", None) is not None:
        # 在 TensorFlow 中使用命名空间为位置嵌入层命名，并构建该层
        with tf.name_scope(self.position_embeddings.name):
            self.position_embeddings.build(None)
    
    # 如果存在 LayerNorm 层对象，构建该层
    if getattr(self, "LayerNorm", None) is not None:
        # 在 TensorFlow 中使用命名空间为 LayerNorm 层命名，并构建该层
        with tf.name_scope(self.LayerNorm.name):
            # 构建 LayerNorm 层，传入输入形状 [None, None, self.config.hidden_size]
            self.LayerNorm.build([None, None, self.config.hidden_size])
    
    # 如果存在 dropout 层对象，构建该层
    if getattr(self, "dropout", None) is not None:
        # 在 TensorFlow 中使用命名空间为 dropout 层命名，并构建该层
        with tf.name_scope(self.dropout.name):
            self.dropout.build(None)
# Adapted from https://github.com/salesforce/BLIP/blob/main/models/med.py#L97
class TFBlipTextSelfAttention(keras.layers.Layer):
    def __init__(self, config, is_cross_attention, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        # 检查隐藏层大小是否能被注意力头数整除，如果不能且没有嵌入大小的属性，则引发错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 创建用于查询、键、值的全连接层，初始化方法为指定范围的初始值
        self.query = keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        self.key = keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        self.value = keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )

        # 添加 dropout 层，使用配置中的注意力概率
        self.dropout = keras.layers.Dropout(config.attention_probs_dropout_prob)
        
        # 根据配置选择位置嵌入类型，默认为绝对位置
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 如果位置嵌入类型为相对键或相对键查询，则创建距离嵌入层
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = keras.layers.Embedding(
                2 * config.max_position_embeddings - 1, self.attention_head_size
            )
        self.is_cross_attention = is_cross_attention

    # 调整张量形状以便计算注意力分数
    def transpose_for_scores(self, x):
        new_x_shape = tf.concat(
            [tf.shape(x)[:-1], tf.constant([self.num_attention_heads, self.attention_head_size], dtype=tf.int32)],
            axis=0,
        )
        x = tf.reshape(x, new_x_shape)
        return tf.transpose(x, perm=(0, 2, 1, 3))

    # 定义层的调用方法，接收多个参数用于注意力计算和输出
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
    # 如果已经构建过网络结构，则直接返回，不再重复构建
    def build(self, input_shape=None):
        if self.built:
            return
        
        # 标记网络已经构建
        self.built = True
        
        # 如果存在查询（query）模块，则构建其网络结构
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.hidden_size])
        
        # 如果是交叉注意力机制，构建键（key）和值（value）的网络结构
        if self.is_cross_attention:
            # 如果存在键（key）模块，则构建其网络结构
            if getattr(self, "key", None) is not None:
                with tf.name_scope(self.key.name):
                    self.key.build([None, None, self.config.encoder_hidden_size])
            # 如果存在值（value）模块，则构建其网络结构
            if getattr(self, "value", None) is not None:
                with tf.name_scope(self.value.name):
                    self.value.build([None, None, self.config.encoder_hidden_size])
        else:
            # 如果存在键（key）模块，则构建其网络结构
            if getattr(self, "key", None) is not None:
                with tf.name_scope(self.key.name):
                    self.key.build([None, None, self.config.hidden_size])
            # 如果存在值（value）模块，则构建其网络结构
            if getattr(self, "value", None) is not None:
                with tf.name_scope(self.value.name):
                    self.value.build([None, None, self.config.hidden_size])
# TFBlipTextSelfOutput 类定义，继承自 keras.layers.Layer
class TFBlipTextSelfOutput(keras.layers.Layer):
    # 初始化方法，接收 BlipTextConfig 类型的 config 对象和其他关键字参数
    def __init__(self, config: BlipTextConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个 Dense 层，用于线性变换，units 参数为 config.hidden_size，使用指定的初始化器初始化权重
        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # 创建 LayerNormalization 层，epsilon 参数为 config.layer_norm_eps，用于归一化
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")

        # 创建 Dropout 层，丢弃率为 config.hidden_dropout_prob，用于随机丢弃部分神经元的输出
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)

        # 将 config 对象保存为实例变量，用于后续调用
        self.config = config

    # call 方法重写，定义了层的正向传播逻辑
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        # 将 hidden_states 输入到 Dense 层进行线性变换
        hidden_states = self.dense(inputs=hidden_states)
        
        # 在训练时，对 hidden_states 使用 Dropout 进行随机丢弃部分神经元的输出
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        
        # 将经过 Dropout 处理后的 hidden_states 与 input_tensor 相加，然后输入到 LayerNorm 层进行归一化处理
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        # 返回处理后的 hidden_states
        return hidden_states

    # build 方法用于构建层，在第一次调用时构建层的权重
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        
        # 将标志 built 设置为 True，表示已构建
        self.built = True
        
        # 如果实例中存在 dense 层，使用 tf.name_scope 创建命名空间，并构建 dense 层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        
        # 如果实例中存在 LayerNorm 层，使用 tf.name_scope 创建命名空间，并构建 LayerNorm 层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# 从 https://github.com/salesforce/BLIP/blob/main/models/med.py#242 适配而来
# TFBlipTextAttention 类定义，继承自 keras.layers.Layer
class TFBlipTextAttention(keras.layers.Layer):
    # 初始化方法，接收 config 和 is_cross_attention 参数以及其他关键字参数
    def __init__(self, config, is_cross_attention=False, **kwargs):
        super().__init__(**kwargs)
        
        # 创建 TFBlipTextSelfAttention 类实例 self.self，用于自注意力计算，is_cross_attention 表示是否是跨注意力
        self.self = TFBlipTextSelfAttention(config, is_cross_attention, name="self")
        
        # 创建 TFBlipTextSelfOutput 类实例 self.self_output，用于自注意力层的输出处理
        self.self_output = TFBlipTextSelfOutput(config, name="output")

    # call 方法重写，定义了层的正向传播逻辑
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
        # 调用 self.self 的 call 方法进行自注意力计算，得到 self_outputs
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
        
        # 将 self_outputs[0] 作为输入，hidden_states 作为 input_tensor，传入 self.self_output 进行处理
        attention_output = self.self_output(self_outputs[0], hidden_states, training=training)
        
        # 构建输出元组，如果需要输出 attentions，将 attentions 添加到 outputs 中
        outputs = (attention_output,) + self_outputs[1:]  # 如果输出 attentions，则添加到 outputs 中
        
        # 返回 outputs
        return outputs
    # 如果模型已经构建完成，直接返回，不做任何操作
    if self.built:
        return
    # 标记模型已经构建
    self.built = True
    # 检查是否存在self属性，并且不为None
    if getattr(self, "self", None) is not None:
        # 使用self的名称创建一个命名空间，并在其中构建self对象
        with tf.name_scope(self.self.name):
            self.self.build(None)
    # 检查是否存在self_output属性，并且不为None
    if getattr(self, "self_output", None) is not None:
        # 使用self_output的名称创建一个命名空间，并在其中构建self_output对象
        with tf.name_scope(self.self_output.name):
            self.self_output.build(None)
# Copied from transformers.models.bert.modeling_tf_bert.TFBertIntermediate with Bert->BlipText
class TFBlipTextIntermediate(keras.layers.Layer):
    def __init__(self, config: BlipTextConfig, **kwargs):
        super().__init__(**kwargs)

        # 定义一个全连接层，用于转换输入的隐藏状态到中间层大小
        self.dense = keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # 根据配置选择激活函数，如果是字符串则转换为对应的 TensorFlow 激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 应用全连接层到输入的隐藏状态
        hidden_states = self.dense(inputs=hidden_states)
        # 应用中间层激活函数到全连接层的输出
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果层已经构建且存在密集层，则构建密集层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


class TFBlipTextOutput(keras.layers.Layer):
    def __init__(self, config: BlipTextConfig, **kwargs):
        super().__init__(**kwargs)

        # 定义一个全连接层，用于转换输入的隐藏状态到输出大小
        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 应用层归一化，用于调整输出层的数据分布
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 应用丢弃层，用于随机丢弃部分神经元以防止过拟合
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 应用全连接层到输入的隐藏状态
        hidden_states = self.dense(inputs=hidden_states)
        # 如果训练模式开启，则应用丢弃层，否则跳过
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 将输入张量和归一化后的隐藏状态相加，并应用层归一化
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果层已经构建且存在密集层，则构建密集层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])
        # 如果层已经构建且存在层归一化层，则构建层归一化层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


class TFBlipTextLayer(keras.layers.Layer):
    # 初始化函数，用于创建一个新的TFBlipTextLayer对象
    def __init__(self, config, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 将传入的config参数保存到对象的config属性中
        self.config = config
        # 创建一个TFBlipTextAttention对象，并命名为"attention"
        self.attention = TFBlipTextAttention(config, name="attention")
        # 如果config中指定为decoder模式，则创建一个用于跨attention的TFBlipTextAttention对象，并命名为"crossattention"
        if self.config.is_decoder:
            self.crossattention = TFBlipTextAttention(
                config, is_cross_attention=self.config.is_decoder, name="crossattention"
            )
        # 创建一个TFBlipTextIntermediate对象，并命名为"intermediate"
        self.intermediate = TFBlipTextIntermediate(config, name="intermediate")
        # 创建一个TFBlipTextOutput对象，并命名为"output"
        self.self_output = TFBlipTextOutput(config, name="output")

    # call方法，用于执行前向传播操作
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
        # 如果存在过去的key/value，则从中获取decoder单向self-attention的缓存
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 执行self attention操作，并获取输出结果
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            training=training,
        )
        # 从self attention输出中提取注意力输出
        attention_output = self_attention_outputs[0]

        # 提取除了注意力输出以外的所有输出
        outputs = self_attention_outputs[1:-1]
        # 获取当前的key/value
        present_key_value = self_attention_outputs[-1]

        # 如果存在encoder的隐藏状态，则执行cross attention操作
        if encoder_hidden_states is not None:
            # 执行cross attention操作，并获取输出结果
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions=output_attentions,
                training=training,
            )
            # 从cross attention输出中提取注意力输出
            attention_output = cross_attention_outputs[0]
            # 如果输出注意力权重，则将cross attention的输出添加到已有的outputs中
            outputs = outputs + cross_attention_outputs[1:-1]

        # 执行intermediate层的操作
        intermediate_output = self.intermediate(attention_output)
        # 执行self output层的操作，并获取最终的层输出
        layer_output = self.self_output(intermediate_output, attention_output, training=training)
        # 将最终的层输出和之前的outputs一起返回
        outputs = (layer_output,) + outputs

        # 将当前的key/value添加到outputs中
        outputs = outputs + (present_key_value,)

        # 返回所有的outputs
        return outputs

    # build方法，用于构建层，并确保每个组件被正确构建
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 将构建状态标记为已构建
        self.built = True
        # 如果存在attention对象，则构建attention对象
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        # 如果存在intermediate对象，则构建intermediate对象
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        # 如果存在self_output对象，则构建self_output对象
        if getattr(self, "self_output", None) is not None:
            with tf.name_scope(self.self_output.name):
                self.self_output.build(None)
        # 如果存在crossattention对象，则构建crossattention对象
        if getattr(self, "crossattention", None) is not None:
            with tf.name_scope(self.crossattention.name):
                self.crossattention.build(None)
# 基于 keras_serializable 装饰器将该类声明为可序列化的 Keras 层
@keras_serializable
class TFBlipTextEncoder(keras.layers.Layer):
    # 指定配置类为 BlipTextConfig
    config_class = BlipTextConfig

    # 初始化方法，接受配置对象 config 和可选的名称参数
    def __init__(self, config, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        # 将传入的配置对象保存为实例变量
        self.config = config
        # 创建一个由 TFBlipTextLayer 实例组成的列表，命名为 layer，用于表示隐藏层
        self.layer = [TFBlipTextLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

    # 使用 unpack_inputs 装饰器定义 call 方法，接收多个输入参数并进行处理
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
        # 如果设置了 output_hidden_states 标志，则初始化 all_hidden_states 为一个空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果设置了 output_attentions 标志，则初始化 all_self_attentions 为一个空元组
        all_self_attentions = () if output_attentions else None
        # 如果是解码器且设置了 output_attentions 标志，则初始化 all_cross_attentions 为一个空元组
        all_cross_attentions = () if output_attentions and self.config.is_decoder else None

        # 如果设置了 use_cache 标志，则初始化 next_decoder_cache 为一个空元组
        next_decoder_cache = () if use_cache else None

        # 循环遍历每个隐藏层
        for i in range(self.config.num_hidden_layers):
            # 获取当前层的模块对象
            layer_module = self.layer[i]
            # 如果设置了 output_hidden_states 标志，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 获取当前层的过去键值
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 调用当前层的模块对象进行前向传播
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

            # 更新隐藏状态为当前层输出的第一个元素
            hidden_states = layer_outputs[0]
            # 如果设置了 use_cache 标志，则将当前层输出的最后一个元素添加到 next_decoder_cache 中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果设置了 output_attentions 标志，则将当前层输出的注意力值添加到对应的元组中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果设置了 output_hidden_states 标志，则将最终隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果 return_dict 为 False，则返回一个元组，其中包含非 None 的值
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
        
        # 如果 return_dict 为 True，则返回 TFBaseModelOutputWithPastAndCrossAttentions 对象
        return TFBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
    # 定义 build 方法，用于构建模型的层
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回，不再重复构建
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        # 检查是否存在 layer 属性，并对其进行迭代
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                # 使用 tf.name_scope 为当前层设置命名空间，确保命名唯一性
                with tf.name_scope(layer.name):
                    # 调用每一层的 build 方法，传入 input_shape 为 None
                    layer.build(None)
# Copied from transformers.models.bert.modeling_tf_bert.TFBertPooler with Bert->BlipText
class TFBlipTextPooler(keras.layers.Layer):
    def __init__(self, config: BlipTextConfig, **kwargs):
        super().__init__(**kwargs)

        # 定义一个全连接层，用于池化操作，输出维度为配置文件中的隐藏大小
        self.dense = keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 从输入的隐藏状态中获取第一个 token 的隐藏状态
        first_token_tensor = hidden_states[:, 0]
        # 将第一个 token 的隐藏状态输入到全连接层中进行池化操作
        pooled_output = self.dense(inputs=first_token_tensor)

        return pooled_output

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果定义了 dense 层，则在 tf 的命名空间下构建该层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# Copied from transformers.models.bert.modeling_tf_bert.TFBertPredictionHeadTransform with Bert->BlipText
class TFBlipTextPredictionHeadTransform(keras.layers.Layer):
    def __init__(self, config: BlipTextConfig, **kwargs):
        super().__init__(**kwargs)

        # 定义一个全连接层，用于预测头部转换，输出维度为配置文件中的隐藏大小
        self.dense = keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )

        # 根据配置文件中的激活函数类型，选择相应的激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.transform_act_fn = config.hidden_act

        # 定义 LayerNormalization 层，用于规范化隐藏状态
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 将输入的隐藏状态输入到全连接层中
        hidden_states = self.dense(inputs=hidden_states)
        # 应用预定义的激活函数
        hidden_states = self.transform_act_fn(hidden_states)
        # 将处理后的隐藏状态输入到 LayerNormalization 层中进行规范化
        hidden_states = self.LayerNorm(inputs=hidden_states)

        return hidden_states

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果定义了 dense 层，则在 tf 的命名空间下构建该层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果定义了 LayerNorm 层，则在 tf 的命名空间下构建该层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


class TFBlipTextLMPredictionHead(keras.layers.Layer):
    # 这里是 TFBlipTextLMPredictionHead 类的定义，需要进一步补充注释
    pass
    # 初始化函数，接收配置参数和可选的关键字参数
    def __init__(self, config, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 使用配置初始化一个文本预测头部的转换器对象
        self.transform = TFBlipTextPredictionHeadTransform(config, name="transform")

        # 输出权重与输入嵌入相同，但每个标记有一个仅输出的偏置项
        # 创建一个全连接层，输出大小为词汇表大小，使用指定的初始化器初始化权重
        self.decoder = keras.layers.Dense(
            config.vocab_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="decoder",
            use_bias=False,
        )
        # 存储配置对象
        self.config = config

    # 构建模型
    def build(self, input_shape=None):
        # 添加一个名为bias的可训练权重，形状为词汇表大小，初始化为全零
        self.bias = self.add_weight(name="bias", shape=(self.config.vocab_size,), initializer="zeros", trainable=True)

        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True

        # 如果存在transform属性，则构建transform层
        if getattr(self, "transform", None) is not None:
            with tf.name_scope(self.transform.name):
                self.transform.build(None)

        # 如果存在decoder属性，则构建decoder层
        if getattr(self, "decoder", None) is not None:
            with tf.name_scope(self.decoder.name):
                # 构建时指定decoder层的输入形状为[None, None, 隐藏大小]
                self.decoder.build([None, None, self.config.hidden_size])

    # 调用模型，传入隐藏状态并返回预测结果
    def call(self, hidden_states):
        # 使用transform层处理隐藏状态
        hidden_states = self.transform(hidden_states)
        # 使用decoder层处理transform后的隐藏状态，并加上偏置
        hidden_states = self.decoder(hidden_states) + self.bias
        # 返回处理后的隐藏状态，即预测结果
        return hidden_states
class TFBlipTextOnlyMLMHead(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 使用给定的配置创建 BLIP 文本预测头部对象
        self.predictions = TFBlipTextLMPredictionHead(config, name="predictions")

    def call(self, sequence_output: tf.Tensor) -> tf.Tensor:
        # 基于序列输出进行预测分数计算
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果 predictions 属性存在，则构建预测头部对象
        if getattr(self, "predictions", None) is not None:
            with tf.name_scope(self.predictions.name):
                self.predictions.build(None)


# Adapted from https://github.com/salesforce/BLIP/blob/main/models/med.py#L548
class TFBlipTextPreTrainedModel(TFPreTrainedModel):
    """
    处理权重初始化和预训练模型下载加载的抽象类，提供简单的接口。
    """

    config_class = BlipTextConfig
    base_model_prefix = "bert"
    _keys_to_ignore_on_load_missing = [r"position_ids"]


# Adapted from https://github.com/salesforce/BLIP/blob/3a29b7410476bf5f2ba0955827390eb6ea1f4f9d/models/med.py#L571
class TFBlipTextModel(TFBlipTextPreTrainedModel):
    """
    该模型可以作为编码器（仅具有自注意力）或解码器行事，在后一种情况下，将在自注意力层之间添加交叉注意力层，遵循
    [Attention is all you need](https://arxiv.org/abs/1706.03762) 的架构描述。
    """

    def __init__(self, config, add_pooling_layer=True, name=None, **kwargs):
        super().__init__(config, name=name, **kwargs)
        self.config = config

        # 创建 BLIP 文本嵌入层、编码层和池化层（如果需要）
        self.embeddings = TFBlipTextEmbeddings(config, name="embeddings")
        self.encoder = TFBlipTextEncoder(config, name="encoder")
        self.pooler = TFBlipTextPooler(config, name="pooler") if add_pooling_layer else None

    def get_input_embeddings(self):
        # 返回输入嵌入层的权重
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        # 设置输入嵌入层的权重值
        self.embeddings.word_embeddings = value

    @tf.function
    def get_extended_attention_mask(
        self, attention_mask: tf.Tensor, input_shape: Tuple[int], is_decoder: bool
    ):
        # 返回扩展的注意力遮罩张量
        pass

    @add_start_docstrings_to_model_forward(BLIP_TEXT_INPUTS_DOCSTRING)
    @unpack_inputs
    # 添加开始的文档字符串到模型前向传递
    # 定义一个方法 `call`，用于执行模型的前向传播。
    # 参数说明：
    # input_ids: 输入的 token IDs
    # attention_mask: 注意力遮罩张量
    # position_ids: 位置 ID 张量
    # head_mask: 头部遮罩张量
    # inputs_embeds: 输入的嵌入张量
    # encoder_embeds: 编码器的嵌入张量
    # encoder_hidden_states: 编码器的隐藏状态张量
    # encoder_attention_mask: 编码器的注意力遮罩张量
    # past_key_values: 缓存的键值对元组
    # use_cache: 是否使用缓存
    # output_attentions: 是否输出注意力权重
    # output_hidden_states: 是否输出隐藏状态
    # return_dict: 是否返回字典形式的结果
    # is_decoder: 是否作为解码器运行
    # training: 是否在训练模式下运行

    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果 `embeddings` 属性存在，则构建 `embeddings` 层
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        # 如果 `encoder` 属性存在，则构建 `encoder` 层
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果 `pooler` 属性存在，则构建 `pooler` 层
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build(None)
# Adapted from https://github.com/salesforce/BLIP/blob/main/models/med.py#L811
class TFBlipTextLMHeadModel(TFBlipTextPreTrainedModel):
    # 在加载模型时忽略的不期望键名列表
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    # 在加载模型时忽略的丢失键名列表
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        # 初始化BERT模型，不包含池化层
        self.bert = TFBlipTextModel(config, add_pooling_layer=False, name="bert")
        # 初始化仅包含MLM头部的模型
        self.cls = TFBlipTextOnlyMLMHead(config, name="cls")
        # 获取标签平滑参数
        self.label_smoothing = config.label_smoothing

    # 获取输出的嵌入层
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出的嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 调用函数，按照模型前向传播的文档字符串注释，解包输入参数
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
        input_shape = input_ids.shape
        # 如果没有提供注意力掩码，则创建一个全为1的掩码
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # 如果使用了过去的键值对，则截断输入的input_ids
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

    # 重新排序缓存，以便于beam search
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
        # 如果存在BERT模型，则构建BERT
        if getattr(self, "bert", None) is not None:
            with tf.name_scope(self.bert.name):
                self.bert.build(None)
        # 如果存在CLS模型，则构建CLS
        if getattr(self, "cls", None) is not None:
            with tf.name_scope(self.cls.name):
                self.cls.build(None)
```