# `.\models\electra\modeling_tf_electra.py`

```py
# 设置编码为 utf-8
# 版权声明
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的要求，否则您不得使用此文件
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或经书面同意，否则软件按"原样"分发，
# 没有任何明示或暗示的担保或条件
# 请参见特定语言权限和限制的许可证
# TF Electra 模型

# 导入必要的库
from __future__ import annotations
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFMaskedLMOutput,
    TFMultipleChoiceModelOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (
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
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_electra import ElectraConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点名称
_CHECKPOINT_FOR_DOC = "google/electra-small-discriminator"
# 用于文档的配置名称
_CONFIG_FOR_DOC = "ElectraConfig"
# TF Electra 预训练模型存档列表
TF_ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/electra-small-generator",
    "google/electra-base-generator",
    "google/electra-large-generator",
    "google/electra-small-discriminator",
    "google/electra-base-discriminator",
    "google/electra-large-discriminator",
    # 查看所有 ELECTRA 模型 https://huggingface.co/models?filter=electra
]

# 从 transformers.models.bert.modeling_tf_bert.TFBertSelfAttention 中复制的 TFElectraSelfAttention 类
class TFElectraSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config: ElectraConfig, **kwargs):
        # 调用父类的构造函数
        super().__init__(**kwargs)

        # 检查隐藏层大小是否能被注意力头的数量整除
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        # 设置注意力头的数量
        self.num_attention_heads = config.num_attention_heads
        # 计算每个注意力头的大小
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        # 计算所有注意力头的总大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # 计算注意力头大小的平方根
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        # 创建查询权重矩阵
        self.query = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        # 创建键权重矩阵
        self.key = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        # 创建值权重矩阵
        self.value = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )
        # 创建 dropout 层，用于处理注意力概率的 dropout
        self.dropout = tf.keras.layers.Dropout(rate=config.attention_probs_dropout_prob)

        # 标记是否为解码器
        self.is_decoder = config.is_decoder
        # 保存配置信息
        self.config = config

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # 将张量从 [batch_size, seq_length, all_head_size] 转换为 [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

        # 将张量从 [batch_size, seq_length, num_attention_heads, attention_head_size] 转置为 [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: tf.Tensor,
        encoder_attention_mask: tf.Tensor,
        past_key_value: Tuple[tf.Tensor],
        output_attentions: bool,
        training: bool = False,
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 标记已经构建过
        self.built = True
        # 如果存在查询权重矩阵，则构建查询权重矩阵
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.hidden_size])
        # 如果存在键权重矩阵，则构建键权重矩阵
        if getattr(self, "key", None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.config.hidden_size])
        # 如果存在值权重矩阵，则构建值权重矩阵
        if getattr(self, "value", None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.config.hidden_size])
# 定义一个名为 TFElectraSelfOutput 的自定义层，继承自 tf.keras.layers.Layer 类
# 参数包括一个 ElectraConfig 类型的 config 对象和额外的关键字参数
class TFElectraSelfOutput(tf.keras.layers.Layer):
    def __init__(self, config: ElectraConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，输出单元数为 config.hidden_size，使用指定的初始化器初始化权重矩阵，命名为"dense"
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个 LayerNormalization 层，epsilon 值设置为 config.layer_norm_eps，命名为"LayerNorm"
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建一个 Dropout 层，丢弃率为 config.hidden_dropout_prob
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        # 保存 config 对象
        self.config = config

    # 定义层的前向传播逻辑
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 将输入 hidden_states 通过全连接层 dense
        hidden_states = self.dense(inputs=hidden_states)
        # 对 hidden_states 进行丢弃操作
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 将 input_tensor 加上经过处理的 hidden_states，再进行 LayerNormalization
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    # 定义层的构建逻辑
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在 dense 层，则构建 dense 层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果存在 LayerNorm 层，则构建 LayerNorm 层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# 定义一个名为 TFElectraAttention 的自定义层，继承自 tf.keras.layers.Layer 类
# 参数包括一个 ElectraConfig 类型的 config 对象和额外的关键字参数
class TFElectraAttention(tf.keras.layers.Layer):
    def __init__(self, config: ElectraConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个 TFElectraSelfAttention 层，命名为"self"
        self.self_attention = TFElectraSelfAttention(config, name="self")
        # 创建一个 TFElectraSelfOutput 层，命名为"output"
        self.dense_output = TFElectraSelfOutput(config, name="output")

    # 定义重新分配注意力头部的方法
    def prune_heads(self, heads):
        raise NotImplementedError

    # 定义层的前向传播逻辑
    def call(
        self,
        input_tensor: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: tf.Tensor,
        encoder_attention_mask: tf.Tensor,
        past_key_value: Tuple[tf.Tensor],
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        # 使用 self_attention 层处理输入，并返回处理后的输出
        self_outputs = self.self_attention(
            hidden_states=input_tensor,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            training=training,
        )
        # 使用 dense_output 层处理 self_attention 的输出，并返回处理后的输出
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=input_tensor, training=training
        )
        # 如果输出注意力信息，则添加到输出中
        outputs = (attention_output,) + self_outputs[1:]

        return outputs
    # 根据输入形状构建模型
    def build(self, input_shape=None):
        # 如果模型已构建好，则直接返回
        if self.built:
            return
        # 设置模型的构建状态为已完成
        self.built = True
        # 如果存在自注意力模块，则构建自注意力模块
        if getattr(self, "self_attention", None) is not None:
            # 使用 TensorFlow 的命名空间定义自注意力模块的名称
            with tf.name_scope(self.self_attention.name):
                # 构建自注意力模块
                self.self_attention.build(None)
        # 如果存在密集输出模块，则构建密集输出模块
        if getattr(self, "dense_output", None) is not None:
            # 使用 TensorFlow 的命名空间定义密集输出模块的名称
            with tf.name_scope(self.dense_output.name):
                # 构建密集输出模块
                self.dense_output.build(None)
# 从transformers.models.bert.modeling_tf_bert.TFBertIntermediate复制代码，并将Bert->Electra
class TFElectraIntermediate(tf.keras.layers.Layer):
    def __init__(self, config: ElectraConfig, **kwargs):
        super().__init__(**kwargs)

        # 使用config中指定的参数创建一个全连接层（Dense）
        self.dense = tf.keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # 如果config中的hidden_act是字符串，则使用get_tf_activation函数获得激活函数，否则直接使用config中的hidden_act
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        self.config = config

    # 对输入的hidden_states进行全连接运算
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        # 使用中间激活函数处理全连接层的输出
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

    # 搭建层的结构
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建dense层的结构
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# 从transformers.models.bert.modeling_tf_bert.TFBertOutput复制代码，并将Bert->Electra
class TFElectraOutput(tf.keras.layers.Layer):
    def __init__(self, config: ElectraConfig, **kwargs):
        super().__init__(**kwargs)

        # 使用config中指定的参数创建一个全连接层（Dense）
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 使用config中指定的参数创建一个LayerNormalization层
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 使用config中指定的参数创建一个Dropout层
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    # 对输入的hidden_states进行全连接运算、dropout和LayerNormalization处理
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    # 搭建层的结构
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建dense层的结构
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])
        # 构建LayerNorm层的结构
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# 从transformers.models.bert.modeling_tf_bert.TFBertLayer复制代码，并将Bert->Electra
class TFElectraLayer(tf.keras.layers.Layer):
    # 初始化 ElectraModel 类的实例
    def __init__(self, config: ElectraConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 创建 ElectraAttention 层的实例，并命名为 "attention"
        self.attention = TFElectraAttention(config, name="attention")
        
        # 是否为解码器模型
        self.is_decoder = config.is_decoder
        
        # 是否添加跨注意力机制
        self.add_cross_attention = config.add_cross_attention
        
        # 如果添加了跨注意力机制，并且不是解码器模型，则抛出数值错误
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
        
        # 创建 TFElectraIntermediate 层的实例，并命名为 "intermediate"
        self.intermediate = TFElectraIntermediate(config, name="intermediate")
        
        # 创建 TFElectraOutput 层的实例，并命名为 "output"
        self.bert_output = TFElectraOutput(config, name="output")

    # 定义调用 ElectraModel 类的方法
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: tf.Tensor | None,
        encoder_attention_mask: tf.Tensor | None,
        past_key_value: Tuple[tf.Tensor] | None,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        # 定义decoder单向自注意力缓存键值元组为位置1, 2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 对输入的隐藏状态进行自注意力计算
        self_attention_outputs = self.attention(
            input_tensor=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=self_attn_past_key_value,
            output_attentions=output_attentions,
            training=training,
        )
        # 获取自注意力输出
        attention_output = self_attention_outputs[0]

        # 如果是解码器，最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # 如果输出注意力权重，则添加自注意力

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 定义交叉注意力缓存键值元组为过去键值元组的第3、4位置
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 计算交叉注意力
            cross_attention_outputs = self.crossattention(
                input_tensor=attention_output,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
                training=training,
            )
            # 获取交叉注意力输出
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # 如果输出注意力权重，则添加交叉注意力

            # 将交叉注意力缓存添加到present_key_value元组的第3、4位置
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 对attention_output应用一层中间输出的全连接层
        intermediate_output = self.intermediate(hidden_states=attention_output)
        # 对intermediate_output应���BERT输出层
        layer_output = self.bert_output(
            hidden_states=intermediate_output, input_tensor=attention_output, training=training
        )
        outputs = (layer_output,) + outputs  # 如果输出注意力，则添加之

        # 如果是解码器，将注意力的键值作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs
    # 构建函数，用于构建模型结构
    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回
        if self.built:
            return
        # 设置构建标志为 True
        self.built = True
        # 如果存在注意力机制对象，则构建其内部结构
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        # 如果存在中间层对象，则构建其内部结构
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        # 如果存在 BERT 输出对象，则构建其内部结构
        if getattr(self, "bert_output", None) is not None:
            with tf.name_scope(self.bert_output.name):
                self.bert_output.build(None)
        # 如果存在交叉注意力机制对象，则构建其内部结构
        if getattr(self, "crossattention", None) is not None:
            with tf.name_scope(self.crossattention.name):
                self.crossattention.build(None)
# 从transformers.models.bert.modeling_tf_bert.TFBertEncoder复制并修改为Bert->Electra
class TFElectraEncoder(tf.keras.layers.Layer):
    def __init__(self, config: ElectraConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.layer = [TFElectraLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: tf.Tensor | None,
        encoder_attention_mask: tf.Tensor | None,
        past_key_values: Tuple[Tuple[tf.Tensor]] | None,
        use_cache: Optional[bool],
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPastAndCrossAttentions, Tuple[tf.Tensor]]:
        # 初始化用于存储所有隐藏状态的变量，如果需要输出隐藏状态的话
        all_hidden_states = () if output_hidden_states else None
        # 初始化用于存储所有注意力权重的变量，如果需要输出注意力权重的话
        all_attentions = () if output_attentions else None
        # 初始化用于存储所有跨层注意力权重的变量，如果需要输出跨层注意力权重且配置中允许的话
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 初始化用于存储下一个解码器缓存的变量，如果需要使用缓存的话
        next_decoder_cache = () if use_cache else None
        # 遍历每个层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前的隐藏状态添加到变量中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果有过去的键值对存在，则取出当前层对应的过去的键值对
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 调用当前层模块进行前向传播，并获取输出
            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                training=training,
            )
            # 更新隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]

            # 如果需要使用缓存，则添加当前层的输出到下一个解码器缓存中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            # 如果需要输出注意力权重，则将当前层的注意力权重添加到变量中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                # 如果需要输出跨层注意力权重且有编码器的隐藏状态存在，则将当前层的跨层注意力权重添加到变量中
                if self.config.add_cross_attention and encoder_hidden_states is not None:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 添加最后一层的隐藏状态到变量中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要返回字典，则返回非None的变量
        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_attentions, all_cross_attentions] if v is not None
            )

        # 返回带有过去键值对和跨层注意力权重的输出
        return TFBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )
        # 构建神经网络模型
        def build(self, input_shape=None):
            # 如果模型已经构建完成，则直接返回
            if self.built:
                return
            # 将模型标记为已构建
            self.built = True
            # 如果模型对象中存在层属性
            if getattr(self, "layer", None) is not None:
                # 遍历每个层，并设置 TensorFlow 中的作用域
                for layer in self.layer:
                    with tf.name_scope(layer.name):
                        # 对每个层进行构建，输入形状暂时设为 None
                        layer.build(None)
# 从transformers.models.bert.modeling_tf_bert.TFBertPooler复制，并将Bert->Electra
class TFElectraPooler(tf.keras.layers.Layer):
    def __init__(self, config: ElectraConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，用于池化操作，输出维度为config.hidden_size，激活函数为tanh
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        # 保存config配置
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 通过取第一个token的hidden state来进行"池化"操作
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(inputs=first_token_tensor)

        return pooled_output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# 从transformers.models.albert.modeling_tf_albert.TFAlbertEmbeddings复制，并将Albert->Electra
class TFElectraEmbeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config: ElectraConfig, **kwargs):
        super().__init__(**kwargs)

        # 保存config配置和embedding_size等参数
        self.config = config
        self.embedding_size = config.embedding_size
        self.max_position_embeddings = config.max_position_embeddings
        self.initializer_range = config.initializer_range
        # 创建LayerNormalization层和Dropout层
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def build(self, input_shape=None):
        with tf.name_scope("word_embeddings"):
            # 创建word的embedding权重矩阵
            self.weight = self.add_weight(
                name="weight",
                shape=[self.config.vocab_size, self.embedding_size],
                initializer=get_initializer(self.initializer_range),
            )

        with tf.name_scope("token_type_embeddings"):
            # 创建token type的embedding权重矩阵
            self.token_type_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.config.type_vocab_size, self.embedding_size],
                initializer=get_initializer(self.initializer_range),
            )

        with tf.name_scope("position_embeddings"):
            # 创建position的embedding权重矩阵
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.embedding_size],
                initializer=get_initializer(self.initializer_range),
            )

        if self.built:
            return
        self.built = True
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.embedding_size])
    # 从transformers.models.bert.modeling_tf_bert.TFBertEmbeddings.call添加的代码
    # 定义了一个函数，对输入进行嵌入操作，返回输出嵌入张量

    def call(
        self,
        input_ids: tf.Tensor = None,
        position_ids: tf.Tensor = None,
        token_type_ids: tf.Tensor = None,
        inputs_embeds: tf.Tensor = None,
        past_key_values_length=0,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """

        if input_ids is None and inputs_embeds is None:
            raise ValueError("Need to provide either `input_ids` or `input_embeds`.")

        # 如果没有提供input_ids或inputs_embeds，则抛出值错误

        if input_ids is not None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        # 如果提供了input_ids，则检查嵌入是否在范围内，并根据input_ids从权重中获取对应的嵌入向量

        input_shape = shape_list(inputs_embeds)[:-1]

        # 获取inputs_embeds的形状，排除最后一个维度

        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)

        # 如果没有提供token_type_ids，则使用0填充形状与inputs_embeds相同的张量

        if position_ids is None:
            position_ids = tf.expand_dims(
                tf.range(start=past_key_values_length, limit=input_shape[1] + past_key_values_length), axis=0
            )

        # 如果没有提供position_ids，则使用范围从past_key_values_length到input_shape[1] + past_key_values_length的张量

        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
        # 从position_embeddings中获取对应索引的位置嵌入向量
        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)
        # 从token_type_embeddings中获取对应索引的标记类型嵌入向量
        final_embeddings = inputs_embeds + position_embeds + token_type_embeds
        # 最终的嵌入向量为输入嵌入、位置嵌入和标记类型嵌入的总和
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        # 对最终嵌入向量进行LayerNorm
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)
        # 根据training参数对最终嵌入向量进行dropout操作

        return final_embeddings
class TFElectraDiscriminatorPredictions(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，输出维度为 config.hidden_size，用于转换判别器的隐藏状态
        self.dense = tf.keras.layers.Dense(config.hidden_size, name="dense")
        # 创建一个全连接层，输出维度为 1，用于生成判别器的预测
        self.dense_prediction = tf.keras.layers.Dense(1, name="dense_prediction")
        self.config = config

    def call(self, discriminator_hidden_states, training=False):
        # 将判别器的隐藏状态通过全连接层进行线性变换
        hidden_states = self.dense(discriminator_hidden_states)
        # 使用激活函数对变换后的隐藏状态进行非线性变换
        hidden_states = get_tf_activation(self.config.hidden_act)(hidden_states)
        # 压缩隐藏状态，得到预测结果
        logits = tf.squeeze(self.dense_prediction(hidden_states), -1)

        return logits

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果定义了 self.dense 层，则对其进行构建
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果定义了 self.dense_prediction 层，则对其进行构建
        if getattr(self, "dense_prediction", None) is not None:
            with tf.name_scope(self.dense_prediction.name):
                self.dense_prediction.build([None, None, self.config.hidden_size])


class TFElectraGeneratorPredictions(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        # 创建一个 LayerNormalization 层，用于生成器的隐藏状态归一化
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建一个全连接层，输出维度为 config.embedding_size，用于转换生成器的隐藏状态
        self.dense = tf.keras.layers.Dense(config.embedding_size, name="dense")
        self.config = config

    def call(self, generator_hidden_states, training=False):
        # 将生成器的隐藏状态通过全连接层进行线性变换
        hidden_states = self.dense(generator_hidden_states)
        # 使用 GELU 激活函数对变换后的隐藏状态进行非线性变换
        hidden_states = get_tf_activation("gelu")(hidden_states)
        # 对变换后的隐藏状态进行归一化处理
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果定义了 self.LayerNorm 层，则对其进行构建
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.embedding_size])
        # 如果定义了 self.dense 层，则对其进行构建
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


class TFElectraPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # ElectraConfig 类型的配置类
    config_class = ElectraConfig
    # 模型的基础名称前缀
    base_model_prefix = "electra"
    # 从 PT 模型加载时忽略的键列表
    _keys_to_ignore_on_load_unexpected = [r"generator_lm_head.weight"]
    # 从 PT 模型加载时缺失的键列表
    _keys_to_ignore_on_load_missing = [r"dropout"]


@keras_serializable
class TFElectraMainLayer(tf.keras.layers.Layer):
    # ElectraConfig 类型的配置类
    config_class = ElectraConfig
    # 初始化 Electra 模型的参数和配置
    def __init__(self, config, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 保存配置信息
        self.config = config
        # 标志是否为解码器
        self.is_decoder = config.is_decoder

        # 创建 ElectraEmbeddings 对象
        self.embeddings = TFElectraEmbeddings(config, name="embeddings")

        # 如果嵌入层的维度大小不等于隐藏层的维度大小
        if config.embedding_size != config.hidden_size:
            # 创建一个全连接层，将嵌入层的维度转换为隐藏层的维度
            self.embeddings_project = tf.keras.layers.Dense(config.hidden_size, name="embeddings_project")

        # 创建 ElectraEncoder 对象
        self.encoder = TFElectraEncoder(config, name="encoder")

    # 获取输入嵌入层的方法
    def get_input_embeddings(self):
        return self.embeddings

    # 设置输入嵌入层的方法
    def set_input_embeddings(self, value):
        # 设置嵌入层的权重
        self.embeddings.weight = value
        # 设置嵌入层的词汇表大小
        self.embeddings.vocab_size = shape_list(value)[0]

    # 剪枝模型中注意力头的方法
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 抛出未实现错误，需要在子类中实现
        raise NotImplementedError
    # 获取扩展的注意力遮罩，根据输入的注意力遮罩、输入形状、数据类型和过去键值的长度来生成
    def get_extended_attention_mask(self, attention_mask, input_shape, dtype, past_key_values_length=0):
        # 获取批次大小和序列长度
        batch_size, seq_length = input_shape

        # 如果注意力遮罩为空，则以值为1填充，维度为（批次大小，序列长度 + 过去键值长度）
        if attention_mask is None:
            attention_mask = tf.fill(dims=(batch_size, seq_length + past_key_values_length), value=1)

        # 创建一个3D注意力遮罩，用于将2D张量转换为3D
        # 大小为[批次大小，1，1，目标序列长度]
        # 这样就能够广播到[批次大小，注意头数，源序列长度，目标序列长度]
        # 这个注意力遮罩比OpenAI GPT中的因果注意力的三角形遮罩更简单，我们只需要准备广播维度即可
        attention_mask_shape = shape_list(attention_mask)

        mask_seq_length = seq_length + past_key_values_length
        # 从`modeling_tf_t5.py`复制的部分
        # 提供维度为[批次大小，遮罩序列长度]的填充遮罩
        # - 如果模型是一个解码器，则在填充遮罩的基础上应用一个因果遮罩
        # - 如果模型是一个编码器，则使遮罩能够广播到[批次大小，注意头数，遮罩序列长度，遮罩序列长度]
        if self.is_decoder:
            seq_ids = tf.range(mask_seq_length)
            causal_mask = tf.less_equal(
                tf.tile(seq_ids[None, None, :], (batch_size, mask_seq_length, 1)),
                seq_ids[None, :, None],
            )
            causal_mask = tf.cast(causal_mask, dtype=attention_mask.dtype)
            extended_attention_mask = causal_mask * attention_mask[:, None, :]
            attention_mask_shape = shape_list(extended_attention_mask)
            extended_attention_mask = tf.reshape(
                extended_attention_mask, (attention_mask_shape[0], 1, attention_mask_shape[1], attention_mask_shape[2])
            )
            if past_key_values_length > 0:
                extended_attention_mask = extended_attention_mask[:, :, -seq_length:, :]
        else:
            extended_attention_mask = tf.reshape(
                attention_mask, (attention_mask_shape[0], 1, 1, attention_mask_shape[1])
            )

        # 因为注意力遮罩对于我们想要关注的位置为1.0，对于遮罩位置为0.0，
        # 这个操作将为我们创建一个张量，其中想要关注的位置为0.0，遮罩位置为-10000.0
        # 由于我们在softmax之前将其加到原始分数中，这实际上等同于完全移除这些位置
        extended_attention_mask = tf.cast(extended_attention_mask, dtype=dtype)
        one_cst = tf.constant(1.0, dtype=dtype)
        ten_thousand_cst = tf.constant(-10000.0, dtype=dtype)
        extended_attention_mask = tf.multiply(tf.subtract(one_cst, extended_attention_mask), ten_thousand_cst)

        return extended_attention_mask
```  
    # 获取头部掩码，用于控制 Transformer 中每个头部的注意力权重
    def get_head_mask(self, head_mask):
        # 如果头部掩码已经给定，则抛出未实现错误
        if head_mask is not None:
            raise NotImplementedError
        # 否则，初始化头部掩码为一个列表，列表长度为模型配置中指定的隐藏层数
        else:
            head_mask = [None] * self.config.num_hidden_layers

        # 返回头部掩码
        return head_mask

    # 调用 Transformer 模型
    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
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
    # 构建 Transformer 模型
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回，不再重复构建
        if self.built:
            return
        # 标记模型已构建
        self.built = True
        # 构建嵌入层
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        # 构建编码器层
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 构建嵌入层后的投影层
        if getattr(self, "embeddings_project", None) is not None:
            with tf.name_scope(self.embeddings_project.name):
                # 构建投影层，输入形状为 [None, None, 配置中的嵌入维度]
                self.embeddings_project.build([None, None, self.config.embedding_size])
# 使用 dataclass 装饰器定义 TFElectraForPreTrainingOutput 类，它是 ModelOutput 的子类
@dataclass
class TFElectraForPreTrainingOutput(ModelOutput):
    """
    Output type of [`TFElectraForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `tf.Tensor` of shape `(1,)`):
            Total loss of the ELECTRA objective.
        logits (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Prediction scores of the head (scores for each token before SoftMax).
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    logits: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None


# 定义 ELECTRA_START_DOCSTRING 字符串，它会被用作文档字符串
ELECTRA_START_DOCSTRING = r"""

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
    """
    在调用模型时，可以使用以下两种方式之一传递输入参数：
    `model([input_ids, attention_mask])` 或 `model([input_ids, attention_mask, token_type_ids])`
    - 返回一个字典，其中包含一个或多个输入 Tensor，与文档字符串中给定的输入名称相关联：
    `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    需要注意的是，当使用[子类化](https://keras.io/guides/making_new_layers_and_models_via_subclassing/)创建模型和层时，
    你无需担心这些，因为你可以像对待任何其他 Python 函数一样传递输入参数！

    参数：
        config ([`ElectraConfig`]): 模型配置类，包含模型的所有参数。
            使用配置文件初始化模型时不会加载与模型关联的权重，只会加载配置。
            可以查看 [`~PreTrainedModel.from_pretrained`] 方法以加载模型权重。
    """
"""

# ELECTRA 模型输入参数的文档字符串，描述了输入参数的含义和格式
ELECTRA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`Numpy array` or `tf.Tensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.__call__`] and
            [`PreTrainedTokenizer.encode`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`Numpy array` or `tf.Tensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`Numpy array` or `tf.Tensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`Numpy array` or `tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`tf.Tensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
            config will be used instead.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in
            eager mode, in graph mode the value will always be set to True.
        training (`bool`, *optional*, defaults to `False`):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""

@add_start_docstrings(
    # 以下是代码片段的多行字符串，用于生成 Electra 模型的文档字符串
    # 该模型是一个裸的 Electra 模型转换器，输出未经任何特定头部处理的原始隐藏状态。
    # 与 BERT 模型相同，但如果隐藏大小和嵌入大小不同，则在嵌入层和编码器之间使用额外的线性层。
    # 可以将生成器和判别器的检查点都加载到这个模型中。
# 定义 TF Electra 模型类，继承自 TF Electra 预训练模型类
class TFElectraModel(TFElectraPreTrainedModel):
    # 初始化方法
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 创建 Electra 主要层对象
        self.electra = TFElectraMainLayer(config, name="electra")

    # 定义模型调用方法
    @unpack_inputs
    # 添加模型前向传播的文档字符串
    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 添加代码示例的文档字符串
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
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
    def call(self, 
        input_ids: tf.Tensor, 
        attention_mask: tf.Tensor = None, 
        token_type_ids: tf.Tensor = None, 
        position_ids: tf.Tensor = None, 
        head_mask: tf.Tensor = None,
        inputs_embeds: tf.Tensor = None, 
        output_attentions: bool = False, 
        output_hidden_states: bool = False,
        return_dict: bool = True, 
        training: bool = False, 
        **kwargs) -> Union[TFBaseModelOutputWithPastAndCrossAttentions, Tuple[tf.Tensor]]:
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
        # 调用 Electra 模型，传递各种参数并返回输出
        outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        return outputs

    def build(self, input_shape=None):
        # 如果模型已经构建完成，则直接返回
        if self.built:
            return
        # 标记模型为已构建状态
        self.built = True
        # 如果存在 Electra 模型，则构建 Electra 模型
        if getattr(self, "electra", None) is not None:
            with tf.name_scope(self.electra.name):
                # 构建 Electra 模型
                self.electra.build(None)
# 导入所需的库或模块
@add_start_docstrings(
    """
    Electra model with a binary classification head on top as used during pretraining for identifying generated tokens.

    Even though both the discriminator and generator may be loaded into this model, the discriminator is the only model
    of the two to have the correct classification head to be used for this model.
    """,
    ELECTRA_START_DOCSTRING, # 添加预训练过程中用于识别生成标记的二进制分类头部的 Electra 模型的说明文档
)
class TFElectraForPreTraining(TFElectraPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.electra = TFElectraMainLayer(config, name="electra") # 实例化 Electra 主层对象
        self.discriminator_predictions = TFElectraDiscriminatorPredictions(config, name="discriminator_predictions") # 实例化分类器模型对象

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFElectraForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    # 定义调用方法，传入多个参数
    def call(
        self,
        input_ids: TFModelInputType | None = None, # 输入的 ID 序列数据
        attention_mask: np.ndarray | tf.Tensor | None = None, # 注意力掩码
        token_type_ids: np.ndarray | tf.Tensor | None = None, # 标记类型 ID
        position_ids: np.ndarray | tf.Tensor | None = None, # 位置 ID
        head_mask: np.ndarray | tf.Tensor | None = None, # 头部屏蔽
        inputs_embeds: np.ndarray | tf.Tensor | None = None, # 输入嵌入
        output_attentions: Optional[bool] = None, # 是否输出注意力
        output_hidden_states: Optional[bool] = None, # 是否输出隐藏状态
        return_dict: Optional[bool] = None, # 是否返回字典
        training: Optional[bool] = False, # 是否处于训练模式
    ) -> Union[TFElectraForPreTrainingOutput, Tuple[tf.Tensor]]:
        r"""
        Returns:

        Examples:

        ```py
        >>> import tensorflow as tf
        >>> from transformers import AutoTokenizer, TFElectraForPreTraining

        >>> tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")
        >>> model = TFElectraForPreTraining.from_pretrained("google/electra-small-discriminator")
        >>> input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
        >>> outputs = model(input_ids)
        >>> scores = outputs[0]
        ```"""
        discriminator_hidden_states = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 获取 discriminator 的隐藏状态
        discriminator_sequence_output = discriminator_hidden_states[0]
        # 使用 discriminator 的输出进行预测
        logits = self.discriminator_predictions(discriminator_sequence_output)

        if not return_dict:
            # 如果不返回字典格式结果，则返回 logits 和 discriminator 的隐藏状态
            return (logits,) + discriminator_hidden_states[1:]

        # 返回 TFElectraForPreTrainingOutput 对象，包括 logits、隐藏状态和注意力
        return TFElectraForPreTrainingOutput(
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )

    def build(self, input_shape=None):
        # 如果模型已经构建完毕，则直接返回，避免重复构建
        if self.built:
            return
        self.built = True
        # 构建模型
        if getattr(self, "electra", None) is not None:
            # 使用电路名称作为作用域，构建 electra 模型
            with tf.name_scope(self.electra.name):
                self.electra.build(None)
        if getattr(self, "discriminator_predictions", None) is not None:
            # 使用 discriminator 名称作为作用域，构建 discriminator_predictions 模型
            with tf.name_scope(self.discriminator_predictions.name):
                self.discriminator_predictions.build(None)
class TFElectraMaskedLMHead(tf.keras.layers.Layer):
    # 创建一个 Electra Masked LM 头部的类
    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.embedding_size = config.embedding_size
        self.input_embeddings = input_embeddings

    def build(self, input_shape):
        # 创建一个可训练的偏置项
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")

        super().build(input_shape)

    def get_output_embeddings(self):
        # 返回输入嵌入层
        return self.input_embeddings

    def set_output_embeddings(self, value):
        # 设置输入嵌入的权重和词汇量大小
        self.input_embeddings.weight = value
        self.input_embeddings.vocab_size = shape_list(value)[0]

    def get_bias(self):
        # 返回偏置项
        return {"bias": self.bias}

    def set_bias(self, value):
        # 设置偏置项和词汇量大小
        self.bias = value["bias"]
        self.config.vocab_size = shape_list(value["bias"])[0]

    def call(self, hidden_states):
        # 根据隐藏层状态计算模型的输出
        seq_length = shape_list(tensor=hidden_states)[1]
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.embedding_size])
        hidden_states = tf.matmul(a=hidden_states, b=self.input_embeddings.weight, transpose_b=True)
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)

        return hidden_states


@add_start_docstrings(
    """
    Electra model with a language modeling head on top.

    Even though both the discriminator and generator may be loaded into this model, the generator is the only model of
    the two to have been trained for the masked language modeling task.
    """,
    ELECTRA_START_DOCSTRING,
)
class TFElectraForMaskedLM(TFElectraPreTrainedModel, TFMaskedLanguageModelingLoss):
    # 创建一个带有语言模型头部的 Electra 模型
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.config = config
        self.electra = TFElectraMainLayer(config, name="electra")
        self.generator_predictions = TFElectraGeneratorPredictions(config, name="generator_predictions")

        if isinstance(config.hidden_act, str):
            self.activation = get_tf_activation(config.hidden_act)
        else:
            self.activation = config.hidden_act

        self.generator_lm_head = TFElectraMaskedLMHead(config, self.electra.embeddings, name="generator_lm_head")

    def get_lm_head(self):
        # 返回生成器的 LM 头部
        return self.generator_lm_head

    def get_prefix_bias_name(self):
        # 返回已弃用的方法警告
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        return self.name + "/" + self.generator_lm_head.name

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="google/electra-small-generator",
        output_type=TFMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="[MASK]",
        expected_output="'paris'",
        expected_loss=1.22,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 定义输入的 token IDs
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 定义注意力掩码
        token_type_ids: np.ndarray | tf.Tensor | None = None,  # 定义 token 类型 IDs
        position_ids: np.ndarray | tf.Tensor | None = None,  # 定义位置 IDs
        head_mask: np.ndarray | tf.Tensor | None = None,  # 定义头部掩码
        inputs_embeds: np.ndarray | tf.Tensor | None = None,  # 定义嵌入输入
        output_attentions: Optional[bool] = None,  # 定义是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 定义是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 定义是否以字典形式返回结果
        labels: np.ndarray | tf.Tensor | None = None,  # 定义标签用于计算 MLN 损失
        training: Optional[bool] = False,  # 定义是否为训练模式
    ) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        generator_hidden_states = self.electra(  # 调用电力制造商模型
            input_ids=input_ids,  # 输入 token IDs
            attention_mask=attention_mask,  # 注意力掩码
            token_type_ids=token_type_ids,  # token 类型 IDs
            position_ids=position_ids,  # 位置 IDs
            head_mask=head_mask,  # 头部掩码
            inputs_embeds=inputs_embeds,  # 嵌入输入
            output_attentions=output_attentions,  # 输出注意力权重
            output_hidden_states=output_hidden_states,  # 输出隐藏状态
            return_dict=return_dict,  # 以字典形式返回结果
            training=training,  # 训练模式
        )
        generator_sequence_output = generator_hidden_states[0]  # 获取 generator_hidden_states 中的第一个元素
        prediction_scores = self.generator_predictions(generator_sequence_output, training=training)  # 利用 generator_sequence_output 进行生成器预测
        prediction_scores = self.generator_lm_head(prediction_scores, training=training)  # 使用生成器的 LM 头来预测
        loss = None if labels is None else self.hf_compute_loss(labels, prediction_scores)  # 如果没有标签，则损失为 None，否则计算损失
    
        if not return_dict:  # 如果不以字典形式返回结果
            output = (prediction_scores,) + generator_hidden_states[1:]  # 构造输出元组
    
            return ((loss,) + output) if loss is not None else output  # 如果有损失则返回损失和输出，否则只返回输出
    
        return TFMaskedLMOutput(  # 使用 TFMaskedLMOutput 返回结果
            loss=loss,  # 损失
            logits=prediction_scores,  # 预测值
            hidden_states=generator_hidden_states.hidden_states,  # 隐藏状态
            attentions=generator_hidden_states.attentions,  # 注意力权重
        )
    # 构建模型，如果已经构建完毕则直接返回
    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回
        if self.built:
            return
        # 更新状态，表示模型已经构建
        self.built = True
        # 如果存在电力生成器模型，则构建电力生成器模型
        if getattr(self, "electra", None) is not None:
            # 在电力生成器模型的命名空间内进行构建
            with tf.name_scope(self.electra.name):
                # 构建电力生成器模型
                self.electra.build(None)
        # 如果存在生成器预测模型，则构建生成器预测模型
        if getattr(self, "generator_predictions", None) is not None:
            # 在生成器预测模型的命名空间内进行构建
            with tf.name_scope(self.generator_predictions.name):
                # 构建生成器预测模型
                self.generator_predictions.build(None)
        # 如果存在生成器 lm 头模型，则构建生成器 lm 头模型
        if getattr(self, "generator_lm_head", None) is not None:
            # 在生成器 lm 头模型的命名空间内进行构建
            with tf.name_scope(self.generator_lm_head.name):
                # 构建生成器 lm 头模型
                self.generator_lm_head.build(None)
class TFElectraClassificationHead(tf.keras.layers.Layer):
    """对句子级分类任务的头部。"""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，用于分类任务
        self.dense = tf.keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 根据配置设置分类器的 dropout 层
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = tf.keras.layers.Dropout(classifier_dropout)
        # 创建一个全连接层，用于输出分类结果
        self.out_proj = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="out_proj"
        )
        self.config = config

    def call(self, inputs, **kwargs):
        # 取第一个位置的 token，即 <s> token (等同于 [CLS])
        x = inputs[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = get_tf_activation("gelu")(x)  # 虽然 BERT 在这里使用 tanh，但似乎 Electra 作者在这里使用了 gelu
        x = self.dropout(x)
        x = self.out_proj(x)

        return x

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
    在顶部添加一个序列分类/回归头部的 ELECTRA 模型变换器（在 pooled 输出的顶部是一个线性层），例如 GLUE 任务。
    """,
    ELECTRA_START_DOCSTRING,
)
class TFElectraForSequenceClassification(TFElectraPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        # 创建 ELECTRA 主层和分类头部
        self.electra = TFElectraMainLayer(config, name="electra")
        self.classifier = TFElectraClassificationHead(config, name="classifier")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="bhadresh-savani/electra-base-emotion",
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'joy'",
        expected_loss=0.06,
    )
    # 定义一个 call 方法，接受多个参数，返回 TFSequenceClassifierOutput 或 Tuple[tf.Tensor]
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 输入的序列的 token IDs
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力掩码
        token_type_ids: np.ndarray | tf.Tensor | None = None,  # token 类型 IDs
        position_ids: np.ndarray | tf.Tensor | None = None,  # 位置 IDs
        head_mask: np.ndarray | tf.Tensor | None = None,  # 头部掩码
        inputs_embeds: np.ndarray | tf.Tensor | None = None,  # 输入的嵌入
        output_attentions: Optional[bool] = None,  # 是否输出注意力
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否以字典形式返回结果
        labels: np.ndarray | tf.Tensor | None = None,  # 用于计算损失的标签
        training: Optional[bool] = False,  # 是否处于训练模式
    ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 使用 Electra 模型对输入进行处理
        outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 对 Electra 模型输出应用分类器
        logits = self.classifier(outputs[0])
        # 如果存在标签，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果不以字典形式返回结果，则构建输出
        if not return_dict:
            output = (logits,) + outputs[1:]

            return ((loss,) + output) if loss is not None else output

        # 以 TFSequenceClassifierOutput 形式返回结果
        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 构建模型
    def build(self, input_shape=None):
        # 如果已经构建，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 Electra 模型，则构建其结构
        if getattr(self, "electra", None) is not None:
            with tf.name_scope(self.electra.name):
                self.electra.build(None)
        # 如果存在分类器模型，则构建其结构
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build(None)
# 添加模型文档字符串，说明该模型是在 ELECTRA 模型的基础上添加了多选分类头部（线性层和 softmax），用于 RocStories/SWAG 任务等
@add_start_docstrings(
    """
    ELECTRA Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    ELECTRA_START_DOCSTRING,
)
class TFElectraForMultipleChoice(TFElectraPreTrainedModel, TFMultipleChoiceLoss):
    # 初始化方法
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 使用 ELECTRA 主层创建 ELECTRA 对象
        self.electra = TFElectraMainLayer(config, name="electra")
        # 创建序列摘要对象
        self.sequence_summary = TFSequenceSummary(
            config, initializer_range=config.initializer_range, name="sequence_summary"
        )
        # 创建分类器层，用于多选分类
        self.classifier = tf.keras.layers.Dense(
            1, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        # 保存配置信息
        self.config = config

    # 模型调用方法
    @unpack_inputs
    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
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
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    # 定义一个方法，接受输入和输出参数，并指定返回类型为TFMultipleChoiceModelOutput或者Tuple[tf.Tensor]
    def encode_plus(self, input_ids: tf.Tensor, attention_mask: tf.Tensor, token_type_ids: tf.Tensor = None,
                    position_ids: tf.Tensor = None, inputs_embeds: tf.Tensor = None, head_mask: tf.Tensor = None,
                    output_attentions: bool = False, output_hidden_states: bool = False, return_dict: bool = True,
                    training: bool = False, labels: tf.Tensor = None) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
        """
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
        Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
        where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)
        """
    
        # 如果input_ids不为空，则获取其对应的num_choices和seq_length；否则，获取inputs_embeds对应的num_choices和seq_length
        if input_ids is not None:
            num_choices = shape_list(input_ids)[1]
            seq_length = shape_list(input_ids)[2]
        else:
            num_choices = shape_list(inputs_embeds)[1]
            seq_length = shape_list(inputs_embeds)[2]
    
        # 将input_ids、attention_mask、token_type_ids、position_ids和inputs_embeds各自展开为两维数组
        flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
        flat_attention_mask = tf.reshape(attention_mask, (-1, seq_length)) if attention_mask is not None else None
        flat_token_type_ids = tf.reshape(token_type_ids, (-1, seq_length)) if token_type_ids is not None else None
        flat_position_ids = tf.reshape(position_ids, (-1, seq_length)) if position_ids is not None else None
        flat_inputs_embeds = (
            tf.reshape(inputs_embeds, (-1, seq_length, shape_list(inputs_embeds)[3]))
            if inputs_embeds is not None
            else None
        )
        
        # 调用electra方法进行处理
        outputs = self.electra(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids,
            position_ids=flat_position_ids,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        
        # 对输出结果进行序列摘要处理
        logits = self.sequence_summary(outputs[0])
        logits = self.classifier(logits)
        
        # 将logits重塑为二维数组
        reshaped_logits = tf.reshape(logits, (-1, num_choices))
        
        # 如果labels不为空，则计算损失值；否则，损失值设为None
        loss = None if labels is None else self.hf_compute_loss(labels, reshaped_logits)
    
        if not return_dict:
            # 如果不需要返回字典形式的结果，则直接返回reshaped_logits和其他outputs
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
    
        # 构建TFMultipleChoiceModelOutput对象并返回
        return TFMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    # 定义build方法，用于搭建模型参数和结构
    def build(self, input_shape=None):
        if self.built:
            return
        # 设定模型已经构建完毕
        self.built = True
        # 根据模型参数构建electra、sequence_summary和classifier
        if getattr(self, "electra", None) is not None:
            with tf.name_scope(self.electra.name):
                self.electra.build(None)
        if getattr(self, "sequence_summary", None) is not None:
            with tf.name_scope(self.sequence_summary.name):
                self.sequence_summary.build(None)
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
# 导入必要的库
@add_start_docstrings(
    """
    Electra model with a token classification head on top.

    Both the discriminator and generator may be loaded into this model.
    """,
    ELECTRA_START_DOCSTRING,
)
# 定义一个继承自 TFElectraPreTrainedModel 和 TFTokenClassificationLoss 的类 TFElectraForTokenClassification
class TFElectraForTokenClassification(TFElectraPreTrainedModel, TFTokenClassificationLoss):
    # 定义初始化方法
    def __init__(self, config, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, **kwargs)

        # 创建 Electra 主层对象
        self.electra = TFElectraMainLayer(config, name="electra")
        # 定义分类器的 dropout 层
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 创建 dropout 层
        self.dropout = tf.keras.layers.Dropout(classifier_dropout)
        # 创建分类器全连接层
        self.classifier = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        # 保存配置
        self.config = config

    # 定义 call 方法
    @unpack_inputs
    # 添加文档字符串
    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 添加代码示例文档字符串
    @add_code_sample_docstrings(
        checkpoint="bhadresh-savani/electra-base-discriminator-finetuned-conll03-english",
        output_type=TFTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="['B-LOC', 'B-ORG', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'O', 'B-LOC', 'I-LOC']",
        expected_loss=0.11,
    )
    # 定义模型前向传播方法
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
        ) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 使用Electra模型对输入进行处理，获取鉴别器的隐藏状态
        discriminator_hidden_states = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 获取鉴别器的序列输出
        discriminator_sequence_output = discriminator_hidden_states[0]
        # 在鉴别器的序列输出上应用dropout
        discriminator_sequence_output = self.dropout(discriminator_sequence_output)
        # 通过分类器获取logits
        logits = self.classifier(discriminator_sequence_output)
        # 如果存在标签，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        if not return_dict:
            # 如果不返回字典，则返回logits和鉴别器的隐藏状态
            output = (logits,) + discriminator_hidden_states[1:]

            return ((loss,) + output) if loss is not None else output

        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "electra", None) is not None:
            with tf.name_scope(self.electra.name):
                self.electra.build(None)
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                # 在分类器上应用build方法
                self.classifier.build([None, None, self.config.hidden_size])
@add_start_docstrings(
    """
    Electra Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    ELECTRA_START_DOCSTRING,
)

这里是一个装饰器，用于给下面定义的类 `TFElectraForQuestionAnswering` 添加文档字符串。文档字符串描述了这个类的作用和功能。


class TFElectraForQuestionAnswering(TFElectraPreTrainedModel, TFQuestionAnsweringLoss):

定义了一个类 `TFElectraForQuestionAnswering`，该类继承自 `TFElectraPreTrainedModel` 和 `TFQuestionAnsweringLoss`。


def __init__(self, config, *inputs, **kwargs):
    super().__init__(config, *inputs, **kwargs)

    self.num_labels = config.num_labels
    self.electra = TFElectraMainLayer(config, name="electra")
    self.qa_outputs = tf.keras.layers.Dense(
        config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
    )
    self.config = config

初始化方法，接受一个 `config` 参数和可变数量的输入参数。首先调用父类的初始化方法。然后，将 `config.num_labels` 赋值给 `self.num_labels`，创建一个 `TFElectraMainLayer` 实例赋值给 `self.electra`，创建一个带有配置参数的全连接层 `self.qa_outputs`，最后将配置参数赋值给 `self.config`。


@unpack_inputs
@add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
@add_code_sample_docstrings(
    checkpoint="bhadresh-savani/electra-base-squad2",
    output_type=TFQuestionAnsweringModelOutput,
    config_class=_CONFIG_FOR_DOC,
    qa_target_start_index=11,
    qa_target_end_index=12,
    expected_output="'a nice puppet'",
    expected_loss=2.64,
)

这是一系列装饰器，用于添加文档字符串和样例代码说明给下面的 `call` 方法。这些文档字符串和样例代码会在文档中显示给用户，用于说明模型的输入、输出和使用方法。


def call(
    self,
    input_ids: TFModelInputType | None = None,
    attention_mask: np.ndarray | tf.Tensor | None = None,
    token_type_ids: np.ndarray | tf.Tensor | None = None,
    position_ids: np.ndarray | tf.Tensor | None = None,
    head_mask: np.ndarray | tf.Tensor | None = None,
    inputs_embeds: np.ndarray | tf.Tensor | None = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    start_positions: np.ndarray | tf.Tensor | None = None,
    end_positions: np.ndarray | tf.Tensor | None = None,
    training: Optional[bool] = False,

定义了一个 `call` 方法，用于执行模型的前向传播。它接受多个参数，包括输入的各种特征，输出控制标志和训练标志。
    ) -> Union[TFQuestionAnsweringModelOutput, Tuple[tf.Tensor]]:
        r"""
        start_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        # 利用 Electra 模型进行预测，获取鉴别器的隐藏状态
        discriminator_hidden_states = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 从鉴别器的隐藏状态中提取序列输出
        discriminator_sequence_output = discriminator_hidden_states[0]
        # 使用 QA 输出层生成起始位置和结束位置的对数概率
        logits = self.qa_outputs(discriminator_sequence_output)
        # 将对数概率拆分为起始位置和结束位置的对数概率
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        # 移除单维度，以便计算损失
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)
        loss = None

        # 如果提供了起始位置和结束位置，则计算损失
        if start_positions is not None and end_positions is not None:
            # 构建标签字典
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            # 计算损失
            loss = self.hf_compute_loss(labels, (start_logits, end_logits))

        # 如果不返回字典，则返回元组形式的输出
        if not return_dict:
            output = (
                start_logits,
                end_logits,
            ) + discriminator_hidden_states[1:]

            return ((loss,) + output) if loss is not None else output

        # 返回 TFQuestionAnsweringModelOutput 对象，包含损失、起始位置和结束位置的对数概率、鉴别器的隐藏状态和注意力权重
        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记模型为已构建
        self.built = True
        # 如果存在 Electra 模型，则构建其权重
        if getattr(self, "electra", None) is not None:
            with tf.name_scope(self.electra.name):
                self.electra.build(None)
        # 如果存在 QA 输出层，则构建其权重
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])
```