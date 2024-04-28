# `.\transformers\models\albert\modeling_tf_albert.py`

```py
# 导入所需模块和库
from __future__ import annotations

import math  # 导入数学函数库
from dataclasses import dataclass  # 从 dataclasses 模块导入 dataclass 装饰器
from typing import Dict, Optional, Tuple, Union  # 导入类型提示相关的模块

import numpy as np  # 导入 NumPy 库并使用别名 np
import tensorflow as tf  # 导入 TensorFlow 库并使用别名 tf

from ...activations_tf import get_tf_activation  # 从 activations_tf 模块导入 get_tf_activation 函数
from ...modeling_tf_outputs import (  # 从 modeling_tf_outputs 模块导入多个输出类
    TFBaseModelOutput,
    TFBaseModelOutputWithPooling,
    TFMaskedLMOutput,
    TFMultipleChoiceModelOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (  # 从 modeling_tf_utils 模块导入多个实用函数和类
    TFMaskedLanguageModelingLoss,
    TFModelInputType,
    TFMultipleChoiceLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    get_initializer,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import (  # 从 tf_utils 模块导入实用函数
    check_embeddings_within_bounds,
    shape_list,
    stable_softmax,
)
from ...utils import (  # 从 utils 模块导入实用函数
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

# 设置日志记录器
logger = logging.get_logger(__name__)

# 用于文档的模型检查点和配置
_CHECKPOINT_FOR_DOC = "albert-base-v2"
_CONFIG_FOR_DOC = "AlbertConfig"

# 预训练的 TF Albert 模型存档列表
TF_ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "albert-base-v1",
    "albert-large-v1",
    "albert-xlarge-v1",
    "albert-xxlarge-v1",
    "albert-base-v2",
    "albert-large-v2",
    "albert-xlarge-v2",
    "albert-xxlarge-v2",
    # 在 https://huggingface.co/models?filter=albert 查看所有 ALBERT 模型
]

class TFAlbertPreTrainingLoss:
    """
    适用于 ALBERT 预训练的损失函数，即通过结合 SOP + MLM 预训练语言模型的任务。
    .. 注意：任何标签为 -100 的将在损失计算中被忽略（以及相应的对数概率）。
    """

class TFAlbertEmbeddings(tf.keras.layers.Layer):
    """
    从单词、位置和 token_type 嵌入构建嵌入。
    """
    # 初始化函数，用于创建一个新的AlbertEmbeddings对象
    def __init__(self, config: AlbertConfig, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 将传入的配置参数保存到对象中
        self.config = config
        # 获取嵌入向量的大小
        self.embedding_size = config.embedding_size
        # 获取最大位置嵌入的长度
        self.max_position_embeddings = config.max_position_embeddings
        # 获取初始化权重的范围
        self.initializer_range = config.initializer_range
        # 创建一个 LayerNormalization 层，用于规范化输入数据
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建一个 Dropout 层，用于在训练中进行随机失活
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    # 构建函数，用于构建模型中的各个层
    def build(self, input_shape=None):
        # 在 "word_embeddings" 命名空间下创建权重张量
        with tf.name_scope("word_embeddings"):
            self.weight = self.add_weight(
                name="weight",
                # 设置权重的形状为 [词汇表大小, 嵌入向量大小]
                shape=[self.config.vocab_size, self.embedding_size],
                # 使用指定的初始化器来初始化权重
                initializer=get_initializer(self.initializer_range),
            )

        # 在 "token_type_embeddings" 命名空间下创建权重张量
        with tf.name_scope("token_type_embeddings"):
            self.token_type_embeddings = self.add_weight(
                name="embeddings",
                # 设置权重的形状为 [token 类型数, 嵌入向量大小]
                shape=[self.config.type_vocab_size, self.embedding_size],
                # 使用指定的初始化器来初始化权重
                initializer=get_initializer(self.initializer_range),
            )

        # 在 "position_embeddings" 命名空间下创建权重张量
        with tf.name_scope("position_embeddings"):
            self.position_embeddings = self.add_weight(
                name="embeddings",
                # 设置权重的形状为 [最大位置嵌入长度, 嵌入向量大小]
                shape=[self.max_position_embeddings, self.embedding_size],
                # 使用指定的初始化器来初始化权重
                initializer=get_initializer(self.initializer_range),
            )

        # 如果已经构建过了，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在 LayerNorm 层，则构建该层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                # 构建 LayerNorm 层，指定输入的形状为 [None, None, 嵌入向量大小]
                self.LayerNorm.build([None, None, self.config.embedding_size])

    # 模型调用函数，用于执行前向传播
    # 从 transformers.models.bert.modeling_tf_bert.TFBertEmbeddings.call 复制而来
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
        # 检查是否提供了 input_ids 或 input_embeds，若没有则引发 ValueError
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Need to provide either `input_ids` or `input_embeds`.")

        # 如果提供了 input_ids，则根据其从权重中获取嵌入向量
        if input_ids is not None:
            # 检查输入的 id 是否在有效范围内
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        # 获取输入嵌入向量的形状
        input_shape = shape_list(inputs_embeds)[:-1]

        # 如果未提供 token_type_ids，则创建与输入形状相同的全零张量
        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)

        # 如果未提供 position_ids，则创建一个范围从 past_key_values_length 到 input_shape[1]+past_key_values_length 的张量
        if position_ids is None:
            position_ids = tf.expand_dims(
                tf.range(start=past_key_values_length, limit=input_shape[1] + past_key_values_length), axis=0
            )

        # 根据 position_ids 获取位置嵌入向量
        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
        # 根据 token_type_ids 获取 token 类型嵌入向量
        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)
        # 将输入嵌入向量、位置嵌入向量和 token 类型嵌入向量相加得到最终的嵌入向量
        final_embeddings = inputs_embeds + position_embeds + token_type_embeds
        # 对最终嵌入向量进行 LayerNormalization
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        # 对最终嵌入向量进行 dropout 处理
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)

        return final_embeddings
class TFAlbertAttention(tf.keras.layers.Layer):
    """Contains the complete attention sublayer, including both dropouts and layer norm."""

    def __init__(self, config: AlbertConfig, **kwargs):
        super().__init__(**kwargs)

        # 检查隐藏层大小是否能被注意力头数整除
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        # 设置注意力头数和注意力头大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)
        self.output_attentions = config.output_attentions

        # 定义查询、键、值以及密集层
        self.query = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        self.key = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        self.value = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 两个不同的 dropout 概率；参见 https://github.com/google-research/albert/blob/master/modeling.py#L971-L993
        self.attention_dropout = tf.keras.layers.Dropout(rate=config.attention_probs_dropout_prob)
        self.output_dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # 从 [batch_size, seq_length, all_head_size] 重塑为 [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

        # 将张量从 [batch_size, seq_length, num_attention_heads, attention_head_size] 转置为 [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(
        self,
        input_tensor: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    # 定义函数的输入和输出类型，此函数返回一个包含 Tensor 的元组
    ) -> Tuple[tf.Tensor]:
        # 获取输入张量的批大小
        batch_size = shape_list(input_tensor)[0]
        # 使用输入张量计算混合查询层
        mixed_query_layer = self.query(inputs=input_tensor)
        # 使用输入张量计算混合键层
        mixed_key_layer = self.key(inputs=input_tensor)
        # 使用输入张量计算混合值层
        mixed_value_layer = self.value(inputs=input_tensor)
        # 通过变换对分数进行归一化
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        # 计算原始注意力分数，使用“查询”和“键”的点积
        # (batch size, num_heads, seq_len_q, seq_len_k)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        # 计算缩放的点积注意力分数
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, dk)

        if attention_mask is not None:
            # 应用注意力掩码（在 TFAlbertModel 的 call() 函数中预先计算）
            attention_scores = tf.add(attention_scores, attention_mask)

        # 将注意力分数归一化为概率
        attention_probs = stable_softmax(logits=attention_scores, axis=-1)

        # 对注意力概率进行丢弃
        attention_probs = self.attention_dropout(inputs=attention_probs, training=training)

        # 如果有需要，对头进行掩码处理
        if head_mask is not None:
            attention_probs = tf.multiply(attention_probs, head_mask)

        # 计算上下文向量，将注意力概率与值层相乘
        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])

        # 重新调整张量形状
        # (batch_size, seq_len_q, all_head_size)
        context_layer = tf.reshape(tensor=context_layer, shape=(batch_size, -1, self.all_head_size))
        # 输出包含上下文层和注意力概率的元组
        self_outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        # 获取隐藏状态
        hidden_states = self_outputs[0]
        # 全连接层
        hidden_states = self.dense(inputs=hidden_states)
        # 对输出进行丢弃
        hidden_states = self.output_dropout(inputs=hidden_states, training=training)
        # 添加输入张量到注意力输出并进行层归一化
        attention_output = self.LayerNorm(inputs=hidden_states + input_tensor)

        # 如果输出注意力，则添加到输出中
        outputs = (attention_output,) + self_outputs[1:]

        # 返回输出
        return outputs
```  
    # 构建神经网络模型，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 设置标志位表示已经构建过
        self.built = True
        # 如果存在查询操作，则构建查询操作
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.hidden_size])
        # 如果存在键操作，则构建键操作
        if getattr(self, "key", None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.config.hidden_size])
        # 如果存在值操作，则构建值操作
        if getattr(self, "value", None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.config.hidden_size])
        # 如果存在密集层操作，则构建密集层操作
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果存在层归一化操作，则构建层归一化操作
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
# 定义 TFAlbertLayer 类，继承自 tf.keras.layers.Layer
class TFAlbertLayer(tf.keras.layers.Layer):
    # 初始化函数，接受配置参数 config 和其他关键字参数
    def __init__(self, config: AlbertConfig, **kwargs):
        # 调用父类初始化函数
        super().__init__(**kwargs)

        # 创建 self.attention 属性，值为 TFAlbertAttention 类的实例对象，命名为 "attention"
        self.attention = TFAlbertAttention(config, name="attention")
        # 创建 self.ffn 属性，值为全连接层(Dense)对象，用于前馈网络，设置神经元数和初始化方式
        self.ffn = tf.keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="ffn"
        )

        # 根据配置中隐藏层激活函数的类型，选择相应的激活函数或者使用配置中指定的激活函数
        if isinstance(config.hidden_act, str):
            self.activation = get_tf_activation(config.hidden_act)
        else:
            self.activation = config.hidden_act

        # 创建 self.ffn_output 属性，值为全连接层(Dense)对象，用于前馈网络输出，设置神经元数和初始化方式
        self.ffn_output = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="ffn_output"
        )
        # 创建 self.full_layer_layer_norm 属性，值为 LayerNormalization 层对象，设置 epsilon 值
        self.full_layer_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="full_layer_layer_norm"
        )
        # 创建 self.dropout 属性，值为 Dropout 层对象，设置 dropout rate
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        # 保存配置参数到 self.config 属性
        self.config = config

    # call 方法用于实现层的正向传播，接受输入张量和一些掩码张量等参数，返回一个元组
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        # 调用 self.attention 对象的 call 方法进行注意力计算，并返回相关输出
        attention_outputs = self.attention(
            input_tensor=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            training=training,
        )
        # 使用前馈网络进行计算
        ffn_output = self.ffn(inputs=attention_outputs[0])
        # 应用激活函数
        ffn_output = self.activation(ffn_output)
        # 再次使用全连接层进行计算
        ffn_output = self.ffn_output(inputs=ffn_output)
        # 对输出进行 dropout 处理
        ffn_output = self.dropout(inputs=ffn_output, training=training)
        # 将 dropout 处理后的输出和注意力计算的结果相加，并进行 LayerNormalization 处理
        hidden_states = self.full_layer_layer_norm(inputs=ffn_output + attention_outputs[0])

        # 如果需要输出注意力权重，则将注意力输出添加到返回结果中
        outputs = (hidden_states,) + attention_outputs[1:]

        # 返回计算结果
        return outputs

    # build 方法用于构建层，根据输入形状构建内部的网络层
    def build(self, input_shape=None):
        # 如果已经构建过了，则直接返回
        if self.built:
            return
        # 设置标志为已构建
        self.built = True
        # 构建注意力层
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        # 构建前馈网络
        if getattr(self, "ffn", None) is not None:
            with tf.name_scope(self.ffn.name):
                self.ffn.build([None, None, self.config.hidden_size])
        # 构建前馈网络输出层
        if getattr(self, "ffn_output", None) is not None:
            with tf.name_scope(self.ffn_output.name):
                self.ffn_output.build([None, None, self.config.intermediate_size])
        # 构建 LayerNormalization 层
        if getattr(self, "full_layer_layer_norm", None) is not None:
            with tf.name_scope(self.full_layer_layer_norm.name):
                self.full_layer_layer_norm.build([None, None, self.config.hidden_size])


class TFAlbertLayerGroup(tf.keras.layers.Layer):
    # 初始化函数，接受一个 AlbertConfig 对象和其他关键字参数
    def __init__(self, config: AlbertConfig, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 创建 Albert 层列表，包含指定数量的 Albert 层
        self.albert_layers = [
            TFAlbertLayer(config, name=f"albert_layers_._{i}") for i in range(config.inner_group_num)
        ]

    # 调用函数，对输入的隐藏状态进行 Albert 层的处理
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        output_hidden_states: bool,
        training: bool = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        # 如果需要输出隐藏状态，则初始化一个空的元组
        layer_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力权重，则初始化一个空的元组
        layer_attentions = () if output_attentions else None

        # 遍历 Albert 层列表，并对每一层进行处理
        for layer_index, albert_layer in enumerate(self.albert_layers):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到隐藏状态元组中
            if output_hidden_states:
                layer_hidden_states = layer_hidden_states + (hidden_states,)

            # 调用当前 Albert 层的处理函数，得到该层的输出
            layer_output = albert_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[layer_index],
                output_attentions=output_attentions,
                training=training,
            )
            # 更新隐藏状态为当前层的输出
            hidden_states = layer_output[0]

            # 如果需要输出注意力权重，则将当前层的注意力权重添加到注意力权重元组中
            if output_attentions:
                layer_attentions = layer_attentions + (layer_output[1],)

        # 添加最后一层的隐藏状态到隐藏状态元组中
        if output_hidden_states:
            layer_hidden_states = layer_hidden_states + (hidden_states,)

        # 返回处理后的结果，注意过滤掉空值
        return tuple(v for v in [hidden_states, layer_hidden_states, layer_attentions] if v is not None)

    # 构建函数，用于构建模型的层
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 设置构建标志为 True
        self.built = True
        # 如果已经初始化了 Albert 层列表，则对每一层进行构建
        if getattr(self, "albert_layers", None) is not None:
            for layer in self.albert_layers:
                # 在当前层的命名空间下构建该层
                with tf.name_scope(layer.name):
                    layer.build(None)
class TFAlbertTransformer(tf.keras.layers.Layer):
    # 定义 TFAlbertTransformer 类，继承自 tf.keras.layers.Layer
    def __init__(self, config: AlbertConfig, **kwargs):
        # 初始化函数，接受 AlbertConfig 类型的 config 参数和其他关键字参数
        super().__init__(**kwargs)
        # 调用父类的初始化函数

        self.num_hidden_layers = config.num_hidden_layers
        # 隐藏层的数量
        self.num_hidden_groups = config.num_hidden_groups
        # 隐藏层分组的数量
        self.layers_per_group = int(config.num_hidden_layers / config.num_hidden_groups)
        # 每个隐藏组中的层数
        self.embedding_hidden_mapping_in = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="embedding_hidden_mapping_in",
        )
        # 创建一个全连接层，用于将输入映射到隐藏层
        self.albert_layer_groups = [
            TFAlbertLayerGroup(config, name=f"albert_layer_groups_._{i}") for i in range(config.num_hidden_groups)
        ]
        # 创建 AlbertLayerGroup 对象的列表
        self.config = config
        # 保存配置信息

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
        # 定义 call 方法，接受多个参数并返回 TFBaseModelOutput 或 Tuple[tf.Tensor] 类型的结果
        hidden_states = self.embedding_hidden_mapping_in(inputs=hidden_states)
        # 将输入数据映射到隐藏层
        all_attentions = () if output_attentions else None
        # 如果需要输出注意力权重，则初始化 all_attentions 为空元组，否则为 None
        all_hidden_states = (hidden_states,) if output_hidden_states else None
        # 如果需要输出隐藏状态，则初始化 all_hidden_states 为包含 hidden_states 的元组，否则为 None

        for i in range(self.num_hidden_layers):
            # 遍历隐藏层
            group_idx = int(i / (self.num_hidden_layers / self.num_hidden_groups))
            # 计算隐藏组的索引
            layer_group_output = self.albert_layer_groups[group_idx](
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[group_idx * self.layers_per_group : (group_idx + 1) * self.layers_per_group],
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                training=training,
            )
            # 调用 AlbertLayerGroup 对象的 call 方法
            hidden_states = layer_group_output[0]

            if output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]
                # 如果需要输出注意力权重，则更新 all_attentions

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                # 如果需要输出隐藏状态，则更新 all_hidden_states

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
            # 如果不需要返回字典，则返回包含非空值的元组

        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )
        # 返回 TFBaseModelOutput 对象
    # 构建模型，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果存在嵌入层映射，则构建嵌入层映射
        if getattr(self, "embedding_hidden_mapping_in", None) is not None:
            # 使用嵌入层映射的名称作为命名空间
            with tf.name_scope(self.embedding_hidden_mapping_in.name):
                # 构建嵌入层映射
                self.embedding_hidden_mapping_in.build([None, None, self.config.embedding_size])
        # 如果存在 ALBERT 层组，则逐个构建每个 ALBERT 层
        if getattr(self, "albert_layer_groups", None) is not None:
            for layer in self.albert_layer_groups:
                # 使用 ALBERT 层的名称作为命名空间
                with tf.name_scope(layer.name):
                    # 构建 ALBERT 层
                    layer.build(None)
# TFAlbertPreTrainedModel 类，用于处理权重初始化以及下载和加载预训练模型的简单接口
class TFAlbertPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # AlbertConfig 类的引用，用于配置模型
    config_class = AlbertConfig
    # 模型的基础名称前缀
    base_model_prefix = "albert"


# TFAlbertMLMHead 类，用于处理 Albert 模型的 Masked Language Model 头部
class TFAlbertMLMHead(tf.keras.layers.Layer):
    # 初始化函数
    def __init__(self, config: AlbertConfig, input_embeddings: tf.keras.layers.Layer, **kwargs):
        super().__init__(**kwargs)

        # 配置对象
        self.config = config
        # 嵌入维度大小
        self.embedding_size = config.embedding_size
        # 全连接层，用于转换输入特征
        self.dense = tf.keras.layers.Dense(
            config.embedding_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 激活函数
        if isinstance(config.hidden_act, str):
            self.activation = get_tf_activation(config.hidden_act)
        else:
            self.activation = config.hidden_act

        # LayerNormalization 层，用于标准化输入特征
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")

        # 解码器，用于输出权重
        self.decoder = input_embeddings

    # 构建函数
    def build(self, input_shape=None):
        # 输出偏置
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")
        self.decoder_bias = self.add_weight(
            shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="decoder/bias"
        )

        if self.built:
            return
        self.built = True
        # 构建全连接层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 构建 LayerNormalization 层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.embedding_size])

    # 获取输出权重
    def get_output_embeddings(self) -> tf.keras.layers.Layer:
        return self.decoder

    # 设置输出权重
    def set_output_embeddings(self, value: tf.Variable):
        self.decoder.weight = value
        self.decoder.vocab_size = shape_list(value)[0]

    # 获取偏置
    def get_bias(self) -> Dict[str, tf.Variable]:
        return {"bias": self.bias, "decoder_bias": self.decoder_bias}

    # 设置偏置
    def set_bias(self, value: tf.Variable):
        self.bias = value["bias"]
        self.decoder_bias = value["decoder_bias"]
        self.config.vocab_size = shape_list(value["bias"])[0]
    # 使用全连接层对隐藏状态进行线性变换
    hidden_states = self.dense(inputs=hidden_states)
    # 应用激活函数
    hidden_states = self.activation(hidden_states)
    # 对隐藏状态进行 Layer Normalization
    hidden_states = self.LayerNorm(inputs=hidden_states)
    # 获取序列长度
    seq_length = shape_list(tensor=hidden_states)[1]
    # 将隐藏状态重塑为二维张量
    hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.embedding_size])
    # 使用矩阵乘法进行线性变换，其中权重为 decoder 的权重矩阵的转置
    hidden_states = tf.matmul(a=hidden_states, b=self.decoder.weight, transpose_b=True)
    # 将结果重新塑造为三维张量
    hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
    # 添加解码器偏置
    hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.decoder_bias)

    # 返回隐藏状态
    return hidden_states
# 使用 keras_serializable 装饰器将类 TFAlbertMainLayer 序列化为 Keras 模型
@keras_serializable
class TFAlbertMainLayer(tf.keras.layers.Layer):
    # 设置配置类为 AlbertConfig
    config_class = AlbertConfig

    # 初始化方法，接受 AlbertConfig 对象和是否添加池化层的参数
    def __init__(self, config: AlbertConfig, add_pooling_layer: bool = True, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 保存传入的配置对象
        self.config = config

        # 创建 TFAlbertEmbeddings 对象
        self.embeddings = TFAlbertEmbeddings(config, name="embeddings")
        # 创建 TFAlbertTransformer 对象
        self.encoder = TFAlbertTransformer(config, name="encoder")
        # 如果需要添加池化层，则创建 Dense 层作为池化层
        self.pooler = (
            tf.keras.layers.Dense(
                units=config.hidden_size,
                kernel_initializer=get_initializer(config.initializer_range),
                activation="tanh",
                name="pooler",
            )
            if add_pooling_layer
            else None
        )

    # 获取输入嵌入层对象
    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        return self.embeddings

    # 设置输入嵌入层的权重
    def set_input_embeddings(self, value: tf.Variable):
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    # 剪枝模型中的头部
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    # 调用方法，接受多个输入参数
    @unpack_inputs
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
        training: bool = False,
    # 构建模型，接受输入形状参数
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在嵌入层对象，则构建嵌入层
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        # 如果存在编码器对象，则构建编码器
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果存在池化层对象，则构建池化层
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build([None, None, self.config.hidden_size])


# 使用 dataclass 装饰器将类 TFAlbertForPreTrainingOutput 转换为数据类
@dataclass
class TFAlbertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`TFAlbertForPreTraining`].
    Args:
        prediction_logits (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            预测语言建模头部的预测分数（SoftMax之前每个词汇标记的分数）。
        sop_logits (`tf.Tensor` of shape `(batch_size, 2)`):
            下一个序列预测（分类）头部的预测分数（SoftMax之前的True/False延续分数）。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            形状为`(batch_size, sequence_length, hidden_size)`的`tf.Tensor`元组。

            每个层的输出隐藏状态加上初始嵌入输出。
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            形状为`(batch_size, num_heads, sequence_length, sequence_length)`的`tf.Tensor`元组。

            在注意力SoftMax之后的注意力权重，用于计算自注意力头部中的加权平均值。
    """

    loss: tf.Tensor = None
    prediction_logits: tf.Tensor = None
    sop_logits: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None
# 定义 ALBERT 模型的文档字符串，包含了模型的继承关系和使用提示
ALBERT_START_DOCSTRING = r"""

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
        config ([`AlbertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义 ALBERT 模型的输入文档字符串
ALBERT_INPUTS_DOCSTRING = r"""
"""

# 添加起始文档字符串到 ALBERT 模型类中
@add_start_docstrings(
    "The bare Albert Model transformer outputting raw hidden-states without any specific head on top.",
    ALBERT_START_DOCSTRING,
)
# 定义 TFAlbertModel 类，继承自 TFAlbertPreTrainedModel
class TFAlbertModel(TFAlbertPreTrainedModel):
    # 初始化方法
    def __init__(self, config: AlbertConfig, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 创建 ALBERT 主层对象
        self.albert = TFAlbertMainLayer(config, name="albert")

    # 解包输入参数
    @unpack_inputs
    # 使用装饰器添加模型输入的文档字符串，描述输入参数的含义和格式
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 使用装饰器添加代码示例的文档字符串，描述模型的检查点、输出类型和配置类
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义模型的调用方法，接受输入参数并返回模型输出
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 输入的 token ID 序列，默认为 None
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力遮罩，默认为 None
        token_type_ids: np.ndarray | tf.Tensor | None = None,  # token 类型 ID，默认为 None
        position_ids: np.ndarray | tf.Tensor | None = None,  # 位置 ID，默认为 None
        head_mask: np.ndarray | tf.Tensor | None = None,  # 头部遮罩，默认为 None
        inputs_embeds: np.ndarray | tf.Tensor | None = None,  # 嵌入向量，默认为 None
        output_attentions: Optional[bool] = None,  # 是否输出注意力，默认为 None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，默认为 None
        return_dict: Optional[bool] = None,  # 是否以字典形式返回，默认为 None
        training: Optional[bool] = False,  # 是否处于训练模式，默认为 False
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:  # 返回类型为模型输出或元组
        # 调用 ALBERT 模型的 call 方法，传递参数并获取输出
        outputs = self.albert(
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
        # 返回模型输出
        return outputs

    # 构建模型，设置模型的构建过程
    def build(self, input_shape=None):
        # 如果模型已构建，则直接返回
        if self.built:
            return
        # 标记模型已构建
        self.built = True
        # 如果模型包含 ALBERT 模型，则构建 ALBERT 模型
        if getattr(self, "albert", None) is not None:
            # 在指定的命名空间下构建 ALBERT 模型
            with tf.name_scope(self.albert.name):
                # 构建 ALBERT 模型，传入输入形状为 None
                self.albert.build(None)
# 使用 Albert 模型进行预训练，包含一个用于掩码语言建模的头部和一个用于句子顺序预测（分类）的头部
@add_start_docstrings(
    """
    Albert Model with two heads on top for pretraining: a `masked language modeling` head and a `sentence order
    prediction` (classification) head.
    """,
    ALBERT_START_DOCSTRING,
)
class TFAlbertForPreTraining(TFAlbertPreTrainedModel, TFAlbertPreTrainingLoss):
    # 在从 PT 模型加载 TF 模型时，带有 '.' 的名称表示授权的意外/缺失层
    _keys_to_ignore_on_load_unexpected = [r"predictions.decoder.weight"]

    def __init__(self, config: AlbertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 获取标签数量
        self.num_labels = config.num_labels

        # 创建 Albert 主层
        self.albert = TFAlbertMainLayer(config, name="albert")
        # 创建掩码语言建模头部
        self.predictions = TFAlbertMLMHead(config, input_embeddings=self.albert.embeddings, name="predictions")
        # 创建句子顺序预测头部
        self.sop_classifier = TFAlbertSOPHead(config, name="sop_classifier")

    # 获取语言建模头部
    def get_lm_head(self) -> tf.keras.layers.Layer:
        return self.predictions

    # 模型调用方法，包括输入参数和输出文档字符串
    @unpack_inputs
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFAlbertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
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
        sentence_order_label: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[TFAlbertForPreTrainingOutput, Tuple[tf.Tensor]]:
        r"""
        Return:

        Example:

        ```py
        >>> import tensorflow as tf
        >>> from transformers import AutoTokenizer, TFAlbertForPreTraining

        >>> tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
        >>> model = TFAlbertForPreTraining.from_pretrained("albert-base-v2")

        >>> input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True))[None, :]
        >>> # Batch size 1
        >>> outputs = model(input_ids)

        >>> prediction_logits = outputs.prediction_logits
        >>> sop_logits = outputs.sop_logits
        ```"""

        # 调用 albert 模型进行预测
        outputs = self.albert(
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
        # 获取序列输出和汇总输出
        sequence_output, pooled_output = outputs[:2]
        # 通过预测模型获取预测分数
        prediction_scores = self.predictions(hidden_states=sequence_output)
        # 通过句子顺序分类器获取句子顺序分数
        sop_scores = self.sop_classifier(pooled_output=pooled_output, training=training)
        total_loss = None

        # 如果存在标签和句子顺序标签
        if labels is not None and sentence_order_label is not None:
            d_labels = {"labels": labels}
            d_labels["sentence_order_label"] = sentence_order_label
            # 计算总损失
            total_loss = self.hf_compute_loss(labels=d_labels, logits=(prediction_scores, sop_scores))

        # 如果不返回字典
        if not return_dict:
            output = (prediction_scores, sop_scores) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 返回 TFAlbertForPreTrainingOutput 对象
        return TFAlbertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            sop_logits=sop_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 构建模型
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 albert 模型
        if getattr(self, "albert", None) is not None:
            with tf.name_scope(self.albert.name):
                self.albert.build(None)
        # 如果存在预测模型
        if getattr(self, "predictions", None) is not None:
            with tf.name_scope(self.predictions.name):
                self.predictions.build(None)
        # 如果存在句子顺序分类器
        if getattr(self, "sop_classifier", None) is not None:
            with tf.name_scope(self.sop_classifier.name):
                self.sop_classifier.build(None)
class TFAlbertSOPHead(tf.keras.layers.Layer):
    def __init__(self, config: AlbertConfig, **kwargs):
        super().__init__(**kwargs)  # 调用父类构造函数初始化对象

        self.dropout = tf.keras.layers.Dropout(rate=config.classifier_dropout_prob)  # 定义 dropout 层，用于随机失活
        self.classifier = tf.keras.layers.Dense(  # 定义全连接层，用于分类任务
            units=config.num_labels,  # 分类的类别数
            kernel_initializer=get_initializer(config.initializer_range),  # 权重初始化器
            name="classifier",  # 层名称
        )
        self.config = config  # 存储配置信息

    def call(self, pooled_output: tf.Tensor, training: bool) -> tf.Tensor:
        dropout_pooled_output = self.dropout(inputs=pooled_output, training=training)  # 应用 dropout
        logits = self.classifier(inputs=dropout_pooled_output)  # 计算分类结果的 logits

        return logits  # 返回分类结果的 logits

    def build(self, input_shape=None):
        if self.built:  # 如果已经构建过，直接返回
            return
        self.built = True  # 标记已构建
        if getattr(self, "classifier", None) is not None:  # 如果分类器已定义
            with tf.name_scope(self.classifier.name):  # 使用分类器的名称作为命名空间
                self.classifier.build([None, None, self.config.hidden_size])  # 构建分类器的参数

@add_start_docstrings("""Albert Model with a `language modeling` head on top.""", ALBERT_START_DOCSTRING)
class TFAlbertForMaskedLM(TFAlbertPreTrainedModel, TFMaskedLanguageModelingLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"predictions.decoder.weight"]

    def __init__(self, config: AlbertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)  # 调用父类构造函数初始化对象

        self.albert = TFAlbertMainLayer(config, add_pooling_layer=False, name="albert")  # Albert 主层
        self.predictions = TFAlbertMLMHead(config, input_embeddings=self.albert.embeddings, name="predictions")  # 预测层

    def get_lm_head(self) -> tf.keras.layers.Layer:
        return self.predictions  # 返回语言模型头部

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFMaskedLMOutput, config_class=_CONFIG_FOR_DOC)
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
    ) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

        Returns:

        Example:

        ```py
        >>> import tensorflow as tf
        >>> from transformers import AutoTokenizer, TFAlbertForMaskedLM

        >>> tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
        >>> model = TFAlbertForMaskedLM.from_pretrained("albert-base-v2")

        >>> # add mask_token
        >>> inputs = tokenizer(f"The capital of [MASK] is Paris.", return_tensors="tf")
        >>> logits = model(**inputs).logits

        >>> # retrieve index of [MASK]
        >>> mask_token_index = tf.where(inputs.input_ids == tokenizer.mask_token_id)[0][1]
        >>> predicted_token_id = tf.math.argmax(logits[0, mask_token_index], axis=-1)
        >>> tokenizer.decode(predicted_token_id)
        'france'
        ```

        ```py
        >>> labels = tokenizer("The capital of France is Paris.", return_tensors="tf")["input_ids"]
        >>> labels = tf.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)
        >>> outputs = model(**inputs, labels=labels)
        >>> round(float(outputs.loss), 2)
        0.81
        ```
        """
        # 调用 Albert 模型，传入输入参数并获取输出
        outputs = self.albert(
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
        # 从输出中获取序列输出
        sequence_output = outputs[0]
        # 根据序列输出计算预测分数
        prediction_scores = self.predictions(hidden_states=sequence_output, training=training)
        # 如果存在标签，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=prediction_scores)

        # 如果不需要以字典形式返回结果
        if not return_dict:
            # 构造输出元组，如果损失不为空，则添加损失到输出元组中
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 以 TFMaskedLMOutput 对象的形式返回结果
        return TFMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    # 构建模型，根据输入形状进行构建
    def build(self, input_shape=None):
        # 如果模型已经构建完成，则直接返回，不进行重复构建
        if self.built:
            return
        # 标记模型已构建
        self.built = True
        # 如果存在 Albert 模型，则构建 Albert 模型
        if getattr(self, "albert", None) is not None:
            # 在 TensorFlow 中为 Albert 模型创建命名空间
            with tf.name_scope(self.albert.name):
                # 根据输入形状构建 Albert 模型
                self.albert.build(None)
        # 如果存在预测层，则构建预测层
        if getattr(self, "predictions", None) is not None:
            # 在 TensorFlow 中为预测层创建命名空间
            with tf.name_scope(self.predictions.name):
                # 根据输入形状构建预测层
                self.predictions.build(None)
@add_start_docstrings(
    """
    Albert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    ALBERT_START_DOCSTRING,
)
class TFAlbertForSequenceClassification(TFAlbertPreTrainedModel, TFSequenceClassificationLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [r"predictions"]  # 在从 PT 模型加载 TF 模型时忽略的未预期/缺失的层
    _keys_to_ignore_on_load_missing = [r"dropout"]  # 在从 PT 模型加载 TF 模型时忽略的未找到的层

    def __init__(self, config: AlbertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels  # 分类标签数量

        self.albert = TFAlbertMainLayer(config, name="albert")  # Albert 主层
        self.dropout = tf.keras.layers.Dropout(rate=config.classifier_dropout_prob)  # dropout 层
        self.classifier = tf.keras.layers.Dense(
            units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )  # 分类器线性层
        self.config = config  # 模型配置

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="vumichien/albert-base-v2-imdb",
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'LABEL_1'",
        expected_loss=0.12,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 输入 token IDs
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力遮罩
        token_type_ids: np.ndarray | tf.Tensor | None = None,  # token 类型 IDs
        position_ids: np.ndarray | tf.Tensor | None = None,  # 位置 IDs
        head_mask: np.ndarray | tf.Tensor | None = None,  # 头部遮罩
        inputs_embeds: np.ndarray | tf.Tensor | None = None,  # 输入嵌入
        output_attentions: Optional[bool] = None,  # 是否返回注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否返回隐藏状态
        return_dict: Optional[bool] = None,  # 是否以字典形式返回结果
        labels: np.ndarray | tf.Tensor | None = None,  # 标签
        training: Optional[bool] = False,  # 是否处于训练模式
    ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 调用 ALBERT 模型，传入输入的各项参数，返回模型输出
        outputs = self.albert(
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
        # 从 ALBERT 模型输出中获取池化后的输出
        pooled_output = outputs[1]
        # 对池化输出进行 dropout 处理
        pooled_output = self.dropout(inputs=pooled_output, training=training)
        # 将池化后的输出传入分类器，得到 logits
        logits = self.classifier(inputs=pooled_output)
        # 计算损失，如果没有标签则损失为 None
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        # 如果不要求返回字典，则构建返回的元组
        if not return_dict:
            # 将 logits 与额外的输出组成元组返回，如果损失不为 None 则加入损失
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果要求返回字典，则构建 TFSequenceClassifierOutput 对象返回
        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建过了，则直接返回
        if self.built:
            return
        # 将构建标志置为 True
        self.built = True
        # 如果 ALBERT 模型已经存在，则构建 ALBERT 模型
        if getattr(self, "albert", None) is not None:
            with tf.name_scope(self.albert.name):
                self.albert.build(None)
        # 如果分类器存在，则构建分类器
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
    # 定义一个带有标记分类头部的 Albert 模型，用于命名实体识别（NER）等任务
    @add_start_docstrings(
        """
        Albert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
        Named-Entity-Recognition (NER) tasks.
        """,
        ALBERT_START_DOCSTRING,
    )
    class TFAlbertForTokenClassification(TFAlbertPreTrainedModel, TFTokenClassificationLoss):
        # 在加载 TF 模型时，带有 '.' 的名称表示授权的意外/缺失层
        _keys_to_ignore_on_load_unexpected = [r"pooler", r"predictions"]
        _keys_to_ignore_on_load_missing = [r"dropout"]

        def __init__(self, config: AlbertConfig, *inputs, **kwargs):
            super().__init__(config, *inputs, **kwargs)

            # 获取标签数量
            self.num_labels = config.num_labels

            # 创建 Albert 主层
            self.albert = TFAlbertMainLayer(config, add_pooling_layer=False, name="albert")
            # 获取分类器的丢弃概率
            classifier_dropout_prob = (
                config.classifier_dropout_prob
                if config.classifier_dropout_prob is not None
                else config.hidden_dropout_prob
            )
            # 创建丢弃层
            self.dropout = tf.keras.layers.Dropout(rate=classifier_dropout_prob)
            # 创建分类器层
            self.classifier = tf.keras.layers.Dense(
                units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
            )
            self.config = config

        # 模型调用方法
        @unpack_inputs
        @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        # 调用 ALBERT 模型进行前向传播，获取输出结果
        outputs = self.albert(
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
        # 获取 ALBERT 模型的输出序列
        sequence_output = outputs[0]
        # 对输出序列进行 dropout 处理
        sequence_output = self.dropout(inputs=sequence_output, training=training)
        # 将处理后的序列输入分类器，得到分类结果
        logits = self.classifier(inputs=sequence_output)
        # 如果有标签，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        # 如果不返回字典形式的结果
        if not return_dict:
            # 组装输出结果
            output = (logits,) + outputs[2:]
            # 返回结果
            return ((loss,) + output) if loss is not None else output

        # 返回字典形式的结果
        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 构建模型
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果 ALBERT 模型存在，则构建 ALBERT 模型
        if getattr(self, "albert", None) is not None:
            with tf.name_scope(self.albert.name):
                self.albert.build(None)
        # 如果分类器存在，则构建分类器
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
# 定义一个带有用于提取问题-回答任务的跨度分类头部的 Albert 模型，例如 SQuAD（在隐藏状态输出之上的线性层，用于计算“跨度起始对数”和“跨度结束对数”）。
@add_start_docstrings(
    """
    Albert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    ALBERT_START_DOCSTRING,
)
class TFAlbertForQuestionAnswering(TFAlbertPreTrainedModel, TFQuestionAnsweringLoss):
    # 在加载 TF 模型时，带有 '.' 的名称表示授权的意外/缺失层
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"predictions"]

    def __init__(self, config: AlbertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 获取标签数量
        self.num_labels = config.num_labels

        # 创建 Albert 主层对象
        self.albert = TFAlbertMainLayer(config, add_pooling_layer=False, name="albert")
        # 创建用于问题-回答任务的输出层
        self.qa_outputs = tf.keras.layers.Dense(
            units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )
        self.config = config

    # 调用模型前向传播
    @unpack_inputs
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="vumichien/albert-base-v2-squad2",
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        qa_target_start_index=12,
        qa_target_end_index=13,
        expected_output="'a nice puppet'",
        expected_loss=7.36,
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
        start_positions: np.ndarray | tf.Tensor | None = None,
        end_positions: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
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
        # 传入的参数包括起始位置和结束位置的标签张量，用于计算标记分类损失
        outputs = self.albert(
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
        # 获取模型输出中的序列输出
        sequence_output = outputs[0]
        # 将序列输出传递给QA输出层，获得logits
        logits = self.qa_outputs(inputs=sequence_output)
        # 将logits分割为起始和结束logits
        start_logits, end_logits = tf.split(value=logits, num_or_size_splits=2, axis=-1)
        # 去除多余的维度
        start_logits = tf.squeeze(input=start_logits, axis=-1)
        end_logits = tf.squeeze(input=end_logits, axis=-1)
        # 初始化损失为None
        loss = None

        # 如果存在起始位置和结束位置的标签，则计算损失
        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            loss = self.hf_compute_loss(labels=labels, logits=(start_logits, end_logits))

        # 如果不要求返回字典，则返回输出
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]

            return ((loss,) + output) if loss is not None else output

        # 如果要求返回字典，则返回TFQuestionAnsweringModelOutput对象
        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 将构建状态设置为True
        self.built = True
        # 如果存在albert模型，则构建albert模型
        if getattr(self, "albert", None) is not None:
            with tf.name_scope(self.albert.name):
                self.albert.build(None)
        # 如果存在qa_outputs模型，则构建qa_outputs模型
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])
# 使用 Albert 模型在顶部添加了一个用于多选分类的分类头部（在池化输出的顶部添加了一个线性层和一个 softmax 层），例如用于 RocStories/SWAG 任务
@add_start_docstrings(
    """
    Albert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    ALBERT_START_DOCSTRING,
)
class TFAlbertForMultipleChoice(TFAlbertPreTrainedModel, TFMultipleChoiceLoss):
    # 当从 PT 模型加载 TF 模型时，以 '.' 结尾的名称代表了预授权的意外/缺失的层
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"predictions"]
    # 当从 PT 模型加载 TF 模型时，代表了预授权的缺失的层
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config: AlbertConfig, *inputs, **kwargs):
        # 调用 TFAlbertPreTrainedModel 的初始化函数
        super().__init__(config, *inputs, **kwargs)

        # 初始化 Albert 主层
        self.albert = TFAlbertMainLayer(config, name="albert")
        # 添加 dropout 层
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        # 添加分类器
        self.classifier = tf.keras.layers.Dense(
            units=1, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        # 设置配置参数
        self.config = config

    # 模型前向传播函数
    @unpack_inputs
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
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
    ) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)
        """
        # 如果提供了 input_ids，则获取其第二维度的大小作为选择数量，获取序列长度
        if input_ids is not None:
            num_choices = shape_list(input_ids)[1]
            seq_length = shape_list(input_ids)[2]
        else:
            # 否则，获取 inputs_embeds 的第二维度的大小作为选择数量，获取序列长度
            num_choices = shape_list(inputs_embeds)[1]
            seq_length = shape_list(inputs_embeds)[2]

        # 将 input_ids 平坦化为二维张量（若提供），否则设为 None
        flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
        # 将 attention_mask 平坦化为二维张量（若提供），否则设为 None
        flat_attention_mask = (
            tf.reshape(tensor=attention_mask, shape=(-1, seq_length)) if attention_mask is not None else None
        )
        # 将 token_type_ids 平坦化为二维张量（若提供），否则设为 None
        flat_token_type_ids = (
            tf.reshape(tensor=token_type_ids, shape=(-1, seq_length)) if token_type_ids is not None else None
        )
        # 将 position_ids 平坦化为二维张量（若提供），否则设为 None
        flat_position_ids = (
            tf.reshape(tensor=position_ids, shape=(-1, seq_length)) if position_ids is not None else None
        )
        # 将 inputs_embeds 平坦化为三维张量（若提供），否则设为 None
        flat_inputs_embeds = (
            tf.reshape(tensor=inputs_embeds, shape=(-1, seq_length, shape_list(inputs_embeds)[3]))
            if inputs_embeds is not None
            else None
        )
        # 使用 ALBERT 模型处理平坦化后的输入
        outputs = self.albert(
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
        # 提取汇聚的输出，应用 dropout，然后通过分类器获取 logits
        pooled_output = outputs[1]
        pooled_output = self.dropout(inputs=pooled_output, training=training)
        logits = self.classifier(inputs=pooled_output)
        # 将 logits 重新整形为二维张量
        reshaped_logits = tf.reshape(tensor=logits, shape=(-1, num_choices))
        # 若提供了 labels，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=reshaped_logits)

        # 若不要求返回字典，则返回相应的输出
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 否则，返回包含 loss、logits、hidden_states 和 attentions 的 TFMultipleChoiceModelOutput 对象
        return TFMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    # 构建模型，如果已经构建过则直接返回
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在 albert 属性，则构建 albert 模型
        if getattr(self, "albert", None) is not None:
            # 使用 albert 的名称作为命名空间
            with tf.name_scope(self.albert.name):
                # 构建 albert 模型
                self.albert.build(None)
        # 如果存在 classifier 属性，则构建 classifier 模型
        if getattr(self, "classifier", None) is not None:
            # 使用 classifier 的名称作为命名空间
            with tf.name_scope(self.classifier.name):
                # 构建 classifier 模型，输入形状为 [None, None, self.config.hidden_size]
                self.classifier.build([None, None, self.config.hidden_size])
```