# `.\transformers\models\xlm_roberta\modeling_tf_xlm_roberta.py`

```py
# 设置文本编码为 utf-8
# 版权声明
# 2021年，Facebook AI Research 和 HuggingFace Inc. team 版权所有
# 版权所有 2018, NVIDIA 公司。 保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可；
# 除非遵守许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 基于“AS IS”“不提供任何形式的担保或条件”分发，
# 无论是明示的还是暗示的。
# 有关具体语言管理权限和
# 许可证下的限制
""" TF 2.0 XLM-RoBERTa model."""

# 导入必要的库，包括未来的注释
from __future__ import annotations
import math  # 导入数学库
import warnings  # 导入警告模块
from typing import Optional, Tuple, Union  # 导入类型提示

import numpy as np  # 导入numpy库
import tensorflow as tf  # 导入tensorflow库

# 导入必要的输出工具，包括不同类型的模型输出
from ...activations_tf import get_tf_activation  # 从transformers库中导入激活函数
from ...modeling_tf_outputs import (  # 从transformers库中导入模型输出
    TFBaseModelOutputWithPastAndCrossAttentions,  # 基础模型输出和过去注意力权重
    TFBaseModelOutputWithPoolingAndCrossAttentions,  # 带池化和交叉注意力的基础模型输出
    TFCausalLMOutputWithCrossAttentions,  # 带交叉注意力的因果语言模型输出
    TFMaskedLMOutput,  # 掩码语言模型输出
    TFMultipleChoiceModelOutput,  # 多选模型输出
    TFQuestionAnsweringModelOutput,  # 问答模型输出
    TFSequenceClassifierOutput,  # 序列分类器输出
    TFTokenClassifierOutput,  # 令牌分类器输出
)
# 导入常用工具和模型配置
from ...modeling_tf_utils import (  # 从transformers库中导入TF模型工具
    TFCausalLanguageModelingLoss,  # 因果语言模型损失函数
    TFMaskedLanguageModelingLoss,  # 掩码语言模型损失函数
    TFModelInputType,  # 模型输入类型
    TFMultipleChoiceLoss,  # 多选任务损失函数
    TFPreTrainedModel,  # 预训练模型基类
    TFQuestionAnsweringLoss,  # 问答任务损失函数
    TFSequenceClassificationLoss,  # 序列分类任务损失函数
    TFTokenClassificationLoss,  # 令牌分类任务损失函数
    get_initializer,  # 获取初始化器
    keras_serializable,  # keras可序列化
    unpack_inputs,  # 解包输入
)
# 导入工具，包括嵌入范围检查、形状列表和稳定的 softmax 函数
from ...tf_utils import (  # 从transformers库中导入tf工具
    check_embeddings_within_bounds,  # 检查嵌入是否在范围内
    shape_list,  # 获取张量的形状列表
    stable_softmax,  # 稳定的 softmax 函数
)
# 导入工具，包括添加代码示例文档字符串、添加模型前文档字符串、添加转发模型前文档字符串和记录
from ...utils import (  # 从transformers库中导入工具
    add_code_sample_docstrings,  # 添加代码示例文档字符串
    add_start_docstrings,  # 添加起始文档字符串
    add_start_docstrings_to_model_forward,  # 添加模型前转发前文档字符串
    logging,  # 记录工具
)

# 获取记录器
logger = logging.get_logger(__name__)

# 获取记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置
_CHECKPOINT_FOR_DOC = "xlm-roberta-base"
_CONFIG_FOR_DOC = "XLMRobertaConfig"

# XLM-RoBERTa 预训练模型列表
TF_XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "xlm-roberta-base",
    "xlm-roberta-large",
    "joeddav/xlm-roberta-large-xnli",
    "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    # 查看所有 XLM-RoBERTa 模型，请访问 https://huggingface.co/models?filter=xlm-roberta
]

# XLM-RoBERTa 模型起始文档字符串
XLM_ROBERTA_START_DOCSTRING = r"""

    该模型继承自 [`TFPreTrainedModel`]。查看超类文档了解库实现的通用方法（如下载或保存、调整输入嵌入、修剪头等）。

    该模型还是 [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) 的子类。
    使用它作为常规 TF 2.0 Keras 模型，并参考 TF 2.0 文档，了解一切与通用用法和行为相关的事情。

    <提示>

    `transformers` 中的 TensorFlow 模型和层接受两种格式的输入:
    # 定义了一些关于输入格式的说明，说明了两种常见的输入格式：关键字参数和列表、元组或字典作为第一个位置参数
    # 支持第二种格式的原因是，当将输入传递给模型和层时，Keras 方法更喜欢这种格式
    # 对于使用诸如 `model.fit()` 这样的方法，只需要以 `model.fit()` 支持的任何格式传递输入和标签，即可正常工作！
    # 然而，如果要在 Keras 方法之外使用第二种格式，比如在使用 Keras `Functional` API 创建自己的层或模型时，有三种可能性可以用来收集第一个位置参数中的所有输入张量：
    # - 仅使用 `input_ids` 作为参数传递：`model(input_ids)`
    # - 使用长度可变的列表，按照文档字符串中给定的顺序，包含一个或多个输入张量：`model([input_ids, attention_mask])` 或 `model([input_ids, attention_mask, token_type_ids])`
    # - 使用一个字典，将一个或多个输入张量与文档字符串中给定的输入名称关联起来：`model({"input_ids": input_ids, "token_type_ids": token_type_ids})`
    # 注意，在使用[子类化](https://keras.io/guides/making_new_layers_and_models_via_subclassing/)创建模型和层时，不需要担心以上任何问题，因为可以像对待其他 Python 函数一样传递输入！
    
    # Parameters 是一个模型的参数说明部分
    # config 是一个列表，包含了模型的所有参数，使用 `XLMRobertaConfig` 类进行定义
    # 通过使用配置文件初始化模型不会加载与模型关联的权重，只会加载配置信息
    # 若要加载模型权重，可以查看 [`~PreTrainedModel.from_pretrained`] 方法
# XLM_ROBERTA_INPUTS_DOCSTRING是一个文档字符串，用于描述XLMRoberta的输入参数的说明
XLM_ROBERTA_INPUTS_DOCSTRING = r"""
"""

# 定义TFXLMRobertaEmbeddings类，继承自tf.keras.layers.Layer
# 用于生成XLMRoberta模型的嵌入层
class TFXLMRobertaEmbeddings(tf.keras.layers.Layer):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    # 初始化函数
    def __init__(self, config, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 设定padding的索引为1
        self.padding_idx = 1
        # 保存配置参数
        self.config = config
        # 隐藏层的大小
        self.hidden_size = config.hidden_size
        # 位置编码的最大长度
        self.max_position_embeddings = config.max_position_embeddings
        # 初始化范围
        self.initializer_range = config.initializer_range
        # LayerNormalization层，用于标准化
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # Dropout层
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    # 构建函数
    def build(self, input_shape=None):
        # 构建单词的嵌入权重
        with tf.name_scope("word_embeddings"):
            self.weight = self.add_weight(
                name="weight",
                shape=[self.config.vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        # 构建token_type的嵌入权重
        with tf.name_scope("token_type_embeddings"):
            self.token_type_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.config.type_vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        # 构建position的嵌入权重
        with tf.name_scope("position_embeddings"):
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        # 若已构建则返回
        if self.built:
            return
        self.built = True
        if getattr(self, "LayerNorm", None) is not None:
            # 构建LayerNormalization层
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])

    # 根据输入的input_ids生成position_ids
    def create_position_ids_from_input_ids(self, input_ids, past_key_values_length=0):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
        symbols are ignored. This is modified from fairseq's `utils.make_positions`.

        Args:
            input_ids: tf.Tensor
        Returns: tf.Tensor
        """
        # 构建一个掩码，将input_ids中不是padding的符号置为1，是padding的置为0
        mask = tf.cast(tf.math.not_equal(input_ids, self.padding_idx), dtype=input_ids.dtype)
        # 来自于fairseq的`utils.make_positions`修改版，将每个非padding的符号替换为其位置编号
        incremental_indices = (tf.math.cumsum(mask, axis=1) + past_key_values_length) * mask

        return incremental_indices + self.padding_idx

    # 调用函数
    def call(
        self,
        input_ids=None,
        position_ids=None,
        token_type_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
        training=False,
    ):
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        # 确保输入标识符 input_ids 和 inputs_embeds 中至少有一个不是 None
        assert not (input_ids is None and inputs_embeds is None)

        # 如果提供了 input_ids，则验证它们的有效性并获取相应的嵌入
        if input_ids is not None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        # 获取 inputs_embeds 的形状，不包括最后一个维度
        input_shape = shape_list(inputs_embeds)[:-1]

        # 如果没有提供 token_type_ids，将其初始化为全0
        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)

        # 如果没有提供 position_ids，根据是否提供 input_ids 来生成 position_ids
        if position_ids is None:
            if input_ids is not None:
                # 根据 input_ids 生成 position_ids，考虑过去的键值对长度 past_key_values_length
                position_ids = self.create_position_ids_from_input_ids(
                    input_ids=input_ids, past_key_values_length=past_key_values_length
                )
            else:
                # 如果 input_ids 没有提供，生成从 padding_idx 开始的连续位置标识符
                position_ids = tf.expand_dims(
                    tf.range(start=self.padding_idx + 1, limit=input_shape[-1] + self.padding_idx + 1), axis=0
                )

        # 获取位置嵌入和类型嵌入
        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)

        # 将输入嵌入、位置嵌入和类型嵌入相加获得最终的嵌入
        final_embeddings = inputs_embeds + position_embeds + token_type_embeds
        # 应用层归一化
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        # 应用 dropout
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)

        # 返回最终的嵌入张量
        return final_embeddings
# 从transformers.models.bert.modeling_tf_bert.TFBertPooler复制，将Bert->XLMRoberta
class TFXLMRobertaPooler(tf.keras.layers.Layer):
    def __init__(self, config: XLMRobertaConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，用于池化操作
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, # 输出单元数为config.hidden_size
            kernel_initializer=get_initializer(config.initializer_range), # 使用config.initializer_range进行参数初始化
            activation="tanh", # 激活函数为tanh
            name="dense",
        )
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 通过取第一个token的隐藏状态来进行"池化"模型
        first_token_tensor = hidden_states[:, 0] # 取出第一个token的隐藏状态
        pooled_output = self.dense(inputs=first_token_tensor) # 将第一个token的隐藏状态通过全连接层得到池化输出

        return pooled_output

    def build(self, input_shape=None):
        if self.built: # 如果已经构建过，直接返回
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size]) # 构建全连接层


# 从transformers.models.bert.modeling_tf_bert.TFBertSelfAttention复制，将Bert->XLMRoberta
class TFXLMRobertaSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config: XLMRobertaConfig, **kwargs):
        super().__init__(**kwargs)

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads # 注意力头的数量
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads) # 每个注意力头的大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size # 所有注意力头的总大小
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size) # 注意力头大小的平方根

        self.query = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query" # 创建一个Dense层，用于处理查询
        )
        self.key = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key" # 创建一个Dense层，用于处理键
        )
        self.value = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value" # 创建一个Dense层，用于处理值
        )
        self.dropout = tf.keras.layers.Dropout(rate=config.attention_probs_dropout_prob) # 使用config.attention_probs_dropout_prob作为dropout率

        self.is_decoder = config.is_decoder # 是否为解码器
        self.config = config
    # 将输入的张量重塑为 [batch_size, seq_length, num_attention_heads, attention_head_size] 的形状
    tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

    # 将张量从 [batch_size, seq_length, num_attention_heads, attention_head_size] 转置为 [batch_size, num_attention_heads, seq_length, attention_head_size] 的形状
    return tf.transpose(tensor, perm=[0, 2, 1, 3])

    # 定义 call 函数，传入隐藏状态、注意力掩码、头部掩码、编码器隐藏状态、编码器注意力掩码、过去键值对、输出注意力标志、训练标志
    # 返回输出结果
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
    
    # 构建函数，用来构建自注意力层的查询、键和值权重
    def build(self, input_shape=None):
        # 如果已构建过，直接返回
        if self.built:
            return
        # 设置已构建标志为真
        self.built = True
        # 如果存在查询权重，用tf.name_scope包裹，并构建查询权重
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.hidden_size])
        # 如果存在键权重，用tf.name_scope包裹，并构建键权重
        if getattr(self, "key", None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.config.hidden_size])
        # 如果存在值权重，用tf.name_scope包裹，并构建值权重
        if getattr(self, "value", None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.config.hidden_size])
# 用于替换 Transformers 库中 Bert 模型的自注意力层，改为 XLMRobertaSelfOutput 类
class TFXLMRobertaSelfOutput(tf.keras.layers.Layer):
    # 构造函数，接受 XLMRobertaConfig 类型的配置参数
    def __init__(self, config: XLMRobertaConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，用于将输入特征维度转换为隐藏层大小
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建 LayerNormalization 层，用于规范化隐藏状态
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建 Dropout 层，用于在训练时随机置零输入单元，防止过拟合
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        # 保存配置参数
        self.config = config

    # 自注意力层的前向传播函数，接收输入的隐藏状态、输入张量和训练标志
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 使用全连接层处理输入的隐藏状态
        hidden_states = self.dense(inputs=hidden_states)
        # 在训练时对处理后的隐藏状态进行 Dropout
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 将处理后的隐藏状态和输入张量相加，并进行 LayerNormalization
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        # 返回处理后的隐藏状态
        return hidden_states

    # 构建层，用于设置层的输入形状
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在全连接层 dense，则设置其输入形状
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果存在 LayerNormalization 层 LayerNorm，则设置其输入形状
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# 用于替换 Transformers 库中 Bert 模型的注意力层，改为 XLMRobertaAttention 类
class TFXLMRobertaAttention(tf.keras.layers.Layer):
    # 构造函数，接受 XLMRobertaConfig 类型的配置参数
    def __init__(self, config: XLMRobertaConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建 XLMRobertaSelfAttention 层，用于自注意力计算
        self.self_attention = TFXLMRobertaSelfAttention(config, name="self")
        # 创建 XLMRobertaSelfOutput 层，用于处理自注意力计算的输出
        self.dense_output = TFXLMRobertaSelfOutput(config, name="output")

    # 头部剪枝函数，目前未实现
    def prune_heads(self, heads):
        raise NotImplementedError

    # 注意力层的前向传播函数，接收多个输入张量和训练标志
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
        # 使用自注意力层处理输入张量，得到自注意力输出
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
        # 使用 XLMRobertaSelfOutput 层处理自注意力输出，得到最终的注意力输出
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=input_tensor, training=training
        )
        # 如果需要输出注意力矩阵，则将注意力输出和其他信息一并返回
        outputs = (attention_output,) + self_outputs[1:]

        # 返回输出结果
        return outputs
    # 构建神经网络模型，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 置标志已构建为真
        self.built = True
        # 如果存在自注意力机制，则构建自注意力层
        if getattr(self, "self_attention", None) is not None:
            with tf.name_scope(self.self_attention.name):
                self.self_attention.build(None)
        # 如果存在密集输出层，则构建密集输出层
        if getattr(self, "dense_output", None) is not None:
            with tf.name_scope(self.dense_output.name):
                self.dense_output.build(None)
# Copied from transformers.models.bert.modeling_tf_bert.TFBertIntermediate with Bert->XLMRoberta
# 从transformers.models.bert.modeling_tf_bert.TFBertIntermediate复制而来，将Bert替换为XLMRoberta
class TFXLMRobertaIntermediate(tf.keras.layers.Layer):
    def __init__(self, config: XLMRobertaConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，用于中间层，单元数为config.intermediate_size，使用指定的初始化器初始化权重
        self.dense = tf.keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # 如果config.hidden_act是字符串，则将其转换为相应的激活函数；否则直接使用config.hidden_act作为激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        self.config = config

    # 前向传播函数，对输入的hidden_states进行全连接处理并使用中间激活函数
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

    # 构建函数，在此之前没有构建则构建全连接层
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# Copied from transformers.models.bert.modeling_tf_bert.TFBertOutput with Bert->XLMRoberta
# 从transformers.models.bert.modeling_tf_bert.TFBertOutput复制而来，将Bert替换为XLMRoberta
class TFXLMRobertaOutput(tf.keras.layers.Layer):
    def __init__(self, config: XLMRobertaConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，用于输出层，单元数为config.hidden_size，使用指定的初始化器初始化权重
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个LayerNormalization层，用于归一化处理
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建一个Dropout层，用于训练时的随机失活
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    # 前向传播函数，对输入的hidden_states进行全连接处理、dropout处理和LayerNormalization处理
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    # 构建函数，在此之前没有构建则构建全连接层和LayerNormalization层
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# Copied from transformers.models.bert.modeling_tf_bert.TFBertLayer with Bert->XLMRoberta
# 从transformers.models.bert.modeling_tf_bert.TFBertLayer复制而来，将Bert替换为XLMRoberta
class TFXLMRobertaLayer(tf.keras.layers.Layer):
    # 初始化函数，用于创建一个新的 XLMRobertaModel 对象
    def __init__(self, config: XLMRobertaConfig, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 创建一个 XLMRobertaAttention 层对象，命名为 "attention"
        self.attention = TFXLMRobertaAttention(config, name="attention")
        # 判断当前模型是否为解码器
        self.is_decoder = config.is_decoder
        # 判断是否添加了跨注意力机制
        self.add_cross_attention = config.add_cross_attention
        # 如果添加了跨注意力机制
        if self.add_cross_attention:
            # 如果当前模型不是解码器，则抛出异常
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 创建一个新的 XLMRobertaAttention 层对象，命名为 "crossattention"
            self.crossattention = TFXLMRobertaAttention(config, name="crossattention")
        # 创建一个 XLMRobertaIntermediate 层对象，命名为 "intermediate"
        self.intermediate = TFXLMRobertaIntermediate(config, name="intermediate")
        # 创建一个 XLMRobertaOutput 层对象，命名为 "output"
        self.bert_output = TFXLMRobertaOutput(config, name="output")

    # 调用函数，用于实现模型的前向传播
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
        # 如果传入了上一个时间步的键值对缓存，则解码器自注意力的缓存键/值对在位置1和2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 使用self.attention函数计算自注意力
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
        # 获得自注意力的输出
        attention_output = self_attention_outputs[0]

        # 如果是解码器，最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # 如果输出注意力权重，则添加自注意力

        cross_attn_present_key_value = None
        # 如果是解码器且传入了编码器的隐藏状态
        if self.is_decoder and encoder_hidden_states is not None:
            # 如果没有交叉注意力层，则报错
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 交叉注意力缓存的键/值对在past_key_value元组的位置3和4
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 使用self.crossattention函数计算交叉注意力
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
            # 获取交叉注意力的输出
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # 如果输出注意力权重，则添加交叉注意力

            # 将交叉注意力缓存添加到present_key_value元组的位置3和4
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 使用self.intermediate函数计算中间输出
        intermediate_output = self.intermediate(hidden_states=attention_output)
        # 使用self.bert_output函数计算BERT输出
        layer_output = self.bert_output(
            hidden_states=intermediate_output, input_tensor=attention_output, training=training
        )
        outputs = (layer_output,) + outputs  # 如果输出注意力，则添加它们

        # 如果是解码器，将注意力的键/值对作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs
    # 构建模型
    def build(self, input_shape=None):
        # 如果已经构建过则返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在 attention 层则构建它
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        # 如果存在 intermediate 层则构建它
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        # 如果存在 bert_output 层则构建它
        if getattr(self, "bert_output", None) is not None:
            with tf.name_scope(self.bert_output.name):
                self.bert_output.build(None)
        # 如果存在 crossattention 层则构建它
        if getattr(self, "crossattention", None) is not None:
            with tf.name_scope(self.crossattention.name):
                self.crossattention.build(None)
# 从transformers.models.bert.modeling_tf_bert.TFBertEncoder复制代码，并将Bert->XLMRoberta
class TFXLMRobertaEncoder(tf.keras.layers.Layer):
    def __init__(self, config: XLMRobertaConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        # 初始化XLMRoberta的多层编码器，每一层都有一个独特的名称
        self.layer = [TFXLMRobertaLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

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
        # 初始化用于存储所有隐藏状态的元组，如果不输出隐藏状态则为None
        all_hidden_states = () if output_hidden_states else None
        # 初始化用于存储所有注意力权重的元组，如果不输出注意力权重则为None
        all_attentions = () if output_attentions else None
        # 初始化用于存储所有跨层注意力权重的元组，如果不输出跨层注意力权重或者模型没有跨层注意力则为None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 初始化用于存储下一个解码器缓存的元组，如果不使用缓存则为None
        next_decoder_cache = () if use_cache else None
        # 遍历所有编码器层
        for i, layer_module in enumerate(self.layer):
            # 如果输出隐藏状态，则将当前隐藏状态添加到所有隐藏状态元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的过去键值对，如果没有则为None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 调用当前编码器层，得到该层的输出
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
            # 更新当前隐藏状态为当前层的输出中的第一个元素
            hidden_states = layer_outputs[0]

            # 如果使用缓存，则将当前层的输出缓存添加到下一个解码器缓存中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            # 如果输出注意力权重，则将当前层的注意力权重添加到所有注意力权重元组中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                # 如果模型包含跨层注意力并且编码器隐藏状态不为None，则将当前层的跨层注意力权重添加到所有跨层注意力权重元组中
                if self.config.add_cross_attention and encoder_hidden_states is not None:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 添加最后一层的隐藏状态到所有隐藏状态元组中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典，则将非None的输出组成元组返回
        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_attentions, all_cross_attentions] if v is not None
            )

        # 返回字典形式的模型输出
        return TFBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )
    # 构建神经网络模型
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 检查是否存在层，如果存在则逐个构建每一层
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                # 使用 TensorFlow 的命名空间创建层
                with tf.name_scope(layer.name):
                    # 构建层
                    layer.build(None)
# 将该类标记为可序列化的，以便能够在 Keras 中使用
@keras_serializable
# 从 transformers.models.roberta.modeling_tf_roberta.TFRobertaMainLayer 复制而来，将 Roberta 改为 XLMRoberta
class TFXLMRobertaMainLayer(tf.keras.layers.Layer):
    # 配置类变量设置为 XLMRobertaConfig
    config_class = XLMRobertaConfig

    def __init__(self, config, add_pooling_layer=True, **kwargs):
        # 调用父类的构造函数
        super().__init__(**kwargs)

        # 设置类的配置属性
        self.config = config
        self.is_decoder = config.is_decoder

        # 设置隐藏层数、初始化范围等属性
        self.num_hidden_layers = config.num_hidden_layers
        self.initializer_range = config.initializer_range
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.return_dict = config.use_return_dict

        # 创建 XLMRobertaEncoder 对象，名为 encoder
        self.encoder = TFXLMRobertaEncoder(config, name="encoder")
        # 如果需要添加池化层，则创建 XLMRobertaPooler 对象，名为 pooler，否则设为 None
        self.pooler = TFXLMRobertaPooler(config, name="pooler") if add_pooling_layer else None
        # embeddings 必须是最后声明的，以便遵循权重顺序
        # 创建 TFXLMRobertaEmbeddings 对象，名为 embeddings
        self.embeddings = TFXLMRobertaEmbeddings(config, name="embeddings")

    # 从 transformers.models.bert.modeling_tf_bert.TFBertMainLayer.get_input_embeddings 复制而来
    # 获取输入嵌入层对象
    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        return self.embeddings

    # 从 transformers.models.bert.modeling_tf_bert.TFBertMainLayer.set_input_embeddings 复制而来
    # 设置输入嵌入层对象
    def set_input_embeddings(self, value: tf.Variable):
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    # 从 transformers.models.bert.modeling_tf_bert.TFBertMainLayer._prune_heads 复制而来
    # 剪枝模型的注意力头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    @unpack_inputs
    # 从 transformers.models.bert.modeling_tf_bert.TFBertMainLayer.call 复制而来
    # 调用模型
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
        training: bool = False,
    # 定义神经网络的 build 方法，并指定输入形状为 None
    def build(self, input_shape=None):
        # 如果已经构建过网络，则直接返回
        if self.built:
            return
        # 标记网络为已构建
        self.built = True
        # 检查是否存在名为 "encoder" 的属性
        if getattr(self, "encoder", None) is not None:
            # 在 TensorFlow 中创建名为 "encoder" 的命名作用域，构建 encoder 层
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 检查是否存在名为 "pooler" 的属性
        if getattr(self, "pooler", None) is not None:
            # 在 TensorFlow 中创建名为 "pooler" 的命名作用域，构建 pooler 层
            with tf.name_scope(self.pooler.name):
                self.pooler.build(None)
        # 检查是否存在名为 "embeddings" 的属性
        if getattr(self, "embeddings", None) is not None:
            # 在 TensorFlow 中创建名为 "embeddings" 的命名作用域，构建 embeddings 层
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
# 从 transformers.models.roberta.modeling_tf_roberta.TFRobertaPreTrainedModel 复制代码，将 Roberta 替换为 XLMRoberta
class TFXLMRobertaPreTrainedModel(TFPreTrainedModel):
    """
    一个抽象类，处理权重初始化和一个简单的接口来下载和加载预训练模型。
    """

    config_class = XLMRobertaConfig  # 使用 XLMRobertaConfig 配置类
    base_model_prefix = "roberta"  # 指定基础模型前缀为 "roberta"


@add_start_docstrings(
    "The bare XLM RoBERTa Model transformer outputting raw hidden-states without any specific head on top.",  # 添加模型的文档字符串
    XLM_ROBERTA_START_DOCSTRING,
)
# 从 transformers.models.roberta.modeling_tf_roberta.TFRobertaModel 复制代码，将 Roberta 替换为 XLMRoberta, ROBERTA 替换为 XLM_ROBERTA
class TFXLMRobertaModel(TFXLMRobertaPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.roberta = TFXLMRobertaMainLayer(config, name="roberta")  # 初始化 XLMRoberta 主层，并指定名称为 "roberta"

    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPoolingAndCrossAttentions,
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
        ) -> Union[Tuple, TFBaseModelOutputWithPoolingAndCrossAttentions]:
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
        outputs = self.roberta(
            input_ids=input_ids,  # 输入的token IDs
            attention_mask=attention_mask,  # 表示哪些token是填充的，哪些是真实的
            token_type_ids=token_type_ids,  # 分段token IDs，用于处理两个句子的输入
            position_ids=position_ids,  # token在序列中的位置的索引
            head_mask=head_mask,  # 用于掩盖attention的头
            inputs_embeds=inputs_embeds,  # token的嵌入表示
            encoder_hidden_states=encoder_hidden_states,  # 编码器隐藏层的状态，用于decoder的交叉attention
            encoder_attention_mask=encoder_attention_mask,  # 编码器的注意力掩码，用于decoder的交叉attention
            past_key_values=past_key_values,  # 预先计算的attention块的键和值的隐藏状态，用于加速解码
            use_cache=use_cache,  # 是否返回 past_key_values，并且用于加速解码
            output_attentions=output_attentions,  # 是否返回注意力权重
            output_hidden_states=output_hidden_states,  # 是否返回隐藏状态
            return_dict=return_dict,  # 返回输出的字典格式
            training=training,  # 是否处于训练阶段
        )

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "roberta", None) is not None:
            with tf.name_scope(self.roberta.name):
                self.roberta.build(None)
# 从transformers.models.roberta.modeling_tf_roberta.TFRobertaLMHead复制代码到XLMRoberta
class TFXLMRobertaLMHead(tf.keras.layers.Layer):
    """XLMRoberta Head for masked language modeling."""
    # 初始化函数
    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.hidden_size = config.hidden_size
        self.dense = tf.keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        self.act = get_tf_activation("gelu")

        # 输出权重与输入嵌入相同，但每个令牌有一个仅输出的偏差。
        self.decoder = input_embeddings

    # 构建函数
    def build(self, input_shape=None):
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")

        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.hidden_size])

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.decoder

    # 设置输出嵌入
    def set_output_embeddings(self, value):
        self.decoder.weight = value
        self.decoder.vocab_size = shape_list(value)[0]

    # 获取偏差
    def get_bias(self):
        return {"bias": self.bias}

    # 设置偏差
    def set_bias(self, value):
        self.bias = value["bias"]
        self.config.vocab_size = shape_list(value["bias"])[0]

    # 调用函数
    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.layer_norm(hidden_states)

        # 通过偏差将其投影回词汇表大小
        seq_length = shape_list(tensor=hidden_states)[1]
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.hidden_size])
        hidden_states = tf.matmul(a=hidden_states, b=self.decoder.weight, transpose_b=True)
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)

        return hidden_states


@add_start_docstrings("""XLM RoBERTa Model with a `language modeling` head on top.""", XLM_ROBERTA_START_DOCSTRING)
# 从transformers.models.roberta.modeling_tf_roberta.TFRobertaForMaskedLM复制代码到XLMRoberta, ROBERTA->XLM_ROBERTA
class TFXLMRobertaForMaskedLM(TFXLMRobertaPreTrainedModel, TFMaskedLanguageModelingLoss):
    # 以'.'开头的名称表示从PT模型加载TF模型时授权的意外/缺失的层
    # 初始化类实例时要忽略的键列表
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head.decoder.weight"]

    # 初始化方法，接收配置和输入，调用父类的初始化方法
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 创建 XLM-Roberta 主层，不添加池化层，命名为 "roberta"
        self.roberta = TFXLMRobertaMainLayer(config, add_pooling_layer=False, name="roberta")
        # 创建 XLM-Roberta 语言模型头层，连接到 embeddings 层，命名为 "lm_head"
        self.lm_head = TFXLMRobertaLMHead(config, self.roberta.embeddings, name="lm_head")

    # 返回语言模型头层
    def get_lm_head(self):
        return self.lm_head

    # 返回前缀偏置名称
    def get_prefix_bias_name(self):
        # 发出警告提示
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        # 返回合成的前缀偏置名称
        return self.name + "/" + self.lm_head.name

    # 使用装饰器添加输入解包、前向模型文档注释和代码示例文档注释后，定义模型前向方法
    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="<mask>",
        expected_output="' Paris'",
        expected_loss=0.1,
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
    ) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        r""" 
        定义函数的返回注解，表明返回值为TFMaskedLMOutput类型或tf.Tensor类型的元组
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional`):
        标签，用于计算掩码语言建模损失。索引应在`[-100, 0, ..., config.vocab_size]`范围内（参见`input_ids` docstring）。索引设置为`-100`的标记将被忽略（被掩码），损失仅计算标签为`[0, ..., config.vocab_size]`范围内的标记
        
        outputs = self.roberta(
            input_ids,
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
        使用self.roberta模型进行预测
        
        sequence_output = outputs[0]
        从输出中获取序列输出
        
        prediction_scores = self.lm_head(sequence_output)
        使用lm_head模型对序列输出进行预测
        
        loss = None if labels is None else self.hf_compute_loss(labels, prediction_scores)
        如果标签为None，则损失为None，否则使用hf_compute_loss函数计算损失
        
        if not return_dict:
            如果不返回字典，则输出元组
            output = (prediction_scores,) + outputs[2:]
            存储预测得分和其他输出
            return ((loss,) + output) if loss is not None else output
            如果损失不为None，则返回包含损失和输出的元组，否则只返回输出
        
        return TFMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        返回TFMaskedLMOutput对象，包含损失、logits、隐藏状态和注意力权重信息

    def build(self, input_shape=None):
        如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        如果self.roberta存在，则在self.roberta的命名空间下构建self.roberta
        if getattr(self, "roberta", None) is not None:
            with tf.name_scope(self.roberta.name):
                self.roberta.build(None)
        如果self.lm_head存在，则在self.lm_head的命名空间下构建self.lm_head
        if getattr(self, "lm_head", None) is not None:
            with tf.name_scope(self.lm_head.name):
                self.lm_head.build(None)
# 添加 XLM-RoBERTa 模型的文档字符串
@add_start_docstrings(
    "XLM-RoBERTa Model with a `language modeling` head on top for CLM fine-tuning.",
    XLM_ROBERTA_START_DOCSTRING,
)
# 从 transformers.models.roberta.modeling_tf_roberta.TFRobertaForCausalLM 复制, 将 Roberta 替换为 XLMRoberta, ROBERTA 替换为 XLM_ROBERTA
class TFXLMRobertaForCausalLM(TFXLMRobertaPreTrainedModel, TFCausalLanguageModelingLoss):
    # 忽略在加载 TF 模型时未期望存在的层，例如 "pooler" 和 "lm_head.decoder.weight"
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head.decoder.weight"]

    def __init__(self, config: XLMRobertaConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 如果配置不是解码器, 则输出警告
        if not config.is_decoder:
            logger.warning("If you want to use `TFXLMRobertaLMHeadModel` as a standalone, add `is_decoder=True.`")

        # 创建 TFXLMRobertaMainLayer 和 TFXLMRobertaLMHead 对象
        self.roberta = TFXLMRobertaMainLayer(config, add_pooling_layer=False, name="roberta")
        self.lm_head = TFXLMRobertaLMHead(config, input_embeddings=self.roberta.embeddings, name="lm_head")

    # 获取 lm_head 对象
    def get_lm_head(self):
        return self.lm_head

    # 获取前缀偏置名称, 这个方法已经被弃用, 请使用 get_bias 代替
    def get_prefix_bias_name(self):
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        return self.name + "/" + self.lm_head.name

    # 复制自 transformers.models.bert.modeling_tf_bert.TFBertLMHeadModel.prepare_inputs_for_generation
    # 准备输入数据以进行生成
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        # 如果模型作为编码器-解码器模型中的解码器使用, 则动态创建解码器注意力掩码
        if attention_mask is None:
            attention_mask = tf.ones(input_shape)

        # 如果使用了过去的 key/value, 则只使用最后一个输入 ID
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}
    # 定义一个 call 方法，接受各种类型的输入参数
    def call(
        self,
        # 输入序列的 ID
        input_ids: TFModelInputType | None = None,
        # 输入序列的注意力掩码
        attention_mask: np.ndarray | tf.Tensor | None = None,
        # 输入序列的类型 ID
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        # 输入序列的位置 ID
        position_ids: np.ndarray | tf.Tensor | None = None,
        # 用于修改注意力的掩码
        head_mask: np.ndarray | tf.Tensor | None = None,
        # 输入的词嵌入表示
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        # 编码器的隐藏状态
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,
        # 编码器的注意力掩码
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        # 用于快速推理的前一个时间步的输出
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        # 是否使用缓存
        use_cache: Optional[bool] = None,
        # 是否输出注意力权重
        output_attentions: Optional[bool] = None,
        # 是否输出所有隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典格式的输出
        return_dict: Optional[bool] = None,
        # 标签输入
        labels: np.ndarray | tf.Tensor | None = None,
        # 是否为训练模式
        training: Optional[bool] = False,
    ):
        # 如果模型还未构建，则进行构建
        def build(self, input_shape=None):
            if self.built:
                return
            self.built = True
            # 如果有 roberta 子层，则构建它
            if getattr(self, "roberta", None) is not None:
                with tf.name_scope(self.roberta.name):
                    self.roberta.build(None)
            # 如果有 lm_head 子层，则构建它
            if getattr(self, "lm_head", None) is not None:
                with tf.name_scope(self.lm_head.name):
                    self.lm_head.build(None)
# 从transformers.models.roberta.modeling_tf_roberta.TFRobertaClassificationHead复制并修改为XLMRoberta
class TFXLMRobertaClassificationHead(tf.keras.layers.Layer):
    """用于句子级分类任务的头部。"""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 创建一个全连接层，用于分类任务
        self.dense = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        # 定义分类器的dropout，如果未定义则使用隐藏层的dropout概率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = tf.keras.layers.Dropout(classifier_dropout)
        # 最后的线性映射层
        self.out_proj = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="out_proj"
        )
        # 保存配置信息
        self.config = config

    def call(self, features, training=False):
        # 取第一个位置的特征（相当于[CLS]符号）
        x = features[:, 0, :]
        x = self.dropout(x, training=training)
        x = self.dense(x)
        x = self.dropout(x, training=training)
        x = self.out_proj(x)
        return x

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建全连接层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 构建线性映射层
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.config.hidden_size])


@add_start_docstrings(
    """
    XLM RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
# 从transformers.models.roberta.modeling_tf_roberta.TFRobertaForSequenceClassification复制并修改为XLMRoberta，ROBERTA->XLM_ROBERTA
class TFXLMRobertaForSequenceClassification(TFXLMRobertaPreTrainedModel, TFSequenceClassificationLoss):
    # 当从PT模型加载TF模型时，包含'.'的名称表示授权的意外/缺失层
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        # 创建XLM-RoBERTa主体层
        self.roberta = TFXLMRobertaMainLayer(config, add_pooling_layer=False, name="roberta")
        # 创建分类头部
        self.classifier = TFXLMRobertaClassificationHead(config, name="classifier")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 为 TFSequenceClassifierOutput 添加代码示例文档
    @add_code_sample_docstrings(
        # 使用的 checkpoint 模型为 "cardiffnlp/twitter-roberta-base-emotion"
        checkpoint="cardiffnlp/twitter-roberta-base-emotion",
        # 输出类型为 TFSequenceClassifierOutput
        output_type=TFSequenceClassifierOutput,
        # 使用的配置类为 _CONFIG_FOR_DOC
        config_class=_CONFIG_FOR_DOC,
        # 期望的输出为 'optimism'
        expected_output="'optimism'",
        # 期望的损失为 0.08
        expected_loss=0.08,
    )
    def call(
        self,
        # 输入 ID
        input_ids: TFModelInputType | None = None,
        # 注意力掩码
        attention_mask: np.ndarray | tf.Tensor | None = None,
        # Token 类型 ID
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        # 位置 ID
        position_ids: np.ndarray | tf.Tensor | None = None,
        # 头部掩码
        head_mask: np.ndarray | tf.Tensor | None = None,
        # 输入嵌入
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        # 是否输出注意力权重
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典格式的输出
        return_dict: Optional[bool] = None,
        # 标签
        labels: np.ndarray | tf.Tensor | None = None,
        # 是否为训练模式
        training: Optional[bool] = False,
    ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        """
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 调用 roberta 模型，获取输出
        outputs = self.roberta(
            input_ids,
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
        # 获取序列输出
        sequence_output = outputs[0]
        # 通过分类器得到logits
        logits = self.classifier(sequence_output, training=training)
    
        # 如果存在标签，计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)
    
        # 如果不返回字典格式，返回一个元组
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
    
        # 返回 TFSequenceClassifierOutput
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
        # 构建 roberta 模型
        if getattr(self, "roberta", None) is not None:
            with tf.name_scope(self.roberta.name):
                self.roberta.build(None)
        # 构建分类器
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build(None)
# 添加模型文档字符串，说明这是一个在XLM Roberta模型基础上添加了多选分类头的模型，用于诸如RocStories/SWAG任务的分类，多选分类头由池化输出上方的线性层和softmax组成
@add_start_docstrings(
    """
    XLM Roberta Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
    a softmax) e.g. for RocStories/SWAG tasks.
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
# 从transformers.models.roberta.modeling_tf_roberta.TFRobertaForMultipleChoice复制并修改，将Roberta->XLMRoberta, ROBERTA->XLM_ROBERTA
class TFXLMRobertaForMultipleChoice(TFXLMRobertaPreTrainedModel, TFMultipleChoiceLoss):
    # 在加载TF模型时，'.'表示的是授权的意外/缺失层
    _keys_to_ignore_on_load_unexpected = [r"lm_head"]
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 初始化XLM Roberta主层
        self.roberta = TFXLMRobertaMainLayer(config, name="roberta")
        # 初始化Dropout层，以防止过拟合
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        # 初始化分类器，是一个全连接层，用于多选分类任务，输出维度为1
        self.classifier = tf.keras.layers.Dense(
            1, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        # 保存模型配置
        self.config = config

    # 对模型的前向传播进行装饰，添加模型文档字符串，描述输入
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    # 添加代码示例文档字符串
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

        # 如果输入的 input_ids 不为空，则获取 num_choices 和 seq_length
        if input_ids is not None:
            num_choices = shape_list(input_ids)[1]  # 获取 input_ids 的第二个维度的大小
            seq_length = shape_list(input_ids)[2]   # 获取 input_ids 的第三个维度的大小
        else:
            num_choices = shape_list(inputs_embeds)[1]  # 获取 inputs_embeds 的第二个维度的大小
            seq_length = shape_list(inputs_embeds)[2]   # 获取 inputs_embeds 的第三个维度的大小

        # 将 input_ids、attention_mask、token_type_ids 和 position_ids 展平，以便传入模型
        flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
        flat_attention_mask = tf.reshape(attention_mask, (-1, seq_length)) if attention_mask is not None else None
        flat_token_type_ids = tf.reshape(token_type_ids, (-1, seq_length)) if token_type_ids is not None else None
        flat_position_ids = tf.reshape(position_ids, (-1, seq_length)) if position_ids is not None else None

        # 将输入传递给 Roberta 模型进行处理
        outputs = self.roberta(
            flat_input_ids,
            flat_attention_mask,
            flat_token_type_ids,
            flat_position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 提取池化输出，并应用 dropout
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, training=training)
        # 将池化输出传递给分类器以获取 logits
        logits = self.classifier(pooled_output)
        # 将 logits 重新形状为(batch_size * num_choices, ...)的形式
        reshaped_logits = tf.reshape(logits, (-1, num_choices))

        # 如果有标签，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, reshaped_logits)

        # 如果 return_dict 为 False，则返回一个元组，否则返回 TFMultipleChoiceModelOutput
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 构建模型
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 构建 Roberta 模型和分类器
        if getattr(self, "roberta", None) is not None:
            with tf.name_scope(self.roberta.name):
                self.roberta.build(None)
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
```  
# 使用add_start_docstrings装饰器添加模型说明文档
# 定义了XLM RoBERTa模型，并在顶部添加了一个标记分类头（隐藏状态输出的线性层），用于命名实体识别（NER）任务
@add_start_docstrings(
    """
    XLM RoBERTa Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
# 从transformers.models.roberta.modeling_tf_roberta.TFRobertaForTokenClassification中复制得到，将Roberta->XLMRoberta，ROBERTA->XLM_ROBERTA
class TFXLMRobertaForTokenClassification(TFXLMRobertaPreTrainedModel, TFTokenClassificationLoss):
    # 使用'.'表示在从PT模型加载TF模型时，预期的未授权的意外/缺失的层
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head"]
    # 预期的加载缺失的层
    _keys_to_ignore_on_load_missing = [r"dropout"]

    # 初始化函数
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化函数
        super().__init__(config, *inputs, **kwargs)
        # 设置标签数
        self.num_labels = config.num_labels

        # 创建XLMRoBerta主层对象，不添加池化层
        self.roberta = TFXLMRobertaMainLayer(config, add_pooling_layer=False, name="roberta")
        # 计算分类器的丢弃率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 添加dropout层
        self.dropout = tf.keras.layers.Dropout(classifier_dropout)
        # 添加分类器线性层
        self.classifier = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        # 保存配置信息
        self.config = config

    # 使用unpack_inputs装饰器，将输入展开为单独的参数
    # 添加模型前向传播的说明文档
    # 添加代码示例的说明文档
    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="ydshieh/roberta-large-ner-english",
        output_type=TFTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="['O', 'ORG', 'ORG', 'O', 'O', 'O', 'O', 'O', 'LOC', 'O', 'LOC', 'LOC']",
        expected_loss=0.01,
    )
    # 定义模型的前向传播函数
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
# 这里应该还有其他的代码，但由于字数限制，无法完全列出
        ) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 对输入进行 RoBERTa 模型的处理
        outputs = self.roberta(
            input_ids,
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
        # 获取 RoBERTa 模型的输出中的 sequence_output
        sequence_output = outputs[0]

        # 对 sequence_output 进行 dropout 处理
        sequence_output = self.dropout(sequence_output, training=training)
        # 将处理后的 sequence_output 传入分类器得到 logits
        logits = self.classifier(sequence_output)

        # 如果 labels 存在，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果 return_dict 为 False，则返回 (loss, logits, hidden_states, attentions)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则返回 TFTokenClassifierOutput 类的对象
        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 定义 build 方法
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果 RoBERTa 模型存在，则构建 RoBERTa 模型
        if getattr(self, "roberta", None) is not None:
            with tf.name_scope(self.roberta.name):
                self.roberta.build(None)
        # 如果分类器存在，则构建分类器
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
# 导入所需模块或函数
@add_start_docstrings(
    """
    XLM RoBERTa Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
    linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
# 定义了一个用于抽取式问答任务（如 SQuAD）的 XLM RoBERTa 模型，其在隐藏状态输出的顶部带有一个用于跨度分类的头部
# 包含线性层，用于计算“跨度起始对数”和“跨度结束对数”
class TFXLMRobertaForQuestionAnswering(TFXLMRobertaPreTrainedModel, TFQuestionAnsweringLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    # 定义了一个用于加载预训练 PyTorch 模型时忽略的层的名称列表
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head"]

    # 初始化函数，接受配置参数和其他输入，设置模型的基本结构
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化函数
        super().__init__(config, *inputs, **kwargs)
        # 将标签数量设置为配置中的标签数量
        self.num_labels = config.num_labels

        # 创建 XLM RoBERTa 主层对象，不添加池化层
        self.roberta = TFXLMRobertaMainLayer(config, add_pooling_layer=False, name="roberta")
        # 创建用于输出的全连接层，参数初始化方式为配置中指定的初始化范围
        self.qa_outputs = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )
        # 保存配置对象
        self.config = config

    # 对输入进行解包，并添加模型前向传播的注释
    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="ydshieh/roberta-base-squad2",
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="' puppet'",
        expected_loss=0.86,
    )
    # 模型的前向传播方法，接受多种输入，并返回模型输出或损失
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
        # 使用 Type Hinting 标注函数的返回类型为 TFQuestionAnsweringModelOutput 或者 Tuple[tf.Tensor]
        outputs = self.roberta(
            input_ids,
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
        # 获取 RoBERTa 模型的输出中的序列输出
        sequence_output = outputs[0]

        # 使用 QA 输出层得到预测的开始和结束位置的对数概率
        logits = self.qa_outputs(sequence_output)
        # 拆分对数概率张量，分别表示开始和结束位置的对数概率
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        # 去除不必要的维度，得到一维张量
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        loss = None
        # 计算损失函数
        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            loss = self.hf_compute_loss(labels, (start_logits, end_logits))

        # 如果不需要返回字典，则输出包括损失在内的结果元组
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典，则输出 TFQuestionAnsweringModelOutput 类的对象
        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建 RoBERTa 模型
        if getattr(self, "roberta", None) is not None:
            with tf.name_scope(self.roberta.name):
                self.roberta.build(None)
        # 构建 QA 输出层
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])
```