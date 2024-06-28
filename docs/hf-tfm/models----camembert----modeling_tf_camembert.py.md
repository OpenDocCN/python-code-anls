# `.\models\camembert\modeling_tf_camembert.py`

```py
# 导入必要的库和模块
import math
import warnings
from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf

# 导入transformers库中的相关模块和类
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFBaseModelOutputWithPoolingAndCrossAttentions,
    TFCausalLMOutputWithCrossAttentions,
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
from .configuration_camembert import CamembertConfig

# 获取全局的日志记录器
logger = logging.get_logger(__name__)

# Transformer模型的checkpoint路径
_CHECKPOINT_FOR_DOC = "almanach/camembert-base"
# Camembert配置文件的名称
_CONFIG_FOR_DOC = "CamembertConfig"

# Camembert预训练模型的存档列表
TF_CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    # 请查看所有Camembert模型的列表：https://huggingface.co/models?filter=camembert
]

# Camembert模型的起始文档字符串，包含模型的基本信息和用法说明
CAMEMBERT_START_DOCSTRING = r"""

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
    """
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
    """

    """
    Parameters:
        config ([`CamembertConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """
"""

CAMEMBERT_INPUTS_DOCSTRING = r"""
"""


# 从 transformers.models.roberta.modeling_tf_roberta.TFRobertaEmbeddings 复制过来的
class TFCamembertEmbeddings(keras.layers.Layer):
    """
    与 BertEmbeddings 相同，但在位置嵌入索引方面有微小调整。
    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.padding_idx = 1  # 定义填充索引，用于表示填充的位置
        self.config = config  # 保存配置对象
        self.hidden_size = config.hidden_size  # 获取隐藏层大小
        self.max_position_embeddings = config.max_position_embeddings  # 获取最大位置嵌入数
        self.initializer_range = config.initializer_range  # 获取初始化范围
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")  # 创建层归一化层对象
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)  # 创建 dropout 层对象

    def build(self, input_shape=None):
        with tf.name_scope("word_embeddings"):
            # 创建词嵌入权重矩阵
            self.weight = self.add_weight(
                name="weight",
                shape=[self.config.vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        with tf.name_scope("token_type_embeddings"):
            # 创建 token 类型嵌入权重矩阵
            self.token_type_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.config.type_vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        with tf.name_scope("position_embeddings"):
            # 创建位置嵌入权重矩阵
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        if self.built:
            return
        self.built = True
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                # 构建层归一化层
                self.LayerNorm.build([None, None, self.config.hidden_size])

    def create_position_ids_from_input_ids(self, input_ids, past_key_values_length=0):
        """
        用输入的 id 创建位置 id，非填充符号被替换为它们的位置数字。位置数字从 padding_idx+1 开始。
        填充符号被忽略。这是从 fairseq 的 `utils.make_positions` 修改而来。

        Args:
            input_ids: tf.Tensor 输入的 id 张量
        Returns: tf.Tensor 输出的位置 id 张量
        """
        mask = tf.cast(tf.math.not_equal(input_ids, self.padding_idx), dtype=input_ids.dtype)
        incremental_indices = (tf.math.cumsum(mask, axis=1) + past_key_values_length) * mask

        return incremental_indices + self.padding_idx

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
        assert not (input_ids is None and inputs_embeds is None)

        # 如果没有提供 input_ids 或 inputs_embeds，抛出异常
        if input_ids is not None:
            # 检查 input_ids 是否在词汇表大小内
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            # 根据 input_ids 从权重矩阵中获取对应的 embeddings
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        # 获取输入 embeds 的形状，去掉最后一个维度
        input_shape = shape_list(inputs_embeds)[:-1]

        # 如果没有提供 token_type_ids，则用0填充形状
        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)

        # 如果没有提供 position_ids
        if position_ids is None:
            if input_ids is not None:
                # 根据输入的 token ids 创建 position ids。任何填充的 token 仍然保持填充状态。
                position_ids = self.create_position_ids_from_input_ids(
                    input_ids=input_ids, past_key_values_length=past_key_values_length
                )
            else:
                # 创建默认的 position ids，范围从 padding_idx + 1 到 input_shape[-1] + padding_idx + 1
                position_ids = tf.expand_dims(
                    tf.range(start=self.padding_idx + 1, limit=input_shape[-1] + self.padding_idx + 1), axis=0
                )

        # 根据 position_ids 获取 position embeddings
        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
        # 根据 token_type_ids 获取 token type embeddings
        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)
        # 计算最终的 embeddings，组合 inputs_embeds、position_embeds 和 token_type_embeds
        final_embeddings = inputs_embeds + position_embeds + token_type_embeds
        # 对最终的 embeddings 进行 LayerNorm 处理
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        # 对最终的 embeddings 进行 dropout 处理
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)

        # 返回最终的 embeddings 结果
        return final_embeddings
# Copied from transformers.models.bert.modeling_tf_bert.TFBertPooler with Bert->Camembert
class TFCamembertPooler(keras.layers.Layer):
    def __init__(self, config: CamembertConfig, **kwargs):
        super().__init__(**kwargs)

        # 定义一个全连接层，用于池化隐藏状态的第一个令牌
        self.dense = keras.layers.Dense(
            units=config.hidden_size,  # 全连接层的输出大小为配置文件中定义的隐藏大小
            kernel_initializer=get_initializer(config.initializer_range),  # 使用配置中的初始化器范围进行权重初始化
            activation="tanh",  # 激活函数为双曲正切函数
            name="dense",  # 层的名称为dense
        )
        self.config = config  # 保存配置参数

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 池化模型的方法是简单地选择与第一个令牌对应的隐藏状态
        first_token_tensor = hidden_states[:, 0]  # 获取每个样本的第一个令牌的隐藏状态
        pooled_output = self.dense(inputs=first_token_tensor)  # 使用全连接层池化第一个令牌的隐藏状态

        return pooled_output  # 返回池化输出

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])



# Copied from transformers.models.bert.modeling_tf_bert.TFBertSelfAttention with Bert->Camembert
class TFCamembertSelfAttention(keras.layers.Layer):
    def __init__(self, config: CamembertConfig, **kwargs):
        super().__init__(**kwargs)

        # 检查隐藏大小是否能够被注意力头数整除
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        # 初始化注意力头数和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        # 定义查询、键、值的全连接层，并使用配置中的初始化器范围初始化权重
        self.query = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        self.key = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        self.value = keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )
        self.dropout = keras.layers.Dropout(rate=config.attention_probs_dropout_prob)  # 定义注意力概率的dropout层

        self.is_decoder = config.is_decoder  # 记录是否为解码器
        self.config = config  # 保存配置参数
    # 将输入张量重新调整形状从 [batch_size, seq_length, all_head_size] 到 [batch_size, seq_length, num_attention_heads, attention_head_size]
    tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

    # 将张量转置从 [batch_size, seq_length, num_attention_heads, attention_head_size] 到 [batch_size, num_attention_heads, seq_length, attention_head_size]
    return tf.transpose(tensor, perm=[0, 2, 1, 3])

# 神经网络模型的调用方法，接受多个输入张量和参数，执行注意力机制相关的计算
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
):
    # 在构建模型时调用，用于设置层的结构
    def build(self, input_shape=None):
        # 如果已经构建过一次，直接返回
        if self.built:
            return
        # 标记该层已构建
        self.built = True
        # 如果存在查询张量，构建查询张量的结构
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.hidden_size])
        # 如果存在键张量，构建键张量的结构
        if getattr(self, "key", None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.config.hidden_size])
        # 如果存在值张量，构建值张量的结构
        if getattr(self, "value", None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.config.hidden_size])
# Copied from transformers.models.bert.modeling_tf_bert.TFBertSelfOutput with Bert->Camembert
class TFCamembertSelfOutput(keras.layers.Layer):
    def __init__(self, config: CamembertConfig, **kwargs):
        super().__init__(**kwargs)

        # 初始化一个全连接层，用于转换隐藏状态的维度
        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 初始化 LayerNormalization 层，用于归一化隐藏状态
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 初始化 Dropout 层，用于在训练时随机置零输入张量的一部分
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 将隐藏状态通过全连接层 dense 进行线性转换
        hidden_states = self.dense(inputs=hidden_states)
        # 在训练时应用 Dropout，随机置零一部分输入张量
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 对转换后的隐藏状态应用 LayerNormalization，加上输入张量 input_tensor
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建 dense 层，设置其输入维度为 config.hidden_size
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 构建 LayerNorm 层，设置其输入维度为 config.hidden_size
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# Copied from transformers.models.bert.modeling_tf_bert.TFBertAttention with Bert->Camembert
class TFCamembertAttention(keras.layers.Layer):
    def __init__(self, config: CamembertConfig, **kwargs):
        super().__init__(**kwargs)

        # 初始化自注意力层 TFCamembertSelfAttention
        self.self_attention = TFCamembertSelfAttention(config, name="self")
        # 初始化输出层 TFCamembertSelfOutput
        self.dense_output = TFCamembertSelfOutput(config, name="output")

    def prune_heads(self, heads):
        raise NotImplementedError

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
        # 调用自注意力层进行注意力计算，返回自注意力层的输出
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
        # 将自注意力层的输出作为输入，通过输出层进行转换
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=input_tensor, training=training
        )
        # 如果需要输出注意力值，则将其添加到输出元组中
        outputs = (attention_output,) + self_outputs[1:]

        return outputs
    # 定义神经网络层的构建方法，用于在给定输入形状时构建层
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回，避免重复构建
        if self.built:
            return
        # 设置标志位，表示该层已经构建完成
        self.built = True
        
        # 检查是否存在自注意力层，并构建其名称作用域下的层
        if getattr(self, "self_attention", None) is not None:
            with tf.name_scope(self.self_attention.name):
                self.self_attention.build(None)
        
        # 检查是否存在密集输出层，并构建其名称作用域下的层
        if getattr(self, "dense_output", None) is not None:
            with tf.name_scope(self.dense_output.name):
                self.dense_output.build(None)
# Copied from transformers.models.bert.modeling_tf_bert.TFBertIntermediate with Bert->Camembert
class TFCamembertIntermediate(keras.layers.Layer):
    def __init__(self, config: CamembertConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，用于中间状态转换，输出单元数由配置文件决定
        self.dense = keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # 根据配置文件中指定的激活函数类型，获取对应的 TensorFlow 激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 将输入的隐藏状态通过全连接层处理
        hidden_states = self.dense(inputs=hidden_states)
        # 使用配置中指定的中间激活函数处理转换后的隐藏状态
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建层次结构，若已存在 dense 层则使用其名字的命名空间，构建时指定输入形状
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# Copied from transformers.models.bert.modeling_tf_bert.TFBertOutput with Bert->Camembert
class TFCamembertOutput(keras.layers.Layer):
    def __init__(self, config: CamembertConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，用于输出层，输出单元数由配置文件决定
        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个 LayerNormalization 层，用于规范化层次，epsilon 值由配置文件决定
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建一个 Dropout 层，用于在训练时进行随机失活，失活率由配置文件决定
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 将输入的隐藏状态通过全连接层处理
        hidden_states = self.dense(inputs=hidden_states)
        # 若在训练状态下，对输出的隐藏状态进行随机失活处理
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 将失活后的隐藏状态与输入张量进行加和，并通过 LayerNormalization 处理
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建层次结构，若已存在 dense 层则使用其名字的命名空间，构建时指定输入形状
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])
        # 构建层次结构，若已存在 LayerNorm 层则使用其名字的命名空间，构建时指定输入形状
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# Copied from transformers.models.bert.modeling_tf_bert.TFBertLayer with Bert->Camembert
class TFCamembertLayer(keras.layers.Layer):
    # 使用指定的配置初始化 Camembert 模型
    def __init__(self, config: CamembertConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 创建注意力层对象，使用给定的配置，并命名为"attention"
        self.attention = TFCamembertAttention(config, name="attention")
        
        # 设置是否为解码器的标志
        self.is_decoder = config.is_decoder
        
        # 设置是否添加交叉注意力的标志
        self.add_cross_attention = config.add_cross_attention
        
        # 如果要添加交叉注意力，需检查当前模型是否为解码器模型
        if self.add_cross_attention:
            if not self.is_decoder:
                # 如果不是解码器模型且添加了交叉注意力，则引发错误
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            
            # 创建交叉注意力层对象，使用给定的配置，并命名为"crossattention"
            self.crossattention = TFCamembertAttention(config, name="crossattention")
        
        # 创建中间层对象，使用给定的配置，并命名为"intermediate"
        self.intermediate = TFCamembertIntermediate(config, name="intermediate")
        
        # 创建输出层对象，使用给定的配置，并命名为"output"
        self.bert_output = TFCamembertOutput(config, name="output")

    # 定义模型的调用方法
    def call(
        self,
        hidden_states: tf.Tensor,                    # 输入的隐藏状态张量
        attention_mask: tf.Tensor,                   # 注意力掩码张量
        head_mask: tf.Tensor,                        # 头部掩码张量
        encoder_hidden_states: tf.Tensor | None,     # 编码器的隐藏状态张量或空值
        encoder_attention_mask: tf.Tensor | None,    # 编码器的注意力掩码张量或空值
        past_key_value: Tuple[tf.Tensor] | None,     # 过去的键-值张量元组或空值
        output_attentions: bool,                     # 是否输出注意力权重
        training: bool = False,                      # 是否处于训练模式，默认为False
    ````
    # 定义方法签名，指定返回类型为包含单个元素的元组，该元素类型为 tf.Tensor
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        head_mask: Optional[tf.Tensor] = None,
        encoder_hidden_states: Optional[tf.Tensor] = None,
        encoder_attention_mask: Optional[tf.Tensor] = None,
        past_key_value: Optional[Tuple[tf.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        training: Optional[bool] = False,
    ) -> Tuple[tf.Tensor]:
    
        # 如果 past_key_value 不为 None，则提取出 self-attention 的过去键/值缓存
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
    
        # 调用 self.attention 方法进行自注意力计算
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
    
        # 获取 self-attention 的输出
        attention_output = self_attention_outputs[0]
    
        # 如果模型是解码器模型
        if self.is_decoder:
            # 输出中除了 self_attention_outputs 中的第一个元素之外的所有元素
            outputs = self_attention_outputs[1:-1]
            # 提取 self_attention_outputs 中的最后一个元素作为 present_key_value
            present_key_value = self_attention_outputs[-1]
        else:
            # 输出中包含 self_attention_outputs 中除第一个元素外的所有元素（如果输出注意力权重的话）
            outputs = self_attention_outputs[1:]
    
        # 初始化 cross_attn_present_key_value 为 None
        cross_attn_present_key_value = None
    
        # 如果模型是解码器并且存在编码器的隐藏状态
        if self.is_decoder and encoder_hidden_states is not None:
            # 如果模型没有交叉注意力层，则引发 ValueError 异常
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )
    
            # 如果 past_key_value 不为 None，则提取出交叉注意力的过去键/值缓存
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
    
            # 调用 self.crossattention 方法进行交叉注意力计算
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
            # 将交叉注意力的输出中除了第一个和最后一个元素之外的所有元素添加到 outputs 中
            outputs = outputs + cross_attention_outputs[1:-1]
    
            # 将交叉注意力的输出中的最后一个元素添加到 present_key_value 中
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value
    
        # 对注意力输出进行中间层处理
        intermediate_output = self.intermediate(hidden_states=attention_output)
    
        # 对中间层输出进行最终的 Bert 输出层处理
        layer_output = self.bert_output(
            hidden_states=intermediate_output, input_tensor=attention_output, training=training
        )
    
        # 将最终的层输出添加到 outputs 中
        outputs = (layer_output,) + outputs
    
        # 如果模型是解码器，将注意力键/值作为最后的输出添加到 outputs 中
        if self.is_decoder:
            outputs = outputs + (present_key_value,)
    
        # 返回最终的输出元组
        return outputs
    # 构建方法，用于构建模型的层次结构。如果已经构建过，则直接返回。
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回，不再重复构建
        if self.built:
            return
        # 将标志位设置为已构建
        self.built = True
        
        # 如果存在 self.attention 属性，则构建 self.attention 层次结构
        if getattr(self, "attention", None) is not None:
            # 使用 tf.name_scope 为 self.attention 层创建命名空间
            with tf.name_scope(self.attention.name):
                # 调用 self.attention 的 build 方法来构建该层
                self.attention.build(None)
        
        # 如果存在 self.intermediate 属性，则构建 self.intermediate 层次结构
        if getattr(self, "intermediate", None) is not None:
            # 使用 tf.name_scope 为 self.intermediate 层创建命名空间
            with tf.name_scope(self.intermediate.name):
                # 调用 self.intermediate 的 build 方法来构建该层
                self.intermediate.build(None)
        
        # 如果存在 self.bert_output 属性，则构建 self.bert_output 层次结构
        if getattr(self, "bert_output", None) is not None:
            # 使用 tf.name_scope 为 self.bert_output 层创建命名空间
            with tf.name_scope(self.bert_output.name):
                # 调用 self.bert_output 的 build 方法来构建该层
                self.bert_output.build(None)
        
        # 如果存在 self.crossattention 属性，则构建 self.crossattention 层次结构
        if getattr(self, "crossattention", None) is not None:
            # 使用 tf.name_scope 为 self.crossattention 层创建命名空间
            with tf.name_scope(self.crossattention.name):
                # 调用 self.crossattention 的 build 方法来构建该层
                self.crossattention.build(None)
# 从 transformers.models.bert.modeling_tf_bert.TFBertEncoder 复制代码，将其中的 Bert 替换为 Camembert
class TFCamembertEncoder(keras.layers.Layer):
    # 初始化函数，接收 CamembertConfig 对象作为参数
    def __init__(self, config: CamembertConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        # 创建 CamembertLayer 的列表，根据层数进行命名
        self.layer = [TFCamembertLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

    # 前向传播函数，接收多个参数和返回类型的注解
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
        # 初始化空元组或 None，用于存储中间结果
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 若 use_cache 为 True，则初始化空元组用于存储下一层的缓存
        next_decoder_cache = () if use_cache else None

        # 遍历每一层的 CamembertLayer
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的过去键值对，如果 past_key_values 不为 None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 调用当前层的前向传播函数，计算当前层的输出
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
            # 更新 hidden_states 为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]

            # 如果 use_cache 为 True，则更新下一层的缓存
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            # 如果 output_attentions 为 True，则将当前层的注意力加入 all_attentions
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                # 如果配置中包含交叉注意力，并且 encoder_hidden_states 不为 None，则将交叉注意力加入 all_cross_attentions
                if self.config.add_cross_attention and encoder_hidden_states is not None:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 添加最后一层的隐藏状态到 all_hidden_states
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果 return_dict 为 False，则返回非空的元组
        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_attentions, all_cross_attentions] if v is not None
            )

        # 返回 TFBaseModelOutputWithPastAndCrossAttentions 对象，包含各类输出结果
        return TFBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )
    # 定义一个方法 `build`，用于构建神经网络模型的层次结构
    def build(self, input_shape=None):
        # 如果模型已经构建过，直接返回，避免重复构建
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        # 检查是否存在 `layer` 属性，并逐层构建每个子层
        if getattr(self, "layer", None) is not None:
            # 遍历每个子层
            for layer in self.layer:
                # 在 TensorFlow 中为每个层次设置命名空间，以层次的名字作为命名空间
                with tf.name_scope(layer.name):
                    # 构建每个子层，此处传入 `None` 作为输入形状参数
                    layer.build(None)
@keras_serializable
# 从 transformers.models.roberta.modeling_tf_roberta.TFRobertaMainLayer 复制并修改为 Camembert
class TFCamembertMainLayer(keras.layers.Layer):
    config_class = CamembertConfig

    def __init__(self, config, add_pooling_layer=True, **kwargs):
        super().__init__(**kwargs)

        self.config = config  # 设置配置对象
        self.is_decoder = config.is_decoder  # 是否为解码器

        self.num_hidden_layers = config.num_hidden_layers  # 隐藏层的数量
        self.initializer_range = config.initializer_range  # 初始化范围
        self.output_attentions = config.output_attentions  # 是否输出注意力权重
        self.output_hidden_states = config.output_hidden_states  # 是否输出隐藏状态
        self.return_dict = config.use_return_dict  # 是否返回字典格式的输出
        self.encoder = TFCamembertEncoder(config, name="encoder")  # Camembert 编码器
        self.pooler = TFCamembertPooler(config, name="pooler") if add_pooling_layer else None  # 可选的池化层
        # embeddings 必须是最后声明的，以保持权重的顺序
        self.embeddings = TFCamembertEmbeddings(config, name="embeddings")  # Camembert embeddings

    # 从 transformers.models.bert.modeling_tf_bert.TFBertMainLayer.get_input_embeddings 复制
    def get_input_embeddings(self) -> keras.layers.Layer:
        return self.embeddings  # 获取输入 embeddings

    # 从 transformers.models.bert.modeling_tf_bert.TFBertMainLayer.set_input_embeddings 复制
    def set_input_embeddings(self, value: tf.Variable):
        self.embeddings.weight = value  # 设置 embeddings 的权重
        self.embeddings.vocab_size = shape_list(value)[0]  # 设置 embeddings 的词汇表大小

    # 从 transformers.models.bert.modeling_tf_bert.TFBertMainLayer._prune_heads 复制
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError  # 未实现的方法，用于剪枝模型的注意力头部

    @unpack_inputs
    # 从 transformers.models.bert.modeling_tf_bert.TFBertMainLayer.call 复制
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
    # 如果模型已经构建完成，则直接返回，不做任何操作
    if self.built:
        return
    # 标记模型已经构建
    self.built = True
    
    # 如果模型中存在编码器（encoder），则构建编码器
    if getattr(self, "encoder", None) is not None:
        # 在TensorFlow的命名空间中，使用编码器的名称
        with tf.name_scope(self.encoder.name):
            # 构建编码器，input_shape设为None
            self.encoder.build(None)
    
    # 如果模型中存在池化器（pooler），则构建池化器
    if getattr(self, "pooler", None) is not None:
        # 在TensorFlow的命名空间中，使用池化器的名称
        with tf.name_scope(self.pooler.name):
            # 构建池化器，input_shape设为None
            self.pooler.build(None)
    
    # 如果模型中存在嵌入层（embeddings），则构建嵌入层
    if getattr(self, "embeddings", None) is not None:
        # 在TensorFlow的命名空间中，使用嵌入层的名称
        with tf.name_scope(self.embeddings.name):
            # 构建嵌入层，input_shape设为None
            self.embeddings.build(None)
# 定义一个名为 TFCamembertPreTrainedModel 的类，继承自 TFPreTrainedModel
class TFCamembertPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    一个抽象类，用于处理权重初始化以及下载和加载预训练模型的简单接口。
    """

    # 指定配置类为 CamembertConfig
    config_class = CamembertConfig
    # 基础模型前缀为 "roberta"
    base_model_prefix = "roberta"


# 引入函数装饰器 add_start_docstrings，并传入文档字符串和 CAMEMBERT_START_DOCSTRING 常量
@add_start_docstrings(
    "The bare CamemBERT Model transformer outputting raw hidden-states without any specific head on top.",
    CAMEMBERT_START_DOCSTRING,
)
# 从 transformers.models.roberta.modeling_tf_roberta.TFRobertaModel 复制代码，并将 Roberta->Camembert, ROBERTA->CAMEMBERT
class TFCamembertModel(TFCamembertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类构造函数，传入配置对象和其他参数
        super().__init__(config, *inputs, **kwargs)
        # 初始化 self.roberta 属性为 TFCamembertMainLayer 类的实例，传入配置对象和名称 "roberta"
        self.roberta = TFCamembertMainLayer(config, name="roberta")

    # 引入函数装饰器 unpack_inputs，用于展开输入参数
    # 引入函数装饰器 add_start_docstrings_to_model_forward，传入格式化字符串 CAMEMBERT_INPUTS_DOCSTRING 和输入参数说明
    # 引入函数装饰器 add_code_sample_docstrings，传入检查点、输出类型和配置类的相关文档信息
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
        # 函数参数说明完毕
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
            input_ids=input_ids,  # 输入的 token IDs
            attention_mask=attention_mask,  # 注意力遮罩，掩盖无效位置的 token
            token_type_ids=token_type_ids,  # token 类型 IDs，用于区分句子 A 和句子 B
            position_ids=position_ids,  # token 的位置编码
            head_mask=head_mask,  # 头部掩码，用于指定哪些注意力头部被屏蔽
            inputs_embeds=inputs_embeds,  # 输入的嵌入表示
            encoder_hidden_states=encoder_hidden_states,  # 编码器的隐藏状态序列
            encoder_attention_mask=encoder_attention_mask,  # 编码器的注意力遮罩
            past_key_values=past_key_values,  # 预计算的键值状态，用于加速解码
            use_cache=use_cache,  # 是否使用缓存以加速解码
            output_attentions=output_attentions,  # 是否输出注意力权重
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
            return_dict=return_dict,  # 返回结果类型，字典还是元组
            training=training,  # 是否处于训练模式
        )

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return  # 如果已经构建过，则直接返回

        self.built = True  # 标记模型已构建

        if getattr(self, "roberta", None) is not None:
            with tf.name_scope(self.roberta.name):  # 在 TensorFlow 中设置命名空间
                self.roberta.build(None)  # 构建 RoBERTa 模型
# 从transformers.models.roberta.modeling_tf_roberta.TFRobertaLMHead复制而来，将Roberta->Camembert
class TFCamembertLMHead(keras.layers.Layer):
    """Camembert模型的masked语言建模头部。"""

    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.hidden_size = config.hidden_size
        self.dense = keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        self.act = get_tf_activation("gelu")

        # 输出权重与输入嵌入相同，但每个token有一个仅输出的偏置项。
        self.decoder = input_embeddings

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

    def get_output_embeddings(self):
        return self.decoder

    def set_output_embeddings(self, value):
        self.decoder.weight = value
        self.decoder.vocab_size = shape_list(value)[0]

    def get_bias(self):
        return {"bias": self.bias}

    def set_bias(self, value):
        self.bias = value["bias"]
        self.config.vocab_size = shape_list(value["bias"])[0]

    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.layer_norm(hidden_states)

        # 投影回词汇表大小，带有偏置项
        seq_length = shape_list(tensor=hidden_states)[1]
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.hidden_size])
        hidden_states = tf.matmul(a=hidden_states, b=self.decoder.weight, transpose_b=True)
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)

        return hidden_states


@add_start_docstrings(
    """在顶部有一个`language modeling`头的CamemBERT模型。""",
    CAMEMBERT_START_DOCSTRING,
)
# 从transformers.models.roberta.modeling_tf_roberta.TFRobertaForMaskedLM复制而来，将Roberta->Camembert, ROBERTA->CAMEMBERT
class TFCamembertForMaskedLM(TFCamembertPreTrainedModel, TFMaskedLanguageModelingLoss):
    # 带有'.'的名称表示在从PT模型加载TF模型时授权的意外/缺失层
    # 初始化一个列表，包含在加载时要忽略的特定键
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head.decoder.weight"]

    # 初始化方法，接受配置对象和其他输入参数
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法，传入配置和其他输入参数
        super().__init__(config, *inputs, **kwargs)

        # 初始化一个 RoBERTa 主层对象，禁用添加池化层，命名为 "roberta"
        self.roberta = TFCamembertMainLayer(config, add_pooling_layer=False, name="roberta")
        # 初始化一个语言模型头部对象，传入配置和 RoBERTa 主层的嵌入层，命名为 "lm_head"
        self.lm_head = TFCamembertLMHead(config, self.roberta.embeddings, name="lm_head")

    # 返回语言模型头部对象的方法
    def get_lm_head(self):
        return self.lm_head

    # 返回前缀偏置名称的方法，已弃用，发出未来警告
    def get_prefix_bias_name(self):
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        # 返回头部名称组成的字符串，使用 "/" 分隔
        return self.name + "/" + self.lm_head.name

    # 调用方法的装饰器，将输入参数解包，并添加模型前向传递的文档字符串和代码示例的文档字符串
    @unpack_inputs
    @add_start_docstrings_to_model_forward(CAMEMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="<mask>",
        expected_output="' Paris'",
        expected_loss=0.1,
    )
    # 模型的前向传递方法，接受多个输入参数，并返回预测输出
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
        """
        # 调用 RoBERTa 模型进行前向传播，获取模型的输出结果
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

        # 从 RoBERTa 模型的输出中提取序列输出
        sequence_output = outputs[0]
        
        # 将序列输出送入语言模型头部，得到预测分数（logits）
        prediction_scores = self.lm_head(sequence_output)

        # 如果提供了标签，则计算损失；否则损失设为 None
        loss = None if labels is None else self.hf_compute_loss(labels, prediction_scores)

        # 如果不要求返回字典形式的输出，则按照元组形式返回结果
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TFMaskedLMOutput 类型的对象，包括损失、预测分数、隐藏状态和注意力权重
        return TFMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回
        if self.built:
            return
        
        # 标记模型已经构建
        self.built = True
        
        # 如果定义了 RoBERTa 模型，则构建 RoBERTa
        if getattr(self, "roberta", None) is not None:
            with tf.name_scope(self.roberta.name):
                self.roberta.build(None)
        
        # 如果定义了语言模型头部，则构建语言模型头部
        if getattr(self, "lm_head", None) is not None:
            with tf.name_scope(self.lm_head.name):
                self.lm_head.build(None)
# 从transformers.models.roberta.modeling_tf_roberta.TFRobertaClassificationHead复制而来，定义了一个用于句子级别分类任务的头部。
class TFCamembertClassificationHead(keras.layers.Layer):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 创建一个全连接层，输出维度为config.hidden_size，激活函数为tanh
        self.dense = keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        # 根据config中的设置，选择分类器的dropout率，如果未指定则使用hidden_dropout_prob
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 定义一个Dropout层，应用于全连接层的输出
        self.dropout = keras.layers.Dropout(classifier_dropout)
        # 创建一个全连接层，输出维度为config.num_labels，用于输出分类任务的结果
        self.out_proj = keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="out_proj"
        )
        self.config = config

    def call(self, features, training=False):
        # 取出features的第一个token的向量表示，通常代表<CLS> token
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        # 应用dropout层到x上，用于训练时进行随机失活
        x = self.dropout(x, training=training)
        # 将x传入全连接层dense中进行线性变换并激活
        x = self.dense(x)
        # 再次应用dropout层到x上，用于训练时进行随机失活
        x = self.dropout(x, training=training)
        # 将x传入全连接层out_proj中，生成最终的分类结果
        x = self.out_proj(x)
        return x

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经建立过网络，则直接返回，否则开始构建
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # 构建全连接层dense，输入维度为config.hidden_size
                self.dense.build([None, None, self.config.hidden_size])
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                # 构建全连接层out_proj，输入维度为config.hidden_size
                self.out_proj.build([None, None, self.config.hidden_size])


@add_start_docstrings(
    """
    CamemBERT Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    CAMEMBERT_START_DOCSTRING,
)
# 从transformers.models.roberta.modeling_tf_roberta.TFRobertaForSequenceClassification中复制，仅将Roberta替换为Camembert，ROBERTA替换为CAMEMBERT
class TFCamembertForSequenceClassification(TFCamembertPreTrainedModel, TFSequenceClassificationLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    # _keys_to_ignore_on_load_unexpected列出了在从PT模型加载TF模型时，可以忽略的意外/丢失的层的名称模式
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 设置分类任务的标签数量
        self.num_labels = config.num_labels

        # 创建Camembert主体层，用于处理输入序列，不包含池化层
        self.roberta = TFCamembertMainLayer(config, add_pooling_layer=False, name="roberta")
        # 创建Camembert分类头部，用于生成分类任务的输出
        self.classifier = TFCamembertClassificationHead(config, name="classifier")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CAMEMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 使用装饰器为方法添加文档字符串，指定模型和输出类型，以及配置类和预期输出和损失
    @add_code_sample_docstrings(
        checkpoint="cardiffnlp/twitter-roberta-base-emotion",
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'optimism'",
        expected_loss=0.08,
    )
    # 定义模型的调用方法，接受多个输入参数和可选的标签，返回分类器输出或者元组包含 logits
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 输入文本的 token IDs
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 表示输入文本中实际词汇的掩码
        token_type_ids: np.ndarray | tf.Tensor | None = None,  # 区分不同句子的标识符
        position_ids: np.ndarray | tf.Tensor | None = None,  # 表示输入中 token 的位置
        head_mask: np.ndarray | tf.Tensor | None = None,  # 多头注意力机制的掩码
        inputs_embeds: np.ndarray | tf.Tensor | None = None,  # 输入 token 的嵌入表示
        output_attentions: Optional[bool] = None,  # 是否返回注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否返回隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回 TFSequenceClassifierOutput 对象
        labels: np.ndarray | tf.Tensor | None = None,  # 计算序列分类/回归损失的标签
        training: Optional[bool] = False,  # 是否处于训练模式
    ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        """
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 使用 RoBERTa 模型处理输入数据，返回模型输出
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
        # 使用分类器模型处理序列输出，得到 logits
        logits = self.classifier(sequence_output, training=training)

        # 如果标签为空，则损失也为空；否则计算标签和 logits 之间的损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果不返回字典，则按顺序返回 logits 和其他输出（如隐藏状态）
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TFSequenceClassifierOutput 对象，包括损失、logits、隐藏状态和注意力权重
        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 构建模型，初始化 RoBERTa 和分类器层
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果 RoBERTa 模型存在，则构建 RoBERTa 层
        if getattr(self, "roberta", None) is not None:
            with tf.name_scope(self.roberta.name):
                self.roberta.build(None)
        # 如果分类器存在，则构建分类器层
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build(None)
# 使用装饰器将以下字符串添加到模型文档字符串的开头，描述了 CamemBERT 模型及其在命名实体识别 (NER) 任务中的用途
@add_start_docstrings(
    """
    CamemBERT Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    CAMEMBERT_START_DOCSTRING,
)
# 从 transformers.models.roberta.modeling_tf_roberta.TFRobertaForTokenClassification 复制，将 Roberta 替换为 Camembert，ROBERTA 替换为 CAMEMBERT
class TFCamembertForTokenClassification(TFCamembertPreTrainedModel, TFTokenClassificationLoss):
    # 在从 PyTorch 模型加载到 TensorFlow 模型时，这些键表示不希望或缺少的层
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head"]
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        # 初始化 Camembert 主层，排除添加池化层，命名为 "roberta"
        self.roberta = TFCamembertMainLayer(config, add_pooling_layer=False, name="roberta")
        
        # 设置分类器的 dropout 比例为 config.classifier_dropout，若未指定则使用 config.hidden_dropout_prob
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = keras.layers.Dropout(classifier_dropout)
        
        # 定义分类器层，输出维度为 config.num_labels，使用给定范围内的初始化器进行初始化，命名为 "classifier"
        self.classifier = keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        
        # 保存配置信息
        self.config = config

    # 使用装饰器解包输入参数，并添加模型前向传播的文档字符串，描述输入格式
    @unpack_inputs
    @add_start_docstrings_to_model_forward(CAMEMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="ydshieh/roberta-large-ner-english",
        output_type=TFTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="['O', 'ORG', 'ORG', 'O', 'O', 'O', 'O', 'O', 'LOC', 'O', 'LOC', 'LOC']",
        expected_loss=0.01,
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
        **kwargs
    ):
        """
        CamemBERT 模型的前向传播方法，支持各种输入参数，返回 TFTokenClassifierOutput 类型的输出结果。
        """
        # 实现前向传播的具体逻辑，包括输入的各种处理和模型输出的计算
        pass
    ) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 使用 Type Hinting 指定函数返回类型，可以是 TFTokenClassifierOutput 或包含 tf.Tensor 的元组
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
        # 获取 RoBERTa 模型的输出序列
        sequence_output = outputs[0]

        # 应用 dropout 操作，用于防止过拟合
        sequence_output = self.dropout(sequence_output, training=training)
        # 对输出序列进行分类器分类
        logits = self.classifier(sequence_output)

        # 如果提供了标签，计算损失函数
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果 return_dict 为 False，则返回不同的输出格式
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则构建 TFTokenClassifierOutput 对象并返回
        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果模型已经构建，直接返回
        if self.built:
            return
        # 设置模型已构建标志
        self.built = True
        # 如果存在 RoBERTa 模型，则构建 RoBERTa 模型
        if getattr(self, "roberta", None) is not None:
            with tf.name_scope(self.roberta.name):
                self.roberta.build(None)
        # 如果存在分类器模型，则构建分类器模型
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
@add_start_docstrings(
    """
    CamemBERT Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    CAMEMBERT_START_DOCSTRING,
)
# 从transformers.models.roberta.modeling_tf_roberta.TFRobertaForMultipleChoice复制过来，将Roberta改为Camembert，ROBERTA改为CAMEMBERT
class TFCamembertForMultipleChoice(TFCamembertPreTrainedModel, TFMultipleChoiceLoss):
    # 当从PyTorch模型加载到TensorFlow模型时，以下带'.'的名称表示授权的意外/缺失层
    _keys_to_ignore_on_load_unexpected = [r"lm_head"]
    # 当从PyTorch模型加载到TensorFlow模型时，以下名称表示授权的缺失层
    _keys_to_ignore_on_load_missing = [r"dropout"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 使用TFCamembertMainLayer初始化Camembert主层，并命名为"roberta"
        self.roberta = TFCamembertMainLayer(config, name="roberta")
        # 使用config.hidden_dropout_prob初始化Dropout层
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        # 使用config.initializer_range初始化Dense层，用于分类
        self.classifier = keras.layers.Dense(
            1, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        CAMEMBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFMultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 模型的前向传播函数，接受多个输入参数并返回相应输出
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
        # 参数training用于指定当前是否处于训练模式，默认为False
    ) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)
        """

        # 如果输入包含 input_ids，则确定 num_choices 和 seq_length
        if input_ids is not None:
            num_choices = shape_list(input_ids)[1]  # 获取选择项的数量
            seq_length = shape_list(input_ids)[2]   # 获取序列长度
        else:
            num_choices = shape_list(inputs_embeds)[1]  # 获取选择项的数量（从 embeddings 中）
            seq_length = shape_list(inputs_embeds)[2]   # 获取序列长度（从 embeddings 中）

        # 根据 input_ids 是否为 None，对输入的张量进行扁平化处理
        flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
        flat_attention_mask = tf.reshape(attention_mask, (-1, seq_length)) if attention_mask is not None else None
        flat_token_type_ids = tf.reshape(token_type_ids, (-1, seq_length)) if token_type_ids is not None else None
        flat_position_ids = tf.reshape(position_ids, (-1, seq_length)) if position_ids is not None else None
        
        # 调用 self.roberta 进行模型的前向传播
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
        
        # 获取池化后的输出（通常是第二个输出）
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, training=training)  # 对池化输出应用 dropout
        logits = self.classifier(pooled_output)  # 使用分类器对池化输出进行分类

        # 将 logits 重新整形为 (batch_size, num_choices)
        reshaped_logits = tf.reshape(logits, (-1, num_choices))

        # 计算损失，如果 labels 不为 None，则计算损失值
        loss = None if labels is None else self.hf_compute_loss(labels, reshaped_logits)

        # 如果不要求返回字典，则返回结果的元组
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果要求返回字典，则返回 TFMultipleChoiceModelOutput 对象
        return TFMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True

        # 如果 self.roberta 存在，则构建 self.roberta 模型
        if getattr(self, "roberta", None) is not None:
            with tf.name_scope(self.roberta.name):
                self.roberta.build(None)

        # 如果 self.classifier 存在，则构建 self.classifier 模型
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
@add_start_docstrings(
    """
    CamemBERT Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    CAMEMBERT_START_DOCSTRING,
)
# 从transformers.models.roberta.modeling_tf_roberta.TFRobertaForQuestionAnswering中复制过来，将Roberta替换为Camembert，将ROBERTA替换为CAMEMBERT
class TFCamembertForQuestionAnswering(TFCamembertPreTrainedModel, TFQuestionAnsweringLoss):
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    # 在从PyTorch模型加载TF模型时，'pooler'和'lm_head'是允许的未预期/缺失的层
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        # 初始化Camembert主层，不添加池化层，命名为"roberta"
        self.roberta = TFCamembertMainLayer(config, add_pooling_layer=False, name="roberta")
        # QA输出层，全连接层，输出维度为config.num_labels，初始化方法为config中定义的initializer_range
        self.qa_outputs = keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CAMEMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="ydshieh/roberta-base-squad2",
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="' puppet'",
        expected_loss=0.86,
    )
    # 定义模型的前向传播方法
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
        **kwargs
    ):
        """
        Perform the forward pass of the model.
        """
        # 调用Camembert主层进行前向传播
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
            **kwargs,
        )

        # 获取Camembert主层的输出
        sequence_output = outputs[0]

        # 计算问题回答的起始位置和结束位置的logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)

        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        outputs = (start_logits, end_logits) + outputs[2:]

        if not return_dict:
            return outputs + (outputs[0],)
        return TFQuestionAnsweringModelOutput(
            start_logits=start_logits, end_logits=end_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
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
        # 调用 RoBERTa 模型进行预测
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
        # 获取模型输出的序列表示
        sequence_output = outputs[0]

        # 对序列表示进行线性变换，得到起始和结束位置的 logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        # 如果提供了起始和结束位置的标签，则计算损失
        loss = None
        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            loss = self.hf_compute_loss(labels, (start_logits, end_logits))

        # 如果不需要返回字典形式的输出，则返回 logits 和可能的其他输出
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TFQuestionAnsweringModelOutput 类型的输出，包括损失、logits、隐藏状态和注意力权重
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
        # 如果定义了 RoBERTa 模型，则构建其结构
        if getattr(self, "roberta", None) is not None:
            with tf.name_scope(self.roberta.name):
                self.roberta.build(None)
        # 如果定义了 QA 输出层，则构建其结构
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])
@add_start_docstrings(
    """CamemBERT Model with a `language modeling` head on top for CLM fine-tuning.""", CAMEMBERT_START_DOCSTRING
)
# 从 transformers.models.roberta.modeling_tf_roberta.TFRobertaForCausalLM 复制并修改为 Camembert，ROBERTA->CAMEMBERT
class TFCamembertForCausalLM(TFCamembertPreTrainedModel, TFCausalLanguageModelingLoss):
    # 在从 PT 模型加载 TF 模型时，以下带有 '.' 的名称表示授权的意外/丢失的层
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head.decoder.weight"]

    def __init__(self, config: CamembertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        if not config.is_decoder:
            logger.warning("如果要将 `TFCamembertLMHeadModel` 作为独立模型使用，请添加 `is_decoder=True.`")

        # 初始化 Camembert 主层，不添加池化层，命名为 "roberta"
        self.roberta = TFCamembertMainLayer(config, add_pooling_layer=False, name="roberta")
        # 初始化 Camembert LM 头部，使用 self.roberta.embeddings 作为输入嵌入，命名为 "lm_head"
        self.lm_head = TFCamembertLMHead(config, input_embeddings=self.roberta.embeddings, name="lm_head")

    def get_lm_head(self):
        return self.lm_head

    def get_prefix_bias_name(self):
        warnings.warn("方法 get_prefix_bias_name 已弃用，请改用 `get_bias`.", FutureWarning)
        # 返回头部名称，以及 LM 头部名称的组合
        return self.name + "/" + self.lm_head.name

    # 从 transformers.models.bert.modeling_tf_bert.TFBertLMHeadModel.prepare_inputs_for_generation 复制
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        # 如果没有提供注意力遮罩，则创建全为1的遮罩
        if attention_mask is None:
            attention_mask = tf.ones(input_shape)

        # 如果存在过去的键值对，则截取最后一个输入 ID
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # 返回包含输入 ID、注意力遮罩和过去键值对的字典
        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CAMEMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFCausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义神经网络模型中的方法，用于执行模型的前向推断或训练
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 输入的 token IDs，类型可以是 TFModelInputType 或 None
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力掩码，类型可以是 numpy 数组、Tensor 或 None
        token_type_ids: np.ndarray | tf.Tensor | None = None,  # token 类型 IDs，类型可以是 numpy 数组、Tensor 或 None
        position_ids: np.ndarray | tf.Tensor | None = None,  # 位置 IDs，类型可以是 numpy 数组、Tensor 或 None
        head_mask: np.ndarray | tf.Tensor | None = None,  # 头部掩码，类型可以是 numpy 数组、Tensor 或 None
        inputs_embeds: np.ndarray | tf.Tensor | None = None,  # 输入的嵌入表示，类型可以是 numpy 数组、Tensor 或 None
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,  # 编码器隐藏状态，类型可以是 numpy 数组、Tensor 或 None
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,  # 编码器的注意力掩码，类型可以是 numpy 数组、Tensor 或 None
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,  # 过去的键值对，类型为可选的嵌套元组
        use_cache: Optional[bool] = None,  # 是否使用缓存，类型为可选的布尔值
        output_attentions: Optional[bool] = None,  # 是否输出注意力信息，类型为可选的布尔值
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，类型为可选的布尔值
        return_dict: Optional[bool] = None,  # 是否返回字典格式的结果，类型为可选的布尔值
        labels: np.ndarray | tf.Tensor | None = None,  # 标签，类型可以是 numpy 数组、Tensor 或 None
        training: Optional[bool] = False,  # 是否处于训练模式，类型为可选的布尔值，默认为 False
    # 构建神经网络模型的结构，设置层的连接关系和参数
    def build(self, input_shape=None):
        if self.built:
            return  # 如果已经构建过，则直接返回

        self.built = True  # 标记模型已经构建

        if getattr(self, "roberta", None) is not None:
            # 如果存在名为 "roberta" 的属性，则在命名空间下构建它
            with tf.name_scope(self.roberta.name):
                self.roberta.build(None)

        if getattr(self, "lm_head", None) is not None:
            # 如果存在名为 "lm_head" 的属性，则在命名空间下构建它
            with tf.name_scope(self.lm_head.name):
                self.lm_head.build(None)
```