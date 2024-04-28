# `.\transformers\models\xlm\modeling_tf_xlm.py`

```py
# 设置脚本使用的字符编码为 UTF-8
# 版权声明和许可证信息
# 版权所有 2019 至今，Facebook, Inc 和 HuggingFace Inc. 团队。
# 根据 Apache 许可证 2.0 版（“许可证”）进行许可；
# 除非符合许可证的要求，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面同意，软件是基于"AS IS"基础分发的，
# 没有任何形式的担保或条件，无论是明示的还是暗示的。
# 请查看许可证，了解特定语言下许可证允许的权限及限制。
"""
# 引入__future__模块，定义：注释中的“annotations”不再引起警告。
import itertools
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFMultipleChoiceModelOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (
    TFModelInputType,
    TFMultipleChoiceLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFSequenceSummary,
    TFSharedEmbeddings,
    TFTokenClassificationLoss,
    get_initializer,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
    MULTIPLE_CHOICE_DUMMY_INPUTS,
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from .configuration_xlm import XLMConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置名称
_CHECKPOINT_FOR_DOC = "xlm-mlm-en-2048"
_CONFIG_FOR_DOC = "XLMConfig"

# XLM 预训练模型的存档列表
TF_XLM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "xlm-mlm-en-2048",
    "xlm-mlm-ende-1024",
    "xlm-mlm-enfr-1024",
    "xlm-mlm-enro-1024",
    "xlm-mlm-tlm-xnli15-1024",
    "xlm-mlm-xnli15-1024",
    "xlm-clm-enfr-1024",
    "xlm-clm-ende-1024",
    "xlm-mlm-17-1280",
    "xlm-mlm-100-1280",
    # 查看所有 XLM 模型：https://huggingface.co/models?filter=xlm
]

# 创建正弦嵌入
def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    out[:, 0::2] = tf.constant(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = tf.constant(np.cos(position_enc[:, 1::2]))

# 生成隐藏状态掩码，以及可选的注意力掩码
def get_masks(slen, lengths, causal, padding_mask=None):
    bs = shape_list(lengths)[0]
    if padding_mask is not None:
        mask = padding_mask
    else:
        # assert lengths.max().item() <= slen
        alen = tf.range(slen, dtype=lengths.dtype)
        mask = alen < tf.expand_dims(lengths, axis=1)

    # 注意力掩码与隐藏状态掩码相同，或是下三角形式的注意力（因果）
    # 如果是因果注意力，则创建一个注意力掩码
    if causal:
        attn_mask = tf.less_equal(
            tf.tile(tf.reshape(alen, (1, 1, slen)), (bs, slen, 1)), tf.reshape(alen, (1, slen, 1))
        )
    # 如果不是因果注意力，则使用给定的掩码
    else:
        attn_mask = mask

    # 对掩码进行形状检查
    # assert shape_list(mask) == [bs, slen]
    tf.debugging.assert_equal(shape_list(mask), [bs, slen])
    # 如果是因果注意力，则对创建的注意力掩码进行形状检查
    if causal:
        tf.debugging.assert_equal(shape_list(attn_mask), [bs, slen, slen])

    # 返回掩码和注意力掩码
    return mask, attn_mask
# 定义一个自定义的多头注意力层类
class TFXLMMultiHeadAttention(tf.keras.layers.Layer):
    # 定义一个类变量，用于生成唯一的标识符
    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, config, **kwargs):
        super().__init__(**kwargs)
        # 为这个实例生成一个唯一的标识符
        self.layer_id = next(TFXLMMultiHeadAttention.NEW_ID)
        self.dim = dim
        self.n_heads = n_heads
        self.output_attentions = config.output_attentions
        assert self.dim % self.n_heads == 0

        # 定义查询、键、值转换层以及输出转换层
        self.q_lin = tf.keras.layers.Dense(dim, kernel_initializer=get_initializer(config.init_std), name="q_lin")
        self.k_lin = tf.keras.layers.Dense(dim, kernel_initializer=get_initializer(config.init_std), name="k_lin")
        self.v_lin = tf.keras.layers.Dense(dim, kernel_initializer=get_initializer(config.init_std), name="v_lin")
        self.out_lin = tf.keras.layers.Dense(dim, kernel_initializer=get_initializer(config.init_std), name="out_lin")
        self.dropout = tf.keras.layers.Dropout(config.attention_dropout)
        self.pruned_heads = set()
        self.dim = dim

    # 头部修剪方法，暂未实现
    def prune_heads(self, heads):
        raise NotImplementedError

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建查询、键、值和输出转换层
        if getattr(self, "q_lin", None) is not None:
            with tf.name_scope(self.q_lin.name):
                self.q_lin.build([None, None, self.dim])
        if getattr(self, "k_lin", None) is not None:
            with tf.name_scope(self.k_lin.name):
                self.k_lin.build([None, None, self.dim])
        if getattr(self, "v_lin", None) is not None:
            with tf.name_scope(self.v_lin.name):
                self.v_lin.build([None, None, self.dim])
        if getattr(self, "out_lin", None) is not None:
            with tf.name_scope(self.out_lin.name):
                self.out_lin.build([None, None, self.dim])

# 定义一个自定义的Transformer前馈神经网络层类
class TFXLMTransformerFFN(tf.keras.layers.Layer):
    def __init__(self, in_dim, dim_hidden, out_dim, config, **kwargs):
        super().__init__(**kwargs)

        # 定义线性转换层
        self.lin1 = tf.keras.layers.Dense(dim_hidden, kernel_initializer=get_initializer(config.init_std), name="lin1")
        self.lin2 = tf.keras.layers.Dense(out_dim, kernel_initializer=get_initializer(config.init_std), name="lin2")
        # 获取GELU或ReLU激活函数
        self.act = get_tf_activation("gelu") if config.gelu_activation else get_tf_activation("relu")
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.in_dim = in_dim
        self.dim_hidden = dim_hidden

    # 前向传播方法
    def call(self, input, training=False):
        x = self.lin1(input)
        x = self.act(x)
        x = self.lin2(x)
        x = self.dropout(x, training=training)

        return x
    # 构建神经网络模型的方法
    def build(self, input_shape=None):
        # 如果模型已经构建完成，则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果存在 self.lin1 属性
        if getattr(self, "lin1", None) is not None:
            # 在 TensorFlow 中使用 name_scope 为线性层指定范围名称
            with tf.name_scope(self.lin1.name):
                # 构建 self.lin1 线性层，指定输入形状
                self.lin1.build([None, None, self.in_dim])
        # 如果存在 self.lin2 属性
        if getattr(self, "lin2", None) is not None:
            # 在 TensorFlow 中使用 name_scope 为线性层指定范围名称
            with tf.name_scope(self.lin2.name):
                # 构建 self.lin2 线性层，指定输入形状
                self.lin2.build([None, None, self.dim_hidden])
# 使用 keras_serializable 装饰器将TFXLMMainLayer类标记为可序列化的
@keras_serializable
class TFXLMMainLayer(tf.keras.layers.Layer):
    # 将config_class属性设置为XLMConfig类
    config_class = XLMConfig

    # 构建方法
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 在"position_embeddings"命名空间下，创建名为position_embeddings的权重，并初始化
        with tf.name_scope("position_embeddings"):
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.dim],
                initializer=get_initializer(self.embed_init_std),
            )

        # 如果包含多种语言并且使用语言嵌入的话，创建名为lang_embeddings的权重，并进行初始化
        if self.n_langs > 1 and self.use_lang_emb:
            with tf.name_scope("lang_embeddings"):
                self.lang_embeddings = self.add_weight(
                    name="embeddings",
                    shape=[self.n_langs, self.dim],
                    initializer=get_initializer(self.embed_init_std),
                )
        # 如果存在embeddings属性，则构建它
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        # 如果存在layer_norm_emb属性，则构建它
        if getattr(self, "layer_norm_emb", None) is not None:
            with tf.name_scope(self.layer_norm_emb.name):
                self.layer_norm_emb.build([None, None, self.dim])
        # 对于每个注意力层，构建它们
        for layer in self.attentions:
            with tf.name_scope(layer.name):
                layer.build(None)
        # 对于每个layer_norm1，构建它们
        for layer in self.layer_norm1:
            with tf.name_scope(layer.name):
                layer.build([None, None, self.dim])
        # 对于每个ffns，构建它们
        for layer in self.ffns:
            with tf.name_scope(layer.name):
                layer.build(None)
        # 对于每个layer_norm2，构建它们
        for layer in self.layer_norm2:
            with tf.name_scope(layer.name):
                layer.build([None, None, self.dim])

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.embeddings

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    # 剪枝模型中的头部
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    # 调用方法
    @unpack_inputs
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        langs=None,
        token_type_ids=None,
        position_ids=None,
        lengths=None,
        cache=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
class TFXLMPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 将config_class属性设置为XLMConfig类
    config_class = XLMConfig
    # 将base_model_prefix属性设置为"transformer"
    base_model_prefix = "transformer"

    # 设置property属性
    @property
    # 定义一个虚拟的输入函数，用于创建输入数据和对应的注意力掩码
    def dummy_inputs(self):
        # 创建一个包含多个数字序列的常量张量作为输入数据
        inputs_list = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]], dtype=tf.int32)
        # 创建一个包含多个数字序列的常量张量作为注意力掩码
        attns_list = tf.constant([[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [1, 0, 0, 1, 1]], dtype=tf.int32)
        # 如果需要使用语言嵌入并且有多个语言，则创建语言张量并返回
        if self.config.use_lang_emb and self.config.n_langs > 1:
            return {
                "input_ids": inputs_list,
                "attention_mask": attns_list,
                "langs": tf.constant([[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [1, 0, 0, 1, 1]], dtype=tf.int32),
            }
        # 否则只返回输入数据和注意力掩码
        else:
            return {"input_ids": inputs_list, "attention_mask": attns_list}
# 定义一个数据类，用于表示带有语言模型头部的 XLM 模型的输出
@dataclass
class TFXLMWithLMHeadModelOutput(ModelOutput):
    """
    Base class for [`TFXLMWithLMHeadModel`] outputs.

    Args:
        logits (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            语言建模头部的预测分数（SoftMax 前每个词汇标记的分数）。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            形状为 `(batch_size, sequence_length, hidden_size)` 的 `tf.Tensor` 元组（当传递 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回）。

            模型在每个层的输出隐藏状态以及初始嵌入输出。
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            形状为 `(batch_size, num_heads, sequence_length, sequence_length)` 的 `tf.Tensor` 元组（当传递 `output_attentions=True` 或 `config.output_attentions=True` 时返回）。

            注意力 softmax 后的注意力权重，用于计算自注意力头部中的加权平均值。
    """

    logits: tf.Tensor = None  # 语言模型头部的预测分数
    hidden_states: Tuple[tf.Tensor] | None = None  # 每个层的隐藏状态输出
    attentions: Tuple[tf.Tensor] | None = None  # 注意力权重


XLM_START_DOCSTRING = r"""

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
    # 调用模型时可以传入不同的输入张量，可以只传入input_ids和attention_mask，也可以传入input_ids、attention_mask和token_type_ids
    - a dictionary with one or several input Tensors associated to the input names given in the docstring:
    # 返回一个字典，其中包含一个或多个与文档字符串中给定的输入名称相关联的输入张量：
    `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    Note that when creating models and layers with
    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don't need to worry
    about any of this, as you can just pass inputs like you would to any other Python function!
    # 注意，在使用子类化创建模型和层时，无需担心这些细节，可以像将输入传递给任何其他Python函数一样传递输入！

    </Tip>

    Parameters:
        config ([`XLMConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    # 参数说明：
    # config ([`XLMConfig`]): 包含模型所有参数的模型配置类。
    # 使用配置文件初始化不会加载与模型关联的权重，仅加载配置。
    # 查看[`~PreTrainedModel.from_pretrained`]方法以加载模型权重。
"""

XLM_INPUTS_DOCSTRING = r"""
"""

# 导入模型所需的库和模块

@add_start_docstrings(
    "The bare XLM Model transformer outputting raw hidden-states without any specific head on top.",
    XLM_START_DOCSTRING,
)
# 定义TFXLMModel类，继承自TFXLMPreTrainedModel

class TFXLMModel(TFXLMPreTrainedModel):
    # 初始化方法
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFXLMMainLayer(config, name="transformer")

    # 调用方法，接收多种输入参数，返回模型输出结果
    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: tf.Tensor | None = None,
        langs: tf.Tensor | None = None,
        token_type_ids: tf.Tensor | None = None,
        position_ids: tf.Tensor | None = None,
        lengths: tf.Tensor | None = None,
        cache: Dict[str, tf.Tensor] | None = None,
        head_mask: tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        training: bool = False,
    ) -> TFBaseModelOutput | Tuple[tf.Tensor]:
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        return outputs

    # 构建模型
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)


class TFXLMPredLayer(tf.keras.layers.Layer):
    """
    Prediction layer (cross_entropy or adaptive_softmax).
    """

    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)

        self.asm = config.asm
        self.n_words = config.n_words
        self.pad_index = config.pad_index

        if config.asm is False:
            # 如果采用交叉熵损失，则直接使用输入嵌入
            self.input_embeddings = input_embeddings
        else:
            # 否则抛出未实现的错误
            raise NotImplementedError
            # self.proj = nn.AdaptiveLogSoftmaxWithLoss(
            #     in_features=dim,
            #     n_classes=config.n_words,
            #     cutoffs=config.asm_cutoffs,
            #     div_value=config.asm_div_value,
            #     head_bias=True,  # default is False
            # )
    # 构建模型的方法，在输入形状上添加权重并初始化偏置
    def build(self, input_shape):
        # 输出权重与输入嵌入相同，但每个标记都有一个仅输出的偏置
        self.bias = self.add_weight(shape=(self.n_words,), initializer="zeros", trainable=True, name="bias")

        # 调用父类的构建方法
        super().build(input_shape)

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.input_embeddings

    # 设置输出嵌入
    def set_output_embeddings(self, value):
        self.input_embeddings.weight = value
        self.input_embeddings.vocab_size = shape_list(value)[0]

    # 获取偏置
    def get_bias(self):
        return {"bias": self.bias}

    # 设置偏置
    def set_bias(self, value):
        self.bias = value["bias"]
        self.vocab_size = shape_list(value["bias"])[0]

    # 模型调用方法，对隐藏状态进行处理
    def call(self, hidden_states):
        # 对隐藏状态进行线性转换
        hidden_states = self.input_embeddings(hidden_states, mode="linear")
        # 添加偏置
        hidden_states = hidden_states + self.bias

        return hidden_states
@add_start_docstrings(
    """
    The XLM Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    XLM_START_DOCSTRING,
)
# 定义包含语言建模头的 XLM 模型变换器类（线性层的权重与输入嵌入层绑定）
class TFXLMWithLMHeadModel(TFXLMPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类构造函数初始化
        super().__init__(config, *inputs, **kwargs)
        # 初始化 transformer 层
        self.transformer = TFXLMMainLayer(config, name="transformer")
        # 初始化预测层
        self.pred_layer = TFXLMPredLayer(config, self.transformer.embeddings, name="pred_layer_._proj")
        # XLM 没有过去缓存特性
        self.supports_xla_generation = False

    # 获取语言模型头部
    def get_lm_head(self):
        return self.pred_layer

    # 获取前缀偏置名称
    def get_prefix_bias_name(self):
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        return self.name + "/" + self.pred_layer.name

    # 为生成准备输入
    def prepare_inputs_for_generation(self, inputs, **kwargs):
        mask_token_id = self.config.mask_token_id
        lang_id = self.config.lang_id

        # 获取有效批次大小
        effective_batch_size = inputs.shape[0]
        # 创建掩码 token
        mask_token = tf.fill((effective_batch_size, 1), 1) * mask_token_id
        # 连接输入和掩码 token
        inputs = tf.concat([inputs, mask_token], axis=1)

        if lang_id is not None:
            langs = tf.ones_like(inputs) * lang_id
        else:
            langs = None
        return {"input_ids": inputs, "langs": langs}

    # 调用方法
    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFXLMWithLMHeadModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        langs: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        lengths: np.ndarray | tf.Tensor | None = None,
        cache: Optional[Dict[str, tf.Tensor]] = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    # 定义模型的前向传播函数，接受一系列输入，并返回模型的输出
    ) -> Union[TFXLMWithLMHeadModelOutput, Tuple[tf.Tensor]]:
        # 调用transformer模型的前向传播函数，并传入相应参数
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 获取transformer模型的输出
        output = transformer_outputs[0]
        # 将transformer模型的输出传入预测层，得到模型的最终输出
        outputs = self.pred_layer(output)

        # 如果return_dict参数为False，则返回一个元组
        if not return_dict:
            return (outputs,) + transformer_outputs[1:]

        # 如果return_dict参数为True，则返回一个TFXLMWithLMHeadModelOutput对象
        return TFXLMWithLMHeadModelOutput(
            logits=outputs, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions
        )

    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回
        if self.built:
            return
        self.built = True
        # 如果transformer模型存在，则构建transformer模型
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        # 如果预测层存在，则构建预测层
        if getattr(self, "pred_layer", None) is not None:
            with tf.name_scope(self.pred_layer.name):
                self.pred_layer.build(None)
# 使用带有序列分类/回归头的 XLM 模型（在汇总输出之上的线性层），例如用于 GLUE 任务
class TFXLMForSequenceClassification(TFXLMPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 获取类别数
        self.num_labels = config.num_labels

        # 创建 XLM 主层对象
        self.transformer = TFXLMMainLayer(config, name="transformer")
        # 创建序列摘要对象
        self.sequence_summary = TFSequenceSummary(config, initializer_range=config.init_std, name="sequence_summary")

    # 模型前向传播函数，添加输入解包、文档字符串和代码示例
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        langs: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        lengths: np.ndarray | tf.Tensor | None = None,
        cache: Optional[Dict[str, tf.Tensor]] = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: bool = False,
    def call(
        self, input_ids: tf.Tensor,  # 输入的 token IDs 张量，shape 为(batch_size, sequence_length)
        attention_mask: Optional[tf.Tensor] = None,  # 注意力遮罩张量，用于指示哪些位置的 token 应被忽略，shape 为(batch_size, sequence_length)
        langs: Optional[tf.Tensor] = None,  # 语言张量，用于多语言模型，shape 为(batch_size, sequence_length)
        token_type_ids: Optional[tf.Tensor] = None,  # Token 类型 IDs 张量，shape 为(batch_size, sequence_length)
        position_ids: Optional[tf.Tensor] = None,  # 位置 IDs 张量，shape 为(batch_size, sequence_length)
        lengths: Optional[tf.Tensor] = None,  # 输入序列的长度张量，shape 为(batch_size,)
        cache: Optional[tf.Tensor] = None,  # 缓存张量，用于缓存注意力权重，shape 为(batch_size, num_heads, sequence_length, sequence_length)
        head_mask: Optional[tf.Tensor] = None,  # 头部掩码张量，shape 为(batch_size, num_heads, sequence_length, sequence_length)
        inputs_embeds: Optional[tf.Tensor] = None,  # 输入嵌入张量，shape 为(batch_size, sequence_length, hidden_size)
        output_attentions: Optional[bool] = None,  # 是否返回注意力权重张量的标志
        output_hidden_states: Optional[bool] = None,  # 是否返回隐藏状态张量的标志
        return_dict: Optional[bool] = None,  # 是否返回字典类型的输出
        training: Optional[bool] = None,  # 是否处于训练模式
        labels: Optional[tf.Tensor] = None,  # 序列分类/回归损失的标签张量，shape 为(batch_size,)
    ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 调用 Transformer 模型进行前向传播
        transformer_outputs = self.transformer(
            input_ids=input_ids,  # 输入的 token IDs 张量
            attention_mask=attention_mask,  # 注意力遮罩张量
            langs=langs,  # 语言张量
            token_type_ids=token_type_ids,  # Token 类型 IDs 张量
            position_ids=position_ids,  # 位置 IDs 张量
            lengths=lengths,  # 输入序列的长度张量
            cache=cache,  # 缓存张量
            head_mask=head_mask,  # 头部掩码张量
            inputs_embeds=inputs_embeds,  # 输入嵌入张量
            output_attentions=output_attentions,  # 是否返回注意力权重张量的标志
            output_hidden_states=output_hidden_states,  # 是否返回隐藏状态张量的标志
            return_dict=return_dict,  # 是否返回字典类型的输出
            training=training,  # 是否处于训练模式
        )
        # 获取 Transformer 模型的输出
        output = transformer_outputs[0]

        # 通过序列摘要层得到 logits
        logits = self.sequence_summary(output)

        # 计算损失（如果有标签）
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果不返回字典类型的输出，则返回元组
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TFSequenceClassifierOutput 类型的输出
        return TFSequenceClassifierOutput(
            loss=loss,  # 损失张量
            logits=logits,  # logits 张量
            hidden_states=transformer_outputs.hidden_states,  # 隐藏状态张量
            attentions=transformer_outputs.attentions,  # 注意力权重张量
        )

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 设置为已构建状态
        self.built = True
        # 构建 Transformer 层
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        # 构建序列摘要层
        if getattr(self, "sequence_summary", None) is not None:
            with tf.name_scope(self.sequence_summary.name):
                self.sequence_summary.build(None)
    # 为多选任务构建一个 XLM 模型，模型中包含一个线性层（放置在池化输出上面）和一个 softmax 层
    # 例如：RocStories/SWAG 任务
    @add_start_docstrings(
        """
        XLM Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
        softmax) e.g. for RocStories/SWAG tasks.
        """,
        XLM_START_DOCSTRING,
    )
    class TFXLMForMultipleChoice(TFXLMPreTrainedModel, TFMultipleChoiceLoss):
        def __init__(self, config, *inputs, **kwargs):
            # 调用父类的构造函数
            super().__init__(config, *inputs, **kwargs)

            # 初始化transformer模块
            self.transformer = TFXLMMainLayer(config, name="transformer")
            # 初始化序列总结模块
            self.sequence_summary = TFSequenceSummary(config, initializer_range=config.init_std, name="sequence_summary")
            # 初始化logits_proj模块
            self.logits_proj = tf.keras.layers.Dense(
                1, kernel_initializer=get_initializer(config.initializer_range), name="logits_proj"
            )
            # 保存配置信息
            self.config = config

        @property
        def dummy_inputs(self):
            """
            Dummy inputs to build the network.

            Returns:
                tf.Tensor with dummy inputs
            """
            # 构建网络的虚拟输入
            # 如果 XLM 有语言嵌入项，也需要构建它们
            if self.config.use_lang_emb and self.config.n_langs > 1:
                return {
                    "input_ids": tf.constant(MULTIPLE_CHOICE_DUMMY_INPUTS, dtype=tf.int32),
                    "langs": tf.constant(MULTIPLE_CHOICE_DUMMY_INPUTS, dtype=tf.int32),
                }
            else:
                return {
                    "input_ids": tf.constant(MULTIPLE_CHOICE_DUMMY_INPUTS, dtype=tf.int32),
                }

        @unpack_inputs
        @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
        @add_code_sample_docstrings(
            checkpoint=_CHECKPOINT_FOR_DOC,
            output_type=TFMultipleChoiceModelOutput,
            config_class=_CONFIG_FOR_DOC,
        )
        def call(
            self,
            input_ids: TFModelInputType | None = None,
            attention_mask: np.ndarray | tf.Tensor | None = None,
            langs: np.ndarray | tf.Tensor | None = None,
            token_type_ids: np.ndarray | tf.Tensor | None = None,
            position_ids: np.ndarray | tf.Tensor | None = None,
            lengths: np.ndarray | tf.Tensor | None = None,
            cache: Optional[Dict[str, tf.Tensor]] = None,
            head_mask: np.ndarray | tf.Tensor | None = None,
            inputs_embeds: np.ndarray | tf.Tensor | None = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            labels: np.ndarray | tf.Tensor | None = None,
            training: bool = False,
    # 该函数用于处理多项选择任务的 TensorFlow 模型输入
    def call(
        self,
        input_ids: Optional[tf.Tensor] = None,
        attention_mask: Optional[tf.Tensor] = None,
        token_type_ids: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        langs: Optional[tf.Tensor] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        lengths: Optional[tf.Tensor] = None,
        cache: Optional[Tuple[tf.Tensor]] = None,
        head_mask: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
        # 如果输入了 input_ids，则获取批量大小和序列长度
        if input_ids is not None:
            num_choices = shape_list(input_ids)[1]
            seq_length = shape_list(input_ids)[2]
        # 否则，从 inputs_embeds 中获取批量大小和序列长度
        else:
            num_choices = shape_list(inputs_embeds)[1]
            seq_length = shape_list(inputs_embeds)[2]
    
        # 将输入的 tensor 展平为 2D 
        flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
        flat_attention_mask = tf.reshape(attention_mask, (-1, seq_length)) if attention_mask is not None else None
        flat_token_type_ids = tf.reshape(token_type_ids, (-1, seq_length)) if token_type_ids is not None else None
        flat_position_ids = tf.reshape(position_ids, (-1, seq_length)) if position_ids is not None else None
        flat_langs = tf.reshape(langs, (-1, seq_length)) if langs is not None else None
        flat_inputs_embeds = (
            tf.reshape(inputs_embeds, (-1, seq_length, shape_list(inputs_embeds)[3]))
            if inputs_embeds is not None
            else None
        )
    
        # 如果传入了 lengths 参数，则给出警告并忽略
        if lengths is not None:
            logger.warning(
                "The `lengths` parameter cannot be used with the XLM multiple choice models. Please use the "
                "attention mask instead.",
            )
            lengths = None
    
        # 调用 transformer 模型获取输出
        transformer_outputs = self.transformer(
            flat_input_ids,
            flat_attention_mask,
            flat_langs,
            flat_token_type_ids,
            flat_position_ids,
            lengths,
            cache,
            head_mask,
            flat_inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        output = transformer_outputs[0]
        # 使用序列摘要层获取 logits
        logits = self.sequence_summary(output)
        logits = self.logits_proj(logits)
        # 将 logits 重塑为 (batch_size, num_choices) 的形状
        reshaped_logits = tf.reshape(logits, (-1, num_choices))
    
        # 计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, reshaped_logits)
    
        # 根据 return_dict 参数返回不同的输出
        if not return_dict:
            output = (reshaped_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
    
        return TFMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
    # 根据输入形状构建模型
    def build(self, input_shape=None):
        # 检查模型是否已经构建，如果是则直接返回
        if self.built:
            return
        # 设置模型为已构建状态
        self.built = True
        # 如果存在transformer属性，则使用transformer的名称作为命名空间，构建transformer
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        # 如果存在sequence_summary属性，则使用sequence_summary的名称作为命名空间，构建sequence_summary
        if getattr(self, "sequence_summary", None) is not None:
            with tf.name_scope(self.sequence_summary.name):
                self.sequence_summary.build(None)
        # 如果存在logits_proj属性，则使用logits_proj的名称作为命名空间，构建logits_proj
        if getattr(self, "logits_proj", None) is not None:
            with tf.name_scope(self.logits_proj.name):
                self.logits_proj.build([None, None, self.config.num_labels])
# 以 XLM 模型为基础，添加了一个顶部的令牌分类头部，用于命名实体识别（NER）等任务
@add_start_docstrings(
    """
    XLM Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    XLM_START_DOCSTRING,
)
class TFXLMForTokenClassification(TFXLMPreTrainedModel, TFTokenClassificationLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 获取标签数目
        self.num_labels = config.num_labels

        # 初始化 XLM 主层
        self.transformer = TFXLMMainLayer(config, name="transformer")
        # 添加丢弃层
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        # 添加分类器
        self.classifier = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.init_std), name="classifier"
        )
        # 保存配置
        self.config = config

    # 解压输入
    @unpack_inputs
    # 向模型前向传播添加起始文档字符串
    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 向模型前向传播添加代码示例文档字符串
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        langs: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        lengths: np.ndarray | tf.Tensor | None = None,
        cache: Optional[Dict[str, tf.Tensor]] = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: bool = False,
        # 其他参数...
    # 定义该函数的返回类型，可以是 TFTokenClassifierOutput 或 Tuple[tf.Tensor]
    def call(
        self,
        input_ids: Optional[tf.Tensor] = None,
        attention_mask: Optional[tf.Tensor] = None,
        langs: Optional[tf.Tensor] = None,
        token_type_ids: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        lengths: Optional[tf.Tensor] = None,
        cache: Optional[Dict[str, tf.Tensor]] = None,
        head_mask: Optional[tf.Tensor] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[tf.Tensor] = None,
        training: Optional[bool] = False,
    ) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        # 获取 transformer 模型的输出，包括隐藏状态、注意力权重等
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 提取 transformer 输出的序列输出
        sequence_output = transformer_outputs[0]
        # 对序列输出应用 Dropout 操作
        sequence_output = self.dropout(sequence_output, training=training)
        # 将 dropout 后的序列输出输入分类器得到logits
        logits = self.classifier(sequence_output)
        # 如果提供了标签，则计算分类损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)
        # 如果不需要返回字典，则返回 logits 和其他 transformer 输出
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        # 否则返回 TFTokenClassifierOutput
        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
    
    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已构建，则直接返回
        if self.built:
            return
        self.built = True
        # 构建 transformer 子模型
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        # 构建分类器子模型
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
# 添加模型文档字符串和XLM模型的起始文档字符串
@add_start_docstrings(
    """
    XLM Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layer
    on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    XLM_START_DOCSTRING,
)
# 创建TFXLMForQuestionAnsweringSimple类，继承自TFXLMPreTrainedModel和TFQuestionAnsweringLoss
class TFXLMForQuestionAnsweringSimple(TFXLMPreTrainedModel, TFQuestionAnsweringLoss):
    # 初始化函数
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类初始化函数
        super().__init__(config, *inputs, **kwargs)
        # 创建XLM主层
        self.transformer = TFXLMMainLayer(config, name="transformer")
        # 创建问题-回答输出层
        self.qa_outputs = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.init_std), name="qa_outputs"
        )
        # 保存配置信息
        self.config = config
    
    # 定义call方法
    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        langs: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        lengths: np.ndarray | tf.Tensor | None = None,
        cache: Optional[Dict[str, tf.Tensor]] = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        start_positions: np.ndarray | tf.Tensor | None = None,
        end_positions: np.ndarray | tf.Tensor | None = None,
        training: bool = False,
        # 剩余的参数，用于模型的输入和训练控制
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
        # 使用 transformer 模型对输入进行处理，并获取 transformer 的输出
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 从 transformer 的输出中获取序列输出
        sequence_output = transformer_outputs[0]

        # 使用 qa_outputs 层处理序列输出，得到答案的起始位置和结束位置的预测结果
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        # 去除两个 logit 中的冗余维度
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        loss = None
        # 如果传入了答案的起始位置和结束位置，则计算损失函数
        if start_positions is not None and end_positions is not None:
            # 组合起始位置和结束位置标签
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            # 使用交叉熵损失函数计算损失
            loss = self.hf_compute_loss(labels, (start_logits, end_logits))

        # 如果不需要返回字典类型的结果，直接返回起始位置和结束位置的 logit
        if not return_dict:
            output = (start_logits, end_logits) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回字典类型的结果
        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 transformer 层，则构建 transformer
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        # 如果存在 qa_outputs 层，则构建 qa_outputs
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])


注释：
```