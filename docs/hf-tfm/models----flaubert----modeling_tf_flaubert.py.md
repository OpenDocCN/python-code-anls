# `.\models\flaubert\modeling_tf_flaubert.py`

```py
# 引入必要的库和模块
import itertools  # itertools模块用于高效地迭代操作
import random  # random模块用于生成随机数和随机选择
import warnings  # warnings模块用于管理警告信息
from dataclasses import dataclass  # dataclass用于创建数据类，简化数据对象的创建和操作
from typing import Dict, Optional, Tuple, Union  # 引入类型提示，用于静态类型检查

import numpy as np  # 引入numpy库，用于数值计算
import tensorflow as tf  # 引入TensorFlow库，用于构建和训练神经网络模型

# 从transformers库中引入所需的TensorFlow相关模块和类
from ...activations_tf import get_tf_activation  # 从transformers.activations_tf模块导入get_tf_activation函数
from ...modeling_tf_outputs import (  # 导入transformers.modeling_tf_outputs模块中的各种输出类
    TFBaseModelOutput,
    TFMultipleChoiceModelOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (  # 导入transformers.modeling_tf_utils模块中的各种实用函数和类
    TFModelInputType,
    TFMultipleChoiceLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFSequenceSummary,
    TFSharedEmbeddings,
    TFTokenClassificationLoss,
    get_initializer,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import (  # 导入transformers.tf_utils模块中的各种实用函数
    check_embeddings_within_bounds,
    shape_list,
    stable_softmax,
)
from ...utils import (  # 导入transformers.utils模块中的各种实用函数和类
    MULTIPLE_CHOICE_DUMMY_INPUTS,
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from .configuration_flaubert import FlaubertConfig  # 导入当前目录下的FlaubertConfig配置类

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义文档中用到的一些常量
_CHECKPOINT_FOR_DOC = "flaubert/flaubert_base_cased"
_CONFIG_FOR_DOC = "FlaubertConfig"

# 定义Flaubert预训练模型存档列表
TF_FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    # 查看所有Flaubert模型的存档列表：https://huggingface.co/models?filter=flaubert
]

# 定义Flaubert模型的起始文档字符串
FLAUBERT_START_DOCSTRING = r"""

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
    # Parameters: config ([`FlaubertConfig`]): Model configuration class with all the parameters of the model.
    # Initializing with a config file does not load the weights associated with the model, only the
    # configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    config ([`FlaubertConfig`]):
        # Model configuration class containing all model parameters.
        Model configuration class with all the parameters of the model.
        
        # Initializing with a config file does not load the weights associated with the model, only the configuration.
        Initializing with a config file does not load the weights associated with the model, only the configuration.
        
        # Check out the [`~PreTrained Model method weights associated Model from explained. contains meaning unknown.
"""

FLAUBERT_INPUTS_DOCSTRING = r"""
"""


def get_masks(slen, lengths, causal, padding_mask=None):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    # 获取批量大小
    bs = shape_list(lengths)[0]
    # 如果提供了padding_mask，则使用之；否则根据lengths生成mask
    if padding_mask is not None:
        mask = padding_mask
    else:
        # 生成一个长度为slen的序列
        alen = tf.range(slen, dtype=lengths.dtype)
        # 生成mask，标识每个位置是否有效（小于对应的lengths）
        mask = alen < tf.expand_dims(lengths, axis=1)

    # attention mask可以是与mask相同，或者是下三角形式的（因果性）
    if causal:
        # 下三角形式的注意力mask
        attn_mask = tf.less_equal(
            tf.tile(tf.reshape(alen, (1, 1, slen)), (bs, slen, 1)), tf.reshape(alen, (1, slen, 1))
        )
    else:
        attn_mask = mask

    # 断言检查
    tf.debugging.assert_equal(shape_list(mask), [bs, slen])
    if causal:
        tf.debugging.assert_equal(shape_list(attn_mask), [bs, slen, slen])

    return mask, attn_mask


class TFFlaubertPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = FlaubertConfig
    base_model_prefix = "transformer"

    @property
    def dummy_inputs(self):
        # 有时候Flaubert模型包含语言嵌入，如果需要，不要忘记同时构建它们
        inputs_list = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]], dtype=tf.int32)
        attns_list = tf.constant([[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [1, 0, 0, 1, 1]], dtype=tf.int32)
        if self.config.use_lang_emb and self.config.n_langs > 1:
            return {
                "input_ids": inputs_list,
                "attention_mask": attns_list,
                "langs": tf.constant([[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [1, 0, 0, 1, 1]], dtype=tf.int32),
            }
        else:
            return {"input_ids": inputs_list, "attention_mask": attns_list}


@add_start_docstrings(
    "The bare Flaubert Model transformer outputting raw hidden-states without any specific head on top.",
    FLAUBERT_START_DOCSTRING,
)
class TFFlaubertModel(TFFlaubertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 初始化transformer模块
        self.transformer = TFFlaubertMainLayer(config, name="transformer")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: np.ndarray | tf.Tensor | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        langs: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        lengths: np.ndarray | tf.Tensor | None = None,
        cache: Optional[Dict[str, tf.Tensor]] = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[Tuple, TFBaseModelOutput]:
        # 调用 Transformer 模型进行前向传播，并返回输出结果
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
        # 返回 Transformer 的输出结果
        return outputs

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果存在 Transformer 模型，则在 TensorFlow 图中构建它
        if getattr(self, "transformer", None) is not None:
            # 使用 Transformer 的名称作为 TensorFlow 名称空间
            with tf.name_scope(self.transformer.name):
                # 构建 Transformer 模型
                self.transformer.build(None)
# 从transformers.models.xlm.modeling_tf_xlm.TFXLMMultiHeadAttention复制并将XLM->Flaubert
class TFFlaubertMultiHeadAttention(keras.layers.Layer):
    # 类变量，用于生成每个实例的唯一标识符
    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, config, **kwargs):
        super().__init__(**kwargs)
        # 为当前实例分配唯一的标识符
        self.layer_id = next(TFFlaubertMultiHeadAttention.NEW_ID)
        self.dim = dim  # 注意力机制中向量的维度
        self.n_heads = n_heads  # 注意力头的数量
        self.output_attentions = config.output_attentions  # 控制是否输出注意力权重
        assert self.dim % self.n_heads == 0  # 确保维度可以被注意力头数量整除

        # 初始化查询、键、值的线性层
        self.q_lin = keras.layers.Dense(dim, kernel_initializer=get_initializer(config.init_std), name="q_lin")
        self.k_lin = keras.layers.Dense(dim, kernel_initializer=get_initializer(config.init_std), name="k_lin")
        self.v_lin = keras.layers.Dense(dim, kernel_initializer=get_initializer(config.init_std), name="v_lin")
        # 初始化输出线性层
        self.out_lin = keras.layers.Dense(dim, kernel_initializer=get_initializer(config.init_std), name="out_lin")
        # 注意力机制中的dropout层
        self.dropout = keras.layers.Dropout(config.attention_dropout)
        # 被剪枝掉的注意力头集合
        self.pruned_heads = set()
        self.dim = dim  # 注意力机制中向量的维度

    # 未实现的方法，用于剪枝注意力头
    def prune_heads(self, heads):
        raise NotImplementedError

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在查询线性层，则构建查询线性层
        if getattr(self, "q_lin", None) is not None:
            with tf.name_scope(self.q_lin.name):
                self.q_lin.build([None, None, self.dim])
        # 如果存在键线性层，则构建键线性层
        if getattr(self, "k_lin", None) is not None:
            with tf.name_scope(self.k_lin.name):
                self.k_lin.build([None, None, self.dim])
        # 如果存在值线性层，则构建值线性层
        if getattr(self, "v_lin", None) is not None:
            with tf.name_scope(self.v_lin.name):
                self.v_lin.build([None, None, self.dim])
        # 如果存在输出线性层，则构建输出线性层
        if getattr(self, "out_lin", None) is not None:
            with tf.name_scope(self.out_lin.name):
                self.out_lin.build([None, None, self.dim])


# 从transformers.models.xlm.modeling_tf_xlm.TFXLMTransformerFFN复制
class TFFlaubertTransformerFFN(keras.layers.Layer):
    def __init__(self, in_dim, dim_hidden, out_dim, config, **kwargs):
        super().__init__(**kwargs)

        # 第一个全连接层，用于FFN的第一步变换
        self.lin1 = keras.layers.Dense(dim_hidden, kernel_initializer=get_initializer(config.init_std), name="lin1")
        # 第二个全连接层，用于FFN的第二步变换
        self.lin2 = keras.layers.Dense(out_dim, kernel_initializer=get_initializer(config.init_std), name="lin2")
        # 激活函数选择，如果配置为GELU激活函数，则使用GELU，否则使用ReLU
        self.act = get_tf_activation("gelu") if config.gelu_activation else get_tf_activation("relu")
        # dropout层，用于防止过拟合
        self.dropout = keras.layers.Dropout(config.dropout)
        self.in_dim = in_dim  # 输入维度
        self.dim_hidden = dim_hidden  # 隐藏层维度

    def call(self, input, training=False):
        # 第一步变换：输入经过第一个全连接层
        x = self.lin1(input)
        # 应用激活函数
        x = self.act(x)
        # 第二步变换：经过第二个全连接层
        x = self.lin2(x)
        # 应用dropout，只有在训练时才应用dropout
        x = self.dropout(x, training=training)

        return x
    # 定义一个方法 `build`，用于构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建完成，直接返回，不进行重复构建
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        
        # 检查并构建第一个线性层 `lin1`
        if getattr(self, "lin1", None) is not None:
            # 使用 `lin1` 的名称作为命名空间，构建 `lin1` 层，输入维度为 [None, None, self.in_dim]
            with tf.name_scope(self.lin1.name):
                self.lin1.build([None, None, self.in_dim])
        
        # 检查并构建第二个线性层 `lin2`
        if getattr(self, "lin2", None) is not None:
            # 使用 `lin2` 的名称作为命名空间，构建 `lin2` 层，输入维度为 [None, None, self.dim_hidden]
            with tf.name_scope(self.lin2.name):
                self.lin2.build([None, None, self.dim_hidden])
# 定义一个可序列化的 Keras 层，用于处理 Flaubert 模型的主要功能
@keras_serializable
class TFFlaubertMainLayer(keras.layers.Layer):
    # 配置类指定为 FlaubertConfig
    config_class = FlaubertConfig

    # 初始化函数，接受配置参数 config 和其他关键字参数
    def __init__(self, config, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 将传入的配置参数 config 存储在实例变量 self.config 中
        self.config = config
        # 从配置中获取并存储各种属性，如头数、语言数、嵌入维度等
        self.n_heads = config.n_heads
        self.n_langs = config.n_langs
        self.dim = config.emb_dim
        self.hidden_dim = self.dim * 4
        self.n_words = config.n_words
        self.pad_index = config.pad_index
        self.causal = config.causal
        self.n_layers = config.n_layers
        self.use_lang_emb = config.use_lang_emb
        self.layerdrop = getattr(config, "layerdrop", 0.0)
        self.pre_norm = getattr(config, "pre_norm", False)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.return_dict = config.use_return_dict
        self.max_position_embeddings = config.max_position_embeddings
        self.embed_init_std = config.embed_init_std

        # 创建一个 Dropout 层，用于后续的 Dropout 操作
        self.dropout = keras.layers.Dropout(config.dropout)
        
        # 创建共享的嵌入层 TFSharedEmbeddings，用于词嵌入
        self.embeddings = TFSharedEmbeddings(
            self.n_words, self.dim, initializer_range=config.embed_init_std, name="embeddings"
        )
        
        # 创建层归一化层，用于嵌入层的输出
        self.layer_norm_emb = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm_emb")
        
        # 初始化存储注意力、层归一化、前馈神经网络和第二层归一化的列表
        self.attentions = []
        self.layer_norm1 = []
        self.ffns = []
        self.layer_norm2 = []

        # 根据层数 self.n_layers 迭代创建注意力、层归一化、前馈神经网络和第二层归一化
        for i in range(self.n_layers):
            # 创建多头注意力层 TFFlaubertMultiHeadAttention
            self.attentions.append(
                TFFlaubertMultiHeadAttention(self.n_heads, self.dim, config=config, name=f"attentions_._{i}")
            )
            
            # 创建第一层归一化层
            self.layer_norm1.append(
                keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name=f"layer_norm1_._{i}")
            )
            
            # 创建 Transformer 中的前馈神经网络层 TFFlaubertTransformerFFN
            self.ffns.append(
                TFFlaubertTransformerFFN(self.dim, self.hidden_dim, self.dim, config=config, name=f"ffns_._{i}")
            )
            
            # 创建第二层归一化层
            self.layer_norm2.append(
                keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name=f"layer_norm2_._{i}")
            )
    # 在 TensorFlow 名称空间 "position_embeddings" 下创建位置嵌入权重矩阵
    def build(self, input_shape=None):
        with tf.name_scope("position_embeddings"):
            # 添加权重变量，用于存储位置嵌入矩阵，形状为 [最大位置数, 嵌入维度]
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.dim],
                initializer=get_initializer(self.embed_init_std),
            )

        # 如果有多语言且使用语言嵌入，创建语言嵌入权重矩阵
        if self.n_langs > 1 and self.use_lang_emb:
            with tf.name_scope("lang_embeddings"):
                # 添加权重变量，用于存储语言嵌入矩阵，形状为 [语言数, 嵌入维度]
                self.lang_embeddings = self.add_weight(
                    name="embeddings",
                    shape=[self.n_langs, self.dim],
                    initializer=get_initializer(self.embed_init_std),
                )

        # 如果已经建立过网络结构，则直接返回
        if self.built:
            return
        self.built = True

        # 如果存在 embeddings 属性，则构建 embeddings 属性
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)

        # 如果存在 layer_norm_emb 属性，则构建 layer_norm_emb 属性
        if getattr(self, "layer_norm_emb", None) is not None:
            with tf.name_scope(self.layer_norm_emb.name):
                # 构建 layer_norm_emb 属性，形状为 [None, None, 嵌入维度]
                self.layer_norm_emb.build([None, None, self.dim])

        # 遍历注意力层列表，构建每个注意力层
        for layer in self.attentions:
            with tf.name_scope(layer.name):
                layer.build(None)

        # 遍历第一个层归一化列表，构建每个层归一化层
        for layer in self.layer_norm1:
            with tf.name_scope(layer.name):
                # 构建每个层归一化层，形状为 [None, None, 嵌入维度]
                layer.build([None, None, self.dim])

        # 遍历前馈神经网络列表，构建每个前馈神经网络层
        for layer in self.ffns:
            with tf.name_scope(layer.name):
                layer.build(None)

        # 遍历第二个层归一化列表，构建每个层归一化层
        for layer in self.layer_norm2:
            with tf.name_scope(layer.name):
                # 构建每个层归一化层，形状为 [None, None, 嵌入维度]
                layer.build([None, None, self.dim])

    # 返回 embeddings 属性
    def get_input_embeddings(self):
        return self.embeddings

    # 设置 embeddings 属性的值，并更新其词汇大小
    def set_input_embeddings(self, value):
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    # 装饰器，用于解包输入参数并调用模型
    @unpack_inputs
    def call(
        self,
        input_ids: np.ndarray | tf.Tensor | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        langs: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        lengths: np.ndarray | tf.Tensor | None = None,
        cache: Optional[Dict[str, tf.Tensor]] = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
# Copied from transformers.models.xlm.modeling_tf_xlm.TFXLMPredLayer
class TFFlaubertPredLayer(keras.layers.Layer):
    """
    Prediction layer (cross_entropy or adaptive_softmax).
    """

    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)

        self.asm = config.asm  # 是否使用自适应 softmax
        self.n_words = config.n_words  # 词汇表中的单词数量
        self.pad_index = config.pad_index  # 填充索引

        if config.asm is False:
            self.input_embeddings = input_embeddings  # 输入的嵌入层对象
        else:
            raise NotImplementedError  # 如果使用自适应 softmax，暂未实现的情况
            # self.proj = nn.AdaptiveLogSoftmaxWithLoss(
            #     in_features=dim,
            #     n_classes=config.n_words,
            #     cutoffs=config.asm_cutoffs,
            #     div_value=config.asm_div_value,
            #     head_bias=True,  # 默认为 False
            # )

    def build(self, input_shape):
        # 输出的权重与输入的嵌入层相同，但是每个标记有一个独立的输出偏置
        self.bias = self.add_weight(shape=(self.n_words,), initializer="zeros", trainable=True, name="bias")

        super().build(input_shape)

    def get_output_embeddings(self):
        return self.input_embeddings  # 返回输入的嵌入层对象

    def set_output_embeddings(self, value):
        self.input_embeddings.weight = value  # 设置输入的嵌入层权重
        self.input_embeddings.vocab_size = shape_list(value)[0]  # 设置词汇表大小为权重的第一维大小

    def get_bias(self):
        return {"bias": self.bias}  # 返回偏置参数

    def set_bias(self, value):
        self.bias = value["bias"]  # 设置偏置参数
        self.vocab_size = shape_list(value["bias"])[0]  # 设置词汇表大小为偏置参数的第一维大小

    def call(self, hidden_states):
        hidden_states = self.input_embeddings(hidden_states, mode="linear")  # 使用线性模式进行嵌入
        hidden_states = hidden_states + self.bias  # 加上偏置参数

        return hidden_states


@dataclass
class TFFlaubertWithLMHeadModelOutput(ModelOutput):
    """
    Base class for [`TFFlaubertWithLMHeadModel`] outputs.

    Args:
        logits (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
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

    logits: tf.Tensor = None  # 语言建模头部的预测分数（SoftMax 前的每个词汇标记的分数）
    # 定义变量 hidden_states，类型为 Tuple[tf.Tensor] 或 None，初始值为 None
    hidden_states: Tuple[tf.Tensor] | None = None
    # 定义变量 attentions，类型为 Tuple[tf.Tensor] 或 None，初始值为 None
    attentions: Tuple[tf.Tensor] | None = None
"""
The Flaubert Model transformer with a language modeling head on top (linear layer with weights tied to the input
embeddings).
"""
# 导入必要的库和模块
@add_start_docstrings(
    """
    The Flaubert Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    FLAUBERT_START_DOCSTRING,
)
# 定义 TFFlaubertWithLMHeadModel 类，继承自 TFFlaubertPreTrainedModel
class TFFlaubertWithLMHeadModel(TFFlaubertPreTrainedModel):
    
    # 初始化方法
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 创建 Flaubert 主层，并命名为 "transformer"
        self.transformer = TFFlaubertMainLayer(config, name="transformer")
        # 创建 Flaubert 预测层，并将 embeddings 传入，命名为 "pred_layer_._proj"
        self.pred_layer = TFFlaubertPredLayer(config, self.transformer.embeddings, name="pred_layer_._proj")
        # Flaubert 模型不支持过去的缓存特性
        self.supports_xla_generation = False

    # 获取语言模型头部的方法
    def get_lm_head(self):
        return self.pred_layer

    # 获取前缀偏置名字的方法（已弃用）
    def get_prefix_bias_name(self):
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        return self.name + "/" + self.pred_layer.name

    # 准备生成所需输入的方法
    def prepare_inputs_for_generation(self, inputs, **kwargs):
        # 获取配置中的 mask_token_id 和 lang_id
        mask_token_id = self.config.mask_token_id
        lang_id = self.config.lang_id

        # 获取有效的批处理大小
        effective_batch_size = inputs.shape[0]
        # 创建 mask_token，填充到输入张量的末尾
        mask_token = tf.fill((effective_batch_size, 1), 1) * mask_token_id
        inputs = tf.concat([inputs, mask_token], axis=1)

        # 如果 lang_id 不为 None，则创建相应语言标识符张量
        if lang_id is not None:
            langs = tf.ones_like(inputs) * lang_id
        else:
            langs = None
        return {"input_ids": inputs, "langs": langs}

    # 定义模型的调用方法，处理输入并执行前向传播
    @unpack_inputs
    @add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFFlaubertWithLMHeadModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: np.ndarray | tf.Tensor | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        langs: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        lengths: np.ndarray | tf.Tensor | None = None,
        cache: Optional[Dict[str, tf.Tensor]] = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
        **kwargs
    ):
        # 方法用于执行模型的前向传播，并接受多种类型的输入参数
    ) -> Union[Tuple, TFFlaubertWithLMHeadModelOutput]:
        # 调用 Transformer 模型处理输入数据，返回变换后的输出
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
        # 从 Transformer 输出中获取主要的输出
        output = transformer_outputs[0]
        # 通过预测层对主要输出进行预测
        outputs = self.pred_layer(output)

        # 如果不要求返回字典，则以元组形式返回预测输出和 Transformer 的其余输出
        if not return_dict:
            return (outputs,) + transformer_outputs[1:]

        # 如果要求返回字典，则创建 TFFlaubertWithLMHeadModelOutput 对象
        return TFFlaubertWithLMHeadModelOutput(
            logits=outputs, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions
        )

    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回
        if self.built:
            return
        # 标记模型为已构建状态
        self.built = True
        # 如果存在 Transformer 属性，则构建 Transformer
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        # 如果存在预测层属性，则构建预测层
        if getattr(self, "pred_layer", None) is not None:
            with tf.name_scope(self.pred_layer.name):
                self.pred_layer.build(None)
# 添加模型的文档字符串，描述该模型在顶部具有一个用于序列分类/回归的头部（在汇总输出之上的线性层），例如用于GLUE任务。
# 这里使用了FLAUBERT_START_DOCSTRING和其他相关文档字符串作为起始。
@add_start_docstrings(
    """
    Flaubert Model with a sequence classification/regression head on top (a linear layer on top of the pooled output)
    e.g. for GLUE tasks.
    """,
    FLAUBERT_START_DOCSTRING,
)
# 从transformers.models.xlm.modeling_tf_xlm.TFXLMForSequenceClassification复制而来，
# 将XLM_INPUTS替换为FLAUBERT_INPUTS，将XLM替换为Flaubert
class TFFlaubertForSequenceClassification(TFFlaubertPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        # 初始化transformer层，使用TFFlaubertMainLayer
        self.transformer = TFFlaubertMainLayer(config, name="transformer")
        # 初始化序列汇总层，使用TFSequenceSummary
        self.sequence_summary = TFSequenceSummary(config, initializer_range=config.init_std, name="sequence_summary")

    # 将输入解包后传递给模型的前向计算，添加了FLAUBERT_INPUTS_DOCSTRING作为模型前向计算的文档字符串
    # 还包括了checkpoint、output_type、config_class作为代码示例的文档字符串
    @unpack_inputs
    @add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSequenceClassifierOutput,
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
    ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 调用transformer模型，传入各种参数，获取transformer的输出
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
        # 获取transformer输出的第一个元素作为输出
        output = transformer_outputs[0]

        # 对输出进行序列摘要，生成logits
        logits = self.sequence_summary(output)

        # 如果labels不为None，计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果不返回字典形式的结果，则将logits与其余transformer输出拼接在一起返回
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回TFSequenceClassifierOutput对象，包含损失、logits、隐藏状态和注意力权重
        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建过网络，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在transformer模型，则在其名字域内构建transformer
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        # 如果存在sequence_summary模型，则在其名字域内构建sequence_summary
        if getattr(self, "sequence_summary", None) is not None:
            with tf.name_scope(self.sequence_summary.name):
                self.sequence_summary.build(None)
@add_start_docstrings(
    """
    Flaubert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    FLAUBERT_START_DOCSTRING,
)
# 基于 Flaubert 模型，添加了一个面向提取式问答任务（如 SQuAD）的跨度分类头部（在隐藏状态输出之上的线性层，用于计算 `span start logits` 和 `span end logits`）。
class TFFlaubertForQuestionAnsweringSimple(TFFlaubertPreTrainedModel, TFQuestionAnsweringLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFFlaubertMainLayer(config, name="transformer")
        # 初始化一个全连接层用于问答输出，其输出维度为 config.num_labels，使用指定的初始化方式初始化权重
        self.qa_outputs = keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.init_std), name="qa_outputs"
        )
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义模型的前向传播函数，接受多种输入，并返回一个字典或 TFQuestionAnsweringModelOutput 类型的输出
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
        # 可能的训练参数包括起始位置和结束位置，用于指定答案的位置
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
        # 获取transformer模型的输出，包括各种参数和特征
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
        # 从transformer输出的序列输出中提取结果
        sequence_output = transformer_outputs[0]

        # 使用qa_outputs层对序列输出进行转换，得到起始和结束位置的logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        # 计算损失函数
        loss = None
        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            loss = self.hf_compute_loss(labels, (start_logits, end_logits))

        # 如果不返回字典，则输出start_logits、end_logits和transformer_outputs的其余部分
        if not return_dict:
            output = (start_logits, end_logits) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回TFQuestionAnsweringModelOutput对象，包括loss、start_logits、end_logits以及其他transformer输出
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
        # 如果存在transformer模型，则构建transformer层
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        # 如果存在qa_outputs层，则构建qa_outputs层
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])
@add_start_docstrings(
    """
    Flaubert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    FLAUBERT_START_DOCSTRING,
)
# 定义了一个用于标记分类任务的 Flaubert 模型，该模型在隐藏状态输出之上添加了一个线性层，用于例如命名实体识别（NER）任务。
# 此处的注释是一个函数装饰器，用于在类的开头添加文档字符串。

class TFFlaubertForTokenClassification(TFFlaubertPreTrainedModel, TFTokenClassificationLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        # 初始化模型的主要组件
        self.transformer = TFFlaubertMainLayer(config, name="transformer")
        # 添加一个用于防止过拟合的 dropout 层
        self.dropout = keras.layers.Dropout(config.dropout)
        # 分类器层，使用全连接层，输出维度为标签数目
        self.classifier = keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.init_std), name="classifier"
        )
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义了模型的前向传播方法
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
        # 以下是函数参数列表，包括输入数据和模型的控制参数
    ) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 调用Transformer模型，传入各种输入参数，并获取输出结果
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
        # 从Transformer的输出中获取序列输出
        sequence_output = transformer_outputs[0]

        # 对序列输出进行dropout操作，根据训练状态进行不同的处理
        sequence_output = self.dropout(sequence_output, training=training)
        # 将dropout后的输出传入分类器获取logits
        logits = self.classifier(sequence_output)

        # 如果没有提供labels，则损失为None；否则计算token分类损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果不返回字典形式的输出，则按元组形式组织返回结果
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果返回字典形式的输出，则组织为TFTokenClassifierOutput对象
        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        # 标记模型已构建
        self.built = True
        # 如果存在Transformer模型，则构建Transformer模型
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        # 如果存在分类器模型，则构建分类器模型
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
@add_start_docstrings(
    """
    Flaubert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    FLAUBERT_START_DOCSTRING,
)
# 定义了一个新的 TensorFlow 模型类 TFFlaubertForMultipleChoice，继承自 TFFlaubertPreTrainedModel 和 TFMultipleChoiceLoss
class TFFlaubertForMultipleChoice(TFFlaubertPreTrainedModel, TFMultipleChoiceLoss):
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 创建 Flaubert 主层，并命名为 'transformer'
        self.transformer = TFFlaubertMainLayer(config, name="transformer")
        # 创建序列摘要层，使用 TFSequenceSummary 类，初始化范围为 config.init_std，命名为 'sequence_summary'
        self.sequence_summary = TFSequenceSummary(config, initializer_range=config.init_std, name="sequence_summary")
        # 创建 logits 投影层，使用 Dense 层，输出维度为 1，初始化器使用 config.initializer_range，命名为 'logits_proj'
        self.logits_proj = keras.layers.Dense(
            1, kernel_initializer=get_initializer(config.initializer_range), name="logits_proj"
        )
        # 设置实例变量 config 为传入的配置对象
        self.config = config

    @property
    def dummy_inputs(self):
        """
        Dummy inputs to build the network.

        Returns:
            tf.Tensor with dummy inputs
        """
        # 如果配置要求使用语言嵌入并且语言数量大于 1，则返回带有语言信息的输入字典
        if self.config.use_lang_emb and self.config.n_langs > 1:
            return {
                "input_ids": tf.constant(MULTIPLE_CHOICE_DUMMY_INPUTS, dtype=tf.int32),
                "langs": tf.constant(MULTIPLE_CHOICE_DUMMY_INPUTS, dtype=tf.int32),
            }
        else:
            # 否则，只返回包含 input_ids 的输入字典
            return {
                "input_ids": tf.constant(MULTIPLE_CHOICE_DUMMY_INPUTS, dtype=tf.int32),
            }

    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        FLAUBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
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
    # 定义方法的返回类型，可以是 TFMultipleChoiceModelOutput 或者包含 tf.Tensor 的元组
    -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
        # 如果存在 input_ids，则获取选择项数量和序列长度
        if input_ids is not None:
            num_choices = shape_list(input_ids)[1]
            seq_length = shape_list(input_ids)[2]
        else:
            # 否则，使用 inputs_embeds 获取选择项数量和序列长度
            num_choices = shape_list(inputs_embeds)[1]
            seq_length = shape_list(inputs_embeds)[2]

        # 将 input_ids 重塑为二维张量 (-1, seq_length)，如果 input_ids 不为 None
        flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
        # 将 attention_mask 重塑为二维张量 (-1, seq_length)，如果 attention_mask 不为 None
        flat_attention_mask = tf.reshape(attention_mask, (-1, seq_length)) if attention_mask is not None else None
        # 将 token_type_ids 重塑为二维张量 (-1, seq_length)，如果 token_type_ids 不为 None
        flat_token_type_ids = tf.reshape(token_type_ids, (-1, seq_length)) if token_type_ids is not None else None
        # 将 position_ids 重塑为二维张量 (-1, seq_length)，如果 position_ids 不为 None
        flat_position_ids = tf.reshape(position_ids, (-1, seq_length)) if position_ids is not None else None
        # 将 langs 重塑为二维张量 (-1, seq_length)，如果 langs 不为 None
        flat_langs = tf.reshape(langs, (-1, seq_length)) if langs is not None else None
        # 将 inputs_embeds 重塑为三维张量 (-1, seq_length, shape_list(inputs_embeds)[3])，如果 inputs_embeds 不为 None
        flat_inputs_embeds = (
            tf.reshape(inputs_embeds, (-1, seq_length, shape_list(inputs_embeds)[3]))
            if inputs_embeds is not None
            else None
        )

        # 如果 lengths 不为 None，则发出警告并将其设为 None，因为 Flaubert 多选模型不能使用 lengths 参数，应使用 attention mask
        if lengths is not None:
            logger.warning(
                "The `lengths` parameter cannot be used with the Flaubert multiple choice models. Please use the "
                "attention mask instead.",
            )
            lengths = None

        # 调用 transformer 方法，传递所有扁平化后的输入，获取 transformer 的输出
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
        # 从 transformer_outputs 中获取第一个输出，即 transformer 的输出
        output = transformer_outputs[0]
        # 使用 sequence_summary 方法对输出进行汇总
        logits = self.sequence_summary(output)
        # 使用 logits_proj 方法对 logits 进行投影
        logits = self.logits_proj(logits)
        # 将 logits 重塑为二维张量 (-1, num_choices)
        reshaped_logits = tf.reshape(logits, (-1, num_choices))

        # 如果 labels 为 None，则损失为 None；否则使用 hf_compute_loss 方法计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, reshaped_logits)

        # 如果不返回字典形式的结果，则将输出重塑为预期的格式
        if not return_dict:
            output = (reshaped_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TFMultipleChoiceModelOutput 对象，包含损失、logits、隐藏状态和注意力权重
        return TFMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
    # 如果模型已经构建完成，则直接返回，不进行重复构建
    if self.built:
        return
    # 将模型标记为已构建状态
    self.built = True

    # 如果存在 transformer 属性，对 transformer 进行构建
    if getattr(self, "transformer", None) is not None:
        # 在 TensorFlow 中为 transformer 创建命名空间，确保命名唯一性
        with tf.name_scope(self.transformer.name):
            self.transformer.build(None)

    # 如果存在 sequence_summary 属性，对 sequence_summary 进行构建
    if getattr(self, "sequence_summary", None) is not None:
        # 在 TensorFlow 中为 sequence_summary 创建命名空间，确保命名唯一性
        with tf.name_scope(self.sequence_summary.name):
            self.sequence_summary.build(None)

    # 如果存在 logits_proj 属性，对 logits_proj 进行构建
    if getattr(self, "logits_proj", None) is not None:
        # 在 TensorFlow 中为 logits_proj 创建命名空间，确保命名唯一性
        with tf.name_scope(self.logits_proj.name):
            # 构建 logits_proj，指定输入形状为 [None, None, self.config.num_labels]
            self.logits_proj.build([None, None, self.config.num_labels])
```