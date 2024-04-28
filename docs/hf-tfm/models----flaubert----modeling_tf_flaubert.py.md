# `.\models\flaubert\modeling_tf_flaubert.py`

```py
# 设置编码格式为 UTF-8
# 版权声明，遵循 Apache License 2.0 协议
# 引入相关 Python 库
from __future__ import annotations
import itertools
import random
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
from .configuration_flaubert import FlaubertConfig
    # 该部分为代码注释，提供了关于如何在 Keras 模型中传递输入的不同方式的说明
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

    # 注意，在创建模型和层时，可以使用不同的方式传递输入。如果在使用 Keras 的方法之外需要传递输入时，
    # 可以使用这些方式。这些方式包括传递单个张量、传递张量列表或传递张量字典。

    Note that when creating models and layers with
    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don't need to worry
    about any of this, as you can just pass inputs like you would to any other Python function!

    # 当使用子类化创建模型和层时，不需要担心这些，因为可以像传递给任何其他 Python 函数一样传递输入。

    </Tip>

    # Parameters 部分提供了函数的参数说明

    Parameters:
        config ([`FlaubertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
# 定义一个字符串变量，用于存储 Flaubert 模型输入的文档字符串
FLAUBERT_INPUTS_DOCSTRING = r"""
"""
# 定义一个函数，生成隐藏状态掩码和可选的注意力掩码
def get_masks(slen, lengths, causal, padding_mask=None):
    """
    生成隐藏状态掩码，并可选生成注意力掩码。
    """
    # 获取 batch size (bs)
    bs = shape_list(lengths)[0]
    # 如果提供了 padding_mask，则使用该掩码
    if padding_mask is not None:
        mask = padding_mask
    else:
        # 否则，创建一个与输入长度相同的范围（alen）
        alen = tf.range(slen, dtype=lengths.dtype)
        # 创建掩码，其中 alen 的元素小于长度的值
        mask = alen < tf.expand_dims(lengths, axis=1)

    # 注意力掩码与 mask 相同，或使用三角形下部注意力（因果的）
    if causal:
        # 生成因果注意力掩码
        attn_mask = tf.less_equal(
            tf.tile(tf.reshape(alen, (1, 1, slen)), (bs, slen, 1)), tf.reshape(alen, (1, slen, 1))
        )
    else:
        # 否则，注意力掩码与 mask 相同
        attn_mask = mask

    # 进行健全性检查，确保掩码形状与预期相符
    tf.debugging.assert_equal(shape_list(mask), [bs, slen])
    if causal:
        # 进行健全性检查，确保注意力掩码形状与预期相符
        tf.debugging.assert_equal(shape_list(attn_mask), [bs, slen, slen])

    # 返回 mask 和 attn_mask
    return mask, attn_mask


# 定义一个抽象类，用于处理权重初始化和预训练模型的下载和加载
class TFFlaubertPreTrainedModel(TFPreTrainedModel):
    """
    处理权重初始化和预训练模型下载和加载的抽象类。
    """

    # 指定配置类为 FlaubertConfig
    config_class = FlaubertConfig
    # 指定基础模型前缀为 "transformer"
    base_model_prefix = "transformer"

    # 定义属性，返回模拟输入
    @property
    def dummy_inputs(self):
        # 定义一个输入列表
        inputs_list = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]], dtype=tf.int32)
        # 定义一个注意力列表
        attns_list = tf.constant([[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [1, 0, 0, 1, 1]], dtype=tf.int32)
        # 如果模型使用语言嵌入并且支持多种语言
        if self.config.use_lang_emb and self.config.n_langs > 1:
            # 返回包含输入 ID、注意力掩码和语言的字典
            return {
                "input_ids": inputs_list,
                "attention_mask": attns_list,
                "langs": tf.constant([[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [1, 0, 0, 1, 1]], dtype=tf.int32),
            }
        else:
            # 否则，只返回包含输入 ID 和注意力掩码的字典
            return {"input_ids": inputs_list, "attention_mask": attns_list}


# 使用装饰器为模型类添加起始文档字符串
@add_start_docstrings(
    "The bare Flaubert Model transformer outputting raw hidden-states without any specific head on top.",
    FLAUBERT_START_DOCSTRING,
)
# 定义 Flaubert 模型类
class TFFlaubertModel(TFFlaubertPreTrainedModel):
    # 初始化模型
    def __init__(self, config, *inputs, **kwargs):
        # 使用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 创建 transformer 层
        self.transformer = TFFlaubertMainLayer(config, name="transformer")

    # 使用装饰器为模型前向传递方法添加文档字符串
    @unpack_inputs
    @add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义一个方法，用于调用模型
    def call(
        self,
        input_ids: np.ndarray | tf.Tensor | None = None,  # 输入的标识符，可以是 numpy 数组、张量或空值
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力掩码，可以是 numpy 数组、张量或空值
        langs: np.ndarray | tf.Tensor | None = None,  # 语言标识符，可以是 numpy 数组、张量或空值
        token_type_ids: np.ndarray | tf.Tensor | None = None,  # 令牌类型标识符，可以是 numpy 数组、张量或空值
        position_ids: np.ndarray | tf.Tensor | None = None,  # 位置标识符，可以是 numpy 数组、张量或空值
        lengths: np.ndarray | tf.Tensor | None = None,  # 长度，可以是 numpy 数组、张量或空值
        cache: Optional[Dict[str, tf.Tensor]] = None,  # 缓存，可以是空值或包含张量的字典
        head_mask: np.ndarray | tf.Tensor | None = None,  # 头部遮罩，可以是 numpy 数组、张量或空值
        inputs_embeds: tf.Tensor | None = None,  # 输入的嵌入向量，可以是张量或空值
        output_attentions: Optional[bool] = None,  # 输出注意力权重，可以是布尔类型或空值
        output_hidden_states: Optional[bool] = None,  # 输出隐藏状态，可以是布尔类型或空值
        return_dict: Optional[bool] = None,  # 是否返回字典形式的结果，可以是布尔类型或空值
        training: Optional[bool] = False,  # 是否训练模型，可以是布尔类型或空值，默认为 False
    ) -> Union[Tuple, TFBaseModelOutput]:  # 定义返回值的类型注解
        # 调用转换器模型，传入相应的参数
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
        # 返回模型输出
        return outputs

    # 定义一个方法，用于构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回
        if self.built:
            return
        # 设置模型为已构建
        self.built = True
        # 如果模型中存在转换器属性
        if getattr(self, "transformer", None) is not None:
            # 打开一个名叫 self.transformer.name 的命名空间，构建转换器模型
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
# 从 transformers.models.xlm.modeling_tf_xlm.TFXLMMultiHeadAttention 复制并修改为 Flaubert 模型的多头注意力机制层
class TFFlaubertMultiHeadAttention(tf.keras.layers.Layer):
    NEW_ID = itertools.count()  # 生成唯一标识符计数器

    def __init__(self, n_heads, dim, config, **kwargs):
        super().__init__(**kwargs)
        self.layer_id = next(TFFlaubertMultiHeadAttention.NEW_ID)  # 分配本层的唯一标识符
        self.dim = dim  # 注意力机制层输出维度
        self.n_heads = n_heads  # 头的数量
        self.output_attentions = config.output_attentions  # 是否输出注意力权重
        assert self.dim % self.n_heads == 0  # 确保输出维度可以被头的数量整除

        # 初始化查询、键、值、输出的全连接层
        self.q_lin = tf.keras.layers.Dense(dim, kernel_initializer=get_initializer(config.init_std), name="q_lin")
        self.k_lin = tf.keras.layers.Dense(dim, kernel_initializer=get_initializer(config.init_std), name="k_lin")
        self.v_lin = tf.keras.layers.Dense(dim, kernel_initializer=get_initializer(config.init_std), name="v_lin")
        self.out_lin = tf.keras.layers.Dense(dim, kernel_initializer=get_initializer(config.init_std), name="out_lin")
        self.dropout = tf.keras.layers.Dropout(config.attention_dropout)  # 注意力机制中的 dropout 层
        self.pruned_heads = set()  # 初始化被修剪的注意力头集合
        self.dim = dim  # 更新维度

    def prune_heads(self, heads):
        raise NotImplementedError  # 修剪注意力头的方法，未实现

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建查询、键、值、输出的全连接层
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


# 从 transformers.models.xlm.modeling_tf_xlm.TFXLMTransformerFFN 复制并修改为 Flaubert 模型的前馈神经网络层
class TFFlaubertTransformerFFN(tf.keras.layers.Layer):
    def __init__(self, in_dim, dim_hidden, out_dim, config, **kwargs):
        super().__init__(**kwargs)

        # 初始化前馈神经网络层的全连接层和激活函数
        self.lin1 = tf.keras.layers.Dense(dim_hidden, kernel_initializer=get_initializer(config.init_std), name="lin1")
        self.lin2 = tf.keras.layers.Dense(out_dim, kernel_initializer=get_initializer(config.init_std), name="lin2")
        self.act = get_tf_activation("gelu") if config.gelu_activation else get_tf_activation("relu")  # 获取激活函数
        self.dropout = tf.keras.layers.Dropout(config.dropout)  # 前馈神经网络中的 dropout 层
        self.in_dim = in_dim  # 输入维度
        self.dim_hidden = dim_hidden  # 隐藏层维度

    def call(self, input, training=False):
        x = self.lin1(input)  # 第一个全连接层
        x = self.act(x)  # 激活函数
        x = self.lin2(x)  # 第二个全连接层
        x = self.dropout(x, training=training)  # dropout 层

        return x  # 返回结果
    # 在构建模型时，如果已经构建过，则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 设置模型已经构建的标志为 True
        self.built = True
        # 如果存在 self.lin1 属性，则构建 self.lin1
        if getattr(self, "lin1", None) is not None:
            # 使用 TensorFlow 的命名空间为 self.lin1 构建模块
            with tf.name_scope(self.lin1.name):
                self.lin1.build([None, None, self.in_dim])
        # 如果存在 self.lin2 属性，则构建 self.lin2
        if getattr(self, "lin2", None) is not None:
            # 使用 TensorFlow 的命名空间为 self.lin2 构建模块
            with tf.name_scope(self.lin2.name):
                self.lin2.build([None, None, self.dim_hidden])
# 定义一个继承自 tf.keras.layers.Layer 的类 TFFlaubertMainLayer，用于表示 Flaubert 模型中的主要层
@keras_serializable
class TFFlaubertMainLayer(tf.keras.layers.Layer):
    # 定义一个属性 config_class，其值为 FlaubertConfig，用于表示该层使用的配置类
    config_class = FlaubertConfig

    # 初始化函数，接受 config 和其他参数
    def __init__(self, config, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 初始化该层的各种属性
        self.config = config
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
        # 使用 tf.keras.layers.Dropout 创建一个 Dropout 层，用于在模型中应用 dropout
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        # 使用 TFSharedEmbeddings 创建一个共享的 Embeddings 层
        self.embeddings = TFSharedEmbeddings(
            self.n_words, self.dim, initializer_range=config.embed_init_std, name="embeddings"
        )
        # 创建一个 LayerNormalization 层，用于对嵌入层进行归一化处理
        self.layer_norm_emb = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm_emb")
        self.attentions = []  # 用于存储多头注意力的层
        self.layer_norm1 = []  # 用于存储第一个层归一化层
        self.ffns = []  # 用于存储 feed forward 网络的层
        self.layer_norm2 = []  # 用于存储第二个层归一化层

        # 循环创建指定数量的多头注意力、归一化层、feed forward 网络和归一化层
        for i in range(self.n_layers):
            # 创建一个 TFFlaubertMultiHeadAttention 层
            self.attentions.append(
                TFFlaubertMultiHeadAttention(self.n_heads, self.dim, config=config, name=f"attentions_._{i}")
            )
            # 创建一个 LayerNormalization 层
            self.layer_norm1.append(
                tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name=f"layer_norm1_._{i}")
            )
            # 创建一个 TFFlaubertTransformerFFN 层
            self.ffns.append(
                TFFlaubertTransformerFFN(self.dim, self.hidden_dim, self.dim, config=config, name=f"ffns_._{i}")
            )
            # 创建一个 LayerNormalization 层
            self.layer_norm2.append(
                tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name=f"layer_norm2_._{i}")
            )
    # 构建模型，初始化各个变量
    def build(self, input_shape=None):
        # 添加一个“position_embeddings”的命名空间，用于组织相关操作
        with tf.name_scope("position_embeddings"):
            # 添加一个权重变量，名称为“embeddings”，形状为[self.max_position_embeddings, self.dim]，初始化方式为self.embed_init_std
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.dim],
                initializer=get_initializer(self.embed_init_std),
            )

        # 如果有多个语言并且使用语言嵌入，则继续执行以下代码块
        if self.n_langs > 1 and self.use_lang_emb:
            # 添加一个“lang_embeddings”的命名空间，用于组织相关操作
            with tf.name_scope("lang_embeddings"):
                # 添加一个权重变量，名称为“embeddings”，形状为[self.n_langs, self.dim]，初始化方式为self.embed_init_std
                self.lang_embeddings = self.add_weight(
                    name="embeddings",
                    shape=[self.n_langs, self.dim],
                    initializer=get_initializer(self.embed_init_std),
                )

        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 设置built为True，表示已经构建过
        self.built = True
        # 如果self.embeddings存在，则继续执行以下代码块
        if getattr(self, "embeddings", None) is not None:
            # 利用self.embeddings变量的名称作为命名空间，组织相关操作
            with tf.name_scope(self.embeddings.name):
                # 调用self.embeddings的build方法
                self.embeddings.build(None)
        # 如果self.layer_norm_emb存在，则继续执行以下代码块
        if getattr(self, "layer_norm_emb", None) is not None:
            # 利用self.layer_norm_emb变量的名称作为命名空间，组织相关操作
            with tf.name_scope(self.layer_norm_emb.name):
                # 调用self.layer_norm_emb的build方法，输入形状为[None, None, self.dim]
                self.layer_norm_emb.build([None, None, self.dim])
        # 遍历self.attentions列表中的每一个元素
        for layer in self.attentions:
            # 利用layer变量的名称作为命名空间，组织相关操作
            with tf.name_scope(layer.name):
                # 调用layer的build方法，输入形状为None
                layer.build(None)
        # 遍历self.layer_norm1列表中的每一个元素
        for layer in self.layer_norm1:
            # 利用layer变量的名称作为命名空间，组织相关操作
            with tf.name_scope(layer.name):
                # 调用layer的build方法，输���形状为[None, None, self.dim]
                layer.build([None, None, self.dim])
        # 遍历self.ffns列表中的每一个元素
        for layer in self.ffns:
            # 利用layer变量的名称作为命名空间，组织相关操作
            with tf.name_scope(layer.name):
                # 调用layer的build方法，输入形状为None
                layer.build(None)
        # 遍历self.layer_norm2列表中的每一个元素
        for layer in self.layer_norm2:
            # 利用layer变量的名称作为命名空间，组织相关操作
            with tf.name_scope(layer.name):
                # 调用layer的build方法，输入形状为[None, None, self.dim]
                layer.build([None, None, self.dim])

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.embeddings

    # 设置输入嵌入层
    def set_input_embeddings(self, value):
        # 更新self.embeddings的weight属性为value
        self.embeddings.weight = value
        # 更新self.embeddings的vocab_size属性为value的shape[0]

    # 调用函数
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
# 从transformers.models.xlm.modeling_tf_xlm.TFXLMPredLayer中复制了代码，定义了TFFlaubertPredLayer类
class TFFlaubertPredLayer(tf.keras.layers.Layer):
    """
    预测层（交叉熵或自适应softmax）。
    """

    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)

        # 是否采用自适应softmax
        self.asm = config.asm
        # 词汇表大小
        self.n_words = config.n_words
        # 填充索引
        self.pad_index = config.pad_index

        # 如果不使用自适应softmax，则使用输入嵌入
        if config.asm is False:
            self.input_embeddings = input_embeddings
        else:
            # 当asm为True时，抛出NotImplementedError异常
            raise NotImplementedError
            # self.proj = nn.AdaptiveLogSoftmaxWithLoss(
            #     in_features=dim,
            #     n_classes=config.n_words,
            #     cutoffs=config.asm_cutoffs,
            #     div_value=config.asm_div_value,
            #     head_bias=True,  # default is False
            # )

    def build(self, input_shape):
        # 输出权重与输入嵌入相同，但每个标记都有一个仅输出的偏置
        self.bias = self.add_weight(shape=(self.n_words,), initializer="zeros", trainable=True, name="bias")

        super().build(input_shape)

    def get_output_embeddings(self):
        # 返回输入嵌入
        return self.input_embeddings

    def set_output_embeddings(self, value):
        # 设置输出嵌入
        self.input_embeddings.weight = value
        self.input_embeddings.vocab_size = shape_list(value)[0]

    def get_bias(self):
        # 获取偏置
        return {"bias": self.bias}

    def set_bias(self, value):
        # 设置偏置
        self.bias = value["bias"]
        self.vocab_size = shape_list(value["bias"])[0]

    def call(self, hidden_states):
        # 将隐藏状态传递到输入嵌入，并进行线性变换
        hidden_states = self.input_embeddings(hidden_states, mode="linear")
        # 添加偏置
        hidden_states = hidden_states + self.bias

        return hidden_states


@dataclass
class TFFlaubertWithLMHeadModelOutput(ModelOutput):
    """
    [`TFFlaubertWithLMHeadModel`]输出的基类。

    参数:
        logits (`tf.Tensor`，形状为 `(batch_size, sequence_length, config.vocab_size)`):
            语言建模头的预测得分（SoftMax之前每个词汇标记的得分）。
        hidden_states (`tuple(tf.Tensor)`，*可选*，当传递`output_hidden_states=True`或`config.output_hidden_states=True`时返回):
            `tf.Tensor`元组（一个用于嵌入的输出 + 一个用于每个层的输出）的形状为 `(batch_size, sequence_length, hidden_size)`。

            模型在每一层输出的隐藏状态加上初始嵌入的输出。
        attentions (`tuple(tf.Tensor)`，*可选*，当传递`output_attentions=True`或`config.output_attentions=True`时返回):
            `tf.Tensor`元组（每一层一个）的形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            在注意力softmax之后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    logits: tf.Tensor = None
    # 定义hidden_states变量，类型为Tuple[tf.Tensor]，初始赋值为None
    hidden_states: Tuple[tf.Tensor] | None = None
    # 定义attentions变量，类型为Tuple[tf.Tensor]，初始赋值为None
    attentions: Tuple[tf.Tensor] | None = None
# 给 TFFlaubertWithLMHeadModel 类添加文档字符串，描述其作为带有语言建模头的 Flaubert 模型转换器
@add_start_docstrings(
    """
    The Flaubert Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    FLAUBERT_START_DOCSTRING,
)
class TFFlaubertWithLMHeadModel(TFFlaubertPreTrainedModel):

    # 初始化方法，接收配置和其他输入参数
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 创建 transformer 层，使用 TFFlaubertMainLayer 类
        self.transformer = TFFlaubertMainLayer(config, name="transformer")
        # 创建预测层，使用 TFFlaubertPredLayer 类，并且传入 transformer 的 embeddings 作为参数
        self.pred_layer = TFFlaubertPredLayer(config, self.transformer.embeddings, name="pred_layer_._proj")
        # Flaubert 模型不支持过去的缓存特性
        self.supports_xla_generation = False

    # 获取语言建模头
    def get_lm_head(self):
        return self.pred_layer

    # 获取前缀偏差的名称
    def get_prefix_bias_name(self):
        # 引发警告，此方法已经废弃，请使用 `get_bias` 替代
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        return self.name + "/" + self.pred_layer.name

    # 为生成准备输入
    def prepare_inputs_for_generation(self, inputs, **kwargs):
        mask_token_id = self.config.mask_token_id
        lang_id = self.config.lang_id

        # 获取有效批次大小
        effective_batch_size = inputs.shape[0]
        # 创建一个值全为 mask_token_id 的张量
        mask_token = tf.fill((effective_batch_size, 1), 1) * mask_token_id
        # 在输入后面拼接 mask_token
        inputs = tf.concat([inputs, mask_token], axis=1)

        if lang_id is not None:
            # 如果 lang_id 存在，则创建一个和 inputs 维度相同的张量，并且值全为 lang_id
            langs = tf.ones_like(inputs) * lang_id
        else:
            langs = None
        return {"input_ids": inputs, "langs": langs}

    # 调用方法，接收多个输入参数，具体参数类型可以是 np.ndarray 或 tf.Tensor，也可以是 None
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
    # 定义一个方法，接收一些输入并返回一个元组或TFFlaubertWithLMHeadModelOutput对象
    def call(
        self, 
        input_ids, 
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
        return_dict=True, 
        training=False
    ) -> Union[Tuple, TFFlaubertWithLMHeadModelOutput]:
        # 使用transformer处理输入并返回输出
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
        # 获取transformer输出的第一个元素
        output = transformer_outputs[0]
        # 使用pred_layer处理transformer的输出并返回结果
        outputs = self.pred_layer(output)
    
        # 如果不需要返回一个字典，就返回一个元组
        if not return_dict:
            return (outputs,) + transformer_outputs[1:]
        
        # 如果需要返回字典，就返回TFFlaubertWithLMHeadModelOutput对象
        return TFFlaubertWithLMHeadModelOutput(
            logits=outputs, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions
        )
    
    # 构建模型，设置参数input_shape=None
    def build(self, input_shape=None):
        # 如果已经构建过了，就直接返回
        if self.built:
            return
        # 如果没有构建过，就进行构建
        self.built = True
        # 如果存在transformer，就在命名空间下构建transformer
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        # 如果存在pred_layer，就在命名空间下构建pred_layer
        if getattr(self, "pred_layer", None) is not None:
            with tf.name_scope(self.pred_layer.name):
                self.pred_layer.build(None)
# 添加文档字符串说明Flaubert模型具有顺序分类/回归头（在汇总输出之上的线性层）
# 例如用于GLUE任务。
@add_start_docstrings(
    """
    Flaubert Model with a sequence classification/regression head on top (a linear layer on top of the pooled output)
    e.g. for GLUE tasks.
    """,
    FLAUBERT_START_DOCSTRING,
)
# 从transformers.models.xlm.modeling_tf_xlm中复制TFXLMForSequenceClassification代码并将XLM_INPUTS->FLAUBERT_INPUTS,XLM->Flaubert
class TFFlaubertForSequenceClassification(TFFlaubertPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        # 初始化Flaubert模型的transformer和sequence_summary
        self.transformer = TFFlaubertMainLayer(config, name="transformer")
        self.sequence_summary = TFSequenceSummary(config, initializer_range=config.init_std, name="sequence_summary")

    # 使用装饰器设置函数输入参数和文档字符串
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
            用于计算序列分类/回归损失的标签。索引应该在 `[0, ..., config.num_labels - 1]` 范围内。
            如果 `config.num_labels == 1`，则计算回归损失（均方损失），如果 `config.num_labels > 1`，则计算分类损失（交叉熵）。
        """
        # 将输入传递给transformer模型
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
        # 从transformer输出中获取第一个元素
        output = transformer_outputs[0]

        # 通过sequence_summary将输出转化为logits
        logits = self.sequence_summary(output)

        # 如果标签不为None，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果return_dict为False，则处理输出
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回TFSequenceClassifierOutput对象
        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    # 创建模型
    def build(self, input_shape=None):
        # 如果已经构建完毕，则直接返回
        if self.built:
            return
        self.built = True
        # 如果transformer存在，调用其build方法
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        # 如果sequence_summary存在，调用其build方法
        if getattr(self, "sequence_summary", None) is not None:
            with tf.name_scope(self.sequence_summary.name):
                self.sequence_summary.build(None)
# 使用额外的文档字符串装饰器，添加关于模型的介绍性文档字符串
# 复制自 transformers.models.xlm.modeling_tf_xlm.TFXLMForQuestionAnsweringSimple，将 XLM_INPUTS 替换为 FLAUBERT_INPUTS，将 XLM 替换为 Flaubert
class TFFlaubertForQuestionAnsweringSimple(TFFlaubertPreTrainedModel, TFQuestionAnsweringLoss):
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 初始化变量 transformer，使用 TFFlaubertMainLayer 创建 Flaubert 主层
        self.transformer = TFFlaubertMainLayer(config, name="transformer")
        # 初始化变量 qa_outputs，使用 Dense 层创建拥有 num_labels 个节点的输出层
        self.qa_outputs = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.init_std), name="qa_outputs"
        )
        # 存储配置信息
        self.config = config

    # 对输入参数进行解包，进行前向传播
    @unpack_inputs
    # 添加关于模型前向传播的文档字符串
    @add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 添加代码示例文档字符串
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
    ) -> Union[TFQuestionAnsweringModelOutput, Tuple[tf.Tensor]]:
        r"""
        `call`方法的签名，指定了输入和输出的类型。这里返回的类型可以是TFQuestionAnsweringModelOutput或者包含两个tf.Tensor的元组。
        """
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
        sequence_output = transformer_outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        loss = None
        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            loss = self.hf_compute_loss(labels, (start_logits, end_logits))
        
        # 如果返回字典的形式，则将损失与输出拼接后返回，否则仅返回输出
        if not return_dict:
            output = (start_logits, end_logits) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回包含损失、开始位置、结束位置、隐藏状态和注意力权重的TFQuestionAnsweringModelOutput对象
        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记该模型已经构建
        self.built = True
        # 构建transformer模型
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        # 构建qa_outputs模型
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])
# 使用指定的文档字符串来装饰 TFFlaubertForTokenClassification 类，描述了该模型的作用和用途
@add_start_docstrings(
    """
    Flaubert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    FLAUBERT_START_DOCSTRING,
)
# 从 transformers.models.xlm.modeling_tf_xlm.TFXLMForTokenClassification 复制代码并修改一些参数，以适配 Flaubert 模型
class TFFlaubertForTokenClassification(TFFlaubertPreTrainedModel, TFTokenClassificationLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 记录分类的类别数
        self.num_labels = config.num_labels

        # 使用 TFFlaubertMainLayer 构建 transformer 层
        self.transformer = TFFlaubertMainLayer(config, name="transformer")
        # 添加 dropout 层
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        # 添加分类器，使用全连接层，初始化方式为指定的初始化方法
        self.classifier = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.init_std), name="classifier"
        )
        self.config = config

    # 装饰 call 方法，添加输入参数的文档字符串
    @unpack_inputs
    @add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 添加示例代码文档字符串
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 模型的前向传播
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
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 使用transformer处理输入数据，并获取transformer的输出
        transformer_outputs = self.transformer(
            input_ids=input_ids,  # 输入的token id
            attention_mask=attention_mask,  # 注意力遮罩
            langs=langs,  # 语言编码
            token_type_ids=token_type_ids,  # token类型id
            position_ids=position_ids,  # 位置id
            lengths=lengths,  # 输入序列的长度
            cache=cache,  # 缓存
            head_mask=head_mask,  # 头部遮罩
            inputs_embeds=inputs_embeds,  # 输入嵌入
            output_attentions=output_attentions,  # 是否输出注意力
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
            return_dict=return_dict,  # 是否返回字典格式结果
            training=training,  # 是否为训练模式
        )
        # 获取transformer的输出的第一个元素作为序列输出
        sequence_output = transformer_outputs[0]

        # 对序列输出进行dropout
        sequence_output = self.dropout(sequence_output, training=training)
        # 对dropout后的序列输出进行分类得到logits
        logits = self.classifier(sequence_output)

        # 如果没有传入labels则loss为None，否则使用hf_compute_loss函数计算loss
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果不返回字典格式结果，将输出结果和loss组成元组返回，如果loss为None则不包含在返回结果中
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回字典格式的输出结果
        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建过网络则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在transformer则构建transformer
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        # 如果存在classifier则构建classifier
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
# 添加模型文档字符串作为注释
@add_start_docstrings(
    """
    Flaubert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    FLAUBERT_START_DOCSTRING,
)
# 从transformers.models.xlm.modeling_tf_xlm.TFXLMForMultipleChoice复制而来，将XLM_INPUTS->FLAUBERT_INPUTS,XLM->Flaubert
class TFFlaubertForMultipleChoice(TFFlaubertPreTrainedModel, TFMultipleChoiceLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 使用Flaubert的主要层创建transformer对象
        self.transformer = TFFlaubertMainLayer(config, name="transformer")
        # 使用config初始化范围构建TFSequenceSummary层
        self.sequence_summary = TFSequenceSummary(config, initializer_range=config.init_std, name="sequence_summary")
        # 构建Dense层作为logits_proj
        self.logits_proj = tf.keras.layers.Dense(
            1, kernel_initializer=get_initializer(config.initializer_range), name="logits_proj"
        )
        # 设置self.config为config
        self.config = config

    @property
    def dummy_inputs(self):
        """
        Dummy inputs to build the network.

        Returns:
            tf.Tensor with dummy inputs
        """
        # 有时Flaubert具有语言嵌入，如果需要，不要忘记构建它们
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
   # 定义函数，返回值为TFMultipleChoiceModelOutput或包含tf.Tensor的元组
   def forward(
       input_ids: tf.Tensor = None,
       attention_mask: tf.Tensor = None,
       langs: tf.Tensor = None,
       token_type_ids: tf.Tensor = None,
       position_ids: tf.Tensor = None,
       lengths: List[int] = None,
       cache: Optional[Dict[str, tf.Tensor]] = None,
       head_mask: Optional[tf.Tensor] = None,
       inputs_embeds: tf.Tensor = None,
       output_attentions: bool = False,
       output_hidden_states: bool = False,
       return_dict: bool = True,
       training: bool = False,
       labels: tf.Tensor = None,
   ) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
       # 如果input_ids不为空，则获取num_choices和seq_length
       if input_ids is not None:
           num_choices = shape_list(input_ids)[1]
           seq_length = shape_list(input_ids)[2]
       else:
           # 如果input_ids为空，则获取num_choices和seq_length
           num_choices = shape_list(inputs_embeds)[1]
           seq_length = shape_list(inputs_embeds)[2]
   
       # 对input_ids进行reshape，形状为(-1, seq_length)，如果input_ids为空则为None
       flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
       # 对attention_mask进行reshape，形状为(-1, seq_length)，如果attention_mask为空则为None
       flat_attention_mask = tf.reshape(attention_mask, (-1, seq_length)) if attention_mask is not None else None
       # 对token_type_ids进行reshape，形状为(-1, seq_length)，如果token_type_ids为空则为None
       flat_token_type_ids = tf.reshape(token_type_ids, (-1, seq_length)) if token_type_ids is not None else None
       # 对position_ids进行reshape，形状为(-1, seq_length)，如果position_ids为空则为None
       flat_position_ids = tf.reshape(position_ids, (-1, seq_length)) if position_ids is not None else None
       # 对langs进行reshape，形状为(-1, seq_length)，如果langs为空则为None
       flat_langs = tf.reshape(langs, (-1, seq_length)) if langs is not None else None
       # 对inputs_embeds进行reshape，形状为(-1, seq_length, shape_list(inputs_embeds)[3])，如果inputs_embeds为空则为None
       flat_inputs_embeds = (
           tf.reshape(inputs_embeds, (-1, seq_length, shape_list(inputs_embeds)[3]))
           if inputs_embeds is not None
           else None
       )
   
       # 如果lengths不为空，则发出警告提示不能与Flaubert多选模型一起使用长度参数，请使用attention_mask
       if lengths is not None:
           logger.warning(
               "The `lengths` parameter cannot be used with the Flaubert multiple choice models. Please use the "
               "attention mask instead.",
           )
           lengths = None
   
       # 调用transformer方法进行模型的前向传播
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
       # 获取transformer输出中的第一个元素作为output
       output = transformer_outputs[0]
       # 使用sequence_summary方法对output进行处理得到logits
       logits = self.sequence_summary(output)
       # 对logits进行处理得到最终输出结果
       logits = self.logits_proj(logits)
       # 对logits进行reshape，形状为(-1, num_choices)
       reshaped_logits = tf.reshape(logits, (-1, num_choices))
   
       # 根据labels是否为空判断是否计算损失
       loss = None if labels is None else self.hf_compute_loss(labels, reshaped_logits)
   
       # 如果return_dict为False，则将结果组装成元组返回
       if not return_dict:
           output = (reshaped_logits,) + transformer_outputs[1:]
           return ((loss,) + output) if loss is not None else output
   
       # 返回TFMultipleChoiceModelOutput对象
       return TFMultipleChoiceModelOutput(
           loss=loss,
           logits=reshaped_logits,
           hidden_states=transformer_outputs.hidden_states,
           attentions=transformer_outputs.attentions,
       )
    # 构建模型的方法，初始化模型的输入形状
    def build(self, input_shape=None):
        # 如果模型已经构建好了，直接返回，避免重复构建
        if self.built:
            return
        # 将标志位设置为已构建
        self.built = True
        # 如果存在 transformer 属性，构建 transformer
        if getattr(self, "transformer", None) is not None:
            # 使用 transformer 的名称作为命名空间
            with tf.name_scope(self.transformer.name):
                # 构建 transformer
                self.transformer.build(None)
        # 如果存在 sequence_summary 属性，构建 sequence_summary
        if getattr(self, "sequence_summary", None) is not None:
            # 使用 sequence_summary 的名称作为命名空间
            with tf.name_scope(self.sequence_summary.name):
                # 构建 sequence_summary
                self.sequence_summary.build(None)
        # 如果存在 logits_proj 属性，构建 logits_proj
        if getattr(self, "logits_proj", None) is not None:
            # 使用 logits_proj 的名称作为命名空间
            with tf.name_scope(self.logits_proj.name):
                # 构建 logits_proj，指定输入形状为 [None, None, self.config.num_labels]
                self.logits_proj.build([None, None, self.config.num_labels])
```