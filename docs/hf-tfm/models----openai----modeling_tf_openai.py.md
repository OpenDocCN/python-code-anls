# `.\transformers\models\openai\modeling_tf_openai.py`

```py
# 设置编码格式为 utf-8
# 版权声明，包括作者和团队信息
# 遵循 Apache 许可证版本 2.0
# 导入必要的模块和库
# 以下是 OpenAI GPT 模型的 TensorFlow 2.0 实现

from __future__ import annotations  # 对未来版本的注释进行标记，使其可以在 Python 3 中正确显示
from dataclasses import dataclass  # 导入 dataclass 类型
from typing import Optional, Tuple, Union  # 导入类型提示所需的类和函数

import numpy as np  # 导入 NumPy 库，并命名为 np
import tensorflow as tf  # 导入 TensorFlow 库，并命名为 tf
from ...activations_tf import get_tf_activation  # 从指定路径导入 get_tf_activation 函数
from ...modeling_tf_outputs import TFBaseModelOutput, TFCausalLMOutput, TFSequenceClassifierOutput  # 从指定路径导入模型输出相关的类
from ...modeling_tf_utils import (  # 从指定路径中导入模型相关的工具和便捷函数
    TFCausalLanguageModelingLoss,
    TFConv1D,
    TFModelInputType,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    TFSequenceSummary,
    TFSharedEmbeddings,
    get_initializer,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax  # 从指定路径导入 TensorFlow 相关的工具函数
from ...utils import (  # 从指定路径导入其他相关工具函数
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_openai import OpenAIGPTConfig  # 从指定路径导入 OpenAI GPT 模型的配置类

logger = logging.get_logger(__name__)  # 使用指定路径下的 logging 模块创建 logger

_CHECKPOINT_FOR_DOC = "openai-gpt"  # 用于文档的检查点标识
_CONFIG_FOR_DOC = "OpenAIGPTConfig"  # 用于文档的配置标识

TF_OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST = [  # OpenAI GPT 预训练模型的存档列表
    "openai-gpt",
    # 查看所有 OpenAI GPT 模型，请访问 https://huggingface.co/models?filter=openai-gpt
]

# 定义一个 TensorFlow 层类 Attention
class TFAttention(tf.keras.layers.Layer):
    def __init__(self, nx, config, scale=False, **kwargs):  # 初始化函数，包含输入参数和配置信息
        super().__init__(**kwargs)  # 调用父类的初始化函数

        n_state = nx  # 在注意力机制中，隐藏状态的维度等于 nx
        # [在 Attention 中交换 nx 和 n_state 的顺序，以保持与 TF 实现的一致性]
        assert (n_state % config.n_head == 0), f"隐藏维度 {n_state} 不能被注意力头数 {config.n_head} 整除"  # 断言隐藏维度必须可以被注意力头数整除
        self.n_head = config.n_head  # 注意力头的数量
        self.split_size = n_state  # 分割��维度大小
        self.scale = scale  # 是否进行缩放
        self.output_attentions = config.output_attentions  # 是否输出注意力分布

        self.c_attn = TFConv1D(n_state * 3, nx, initializer_range=config.initializer_range, name="c_attn")  # 创建卷积层 c_attn，用于计算注意力权重
        self.c_proj = TFConv1D(n_state, nx, initializer_range=config.initializer_range, name="c_proj")  # 创建卷积层 c_proj，用于计算最终输出
        self.attn_dropout = tf.keras.layers.Dropout(config.attn_pdrop)  # 创建注意力机制的 dropout 层
        self.resid_dropout = tf.keras.layers.Dropout(config.resid_pdrop)  # 创建残差连接的 dropout 层
        self.n_state = n_state  # 隐藏状态的维度
        self.pruned_heads = set()  # 被剪枝的注意力头集合

    def prune_heads(self, heads):  # 剪枝注意力头
        pass

    @staticmethod
    def causal_attention_mask(nd, ns):
        """
        1's in the lower triangle, counting from the lower right corner. Same as tf.matrix_band_part(tf.ones([nd, ns]),
        -1, ns-nd), but doesn't produce garbage on TPUs.
        """
        # 生成一个 causal attention mask，生成 1 的区域在从右下角往上的对角线以下
        i = tf.range(nd)[:, None]
        j = tf.range(ns)
        m = i >= j - ns + nd
        return m

    def _attn(self, q, k, v, attention_mask, head_mask, output_attentions, training=False):
        # q, k, v have shape [batch, heads, sequence, features]
        # 计算 query 与 key 的乘积
        w = tf.matmul(q, k, transpose_b=True)
        if self.scale:
            dk = tf.cast(shape_list(k)[-1], dtype=w.dtype)  # scale attention_scores
            # 对注意力矩阵进行缩放
            w = w / tf.math.sqrt(dk)

        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        # 创建 causal attention mask
        b = tf.cast(self.causal_attention_mask(nd, ns), dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w * b - 1e4 * (1 - b)

        if attention_mask is not None:
            # 加上额外的 attention mask
            attention_mask = tf.cast(attention_mask, dtype=w.dtype)
            w = w + attention_mask

        # 对 w 进行 softmax 操作
        w = stable_softmax(w, axis=-1)
        w = self.attn_dropout(w, training=training)

        # 如果指定了 head_mask，则对 heads 进行 Mask
        if head_mask is not None:
            w = w * head_mask

        outputs = [tf.matmul(w, v)]
        if output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = tf.transpose(x, [0, 2, 1, 3])
        x_shape = shape_list(x)
        new_x_shape = x_shape[:-2] + [x_shape[-2] * x_shape[-1]]
        return tf.reshape(x, new_x_shape)

    def split_heads(self, x):
        x_shape = shape_list(x)
        new_x_shape = x_shape[:-1] + [self.n_head, x_shape[-1] // self.n_head]
        x = tf.reshape(x, new_x_shape)
        return tf.transpose(x, (0, 2, 1, 3))  # (batch, head, seq_length, head_features)

    def call(self, x, attention_mask, head_mask, output_attentions, training=False):
        # 调用 c_attn 函数
        x = self.c_attn(x)
        query, key, value = tf.split(x, 3, axis=2)
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        # 进行自注意力机制
        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions, training=training)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        # 经过全��接层输出
        a = self.c_proj(a)
        a = self.resid_dropout(a, training=training)

        outputs = [a] + attn_outputs[1:]
        return outputs  # a, (attentions)
    # 构建模型的函数
    def build(self, input_shape=None):
        # 如果已构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 检查是否有 c_attn 属性
        if getattr(self, "c_attn", None) is not None:
            # 使用 c_attn 的名字作为命名空间
            with tf.name_scope(self.c_attn.name):
                # 调用 c_attn 的 build 方法，输入shape为 [None, None, self.n_state * 3]
                self.c_attn.build([None, None, self.n_state * 3])
        # 检查是否有 c_proj 属性        
        if getattr(self, "c_proj", None) is not None:
            # 使用 c_proj 的名字作为命名空间
            with tf.name_scope(self.c_proj.name):
                # 调用 c_proj 的 build 方法，输入shape为 [None, None, self.n_state]
                self.c_proj.build([None, None, self.n_state])
# 定义了一个基于 TensorFlow 的多层感知器（MLP）层
class TFMLP(tf.keras.layers.Layer):
    # 初始化函数，接受状态数量、配置和其他关键字参数
    def __init__(self, n_state, config, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 从配置中获取嵌入维度
        nx = config.n_embd
        # 创建一个一维卷积层用于 MLP 的第一层，初始化权重范围使用配置中的值，命名为"c_fc"
        self.c_fc = TFConv1D(n_state, nx, initializer_range=config.initializer_range, name="c_fc")
        # 创建一个一维卷积层用于 MLP 的第二层，初始化权重范围使用配置中的值，命名为"c_proj"
        self.c_proj = TFConv1D(nx, n_state, initializer_range=config.initializer_range, name="c_proj")
        # 获取 GELU 激活函数
        self.act = get_tf_activation("gelu")
        # 创建一个丢弃层，使用配置中的残差 dropout 率
        self.dropout = tf.keras.layers.Dropout(config.resid_pdrop)
        # 记录嵌入维度和状态数量
        self.nx = nx
        self.n_state = n_state

    # 前向传播函数，接受输入张量 x 和训练标志
    def call(self, x, training=False):
        # 使用 GELU 激活函数对 MLP 的第一层进行激活
        h = self.act(self.c_fc(x))
        # 对 MLP 的第二层进行投影
        h2 = self.c_proj(h)
        # 在训练模式下对投影层进行 dropout
        h2 = self.dropout(h2, training=training)
        # 返回 MLP 的输出
        return h2

    # 构建层，用于在第一次调用时创建层的权重
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在第一层卷积层，则构建第一层卷积层的权重
        if getattr(self, "c_fc", None) is not None:
            with tf.name_scope(self.c_fc.name):
                self.c_fc.build([None, None, self.n_state])
        # 如果存在第二层卷积层，则构建第二层卷积层的权重
        if getattr(self, "c_proj", None) is not None:
            with tf.name_scope(self.c_proj.name):
                self.c_proj.build([None, None, self.nx])


# 定义了一个基于 TensorFlow 的块层
class TFBlock(tf.keras.layers.Layer):
    # 初始化函数，接受配置和是否进行缩放的参数
    def __init__(self, config, scale=False, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 从配置中获取嵌入维度
        nx = config.n_embd
        # 创建一个注意力层
        self.attn = TFAttention(nx, config, scale, name="attn")
        # 创建一个层归一化层，用于 MLP 的第一层
        self.ln_1 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="ln_1")
        # 创建一个 MLP 层
        self.mlp = TFMLP(4 * nx, config, name="mlp")
        # 创建一个层归一化层，用于 MLP 的第二层
        self.ln_2 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="ln_2")
        # 记录嵌入维度
        self.nx = nx

    # 前向传播函数，接受输入张量 x、注意力掩码、头部掩码、是否输出注意力权重和训练标志
    def call(self, x, attention_mask, head_mask, output_attentions, training=False):
        # 使用注意力层处理输入，得到注意力输出和注意力权重
        output_attn = self.attn(x, attention_mask, head_mask, output_attentions, training=training)
        a = output_attn[0]  # output_attn: a, (attentions)

        # 对输入进行层归一化和残差连接
        n = self.ln_1(x + a)
        # 使用 MLP 处理归一化后的输入
        m = self.mlp(n, training=training)
        # 对 MLP 的输出进行层归一化和残差连接
        h = self.ln_2(n + m)

        # 将块的输出组成列表，包括 MLP 的输出和注意力权重
        outputs = [h] + output_attn[1:]
        return outputs  # x, (attentions)

    # 构建层，用于在第一次调用时创建层的权重
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在注意力层，则构建注意力层的权重
        if getattr(self, "attn", None) is not None:
            with tf.name_scope(self.attn.name):
                self.attn.build(None)
        # 如果存在第一层层归一化层，则构建第一层层归一化层的权重
        if getattr(self, "ln_1", None) is not None:
            with tf.name_scope(self.ln_1.name):
                self.ln_1.build([None, None, self.nx])
        # 如果存在 MLP 层，则构建 MLP 层的权重
        if getattr(self, "mlp", None) is not None:
            with tf.name_scope(self.mlp.name):
                self.mlp.build(None)
        # 如果存在第二层层归一化层，则构建第二层层归一化层的权重
        if getattr(self, "ln_2", None) is not None:
            with tf.name_scope(self.ln_2.name):
               
    # 初始化模型对象
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类初始化方法
        super().__init__(*inputs, **kwargs)

        # 将传入的配置信息赋值给对象
        self.config = config
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.return_dict = config.use_return_dict
        self.num_hidden_layers = config.n_layer
        self.n_embd = config.n_embd
        self.n_positions = config.n_positions
        self.initializer_range = config.initializer_range

        # 创建共享embedding层
        self.tokens_embed = TFSharedEmbeddings(
            config.vocab_size, config.n_embd, initializer_range=config.initializer_range, name="tokens_embed"
        )
        # 创建dropout层
        self.drop = tf.keras.layers.Dropout(config.embd_pdrop)
        # 创建多层Transformer Block
        self.h = [TFBlock(config, scale=True, name=f"h_._{i}") for i in range(config.n_layer)]

    # 构建模型结构
    def build(self, input_shape=None):
        # 在计算图中创建位置embedding
        with tf.name_scope("positions_embed"):
            self.positions_embed = self.add_weight(
                name="embeddings",
                shape=[self.n_positions, self.n_embd],
                initializer=get_initializer(self.initializer_range),
            )

        # 如果已经构建过则直接返回
        if self.built:
            return
        self.built = True
        # 构建token embedding层和Transformer Block层
        if getattr(self, "tokens_embed", None) is not None:
            with tf.name_scope(self.tokens_embed.name):
                self.tokens_embed.build(None)
        if getattr(self, "h", None) is not None:
            for layer in self.h:
                with tf.name_scope(layer.name):
                    layer.build(None)

    # 获取输入embedding
    def get_input_embeddings(self):
        return self.tokens_embed

    # 设置输入embedding
    def set_input_embeddings(self, value):
        self.tokens_embed.weight = value
        self.tokens_embed.vocab_size = shape_list(value)[0]

    # 剪枝模型中的heads
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        raise NotImplementedError

    # 模型的前向传播方法
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
        training: Optional[bool] = False,
# 这是一个抽象类，用于处理权重初始化和简单的下载和加载预训练模型的接口
class TFOpenAIGPTPreTrainedModel(TFPreTrainedModel):
    # 配置类是 OpenAIGPTConfig
    config_class = OpenAIGPTConfig
    # 基础模型前缀是 "transformer"
    base_model_prefix = "transformer"

# 这是一个数据类，用于保存模型预测两个句子是否连续的输出
@dataclass
class TFOpenAIGPTDoubleHeadsModelOutput(ModelOutput):
    # 语言建模头的预测得分(每个词汇表token的得分)
    logits: tf.Tensor = None
    # 多项选择分类头的预测得分(每个选项的得分)
    mc_logits: tf.Tensor = None
    # 模型每一层的隐藏状态(可选输出)
    hidden_states: Tuple[tf.Tensor] | None = None
    # 模型每一层的注意力权重(可选输出)
    attentions: Tuple[tf.Tensor] | None = None

# 这是 OpenAI GPT 模型的起始文档字符串
OPENAI_GPT_START_DOCSTRING = r"""
    这个模型继承自 TFPreTrainedModel。它实现了库中所有模型的通用方法,比如下载、保存、调整输入嵌入、修剪头等。

    这个模型也是一个 tf.keras.Model 的子类。可以像使用常规的 TF 2.0 Keras 模型一样使用它,并参考 TF 2.0 文档了解一般用法和行为。

    TensorFlow 模型和层在 transformers 中接受两种格式的输入:
    1. 所有输入作为关键字参数(与 PyTorch 模型类似)
    2. 所有输入作为第一个位置参数的列表、元组或字典

    支持第二种格式是因为 Keras 方法更倾向于这种格式来传递输入到模型和层。由于这种支持,当使用像 model.fit() 这样的方法时,只需要以任何 model.fit() 支持的格式传递输入和标签即可。但是,如果您想使用第二种格式,则...
    # 参数配置
    
    **参数**：
    - config ([`OpenAIGPTConfig`]): 模型的配置类，包含模型的所有参数。
        - 使用配置文件进行初始化不会加载与模型关联的权重，只会加载配置。可以查看[`~PreTrainedModel.from_pretrained`]方法以加载模型的权重。
# 这是 OpenAI GPT 模型的输入文档字符串
OPENAI_GPT_INPUTS_DOCSTRING = r"""
"""

# 这是一个装饰器函数，用于向模型的文档添加一些前言
@add_start_docstrings(
    "The bare OpenAI GPT transformer model outputting raw hidden-states without any specific head on top.",
    OPENAI_GPT_START_DOCSTRING,
)
# TFOpenAIGPTModel 类是 OpenAI GPT 模型的 TensorFlow 实现
class TFOpenAIGPTModel(TFOpenAIGPTPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的构造函数，并将配置信息传递给父类
        super().__init__(config, *inputs, **kwargs)
        # 创建 OpenAI GPT 主要层
        self.transformer = TFOpenAIGPTMainLayer(config, name="transformer")

    # 这个装饰器用于解包输入参数
    @unpack_inputs
    # 这个装饰器用于为模型前向传播添加文档字符串
    @add_start_docstrings_to_model_forward(OPENAI_GPT_INPUTS_DOCSTRING)
    # 这个装饰器用于添加代码示例
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
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
        training: Optional[bool] = False,
    ) -> Union[Tuple, TFBaseModelOutput]:
        # 调用 TFOpenAIGPTMainLayer 的前向传播方法
        outputs = self.transformer(
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
        # 返回模型的输出
        return outputs

    # 定义模型的构建方法
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)

# 这个装饰器用于向模型的文档添加一些前言
@add_start_docstrings(
    """
    OpenAI GPT Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    OPENAI_GPT_START_DOCSTRING,
)
# TFOpenAIGPTLMHeadModel 类是带有语言建模头的 OpenAI GPT 模型的 TensorFlow 实现
class TFOpenAIGPTLMHeadModel(TFOpenAIGPTPreTrainedModel, TFCausalLanguageModelingLoss):
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的构造函数，并将配置信息传递给父类
        super().__init__(config, *inputs, **kwargs)
        # 创建 OpenAI GPT 主要层
        self.transformer = TFOpenAIGPTMainLayer(config, name="transformer")
        # OpenAIGPT 不支持过去的缓存特性
        self.supports_xla_generation = False

    # 返回输入嵌入作为输出嵌入
    def get_output_embeddings(self):
        return self.get_input_embeddings()

    # 设置输出嵌入等于输入嵌入
    def set_output_embeddings(self, value):
        self.set_input_embeddings(value)

    # 这个装饰器用于解包输入参数
    @unpack_inputs
    # 这个装饰器用于为模型前向传播添加文档字符串
    @add_start_docstrings_to_model_forward(OPENAI_GPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,  # 用于添加代码示例的文档字符串
        output_type=TFCausalLMOutput,  # 输出类型为 TFCausalLMOutput 类型
        config_class=_CONFIG_FOR_DOC,  # 用于添加代码示例的配置类
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 输入的 token IDs，类型为 TFModelInputType 或者 None
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力掩码，类型为 numpy 数组或 TensorFlow 张量，或者 None
        token_type_ids: np.ndarray | tf.Tensor | None = None,  # token 类型 IDs，类型为 numpy 数组或 TensorFlow 张量，或者 None
        position_ids: np.ndarray | tf.Tensor | None = None,  # 位置 IDs，类型为 numpy 数组或 TensorFlow 张量，或者 None
        head_mask: np.ndarray | tf.Tensor | None = None,  # 头部掩码，类型为 numpy 数组或 TensorFlow 张量，或者 None
        inputs_embeds: np.ndarray | tf.Tensor | None = None,  # 输入的嵌入向量，类型为 numpy 数组或 TensorFlow 张量，或者 None
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，默认为 None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，默认为 None
        return_dict: Optional[bool] = None,  # 是否返回字典，默认为 None
        labels: np.ndarray | tf.Tensor | None = None,  # 标签，类型为 numpy 数组或 TensorFlow 张量，或者 None
        training: Optional[bool] = False,  # 是否处于训练模式，默认为 False
    ) -> Union[Tuple, TFCausalLMOutput]:  # 返回类型可以是元组或 TFCausalLMOutput 类型的实例
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the cross entropy classification loss. Indices should be in `[0, ...,
            config.vocab_size - 1]`.
        """

        # 调用 transformer 模型，获取 transformer 输出
        transformer_outputs = self.transformer(
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
        # 获取 transformer 输出中的隐藏状态
        hidden_states = transformer_outputs[0]

        # 根据隐藏状态计算 logits
        logits = self.transformer.tokens_embed(hidden_states, mode="linear")

        loss = None
        if labels is not None:
            # 将标签向左移动一位，并且去掉最后一个 logits token
            shifted_logits = logits[:, :-1]
            labels = labels[:, 1:]
            # 计算损失
            loss = self.hf_compute_loss(labels, shifted_logits)

        if not return_dict:
            # 如果不返回字典，则返回元组
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果返回字典，则构建 TFCausalLMOutput 实例并返回
        return TFCausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def prepare_inputs_for_generation(self, inputs, **kwargs):
        # 为生成准备输入，返回输入字典
        return {"input_ids": inputs}

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
# 添加示例文档字符串到这个 OpenAI GPT 模型类的文档开始部分
@add_start_docstrings(
    """
    OpenAI GPT Model transformer with a language modeling and a multiple-choice classification head on top e.g. for
    RocStories/SWAG tasks. The two heads are two linear layers. The language modeling head has its weights tied to the
    input embeddings, the classification head takes as input the input of a specified classification token index in the
    input sequence).
    """,
    OPENAI_GPT_START_DOCSTRING,
)
# 定义这个 OpenAI GPT 模型的子类
class TFOpenAIGPTDoubleHeadsModel(TFOpenAIGPTPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 设置分类任务的标签数为 1
        config.num_labels = 1
        # 创建 OpenAI GPT 的主要层
        self.transformer = TFOpenAIGPTMainLayer(config, name="transformer")
        # 创建多个选择的分类头
        self.multiple_choice_head = TFSequenceSummary(
            config, initializer_range=config.initializer_range, name="multiple_choice_head"
        )

    # 解包输入参数
    @unpack_inputs
    # 添加示例输入文档字符串到模型前向传播方法的文档开始部分
    @add_start_docstrings_to_model_forward(OPENAI_GPT_INPUTS_DOCSTRING)
    # 替换返回值文档字符串为 TFOpenAIGPTDoubleHeadsModelOutput 和对应的配置类
    @replace_return_docstrings(output_type=TFOpenAIGPTDoubleHeadsModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        mc_token_ids: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ):
        # 这个模型的前向传播方法的实现

    # 定义这个模型的输入签名
    @property
    def input_signature(self):
        return {
            "input_ids": tf.TensorSpec((None, None, None), tf.int32, name="input_ids"),
            "attention_mask": tf.TensorSpec((None, None, None), tf.int32, name="attention_mask"),
            "mc_token_ids": tf.TensorSpec((None, None), tf.int32, name="token_type_ids"),
        }

    # 构建模型
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        if getattr(self, "multiple_choice_head", None) is not None:
            with tf.name_scope(self.multiple_choice_head.name):
                self.multiple_choice_head.build(None)

# 添加示例文档字符串到这个 OpenAI GPT 分类模型类的文档开始部分
@add_start_docstrings(
    """
    The OpenAI GPT Model transformer with a sequence classification head on top (linear layer).

    [`TFOpenAIGPTForSequenceClassification`] uses the last token in order to do the classification, as other causal
    models (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    """,
)
    # `pad_token_id` 是在配置中定义的，它可以找到每一行中不是填充标记的最后一个标记。
    # 如果没有定义 `pad_token_id`，它只会取每一批中每一行的最后一个值。
    # 因为当传递 `inputs_embeds` 而不是 `input_ids` 时，无法猜测填充标记，它也会做同样的事情（取每一批中每一行的最后一个值）。
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    OPENAI_GPT_START_DOCSTRING,
# 定义 TFOpenAIGPTForSequenceClassification 类，继承自 TFOpenAIGPTPreTrainedModel 和 TFSequenceClassificationLoss
class TFOpenAIGPTForSequenceClassification(TFOpenAIGPTPreTrainedModel, TFSequenceClassificationLoss):
    # 初始化方法
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 设置类属性 num_labels 为配置中的标签数
        self.num_labels = config.num_labels
        # 创建一个全连接层，用于计算分类分数
        self.score = tf.keras.layers.Dense(
            config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="score",
            use_bias=False,
        )
        # 创建一个 OpenAIGPT 主体层对象
        self.transformer = TFOpenAIGPTMainLayer(config, name="transformer")
        # 设置类属性为配置对象
        self.config = config

    # 调用方法
    @unpack_inputs
    # 在模型前向传播方法中添加起始的文档字符串
    @add_start_docstrings_to_model_forward(OPENAI_GPT_INPUTS_DOCSTRING)
    # 添加代码示例的文档字符串
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        # 输入参数
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
````
    ) -> Union[Tuple, TFSequenceClassifierOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the cross entropy classification loss. Indices should be in `[0, ...,
            config.vocab_size - 1]`.
        """
        # 使用 transformer 处理输入，返回包含各种输出的字典
        transformer_outputs = self.transformer(
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

        # 从 transformer 输出中获取隐藏状态
        hidden_states = transformer_outputs[0]
        # 使用评分层计算分类任务的逻辑回归
        logits = self.score(hidden_states)
        in_logits = None
        # 如果没有定义 pad_token_id，则将 sequence_lengths 设为 -1
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            # 如果输入了 input_ids，则计算 sequence_lengths
            if input_ids is not None:
                # 根据 pad_token_id 的位置计算序列长度
                sequence_lengths = (
                    tf.argmax(tf.cast(tf.math.equal(input_ids, self.config.pad_token_id), input_ids.dtype), axis=-1)
                    - 1
                )
                # 处理序列长度小于 0 的情况
                sequence_lengths = tf.where(sequence_lengths >= 0, sequence_lengths, input_ids.shape[-1] - 1)
                # 根据序列长度从 logits 中提取对应位置的输出
                in_logits = tf.gather(logits, sequence_lengths, batch_dims=1, axis=1)
            else:
                # 如果没有 input_ids，则将 sequence_lengths 设为 -1，并发出警告
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )
        loss = None

        # 如果提供了 labels，则计算损失
        if labels is not None:
            # 获取批次大小和序列长度
            if input_ids is not None:
                batch_size, sequence_length = shape_list(input_ids)[:2]
            else:
                batch_size, sequence_length = shape_list(inputs_embeds)[:2]
            # 如果没有定义 pad_token_id，则批次大小必须为 1
            assert (
                self.config.pad_token_id is not None or batch_size == 1
            ), "Cannot handle batch sizes > 1 if no padding token is defined."

            # 如果 sequence_lengths 不是张量，则从 logits 中提取对应位置的输出
            if not tf.is_tensor(sequence_lengths):
                in_logits = logits[0:batch_size, sequence_lengths]

            # 使用损失计算函数计算损失
            loss = self.hf_compute_loss(tf.reshape(labels, [-1, 1]), tf.reshape(in_logits, [-1, self.num_labels]))

        # 如果 in_logits 不为 None，则使用其作为 pooled_logits，否则使用 logits
        pooled_logits = in_logits if in_logits is not None else logits

        # 如果不返回字典，则将输出组合成元组返回
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TFSequenceClassifierOutput 对象，包括损失、logits、隐藏状态和注意力
        return TFSequenceClassifierOutput(
            loss=loss,
            logits=pooled_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建过了，直接返回
        if self.built:
            return
        # 标记模型已经构建完成
        self.built = True
        # 如果模型有 score 属性
        if getattr(self, "score", None) is not None:
            # 创建 score 的名字域
            with tf.name_scope(self.score.name):
                # 构建 score 子模型
                self.score.build([None, None, self.config.n_embd])
        # 如果模型有 transformer 属性
        if getattr(self, "transformer", None) is not None:
            # 创建 transformer 的名字域
            with tf.name_scope(self.transformer.name):
                # 构建 transformer 子模型
                self.transformer.build(None)
```