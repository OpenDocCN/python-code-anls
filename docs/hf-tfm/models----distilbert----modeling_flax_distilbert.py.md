# `.\models\distilbert\modeling_flax_distilbert.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 版权归属于 HuggingFace Inc. 团队，Google AI 语言团队和 Facebook, Inc.
#
# 根据 Apache 许可证 2.0 版对此文件进行授权;
# 您不得使用此文件，除非符合许可证的规定
# 可以在以下网址获得许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
#
# 除非依法要求或书面同意，否则软件按 "原样" 分发，
# 没有任何形式的担保或条款，无论是明示的还是默示的。
# 请查看有关特定语言的具体语法和限制的许可证。
import math
from typing import Callable, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax

from ...modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxMaskedLMOutput,
    FlaxMultipleChoiceModelOutput,
    FlaxQuestionAnsweringModelOutput,
    FlaxSequenceClassifierOutput,
    FlaxTokenClassifierOutput,
)
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring, overwrite_call_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_distilbert import DistilBertConfig

# 获取 logger 实例
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置
_CHECKPOINT_FOR_DOC = "distilbert-base-uncased"
_CONFIG_FOR_DOC = "DistilBertConfig"

# DistilBert 模型的开始文档字符串
FLAX_DISTILBERT_START_DOCSTRING = r"""

    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading, saving and converting weights from PyTorch models)

    This model is also a
    [flax.linen.Module](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html) subclass. Use it as
    a regular Flax linen Module and refer to the Flax documentation for all matter related to general usage and
    behavior.

    Finally, this model supports inherent JAX features such as:

    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    Parameters:
        config ([`DistilBertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# DistilBert 模型输入的文档字符串
DISTILBERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`numpy.ndarray` of shape `({0})`):
            # 输入序列标记在词汇表中的索引。
            # 可以使用 AutoTokenizer 获取索引。参见 PreTrainedTokenizer.encode 和 PreTrainedTokenizer.__call__ 了解详情。
            # 什么是输入 ID？
        attention_mask (`numpy.ndarray` of shape `({0})`, *optional*):
            # 避免在填充标记索引上执行注意力的掩码。掩码值选择在 [0, 1] 之间：
            # - 1 代表**未被掩码**的标记
            # - 0 代表**被掩码**的标记
            # 什么是注意力掩码？
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。查看返回张量中的 attentions 以了解更多细节。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。查看返回张量中的 hidden_states 以了解更多细节。
        return_dict (`bool`, *optional*):
            # 是否返回一个 `~utils.ModelOutput` 而不是普通的元组。
"""
# 定义一个函数，用于生成角度值数组
def get_angles(pos, i, d_model):
    # 根据位置、索引和模型维度计算角度比率
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

# 定义一个函数，用于生成位置编码
def positional_encoding(position, d_model):
    # 创建位置编码的正弦模式
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

    # 对数组中的偶数索引应用正弦函数；2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # 对数组中的奇数索引应用余弦函数；2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return jnp.array(pos_encoding)

# 定义一个FlaxEmbeddings类，用于构建word、position和token_type embeddings
class FlaxEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    config: DistilBertConfig
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

    def setup(self):
        # 初始化单词嵌入
        self.word_embeddings = nn.Embed(
            self.config.vocab_size,
            self.config.dim,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        # 如果不启用正弦位置编码，初始化位置嵌入
        if not self.config.sinusoidal_pos_embds:
            self.position_embeddings = nn.Embed(
                self.config.max_position_embeddings,
                self.config.dim,
                embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            )
        else:
            # 初始化位置编码
            self.pos_encoding = positional_encoding(self.config.max_position_embeddings, self.config.dim)
        self.LayerNorm = nn.LayerNorm(epsilon=1e-12, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.dropout)

    def __call__(self, input_ids, deterministic: bool = True):
        # 嵌入
        batch_size, seq_length = input_ids.shape
        inputs_embeds = self.word_embeddings(input_ids.astype("i4"))
        if not self.config.sinusoidal_pos_embds:
            position_ids = jnp.arange(seq_length).astype("i4")
            position_ids = jnp.broadcast_to(position_ids, shape=(batch_size, seq_length))
            position_embeds = self.position_embeddings(position_ids.astype("i4"))
        else:
            position_embeds = self.pos_encoding[:, :seq_length, :]
            # 在这里明确转换位置数据类型，因为self.embed_positions未注册为参数
            position_embeds = position_embeds.astype(inputs_embeds.dtype)

        # 汇总所有嵌入
        hidden_states = inputs_embeds + position_embeds

        # 层归一化
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states

# 定义一个FlaxMultiHeadSelfAttention类
class FlaxMultiHeadSelfAttention(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型
    # 初始化模型参数和配置项
    def setup(self):
        # 设置注意力头的数量
        self.n_heads = self.config.n_heads
        # 设置向量维度
        self.dim = self.config.dim
        # 设置 dropout 操作
        self.dropout = nn.Dropout(rate=self.config.attention_dropout)

        # 检查向量维度是否可以被注意力头的数量整除
        if not (self.dim % self.n_heads == 0):
            raise ValueError(f"Hidden size {self.dim} not dividable by number of heads {self.n_heads}")

        # 定义操作线性层，并初始化参数
        self.q_lin = nn.Dense(
            self.dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        self.k_lin = nn.Dense(
            self.dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        self.v_lin = nn.Dense(
            self.dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        self.out_lin = nn.Dense(
            self.dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )

    # 定义调用模型时的操作
    def __call__(
        self,
        query,
        key,
        value,
        mask,
        deterministic: bool = True,
        output_attentions: bool = False,
    ):
        bs, q_len, dim = query.shape
        k_len = key.shape[1]

        # 计算每个注意力头的维度
        dim_per_head = self.dim // self.n_heads

        mask_reshp = (bs, 1, 1, k_len)

        # 定义对向量进行分组和合并操作的函数
        def shape(x):
            """separate heads"""
            return x.reshape(bs, -1, self.n_heads, dim_per_head).transpose(0, 2, 1, 3)

        def unshape(x):
            """group heads"""
            return x.transpose(0, 2, 1, 3).reshape(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(query))  # (bs, n_heads, q_len, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_len, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_len, dim_per_head)

        # 除以 sqrt(dim_per_head) 以防止注意力计算过大
        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_len, dim_per_head)
        scores = jnp.matmul(q, k.transpose(0, 1, 3, 2))  # (bs, n_heads, q_len, k_len)
        mask = jnp.reshape(mask, mask_reshp)

        mask = mask.astype(scores.dtype)
        scores = scores - 1e30 * (1.0 - mask)

        # 计算注意力权重
        weights = nn.softmax(scores, axis=-1)  # (bs, n_heads, q_len, k_len)
        weights = self.dropout(weights, deterministic=deterministic)

        # 计算加权和后的 context
        context = jnp.matmul(weights, v)  # (bs, n_heads, q_len, dim_per_head)
        context = unshape(context)  # (bs, q_len, dim)
        context = self.out_lin(context)  # (bs, q_len, dim)

        # 若需要输出注意力权重，则将其一并返回
        if output_attentions:
            return (context, weights)
        else:
            return (context,)
class FlaxFFN(nn.Module):
    config: DistilBertConfig  # 类型声明，指定配置类
    dtype: jnp.dtype = jnp.float32  # 计算时的数据类型，默认为 jnp.float32

    def setup(self):
        self.dropout = nn.Dropout(rate=self.config.dropout)  # 初始化 dropout 层
        self.chunk_size_feed_forward = self.config.chunk_size_feed_forward  # 初始化前馈网络的块大小
        self.seq_len_dim = 1  # 序列长度维度为 1
        self.lin1 = nn.Dense(  # 第一个全连接层
            self.config.hidden_dim,  # 输出维度
            dtype=self.dtype,  # 数据类型
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),  # 权重初始化方法
        )
        self.lin2 = nn.Dense(  # 第二个全连接层
            self.config.dim,  # 输出维度
            dtype=self.dtype,  # 数据类型
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),  # 权重初始化方法
        )

        self.activation = ACT2FN[self.config.activation]  # 激活函数

    def __call__(self, hidden_states, deterministic: bool = True):
        hidden_states = self.lin1(hidden_states)  # 全连接层 1
        hidden_states = self.activation(hidden_states)  # 激活函数
        hidden_states = self.lin2(hidden_states)  # 全连接层 2
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)  # dropout
        return hidden_states  # 返回结果


class FlaxTransformerBlock(nn.Module):
    config: DistilBertConfig  # 类型声明，指定配置类
    dtype: jnp.dtype = jnp.float32  # 计算时的数据类型，默认为 jnp.float32

    def setup(self):
        assert (
            self.config.dim % self.config.n_heads == 0
        ), f"Hidden size {self.config.dim} not dividable by number of heads {self.config.n_heads}"  # 断言，确保隐藏大小可以被头数整除

        self.attention = FlaxMultiHeadSelfAttention(self.config, dtype=self.dtype)  # 多头自注意力层
        self.sa_layer_norm = nn.LayerNorm(epsilon=1e-12, dtype=self.dtype)  # 自注意力层的 LayerNorm 归一化层

        self.ffn = FlaxFFN(self.config, dtype=self.dtype)  # 前馈网络
        self.output_layer_norm = nn.LayerNorm(epsilon=1e-12, dtype=self.dtype)  # 前馈网络的 LayerNorm 归一化层

    def __call__(
        self,
        hidden_states,
        attn_mask,
        output_attentions: bool = False,
        deterministic: bool = True,
    ):
        # Self-Attention
        sa_output = self.attention(  # 自注意力操作
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            mask=attn_mask,
            output_attentions=output_attentions,
            deterministic=deterministic,
        )
        if output_attentions:
            sa_output, sa_weights = sa_output  # 如果需要输出注意力权重，则解包
        else:
            assert type(sa_output) == tuple  # 确保 sa_output 是一个元组
            sa_output = sa_output[0]  # 获取元组中的第一个元素
        sa_output = self.sa_layer_norm(sa_output + hidden_states)  # 执行自注意力层的 LayerNorm 归一化操作

        # Feed Forward Network
        ffn_output = self.ffn(sa_output, deterministic=deterministic)  # 前馈网络操作
        ffn_output = self.output_layer_norm(ffn_output + sa_output)  # 执行前馈网络的 LayerNorm 归一化操作
        output = (ffn_output,)  # 输出结果
        if output_attentions:
            output = (sa_weights,) + output  # 如果需要输出注意力权重，则加入到输出结果中
        return output  # 返回结果


class FlaxTransformer(nn.Module):
    config: DistilBertConfig  # 类型声明，指定配置类
    dtype: jnp.dtype = jnp.float32  # 计算时的数据类型，默认为 jnp.float32

    def setup(self):
        self.layers = [  # 初始化 Transformer 层列表
            FlaxTransformerBlock(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.n_layers)  # 遍历创建 Transformer 层
        ]
    # 定义一个调用方法，用于执行 Transformer 模型的前向传播
    def __call__(
        self,
        hidden_states,
        attention_mask,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        deterministic: bool = True,
        return_dict: bool = False,
    ):
        # 如果需要输出隐藏状态，则初始化一个空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力权重，则初始化一个空元组
        all_attentions = () if output_attentions else None

        # 遍历每一个 Transformer 层
        for layer_module in self.layers:
            # 如果需要输出隐藏状态，则将当前隐藏状态加入到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 调用 Transformer 层的前向传播方法
            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attn_mask=attention_mask,
                output_attentions=output_attentions,
                deterministic=deterministic,
            )
            # 更新隐藏状态为当前层的输出
            hidden_states = layer_outputs[-1]

            # 如果需要输出注意力权重
            if output_attentions:
                # 确保当前层输出包含注意力权重
                assert len(layer_outputs) == 2
                # 提取注意力权重
                attentions = layer_outputs[0]
                # 将当前层的注意力权重加入到 all_attentions 中
                all_attentions = all_attentions + (attentions,)
            else:
                # 确保当前层输出不包含注意力权重
                assert len(layer_outputs) == 1

        # 添加最后一层的隐藏状态到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要以字典形式返回结果
        if not return_dict:
            # 返回非空的结果元组
            return tuple(v for v in [hidden_states, all_attentions, all_hidden_states] if v is not None)
        # 以 FlaxBaseModelOutput 类型返回结果
        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )
# 定义一个名为FlaxTransformerEncoder的类，继承自nn.Module类
class FlaxTransformerEncoder(nn.Module):
    # 保存DistilBertConfig配置
    config: DistilBertConfig
    # 定义dtype变量，默认为jnp.float32，用于计算的数据类型

    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 设置函数，初始化自身的layer属性为一个FlaxTransformer对象
    def setup(self):
        self.layer = FlaxTransformer(self.config, dtype=self.dtype)

    # 定义__call__函数，接收一些输入参数并返回结果
    def __call__(
        self,
        hidden_states,
        attention_mask,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        deterministic: bool = True,
        return_dict: bool = False,
    ):
        # 调用self.layer对象，传入参数并返回结果
        return self.layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            deterministic=deterministic,
            return_dict=return_dict,
        )

# 定义一个名为FlaxDistilBertLMDecoder的类，继承自nn.Module类
class FlaxDistilBertLMDecoder(nn.Module):
    # 保存DistilBertConfig配置
    config: DistilBertConfig
    # 定义dtype变量，默认为jnp.float32，用于计算的数据类型

    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    # 定义bias_init为一个接收参数并返回np.ndarray的函数，初始值为jax.nn.initializers.zeros

    bias_init: Callable[..., np.ndarray] = jax.nn.initializers.zeros

    # 设置函数，初始化自身的bias属性为一个bias_init函数的结果，形状为(self.config.vocab_size,)
    def setup(self):
        self.bias = self.param("bias", self.bias_init, (self.config.vocab_size,))

    # 定义__call__函数，接收inputs和kernel两个参数，并返回结果
    def __call__(self, inputs, kernel):
        # 将inputs和kernel转换为dtype类型的jnp数组
        inputs = jnp.asarray(inputs, self.dtype)
        kernel = jnp.asarray(kernel, self.dtype)
        # 计算y的值，使用lax.dot_general进行矩阵乘法
        y = lax.dot_general(inputs, kernel, (((inputs.ndim - 1,), (0,)), ((), ())))
        # 转换bias为dtype类型的jnp数组，并将y与bias相加
        bias = jnp.asarray(self.bias, self.dtype)
        y = y + bias
        # 返回计算结果y
        return y

# 定义一个名为FlaxDistilBertPreTrainedModel的类，继承自FlaxPreTrainedModel类
class FlaxDistilBertPreTrainedModel(FlaxPreTrainedModel):
    # 一个处理权重初始化的抽象类，用于下载和加载预训练模型

    # 保存DistilBertConfig配置
    config_class = DistilBertConfig
    # 保存base_model_prefix为"distilbert"
    base_model_prefix = "distilbert"
    # 定义module_class为nn.Module，初始值为None

    module_class: nn.Module = None

    # 定义初始化函数，接收一些参数并返回结果
    def __init__(
        self,
        config: DistilBertConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 初始化module为module_class的结果，传入config和dtype等参数
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 调用父类的初始化函数，传入参数并返回结果
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    # 定义初始化权重的函数，接收rng、input_shape和params三个参数，返回FrozenDict结果
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化input_tensors，input_ids为全零数组，attention_mask为和input_ids相同形状的全1数组
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)

        # 使用jax.random.split函数拆分rng为params_rng和dropout_rng
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 使用self.module的init函数，传入rngs、input_ids、attention_mask和return_dict参数
        random_params = self.module.init(rngs, input_ids, attention_mask, return_dict=False)["params"]

        # 若params不为空，则将random_params和params合并，并返回结果；否则返回random_params
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params
    # 将文档字符串添加到模型的前向方法中
    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 定义模型的前向方法
    def __call__(
        self,
        input_ids,  # 输入的词索引
        attention_mask=None,  # 注意力遮罩，默认为None
        head_mask=None,  # 头遮罩，默认为None
        params: dict = None,  # 参数字典，默认为None
        dropout_rng: jax.random.PRNGKey = None,  # 随机数生成器，默认为None
        train: bool = False,  # 是否处于训练模式，默认为False
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，默认为None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏层状态，默认为None
        return_dict: Optional[bool] = None,  # 是否返回字典类型，默认为None
    ):
        # 如果未指定output_attentions，则使用配置文件中的值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定output_hidden_states，则使用配置文件中的值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定return_dict，则使用配置文件中的值
        return_dict = return_dict if return_dict is not None else self.config.return_dict
    
        # 如果attention_mask为None，则创建一个全1的注意力遮罩
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
    
        # 处理可能的随机数生成器
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng
    
        # 调用模型的apply方法，传入参数
        return self.module.apply(
            {"params": params or self.params},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            not train,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
        )
class FlaxDistilBertModule(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        # 初始化 Embeddings 层和 Transformer 编码器
        self.embeddings = FlaxEmbeddings(self.config, dtype=self.dtype)
        self.transformer = FlaxTransformerEncoder(self.config, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 根据参数设定是否输出注意力权重、隐藏状态、返回值字典
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 获取输入的嵌入向量
        input_embeds = self.embeddings(input_ids, deterministic=deterministic)
        # 调用 Transformer 编码器处理输入数据
        return self.transformer(
            hidden_states=input_embeds,
            attention_mask=attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

@add_start_docstrings(
    "The bare DistilBert Model transformer outputting raw hidden-states without any specific head on top.",
    FLAX_DISTILBERT_START_DOCSTRING,
)
class FlaxDistilBertModel(FlaxDistilBertPreTrainedModel):
    # 设定模块类为 FlaxDistilBertModule
    module_class = FlaxDistilBertModule

# 添加调用示例文档字符串
append_call_sample_docstring(FlaxDistilBertModel, _CHECKPOINT_FOR_DOC, None, _CONFIG_FOR_DOC)

class FlaxDistilBertForMaskedLMModule(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        # 初始化 DistilBERT 模型、词汇转换层和词汇归一化层
        self.distilbert = FlaxDistilBertModule(self.config, dtype=self.dtype)
        self.vocab_transform = nn.Dense(
            self.config.dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        self.vocab_layer_norm = nn.LayerNorm(epsilon=1e-12, dtype=self.dtype)
        # 根据是否共享词嵌入权重创建相应的词汇投影器
        if self.config.tie_word_embeddings:
            self.vocab_projector = FlaxDistilBertLMDecoder(
                self.config,
                dtype=self.dtype,
            )
        else:
            self.vocab_projector = nn.Dense(
                self.config.vocab_size,
                dtype=self.dtype,
                kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            )

    def __call__(
        self,
        input_ids,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        # 如果 return_dict 不是 None，则使用 return_dict；否则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 distilbert 模型进行推理，得到输出
        dlbrt_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            deterministic=deterministic,
            return_dict=return_dict,
        )

        # 从输出中取出隐藏状态
        hidden_states = dlbrt_output[0]

        # 通过预测原始信息得到预测的逻辑回归
        prediction_logits = self.vocab_transform(hidden_states)

        # 应用激活函数到预测的逻辑回归
        prediction_logits = ACT2FN[self.config.activation](prediction_logits)

        # 对预测的逻辑回归进行归一化
        prediction_logits = self.vocab_layer_norm(prediction_logits)

        # 如果配置中设置了共享词嵌入，则使用共享的词嵌入来投影预测逻辑回归
        if self.config.tie_word_embeddings:
            shared_embedding = self.distilbert.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
            prediction_logits = self.vocab_projector(prediction_logits, shared_embedding.T)
        else:
            prediction_logits = self.vocab_projector(prediction_logits)

        # 如果 return_dict 为 False，返回 output
        if not return_dict:
            output = (prediction_logits,) + dlbrt_output[1:]
            return output

        # 返回 FlaxMaskedLMOutput 形式的输出
        return FlaxMaskedLMOutput(
            logits=prediction_logits,
            hidden_states=dlbrt_output.hidden_states,
            attentions=dlbrt_output.attentions,
        )
# 为FlaxDistilBertForMaskedLM类添加文档字符串，描述其为在DistilBert模型上方带有`语言建模`头的模型
@add_start_docstrings("""DistilBert Model with a `language modeling` head on top.""", FLAX_DISTILBERT_START_DOCSTRING)
class FlaxDistilBertForMaskedLM(FlaxDistilBertPreTrainedModel):
    module_class = FlaxDistilBertForMaskedLMModule

# 为FlaxDistilBertForMaskedLM类添加调用示例文档字符串
append_call_sample_docstring(FlaxDistilBertForMaskedLM, _CHECKPOINT_FOR_DOC, FlaxMaskedLMOutput, _CONFIG_FOR_DOC)

# 定义FlaxDistilBertForSequenceClassificationModule类
class FlaxDistilBertForSequenceClassificationModule(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 初始化DistilBert模型
        self.distilbert = FlaxDistilBertModule(config=self.config, dtype=self.dtype)
        # 初始化预分类器
        self.pre_classifier = nn.Dense(
            self.config.dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        # 初始化Dropout层
        self.dropout = nn.Dropout(rate=self.config.seq_classif_dropout)
        # 初始化分类器
        self.classifier = nn.Dense(
            self.config.num_labels,
            dtype=self.dtype,
        )

    def __call__(
        self,
        input_ids,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 确定是否返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取DistilBert模型输出
        distilbert_output = self.distilbert(
            input_ids,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = ACT2FN["relu"](pooled_output)
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)
        logits = self.classifier(pooled_output)  # (bs, dim)

        if not return_dict:
            return (logits,) + distilbert_output[1:]

        return FlaxSequenceClassifierOutput(
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )

# 为FlaxDistilBertForSequenceClassification类添加文档字符串，描述其为在DistilBert模型上方带有序列分类/回归头的模型
@add_start_docstrings(
    """
    DistilBert Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    FLAX_DISTILBERT_START_DOCSTRING,
)
class FlaxDistilBertForSequenceClassification(FlaxDistilBertPreTrainedModel):
    module_class = FlaxDistilBertForSequenceClassificationModule

# 为FlaxDistilBertForSequenceClassification类添加调用示例文档字符串
append_call_sample_docstring(
    FlaxDistilBertForSequenceClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxSequenceClassifierOutput,
    _CONFIG_FOR_DOC,
)

# 定义FlaxDistilBertForMultipleChoiceModule类
class FlaxDistilBertForMultipleChoiceModule(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = jnp.float32
    # 初始化模型的各个组件
    def setup(self):
        # 初始化 DistilBERT 模型
        self.distilbert = FlaxDistilBertModule(config=self.config, dtype=self.dtype)
        # 初始化预分类器
        self.pre_classifier = nn.Dense(
            self.config.dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        # 初始化 Dropout 层
        self.dropout = nn.Dropout(rate=self.config.seq_classif_dropout)
        # 初始化分类器
        self.classifier = nn.Dense(
            1,
            dtype=self.dtype,
        )

    # 模型调用函数
    def __call__(
        self,
        input_ids,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 确定是否返回字典形式的结果
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取选择数量
        num_choices = input_ids.shape[1]
        # 重塑输入数据形状
        input_ids = input_ids.reshape(-1, input_ids.shape[-1]) if input_ids is not None else None
        attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1]) if attention_mask is not None else None

        # 模型前向传播
        outputs = self.distilbert(
            input_ids,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取隐藏状态和池化输出
        hidden_state = outputs[0]
        pooled_output = hidden_state[:, 0]
        # 预分类器处理池化输出
        pooled_output = self.pre_classifier(pooled_output)
        # 使用激活函数处理池化输出
        pooled_output = ACT2FN["relu"](pooled_output)
        # 使用 Dropout 处理池化输出
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)
        # 分类器处理池化输出得到最终输出
        logits = self.classifier(pooled_output)

        # 重塑输出形状
        reshaped_logits = logits.reshape(-1, num_choices)

        # 根据是否返回字典形式的结果进行返回
        if not return_dict:
            return (reshaped_logits,) + outputs[2:]

        return FlaxMultipleChoiceModelOutput(
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 定义一个带有多选分类头部的 DistilBert 模型（在池化输出的顶部有一个线性层和 softmax），例如用于 RocStories/SWAG 任务
@add_start_docstrings(
    """
    DistilBert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
    a softmax) e.g. for RocStories/SWAG tasks.
    """,
    FLAX_DISTILBERT_START_DOCSTRING,
)
class FlaxDistilBertForMultipleChoice(FlaxDistilBertPreTrainedModel):
    module_class = FlaxDistilBertForMultipleChoiceModule

# 覆盖调用文档字符串
overwrite_call_docstring(
    FlaxDistilBertForMultipleChoice, DISTILBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
)
# 追加调用示例文档字符串
append_call_sample_docstring(
    FlaxDistilBertForMultipleChoice,
    _CHECKPOINT_FOR_DOC,
    FlaxMultipleChoiceModelOutput,
    _CONFIG_FOR_DOC,
)

# 定义一个带有标记分类头部的 DistilBert 模型（在隐藏状态输出的顶部有一个线性层），例如用于命名实体识别（NER）任务
@add_start_docstrings(
    """
    DistilBert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    FLAX_DISTILBERT_START_DOCSTRING,
)
class FlaxDistilBertForTokenClassification(FlaxDistilBertPreTrainedModel):
    module_class = FlaxDistilBertForTokenClassificationModule

# 追加调用示例文档字符串
append_call_sample_docstring(
    FlaxDistilBertForTokenClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxTokenClassifierOutput,
    _CONFIG_FOR_DOC,
)

# 定义一个用于问答任务的 DistilBert 模型
class FlaxDistilBertForQuestionAnsweringModule(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = jnp.float32
    # 初始化方法，设置模型的各个组件
    def setup(self):
        # 初始化 DistilBERT 模型
        self.distilbert = FlaxDistilBertModule(config=self.config, dtype=self.dtype)
        # 初始化输出层
        self.qa_outputs = nn.Dense(self.config.num_labels, dtype=self.dtype)
        # 断言模型输出的标签数为2
        assert self.config.num_labels == 2
        # 初始化 Dropout 层
        self.dropout = nn.Dropout(rate=self.config.qa_dropout)

    # 模型调用方法
    def __call__(
        self,
        input_ids,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 如果 return_dict 为 None，则使用配置中的 use_return_dict 值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 模型前向传播
        distilbert_output = self.distilbert(
            input_ids,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取 DistilBERT 输出的隐藏状态
        hidden_states = distilbert_output[0]

        # 对隐藏状态应用 Dropout
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 通过输出层获取 logits
        logits = self.qa_outputs(hidden_states)
        # 将 logits 拆分为起始和结束 logits
        start_logits, end_logits = logits.split(self.config.num_labels, axis=-1)
        # 去除多余的维度
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # 如果不需要返回字典，则返回 logits 和 DistilBERT 输出的其他部分
        if not return_dict:
            return (start_logits, end_logits) + distilbert_output[1:]

        # 返回包含起始和结束 logits、隐藏状态和注意力权重的输出对象
        return FlaxQuestionAnsweringModelOutput(
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )
# 使用 DistilBert 模型，在顶部添加一个用于提取性问题回答任务（如 SQuAD）的跨度分类头（在隐藏状态输出之上的线性层，用于计算“跨度起始对数”和“跨度结束对数”）。
@add_start_docstrings(
    """
    DistilBert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
    linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    FLAX_DISTILBERT_START_DOCSTRING,
)
class FlaxDistilBertForQuestionAnswering(FlaxDistilBertPreTrainedModel):
    module_class = FlaxDistilBertForQuestionAnsweringModule

# 添加调用示例的文档字符串
append_call_sample_docstring(
    FlaxDistilBertForQuestionAnswering,
    _CHECKPOINT_FOR_DOC,
    FlaxQuestionAnsweringModelOutput,
    _CONFIG_FOR_DOC,
)
```