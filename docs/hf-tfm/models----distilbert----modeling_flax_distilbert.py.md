# `.\models\distilbert\modeling_flax_distilbert.py`

```
# 导入所需的库和模块
import math
from typing import Callable, Optional, Tuple

import flax.linen as nn  # 导入flax中的linen模块作为nn别名
import jax  # 导入jax库
import jax.numpy as jnp  # 导入jax中的numpy模块作为jnp别名
import numpy as np  # 导入numpy库作为np别名
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze  # 从flax.core.frozen_dict模块导入FrozenDict、freeze、unfreeze函数
from flax.traverse_util import flatten_dict, unflatten_dict  # 从flax.traverse_util模块导入flatten_dict、unflatten_dict函数
from jax import lax  # 从jax模块导入lax模块

# 导入模型输出相关的类
from ...modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxMaskedLMOutput,
    FlaxMultipleChoiceModelOutput,
    FlaxQuestionAnsweringModelOutput,
    FlaxSequenceClassifierOutput,
    FlaxTokenClassifierOutput,
)
# 导入模型工具函数和常量
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring, overwrite_call_docstring
# 导入工具函数和常量
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
# 导入DistilBertConfig配置类
from .configuration_distilbert import DistilBertConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的模型检查点名称
_CHECKPOINT_FOR_DOC = "distilbert-base-uncased"
# 用于文档的模型配置名称
_CONFIG_FOR_DOC = "DistilBertConfig"

# DistilBERT模型的起始文档字符串
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

# DistilBERT模型输入文档字符串
DISTILBERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`numpy.ndarray` of shape `({0})`):
            # 输入序列中的token索引数组，索引对应词汇表中的token。

            # 可以使用`AutoTokenizer`获取这些索引。参见`PreTrainedTokenizer.encode`和`PreTrainedTokenizer.__call__`获取详细信息。

            # [什么是input IDs?](../glossary#input-ids)
        attention_mask (`numpy.ndarray` of shape `({0})`, *optional*):
            # 避免对填充token索引执行注意力计算的掩码。掩码的取值范围为`[0, 1]`：

            # - 1表示**不被掩盖**的token，
            # - 0表示**被掩盖**的token。

            # [什么是attention masks?](../glossary#attention-mask)
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。查看返回的张量中`attentions`获取更多细节。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。查看返回的张量中`hidden_states`获取更多细节。
        return_dict (`bool`, *optional*):
            # 是否返回`~utils.ModelOutput`而不是普通的元组。
"""
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates
"""
# 根据位置、索引和模型维度计算角度率，用于位置编码中的角度计算

"""
def positional_encoding(position, d_model):
    # create the sinusoidal pattern for the positional encoding
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return jnp.array(pos_encoding)
"""
# 根据位置和模型维度生成位置编码的正弦和余弦模式

class FlaxEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    config: DistilBertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.word_embeddings = nn.Embed(
            self.config.vocab_size,
            self.config.dim,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        if not self.config.sinusoidal_pos_embds:
            self.position_embeddings = nn.Embed(
                self.config.max_position_embeddings,
                self.config.dim,
                embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            )
        else:
            self.pos_encoding = positional_encoding(self.config.max_position_embeddings, self.config.dim)
        self.LayerNorm = nn.LayerNorm(epsilon=1e-12, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.dropout)

    def __call__(self, input_ids, deterministic: bool = True):
        # Embed
        batch_size, seq_length = input_ids.shape
        inputs_embeds = self.word_embeddings(input_ids.astype("i4"))
        if not self.config.sinusoidal_pos_embds:
            position_ids = jnp.arange(seq_length).astype("i4")
            position_ids = jnp.broadcast_to(position_ids, shape=(batch_size, seq_length))
            position_embeds = self.position_embeddings(position_ids.astype("i4"))
        else:
            position_embeds = self.pos_encoding[:, :seq_length, :]
            # explicitly cast the positions here, since self.embed_positions are not registered as parameters
            position_embeds = position_embeds.astype(inputs_embeds.dtype)

        # Sum all embeddings
        hidden_states = inputs_embeds + position_embeds

        # Layer Norm
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


class FlaxMultiHeadSelfAttention(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    def setup(self):
        self.n_heads = self.config.n_heads  # 从配置中获取注意力头的数量
        self.dim = self.config.dim  # 从配置中获取模型维度
        self.dropout = nn.Dropout(rate=self.config.attention_dropout)  # 根据配置设置注意力机制中的dropout

        if not (self.dim % self.n_heads == 0):
            raise ValueError(f"Hidden size {self.dim} not dividable by number of heads {self.n_heads}")  # 检查隐藏层大小是否可以被注意力头的数量整除

        self.q_lin = nn.Dense(
            self.dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )  # 初始化用于query的线性层，输入维度为dim，输出维度为dim

        self.k_lin = nn.Dense(
            self.dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )  # 初始化用于key的线性层，输入维度为dim，输出维度为dim

        self.v_lin = nn.Dense(
            self.dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )  # 初始化用于value的线性层，输入维度为dim，输出维度为dim

        self.out_lin = nn.Dense(
            self.dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )  # 初始化输出层线性层，输入维度为dim，输出维度为dim

    def __call__(
        self,
        query,
        key,
        value,
        mask,
        deterministic: bool = True,
        output_attentions: bool = False,
    ):
        bs, q_len, dim = query.shape  # 获取query的形状信息，bs为batch size，q_len为query的长度，dim为维度
        k_len = key.shape[1]  # 获取key的长度

        dim_per_head = self.dim // self.n_heads  # 计算每个注意力头的维度

        mask_reshp = (bs, 1, 1, k_len)  # 重塑mask的形状用于后续操作

        def shape(x):
            """分离头部"""
            return x.reshape(bs, -1, self.n_heads, dim_per_head).transpose(0, 2, 1, 3)  # 重塑张量x以分离注意力头

        def unshape(x):
            """合并头部"""
            return x.transpose(0, 2, 1, 3).reshape(bs, -1, self.n_heads * dim_per_head)  # 重塑张量x以合并注意力头

        q = shape(self.q_lin(query))  # 通过query的线性层进行形状分离，得到 (bs, n_heads, q_len, dim_per_head)
        k = shape(self.k_lin(key))  # 通过key的线性层进行形状分离，得到 (bs, n_heads, k_len, dim_per_head)
        v = shape(self.v_lin(value))  # 通过value的线性层进行形状分离，得到 (bs, n_heads, k_len, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # 对query进行缩放，以便更好地计算注意力权重 (bs, n_heads, q_len, dim_per_head)
        scores = jnp.matmul(q, k.transpose(0, 1, 3, 2))  # 计算注意力分数，形状为 (bs, n_heads, q_len, k_len)
        mask = jnp.reshape(mask, mask_reshp)  # 调整mask的形状以匹配注意力分数

        mask = mask.astype(scores.dtype)  # 将mask转换为与scores相同的数据类型
        scores = scores - 1e30 * (1.0 - mask)  # 将mask应用于scores，增加无效位置的大负数

        weights = nn.softmax(scores, axis=-1)  # 计算注意力权重，形状为 (bs, n_heads, q_len, k_len)
        weights = self.dropout(weights, deterministic=deterministic)  # 应用dropout到注意力权重

        context = jnp.matmul(weights, v)  # 计算上下文向量，形状为 (bs, n_heads, q_len, dim_per_head)
        context = unshape(context)  # 合并注意力头，形状为 (bs, q_len, dim)
        context = self.out_lin(context)  # 应用输出层线性层，形状为 (bs, q_len, dim)

        if output_attentions:
            return (context, weights)  # 如果需要输出注意力权重，返回上下文向量和权重
        else:
            return (context,)  # 否则只返回上下文向量
class FlaxFFN(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

    def setup(self):
        self.dropout = nn.Dropout(rate=self.config.dropout)  # 设置dropout层
        self.chunk_size_feed_forward = self.config.chunk_size_feed_forward  # 前馈层的块大小
        self.seq_len_dim = 1  # 序列长度维度为1
        self.lin1 = nn.Dense(
            self.config.hidden_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )  # 第一个全连接层，使用正态分布初始化权重

        self.lin2 = nn.Dense(
            self.config.dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )  # 第二个全连接层，使用正态分布初始化权重

        self.activation = ACT2FN[self.config.activation]  # 激活函数

    def __call__(self, hidden_states, deterministic: bool = True):
        hidden_states = self.lin1(hidden_states)  # 第一个全连接层的计算
        hidden_states = self.activation(hidden_states)  # 激活函数的应用
        hidden_states = self.lin2(hidden_states)  # 第二个全连接层的计算
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)  # dropout操作
        return hidden_states


class FlaxTransformerBlock(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

    def setup(self):
        assert (
            self.config.dim % self.config.n_heads == 0
        ), f"Hidden size {self.config.dim} not dividable by number of heads {self.config.n_heads}"  # 断言，确保隐藏大小可以被头数整除

        self.attention = FlaxMultiHeadSelfAttention(self.config, dtype=self.dtype)  # 多头自注意力机制
        self.sa_layer_norm = nn.LayerNorm(epsilon=1e-12, dtype=self.dtype)  # 自注意力层的LayerNorm

        self.ffn = FlaxFFN(self.config, dtype=self.dtype)  # 前馈网络
        self.output_layer_norm = nn.LayerNorm(epsilon=1e-12, dtype=self.dtype)  # 输出层的LayerNorm

    def __call__(
        self,
        hidden_states,
        attn_mask,
        output_attentions: bool = False,
        deterministic: bool = True,
    ):
        # 自注意力
        sa_output = self.attention(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            mask=attn_mask,
            output_attentions=output_attentions,
            deterministic=deterministic,
        )
        if output_attentions:
            sa_output, sa_weights = sa_output  # 如果需要输出注意力权重，则获取权重

        else:
            assert type(sa_output) == tuple
            sa_output = sa_output[0]  # 否则，获取自注意力的输出

        sa_output = self.sa_layer_norm(sa_output + hidden_states)  # 应用LayerNorm

        # 前馈网络
        ffn_output = self.ffn(sa_output, deterministic=deterministic)  # 前馈网络的计算
        ffn_output = self.output_layer_norm(ffn_output + sa_output)  # 应用LayerNorm
        output = (ffn_output,)  # 输出结果为元组

        if output_attentions:
            output = (sa_weights,) + output  # 如果需要输出注意力权重，则将权重添加到输出中

        return output


class FlaxTransformer(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

    def setup(self):
        self.layers = [
            FlaxTransformerBlock(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.n_layers)
        ]  # 创建多个TransformerBlock层的列表
    # 定义一个可调用的方法，用于执行模型的前向传播
    def __call__(
        self,
        hidden_states,
        attention_mask,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        deterministic: bool = True,
        return_dict: bool = False,
    ):
        # 如果输出隐藏状态，初始化存储所有隐藏状态的元组，否则为None
        all_hidden_states = () if output_hidden_states else None
        # 如果输出注意力权重，初始化存储所有注意力权重的元组，否则为None
        all_attentions = () if output_attentions else None

        # 遍历所有的层模块
        for layer_module in self.layers:
            # 如果需要输出隐藏状态，将当前的隐藏状态添加到所有隐藏状态的元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 调用当前层模块的前向传播方法，获取该层的输出
            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attn_mask=attention_mask,
                output_attentions=output_attentions,
                deterministic=deterministic,
            )
            # 更新隐藏状态为当前层的输出的最后一个值
            hidden_states = layer_outputs[-1]

            # 如果需要输出注意力权重
            if output_attentions:
                # 确保当前层的输出包含两个元素（注意力权重和其他）
                assert len(layer_outputs) == 2
                # 获取注意力权重，并添加到所有注意力权重的元组中
                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                # 确保当前层的输出只包含一个元素（隐藏状态）
                assert len(layer_outputs) == 1

        # 添加最后一层的隐藏状态到所有隐藏状态的元组中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要返回字典，则返回包含非None值的元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_attentions, all_hidden_states] if v is not None)
        # 如果需要返回字典，则创建并返回FlaxBaseModelOutput对象
        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )
class FlaxTransformerEncoder(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layer = FlaxTransformer(self.config, dtype=self.dtype)
        # 初始化 FlaxTransformer 层，使用给定的配置和数据类型

    def __call__(
        self,
        hidden_states,
        attention_mask,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        deterministic: bool = True,
        return_dict: bool = False,
    ):
        return self.layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            deterministic=deterministic,
            return_dict=return_dict,
        )
        # 调用 FlaxTransformer 层，传递输入参数并返回结果
        

class FlaxDistilBertLMDecoder(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    bias_init: Callable[..., np.ndarray] = jax.nn.initializers.zeros

    def setup(self):
        self.bias = self.param("bias", self.bias_init, (self.config.vocab_size,))
        # 初始化偏置参数 self.bias，大小为词汇表大小，使用 bias_init 初始化器

    def __call__(self, inputs, kernel):
        inputs = jnp.asarray(inputs, self.dtype)
        kernel = jnp.asarray(kernel, self.dtype)
        y = lax.dot_general(inputs, kernel, (((inputs.ndim - 1,), (0,)), ((), ())))
        # 执行高效的矩阵乘法操作，inputs 和 kernel 是输入张量
        bias = jnp.asarray(self.bias, self.dtype)
        y = y + bias
        # 将偏置加到输出 y 上
        return y


class FlaxDistilBertPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DistilBertConfig
    base_model_prefix = "distilbert"
    module_class: nn.Module = None

    def __init__(
        self,
        config: DistilBertConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 使用给定的配置和数据类型初始化模块
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化权重函数
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        # 创建输入张量和注意力掩码，使用默认值

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}
        # 分割随机数生成器以用于参数初始化和 dropout

        random_params = self.module.init(rngs, input_ids, attention_mask, return_dict=False)["params"]
        # 使用随机数初始化模块的参数

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
            # 如果有指定的参数，则将缺失的键补充为随机初始化的参数，并返回完整的参数字典
        else:
            return random_params
            # 否则，直接返回随机初始化的参数
    # 添加模型调用的前向传播文档字符串，描述输入参数为批大小和序列长度
    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 定义调用方法，接受多个输入参数用于模型推理
    def __call__(
        self,
        input_ids,  # 输入的token IDs序列
        attention_mask=None,  # 注意力掩码，指示哪些位置是有效的
        head_mask=None,  # 头掩码，控制不同的注意力头的掩码
        params: dict = None,  # 参数字典，用于加载模型参数
        dropout_rng: jax.random.PRNGKey = None,  # 随机数生成器密钥，用于Dropout操作
        train: bool = False,  # 指示是否为训练模式
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典格式的输出
    ):
        # 如果未提供attention_mask，则默认为全1，即所有位置都是有效的
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # 处理可能需要的任何随机数生成器
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        # 调用模型的apply方法进行前向传播
        return self.module.apply(
            {"params": params or self.params},  # 使用给定的参数或默认的模型参数
            jnp.array(input_ids, dtype="i4"),  # 转换输入token IDs为JAX数组
            jnp.array(attention_mask, dtype="i4"),  # 转换注意力掩码为JAX数组
            not train,  # 转换训练标志为相反值，用于控制模型是否在推理模式下运行
            output_attentions,  # 是否输出注意力权重
            output_hidden_states,  # 是否输出隐藏状态
            return_dict,  # 是否返回字典格式的输出
            rngs=rngs,  # 传递随机数生成器密钥到模型的apply方法中
        )
class FlaxDistilBertModule(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = jnp.float32  # 计算时使用的数据类型

    def setup(self):
        # 初始化嵌入层对象，使用给定的配置和数据类型
        self.embeddings = FlaxEmbeddings(self.config, dtype=self.dtype)
        # 初始化变换器编码器对象，使用给定的配置和数据类型
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
        # 如果输出注意力权重未指定，则使用配置中的默认设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果输出隐藏状态未指定，则使用配置中的默认设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果返回字典未指定，则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 获取输入的嵌入表示
        input_embeds = self.embeddings(input_ids, deterministic=deterministic)
        # 调用变换器编码器进行处理
        return self.transformer(
            hidden_states=input_embeds,
            attention_mask=attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


@add_start_docstrings(
    "输出原始隐藏状态的DistilBert模型变换器，没有特定的输出头部。",
    FLAX_DISTILBERT_START_DOCSTRING,
)
class FlaxDistilBertModel(FlaxDistilBertPreTrainedModel):
    module_class = FlaxDistilBertModule


append_call_sample_docstring(FlaxDistilBertModel, _CHECKPOINT_FOR_DOC, None, _CONFIG_FOR_DOC)


class FlaxDistilBertForMaskedLMModule(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = jnp.float32  # 计算时使用的数据类型

    def setup(self):
        # 初始化DistilBert模型对象，使用给定的配置和数据类型
        self.distilbert = FlaxDistilBertModule(self.config, dtype=self.dtype)
        # 初始化词汇变换层，使用给定的维度和正态分布的初始化方式
        self.vocab_transform = nn.Dense(
            self.config.dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        # 初始化词汇层归一化，设定epsilon为1e-12，使用给定的数据类型
        self.vocab_layer_norm = nn.LayerNorm(epsilon=1e-12, dtype=self.dtype)
        # 如果需要绑定词嵌入，则初始化DistilBert语言模型解码器
        if self.config.tie_word_embeddings:
            self.vocab_projector = FlaxDistilBertLMDecoder(
                self.config,
                dtype=self.dtype,
            )
        else:
            # 否则初始化普通的Dense层作为词汇投影器
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
        )
        # 如果 return_dict 为 None，则根据配置决定是否使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 DistilBERT 模型处理输入，获取输出
        dlbrt_output = self.distilbert(
            input_ids=input_ids,                   # 输入的 token IDs
            attention_mask=attention_mask,         # 注意力掩码
            output_attentions=output_attentions,   # 是否输出注意力权重
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
            deterministic=deterministic,           # 是否确定性运行
            return_dict=return_dict,               # 是否返回字典形式的输出
        )
        # 获取隐藏状态作为预测的 logits
        hidden_states = dlbrt_output[0]
        # 使用 vocab_transform 对隐藏状态进行转换得到预测 logits
        prediction_logits = self.vocab_transform(hidden_states)
        # 根据配置中的激活函数对 logits 进行激活
        prediction_logits = ACT2FN[self.config.activation](prediction_logits)
        # 对激活后的 logits 进行 layer normalization
        prediction_logits = self.vocab_layer_norm(prediction_logits)

        # 如果配置指定共享词嵌入，则使用 distilbert 中的词嵌入与 logits 进行投影
        if self.config.tie_word_embeddings:
            shared_embedding = self.distilbert.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
            prediction_logits = self.vocab_projector(prediction_logits, shared_embedding.T)
        else:
            prediction_logits = self.vocab_projector(prediction_logits)

        # 如果不需要以字典形式返回结果，则返回 logits 与其它输出
        if not return_dict:
            output = (prediction_logits,) + dlbrt_output[1:]  # 构建输出元组
            return output

        # 以 FlaxMaskedLMOutput 类型返回输出结果，包含 logits、隐藏状态和注意力权重
        return FlaxMaskedLMOutput(
            logits=prediction_logits,                   # 预测 logits
            hidden_states=dlbrt_output.hidden_states,   # 隐藏状态
            attentions=dlbrt_output.attentions,         # 注意力权重
        )
@add_start_docstrings("""DistilBert Model with a `language modeling` head on top.""", FLAX_DISTILBERT_START_DOCSTRING)
class FlaxDistilBertForMaskedLM(FlaxDistilBertPreTrainedModel):
    module_class = FlaxDistilBertForMaskedLMModule

- 定义了一个基于FlaxDistilBertPreTrainedModel的FlaxDistilBertForMaskedLM类，它具有一个`language modeling`头部。


append_call_sample_docstring(FlaxDistilBertForMaskedLM, _CHECKPOINT_FOR_DOC, FlaxMaskedLMOutput, _CONFIG_FOR_DOC)

- 调用append_call_sample_docstring函数，为FlaxDistilBertForMaskedLM类添加文档字符串示例。


class FlaxDistilBertForSequenceClassificationModule(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.distilbert = FlaxDistilBertModule(config=self.config, dtype=self.dtype)
        self.pre_classifier = nn.Dense(
            self.config.dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        self.dropout = nn.Dropout(rate=self.config.seq_classif_dropout)
        self.classifier = nn.Dense(
            self.config.num_labels,
            dtype=self.dtype,
        )

- 定义了一个FlaxDistilBertForSequenceClassificationModule类，继承自nn.Module，用于序列分类任务。在setup方法中初始化了DistilBERT模块、预分类器、Dropout和分类器。


    def __call__(
        self,
        input_ids,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

- 定义了__call__方法，实现了对输入数据进行处理和前向传播，支持不同的返回格式选项。


        distilbert_output = self.distilbert(
            input_ids,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

- 调用self.distilbert对输入进行处理，得到DistilBERT模型的输出。


        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = ACT2FN["relu"](pooled_output)
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)
        logits = self.classifier(pooled_output)  # (bs, dim)

- 对DistilBERT模型的输出进行处理，包括提取池化输出、通过预分类器和激活函数处理、应用Dropout、最终分类器得到logits。


        if not return_dict:
            return (logits,) + distilbert_output[1:]

        return FlaxSequenceClassifierOutput(
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )

- 根据return_dict的设置决定返回的结果格式，可以选择返回元组或者包含logits、隐藏状态和注意力的FlaxSequenceClassifierOutput对象。


@add_start_docstrings(
    """
    DistilBert Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    FLAX_DISTILBERT_START_DOCSTRING,
)
class FlaxDistilBertForSequenceClassification(FlaxDistilBertPreTrainedModel):
    module_class = FlaxDistilBertForSequenceClassificationModule

- 定义了一个FlaxDistilBertForSequenceClassification类，继承自FlaxDistilBertPreTrainedModel，具有序列分类/回归头部的DistilBERT模型。


append_call_sample_docstring(
    FlaxDistilBertForSequenceClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxSequenceClassifierOutput,
    _CONFIG_FOR_DOC,
)

- 调用append_call_sample_docstring函数，为FlaxDistilBertForSequenceClassification类添加文档字符串示例。


class FlaxDistilBertForMultipleChoiceModule(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = jnp.float32

- 定义了一个FlaxDistilBertForMultipleChoiceModule类，继承自nn.Module，用于多选题任务。
    # 初始化模型的各个组件，包括DistilBERT模块、预分类器、Dropout层和分类器
    def setup(self):
        self.distilbert = FlaxDistilBertModule(config=self.config, dtype=self.dtype)
        self.pre_classifier = nn.Dense(
            self.config.dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        self.dropout = nn.Dropout(rate=self.config.seq_classif_dropout)
        self.classifier = nn.Dense(
            1,
            dtype=self.dtype,
        )

    # 模型的调用方法，接收输入的token IDs和attention mask，并返回多项选择任务的结果
    def __call__(
        self,
        input_ids,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 根据参数设定是否使用配置中指定的返回字典方式
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 计算多项选择的数量
        num_choices = input_ids.shape[1]
        # 将输入的token IDs重新调整形状以便传递给模型
        input_ids = input_ids.reshape(-1, input_ids.shape[-1]) if input_ids is not None else None
        # 将输入的attention mask重新调整形状以便传递给模型
        attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1]) if attention_mask is not None else None

        # 使用DistilBERT模型处理输入，返回模型的输出
        outputs = self.distilbert(
            input_ids,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取模型的隐藏状态
        hidden_state = outputs[0]
        # 从隐藏状态中提取池化输出，一般是第一个位置的隐藏状态
        pooled_output = hidden_state[:, 0]
        # 通过预分类器处理池化输出
        pooled_output = self.pre_classifier(pooled_output)
        # 应用ReLU激活函数到处理后的池化输出
        pooled_output = ACT2FN["relu"](pooled_output)
        # 使用Dropout层对处理后的池化输出进行随机失活
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)
        # 使用分类器计算最终的logits
        logits = self.classifier(pooled_output)

        # 将logits重新调整形状以适应多项选择的格式
        reshaped_logits = logits.reshape(-1, num_choices)

        # 如果不使用返回字典的方式，则返回调整形状后的logits和额外的隐藏状态
        if not return_dict:
            return (reshaped_logits,) + outputs[2:]

        # 如果使用返回字典的方式，则返回FlaxMultipleChoiceModelOutput对象
        return FlaxMultipleChoiceModelOutput(
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    DistilBert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
    a softmax) e.g. for RocStories/SWAG tasks.
    """,
    FLAX_DISTILBERT_START_DOCSTRING,
)
class FlaxDistilBertForMultipleChoice(FlaxDistilBertPreTrainedModel):
    module_class = FlaxDistilBertForMultipleChoiceModule



overwrite_call_docstring(
    FlaxDistilBertForMultipleChoice, DISTILBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
)



append_call_sample_docstring(
    FlaxDistilBertForMultipleChoice,
    _CHECKPOINT_FOR_DOC,
    FlaxMultipleChoiceModelOutput,
    _CONFIG_FOR_DOC,
)



class FlaxDistilBertForTokenClassificationModule(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.distilbert = FlaxDistilBertModule(config=self.config, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.dropout)
        self.classifier = nn.Dense(self.config.num_labels, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Model
        outputs = self.distilbert(
            input_ids,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        logits = self.classifier(hidden_states)

        if not return_dict:
            return (logits,) + outputs[1:]

        return FlaxTokenClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



@add_start_docstrings(
    """
    DistilBert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    FLAX_DISTILBERT_START_DOCSTRING,
)
class FlaxDistilBertForTokenClassification(FlaxDistilBertPreTrainedModel):
    module_class = FlaxDistilBertForTokenClassificationModule



append_call_sample_docstring(
    FlaxDistilBertForTokenClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxTokenClassifierOutput,
    _CONFIG_FOR_DOC,
)



class FlaxDistilBertForQuestionAnsweringModule(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = jnp.float32
    # 初始化模型的方法，设置各个组件
    def setup(self):
        # 创建一个 DistilBERT 模型实例，使用给定的配置和数据类型
        self.distilbert = FlaxDistilBertModule(config=self.config, dtype=self.dtype)
        # 创建一个全连接层，用于输出问题回答的分类数目
        self.qa_outputs = nn.Dense(self.config.num_labels, dtype=self.dtype)
        # 断言模型需要输出的类别数为2
        assert self.config.num_labels == 2
        # 创建一个 Dropout 层，用于在训练过程中随机丢弃部分输入以防止过拟合
        self.dropout = nn.Dropout(rate=self.config.qa_dropout)

    # 模型调用方法，接受输入并返回模型预测结果
    def __call__(
        self,
        input_ids,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 根据参数设置是否返回字典形式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 DistilBERT 模型进行前向传播
        distilbert_output = self.distilbert(
            input_ids,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取模型输出中的隐藏状态
        hidden_states = distilbert_output[0]

        # 使用 Dropout 层对隐藏状态进行随机丢弃
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 将处理后的隐藏状态输入到全连接层中，得到最终的分类 logits
        logits = self.qa_outputs(hidden_states)
        # 将 logits 按照类别数目分割成起始和结束 logits
        start_logits, end_logits = logits.split(self.config.num_labels, axis=-1)
        # 去除不必要的维度
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # 根据是否需要返回字典形式的输出进行处理并返回
        if not return_dict:
            # 如果不返回字典，则返回元组形式的输出
            return (start_logits, end_logits) + distilbert_output[1:]

        # 返回 FlaxQuestionAnsweringModelOutput 类的实例，包含起始 logits、结束 logits、隐藏状态和注意力权重
        return FlaxQuestionAnsweringModelOutput(
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )
@add_start_docstrings(
    """
    DistilBert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
    linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    FLAX_DISTILBERT_START_DOCSTRING,
)


这部分代码是一个装饰器函数调用，用于给 `FlaxDistilBertForQuestionAnswering` 类添加文档字符串。文档字符串描述了该类的作用，说明它是基于 DistilBert 模型的，具有用于提取式问答任务（如 SQuAD）的分类头部（在隐藏状态输出的基础上进行线性层计算，生成 `span start logits` 和 `span end logits`）。


class FlaxDistilBertForQuestionAnswering(FlaxDistilBertPreTrainedModel):
    module_class = FlaxDistilBertForQuestionAnsweringModule


定义了一个新的类 `FlaxDistilBertForQuestionAnswering`，继承自 `FlaxDistilBertPreTrainedModel`。`module_class` 被设置为 `FlaxDistilBertForQuestionAnsweringModule`，用于模型内部的模块处理。


append_call_sample_docstring(
    FlaxDistilBertForQuestionAnswering,
    _CHECKPOINT_FOR_DOC,
    FlaxQuestionAnsweringModelOutput,
    _CONFIG_FOR_DOC,
)


这是一个函数调用，用于向 `FlaxDistilBertForQuestionAnswering` 类添加调用示例的文档字符串。它会附加一个关于模型如何调用的示例文档字符串，包括 `_CHECKPOINT_FOR_DOC`（用于模型检查点）、`FlaxQuestionAnsweringModelOutput`（模型输出）和 `_CONFIG_FOR_DOC`（模型配置）。
```