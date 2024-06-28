# `.\models\big_bird\modeling_flax_big_bird.py`

```
# 导入必要的模块和类
from typing import Callable, Optional, Tuple  # 导入类型提示相关模块

import flax  # 导入Flax框架
import flax.linen as nn  # 导入Flax的linen模块，用于定义神经网络模型
import jax  # 导入JAX，用于自动求导和并行计算
import jax.numpy as jnp  # 导入JAX的NumPy接口，命名为jnp，用于数组操作

# 从Flax的core.frozen_dict模块中导入FrozenDict、freeze、unfreeze等相关函数和类
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze  

# 从Flax的linen模块中导入combine_masks、make_causal_mask等函数和类，用于处理神经网络模型
from flax.linen import combine_masks, make_causal_mask  
from flax.linen import partitioning as nn_partitioning  # 导入linen.partitioning模块，用于模型分区
from flax.linen.attention import dot_product_attention_weights  # 导入dot_product_attention_weights函数，用于注意力机制权重计算
from flax.traverse_util import flatten_dict, unflatten_dict  # 导入flatten_dict和unflatten_dict函数，用于字典扁平化和反扁平化
from jax import lax  # 导入lax模块，用于定义JAX的低级API

# 导入特定的模型输出类和工具函数
from ...modeling_flax_outputs import (
    FlaxBaseModelOutputWithPastAndCrossAttentions,
    FlaxBaseModelOutputWithPooling,
    FlaxBaseModelOutputWithPoolingAndCrossAttentions,
    FlaxCausalLMOutputWithCrossAttentions,
    FlaxMaskedLMOutput,
    FlaxMultipleChoiceModelOutput,
    FlaxSequenceClassifierOutput,
    FlaxTokenClassifierOutput,
)

# 导入特定的模型基类和工具函数
from ...modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
    append_call_sample_docstring,
    append_replace_return_docstrings,
    overwrite_call_docstring,
)

# 导入通用工具函数和类
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging

# 从当前目录下的configuration_big_bird.py文件中导入BigBirdConfig类
from .configuration_big_bird import BigBirdConfig  

# 获取logger对象，用于记录日志信息
logger = logging.get_logger(__name__)

# 定义用于文档的检查点和配置变量
_CHECKPOINT_FOR_DOC = "google/bigbird-roberta-base"
_CONFIG_FOR_DOC = "BigBirdConfig"

# 定义并装饰remat函数，用于对神经网络模型进行分区重组
remat = nn_partitioning.remat

# 定义FlaxBigBirdForPreTrainingOutput类，继承自ModelOutput，用于BigBird预训练模型的输出类型
@flax.struct.dataclass
class FlaxBigBirdForPreTrainingOutput(ModelOutput):
    """
    Output type of [`BigBirdForPreTraining`].
    """
    # `prediction_logits` 是一个形状为 `(batch_size, sequence_length, config.vocab_size)` 的 NumPy 数组，
    # 包含语言建模头部的预测分数（在 SoftMax 之前的每个词汇标记的分数）。
    prediction_logits: jnp.ndarray = None
    
    # `seq_relationship_logits` 是一个形状为 `(batch_size, 2)` 的 NumPy 数组，
    # 包含下一个序列预测（分类）头部的预测分数（在 SoftMax 之前的 True/False 继续的分数）。
    seq_relationship_logits: jnp.ndarray = None
    
    # `hidden_states` 是一个可选的元组，当传递 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回。
    # 其中包含多个 `jnp.ndarray`（一个用于嵌入的输出 + 每个层的输出），
    # 形状为 `(batch_size, sequence_length, hidden_size)`。
    # 这些是模型在每个层输出的隐藏状态以及初始嵌入输出。
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    
    # `attentions` 是一个可选的元组，当传递 `output_attentions=True` 或 `config.output_attentions=True` 时返回。
    # 其中包含多个 `jnp.ndarray`（每个层一个），形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
    # 这些是经过注意力 SoftMax 后的注意力权重，用于计算自注意力头中的加权平均值。
    attentions: Optional[Tuple[jnp.ndarray]] = None
@flax.struct.dataclass
class FlaxBigBirdForQuestionAnsweringModelOutput(ModelOutput):
    """
    Base class for outputs of question answering models.

    Args:
        start_logits (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            Span-start scores (before SoftMax).
        end_logits (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            Span-end scores (before SoftMax).
        pooled_output (`jnp.ndarray` of shape `(batch_size, hidden_size)`):
            pooled_output returned by FlaxBigBirdModel.
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    start_logits: jnp.ndarray = None  # Span-start scores (before SoftMax) for question answering.
    end_logits: jnp.ndarray = None  # Span-end scores (before SoftMax) for question answering.
    pooled_output: jnp.ndarray = None  # Output pooled by FlaxBigBirdModel.
    hidden_states: Optional[Tuple[jnp.ndarray]] = None  # Hidden states of model layers and embeddings.
    attentions: Optional[Tuple[jnp.ndarray]] = None  # Attention weights for self-attention heads.


BIG_BIRD_START_DOCSTRING = r"""

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
"""
    # Parameters: 定义函数参数和其作用
    # config ([`BigBirdConfig`]): 模型配置类，包含模型的所有参数
    #     初始化配置文件不会加载与模型相关的权重，仅加载配置。
    #     若要加载模型权重，请查看 [`~FlaxPreTrainedModel.from_pretrained`] 方法。
    # dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
    #     计算的数据类型。可以是 `jax.numpy.float32`, `jax.numpy.float16` (在GPU上), `jax.numpy.bfloat16` (在TPU上) 之一。
    #
    #     可用于启用混合精度训练或在GPU或TPU上进行半精度推断。如果指定，则所有计算将使用给定的 `dtype` 进行。
    #
    #     **注意，这只是指定计算的数据类型，不影响模型参数的数据类型。**
    #
    #     如果希望更改模型参数的数据类型，请参阅 [`~FlaxPreTrainedModel.to_fp16`] 和 [`~FlaxPreTrainedModel.to_bf16`]。
"""

BIG_BIRD_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`numpy.ndarray` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`numpy.ndarray` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`numpy.ndarray` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`numpy.ndarray` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.
        head_mask (`numpy.ndarray` of shape `({0})`, `optional):
            Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

"""


class FlaxBigBirdEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # Copied from transformers.models.bert.modeling_flax_bert.FlaxBertEmbeddings.setup

    # 此处定义了一个FlaxBigBirdEmbeddings类，用于构建从词嵌入、位置嵌入和token_type嵌入构成的嵌入向量。
    # 初始化模型的各种嵌入层和正则化层
    def setup(self):
        # 初始化词嵌入层，将词汇表大小、隐藏层大小等作为参数传入
        self.word_embeddings = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        # 初始化位置嵌入层，将最大位置嵌入数、隐藏层大小等作为参数传入
        self.position_embeddings = nn.Embed(
            self.config.max_position_embeddings,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        # 初始化类型嵌入层，将类型词汇表大小、隐藏层大小等作为参数传入
        self.token_type_embeddings = nn.Embed(
            self.config.type_vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        # 初始化 Layer Normalization 层，使用指定的 epsilon 参数
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 初始化 Dropout 层，使用指定的 dropout 率
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    # 模型的调用方法，将输入的各种嵌入 ID 进行嵌入，并返回处理后的隐藏状态
    def __call__(self, input_ids, token_type_ids, position_ids, attention_mask, deterministic: bool = True):
        # 嵌入输入的词 ID，将其转换为整数类型并传递给词嵌入层
        inputs_embeds = self.word_embeddings(input_ids.astype("i4"))
        # 嵌入位置 ID，将其转换为整数类型并传递给位置嵌入层
        position_embeds = self.position_embeddings(position_ids.astype("i4"))
        # 嵌入类型 ID，将其转换为整数类型并传递给类型嵌入层
        token_type_embeddings = self.token_type_embeddings(token_type_ids.astype("i4"))

        # 如果配置中指定需要重新缩放嵌入层的权重
        if self.config.rescale_embeddings:
            # 对输入嵌入层的值进行按比例缩放
            inputs_embeds *= self.config.hidden_size**0.5

        # 将所有嵌入层的结果相加，形成隐藏状态的初始表示
        hidden_states = inputs_embeds + token_type_embeddings + position_embeds

        # 应用 Dropout 进行正则化，根据 deterministic 参数决定是否使用确定性模式
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 应用 Layer Normalization 进行正则化，将结果传递给 LayerNorm 层
        hidden_states = self.LayerNorm(hidden_states)
        # 返回最终的隐藏状态作为模型的输出
        return hidden_states
# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertSelfAttention with Bert->BigBird
class FlaxBigBirdSelfAttention(nn.Module):
    # 定义类属性config，表示BigBird模型的配置
    config: BigBirdConfig
    # 是否使用因果注意力，默认为False
    causal: bool = False
    # 计算时使用的数据类型，默认为jnp.float32
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        # 计算每个注意力头的维度
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        # 检查隐藏大小是否能够被注意力头数整除
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                # 如果不能整除，抛出错误提示
                "`config.hidden_size`: {self.config.hidden_size} has to be a multiple of `config.num_attention_heads` "
                "                   : {self.config.num_attention_heads}"
            )

        # 初始化查询层
        self.query = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        # 初始化键层
        self.key = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        # 初始化值层
        self.value = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        # 如果使用因果注意力，创建因果掩码
        if self.causal:
            self.causal_mask = make_causal_mask(
                jnp.ones((1, self.config.max_position_embeddings), dtype="bool"), dtype="bool"
            )

    # 将隐藏状态分割为多个注意力头
    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.config.num_attention_heads, self.head_dim))

    # 合并多个注意力头为一个隐藏状态
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.config.hidden_size,))

    @nn.compact
    # Copied from transformers.models.bart.modeling_flax_bart.FlaxBartAttention._concatenate_to_cache
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slightly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # 检测是否通过检查"cache"变量中"cached_key"来初始化
        is_initialized = self.has_variable("cache", "cached_key")
        # 初始化或获取缓存的key和value，如果不存在则创建全零数组
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 获取或创建缓存索引，初始为0
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            # 获取缓存key的维度信息，从而更新缓存的状态
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 根据当前缓存索引更新key和value的缓存状态
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            # 更新缓存索引，增加已更新的缓存向量数目
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 为缓存的解码器自注意力创建因果掩码：我们的单个查询位置只应该关注已生成和缓存的key位置，而不是剩余的零元素。
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
        
        # 返回更新后的key、value和注意力掩码
        return key, value, attention_mask
    # 定义一个名为 FlaxBigBirdBlockSparseAttention 的类，继承自 nn.Module
    class FlaxBigBirdBlockSparseAttention(nn.Module):
        # 类变量：BigBirdConfig 类型的 config 对象，block_sparse_seed 和 dtype 为可选参数
        config: BigBirdConfig
        block_sparse_seed: int = None
        dtype: jnp.dtype = jnp.float32

        # 初始化方法，设置网络的各个组件
        def setup(self):
            # 创建一个 Dense 层作为查询网络，输出维度为 config.hidden_size
            self.query = nn.Dense(
                self.config.hidden_size,
                dtype=self.dtype,
                use_bias=self.config.use_bias,
                # 使用正态分布初始化权重，标准差为 config.initializer_range
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            )
            # 创建一个 Dense 层作为键网络，输出维度为 config.hidden_size
            self.key = nn.Dense(
                self.config.hidden_size,
                dtype=self.dtype,
                use_bias=self.config.use_bias,
                # 使用正态分布初始化权重，标准差为 config.initializer_range
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            )
            # 创建一个 Dense 层作为值网络，输出维度为 config.hidden_size
            self.value = nn.Dense(
                self.config.hidden_size,
                dtype=self.dtype,
                use_bias=self.config.use_bias,
                # 使用正态分布初始化权重，标准差为 config.initializer_range
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            )

        # 静态方法：将输入 x 转置为 scores 矩阵的形状
        @staticmethod
        def transpose_for_scores(x, n_heads, head_size):
            # 新的形状为 x 的最后一维除去最后一个元素，加上 (n_heads, head_size)
            new_x_shape = x.shape[:-1] + (n_heads, head_size)
            x = x.reshape(*new_x_shape)
            # 交换指定的维度顺序：第一维和第三维互换位置
            return jnp.transpose(x, axes=(0, 2, 1, 3))

        # 实例方法：处理输入的 hidden_states 和 attention_mask，执行注意力计算
        def __call__(
            self,
            hidden_states,
            attention_mask,
            deterministic=True,
            output_attentions=False,
        ):
            # 提取配置中的注意力头数和头大小
            n_heads = self.config.num_attention_heads
            head_size = self.config.hidden_size // n_heads

            # 创建用于块稀疏注意力的掩码
            blocked_encoder_mask, band_mask, from_mask, to_mask = self.create_masks_for_block_sparse_attn(
                attention_mask, self.config.block_size
            )

            # 对查询、键和值进行维度变换，以备进行注意力计算
            query_layer = self.transpose_for_scores(self.query(hidden_states), n_heads, head_size)
            key_layer = self.transpose_for_scores(self.key(hidden_states), n_heads, head_size)
            value_layer = self.transpose_for_scores(self.value(hidden_states), n_heads, head_size)

            # 如果需要非确定性操作，则创建随机数生成器密钥
            indices_prng_key = None
            if not deterministic:
                indices_prng_key = self.make_rng("indices")

            # 执行 BigBird 块稀疏注意力机制
            attn_output, attn_weights = self.bigbird_block_sparse_attention(
                query_layer,
                key_layer,
                value_layer,
                band_mask,
                from_mask,
                to_mask,
                blocked_encoder_mask,
                blocked_encoder_mask,
                n_heads,
                head_size,
                indices_prng_key=indices_prng_key,
                deterministic=deterministic,
                plan_from_length=None,
                plan_num_rand_blocks=None,
                output_attentions=output_attentions,
            )

            # 根据需要返回注意力输出和注意力权重
            outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
            return outputs

        # 静态方法：
        @staticmethod
    def create_masks_for_block_sparse_attn(attention_mask, block_size: int):
        # 获取输入的注意力掩码的批次大小和序列长度
        batch_size, seq_length = attention_mask.shape
        # 检查序列长度是否是块大小的倍数，否则引发数值错误
        if seq_length % block_size != 0:
            raise ValueError(
                f"Sequence length must be multiple of block size, but sequence length is {seq_length}, while block"
                f" size is {block_size}."
            )

        def create_band_mask_from_inputs(from_blocked_mask, to_blocked_mask):
            """
            从二维张量掩码创建三维注意力掩码。

            Args:
                from_blocked_mask: 形状为 [batch_size, from_seq_length//from_block_size, from_block_size] 的二维张量掩码。
                to_blocked_mask: 形状为 [batch_size, to_seq_length//to_block_size, to_block_size] 的整数32位张量掩码。

            Returns:
                形状为 [batch_size, 1, from_seq_length//from_block_size-4, from_block_size, 3*to_block_size] 的浮点张量。
            """
            # 扩展并拼接来自被阻塞的掩码以进行填充
            exp_blocked_to_pad = jnp.concatenate(
                [to_blocked_mask[:, 1:-3], to_blocked_mask[:, 2:-2], to_blocked_mask[:, 3:-1]], axis=2
            )
            # 使用爱因斯坦求和符号计算带状掩码
            band_mask = jnp.einsum("blq,blk->blqk", from_blocked_mask[:, 2:-2], exp_blocked_to_pad)
            band_mask = jnp.expand_dims(band_mask, 1)
            return band_mask

        # 将注意力掩码重新形状为块形式的编码器掩码
        blocked_encoder_mask = attention_mask.reshape(batch_size, seq_length // block_size, block_size)
        # 创建带状掩码
        band_mask = create_band_mask_from_inputs(blocked_encoder_mask, blocked_encoder_mask)

        # 重新形状创建来自掩码和去掩码
        from_mask = attention_mask.reshape(batch_size, 1, seq_length, 1)
        to_mask = attention_mask.reshape(batch_size, 1, 1, seq_length)

        return blocked_encoder_mask, band_mask, from_mask, to_mask

    @staticmethod
    def jax_gather(params, indices, batch_dims=2):
        """
        正确地从参数中聚集指数（相当于tf.gather但有修改）。

        Args:
            params: 形状为 (bsz, n_heads, num_blocks, block_size, head_dim) 的参数。
            indices: 形状为 (<num_blocks, 1) 的索引。

        Returns:
            聚集后的张量，形状为 params.shape[:batch_dims] + indices.shape + params.shape[batch_dims+1:]。
        """

        def _jax_gather(params, indices):
            return params[indices]

        # 使用jax.vmap逐批次维度进行映射
        for _ in range(batch_dims):
            _jax_gather = jax.vmap(_jax_gather, in_axes=(0, 0))

        return _jax_gather(params, indices)  # 返回聚集结果
    def _create_rand_mask_from_inputs(
        self,
        from_blocked_mask,
        to_blocked_mask,
        broadcasted_rand_attn,
        num_attention_heads,
        num_random_blocks,
        batch_size,
        from_seq_length,
        from_block_size,
    ):
        """
        Create 3D attention mask from a 2D tensor mask.

        Args:
            from_blocked_mask: 2D Tensor of shape [batch_size, from_seq_length//from_block_size, from_block_size].
                Mask for the 'from' sequence, divided into blocks.
            to_blocked_mask: int32 Tensor of shape [batch_size, to_seq_length//to_block_size, to_block_size].
                Mask for the 'to' sequence, divided into blocks.
            broadcasted_rand_attn:
                [batch_size, num_attention_heads, from_seq_length//from_block_size-2, num_rand_blocks]
                Random attention distribution broadcasted across heads and sequence blocks.
            num_attention_heads: int. Number of attention heads.
            num_random_blocks: int. Number of random chunks per row.
            batch_size: int. Batch size for computation.
            from_seq_length: int. Length of 'from' sequence.
            from_block_size: int. Size of block in 'from' sequence.

        Returns:
            float Tensor of shape [batch_size, num_attention_heads, from_seq_length//from_block_size-2,
            from_block_size, num_rand_blocks*to_block_size].
            3D attention mask combining information from 'from' and 'to' sequences.
        """
        # Calculate the number of windows in the 'from' sequence
        num_windows = from_seq_length // from_block_size - 2
        
        # Gather the random attention mask using JAX gather operation
        rand_mask = self.jax_gather(to_blocked_mask, broadcasted_rand_attn, batch_dims=1)
        
        # Reshape the random mask to match the required output shape
        rand_mask = rand_mask.reshape(
            batch_size, num_attention_heads, num_windows, num_random_blocks * from_block_size
        )
        
        # Perform Einstein summation to combine 'from' block mask with random attention
        rand_mask = jnp.einsum("blq,bhlk->bhlqk", from_blocked_mask[:, 1:-1], rand_mask)
        
        # Return the final random attention mask
        return rand_mask

    @staticmethod
    def _get_rand_attn_plan(from_seq_length, from_block_size, num_rand_blocks):
        """
        根据给定的参数生成随机注意力的分布计划。

        Args:
            from_seq_length: int. 源序列的长度。
            from_block_size: int. 源序列中的块大小。
            num_rand_blocks: int. 每行随机块的数量。

        Returns:
            plan_from_length: list. 源块的结束位置计划。
            plan_num_rand_blocks: list. 每个块中随机结束位置的数量。
        """

        plan_from_length = []  # 初始化源块的结束位置列表
        plan_num_rand_blocks = []  # 初始化每个块中随机结束位置的数量列表

        # 根据条件生成不同的分布计划
        if (2 * num_rand_blocks + 5) < (from_seq_length // from_block_size):
            plan_from_length.append(int((2 * num_rand_blocks + 5) * from_block_size))
            plan_num_rand_blocks.append(num_rand_blocks)
            plan_from_length.append(from_seq_length)
            plan_num_rand_blocks.append(0)
        elif (num_rand_blocks + 5) < (from_seq_length // from_block_size):
            plan_from_length.append(int((num_rand_blocks + 5) * from_block_size))
            plan_num_rand_blocks.append(num_rand_blocks // 2)
            plan_from_length.append(from_seq_length)
            plan_num_rand_blocks.append(num_rand_blocks - (num_rand_blocks // 2))
        else:
            plan_from_length.append(from_seq_length)
            plan_num_rand_blocks.append(num_rand_blocks)

        return plan_from_length, plan_num_rand_blocks

    @staticmethod
    def _bigbird_block_rand_mask(
        from_seq_length,
        to_seq_length,
        from_block_size,
        to_block_size,
        num_rand_blocks,
        indices_prng_key: Optional[jax.random.PRNGKey] = None,
        deterministic: Optional[bool] = True,
        last_idx: Optional[int] = -1,
    ):
        """
        生成BigBird模型中块随机掩码。

        Args:
            from_seq_length: int. 源序列的长度。
            to_seq_length: int. 目标序列的长度。
            from_block_size: int. 源序列中的块大小。
            to_block_size: int. 目标序列中的块大小。
            num_rand_blocks: int. 每行随机块的数量。
            indices_prng_key: Optional[jax.random.PRNGKey]. 随机数生成器密钥。
            deterministic: Optional[bool]. 是否确定性生成随机数。
            last_idx: Optional[int]. 最后一个索引位置，默认为-1。

        Returns:
            返回生成的块随机掩码。
        """

    def _bigbird_block_rand_mask_with_head(
        self,
        from_seq_length,
        to_seq_length,
        from_block_size,
        to_block_size,
        num_heads,
        plan_from_length,
        plan_num_rand_blocks,
        indices_prng_key: Optional[jax.random.PRNGKey] = None,
        deterministic: Optional[bool] = True,
        window_block_left=1,
        window_block_right=1,
        global_block_top=1,
        global_block_bottom=1,
        global_block_left=1,
        global_block_right=1,
    ):
        """
        生成带有头信息的BigBird模型中的块随机掩码。

        Args:
            from_seq_length: int. 源序列的长度。
            to_seq_length: int. 目标序列的长度。
            from_block_size: int. 源序列中的块大小。
            to_block_size: int. 目标序列中的块大小。
            num_heads: int. 头的数量。
            plan_from_length: list. 源块的结束位置计划。
            plan_num_rand_blocks: list. 每个块中随机结束位置的数量。
            indices_prng_key: Optional[jax.random.PRNGKey]. 随机数生成器密钥。
            deterministic: Optional[bool]. 是否确定性生成随机数。
            window_block_left: int. 左侧窗口块大小，默认为1。
            window_block_right: int. 右侧窗口块大小，默认为1。
            global_block_top: int. 顶部全局块大小，默认为1。
            global_block_bottom: int. 底部全局块大小，默认为1。
            global_block_left: int. 左侧全局块大小，默认为1。
            global_block_right: int. 右侧全局块大小，默认为1.

        Returns:
            返回生成的带有头信息的块随机掩码。
        """

    @staticmethod
    def _get_single_block_row_attention(
        block_id,
        to_start_block_id,
        to_end_block_id,
        num_rand_blocks,
        indices_prng_key: Optional[jax.random.PRNGKey] = None,
        window_block_left=1,
        window_block_right=1,
        global_block_left=1,
        global_block_right=1,
    ):
        """
        获取单个块行注意力的实现。

        Args:
            block_id: int. 块的ID。
            to_start_block_id: int. 目标序列起始块的ID。
            to_end_block_id: int. 目标序列结束块的ID。
            num_rand_blocks: int. 每行随机块的数量。
            indices_prng_key: Optional[jax.random.PRNGKey]. 随机数生成器密钥。
            window_block_left: int. 左侧窗口块大小，默认为1。
            window_block_right: int. 右侧窗口块大小，默认为1。
            global_block_left: int. 左侧全局块大小，默认为1。
            global_block_right: int. 右侧全局块大小，默认为1。
        """
    ):
        """
        For a single row block get random row attention.

        Args:
            block_id: int. block id of row.
                表示行块的块标识号。
            to_start_block_id: int. random attention column start id.
                随机注意力列开始的块标识号。
            to_end_block_id: int. random attention column end id.
                随机注意力列结束的块标识号。
            num_rand_blocks: int. number of random blocks to be selected.
                要选择的随机块的数量。
            indices_prng_key: jax.random.PRNGKey. PRNG key that is used to perform random jax operations
                用于执行随机 JAX 操作的 PRNG 密钥。
            window_block_left: int. number of blocks of window to left of a block.
                在一个块左边的窗口中的块数。
            window_block_right: int. number of blocks of window to right of a block.
                在一个块右边的窗口中的块数。
            global_block_left: int. Number of blocks globally used to the left.
                左侧全局使用的块数。
            global_block_right: int. Number of blocks globally used to the right.
                右侧全局使用的块数。

        Returns:
            row containing the random attention vector of size num_rand_blocks.
            包含大小为 num_rand_blocks 的随机注意力向量的行。
        """
        # list of to_blocks from which to choose random attention
        to_block_list = jnp.arange(to_start_block_id, to_end_block_id, dtype=jnp.int32)
        # permute the blocks
        perm_block = jax.random.permutation(indices_prng_key, to_block_list)

        # illegal blocks for the current block id, using window
        illegal_blocks = list(range(block_id - window_block_left, block_id + window_block_right + 1))

        # Add blocks at the start and at the end
        illegal_blocks.extend(list(range(global_block_left)))
        illegal_blocks.extend(list(range(to_end_block_id - global_block_right, to_end_block_id)))

        # The second from_block cannot choose random attention on second last to_block
        if block_id == 1:
            illegal_blocks.append(to_end_block_id - 2)

        # The second last from_block cannot choose random attention on second to_block
        if block_id == to_end_block_id - 2:
            illegal_blocks.append(1)

        selected_random_blocks = []

        for i in range(to_end_block_id - to_start_block_id):
            if perm_block[i] not in illegal_blocks:
                selected_random_blocks.append(perm_block[i])
            if len(selected_random_blocks) == num_rand_blocks:
                break
        return jnp.array(selected_random_blocks, dtype=jnp.int32)
# 从 `transformers.models.bert.modeling_flax_bert.FlaxBertSelfOutput` 复制并修改为 BigBird
class FlaxBigBirdSelfOutput(nn.Module):
    # BigBird 的配置信息
    config: BigBirdConfig
    # 计算的数据类型
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 初始化函数，设置层的结构
    def setup(self):
        # 全连接层，将输入的隐藏状态转换为指定大小的输出
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # LayerNorm 层，用于规范化输入数据
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # Dropout 层，用于随机失活，防止过拟合
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    # 对象调用函数，执行层的前向计算
    def __call__(self, hidden_states, input_tensor, deterministic: bool = True):
        # 全连接层计算
        hidden_states = self.dense(hidden_states)
        # Dropout 计算
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # LayerNorm 计算，将残差连接后的结果规范化
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的隐藏状态
        return hidden_states


# BigBird 注意力机制层
class FlaxBigBirdAttention(nn.Module):
    # BigBird 的配置信息
    config: BigBirdConfig
    # 层的编号
    layer_id: int = None
    # 是否使用因果注意力
    causal: bool = False
    # 计算的数据类型
    dtype: jnp.dtype = jnp.float32

    # 初始化函数，设置层的结构
    def setup(self):
        # 根据配置选择不同类型的注意力机制
        if self.config.attention_type == "original_full":
            # 使用原始的全注意力机制
            self.self = FlaxBigBirdSelfAttention(self.config, causal=self.causal, dtype=self.dtype)
        elif self.config.attention_type == "block_sparse":
            # 使用块稀疏注意力机制
            self.self = FlaxBigBirdBlockSparseAttention(self.config, block_sparse_seed=self.layer_id, dtype=self.dtype)
        else:
            # 抛出错误，如果配置不匹配
            raise ValueError(
                f"Your `config.attention_type` is {self.config.attention_type} but it can either be `original_full` or"
                " `block_sparse`"
            )

        # 输出层，用于处理自注意力的输出结果
        self.output = FlaxBigBirdSelfOutput(self.config, dtype=self.dtype)

    # 对象调用函数，执行注意力计算
    def __call__(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask,
        key_value_states=None,
        init_cache=False,
        deterministic=True,
        output_attentions: bool = False,
        # 如果 attention_mask 的形状为 (*batch_sizes, kv_length)，FLAX 要求形状为 (*batch_sizes, 1, 1, kv_length)，以便广播匹配 attn_weights 的形状为 (*batch_sizes, num_heads, q_length, kv_length)
        # 当 self.config.attention_type == "original_full" 时，使用带有额外参数的 self.self 方法进行注意力计算
        attn_outputs = self.self(
            hidden_states,
            attention_mask,
            layer_head_mask=layer_head_mask,
            key_value_states=key_value_states,
            init_cache=init_cache,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )
        # 否则使用默认参数调用 self.self 方法进行注意力计算
        else:
            attn_outputs = self.self(
                hidden_states,
                attention_mask,
                deterministic=deterministic,
                output_attentions=output_attentions,
            )
        # 获取注意力输出的第一个元素
        attn_output = attn_outputs[0]
        # 通过 self.output 方法计算最终的输出 hidden_states
        hidden_states = self.output(attn_output, hidden_states, deterministic=deterministic)

        # 构建输出元组，至少包含 hidden_states
        outputs = (hidden_states,)

        # 如果需要输出注意力信息，则在输出元组中添加 attn_outputs 的第二个元素
        if output_attentions:
            outputs += (attn_outputs[1],)

        # 返回最终输出元组
        return outputs
# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertIntermediate with Bert->BigBird
class FlaxBigBirdIntermediate(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        # 定义一个全连接层，输出大小为中间层大小，使用正态分布初始化权重
        self.dense = nn.Dense(
            self.config.intermediate_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 根据配置选择激活函数
        self.activation = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states):
        # 将输入的隐藏状态通过全连接层处理
        hidden_states = self.dense(hidden_states)
        # 应用激活函数到处理后的隐藏状态
        hidden_states = self.activation(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertOutput with Bert->BigBird
class FlaxBigBirdOutput(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        # 定义一个全连接层，输出大小为隐藏大小，使用正态分布初始化权重
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 定义一个 dropout 层，用于隐藏层的输出
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        # 定义一个 LayerNorm 层，用于归一化隐藏层的输出
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(self, hidden_states, attention_output, deterministic: bool = True):
        # 将隐藏状态输入全连接层进行处理
        hidden_states = self.dense(hidden_states)
        # 对处理后的隐藏状态应用 dropout 操作
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 将 dropout 后的结果与注意力输出进行残差连接，并通过 LayerNorm 层进行归一化
        hidden_states = self.LayerNorm(hidden_states + attention_output)
        return hidden_states


class FlaxBigBirdLayer(nn.Module):
    config: BigBirdConfig
    layer_id: int = None
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        # 定义 BigBird 层的注意力机制
        self.attention = FlaxBigBirdAttention(
            self.config, layer_id=self.layer_id, causal=self.config.is_decoder, dtype=self.dtype
        )
        # 定义 BigBird 层的中间层
        self.intermediate = FlaxBigBirdIntermediate(self.config, dtype=self.dtype)
        # 定义 BigBird 层的输出层
        self.output = FlaxBigBirdOutput(self.config, dtype=self.dtype)
        # 如果配置中包含跨注意力机制，定义 BigBird 层的跨注意力机制
        if self.config.add_cross_attention:
            self.crossattention = FlaxBigBirdAttention(self.config, causal=False, dtype=self.dtype)

    # Copied from transformers.models.bert.modeling_flax_bert.FlaxBertLayer.__call__ with Bert->BigBird
    def __call__(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic: bool = True,
        output_attentions: bool = False,

        # BigBird 层的调用方法，接收隐藏状态、注意力掩码、层头掩码等输入参数
        # 如果需要初始化缓存，则传入 True
        # deterministic 参数指定是否使用确定性计算，默认为 True
        # output_attentions 参数指定是否输出注意力权重，默认为 False
        # Self Attention
        # 使用 self.attention 方法对输入的 hidden_states 进行自注意力计算
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            layer_head_mask=layer_head_mask,
            init_cache=init_cache,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )
        # 获取自注意力计算后的输出
        attention_output = attention_outputs[0]

        # Cross-Attention Block
        # 如果存在 encoder_hidden_states，则执行交叉注意力计算
        if encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask=encoder_attention_mask,
                layer_head_mask=layer_head_mask,
                key_value_states=encoder_hidden_states,
                deterministic=deterministic,
                output_attentions=output_attentions,
            )
            # 获取交叉注意力计算后的输出
            attention_output = cross_attention_outputs[0]

        # 经过注意力计算后的输出再经过 intermediate 层处理
        hidden_states = self.intermediate(attention_output)
        # 经过输出层处理，得到最终的 hidden_states
        hidden_states = self.output(hidden_states, attention_output, deterministic=deterministic)

        # 输出结果为 hidden_states 组成的元组
        outputs = (hidden_states,)

        # 如果需要输出 attentions，则将 attentions 添加到输出结果中
        if output_attentions:
            outputs += (attention_outputs[1],)
            # 如果存在 encoder_hidden_states，则将交叉注意力也添加到输出结果中
            if encoder_hidden_states is not None:
                outputs += (cross_attention_outputs[1],)
        
        # 返回最终的输出结果
        return outputs
# 定义一个名为 FlaxBigBirdLayerCollection 的类，继承自 nn.Module
class FlaxBigBirdLayerCollection(nn.Module):
    # config 属性，类型为 BigBirdConfig，用于存储 BigBird 的配置信息
    config: BigBirdConfig
    # dtype 属性，默认为 jnp.float32，用于定义计算的数据类型
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型
    # gradient_checkpointing 属性，默认为 False，表示是否开启梯度检查点
    gradient_checkpointing: bool = False

    # 定义类的初始化方法 setup
    def setup(self):
        # 如果开启了梯度检查点
        if self.gradient_checkpointing:
            # 定义一个经过 remat 处理的 FlaxBigBirdCheckpointLayer 类，用于梯度检查点
            FlaxBigBirdCheckpointLayer = remat(FlaxBigBirdLayer, static_argnums=(5, 6, 7))
            # 初始化 self.layers，创建包含多个 FlaxBigBirdCheckpointLayer 实例的列表
            self.layers = [
                FlaxBigBirdCheckpointLayer(self.config, layer_id=i, name=str(i), dtype=self.dtype)
                for i in range(self.config.num_hidden_layers)
            ]
        else:
            # 如果未开启梯度检查点，初始化 self.layers，创建包含多个 FlaxBigBirdLayer 实例的列表
            self.layers = [
                FlaxBigBirdLayer(self.config, layer_id=i, name=str(i), dtype=self.dtype)
                for i in range(self.config.num_hidden_layers)
            ]

    # 定义类的调用方法 __call__，用于执行实例化对象时的操作
    # 该方法功能与 transformers.models.bert.modeling_flax_bert.FlaxBertLayerCollection.__call__ 相似，替换了 Bert 为 BigBird
    def __call__(
        self,
        hidden_states,  # 输入参数，表示隐藏状态
        attention_mask,  # 输入参数，表示注意力掩码
        head_mask,  # 输入参数，表示头掩码
        encoder_hidden_states: Optional[jnp.ndarray] = None,  # 可选输入参数，编码器隐藏状态
        encoder_attention_mask: Optional[jnp.ndarray] = None,  # 可选输入参数，编码器注意力掩码
        init_cache: bool = False,  # 是否初始化缓存，默认为 False
        deterministic: bool = True,  # 是否确定性计算，默认为 True
        output_attentions: bool = False,  # 是否输出注意力，默认为 False
        output_hidden_states: bool = False,  # 是否输出隐藏状态，默认为 False
        return_dict: bool = True,  # 是否返回字典，默认为 True
        # 返回值：根据参数执行 BigBird 相关计算并返回相应结果
        ):
            all_attentions = () if output_attentions else None
            all_hidden_states = () if output_hidden_states else None
            all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

            # 检查是否需要为每个层级指定正确数量的头部掩码
            if head_mask is not None:
                if head_mask.shape[0] != (len(self.layers)):
                    raise ValueError(
                        f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.shape[0]}."
                    )

            # 遍历每一层的 Transformer 层进行处理
            for i, layer in enumerate(self.layers):
                # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                # 调用当前层的 Transformer 层进行前向传播
                layer_outputs = layer(
                    hidden_states,
                    attention_mask,
                    head_mask[i] if head_mask is not None else None,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    init_cache,
                    deterministic,
                    output_attentions,
                )

                # 更新隐藏状态为当前层输出的第一个元素
                hidden_states = layer_outputs[0]

                # 如果需要输出注意力权重，则将当前层的注意力权重添加到 all_attentions 中
                if output_attentions:
                    all_attentions += (layer_outputs[1],)

                    # 如果有编码器的隐藏状态，将当前层的交叉注意力权重添加到 all_cross_attentions 中
                    if encoder_hidden_states is not None:
                        all_cross_attentions += (layer_outputs[2],)

            # 如果需要输出隐藏状态，则将最终的隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # 构建最终的输出元组
            outputs = (hidden_states, all_hidden_states, all_attentions, all_cross_attentions)

            # 如果不需要返回字典形式的结果，则以元组形式返回所有非空结果
            if not return_dict:
                return tuple(v for v in outputs if v is not None)

            # 如果需要返回字典形式的结果，则构建特定格式的输出对象并返回
            return FlaxBaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_attentions,
                cross_attentions=all_cross_attentions,
            )
# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertEncoder with Bert->BigBird
class FlaxBigBirdEncoder(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32  # 计算过程中使用的数据类型
    gradient_checkpointing: bool = False  # 梯度检查点是否启用，默认为 False

    def setup(self):
        # 初始化 BigBird 编码器层集合，配置包括数据类型和梯度检查点设置
        self.layer = FlaxBigBirdLayerCollection(
            self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )

    def __call__(
        self,
        hidden_states,
        attention_mask,
        head_mask,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 调用 BigBird 编码器层集合来处理输入
        return self.layer(
            hidden_states,
            attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            init_cache=init_cache,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertPredictionHeadTransform with Bert->BigBird
class FlaxBigBirdPredictionHeadTransform(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 初始化 BigBird 预测头转换层，包括稠密层、激活函数和 LayerNorm 层
        self.dense = nn.Dense(self.config.hidden_size, dtype=self.dtype)
        self.activation = ACT2FN[self.config.hidden_act]
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(self, hidden_states):
        # 通过稠密层、激活函数和 LayerNorm 层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return self.LayerNorm(hidden_states)


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertLMPredictionHead with Bert->BigBird, np.ndarray->jnp.ndarray
class FlaxBigBirdLMPredictionHead(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32
    bias_init: Callable[..., jnp.ndarray] = jax.nn.initializers.zeros

    def setup(self):
        # 初始化 BigBird 语言模型预测头，包括预测头转换和输出稠密层
        self.transform = FlaxBigBirdPredictionHeadTransform(self.config, dtype=self.dtype)
        self.decoder = nn.Dense(self.config.vocab_size, dtype=self.dtype, use_bias=False)
        self.bias = self.param("bias", self.bias_init, (self.config.vocab_size,))
    # 定义一个特殊方法 __call__，使得对象可以像函数一样被调用
    def __call__(self, hidden_states, shared_embedding=None):
        # 调用 transform 方法对隐藏状态进行变换处理
        hidden_states = self.transform(hidden_states)

        # 如果提供了共享的嵌入矩阵，则使用 decoder 对象应用该共享嵌入
        if shared_embedding is not None:
            hidden_states = self.decoder.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
        else:
            # 否则，直接使用 decoder 对象处理隐藏状态
            hidden_states = self.decoder(hidden_states)

        # 将 bias 转换为与当前数据类型相匹配的 JAX 数组
        bias = jnp.asarray(self.bias, self.dtype)
        # 将隐藏状态加上偏置项
        hidden_states += bias
        # 返回处理后的隐藏状态
        return hidden_states
# 从 transformers.models.bert.modeling_flax_bert.FlaxBertOnlyMLMHead 复制并修改为 BigBird
class FlaxBigBirdOnlyMLMHead(nn.Module):
    # 使用 BigBirdConfig 配置类初始化模块
    config: BigBirdConfig
    # 默认数据类型为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 使用 BigBirdLMPredictionHead 初始化预测头部
        self.predictions = FlaxBigBirdLMPredictionHead(self.config, dtype=self.dtype)

    def __call__(self, hidden_states, shared_embedding=None):
        # 使用预测头部处理隐藏状态并返回结果
        hidden_states = self.predictions(hidden_states, shared_embedding=shared_embedding)
        return hidden_states


# 从 transformers.models.bert.modeling_flax_bert.FlaxBertPreTrainingHeads 复制并修改为 BigBird
class FlaxBigBirdPreTrainingHeads(nn.Module):
    # 使用 BigBirdConfig 配置类初始化模块
    config: BigBirdConfig
    # 默认数据类型为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 使用 BigBirdLMPredictionHead 初始化预测头部
        self.predictions = FlaxBigBirdLMPredictionHead(self.config, dtype=self.dtype)
        # 使用 Dense 层初始化序列关系预测
        self.seq_relationship = nn.Dense(2, dtype=self.dtype)

    def __call__(self, hidden_states, pooled_output, shared_embedding=None):
        # 使用预测头部处理隐藏状态并返回预测分数
        prediction_scores = self.predictions(hidden_states, shared_embedding=shared_embedding)
        # 使用序列关系预测处理池化输出并返回结果
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class FlaxBigBirdPreTrainedModel(FlaxPreTrainedModel):
    """
    一个抽象类，处理权重初始化以及下载和加载预训练模型的简单接口。
    """

    # 使用 BigBirdConfig 配置类作为配置类
    config_class = BigBirdConfig
    # 基础模型前缀为 "bert"
    base_model_prefix = "bert"
    # 模块类默认为空
    module_class: nn.Module = None

    def __init__(
        self,
        config: BigBirdConfig,
        input_shape: Optional[tuple] = None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        gradient_checkpointing: bool = False,
        **kwargs,
    ):
        # 使用模块类初始化模块，根据配置和其他参数设置输入形状等
        module = self.module_class(config=config, dtype=dtype, gradient_checkpointing=gradient_checkpointing, **kwargs)
        # 根据注意力类型和输入形状设置默认输入形状
        if config.attention_type == "block_sparse" and input_shape is None:
            input_shape = (1, 12 * config.block_size)
        elif input_shape is None:
            input_shape = (1, 1)

        # 调用父类初始化方法
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    # 从 transformers.models.bert.modeling_flax_bert.FlaxBertPreTrainedModel.enable_gradient_checkpointing 复制
    def enable_gradient_checkpointing(self):
        # 使用模块类初始化模块，并启用梯度检查点
        self._module = self.module_class(
            config=self.config,
            dtype=self.dtype,
            gradient_checkpointing=True,
        )
    # 初始化模型权重的函数，使用给定的随机数种子和输入形状初始化模型参数
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        input_ids = jnp.zeros(input_shape, dtype="i4")
        # 根据输入张量创建与其相同形状的 token 类型张量，初始化为零
        token_type_ids = jnp.zeros_like(input_ids)
        # 创建位置张量，广播到与 input_ids 相同的形状
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        # 创建注意力掩码张量，初始化为全 1
        attention_mask = jnp.ones_like(input_ids)
        # 创建头掩码张量，形状为 (层数, 注意力头数)，初始化为全 1
        head_mask = jnp.ones((self.config.num_hidden_layers, self.config.num_attention_heads))

        # 使用随机数种子 rng 拆分出三个新的随机数种子
        params_rng, dropout_rng, indices_rng = jax.random.split(rng, num=3)
        # 将拆分后的随机数种子保存在字典中
        rngs = {"params": params_rng, "dropout": dropout_rng, "indices": indices_rng}

        # 如果配置中包含跨注意力机制，则初始化编码器隐藏状态和编码器注意力掩码
        if self.config.add_cross_attention:
            encoder_hidden_states = jnp.zeros(input_shape + (self.config.hidden_size,))
            encoder_attention_mask = attention_mask
            # 使用模块的初始化方法初始化模型，返回结果不作为字典返回
            module_init_outputs = self.module.init(
                rngs,
                input_ids,
                attention_mask,
                token_type_ids,
                position_ids,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                return_dict=False,
            )
        else:
            # 使用模块的初始化方法初始化模型，返回结果不作为字典返回
            module_init_outputs = self.module.init(
                rngs,
                input_ids,
                attention_mask,
                token_type_ids,
                position_ids,
                head_mask,
                return_dict=False,
            )

        # 获取初始化后的随机参数
        random_params = module_init_outputs["params"]

        # 如果给定了预训练参数，则将随机参数展平并添加到参数中
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            # 冻结参数并返回
            return freeze(unflatten_dict(params))
        else:
            # 否则直接返回随机初始化的参数
            return random_params

    # 从 transformers.models.bart.modeling_flax_bart.FlaxBartDecoderPreTrainedModel.init_cache 复制而来
    # 初始化缓存的函数，用于快速自回归解码
    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                用于快速自回归解码的批量大小，定义了初始化缓存的批量大小。
            max_length (`int`):
                自回归解码的最大可能长度，定义了初始化缓存的序列长度。
        """
        # 初始化用于检索缓存的输入变量
        input_ids = jnp.ones((batch_size, max_length), dtype="i4")
        # 创建与 input_ids 相同形状的注意力掩码张量，初始化为全 1
        attention_mask = jnp.ones_like(input_ids, dtype="i4")
        # 广播位置张量到与 input_ids 相同的形状
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        # 使用模块的初始化方法初始化模型，返回结果不作为字典返回，并标记为初始化缓存
        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        # 返回解冻后的缓存变量
        return unfreeze(init_variables["cache"])
    # 将模型调用函数装饰为使用指定的文档字符串格式
    @add_start_docstrings_to_model_forward(BIG_BIRD_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 定义模型调用函数，接受多个输入参数
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        params: dict = None,
        dropout_rng: Optional[jax.random.PRNGKey] = None,
        indices_rng: Optional[jax.random.PRNGKey] = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        past_key_values: dict = None,
# 定义了一个 FlaxBigBirdModule 类，继承自 nn.Module
class FlaxBigBirdModule(nn.Module):
    # 类属性，存储 BigBirdConfig 配置对象
    config: BigBirdConfig
    # 计算时使用的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    # 是否添加池化层的标志，默认为 True
    add_pooling_layer: bool = True
    # 是否使用梯度检查点的标志，默认为 False
    gradient_checkpointing: bool = False

    # 模块初始化方法
    def setup(self):
        # 初始化 embeddings 属性，调用 FlaxBigBirdEmbeddings 构造方法
        self.embeddings = FlaxBigBirdEmbeddings(self.config, dtype=self.dtype)
        # 初始化 encoder 属性，调用 FlaxBigBirdEncoder 构造方法
        self.encoder = FlaxBigBirdEncoder(
            self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )
        # 初始化 pooler 属性，调用 nn.Dense 构造方法
        self.pooler = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )

    # 对象调用方法，实现模块的前向计算
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 调用 embeddings 属性的方法，获取输入序列的嵌入表示
        hidden_states = self.embeddings(
            input_ids, token_type_ids, position_ids, attention_mask, deterministic=deterministic
        )
        # 调用 encoder 属性的方法，对输入的隐藏状态进行编码
        outputs = self.encoder(
            hidden_states,
            attention_mask,
            head_mask=head_mask,
            deterministic=deterministic,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 从 encoder 输出中获取隐藏状态
        hidden_states = outputs[0]

        # 如果设置了添加池化层的标志，则对隐藏状态进行池化操作
        pooled = nn.tanh(self.pooler(hidden_states[:, 0, :])) if self.add_pooling_layer else None

        # 如果 return_dict 为 False，则根据 pooled 是否为 None 返回不同的输出
        if not return_dict:
            if pooled is None:
                return (hidden_states,) + outputs[1:]
            return (hidden_states, pooled) + outputs[1:]

        # 构建返回的输出对象，包括最终的隐藏状态和池化输出
        return FlaxBaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


# 添加了关于 BigBird 模型的文档字符串
@add_start_docstrings(
    "The bare BigBird Model transformer outputting raw hidden-states without any specific head on top.",
    BIG_BIRD_START_DOCSTRING,
)
# 从 FlaxBigBirdPreTrainedModel 继承，并将 module_class 设置为 FlaxBigBirdModule
class FlaxBigBirdModel(FlaxBigBirdPreTrainedModel):
    module_class = FlaxBigBirdModule


# 复制自 transformers.models.bert.modeling_flax_bert.FlaxBertModel，将其中的 Bert 替换为 BigBird
# 添加了对 FlaxBigBirdModel 的调用样例文档字符串
append_call_sample_docstring(FlaxBigBirdModel, _CHECKPOINT_FOR_DOC, FlaxBaseModelOutputWithPooling, _CONFIG_FOR_DOC)


# 复制自 transformers.models.bert.modeling_flax_bert.FlaxBertForPreTrainingModule，将其中的 Bert 替换为 BigBird
class FlaxBigBirdForPreTrainingModule(nn.Module):
    # 定义类的属性，BigBirdConfig 类型的 config，默认数据类型为 jnp.float32 的 dtype，是否开启梯度检查点的 gradient_checkpointing
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    # 类的初始化方法
    def setup(self):
        # 初始化 FlaxBigBirdModule 类对象 self.bert，传入配置 config、数据类型 dtype、梯度检查点设置 gradient_checkpointing
        self.bert = FlaxBigBirdModule(
            config=self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 初始化 FlaxBigBirdPreTrainingHeads 类对象 self.cls，传入配置 config、数据类型 dtype
        self.cls = FlaxBigBirdPreTrainingHeads(config=self.config, dtype=self.dtype)

    # 类的调用方法，接收多个参数，包括输入的各种 IDs、掩码、位置 IDs、头掩码，以及一些控制参数
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 调用 self.bert 对象进行模型前向传播，传入所有参数，并指定返回的数据类型是字典（return_dict=True）
        outputs = self.bert(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 根据配置决定是否共享词嵌入矩阵
        if self.config.tie_word_embeddings:
            # 如果要求共享词嵌入矩阵，则获取 self.bert 对象中的共享词嵌入
            shared_embedding = self.bert.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
        else:
            shared_embedding = None

        # 从模型输出中获取隐藏状态和池化输出
        hidden_states = outputs[0]
        pooled_output = outputs[1]

        # 调用 self.cls 对象进行预测头部预训练任务的预测，传入隐藏状态、池化输出以及可能的共享词嵌入
        prediction_scores, seq_relationship_score = self.cls(
            hidden_states, pooled_output, shared_embedding=shared_embedding
        )

        # 根据 return_dict 的值确定返回的数据结构
        if not return_dict:
            # 如果 return_dict=False，则返回元组形式的输出，包括预测得分、序列关系得分以及额外的隐藏状态和注意力权重
            return (prediction_scores, seq_relationship_score) + outputs[2:]

        # 如果 return_dict=True，则返回 FlaxBigBirdForPreTrainingOutput 类的实例，包含预测得分、序列关系得分、隐藏状态和注意力权重
        return FlaxBigBirdForPreTrainingOutput(
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    BigBird Model with two heads on top as done during the pretraining: a `masked language modeling` head and a `next
    sentence prediction (classification)` head.
    """,
    BIG_BIRD_START_DOCSTRING,
)
# 定义一个BigBird模型，包含预训练过程中的两个头部：掩码语言建模头部和下一个句子预测头部
# 这段注释是为了说明该类是从FlaxBigBirdPreTrainedModel继承而来的，并设置了模块类为FlaxBigBirdForPreTrainingModule
class FlaxBigBirdForPreTraining(FlaxBigBirdPreTrainedModel):
    module_class = FlaxBigBirdForPreTrainingModule


FLAX_BIG_BIRD_FOR_PRETRAINING_DOCSTRING = """
    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, FlaxBigBirdForPreTraining

    >>> tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")
    >>> model = FlaxBigBirdForPreTraining.from_pretrained("google/bigbird-roberta-base")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="np")
    >>> outputs = model(**inputs)

    >>> prediction_logits = outputs.prediction_logits
    >>> seq_relationship_logits = outputs.seq_relationship_logits
    ```
"""
# 更新FlaxBigBirdForPreTraining类的文档字符串，包含了输入说明和示例
overwrite_call_docstring(
    FlaxBigBirdForPreTraining,
    BIG_BIRD_INPUTS_DOCSTRING.format("batch_size, sequence_length") + FLAX_BIG_BIRD_FOR_PRETRAINING_DOCSTRING,
)
# 向FlaxBigBirdForPreTraining类中追加或替换返回文档字符串，指定了输出类型为FlaxBigBirdForPreTrainingOutput，配置类为_CONFIG_FOR_DOC


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertForMaskedLMModule with Bert->BigBird
# 从transformers.models.bert.modeling_flax_bert.FlaxBertForMaskedLMModule复制过来，将Bert更换为BigBird
class FlaxBigBirdForMaskedLMModule(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        # 初始化BigBird模块，不添加池化层
        self.bert = FlaxBigBirdModule(
            config=self.config,
            add_pooling_layer=False,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 初始化BigBird模型的MLM头部
        self.cls = FlaxBigBirdOnlyMLMHead(config=self.config, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        # 定义了FlaxBigBirdForMaskedLMModule的调用方法，接受多个输入参数
        # 调用 BERT 模型进行前向传播，获取模型输出
        outputs = self.bert(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从模型输出中提取隐藏状态
        hidden_states = outputs[0]

        # 如果配置要求共享词嵌入，则获取共享的词嵌入向量
        if self.config.tie_word_embeddings:
            shared_embedding = self.bert.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
        else:
            shared_embedding = None

        # 使用分类头部模型计算预测分数
        logits = self.cls(hidden_states, shared_embedding=shared_embedding)

        # 如果不返回字典形式的输出，则返回元组
        if not return_dict:
            return (logits,) + outputs[1:]

        # 返回 FlaxMaskedLMOutput 类的实例作为字典形式的输出
        return FlaxMaskedLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings("""BigBird Model with a `language modeling` head on top.""", BIG_BIRD_START_DOCSTRING)
# 添加起始文档字符串，说明这是在 BigBird 模型基础上加上语言建模头部的类
# 从 transformers.models.bert.modeling_flax_bert.FlaxBertForMaskedLM 复制并将 Bert 改为 BigBird
class FlaxBigBirdForMaskedLM(FlaxBigBirdPreTrainedModel):
    module_class = FlaxBigBirdForMaskedLMModule

# 添加调用示例文档字符串，描述如何在 FlaxBigBirdForMaskedLM 类上附加检查点的说明
append_call_sample_docstring(FlaxBigBirdForMaskedLM, _CHECKPOINT_FOR_DOC, FlaxMaskedLMOutput, _CONFIG_FOR_DOC)

# BigBird 分类头部，用于句子级别分类任务
class FlaxBigBirdClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dense = nn.Dense(self.config.hidden_size, dtype=self.dtype)
        # 设置分类器的 dropout，如果未提供特定的分类器 dropout，则使用隐藏层 dropout
        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Dense(self.config.num_labels, dtype=self.dtype)

    def __call__(self, features, deterministic=True):
        x = features[:, 0, :]  # 取 <s> token（相当于 [CLS]）
        x = self.dropout(x, deterministic=deterministic)
        x = self.dense(x)
        x = ACT2FN[self.config.hidden_act](x)  # 使用指定的激活函数处理隐藏层输出
        x = self.dropout(x, deterministic=deterministic)
        x = self.out_proj(x)
        return x

# BigBird 序列分类模块
class FlaxBigBirdForSequenceClassificationModule(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        # 设置 BigBird 模块作为 BERT
        self.bert = FlaxBigBirdModule(
            config=self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )
        self.classifier = FlaxBigBirdClassificationHead(self.config, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 模型计算
        outputs = self.bert(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]  # 获取序列输出
        logits = self.classifier(sequence_output, deterministic=deterministic)  # 使用分类头部进行分类

        if not return_dict:
            return (logits,) + outputs[2:]

        # 返回序列分类器输出对象
        return FlaxSequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

@add_start_docstrings(
    """
    BigBird Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    """
)
# 添加起始文档字符串，说明这是在 BigBird 模型基础上加上序列分类/回归头部的类（线性层在顶部）
    pooled output) e.g. for GLUE tasks.
    ```
    这部分代码是一个多行字符串，描述了`BigBirdForSequenceClassification`类的用途和功能，特别是在GLUE任务中如何使用汇集输出（pooled output）。
    ```
    BIG_BIRD_START_DOCSTRING,
    ```
    这里调用了`BIG_BIRD_START_DOCSTRING`，它可能是一个预定义的常量或函数，用于指示文档字符串的开始位置。
    ```
# 从transformers.models.bert.modeling_flax_bert.FlaxBertForSequenceClassification复制代码，将Bert改为BigBird
class FlaxBigBirdForSequenceClassification(FlaxBigBirdPreTrainedModel):
    # 将模块类指定为FlaxBigBirdForSequenceClassificationModule
    module_class = FlaxBigBirdForSequenceClassificationModule


# 将样本调用文档字符串附加到FlaxBigBirdForSequenceClassification类上
append_call_sample_docstring(
    FlaxBigBirdForSequenceClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxSequenceClassifierOutput,
    _CONFIG_FOR_DOC,
)


# 从transformers.models.bert.modeling_flax_bert.FlaxBertForMultipleChoiceModule复制代码，将Bert改为BigBird
class FlaxBigBirdForMultipleChoiceModule(nn.Module):
    # BigBird配置
    config: BigBirdConfig
    # 数据类型，默认为32位浮点数
    dtype: jnp.dtype = jnp.float32
    # 梯度检查点，默认关闭
    gradient_checkpointing: bool = False

    def setup(self):
        # 初始化BigBird模块
        self.bert = FlaxBigBirdModule(
            config=self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # Dropout层，使用隐藏层dropout比率
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        # 分类器，输出为1，使用指定数据类型
        self.classifier = nn.Dense(1, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 获取选项数量
        num_choices = input_ids.shape[1]
        # 重新整形输入数据，用于模型输入
        input_ids = input_ids.reshape(-1, input_ids.shape[-1]) if input_ids is not None else None
        attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1]) if attention_mask is not None else None
        token_type_ids = token_type_ids.reshape(-1, token_type_ids.shape[-1]) if token_type_ids is not None else None
        position_ids = position_ids.reshape(-1, position_ids.shape[-1]) if position_ids is not None else None

        # 模型前向传播
        outputs = self.bert(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 提取池化输出
        pooled_output = outputs[1]
        # 应用dropout到池化输出
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)
        # 应用分类器得到logits
        logits = self.classifier(pooled_output)

        # 重新整形logits，以匹配选项数量
        reshaped_logits = logits.reshape(-1, num_choices)

        # 如果不返回字典，则返回重整后的logits和额外的输出
        if not return_dict:
            return (reshaped_logits,) + outputs[2:]

        # 返回多选模型输出对象
        return FlaxMultipleChoiceModelOutput(
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    BigBird模型，顶部带有多选分类头（池化输出上的线性层和softmax），例如用于RocStories/SWAG任务。
    """,
    BIG_BIRD_START_DOCSTRING,
)
class FlaxBigBirdForMultipleChoice(FlaxBigBirdPreTrainedModel):
    # 将模块类指定为FlaxBigBirdForMultipleChoiceModule
    module_class = FlaxBigBirdForMultipleChoiceModule
    # 初始化函数，用于创建一个 BigBirdLayer 的实例
    def __init__(
        self,
        config: BigBirdConfig,  # 参数：BigBird 模型的配置对象
        input_shape: Optional[tuple] = None,  # 参数：输入数据的形状，可选，默认为 None
        seed: int = 0,  # 参数：随机种子，默认为 0
        dtype: jnp.dtype = jnp.float32,  # 参数：数据类型，默认为 jnp.float32
        _do_init: bool = True,  # 参数：是否执行初始化，默认为 True
        **kwargs,  # 其他关键字参数
    ):
        # 如果配置的注意力类型是 "block_sparse" 并且输入形状是 None
        if config.attention_type == "block_sparse" and input_shape is None:
            # 设置输入形状为 (1, 1, 12 * config.block_size)
            input_shape = (1, 1, 12 * config.block_size)
        # 如果输入形状仍然是 None
        elif input_shape is None:
            # 设置输入形状为 (1, 1)
            input_shape = (1, 1)
        
        # 调用父类的初始化方法，传递配置对象、输入形状、随机种子、数据类型、是否执行初始化标志位
        super().__init__(config, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)
# 调用函数 overwrite_call_docstring，为 FlaxBigBirdForMultipleChoice 类重写文档字符串
overwrite_call_docstring(
    FlaxBigBirdForMultipleChoice, BIG_BIRD_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
)

# 调用函数 append_call_sample_docstring，为 FlaxBigBirdForMultipleChoice 类附加示例文档字符串
append_call_sample_docstring(
    FlaxBigBirdForMultipleChoice,
    _CHECKPOINT_FOR_DOC,
    FlaxMultipleChoiceModelOutput,
    _CONFIG_FOR_DOC,
)


# 从 transformers.models.bert.modeling_flax_bert.FlaxBertForTokenClassificationModule 复制，并将 Bert 替换为 BigBird
class FlaxBigBirdForTokenClassificationModule(nn.Module):
    config: BigBirdConfig  # 定义配置项为 BigBirdConfig 类型
    dtype: jnp.dtype = jnp.float32  # 数据类型设置为 jnp.float32，默认为浮点数
    gradient_checkpointing: bool = False  # 梯度检查点设置为 False，默认不启用

    def setup(self):
        # 初始化 self.bert，使用 FlaxBigBirdModule 构建 BigBird 模型，设置一些参数
        self.bert = FlaxBigBirdModule(
            config=self.config,
            dtype=self.dtype,
            add_pooling_layer=False,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 设置 dropout 层，使用配置中的 classifier_dropout，若未指定则使用 hidden_dropout_prob
        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(rate=classifier_dropout)  # 设置 dropout 层
        self.classifier = nn.Dense(self.config.num_labels, dtype=self.dtype)  # 设置分类器层

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 调用 BigBird 模型 self.bert 进行前向传播
        outputs = self.bert(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]  # 获取模型输出的隐藏状态
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)  # 应用 dropout
        logits = self.classifier(hidden_states)  # 使用分类器得到 logits

        if not return_dict:
            return (logits,) + outputs[1:]  # 返回 logits 和其它输出

        # 返回 FlaxTokenClassifierOutput 类型的对象，包含 logits、隐藏状态和注意力分布
        return FlaxTokenClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    BigBird 模型添加了一个 token 分类头部（线性层在隐藏状态输出之上），例如用于命名实体识别（NER）任务。
    """,
    BIG_BIRD_START_DOCSTRING,
)
# 从 transformers.models.bert.modeling_flax_bert.FlaxBertForTokenClassification 复制，并将 Bert 替换为 BigBird
class FlaxBigBirdForTokenClassification(FlaxBigBirdPreTrainedModel):
    module_class = FlaxBigBirdForTokenClassificationModule  # 指定模型类为 FlaxBigBirdForTokenClassificationModule


# 附加文档字符串示例到 FlaxBigBirdForTokenClassification 类
append_call_sample_docstring(
    FlaxBigBirdForTokenClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxTokenClassifierOutput,
    _CONFIG_FOR_DOC,
)


# 为问答任务头部定义类 FlaxBigBirdForQuestionAnsweringHead
class FlaxBigBirdForQuestionAnsweringHead(nn.Module):
    config: BigBirdConfig  # 定义配置项为 BigBirdConfig 类型
    dtype: jnp.dtype = jnp.float32  # 数据类型设置为 jnp.float32，默认为浮点数
    # 在模型设置过程中初始化 dropout 层，使用给定的隐藏层dropout概率
    def setup(self):
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        # 初始化一个中间层对象，用于处理 BigBird 模型的中间输出
        self.intermediate = FlaxBigBirdIntermediate(self.config, dtype=self.dtype)
        # 初始化一个输出层对象，用于处理 BigBird 模型的最终输出
        self.output = FlaxBigBirdOutput(self.config, dtype=self.dtype)
        # 初始化一个全连接层，用于执行问题回答任务的最终输出
        self.qa_outputs = nn.Dense(self.config.num_labels, dtype=self.dtype)

    # 模型调用方法，接收编码器的输出和一个确定性标志
    def __call__(self, encoder_output, deterministic=True):
        # 对编码器输出应用 dropout 层，根据确定性标志确定是否随机失活
        hidden_states = self.dropout(encoder_output, deterministic=deterministic)
        # 将 dropout 处理后的隐藏状态传递给中间层对象处理
        hidden_states = self.intermediate(hidden_states)
        # 将中间层处理后的输出传递给输出层对象处理，并结合编码器的原始输出
        hidden_states = self.output(hidden_states, encoder_output)
        # 将输出层处理后的结果传递给问题回答的全连接层，生成最终的模型输出
        hidden_states = self.qa_outputs(hidden_states)
        # 返回问题回答任务的最终输出
        return hidden_states
class FlaxBigBirdForQuestionAnsweringModule(nn.Module):
    # 定义模型配置
    config: BigBirdConfig
    # 定义数据类型，默认为32位浮点数
    dtype: jnp.dtype = jnp.float32
    # 是否添加池化层，默认为False
    add_pooling_layer: bool = False
    # 是否使用梯度检查点，默认为False
    gradient_checkpointing: bool = False

    def setup(self):
        # 设置模型的类别数为2
        self.config.num_labels = 2
        # 初始化 BigBird 模型
        self.bert = FlaxBigBirdModule(
            self.config,
            dtype=self.dtype,
            add_pooling_layer=self.add_pooling_layer,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 初始化用于问答的分类器
        self.qa_classifier = FlaxBigBirdForQuestionAnsweringHead(self.config, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        logits_mask=None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 调用模型计算
        outputs = self.bert(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 提取模型输出的隐藏状态
        hidden_states = outputs[0]
        # 如果启用池化层，则提取池化后的输出
        pooled_output = outputs[1] if self.add_pooling_layer else None
        # 使用问答分类器计算 logits
        logits = self.qa_classifier(hidden_states, deterministic=deterministic)

        if logits_mask is not None:
            # 如果提供了 logits_mask，则在竞赛中移除问题标记
            logits = logits - logits_mask * 1e6

        # 将 logits 分割为起始位置和结束位置的预测
        start_logits, end_logits = logits.split(self.config.num_labels, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if not return_dict:
            # 如果不要求返回字典，则返回元组形式的结果
            return (start_logits, end_logits) + outputs[1:]

        # 返回问答模型的输出，包括起始和结束 logits，以及其它可选输出
        return FlaxBigBirdForQuestionAnsweringModelOutput(
            start_logits=start_logits,
            end_logits=end_logits,
            pooled_output=pooled_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    BigBird 模型，顶部带有用于抽取式问答任务（如 SQuAD）的跨度分类头部（线性层在隐藏状态输出之上计算 'span start logits' 和 'span end logits'）。
    """,
    BIG_BIRD_START_DOCSTRING,
)
class FlaxBigBirdForQuestionAnswering(FlaxBigBirdPreTrainedModel):
    module_class = FlaxBigBirdForQuestionAnsweringModule

    @add_start_docstrings_to_model_forward(BIG_BIRD_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 定义一个静态方法，用于静态调用或实例调用
    @staticmethod
    # 以下是 __call__ 方法的定义，用于模型类实例的调用
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        question_lengths=None,
        params: dict = None,
        dropout_rng: Optional[jax.random.PRNGKey] = None,
        indices_rng: Optional[jax.random.PRNGKey] = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 根据需求设置是否输出注意力权重
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 根据需求设置是否输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 根据需求设置是否返回字典格式的输出结果
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 如果未提供位置编码，使用输入张量形状的广播操作生成位置编码
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        # 如果未提供注意力掩码，使用与输入张量形状相同的全 1 张量作为注意力掩码
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # 如果未提供头部掩码，使用形状为 (层数, 注意力头数) 的全 1 张量作为头部掩码
        if head_mask is None:
            head_mask = jnp.ones((self.config.num_hidden_layers, self.config.num_attention_heads))

        # 如果未提供问题长度并且输入不为空，则计算问题长度
        if question_lengths is None and input_ids is not None:
            # 假设输入格式为：<cls> <question> <sep> context <sep>
            question_lengths = jnp.argmax((input_ids == self.config.sep_token_id).astype("i4"), axis=-1) + 1
            question_lengths = jnp.expand_dims(question_lengths, axis=1)

        # 计算输入张量的序列长度
        seqlen = input_ids.shape[1]

        # 初始化 logits_mask 为 None
        logits_mask = None
        # 如果存在问题长度，则准备问题掩码
        if question_lengths is not None:
            # 将长度为问题的 logits 设置为 `-inf`
            logits_mask = self.prepare_question_mask(question_lengths, seqlen)
            # 如果未提供 token_type_ids，则使用 logits_mask 的反向值
            if token_type_ids is None:
                token_type_ids = (~logits_mask).astype("i4")
            logits_mask = jnp.expand_dims(logits_mask, axis=2)
            logits_mask = logits_mask.at[:, 0].set(False)

        # 如果未提供 token_type_ids，则初始化为与 input_ids 形状相同的全 0 张量
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        # 如果需要处理任何伪随机数生成器（PRNG）
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        if indices_rng is not None:
            rngs["indices"] = indices_rng

        # 调用 self.module 的 apply 方法，传递各种输入参数
        return self.module.apply(
            {"params": params or self.params},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            token_type_ids,
            jnp.array(position_ids, dtype="i4"),
            jnp.array(head_mask, dtype="i4"),
            logits_mask,
            not train,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
        )
    # 定义函数 prepare_question_mask，准备问题的掩码
    def prepare_question_mask(q_lengths, maxlen: int):
        # q_lengths -> (bz, 1)
        # 创建一个长度为 maxlen 的数组 mask，其中包含从 0 到 maxlen-1 的整数
        mask = jnp.arange(0, maxlen)
        # 将 mask 扩展为二维数组，与 q_lengths 比较，生成布尔型掩码
        mask = jnp.expand_dims(mask, axis=0) < q_lengths
        # 返回生成的掩码
        return mask
# 将示例文档字符串添加到指定的模型类中
append_call_sample_docstring(
    FlaxBigBirdForQuestionAnswering,  # 要添加文档字符串的模型类
    _CHECKPOINT_FOR_DOC,  # 用于文档的检查点
    FlaxBigBirdForQuestionAnsweringModelOutput,  # 模型输出类
    _CONFIG_FOR_DOC,  # 用于文档的配置
)


# 定义一个用于语言建模的 BigBird 模型类
class FlaxBigBirdForCausalLMModule(nn.Module):
    config: BigBirdConfig  # BigBird 模型的配置类
    dtype: jnp.dtype = jnp.float32  # 数据类型，默认为 jnp.float32
    gradient_checkpointing: bool = False  # 是否使用梯度检查点，默认为 False

    def setup(self):
        # 初始化 BigBird 模型，不添加池化层
        self.bert = FlaxBigBirdModule(
            config=self.config,
            add_pooling_layer=False,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 初始化仅包含 MLM 头部的 BigBird 模型
        self.cls = FlaxBigBirdOnlyMLMHead(config=self.config, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        token_type_ids: Optional[jnp.ndarray] = None,
        head_mask: Optional[jnp.ndarray] = None,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 模型前向传播
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            init_cache=init_cache,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.tie_word_embeddings:
            # 如果需要共享词嵌入，则获取共享的词嵌入
            shared_embedding = self.bert.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
        else:
            shared_embedding = None

        # 计算预测分数
        logits = self.cls(hidden_states, shared_embedding=shared_embedding)

        if not return_dict:
            return (logits,) + outputs[1:]

        # 返回带有交叉注意力的 FlaxCausalLMOutputWithCrossAttentions 类的输出
        return FlaxCausalLMOutputWithCrossAttentions(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


@add_start_docstrings(
    """
    在 BigBird 模型顶部添加一个语言建模头部的模型（在隐藏状态输出的顶部添加一个线性层），例如用于自回归任务。
    """,
    BIG_BIRD_START_DOCSTRING,  # BigBird 模型的起始文档字符串
)
# 从 transformers.models.bert.modeling_flax_bert.FlaxBertForCausalLM 复制并修改为 BigBird
class FlaxBigBirdForCausalLM(FlaxBigBirdPreTrainedModel):
    module_class = FlaxBigBirdForCausalLMModule  # 使用的模块类
    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        # initializing the cache
        # 获取输入的批次大小和序列长度
        batch_size, seq_length = input_ids.shape

        # 使用 self.init_cache 方法初始化过去的键值对
        past_key_values = self.init_cache(batch_size, max_length)

        # 注意：通常需要为 attention_mask 中大于 input_ids.shape[-1] 和小于 cache_length 的位置放置 0
        # 但由于解码器使用因果 mask，这些位置已经被屏蔽了。
        # 因此，我们可以在这里创建一个静态的 attention_mask，这样更有效率。
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        
        # 如果提供了 attention_mask，则根据其累积求和计算 position_ids
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            # 使用 lax.dynamic_update_slice 更新 extended_attention_mask 的部分值
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            # 否则，使用广播方式创建 position_ids
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        # 返回准备好的输入参数字典
        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        # 更新 model_kwargs 中的 past_key_values 和 position_ids
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs
# 调用函数append_call_sample_docstring，将以下参数传递给它：
# - FlaxBigBirdForCausalLM: 类型，表示为生成样例文档字符串时用到的模型类
# - _CHECKPOINT_FOR_DOC: 常量，表示为生成样例文档字符串时用到的检查点名称
# - FlaxCausalLMOutputWithCrossAttentions: 类型，表示为生成样例文档字符串时用到的模型输出类
# - _CONFIG_FOR_DOC: 常量，表示为生成样例文档字符串时用到的配置信息名称
append_call_sample_docstring(
    FlaxBigBirdForCausalLM,
    _CHECKPOINT_FOR_DOC,
    FlaxCausalLMOutputWithCrossAttentions,
    _CONFIG_FOR_DOC,
)
```