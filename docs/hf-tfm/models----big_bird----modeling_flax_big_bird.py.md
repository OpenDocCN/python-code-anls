# `.\transformers\models\big_bird\modeling_flax_big_bird.py`

```
# 设置文件编码为 UTF-8
# 版权声明：版权归 2021 年的 Google Flax 团队作者和 HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版（"许可证"）授权；除非遵守许可证，否则不得使用此文件
# 您可以在以下网址获取许可证副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件以"原样"提供，不提供任何明示或暗示的保证或条件
# 有关特定语言的许可证，请参阅许可证

# 导入必要的模块和类型
from typing import Callable, Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax

# 导入模型输出相关的类型
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
# 导入模型相关的实用函数和常量
from ...modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
    append_call_sample_docstring,
    append_replace_return_docstrings,
    overwrite_call_docstring,
)
# 导入其他实用工具和日志记录器
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
# 导入 BigBird 配置
from .configuration_big_bird import BigBirdConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的模型检查点
_CHECKPOINT_FOR_DOC = "google/bigbird-roberta-base"
# 用于文档的配置文件
_CONFIG_FOR_DOC = "BigBirdConfig"

# 定义 remat 函数，用于重新计算注意力
remat = nn_partitioning.remat

# 定义 FlaxBigBirdForPreTrainingOutput 类，作为 BigBirdForPreTraining 的输出类型
@flax.struct.dataclass
class FlaxBigBirdForPreTrainingOutput(ModelOutput):
    """
    Output type of [`BigBirdForPreTraining`].
```  
    # 预测逻辑回归：语言建模头的预测分数（SoftMax 前每个词汇标记的分数）
    prediction_logits: jnp.ndarray = None
    # 序列关系逻辑回归：下一个序列预测（分类）头的预测分数（SoftMax 前 True/False 继续的分数）
    seq_relationship_logits: jnp.ndarray = None
    # 隐藏状态：当传递 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回，包含元组的 `jnp.ndarray`（嵌入输出和每个层的输出）的形状 `(batch_size, sequence_length, hidden_size)`
    hidden_states: Optional[Tuple[jnp.ndarray]] = None

    # 注意力：当传递 `output_attentions=True` 或 `config.output_attentions=True` 时返回，包含元组的 `jnp.ndarray`（每层一个）的形状 `(batch_size, num_heads, sequence_length, sequence_length)`
    attentions: Optional[Tuple[jnp.ndarray]] = None
# 定义一个数据类，用于存储问答模型的输出结果
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

    start_logits: jnp.ndarray = None
    end_logits: jnp.ndarray = None
    pooled_output: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None


# 定义一个文档字符串，描述了该模型的特性和使用方法
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
    # 定义函数参数
    Parameters:
        # config参数: 模型配置类，包含模型的所有参数。使用配置文件初始化不会加载与模型相关的权重，只会加载配置。
        # 若要加载模型权重，请参阅`~FlaxPreTrainedModel.from_pretrained`方法。
        config ([`BigBirdConfig`]):
        # dtype参数（可选）：计算的数据类型。可以是`jax.numpy.float32`，`jax.numpy.float16`（在GPU上），以及`jax.numpy.bfloat16`（在TPU上）。
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            # 计算中的数据类型。可以使用`jax.numpy.float32`、`jax.numpy.float16`（在GPU上）、`jax.numpy.bfloat16`（在TPU上）中的任一类型。
            # 可以用于在GPU或TPU上启用混合精度训练或半精度推断。如果指定，所有计算将使用给定的`dtype`执行。
            **注意：这仅指定计算的数据类型，并不影响模型参数的数据类型。**
            # 如果要更改模型参数的数据类型，请参阅`~FlaxPreTrainedModel.to_fp16`和`~FlaxPreTrainedModel.to_bf16`。
"""
# 定义一个文档字符串，用于描述 BigBird 模型的输入参数
BIG_BIRD_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`numpy.ndarray` of shape `({0})`):
            词汇表中输入序列标记的索引。

            可以使用 [`AutoTokenizer`] 获取索引。有关详细信息，请参阅 [`PreTrainedTokenizer.encode`] 和
            [`PreTrainedTokenizer.__call__`]。

            [什么是输入 ID?](../glossary#input-ids)
        attention_mask (`numpy.ndarray` of shape `({0})`, *optional*):
            避免在填充标记索引上执行注意力的掩码。选择 `[0, 1]` 中的掩码值：

            - 对于 **未掩码** 的标记，使用 1，
            - 对于 **已掩码** 的标记，使用 0。

            [什么是注意力掩码?](../glossary#attention-mask)
        token_type_ids (`numpy.ndarray` of shape `({0})`, *optional*):
            指示输入的第一部分和第二部分的段标记索引。索引选择 `[0, 1]`：

            - 0 对应于 *句子 A* 的标记，
            - 1 对应于 *句子 B* 的标记。

            [什么是标记类型 ID?](../glossary#token-type-ids)
        position_ids (`numpy.ndarray` of shape `({0})`, *optional*):
            每个输入序列标记在位置嵌入中的位置索引。在范围 `[0, config.max_position_embeddings - 1]` 中选择。
        head_mask (`numpy.ndarray` of shape `({0})`, `optional):
            空化注意力模块的选定头部的掩码。选择的掩码值在 `[0, 1]` 中：

            - 1 表示头部**未掩码**，
            - 0 表示头部**已掩码**。

        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是普通元组。

"""


class FlaxBigBirdEmbeddings(nn.Module):
    """构建来自词嵌入、位置嵌入和标记类型嵌入的嵌入。"""

    config: BigBirdConfig  # BigBird 模型配置
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

    # 从 transformers.models.bert.modeling_flax_bert.FlaxBertEmbeddings.setup 复制过来的
    # 初始化模型参数
    def setup(self):
        # 初始化词嵌入层
        self.word_embeddings = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        # 初始化位置嵌入层
        self.position_embeddings = nn.Embed(
            self.config.max_position_embeddings,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        # 初始化标记类型嵌入层
        self.token_type_embeddings = nn.Embed(
            self.config.type_vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        # 初始化 LayerNorm 层
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 初始化 Dropout 层
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, input_ids, token_type_ids, position_ids, attention_mask, deterministic: bool = True):
        # 嵌入
        inputs_embeds = self.word_embeddings(input_ids.astype("i4"))
        position_embeds = self.position_embeddings(position_ids.astype("i4"))
        token_type_embeddings = self.token_type_embeddings(token_type_ids.astype("i4"))

        if self.config.rescale_embeddings:
            # 如果需要重新缩放嵌入向量
            inputs_embeds *= self.config.hidden_size**0.5

        # 将所有嵌入向量相加
        hidden_states = inputs_embeds + token_type_embeddings + position_embeds

        # Layer Norm
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states
# 从transformers.models.bert.modeling_flax_bert.FlaxBertSelfAttention复制代码，并将Bert->BigBird
class FlaxBigBirdSelfAttention(nn.Module):
    config: BigBirdConfig
    causal: bool = False
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

    def setup(self):
        # 头维度是隐藏尺寸除以注意力头数
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        # 如果隐藏尺寸不能被注意力头数整除，则引发值错误
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                "`config.hidden_size`: {self.config.hidden_size} 必须是 `config.num_attention_heads` 的倍数 "
                "                   : {self.config.num_attention_heads}"
            )

        # 初始化查询、键和值的Dense层
        self.query = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.key = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.value = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        # 如果是因果的，创建一个因果遮罩
        if self.causal:
            self.causal_mask = make_causal_mask(
                jnp.ones((1, self.config.max_position_embeddings), dtype="bool"), dtype="bool"
            )

    # 将隐藏状态拆分成多头形式
    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.config.num_attention_heads, self.head_dim))

    # 将多头形式的隐藏状态合并
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.config.hidden_size,))

    @nn.compact
    # 从transformers.models.bart.modeling_flax_bart.FlaxBartAttention._concatenate_to_cache中复制
    # 定义一个函数，用于将来自单个输入标记的投影键、值状态与先前步骤的缓存状态连接起来。
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slightly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # 检测是否通过缺少现有缓存数据来初始化。
        is_initialized = self.has_variable("cache", "cached_key")
        # 获取缓存的键
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        # 获取缓存的值
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 获取缓存索引
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        # 如果已经初始化了缓存
        if is_initialized:
            # 获取批次维度、最大长度、头数和每个头的深度
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 更新键、值缓存，使用新的一维空间切片
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            # 更新缓存的键和值
            cached_key.value = key
            cached_value.value = value
            # 计算更新的缓存向量数
            num_updated_cache_vectors = query.shape[1]
            # 更新缓存索引
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 用于缓存的因果掩码: 我们单个查询位置只能参与已生成并缓存的键位置，而不是剩余的零元素。
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            # 结合掩码
            attention_mask = combine_masks(pad_mask, attention_mask)
        # 返回键、值和注意力掩码
        return key, value, attention_mask

    # 实例调用方法，用于执行自注意力机制的计算
    def __call__(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask,
        key_value_states: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic=True,
        output_attentions: bool = False,
    # 定义一个名为 FlaxBigBirdBlockSparseAttention 的类，继承自 nn.Module
    class FlaxBigBirdBlockSparseAttention(nn.Module):
        # 定义类属性 config，类型为 BigBirdConfig
        config: BigBirdConfig
        # 定义类属性 block_sparse_seed，默认值为 None
        block_sparse_seed: int = None
        # 定义类属性 dtype，默认值为 jnp.float32
        dtype: jnp.dtype = jnp.float32

        # 定义 setup 方法
        def setup(self):
            # 初始化 query 层，使用 nn.Dense 类
            self.query = nn.Dense(
                self.config.hidden_size,
                dtype=self.dtype,
                use_bias=self.config.use_bias,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            )
            # 初始化 key 层，使用 nn.Dense 类
            self.key = nn.Dense(
                self.config.hidden_size,
                dtype=self.dtype,
                use_bias=self.config.use_bias,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            )
            # 初始化 value 层，使用 nn.Dense 类
            self.value = nn.Dense(
                self.config.hidden_size,
                dtype=self.dtype,
                use_bias=self.config.use_bias,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            )

        # 定义 transpose_for_scores 静态方法
        @staticmethod
        def transpose_for_scores(x, n_heads, head_size):
            # 重新构造 x 的形状，以便进行矩阵转置
            new_x_shape = x.shape[:-1] + (n_heads, head_size)
            x = x.reshape(*new_x_shape)
            return jnp.transpose(x, axes=(0, 2, 1, 3))

        # 定义 __call__ 方法
        def __call__(
            self,
            hidden_states,
            attention_mask,
            deterministic=True,
            output_attentions=False,
        ):
            # 计算每个头的大小
            n_heads = self.config.num_attention_heads
            head_size = self.config.hidden_size // n_heads

            # 创建用于稀疏注意力的掩码
            blocked_encoder_mask, band_mask, from_mask, to_mask = self.create_masks_for_block_sparse_attn(
                attention_mask, self.config.block_size
            )

            # 对查询、键和值进行转置以便计算注意力分数
            query_layer = self.transpose_for_scores(self.query(hidden_states), n_heads, head_size)
            key_layer = self.transpose_for_scores(self.key(hidden_states), n_heads, head_size)
            value_layer = self.transpose_for_scores(self.value(hidden_states), n_heads, head_size)

            indices_prng_key = None
            # 如果不是确定性的，则生成随机数种子
            if not deterministic:
                indices_prng_key = self.make_rng("indices")

            # 进行大鸟块稀疏注意力计算
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

            # 如果需要输出注意力权重，则将其包含在输出中
            outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
            return outputs

        # 定义静态方法
        @staticmethod
    def create_masks_for_block_sparse_attn(attention_mask, block_size: int):
        # 获取注意力掩码的批量大小和序列长度
        batch_size, seq_length = attention_mask.shape
        # 如果序列长度不能被块大小整除，则引发 ValueError 异常
        if seq_length % block_size != 0:
            raise ValueError(
                f"Sequence length must be multiple of block size, but sequence length is {seq_length}, while block"
                f" size is {block_size}."
            )

        def create_band_mask_from_inputs(from_blocked_mask, to_blocked_mask):
            """
            Create 3D attention mask from a 2D tensor mask.

            Args:
                from_blocked_mask: 2D Tensor of shape [batch_size,
                from_seq_length//from_block_size, from_block_size].
                to_blocked_mask: int32 Tensor of shape [batch_size,
                to_seq_length//to_block_size, to_block_size].

            Returns:
                float Tensor of shape [batch_size, 1, from_seq_length//from_block_size-4, from_block_size,
                3*to_block_size].
            """
            # 将 to_blocked_mask 进行扩展以匹配形状要求
            exp_blocked_to_pad = jnp.concatenate(
                [to_blocked_mask[:, 1:-3], to_blocked_mask[:, 2:-2], to_blocked_mask[:, 3:-1]], axis=2
            )
            # 使用 einsum 创建带状掩码
            band_mask = jnp.einsum("blq,blk->blqk", from_blocked_mask[:, 2:-2], exp_blocked_to_pad)
            # 增加维度以匹配输出形状要求
            band_mask = jnp.expand_dims(band_mask, 1)
            return band_mask

        # 将注意力掩码重塑为块形式
        blocked_encoder_mask = attention_mask.reshape(batch_size, seq_length // block_size, block_size)
        # 创建带状掩码
        band_mask = create_band_mask_from_inputs(blocked_encoder_mask, blocked_encoder_mask)

        # 重塑注意力掩码以匹配不同的形状要求
        from_mask = attention_mask.reshape(batch_size, 1, seq_length, 1)
        to_mask = attention_mask.reshape(batch_size, 1, 1, seq_length)

        return blocked_encoder_mask, band_mask, from_mask, to_mask

    def bigbird_block_sparse_attention(
        self,
        query_layer,
        key_layer,
        value_layer,
        band_mask,
        from_mask,
        to_mask,
        from_blocked_mask,
        to_blocked_mask,
        n_heads,
        head_size,
        indices_prng_key: Optional[jax.random.PRNGKey] = None,
        deterministic: Optional[bool] = True,
        plan_from_length=None,
        plan_num_rand_blocks=None,
        output_attentions=None,
    @staticmethod
    def jax_gather(params, indices, batch_dims=2):
        """
        Gather the indices from params correctly (equivalent to tf.gather but with modifications)

        Args:
            params: (bsz, n_heads, num_blocks, block_size, head_dim)
            indices: (<num_blocks, 1)
        """

        def _jax_gather(params, indices):
            return params[indices]

        # 通过 jax.vmap 对 _jax_gather 进行批处理
        for _ in range(batch_dims):
            _jax_gather = jax.vmap(_jax_gather, in_axes=(0, 0))

        return _jax_gather(params, indices)  # params.shape[:batch_dims] + indices.shape + params.shape[batch_dims+1:]
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
            to_blocked_mask: int32 Tensor of shape [batch_size, to_seq_length//to_block_size, to_block_size].
            broadcasted_rand_attn:
                [batch_size, num_attention_heads, from_seq_length//from_block_size-2, num_rand_blocks]
            num_attention_heads: int. Number of attention heads.
            num_random_blocks: int. Number of random chunks per row.
            batch_size: int. Batch size for computation.
            from_seq_length: int. length of from sequence.
            from_block_size: int. size of block in from sequence.

        Returns:
            float Tensor of shape [batch_size, num_attention_heads, from_seq_length//from_block_size-2,
            from_block_size, num_rand_blocks*to_block_size].
        """
        # 计算窗口数量
        num_windows = from_seq_length // from_block_size - 2
        # 从 to_blocked_mask 中根据 broadcasted_rand_attn 提取随机掩码
        rand_mask = self.jax_gather(to_blocked_mask, broadcasted_rand_attn, batch_dims=1)
        # 重塑随机掩码的形状
        rand_mask = rand_mask.reshape(
            batch_size, num_attention_heads, num_windows, num_random_blocks * from_block_size
        )
        # 使用 einsum 计算注意力掩码
        rand_mask = jnp.einsum("blq,bhlk->bhlqk", from_blocked_mask[:, 1:-1], rand_mask)
        # 返回注意力掩码
        return rand_mask

    @staticmethod
    # 获取随机注意力分布计划
    def _get_rand_attn_plan(from_seq_length, from_block_size, num_rand_blocks):
        """
        Gives the plan of where to put random attention.

        Args:
            from_seq_length: int. length of from sequence.
            from_block_size: int. size of block in from sequence.
            num_rand_blocks: int. Number of random chunks per row.

        Returns:
            plan_from_length: ending location of from block plan_num_rand_blocks: number of random ending location for
            each block
        """

        plan_from_length = []
        plan_num_rand_blocks = []
        # 根据不同条件生成随机注意力分布计划
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

    # 生成大鸟模型的随机掩码
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
    # 生成带有头信息的大鸟模型的随机掩码
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
    # 获取单个块行注意力
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
        """
        对于单行块，获取随机行注意力。

        Args:
            block_id: int。行的块ID。
            to_start_block_id: int。随机注意力列的起始ID。
            to_end_block_id: int。随机注意力列的结束ID。
            num_rand_blocks: int。要选择的随机块数。
            indices_prng_key: jax.random.PRNGKey。用于执行随机jax操作的PRNG密钥。
            window_block_left: int。窗口左侧的块数。
            window_block_right: int。窗口右侧的块数。
            global_block_left: int。全局左侧使用的块数。
            global_block_right: int。全局右侧使用的块数。

        Returns:
            包含大小为num_rand_blocks的随机注意力向量的行。
        """
        # 从中选择随机注意力的to_block列表
        to_block_list = jnp.arange(to_start_block_id, to_end_block_id, dtype=jnp.int32)
        # 对块进行置换
        perm_block = jax.random.permutation(indices_prng_key, to_block_list)

        # 当前块ID的非法块，使用窗口
        illegal_blocks = list(range(block_id - window_block_left, block_id + window_block_right + 1))

        # 在开头和结尾添加块
        illegal_blocks.extend(list(range(global_block_left)))
        illegal_blocks.extend(list(range(to_end_block_id - global_block_right, to_end_block_id)))

        # 第二个from_block不能在倒数第二个to_block上选择随机注意力
        if block_id == 1:
            illegal_blocks.append(to_end_block_id - 2)

        # 倒数第二个from_block不能在第二个to_block上选择随机注意力
        if block_id == to_end_block_id - 2:
            illegal_blocks.append(1)

        selected_random_blocks = []

        for i in range(to_end_block_id - to_start_block_id):
            if perm_block[i] not in illegal_blocks:
                selected_random_blocks.append(perm_block[i])
            if len(selected_random_blocks) == num_rand_blocks:
                break
        return jnp.array(selected_random_blocks, dtype=jnp.int32)
# 从 transformers.models.bert.modeling_flax_bert.FlaxBertSelfOutput 复制并修改为 BigBird
class FlaxBigBirdSelfOutput(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

    def setup(self):
        # 创建一个全连接层，输出维度为配置的隐藏层大小，使用正态分布初始化权重
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 创建一个 LayerNorm 层，用于规范化隐藏层状态
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 创建一个 Dropout 层，用于随机丢弃隐藏状态中的一些元素，以减少过拟合
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, hidden_states, input_tensor, deterministic: bool = True):
        # 将隐藏状态输入到全连接层中
        hidden_states = self.dense(hidden_states)
        # 在隐藏状态上应用 Dropout
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 将规范化后的隐藏状态与输入张量相加
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回规范化后的隐藏状态
        return hidden_states


class FlaxBigBirdAttention(nn.Module):
    config: BigBirdConfig
    layer_id: int = None
    causal: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 根据配置选择不同类型的注意力机制
        if self.config.attention_type == "original_full":
            # 如果是原始全连接类型的注意力机制，则使用 FlaxBigBirdSelfAttention
            self.self = FlaxBigBirdSelfAttention(self.config, causal=self.causal, dtype=self.dtype)
        elif self.config.attention_type == "block_sparse":
            # 如果是块稀疏类型的注意力机制，则使用 FlaxBigBirdBlockSparseAttention
            self.self = FlaxBigBirdBlockSparseAttention(self.config, block_sparse_seed=self.layer_id, dtype=self.dtype)
        else:
            # 若配置中的注意力类型不是支持的类型，则引发 ValueError
            raise ValueError(
                f"Your `config.attention_type` is {self.config.attention_type} but it can either be `original_full` or"
                " `block_sparse`"
            )

        # 创建一个输出层，用于对注意力机制的输出进行处理
        self.output = FlaxBigBirdSelfOutput(self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask,
        key_value_states=None,
        init_cache=False,
        deterministic=True,
        output_attentions: bool = False,
        # Attention mask comes in as attention_mask.shape == (*batch_sizes, kv_length)
        # FLAX expects: attention_mask.shape == (*batch_sizes, 1, 1, kv_length) such that it is broadcastable
        # with attn_weights.shape == (*batch_sizes, num_heads, q_length, kv_length)
        # 如果注意力掩码为 "original_full" 类型，则调用 self 方法计算注意力输出
        if self.config.attention_type == "original_full":
            attn_outputs = self.self(
                hidden_states,
                attention_mask,
                layer_head_mask=layer_head_mask,
                key_value_states=key_value_states,
                init_cache=init_cache,
                deterministic=deterministic,
                output_attentions=output_attentions,
            )
        else:
            # 如果注意力掩码不是 "original_full" 类型，则调用 self 方法计算注意力输出
            attn_outputs = self.self(
                hidden_states,
                attention_mask,
                deterministic=deterministic,
                output_attentions=output_attentions,
            )
        # 获取注意力输出
        attn_output = attn_outputs[0]
        # 将注意力输出传入输出层，并更新隐藏状态
        hidden_states = self.output(attn_output, hidden_states, deterministic=deterministic)

        # 将隐藏状态作为输出
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则将其添加到输出中
        if output_attentions:
            outputs += (attn_outputs[1],)

        # 返回输出
        return outputs
# 从transformers.models.bert.modeling_flax_bert.FlaxBertIntermediate复制并修改为Bert->BigBird
class FlaxBigBirdIntermediate(nn.Module):
    config: BigBirdConfig  # BigBird模型配置
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

    def setup(self):
        # 创建一个全连接层，用于处理中间表示，设置输入大小为配置中的中间尺寸，使用正态分布初始化权重
        self.dense = nn.Dense(
            self.config.intermediate_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 激活函数，根据配置中的隐藏层激活函数选择对中间表示进行激活
        self.activation = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states):
        # 使用全连接层处理隐藏状态（中间表示）
        hidden_states = self.dense(hidden_states)
        # 使用激活函数对处理后的中间表示进行激活
        hidden_states = self.activation(hidden_states)
        return hidden_states


# 从transformers.models.bert.modeling_flax_bert.FlaxBertOutput复制并修改为Bert->BigBird
class FlaxBigBirdOutput(nn.Module):
    config: BigBirdConfig  # BigBird模型配置
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

    def setup(self):
        # 创建一个全连接层，用于处理输出表示，设置输入大小为配置中的隐藏尺寸，使用正态分布初始化权重
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # Dropout层，根据配置中的隐藏层dropout比率对输出表示进行dropout
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        # LayerNorm层，用于对输出表示进行归一化
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(self, hidden_states, attention_output, deterministic: bool = True):
        # 使用全连接层处理隐藏状态（输出表示）
        hidden_states = self.dense(hidden_states)
        # 使用Dropout对输出表示进行dropout
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 对dropout后的表示与注意力输出进行加和，并应用LayerNorm进行归一化
        hidden_states = self.LayerNorm(hidden_states + attention_output)
        return hidden_states


class FlaxBigBirdLayer(nn.Module):
    config: BigBirdConfig  # BigBird模型配置
    layer_id: int = None  # 层的ID
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

    def setup(self):
        # 创建BigBird自注意力层，根据配置设置是否为解码器，以及数据类型
        self.attention = FlaxBigBirdAttention(
            self.config, layer_id=self.layer_id, causal=self.config.is_decoder, dtype=self.dtype
        )
        # 创建BigBird中间层，用于处理自注意力层的输出，设置数据类型
        self.intermediate = FlaxBigBirdIntermediate(self.config, dtype=self.dtype)
        # 创建BigBird输出层，用于处理中间层的输出与自注意力层的输出，设置数据类型
        self.output = FlaxBigBirdOutput(self.config, dtype=self.dtype)
        # 如果配置中有跨注意力，则创建BigBird跨注意力层，设置数据类型
        if self.config.add_cross_attention:
            self.crossattention = FlaxBigBirdAttention(self.config, causal=False, dtype=self.dtype)

    # 从transformers.models.bert.modeling_flax_bert.FlaxBertLayer.__call__复制并修改为Bert->BigBird
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
        # Self Attention
        # 使用注意力机制处理输入隐藏状态，得到注意力输出
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            layer_head_mask=layer_head_mask,
            init_cache=init_cache,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )
        # 获取注意力输出的第一个元素，即自注意力机制的输出
        attention_output = attention_outputs[0]

        # Cross-Attention Block
        # 如果存在编码器的隐藏状态，则进行交叉注意力机制
        if encoder_hidden_states is not None:
            # 使用交叉注意力机制处理注意力输出和编码器的隐藏状态，得到交叉注意力输出
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask=encoder_attention_mask,
                layer_head_mask=layer_head_mask,
                key_value_states=encoder_hidden_states,
                deterministic=deterministic,
                output_attentions=output_attentions,
            )
            # 获取交叉注意力输出的第一个元素
            attention_output = cross_attention_outputs[0]

        # 使用中间层处理注意力输出
        hidden_states = self.intermediate(attention_output)
        # 使用输出层处理中间层的输出和注意力输出，得到最终的隐藏状态
        hidden_states = self.output(hidden_states, attention_output, deterministic=deterministic)

        # 返回最终的隐藏状态
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则将注意力权重添加到输出中
        if output_attentions:
            outputs += (attention_outputs[1],)
            # 如果存在编码器的隐藏状态，则将交叉注意力权重也添加到输出中
            if encoder_hidden_states is not None:
                outputs += (cross_attention_outputs[1],)
        return outputs
class FlaxBigBirdLayerCollection(nn.Module):
    # 定义一个类，用于存储 BigBird 模型的层集合
    config: BigBirdConfig
    # 定义一个 BigBirdConfig 类型的属性 config，用于存储 BigBird 模型的配置信息
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    # 定义一个 jnp.float32 类型的属性 dtype，用于存储计算时的数据类型，默认为 jnp.float32
    gradient_checkpointing: bool = False
    # 定义一个布尔类型的属性 gradient_checkpointing，用于标记是否使用梯度检查点，默认为 False

    def setup(self):
        # 定义一个方法 setup，用于初始化模型的层集合
        if self.gradient_checkpointing:
            # 如果使用梯度检查点
            FlaxBigBirdCheckpointLayer = remat(FlaxBigBirdLayer, static_argnums=(5, 6, 7))
            # 创建一个支持梯度检查点的 BigBird 层
            self.layers = [
                FlaxBigBirdCheckpointLayer(self.config, layer_id=i, name=str(i), dtype=self.dtype)
                for i in range(self.config.num_hidden_layers)
            ]
            # 根据配置信息创建多个支持梯度检查点的 BigBird 层
        else:
            # 如果不使用梯度检查点
            self.layers = [
                FlaxBigBirdLayer(self.config, layer_id=i, name=str(i), dtype=self.dtype)
                for i in range(self.config.num_hidden_layers)
            ]
            # 根据配置信息创建多个 BigBird 层

    # Copied from transformers.models.bert.modeling_flax_bert.FlaxBertLayerCollection.__call__ with Bert->BigBird
    # 从 transformers.models.bert.modeling_flax_bert.FlaxBertLayerCollection.__call__ 复制并修改为 BigBird
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
        # 如果不输出注意力权重，则将 all_attentions 初始化为空元组；否则设为 None
        all_attentions = () if output_attentions else None
        # 如果不输出隐藏状态，则将 all_hidden_states 初始化为空元组；否则设为 None
        all_hidden_states = () if output_hidden_states else None
        # 如果不输出交叉注意力权重或者编码器隐藏状态为空，则将 all_cross_attentions 初始化为空元组；否则设为 None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        # 检查是否正确指定了 head_mask 的层数
        if head_mask is not None:
            # 如果指定的 head_mask 的层数与层数不匹配，则引发 ValueError 异常
            if head_mask.shape[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for                  "
                    f"       {head_mask.shape[0]}."
                )

        # 遍历每一层 Transformer 编码器
        for i, layer in enumerate(self.layers):
            # 如果输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # 调用当前层的 forward 方法，得到当前层的输出
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

            # 更新隐藏状态为当前层的输出的第一个元素（即下一层的输入）
            hidden_states = layer_outputs[0]

            # 如果输出注意力权重，则将当前层的注意力权重添加到 all_attentions
            if output_attentions:
                all_attentions += (layer_outputs[1],)

                # 如果编码器隐藏状态不为空，则将当前层的交叉注意力权重添加到 all_cross_attentions
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # 如果输出隐藏状态，则将最终隐藏状态添加到 all_hidden_states
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 整合所有输出
        outputs = (hidden_states, all_hidden_states, all_attentions, all_cross_attentions)

        # 如果不使用字典返回，则将输出中为空的部分去除后返回
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        # 使用 FlaxBaseModelOutputWithPastAndCrossAttentions 封装输出并返回
        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )
# 从transformers.models.bert.modeling_flax_bert.FlaxBertEncoder复制并将Bert->BigBird
class FlaxBigBirdEncoder(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型
    gradient_checkpointing: bool = False

    def setup(self):
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


# 从transformers.models.bert.modeling_flax_bert.FlaxBertPredictionHeadTransform复制并将Bert->BigBird
class FlaxBigBirdPredictionHeadTransform(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dense = nn.Dense(self.config.hidden_size, dtype=self.dtype)
        self.activation = ACT2FN[self.config.hidden_act]
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return self.LayerNorm(hidden_states)


# 从transformers.models.bert.modeling_flax_bert.FlaxBertLMPredictionHead复制并将Bert->BigBird, np.ndarray->jnp.ndarray
class FlaxBigBirdLMPredictionHead(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32
    bias_init: Callable[..., jnp.ndarray] = jax.nn.initializers.zeros

    def setup(self):
        self.transform = FlaxBigBirdPredictionHeadTransform(self.config, dtype=self.dtype)
        self.decoder = nn.Dense(self.config.vocab_size, dtype=self.dtype, use_bias=False)
        self.bias = self.param("bias", self.bias_init, (self.config.vocab_size,))
    # 定义一个类的调用方法，接受隐藏状态和共享嵌入作为参数
    def __call__(self, hidden_states, shared_embedding=None):
        # 对隐藏状态进行转换
        hidden_states = self.transform(hidden_states)

        # 如果存在共享嵌入，则使用共享嵌入进行解码
        if shared_embedding is not None:
            hidden_states = self.decoder.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
        else:
            # 否则直接使用解码器对隐藏状态进行解码
            hidden_states = self.decoder(hidden_states)

        # 将偏置转换为与数据类型相匹配的JAX数组，并添加到隐藏状态中
        bias = jnp.asarray(self.bias, self.dtype)
        hidden_states += bias
        # 返回处理后的隐藏状态
        return hidden_states
# 从transformers.models.bert.modeling_flax_bert.FlaxBertOnlyMLMHead中复制并修改为BigBird
class FlaxBigBirdOnlyMLMHead(nn.Module):
    # 类变量config，指定为BigBirdConfig类型
    config: BigBirdConfig
    # 类变量dtype，默认为jnp.float32类型
    dtype: jnp.dtype = jnp.float32

    # 模块初始化方法
    def setup(self):
        # 实例化FlaxBigBirdLMPredictionHead对象，并赋值给self.predictions
        self.predictions = FlaxBigBirdLMPredictionHead(self.config, dtype=self.dtype)

    # 模块调用方法
    def __call__(self, hidden_states, shared_embedding=None):
        # 调用self.predictions进行预测，得到预测结果，并返回
        hidden_states = self.predictions(hidden_states, shared_embedding=shared_embedding)
        return hidden_states


# 从transformers.models.bert.modeling_flax_bert.FlaxBertPreTrainingHeads中复制并修改为BigBird
class FlaxBigBirdPreTrainingHeads(nn.Module):
    # 类变量config，指定为BigBirdConfig类型
    config: BigBirdConfig
    # 类变量dtype，默认为jnp.float32类型
    dtype: jnp.dtype = jnp.float32

    # 模块初始化方法
    def setup(self):
        # 实例化FlaxBigBirdLMPredictionHead对象，并赋值给self.predictions
        self.predictions = FlaxBigBirdLMPredictionHead(self.config, dtype=self.dtype)
        # 实例化nn.Dense对象，用于计算seq_relationship_score
        self.seq_relationship = nn.Dense(2, dtype=self.dtype)

    # 模块调用方法
    def __call__(self, hidden_states, pooled_output, shared_embedding=None):
        # 调用self.predictions进行预测，得到预测结果
        prediction_scores = self.predictions(hidden_states, shared_embedding=shared_embedding)
        # 调用self.seq_relationship计算seq_relationship_score
        seq_relationship_score = self.seq_relationship(pooled_output)
        # 返回预测结果和seq_relationship_score
        return prediction_scores, seq_relationship_score


# 从transformers.models.bert.modeling_flax_bert.FlaxBertPreTrainedModel中复制并修改为BigBird
class FlaxBigBirdPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    # 类变量config_class，指定为BigBirdConfig类型
    config_class = BigBirdConfig
    # 类变量base_model_prefix，指定为"bert"
    base_model_prefix = "bert"
    # 类变量module_class，默认为None，需要在子类中指定具体模块类
    module_class: nn.Module = None

    # 模块初始化方法
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
        # 根据配置创建模块对象
        module = self.module_class(config=config, dtype=dtype, gradient_checkpointing=gradient_checkpointing, **kwargs)
        # 如果配置中attention_type为"block_sparse"且input_shape为None，则设置input_shape为(1, 12 * config.block_size)，否则为(1, 1)
        if config.attention_type == "block_sparse" and input_shape is None:
            input_shape = (1, 12 * config.block_size)
        elif input_shape is None:
            input_shape = (1, 1)

        # 调用父类构造方法初始化模型
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    # 从transformers.models.bert.modeling_flax_bert.FlaxBertPreTrainedModel中复制并修改为BigBird
    # 方法用于启用梯度检查点
    def enable_gradient_checkpointing(self):
        self._module = self.module_class(
            config=self.config,
            dtype=self.dtype,
            gradient_checkpointing=True,
        )
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化权重函数，用于初始化模型参数
        # 初始化输入张量
        input_ids = jnp.zeros(input_shape, dtype="i4")  # 初始化输入张量为零矩阵
        token_type_ids = jnp.zeros_like(input_ids)  # 初始化token类型张量为与输入张量相同形状的零矩阵
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)  # 根据输入张量形状广播位置张量
        attention_mask = jnp.ones_like(input_ids)  # 初始化注意力掩码张量为与输入张量相同形状的全一矩阵
        head_mask = jnp.ones((self.config.num_hidden_layers, self.config.num_attention_heads))  # 初始化头掩码张量为全一矩阵

        params_rng, dropout_rng, indices_rng = jax.random.split(rng, num=3)  # 使用随机种子分割成三个子种子
        rngs = {"params": params_rng, "dropout": dropout_rng, "indices": indices_rng}  # 将子种子放入字典中

        if self.config.add_cross_attention:
            # 如果模型配置中包含跨注意力，初始化encoder_hidden_states和encoder_attention_mask
            encoder_hidden_states = jnp.zeros(input_shape + (self.config.hidden_size,))
            encoder_attention_mask = attention_mask
            # 使用模型的init函数初始化模型参数
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
            # 否则只使用模型的init函数初始化模型参数
            module_init_outputs = self.module.init(
                rngs,
                input_ids,
                attention_mask,
                token_type_ids,
                position_ids,
                head_mask,
                return_dict=False,
            )

        random_params = module_init_outputs["params"]  # 从初始化输出中获取随机参数

        if params is not None:
            # 如果提供了预训练参数，将随机参数与预训练参数进行融合
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()  # 清空缺失键的集合
            return freeze(unflatten_dict(params))  # 返回融合后的参数
        else:
            return random_params  # 返回随机初始化的参数

    # 从transformers.models.bart.modeling_flax_bart.FlaxBartDecoderPreTrainedModel.init_cache中复制过来的函数
    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                用于快速自回归解码的批量大小。定义了初始化缓存时的批量大小。
            max_length (`int`):
                自回归解码的最大可能长度。定义了初始化缓存时的序列长度。
        """
        # 初始化用于检索缓存的输入变量
        input_ids = jnp.ones((batch_size, max_length), dtype="i4")  # 初始化输入ID为全一矩阵
        attention_mask = jnp.ones_like(input_ids, dtype="i4")  # 初始化注意力掩码为与输入ID相同形状的全一矩阵
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)  # 广播位置ID
        # 使用模型的init函数初始化模型参数，同时初始化缓存
        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        return unfreeze(init_variables["cache"])  # 返回初始化的缓存
    # 使用装饰器为模型前向传播函数添加文档字符串，文档字符串使用 BIG_BIRD_INPUTS_DOCSTRING 格式，参数为批量大小和序列长度
    @add_start_docstrings_to_model_forward(BIG_BIRD_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 定义模型的前向传播函数，接受一系列输入参数
    def __call__(
        self,
        # 输入的 token IDs，表示输入序列中每个 token 的编码
        input_ids,
        # 注意力掩码，指定哪些位置的 token 参与注意力计算
        attention_mask=None,
        # token 类型 IDs，用于区分输入序列中不同的句子或片段
        token_type_ids=None,
        # 位置 IDs，指定输入序列中每个 token 的位置信息
        position_ids=None,
        # 头部掩码，用于指定哪些注意力头应该被屏蔽
        head_mask=None,
        # 编码器的隐藏状态，用于实现 encoder-decoder 架构
        encoder_hidden_states=None,
        # 编码器注意力掩码，指定编码器中哪些位置的 token 参与注意力计算
        encoder_attention_mask=None,
        # 参数字典，用于传递额外的模型参数
        params: dict = None,
        # 随机数生成器用于执行 dropout 操作
        dropout_rng: Optional[jax.random.PRNGKey] = None,
        # 随机数生成器用于执行索引操作
        indices_rng: Optional[jax.random.PRNGKey] = None,
        # 是否处于训练模式
        train: bool = False,
        # 是否输出注意力权重
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典形式的输出
        return_dict: Optional[bool] = None,
        # 过去键值对，用于存储过去的注意力值，用于实现缓存
        past_key_values: dict = None,
# 定义 FlaxBigBirdModule 类，继承自 nn.Module
class FlaxBigBirdModule(nn.Module):
    # 声明 config 属性为 BigBirdConfig 类型
    config: BigBirdConfig
    # 设置 dtype 属性为 jnp.float32，表示计算的数据类型，默认为 float32
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    # 设置 add_pooling_layer 属性为 True，表示是否添加池化层，默认为 True
    add_pooling_layer: bool = True
    # 设置 gradient_checkpointing 属性为 False，表示是否使用梯度检查点，默认为 False
    gradient_checkpointing: bool = False

    # 初始化方法
    def setup(self):
        # 创建 FlaxBigBirdEmbeddings 实例，传入配置参数和数据类型
        self.embeddings = FlaxBigBirdEmbeddings(self.config, dtype=self.dtype)
        # 创建 FlaxBigBirdEncoder 实例，传入配置参数、数据类型和梯度检查点参数
        self.encoder = FlaxBigBirdEncoder(
            self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing
        )
        # 创建 nn.Dense 实例作为池化层，设置输出大小、初始化方式和数据类型
        self.pooler = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )

    # 调用方法，接收多个输入参数
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
        # 将输入序列转换为嵌入向量
        hidden_states = self.embeddings(
            input_ids, token_type_ids, position_ids, attention_mask, deterministic=deterministic
        )
        # 使用编码器对嵌入向量进行编码
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
        # 获取编码器的输出隐藏状态
        hidden_states = outputs[0]

        # 如果需要添加池化层
        pooled = nn.tanh(self.pooler(hidden_states[:, 0, :])) if self.add_pooling_layer else None

        # 如果不需要返回字典
        if not return_dict:
            # 如果池化结果为空，则不返回池化结果
            if pooled is None:
                return (hidden_states,) + outputs[1:]
            # 否则返回隐藏状态和池化结果
            return (hidden_states, pooled) + outputs[1:]

        # 返回带池化和交叉注意力的基础模型输出
        return FlaxBaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


# 为 FlaxBigBirdModel 类添加文档字符串，描述其功能和使用方式
@add_start_docstrings(
    "The bare BigBird Model transformer outputting raw hidden-states without any specific head on top.",
    BIG_BIRD_START_DOCSTRING,
)
# 从 transformers.models.bert.modeling_flax_bert.FlaxBertModel 复制代码并修改为 BigBird
# 作为 FlaxBigBirdModel 的父类
class FlaxBigBirdModel(FlaxBigBirdPreTrainedModel):
    module_class = FlaxBigBirdModule


# 从 transformers.models.bert.modeling_flax_bert.FlaxBertForPreTrainingModule 复制代码并修改为 BigBird
# 定义 FlaxBigBirdForPreTrainingModule 类，继承自 nn.Module
class FlaxBigBirdForPreTrainingModule(nn.Module):
    # 定义类的属性，BigBirdConfig类型的config，jnp.float32类型的dtype，默认为False的bool类型的gradient_checkpointing
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    # 初始化方法，创建FlaxBigBirdModule和FlaxBigBirdPreTrainingHeads对象
    def setup(self):
        self.bert = FlaxBigBirdModule(
            config=self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        self.cls = FlaxBigBirdPreTrainingHeads(config=self.config, dtype=self.dtype)

    # 调用方法，接收多个参数，进行模型推理
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
        # 调用self.bert进行模型推理，获取输出
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

        # 根据配置判断是否共享词嵌入
        if self.config.tie_word_embeddings:
            shared_embedding = self.bert.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
        else:
            shared_embedding = None

        # 获取隐藏状态和池化输出
        hidden_states = outputs[0]
        pooled_output = outputs[1]

        # 调用self.cls进行预测
        prediction_scores, seq_relationship_score = self.cls(
            hidden_states, pooled_output, shared_embedding=shared_embedding
        )

        # 如果不返回字典，则返回元组
        if not return_dict:
            return (prediction_scores, seq_relationship_score) + outputs[2:]

        # 返回FlaxBigBirdForPreTrainingOutput对象
        return FlaxBigBirdForPreTrainingOutput(
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 添加模型文档字符串，描述 BigBird 模型的两个头部：掩码语言建模头和下一个句子预测（分类）头
@add_start_docstrings(
    """
    BigBird Model with two heads on top as done during the pretraining: a `masked language modeling` head and a `next
    sentence prediction (classification)` head.
    """,
    BIG_BIRD_START_DOCSTRING,
)
# 从 transformers.models.bert.modeling_flax_bert.FlaxBertForPreTraining 复制并将 Bert 替换为 BigBird
class FlaxBigBirdForPreTraining(FlaxBigBirdPreTrainedModel):
    module_class = FlaxBigBirdForPreTrainingModule


# BigBird 预训练模型的文档字符串
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

# 覆盖 FlaxBigBirdForPreTraining 的调用文档字符串
overwrite_call_docstring(
    FlaxBigBirdForPreTraining,
    BIG_BIRD_INPUTS_DOCSTRING.format("batch_size, sequence_length") + FLAX_BIG_BIRD_FOR_PRETRAINING_DOCSTRING,
)
# 追加替换 FlaxBigBirdForPreTraining 的返回文档字符串
append_replace_return_docstrings(
    FlaxBigBirdForPreTraining, output_type=FlaxBigBirdForPreTrainingOutput, config_class=_CONFIG_FOR_DOC
)


# 从 transformers.models.bert.modeling_flax_bert.FlaxBertForMaskedLMModule 复制并将 Bert 替换为 BigBird
class FlaxBigBirdForMaskedLMModule(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        # 初始化 BigBird 模块和仅包含 MLM 头的 BigBird 模块
        self.bert = FlaxBigBirdModule(
            config=self.config,
            add_pooling_layer=False,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
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
        # Model
        # 调用 BERT 模型，传入输入的各种参数
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

        # 获取 BERT 模型输出的隐藏状态
        hidden_states = outputs[0]
        
        # 如果配置了共享词嵌入，则获取共享的词嵌入
        if self.config.tie_word_embeddings:
            shared_embedding = self.bert.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
        else:
            shared_embedding = None

        # 计算预测分数
        logits = self.cls(hidden_states, shared_embedding=shared_embedding)

        # 如果不需要返回字典，则返回预测分数和其他输出
        if not return_dict:
            return (logits,) + outputs[1:]

        # 返回 FlaxMaskedLMOutput 对象，包含预测分数、隐藏状态和注意力权重
        return FlaxMaskedLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings("""BigBird Model with a `language modeling` head on top.""", BIG_BIRD_START_DOCSTRING)
# 从transformers.models.bert.modeling_flax_bert.FlaxBertForMaskedLM复制并修改为Bert->BigBird
# 这个类定义了带有语言建模头部的BigBird模型
class FlaxBigBirdForMaskedLM(FlaxBigBirdPreTrainedModel):
    module_class = FlaxBigBirdForMaskedLMModule

# 添加调用示例的文档字符串
append_call_sample_docstring(FlaxBigBirdForMaskedLM, _CHECKPOINT_FOR_DOC, FlaxMaskedLMOutput, _CONFIG_FOR_DOC)

# 定义用于句子级分类任务的头部
class FlaxBigBirdClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dense = nn.Dense(self.config.hidden_size, dtype=self.dtype)
        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Dense(self.config.num_labels, dtype=self.dtype)

    def __call__(self, features, deterministic=True):
        # 取特征的第一个token (<s>)，相当于 [CLS]
        x = features[:, 0, :]
        x = self.dropout(x, deterministic=deterministic)
        x = self.dense(x)
        x = ACT2FN[self.config.hidden_act](x)
        x = self.dropout(x, deterministic=deterministic)
        x = self.out_proj(x)
        return x

# 定义用于序列分类的模块
class FlaxBigBirdForSequenceClassificationModule(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        # 实例化BigBird模型和分类头部
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

        sequence_output = outputs[0]
        # 使用分类头部得到logits
        logits = self.classifier(sequence_output, deterministic=deterministic)

        if not return_dict:
            return (logits,) + outputs[2:]

        # 返回分类器输出以及可能的隐藏状态和注意力
        return FlaxSequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

@add_start_docstrings(
    """
    BigBird Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    # 创建一个多头注意力池化器，用于生成汇总输出，例如用于 GLUE 任务
    BIG_BIRD_START_DOCSTRING,
# 从transformers.models.bert.modeling_flax_bert.FlaxBertForSequenceClassification复制并将Bert->BigBird
class FlaxBigBirdForSequenceClassification(FlaxBigBirdPreTrainedModel):
    module_class = FlaxBigBirdForSequenceClassificationModule

# 添加调用示例的文档字符串
append_call_sample_docstring(
    FlaxBigBirdForSequenceClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxSequenceClassifierOutput,
    _CONFIG_FOR_DOC,
)

# 从transformers.models.bert.modeling_flax_bert.FlaxBertForMultipleChoiceModule复制并将Bert->BigBird
class FlaxBigBirdForMultipleChoiceModule(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        self.bert = FlaxBigBirdModule(
            config=self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
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
        num_choices = input_ids.shape[1]
        input_ids = input_ids.reshape(-1, input_ids.shape[-1]) if input_ids is not None else None
        attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1]) if attention_mask is not None else None
        token_type_ids = token_type_ids.reshape(-1, token_type_ids.shape[-1]) if token_type_ids is not None else None
        position_ids = position_ids.reshape(-1, position_ids.shape[-1]) if position_ids is not None else None

        # 模型
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

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)
        logits = self.classifier(pooled_output)

        reshaped_logits = logits.reshape(-1, num_choices)

        if not return_dict:
            return (reshaped_logits,) + outputs[2:]

        return FlaxMultipleChoiceModelOutput(
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# 添加起始文档字符串
@add_start_docstrings(
    """
    BigBird模型，顶部带有多选分类头（池化输出之上的线性层和softmax），例如RocStories/SWAG任务。
    """,
    BIG_BIRD_START_DOCSTRING,
)
class FlaxBigBirdForMultipleChoice(FlaxBigBirdPreTrainedModel):
    module_class = FlaxBigBirdForMultipleChoiceModule
    # 初始化函数，用于创建 BigBirdForPreTraining 类的实例
    def __init__(
        self,
        # BigBird 模型的配置对象，指定模型的各项参数
        config: BigBirdConfig,
        # 输入数据的形状，可选参数，默认为 None
        input_shape: Optional[tuple] = None,
        # 随机种子，用于初始化模型参数，默认为 0
        seed: int = 0,
        # 数据类型，默认为 jnp.float32
        dtype: jnp.dtype = jnp.float32,
        # 是否进行初始化，默认为 True
        _do_init: bool = True,
        # 其他关键字参数
        **kwargs,
    ):
        # 如果注意力类型为 "block_sparse" 且输入形状为 None，则指定输入形状为 (1, 1, 12 * config.block_size)
        if config.attention_type == "block_sparse" and input_shape is None:
            input_shape = (1, 1, 12 * config.block_size)
        # 如果输入形状为 None，则指定输入形状为 (1, 1)
        elif input_shape is None:
            input_shape = (1, 1)
        # 调用父类的初始化方法，传入参数配置、输入形状、随机种子、数据类型、是否初始化等参数
        super().__init__(config, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)
# 覆盖指定模型的调用文档字符串，格式化指定的输入说明信息，如批量大小、选择个数、序列长度
overwrite_call_docstring(
    FlaxBigBirdForMultipleChoice, BIG_BIRD_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
)
# 添加指定模型的调用示例文档字符串，引用检查点、模型输出、配置信息
append_call_sample_docstring(
    FlaxBigBirdForMultipleChoice,
    _CHECKPOINT_FOR_DOC,
    FlaxMultipleChoiceModelOutput,
    _CONFIG_FOR_DOC,
)

# 从transformers.models.bert.modeling_flax_bert.FlaxBertForTokenClassificationModule中复制代码，并将Bert替换为BigBird
class FlaxBigBirdForTokenClassificationModule(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        # 初始化BigBird模块，设置配置、数据类型、是否添加池化层、是否启用梯度检查点
        self.bert = FlaxBigBirdModule(
            config=self.config,
            dtype=self.dtype,
            add_pooling_layer=False,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 根据配置设定分类器的丢弃率，若未指定则使用隐藏层的丢弃率
        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        # 初始化丢弃层
        self.dropout = nn.Dropout(rate=classifier_dropout)
        # 初始化分类器，输出维度为标签数目
        self.classifier = nn.Dense(self.config.num_labels, dtype=self.dtype)

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

        # 获取隐藏状态并进行丢弃处理
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 获取分类器的输出（对隐藏状态进行线性变换）
        logits = self.classifier(hidden_states)

        # 若不返回字典，则返回元组
        if not return_dict:
            return (logits,) + outputs[1:]

        # 返回分类器输出，同时保留隐藏状态和注意力权重（如果有）
        return FlaxTokenClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

@add_start_docstrings(
    """
    BigBird模型的标记分类头部（线性层叠加在隐藏状态输出之上），例如用于命名实体识别（NER）任务。
    """,
    BIG_BIRD_START_DOCSTRING,
)
# 从transformers.models.bert.modeling_flax_bert.FlaxBertForTokenClassification中复制代码，并将Bert替换为BigBird
class FlaxBigBirdForTokenClassification(FlaxBigBirdPreTrainedModel):
    module_class = FlaxBigBirdForTokenClassificationModule

# 添加指定模型的调用示例文档字符串，引用检查点、标记分类器输出、配置信息
append_call_sample_docstring(
    FlaxBigBirdForTokenClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxTokenClassifierOutput,
    _CONFIG_FOR_DOC,
)

# BigBird模型的问答头部
class FlaxBigBirdForQuestionAnsweringHead(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32
    # 在模型初始化时设置各个模块
    def setup(self):
        # 初始化 dropout 模块，使用给定的隐藏层 dropout 概率
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        # 初始化中间层模块，使用 FlaxBigBirdIntermediate 类
        self.intermediate = FlaxBigBirdIntermediate(self.config, dtype=self.dtype)
        # 初始化输出层模块，使用 FlaxBigBirdOutput 类
        self.output = FlaxBigBirdOutput(self.config, dtype=self.dtype)
        # 初始化输出层的全连接层模块，设置输出的维度为预测的标签数量
        self.qa_outputs = nn.Dense(self.config.num_labels, dtype=self.dtype)

    # 模型调用时的前向传播过程
    def __call__(self, encoder_output, deterministic=True):
        # 对 encoder_output 进行 dropout 处理，根据 deterministic 参数确定是否使用确定性的 dropout
        hidden_states = self.dropout(encoder_output, deterministic=deterministic)
        # 将处理后的隐藏状态输入到中间层模块
        hidden_states = self.intermediate(hidden_states)
        # 将中间层模块的输出输入到输出层模块，并且传入原始的 encoder_output
        hidden_states = self.output(hidden_states, encoder_output)
        # 将输出层模块的输出输入到全连接层模块
        hidden_states = self.qa_outputs(hidden_states)
        # 返回模型的最终输出
        return hidden_states
# 定义了一个用于提问与回答任务的 FlaxBigBird 模型，包含了一个用于抽取式问答任务（如 SQuAD）的跨度分类头部
class FlaxBigBirdForQuestionAnsweringModule(nn.Module):
    # BigBird 模型的配置信息
    config: BigBirdConfig
    # 数据类型，默认为 32 位浮点数
    dtype: jnp.dtype = jnp.float32
    # 是否添加池化层，默认为 False
    add_pooling_layer: bool = False
    # 是否使用梯度检查点，默认为 False
    gradient_checkpointing: bool = False

    # 初始化模型
    def setup(self):
        # 设置类别数量为 2（用于二分类任务）
        self.config.num_labels = 2
        # 初始化 BigBird 模型
        self.bert = FlaxBigBirdModule(
            self.config,
            dtype=self.dtype,
            add_pooling_layer=self.add_pooling_layer,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # 初始化用于提问与回答任务的分类头部
        self.qa_classifier = FlaxBigBirdForQuestionAnsweringHead(self.config, dtype=self.dtype)

    # 调用模型进行前向传播
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

        # 获取隐藏状态
        hidden_states = outputs[0]
        # 如果添加了池化层，则获取池化后的输出
        pooled_output = outputs[1] if self.add_pooling_layer else None
        # 通过分类头部获取 logits
        logits = self.qa_classifier(hidden_states, deterministic=deterministic)

        # 如果提供了 logits_mask，则对 logits 进行掩码操作
        if logits_mask is not None:
            # 从竞争中移除问题标记
            logits = logits - logits_mask * 1e6

        # 将 logits 分割为起始和结束 logits
        start_logits, end_logits = logits.split(self.config.num_labels, axis=-1)
        # 压缩 logits 的最后一个维度
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # 如果不返回字典，则返回元组形式的结果
        if not return_dict:
            return (start_logits, end_logits) + outputs[1:]

        # 返回字典形式的结果
        return FlaxBigBirdForQuestionAnsweringModelOutput(
            start_logits=start_logits,
            end_logits=end_logits,
            pooled_output=pooled_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# 添加模型文档字符串
@add_start_docstrings(
    """
    BigBird 模型，顶部带有用于抽取式问答任务（如 SQuAD）的跨度分类头部（在隐藏状态输出的线性层上计算 `span start logits` 和 `span end logits`）。
    """,
    BIG_BIRD_START_DOCSTRING,
)
class FlaxBigBirdForQuestionAnswering(FlaxBigBirdPreTrainedModel):
    # 模型类
    module_class = FlaxBigBirdForQuestionAnsweringModule

    # 将模型前向传播的文档字符串添加到模型上
    @add_start_docstrings_to_model_forward(BIG_BIRD_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 定义一个调用函数，接受多个参数
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
        # 设置输出注意力的标志，如果未指定则使用配置中的值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置输出隐藏状态的标志，如果未指定则使用配置中的值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置返回字典的标志，如果未指定则使用配置中的值
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 如果未提供位置 ID，则根据输入 ID 的形状广播生成位置 ID
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        # 如果未提供注意力掩码，则生成一个全为1的掩码
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # 如果未提供头掩码，则生成一个全为1的头掩码
        if head_mask is None:
            head_mask = jnp.ones((self.config.num_hidden_layers, self.config.num_attention_heads))

        # 如果未提供问题长度并且提供了输入 ID，则根据分隔符位置计算问题长度
        if question_lengths is None and input_ids is not None:
            # 假设输入 ID 格式为：<cls> <question> <sep> context <sep>
            question_lengths = jnp.argmax((input_ids == self.config.sep_token_id).astype("i4"), axis=-1) + 1
            question_lengths = jnp.expand_dims(question_lengths, axis=1)

        # 获取序列长度
        seqlen = input_ids.shape[1]

        # 初始化 logits 掩码为 None
        logits_mask = None
        if question_lengths is not None:
            # 将长度 logits 设置为 `-inf`
            logits_mask = self.prepare_question_mask(question_lengths, seqlen)
            if token_type_ids is None:
                token_type_ids = (~logits_mask).astype("i4")
            logits_mask = jnp.expand_dims(logits_mask, axis=2)
            logits_mask = logits_mask.at[:, 0].set(False)

        # 如果未提供 token 类型 ID，则初始化为全为0的数组
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        # 处理需要的 PRNG
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        if indices_rng is not None:
            rngs["indices"] = indices_rng

        # 调用模块的 apply 方法，传入参数
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

    @staticmethod
    # 准备问题掩码函数，用于生成问题长度的掩码
    def prepare_question_mask(q_lengths, maxlen: int):
        # 创建一个长度为最大长度的张量，包含从0到maxlen-1的整数
        mask = jnp.arange(0, maxlen)
        # 在第0维度上扩展张量，变成形状为(1, maxlen)的张量
        mask = jnp.expand_dims(mask, axis=0) < q_lengths
        # 返回生成的掩码张量，掩码的每个元素表示该位置是否需要被保留
        return mask
# 调用函数，向模型类中添加示例函数的文档字符串
append_call_sample_docstring(
    FlaxBigBirdForQuestionAnswering,  # 要添加文档字符串的模型类
    _CHECKPOINT_FOR_DOC,  # 检查点路径的文档字符串
    FlaxBigBirdForQuestionAnsweringModelOutput,  # 模型输出的文档字符串
    _CONFIG_FOR_DOC,  # 配置的文档字符串
)

# 定义一个 FlaxBigBirdForCausalLMModule 类
class FlaxBigBirdForCausalLMModule(nn.Module):
    # 类属性，存储 BigBirdConfig 对象
    config: BigBirdConfig
    # 类属性，数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 类属性，梯度检查点，默认为 False
    gradient_checkpointing: bool = False

    # 初始化函数
    def setup(self):
        # 创建一个 FlaxBigBirdModule 对象，用于处理 BigBird 模型
        self.bert = FlaxBigBirdModule(
            config=self.config,  # BigBird 模型的配置
            add_pooling_layer=False,  # 是否添加池化层，默认为 False
            dtype=self.dtype,  # 数据类型
            gradient_checkpointing=self.gradient_checkpointing,  # 梯度检查点
        )
        # 创建一个 FlaxBigBirdOnlyMLMHead 对象，用于处理 Masked Language Modeling 任务
        self.cls = FlaxBigBirdOnlyMLMHead(config=self.config, dtype=self.dtype)

    # 调用函数
    def __call__(
        self,
        input_ids,  # 输入 token 的 ID
        attention_mask,  # 注意力遮罩
        position_ids,  # 位置 ID
        token_type_ids: Optional[jnp.ndarray] = None,  # token 类型的 ID，可选
        head_mask: Optional[jnp.ndarray] = None,  # 头部遮罩，可选
        encoder_hidden_states: Optional[jnp.ndarray] = None,  # 编码器的隐藏状态，可选
        encoder_attention_mask: Optional[jnp.ndarray] = None,  # 编码器的注意力遮罩，可选
        init_cache: bool = False,  # 是否初始化缓存，默认为 False
        deterministic: bool = True,  # 是否确定性，默认为 True
        output_attentions: bool = False,  # 是否输出注意力权重，默认为 False
        output_hidden_states: bool = False,  # 是否输出隐藏状态，默认为 False
        return_dict: bool = True,  # 是否返回字典，默认为 True
    ):
        # 模型计算
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

        # 获取隐藏状态
        hidden_states = outputs[0]
        # 如果配置了词嵌入共享，则获取共享的嵌入层
        if self.config.tie_word_embeddings:
            shared_embedding = self.bert.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
        else:
            shared_embedding = None

        # 计算预测得分
        logits = self.cls(hidden_states, shared_embedding=shared_embedding)

        # 如果不返回字典，则返回元组
        if not return_dict:
            return (logits,) + outputs[1:]

        # 返回包含交叉注意力的 FlaxCausalLMOutputWithCrossAttentions 对象
        return FlaxCausalLMOutputWithCrossAttentions(
            logits=logits,  # 预测得分
            hidden_states=outputs.hidden_states,  # 隐藏状态
            attentions=outputs.attentions,  # 注意力权重
            cross_attentions=outputs.cross_attentions,  # 交叉注意力权重
        )


# 为 FlaxBigBirdForCausalLM 类添加文档字符串
@add_start_docstrings(
    """
    BigBird Model with a language modeling head on top (a linear layer on top of the hidden-states output) e.g for
    autoregressive tasks.
    """,
    BIG_BIRD_START_DOCSTRING,  # 大鸟模型的基本文档字符串
)
# 从 transformers.models.bert.modeling_flax_bert.FlaxBertForCausalLM 复制并修改为 BigBird
class FlaxBigBirdForCausalLM(FlaxBigBirdPreTrainedModel):
    # 模块类为 FlaxBigBirdForCausalLMModule
    module_class = FlaxBigBirdForCausalLMModule
    # 为生成准备输入，根据输入的input_ids、max_length和attention_mask（可选）初始化缓存
    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        # 获取输入的批量大小和序列长度
        batch_size, seq_length = input_ids.shape

        # 初始化缓存，获取过去的键值对
        past_key_values = self.init_cache(batch_size, max_length)
        
        # 注意：通常情况下，需要在attention_mask中对超出input_ids.shape[-1]和小于cache_length的位置填充0。
        # 但由于解码器使用的是因果mask，这些位置已经被mask了。
        # 因此，我们可以在这里创建一个固定的attention_mask，这对编译更有效率
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            # 计算位置ID，注意用cumsum来获取每个位置的累积和并减去1
            position_ids = attention_mask.cumsum(axis=-1) - 1
            # 使用lax.dynamic_update_slice将attention_mask更新到extended_attention_mask中
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            # 如果没有提供attention_mask，则默认为序列长度
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        # 返回准备好的输入
        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    # 更新生成的输入
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        # 更新过去的键值对为模型输出的过去的键值对
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        # 更新位置ID为最后一个位置ID加1
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs
# 将样本文档字符串添加到模型的调用函数中
append_call_sample_docstring(
    # 模型类：FlaxBigBirdForCausalLM，用于生成文本的模型
    FlaxBigBirdForCausalLM,
    # 用于文档的检查点
    _CHECKPOINT_FOR_DOC,
    # 携带交叉注意力的条件语言建模输出
    FlaxCausalLMOutputWithCrossAttentions,
    # 用于文档的配置
    _CONFIG_FOR_DOC,
)
```