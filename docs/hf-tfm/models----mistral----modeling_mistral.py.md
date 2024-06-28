# `.\models\mistral\modeling_mistral.py`

```
# 设置文件编码为UTF-8
# 版权声明和许可信息，基于Mistral AI和HuggingFace Inc.团队的代码
# 本代码基于EleutherAI的GPT-NeoX库和GPT-NeoX和OPT实现进行了修改，以适应Meta AI团队训练模型时的架构差异
# 根据Apache许可证2.0版授权使用本文件，除非符合许可证要求，否则不得使用此文件

""" PyTorch Mistral model. """
# 导入Python标准库和第三方库
import inspect
import math
import warnings
from typing import List, Optional, Tuple, Union

# 导入PyTorch相关库
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入本地模块
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from .configuration_mistral import MistralConfig

# 检查是否支持Flash Attention 2
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

# 获取日志记录器
logger = logging.get_logger(__name__)

# 文档字符串中的配置信息
_CONFIG_FOR_DOC = "MistralConfig"

# 从transformers.models.llama.modeling_llama._get_unpad_data复制而来
# 根据注意力掩码获取未填充数据的函数
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

# 从transformers.models.llama.modeling_llama.LlamaRMSNorm中复制而来，将Llama->Mistral
# MistralRMSNorm类，继承自nn.Module，用于实现Mistral模型的RMS归一化
class MistralRMSNorm(nn.Module):
    # 初始化函数，定义了一个自定义的归一化层 MistralRMSNorm，功能类似于 T5 模型的 LayerNorm
    def __init__(self, hidden_size, eps=1e-6):
        """
        MistralRMSNorm is equivalent to T5LayerNorm
        """
        # 调用父类的初始化方法
        super().__init__()
        # 初始化权重参数，这些参数将被优化
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # 定义方差中添加的小常数值
        self.variance_epsilon = eps

    # 前向传播函数，计算归一化后的隐藏状态
    def forward(self, hidden_states):
        # 记录输入的数据类型
        input_dtype = hidden_states.dtype
        # 将输入的隐藏状态转换为 float32 类型
        hidden_states = hidden_states.to(torch.float32)
        # 计算隐藏状态的方差
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # 对隐藏状态进行归一化处理，通过除以标准差加上一个小常数来实现数值稳定性
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # 返回经过权重调节后的归一化隐藏状态
        return self.weight * hidden_states.to(input_dtype)
# 从transformers.models.llama.modeling_llama.LlamaRotaryEmbedding复制并修改为MistralRotaryEmbedding
# TODO @Arthur 在静态缓存后不再从LLama复制
class MistralRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # 计算频率倒数，用于正弦和余弦计算
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 为了使`torch.jit.trace`正常工作，在这里构建缓存
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        # 计算正弦和余弦缓存
        freqs = torch.outer(t, self.inv_freq)
        # 与论文中不同，但使用不同的排列顺序以获得相同的计算结果
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# 从transformers.models.llama.modeling_llama.rotate_half复制的函数
def rotate_half(x):
    """对输入的隐藏维度的一半进行旋转。"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# 从transformers.models.llama.modeling_llama.apply_rotary_pos_emb复制并修改
# TODO @Arthur 在静态缓存后不再从LLama复制
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """将Rotary位置嵌入应用到查询和键张量中。"""
    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # Unsqueezing cos and sin tensors along the specified dimension to match q and k tensor shapes
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    # Applying rotary position embedding to q and k tensors
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
class MistralMLP(nn.Module):
    # MistralMLP 类，用于定义一个 MLP 模型
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size  # 从配置中获取隐藏层大小
        self.intermediate_size = config.intermediate_size  # 从配置中获取中间层大小
        # 创建一个线性层，用于门控投影，输入大小为 hidden_size，输出大小为 intermediate_size，无偏置
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # 创建一个线性层，用于上投影，输入大小为 hidden_size，输出大小为 intermediate_size，无偏置
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # 创建一个线性层，用于下投影，输入大小为 intermediate_size，输出大小为 hidden_size，无偏置
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # 根据配置中的隐藏激活函数选择对应的激活函数
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # 前向传播函数，利用门控投影、激活函数、上投影计算最终输出
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    # 将 hidden_states 在维度 1 上重复 n_rep 次，实现扩展
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # 扩展 hidden_states 维度
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class MistralAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """
    # MistralAttention 类，实现多头注意力机制，基于 'Attention Is All You Need' 的方法，并支持滑动窗口注意力
    # 初始化函数，用于创建一个新的Mistral注意力层对象
    def __init__(self, config: MistralConfig, layer_idx: Optional[int] = None):
        # 调用父类的初始化方法
        super().__init__()
        # 将传入的配置对象保存到实例变量中
        self.config = config
        # 保存传入的层索引到实例变量中
        self.layer_idx = layer_idx
        # 如果未传入层索引，发出警告，并说明在使用缓存时可能会导致错误
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
    
        # 从配置中获取隐藏单元大小并保存到实例变量中
        self.hidden_size = config.hidden_size
        # 从配置中获取注意力头的数量并保存到实例变量中
        self.num_heads = config.num_attention_heads
        # 计算每个注意力头的维度并保存到实例变量中
        self.head_dim = self.hidden_size // self.num_heads
        # 从配置中获取键值头的数量并保存到实例变量中
        self.num_key_value_heads = config.num_key_value_heads
        # 计算每个键值头的组数并保存到实例变量中
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        # 从配置中获取最大位置嵌入数并保存到实例变量中
        self.max_position_embeddings = config.max_position_embeddings
        # 从配置中获取Rope Theta并保存到实例变量中
        self.rope_theta = config.rope_theta
        # 设置是否因果化为True，并保存到实例变量中
        self.is_causal = True
        # 从配置中获取注意力丢弃率并保存到实例变量中
        self.attention_dropout = config.attention_dropout
    
        # 检查隐藏单元大小是否能被注意力头的数量整除，否则抛出值错误
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        # 创建查询投影矩阵，将隐藏状态映射到注意力头维度的空间
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        # 创建键投影矩阵，将隐藏状态映射到键值头维度的空间
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # 创建值投影矩阵，将隐藏状态映射到键值头维度的空间
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # 创建输出投影矩阵，将注意力头的结果映射回隐藏状态的空间
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
    
        # 创建旋转嵌入对象，用于引入循环旋转机制以捕捉序列位置信息
        self.rotary_emb = MistralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
    
    # 定义形状函数，用于调整张量的形状以适应注意力计算的需要
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    
    # 前向传播函数，执行Mistral注意力层的计算过程
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
# 定义一个名为 MistralFlashAttention2 的类，继承自 MistralAttention 类。
# 这个类是 Mistral flash attention 模块，其权重继承自 MistralAttention，没有进行修改。
class MistralFlashAttention2(MistralAttention):
    """
    Mistral flash attention module. This module inherits from `MistralAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # 从 transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__ 中复制而来
    # 初始化函数，接受任意参数并传递给父类的初始化函数
    def __init__(self, *args, **kwargs):
        # 调用父类的初始化函数
        super().__init__(*args, **kwargs)

        # TODO: 在 Flash Attention for RoCm 更新到 2.1 后应移除这段代码。
        # flash_attn<2.1 生成左上角对齐的因果蒙版，而这里需要的是右下角对齐，默认情况下 flash_attn>=2.1 已经实现了这个变更。这个属性用于处理这个差异。
        # 参考链接：https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0
        # 需要注意的是，在 flash_attn<2.1 中，当 q_seqlen != k_seqlen（除了 q_seqlen == 1 的情况）时会生成错误的蒙版（左上角）。
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    # 正向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        # 该函数定义了模块的正向传播逻辑，接受多个参数，其中 hidden_states 是必传的 Tensor 类型参数。
        # attention_mask, position_ids, past_key_value, output_attentions, use_cache 等参数是可选的。
        # **kwargs 允许传递任意额外的关键字参数。

    # 私有方法 _flash_attention_forward 的定义
    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        use_sliding_windows=False,
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        # 如果键值序列长度与注意力掩码长度不一致，需要调整注意力掩码
        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len :]

        # 获取未填充数据的索引和相关的序列长度信息
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

        # 根据索引重新组织键和值的层，以便与查询层对齐
        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)

        if query_length == kv_seq_len:
            # 如果查询长度与键值序列长度相同，则直接使用相同的索引和序列长度信息
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            # 如果查询长度为1，特殊处理序列长度信息和查询层
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # 这里有一个内存复制操作，效率较低。
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # 对于其他情况，假设左填充，调整注意力掩码，然后调用unpad_input函数处理查询层
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        # 返回更新后的查询层、键层、值层，以及相关的索引和序列长度信息
        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
# 从 transformers.models.llama.modeling_llama.LlamaSdpaAttention 复制代码并将 LLama 改为 Mistral
# TODO @Arthur 在静态缓存后不再从 LLama 复制代码
class MistralSdpaAttention(MistralAttention):
    """
    Mistral 注意力模块使用 torch.nn.functional.scaled_dot_product_attention。该模块继承自
    `MistralAttention`，模块的权重保持不变。唯一的改动在于前向传播部分以适应 SDPA API。
    """

    # 改编自 MistralAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        """
        前向传播方法用于执行注意力计算。

        Args:
            hidden_states (torch.Tensor): 输入的隐藏状态张量。
            attention_mask (Optional[torch.Tensor], optional): 注意力掩码张量，默认为None。
            position_ids (Optional[torch.LongTensor], optional): 位置标识符张量，默认为None。
            past_key_value (Optional[Cache], optional): 过去的键值对缓存，默认为None。
            output_attentions (bool, optional): 是否输出注意力权重，默认为False。
            use_cache (bool, optional): 是否使用缓存，默认为False。

        Returns:
            根据模块的具体实现不同，返回不同的结果。
        """
        # 实现具体的注意力计算逻辑
        # (具体实现部分可能包括 scaled_dot_product_attention 的调用或其它实现方式)

MISTRAL_ATTENTION_CLASSES = {
    "eager": MistralAttention,
    "flash_attention_2": MistralFlashAttention2,
    "sdpa": MistralSdpaAttention,
}

class MistralDecoderLayer(nn.Module):
    def __init__(self, config: MistralConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # 初始化自注意力机制，根据配置选择不同的实现类
        self.self_attn = MISTRAL_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        # MLP 部分的初始化
        self.mlp = MistralMLP(config)

        # 输入层归一化，使用 MistralRMSNorm 类进行初始化
        self.input_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 注意力后归一化，同样使用 MistralRMSNorm 类进行初始化
        self.post_attention_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ):
        """
        Mistral 解码器层的前向传播方法。

        Args:
            hidden_states (torch.Tensor): 输入的隐藏状态张量。
            attention_mask (Optional[torch.Tensor], optional): 注意力掩码张量，默认为None。
            position_ids (Optional[torch.LongTensor], optional): 位置标识符张量，默认为None。
            past_key_value (Optional[Tuple[torch.Tensor]], optional): 过去的键值对缓存，默认为None。
            output_attentions (Optional[bool], optional): 是否输出注意力权重，默认为False。
            use_cache (Optional[bool], optional): 是否使用缓存，默认为False。
            **kwargs: 其他可选参数。

        Returns:
            根据模块的具体实现不同，返回不同的结果。
        """
        # 实现具体的前向传播逻辑
        # (具体实现部分包括自注意力、MLP处理和归一化处理等步骤)
        ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # 如果传入了 `padding_mask` 参数，发出警告，提示在 v4.37 版本中将移除，请使用 `attention_mask` 替代
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        
        """
        Args:
            hidden_states (`torch.FloatTensor`): 输入到层的张量，形状为 `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): 注意力掩码，形状为 `(batch, sequence_length)`，
                其中填充元素由0表示。
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力张量。查看返回张量中的 `attentions` 以获取更多详细信息。
            use_cache (`bool`, *optional*):
                如果设置为 `True`，将返回 `past_key_values` 键值状态，可用于加速解码（参见 `past_key_values`）。
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): 缓存的过去键和值投影状态
        """

        residual = hidden_states

        # 输入层归一化
        hidden_states = self.input_layernorm(hidden_states)

        # 自注意力
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        # 残差连接
        hidden_states = residual + hidden_states

        # 全连接层归一化
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # 残差连接
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
# 定义一个长文档字符串，描述了 MistralPreTrainedModel 类的继承关系和使用方法
MISTRAL_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MistralConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 为 MistralPreTrainedModel 类添加文档注释，指明它是一个输出原始隐藏状态的模型，没有特定的输出层
@add_start_docstrings(
    "The bare Mistral Model outputting raw hidden-states without any specific head on top.",
    MISTRAL_START_DOCSTRING,
)
class MistralPreTrainedModel(PreTrainedModel):
    # 指定 MistralConfig 作为配置类
    config_class = MistralConfig
    # 基础模型前缀名称为 "model"
    base_model_prefix = "model"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 不进行模块拆分的模块列表
    _no_split_modules = ["MistralDecoderLayer"]
    # 跳过设备放置的键名 "past_key_values"
    _skip_keys_device_placement = "past_key_values"
    # 支持 Flash Attention 2
    _supports_flash_attn_2 = True
    # 支持 SDPA
    _supports_sdpa = True
    # 支持缓存类
    _supports_cache_class = True

    # 初始化权重的方法，根据模块类型设置权重
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


MISTRAL_INPUTS_DOCSTRING = r"""
"""


# 为 MistralModel 类添加文档注释，描述它是一个 Transformer 解码器模型，由多个 MistralDecoderLayer 组成
@add_start_docstrings(
    "The bare Mistral Model outputting raw hidden-states without any specific head on top.",
    MISTRAL_START_DOCSTRING,
)
class MistralModel(MistralPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MistralDecoderLayer`]

    Args:
        config: MistralConfig
    """

    # 初始化方法，接受一个 MistralConfig 的配置对象
    def __init__(self, config: MistralConfig):
        super().__init__(config)
        # 设置填充索引为配置中的 pad_token_id
        self.padding_idx = config.pad_token_id
        # 设置词汇表大小为配置中的 vocab_size
        self.vocab_size = config.vocab_size

        # 初始化词嵌入层，指定词汇表大小、隐藏大小和填充索引
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # 初始化多个 MistralDecoderLayer 层，根据 num_hidden_layers 参数
        self.layers = nn.ModuleList(
            [MistralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # 设置注意力实现类型为配置中的 _attn_implementation
        self._attn_implementation = config._attn_implementation
        # 初始化 RMS 归一化层，指定隐藏大小和 RMS 归一化的 epsilon 值
        self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 默认关闭梯度检查点
        self.gradient_checkpointing = False
        # 初始化权重并进行最终处理
        self.post_init()
    # 返回当前模型的输入嵌入（embedding）
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置当前模型的输入嵌入（embedding）
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # 使用 MISTRAL_INPUTS_DOCSTRING 将文档字符串添加到模型前向传播方法上
    def forward(
        self,
        input_ids: torch.LongTensor = None,  # 输入的 token IDs，数据类型为 LongTensor
        attention_mask: Optional[torch.Tensor] = None,  # 注意力遮罩，可选的 Torch 张量
        position_ids: Optional[torch.LongTensor] = None,  # 位置 IDs，可选的 LongTensor
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 过去的键值对列表，可选的 FloatTensor 列表
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入张量，可选的 FloatTensor
        use_cache: Optional[bool] = None,  # 是否使用缓存，可选的布尔值
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选的布尔值
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选的布尔值
        return_dict: Optional[bool] = None,  # 是否以字典形式返回结果，可选的布尔值
class MistralForCausalLM(MistralPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        # 使用MistralModel构建模型
        self.model = MistralModel(config)
        # 设置词汇表大小
        self.vocab_size = config.vocab_size
        # 线性层，将隐藏状态映射到词汇表大小的空间，无偏置项
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回模型的输入嵌入层
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        # 设置模型的输入嵌入层
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        # 返回语言模型头部的输出嵌入层
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        # 设置语言模型头部的输出嵌入层
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        # 设置解码器
        self.model = decoder

    def get_decoder(self):
        # 获取解码器
        return self.model

    @add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 模型前向传播函数，详细说明见函数装饰器的注释
        pass

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # 为生成准备输入的函数，包括输入ID、过去键值、注意力掩码和输入嵌入
        pass
        # 检查是否提供了 past_key_values 参数，如果是则根据其内容进行处理
        if past_key_values is not None:
            # 如果 past_key_values 是 Cache 类型，则获取其相关属性
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()  # 获取缓存序列的长度
                past_length = past_key_values.seen_tokens  # 获取已处理的标记数
                max_cache_length = past_key_values.get_max_length()  # 获取最大缓存长度
            else:
                # 否则假设 past_key_values 是一个元组，获取其第一个元素的第三维长度作为 cache_length 和 past_length
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # 保留未处理的标记：
            # 1 - 如果 attention_mask 的长度超过 input_ids 的长度，则表明一些输入是作为缓存的一部分传递的
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - 如果 past_length 小于 input_ids 的长度，则 input_ids 包含所有输入标记。根据 past_length 截断 input_ids。
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - 否则（past_length >= input_ids.shape[1]），假设 input_ids 只包含未处理的标记。

            # 如果即将超过最大缓存长度，则需要裁剪输入的 attention_mask。
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        # 获取可选的 position_ids 参数，如果 attention_mask 存在且 position_ids 为 None，则动态生成 position_ids 以用于批次生成
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1  # 在 attention_mask 上累积和计算 position_ids
            position_ids.masked_fill_(attention_mask == 0, 1)  # 将 attention_mask 为 0 的位置填充为 1
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]  # 如果 past_key_values 存在，只保留与 input_ids 相关的部分

        # 如果传入了 inputs_embeds 参数，并且 past_key_values 为 None，则只在第一代中使用它们
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}  # 使用 inputs_embeds 作为模型输入
        else:
            model_inputs = {"input_ids": input_ids}  # 否则使用 input_ids 作为模型输入

        # 更新 model_inputs 字典，添加 position_ids、past_key_values、use_cache 和 attention_mask
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs  # 返回最终的模型输入字典

    @staticmethod
    # 定义一个函数 `_reorder_cache`，用于重新排序缓存 `past_key_values` 中的数据
    def _reorder_cache(past_key_values, beam_idx):
        # 初始化一个空元组，用于存储重新排序后的缓存数据
        reordered_past = ()
        # 遍历 past_key_values 中的每一层的缓存数据
        for layer_past in past_key_values:
            # 对每层的缓存数据进行重新排序，并将重新排序后的结果添加到 reordered_past 中
            reordered_past += (
                # 对每个 past_state 执行索引选择操作，使用 beam_idx 作为索引，转移到 past_state 的设备上
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新排序后的缓存数据 reordered_past
        return reordered_past
# 定义了一个用于序列分类的 Mistral 模型，其顶部有一个线性层用于分类。
# 该模型使用最后一个 token 进行分类，类似于其他因果模型（如 GPT-2）的做法。
# 如果配置中定义了 `pad_token_id`，则找到每行中不是填充 token 的最后一个 token 进行分类。
# 如果没有定义 `pad_token_id`，则直接取每个批次中每行的最后一个值作为分类的 token。
# 当传入 `inputs_embeds` 而不是 `input_ids` 时，由于无法猜测填充 token，也采用相同的策略（取每行的最后一个值）。
@add_start_docstrings(
    """
    The Mistral Model transformer with a sequence classification head on top (linear layer).

    [`MistralForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    MISTRAL_START_DOCSTRING,
)
# 从 transformers.models.llama.modeling_llama.LlamaForSequenceClassification 复制并修改为使用 Mistral 模型
class MistralForSequenceClassification(MistralPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = MistralModel(config)
        # 使用线性层进行分类，输出维度为类别数，没有偏置项
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # 初始化权重并进行最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 获取模型的输入嵌入层
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        # 设置模型的输入嵌入层
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
```