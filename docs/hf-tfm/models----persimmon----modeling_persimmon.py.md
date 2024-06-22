# `.\transformers\models\persimmon\modeling_persimmon.py`

```py
# 指定文件编码为utf-8

# 版权声明，说明代码的版权信息
# 版权归EleutherAI和HuggingFace Inc.团队所有
#
# 本代码基于EleutherAI的GPT-NeoX库以及该库中的GPT-NeoX和OPT实现。已对其进行修改，
# 以适应与训练模型的Meta AI团队使用的GPT-NeoX和OPT之间的轻微架构差异
# 
# 根据Apache许可证2.0版权许可
# 除非符合许可证，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，
# 没有任何明示或暗示的担保或条件
# 请查看许可证以获得更多有关权限和限制的详细信息
""" PyTorch Persimmon model. """

# 导入所需的库
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入自定义的激活函数映射表和缓存工具
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
# 导入Persimmon模型的配置类
from .configuration_persimmon import PersimmonConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 为文档添加配置说明
_CONFIG_FOR_DOC = "PersimmonConfig"

# 从transformers.models.llama.modeling_llama.LlamaRotaryEmbedding复制到PersimmonRotaryEmbedding
class PersimmonRotaryEmbedding(nn.Module):
    # 初始化函数
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        # 初始化维度、最大位置嵌入、基础值等参数
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # 计算频率倒数，根据维度和基础值计算
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        # 将频率倒数注册为缓冲区
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 构建cos和sin的缓存，以使torch.jit.trace正常工作
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )
    # 设置余弦和正弦的缓存，用于后续计算
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # 设置当前缓存的最大序列长度
        self.max_seq_len_cached = seq_len
        # 创建一个序列从 0 到最大序列长度的张量 t，并移动到指定设备和指定数据类型
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
    
        # 计算频率乘以序列 t，构成一个矩阵
        freqs = torch.outer(t, self.inv_freq)
        # 连接频率矩阵两个副本，用于后续的正弦和余弦计算
        emb = torch.cat((freqs, freqs), dim=-1)
        # 将计算出的余弦值注册为缓存数据，数据类型转换为指定类型，并且不持久化到模型状态字典中
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        # 将计算出的正弦值注册为缓存数据，数据类型转换为指定类型，并且不持久化到模型状态字典中
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
    
    # 前向传播函数
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # 如果输入序列长度超过当前缓存的最大序列长度，重新计算余弦和正弦缓存
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
    
        # 返回存储的余弦和正弦缓存，取其中部分数据，并且转换为输入张量的数据类型
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )
# 定义 PersimmonLinearScalingRotaryEmbedding 类继承自 PersimmonRotaryEmbedding 类，添加了线性缩放功能
class PersimmonLinearScalingRotaryEmbedding(PersimmonRotaryEmbedding):
    # 类描述，提到了来源和功劳归于 Reddit 用户 /u/kaiokendev
    """PersimmonRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    # 初始化函数，设置参数及继承父类初始化
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        # 存储缩放因子
        self.scaling_factor = scaling_factor
        # 调用父类构造函数完成初始化
        super().__init__(dim, max_position_embeddings, base, device)

    # 定义一个设置余弦和正弦缓存的私有方法
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # 缓存序列长度
        self.max_seq_len_cached = seq_len
        # 生成一个序列，应用设备和数据类型设置
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        # 将序列除以缩放因子进行线性缩放
        t = t / self.scaling_factor

        # 计算频率作为 t 和逆频率的外积
        freqs = torch.outer(t, self.inv_freq)
        # 与论文不同的处理方式，但实现了相同的计算，合并频率以创建嵌入
        emb = torch.cat((freqs, freqs), dim=-1)
        # 创建并注册余弦缓存
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        # 创建并注册正弦缓存
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

# 定义 PersimmonDynamicNTKScalingRotaryEmbedding 类继承自 PersimmonRotaryEmbedding 类，添加了动态 NTK 缩放功能
class PersimmonDynamicNTKScalingRotaryEmbedding(PersimmonRotaryEmbedding):
    # 类描述，提到了来源和功劳归于 Reddit 用户 /u/bloc97 和 /u/emozilla
    """PersimmonRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    # 初始化函数，设置参数及继承父类初始化
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        # 存储缩放因子
        self.scaling_factor = scaling_factor
        # 调用父类构造函数完成初始化
        super().__init__(dim, max_position_embeddings, base, device)

    # 定义一个设置余弦和正弦缓存的私有方法
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # 缓存序列长度
        self.max_seq_len_cached = seq_len

        # 如果序列长度超过最大位置嵌入，则调整基数
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            # 计算逆频率
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            # 注册逆频率缓存
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 生成一个序列，应用设备和数据类型设置
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        # 计算频率作为 t 和逆频率的外积
        freqs = torch.outer(t, self.inv_freq)
        # 与论文不同的处理方式，但实现了相同的计算，合并频率以创建嵌入
        emb = torch.cat((freqs, freqs), dim=-1)
        # 创建并注册余弦缓存
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        # 创建并注册正弦缓存
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

# 定义 rotate_half 函数用于处理输入向量的一半维度
def rotate_half(x):
    # 提取输入向量的前半部分
    x1 = x[..., : x.shape[-1] // 2]
    # 提取输入向量的后半部分
    x2 = x[..., x.shape[-1] // 2 :]
    # 将后半部分取负并与前半部分进行拼接
    return torch.cat((-x2, x1), dim=-1)

# 被复制的标记，但没有给出函数的定义，可能在代码的其他部分
# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
# 将旋转位置嵌入应用于查询和键张量
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor. 查询张量
        k (`torch.Tensor`): The key tensor. 键张量
        cos (`torch.Tensor`): The cosine part of the rotary embedding. 旋转嵌入的余弦部分
        sin (`torch.Tensor`): The sine part of the rotary embedding. 旋转嵌入的正弦部分
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. 与查询和键张量对应的令牌的位置索引
            For example, this can be used to pass offsetted position ids when working with a KV-cache. 例如，在使用 KV-cache 时，可以传递偏移的位置 id。
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. 'unsqueeze_dim' 参数指定了沿着哪个维度对 cos[position_ids] 和 sin[position_ids] 进行展开，以便它们可以正确地广播到 q 和 k 的维度。
            For example, note that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. 例如，注意 cos[position_ids] 和 sin[position_ids] 的形状为 [batch_size, seq_len, head_dim]。
            Then, if q and k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. 如果 q 和 k 的形状为 [batch_size, heads, seq_len, head_dim]，则设置 unsqueeze_dim=1 使 cos[position_ids] 和 sin[position_ids] 可以广播到 q 和 k 的形状。
            Similarly, if q and k have the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2. 类似地，如果 q 和 k 的形状为 [batch_size, seq_len, heads, head_dim]，则设置 unsqueeze_dim=2。
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding. 返回旋转使用旋转位置嵌入的查询和键张量的元组。
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)  # 沿指定维度展开 cos[position_ids]
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)  # 沿指定维度展开 sin[position_ids]
    q_embed = (q * cos) + (rotate_half(q) * sin)  # 计算旋转后的查询张量
    k_embed = (k * cos) + (rotate_half(k) * sin)  # 计算旋转后的键张量
    return q_embed, k_embed  # 返回旋转后的查询���键张量


# 从 transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXMLP 复制并修改为 Persimmon
class PersimmonMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.intermediate_size)  # 线性变换层，从隐藏大小到中间大小
        self.dense_4h_to_h = nn.Linear(config.intermediate_size, config.hidden_size)  # 线性变换层，从中间大小到隐藏大小
        self.act = ACT2FN[config.hidden_act]  # 激活函数

    def forward(self, hidden_states):
        hidden_states = self.dense_h_to_4h(hidden_states)  # 隐藏状态经过第一个线性变换层
        hidden_states = self.act(hidden_states)  # 经过激活函数
        hidden_states = self.dense_4h_to_h(hidden_states)  # 经过第二个线性变换层
        return hidden_states  # 返回处理后的隐藏状态


class PersimmonAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    # 初始化函数，接受配置和层索引作为参数
    def __init__(self, config: PersimmonConfig, layer_idx: Optional[int] = None):
        # 调用父类的初始化函数
        super().__init__()
        # 保存配置和层索引到对象属性中
        self.config = config
        self.layer_idx = layer_idx
        # 如果未传入层索引，则发出警告
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        # 初始化隐藏层大小、注意力头数、头维度、最大位置嵌入、绳子角度和部分旋转因子
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.partial_rotary_factor = config.partial_rotary_factor
        self.is_causal = True

        # 检查隐藏层大小是否能被注意力头数整除
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        # 初始化查询、键、值的线性层和全连接层
        self.query_key_value = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=True)
        self.dense = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)
        self.qk_layernorm = config.qk_layernorm

        # 如果需要对查询和键进行 LayerNorm
        if self.qk_layernorm:
            self.q_layernorm = nn.LayerNorm(
                config.hidden_size // self.num_heads, eps=config.layer_norm_eps, elementwise_affine=True
            )
            self.k_layernorm = nn.LayerNorm(
                config.hidden_size // self.num_heads, eps=config.layer_norm_eps, elementwise_affine=True
            )
        
        # 初始化注意力丢弃层和绳子初始化函数
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self._init_rope()
    # 初始化 RoPE（Rotary Positional Embedding）模块
    def _init_rope(self):
        # 如果配置中未指定 RoPE 缩放参数，则使用 PersimmonRotaryEmbedding 初始化
        if self.config.rope_scaling is None:
            self.rotary_emb = PersimmonRotaryEmbedding(
                int(self.partial_rotary_factor * self.head_dim),
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            # 获取 RoPE 缩放类型和缩放因子
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            # 根据不同的缩放类型选择不同的 RoPE 初始化方式
            if scaling_type == "linear":
                self.rotary_emb = PersimmonLinearScalingRotaryEmbedding(
                    int(self.partial_rotary_factor * self.head_dim),
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = PersimmonDynamicNTKScalingRotaryEmbedding(
                    int(self.partial_rotary_factor * self.head_dim),
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                # 抛出异常，未知的 RoPE 缩放类型
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    # 从融合的 QKV 张量中拆分出 query、key、value
    # 没有复制任何数据，结果共享与 fused_qkv 相同的内存存储
    def _split_heads(self, fused_qkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split the last dimension into (num_heads, head_dim) without making any copies, results share same memory
        storage as `fused_qkv`

        Args:
            fused_qkv (`torch.tensor`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]

        Returns:
            query: [batch_size, seq_length, num_heads, head_dim] key: [batch_size, seq_length, num_heads, head_dim]
            value: [batch_size, seq_length, num_heads, head_dim]
        """
        batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
        # 重塑张量形状，拆分出 query、key、value
        fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads, 3, self.head_dim)
        return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
# 定义 PersimmonDecoderLayer 类，继承自 nn.Module
class PersimmonDecoderLayer(nn.Module):
    # 初始化方法，接受 PersimmonConfig 和 layer_idx 作为参数
    def __init__(self, config: PersimmonConfig, layer_idx: int):
        # 调用父类的初始化方法
        super().__init__()
        # 从配置中获取隐藏层大小
        self.hidden_size = config.hidden_size
        # 创建自注意力层对象
        self.self_attn = PersimmonAttention(config=config, layer_idx=layer_idx)
        # 创建多层感知机对象
        self.mlp = PersimmonMLP(config)
        # 创建输入层归一化对象
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建注意力后归一化对象
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建 dropout 层对象
        self.dropout = nn.Dropout(config.hidden_dropout)

    # 前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range
                `[0, config.n_positions - 1]`.

                [What are position IDs?](../glossary#position-ids)
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*):
                cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """

        residual = hidden_states

        # 对输入进行 Layer Normalization
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        # 进行自注意力计算
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        # 将自注意力计算结果与残差相加
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        # 对自注意力计算结果进行 Layer Normalization
        hidden_states = self.post_attention_layernorm(hidden_states)
        # 进行全连接层计算
        hidden_states = self.mlp(hidden_states)

        # Dropout
        hidden_states = self.dropout(hidden_states)
        # 将全连接层计算结果与残差相加
        hidden_states = hidden_states + residual

        outputs = (hidden_states,)

        # 如果需要输出注意力权重
        if output_attentions:
            outputs += (self_attn_weights,)

        # 如果需要使用缓存
        if use_cache:
            outputs += (present_key_value,)

        return outputs
# 定义 Persimmon 模型的文档字符串，包含了模型的继承关系、参数说明等信息
PERSIMMON_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`PersimmonConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 添加文档字符串到 PersimmonModel 类
@add_start_docstrings(
    "The bare Persimmon Model outputting raw hidden-states without any specific head on top.",
    PERSIMMON_START_DOCSTRING,
)
class PersimmonPreTrainedModel(PreTrainedModel):
    config_class = PersimmonConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PersimmonDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True

    # 初始化权重
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

# 定义 Persimmon 模型的输入文档字符串
PERSIMMON_INPUTS_DOCSTRING = r"""
"""

# 添加文档字符串到 PersimmonModel 类
@add_start_docstrings(
    "The bare Persimmon Model outputting raw hidden-states without any specific head on top.",
    PERSIMMON_START_DOCSTRING,
)
class PersimmonModel(PersimmonPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`PersimmonDecoderLayer`]

    Args:
        config: PersimmonConfig
    """

    # 初始化 PersimmonModel 类
    def __init__(self, config: PersimmonConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # 初始化嵌入层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # 初始化层列表
        self.layers = nn.ModuleList(
            [PersimmonDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # 初始化最终的 LayerNorm
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.embed_tokens
    # 设置输入嵌入层的数值
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # 在模型前向传播函数中添加文档字符串
    @add_start_docstrings_to_model_forward(PERSIMMON_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,  # 输入的 token ID
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，指示哪些 token 需要被注意
        position_ids: Optional[torch.LongTensor] = None,  # 位置 ID，指示每个 token 的位置
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 用于存储过去的键值对
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入向量
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典形式的输出
class PersimmonForCausalLM(PersimmonPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    # 从transformers.models.llama.modeling_llama.LlamaForCausalLM.__init__中复制而来，初始化PersimmonForCausalLM类
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建PersimmonModel对象
        self.model = PersimmonModel(config)
        # 获取词汇表大小
        self.vocab_size = config.vocab_size
        # 创建线性层，用于LM头部
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 从transformers.models.llama.modeling_llama.LlamaForCausalLM.get_input_embeddings中复制而来，获取输入嵌入
    def get_input_embeddings(self):
        return self.model.embed_tokens

    # 从transformers.models.llama.modeling_llama.LlamaForCausalLM.set_input_embeddings中复制而来，设置输入嵌入
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # 从transformers.models.llama.modeling_llama.LlamaForCausalLM.get_output_embeddings中复制而来，获取输出嵌入
    def get_output_embeddings(self):
        return self.lm_head

    # 从transformers.models.llama.modeling_llama.LlamaForCausalLM.set_output_embeddings中复制而来，设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 从transformers.models.llama.modeling_llama.LlamaForCausalLM.set_decoder中复制而来，设置解码器
    def set_decoder(self, decoder):
        self.model = decoder

    # 从transformers.models.llama.modeling_llama.LlamaForCausalLM.get_decoder中复制而来，获取解码器
    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(PERSIMMON_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    # 从transformers.models.llama.modeling_llama.LlamaForCausalLM.forward中复制而来，前向传播方法
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
    # 从transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation中复制而来，为生成准备输入
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
        ):
        # 如果过去的键值不为空
        if past_key_values is not None:
            # 如果过去的键值是缓存对象
            if isinstance(past_key_values, Cache):
                # 获取缓存对象的序列长度
                cache_length = past_key_values.get_seq_length()
                # 获取缓存对象已见标记的数量
                past_length = past_key_values.seen_tokens
                # 获取缓存对象的最大长度
                max_cache_length = past_key_values.get_max_length()
            else:
                # 如果过去的键值不是缓存对象，则获取其维度信息
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # 保留未处理的标记：
            # 1 - 如果注意力掩码的长度超过了输入标记的长度，则说明一些输入是作为缓存的一部分传递的
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - 如果过去的长度小于输入标记的长度，则输入标记包含所有输入标记。我们可以根据过去的长度丢弃输入标记。
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - 否则（past_length >= input_ids.shape[1]），假设输入标记只包含未处理的标记。

            # 如果我们即将超过最大缓存长度，则需要裁剪输入注意力掩码。
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # 为批量生成动态创建位置标识
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # 如果传递了`inputs_embeds`，我们只想在第1代步骤中使用它们
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    # 重新排序缓存中的过去键值对
    def _reorder_cache(past_key_values, beam_idx):
        # 初始化重新排序后的过去键值对元组
        reordered_past = ()
        # 遍历每一层的过去键值对
        for layer_past in past_key_values:
            # 将每一层的过去状态按照beam_idx重新排序，并转移到相同设备上
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新排序后的过去键值对
        return reordered_past
# 导入函数 add_start_docstrings
# 定义了一个带有线性层的 Persimmon 转换器，用于序列分类
# PersimmonForSequenceClassification 使用最后一个标记进行分类，类似于其他因果模型（例如 GPT-2）
# 如果配置中定义了 pad_token_id，则找到每行中不是填充标记的最后一个标记；如果未定义 pad_token_id，则取批次中每行的最后一个值
# 当传递 inputs_embeds 而不是 input_ids 时，无法猜测填充标记，因此也取批次中每行的最后一个值
class PersimmonForSequenceClassification(PersimmonPreTrainedModel):
    # 初始化函数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)
        # 获取标签数量
        self.num_labels = config.num_labels
        # 创建 PersimmonModel 模型
        self.model = PersimmonModel(config)
        # 创建线性层，用于分类
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.model.embed_tokens

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # 前向传播函数
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