# `.\pytorch\benchmarks\gpt_fast\mixtral_moe_model.py`

```
# 导入必要的模块和库
from dataclasses import dataclass
from typing import Optional

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块
from torch import Tensor  # 导入张量类型
from torch.nn import functional as F  # 导入PyTorch中的函数模块


# 定义一个函数，用于找到大于等于n且能被k整除的最小整数
def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


# 数据类，用于存储模型的参数
@dataclass
class ModelArgs:
    block_size: int = 2048  # 块大小，默认为2048
    vocab_size: int = 32000  # 词汇表大小，默认为32000
    n_layer: int = 32  # 层数，默认为32
    n_head: int = 32  # 注意力头数，默认为32
    dim: int = 4096  # 维度，默认为4096
    intermediate_size: int = None  # 中间层大小，默认为None
    n_local_heads: int = -1  # 本地注意力头数，默认为-1
    head_dim: int = 64  # 注意力头的维度，默认为64
    rope_base: float = 10000  # 绳基数，默认为10000
    norm_eps: float = 1e-5  # 归一化的epsilon，默认为1e-5
    num_experts: int = 8  # 专家数，默认为8
    num_activated_experts: int = 2  # 激活的专家数，默认为2

    def __post_init__(self):
        # 如果本地注意力头数为-1，则设为注意力头数
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        # 如果中间层大小为None，则根据规则计算
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        # 计算每个注意力头的维度
        self.head_dim = self.dim // self.n_head

    @classmethod
    def from_name(cls, name: str):
        # 如果名称在transformer_configs中，则使用给定配置创建实例
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        # 否则进行模糊搜索
        config = [
            config
            for config in transformer_configs
            if config in str(name).upper() or config in str(name)
        ]
        # 确保找到的配置只有一个
        assert len(config) == 1, name
        # 使用找到的配置创建实例
        return cls(**transformer_configs[config[0]])


# 预定义的transformer配置字典
transformer_configs = {
    "Mixtral-8x7B-v0.1": dict(
        block_size=32768,
        n_layer=16,
        n_head=32,
        n_local_heads=8,
        dim=4096,
        intermediate_size=14336,
        rope_base=1000000.0,
        num_experts=8,
        num_activated_experts=2,
    ),
}


# KVCache类，用于缓存键值对
class KVCache(nn.Module):
    def __init__(
        self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16
    ):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        # 注册缓存的键和值，均初始化为0的张量
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # 更新缓存
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


# Transformer类，用于实现Transformer模型
class Transformer(nn.Module):
    # 初始化函数，用于设置模型参数和层次结构
    def __init__(self, config: ModelArgs) -> None:
        # 调用父类初始化函数
        super().__init__()
        # 将配置参数保存在实例中
        self.config = config

        # 创建词嵌入层，将词汇表大小映射到指定维度的向量空间
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        
        # 创建多个 TransformerBlock 组成的层列表
        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layer)
        )
        
        # 创建用于层归一化的 RMSNorm 层
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        
        # 创建线性层，用于生成最终的输出结果
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # 初始化缓存变量
        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    # 设置缓存函数，用于初始化或更新缓存
    def setup_caches(self, max_batch_size, max_seq_length):
        # 如果当前缓存已经满足要求，则无需更新
        if (
            self.max_seq_length >= max_seq_length
            and self.max_batch_size >= max_batch_size
        ):
            return
        
        # 计算每个头部的维度
        head_dim = self.config.dim // self.config.n_head
        
        # 将最大序列长度向上取整至8的倍数
        max_seq_length = find_multiple(max_seq_length, 8)
        
        # 更新最大序列长度和最大批处理大小
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        
        # 遍历每个 TransformerBlock 层，并初始化其注意力机制的缓存
        for b in self.layers:
            b.attention.kv_cache = KVCache(
                max_batch_size, max_seq_length, self.config.n_local_heads, head_dim
            )

        # 预计算频率偏置项
        self.freqs_cis = precompute_freqs_cis(
            self.config.block_size,
            self.config.dim // self.config.n_head,
            self.config.rope_base,
        )
        
        # 创建因果掩码，用于控制模型在预测时不能看到未来的信息
        self.causal_mask = torch.tril(
            torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)
        )

    # 前向传播函数，用于执行模型的前向计算
    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        # 确保缓存变量已经初始化
        assert self.freqs_cis is not None, "Caches must be initialized first"
        
        # 生成因果掩码，限制模型只能依赖当前或过去的信息
        mask = self.causal_mask[None, None, input_pos]
        
        # 获取当前位置的频率偏置项
        freqs_cis = self.freqs_cis[input_pos]
        
        # 将输入的索引映射到词嵌入空间
        x = self.tok_embeddings(idx)

        # 依次经过每个 TransformerBlock 层
        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, freqs_cis, mask)
        
        # 应用层归一化
        x = self.norm(x)
        
        # 通过线性层生成最终的预测结果
        logits = self.output(x)
        
        # 返回预测结果
        return logits

    # 类方法，通过模型名称创建一个模型实例
    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))
class TransformerBlock(nn.Module):
    # 定义一个 Transformer 模块的类，继承自 nn.Module
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        # 初始化函数，设置 TransformerBlock 的各个组件
        self.attention = Attention(config)
        # 初始化注意力机制模块
        self.block_sparse_moe = MOEFeedForward(config)
        # 初始化分块稀疏多路注意力前馈模块
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        # 初始化前馈神经网络层的归一化模块
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        # 初始化注意力机制的归一化模块

    def forward(
        self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor
    ) -> Tensor:
        # 定义 TransformerBlock 的前向传播函数
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        # 使用注意力机制对输入 x 进行处理，并与 x 相加得到 h
        out = h + self.block_sparse_moe(self.ffn_norm(h))
        # 使用分块稀疏多路注意力前馈模块对 h 进行处理，并与 h 相加得到 out
        return out


class Attention(nn.Module):
    # 定义一个注意力机制的类，继承自 nn.Module
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # 计算总的头维度

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        # 线性层，用于 key, query, value 的投影
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        # 线性层，用于输出的投影
        self.kv_cache = None
        # 初始化键值缓存为空

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        # 设置注意力机制的参数

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        # 加载模型时的钩子函数，用于处理权重的加载
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])
        # 如果存在特定的权重键，则将分开的权重拼接为一个 wqkv 权重

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
        # 定义注意力机制的前向传播函数
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        # 计算键值的大小
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)
        # 对输入 x 进行 key, query, value 的投影并分割得到 q, k, v

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        # 重塑 q, k, v 的形状

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)
        # 应用旋转嵌入到 q 和 k 上

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))
        # 转置 q, k, v 的维度

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)
        # 如果存在键值缓存，则更新 k, v

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        # 重复 k, v 以适应多头的形状

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        # 使用缩放点积注意力计算 y

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        # 转置 y 的维度并重新整形为输出形状

        y = self.wo(y)
        # 使用输出投影层处理 y
        return y
    # 初始化函数，用于创建一个新的对象实例
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建参数 w1，是一个神经网络参数张量，形状为 [num_experts, intermediate_size, dim]
        self.w1 = nn.Parameter(
            torch.empty(config.num_experts, config.intermediate_size, config.dim)
        )
        # 创建参数 w2，是一个神经网络参数张量，形状为 [num_experts, dim, intermediate_size]
        self.w2 = nn.Parameter(
            torch.empty(config.num_experts, config.dim, config.intermediate_size)
        )
        # 创建参数 w3，是一个神经网络参数张量，形状为 [num_experts, intermediate_size, dim]
        self.w3 = nn.Parameter(
            torch.empty(config.num_experts, config.intermediate_size, config.dim)
        )

    # 前向传播函数，用于计算神经网络的输出
    def forward(self, x: Tensor, expert_indices: Tensor) -> Tensor:
        # 获取当前专家索引对应的 w1 参数，形状为 [T, A, D, D]
        w1_weights = self.w1[expert_indices]
        # 获取当前专家索引对应的 w3 参数，形状为 [T, A, D, D]
        w3_weights = self.w3[expert_indices]
        # 获取当前专家索引对应的 w2 参数，形状为 [T, A, D, D]
        w2_weights = self.w2[expert_indices]
        # 计算第一部分输出 x1，通过 torch.einsum 对输入 x 和 w1_weights 进行张量乘法和求和，然后应用 silu 激活函数
        x1 = F.silu(torch.einsum("ti,taoi -> tao", x, w1_weights))
        # 计算第三部分输出 x3，通过 torch.einsum 对输入 x 和 w3_weights 进行张量乘法和求和
        x3 = torch.einsum("ti, taoi -> tao", x, w3_weights)
        # 计算最终的专家输出，通过 torch.einsum 对 x1 和 x3 进行逐元素乘法，然后与 w2_weights 进行张量乘法和求和
        expert_outs = torch.einsum("tao, taio -> tai", (x1 * x3), w2_weights)
        # 返回专家输出张量
        return expert_outs
# 定义一个基于多门控前馈网络的模块，用于MOE（多专家）模型
class MOEFeedForward(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        # 初始化一个线性层作为门控，输入维度为 config.dim，输出维度为 config.num_experts，无偏置
        self.gate = nn.Linear(config.dim, config.num_experts, bias=False)
        # 初始化一个条件前馈网络，根据 config 参数
        self.cond_ffn = ConditionalFeedForward(config)
        # 保存输入维度和激活的专家数到实例变量中
        self.dim = config.dim
        self.num_activated_experts = config.num_activated_experts

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(-1, self.dim)
        # x 的形状变为 [T, D]，其中 T 为序列长度，D 为隐藏层维度
        scores = self.gate(x)  # 计算每个输入向量的专家分数，形状为 [T, E]
        expert_weights = F.softmax(scores, dim=-1)
        # 对专家权重进行 softmax 归一化，dim=-1 表示在最后一个维度进行归一化，形状为 [T, E]
        expert_weights, expert_indices = torch.topk(
            expert_weights, self.num_activated_experts, dim=-1
        )
        # 取出每个输入向量的前 num_activated_experts 个最高权重的专家，形状分别为 [T, A] 和 [T, A]
        expert_weights /= expert_weights.sum(dim=-1, keepdim=True)
        # 对选择的专家权重进行归一化，使得每个输入向量的权重之和为 1，形状为 [T, A]
        expert_outs = self.cond_ffn(x, expert_indices)
        # 使用条件前馈网络处理输入向量和选定的专家索引，得到专家的输出，形状为 [T, D]
        return torch.einsum("tai,ta -> ti", expert_outs, expert_weights)
        # 使用 Einstein Summation Notation 计算加权和，形状为 [T, D]


# 定义一个带有 RMS 标准化的模块
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # 初始化标准化的权重参数，形状为 [dim]
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # 计算 RMS 标准化的函数
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        # 对输入进行 RMS 标准化处理，并保持与输入的数据类型一致
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
        # 返回标准化后的结果乘以权重参数


# 预计算序列长度和元素数的频率和余弦正弦值
def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000) -> Tensor:
    # 计算频率，使得 base 的递增幂次方与元素数对应
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    # 生成长度为 seq_len 的序列，形状为 [seq_len, n_elem // 2]，表示频率
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    # 计算频率的极坐标形式（复数），实部和虚部组成一个缓存张量，形状为 [seq_len, n_elem // 2, 2]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    # 返回频率的实部和虚部组成的缓存张量，数据类型为 torch.bfloat16


# 应用旋转嵌入到输入张量中
def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    # 将输入张量按照最后两个维度拆分成形状为 [..., 2] 的张量
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    # 将频率的复数形式频率张量变为与输入张量相同的形状，形状为 [1, T, 1, n_elem // 2, 2]
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    # 使用旋转嵌入公式，应用频率的正弦和余弦值到输入张量，形状为与 xshaped 相同
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )
    # 将处理后的张量展平，形状变为 [..., 2*n_elem // 2]
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)
    # 返回处理后的张量，数据类型与输入张量相同
```