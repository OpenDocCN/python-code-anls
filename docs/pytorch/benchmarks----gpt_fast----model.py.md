# `.\pytorch\benchmarks\gpt_fast\model.py`

```
# 引入必要的库和模块
from dataclasses import dataclass  # 导入用于定义数据类的装饰器
from typing import Optional  # 导入类型提示中的可选类型

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块
from torch import Tensor  # 导入PyTorch张量类型
from torch.nn import functional as F  # 导入PyTorch中的函数模块

# 定义一个函数，用于找到大于等于给定数字n的最小的k的倍数
def find_multiple(n: int, k: int) -> int:
    if n % k == 0:  # 如果n能被k整除
        return n  # 直接返回n
    return n + k - (n % k)  # 否则返回大于等于n的最小的k的倍数

# 定义一个数据类，用于存储模型参数
@dataclass
class ModelArgs:
    block_size: int = 2048  # 模型的块大小
    vocab_size: int = 32000  # 词汇表大小
    n_layer: int = 32  # 层数
    n_head: int = 32  # 头数
    dim: int = 4096  # 维度
    intermediate_size: int = None  # 中间层大小，默认为None
    n_local_heads: int = -1  # 本地头数，默认为-1
    head_dim: int = 64  # 头维度
    rope_base: float = 10000  # 绳子基数
    norm_eps: float = 1e-5  # 归一化的小量值

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head  # 如果本地头数未指定，则与总头数相同
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)  # 计算中间层大小
        self.head_dim = self.dim // self.n_head  # 计算头维度

    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])  # 根据名称从预定义的配置中初始化模型参数
        # 模糊搜索匹配名称
        config = [
            config
            for config in transformer_configs
            if config in str(name).upper() or config in str(name)
        ]

        # 可能会有两个或更多匹配的配置（例如，"7B" 和 "Mistral-7B"）。找到最佳匹配的配置，
        # 选择名称较长的（因为它匹配了更多的符号）
        if len(config) > 1:
            config.sort(key=len, reverse=True)
            assert len(config[0]) != len(
                config[1]
            ), name  # 确保只有一个最佳匹配

        return cls(**transformer_configs[config[0]])  # 返回最佳匹配的模型参数配置

# 预定义的Transformer模型配置
transformer_configs = {
    "CodeLlama-7b-Python-hf": dict(
        block_size=16384, vocab_size=32000, n_layer=32, dim=4096, rope_base=1000000
    ),
    "7B": dict(n_layer=32, n_head=32, dim=4096),
    "13B": dict(n_layer=40, n_head=40, dim=5120),
    "30B": dict(n_layer=60, n_head=52, dim=6656),
    "34B": dict(
        n_layer=48,
        n_head=64,
        dim=8192,
        vocab_size=32000,
        n_local_heads=8,
        intermediate_size=22016,
        rope_base=1000000,
    ),  # CodeLlama-34B-Python-hf
    "70B": dict(
        n_layer=80, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672
    ),
    "Mistral-7B": dict(
        n_layer=32,
        n_head=32,
        n_local_heads=8,
        dim=4096,
        intermediate_size=14336,
        vocab_size=32000,
    ),
}

# 定义一个缓存键值对的模块
class KVCache(nn.Module):
    def __init__(
        self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16
    ):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))
    # 定义一个方法 `update`，用于更新缓存中的键值对
    def update(self, input_pos, k_val, v_val):
        # 断言输入位置的维度是否与键值数组的最后一个维度大小相匹配
        assert input_pos.shape[0] == k_val.shape[2]

        # 将类成员变量 self.k_cache 赋值给局部变量 k_out
        k_out = self.k_cache
        # 将类成员变量 self.v_cache 赋值给局部变量 v_out
        v_out = self.v_cache

        # 在 k_out 中的指定位置 input_pos 处更新键值 k_val
        k_out[:, :, input_pos] = k_val
        # 在 v_out 中的指定位置 input_pos 处更新值 v_val
        v_out[:, :, input_pos] = v_val

        # 返回更新后的 k_out 和 v_out
        return k_out, v_out
class Transformer(nn.Module):
    # Transformer 模型的主要定义，继承自 nn.Module
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        # Token embeddings 层，使用 nn.Embedding 创建
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        
        # Transformer 的层列表，每层为 TransformerBlock 的实例化对象
        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layer)
        )
        
        # 层归一化模块，使用 RMSNorm，对应维度为 config.dim
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        
        # 输出层，使用 nn.Linear 进行线性变换，输出维度为 config.vocab_size，无偏置
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # 额外的属性初始化
        self.freqs_cis: Optional[Tensor] = None  # 频率缓存
        self.mask_cache: Optional[Tensor] = None  # 掩码缓存
        self.max_batch_size = -1  # 最大批次大小
        self.max_seq_length = -1  # 最大序列长度

    def setup_caches(self, max_batch_size, max_seq_length):
        # 设置缓存函数，用于初始化和更新缓存
        if (
            self.max_seq_length >= max_seq_length
            and self.max_batch_size >= max_batch_size
        ):
            return
        
        # 计算每个头部的维度
        head_dim = self.config.dim // self.config.n_head
        
        # 找到最接近的 8 的倍数的最大序列长度
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        
        # 初始化每个 TransformerBlock 的 kv_cache 属性
        for b in self.layers:
            b.attention.kv_cache = KVCache(
                max_batch_size, max_seq_length, self.config.n_local_heads, head_dim
            )

        # 预计算频率信息，用于注意力机制
        self.freqs_cis = precompute_freqs_cis(
            self.config.block_size,
            self.config.dim // self.config.n_head,
            self.config.rope_base,
        )
        
        # 创建因果掩码，对角线以下为 True，以上为 False
        self.causal_mask = torch.tril(
            torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)
        )

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        # 前向传播函数，接受 idx 和 input_pos 作为输入，返回 logits
        assert self.freqs_cis is not None, "Caches must be initialized first"
        
        # 生成因果掩码
        mask = self.causal_mask[None, None, input_pos]
        
        # 获取频率信息
        freqs_cis = self.freqs_cis[input_pos]
        
        # 获取 token embeddings
        x = self.tok_embeddings(idx)

        # 遍历每个 TransformerBlock 层，并执行前向传播
        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, freqs_cis, mask)
        
        # 执行层归一化
        x = self.norm(x)
        
        # 计算最终输出 logits
        logits = self.output(x)
        return logits

    @classmethod
    def from_name(cls, name: str):
        # 类方法，根据名称创建 Transformer 模型
        return cls(ModelArgs.from_name(name))


class TransformerBlock(nn.Module):
    # Transformer 模型中的每个块的定义，继承自 nn.Module
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        
        # 注意力机制层，使用 Attention 类
        self.attention = Attention(config)
        
        # 前馈网络层，使用 FeedForward 类
        self.feed_forward = FeedForward(config)
        
        # 注意力层的归一化模块
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        
        # 前馈网络层的归一化模块
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(
        self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor
    ) -> Tensor:
        # 块的前向传播函数，接受输入 x、位置信息 input_pos、频率信息 freqs_cis 和掩码 mask，返回输出
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Attention(nn.Module):
    # 注意力机制的定义，继承自 nn.Module
    # 初始化方法，接受一个配置参数对象 ModelArgs
    def __init__(self, config: ModelArgs):
        # 调用父类的初始化方法
        super().__init__()
        # 断言确保 dim 能被 n_head 整除
        assert config.dim % config.n_head == 0

        # 计算总的头部维度
        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        
        # 创建一个线性层，用于 key, query, value 投影到所有头部，但是在一个批次中
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        
        # 创建一个线性层，用于最后的输出映射
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        
        # 初始化缓存为 None
        self.kv_cache = None

        # 将配置参数中的各项赋值给对象的属性
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        
        # 注册一个加载状态字典前的钩子函数
        self._register_load_state_dict_pre_hook(self.load_hook)

    # 加载钩子函数，用于加载特定键的权重并合并
    def load_hook(self, state_dict, prefix, *args):
        # 如果特定键存在于状态字典中，则弹出相应的权重并合并
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    # 前向传播方法
    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
        # 获取输入张量 x 的形状信息
        bsz, seqlen, _ = x.shape

        # 计算局部头部的 key, query, value 大小
        kv_size = self.n_local_heads * self.head_dim
        
        # 对输入张量 x 进行 key, query, value 的投影
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        # 将每个头部的维度重新组织成四维张量
        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        # 对 query 和 key 应用旋转嵌入
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # 将 q, k, v 张量的头部维度和序列长度维度转置
        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        # 如果存在 kv_cache，则更新 k, v
        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        # 将 k, v 沿着头部维度重复 n_head // n_local_heads 次
        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        
        # 执行缩放点积注意力机制
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        # 将结果张量转置并重新组织形状
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        # 应用最后的输出映射层
        y = self.wo(y)
        
        # 返回输出张量 y
        return y
# 定义一个前馈神经网络模型类 FeedForward，继承自 nn.Module
class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        # 初始化输入到中间层的线性变换 w1，维度为 config.dim 到 config.intermediate_size，无偏置
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        # 初始化输入到中间层的线性变换 w3，维度同上，无偏置
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        # 初始化中间层到输出的线性变换 w2，维度为 config.intermediate_size 到 config.dim，无偏置
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    # 前向传播函数，接受输入张量 x，返回输出张量
    def forward(self, x: Tensor) -> Tensor:
        # 执行前向传播：先经过 w1 和 silu 激活函数，再乘以 w3，最后经过 w2 得到输出
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# 定义一个 RMS 归一化模型类 RMSNorm，继承自 nn.Module
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        # 设定归一化时的小数值常数 eps
        self.eps = eps
        # 初始化权重参数为全1的张量，维度为 dim
        self.weight = nn.Parameter(torch.ones(dim))

    # 私有方法 _norm，执行 RMS 归一化
    def _norm(self, x):
        # 计算 RMS 归一化结果
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    # 前向传播函数，接受输入张量 x，返回 RMS 归一化后的输出张量
    def forward(self, x: Tensor) -> Tensor:
        # 将输入张量 x 转换为 float 类型，并进行 RMS 归一化，最后乘以权重参数 self.weight
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# 定义一个预计算频率的余弦/正弦信息的函数 precompute_freqs_cis
def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000) -> Tensor:
    # 计算频率
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    # 创建时间步长张量 t
    t = torch.arange(seq_len, device=freqs.device)
    # 计算频率信息的外积
    freqs = torch.outer(t, freqs)
    # 使用极坐标形式创建频率信息的余弦/正弦张量
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    # 将实部和虚部堆叠成一个张量，返回结果作为缓存 cache
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=torch.bfloat16)


# 定义一个应用旋转嵌入的函数 apply_rotary_emb
def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    # 将输入张量 x 转换为 float，并重塑为形状为 (*x.shape[:-1], -1, 2) 的张量
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    # 将频率信息 freqs_cis 重塑为形状 (1, xshaped.size(1), 1, xshaped.size(3), 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    # 执行旋转嵌入操作，使用频率信息进行旋转变换
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )
    # 将结果展平至第3维度，返回处理后的张量，类型与输入张量 x 一致
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)
```