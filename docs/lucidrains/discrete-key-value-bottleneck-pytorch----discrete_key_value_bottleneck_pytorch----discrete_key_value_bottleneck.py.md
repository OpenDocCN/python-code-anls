# `.\lucidrains\discrete-key-value-bottleneck-pytorch\discrete_key_value_bottleneck_pytorch\discrete_key_value_bottleneck.py`

```py
# 导入 torch 库
import torch
# 从 torch 库中导入 nn, einsum 模块
from torch import nn, einsum
# 从 einops 库中导入 rearrange, repeat, reduce 函数
from einops import rearrange, repeat, reduce
# 从 vector_quantize_pytorch 模块中导入 VectorQuantize 类

# 辅助函数

# 判断变量是否存在的函数
def exists(val):
    return val is not None

# 返回默认值的函数
def default(val, d):
    return val if exists(val) else d

# 主类

class DiscreteKeyValueBottleneck(nn.Module):
    def __init__(
        self,
        dim,
        *,
        num_memories,
        dim_embed = None,
        num_memory_codebooks = 1,
        encoder = None,
        dim_memory = None,
        average_pool_memories = True,
        **kwargs
    ):
        super().__init__()
        self.encoder = encoder
        dim_embed = default(dim_embed, dim)
        self.dim_embed = dim_embed

        # 创建 VectorQuantize 对象
        self.vq = VectorQuantize(
            dim = dim * num_memory_codebooks,
            codebook_size = num_memories,
            heads = num_memory_codebooks,
            separate_codebook_per_head = True,
            **kwargs
        )

        dim_memory = default(dim_memory, dim)
        # 创建 nn.Parameter 对象
        self.values = nn.Parameter(torch.randn(num_memory_codebooks, num_memories, dim_memory))

        # 创建随机投影矩阵
        rand_proj = torch.empty(num_memory_codebooks, dim_embed, dim)
        nn.init.xavier_normal_(rand_proj)

        # 将随机投影矩阵注册为 buffer
        self.register_buffer('rand_proj', rand_proj)
        self.average_pool_memories = average_pool_memories

    def forward(
        self,
        x,
        return_intermediates = False,
        average_pool_memories = None,
        **kwargs
    ):
        average_pool_memories = default(average_pool_memories, self.average_pool_memories)

        if exists(self.encoder):
            self.encoder.eval()
            with torch.no_grad():
                x = self.encoder(x, **kwargs)
                x.detach_()

        # 检查输入张量的最后一个维度是否与 dim_embed 相同
        assert x.shape[-1] == self.dim_embed, f'encoding has a dimension of {x.shape[-1]} but dim_embed (defaults to dim) is set to {self.dim_embed} on init'

        # 线性变换
        x = einsum('b n d, c d e -> b n c e', x, self.rand_proj)
        # 重排张量维度
        x = rearrange(x, 'b n c e -> b n (c e)')

        # 对 x 进行向量量化
        vq_out = self.vq(x)

        quantized, memory_indices, commit_loss = vq_out

        if memory_indices.ndim == 2:
            memory_indices = rearrange(memory_indices, '... -> ... 1')

        memory_indices = rearrange(memory_indices, 'b n h -> b h n')

        values = repeat(self.values, 'h n d -> b h n d', b = memory_indices.shape[0])
        memory_indices = repeat(memory_indices, 'b h n -> b h n d', d = values.shape[-1])

        memories = values.gather(2, memory_indices)

        if average_pool_memories:
            memories = reduce(memories, 'b h n d -> b n d', 'mean')

        if return_intermediates:
            return memories, vq_out

        return memories
```