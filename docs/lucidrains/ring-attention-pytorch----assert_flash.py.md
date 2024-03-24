# `.\lucidrains\ring-attention-pytorch\assert_flash.py`

```
# 导入 torch 库
import torch

# 从 ring_attention_pytorch 模块中导入 default_attention 和 ring_flash_attn 函数
from ring_attention_pytorch import (
    default_attention,
    ring_flash_attn
)

# 定义变量

# 是否使用因果关系
causal = True
# 序列长度
seq_len = 62
# 桶大小
bucket_size = 4

# 基础的 qkv

# 随机生成 q 张量，形状为 (2, seq_len, 2, 16)
q = torch.randn(2, seq_len, 2, 16)
# 随机生成 k 张量，形状为 (2, seq_len, 2, 16)
k = torch.randn(2, seq_len, 2, 16)
# 随机生成 v 张量，形状为 (2, seq_len, 2, 16)
v = torch.randn(2, seq_len, 2, 16)

# flash 和 regular qkv

# 克隆 q 张量，并设置 requires_grad 为 True
fq = q.clone().requires_grad_()
# 克隆 k 张量，并设置 requires_grad 为 True
fk = k.clone().requires_grad_()
# 克隆 v 张量，并设置 requires_grad 为 True
fv = v.clone().requires_grad_()

# 克隆 q 张量，并设置 requires_grad 为 True
rq = q.clone().requires_grad_()
# 克隆 k 张量，并设置 requires_grad 为 True
rk = k.clone().requires_grad_()
# 克隆 v 张量，并设置 requires_grad 为 True
rv = v.clone().requires_grad_()

# 前向传播

# 使用 default_attention 函数计算输出 o
o = default_attention(rq, rk, rv, causal=causal)
# 使用 ring_flash_attn 函数计算输出 fo
fo = ring_flash_attn(fq, fk, fv, bucket_size=bucket_size, causal=causal)

# 断言 o 和 fo 的值在给定的容差范围内相等
assert torch.allclose(o, fo, atol=1e-6)

# 反向传播

# 对 o 求和并进行反向传播
o.sum().backward()
# 对 fo 求和并进行反向传播
fo.sum().backward()

# 断言 rq.grad 和 fq.grad 的值在给定的容差范围内相等
assert torch.allclose(rq.grad, fq.grad, atol=1e-6)
# 断言 rk.grad 和 fk.grad 的值在给定的容差范围内相等
assert torch.allclose(rk.grad, fk.grad, atol=1e-6)
# 断言 rv.grad 和 fv.grad 的值在给定的容差范围内相等
assert torch.allclose(rv.grad, fv.grad, atol=1e-6)
```