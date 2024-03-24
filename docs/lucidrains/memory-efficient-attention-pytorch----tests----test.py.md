# `.\lucidrains\memory-efficient-attention-pytorch\tests\test.py`

```
# 导入 torch 库
import torch
# 从 memory_efficient_attention_pytorch 中导入 Attention 类
from memory_efficient_attention_pytorch import Attention

# 从 memory_efficient_attention_pytorch.memory_efficient_attention 中导入 attention 函数
from memory_efficient_attention_pytorch.memory_efficient_attention import attention
# 从 memory_efficient_attention_pytorch.flash_attention 中导入 FlashAttention 和 FlashAttentionFunction 类

# 定义常量

# 判断两个张量是否接近
def isclose(a, b, atol = 1e-6):
    # 计算两个张量的最大差值
    diff = (a - b).abs().amax()
    # 返回是否小于给定的阈值
    return diff < atol

# 测试输出是否相等

def test_output_equal():
    # 创建 Attention 对象
    attn = Attention(
        dim = 512,
        dim_head = 64,
        heads = 8,
        q_bucket_size = 64,
        k_bucket_size = 64,
        causal = True
    )

    # 创建随机张量 x 和掩码 mask
    x = torch.randn(2, 2048, 512)
    mask = torch.ones(2, 2048).bool()

    # 使用 Attention 对象计算输出
    out = attn(x, mask = mask)
    # 使用 Attention 对象计算输出（启用内存效率模式）
    mem_efficient_out = attn(x, mask = mask, memory_efficient = True)

    # 断言内存效率输出与普通输出是否接近
    assert isclose(mem_efficient_out, out, atol = 1e-6)

# 测试梯度是否相等

def test_gradients_equal():
    # 创建 Attention 对象
    attn = Attention(
        dim = 512,
        dim_head = 64,
        heads = 8,
        q_bucket_size = 64,
        k_bucket_size = 64,
        causal = True
    )

    # 定义损失函数
    def loss_fn(inp, **kwargs):
        return attn(inp, **kwargs).sum()

    # 创建随机张量 x 和掩码 mask
    x = torch.randn(2, 2048, 512).requires_grad_()
    mask = torch.ones(2, 2048).bool()

    # 计算损失并反向传播
    loss_fn(x, mask = mask).backward()
    out_grad = x.grad.clone()

    x.grad.zero_()
    # 计算损失并反向传播（启用内存效率模式）
    loss_fn(x, mask = mask, memory_efficient = True).backward()
    mem_efficient_out_grad = x.grad.clone()

    # 断言内存效率梯度与普通梯度是否接近
    assert isclose(out_grad, mem_efficient_out_grad, atol = 1e-5)

# 测试 Flash Attention

def test_flash_attn_output_equal():
    attn_kwargs = dict(
        dim = 512,
        dim_head = 64,
        heads = 8,
        q_bucket_size = 64,
        k_bucket_size = 64,
        causal = True
    )

    # 创建 Attention 和 FlashAttention 对象
    attn = Attention(**attn_kwargs)
    flash_attn = FlashAttention(**attn_kwargs)

    # 将 Attention 对象的权重赋值给 FlashAttention 对象
    flash_attn.to_q = attn.to_q
    flash_attn.to_kv = attn.to_kv
    flash_attn.to_out = attn.to_out

    # 创建随机张量 x 和掩码 mask
    x = torch.randn(2, 2048, 512)
    mask = torch.ones(2, 2048).bool()

    # 使用 Attention 和 FlashAttention 对象计算输出
    out = attn(x, mask = mask)
    mem_efficient_out = flash_attn(x, mask = mask)

    # 断言内存效率输出与普通输出是否接近
    assert isclose(mem_efficient_out, out, atol = 1e-6)

# 测试 Flash Attention 梯度是否相等

def test_flash_attn_gradients_equal():
    q = torch.randn(1, 8, 1024, 512).requires_grad_()
    k = torch.randn(1, 8, 1024, 512).requires_grad_()
    v = torch.randn(1, 8, 1024, 512).requires_grad_()

    mask = torch.ones(1, 1024).bool()

    # 使用 attention 函数计算输出并反向传播
    o = attention(q, k, v, mask = mask, causal = True)
    o.sum().backward()

    dq_grad = q.grad.clone()
    dk_grad = k.grad.clone()
    dv_grad = v.grad.clone()

    q.grad.zero_()
    k.grad.zero_()
    v.grad.zero_()

    # 使用 FlashAttentionFunction 计算输出并反向传播
    flash_o = FlashAttentionFunction.apply(q, k, v, mask, True, 64, 64)
    flash_o.sum().backward()

    flash_dq_grad = q.grad.clone()
    flash_dk_grad = k.grad.clone()
    flash_dv_grad = v.grad.clone()

    # 断言 FlashAttention 梯度与 attention 函数梯度是否接近
    assert isclose(flash_dq_grad, dq_grad, atol = 1e-5)
    assert isclose(flash_dk_grad, dk_grad, atol = 1e-5)
    assert isclose(flash_dv_grad, dv_grad, atol = 1e-5)

# 测试 Flash Attention - 完全注意力掩码

def test_flash_attn_full_attn_mask_output_equal():
    attn_kwargs = dict(
        dim = 512,
        dim_head = 64,
        heads = 8,
        q_bucket_size = 64,
        k_bucket_size = 64,
        causal = True
    )

    # 创建 Attention 和 FlashAttention 对象
    attn = Attention(**attn_kwargs)
    flash_attn = FlashAttention(**attn_kwargs)

    # 将 Attention 对象的权重赋值给 FlashAttention 对象
    flash_attn.to_q = attn.to_q
    flash_attn.to_kv = attn.to_kv
    flash_attn.to_out = attn.to_out

    # 创建随机张量 x 和完全注意力掩码 mask
    x = torch.randn(2, 2048, 512)
    mask = torch.ones(2, 1, 2048, 2048).bool()

    # 使用 Attention 和 FlashAttention 对象计算输出
    out = attn(x, mask = mask)
    mem_efficient_out = flash_attn(x, mask = mask)

    # 断言内存效率输出与普通输出是否接近
    assert isclose(mem_efficient_out, out, atol = 1e-6)

# 测试梯度是否相等 - 完全注意力掩码

def test_flash_attn_full_attn_mask_gradients_equal():
    q = torch.randn(1, 8, 1024, 512).requires_grad_()
    k = torch.randn(1, 8, 1024, 512).requires_grad_()
    v = torch.randn(1, 8, 1024, 512).requires_grad_()

    mask = torch.ones(1, 1, 1024, 1024).bool()

    # 使用 attention 函数计算输出
    o = attention(q, k, v, mask = mask, causal = True)
    # 对输出进行求和并计算反向传播
    o.sum().backward()

    # 克隆梯度信息
    dq_grad = q.grad.clone()
    dk_grad = k.grad.clone()
    dv_grad = v.grad.clone()

    # 将梯度信息清零
    q.grad.zero_()
    k.grad.zero_()
    v.grad.zero_()

    # 使用自定义的FlashAttentionFunction进行注意力计算，并进行反向传播
    flash_o = FlashAttentionFunction.apply(q, k, v, mask, True, 64, 64)
    flash_o.sum().backward()

    # 克隆FlashAttentionFunction计算后的梯度信息
    flash_dq_grad = q.grad.clone()
    flash_dk_grad = k.grad.clone()
    flash_dv_grad = v.grad.clone()

    # 断言FlashAttentionFunction计算后的梯度信息与原始梯度信息在一定误差范围内相等
    assert isclose(flash_dq_grad, dq_grad, atol = 1e-5)
    assert isclose(flash_dk_grad, dk_grad, atol = 1e-5)
    assert isclose(flash_dv_grad, dv_grad, atol = 1e-5)
```