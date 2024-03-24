# `.\lucidrains\flash-cosine-sim-attention\tests\test.py`

```py
import torch
import pytest
from flash_cosine_sim_attention import plain_cosine_sim_attention, flash_cosine_sim_attention

# 检查是否CUDA可用
assert torch.cuda.is_available(), 'cuda must be available'

# 辅助函数

# 检查张量中是否存在NaN或无穷大值
def not_nan_or_infs(t):
    return not (torch.any(torch.isnan(t)) or torch.any(torch.isinf(t)))

# 检查两个张量是否在指定的绝对误差范围内相等
def allclose(a, b, atol = 1e-4):
    diff = (a - b).abs().amax()

    if torch.any(diff > atol):
        print(f'diff: {diff}')

    return diff <= atol

# 检查张量是否存在
def exists(t):
    return t is not None

# 如果张量存在，则将其移动到CPU上
def maybe_cpu(t):
    if not exists(t):
        return None

    return t.cpu()

# 测试

# 参数化测试用例
@pytest.mark.parametrize('causal,mask', [(True, False), (False, True), (False, False)])
@pytest.mark.parametrize('attn_bias', [True, False])
@pytest.mark.parametrize('seq_len', [63, 127])
@pytest.mark.parametrize('dim_head', [32, 64, 96, 128])
@pytest.mark.parametrize('float16', [False, True])
@pytest.mark.parametrize('attn_bias_batch_dim', [False, True])
@pytest.mark.parametrize('single_head_kv', [False, True])
def test_output_equal(
    causal,
    mask,
    attn_bias,
    seq_len,
    dim_head,
    float16,
    attn_bias_batch_dim,
    single_head_kv
):
    batch, heads = 4, 8
    dtype, atol = (torch.float16, 1e-1) if float16 else (torch.float32, 1e-4)

    kv_shape = (batch, heads, seq_len, dim_head) if not single_head_kv else (batch, seq_len, dim_head)

    q = torch.randn(batch, heads, seq_len, dim_head, dtype = dtype).cuda()
    k = torch.randn(kv_shape, dtype = dtype).cuda()
    v = torch.randn(kv_shape, dtype = dtype).cuda()

    attn_mask = torch.randint(0, 2, (batch, seq_len), dtype = torch.bool).cuda() if mask else None
    bias = torch.randn(batch if attn_bias_batch_dim else heads, seq_len, seq_len, dtype = dtype).cuda() if attn_bias else None

    plain_output = plain_cosine_sim_attention(q, k, v, causal = causal, mask = attn_mask, attn_bias = bias, attn_bias_batch_dim = attn_bias_batch_dim)
    flash_output = flash_cosine_sim_attention(q, k, v, causal = causal, mask = attn_mask, attn_bias = bias, attn_bias_batch_dim = attn_bias_batch_dim)

    # 断言flash_output中不存在NaN或无穷大值
    assert not_nan_or_infs(flash_output)
    # 断言plain_output和flash_output在指定的绝对误差范围内相等
    assert allclose(plain_output, flash_output, atol = atol)

# 参数化测试用例
@pytest.mark.parametrize('causal,mask', [(True, False), (False, True), (False, False)])
@pytest.mark.parametrize('attn_bias', [True, False])
@pytest.mark.parametrize('seq_len', [63, 127])
@pytest.mark.parametrize('dim_head', [32, 64, 96, 128])
@pytest.mark.parametrize('float16', [False, True])
@pytest.mark.parametrize('attn_bias_batch_dim', [False, True])
@pytest.mark.parametrize('single_head_kv', [False, True])
def test_grad_equal(
    causal,
    mask,
    attn_bias,
    seq_len,
    dim_head,
    float16,
    attn_bias_batch_dim,
    single_head_kv
):
    batch, heads = 4, 8
    dtype, atol = (torch.float16, 1e-1) if float16 else (torch.float32, 1e-4)

    kv_shape = (batch, heads, seq_len, dim_head)

    q = torch.randn(batch, heads, seq_len, dim_head, dtype = dtype).cuda().requires_grad_()
    k = torch.randn(kv_shape, dtype = dtype).cuda().requires_grad_()
    v = torch.randn(kv_shape, dtype = dtype).cuda().requires_grad_()

    attn_mask = torch.randint(0, 2, (batch, seq_len), dtype = torch.bool).cuda() if mask else None
    bias = torch.randn(batch if attn_bias_batch_dim else heads, seq_len, seq_len, dtype = dtype).cuda().requires_grad_() if attn_bias else None

    plain_output = plain_cosine_sim_attention(q, k, v, causal = causal, mask = attn_mask, attn_bias = bias, attn_bias_batch_dim = attn_bias_batch_dim)
    plain_output.sum().backward()

    dq, dk, dv = q.grad, k.grad, v.grad

    db = bias.grad if attn_bias else None

    q.grad, k.grad, v.grad = None, None, None

    if attn_bias:
        bias.grad = None

    flash_output = flash_cosine_sim_attention(q, k, v, causal = causal, mask = attn_mask, attn_bias = bias, attn_bias_batch_dim = attn_bias_batch_dim)
    flash_output.sum().backward()

    fdq, fdk, fdv = q.grad, k.grad, v.grad

    fdb = bias.grad if attn_bias else None
    # 断言 fdv 中不存在 NaN 或 无穷大值
    assert not_nan_or_infs(fdv)
    # 断言 fdk 中不存在 NaN 或 无穷大值
    assert not_nan_or_infs(fdk)
    # 断言 fdq 中不存在 NaN 或 无穷大值
    assert not_nan_or_infs(fdq)
    
    # 断言 dv 与 fdv 之间的所有元素在指定的容差范围内相等
    assert allclose(dv, fdv, atol=atol)
    
    # 断言 dk 与 fdk 之间的所有元素在指定的容差范围内相等
    assert allclose(dk, fdk, atol=atol)
    # 断言 dq 与 fdq 之间的所有元素在指定的容差范围内相等
    assert allclose(dq, fdq, atol=atol)
    
    # 如果存在注意力偏置，则断言 fdb 中不存在 NaN 或 无穷大值
    if attn_bias:
        assert not_nan_or_infs(fdb)
        # 断言 db 与 fdb 之间的所有元素在指定的容差范围内相等
        assert allclose(db, fdb, atol=atol)
# 测试 CPU 上的函数

# 参数化测试，测试不同的组合情况
@pytest.mark.parametrize('causal,mask', [(True, False), (False, True), (False, False)])
@pytest.mark.parametrize('attn_bias', [True, False])
@pytest.mark.parametrize('seq_len', [63, 127])
@pytest.mark.parametrize('dim_head', [32, 64, 96, 128])
@pytest.mark.parametrize('float16', [False, True])
@pytest.mark.parametrize('attn_bias_batch_dim', [False, True])
@pytest.mark.parametrize('single_head_kv', [False, True])
def test_output_equal_cuda_and_cpu_forward(
    causal,
    mask,
    attn_bias,
    seq_len,
    dim_head,
    float16,
    attn_bias_batch_dim,
    single_head_kv
):
    # 定义 batch 和 heads 的值
    batch, heads = 4, 8
    # 根据 float16 参数选择数据类型和容差值
    dtype, atol = (torch.float16, 1e-1) if float16 else (torch.float32, 1e-4)

    # 根据 single_head_kv 参数确定 kv_shape 的形状
    kv_shape = (batch, heads, seq_len, dim_head) if not single_head_kv else (batch, seq_len, dim_head)

    # 生成随机的 q, k, v 张量，并移动到 GPU 上
    q = torch.randn(batch, heads, seq_len, dim_head, dtype = dtype).cuda()
    k = torch.randn(kv_shape, dtype = dtype).cuda()
    v = torch.randn(kv_shape, dtype = dtype).cuda()

    # 根据 mask 参数生成注意力掩码
    attn_mask = torch.randint(0, 2, (batch, seq_len), dtype = torch.bool).cuda() if mask else None
    # 根据 attn_bias 参数生成偏置
    bias = torch.randn(batch if attn_bias_batch_dim else heads, seq_len, seq_len, dtype = dtype).cuda() if attn_bias else None

    # 在 GPU 上调用 flash_cosine_sim_attention 函数
    flash_output = flash_cosine_sim_attention(q, k, v, causal = causal, mask = attn_mask, attn_bias = bias, attn_bias_batch_dim = attn_bias_batch_dim)
    # 在 CPU 上调用 flash_cosine_sim_attention 函数
    flash_output_cpu = flash_cosine_sim_attention(q.cpu(), k.cpu(), v.cpu(), causal = causal, mask = maybe_cpu(attn_mask), attn_bias = maybe_cpu(bias), attn_bias_batch_dim = attn_bias_batch_dim)

    # 断言两个输出是否相等
    assert allclose(flash_output.cpu(), flash_output_cpu, atol = atol)
```