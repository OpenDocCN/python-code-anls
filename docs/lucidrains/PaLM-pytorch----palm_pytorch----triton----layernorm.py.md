# `.\lucidrains\PaLM-pytorch\palm_pytorch\triton\layernorm.py`

```py
# 从 Phil Tillet 的 Triton 的 layernorm 教程中获取的代码

# Triton - https://triton-lang.org
# Layernorm 教程 - https://triton-lang.org/master/getting-started/tutorials/05-layer-norm.html#sphx-glr-getting-started-tutorials-05-layer-norm-py
# 修改为无偏置

# 导入必要的库
import torch
import triton
import triton.language as tl

# 前向传播的 Triton 内核函数
@triton.jit
def _layer_norm_fwd_fused(X, Y, W, M, V, stride, N,
                          BLOCK_SIZE: tl.constexpr):

    # 获取当前行号
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    X += row * stride
    Y += row * stride

    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N

    xmean = tl.where(mask, x - mean, 0.)
    var = tl.sum(xmean * xmean, axis=0) / N
    rstd = 1 / tl.sqrt(var + 1e-5)
    xhat = xmean * rstd

    tl.store(M + row, mean)
    tl.store(V + row, rstd)

    w = tl.load(W + cols, mask=mask)
    y = xhat * w

    tl.store(Y + cols, y, mask=mask)

# 反向传播的 Triton 内核函数
@triton.jit
def _layer_norm_bwd_dx_fused(DX, DY, DW, X, W, M, V, Lock, stride, N,
                             GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):

    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N

    X += row * stride
    DY += row * stride
    DX += row * stride

    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW = DW + lock_id * N + cols

    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    mean = tl.load(M + row)
    rstd = tl.load(V + row)

    xhat = (x - mean) * rstd
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.)
    wdy = tl.where(mask, wdy, 0.)
    mean1 = tl.sum(xhat * wdy, axis=0) / N
    mean2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * mean1 + mean2)) * rstd
    
    tl.store(DX + cols, dx, mask=mask)

    partial_dw = (dy * xhat).to(w.dtype)

    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)

    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)

    tl.store(DW, partial_dw, mask=mask)

    tl.atomic_xchg(Lock, 0)

# 计算权重梯度的 Triton 内核函数
@triton.jit
def _layer_norm_bwd_dw(DW, FINAL_DW, M, N,
                         BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.)

    sum_dw = tl.sum(dw, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)

# LayerNorm 类，继承自 torch.autograd.Function
class LayerNorm(torch.autograd.Function):

    # 前向传播函数
    @staticmethod
    def forward(ctx, x, normalized_shape, weight):
        y = torch.empty_like(x)

        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        mean = torch.empty((M, ), dtype=torch.float32, device='cuda')
        rstd = torch.empty((M, ), dtype=torch.float32, device='cuda')

        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

        _layer_norm_fwd_fused[(M,)](x_arg, y, weight, mean, rstd,
                                    x_arg.stride(0), N,
                                    BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
        ctx.save_for_backward(x, weight, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        return y

    @staticmethod
    # 反向传播函数，计算梯度
    def backward(ctx, dy):
        # 从上下文中获取保存的张量 x, w, m, v
        x, w, m, v = ctx.saved_tensors

        # 获取 w 的形状信息
        N = w.shape[0]
        GROUP_SIZE_M = 64
        # 根据 w 的大小确定 GROUP_SIZE_M 的值
        if N <= 8192: GROUP_SIZE_M = 96
        if N <= 4096: GROUP_SIZE_M = 128
        if N <= 1024: GROUP_SIZE_M = 256

        # 创建用于同步的锁
        locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device='cuda')
        # 创建用于存储梯度的 _dw 张量
        _dw = torch.empty((GROUP_SIZE_M, w.shape[0]), dtype=x.dtype, device=w.device)

        # 创建用于存储 w 梯度的 dw 张量
        dw = torch.empty((w.shape[0],), dtype=w.dtype, device=w.device)
        # 创建用于存储输入 x 的梯度的 dx 张量
        dx = torch.empty_like(dy)

        # 将输入 x 重塑为二维张量
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        # 调用 _layer_norm_bwd_dx_fused 函数计算 dx
        _layer_norm_bwd_dx_fused[(M,)](dx, dy, _dw, x, w, m, v, locks,
                                       x_arg.stride(0), N,
                                       BLOCK_SIZE_N=ctx.BLOCK_SIZE,
                                       GROUP_SIZE_M=GROUP_SIZE_M,
                                       num_warps=ctx.num_warps)
        # 定义 grid 函数用于计算网格大小
        grid = lambda meta: [triton.cdiv(N, meta['BLOCK_SIZE_N'])]

        # 调用 _layer_norm_bwd_dw 函数计算 dw
        _layer_norm_bwd_dw[grid](_dw, dw, GROUP_SIZE_M, N,
                                   BLOCK_SIZE_M=32,
                                   BLOCK_SIZE_N=128)
        # 返回计算得到的 dx 和 dw
        return dx, None, dw, None
# 将LayerNorm类的apply方法赋值给layernorm_without_bias变量
layernorm_without_bias = LayerNorm.apply
```