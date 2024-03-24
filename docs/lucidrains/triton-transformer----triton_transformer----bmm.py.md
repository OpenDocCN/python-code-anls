# `.\lucidrains\triton-transformer\triton_transformer\bmm.py`

```py
# 导入 torch 库
import torch
# 从 torch 库中导入 autograd 模块
from torch import autograd
# 从 torch.nn.functional 模块中导入 F 函数
import torch.nn.functional as F

# 从 triton_transformer.utils 模块中导入 calc_num_warps 和 exists 函数
from triton_transformer.utils import calc_num_warps, exists

# 导入 triton 库
import triton
# 从 triton.language 模块中导入 tl
import triton.language as tl

# 使用 triton.autotune 装饰器，配置自动调优参数
@triton.autotune(
    configs=[
        # 配置不同的参数组合
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64 , 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64 , 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32 , 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64 , 'BLOCK_SIZE_N': 32 , 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32 , 'BLOCK_SIZE_N': 64 , 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
# 使用 triton.jit 装饰器，编译 bmm_kernel 函数
@triton.jit
def bmm_kernel(
    x_ptr, y_ptr, o_ptr,
    M, N, K,
    stride_al, stride_am, stride_ak,
    stride_bl, stride_bk, stride_bn,
    stride_ol, stride_om, stride_on,
    **meta,
):
    # 定义常量
    BLOCK_SIZE_M = meta['BLOCK_SIZE_M']
    BLOCK_SIZE_N = meta['BLOCK_SIZE_N']
    BLOCK_SIZE_K = meta['BLOCK_SIZE_K']
    GROUP_SIZE_M = 8

    # 计算程序 ID
    pid_batch = tl.program_id(0)
    pid = tl.program_id(1)

    # 计算分组数量
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # 计算偏移量
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_am[:, None]*stride_am + offs_k [None, :]*stride_ak + pid_batch*stride_al)
    y_ptrs = y_ptr + (offs_k [:, None]*stride_bk + offs_bn[None, :]*stride_bn + pid_batch*stride_bl)

    # 初始化输出矩阵 o
    o = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # 循环计算矩阵乘法
    for k in range(0, K, BLOCK_SIZE_K):
        x = tl.load(x_ptrs)
        y = tl.load(y_ptrs)
        o += tl.dot(x, y)

        x_ptrs += BLOCK_SIZE_K * stride_ak
        y_ptrs += BLOCK_SIZE_K * stride_bk

    # 如果存在激活函数，则应用激活函数
    if exists(meta['ACTIVATION']):
        o = meta['ACTIVATION'](o)

    # 计算偏移量
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # 创建掩码
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # 计算输出指针
    o_ptrs = o_ptr + stride_om * offs_m[:, None] + stride_on * offs_n[None, :] + stride_ol * pid_batch
    # 存储结果到输出指针
    tl.store(o_ptrs, o, mask=mask)

# 定义 triton_bmm 函数
def triton_bmm(x, y, activation = None):
    # 获取 x 的形状信息
    B, M, K = x.shape

    # 如果 y 的维度为 2，则扩展维度
    if y.ndim == 2:
        y = y.unsqueeze(0).expand(B, -1, -1)

    # 获取 y 的形状信息
    _, K, N = y.shape
    # 断言 K 必须能被 32 整除
    assert (K % 32 == 0), "K must be divisible by 32"

    # 创建输出张量 o
    o = torch.empty((B, M, N), device = x.device, dtype = x.dtype)

    # 定义 grid 函数
    grid = lambda META: (
        B, triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    # 调用 bmm_kernel 函数
    bmm_kernel[grid](
        x, y, o,
        M, N, K,
        x.stride(0), x.stride(1), x.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        ACTIVATION = activation
    )
    # 返回结果张量 o
    return o

# 使用 triton.jit 装饰器，编译 relu_squared_activation 函数
@triton.jit
def relu_squared_activation(x):
    return tl.where(x > 0, x * x, 0.)

# 定义 _relu_squared 类
class _relu_squared(autograd.Function):
    # 前向传播函数
    @classmethod
    def forward(self, ctx, x, w):
        # 调用 triton_bmm 函数，应用 relu_squared_activation 激活函数
        o = triton_bmm(x, w, activation = relu_squared_activation)
        # 如果 x 需要梯度，则保存相关信息
        if x.requires_grad:
            ctx.save_for_backward(x, w, o)
        return o

    @classmethod
    # 反向传播函数，接收上下文和梯度作为输入
    def backward(self, ctx, dy):
        # 从上下文中获取保存的张量 x, w, o
        x, w, o = ctx.saved_tensors
        # 计算 dy 乘以 o 的平方根乘以 2，得到新的梯度 dy
        dy = torch.sqrt(o) * 2 * dy
        # 计算 dy 与权重 w 的转置的矩阵乘积，得到输入 x 的梯度 dx
        dx = triton_bmm(dy, w.t())
        # 计算输入 x 的转置与梯度 dy 的矩阵乘积，得到权重 w 的梯度 dw
        dw = triton_bmm(x.transpose(-1, -2), dy)
        # 返回输入 x 和权重 w 的梯度
        return dx, dw
# 将 _relu_squared.apply 赋值给 triton_relu_squared，用于后续调用
triton_relu_squared = _relu_squared.apply

# 定义一个融合了 ReLU 和平方操作的函数
def fused_relu_squared(x, w, use_triton = False):
    # 如果 use_triton 为 True，则调用 triton_relu_squared 函数
    if use_triton:
        return triton_relu_squared(x, w)

    # 如果 use_triton 为 False，则计算 x @ w 的矩阵乘法结果，然后对结果进行 ReLU 和平方操作
    return F.relu(x @ w) ** 2
```