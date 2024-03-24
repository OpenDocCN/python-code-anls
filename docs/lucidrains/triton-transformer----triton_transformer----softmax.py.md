# `.\lucidrains\triton-transformer\triton_transformer\softmax.py`

```
# 导入 torch 库
import torch
# 从 torch 库中导入 autograd 模块
from torch import autograd
# 从 torch.nn.functional 模块中导入 F 函数
import torch.nn.functional as F

# 导入 triton 库
import triton
# 从 triton.language 模块中导入 tl
import triton.language as tl
# 从 triton_transformer.utils 模块中导入 calc_num_warps 函数
from triton_transformer.utils import calc_num_warps

# 定义 softmax_kernel_forward 函数，使用 triton.jit 装饰器
@triton.jit
def softmax_kernel_forward(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    causal,
    **meta
):
    # 获取当前程序的行索引
    row_idx = tl.program_id(0)
    # 获取 meta 字典中的 BLOCK_SIZE 值
    BLOCK_SIZE = meta['BLOCK_SIZE']

    # 计算当前行的起始指针
    row_start_ptr = input_ptr + row_idx * input_row_stride

    # 生成列偏移量
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # 计算输入指针
    input_ptrs = row_start_ptr + col_offsets

    # 创建掩码，用于处理超出列数的情况
    mask = col_offsets < n_cols

    # 从输入指针加载数据到 row 变量，处理超出列数的情况
    row = tl.load(input_ptrs, mask = mask, other = -float('inf'))

    # 如果是因果的情况，进行处理
    if causal:
        causal_mask = col_offsets > (row_idx % n_cols)
        row = row + tl.where(causal_mask, -float('inf'), 0.)

    # 计算 row 减去最大值
    row_minus_max = row - tl.max(row, axis=0)

    # 计算 softmax 的分子
    numerator = tl.exp(row_minus_max)
    # 计算 softmax 的分母
    denominator = tl.sum(numerator, axis=0)
    # 计算 softmax 输出
    softmax_output = numerator / denominator

    # 计算输出行的起始指针
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    # 计算输出指针
    output_ptrs = output_row_start_ptr + col_offsets
    # 存储 softmax 输出到输出指针，处理超出列数的情况
    tl.store(output_ptrs, softmax_output, mask = mask)

# 定义 softmax_kernel_backward 函数，使用 triton.jit 装饰器
@triton.jit
def softmax_kernel_backward(
    output_ptr,
    input_ptr,
    grad_ptr,
    grad_row_stride,
    input_row_stride,
    output_row_stride,
    n_cols,
    **meta
):
    # 获取当前程序的行索引
    row_idx = tl.program_id(0)
    # 获取 meta 字典中的 BLOCK_SIZE 值
    BLOCK_SIZE = meta['BLOCK_SIZE']

    # 计算当前行的起始指针
    row_start_ptr = input_ptr + row_idx * input_row_stride
    grad_row_start_ptr = grad_ptr + row_idx * grad_row_stride

    # 生成列偏移量
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # 计算输入指针和梯度指���
    input_ptrs = row_start_ptr + col_offsets
    grad_ptrs = grad_row_start_ptr + col_offsets

    # 创建掩码，用于处理超出列数的情况
    mask = col_offsets < n_cols

    # 从输入指针加载数据到 probs_row 变量，处理超出列数的情况
    probs_row = tl.load(input_ptrs, mask = mask, other = 0.)
    # 从梯度指针加载数据到 grad_row 变量，处理超出列数的情况
    grad_row = tl.load(grad_ptrs, mask = mask, other = 0.)

    # 计算 dxhat
    dxhat = probs_row * grad_row
    # 计算 softmax 梯度输出
    softmax_grad_output = dxhat - probs_row * tl.sum(dxhat, axis = 0)

    # 计算输出行的起始指针
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    # 计算输出指针
    output_ptrs = output_row_start_ptr + col_offsets
    # 存储 softmax 梯度输出到输出指针，处理超出列数的情况
    tl.store(output_ptrs, softmax_grad_output, mask = mask)

# 定义 _softmax 类，继承自 autograd.Function
class _softmax(autograd.Function):
    # 前向传播函数
    @classmethod
    def forward(self, ctx, x, causal):
        # 获取输入张量的形状
        shape = x.shape
        # 将输入张量展平为二维张量
        x = x.view(-1, shape[-1])
        n_rows, n_cols = x.shape

        # 计算 BLOCK_SIZE 和 num_warps
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = calc_num_warps(BLOCK_SIZE)

        # 创建与输入张量相同形状的空张量 y
        y = torch.empty_like(x)

        # 调用 softmax_kernel_forward 函数进行前向传播计算
        softmax_kernel_forward[(n_rows,)](
            y,
            x,
            x.stride(0),
            y.stride(0),
            n_cols,
            causal,
            num_warps = num_warps,
            BLOCK_SIZE = BLOCK_SIZE,
        )

        # 如果输入张量需要梯度，保存 y 用于反向传播
        if x.requires_grad:
            ctx.save_for_backward(y)
        return y.view(*shape)

    # 反向传播函数
    @classmethod
    def backward(self, ctx, grad_probs):
        # 获取梯度张量的形状
        shape = grad_probs.shape
        # 从上下文中获取保存的张量 probs
        probs, = ctx.saved_tensors

        # 将梯度张量展平为二维张量
        grad_probs = grad_probs.view(-1, grad_probs.shape[-1])
        n_rows, n_cols = grad_probs.shape

        # 计算 BLOCK_SIZE 和 num_warps
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = calc_num_warps(BLOCK_SIZE)

        # 创建与 probs 张量相同形状的空张量 dx
        dx = torch.empty_like(probs)

        # 调用 softmax_kernel_backward 函数进行反向传播计算
        softmax_kernel_backward[(n_rows,)](
            dx,
            probs,
            grad_probs,
            grad_probs.stride(0),
            probs.stride(0),
            dx.stride(0),
            n_cols,
            num_warps = num_warps,
            BLOCK_SIZE = BLOCK_SIZE
        )

        # 返回 dx 和 None，None 表示不需要额外的梯度信息
        return dx.view(*shape), None

# 定义 triton_softmax 函数，调用 _softmax 类的 apply 方法
triton_softmax = _softmax.apply

# 定义 softmax 函数，实现 softmax 操作
def softmax(x, causal = False, use_triton = False):
    # 如果使用 triton 进行计算
    if use_triton:
        # 调用 triton_softmax 函数
        return triton_softmax(x, causal)
    else:
        # 使用 PyTorch 的 F.softmax 函数
        return F.softmax(x, dim = -1)
```