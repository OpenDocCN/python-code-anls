# `.\lucidrains\PaLM-pytorch\palm_pytorch\triton\softmax.py`

```py
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
    BLOCK_SIZE: tl.constexpr
):
    # 获取当前程序的 ID
    row_idx = tl.program_id(0)

    # 计算当前行的起始指针
    row_start_ptr = input_ptr + row_idx * input_row_stride

    # 计算列偏移量
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # 计算输入指针
    input_ptrs = row_start_ptr + col_offsets

    # 创建一个掩码，用于过滤超出列数的列
    mask = col_offsets < n_cols

    # 从输入指针加载数据到行
    row = tl.load(input_ptrs, mask = mask, other = -float('inf'))

    # 创建一个因果掩码
    causal_mask = col_offsets > (row_idx % n_cols)
    # 对行应用因果掩码
    row = row + tl.where(causal_mask, -float('inf'), 0.)

    # 计算行减去最大值
    row_minus_max = row - tl.max(row, axis=0)

    # 计算指数
    numerator = tl.exp(row_minus_max)
    # 计算分母
    denominator = tl.sum(numerator, axis=0)
    # 计算 softmax 输出
    softmax_output = numerator / denominator

    # 计算输出行的起始指针
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    # 计算输出指针
    output_ptrs = output_row_start_ptr + col_offsets
    # 存储 softmax 输出
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
    BLOCK_SIZE: tl.constexpr
):
    # 获取当前程序的 ID
    row_idx = tl.program_id(0)

    # 计算当前行的起始指针
    row_start_ptr = input_ptr + row_idx * input_row_stride
    grad_row_start_ptr = grad_ptr + row_idx * grad_row_stride

    # 计算列偏移量
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # 计算输入指针和梯度指针
    input_ptrs = row_start_ptr + col_offsets
    grad_ptrs = grad_row_start_ptr + col_offsets

    # 创建一个掩码，用于过滤超出列数的列
    mask = col_offsets < n_cols

    # 从输入指针加载概率行和梯度行
    probs_row = tl.load(input_ptrs, mask = mask, other = 0.)
    grad_row = tl.load(grad_ptrs, mask = mask, other = 0.)

    # 计算 dxhat
    dxhat = probs_row * grad_row
    # 计算 softmax 梯度输出
    softmax_grad_output = dxhat - probs_row * tl.sum(dxhat, axis = 0)

    # 计算输出行的起始指针
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    # 计算输出指针
    output_ptrs = output_row_start_ptr + col_offsets
    # 存储 softmax 梯度输出
    tl.store(output_ptrs, softmax_grad_output, mask = mask)

# 定义 _softmax 类，继承自 autograd.Function
class _softmax(autograd.Function):
    # 定义前向传播函数
    @classmethod
    def forward(self, ctx, x):
        # 获取输入张量的形状
        shape = x.shape
        # 将输入张量展平成二维张量
        x = x.view(-1, shape[-1])
        n_rows, n_cols = x.shape

        # 计算 BLOCK_SIZE 和 num_warps
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = calc_num_warps(BLOCK_SIZE)

        # 创建一个与输入张量相同形状的空张量
        y = torch.empty_like(x)

        # 调用 softmax_kernel_forward 函数
        softmax_kernel_forward[(n_rows,)](
            y,
            x,
            x.stride(0),
            y.stride(0),
            n_cols,
            num_warps = num_warps,
            BLOCK_SIZE = BLOCK_SIZE,
        )

        # 如果输入张量需要梯度，则保存中间结果
        if x.requires_grad:
            ctx.save_for_backward(y)
        return y.view(*shape)

    # 定义反向传播函数
    @classmethod
    def backward(self, ctx, grad_probs):
        # 获取梯度张量的形状
        shape = grad_probs.shape
        # 获取前向传播保存的中间结果
        probs, = ctx.saved_tensors

        # 将梯度张量展平成二维张量
        grad_probs = grad_probs.view(-1, grad_probs.shape[-1])
        n_rows, n_cols = grad_probs.shape

        # 计算 BLOCK_SIZE 和 num_warps
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = calc_num_warps(BLOCK_SIZE)

        # 创建一个与概率张量相同形状的空张量
        dx = torch.empty_like(probs)

        # 调用 softmax_kernel_backward 函数
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

        return dx.view(*shape), None

# 定义 causal_softmax 函数，调用 _softmax 类的 apply 方法
causal_softmax = _softmax.apply
```