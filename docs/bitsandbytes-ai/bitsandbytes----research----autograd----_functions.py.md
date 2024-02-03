# `bitsandbytes\bitsandbytes\research\autograd\_functions.py`

```py
from functools import reduce  # Required in Python 3
import operator
from typing import Optional
import warnings

import torch

from bitsandbytes.autograd._functions import GlobalOutlierPooler, MatmulLtState
import bitsandbytes.functional as F


# math.prod not compatible with python < 3.8
def prod(iterable):
    # 使用 reduce 函数计算可迭代对象中所有元素的乘积
    return reduce(operator.mul, iterable, 1)


class MatMulFP8Mixed(torch.autograd.Function):
    # forward is the same, but we added the fallback for pre-turing GPUs
    # backward is mostly the same, but adds one extra clause (see "elif state.CxB is not None")

    @staticmethod
    def forward(ctx, A, B, out=None, fw_code=None, bw_code=None, bsz=1024, bsz2=1024):
        # default of pytorch behavior if inputs are empty
        # 检查输入是否为空，如果是空的则返回对应形状的空张量
        ctx.is_empty = False
        if prod(A.shape) == 0:
            ctx.is_empty = True
            ctx.A = A
            ctx.B = B

            B_shape = B.shape
            if A.shape[-1] == B_shape[0]:
                return torch.empty(A.shape[:-1] + B_shape[1:], dtype=A.dtype, device=A.device)
            else:
                return torch.empty(A.shape[:-1] + B_shape[:1], dtype=A.dtype, device=A.device)

        # 1. Dequantize
        # 2. MatmulnN
        # 对输入进行量化和反量化操作，然后进行矩阵乘法
        cA, state = F.quantize_blockwise(A, code=fw_code, blocksize=bsz)
        fp8A = F.dequantize_blockwise(cA, state, blocksize=bsz).to(A.dtype)

        cB, state = F.quantize(B.float(), code=fw_code)
        fp8B = F.dequantize(cB, state).to(B.dtype)

        output = torch.matmul(fp8A, fp8B)

        # output is half

        # 3. Save state
        # 保存状态信息，用于反向传播
        ctx.fw_code = fw_code
        ctx.bw_code = bw_code
        ctx.bsz = bsz
        ctx.bsz2 = bsz2
        ctx.dtype_A, ctx.dtype_B = A.dtype, B.dtype

        if any(ctx.needs_input_grad[:2]):
            # NOTE: we send back A, and re-quant.
            # 如果需要计算输入梯度，则返回 A，并重新量化
            ctx.tensors = (A, fp8B)
        else:
            ctx.tensors = (None, None)

        return output

    @staticmethod
    # 定义反向传播函数，计算梯度
    def backward(ctx, grad_output):
        # 如果上下文为空，则返回与输入张量相同形状的零张量
        if ctx.is_empty:
            return torch.zeros_like(ctx.A), torch.zeros_like(ctx.B), None, None, None, None, None

        # 获取是否需要计算梯度的标志
        req_gradA, req_gradB, _, _, _, _, _ = ctx.needs_input_grad
        # 获取输入张量 A 和 B
        A, B = ctx.tensors

        grad_A, grad_B = None, None

        # TODO: 修复块大小为输出维度
        # 对梯度输出进行分块量化和反量化
        cgrad_out, state = F.quantize_blockwise(grad_output, code=ctx.bw_code, blocksize=ctx.bsz2)
        fp8out = F.dequantize_blockwise(cgrad_out, state, blocksize=ctx.bsz2).to(grad_output.dtype)

        # 不支持的操作，需要创建解决方法
        if req_gradA:
            # 计算 A 的梯度
            grad_A = torch.matmul(fp8out, B.t().to(fp8out.dtype)).to(A.dtype)

        if req_gradB:
            if len(A.shape) == 3:
                At = A.transpose(2, 1).contiguous()
            else:
                At = A.transpose(1, 0).contiguous()
            # 计算 B 的梯度
            grad_B = torch.matmul(At.to(grad_output.dtype), grad_output).to(B.dtype)

        return grad_A, grad_B, None, None, None, None, None
class MatMulFP8Global(torch.autograd.Function):
    # 定义一个自定义的 PyTorch 自动求导函数 MatMulFP8Global

    # forward is the same, but we added the fallback for pre-turing GPUs
    # backward is mostly the same, but adds one extra clause (see "elif state.CxB is not None")

    @staticmethod
    def forward(ctx, A, B, out=None, fw_code=None, bw_code=None, bsz=1024, bsz2=1024):
        # 定义 forward 方法，接受输入 A, B 和其他参数，执行前向传播计算

        # default of pytorch behavior if inputs are empty
        ctx.is_empty = False
        # 检查输入是否为空，如果是空的则返回对应形状的空张量
        if prod(A.shape) == 0:
            ctx.is_empty = True
            ctx.A = A
            ctx.B = B

            B_shape = B.shape
            if A.shape[-1] == B_shape[0]:
                return torch.empty(A.shape[:-1] + B_shape[1:], dtype=A.dtype, device=A.device)
            else:
                return torch.empty(A.shape[:-1] + B_shape[:1], dtype=A.dtype, device=A.device)

        # 1. Dequantize
        # 2. MatmulnN
        # 对输入 A 和 B 进行量化和反量化操作，然后进行矩阵乘法计算
        cA, state = F.quantize(A.float(), code=fw_code)
        fp8A = F.dequantize(cA, state).to(A.dtype)

        cB, state = F.quantize(B.float(), code=fw_code)
        fp8B = F.dequantize(cB, state).to(B.dtype)

        output = torch.matmul(fp8A, fp8B)

        # output is half

        # 3. Save state
        # 保存状态信息和参数
        ctx.fw_code = fw_code
        ctx.bw_code = bw_code
        ctx.bsz = bsz
        ctx.bsz2 = bsz2
        ctx.dtype_A, ctx.dtype_B = A.dtype, B.dtype

        if any(ctx.needs_input_grad[:2]):
            # NOTE: we send back A, and re-quant.
            # 如果需要计算梯度，则返回 A 和 fp8B，并重新量化
            ctx.tensors = (A, fp8B)
        else:
            ctx.tensors = (None, None)

        return output

    @staticmethod
    # 定义反向传播函数，接收上下文和梯度输出作为参数
    def backward(ctx, grad_output):
        # 如果上下文为空，则返回与输入张量相同形状的零张量和空值
        if ctx.is_empty:
            return torch.zeros_like(ctx.A), torch.zeros_like(ctx.B), None, None, None, None, None

        # 获取是否需要计算梯度的标志
        req_gradA, req_gradB, _, _, _, _, _ = ctx.needs_input_grad
        # 获取输入张量 A 和 B
        A, B = ctx.tensors

        grad_A, grad_B = None, None

        # TODO: 修复块大小为输出维度
        # 将梯度输出量化为指定精度，然后反量化为原始精度
        cgrad_out, state = F.quantize(grad_output.float(), code=ctx.bw_code)
        fp8out = F.dequantize(cgrad_out, state).to(grad_output.dtype)

        # 下面的代码块被注释掉，可能是暂时不需要执行的部分

        # 如果需要计算 A 的梯度
        if req_gradA:
            # 计算 A 的梯度
            grad_A = torch.matmul(fp8out, B.t().to(fp8out.dtype)).to(A.dtype)

        # 如果需要计算 B 的梯度
        if req_gradB:
            # 如果 A 的维度为 3，则转置 A
            if len(A.shape) == 3:
                At = A.transpose(2, 1).contiguous()
            else:
                At = A.transpose(1, 0).contiguous()
            # 将转置后的 A 量化为指定精度，然后反量化为原始精度
            cA, state = F.quantize(At.float(), code=ctx.fw_code)
            fp8At = F.dequantize(cA, state).to(A.dtype)
            # 计算 B 的梯度
            grad_B = torch.matmul(fp8At.to(fp8out.dtype), fp8out).to(B.dtype)

        # 返回计算得到的梯度
        return grad_A, grad_B, None, None, None, None, None
class SwitchBackBnb(torch.autograd.Function):
    @staticmethod
    # TODO: the B008 on the line below is a likely bug; the current implementation will
    #       have each SwitchBackBnb instance share a single MatmulLtState instance!!!
    @staticmethod
    # 定义一个静态方法，用于创建 SwitchBackBnb 类的实例

def get_block_sizes(input_matrix, weight_matrix):
    # 获取输入矩阵的特征数
    input_features = input_matrix.shape[-1]
    # 获取输出特征数，如果权重矩阵的行数等于输入特征数，则使用权重矩阵的列数，否则使用行数
    output_features = (weight_matrix.shape[0] if weight_matrix.shape[1] == input_features else weight_matrix.shape[1])
    # 定义一个数组，存储块大小
    array = [4096, 2048, 1024, 512, 256, 128, 64, 0]
    # 初始化块大小为 1024
    bsz, bsz2 = 1024, 1024
    # 遍历数组，找到适合输入特征数的块大小
    for i, k in enumerate(array):
        if input_features > array[i + 1]:
            bsz = k
            break
    # 遍历数组，找到适合输出特征数的块大小
    for i, k in enumerate(array):
        if output_features > array[i + 1]:
            bsz2 = k
            break

    return bsz, bsz2

def matmul_fp8_global(
    A: torch.Tensor,
    B: torch.Tensor,
    fw_code: torch.Tensor,
    bw_code: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    bsz: int = -1,
    bsz2: int = -1,
):
    # 如果未指定块大小，则调用 get_block_sizes 函数获取
    if bsz == -1 or bsz2 == -1: bsz, bsz2 = get_block_sizes(A, B)
    return MatMulFP8Global.apply(A, B, out, fw_code, bw_code, bsz, bsz2)

def matmul_fp8_mixed(
    A: torch.Tensor,
    B: torch.Tensor,
    fw_code: torch.Tensor,
    bw_code: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    bsz: int = -1,
    bsz2: int = -1,
):
    # 如果未指定块大小，则调用 get_block_sizes 函数获取
    if bsz == -1 or bsz2 == -1: bsz, bsz2 = get_block_sizes(A, B)
    return MatMulFP8Mixed.apply(A, B, out, fw_code, bw_code, bsz, bsz2)

def switchback_bnb(
    A: torch.Tensor,
    B: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    state: Optional[MatmulLtState] = None,
    threshold=0.0,
    bias=None
):
    # 如果未提供状态，则创建一个新的 MatmulLtState 实例
    state = state or MatmulLtState()
    # 如果阈值大于 0.0，则更新状态的阈值
    if threshold > 0.0:
        state.threshold = threshold
    return SwitchBackBnb.apply(A, B, out, bias, state)
```