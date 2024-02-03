# `bitsandbytes\bitsandbytes\triton\quantize_global.py`

```py
# 导入 torch 库
import torch

# 从 bitsandbytes.triton.triton_utils 模块中导入 is_triton_available 函数
from bitsandbytes.triton.triton_utils import is_triton_available

# 如果 Triton 不可用，则定义 quantize_global_transpose 和 quantize_global 函数返回 None
if not is_triton_available():
    def quantize_global_transpose(input): return None
    def quantize_global(x: torch.Tensor): return None
else:
    # 导入 triton 和 triton.language 模块
    import triton
    import triton.language as tl

    # 定义全局量化函数 _quantize_global
    @triton.autotune(
            configs=[
                triton.Config({'BLOCK_SIZE': 1024,}, num_warps=4),
                triton.Config({'BLOCK_SIZE': 2048,}, num_stages=1),

            ],
            key=['n_elements']
    )
    @triton.jit
    def _quantize_global(
        x_ptr,
        absmax_inv_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        # 获取程序 ID
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        absmax_inv = tl.load(absmax_inv_ptr)
        output = tl.libdevice.llrint(127. * (x * absmax_inv))
        tl.store(output_ptr + offsets, output, mask=mask)

    # 定义全局量化函数 quantize_global
    def quantize_global(x: torch.Tensor):
        absmax = x.abs().max().unsqueeze(0)
        absmax_inv = 1./ absmax
        output = torch.empty(*x.shape, device='cuda', dtype=torch.int8)
        assert x.is_cuda and output.is_cuda
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        _quantize_global[grid](x, absmax_inv, output, n_elements)
        return output, absmax

    # 定义全局量化和转置函数
    @triton.autotune(
            configs=[
                triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'GROUP_M': 8}, num_warps=4),
                triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'GROUP_M': 8}, num_warps=4),

                # ...
            ],
            key=['M', 'N']
    )
    @triton.jit
    # 定义一个函数，用于对输入矩阵进行全局量化和转置操作
    def _quantize_global_transpose(A, absmax_inv_ptr, B, stride_am, stride_an, stride_bn, stride_bm, M, N,
                          BLOCK_M : tl.constexpr,
                          BLOCK_N : tl.constexpr,
                          GROUP_M : tl.constexpr):
        # 获取当前程序的 ID
        pid = tl.program_id(0)
        # 计算 M 和 N 方向的网格数量
        grid_m = (M + BLOCK_M - 1) // BLOCK_M
        grid_n = (N + BLOCK_N - 1) // BLOCK_N

        # 计算每个组的宽度
        width = GROUP_M * grid_n
        # 计算当前程序所在的组 ID
        group_id = pid // width
        # 计算当前组的大小
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
        # 计算当前程序在 M 方向上的 ID
        pid_m = group_id * GROUP_M + (pid % group_size)
        # 计算当前程序在 N 方向上的 ID
        pid_n = (pid % width) // group_size

        # 计算当前程序需要处理的行索引
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        # 计算当前程序需要处理的列索引
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        # 更新 A 的索引
        A = A + (rm[:, None] * stride_am + rn[None, :] * stride_an)
        # 创建一个掩码，用于过滤超出矩阵范围的索引
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        # 从 A 中加载数据
        a = tl.load(A, mask=mask)
        # 从 absmax_inv_ptr 中加载数据
        absmax_inv = tl.load(absmax_inv_ptr)

        # 重新生成 rm 和 rn，以节省寄存器
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        # 更新 B 的索引
        B = B + (rm[:, None] * stride_bm + rn[None, :] * stride_bn)
        # 创建一个掩码，用于过滤超出矩阵范围的索引
        mask = (rm < M)[:, None] & (rn < N)[None, :]

        # 对数据进行量化操作
        output = tl.libdevice.llrint(127. * (a * absmax_inv))
        # 将结果存储到 B 中
        tl.store(B, output, mask=mask)

    # 定义一个函数，用于对输入矩阵进行全局量化和转置操作
    def quantize_global_transpose(input):
        # 计算输入矩阵的绝对值的最大值，并在第 0 维度上增加一个维度
        absmax = input.abs().max().unsqueeze(0)
        # 计算 absmax 的倒数
        absmax_inv = 1./ absmax
        # 获取输入矩阵的形状
        M, N = input.shape
        # 创建一个空的输出矩阵，数据类型为 torch.int8，存储在 GPU 上
        out = torch.empty(N, M, device='cuda', dtype=torch.int8)

        # 断言输出矩阵的大小和输入矩阵的大小匹配
        assert out.size(0) == N and out.size(1) == M
        # 断言输入矩阵的步长为 1 或者输出矩阵的步长为 1
        assert input.stride(0) == 1 or input.stride(1) == 1
        assert out.stride(0) == 1 or out.stride(1) == 1

        # 定义一个 lambda 函数，用于计算网格大小
        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
        # 调用 _quantize_global_transpose 函数进行全局量化和转置操作
        _quantize_global_transpose[grid](input, absmax_inv, out, input.stride(0), input.stride(1), out.stride(0), out.stride(1), M, N)
        # 返回输出矩阵和 absmax
        return out, absmax
```