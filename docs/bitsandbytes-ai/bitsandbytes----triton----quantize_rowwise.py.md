# `bitsandbytes\bitsandbytes\triton\quantize_rowwise.py`

```py
# 导入数学库
import math

# 导入 PyTorch 库
import torch

# 导入自定义模块，检查是否可用 Triton
from bitsandbytes.triton.triton_utils import is_triton_available

# 如果 Triton 不可用，则定义一个空的函数 quantize_rowwise
if not is_triton_available():
    def quantize_rowwise(x: torch.Tensor): return None
else:
    # 导入 Triton 库
    import triton
    import triton.language as tl

    # 定义一个函数 _quantize_rowwise，用于行压缩量化

    # TODO: 更好地自动调整这个函数。
    @triton.autotune(
            configs=[
                triton.Config({}, num_stages=1, num_warps=8),
                triton.Config({}, num_stages=2, num_warps=8),
                triton.Config({}, num_stages=4, num_warps=8),
                triton.Config({}, num_stages=8, num_warps=8),
                triton.Config({}, num_stages=1),
                triton.Config({}, num_stages=2),
                triton.Config({}, num_stages=4),
                triton.Config({}, num_stages=8),
                triton.Config({}, num_warps=1),
                triton.Config({}, num_warps=2),
                triton.Config({}, num_warps=4),
                triton.Config({}, num_warps=8),
            ],
            key=['n_elements']
    )
    @triton.jit
    def _quantize_rowwise(
        x_ptr,
        output_ptr,
        output_maxs,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        P2: tl.constexpr,
    ):
        # 获取程序 ID
        pid = tl.program_id(axis=0)
        # 计算块的起始位置
        block_start = pid * BLOCK_SIZE
        # 生成一个范围
        arange = tl.arange(0, P2)
        # 计算偏移量
        offsets = block_start + arange
        # 创建行掩码
        row_mask = arange < BLOCK_SIZE
        # 从内存中加载数据
        x = tl.load(x_ptr + offsets, mask=row_mask)

        # 计算绝对值
        abs_x = tl.abs(x)
        # 计算最大值
        max_val = tl.max(tl.where(row_mask, abs_x, 0), axis=0)
        # 进行量化
        output = tl.libdevice.llrint(127. * (x / max_val))
        # 将结果存储到内存中
        tl.store(output_ptr + offsets, output, mask=row_mask)
        # 存储最大值
        tl.store(output_maxs + pid, max_val)
    # 定义一个函数，用于按行对输入张量进行量化
    def quantize_rowwise(x: torch.Tensor):
        # 创建一个与输入张量相同形状的空张量，用于存储量化后的结果，数据类型为int8
        output = torch.empty(*x.shape, device=x.device, dtype=torch.int8)
        # 创建一个与输入张量行数相同的空张量，用于存储每行的最大值，数据类型为float16
        output_maxs = torch.empty(x.shape[0], device=x.device, dtype=torch.float16)

        # 计算大于等于输入张量列数的最小的2的幂次方
        P2 = int(2 ** (math.ceil(math.log2(x.shape[1]))))

        # 断言输入张量和输出张量都在GPU上
        assert x.is_cuda and output.is_cuda
        # 计算输出张量中元素的总数
        n_elements = output.numel()
        # 定义一个lambda函数，用于生成计算的网格大小
        grid = lambda meta: (x.shape[0],)
        # 调用CUDA函数_quantize_rowwise，对输入张量进行量化
        _quantize_rowwise[grid](x, output, output_maxs, n_elements, BLOCK_SIZE=x.shape[1], P2=P2)
        # 返回量化后的结果张量和每行的最大值
        return output, output_maxs
```