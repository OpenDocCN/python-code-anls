# `.\lucidrains\triton-transformer\triton_transformer\layernorm.py`

```
# 导入 torch 库
import torch
# 从 torch 库中导入 autograd 模块
from torch import autograd
# 从 torch 库中导入 functional 模块
import torch.nn.functional as F

# 导入 triton 库
import triton
# 从 triton 库中导入 language 模块并重命名为 tl
import triton.language as tl

# 从 triton_transformer.utils 模块中导入 calc_num_warps 和 exists 函数
from triton_transformer.utils import calc_num_warps, exists

# 定义 GAMMA_BLOCK_SIZE 常量为 64
GAMMA_BLOCK_SIZE = 64
# 定义 GAMMA_ROW_BLOCK_SIZE 常量为 64
GAMMA_ROW_BLOCK_SIZE = 64

# 定义 layernorm_kernel_forward_training 函数
@triton.jit
def layernorm_kernel_forward_training(
    output_ptr,
    mean_centered_ptr,
    normed_ptr,
    input_ptr,
    gamma_ptr,
    input_row_stride,
    gamma_row_stride,
    output_row_stride,
    mean_centered_row_stride,
    normed_row_stride,
    n_cols,
    stable,
    eps,
    **meta
):
    # 获取当前程序的 ID
    row_idx = tl.program_id(0)
    # 从 meta 中获取 BLOCK_SIZE 常量
    BLOCK_SIZE = meta['BLOCK_SIZE']

    # 计算当前行的起始指针
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # 计算当前行 gamma 的起始指针
    gamma_row_start_ptr = gamma_ptr + row_idx * gamma_row_stride

    # 生成列偏移量
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # 计算当前行的输入指针
    input_ptrs = row_start_ptr + col_offsets
    # 计算当前行的 gamma 指针
    gamma_ptrs = gamma_row_start_ptr + col_offsets

    # 创建一个掩码，用于处理列偏移量小于 n_cols 的情况
    mask = col_offsets < n_cols
    # 从输入指针处加载数据到 row，如果掩码为 False，则加载 0.0
    row = tl.load(input_ptrs, mask=mask, other=0.)
    # 从 gamma 指针处加载数据到 gammas，如果掩码为 False，则加载 0.0
    gammas = tl.load(gamma_ptrs, mask=mask, other=0.)

    # 如果 stable 为 True
    if stable:
        # 计算当前行的最大值
        row_max = tl.max(tl.where(mask, row, float('-inf')), axis=0)
        # 对当前行进行归一化
        row /= row_max

    # 计算当前行的均值
    row_mean = tl.sum(row, axis=0) / n_cols
    # 计算当前行的中心化值
    row_mean_centered = tl.where(mask, row - row_mean, 0.)
    # 计算当前行的方差
    row_var = tl.sum(row_mean_centered * row_mean_centered, axis=0) / n_cols
    # 计算当前行的标准差的倒数
    inv_var = 1. / tl.sqrt(row_var + eps)
    # 计算当前行的归一化值
    normed = row_mean_centered * inv_var

    # 计算输出值
    output = normed * gammas

    # 计算输出行的起始指针
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    # 计算输出指针
    output_ptrs = output_row_start_ptr + col_offsets
    # 将输出值存储到输出指针处
    tl.store(output_ptrs, output, mask=mask)

    # 计算中心化行的起始指针
    mean_centered_row_start_ptr = mean_centered_ptr + row_idx * mean_centered_row_stride
    # 计算中心化指针
    mean_centered_ptrs = mean_centered_row_start_ptr + col_offsets
    # 将中心化值存储到中心化指针处
    tl.store(mean_centered_ptrs, row_mean_centered, mask=mask)

    # 计算归一化行的起始指针
    normed_row_start_ptr = normed_ptr + row_idx * normed_row_stride
    # 计算归一化指针
    normed_ptrs = normed_row_start_ptr + col_offsets
    # 将归一化值存储到归一化指针处
    tl.store(normed_ptrs, normed, mask=mask)

# 定义 layernorm_kernel_forward_inference 函数
@triton.jit
def layernorm_kernel_forward_inference(
    output_ptr,
    input_ptr,
    gamma_ptr,
    input_row_stride,
    gamma_row_stride,
    output_row_stride,
    n_cols,
    stable,
    eps,
    **meta
):
    # 获取当前程序的 ID
    row_idx = tl.program_id(0)
    # 从 meta 中获取 BLOCK_SIZE 常量
    BLOCK_SIZE = meta['BLOCK_SIZE']

    # 计算当前行的起始指针
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # 计算当前行 gamma 的起始指针
    gamma_row_start_ptr = gamma_ptr + row_idx * gamma_row_stride

    # 生成列偏移量
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # 计算当前行的输入指针
    input_ptrs = row_start_ptr + col_offsets
    # 计算当前行的 gamma 指针
    gamma_ptrs = gamma_row_start_ptr + col_offsets

    # 创建一个掩码，用于处理列偏移量小于 n_cols 的情况
    mask = col_offsets < n_cols
    # 从输入指针处加载数据到 row，如果掩码为 False，则加载 0.0
    row = tl.load(input_ptrs, mask=mask, other=0.)
    # 从 gamma 指针处加载数据到 gammas，如果掩码为 False，则加载 0.0
    gammas = tl.load(gamma_ptrs, mask=mask, other=0.)

    # 如果 stable 为 True
    if stable:
        # 计算当前行的最大值
        row_max = tl.max(tl.where(mask, row, float('-inf')), axis=0)
        # 对当前行进行归一化
        row /= row_max

    # 计算当前行的均值
    row_mean = tl.sum(row, axis=0) / n_cols
    # 计算当前行的中心化值
    row_mean_centered = tl.where(mask, row - row_mean, 0.)
    # 计算当前行的方差
    row_var = tl.sum(row_mean_centered * row_mean_centered, axis=0) / n_cols
    # 计算当前行的标准差的倒数
    inv_var = 1. / tl.sqrt(row_var + eps)
    # 计算当前行的归一化值
    normed = row_mean_centered * inv_var

    # 计算输出值
    output = normed * gammas

    # 计算输出行的起始指针
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    # 计算输出指针
    output_ptrs = output_row_start_ptr + col_offsets
    # 将输出值存储到输出指针处
    tl.store(output_ptrs, output, mask=mask)

# 定义 layernorm_kernel_backward 函数
@triton.jit
def layernorm_kernel_backward(
    output_ptr,
    dy_ptr,
    mean_centered_ptr,
    output_row_stride,
    dy_row_stride,
    mean_centered_row_stride,
    n_cols,
    eps,
    **meta
):
    # 获取当前程序的 ID
    row_idx = tl.program_id(0)
    # 从 meta 中获取 BLOCK_SIZE 常量
    BLOCK_SIZE = meta['BLOCK_SIZE']

    # 计算当前行的 dy 起始指针
    dy_row_start_ptr = dy_ptr + row_idx * dy_row_stride
    # 计算当前行的中心化值起始指针
    mean_centered_row_start_ptr = mean_centered_ptr + row_idx * mean_centered_row_stride

    # 生成列偏移量
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # 计算当前行的 dy 指针
    dy_ptrs = dy_row_start_ptr + col_offsets
    # 计算当前行的中心化值指针
    mean_centered_ptrs = mean_centered_row_start_ptr + col_offsets

    # 创建一个掩码，用于处理列偏移量小于 n_cols 的情况
    mask = col_offsets < n_cols

    # 从 dy 指针处加载数据到 dy，如果掩码为 False，则加载 0.0
    dy = tl.load(dy_ptrs, mask=mask, other=0.)
    # 从中心化值指针处加载数据到 mean_centered，如果掩码为 False，则加载 0.0
    mean_centered = tl.load(mean_centered_ptrs, mask=mask, other=0.)
    # 计算每行的方差
    row_var = tl.sum(mean_centered * mean_centered, axis=0) / n_cols
    # 计算每行的标准差的倒数
    inv_var = 1. / tl.sqrt(row_var + eps)
    # 对数据进行标准化处理
    normed = mean_centered * inv_var

    # 计算输出值
    output = 1. / n_cols * inv_var * (n_cols * dy - tl.sum(dy, axis=0) - normed * tl.sum(dy * normed, axis=0))

    # 计算输出行的起始指针
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    # 计算输出指针数组
    output_ptrs = output_row_start_ptr + col_offsets
    # 存储输出数据到指定的指针位置，使用掩码进行过滤
    tl.store(output_ptrs, output, mask=mask)
# 定义一个使用 Triton JIT 编译的函数，用于计算 LayerNorm 操作的 gamma 反向传播
def layernorm_gamma_kernel_backward(
    dgamma_ptr,  # 存储计算得到的 dgamma 结果的指针
    norm_ptr,  # 存储 norm 数据的指针
    dy_ptr,  # 存储 dy 数据的指针
    norm_stride,  # norm 数据的步长
    dy_stride,  # dy 数据的步长
    dgamma_row_stride,  # dgamma 行步长
    n_rows,  # 数据行数
    n_cols,  # 数据列数
    **meta  # 其他元数据
):
    # 获取当前程序的列索引和行索引
    col_idx = tl.program_id(0)
    row_idx = tl.program_id(1)
    # 从元数据中获取 BLOCK_SIZE 和 ROW_BLOCK_SIZE
    BLOCK_SIZE = meta['BLOCK_SIZE']
    ROW_BLOCK_SIZE = meta['BLOCK_SIZE_ROW']

    # 创建列偏移量和行偏移量
    col_offsets = tl.arange(0, BLOCK_SIZE)
    row_offsets = tl.arange(0, ROW_BLOCK_SIZE)

    # 计算列范围和行范围
    col_range = col_idx * BLOCK_SIZE + col_offsets
    row_range = row_idx * ROW_BLOCK_SIZE + row_offsets

    # 创建列掩码
    col_mask = col_range < n_cols
    # 创建掩码，用于过滤超出数据范围的行列
    mask = (row_range < n_rows)[:, None] & col_mask[None, :]

    # 更新 dy_ptr 和 norm_ptr 指针位置
    dy_ptr += row_range[:, None] * dy_stride + col_range[None, :]
    norm_ptr += row_range[:, None] * norm_stride + col_range[None, :]

    # 从指定位置加载 dy 和 norm 数据
    dy = tl.load(dy_ptr, mask=mask, other=0.)
    norm = tl.load(norm_ptr, mask=mask, other=0.)

    # 计算 dgamma
    dgamma = tl.sum(dy * norm, axis=0)

    # 更新 dgamma_ptr 指针位置
    dgamma_ptr += row_idx * dgamma_row_stride + col_range

    # 存储计算得到的 dgamma 结果
    tl.store(dgamma_ptr, dgamma, mask=col_mask)

# 定义一个 autograd 函数 _layernorm
class _layernorm(autograd.Function):
    @classmethod
    def forward(cls, ctx, x, gamma, eps, stable):
        # 获取输入 x 的形状和维度
        shape = x.shape
        dim = shape[-1]
        x = x.view(-1, dim)
        n_rows, n_cols = x.shape

        # 扩展 gamma 到与 x 相同的形状
        expanded_gamma = gamma[None, :].expand(n_rows, -1)

        # 计算 BLOCK_SIZE 和 num_warps
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = calc_num_warps(BLOCK_SIZE)

        # 创建一个与 x 相同形状的输出张量
        out = torch.empty_like(x)

        # 保存 eps 到上下文中
        ctx.eps = eps

        if x.requires_grad:
            # 创建 scaled_x 和 normed 张量
            scaled_x = torch.empty_like(x)
            normed = torch.empty_like(x)

            # 调用 layernorm_kernel_forward_training 函数进行前向传播计算
            layernorm_kernel_forward_training[(n_rows,)](
                out,
                scaled_x,
                normed,
                x,
                expanded_gamma,
                x.stride(0),
                expanded_gamma.stride(0),
                out.stride(0),
                scaled_x.stride(0),
                normed.stride(0),
                n_cols,
                stable,
                eps,
                num_warps=num_warps,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            # 保存 scaled_x, gamma, out 到上下文中
            ctx.save_for_backward(scaled_x, gamma, out)
        else:
            # 调用 layernorm_kernel_forward_inference 函数进行前向传播计算（无梯度）
            layernorm_kernel_forward_inference[(n_rows,)](
                out,
                x,
                expanded_gamma,
                x.stride(0),
                expanded_gamma.stride(0),
                out.stride(0),
                n_cols,
                stable,
                eps,
                num_warps=num_warps,
                BLOCK_SIZE=BLOCK_SIZE,
            )

        # 返回输出张量，并恢复原始形状
        return out.view(*shape)

    @classmethod
    def backward(cls, ctx, dy):
        # 获取 dy 的形状和设备信息
        shape, device = dy.shape, dy.device
        dim = shape[-1]
        dy = dy.view(-1, dim)

        # 从上下文中获取保存的 scaled_x, gamma, normed 张量
        scaled_x, gamma, normed = ctx.saved_tensors

        n_rows, n_cols = dy.shape

        # 计算 num_col_programs 和 num_row_programs
        num_col_programs = triton.cdiv(n_cols, GAMMA_BLOCK_SIZE)
        num_row_programs = triton.cdiv(n_rows, GAMMA_ROW_BLOCK_SIZE)

        # 创建一个用于存储 dgamma 的张量
        dgamma = torch.empty((num_row_programs, n_cols), device=device)

        # 调用 layernorm_gamma_kernel_backward 函数进行 gamma 反向传播计算
        layernorm_gamma_kernel_backward[(num_col_programs, num_row_programs)](
            dgamma,
            normed,
            dy,
            normed.stride(0),
            dy.stride(0),
            dgamma.stride(0),
            n_rows,
            n_cols,
            num_warps=4,
            BLOCK_SIZE=GAMMA_BLOCK_SIZE,
            BLOCK_SIZE_ROW=GAMMA_ROW_BLOCK_SIZE
        )

        # 对 dgamma 沿指定维度求和
        dgamma = dgamma.sum(dim=0)

        # 计算 dxhat 和 dx
        dxhat = dy * gamma
        dx = torch.empty_like(dy)

        # 计算 BLOCK_SIZE 和 num_warps
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = calc_num_warps(BLOCK_SIZE)

        # 调用 layernorm_kernel_backward 函数进行反向传播计算
        layernorm_kernel_backward[(n_rows,)](
            dx,
            dxhat,
            scaled_x,
            dx.stride(0),
            dxhat.stride(0),
            scaled_x.stride(0),
            n_cols,
            ctx.eps,
            num_warps=num_warps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # 恢复原始形状并返回 dx, dgamma
        dx = dx.view(*shape)
        return dx, dgamma, None, None
# 对输入数据进行 Layer Normalization 处理
def layernorm(x, gamma, eps = 1e-5, use_triton = False, stable = False):
    # 如果使用 Triton 加速库
    if use_triton:
        # 调用 Triton 提供的 Layer Normalization 函数
        out = _layernorm.apply(x, gamma, eps, stable)
    else:
        # 如果不使用 Triton 加速库
        if stable:
            # 对输入数据进行稳定处理，将每个元素除以最大值
            x = x / torch.amax(x, dim = -1, keepdim = True)
        # 使用 PyTorch 提供的 Layer Normalization 函数
        out = F.layer_norm(x, (x.shape[-1],), gamma, torch.zeros_like(gamma), eps = eps)
    # 返回处理后的数据
    return out
```