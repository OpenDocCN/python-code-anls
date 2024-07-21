# `.\pytorch\torch\sparse\_triton_ops.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和库
import math  # 导入数学模块
import os  # 导入操作系统接口模块
import torch  # 导入PyTorch库
import weakref  # 导入弱引用模块
from functools import lru_cache  # 从functools模块中导入lru_cache函数
from torch.utils._triton import has_triton  # 从torch.utils._triton中导入has_triton函数
from ._triton_ops_meta import get_meta  # 从当前模块的_triton_ops_meta中导入get_meta函数
from typing import Optional, Tuple  # 导入类型提示中的Optional和Tuple类型

# 从环境变量中获取并设置TORCH_SPARSE_BSR_SCATTER_MM_LRU_CACHE_SIZE
TORCH_SPARSE_BSR_SCATTER_MM_LRU_CACHE_SIZE = int(os.getenv('TORCH_SPARSE_BSR_SCATTER_MM_LRU_CACHE_SIZE', 2))


# 自定义检查函数，如果条件cond不满足，则抛出ValueError异常并显示msg消息
def check(cond, msg):
    if not cond:
        raise ValueError(msg)


# 检查稀疏张量t的布局是否为torch.sparse_bsr，否则抛出异常
def check_bsr_layout(f_name, t):
    check(
        t.layout == torch.sparse_bsr,
        f"{f_name}(): only BSR sparse format is supported for the sparse argument.",
    )


# 检查稀疏张量t的设备是否为指定的device且设备类型为cuda，否则抛出异常
def check_device(f_name, t, device):
    check(
        t.device == device and t.device.type == "cuda",
        f"{f_name}(): all inputs are expected to be on the same GPU device.",
    )


# 检查参与矩阵乘积的两个张量lhs和rhs是否维度至少为2，否则抛出异常
def check_mm_compatible_shapes(f_name, lhs, rhs):
    check(
        lhs.dim() >= 2 and rhs.dim() >= 2,
        f"{f_name}(): all inputs involved in the matrix product are expected to be at least 2D, "
        f"but got lhs.dim() == {lhs.dim()} and rhs.dim() == {rhs.dim()}."
    )

    m, kl = lhs.shape[-2:]
    kr, n = rhs.shape[-2:]

    # 检查lhs和rhs的最后两个维度是否匹配，否则抛出异常
    check(
        kl == kr,
        f"{f_name}(): arguments' sizes involved in the matrix product are not compatible for matrix multiplication, "
        f"got lhs.shape[-1] == {kl} which is not equal to rhs.shape[-2] == {kr}.",
    )


# 检查稀疏张量t的dtype是否为指定的dtype及其他可选的additional_dtypes，否则抛出异常
def check_dtype(f_name, t, dtype, *additional_dtypes):
    check(
        t.dtype == dtype
        and t.dtype in ((torch.half, torch.bfloat16, torch.float) + tuple(*additional_dtypes)),
        f"{f_name}(): all inputs are expected to be of the same dtype "
        f"and one of (half, bfloat16, float32) or {additional_dtypes}, "
        f"but got dtype == {t.dtype}.",
    )


# 检查块大小是否满足特定条件，如果不满足则抛出异常
def check_blocksize(f_name, blocksize):
    assert len(blocksize) == 2

    # 检查块大小是否为大于等于16且为2的幂次方，否则抛出异常
    def is_power_of_two(v):
        return not (v & (v - 1))

    def is_compatible_blocksize(b):
        res = True
        for blocksize in b:
            # Triton加载的块大小必须至少为16且是2的幂次方
            res = (blocksize >= 16 and is_power_of_two(blocksize)) and res
        return res

    check(
        is_compatible_blocksize(blocksize),
        f"{f_name}(): sparse inputs' blocksize ({blocksize[0]}, {blocksize[1]}) "
        "should be at least 16 and a power of 2 in each dimension.",
    )


# 返回一个triton-contiguous张量，如果输入张量t不符合条件则进行连续化操作
def make_triton_contiguous(t):
    """Return input as a triton-contiguous tensor.

    A triton-contiguous tensor is defined as a tensor that has strides
    with minimal value equal to 1.

    While triton kernels support triton-non-contiguous tensors (all
    strides being greater than 1 or having 0 strides) arguments, a
    considerable slow-down occurs because tensor data is copied
    element-wise rather than chunk-wise.
    """
    if min(t.stride()) != 1:
        # 如果最小的步长不为1，则进行连续化操作
        # TODO: investigate if contiguity along other axes than the
        # last one can be beneficial for performance
        return t.contiguous()
    else:
        # 如果已经是按照triton-contiguous的要求，则直接返回原始张量
        return t
# 定义一个函数，用于计算一批张量的广播维度
def broadcast_batch_dims(f_name, *tensors):
    try:
        # 调用 PyTorch 的 broadcast_shapes 函数，返回张量各自的广播维度形状
        return torch.broadcast_shapes(*(t.shape[:-2] for t in tensors))
    except Exception:
        # 如果广播维度失败，则触发检查点错误，并输出错误信息
        check(False, f"{f_name}(): inputs' batch dimensions are not broadcastable!")


# 定义一个生成器函数，用于在指定维度上切片一批张量
def slicer(dim, slice_range, *tensors):
    for t in tensors:
        slices = [slice(None)] * t.dim()
        slices[dim] = slice_range
        yield t[slices]


# 定义一个生成器函数，用于在多个维度上按照给定的切片范围切片一批张量
def multidim_slicer(dims, slices, *tensors):
    for t in tensors:
        s = [slice(None)] * t.dim()
        for d, d_slice in zip(dims, slices):
            if d is not None:
                s[d] = d_slice
        yield t[s]


# 定义一个生成器函数，用于提取一批张量的指针和步长信息
def ptr_stride_extractor(*tensors):
    for t in tensors:
        yield t
        yield from t.stride()


# 定义一个函数，用于将整个网格分割为块，并切片相关张量以便进行操作
def grid_partitioner(full_grid, grid_blocks, tensor_dims_map):
    assert 0 <= len(full_grid) <= 3
    assert 0 <= len(grid_blocks) <= 3

    import itertools

    # 定义生成网格点的函数
    def generate_grid_points():
        for fg, mg in zip(full_grid, grid_blocks):
            yield range(0, fg, mg)

    # 定义生成切片张量的函数
    def generate_sliced_tensors(slices):
        for t, t_dims in tensor_dims_map.items():
            yield next(multidim_slicer(t_dims, slices, t))

    # 使用 itertools.product 生成网格中的每个点
    for grid_point in itertools.product(*generate_grid_points()):
        # 计算每个网格块的大小
        grid = [min(fg - gp, mg) for fg, gp, mg in zip(full_grid, grid_point, grid_blocks)]
        # 构建每个网格块的切片范围
        slices = [slice(gp, gp + g) for gp, g in zip(grid_point, grid)]
        # 输出说明：网格点按照“连续”的顺序迭代，即左侧维度比右侧维度慢。
        # 对于 CUDA 网格，这种顺序是相反的。
        yield grid[::-1], *generate_sliced_tensors(slices)


# 定义一个函数，用于启动一个 CUDA 核函数，并管理相关的张量维度映射、网格和块大小
def launch_kernel(kernel, tensor_dims_map, full_grid, grid_blocks=None):
    # CUDA 的最大网格大小限制
    cuda_max_grid = (2147483647, 65535, 65535)[::-1]
    # 如果未提供网格块大小，则使用默认的 CUDA 最大网格大小
    if grid_blocks is None:
        grid_blocks = cuda_max_grid
    else:
        # 定义函数，用于验证网格维度是否有效
        def valid_grid_dim(g, mg):
            if g is None:
                return mg
            else:
                # 确保网格大小至少为 1，且不超过最大网格大小
                return max(1, min(g, mg))

        # 对给定的网格块大小进行验证和调整
        grid_blocks = tuple(
            valid_grid_dim(g, mg) for g, mg in zip(grid_blocks, cuda_max_grid)
        )  # type: ignore[assignment]

    # 使用 grid_partitioner 函数生成的网格块，迭代执行指定的核函数
    for grid, *sliced_tensors in grid_partitioner(full_grid, grid_blocks, tensor_dims_map):
        kernel(grid, *sliced_tensors)


# 定义一个函数，用于准备输入数据，处理稀疏张量和密集张量，并确保存在批次维度以便处理
def prepare_inputs(bsr, *dense_tensors):
    # 如果不存在批次维度，则引入一个虚拟的批次维度以方便处理
    crow_indices = bsr.crow_indices().unsqueeze(0)
    col_indices = bsr.col_indices().unsqueeze(0)
    values = make_triton_contiguous(bsr.values().unsqueeze(0))
    tensors = [make_triton_contiguous(t.unsqueeze(0)) for t in dense_tensors]

    # 计算广播后的批次维度形状
    batch_dims_broadcasted = torch.broadcast_shapes(values.shape[:-3], *(t.shape[:-2] for t in tensors))

    # 广播批次维度并压缩
    # 结果可以是一个视图或者是一个复制
    # 定义一个函数，用于在张量 `t` 上进行批处理广播和压缩操作，返回结果张量扁平化后的形状
    def batch_broadcast_and_squash(t, batch_dims, invariant_dims):
        return t.broadcast_to(batch_dims + invariant_dims).flatten(
            0, len(batch_dims) - 1
        )
    
    # 对 `crow_indices` 进行批处理广播和压缩操作，将其扁平化为指定形状
    crow_indices = batch_broadcast_and_squash(
        crow_indices, batch_dims_broadcasted, (-1,)
    )
    
    # 对 `col_indices` 进行批处理广播和压缩操作，将其扁平化为指定形状
    col_indices = batch_broadcast_and_squash(
        col_indices, batch_dims_broadcasted, (-1,)
    )
    
    # 对 `values` 进行批处理广播和压缩操作，将其扁平化为指定形状，该形状由 `values` 的最后三个维度决定
    values = batch_broadcast_and_squash(
        values, batch_dims_broadcasted, values.shape[-3:]
    )
    
    # 对列表 `tensors` 中的每个张量 `t` 进行批处理广播和压缩操作，将它们扁平化为指定形状，该形状由每个张量 `t` 的最后两个维度决定
    tensors = [
        batch_broadcast_and_squash(t, batch_dims_broadcasted, t.shape[-2:]) for t in tensors
    ]
    
    # 返回压缩后的 `crow_indices`、`col_indices`、`values`，以及所有 `tensors` 的扁平化结果
    return crow_indices, col_indices, values, *tensors
# 根据给定的参数 `f_name`, `bsr` 以及多个张量 `tensors`，广播它们的批次维度，返回广播后的批次形状
def broadcast_batch_dims_bsr(f_name, bsr, *tensors):
    # 调用 `broadcast_batch_dims` 函数计算并返回批次形状
    batch_shape = broadcast_batch_dims(f_name, bsr, *tensors)

    # 获取 BSR 稀疏矩阵的行压缩索引，并将其广播到批次形状加上 (-1,) 的形状
    crow_indices = bsr.crow_indices().broadcast_to(batch_shape + (-1,))
    # 获取 BSR 稀疏矩阵的列索引，并将其广播到批次形状加上 (-1,) 的形状
    col_indices = bsr.col_indices().broadcast_to(batch_shape + (-1,))
    # 获取 BSR 稀疏矩阵的值，并将其广播到批次形状加上 BSR 矩阵值的最后三个维度的形状
    values = bsr.values().broadcast_to(batch_shape + bsr.values().shape[-3:])
    # 计算稀疏张量的大小，形状为批次形状加上 BSR 矩阵的最后两个维度
    size = batch_shape + bsr.shape[-2:]
    # 使用给定的广播后的数据创建稀疏压缩张量，并返回
    return torch.sparse_compressed_tensor(crow_indices, col_indices, values, size=size, layout=bsr.layout)


# NOTE: this function will ALWAYS create a view
# 将输入张量 `t` 重新调整为指定的 `blocksize` 块大小的形状，保证结果是一个视图而非新的数据拷贝
def tile_to_blocksize(t, blocksize):
    # 获取除最后两个维度外的其余维度，并分别获取最后两个维度的大小 m 和 n
    *rest, m, n = t.shape
    # 计算新的形状，将 t 调整为指定块大小的形状
    new_shape = rest + [
        m // blocksize[0],
        blocksize[0],
        n // blocksize[1],
        blocksize[1],
    ]
    # 使用 .view 方法而非 .reshape 方法来确保返回的是视图
    return t.view(new_shape).transpose(-3, -2)


# 将输入的张量 `tensor` 转换为一个三维张量，方法是在张量的形状前面插入新的维度（当 `tensor.ndim < 3` 时）或将起始维度合并为第一维度（当 `tensor.ndim > 3` 时）
def as1Dbatch(tensor):
    while tensor.ndim < 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim > 3:
        tensor = tensor.flatten(0, tensor.ndim - 3)
    assert tensor.ndim == 3, tensor.shape
    return tensor


# 执行分散矩阵乘法操作，根据指定的索引数据 `indices_data` 在输入的 `blocks` 和 `others` 张量上执行矩阵乘法，并将结果累加到给定的 `accumulators` 张量上
def scatter_mm(blocks, others, indices_data, *, accumulators=None):
    """Scattered matrix multiplication of tensors.

    A scattered matrix multiplication is defined as a series of matrix
    multiplications applied to input tensors according to the input
    and output mappings specified by indices data.

    The following indices data formats are supported for defining a
    scattered matrix multiplication operation (:attr:`indices_data[0]`
    holds the name of the indices data format as specified below):

    - ``"scatter_mm"`` - matrix multiplications scattered in batches
      of tensors.

      If :attr:`blocks` is a :math:`(* \times M \times K) tensor,
      :attr:`others` is a :math:`(* \times K \times N)` tensor,
      :attr:`accumulators` is a :math:`(* \times M \times N)` tensor,
      and :attr:`indices = indices_data['indices']` is a :math:`(*
      \times 3)` tensor, then the operation is equivalent to the
      following code::

        c_offsets, pq = indices_data[1:]
        for r in range(len(c_offsets) - 1):
            for g in range(c_offsets[r], c_offsets[r + 1]):
                p, q = pq[g]
                accumulators[r] += blocks[p] @ others[q]
    """
    # "bsr_strided_mm" - 稀疏分块矩阵乘法，将张量批次和张量分块进行乘法操作。
    
    # 如果 blocks 是一个 (Ms × Ks) 的张量，
    # others 是一个 (* × K × N) 的张量，
    # accumulators 是一个 (* × M × N) 的张量，
    # 则该操作等同于以下代码：
    
    c_indices, r_offsets, p_offsets, q_offsets, meta = indices_data[1:]
    # 解包 indices_data 中的参数，包括列索引、行偏移、P 偏移、Q 偏移和元数据
    
    for b in range(nbatches):
        # 遍历批次数
        for i, r in enumerate(r_offsets):
            # 遍历行偏移列表中的每一项
            r0, r1 = divmod(r, N)
            # 计算行索引的起始和终止位置
            acc = accumulators[b, r0:r0 + Ms, r1:r1 + Ns]
            # 获取累加器的对应部分
            for g in range(c_indices[i], c_indices[i+1]):
                # 遍历列索引范围内的每一个元素
                p = p_offsets[g]
                # 获取P偏移量中的值
                q0, q1 = divmod(q_offsets[g], N)
                # 计算Q偏移的起始位置和终止位置
                acc += blocks[p] @ others[b, q0:q0 + Ks, q1:q1 + Ns]
                # 执行块矩阵乘法运算
    
    # 其中 Ns = N // meta['SPLIT_N']，M 和 K 是 Ms 和 Ks 的整数倍。
    
    # "bsr_strided_mm_compressed" - 压缩的稀疏分块矩阵乘法，是 "bsr_strided_mm" 的内存和处理器高效版本。
    
    # 如果 blocks 是一个 (Ms × Ks) 的张量，
    # others 是一个 (* × K × N) 的张量，
    # accumulators 是一个 (* × M × N) 的张量，
    # 则该操作等同于以下代码：
    
    c_indices, r_offsets, q_offsets, meta = indices_data[1:]
    # 解包 indices_data 中的参数，包括列索引、行偏移、Q 偏移和元数据
    
    for b in range(nbatches):
        # 遍历批次数
        for r in r_offsets:
            # 遍历行偏移列表中的每一项
            m = (r // N) // Ms
            n = (r % N) // Ns
            r0, r1 = divmod(r, N)
            # 计算行索引的起始和终止位置
            c0, c1 = c_indices[m], c_indices[m + 1]
            # 获取列索引的起始和终止位置
            acc = accumulators[b, r0:r0 + Ms, r1:r1 + Ns]
            # 获取累加器的对应部分
            for i, p in enumerate(range(c0, c1)):
                # 遍历列索引范围内的每一个元素
                q = q_offsets[n * c1 + (SPLIT_N - n) * c0 + i]
                # 计算Q偏移的位置
                q0, q1 = divmod(q, N)
                # 计算Q偏移的起始位置和终止位置
                acc += blocks[p] @ others[b, q0:q0 + Ks, q1:q1 + Ns]
                # 执行块矩阵乘法运算
    
    # 其中 Ns = N // meta['SPLIT_N']，M 和 K 是 Ms 和 Ks 的整数倍。
    
    # 注意，r_offsets 中的项的顺序可以是任意的；这个属性允许通过重新排列 r_offsets 的项来定义 swizzle 操作符。
    
    # 提供了用于预计算 indices_data 的辅助函数，例如 bsr_scatter_mm_indices_data 用于定义 BSR 和分块张量的矩阵乘法的索引数据。
    
    # 参数：
    # blocks（Tensor）：第一个要相乘的矩阵的三维张量。
    # others（Tensor）：要相乘的第二个矩阵。如果 indices_data[0]=="scatter_mm"，则该张量是第二个输入矩阵的批量张量。否则，第二个输入矩阵是 others 张量的切片。
    indices_data (tuple): a format data that defines the inputs and
      outputs of scattered matrix multiplications.
    """
    # 解析输入参数，indices_data 是一个元组，用于定义散列矩阵乘法的输入和输出
    indices_format = indices_data[0]

    # 确保 blocks 的维度为3
    assert blocks.ndim == 3
    P, Ms, Ks = blocks.shape

    # 根据 indices_format 的值进行不同的操作
    if indices_format == 'scatter_mm':
        # 解析 indices_data 中的具体参数
        c_offsets, pq = indices_data[1:]

        # 确保 others 的维度为3，同时确认维度匹配条件
        assert others.ndim == 3
        Q, Ks_, Ns = others.shape
        assert Ks == Ks_

        # 如果 accumulators 为 None，则初始化为零张量
        if accumulators is None:
            R = c_offsets.shape[0] - 1
            accumulators = torch.zeros((R, Ms, Ns), dtype=blocks.dtype, device=blocks.device)
        else:
            # 否则，确认 accumulators 的维度匹配条件
            R, Ms_, Ns_ = accumulators.shape
            assert Ms_ == Ms
            assert Ns_ == Ns

        # 如果任何一个维度不是16的倍数或者没有提供 _scatter_mm2 函数，则使用循环计算
        if Ms % 16 or Ks % 16 or Ns % 16 or _scatter_mm2 is None:
            # 遍历 c_offsets 中的索引范围
            for r in range(c_offsets.shape[0] - 1):
                g0 = c_offsets[r]
                g1 = c_offsets[r + 1]
                # 遍历每个索引 g 并计算累加器
                for g in range(g0, g1):
                    p, q = pq[g]
                    accumulators[r] += blocks[p] @ others[q]
        else:
            # 否则，调用优化后的函数 _scatter_mm2 进行计算
            _scatter_mm2(blocks, others, c_offsets, pq, accumulators)

        # 返回计算后的 accumulators 张量
        return accumulators

    elif indices_format == 'bsr_strided_mm':
        # 解析 indices_data 中的具体参数
        others_shape = others.shape
        others = as1Dbatch(others)

        B, K, N = others.shape
        assert K % Ks == 0

        c_indices, r_offsets, p_offsets, q_offsets, meta = indices_data[1:]
        SPLIT_N = meta['SPLIT_N']

        # 如果 accumulators 为 None，则根据 r_offsets 的最大值初始化 M 维度
        if accumulators is None:
            M = Ms + (r_offsets.max().item() + 1) // N
            accumulators = torch.zeros((*others_shape[:-2], M, N), dtype=blocks.dtype, device=blocks.device)
        else:
            # 否则，确认 accumulators 的最后两个维度匹配条件
            M, N_ = accumulators.shape[-2:]
            assert N_ == N

        # 重新定义 accumulators 的形状，并调用 as1Dbatch 进行处理
        accumulators_shape = accumulators.shape
        accumulators = as1Dbatch(accumulators)

        Ns = N // SPLIT_N

        # 如果任何一个维度不是16的倍数或者没有提供 _scatter_mm6 函数，则使用循环计算
        if Ms % 16 or Ks % 16 or Ns % 16 or _scatter_mm6 is None:
            # 清零 accumulators
            accumulators.zero_()
            # 遍历 B 和 r_offsets 中的索引范围
            for b in range(B):
                for r in range(r_offsets.shape[0]):
                    r_ = r_offsets[r].item()
                    g0 = c_indices[r].item()
                    g1 = c_indices[r + 1].item()
                    r0, r1 = divmod(r_, N)
                    acc = accumulators[b, r0:r0 + Ms, r1:r1 + Ns]
                    # 遍历每个索引 g 并计算累加器
                    for g in range(g0, g1):
                        p, q = p_offsets[g], q_offsets[g]
                        q0, q1 = divmod(q.item(), N)
                        acc += blocks[p] @ others[b, q0:q0 + Ks, q1:q1 + Ns]
        else:
            # 否则，调用优化后的函数 _scatter_mm6 进行计算
            _scatter_mm6(blocks, others, c_indices, r_offsets, p_offsets, q_offsets, meta, accumulators)

        # 返回形状恢复后的 accumulators 张量
        return accumulators.view(accumulators_shape)
    # 如果索引格式为 'bsr_strided_mm_compressed'
    elif indices_format == 'bsr_strided_mm_compressed':
        # 获取 others 的形状，并保存原始形状
        others_shape = others.shape
        # 将 others 转换为一维批量
        others = as1Dbatch(others)

        # 获取 B, K, N 的值，这些是 others 的维度
        B, K, N = others.shape
        # 断言 K 能够整除 Ks
        assert K % Ks == 0

        # 解压 indices_data 的各个部分
        c_indices, r_offsets, q_offsets, meta = indices_data[1:]
        # 获取 SPLIT_N 的值
        SPLIT_N = meta['SPLIT_N']

        # 如果 accumulators 为空
        if accumulators is None:
            # 计算 M 的值，并确保 Ms 是 M 的初始值
            M = Ms + (r_offsets.max().item() + 1) // N
            # 创建一个与 others_shape 除了最后两个维度外相同的零张量作为 accumulators
            accumulators = torch.zeros((*others_shape[:-2], M, N), dtype=blocks.dtype, device=blocks.device)
        else:
            # 否则，获取 accumulators 的最后两个维度的大小，并确保 N_ 等于 N
            M, N_ = accumulators.shape[-2:]
            assert N_ == N

        # 保存 accumulators 的形状，并将其转换为一维批量
        accumulators_shape = accumulators.shape
        accumulators = as1Dbatch(accumulators)

        # 计算 Ns 的值，即 N 除以 SPLIT_N
        Ns = N // SPLIT_N

        # 如果 Ms、Ks、Ns 不是 16 的倍数，或者 _scatter_mm6 为空
        if Ms % 16 or Ks % 16 or Ns % 16 or _scatter_mm6 is None:
            # 对每个批次中的每个元素进行循环
            for b in range(B):
                # 对 r_offsets 中的每个元素进行循环
                for j in range(len(r_offsets)):
                    # 计算 r_offsets[j] 的商和余数
                    r0, r1 = divmod(r_offsets[j].item(), N)
                    # 计算 m、n 和 c0、c1
                    m = r0 // Ms
                    n = r1 // Ns
                    c0 = c_indices[m].item()
                    c1 = c_indices[m + 1].item()
                    # 获取 accumulators 的部分并与 blocks 和 others 的乘积相加
                    acc = accumulators[b, r0:r0 + Ms, r1:r1 + Ns]
                    for i, p in enumerate(range(c0, c1)):
                        q = q_offsets[n * c1 + (SPLIT_N - n) * c0 + i].item()
                        q0, q1 = divmod(q, N)
                        acc += blocks[p] @ others[b, q0:q0 + Ks, q1:q1 + Ns]
        else:
            # 否则，创建一个空的 p_offsets 张量，并使用 _scatter_mm6 函数
            p_offsets = torch.empty((0, ), dtype=q_offsets.dtype, device=q_offsets.device)
            _scatter_mm6(blocks, others, c_indices, r_offsets, p_offsets, q_offsets, meta, accumulators)
        
        # 返回重塑后的 accumulators
        return accumulators.view(accumulators_shape)

    # 如果索引格式不支持，则抛出未实现错误
    else:
        raise NotImplementedError(indices_format)
# 定义一个函数，用于计算scatter_mm操作的元数据，包括矩阵尺寸和分割信息
def scatter_mm_meta(M, K, N, Ms, Ks,
                    GROUP_SIZE=None, TILE_M=None, TILE_N=None, SPLIT_N=None, num_warps=None, num_stages=None, **extra):
    # 如果未指定SPLIT_N，则根据N的大小进行分割策略选择
    if SPLIT_N is None:
        # 假设为NVIDIA GeForce RTX 2060 SUPER，根据N的大小选择合适的分割因子
        SPLIT_N = {16: 1, 32: 2, 64: 4, 128: 8, 256: 16, 512: 8, 1024: 16, 4096: 32, 8192: 64}.get(N, 16)
        # 当Ms大于等于512且N大于等于2048时，优先选择SPLIT_N为1
        if Ms >= 512 and N >= 2048:
            SPLIT_N = 1
    # 根据选择的SPLIT_N计算Ns（分割后的N大小）
    Ns = N // SPLIT_N
    # 如果未指定TILE_M，则根据Ns的大小选择合适的TILE_M
    if TILE_M is None:
        TILE_M = min(64 if Ns < 512 else 32, Ms)
    # 如果未指定TILE_N，则根据Ns的大小选择合适的TILE_N
    if TILE_N is None:
        TILE_N = min(64 if Ns < 512 else 32, Ns)
    # 如果未指定num_stages，则设置为1
    num_stages = num_stages or 1
    # 如果未指定num_warps，则根据M和N的最小值选择合适的num_warps
    if num_warps is None:
        if min(M, N) > 1024:
            num_warps = {16: 1, 32: 1, 64: 2}.get(Ms, 4)
        elif min(M, N) == 1024:
            num_warps = {16: 1, 32: 1, 64: 2}.get(Ms, 4)
        elif min(M, N) == 256:
            num_warps = {16: 1, 32: 4}.get(Ms, 4)
        else:
            num_warps = {16: 1, 32: 2}.get(Ms, 4)
    # 如果未指定GROUP_SIZE，则设置为4
    GROUP_SIZE = GROUP_SIZE or 4

    # 断言确保选择的TILE_M和TILE_N不超出允许的Ms和Ns大小
    assert TILE_M <= Ms, dict(TILE_M=TILE_M, Ms=Ms)
    assert TILE_N <= Ns, dict(TILE_N=TILE_N, Ns=Ns)
    # 断言确保Ms和Ns不超出给定的M和N的大小
    assert Ms <= M, dict(M=M, Ms=Ms)
    assert Ns <= N, dict(N=N, Ns=Ns)
    # 断言确保Ks不超出给定的K的大小
    assert Ks <= K, dict(K=K, Ks=Ks)

    # 返回计算得到的元数据字典
    return dict(TILE_M=TILE_M, TILE_N=TILE_N, GROUP_SIZE=GROUP_SIZE,
                num_stages=num_stages, num_warps=num_warps, SPLIT_N=SPLIT_N, **extra)


# 定义一个函数，用于计算bsr_dense_addmm操作的元数据，包括矩阵尺寸、稀疏度和分割信息
def bsr_dense_addmm_meta(M, K, N, Ms, Ks, beta, alpha,
                         SPLIT_N=None, GROUP_SIZE_ROW=None, num_warps=None, num_stages=None, sparsity=None, dtype=None, **extra):
    # 如果未指定dtype，则默认为torch.float16
    if dtype is None:
        dtype = torch.float16
    # 如果未指定sparsity，则默认为0.5
    if sparsity is None:
        sparsity = 0.5
    # 如果SPLIT_N、num_warps、num_stages、GROUP_SIZE_ROW都未指定，则根据设备信息和版本选择合适的元数据
    if {SPLIT_N, num_warps, num_stages, GROUP_SIZE_ROW} == {None}:
        device_name = torch.cuda.get_device_name()
        key = (M, K, N, Ms, Ks, beta == 0, beta == 1, alpha == 1)
        # 获取bsr_dense_addmm操作的元数据，根据不同的稀疏度和dtype版本进行选择
        meta = get_meta('bsr_dense_addmm', key,
                        device_name, version=(0, dtype, sparsity))
        if meta is None and sparsity != 0.5:
            meta = get_meta('bsr_dense_addmm', key,
                            device_name, version=(0, dtype, 0.5))
            if meta is None:
                # 寻找一个近似的元数据，使得N % SPLIT_N == 0
                matching_meta = get_meta(
                    'bsr_dense_addmm',
                    (*key[:2], '*', *key[3:]),
                    device_name, version=(0, dtype, 0.5))
                for mkey in sorted(matching_meta or {}):
                    meta_ = matching_meta[mkey]
                    if N % meta_['SPLIT_N'] == 0 and mkey[2] <= N:
                        meta = meta_
        # 如果找到合适的元数据，则更新并返回
        if meta is not None:
            meta.update(**extra)
            return meta
    # 如果 SPLIT_N 为 None 或者 0，将其设为 N 除以 Ms 后的最大整数（至少为 1）
    SPLIT_N = SPLIT_N or max(N // Ms, 1)
    # 如果 GROUP_SIZE_ROW 为 None 或者 0，将其设为 4
    GROUP_SIZE_ROW = GROUP_SIZE_ROW or 4
    # 如果 num_stages 为 None 或者 0，将其设为 1
    num_stages = num_stages or 1
    # 如果 num_warps 为 None 或者 0，将其设为 4
    num_warps = num_warps or 4
    # 返回一个包含 SPLIT_N、GROUP_SIZE_ROW、num_stages 和 num_warps 的字典，
    # 并且将 extra 字典中的所有键值对也加入到返回的字典中
    return dict(SPLIT_N=SPLIT_N, GROUP_SIZE_ROW=GROUP_SIZE_ROW, num_stages=num_stages, num_warps=num_warps, **extra)
# 表示一个将张量包装为键的轻量级包装器，以便通过内存引用比较来近似数据相等性的键
class TensorAsKey:
    """A light-weight wrapper of a tensor that enables storing tensors as
    keys with efficient memory reference based comparision as an
    approximation to data equality based keys.
    
    Motivation: the hash value of a torch tensor is tensor instance
    based that does not use data equality and makes the usage of
    tensors as keys less useful. For instance, the result of
    ``len({a.crow_indices(), a.crow_indices()})`` is `2`, although,
    the tensor results from `crow_indices` method call are equal, in
    fact, these share the same data storage.
    On the other hand, for efficient caching of tensors we want to
    avoid calling torch.equal that compares tensors item-wise.

    TensorAsKey offers a compromise in that it guarantees key equality
    of tensors that references data in the same storage in the same
    manner and without accessing underlying data. However, this
    approach does not always guarantee correctness. For instance, for
    a complex tensor ``x``, we have ``TensorAsKey(x) ==
    TensorAsKey(x.conj())`` while ``torch.equal(x, x.conj())`` would
    return False.
    """

    def __init__(self, obj):
        # 定义获取张量键的函数，不追踪输入对象的负值或共轭位的警告，因为在压缩/普通索引的压缩稀疏张量的用例中（这些张量始终是具有非负项的整数张量），这些位从不设置。但是，当将TensorAsKey的使用扩展到浮点数或复数张量时，这些位（参见is_neg和is_conj方法）必须包含在键中。
        assert not (obj.dtype.is_floating_point or obj.dtype.is_complex), obj.dtype
        # 返回包含张量关键信息的元组
        return (obj.data_ptr(), obj.storage_offset(), obj.shape, obj.stride(), obj.dtype)

        self._obj_ref = weakref.ref(obj)
        if obj.layout is torch.strided:
            # 对于分步布局的张量，直接使用获取的张量键
            self.key = get_tensor_key(obj)
        elif obj.layout in {torch.sparse_csr, torch.sparse_bsr}:
            # 对于CSR或BSR稀疏布局的张量，使用行和列索引的张量键
            self.key = (get_tensor_key(obj.crow_indices()), get_tensor_key(obj.col_indices()))
        elif obj.layout in {torch.sparse_csc, torch.sparse_bsc}:
            # 对于CSC或BSC稀疏布局的张量，使用列和行索引的张量键
            self.key = (get_tensor_key(obj.ccol_indices()), get_tensor_key(obj.row_indices()))
        else:
            # 如果布局类型未实现，则引发NotImplementedError
            raise NotImplementedError(obj.layout)
        # 计算对象的哈希值
        self._hash = hash(self.key)

    def __hash__(self):
        # 返回对象的哈希值
        return self._hash

    def __eq__(self, other):
        # 检查两个TensorAsKey对象是否相等
        if not isinstance(other, TensorAsKey):
            return False
        # 如果自身或其他对象为None，则比较两者是否为同一对象
        if self.obj is None or other.obj is None:
            return self is other
        # 比较两个对象的键是否相等
        return self.key == other.key

    @property
    def obj(self):
        """Return object if alive, otherwise None."""
        # 返回对象的弱引用，如果对象不存在则返回None
        return self._obj_ref()
# 使用 lru_cache 装饰器缓存函数调用结果，最大缓存大小由 TORCH_SPARSE_BSR_SCATTER_MM_LRU_CACHE_SIZE 指定
@lru_cache(maxsize=TORCH_SPARSE_BSR_SCATTER_MM_LRU_CACHE_SIZE)
def _bsr_scatter_mm_indices_data(indices_format, M, K, N, Ms, Ks, nbatches, SPLIT_N, compressed_sparse_tensor_as_key):
    # 从 compressed_sparse_tensor_as_key 中获取 BSR 稀疏张量对象 bsr，并确保其不为 None
    bsr = compressed_sparse_tensor_as_key.obj
    assert bsr is not None
    # 提取 BSR 稀疏张量的行索引和列索引
    crow_indices, col_indices = bsr.crow_indices(), bsr.col_indices()
    # 获取张量的设备信息（设备类型）
    device = crow_indices.device
    # 设置索引数据类型为 torch.int32
    indices_dtype = torch.int32

    # 若 indices_format 为 'bsr_strided_mm_compressed'，执行以下操作
    if indices_format == 'bsr_strided_mm_compressed':
        # 计算每个分段的大小 Ns
        Ns = N // SPLIT_N
        # 初始化 q_offsets_lst 作为列表
        q_offsets_lst = []
        # 创建偏移量 b，用于后续计算
        b = torch.arange(SPLIT_N, dtype=indices_dtype, device=device) * Ns
        # 遍历每个 M // Ms 段
        for m in range(M // Ms):
            # 提取当前段的起始和结束行索引 r0, r1
            r0 = crow_indices[m].item()
            r1 = crow_indices[m + 1].item()
            # 若当前段为空，跳过
            if r1 == r0:
                continue
            # 计算当前段的 q_offsets，并添加到 q_offsets_lst 中
            q_offsets_lst.append((col_indices[r0:r1] * (Ks * N)).repeat(SPLIT_N) + b.repeat_interleave(r1 - r0))
        # 将所有 q_offsets 拼接成一个张量 q_offsets
        q_offsets = torch.cat(q_offsets_lst)
        # 计算 crow_indices 的差分 crow_indices_diff
        crow_indices_diff = crow_indices.diff()
        # 找到非零行索引的位置
        non_zero_row_indices = crow_indices_diff.nonzero()
        # 计算 r_offsets
        a = non_zero_row_indices * (Ms * N)
        r_offsets = (a + b).view(-1)
        # 设置 c_indices 为 crow_indices
        c_indices = crow_indices
        # 返回格式化后的结果元组
        return (indices_format, c_indices, r_offsets, q_offsets)

    # 若 indices_format 为 'bsr_strided_mm'，执行以下操作
    elif indices_format == 'bsr_strided_mm':
        # 计算每个分段的大小 Ns
        Ns = N // SPLIT_N
        # 初始化 p_offsets_lst 和 q_offsets_lst 作为列表
        p_offsets_lst = []
        q_offsets_lst = []
        # 创建偏移量 b，用于后续计算
        b = torch.arange(SPLIT_N, dtype=indices_dtype, device=device) * Ns
        # 遍历每个 M // Ms 段
        for m in range(M // Ms):
            # 提取当前段的起始和结束行索引 r0, r1
            r0 = crow_indices[m].item()
            r1 = crow_indices[m + 1].item()
            # 若当前段为空，跳过
            if r1 == r0:
                continue
            # 计算当前段的 p_offsets 和 q_offsets，并添加到相应的列表中
            p_offsets_lst.append(torch.arange(r0, r1, dtype=indices_dtype, device=device).repeat(SPLIT_N))
            q_offsets_lst.append((col_indices[r0:r1] * (Ks * N)).repeat(SPLIT_N) + b.repeat_interleave(r1 - r0))
        # 将所有 q_offsets 拼接成一个张量 q_offsets
        q_offsets = torch.cat(q_offsets_lst)
        # 计算 crow_indices 的差分 crow_indices_diff
        crow_indices_diff = crow_indices.diff()
        # 找到非零行索引的位置
        non_zero_row_indices = crow_indices_diff.nonzero()
        # 计算 r_offsets
        a = non_zero_row_indices * (Ms * N)
        r_offsets = (a + b).view(-1)
        # 计算 c_indices
        c_indices = torch.cat((crow_indices[:1],
                               torch.cumsum(crow_indices_diff[non_zero_row_indices].repeat_interleave(SPLIT_N), 0)))
        # 将 p_offsets_lst 拼接成一个张量 p_offsets
        p_offsets = torch.cat(p_offsets_lst)
        # 返回格式化后的结果元组
        return (indices_format, c_indices, r_offsets, p_offsets, q_offsets)
    elif indices_format == 'scatter_mm':
        Ns = Ms
        c_indices = [0]
        pq_offsets = []
        # todo: eliminate inner for-loops for efficiency
        # 对于每个批次中的每个子矩阵块
        for b in range(nbatches):
            # 对于每个主矩阵块内的小矩阵块
            for m in range(M // Ms):
                r0 = crow_indices[m].item()
                r1 = crow_indices[m + 1].item()
                # 对于每个小矩阵块内的每个行索引
                for n in range(N // Ns):
                    # 更新列索引的偏移
                    c_indices.append(c_indices[-1] + r1 - r0)
                    # 对于每个行索引对应的列索引
                    for t in range(r1 - r0):
                        p = r0 + t
                        # 计算全局稀疏矩阵索引
                        q = (col_indices[p].item() + b * (K // Ks)) * (N // Ns) + n
                        pq_offsets.append([p, q])

        # 返回稀疏矩阵的索引格式、压缩的行索引和偏移量
        return (indices_format,
                torch.tensor(c_indices, dtype=indices_dtype, device=device),
                torch.tensor(pq_offsets, dtype=indices_dtype, device=device))

    else:
        # 抛出错误，因为提供的索引格式不是支持的格式之一
        raise ValueError(f'Invalid {indices_format=}. Expected bsr_strided_mm_compressed|bsr_strided_mm|scatter_mm')
# 确保稀疏张量的 dense 维度为 0
assert bsr.dense_dim() == 0
# 确保稀疏张量的维度为 2，即没有批次维度
assert bsr.ndim == 2

# 获取稀疏张量的行索引
crow_indices = bsr.crow_indices()
# 获取稀疏张量的列索引
col_indices = bsr.col_indices()
# 获取稀疏张量值的块大小
blocksize = bsr.values().shape[-2:]
# 获取稀疏张量的形状 M, K
M, K = bsr.shape
# 获取块大小 Ms, Ks
Ms, Ks = blocksize
# 获取另一个张量 other 的形状 K_, N
K_, N = other.shape[-2:]

# 确保 K_ 等于 K
assert K_ == K
# 计算 other 张量的批次数目
nbatches = other.shape[:-2].numel()

# 根据输入的元数据计算 scatter_mm 的元数据
meta = scatter_mm_meta(M, K, N, Ms, Ks, **meta_input)
# 如果 meta_input 中没有 'allow_tf32'，则根据 bsr 的 dtype 更新 allow_tf32 属性
if 'allow_tf32' not in meta_input:
    meta.update(allow_tf32=bsr.dtype in {torch.float16, torch.bfloat16})
# 获取元数据中的 SPLIT_N
SPLIT_N = meta['SPLIT_N']

# 根据指定的 indices_format 计算 indices_data
indices_data = _bsr_scatter_mm_indices_data(
    indices_format, M, K, N, Ms, Ks, nbatches, SPLIT_N, TensorAsKey(bsr))

# 根据 indices_format 返回不同的结果和元数据
if indices_format == 'bsr_strided_mm_compressed':
    # 更新 meta 表示结果是压缩的
    meta.update(is_compressed=True)
    return indices_data + (meta,)
elif indices_format == 'bsr_strided_mm':
    # 更新 meta 表示结果不是压缩的
    meta.update(is_compressed=False)
    return indices_data + (meta,)
else:
    # 如果 indices_format 不匹配预期，则返回 indices_data
    return indices_data


# 稀疏张量与 strided 张量矩阵乘法
def bsr_scatter_mm(bsr, other, indices_data=None, out=None):
    """BSR @ strided -> strided
    """

    # 确保稀疏张量 bsr 的维度为 2
    assert bsr.ndim == 2
    # 确保另一个张量 other 的维度至少为 2
    assert other.ndim >= 2

    # 获取 bsr 张量的 Ms, Ks, Ns
    Ms, Ks, Ns = bsr.shape[-2], bsr.shape[-1], other.shape[-1]
    # 获取 bsr 张量值的块大小
    blocksize = bsr.values().shape[-2:]

    # 如果没有提供 indices_data，则根据 bsr 和 other 计算 indices_data
    if indices_data is None:
        indices_data = bsr_scatter_mm_indices_data(bsr, other, indices_format='bsr_strided_mm_compressed')

    # 获取 indices_data 中的 indices_format
    indices_format = indices_data[0]

    # 如果没有提供输出张量 out，则创建一个与 other 形状相匹配的空张量，使用 bsr 的 dtype 和 device
    if out is None:
        out = torch.empty((*other.shape[:-2], Ms, Ns), dtype=bsr.dtype, device=bsr.device)
    # 记录输出张量的形状
    out_shape = out.shape
    # 将输出张量转换为 1D 批次张量
    out = as1Dbatch(out)

    # 如果 bsr 的非零值数目为 0，则将输出张量置零
    if bsr._nnz() == 0:
        out.zero_()
    # 如果 indices_format 属于 {'bsr_strided_mm_compressed', 'bsr_strided_mm'}，则执行 scatter_mm 操作
    elif indices_format in {'bsr_strided_mm_compressed', 'bsr_strided_mm'}:
        out.zero_()
        scatter_mm(bsr.values(), other, indices_data, accumulators=out)
    elif indices_format == 'scatter_mm':
        # 计算非零元素的批次数
        nbatches = other.shape[:-2].numel()
        # 创建用于累加结果的张量，其形状为 (nbatches * Ms // blocksize[0] * Ns // blocksize[0], blocksize[0], blocksize[0])
        accumulators = torch.zeros((nbatches * Ms // blocksize[0] * Ns // blocksize[0], blocksize[0], blocksize[0]),
                                   dtype=bsr.dtype, device=bsr.device)
        # 将 other 张量转换为一维批次，并重新排列维度以便进行矩阵乘法散列操作
        others = (as1Dbatch(other)
                  .transpose(-2, -1)
                  .view(nbatches, Ns // blocksize[0], blocksize[0], Ks // blocksize[1], blocksize[1])
                  .movedim((3, 1, 4, 2), (1, 2, 3, 4))  # 相当于 .transpose(-3, -2).transpose(-2, -1).transpose(-4, -3)
                  .flatten(0, 2)
                  )
        # 执行散列矩阵乘法操作，将结果累加到 accumulators 张量中
        scatter_mm(bsr.values(), others, indices_data, accumulators=accumulators)
        # 将累加结果重新整形为所需输出的形状
        out.copy_(accumulators
                  .unflatten(0, (nbatches, Ms // blocksize[0], Ns // blocksize[0]))
                  .movedim((1, 2, 3, 4), (3, 1, 4, 2))  # 相当于 .transpose(-4, -3).transpose(-2, -1).transpose(-3, -2)
                  .reshape(nbatches, Ns, Ms)
                  .transpose(-2, -1))
    else:
        # 如果 indices_format 不支持，则抛出未实现错误
        raise NotImplementedError(indices_format)

    # 返回整形后的输出张量
    return out.view(out_shape)
def bsr_dense_addmm(
        input: torch.Tensor,
        bsr: torch.Tensor,
        dense: torch.Tensor,
        *,
        beta=1,
        alpha=1,
        out: Optional[torch.Tensor] = None,
        skip_checks: bool = False,
        max_grid: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None,
        meta: Optional[dict] = None):
    # 定义函数名
    f_name = 'bsr_dense_addmm'
    # 获取 BSR 格式张量的数值部分
    values = bsr.values()
    # 获取 BSR 格式张量的行索引
    crow_indices = bsr.crow_indices()
    # 获取 BSR 格式张量的列索引
    col_indices = bsr.col_indices()
    # 计算批次的维度数量
    batch_ndim = crow_indices.dim() - 1
    # 获取 BSR 格式张量的形状中的 M 和 K
    M, K = bsr.shape[batch_ndim:batch_ndim + 2]
    # 获取块大小
    blocksize = values.shape[batch_ndim + 1:batch_ndim + 3]
    # 获取 dense 张量的最后一个维度大小 N
    N = dense.shape[-1]

    # todo: implement checks

    # 如果输出张量未提供，则创建一个空的张量作为输出
    if out is None:
        original_batch_dims_broadcasted = broadcast_batch_dims(f_name, bsr, dense)
        out = dense.new_empty(original_batch_dims_broadcasted + (M, N))

    # 如果 BSR 格式张量的非零元素个数为 0，或者 alpha 为 0，或者 N、M、K 中有任何一个为 0，则根据 beta 的值进行处理
    if bsr._nnz() == 0 or alpha == 0 or N == 0 or M == 0 or K == 0:
        if beta == 0:
            # 如果 beta 为 0，则将输出张量置零
            out.zero_()
        else:
            # 否则将输入张量复制到输出张量，并根据 beta 的值进行缩放
            out.copy_(input)
            if beta != 1:
                out.mul_(beta)
        # 返回处理后的输出张量
        return out

    # 如果未提供元数据，则计算稀疏性并生成元数据
    if meta is None:
        sparsity = round(1 - bsr._nnz() * blocksize[0] * blocksize[1] / (M * K), 2)
        meta = bsr_dense_addmm_meta(M, K, N, blocksize[0], blocksize[1], beta, alpha, sparsity=sparsity, dtype=out.dtype)
    # 备份输出张量
    out_backup = out

    # 准备输入张量，调整它们的格式以匹配处理要求
    crow_indices, col_indices, values, input, dense, out = prepare_inputs(bsr, input, dense, out)

    # 获取块的大小 BM 和 BK
    BM, BK = blocksize
    # 获取元数据中的拆分 N
    SPLIT_N = meta.get('SPLIT_N', N // BM)
    # 计算 BN
    BN = N // SPLIT_N

    # 将输出张量进行块大小的调整
    out_untiled = out
    out = tile_to_blocksize(out, (BM, BN))
    dense = tile_to_blocksize(dense, (BK, BN))
    input = tile_to_blocksize(input, (BM, BN))

    # 根据输出张量的数据类型选择适当的点积输出数据类型
    dot_out_dtype = {torch.float16: tl.float32,
                     torch.bfloat16: tl.float32,
                     torch.float32: tl.float64,
                     torch.float64: tl.float64}[out.dtype]

    # 获取 dense 张量的批次数、行块数和列块数
    n_batches = dense.size(0)
    n_block_rows = crow_indices.size(-1) - 1
    n_block_cols = dense.size(-3)

    # 定义全网格和最大网格
    full_grid = (n_batches, n_block_cols, n_block_rows)
    if max_grid is not None:
        grid_blocks = tuple(max_grid[:3][::-1]) + (None,) * (3 - len(max_grid[:3]))
    else:
        grid_blocks = None

    # 定义张量维度映射关系
    tensor_dims_map = {
        values: (0, None, None),
        crow_indices: (0, None, -1),
        col_indices: (0, None, None),
        input: (0, -3, -4),
        dense: (0, -3, None),
        out: (0, -3, -4),
    }

    # 断言 alpha 不为 0
    assert alpha != 0

    # 定义内核函数，执行稀疏张量矩阵乘法的计算
    def kernel(grid, *sliced_tensors):
        _bsr_strided_addmm_kernel[grid](
            *ptr_stride_extractor(*sliced_tensors),
            beta, alpha,
            beta_is_one=beta == 1,
            beta_is_nonzero=beta != 0,
            alpha_is_one=alpha == 1,
            BLOCKSIZE_ROW=BM,
            BLOCKSIZE_INNER=BK,
            BLOCKSIZE_COL=BN,
            allow_tf32=dot_out_dtype == tl.float32,
            acc_dtype=dot_out_dtype,
            **meta)
    # 调用函数 launch_kernel，并传入参数 kernel, tensor_dims_map, full_grid, grid_blocks
    launch_kernel(kernel, tensor_dims_map, full_grid, grid_blocks)

    # 检查 out 和 out_backup 的数据指针是否不同
    if out.data_ptr() != out_backup.data_ptr():
        # 如果 prepare_inputs 函数复制了 out 的内容，将其内容复制回 out_backup:
        out_backup.copy_(out_untiled.view(out_backup.shape))

    # 返回变量 out_backup
    return out_backup
# 如果检测到 Triton 硬件加速库可用
if has_triton():
    # 导入 Triton 库及其语言模块
    import triton
    import triton.language as tl

    # 定义一个 Triton JIT 编译的函数，用于执行稀疏矩阵-稠密矩阵乘法中的采样操作
    @triton.jit
    def _sampled_addmm_kernel(
        alpha,                          # 系数 alpha
        beta,                           # 系数 beta
        IS_BETA_ZERO: tl.constexpr,     # 是否 beta 为零的常量表达式
        BLOCKSIZE_ROW: tl.constexpr,    # 块的行大小的常量表达式
        BLOCKSIZE_COL: tl.constexpr,    # 块的列大小的常量表达式
        k,                              # 稠密矩阵的维度
        TILE_K: tl.constexpr,           # 稀疏矩阵块的大小的常量表达式
        values_ptr,                     # 稀疏矩阵值的指针
        values_batch_stride,            # 稀疏矩阵值的批次步长
        values_nnz_stride,              # 稀疏矩阵值的非零值步长
        values_row_block_stride,        # 稀疏矩阵值的行块步长
        values_col_block_stride,        # 稀疏矩阵值的列块步长
        crow_indices_ptr,               # 稀疏矩阵压缩行指针的指针
        crow_indices_batch_stride,      # 稀疏矩阵压缩行指针的批次步长
        crow_indices_stride,            # 稀疏矩阵压缩行指针的步长
        col_indices_ptr,                # 稀疏矩阵列指针的指针
        col_indices_batch_stride,       # 稀疏矩阵列指针的批次步长
        col_indices_stride,             # 稀疏矩阵列指针的步长
        mat1_ptr,                       # 第一个稠密矩阵的指针
        mat1_batch_stride,              # 第一个稠密矩阵的批次步长
        mat1_tiled_row_stride,          # 第一个稠密矩阵的行块步长
        mat1_tiled_col_stride,          # 第一个稠密矩阵的列块步长
        mat1_row_block_stride,          # 第一个稠密矩阵的行块步长
        mat1_col_block_stride,          # 第一个稠密矩阵的列块步长
        mat2_ptr,                       # 第二个稠密矩阵的指针
        mat2_batch_stride,              # 第二个稠密矩阵的批次步长
        mat2_tiled_row_stride,          # 第二个稠密矩阵的行块步长
        mat2_tiled_col_stride,          # 第二个稠密矩阵的列块步长
        mat2_row_block_stride,          # 第二个稠密矩阵的行块步长
        mat2_col_block_stride,          # 第二个稠密矩阵的列块步长
        acc_dtype: tl.constexpr,        # 累加数据类型的常量表达式
        allow_tf32: tl.constexpr,       # 是否允许 TF32 的常量表达式
    ):
        # Triton JIT 编译的函数内部，实现稀疏矩阵-稠密矩阵乘法的具体计算细节
        pass

    # Triton JIT 编译的函数，用于执行稀疏矩阵-稠密矩阵乘法中的 BSR 格式的行空间操作
    @triton.jit
    def _bsr_strided_dense_rowspace_kernel(
        # values prologue
        values_ptr,                     # 稀疏矩阵值的指针
        values_batch_stride,            # 稀疏矩阵值的批次步长
        values_nnz_stride,              # 稀疏矩阵值的非零值步长
        values_row_block_stride,        # 稀疏矩阵值的行块步长
        values_col_block_stride,        # 稀疏矩阵值的列块步长
        # values epilogue
        # crow_indices prologue
        crow_indices_ptr,               # 稀疏矩阵压缩行指针的指针
        crow_indices_batch_stride,      # 稀疏矩阵压缩行指针的批次步长
        crow_indices_stride,            # 稀疏矩阵压缩行指针的步长
        # crow_indices epilogue
        # col_indices prologue
        col_indices_ptr,                # 稀疏矩阵列指针的指针
        col_indices_batch_stride,       # 稀疏矩阵列指针的批次步长
        col_indices_stride,             # 稀疏矩阵列指针的步长
        # col_indices epilogue
        # dense prologue
        dense_ptr,                      # 稠密矩阵的指针
        dense_batch_stride,             # 稠密矩阵的批次步长
        dense_tiled_row_stride,         # 稠密矩阵的行块步长
        dense_tiled_col_stride,         # 稠密矩阵的列块步长
        dense_row_block_stride,         # 稠密矩阵的行块步长
        dense_col_block_stride,         # 稠密矩阵的列块步长
        # dense epilogue
        # output prologue
        output_ptr,                     # 输出的指针
        output_batch_stride,            # 输出的批次步长
        output_tiled_row_stride,        # 输出的行块步长
        output_tiled_col_stride,        # 输出的列块步长
        output_row_block_stride,        # 输出的行块步长
        output_col_block_stride,        # 输出的列块步长
        # output epilogue
        #
        # gh-113754: Always keep all constexpr arguments at the end of
        # triton kernel arguments list because with triton 2.1 or
        # earlier non-contiguous outputs will corrupt CUDA state due
        # to a triton bug (fixed in openai/triton#2262).
        BLOCKSIZE_ROW: tl.constexpr,    # 块的行大小的常量表达式
        BLOCKSIZE_COL: tl.constexpr,    # 块的列大小的常量表达式
        acc_dtype: tl.constexpr,        # 累加数据类型的常量表达式
        allow_tf32: tl.constexpr,       # 是否允许 TF32 的常量表达式
        GROUP_SIZE_ROW: tl.constexpr,   # 行组大小的常量表达式
    ):
        # Triton JIT 编译的函数内部，实现 BSR 格式的稀疏矩阵-稠密矩阵乘法的具体计算细节
        pass

    # 定义一个函数，用于执行采样的稠密矩阵乘法操作
    def _run_sampled_addmm_kernel(
        alpha,              # 系数 alpha
        beta,               # 系数 beta
        is_beta_zero,       # 是否 beta 为零
        blocksize,          # 块大小
        k,                  # 矩阵的维度
        tile_k,             # 块的大小
        values,             # 稀疏矩阵值
        crow_indices,       # 稀疏矩
    ):
        # 计算输入张量的批次数
        n_batches = values.size(0)
        # 计算稀疏矩阵的块行数
        n_block_rows = crow_indices.size(-1) - 1

        # 创建完整的网格形状
        full_grid = (n_batches, n_block_rows)
        
        # 根据 max_grid 设置网格块的大小，如果 max_grid 未指定则设置为 None
        if max_grid is not None:
            grid_blocks = tuple(max_grid[:2][::-1]) + (None,) * (2 - len(max_grid[:2]))
        else:
            grid_blocks = None
        
        # 定义张量维度映射关系
        tensor_dims_map = {
            values: (0, None),
            crow_indices: (0, -1),
            col_indices: (0, None),
            mat1: (0, -4),
            mat2: (0, None),
        }
        
        # 根据值的数据类型设置累加的数据类型和是否允许 TF32 加速
        if values.dtype in (torch.half, torch.bfloat16):
            acc_dtype = tl.float32
            allow_tf32 = True
        else:
            acc_dtype = tl.float64
            allow_tf32 = False

        # 定义内核函数，启动相应的 CUDA 内核
        def kernel(grid, *sliced_tensors):
            _sampled_addmm_kernel[grid](
                alpha, beta, is_beta_zero,
                *blocksize, k, tile_k,
                *ptr_stride_extractor(*sliced_tensors),
                acc_dtype=acc_dtype,
                allow_tf32=allow_tf32,
                num_stages=1,
                num_warps=4
            )

        # 调用 launch_kernel 函数，执行内核函数
        launch_kernel(kernel, tensor_dims_map, full_grid, grid_blocks)


    def sampled_addmm(
        input: torch.Tensor,
        mat1: torch.Tensor,
        mat2: torch.Tensor,
        *,
        beta=1.0,
        alpha=1.0,
        out: Optional[torch.Tensor] = None,
        skip_checks: bool = False,
        max_grid: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None,
    # 声明函数以执行稀疏矩阵乘法操作
    def sampled_addmm(
        input, mat1, mat2, alpha, beta, max_grid, out=None, *,
        skip_checks=False
    ):
        # 默认使用的函数名
        f_name = "sampled_addmm"
    
        # 检查输入是否符合BSR布局要求
        check_bsr_layout(f_name, input)
        # 对输入进行扩展以匹配BSR布局
        input_broadcasted = broadcast_batch_dims_bsr(f_name, input, mat1, mat2)
    
        # 如果不跳过检查，则执行以下验证
        if not skip_checks:
            # 检查mat1的设备与输入设备是否相同
            check_device(f_name, mat1, input.device)
            # 检查mat2的设备与输入设备是否相同
            check_device(f_name, mat2, input.device)
            # 如果beta不为0且输入数据类型为torch.bool，则发出警告
            if beta != 0.0 and input.dtype is torch.bool:
                check(
                    False,
                    f"{f_name}(): having beta == {beta} not equal to 0.0 with boolean mask is not allowed."
                )
            # 如果输入数据类型不为torch.bool，则验证数据类型
            if input.dtype is not torch.bool:
                check_dtype(f_name, mat1, input.dtype)
                check_dtype(f_name, mat2, input.dtype)
            else:
                # 如果输入数据类型为torch.bool，则验证mat2的数据类型
                check_dtype(f_name, mat1, mat2.dtype)
            # 检查矩阵乘法的兼容形状
            check_mm_compatible_shapes(f_name, mat1, mat2)
            # 如果输出out不为None，则进行以下验证
            if out is not None:
                # 检查输出out是否符合BSR布局要求
                check_bsr_layout(f_name, out)
                # 检查输出out的设备与mat1的设备是否相同
                check_device(f_name, out, mat1.device)
                # 检查输出out的数据类型与输入的数据类型是否相同
                check_dtype(f_name, out, input.dtype)
                # 检查输出out的形状和稀疏结构是否符合预期
                check(
                    out.shape == input_broadcasted.shape
                    and out._nnz() == input._nnz(),
                    f"{f_name}(): Expects `out` to be of shape {input_broadcasted.shape} "
                    f"and with nnz equal to {input_broadcasted._nnz()} "
                    f"but got out.shape = {out.shape} and out.nnz = {out._nnz()}"
                )
    
        # 如果输出out为None，则将其初始化为input_broadcasted的拷贝
        if out is None:
            out = input_broadcasted.to(mat1.dtype, copy=True)
        else:
            # 否则，将input_broadcasted的数据拷贝到输出out中
            out.copy_(input_broadcasted)
    
        # 如果输出out为空或稀疏值为0，则直接返回输出out
        if out.numel() == 0 or out._nnz() == 0:
            return out
    
        # 计算块大小和矩阵的维度
        blocksize = out.values().shape[-2:]
        m = mat1.size(-2)
        n = mat2.size(-1)
        k = mat1.size(-1)
    
        # 如果alpha为0或k为0，则将输出out的值乘以beta并返回结果
        if alpha == 0.0 or k == 0:
            out.values().mul_(beta)
            return out
    
        # 备份输出out的数据
        out_backup = out
    
        # 准备输入数据，使其适应核函数的要求
        crow_indices, col_indices, values, mat1, mat2 = prepare_inputs(out, mat1, mat2)
    
        # 将mat1和mat2调整到指定的块大小
        mat1 = tile_to_blocksize(mat1, (blocksize[0], k))
        mat2 = tile_to_blocksize(mat2, (k, blocksize[1]))
        tile_k = max(*blocksize)
    
        # 运行采样的矩阵乘法核函数
        _run_sampled_addmm_kernel(
            alpha, beta, beta == 0.0,
            blocksize, k, tile_k,
            values, crow_indices, col_indices,
            mat1, mat2,
            max_grid
        )
    
        # 如果out_backup.values和values的nnz x block strides不相同，
        # 则需要复制数据到out_backup.values中以保证一致性
        if out_backup.values().stride()[-3:] != values.stride()[-3:]:
            out_backup.values().copy_(values.reshape(out_backup.values().shape))
        # 返回处理后的输出
        return out_backup
    # 定义稀疏-密集矩阵乘法函数
    def bsr_dense_mm(
        bsr: torch.Tensor,
        dense: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        skip_checks: bool = False,
        max_grid: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None,
        meta: Optional[dict] = None
    ):
        # 函数名
        f_name = "bsr_dense_mm"
        # 获取稀疏矩阵的行数和每行非零元素块的数目
        m, kl = bsr.shape[-2:]

        # 如果不跳过检查，执行以下检查：
        if not skip_checks:
            # 检查稀疏矩阵的布局
            check_bsr_layout(f_name, bsr)
            # 检查稀疏矩阵和密集矩阵的设备是否一致
            check_device(f_name, bsr, dense.device)
            # 检查稀疏矩阵和密集矩阵的数据类型是否一致
            check_dtype(f_name, bsr, dense.dtype)
            # 检查稀疏矩阵和密集矩阵是否可以进行乘法操作
            check_mm_compatible_shapes(f_name, bsr, dense)

            # 获取密集矩阵的列数
            n = dense.size(-1)
            # 获取稀疏矩阵每个非零元素块的行数和列数
            row_block, col_block = bsr.values().shape[-2:]
            # 检查非零元素块的大小
            check_blocksize(f_name, (row_block, col_block))
            # 检查密集矩阵的列数是否可以被16整除
            check(
                not n % 16,
                f"{f_name}(): dense.size(-1) == {n} should be divisible by 16"
            )
        else:
            # 如果跳过检查，获取密集矩阵的行数和列数
            kr, n = dense.shape[-2:]

        # 广播批处理维度
        original_batch_dims_broadcasted = broadcast_batch_dims(f_name, bsr, dense)

        # 如果提供了输出张量 `out`，且未跳过检查，则执行以下检查：
        if out is not None and not skip_checks:
            # 预期的输出张量形状
            expected_out_shape = original_batch_dims_broadcasted + (m, n)
            # 检查输出张量的形状是否正确
            check(
                out.shape == expected_out_shape,
                "bsr_dense_mm(): `out` argument has wrong shape, "
                f"expected {expected_out_shape}, but got {out.shape}.",
            )
            # 检查输出张量是否是连续的行或列主序
            check(
                out.is_contiguous() or out.transpose(-2, -1).is_contiguous(),
                "bsr_dense_mm(): only row-major/col-major `out` arguments are supported, "
                "i.e. (out.is_contiguous() or out.transpose(-2, -1).is_contiguous()) "
                "should be True.",
            )

        # 如果未提供输出张量 `out`，则分配一个与 `dense` 相同形状的空张量
        if out is None:
            out = dense.new_empty(original_batch_dims_broadcasted + (m, n))

        # 如果稀疏矩阵 `bsr` 的非零元素个数为零，直接返回全零的输出张量
        if bsr._nnz() == 0:
            return out.zero_()

        # 当 `beta==0` 时，addmm 函数忽略输入内容，因此可以使用 `out` 作为输入的占位符，
        # 因为它们的形状匹配：
        return bsr_dense_addmm(out, bsr, dense, alpha=1, beta=0, out=out)


    # 定义 BSR 格式稀疏矩阵的 softmax 核函数
    @triton.jit
    def _bsr_softmax_kernel(
        crow_indices_ptr,
        crow_indices_batch_stride,
        crow_indices_stride,
        values_ptr,
        values_batch_stride,
        values_row_block_stride,
        values_nnz_col_block_stride,
        row_block, col_block,
        MAX_ROW_NNZ: tl.constexpr,
        TILE: tl.constexpr
        ):
            # 获取当前处理的批次的程序 ID，按照第2轴（axis=2）确定
            batch_pid = tl.program_id(axis=2)
            # 获取当前处理的行块偏移的程序 ID，按照第1轴（axis=1）确定
            row_block_offset_pid = tl.program_id(axis=1)
            # 获取当前处理的行块的程序 ID，按照第0轴（axis=0）确定
            row_block_pid = tl.program_id(axis=0)

            # 计算当前行的非零元素的索引指针
            crow_indices_offset_ptr = (
                crow_indices_ptr
                + crow_indices_batch_stride * batch_pid
                + crow_indices_stride * row_block_pid
            )
            # 获取当前行和下一行的非零元素偏移量
            nnz_offset = tl.load(crow_indices_offset_ptr)
            nnz_offset_next = tl.load(crow_indices_offset_ptr + crow_indices_stride)

            # 计算当前行的非零元素数量
            row_nnz = nnz_offset_next - nnz_offset
            # 如果当前行的非零元素数量为零，则跳过该行
            if row_nnz == 0:
                return

            # 创建一个范围数组，范围为 [0, TILE)
            row_arange = tl.arange(0, TILE)
            # 创建一个布尔掩码，用于确定哪些列块需要处理
            mask = row_arange < row_nnz * col_block

            # 获取当前行的值指针
            curr_row_values_ptrs = (
                values_ptr
                + values_batch_stride * batch_pid
                + values_row_block_stride * row_block_offset_pid
                + nnz_offset * col_block
            )

            # 查找当前行中的最大值
            row_tile = tl.load(curr_row_values_ptrs + row_arange, mask=mask, other=-float('inf')).to(tl.float32)
            max_row_value = tl.max(row_tile, axis=0)

            # 循环处理 TILE 到 MAX_ROW_NNZ 的范围，每次增加 TILE
            for _ in range(TILE, MAX_ROW_NNZ, TILE):
                row_arange += TILE
                mask = row_arange < row_nnz * col_block
                row_tile = tl.load(curr_row_values_ptrs + row_arange, mask=mask, other=-float('inf')).to(tl.float32)
                curr_max_row_value = tl.max(row_tile, axis=0)
                max_row_value = tl.where(max_row_value > curr_max_row_value, max_row_value, curr_max_row_value)

            # 计算稳定 softmax 的分母
            num = tl.exp(row_tile - max_row_value)
            denom = tl.sum(num, axis=0)
            for _ in range(TILE, MAX_ROW_NNZ, TILE):
                row_arange -= TILE
                mask = row_arange < row_nnz * col_block
                row_tile = tl.load(curr_row_values_ptrs + row_arange, mask=mask, other=-float('inf')).to(tl.float32)
                num = tl.exp(row_tile - max_row_value)
                denom += tl.sum(num, axis=0)

            # 填充输出值
            tl.store(curr_row_values_ptrs + row_arange, (num / denom).to(values_ptr.dtype.element_ty), mask=mask)
            for _ in range(TILE, MAX_ROW_NNZ, TILE):
                row_arange += TILE
                mask = row_arange < row_nnz * col_block
                row_tile = tl.load(curr_row_values_ptrs + row_arange, mask=mask, other=-float('inf')).to(tl.float32)
                num = tl.exp(row_tile - max_row_value)
                tl.store(curr_row_values_ptrs + row_arange, (num / denom).to(values_ptr.dtype.element_ty), mask=mask)
    # 定义函数名
    def bsr_softmax(input, max_row_nnz=None):
        # 函数名
        f_name = "bsr_softmax"

        # 检查输入的 BSR 布局是否正确
        check_bsr_layout(f_name, input)
        # 检查输入的数据类型是否正确
        check_dtype(f_name, input, input.dtype)

        # 如果输入稀疏张量的非零元素数为 0 或者总元素数为 0，则直接返回输入的克隆副本
        if input._nnz() == 0 or input.numel() == 0:
            return input.clone()

        # 获取输入稀疏张量的形状信息
        m, n = input.shape[-2:]
        # 获取输入稀疏张量的非零元素数
        nnz = input._nnz()
        # 获取输入稀疏张量的行块和列块大小
        row_block, col_block = input.values().shape[-2:]

        # 计算行的最大非零元素数（如果未提供，则使用列数 n 的下一个 2 的幂）
        if max_row_nnz is None:
            max_row_nnz = triton.next_power_of_2(n)
        else:
            max_row_nnz = triton.next_power_of_2(max_row_nnz)

        # 获取输入稀疏张量的行索引并扁平化处理
        crow_indices = input.crow_indices().unsqueeze(0).flatten(0, -2)

        # 重塑张量的值，以便简化批次维度操作，并解锁访问任何给定行中所有非零元素的可能性
        # 从 (b1, ..., bn, nnz, row_block, col_block) -> (b1 * ... * bn, row_block, nnz * col_block)
        if input.values().transpose(-3, -2).is_contiguous():
            # 需要克隆以避免 `contiguous` 返回视图
            values = input.values().clone()
        else:
            values = input.values()
        values = values.transpose(-3, -2).contiguous().unsqueeze(0).flatten(0, -4).reshape(-1, row_block, nnz * col_block)

        # 定义全局网格大小和块大小
        full_grid = (values.shape[0], row_block, m // row_block)
        grid_blocks = None

        # 张量维度映射
        tensor_dims_map = {
            # 我们跨越 nnz 个块，而不是 nnz + 1，因此 crow_indices[..., :-1]
            crow_indices[..., :-1]: (0, None, -1),
            values: (0, None, None),
        }

        # 定义内核函数并启动
        def kernel(grid, *sliced_tensors):
            _bsr_softmax_kernel[grid](
                *ptr_stride_extractor(*sliced_tensors),
                row_block, col_block,
                max_row_nnz,
                # Triton 的最大元素数受到 2 ** 17 的限制
                min(2 ** 17, max_row_nnz)
            )

        # 启动内核函数
        launch_kernel(kernel, tensor_dims_map, full_grid, grid_blocks)

        # 重塑值张量为其原始形状
        values = values.reshape(-1, row_block, nnz, col_block).transpose(-3, -2).reshape(*input.values().shape)

        # 返回稀疏压缩张量，保留原始的行索引、列索引和重塑后的值，以及输入的形状和布局
        return torch.sparse_compressed_tensor(
            input.crow_indices().clone(),
            input.col_indices().clone(),
            values,
            size=input.shape,
            layout=input.layout
        )

    # 定义缩放点积注意力函数
    def _scaled_dot_product_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None
    ):
        # 函数名后缀，用于生成特定错误消息
        f_name = "_scaled_dot_product_attention"
        # 检查是否不是因果关系注意力
        check(
            not is_causal,
            f"{f_name}(): is_causal == True is not supported."
        )
        # 检查是否存在注意力掩码
        check(
            attn_mask is not None,
            f"{f_name}(): attn_mask == None is not supported."
        )
        # 断言确保注意力掩码不为None
        assert attn_mask is not None

        # 检查注意力掩码的布局是否为稀疏 BSR 格式
        check(
            attn_mask.layout == torch.sparse_bsr,
            f"{f_name}(): "
            f"attn_mask.layout must be {torch.sparse_bsr}, but got "
            f"attn_mask.layout == {attn_mask.layout}."
        )

        # 检查输入张量 key, query, attn_mask 的设备是否一致
        check_device(f_name, key, query.device)
        check_device(f_name, value, query.device)
        check_device(f_name, attn_mask, query.device)

        # 检查输入张量 key, value, attn_mask 的数据类型是否一致
        check_dtype(f_name, key, query.dtype)
        check_dtype(f_name, value, query.dtype)
        if attn_mask.dtype is not torch.bool:
            check_dtype(f_name, attn_mask, query.dtype)

        # 执行带采样的矩阵乘法操作
        sdpa = sampled_addmm(attn_mask, query, key.transpose(-2, -1), beta=0.0, skip_checks=False)
        
        # 检查是否需要进行缩放操作，并防止除以零
        if scale is None and query.size(-1) == 0 or scale == 0.0:
            check(
                False,
                f"{f_name}(): current value of scale == {scale} "
                "results in division by zero."
            )
        
        # 计算缩放因子，如果未指定缩放因子则根据查询向量的维度计算
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        
        # 对注意力分数进行缩放操作
        sdpa.values().mul_(scale_factor)
        
        # 对缩放后的注意力分数进行稀疏softmax操作
        sdpa = bsr_softmax(sdpa)
        
        # 对注意力分数应用dropout操作
        torch.nn.functional.dropout(sdpa.values(), p=dropout_p, inplace=True)
        
        # 执行稀疏矩阵乘法，计算最终的注意力加权值
        sdpa = bsr_dense_mm(sdpa, value)
        
        # 返回计算得到的注意力加权值
        return sdpa
    # 定义一个内核函数，用于执行稀疏矩阵乘法的计算。
    def _scatter_mm2_kernel(
            M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,  # 接收输入的尺寸参数 M, K, N
            blocks_ptr, blocks_stride_P, blocks_stride_M, blocks_stride_K,  # 指向块数据的指针及步长参数
            others_ptr, others_stride_Q, others_stride_K, others_stride_N,  # 指向其他数据的指针及步长参数
            accumulators_ptr, accumulators_stride_R, accumulators_stride_M, accumulators_stride_N,  # 指向累加器的指针及步长参数
            pq_offsets_ptr, pq_offsets_stride,  # 指向偏移量的指针及步长参数
            pq_ptr, pq_stride_T, pq_stride_1,  # 指向 pq 数据的指针及步长参数
            dot_out_dtype: tl.constexpr,  # 指定输出数据类型
            TILE_M: tl.constexpr,  # 定义 M 方向上的瓦片尺寸
            TILE_N: tl.constexpr,  # 定义 N 方向上的瓦片尺寸
            allow_tf32: tl.constexpr):  # 是否允许使用 TF32 精度
    
        # 计算瓦片数量 Ms 和 Ns
        Ms = M // TILE_M
        Ns = N // TILE_N
    
        # 获取当前线程的程序 ID
        pid_t = tl.program_id(axis=0)
    
        # 获取当前线程内部的子程序 ID
        pid = tl.program_id(axis=1)
        pid_m = pid // Ms  # 计算子程序在 M 方向上的索引
        pid_n = pid % Ms   # 计算子程序在 N 方向上的索引
    
        # 计算当前子程序处理的块索引 rm, rn, rk
        rm = (pid_m * TILE_M + tl.arange(0, TILE_M))
        rn = (pid_n * TILE_N + tl.arange(0, TILE_N))
        rk = tl.arange(0, K)
    
        # 计算 A 和 B 的指针位置
        A_ptr = blocks_ptr + (rm[:, None] * blocks_stride_M + rk[None, :] * blocks_stride_K)
        B_ptr = others_ptr + (rk[:, None] * others_stride_K + rn[None, :] * others_stride_N)
    
        # 加载并计算偏移量 g0 和 g1
        g0 = tl.load(pq_offsets_ptr + pid_t * pq_offsets_stride)
        g1 = tl.load(pq_offsets_ptr + (pid_t + 1) * pq_offsets_stride)
    
        # 如果 g0 等于 g1，则返回，表示没有需要处理的数据
        if g0 == g1:
            return
    
        # 初始化累加器块 acc_block，用于存储计算结果
        acc_block = tl.zeros((TILE_M, TILE_N), dtype=dot_out_dtype)
    
        # 循环处理每个非零元素对 (p, q)
        for i in range(g0, g1):
            p = tl.load(pq_ptr + i * pq_stride_T)
            q = tl.load(pq_ptr + i * pq_stride_T + pq_stride_1)
            A = tl.load(A_ptr + p * blocks_stride_P)
            B = tl.load(B_ptr + q * others_stride_Q)
            acc_block += tl.dot(A, B, out_dtype=dot_out_dtype, allow_tf32=allow_tf32)
    
        # 计算并存储结果 C_ptr 到累加器中
        C_ptr = accumulators_ptr + pid_t * accumulators_stride_R + (
            rm[:, None] * accumulators_stride_M + rn[None, :] * accumulators_stride_N)
        tl.store(C_ptr, acc_block.to(accumulators_ptr.dtype.element_ty))
    
    # 定义稀疏矩阵乘法函数 _scatter_mm2，接收多个参数
    def _scatter_mm2(
            blocks: torch.Tensor,  # 包含块数据的张量
            others: torch.Tensor,  # 包含其他数据的张量
            pq_offsets: torch.Tensor,  # 包含偏移量的张量
            pq_indices: torch.Tensor,  # 包含索引数据的张量
            accumulators: torch.Tensor  # 包含累加器数据的张量
    ):
        # 获取输入张量 blocks 的形状 P, M, K
        P, M, K = blocks.shape
        # 获取输入张量 others 的形状 Q, _, N
        Q, _, N = others.shape
        # 获取输入张量 accumulators 的形状 R, _, _
        R, _, _ = accumulators.shape

        # 定义元数据字典，包括 TILE_M 和 TILE_N 的计算，以及一些阶段和线程块的数量
        meta = dict(TILE_M=max(16, M // 4), TILE_N=max(16, N // 4), num_stages=1, num_warps=2)

        # 定义内部函数 grid，用于计算并返回 grid 的大小
        def grid(META):
            return (pq_offsets.shape[0] - 1, triton.cdiv(M, META['TILE_M']) * triton.cdiv(N, META['TILE_N']), 1)

        # 根据 accumulators 的 dtype，设置 dot_out_dtype，用于后续的数据类型转换
        dot_out_dtype = {torch.float16: tl.float32,
                         torch.bfloat16: tl.float32,
                         torch.float32: tl.float64,
                         torch.float64: tl.float64}[accumulators.dtype]

        # 如果 meta 中没有 'allow_tf32' 键，则根据 dot_out_dtype 设置 'allow_tf32'
        if 'allow_tf32' not in meta:
            meta.update(allow_tf32=dot_out_dtype == tl.float32)

        # 调用 triton.jit 编译的 _scatter_mm2_kernel 函数，传入各种参数进行矩阵乘法计算
        _scatter_mm2_kernel[grid](
            M, K, N,
            blocks, blocks.stride(0), blocks.stride(1), blocks.stride(2),
            others, others.stride(0), others.stride(1), others.stride(2),
            accumulators, accumulators.stride(0), accumulators.stride(1), accumulators.stride(2),
            pq_offsets, pq_offsets.stride(0),
            pq_indices, pq_indices.stride(0), pq_indices.stride(1),
            dot_out_dtype=dot_out_dtype,
            **meta
        )

    @triton.jit
    # 定义 triton.jit 装饰的 _scatter_mm6 函数，用于执行稀疏矩阵乘法
    def _scatter_mm6(
            blocks: torch.Tensor,        # 输入张量 blocks
            others: torch.Tensor,        # 输入张量 others
            c_indices: torch.Tensor,     # 索引张量 c_indices
            r_offsets: torch.Tensor,     # 索引张量 r_offsets
            p_offsets: torch.Tensor,     # 索引张量 p_offsets
            q_offsets: torch.Tensor,     # 索引张量 q_offsets
            meta: dict,                  # 元数据字典
            accumulators: torch.Tensor,  # 累加器张量
            force_contiguous: bool = True,  # 是否强制连续性
    ):
        # 获取元数据中的 SPLIT_N 值
        SPLIT_N = meta['SPLIT_N']
        # 获取 blocks 张量的维度信息 P, Ms, Ks
        P, Ms, Ks = blocks.shape
        # 获取 others 张量的维度信息 B, K_, N
        B, K_, N = others.shape
        # 获取 accumulators 张量的维度信息 B_, M, N_
        B_, M, N_ = accumulators.shape
        # 断言 N_ 等于 N，确保维度匹配
        assert N_ == N
        # 计算 Ns，作为 N 除以 SPLIT_N 的结果
        Ns = N // SPLIT_N
        # 断言 B_ 等于 B，确保维度匹配

        assert B_ == B

        # 定义一个内部函数 grid，计算输出的形状
        def grid(META):
            return (r_offsets.shape[0] * B, triton.cdiv(Ms, META['TILE_M']) * triton.cdiv(Ns, META['TILE_N']))

        # 根据 accumulators 的 dtype 确定 dot_out_dtype
        dot_out_dtype = {torch.float16: tl.float32,
                         torch.bfloat16: tl.float32,
                         torch.float32: tl.float64,
                         torch.float64: tl.float64}[accumulators.dtype]
        # 如果 meta 中不存在 'allow_tf32' 键，根据 dot_out_dtype 更新 meta
        if 'allow_tf32' not in meta:
            meta.update(allow_tf32=dot_out_dtype == tl.float32)

        # 断言 c_indices 的 stride(0) 等于 1
        assert c_indices.stride(0) == 1
        # 断言 r_offsets 的 stride(0) 等于 1
        assert r_offsets.stride(0) == 1
        # 断言 p_offsets 的 stride(0) 等于 1
        assert p_offsets.stride(0) == 1
        # 断言 q_offsets 的 stride(0) 等于 1
        assert q_offsets.stride(0) == 1

        # 如果 force_contiguous 为真，则将 blocks 和 others 张量转为连续的张量
        if force_contiguous:
            blocks = blocks.contiguous()
            others = others.contiguous()
            # 如果 accumulators 不是连续的张量，则将其转为连续的张量
            if not accumulators.is_contiguous():
                accumulators_ = accumulators.contiguous()
            else:
                accumulators_ = accumulators
        else:
            # 如果 force_contiguous 为假，则直接使用 accumulators
            accumulators_ = accumulators

        # 调用 _scatter_mm6_kernel 函数进行计算
        _scatter_mm6_kernel[grid](
            B, Ms, Ks, N,
            blocks, blocks.stride(0), blocks.stride(1), blocks.stride(2),
            others, others.stride(0), others.stride(1), others.stride(2),
            accumulators_, accumulators_.stride(0), accumulators_.stride(1), accumulators_.stride(2),
            c_indices,
            r_offsets,
            p_offsets,
            q_offsets,
            dot_out_dtype=dot_out_dtype,
            **meta
        )

        # 如果 force_contiguous 为真且 accumulators 不是连续的张量，则将结果复制回 accumulators
        if force_contiguous and not accumulators.is_contiguous():
            accumulators.copy_(accumulators_)

    @triton.jit
    def _bsr_strided_addmm_kernel(
        # values prologue
        values_ptr,                      # 指向稀疏值的指针
        values_batch_stride,             # 批次之间稀疏值的步长
        values_nnz_stride,               # 非零值之间的步长
        values_row_block_stride,         # 行块之间稀疏值的步长
        values_col_block_stride,         # 列块之间稀疏值的步长
        # values epilogue

        # crow_indices prologue
        crow_indices_ptr,                # 指向行压缩指数的指针
        crow_indices_batch_stride,       # 批次之间行压缩指数的步长
        crow_indices_stride,             # 行压缩指数之间的步长
        # crow_indices epilogue

        # col_indices prologue
        col_indices_ptr,                 # 指向列索引的指针
        col_indices_batch_stride,        # 批次之间列索引的步长
        col_indices_stride,              # 列索引之间的步长
        # col_indices epilogue

        # input prologue
        input_ptr,                       # 指向输入数据的指针
        input_batch_stride,              # 批次之间输入数据的步长
        input_tiled_row_stride,          # 输入瓦片行之间的步长
        input_tiled_col_stride,          # 输入瓦片列之间的步长
        input_row_block_stride,          # 行块之间输入数据的步长
        input_col_block_stride,          # 列块之间输入数据的步长
        # input epilogue

        # dense prologue
        dense_ptr,                       # 指向稠密数据的指针
        dense_batch_stride,              # 批次之间稠密数据的步长
        dense_tiled_row_stride,          # 稠密数据瓦片行之间的步长
        dense_tiled_col_stride,          # 稠密数据瓦片列之间的步长
        dense_row_block_stride,          # 行块之间稠密数据的步长
        dense_col_block_stride,          # 列块之间稠密数据的步长
        # dense epilogue

        # output prologue
        output_ptr,                      # 指向输出数据的指针
        output_batch_stride,             # 批次之间输出数据的步长
        output_tiled_row_stride,         # 输出数据瓦片行之间的步长
        output_tiled_col_stride,         # 输出数据瓦片列之间的步长
        output_row_block_stride,         # 行块之间输出数据的步长
        output_col_block_stride,         # 列块之间输出数据的步长
        # output epilogue

        beta,                            # 缩放因子 beta
        alpha,                           # 缩放因子 alpha
        beta_is_one: tl.constexpr,       # 是否 beta 等于 1 的编译时常量
        beta_is_nonzero: tl.constexpr,   # 是否 beta 非零的编译时常量
        alpha_is_one: tl.constexpr,      # 是否 alpha 等于 1 的编译时常量
        BLOCKSIZE_ROW: tl.constexpr,     # 行块大小的编译时常量
        BLOCKSIZE_COL: tl.constexpr,     # 列块大小的编译时常量
        BLOCKSIZE_INNER: tl.constexpr,   # 内部块大小的编译时常量
        acc_dtype: tl.constexpr,         # 累加结果数据类型的编译时常量
        allow_tf32: tl.constexpr,        # 是否允许 TF32 的编译时常量
        GROUP_SIZE_ROW: tl.constexpr,    # 行组大小的编译时常量
        SPLIT_N: tl.constexpr            # 分割 N 的编译时常量
else:
    # 如果条件不满足，以下变量将被赋值为None，并标注为类型忽略赋值错误
    bsr_softmax = None  # type: ignore[assignment]
    bsr_dense_mm = None  # type: ignore[assignment]
    sampled_addmm = None  # type: ignore[assignment]
    _scaled_dot_product_attention = None  # type: ignore[assignment]
    _scatter_mm2 = None  # type: ignore[assignment]
    _scatter_mm6 = None  # type: ignore[assignment]
    _bsr_strided_addmm_kernel = None  # type: ignore[assignment]
```