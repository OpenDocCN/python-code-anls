# `.\pytorch\torch\sparse\_semi_structured_conversions.py`

```
# mypy: allow-untyped-defs
import torch

# 计算元数据重排序的散列偏移量
def _calculate_meta_reordering_scatter_offsets(m, meta_ncols, meta_dtype, device):
    """
    This is PyTorch implementation of main part of reorder_meta()
    function, from tools/util/include/cutlass/util/host_reorder.h file
    of CUTLASS source tree.  Furthermore, CUTLASS template for sparse
    GEMM decides upon layout of this matrix, and at the moment for the
    sparse GEMM executed on tensor cores, this is layout described by
    ColumnMajorInterleaved<2> data structure, in
    include/cutlass/layout/matrix.h of CUTLASS source tree.  The
    reordering of meta matrix into meta_reordered matrix calculated
    according to these segments of CUTLASS code is re-implemented here.
    Note that this calculation produces offsets for scattering metadata
    matrix elements into reordered metadata matrix elements (or,
    equivalently, for gathering reordered metadata matrix element back
    into metadata matrix elements).
    """
    # 创建一个包含从 0 到 m-1 的整数的列向量，设备上执行操作
    dst_rows = torch.arange(0, m, device=device)[:, None].repeat(1, meta_ncols)
    # 创建一个包含从 0 到 meta_ncols-1 的整数的行向量，设备上执行操作
    dst_cols = torch.arange(0, meta_ncols, device=device).repeat(m, 1)

    # 重新排序行，然后交换 2x2 块
    group = 32 if meta_dtype.itemsize == 2 else 16
    interweave = 4 if meta_dtype.itemsize == 2 else 2
    dst_rows = (
        dst_rows // group * group
        + (dst_rows % 8) * interweave
        + (dst_rows % group) // 8
    )

    # 计算元素的偏移量，以便将其散布到重新排序后的元数据矩阵中
    topright = ((dst_rows % 2 == 0) & (dst_cols % 2 == 1)).to(torch.int8)
    bottomleft = ((dst_rows % 2 == 1) & (dst_cols % 2 == 0)).to(torch.int8)
    dst_rows += topright - bottomleft
    dst_cols -= topright - bottomleft

    # 假设元数据张量存储在 CUTLASS InterleavedColumnMajor 布局中，并且
    # 反向工程对应的代码以将值存储到该张量中
    interleave = 2
    cols_maj = dst_cols // interleave
    cols_min = dst_cols % interleave
    return (cols_maj * m * interleave + dst_rows * interleave + cols_min).view(-1)


# 将稠密矩阵转换为稀疏半结构化表示，使用 CUTLASS 后端使用的布局和对应的元数据矩阵
def sparse_semi_structured_from_dense_cutlass(dense):
    """
    This function converts dense matrix into sparse semi-structured
    representation, producing "compressed" matrix, in the layout used by
    CUTLASS backend, and corresponding metadata matrix.
    """
    if dense.dim() != 2:
        raise RuntimeError(
            f"Expected 2-dimensional dense tensor, got {dense.dim()}-dimensional tensor"
        )

    m, k = dense.shape
    device = dense.device

    # 确定元数据的数据类型
    meta_dtype = torch.int8
    if dense.dtype == torch.int8:
        meta_dtype = torch.int32
    elif dense.dtype in [torch.half, torch.bfloat16, torch.float]:
        meta_dtype = torch.int16
    else:
        raise RuntimeError(f"Invalid datatype {dense.dtype} of dense matrix")

    # 计算每个元数据元素的四重位数
    quadbits_per_meta_elem = meta_dtype.itemsize * 8 // 4
    if quadbits_per_meta_elem not in (4, 8):
        raise RuntimeError("Invalid number of elements per meta element calculated")
    # 如果元数据类型为 torch.int32
    if meta_dtype == torch.int32:
        # 检查 dense 矩阵的行数 m 是否能被 16 整除，若不能则引发运行时错误
        if m % 16 != 0:
            raise RuntimeError(
                f"Number of rows of dense matrix {m} must be divisible by 16"
            )
    else:
        # 否则检查 dense 矩阵的行数 m 是否能被 32 整除，若不能则引发运行时错误
        if m % 32 != 0:
            raise RuntimeError(
                f"Number of rows of dense matrix {m} must be divisible by 32"
            )
    
    # 检查 dense 矩阵的列数 k 是否能被 4 * quadbits_per_meta_elem 整除，若不能则引发运行时错误
    if k % (4 * quadbits_per_meta_elem) != 0:
        raise RuntimeError(
            f"Number of columns of dense matrix {k} must be divisible by {4 * quadbits_per_meta_elem}"
        )

    # 如果 dense 矩阵的数据类型不是 torch.float
    if dense.dtype != torch.float:
        # 设置 ksparse 为 4，将 dense 矩阵重塑为 ksparse 列的视图
        ksparse = 4
        dense_4 = dense.view(-1, k // ksparse, ksparse)
        # 将 dense_4 矩阵的非零值与零值解绑定，分别得到 m0, m1, m2, m3
        m0, m1, m2, m3 = (dense_4 != 0).unbind(-1)
    else:
        # 否则设置 ksparse 为 2，将 dense 矩阵重塑为 ksparse 列的视图
        ksparse = 2
        dense_2 = dense.view(-1, k // ksparse, ksparse)
        # 将 dense_2 矩阵的非零值与零值解绑定，分别得到 m0, m2
        m0, m2 = m1, m3 = (dense_2 != 0).unbind(-1)
    
    # 计算 meta 数据的列数，每个 quadbits_per_meta_elem 对应的列数为 k / (ksparse * quadbits_per_meta_elem)
    meta_ncols = k // (ksparse * quadbits_per_meta_elem)

    # 编码四元 True/False 值的含义说明如下：
    #     [True,  True,  False, False] -> 0b0100
    #     [True,  False, True,  False] -> 0b1000
    #     [False, True,  True,  False] -> 0b1001
    #     [True,  False, False, True ] -> 0b1100
    #     [False, True,  False, True ] -> 0b1101
    #     [False, False, True,  True ] -> 0b1110
    # 具体编码细节：
    #     低两位编码的是四元组中第一个 True 值的索引，高两位编码的是其他 True 值的索引
    # 不足两个 True 值的情况下，将 False 值或者某些索引处的值视为 True 进行编码
    # 超过两个 True 值的情况下，将多余的 True 值视为 False 进行编码

    # 根据 m0, m1 计算表达式 expr0, expr1, expr2
    expr0 = m0 & m1
    expr1 = ~m0 & m1
    expr2 = ~m0 & ~m1
    # 计算编码的四个位
    bit0 = expr1
    bit1 = expr2
    bit2 = expr0 | expr2 | m3
    bit3 = expr1 | ~m1
    # 根据不同的表达式和位值计算索引 idxs0 和 idxs1
    idxs0 = bit0 | (bit1.to(torch.int64) << 1)
    idxs1 = bit2 | (bit3.to(torch.int64) << 1)
    # 如果 dense 的数据类型不是 torch.float
    if dense.dtype != torch.float:
        # 从 dense_4 中按照索引 idxs0 和 idxs1 指定的位置收集数据，形成稀疏张量 sparse0 和 sparse1
        sparse0 = dense_4.gather(-1, idxs0.unsqueeze(-1))  # type: ignore[possibly-undefined]
        sparse1 = dense_4.gather(-1, idxs1.unsqueeze(-1))
        # 将 sparse0 和 sparse1 拼接成形状为 (m, k // 2) 的张量 sparse
        sparse = torch.stack((sparse0, sparse1), dim=-1).view(m, k // 2)
    else:
        # 如果 dense 的数据类型是 torch.float
        # 从 dense_2 中按照索引 idxs0 指定的位置收集数据，形成稀疏张量 sparse，且索引除以 2
        sparse = dense_2.gather(-1, idxs0.unsqueeze(-1) // 2).view(m, k // 2)  # type: ignore[possibly-undefined]

    # 根据 idxs0 和 idxs1 创建元信息 meta_4，将两者按位或操作得到元信息
    meta_4 = idxs0 | (idxs1 << 2)
    # 将元信息 meta_4 重塑为形状为 (-1, meta_ncols, quadbits_per_meta_elem) 的张量，并转换为指定类型 meta_dtype
    meta_n = meta_4.view((-1, meta_ncols, quadbits_per_meta_elem)).to(meta_dtype)

    # 根据 quadbits_per_meta_elem 的值不同，构建不同的 meta 张量
    if quadbits_per_meta_elem == 4:
        meta = (
            meta_n[:, :, 0]
            | (meta_n[:, :, 1] << 4)
            | (meta_n[:, :, 2] << 8)
            | (meta_n[:, :, 3] << 12)
        )
    elif quadbits_per_meta_elem == 8:
        meta = (
            meta_n[:, :, 0]
            | (meta_n[:, :, 1] << 4)
            | (meta_n[:, :, 2] << 8)
            | (meta_n[:, :, 3] << 12)
            | (meta_n[:, :, 4] << 16)
            | (meta_n[:, :, 5] << 20)
            | (meta_n[:, :, 6] << 24)
            | (meta_n[:, :, 7] << 28)
        )

    # 创建一个新的空 meta_reordered 张量，形状为 (m * meta_ncols)，用于重新排序元信息
    meta_reordered = meta.new_empty((m * meta_ncols,))  # type: ignore[possibly-undefined]
    # 计算元信息重排序的偏移量
    meta_offsets = _calculate_meta_reordering_scatter_offsets(
        m, meta_ncols, meta_dtype, device
    )
    # 使用 scatter_ 函数按照 meta_offsets 重新排序 meta 张量的元素，结果保存到 meta_reordered 中
    meta_reordered.scatter_(0, meta_offsets, meta.view(-1))

    # 返回稀疏张量 sparse 和重新排序后的 meta_reordered 张量，形状为 (m, meta_ncols)
    return (sparse, meta_reordered.view(m, meta_ncols))
# 将稀疏张量转换为密集张量，使用 CUTLASS 后端的布局
def sparse_semi_structured_to_dense_cutlass(sparse, meta_reordered):
    """
    This function performs reverse of the function above - it
    reconstructs dense matrix from a pair of "compressed" matrix, given
    in the layout used by CUTLASS backend, and accompanying metadata
    matrix.
    """
    # 检查稀疏张量是否为二维张量，如果不是则抛出运行时错误
    if sparse.dim() != 2:
        raise RuntimeError(
            f"Expected 2-dimensional sparse tensor, got {sparse.dim()}-dimensional tensor"
        )

    # 获取稀疏张量的形状 m 行 k 列
    m, k = sparse.shape
    # 获取稀疏张量所在的设备
    device = sparse.device

    # 检查重新排序后的元数据张量是否为二维张量，如果不是则抛出运行时错误
    if meta_reordered.dim() != 2:
        raise RuntimeError(
            f"Expected 2-dimensional meta tensor, got {meta_reordered.dim()}-dimensional tensor"
        )
    # 检查元数据张量所在的设备是否与稀疏张量一致，如果不一致则抛出运行时错误
    if meta_reordered.device != device:
        raise RuntimeError(
            f"Expected meta matrix to be on {device} device, got matrix on {meta_reordered.device} device"
        )

    # 获取元数据张量的数据类型
    meta_dtype = meta_reordered.dtype
    # 检查元数据类型是否为 torch.int16 或 torch.int32，如果不是则抛出运行时错误
    if meta_dtype not in (torch.int16, torch.int32):
        raise RuntimeError(f"Invalid datatype {meta_dtype} of meta matrix")
    
    # 计算每个元数据元素所占的四位数据（quadbits）数量
    quadbits_per_meta_elem = meta_dtype.itemsize * 8 // 4

    # 根据稀疏张量的数据类型确定 ksparse 的值
    if sparse.dtype != torch.float:
        ksparse = 4
    else:
        ksparse = 2

    # 获取重新排序后的元数据张量的形状
    meta_nrows, meta_ncols = meta_reordered.shape
    # 检查元数据张量的行数是否等于稀疏矩阵的行数，如果不等则抛出运行时错误
    if meta_nrows != m:
        raise RuntimeError(
            f"Number of rows of meta matrix {meta_nrows} must be equal to number of columns of spase matrix {m}"
        )
    # 检查元数据张量的列数乘以 ksparse 乘以 quadbits_per_meta_elem 是否等于 2 * k，如果不等则抛出运行时错误
    if meta_ncols * ksparse * quadbits_per_meta_elem != 2 * k:
        raise RuntimeError(
            f"Number of columns of sparse matrix {k} different from the {meta_ncols * ksparse * quadbits_per_meta_elem // 2}, "
            "expected according to the number of columns of meta matrix"
        )

    # 根据重新排序的元数据张量计算元数据的偏移量
    meta_offsets = _calculate_meta_reordering_scatter_offsets(
        m, meta_ncols, meta_dtype, device
    )
    # 使用元数据偏移量重新排列元数据张量
    meta = torch.gather(meta_reordered.view(-1), 0, meta_offsets).view(m, meta_ncols)

    # 创建一个空的元数据张量 meta_2，用于将稀疏张量解压缩为原始的密集张量
    meta_2 = torch.empty(
        (m, meta_ncols, 2 * quadbits_per_meta_elem),
        dtype=meta_dtype,
        device=device,
    )
    # 根据 quadbits_per_meta_elem 的值解码元数据张量 meta，将其填充到 meta_2 中的对应位置
    if quadbits_per_meta_elem == 4:
        meta_2[:, :, 0] = meta & 0b11
        meta_2[:, :, 1] = (meta >> 2) & 0b11
        meta_2[:, :, 2] = (meta >> 4) & 0b11
        meta_2[:, :, 3] = (meta >> 6) & 0b11
        meta_2[:, :, 4] = (meta >> 8) & 0b11
        meta_2[:, :, 5] = (meta >> 10) & 0b11
        meta_2[:, :, 6] = (meta >> 12) & 0b11
        meta_2[:, :, 7] = (meta >> 14) & 0b11
    # 如果每个元数据元素占用8个四位二进制数
    elif quadbits_per_meta_elem == 8:
        # 将 meta 的低位4位与0b11进行按位与操作，存入 meta_2 的第一个通道
        meta_2[:, :, 0] = meta & 0b11
        # 将 meta 右移2位后的低位4位与0b11进行按位与操作，存入 meta_2 的第二个通道
        meta_2[:, :, 1] = (meta >> 2) & 0b11
        # 依此类推，将 meta 右移4、6、8、10、12、14、16、18、20、22、24、26、28、30位后的低位4位与0b11进行按位与操作，分别存入 meta_2 的后续通道
        meta_2[:, :, 2] = (meta >> 4) & 0b11
        meta_2[:, :, 3] = (meta >> 6) & 0b11
        meta_2[:, :, 4] = (meta >> 8) & 0b11
        meta_2[:, :, 5] = (meta >> 10) & 0b11
        meta_2[:, :, 6] = (meta >> 12) & 0b11
        meta_2[:, :, 7] = (meta >> 14) & 0b11
        meta_2[:, :, 8] = (meta >> 16) & 0b11
        meta_2[:, :, 9] = (meta >> 18) & 0b11
        meta_2[:, :, 10] = (meta >> 20) & 0b11
        meta_2[:, :, 11] = (meta >> 22) & 0b11
        meta_2[:, :, 12] = (meta >> 24) & 0b11
        meta_2[:, :, 13] = (meta >> 26) & 0b11
        meta_2[:, :, 14] = (meta >> 28) & 0b11
        meta_2[:, :, 15] = (meta >> 30) & 0b11

    # 计算稠密偏移量，将 meta_2 展平后的数据与 torch.arange 生成的偏移量相加
    dense_offsets = meta_2.view(-1) + (
        torch.arange(0, 2 * m * k // ksparse, device=device) * 4
    ).view(-1, 1).repeat(1, 2).view(-1)

    # 创建全零的稠密张量 dense，类型与 sparse 张量相同
    dense = torch.zeros((m * 2 * k,), dtype=sparse.dtype, device=device)
    
    # 如果 sparse 张量的数据类型不是 torch.float，使用 scatter_ 函数将 sparse 张量的展平版本按照 dense_offsets 分散到 dense 张量中
    if sparse.dtype != torch.float:
        dense.scatter_(0, dense_offsets, sparse.view(-1))
    # 如果 sparse 张量的数据类型是 torch.float，将 dense 张量视为 torch.half 类型后，使用 scatter_ 函数将 sparse 张量视为 torch.half 类型后的展平版本按照 dense_offsets 分散到 dense 张量中
    else:
        dense.view(torch.half).scatter_(
            0, dense_offsets, sparse.view(torch.half).view(-1)
        )

    # 将 dense 张量重新视图为形状 (m, 2 * k) 并返回
    return dense.view(m, 2 * k)
def _sparse_semi_structured_tile(dense):
    """
    This function computes a 2:4 sparse tile by greedily taking the largest values.

    Since we take the largest values greedily, how the sorting algorithm handles duplicates affects
    the ultimate sparsity pattern.

    Note that this function does not have the same sorting semantics as our CUDA backend,
    which is exposed via `torch._sparse_semi_structured_tile` and thus returns a different pattern.
    """

    def greedy_prune_tile(tile):
        num_kept_row = [0, 0, 0, 0]  # 列表，记录每行保留的元素数量
        num_kept_col = [0, 0, 0, 0]  # 列表，记录每列保留的元素数量

        # 对 tile 中展开后的元素按降序排序，并稳定排序（处理相同值的情况）
        for x in tile.flatten().sort(descending=True, stable=True).indices:
            r, c = x // 4, x % 4  # 计算出展开后索引对应的行和列
            if num_kept_row[r] < 2 and num_kept_col[c] < 2:
                num_kept_row[r] += 1  # 如果行 r 和列 c 上还可以保留元素，则加一
                num_kept_col[c] += 1
            else:
                tile[r, c] = 0  # 否则将该位置置为0（稀疏化处理）

    # 将 dense 张量按照步长4展开成大小为4x4的块，并逐个块应用 greedy_prune_tile 函数
    for batch in dense.unfold(0, 4, 4).unfold(1, 4, 4):
        for tile in batch:
            greedy_prune_tile(tile)

    return dense


def _compute_compressed_swizzled_bitmask(dense):
    """
    Calculates the compressed swizzled bitmask from a dense tensor
    """

    # 将 dense 张量转换为布尔型张量，表示为位掩码
    int_bitmask = dense.bool().to(torch.uint8)

    # 每个线程负责一个8x8块，其中包含4个4x4块：A, B, C 和 D
    # 如下所示的模式：
    # +---+---+
    # | A | B |
    # +---+---+
    # | C | D |
    # +---+---+

    # 先将张量切分成8x8大小的块
    bitmask_8x8_chunks = int_bitmask.unfold(0, 8, 8).unfold(1, 8, 8)

    # 再次切分以获得单独的4x4块
    bitmask_4x4_chunks = bitmask_8x8_chunks.unfold(2, 4, 4).unfold(3, 4, 4)

    # 每个4x4位掩码定义了两个8位整数，编码了该块的稀疏模式
    # 最低位优先存储
    # [1 1 0 0]
    # [1 1 0 0]  ->  0011 0011 ->   51
    # [0 0 1 1]      1100 1100      204
    # [0 0 1 1]

    # 将4x4块的二进制表示重塑为8位向量
    bitmask_binary_representation = bitmask_4x4_chunks.reshape(*bitmask_4x4_chunks.shape[:2], 4, 2, 8)

    # 用2的幂进行矩阵乘法以将二进制表示转换为整数
    powers_of_two = 2**torch.arange(8, dtype=torch.float, device="cuda")
    # 在GPU上运行：转换为float执行矩阵乘法，然后再转换回uint8
    compressed_swizzled_bitmask = (bitmask_binary_representation.to(torch.float) @ powers_of_two).to(torch.uint8)

    return compressed_swizzled_bitmask
```