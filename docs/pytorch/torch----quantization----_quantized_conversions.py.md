# `.\pytorch\torch\quantization\_quantized_conversions.py`

```py
# mypy: allow-untyped-defs
import torch  # 导入PyTorch库


# Pack pairs of int4 values into int8, in row major order; first int4
# value goes into lower order bits, and second int4 value into higher
# order bits of resulting int8 value.
def pack_int4_to_int8(weight):
    assert weight.dim() == 2  # 断言输入张量维度为2
    assert weight.shape[1] % 2 == 0  # 断言输入张量第二维度为偶数
    assert weight.dtype == torch.int8  # 断言输入张量数据类型为int8
    return ((weight[:, 1::2] & 0xF) << 4) | (weight[:, 0::2] & 0xF)  # 打包int4值为int8


# Unpack quandruples of bits in int8 values into int4 values, in row
# major order; lower 4 bits go into first int4 value goes, and upper 4
# bits go into second int4 value.
def unpack_int8_to_int4(weight):
    assert weight.dim() == 2  # 断言输入张量维度为2
    assert weight.dtype == torch.int8  # 断言输入张量数据类型为int8
    return torch.stack((weight & 0xF, (weight >> 4) & 0xF), dim=2).view(
        weight.shape[0], 2 * weight.shape[1]
    )  # 解包int8值为int4


# Transpose the weight matrix, and then reorder its elements according
# to underlying requirements of CUTLASS library, so that it could be
# used for CUTLASS-based mixed datatypes linear operation.
def quantized_weight_reorder_for_mixed_dtypes_linear_cutlass(
    weight, dtypeq, transpose=False
):
    assert weight.dim() == 2  # 断言输入张量维度为2
    assert weight.dtype == torch.int8  # 断言输入张量数据类型为int8
    assert dtypeq == torch.int8 or dtypeq == torch.quint4x2  # 断言量化数据类型为int8或quint4x2
    assert weight.device.type == "cuda"  # 断言张量位于CUDA设备上

    device = weight.device  # 获取张量所在设备

    # subbyte_transpose
    if not transpose:
        if dtypeq == torch.int8:
            outp = weight.T  # 如果量化数据类型为int8，则转置输入张量
        elif dtypeq == torch.quint4x2:
            outp = pack_int4_to_int8(unpack_int8_to_int4(weight.view(torch.int8)).T)  # 如果量化数据类型为quint4x2，则解包和打包int4到int8，并转置
    else:
        outp = weight  # 否则输出为未变换的输入张量

    ncols, nrows = outp.shape  # 获取输出张量的列数和行数

    assert nrows % (32 if dtypeq == torch.quint4x2 else 64) == 0  # 断言行数符合量化数据类型的要求
    assert ncols % 64 == 0  # 断言列数为64的倍数

    # permute_B_rows_for_mixed_gemm
    # (permute cols actually, as transpose is applied first here)
    if dtypeq == torch.quint4x2:
        cols_permuted = (
            torch.tensor(
                [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15],
                device=device,
            )
            + (torch.arange(0, nrows // 16, device=device).reshape(-1, 1) * 16).expand(
                nrows // 16, 16
            )
        ).view(-1)  # 按照指定顺序排列列的索引
    else:
        cols_permuted = (
            torch.tensor(
                [0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15],
                device=device,
            )
            + (torch.arange(0, nrows // 16, device=device).reshape(-1, 1) * 16).expand(
                nrows // 16, 16
            )
        ).view(-1)  # 按照指定顺序排列列的索引
    outp = outp.index_copy(1, cols_permuted, outp)  # 使用排列后的索引重排输出张量的列

    # interleave_column_major_tensor
    magic0 = 4 if dtypeq == torch.quint4x2 else 2  # 计算magic0的值
    magic1 = 32 // magic0  # 计算magic1的值

    tmp0 = (
        (torch.arange(0, ncols // magic0, device=device) * (nrows // 4 * magic0))
        .view(-1, 1)
        .repeat(1, nrows // 4 * magic0)
        .view(-1)
    )  # 计算临时变量tmp0
    # 计算 tmp1：创建一个一维张量，表示输出张量的偏移量
    tmp1 = (
        (torch.arange(0, nrows // 4 // magic1, device=device) * (magic0 * magic1))
        .view(-1, 1)  # 将一维张量转换为列向量
        .repeat(1, magic1)  # 在列方向上重复 magic1 次
        .view(-1)  # 将张量展平为一维
        .repeat(ncols)  # 在一维方向上重复 ncols 次
    )

    # 计算 tmp2：创建一个一维张量，表示输出张量的偏移量
    tmp2 = (
        (torch.arange(0, magic0, device=device) * magic1)
        .view(-1, 1)  # 将一维张量转换为列向量
        .repeat(1, nrows // 4)  # 在列方向上重复 nrows // 4 次
        .view(-1)  # 将张量展平为一维
        .repeat(ncols // magic0)  # 在一维方向上重复 ncols // magic0 次
    )

    # 计算 tmp3：创建一个一维张量，表示输出张量的偏移量
    tmp3 = torch.arange(0, magic1, device=device).repeat(nrows // 4 * ncols // magic1)

    # 计算输出张量的偏移量
    outp_offsets = tmp0 + tmp1 + tmp2 + tmp3

    # 将输出张量展平为一维，类型转换为 torch.int32
    tmp = outp.view(-1).view(torch.int32)
    
    # 初始化输出张量为与 tmp 张量相同大小的零张量
    outp = torch.zeros_like(tmp)
    
    # 根据 dtypeq 的类型进行量化和偏置添加
    outp.scatter_(0, outp_offsets, tmp)
    outp = outp.view(weight.dtype)  # 将输出张量重新视图为 weight 的数据类型

    # 根据 dtypeq 的类型进行偏置和交错量化张量的处理
    tmp = outp.view(-1)

    # 初始化 outp 为与 tmp 张量相同大小的空张量
    outp = torch.empty_like(tmp)
    if dtypeq == torch.int8:
        # 对 tmp 张量进行量化转换为 torch.int，并添加偏置
        tmp = (tmp.to(torch.int) + 128).to(tmp.dtype)
        # 交错插入量化张量的值到 outp 张量中
        outp[0::4] = tmp[0::4]
        outp[1::4] = tmp[2::4]
        outp[2::4] = tmp[1::4]
        outp[3::4] = tmp[3::4]
    elif dtypeq == torch.quint4x2:
        # 对 tmp 张量进行 4 位二进制量化转换和添加偏置
        tmp0 = ((tmp & 0xF) + 8) & 0xF
        tmp0 = (tmp0[1::2] << 4) | tmp0[0::2]
        tmp1 = (((tmp >> 4) & 0xF) + 8) & 0xF
        tmp1 = (tmp1[1::2] << 4) | tmp1[0::2]
        # 交错插入量化张量的值到 outp 张量中
        outp[0::4] = tmp0[0::2]
        outp[1::4] = tmp0[1::2]
        outp[2::4] = tmp1[0::2]
        outp[3::4] = tmp1[1::2]

    # 如果 dtypeq 是 torch.quint4x2 类型，则调整 nrows 和 ncols 的值
    if dtypeq == torch.quint4x2:
        nrows *= 2
        ncols //= 2

    # 将 outp 张量重新视图为 nrows x ncols 的 uint8 类型张量，并返回
    return outp.view(nrows, ncols).view(torch.uint8)
```