# `.\pytorch\torch\jit\_shape_functions.py`

```py
# mypy: allow-untyped-defs
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

number = Union[int, float]
# flake8: noqa

###
# There are generated files that depend on this file
# To re-generate, please run from the root of the repo:
# python torchgen/shape_functions/gen_jit_shape_functions.py

# How to test:
# After regenerating files, compile PyTorch.
# Then run: ./build/bin/test_jit --gtest_filter=TestShapeGraphLinting.Basic
# If you have enabled opinfo testing for the op, also run:
# python test/test_ops_jit.py TestJitCPU.test_variant_consistency_jit_[FAILING_OP]_cpu_float32
# to reproduce errors from opinfo tests.

# Example PR: https://github.com/pytorch/pytorch/pull/80860/files
####

import torch


def broadcast(a: List[int], b: List[int]):
    # 获取列表 a 和 b 的维度
    dimsA = len(a)
    dimsB = len(b)
    # 计算广播后的维度，取两者中较大的作为广播后的维度
    ndim = max(dimsA, dimsB)
    expandedSizes: List[int] = []

    for i in range(ndim):
        offset = ndim - 1 - i
        dimA = dimsA - 1 - offset
        dimB = dimsB - 1 - offset
        # 获取当前维度下的大小
        sizeA = a[dimA] if (dimA >= 0) else 1
        sizeB = b[dimB] if (dimB >= 0) else 1

        # 如果大小不相等且不为 1，则抛出错误
        if sizeA != sizeB and sizeA != 1 and sizeB != 1:
            # TODO: only assertion error is bound in C++ compilation right now
            raise AssertionError(
                f"The size of tensor a {sizeA} must match the size of tensor b ({sizeB}) at non-singleton dimension {i}"
            )

        # 将广播后的大小添加到列表中
        expandedSizes.append(sizeB if sizeA == 1 else sizeA)

    return expandedSizes


def broadcast_three(a: List[int], b: List[int], c: List[int]):
    # 对三个列表进行广播操作
    return broadcast(broadcast(a, b), c)


def broadcast_one_three(a: List[int], b: Any, c: List[int]):
    # 对第一个和第三个列表进行广播操作
    return broadcast(a, c)


def adaptive_avg_pool2d(self: List[int], out: List[int]):
    # 自适应平均池化的输出形状
    assert len(out) == 2
    assert len(self) == 3 or len(self) == 4
    # 确保 self 中的每个维度都不为零
    for i in range(1, len(self)):
        assert self[i] != 0

    shape: List[int] = []
    # 将 self 中除了最后两个元素外的维度添加到 shape 中
    for i in range(0, len(self) - 2):
        shape.append(self[i])
    # 将 out 中的元素添加到 shape 中
    for elem in out:
        shape.append(elem)
    return shape


def _copy(self: List[int]):
    # 复制列表中的元素到新的列表中
    out: List[int] = []
    for elem in self:
        out.append(elem)
    return out


def unary(self: List[int]):
    # 对列表进行一元操作，即复制列表
    return _copy(self)


def broadcast_inplace(a: List[int], b: List[int]):
    # 就地广播两个列表
    dimsA = len(a)
    dimsB = len(b)
    if dimsB > dimsA:
        raise AssertionError(
            f"The dims of tensor b ({dimsB}) must be less than or equal tothe dims of tensor a ({dimsA}) "
        )
    for dimA in range(dimsA):
        dimB = dimsB - dimsA + dimA
        sizeA = a[dimA]
        sizeB = b[dimB] if (dimB >= 0) else 1
        if sizeA != sizeB and sizeB != 1:
            # TODO: only assertion error is bound in C++ compilation right now
            raise AssertionError(
                "The size of tensor a {} must match the size of tensor b ({}) at non-singleton dimension {}".format(sizeA, sizeB, dimA)
            )
    return _copy(a)
# 将 self 扩展到指定 sizes 的长度，确保 sizes 的长度不小于 self 的长度
def expand(self: List[int], sizes: List[int]):
    # 确定 sizes 的维度数
    ndim = len(sizes)
    # 确定 self 的维度数
    tensor_dim = len(self)
    # 如果 sizes 的维度数为 0，则返回 sizes 的副本
    if ndim == 0:
        return _copy(sizes)
    # 初始化输出列表
    out: List[int] = []
    # 遍历 sizes 的维度
    for i in range(ndim):
        # 计算当前维度的偏移量
        offset = ndim - 1 - i
        # 计算 self 对应的维度索引
        dim = tensor_dim - 1 - offset
        # 获取当前维度的大小，如果维度索引非负则取 self 中对应维度的大小，否则设为 1
        size = self[dim] if dim >= 0 else 1
        # 获取目标维度的大小
        targetSize = sizes[i]
        # 如果目标大小为 -1，则确保对应维度索引非负，并将目标大小设为该维度的大小
        if targetSize == -1:
            assert dim >= 0
            targetSize = size
        # 如果当前大小不等于目标大小，且当前大小为 1，则将当前大小设为目标大小
        if size != targetSize:
            assert size == 1
            size = targetSize
        # 将当前大小添加到输出列表
        out.append(size)
    # 返回输出列表
    return out


# 未使用的函数，直接调用 expand 函数并返回结果
def expand_one_unused(self: List[int], sizes: List[int], inp0: Any):
    return expand(self, sizes)


# 根据给定的 shape 推断大小，确保元素数目为 numel
def infer_size_impl(shape: List[int], numel: int) -> List[int]:
    # 初始化新的大小为 1
    newsize = 1
    # 推断的维度为 None
    infer_dim: Optional[int] = None
    # 遍历 shape 的维度
    for dim in range(len(shape)):
        # 如果当前维度为 -1，确保只有一个推断的维度
        if shape[dim] == -1:
            if infer_dim is not None:
                raise AssertionError("only one dimension can be inferred")
            infer_dim = dim
        # 如果当前维度大于等于 0，则乘以当前维度的大小
        elif shape[dim] >= 0:
            newsize *= shape[dim]
        # 否则，抛出异常，表示维度无效
        else:
            raise AssertionError("invalid shape dimensions")
    # 如果元素数目不等于新的大小，或者有推断的维度但新的大小为 0 或者元素数目不能整除新的大小，则抛出异常，表示形状无效
    if not (
        numel == newsize
        or (infer_dim is not None and newsize > 0 and numel % newsize == 0)
    ):
        raise AssertionError("invalid shape")
    # 复制 shape 并更新推断的维度
    out = _copy(shape)
    if infer_dim is not None:
        out[infer_dim] = numel // newsize
    # 返回更新后的 shape
    return out


# 计算给定大小列表的元素数目
def numel(sizes: List[int]):
    numel = 1
    for elem in sizes:
        numel *= elem
    return numel


# 根据给定的 sizes 视图调整 self 的大小
def view(self: List[int], sizes: List[int]):
    return infer_size_impl(sizes, numel(self))


# 未使用的函数，直接调用 view 函数并返回结果
def view_one_unused(self: List[int], sizes: List[int], *, implicit: bool = False):
    return view(self, sizes)


# 对给定的维度进行求和或求平均操作
def sum_mean_dim(
    self: List[int], opt_dims: Optional[List[int]], keep_dim: bool, dt: Any
):
    # 初始化输出列表
    out: List[int] = []
    # 如果 opt_dims 为空或者长度为 0，则将 dims 设置为 self 的所有维度索引列表
    if opt_dims is None or len(opt_dims) == 0:
        dims: List[int] = list(range(len(self)))
    else:
        dims = opt_dims

    # 遍历 self 的维度
    for idx in range(len(self)):
        # 初始化是否为平均维度为 False
        is_mean_dim: bool = False
        # 遍历需要减少的维度列表 dims
        for reduce_dim in dims:
            # 如果当前维度索引等于需要减少的维度索引的包装值，则标记为平均维度
            if idx == maybe_wrap_dim(reduce_dim, len(self)):
                is_mean_dim = True
        # 如果是平均维度且需要保持维度，则将 1 添加到输出列表
        if is_mean_dim:
            if keep_dim:
                out.append(1)
        # 否则，将当前维度的大小添加到输出列表
        else:
            out.append(self[idx])
    # 返回输出列表
    return out


# 对给定维度进行求最大值操作
def max_dim(self: List[int], dim: int, keep_dim: bool):
    # 调用 sum_mean_dim 函数进行求和操作，并返回结果
    out = sum_mean_dim(self, [dim], keep_dim, None)
    return out, out


# 注意：Python 在整数除法时已向负无穷方向取整，因此不需要特殊的算术处理
# 对 x 和 y 进行整数除法运算并返回结果
def div_rtn(x: int, y: int):
    return x // y


# 计算池化操作的输出形状
def pooling_output_shape_pad_lr(
    inputSize: int,
    kernelSize: int,
    pad_l: int,
    pad_r: int,
    stride: int,
    dilation: int,
    ceil_mode: bool,
):
    # 计算输出大小，根据输入大小、填充左右边界、膨胀率、核大小、步幅等参数进行计算
    outputSize = (
        div_rtn(  # 调用div_rtn函数，计算以下表达式结果
            inputSize  # 输入大小
            + pad_l  # 左填充
            + pad_r  # 右填充
            - dilation * (kernelSize - 1)  # 考虑卷积核膨胀后的大小调整
            - 1  # 考虑索引从0开始的调整
            + (stride - 1 if ceil_mode else 0),  # 如果使用ceil_mode，考虑步幅减一的调整
            stride,  # 步幅
        )
        + 1  # 计算总大小时加一
    )
    # 如果使用ceil_mode，再次检查输出大小是否需要向上取整调整
    if ceil_mode:
        if (outputSize - 1) * stride >= inputSize + pad_l:
            outputSize = outputSize - 1  # 调整输出大小以满足ceil_mode要求
    # 返回最终计算得到的输出大小
    return outputSize
# 计算池化层输出的形状
def pooling_output_shape(
    inputSize: int,
    kernelSize: int,
    pad_l: int,
    stride: int,
    dilation: int,
    ceil_mode: bool,
):
    # 断言步长不为零
    assert stride != 0, "stride should not be zeero"
    # 调用带有填充参数的池化输出形状函数
    return pooling_output_shape_pad_lr(
        inputSize, kernelSize, pad_l, pad_l, stride, dilation, ceil_mode
    )


# 检查池化层的形状
def pool2d_shape_check(
    input: List[int],
    kH: int,
    kW: int,
    dH: int,
    dW: int,
    padH: int,
    padW: int,
    dilationH: int,
    dilationW: int,
    nInputPlane: int,
    inputHeight: int,
    inputWidth: int,
    outputHeight: int,
    outputWidth: int,
):
    # 获取输入维度
    ndim = len(input)
    nOutputPlane = nInputPlane

    # 断言核大小、步长和膨胀率均大于零
    assert kW > 0 and kH > 0
    assert dW > 0 and dH > 0
    assert dilationH > 0 and dilationW > 0

    # 检查输入维度是否符合要求
    valid_dims = input[1] != 0 and input[2] != 0
    assert (
        ndim == 3
        and input[0] != 0
        and valid_dims
        or (ndim == 4 and valid_dims and input[3] != 0)
    )

    # 断言填充大小不超过核大小的一半，输出高度和宽度大于等于1
    assert kW // 2 >= padW and kH // 2 >= padH
    assert outputWidth >= 1 and outputHeight >= 1


# 最大池化操作
def max_pool2d(
    input: List[int],
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    ceil_mode: bool,
):
    # 断言核大小、步长、填充和膨胀率的维度符合要求
    assert (
        len(kernel_size) == 1 or len(kernel_size) == 2
    ), "max_pool2d: kernel_size must either be a single int, or a tuple of two ints"
    kH = kernel_size[0]
    kW = kH if len(kernel_size) == 1 else kernel_size[1]

    assert (
        len(stride) == 0 or len(stride) == 1 or len(stride) == 2
    ), "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints"
    dH = kH if len(stride) == 0 else stride[0]
    if len(stride) == 0:
        dW = kW
    elif len(stride) == 1:
        dW = dH
    else:
        dW = stride[1]

    assert (
        len(padding) == 1 or len(padding) == 2
    ), "max_pool2d: padding must either be a single int, or a tuple of two ints"
    padH = padding[0]
    padW = padH if len(padding) == 1 else padding[1]

    assert (
        len(dilation) == 1 or len(dilation) == 2
    ), "max_pool2d: dilation must be either a single int, or a tuple of two ints"
    dilationH = dilation[0]
    dilationW = dilationH if len(dilation) == 1 else dilation[1]

    # 断言输入维度为3或4
    assert len(input) == 3 or len(input) == 4

    # 获取输入的批次数、输入通道数、输入高度和宽度
    nbatch = input[-4] if len(input) == 4 else 1
    nInputPlane = input[-3]
    inputHeight = input[-2]
    inputWidth = input[-1]

    # 计算输出高度和宽度
    outputHeight = pooling_output_shape(inputHeight, kH, padH, dH, dilationH, ceil_mode)
    outputWidth = pooling_output_shape(inputWidth, kW, padW, dW, dilationW, ceil_mode)

    # 检查池化层的形状是否符合要求
    pool2d_shape_check(
        input,
        kH,
        kW,
        dH,
        dW,
        padH,
        padW,
        dilationH,
        dilationW,
        nInputPlane,
        inputHeight,
        inputWidth,
        outputHeight,
        outputWidth,
    )

    # 如果输入维度为3，返回包含通道数、输出高度和宽度的列表
    if len(input) == 3:
        return [nInputPlane, outputHeight, outputWidth]
    else:
        # 如果条件不满足，则执行以下代码块
        return [nbatch, nInputPlane, outputHeight, outputWidth]
# 定义一个函数 max_pool2d_with_indices，执行最大池化操作并返回结果和原始输出
def max_pool2d_with_indices(
    input: List[int],  # 输入列表
    kernel_size: List[int],  # 池化核大小
    stride: List[int],  # 步幅大小
    padding: List[int],  # 填充大小
    dilation: List[int],  # 膨胀大小
    ceil_mode: bool,  # 是否使用 ceil 模式
):
    out = max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)  # 调用最大池化函数
    return (out, out)  # 返回池化结果和其自身的元组


# 定义一个函数 upsample_nearest2d，对输入进行最近邻插值上采样
def upsample_nearest2d(
    input: List[int],  # 输入列表
    output_size: Optional[List[int]],  # 输出大小（可选）
    scale_factors: Optional[List[float]],  # 缩放因子（可选）
):
    out: List[int] = []  # 初始化输出列表
    out.append(input[0])  # 添加输入的前两个元素到输出中
    out.append(input[1])

    # 如果 scale_factors 和 output_size 都未指定，抛出断言错误
    if scale_factors is None and output_size is None:
        assert 0, "Either output_size or scale_factors must be presented"

    # 如果指定了 output_size
    if output_size is not None:
        assert (
            scale_factors is None
        ), "Must specify exactly one of output_size and scale_factors"
        assert len(output_size) == 2  # 输出大小必须是二维
        out.append(output_size[0])  # 添加 output_size 的两个维度到输出中
        out.append(output_size[1])

    # 如果指定了 scale_factors
    if scale_factors is not None:
        assert (
            output_size is None
        ), "Must specify exactly one of output_size and scale_factors"
        assert len(scale_factors) == 2  # 缩放因子必须是二维
        out.append(int(input[2] * scale_factors[0]))  # 根据缩放因子计算新的维度大小并添加到输出中
        out.append(int(input[3] * scale_factors[1]))

    return out  # 返回最终的输出列表


# 定义一个函数 mm，执行矩阵乘法
def mm(self: List[int], mat2: List[int]):
    assert len(self) == 2, "self must be a matrix"  # 断言确保 self 是一个二维矩阵
    assert len(mat2) == 2, "mat2 must be a matrix"  # 断言确保 mat2 是一个二维矩阵

    assert self[1] == mat2[0]  # 断言确保矩阵乘法的维度匹配
    return [self[0], mat2[1]]  # 返回矩阵乘法的结果


# 定义一个函数 dot，执行张量点积运算
def dot(self: List[int], tensor: List[int]):
    assert len(self) == 1 and len(tensor) == 1  # 断言确保输入是一维张量
    assert self[0] == tensor[0]  # 断言确保张量维度匹配
    out: List[int] = []  # 初始化输出列表
    return out  # 返回空的输出列表


# 定义一个函数 mv，执行矩阵向量乘法
def mv(self: List[int], vec: List[int]):
    assert len(self) == 2 and len(vec) == 1  # 断言确保 self 是一个二维矩阵，vec 是一个一维向量
    assert self[1] == vec[0]  # 断言确保矩阵和向量的维度匹配
    # TODO: return self
    return [self[0]]  # 返回矩阵向量乘法的结果


# 定义一个函数 unsqueeze，在指定维度上对列表进行扩展
def unsqueeze(li: List[int], dim: int):
    dim = maybe_wrap_dim(dim, len(li) + 1)  # 确定扩展维度，并确保其有效性
    out = _copy(li)  # 复制输入列表
    out.insert(dim, 1)  # 在指定维度上插入值为 1
    return out  # 返回扩展后的列表


# 定义一个函数 squeeze_nodim，在不指定维度的情况下对列表进行压缩
def squeeze_nodim(li: List[int]):
    out: List[int] = []  # 初始化输出列表
    for i in range(len(li)):
        if li[i] != 1:  # 如果元素不为 1，则添加到输出列表中
            out.append(li[i])
    return out  # 返回压缩后的列表


# 定义一个函数 squeeze，在指定维度上对列表进行压缩
def squeeze(li: List[int], dim: int):
    out: List[int] = []  # 初始化输出列表
    wrapped_dim = maybe_wrap_dim(dim, len(li))  # 确定压缩维度，并确保其有效性
    for i in range(len(li)):
        if i == wrapped_dim:
            if li[i] != 1:  # 如果指定维度上元素不为 1，则添加到输出列表中
                out.append(li[i])
        else:
            out.append(li[i])  # 其他维度上的元素直接添加到输出列表中
    return out  # 返回压缩后的列表


# 定义一个函数 squeeze_dims，在指定维度列表上对列表进行压缩
def squeeze_dims(li: List[int], dims: List[int]):
    if len(dims) == 0:  # 如果维度列表为空，则直接返回原列表
        return li
    wrapped_dims = _copy(dims)  # 复制维度列表
    for i in range(len(dims)):
        wrapped_dims[i] = maybe_wrap_dim(wrapped_dims[i], len(li))  # 确定每个维度的有效性
    result: List[int] = []  # 初始化结果列表
    for i in range(len(li)):
        if li[i] == 1:  # 如果元素为 1
            if i not in wrapped_dims:  # 并且不在压缩维度列表中，则忽略
                result.append(li[i])
        else:
            result.append(li[i])  # 非 1 元素直接添加到结果列表中
    return result  # 返回压缩后的列表


# 定义一个函数 index_select，在指定维度上对列表进行索引选择
def index_select(self: List[int], dim: int, index: List[int]):
    dim = maybe_wrap_dim(dim, len(self))  # 确定索引选择的维度，并确保其有效性
    numel = multiply_integers(index)  # 计算索引列表中元素的乘积
    assert len(index) <= 1  # 断言确保索引列表长度最多为 1
    assert dim == 0 or dim < len(self)  # 断言确保选择的维度有效
    result_size: List[int] = []  # 初始化结果大小列表
    # 对于对象的每个维度进行遍历
    for i in range(len(self)):
        # 如果当前维度索引等于指定的维度参数 dim
        if dim == i:
            # 将指定维度的大小（元素数）添加到结果大小列表中
            result_size.append(numel)
        else:
            # 将当前维度的大小添加到结果大小列表中
            result_size.append(self[i])
    # 返回最终的结果大小列表
    return result_size
def embedding(
    weight: List[int],
    indices: List[int],
    padding_idx: int = -1,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
):
    assert len(weight) == 2
    # 检查权重列表长度是否为2，用于嵌入层的权重维度验证

    if len(indices) == 1:
        return index_select(weight, 0, indices)
    # 如果索引列表长度为1，直接使用索引选择函数对权重进行选择并返回

    size = _copy(indices)
    size.append(weight[1])
    # 复制索引列表并加入权重的第二维度大小，用于返回嵌入向量的大小
    return size
    # 返回嵌入向量的大小


def max_int():
    return 9223372036854775807
    # 返回Python中能表示的最大整数


def slice(
    self: List[int], dim: int, start: Optional[int], end: Optional[int], step: int
):
    ndim = len(self)
    assert ndim != 0
    # 检查张量的维度是否为0

    dim = maybe_wrap_dim(dim, ndim)
    # 将维度索引包装在张量维度内，处理负数索引的情况

    start_val = start if start is not None else 0
    end_val = end if end is not None else max_int()
    # 设置起始和结束值，如果未提供则使用默认值或最大整数

    assert step > 0
    # 检查步长是否大于0

    if start_val == max_int():
        start_val = 0
    # 如果起始值为最大整数，将其设置为0

    if start_val < 0:
        start_val += self[dim]
    # 处理负数起始索引，将其转换为正数索引

    if end_val < 0:
        end_val += self[dim]
    # 处理负数结束索引，将其转换为正数索引

    if start_val < 0:
        start_val = 0
    elif start_val > self[dim]:
        start_val = self[dim]
    # 处理起始索引超出张量维度范围的情况

    if end_val < start_val:
        end_val = start_val
    elif end_val >= self[dim]:
        end_val = self[dim]
    # 处理结束索引超出或小于起始索引的情况

    slice_len = end_val - start_val
    # 计算切片的长度

    out = _copy(self)
    out[dim] = (slice_len + step - 1) // step
    # 使用切片长度和步长计算切片后张量在给定维度上的大小

    return out
    # 返回切片后的张量


def check_cat_no_zero_dim(tensors: List[List[int]]):
    for tensor in tensors:
        assert len(tensor) > 0
    # 检查每个张量列表中的张量是否至少有一个维度


def legacy_cat_wrap_dim(dim: int, tensor_sizes: List[List[int]]):
    out_dim: Optional[int] = None
    for size in tensor_sizes:
        if not (len(size) == 1 and size[0] == 0):
            if out_dim is None:
                out_dim = maybe_wrap_dim(dim, len(size))
    # 在传统的张量拼接中，确定用于拼接的维度索引

    if out_dim is None:
        out_dim = dim
    # 如果没有明确指定拼接维度，则使用原始维度索引

    return out_dim
    # 返回用于拼接的维度索引


def should_skip(tensor: List[int]):
    return numel(tensor) == 0 and len(tensor) == 1
    # 检查张量是否为空或只包含一个元素


def check_cat_shape_except_dim(
    first: List[int], second: List[int], dimension: int, index: int
):
    first_dims = len(first)
    second_dims = len(second)
    assert first_dims == second_dims, "Tensors must have same number of dimensions"
    # 检查两个张量的维度数是否相同

    for dim in range(0, first_dims):
        if dim != dimension:
            assert (
                first[dim] == second[dim]
            ), "Sizes of tensors must match except in dimension"
    # 检查除了指定维度以外的所有维度上的大小是否相同


def cat(tensors: List[List[int]], dim: int):
    check_cat_no_zero_dim(tensors)
    # 检查拼接的张量列表中是否有空张量

    dim = legacy_cat_wrap_dim(dim, tensors)
    # 确定用于拼接的维度索引

    assert len(tensors) > 0
    # 至少要有一个张量用于拼接

    not_skipped_tensor: Optional[List[int]] = None
    for tensor in tensors:
        if not should_skip(tensor):
            not_skipped_tensor = tensor
    # 查找第一个不应跳过的张量

    if not_skipped_tensor is None:
        return [0]
    # 如果没有找到可用于拼接的张量，则返回一个空张量

    cat_dim_size = 0
    # 初始化拼接维度的大小为0

    for i in range(len(tensors)):
        tensor = tensors[i]
        if not should_skip(tensor):
            check_cat_shape_except_dim(not_skipped_tensor, tensor, dim, i)
            cat_dim_size = cat_dim_size + tensor[dim]
    # 计算拼接维度的总大小

    result_size = _copy(not_skipped_tensor)
    result_size[dim] = cat_dim_size
    # 构建并返回拼接后张量的大小

    return result_size
    # 返回拼接后张量的大小


def stack(tensors: List[List[int]], dim: int):
    unsqueezed_tensors: List[List[int]] = []
    # 遍历输入的张量列表
    for tensor in tensors:
        # 对当前张量在指定维度上进行展开操作，返回展开后的张量
        unsqueezed = unsqueeze(tensor, dim)
        # 将展开后的张量添加到结果列表中
        unsqueezed_tensors.append(unsqueezed)
    # 对所有展开后的张量按指定维度进行拼接操作，返回拼接后的张量
    return cat(unsqueezed_tensors, dim)
def select(self: List[int], dim: int, index: int):
    # 获取列表的维度数
    ndim = len(self)
    # 确保列表维度不为零
    assert ndim != 0
    # 将维度索引可能包装成有效的维度索引
    dim = maybe_wrap_dim(dim, ndim)
    # 获取指定维度上的大小
    size = self[dim]
    # 确保索引在有效范围内
    assert not (index < -size or index >= size)
    # 如果索引为负数，转换为对应正数索引
    if index < 0:
        index += size
    # 初始化输出列表
    out: List[int] = []
    # 遍历所有维度
    for i in range(ndim):
        # 排除指定的维度，将其他维度的元素添加到输出列表中
        if i != dim:
            out.append(self[i])
    # 返回输出列表
    return out


def matmul(tensor1: List[int], tensor2: List[int]):
    # 获取张量1和张量2的维度数
    dim_tensor1 = len(tensor1)
    dim_tensor2 = len(tensor2)
    # 根据不同的维度情况进行张量乘法
    if dim_tensor1 == 1 and dim_tensor2 == 1:
        return dot(tensor1, tensor2)
    elif dim_tensor1 == 2 and dim_tensor2 == 1:
        return mv(tensor1, tensor2)
    elif dim_tensor1 == 1 and dim_tensor2 == 2:
        return squeeze(mm(unsqueeze(tensor1, 0), tensor2), 0)
    elif dim_tensor1 == 2 and dim_tensor2 == 2:
        return mm(tensor1, tensor2)
    elif dim_tensor1 >= 1 and dim_tensor2 >= 1:
        # 计算张量乘法的输出形状
        n = tensor1[-2] if dim_tensor1 > 1 else 1
        m1 = tensor1[-1]
        batch_tensor1: List[int] = []
        # TODO: 处理切片
        for i in range(dim_tensor1 - 2):
            batch_tensor1.append(tensor1[i])
        m2 = tensor2[-1] if dim_tensor2 > 1 else 1
        p = tensor2[-1]
        batch_tensor2: List[int] = []
        # TODO: 处理切片
        for i in range(dim_tensor2 - 2):
            batch_tensor2.append(tensor2[i])

        # 扩展批处理部分（即截取矩阵维度并扩展剩余部分）
        expand_batch_portion = broadcast(batch_tensor1, batch_tensor2)

        # 输出形状
        output_shape = expand_batch_portion
        if dim_tensor1 > 1:
            output_shape.append(n)
        if dim_tensor2 > 1:
            output_shape.append(p)

        return output_shape
    else:
        assert False, "matmul的两个参数必须至少是1维的"


def t(self: List[int]):
    # 确保列表长度不超过2
    assert len(self) <= 2
    self_len = len(self)
    # 根据列表长度返回相应转置后的列表
    if self_len == 0:
        out: List[int] = []
        return out
    elif self_len == 1:
        return [self[0]]
    else:
        return [self[1], self[0]]


def transpose(self: List[int], dim0: int, dim1: int):
    # 获取列表的维度数
    ndims = len(self)
    # 将维度索引可能包装成有效的维度索引
    dim0 = maybe_wrap_dim(dim0, ndims)
    dim1 = maybe_wrap_dim(dim1, ndims)
    # 如果两个维度索引相同，返回列表的副本
    if dim0 == dim1:
        return _copy(self)
    # 初始化输出列表
    out: List[int] = []
    # 遍历所有维度
    for i in range(ndims):
        # 将维度索引为dim0的元素放在维度索引为dim1的位置，反之亦然
        if i == dim0:
            out.append(self[dim1])
        elif i == dim1:
            out.append(self[dim0])
        else:
            out.append(self[i])
    # 返回输出列表
    return out


def linear(input: List[int], weight: List[int], bias: Optional[List[int]]):
    # 使用matmul函数计算输入和权重的乘积，并使用权重的转置
    out = matmul(input, t(weight))
    # 如果存在偏置项，确保其与输出兼容
    if bias is not None:
        assert broadcast(bias, out) == out
    # 返回结果
    return out


def addmm(self: List[int], mat1: List[int], mat2: List[int], beta: Any, alpha: Any):
    # 返回张量和两个矩阵相乘后的广播结果
    return broadcast(self, mm(mat1, mat2))
# 检查给定数组中是否有负数，返回布尔值。若存在负数，则返回 True，否则返回 False。
def check_non_negative(array: List[int]) -> bool:
    non_negative = False
    for val in array:
        if val < 0:
            non_negative = True
    return non_negative


# 检查卷积层正向传播的输入形状和参数的合法性
def check_shape_forward(
    input: List[int],
    weight_sizes: List[int],
    bias: Optional[List[int]],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int,
):
    k = len(input)  # 获取输入维度
    weight_dim = len(weight_sizes)  # 获取权重尺寸的维度

    # 确保填充和步幅参数非负
    assert not check_non_negative(padding)
    assert not check_non_negative(stride)

    # 确保输入维度与权重尺寸维度相同
    assert weight_dim == k
    # 确保权重的第一个维度大于等于分组数，并且可以被分组数整除
    assert weight_sizes[0] >= groups
    assert (weight_sizes[0] % groups) == 0
    # 确保输入的第二个维度与权重的第二个维度乘以分组数相同
    assert input[1] == weight_sizes[1] * groups
    # 确保偏置为空或者长度为1且与权重的第一个维度相同
    assert bias is None or (len(bias) == 1 and bias[0] == weight_sizes[0])

    # 对于从第二个维度开始的每个维度，确保输入大小至少与计算出的卷积核大小一样大
    for i in range(2, k):
        assert (input[i] + 2 * padding[i - 2]) >= (
            dilation[i - 2] * (weight_sizes[i] - 1) + 1
        )

    # 这里暂未处理转置卷积


# 计算卷积层输出的大小
def conv_output_size(
    input_size: List[int],
    weight_size: List[int],
    bias: Optional[List[int]],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int,
):
    check_shape_forward(
        input_size, weight_size, bias, stride, padding, dilation, groups
    )

    has_dilation = len(dilation) > 0  # 检查是否存在膨胀参数
    dim = len(input_size)  # 获取输入的维度
    output_size: List[int] = []
    input_batch_size_dim = 0
    weight_output_channels_dim = 0
    output_size.append(input_size[input_batch_size_dim])  # 将输入的批次大小添加到输出大小列表
    output_size.append(weight_size[weight_output_channels_dim])  # 将权重的输出通道数添加到输出大小列表

    # 计算从第二个维度开始的每个维度的输出大小
    for d in range(2, dim):
        dilation_ = dilation[d - 2] if has_dilation else 1  # 获取膨胀系数或者默认为1
        kernel = dilation_ * (weight_size[d] - 1) + 1  # 计算卷积核的大小
        output_size.append(
            (input_size[d] + (2 * padding[d - 2]) - kernel) // stride[d - 2] + 1
        )  # 计算输出大小并添加到输出大小列表

    return output_size  # 返回计算得到的输出大小列表


# 对一维卷积的输入参数进行检查，并计算输出大小
def conv1d(
    input: List[int],
    weight: List[int],
    bias: Optional[List[int]],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int,
):
    assert len(weight) == 3  # 确保权重的维度为3
    assert len(input) == 3  # 确保输入的维度为3
    return conv_output_size(input, weight, bias, stride, padding, dilation, groups)


# 对二维卷积的输入参数进行检查，并计算输出大小
def conv2d(
    input: List[int],
    weight: List[int],
    bias: Optional[List[int]],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int,
):
    assert len(weight) == 4  # 确保权重的维度为4
    assert len(input) == 4  # 确保输入的维度为4
    return conv_output_size(input, weight, bias, stride, padding, dilation, groups)


# 计算卷积层反向传播时的输入参数
def conv_backwards(
    grad_output: List[int],
    input: List[int],
    weight: List[int],
    biases: Optional[List[int]],
):
    # 偏置梯度总是生成的，无论是否提供偏置
    return _copy(input), _copy(weight), [grad_output[1]]


# 对二维转置卷积的输入参数进行处理，尚未实现
def conv_transpose2d_input(
    input: List[int],
    weight: List[int],
    # 定义可选的整数列表bias，若未提供则默认为None
    bias: Optional[List[int]] = None,
    # 定义可选的整数列表stride，若未提供则默认为None
    stride: Optional[List[int]] = None,
    # 定义可选的整数列表padding，若未提供则默认为None
    padding: Optional[List[int]] = None,
    # 定义可选的整数列表output_padding，若未提供则默认为None
    output_padding: Optional[List[int]] = None,
    # 定义整数groups，默认为1，用于指定卷积操作中的分组数量
    groups: int = 1,
    # 定义可选的整数列表dilation，若未提供则默认为None，用于指定卷积核元素之间的间隔
    dilation: Optional[List[int]] = None,
# 定义一个函数，计算卷积层的输出尺寸
def conv_forwards(
    input: List[int],            # 输入张量的维度列表
    weight: List[int],           # 卷积核张量的维度列表
    bias: Optional[List[int]],   # 可选的偏置张量的维度列表
    stride: List[int],           # 卷积操作的步长列表
    padding: List[int],          # 卷积操作的填充列表
    dilation: List[int],         # 卷积核的膨胀率列表
    transposed: bool,            # 是否是转置卷积
    output_padding: List[int],   # 转置卷积的输出填充列表
    groups: int,                 # 卷积分组数
) -> List[int]:                  # 返回值是输出张量的维度列表
    has_dilation = len(dilation) > 0               # 检查是否设置了膨胀率
    has_output_padding = len(output_padding) > 0   # 检查是否设置了输出填充
    dim = len(input)                               # 获取输入张量的维度数
    output_size: List[int] = []                    # 初始化输出尺寸的空列表
    input_batch_size_dim = 0                       # 输入张量的批次大小所在维度
    weight_output_channels_dim = 1 if transposed else 0  # 卷积核输出通道所在维度

    # 将批次大小添加到输出尺寸列表中
    output_size.append(input[input_batch_size_dim])

    # 根据是否是转置卷积，确定卷积核输出通道数的计算方式
    if transposed:
        output_size.append(weight[weight_output_channels_dim] * groups)
    else:
        output_size.append(weight[weight_output_channels_dim])

    # 遍历除批次大小和输出通道大小外的每个维度
    for d in range(2, dim):
        dilation_ = dilation[d - 2] if has_dilation else 1         # 获取当前维度的膨胀率
        output_padding_ = output_padding[d - 2] if has_output_padding else 0  # 获取当前维度的输出填充

        if transposed:
            kernel = dilation_ * (weight[d] - 1)                    # 计算转置卷积的卷积核大小
            # 计算转置卷积的输出尺寸，并添加到输出尺寸列表中
            output_size.append(
                (input[d] - 1) * stride[d - 2]
                - 2 * padding[d - 2]
                + kernel
                + output_padding_
                + 1
            )
        else:
            kernel = dilation_ * (weight[d] - 1) + 1                # 计算正常卷积的卷积核大小
            # 计算正常卷积的输出尺寸，并添加到输出尺寸列表中
            output_size.append(
                (input[d] + (2 * padding[d - 2]) - kernel) // stride[d - 2] + 1
            )

    return output_size   # 返回最终的输出尺寸列表


# 定义一个辅助函数，调用conv_forwards函数进行卷积操作
def _conv_forwards(
    input: List[int],            # 输入张量的维度列表
    weight: List[int],           # 卷积核张量的维度列表
    bias: Optional[List[int]],   # 可选的偏置张量的维度列表
    stride: List[int],           # 卷积操作的步长列表
    padding: List[int],          # 卷积操作的填充列表
    dilation: List[int],         # 卷积核的膨胀率列表
    transposed: bool,            # 是否是转置卷积
    output_padding: List[int],   # 转置卷积的输出填充列表
    groups: int,                 # 卷积分组数
    benchmark: bool,             # 是否启用基准模式
    deterministic: bool,         # 是否使用确定性计算
    cudnn_enabled: bool,         # 是否启用cudnn加速
    allow_tf32: bool,            # 是否允许tf32加速
) -> List[int]:                  # 返回值是输出张量的维度列表
    return conv_forwards(        # 调用conv_forwards函数进行卷积操作
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    )


# 定义一个函数，实现批量归一化操作
def batch_norm(
    input: List[int],                # 输入张量的维度列表
    weight: Optional[List[int]],     # 可选的权重张量的维度列表
    bias: Optional[List[int]],       # 可选的偏置张量的维度列表
    running_mean: Optional[List[int]],   # 可选的运行时均值张量的维度列表
    running_var: Optional[List[int]],    # 可选的运行时方差张量的维度列表
    training: bool,                 # 是否处于训练模式
    momentum: float,                # 动量参数
    eps: float,                     # 用于数值稳定性的小值
    cudnn_enabled: bool,            # 是否启用cudnn加速
def zero_dim_tensor(input: Any):
    # 初始化一个空列表，用于表示零维张量
    out: List[int] = []
    # 返回空列表作为零维张量的表示
    return out


def arange_end(end: number, inp0: Any, inp1: Any, inp2: Any, inp3: Any):
    # 断言终止值必须为非负数
    assert end >= 0
    # 返回一个包含终止值向上取整后的整数的列表
    return [int(math.ceil(end))]


def arange_start(
    start: number, end: number, inp0: Any, inp1: Any, inp2: Any, inp3: Any
):
    # 断言终止值和起始值必须为非负数，且终止值必须大于等于起始值
    assert end >= 0
    assert end >= start
    # 返回一个包含从起始值到终止值向上取整后的整数个数的列表
    return [int(math.ceil(end - start))]


def arange_start_step(
    start: number, end: number, step: number, inp0: Any, inp1: Any, inp2: Any, inp3: Any
):
    # 断言步长不能为零
    assert step != 0
    # 如果步长为负数，则起始值必须大于等于终止值；否则，终止值必须大于等于起始值
    if step < 0:
        assert start >= end
    else:
        assert end >= start
    # 返回一个包含从起始值到终止值，按照给定步长向上取整后的整数个数的列表
    return [int(math.ceil((end - start) / step))]


def permute(input: List[int], dims: List[int]):
    # 断言输入的维度列表和指定的维度列表长度相同
    assert len(input) == len(dims)
    # 获取输入张量的总维度数
    ndim = len(dims)
    # 初始化已经处理过的维度和新尺寸列表
    seen_dims: List[int] = []
    newSizes: List[int] = []
    # 遍历每个维度
    for i in range(ndim):
        # 对当前维度进行可能的包装
        dim = maybe_wrap_dim(dims[i], ndim)
        # 记录已处理的维度和新的尺寸
        seen_dims.append(dim)
        newSizes.append(input[dim])
    # 断言所有的维度都是唯一的
    for i in range(1, ndim):
        for j in range(i):
            assert seen_dims[i] != seen_dims[j]
    # 返回新的尺寸列表
    return newSizes


def movedim(self: List[int], source: List[int], destination: List[int]) -> List[int]:
    # 获取输入张量的维度数
    self_dim = len(self)
    # 如果张量维度小于等于1，则直接返回原张量
    if self_dim <= 1:
        return self
    # 初始化规范化后的源维度和目标维度列表
    normalized_src: List[int] = []
    normalized_dst: List[int] = []
    # 遍历源和目标维度列表，进行可能的包装，并记录
    for i in range(len(source)):
        normalized_src.append(maybe_wrap_dim(source[i], self_dim))
        normalized_dst.append(maybe_wrap_dim(destination[i], self_dim))
    # 初始化顺序和维度列表
    order = [-1 for i in range(self_dim)]
    src_dims = [i for i in range(self_dim)]
    dst_dims = [i for i in range(self_dim)]

    # 根据规范化后的源和目标维度列表，确定顺序和维度列表
    for i in range(len(source)):
        order[normalized_dst[i]] = normalized_src[i]
        src_dims[normalized_src[i]] = -1
        dst_dims[normalized_dst[i]] = -1

    # 初始化源和目标维度列表
    source_dims: List[int] = []
    destination_dims: List[int] = []
    for ele in src_dims:
        if ele != -1:
            source_dims.append(ele)
    for ele in dst_dims:
        if ele != -1:
            destination_dims.append(ele)

    # 计算剩余维度数
    rest_dim = self_dim - len(source)
    # 根据源和目标维度列表，确定顺序
    for i in range(rest_dim):
        order[destination_dims[i]] = source_dims[i]
    # 调用当前对象（self）的 permute 方法，传入 order 参数，并返回结果
    return permute(self, order)
def flatten(input: List[int], start_dim: int, end_dim: int):
    # 使用 maybe_wrap_dim 函数确保 start_dim 在有效范围内
    start_dim = maybe_wrap_dim(start_dim, len(input))
    # 使用 maybe_wrap_dim 函数确保 end_dim 在有效范围内
    end_dim = maybe_wrap_dim(end_dim, len(input))
    # 断言 start_dim 不大于 end_dim
    assert start_dim <= end_dim
    # 如果 input 列表为空，返回包含数字 1 的列表
    if len(input) == 0:
        return [1]
    # 如果 start_dim 等于 end_dim，返回 input 的副本
    if start_dim == end_dim:
        # TODO: return self
        out: List[int] = []
        # 将 input 中的每个元素添加到 out 列表中
        for elem in input:
            out.append(elem)
        return out
    # 计算从 start_dim 到 end_dim 的所有维度的元素总数
    slice_numel = 1
    for i in range(start_dim, end_dim + 1):
        slice_numel *= input[i]
    # TODO: 当切片优化可用时使用切片操作
    # slice_numel = multiply_integers(input[start_dim:end_dim - start_dim + 1])
    # 构建新的形状列表，复制 input 中除了指定范围外的维度
    shape: List[int] = []
    for i in range(start_dim):
        shape.append(input[i])
    # 添加 slice_numel 作为新的维度大小
    shape.append(slice_numel)
    # 复制 end_dim 后面的所有维度到 shape 中
    for i in range(end_dim + 1, len(input)):
        shape.append(input[i])
    return shape


def nonzero_lower_bound(input: List[int]):
    # 返回一个列表，包含零和 input 的长度
    return [0, len(input)]


def nonzero_upper_bound(input: List[int]):
    # 返回一个列表，包含 input 的元素总数和 input 的长度
    return [numel(input), len(input)]


def _reduce_along_dim(self: List[int], dim: int, keepdim: bool):
    # 使用 maybe_wrap_dim 函数确保 dim 在有效范围内
    dim = maybe_wrap_dim(dim, len(self))
    # 创建一个空列表 out 用于存储结果
    out: List[int] = []
    # 遍历 self 中的每个维度 self_dim 和对应的索引 i
    for i, self_dim in enumerate(self):
        # 如果 i 等于 dim，且 keepdim 为 True，则添加数字 1 到 out
        if i == dim:
            if keepdim:
                out.append(1)
        else:
            # 否则将 self_dim 添加到 out 中
            out.append(self_dim)
    return out


def argmax(
    self: List[int], dim: Optional[int] = None, keepdim: bool = False
) -> List[int]:
    # 如果 dim 是 None，返回一个空列表
    if dim is None:
        return []
    # 调用 _reduce_along_dim 函数，返回在指定维度上的最大值
    return _reduce_along_dim(self, dim, keepdim)


def bmm(self: List[int], mat2: List[int]) -> List[int]:
    # 断言 self 和 mat2 是 3D 张量
    assert len(self) == 3, "bmm only supports 3D tensors"
    assert len(mat2) == 3, "bmm only supports 3D tensors"
    # 断言 self 和 mat2 的批次维度相同
    assert self[0] == mat2[0], "mismatching batch dimension"
    # 断言 self 和 mat2 的合并维度相同
    assert self[2] == mat2[1], "mismatching contracting dimension"
    # 返回一个列表，包含 self 的批次和 mat2 的第二个维度
    return [self[0], self[1], mat2[2]]


def _shape_as_tensor(self: List[int]) -> List[int]:
    # 返回一个列表，包含 self 的长度
    return [len(self)]


def topk(self: List[int], k: int, dim: int = -1) -> Tuple[List[int], List[int]]:
    # 如果 self 列表为空，返回一个空列表作为结果
    if len(self) == 0:
        result: List[int] = []
    else:
        # 断言 k 不大于指定维度的大小
        assert (
            k <= self[dim]
        ), f"k ({k}) is too big for dimension {dim} of size {self[dim]}"
        # 复制 self 列表到 result 中
        result = _copy(self)
        # 将 result 在指定维度 dim 处的大小设置为 k
        result[dim] = k
    return result, result


def nll_loss_forward(
    self: List[int], target: List[int], weight: Optional[List[int]], reduction: int
) -> Tuple[List[int], List[int]]:
    # 这部分直接从 LossNLL.cpp 中的元函数中复制而来
    # 获取 self 和 target 的维度
    self_dim = len(self)
    target_dim = len(target)
    # 断言 self 的维度在 1 到 2 之间，target 的维度在 0 到 1 之间
    assert 0 < self_dim <= 2
    assert target_dim <= 1
    # 如果没有批次维度，确保 self 的第一个维度与 target 的第一个维度相同
    no_batch_dim = self_dim == 1 and target_dim == 0
    assert no_batch_dim or (self[0] == target[0])
    # 获取类别数量
    n_classes = self[-1]
    # scalar_shape 列表存储标量形状
    scalar_shape: List[int] = []
    # 断言 weight 为 None 或者 weight 的长度为 1，且等于类别数量
    assert weight is None or (len(weight) == 1 and weight[0] == n_classes)
    # 如果 reduction 等于 0 且 self 的维度为 2，返回 reduction_shape，否则返回 scalar_shape
    if reduction == 0 and self_dim == 2:
        reduction_shape = [self[0]]
    else:
        reduction_shape = scalar_shape
    return reduction_shape, scalar_shape
# 定义一个函数，实现原生层的归一化操作，接受输入和归一化形状作为参数，并返回归一化后的结果
def native_layer_norm(
    input: List[int], normalized_shape: List[int]
) -> Tuple[List[int], List[int], List[int]]:
    # 初始化一个空的减少形状列表
    reduction_shape: List[int] = []
    # 计算未减少维度的数量
    num_unreduced_dimensions = len(input) - len(normalized_shape)
    # 确保未减少的维度数量不小于零
    assert num_unreduced_dimensions >= 0
    # 将前面未减少的维度添加到减少形状列表中
    for i in range(num_unreduced_dimensions):
        reduction_shape.append(input[i])
    # 将后面减少的维度（置为1）添加到减少形状列表中
    for i in range(num_unreduced_dimensions, len(input)):
        reduction_shape.append(1)
    # 返回输入的拷贝、减少形状列表及其拷贝
    return _copy(input), reduction_shape, reduction_shape


# 定义一个函数，实现原生批量归一化操作，接受输入、权重、偏置、移动均值、移动方差、训练标志作为参数，并返回处理后的结果
def native_batch_norm(
    input: List[int],
    weight: Optional[List[int]],
    bias: Optional[List[int]],
    running_mean: Optional[List[int]],
    running_var: Optional[List[int]],
    training: bool,
) -> Tuple[List[int], List[int], List[int]]:
    # 如果处于训练模式，将_size设置为输入的第二个维度的列表，否则设置为[0]
    if training:
        _size = [input[1]]
    else:
        _size = [0]
    # 返回输入的拷贝、_size及其拷贝
    return _copy(input), _size, _size


# 定义一个函数，实现带更新的批量归一化操作，接受输入、权重、偏置、移动均值、移动方差作为参数，并返回处理后的结果
def _batch_norm_with_update(
    input: List[int],
    weight: Optional[List[int]],
    bias: Optional[List[int]],
    running_mean: Optional[List[int]],
    running_var: Optional[List[int]],
) -> Tuple[List[int], List[int], List[int], List[int]]:
    # 设置_size为输入的第二个维度的列表
    _size = [input[1]]
    # 返回输入的拷贝、_size及其拷贝，以及包含0的列表
    return _copy(input), _size, _size, [0]


# 定义一个交叉熵损失函数，接受自身、目标、权重、减少标志、忽略索引和标签平滑值作为参数，并返回结果的形状
def cross_entropy_loss(
    self: List[int],
    target: List[int],
    weight: Optional[List[int]] = None,
    reduction: int = 1,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> List[int]:
    # 调用nll_loss_forward函数计算损失结果的形状，并返回
    result_shape = nll_loss_forward(self, target, weight, reduction)[0]
    return result_shape


# 以下部分未启用，可能因为暂停添加操作
# 定义一个索引Tensor的函数，接受自身和索引作为参数，并返回广播后的形状
# def index_Tensor(self: List[int], indices: List[Optional[List[int]]]) -> List[int]:
#     assert len(indices) <= len(self), "More indices than dimensions to index"
#     broadcasted_shape: List[int] = []
#     for index_tensor_shape in indices:
#         if index_tensor_shape is not None:
#             broadcasted_shape = broadcast(broadcasted_shape, index_tensor_shape)
#     return broadcasted_shape


# 初始化一个脚本函数映射字典和一个有界计算图映射字典
ScriptFn = torch._C.ScriptFunction
shape_compute_graph_mapping: Dict[str, ScriptFn] = {}
bounded_compute_graph_mapping: Dict[str, Tuple[ScriptFn, ScriptFn]] = {}
script_func_map: Dict[Callable, ScriptFn] = {}


# 定义一个处理函数，接受一个函数作为参数，并返回相应的脚本化函数
def process_func(func: Callable):
    # 如果函数不在脚本函数映射中，则将其脚本化并进行一系列优化处理，最终存储到映射字典中并返回
    if func not in script_func_map:
        scripted_func = torch.jit.script(func)

        torch._C._jit_pass_inline(scripted_func.graph)

        for _ in range(2):
            torch._C._jit_pass_peephole(scripted_func.graph)
            torch._C._jit_pass_constant_propagation(scripted_func.graph)

        script_func_map[func] = scripted_func
    return script_func_map[func]
# 添加形状计算映射函数，将操作符模式(operator_schema)映射到相应的处理函数(func)上
def add_shape_compute_mapping(operator_schema: str, func: Callable):
    # 将操作符模式(operator_schema)和处理函数(func)添加到全局形状计算图映射表(shape_compute_graph_mapping)中
    global shape_compute_graph_mapping
    shape_compute_graph_mapping[operator_schema] = process_func(func)

# 添加有界计算映射函数，为操作符模式(operator_schema)添加上界和下界的形状计算函数
def add_bounded_compute_mapping(
    operator_schema: str, lower_bound_func: Callable, upper_bound_func: Callable
):
    # 处理上界和下界计算函数，并存储为元组
    fns = (process_func(lower_bound_func), process_func(upper_bound_func))
    # 将操作符模式(operator_schema)和计算函数元组(fns)添加到有界计算图映射表(bounded_compute_graph_mapping)中
    bounded_compute_graph_mapping[operator_schema] = fns

# 调用 add_shape_compute_mapping 函数，添加对特定操作符的形状计算函数映射
add_shape_compute_mapping(
    "aten::contiguous(Tensor(a) self, *, MemoryFormat memory_format=contiguous_format) -> Tensor(a)",
    unary,
)
add_shape_compute_mapping(
    "aten::rsub.Tensor(Tensor self, Scalar other, Scalar alpha=1) -> Tensor", unary
)
add_shape_compute_mapping(
    "aten::dropout(Tensor input, float p, bool train) -> Tensor", unary
)
add_shape_compute_mapping(
    "aten::adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor",
    adaptive_avg_pool2d,
)
add_shape_compute_mapping(
    "prim::NumToTensor.Scalar(Scalar a) -> Tensor", zero_dim_tensor
)
add_shape_compute_mapping("prim::NumToTensor.bool(bool a) -> Tensor", zero_dim_tensor)
add_shape_compute_mapping(
    "aten::zeros(int[] size, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor)",
    unary,
)
add_shape_compute_mapping(
    "aten::to.dtype(Tensor(a) self, int dtype, bool non_blocking=False, bool copy=False, int? memory_format=None) -> (Tensor(a))",
    unary,
)
add_shape_compute_mapping(
    "aten::arange(Scalar end, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor)",
    arange_end,
)
add_shape_compute_mapping(
    "aten::arange.start(Scalar start, Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
    arange_start,
)
add_shape_compute_mapping(
    "aten::arange.start_step(Scalar start, Scalar end, Scalar step, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
    arange_start_step,
)
add_shape_compute_mapping("aten::squeeze(Tensor(a) self) -> Tensor(a)", squeeze_nodim)
add_shape_compute_mapping(
    "aten::squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)", squeeze
)
add_shape_compute_mapping(
    "aten::squeeze.dims(Tensor(a) self, int[] dim) -> Tensor(a)", squeeze_dims
)
add_shape_compute_mapping(
    "aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)", unsqueeze
)
add_shape_compute_mapping(
    "aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)",
    slice,
)
add_shape_compute_mapping(
    "aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)", select
)
add_shape_compute_mapping(
    "aten::index_select(Tensor self, int dim, Tensor index) -> Tensor", index_select
)
# 操作符模式字符串未完全显示，将在代码编辑器中继续显示和注释
    # 定义一个函数签名，参数包括一个 float 类型的 eps（默认值为 1e-05）和一个布尔类型的 cudnn_enable（默认值为 True），返回一个 Tensor 对象
    "float eps=1e-05, bool cudnn_enable=True) -> Tensor",
    # 调用一个名为 unary 的函数或方法
    unary,
# 添加形状计算映射函数，处理 "aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor" 这类操作，使用 unary 函数
add_shape_compute_mapping(
    "aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor", unary
)

# 添加形状计算映射函数，处理 "aten::_no_grad_embedding_renorm_(Tensor weight, Tensor input, float max_norm, float norm_type) -> Tensor" 这类操作，使用 unary 函数
add_shape_compute_mapping(
    "aten::_no_grad_embedding_renorm_(Tensor weight, Tensor input, float max_norm, float norm_type) -> Tensor",
    unary,
)

# 添加形状计算映射函数，处理 "aten::embedding_renorm_(Tensor(a!) self, Tensor indices, float max_norm, float norm_type) -> Tensor(a!)" 这类操作，使用 unary 函数
add_shape_compute_mapping(
    "aten::embedding_renorm_(Tensor(a!) self, Tensor indices, float max_norm, float norm_type) -> Tensor(a!)",
    unary,
)

# 添加形状计算映射函数，处理 "aten::embedding(Tensor weight, Tensor indices, int padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor" 这类操作，使用 embedding 函数
add_shape_compute_mapping(
    "aten::embedding(Tensor weight, Tensor indices, int padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor",
    embedding,
)

# 添加形状计算映射函数，处理 "aten::mm(Tensor self, Tensor mat2) -> Tensor" 这类操作，使用 mm 函数
add_shape_compute_mapping("aten::mm(Tensor self, Tensor mat2) -> Tensor", mm)

# 添加形状计算映射函数，处理 "aten::dot(Tensor self, Tensor tensor) -> Tensor" 这类操作，使用 dot 函数
add_shape_compute_mapping("aten::dot(Tensor self, Tensor tensor) -> Tensor", dot)

# 添加形状计算映射函数，处理 "aten::mv(Tensor self, Tensor vec) -> Tensor" 这类操作，使用 mv 函数
add_shape_compute_mapping("aten::mv(Tensor self, Tensor vec) -> Tensor", mv)

# 添加形状计算映射函数，处理 "aten::matmul(Tensor self, Tensor other) -> Tensor" 这类操作，使用 matmul 函数
add_shape_compute_mapping("aten::matmul(Tensor self, Tensor other) -> Tensor", matmul)

# 添加形状计算映射函数，处理 "aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor" 这类操作，使用 linear 函数
add_shape_compute_mapping(
    "aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor", linear
)

# 添加形状计算映射函数，处理 "aten::max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor" 这类操作，使用 max_pool2d 函数
add_shape_compute_mapping(
    "aten::max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor",
    max_pool2d,
)

# 添加形状计算映射函数，处理 "aten::max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)" 这类操作，使用 max_pool2d_with_indices 函数
add_shape_compute_mapping(
    "aten::max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)",
    max_pool2d_with_indices,
)

# 添加形状计算映射函数，处理 "aten::t(Tensor(a) self) -> Tensor(a)" 这类操作，使用 t 函数
add_shape_compute_mapping("aten::t(Tensor(a) self) -> Tensor(a)", t)

# 添加形状计算映射函数，处理 "aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)" 这类操作，使用 transpose 函数
add_shape_compute_mapping(
    "aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)", transpose
)

# 添加形状计算映射函数，处理 "aten::conv1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, int[1] padding=0, int[1] dilation=1, int groups=1) -> Tensor" 这类操作，使用 conv1d 函数
add_shape_compute_mapping(
    "aten::conv1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, int[1] padding=0, int[1] dilation=1, int groups=1) -> Tensor",
    conv1d,
)

# 添加形状计算映射函数，处理 "aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor" 这类操作，使用 conv2d 函数
add_shape_compute_mapping(
    "aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor",
    conv2d,
)

# 添加形状计算映射函数，处理 "aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor" 这类操作，使用 batch_norm 函数
add_shape_compute_mapping(
    "aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor",
    batch_norm,
)

# 添加形状计算映射函数，处理 "aten::conv3d(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] dilation=1, int groups=1) -> Tensor" 这类操作，使用 conv3d 函数
add_shape_compute_mapping(
    "aten::conv3d(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] dilation=1, int groups=1) -> Tensor",
    conv3d,
)

# 添加形状计算映射函数，处理 "aten::convolution_backward(Tensor grad_output, Tensor input, Tensor weight, int[]? bias_sizes, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)" 这类操作，使用 conv_backwards 函数
add_shape_compute_mapping(
    "aten::convolution_backward(Tensor grad_output, Tensor input, Tensor weight, int[]? bias_sizes, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)",
    conv_backwards,
)

# 添加形状计算映射函数，处理 "aten::convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor" 这类操作，使用 conv_forwards 函数
add_shape_compute_mapping(
    "aten::convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor",
    conv_forwards,
)
    # 定义字符串，描述了一个特定的函数签名和其对应的函数实现
    "aten::_convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) -> Tensor",
    # 将函数实现的引用赋值给变量 _conv_forwards
    _conv_forwards,
# 添加形状计算映射函数，处理转置卷积操作的输入
add_shape_compute_mapping(
    "aten::conv_transpose2d.input(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] output_padding=0, int groups=1, int[2] dilation=1) -> Tensor",
    conv_transpose2d_input,
)

# 添加形状计算映射函数，处理张量展平操作
add_shape_compute_mapping(
    "aten::flatten.using_ints(Tensor(a) self, int start_dim=0, int end_dim=-1) -> Tensor(a)",
    flatten,
)

# 添加形状计算映射函数，处理张量连接操作
add_shape_compute_mapping("aten::cat(Tensor[] tensors, int dim=0) -> Tensor", cat)

# 添加形状计算映射函数，处理张量堆叠操作
add_shape_compute_mapping("aten::stack(Tensor[] tensors, int dim=0) -> Tensor", stack)

# 添加形状计算映射函数，处理张量维度重排操作
add_shape_compute_mapping(
    "aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)", permute
)

# 添加形状计算映射函数，处理张量维度移动操作
add_shape_compute_mapping(
    "aten::movedim.intlist(Tensor(a) self, int[] source, int[] destination) -> Tensor(a)",
    movedim,
)

# 添加形状计算映射函数，处理张量视图重塑操作
add_shape_compute_mapping("aten::view(Tensor(a) self, int[] size) -> Tensor(a)", view)

# 添加形状计算映射函数，处理张量扩展操作
add_shape_compute_mapping(
    "aten::expand_as(Tensor(a) self, Tensor other) -> Tensor(a)", expand
)

# 添加形状计算映射函数，处理张量扩展操作（未使用）
add_shape_compute_mapping(
    "aten::expand(Tensor(a) self, int[] size, *, bool implicit=False) -> Tensor(a)",
    expand_one_unused,
)

# 添加形状计算映射函数，处理张量按维度求均值或求和操作
add_shape_compute_mapping(
    "aten::mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor",
    sum_mean_dim,
)

# 添加形状计算映射函数，处理张量按维度求均值或求和操作
add_shape_compute_mapping(
    "aten::sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor",
    sum_mean_dim,
)

# 添加形状计算映射函数，处理张量按维度求最大值操作
add_shape_compute_mapping(
    "aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)",
    max_dim,
)

# 添加形状计算映射函数，处理零维张量的均值计算操作
add_shape_compute_mapping(
    "aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor", zero_dim_tensor
)

# 添加形状计算映射函数，处理零维张量的求和计算操作
add_shape_compute_mapping(
    "aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor", zero_dim_tensor
)

# 添加形状计算映射函数，处理矩阵乘法加法操作
add_shape_compute_mapping(
    "aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor",
    addmm,
)

# 添加形状计算映射函数，处理二维最近邻上采样操作
add_shape_compute_mapping(
    "aten::upsample_nearest2d.vec(Tensor input, int[]? output_size, float[]? scale_factors) -> (Tensor)",
    upsample_nearest2d,
)

# 添加形状计算映射函数，处理张量量化操作
add_shape_compute_mapping(
    "aten::quantize_per_tensor(Tensor self, float scale, int zero_point, ScalarType dtype) -> Tensor",
    unary,
)

# 添加形状计算映射函数，处理张量量化操作
add_shape_compute_mapping(
    "aten::quantize_per_tensor.tensor_qparams(Tensor self, Tensor scale, Tensor zero_point, ScalarType dtype) -> Tensor",
    unary,
)

# 添加形状计算映射函数，处理张量反量化操作
add_shape_compute_mapping("aten::dequantize(Tensor self) -> Tensor", unary)

# 添加形状计算映射函数，处理量化操作的广播加法
add_shape_compute_mapping(
    "quantized::add(Tensor qa, Tensor qb, float scale, int zero_point) -> Tensor qc",
    broadcast,
)

# 添加形状计算映射函数，处理张量按维度求最大值的索引操作
add_shape_compute_mapping(
    "aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor", argmax
)

# 添加形状计算映射函数，处理批量矩阵乘法操作
add_shape_compute_mapping("aten::bmm(Tensor self, Tensor mat2) -> Tensor", bmm)

# 添加形状计算映射函数，将张量形状作为张量输出
add_shape_compute_mapping(
    "aten::_shape_as_tensor(Tensor self) -> Tensor", _shape_as_tensor
)
    # 定义了一个字符串，描述了一个 PyTorch 的操作符（at::topk），参数及其返回值
    "aten::topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)",
    # 将函数 topk 赋值给变量 topk，这个函数可以执行上述描述的操作
    topk,
add_shape_compute_mapping(
    "aten::nll_loss_forward(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index) -> (Tensor output, Tensor total_weight)",
    nll_loss_forward,
)
# 将 nll_loss_forward 函数映射到其符号计算函数 nll_loss_forward 上

add_shape_compute_mapping(
    "aten::native_layer_norm(Tensor input, int[] normalized_shape, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)",
    native_layer_norm,
)
# 将 native_layer_norm 函数映射到其符号计算函数 native_layer_norm 上

add_shape_compute_mapping(
    "aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)",
    native_batch_norm,
)
# 将 native_batch_norm 函数映射到其符号计算函数 native_batch_norm 上

add_shape_compute_mapping(
    "aten::_native_batch_norm_legit(Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, Tensor running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)",
    native_batch_norm,
)
# 将 _native_batch_norm_legit 函数映射到其符号计算函数 native_batch_norm 上

add_shape_compute_mapping(
    "aten::_native_batch_norm_legit.no_stats(Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, Tensor running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)",
    native_batch_norm,
)
# 将 _native_batch_norm_legit.no_stats 函数映射到其符号计算函数 native_batch_norm 上

add_shape_compute_mapping(
    "_batch_norm_with_update(Tensor input, Tensor? weight, Tensor? bias, Tensor(a!) running_mean, Tensor(b!) running_var, float momentum, float eps) -> (Tensor, Tensor, Tensor, Tensor)",
    _batch_norm_with_update,
)
# 将 _batch_norm_with_update 函数映射到其符号计算函数 _batch_norm_with_update 上

add_shape_compute_mapping(
    "aten::cross_entropy_loss(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, SymInt ignore_index=-100, float label_smoothing=0.0) -> Tensor",
    cross_entropy_loss,
)
# 将 cross_entropy_loss 函数映射到其符号计算函数 cross_entropy_loss 上

# add_shape_compute_mapping("aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor", index_Tensor)
# 注释掉的代码行，不执行映射，仅作为注释保留

# TODO: migrate over all of symbolic_shape_registry_util.cpp
# These are duplicated here so that the functions will be serialiazed

add_shape_compute_mapping(
    "aten::lerp.Tensor(Tensor self, Tensor end, Tensor weight) -> Tensor",
    broadcast_three,
)
# 将 lerp 函数映射到其符号计算函数 broadcast_three 上

add_shape_compute_mapping(
    "aten::where.ScalarSelf(Tensor condition, Scalar self, Tensor other) -> Tensor",
    broadcast_one_three,
)
# 将 where 函数映射到其符号计算函数 broadcast_one_three 上

add_shape_compute_mapping(
    "aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)",
    broadcast_inplace,
)
# 将 add_ 函数映射到其符号计算函数 broadcast_inplace 上

# quantized_conv_prepack TODO
# quantized_conv_prepack 函数尚未实现，TODO 表示待完成的工作

# Shape Compute Fn with upper and lower bounds

add_bounded_compute_mapping(
    "aten::nonzero(Tensor self) -> (Tensor)", nonzero_lower_bound, nonzero_upper_bound
)
# 将 nonzero 函数映射到其上下界计算函数 nonzero_lower_bound 和 nonzero_upper_bound 上
```