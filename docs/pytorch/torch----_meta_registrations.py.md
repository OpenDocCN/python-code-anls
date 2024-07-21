# `.\pytorch\torch\_meta_registrations.py`

```
# mypy: allow-untyped-defs
# 导入数学库
import math
# 导入枚举类型
from enum import Enum
# 导入类型提示相关模块
from typing import List, Optional, Sequence, Tuple, Union

# 导入 PyTorch 库
import torch
# 导入 Torch 内部的通用函数
import torch._prims_common as utils
# 导入 Torch 相关模块和类
from torch import SymBool, SymFloat, Tensor
# 导入 Torch 内部的分解模块
from torch._decomp import (
    _add_op_to_registry,
    _convert_out_params,
    global_decomposition_table,
    meta_table,
)
# 导入 Torch 操作重载相关模块
from torch._ops import OpOverload
# 导入 Torch 元素级运算相关模块
from torch._prims import _prim_elementwise_meta, ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND
# 导入 Torch 公共的元素级操作类型
from torch._prims_common import (
    corresponding_complex_dtype,
    corresponding_real_dtype,
    elementwise_dtypes,
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    IntLike,
    make_contiguous_strides_for,
    Number,
    TensorLike,
)
# 导入 Torch 公共的函数和包装器
from torch._prims_common.wrappers import (
    _maybe_convert_to_dtype,
    _maybe_resize_out,
    _resize_output_check,
    _safe_copy_out,
    out_wrapper,
)
# 导入 Torch 引用相关函数
from torch._refs import _broadcast_shapes, _maybe_broadcast
# 导入 Torch 的 PyTree 模块
from torch.utils import _pytree as pytree

# 使用 torch.ops 模块引用 aten 命名空间
aten = torch.ops.aten

# 定义一个 torch.library.Library 对象
_meta_lib_dont_use_me_use_register_meta = torch.library.Library("aten", "IMPL", "Meta")

# 注册元数据的装饰器函数
def register_meta(op):
    def wrapper(fn):
        # 将 fn 函数转换为支持输出参数的版本
        fn = _convert_out_params(fn)

        # 注册操作到 meta_table 中
        def register(op):
            _add_op_to_registry(meta_table, op, fn)

        # 使用 pytree.tree_map_ 遍历 op，并注册到 meta_table
        pytree.tree_map_(register, op)
        return fn

    return wrapper

# 元素级元数据函数装饰器
def elementwise_meta(
    *args,
    type_promotion: ELEMENTWISE_TYPE_PROMOTION_KIND,
):
    # 根据 type_promotion 执行类型提升
    _, result_dtype = utils.elementwise_dtypes(
        *args,
        type_promotion_kind=type_promotion,
    )
    # 将参数列表中的每个参数转换为结果类型
    args = [_maybe_convert_to_dtype(x, result_dtype) for x in args]

    # 广播参数列表中的参数
    args = _maybe_broadcast(*args)

    # 执行元素级原语检查
    return _prim_elementwise_meta(
        *args, type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT
    )

# 根据复数类型获取对应的实数类型函数
def toRealValueType(dtype):
    from_complex = {
        torch.complex32: torch.half,
        torch.cfloat: torch.float,
        torch.cdouble: torch.double,
    }
    return from_complex.get(dtype, dtype)

# 检查原位广播的函数
def check_inplace_broadcast(self_shape, *args_shape):
    # 计算广播后的形状
    broadcasted_shape = tuple(_broadcast_shapes(self_shape, *args_shape))
    # 检查广播后的形状是否与原始形状相匹配
    torch._check(
        broadcasted_shape == self_shape,
        lambda: f"output with shape {self_shape} doesn't match the broadcast shape {broadcasted_shape}",
    )

# 注册 meta_linspace_logspace 函数到 meta_table
@register_meta([aten.linspace, aten.logspace])
# 包装输出结果的装饰器函数
@out_wrapper()
# 定义 meta_linspace_logspace 函数，用于处理 linspace 和 logspace 的元数据
def meta_linspace_logspace(
    start,
    end,
    steps,
    base=None,
    dtype=None,
    device=None,
    layout=torch.strided,
    pin_memory=False,
    requires_grad=False,
):
    # 如果 start 是 torch.Tensor 类型，检查其是否为 0 维张量
    if isinstance(start, torch.Tensor):
        torch._check(
            start.dim() == 0,
            lambda: "linspace only supports 0-dimensional start and end tensors",
        )
    # 如果 end 是 torch.Tensor 类型，检查其是否为 0 维张量
    if isinstance(end, torch.Tensor):
        torch._check(
            end.dim() == 0,
            lambda: "linspace only supports 0-dimensional start and end tensors",
        )
    # 检查是否有复数类型的参数（start, end, steps 中的任何一个），如果有则需特殊处理
    if any(isinstance(arg, complex) for arg in (start, end, steps)):
        # 获取默认的复数数据类型
        default_complex_dtype = utils.corresponding_complex_dtype(
            torch.get_default_dtype()
        )
        # 如果未指定 dtype，则使用默认的复数数据类型
        if dtype is None:
            dtype = default_complex_dtype
        else:
            # 检查指定的 dtype 是否为合法的复数数据类型
            torch._check(
                utils.is_complex_dtype(dtype),
                lambda: f"linspace(): inferred dtype {default_complex_dtype} can't be safely cast to passed dtype {dtype}",
            )
    else:
        # 如果没有复数类型的参数，则使用用户指定的 dtype 或默认的 Torch 数据类型
        dtype = dtype or torch.get_default_dtype()
    
    # 断言确保 dtype 是 Torch 的数据类型对象
    assert isinstance(dtype, torch.dtype)

    # steps 参数不影响 dtype 的计算，这里进行类型检查确保 steps 是 IntLike 类型
    torch._check_type(
        isinstance(steps, IntLike),
        lambda: f"received an invalid combination of arguments - got \
# 构建字符串，描述输入的起始、结束和步长的类型
({type(start).__name__}, {type(end).__name__}, {type(steps).__name__})",
    )
    # 断言步长为 IntLike 类型（用于类型检查）
    assert isinstance(steps, IntLike)  # for mypy
    # 检查步长是否非负
    torch._check(steps >= 0, lambda: "number of steps must be non-negative")

    # 返回一个空的 Torch 张量
    return torch.empty(
        (steps,),  # type: ignore[arg-type]
        dtype=dtype,
        layout=layout,
        device="meta",
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )


@register_meta([aten.take.default, aten.take.out])
@out_wrapper()
def meta_take(self, index):
    # 类型和设备检查
    torch._check(
        index.dtype == torch.long,
        lambda: f"take(): Expected a long tensor for index, but got {index.dtype}",
    )
    # 索引检查
    torch._check_index(
        not (self.numel() == 0 and index.numel() != 0),
        lambda: "take(): tried to take from an empty tensor",
    )
    # 返回一个新的空张量，形状与索引相同
    return self.new_empty(index.shape)


@register_meta([aten.linalg_cross.default, aten.linalg_cross.out])
@out_wrapper()
def linalg_cross(self, other, *, dim=-1):
    x_d = self.ndim
    y_d = other.ndim
    # 检查输入张量的维度是否相同
    torch._check(
        x_d == y_d,
        lambda: "linalg.cross: inputs must have the same number of dimensions.",
    )
    # 检查指定维度上的长度是否为3
    torch._check(
        self.size(dim) == 3 and other.size(dim) == 3,
        lambda: (
            f"linalg.cross: inputs dimension {dim} must have length 3. "
            f"Got {self.size(dim)} and {other.size(dim)}"
        ),
    )
    # 计算输出形状并返回一个新的空张量
    out_shape = _broadcast_shapes(self.shape, other.shape)
    return self.new_empty(out_shape)


@register_meta(aten.linalg_matrix_exp)
@out_wrapper()
def linalg_matrix_exp(self):
    # 检查输入张量是否为方阵
    squareCheckInputs(self, "linalg.matrix_exp")
    # 检查输入张量是否为浮点数或复数类型
    checkFloatingOrComplex(self, "linalg.matrix_exp")
    # 返回与输入张量相同形状的新空张量，使用连续内存格式
    return torch.empty_like(self, memory_format=torch.contiguous_format)


@register_meta(
    [aten.cummax.default, aten.cummax.out, aten.cummin.default, aten.cummin.out]
)
@out_wrapper("values", "indices")
def cummaxmin(self, dim):
    # 创建用于存储值和索引的空张量
    values = torch.empty(self.shape, device=self.device, dtype=self.dtype)
    indices = torch.empty(self.shape, device=self.device, dtype=torch.int64)
    # 如果张量非空且维度不为0，检查指定维度是否合法
    if self.numel() != 0 and self.ndim != 0:
        maybe_wrap_dim(dim, self.ndim)
    # 返回值张量和索引张量
    return values, indices


@register_meta([aten.logcumsumexp.default, aten.logcumsumexp.out])
@out_wrapper()
def logcumsumexp(self, dim):
    # 检查指定维度是否合法
    maybe_wrap_dim(dim, self.ndim)
    # 返回与输入张量相同形状的新张量，使用连续内存格式
    return torch.empty_like(self).contiguous()


# Stride-related code from _exec_fft in aten/src/ATen/native/cuda/SpectralOps.cpp
def _exec_fft(out, self, out_sizes, dim, forward):
    ndim = self.ndim
    signal_ndim = len(dim)
    batch_dims = ndim - signal_ndim

    # 重新排列维度，使批处理维度位于最前，并按步长顺序排列
    dim_permute = list(range(ndim))

    is_transformed_dim = [False for _ in range(ndim)]
    for d in dim:
        is_transformed_dim[d] = True

    # std::partition
    # 遍历维度排列列表 dim_permute 中的每一个维度 d
    for d in dim_permute:
        # 如果当前维度 d 没有被转换过
        if not is_transformed_dim[d]:
            # 将未转换的维度 d 添加到 left 列表中
            left.append(d)
        else:
            # 如果当前维度 d 已经被转换过，将其添加到 right 列表中
            right.append(d)
    
    # 更新 dim_permute 为先 left 列表后 right 列表的顺序
    dim_permute = left + right
    
    # 计算 batch 维度的末尾索引
    batch_end = len(left)

    # 获取当前对象的步长信息
    self_strides = self.stride()
    
    # 提取 dim_permute 中前 batch_end 个维度到 tmp 列表中，并按照步长降序排序
    tmp = dim_permute[:batch_end]
    tmp.sort(key=lambda x: self_strides[x], reverse=True)
    
    # 将排序后的 tmp 列表与 dim_permute 中剩余部分合并
    dim_permute = tmp + dim_permute[batch_end:]
    
    # 使用新的维度排列对输入数据进行置换
    input = self.permute(dim_permute)

    # 将 batch 维度折叠成一个单独的维度
    batched_sizes = [-1] + list(input.shape[batch_dims:])
    input = input.reshape(batched_sizes)

    # 计算输入数据的 batch 大小
    batch_size = input.size(0)
    
    # 更新 batched_sizes 中的 batch 维度大小为当前输入数据的 batch 大小
    batched_sizes[0] = batch_size
    
    # 初始化 batched_out_sizes 为 batched_sizes 的拷贝
    batched_out_sizes = batched_sizes
    
    # 根据 dim 数组重新调整 batched_out_sizes 中的维度大小
    for i in range(len(dim)):
        batched_out_sizes[i + 1] = out_sizes[dim[i]]
    
    # 将输出 out 重新调整为 batched_out_sizes 的形状
    out = out.reshape(batched_out_sizes)

    # 初始化 out_strides 为长度为 ndim 的零数组
    out_strides = [0 for _ in range(ndim)]
    
    # 初始化 batch_numel 为 1，用于计算 batch 维度的元素数量
    batch_numel = 1
    
    # 遍历 batch 维度之前的维度
    i = batch_dims - 1
    while i >= 0:
        # 计算每个维度在输出张量中的步长
        out_strides[dim_permute[i]] = batch_numel * out.stride(0)
        batch_numel *= out_sizes[dim_permute[i]]
        i -= 1
    
    # 遍历 batch 维度之后的维度
    for i in range(batch_dims, ndim):
        out_strides[dim_permute[i]] = out.stride(1 + (i - batch_dims))
    
    # 使用指定的形状和步长信息创建一个新的张量 out，并返回
    return out.as_strided(out_sizes, out_strides, out.storage_offset())
# 注册元信息，用于处理 torch 的复数到复数的 FFT 操作，默认和指定输出
@register_meta([aten._fft_c2c.default, aten._fft_c2c.out])
@out_wrapper()
def meta_fft_c2c(self, dim, normalization, forward):
    # 断言输入张量的数据类型为复数
    assert self.dtype.is_complex

    # 输出张量的大小与输入张量相同
    out_sizes = self.shape
    # 创建一个空的输出张量
    output = self.new_empty(out_sizes)

    # 如果 dim 为空，直接返回空的输出张量
    if not dim:
        return output

    # 复制 dim 列表，按照输入张量各维度的步长排序
    sorted_dims = dim[:]
    self_strides = self.stride()
    sorted_dims.sort(key=lambda x: self_strides[x], reverse=True)
    # 执行 FFT 操作，将结果存储到 output 中
    output = _exec_fft(output, self, out_sizes, sorted_dims, forward)

    # 返回 FFT 结果的输出张量
    return output


# 注册元信息，用于处理 torch 的实数到复数的 FFT 操作，默认和指定输出
@register_meta([aten._fft_r2c.default, aten._fft_r2c.out])
@out_wrapper()
def meta_fft_r2c(self, dim, normalization, onesided):
    # 断言输入张量的数据类型为浮点数
    assert self.dtype.is_floating_point
    # 获取输出张量的大小列表
    output_sizes = list(self.size())

    # 如果 onesided 为 True，修改最后一个维度的大小为其一半加一
    if onesided:
        last_dim = dim[-1]
        last_dim_halfsize = (output_sizes[last_dim] // 2) + 1
        output_sizes[last_dim] = last_dim_halfsize

    # 创建一个空的输出张量，数据类型为对应的复数类型
    return self.new_empty(
        output_sizes, dtype=utils.corresponding_complex_dtype(self.dtype)
    )


# 注册元信息，用于处理生成指定大小的随机排列张量，输出结果存储在指定的输出张量 out 中
@register_meta(aten.randperm.generator_out)
def meta_randperm(n, *, generator=None, out):
    return _maybe_resize_out(out, torch.Size([n]))


# 注册元信息，用于生成指定大小的随机排列张量，默认输出结果
@register_meta(aten.randperm.default)
def meta_randperm_default(
    n,
    *,
    dtype=torch.long,
    layout=None,
    device=None,
    pin_memory=None,
):
    # 创建一个空的张量，存储生成的随机排列
    return torch.empty(
        n, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


# 注册元信息，用于生成指定大小的随机整数张量，默认和指定输出
@register_meta([aten.randint.default, aten.randint.out])
@out_wrapper()
def meta_randint(
    high,
    size,
    *,
    dtype=torch.long,
    layout=None,
    device=None,
    pin_memory=None,
):
    # 创建一个空的张量，存储生成的随机整数
    return torch.empty(
        size, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


# 注册元信息，用于生成指定范围内的随机整数张量，默认和指定输出
@register_meta([aten.randint.low, aten.randint.low_out])
@out_wrapper()
def meta_randint_low(
    low,
    high,
    size,
    *,
    dtype=torch.long,
    layout=None,
    device=None,
    pin_memory=None,
):
    # 创建一个空的张量，存储生成的随机整数
    return torch.empty(
        size, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


# 注册元信息，用于生成指定大小的随机张量，默认和指定输出
@register_meta([aten.rand.default, aten.rand.out])
@out_wrapper()
def meta_rand_default(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    # 创建一个空的张量，存储生成的随机数
    return torch.empty(
        size, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


# 注册元信息，用于处理 torch 的复数到实数的 FFT 操作，默认和指定输出
@register_meta([aten._fft_c2r.default, aten._fft_c2r.out])
@out_wrapper()
def meta_fft_c2r(self, dim, normalization, lastdim):
    # 断言输入张量的数据类型为复数
    assert self.dtype.is_complex
    # 获取输出张量的大小列表
    output_sizes = list(self.size())
    # 修改指定维度的大小为 lastdim
    output_sizes[dim[-1]] = lastdim
    # 创建一个空的输出张量，数据类型为对应的实数类型
    return self.new_empty(output_sizes, dtype=toRealValueType(self.dtype))


# 注册元信息，用于处理 torch 的张量复制操作，默认输出
@register_meta(aten.copy_.default)
def meta_copy_(self, src, non_blocking=False):
    # 这段代码模拟了从 inductor 获取的原始分解，
    # 它运行了我们关心的大多数元检查。
    # 理论上，我们应该通过仔细地实现它来使其更加健壮
    # 导入符号形状模块中的free_unbacked_symbols函数，用于分析未支持的符号
    from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

    # TODO: 理想情况下，在此处插入延迟运行时断言，但如果调用实际的复制函数，将自动触发
    # https://github.com/pytorch/pytorch/issues/122477
    # 检查当前张量是否有未支持的符号并且是否存在内存重叠情况
    if (
        not free_unbacked_symbols(self) and torch._debug_has_internal_overlap(self) == 1
    ):  # 1 == MemOverlap::Yes
        # 如果写入的张量中超过一个元素引用同一内存位置，则引发运行时错误
        raise RuntimeError(
            "more than one element of the written-to tensor refers to a single memory location"
        )

    # 如果源张量是Tensor类型，则创建一个中间张量以进行非阻塞同步
    if isinstance(src, Tensor):
        intermediate = src.to(self, non_blocking)
        # 如果当前张量的大小与中间张量的大小不匹配，则进行扩展复制操作
        if self.size() != intermediate.size():
            aten.expand_copy.default(intermediate, self.size())
    
    # 返回当前张量对象
    return self
# 推断并调整张量的几何属性，使其支持指定维度的unsqueeze操作
def inferUnsqueezeGeometry(tensor, dim):
    # 获取张量的大小列表
    result_sizes = list(tensor.size())
    # 获取张量的步幅列表
    result_strides = list(tensor.stride())
    # 计算新的步幅，如果指定的维度大于等于张量的维度，则步幅为1，否则为维度大小乘以原步幅
    new_stride = 1 if dim >= tensor.dim() else result_sizes[dim] * result_strides[dim]
    # 在结果大小列表中插入1，以支持unsqueeze操作
    result_sizes.insert(dim, 1)
    # 在结果步幅列表中插入新的步幅值
    result_strides.insert(dim, new_stride)
    # 返回调整后的大小和步幅列表
    return result_sizes, result_strides


# 注册aten.unsqueeze_的元信息，实现在指定维度进行unsqueeze操作
@register_meta(aten.unsqueeze_.default)
def meta_unsqueeze_(self, dim):
    # 根据当前张量的维度和给定的dim，确定最终的操作维度
    dim = maybe_wrap_dim(dim, self.dim() + 1)
    # 推断并调整张量的几何属性以支持unsqueeze操作
    g_sizes, g_strides = inferUnsqueezeGeometry(self, dim)
    # 将张量自身视为步幅的子集，以支持unsqueeze操作
    self.as_strided_(g_sizes, g_strides)
    # 返回操作后的张量本身
    return self


# 注册aten._sparse_semi_structured_linear的元信息，实现稀疏结构线性变换
@register_meta(aten._sparse_semi_structured_linear)
def meta_sparse_structured_linear(
    input: Tensor,
    weight: Tensor,
    _meta: Tensor,
    bias: Optional[Tensor] = None,
    _activation_opt: Optional[str] = None,
    out_dtype: Optional[torch.dtype] = None,
):
    # 复制输入张量的尺寸作为输出尺寸
    output_sizes = list(input.shape)
    # 如果存在偏置项，验证权重的第一维度与偏置的第一维度匹配
    if bias is not None:
        assert weight.size(0) == bias.size(0), "output size mismatch"
    # 验证权重的第二维度为输入的一半
    assert weight.size(1) == input.size(-1) / 2
    # 更新输出尺寸的最后一个维度为权重的第一维度
    output_sizes[-1] = weight.size(0)

    # 引用Github上的注释链接，指导处理已压缩成2D张量的输入情况
    # 输出为转置形式，需要将转置后的步幅信息传播到输出张量
    assert len(input.shape) == 2, "we can only handle the squashed input case"
    transposed_strides = (1, input.size(0))

    # 如果指定了输出数据类型，验证输入和输出数据类型的匹配性
    if out_dtype is not None:
        assert (
            input.dtype == torch.int8 and out_dtype == torch.int32
        ), "out_dtype is only supported for i8i8->i32 linear operator"
    # 创建一个新的空张量作为输出，并按指定步幅视为张量
    output = input.new_empty(
        output_sizes,
        dtype=input.dtype if out_dtype is None else out_dtype,
    ).as_strided(output_sizes, transposed_strides)

    # 返回生成的输出张量
    return output


# 注册aten._sparse_semi_structured_mm的元信息，实现稀疏结构的矩阵乘法
@register_meta(aten._sparse_semi_structured_mm)
def meta_sparse_structured_mm(
    mat1: Tensor,
    mat1_meta: Tensor,
    mat2: Tensor,
    out_dtype: Optional[torch.dtype] = None,
):
    # 验证mat1和mat2的维度为2
    assert len(mat1.shape) == 2
    assert len(mat1_meta.shape) == 2
    assert len(mat2.shape) == 2
    # 验证mat1的列数与mat2的行数的一半匹配
    assert mat1.size(1) == mat2.size(0) / 2
    # 更新输出尺寸为mat1的行数和mat2的列数
    output_sizes = [mat1.size(0), mat2.size(1)]

    # 如果指定了输出数据类型，验证输入和输出数据类型的匹配性
    if out_dtype is not None:
        assert (
            mat2.dtype == torch.int8 and out_dtype == torch.int32
        ), "out_dtype is only supported for i8i8->i32 linear operator"
    # 创建一个新的空张量作为输出
    output = mat2.new_empty(
        output_sizes,
        dtype=mat2.dtype if out_dtype is None else out_dtype,
    )

    # 返回生成的输出张量
    return output


# 注册aten._sparse_semi_structured_addmm的元信息，实现稀疏结构的加法和矩阵乘法
@register_meta(aten._sparse_semi_structured_addmm)
def meta_sparse_structured_addmm(
    input: Tensor,
    mat1: Tensor,
    mat1_meta: Tensor,
    mat2: Tensor,
    *,
    alpha=1,
    beta=1,
    out_dtype: Optional[torch.dtype] = None,
):
    # 验证input的维度为1，支持与mat1 * mat2乘积的列广播
    assert (
        len(input.shape) == 1
    ), "only input broadcasted to columns of mat1 * mat2 product is supported"
    # 验证mat1和mat2的维度为2
    assert len(mat1.shape) == 2
    assert len(mat1_meta.shape) == 2
    assert len(mat2.shape) == 2
    # 确保输入张量的第一个维度与 mat1 的第一个维度相同，以支持 mat1 * mat2 的乘积中广播到列的输入
    assert input.size(0) == mat1.size(0), "only input broadcasted to columns of mat1 * mat2 product is supported"
    
    # 确保 mat1 的第二个维度是 mat2 的第一个维度的一半，用于乘积运算
    assert mat1.size(1) == mat2.size(0) / 2
    
    # 设置输出张量的大小为 [mat1 的行数, mat2 的列数]
    output_sizes = [mat1.size(0), mat2.size(1)]

    # 如果指定了输出数据类型 out_dtype，则验证 mat2 的数据类型为 int8，out_dtype 为 int32，
    # 仅支持 i8i8->i32 的线性操作符
    if out_dtype is not None:
        assert mat2.dtype == torch.int8 and out_dtype == torch.int32, "out_dtype is only supported for i8i8->i32 linear operator"
    
    # 根据指定的输出大小和数据类型创建一个新的空张量 output，数据类型为 mat2 的数据类型（或者指定的 out_dtype）
    output = mat2.new_empty(
        output_sizes,
        dtype=mat2.dtype if out_dtype is None else out_dtype,
    )

    # 返回创建的输出张量
    return output
# 注册元数据函数，用于 _cslt_sparse_mm 操作
@register_meta(aten._cslt_sparse_mm)
def meta__cslt_sparse_mm(
    compressed_A: torch.Tensor,  # 压缩的稀疏矩阵 A，类型为 torch.Tensor
    dense_B: torch.Tensor,  # 密集矩阵 B，类型为 torch.Tensor
    bias: Optional[Tensor] = None,  # 可选参数，偏置向量，类型为 torch.Tensor 或 None
    alpha: Optional[Tensor] = None,  # 可选参数，alpha 系数，类型为 torch.Tensor 或 None
    out_dtype: Optional[torch.dtype] = None,  # 可选参数，输出数据类型，类型为 torch.dtype 或 None
    transpose_result: bool = False,  # 布尔型参数，是否转置结果，默认为 False
):
    # 断言 dense_B 的数据类型必须为 float32, float16, bfloat16 或 int8
    assert dense_B.dtype in {
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int8,
    }, "_cslt_sparse_mm only supports fp16, bf16, and int8"
    # 断言 compressed_A 和 dense_B 的数据类型必须相同
    assert compressed_A.dtype == dense_B.dtype, "inputs must have the same dtype"
    # 断言 dense_B 的维度必须为 2
    assert len(dense_B.shape) == 2, "_cslt_sparse_mm only supports 2d inputs"

    # 判断 compressed_A 的数据类型是否为 int8
    is_int8_input_type = compressed_A.dtype == torch.int8
    # 根据输入类型选择压缩因子
    compression_factor = 10 if is_int8_input_type else 9
    # 获取 dense_B 的行数 k 和列数 n
    k = dense_B.size(0)
    n = dense_B.size(1)
    # 计算压缩矩阵 A 的行数 m
    m = (compressed_A.numel() * 16) // (compression_factor * k)
    # 如果存在偏置向量，断言 m 必须与偏置向量的长度相同
    if bias is not None:
        assert m == bias.size(0)

    # 如果指定了输出数据类型 out_dtype
    if out_dtype is not None:
        # 断言输入类型为 int8，并且 out_dtype 必须为 float16, bfloat16 或 int32
        assert is_int8_input_type and out_dtype in {
            torch.float16,
            torch.bfloat16,
            torch.int32,
        }, "out_dtype is only supported for i8i8->fp16, bf16, or i32 matmul"
    
    # 根据是否需要转置结果，确定输出的形状
    output_shape = (n, m) if transpose_result else (m, n)
    # 创建一个与 dense_B 相同形状的新张量 result，数据类型为 out_dtype
    result = dense_B.new_empty(output_shape, dtype=out_dtype)
    # 返回结果张量
    return result


# 注册元数据函数，用于 index_reduce 操作
@register_meta(aten.index_reduce.default)
def meta_index_reduce(
    self: Tensor,  # 自身张量，类型为 torch.Tensor
    dim: int,  # 维度参数，指定要操作的维度，类型为 int
    index: Tensor,  # 索引张量，类型为 torch.Tensor
    source: torch.Tensor,  # 源张量，类型为 torch.Tensor
    reduce: str,  # 减少操作的类型，类型为 str
    *,
    include_self: bool = True,  # 可选参数，是否包含自身，默认为 True
) -> Tensor:  # 返回类型为 torch.Tensor
    # 返回一个与 self 张量相同形状的新空张量，使用连续内存格式
    return torch.empty_like(self, memory_format=torch.contiguous_format)


# 注册元数据函数，用于 index_reduce_ 操作
@register_meta(aten.index_reduce_.default)
def meta_index_reduce_(
    self: Tensor,  # 自身张量，类型为 torch.Tensor
    dim: int,  # 维度参数，指定要操作的维度，类型为 int
    index: Tensor,  # 索引张量，类型为 torch.Tensor
    source: torch.Tensor,  # 源张量，类型为 torch.Tensor
    reduce: str,  # 减少操作的类型，类型为 str
    *,
    include_self: bool = True,  # 可选参数，是否包含自身，默认为 True
) -> Tensor:  # 返回类型为 torch.Tensor
    # 直接返回自身张量，不做任何修改
    return self


# 注册元数据函数，用于 index_select 操作
@out_wrapper()
@register_meta(aten.index_select.default)
def meta_index_select(self, dim, index):
    # 获取结果张量的形状，初始化为 self 张量的形状列表
    result_size = list(self.size())
    # 如果 self 张量的维度大于 0
    if self.dim() > 0:
        # 更新结果张量在指定维度 dim 的长度为 index 的元素数量
        result_size[dim] = index.numel()
    # 创建一个与 self 张量相同形状的新空张量
    return self.new_empty(result_size)


# 注册元数据函数，用于 segment_reduce 操作
@register_meta(aten.segment_reduce.default)
def meta_segment_reduce(
    data: Tensor,  # 数据张量，类型为 torch.Tensor
    reduce: str,  # 减少操作的类型，类型为 str
    *,
    lengths: Optional[Tensor] = None,  # 可选参数，长度张量，类型为 torch.Tensor 或 None
    indices: Optional[Tensor] = None,  # 可选参数，索引张量，类型为 torch.Tensor 或 None
    offsets: Optional[Tensor] = None,  # 可选参数，偏移张量，类型为 torch.Tensor 或 None
    axis: int = 0,  # 轴参数，指定操作的轴，默认为 0
    unsafe: bool = False,  # 布尔型参数，是否安全操作，默认为 False
    initial=None,  # 初始值，默认为 None
) -> Tensor:  # 返回类型为 torch.Tensor
    # 如果存在索引张量 indices，则抛出未实现异常
    if indices is not None:
        raise NotImplementedError(
            "segment_reduce(): indices based reduction is not supported yet."
        )

    # 定义一个函数，根据长度张量形状创建一个新空张量
    def segment_reduce_lengths_tensor(lengths_shape):
        return torch.empty(
            lengths_shape + data.shape[axis + 1 :],
            dtype=data.dtype,
            device="meta",
            memory_format=torch.contiguous_format,
        )

    # 如果存在长度张量 lengths，则返回对应形状的空张量
    if lengths is not None:
        return segment_reduce_lengths_tensor(lengths.shape)
    # 如果提供了offsets参数，则执行以下逻辑。注意，ATen实现也忽略了长度和偏移都设置的情况。
    if offsets is not None:
        # 计算长度：lengths == torch.diff(offsets)
        lengths_shape = offsets.shape[:-1] + (offsets.shape[-1] - 1,)
        # 调用segment_reduce_lengths_tensor函数，传入计算出的长度形状
        return segment_reduce_lengths_tensor(lengths_shape)
    
    # 如果没有提供offsets参数，则抛出运行时错误
    raise RuntimeError("segment_reduce(): Either lengths or offsets must be defined.")
# 注册元信息为 aten.max.default 和 aten.max.unary_out 的装饰器
@register_meta([aten.max.default, aten.max.unary_out])
# 对输出进行包装
@out_wrapper()
# 定义函数 meta_max，返回一个空的张量，形状为 ()
def meta_max(self):
    return self.new_empty(())


# 注册元信息为 aten.max.dim 的装饰器
@register_meta(aten.max.dim)
# 定义函数 meta_max_dim，接受维度 dim 和 keepdim 标志，默认为 False
def meta_max_dim(self, dim, keepdim=False):
    # 计算需要缩减的维度，根据当前形状和给定的维度
    dim = utils.reduction_dims(self.shape, (dim,))
    # 计算缩减后的输出形状
    output_shape = _compute_reduction_shape(self, dim, keepdim)
    # 返回一个包含两个空张量的元组，形状为 output_shape，并且第二个张量的数据类型为 torch.long
    return (
        self.new_empty(output_shape),
        self.new_empty(output_shape, dtype=torch.long),
    )


# 注册元信息为 aten.min.default 和 aten.min.unary_out 的装饰器
@register_meta([aten.min.default, aten.min.unary_out])
# 对输出进行包装
@out_wrapper()
# 定义函数 meta_min，返回一个空的张量，形状为 ()
def meta_min(self):
    return self.new_empty(())


# 注册元信息为 aten.min.dim 的装饰器
@register_meta(aten.min.dim)
# 定义函数 meta_min_dim，接受维度 dim 和 keepdim 标志，默认为 False
def meta_min_dim(self, dim, keepdim=False):
    # 计算需要缩减的维度，根据当前形状和给定的维度
    dim = utils.reduction_dims(self.shape, (dim,))
    # 计算缩减后的输出形状
    output_shape = _compute_reduction_shape(self, dim, keepdim)
    # 返回一个包含两个空张量的元组，形状为 output_shape，并且第二个张量的数据类型为 torch.long
    return (
        self.new_empty(output_shape),
        self.new_empty(output_shape, dtype=torch.long),
    )


# 注册元信息为 aten.angle.default 的装饰器
@register_meta(aten.angle.default)
# 定义函数 meta_angle
def meta_angle(self):
    # 如果张量是复数类型
    if self.is_complex():
        # 使用对应的实数数据类型作为结果数据类型
        result_dtype = corresponding_real_dtype(self.dtype)
    else:
        # 否则，通过 elementwise_dtypes 函数确定结果数据类型
        _, result_dtype = elementwise_dtypes(
            self,
            type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        )
    # 返回一个与输入张量相同形状的空张量，数据类型为 result_dtype
    return torch.empty_like(self, dtype=result_dtype)


# 注册元信息为 aten.angle.out 的装饰器
@register_meta(aten.angle.out)
# 定义函数 meta_angle_out，接受一个输出张量 out
def meta_angle_out(self, out):
    # 调整输出张量 out 的大小，使其与当前张量的大小和设备相匹配
    torch._resize_output_(out, self.size(), self.device)
    # 将计算当前张量的角度，并复制到输出张量 out
    return out.copy_(torch.angle(self))


# 注册元信息为 aten._assert_async.default 的装饰器
@register_meta(aten._assert_async.default)
# 定义函数 assert_async，接受一个值 val
def assert_async(val):
    # 空函数，什么也不做
    return


# 注册元信息为 aten._assert_async.msg 的装饰器
@register_meta(aten._assert_async.msg)
# 定义函数 assert_async_meta，接受值 val 和断言消息 assert_msg
def assert_async_meta(val, assert_msg):
    # 空函数，什么也不做
    return


# 注册元信息为 aten._print.default 的装饰器
@register_meta(aten._print.default)
# 定义函数 print_meta，接受一个字符串 s
def print_meta(s):
    # 空函数，什么也不做
    return


# 注册元信息为 aten._make_dep_token.default 的装饰器
@register_meta(aten._make_dep_token.default)
# 定义函数 make_dep_token，接受一些可选参数，返回一个空张量，形状为 (0,)，设备为 "meta"
def make_dep_token(
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
):
    return torch.empty(0, device="meta")


# 注册元信息为 aten.sym_constrain_range.default 的装饰器
@register_meta(aten.sym_constrain_range.default)
# 定义函数 sym_constrain_range，接受参数 size、min 和 max
def sym_constrain_range(size, min=None, max=None):
    # 避免在模块级别导入 sympy
    from torch.fx.experimental.symbolic_shapes import constrain_range

    # 如果 size 是 SymFloat 或 SymBool 类型，则引发 ValueError
    if isinstance(size, (SymFloat, SymBool)):
        raise ValueError("Constraining SymFloat or Symbool is nyi")
    # 对给定的 size、min 和 max 进行约束处理
    constrain_range(size, min=min, max=max)


# 注册元信息为 aten._functional_sym_constrain_range.default 的装饰器
@register_meta(aten._functional_sym_constrain_range.default)
# 定义函数 functional_sym_constrain_range，接受参数 size、min、max 和 dep_token
def functional_sym_constrain_range(size, min=None, max=None, dep_token=None):
    # 调用 sym_constrain_range 函数，对 size、min 和 max 进行符号约束
    aten.sym_constrain_range(size, min=min, max=max)
    # 返回 dep_token
    return dep_token


# 注册元信息为 aten.sym_constrain_range_for_size.default 的装饰器
@register_meta(aten.sym_constrain_range_for_size.default)
# 定义函数 sym_constrain_range_for_size，接受参数 size、min 和 max
def sym_constrain_range_for_size(size, min=None, max=None):
    # 避免在模块级别导入 sympy
    from torch.fx.experimental.symbolic_shapes import _constrain_range_for_size

    # 如果 size 是 SymFloat 或 SymBool 类型，则引发 ValueError
    if isinstance(size, (SymFloat, SymBool)):
        raise ValueError("Constraining SymFloat or Symbool is nyi")
    # 对给定的 size、min 和 max 进行约束处理
    _constrain_range_for_size(size, min=min, max=max)


# 注册元信息为 aten._functional_sym_constrain_range_for_size.default 的装饰器
@register_meta(aten._functional_sym_constrain_range_for_size.default)
# 定义函数 functional_sym_constrain_range_for_size，接受参数 size、min、max 和 dep_token
def functional_sym_constrain_range_for_size(size, min, max, dep_token):
    # 使用 `aten.sym_constrain_range_for_size` 函数，对给定的 `size` 进行范围约束，限制在指定的最小值 `min` 和最大值 `max` 之间。
    aten.sym_constrain_range_for_size(size, min=min, max=max)
    # 返回 `dep_token` 变量作为函数的结果
    return dep_token
# 注册元信息，使用特定消息来注册
@register_meta(aten._functional_assert_async.msg)
def functional_assert_async_meta(val, assert_msg, dep_token):
    # 返回依赖令牌
    return dep_token


# 检查输入张量是否为方阵
# From aten/src/ATen/native/LinearAlgebraUtils.h
def squareCheckInputs(self: Tensor, f_name: str):
    # 断言输入张量至少有两个维度
    assert (
        self.dim() >= 2
    ), f"{f_name}: The input tensor must have at least 2 dimensions."
    # 断言输入张量的最后两个维度大小相等，即为方阵
    assert self.size(-1) == self.size(
        -2
    ), f"{f_name}: A must be batches of square matrices, but they are {self.size(-2)} by {self.size(-1)} matrices"


# 验证线性求解方法（solve, cholesky_solve, lu_solve, triangular_solve）的输入形状和设备
# From aten/src/ATen/native/LinearAlgebraUtils.h
def linearSolveCheckInputs(
    self: Tensor,
    A: Tensor,
    name: str,
):
    # 检查张量 self 和 A 是否在同一设备上
    torch._check(
        self.device == A.device,
        lambda: (
            f"Expected b and A to be on the same device, but found b on "
            f"{self.device} and A on {A.device} instead."
        ),
    )

    # 检查张量 self 和 A 是否具有相同的数据类型
    torch._check(
        self.dtype == A.dtype,
        lambda: (
            f"Expected b and A to have the same dtype, but found b of type "
            f"{self.dtype} and A of type {A.dtype} instead."
        ),
    )

    # 检查张量 A 是否为方阵
    torch._check(
        A.size(-1) == A.size(-2),
        lambda: (
            f"A must be batches of square matrices, "
            f"but they are {A.size(-2)} by {A.size(-1)} matrices"
        ),
    )

    # 检查张量 A 的最后两个维度与张量 self 的维度是否兼容
    torch._check(
        A.size(-1) == self.size(-2),
        lambda: (
            f"Incompatible matrix sizes for {name}: each A "
            f"matrix is {A.size(-1)} by {A.size(-1)}"
            f" but each b matrix is {self.size(-2)} by {self.size(-1)}"
        ),
    )


# 检查张量是否为浮点数或复数类型
# From aten/src/ATen/native/LinearAlgebraUtils.h
def checkFloatingOrComplex(
    t: Tensor,
    f_name: str,
    allow_low_precision_dtypes: bool = True,
):
    dtype = t.dtype
    # 检查张量是否为浮点数或复数类型
    torch._check(
        t.is_floating_point() or t.is_complex(),
        lambda: f"{f_name}: Expected a floating point or complex tensor as input. Got {dtype}",
    )
    # 如果不允许低精度数据类型，则进一步检查
    if not allow_low_precision_dtypes:
        torch._check(
            dtype in (torch.float, torch.double, torch.cfloat, torch.cdouble),
            lambda: f"{f_name}: Low precision dtypes not supported. Got {dtype}",
        )


# 检查张量是否为矩阵
# From aten/src/ATen/native/LinearAlgebraUtils.h
def checkIsMatrix(A: Tensor, f_name: str, arg_name: str = "A"):
    # 检查张量 A 的维度是否至少为 2
    torch._check(
        A.dim() >= 2,
        lambda: f"{f_name}: The input tensor {arg_name} must have at least 2 dimensions.",
    )


# 检查线性求解方法的输入张量是否兼容
def checkInputsSolver(
    A: Tensor,
    B: Tensor,
    left: bool,
    f_name: str,
):
    # 检查 A 是否为方阵
    squareCheckInputs(A, f_name)
    # 检查 B 是否为矩阵
    checkIsMatrix(B, f_name)
    # 根据左右乘关系检查 A 和 B 的形状是否兼容
    torch._check(
        A.size(-2) == B.size(-2) if left else A.size(-1) == B.size(-1),
        lambda: (
            f"{f_name}: Incompatible shapes of A and B for the equation "
            f"{'AX = B' if left else 'XA = B'}"
            f" ({A.size(-2)}x{A.size(-1)} and {B.size(-2)}x{B.size(-1)})"
        ),
    )
# 检查函数，确保结果张量和输入张量在同一设备上
def checkSameDevice(
    fn_name: str,
    result: Tensor,
    input: Tensor,
    result_name: str = "result",
):
    torch._check(
        result.device == input.device,
        lambda: (
            f"{fn_name}: Expected {result_name} and input tensors to be on the same device, but got "
            f"{result_name} on {result.device} and input on {input.device}"
        ),
    )

# 检查 UPLO 参数是否为单个字符且为 'L' 或 'U'
def checkUplo(UPLO: str):
    UPLO_uppercase = UPLO.upper()
    torch._check(
        len(UPLO) == 1 and (UPLO_uppercase == "U" or UPLO_uppercase == "L"),
        lambda: f"Expected UPLO argument to be 'L' or 'U', but got {UPLO}",
    )

# 注册函数的元数据装饰器，适用于 _linalg_eigh.default 和 _linalg_eigh.eigenvalues
@out_wrapper("eigenvalues", "eigenvectors")
def meta__linalg_eigh(
    A: Tensor,
    UPLO: str = "L",
    compute_v: bool = True,
):
    # 检查 A 是否为方阵
    squareCheckInputs(A, "linalg.eigh")
    # 检查 UPLO 参数
    checkUplo(UPLO)

    # 创建与 A 相同形状的空张量 vecs
    shape = list(A.shape)
    if compute_v:
        vecs = A.new_empty(shape)
        vecs.as_strided_(shape, make_contiguous_strides_for(shape, row_major=False))
    else:
        vecs = A.new_empty([0])

    # 创建一个比 A 少一维的空张量 vals，类型为 A 的实数值类型
    shape.pop()
    vals = A.new_empty(shape, dtype=toRealValueType(A.dtype))

    return vals, vecs

# 注册函数的元数据装饰器，适用于 aten._linalg_eigvals.default 和 aten.linalg_eigvals.out
@out_wrapper()
def meta__linalg_eigvals(input: Tensor) -> Tensor:
    # 检查 input 是否为方阵
    squareCheckInputs(input, "linalg.eigvals")
    # 确定复数类型，与 input 的数据类型相关
    complex_dtype = (
        input.dtype
        if utils.is_complex_dtype(input.dtype)
        else utils.corresponding_complex_dtype(input.dtype)
    )
    # 创建一个与 input 形状的前 n-1 维相同的空张量，类型为 complex_dtype
    return input.new_empty(input.shape[:-1], dtype=complex_dtype)

# 注册函数的元数据装饰器，适用于 aten.linalg_eig
@out_wrapper("eigenvalues", "eigenvectors")
def meta_linalg_eig(input: Tensor):
    # 检查 input 是否为方阵
    squareCheckInputs(input, "linalg.eig")
    # 确定复数类型，与 input 的数据类型相关
    complex_dtype = (
        input.dtype
        if utils.is_complex_dtype(input.dtype)
        else utils.corresponding_complex_dtype(input.dtype)
    )
    # 创建一个与 input 形状的前 n-1 维相同的空张量 values 和与 input 相同形状的空张量 vectors，类型为 complex_dtype
    values = input.new_empty(input.shape[:-1], dtype=complex_dtype)
    vectors = input.new_empty(input.shape, dtype=complex_dtype)
    return values, vectors

# 复制批处理列主格式的张量，并进行转置
def cloneBatchedColumnMajor(src: Tensor) -> Tensor:
    return src.mT.clone(memory_format=torch.contiguous_format).transpose(-2, -1)

# 注册函数的元数据装饰器，适用于 aten._cholesky_solve_helper
@out_wrapper()
def _cholesky_solve_helper(self: Tensor, A: Tensor, upper: bool) -> Tensor:
    # 调用 cloneBatchedColumnMajor 函数复制并转置输入张量 self
    return cloneBatchedColumnMajor(self)

# 注册函数的元数据装饰器，适用于 aten.cholesky_solve
@out_wrapper()
def cholesky_solve(self: Tensor, A: Tensor, upper: bool = False) -> Tensor:
    # 检查 self 和 A 张量是否至少有两个维度
    torch._check(
        self.ndim >= 2,
        lambda: f"b should have at least 2 dimensions, but has {self.ndim} dimensions instead",
    )
    torch._check(
        A.ndim >= 2,
        lambda: f"u should have at least 2 dimensions, but has {A.ndim} dimensions instead",
    )
    # 对 self 和 A 进行广播处理，并调用 _cholesky_solve_helper 函数
    self_broadcasted, A_broadcasted = _linalg_broadcast_batch_dims_name(
        self, A, "cholesky_solve"
    )
    return _cholesky_solve_helper(self_broadcasted, A_broadcasted, upper)
# 注册 cholesky 函数为 Torch 操作的元信息
@register_meta(aten.cholesky)
# 应用输出包装器装饰器，用于处理输出参数
@out_wrapper()
# 定义 cholesky 函数，以当前 Tensor 作为 self，可选参数 upper 默认为 False，返回值为 Tensor
def cholesky(self: Tensor, upper: bool = False) -> Tensor:
    # 如果 self 的元素个数为 0，则返回一个与 self 相同大小的空 Tensor
    if self.numel() == 0:
        return torch.empty_like(self, memory_format=torch.legacy_contiguous_format)
    # 对输入进行方阵检查，用于 cholesky 分解
    squareCheckInputs(self, "cholesky")
    # 克隆批次主列优先的 Tensor，并返回
    return cloneBatchedColumnMajor(self)


# 注册 cholesky_inverse 函数为 Torch 操作的元信息
@register_meta(aten.cholesky_inverse)
# 应用输出包装器装饰器，用于处理输出参数
@out_wrapper()
# 定义 cholesky_inverse 函数，以当前 Tensor 作为 self，可选参数 upper 默认为 False，返回值为 Tensor
def cholesky_inverse(self: Tensor, upper: bool = False) -> Tensor:
    # 对输入进行方阵检查，用于 cholesky 逆矩阵计算
    squareCheckInputs(self, "cholesky_inverse")
    # 返回克隆批次主列优先的 Tensor
    return cloneBatchedColumnMajor(self)


# 从 aten/src/ATen/native/BatchLinearAlgebra.cpp 导入的 linalg_cholesky_ex 函数
@register_meta(aten.linalg_cholesky_ex.default)
# 定义 linalg_cholesky_ex 函数，接受输入参数 A（Tensor）、upper（bool，默认为 False）、check_errors（bool，默认为 False）
def linalg_cholesky_ex(A: Tensor, upper: bool = False, check_errors: bool = False):
    # 对输入 A 进行方阵检查，用于 linalg cholesky 操作
    squareCheckInputs(A, "linalg.cholesky")
    # 检查 A 的数据类型必须为浮点数或复数
    checkFloatingOrComplex(A, "linalg.cholesky")

    # 获取输入 Tensor A 的形状信息
    A_shape = A.shape
    ndim = len(A_shape)

    # 创建空的 Tensor L，形状与 A 相同，并通过 make_contiguous_strides_for 函数生成连续的步幅信息
    L_strides = make_contiguous_strides_for(A_shape, False)
    L = A.new_empty(A_shape)
    L.as_strided_(A_shape, L_strides)

    # 创建与 A 形状的 infos Tensor，数据类型为 torch.int32
    infos = A.new_empty(A_shape[0 : ndim - 2], dtype=torch.int32)
    # 返回 Tensor L 和 infos
    return L, infos


# 注册 linalg_householder_product 函数为 Torch 操作的元信息，支持默认和输出的两种注册方式
@register_meta(
    [aten.linalg_householder_product.default, aten.linalg_householder_product.out]
)
# 应用输出包装器装饰器，用于处理输出参数
@out_wrapper()
# 定义 linalg_householder_product 函数，接受输入参数 input（Tensor）、tau（Tensor），返回值为 Tensor
def linalg_householder_product(input: Tensor, tau: Tensor) -> Tensor:
    # 检查输入 Tensor input 的维度必须至少为 2
    torch._check(
        input.ndim >= 2,
        lambda: "torch.linalg.householder_product: input must have at least 2 dimensions.",
    )
    # 检查 input 的倒数第二个维度必须大于等于最后一个维度
    torch._check(
        input.size(-2) >= input.size(-1),
        lambda: "torch.linalg.householder_product: input.shape[-2] must be greater than or equal to input.shape[-1]",
    )
    # 检查 input 的最后一个维度必须大于等于 tau 的最后一个维度
    torch._check(
        input.size(-1) >= tau.size(-1),
        lambda: "torch.linalg.householder_product: input.shape[-1] must be greater than or equal to tau.shape[-1]",
    )
    # 检查 input 的维度减去 tau 的维度必须为 1
    torch._check(
        input.ndim - tau.ndim == 1,
        lambda: (
            f"torch.linalg.householder_product: Expected tau to have one dimension less than input, "
            f"but got tau.ndim equal to {tau.ndim} and input.ndim is equal to {input.ndim}"
        ),
    )
    # 如果 input 的维度大于 2，检查 tau 的批处理维度必须与 input 的前两个维度相同
    if input.ndim > 2:
        expected_batch_tau_shape = input.shape[:-2]
        actual_batch_tau_shape = tau.shape[:-1]
        torch._check(
            actual_batch_tau_shape == expected_batch_tau_shape,
            lambda: (
                f"torch.linalg.householder_product: Expected batch dimensions of tau to be "
                f"equal to input.shape[:-2], but got {actual_batch_tau_shape}"
            ),
        )
    # 检查 tau 的数据类型必须与 input 相同
    torch._check(
        tau.dtype == input.dtype,
        lambda: (
            f"torch.linalg.householder_product: tau dtype {tau.dtype}"
            f" does not match input dtype {input.dtype}"
        ),
    )
    # 检查 tau 和 input 必须在相同的设备上
    checkSameDevice("torch.linalg.householder_product", tau, input, "tau")

    # 返回一个与 input 形状相同的空 Tensor，使用非行主排列的连续步幅方式
    return torch.empty_strided(
        size=input.shape,
        stride=make_contiguous_strides_for(input.shape, row_major=False),
        dtype=input.dtype,
        device=input.device,
    )
# From aten/src/ATen/native/BatchLinearAlgebra.cpp

# 注册 linalg_inv_ex.default 元信息的函数装饰器
@register_meta(aten.linalg_inv_ex.default)
def linalg_inv_ex_meta(A: Tensor, check_errors: bool = False):
    # 对输入张量 A 进行方阵检查，用于 linalg.inv_ex 操作
    squareCheckInputs(A, "linalg.inv_ex")
    # 检查 A 是否为浮点数或复数类型，不允许使用低精度数据类型
    checkFloatingOrComplex(A, "linalg.inv_ex", allow_low_precision_dtypes=False)

    # 创建一个与 A 相同形状的新空张量 L
    L = A.new_empty(A.shape)
    # 将张量 L 重设为 A 的形状，并使其具有行优先的连续步长
    L.as_strided_(A.shape, make_contiguous_strides_for(A.shape, row_major=False))

    # 创建一个与 A 的前两维形状相同的空张量 infos，数据类型为 torch.int32
    infos = A.new_empty(A.shape[:-2], dtype=torch.int32)
    # 返回张量 L 和 infos
    return L, infos


# 注册 linalg_ldl_factor_ex.default 和 linalg_ldl_factor_ex.out 元信息的函数装饰器
@register_meta([aten.linalg_ldl_factor_ex.default, aten.linalg_ldl_factor_ex.out])
@out_wrapper("LD", "pivots", "info")
def linalg_ldl_factor_ex_meta(
    self: Tensor,
    *,
    hermitian: bool = False,
    check_errors: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    # 对输入张量 self 进行方阵检查，用于 torch.linalg.ldl_factor_ex 操作
    squareCheckInputs(self, "torch.linalg.ldl_factor_ex")
    # 检查 self 是否为浮点数或复数类型，用于 torch.linalg.ldl_factor_ex 操作
    checkFloatingOrComplex(self, "torch.linalg.ldl_factor_ex")

    # 创建一个与 self 相同形状的空张量 LD，数据类型与 self 相同，位于相同设备上
    LD = torch.empty_strided(
        size=self.shape,
        stride=make_contiguous_strides_for(self.shape, row_major=False),
        dtype=self.dtype,
        device=self.device,
    )
    # 创建一个与 self 的最后一维形状相同的空张量 pivots，数据类型为 torch.int
    pivots = self.new_empty(self.shape[:-1], dtype=torch.int)
    # 创建一个与 self 的前两维形状相同的空张量 info，数据类型为 torch.int
    info = self.new_empty(self.shape[:-2], dtype=torch.int)
    # 返回张量 LD、pivots 和 info
    return LD, pivots, info


# 注册 linalg_ldl_solve.default 和 linalg_ldl_solve.out 元信息的函数装饰器
@register_meta([aten.linalg_ldl_solve.default, aten.linalg_ldl_solve.out])
@out_wrapper()
def linalg_ldl_solve_meta(
    LD: Tensor,
    pivots: Tensor,
    B: Tensor,
    *,
    hermitian: bool = False,
) -> Tensor:
    # 对输入张量 LD 进行方阵检查，用于 torch.linalg.ldl_solve 操作
    squareCheckInputs(LD, "torch.linalg.ldl_solve")
    # 检查 LD 是否为浮点数或复数类型，用于 torch.linalg.ldl_solve 操作
    checkFloatingOrComplex(LD, "torch.linalg.ldl_solve")
    # 检查 LD 和 B 的形状匹配，用于 torch.linalg.ldl_solve 操作
    linearSolveCheckInputs(B, LD, "torch.linalg.ldl_solve")
    # 检查 B 的维度至少为 2
    torch._check(
        B.ndim >= 2,
        lambda: (
            f"torch.linalg.ldl_solve: Expected B to have at least 2 dimensions, "
            f"but it has {B.ndim} dimensions instead"
        ),
    )
    # 检查 pivots 的形状与 LD 的前两维形状相同
    expected_pivots_shape = LD.shape[:-1]
    torch._check(
        expected_pivots_shape == pivots.shape,
        lambda: (
            f"torch.linalg.ldl_solve: Expected LD.shape[:-1] and pivots.shape to be the same, "
            f"but got pivots with shape {pivots.shape} instead"
        ),
    )
    # 检查 pivots 的数据类型为整数类型
    torch._check(
        utils.is_integer_dtype(pivots.dtype),
        lambda: f"torch.linalg.ldl_solve: Expected pivots to be integers. Got {pivots.dtype}",
    )
    # 检查 LD 的数据类型与 B 的数据类型相同
    torch._check(
        LD.dtype == B.dtype,
        lambda: f"torch.linalg.ldl_solve: LD dtype {LD.dtype} does not match b dtype {B.dtype}",
    )
    
    # 获取广播后的 B 的大小，并创建一个相同大小的空张量，数据类型与 B 相同，位于相同设备上
    B_broadcast_size, _ = _linalg_broadcast_batch_dims(B, LD)
    return torch.empty_strided(
        size=B_broadcast_size,
        stride=make_contiguous_strides_for(B_broadcast_size, row_major=False),
        dtype=B.dtype,
        device=B.device,
    )


# 注册 linalg_lu.default 和 linalg_lu.out 元信息的函数装饰器
@register_meta([aten.linalg_lu.default, aten.linalg_lu.out])
@out_wrapper("P", "L", "U")
def linalg_lu_meta(A: Tensor, *, pivot: bool = True) -> Tuple[Tensor, Tensor, Tensor]:
    # 检查输入张量 A 的维度至少为 2
    torch._check(
        A.ndim >= 2,
        lambda: f"linalg.lu: Expected tensor with 2 or more dimensions. Got size: {A.shape} instead",
    )

    # 获取矩阵 A 的形状信息，并分别赋值给 m、n
    sizes = list(A.shape)
    m = sizes[-2]  # 矩阵 A 的倒数第二维大小
    n = sizes[-1]  # 矩阵 A 的最后一维大小
    k = min(m, n)  # 取 m 和 n 中较小的作为 k

    sizes[-1] = m  # 将 sizes 列表中最后一位（即矩阵 A 的最后一维大小）设置为 m
    # 如果 pivot 为真，创建一个与 A 形状相同的空数组 P，否则创建一个空数组 [0]
    if pivot:
        P = A.new_empty(sizes)
    else:
        P = A.new_empty([0])

    sizes[-1] = k  # 将 sizes 列表中最后一位设置为 k
    # 创建一个与 A 形状相同的空数组 L
    L = A.new_empty(sizes)

    sizes[-2] = k  # 将 sizes 列表中倒数第二位设置为 k
    sizes[-1] = n  # 将 sizes 列表中最后一位设置为 n
    # 创建一个与 A 形状相同的空数组 U
    U = A.new_empty(sizes)
    # 返回数组 P、L、U 作为结果
    return P, L, U
# 注册元信息，指定适用的操作和输出包装器函数
@register_meta([aten.linalg_lu_factor_ex.default, aten.linalg_lu_factor_ex.out])
@out_wrapper("LU", "pivots", "info")
# LU 分解元信息函数，返回 LU 分解后的结果：LU 矩阵，主元（pivots），信息（info）
def linalg_lu_factor_ex_meta(
    A: Tensor,
    *,
    pivot: bool = True,
    check_errors: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    # 检查输入张量 A 的维度是否大于等于 2
    torch._check(
        A.ndim >= 2,
        lambda: f"torch.lu_factor: Expected tensor with 2 or more dimensions. Got size: {A.shape} instead",
    )

    # 获取张量 A 的尺寸
    sizes = list(A.shape)
    m = sizes[-2]
    n = sizes[-1]

    # 创建一个空的、按步长连续的张量 LU，与 A 的数据类型和设备相同
    LU = torch.empty_strided(
        size=sizes,
        stride=make_contiguous_strides_for(sizes, row_major=False),
        dtype=A.dtype,
        device=A.device,
    )

    # 设置 sizes 为主元 pivots 的尺寸
    sizes.pop()
    sizes[-1] = min(m, n)
    pivots = A.new_empty(sizes, dtype=torch.int)

    # 设置 sizes 为信息 info 的尺寸
    sizes.pop()
    info = A.new_empty(sizes, dtype=torch.int)

    # 返回 LU 分解后的结果：LU 矩阵，主元（pivots），信息（info）
    return LU, pivots, info


# 注册元信息，指定适用的操作和输出包装器函数
@register_meta([aten.linalg_lu_solve.default, aten.linalg_lu_solve.out])
@out_wrapper()
# LU 求解元信息函数，返回解 B
def linalg_lu_solve_meta(
    LU: Tensor,
    pivots: Tensor,
    B: Tensor,
    *,
    left: bool = True,
    adjoint: bool = False,
) -> Tensor:
    # 检查 LU 和 B 张量的数据类型
    checkFloatingOrComplex(LU, "torch.linalg.lu_solve")
    # 检查 LU 和 B 的数据类型是否一致
    torch._check(
        LU.dtype == B.dtype,
        lambda: (
            f"linalg.lu_solve: Expected LU and B to have the same dtype, "
            f"but found LU of type {LU.dtype} and B of type {B.dtype} instead"
        ),
    )
    # 检查主元 pivots 的数据类型
    torch._check(
        pivots.dtype == torch.int,
        lambda: "linalg.lu_solve: pivots should be a Tensor of scalar type torch.int32",
    )

    # 检查输入张量 LU 和 B 的形状是否适合求解
    squareCheckInputs(LU, "torch.linalg.lu_solve")
    checkInputsSolver(LU, B, left, "linalg.lu_solve")
    # 检查每个批次的主元数是否与矩阵的维度相同
    torch._check(
        LU.size(-1) == pivots.size(-1),
        lambda: "linalg.lu_solve: Number of pivots per batch should be same as the dimension of the matrix",
    )

    # 检查 LU 和 pivots 的批次形状是否匹配
    torch._check(
        LU.shape[:-1] == pivots.shape,
        lambda: (
            f"linalg.lu_solve: Expected LU.shape[:-1] and pivots.shape to be the same, "
            f"but got pivots with shape {pivots.shape} instead"
        ),
    )

    # 计算广播后的张量 B 的尺寸
    B_broadcast_size, _ = _linalg_broadcast_batch_dims(B, LU)

    # 创建一个空的、按步长连续的张量 result，与 B 的数据类型和设备相同
    result = torch.empty_strided(
        size=B_broadcast_size,
        stride=make_contiguous_strides_for(B_broadcast_size, row_major=not left),
        dtype=B.dtype,
        device=B.device,
    )

    # 如果 result 非空且左求解，且结果是复数，则求共轭
    if result.numel() != 0 and not left:
        if result.is_complex():
            result = result.conj()

    # 返回求解结果张量 result
    return result


# 注册元信息，指定适用的操作和输出包装器函数
@register_meta(aten.lu_unpack)
@out_wrapper("P", "L", "U")
# LU 解包元信息函数，返回 P、L、U 三个张量元组
def lu_unpack_meta(
    LU: Tensor,
    pivots: Tensor,
    unpack_data: bool = True,
    unpack_pivots: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    # 检查输入张量 LU 的维度是否大于等于 2
    torch._check(
        LU.ndim >= 2,
        lambda: f"torch.lu_unpack: Expected tensor with 2 or more dimensions. Got size: {LU.shape} instead",
    )
    # 如果需要解压 LU 因子，则进行以下操作
    if unpack_pivots:
        # 检查 LU_pivots 张量的数据类型是否为 torch.int32
        torch._check(
            pivots.dtype == torch.int32,
            lambda: (
                "torch.lu_unpack: LU_pivots is expected to be a contiguous tensor of torch.int32 dtype.\n"
                "Note: this function is intended to be used with the output produced by torch.linalg.lu_factor"
            ),
        )
    
    # 获取 LU 矩阵的维度大小
    sizes = list(LU.shape)
    m = sizes[-2]  # 获取矩阵的行数
    n = sizes[-1]  # 获取矩阵的列数
    k = min(m, n)  # 计算 LU 分解的秩
    sizes[-1] = m  # 更新大小列表中最后一个维度的值为 m
    
    # 如果需要解压 LU 因子，则创建相应大小的 P 矩阵；否则创建大小为 [0] 的 P 矩阵
    if unpack_pivots:
        P = LU.new_empty(sizes)
    else:
        P = LU.new_empty([0])
    
    # 如果需要解压 LU 数据，则创建相应大小的 L 和 U 矩阵
    if unpack_data:
        sizes[-1] = k  # 更新大小列表中最后一个维度的值为 k
        L = LU.new_empty(sizes)
        sizes[-2] = k  # 更新大小列表中倒数第二个维度的值为 k
        sizes[-1] = n  # 恢复大小列表中最后一个维度的值为 n
        U = LU.new_empty(sizes)
    else:
        L = LU.new_empty([0])  # 创建大小为 [0] 的 L 矩阵
        U = LU.new_empty([0])  # 创建大小为 [0] 的 U 矩阵
    
    # 返回创建的 P, L, U 矩阵
    return P, L, U
# 解析 linalg_qr 函数的 mode 参数，返回一个布尔值元组 (compute_q, reduced)
def _parse_qr_mode(mode: str) -> Tuple[bool, bool]:
    # 如果 mode 是 "reduced"，则计算 Q，并使用 reduced 模式
    if mode == "reduced":
        compute_q = True
        reduced = True
    # 如果 mode 是 "complete"，则计算 Q，但不使用 reduced 模式
    elif mode == "complete":
        compute_q = True
        reduced = False
    # 如果 mode 是 "r"，则不计算 Q，reduced 模式在此情况下无关紧要
    elif mode == "r":
        compute_q = False
        reduced = True  # 实际上在这种模式下这个值是无关紧要的
    else:
        # 如果 mode 是其他未知值，则引发错误并显示预期的有效 mode 值
        torch._check(
            False,
            lambda: (
                f"qr received unrecognized mode '{mode}' "
                f"but expected one of 'reduced' (default), 'r', or 'complete'"
            ),
        )
    return compute_q, reduced  # type: ignore[possibly-undefined]


# 注册 linalg_qr_meta 函数为 aten.linalg_qr.default 和 aten.linalg_qr.out 的元信息
@register_meta([aten.linalg_qr.default, aten.linalg_qr.out])
# 用于封装输出的装饰器，输出结果包括 "Q" 和 "R"
@out_wrapper("Q", "R")
# 计算 QR 分解的元信息函数
def linalg_qr_meta(
    A: Tensor,
    mode: str = "reduced",
) -> Tuple[Tensor, Tensor]:
    # 检查输入张量 A 是否是矩阵，用于 linalg.qr 函数
    checkIsMatrix(A, "linalg.qr")
    # 检查输入张量 A 是否是浮点数或复数类型，用于 linalg.qr 函数
    checkFloatingOrComplex(A, "linalg.qr")

    # 解析 QR 分解的 mode 参数，得到是否计算 Q 和使用的 reduced 模式
    compute_q, reduced_mode = _parse_qr_mode(mode)

    m = A.shape[-2]  # 获取矩阵 A 的行数
    n = A.shape[-1]  # 获取矩阵 A 的列数
    k = min(m, n)    # 计算用于 QR 分解的较小维度

    if compute_q:
        # 如果需要计算 Q，则创建相应形状的空张量 Q
        Q_shape = list(A.shape)
        Q_shape[-1] = k if reduced_mode else m
        Q = A.new_empty(Q_shape)
        Q.as_strided_(Q_shape, make_contiguous_strides_for(Q_shape, row_major=False))
    else:
        # 如果不需要计算 Q，则创建空的零维张量 Q
        Q = A.new_empty([0])

    # 创建相应形状的空张量 R，用于存储 QR 分解的 R 矩阵
    R_shape = list(A.shape)
    R_shape[-2] = k if reduced_mode or not compute_q else m
    R = A.new_empty(R_shape)
    R.as_strided_(R_shape, make_contiguous_strides_for(R_shape, row_major=False))
    return Q, R


# 注册 _linalg_slogdet 函数为 aten._linalg_slogdet.default 和 aten._linalg_slogdet.sign 的元信息
@register_meta([aten._linalg_slogdet.default, aten._linalg_slogdet.sign])
# 用于封装输出的装饰器，输出结果包括 "sign", "logabsdet", "LU", "pivots"
@out_wrapper("sign", "logabsdet", "LU", "pivots")
# 计算行列式的对数和符号的元信息函数
def _linalg_slogdet(A: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # 检查输入张量 A 是否是方阵，用于 linalg.slogdet 函数
    squareCheckInputs(A, "linalg.slogdet")
    # 检查输入张量 A 是否是浮点数或复数类型，用于 linalg.slogdet 函数，不允许复数
    checkFloatingOrComplex(A, "linalg.slogdet", False)
    shape = A.shape
    # 创建空的张量用于存储符号 sign
    sign = A.new_empty(shape[:-2])
    # 创建空的张量用于存储行列式的对数的绝对值 logabsdet
    logabsdet = A.new_empty(shape[:-2], dtype=toRealValueType(A.dtype))
    # 创建空的张量用于存储 LU 分解的结果 LU
    LU = torch.empty_strided(
        size=shape,
        stride=make_contiguous_strides_for(shape, False),
        dtype=A.dtype,
        device=A.device,
    )
    # 创建空的张量用于存储 LU 分解的主元索引 pivots
    pivots = A.new_empty(shape[:-1], dtype=torch.int32)
    return sign, logabsdet, LU, pivots


# 从 aten/src/ATen/native/BatchLinearAlgebra.cpp 文件中获取函数的默认信息
# 注意：与 aten/src/ATen/native/native_functions.yaml 中的默认值匹配
@register_meta(aten._linalg_svd.default)
# 计算奇异值分解的元信息函数
def _linalg_svd_meta(
    A: Tensor,
    full_matrices: bool = False,
    compute_uv: bool = True,
    driver: Optional[str] = None,
):
    # 检查输入张量 A 是否是矩阵，用于 linalg.svd 函数
    checkIsMatrix(A, "linalg.svd")
    # 检查输入张量 A 是否是浮点数或复数类型，用于 linalg.svd 函数
    checkFloatingOrComplex(A, "linalg.svd")

    batch_dims = list(A.shape[:-2])  # 获取批处理维度
    m = A.shape[-2]  # 获取矩阵 A 的行数
    n = A.shape[-1]  # 获取矩阵 A 的列数
    k = min(m, n)    # 计算用于奇异值分解的较小维度
    # 如果需要计算 U 和 V
    if compute_uv:
        # 设置 U 的形状，根据是否需要完整矩阵决定最后两维的大小
        U_shape = batch_dims + [m, m if full_matrices else k]
        # 创建一个空的 U 张量
        U = A.new_empty(U_shape)
        # 使用 as_strided_ 方法设置 U 张量的形状和步长，确保是按行主序（row_major=False）
        U.as_strided_(U_shape, make_contiguous_strides_for(U_shape, row_major=False))

        # 设置 V 的形状，根据是否需要完整矩阵决定前两维的大小
        V_shape = batch_dims + [n if full_matrices else k, n]
        # 创建一个空的 V 张量
        V = A.new_empty(V_shape)
        
        # 注意：以下注释解释了这段代码的一些技术细节和条件
        # NB: 这里检查是否在 CUDA 上运行，因为没有办法检查 cuSolver 的存在。
        # 在 CPU 上可能无法正常工作，当 fake_device 不可用时，默认设备提示为 CUDA。
        # 参见 core 中 _linalg_svd 元信息。
        is_cuda = device_hint(A) == "cuda"
        # 使用 as_strided_ 方法设置 V 张量的形状和步长，根据是否在 CUDA 上决定行主序
        V.as_strided_(V_shape, make_contiguous_strides_for(V_shape, row_major=is_cuda))
    else:
        # 如果不需要计算 U 和 V，则创建空的 U 和 V 张量
        # 不过这段代码看起来似乎永远不会执行，因为函数末尾总是返回 U, S, V
        U = A.new_empty([0])
        V = A.new_empty([0])

    # 创建一个空的 S 张量，形状为 batch_dims + [k]
    # S 总是实数，即使 A 是复数
    S = A.new_empty(batch_dims + [k], dtype=toRealValueType(A.dtype))
    # 返回计算得到的 U, S, V 张量
    return U, S, V
def _linalg_broadcast_batch_dims(
    arg1: Tensor,
    arg2: Tensor,
) -> Tuple[List[int], List[int]]:
    # 广播 arg1 和 arg2 的批次维度。

    # 获取 arg1 的批次维度，不包括最后两个维度（通常是行和列）
    arg1_batch_sizes = arg1.shape[:-2]
    # 获取 arg2 的批次维度，不包括最后两个维度
    arg2_batch_sizes = arg2.shape[:-2]

    # 调用函数 _broadcast_shapes 广播 arg1 和 arg2 的批次部分
    expand_batch_portion = _broadcast_shapes(arg1_batch_sizes, arg2_batch_sizes)

    # 构建扩展后的 arg1 的尺寸
    arg1_expand_size = list(expand_batch_portion)
    arg1_expand_size += [arg1.size(-2), arg1.size(-1)]

    # 构建扩展后的 arg2 的尺寸
    arg2_expand_size = list(expand_batch_portion)
    arg2_expand_size += [arg2.size(-2), arg2.size(-1)]

    # 返回扩展后的尺寸信息
    return arg1_expand_size, arg2_expand_size


def _linalg_broadcast_batch_dims_name(
    arg1: Tensor,
    arg2: Tensor,
    name: Optional[str],
) -> Tuple[Tensor, Tensor]:
    # 如果没有提供名称，假设不需要检查错误
    if name:
        # 调用 linearSolveCheckInputs 函数检查输入参数 arg1 和 arg2
        linearSolveCheckInputs(arg1, arg2, name)

    # 获取扩展后的 arg1 和 arg2 的尺寸
    arg1_expand_size, arg2_expand_size = _linalg_broadcast_batch_dims(arg1, arg2)

    # 如果扩展后的尺寸与原始尺寸相同，则直接使用原始张量；否则使用 expand 扩展
    arg1_broadcasted = (
        arg1 if arg1_expand_size == arg1.shape else arg1.expand(arg1_expand_size)
    )
    arg2_broadcasted = (
        arg2 if arg2_expand_size == arg2.shape else arg2.expand(arg2_expand_size)
    )

    # 返回扩展后的张量
    return arg1_broadcasted, arg2_broadcasted


def linalg_solve_is_vector_rhs(input: Tensor, other: Tensor) -> bool:
    # 获取期望的右手边张量（rhs）的批次形状
    expected_batched_rhs_shape = input.shape[:-1]

    # 检查 other 是否是向量，或者是否符合批次形状的条件
    vector_case = other.ndim == 1 or (
        input.ndim - 1 == other.ndim and other.shape == expected_batched_rhs_shape
    )

    # 返回是否为向量的布尔值
    return vector_case


@register_meta(aten._linalg_solve_ex)
def _linalg_solve_ex(
    A: Tensor,
    B: Tensor,
    *,
    left: bool = True,
    check_errors: bool = False,
    result: Optional[Tensor] = None,
    LU: Optional[Tensor] = None,
    pivots: Optional[Tensor] = None,
    info: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # 检查 A 是否为浮点数或复数类型，用于 linalg.solve
    checkFloatingOrComplex(A, "linalg.solve")

    # 检查 A 和 B 的数据类型是否相同
    torch._check(
        A.dtype == B.dtype,
        lambda: (
            f"linalg.solve: Expected A and B to have the same dtype, but found A of type "
            f"{A.dtype} and B of type {B.dtype} instead"
        ),
    )

    # 判断是否为向量的情况
    vector_case = linalg_solve_is_vector_rhs(A, B)

    # 如果是向量，则在最后维度上增加一个维度，否则保持不变
    B_ = B.unsqueeze(-1) if vector_case else B

    # 检查输入参数 A 和 B_，并根据 left 参数进行特定的检查
    checkInputsSolver(A, B_, left, "linalg.solve")

    # 获取 B_ 和 A 的批次广播形状
    B_broad_shape, _ = _linalg_broadcast_batch_dims(B_, A)

    # 如果 left 为 False 且为向量情况，则报错
    torch._check(
        left or not vector_case,
        lambda: (
            "linalg.solve: Vector broadcasting of the left hand side is not supported for left=False. "
            "In this case linalg.solve is equivalent to B / A.squeeze(-1)"
        ),
    )

    # 根据结果的形状创建一个空张量 result_
    result_shape = B_broad_shape[:-1] if vector_case else B_broad_shape
    result_ = torch.empty_strided(
        size=result_shape,
        stride=make_contiguous_strides_for(result_shape, not left),
        dtype=B.dtype,
        device=B.device,
    )

    # 获取 A 的形状和维度
    shape = A.shape
    ndim = A.ndim
    # 创建一个空的张量 LU_，其形状与给定的 shape 相同，但是通过 make_contiguous_strides_for 函数生成非连续的步幅
    LU_ = torch.empty_strided(
        size=shape,
        stride=make_contiguous_strides_for(shape, False),
        dtype=A.dtype,
        device=A.device,
    )
    # 创建一个新的空张量 pivots_，形状为 shape[:-1]，数据类型为 torch.int32
    pivots_ = A.new_empty(shape[:-1], dtype=torch.int32)
    # 创建一个新的空张量 info_，形状为 shape[:-2]，数据类型为 torch.int32
    info_ = A.new_empty(shape[:-2], dtype=torch.int32)
    # 将结果张量、LU_、pivots_ 和 info_ 放入元组 out 中
    out = (result, LU, pivots, info)
    # 创建一个结果的副本元组 res，其中每个元素都与 out 中对应元素的形状相同
    res = (result_, LU_, pivots_, info_)
    # 如果 out 中所有元素均不为 None
    if all(x is not None for x in out):
        # 对于 res 和 out 中的每一对元素 (r, o)，执行以下操作
        for r, o in zip(res, out):
            # 调整 o 的大小并执行拷贝操作（in-place）
            _maybe_resize_out(o, r.shape)  # type: ignore[arg-type]
            # 使用 r 的形状对 o 进行重建，并且不复制步幅（in-place）
            o.as_strided_(r.shape, r.stride())  # type: ignore[union-attr]
            # 安全地从 r 拷贝数据到 o（in-place），不要求精确的数据类型匹配
            _safe_copy_out(copy_from=r, copy_to=o, exact_dtype=False)  # type: ignore[arg-type]
    # 返回结果元组 res
    return res
@register_meta([aten.linalg_solve_triangular.default, aten.linalg_solve_triangular.out])
def linalg_solve_triangular_meta(
    A: Tensor,
    B: Tensor,
    *,
    upper: bool,
    left: bool = True,
    unitriangular: bool = False,
    out: Optional[Tensor] = None,
) -> Tensor:
    # 如果输出张量为空，则创建一个空张量作为输出
    if out is None:
        out = A.new_empty([0])
    # 断言确保输出张量为 TensorLike 类型
    assert isinstance(out, TensorLike)
    # 检查输入参数的有效性
    checkInputsSolver(A, B, left, "linalg.solve_triangular")
    # 调整输入张量的批处理维度，保证形状匹配
    B_, A_ = _linalg_broadcast_batch_dims_name(B, A, None)
    # 避免复制 A 张量的情况
    avoid_copy_A = A_.transpose(-2, -1).is_contiguous() and A_.is_conj()
    # 如果避免复制，则可能调整输出张量的大小
    if avoid_copy_A:
        out = _maybe_resize_out(out, B_.shape)
    else:
        # 使用 F-contig 结果重新实现 resize_output
        if _resize_output_check(out, B_.shape):
            out.resize_(B_.transpose(-2, -1).shape)
            out.transpose_(-2, -1)
    return out  # 返回输出张量


@register_meta(aten.triangular_solve)
@out_wrapper("solution", "cloned_coefficient")
def triangular_solve_meta(
    self: Tensor,
    A: Tensor,
    upper: bool = True,
    transpose: bool = False,
    unitriangular: bool = False,
) -> Tuple[Tensor, Tensor]:
    # 检查 self 张量至少有两个维度
    torch._check(
        self.ndim >= 2,
        lambda: (
            f"torch.triangular_solve: Expected b to have at least 2 dimensions, "
            f"but it has {self.ndim} dimensions instead"
        ),
    )
    # 检查 A 张量至少有两个维度
    torch._check(
        A.ndim >= 2,
        lambda: (
            f"torch.triangular_solve: Expected A to have at least 2 dimensions, "
            f"but it has {A.ndim} dimensions instead"
        ),
    )

    # 检查输入参数的有效性
    linearSolveCheckInputs(self, A, "triangular_solve")

    # 根据 A 张量的布局选择不同的操作路径
    if A.layout == torch.strided:
        # 创建新的步幅张量作为解的输出
        self_broadcast_size, A_broadcast_size = _linalg_broadcast_batch_dims(self, A)
        solution = torch.empty_strided(
            size=self_broadcast_size,
            stride=make_contiguous_strides_for(self_broadcast_size, row_major=False),
            dtype=self.dtype,
            device=self.device,
        )
        # 创建新的步幅张量作为系数的克隆输出
        cloned_coefficient = torch.empty_strided(
            size=A_broadcast_size,
            stride=make_contiguous_strides_for(A_broadcast_size, row_major=False),
            dtype=A.dtype,
            device=A.device,
        )
    elif A.layout == torch.sparse_csr or A.layout == torch.sparse_bsr:
        # 使用 self 张量的形状创建解的稀疏张量
        solution = torch.empty_like(self)
        # 创建新的空张量作为系数的克隆输出
        cloned_coefficient = self.new_empty([0])
    else:
        # 如果布局不是预期的任何一种，则引发错误
        torch._check(False, lambda: "triangular_solve: Got an unexpected layout.")
    return solution, cloned_coefficient  # 返回解和系数的克隆输出


# From aten/src/ATen/native/LinearAlgebra.cpp
@register_meta(aten._linalg_det.default)
def _linalg_det_meta(A):
    # 检查输入张量是否为方阵
    squareCheckInputs(A, "linalg.det")
    # 检查输入张量是否为浮点数或复数类型
    checkFloatingOrComplex(A, "linalg.det")

    # 创建新的空张量作为行和列数减少后的 det 结果
    det = A.new_empty(A.shape[:-2])

    # 创建新的空张量作为 LU 分解的输出
    LU = A.new_empty(A.shape)
    # 将 LU 张量视为步幅张量，确保行和列是紧密排列的
    LU.as_strided_(A.shape, make_contiguous_strides_for(A.shape, row_major=False))

    # 创建新的空张量作为主元素的索引
    pivots = A.new_empty(A.shape[:-1], dtype=torch.int32)
    return det, LU, pivots  # 返回 det、LU 分解和主元素索引
# 注册元数据，将本函数与指定的aten.ormqr函数关联
@register_meta(aten.ormqr)
# 使用out_wrapper装饰器，可能是为了包装输出结果
@out_wrapper()
# 定义函数ormqr，接受以下参数，并返回一个Tensor
def ormqr(
    input: Tensor,
    tau: Tensor,
    other: Tensor,
    left: bool = True,
    transpose: bool = False,
) -> Tensor:
    # 检查输入张量input至少有两个维度
    torch._check(
        input.ndim >= 2, lambda: "torch.ormqr: input must have at least 2 dimensions."
    )
    # 检查其他张量other至少有两个维度
    torch._check(
        other.ndim >= 2, lambda: "torch.ormqr: other must have at least 2 dimensions."
    )

    # 根据left参数确定条件索引值
    left_size_condition = -2 if left else -1
    # 检查other张量在left_size_condition指定的维度上的大小至少为tau张量最后一个维度的大小
    torch._check(
        other.shape[left_size_condition] >= tau.shape[-1],
        lambda: f"torch.ormqr: other.shape[{left_size_condition}] must be greater than or equal to tau.shape[-1]",
    )
    # 检查other张量在left_size_condition指定的维度上的大小等于input张量倒数第二个维度的大小
    torch._check(
        other.shape[left_size_condition] == input.shape[-2],
        lambda: f"torch.ormqr: other.shape[{left_size_condition}] must be equal to input.shape[-2]",
    )

    # 检查tau张量最后一个维度的大小不大于input张量最后一个维度的大小
    torch._check(
        tau.shape[-1] <= input.shape[-1],
        lambda: "torch.ormqr: tau.shape[-1] must be less than or equal to input.shape[-1]",
    )

    # 检查tau张量维度数比input张量少1
    torch._check(
        input.ndim - tau.ndim == 1,
        lambda: (
            f"torch.ormqr: Expected tau to have one dimension less than input, "
            f"but got tau.ndim equal to {tau.ndim} and input.ndim is equal to {input.ndim}"
        ),
    )
    # 检查other张量与input张量具有相同的维度数
    torch._check(
        input.ndim == other.ndim,
        lambda: (
            f"torch.ormqr: Expected other to have the same number of dimensions as input, "
            f"but got other.ndim equal to {other.ndim} and input.ndim is equal to {input.ndim}"
        ),
    )

    # 如果input张量的维度大于2，进一步检查批处理维度
    if input.ndim > 2:
        expected_batch_shape = input.shape[:-2]
        actual_batch_tau_shape = tau.shape[:-1]
        # 检查tau张量的批处理维度与input张量除了最后两个维度外的维度相匹配
        torch._check(
            actual_batch_tau_shape == expected_batch_shape,
            lambda: (
                f"torch.ormqr: Expected batch dimensions of tau to be "
                f"equal to input.shape[:-2], but got {actual_batch_tau_shape}"
            ),
        )

        actual_batch_other_shape = other.shape[:-2]
        # 检查other张量的批处理维度与input张量除了最后两个维度外的维度相匹配
        torch._check(
            actual_batch_other_shape == expected_batch_shape,
            lambda: (
                f"torch.ormqr: Expected batch dimensions of other to be "
                f"equal to input.shape[:-2], but got {actual_batch_other_shape}"
            ),
        )

    # 检查input张量与tau张量具有相同的数据类型
    torch._check(
        tau.dtype == input.dtype,
        lambda: (
            f"torch.ormqr: Expected input and tau to have the same dtype, "
            f"but input has dtype {input.dtype} and tau has dtype {tau.dtype}"
        ),
    )
    # 检查input张量与other张量具有相同的数据类型
    torch._check(
        other.dtype == input.dtype,
        lambda: (
            f"torch.ormqr: Expected input and other to have the same dtype, "
            f"but input has dtype {input.dtype} and other has dtype {other.dtype}"
        ),
    )

    # 检查tau张量与input张量在相同设备上
    checkSameDevice("torch.ormqr", tau, input, "tau")
    # 检查other张量与input张量在相同设备上
    checkSameDevice("torch.ormqr", other, input, "other")
    # 返回一个具有指定大小、步长、数据类型和设备的空张量，基于输入张量的形状
    return torch.empty_strided(
        size=other.shape,  # 使用输入张量 `other` 的形状作为新张量的大小
        stride=make_contiguous_strides_for(other.shape, row_major=False),  # 根据输入张量形状生成连续的步长
        dtype=other.dtype,  # 使用输入张量 `other` 的数据类型作为新张量的数据类型
        device=other.device,  # 使用输入张量 `other` 的设备作为新张量的设备
    )
# 检查输入的填充是否有效，确保填充尺寸与输入维度匹配
def _padding_check_valid_input(input, padding, *, dim):
    # 检查填充的长度是否符合预期（应为 2*dim）
    torch._check(
        len(padding) == 2 * dim,
        lambda: f"padding size is expected to be {2 * dim}, but got: {len(padding)}",
    )

    # 获取输入的维度
    input_dim = input.ndim

    # 判断是否处于批处理模式
    is_batch_mode = input_dim == (dim + 2)

    # 初始化批处理模式和非批处理模式的有效性
    valid_batch_mode = is_batch_mode
    valid_non_batch_mode = not is_batch_mode

    # 如果处于批处理模式
    if is_batch_mode:
        # 允许批处理大小为 0 维度
        for d in range(1, input_dim):
            valid_batch_mode = valid_batch_mode and input.size(d) != 0
    else:
        # 如果非批处理模式
        for d in range(0, input_dim):
            valid_non_batch_mode = valid_non_batch_mode and input.size(d) != 0

    # 检查批处理模式或非批处理模式是否有效
    torch._check(
        valid_batch_mode or valid_non_batch_mode,
        lambda: (
            f"Expected {dim + 1}D or {dim + 2}D (batch mode) tensor with possibly 0 batch size "
            f"and other non-zero dimensions for input, but got: {input.shape}"
        ),
    )


# 对输入进行一维填充的通用函数，支持反射填充或复制填充
def _pad1d_common(input, padding, *, is_reflection):
    # 初始化维度相关变量
    dim_plane = 0
    dim_w = 1
    nbatch = 1

    # 如果输入为三维张量
    if input.ndim == 3:
        # 获取批处理大小，并调整维度索引
        nbatch = input.size(0)
        dim_w += 1
        dim_plane += 1

    # 检查输入填充的有效性
    _padding_check_valid_input(input, padding, dim=1)

    # 获取左右填充值
    pad_l, pad_r = padding

    # 获取平面数和输入的宽度
    nplane = input.size(dim_plane)
    input_w = input.size(dim_w)

    # 计算输出的宽度
    output_w = input_w + pad_l + pad_r

    # 如果是反射填充
    if is_reflection:
        # 检查填充大小是否小于输入维度，确保反射填充有效
        torch._check(
            pad_l < input_w and pad_r < input_w,
            lambda: (
                f"Argument #4: Padding size should be less than the corresponding input dimension, "
                f"but got: padding ({pad_l}, {pad_r}) at dimension {dim_w} of input {input.shape}"
            ),
        )

    # 检查输出宽度是否合理
    torch._check(
        output_w >= 1,
        lambda: f"input (W: {input_w}) is too small. Calculated output W: {output_w}",
    )

    # 如果输入为二维张量，返回一个新的空张量（保留设备和数据类型）
    if input.ndim == 2:
        return input.new_empty((nplane, output_w))
    else:
        # 否则返回一个新的空张量（保留批处理、平面数和输出宽度）
        return input.new_empty((nbatch, nplane, output_w))


# 注册元数据：反射填充一维操作的包装器
@register_meta(aten.reflection_pad1d)
@out_wrapper()
def meta_reflection_pad1d(input, padding):
    return _pad1d_common(input, padding, is_reflection=True)


# 注册元数据：复制填充一维操作的包装器
@register_meta(aten.replication_pad1d)
@out_wrapper()
def meta_replication_pad1d(input, padding):
    return _pad1d_common(input, padding, is_reflection=False)


# 对一维填充操作的反向传播的通用函数
def _pad1d_backward_common(grad_output, input, padding, *, is_reflection):
    dim_w = 1

    # 如果不是反射填充，检查填充大小是否为2
    if not is_reflection:
        torch._check(len(padding) == 2, lambda: "padding size is expected to be 2")

    # 如果输入为三维张量，调整维度索引
    if input.ndim == 3:
        dim_w += 1

    # 获取左右填充值
    pad_l, pad_r = padding

    # 获取输入的宽度和输出的宽度
    input_w = input.size(dim_w)
    output_w = input_w + pad_l + pad_r
    # 如果开启反射模式，则进行以下检查
    if is_reflection:
        # 使用 torch._check 函数检查条件，确保左右填充大小小于输入宽度
        torch._check(
            pad_l < input_w and pad_r < input_w,
            lambda: (
                f"Argument #4: Padding size should be less than the corresponding input dimension, "
                f"but got: padding ({pad_l}, {pad_r}) at dimension {dim_w} of input {input.shape}"
            ),
        )

    # 使用 torch._check 函数检查条件，确保输出宽度与梯度输出张量的指定维度大小一致
    torch._check(
        output_w == grad_output.size(dim_w),
        lambda: f"grad_output width unexpected. Expected: {output_w}, Got: {grad_output.size(dim_w)}",
    )

    # 返回一个与输入形状相同的空张量
    return input.new_empty(input.shape)
# 注册元数据处理函数，用于反射填充1维的反向传播
@register_meta(aten.reflection_pad1d_backward)
@out_wrapper("grad_input")
def meta_reflection_pad1d_backward(grad_output, input, padding):
    return _pad1d_backward_common(grad_output, input, padding, is_reflection=True)


# 注册元数据处理函数，用于复制填充1维的反向传播
@register_meta(aten.replication_pad1d_backward)
@out_wrapper("grad_input")
def meta_replication_pad1d_backward(grad_output, input, padding):
    return _pad1d_backward_common(grad_output, input, padding, is_reflection=False)


# 定义2维填充通用函数，处理输入、填充和反射填充标志
def _pad2d_common(input, padding, *, is_reflection):
    # 定义维度索引
    dim_w = 2
    dim_h = 1
    dim_slices = 0
    nbatch = 1

    # 检查填充参数的有效性
    _padding_check_valid_input(input, padding, dim=2)

    # 获取输入数据的维度数
    ndim = input.ndim
    if ndim == 4:
        # 如果输入是4维张量，更新相关维度索引和批次数
        nbatch = input.size(0)
        dim_w += 1
        dim_h += 1
        dim_slices += 1

    # 解包填充参数
    pad_l, pad_r, pad_t, pad_b = padding

    # 获取平面数、输入高度和宽度，计算输出的高度和宽度
    nplane = input.size(dim_slices)
    input_h = input.size(dim_h)
    input_w = input.size(dim_w)
    output_h = input_h + pad_t + pad_b
    output_w = input_w + pad_l + pad_r

    # 如果是反射填充模式，检查填充边界
    if is_reflection:
        torch._check(
            pad_l < input_w and pad_r < input_w,
            lambda: (
                f"Argument #4: Padding size should be less than the corresponding input dimension, "
                f"but got: padding ({pad_l}, {pad_r}) at dimension {dim_w} of input {input.shape}"
            ),
        )
        torch._check(
            pad_t < input_h and pad_b < input_h,
            lambda: (
                f"Argument #6: Padding size should be less than the corresponding input dimension, "
                f"but got: padding ({pad_t}, {pad_b}) at dimension {dim_h} of input {input.shape}"
            ),
        )

    # 检查输出高度和宽度是否合法
    torch._check(
        output_w >= 1 or output_h >= 1,
        lambda: (
            f"input (H: {input_h} W: {input_w}) is too small. "
            f"Calculated output H: {output_h} W: {output_w}"
        ),
    )

    # 根据输入数据维度返回新的空张量
    if input.ndim == 3:
        return input.new_empty((nplane, output_h, output_w))
    else:
        return input.new_empty((nbatch, nplane, output_h, output_w))


# 注册元数据处理函数，用于反射填充2维
@register_meta(aten.reflection_pad2d)
@out_wrapper()
def meta_reflection_pad2d(input, padding):
    return _pad2d_common(input, padding, is_reflection=True)


# 注册元数据处理函数，用于复制填充2维
@register_meta(aten.replication_pad2d)
@out_wrapper()
def meta_replication_pad2d(input, padding):
    return _pad2d_common(input, padding, is_reflection=False)


# 注册元数据处理函数，用于2维填充的反向传播
@register_meta(
    [
        aten.reflection_pad2d_backward.default,
        aten.reflection_pad2d_backward.grad_input,
        aten.replication_pad2d_backward.default,
        aten.replication_pad2d_backward.grad_input,
    ]
)
@out_wrapper("grad_input")
def meta_pad2d_backward(grad_output, self, padding):
    dim_w = 2
    dim_h = 1
    dim_plane = 0
    nbatch = 1

    # 获取self张量的形状
    self_shape = self.shape
    if self.dim() == 4:
        # 如果self是4维张量，更新相关维度索引和批次数
        nbatch = self_shape[0]
        dim_w += 1
        dim_h += 1
        dim_plane += 1

    # 解包填充参数
    pad_l, pad_r, pad_t, pad_b = padding

    # 获取平面数、输入高度和宽度
    nplane = self_shape[dim_plane]
    input_h = self_shape[dim_h]
    input_w = self_shape[dim_w]
    # 计算输出高度，根据输入高度和顶部/底部填充量相加得到
    output_h = input_h + pad_t + pad_b
    # 计算输出宽度，根据输入宽度和左侧/右侧填充量相加得到
    output_w = input_w + pad_l + pad_r

    # 使用 torch._check 函数检查梯度输出的宽度是否符合预期
    torch._check(
        output_w == grad_output.size(dim_w),
        lambda: f"grad_output width unexpected. Expected: {output_w}, Got: {grad_output.size(dim_w)}",
    )
    # 使用 torch._check 函数检查梯度输出的高度是否符合预期
    torch._check(
        output_h == grad_output.size(dim_h),
        lambda: f"grad_output height unexpected. Expected: {output_h}, Got: {grad_output.size(dim_h)}",
    )
    # 返回一个新的空张量，形状由 self.shape 决定
    return self.new_empty(self.shape)
# 定义一个内部函数，用于在3D输入张量上应用通用的填充操作，支持反射和复制两种填充方式
def _pad3d_common(input, padding, *, is_reflection):
    # 定义张量的维度索引
    dim_w = 3  # 宽度维度索引
    dim_h = 2  # 高度维度索引
    dim_d = 1  # 深度维度索引
    dim_plane = 0  # 平面维度索引

    # 检查输入和填充是否有效
    _padding_check_valid_input(input, padding, dim=3)

    # 检查是否处于批处理模式
    batch_mode = input.ndim == 5
    if batch_mode:
        # 如果是批处理模式，获取批次大小并调整维度索引
        nbatch = input.size(0)
        dim_w += 1
        dim_h += 1
        dim_d += 1
        dim_plane += 1

    # 分别获取填充值
    pad_l, pad_r, pad_t, pad_b, pad_f, pad_bk = padding

    # 获取输入张量的平面数和各个维度的大小
    nplane = input.size(dim_plane)
    input_d = input.size(dim_d)
    input_h = input.size(dim_h)
    input_w = input.size(dim_w)

    # 计算输出张量的各个维度大小
    output_d = input_d + pad_f + pad_bk
    output_h = input_h + pad_t + pad_b
    output_w = input_w + pad_l + pad_r

    # 如果是反射填充，进行额外的检查
    if is_reflection:
        # 检查宽度维度的填充大小是否有效
        torch._check(
            pad_l < input_w and pad_r < input_w,
            lambda: (
                f"Argument #4: Padding size should be less than the corresponding input dimension, "
                f"but got: padding ({pad_l}, {pad_r}) at dimension {dim_w} of input {input.shape}"
            ),
        )
        # 检查高度维度的填充大小是否有效
        torch._check(
            pad_t < input_h and pad_b < input_h,
            lambda: (
                f"Argument #6: Padding size should be less than the corresponding input dimension, "
                f"but got: padding ({pad_t}, {pad_b}) at dimension {dim_h} of input {input.shape}"
            ),
        )
        # 检查深度维度的填充大小是否有效
        torch._check(
            pad_f < input_d and pad_bk < input_d,
            lambda: (
                f"Argument #8: Padding size should be less than the corresponding input dimension, "
                f"but got: padding ({pad_f}, {pad_bk}) at dimension {dim_d} of input {input.shape}"
            ),
        )

    # 检查输出张量的维度是否合理
    torch._check(
        output_w >= 1 or output_h >= 1 or output_d >= 1,
        lambda: (
            f"input (D: {input_d} H: {input_h} W: {input_w}) is too small. "
            f"Calculated output D: {output_d} H: {output_h} W: {output_w}"
        ),
    )

    # 如果处于批处理模式，返回一个新的空张量，否则返回一个新的空张量
    if batch_mode:
        return input.new_empty((nbatch, nplane, output_d, output_h, output_w))  # type: ignore[possibly-undefined]
    else:
        return input.new_empty((nplane, output_d, output_h, output_w))


# 注册反射填充3D操作的元信息，并将其包装为输出函数
@register_meta(aten.reflection_pad3d)
@out_wrapper()
def meta_reflection_pad3d(input, padding):
    return _pad3d_common(input, padding, is_reflection=True)


# 注册复制填充3D操作的元信息，并将其包装为输出函数
@register_meta(aten.replication_pad3d)
@out_wrapper()
def meta_replication_pad3d(input, padding):
    return _pad3d_common(input, padding, is_reflection=False)


# 注册3D填充操作的反向传播元信息，并将其包装为输出函数
@register_meta(
    [
        aten.reflection_pad3d_backward.default,
        aten.reflection_pad3d_backward.grad_input,
        aten.replication_pad3d_backward.default,
        aten.replication_pad3d_backward.grad_input,
    ]
)
@out_wrapper("grad_input")
def meta_pad3d_backward(grad_output, input, padding):
    # 检查填充大小是否为6
    torch._check(len(padding) == 6, lambda: "padding size is expected to be 6")
    # 断言输入张量的维度大于3
    assert input.ndim > 3
    # 断言梯度输出张量的维度与输入张量的维度相同
    assert grad_output.ndim == input.ndim

    dim_w = 3  # 宽度维度索引
    dim_h = 2  # 高度维度索引
    dim_d = 1  # 深度维度索引
    # 检查输入张量的维度是否为5维，如果是则增加相应的维度值
    if input.ndim == 5:
        dim_w += 1
        dim_h += 1
        dim_d += 1

    # 将padding元组解包分配给相应的变量
    pad_l, pad_r, pad_t, pad_b, pad_f, pad_bk = padding

    # 获取输入张量在dim_d维度上的大小
    input_d = input.size(dim_d)
    # 获取输入张量在dim_h维度上的大小
    input_h = input.size(dim_h)
    # 获取输入张量在dim_w维度上的大小
    input_w = input.size(dim_w)
    
    # 计算输出张量在dim_d维度上的大小，加上前向和后向的padding值
    output_d = input_d + pad_f + pad_bk
    # 计算输出张量在dim_h维度上的大小，加上上方和下方的padding值
    output_h = input_h + pad_t + pad_b
    # 计算输出张量在dim_w维度上的大小，加上左侧和右侧的padding值
    output_w = input_w + pad_l + pad_r

    # 检查输出张量在dim_w维度上的大小是否与grad_output张量相同
    torch._check(
        output_w == grad_output.size(dim_w),
        lambda: f"grad_output width unexpected. Expected: {output_w}, Got: {grad_output.size(dim_w)}",
    )
    # 检查输出张量在dim_h维度上的大小是否与grad_output张量相同
    torch._check(
        output_h == grad_output.size(dim_h),
        lambda: f"grad_output height unexpected. Expected: {output_h}, Got: {grad_output.size(dim_h)}",
    )
    # 检查输出张量在dim_d维度上的大小是否与grad_output张量相同
    torch._check(
        output_d == grad_output.size(dim_d),
        lambda: f"grad_output depth unexpected. Expected: {output_d}, Got: {grad_output.size(dim_d)}",
    )

    # 返回与输入张量相同设备的新空张量，形状与输入张量相同
    return input.new_empty(input.shape)
@register_meta(aten._pdist_forward)
@out_wrapper()
def meta__pdist_forward(self: Tensor, p: float = 2) -> Tensor:
    # 检查输入张量是否是连续的
    torch._check(
        self.is_contiguous(), lambda: "_pdist_forward requires contiguous input"
    )
    # 获取张量的大小
    n = self.size(0)
    # 如果张量大小小于等于1，则返回一个空张量，使用旧的内存格式
    if n <= 1:
        return self.new_empty([0]).to(memory_format=torch.legacy_contiguous_format)  # type: ignore[call-overload]
    else:
        # 否则返回一个指定形状的空张量，使用旧的内存格式
        return self.new_empty((n * (n - 1) // 2,)).to(
            memory_format=torch.legacy_contiguous_format
        )  # type: ignore[call-overload]


@register_meta(aten._pdist_backward)
@out_wrapper()
def meta__pdist_backward(grad: Tensor, self: Tensor, p: float, pdist: Tensor) -> Tensor:
    # 检查 self 张量是否是连续的
    torch._check(
        self.is_contiguous(), lambda: "_pdist_backward requires self to be contiguous"
    )
    # 检查 pdist 张量是否是连续的
    torch._check(
        pdist.is_contiguous(), lambda: "_pdist_backward requires pdist to be contiguous"
    )
    # 返回一个和 self 张量相同形状的空张量，使用旧的内存格式
    return torch.empty_like(self, memory_format=torch.legacy_contiguous_format)


@register_meta([aten.baddbmm.default, aten.baddbmm.out])
@out_wrapper()
def meta_baddbmm(self, batch1, batch2, *, beta=1, alpha=1):
    # 获取 batch1 的三个维度大小
    dim1 = batch1.size(0)
    dim2 = batch1.size(1)
    dim3 = batch2.size(2)
    # 将 self 张量扩展为 (dim1, dim2, dim3) 的形状
    self = self.expand((dim1, dim2, dim3))
    # 检查 batch1 是否是一个三维张量
    torch._check(batch1.dim() == 3, lambda: "batch1 must be a 3D tensor")
    # 检查 batch2 是否是一个三维张量
    torch._check(batch2.dim() == 3, lambda: "batch2 must be a 3D tensor")
    # 检查 self、batch1、batch2 的数据类型是否一致
    torch._check(
        self.dtype == batch1.dtype == batch2.dtype,
        lambda: f"Input dtypes must be the same, got: input: {self.dtype}, batch1: {batch1.dtype}, batch2: {batch2.dtype}",
    )
    # 获取 batch1、batch2 的形状
    batch1_sizes = batch1.shape
    batch2_sizes = batch2.shape
    bs = batch1_sizes[0]
    contraction_size = batch1_sizes[2]
    # 检查 batch2 的第一、二维度大小是否符合预期
    torch._check(
        batch2_sizes[0] == bs and batch2_sizes[1] == contraction_size,
        lambda: (
            f"Expected size for first two dimensions of batch2 tensor to be: "
            f"[{bs}, {contraction_size}] but got: [{batch2_sizes[0]}, {batch2_sizes[1]}]."
        ),
    )
    # 返回一个和 self 张量相同形状的空张量
    return self.new_empty(self.size())


@register_meta([aten.bernoulli.default, aten.bernoulli.out])
@out_wrapper()
def meta_bernoulli(self, *, generator=None):
    # 返回一个和 self 张量相同形状的空张量，并确保是连续的
    return torch.empty_like(self).contiguous()


@register_meta(aten.bernoulli_.float)
def meta_bernoulli_(self, p=0.5, generator=None):
    # 直接返回 self 张量，不做任何修改
    return self


@register_meta(aten.bernoulli.p)
def meta_bernoulli_p(self, p=0.5, generator=None):
    # 返回一个和 self 张量相同形状的空张量，并确保是连续的
    return torch.empty_like(self).contiguous()


@register_meta(aten._fused_moving_avg_obs_fq_helper.default)
def meta__fused_moving_avg_obs_fq_helper(
    self,
    observer_on,
    fake_quant_on,
    running_min,
    running_max,
    scale,
    zero_point,
    averaging_const,
    quant_min,
    quant_max,
    ch_axis,
    per_row_fake_quant=False,
    symmetric_quant=False,
):
    # 这个函数暂时没有实现内容
    pass
    # 使用 torch._check 方法检查 ch_axis 是否小于 self 的维度数，确保其合法性
    torch._check(
        ch_axis < self.dim(),
        lambda: "Error in fused_moving_avg_obs_fake_quant_cpu: ch_axis must be < self.dim()",
    )
    # 创建一个与 self 张量相同形状的空张量 mask，数据类型为 torch.bool
    mask = torch.empty_like(self, dtype=torch.bool)
    # 返回一个元组，包含与 self 张量相同形状的空张量和刚创建的 mask 张量
    return (torch.empty_like(self), mask)
@register_meta(aten.mm)
@out_wrapper()
# 注册为元操作，并返回装饰器包装的函数
def meta_mm(a, b):
    torch._check(a.dim() == 2, lambda: "a must be 2D")
    # 检查张量 a 是否为二维
    torch._check(b.dim() == 2, lambda: "b must be 2D")
    # 检查张量 b 是否为二维
    N, M1 = a.shape
    # 获取张量 a 的形状，并将其分解为 N 和 M1
    M2, P = b.shape
    # 获取张量 b 的形状，并将其分解为 M2 和 P
    torch._check(
        M1 == M2,
        lambda: f"a and b must have same reduction dim, but got [{N}, {M1}] X [{M2}, {P}].",
    )
    # 检查 M1 和 M2 是否相等，若不相等则引发异常
    return a.new_empty(N, P)
    # 返回一个与 a 相同类型的新的空张量，形状为 N x P


def _compute_reduction_shape(self, dims, keepdim):
    if keepdim:
        return tuple(self.shape[i] if i not in dims else 1 for i in range(self.ndim))
    # 如果 keepdim 为 True，返回保持维度的形状
    return utils.compute_reduction_output_shape(self.shape, dims)
    # 否则，使用工具函数计算按给定维度减少后的输出形状


# FakeTensors (meta tensors with a device) will report device as meta
# when running meta kernels. Here, access the "fake device" of FakeTensor if it
# exists so meta kernels which have diverge per device will be more
# accurate when run with FakeTensors
# 在运行元内核时，FakeTensors（具有设备的元张量）将其设备报告为元设备。这里，如果存在 FakeTensor 的 "fake device"，则访问它，
# 因此使用 FakeTensors 运行时，能够更准确地根据设备分歧运行元内核
def device_hint(tensor) -> "str":
    if isinstance(tensor, torch._subclasses.FakeTensor):
        return tensor.fake_device.type
    # 如果 tensor 是 FakeTensor 类型，返回其伪设备类型
    else:
        return "cuda"  # default to cuda
    # 否则，默认返回 "cuda"


def calc_conv_nd_return_shape(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    stride: Union[List[int], int],
    padding: Union[List[int], int],
    dilation: Union[List[int], int],
    is_transposed: bool,
    groups: int,
    output_padding: Optional[Union[List[int], int]] = None,
):
    def _formula(ln: int, p: int, d: int, k: int, s: int) -> int:
        """
        Formula to apply to calculate the length of some dimension of the output

        See: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

        Args:
            ln: length of the dimension
            p: padding in that dim
            d: dilation in that dim
            k: kernel size in that dim
            s: stride in that dim
        Returns:
            The output length
        """
        return (ln + 2 * p - d * (k - 1) - 1) // s + 1
        # 返回输出某维度的长度，应用于计算输出的公式

    def _formula_transposed(ln: int, p: int, d: int, k: int, s: int, op: int) -> int:
        """
        Formula to apply to calculate the length of some dimension of the output
        if transposed convolution is used.
        See: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html

        Args:
            ln: length of the dimension
            p: padding in that dim
            d: dilation in that dim
            k: kernel size in that dim
            s: stride in that dim
            op: output padding in that dim

        Returns:
            The output length
        """
        return (ln - 1) * s - 2 * p + d * (k - 1) + op + 1
        # 返回输出某维度的长度，应用于计算转置卷积输出的公式

    kernel_size = weight.shape[2:]
    # 获取权重张量的核心大小
    dims = input_tensor.shape[2:]
    # 获取输入张量的维度大小
    if is_transposed:
        out_channels = groups * weight.shape[1]
        # 如果是转置卷积，计算输出通道数
    else:
        out_channels = weight.shape[0]
        # 否则，计算输出通道数为权重张量的第一维大小
        if weight.shape[1] * groups != input_tensor.shape[1]:
            raise RuntimeError("Invalid channel dimensions")
        # 如果权重的第二维乘以组数不等于输入张量的第二维大小，则引发异常

    ret_shape = [input_tensor.shape[0], out_channels]
    # 返回的形状为输入张量的第一维大小和输出通道数
    if isinstance(stride, IntLike):
        stride = [stride] * len(dims)
        # 如果步长为整数，转换为与维度大小相同的列表步长
    # 如果步长参数是一个整数，则将其扩展为与维度数相同的列表
    elif len(stride) == 1:
        stride = [stride[0]] * len(dims)

    # 如果填充参数是一个整数类型，则将其扩展为与维度数相同的列表
    if isinstance(padding, IntLike):
        padding = [padding] * len(dims)
    elif len(padding) == 1:
        padding = [padding[0]] * len(dims)

    # 如果扩张参数是一个整数类型，则将其扩展为与维度数相同的列表
    if isinstance(dilation, IntLike):
        dilation = [dilation] * len(dims)
    elif len(dilation) == 1:
        dilation = [dilation[0]] * len(dims)

    # 初始化一个可选的输出填充列表
    output_padding_list: Optional[List[int]] = None
    if output_padding:
        # 如果输出填充是一个整数类型，则将其扩展为与维度数相同的列表
        if isinstance(output_padding, IntLike):
            output_padding_list = [output_padding] * len(dims)
        elif len(output_padding) == 1:
            output_padding_list = [output_padding[0]] * len(dims)
        else:
            output_padding_list = output_padding

    # 遍历维度列表，计算每个维度的输出形状
    for i in range(len(dims)):
        # 如果存在输出填充，则表示处理转置卷积
        if output_padding_list:
            ret_shape.append(
                _formula_transposed(
                    dims[i],
                    padding[i],
                    dilation[i],
                    kernel_size[i],
                    stride[i],
                    output_padding_list[i],
                )
            )
        else:
            # 否则，按正常卷积的方式计算输出形状
            ret_shape.append(
                _formula(dims[i], padding[i], dilation[i], kernel_size[i], stride[i])
            )

    # 返回计算得到的形状列表
    return ret_shape
# 定义一个函数，用于判断给定的张量是否按照通道优先的内存格式存储
def is_channels_last(ten):
    return torch._prims_common.suggest_memory_format(ten) == torch.channels_last


# 注册一个元信息函数，用于处理普通的卷积操作
@register_meta(aten.convolution.default)
def meta_conv(
    input_tensor: torch.Tensor,        # 输入张量
    weight: torch.Tensor,              # 卷积核张量
    bias: torch.Tensor,                # 偏置张量
    stride: List[int],                 # 步长列表
    padding: List[int],                # 填充列表
    dilation: List[int],               # 膨胀列表
    is_transposed: bool,               # 是否为反卷积
    output_padding: List[int],         # 输出填充列表（仅反卷积时有效）
    groups: int,                       # 分组卷积数
):
    # 内部函数，选择合适的内存格式
    def pick_memory_format():
        # 如果设备提示为 CUDA，并且输入张量或卷积核张量按通道优先格式存储，则选择通道优先格式
        if device_hint(input_tensor) == "cuda":
            if is_channels_last(input_tensor) or is_channels_last(weight):
                return torch.channels_last
        else:
            # 如果在非 CUDA 设备上，并且输入张量按通道优先格式存储，则选择通道优先格式
            if is_channels_last(input_tensor):
                return torch.channels_last
        # 如果张量已经是连续的（按 torch.contiguous_format 存储），则保持其内存格式
        if input_tensor.is_contiguous(memory_format=torch.contiguous_format):
            return torch.contiguous_format
        # 否则，保持当前内存格式
        elif input_tensor.is_contiguous(memory_format=torch.preserve_format):
            return torch.preserve_format

    # 计算卷积操作的输出形状
    shape_out = calc_conv_nd_return_shape(
        input_tensor,
        weight,
        stride,
        padding,
        dilation,
        is_transposed,
        groups,
        output_padding if is_transposed else None,
    )

    input_channels_dim = 1
    output_channels_dim = 1
    # 如果输入张量的输入通道数为0，则输出形状中的输出通道数维度置为0
    if input_tensor.size(input_channels_dim) == 0:
        shape_out[output_channels_dim] = 0

    # 创建一个与输入张量形状相同的空张量作为输出
    out = input_tensor.new_empty(shape_out)
    # 将输出张量转换为指定的内存格式
    out = out.to(memory_format=pick_memory_format())  # type: ignore[call-overload]
    return out


# 如果 torch 支持 MKL-DNN，则注册一个特定操作的元信息函数
if torch._C._has_mkldnn:
    # 创建一个 MKL-DNN 库的元信息函数
    _meta_lib_dont_use_me_use_register_meta_for_mkldnn = torch.library.Library(
        "mkldnn", "IMPL", "Meta"
    )

    # 注册一个 MKL-DNN 的卷积操作的元信息函数
    @register_meta(torch.ops.mkldnn._convolution_pointwise.default)
    def meta_mkldnn_convolution_default(
        input_tensor,        # 输入张量
        weight,              # 卷积核张量
        bias,                # 偏置张量
        padding,             # 填充列表
        stride,              # 步长列表
        dilation,            # 膨胀列表
        groups,              # 分组卷积数
        attr,                # 属性
        scalars,             # 标量
        algorithm,           # 算法
    ):
        # 计算 MKL-DNN 卷积操作的输出形状
        shape_out = calc_conv_nd_return_shape(
            input_tensor, weight, stride, padding, dilation, False, groups, []
        )
        # 创建一个与输入张量形状相同的空张量作为输出
        out = input_tensor.new_empty(shape_out)
        out_memory_format = torch.channels_last
        # 如果输入张量的维度为5，则输出张量选择 3D 通道优先格式
        if input_tensor.dim() == 5:
            out_memory_format = torch.channels_last_3d
        # 将输出张量转换为指定的内存格式
        out = out.to(memory_format=out_memory_format)  # type: ignore[call-overload]
        return out

    # 注册一个 MKL-DNN 的线性操作的元信息函数
    @register_meta(torch.ops.mkldnn._linear_pointwise.default)
    def meta_linear_pointwise_default(
        input_tensor,        # 输入张量
        weight,              # 权重张量
        bias,                # 偏置张量
        attr,                # 属性
        scalars,             # 标量
        algorithm            # 算法
    ):
        # 创建一个形状与输入张量的除最后一维外相同的空张量作为输出
        return input_tensor.new_empty((*input_tensor.shape[:-1], weight.shape[0]))
    # 检查是否 Torch 使用了 MKL（Math Kernel Library）
    if torch._C.has_mkl:
        # 创建一个名为 _meta_lib_dont_use_me_use_register_meta_for_mkl 的 Torch 库对象，
        # 用于注册 MKL 的元数据信息
        _meta_lib_dont_use_me_use_register_meta_for_mkl = torch.library.Library(
            "mkl", "IMPL", "Meta"
        )

        # 定义一个装饰器，用于注册 MKL 版本的线性操作的元数据信息
        @register_meta(torch.ops.mkl._mkl_linear)
        def meta_mkl_linear(
            input_tensor,
            packed_weight,
            orig_weight,
            bias,
            batch_size,
        ):
            # 返回一个新的空张量，与输入张量除最后一个维度外相同，最后一个维度为原始权重的大小
            return input_tensor.new_empty(
                (*input_tensor.shape[:-1], orig_weight.shape[0])
            )

    # 创建一个名为 _meta_lib_dont_use_me_use_register_meta_for_onednn 的 Torch 库对象，
    # 用于注册 OneDNN 的元数据信息
    _meta_lib_dont_use_me_use_register_meta_for_onednn = torch.library.Library(
        "onednn", "IMPL", "Meta"
    )

    # 定义一个装饰器，用于注册 OneDNN 版本的点卷积操作的元数据信息
    @register_meta(torch.ops.onednn.qconv2d_pointwise.default)
    def meta_qconv2d_pointwise(
        x,
        x_scale,
        x_zp,
        w,  # prepacked_weight
        w_scale,
        w_zp,
        bias,
        stride,
        padding,
        dilation,
        groups,
        output_scale,
        output_zero_point,
        output_dtype,
        attr,
        scalars,
        algorithm,
    ):
        # 计算卷积操作的输出形状
        shape_out = calc_conv_nd_return_shape(
            x,
            w,
            stride,
            padding,
            dilation,
            False,
            groups,
            None,
        )
        # 断言输出数据类型为 torch.float32 或 torch.bfloat16
        assert output_dtype in [torch.float32, torch.bfloat16]
        # 创建一个与输入张量 x 形状相同的空张量 out，数据类型为 output_dtype
        out = x.new_empty(shape_out, dtype=output_dtype)
        # 将张量 out 转换为以通道优先的内存格式
        out = out.to(memory_format=torch.channels_last)
        return out

    # 定义一个装饰器，用于注册 OneDNN 版本的线性操作的元数据信息
    @register_meta(torch.ops.onednn.qlinear_pointwise.default)
    @register_meta(torch.ops.onednn.qlinear_pointwise.tensor)
    def meta_qlinear_pointwise(
        x,
        x_scale,
        x_zp,
        w,
        w_scale,
        w_zp,
        bias,
        output_scale,
        output_zero_point,
        output_dtype,
        post_op_name,
        post_op_args,
        post_op_algorithm,
    ):
        # 获取输入张量 x 的形状，并存储在 output_shape 变量中
        output_shape = list(x.shape)
        # 线性权重在预打包过程中已被转置
        output_shape[-1] = w.shape[1]
        # 断言输出数据类型为 torch.float32 或 torch.bfloat16
        assert output_dtype in [torch.float32, torch.bfloat16]
        # 创建一个形状与 output_shape 相同的新空张量 out，数据类型为 output_dtype
        out = x.new_empty(output_shape, dtype=output_dtype)
        return out

    # 创建一个名为 _meta_lib_dont_use_me_use_register_meta_for_quantized 的 Torch 库对象，
    # 用于注册量化操作的元数据信息
    _meta_lib_dont_use_me_use_register_meta_for_quantized = torch.library.Library(
        "quantized", "IMPL", "Meta"
    )

    # 定义一个装饰器，用于注册量化版本的最大池化操作的元数据信息
    @register_meta(torch.ops.quantized.max_pool2d)
    def meta_quantized_max_pool2d(
        input,
        kernel_size,
        stride=(),
        padding=(0,),
        dilation=(1,),
        ceil_mode=False,
    ):
        (
            nInputPlane,  # 提取 max_pool2d_checks_and_compute_shape 函数返回的 nInputPlane 变量
            outputHeight,  # 提取 max_pool2d_checks_and_compute_shape 函数返回的 outputHeight 变量
            outputWidth,  # 提取 max_pool2d_checks_and_compute_shape 函数返回的 outputWidth 变量
        ) = max_pool2d_checks_and_compute_shape(
            input, kernel_size, stride, padding, dilation, ceil_mode  # 调用 max_pool2d_checks_and_compute_shape 函数，计算池化层输出的形状
        )
        nbatch = input.size(-4) if input.dim() == 4 else 1  # 计算输入张量的批次数，如果维度为4则取第一个维度大小，否则为1
        memory_format = torch.channels_last  # 设置内存布局格式为通道优先
        if input.dim() == 3:
            size = [nInputPlane, outputHeight, outputWidth]  # 如果输入维度为3，设置输出张量大小为 [nInputPlane, outputHeight, outputWidth]
        else:
            size = [nbatch, nInputPlane, outputHeight, outputWidth]  # 如果输入维度不为3，设置输出张量大小为 [nbatch, nInputPlane, outputHeight, outputWidth]
        return torch.empty(
            size,  # 创建一个空的张量，大小由上面确定，用来作为池化层的输出
            dtype=input.dtype,  # 设置输出张量的数据类型与输入张量相同
            device=input.device,  # 设置输出张量的设备与输入张量相同
            memory_format=memory_format,  # 设置输出张量的内存布局格式
        )
# 从 aten/src/ATen/TensorUtils.cpp 的 check_dim_size() 函数中引入。

def check_dim_size(tensor, dim, dim_size, size):
    # 检查张量 tensor 的维度是否为 dim，并且指定维度 dim_size 的大小是否为 size
    torch._check(
        tensor.dim() == dim and tensor.shape[dim_size] == size,
        lambda: f"Expected a tensor of dimension {dim} and tensor.size[{dim_size}] == {size}, "
        + f"but got : dimension {tensor.dim()} and tensor.size[{dim_size}] = {tensor.shape[dim_size]}",
    )


# 注册 aten.avg_pool2d.default 的元信息函数。
@register_meta(aten.avg_pool2d.default)
def meta_avg_pool2d(
    input,
    kernel_size,
    stride=(),
    padding=(0,),
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    # 解包 kernel_size，确定 kH 和 kW
    def unpack(name, val):
        torch._check(
            len(val) in [1, 2],
            lambda: f"avg_pool2d: {name} must either be a single int, or a tuple of two ints",
        )
        H = val[0]
        W = H if len(val) == 1 else val[1]
        return H, W

    kH, kW = unpack("kernel_size", kernel_size)

    # 检查 stride 是否为合法值
    torch._check(
        len(stride) in [0, 1, 2],
        lambda: "avg_pool2d: stride must either be omitted, a single int, or a tuple of two ints",
    )
    if len(stride) == 0:
        dH, dW = kH, kW
    elif len(stride) == 1:
        dH, dW = stride[0], stride[0]
    else:
        dH, dW = unpack("stride", stride)

    # 解包 padding，确定 padH 和 padW
    padH, padW = unpack("padding", padding)

    # 检查 divisor_override 是否合法
    torch._check(
        divisor_override is None or divisor_override != 0,
        lambda: "divisor must be not zero",
    )

    # 确定输入的批次数、输入通道数、输入高度和宽度
    nbatch = input.size(-4) if input.dim() == 4 else 1
    nInputPlane = input.size(-3)
    inputHeight = input.size(-2)
    inputWidth = input.size(-1)

    # 计算池化操作后的输出高度和宽度
    outputHeight = pooling_output_shape(inputHeight, kH, padH, dH, 1, ceil_mode)
    outputWidth = pooling_output_shape(inputWidth, kW, padW, dW, 1, ceil_mode)

    # 建议的内存格式
    memory_format = utils.suggest_memory_format(input)

    # 进行池化形状检查
    pool2d_shape_check(
        input,
        kH,
        kW,
        dH,
        dW,
        padH,
        padW,
        1,
        1,
        nInputPlane,
        inputHeight,
        inputWidth,
        outputHeight,
        outputWidth,
        memory_format,
    )

    # 如果输入是三维张量，确定输出尺寸为 [nInputPlane, outputHeight, outputWidth]，否则为 [nbatch, nInputPlane, outputHeight, outputWidth]
    if input.dim() == 3:
        size = [nInputPlane, outputHeight, outputWidth]
    else:
        size = [nbatch, nInputPlane, outputHeight, outputWidth]

    # 返回一个空的张量，具有指定的尺寸、数据类型、设备和内存格式
    return torch.empty(
        size,
        dtype=input.dtype,
        device=input.device,
        memory_format=memory_format,
    )


# 从 aten/src/ATen/native/Pool.h 的 avg_pool2d_backward_shape_check() 函数中引入。

def avg_pool2d_backward_shape_check(
    input,
    gradOutput,
    nbatch,
    kH,
    kW,
    dH,
    dW,
    padH,
    padW,
    nInputPlane,
    inputHeight,
    inputWidth,
    outputHeight,
    outputWidth,
    mem_format,
):
    # 调用池化形状检查函数，确保输入和输出形状符合预期
    pool2d_shape_check(
        input,
        kH,
        kW,
        dH,
        dW,
        padH,
        padW,
        1,
        1,
        nInputPlane,
        inputHeight,
        inputWidth,
        outputHeight,
        outputWidth,
        mem_format,
    )

    # 确定输入张量的维度数，并设置输出通道数为输入通道数
    ndim = input.dim()
    nOutputPlane = nInputPlane
    # 检查梯度张量的维度大小是否符合预期，第一个参数 gradOutput 是梯度张量，
    # ndim 是张量的总维度数，ndim - 3 是梯度张量在总维度中的索引位置，nOutputPlane 是期望的大小
    check_dim_size(gradOutput, ndim, ndim - 3, nOutputPlane)
    
    # 检查梯度张量的维度大小是否符合预期，第一个参数 gradOutput 是梯度张量，
    # ndim 是张量的总维度数，ndim - 2 是梯度张量在总维度中的索引位置，outputHeight 是期望的大小
    check_dim_size(gradOutput, ndim, ndim - 2, outputHeight)
    
    # 检查梯度张量的维度大小是否符合预期，第一个参数 gradOutput 是梯度张量，
    # ndim 是张量的总维度数，ndim - 1 是梯度张量在总维度中的索引位置，outputWidth 是期望的大小
    check_dim_size(gradOutput, ndim, ndim - 1, outputWidth)
# 不要覆盖 C++ 注册。
@register_meta(aten.avg_pool2d_backward.default)
def meta_avg_pool2d_backward(
    gradOutput_,
    input,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
):
    # 从 aten/src/ATen/native/AveragePool2d.cpp 结构化内核元函数。
    torch._check(
        len(kernel_size) == 1 or len(kernel_size) == 2,
        lambda: "avg_pool2d: kernel_size must either be a single int, or a tuple of two ints",
    )
    kH = kernel_size[0]  # 提取池化核大小的高度
    kW = kH if len(kernel_size) == 1 else kernel_size[1]  # 提取池化核大小的宽度
    torch._check(
        len(stride) == 0 or len(stride) == 1 or len(stride) == 2,
        lambda: "avg_pool2d: stride must either be omitted, a single int, or a tuple of two ints",
    )
    dH = kH if len(stride) == 0 else stride[0]  # 提取池化步长的高度
    dW = kW if len(stride) == 0 else dH if len(stride) == 1 else stride[1]  # 提取池化步长的宽度
    torch._check(
        len(padding) == 1 or len(padding) == 2,
        lambda: "avg_pool2d: padding must either be a single int, or a tuple of two ints",
    )
    padH = padding[0]  # 提取池化填充的高度
    padW = padH if len(padding) == 1 else padding[1]  # 提取池化填充的宽度

    torch._check(
        divisor_override is None or divisor_override != 0,
        lambda: "divisor must be not zero",
    )

    input_size = input.shape  # 获取输入张量的形状
    nbatch = input_size[-4] if input.dim() == 4 else 1  # 获取批次大小
    nInputPlane = input_size[-3]  # 获取输入平面数
    inputHeight = input_size[-2]  # 获取输入图像的高度
    inputWidth = input_size[-1]  # 获取输入图像的宽度

    outputHeight = pooling_output_shape(inputHeight, kH, padH, dH, 1, ceil_mode)  # 计算池化后的输出高度
    outputWidth = pooling_output_shape(inputWidth, kW, padW, dW, 1, ceil_mode)  # 计算池化后的输出宽度

    mem_format = utils.suggest_memory_format(input)  # 根据输入建议内存格式

    avg_pool2d_backward_shape_check(
        input,
        gradOutput_,
        nbatch,
        kH,
        kW,
        dH,
        dW,
        padH,
        padW,
        nInputPlane,
        inputHeight,
        inputWidth,
        outputHeight,
        outputWidth,
        mem_format,
    )

    return torch.empty(
        input_size,
        dtype=input.dtype,
        device=input.device,
        memory_format=mem_format,
    )


@register_meta(aten.avg_pool3d)
@out_wrapper()
def meta_avg_pool3d(
    input,
    kernel_size,
    stride=(),
    padding=(0,),
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    torch._check(
        len(kernel_size) in (1, 3),
        lambda: "avg_pool3d: kernel_size must be a single int, or a tuple of three ints",
    )
    kT = kernel_size[0]  # 提取池化核大小的时间维度
    kH = kT if len(kernel_size) == 1 else kernel_size[1]  # 提取池化核大小的高度
    kW = kT if len(kernel_size) == 1 else kernel_size[2]  # 提取池化核大小的宽度

    torch._check(
        not stride or len(stride) in (1, 3),
        lambda: "avg_pool3d: stride must be omitted, a single int, or a tuple of three ints",
    )
    dT = kT if not stride else stride[0]  # 提取池化步长的时间维度
    dH = kH if not stride else (dT if len(stride) == 1 else stride[1])  # 提取池化步长的高度
    dW = kW if not stride else (dT if len(stride) == 1 else stride[2])  # 提取池化步长的宽度
    # 检查 padding 参数的长度，必须是1或3，否则抛出错误信息
    torch._check(
        len(padding) in (1, 3),
        lambda: "avg_pool3d: padding must be a single int, or a tuple of three ints",
    )
    # 根据 padding 的长度选择性地获取 padT, padH, padW 的数值
    padT = padding[0]
    padH = padT if len(padding) == 1 else padding[1]
    padW = padT if len(padding) == 1 else padding[2]

    # 检查输入张量的维度，必须是4或5，否则抛出错误信息
    torch._check(
        input.ndim in (4, 5),
        lambda: "non-empty 4D or 5D (batch mode) tensor expected for input",
    )

    # 检查 divisor_override 是否为0，若为0且其值不为 None，则抛出错误信息
    torch._check(
        not divisor_override or divisor_override != 0,
        lambda: "divisor must be not zero",
    )

    # 获取输入张量的批次数、通道数、时间维度、高度和宽度
    nbatch = input.size(0)
    nslices = input.size(-4)
    itime = input.size(-3)
    iheight = input.size(-2)
    iwidth = input.size(-1)

    # 计算池化后的输出时间、高度和宽度的形状
    otime = pooling_output_shape(itime, kT, padT, dT, 1, ceil_mode)
    oheight = pooling_output_shape(iheight, kH, padH, dH, 1, ceil_mode)
    owidth = pooling_output_shape(iwidth, kW, padW, dW, 1, ceil_mode)

    # 检查池化函数的输入和输出形状是否符合预期，若不符合则抛出错误信息
    pool3d_shape_check(
        input,
        nslices,
        kT,
        kH,
        kW,
        dT,
        dH,
        dW,
        padT,
        padH,
        padW,
        1,
        1,
        1,
        itime,
        iheight,
        iwidth,
        otime,
        oheight,
        owidth,
        "avg_pool3d()",
        check_input_size=True,
    )

    # 根据输入张量的维度返回相应形状的空张量作为输出
    if input.ndim == 4:
        return input.new_empty((nslices, otime, oheight, owidth))
    else:
        return input.new_empty((nbatch, nslices, otime, oheight, owidth))
@register_meta(aten.avg_pool3d_backward)
@out_wrapper("grad_input")
def meta_avg_pool3d_backward(
    grad_output,
    input,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
):
    # 检查 kernel_size 的长度是否为 1 或 3，如果不是则抛出异常
    torch._check(
        len(kernel_size) in (1, 3),
        lambda: "avg_pool3d: kernel_size must be a single int, or a tuple of three ints",
    )
    kT = kernel_size[0]
    kH = kT if len(kernel_size) == 1 else kernel_size[1]
    kW = kT if len(kernel_size) == 1 else kernel_size[2]

    # 检查 stride 是否为空或长度为 1 或 3，如果不是则抛出异常
    torch._check(
        not stride or len(stride) in (1, 3),
        lambda: "avg_pool3d: stride must be omitted, a single int, or a tuple of three ints",
    )
    dT = kT if not stride else stride[0]
    dH = kH if not stride else (dT if len(stride) == 1 else stride[1])
    dW = kW if not stride else (dT if len(stride) == 1 else stride[2])

    # 检查 padding 的长度是否为 1 或 3，如果不是则抛出异常
    torch._check(
        len(padding) in (1, 3),
        lambda: "avg_pool3d: padding must be a single int, or a tuple of three ints",
    )
    padT = padding[0]
    padH = padT if len(padding) == 1 else padding[1]
    padW = padT if len(padding) == 1 else padding[2]

    # 检查 input 的维度是否为 4 或 5，如果不是则抛出异常
    torch._check(
        input.ndim in (4, 5),
        lambda: "non-empty 4D or 5D (batch mode) tensor expected for input",
    )

    # 检查 divisor_override 是否为空或不为零，如果是则抛出异常
    torch._check(
        not divisor_override or divisor_override != 0,
        lambda: "divisor must be not zero",
    )

    # 获取 input 的各个维度大小
    nslices = input.size(-4)
    itime = input.size(-3)
    iheight = input.size(-2)
    iwidth = input.size(-1)

    # 计算用于形状检查的输出时间、高度、宽度
    otime_for_shape_check = pooling_output_shape(itime, kT, padT, dT, 1, ceil_mode)
    oheight_for_shape_check = pooling_output_shape(iheight, kH, padH, dH, 1, ceil_mode)
    owidth_for_shape_check = pooling_output_shape(iwidth, kW, padW, dW, 1, ceil_mode)

    # 进行平均池化反向传播的形状检查
    avg_pool3d_backward_shape_check(
        input,
        grad_output,
        nslices,
        kT,
        kH,
        kW,
        dT,
        dH,
        dW,
        padT,
        padH,
        padW,
        itime,
        iheight,
        iwidth,
        otime_for_shape_check,
        oheight_for_shape_check,
        owidth_for_shape_check,
        "avg_pool3d_backward()",
    )

    # 返回一个与 input 形状相同的空张量
    return input.new_empty(input.shape)


@register_meta(aten._adaptive_avg_pool2d.default)
def meta_adaptive_avg_pool2d(self, output_size):
    # 检查 self 的维度是否为 3 或 4，如果不是则抛出异常
    torch._check(
        self.ndim == 3 or self.ndim == 4,
        lambda: f"Expected 3D or 4D tensor, but got {self.shape}",
    )
    output_shape = self.shape[:-2] + tuple(output_size)
    memory_format = utils.suggest_memory_format(self)
    # 需要设置 memory_format 以保持输入的内存格式
    # 输入的通道顺序应该与输出的通道顺序相同
    return torch.empty(
        output_shape,
        dtype=self.dtype,
        device=self.device,
        memory_format=memory_format,
    )


@register_meta(aten._adaptive_avg_pool3d.default)
def meta_adaptive_avg_pool3d(self, output_size):
    # 待实现的函数，暂无内容
    # 检查张量的维度是否为4维或5维，否则抛出异常
    torch._check(
        self.ndim == 4 or self.ndim == 5,
        lambda: f"Expected 4D or 5D tensor, but got {self.shape}",
    )
    # 返回一个与当前张量相同类型和设备的新空张量，形状为当前张量去掉最后三个维度加上指定的输出大小
    return self.new_empty(self.shape[:-3] + tuple(output_size))
@register_meta(aten._adaptive_avg_pool2d_backward.default)
# 注册一个元数据处理函数，处理aten._adaptive_avg_pool2d_backward.default函数
def meta__adaptive_avg_pool2d_backward(grad_out, self):
    # 计算grad_out张量的维度数
    ndim = grad_out.ndim
    # 遍历除了批次维度之外的所有维度，检查它们是否大于零
    for i in range(1, ndim):
        torch._check(
            grad_out.size(i) > 0,
            lambda: f"adaptive_avg_pool2d_backward(): Expected grad_output to have non-zero \
                      size for non-batch dimensions, {grad_out.shape} with dimension {i} being empty",
        )
    # 检查self张量的维度是否为3或4
    torch._check(
        ndim == 3 or ndim == 4,
        lambda: f"adaptive_avg_pool2d_backward(): Expected 3D or 4D tensor, but got {self.shape}",
    )
    # 检查grad_out的数据类型是否与self的数据类型相同
    torch._check(
        self.dtype == grad_out.dtype,
        lambda: f"expected dtype {self.dtype} for `grad_output` but got dtype {grad_out.dtype}",
    )
    # 设置内存格式为torch.contiguous_format，除非self具有通道为最后一维的格式
    memory_format = torch.contiguous_format
    if is_channels_last(self):
        memory_format = torch.channels_last
    # 返回一个与self形状相同的新空张量，并按指定的内存格式进行配置
    return self.new_empty(self.shape).to(memory_format=memory_format)


@register_meta(aten._adaptive_avg_pool3d_backward)
# 注册一个元数据处理函数，处理aten._adaptive_avg_pool3d_backward函数，并且使用grad_input作为输出
@out_wrapper("grad_input")
def meta__adaptive_avg_pool3d_backward(grad_output, self):
    # 调用内部函数检查grad_output是否为空
    _adaptive_pool_empty_output_check(grad_output, "adaptive_avg_pool3d_backward")
    # 返回一个与self具有相同形状的新空张量，并使用torch.legacy_contiguous_format内存格式
    return torch.empty_like(self, memory_format=torch.legacy_contiguous_format)


def _adaptive_pool_empty_output_check(grad_output: Tensor, arg_name: str):
    # 获取grad_output张量的维度数
    ndim = grad_output.ndim
    # 遍历除了批次维度之外的所有维度，检查它们是否大于零
    for i in range(1, ndim):
        torch._check(
            grad_output.size(i) > 0,
            lambda: (
                f"{arg_name}(): Expected grad_output to have non-zero size for non-batch dimensions, "
                f"but grad_output has sizes {grad_output.shape} with dimension {i} being empty"
            ),
        )


@register_meta(aten.adaptive_max_pool2d)
# 注册一个元数据处理函数，处理aten.adaptive_max_pool2d函数，并将其包装为out类型输出
@out_wrapper("out", "indices")
def meta_adaptive_max_pool2d(input, output_size):
    # 获取输入张量的维度数
    ndim = input.ndim
    # 检查输入张量的维度是否为3或4
    torch._check(
        ndim in (3, 4),
        lambda: f"adaptive_max_pool2d(): Expected 3D or 4D tensor, but got: {input.shape}",
    )
    # 遍历除了批次维度之外的所有维度，检查它们是否大于零
    for i in range(1, ndim):
        torch._check(
            input.size(i) > 0,
            lambda: (
                f"adaptive_max_pool2d(): Expected input to have non-zero size for non-batch dimensions, "
                f"but input has sizes {input.shape} with dimension {i} being empty"
            ),
        )
    # 检查output_size是否包含两个元素
    torch._check(
        len(output_size) == 2,
        lambda: "adaptive_max_pool2d(): internal error: output_size.size() must be 2",
    )

    dimH = 1
    sizeB = 1
    sizeD = 0

    # 如果输入张量的维度为4，则更新sizeB和dimH
    if input.ndim == 4:
        sizeB = input.size(0)
        dimH += 1

    # 更新sizeD为输入张量的最后一个维度的大小
    sizeD = input.size(dimH - 1)
    # 从output_size中提取osizeH和osizeW
    osizeH, osizeW = output_size

    # 如果输入张量的维度为3，创建形状为(sizeD, osizeH, osizeW)的新空张量和索引张量
    if input.ndim == 3:
        out_shape = (sizeD, osizeH, osizeW)
        out = input.new_empty(out_shape)
        indices = input.new_empty(out_shape, dtype=torch.int64)
        return out, indices
    else:
        # 定义输出张量的形状为(sizeB, sizeD, osizeH, osizeW)，忽略类型检查
        out_shape = (sizeB, sizeD, osizeH, osizeW)  # type: ignore[assignment]
        # 根据输入张量建议的内存格式，获取建议的内存格式
        memory_format = utils.suggest_memory_format(input)
        # 创建一个新的空张量，形状为out_shape，并指定内存格式
        out = input.new_empty(out_shape).to(memory_format=memory_format)
        # 创建一个新的空索引张量，形状与out相同，数据类型为torch.int64，并指定内存格式
        indices = input.new_empty(out_shape, dtype=torch.int64).to(
            memory_format=memory_format
        )
        # 返回创建的输出张量和索引张量
        return out, indices
@register_meta(aten.adaptive_max_pool2d_backward)
@out_wrapper("grad_input")
def meta_adaptive_max_pool2d_backward(grad_output, input, indices):
    # 获取 grad_output 的维度
    ndim = grad_output.ndim
    # 检查 grad_output 的维度是否为 3 或 4
    torch._check(
        ndim in (3, 4),
        lambda: f"adaptive_max_pooling2d_backward(): Expected 3D or 4D grad_output, but got: {grad_output.shape}",
    )

    # 检查输出是否为空，针对 adaptive_max_pool2d_backward
    _adaptive_pool_empty_output_check(grad_output, "adaptive_max_pool2d_backward")

    # 检查 input 和 grad_output 的数据类型是否相同
    torch._check(
        input.dtype == grad_output.dtype,
        lambda: f"expected dtype {input.dtype} for `grad_output` but got dtype {grad_output.dtype}",
    )

    # 推荐 input 的内存格式
    memory_format = utils.suggest_memory_format(input)
    # 返回一个具有 input 形状的新空张量，并指定内存格式
    return input.new_empty(input.shape).to(memory_format=memory_format)


@register_meta(aten.adaptive_max_pool3d)
@out_wrapper("out", "indices")
def meta_adaptive_max_pool3d(input, output_size):
    # 获取输入张量的维度
    ndim = input.ndim
    # 检查输入张量的维度是否为 4 或 5
    torch._check(
        ndim in (4, 5),
        lambda: f"adaptive_max_pool3d(): Expected 4D or 5D tensor, but got: {input.shape}",
    )
    # 遍历非批处理维度，并确保其大小大于零
    for i in range(1, ndim):
        torch._check(
            input.size(i) > 0,
            lambda: (
                f"adaptive_max_pool3d(): Expected input to have non-zero size for non-batch dimensions, "
                f"but input has sizes {input.shape} with dimension {i} being empty"
            ),
        )

    # 检查输出大小是否为三元组
    torch._check(
        len(output_size) == 3,
        lambda: "adaptive_max_pool3d(): internal error: output_size.size() must be 3",
    )

    dimD = 0
    sizeB = 1
    sizeD = 0

    # 如果输入张量的维度为 5，则获取批处理大小和 dimD 的值
    if ndim == 5:
        sizeB = input.size(0)
        dimD += 1

    sizeD = input.size(dimD)
    osizeT, osizeH, osizeW = output_size

    # 根据输入张量的维度，确定输出张量的形状
    if ndim == 4:
        out_shape = (sizeD, osizeT, osizeH, osizeW)
    else:
        out_shape = (sizeB, sizeD, osizeT, osizeH, osizeW)  # type: ignore[assignment]

    # 返回一个具有指定形状的新空张量和索引张量
    out = input.new_empty(out_shape)
    indices = input.new_empty(out_shape, dtype=torch.int64)

    return out, indices


@register_meta(aten.adaptive_max_pool3d_backward)
@out_wrapper("grad_input")
def meta_adaptive_max_pool3d_backward(grad_output, input, indices):
    # 检查输出是否为空，针对 adaptive_max_pool3d_backward
    _adaptive_pool_empty_output_check(grad_output, "adaptive_max_pool3d_backward")
    # 返回一个具有 input 形状的新空张量作为梯度输入
    return input.new_empty(input.shape)


@register_meta(aten.repeat_interleave.Tensor)
def meta_repeat_interleave_Tensor(repeats, output_size=None):
    # 如果没有提供 output_size，则抛出运行时错误
    if output_size is None:
        raise RuntimeError("cannot repeat_interleave a meta tensor without output_size")
    # 返回一个具有指定输出大小的新空张量
    return repeats.new_empty(output_size)


@register_meta([aten.complex.default, aten.complex.out])
@out_wrapper()
def meta_complex(real, imag):
    # 断言实部和虚部的数据类型为浮点型
    assert real.dtype.is_floating_point
    assert imag.dtype.is_floating_point
    # 计算广播后的输出形状
    out_shape = _broadcast_shapes(real.shape, imag.shape)
    # 返回一个具有指定形状和对应复数数据类型的新空张量
    return real.new_empty(out_shape, dtype=corresponding_complex_dtype(real.dtype))


@register_meta([aten.nonzero_static.default, aten.nonzero_static.out])
@out_wrapper()
def nonzero_static(self, *, size: int, fill_value: int = -1):
    # 这个函数尚未提供具体的实现，因此没有注释
    pass
    # 返回一个新的形状为 (size, self.dim()) 的长整型张量
    return self.new_empty((size, self.dim()), dtype=torch.long)
@register_meta([aten.index.Tensor, aten._unsafe_index.Tensor])
# 注册元信息处理函数，接受aten.index.Tensor和aten._unsafe_index.Tensor类型的输入参数
def meta_index_Tensor(self, indices):
    # 检查indices至少包含一个索引，否则抛出异常
    torch._check(bool(indices), lambda: "at least one index must be provided")

    # aten::index 是内部的高级索引实现
    # checkIndexTensorTypes 和 expandTensors

    result: List[Optional[Tensor]] = []
    # 初始化结果列表，用于存储处理后的索引结果

    for i, index in enumerate(indices):
        # 遍历索引列表，同时追踪索引的序号i和具体的索引对象index

        if index is not None:
            # 如果索引不为None，则进行类型检查
            torch._check(
                index.dtype in [torch.long, torch.int, torch.int8, torch.bool],
                lambda: "tensors used as indices must be long, int, byte or bool tensors",
            )

            if index.dtype in [torch.int8, torch.bool]:
                # 如果索引类型是torch.int8或torch.bool，则处理非零索引
                nonzero = index.nonzero()
                k = len(result)
                # 检查索引维度是否超出当前张量的维度
                torch._check_index(
                    k + index.ndim <= self.ndim,
                    lambda: f"too many indices for tensor of dimension {self.ndim}",
                )

                for j in range(index.ndim):
                    # 检查索引形状是否与张量的形状匹配
                    torch._check_index(
                        index.shape[j] == self.shape[k + j],
                        lambda: f"The shape of the mask {index.shape} at index {i} "
                        f"does not match the shape of the indexed tensor {self.shape} at index {k + j}",
                    )
                    result.append(nonzero.select(1, j))
            else:
                # 否则直接将索引添加到结果列表中
                result.append(index)
        else:
            # 如果索引为None，则直接将None添加到结果列表中
            result.append(index)

    indices = result
    # 更新indices为处理后的结果列表

    torch._check(
        len(indices) <= self.ndim,
        lambda: f"too many indices for tensor of dimension {self.ndim} (got {len(indices)})",
    )

    # expand_outplace
    import torch._refs as refs  # 避免在mypy中出现循环导入

    # 使用refs._maybe_broadcast对indices进行可能的广播操作
    indices = list(refs._maybe_broadcast(*indices))

    # add missing null tensors
    # 补充缺失的空张量
    while len(indices) < self.ndim:
        indices.append(None)

    # hasContiguousSubspace
    # 如果所有非空张量都是相邻的，则为True
    # 参考：
    # https://numpy.org/doc/stable/user/basics.indexing.html#combining-advanced-and-basic-indexing
    # https://stackoverflow.com/questions/53841497/why-does-numpy-mixed-basic-advanced-indexing-depend-on-slice-adjacency
    state = 0
    has_contiguous_subspace = False
    for index in indices:
        if state == 0:
            if index is not None:
                state = 1
        elif state == 1:
            if index is None:
                state = 2
        else:
            if index is not None:
                break
    else:
        has_contiguous_subspace = True

    # transposeToFront
    # 如果新插入的维度不是连续的，则此逻辑将导致它们出现在张量的开头
    # 如果没有连续的子空间索引，需要重新排列张量和索引
    if not has_contiguous_subspace:
        # 初始化空列表，用于存储要重新排列的维度和索引
        dims = []
        transposed_indices = []

        # 遍历索引列表，将非空的索引所在维度加入 dims 和相应索引加入 transposed_indices
        for i, index in enumerate(indices):
            if index is not None:
                dims.append(i)
                transposed_indices.append(index)

        # 再次遍历索引列表，将空索引所在维度加入 dims 和空索引本身加入 transposed_indices
        for i, index in enumerate(indices):
            if index is None:
                dims.append(i)
                transposed_indices.append(index)

        # 使用 dims 对张量进行重新排列
        self = self.permute(dims)
        # 更新 indices 为重新排列后的索引
        indices = transposed_indices

    # 初始化三个空列表，用于存储不同部分的形状信息
    before_shape: List[int] = []
    after_shape: List[int] = []
    replacement_shape: List[int] = []

    # 遍历索引和对应维度，确定每个维度在结果张量的哪个部分
    for dim, index in enumerate(indices):
        if index is None:
            # 如果索引为空，根据 replacement_shape 是否已有值将该维度加入 before_shape 或 after_shape
            if replacement_shape:
                after_shape.append(self.shape[dim])
            else:
                before_shape.append(self.shape[dim])
        else:
            # 如果索引非空，将其形状加入 replacement_shape
            replacement_shape = list(index.shape)

    # 返回一个新的空张量，形状由 before_shape、replacement_shape 和 after_shape 组成
    return self.new_empty(before_shape + replacement_shape + after_shape)
# 注册一个元数据函数，用于处理反向卷积的梯度计算
@register_meta([aten.convolution_backward.default])
def meta_convolution_backward(
    grad_output_,
    input_,
    weight_,
    bias_sizes_opt,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
):
    # 从 slow_conv3d_backward_cpu 中采用的高级逻辑，代表了所有 convolution_backward 实现的典型方法
    backend_grad_input = None
    backend_grad_weight = None
    backend_grad_bias = None

    # 根据 output_mask 的不同标志位，创建相应形状的新 tensor
    if output_mask[0]:
        backend_grad_input = grad_output_.new_empty(input_.size())
    if output_mask[1]:
        backend_grad_weight = grad_output_.new_empty(weight_.size())
    if output_mask[2]:
        backend_grad_bias = grad_output_.new_empty(bias_sizes_opt)

    # 返回包含梯度信息的元组
    return (backend_grad_input, backend_grad_weight, backend_grad_bias)


# 注册一个元数据函数，处理 addbmm 操作，支持输出包装器
@register_meta([aten.addbmm.default, aten.addbmm.out])
@out_wrapper()
def meta_addbmm(self, batch1, batch2, *, beta=1, alpha=1):
    dim1 = batch1.size(1)
    dim2 = batch2.size(2)
    # 扩展当前对象 self 的形状以匹配输出维度
    self = self.expand((dim1, dim2))
    # 检查 batch1 和 batch2 是否为 3D 张量
    torch._check(batch1.dim() == 3, lambda: "batch1 must be a 3D tensor")
    torch._check(batch2.dim() == 3, lambda: "batch2 must be a 3D tensor")
    # 检查 batch1 和 batch2 的批次数是否相同
    torch._check(
        batch1.size(0) == batch2.size(0),
        lambda: f"batch1 and batch2 must have same number of batches, got {batch1.size(0)} and {batch2.size(0)}",
    )
    # 检查 batch1 和 batch2 的矩阵乘法维度是否匹配
    torch._check(
        batch1.size(2) == batch2.size(1),
        lambda: (
            f"Incompatible matrix sizes for bmm ({batch1.size(1)}x{batch1.size(2)} "
            f"and {batch2.size(1)}x{batch2.size(2)})"
        ),
    )
    # 检查 self 的形状是否与矩阵乘法输出一致
    torch._check(
        self.size(0) == dim1 and self.size(1) == dim2,
        lambda: "self tensor does not match matmul output shape",
    )
    # 返回一个与 self 形状相同的新空 tensor
    return self.new_empty(self.size())


# 注册一个元数据函数，处理 _fused_adam_ 操作
@register_meta([aten._fused_adam_.default])
def meta__fused_adam_(
    self,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    *,
    lr,
    beta1,
    beta2,
    weight_decay,
    eps,
    amsgrad,
    maximize,
    grad_scale=None,
    found_inf=None,
):
    # 检查传入参数列表是否均为 tensor 列表
    for l in [self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps]:
        torch._check(
            isinstance(l, List),
            lambda: f"exponent must be a tensor list but got {type(l)}",
        )


# 注册一个元数据函数，处理 _fused_adam 操作
@register_meta([aten._fused_adam.default])
def meta__fused_adam(
    self,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    *,
    lr,
    beta1,
    beta2,
    weight_decay,
    eps,
    amsgrad,
    maximize,
    grad_scale=None,
    found_inf=None,
):
    # 检查传入参数列表是否均为 tensor 列表
    for l in [self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps]:
        torch._check(
            isinstance(l, List),
            lambda: f"exponent must be a tensor list but got {type(l)}",
        )

    # 定义一个函数，返回一个与输入 tensor_list 形状相同的空 tensor 列表
    def empty_like_list(tensor_list):
        return [torch.empty_like(t) for t in tensor_list]
    # 返回一个包含多个空列表的元组，这些列表的结构与传入的对象类似
    return (
        empty_like_list(self),         # 创建一个类似 self 结构的空列表
        empty_like_list(grads),        # 创建一个类似 grads 结构的空列表
        empty_like_list(exp_avgs),     # 创建一个类似 exp_avgs 结构的空列表
        empty_like_list(exp_avg_sqs),  # 创建一个类似 exp_avg_sqs 结构的空列表
        empty_like_list(max_exp_avg_sqs),  # 创建一个类似 max_exp_avg_sqs 结构的空列表
    )
# 注册元信息，指定对 torch 中 _int_mm 函数的元信息装饰
@register_meta([aten._int_mm])
# 输出装饰器的包装函数
@out_wrapper()
# 定义函数 meta__int_mm，接受两个参数 a 和 b
def meta__int_mm(a, b):
    # 检查 a 必须是一个二维张量
    torch._check(a.dim() == 2, lambda: "a must be a 2D tensor")
    # 检查 b 必须是一个二维张量
    torch._check(b.dim() == 2, lambda: "b must be a 2D tensor")
    # 检查 a 的数据类型必须是 int8
    torch._check(
        a.dtype is torch.int8,
        lambda: f"expected self to be int8, got {a.dtype}",
    )
    # 检查 b 的数据类型必须是 int8
    torch._check(
        b.dtype is torch.int8,
        lambda: f"expected mat2 to be int8, got {b.dtype}",
    )
    # 检查 a 的列数必须等于 b 的行数
    torch._check(
        a.size(1) == b.size(0),
        lambda: (
            f"Incompatible matrix sizes for _int_mm ({a.size(0)}x{a.size(1)} "
            f"and {b.size(0)}x{b.size(1)})"
        ),
    )
    # 返回一个与 a 形状相同的新的空张量，数据类型为 int32
    return a.new_empty((a.size(0), b.size(1)), dtype=torch.int32)


# 注册元信息，指定对 torch 中 _convert_weight_to_int4pack 函数的元信息装饰
@register_meta([aten._convert_weight_to_int4pack])
# 定义函数 meta__convert_weight_to_int4pack，接受两个参数 w 和 inner_k_tiles
def meta__convert_weight_to_int4pack(w, inner_k_tiles):
    # 检查 w 必须是一个二维张量
    torch._check(w.dim() == 2, lambda: "w must be a 2D tensor")
    # 检查 w 的数据类型必须是 int32
    torch._check(
        w.dtype is torch.int32,
        lambda: f"expected w to be int32, got {w.dtype}",
    )
    # 获取 w 的行数和列数
    n = w.size(0)
    k = w.size(1)
    # 返回一个新的空张量，形状为 (n // 8, k // (inner_k_tiles * 16), 32, inner_k_tiles // 2)，数据类型为 int32
    return w.new_empty(
        (
            n // 8,
            k // (inner_k_tiles * 16),
            32,
            inner_k_tiles // 2,
        ),
        dtype=torch.int32,
    )


# 注册元信息，指定对 torch 中 _weight_int4pack_mm 函数的元信息装饰
@register_meta([aten._weight_int4pack_mm])
# 定义函数 meta__weight_int4pack_mm，接受四个参数 x, w, q_group_size 和 q_scale_and_zeros
def meta__weight_int4pack_mm(x, w, q_group_size, q_scale_and_zeros):
    # 检查 x 必须是一个二维张量
    torch._check(x.dim() == 2, lambda: "x must be a 2D tensor")
    # 检查 w 必须是一个四维张量
    torch._check(w.dim() == 4, lambda: "w must be a 4D tensor")
    # 检查 x 的数据类型必须是 torch.float32, torch.float16 或 torch.bfloat16
    torch._check(
        x.dtype in [torch.float32, torch.float16, torch.bfloat16],
        lambda: f"expected x to be f32/f16/bf16, got {x.dtype}",
    )
    # 检查 w 的数据类型必须是 int32
    torch._check(
        w.dtype is torch.int32,
        lambda: f"expected w to be int32, got {w.dtype}",
    )
    # 返回一个新的空张量，形状为 (x.size(0), w.size(0) * 8)，数据类型与 x 相同
    return x.new_empty(x.size(0), w.size(0) * 8, dtype=x.dtype)


# 注册元信息，指定对 torch 中 _weight_int8pack_mm 函数的元信息装饰
@register_meta([aten._weight_int8pack_mm])
# 定义函数 meta__weight_int8pack_mm，接受三个参数 x, w 和 q_scales
def meta__weight_int8pack_mm(x, w, q_scales):
    # 检查 x 必须是一个二维张量
    torch._check(x.dim() == 2, lambda: "x must be a 2D tensor")
    # 检查 x 的数据类型必须是 torch.float32, torch.float16 或 torch.bfloat16
    torch._check(
        x.dtype in [torch.float32, torch.float16, torch.bfloat16],
        lambda: f"expected x to be f32/f16/bf16, got {x.dtype}",
    )
    # 检查 w 必须是一个二维张量
    torch._check(w.dim() == 2, lambda: "w must be a 2D tensor")
    # 检查 w 的数据类型必须是 int8
    torch._check(
        w.dtype is torch.int8,
        lambda: f"expected w to be int8, got {w.dtype}",
    )
    # 返回一个新的空张量，形状为 (x.size(0), w.size(0))，数据类型与 x 相同
    return x.new_empty(x.size(0), w.size(0), dtype=x.dtype)


# 注册元信息，指定对 aten._cdist_forward.default 中的函数的元信息装饰
@register_meta(aten._cdist_forward.default)
# 定义函数 meta_cdist_forward，接受四个参数 x1, x2, p 和 compute_mode
def meta_cdist_forward(x1, x2, p, compute_mode):
    # 检查 x1 必须至少是一个二维张量
    torch._check(
        x1.dim() >= 2,
        lambda: f"cdist only supports at least 2D tensors, X1 got: {x1.dim()}D",
    )
    # 检查 x2 必须至少是一个二维张量
    torch._check(
        x2.dim() >= 2,
        lambda: f"cdist only supports at least 2D tensors, X2 got: {x2.dim()}D",
    )
    # 检查 x1 和 x2 的最后一个维度必须相等
    torch._check(
        x1.size(-1) == x2.size(-1),
        lambda: f"X1 and X2 must have the same number of columns. X1: {x1.size(-1)} X2: {x2.size(-1)}",
    )
    # 检查 x1 的数据类型必须是浮点数类型
    torch._check(
        utils.is_float_dtype(x1.dtype),
        lambda: "cdist only supports floating-point dtypes, X1 got: {x1.dtype}",
    )
    # 检查 x2 的数据类型是否为浮点型
    torch._check(
        utils.is_float_dtype(x2.dtype),
        lambda: "cdist only supports floating-point dtypes, X2 got: {x2.dtype}",
    )
    # 检查 p 是否为非负值
    torch._check(p >= 0, lambda: "cdist only supports non-negative p values")
    # 检查 compute_mode 是否在 None, 1, 2 中
    torch._check(
        compute_mode in (None, 1, 2),
        lambda: f"possible modes: None, 1, 2, but was: {compute_mode}",
    )
    # 获取 x1 的倒数第二维的大小
    r1 = x1.size(-2)
    # 获取 x2 的倒数第二维的大小
    r2 = x2.size(-2)
    # 获取 x1 所有维度除了倒数两个维度的形状
    batch_tensor1 = x1.shape[:-2]
    # 获取 x2 所有维度除了倒数两个维度的形状
    batch_tensor2 = x2.shape[:-2]
    # 计算输出张量的形状，通过广播 x1 和 x2 的形状，再附加 r1 和 r2 作为最后两个维度
    output_shape = list(torch.broadcast_shapes(batch_tensor1, batch_tensor2))
    output_shape.extend([r1, r2])
    # 返回一个新的空张量，形状为 output_shape，数据类型与 x1 相同
    return x1.new_empty(output_shape)
# 注册元信息函数，用于处理aten._cdist_backward函数
@register_meta(aten._cdist_backward)
# 对输出进行装饰处理
@out_wrapper()
def meta_cdist_backward(grad, x1, x2, p, cdist):
    # 计算第一个张量的列数
    c1 = x1.shape[-1]
    # 计算第一个张量的行数
    r1 = x1.shape[-2]
    # 计算第二个张量的行数
    r2 = x2.shape[-2]
    # 获取第一个张量的批处理维度
    batch_tensor1 = x1.shape[:-2]
    # 获取第二个张量的批处理维度
    batch_tensor2 = x2.shape[:-2]
    # 计算广播后的批处理维度
    expand_batch_portion = list(torch.broadcast_shapes(batch_tensor1, batch_tensor2))
    # 复制扩展后的第一个张量大小
    tensor1_expand_size = expand_batch_portion.copy()
    # 添加行列数到复制的张量大小中
    tensor1_expand_size.extend([r1, c1])
    # 计算批处理的乘积
    batch_product = math.prod(expand_batch_portion)
    # 如果其中一个张量的行或列数为零，或者批处理乘积为零，则返回与x1相同形状的全零张量
    if r1 == 0 or r2 == 0 or c1 == 0 or batch_product == 0:
        return torch.zeros_like(x1)
    # 如果扩展后的第一个张量大小不等于x1的形状，则将x1张量进行扩展
    if tensor1_expand_size != list(x1.shape):
        x1 = x1.expand(tensor1_expand_size)
    # 返回与x1相同形状的新空张量，内存格式为连续格式
    return torch.empty_like(x1, memory_format=torch.contiguous_format)


# 注意：此元信息函数接受非元信息参数！当这个行为最初引入时是偶然的，
# 但现在它是基本功能，因为人们使用它可以方便地测试涉及嵌入（使用CPU张量输入和元设备EmbeddingBag模块）的代码。
@register_meta(aten._embedding_bag.default)
def meta_embedding_bag(
    weight,
    indices,
    offsets,
    scale_grad_by_freq=False,
    mode=0,
    sparse=False,
    per_sample_weights=None,
    include_last_offset=False,
    padding_idx=-1,
):
    # 检查indices张量的数据类型是否为长整型或整型
    torch._check(
        indices.dtype in (torch.long, torch.int),
        lambda: f"expected indices to be long or int, got {indices.dtype}",
    )
    # 检查offsets张量的数据类型是否为长整型或整型
    torch._check(
        offsets.dtype in (torch.long, torch.int),
        lambda: f"expected offsets to be long or int, got {offsets.dtype}",
    )
    # 检查weight张量的数据类型是否为浮点类型
    torch._check(
        utils.is_float_dtype(weight.dtype),
        lambda: f"expected weight to be floating point type, got {weight.dtype}",
    )

    # 计算offsets张量的行数（袋子数量）
    num_bags = offsets.size(0)
    # 如果include_last_offset为True，则检查num_bags是否大于等于1
    if include_last_offset:
        torch._check(
            num_bags >= 1,
            lambda: "include_last_offset: numBags should be at least 1",
        )
        # 如果include_last_offset为True，则将num_bags减1
        num_bags -= 1

    # 创建一个与weight张量形状相同的空张量output
    output = weight.new_empty(num_bags, weight.size(1))
    # 定义常量：MODE_SUM为0，MODE_MEAN为1，MODE_MAX为2
    MODE_SUM, MODE_MEAN, MODE_MAX = range(3)

    # 如果per_sample_weights不为None，则进行一系列检查
    if per_sample_weights is not None:
        # 检查mode是否为MODE_SUM
        torch._check(
            mode == MODE_SUM,
            lambda: "embedding_bag: per_sample_weights only supported with mode='sum'",
        )
        # 检查per_sample_weights张量的数据类型与weight张量的数据类型是否相同
        torch._check(
            per_sample_weights.dtype == weight.dtype,
            lambda: f"expected weight ({weight.dtype}) and per_sample_weights ({per_sample_weights.dtype}) to have same dtype",
        )
        # 检查per_sample_weights张量是否为1维张量
        torch._check(
            per_sample_weights.ndim == 1,
            lambda: f"expected per_sample_weights to be 1D tensor, got {per_sample_weights.ndim}D",
        )
        # 检查per_sample_weights张量的元素数量是否与indices张量的元素数量相同
        torch._check(
            per_sample_weights.numel() == indices.numel(),
            lambda: (
                f"expected per_sample_weights.numel() ({per_sample_weights.numel()} "
                f"to be the same as indices.numel() ({indices.numel()})"
            ),
        )
    # 检查是否可以使用快速路径进行索引选择和缩放
    def is_fast_path_index_select_scale(src, scale, output, padding_idx):
        # 调用is_fast_path_index_select函数，同时检查缩放张量的步长是否为1
        return (
            is_fast_path_index_select(src, output, padding_idx) and scale.stride(0) == 1
        )

    # 检查是否可以使用快速路径进行索引选择
    def is_fast_path_index_select(src, output, padding_idx):
        # 检查源张量数据类型是否为float或half，以及其在第一维度的步长是否为1，
        # 输出张量在第一维度的步长是否为1，以及填充索引是否小于0
        return (
            (src.dtype == torch.float or src.dtype == torch.half)
            and src.stride(1) == 1
            and output.stride(1) == 1
            and padding_idx < 0
        )

    # 检查是否可以使用快速路径
    def is_fast_path(src, scale, output, padding_idx):
        if scale is not None:
            # 如果存在缩放张量，则调用is_fast_path_index_select_scale函数
            return is_fast_path_index_select_scale(src, scale, output, padding_idx)
        else:
            # 否则调用is_fast_path_index_select函数
            return is_fast_path_index_select(src, output, padding_idx)

    # 如果偏移量不是在CPU上，则初始化offset2bag和bag_size
    if device_hint(offsets) != "cpu":
        offset2bag = indices.new_empty(indices.size(0))
        bag_size = indices.new_empty(offsets.size())
        # 根据模式初始化max_indices张量
        if mode == MODE_MAX:
            max_indices = indices.new_empty(num_bags, weight.size(1))
        else:
            max_indices = indices.new_empty(0)
    else:
        # 在CPU上执行快速路径和汇总逻辑
        fast_path_sum = is_fast_path(weight, per_sample_weights, output, padding_idx)
        # 根据模式和快速路径决定初始化offset2bag、bag_size和max_indices张量
        if mode in (MODE_MEAN, MODE_MAX) or not fast_path_sum:
            offset2bag = offsets.new_empty(indices.size(0))
        else:
            offset2bag = offsets.new_empty(0)
        bag_size = offsets.new_empty(num_bags)
        # 根据模式和include_last_offset初始化max_indices张量，参考EmbeddingBag.cpp中的逻辑
        numBags = offsets.shape[0]
        if mode == MODE_MAX:
            if include_last_offset:
                torch._check(
                    numBags >= 1,
                    lambda: "include_last_offset: numBags should be at least 1",
                )
                numBags -= 1
            max_indices = offsets.new_empty(numBags, weight.shape[1])
        else:
            max_indices = offsets.new_empty(bag_size.size())
    # 返回计算结果output、offset2bag、bag_size和max_indices
    return output, offset2bag, bag_size, max_indices
@register_meta(aten._embedding_bag_forward_only.default)
def meta_embedding_bag_forward_only(weight, indices, offsets, *args):
    # 调用 meta_embedding_bag 函数处理权重、索引和偏移量，返回处理后的结果
    output, offset2bag, bag_size, max_indices = meta_embedding_bag(
        weight, indices, offsets, *args
    )
    # 如果偏移量的设备提示为 "cpu"，则创建一个与偏移量相同大小的空张量
    if device_hint(offsets) == "cpu":
        bag_size = offsets.new_empty(offsets.size())
    # 返回处理后的输出、offset2bag映射、bag_size和max_indices
    return output, offset2bag, bag_size, max_indices


def _get_reduction_dtype(input, dtype, promote_int_to_long=True):
    # 如果指定了 dtype，则优先使用 dtype
    if dtype:
        return dtype

    # 如果输入张量的 dtype 是浮点数或复数类型，则返回该 dtype
    elif input.dtype.is_floating_point or input.dtype.is_complex:
        return input.dtype
    # 否则，如果允许整数提升为长整型，则返回 torch.long
    elif promote_int_to_long:
        return torch.long

    # 返回输入张量的 dtype
    return input.dtype


@register_meta([aten.nansum.default, aten.nansum.out])
@out_wrapper()
def meta_nansum(input, dims=None, keepdim=False, *, dtype=None):
    # 根据输入张量和指定的 dtype 计算输出的数据类型
    output_dtype = _get_reduction_dtype(input, dtype, promote_int_to_long=True)
    # 根据输入张量的形状和指定的维度 dims，计算约简后的形状
    dims = utils.reduction_dims(input.shape, dims)
    # 根据约简后的形状计算输出张量的形状
    output_shape = _compute_reduction_shape(input, dims, keepdim)
    # 返回指定形状和数据类型的新空张量
    return input.new_empty(output_shape, dtype=output_dtype)


@register_meta([aten.median.default, aten.nanmedian.default])
def meta_median(input):
    # 根据输入张量的形状计算约简后的输出形状
    output_shape = utils.compute_reduction_output_shape(
        input.shape, tuple(range(input.dim()))
    )
    # 返回指定形状的新空张量
    return input.new_empty(output_shape)


@register_meta(
    [
        aten.median.dim,
        aten.median.dim_values,
        aten.nanmedian.dim,
        aten.nanmedian.dim_values,
        aten.mode.default,
        aten.mode.values,
    ]
)
@out_wrapper("values", "indices")
def meta_median_mode_dim(input, dim=-1, keepdim=False):
    # 如果输入张量的设备提示为 "cuda"，则发出非确定性警告
    if device_hint(input) == "cuda":
        utils.alert_not_deterministic("median CUDA with indices output")
    # 根据输入张量的形状和指定的维度 dim，计算约简后的形状
    dim = utils.reduction_dims(input.shape, (dim,))
    # 根据约简后的形状计算输出张量的形状
    output_shape = _compute_reduction_shape(input, dim, keepdim)
    # 返回两个指定形状的新空张量，分别用于值和索引
    return (
        input.new_empty(output_shape),
        input.new_empty(output_shape, dtype=torch.long),
    )


@register_meta(aten.logical_not_.default)
def meta_logical_not_(self):
    # 返回自身张量，用于逻辑非操作
    return self


@register_meta(aten.repeat.default)
def meta_repeat(self, repeats):
    # 检查重复维度的数量是否大于等于张量的维度数
    torch._check(
        len(repeats) >= self.dim(),
        lambda: "Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor",
    )
    # 如果目标维度数量大于源张量的维度数量，则在张量前面添加新的维度
    num_new_dimensions = len(repeats) - self.dim()
    padded_size = (1,) * num_new_dimensions + tuple(self.shape)
    # 计算目标张量的形状
    target_size = [padded_size[i] * repeats[i] for i in range(len(repeats))]
    # 返回指定形状的新空张量
    return self.new_empty(target_size)


@register_meta(aten.zero_.default)
def meta_zero_(self):
    # 返回自身张量，用于将张量元素清零
    return self


@register_meta(
    [
        aten.mul_.Scalar,
        aten.div_.Scalar,
        aten.mul_.Tensor,
        aten.div_.Tensor,
        aten.logical_and_.default,
        aten.logical_or_.default,
        aten.logical_xor_.default,
    ],
)
# 注册元信息和操作的函数，用于执行原位二元运算（inplace binary operation）
def meta_binop_inplace(self, other):
    # 如果other是torch.Tensor类型，则检查并确保self和other的形状可以进行原位广播操作
    if isinstance(other, torch.Tensor):
        check_inplace_broadcast(self.shape, other.shape)
    # 返回self本身
    return self


# 注册元信息和操作的函数，用于执行带alpha参数的原位二元运算（inplace binary operation with alpha）
@register_meta(
    [
        aten.add_.Scalar,
        aten.sub_.Scalar,
        aten.add_.Tensor,
        aten.sub_.Tensor,
    ],
)
def meta_binop_inplace_alpha(self, other, alpha=1):
    # 如果other是torch.Tensor类型，则检查并确保self和other的形状可以进行原位广播操作
    if isinstance(other, torch.Tensor):
        check_inplace_broadcast(self.shape, other.shape)
    # 返回self本身
    return self


# 注册元信息和操作的函数，用于执行元素级（elementwise）的round操作
@register_meta([aten.round.default, aten.round.decimals])
def meta_round(self, **kwargs):
    return elementwise_meta(
        self, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )


# 检查shift操作的数据类型
def shift_dtype_check(fn_name, self, val):
    # 检查self的数据类型是否为整数类型
    torch._check(
        utils.is_integer_dtype(self.dtype),
        lambda: f"{fn_name}: Expected input tensor to have an integral dtype. Got {self.dtype}",
    )
    # 如果val是torch.Tensor类型，则检查其数据类型是否为整数类型
    if isinstance(val, torch.Tensor):
        torch._check(
            utils.is_integer_dtype(val.dtype),
            lambda: f"{fn_name}: Expected shift value to have an integral dtype. Got {val.dtype}",
        )
    else:
        # 否则，检查val是否为IntLike类型（类似整数的类型）
        torch._check(
            isinstance(val, IntLike),
            lambda: f"{fn_name}: Expected shift value to be an int. Got {val}",
        )


# 注册元信息和操作的函数，用于执行右移操作（>>）
@register_meta([aten.__rshift__.Tensor, aten.__rshift__.Scalar])
def meta_rshifts(self, other):
    # 检查右移操作的数据类型
    shift_dtype_check("rshift", self, other)
    # 返回元素级的操作结果
    return elementwise_meta(
        self, other, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )


# 注册元信息和操作的函数，用于执行左移操作（<<）
@register_meta([aten.__lshift__.Tensor, aten.__lshift__.Scalar])
def meta_lshifts(self, other):
    # 检查左移操作的数据类型
    shift_dtype_check("lshift", self, other)
    # 返回元素级的操作结果
    return elementwise_meta(
        self, other, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )


# 注册元信息和操作的函数，用于执行全零填充操作
@register_meta(aten.zero.default)
def meta_zero(self):
    # 创建并返回一个与self形状相同的空tensor
    return self.new_empty(self.shape)


# 注册元信息和操作的函数，用于执行原位填充操作（inplace fill）
@register_meta([aten.fill_.Tensor, aten.fill_.Scalar])
def meta_fill_(self, val):
    # 返回self本身
    return self


# 注册元信息和操作的函数，用于执行填充操作
@register_meta([aten.fill.Tensor, aten.fill.Scalar])
def meta_fill(self, val):
    # 创建并返回一个与self形状相同的空tensor
    return torch.empty_like(self)


# 注册元信息和操作的函数，用于执行原位ReLU操作
@register_meta(aten.relu_.default)
def meta_relu_(self):
    # 返回self本身
    return self


# 注册元信息和操作的函数，用于执行索引放置操作
@register_meta([aten.index_put.default, aten._unsafe_index_put.default])
def meta_index_put(self, indices, values, accumulate=False):
    # 创建并返回一个与self形状相同的空tensor
    return torch.empty_like(self)


# 注册元信息和操作的函数，用于执行原位掩码填充操作
@register_meta(aten.masked_fill_.Scalar)
def meta_masked_fill_(self, mask, value):
    # 检查原位掩码填充操作的形状是否可以广播
    check_inplace_broadcast(self.shape, mask.shape)
    # 返回self本身
    return self


# 注册元信息和操作的函数，用于执行掩码缩放操作
@register_meta(aten._masked_scale.default)
def meta__masked_scale(self, mask, scale):
    # 创建一个与self形状相同的新tensor，并使用推荐的内存格式
    masked_scale = self.new_empty(self.size()).to(
        memory_format=utils.suggest_memory_format(self)
    )
    return masked_scale


# 注册元信息和操作的函数，用于执行原位掩码散布操作
@register_meta(aten.masked_scatter_)
def meta_masked_scatter_(self, mask, source):
    # 检查掩码是否为布尔类型或8位无符号整数类型
    torch._check(
        mask.dtype in (torch.bool, torch.uint8), lambda: "Mask must be bool or uint8"
    )
    # 调用 torch._check 方法，验证 self 对象的 dtype 与 source 对象的 dtype 是否相同
    torch._check(
        self.dtype == source.dtype,
        lambda: "masked_scatter: expected self and source to have same "
        "dtypes but got {self.dtype} and {source.dtype}",
    )
    # 返回当前对象 self
    return self
@register_meta(aten.masked_scatter)
@out_wrapper()
def meta_masked_scatter(self, mask, source):
    # 将 self 和 mask 可能进行广播处理
    self, mask = _maybe_broadcast(self, mask)
    # 创建一个与 self 具有相同形状和数据类型的空张量
    output = torch.empty_like(self, memory_format=torch.contiguous_format)
    # 调用 meta_masked_scatter_ 函数进行填充操作
    return meta_masked_scatter_(output, mask, source)


@register_meta(aten.masked_scatter_backward)
def meta_masked_scatter_backward(self, mask, sizes):
    # 创建一个新的空张量，形状由 sizes 指定
    return self.new_empty(sizes)


@register_meta(aten.index_put_.default)
def meta_index_put_(self, indices, values, accumulate=False):
    # 直接返回 self 张量
    return self


@register_meta(aten.alias.default)
def meta_alias(self):
    # 返回一个新的张量，其形状与 self 张量相同
    return self.view(self.shape)


def common_meta_baddbmm_bmm(batch1, batch2, is_bmm, self_baddbmm=None):
    # 检查 batch1 和 batch2 张量的维度是否为 3
    torch._check(batch1.dim() == 3, lambda: "batch1 must be a 3D tensor")
    torch._check(batch2.dim() == 3, lambda: "batch2 must be a 3D tensor")

    # 获取 batch1 和 batch2 的尺寸信息
    batch1_sizes = batch1.size()
    batch2_sizes = batch2.size()

    # 提取批次数、压缩大小和结果行列数
    bs = batch1_sizes[0]
    contraction_size = batch1_sizes[2]
    res_rows = batch1_sizes[1]
    res_cols = batch2_sizes[2]
    output_size = (bs, res_rows, res_cols)

    # 检查 batch2 张量的前两个维度是否符合预期
    torch._check(
        batch2_sizes[0] == bs and batch2_sizes[1] == contraction_size,
        lambda: f"Expected size for first two dimensions of batch2 tensor to be: [{bs}"
        f", {contraction_size}] but got: [{batch2_sizes[0]}, {batch2_sizes[1]}].",
    )

    # TODO: handle out

    # 创建一个新的空张量，形状由 output_size 指定
    output = batch2.new_empty(output_size)

    # 如果不是 bmm 操作且 self_baddbmm 不为空，则进一步检查 self_baddbmm 张量的维度和形状
    if not is_bmm and self_baddbmm is not None:
        torch._check(self_baddbmm.dim() == 3, lambda: "self must be a 3D tensor")
        torch._check(
            self_baddbmm.size() == output_size,
            lambda: f"Expected an input tensor shape with shape {output_size} but got shape: {self_baddbmm.size()}",
        )

    # 返回创建的空张量 output
    return output


@register_meta(aten.bmm.default)
def meta_bmm(self, mat2):
    # 调用 common_meta_baddbmm_bmm 函数进行 bmm 操作
    return common_meta_baddbmm_bmm(self, mat2, True)


def div_rtn(x, y):
    # 计算 x 除以 y 的商和余数
    q = x // y
    r = x % y
    # 显式转换 r 为布尔值，需要进行此步骤；SymBool 可能会解决这个问题
    if r != 0 and (bool(r < 0) != bool(y < 0)):
        q -= 1
    return q


def pooling_output_shape_pad_lr(
    inputSize,
    kernelSize,
    pad_l,
    pad_r,
    stride,
    dilation,
    ceil_mode,
):
    # 计算池化层输出的大小，考虑填充、步幅、膨胀和是否向上取整
    outputSize = (
        div_rtn(
            inputSize
            + pad_l
            + pad_r
            - dilation * (kernelSize - 1)
            - 1
            + (stride - 1 if ceil_mode else 0),
            stride,
        )
        + 1
    )
    # 如果 ceil_mode 为 True，进一步调整输出大小
    if ceil_mode:
        if (outputSize - 1) * stride >= inputSize + pad_l:
            outputSize -= 1
    return outputSize


def pooling_output_shape(inputSize, kernelSize, pad, stride, dilation, ceil_mode):
    # 检查步幅是否为零
    torch._check(stride != 0, lambda: "stride should not be zero")
    # 检查填充是否为非负数
    torch._check(pad >= 0, lambda: f"pad must be non-negative, but got pad: {pad}")
    # 调用 torch._check 函数，检查 pad 是否满足条件
    torch._check(
        # 检查 pad 是否小于等于 ((kernelSize - 1) * dilation + 1) 的一半
        pad <= ((kernelSize - 1) * dilation + 1) // 2,
        # 如果条件不满足，生成错误消息字符串，包括当前的 pad、kernelSize 和 dilation 值
        lambda: (
            f"pad should be at most half of effective kernel size, but got pad={pad}, "
            f"kernel_size={kernelSize} and dilation={dilation}"
        ),
    )
    # 返回通过 pooling_output_shape_pad_lr 函数计算得到的池化操作输出形状
    return pooling_output_shape_pad_lr(
        inputSize, kernelSize, pad, pad, stride, dilation, ceil_mode
    )
# 检查输入张量的维度数
ndim = input.dim()

# 输出通道数与输入通道数相同
nOutputPlane = nInputPlane

# 检查卷积核大小是否大于零，如果不是则抛出异常
torch._check(
    kW > 0 and kH > 0,
    lambda: "kernel size should be greater than zero, but got kH: {kH}, kW: {kW}",
)

# 检查步长是否大于零，如果不是则抛出异常
torch._check(
    dW > 0 and dH > 0,
    lambda: "stride should be greater than zero, but got dH: {dH}, dW: {dW}",
)

# 检查扩展（dilation）是否大于零，如果不是则抛出异常
torch._check(
    dilationH > 0 and dilationW > 0,
    lambda: "dilation should be greater than zero, but got dilationH: {dilationH}, dilationW: {dilationW}",
)

# 检查输入张量的有效维度
valid_dims = input.size(1) != 0 and input.size(2) != 0

# 如果内存格式为通道最后，则检查输入张量的维度和大小是否符合预期
if memory_format == torch.channels_last:
    torch._check(
        ndim == 4 and valid_dims and input.size(3) != 0,
        lambda: "Expected 4D (batch mode) tensor expected for input with channels_last layout"
        " with optional 0 dim batch size for input, but got: {input.size()}",
    )
else:
    # 否则，检查输入张量的维度和大小是否符合预期（3D或4D）
    torch._check(
        (ndim == 3 and input.size(0) != 0 and valid_dims)
        or (ndim == 4 and valid_dims and input.size(3) != 0),
        lambda: f"Expected 3D or 4D (batch mode) tensor with optional 0 dim batch size for input, but got: {input.size()}",
    )

# 检查填充大小是否小于等于卷积核大小的一半，如果不是则抛出异常
torch._check(
    kW // 2 >= padW and kH // 2 >= padH,
    lambda: "pad should be smaller than or equal to half of kernel size, but got "
    f"padW = {padW}, padH = {padH}, kW = {kW}, kH = {kH}",
)

# 检查输出高度和宽度是否至少为1，否则抛出异常
torch._check(
    outputWidth >= 1 and outputHeight >= 1,
    lambda: f"Given input size: ({nInputPlane}x{inputHeight}x{inputWidth}). "
    f"Calculated output size: ({nOutputPlane}x{outputHeight}x{outputWidth}). "
    "Output size is too small",
)


# 检查输入张量的维度数
ndim = input.ndim

# 检查三维池化操作的卷积核大小是否大于零，如果不是则抛出异常
torch._check(
    kT > 0 and kW > 0 and kH > 0,
    lambda: (
        f"kernel size should be greater than zero, but got "
        f"kT: {kT}, kH: {kH}, kW: {kW}"
    ),
)

# 检查三维池化操作的步长是否大于零，如果不是则抛出异常
torch._check(
    dT > 0 and dW > 0 and dH > 0,
    lambda: (
        f"stride should be greater than zero, but got "
        f"dT: {dT}, dH: {dH}, dW: {dW}"
    ),
)

# 检查三维池化操作的扩展（dilation）是否大于零，如果不是则抛出异常
torch._check(
    dilationT > 0 and dilationW > 0 and dilationH > 0,
    lambda: (
        f"dilation should be greater than zero, but got "
        f"dilationT: {dilationT}, dilationH: {dilationH}, dilationW: {dilationW}"
    ),
)
    # 检查输入张量的维度是否为4或5，否则抛出错误信息
    torch._check(
        ndim in (4, 5),
        lambda: f"{fn_name}: Expected 4D or 5D tensor for input, but got: {input.shape}",
    )

    # 遍历张量的维度，检查非批处理维度的长度是否大于0，否则抛出错误信息
    for i in range(ndim):
        if ndim == 5 and i == 0:
            # 如果是5维张量且当前维度为批处理维度（第0维），则跳过检查
            continue
        torch._check(
            input.size(i) > 0,
            lambda: (
                f"{fn_name}: Expected input's non-batch dimensions to have positive length,"
                f" but input has a shape of {input.shape}"
                f" and non-batch dimension {input.size(i)} has length zero!"
            ),
        )

    # 如果需要检查输入大小（如在AveragePool3d情况下），确保输入图像大小不小于卷积核大小
    if check_input_size:
        torch._check(
            itime >= kT and iheight >= kH and iwidth >= kW,
            lambda: (
                f"input image (T: {itime} H: {iheight} W: {iwidth}) smaller than "
                f"kernel size (kT: {kT} kH: {kH} kW: {kW})"
            ),
        )

    # 检查填充大小是否符合要求（应小于或等于卷积核大小的一半）
    torch._check(
        kT / 2 >= pT and kW / 2 >= pW and kH / 2 >= pH,
        lambda: (
            f"pad should be smaller than or equal to half of kernel size, but got "
            f"kT: {kT} kW: {kW} kH: {kH} padT: {pT} padW: {pW} padH: {pH}"
        ),
    )

    # 检查输出大小是否至少为1，否则抛出错误信息
    torch._check(
        otime >= 1 and owidth >= 1 and oheight >= 1,
        lambda: (
            f"Given input size: ({nslices}x{itime}x{iheight}x{iwidth}). "
            f"Calculated output size: ({nslices}x{otime}x{oheight}x{owidth}). "
            f"Output size is too small"
        ),
    )
# 检查输入张量的维度数
ndim = input.ndim

# 调用函数，验证池化操作的形状和参数是否符合要求
pool3d_shape_check(
    input,
    nslices,
    kT,
    kH,
    kW,
    dT,
    dH,
    dW,
    pT,
    pH,
    pW,
    dilationT,
    dilationH,
    dilationW,
    itime,
    iheight,
    iwidth,
    otime,
    oheight,
    owidth,
    fn_name,
)

# 检查梯度张量在特定维度上的大小是否符合要求
check_dim_size(grad_output, ndim, ndim - 4, nslices)
check_dim_size(grad_output, ndim, ndim - 3, otime)
check_dim_size(grad_output, ndim, ndim - 2, oheight)
check_dim_size(grad_output, ndim, ndim - 1, owidth)

# 检查索引张量在特定维度上的大小是否符合要求
check_dim_size(indices, ndim, ndim - 4, nslices)
check_dim_size(indices, ndim, ndim - 3, otime)
check_dim_size(indices, ndim, ndim - 2, oheight)
check_dim_size(indices, ndim, ndim - 1, owidth)


# 平均池化的反向传播形状验证函数
def avg_pool3d_backward_shape_check(
    input: Tensor,
    grad_output: Tensor,
    nslices: int,
    kT: int,
    kH: int,
    kW: int,
    dT: int,
    dH: int,
    dW: int,
    pT: int,
    pH: int,
    pW: int,
    itime: int,
    iheight: int,
    iwidth: int,
    otime: int,
    oheight: int,
    owidth: int,
    fn_name: str,
):
    # 获取输入张量的维度数
    ndim = input.ndim

    # 调用函数，验证池化操作的形状和参数是否符合要求，这里 dilation 使用默认值 1
    pool3d_shape_check(
        input,
        nslices,
        kT,
        kH,
        kW,
        dT,
        dH,
        dW,
        pT,
        pH,
        pW,
        1,
        1,
        1,
        itime,
        iheight,
        iwidth,
        otime,
        oheight,
        owidth,
        fn_name,
        True,
    )

    # 检查梯度张量在特定维度上的大小是否符合要求
    check_dim_size(grad_output, ndim, ndim - 4, nslices)
    check_dim_size(grad_output, ndim, ndim - 3, otime)
    check_dim_size(grad_output, ndim, ndim - 2, oheight)
    check_dim_size(grad_output, ndim, ndim - 1, owidth)


# 最大池化的二维操作参数检查和形状计算
def max_pool2d_checks_and_compute_shape(
    input,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
):
    # 内部函数，用于解包名称和值，验证 kernel_size 是否是单个整数或两个整数的元组
    def unpack(name, val):
        torch._check(
            len(val) in [1, 2],
            lambda: f"max_pool2d: {name} must either be a single int, or a tuple of two ints",
        )
        H = val[0]
        W = H if len(val) == 1 else val[1]
        return H, W

    # 解包 kernel_size 参数，得到 kH 和 kW
    kH, kW = unpack("kernel_size", kernel_size)

    # 验证 stride 是否是省略、单个整数或两个整数的元组
    torch._check(
        len(stride) in [0, 1, 2],
        lambda: "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints",
    )
    if len(stride) == 0:
        dH, dW = kH, kW
    else:
        # 解包 stride 参数，得到 dH 和 dW
        dH, dW = unpack("stride", stride)

    # 解包 padding 参数，得到 padH 和 padW
    padH, padW = unpack("padding", padding)
    # 解包 dilation 参数，得到 dilationH 和 dilationW
    dilationH, dilationW = unpack("dilation", dilation)
    # 获取输入张量的第 -3 维度大小（输入平面数）
    nInputPlane = input.size(-3)
    # 获取输入张量的第 -2 维度大小（输入高度）
    inputHeight = input.size(-2)
    # 获取输入张量的最后一维大小（输入宽度）
    inputWidth = input.size(-1)
    # 推荐适合输入的内存格式
    memory_format = utils.suggest_memory_format(input)
    
    # 根据推荐的内存格式进行不同的检查
    if memory_format == torch.channels_last:
        # 如果内存格式为通道最后（ChannelsLast），则要求输入必须是非空的4维张量（批处理模式）
        torch._check(
            input.dim() == 4,
            lambda: "non-empty 4D (batch mode) tensor expected for input with channels_last layout",
        )
    elif memory_format == torch.contiguous_format:
        # 如果内存格式为连续格式（Contiguous），则要求输入必须是非空的3维或4维张量（批处理模式）
        torch._check(
            input.dim() in [3, 4],
            lambda: "non-empty 3D or 4D (batch mode) tensor expected for input",
        )
    else:
        # 如果内存格式不受支持，则抛出错误
        torch._check(
            False,
            lambda: "Unsupport memory format. Supports only ChannelsLast, Contiguous",
        )
    
    # 计算池化操作后的输出高度和宽度
    outputHeight = pooling_output_shape(inputHeight, kH, padH, dH, dilationH, ceil_mode)
    outputWidth = pooling_output_shape(inputWidth, kW, padW, dW, dilationW, ceil_mode)
    
    # 检查池化操作的输入形状是否符合预期
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
        memory_format,
    )
    
    # 返回输入通道数、池化后的输出高度和宽度
    return nInputPlane, outputHeight, outputWidth
@register_meta(aten.max_pool2d_with_indices_backward.default)
# 注册元数据，处理 Torch 的 max_pool2d_with_indices_backward 默认函数
def meta_max_pool2d_with_indices_backward(
    grad_output,  # 梯度输出
    self,  # 当前对象
    kernel_size,  # 卷积核大小
    stride,  # 步幅
    padding,  # 填充
    dilation,  # 扩张率
    ceil_mode,  # 是否使用 ceil 模式
    indices,  # 索引
):
    (
        nInputPlane,  # 输入平面数
        outputHeight,  # 输出高度
        outputWidth,  # 输出宽度
    ) = max_pool2d_checks_and_compute_shape(
        self, kernel_size, stride, padding, dilation, ceil_mode
    )

    torch._check(
        self.dtype == grad_output.dtype,
        lambda: f"Expected dtype {self.dtype} for `gradOutput` but got dtype {grad_output.dtype}",
    )

    nOutputPlane = nInputPlane  # 输出平面数与输入平面数相同
    ndim = self.ndim  # 数据维度

    def _check_dim_size(t):
        # 检查张量维度尺寸是否符合预期
        check_dim_size(t, ndim, ndim - 3, nOutputPlane)
        check_dim_size(t, ndim, ndim - 2, outputHeight)
        check_dim_size(t, ndim, ndim - 1, outputWidth)

    _check_dim_size(grad_output)  # 对梯度输出进行维度尺寸检查
    _check_dim_size(indices)  # 对索引进行维度尺寸检查

    memory_format = utils.suggest_memory_format(self)  # 建议的内存格式
    # 返回一个空张量，用于存储结果，设备、数据类型和内存格式与当前对象相同
    return torch.empty(
        self.shape,
        dtype=self.dtype,
        device=self.device,
        memory_format=memory_format,
    )


@register_meta(aten.max_pool2d_with_indices.default)
# 注册元数据，处理 Torch 的 max_pool2d_with_indices 默认函数
def meta_max_pool2d_with_indices(
    input,  # 输入张量
    kernel_size,  # 卷积核大小
    stride=(),  # 步幅，默认为空元组
    padding=(0,),  # 填充，默认为 (0,)
    dilation=(1,),  # 扩张率，默认为 (1,)
    ceil_mode=False,  # 是否使用 ceil 模式，默认为 False
):
    (
        nInputPlane,  # 输入平面数
        outputHeight,  # 输出高度
        outputWidth,  # 输出宽度
    ) = max_pool2d_checks_and_compute_shape(
        input, kernel_size, stride, padding, dilation, ceil_mode
    )

    nbatch = input.size(-4) if input.dim() == 4 else 1  # 批次大小
    memory_format = utils.suggest_memory_format(input)  # 建议的内存格式
    if input.dim() == 3:
        size = [nInputPlane, outputHeight, outputWidth]  # 如果输入为 3 维，则大小为 [nInputPlane, outputHeight, outputWidth]
    else:
        size = [nbatch, nInputPlane, outputHeight, outputWidth]  # 否则大小为 [nbatch, nInputPlane, outputHeight, outputWidth]
    # 返回两个空张量，一个用于存储结果，一个用于存储索引，设备、数据类型和内存格式与输入张量相同
    return (
        torch.empty(
            size,
            dtype=input.dtype,
            device=input.device,
            memory_format=memory_format,
        ),
        torch.empty(
            size,
            dtype=torch.int64,
            device=input.device,
            memory_format=memory_format,
        ),
    )


@register_meta(aten.fractional_max_pool2d.default)
# 注册元数据，处理 Torch 的 fractional_max_pool2d 默认函数
def meta_fractional_max_pool2d(self_, kernel_size, output_size, random_samples):
    torch._check(
        self_.ndim in (3, 4),  # 检查输入张量的维度是否为 3 或 4
        lambda: f"fractional_max_pool2d: Expected 3D or 4D tensor, but got: {self_.ndim}",
    )
    ndim = self_.ndim  # 数据维度

    for d in range(ndim - 3, ndim):
        torch._check(
            self_.size(d) > 0,
            f"fractional_max_pool2d: Expected input to have non-zero "
            f" size for non-batch dimenions, but got {self_.size()} with dimension {d} empty",
        )

    # 检查卷积核大小是否为二元组
    torch._check(
        len(kernel_size) == 2,
        lambda: "fractional_max_pool2d: kernel_size must"
        "either be a single int or tuple of Ints",
    )
    # 检查输出大小是否为二维，否则抛出异常信息
    torch._check(
        len(output_size) == 2,
        lambda: "fractional_max_pool2d: output_size must "
        "either be a single int or tuple of Ints",
    )

    # 获取输入张量的通道数
    input_channels = self_.size(-3)
    # 获取输入张量的高度
    input_height = self_.size(-2)
    # 获取输入张量的宽度
    input_width = self_.size(-1)
    # 如果输入张量维度为4，获取输入批次大小；否则，默认为1
    if ndim == 4:
        input_batch = self_.size(0)
    else:
        input_batch = 1

    # 检查随机样本张量的数据类型是否与输入张量相同，否则抛出异常信息
    torch._check(
        self_.dtype == random_samples.dtype,
        lambda: "Expect _random_samples to have the same dtype as input",
    )
    # 检查随机样本张量的维度是否为3，否则抛出异常信息
    torch._check(
        random_samples.ndim == 3,
        lambda: f"Expect _random samples to have 3 dimensions got, {random_samples.ndim}",
    )

    # 获取随机样本张量的大小：第一维度
    n = random_samples.size(0)
    # 获取随机样本张量的大小：第二维度
    c = random_samples.size(1)
    # 获取随机样本张量的大小：第三维度
    d = random_samples.size(2)
    # 检查随机样本张量的第一维度是否大于等于输入批次大小，否则抛出异常信息
    torch._check(
        n >= input_batch,
        "Expect _random_samples.size(0) no less then input batch size.",
    )
    # 检查随机样本张量的第二维度是否等于输入张量的通道数，否则抛出异常信息
    torch._check(
        c == input_channels,
        lambda: "Expect _random_samples.size(1) equals to input channel size.",
    )
    # 检查随机样本张量的第三维度是否等于2，否则抛出异常信息
    torch._check(d == 2, lambda: f"Expect _random_samples.size(2) equals to 2 got {d}.")

    # 检查输出高度是否小于等于输入高度与卷积核高度之和减1，否则抛出异常信息
    torch._check(
        output_size[0] + kernel_size[0] - 1 <= input_height,
        lambda: f"fractional_max_pool2d: kernel height {kernel_size[0]} is too large relative to input height {input_height}",
    )
    # 检查输出宽度是否小于等于输入宽度与卷积核宽度之和减1，否则抛出异常信息
    torch._check(
        output_size[1] + kernel_size[1] - 1 <= input_width,
        lambda: f"fractional_max_pool2d: kernel width {kernel_size[1]} is too large relative to input width {input_width}",
    )

    # 根据输入张量维度确定返回的空张量的大小
    if self_.dim() == 4:
        size = [input_batch, input_channels, output_size[0], output_size[1]]
    else:
        size = [input_channels, output_size[0], output_size[1]]

    # 返回两个空张量：一个数据类型与输入张量相同，另一个数据类型为torch.int64
    return (
        torch.empty(
            size,
            dtype=self_.dtype,
            device=self_.device,
        ),
        torch.empty(
            size,
            dtype=torch.int64,
            device=self_.device,
        ),
    )
# 注册函数为元数据处理器，用于 aten.max_unpool2d 函数
# 返回包装后的输出结果
@register_meta(aten.max_unpool2d)
@out_wrapper()
def meta_max_unpool2d(self_, indices, output_size):
    # 提示可能导致非确定性的操作 "max_unpooling2d_forward_out"
    utils.alert_not_deterministic("max_unpooling2d_forward_out")

    # 检查 indices 张量的数据类型是否为 int64
    torch._check(
        indices.dtype == torch.int64,
        lambda: f"elements in indices should be type int64 but got: {indices.dtype}",
    )
    # 检查 output_size 应该包含两个元素 (height, width)
    torch._check(
        len(output_size) == 2,
        lambda: (
            f"There should be exactly two elements (height, width) in output_size, "
            f"but got {len(output_size)} elements."
        ),
    )

    # 解包 output_size 中的高度和宽度
    oheight, owidth = output_size

    # 检查输入张量的维度应为 3 或 4
    torch._check(
        self_.ndim in (3, 4),
        lambda: (
            f"Input to max_unpooling2d should be a 3d or 4d Tensor, "
            f"but got a tensor with {self_.ndim} dimensions."
        ),
    )
    # 检查输入张量的形状与 indices 张量的形状是否相同
    torch._check(
        self_.shape == indices.shape,
        lambda: (
            f"Expected shape of indices to be same as that of the input tensor ({self_.shape}) "
            f"but got indices tensor with shape: {indices.shape}"
        ),
    )

    # 遍历输入张量的非批次维度，检查其大小是否非零
    for i in range(1, self_.ndim):
        torch._check(
            self_.size(i) > 0,
            lambda: (
                f"max_unpooling2d(): "
                f"Expected input to have non-zero size for non-batch dimensions, "
                f"but got {self_.shape} with dimension {i} being empty."
            ),
        )

    # 将输入张量转换为连续内存布局的张量
    self = self_.contiguous()

    # 根据输入张量的维度创建相应形状的空张量 result
    if self_.ndim == 3:
        nchannels = self.size(0)
        result = self.new_empty((nchannels, oheight, owidth))
    else:
        nbatch = self.size(0)
        nchannels = self.size(1)
        result = self.new_empty((nbatch, nchannels, oheight, owidth))

    # 返回创建的结果张量
    return result


# 检查 max_unpooling3d 函数的输入形状等参数是否符合预期
def _max_unpooling3d_shape_check(input, indices, output_size, stride, padding, fn_name):
    # 检查 indices 张量的数据类型是否为 int64
    torch._check(
        indices.dtype == torch.int64, lambda: "elements in indices should be type int64"
    )
    # 检查输入张量的维度应为 4 或 5
    torch._check(
        input.ndim in (4, 5),
        lambda: f"Input to max_unpooling3d should be a 4d or 5d Tensor, but got a tensor with {input.ndim} dimensions.",
    )
    # 检查 output_size 应该包含三个元素 (depth, height, width)
    torch._check(
        len(output_size) == 3,
        lambda: (
            f"There should be exactly three elements (depth, height, width) in output_size, "
            f"but got {len(output_size)} elements."
        ),
    )
    # 检查 stride 应该包含三个元素 (depth, height, width)
    torch._check(
        len(stride) == 3,
        lambda: f"There should be exactly three elements (depth, height, width) in stride, but got: {len(stride)} elements.",
    )
    # 检查 padding 应该包含三个元素 (depth, height, width)
    torch._check(
        len(padding) == 3,
        lambda: f"There should be exactly three elements (depth, height, width) in padding, but got: {len(padding)} elements.",
    )
    # 检查输入张量的形状与 indices 张量的形状是否相同
    torch._check(
        input.shape == indices.shape,
        lambda: (
            f"Expected shape of indices to be same as that of the input tensor ({input.shape}) "
            f"but got indices tensor with shape: {indices.shape}"
        ),
    )
    # 遍历输入张量的非批次维度，确保它们的尺寸大于零
    for i in range(1, input.ndim):
        # 使用 torch._check 函数检查当前维度的尺寸是否大于零，否则生成错误消息
        torch._check(
            input.size(i) > 0,
            lambda: (
                f"{fn_name}: "
                f"Expected input to have non-zero size for non-batch dimensions, "
                f"but got {input.shape} with dimension {i} being empty."
            ),
        )
    
    # 检查步幅数组的各个元素是否大于零
    torch._check(
        stride[0] > 0 and stride[1] > 0 and stride[2] > 0,
        lambda: f"strides should be greater than zero, but got stride: {stride}",
    )
# 注册元信息装饰器，用于函数aten.max_unpool3d
# 使用out_wrapper装饰器修饰函数，返回其包装后的结果
@register_meta(aten.max_unpool3d)
@out_wrapper()
def meta_max_unpool3d(self_, indices, output_size, stride, padding):
    # 提示非确定性操作，记录警告信息
    utils.alert_not_deterministic("max_unpooling3d_forward_out")

    # 执行形状检查函数_max_unpooling3d_shape_check，确保参数一致性
    _max_unpooling3d_shape_check(
        self_, indices, output_size, stride, padding, "max_unpooling3d()"
    )

    # 获得self_的连续化版本，并赋值给self
    self = self_.contiguous()

    # 解包output_size元组，获取odepth, oheight, owidth
    odepth, oheight, owidth = output_size

    # 根据self_的维度情况分别处理
    if self_.ndim == 4:
        # 如果self_是4维张量，获取通道数nchannels
        nchannels = self.size(0)
        # 创建一个与self形状相同但未初始化的结果张量result
        result = self.new_empty((nchannels, odepth, oheight, owidth))
    else:
        # 如果self_是5维张量，获取批次数nbatch和通道数nchannels
        nbatch = self.size(0)
        nchannels = self.size(1)
        # 创建一个与self形状相同但未初始化的结果张量result
        result = self.new_empty((nbatch, nchannels, odepth, oheight, owidth))

    # 返回创建的结果张量result
    return result


# 注册元信息装饰器，用于函数aten.max_pool3d_with_indices
# 使用out_wrapper装饰器修饰函数，指定函数的输出参数为"out"和"indices"
@register_meta(aten.max_pool3d_with_indices)
@out_wrapper("out", "indices")
def meta_max_pool3d_with_indices(
    input,
    kernel_size,
    stride=(),
    padding=(0,),
    dilation=(1,),
    ceil_mode=False,
):
    # 检查kernel_size的长度是否为1或3，否则抛出错误
    torch._check(
        len(kernel_size) in (1, 3),
        lambda: "max_pool3d: kernel_size must either be a single int, or a tuple of three ints",
    )
    # 解包kernel_size元组，获取kT, kH, kW
    kT = kernel_size[0]
    kH = kT if len(kernel_size) == 1 else kernel_size[1]
    kW = kT if len(kernel_size) == 1 else kernel_size[2]

    # 检查stride是否为空或长度为1或3，否则抛出错误
    torch._check(
        not stride or len(stride) in (1, 3),
        lambda: "max_pool3d: stride must either be omitted, a single int, or a tuple of three ints",
    )
    # 根据stride的情况解包，获取dT, dH, dW
    dT = kT if not stride else stride[0]
    dH = kH if not stride else (dT if len(stride) == 1 else stride[1])
    dW = kW if not stride else (dT if len(stride) == 1 else stride[2])

    # 检查padding的长度是否为1或3，否则抛出错误
    torch._check(
        len(padding) in (1, 3),
        lambda: "max_pool3d: padding must either be a single int, or a tuple of three ints",
    )
    # 解包padding元组，获取pT, pH, pW
    pT = padding[0]
    pH = pT if len(padding) == 1 else padding[1]
    pW = pT if len(padding) == 1 else padding[2]

    # 检查dilation的长度是否为1或3，否则抛出错误
    torch._check(
        len(dilation) in (1, 3),
        lambda: "max_pool3d: dilation must be either a single int, or a tuple of three ints",
    )
    # 解包dilation元组，获取dilationT, dilationH, dilationW
    dilationT = dilation[0]
    dilationH = dilationT if len(dilation) == 1 else dilation[1]
    dilationW = dilationT if len(dilation) == 1 else dilation[2]

    # 检查input的维度是否为4或5，否则抛出错误
    torch._check(
        input.ndim in (4, 5),
        lambda: "non-empty 4D or 5D (batch mode) tensor expected for input",
    )

    # 获取input的批次数nbatch和通道数nslices
    nbatch = input.size(-5) if input.ndim == 5 else 1
    nslices = input.size(-4)
    itime = input.size(-3)
    iheight = input.size(-2)
    iwidth = input.size(-1)

    # 计算输出的otime, oheight, owidth，使用pooling_output_shape函数
    otime = pooling_output_shape(itime, kT, pT, dT, dilationT, ceil_mode)
    oheight = pooling_output_shape(iheight, kH, pH, dH, dilationH, ceil_mode)
    owidth = pooling_output_shape(iwidth, kW, pW, dW, dilationW, ceil_mode)

    # 调用pool3d_shape_check函数，检查各参数的有效性
    pool3d_shape_check(
        input,
        nslices,
        kT,
        kH,
        kW,
        dT,
        dH,
        dW,
        pT,
        pH,
        pW,
        dilationT,
        dilationH,
        dilationW,
        itime,
        iheight,
        iwidth,
        otime,
        oheight,
        owidth,
        "max_pool3d_with_indices()",
    )

    # 检查输入张量是否是五维且内存格式建议为 torch.channels_last_3d
    channels_last = (
        input.ndim == 5 and utils.suggest_memory_format(input) == torch.channels_last_3d
    )
    # 如果输入张量是四维
    if input.ndim == 4:
        # 增加一个维度，以便后续检查通道是否按最后一维存储
        input_channels_last_check = input.unsqueeze(0)
        # 检查增加维度后的张量是否不是连续的，并且按照 channels_last_3d 内存格式连续
        channels_last = (
            not input_channels_last_check.is_contiguous()
        ) and input_channels_last_check.is_contiguous(
            memory_format=torch.channels_last_3d
        )
        # 定义输出形状为 (nslices, otime, oheight, owidth)
        out_shape = (nslices, otime, oheight, owidth)
    else:
        # 如果输入张量不是四维，则定义输出形状为 (nbatch, nslices, otime, oheight, owidth)
        out_shape = (nbatch, nslices, otime, oheight, owidth)  # type: ignore[assignment]

    # 根据输入张量的 dtype 创建一个新的空输出张量
    out = input.new_empty(out_shape)
    # 根据输入张量的 dtype 创建一个新的空索引张量
    indices = input.new_empty(out_shape, dtype=torch.int64)

    # 如果 channels_last 为 True，则将 out 和 indices 转换为 channels_last_3d 内存格式
    if channels_last:
        out = out.to(memory_format=torch.channels_last_3d)
        indices = indices.to(memory_format=torch.channels_last_3d)

    # 返回输出张量和索引张量
    return out, indices
# 注册一个元数据处理函数，用于计算 max_pool3d_with_indices_backward 的梯度输入
@register_meta(aten.max_pool3d_with_indices_backward)
# 用于输出包装的装饰器函数，指定返回值为 "grad_input"
@out_wrapper("grad_input")
def meta_max_pool3d_with_indices_backward(
    grad_output,  # 输入参数：梯度输出
    input,         # 输入参数：输入张量
    kernel_size,   # 输入参数：池化核大小
    stride,        # 输入参数：步幅
    padding,       # 输入参数：填充
    dilation,      # 输入参数：扩张
    ceil_mode,     # 输入参数：向上取整模式
    indices,       # 输入参数：池化索引
):
    # 检查池化核大小是否符合要求，必须是单个整数或三个整数的元组
    torch._check(
        len(kernel_size) in (1, 3),
        lambda: "max_pool3d: kernel_size must either be a single int, or a tuple of three ints",
    )
    kT = kernel_size[0]  # 如果 kernel_size 长度为1，则 kT 为 kernel_size 的第一个元素；否则 kT 为 kernel_size 的第一个元素
    kH = kT if len(kernel_size) == 1 else kernel_size[1]  # 如果 kernel_size 长度为1，则 kH 和 kT 相同；否则 kH 为 kernel_size 的第二个元素
    kW = kT if len(kernel_size) == 1 else kernel_size[2]  # 如果 kernel_size 长度为1，则 kW 和 kT 相同；否则 kW 为 kernel_size 的第三个元素

    # 检查步幅是否符合要求，可以省略或者是单个整数或三个整数的元组
    torch._check(
        not stride or len(stride) in (1, 3),
        lambda: "max_pool3d: stride must either be omitted, a single int, or a tuple of three ints",
    )
    dT = kT if not stride else stride[0]  # 如果 stride 未指定，则 dT 和 kT 相同；否则 dT 为 stride 的第一个元素
    dH = kH if not stride else (dT if len(stride) == 1 else stride[1])  # 如果 stride 未指定，则 dH 和 kH 相同；否则根据 stride 的长度确定 dH
    dW = kW if not stride else (dT if len(stride) == 1 else stride[2])  # 如果 stride 未指定，则 dW 和 kW 相同；否则根据 stride 的长度确定 dW

    # 检查填充是否符合要求，必须是单个整数或三个整数的元组
    torch._check(
        len(padding) in (1, 3),
        lambda: "max_pool3d: padding must either be a single int, or a tuple of three ints",
    )
    pT = padding[0]  # 获取填充的第一个元素
    pH = pT if len(padding) == 1 else padding[1]  # 如果 padding 长度为1，则 pH 和 pT 相同；否则 pH 为 padding 的第二个元素
    pW = pT if len(padding) == 1 else padding[2]  # 如果 padding 长度为1，则 pW 和 pT 相同；否则 pW 为 padding 的第三个元素

    # 检查扩张是否符合要求，必须是单个整数或三个整数的元组
    torch._check(
        len(dilation) in (1, 3),
        lambda: "max_pool3d: dilation must be either a single int, or a tuple of three ints",
    )
    dilationT = dilation[0]  # 获取扩张的第一个元素
    dilationH = dilationT if len(dilation) == 1 else dilation[1]  # 如果 dilation 长度为1，则 dilationH 和 dilationT 相同；否则 dilationH 为 dilation 的第二个元素
    dilationW = dilationT if len(dilation) == 1 else dilation[2]  # 如果 dilation 长度为1，则 dilationW 和 dilationT 相同；否则 dilationW 为 dilation 的第三个元素

    # 检查输入张量的维度是否为4D或5D（批处理模式），否则抛出异常
    torch._check(
        input.ndim in (4, 5),
        lambda: "non-empty 4D or 5D (batch mode) tensor expected for input",
    )

    # 获取输入张量的通道数、时间维度、高度和宽度
    nslices = input.size(-4)  # 获取切片数（通道数）
    itime = input.size(-3)     # 获取时间维度
    iheight = input.size(-2)   # 获取高度
    iwidth = input.size(-1)    # 获取宽度

    # 获取梯度输出张量的时间维度、高度和宽度
    otime = grad_output.size(-3)   # 获取梯度输出的时间维度
    oheight = grad_output.size(-2) # 获取梯度输出的高度
    owidth = grad_output.size(-1)  # 获取梯度输出的宽度

    # 调用函数，检查形状是否符合要求，用于 max_pool3d_with_indices_backward 函数
    max_pool3d_backward_shape_check(
        input,
        grad_output,
        indices,
        nslices,
        kT,
        kH,
        kW,
        dT,
        dH,
        dW,
        pT,
        pH,
        pW,
        dilationT,
        dilationH,
        dilationW,
        itime,
        iheight,
        iwidth,
        otime,
        oheight,
        owidth,
        "max_pool3d_with_indices_backward()",
    )

    # 检查输入张量是否为5D且推荐内存格式为 torch.channels_last_3d
    channels_last = (
        input.ndim == 5 and utils.suggest_memory_format(input) == torch.channels_last_3d
    )
    if input.ndim == 4:
        # 将输入张量扩展为包含一个批次维度的新张量，并检查是否不是连续的但符合 torch.channels_last_3d 的内存格式
        input_channels_last_check = input.unsqueeze(0)
        channels_last = (
            not input_channels_last_check.is_contiguous()  # 检查张量是否不是连续的
        ) and input_channels_last_check.is_contiguous(
            memory_format=torch.channels_last_3d  # 检查张量是否符合指定的内存格式
        )

    # 创建一个与输入张量形状相同的空梯度张量
    grad_input = input.new_empty(input.shape)

    # 如果 channels_last 为 True，则将 grad_input 转换为 torch.channels_last_3d 内存格式
    if channels_last:
        grad_input = grad_input.to(memory_format=torch.channels_last_3d)

    # 返回计算得到的梯度输入张量
    return grad_input
    # 检查输入张量和网格张量是否在同一设备上
    torch._check(
        input.device == grid.device,
        lambda: (
            f"grid_sampler(): expected input and grid to be on same device, but input "
            f"is on {input.device} and grid is on {grid.device}"
        ),
    )
    # 检查输入张量和网格张量是否具有 torch.strided 布局
    torch._check(
        input.layout == torch.strided and grid.layout == torch.strided,
        lambda: (
            f"grid_sampler(): expected input and grid to have torch.strided layout, but "
            f"input has {input.layout} and grid has {grid.layout}"
        ),
    )
    # 检查输入张量和网格张量的批次大小是否相同
    torch._check(
        input.shape[0] == grid.shape[0],
        lambda: (
            f"grid_sampler(): expected grid and input to have same batch size, but got "
            f"input with sizes {input.shape} and grid with sizes {grid.shape}"
        ),
    )
    # 检查网格张量的最后一个维度大小是否符合预期
    torch._check(
        grid.shape[-1] == input.ndim - 2,
        lambda: (
            f"grid_sampler(): expected grid to have size {input.ndim - 2} in last "
            f"dimension, but got grid with sizes {grid.shape}"
        ),
    )

    # 遍历输入张量的空间维度（从第三个维度开始）
    for i in range(2, input.ndim):
        # 检查输入张量的空间维度是否大于零
        torch._check(
            input.shape[i] > 0,
            lambda: (
                f"grid_sampler(): expected input to have non-empty spatial dimensions, "
                f"but input has sizes {input.shape} with dimension {i} being empty"
            ),
        )
class GridSamplerInterpolation(Enum):
    # 定义枚举类型，包括三种插值模式：双线性插值、最近邻插值、双三次插值
    BILINEAR = 0
    NEAREST = 1
    BICUBIC = 2


def check_grid_sampler_3d(input: Tensor, grid: Tensor, interpolation_mode: int):
    # 检查输入张量和网格张量的维度是否为5
    torch._check(
        input.ndim == 5 and input.ndim == grid.ndim,
        lambda: (
            f"grid_sampler(): expected 5D input and grid with same number of "
            f"dimensions, but got input with sizes {input.shape}"
            f" and grid with sizes {grid.shape}"
        ),
    )
    # 检查是否为5维输入且插值模式为双三次插值，这种情况下不支持
    torch._check(
        not (
            input.ndim == 5
            and interpolation_mode == GridSamplerInterpolation.BICUBIC.value
        ),
        lambda: "grid_sampler(): bicubic interpolation only supports 4D input",
    )


@register_meta(aten.grid_sampler_2d_backward.default)
def grid_sampler_2d_backward_meta(
    grad_output,
    input,
    grid,
    interpolation_mode,
    padding_mode,
    align_corners,
    output_mask,
):
    # 检查是否需要对输入进行梯度计算，如果需要，则创建与输入相同形状的零张量作为梯度输入
    input_requires_grad = output_mask[0]
    if input_requires_grad:
        grad_input = torch.zeros_like(input, memory_format=torch.contiguous_format)
    else:
        grad_input = None
    # 创建与网格张量相同形状的空梯度网格
    grad_grid = torch.empty_like(grid, memory_format=torch.contiguous_format)
    return (grad_input, grad_grid)


@register_meta(aten.grid_sampler_3d)
@out_wrapper()
def grid_sampler_3d(
    input,
    grid,
    interpolation_mode,
    padding_mode,
    align_corners,
):
    # 调用公共的网格采样检查函数，确保输入和网格的兼容性
    check_grid_sampler_common(input, grid)
    # 调用检查3D网格采样函数，确保输入和插值模式的兼容性
    check_grid_sampler_3d(input, grid, interpolation_mode)
    # 返回一个新的空张量，形状为(N, C, out_D, out_H, out_W)，与输入张量的设备和数据类型相同
    N = input.shape[0]
    C = input.shape[1]
    out_D = grid.shape[1]
    out_H = grid.shape[2]
    out_W = grid.shape[3]
    return input.new_empty((N, C, out_D, out_H, out_W))


@register_meta(aten.grid_sampler_3d_backward)
@out_wrapper("grad_input", "grad_grid")
def grid_sampler_3d_backward(
    grad_output,
    input,
    grid,
    interpolation_mode,
    padding_mode,
    align_corners,
    output_mask,
):
    # 调用公共的网格采样检查函数，确保输入和网格的兼容性
    check_grid_sampler_common(input, grid)
    # 调用检查3D网格采样函数，确保输入和插值模式的兼容性
    check_grid_sampler_3d(input, grid, interpolation_mode)
    # 检查是否需要对输入进行梯度计算，如果需要，则创建与输入相同形状的零张量作为梯度输入
    input_requires_grad = output_mask[0]
    if input_requires_grad:
        grad_input = torch.zeros_like(
            input, memory_format=torch.legacy_contiguous_format
        )
    else:
        grad_input = None
    # 创建与网格张量相同形状的空梯度网格
    grad_grid = torch.empty_like(grid, memory_format=torch.legacy_contiguous_format)
    return grad_input, grad_grid


@register_meta([aten.full.default])
def full(size, fill_value, *args, **kwargs):
    # 获取填充值的数据类型，如果未指定，则根据填充值获取数据类型
    dtype = kwargs.get("dtype", None)
    if not dtype:
        dtype = utils.get_dtype(fill_value)
    kwargs["dtype"] = dtype
    # 返回一个空张量，形状由size参数指定，使用给定的填充值和数据类型
    return torch.empty(size, *args, **kwargs)


# zeros_like is special cased to work for sparse
@register_meta(aten.zeros_like.default)
def zeros_like(
    self,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
):
    # 如果指定的张量布局是稀疏 COO 格式
    if layout == torch.sparse_coo:
        # 检查是否指定了内存格式选项，稀疏张量只支持分步张量（strided tensors）
        torch._check(
            memory_format is None,
            lambda: "memory format option is only supported by strided tensors",
        )

        # 创建一个空张量，大小为 0，数据类型根据传入参数确定，布局为稀疏 COO，
        # 设备为当前张量的设备，如果未指定则为默认设备，可以固定在内存中（pin_memory）
        res = torch.empty(
            0,
            dtype=self.dtype if dtype is None else dtype,
            layout=layout,
            device=self.device if device is None else device,
            pin_memory=pin_memory,
        )

        # 如果当前张量是稀疏张量
        if self.is_sparse:
            # 调整稀疏张量的大小并清空数据
            res.sparse_resize_and_clear_(
                self.size(), self.sparse_dim(), self.dense_dim()
            )
        else:
            # 对于非稀疏张量，调整大小并清空数据
            res.sparse_resize_and_clear_(self.size(), self.dim(), 0)

        # 对结果张量进行合并操作，确保其为稀疏张量
        res._coalesced_(True)
        # 返回结果张量
        return res
    
    # 如果不是稀疏 COO 格式，调用默认的 empty_like 方法创建张量
    res = aten.empty_like.default(
        self,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        memory_format=memory_format,
    )
    # 为新创建的张量填充零值
    res.fill_(0)
    # 返回结果张量
    return res
# 注册元数据为 aten.select.int 的函数 meta_select，用于实现在张量上的选择操作
@register_meta(aten.select.int)
def meta_select(self, dim, index):
    # 获取张量的维度数
    ndim = self.dim()
    # 检查张量维度不为 0，否则抛出异常
    torch._check_index(
        ndim != 0,
        lambda: "select() cannot be applied to a 0-dim tensor.",
    )

    # 将负数维度转换为正数
    dim = dim if dim >= 0 else dim + ndim
    # 获取指定维度的大小
    size = self.size(dim)

    # 检查索引是否在有效范围内
    torch._check_index(
        not (-index > size or index >= size),
        lambda: f"select(): index {index} out of range for tensor of size "
        f"{self.size()} at dimension {dim}",
    )

    # 将负数索引转换为正数
    index = index if index >= 0 else index + size

    # 创建新的尺寸和步幅列表，以及新的存储偏移量
    new_size = list(self.size())
    new_stride = list(self.stride())
    new_storage_offset = self.storage_offset() + index * new_stride[dim]

    # 删除指定维度的尺寸和步幅信息
    del new_size[dim]
    del new_stride[dim]

    # 返回基于新尺寸、新步幅和新存储偏移量的张量视图
    return self.as_strided(new_size, new_stride, new_storage_offset)


# 注册元数据为 aten.select_scatter.default 的函数 meta_select_scatter，用于克隆保持步幅的张量
@register_meta(aten.select_scatter.default)
def meta_select_scatter(self, src, dim, index):
    return utils.clone_preserve_strides(self)


# 注册元数据为 aten.slice_scatter.default 的函数 meta_slice_scatter，用于克隆保持步幅的张量
@register_meta(aten.slice_scatter.default)
def meta_slice_scatter(self, src, dim=0, start=None, end=None, step=1):
    return utils.clone_preserve_strides(self)


# TODO: 与 canonicalize_dim 合并
# 定义函数 maybe_wrap_dim，用于处理可能需要包装的维度信息
def maybe_wrap_dim(dim: int, dim_post_expr: int, wrap_scalar: bool = True):
    # 如果 dim_post_expr 小于等于 0，则确保 wrap_scalar 为真
    if dim_post_expr <= 0:
        assert wrap_scalar
        dim_post_expr = 1
    
    # 计算维度的最小值和最大值
    min = -dim_post_expr
    max = dim_post_expr - 1
    # 确保 dim 在有效范围内，否则抛出异常
    assert not (dim < min or dim > max), f"dim {dim} out of bounds ({min}, {max})"
    
    # 如果 dim 为负数，则将其转换为正数
    if dim < 0:
        dim += dim_post_expr
    
    # 返回处理后的维度值
    return dim


# 定义函数 ensure_nonempty_size，用于确保张量在指定维度上的大小非空
def ensure_nonempty_size(t, dim):
    return 1 if t.dim() == 0 else t.shape[dim]


# 从 aten/src/ATen/native/ScatterGatherChecks.h 中获取的函数 gather_shape_check
# 用于检查 gather 操作中的形状匹配
def gather_shape_check(self, dim, index):
    # 获取输入张量和索引张量的最大维度数（至少为 1）
    self_dims = max(self.dim(), 1)
    index_dims = max(index.dim(), 1)
    
    # 检查索引张量的维度与输入张量的维度相同，否则抛出异常
    torch._check(
        self_dims == index_dims,
        lambda: "Index tensor must have the same number of dimensions as input tensor",
    )
    
    # 遍历张量的每个维度，检查除了指定的 dim 维度外其他维度的尺寸匹配
    for i in range(self_dims):
        if i != dim:
            torch._check(
                ensure_nonempty_size(index, i) <= ensure_nonempty_size(self, i),
                lambda: f"Size does not match at dimension {i} expected index {index.shape}"
                + f" to be smaller than self {self.shape} apart from dimension {dim}",
            )


# 注册元数据为 aten.gather.default 的函数 meta_gather，用于实现 gather 操作
@register_meta(aten.gather.default)
def meta_gather(self, dim, index, sparse_grad=False):
    # 导入 guard_size_oblivious 函数用于处理尺寸不确定性
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

    # 包装维度信息，确保维度合法
    wrapped_dim = maybe_wrap_dim(dim, self.dim())
    
    # 检查索引张量是否为空
    is_index_empty = guard_size_oblivious(index.numel() == 0)
    if not is_index_empty:
        # 检查索引张量的数据类型为 int64
        torch._check(
            index.dtype == torch.long,
            lambda: f"gather(): Expected dtype int64 for index, but got {index.dtype}",
        )
        # 执行 gather 操作的形状检查
        gather_shape_check(self, wrapped_dim, index)
    
    # 返回一个新的空张量，其形状与索引张量相同
    return self.new_empty(index.shape)


# 从 aten/src/ATen/native/TensorAdvancedIndexing.cpp 中获取的函数 get_operator_enum
# 用于获取操作符的枚举值
def get_operator_enum(reduce_, use_new_options=False):
    pass
    # 如果使用新的选项
    if use_new_options:
        # 如果 reduce_ 等于 "sum"，返回字符串 "REDUCE_ADD"
        if reduce_ == "sum":
            return "REDUCE_ADD"
        # 如果 reduce_ 等于 "prod"，返回字符串 "REDUCE_MULTIPLY"
        elif reduce_ == "prod":
            return "REDUCE_MULTIPLY"
        # 如果 reduce_ 等于 "mean"，返回字符串 "REDUCE_MEAN"
        elif reduce_ == "mean":
            return "REDUCE_MEAN"
        # 如果 reduce_ 等于 "amax"，返回字符串 "REDUCE_MAXIMUM"
        elif reduce_ == "amax":
            return "REDUCE_MAXIMUM"
        # 如果 reduce_ 等于 "amin"，返回字符串 "REDUCE_MINIMUM"
        elif reduce_ == "amin":
            return "REDUCE_MINIMUM"
        # 如果 reduce_ 的值不是预期的任何一个，则抛出异常并显示错误消息
        torch._check(
            False,
            lambda: "reduce argument must be either sum, prod, mean, amax or amin.",
        )
        return  # 返回空，表示结束函数执行
    else:
        # 如果不使用新选项，仅支持 "add" 和 "multiply" 两种 reduce 操作
        # 如果 reduce_ 等于 "add"，返回字符串 "REDUCE_ADD"
        if reduce_ == "add":
            return "REDUCE_ADD"
        # 如果 reduce_ 等于 "multiply"，返回字符串 "REDUCE_MULTIPLY"
        elif reduce_ == "multiply":
            return "REDUCE_MULTIPLY"
        # 如果 reduce_ 的值不是 "add" 或 "multiply"，则抛出异常并显示错误消息
        torch._check(False, lambda: "reduce argument must be either add or multiply.")
        return  # 返回空，表示结束函数执行
# 从 ATen 源码中导入 ScatterGatherChecks.h 模块
def scatter_gather_dtype_check(method_name, self, index, src_opt=None):
    # 从 torch.fx.experimental.symbolic_shapes 模块中导入 guard_size_oblivious 函数
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

    # 如果 index 的元素数量不为零
    if guard_size_oblivious(index.numel() != 0):
        # 检查 index 的数据类型是否为 torch.long
        torch._check(
            index.dtype == torch.long,
            lambda: f"{method_name}(): Expected dtype int64 for index",
        )

    # 如果 src_opt 不为 None
    if src_opt is not None:
        # 检查 self 的数据类型是否与 src_opt 的数据类型相等
        torch._check(
            self.dtype == src_opt.dtype,
            lambda: f"{method_name}(): Expected self.dtype to be equal to src.dtype",
        )


# 从 ATen 源码中导入 ScatterGatherChecks.h 模块
def scatter_shape_check(self, dim, index, src_opt=None):
    # 从 torch.fx.experimental.symbolic_shapes 模块中导入 guard_size_oblivious 函数
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

    # 如果 index 的元素数量为零
    if guard_size_oblivious(index.numel() == 0):
        return
    # 检查 self 和 index 的维度是否相等
    torch._check(
        ensure_nonempty_dim(self.dim()) == ensure_nonempty_dim(index.dim()),
        lambda: "Index tensor must have the same number of dimensions as self tensor",
    )

    is_wrong_shape = False
    self_dims = ensure_nonempty_dim(self.dim())

    # 检查：index.size(d) <= self.size(d) 对于所有 d != dim
    for d in range(self_dims):
        index_d_size = ensure_nonempty_size(index, d)
        if d == dim:
            continue
        if index_d_size > ensure_nonempty_size(self, d):
            is_wrong_shape = True
            break

    # 检查：index.size(d) <= src.size(d) 对于所有 d，如果 src 是 Tensor
    if not is_wrong_shape and src_opt is not None:
        for d in range(self_dims):
            index_d_size = ensure_nonempty_size(index, d)
            if index_d_size > ensure_nonempty_size(src_opt, d):
                is_wrong_shape = True
                break

    if src_opt is not None:
        # 检查 self 和 index 的维度是否相等
        torch._check(
            ensure_nonempty_dim(self.dim()) == ensure_nonempty_dim(index.dim()),
            lambda: "Index tensor must have the same number of dimensions as self tensor",
        )
        # 检查是否有形状不匹配的情况
        torch._check(
            not is_wrong_shape,
            lambda: f"Expected index {index.shape} to be smaller than self {self.shape}"
            + f" apart from dimension {dim} and to be smaller than src {src_opt.shape}",
        )
    else:
        # 检查是否有形状不匹配的情况
        torch._check(
            not is_wrong_shape,
            lambda: f"Expected index {index.shape} to be smaller than self {self.shape}"
            + f" apart from dimension {dim}",
        )


# 从 ATen 源码中导入 TensorAdvancedIndexing.cpp 模块
def scatter_meta_impl(self, dim, index, src=None, reduce_=None, use_new_options=False):
    # 将 dim 包装为可能的维度
    wrapped_dim = maybe_wrap_dim(dim, self.dim())
    # 进行 scatter 操作的数据类型检查
    scatter_gather_dtype_check("scatter", self, index, src)
    # 进行 scatter 操作的形状检查
    scatter_shape_check(self, wrapped_dim, index, src)
    if reduce_ is not None:
        # 检查是否有有效的 reduce 运算符
        get_operator_enum(reduce_, use_new_options)


# 注册 scatter_add 操作的元信息
@register_meta(aten.scatter_add.default)
def meta_scatter_add(self, dim, index, src):
    # 调用 scatter_meta_impl 函数，用于在张量上执行按索引散布操作
    scatter_meta_impl(self, dim, index, src, "add")
    # 返回一个新的空张量，形状与当前张量相同
    return self.new_empty(self.shape)
@register_meta(aten.scatter_add_)
def meta_scatter_add_(self, dim, index, src):
    # 调用 scatter_meta_impl 函数处理 scatter_add_ 操作
    scatter_meta_impl(self, dim, index, src, "add")
    # 返回 self 对象本身
    return self


@register_meta(
    [
        aten.scatter.src,
        aten.scatter.value,
        aten.scatter.reduce,
        aten.scatter.value_reduce,
    ]
)
@out_wrapper()
def meta_scatter(self, dim, index, src_or_value, reduce=None):
    # 根据参数类型判断 src 是 tensor 还是 None
    src = src_or_value if isinstance(src_or_value, torch.Tensor) else None
    # 调用 scatter_meta_impl 函数处理 scatter 操作
    scatter_meta_impl(self, dim, index, src, reduce)
    # 返回一个新的空 tensor，形状与当前对象相同
    return self.new_empty(self.shape)


@register_meta(
    [
        aten.scatter_.src,
        aten.scatter_.value,
        aten.scatter_.reduce,
        aten.scatter_.value_reduce,
    ]
)
def meta_scatter_(self, dim, index, src_or_value, reduce=None):
    # 根据参数类型判断 src 是 tensor 还是 None
    src = src_or_value if isinstance(src_or_value, torch.Tensor) else None
    # 调用 scatter_meta_impl 函数处理 scatter_ 操作
    scatter_meta_impl(self, dim, index, src, reduce)
    # 返回 self 对象本身
    return self


@register_meta(
    [
        aten._scaled_dot_product_flash_attention_backward,
    ]
)
def meta__scaled_dot_product_flash_backward(
    grad_out: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    out: Tensor,
    logsumexp: Tensor,
    cum_seq_q: Tensor,
    cum_seq_k: Tensor,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    philox_seed: Tensor,
    philox_offset: Tensor,
    scale: Optional[float] = None,
):
    # 创建与 query, key, value 大小相同的空 tensor 用于梯度
    grad_q = torch.empty_like(query.transpose(1, 2)).transpose(1, 2)
    grad_k = torch.empty_like(key.transpose(1, 2)).transpose(1, 2)
    grad_v = torch.empty_like(value.transpose(1, 2)).transpose(1, 2)
    # 返回 query, key, value 的梯度
    return grad_q, grad_k, grad_v


@register_meta(
    [
        aten._scaled_dot_product_flash_attention_for_cpu,
    ]
)
def meta__scaled_dot_product_flash_attention_for_cpu(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    attn_mask: Optional[Tensor] = None,
    scale: Optional[float] = None,
):
    # 获取 query 的批量大小、头数、序列长度、头维度
    batch_size = query.size(0)
    num_heads = query.size(1)
    max_seqlen_batch_q = query.size(2)
    head_dim = query.size(3)

    # 创建用于注意力计算的 attention tensor 和 logsumexp tensor
    attention = torch.empty(
        (batch_size, max_seqlen_batch_q, num_heads, head_dim),
        dtype=query.dtype,
        device=query.device,
    ).transpose(1, 2)
    logsumexp = torch.empty(
        (
            batch_size,
            max_seqlen_batch_q,
            num_heads,
        ),
        dtype=torch.float,
        device=query.device,
    ).transpose(1, 2)
    # 返回 attention 和 logsumexp tensor
    return (
        attention,
        logsumexp,
    )


@register_meta(
    [
        aten._scaled_dot_product_flash_attention_for_cpu_backward,
    ]
)
def meta__scaled_dot_product_flash_attention_for_cpu_backward(
    grad_out: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    out: Tensor,
    logsumexp: Tensor,
    dropout_p: float,
    is_causal: bool,
    attn_mask: Optional[Tensor] = None,
    scale: Optional[float] = None,
):
    # CPU 的梯度布局与 CUDA 不同，
    # 获取查询张量的批量大小
    batch_size = query.size(0)
    # 获取查询张量的头数（注意力头的数量）
    num_heads = query.size(1)
    # 获取查询张量中每个头的维度大小
    head_dim = query.size(3)
    # 获取查询张量的长度（序列长度）
    len_q = query.size(2)
    # 获取键张量的长度（用于注意力计算）
    len_k = key.size(2)
    
    # 创建一个与查询张量形状相同的空张量，但维度重新排列
    # 维度顺序为：(批量大小, 序列长度, 注意力头数, 头维度)
    grad_q = torch.empty_permuted(
        (batch_size, num_heads, len_q, head_dim),
        (0, 2, 1, 3),
        dtype=query.dtype,
        device=query.device,
    )
    # 创建一个与键张量形状相同的空张量，维度也重新排列
    # 维度顺序为：(批量大小, 序列长度, 注意力头数, 头维度)
    grad_k = torch.empty_permuted(
        (batch_size, num_heads, len_k, head_dim),
        (0, 2, 1, 3),
        dtype=key.dtype,
        device=key.device,
    )
    # 创建一个与值张量形状相同的空张量，维度重新排列
    # 维度顺序为：(批量大小, 序列长度, 注意力头数, 头维度)
    grad_v = torch.empty_permuted(
        (batch_size, num_heads, len_k, head_dim),
        (0, 2, 1, 3),
        dtype=value.dtype,
        device=value.device,
    )
    
    # 返回重新排列后的查询、键、值张量
    return grad_q, grad_k, grad_v
# 注册元数据，指定了一个或多个函数，这些函数将在后续被元编程操作使用
@register_meta(
    [
        aten._scaled_dot_product_efficient_attention_backward,
    ]
)
# 定义了一个名为 meta__scaled_dot_product_efficient_backward 的函数，用于计算缩放点乘效率注意力机制的反向传播
def meta__scaled_dot_product_efficient_backward(
    grad_out: Tensor,               # 梯度输出张量
    query: Tensor,                  # 查询张量
    key: Tensor,                    # 键张量
    value: Tensor,                  # 值张量
    attn_bias: Optional[Tensor],    # 注意力偏置张量（可选）
    out: Tensor,                    # 输出张量
    logsumexp: Tensor,              # 对数求和指数张量
    philox_seed: Tensor,            # 随机数种子张量
    philox_offset: Tensor,          # 随机数偏移张量
    dropout_p: float,               # 丢弃率
    grad_input_mask: List[bool],    # 梯度输入掩码列表
    is_causal: bool = False,        # 是否为因果（可选，默认为 False）
    scale: Optional[float] = None,  # 缩放因子（可选）
):
    batch_size = query.size(0)      # 批量大小
    num_heads = query.size(1)       # 注意力头数
    max_q = query.size(2)           # 查询的最大长度
    head_dim = query.size(3)        # 注意力头维度
    head_dim_v = value.size(3)      # 值的注意力头维度

    max_k = key.size(2)             # 键的最大长度

    # 创建排列后的空张量，用于存储查询、键、值的梯度
    grad_q = torch.empty_permuted(
        (batch_size, num_heads, max_q, head_dim),   # 维度为 (批量大小, 注意力头数, 最大查询长度, 注意力头维度)
        (0, 2, 1, 3),                              # 维度排列顺序为 (0, 2, 1, 3)
        dtype=query.dtype,
        device=query.device,
    )
    grad_k = torch.empty_permuted(
        (batch_size, num_heads, max_k, head_dim),   # 维度为 (批量大小, 注意力头数, 最大键长度, 注意力头维度)
        (0, 2, 1, 3),                              # 维度排列顺序为 (0, 2, 1, 3)
        dtype=key.dtype,
        device=key.device,
    )
    grad_v = torch.empty_permuted(
        (batch_size, num_heads, max_k, head_dim_v), # 维度为 (批量大小, 注意力头数, 最大键长度, 值的注意力头维度)
        (0, 2, 1, 3),                              # 维度排列顺序为 (0, 2, 1, 3)
        dtype=value.dtype,
        device=value.device,
    )
    
    grad_bias = None
    # 如果存在注意力偏置且梯度输入掩码的第4个元素为真，则创建梯度偏置张量
    if attn_bias is not None and grad_input_mask[3]:
        lastDim = attn_bias.size(-1)    # 获取注意力偏置张量的最后一个维度大小
        lastDimAligned = lastDim if lastDim % 16 == 0 else lastDim + 16 - lastDim % 16
        new_sizes = list(attn_bias.size())
        new_sizes[-1] = lastDimAligned
        grad_bias = torch.empty(
            new_sizes,                    # 新的张量大小
            dtype=attn_bias.dtype,        # 数据类型与注意力偏置相同
            device=attn_bias.device       # 设备与注意力偏置相同
        )
        grad_bias = grad_bias[..., :lastDim]  # 裁剪张量到原始大小

    # 返回查询、键、值和梯度偏置（如果存在）的梯度张量
    return grad_q, grad_k, grad_v, grad_bias


@register_meta(
    [
        aten._flash_attention_backward,
    ]
)
# 定义了一个名为 meta__flash_attention_backward 的函数，用于计算闪电注意力机制的反向传播
def meta__flash_attention_backward(
    grad_out: Tensor,               # 梯度输出张量
    query: Tensor,                  # 查询张量
    key: Tensor,                    # 键张量
    value: Tensor,                  # 值张量
    out: Tensor,                    # 输出张量
    logsumexp: Tensor,              # 对数求和指数张量
    cum_seq_q: Tensor,              # 查询序列累积张量
    cum_seq_k: Tensor,              # 键序列累积张量
    max_q: int,                     # 最大查询长度
    max_k: int,                     # 最大键长度
    dropout_p: float,               # 丢弃率
    is_causal: bool,                # 是否为因果
    philox_seed: Tensor,            # 随机数种子张量
    philox_offset: Tensor,          # 随机数偏移张量
    scale: Optional[float] = None,  # 缩放因子（可选）
    window_size_left: Optional[int] = None,   # 左侧窗口大小（可选）
    window_size_right: Optional[int] = None,  # 右侧窗口大小（可选）
):
    # 创建与查询、键、值张量相同形状的空梯度张量
    grad_query = torch.empty_like(query)
    grad_key = torch.empty_like(key)
    grad_value = torch.empty_like(value)

    # 返回查询、键、值的梯度张量
    return grad_query, grad_key, grad_value


@register_meta(
    [
        aten._efficient_attention_backward,
    ]
)
# 定义了一个名为 meta__efficient_attention_backward 的函数，用于计算高效注意力机制的反向传播
def meta__efficient_attention_backward(
    grad_out: Tensor,               # 梯度输出张量
    query: Tensor,                  # 查询张量
    key: Tensor,                    # 键张量
    value: Tensor,                  # 值张量
    bias: Optional[Tensor],         # 注意力偏置张量（可选）
    cu_seqlens_q: Optional[Tensor], # 查询序列长度张量（可选）
    cu_seqlens_k: Optional[Tensor], # 键序列长度张量（可选）
    max_seqlen_q: torch.SymInt,     # 查询的最大序列长度
    max_seqlen_k: torch.SymInt,     # 键的最大序列长度
    logsumexp: Tensor,              # 对数求和指数张量
    dropout_p: float,               # 丢弃率
    philox_seed: Tensor,            # 随机数种子张量
    philox_offset: Tensor,          # 随机数偏移张量
    custom_mask_type: int,          # 自定义掩码类型
    bias_requires_grad: bool,       # 是否需要计算偏置的梯度
    scale: Optional[float] = None,  # 缩放因子（可选）
    num_splits_key: Optional[int] = None,     # 键的分割数（可选）
    shared_storage_dqdkdv: bool = False,      # 是否共享 dq、dk、dv 的存储空间（默认为 False）
):
    # 创建与查询、键、值张量相同形状的空梯度张量
    grad_query = torch.empty_like(query)
    grad_key = torch.empty_like(key)
    grad_value = torch.empty_like(value)

    # 返回查询、键、值的梯度张量
    return grad_query, grad_key, grad_value
    # 如果启用了共享存储 `shared_storage_dqdkdv`，则进行以下操作
    if shared_storage_dqdkdv:
        # 检查查询张量和键张量的第二维度（即序列长度）是否相等，若不相等则抛出错误信息
        torch._check(
            query.shape[1] == key.shape[1],
            lambda: "seqlen must match for `shared_storage_dqdkdv",
        )
        # 检查查询张量和键张量的第四维度（即嵌入维度）是否相等，若不相等则抛出错误信息
        torch._check(
            query.shape[3] == key.shape[3],
            lambda: "embedding dim must match for `shared_storage_dqdkdv",
        )
        # 创建一个空的张量块，形状为(*query.shape[0:-2], 3, query.shape[-2], query.shape[-1])
        # 这里(*query.shape[0:-2])保持 query 的前面维度不变，增加了一个维度为3来存放梯度信息
        chunk = torch.empty(
            (*query.shape[0:-2], 3, query.shape[-2], query.shape[-1]),
            dtype=query.dtype,
            device=query.device,
        )
        # 分别为梯度块中的查询、键、值分配张量视图
        grad_query = chunk.select(-3, 0)
        grad_key = chunk.select(-3, 1)
        grad_value = chunk.select(-3, 2)
    else:
        # 如果未启用共享存储，则创建与查询、键、值张量相同形状的空张量作为梯度张量
        grad_query = torch.empty_like(query)
        grad_key = torch.empty_like(key)
        grad_value = torch.empty_like(value)

    # 如果存在偏置项 bias
    if bias is not None:
        # 获取偏置项的最后一个维度大小
        lastDim = bias.size(-1)
        # 将最后一个维度大小调整为能被16整除的数，以保证对齐要求
        lastDimAligned = lastDim if lastDim % 16 == 0 else lastDim + 16 - lastDim % 16
        # 创建一个新的偏置张量，将其最后一个维度大小调整为 lastDimAligned
        new_sizes = list(bias.size())
        new_sizes[-1] = lastDimAligned
        grad_bias = torch.empty(new_sizes, dtype=bias.dtype, device=bias.device)
        # 从新创建的偏置张量中选择与原始偏置张量大小相同的部分作为梯度偏置
        grad_bias = grad_bias[..., :lastDim]
    else:
        # 如果不存在偏置项，则创建一个空的标量张量作为梯度偏置，设备与查询张量相同
        grad_bias = torch.empty((), device=query.device)

    # 返回计算得到的梯度张量：查询、键、值以及偏置
    return grad_query, grad_key, grad_value, grad_bias
# 注册一个元信息函数，用于处理 torch.Tensor 的 scaled_mm 操作
@register_meta([aten._scaled_mm.default])
def meta_scaled_mm(
    self: torch.Tensor,
    mat2: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    scale_result: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    use_fast_accum: bool = False,
):
    # 检查 self 和 mat2 是否为二维张量
    def is_row_major(stride):
        return stride[0] > stride[1] and stride[1] == 1

    def is_col_major(stride):
        return stride[0] == 1 and stride[1] > 1

    def is_fp8_type(dtype):
        return dtype in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2fnuz,
        )

    # 检查输入张量维度是否为二维
    torch._check(
        self.dim() == 2 and mat2.dim() == 2,
        lambda: f"Inputs must be 2D but got self.dim()={self.dim()} and mat2.dim()={mat2.dim()}",
    )
    # 检查 self 是否为行主序
    torch._check(
        is_row_major(self.stride()),
        lambda: "self must be row_major",
    )
    # 检查 mat2 是否为列主序
    torch._check(
        is_col_major(mat2.stride()),
        lambda: "mat2 must be col_major",
    )
    # 检查 self 的第二维度是否能被 16 整除
    torch._check(
        self.size(1) % 16 == 0,
        lambda: f"Expected self.size(0) to be divisible by 16, but got self.size(1)={self.size(1)}",
    )
    # 检查 mat2 的两个维度是否能被 16 整除
    torch._check(
        mat2.size(0) % 16 == 0 and mat2.size(1) % 16 == 0,
        lambda: f"Expected both dimensions of mat2 to be divisble by 16 but got {mat2.shape}",
    )
    # 检查 self 和 mat2 的数据类型是否为 fp8 类型
    torch._check(
        is_fp8_type(self.dtype) and is_fp8_type(mat2.dtype),
        lambda: f"Expected both inputs to be fp8 types but got self.dtype={self.dtype} and mat2.dtype={mat2.dtype}",
    )
    # 如果指定了输出数据类型，则使用指定的类型；否则使用 self 的数据类型作为输出数据类型
    _out_dtype = out_dtype if out_dtype is not None else self.dtype
    # 返回一个新的空张量，形状为 self.size(0) x mat2.size(1)，数据类型为 _out_dtype，存储设备为 self 的设备
    return torch.empty(self.size(0), mat2.size(1), dtype=_out_dtype, device=self.device)


# 注册一个元信息函数，处理 torch.Tensor 的 scatter_reduce_two 操作
@register_meta([aten.scatter_reduce.two, aten.scatter_reduce.two_out])
@out_wrapper()
def meta_scatter_reduce_two(self, dim, index, src, reduce, include_self=True):
    # 调用 scatter_meta_impl 函数进行 scatter_reduce 操作的实现
    scatter_meta_impl(self, dim, index, src, reduce, use_new_options=True)
    # 返回一个新的空张量，形状与 self 相同
    return self.new_empty(self.shape)


# 注册一个元信息函数，处理 torch.Tensor 的 in-place scatter_reduce_.two 操作
@register_meta(aten.scatter_reduce_.two)
def meta_scatter_reduce__two(self, dim, index, src, reduce, include_self=True):
    # 调用 scatter_meta_impl 函数进行 scatter_reduce 操作的实现
    scatter_meta_impl(self, dim, index, src, reduce, use_new_options=True)
    # 返回 self 本身
    return self


# 注册一个元信息函数，处理 torch.multinomial 操作
@register_meta([aten.multinomial.default, aten.multinomial.out])
@out_wrapper()
def meta_multinomial(input, num_samples, replacement=False, *, generator=None):
    # 检查输入张量的维度是否为 1 或 2
    torch._check(
        0 < input.dim() <= 2,
        lambda: f"The probabilty distributions dimensions must be 1 or 2, but got {input.dim()}",
    )
    # 如果输入张量维度为 1，则返回一个新的空张量，形状为 num_samples x 1，数据类型为 torch.long，存储设备为 input 的设备
    if input.dim() == 1:
        return torch.empty(num_samples, dtype=torch.long, device=input.device)
    # 否则返回一个新的空张量，形状为 input.size(0) x num_samples，数据类型为 torch.long，存储设备为 input 的设备
    return torch.empty(
        input.size(0), num_samples, dtype=torch.long, device=input.device
    )


# 定义一个函数，用于计算一组整数的乘积
def multiply_integers(vs):
    r = 1
    for v in vs:
        r *= v
    return r


# 定义一个函数，用于检查上采样操作的共同条件
def upsample_common_check(input_size, output_size, num_spatial_dims):
    # 在这里填写上采样操作的具体共同检查逻辑
    pass
    #`
        # 检查输出尺寸的长度是否等于空间维度的数量，若不等，抛出异常
        torch._check(
            len(output_size) == num_spatial_dims,
            lambda: f"It is expected output_size equals to {num_spatial_dims}, but got size {len(output_size)}",
        )
        # 计算期望的输入维度大小，通常为空间维度数量加 2（N, C, ...）
        expected_input_dims = num_spatial_dims + 2  # N, C, ...
        # 检查输入尺寸的长度是否等于期望的输入维度大小，若不等，抛出异常
        torch._check(
            len(input_size) == expected_input_dims,
            lambda: f"It is expected input_size equals to {expected_input_dims}, but got size {len(input_size)}",
        )
    
        # 检查输入尺寸的空间维度和输出尺寸的所有值是否都大于 0，若有一个不满足，抛出异常
        torch._check(
            all(s > 0 for s in input_size[2:]) and all(s > 0 for s in output_size),
            lambda: f"Input and output sizes should be greater than 0, but got "
            f"input size {input_size} and output size {output_size}",
        )
    
        # 提取输入尺寸的批次大小和通道数
        nbatch, channels = input_size[:2]
        # 返回包含批次大小、通道数和输出尺寸的元组
        return (nbatch, channels, *output_size)
@register_meta(
    [aten.upsample_nearest1d.default, aten._upsample_nearest_exact1d.default]
)
def upsample_nearest1d(input, output_size, scales=None):
    torch._check(
        input.numel() != 0 or multiply_integers(input.size()[1:]),
        lambda: f"Non-empty 3D data tensor expected but got a tensor with sizes {input.size()}",
    )
    # 检查输入是否符合要求，要求为非空的 3D 数据张量
    full_output_size = upsample_common_check(
        input.size(), output_size, num_spatial_dims=1
    )
    # 调用通用的上采样检查函数，返回完整的输出大小
    return input.new_empty(full_output_size).to(
        memory_format=utils.suggest_memory_format(input)
    )


@register_meta(
    [aten.upsample_nearest2d.default, aten._upsample_nearest_exact2d.default]
)
def upsample_nearest2d(input, output_size, scales_h=None, scales_w=None):
    torch._check(
        input.numel() != 0 or multiply_integers(input.size()[1:]),
        lambda: f"Non-empty 4D data tensor expected but got a tensor with sizes {input.size()}",
    )
    # 检查输入是否符合要求，要求为非空的 4D 数据张量
    full_output_size = upsample_common_check(
        input.size(), output_size, num_spatial_dims=2
    )
    output = input.new_empty(full_output_size)

    # 将输出转换为正确的内存格式（如果必要）
    memory_format = utils.suggest_memory_format(input)

    # 根据启发式策略："只有在通道数量小于4且使用 CUDA 设备时才使用 channels_last 路径"
    _, n_channels, _, _ = input.shape
    if input.device.type == "cuda" and n_channels < 4:
        memory_format = torch.contiguous_format

    output = output.contiguous(memory_format=memory_format)

    return output


@register_meta(
    [
        aten.upsample_nearest2d_backward.default,
        aten._upsample_nearest_exact2d_backward.default,
    ]
)
def upsample_nearest2d_backward(
    grad_output: Tensor,
    output_size: Sequence[Union[int, torch.SymInt]],
    input_size: Sequence[Union[int, torch.SymInt]],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
):
    full_output_size = upsample_common_check(
        input_size, output_size, num_spatial_dims=2
    )
    torch._check(
        grad_output.ndim == 4,
        lambda: f"Expected grad_output to be a tensor of dimension 4 but got: dimension {grad_output.ndim}",
    )
    for i in range(4):
        torch._check(
            grad_output.size(i) == full_output_size[i],
            lambda: (
                f"Expected grad_output to have the same shape as output;"
                f" output.size({i}) = {full_output_size[i]}"
                f" but got grad_output.size({i}) = {grad_output.size(i)}"
            ),
        )

    return grad_output.new_empty(input_size).to(
        memory_format=utils.suggest_memory_format(grad_output)
    )  # type: ignore[call-overload]


@register_meta(
    [aten.upsample_nearest3d.default, aten._upsample_nearest_exact3d.default]
)
def upsample_nearest3d(input, output_size, scales_d=None, scales_h=None, scales_w=None):
    # 这部分代码段未提供，需要在注释中解释该函数的作用和实现方式
    pass
    # 检查输入张量是否为非空的5维数据张量，或者尺寸不为0的多维整数乘积
    torch._check(
        input.numel() != 0 or multiply_integers(input.size()[1:]),
        lambda: f"Non-empty 5D data tensor expected but got a tensor with sizes {input.size()}",
    )
    
    # 根据输入张量的大小和输出大小进行上采样操作的通用检查，并返回完整的输出尺寸
    full_output_size = upsample_common_check(
        input.size(), output_size, num_spatial_dims=3
    )
    
    # 创建一个与输入张量尺寸相同的空张量，并按建议的内存格式转换
    return input.new_empty(full_output_size).to(
        memory_format=utils.suggest_memory_format(input)
    )
# 注册元信息，指定了一系列的排序操作函数
@register_meta(
    [
        aten.sort.default,
        aten.sort.stable,
        aten.sort.values,
        aten.sort.values_stable,
    ]
)
def meta_sort(self, stable=None, dim=-1, descending=False, values=None, indices=None):
    # 创建与 self 相同形状的空张量 v 和 int64 类型的空张量 i
    v, i = torch.empty_like(self), torch.empty_like(self, dtype=torch.int64)
    
    # 如果 values 和 indices 都不为 None，则进行下列操作
    if values is not None and indices is not None:
        assert isinstance(values, TensorLike)
        assert isinstance(indices, TensorLike)
        
        # 确保 values 和 indices 具有相同的步幅。对于不同形状的情况（例如在 msort 中的 (5, 10, 5) 和 (0)），需要调整大小。
        out_shape = v.shape
        out_stride = v.stride()
        values = _maybe_resize_out(values, out_shape)
        indices = _maybe_resize_out(indices, out_shape)
        
        # 使用 as_strided_ 方法将 values 和 indices 调整为指定的形状和步幅
        values.as_strided_(out_shape, out_stride)
        indices.as_strided_(out_shape, out_stride)
        
        # 将 self 的数据复制到 values 和 indices 中
        _safe_copy_out(copy_from=v, copy_to=values)  # type: ignore[arg-type]
        _safe_copy_out(copy_from=i, copy_to=indices)  # type: ignore[arg-type]
        
        # 返回处理后的 values 和 indices
        return values, indices
    
    # 如果 values 和 indices 有任意一个为 None，则直接返回空张量 v 和 i
    return v, i


def rnn_cell_checkSizes(
    input_gates,
    hidden_gates,
    input_bias,
    hidden_bias,
    factor,
    prev_hidden,
):
    # 检查 input_gates 张量的维度是否为 2
    torch._check(input_gates.ndim == 2, lambda: f"{input_gates.ndim} != 2")
    
    # 检查 input_gates 和 hidden_gates 的形状是否相同
    torch._check(
        input_gates.shape == hidden_gates.shape,
        lambda: f"{input_gates.shape} != {hidden_gates.shape}",
    )
    
    # 计算 input_gates 的第二个维度大小
    gates_size = input_gates.size(1)
    
    # 如果 input_bias 不为 None，则继续进行下列检查
    if input_bias is not None:
        # 检查 input_bias 张量的维度是否为 1
        torch._check(input_bias.ndim == 1, lambda: f"{input_bias.ndim} != 1")
        
        # 检查 input_bias 张量元素的数量是否等于 gates_size
        torch._check(
            input_bias.numel() == gates_size,
            lambda: f"{input_bias.numel()} != {gates_size}",
        )
        
        # 检查 input_bias 和 hidden_bias 的形状是否相同
        torch._check(
            input_bias.shape == hidden_bias.shape,
            lambda: f"{input_bias.shape} != {hidden_bias.shape}",
        )
    
    # 检查 prev_hidden 张量的维度是否为 2
    torch._check(prev_hidden.ndim == 2, lambda: f"{prev_hidden.ndim} != 2")
    
    # 计算预期的 prev_hidden 张量的元素数量
    expected_prev_hidden_numel = input_gates.size(0) * gates_size // factor
    
    # 检查 prev_hidden 张量的元素数量是否等于预期数量
    torch._check(
        prev_hidden.numel() == expected_prev_hidden_numel,
        lambda: f"{prev_hidden.numel()} != {input_gates.size(0)} * {gates_size} // {factor} (aka {expected_prev_hidden_numel})",
    )
    
    # 检查 hidden_gates、input_bias、hidden_bias 和 prev_hidden 张量是否都在同一个设备上
    torch._check(
        all(
            x.device == input_gates.device
            for x in [hidden_gates, input_bias, hidden_bias, prev_hidden]
        ),
        lambda: "expected all inputs to be same device",
    )


# 注册元信息，指定了针对 _thnn_fused_lstm_cell 的元信息处理函数
@register_meta(aten._thnn_fused_lstm_cell.default)
def _thnn_fused_lstm_cell_meta(
    input_gates,
    hidden_gates,
    cx,
    input_bias=None,
    hidden_bias=None,
):
    # 调用 rnn_cell_checkSizes 函数检查输入张量的尺寸
    rnn_cell_checkSizes(input_gates, hidden_gates, input_bias, hidden_bias, 4, cx)
    
    # 创建与 input_gates、cx 张量相同形状的空张量 workspace、hy、cy
    workspace = torch.empty_like(input_gates, memory_format=torch.contiguous_format)
    hy = torch.empty_like(cx, memory_format=torch.contiguous_format)
    cy = torch.empty_like(cx, memory_format=torch.contiguous_format)
    
    # 返回包含 hy、cy、workspace 的元组
    return (hy, cy, workspace)


# 注册元信息，指定了针对 _cudnn_rnn 的默认元信息处理函数
@register_meta(aten._cudnn_rnn.default)
# 定义一个私有函数 _cudnn_rnn，用于执行 CUDNN 的 RNN 操作
def _cudnn_rnn(
    input,
    weight,
    weight_stride0,
    weight_buf,
    hx,
    cx,
    mode,
    hidden_size,
    proj_size,
    num_layers,
    batch_first,
    dropout,
    train,
    bidirectional,
    batch_sizes,
    dropout_state,
):
    # 检查输入是否为打包格式
    is_input_packed = len(batch_sizes) != 0
    if is_input_packed:
        # 获取打包后的序列长度和第一个迷你批次大小
        seq_length = len(batch_sizes)
        mini_batch = batch_sizes[0]
        # 计算打包后的总批次大小
        batch_sizes_sum = input.shape[0]
    else:
        # 根据是否 batch_first 设置，获取序列长度和迷你批次大小
        seq_length = input.shape[1] if batch_first else input.shape[0]
        mini_batch = input.shape[0] if batch_first else input.shape[1]
        # 未打包时，批次大小总和置为 -1
        batch_sizes_sum = -1

    # 根据是否双向设置方向数量
    num_directions = 2 if bidirectional else 1
    # 确定输出大小，可以是投影大小或隐藏状态大小的两倍
    out_size = proj_size if proj_size != 0 else hidden_size
    if is_input_packed:
        # 设置输出的形状，考虑到打包后的总批次大小和方向数量
        out_shape = [batch_sizes_sum, out_size * num_directions]
    else:
        # 根据是否 batch_first 设置输出形状，考虑到序列长度、迷你批次和方向数量
        out_shape = (
            [mini_batch, seq_length, out_size * num_directions]
            if batch_first
            else [seq_length, mini_batch, out_size * num_directions]
        )
    # 创建一个新的空输出张量，形状为 out_shape，与输入张量具有相同的设备和数据类型
    output = input.new_empty(out_shape)

    # 设置细胞状态形状
    cell_shape = [num_layers * num_directions, mini_batch, hidden_size]
    if cx is None:
        # 如果没有提供初始细胞状态，创建一个空张量
        cy = torch.empty(0, device=input.device)
    else:
        # 根据提供的细胞状态形状创建一个新的空张量
        cy = cx.new_empty(cell_shape)

    # 创建一个新的空隐藏状态张量，形状为 num_layers * num_directions, mini_batch, out_size
    hy = hx.new_empty([num_layers * num_directions, mini_batch, out_size])

    # TODO: 查询 cudnnGetRNNTrainingReserveSize（向 Python 公开）
    # 根据训练状态确定保留内存大小，如果是训练状态，则为 0，否则为 0
    reserve_shape = 0 if train else 0
    # 创建一个新的空保留内存张量，数据类型为 torch.uint8
    reserve = input.new_empty(reserve_shape, dtype=torch.uint8)

    # 返回输出张量、隐藏状态、细胞状态、保留内存和权重缓冲区
    return output, hy, cy, reserve, weight_buf


# 注册 mkldnn_rnn_layer 函数作为 aten.mkldnn_rnn_layer.default 的元数据处理函数
@register_meta(aten.mkldnn_rnn_layer.default)
def mkldnn_rnn_layer(
    input,
    w0,
    w1,
    w2,
    w3,
    hx_,
    cx_,
    reverse,
    batch_sizes,
    mode,
    hidden_size,
    num_layers,
    has_biases,
    bidirectional,
    batch_first,
    train,
):
    # 根据是否 batch_first 设置，获取序列长度和迷你批次大小
    seq_length = input.shape[1] if batch_first else input.shape[0]
    mini_batch = input.shape[0] if batch_first else input.shape[1]
    # 设置输出通道数为隐藏状态大小
    output_chanels = hidden_size
    # 根据是否 batch_first 设置输出形状，考虑到序列长度、迷你批次和输出通道数
    out_shape = (
        [mini_batch, seq_length, output_chanels]
        if batch_first
        else [seq_length, mini_batch, output_chanels]
    )
    # 创建一个新的空输出张量，形状为 out_shape，与输入张量具有相同的设备和数据类型
    output = input.new_empty(out_shape)
    if hx_ is None:
        # 如果没有提供初始隐藏状态，创建一个空张量
        hy = torch.empty(0, device=input.device)
    else:
        # 根据提供的隐藏状态形状创建一个新的空张量
        hy = hx_.new_empty(hx_.shape)
    if cx_ is None:
        # 如果没有提供初始细胞状态，创建一个空张量
        cy = torch.empty(0, device=input.device)
    else:
        # 根据提供的细胞状态形状创建一个新的空张量
        cy = cx_.new_empty(cx_.shape)
    # 创建一个新的空工作空间张量，数据类型为 torch.uint8
    workspace = torch.empty(0, device=input.device, dtype=torch.uint8)
    # 返回输出张量、隐藏状态、细胞状态和工作空间
    return output, hy, cy, workspace


# 检查零元素数量并验证维度
def zero_numel_check_dims(self, dim, fn_name):
    if self.ndim == 0:
        # 如果张量是标量，检查维度是否为 0 或 -1
        torch._check_index(
            dim == 0 or dim == -1,
            lambda: f"{fn_name}: Expected reduction dim -1 or 0 for scalar but got {dim}",
        )
    else:
        # 如果张量不是标量，检查指定维度是否具有非零大小
        torch._check_index(
            self.size(dim) != 0,
            lambda: f"{fn_name}: Expected reduction dim {dim} to have non-zero size.",
        )


# 从 aten/src/ATen/native/ReduceOps.cpp 导入的函数，用于检查 argmax 和 argmin 的参数
def check_argmax_argmin(name, self, dim):
    # 如果 dim 参数不为 None，则执行以下操作：
    if dim is not None:
        # 调用 maybe_wrap_dim 函数，确保 dim 参数与当前对象的维度相匹配
        dim = maybe_wrap_dim(dim, self.dim())
        # 调用 zero_numel_check_dims 函数，检查当前对象的形状和指定的 dim 参数是否符合要求
        zero_numel_check_dims(self, dim, name)
    # 如果 dim 参数为 None，则执行以下操作：
    else:
        # 使用 torch._check 函数验证当前对象的元素数不为零，否则引发错误信息
        torch._check(
            self.numel() != 0,
            lambda: f"{name}: Expected reduction dim to be specified for input.numel() == 0.",
        )
# 注册元数据，用于处理 torch.aten.argmax 和 torch.aten.argmin 的默认情况
@register_meta([aten.argmax.default, aten.argmin.default])
def argmax_argmin_meta(self, dim=None, keepdim=False):
    # 检查并确保执行 argmax 操作的合法性
    check_argmax_argmin("argmax", self, dim)
    # 计算约简操作的维度列表
    dims = utils.reduction_dims(self.shape, (dim,) if dim is not None else None)
    # 计算约简操作后的输出形状
    shape = _compute_reduction_shape(self, dims, keepdim)
    # 返回一个新的空张量，形状为 shape，数据类型为 torch.int64
    return self.new_empty(shape, dtype=torch.int64)


# 注册元数据，用于处理 torch.aten.scalar_tensor 的默认情况
def scalar_tensor(s, dtype=None, layout=None, device=None, pin_memory=None):
    # 返回一个空张量，形状为 ()，数据类型、布局、设备及是否固定在内存中可选
    return torch.empty(
        (), dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


# 注册元数据，用于处理 torch.aten.topk 的默认情况
def topk_meta(self, k, dim=-1, largest=True, sorted=True):
    # 从 aten/src/ATen/native/Sorting.cpp 中获取的函数
    # 确保维度参数在合理范围内
    dim = maybe_wrap_dim(dim, self.dim(), wrap_scalar=True)
    torch._check(
        k >= 0 and k <= (self.size(dim) if self.dim() > 0 else 1),
        lambda: "selected index k out of range",
    )
    sliceSize = 1 if self.dim() == 0 else self.size(dim)
    torch._check(k >= 0 and k <= sliceSize, lambda: "k not in range for dimension")

    # 创建一个与 self 同形状的空张量，用于存储 topk 的值
    topKSize = list(self.shape)
    if len(topKSize) > 0:
        topKSize[dim] = k
    return self.new_empty(topKSize), self.new_empty(topKSize, dtype=torch.int64)


# 定义一个 legacy_contiguous_memory_format 变量，用于存储 torch.contiguous_format
legacy_contiguous_memory_format = torch.contiguous_format


# 从 aten/src/ATen/native/cuda/RNN.cu 中获取的函数
# 检查 LSTM 反向传播时的张量大小
def checkLSTMBackwardSizes(grad_hy, grad_cy, cx, cy, workspace):
    defined_grad = grad_hy if grad_hy is not None else grad_cy
    torch._check(defined_grad.dim() == 2, lambda: "")
    exp_size = defined_grad.size()
    if grad_hy is not None:
        torch._check(grad_hy.size() == exp_size, lambda: "")
    if grad_cy is not None:
        torch._check(grad_cy.size() == exp_size, lambda: "")
    torch._check(cx.size() == exp_size, lambda: "")
    torch._check(cy.size() == exp_size, lambda: "")
    torch._check(workspace.dim() == 2, lambda: "")
    torch._check(workspace.numel() == exp_size[0] * exp_size[1] * 4, lambda: "")


# 从 aten/src/ATen/native/cuda/RNN.cu 中获取的函数
# 注册元数据，用于处理 torch.aten._thnn_fused_lstm_cell_backward_impl 的默认情况
def _thnn_fused_lstm_cell_backward_impl(grad_hy, grad_cy, cx, cy, workspace, has_bias):
    if grad_hy is None and grad_cy is None:
        return None, None, None
    # 检查 LSTM 反向传播时的张量大小
    checkLSTMBackwardSizes(grad_hy, grad_cy, cx, cy, workspace)
    # 创建一个与 workspace 形状相同的空张量，用于存储梯度门控单元的梯度
    grad_gates = torch.empty_like(
        workspace, memory_format=legacy_contiguous_memory_format
    )
    # 创建一个与 cx 形状相同的空张量，用于存储 cx 的梯度
    grad_cx = torch.empty_like(cx, memory_format=legacy_contiguous_memory_format)
    # 如果有偏置，计算并返回偏置的梯度，否则返回 None
    grad_bias = grad_gates.sum(0, keepdim=False) if has_bias else None
    return grad_gates, grad_cx, grad_bias


# 从 aten/src/ATen/native/mps/operations/Linear.mm 中获取的函数
# 注册元数据，用于处理 torch.aten.linear_backward 的默认情况
def linear_backward(input_, grad_output_, weight_, output_mask):
    grad_input = None
    grad_weight = None
    grad_bias = None
    # 如果 output_mask 的第一个元素为 True，创建一个与 input_ 相同形状的空张量，作为 grad_input
    if output_mask[0]:
        grad_input = grad_output_.new_empty(input_.size())
    # 如果 output_mask 中索引 1 或 2 对应的值为 True，则执行以下操作
    if output_mask[1] or output_mask[2]:
        # 创建一个与 grad_output_ 相同类型和形状的新空张量，用于存储权重梯度
        grad_weight = grad_output_.new_empty((grad_output_.size(-1), input_.size(-1)))
        # 创建一个与 grad_output_ 相同类型和形状的新空张量，用于存储偏置梯度
        grad_bias = grad_output_.new_empty(grad_output_.size(-1))
    
    # 返回三个变量作为元组：grad_input 是输入的梯度，grad_weight 是权重的梯度，grad_bias 是偏置的梯度
    return (grad_input, grad_weight, grad_bias)
# 注册元信息为 aten.pixel_shuffle.default 的函数
@register_meta(aten.pixel_shuffle.default)
def meta_pixel_shuffle(self, upscale_factor):
    # 断言条件：self 的维度大于2，且倒数第三维的长度可以被 upscale_factor * upscale_factor 整除
    assert (
        len(self.shape) > 2 and self.shape[-3] % (upscale_factor * upscale_factor) == 0
    ), f"Invalid input shape for pixel_shuffle: {self.shape} with upscale_factor = {upscale_factor}"

    # 定义内部函数 is_channels_last，用于检查是否推荐使用 channels_last 内存格式
    def is_channels_last(ten):
        return torch._prims_common.suggest_memory_format(ten) == torch.channels_last

    # 定义内部函数 pick_memory_format，根据条件选择合适的内存格式
    def pick_memory_format():
        if is_channels_last(self):
            if device_hint(self) == "cuda":
                return torch.contiguous_format
            else:
                return torch.channels_last
        elif self.is_contiguous(memory_format=torch.contiguous_format):
            return torch.contiguous_format
        elif self.is_contiguous(memory_format=torch.preserve_format):
            return torch.preserve_format

    # 计算输出的通道数 C，高度 Hr，宽度 Wr
    C = self.shape[-3] // (upscale_factor * upscale_factor)
    Hr = self.shape[-2] * upscale_factor
    Wr = self.shape[-1] * upscale_factor
    # 构造输出的形状 out_shape
    out_shape = (*self.shape[:-3], C, Hr, Wr)

    # 创建一个新的空输出张量 out
    out = self.new_empty(out_shape)
    # 将 out 转换为选择的内存格式
    out = out.to(memory_format=pick_memory_format())  # type: ignore[call-overload]
    return out


# 注册元信息为 aten.mkldnn_rnn_layer_backward.default 的函数
@register_meta(aten.mkldnn_rnn_layer_backward.default)
def mkldnn_rnn_layer_backward(
    input,
    weight0,
    weight1,
    weight2,
    weight3,
    hx_,
    cx_tmp,
    output,
    hy_,
    cy_,
    grad_output_r_opt,
    grad_hy_r_opt,
    grad_cy_r_opt,
    reverse,
    mode,
    hidden_size,
    num_layers,
    has_biases,
    train,
    bidirectional,
    batch_sizes,
    batch_first,
    workspace,
):
    # 创建与 input 形状相同的空张量 diff_x
    diff_x = input.new_empty(input.shape)
    # 创建与 hx_ 形状相同的空张量 diff_hx
    diff_hx = hx_.new_empty(hx_.shape)
    # 创建与 cx_tmp 形状相同的空张量 diff_cx
    diff_cx = cx_tmp.new_empty(cx_tmp.shape)
    # 创建与 weight0 形状相同的空张量 diff_w1
    diff_w1 = weight0.new_empty(weight0.shape)
    # 创建与 weight1 形状相同的空张量 diff_w2
    diff_w2 = weight1.new_empty(weight1.shape)
    # 创建与 weight2 形状相同的空张量 diff_b
    diff_b = weight2.new_empty(weight2.shape)
    # 返回所有创建的空张量作为结果
    return diff_x, diff_w1, diff_w2, diff_b, diff_b, diff_hx, diff_cx


# 注册元信息为 aten.bucketize.Tensor 和 aten.bucketize.Tensor_out 的函数
@register_meta([aten.bucketize.Tensor, aten.bucketize.Tensor_out])
@out_wrapper()
def meta_bucketize(self, boundaries, *, out_int32=False, right=False):
    # 创建一个与 self 形状相同的空张量，根据 out_int32 和 right 参数选择相应的数据类型
    return torch.empty_like(
        self, dtype=torch.int32 if out_int32 else torch.int64
    ).contiguous()


# 注册元信息为 aten.histc 的函数
@register_meta([aten.histc])
@out_wrapper()
def meta_histc(input, bins=100, min=0, max=0):
    # 函数名称字符串
    fn_name = "histc()"
    # 如果 input 的设备提示为 "cpu"，则检查 input 是否为浮点数类型
    if device_hint(input) == "cpu":
        torch._check(
            input.is_floating_point(),
            lambda: f"\"histogram_cpu\" not implemented for '{input.dtype}'",
        )
    # 检查 bins 是否为整数类型
    torch._check(
        isinstance(bins, IntLike),
        lambda: f"{fn_name}: argument 'bins' must be int, not {type(bins)}",
    )
    # 检查 bins 是否大于 0
    torch._check(bins > 0, lambda: f"{fn_name}: bins must be > 0, but got {bins}")
    # 检查 min 是否为数值类型
    torch._check(
        isinstance(min, Number),
        lambda: f"{fn_name}: argument 'min' must be Number, not {type(min)}",
    )
    # 检查 max 是否为数值类型
    torch._check(
        isinstance(max, Number),
        lambda: f"{fn_name}: argument 'max' must be Number, not {type(max)}",
    )
    # 检查最大值是否大于等于最小值，如果不是则抛出异常信息，包含函数名
    torch._check(max >= min, lambda: "{fn_name}: max must be larger than min")
    # 返回一个新的张量，形状由 bins 指定，存储设备与输入张量相同，数据类型也与输入张量相同
    return torch.empty(bins, device=input.device, dtype=input.dtype)
@register_meta(
    [aten._upsample_bilinear2d_aa.default, aten._upsample_bicubic2d_aa.default]
)
# 注册元数据处理函数，处理双线性和双三次插值的上采样操作
def meta_upsample_bimode2d_aa(
    input,
    output_size,
    align_corners,
    scales_h=None,
    scales_w=None,
):
    # 根据输入的尺寸和输出尺寸检查并计算完整的输出尺寸
    full_output_size = upsample_common_check(
        input.size(), output_size, num_spatial_dims=2
    )
    # 检查输入张量是否非空，或者其各维度大小是否大于0
    torch._check(
        input.numel() != 0 or all(size > 0 for size in input.size()[1:]),
        lambda: f"Non-empty 4D data tensor expected but got a tensor with sizes {input.size()}",
    )
    # 创建一个新的空张量，具有指定的完整输出尺寸和建议的内存格式
    return input.new_empty(full_output_size).to(
        memory_format=utils.suggest_memory_format(input)
    )


# From aten/src/ATen/native/cuda/AmpKernels.cu
@register_meta(aten._amp_foreach_non_finite_check_and_unscale_.default)
# 注册元数据处理函数，处理非有限检查和反缩放操作
def _amp_foreach_non_finite_check_and_unscale_(self, found_inf, inv_scale):
    # 检查 found_inf 张量是否只有一个元素
    torch._check(
        found_inf.numel() == 1, lambda: "found_inf must be a 1-element tensor."
    )
    # 检查 inv_scale 张量是否只有一个元素
    torch._check(
        inv_scale.numel() == 1, lambda: "inv_scale must be a 1-element tensor."
    )
    # 检查 found_inf 张量是否为浮点型
    torch._check(
        found_inf.dtype.is_floating_point,
        lambda: "found_inf must be a float tensor.",
    )
    # 检查 inv_scale 张量是否为浮点型
    torch._check(
        inv_scale.dtype.is_floating_point,
        lambda: "inv_scale must be a float tensor.",
    )


# From aten/src/ATen/native/UnaryOps.cpp
@register_meta([aten.nan_to_num.default, aten.nan_to_num.out])
@out_wrapper()
# 注册元数据处理函数，处理将 NaN 替换为数值的操作
def nan_to_num(self, nan=None, posinf=None, neginf=None):
    # 计算结果张量的尺寸
    result_size = list(self.size())
    # 返回一个新的空张量，具有和 self 相同的尺寸
    return self.new_empty(result_size)


@register_meta(torch.ops.aten.transpose_)
# 注册元数据处理函数，处理张量的转置操作
def transpose_(self, dim0, dim1):
    # 检查 self 的布局是否为稀疏布局，如果是则抛出异常
    assert self.layout not in {
        torch.sparse_csr,
        torch.sparse_csc,
        torch.sparse_bsr,
        torch.sparse_bsc,
    }, f"torch.transpose_: in-place transposition is not supported for {self.layout} layout"

    # 获取张量的维度数
    ndims = self.ndim

    # 将 dim0 和 dim1 转换为有效的维度索引
    dim0 = maybe_wrap_dim(dim0, ndims)
    dim1 = maybe_wrap_dim(dim1, ndims)

    # 如果 dim0 和 dim1 相同，则直接返回 self
    if dim0 == dim1:
        return self

    # 获取张量的尺寸和步幅
    size = list(self.size())
    stride = list(self.stride())

    # 交换 dim0 和 dim1 的步幅和尺寸
    stride[dim0], stride[dim1] = stride[dim1], stride[dim0]
    size[dim0], size[dim1] = size[dim1], size[dim0]

    # 使用新的步幅和尺寸创建一个新的张量视图
    self.as_strided_(size, stride)
    return self


@register_meta(torch.ops.aten.t_)
# 注册元数据处理函数，处理张量的转置操作
def t_(self):
    # 获取张量的维度数
    ndims = self.ndim

    # 如果张量是稀疏张量，则检查其稀疏和稠密维度是否符合要求
    if self.is_sparse:
        sparse_dim = self.sparse_dim()
        dense_dim = self.dense_dim()
        assert (
            sparse_dim <= 2 and dense_dim == 0
        ), f"t_ expects a tensor with <= 2 sparse and 0 dense dimensions, but got {sparse_dim} sparse and {dense_dim} dense dimensions"  # noqa: B950
    else:
        # 检查张量的维度数是否不超过2
        assert (
            self.dim() <= 2
        ), f"t_ expects a tensor with <= 2 dimensions, but self is {ndims}D"

    # 调用 transpose_ 函数进行转置操作
    return transpose_(self, 0, 0 if ndims < 2 else 1)


@register_meta(aten.searchsorted)
@out_wrapper()
# 注册元数据处理函数，处理有序序列的二分搜索操作
def meta_searchsorted(
    sorted_sequence,
    self,
    *,
    out_int32=False,
    right=False,
    side=None,
    sorter=None,
):
    # 根据是否需要输出 int32 还是 int64 决定数据类型
    dtype = torch.int32 if out_int32 else torch.int64
    # 检查当前对象是否是 torch.Tensor 类型
    if isinstance(self, torch.Tensor):
        # 返回一个和当前张量相同大小和数据类型的新张量，保持存储顺序连续
        return torch.empty_like(self, dtype=dtype).contiguous()
    else:  # 如果是标量
        # 返回一个空的标量张量，数据类型为 dtype，在 sorted_sequence 的设备上
        return torch.empty((), dtype=dtype, device=sorted_sequence.device)
@register_meta(aten._embedding_bag_dense_backward)
def meta_embedding_bag_dense_backward(
    grad,
    indices,
    offset2bag,
    bag_size,
    maximum_indices,
    num_weights,
    scale_grad_by_freq,
    mode,
    per_sample_weights,
    padding_idx=-1,
):
    # 检查梯度张量的数据类型是否为浮点数类型
    torch._check(
        grad.dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64],
        lambda: f"Unsupported input type encountered: {grad.dtype}",
    )
    MODE_SUM, MODE_MEAN, MODE_MAX = range(3)
    # 如果模式为最大值模式，确保最大索引不为None
    if mode == MODE_MAX:
        torch._check(maximum_indices is not None)
    # 创建一个新的空梯度权重张量，形状为(num_weights, grad.size(1))
    index_grad_weight = grad.new_empty((num_weights, grad.size(1)))
    return index_grad_weight


@register_meta(aten._embedding_bag_per_sample_weights_backward)
def meta_embedding_bag_per_sample_weights_backward(
    grad,
    weight,
    indices,
    offsets,
    offset2bag,
    mode,
    padding_idx=-1,
):
    MODE_SUM, MODE_MEAN, MODE_MAX = range(3)
    # 检查模式是否为SUM，因为per_sample_weights仅在模式为'sum'时支持
    torch._check(
        mode == MODE_SUM,
        "embedding_bag_backward: per_sample_weights only supported for mode='sum'",
    )
    # 检查梯度张量的维度是否为2
    torch._check(grad.dim() == 2)
    # 检查索引张量的维度是否为1
    torch._check(indices.dim() == 1)
    # 获取索引张量的样本数
    num_samples = indices.size(0)
    # 检查权重张量的维度是否为2，并且检查其第二维是否等于嵌入特征的数量
    torch._check(weight.dim() == 2)
    torch._check(weight.size(1) == grad.size(1))  # grad.size(1) is embedding_features
    # 创建一个新的空输出张量，形状为(num_samples,)
    output = grad.new_empty((num_samples,))
    return output


@register_meta(aten.isin)
@out_wrapper()
def meta_isin(elements, test_elements, *, assume_unique=False, invert=False):
    # 检查输入的elements和test_elements至少有一个是Tensor类型
    torch._check(
        isinstance(elements, Tensor) or isinstance(test_elements, Tensor),
        lambda: "At least one of elements and test_elements must be a Tensor.",
    )
    # 如果elements不是Tensor类型，则将其转换为Tensor类型，并使用test_elements的设备
    if not isinstance(elements, Tensor):
        elements = torch.tensor(elements, device=test_elements.device)

    # 如果test_elements不是Tensor类型，则将其转换为Tensor类型，并使用elements的设备
    if not isinstance(test_elements, Tensor):
        test_elements = torch.tensor(test_elements, device=elements.device)

    # 检查elements和test_elements的数据类型是否为支持的类型
    _check_for_unsupported_isin_dtype(elements.dtype)
    _check_for_unsupported_isin_dtype(test_elements.dtype)
    # 创建一个与elements形状相同、数据类型为torch.bool的空张量
    return torch.empty_like(elements, dtype=torch.bool)


@register_meta(aten.polygamma)
@out_wrapper()
def meta_polygamma(n: int, self: Tensor) -> Tensor:
    # 检查n是否为非负数，polygamma函数不支持负的n值
    torch._check(n >= 0, lambda: "polygamma(n, x) does not support negative n.")
    # 调用elementwise_dtypes函数获取self张量的数据类型
    _, result_dtype = elementwise_dtypes(
        self,
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    )
    # 创建一个与self形状相同、数据类型为result_dtype的空张量
    return torch.empty_like(self, dtype=result_dtype)


@register_meta(aten.channel_shuffle.default)
def meta_channel_shuffle(input, groups):
    # 假设输入的形状为(*, C, H, W)，其中*代表任意数量的前导维度
    *leading_dims, C, H, W = input.size()
    # 输出的形状与输入相同，没有形状变化
    # 返回一个未初始化的张量，具有指定的维度和设备参数
    return torch.empty(
        *leading_dims,    # 使用可变数量的 leading_dims 参数作为张量的前导维度
        C,                # 通道数为 C
        H,                # 图像高度为 H
        W,                # 图像宽度为 W
        dtype=input.dtype,  # 使用输入张量的数据类型作为返回张量的数据类型
        layout=input.layout,  # 使用输入张量的布局作为返回张量的布局
        device=input.device,  # 使用输入张量的设备作为返回张量的设备
    )
# 注册元数据处理函数，处理本地标量稠密张量
@register_meta(aten._local_scalar_dense)
def meta_local_scalar_dense(self: Tensor):
    raise RuntimeError("Tensor.item() cannot be called on meta tensors")


# 注册元数据处理函数，处理将不规则张量转换为填充稠密张量的操作
@register_meta(aten._jagged_to_padded_dense_forward.default)
def meta__jagged_to_padded_dense_forward(
    values: Tensor,
    offsets: List[Tensor],
    max_lengths: List[int],
    padding_value: float = 0.0,
):
    # 只支持一个不规则维度
    assert len(offsets) == 1
    assert len(max_lengths) == 1

    # 计算批次数 B 和最大长度 S
    B = offsets[0].shape[0] - 1
    S = max_lengths[0]
    # 计算输出形状
    output_shape = (B, S, *values.shape[1:])
    return values.new_empty(output_shape)


# 注册元数据处理函数，处理将填充稠密张量转换为不规则张量的操作
@register_meta(aten._padded_dense_to_jagged_forward.default)
def meta__padded_dense_to_jagged_forward(
    padded: Tensor,
    offsets: List[Tensor],
    total_L: Optional[int] = None,
):
    # 只支持一个不规则维度
    assert len(offsets) == 1

    # 如果没有提供 total_L，则使用 FakeTensor 的信息推断
    if not total_L:
        assert isinstance(padded, torch._subclasses.FakeTensor)
        shape_env = padded.fake_mode.shape_env
        assert shape_env is not None
        total_L = shape_env.create_unbacked_symint()
        # 约束 total_L 的取值范围
        torch.fx.experimental.symbolic_shapes._constrain_range_for_size(
            total_L, min=0, max=None
        )

    # 计算输出形状
    output_shape = (total_L, *padded.shape[2:])
    return padded.new_empty(output_shape)


# 创建一元浮点数元数据处理函数的模板
def _create_unary_float_meta_func(func):
    @register_meta(func)
    @out_wrapper()
    def _f(x):
        return elementwise_meta(
            x, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
        )

    return _f


# 创建二元浮点数元数据处理函数的模板
def _create_binary_float_meta_func(func):
    @register_meta(func)
    @out_wrapper()
    def _f(x, y):
        return elementwise_meta(
            x, y, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
        )

    return _f


# 为一元特殊函数创建浮点数元数据处理函数
_create_unary_float_meta_func(aten.special_airy_ai)
_create_unary_float_meta_func(aten.special_bessel_y0)
_create_unary_float_meta_func(aten.special_bessel_y1)
_create_unary_float_meta_func(aten.special_modified_bessel_i0)
_create_unary_float_meta_func(aten.special_modified_bessel_i1)
_create_unary_float_meta_func(aten.special_modified_bessel_k0)
_create_unary_float_meta_func(aten.special_modified_bessel_k1)
_create_unary_float_meta_func(aten.special_scaled_modified_bessel_k0)
_create_unary_float_meta_func(aten.special_scaled_modified_bessel_k1)


# 为二元特殊函数创建浮点数元数据处理函数
_create_binary_float_meta_func(aten.special_chebyshev_polynomial_t)
_create_binary_float_meta_func(aten.special_chebyshev_polynomial_u)
_create_binary_float_meta_func(aten.special_chebyshev_polynomial_v)
_create_binary_float_meta_func(aten.special_chebyshev_polynomial_w)
_create_binary_float_meta_func(aten.special_shifted_chebyshev_polynomial_t)
_create_binary_float_meta_func(aten.special_shifted_chebyshev_polynomial_u)
_create_binary_float_meta_func(aten.special_shifted_chebyshev_polynomial_v)
_create_binary_float_meta_func(aten.special_shifted_chebyshev_polynomial_w)
_create_binary_float_meta_func(aten.special_hermite_polynomial_h)
# 调用函数 _create_binary_float_meta_func，传入 aten.special_hermite_polynomial_he 函数作为参数
_create_binary_float_meta_func(aten.special_hermite_polynomial_he)
# 调用函数 _create_binary_float_meta_func，传入 aten.special_laguerre_polynomial_l 函数作为参数
_create_binary_float_meta_func(aten.special_laguerre_polynomial_l)
# 调用函数 _create_binary_float_meta_func，传入 aten.special_legendre_polynomial_p 函数作为参数
_create_binary_float_meta_func(aten.special_legendre_polynomial_p)

# 引入 PrimTorch 参考的 meta 注册，以及 ref 分解
import torch._refs
import torch._refs.nn.functional
import torch._refs.special

# 定义函数 activate_meta，用于激活 meta 表
def activate_meta():
    # 初始化空的激活 meta 表
    activate_meta_table = {}

    # 遍历类型列表 ["meta", "post_autograd", "pre_autograd"]
    for type in ["meta", "post_autograd", "pre_autograd"]:
        # 从全局分解表 global_decomposition_table 中获取对应类型的注册表
        registry = global_decomposition_table[type]

        # 遍历注册表中的操作符 opo
        for opo in registry:
            # 如果操作符 opo 不在激活 meta 表中，将其添加到激活 meta 表中
            if opo not in activate_meta_table:
                activate_meta_table[opo] = registry[opo]

# 调用 activate_meta 函数，执行激活 meta 表的操作
activate_meta()
```