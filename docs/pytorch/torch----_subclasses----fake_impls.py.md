# `.\pytorch\torch\_subclasses\fake_impls.py`

```py
# 忽略 mypy 的类型检查错误
# 导入必要的模块和函数
import functools  # 导入 functools 模块，用于实现函数的装饰器等工具
import itertools  # 导入 itertools 模块，用于创建迭代器的函数
import math  # 导入 math 模块，提供数学运算函数
import sys  # 导入 sys 模块，提供与 Python 解释器相关的函数和变量
from typing import Callable, Union  # 从 typing 模块导入 Callable 和 Union 类型注解

import torch  # 导入 PyTorch 库
import torch._custom_op  # 导入 PyTorch 私有模块 _custom_op
import torch._logging  # 导入 PyTorch 私有模块 _logging

from torch._ops import OpOverload  # 从 torch._ops 模块导入 OpOverload 类
from torch._prims_common import (  # 从 torch._prims_common 模块导入多个函数和常量
    elementwise_dtypes,
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    is_boolean_dtype,
    is_float_dtype,
    is_integer_dtype,
)

from torch._subclasses.fake_tensor import (  # 从 torch._subclasses.fake_tensor 导入多个异常和类
    DataDependentOutputException,
    DynamicOutputShapeException,
    FakeTensor,
    in_kernel_invocation_manager,
    run_fallback_kernel,
    UnsupportedOperatorException,
)
from torch.fx.operator_schemas import normalize_function  # 导入 torch.fx.operator_schemas 模块的 normalize_function 函数

from torch.utils._stats import count_label  # 从 torch.utils._stats 导入 count_label 函数

pytree = torch.utils._pytree  # 将 torch.utils._pytree 赋值给变量 pytree

__all__ = [  # 定义公开接口列表 __all__
    "op_implementations_checks",  # 将 "op_implementations_checks" 添加到 __all__
    "get_fast_op_impls",  # 将 "get_fast_op_impls" 添加到 __all__
    "stride_incorrect_op",  # 将 "stride_incorrect_op" 添加到 __all__
    "has_meta",  # 将 "has_meta" 添加到 __all__
]

op_implementations_dict = {}  # 初始化 op_implementations_dict 空字典
op_implementations_checks = []  # 初始化 op_implementations_checks 空列表

aten = torch._ops.ops.aten  # 将 torch._ops.ops.aten 赋值给变量 aten

# 定义一个函数 ordered_set，返回包含输入项目的字典
def ordered_set(*items):
    return dict.fromkeys(items, True)

# 这个函数指示后端设备是否支持非连续张量
def is_noncontiguous_supported(device):
    if device.type == "hpu":  # 如果设备类型是 "hpu"
        return False  # 返回 False
    return True  # 否则返回 True

# 定义一个有序集合 _like_tensor_constructors，包含各种张量相关的构造函数
_like_tensor_constructors = ordered_set(
    aten.empty_like.default,
    aten.empty_like.out,
    aten.full_like.default,
    aten.full_like.out,
    aten.ones_like.default,
    aten.ones_like.out,
    aten.rand_like.default,
    aten.rand_like.out,
    aten.randn_like.default,
    aten.randn_like.out,
    aten.randint_like.default,
    aten.randint_like.out,
    aten.randint_like.low_dtype,
    aten.randint_like.low_dtype_out,
    aten.zeros_like.default,
    aten.zeros_like.out,
    aten.new_empty.default,
    aten.new_empty.out,
    aten.new_empty_strided.default,
    aten.new_empty_strided.out,
    aten.new_full.default,
    aten.new_full.out,
    aten.new_zeros.default,
    aten.new_zeros.out,
    aten.new_ones.default,
    aten.new_ones.out,
)

# 定义一个有序集合 _device_not_kwarg_ops，包含不使用关键字参数的操作集合
_device_not_kwarg_ops = ordered_set(
    aten._resize_output_.default,
    aten._nested_tensor_from_tensor_list.default,
    aten._nested_tensor_from_tensor_list.out,
    aten.pin_memory.default,
    aten.is_pinned.default,
    aten.to.device,
    aten.to.prim_Device,
    aten._pin_memory.default,
    aten._pin_memory.out,
    aten._resize_output.default,
    aten._resize_output.out,
)

# 这些操作实际上从未使用过
_non_kwarg_device_constructors = (aten._list_to_tensor,)

# 检查类型是否包含张量类型
def contains_tensor_types(type):
    tensor_type = torch._C.TensorType.get()
    return type.isSubtypeOf(tensor_type) or any(
        contains_tensor_types(e) for e in type.containedTypes()
    )

# 装饰器函数，使用 functools.lru_cache(None) 来缓存结果的函数 _is_tensor_constructor
@functools.lru_cache(None)
def _is_tensor_constructor(func: OpOverload):
    assert isinstance(func, OpOverload)  # 断言 func 是 OpOverload 类的实例
    schema = func._schema  # 获取 func 的模式 schema
    if any(contains_tensor_types(arg.type) for arg in schema.arguments):  # 如果模式的参数包含张量类型
        return False  # 返回 False
    # TODO: no real reason to restrict multiple outputs
    # 返回一个布尔值，判断 schema.returns 中是否只有一个元素，并且这个元素的类型是 torch._C.TensorType.get()
    return (
        len(schema.returns) == 1 and schema.returns[0].type is torch._C.TensorType.get()
    )
# 注册操作实现函数的装饰器，根据给定的运行实现检查器注册操作
def register_op_impl(run_impl_check: Union[Callable[[OpOverload], bool], OpOverload]):
    # 实现装饰器函数，用于注册操作实现
    def impl_decorator(op_impl):
        # 如果运行实现检查器是 OpOverload 类型
        if isinstance(run_impl_check, OpOverload):
            # 断言确保不重复注册同一个 OpOverload
            assert (
                run_impl_check not in op_implementations_dict
            ), f"duplicate registration: {run_impl_check}"
            # 将 op_impl 注册到 op_implementations_dict 中
            op_implementations_dict[run_impl_check] = op_impl
        # 如果运行实现检查器是列表或元组
        elif isinstance(run_impl_check, (list, tuple)):
            # 对列表中的每个运行实现检查器递归地注册 op_impl
            for op in run_impl_check:
                register_op_impl(op)(op_impl)
        else:
            # 如果运行实现检查器是可调用对象
            assert callable(run_impl_check)
            # 将运行实现检查器与 op_impl 组成的元组添加到 op_implementations_checks 列表中
            op_implementations_checks.append((run_impl_check, op_impl))

        return op_impl

    return impl_decorator


# 使用 op_implementations_dict 中的函数来调度操作实现
@register_op_impl(op_implementations_dict.__contains__)
def dispatch_to_op_implementations_dict(fake_mode, func, *args, **kwargs):
    return op_implementations_dict[func](fake_mode, func, *args, **kwargs)


# 注册构造函数的装饰器，用于处理像 _is_tensor_constructor 这样的构造函数
@register_op_impl(_is_tensor_constructor)
# 同时注册多个构造函数，如 _like_tensor_constructors 列表中的函数
@register_op_impl([*_like_tensor_constructors])
def constructors(fake_mode, func, *args, **kwargs):
    # 断言确保 func 不在 _non_kwarg_device_constructors 中
    assert func not in _non_kwarg_device_constructors
    # 根据函数签名规范化参数，仅使用关键字参数
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    # 如果 kwargs 中包含 "names" 键
    if "names" in kwargs:
        # 抛出异常，不支持命名张量
        raise UnsupportedOperatorException(
            "torch.compile doesn't support named tensors"
        )

    # 如果 func 在 _like_tensor_constructors 中
    if func in _like_tensor_constructors:
        # 默认设备为 new_kwargs["input"] 的设备
        default_device = new_kwargs["input"].device
        # TODO: file issue
        # 参数设置为 new_kwargs.pop("input")，因为其它情况下为空
        args = (new_kwargs.pop("input"),)
    else:
        # 如果未指定设备，默认为 CPU
        default_device = torch.device("cpu")
        args = ()

    # 输出设备根据参数指定或默认为 meta
    out_device = new_kwargs.pop("device", None)
    out_device = out_device if out_device is not None else default_device
    new_kwargs["device"] = torch.device("meta")
    
    # 使用 fake_mode 进入核心调用管理器
    with in_kernel_invocation_manager(fake_mode):
        # 调用 func 函数，传入参数 args 和 new_kwargs
        r = func(*args, **new_kwargs)
    
    # 返回伪造的张量，传入 fake_mode、r 和 out_device
    return FakeTensor(fake_mode, r, out_device)


# 注册操作函数的装饰器，处理像 aten.to.prim_Device 这样的非关键字参数转换函数
@register_op_impl(aten.to.prim_Device)
@register_op_impl(aten.to.device)
def non_kwarg_to(fake_mode, func, *args, **kwargs):
    # 根据函数签名规范化参数，仅使用关键字参数
    _, new_kwargs = normalize_function(
        func, args, kwargs, normalize_to_only_use_kwargs=True
    )
    # 获取输入参数的设备
    input_device = new_kwargs["device"]
    # 输出设备默认为输入设备，或者为 new_kwargs["input"] 的设备
    out_device = input_device if input_device else new_kwargs["input"].device
    new_kwargs["device"] = torch.device("meta")
    inp = new_kwargs.pop("input")
    
    # 使用 fake_mode 进入核心调用管理器
    with in_kernel_invocation_manager(fake_mode):
        # 调用 func 函数，传入参数 inp 和 new_kwargs
        r = func(inp, **new_kwargs)
    
    # 返回从 meta 和设备中创建的伪张量
    return fake_mode.fake_tensor_converter.from_meta_and_device(
        fake_mode, r, out_device
    )


# 检查操作是否是错位的操作函数
def stride_incorrect_op(op):
    # 如果操作不在 "aten" 或 "prims" 命名空间内
    if op.namespace not in ("aten", "prims"):
        return False
    # 如果操作是 aten._fft_c2c.default
    if op is aten._fft_c2c.default:
        return False

    # 获取操作的名称
    op_name = op.name()
    # 如果操作名称中包含 "fft"
    if "fft" in op_name:
        return True
    return False
# 注册一个操作的实现，用于处理具有不正确步幅的元实现
@register_op_impl(stride_incorrect_op)
def wordaround_stride_incorrect_op(fake_mode, func, *args, **kwargs):
    # 这是解决具有不正确步幅的元实现的一种方法

    # 检查参数是否是符号类型
    def is_symbolic(x):
        if isinstance(x, FakeTensor):
            return x._has_symbolic_sizes_strides
        if isinstance(x, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            return True
        return False

    # 如果允许回退到默认内核
    if fake_mode.allow_fallback_kernels:
        # 检查是否需要动态计算
        require_dynamic = any(
            is_symbolic(x) for x in itertools.chain(args, kwargs.values())
        )
        if not require_dynamic:
            # 展平参数列表，并执行回退内核
            flat_args, args_spec = pytree.tree_flatten((args, kwargs))
            return run_fallback_kernel(fake_mode, func, flat_args, args_spec, None)

    # 如果不满足回退条件，抛出不支持的操作异常
    raise UnsupportedOperatorException(func)


# 不要默认处理默认设备，因为忽略了 `the_template` 的设备
@register_op_impl(aten.resize_as_.default)
def resize_as_(fake_mode, func, *args, **kwargs):
    # 在内核调用管理器中执行函数
    with in_kernel_invocation_manager(fake_mode):
        return func(*args, **kwargs)


# 用于处理具有指定维度和张量的稀疏 COO 张量
@register_op_impl(aten._sparse_coo_tensor_with_dims_and_tensors.default)
def _sparse_coo_tensor_with_dims_and_tensors(fake_mode, func, *args, **kwargs):
    # TODO: remove me
    # 调用构造函数处理稀疏 COO 张量
    return constructors(fake_mode, func, *args, **kwargs)


# 在特定条件下处理依赖于索引张量的动态输出形状
@register_op_impl(
    lambda func: torch.Tag.dynamic_output_shape in func.tags
    and func
    not in [aten.index.Tensor, aten.nonzero.default, aten.repeat_interleave.Tensor]
)
def dyn_shape(fake_mode, func, *args, **kwargs):
    # 抛出动态输出形状异常
    raise DynamicOutputShapeException(func)


# 处理唯一化操作的函数，用于处理动态输出形状操作异常
def _unique(
    fake_mode, func, arg, dim, sorted=True, return_inverse=False, return_counts=False
):
    # 如果没有符号整数/符号浮点数，无法处理此操作
    if (
        fake_mode.shape_env is None
        or not fake_mode.shape_env.allow_dynamic_output_shape_ops
    ):
        raise DynamicOutputShapeException(func)

    # 不使用唯一维度的备忘录
    # 如果 dim 不是 None，或者 arg.unique_memo 是 None，则执行以下操作
    if dim is not None or (nnz := arg.unique_memo) is None:
        # 避免在模块级别导入 sympy
        from torch.fx.experimental.symbolic_shapes import (
            _constrain_range_for_size,
            has_free_symbols,
        )

        # 检查 arg 的元素个数是否为零且没有自由符号
        if not has_free_symbols(arg.numel()) and arg.numel() == 0:
            # 如果元素个数为零，则 nnz 必须为零。
            # 在这种情况下，我们不应该分配一个未支持的 SymInt，
            # 因为如果分配了，它会立即被细化为零，
            # 但这与大小无关的测试不一致（测试将继续声称未支持的
            # symint 不能等于零）。我们也可以无条件地分配一个未支持的
            # SymInt，并且不细化其范围，但这似乎更精确。
            nnz = 0
        else:
            # 创建一个未支持的 SymInt
            nnz = fake_mode.shape_env.create_unbacked_symint()

            # 设置 maxval 的初始值为 sys.maxsize - 1
            maxval = sys.maxsize - 1

            # 如果 dim 为 None，则使用 arg 的元素个数；否则使用指定维度的大小
            numel = arg.numel() if dim is None else arg.size(dim)
            # 如果 numel 没有自由符号，则将其转换为整数，并将其作为 maxval 的值
            if not has_free_symbols(numel):
                maxval = int(numel)

            # 对 nnz 应用大小的约束范围
            _constrain_range_for_size(nnz, max=maxval)

        # 如果 dim 是 None，则将 nnz 存储在 arg.unique_memo 中
        if dim is None:
            arg.unique_memo = nnz

    # 如果 dim 是 None，则创建一个新的空张量列表 ret，其大小为 (nnz,)
    if dim is None:
        ret = [arg.new_empty((nnz,))]
    else:
        # 否则，创建一个新的空张量列表 ret，其大小为 (*arg.shape[:dim], nnz, *arg.shape[dim + 1 :])
        ret = [arg.new_empty(*arg.shape[:dim], nnz, *arg.shape[dim + 1 :])]

    # 如果 dim 不是 None 且 arg 的虚拟设备为 "cpu"，则设置 return_if_dim_and_cpu 为 True
    return_if_dim_and_cpu = dim is not None and arg.fake_device == torch.device("cpu")

    # 如果需要返回逆和 return_inverse 为 True，或者 return_if_dim_and_cpu 为 True
    if return_inverse or return_if_dim_and_cpu:
        # 创建一个与 arg 的形状相同的新空张量 inverse
        inverse = arg.new_empty(arg.shape if dim is None else (arg.shape[dim],))
    else:
        # 否则，创建一个大小为 0 的新空张量 inverse
        inverse = arg.new_empty(0)
    # 将 inverse 添加到 ret 中
    ret.append(inverse)

    # 如果需要返回计数和 return_counts 为 True，或者 return_if_dim_and_cpu 为 True
    if return_counts or return_if_dim_and_cpu:
        # 创建一个与 ret[0] 的形状相同的新空张量 counts
        counts = arg.new_empty(ret[0].shape if dim is None else (ret[0].shape[dim],))
    else:
        # 否则，创建一个大小为 0 的新空张量 counts
        counts = arg.new_empty(0)
    # 将 counts 添加到 ret 中
    ret.append(counts)

    # 返回 ret 中的所有张量作为元组
    return tuple(ret)
# 注册一个自定义的操作实现函数，用于处理 torch.aten._unique2.default 操作
@register_op_impl(aten._unique2.default)
def unique2(
    fake_mode, func, arg, sorted=True, return_inverse=False, return_counts=False
):
    # 调用通用的唯一值查找函数 _unique，并返回结果
    return _unique(fake_mode, func, arg, None, sorted, return_inverse, return_counts)


# 注册一个自定义的操作实现函数，用于处理 torch.aten.unique_dim.default 操作
@register_op_impl(aten.unique_dim.default)
def unique_dim(
    fake_mode, func, arg, dim, sorted=True, return_inverse=False, return_counts=False
):
    # 调用通用的唯一值查找函数 _unique，将维度参数标准化为非负数后再传递，并返回结果
    return _unique(
        fake_mode,
        func,
        arg,
        dim if dim >= 0 else dim % max(arg.ndim, 1),
        sorted,
        return_inverse,
        return_counts,
    )


# 注册一个自定义的操作实现函数，用于处理 torch.aten.repeat_interleave.Tensor 操作
def repeat_interleave_tensor(fake_mode, func, repeats, output_size=None):
    if output_size is None:
        # 如果输出大小未指定，根据条件判断是否允许动态输出形状操作，若不允许则引发异常
        if (
            fake_mode.shape_env is None
            or not fake_mode.shape_env.allow_dynamic_output_shape_ops
        ):
            raise DynamicOutputShapeException(func)

        # 创建一个未支持的符号整数用于输出大小
        output_size = fake_mode.shape_env.create_unbacked_symint()

        # 避免在模块级别导入 sympy
        from torch.fx.experimental.symbolic_shapes import _constrain_range_for_size

        # 对输出大小应用范围约束
        _constrain_range_for_size(output_size)
        # TODO: 考虑使用备忘录
    return repeats.new_empty(output_size)


# 注册一个自定义的操作实现函数，用于处理 torch.ops.aten._local_scalar_dense.default 操作
@register_op_impl(torch.ops.aten._local_scalar_dense.default)
def local_scalar_dense(fake_mode, func, arg):
    # 如果 arg.item_memo 不为 None，直接返回其值
    if (r := arg.item_memo) is not None:
        return r
    # 如果形状环境未定义或不允许标量输出，并且不允许标量输出，引发数据依赖输出异常
    if fake_mode.shape_env is None or (
        not fake_mode.shape_env.allow_scalar_outputs
        and not fake_mode.allow_scalar_outputs
    ):
        raise DataDependentOutputException(func)
    # 根据 arg 的数据类型创建对应的符号值并缓存到 arg.item_memo 中
    if is_float_dtype(arg.dtype):
        r = fake_mode.shape_env.create_unbacked_symfloat()
    elif is_integer_dtype(arg.dtype):
        r = fake_mode.shape_env.create_unbacked_symint()
    elif is_boolean_dtype(arg.dtype):
        r = fake_mode.shape_env.create_unbacked_symbool()
    else:
        raise NotImplementedError(f"local_scalar_dense/item NYI for {arg.dtype}")
    arg.item_memo = r
    return r


# 注册一个自定义的操作实现函数，用于处理 torch.ops.aten.nonzero.default 操作
@register_op_impl(torch.ops.aten.nonzero.default)
def nonzero(fake_mode, func, arg):
    # 如果形状环境未定义或不允许动态输出形状操作，引发动态输出形状异常
    if (
        fake_mode.shape_env is None
        or not fake_mode.shape_env.allow_dynamic_output_shape_ops
    ):
        raise DynamicOutputShapeException(func)
    # 如果 nnz 存在并且非空，则直接使用现有的非零元素数量
    if (nnz := arg.nonzero_memo) is None:
        # 避免在模块级别导入 sympy
        from torch.fx.experimental.symbolic_shapes import (
            _constrain_range_for_size,
            has_free_symbols,
        )

        # 检查 arg 的元素数量是否为零且不含自由符号
        if not has_free_symbols(arg.numel()) and arg.numel() == 0:
            # 如果元素数量为零，则输出大小必须为零。
            # 在这种情况下，我们不应分配一个未支持的 SymInt，
            # 因为如果这样做，它将立即被细化为零，但这与大小无关的
            # 测试不一致（这些测试将继续声称未支持的 symint 不能等于零）。
            # 我们也可以无条件地分配一个未支持的 SymInt 并且不细化其范围，
            # 但这似乎更加精确。
            nnz = 0
        else:
            # 否则，创建一个未支持的 SymInt
            nnz = fake_mode.shape_env.create_unbacked_symint()

            # 设置最大值为 sys.maxsize - 1
            maxval = sys.maxsize - 1

            # 如果 arg 的元素数量不含自由符号，则将 maxval 设置为其整数值
            if not has_free_symbols(arg.numel()):
                maxval = int(arg.numel())

            # 约束 nnz 的范围以符合给定的大小
            _constrain_range_for_size(nnz, max=maxval)

        # 将计算得到的 nnz 存储到 arg 的 nonzero_memo 属性中
        arg.nonzero_memo = nnz

    # 返回一个新的、未初始化的张量，形状为 (nnz, arg.dim())，数据类型为 torch.int64
    return arg.new_empty((nnz, arg.dim()), dtype=torch.int64)
@register_oppython
# 注册函数实现，用于 torch.ops.aten.masked_select.default 操作
@register_op_impl(torch.ops.aten.masked_select.default)
def masked_select(fake_mode, func, self, mask):
    # 如果没有符号整数或符号浮点数环境，无法处理此操作
    if (
        fake_mode.shape_env is None
        or not fake_mode.shape_env.allow_dynamic_output_shape_ops
    ):
        # 抛出动态输出形状异常，指定函数为 func
        raise DynamicOutputShapeException(func)

    # 创建一个未支持的符号整数 nnz
    nnz = fake_mode.shape_env.create_unbacked_symint()

    # 查看 nonzero 函数的评论
    maxval = sys.maxsize - 1  # 最大值为系统最大值减去1

    # 避免在模块级别导入 sympy
    from torch.fx.experimental.symbolic_shapes import (
        _constrain_range_for_size,
        has_free_symbols,
    )

    # 如果 self.numel() 没有自由符号
    if not has_free_symbols(self.numel()):
        # 如果 self.numel() 大于2，将 maxval 设置为 self.numel() 的整数值
        if self.numel() > 2:
            maxval = int(self.numel())

    # 对 nnz 进行大小范围约束，最大值为 maxval
    _constrain_range_for_size(nnz, max=maxval)

    # 返回一个新的空张量，形状为 (nnz,)
    return self.new_empty((nnz,))


# 注意：此处必须在 local_scalar_dense 之后进行排序
@register_op_impl(lambda func: torch.Tag.data_dependent_output in func.tags)
def data_dep(fake_mode, func, *args, **kwargs):
    # 抛出数据依赖输出异常，指定函数为 func
    raise DataDependentOutputException(func)


# 布尔索引将被扩展为掩码
# 参见：IndexingUtils.h:expandTensors
def check_no_bool_index_tensors(func, self, indices):
    # 遍历索引列表 indices
    for index in indices:
        # 如果索引不为 None 且其数据类型为 torch.bool 或 torch.uint8
        if index is not None and index.dtype in (torch.bool, torch.uint8):
            # 抛出动态输出形状异常，指定函数为 func
            raise DynamicOutputShapeException(func)


# 运行并返回与输入设备相同的新张量
def run_and_return_new_tensor_of_input_device(fake_mode, func, args, kwargs):
    # 规范化函数，仅使用 kwargs 进行规范化
    _, new_kwargs = normalize_function(
        func, args=args, kwargsnew_kwargs["input"]:
        return out  # 返回复制后的张量
    return FakeTensor(fake_mode, out, out_device)


_is_builtin_namespaces = ordered_set("aten", "prims", "prim")


def is_builtin(op):
    # 检查操作是否在内置命名空间中
    return op.namespace in _is_builtin_namespaces


def has_meta(func):
    # 检查函数是否具有元数据
    return torch._C._dispatch_has_computed_kernel_for_dispatch_key(func.name(), "Meta")


@register_op_impl(
    lambda func: is_builtin(func) and "foreach" in func.name() and has_meta(func)
)
def foreach_run_and_map_input_device(fake_mode, func, *args, **kwargs):
    tensor_lists = []
    for arg in itertools.chain(args, kwargs.values()):
        if (
            isinstance(arg, (list, tuple))
            and len(arg)
            and isinstance(arg[0], torch.Tensor)
        ):
            tensor_lists.append(arg)

    try:
        with in_kernel_invocation_manager(fake_mode):
            out_meta = func(*args, **kwargs)
    except NotImplementedError as not_implemented_error:
        return NotImplemented

    if not out_meta:
        return out_meta

    assert tensor_lists
    out_fake = []
    # 遍历 out_meta 列表中的元数据，并返回元素索引 i 及其值 meta_t
    for i, meta_t in enumerate(out_meta):
        # 调用 FakeTensor 类的 _find_common_device 方法，查找 func 函数和 tensor_lists 中每个列表 tl[i] 共同的设备
        device, _ = FakeTensor._find_common_device(func, [tl[i] for tl in tensor_lists])
        # 调用 fake_mode.fake_tensor_converter 中的 from_meta_and_device 方法，
        # 使用 fake_mode、meta_t 和找到的设备 device 来创建一个新的 fake_tensor 对象，并将其添加到 out_fake 列表中
        out_fake.append(
            fake_mode.fake_tensor_converter.from_meta_and_device(
                fake_mode, meta_t, device
            )
        )
    
    # 返回填充了 fake_tensor 对象的 out_fake 列表作为函数的结果
    return out_fake
# 不要默认处理默认设备，因为 op 可以处理带有 cuda 自身的非零大小的 CPU 索引张量
@register_op_impl(aten.index.Tensor)
def index_tensor(fake_mode, func, *args, **kwargs):
    # 从 torch._meta_registrations 导入 meta_index_Tensor
    from torch._meta_registrations import meta_index_Tensor

    # 根据函数标准化参数，仅使用关键字参数
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # 获取输入张量的设备
    out_device = new_kwargs["input"].device
    # 确保非零调用传递到虚拟张量
    with fake_mode:
        # 调用 meta_index_Tensor 函数并返回结果到指定设备
        out = meta_index_Tensor(*args, **kwargs)
        return out.to(out_device)


# 可以接受混合的元数据/非元数据参数；即使给定真实设备，元注册也会大致做正确的事情
@register_op_impl(aten._embedding_bag.default)
def embedding_bag(fake_mode, func, *args, **kwargs):
    # 从 torch._meta_registrations 导入 meta_embedding_bag
    from torch._meta_registrations import meta_embedding_bag

    # 使用 fake_mode 执行 meta_embedding_bag 函数并返回结果
    with fake_mode:
        return meta_embedding_bag(*args, **kwargs)


# 接受多设备输入，不要默认处理默认设备
@register_op_impl(aten._unsafe_index_put.default)
@register_op_impl(aten.copy.default)
@register_op_impl(aten.copy_.default)
@register_op_impl(aten.slice_scatter.default)
def multi_device_op_default(fake_mode, func, *args, **kwargs):
    # 使用 run_and_return_new_tensor_of_input_device 函数执行并返回新的张量，使用输入设备
    return run_and_return_new_tensor_of_input_device(fake_mode, func, args, kwargs)


# 和 multi_device_op_default 相同，但返回输入
@register_op_impl(aten.copy.out)
@register_op_impl(aten.slice_scatter.out)
def multi_device_op_out(fake_mode, func, *args, **kwargs):
    # 在内核调用管理器中使用 fake_mode 执行 func 函数
    with in_kernel_invocation_manager(fake_mode):
        out = func(*args, **kwargs)

    # 根据函数标准化参数，仅使用关键字参数
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # 返回输入张量
    return new_kwargs["input"]


# 索引放置操作的实现
@register_op_impl(aten.index_put.default)
@register_op_impl(aten.index_put_.default)
def index_put_impl(fake_mode, func, *args, **kwargs):
    # 根据函数标准化参数，仅使用关键字参数
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # 获取值和输入张量的设备
    values = new_kwargs["values"]
    self_device = new_kwargs["input"].fake_device
    # 检查自身设备与值设备是否匹配，或者是否值为标量
    torch._check(
        self_device == values.fake_device or (values.ndim == 0 and values.numel() == 1),
        lambda: f"Mismatching {func} device between self ({self_device}) and values ({values.device})",
    )

    # 使用 run_and_return_new_tensor_of_input_device 函数执行并返回新的张量
    out = run_and_return_new_tensor_of_input_device(fake_mode, func, args, kwargs)
    # 如果 func 是 aten.index_put_.default，则返回输入张量
    if func is aten.index_put_.default:
        return new_kwargs["input"]
    else:
        return out


# 不支持嵌套张量的操作
@register_op_impl(aten._nested_tensor_from_tensor_list.default)
@register_op_impl(aten._nested_tensor_from_tensor_list.out)
@register_op_impl(aten._nested_view_from_buffer.default)
@register_op_impl(aten._nested_view_from_buffer_copy.default)
def nested_tensors_unsupported(fake_mode, func, *args, **kwargs):
    # 抛出不支持的操作异常，因为 torch.compile 不支持 strided NestedTensor
    raise UnsupportedOperatorException(
        "torch.compile does not support strided NestedTensor"
    )
    # 列表推导式，遍历 _device_not_kwarg_ops 中的每个元素 x
    [
        x
        for x in _device_not_kwarg_ops
        # 筛选条件：x 不在以下已经在其他地方注册的函数中
        if x
        not in (
            # 这些函数已经在其他地方注册过了
            aten.to.device,
            aten.to.prim_Device,
            aten._nested_tensor_from_tensor_list.default,
            aten._nested_tensor_from_tensor_list.out,
        )
    ]
# 定义一个函数 nyi，用于标记未实现的功能
def nyi(fake_mode, func, *args, **kwargs):
    # 断言 func 不在 _device_not_kwarg_ops 中，否则抛出异常，提示未实现该功能
    assert func not in _device_not_kwarg_ops, f"NYI: {func}"

# 注册 op 实现函数，处理卷积操作
@register_op_impl([aten.convolution.default, aten.convolution_backward.default])
def conv(fake_mode, func, *args, **kwargs):
    # 根据参数规范化函数调用，只使用关键字参数
    _, kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    # 获取输入的虚拟设备信息
    device = kwargs["input"].fake_device
    # 需要重新启用虚拟模式，以便张量能够报告虚拟设备信息
    with fake_mode:
        # 如果输入在 Convolution.cpp 中未被挤压，会导致段错误
        k = kwargs["weight"].ndim
        batch = kwargs["input"].shape[0]

        # 避免在模块级别导入 sympy
        from torch.fx.experimental.symbolic_shapes import has_hint

        if not has_hint(batch):
            # TODO: 可以稍微更加忠实地检测通道顺序（如果静态上明显的话）
            mem_fmt = None
        elif k == 3 and not kwargs["input"].is_mkldnn and not kwargs["input"].is_xpu:
            mem_fmt = None
        else:
            # 根据函数类型选择卷积后端
            if func is aten.convolution.default:
                conv_backend = torch._C._select_conv_backend(**kwargs)
            else:
                conv_backend = torch._C._select_conv_backend(
                    kwargs["input"],
                    kwargs["weight"],
                    bias=None,
                    stride=kwargs["stride"],
                    padding=kwargs["padding"],
                    dilation=kwargs["dilation"],
                    transposed=kwargs["transposed"],
                    output_padding=kwargs["output_padding"],
                    groups=kwargs["groups"],
                    bias_sizes=kwargs["bias_sizes"],
                )
            # 确定卷积操作的内存格式
            mem_fmt = torch._C._conv_determine_backend_memory_format(
                kwargs["input"], kwargs["weight"], conv_backend
            )

    # 定义转换函数，将张量转换为虚拟设备格式
    def convert(t, mem_fmt):
        if t is None:
            return t
        if mem_fmt is not None:
            t = t.to(memory_format=mem_fmt)
        return FakeTensor(fake_mode, t, device)

    # 使用内核调用管理器，执行函数计算
    with in_kernel_invocation_manager(fake_mode):
        out = func(**kwargs)

        # 根据函数类型进行结果转换
        if func is aten.convolution.default:
            return convert(out, mem_fmt)
        else:
            return (
                convert(out[0], mem_fmt),
                convert(out[1], mem_fmt),
                convert(out[2], None),
            )


# 注册 op 实现函数，处理 scaled dot product flash attention 操作
@register_op_impl(aten._scaled_dot_product_flash_attention.default)
def meta__scaled_dot_product_flash(fake_mode, func, *args, **kwargs):
    # 根据参数规范化函数调用，只使用关键字参数
    _, kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # 获取 query、key 和 return_debug_mask 参数
    query = kwargs["query"]
    key = kwargs["key"]
    return_debug_mask = kwargs["return_debug_mask"]
    # unused: value, dropout_p, is_causal, scale

    # 定义转换张量函数，将张量转换为虚拟设备格式
    def convert_tensor(t, device):
        return FakeTensor(fake_mode, t, device)

    # 获取 batch_size
    batch_size = query.size(0)
    # 获取查询张量的第二维大小，即头数
    num_heads = query.size(1)
    # 获取查询张量的第三维大小，即每个批次中的最大序列长度
    max_seqlen_batch_q = query.size(2)
    # 获取查询张量的第四维大小，即每个头的维度
    head_dim = query.size(3)
    # 获取键张量的第三维大小，即每个批次中的最大序列长度
    max_seqlen_batch_k = key.size(2)

    # 将查询张量按照第一维和第二维进行转置
    query_t = query.transpose(1, 2)
    # 创建一个与转置后的查询张量相同大小的空张量，并将其按照第一维和第二维进行再次转置
    attention = torch.empty_like(query_t).transpose(1, 2)
    
    # 创建一个形状为(batch_size, num_heads, max_seqlen_batch_q)的空张量，用于计算对数求和指数
    logsumexp = convert_tensor(
        torch.empty(
            (batch_size, num_heads, max_seqlen_batch_q),
            dtype=torch.float,
            device="meta",
        ),
        device=query.device,
    )

    if return_debug_mask:
        # 如果需要返回调试掩码
        # 根据头维度大小选择块大小
        blocksize_c = 128 if head_dim > 64 else 256
        # 计算键的最大序列长度
        max_seqlen_k = math.ceil(max_seqlen_batch_q / blocksize_c)
        # 根据键的最大序列长度重新调整最大序列长度，保证其符合条件
        if max_seqlen_batch_k <= 128:
            max_seqlen_k = 128
        elif max_seqlen_batch_k <= 256:
            max_seqlen_k = 256
        # 创建一个形状为(batch_size, num_heads, max_seqlen_batch_q, max_seqlen_k)的调试掩码张量
        debug_mask = convert_tensor(
            torch.empty(
                (batch_size, num_heads, max_seqlen_batch_q, max_seqlen_k),
                dtype=query.dtype,
                device="meta",
            ),
            device=query.device,
        )
    else:
        # 如果不需要返回调试掩码，则创建一个空张量
        debug_mask = convert_tensor(
            torch.empty(0, dtype=query.dtype, device="meta"),
            query.device,
        )

    # 返回一系列张量和值，包括注意力、对数求和指数、最大序列长度等
    return (
        attention,
        logsumexp,
        None,
        None,
        max_seqlen_batch_q,
        max_seqlen_batch_k,
        convert_tensor(torch.empty((), dtype=torch.long, device="meta"), query.device),
        convert_tensor(torch.empty((), dtype=torch.long, device="meta"), query.device),
        debug_mask,
    )
@register_op_impl(aten._scaled_dot_product_efficient_attention.default)
def meta__scaled_dot_product_efficient(fake_mode, func, *args, **kwargs):
    _, kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # 获取关键字参数中的查询(query)、键(key)、值(value)和计算 logsumexp 的标志
    query = kwargs["query"]
    key = kwargs["key"]
    value = kwargs["value"]
    compute_log_sumexp = kwargs["compute_log_sumexp"]
    # unused: attn_bias, dropout_p, is_causal, scale

    def convert_tensor(t, device):
        return FakeTensor(fake_mode, t, device)

    # 将查询(query)、键(key)、值(value)转置，调整维度顺序
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # 获取查询(query)的批量大小(B)、序列长度(M)、键的序列长度(N)、注意力头数(num_heads)、查询的维度(K)、值的维度(Kv)
    B = query.size(0)
    M = query.size(1)
    N = key.size(1)
    num_heads = query.size(-2)
    K = query.size(-1)
    Kv = value.size(-1)

    # 创建一个结果张量(res)，用于存储注意力计算的输出
    res = convert_tensor(
        torch.empty(B, M, num_heads, Kv, dtype=query.dtype, device="meta"),
        query.device,
    )

    # 计算 logsumexp 所需的维度(logsumexp_dim)，如果 compute_log_sumexp 为真，则向上取整到最近的32的倍数，否则为0
    logsumexp_dim = math.ceil(M / 32) * 32 if compute_log_sumexp else 0
    # 创建 logsumexp 张量，用于存储 logsumexp 计算的结果
    logsum_exp = convert_tensor(
        torch.empty(
            (B, num_heads, logsumexp_dim),
            dtype=torch.float,
            device="meta",
        ),
        query.device,
    )

    # 将结果张量(res)的维度再次转置，使其符合输出要求
    res = res.transpose(1, 2)

    # 创建种子(seed)和偏移量(offset)，用于随机数生成的种子和偏移
    # See Note [Seed and Offset]:
    seed = convert_tensor(
        torch.empty((), dtype=torch.long, device="meta"), query.device
    )
    offset = convert_tensor(
        torch.empty((), dtype=torch.long, device="meta"), query.device
    )

    return res, logsum_exp, seed, offset


@register_op_impl(aten._flash_attention_forward.default)
def meta__flash_attention_forward(fake_mode, func, *args, **kwargs):
    _, kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # 获取关键字参数中的查询(query)、键(key)、累计查询序列(cum_seq_q)、累计键序列(cum_seq_k)、最大查询长度(max_q)、最大键长度(max_k)和返回调试掩码(return_debug_mask)标志
    query = kwargs["query"]
    key = kwargs["key"]
    cum_seq_q = kwargs["cum_seq_q"]
    cum_seq_k = kwargs["cum_seq_k"]
    max_q = kwargs["max_q"]
    max_k = kwargs["max_k"]
    return_debug_mask = kwargs["return_debug_mask"]
    # unused: value, dropout_p, is_causal, scale
    # unused: seqused_k, alibi_slopes, window_size_left, window_size_right

    def convert_tensor(t, device):
        return FakeTensor(fake_mode, t, device)

    # 根据累计序列的情况确定批处理大小和最大序列长度
    # NB: there are two underlying paths:
    # 1. normal dense path; expect 4D inputs of shape (batch_size, seqlen, num_heads, head_dim)
    # 2. varseqlen path; expect 3D inputs of shape (total, num_heads, head_dim) where total
    #    includes all batch item sequences. cum_seq_q / cum_seq_k contain offsets into total
    batch_size = query.size(0) if cum_seq_q is None else cum_seq_q.numel() - 1
    max_seqlen_batch_q = query.size(1) if cum_seq_q is None else max_q
    max_seqlen_batch_k = key.size(1) if cum_seq_k is None else max_k
    num_heads = query.size(-2)
    head_dim = query.size(-1)

    # 创建一个注意力张量(attention)，形状与查询(query)相同，用于存储注意力计算的输出
    # Cuda Path
    # 注意: empty_like 已经返回一个伪造的张量，我们不需要再次封装它
    attention = torch.empty_like(query)
    # 创建一个张量 `logsumexp`，用于存储对数求和指数的计算结果，形状为(batch_size, num_heads, max_seqlen_batch_q)，数据类型为浮点型，存储在"meta"设备上。
    logsumexp = convert_tensor(
        torch.empty(
            (batch_size, num_heads, max_seqlen_batch_q),
            dtype=torch.float,
            device="meta",
        ),
        device=query.device,
    )
    
    # 如果需要返回调试掩码 (`return_debug_mask` 为真)：
    if return_debug_mask:
        # 根据头部维度 (`head_dim`) 的大小确定块大小 (`blocksize_c`)，128 或 256。
        blocksize_c = 128 if head_dim > 64 else 256
        # 计算 `max_seqlen_k`，用于限制序列长度。
        max_seqlen_k = math.ceil(max_seqlen_batch_q / blocksize_c)
        # 根据 `max_seqlen_batch_k` 的大小调整 `max_seqlen_k`。
        if max_seqlen_batch_k <= 128:
            max_seqlen_k = 128
        elif max_seqlen_batch_k <= 256:
            max_seqlen_k = 256
        # 创建调试掩码张量 `debug_mask`，形状为(batch_size, num_heads, max_seqlen_batch_q, max_seqlen_k)，数据类型与查询张量 (`query`) 相同，存储在"meta"设备上。
        debug_mask = convert_tensor(
            torch.empty(
                (batch_size, num_heads, max_seqlen_batch_q, max_seqlen_k),
                dtype=query.dtype,
                device="meta",
            ),
            query.device,
        )
    # 如果不需要返回调试掩码：
    else:
        # 创建空张量 `debug_mask`，形状为(0,)，数据类型与查询张量 (`query`) 相同，存储在"meta"设备上。
        debug_mask = convert_tensor(
            torch.empty(0, dtype=query.dtype, device="meta"),
            query.device,
        )
    
    # 返回以下张量作为结果：
    return (
        attention,  # 注意力机制计算结果
        logsumexp,  # 对数求和指数计算结果
        convert_tensor(torch.empty((), dtype=torch.long, device="meta"), query.device),  # 空的长整型张量
        convert_tensor(torch.empty((), dtype=torch.long, device="meta"), query.device),  # 空的长整型张量
        debug_mask,  # 调试掩码张量
    )
@register_op_impl(aten._efficient_attention_forward.default)
# 注册一个函数实现，用于处理 efficient_attention_forward 的默认实现
def meta__efficient_attention_forward(fake_mode, func, *args, **kwargs):
    _, kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    query = kwargs["query"]
    key = kwargs["key"]
    value = kwargs["value"]
    cu_seqlens_q = kwargs["cu_seqlens_q"]
    max_seqlen_q = kwargs["max_seqlen_q"]
    max_seqlen_k = kwargs["max_seqlen_k"]
    compute_log_sumexp = kwargs["compute_log_sumexp"]
    # unused: bias, cu_seqlens_k, dropout_p, custom_mask_type, scale, seqlen_k

    def convert_tensor(t, device):
        return FakeTensor(fake_mode, t, device)

    B = query.size(0)           # 获取 query 的 batch size
    M = query.size(1)           # 获取 query 的序列长度
    N = key.size(1)             # 获取 key 的序列长度
    num_heads = query.size(-2)  # 获取注意力头的数量
    K = query.size(-1)          # 获取 query 的特征维度
    Kv = value.size(-1)         # 获取 value 的特征维度

    res = convert_tensor(
        torch.empty(B, M, num_heads, Kv, dtype=query.dtype, device="meta"),
        query.device,
    )

    logsumexp_batch_dim = cu_seqlens_q.size(0) - 1 if (cu_seqlens_q is not None) else B
    actual_max_seqlen_q = M
    if cu_seqlens_q is not None:
        assert max_seqlen_q is not None
        actual_max_seqlen_q = max_seqlen_q
    actual_max_seqlen_k = max_seqlen_k if max_seqlen_k is not None else N
    logsumexp_dim = (
        math.ceil(actual_max_seqlen_q / 32) * 32 if compute_log_sumexp else 0
    )
    logsum_exp = convert_tensor(
        torch.empty(
            (logsumexp_batch_dim, num_heads, logsumexp_dim),
            dtype=torch.float,
            device="meta",
        ),
        query.device,
    )

    # See Note [Seed and Offset]:
    seed = convert_tensor(
        torch.empty((), dtype=torch.long, device="meta"), query.device
    )
    offset = convert_tensor(
        torch.empty((), dtype=torch.long, device="meta"), query.device
    )

    return res, logsum_exp, seed, offset, actual_max_seqlen_q, actual_max_seqlen_k


@register_op_impl(torch.ops.aten._pack_padded_sequence.default)
# 注册一个函数实现，用于处理 _pack_padded_sequence 的默认实现
def _pack_padded_sequence(fake_mode, func, inputs, lengths, batch_first):
    if (
        fake_mode.shape_env is None
        or not fake_mode.shape_env.allow_dynamic_output_shape_ops
    ):
        # 如果没有符号整数或符号浮点数支持，则无法处理此操作
        raise DynamicOutputShapeException(func)

    new_batch_size = fake_mode.shape_env.create_unbacked_symint()

    from torch.fx.experimental.symbolic_shapes import _constrain_range_for_size

    _constrain_range_for_size(new_batch_size)

    if not batch_first:
        # 如果不是 batch_first，将输入张量进行转置，使其形状变为 (seq_len, batch_size, *)
        inputs = inputs.transpose(0, 1)

    res_size = inputs.shape[1:]         # 获取输入数据的形状，去除第一个维度（batch_size）
    packed_data = inputs.new_empty(res_size)
    batch_size = inputs.new_empty((new_batch_size,))
    return (packed_data, batch_size)


FAST_OP_IMPLEMENTATIONS = {}
# 空的快速操作实现字典，用于存储快速操作的注册信息

# Unlike register_op_impl, these don't do the slow iteration for
# run_impl_check, and these run BEFORE decompositions
# 与 register_op_impl 不同，这些函数不会进行缓慢的迭代检查，且在分解之前运行
def register_fast_op_impl(func: OpOverload):
    # 注册一个快速操作实现函数
    # 定义一个装饰器函数，接受一个操作实现函数作为参数
    def impl_decorator(op_impl):
        # 将操作实现函数存储在全局字典 FAST_OP_IMPLEMENTATIONS 中，键为 func
        FAST_OP_IMPLEMENTATIONS[func] = op_impl
        # 返回操作实现函数
        return op_impl

    # 返回定义的装饰器函数
    return impl_decorator
# 定义函数 infer_size，用于推断两个张量的扩展尺寸
def infer_size(a, b):
    # 导入 torch.fx.experimental.symbolic_shapes 模块中的 guard_size_oblivious 函数
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

    # 计算张量 a 和 b 的维度
    dimsA = len(a)
    dimsB = len(b)
    # 确定最大维度作为扩展后的维度
    ndim = max(dimsA, dimsB)
    # 初始化扩展后的尺寸列表为全零
    expandedSizes = [0] * ndim
    # 从最高维到最低维遍历
    for i in range(ndim - 1, -1, -1):
        # 计算当前维度的偏移量
        offset = ndim - 1 - i
        # 计算在张量 a 和 b 中对应的维度索引
        dimA = dimsA - 1 - offset
        dimB = dimsB - 1 - offset
        # 获取张量 a 和 b 在当前维度上的尺寸，若索引为负数则为 1
        sizeA = a[dimA] if dimA >= 0 else 1
        sizeB = b[dimB] if dimB >= 0 else 1

        # 重要提示：在测试 sizeA == sizeB 之前，先测试是否需要广播
        # 这是因为广播测试可能在静态情况下已知（特别是如果 sizeA/sizeB 是未备份但类似大小的情况，我们会错误地假设它们永远不等于 1），
        # 但 sizeA == sizeB 的测试可能在静态情况下未知。
        # 然而，一旦我们确定没有发生广播，sizeA == sizeB 现在是预期的结果，我们可以将其推迟为运行时的断言（这是因为 Python 会直接返回 or 语句的终端表达式，而不会对其进行 bool() 处理；
        # 如果情况不是这样，我们需要使用 torch.sym_or() 或类似的方法编写此代码）。
        torch._check(
            guard_size_oblivious(sizeA == 1)
            or guard_size_oblivious(sizeB == 1)
            or sizeA == sizeB,
            lambda: f"The size of tensor a ({sizeA}) "
            f"must match the size of tensor b ({sizeB}) "
            f"at non-singleton dimension {i})",
        )
        # 根据 sizeA 是否为 1 来确定在当前维度上选择 sizeB 还是 sizeA
        expandedSizes[i] = sizeB if guard_size_oblivious(sizeA == 1) else sizeA
    # 返回扩展后的尺寸元组
    return tuple(expandedSizes)


# 定义函数 make_fast_binary_impl，返回 fast_binary_impl 函数
def make_fast_binary_impl(slow_ref):
    return fast_binary_impl


# 使用 functools.lru_cache(None) 装饰，定义函数 get_fast_op_impls，返回快速操作的实现
@functools.lru_cache(None)
def get_fast_op_impls():
    # 导入 torch._refs 模块
    import torch._refs

    # 分别注册 torch.ops.aten.add.Tensor、torch.ops.aten.sub.Tensor、torch.ops.aten.mul.Tensor、torch.ops.aten.div.Tensor 的快速操作实现
    register_fast_op_impl(torch.ops.aten.add.Tensor)(
        make_fast_binary_impl(torch._refs.add)
    )
    register_fast_op_impl(torch.ops.aten.sub.Tensor)(
        make_fast_binary_impl(torch._refs.sub)
    )
    register_fast_op_impl(torch.ops.aten.mul.Tensor)(
        make_fast_binary_impl(torch._refs.mul))  # type: ignore[has-type]
    register_fast_op_impl(torch.ops.aten.div.Tensor)(
        make_fast_binary_impl(torch._refs.div)
    )
    # 返回 FAST_OP_IMPLEMENTATIONS
    return FAST_OP_IMPLEMENTATIONS
```