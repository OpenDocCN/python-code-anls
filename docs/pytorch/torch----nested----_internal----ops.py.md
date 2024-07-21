# `.\pytorch\torch\nested\_internal\ops.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和函数
import functools
import math
import operator

import torch
from torch.nested._internal.sdpa import jagged_scaled_dot_product_attention

# 从当前目录导入NestedTensor类
from .nested_tensor import NestedTensor
from typing import *  # noqa: F403
import torch.nn.functional as F
from torch.fx.operator_schemas import normalize_function

# 定义全局变量，用于存储公开的模块成员
__all__: List[Any] = []

# 定义空的字典，用于存储操作表
JAGGED_OPS_TABLE: Dict[Any, Any] = {}


# 函数用于确定将指定维度dim进行转换为内部维度还是外部维度
# 简化假设：我们假设批处理维度始终是最左边的维度，不规则维度始终是第二个维度。
def _outer_to_inner_dim(ndim, dim):
    assert dim >= 0 and dim < ndim
    return 0 if dim < 2 else dim - 1


# 函数用于将指定维度dim封装为内部维度，并检查是否允许批处理维度。
def _wrap_jagged_dim(
    ndim, dim, op_name, convert_to_inner_dim=True, allow_batch_dim=False
):
    from torch._prims_common import canonicalize_dims

    # 使用canonicalize_dims函数将维度进行规范化处理
    wrapped = canonicalize_dims(ndim, dim)
    if wrapped == 1:
        # 如果封装后的维度为1，则抛出异常，因为不支持在dim=1上进行操作
        raise RuntimeError(f"{op_name}(): not supported for NestedTensor on dim=1")
    elif wrapped == 0 and not allow_batch_dim:
        # 如果封装后的维度为0且不允许批处理维度，则抛出异常
        raise RuntimeError(f"{op_name}(): not supported for NestedTensor on dim=0")
    # 将封装后的维度转换为内部维度，如果convert_to_inner_dim为True
    return _outer_to_inner_dim(ndim, wrapped) if convert_to_inner_dim else wrapped


# 函数用于将指定维度dims封装为内部维度，并检查是否允许在批处理维度上操作。
def _wrap_jagged_dims(ndim, dims, op_name):
    # ex: (2, 3, 4) -> (1, 2, 3)
    # ex: (0, 1, 4) -> (0, 3)
    from torch._prims_common import canonicalize_dims

    # 使用canonicalize_dims函数将维度进行规范化处理
    wrapped_dims = [canonicalize_dims(ndim, d) for d in dims]
    # 在进行映射到内部维度之前，需要进行下面的逻辑处理，以便能够打印更友好的错误消息。
    zero_in_dims = 0 in wrapped_dims
    one_in_dims = 1 in wrapped_dims
    if zero_in_dims ^ one_in_dims:
        apply, not_apply = ("batch", "ragged") if zero_in_dims else ("ragged", "batch")
        # 如果在应用于batch维度时不适用于ragged维度，则抛出异常。
        raise RuntimeError(
            f"{op_name}(): applying over the {apply} dimension, but not the {not_apply}"
            " dimension is not supported for NestedTensor"
        )
    # 返回封装后的维度和是否存在0维的标志
    return (
        tuple(_outer_to_inner_dim(ndim, d) for d in dims if d != 0),
        zero_in_dims,
    )


# 函数用于检查函数的参数个数是否符合预期，并根据特定的schema_str进行检查。
def check_schema(schema_str: str, func, *args, **kwargs) -> None:
    named_arg_types = schema_str.split(", ")
    num_optional_args = [x.endswith("?") for x in named_arg_types].count(True)
    min_args = len(named_arg_types) - num_optional_args

    # 特殊情况：如果schema_str以"..."结尾，则允许任意数量的未检查参数
    if named_arg_types[-1] == "...":
        named_arg_types = named_arg_types[:-1]
    else:
        # 否则，检查传入参数的数量是否在预期范围内
        if not (len(args) >= min_args and len(args) <= len(named_arg_types)):
            raise ValueError(
                f"NestedTensor {func.__name__}({schema_str}): expected at least {min_args} "
                f"arguments and at most {len(named_arg_types)} arguments, but got: "
                f"{len(args)} arguments"
            )
    # 定义用于参数类型检查的函数字典，每个键是参数类型的缩写，值是对应的类型检查函数
    arg_type_check_fns = {
        "t": lambda x: isinstance(x, torch.Tensor) and not isinstance(x, NestedTensor),
        # 对于类型为 "t" 的参数，要求其为 torch.Tensor 类型且不是 NestedTensor 类型的实例
        "jt": lambda x: isinstance(x, NestedTensor)
        and x._lengths is None
        and x._ragged_idx == 1,  # 对于类型为 "jt" 的参数，要求其为具有连续布局的 NestedTensor
        "jt_all": lambda x: isinstance(
            x, NestedTensor
        ),  # 对于类型为 "jt_all" 的参数，要求其为任意布局的 NestedTensor
        "any": lambda x: True,  # 对于类型为 "any" 的参数，接受任意类型
    }
    # 遍历命名参数类型列表
    for i, named_arg_type in enumerate(named_arg_types):
        # 解析参数名和类型
        name, arg_type = named_arg_type.split(": ")
        # 检查类型是否可选
        is_optional = arg_type.endswith("?")
        # 如果类型可选，截取类型字符串
        normalized_arg_type = arg_type[:-1] if is_optional else arg_type
        # 如果参数类型不在预定义的类型检查函数中，则抛出异常
        if normalized_arg_type not in arg_type_check_fns.keys():
            raise AssertionError(f"Unknown arg type: {normalized_arg_type}")

        # 如果参数索引超出实际参数个数，则检查是否可选，若非可选则抛出缺少参数异常
        if i >= len(args):
            if not is_optional:
                raise ValueError(
                    f"NestedTensor {func.__name__}({schema_str}) "
                    f"missing required argument: {name}"
                )
            continue

        # 获取当前参数类型对应的检查函数
        _check_fn = arg_type_check_fns[normalized_arg_type]

        # 定义参数检查函数，可选参数默认为 None，检查参数类型是否符合预期
        def check_fn(x, is_optional=is_optional):
            if is_optional:
                return x is None or _check_fn(x)
            else:
                return _check_fn(x)

        # 如果当前参数不符合预期类型，则抛出值错误异常，描述预期的参数类型
        if not check_fn(args[i]):
            type_to_desc = {
                "t": "tensor",
                "t?": "optional tensor",
                "jt": "contiguous jagged layout NestedTensor",
                "jt_all": "jagged layout NestedTensor",
                "any": "<any type>",
            }
            raise ValueError(
                f"NestedTensor {func.__name__}({schema_str}): expected {name} to be a "
                f"{type_to_desc[arg_type]}"
            )
# 检查两个NestedTensor对象的指定维度是否相同
def check_ragged_dim_same(
    func, a: NestedTensor, a_name: str, b: NestedTensor, b_name: str
) -> None:
    # 调用shape属性检查维度信息
    if a._size[a._ragged_idx] != b._size[b._ragged_idx]:
        # 如果指定维度的大小不同，则抛出运行时错误
        raise RuntimeError(
            f"NestedTensor {func.__name__}: expected {a_name} and {b_name} to have the "
            "same exact offsets tensor."
        )


# 返回True，如果NestedTensor的raggedness相关部分的形状与指定的大小匹配
def raggedness_matches(nt, size):
    end = nt._ragged_idx + 1
    # 提取NestedTensor的raggedness相关部分
    nt_ragged = nt._size[:end]
    # 提取指定大小的raggedness相关部分
    size_ragged = size[:end]
    # 返回是否长度相同且每个对应位置的元素相等或者指定为-1
    return len(nt_ragged) == len(size_ragged) and (
        all(ns == s or s == -1 for ns, s in zip(nt_ragged, size_ragged))
    )


def squeeze_leading_ones(t):
    # 注意: [挤压前导的1]
    #
    # 从张量t中挤压前导的1。
    #
    # 我们希望:
    #   (B, j0, ?, ?) + (1, 1, ?, ?) -> (B, j0, ?, ?)
    #   (B, j0, ?, ?) + (1, 1, 1, ?, ?) -> (1, B, j0, ?, ?)  (目前不支持)
    #
    # 1) 挤压额外的1并从NT中获取值
    #   (1, 1, ?, ?) -> (?, ?)   和   (sum(*), ?, ?) -> (B, j0, ?, ?)
    # 2) 进行密集的广播:
    #   (sum(*), ?, ?) + (?, ?) -> (sum(*), ?, ?)
    # 3) 构造嵌套张量
    #   (sum(*), ?, ?) -> (B, j0, ?, ?)
    #
    # 如果在第0维上的unsqueeze操作得到支持，我们将在第(4)步进行unsqueeze，
    # 并需要更新此函数以记录我们unsqueeze了多少个1。
    while t.shape[0] == 1:
        t = t.squeeze(0)
    return t


def register_func(tables, aten_ops, schema_str):
    if not isinstance(aten_ops, list):
        aten_ops = [aten_ops]
    if not isinstance(tables, list):
        tables = [tables]

    def wrapper(func):
        for aten_op in aten_ops:

            def get_inner(aten_op):
                def inner(*args, **kwargs):
                    # 检查schema是否匹配，然后调用func函数
                    check_schema(schema_str, func, *args, **kwargs)
                    return func(aten_op, *args, **kwargs)

                return inner

            for table in tables:
                # 将inner函数注册到tables中的aten_op上
                table[aten_op] = get_inner(aten_op)
        return func

    return wrapper


# 使用register_func函数部分应用，将JAGGED_OPS_TABLE作为tables参数传递
register_jagged_func = functools.partial(register_func, JAGGED_OPS_TABLE)


def lookup_jagged(func, *args, **kwargs) -> Optional[Callable]:
    # 获取func在JAGGED_OPS_TABLE中对应的dispatch_func
    dispatch_func = JAGGED_OPS_TABLE.get(func, None)
    if dispatch_func is not None:
        return dispatch_func

    # 处理逐点回退
    if torch.Tag.pointwise in func.tags:
        # 假设不存在不是"unary/binary"参数的其他张量
        num_tensor_args = sum(isinstance(x, torch.Tensor) for x in args)
        if num_tensor_args == 1:
            # 检查schema是否匹配"self: jt_all, ..."
            check_schema("self: jt_all, ...", func, *args, **kwargs)
            # 返回部分应用的jagged_unary_pointwise函数
            return functools.partial(jagged_unary_pointwise, func)
        elif num_tensor_args == 2:
            # 检查schema是否匹配"lhs: any, rhs: any, ..."
            check_schema("lhs: any, rhs: any, ...", func, *args, **kwargs)
            # 返回部分应用的jagged_binary_pointwise函数
            return functools.partial(jagged_binary_pointwise, func)

    return None
# 定义一个函数，从输入参数 `arg` 中提取关键字参数并返回
def extract_kwargs(arg):
    kwargs = {
        "offsets": arg.offsets(),  # 调用 `arg` 对象的 `offsets()` 方法，获取偏移量信息
        "_metadata_cache": arg._metadata_cache,  # 将 `arg` 对象的 `_metadata_cache` 属性赋给关键字参数
        "_ragged_idx": arg._ragged_idx,  # 将 `arg` 对象的 `_ragged_idx` 属性赋给关键字参数
    }
    return kwargs  # 返回包含关键字参数的字典


# 定义一个函数，执行针对嵌套张量的一元逐点操作
def jagged_unary_pointwise(func, *args, **kwargs):
    return NestedTensor(
        func(args[0]._values, *args[1:], **kwargs), **extract_kwargs(args[0])
    )
    # 使用给定的函数 `func` 对第一个参数 `args[0]._values` 进行一元逐点操作，
    # 并用 `extract_kwargs` 函数提取第一个参数的关键字参数，创建一个新的 NestedTensor 对象


# 定义一个函数，执行针对嵌套张量的二元逐点操作
def jagged_binary_pointwise(func, *args, **kwargs):
    a, b = args[0], args[1]
    assert isinstance(a, NestedTensor) or isinstance(b, NestedTensor)

    mismatch_error_msg = (
        "cannot call binary pointwise function {} with inputs of shapes {} and {}"
    )
    
    # 如果 a 和 b 都是 NestedTensor 类型的对象
    if isinstance(a, NestedTensor) and isinstance(b, NestedTensor):
        # 检查嵌套度是否匹配
        if raggedness_matches(a, b._size):
            return NestedTensor(
                func(a._values, b._values, *args[2:], **kwargs), **extract_kwargs(a)
            )
        raise RuntimeError(mismatch_error_msg.format(func.__name__, a._size, b._size))
    
    # 如果 a 或者 b 是 NestedTensor 类型的对象
    a_is_nt = isinstance(a, NestedTensor)
    extracted_kwargs = extract_kwargs(a) if a_is_nt else extract_kwargs(b)

    # 处理跨批次/不规则维度的广播情况
    nt, t = (a, b) if a_is_nt else (b, a)
    
    # 如果 t 的维度大于 nt 的维度，抛出未实现错误
    if t.dim() > nt.dim():
        raise NotImplementedError("NYI: broadcasting NT with T with larger dim")
    
    # 对 t 进行挤压操作，去除前导的 1 维度
    t_squeezed = squeeze_leading_ones(t)
    
    # 如果 nt 的维度大于或等于 t_squeezed 的维度加 2
    if nt.dim() >= t_squeezed.dim() + 2:
        lhs, rhs = (nt._values, t_squeezed) if a_is_nt else (t_squeezed, nt._values)
        return NestedTensor(func(lhs, rhs, *args[2:], **kwargs), **extracted_kwargs)
    
    # 处理手动广播情况
    if a.dim() == b.dim():
        if a.shape[0] != b.shape[0]:  # 如果两个张量的第一维度大小不同，抛出运行时错误
            raise RuntimeError(mismatch_error_msg.format(func.__name__, a.shape, b.shape))
        
        # 使用偏移量手动广播未绑定的组件
        outputs = []
        for a_comp, b_comp in zip(a.unbind(), b.unbind()):
            outputs.append(func(a_comp, b_comp, *args[2:], **kwargs))
        new_values = torch.cat(outputs, dim=0)
        return NestedTensor(new_values, **extracted_kwargs)

    # 如果以上条件都不满足，抛出错误，因为这会破坏不规则维度相对于左侧最批次维度的不变性
    raise RuntimeError("Unexpected condition encountered in binary pointwise operation")
    # 抛出运行时错误，指示函数名称、a 和 b 的形状不匹配
    raise RuntimeError(mismatch_error_msg.format(func.__name__, a.shape, b.shape))
# 定义一个函数，用于处理包含嵌套张量的函数调用
def jagged_torch_function(func, *args, **kwargs):
    # SDPA 有专门处理嵌套张量的内核，这里分发到正确的实现
    if func is torch._C._nn.scaled_dot_product_attention:
        return jagged_scaled_dot_product_attention(*args, **kwargs)

    # 如果函数名为 "apply_"，调用它并返回第一个参数
    if func.__name__ == "apply_":
        func(args[0]._values, *args[1:], **kwargs)
        return args[0]

    # 处理 flatten() 函数，因为它是 CompositeImplicit 的一部分
    if func.__name__ == "flatten":

        # 定义一个内部函数 _flatten_sig
        def _flatten_sig(input, start_dim=0, end_dim=-1):
            pass

        # 调用 normalize_function 函数，规范化参数
        _, new_kwargs = normalize_function(
            _flatten_sig, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
        )

        # 从新规范化的参数中提取 input
        inp = new_kwargs.pop("input")

        # 确定 start_dim 和 end_dim 的值，确保在外部维度空间中，因为将在 NT 输入上重新分派
        start_dim = _wrap_jagged_dim(
            inp.dim(), new_kwargs["start_dim"], "flatten", convert_to_inner_dim=False
        )
        end_dim = _wrap_jagged_dim(
            inp.dim(), new_kwargs["end_dim"], "flatten", convert_to_inner_dim=False
        )

        # 如果 start_dim 等于 end_dim，则返回输入 inp
        if start_dim == end_dim:
            return inp

        # 计算新的形状，并返回重塑后的张量
        product = functools.reduce(operator.mul, inp.shape[start_dim : end_dim + 1])
        new_shape = (*inp.shape[:start_dim], product, *inp.shape[end_dim + 1 :])
        return inp.reshape(*new_shape)

    # 如果未匹配到任何已知函数，则抛出 NotImplementedError
    raise NotImplementedError(func)


# 注册 jagged 函数，包括一系列函数和其别名
@register_jagged_func(
    [
        torch.ops.aten.is_non_overlapping_and_dense.default,
        torch.ops.aten.sym_size.default,
        torch.ops.aten.dim.default,
        torch.ops.aten.numel.default,
        torch.ops.aten.sym_numel.default,
        torch.ops.aten.sym_stride.default,
        torch.ops.aten.sym_storage_offset.default,
    ],
    "self: jt_all",
)
# 定义一个函数，用于获取张量属性，根据不同的函数名返回不同的属性值
def tensor_attr_supported_getter(func, *args, **kwargs):
    if func == torch.ops.aten.is_non_overlapping_and_dense.default:
        return False

    if func == torch.ops.aten.sym_size.default:
        return args[0]._size

    if func == torch.ops.aten.dim.default:
        return len(args[0]._size)

    if func in (torch.ops.aten.sym_numel.default, torch.ops.aten.numel.default):
        if args[0]._lengths is not None:
            return int(sum(args[0]._lengths) * math.prod(args[0]._size[2:]))
        return args[0]._values.numel()

    if func == torch.ops.aten.sym_stride.default:
        return args[0]._strides

    if func == torch.ops.aten.sym_storage_offset.default:
        return args[0]._values.storage_offset()


# 注册 jagged 函数，处理 prim.layout.default 函数的调用
@register_jagged_func(torch.ops.prim.layout.default, "self: jt_all")
def prim_layout_default(func, *args, **kwargs):
    return torch.jagged


# 注册 jagged 函数，处理 tensor_attr_unsupported_getter 函数的调用
@register_jagged_func(
    [torch.ops.aten.size.default],
    "self: jt_all",
)
def tensor_attr_unsupported_getter(func, *args, **kwargs):
    pass
    # 检查 func 是否等于 torch.ops.aten.size.default
    if func == torch.ops.aten.size.default:
        # 如果相等，抛出运行时错误并显示提示信息
        raise RuntimeError(
            "NestedTensors does not support directly calling torch.ops.aten.size "
            "please use `nested_tensor.size()` instead."
        )
# 注册一个自定义的装饰器函数，用于将 torch.ops.aten.is_contiguous.default 函数注册为 jagged_func，并指定参数 "self: jt_all"
@register_jagged_func(torch.ops.aten.is_contiguous.default, "self: jt_all")
def is_contiguous_general(func, *args, **kwargs):
    # 从 torch._prims_common 模块导入 is_contiguous_for_memory_format 函数
    from torch._prims_common import is_contiguous_for_memory_format

    # 调用 normalize_function 函数对输入的 func 函数进行规范化，并仅使用 kwargs
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    # 从 new_kwargs 中弹出名为 "input" 的参数，赋值给 inp
    inp = new_kwargs.pop("input")

    # 如果 inp 是由 narrow() 函数创建的，则检查其长度是否为 None
    if inp.lengths() is not None:
        return False

    # 设置 new_kwargs 中的 "memory_format" 参数，默认为 torch.contiguous_format
    new_kwargs["memory_format"] = new_kwargs.get(
        "memory_format", torch.contiguous_format
    )
    # 如果 "memory_format" 等于 torch.preserve_format，则返回 True
    if new_kwargs["memory_format"] == torch.preserve_format:
        return True
    # 否则调用 is_contiguous_for_memory_format 函数，检查 inp._values 是否符合给定的内存格式要求
    return is_contiguous_for_memory_format(inp._values, **new_kwargs)


# 将 is_contiguous_general 函数注册为 jagged_func，并指定参数 "self: jt_all, memory_format: any?"
register_jagged_func(
    torch.ops.aten.is_contiguous.memory_format, "self: jt_all, memory_format: any?"
)(is_contiguous_general)


# 注册一个自定义的装饰器函数，用于将 torch.ops.aten.linear.default 函数注册为 jagged_func，并指定参数 "input: jt, weight: t, bias: t?"
@register_jagged_func(torch.ops.aten.linear.default, "input: jt, weight: t, bias: t?")
def linear_default(func, *args, **kwargs):
    # 调用 normalize_function 函数对输入的 func 函数进行规范化，并仅使用 kwargs
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # 从 new_kwargs 中弹出名为 "input" 的参数，赋值给 inp
    inp = new_kwargs.pop("input")

    # 调用 func 函数对 inp._values 进行线性操作，并返回 NestedTensor 对象
    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


# 注册一个自定义的装饰器函数，用于将 torch.ops.aten.linear_backward.default 函数注册为 jagged_func，并指定参数 "self: jt, grad_output: jt, weight: t, output_mask: any"
@register_jagged_func(
    torch.ops.aten.linear_backward.default,
    "self: jt, grad_output: jt, weight: t, output_mask: any",
)
def linear_backward_default(func, *args, **kwargs):
    # 调用 normalize_function 函数对输入的 func 函数进行规范化，并仅使用 kwargs
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # 从 new_kwargs 中依次弹出名为 "input", "grad_output", "weight" 的参数，分别赋值给 inp, grad_output, weight
    inp = new_kwargs.pop("input")
    grad_output = new_kwargs.pop("grad_output")
    weight = new_kwargs.pop("weight")

    # 检查输入张量 inp 和 grad_output 的 ragged 维度是否相同
    check_ragged_dim_same(func, inp, "self", grad_output, "grad_output")
    # 计算 ds（损失函数对 self 的梯度）、dw（损失函数对 weight 的梯度）、db（暂未实现对 bias 的梯度计算）
    ds = NestedTensor(
        torch.matmul(grad_output._values, weight), **extract_kwargs(grad_output)
    )
    dw = torch.matmul(grad_output._values.transpose(-2, -1), inp._values)
    db = None  # NYI: gradient for bias, need to reduce over ragged dim
    return (ds, dw, db)


# 注册一个自定义的装饰器函数，用于将 torch.ops.aten._to_copy.default 函数注册为 jagged_func，并指定参数 "self: jt_all"
@register_jagged_func(torch.ops.aten._to_copy.default, "self: jt_all")
def to_copy_default(func, *args, **kwargs):
    # 从当前模块的 nested_tensor 中导入 _tensor_symint_registry 对象
    from .nested_tensor import _tensor_symint_registry

    # 调用 normalize_function 函数对输入的 func 函数进行规范化，并仅使用 kwargs
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # 从 new_kwargs 中弹出名为 "input" 的参数，赋值给 inp
    inp = new_kwargs.pop("input")
    # 从 new_kwargs 中移除 "layout" 参数
    new_kwargs.pop("layout")

    # 调用 func 函数对 inp._values 进行复制操作，并返回新的 NestedTensor 对象
    new_values = func(inp._values, **new_kwargs)
    # 将原始输入 inp 的偏移信息转移到新的 NestedTensor 对象中
    new_offsets = inp._offsets.to(device=new_values.device)
    _tensor_symint_registry[new_offsets] = _tensor_symint_registry[inp._offsets]
    inp_kwargs = extract_kwargs(inp)
    inp_kwargs["offsets"] = new_offsets

    return NestedTensor(new_values, **inp_kwargs)


# 将 to_copy_default 函数注册为 jagged_func，并指定多个 torch.ops.aten.*.default 函数列表作为其函数对象，参数为 "self: jt_all"
register_jagged_func(
    [
        torch.ops.aten.empty_like.default,
        torch.ops.aten.ones_like.default,
        torch.ops.aten.zeros_like.default,
        torch.ops.aten.randn_like.default,
        torch.ops.aten.detach.default,
    ],
    "self: jt_all",
)(jagged_unary_pointwise)
# 注册一个函数，将 torch.ops.aten.zero_.default 映射为 "self: jt_all" 的 jagged 函数
@register_jagged_func(torch.ops.aten.zero_.default, "self: jt_all")
def zero__default(func, *args, **kwargs):
    # 调用 normalize_function 函数，将 func、args、kwargs 标准化，仅使用 kwargs
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # 从 new_kwargs 中弹出 "input" 对应的值
    inp = new_kwargs.pop("input")
    # 调用 func 函数，作用于 inp._values
    func(inp._values)
    # 返回原始的 inp
    return inp


# 注册一个函数，将 torch.ops.aten._softmax.default 映射为 "self: jt, dim: any, half_to_float: any" 的 jagged 函数
def _softmax_default(func, *args, **kwargs):
    # 调用 normalize_function 函数，将 func、args、kwargs 标准化，仅使用 kwargs
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # 从 new_kwargs 中弹出 "input" 对应的值
    inp = new_kwargs.pop("input")
    # 从 new_kwargs 中取出 "dim"
    dim = new_kwargs["dim"]
    # 将 "dim" 标准化为包含 jagged 格式信息的形式
    new_kwargs["dim"] = _wrap_jagged_dim(len(inp._size), dim, "softmax")

    # 调用 func 函数，作用于 inp._values，返回结果构造成 NestedTensor
    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


# 注册一个函数，将 torch.ops.aten.native_dropout.default 映射为 "self: jt, float: any, train: any?" 的 jagged 函数
def native_dropout_default(func, *args, **kwargs):
    # 调用 normalize_function 函数，将 func、args、kwargs 标准化，仅使用 kwargs
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # 从 new_kwargs 中弹出 "input" 对应的值
    inp = new_kwargs.pop("input")
    # 调用 func 函数，作用于 inp._values，返回结果构造成两个 NestedTensor
    out1, out2 = func(inp._values, **new_kwargs)
    return (
        NestedTensor(out1, **extract_kwargs(inp)),
        NestedTensor(out2, **extract_kwargs(inp)),
    )


# 注册一个函数，将 torch.ops.aten.native_dropout_backward.default 映射为 "grad_output: jt, mask: jt, scale: any" 的 jagged 函数
def native_dropout_backward_default(func, *args, **kwargs):
    # 调用 normalize_function 函数，将 func、args、kwargs 标准化，仅使用 kwargs
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    # 从 new_kwargs 中弹出 "grad_output" 和 "mask" 对应的值
    grad_output = new_kwargs.pop("grad_output")
    mask = new_kwargs.pop("mask")
    # 调用 func 函数，作用于 grad_output._values 和 mask._values，返回结果构造成 NestedTensor
    return NestedTensor(
        func(grad_output._values, mask._values, **new_kwargs),
        **extract_kwargs(grad_output),
    )


# 注册一个函数，将 torch.ops.aten.prod.dim_int 映射为 "self: jt, dim: any, keepdim: any?" 的 jagged 函数
def prod_dim_int(func, *args, **kwargs):
    # 调用 normalize_function 函数，将 func、args、kwargs 标准化，仅使用 kwargs
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # 从 new_kwargs 中弹出 "input" 对应的值
    inp = new_kwargs.pop("input")
    # 如果 new_kwargs["keepdim"] 为 False，则抛出异常
    if not new_kwargs["keepdim"]:
        raise RuntimeError("prod(): keepdim=True must be set for NestedTensor")
    # 从 new_kwargs 中取出 "dim"
    dim = new_kwargs["dim"]
    # 将 "dim" 标准化为包含 jagged 格式信息的形式
    new_kwargs["dim"] = _wrap_jagged_dim(len(inp._size), dim, "prod")

    # 调用 func 函数，作用于 inp._values，返回结果构造成 NestedTensor
    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(args[0]))


# 注册一个函数，将 torch.ops.aten.split.Tensor 映射为 "self: jt, split_size: any, dim: any" 的 jagged 函数
def split_tensor(func, *args, **kwargs):
    # 调用 normalize_function 函数，将 func、args、kwargs 标准化，仅使用 kwargs
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # 从 new_kwargs 中弹出 "input" 对应的值
    inp = new_kwargs.pop("input")
    # 将 "dim" 标准化为包含 jagged 格式信息的形式
    new_kwargs["dim"] = _wrap_jagged_dim(inp.dim(), new_kwargs["dim"], "split")

    # 调用 func 函数，作用于 inp._values，返回结果构造成多个 NestedTensor
    return tuple(
        NestedTensor(values=x, **extract_kwargs(inp))
        for x in func(inp._values, **new_kwargs)
    )
@register_jagged_func(
    torch.ops.aten.split_with_sizes.default, "self: jt, split_sizes: any, dim: any"
)
# 注册一个自定义的函数装饰器，将 torch.ops.aten.split_with_sizes.default 函数映射到该装饰的函数上
def split_with_sizes_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    
    # 从参数中提取 input，并移除 new_kwargs 中的 input 键
    inp = new_kwargs.pop("input")
    
    # 根据 inp 的维度和 new_kwargs 中的 dim 参数，调用 _wrap_jagged_dim 函数进行包装
    new_kwargs["dim"] = _wrap_jagged_dim(
        inp.dim(), new_kwargs["dim"], "split_with_sizes"
    )
    
    # 调用 func 函数，并将结果封装成 NestedTensor 对象的列表
    return [
        NestedTensor(values=x, **extract_kwargs(inp))
        for x in func(inp._values, **new_kwargs)
    ]


@register_jagged_func(torch.ops.aten.chunk.default, "self: jt, chunks: any, dim: any?")
# 注册一个自定义的函数装饰器，将 torch.ops.aten.chunk.default 函数映射到该装饰的函数上
def chunk_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    
    # 从参数中提取 input，并移除 new_kwargs 中的 input 键
    inp = new_kwargs.pop("input")
    
    # 根据 inp 的维度和 new_kwargs 中的 dim 参数，调用 _wrap_jagged_dim 函数进行包装
    new_kwargs["dim"] = _wrap_jagged_dim(
        inp.dim(), new_kwargs["dim"], "chunk", allow_batch_dim=True
    )
    
    # 如果 new_kwargs 中的 dim 参数为 0，则执行以下操作
    if new_kwargs["dim"] == 0:
        chunks = new_kwargs["chunks"]
        dim0_size = inp._size[0]
        chunk_size = math.ceil(dim0_size / chunks)
        
        # 计算 chunk 的长度，并获取各 chunk 的偏移量
        lengths = inp._offsets.diff()
        chunked_lengths = lengths.chunk(chunks)
        chunked_offsets = [torch.cumsum(x, dim=0) for x in chunked_lengths]
        chunked_offsets = [F.pad(x, (1, 0), value=0) for x in chunked_offsets]
        
        # 为每个 chunk 准备嵌套的参数字典
        nested_kwargs = [
            {"offsets": per_offsets, "_ragged_idx": inp._ragged_idx}
            for per_offsets in chunked_offsets
        ]
        
        # 获取每个 chunk 的值，并封装成 NestedTensor 对象的列表
        split_sizes = [x.sum().item() for x in chunked_lengths]
        chunk_values = inp._values.split(split_sizes)
        return [
            NestedTensor(values=chunk_values[i], **(nested_kwargs[i]))
            for i in range(0, chunk_size)
        ]
    else:
        # 如果 new_kwargs 中的 dim 参数不为 0，则直接调用 func 函数，并封装成 NestedTensor 对象的列表
        return [
            NestedTensor(values=x, **extract_kwargs(inp))
            for x in func(inp._values, **new_kwargs)
        ]


@register_jagged_func(torch.ops.aten.unbind.int, "self: jt_all, dim: any?")
# 注册一个自定义的函数装饰器，将 torch.ops.aten.unbind.int 函数映射到该装饰的函数上
def unbind_int(func, *args, **kwargs):
    # 注意这里专门处理 offsets 长度的情况
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    
    # 获取 new_kwargs 中的 dim 参数
    dim = new_kwargs["dim"]
    
    # 如果 dim 不等于 0，则抛出运行时错误
    if dim != 0:
        raise RuntimeError("unbind(): only supported for NestedTensor on dim=0")
    
    # 从参数中提取 input，并移除 new_kwargs 中的 input 键
    inp = new_kwargs.pop("input")
    
    # 获取 inp 的值、偏移量和长度
    values = inp.values()
    offsets = inp.offsets()
    lengths = inp.lengths()
    ragged_idx = inp._ragged_idx
    
    # 如果 lengths 为 None，则使用 offsets 的差分值进行分割，并根据 ragged_idx 执行分割操作
    if lengths is None:
        return torch.split(values, offsets.diff().tolist(), dim=(ragged_idx - 1))
    
    # 如果 ragged_idx 小于等于 0，则抛出运行时错误
    if ragged_idx <= 0:
        raise RuntimeError(
            "unbind(): nested tensor ragged_idx out of bounds (should be >= 1)"
        )
    # 遍历长度数组的每个元素
    for i in range(lengths.shape[0]):
        # 检查当前偏移量加上长度是否超过了值数组在 ragged_idx - 1 维度上的长度
        if offsets[i] + lengths[i] > values.shape[ragged_idx - 1]:
            # 如果超出范围，抛出运行时错误，指示嵌套张量的偏移量和长度与 ragged_idx 维度不匹配
            raise RuntimeError(
                "unbind(): nested tensor offsets and lengths do not match ragged_idx dimension"
            )
    # 返回一个列表，其中每个元素是通过在 values 张量的 ragged_idx - 1 维度上进行切片得到的结果
    return [
        torch.narrow(values, dim=(ragged_idx - 1), start=offsets[i], length=lengths[i])
        for i in range(lengths.shape[0])
    ]
# 注册一个装饰器函数，用于对 torch.ops.aten.squeeze.dim 操作进行注册
# "self: jt, dim: any" 表示函数的签名，其中 jt 是输入的第一个参数，dim 是任意类型的维度参数
@register_jagged_func(torch.ops.aten.squeeze.dim, "self: jt, dim: any")
def squeeze_dim(func, *args, **kwargs):
    # 根据函数和参数规范化关键字参数，确保仅使用规范化后的关键字参数
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # 从规范化后的关键字参数中取出输入
    inp = new_kwargs.pop("input")
    # 获取输入的值部分（假设为 NestedTensor 类的实例），保存在 values 中
    values = inp._values

    # 对于维度参数进行包装，处理可能的嵌套维度，用于 squeeze 操作
    new_kwargs["dim"] = _wrap_jagged_dim(len(inp._size), new_kwargs["dim"], "squeeze")
    
    # 调用底层的 aten.squeeze.dim 操作，返回一个 NestedTensor 对象，提取输入的其他关键字参数
    return NestedTensor(func(values, **new_kwargs), **extract_kwargs(inp))


# 注册一个装饰器函数，用于对 torch.ops.aten.unsqueeze.default 操作进行注册
# "self: jt, dim: any" 表示函数的签名，其中 jt 是输入的第一个参数，dim 是任意类型的维度参数
@register_jagged_func(torch.ops.aten.unsqueeze.default, "self: jt, dim: any")
def unsqueeze_default(func, *args, **kwargs):
    # 根据函数和参数规范化关键字参数，确保仅使用规范化后的关键字参数
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # 从规范化后的关键字参数中取出输入
    inp = new_kwargs.pop("input")
    # 获取输入的值部分（假设为 NestedTensor 类的实例），保存在 values 中
    values = inp._values

    # 负责处理嵌套维度的展开，用于 unsqueeze 操作
    dim = new_kwargs["dim"]
    new_kwargs["dim"] = _wrap_jagged_dim(len(inp._size) + 1, dim, "unsqueeze")
    
    # 调用底层的 aten.unsqueeze.default 操作，返回一个 NestedTensor 对象，提取输入的其他关键字参数
    return NestedTensor(func(values, **new_kwargs), **extract_kwargs(inp))


# 注册一个装饰器函数，用于对 torch.ops.aten.cat.default 操作进行注册
# "tensors: any, dim: any" 表示函数的签名，其中 tensors 是任意类型的张量参数，dim 是任意类型的维度参数
@register_jagged_func(torch.ops.aten.cat.default, "tensors: any, dim: any")
def cat_default(func, *args, **kwargs):
    # 根据函数和参数规范化关键字参数，确保仅使用规范化后的关键字参数
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # 从规范化后的关键字参数中取出 tensors 参数
    tensors = new_kwargs.pop("tensors")

    # 将非嵌套张量转换为嵌套张量
    nested = [t for t in tensors if t.is_nested]
    assert len(nested) > 0
    first = nested[0]
    tensors = [t if t.is_nested else t.expand_as(first) for t in tensors]

    # 负责处理嵌套维度的合并，用于 cat 操作
    dim = new_kwargs["dim"]
    new_kwargs["dim"] = _wrap_jagged_dim(len(first.shape), dim, "cat")

    # 调用底层的 aten.cat.default 操作，返回一个 NestedTensor 对象，提取第一个张量的其他关键字参数
    return NestedTensor(
        func([t._values for t in tensors], **new_kwargs), **extract_kwargs(tensors[0])
    )


# 注册一个装饰器函数，用于对 torch.ops.aten.matmul.default 操作进行注册
# "self: jt, other: any" 表示函数的签名，其中 jt 是输入的第一个参数，other 是任意类型的张量参数
@register_jagged_func(torch.ops.aten.matmul.default, "self: jt, other: any")
def matmul_default(func, *args, **kwargs):
    # 根据函数和参数规范化关键字参数，确保仅使用规范化后的关键字参数
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # 从规范化后的关键字参数中取出输入和其他参数
    inp = new_kwargs.pop("input")
    other = new_kwargs.pop("other")

    # 处理嵌套输入情况下的矩阵乘法操作
    if inp.is_nested and not other.is_nested:
        return NestedTensor(
            func(inp._values, other, **new_kwargs), **extract_kwargs(inp)
        )
    elif inp.is_nested and other.is_nested:
        # 处理具有相同不规则维度的两个输入的批量矩阵乘法（BMM）
        if inp.dim() > 3 and other.dim() > 3 and raggedness_matches(inp, other._size):
            return NestedTensor(func(inp._values, other._values), **extract_kwargs(inp))

    # 若无法处理的情况，抛出运行时错误
    raise RuntimeError(
        f"matmul(): not supported between inputs of shapes {inp._size} and {other.shape}"
    )


# 注册一个装饰器函数，用于对 torch.ops.aten.expand.default 操作进行注册
# "self: jt, size: any, implicit: any?" 表示函数的签名，其中 jt 是输入的第一个参数，size 是任意类型的大小参数，implicit 是可选的隐式参数
@register_jagged_func(
    torch.ops.aten.expand.default, "self: jt, size: any, implicit: any?"
)
def expand_default(func, *args, **kwargs):
    # 根据函数和参数规范化关键字参数，确保仅使用规范化后的关键字参数
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # 从规范化后的关键字参数中取出输入和 size 参数
    inp = new_kwargs.pop("input")
    size = new_kwargs["size"]
    # 检查是否在 new_kwargs 中包含 "implicit" 键，并且其对应的值为 False，如果是则移除该键值对
    assert ("implicit" not in new_kwargs) or (not new_kwargs.pop("implicit"))
    # 检查输入的形状是否符合要求，如果不符合则抛出异常
    if not raggedness_matches(inp, size):
        raise RuntimeError(f"expand(): cannot expand shape {inp._size} -> {size}")

    # 创建一个用于扩展的参数列表，将 -1 添加到列表开头，其余元素从 size 的第三个元素开始复制
    expand_arg = [-1, *size[2:]]
    # 使用 func 函数对 inp._values 进行处理，使用 expand_arg 进行扩展，同时将输入 inp 的其他关键字参数提取并作为关键字参数传递
    return NestedTensor(func(inp._values, expand_arg), **extract_kwargs(inp))
@register_jagged_func(torch.ops.aten.expand_as.default, "self: t, other: jt")
def expand_as_default(func, *args, **kwargs):
    # 标准化函数参数，仅使用关键字参数进行标准化
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # 从标准化后的参数中取出输入张量和其他张量
    inp = new_kwargs.pop("input")
    other = new_kwargs.pop("other")

    # 调用底层函数，对输入张量进行形状扩展操作
    return NestedTensor(func(inp, other._values), **extract_kwargs(other))


@register_jagged_func(torch.ops.aten.where.self, "condition: jt, self: jt, other: jt")
def where_self(func, *args, **kwargs):
    # 标准化函数参数，仅使用关键字参数进行标准化
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # 从标准化后的参数中取出条件张量、输入张量和其他张量
    condition = new_kwargs.pop("condition")
    inp = new_kwargs.pop("input")
    other = new_kwargs.pop("other")

    # 断言条件张量、其他张量和输入张量的尺寸相同
    assert condition._size == other._size == inp._size

    # 调用底层函数，在满足条件的位置选择输入张量或其他张量的值
    return NestedTensor(
        func(condition._values, inp._values, other._values, **new_kwargs),
        **extract_kwargs(condition),
    )


@register_jagged_func(torch.ops.aten._pin_memory.default, "self: jt, device: any?")
def _pin_memory_default(func, *args, **kwargs):
    # 标准化函数参数，仅使用关键字参数进行标准化
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # 从标准化后的参数中取出输入张量
    inp = new_kwargs.pop("input")

    # 调用底层函数，将输入张量的数据存储到固定内存中
    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(torch.ops.aten.is_pinned.default, "self: jt, device: any?")
def is_pinned_default(func, *args, **kwargs):
    # 标准化函数参数，仅使用关键字参数进行标准化
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # 从标准化后的参数中取出输入张量
    inp = new_kwargs.pop("input")

    # 调用底层函数，判断输入张量的数据是否存储在固定内存中
    return func(inp._values, **new_kwargs)


@register_jagged_func(
    torch.ops.aten.is_same_size.default, "self: jt_all, other: jt_all"
)
def is_same_size_default(func, *args, **kwargs):
    # 直接返回两个输入张量是否尺寸相同的布尔值
    return args[0]._size == args[1]._size


@register_jagged_func(
    torch.ops.aten.sum.dim_IntList, "self: jt, dim: any?, keepdim: any?, dtype: any?"
)
def sum_dim_IntList(func, *args, **kwargs):
    # sum_dim_IntList 函数根据是否减少不规则维度返回 NestedTensor 或 Tensor
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    # 从标准化后的参数中取出输入张量
    inp = new_kwargs.pop("input")
    # 断言输入张量是不规则的
    assert inp._ragged_idx == 1
    # 包装不规则维度并更新关键字参数中的维度信息
    new_kwargs["dim"], ragged_reduced_away = _wrap_jagged_dims(
        inp.dim(), new_kwargs["dim"], "sum"
    )

    if not ragged_reduced_away:
        # 如果未减少不规则维度，则返回 NestedTensor
        return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))
    else:
        # 如果减少了不规则维度，则不包装结果，可能需要保持维度
        out = func(inp._values, **new_kwargs)
        if new_kwargs["keepdim"]:
            out = out.unsqueeze(0)
        return out


@register_jagged_func(
    torch.ops.aten.transpose.int, "self: jt_all, dim0: any, dim1: any"
)
def transpose_int(func, *args, **kwargs):
    # 标准化函数参数，仅使用关键字参数进行标准化
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # 直接调用底层函数执行张量转置操作
    # 导入torch._prims_common模块中的canonicalize_dims函数
    from torch._prims_common import canonicalize_dims

    # 从new_kwargs字典中弹出键为"input"的值，赋给inp变量
    inp = new_kwargs.pop("input")
    
    # 使用canonicalize_dims函数对inp的维度进行规范化，得到规范后的dim0和dim1
    dim0, dim1 = canonicalize_dims(inp.dim(), (new_kwargs["dim0"], new_kwargs["dim1"]))

    # 如果inp的_lengths属性不为None，抛出ValueError异常
    if inp._lengths is not None:
        raise ValueError(
            "transpose(): not supported on jagged layout nested tensor with holes"
        )

    # 为了支持SDPA API，需要将输入的ragged idx（不规则索引）转置到dim 2而不是dim 1，
    # 尽管内部的Flash和内存高效实现会使用在dim 1 上有raggedness的输入。
    if dim0 == inp._ragged_idx or dim1 == inp._ragged_idx:
        # 如果dim0或dim1等于inp的ragged idx
        if dim0 == 0 or dim1 == 0:
            # 如果dim0等于0或dim1等于0，抛出异常
            raise ValueError(
                "Transpose is not supported on the batch dimension for jagged NT"
            )
        # 如果dim0等于inp的ragged idx，则将to_dim设置为dim1，否则设置为dim0
        if dim0 == inp._ragged_idx:
            to_dim = dim1
        else:
            to_dim = dim0
        
        # 提取inp的关键字参数
        inp_kwargs = extract_kwargs(inp)
        # 设置inp_kwargs中的"_ragged_idx"键对应的值为to_dim
        inp_kwargs["_ragged_idx"] = to_dim
        
        # 返回一个NestedTensor对象，其值为inp的值经过transpose操作后的结果，
        # 使用_outer_to_inner_dim函数将dim0和dim1从外部维度转换为内部维度
        return NestedTensor(
            inp.values().transpose(
                _outer_to_inner_dim(len(inp._size), dim0),
                _outer_to_inner_dim(len(inp._size), dim1),
            ),
            **inp_kwargs,
        )

    # 更新new_kwargs字典中"dim0"和"dim1"的值，通过_wrap_jagged_dim函数对inp的维度进行包装
    new_kwargs["dim0"] = _wrap_jagged_dim(inp.dim(), new_kwargs["dim0"], "transpose")
    new_kwargs["dim1"] = _wrap_jagged_dim(inp.dim(), new_kwargs["dim1"], "transpose")

    # 返回一个NestedTensor对象，其值为调用func函数对inp._values和new_kwargs进行处理后的结果，
    # 使用extract_kwargs函数提取inp的关键字参数
    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))
@register_jagged_func(
    [torch.ops.aten.view.default, torch.ops.aten._unsafe_view.default],
    "self: jt_all, size: any",
)
# 定义装饰器函数，用于注册视图操作函数，支持两种视图操作：默认安全视图和不安全视图
def view_default(func, *args, **kwargs):
    # 根据函数和参数规范化函数调用
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # 从规范化后的关键字参数中取出输入张量和目标大小
    inp = new_kwargs.pop("input")
    size = new_kwargs.pop("size")

    # 检查输入张量的不规则索引和大小是否符合条件
    if inp._ragged_idx != 1 and tuple(inp._size) != tuple(size):
        raise RuntimeError(
            f"view(): does not support ragged_idx != 1 except when inp._size == size. "
            f"inp._size is ({inp._size}) and size is ({size})."
        )

    # 确保指定的大小仍然包含批次和不规则维度
    if len(size) < 3 or not raggedness_matches(inp, size):
        raise RuntimeError(f"view(): cannot view shape {inp._size} as {size}")

    # 外部大小：NT 的大小，例如 [3, j0, 10]
    # 内部大小：值的大小，例如 [8, 10]（例如偏移为 [0, 3, 5, 8]）
    # 此函数获取给定内部索引的内部大小。
    #
    # 示例：对于外部大小 [a, b, c, j0, d, e, f]
    # 假设 j0 是不规则的，其他是具体的整数
    # 并且 ragged_idx=3
    # 内部大小将是 [b, c, inp._values.size(ragged_idx), d, e, f]
    # 因此：
    #    inner_size[0] = outer_size[1]
    #    inner_size[1] = outer_size[2]
    #    inner_size[0] = inp._values.size(ragged_idx - 1)
    #    inner_size[3] = outer_size[4]
    #    inner_size[4] = outer_size[5]
    def get_inner_size(inner_idx):
        nonlocal inp, size
        if inner_idx == inp._ragged_idx - 1:
            return inp._values.size(inner_idx)
        else:
            return size[inner_idx + 1]

    # 根据目标大小获取内部大小列表
    inner_size = [get_inner_size(i) for i in range(len(size) - 1)]

    # 调用函数对输入张量的内部值进行视图操作，并将额外参数提取为关键字参数传递
    return NestedTensor(func(inp._values, inner_size), **extract_kwargs(inp))


@register_jagged_func(
    torch.ops.aten.native_layer_norm.default,
    "input: jt, normalized_shape: any, weight: any?, bias: any?, eps: any",
)
# 注册原生层归一化默认函数
def native_layer_norm_default(func, *args, **kwargs):
    # 根据函数和参数规范化函数调用
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # 从规范化后的关键字参数中取出输入张量和归一化形状
    inp = new_kwargs.pop("input")
    normalized_shape = new_kwargs["normalized_shape"]

    # 确保不是在不规则维度上进行归一化
    if inp.dim() < 3 or (inp.dim() - len(normalized_shape)) < 2:
        raise RuntimeError(
            "layer_norm(): normalizing over ragged dim not supported for nested tensors"
        )

    # 调用函数对输入张量的值进行原生层归一化，并返回输出、均值和标准差
    output, mean, std = func(inp._values, **new_kwargs)
    return (NestedTensor(output, **extract_kwargs(inp)), mean, std)


@register_jagged_func(
    torch.ops.aten.native_layer_norm_backward.default,
    "grad_out: jt, input: jt, normalized_shape: any, mean: any, rstd: any, weight: any?, bias: any?, output_mask: any",
)
# 注册原生层归一化反向传播默认函数
def native_layer_norm_backward_default(func, *args, **kwargs):
    # 调用 normalize_function 函数，将输入参数 func、args、kwargs 标准化为仅使用 kwargs 的形式，并获取返回结果中的新 kwargs
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    # 从 new_kwargs 中弹出键为 "grad_out" 的值，该值将作为函数 func 的梯度输出参数
    grad_out = new_kwargs.pop("grad_out")
    # 从 new_kwargs 中弹出键为 "input" 的值，该值将作为函数 func 的输入参数
    inp = new_kwargs.pop("input")
    # 调用函数 func，传入 grad_out 和 inp 的数值部分以及剩余的 new_kwargs 作为关键字参数，获取结果 d_input、d_gamma 和 d_beta
    d_input, d_gamma, d_beta = func(grad_out._values, inp._values, **new_kwargs)
    # 如果 d_input 为 None，则返回一个元组 (None, d_gamma, d_beta)
    if d_input is None:
        return (None, d_gamma, d_beta)

    # 否则，返回一个元组，其中包含使用 d_input 创建的 NestedTensor 对象和从 inp 中提取的关键字参数作为初始化参数，以及 d_gamma 和 d_beta
    return (NestedTensor(d_input, **extract_kwargs(inp)), d_gamma, d_beta)
@register_jagged_func(torch.ops.aten.select.int, "self: jt, dim: any, index: any")
def select_int(func, *args, **kwargs):
    # 根据函数签名规范化函数参数
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # 从参数中提取输入张量
    inp = new_kwargs.pop("input")
    # 规范化维度参数并封装，用于处理不规则张量的选择操作
    new_kwargs["dim"] = _wrap_jagged_dim(inp.dim(), new_kwargs["dim"], "select")

    # 调用底层操作函数执行选择操作，并创建新的 NestedTensor 对象
    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(
    torch.ops.aten.slice.Tensor,
    "self: jt, dim: any?, start: any?, end: any?, step: any?",
)
def slice_tensor(func, *args, **kwargs):
    # 根据函数签名规范化函数参数
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # 从参数中提取输入张量
    inp = new_kwargs.pop("input")
    # 规范化维度参数并封装，用于处理不规则张量的切片操作
    new_kwargs["dim"] = _wrap_jagged_dim(inp.dim(), new_kwargs["dim"], "slice")

    # 调用底层操作函数执行切片操作，并创建新的 NestedTensor 对象
    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(
    torch.ops.aten.convolution.default,
    "input: jt, weight: t, bias: t?, stride: any, padding: any, "
    "dilation: any, transposed: any, output_padding: any, groups: any",
)
def convolution_default(func, *args, **kwargs):
    # 根据函数签名规范化函数参数
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # 从参数中提取输入张量
    inp = new_kwargs.pop("input")

    # 调用底层卷积函数执行卷积操作，并创建新的 NestedTensor 对象
    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(
    torch.ops.aten.mean.dim, "self: jt, dim: any?, keepdim: any, dtype: any?"
)
def mean_dim(func, *args, **kwargs):
    # 根据函数签名规范化函数参数
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # 从参数中提取输入张量
    inp = new_kwargs.pop("input")
    # 对维度参数进行特殊处理，以符合 mean 函数的预期输入格式
    new_kwargs["dim"] = [_wrap_jagged_dim(inp.dim(), new_kwargs["dim"][0], "mean")]

    # 调用底层求均值函数执行操作，并创建新的 NestedTensor 对象
    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(torch.ops.aten.stack.default, "tensors: any, dim: any")
def stack_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # 保证 tensors 非空
    tensors = new_kwargs.pop("tensors")
    for t in tensors:
        if not isinstance(t, NestedTensor):
            raise RuntimeError("stack(): expected all nested tensors inputs")

        if t.dim() != tensors[0].dim():
            raise RuntimeError(
                "stack(): expected all nested tensors to have the same dim"
            )

        if not raggedness_matches(t, tensors[0].shape):
            raise RuntimeError(
                "stack(): expected all nested tensors to have the same nested structure"
            )

    # 规范化维度参数并封装，用于处理不规则张量的堆叠操作
    new_kwargs["dim"] = _wrap_jagged_dim(
        tensors[0].dim() + 1, new_kwargs["dim"], "stack"
    )

    # 调用底层堆叠函数执行操作，并创建新的 NestedTensor 对象
    return NestedTensor(
        func([t._values for t in tensors], **new_kwargs), **extract_kwargs(tensors[0])
)
    # 导入必要的模块
    import os
    import zipfile
    
    # 定义函数 unzip_file，接受两个参数：压缩文件路径（source_file）和目标路径（dest_dir）
    def unzip_file(source_file, dest_dir):
        # 确保目标路径存在，若不存在则创建
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        
        # 打开压缩文件
        with zipfile.ZipFile(source_file, 'r') as zip_ref:
            # 解压缩文件到目标路径
            zip_ref.extractall(dest_dir)
    
    # 调用 unzip_file 函数，解压缩 example.zip 文件到当前目录下的 /extracted 文件夹中
    unzip_file('example.zip', './extracted')
@register_jagged_func(
    torch.ops.aten.embedding.default,
    "weight: t, indices: jt, padding_idx: any?, scale_grad_by_freq: any?, sparse: any?",
)
def embedding_default(func, *args, **kwargs):
    # 根据函数签名规范化参数，仅使用关键字参数
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # 从规范化后的参数中弹出 "indices" 和 "weight"
    indices = new_kwargs.pop("indices")
    weight = new_kwargs.pop("weight")

    # 调用嵌套张量构造函数，并返回结果
    return NestedTensor(
        func(weight, indices._values, **new_kwargs), **extract_kwargs(indices)
    )


@register_jagged_func(
    [
        torch.ops.aten.values.default,
        torch.ops.aten._nested_get_values.default,
    ],
    "self: jt_all",
)
def values_default(func, *args, **kwargs):
    # 根据函数签名规范化参数，仅使用关键字参数
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # 从规范化后的参数中获取 "input"
    inp = new_kwargs.pop("input")

    # TODO: 处理推断模式的情况。
    # 参见 https://github.com/pytorch/pytorch/issues/112024#issuecomment-1779554292
    # 返回输入张量的 _values 属性的分离视图
    return inp._values.detach()


@register_jagged_func(
    torch.ops.aten._nested_view_from_jagged.default,
    "values: t, offsets: t, dummy: jt_all, lengths: t?, ragged_idx: any?, min_seqlen: t?, max_seqlen: t?",
)
def _nested_view_from_jagged_default(func, *args, **kwargs):
    # 根据函数签名规范化参数，仅使用关键字参数
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # 从规范化后的参数中获取 "values", "offsets", "lengths"
    values, offsets, lengths = (
        new_kwargs["input"],
        new_kwargs["offsets"],
        new_kwargs["lengths"],
    )
    ragged_idx = new_kwargs["ragged_idx"]
    min_seqlen = new_kwargs["min_seqlen"]
    max_seqlen = new_kwargs["max_seqlen"]
    metadata_cache = {}

    # 如果指定了 min_seqlen，则将其存储到 metadata_cache
    if min_seqlen is not None:
        metadata_cache["min_seqlen"] = min_seqlen

    # 如果指定了 max_seqlen，则将其存储到 metadata_cache
    if max_seqlen is not None:
        metadata_cache["max_seqlen"] = max_seqlen

    # 返回嵌套张量对象，包括 values, offsets, lengths 和 metadata_cache
    return NestedTensor(
        values,
        offsets,
        lengths=lengths,
        _ragged_idx=ragged_idx,
        _metadata_cache=metadata_cache,
    )


@register_jagged_func(torch.ops.aten._nested_get_offsets.default, "self: jt_all")
def _nested_get_offsets(func, *args, **kwargs):
    # 根据函数签名规范化参数，仅使用关键字参数
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # 从规范化后的参数中获取 "input"
    inp = new_kwargs.pop("input")

    # 返回输入张量的 _offsets 属性
    return inp._offsets


@register_jagged_func(torch.ops.aten._nested_get_lengths.default, "self: jt_all")
def _nested_get_lengths(func, *args, **kwargs):
    # 根据函数签名规范化参数，仅使用关键字参数
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # 从规范化后的参数中获取 "input"
    inp = new_kwargs.pop("input")

    # 返回输入张量的 _lengths 属性
    return inp._lengths


@register_jagged_func(torch.ops.aten._nested_get_ragged_idx.default, "self: jt_all")
def _nested_get_ragged_idx(func, *args, **kwargs):
    # 根据函数签名规范化参数，仅使用关键字参数
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # 从规范化后的参数中获取 "input"
    inp = new_kwargs.pop("input")

    # 返回输入张量的 _ragged_idx 属性
    return inp._ragged_idx
# 将 `_nested_get_min_seqlen` 函数注册为 `torch.ops.aten._nested_get_min_seqlen.default` 的 jagged 函数，使用 `jt_all` 作为注册的条件
@register_jagged_func(torch.ops.aten._nested_get_min_seqlen.default, "self: jt_all")
def _nested_get_min_seqlen(func, *args, **kwargs):
    # 根据函数签名规范化参数和关键字参数，仅使用关键字参数进行规范化
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # 从规范化后的参数中弹出名为 "input" 的参数
    inp = new_kwargs.pop("input")
    # 返回输入张量的元数据缓存中的最小序列长度，若不存在则返回 None
    return inp._metadata_cache.get("min_seqlen", None)


# 将 `_nested_get_max_seqlen` 函数注册为 `torch.ops.aten._nested_get_max_seqlen.default` 的 jagged 函数，使用 `jt_all` 作为注册的条件
@register_jagged_func(torch.ops.aten._nested_get_max_seqlen.default, "self: jt_all")
def _nested_get_max_seqlen(func, *args, **kwargs):
    # 根据函数签名规范化参数和关键字参数，仅使用关键字参数进行规范化
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # 从规范化后的参数中弹出名为 "input" 的参数
    inp = new_kwargs.pop("input")
    # 返回输入张量的元数据缓存中的最大序列长度，若不存在则返回 None
    return inp._metadata_cache.get("max_seqlen", None)


# 将 `_nested_get_jagged_dummy` 函数注册为 `torch.ops.aten._nested_get_jagged_dummy.default` 的 jagged 函数，使用 `any` 作为注册的条件
@register_jagged_func(torch.ops.aten._nested_get_jagged_dummy.default, "self: any")
def _nested_get_jagged_dummy(func, *args, **kwargs):
    # 导入 `_nt_view_dummy` 函数，用于创建虚拟的嵌套张量视图
    from torch.nested._internal.nested_tensor import _nt_view_dummy

    # 返回调用 `_nt_view_dummy` 函数的结果，该函数创建虚拟的嵌套张量视图
    return _nt_view_dummy()


# 使用 `torch.library._scoped_library` 上下文管理器，将 `_nested_get_jagged_dummy` 函数注册到不同设备类型上
with torch.library._scoped_library("aten", "IMPL") as aten:
    # 在 CPU 设备上注册 `_nested_get_jagged_dummy` 函数的实现
    aten.impl("_nested_get_jagged_dummy", _nested_get_jagged_dummy, "CPU")
    # 在 CUDA 设备上注册 `_nested_get_jagged_dummy` 函数的实现
    aten.impl("_nested_get_jagged_dummy", _nested_get_jagged_dummy, "CUDA")
    # 在 Meta 设备上注册 `_nested_get_jagged_dummy` 函数的实现
    aten.impl("_nested_get_jagged_dummy", _nested_get_jagged_dummy, "Meta")
```