# `.\pytorch\torch\_higher_order_ops\out_dtype.py`

```
# 引入 torch 库和相关子模块
import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.utils import autograd_not_implemented
from torch._ops import HigherOrderOperator
from torch._prims_common import elementwise_dtypes, ELEMENTWISE_TYPE_PROMOTION_KIND
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    maybe_handle_decomp,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)

# 允许未定义类型的函数声明（mypy）
# TODO: 需要找到更通用的方法来处理
ALLOWABLE_OPS = [
    torch.ops.aten.linear.default,
    torch.ops.aten.mm.default,
    torch.ops.aten.conv2d.default,
    torch.ops.aten.convolution.default,
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.mul.Scalar,
    torch.ops.aten.div.Tensor,
    torch.ops.aten.div.Scalar,
]

class OutDtypeOperator(HigherOrderOperator):
    """
    The out_dtype operator takes an existing ATen functional operator, an
    `out_dtype` argument, and arguments to the original operator, and executes
    the original operator and returns a Tensor with the `out_dtype` precision.
    This operator does not mandate a compute precision so it allows the
    representation to not be opinionated about the exact implementation.

    The general implementation for all operators will be the following:
        1. Promote inputs dtypes based on default PyTorch dtype promotion rules,
            using the dtypes of all input Tensors/Scalars and the `out_dtype`
            arugument.
        2. Execute the operator
        3. Cast the output to `out_dtype`
    """

    def __init__(self):
        super().__init__("out_dtype")
        # TODO(ydwu4): Subclassing HigherOrderOperator causes __module__ to
        # become different (torch._higher_order_ops.out_dtype) which will result
        # in torch.fx to record the op incorrectly in the graph.
        # 设置模块名以便在图中正确记录操作
        self.__module__ = "torch.ops.higher_order"

    def __call__(self, op, output_dtype, *args):
        # 检查第一个参数是否为 OpOverload 类型
        if not isinstance(op, torch._ops.OpOverload):
            raise ValueError("out_dtype's first argument must be an OpOverload")
        # 检查操作是否为不可变的函数操作
        if op._schema.is_mutable:
            raise ValueError(
                "out_dtype's first argument needs to be a functional operator"
            )
        # 检查操作是否仅返回单个张量类型
        if not (
            len(op._schema.returns) == 1
            and isinstance(op._schema.returns[0].type, torch.TensorType)
        ):
            raise ValueError(
                "out_dtype's can only apply to ops that return a single tensor"
                f"Instead got {[r.type for r in op._schema.returns]}"
            )

        # 检查操作是否在允许的操作列表中
        if op not in ALLOWABLE_OPS:
            raise ValueError(
                f"out_dtype only allows the following operators: {ALLOWABLE_OPS}."
            )

        # 调用父类方法执行操作
        res = super().__call__(op, output_dtype, *args)

        return res


# 创建 OutDtypeOperator 类的实例
out_dtype = OutDtypeOperator()


def trace_out_dtype(proxy_mode, func_overload, op, output_dtype, *args):
    # NB: Long-term we should put the decomposition logic into
    # ProxyTorchDispatchMode so that people do not need to call maybe_handle_decomp
    # in all HigherOrderOp proxy implementations.
    # 长期来看，我们应该将分解逻辑放入 ProxyTorchDispatchMode 中，
    # 这样人们就不需要在所有 HigherOrderOp 代理实现中调用 maybe_handle_decomp 了。

    r = maybe_handle_decomp(proxy_mode, func_overload, (op, output_dtype, *args), {})
    # 调用 maybe_handle_decomp 函数处理代理模式、函数重载和参数，以尝试处理分解操作
    if r is not NotImplemented:
        # 如果返回值 r 不是 NotImplemented，则直接返回 r
        return r

    with disable_proxy_modes_tracing():
        # 这是一个简化的操作符实现，专门用于追踪。
        # 实际实现可能首先对参数进行提升（promotion）
        # 使用 disable_proxy_modes_tracing() 禁用代理模式的追踪功能
        out = op(*args).to(dtype=output_dtype)
        # 执行操作 op，并将结果转换为指定的数据类型 output_dtype

    node_args = (op, output_dtype, *args)
    # 组成节点参数元组
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)
    # 使用 proxy_mode.tracer.unwrap_proxy 映射解包代理对象
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}, name="out_dtype"
    )
    # 使用代理模式的追踪器创建一个代理对象，用于追踪函数调用，并命名为 "out_dtype"
    
    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)
    # 返回追踪过的张量树，使用指定的追踪器追踪
# 使用装饰器声明函数，指定实现的调度键为CompositeExplicitAutograd
@out_dtype.py_impl(DispatchKey.CompositeExplicitAutograd)
def out_dtype_dense(op: torch._ops.OpOverload, output_dtype: torch.dtype, *args):
    # 检查是否可以使用整数矩阵乘法实现
    if is_int_mm(op, output_dtype, args):
        # 调用整数矩阵乘法的特定实现
        return torch._int_mm(*args)
    # 否则，使用通用的类型推断回退函数
    return out_dtype_fallback(op, output_dtype, *args)


# 判断是否可以使用整数矩阵乘法实现
def is_int_mm(op, output_dtype, args):
    return (
        op == torch.ops.aten.mm.default
        and output_dtype == torch.int32
        and len(args) == 2
        and args[0].dtype == torch.int8
        and args[1].dtype == torch.int8
        and args[0].is_cuda
        and args[1].is_cuda
    )


# 通用的类型推断回退函数
def out_dtype_fallback(op, output_dtype, *args):
    # 将所有输入展平，并加入一个dtype为output_dtype的张量
    flat_inputs = pytree.arg_tree_leaves(*args) + [torch.ones(1, dtype=output_dtype)]
    # 推断出要提升的数据类型
    promote_dtype: torch.dtype = elementwise_dtypes(
        *flat_inputs,
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    )[0]

    # 将所有输入转换为推断出的数据类型
    casted_args = pytree.tree_map_only(
        torch.Tensor, lambda arg: arg.to(dtype=promote_dtype), args
    )
    # 执行操作，并将结果转换为指定的输出数据类型
    res = op(*casted_args).to(dtype=output_dtype)
    return res


# 使用装饰器声明函数，指定实现的调度键为Autograd
out_dtype.py_impl(DispatchKey.Autograd)(
    autograd_not_implemented(out_dtype, deferred_error=True)
)


# 使用装饰器声明函数，指定实现的调度键为ProxyTorchDispatchMode
@out_dtype.py_impl(ProxyTorchDispatchMode)
def out_dtype_proxy(
    mode: ProxyTorchDispatchMode,
    op: torch._ops.OpOverload,
    output_dtype: torch.dtype,
    *args,
):
    # 如果启用追踪模式，则调用追踪输出数据类型函数
    if mode.enable_tracing:
        return trace_out_dtype(mode, out_dtype, op, output_dtype, *args)
    else:
        # 否则，调用默认的输出数据类型函数
        return out_dtype(op, output_dtype, *args)


# 使用装饰器声明函数，指定实现的调度键为FakeTensorMode
@out_dtype.py_impl(FakeTensorMode)
def out_dtype_fake_tensor_mode(
    mode: FakeTensorMode,
    op: torch._ops.OpOverload,
    output_dtype: torch.dtype,
    *args,
):
    # 使用伪张量模式执行函数体
    with mode:
        return out_dtype_dense(op, output_dtype, *args)


# 使用装饰器声明函数，指定为功能化实现的调度键
@out_dtype.py_functionalize_impl
def out_dtype_func(ctx, op, output_dtype, *args):
    # 对所有参数进行解包
    unwrapped_args = tuple(ctx.unwrap_tensors(arg) for arg in args)

    # 重新分派到下一个实现函数
    with ctx.redispatch_to_next():
        # 调用原始的输出数据类型函数，并包装结果
        res = out_dtype(op, output_dtype, *unwrapped_args)
    return ctx.wrap_tensors(res)
```