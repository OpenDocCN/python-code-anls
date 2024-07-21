# `.\pytorch\torch\_prims_common\wrappers.py`

```py
# 设置 mypy 的选项，允许未标注的函数定义
mypy: allow-untyped-defs

# 导入必要的模块和函数
import inspect              # 提供对 Python 对象内省的支持
import warnings             # 提供警告管理的功能
from functools import wraps  # 提供创建装饰器的功能
from itertools import chain  # 提供迭代器和生成器操作的工具函数

from typing import (        # 提供静态类型检查支持
    Callable,               # 可调用对象的类型
    NamedTuple,             # 命名元组的类型
    Optional,               # 可选类型的支持
    overload,               # 函数重载的装饰器
    Sequence,               # 序列类型的支持
    Tuple                   # 元组类型的支持
)

import torch                # PyTorch 张量库
import torch._prims_common as utils  # PyTorch 内部通用函数
from torch._prims_common import (    # 导入 PyTorch 内部通用类型和常量
    CustomOutParamAnnotation,        # 自定义输出参数注解
    ELEMENTWISE_TYPE_PROMOTION_KIND, # 元素级类型提升种类
    Number,                         # 数字类型
    NumberType,                     # 数字类型
    ShapeType,                      # 形状类型
    TensorLike,                     # 类张量类型
    TensorLikeType                  # 类张量类型
)
from torch.utils import _pytree as pytree  # PyTorch 工具模块
from torch.utils._pytree import (          # 导入 PyTorch 树形操作函数
    tree_flatten,                          # 树形展平函数
    tree_unflatten                         # 树形还原函数
)


@overload
# 定义函数重载，用于将输入转换为指定数据类型的张量类似对象
def _maybe_convert_to_dtype(a: TensorLikeType, dtype: torch.dtype) -> TensorLikeType:
    pass


@overload
# 定义函数重载，用于将输入转换为指定数据类型的数值类似对象
def _maybe_convert_to_dtype(a: NumberType, dtype: torch.dtype) -> NumberType:
    pass


@overload
# 定义函数重载，用于将输入序列转换为指定数据类型的序列对象
def _maybe_convert_to_dtype(a: Sequence, dtype: torch.dtype) -> Sequence:
    pass


@overload
# 定义函数重载，用于将空值保持为空值
def _maybe_convert_to_dtype(a: None, dtype: torch.dtype) -> None:
    pass


# TODO: 实现带有强制安全类型转换选项的 ref.cast
def _maybe_convert_to_dtype(a, dtype):
    # 如果输入对象类似张量
    if isinstance(a, TensorLike):
        # 如果输入对象的数据类型不是指定的数据类型
        if a.dtype != dtype:
            return a.to(dtype)  # 转换为指定数据类型
        return a  # 返回原对象
    # 如果输入对象是数字类型
    if isinstance(a, Number):
        return utils.dtype_to_type_ctor(dtype)(a)  # 使用指定数据类型构造器转换成类型
    # 如果输入对象是序列类型
    if isinstance(a, Sequence):
        return tuple(_maybe_convert_to_dtype(x, dtype) for x in a)  # 递归转换序列中每个元素的数据类型
    # 对于空值直接返回空值，因为某些使用类型提升包装的函数可能具有可选参数
    if a is None:
        return None

    # 如果输入对象不是张量或数字，则抛出数值错误异常
    raise ValueError(f"Received type {type(a)} that is neither a tensor or a number!")


# 将输入对象转换为指定类型的数值类型
def _maybe_convert_to_type(a: NumberType, typ: type) -> NumberType:
    # 如果输入对象不是数字类型，则抛出数值错误异常
    if not isinstance(a, Number):
        msg = f"Found unknown type {type(a)} when trying to convert scalars!"
        raise ValueError(msg)
    # 如果输入对象的类型不符合类型安全转换的要求，则抛出数值错误异常
    if not utils.is_weakly_lesser_type(type(a), typ):
        msg = f"Scalar {a} of type {type(a)} cannot be safely cast to type {typ}!"
        raise ValueError(msg)

    return typ(a)


# 检查注解是否具有指定类型
def _annotation_has_type(*, typ, annotation):
    # 如果注解具有多个参数
    if hasattr(annotation, "__args__"):
        # 遍历每个参数
        for a in annotation.__args__:
            # 如果有参数具有指定类型，则返回真
            if _annotation_has_type(typ=typ, annotation=a):
                return True
        return False

    return typ is annotation


# 元素级类型提升装饰器类
class elementwise_type_promotion_wrapper:
    """
    给 Python 引用实现添加元素级类型提升功能。

    接受两个关键字参数，type_promoting_args 和 type_promotion_kind。

    type_promoting_args 必须是一个字符串序列，指定参与类型提升的所有参数名称（应进行类型提升）。
    如果参数指定为 Sequence 类型，则序列的每个元素将参与类型提升。

    type_promotion_kind 必须是 ELEMENTWISE_TYPE_PROMOTION_KIND 中指定的一种种类。
    查看其文档以获取详细信息。

    如果 return_dtype 可用且不为 None，则将其强制转换为包装函数的 dtype 参数。

    """
    def __init__(self, type_promoting_args: Sequence[str], type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND):
        pass  # 构造函数不需要额外操作，只需存储参数

    def __call__(self, f: Callable) -> Callable:
        pass  # 装饰器类需要实现 __call__ 方法，用于装饰函数
    """
    Other type promotion behavior, like validating the Python type of scalar arguments, must
    be handled separately.
    """

    # 定义一个装饰器类，用于执行元素级类型提升
    def __init__(
        self,
        *,
        type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND,
        type_promoting_args: Optional[Sequence[str]] = None,
    ):
        # 设置类型提升参数的名称列表和类型提升的种类
        self.type_promoting_arg_names = type_promoting_args
        self.type_promotion_kind = type_promotion_kind

    # 实现装饰器的__call__方法，用于装饰函数fn
    def __call__(self, fn: Callable) -> Callable:
        # 获取函数fn的参数签名
        sig = inspect.signature(fn)

        @wraps(fn)
        def _fn(*args, **kwargs):
            # 使用函数签名绑定传入的参数
            bound = sig.bind(*args, **kwargs)

            # 提取类型提升参数的实际值
            type_promoting_args = tuple(
                bound.arguments[x]
                for x in self.type_promoting_arg_names  # type: ignore[union-attr]
                if x in bound.arguments.keys()
            )

            # 展平类型提升参数的树形结构
            flattened_type_promoting_args = pytree.arg_tree_leaves(*type_promoting_args)

            # 计算计算数据类型和结果数据类型
            compute_dtype, result_dtype = utils.elementwise_dtypes(
                *flattened_type_promoting_args,
                type_promotion_kind=self.type_promotion_kind,
            )

            # 将参数按照计算数据类型进行可能的转换
            promoted_args = {
                x: _maybe_convert_to_dtype(bound.arguments[x], compute_dtype)
                for x in self.type_promoting_arg_names  # type: ignore[union-attr]
                if x in bound.arguments.keys()
            }
            bound.arguments.update(promoted_args)

            # 调用原始函数并获取结果
            result = fn(**bound.arguments)

            # 如果存在dtype参数且不为None，则覆盖结果数据类型
            if "dtype" in bound.arguments:
                maybe_dtype = bound.arguments["dtype"]
                if maybe_dtype:  # dtype cannot be None
                    result_dtype = maybe_dtype

            # 根据结果的类型进行可能的数据类型转换
            if isinstance(result, TensorLike):
                return _maybe_convert_to_dtype(result, result_dtype)
            if isinstance(result, Sequence):
                return tuple(_maybe_convert_to_dtype(x, result_dtype) for x in result)

            # 如果结果类型未处理，则抛出断言错误
            raise AssertionError(f"Unhandled result type: {type(result)}")

        _fn.__signature__ = sig  # type: ignore[attr-defined]
        return _fn
# Returns True if resize is necessary
def _resize_output_check(out: TensorLikeType, shape: ShapeType):
    # If the shapes are correct there's nothing to do
    if utils.same_shape(out.shape, shape):
        return False
    # Check if the output tensor is non-empty
    if out.numel() != 0:
        # Warn about resizing non-empty tensors, which will be deprecated
        msg = (
            f"An output with one or more elements was resized since it had shape {str(out.shape)} "
            "which does not match the required output shape {str(shape)}. "
            "This behavior is deprecated, and in a future PyTorch release outputs will not "
            "be resized unless they have zero elements. "
            "You can explicitly reuse an out tensor t by resizing it, inplace, to zero elements with t.resize_(0)."
        )
        warnings.warn(msg)
    return True


# TODO: handle tuples of tensors
def _maybe_resize_out(
    out: TensorLikeType,
    shape: ShapeType,
    memory_format: Optional[torch.memory_format] = None,
):
    # Check if resizing is necessary
    if _resize_output_check(out, shape):
        # Resize the output tensor to the specified shape
        return out.resize_(shape, memory_format=memory_format)
    else:
        return out


def _safe_copy_out(
    *, copy_from: TensorLikeType, copy_to: TensorLikeType, exact_dtype: bool = False
):
    # Checks if both tensors are on the same device
    if copy_from.device != copy_to.device:
        msg = (
            f"Attempting to copy from device {copy_from.device} "
            f"to device {copy_to.device}, but cross-device copies are not allowed!"
        )
        raise RuntimeError(msg)

    # Checks if exact dtype matching is required
    if exact_dtype:
        # Ensures exact dtype match between source and destination tensors
        torch._check(
            copy_from.dtype == copy_to.dtype,
            lambda: f"Expected out tensor to have dtype {copy_from.dtype} "
            f"but got {copy_to.dtype} instead",
        )
    else:
        # Checks if casting between dtypes is safe
        torch._check(
            utils.can_safe_cast_to(cast_from=copy_from.dtype, cast_to=copy_to.dtype),
            lambda: f"Attempting to cast from {copy_from.dtype} to out tensor with dtype {copy_to.dtype}, "
            "but this can't be cast because it is not safe!",
        )

    return copy_to.copy_(copy_from)


def out_wrapper(
    *out_names: str,
    exact_dtype: bool = False,
    pass_is_out: bool = False,
    preserve_memory_format=False,
):
    # The wrapped function needs to convert the output parameters to ensure
    # compatibility between the Python API (which always uses "out" as the
    # parameter name and may be a tuple) and the Aten API (which may have
    # multiple output parameters and use different parameter names such as
    # "grad_input", "indices" or "values".)

    default_out_names = ("out",)
    if len(out_names) == 0:
        # Use default in out name if no specific names are provided
        out_names = default_out_names

    # Check if only a single tensor output is expected
    is_tensor = len(out_names) == 1

    def maybe_compute_memory_format(t):
        # Determine memory format suggestion if preservation is enabled
        return utils.suggest_memory_format(t) if preserve_memory_format else None

    # Return the internal wrapper function
    return _out_wrapper


def _maybe_remove_out_wrapper(fn: Callable):
    # Unwrap the function to its original form, checking for the presence of the wrapper attribute
    return inspect.unwrap(
        fn,
        stop=lambda f: not hasattr(f, "_torch_decompositions_out_wrapper"),
    )
# 定义一个装饰器函数，用于处理不支持反向传播的操作
def backwards_not_supported(prim):
    # 定义一个函数，重新分发原始操作
    def redispatch_prim(args, kwargs):
        # 设置自动调度以下自动求导过程
        with torch._C._AutoDispatchBelowAutograd():
            # 临时排除指定的调度键，确保执行原始操作
            old = torch._C._dispatch_tls_is_dispatch_key_excluded(
                torch._C.DispatchKey.ADInplaceOrView
            )
            # 调用原始操作并返回结果
            return prim(*args, **kwargs)

    # 定义一个继承自torch.autograd.Function的类，用于不支持反向传播的情况
    class BackwardsNotSupported(torch.autograd.Function):
        @staticmethod
        def forward(ctx, args_spec, *flat_args):
            # 解构扁平化的参数为原始参数和关键字参数
            args, kwargs = tree_unflatten(flat_args, args_spec)  # type: ignore[arg-type]
            # 调用redispatch_prim函数处理前向传播
            return redispatch_prim(args, kwargs)

        @staticmethod
        def backward(ctx, *args):
            # 抛出异常，不支持后向传播
            raise RuntimeError("backwards not supported on prim")

    # 使用prim函数的装饰器，处理支持自动求导的情况
    @wraps(prim)
    def _autograd_impl(*args, **kwargs):
        # 扁平化参数并获取参数规范
        flat_args, args_spec = tree_flatten((args, kwargs))
        # 如果梯度开启且存在需要梯度的张量，则执行以下操作
        if torch.is_grad_enabled() and any(
            a.requires_grad for a in flat_args if isinstance(a, torch.Tensor)
        ):
            # 返回BackwardsNotSupported类的应用结果
            return BackwardsNotSupported.apply(args_spec, *flat_args)
        else:
            # 否则执行redispatch_prim函数处理前向传播
            return redispatch_prim(args, kwargs)

    return _autograd_impl


# TODO: 当进行追踪时，此装饰器将添加torch张量而不是TensorMeta对象到追踪结果中
# 我们应该通过添加追踪上下文和NumberMeta类来修复这个问题
# TODO: 当前这个装饰器没有经过测试
def elementwise_unary_scalar_wrapper(fn: Callable) -> Callable:
    """
    允许接受张量的一元运算符与Python数字一起使用。
    """
    # 获取函数的签名信息
    sig = inspect.signature(fn)

    @wraps(fn)
    def _fn(*args, **kwargs):
        # 如果参数长度大于0且第一个参数是Number类型，则执行以下操作
        if len(args) > 0 and isinstance(args[0], Number):
            # 将第一个参数转换为对应的torch张量
            dtype = utils.type_to_dtype(type(args[0]))
            args_ = list(args)
            args_[0] = torch.tensor(args[0], dtype=dtype)
            # 执行函数并确保返回的是torch张量
            result = fn(*args_, **kwargs)
            assert isinstance(result, torch.Tensor)
            # 返回张量的标量值
            return result.item()

        # 否则直接执行原始函数
        return fn(*args, **kwargs)

    # 将函数签名设置回装饰后的函数
    _fn.__signature__ = sig  # type: ignore[attr-defined]
    return _fn
```