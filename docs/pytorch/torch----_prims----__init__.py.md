# `.\pytorch\torch\_prims\__init__.py`

```py
# mypy: allow-untyped-defs
# 引入上下文管理模块
import contextlib
# 引入迭代工具模块
import itertools
# 引入运算符模块
import operator
# 引入弱引用模块
import weakref
# 引入枚举类型模块
from enum import Enum
# 引入偏函数模块和reduce函数
from functools import partial, reduce
# 引入类型提示相关模块
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union

# 引入PyTorch核心模块
import torch

# 引入PyTorch私有API中的公共函数模块
import torch._prims_common as utils
# 引入PyTorch库模块
import torch.library
# 引入torch库中的相关功能
from torch import sym_float, Tensor, TypedStorage
# 引入PyTorch C++扩展模块中的默认设备获取函数
from torch._C import _get_default_device
# 引入torch._library.utils模块中的函数
from torch._library.utils import is_functional_schema
# 引入PyTorch调试基元操作模块
from torch._prims.debug_prims import register_debug_prims
# 引入PyTorch随机数生成基元操作模块
from torch._prims.rng_prims import register_rng_prims
# 引入PyTorch公共基元操作模块
from torch._prims_common import (
    Dim,
    DimsSequenceType,
    DimsType,
    IntLike,
    Number,
    NumberType,
    RETURN_TYPE,
    ShapeType,
    StrideType,
    TensorLike,
    TensorLikeType,
    type_to_dtype,
)
# 引入PyTorch公共基元操作模块中的后向不支持函数
from torch._prims_common.wrappers import backwards_not_supported
# 引入PyTorch虚拟张量模块中的FakeTensor类和FakeTensorMode类
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
# 引入PyTorch函数重载模块中的函数
from torch.overrides import handle_torch_function, has_torch_function
# 引入PyTorch工具模块中的树操作函数
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

# 创建名为prim的PyTorch库对象，指定名称和版本
prim = torch.library.Library("prims", "DEF")
# 创建名为prim_impl的PyTorch库对象，指定名称、版本和具体实现类型
prim_impl = torch.library.Library("prims", "IMPL", "CompositeExplicitAutograd")
# 创建名为prim_backend_select_impl的PyTorch库对象，指定名称、版本和具体实现类型
prim_backend_select_impl = torch.library.Library("prims", "IMPL", "BackendSelect")
# 创建名为prim_autograd_impl的PyTorch库对象，指定名称、版本和具体实现类型
prim_autograd_impl = torch.library.Library("prims", "IMPL", "Autograd")
# 创建名为prim_meta_impl的PyTorch库对象，指定名称、版本和具体实现类型
prim_meta_impl = torch.library.Library("prims", "IMPL", "Meta")

# 实验性模块，包含原型“primitive”操作的定义，以下为公开接口列表
__all__ = [
    #
    # Common datastructures and helpers
    #
    "RETURN_TYPE",
    #
    # Elementwise unary prims
    #
    "abs",
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "cos",
    "cosh",
    "bessel_i0",
    "bessel_i0e",
    "bessel_i1",
    "bessel_i1e",
    "bessel_j0",
    "bessel_j1",
    "bitwise_not",
    "cbrt",
    "ceil",
    "conj_physical",
    "digamma",
    "erf",
    "erf_inv",
    "erfc",
    "erfcx",
    "exp",
    "expm1",
    "exp2",
    "fill",
    "floor",
    "imag",
    "isfinite",
    "lgamma",
    "log",
    "log1p",
    "log2",
    "log10",
    "ndtri",
    "neg",
    "real",
    "reciprocal",
    "round",
    "sign",
    "signbit",
    "sin",
    "sinh",
    "spherical_bessel_j0",
    "sqrt",
    "tan",
    "tanh",
    "trunc",
    #
    # Elementwise binary prims
    #
    "add",
    "atan2",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    # 'complex',  # 需要自定义元数据
    "div",
    "eq",
    "fmax",
    "fmin",
    "fmod",
    "frexp",
    "gcd",
    "ge",
    "gt",
    "hypot",
    "igamma",
    "igammac",
    "le",
    "lt",
    "maximum",
    "minimum",
    "mul",
    "ne",
    "nextafter",
    "pow",
    "remainder",
    "rsqrt",
    "shift_left",
    "shift_right_arithmetic",
    "shift_right_logical",  # 尚未实现
    "sub",
    "zeta",
    #
    # View prims
    #
    "as_strided",
    "broadcast_in_dim",
    "collapse_view",
    "conj",
    "expand_dims",
    "slice",
    "slice_in_dim",  # 使用切片实现 -- 将其作为引用？
    "split_dim",  # 分割维度
    "squeeze",  # 压缩维度
    "transpose",  # 转置
    "view_of",  # 获取视图
    "view_element_type",  # 获取视图元素类型
    #
    # 函数化视图变异
    #
    "as_strided_scatter",  # 使用 strided 散列
    #
    # 形状基元
    #
    "collapse",  # 折叠
    "cat",  # 连接
    "reshape",  # 重塑
    "rev",  # 反转
    #
    # 条件基元
    #
    "where",  # 条件判断
    #
    # 数据转换和移动基元
    #
    "clone",  # 克隆
    "convert_element_type",  # 转换元素类型
    "device_put",  # 设备放置
    "item",  # 获取元素
    "maximum_value",  # 最大值
    "minimum_value",  # 最小值
    "copy_strided",  # 复制 strided
    #
    # 原地基元
    #
    "copy_to",  # 复制到
    "resize",  # 调整大小
    # "_set",  # 已注释，参见下面的说明
    #
    # 缩减基元
    #
    "amax",  # 最大值
    "amin",  # 最小值
    "prod",  # 乘积
    "sum",  # 求和
    "xor_sum",  # 异或和
    "var",  # 方差
    #
    # 张量创建基元
    #
    "empty_strided",  # 创建空 strided 张量
    "empty_permuted",  # 创建空置换张量
    "scalar_tensor",  # 标量张量
    "iota",  # 递增序列
    #
    # 线性代数 (linalg) 基元
    #
    "svd",  # 奇异值分解
    #
    # 随机性基元
    #
    "normal",  # 正态分布
    "_uniform_helper",  # 均匀分布辅助函数
    #
    # FFT 基元
    #
    "fft_r2c",  # 实数到复数的快速傅里叶变换
    "fft_c2c",  # 复数到复数的快速傅里叶变换
    "fft_c2r",  # 复数到实数的快速傅里叶逆变换
    #
    # 用于生成/接收令牌的基元
    #
    "_make_token",  # 生成令牌
    "_sink_tokens",  # 接收令牌
# 定义一个函数 `TensorMeta`，用于生成包含张量元信息的张量
def TensorMeta(
    tensorlike: Optional[Union[NumberType, torch.Tensor]] = None,
    *,
    shape: Optional[ShapeType] = None,
    strides: Optional[StrideType] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[torch.device, str]] = None,
):
    # 如果 `tensorlike` 是一个数字类型
    if isinstance(tensorlike, Number):
        # 断言 `shape` 为空，并且 `shape` 是 `None` 或者是一个序列
        assert not shape and (shape is None or isinstance(shape, Sequence))
        # 断言 `strides` 为空，并且 `strides` 是 `None` 或者是一个序列
        assert not strides and (strides is None or isinstance(strides, Sequence))
        inferred_shape: Tuple[int, ...] = ()
        inferred_strides: Tuple[int, ...] = ()
        inferred_dtype = type_to_dtype(type(tensorlike))  # 推断数据类型
        inferred_device = torch.device("cpu")  # 推断设备为 CPU
        # TODO: This looks wrong, a number that is wrapped into a tensor
        # needs to behave differently than a scalar tensor for type
        # promotion purposes
    # 如果 `tensorlike` 不为空且是 `torch.Tensor` 类型
    elif tensorlike is not None:
        inferred_shape = tuple(tensorlike.shape)  # 推断形状
        inferred_strides = tuple(tensorlike.stride())  # 推断步长
        inferred_dtype = tensorlike.dtype  # 推断数据类型
        inferred_device = tensorlike.device  # 推断设备
    else:
        # 如果没有给定 `tensorlike` 作为例子，则必须明确提供所有元数据
        assert shape is not None  # 断言 `shape` 不为空
        assert strides is not None  # 断言 `strides` 不为空
        assert dtype is not None  # 断言 `dtype` 不为空
        assert device is not None  # 断言 `device` 不为空

    # 如果 `shape` 是 `None`，则使用推断的形状；否则将其转换为元组形式
    shape = inferred_shape if shape is None else tuple(shape)  # type: ignore[possibly-undefined]
    # 如果 `strides` 是 `None`，则使用推断的步长；否则将其转换为元组形式
    strides = inferred_strides if strides is None else tuple(strides)  # type: ignore[possibly-undefined]
    # 如果 `dtype` 是 `None`，则使用推断的数据类型；否则使用明确提供的数据类型
    dtype = inferred_dtype if dtype is None else dtype  # type: ignore[possibly-undefined]
    # 如果 `device` 是 `None`，则使用推断的设备；否则根据情况转换为 `torch.device` 对象
    device = inferred_device if device is None else device  # type: ignore[possibly-undefined]

    # 如果 `device` 是字符串类型，则将其转换为 `torch.device` 对象
    if isinstance(device, str):
        device = torch.device(device)

    # 返回一个空的张量，指定形状、步长、数据类型和设备
    return torch.empty_strided(shape, strides, dtype=dtype, device=device)


# 定义一个函数 `_make_prim`，用于创建原始操作
def _make_prim(
    *,
    schema: str,  # 操作的模式描述
    return_type: Union[RETURN_TYPE, Tuple[RETURN_TYPE, ...]],  # 返回值的类型
    meta: Callable,  # 元信息函数
    impl_aten: Callable,  # ATen 实现函数
    doc: str,  # 操作的文档字符串
    tags: Optional[Sequence[torch.Tag]] = None,  # 操作的标签（可选）
    use_old_custom_ops_api: bool = False,  # 是否使用旧的自定义操作 API
):
    """
    创建一个原始操作。
    """

    # 定义 `_prim_impl` 函数，用于实际执行操作
    def _prim_impl(*args, **kwargs):
        # 总是调用 `meta` 函数，因为 ATen 实现通常接受更多的输入（例如，进行类型提升和广播），我们希望拒绝这些情况
        meta(*args, **kwargs)
        # 返回 `impl_aten` 函数的执行结果
        return impl_aten(*args, **kwargs)

    # 现在 prims 不支持自动求导（我们可以和应该在这里添加一个参数，为反向传播提供实现）
    # 因为我们没有导数公式，必须设置一个自定义的自动求导函数，如果调用了反向传播则会抛出错误
    def _autograd_impl(*args, **kwargs):
        return backwards_not_supported(_prim)(*args, **kwargs)
    # 定义内部函数 _backend_select_impl，根据参数和关键字参数选择合适的实现函数
    def _backend_select_impl(*args, **kwargs):
        # 如果关键字参数中包含 "device" 并且其类型为 "meta"，则调用 meta 函数处理
        if kwargs.get("device") and kwargs["device"].type == "meta":
            return meta(*args, **kwargs)
        # 如果 args 中有任何一个元素是 torch.device 类型且类型为 "meta"，同样调用 meta 函数处理
        if any(isinstance(x, torch.device) and x.type == "meta" for x in args):
            return meta(*args, **kwargs)
        else:
            # 否则调用 _prim_impl 函数处理
            return _prim_impl(*args, **kwargs)

    # 根据 schema 字符串分割出操作名称 name
    name = schema.split("(")[0]
    # 将 schema 字符串去除操作名称部分，保留其余部分
    schema = schema[len(name):]

    # 使用旧的自定义操作 API 或者 schema 不是功能性的时候，注册非功能性操作
    cpp_schema = torch._C.parse_schema(name + schema)
    if use_old_custom_ops_api or not is_functional_schema(cpp_schema):
        # 定义操作并标记为兼容 pt2 的标签
        prim.define(name + schema, tags=torch.Tag.pt2_compliant_tag)
        # 注册操作的实现函数和自动求导实现函数
        prim_impl.impl(name, _prim_impl)
        prim_autograd_impl.impl(name, _autograd_impl)
        # 注册 meta 函数作为操作的元实现函数
        prim_meta_impl.impl(name, meta)
    else:
        # 如果 schema 中的参数具有写入别名信息，则记录其参数名称
        mutates_args = []
        for arg in cpp_schema.arguments:
            if arg.alias_info is not None and arg.alias_info.is_write:
                mutates_args.append(arg.name)
        # 注册自定义操作，并设置 mutates_args 和 schema
        prim_def = torch.library.custom_op(
            "prims::" + name,
            _prim_impl,
            mutates_args=tuple(mutates_args),
            schema=schema,
        )
        # 注册 meta 函数作为伪造的实现函数
        prim_def.register_fake(meta)

    # 获取 _ops.prims 中的操作包 _prim_packet
    _prim_packet = getattr(torch._ops.ops.prims, name)
    # 获取默认的操作 _prim
    _prim = _prim_packet.default
    # 如果有 tags 参数，则设置操作的标签
    if tags:
        _prim._tags = tags

    # 导入包含张量类型的假张量子类，检查操作的参数是否包含张量类型
    from torch._subclasses.fake_tensor import contains_tensor_types
    # 如果操作的参数中不包含张量类型，或者操作名称在特定列表中，设置对应的 backend_select 实现函数
    if not any(contains_tensor_types(a.type) for a in _prim._schema.arguments) or str(
        _prim
    ) in [
        "prims.device_put.default"  # 参考链接中的问题
    ]:
        prim_backend_select_impl.impl(name, _backend_select_impl)

    # 对 _prim_packet 和 _prim 进行文档设置、返回类型设置、schema 设置、实现函数设置
    for p in (_prim_packet, _prim):
        p.__doc__ = doc
        p.return_type = return_type  # type: ignore[attr-defined]
        p.schema = schema
        p.prim_impl = _prim_impl
        p.prim_meta_impl = meta
        p.impl_aten = impl_aten

    # 返回最终确定的 _prim 函数
    return _prim
# 定义枚举类型 ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND，用于指定元素级操作的类型提升行为
class ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND(Enum):
    # 默认类型提升行为，对应值为 (0,)
    DEFAULT = (0,)
    # 整数到浮点数的类型提升行为，对应值为 (2,)
    INT_TO_FLOAT = (2,)
    # 布尔类型始终保持不变的类型提升行为，对应值为 (3,)
    ALWAYS_BOOL = (3,)
    # 复数到浮点数的类型提升行为，对应值为 (4,)
    COMPLEX_TO_FLOAT = (4,)


# TODO: 在此处实现 dtype 的验证，或者在相应的引用位置上进行验证
# 定义元素级操作的元函数 _prim_elementwise_meta，其输出结果与输入保持相同的数据类型
def _prim_elementwise_meta(
    *args,
    type_promotion: ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND,
    args_with_fixed_dtypes: Optional[Tuple[TensorLikeType, ...]] = None,
) -> FakeTensor:
    """
    Meta function for elementwise operations that produce outputs in the same dtype
    as their inputs.

    Stride logic is currently incorrect.
    """

    # 断言参数 args 的长度大于 0
    assert len(args) > 0

    # 使用工具函数检查所有输入参数的数据类型是否相同
    utils.check_same_dtype(*args)

    # 如果提供了固定数据类型的参数 args_with_fixed_dtypes，则使用这些固定类型参数代替原始 args 中的对应位置参数
    args_ = list(args)
    if args_with_fixed_dtypes is not None:
        args_ = list(args_with_fixed_dtypes) + args_

    # 使用工具函数检查所有输入参数的设备是否相同，允许包含 CPU 标量张量
    utils.check_same_device(*args_, allow_cpu_scalar_tensors=True)
    # 使用工具函数检查所有输入参数的形状是否相同，允许包含 CPU 标量张量
    utils.check_same_shape(*args_, allow_cpu_scalar_tensors=True)

    # 计算元素级输出的逻辑到物理排列的排列顺序
    l2p_perm = utils.compute_elementwise_output_logical_to_physical_perm(*args_)
    # 提取所有输入参数的形状信息，允许包含 CPU 标量张量
    shape = utils.extract_shape(*args_, allow_cpu_scalar_tensors=True)

    # 获取数据类型 dtype 和标量类型 scalar_type
    dtype = None
    scalar_type = None
    for arg in args:
        if isinstance(arg, TensorLike):
            if not utils.is_cpu_scalar_tensor(arg):
                dtype = arg.dtype
                break
            else:
                dtype = arg.dtype
        elif isinstance(arg, Number):
            scalar_type = type(arg)

    # 如果未能从 TensorLike 参数中获取到 dtype，但获取到了标量类型 scalar_type，则通过工具函数将标量类型转换为 dtype
    if dtype is None and scalar_type is not None:
        dtype = utils.type_to_dtype(scalar_type)

    # 获取设备 device 或数字 number
    device = None
    number = None
    for arg in args_:
        if isinstance(arg, TensorLike):
            if utils.is_cpu_scalar_tensor(arg):
                if device is None:
                    device = arg.device
                # 继续遍历，以便在后面找到可能的 CUDA 张量
            else:
                device = arg.device
                break

        elif isinstance(arg, Number):
            if number is None:
                number = arg

    # 注意：这里的类型提升行为大部分被隐藏在测试中，因为通常情况下引用位置会正确处理类型提升，
    # 即使在这里处理不正确也会在追踪中插入过多的类型转换！
    # 如果设备不为 None，则需要指定数据类型 dtype
    if device is not None:
        assert dtype is not None  # 断言确保数据类型不为 None

        # 根据 type_promotion 的不同类型进行数据类型推断
        if type_promotion == ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT:
            dtype = dtype
        elif type_promotion == ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL:
            dtype = torch.bool
        elif type_promotion == ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.INT_TO_FLOAT:
            # 如果 dtype 是整数或布尔类型，则将其转换为默认浮点数类型
            if utils.is_integer_dtype(dtype) or utils.is_boolean_dtype(dtype):
                dtype = torch.get_default_dtype()
        elif type_promotion == ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT:
            # 如果 dtype 是复数类型，则将其转换为对应的实数类型；否则保持不变
            if utils.is_complex_dtype(dtype):
                dtype = utils.corresponding_real_dtype(dtype)
            else:
                dtype = dtype

        assert shape is not None  # 断言确保形状参数不为 None
        # 返回一个空的张量，按指定的排列顺序 l2p_perm，使用指定的设备和数据类型 dtype
        return torch.empty_permuted(shape, l2p_perm, device=device, dtype=dtype)  # type: ignore[return-value]

    # 处理数字类型的情况
    # TODO: 修复数字类型推断问题（布尔型，复数 -> 浮点数）

    # 目前仅实现整数和浮点数的常见/简单情况，例如 (int, float, symint, symfloat)
    seen_float = False
    if isinstance(number, (torch.SymInt, torch.SymFloat)):
        # 检查参数中是否有不支持的类型，目前暂不支持
        for a in args:
            assert isinstance(a, (int, float, torch.SymInt, torch.SymFloat)), "NYI"
            seen_float = seen_float or isinstance(a, (float, torch.SymFloat))
        if seen_float:
            # 如果参数中包含浮点数或符号浮点数，则将 number 转换为符号浮点数
            number = sym_float(number)

    # 返回一个 TensorMeta 对象，表示处理后的结果
    return TensorMeta(number)  # type: ignore[arg-type]
def _complex_only_elementwise_meta(*args, **kwargs):
    # 检查第一个参数的数据类型是否为复数类型
    torch._check(
        utils.is_complex_dtype(args[0].dtype), lambda: "Only complex dtype is supported"
    )
    # 调用基本元素级元信息函数，并返回结果
    return _prim_elementwise_meta(*args, **kwargs)


def _make_elementwise_unary_prim(
    name: str, *, type_promotion: ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND, **kwargs
):
    """
    Creates an elementwise unary prim.
    """
    # 创建一个元素级一元原语

    return _make_prim(
        schema=f"{name}(Tensor self) -> Tensor",
        meta=partial(_prim_elementwise_meta, type_promotion=type_promotion),
        return_type=RETURN_TYPE.NEW,
        **kwargs,
    )


def _make_elementwise_binary_prim(
    name: str, *, type_promotion: ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND, **kwargs
):
    """
    Creates an elementwise binary prim.
    """
    # 创建一个元素级二元原语

    return _make_prim(
        schema=f"{name}(Tensor self, Tensor other) -> Tensor",
        meta=partial(_prim_elementwise_meta, type_promotion=type_promotion),
        return_type=RETURN_TYPE.NEW,
        **kwargs,
    )


def _not_impl(*args, **kwargs):
    # 抛出未实现错误
    raise NotImplementedError


#
# Elementwise unary operations
#


abs = _make_elementwise_unary_prim(
    "abs",
    impl_aten=torch.abs,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT,
)

acos = _make_elementwise_unary_prim(
    "acos",
    impl_aten=torch.acos,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

acosh = _make_elementwise_unary_prim(
    "acosh",
    impl_aten=torch.acosh,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

asin = _make_elementwise_unary_prim(
    "asin",
    impl_aten=torch.asin,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

asinh = _make_elementwise_unary_prim(
    "asinh",
    impl_aten=torch.asinh,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

atan = _make_elementwise_unary_prim(
    "atan",
    impl_aten=torch.atan,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

atanh = _make_elementwise_unary_prim(
    "atanh",
    impl_aten=torch.atanh,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

cos = _make_elementwise_unary_prim(
    "cos",
    impl_aten=torch.cos,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

cosh = _make_elementwise_unary_prim(
    "cosh",
    impl_aten=torch.cosh,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bessel_j0 = _make_elementwise_unary_prim(
    "bessel_j0",
    impl_aten=torch.special.bessel_j0,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bessel_j1 = _make_elementwise_unary_prim(
    "bessel_j1",
    impl_aten=torch.special.bessel_j1,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bessel_i0 = _make_elementwise_unary_prim(
    "bessel_i0",
    impl_aten=torch.i0,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,


注释：


# 设置一个变量 type_promotion，并指定其值为 ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT


这行代码简单地将变量 `type_promotion` 设置为 `ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT` 的值。
)

bessel_i0e = _make_elementwise_unary_prim(
    "bessel_i0e",
    impl_aten=torch.special.i0e,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)
# 创建一个元素级的一元原语，计算修改的贝塞尔函数I₀e
# 使用 torch.special.i0e 实现，无额外文档说明

bessel_i1 = _make_elementwise_unary_prim(
    "bessel_i1",
    impl_aten=torch.special.i1,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)
# 创建一个元素级的一元原语，计算修改的贝塞尔函数I₁
# 使用 torch.special.i1 实现，无额外文档说明

bessel_i1e = _make_elementwise_unary_prim(
    "bessel_i1e",
    impl_aten=torch.special.i1e,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)
# 创建一个元素级的一元原语，计算修改的贝塞尔函数I₁e
# 使用 torch.special.i1e 实现，无额外文档说明

bitwise_not = _make_elementwise_unary_prim(
    "bitwise_not",
    impl_aten=torch.bitwise_not,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)
# 创建一个元素级的一元原语，按位取反操作
# 使用 torch.bitwise_not 实现，无额外文档说明

def _cbrt_aten(a: torch.Tensor) -> Tensor:
    torch._check(
        not a.is_complex(),
        lambda: "cbrt: Complex inputs not supported. Consider calling torch.pow(a, 1.0/3.0)",
    )
    # 检查输入张量是否为复数类型，若是则抛出异常
    # 返回输入张量的实数立方根
    # 注意，如果 a < 0，pow(a, (1. / 3.)) 返回复数
    # exp(1/3 * log(a)) = exp(1/3 * (log(abs(a)) + pi*i)) = cbrt(abs(a)) * e^{pi/3*i}
    # 这是一个复数
    # 更多信息请参见 https://en.cppreference.com/w/cpp/numeric/math/cbrt 中的 Note 部分
    return torch.copysign(torch.pow(a.abs(), 1 / 3), a)

cbrt = _make_elementwise_unary_prim(
    "cbrt",
    impl_aten=_cbrt_aten,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)
# 创建一个元素级的一元原语，计算立方根
# 使用 _cbrt_aten 实现，无额外文档说明

ceil = _make_elementwise_unary_prim(
    "ceil",
    impl_aten=torch.ceil,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)
# 创建一个元素级的一元原语，向上取整操作
# 使用 torch.ceil 实现，无额外文档说明

def _conj_physical_meta(input: TensorLikeType) -> TensorLikeType:
    if not input.dtype.is_complex:
        raise RuntimeError("prims.conj_physical is only defined for complex dtypes")

    strides = utils.compute_elementwise_output_strides(input)
    return TensorMeta(input, strides=strides)

conj_physical = _make_prim(
    schema="conj_physical(Tensor self) -> Tensor",
    meta=_conj_physical_meta,
    impl_aten=torch._conj_physical,
    doc="Returns the physical conjugation of a complex tensor",
    return_type=RETURN_TYPE.NEW,
)
# 创建一个原语，计算复数张量的物理共轭
# 使用 _conj_physical_meta 实现，返回新的张量类型
# 文档说明返回结果是复数张量的物理共轭

def _clone_meta(
    input: TensorLikeType, *, memory_format: torch.memory_format = torch.preserve_format
) -> TensorLikeType:
    if memory_format != torch.preserve_format:
        return torch.empty(
            input.shape,
            dtype=input.dtype,
            layout=input.layout,
            device=input.device,
            memory_format=memory_format,
        )

    # memory_format == torch.preserve_format
    strides = utils.compute_elementwise_output_strides(input)
    return torch.empty_strided(
        input.shape,
        strides,
        dtype=input.dtype,
        layout=input.layout,
        device=input.device,
    )

clone = _make_prim(
    schema="clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor",
    meta=_clone_meta,
    impl_aten=torch.clone,
    doc="Returns the copy of a tensor",
)
# 创建一个原语，复制张量
# 使用 _clone_meta 实现，支持指定内存格式
# 文档说明返回结果是张量的复制
    return_type=RETURN_TYPE.NEW,


注释：


# 设置函数的返回类型为 RETURN_TYPE.NEW


这行代码设置了一个名为 `return_type` 的关键字参数，它被赋值为 `RETURN_TYPE.NEW`。根据上下文来看，这很可能是函数定义的一部分，用来指定函数的返回类型。
# 创建 digamma 函数原语，使用 torch.digamma 实现
digamma = _make_elementwise_unary_prim(
    "digamma",
    impl_aten=torch.digamma,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

# 创建 erf 函数原语，使用 torch.erf 实现
erf = _make_elementwise_unary_prim(
    "erf",
    impl_aten=torch.erf,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

# 创建 erf_inv 函数原语，使用 torch.special.erfinv 实现
erf_inv = _make_elementwise_unary_prim(
    "erf_inv",
    impl_aten=torch.special.erfinv,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

# 创建 erfc 函数原语，使用 torch.special.erfc 实现
erfc = _make_elementwise_unary_prim(
    "erfc",
    impl_aten=torch.special.erfc,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

# 创建 erfcx 函数原语，使用 torch.special.erfcx 实现
erfcx = _make_elementwise_unary_prim(
    "erfcx",
    impl_aten=torch.special.erfcx,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

# 创建 exp 函数原语，使用 torch.exp 实现
exp = _make_elementwise_unary_prim(
    "exp",
    impl_aten=torch.exp,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

# 创建 expm1 函数原语，使用 torch.special.expm1 实现
expm1 = _make_elementwise_unary_prim(
    "expm1",
    impl_aten=torch.special.expm1,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

# 创建 exp2 函数原语，使用 torch.special.exp2 实现
exp2 = _make_elementwise_unary_prim(
    "exp2",
    impl_aten=torch.special.exp2,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

# 定义一个函数 _fill_meta，用于处理元数据，无实现细节注明
def _fill_meta(a: TensorLikeType, value: NumberType) -> TensorLikeType:
    return _prim_elementwise_meta(
        a, type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT
    )

# 创建 fill 函数原语，使用 torch.fill 实现
fill = _make_prim(
    schema="fill(Tensor self, Scalar value) -> Tensor",
    return_type=RETURN_TYPE.NEW,
    meta=_fill_meta,
    impl_aten=torch.fill,
    doc="",
)

# 创建 floor 函数原语，使用 torch.floor 实现
floor = _make_elementwise_unary_prim(
    "floor",
    impl_aten=torch.floor,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

# 创建 imag 函数原语，使用 torch.imag 实现，处理复数类型数据
imag = _make_prim(
    schema="imag(Tensor(a) self) -> Tensor(a)",
    meta=partial(
        _complex_only_elementwise_meta,
        type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT,
    ),
    return_type=RETURN_TYPE.VIEW,
    impl_aten=torch.imag,
    doc="",
)

# 创建 isfinite 函数原语，使用 torch.isfinite 实现，返回布尔值
isfinite = _make_elementwise_unary_prim(
    "isfinite",
    impl_aten=torch.isfinite,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)

# 创建 lgamma 函数原语，使用 torch.lgamma 实现
lgamma = _make_elementwise_unary_prim(
    "lgamma",
    impl_aten=torch.lgamma,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

# 创建 log 函数原语，使用 torch.log 实现
log = _make_elementwise_unary_prim(
    "log",
    impl_aten=torch.log,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

# 创建 log1p 函数原语，使用 torch.log1p 实现
log1p = _make_elementwise_unary_prim(
    "log1p",
    impl_aten=torch.log1p,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

# 创建 log2 函数原语，使用 torch.log2 实现
log2 = _make_elementwise_unary_prim(
    "log2",
    impl_aten=torch.log2,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)
log10 = _make_elementwise_unary_prim(
    "log10",
    impl_aten=torch.log10,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)
# 创建一个名为 'log10' 的一元操作，使用 torch.log10 实现，没有文档说明，采用默认的类型提升方式

real = _make_prim(
    schema="real(Tensor(a) self) -> Tensor(a)",
    meta=partial(
        _complex_only_elementwise_meta,
        type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT,
    ),
    return_type=RETURN_TYPE.VIEW,
    impl_aten=torch.real,
    doc="",
)
# 创建一个名为 'real' 的操作原语，定义了其模式和返回类型，使用 torch.real 实现，没有文档说明

reciprocal = _make_elementwise_unary_prim(
    "reciprocal",
    impl_aten=torch.reciprocal,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)
# 创建一个名为 'reciprocal' 的一元操作，使用 torch.reciprocal 实现，没有文档说明，采用默认的类型提升方式

ndtri = _make_elementwise_unary_prim(
    "ndtri",
    impl_aten=torch.special.ndtri,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)
# 创建一个名为 'ndtri' 的一元操作，使用 torch.special.ndtri 实现，没有文档说明，采用默认的类型提升方式

neg = _make_elementwise_unary_prim(
    "neg",
    impl_aten=torch.neg,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)
# 创建一个名为 'neg' 的一元操作，使用 torch.neg 实现，没有文档说明，采用默认的类型提升方式

round = _make_elementwise_unary_prim(
    "round",
    impl_aten=torch.round,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)
# 创建一个名为 'round' 的一元操作，使用 torch.round 实现，没有文档说明，采用默认的类型提升方式

rsqrt = _make_elementwise_unary_prim(
    "rsqrt",
    impl_aten=torch.rsqrt,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)
# 创建一个名为 'rsqrt' 的一元操作，使用 torch.rsqrt 实现，没有文档说明，采用默认的类型提升方式

sign = _make_elementwise_unary_prim(
    "sign",
    impl_aten=torch.sign,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)
# 创建一个名为 'sign' 的一元操作，使用 torch.sign 实现，没有文档说明，采用默认的类型提升方式

signbit = _make_elementwise_unary_prim(
    "signbit",
    impl_aten=torch.signbit,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)
# 创建一个名为 'signbit' 的一元操作，使用 torch.signbit 实现，没有文档说明，采用默认的类型提升方式

sin = _make_elementwise_unary_prim(
    "sin",
    impl_aten=torch.sin,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)
# 创建一个名为 'sin' 的一元操作，使用 torch.sin 实现，没有文档说明，采用默认的类型提升方式

sinh = _make_elementwise_unary_prim(
    "sinh",
    impl_aten=torch.sinh,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)
# 创建一个名为 'sinh' 的一元操作，使用 torch.sinh 实现，没有文档说明，采用默认的类型提升方式

spherical_bessel_j0 = _make_elementwise_unary_prim(
    "spherical_bessel_j0",
    impl_aten=torch.special.spherical_bessel_j0,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)
# 创建一个名为 'spherical_bessel_j0' 的一元操作，使用 torch.special.spherical_bessel_j0 实现，没有文档说明，采用默认的类型提升方式

sqrt = _make_elementwise_unary_prim(
    "sqrt",
    impl_aten=torch.sqrt,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)
# 创建一个名为 'sqrt' 的一元操作，使用 torch.sqrt 实现，没有文档说明，采用默认的类型提升方式

tan = _make_elementwise_unary_prim(
    "tan",
    impl_aten=torch.tan,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)
# 创建一个名为 'tan' 的一元操作，使用 torch.tan 实现，没有文档说明，采用默认的类型提升方式

tanh = _make_elementwise_unary_prim(
    "tanh",
    impl_aten=torch.tanh,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)
# 创建一个名为 'tanh' 的一元操作，使用 torch.tanh 实现，没有文档说明，采用默认的类型提升方式

trunc = _make_elementwise_unary_prim(
    "trunc",
    impl_aten=torch.trunc,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)
# 创建一个名为 'trunc' 的一元操作，使用 torch.trunc 实现，没有文档说明，采用默认的类型提升方式

add = _make_elementwise_binary_prim(
    name="add",
    impl_aten=torch.add,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)
# 创建一个名为 'add' 的二元操作，使用 torch.add 实现，没有文档说明，采用默认的类型提升方式

atan2 = _make_elementwise_binary_prim(
    name="atan2",
    impl_aten=torch.atan2,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)
# 创建一个名为 'atan2' 的二元操作，使用 torch.atan2 实现，没有文档说明，采用默认的类型提升方式
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,


# 设置变量type_promotion为ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT
# 创建一个二进制元素级别操作的原语函数，并命名为 bitwise_and
bitwise_and = _make_elementwise_binary_prim(
    "bitwise_and",  # 函数名
    impl_aten=torch.bitwise_and,  # 实现使用 torch 库中的 bitwise_and 函数
    doc="",  # 文档字符串为空
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,  # 元素级操作类型推广策略为默认
)

# 创建一个二进制元素级别操作的原语函数，并命名为 bitwise_or
bitwise_or = _make_elementwise_binary_prim(
    "bitwise_or",  # 函数名
    impl_aten=torch.bitwise_or,  # 实现使用 torch 库中的 bitwise_or 函数
    doc="",  # 文档字符串为空
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,  # 元素级操作类型推广策略为默认
)

# 创建一个二进制元素级别操作的原语函数，并命名为 bitwise_xor
bitwise_xor = _make_elementwise_binary_prim(
    "bitwise_xor",  # 函数名
    impl_aten=torch.bitwise_xor,  # 实现使用 torch 库中的 bitwise_xor 函数
    doc="",  # 文档字符串为空
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,  # 元素级操作类型推广策略为默认
)

# 创建一个二进制元素级别操作的原语函数，并命名为 div
div = _make_elementwise_binary_prim(
    "div",  # 函数名
    impl_aten=_div_aten,  # 实现使用 _div_aten 函数
    doc="",  # 文档字符串为空
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,  # 元素级操作类型推广策略为默认
)

# 创建一个二进制元素级别操作的原语函数，并命名为 eq
eq = _make_elementwise_binary_prim(
    "eq",  # 函数名
    impl_aten=torch.eq,  # 实现使用 torch 库中的 eq 函数
    doc="",  # 文档字符串为空
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,  # 元素级操作类型推广策略总是返回布尔类型
)

# 创建一个二进制元素级别操作的原语函数，并命名为 fmax
fmax = _make_elementwise_binary_prim(
    "fmax",  # 函数名
    impl_aten=torch.fmax,  # 实现使用 torch 库中的 fmax 函数
    doc="",  # 文档字符串为空
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,  # 元素级操作类型推广策略为默认
)

# 创建一个二进制元素级别操作的原语函数，并命名为 fmin
fmin = _make_elementwise_binary_prim(
    "fmin",  # 函数名
    impl_aten=torch.fmin,  # 实现使用 torch 库中的 fmin 函数
    doc="",  # 文档字符串为空
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,  # 元素级操作类型推广策略为默认
)

# 创建一个二进制元素级别操作的原语函数，并命名为 fmod
fmod = _make_elementwise_binary_prim(
    "fmod",  # 函数名
    impl_aten=torch.fmod,  # 实现使用 torch 库中的 fmod 函数
    doc="",  # 文档字符串为空
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,  # 元素级操作类型推广策略为默认
)

# 创建一个二进制元素级别操作的原语函数，并命名为 gcd
gcd = _make_elementwise_binary_prim(
    "gcd",  # 函数名
    impl_aten=torch.gcd,  # 实现使用 torch 库中的 gcd 函数
    doc="",  # 文档字符串为空
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,  # 元素级操作类型推广策略为默认
)

# 创建一个二进制元素级别操作的原语函数，并命名为 ge
ge = _make_elementwise_binary_prim(
    "ge",  # 函数名
    impl_aten=torch.ge,  # 实现使用 torch 库中的 ge 函数
    doc="",  # 文档字符串为空
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,  # 元素级操作类型推广策略总是返回布尔类型
)

# 创建一个二进制元素级别操作的原语函数，并命名为 gt
gt = _make_elementwise_binary_prim(
    "gt",  # 函数名
    impl_aten=torch.gt,  # 实现使用 torch 库中的 gt 函数
    doc="",  # 文档字符串为空
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,  # 元素级操作类型推广策略总是返回布尔类型
)

# 创建一个二进制元素级别操作的原语函数，并命名为 hypot
hypot = _make_elementwise_binary_prim(
    "hypot",  # 函数名
    impl_aten=torch.hypot,  # 实现使用 torch 库中的 hypot 函数
    doc="",  # 文档字符串为空
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,  # 元素级操作类型推广策略为默认
)

# 创建一个二进制元素级别操作的原语函数，并命名为 igamma
igamma = _make_elementwise_binary_prim(
    "igamma",  # 函数名
    impl_aten=torch.special.gammainc,  # 实现使用 torch 库中的 gammainc 函数
    doc="",  # 文档字符串为空
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,  # 元素级操作类型推广策略为默认
)

# 创建一个二进制元素级别操作的原语函数，并命名为 igammac
igammac = _make_elementwise_binary_prim(
    "igammac",  # 函数名
    impl_aten=torch.special.gammaincc,  # 实现使用 torch 库中的 gammaincc 函数
    doc="",  # 文档字符串为空
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,  # 元素级操作类型推广策略为默认
)

# 创建一个二进制元素级别操作的原语函数，并命名为 le
le = _make_elementwise_binary_prim(
    "le",  # 函数名
    impl_aten=torch.le,  # 实现使用 torch 库中的 le 函数
    doc="",  # 文档字符串为空
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,  # 元素级操作类型推广策略总是返回布尔类型
)
# 创建一个名为 `lt` 的元素级别二元原语，实现由 torch.lt 提供，无文档说明，类型提升默认为布尔型
lt = _make_elementwise_binary_prim(
    "lt",
    impl_aten=torch.lt,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)

# 定义一个私有函数 _maximum_aten，实现 torch.maximum，处理标量输入的特殊情况
def _maximum_aten(
    a: Union[TensorLikeType, NumberType], b: Union[TensorLikeType, NumberType]
) -> TensorLikeType:
    if isinstance(a, TensorLike) and isinstance(b, Number):
        b = scalar_tensor(b, dtype=a.dtype, device=a.device)
    elif isinstance(b, TensorLike) and isinstance(a, Number):
        a = scalar_tensor(a, dtype=b.dtype, device=b.device)

    return torch.maximum(a, b)  # type: ignore[arg-type]

# 创建一个名为 `maximum` 的元素级别二元原语，实现由 _maximum_aten 提供，无文档说明，类型提升默认
maximum = _make_elementwise_binary_prim(
    "maximum",
    impl_aten=_maximum_aten,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

# 定义一个私有函数 _minimum_aten，实现 torch.minimum，处理标量输入的特殊情况
def _minimum_aten(
    a: Union[TensorLikeType, NumberType], b: Union[TensorLikeType, NumberType]
) -> TensorLikeType:
    if isinstance(a, TensorLike) and isinstance(b, Number):
        b = scalar_tensor(b, dtype=a.dtype, device=a.device)
    elif isinstance(b, TensorLike) and isinstance(a, Number):
        a = scalar_tensor(a, dtype=b.dtype, device=b.device)

    return torch.minimum(a, b)  # type: ignore[arg-type]

# 创建一个名为 `minimum` 的元素级别二元原语，实现由 _minimum_aten 提供，无文档说明，类型提升默认
minimum = _make_elementwise_binary_prim(
    "minimum",
    impl_aten=_minimum_aten,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

# 创建一个名为 `mul` 的元素级别二元原语，实现由 torch.mul 提供，无文档说明，类型提升默认
mul = _make_elementwise_binary_prim(
    "mul",
    impl_aten=torch.mul,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

# 创建一个名为 `ne` 的元素级别二元原语，实现由 torch.ne 提供，无文档说明，类型提升总是布尔型
ne = _make_elementwise_binary_prim(
    "ne",
    impl_aten=torch.ne,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)

# 创建一个名为 `nextafter` 的元素级别二元原语，实现由 torch.nextafter 提供，无文档说明，类型提升默认
nextafter = _make_elementwise_binary_prim(
    "nextafter",
    impl_aten=torch.nextafter,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

# 创建一个名为 `pow` 的元素级别二元原语，实现由 torch.pow 提供，无文档说明，类型提升默认
pow = _make_elementwise_binary_prim(
    "pow",
    impl_aten=torch.pow,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

# 创建一个名为 `remainder` 的元素级别二元原语，实现由 torch.remainder 提供，无文档说明，类型提升默认
remainder = _make_elementwise_binary_prim(
    "remainder",
    impl_aten=torch.remainder,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

# 创建一个名为 `shift_left` 的元素级别二元原语，实现由 torch.bitwise_left_shift 提供，无文档说明，类型提升默认
shift_left = _make_elementwise_binary_prim(
    "shift_left",
    impl_aten=torch.bitwise_left_shift,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

# 创建一个名为 `shift_right_arithmetic` 的元素级别二元原语，实现由 torch.bitwise_right_shift 提供，无文档说明，类型提升默认
shift_right_arithmetic = _make_elementwise_binary_prim(
    "shift_right_arithmetic",
    impl_aten=torch.bitwise_right_shift,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

# shift_right_logical 是未实现的元素级别二元原语，无法提供说明或类型提升
shift_right_logical = _not_impl

# 创建一个名为 `sub` 的元素级别二元原语，实现由 torch.sub 提供，无文档说明，类型提升默认
sub = _make_elementwise_binary_prim(
    "sub",
    impl_aten=torch.sub,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

# 创建一个名为 `zeta` 的元素级别二元原语，实现由 torch.special.zeta 提供，无文档说明，类型提升默认
zeta = _make_elementwise_binary_prim(
    "zeta",
    impl_aten=torch.special.zeta,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)
# 定义一个函数 _as_strided_meta，接受四个参数：a（类似张量类型）、size（形状类型）、stride（步幅类型）、storage_offset（整数类型），返回类似张量类型
def _as_strided_meta(
    a: TensorLikeType, size: ShapeType, stride: StrideType, storage_offset: int
) -> TensorLikeType:
    # 断言确保 size 和 stride 的长度相同
    assert len(size) == len(stride)
    # 断言确保 storage_offset 大于等于 0
    assert storage_offset >= 0
    # 使用工具函数 utils.validate_strides 验证步幅的有效性
    utils.validate_strides(stride)
    # 使用工具函数 utils.validate_shape 验证形状的有效性
    utils.validate_shape(size)

    # 如果 size 中所有维度的乘积为 0，处理特殊情况以避免需要获取下面的存储
    if reduce(operator.mul, size) == 0:
        # 注意：这种特殊情况是为了避免需要获取下面的存储
        # 对于没有元素的形状，as_strided 是平凡有效的，所以可以通过
        pass
    # 如果 a 是 torch.Tensor 类型
    elif isinstance(a, torch.Tensor):
        # 使用 a._typed_storage() 获取类型化存储，并检查边界
        utils.check_in_bounds_for_storage(
            a._typed_storage(), size, stride, storage_offset
        )

    # 返回使用 torch.as_strided 创建的张量视图，使用给定的 size、stride 和 storage_offset
    return torch.as_strided(a, size, stride, storage_offset)


# 定义一个函数 _as_strided_aten，接受四个参数：a（张量类型）、size（形状类型）、stride（步幅类型）、storage_offset（整数类型），返回张量类型
def _as_strided_aten(
    a: Tensor, size: ShapeType, stride: StrideType, storage_offset: int
) -> Tensor:
    # 直接使用 torch.as_strided 创建张量视图，使用给定的 size、stride 和 storage_offset
    return torch.as_strided(a, size, stride, storage_offset)


# 定义一个多行字符串 _as_strided_doc，描述 as_strided 函数的作用
_as_strided_doc = """
    Creates a view of the tensor with the given shape (size), strides (stride) and
    storage offset (storage_offset).
"""

# 定义 as_strided 变量，使用 _make_prim 创建一个原语，其 schema 描述参数和返回类型，meta 使用 _as_strided_meta 函数，impl_aten 使用 _as_strided_aten 函数，doc 使用 _as_strided_doc 描述
as_strided = _make_prim(
    schema="as_strided(Tensor(a!) a, SymInt[] size, SymInt[] stride, SymInt storage_offset) -> Tensor(a!)",
    meta=_as_strided_meta,
    impl_aten=_as_strided_aten,
    return_type=RETURN_TYPE.VIEW,
    doc=_as_strided_doc,
)


# 定义一个函数 _broadcast_in_dim_meta，接受三个参数：a（类似张量类型）、shape（形状类型）、broadcast_dimensions（整数序列），无返回值
def _broadcast_in_dim_meta(
    a: TensorLikeType, shape: ShapeType, broadcast_dimensions: Sequence[int]
):
    # 导入 torch.fx.experimental.symbolic_shapes 的 guard_size_oblivious 函数
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

    # 类型检查，确保 a 是 TensorLike 类型，shape 和 broadcast_dimensions 是序列类型
    assert isinstance(a, TensorLike)
    assert isinstance(shape, Sequence)
    assert isinstance(broadcast_dimensions, Sequence)

    # 每个维度必须被考虑到
    assert a.ndim == len(broadcast_dimensions)

    # 广播形状必须至少有更多维度
    assert len(shape) >= a.ndim

    # broadcast_dimensions 必须是升序序列（没有相对重新排序的维度），每个维度必须在新形状内
    def _greater_than_reduce(acc, x):
        assert isinstance(x, Dim)
        assert x > acc
        assert x < len(shape)
        return x

    reduce(_greater_than_reduce, broadcast_dimensions, -1)

    # 对于每个索引 idx 和 broadcast_dimensions 中的新索引 new_idx
    for idx, new_idx in enumerate(broadcast_dimensions):
        # 如果 a 的形状的 idx 维度不是 1，则使用 guard_size_oblivious 确保其可以广播到 shape[new_idx]
        if not guard_size_oblivious(a.shape[idx] == 1):
            # 使用 torch._check 检查条件，确保形状可以广播
            torch._check(
                a.shape[idx] == shape[new_idx],
                lambda: f"{a.shape[idx]} must be broadcastable to {shape[new_idx]}",
            )

    # 创建空列表 new_strides，用于存储新的步幅值
    new_strides = []
    # 初始索引为 0
    original_idx = 0
    # 遍历给定形状（shape）的索引范围
    for idx in range(len(shape)):
        # 检查当前索引是否在广播维度列表中
        if idx in broadcast_dimensions:
            # 如果当前维度是广播的，将其步长设为零
            if guard_size_oblivious(a.shape[original_idx] != shape[idx]):
                # 如果形状不匹配，则使用零步长
                new_strides.append(0)
            else:
                # 否则使用原数组在原始索引处的步长
                new_strides.append(a.stride()[original_idx])
            # 增加原始索引以继续处理下一个维度
            original_idx = original_idx + 1
        else:
            # 如果当前维度不是广播的
            if guard_size_oblivious(shape[idx] != 1):
                # 如果形状不为1，则使用零步长
                new_strides.append(0)
            elif original_idx == a.ndim:
                # 如果原始索引已经超过数组的维度数，则步长设为1
                new_strides.append(1)
            else:
                # 否则计算步长为原数组在原始索引处的步长乘以原数组在原始索引处的大小
                new_strides.append(a.stride()[original_idx] * a.size()[original_idx])

    # 使用给定的形状和新的步长创建一个新的视图数组
    return a.as_strided(shape, new_strides, a.storage_offset())
# 定义函数 `_broadcast_in_dim_aten`，用于在指定的维度上广播张量 `a` 到指定的 `shape` 形状
def _broadcast_in_dim_aten(a, shape, broadcast_dimensions):
    # 将形状列表转换为可修改的列表
    s = list(shape)
    # 对于每个广播维度，将其对应位置的形状设置为 -1，表示允许广播
    for broadcast_dimension in broadcast_dimensions:
        s[broadcast_dimension] = -1

    # 将变量 v 初始化为张量 a
    v = a
    # 遍历修改后的形状 s
    for idx, x in enumerate(s):
        # 如果当前维度 x 不为 -1，则在该维度上扩展张量 v
        if x != -1:
            v = v.unsqueeze(idx)

    # 返回扩展后的张量 v，形状为 shape
    return v.expand(shape)


# 定义文档字符串，描述函数 `broadcast_in_dim` 的功能和使用注意事项
_broadcast_in_dim_doc = """
  Creates a view of a with the specified shape.

  Allows adding dimensions of any length and broadcasting
  dimensions of length one in a to any length.

  The location of the broadcast dimensions must be specified
  using the broadcast_dimensions argument. Changing the
  relative order of dimensions is not supported.
  """

# 使用 _make_prim 函数创建一个名为 broadcast_in_dim 的原语函数
broadcast_in_dim = _make_prim(
    schema="broadcast_in_dim(Tensor(a) a, SymInt[] shape, int[] broadcast_dimensions) -> Tensor(a)",
    meta=_broadcast_in_dim_meta,
    impl_aten=_broadcast_in_dim_aten,
    return_type=RETURN_TYPE.VIEW,
    doc=_broadcast_in_dim_doc,
)


# 定义函数 `_validate_collapse_args`，用于验证在进行折叠操作时的参数有效性
def _validate_collapse_args(a: Tensor, start: int, end: int) -> None:
    # 对于零维张量的特殊情况，将维度数设为最小为 1
    ndim = max(1, a.dim())
    # 使用 utils 模块中的函数验证起始索引和结束索引的有效性
    utils.validate_idx(ndim, start)
    utils.validate_idx(ndim, end)

    # 确保结束索引大于等于起始索引，以保证折叠操作不是空的区间
    torch._check_value(
        end >= start,
        lambda: f"Attempting to collapse but end, {end}, is less than start, {start}!",
    )


# 定义函数 `_collapsed_shape`，返回将指定范围内的维度合并为一个的张量形状
def _collapsed_shape(shape: ShapeType, start: int, end: int) -> Tuple[int, ...]:
    """
    Returns the shape of a with dims in [start, end) merged into a single dimension.
    """
    # 对于零维张量的特殊情况，将形状设为 (1,)
    shape = (1,) if len(shape) == 0 else tuple(shape)

    # 计算在指定范围内维度长度的乘积，作为合并后的维度长度
    dim_length = 1
    for s in shape[start : end + 1]:
        dim_length = dim_length * s

    # 返回合并维度后的新形状
    return shape[0:start] + (dim_length,) + shape[end + 1 :]


# 定义函数 `_collapse_view_helper`，帮助对指定的张量在指定维度范围内进行折叠操作
def _collapse_view_helper(
    a: TensorLikeType, start: int, end: int
) -> Tuple[Optional[ShapeType], Optional[StrideType]]:
    # 断言参数 a 是 TensorLike 类型的对象
    assert isinstance(a, TensorLike)

    # 引入 torch.fx.experimental.symbolic_shapes 模块中的函数
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

    # 验证折叠操作的参数有效性
    _validate_collapse_args(a, start, end)

    # 对于零维张量的特殊情况，设置默认的形状和步长
    if a.ndim == 0:
        shape = (1,)
        strides = (1,)
    else:
        shape = a.shape  # type: ignore[assignment]
        strides = a.stride()  # type: ignore[assignment]

    # 如果张量是零维或者起始索引等于结束索引，则直接返回当前的形状和步长
    if a.ndim == 0 or (end == start):
        return shape, strides

    # 否则，获取结束维度的长度和步长
    length = shape[end]
    stride = strides[end]
    # 从 `end - 1` 到 `start - 1` 的逆序遍历索引范围
    for idx in range(end - 1, start - 1, -1):
        # 检查 shape[idx] 和 shape[idx + 1] 是否为零，如果是则将 length 和 stride 设为零并跳出循环
        if guard_size_oblivious(shape[idx] == 0) or guard_size_oblivious(
            shape[idx + 1] == 0
        ):
            length = 0
            stride = 0
            break

        # 如果 shape[idx] 等于 1，则继续下一次循环
        if guard_size_oblivious(shape[idx] == 1):
            continue

        # 计算新的 length，乘以 shape[idx]
        length = length * shape[idx]

        # 更新 stride，取 strides[idx] 和当前 stride 的较大值
        if guard_size_oblivious(stride < strides[idx]):
            stride = stride
        else:
            stride = strides[idx]

        # 如果 a 中的元素数量大于零，并且 shape[idx + 1] 不为 1，并且 strides[idx] 不等于 strides[idx + 1] * shape[idx + 1]
        # 则返回 None, None
        if (
            guard_size_oblivious(a.numel() > 0)
            and guard_size_oblivious(shape[idx + 1] != 1)
            and not guard_size_oblivious(
                strides[idx] == strides[idx + 1] * shape[idx + 1]
            )
        ):
            return None, None

    # 根据处理后的 new_shape 和 new_strides 更新形状和步长
    new_shape = shape[:start] + (length,) + shape[end + 1 :]
    new_strides = strides[:start] + (stride,) + strides[end + 1 :]

    # 当输入的张量没有元素时，重新调整步长使其连续化
    if guard_size_oblivious(a.numel() == 0):
        new_strides = utils.make_contiguous_strides_for(new_shape)

    # 返回更新后的形状和步长
    return new_shape, new_strides
# 定义一个函数 `_collapse_view_meta`，接受一个张量或张量样式类型 `a`，以及开始和结束的整数索引 `start` 和 `end`，返回一个张量或张量样式类型
def _collapse_view_meta(a: TensorLikeType, start: int, end: int) -> TensorLikeType:
    # 调用辅助函数 `_collapse_view_helper`，返回新的形状和步幅
    new_shape, new_strides = _collapse_view_helper(a, start, end)

    # 如果新形状为 None，则抛出值错误
    if new_shape is None:
        msg = "Attempting to view a collapsed tensor, but no such view exists!"
        raise ValueError(msg)

    # 确保新步幅不为 None
    assert new_strides is not None
    # 返回通过 `as_strided` 方法创建的视图张量
    return a.as_strided(new_shape, new_strides, a.storage_offset())


# 定义函数 `_collapse_view_aten`，接受一个张量 `a`，以及开始和结束的整数索引 `start` 和 `end`，返回一个张量
def _collapse_view_aten(a: Tensor, start: int, end: int) -> Tensor:
    # 调用 `_collapsed_shape` 函数，计算新的折叠形状
    new_shape = _collapsed_shape(a.shape, start, end)
    # 使用 `view` 方法返回具有新形状的视图张量
    return a.view(new_shape)


# 字符串文档，描述了 `collapse_view` 函数的作用和用法
_collapse_view_doc = """
  Creates a view of a with the dimensions between
  start (inclusive) and end (exclusive) merged into a
  single dimension.

  If it's not possible to take such a view then an error
  is thrown. See collapse instead.

  The dimensions can be merged if and only if
  they are all "nested" with each other. That is, they all
  have the property that

  stride[i] = stride[i+1] * shape[i+1]

  for all i in [start, end - 1).
  """

# 定义 `collapse_view`，使用 `_make_prim` 函数创建一个原语
collapse_view = _make_prim(
    schema="collapse_view(Tensor(a) a, int start, int end) -> Tensor(a)",
    meta=_collapse_view_meta,  # 元数据函数
    impl_aten=_collapse_view_aten,  # ATen 实现函数
    return_type=RETURN_TYPE.VIEW,  # 返回类型为视图
    doc=_collapse_view_doc,  # 使用之前定义的文档字符串作为文档
)


# 定义 `_conj_meta` 函数，接受一个张量或张量样式类型 `a`，返回一个张量或张量样式类型
def _conj_meta(a: TensorLikeType) -> TensorLikeType:
    # 如果 `a` 的数据类型不是复数类型，则引发运行时错误
    if not a.dtype.is_complex:
        raise RuntimeError("Expected complex dtype in prims.conj")
    # 创建 `a` 的一个共轭视图，并更新其共轭状态
    out = a.as_strided(a.shape, a.stride(), a.storage_offset())
    torch._C._set_conj(out, not a.is_conj())
    return out


# 字符串文档，描述 `conj` 函数的作用和用法
_conj_doc = """
Returns a conjugated view of the original tensor
"""

# 定义 `conj`，使用 `_make_prim` 函数创建一个原语
conj = _make_prim(
    schema="conj(Tensor(a) a) -> Tensor(a)",
    meta=_conj_meta,  # 元数据函数
    impl_aten=torch.conj,  # ATen 实现函数
    return_type=RETURN_TYPE.VIEW,  # 返回类型为视图
    doc=_conj_doc,  # 使用之前定义的文档字符串作为文档
)


# 定义 `expand_dims` 函数，接受一个张量或张量样式类型 `a`，维度序列 `dimensions` 和可选的维度数 `ndim`，返回一个张量或张量样式类型
def expand_dims(
    a: TensorLikeType, dimensions: DimsSequenceType, ndim=None
) -> TensorLikeType:
    """
    Creates a view of a with a.ndim + len(dimensions) dimensions, with new
    dimensions of length one at the dimensions specified by dimensions.
    """
    # 如果提供了 `ndim`，则规范化维度序列，用于展开操作
    if ndim is not None:
        dims = sorted(utils.canonicalize_dims(ndim, dimensions))  # 类型：忽略参数类型
    else:
        dims = sorted(utils.canonicalize_dims(a.ndim, dimensions))  # 类型：忽略参数类型
    # 如果存在重复维度，引发值错误
    if len(set(dims)) != len(dims):
        msg = f"Received duplicate dimensions to expand in {str(dimensions)}"
        raise ValueError(msg)

    # 复制原始张量的形状
    new_shape = list(a.shape)
    # 在指定的维度上插入长度为一的新维度
    for idx in dims:
        new_shape.insert(idx, 1)

    # 计算广播维度，即不在 `dimensions` 中的维度
    broadcast_dimensions = [
        idx for idx in range(len(new_shape)) if idx not in dimensions
    ]
    # 调用 `broadcast_in_dim` 函数，返回在指定维度上广播的张量
    return broadcast_in_dim(a, new_shape, broadcast_dimensions)


# 注意：保存 Python 的切片对象，因为即将使用切片原语来覆盖其名称
pyslice: Type[slice] = slice  # 类型：忽略[有类型]


# 定义 `_slice_meta` 函数，接受一个张量或张量样式类型 `a`，起始索引序列 `start_indices` 和结束索引序列 `limit_indices`
def _slice_meta(
    a: TensorLikeType,
    start_indices: DimsSequenceType,
    limit_indices: DimsSequenceType,
    strides: Optional[StrideType] = None,


# 定义一个名为strides的变量，类型为Optional[StrideType]，默认值为None
# 定义一个函数 _slice_aten，用于在张量中进行切片操作
def _slice_aten(
    a: Tensor,  # 参数 a 是一个张量
    start_indices: DimsSequenceType,  # start_indices 是切片的起始索引的序列
    limit_indices: DimsSequenceType,  # limit_indices 是切片的终止索引的序列
    strides: Optional[StrideType] = None,  # 可选参数 strides 是切片的步长序列
) -> Tensor:  # 函数返回一个张量

    # 如果未指定步长 strides，则将其设为每个维度的默认值 1
    _strides = strides if strides is not None else [1] * len(start_indices)

    # 检查张量 a 的维度是否与 start_indices 的长度一致
    if a.ndim != len(start_indices):
        msg = f"Attempting to slice tensor of rank {a.ndim} with start_indices of length {len(start_indices)}!"
        raise ValueError(msg)

    # 检查张量 a 的维度是否与 limit_indices 的长度一致
    if a.ndim != len(limit_indices):
        msg = f"Attempting to slice tensor of rank {a.ndim} with limit_indices of length {len(limit_indices)}!"
        raise ValueError(msg)

    # 检查张量 a 的维度是否与 _strides 的长度一致
    if a.ndim != len(_strides):
        msg = f"Attempting to slice tensor of rank {a.ndim} with strides of length {len(limit_indices)}!"
        raise ValueError(msg)

    # 检查 start_indices 中是否有负数索引
    for x, y in zip(start_indices, a.shape):
        if x < 0:
            msg = f"Attempting to slice a tensor with a negative start index of {x}!"
            raise ValueError(msg)
        if x > y:
            msg = (
                f"Attempting to slice a tensor but a start index in {start_indices} is greater than"
                f" the length of its corresponding dimension in shape {a.shape}"
            )
            raise ValueError(msg)

    # 检查 limit_indices 中是否有负数索引或大于对应维度长度的索引
    for x, y, z in zip(limit_indices, a.shape, start_indices):
        if x < 0:
            msg = f"Attempting to slice a tensor with a negative stop index of {x}!"
            raise ValueError(msg)
        if x > y:
            msg = (
                f"Attempting to slice a tensor but a stop index in {limit_indices} is greater than the length of "
                f" its corresponding dimension in shape {a.shape}"
            )
            raise ValueError(msg)
        if x < z:
            msg = (
                f"Attempting to slice a tensor but a start index in {x} is greater than "
                f" its corresponding stop index {z}"
            )
            raise ValueError(msg)

    # 检查 _strides 中的步长是否为正数
    for x in _strides:
        if x <= 0:
            msg = f"Attempting to slice a tensor with a non-positive step of {x}!"
            raise ValueError(msg)

    # 计算切片后的新形状 new_shape
    new_shape = []
    for x, y, z in zip(start_indices, limit_indices, _strides):
        new_shape.append(1 + (y - x - 1) // z)

    # 计算新的步长 new_strides
    new_strides = []
    for x, y in zip(a.stride(), _strides):
        new_strides.append(x * y)

    # 返回使用 as_strided 方法创建的张量视图，表示切片后的结果
    return a.as_strided(new_shape, new_strides, a.storage_offset())


_slice_doc = """
    Creates a view of a "bounding box" within the tensor.

    The bounding box is specified independently in each of the tensor's dimensions.
    start_indices and limit_indices describe the box's boundaries for their corresponding
    dimensions. If strides is specified then they specify the step size between elements
"""
    # 返回在每个维度上按照给定的索引范围切片后的数据
    def slice_by_indices(data, indices):
        # 确保索引列表的长度与数据的维度相同
        assert len(indices) == data.ndim, "Number of indices must match number of dimensions"

        # 生成切片对象的元组，用于在每个维度上执行切片操作
        slices = tuple(slice(idx.start, idx.stop) for idx in indices)

        # 返回按照生成的切片对象切片后的数据
        return data[slices]
# 创建名为 `slice` 的函数，用于生成切片操作的原语
slice = _make_prim(
    schema="slice(Tensor(a) a, SymInt[] start_indices, SymInt[] limit_indices, SymInt[]? strides=None) -> Tensor(a)",
    meta=_slice_meta,  # 使用 `_slice_meta` 提供的元信息
    impl_aten=_slice_aten,  # 使用 `_slice_aten` 提供的实现函数
    return_type=RETURN_TYPE.VIEW,  # 返回类型为视图类型
    doc=_slice_doc,  # 使用 `_slice_doc` 中的文档字符串
)


def _slice_in_dim_meta(
    a: TensorLikeType,
    start_index: int,
    limit_index: int,
    stride: int = 1,
    axis: int = 0,
) -> TensorLikeType:
    if axis < 0:
        msg = f"slice_in_dim: received a negative axis {axis}"
        raise ValueError(msg)  # 如果 `axis` 是负数，抛出值错误异常

    if axis >= a.ndim:
        msg = f"slice_in_dim: axis {axis} is greater or equal to the rank {a.ndim} of the tensor"
        raise ValueError(msg)  # 如果 `axis` 大于或等于 `a` 的秩数，抛出值错误异常

    if start_index < 0:
        msg = f"slice_in_dim: received a negative start_index {start_index}"
        raise ValueError(msg)  # 如果 `start_index` 是负数，抛出值错误异常

    if start_index > a.shape[axis]:
        msg = f"slice_in_dim: start_index is greater than the length {start_index} of dimension {axis}"
        raise ValueError(msg)  # 如果 `start_index` 大于指定维度 `axis` 的长度，抛出值错误异常

    if limit_index > a.shape[axis]:
        msg = f"slice_in_dim: limit_index is greater than the length {limit_index} of dimension {axis}"
        raise ValueError(msg)  # 如果 `limit_index` 大于指定维度 `axis` 的长度，抛出值错误异常

    if limit_index < start_index:
        msg = f"slice_in_dim: received a limit_index {limit_index} less than the start_index {start_index}"
        raise ValueError(msg)  # 如果 `limit_index` 小于 `start_index`，抛出值错误异常

    if stride < 0:
        msg = f"slice_in_dim: received a non-positive stride of {stride}!"
        raise ValueError(msg)  # 如果 `stride` 是负数或零，抛出值错误异常

    start_indices = [0] * a.ndim  # 创建一个长度为 `a` 的秩数的列表，初始值为 0
    limit_indices = list(a.shape)  # 创建一个 `a.shape` 的副本作为列表
    strides = [1] * a.ndim  # 创建一个长度为 `a` 的秩数的列表，初始值为 1

    start_indices[axis] = start_index  # 将指定维度 `axis` 的起始索引设置为 `start_index`
    limit_indices[axis] = limit_index  # 将指定维度 `axis` 的结束索引设置为 `limit_index`
    strides[axis] = stride  # 将指定维度 `axis` 的步长设置为 `stride`

    return _slice_meta(a, start_indices, limit_indices, strides)  # 调用 `_slice_meta` 函数进行切片操作的元信息


def _slice_in_dim_aten(
    a: Tensor,
    start_index: int,
    limit_index: int,
    stride: int = 1,
    axis: int = 0,
) -> Tensor:
    start_indices = [0] * a.ndim  # 创建一个长度为 `a` 的秩数的列表，初始值为 0
    limit_indices = list(a.shape)  # 创建一个 `a.shape` 的副本作为列表
    strides = [1] * a.ndim  # 创建一个长度为 `a` 的秩数的列表，初始值为 1

    start_indices[axis] = start_index  # 将指定维度 `axis` 的起始索引设置为 `start_index`
    limit_indices[axis] = limit_index  # 将指定维度 `axis` 的结束索引设置为 `limit_index`
    strides[axis] = stride  # 将指定维度 `axis` 的步长设置为 `stride`

    return slice(a, start_indices, limit_indices, strides)  # 调用 `slice` 函数执行切片操作


_slice_in_dim_doc = """
    Convenience wrapper for slicing just one dimension using slice.
    """
# `slice_in_dim` 函数的文档字符串，为仅使用 `slice` 切片一个维度的便捷包装器
slice_in_dim = _make_prim(
    schema="slice_in_dim(Tensor(a) a, SymInt start_index, SymInt limit_index, int stride=1, int axis=0) -> Tensor(a)",
    meta=_slice_in_dim_meta,  # 使用 `_slice_in_dim_meta` 函数提供的元信息
    impl_aten=_slice_in_dim_aten,  # 使用 `_slice_in_dim_aten` 函数提供的实现
    return_type=RETURN_TYPE.VIEW,  # 返回类型为视图类型
    doc=_slice_in_dim_doc,  # 使用 `_slice_in_dim_doc` 中的文档字符串
)


def _split_dim_meta(a: TensorLikeType, dim: int, outer_length: int) -> TensorLikeType:
    assert isinstance(a, TensorLike)  # 断言 `a` 是 `TensorLike` 类型
    utils.validate_idx(a.ndim, dim)  # 使用 `utils` 模块验证 `a.ndim` 和 `dim`
    utils.validate_dim_length(outer_length)  # 使用 `utils` 模块验证 `outer_length`

    # Verifies the dim can be split with the specified lhs_length
    inner_length = a.shape[dim] // outer_length
    # 检查给定维度是否可以被 outer_length 整除，否则抛出数值错误异常
    if (a.shape[dim] % outer_length) != 0:
        # 构造错误信息，指示尝试分割的维度长度及无法整除的外部长度
        msg = (
            f"Attempting to split dimension of length {a.shape[dim]}, "
            f"but outer length of {outer_length} divides it with a remainder!"
        )
        raise ValueError(msg)

    # 初始化新的形状和步长列表
    new_shape: List[int] = []
    new_strides: List[int] = []
    # 遍历数组 a 的维度索引
    for idx in range(a.ndim):
        # 如果当前索引等于指定的维度 dim
        if idx == dim:
            # 在新形状列表中添加外部长度和内部长度
            new_shape.extend((outer_length, inner_length))
            # 在新步长列表中添加根据内部长度调整后的步长
            new_strides.extend((a.stride()[idx] * inner_length, a.stride()[idx]))
        else:
            # 如果当前索引不是指定的维度 dim，则保持原始形状和步长
            new_shape.append(a.shape[idx])
            new_strides.append(a.stride()[idx])

    # 使用给定的新形状、新步长和存储偏移创建一个新的扩展视图
    return a.as_strided(new_shape, new_strides, a.storage_offset())
# 根据给定的张量 `a`，在指定的维度 `dim` 上进行切分，使其分为两个维度：外部维度长度为 `outer_length`，内部维度长度为计算得出的 `inner_length`，即 `outer_length * inner_length = a.shape[dim]`。
def _split_dim_aten(a: Tensor, dim: int, outer_length: int) -> Tensor:
    # 计算内部维度的长度
    inner_length = a.shape[dim] // outer_length
    # 构建新的张量形状，将指定维度 `dim` 分割为 `(outer_length, inner_length)`
    new_shape = a.shape[0:dim] + (outer_length, inner_length) + a.shape[dim + 1 :]
    
    # 返回按新形状视图的张量
    return a.view(new_shape)


# 定义 `_split_dim` 函数的文档字符串，描述其功能和用法
_split_dim_doc = """
  Creates a view of a with the given dimension (of length l) split
  into two dimensions, with the outer of the two having
  length outer_length and the inner of the two having computed
  length inner_length such outer_length * inner_length = l.
  """

# 创建 `_split_dim` 函数，将其包装为原语 `_make_prim`
split_dim = _make_prim(
    schema="split_dim(Tensor(a) a, int dim, SymInt outer_length) -> Tensor(a)",
    meta=_split_dim_meta,  # 使用元数据 `_split_dim_meta`
    impl_aten=_split_dim_aten,  # 使用 ATen 实现 `_split_dim_aten`
    return_type=RETURN_TYPE.VIEW,  # 返回视图类型的张量
    doc=_split_dim_doc,  # 使用上面定义的文档字符串 `_split_dim_doc`
)


# 注释：允许冗余地指定维度
def _squeeze_meta(a: TensorLikeType, dimensions: Sequence) -> TensorLikeType:
    assert isinstance(a, TensorLike)

    # 验证并确保每个指定的维度都为 1
    for idx in dimensions:
        utils.validate_idx(a.ndim, idx)
        assert a.shape[idx] == 1

    # 构建新的张量形状和步长
    new_shape = []
    new_strides = []
    for idx in range(len(a.shape)):
        if idx in dimensions:
            continue
        
        new_shape.append(a.shape[idx])
        new_strides.append(a.stride()[idx])

    # 返回按指定形状和步长展开的张量视图
    return a.as_strided(new_shape, new_strides, a.storage_offset())


# 定义 `_squeeze_meta` 函数的文档字符串，描述其功能和用法
_squeeze_doc = """
  Creates a view of the tensor with the specified dimensions removed.

  The removed dimensions must each have length one.
  """

# 创建 `_squeeze` 函数，将其包装为原语 `_make_prim`
squeeze = _make_prim(
    schema="squeeze(Tensor(a) a, int[] dimensions) -> Tensor(a)",
    meta=_squeeze_meta,  # 使用元数据 `_squeeze_meta`
    impl_aten=torch.squeeze,  # 使用 PyTorch 的 `squeeze` 函数实现
    return_type=RETURN_TYPE.VIEW,  # 返回视图类型的张量
    doc=_squeeze_doc,  # 使用上面定义的文档字符串 `_squeeze_doc`
)


# 根据指定的排列重排张量的维度
def _transpose_meta(a: TensorLikeType, permutation: DimsSequenceType) -> TensorLikeType:
    # 检查排列的长度是否与张量的维数相同
    if a.ndim != len(permutation):
        msg = f"Attempting to permute a tensor of rank {a.ndim}, but received a permutation of length {len(permutation)}!"
        raise ValueError(msg)

    # 检查排列是否有效
    if not utils.is_valid_permutation(a.ndim, permutation):
        msg = f"Received an invalid permutation, {permutation}!"
        raise ValueError(msg)

    # 构建新的张量形状和步长
    new_shape = [0] * a.ndim
    new_strides = [0] * a.ndim
    for idx, dim in enumerate(permutation):
        new_shape[idx] = a.shape[dim]
        new_strides[idx] = a.stride()[dim]

    # 返回按指定排列重排后的张量视图
    return a.as_strided(tuple(new_shape), tuple(new_strides), a.storage_offset())


# 根据指定的排列重排张量的维度
def _transpose_aten(a: Tensor, permutation: DimsSequenceType) -> Tensor:
    return torch.permute(a, permutation)


# 定义 `_transpose_meta` 函数的文档字符串，描述其功能和用法
_transpose_doc = """
    Creates a view of the tensor with its dimensions permuted.

    The length of the permutation must be the rank of the tensor,
    and each element of the permutation specifies the new order
    for the corresponding dimension.
    """

# 创建 `_transpose` 函数，将其包装为原语 `_make_prim`
transpose = _make_prim(
    schema="transpose(Tensor(a) a, int[] permutation) -> Tensor(a)",
    meta=_transpose_meta,  # 使用元数据 `_transpose_meta`
    impl_aten=_transpose_aten,  # 使用 ATen 实现 `_transpose_aten`
    return_type=RETURN_TYPE.VIEW,  # 返回视图类型的张量
    doc=_transpose_doc,  # 使用上面定义的文档字符串 `_transpose_doc`
)
# 定义一个函数，返回给定张量的一个视图
def _view_of_meta(a: TensorLikeType) -> TensorLikeType:
    return a.as_strided(a.shape, a.stride(), a.storage_offset())

# 定义一个函数，使用`torch.view`方法返回给定张量的视图
def _view_of_aten(a: Tensor) -> Tensor:
    return a.view(a.shape)

# 定义文档字符串，描述创建张量视图的操作
_view_of_doc = """
    Creates a view of the tensor.
"""

# 创建一个基本操作，生成一个张量视图
view_of = _make_prim(
    schema="view_of(Tensor(a) a) -> Tensor(a)",
    meta=_view_of_meta,
    impl_aten=_view_of_aten,
    return_type=RETURN_TYPE.VIEW,
    doc=_view_of_doc,
)


# 定义一个函数，返回给定张量以不同数据类型的视图
def _view_element_type_meta(a: TensorLikeType, dtype: torch.dtype) -> TensorLikeType:
    return a.view(dtype)

# 定义一个函数，使用`torch.view`方法返回给定张量以不同数据类型的视图
def _view_element_type_aten(a: Tensor, dtype: torch.dtype) -> Tensor:
    return a.view(dtype)

# 定义文档字符串，描述创建具有不同数据类型的张量视图的操作
_view_element_type_doc = """
    Creates a view of the tensor with a different dtype.
"""

# 创建一个基本操作，生成一个具有不同数据类型的张量视图
view_element_type = _make_prim(
    schema="view_of_dtype(Tensor(a) a, ScalarType dtype) -> Tensor(a)",
    meta=_view_element_type_meta,
    impl_aten=_view_element_type_aten,
    return_type=RETURN_TYPE.VIEW,
    doc=_view_element_type_doc,
)

#
# Functionalized view mutations
#

# 定义一个函数，返回一个经过变异的与输入相等的张量
def _as_strided_scatter_meta(
    input: TensorLikeType,
    src: TensorLikeType,
    size: ShapeType,
    stride: StrideType,
    storage_offset: int,
) -> TensorLikeType:
    # 验证尺寸参数的有效性
    utils.validate_shape(size)
    # 验证步长参数的有效性
    utils.validate_strides(stride)

    # 计算需要的存储长度并验证输入张量的容量是否足够
    required_size = utils.compute_required_storage_length(size, stride, storage_offset)
    torch._check(
        input.numel() >= required_size,
        lambda: (
            f"as_strided_scatter: sizes {size}, strides {stride}, storage offset {storage_offset} "
            f" and itemsize {input.element_size()} requiring a storage size of "
            f"{required_size * input.element_size()} are out of bounds "
            f"for storage of size {input.numel() * input.element_size()}"
        ),
    )
    # 验证源张量的形状与指定的尺寸是否一致
    torch._check(
        utils.is_same_shape(src.shape, size),
        lambda: f"expected src to have a size equal to the slice of self. src size = {src.shape}, slice size = {size}",
    )

    # 创建一个保留步长的克隆张量
    return utils.clone_preserve_strides(input)


# 定义文档字符串，描述创建一个经过变异的张量的操作
_as_strided_scatter_doc = """
    Creates a new tensor equivalent to ``out = input.clone()`` after mutation by
    ``out.as_strided(size, stride, storage_offset).copy_(src)``.
"""

# 创建一个基本操作，生成一个经过变异的张量
as_strided_scatter = _make_prim(
    schema="as_strided_scatter(Tensor self, Tensor src, SymInt[] size, SymInt[] stride, SymInt storage_offset) -> Tensor",
    meta=_as_strided_scatter_meta,
    impl_aten=torch.as_strided_scatter,
    return_type=RETURN_TYPE.NEW,
    doc=_as_strided_scatter_doc,
)

#
# Shape operations
#

# 定义一个函数，折叠给定张量的特定维度范围
def _collapse_meta(a: Tensor, start: int, end: int) -> Tensor:
    # 针对零维张量进行特殊处理
    _validate_collapse_args(a, start, end)
    # 计算折叠后的新形状
    new_shape = _collapsed_shape(a.shape, start, end)
    # 创建一个新的空张量，形状为新形状
    return a.new_empty(new_shape)


# 定义一个函数，使用`torch.view_as`方法折叠给定张量的特定维度范围
def _collapse_aten(a: Tensor, start: int, end: int) -> Tensor:
    # 计算折叠后的新形状
    new_shape = _collapsed_shape(a.shape, start, end)
    # 创建一个新的空张量，形状为新形状
    out = a.new_empty(new_shape)
    # 使用`torch.no_grad`上下文，将输入张量复制到新张量
    with torch.no_grad():
        out.view_as(a).copy_(a)
    return out
"""
Collapse a span of neighboring dimensions into one.

See collapse_view for the corresponding view operation.
"""
collapse = _make_prim(
    schema="collapse(Tensor a, int start, int end) -> Tensor",
    meta=_collapse_meta,
    impl_aten=_collapse_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_collapse_doc,
)

# TODO: review stride logic
# NB: unlike torch.cat, this is more strict about empty tensors and dim is
# never negative
def _cat_meta(tensors: Sequence[TensorLikeType], dim: int) -> TensorLikeType:
    # Verifies same shape (except in the concat dimension)
    assert dim >= 0
    shape = tensors[0].shape
    concat_length = 0
    for tensor_idx, tensor in enumerate(tensors):
        assert len(shape) == len(tensor.shape)
        for idx, (common_length, length) in enumerate(zip(shape, tensor.shape)):
            if idx == dim:
                concat_length = concat_length + length
            else:
                torch._check(
                    length == common_length,
                    lambda: f"Sizes of tensors must match except in dimension {dim}. "
                    f"Expected {common_length} but got {length} for tensor number "
                    f"{tensor_idx} in the list",
                )

    new_shape = list(tensors[0].shape).copy()
    new_shape[dim] = concat_length
    return TensorMeta(
        tensors[0],
        shape=new_shape,
        strides=utils.make_contiguous_strides_for(new_shape),
    )


def _cat_aten(tensors: Union[Tuple[Tensor, ...], List[Tensor]], dim: int) -> Tensor:
    return torch.cat(tensors, dim)


_cat_doc = """
  Concatenates tensors along the specified dimension.

  The tensors' shapes must have the same rank and same length for other dimensions.
  """

cat = _make_prim(
    schema="cat(Tensor[] tensors, int dim) -> Tensor",
    meta=_cat_meta,
    impl_aten=_cat_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_cat_doc,
)


def _reshape_meta(a: TensorLikeType, shape: ShapeType):
    assert isinstance(a, TensorLike)
    utils.validate_shape(shape)

    # Validates the tensor and the requested shape have the
    # same number of elements
    numel = reduce(operator.mul, shape)
    if numel != a.numel():
        msg = f"Attempting to reshape a tensor with {a.numel()} elements to a shape with {numel} elements!"
        raise ValueError(msg)

    return TensorMeta(a, shape=shape, strides=utils.make_contiguous_strides_for(shape))


def _reshape_aten(a: Tensor, shape: ShapeType) -> Tensor:
    return a.reshape(shape).contiguous().clone()


_reshape_doc = """
  Creates a contiguous tensor with the specified shape
  containing a copy of the data in a.
  """
reshape = _make_prim(
    schema="reshape(Tensor a, SymInt[] shape) -> Tensor",
    meta=_reshape_meta,
    impl_aten=_reshape_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_reshape_doc,
)


def _rev_meta(a: TensorLikeType, dims: DimsSequenceType) -> TensorLikeType:
    utils.validate_dimension_indices(a.ndim, dims)
    # 返回一个和输入张量 `a` 相同大小的空张量，但保留其内存格式
    return torch.empty_like(a, memory_format=torch.preserve_format)
# _rev_doc 是用来描述函数功能的字符串，用于生成文档
_rev_doc = """
    Reverses the order of elements along the given dimensions.
    """

# 创建名为 rev 的原语函数，用于反转张量的元素顺序
rev = _make_prim(
    schema="rev(Tensor a, int[] dims) -> Tensor",  # 函数签名说明接受一个张量和一个整数数组作为参数，并返回一个张量
    meta=_rev_meta,  # 元数据，通常用于描述函数的其他特性
    impl_aten=torch.flip,  # 实现函数使用 torch.flip 进行张量元素的反转操作
    return_type=RETURN_TYPE.NEW,  # 返回类型为新创建的张量
    doc=_rev_doc,  # 函数的文档字符串，描述了函数的具体操作和用途
)

#
# Conditional prims
#

# 函数 _where_meta 用于生成条件操作的元数据
def _where_meta(
    pred: TensorLikeType, a: TensorLikeType, b: TensorLikeType
) -> TensorLikeType:
    return _prim_elementwise_meta(
        a,
        b,
        type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
        args_with_fixed_dtypes=(pred,),
    )

# _where_doc 描述了函数 where 的功能，根据条件 pred 从 a 和 b 中选择元素
_where_doc = """
  Selects elements from a and b according to pred.

  Where pred is true the result contains the element from a, and
  where pred is false the result contains the element from b.
  """

# 创建名为 where 的原语函数，用于根据条件选择张量中的元素
where = _make_prim(
    schema="where(Tensor pred, Tensor a, Tensor b) -> Tensor",  # 函数签名说明接受一个布尔张量和两个张量作为参数，并返回一个张量
    meta=_where_meta,  # 元数据，用于描述函数的其他特性
    impl_aten=torch.where,  # 实现函数使用 torch.where 进行条件选择操作
    return_type=RETURN_TYPE.NEW,  # 返回类型为新创建的张量
    doc=_where_doc,  # 函数的文档字符串，描述了函数的具体操作和用途
)

#
# Type conversions
#

# 函数 _convert_element_type_meta 用于生成类型转换操作的元数据
def _convert_element_type_meta(a: TensorLikeType, dtype: torch.dtype) -> TensorLikeType:
    # 类型检查
    assert isinstance(a, TensorLike)
    assert isinstance(dtype, torch.dtype)

    # 若输入张量是非重叠且密集的，则保留其步长
    if torch._prims_common.is_non_overlapping_and_dense(a):
        strides = a.stride()
    else:
        strides = utils.compute_elementwise_output_strides(a)

    return TensorMeta(a, strides=strides, dtype=dtype)

# 函数 _convert_element_type_aten 实现了张量类型转换操作
def _convert_element_type_aten(a: Tensor, dtype: torch.dtype) -> Tensor:
    # 在可能的情况下传播 requires_grad 属性
    if not utils.is_grad_dtype(dtype):
        requires_grad = False
    else:
        # TODO: 更新元对象以便可以直接获取此属性
        try:
            requires_grad = a.requires_grad
        except Exception as e:
            requires_grad = False

    # 创建一个与 a 类型相同、设备相同、dtype 为指定类型的新张量
    result = torch.empty_like(
        a, device=a.device, dtype=dtype, requires_grad=requires_grad
    )
    with torch.no_grad():
        return copy_to(result, a)

# _convert_element_type_doc 描述了函数 convert_element_type 的功能，即创建指定 dtype 的张量副本
_convert_element_type_doc = """
  Creates a copy of a tensor with the given dtype.
  """

# 创建名为 convert_element_type 的原语函数，用于执行张量类型转换操作
convert_element_type = _make_prim(
    schema="convert_element_type(Tensor a, ScalarType dtype) -> Tensor",  # 函数签名说明接受一个张量和一个标量类型作为参数，并返回一个张量
    meta=_convert_element_type_meta,  # 元数据，用于描述函数的其他特性
    impl_aten=_convert_element_type_aten,  # 实现函数使用 _convert_element_type_aten 进行类型转换操作
    return_type=RETURN_TYPE.NEW,  # 返回类型为新创建的张量
    doc=_convert_element_type_doc,  # 函数的文档字符串，描述了函数的具体操作和用途
    tags=(torch.Tag.pointwise,),  # 函数的标签，表示该函数是一个逐点操作
)

# 函数 _device_put_meta 用于生成将张量放置到指定设备的元数据
def _device_put_meta(
    a: TensorLikeType, device: Union[str, torch.device]
) -> TensorLikeType:
    assert isinstance(a, TensorLike)
    assert isinstance(device, (str, torch.device))

    return TensorMeta(a, device=utils.canonicalize_device(device))

# 函数 _device_put_aten 实现了将张量放置到指定设备的操作
def _device_put_aten(a: Tensor, device: Union[str, torch.device]) -> Tensor:
    return a.to(device)

# _device_put_doc 描述了函数 device_put 的功能，即将张量放置到指定设备上
_device_put_doc = """
  Creates a copy of a tensor on the given device.
  """

# 创建名为 device_put 的原语函数，用于将张量放置到指定设备上
device_put = _make_prim(
    schema="device_put(Tensor a, Device device) -> Tensor",  # 函数签名说明接受一个张量和一个设备作为参数，并返回一个张量
    meta=_device_put_meta,  # 元数据，用于描述函数的其他特性
    impl_aten=_device_put_aten,  # 实现函数使用 _device_put_aten 进行设备放置操作
    return_type=RETURN_TYPE.NEW,  # 返回类型为新创建的张量
    doc=_device_put_doc,  # 函数的文档字符串，描述了函数的具体操作和用途
)
    doc=_device_put_doc,


注释：


    # 将变量 _device_put_doc 赋值给变量 doc
# NOTE: need to model meta scalars
# See https://github.com/pytorch/pytorch/issues/78070
# 定义一个函数用于生成元信息，用于处理输入参数 a 的类型转换和包装成 FakeTensor 对象
def _item_meta(a: TensorLikeType) -> FakeTensor:
    # 获取输入张量的数据类型，并转换为对应的 Python 数值类型
    number_type = utils.dtype_to_type(a.dtype)
    # 创建并返回一个包含元信息的 TensorMeta 对象，其中值为 -1
    return TensorMeta(number_type(-1))


# 定义一个字符串，描述了 _item 函数的作用
_item_doc = """
    Converts a tensor with one element to a Python number.
"""

# 创建 _item 函数
# TODO: create a new return type for scalars?
# FIXME: currently returns integers for boolean tensors
# https://github.com/pytorch/pytorch/issues/78071
item = _make_prim(
    schema="item(Tensor a) -> Scalar",  # 指定函数原型
    meta=_item_meta,  # 指定元信息生成函数
    impl_aten=torch.Tensor.item,  # 指定实现函数的位置
    return_type=RETURN_TYPE.NEW,  # 指定返回类型
    doc=_item_doc,  # 指定函数文档字符串
)


# NOTE: need to model meta scalars
# See https://github.com/pytorch/pytorch/issues/78070
# 定义一个函数用于生成最大值元信息，处理输入参数 dtype 的类型转换和包装成 FakeTensor 对象
def _maximum_value_meta(dtype: torch.dtype) -> FakeTensor:
    # 获取输入数据类型，并转换为对应的 Python 数值类型
    number_type = utils.dtype_to_type(dtype)
    # 创建并返回一个包含最大值的 TensorMeta 对象，其中值为 -1
    return TensorMeta(number_type(-1))


# 定义一个函数，返回指定数据类型的最大有限值
def _maximum_value_aten(dtype: torch.dtype):
    if dtype == torch.bool:  # 如果数据类型是布尔型
        return True  # 返回 True
    elif dtype.is_complex or dtype.is_floating_point:  # 如果是复数或浮点数
        return torch.finfo(dtype).max  # 返回对应数据类型的最大值
    else:  # 其他情况
        return torch.iinfo(dtype).max  # 返回对应数据类型的最大值


# 定义一个字符串，描述了 _maximum_value 函数的作用
_maximum_value_doc = """
    Return the maximum finite value for a dtype.
"""

# 创建 _maximum_value 函数
# TODO: create a new return type for scalars?
# FIXME: currently returns integers for boolean tensors
# https://github.com/pytorch/pytorch/issues/78071
maximum_value = _make_prim(
    schema="maximum_value(ScalarType dtype) -> Scalar",  # 指定函数原型
    meta=_maximum_value_meta,  # 指定元信息生成函数
    impl_aten=_maximum_value_aten,  # 指定实现函数的位置
    return_type=RETURN_TYPE.NEW,  # 指定返回类型
    doc=_maximum_value_doc,  # 指定函数文档字符串
)


# NOTE: need to model meta scalars
# See https://github.com/pytorch/pytorch/issues/78070
# 定义一个函数用于生成最小值元信息，处理输入参数 dtype 的类型转换和包装成 FakeTensor 对象
def _minimum_value_meta(dtype: torch.dtype) -> FakeTensor:
    # 获取输入数据类型，并转换为对应的 Python 数值类型
    number_type = utils.dtype_to_type(dtype)
    # 创建并返回一个包含最小值的 TensorMeta 对象，其中值为 -1
    return TensorMeta(number_type(-1))


# 定义一个函数，返回指定数据类型的最小有限值
def _minimum_value_aten(dtype: torch.dtype):
    if dtype == torch.bool:  # 如果数据类型是布尔型
        return False  # 返回 False
    elif dtype.is_complex or dtype.is_floating_point:  # 如果是复数或浮点数
        return torch.finfo(dtype).min  # 返回对应数据类型的最小值
    else:  # 其他情况
        return torch.iinfo(dtype).min  # 返回对应数据类型的最小值


# 定义一个字符串，描述了 _minimum_value 函数的作用
_minimum_value_doc = """
    Return the minimum finite value for a dtype.
"""

# 创建 _minimum_value 函数
# TODO: create a new return type for scalars?
# FIXME: currently returns integers for boolean tensors
# https://github.com/pytorch/pytorch/issues/78071
minimum_value = _make_prim(
    schema="minimum_value(ScalarType dtype) -> Scalar",  # 指定函数原型
    meta=_minimum_value_meta,  # 指定元信息生成函数
    impl_aten=_minimum_value_aten,  # 指定实现函数的位置
    return_type=RETURN_TYPE.NEW,  # 指定返回类型
    doc=_minimum_value_doc,  # 指定函数文档字符串
)


# Inplace operators
# 原地操作符


# 定义一个函数，用于验证两个张量的数据类型是否兼容，并在安全转换时执行验证
def _copy_to_meta(a: TensorLikeType, b: TensorLikeType):
    assert isinstance(a, TensorLike)  # 断言参数 a 是 TensorLike 类型
    assert isinstance(b, TensorLike)  # 断言参数 b 是 TensorLike 类型

    # Validates the cast is safe
    # TODO: move this as an option on the reference
    # a_typ = utils.dtype_to_type(a.dtype)
    # b_typ = utils.dtype_to_type(b.dtype)
    # if a_typ is not utils.get_higher_type(a_typ, b_typ):
    #     raise RuntimeError(str(b.dtype), " can't be cast safely to ", str(a.dtype), "!")

    # Validates the tensors have the same number of elements
    # 检查张量a和张量b的元素数量是否相等
    if a.numel() != b.numel():
        # 如果张量a和张量b的元素数量不相等，生成错误消息
        msg = f"Attempting to copy {b.numel()} elements to a tensor with {a.numel()} elements!"
        # 抛出运行时错误，显示错误消息
        raise RuntimeError(msg)
    
    # 如果张量a和张量b的元素数量相等，返回张量a
    return a
# 复制 b 中的数据到 a，并返回修改后的 a 张量
def _copy_to_aten(a: Tensor, b: Tensor) -> Tensor:
    return a.copy_(b)

# 复制操作的文档字符串
_copy_to_doc = """
  Copies the data in b to a and returns the modified a.
  """

# TODO: 移除安全类型转换，并在引用上实现
# 创建 copy_to 原语
copy_to = _make_prim(
    schema="copy_to(Tensor(a!) a, Tensor b) -> Tensor(a!)",
    meta=_copy_to_meta,
    impl_aten=_copy_to_aten,
    return_type=RETURN_TYPE.INPLACE,
    doc=_copy_to_doc,
)


# 复制到新张量的元信息函数
def _copy_strided_meta(a: TensorLikeType, stride: ShapeType):
    assert isinstance(a, TensorLike)
    return torch.empty_strided(
        a.shape,
        stride,
        dtype=a.dtype,
        layout=a.layout,
        device=a.device,
        requires_grad=a.requires_grad,
    )

# 使用张量的步长进行复制的底层实现
def _copy_strided_aten(a: Tensor, stride: ShapeType) -> Tensor:
    out = torch.empty_strided(
        a.size(),
        stride=stride,
        dtype=a.dtype,
        layout=a.layout,
        device=a.device,
        requires_grad=a.requires_grad,
    )
    out.copy_(a)
    return out

# 复制到新张量操作的文档字符串
_copy_strided_doc = """
  Copies the data in a to a new tensor, the new tensor has same shape with a size, but has different stride.
  """

# 创建 copy_strided 原语
copy_strided = _make_prim(
    schema="copy_strided(Tensor a, SymInt[] stride) -> Tensor",
    meta=_copy_strided_meta,
    impl_aten=_copy_strided_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_copy_strided_doc,
)


# 调整张量形状的元信息函数
def _resize_meta(a: TensorLikeType, shape: ShapeType):
    return a.resize_(shape)

# 使用底层 ATen 实现调整张量形状的函数
def _resize_aten(a: Tensor, shape: ShapeType) -> Tensor:
    return a.resize_(shape)

# 调整张量形状操作的文档字符串
_resize_doc = """
  Gives a tensor with no elements a new shape, returning the modified tensor.

  The tensor's strides are contiguous and its values are unitialized.
  """

# TODO: 评估支持任意大小调整
# 创建 resize 原语
resize = _make_prim(
    schema="resize(Tensor(a!) a, SymInt[] shape) -> Tensor(a!)",
    meta=_resize_meta,
    impl_aten=_resize_aten,
    return_type=RETURN_TYPE.INPLACE,
    doc=_resize_doc,
)


# 单输出缩减操作的元信息函数
def _reduction_meta(inp, dims, *, output_dtype=None):
    """
    Meta function for single output reduction operations
    Stride logic is incorrect
    """
    assert isinstance(inp, TensorLike)
    if output_dtype is None:
        output_dtype = inp.dtype
    output_shape = utils.compute_reduction_output_shape(inp.shape, dims)
    return TensorMeta(
        shape=output_shape,
        strides=utils.make_contiguous_strides_for(output_shape),
        dtype=output_dtype,
        device=inp.device,
    )

# 变量缩减操作的元信息函数
def _var_reduction_meta(inp, dims, correction):
    if utils.is_complex_dtype(inp.dtype):
        output_dtype = utils.corresponding_real_dtype(inp.dtype)
    else:
        output_dtype = inp.dtype
    return _reduction_meta(inp, dims, output_dtype=output_dtype)

# 求和操作的文档字符串
_sum_doc = """
    Computes the sum of elements in the input tensor over the list of dimensions
    specified in the dim argument
    """

# 按位异或求和操作的文档字符串
_xor_sum_doc = """
    Computes the xor sum of elements in the input tensor over the list of dimensions
    specified in the dim argument
    """
_iota_doc = """
    Constructs a 1-D tensor t where ``t[i] == start + i * step``.
"""
# 定义了一个文档字符串，描述了 _iota_meta 函数的作用：创建一个从 start 开始、以 step 为步长的长度为 length 的一维张量

# TODO: layout, pin_memory, memory_format
# TODO: model requires_grad on TensorMeta
# 上面两行是注释，指出了未来需要添加的功能或改进的地方，但目前未实现

def _iota_meta(
    length: int,
    *,
    start: int,
    step: int,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
) -> TensorLikeType:
    # 创建一个一维张量，长度为 length，每个元素的值为 start + i * step
    # 其中 i 是索引，从 0 到 length-1
    # 张量的数据类型由 dtype 指定，存储设备由 device 指定
    # requires_grad 指定张量是否需要梯度计算
    # 检查给定的 dtype 是否为整数类型，如果不是抛出错误信息
    torch._check(
        utils.is_integer_dtype(dtype),
        lambda: "prims.iota only supports integer dtypes",
    )
    # 检查 step 是否为非零值，如果为零则抛出错误信息
    torch._check(step != 0, lambda: "step must be nonzero")
    # 返回一个未初始化的张量，指定长度、dtype、设备和梯度跟踪需求
    return torch.empty(
        length,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
def _iota_aten(
    length: int,
    *,
    start: int,
    step: int,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
) -> TensorLikeType:
    # 计算生成序列的结束值
    end = start + length * step
    # 使用 torch.arange 生成等差数列的张量，并指定数据类型、设备和是否需要梯度
    return torch.arange(
        start, end, step, dtype=dtype, device=device, requires_grad=requires_grad
    )


iota = _make_prim(
    schema="iota(SymInt length, *, SymInt start, SymInt step, ScalarType dtype, Device device, bool requires_grad) -> Tensor",  # noqa: B950
    return_type=RETURN_TYPE.NEW,
    meta=_iota_meta,
    impl_aten=_iota_aten,
    doc=_iota_doc,
)

# TODO: layout, pin_memory, memory_format
# TODO: model requires_grad on TensorMeta
def _empty_meta(
    shape: ShapeType, *, dtype: torch.dtype, device: torch.device, requires_grad: bool
) -> TensorLikeType:
    # 根据输入的形状生成连续的步幅
    strides = utils.make_contiguous_strides_for(shape)
    # 返回一个 TensorMeta 对象，包含指定的形状、步幅、数据类型和设备信息
    return TensorMeta(shape=shape, strides=strides, dtype=dtype, device=device)


def _empty_aten(
    shape: ShapeType, *, dtype: torch.dtype, device: torch.device, requires_grad: bool
) -> Tensor:
    # 使用 torch.empty 创建一个未初始化值的张量，并指定形状、数据类型、设备和是否需要梯度
    return torch.empty(shape, dtype=dtype, device=device, requires_grad=requires_grad)


_empty_doc = """
    Creates a tensor with uninitialized values and the specified shape, dtype, and device.
"""

empty = _make_prim(
    schema="empty(SymInt[] shape, *, ScalarType dtype, Device device, bool requires_grad) -> Tensor",
    meta=_empty_meta,
    impl_aten=_empty_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_empty_doc,
)


def _empty_strided_meta(
    shape: ShapeType,
    strides: StrideType,
    *,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
) -> TensorLikeType:
    # 返回一个 TensorMeta 对象，包含指定的形状、步幅、数据类型和设备信息
    return TensorMeta(shape=shape, strides=strides, dtype=dtype, device=device)


_empty_strided_doc = """
    Creates a tensor with uninitialized values.
"""

# TODO: add layout, pin_memory
empty_strided = _make_prim(
    schema="empty_strided(SymInt[] shape, SymInt[] strides, *, ScalarType dtype, Device device, bool requires_grad) -> Tensor",
    return_type=RETURN_TYPE.NEW,
    meta=_empty_strided_meta,
    impl_aten=torch.empty_strided,
    doc=_empty_strided_doc,
)


def _empty_permuted_meta(
    shape: ShapeType,
    physical_layout: DimsSequenceType,
    *,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
) -> TensorLikeType:
    # 根据物理布局重新生成连续的步幅
    p_strides = utils.make_contiguous_strides_for([shape[l] for l in physical_layout])
    # 检查输入形状的维度数与物理布局列表长度是否一致
    dim = len(shape)
    torch._check(
        len(physical_layout) == dim,
        lambda: (
            "Number of dimensions in the tensor input does not match the "
            f"length of the physical layout; i.e. len(size) = {dim} "
            f"is not equal to len(physical_layout) = {len(physical_layout)}"
        ),
    )
    # 初始化步幅列表
    strides = [0] * len(shape)
    # 初始化已见维度的集合
    seen_dims = set()
    # 遍历物理布局列表中的元素和索引
    for p, l in enumerate(physical_layout):
        # 调用 torch._check 函数，检查维度 l 是否在合理范围内 [0, dim)
        torch._check(
            0 <= l < dim,
            lambda: (
                f"Dimension out of range (expected to be between 0 and {dim - 1}, but got "
                f"{l} at index {p}).  NB: negative dims "
                "not currently supported; file an issue if you want it."
            ),
        )
        # 调用 torch._check 函数，检查维度 l 是否已经在 seen_dims 集合中存在，避免重复
        torch._check(l not in seen_dims, lambda: "Duplicate dim not allowed")
        # 将物理布局中的步幅信息 p_strides[p] 分配给 strides 字典的维度 l
        strides[l] = p_strides[p]
        # 将维度 l 添加到 seen_dims 集合中，标记为已见
        seen_dims.add(l)
    # 返回一个 TensorMeta 对象，包含形状 shape、步幅 strides、数据类型 dtype、设备信息 device
    return TensorMeta(
        shape=shape,
        strides=strides,
        dtype=dtype,
        device=device,
    )
# 空字符串变量，包含一个描述性文档字符串，说明了创建一个张量的方法，其物理布局非重叠且密集。
_empty_permuted_doc = """
    Creates a tensor with uninitialized values according to some physical layout,
    that is guaranteed to be non-overlapping and dense.
"""

# TODO: add layout, pin_memory
# 创建一个名为 `empty_permuted` 的原语函数，根据指定的 schema、meta、impl_aten 和文档字符串定义。
empty_permuted = _make_prim(
    schema="empty_permuted(SymInt[] shape, int[] physical_layout, *, ScalarType dtype, Device device, bool requires_grad) -> Tensor",  # noqa: B950
    return_type=RETURN_TYPE.NEW,
    meta=_empty_permuted_meta,
    impl_aten=torch.empty_permuted,
    doc=_empty_permuted_doc,
)


def _full_meta(
    shape: ShapeType,
    fill_value: NumberType,
    *,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
) -> TensorLikeType:
    # 根据给定的形状计算连续的步幅，返回张量的元数据。
    strides = utils.make_contiguous_strides_for(shape)
    return TensorMeta(shape=shape, strides=strides, dtype=dtype, device=device)


def _full_aten(
    shape: ShapeType,
    fill_value: NumberType,
    *,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
) -> Tensor:
    # 注意，Mypy 认为 torch.full 不能接受复数填充值。
    # 使用 torch.full 创建一个张量，填充给定的值，并指定形状、dtype、设备和梯度属性。
    return torch.full(
        shape, fill_value, dtype=dtype, device=device, requires_grad=requires_grad  # type: ignore[arg-type]
    )


_full_doc = """
    Creates a tensor filled with the given fill value, and with the specified shape, dtype, and device.
"""

# TODO: add layout
# 创建一个名为 `full` 的原语函数，根据指定的 schema、meta、impl_aten 和文档字符串定义。
full = _make_prim(
    schema="full(SymInt[] shape, Scalar fill_value, *, ScalarType dtype, Device device, bool requires_grad) -> Tensor",
    meta=_full_meta,
    impl_aten=_full_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_full_doc,
)


def _full_like_meta(
    a: TensorLikeType,
    fill_value: NumberType,
    *,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
) -> TensorLikeType:
    # 根据输入张量 `a` 计算逐元素输出的步幅，返回填充值、dtype 和设备相同的张量元数据。
    strides = utils.compute_elementwise_output_strides(a)
    if a.numel() == 0:
        strides = a.stride()

    return TensorMeta(a, strides=strides, dtype=dtype, device=device)


def _full_like_aten(
    a: Tensor,
    fill_value: NumberType,
    *,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
) -> Tensor:
    # 注意，Mypy 认为 torch.full_like 不能接受复数填充值。
    # 使用 torch.full_like 创建一个形状、dtype 和设备与给定张量 `a` 相同的张量，填充给定的值。
    return torch.full_like(
        a, fill_value, dtype=dtype, device=device, requires_grad=requires_grad  # type: ignore[arg-type]
    )


_full_like_doc = """
    Creates a tensor filled with the given fill value, and the same shape, dtype, and device as the
    given tensor by default. The dtype and device settings can be overridden
    by specifying them explicitly.
"""

# 创建一个名为 `full_like` 的原语函数，根据指定的 schema、meta、impl_aten 和文档字符串定义。
full_like = _make_prim(
    schema="full_like(Tensor a, Scalar fill_value, *, ScalarType dtype, Device device, bool requires_grad) -> Tensor",
    meta=_full_like_meta,
    impl_aten=_full_like_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_full_like_doc,
)


def _scalar_tensor_meta(
    scalar: NumberType,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> TensorLikeType:
    # 初始化一个空形状的张量。
    shape: ShapeType = []
    # 使用给定的形状生成连续的步幅
    strides = utils.make_contiguous_strides_for(shape)
    # 创建一个TensorMeta对象，包括标量、形状、步幅、数据类型和设备信息，并返回
    return TensorMeta(scalar, shape=shape, strides=strides, dtype=dtype, device=device)
# 将标量包装成具有指定数据类型和设备的张量
def _scalar_tensor_aten(
    scalar: NumberType,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    # 如果标量是复数且数据类型不是复数类型，引发类型错误异常
    if isinstance(scalar, complex) and (
        dtype is None or not utils.is_complex_dtype(dtype)
    ):
        raise TypeError("Complex scalar requires complex tensor dtype.")
    # 创建一个 torch 标量张量，类型为指定的 dtype 和设备为指定的 device
    return torch.scalar_tensor(scalar, dtype=dtype, device=device)  # type: ignore[arg-type]


_scalar_tensor_doc = """
    Wraps a Number into a Tensor with the specified dtype and device.
"""

# TODO: add layout and pin_memory support
# 将 _scalar_tensor_aten 函数封装为原语，并添加文档和元信息
scalar_tensor = _make_prim(
    schema="scalar_tensor(Scalar s, *, ScalarType? dtype=None, Device? device=None) -> Tensor",
    meta=_scalar_tensor_meta,
    impl_aten=_scalar_tensor_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_scalar_tensor_doc,
)


#
# Linear algebra (linalg) prims
#


def _svd_meta(
    A: TensorLikeType, *, full_matrices: bool
) -> Tuple[TensorLikeType, TensorLikeType, TensorLikeType]:
    # 检查 A 是否是矩阵，并限制 A 的数据类型为浮点数或复数
    utils.check_is_matrix(A, "linalg.svd")
    utils.check_fp_or_complex(A.dtype, "linalg.svd", allow_low_precision_dtypes=False)

    # 获取 A 的形状信息
    A_shape = A.shape
    batch = A_shape[:-2]
    m, n = A_shape[-2:]
    k = min(m, n)

    # 计算 U 的形状和步长信息
    shape_U = batch + (m, m if full_matrices else k)
    strides_U = utils.make_contiguous_strides_for(shape_U, row_major=False)
    # 创建 U 的元信息对象
    U = TensorMeta(shape=shape_U, strides=strides_U, dtype=A.dtype, device=A.device)

    # 计算 S 的形状和步长信息
    shape_S = batch + (k,)
    strides_S = utils.make_contiguous_strides_for(shape_S)
    # 创建 S 的元信息对象
    S = TensorMeta(
        shape=shape_S,
        strides=strides_S,
        dtype=utils.corresponding_real_dtype(A.dtype) if A.is_complex() else A.dtype,
        device=A.device,
    )

    # 计算 Vh 的形状和步长信息
    shape_Vh = batch + (n if full_matrices else k, n)
    # 根据设备类型确定 Vh 的行优先或列优先顺序
    is_cuda = A.device.type == "cuda"
    strides_Vh = utils.make_contiguous_strides_for(shape_Vh, row_major=is_cuda)
    # 创建 Vh 的元信息对象
    Vh = TensorMeta(shape=shape_Vh, strides=strides_Vh, dtype=A.dtype, device=A.device)
    # 如果 A 不为空且 Vh 是复数类型且 CUDA 可用，则对 Vh 进行共轭操作
    if A.numel() != 0 and Vh.is_complex() and torch.cuda.is_available():
        Vh = Vh.conj()
    return U, S, Vh


def _svd_aten(
    A: TensorLikeType, *, full_matrices: bool
) -> Tuple[Tensor, Tensor, Tensor]:
    # 调用 torch.linalg.svd 函数计算矩阵 A 的奇异值分解
    return torch.linalg.svd(A, full_matrices=full_matrices)


_svd_doc = """
    Returns the SVD of a matrix or batch of matrices.

    The `full_matrices` flag controls whether the full or reduced SVD decomposition is returned.
"""

# 将 _svd_aten 函数封装为原语，并添加文档和元信息
svd = _make_prim(
    schema="svd(Tensor A, *, bool full_matrices) -> (Tensor U, Tensor S, Tensor Vh)",
    meta=_svd_meta,
    impl_aten=_svd_aten,
    return_type=(RETURN_TYPE.NEW, RETURN_TYPE.NEW, RETURN_TYPE.NEW),
    doc=_svd_doc,
)


#
# Randomness Prims
#


def _normal_meta(
    shape: ShapeType,
    # 定义一个参数 shape，类型为 ShapeType，表示张量的形状

    *,
    # * 表示这之后的参数必须使用关键字传递，不能位置传递

    mean: Union[float, complex],
    # 定义一个参数 mean，可以是 float 或 complex 类型，表示张量的均值

    std: float,
    # 定义一个参数 std，类型为 float，表示张量的标准差

    dtype: torch.dtype,
    # 定义一个参数 dtype，类型为 torch.dtype，表示张量的数据类型

    device: torch.device,
    # 定义一个参数 device，类型为 torch.device，表示张量所在的设备

    requires_grad: bool,
    # 定义一个参数 requires_grad，类型为 bool，表示张量是否需要梯度计算

    generator: Optional[torch.Generator] = None,
    # 定义一个可选参数 generator，类型为 torch.Generator 或 NoneType，默认为 None，
    # 表示生成张量的随机数生成器，如果不提供则不使用随机数生成器
# 确保标准差为非负数
torch._check(
    std >= 0.0,
    lambda: f"expected non-negative standard deviation, but got std={std}",
)

# 确保数据类型为浮点数或复数类型
torch._check(
    utils.is_float_dtype(dtype) or utils.is_complex_dtype(dtype),
    lambda: f"expected a floating-point or complex dtype, but got dtype={dtype}",
)

# 生成连续的步幅用于给定形状
strides = utils.make_contiguous_strides_for(shape)

# 创建一个 TensorMeta 对象，表示具有指定形状、步幅、数据类型和设备的张量元信息
return TensorMeta(shape=shape, strides=strides, dtype=dtype, device=device)


def _normal_aten(
    shape: ShapeType,
    *,
    mean: Union[float, complex],
    std: float,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    # 创建一个未初始化的张量，具有指定形状、数据类型和设备，并指定是否需要梯度
    a = torch.empty(shape, dtype=dtype, device=device, requires_grad=requires_grad)
    with torch.no_grad():
        # 使用正态分布填充张量。注意，此处的 mean 可能是 float 或 complex 类型，但其类型注解不正确
        a.normal_(mean, std, generator=generator)  # type: ignore[arg-type]
    return a


_normal_doc = """
    构造一个张量，其中值来自具有指定均值和标准差的正态分布。

    仅支持浮点数类型。
"""

# 创建 normal 函数的原语
normal = _make_prim(
    schema=(
        "normal(SymInt[] shape, *, Scalar mean, Scalar std, ScalarType dtype, Device device, bool requires_grad, Generator? generator=None) -> Tensor"  # noqa: B950
    ),
    return_type=RETURN_TYPE.NEW,
    meta=_normal_meta,
    impl_aten=_normal_aten,
    doc=_normal_doc,
)


def _uniform_meta(
    shape: ShapeType,
    *,
    low: float,
    high: float,
    dtype: torch.dtype,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> TensorLikeType:
    # 生成连续的步幅用于给定形状
    strides = utils.make_contiguous_strides_for(shape)
    return TensorMeta(shape=shape, strides=strides, dtype=dtype, device=device)


def _uniform_aten(
    shape: ShapeType,
    *,
    low: float,
    high: float,
    dtype: torch.dtype,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    # 创建一个未初始化的张量，具有指定形状、数据类型和设备
    a = torch.empty(shape, dtype=dtype, device=device)
    # 使用均匀分布填充张量
    a.uniform_(low, high, generator=generator)
    return a


_uniform_doc = """
    构造一个张量，其中值从 low 到 high 均匀抽样。
"""

# 创建 uniform 函数的原语
_uniform_helper = _make_prim(
    schema=(
        "uniform(SymInt[] shape, *, Scalar low, Scalar high, ScalarType dtype, Device device, Generator? generator=None) -> Tensor"
    ),
    return_type=RETURN_TYPE.NEW,
    meta=_uniform_meta,
    impl_aten=_uniform_aten,
    doc=_uniform_doc,
)


#
# FFT prims
#


def _fft_r2c_meta(
    input: TensorLike,
    *,
    dim: DimsSequenceType,
    onesided: bool,
) -> TensorLikeType:
    # 规范化维度参数，确保它们有效
    dim = utils.canonicalize_dims(input.ndim, dim)
    utils.validate_no_repeating_dims(dim)

    shape = list(input.shape)
    # 如果是单边 FFT，调整输出张量的形状
    if onesided:
        last_dim = dim[-1]
        shape[last_dim] = shape[last_dim] // 2 + 1
    # 根据输入数据的dtype找到对应的复杂数据类型
    dtype = utils.corresponding_complex_dtype(input.dtype)
    
    # 根据给定的形状创建连续的步幅
    strides = utils.make_contiguous_strides_for(shape)
    
    # 返回一个TensorMeta对象，其中包括指定的形状、步幅、数据类型和设备信息
    return TensorMeta(shape=shape, strides=strides, dtype=dtype, device=input.device)
def _fft_r2c_aten(
    input: TensorLike,
    *,
    dim: DimsSequenceType,
    onesided: bool,
) -> TensorLikeType:
    normalization = 0  # No normalization
    return torch._fft_r2c(input, dim, normalization, onesided)



_fft_r2c_doc = """
    Performs a real to complex Fast Fourier Transform
"""

fft_r2c = _make_prim(
    schema="fft_r2c(Tensor self, *, int[] dim, bool onesided) -> Tensor",
    meta=_fft_r2c_meta,
    impl_aten=_fft_r2c_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_fft_r2c_doc,
)



def _fft_c2c_meta(
    input: TensorLike,
    *,
    dim: DimsSequenceType,
    forward: bool,
) -> TensorLikeType:
    dim = utils.canonicalize_dims(input.ndim, dim)
    utils.validate_no_repeating_dims(dim)

    shape = input.shape
    strides = utils.make_contiguous_strides_for(shape)
    return TensorMeta(
        shape=shape, strides=strides, dtype=input.dtype, device=input.device
    )



def _fft_c2c_aten(
    input: TensorLike,
    *,
    dim: DimsSequenceType,
    forward: bool,
) -> TensorLikeType:
    normalization = 0  # No normalization
    return torch._fft_c2c(input, dim, normalization, forward)



_fft_c2c_doc = """
    Performs either a Fast Fourier Transform, or its inverse
"""

fft_c2c = _make_prim(
    schema="fft_c2c(Tensor self, *, int[] dim, bool forward) -> Tensor",
    meta=_fft_c2c_meta,
    impl_aten=_fft_c2c_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_fft_c2c_doc,
)



def _fft_c2r_meta(
    input: TensorLike,
    *,
    dim: DimsSequenceType,
    last_dim_size: int,
) -> TensorLikeType:
    dim = utils.canonicalize_dims(input.ndim, dim)
    utils.validate_no_repeating_dims(dim)

    shape = list(input.shape)
    shape[dim[-1]] = last_dim_size
    dtype = utils.corresponding_real_dtype(input.dtype)
    strides = utils.make_contiguous_strides_for(shape)
    return TensorMeta(shape=shape, strides=strides, dtype=dtype, device=input.device)



def _fft_c2r_aten(
    input: TensorLike,
    *,
    dim: DimsSequenceType,
    last_dim_size: int,
) -> TensorLikeType:
    normalization = 0  # No normalization
    return torch._fft_c2r(input, dim, normalization, last_dim_size)



_fft_c2r_doc = """
    Performs a complex to real Inverse Fast Fourier Transform
"""

fft_c2r = _make_prim(
    schema="fft_c2r(Tensor self, *, int[] dim, SymInt last_dim_size) -> Tensor",
    meta=_fft_c2r_meta,
    impl_aten=_fft_c2r_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_fft_c2r_doc,
)



def _frexp_meta(self: TensorLikeType) -> Tuple[TensorLikeType, TensorLikeType]:
    torch._check(
        self.dtype.is_floating_point,
        lambda: "torch.frexp() only supports floating-point dtypes",
    )
    return torch.empty_like(self), torch.empty_like(self, dtype=torch.int32)



frexp = _make_prim(
    schema="frexp(Tensor self) -> (Tensor mantissa, Tensor exponent)",
    meta=_frexp_meta,
    return_type=(RETURN_TYPE.NEW, RETURN_TYPE.NEW),
    impl_aten=torch.frexp,
    doc="",
)



def _make_token_aten() -> TensorLikeType:
    # 返回一个空的 PyTorch 张量，形状为 0 维度
    return torch.empty(0)
# 创建一个名为 _make_token 的全局变量，其值是 _make_prim 函数的返回结果
_make_token = _make_prim(
    schema="_make_token() -> Tensor",  # 定义 _make_token 函数的签名
    meta=_make_token_aten,  # 使用 _make_token_aten 函数作为元信息
    return_type=RETURN_TYPE.NEW,  # 指定函数返回类型为 NEW
    impl_aten=_make_token_aten,  # 指定实现函数为 _make_token_aten
    doc="Creates a token used for keeping track of side effects."  # 函数的文档字符串，描述其作用
)


def _sink_tokens_aten(tokens) -> None:
    pass  # 空函数，未实现具体功能


# 创建一个名为 _sink_tokens 的全局变量，其值是 _make_prim 函数的返回结果
_sink_tokens = _make_prim(
    schema="_sink_tokens(Tensor[] tokens) -> ()",  # 定义 _sink_tokens 函数的签名
    meta=_sink_tokens_aten,  # 使用 _sink_tokens_aten 函数作为元信息
    return_type=RETURN_TYPE.NONE,  # 指定函数返回类型为 NONE
    impl_aten=_sink_tokens_aten,  # 指定实现函数为 _sink_tokens_aten
    doc="Sink all of the tokens which were previously used for keeping track of side effects."  # 函数的文档字符串，描述其作用
)


# 调用 register_rng_prims() 函数，注册随机数生成相关的原语
register_rng_prims()

# 调用 register_debug_prims() 函数，注册调试相关的原语
register_debug_prims()
```