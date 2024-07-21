# `.\pytorch\torch\_refs\linalg\__init__.py`

```
# mypy: allow-untyped-defs
# 从 functools 模块导入 partial 函数
from functools import partial

# 从 typing 模块导入 List, Optional, Tuple, Union 类型
from typing import List, Optional, Tuple, Union

# 导入 torch 库
import torch

# 导入 torch._prims 模块中的 prims 对象
import torch._prims as prims

# 导入 torch._prims_common 模块中的 utils 对象
import torch._prims_common as utils

# 导入 torch._refs 模块中的 refs 对象
import torch._refs as refs

# 导入 torch._refs.linalg 模块中的 linalg 对象
import torch._refs.linalg as linalg

# 从 torch 模块中导入 Tensor 类型
from torch import Tensor

# 从 torch._prims_common 模块中导入以下函数和类
from torch._prims_common import (
    check_fp_or_complex,
    check_is_matrix,
    Dim,
    DimsType,
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    IntLike,
    NumberType,
    TensorLikeType,
)

# 从 torch._prims_common.wrappers 模块中导入以下函数
from torch._prims_common.wrappers import (
    _maybe_convert_to_dtype,
    elementwise_type_promotion_wrapper,
    out_wrapper,
)

# 声明 __all__ 列表，指定模块中公开的函数和类名
__all__ = [
    "diagonal",
    "matrix_norm",
    "norm",
    "svd",
    "svdvals",
    "vector_norm",
    "vecdot",
    "cross",
]

# 定义 _check_norm_dtype 函数，用于检查 linalg.*norm 函数中的 dtype 参数
def _check_norm_dtype(dtype: Optional[torch.dtype], x_dtype: torch.dtype, fn_name: str):
    """
    Checks related to the dtype kwarg in `linalg.*norm` functions
    """
    # 如果 dtype 不为 None，则进行以下检查
    if dtype is not None:
        # 检查 dtype 是否为浮点型或复数型
        torch._check(
            utils.is_float_dtype(dtype) or utils.is_complex_dtype(dtype),
            lambda: f"{fn_name}: dtype should be floating point or complex. Got {dtype}",
        )
        # 检查输入 x 的 dtype 是否与指定的 dtype 一致（如果 x 是复数，则 dtype 也应为复数）
        torch._check(
            utils.is_complex_dtype(dtype) == utils.is_complex_dtype(x_dtype),
            lambda: "{fn_name}: dtype should be {d} for {d} inputs. Got {dtype}".format(
                fn_name=fn_name,
                d="complex" if utils.is_complex_dtype(x_dtype) else "real",
                dtype=dtype,
            ),
        )
        # 检查是否可以将输入 x 的 dtype 转换为指定的 dtype，而不会损失精度
        torch._check(
            utils.get_higher_dtype(dtype, x_dtype) == dtype,
            lambda: f"{fn_name}: the dtype of the input ({x_dtype}) should be convertible "
            "without narrowing to the specified dtype ({dtype})",
        )

# 导入 operator 模块

# 从 torch._decomp 模块中注册 register_decomposition 函数
from torch._decomp import register_decomposition

# 从 torch._decomp.decompositions 模块中导入 pw_cast_for_opmath 函数
from torch._decomp.decompositions import pw_cast_for_opmath

# 定义 cross 函数，计算两个张量的叉乘
@register_decomposition(torch._ops.ops.aten.linalg_cross)
@out_wrapper()
@pw_cast_for_opmath
def cross(a: Tensor, b: Tensor, dim: int = -1):
    # 检查输入张量 a 和 b 的维度是否相同
    torch._check(
        a.ndim == b.ndim,
        lambda: "linalg.cross: inputs must have the same number of dimensions.",
    )
    # 检查指定维度 dim 上，张量 a 和 b 的长度是否为 3
    torch._check(
        a.size(dim) == 3 and b.size(dim) == 3,
        lambda: f"linalg.cross: inputs dim {dim} must have length 3, got {a.size(dim)} and {b.size(dim)}",
    )
    # 广播张量 a 和 b，使其具有相同的形状
    a, b = torch.broadcast_tensors(a, b)
    # 规范化维度 dim 的值
    dim = utils.canonicalize_dim(a.ndim, dim)
    # 创建索引数组 idx，长度为 3
    idx = torch.arange(3, device=a.device)
    # 计算叉乘结果并返回
    return a.index_select(dim, (idx + 1) % 3) * b.index_select(
        dim, (idx + 2) % 3
    ) - a.index_select(dim, (idx + 2) % 3) * b.index_select(dim, (idx + 1) % 3)

# 定义 diagonal 函数，返回输入张量的对角线元素或指定对角线的元素
def diagonal(
    input: TensorLikeType,
    *,
    offset: int = 0,
    dim1: int = -2,
    dim2: int = -1,
) -> TensorLikeType:
    return torch.diagonal(input, offset=offset, dim1=dim1, dim2=dim2)

# 定义 vector_norm 函数，计算输入张量的向量范数
@register_decomposition(torch._ops.ops.aten.linalg_vector_norm)
@out_wrapper(exact_dtype=True)
def vector_norm(
    x: TensorLikeType,
    # ord参数的类型注释，可以是float或int，初始值设为2
    ord: Union[float, int] = 2,
    # dim参数的类型注释，可以是DimsType类型或None，默认为None
    dim: Optional[DimsType] = None,
    # keepdim参数的类型注释，布尔类型，默认为False，表示是否保持维度
    keepdim: bool = False,
    # dtype参数是一个命名关键字参数，用星号(*)标记，表示其后面的参数必须以关键字方式传入
    # dtype参数的类型注释，可以是torch.dtype类型或None，默认为None
    dtype: Optional[torch.dtype] = None,
# 返回值类型注解，表示函数返回一个 Tensor 类型的对象
) -> Tensor:
    # 导入符号形状的保护函数，用于符号形状处理
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

    # 检查输入张量 x 的数据类型，要求是浮点数或复数类型
    check_fp_or_complex(x.dtype, "linalg.vector_norm")

    # 如果 dim 是 Dim 类型的实例，则转换为列表类型，忽略类型检查
    if isinstance(dim, Dim):
        dim = [dim]  # type: ignore[assignment]

    # 如果张量 x 的元素数量为 0，并且 ord 小于 0 或者 ord 为无穷大
    if guard_size_oblivious(x.numel() == 0) and (ord < 0.0 or ord == float("inf")):
        # 断言 dim 不为 None 且长度不为 0
        torch._check(
            dim is not None and len(dim) != 0,
            lambda: f"linalg.vector_norm cannot compute the {ord} norm on an empty tensor "
            "because the operation does not have an identity",
        )
        # 获取张量 x 的形状
        shape = x.shape
        # 断言 dim 不为 None
        assert dim is not None  # mypy does not seem to be able to see through check?
        # 遍历 dim 中的维度
        for d in dim:
            # 检查张量 x 在维度 d 上的形状是否不为 0
            torch._check(
                shape[d] != 0,
                lambda: f"linalg.vector_norm cannot compute the {ord} norm on the "
                f"dimension {d} because this dimension is empty and the "
                "operation does not have an identity",
            )

    # 检查规范化操作的数据类型是否符合要求
    _check_norm_dtype(dtype, x.dtype, "linalg.vector_norm")

    # 计算类型和结果类型，用于后续的规约操作
    computation_dtype, result_dtype = utils.reduction_dtypes(
        x, utils.REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT, dtype
    )

    # partial 函数，将 _maybe_convert_to_dtype 函数的 dtype 参数固定为 result_dtype
    to_result_dtype = partial(_maybe_convert_to_dtype, dtype=result_dtype)

    # 实现部分
    # 如果 ord 等于 0.0，则返回张量 x 中非零元素的数量，按 dim 进行求和
    if ord == 0.0:
        return torch.sum(torch.ne(x, 0.0), dim=dim, keepdim=keepdim, dtype=result_dtype)
    # 如果 ord 等于无穷大，则返回张量 x 的绝对值的最大值，按 dim 进行求解
    elif ord == float("inf"):
        return to_result_dtype(torch.amax(torch.abs(x), dim=dim, keepdim=keepdim))  # type: ignore[return-value,arg-type]
    # 如果 ord 等于负无穷，则返回张量 x 的绝对值的最小值，按 dim 进行求解
    elif ord == float("-inf"):
        return to_result_dtype(torch.amin(torch.abs(x), dim=dim, keepdim=keepdim))  # type: ignore[return-value,arg-type]
    else:
        # 如果 ord 不是特殊值，则需要根据计算数据类型处理张量 x
        x = _maybe_convert_to_dtype(x, computation_dtype)  # type: ignore[assignment]

        # partial 函数，固定了 torch.sum 的 dim 和 keepdim 参数
        reduce_sum = partial(torch.sum, dim=dim, keepdim=keepdim)

        # 判断 ord 是否为偶数，如果是偶数且 x 的数据类型为浮点数，则对 x 取绝对值
        is_ord_even = ord % 2 == 0 if isinstance(ord, IntLike) else ord % 2.0 == 0.0
        if not (is_ord_even and utils.is_float_dtype(x.dtype)):
            x = torch.abs(x)

        # 对规约求幂和开方的操作，最终返回规约后的结果，使用 to_result_dtype 转换数据类型
        return to_result_dtype(torch.pow(reduce_sum(torch.pow(x, ord)), 1.0 / ord))  # type: ignore[return-value]


# 辅助函数，用于 matrix_norm 函数中，计算将两个给定维度移至末尾的排列
def _backshift_permutation(dim0, dim1, ndim):
    # 返回排列列表，排列中排除 dim0 和 dim1，将它们放置在末尾
    ret = [i for i in range(ndim) if i != dim0 and i != dim1]
    ret.extend((dim0, dim1))
    return ret


# 给定一个排列，返回其逆序排列，相当于对数组进行 argsort 操作
def _inverse_permutation(perm):
    return [i for i, j in sorted(enumerate(perm), key=operator.itemgetter(1))]


# CompositeImplicitAutograd
# 矩阵范数的计算函数装饰器，确保精确的数据类型匹配
@out_wrapper(exact_dtype=True)
def matrix_norm(
    A: TensorLikeType,
    ord: Union[float, str] = "fro",
    dim: DimsType = (-2, -1),
    keepdim: bool = False,
    *,
    dtype: Optional[torch.dtype] = None,
) -> TensorLikeType:
    # shape
    # 检查输入矩阵 A 是否符合矩阵标准，用于 "linalg.matrix_norm"
    check_is_matrix(A, "linalg.matrix_norm")
    # 将 dim 规范化为二维元组
    dim = utils.canonicalize_dims(A.ndim, dim)
    if isinstance(dim, Dim):
        dim = (dim,)  # type: ignore[assignment]
    # 检查 dim 是否为二维元组，否则报错
    torch._check(
        len(dim) == 2, lambda: "linalg.matrix_norm: dim must be a 2-tuple. Got {dim}"
    )
    # 检查 dim 中的维度是否不同，否则报错
    torch._check(
        dim[0] != dim[1],
        lambda: "linalg.matrix_norm: dims must be different. Got ({dim[0]}, {dim[1]})",
    )
    # 检查 dtype 参数的合法性
    _check_norm_dtype(dtype, A.dtype, "linalg.matrix_norm")

    if isinstance(ord, str):
        # 检查 ord 是否为 "fro" 或 "nuc"，否则报错
        torch._check(
            ord in ("fro", "nuc"),
            lambda: "linalg.matrix_norm: Order {ord} not supported.",
        )
        # 检查矩阵数据类型的合法性
        check_fp_or_complex(
            A.dtype, "linalg.matrix_norm", allow_low_precision_dtypes=ord != "nuc"
        )

        if ord == "fro":
            # 返回 Frobenius 范数
            return vector_norm(A, 2, dim, keepdim, dtype=dtype)
        else:  # ord == "nuc"
            if dtype is not None:
                # 将 A 转换为指定的数据类型
                A = _maybe_convert_to_dtype(A, dtype)  # type: ignore[assignment]
            # 计算核范数
            perm = _backshift_permutation(dim[0], dim[1], A.ndim)
            result = torch.sum(svdvals(prims.transpose(A, perm)), -1, keepdim)
            if keepdim:
                inv_perm = _inverse_permutation(perm)
                result = prims.transpose(torch.unsqueeze(result, -1), inv_perm)
            return result
    else:
        # 计算绝对值后的 ord
        abs_ord = abs(ord)
        # 检查 ord 是否为 2, 1, 或者无穷大，否则报错
        torch._check(
            abs_ord in (2, 1, float("inf")),
            lambda: "linalg.matrix_norm: Order {ord} not supported.",
        )
        # 检查矩阵数据类型的合法性
        check_fp_or_complex(
            A.dtype, "linalg.matrix_norm", allow_low_precision_dtypes=ord != 2
        )

        # 根据 abs_ord 的值选择最大或最小值的计算方式
        max_min = partial(torch.amax if ord > 0.0 else torch.amin, keepdim=keepdim)

        if abs_ord == 2.0:
            if dtype is not None:
                # 将 A 转换为指定的数据类型
                A = _maybe_convert_to_dtype(A, dtype)  # type: ignore[assignment]
            # 计算二范数
            perm = _backshift_permutation(dim[0], dim[1], A.ndim)
            result = max_min(svdvals(prims.transpose(A, perm)), dim=-1)
            if keepdim:
                inv_perm = _inverse_permutation(perm)
                result = prims.transpose(torch.unsqueeze(result, -1), inv_perm)
            return result
        else:  # abs_ord == 1, -1, inf, -inf
            dim0, dim1 = dim
            if abs_ord == float("inf"):
                dim0, dim1 = dim1, dim0
            if not keepdim and (dim0 < dim1):
                dim1 -= 1
            # 返回 max_min 函数应用于一范数的结果
            return max_min(
                vector_norm(A, 1.0, dim=dim0, keepdim=keepdim, dtype=dtype), dim1
            )
# CompositeImplicitAutograd
@out_wrapper(exact_dtype=True)
# 定义了一个函数 `norm`，用于计算张量的范数
def norm(
    A: TensorLikeType,
    ord: Optional[Union[float, str]] = None,
    dim: Optional[DimsType] = None,
    keepdim: bool = False,
    *,
    dtype: Optional[torch.dtype] = None,
) -> TensorLikeType:
    # 如果指定了 `dim` 参数
    if dim is not None:
        # 如果 `dim` 是 `Dim` 类型的实例，将其转换成元组
        if isinstance(dim, Dim):
            dim = (dim,)  # type: ignore[assignment]
        # 检查 `dim` 的长度是否为 1 或 2
        torch._check(
            len(dim) in (1, 2),
            lambda: "linalg.norm: If dim is specified, it must be of length 1 or 2. Got {dim}",
        )
    # 如果未指定 `dim` 参数但指定了 `ord` 参数
    elif ord is not None:
        # 检查输入张量 `A` 的维度是否为 1 或 2
        torch._check(
            A.ndim in (1, 2),
            lambda: "linalg.norm: If dim is not specified but ord is, the input must be 1D or 2D. Got {A.ndim}D",
        )

    # 如果指定了 `ord` 参数并且满足条件（`dim` 为 2 或者未指定 `dim` 但 `A` 的维度为 2）
    if ord is not None and (
        (dim is not None and len(dim) == 2) or (dim is None and A.ndim == 2)
    ):
        # 如果未指定 `dim`，则默认为 (0, 1)，计算矩阵的范数
        if dim is None:
            dim = (0, 1)
        return matrix_norm(A, ord, dim, keepdim, dtype=dtype)
    else:
        # 如果未指定 `ord` 参数，则默认为 2.0，计算向量的范数
        if ord is None:
            ord = 2.0
        return vector_norm(A, ord, dim, keepdim, dtype=dtype)


# CompositeImplicitAutograd
@out_wrapper("U", "S", "Vh", exact_dtype=True)
# 定义了一个函数 `svd`，用于计算张量的奇异值分解（SVD）
def svd(A: TensorLikeType, full_matrices: bool = True) -> Tuple[Tensor, Tensor, Tensor]:
    # 调用底层的 `prims.svd` 函数进行奇异值分解，并返回结果
    return prims.svd(A, full_matrices=full_matrices)


# CompositeImplicitAutograd
@out_wrapper(exact_dtype=True)
# 定义了一个函数 `svdvals`，用于计算张量的奇异值
def svdvals(A: TensorLikeType) -> Tensor:
    # 调用 `svd` 函数进行奇异值分解，并返回其中的奇异值部分
    return svd(A, full_matrices=False)[1]


# CompositeImplicitAutograd
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("x", "y"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
# 定义了一个函数 `vecdot`，用于计算两个张量的逐元素乘积之和
def vecdot(x: Tensor, y: Tensor, dim: int = -1) -> Tensor:
    # 检查张量 `x` 的数据类型，确保是浮点数或复数类型
    check_fp_or_complex(x.dtype, "linalg.vecdot")
    # 计算 `x` 与 `y` 的共轭乘积，并沿着指定维度求和，返回结果张量
    return (x.conj() * y).sum(dim=dim)
```