# `.\pytorch\torch\masked\_ops.py`

```
# mypy: allow-untyped-defs
# 引入警告模块，用于发出警告信息
import warnings
# 引入类型相关模块
from typing import Any, List, Optional, Tuple, TYPE_CHECKING, Union

# 引入PyTorch核心模块
import torch
# 引入符号浮点数相关模块
from torch import sym_float, Tensor
# 引入对应的实数数据类型
from torch._prims_common import corresponding_real_dtype
# 引入文档模块
from torch.masked import _docs
# 引入MaskedTensor相关模块
from torch.masked.maskedtensor.core import is_masked_tensor, MaskedTensor
# 引入创建MaskedTensor相关模块
from torch.masked.maskedtensor.creation import as_masked_tensor

# 如果类型检查开启，则引入特定类型
if TYPE_CHECKING:
    from torch.types import _dtype as DType
    # DimOrDims 可以是一个整数、整数元组、整数列表的可选类型
    DimOrDims = Optional[Union[int, Tuple[int], List[int]]]
else:
    # JIT 不理解 Union 和 torch.dtype 的情况下
    # DType 被定义为整数类型
    DType = int
    # DimOrDims 被定义为整数元组的可选类型
    DimOrDims = Optional[Tuple[int]]

# 导出的符号列表为空
__all__: List[str] = []

# 所有掩码化的规约/归一化操作具有相同的签名。
# 这里定义了用于规约/归一化函数文档字符串的模板，
# 通过 _apply_docstring_templates 装饰器应用于函数的文档字符串中。

def _apply_docstring_templates(func):
    """装饰器，将文档字符串模板应用于函数文档字符串，并返回函数实例。"""
    # 获取与函数名对应的文档字符串模板
    doc_string = getattr(_docs, f"{func.__name__}_docstring", None)
    if doc_string is None:
        # 如果找不到文档字符串模板，则发出警告
        warnings.warn(
            f"No documentation string available for {func.__name__}."
            " PyTorch team should run `python tools/update_masked_docs.py`"
            " to generate the missing docstrings."
        )
    else:
        # 否则，将函数的文档字符串设置为找到的文档字符串模板
        func.__doc__ = doc_string

    # 将函数名添加到导出符号列表中
    __all__.append(func.__name__)

    return func

def _generate_docstring(func):
    """工具函数，从 tools/update_masked_docs.py 脚本调用，
    用于更新模块 torch.masked._docs.py 的文档字符串。
    """
    # 定义文档字符串模板字典
    docstring_templates = dict(
        reduction_signature="""\
{function_name}(input, {operation_args}, *, {operation_kwargs}) -> Tensor""",
        reduction_descr="""\
Returns {operation name} of all the elements in the :attr:`input`
tensor along the given dimension(s) :attr:`dim` while the :attr:`input`
elements are masked out according to the boolean tensor
:attr:`mask`.""",
        reduction_args="""\
If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of
size 1. Otherwise, :attr:`dim` is squeezed (see
:func:`torch.squeeze`), resulting in the output tensor having 1 (or
``len(dim)``) fewer dimension(s).

The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True
then the corresponding element in :attr:`input` tensor will be
included in {operation name} computation, otherwise the element is
ignored.

When all elements of :attr:`input` along the given dimension
:attr:`dim` are ignored (fully masked-out), the corresponding element
of the output tensor will have undefined value: it may or may not
correspond to the identity value of {operation name} operation; the
choice may correspond to the value that leads to the most efficient
storage of :attr:`output` tensor.
"""
The mask of the output tensor can be computed as
``torch.any(torch.broadcast_to(mask, input.shape), dim, keepdim=keepdim,
dtype=torch.bool)``.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
don't need to match, but they must be :ref:`broadcastable
<broadcasting-semantics>` and the dimensionality of the :attr:`mask`
tensor must not be greater than of the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor
    {args_declarations}

Keyword args:
    {kwargs_declarations}""",
        reduction_example="""\
Example::

    >>> input = {example_input}
    >>> input
    {indent_example_input}
    >>> mask = {example_mask}
    >>> mask
    {indent_example_mask}
    >>> {full_function_name}(input, {example_args}, mask=mask)
    {indent_example_output}
""",
        reduction_identity="""\
The identity value of {operation name} operation, which is used to start the reduction, is ``{identity_int32}``.""",
        reduction_identity_dtype="""\
The identity value of {operation name} operation, which is used to start the
reduction, depends on input dtype. For instance, for float32, uint8,
and int32 dtypes, the identity values are ``{identity_float32}``, ``{identity_uint8}``, and ``{identity_int32}``, respectively.""",
        normalization_signature="""\
{function_name}(input, {operation_args}, *, {operation_kwargs}) -> Tensor""",
        normalization_descr="""\
Returns {operation name} of all the slices in the :attr:`input` tensor
along :attr:`dim` while the :attr:`input` elements are masked out
according to the boolean tensor :attr:`mask`.

{definition}""",
        normalization_args="""\
The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True then
the corresponding element in :attr:`input` tensor will be included in
{operation name} computation, otherwise the element is ignored.

The values of masked-out elements of the output tensor have undefined
value: it may or may not be set to zero or nan; the choice may correspond to
the value that leads to the most efficient storage of :attr:`output`
tensor.

The mask of the {operation name} output tensor can be computed as
``torch.broadcast_to(mask, input.shape)``.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
don't need to match, but they must be :ref:`broadcastable
<broadcasting-semantics>` and the dimensionality of the :attr:`mask`
tensor must not be greater than of the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor
    {args_declarations}

Keyword args:
    {kwargs_declarations}
    args_and_kwargs = dict(
        # 定义函数的参数和关键字参数列表，以字典形式存储
        sum=(("dim",), ("keepdim=False", "dtype=None", "mask=None")),
        # sum 函数的参数：dim（指定维度），关键字参数：keepdim（是否保持维度），dtype（数据类型），mask（掩码）
        prod=(("dim",), ("keepdim=False", "dtype=None", "mask=None")),
        # prod 函数的参数同上
        cumsum=(("dim__as_int",), ("dtype=None", "mask=None")),
        # cumsum 函数的参数：dim__as_int（指定维度，作为整数），dtype（数据类型），mask（掩码）
        cumprod=(("dim__as_int",), ("dtype=None", "mask=None")),
        # cumprod 函数的参数同上
        amin=(("dim",), ("keepdim=False", "dtype=None", "mask=None")),
        # amin 函数的参数同上
        amax=(("dim",), ("keepdim=False", "dtype=None", "mask=None")),
        # amax 函数的参数同上
        argmin=(("dim__as_int",), ("keepdim=False", "dtype=None", "mask=None")),
        # argmin 函数的参数同上
        argmax=(("dim__as_int",), ("keepdim=False", "dtype=None", "mask=None")),
        # argmax 函数的参数同上
        mean=(("dim",), ("keepdim=False", "dtype=None", "mask=None")),
        # mean 函数的参数同上
        median=(("dim__as_int",), ("keepdim=False", "dtype=None", "mask=None")),
        # median 函数的参数同上
        norm=(
            (
                "ord",
                "dim",
            ),
            ("keepdim=False", "dtype=None", "mask=None"),
        ),
        # norm 函数的参数：ord（指定范数的阶数），dim（指定维度），关键字参数同上
        var=(("dim", "unbiased"), ("keepdim=False", "dtype=None", "mask=None")),
        # var 函数的参数：dim（指定维度），unbiased（是否使用无偏估计），关键字参数同上
        std=(("dim", "unbiased"), ("keepdim=False", "dtype=None", "mask=None")),
        # std 函数的参数同上
        logsumexp=(("dim",), ("keepdim=False", "dtype=None", "mask=None")),
        # logsumexp 函数的参数同上
        softmax=(("dim__as_int",), ("dtype=None", "mask=None")),
        # softmax 函数的参数：dim__as_int（指定维度，作为整数），dtype（数据类型），mask（掩码）
        log_softmax=(("dim__as_int",), ("dtype=None", "mask=None")),
        # log_softmax 函数的参数同上
        softmin=(("dim__as_int",), ("dtype=None", "mask=None")),
        # softmin 函数的参数同上
        normalize=(
            (
                "ord__required",
                "dim__as_int",
            ),
            ("eps=1e-12", "dtype=None", "mask=None"),
        ),
        # normalize 函数的参数：ord__required（指定的必需的范数阶数），dim__as_int（指定维度，作为整数），关键字参数同上
    )

    argument_declarations = dict(
        dim="""\
dim (int or tuple of ints, optional): the dimension or dimensions to reduce.
  Default: None that is equivalent to ``tuple(range(input.ndim))``."""
        dim__as_int="""\
dim (int): the dimension along which {operation name} is computed."""
        ord="""\
ord (int, float, optional): the order of vector norm. Default: 2.
  See :func:`torch.linalg.vector_norm` for a list of supported norms."""
        ord__required="""\
ord (int, float): the order of vector norm. Default: 2.
  See :func:`torch.linalg.vector_norm` for a list of supported norms."""
        unbiased="""\
unbiased (bool): when True, use Bessel's correction, otherwise, compute
  the uncorrected sample variance."""
        eps="""\
eps (float, optional): small value to avoid division by zero. Default: {default}."""
        keepdim="""\
keepdim (bool, optional): whether the output tensor has
  :attr:`dim` retained or not. Default: {default}."""
        dtype="""\
dtype (:class:`torch.dtype`, optional): the desired data type
  of returned tensor.  If specified, the input tensor is
  casted to :attr:`dtype` before the operation is
  performed. Default: {default}."""
        mask="""\
mask (:class:`torch.Tensor`, optional): the boolean tensor
  containing the binary mask of validity of input tensor
  elements.
  Default: None that is equivalent to ``torch.ones(input.shape, dtype=torch.bool)``."""

definitions = dict(
    softmax="""\
Let ``x`` be a sequence of unmasked elements of one-dimensional slice
of the :attr:`input` tensor. Softmax of i-th element in ``x`` is
defined as ``exp(x[i])/sum(exp(x))``.""",
    log_softmax="""\
Let ``x`` be a sequence of unmasked elements of one-dimensional slice
of the :attr:`input` tensor. LogSoftmax of i-th element in ``x`` is
defined as ``log(exp(x[i])/sum(exp(x)))``.""",
    softmin="""\
Let ``x`` be a sequence of unmasked elements of one-dimensional slice
of the :attr:`input` tensor. Softmin of i-th element in ``x`` is
defined as ``exp(-x[i])/sum(exp(-x))``.""",
    normalize="""\
Let ``x`` be a sequence of unmasked elements of one-dimensional slice
of the :attr:`input` tensor. Normalize of i-th element in ``x`` is
defined as ``x[i]/max(norm(x, p), eps)``.""",
    cumsum="""\
Let ``x`` be a sequence of unmasked elements of one-dimensional slice
of the :attr:`input` tensor. Cumsum of i-th element in ``x`` is
defined as ``sum(x[:i])``.""",
    cumprod="""\
Let ``x`` be a sequence of unmasked elements of one-dimensional slice
of the :attr:`input` tensor. Cumsum of i-th element in ``x`` is
defined as ``prod(x[:i])``.""",
)

reduction_names = dict(
    sum="sum",
    prod="product",
    amax="maximum",
    amin="minimum",
    argmax="argmax",
    argmin="argmin",
    mean="mean",
    median="median",
    norm="norm",
    var="variance",
    std="standard_deviation",
    logsumexp="logsumexp",
)
    normalization_names = dict(
        softmax="softmax",
        log_softmax="log_softmax",
        softmin="softmin",
        normalize="normalize",
        cumsum="cumulative_sum",
        cumprod="cumulative_prod",
    )

    operation_names = {}
    operation_names.update(reduction_names)  # 将 reduction_names 中的内容更新到 operation_names 中
    operation_names.update(normalization_names)  # 将 normalization_names 中的内容更新到 operation_names 中

    # 默认示例数据：
    example_dim = 1  # 示例数据维度
    example_input = torch.tensor([[-3, -2, -1], [0, 1, 2]])  # 示例输入张量
    example_mask = torch.tensor([[True, False, True], [False, False, False]])  # 示例掩码张量
    example_args: Tuple[Any, ...]  # 示例参数元组

    # 根据函数名称设置示例参数和转换示例输入的数据类型
    if func.__name__ in {"norm", "normalize"}:
        example_args = (2.0, example_dim)  # 标准化相关函数的示例参数
        example_input = example_input.to(dtype=torch.float32)  # 将示例输入转换为 float32 类型
    elif func.__name__ in {"var", "std"}:
        example_args = (example_dim, False)  # 方差和标准差相关函数的示例参数
    elif func.__name__ == "median":
        example_args = (example_dim,)  # 中位数相关函数的示例参数
        example_input = example_input.to(dtype=torch.float32)  # 将示例输入转换为 float32 类型
    else:
        example_args = (example_dim,)  # 默认情况下的示例参数

    operation_args: Tuple[str, ...]  # 操作的位置参数元组
    operation_kwargs: Tuple[str, ...]  # 操作的关键字参数元组
    operation_args, operation_kwargs = args_and_kwargs[func.__name__]  # 获取函数名称对应的参数和关键字参数
    arg_declarations = [
        "\n    ".join(
            argument_declarations.get(a, f'{a.split("__", 1)[0]}: TBD.')  # 获取参数声明，若不存在则标记为待定
            .splitlines()
        )
        for a in operation_args
    ]
    kwarg_declarations = [
        "\n    ".join(
            argument_declarations.get(
                a.split("=", 1)[0], f'{a.split("__", 1)[0]}: TBD.'  # 获取关键字参数声明，若不存在则标记为待定
            )
            .format(default=a.split("=", 1)[1])  # 格式化默认值
            .splitlines()
        )
        for a in operation_kwargs
    ]

    if func.__name__ in reduction_names:
        op_kind = "reduction"  # 操作类型为减少（reduction）
        doc_sections = ["signature", "descr", "identity", "args", "example"]  # 文档部分包括签名、描述、标识、参数、示例
    elif func.__name__ in normalization_names:
        op_kind = "normalization"  # 操作类型为标准化（normalization）
        doc_sections = ["signature", "descr", "args", "example"]  # 文档部分包括签名、描述、参数、示例
        example_input = example_input.to(dtype=torch.float32)  # 将示例输入转换为 float32 类型
    else:
        assert 0  # 如果不是已知的操作类型，引发断言错误

    example_output = func(example_input, *example_args, mask=example_mask)  # 使用函数计算示例输出
    # 构建模板数据字典，用于生成函数文档字符串的各个部分
    template_data = {
        "function_name": func.__name__,  # 函数名
        "full_function_name": func.__module__ + "." + func.__name__,  # 完整的函数名（模块名.函数名）
        "operation name": operation_names[func.__name__],  # 操作名称
        "operation_args": ", ".join(a.split("__", 1)[0] for a in operation_args),  # 操作的参数列表
        "operation_kwargs": ", ".join(a.split("__", 1)[0] for a in operation_kwargs),  # 操作的关键字参数列表
        # 以下是示例输入的单行表示
        "example_input": " ".join(str(example_input).split()),  # 示例输入
        "example_args": ", ".join(map(str, example_args)),  # 示例参数
        "example_mask": " ".join(str(example_mask).split()),  # 示例掩码
        # 以下是示例输入的多行表示，带有缩进
        "indent_example_input": ("\n    ").join(str(example_input).splitlines()),  # 示例输入（带缩进）
        "indent_example_mask": ("\n    ").join(str(example_mask).splitlines()),  # 示例掩码（带缩进）
        "indent_example_output": ("\n    ").join(str(example_output).splitlines()),  # 示例输出（带缩进）
    }
    
    # 如果函数名在 reduction_names 中
    if func.__name__ in reduction_names:
        # 更新模板数据字典，添加不同数据类型的缩减操作的单位元
        template_data.update(
            identity_uint8=_reduction_identity(
                func.__name__, torch.tensor(0, dtype=torch.uint8)
            ),
            identity_int32=_reduction_identity(
                func.__name__, torch.tensor(0, dtype=torch.int32)
            ),
            identity_float32=_reduction_identity(
                func.__name__, torch.tensor(0, dtype=torch.float32)
            ),
        )
        # 如果函数名为 "norm"，添加额外的单位元（负无穷）
        if func.__name__ == "norm":
            template_data.update(
                identity_ord_ninf=_reduction_identity(
                    func.__name__, torch.tensor(0, dtype=torch.float32), float("-inf")
                )
            )
    # 如果函数名在 normalization_names 中
    elif func.__name__ in normalization_names:
        # 更新模板数据字典，添加归一化操作的定义
        template_data.update(definition=definitions[func.__name__])
    else:
        assert 0  # 如果不属于以上任何一种情况，则断言错误（应将函数名添加到操作名称字典中）
    
    # 更新模板数据字典，添加参数声明和关键字参数声明
    template_data.update(
        args_declarations=("\n    ".join(arg_declarations)).format_map(template_data),
    )
    template_data.update(
        kwargs_declarations=("\n    ".join(kwarg_declarations)).format_map(
            template_data
        )
    )
    
    # 将模板数据字典应用于文档字符串模板，生成各个部分的文本内容
    templates = {
        k: v.format_map(template_data)
        for k, v in docstring_templates.items()
        if k.startswith(op_kind)
    }
    templates.update(
        (k, v.format_map(template_data) if isinstance(v, str) else v)
        for k, v in template_data.items()
    )
    
    # 将文档字符串模板应用于函数的文档字符串
    if func.__doc__ is None:
        doc_template = "\n\n".join([f"{{{op_kind}_{sec}}}" for sec in doc_sections])
    else:
        doc_template = func.__doc__
    # 返回填充了模板的最终文档字符串
    return doc_template.format_map(templates)
# 返回给定输入的减少操作的标量张量的身份值，或者如果无法为给定输入唯一定义身份值，则返回None。
# 函数接受一个操作名称字符串，一个输入张量和任意数量的其他参数。
def _reduction_identity(op_name: str, input: Tensor, *args):
    """Return identity value as scalar tensor of a reduction operation on
    given input, or None, if the identity value cannot be uniquely
    defined for the given input.

    The identity value of the operation is defined as the initial
    value to reduction operation that has a property ``op(op_identity,
    value) == value`` for any value in the domain of the operation.
    Or put it another way, including or excluding the identity value in
    a list of operands will not change the reduction result.

    See https://github.com/pytorch/rfcs/pull/27 for more information.

    """
    # 获取输入张量的数据类型
    dtype: DType = input.dtype
    # 获取输入张量的设备
    device = input.device
    # 从操作名称中去除模块名称（如果有）
    op_name = op_name.rsplit(".", 1)[-1]  # lstrip module name when present
    
    # 根据不同的操作名称返回对应的标量张量身份值
    if op_name in {"sum", "cumsum"}:
        return torch.tensor(0, dtype=dtype, device=device)
    elif op_name in {"prod", "cumprod"}:
        return torch.tensor(1, dtype=dtype, device=device)
    elif op_name in {"amax", "argmax", "logsumexp"}:
        # 对于浮点数类型的输入，返回负无穷
        if torch.is_floating_point(input):
            return torch.tensor(-torch.inf, dtype=dtype, device=device)
        # 对于带符号的输入或者 uint8 类型，返回类型的最小值
        elif torch.is_signed(input) or dtype == torch.uint8:
            return torch.tensor(torch.iinfo(dtype).min, dtype=dtype, device=device)
    elif op_name in {"amin", "argmin"}:
        # 对于浮点数类型的输入，返回正无穷
        if torch.is_floating_point(input):
            return torch.tensor(torch.inf, dtype=dtype, device=device)
        # 对于带符号的输入或者 uint8 类型，返回类型的最大值
        elif torch.is_signed(input) or dtype == torch.uint8:
            return torch.tensor(torch.iinfo(dtype).max, dtype=dtype, device=device)
    elif op_name == "mean":
        # 对于均值操作，身份值的定义存在歧义，因为均值依赖于 dim 参数，并且可能是非标量张量
        # 空输入的均值未定义
        return None
    elif op_name == "norm":
        # 对于范数操作，如果 ord 参数为负无穷，返回正无穷
        ord = args[0] if args else 2
        if ord == float("-inf"):
            assert torch.is_floating_point(input), input.dtype
            return torch.tensor(torch.inf, dtype=dtype, device=device)
        # 否则返回零张量
        return torch.tensor(0, dtype=dtype, device=device)
    elif op_name == "median":
        # 目前使用 NaN，因为实现当前使用 torch.nanmedian，
        # 而 NaN 是该函数的身份值，因为它会被忽略
        dtype = input.dtype if torch.is_floating_point(input) else torch.float
        return torch.tensor(torch.nan, dtype=dtype, device=device)
    elif op_name in {"var", "std"}:
        # 方差和标准差的身份值为 None
        return None
    # 如果没有匹配的操作名称，抛出未实现错误
    raise NotImplementedError(f"identity of {op_name} on {dtype} input")


# 返回维度参数作为已排序维度值的元组
def _canonical_dim(dim: DimOrDims, ndim: int) -> Tuple[int, ...]:
    """Return dim argument as a tuple of sorted dim values."""
    # 初始化一个空列表用于存放维度值
    dims: List[int] = []
    # 如果 `dim` 是一个空元组 `()`，表示在当前的减少操作中“在所有维度上减少”，
    # 但未来可能会表示“不减少”。参见 https://github.com/pytorch/pytorch/issues/29137
    # 一旦问题 gh-29137 得到解决，这个 if 块必须删除。
    if dim == ():
        dim = None  # 将空元组 `()` 转换为 `None`
    
    # 如果 `dim` 是 `None`，则返回所有维度的范围作为元组
    if dim is None:
        return tuple(range(ndim))
    
    ndim = max(ndim, 1)  # 确保 `ndim` 至少为 1
    
    # 如果 `dim` 是整数或者 torch.SymInt 类型的单个值，则转换为包含一个元素的元组
    dim_ = (dim,) if isinstance(dim, (int, torch.SymInt)) else dim
    
    # 遍历 `dim_` 中的每个维度
    for d in dim_:
        # 如果维度 `d` 已经在 `dims` 列表中出现过，则抛出运行时错误
        if d in dims:
            raise RuntimeError(f"dim={d} appears multiple times in the list of dims")
        
        # 如果维度 `d` 超出了有效范围，抛出索引错误
        if d >= ndim or d < -ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], but got {d})"
            )
        
        # 将维度 `d` 添加到 `dims` 列表中，使用取模确保在有效范围内
        dims.append(d % ndim)
    
    # 返回排序后的 `dims` 列表作为元组
    return tuple(sorted(dims))
def _sparse_coo_flatten_indices(indices: Tensor, shape: tuple):
    # 将稀疏 COO 格式的多维索引扁平化为一维索引
    flat_indices = indices.new_zeros(indices.size(1))
    for d, sz in enumerate(shape):
        flat_indices.mul_(sz)
        flat_indices.add_(indices[d])
    return flat_indices


def _any(input: Tensor, dim: tuple, keepdim: bool):
    # 支持带有元组维度参数的 torch.any 函数。
    # 解决 https://github.com/pytorch/pytorch/issues/56586 的问题
    r = input
    for d in reversed(dim):
        r = r.any(dim=d, keepdim=keepdim)
    return r


def _sparse_coo_where(mask: Tensor, input: Tensor, fill_value: Tensor) -> Tensor:
    """Sparse variant of torch.where. Supports sparse COO and hybrid sparse COO tensors.

    _sparse_coo_where implements the following invariant:

      _sparse_coo_where(mask, input, fill_value).to_dense(fill_value) ==
        torch.where(mask.to_dense(), input.to_dense(), torch.full(input.shape, fill_value))

    where `a == b` means `assertEqual(a, b)`, mask is boolean sparse
    tensor, and `to_dense(fill_value)` is like `to_dense()` except
    that the unspecified elements are mapped to `fill_value` rather
    than to `0`.

    Returns a sparse COO tensor with the following features:

    - all specified elements correspond to masked-in elements that
      have the values of the input tensor. If there exists a masked-in
      element (as specified by mask) that is not specified in the
      input, in the result tensor, the corresponding element has value
      0. In the dense part of the sparse tensor, the masked-out
      elements are replaced with fill_value.

    - all unspecified elements correspond to masked-out elements.
    """

    assert input.layout == torch.sparse_coo
    assert mask.layout == input.layout
    assert mask.shape == input.shape
    assert mask.dense_dim() == input.dense_dim()  # TODO: eliminate this restriction

    input = input.coalesce()

    # 对稀疏张量的索引进行集合操作，为了提高效率，将多维索引转换为一维索引。
    input_flat_indices = _sparse_coo_flatten_indices(
        input.indices(), input.shape[: input.sparse_dim()]
    )
    mask_flat_indices = _sparse_coo_flatten_indices(
        mask.indices(), mask.shape[: mask.sparse_dim()]
    )

    # 定义交集和差集函数
    def intersection(i1, i2):
        union, counts = torch.cat([i1, i2]).unique(return_counts=True)
        return union, torch.where(counts.gt(1))

    def minus(i1, i2):
        union, counts = torch.cat([i1, i2]).unique(return_counts=True)
        return intersection(union[torch.where(counts.eq(1))], i1)

    # 应用函数，这里是一个占位注释
    def _apply(a):
        obj, w = a
        return obj[w]
    # 获取指定和掩码输入元素的平面索引的集合：
    maskin_input_flat_indices = _apply(
        intersection(maskin_flat_indices, input_flat_indices)
    )
    # 获取掩码输入元素和输入元素的交集的索引和值：
    _, w = intersection(input_flat_indices, maskin_input_flat_indices)
    
    # 获取掩码输入元素的索引：
    where_input_indices = input.indices()[(slice(None),) + w]
    # 获取掩码输入元素的值：
    where_input_values = input.values()[w]
    
    if mask.dense_dim() > 0:
        # 将掩码应用于输入值的稠密部分：
        _, w1 = intersection(mask_flat_indices, maskin_input_flat_indices)
        where_mask_values = mask.values()[w1]
        where_input_values = torch.where(
            where_mask_values, where_input_values, fill_value
        )
    
    # 获取未指定输入和掩码输入元素的平面索引的集合：
    maskin_zero_flat_indices = _apply(
        minus(maskin_flat_indices, maskin_input_flat_indices)
    )
    
    # 获取掩码输入为零元素的索引：
    _, w = intersection(mask_flat_indices, maskin_zero_flat_indices)
    where_zero_indices = mask.indices()[(slice(None),) + w]
    
    # 构造结果
    n = where_zero_indices.size(1)
    if n == 0:
        # 输入已合并，因此输入_flat_indices已排序，并且结果保证合并：
        result = torch.sparse_coo_tensor(
            where_input_indices, where_input_values, input.shape
        )
        return result._coalesced_(True)
    
    # 合并输入和零元素的索引：
    where_indices = torch.cat([where_input_indices, where_zero_indices], dim=1)
    # 合并输入和零元素的值：
    where_values = torch.cat(
        [
            where_input_values,
            where_input_values.new_zeros((n,) + where_input_values.shape[1:]),
        ]
    )
    result = torch.sparse_coo_tensor(where_indices, where_values, input.shape)
    
    # 添加零元素导致不合并的稀疏张量
    return result.coalesce()
# 定义一个辅助函数用于稀疏 COO 格式张量的按位运算归约操作
def _sparse_coo_scatter_reduction_helper(
    op,
    mask_input: Tensor,
    dims: Tuple[int, ...],
    keepdim: bool,
    dtype: Optional[DType] = None,
) -> Tensor:
    # 获取归约操作的名称
    reduce = op.__name__
    # 可接受的归约操作列表
    valid_reductions = ["sum", "prod", "amax", "amin"]
    # 如果指定的操作不在有效操作列表中，抛出 ValueError 异常
    if reduce not in valid_reductions:
        raise ValueError(
            f"op must be one of {' '.join(valid_reductions)}, but got {reduce} instead"
        )

    # 输出张量的数据类型
    output_dtype = dtype
    # 获取稀疏张量的值和索引
    values, indices = mask_input._values(), mask_input._indices()
    # 输入张量的维度
    input_dims = mask_input.dim()
    # 稀疏张量的稀疏维度数
    num_sparse_dims = mask_input.sparse_dim()
    # 被归约的稀疏维度
    reduced_sparse_dims = []
    # 保留的稀疏维度
    retained_sparse_dims = []
    # 被归约的密集维度
    reduced_dense_dims = []

    # 如果指定了数据类型，提升值的数据类型
    if values.dtype != output_dtype:
        values = values.to(output_dtype)

    # 如果保持维度，设置输出形状为在指定维度上为1的形状
    if keepdim:
        output_shape = tuple(
            1 if i in dims else si for (i, si) in enumerate(mask_input.shape)
        )
    else:
        # 否则，输出形状为剔除指定维度后的形状
        output_shape = tuple(
            si for (i, si) in enumerate(mask_input.shape) if i not in dims
        )

    # 对于每个指定的维度
    for d in dims:
        # 如果维度超出输入张量的维度，跳过
        if d >= input_dims:
            continue

        # 如果维度小于稀疏维度数，将其添加到被归约的稀疏维度列表中
        if d < num_sparse_dims:
            reduced_sparse_dims.append(d)
        else:
            # 否则，添加到被归约的密集维度列表中
            reduced_dense_dims.append(d + 1 - num_sparse_dims)

    # 归约密集维度
    if len(reduced_dense_dims) > 0:
        if reduce == "sum":
            # 对值进行求和归约操作
            new_values = op(new_values, dim=reduced_dense_dims, keepdim=bool(keepdim))
        else:
            # FIXME: 实现具有非零归约标识的操作的密集维度归约
            # 对于非零归约标识的操作暂不支持
            return NotImplemented
    else:
        # 复制值，不进行密集维度归约
        new_values = values.clone()

    # 归约稀疏维度
    if len(reduced_sparse_dims) == num_sparse_dims:
        if reduce in {"amax", "amin"} and new_values.size(0) == 0:
            # 如果是 amax 或 amin 操作且新值大小为0，则返回对应的归约标识
            # sum()/prod() 在维度为0且大小为0时返回归约标识，但amax()/amin()不会
            # 参见 https://github.com/pytorch/pytorch/issues/61901
            new_values = _reduction_identity(reduce, new_values)
        else:
            # 对值进行稀疏维度归约操作
            new_values = op(new_values, dim=0)
        if keepdim:
            # 如果保持维度，对稀疏维度进行扩展
            for _ in range(num_sparse_dims):
                new_values = new_values.unsqueeze(0)
        # 将结果转换为指定的数据类型，并转换为稀疏张量
        return new_values.to(dtype=output_dtype).to_sparse()
    else:
        # Clone the indices tensor to operate on
        new_indices = indices.clone()
        
        if keepdim:
            # If keepdim is True, zero out reduced sparse dimensions
            # This ensures that duplicate indices are folded together while preserving dimension
            new_indices[reduced_sparse_dims, :] = 0
        else:
            # If keepdim is False, remove reduced sparse dimensions
            if len(reduced_sparse_dims) > 0:
                # Filter out reduced sparse dimensions from the list of all dimensions
                retained_sparse_dims = [
                    i
                    for i in range(num_sparse_dims)
                    if i not in set(reduced_sparse_dims)
                ]
                # Select indices corresponding to retained sparse dimensions
                new_indices = new_indices.index_select(
                    0, torch.tensor(retained_sparse_dims).to(mask_input.device)
                )

    # Use scatter_reduce to reduce items in the new_values tensor that correspond to the same indices in new_indices
    if new_indices.numel() > 0:
        # Sort and get unique indices, along with the index tensor for scatter reduction
        new_indices, inverse_indices = torch.unique(
            new_indices, return_inverse=True, dim=1
        )
        # Determine the shape of the output tensor
        out_shape = list(new_values.shape)
        out_shape[0] = new_indices.shape[1]
        # Expand inverse indices to match the shape of new_values
        for _ in range(new_values.ndim - 1):
            inverse_indices = inverse_indices.unsqueeze(-1)
        scatter_indices = inverse_indices.expand(new_values.shape)
        
        # FIXME: temporary workaround for issue with bfloat16/float16 remove when acctype is implemented for scatter_reduce
        if output_dtype in {torch.bfloat16, torch.float16}:
            # Convert new_values to float temporarily if output_dtype is bfloat16 or float16
            new_values = new_values.to(torch.float)
            # Create an empty tensor out with the determined shape
            out = new_values.new_empty(out_shape)
            # Perform scatter reduction operation on new_values tensor
            new_values = out.scatter_reduce_(
                0, scatter_indices, new_values, reduce=reduce, include_self=False
            )
            # Convert back new_values to the specified output_dtype
            new_values = new_values.to(dtype=output_dtype)
        else:
            # Create an empty tensor out with the determined shape
            out = new_values.new_empty(out_shape)
            # Perform scatter reduction operation on new_values tensor
            new_values = out.scatter_reduce_(
                0, scatter_indices, new_values, reduce=reduce, include_self=False
            )

    # Return a sparse COO tensor with specified properties
    return torch.sparse_coo_tensor(
        new_indices,
        new_values,
        output_shape,
        dtype=output_dtype,
        device=mask_input.device,
    )
    # 辅助函数用于稀疏CSR张量的段约简操作
    # op: 用于约简操作的函数
    # mask_input: 输入的稀疏CSR张量
    # dims: 要进行约简的维度，通常为一个或两个维度的元组
    # keepdim: 是否保持约简后的维度
    # dtype: 输出张量的数据类型（可选）
def _sparse_csr_segment_reduction_helper(
    op,
    mask_input: Tensor,
    dims: Tuple[int, ...],
    keepdim: bool,
    dtype: Optional[DType] = None,
) -> Tensor:
    # 当前稀疏CSR张量总是2D且没有稠密维度，因此keepdim必须为True
    # FIXME: 当为CSR张量实现稠密维度时需要修复此处
    assert (
        keepdim
    ), "reduction operations on CSR tensors with keepdim=False is unsupported"
    
    # 获取约简操作的函数名
    reduce = op.__name__
    # 支持的有效约简操作列表
    valid_reductions = ["sum", "prod", "mean", "amax", "amin"]
    if reduce not in valid_reductions:
        raise ValueError(
            f"op must be one of {' '.join(valid_reductions)}, but got {reduce} instead"
        )
    
    # 获取输入张量的设备信息
    device = mask_input.device
    # 输出张量的数据类型
    output_dtype = dtype
    # 提取输入张量的值、行索引和列索引
    values, crow_indices, col_indices = (
        mask_input.values(),
        mask_input.crow_indices(),
        mask_input.col_indices(),
    )

    # 如果指定了输出数据类型，则将值提升为该类型
    if values.dtype != output_dtype:
        values = values.to(output_dtype)

    # 如果dims为空，则直接返回输入张量
    if len(dims) == 0:
        return mask_input
    
    # 如果dims长度为1
    if len(dims) == 1:
        # 如果dims的唯一元素为0
        if dims[0] == 0:
            # 对列索引进行唯一化并返回逆向索引，以及新的非零元素个数
            new_col_indices, scatter_indices = torch.unique(
                col_indices, return_inverse=True
            )
            new_nnz = new_col_indices.shape[0]
            # 新的行索引只有一个元素为0和新的非零元素个数
            new_crow_indices = torch.tensor([0, new_nnz])
            # 创建一个新的空值张量，使用scatter_reduce_进行归约操作
            new_values = values.new_empty(new_col_indices.shape)
            new_values.scatter_reduce_(
                0, scatter_indices, values, reduce, include_self=False
            )
            # 新的形状为[1, 输入张量的列数]
            new_shape = [1, mask_input.size(1)]
        else:
            # 否则，dims的唯一元素必须为1
            assert (
                dims[0] == 1
            ), "Sparse CSR tensors are 2D and only support reduction along dim 0 or 1."
            # 创建新的行索引，其中除了crow_indices[i] == crow_indices[i-1]的地方外，其他地方都为1
            new_crow_indices = torch.cat(
                (
                    crow_indices.new_zeros(1),
                    torch.cumsum(torch.diff(crow_indices) != 0, 0),
                ),
                0,
            )
            # 新的非零元素个数为新行索引的最后一个元素
            new_nnz = new_crow_indices[-1]
            # 创建新的全零列索引
            new_col_indices = col_indices.new_zeros(new_nnz)
            # 使用torch._segment_reduce进行分段约简操作
            new_values = torch._segment_reduce(values, reduce, offsets=crow_indices)  # type: ignore[attr-defined]
            # 新的形状为[输入张量的行数, 1]
            new_shape = [mask_input.size(0), 1]
    else:
        # 否则，dims的长度必须为2
        assert len(dims) == 2
        # 计算非零元素的数量，取1和值的元素数量的较小值
        nnz = min(1, values.numel())
        if nnz == 1:
            # 如果非零元素数量为1，则进行操作的关键字参数
            op_kwargs = {"keepdim": True, "dtype": output_dtype}
            # amax和amin不支持dtype关键字参数
            if reduce in ["amax", "amin"]:
                del op_kwargs["dtype"]
            # 对值进行操作，并返回新的值张量
            new_values = op(values, 0, **op_kwargs)
        else:
            # 否则，创建一个新的空值张量
            new_values = torch.empty(0, dtype=output_dtype)
        # 创建新的全零列索引
        new_col_indices = col_indices.new_zeros(nnz)
        # 新的行索引只有一个元素为0和nnz
        new_crow_indices = torch.tensor([0, nnz])
        # 新的形状为[1, nnz]
    # 使用给定的参数创建一个稀疏的 CSR (Compressed Sparse Row) 张量对象并返回
    return torch.sparse_csr_tensor(
        new_crow_indices,  # 新的行索引数组，用于构建稀疏张量的行索引
        new_col_indices,   # 新的列索引数组，用于构建稀疏张量的列索引
        new_values,        # 新的值数组，用于构建稀疏张量的非零值
        new_shape,         # 新的形状数组，指定稀疏张量的形状
        dtype=output_dtype,  # 输出稀疏张量的数据类型
        device=device,     # 稀疏张量的计算设备（如 CPU 或 GPU）
    )
# 返回一个稀疏 CSR 格式的张量，用于代替 torch.where 函数。支持稀疏的 CSR 张量。
def _sparse_csr_where(mask: Tensor, input: Tensor, fill_value: Tensor) -> Tensor:
    """Sparse variant of torch.where. Supports sparse CSR tensors."""
    # TODO: implement sparse CSR specific where operator for efficiency
    # 调用 _sparse_coo_where 函数，将输入的 mask 和 input 张量转换为稀疏 COO 格式，然后应用 where 操作，
    # 最后将结果转换为稀疏的 CSR 格式返回
    return _sparse_coo_where(
        mask.to_sparse_coo(), input.to_sparse_coo(), fill_value
    ).to_sparse_csr()


# 实现了支持稀疏输入的 torch.where 函数。
def _where(mask: Tensor, input: Tensor, fill_value: Tensor) -> Tensor:
    """torch.where with sparse inputs support.

    _where implements the following invariant:

      _where(mask, input, fill_value).to_dense(fill_value) ==
        torch.where(mask.to_dense(), input.to_dense(), torch.full(input.shape, fill_value))

    where `a == b` means `assertEqual(a, b)`, mask is boolean sparse
    tensor, and `to_dense(fill_value)` is like `to_dense()` except
    that the unspecified elements are mapped to `fill_value` rather
    than to `0`.

    Returns a sparse tensor with the following features:

    - all specified elements correspond to masked-in elements that
      have the values of the input tensor. If there exists a masked-in
      element (as specified by mask) that is not specified in the
      input, in the result tensor, the corresponding element has value
      0. In the dense part of the sparse tensor, the masked-out
      elements are replaced with fill_value.

    - all unspecified elements correspond to masked-out elements.
    """
    # 根据 mask 张量的布局类型选择相应的操作分支
    if mask.layout == torch.strided:
        # 如果 mask 的布局是 strided，则直接调用 torch.where 函数
        return torch.where(mask, input, fill_value)
    elif mask.layout == torch.sparse_coo:
        # 如果 mask 的布局是 sparse COO，则调用 _sparse_coo_where 函数
        return _sparse_coo_where(mask, input, fill_value)
    elif mask.layout == torch.sparse_csr:
        # 如果 mask 的布局是 sparse CSR，则调用 _sparse_csr_where 函数
        return _sparse_csr_where(mask, input, fill_value)
    else:
        # 如果 mask 的布局不是 strided、sparse COO 或 sparse CSR，则抛出异常
        raise ValueError(
            f"_where expects strided or sparse COO or sparse CSR tensor but got {mask.layout}"
        )


# 返回规范化的输入 mask 张量，确保形状、布局和 dtype 与输入张量匹配
def _input_mask(input: Union[Tensor, MaskedTensor], *args, **kwargs) -> Tensor:
    """Return canonical input mask.

    A canonical input mask is defined as a boolean mask tensor that
    shape and layout matches with the shape and the layout of the
    input.

    The canonical input mask is computed from the :attr:`mask` tensor
    content to meet the following criteria:

    1. The shape of the canonical input mask is the same as the shape
       of :attr:`input` tensor. If the mask tensor has a smaller shape
       than the shape of the :attr:`input`, broadcasting rules will be
       applied. Downcasting of mask is not supported.

    2. The layout of the canonical input mask is the same as the
       layout of the :attr:`input` tensor. If the mask has different
       layout, it will be converted to the expected layout.  In the
       case of sparse COO layout, the canonical input mask will be
       coalesced.

    3. The dtype of the canonical input mask is torch.bool. If the
       mask dtype is not bool then it will be converted to bool dtype
       using `.to(dtype=bool)` method call.
    """
    # 返回一个规范化的输入 mask 张量，确保其形状、布局和 dtype 与输入张量一致
    # 检查输入张量的布局，必须是 strided、sparse COO 或 sparse CSR
    if input.layout not in {torch.strided, torch.sparse_coo, torch.sparse_csr}:
        raise ValueError(
            f"_input_mask expects strided or sparse COO or sparse CSR tensor but got {input.layout}"
        )

    # 获取关键字参数中的 mask
    mask = kwargs.get("mask")

    # 默认情况下，必须提供 mask 参数
    if mask is None:
        raise ValueError("_input_mask requires explicit mask")

    # 确保 mask 的形状与输入张量的形状相匹配
    if mask.shape != input.shape:
        # 如果 mask 的维度高于输入张量的维度，则无法广播
        if mask.ndim > input.ndim:
            raise IndexError(
                "_input_mask expected broadcastable mask (got mask dimensionality higher than of the input)"
            )
        # 如果 mask 的布局是 strided，则将其广播到与输入张量相同的形状，并转换为布尔类型
        if mask.layout == torch.strided:
            mask = torch.broadcast_to(mask.clone(), input.shape).to(dtype=torch.bool)
        # 如果 mask 的布局是 sparse COO，则使用 sparse 广播
        elif mask.layout == torch.sparse_coo:
            mask = torch._sparse_broadcast_to(mask, input.shape)
        else:
            # 如果 mask 的布局是 sparse CSR，则通过先转换为 COO 布局来实现广播
            assert mask.layout == torch.sparse_csr
            mask = torch._sparse_broadcast_to(
                mask.to_sparse(), input.shape
            ).to_sparse_csr()

    # 确保 mask 的布局与输入张量的布局相匹配
    if mask.layout != input.layout:
        # 如果输入张量的布局是 strided，则将 mask 转换为 dense tensor
        if input.layout == torch.strided:
            mask = mask.to_dense()
        # 如果输入张量的布局是 sparse COO，则根据 mask 的布局进行转换
        elif input.layout == torch.sparse_coo:
            if mask.layout == torch.strided:
                mask = mask.to_sparse(input.sparse_dim())
            else:
                mask = mask.to_sparse()
        else:
            # 如果输入张量的布局是 sparse CSR，则将 mask 转换为 sparse CSR 布局
            assert input.layout == torch.sparse_csr
            mask = mask.to_sparse_csr()

    # 如果 mask 的布局是 sparse COO，则确保其是 coalesced 的
    if mask.layout == torch.sparse_coo:
        mask = mask.coalesce()

    # 将 mask 转换为布尔类型的张量
    mask = mask.to(dtype=torch.bool)

    return mask
# 返回给定参数应用掩码操作的输出掩码
def _output_mask(op, input: Tensor, *args, **kwargs) -> Tensor:
    # 如果操作是可调用的
    if callable(op):
        # 检查是否为减少操作
        is_reduction = op.__name__ in {
            "sum",
            "prod",
            "amax",
            "amin",
            "argmax",
            "argmin",
            "mean",
            "median",
            "norm",
            "var",
            "std",
            "logsumexp",
        }
        # 检查是否为归一化操作
        is_normalization = op.__name__ in {
            "softmax",
            "log_softmax",
            "softmin",
            "normalize",
            "cumsum",
            "cumprod",
        }
        # 如果是减少操作
        if is_reduction:
            # 对于 norm 操作，如果存在参数则去除第一个参数（即 lstrip ord 参数）
            if op.__name__ == "norm":
                if args:
                    args = args[1:]  # lstrip ord argument
            # 获取维度参数
            dim = args[0] if args else kwargs.get("dim")
            # 获取输入的输出掩码
            outmask = _input_mask(input, *args, **kwargs)
            # 获取是否保持维度的标志
            keepdim = kwargs.get("keepdim", False)
            # 规范化维度
            dim_ = _canonical_dim(dim, input.ndim)
            # 返回输出掩码
            return _any(outmask, dim_, bool(keepdim))
        # 如果是归一化操作
        elif is_normalization:
            # 返回输入的输出掩码
            return _input_mask(input, *args, **kwargs)
        else:
            # 抛出数值错误，指示预期的掩码操作类型不正确
            raise ValueError(
                f"_output_mask expected masked operation (got callable {op.__module__}.{op.__name__})"
            )
    else:
        # 抛出数值错误，指示预期的掩码操作类型不正确
        raise ValueError(
            f"_output_mask expected masked operation (got {type(op).__name__} object)"
        )


# 合并输入和掩码
def _combine_input_and_mask(
    op, input: Union[MaskedTensor, Tensor], mask, *args
) -> Tensor:
    # 内部辅助函数，处理输入和掩码
    def helper(input, mask):
        # 如果掩码为 None，则返回输入本身
        if mask is None:
            return input
        # 规范化掩码
        canonical_mask = _input_mask(input, mask=mask)
        # 如果操作是可调用的
        if callable(op):
            # 获取填充值
            fill_value = _reduction_identity(op.__name__, input, *args)
            # 返回应用掩码后的值
            return _where(canonical_mask, input, fill_value)
        else:
            # 抛出数值错误，指示预期的掩码操作类型不正确
            raise ValueError(
                f"_combine_input_and_mask expected masked operation (got {type(op).__name__} object)"
            )

    # 内部类，代表组合操作
    class Combine(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, mask):
            """返回输入的掩码值，为给定操作消除掩码的元素。"""
            # 保存掩码张量以备后用
            ctx.save_for_backward(mask)

            # 如果掩码不为 None，则标记为不可微分
            if mask is not None:
                ctx.mark_non_differentiable(mask)

            # 调用辅助函数处理输入和掩码
            return helper(input, mask)

        @staticmethod
        def backward(ctx, grad_output):
            # 从上下文中恢复保存的掩码
            (mask,) = ctx.saved_tensors
            # 获取梯度数据，如果是掩码张量则获取其数据
            grad_data = (
                grad_output.get_data() if is_masked_tensor(grad_output) else grad_output
            )
            # 返回作为掩码张量的结果
            result = as_masked_tensor(grad_data, mask)
            return result, None

    # 应用组合操作
    return (
        Combine.apply(input.get_data(), input.get_mask())  # type: ignore[union-attr]
        if is_masked_tensor(input)
        else helper(input, mask)
    )


# 应用文档字符串模板，计算输入的总和
def sum(
    input: Union[Tensor, MaskedTensor],
    dim: DimOrDims = None,
    *,
    keepdim: Optional[bool] = False,
    # keepdim 是一个可选的布尔类型参数，默认为 False，表示是否保持输出张量的维度。
    dtype: Optional[DType] = None,
    # dtype 是一个可选的数据类型参数，默认为 None，用于指定输出张量的数据类型。
    mask: Optional[Tensor] = None,
    # mask 是一个可选的张量类型参数，默认为 None，用于指定要应用的遮罩（掩码）。
# 返回类型为 Tensor 的函数定义
) -> Tensor:
    # 函数文档字符串由 _apply_docstring_templates 装饰器生成
    if dtype is None:
        # 当输出数据类型未指定时，将整数类型提升为 int64
        if input.layout == torch.sparse_csr:
            # 对于稀疏 CSR 格式的输入，特别处理整数类型
            if input.dtype in {
                torch.uint8,
                torch.bool,
                torch.int8,
                torch.int16,
                torch.int32,
            }:
                # 由于 csr.to(dtype=torch.int64) 方法未实现，因此通过先将输入转换为 COO 格式，再转换为 CSR 格式来确保数据类型提升
                input = input.to_sparse_coo().to(dtype=torch.int64).to_sparse_csr()
            else:
                dtype = input.dtype
        else:
            # 对于非稀疏格式的输入，保持输入数据类型不变
            dtype = input.dtype
            if input.dtype in {
                torch.uint8,
                torch.bool,
                torch.int8,
                torch.int16,
                torch.int32,
            }:
                # 如果输入数据类型为特定整数类型，则将输出数据类型设置为 int64
                dtype = torch.int64
    # 规范化维度参数，确保它符合输入张量的维度数量
    dim_ = _canonical_dim(dim, input.ndim)
    # 结合输入张量和掩码，生成掩码后的输入
    mask_input = _combine_input_and_mask(sum, input, mask)
    # 根据掩码后输入的布局类型选择不同的求和操作
    if mask_input.layout == torch.strided:
        # 对于分块连续布局的张量，进行求和操作并返回结果
        return torch.sum(mask_input, dim_, bool(keepdim), dtype=dtype)
    elif mask_input.layout == torch.sparse_coo:
        # 对于稀疏 COO 格式的张量，调用辅助函数进行稀疏矩阵的散布约简操作
        return _sparse_coo_scatter_reduction_helper(
            torch.sum, mask_input, dim_, bool(keepdim), dtype
        )
    elif mask_input.layout == torch.sparse_csr:
        # 对于稀疏 CSR 格式的张量，调用内部函数进行 CSR 格式的求和操作
        return torch._sparse_csr_sum(
            mask_input, dim=list(dim_), keepdim=bool(keepdim), dtype=dtype
        )
    else:
        # 如果输入张量布局不是预期的 strided、sparse_coo 或 sparse_csr，则引发值错误异常
        raise ValueError(
            f"masked sum expects strided, sparse_coo or sparse_csr tensor (got {mask_input.layout} tensor)"
        )


# 应用文档字符串模板的装饰器函数定义
@_apply_docstring_templates
def prod(
    input: Union[Tensor, MaskedTensor],
    dim: DimOrDims = None,
    *,
    keepdim: Optional[bool] = False,
    dtype: Optional[DType] = None,
    mask: Optional[Tensor] = None,
) -> Tensor:
    # 函数文档字符串由 _apply_docstring_templates 装饰器生成
    if dtype is None:
        # 当输出数据类型未指定时，将整数类型提升为 int64
        if input.layout == torch.sparse_csr:
            # 对于稀疏 CSR 格式的输入，特别处理整数类型
            if input.dtype in {
                torch.uint8,
                torch.bool,
                torch.int8,
                torch.int16,
                torch.int32,
            }:
                # 由于 csr.to(dtype=torch.int64) 方法未实现，因此通过先将输入转换为 COO 格式，再转换为 CSR 格式来确保数据类型提升
                input = input.to_sparse_coo().to(dtype=torch.int64).to_sparse_csr()
            else:
                dtype = input.dtype
        else:
            # 对于非稀疏格式的输入，保持输入数据类型不变
            dtype = input.dtype
            if input.dtype in {
                torch.uint8,
                torch.bool,
                torch.int8,
                torch.int16,
                torch.int32,
            }:
                # 如果输入数据类型为特定整数类型，则将输出数据类型设置为 int64
                dtype = torch.int64
    # 规范化维度参数，确保它符合输入张量的维度数量
    dim_ = _canonical_dim(dim, input.ndim)
    # 结合输入张量和掩码，生成掩码后的输入
    mask_input = _combine_input_and_mask(prod, input, mask)
    # 检查输入的张量布局是否为 strided
    if mask_input.layout == torch.strided:
        # 由于存在问题 https://github.com/pytorch/pytorch/issues/56586，需要进行以下处理
        result = mask_input
        # 将结果张量的数据类型转换为指定的 dtype
        result = result.to(dtype=dtype)
        # 对指定的维度进行反向迭代
        for d in reversed(dim_):
            # 在指定维度上计算乘积，保持维度信息由 keepdim 决定
            result = result.prod(dim=d, keepdim=bool(keepdim))
        return result
    # 检查输入的张量布局是否为 sparse_coo
    elif mask_input.layout == torch.sparse_coo:
        if mask is None:
            # 在 sparse_coo 分支中存在与 sparse_csr 分支相同的问题，详见注释
            raise ValueError(
                "masked prod expects explicit mask for sparse_coo tensor input"
            )
        # 调用辅助函数处理 sparse_coo 张量的乘积操作
        return _sparse_coo_scatter_reduction_helper(
            torch.prod, mask_input, dim_, bool(keepdim), dtype
        )
    # 检查输入的张量布局是否为 sparse_csr
    elif mask_input.layout == torch.sparse_csr:
        if mask is None:
            # 当 mask 为 None 时，对应所有元素为真的掩码。CSR 稀疏张量中未指定的元素对应值为零。
            # 因此，乘积的结果会自动为零，除非所有元素都指定了。为了半优化地考虑这一点，可以使用以下方法：
            #
            #   masked_prod(csr, ..., mask=None) == torch._sparse_csr_prod(csr, ...) * all(csr.nonzero(), ...)
            #
            # 但这需要为稀疏 csr 张量实现 `all` 和 `nonzero` 的支持。
            raise ValueError(
                "masked prod expects explicit mask for sparse_csr tensor input"
            )
        # 调用 PyTorch 的稀疏 CSR 张量乘积函数
        return torch._sparse_csr_prod(
            mask_input, dim=list(dim_), keepdim=bool(keepdim), dtype=dtype
        )
    else:
        # 如果输入的张量布局既不是 strided、sparse_coo、也不是 sparse_csr，则引发值错误异常
        raise ValueError(
            f"masked prod expects strided, sparse_coo or sparse_csr tensor (got {mask_input.layout} tensor)"
        )
# 应用文档字符串模板到函数上
@_apply_docstring_templates
# 计算输入张量沿指定维度的累积和
def cumsum(
    input: Tensor,
    dim: int,
    *,
    dtype: Optional[DType] = None,
    mask: Optional[Tensor] = None,
) -> Tensor:
    # 如果未指定数据类型，则使用输入张量的数据类型
    if dtype is None:
        dtype = input.dtype
    # 规范化维度参数，确保其有效性
    dim_ = _canonical_dim(dim, input.ndim)[0]
    # 将输入张量和掩码合并
    mask_input = _combine_input_and_mask(sum, input, mask)
    # 如果合并后的张量布局为步进型（strided）
    if mask_input.layout == torch.strided:
        # 计算累积和并转换数据类型为指定的 dtype
        return torch.cumsum(mask_input, dim_, dtype=dtype).to(dtype=dtype)
    else:
        # 如果张量布局不是步进型，则抛出数值错误
        raise ValueError(
            f"masked cumsum expects strided tensor (got {mask_input.layout} tensor)"
        )


# 应用文档字符串模板到函数上
@_apply_docstring_templates
# 计算输入张量沿指定维度的累积积
def cumprod(
    input: Tensor,
    dim: int,
    *,
    dtype: Optional[DType] = None,
    mask: Optional[Tensor] = None,
) -> Tensor:
    # 如果未指定数据类型，则使用输入张量的数据类型
    if dtype is None:
        dtype = input.dtype
    # 规范化维度参数，确保其有效性
    dim_ = _canonical_dim(dim, input.ndim)[0]
    # 将输入张量和掩码合并
    mask_input = _combine_input_and_mask(prod, input, mask)
    # 如果合并后的张量布局为步进型（strided）
    if mask_input.layout == torch.strided:
        # 计算累积积并转换数据类型为指定的 dtype
        return torch.cumprod(mask_input, dim_, dtype=dtype).to(dtype=dtype)
    else:
        # 如果张量布局不是步进型，则抛出数值错误
        raise ValueError(
            f"masked cumprod expects strided tensor (got {mask_input.layout} tensor)"
        )


# 应用文档字符串模板到函数上
@_apply_docstring_templates
# 计算输入张量或掩码张量沿指定维度的最大值
def amax(
    input: Union[Tensor, MaskedTensor],
    dim: DimOrDims = None,
    *,
    keepdim: Optional[bool] = False,
    dtype: Optional[DType] = None,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """\
    {reduction_signature}

    {reduction_descr}

    {reduction_identity_dtype}

    {reduction_args}

    {reduction_example}"""
    # 如果未指定数据类型，则使用输入张量的数据类型
    if dtype is None:
        dtype = input.dtype
    # 将输入张量和掩码合并
    mask_input = _combine_input_and_mask(amax, input, mask)
    # 规范化维度参数，确保其有效性
    dim_ = _canonical_dim(dim, mask_input.ndim)
    # 根据合并后的张量布局类型执行不同的最大值计算操作
    if mask_input.layout == torch.strided:
        # 如果是步进型布局，则计算最大值并根据 keepdim 转换形状，最后转换数据类型为指定的 dtype
        return torch.amax(mask_input, dim_, bool(keepdim)).to(dtype=dtype)
    elif mask_input.layout == torch.sparse_coo:
        # 如果是稀疏 COO 布局，且未提供掩码，则抛出数值错误
        if mask is None:
            raise ValueError(
                "masked amax expects explicit mask for sparse_coo tensor input"
            )
        # 否则，调用辅助函数处理稀疏 COO 张量的最大值计算
        return _sparse_coo_scatter_reduction_helper(
            torch.amax, mask_input, dim_, bool(keepdim), dtype
        )
    elif mask_input.layout == torch.sparse_csr:
        # 如果是稀疏 CSR 布局，且未提供掩码，则抛出数值错误
        if mask is None:
            raise ValueError(
                "masked amax expects explicit mask for sparse_csr tensor input"
            )
        # 否则，调用辅助函数处理稀疏 CSR 张量的最大值计算
        return _sparse_csr_segment_reduction_helper(
            torch.amax, mask_input, dim_, bool(keepdim), dtype
        )
    else:
        # 如果张量布局不是步进型、稀疏 COO 或稀疏 CSR，则抛出数值错误
        raise ValueError(
            f"masked amax expects strided, sparse_coo or sparse_csr tensor (got {mask_input.layout} tensor)"
        )


# 应用文档字符串模板到函数上
@_apply_docstring_templates
# 计算输入张量或掩码张量沿指定维度的最小值
def amin(
    input: Union[Tensor, MaskedTensor],
    dim: DimOrDims = None,
    *,
    keepdim: Optional[bool] = False,
    dtype: Optional[DType] = None,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """
    这是一个多行字符串的起始标记，用于创建多行注释块
    """\
{reduction_signature}
{reduction_descr}
{reduction_identity_dtype}
{reduction_args}
{reduction_example}"""
    # 如果未指定数据类型，使用输入张量的数据类型
    if dtype is None:
        dtype = input.dtype

    # 将输入张量与掩码结合，生成掩码输入
    mask_input = _combine_input_and_mask(amin, input, mask)

    # 规范化维度参数，确保其在合法范围内
    dim_ = _canonical_dim(dim, mask_input.ndim)

    # 根据掩码输入的布局类型选择相应的操作
    if mask_input.layout == torch.strided:
        # 对于布局为 strided 的张量，计算沿指定维度的最小值，并根据 keepdim 转换数据类型后返回结果
        return torch.amin(mask_input, dim_, bool(keepdim)).to(dtype=dtype)
    elif mask_input.layout == torch.sparse_coo:
        if mask is None:
            # 对于稀疏张量 sparse_coo，如果未提供掩码，则引发值错误
            # 这类似于 prod 中 sparse_csr 分支的注释，可能需要在某些维度上以结果减少未指定的元素
            raise ValueError(
                "masked amax expects explicit mask for sparse_coo tensor input"
            )
        # 使用辅助函数处理 sparse_coo 张量的带掩码最小值计算
        return _sparse_coo_scatter_reduction_helper(
            torch.amin, mask_input, dim_, bool(keepdim), dtype
        )
    elif mask_input.layout == torch.sparse_csr:
        if mask is None:
            # 对于稀疏张量 sparse_csr，如果未提供掩码，则引发值错误
            raise ValueError(
                "masked amin expects explicit mask for sparse_csr tensor input"
            )
        # 使用辅助函数处理 sparse_csr 张量的分段带掩码最小值计算
        return _sparse_csr_segment_reduction_helper(
            torch.amin, mask_input, dim_, bool(keepdim), dtype
        )
    else:
        # 对于未知布局的张量，引发值错误
        raise ValueError(
            f"masked amin expects strided, sparse_coo or sparse_csr tensor (got {mask_input.layout} tensor)"
        )


@_apply_docstring_templates
def argmax(
    input: Union[Tensor, MaskedTensor],
    dim: Optional[int] = None,
    *,
    keepdim: Optional[bool] = False,
    dtype: Optional[DType] = None,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """\
{reduction_signature}
{reduction_descr}
{reduction_identity_dtype}
{reduction_args}
{reduction_example}"""
    # 如果未指定数据类型，使用输入张量的数据类型
    if dtype is None:
        dtype = input.dtype

    # 将输入张量与掩码结合，生成掩码输入
    mask_input = _combine_input_and_mask(argmax, input, mask)

    # 根据掩码输入的布局类型选择相应的操作
    if mask_input.layout == torch.strided:
        # 对于布局为 strided 的张量，计算沿指定维度的最大值，并根据 keepdim 转换数据类型后返回结果
        return torch.argmax(mask_input, dim, bool(keepdim)).to(dtype=dtype)
    else:
        # 对于不支持的布局类型，引发值错误
        raise ValueError(
            f"masked argmax expects strided tensor (got {mask_input.layout} tensor)"
        )


@_apply_docstring_templates
def argmin(
    input: Union[Tensor, MaskedTensor],
    dim: Optional[int] = None,
    *,
    keepdim: Optional[bool] = False,
    dtype: Optional[DType] = None,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """\
{reduction_signature}
{reduction_descr}
{reduction_identity_dtype}
{reduction_args}
{reduction_example}"""
    # 如果未指定数据类型，使用输入张量的数据类型
    if dtype is None:
        dtype = input.dtype

    # 将输入张量与掩码结合，生成掩码输入
    mask_input = _combine_input_and_mask(argmin, input, mask)

    # 根据掩码输入的布局类型选择相应的操作
    if mask_input.layout == torch.strided:
        # 对于布局为 strided 的张量，计算沿指定维度的最小值，并根据 keepdim 转换数据类型后返回结果
        return torch.argmin(mask_input, dim, bool(keepdim)).to(dtype=dtype)
    else:
        # 对于不支持的布局类型，引发值错误
        raise ValueError(
            f"masked argmin expects strided tensor (got {mask_input.layout} tensor)"
        )


@_apply_docstring_templates
def mean(
    input: Union[Tensor, MaskedTensor],
    dim: DimOrDims = None,
    *,
    keepdim: Optional[bool] = False,
    dtype: Optional[DType] = None,
    mask: Optional[Tensor] = None,


    # 定义一个名为 mask 的变量，类型为 Optional[Tensor]，默认为 None
    # Optional[Tensor] 表示该变量可以是 Tensor 类型，也可以是 None（即可选的 Tensor 类型）
# 定义函数签名和文档字符串模板
def median(
    input: Union[Tensor, MaskedTensor],  # 输入参数可以是普通张量或掩码张量
    dim: int = -1,                      # 沿指定维度计算中位数，默认为最后一维
    *,
    keepdim: bool = False,              # 是否保持计算结果的维度信息，默认不保持
    dtype: Optional[DType] = None,      # 输出张量的数据类型，默认与输入相同
    mask: Optional[Tensor] = None,      # 可选的掩码张量，用于指定哪些元素参与计算
) -> Tensor:
    """\
{reduction_signature}
{reduction_descr}
By definition, the identity value of a median operation is the median
value of the tensor. If all elements of the input tensor along given
dimension(s) :attr:`dim` are masked-out, the identity value of the
median is undefined.  Due to this ambiguity, the elements of output
tensor with strided layout, that correspond to fully masked-out
elements, have ``nan`` values.
{reduction_args}
{reduction_example}"""
    # 如果未指定数据类型，则使用输入张量的数据类型
    if dtype is None:
        dtype = input.dtype
    # 获取标准化后的维度索引
    dim_ = _canonical_dim(dim, input.ndim)[0]
    # 检查输入张量是否为浮点类型
    is_float = torch.is_floating_point(input)
    # 如果不是浮点类型，则转换为浮点类型
    if not is_float:
        input = input.to(dtype=torch.float)
    # 结合输入和掩码张量，创建掩码后的输入张量
    mask_input = _combine_input_and_mask(median, input, mask)
    # 检查输入张量的布局是否为 strided
    if mask_input.layout == torch.strided:
        # 计算在指定维度上的中位数，忽略 NaN 值，返回值为一个张量
        output = torch.nanmedian(mask_input, dim_, keepdim).values
        # 如果输出是浮点数类型，则直接返回
        if is_float:
            return output
        # 如果输出不是浮点数且没有任何 NaN 值，则转换输出为指定的数据类型并返回
        elif not is_float and not torch.isnan(output).any():
            return output.to(dtype=dtype)
        # 如果输出包含 NaN 值且不是浮点数类型，则抛出值错误异常
        else:
            raise ValueError(
                "masked median expects no fully masked out rows if dtype is not floating point"
            )
    else:
        # 如果输入张量的布局不是 strided，则抛出值错误异常，指出期望的张量布局类型
        raise ValueError(
            f"masked median expects strided tensor (got {mask_input.layout} tensor)"
        )
# 应用文档字符串模板到 logsumexp 函数
@_apply_docstring_templates
# 计算输入张量的 logsumexp 操作
def logsumexp(
    input: Tensor,
    dim: DimOrDims = None,
    *,
    keepdim: bool = False,
    dtype: Optional[DType] = None,
    mask: Optional[Tensor] = None,
) -> Tensor:
    # 如果未指定数据类型，则使用输入张量的数据类型
    if dtype is None:
        dtype = input.dtype
    # 规范化维度参数
    dim_ = _canonical_dim(dim, input.ndim)
    # 合并输入张量和掩码张量
    mask_input = _combine_input_and_mask(logsumexp, input, mask)
    # 如果合并后的张量布局是 strided，则进行 logsumexp 操作，并转换数据类型为指定类型
    if mask_input.layout == torch.strided:
        return torch.logsumexp(mask_input, dim_, keepdim=keepdim).to(dtype=dtype)
    else:
        # 如果合并后的张量布局不是 strided，则抛出 ValueError 异常
        raise ValueError(
            f"masked logsumexp expects strided tensor (got {mask_input.layout} tensor)"
        )


# 由于 _apply_docstring_templates 仅设置用于规范化和归一化，logaddexp 函数不能使用该装饰器
def logaddexp(
    input: Union[Tensor, MaskedTensor],
    other: Union[Tensor, MaskedTensor],
    *,
    dtype: Optional[DType] = None,
    input_mask: Optional[Tensor] = None,
    other_mask: Optional[Tensor] = None,
) -> Tensor:
    """logaddexp(input, other, *, dtype=None, input_mask=None, other_mask=None) -> Tensor

    返回输入张量和另一个张量的 logaddexp。input 的元素根据布尔张量 input_mask 进行掩码，
    other 的元素根据布尔张量 other_mask 进行掩码。

    掩码张量和被掩码张量的形状不需要匹配，但它们必须是可广播的，并且掩码张量的维度不能大于被掩码张量的维度。

    Args:
        input (Tensor): 输入张量
        other (Tensor): 第二个输入张量

    Keyword args:
        dtype (:class:`torch.dtype`, optional): 返回张量的期望数据类型。
          如果指定，操作完成后将转换为 :attr:`dtype`。默认值: None.
        input_mask (:class:`torch.Tensor`, optional): 包含 :attr:`input` 张量元素有效性的二进制掩码张量。
          默认值: None，相当于 ``torch.ones(input.shape, dtype=torch.bool)``.
        other_mask (:class:`torch.Tensor`, optional): 包含 :attr:`other` 张量元素有效性的二进制掩码张量。
          默认值: None，相当于 ``torch.ones(other.shape, dtype=torch.bool)``.

    Example::

        >>> input = torch.tensor([-100.0, -200, -300])
        >>> input
        tensor([-100., -200., -300.])
        >>> other = torch.tensor([-1.0, -2, -3])
        >>> other
        tensor([-1., -2., -3.])
        >>> mask = torch.tensor([True, False, True])
        >>> mask
        tensor([ True, False,  True])
        >>> torch.masked._ops.logaddexp(input, other, input_mask=mask, other_mask=mask)
        tensor([-1., -inf, -3.])"""
    # 如果未指定数据类型，则使用输入张量的数据类型
    if dtype is None:
        dtype = input.dtype
    # 检查输入和其他张量的布局是否为 torch.strided
    if input.layout == torch.strided and other.layout == torch.strided:
        # 使用 _combine_input_and_mask 函数将 logsumexp、input 和 input_mask 组合
        mask_input = _combine_input_and_mask(logsumexp, input, input_mask)
        # 使用 _combine_input_and_mask 函数将 logsumexp、other 和 other_mask 组合
        mask_other = _combine_input_and_mask(logsumexp, other, other_mask)
        # 对组合后的张量应用 torch.logaddexp 运算，并转换为指定的数据类型
        return torch.logaddexp(mask_input, mask_other).to(dtype=dtype)
    else:
        # 如果输入和其他张量的布局不是 torch.strided，则抛出 ValueError 异常
        raise ValueError(
            f"masked logaddexp expects strided tensors (got {input.layout} tensor for input, {other.layout} for other)"
        )
# 应用文档字符串模板到 norm 函数上
@_apply_docstring_templates
# 定义 norm 函数，计算向量或者 MaskedTensor 的范数
def norm(
    input: Union[Tensor, MaskedTensor],  # 输入参数可以是 Tensor 或 MaskedTensor
    ord: Optional[float] = 2.0,           # 范数的阶数，默认为 2.0
    dim: DimOrDims = None,                # 指定计算范数的维度，默认为 None
    *,                                    # 以下参数必须使用关键字传递
    keepdim: Optional[bool] = False,      # 是否保持计算结果的维度信息，默认为 False
    dtype: Optional[DType] = None,        # 指定计算结果的数据类型，默认为 None
    mask: Optional[Tensor] = None,        # 用于屏蔽计算的掩码张量，默认为 None
) -> Tensor:                             # 函数返回值为 Tensor 类型
    """\
{reduction_signature}

{reduction_descr}

The identity value of norm operation, which is used to start the
reduction, is ``{identity_float32}``, except for ``ord=-inf`` it is
``{identity_ord_ninf}``.

{reduction_args}

{reduction_example}"""
    if dtype is None:
        dtype = input.dtype  # 如果未指定 dtype，则使用输入的数据类型
    mask_input = _combine_input_and_mask(norm, input, mask, ord)  # 组合输入和掩码
    if mask_input.layout == torch.strided:  # 检查掩码输入是否为 strided 布局
        dim_ = _canonical_dim(dim, input.ndim)  # 规范化维度参数
        return torch.linalg.vector_norm(
            mask_input, ord, dim_, bool(keepdim), dtype=dtype  # 计算向量范数
        )
    else:
        raise ValueError(
            f"masked norm expects strided tensor (got {mask_input.layout} tensor)"  # 抛出布局错误异常
        )


# 定义 _std_var 函数，计算标准差或方差
def _std_var(
    input: Union[Tensor, MaskedTensor],      # 输入参数可以是 Tensor 或 MaskedTensor
    dim: DimOrDims,                          # 指定计算标准差或方差的维度
    unbiased: Optional[bool],                # 是否使用无偏估计
    *,                                       # 以下参数必须使用关键字传递
    correction_opt: Optional[Union[int, float]],  # 校正参数，可选整数或浮点数
    keepdim: Optional[bool],                 # 是否保持计算结果的维度信息
    dtype: Optional[DType],                  # 指定计算结果的数据类型
    mask: Optional[Tensor],                  # 用于屏蔽计算的掩码张量
    take_sqrt: Optional[bool],               # 是否对结果取平方根
) -> Tensor:                                # 函数返回值为 Tensor 类型
    assert (
        unbiased is None or correction_opt is None
    ), "Only one of unbiased and correction may be given"  # 断言只能给定 unbiased 或 correction_opt 中的一个
    correction = 1.0
    if unbiased is not None:
        correction = 1.0 if unbiased else 0.0  # 根据 unbiased 设置修正值
    if correction_opt is not None:
        correction = sym_float(correction_opt)  # 使用 sym_float 函数处理校正参数

    if dtype is None:
        dtype = input.dtype  # 如果未指定 dtype，则使用输入的数据类型
        if not (dtype.is_floating_point or dtype.is_complex):
            dtype = torch.float32  # 如果数据类型不是浮点型或复数型，则使用 float32

    compute_dtype = dtype
    if not (compute_dtype.is_floating_point or compute_dtype.is_complex):
        compute_dtype = torch.float32  # 如果计算数据类型不是浮点型或复数型，则使用 float32
    # 如果输入的张量布局为 strided
    if input.layout == torch.strided:
        # 如果没有提供掩码 mask
        if mask is None:
            # TODO: 通过解析计算 count
            # 计算张量 input 的元素个数，按指定维度 dim 求和，保持维度不变
            count = sum(
                torch.ones(input.shape, dtype=torch.int64, device=input.device),
                dim,
                keepdim=True,
            )
            # 计算 input 指定维度 dim 上的总和，保持维度不变，使用指定的数据类型 dtype
            sample_total = sum(input, dim, keepdim=True, dtype=dtype)
        else:
            # 根据掩码 mask 创建输入的掩码 inmask
            inmask = _input_mask(input, mask=mask)
            # 使用掩码创建张量 input 形状的新张量，元素为 1，数据类型为 torch.int64
            count = sum(
                inmask.new_ones(input.shape, dtype=torch.int64),
                dim,
                keepdim=True,
                mask=inmask,
            )
            # 计算 input 在掩码 inmask 指定维度 dim 上的总和，保持维度不变，使用指定的数据类型 dtype
            sample_total = sum(input, dim, keepdim=True, dtype=dtype, mask=inmask)
        
        # TODO: 当可以使用掩码版本的 subtract/divide/square/maximum 时，替换为掩码版本
        # 计算样本均值，即 sample_total 除以 count
        sample_mean = torch.divide(sample_total, count)
        
        # 将 input 减去样本均值，得到 x
        x = torch.subtract(input, sample_mean)
        
        # 如果没有提供掩码 mask
        if mask is None:
            # 计算 x 与其共轭的乘积在指定维度 dim 上的总和，保持维度 keepdim，数据类型为 compute_dtype
            total = sum(x * x.conj(), dim, keepdim=keepdim, dtype=compute_dtype)
        else:
            # 使用掩码 inmask 计算 x 与其共轭的乘积在指定维度 dim 上的总和，保持维度 keepdim，数据类型为 compute_dtype
            total = sum(
                x * x.conj(), dim, keepdim=keepdim, dtype=compute_dtype, mask=inmask  # type: ignore[possibly-undefined]
            )
        
        # 如果不保持维度 keepdim
        if not keepdim:
            # 将 count 的形状重塑为 total 的形状
            count = count.reshape(total.shape)
        
        # 如果修正值 correction 不为 0
        if correction != 0:
            # 根据 compute_dtype 类型获取其对应的实部数据类型
            real_dtype = (
                corresponding_real_dtype(compute_dtype)
                if compute_dtype.is_complex
                else compute_dtype
            )
            # 将 count 转换为实部数据类型
            count = count.to(real_dtype)
            # count 减去修正值 correction
            count = torch.subtract(count, correction)
            # count 取 count 和零元素张量的每个元素的最大值
            count = torch.maximum(count, count.new_zeros([]))
        
        # 计算最终的输出，即 total 除以 count，并转换为数据类型 dtype
        output = torch.divide(total, count).to(dtype=dtype)
        
        # 如果需要对输出进行开方处理
        if take_sqrt:
            output = torch.sqrt(output)
        
        # 返回计算得到的输出
        return output
    else:
        # 如果输入的张量布局不是 strided，则抛出 ValueError 异常
        raise ValueError(
            f"masked std/var expects strided tensor (got {input.layout} tensor)"
        )
@_apply_docstring_templates
def var(
    input: Union[Tensor, MaskedTensor],
    dim: DimOrDims = None,
    unbiased: Optional[bool] = None,
    *,
    correction: Optional[Union[int, float]] = None,
    keepdim: Optional[bool] = False,
    dtype: Optional[DType] = None,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """\
{reduction_signature}
{reduction_descr}
The identity value of sample variance operation is undefined. The
elements of output tensor with strided layout, that correspond to
fully masked-out elements, have ``nan`` values.
{reduction_args}
{reduction_example}"""
    # 调用 _std_var 函数计算方差，不进行平方根计算
    return _std_var(
        input=input,
        dim=dim,
        unbiased=unbiased,
        correction_opt=correction,
        keepdim=keepdim,
        dtype=dtype,
        mask=mask,
        take_sqrt=False,
    )


@_apply_docstring_templates
def std(
    input: Union[Tensor, MaskedTensor],
    dim: DimOrDims = None,
    unbiased: Optional[bool] = None,
    *,
    correction: Optional[int] = None,
    keepdim: Optional[bool] = False,
    dtype: Optional[DType] = None,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """\
{reduction_signature}
{reduction_descr}
The identity value of sample standard deviation operation is undefined. The
elements of output tensor with strided layout, that correspond to
fully masked-out elements, have ``nan`` values.
{reduction_args}
{reduction_example}"""
    # 调用 _std_var 函数计算标准差，进行平方根计算
    return _std_var(
        input=input,
        dim=dim,
        unbiased=unbiased,
        correction_opt=correction,
        keepdim=keepdim,
        dtype=dtype,
        mask=mask,
        take_sqrt=True,
    )


@_apply_docstring_templates
def softmax(
    input: Union[Tensor, MaskedTensor],
    dim: int,
    *,
    dtype: Optional[DType] = None,
    mask: Optional[Tensor] = None,
) -> Tensor:
    if dtype is None:
        dtype = input.dtype
    # 规范化维度参数
    dim_ = _canonical_dim(dim, input.ndim)[0]
    # 将输入和掩码合并，以处理掩码
    mask_input = _combine_input_and_mask(amax, input, mask)
    # 检查输入的布局是否是 strided，如果是则调用 PyTorch 的 softmax 函数
    if mask_input.layout == torch.strided:
        return torch.nn.functional.softmax(mask_input, dim_, dtype=dtype)
    else:
        # 如果输入布局不是 strided，则抛出异常
        raise ValueError(
            f"masked softmax expects strided tensor (got {mask_input.layout} tensor)"
        )


@_apply_docstring_templates
def log_softmax(
    input: Union[Tensor, MaskedTensor],
    dim: int,
    *,
    dtype: Optional[DType] = None,
    mask: Optional[Tensor] = None,
) -> Tensor:
    if dtype is None:
        dtype = input.dtype
    dim_ = _canonical_dim(dim, input.ndim)[0]
    mask_input = _combine_input_and_mask(amax, input, mask)
    if mask_input.layout == torch.strided:
        return torch.nn.functional.log_softmax(mask_input, dim_, dtype=dtype)
    else:
        raise ValueError(
            f"masked log_softmax expects strided tensor (got {mask_input.layout} tensor)"
        )


@_apply_docstring_templates
def softmin(
    input: Union[Tensor, MaskedTensor],
    dim: int,
    *,
    dtype: Optional[DType] = None,
    mask: Optional[Tensor] = None,
) -> Tensor:
    # 如果未指定 dtype，则使用输入的 dtype
    if dtype is None:
        dtype = input.dtype
    # 规范化维度参数
    dim_ = _canonical_dim(dim, input.ndim)[0]
    # 将输入和掩码合并，以处理掩码
    mask_input = _combine_input_and_mask(amax, input, mask)
    # 检查输入的布局是否是 strided，如果是则调用 PyTorch 的 softmin 函数
    if mask_input.layout == torch.strided:
        return torch.nn.functional.softmax(mask_input, dim_, dtype=dtype)
    else:
        # 如果输入布局不是 strided，则抛出异常
        raise ValueError(
            f"masked softmax expects strided tensor (got {mask_input.layout} tensor)"
        )
    # 如果未指定数据类型(dtype)，则使用输入张量(input)的数据类型
    if dtype is None:
        dtype = input.dtype
    
    # 规范化维度(dim)，确保维度(dim)参数是有效的，并返回规范化后的维度值
    dim_ = _canonical_dim(dim, input.ndim)[0]
    
    # 将最小值(amin)、输入张量(input)和掩码(mask)组合成一个新的输入对象(mask_input)
    mask_input = _combine_input_and_mask(amin, input, mask)
    
    # 检查组合后的输入对象(mask_input)的布局是否为 torch.strided
    if mask_input.layout == torch.strided:
        # 如果是 strided 布局，则使用 torch.nn.functional.softmin 进行 softmin 操作
        return torch.nn.functional.softmin(mask_input, dim_, dtype=dtype)
    else:
        # 如果不是 strided 布局，则抛出 ValueError 异常
        raise ValueError(
            f"masked softmin expects strided tensor (got {mask_input.layout} tensor)"
        )
@_apply_docstring_templates
def normalize(
    input: Union[Tensor, MaskedTensor],  # 函数接受一个输入参数，可以是普通张量或者带掩码的张量
    ord: float,  # 指定规范化的阶数
    dim: int,  # 指定进行规范化操作的维度
    *,
    eps: float = 1e-12,  # 规范化操作中的小常数，用于避免除以零
    dtype: Optional[DType] = None,  # 输出张量的数据类型，可选
    mask: Optional[Tensor] = None,  # 可选的掩码张量，用于指定哪些元素参与规范化
) -> Tensor:
    if dtype is None:
        dtype = input.dtype  # 如果未指定输出数据类型，则使用输入张量的数据类型
    dim_ = _canonical_dim(dim, input.ndim)[0]  # 规范化维度参数
    # TODO: eliminate mask_input as unnecessary when using masked divide.
    mask_input = _combine_input_and_mask(sum, input, mask)  # 将输入张量和掩码合并为一个张量
    if mask_input.layout == torch.strided:  # 如果合并后的张量布局为 strided
        nrm_ = norm(input, ord, dim, keepdim=True, dtype=dtype, mask=mask)  # 计算规范化的范数
        # TODO: replace torch.maximum with masked maximum when available.
        denom = torch.maximum(nrm_, nrm_.new_full([], eps))  # 计算规范化的分母，避免除以零
        # TODO: replace torch.divide with masked divide when available.
        return torch.divide(mask_input, denom)  # 返回规范化后的张量
    else:
        raise ValueError(
            f"masked normalize expects strided tensor (got {mask_input.layout} tensor)"
        )  # 如果合并后的张量布局不是 strided，则抛出 ValueError 异常
```