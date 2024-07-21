# `.\pytorch\torch\testing\_comparison.py`

```
# mypy: allow-untyped-defs
# 引入必要的模块和库
import abc  # Abstract Base Classes (ABC) 模块
import cmath  # 复数数学函数库
import collections.abc  # 标准集合抽象基类
import contextlib  # 用于创建和管理上下文的工具
from typing import (
    Any,  # 泛型类型声明
    Callable,  # 可调用对象类型声明
    Collection,  # 集合类型声明
    Dict,  # 字典类型声明
    List,  # 列表类型声明
    NoReturn,  # 永不返回类型声明
    Optional,  # 可选类型声明
    Sequence,  # 序列类型声明
    Tuple,  # 元组类型声明
    Type,  # 类型对象类型声明
    Union,  # 联合类型声明
)
from typing_extensions import deprecated  # 引入已弃用的类型声明扩展

import torch  # 引入 PyTorch 库


try:
    import numpy as np  # 尝试引入 NumPy 库

    HAS_NUMPY = True
except ModuleNotFoundError:
    HAS_NUMPY = False
    np = None  # 若未找到 NumPy 模块，则设为 None，并忽略类型检查


class ErrorMeta(Exception):
    """Internal testing exception that makes that carries error metadata."""

    def __init__(
        self, type: Type[Exception], msg: str, *, id: Tuple[Any, ...] = ()
    ) -> None:
        # 初始化异常对象，设置异常类型、消息和可选的 ID
        super().__init__(
            "If you are a user and see this message during normal operation "
            "please file an issue at https://github.com/pytorch/pytorch/issues. "
            "If you are a developer and working on the comparison functions, please `raise ErrorMeta.to_error()` "
            "for user facing errors."
        )
        self.type = type  # 异常类型
        self.msg = msg  # 异常消息
        self.id = id  # 异常的标识信息，为元组类型

    def to_error(
        self, msg: Optional[Union[str, Callable[[str], str]]] = None
    ) -> Exception:
        # 转换为用户可见的异常对象，支持自定义错误消息的生成或传入
        if not isinstance(msg, str):
            generated_msg = self.msg
            if self.id:
                generated_msg += f"\n\nThe failure occurred for item {''.join(str([item]) for item in self.id)}"

            msg = msg(generated_msg) if callable(msg) else generated_msg

        return self.type(msg)


# Some analysis of tolerance by logging tests from test_torch.py can be found in
# https://github.com/pytorch/pytorch/pull/32538.
# {dtype: (rtol, atol)}
_DTYPE_PRECISIONS = {
    torch.float16: (0.001, 1e-5),  # torch.float16 类型的相对误差和绝对误差
    torch.bfloat16: (0.016, 1e-5),  # torch.bfloat16 类型的相对误差和绝对误差
    torch.float32: (1.3e-6, 1e-5),  # torch.float32 类型的相对误差和绝对误差
    torch.float64: (1e-7, 1e-7),  # torch.float64 类型的相对误差和绝对误差
    torch.complex32: (0.001, 1e-5),  # torch.complex32 类型的相对误差和绝对误差
    torch.complex64: (1.3e-6, 1e-5),  # torch.complex64 类型的相对误差和绝对误差
    torch.complex128: (1e-7, 1e-7),  # torch.complex128 类型的相对误差和绝对误差
}
# 对于量化数据类型（quantized dtypes），使用 torch.float32 类型的默认误差作为其误差值
_DTYPE_PRECISIONS.update(
    dict.fromkeys(
        (torch.quint8, torch.quint2x4, torch.quint4x2, torch.qint8, torch.qint32),
        _DTYPE_PRECISIONS[torch.float32],
    )
)


def default_tolerances(
    *inputs: Union[torch.Tensor, torch.dtype],
    dtype_precisions: Optional[Dict[torch.dtype, Tuple[float, float]]] = None,
) -> Tuple[float, float]:
    """Returns the default absolute and relative testing tolerances for a set of inputs based on the dtype.

    See :func:`assert_close` for a table of the default tolerance for each dtype.

    Args:
        *inputs: 一组 PyTorch 张量或数据类型
        dtype_precisions: 可选参数，指定数据类型的精度字典

    Returns:
        (Tuple[float, float]): 所有输入数据类型中最宽松的绝对误差和相对误差
    """
    dtypes = []
    # 遍历输入列表中的每一个元素
    for input in inputs:
        # 检查当前元素是否为 torch.Tensor 类型
        if isinstance(input, torch.Tensor):
            # 如果是 tensor 类型，获取其数据类型并添加到 dtypes 列表中
            dtypes.append(input.dtype)
        # 如果当前元素是 torch.dtype 类型
        elif isinstance(input, torch.dtype):
            # 直接将其添加到 dtypes 列表中
            dtypes.append(input)
        else:
            # 如果元素既不是 tensor 类型也不是 dtype 类型，则引发类型错误异常
            raise TypeError(
                f"Expected a torch.Tensor or a torch.dtype, but got {type(input)} instead."
            )
    
    # 如果未提供 dtype_precisions，则使用默认的 _DTYPE_PRECISIONS
    dtype_precisions = dtype_precisions or _DTYPE_PRECISIONS
    
    # 获取每种数据类型对应的相对容差(rtols)和绝对容差(atols)，若无对应关系则使用默认值 (0.0, 0.0)
    rtols, atols = zip(*[dtype_precisions.get(dtype, (0.0, 0.0)) for dtype in dtypes])
    
    # 返回容差中的最大相对容差和最大绝对容差
    return max(rtols), max(atols)
# 定义函数 get_tolerances，用于获取用于数值比较的绝对容差和相对容差
def get_tolerances(
    *inputs: Union[torch.Tensor, torch.dtype],  # 接受多个参数，类型为 torch.Tensor 或 torch.dtype
    rtol: Optional[float],  # 相对容差，可选参数
    atol: Optional[float],  # 绝对容差，可选参数
    id: Tuple[Any, ...] = (),  # 标识符元组，默认为空元组
) -> Tuple[float, float]:  # 返回一个元组，包含有效的绝对容差和相对容差
    """Gets absolute and relative to be used for numeric comparisons.

    If both ``rtol`` and ``atol`` are specified, this is a no-op. If both are not specified, the return value of
    :func:`default_tolerances` is used.

    Raises:
        ErrorMeta: With :class:`ValueError`, if only ``rtol`` or ``atol`` is specified.

    Returns:
        (Tuple[float, float]): Valid absolute and relative tolerances.
    """
    if (rtol is None) ^ (atol is None):  # 如果只有一个容差值被指定而另一个未被指定，引发错误
        raise ErrorMeta(
            ValueError,
            f"Both 'rtol' and 'atol' must be either specified or omitted, "
            f"but got no {'rtol' if rtol is None else 'atol'}.",
            id=id,
        )
    elif rtol is not None and atol is not None:  # 如果同时指定了 rtol 和 atol，则直接返回这两个值
        return rtol, atol
    else:  # 否则调用 default_tolerances 函数获取默认的容差值并返回
        return default_tolerances(*inputs)


def _make_mismatch_msg(
    *,
    default_identifier: str,  # 默认的比较值描述
    identifier: Optional[Union[str, Callable[[str], str]]] = None,  # 可选的自定义描述或生成器
    extra: Optional[str] = None,  # 可选的额外信息
    abs_diff: float,  # 绝对差值
    abs_diff_idx: Optional[Union[int, Tuple[int, ...]]] = None,  # 可选的绝对差值的索引
    atol: float,  # 允许的绝对容差
    rel_diff: float,  # 相对差值
    rel_diff_idx: Optional[Union[int, Tuple[int, ...]]] = None,  # 可选的相对差值的索引
    rtol: float,  # 允许的相对容差
) -> str:  # 返回生成的错误消息字符串
    """Makes a mismatch error message for numeric values.

    Args:
        default_identifier (str): Default description of the compared values, e.g. "Tensor-likes".
        identifier (Optional[Union[str, Callable[[str], str]]]): Optional identifier that overrides
            ``default_identifier``. Can be passed as callable in which case it will be called with
            ``default_identifier`` to create the description at runtime.
        extra (Optional[str]): Extra information to be placed after the message header and the mismatch statistics.
        abs_diff (float): Absolute difference.
        abs_diff_idx (Optional[Union[int, Tuple[int, ...]]]): Optional index of the absolute difference.
        atol (float): Allowed absolute tolerance. Will only be added to mismatch statistics if it or ``rtol`` are
            ``> 0``.
        rel_diff (float): Relative difference.
        rel_diff_idx (Optional[Union[int, Tuple[int, ...]]]): Optional index of the relative difference.
        rtol (float): Allowed relative tolerance. Will only be added to mismatch statistics if it or ``atol`` are
            ``> 0``.
    """
    equality = rtol == 0 and atol == 0  # 检查是否使用严格相等比较

    def make_diff_msg(
        *,
        type: str,  # 描述差异类型的字符串
        diff: float,  # 差异值
        idx: Optional[Union[int, Tuple[int, ...]]],  # 可选的差异值索引
        tol: float,  # 允许的容差值

        # 返回生成的差异消息字符串
        return ...
    ) -> str:
        # 如果 idx 参数为 None，则创建消息来表示类型和差异
        if idx is None:
            msg = f"{type.title()} difference: {diff}"
        else:
            # 否则创建消息来表示最大差异的类型、差异值和索引
            msg = f"Greatest {type} difference: {diff} at index {idx}"
        # 如果 equality 为 False，则在消息末尾添加允许的容差信息
        if not equality:
            msg += f" (up to {tol} allowed)"
        # 返回最终消息并换行
        return msg + "\n"

    # 如果 identifier 为 None，则使用默认的 identifier
    if identifier is None:
        identifier = default_identifier
    # 如果 identifier 是可调用对象，则使用它来计算新的 identifier
    elif callable(identifier):
        identifier = identifier(default_identifier)

    # 创建消息来表示不相等或者不接近的情况
    msg = f"{identifier} are not {'equal' if equality else 'close'}!\n\n"

    # 如果有额外的信息，则添加到消息中
    if extra:
        msg += f"{extra.strip()}\n"

    # 创建绝对差异的消息并添加到主消息中
    msg += make_diff_msg(type="absolute", diff=abs_diff, idx=abs_diff_idx, tol=atol)
    # 创建相对差异的消息并添加到主消息中
    msg += make_diff_msg(type="relative", diff=rel_diff, idx=rel_diff_idx, tol=rtol)

    # 返回去除末尾空白的最终消息
    return msg.strip()
def make_scalar_mismatch_msg(
    actual: Union[bool, int, float, complex],
    expected: Union[bool, int, float, complex],
    *,
    rtol: float,
    atol: float,
    identifier: Optional[Union[str, Callable[[str], str]]] = None,
) -> str:
    """Makes a mismatch error message for scalars.

    Args:
        actual (Union[bool, int, float, complex]): The actual scalar value.
        expected (Union[bool, int, float, complex]): The expected scalar value.
        rtol (float): Relative tolerance for comparison.
        atol (float): Absolute tolerance for comparison.
        identifier (Optional[Union[str, Callable[[str], str]]]): Optional identifier for the scalars,
            can be a string or a callable returning a string. Defaults to "Scalars".

    Returns:
        str: The formatted error message describing the scalar mismatch.
    """
    # 计算实际值和期望值之间的绝对差异
    abs_diff = abs(actual - expected)
    # 计算相对差异，如果期望值为零，设置为无穷大避免除以零错误
    rel_diff = float("inf") if expected == 0 else abs_diff / abs(expected)
    # 调用内部函数生成错误消息，并返回结果
    return _make_mismatch_msg(
        default_identifier="Scalars",
        identifier=identifier,
        extra=f"Expected {expected} but got {actual}.",
        abs_diff=abs_diff,
        atol=atol,
        rel_diff=rel_diff,
        rtol=rtol,
    )


def make_tensor_mismatch_msg(
    actual: torch.Tensor,
    expected: torch.Tensor,
    matches: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    identifier: Optional[Union[str, Callable[[str], str]]] = None,
):
    """Makes a mismatch error message for tensors.

    Args:
        actual (torch.Tensor): The actual tensor.
        expected (torch.Tensor): The expected tensor.
        matches (torch.Tensor): Boolean mask indicating where matches occur.
        rtol (float): Relative tolerance for comparison.
        atol (float): Absolute tolerance for comparison.
        identifier (Optional[Union[str, Callable[[str], str]]]): Optional identifier for the tensors,
            can be a string or a callable returning a string. Defaults to "Tensor-likes".
    """

    def unravel_flat_index(flat_index: int) -> Tuple[int, ...]:
        """Converts a flattened index into the multi-dimensional index based on matches shape.

        Args:
            flat_index (int): Flattened index to unravel.

        Returns:
            Tuple[int, ...]: Multi-dimensional index corresponding to the flattened index in matches.
        """
        # 如果 matches 的形状为空，返回空元组
        if not matches.shape:
            return ()

        # 初始化反向索引列表
        inverse_index = []
        # 反向遍历 matches 形状，将平面索引转换为多维索引
        for size in matches.shape[::-1]:
            div, mod = divmod(flat_index, size)
            flat_index = div
            inverse_index.append(mod)

        # 返回转换后的多维索引
        return tuple(inverse_index[::-1])

    # 计算总元素数和不匹配的元素数
    number_of_elements = matches.numel()
    total_mismatches = number_of_elements - int(torch.sum(matches))
    # 生成额外的错误信息，描述不匹配的元素数量及比例
    extra = (
        f"Mismatched elements: {total_mismatches} / {number_of_elements} "
        f"({total_mismatches / number_of_elements:.1%})"
    )

    # 将实际值、期望值和匹配情况展平为一维张量
    actual_flat = actual.flatten()
    expected_flat = expected.flatten()
    matches_flat = matches.flatten()
    # 检查实际值和期望值的数据类型是否为浮点数或复数，如果不是，则执行以下操作
    if not actual.dtype.is_floating_point and not actual.dtype.is_complex:
        # TODO: 不必总是将数据类型提升到 int64，可以提升到下一个更高的数据类型，以避免溢出
        actual_flat = actual_flat.to(torch.int64)
        expected_flat = expected_flat.to(torch.int64)

    # 计算实际值与期望值的绝对差值
    abs_diff = torch.abs(actual_flat - expected_flat)
    # 确保仅使用不匹配的值来计算最大绝对差值
    abs_diff[matches_flat] = 0
    # 计算绝对差值的最大值及其索引
    max_abs_diff, max_abs_diff_flat_idx = torch.max(abs_diff, 0)

    # 计算相对差值，即绝对差值除以期望值的绝对值
    rel_diff = abs_diff / torch.abs(expected_flat)
    # 确保仅使用不匹配的值来计算最大相对差值
    rel_diff[matches_flat] = 0
    # 计算相对差值的最大值及其索引
    max_rel_diff, max_rel_diff_flat_idx = torch.max(rel_diff, 0)

    # 调用 _make_mismatch_msg 函数，生成描述不匹配情况的消息
    return _make_mismatch_msg(
        default_identifier="Tensor-likes",
        identifier=identifier,
        extra=extra,
        abs_diff=max_abs_diff.item(),  # 最大绝对差值
        abs_diff_idx=unravel_flat_index(int(max_abs_diff_flat_idx)),  # 最大绝对差值的索引
        atol=atol,
        rel_diff=max_rel_diff.item(),  # 最大相对差值
        rel_diff_idx=unravel_flat_index(int(max_rel_diff_flat_idx)),  # 最大相对差值的索引
        rtol=rtol,
    )
class UnsupportedInputs(Exception):  # 定义一个自定义异常类UnsupportedInputs，用于在构造Pair类时处理不支持的输入情况
    """Exception to be raised during the construction of a :class:`Pair` in case it doesn't support the inputs."""

class Pair(abc.ABC):
    """ABC for all comparison pairs to be used in conjunction with :func:`assert_equal`.

    Each subclass needs to overwrite :meth:`Pair.compare` that performs the actual comparison.

    Each pair receives **all** options, so select the ones applicable for the subclass and forward the rest to the
    super class. Raising an :class:`UnsupportedInputs` during constructions indicates that the pair is not able to
    handle the inputs and the next pair type will be tried.

    All other errors should be raised as :class:`ErrorMeta`. After the instantiation, :meth:`Pair._make_error_meta` can
    be used to automatically handle overwriting the message with a user supplied one and id handling.
    """

    def __init__(
        self,
        actual: Any,
        expected: Any,
        *,
        id: Tuple[Any, ...] = (),
        **unknown_parameters: Any,
    ) -> None:
        """Initialize a Pair object with actual and expected values, and additional optional parameters.

        Args:
            actual: The actual value for comparison.
            expected: The expected value for comparison.
            id: Optional identifier for the pair.
            **unknown_parameters: Additional parameters passed to the constructor.
        """
        self.actual = actual  # 存储实际值
        self.expected = expected  # 存储期望值
        self.id = id  # 存储标识符
        self._unknown_parameters = unknown_parameters  # 存储未知的额外参数

    @staticmethod
    def _inputs_not_supported() -> NoReturn:
        """Raise UnsupportedInputs exception indicating that the inputs are not supported."""
        raise UnsupportedInputs

    @staticmethod
    def _check_inputs_isinstance(*inputs: Any, cls: Union[Type, Tuple[Type, ...]]):
        """Checks if all inputs are instances of a given class and raise UnsupportedInputs otherwise.

        Args:
            inputs: Inputs to check.
            cls: Class or tuple of classes that inputs should be instances of.
        """
        if not all(isinstance(input, cls) for input in inputs):
            Pair._inputs_not_supported()

    def _fail(
        self, type: Type[Exception], msg: str, *, id: Tuple[Any, ...] = ()
    ) -> NoReturn:
        """Raises an ErrorMeta from a given exception type and message and the stored id.

        Args:
            type: Type of exception to raise.
            msg: Error message.
            id: Optional identifier for the error.

        Raises:
            ErrorMeta: Exception containing type, message, and id.
        """
        raise ErrorMeta(type, msg, id=self.id if not id and hasattr(self, "id") else id)

    @abc.abstractmethod
    def compare(self) -> None:
        """Compares the inputs and raises an ErrorMeta in case they mismatch.

        This method should be implemented in subclasses to perform specific comparison logic.
        """

    def extra_repr(self) -> Sequence[Union[str, Tuple[str, Any]]]:
        """Returns extra information that will be included in the representation.

        Should be overwritten by all subclasses that use additional options. The representation of the object will only
        be surfaced in case we encounter an unexpected error and thus should help debug the issue. Can be a sequence of
        key-value-pairs or attribute names.

        Returns:
            Sequence[Union[str, Tuple[str, Any]]]: Extra information for object representation.
        """
        return []
    # 返回对象的字符串表示形式，用于调试和显示
    def __repr__(self) -> str:
        # 创建字符串的头部，表示对象类型
        head = f"{type(self).__name__}("
        # 创建字符串的尾部
        tail = ")"
        # 创建包含对象属性的列表
        body = [
            # 格式化每个属性的名称和值到字符串中
            f"    {name}={value!s},"
            # 迭代属性名和值的元组列表
            for name, value in [
                ("id", self.id),  # 对象的 id 属性
                ("actual", self.actual),  # 对象的 actual 属性
                ("expected", self.expected),  # 对象的 expected 属性
                *[
                    (extra, getattr(self, extra)) if isinstance(extra, str) else extra
                    # 调用额外表示方法获取额外属性的名称和值，如果额外属性是字符串
                    for extra in self.extra_repr()
                ],
            ]
        ]
        # 返回组装好的字符串表示形式，包括头部、属性列表和尾部
        return "\n".join((head, *body, *tail))
class ObjectPair(Pair):
    """Pair for any type of inputs that will be compared with the `==` operator.

    .. note::

        Since this will instantiate for any kind of inputs, it should only be used as fallback after all other pairs
        couldn't handle the inputs.

    """

    def compare(self) -> None:
        try:
            # 尝试使用 `==` 运算符比较实际值和期望值
            equal = self.actual == self.expected
        except Exception as error:
            # 如果出现异常，不使用 `self._raise_error_meta`，而是保留异常链
            raise ErrorMeta(
                ValueError,
                f"{self.actual} == {self.expected} failed with:\n{error}.",
                id=self.id,
            ) from error

        if not equal:
            # 如果不相等，触发断言错误，显示实际值和期望值
            self._fail(AssertionError, f"{self.actual} != {self.expected}")


class NonePair(Pair):
    """Pair for ``None`` inputs."""

    def __init__(self, actual: Any, expected: Any, **other_parameters: Any) -> None:
        if not (actual is None or expected is None):
            # 如果实际值和期望值都不是 None，则调用 _inputs_not_supported 方法
            self._inputs_not_supported()

        super().__init__(actual, expected, **other_parameters)

    def compare(self) -> None:
        if not (self.actual is None and self.expected is None):
            # 如果实际值和期望值不都是 None，则触发断言错误，显示不匹配的信息
            self._fail(
                AssertionError, f"None mismatch: {self.actual} is not {self.expected}"
            )


class BooleanPair(Pair):
    """Pair for :class:`bool` inputs.

    .. note::

        If ``numpy`` is available, also handles :class:`numpy.bool_` inputs.

    """

    def __init__(
        self,
        actual: Any,
        expected: Any,
        *,
        id: Tuple[Any, ...],
        **other_parameters: Any,
    ) -> None:
        # 处理输入的实际值和期望值，并调用父类构造函数
        actual, expected = self._process_inputs(actual, expected, id=id)
        super().__init__(actual, expected, **other_parameters)

    @property
    def _supported_types(self) -> Tuple[Type, ...]:
        # 返回支持的数据类型列表，包括 bool 和 numpy.bool_
        cls: List[Type] = [bool]
        if HAS_NUMPY:
            cls.append(np.bool_)
        return tuple(cls)

    def _process_inputs(
        self, actual: Any, expected: Any, *, id: Tuple[Any, ...]
    ) -> Tuple[bool, bool]:
        # 检查输入的实际值和期望值是否是支持的数据类型
        self._check_inputs_isinstance(actual, expected, cls=self._supported_types)
        # 将实际值和期望值转换为布尔值
        actual, expected = (
            self._to_bool(bool_like, id=id) for bool_like in (actual, expected)
        )
        return actual, expected

    def _to_bool(self, bool_like: Any, *, id: Tuple[Any, ...]) -> bool:
        # 将类似布尔值的对象转换为布尔值
        if isinstance(bool_like, bool):
            return bool_like
        elif isinstance(bool_like, np.bool_):
            return bool_like.item()
        else:
            # 如果遇到未知的布尔类型，抛出错误
            raise ErrorMeta(
                TypeError, f"Unknown boolean type {type(bool_like)}.", id=id
            )

    def compare(self) -> None:
        if self.actual is not self.expected:
            # 如果实际值不等于期望值，触发断言错误，显示不匹配的信息
            self._fail(
                AssertionError,
                f"Booleans mismatch: {self.actual} is not {self.expected}",
            )


class NumberPair(Pair):
    # 此处省略了 NumberPair 类的注释，因为它并未提供完整的代码块
    """
    Pair for Python number (:class:`int`, :class:`float`, and :class:`complex`) inputs.
    
    .. note::
    
        If ``numpy`` is available, also handles :class:`numpy.number` inputs.
    
    Kwargs:
        rtol (Optional[float]): Relative tolerance. If specified ``atol`` must also be specified. If omitted, default
            values based on the type are selected with the below table.
        atol (Optional[float]): Absolute tolerance. If specified ``rtol`` must also be specified. If omitted, default
            values based on the type are selected with the below table.
        equal_nan (bool): If ``True``, two ``NaN`` values are considered equal. Defaults to ``False``.
        check_dtype (bool): If ``True``, the type of the inputs will be checked for equality. Defaults to ``False``.
    
    The following table displays correspondence between Python number type and the ``torch.dtype``'s. See
    :func:`assert_close` for the corresponding tolerances.
    
    +------------------+-------------------------------+
    | ``type``         | corresponding ``torch.dtype`` |
    +==================+===============================+
    | :class:`int`     | :attr:`~torch.int64`          |
    +------------------+-------------------------------+
    | :class:`float`   | :attr:`~torch.float64`        |
    +------------------+-------------------------------+
    | :class:`complex` | :attr:`~torch.complex128`     |
    +------------------+-------------------------------+
    """
    
    _TYPE_TO_DTYPE = {
        int: torch.int64,
        float: torch.float64,
        complex: torch.complex128,
    }
    _NUMBER_TYPES = tuple(_TYPE_TO_DTYPE.keys())
    
    def __init__(
        self,
        actual: Any,
        expected: Any,
        *,
        id: Tuple[Any, ...] = (),
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        equal_nan: bool = False,
        check_dtype: bool = False,
        **other_parameters: Any,
    ) -> None:
        """
        Initialize the Pair object with actual and expected values, along with optional parameters.
    
        Args:
            actual (Any): The actual value to compare.
            expected (Any): The expected value to compare.
        
        Kwargs:
            id (Tuple[Any, ...], optional): Identifier for the comparison. Defaults to ().
            rtol (Optional[float], optional): Relative tolerance for comparison. Must be specified with `atol`. Defaults to None.
            atol (Optional[float], optional): Absolute tolerance for comparison. Must be specified with `rtol`. Defaults to None.
            equal_nan (bool, optional): If True, considers two NaN values as equal. Defaults to False.
            check_dtype (bool, optional): If True, checks whether the types of actual and expected values are equal. Defaults to False.
            **other_parameters (Any): Any additional parameters.
        """
        # Process the inputs to ensure they are in a suitable form for comparison
        actual, expected = self._process_inputs(actual, expected, id=id)
        # Initialize the base class with processed inputs and additional parameters
        super().__init__(actual, expected, id=id, **other_parameters)
    
        # Determine tolerances based on the types of actual and expected inputs
        self.rtol, self.atol = get_tolerances(
            *[self._TYPE_TO_DTYPE[type(input)] for input in (actual, expected)],
            rtol=rtol,
            atol=atol,
            id=id,
        )
        # Set whether NaN values should be considered equal
        self.equal_nan = equal_nan
        # Set whether to check the types of actual and expected values
        self.check_dtype = check_dtype
    
    @property
    def _supported_types(self) -> Tuple[Type, ...]:
        """
        Returns the tuple of supported types for comparison.
    
        Returns:
            Tuple[Type, ...]: Tuple containing supported types for comparison.
        """
        cls = list(self._NUMBER_TYPES)
        if HAS_NUMPY:
            cls.append(np.number)
        return tuple(cls)
    
    def _process_inputs(
        self, actual: Any, expected: Any, *, id: Tuple[Any, ...]
    ) -> Tuple[Union[int, float, complex], Union[int, float, complex]]:
        """
        Process the inputs to ensure they are compatible for comparison.
    
        Args:
            actual (Any): The actual value to be processed.
            expected (Any): The expected value to be processed.
            id (Tuple[Any, ...]): Identifier for the comparison.
    
        Returns:
            Tuple[Union[int, float, complex], Union[int, float, complex]]: Processed actual and expected values.
        """
        # Check if the types of actual and expected values are among the supported types
        self._check_inputs_isinstance(actual, expected, cls=self._supported_types)
        # Convert actual and expected values to numerical form if they are number-like
        actual, expected = (
            self._to_number(number_like, id=id) for number_like in (actual, expected)
        )
        return actual, expected
    # 将类内的任何类似数字的对象转换为标准数字类型（int, float, complex）
    def _to_number(
        self, number_like: Any, *, id: Tuple[Any, ...]
    ) -> Union[int, float, complex]:
        # 如果有 numpy 库并且 number_like 是 numpy 的数值类型，则返回其标量值
        if HAS_NUMPY and isinstance(number_like, np.number):
            return number_like.item()
        # 如果 number_like 是预定义的标准数值类型，则直接返回
        elif isinstance(number_like, self._NUMBER_TYPES):
            return number_like  # type: ignore[return-value]
        else:
            # 抛出错误，指明未知的数字类型
            raise ErrorMeta(
                TypeError, f"Unknown number type {type(number_like)}.", id=id
            )

    # 比较实际值和期望值，如果不符合条件则抛出 AssertionError
    def compare(self) -> None:
        # 如果开启了类型检查并且实际值与期望值的类型不同，则抛出 AssertionError
        if self.check_dtype and type(self.actual) is not type(self.expected):
            self._fail(
                AssertionError,
                f"The (d)types do not match: {type(self.actual)} != {type(self.expected)}.",
            )

        # 如果实际值等于期望值，则返回
        if self.actual == self.expected:
            return

        # 如果开启了 equal_nan 选项，并且实际值和期望值都是 NaN，则返回
        if self.equal_nan and cmath.isnan(self.actual) and cmath.isnan(self.expected):
            return

        # 计算实际值和期望值的绝对差
        abs_diff = abs(self.actual - self.expected)
        # 根据相对误差和绝对误差计算容差值
        tolerance = self.atol + self.rtol * abs(self.expected)

        # 如果 abs_diff 是有限值且小于等于 tolerance，则返回
        if cmath.isfinite(abs_diff) and abs_diff <= tolerance:
            return

        # 否则，抛出 AssertionError，包含详细的标量不匹配消息
        self._fail(
            AssertionError,
            make_scalar_mismatch_msg(
                self.actual, self.expected, rtol=self.rtol, atol=self.atol
            ),
        )

    # 返回一个包含描述对象的可选参数的元组
    def extra_repr(self) -> Sequence[str]:
        return (
            "rtol",
            "atol",
            "equal_nan",
            "check_dtype",
        )
class TensorLikePair(Pair):
    """Pair for :class:`torch.Tensor`-like inputs.

    Kwargs:
        allow_subclasses (bool): Whether subclasses of :class:`torch.Tensor` are allowed.
        rtol (Optional[float]): Relative tolerance for numeric comparison.
            If specified, `atol` must also be specified. Defaults based on type.
            See :func:`assert_close` for details.
        atol (Optional[float]): Absolute tolerance for numeric comparison.
            If specified, `rtol` must also be specified. Defaults based on type.
            See :func:`assert_close` for details.
        equal_nan (bool): Whether NaN values are considered equal.
        check_device (bool): Whether to check if tensors are on the same device.
            If `False`, tensors on different devices are moved to CPU before comparison.
        check_dtype (bool): Whether to check if tensors have the same dtype.
            If `False`, tensors with different dtypes are promoted to a common dtype.
        check_layout (bool): Whether to check if tensors have the same layout.
            If `False`, tensors with different layouts are converted to strided tensors.
        check_stride (bool): Whether to check if strided tensors have the same stride.
    """

    def __init__(
        self,
        actual: Any,
        expected: Any,
        *,
        id: Tuple[Any, ...] = (),
        allow_subclasses: bool = True,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        equal_nan: bool = False,
        check_device: bool = True,
        check_dtype: bool = True,
        check_layout: bool = True,
        check_stride: bool = False,
        **other_parameters: Any,
    ):
        # Process actual and expected inputs, ensuring they match subclasses policy
        actual, expected = self._process_inputs(
            actual, expected, id=id, allow_subclasses=allow_subclasses
        )
        # Initialize superclass with processed inputs and additional parameters
        super().__init__(actual, expected, id=id, **other_parameters)

        # Determine relative and absolute tolerances based on input values or defaults
        self.rtol, self.atol = get_tolerances(
            actual, expected, rtol=rtol, atol=atol, id=self.id
        )
        # Set whether NaN values should be considered equal
        self.equal_nan = equal_nan
        # Set whether to check tensors on the same device
        self.check_device = check_device
        # Set whether to check tensors have the same dtype
        self.check_dtype = check_dtype
        # Set whether to check tensors have the same layout
        self.check_layout = check_layout
        # Set whether to check stride for strided tensors
        self.check_stride = check_stride

    def _process_inputs(
        self, actual: Any, expected: Any, *, id: Tuple[Any, ...], allow_subclasses: bool
    ):
        # Internal method to process inputs ensuring subclass policy is respected
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 检查输入的 actual 和 expected 是否直接相关
        directly_related = isinstance(actual, type(expected)) or isinstance(
            expected, type(actual)
        )
        # 如果不直接相关，则调用 _inputs_not_supported 方法处理
        if not directly_related:
            self._inputs_not_supported()

        # 如果不允许子类，并且 actual 的类型不是 expected 的类型，则调用 _inputs_not_supported 方法处理
        if not allow_subclasses and type(actual) is not type(expected):
            self._inputs_not_supported()

        # 将 actual 和 expected 转换为 torch.Tensor 对象
        actual, expected = (self._to_tensor(input) for input in (actual, expected))
        # 检查转换后的 actual 和 expected 是否被支持，使用 id 标识
        for tensor in (actual, expected):
            self._check_supported(tensor, id=id)
        # 返回处理后的 actual 和 expected
        return actual, expected

    def _to_tensor(self, tensor_like: Any) -> torch.Tensor:
        # 如果 tensor_like 已经是 torch.Tensor 类型，则直接返回
        if isinstance(tensor_like, torch.Tensor):
            return tensor_like

        # 尝试将 tensor_like 转换为 torch.Tensor 类型，失败则调用 _inputs_not_supported 方法处理
        try:
            return torch.as_tensor(tensor_like)
        except Exception:
            self._inputs_not_supported()

    def _check_supported(self, tensor: torch.Tensor, *, id: Tuple[Any, ...]) -> None:
        # 检查 tensor 的布局是否在支持的布局集合中，否则抛出 ErrorMeta 异常
        if tensor.layout not in {
            torch.strided,
            torch.sparse_coo,
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        }:
            raise ErrorMeta(
                ValueError, f"Unsupported tensor layout {tensor.layout}", id=id
            )

    def compare(self) -> None:
        # 获取 self.actual 和 self.expected
        actual, expected = self.actual, self.expected

        # 比较 actual 和 expected 的属性
        self._compare_attributes(actual, expected)
        # 如果 actual 或 expected 中任意一个 tensor 的设备类型为 "meta"，则直接返回
        if any(input.device.type == "meta" for input in (actual, expected)):
            return

        # 使 actual 和 expected 的属性保持一致
        actual, expected = self._equalize_attributes(actual, expected)
        # 比较 actual 和 expected 的值
        self._compare_values(actual, expected)

    def _compare_attributes(
        self,
        actual: torch.Tensor,
        expected: torch.Tensor,
        # 比较 actual 和 expected 的属性
    def _equalize_attributes(
        self, actual: torch.Tensor, expected: torch.Tensor
    ) -> None:
        """Checks if the attributes of two tensors match.
    
        Always checks
    
        - the :attr:`~torch.Tensor.shape`,
        - whether both inputs are quantized or not,
        - and if they use the same quantization scheme.
    
        Checks for
    
        - :attr:`~torch.Tensor.layout`,
        - :meth:`~torch.Tensor.stride`,
        - :attr:`~torch.Tensor.device`, and
        - :attr:`~torch.Tensor.dtype`
    
        are optional and can be disabled through the corresponding ``check_*`` flag during construction of the pair.
        """
        
        def raise_mismatch_error(
            attribute_name: str, actual_value: Any, expected_value: Any
        ) -> NoReturn:
            self._fail(
                AssertionError,
                f"The values for attribute '{attribute_name}' do not match: {actual_value} != {expected_value}.",
            )
    
        # Check if shapes match
        if actual.shape != expected.shape:
            raise_mismatch_error("shape", actual.shape, expected.shape)
    
        # Check if both tensors are quantized and their quantization schemes match
        if actual.is_quantized != expected.is_quantized:
            raise_mismatch_error(
                "is_quantized", actual.is_quantized, expected.is_quantized
            )
        elif actual.is_quantized and actual.qscheme() != expected.qscheme():
            raise_mismatch_error("qscheme()", actual.qscheme(), expected.qscheme())
    
        # Check if layouts match (optional, can be disabled by check_layout flag)
        if actual.layout != expected.layout:
            if self.check_layout:
                raise_mismatch_error("layout", actual.layout, expected.layout)
        # If layouts are strided, check strides (optional, can be disabled by check_stride flag)
        elif (
            actual.layout == torch.strided
            and self.check_stride
            and actual.stride() != expected.stride()
        ):
            raise_mismatch_error("stride()", actual.stride(), expected.stride())
    
        # Check if devices match (optional, can be disabled by check_device flag)
        if self.check_device and actual.device != expected.device:
            raise_mismatch_error("device", actual.device, expected.device)
    
        # Check if data types match (optional, can be disabled by check_dtype flag)
        if self.check_dtype and actual.dtype != expected.dtype:
            raise_mismatch_error("dtype", actual.dtype, expected.dtype)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Equalizes some attributes of two tensors for value comparison.

        If ``actual`` and ``expected`` are ...

        - ... not on the same :attr:`~torch.Tensor.device`, they are moved to CPU memory.
        - ... not of the same ``dtype``, they are promoted to a common ``dtype`` (according to
            :func:`torch.promote_types`).
        - ... not of the same ``layout``, they are converted to strided tensors.

        Args:
            actual (Tensor): Actual tensor.
            expected (Tensor): Expected tensor.

        Returns:
            (Tuple[Tensor, Tensor]): Equalized tensors.
        """
        # The comparison logic uses operators currently not supported by the MPS backends.
        # See https://github.com/pytorch/pytorch/issues/77144 for details.
        # TODO: Remove this conversion as soon as all operations are supported natively by the MPS backend
        # Check if either actual or expected tensor is using MPS backend
        if actual.is_mps or expected.is_mps:  # type: ignore[attr-defined]
            # Move tensors to CPU memory if they are using MPS backend
            actual = actual.cpu()
            expected = expected.cpu()

        # Ensure both tensors are on the same device
        if actual.device != expected.device:
            actual = actual.cpu()
            expected = expected.cpu()

        # Ensure both tensors have the same data type
        if actual.dtype != expected.dtype:
            actual_dtype = actual.dtype
            expected_dtype = expected.dtype
            # Handle special case for unsigned int types by promoting to int64
            if actual_dtype in [torch.uint64, torch.uint32, torch.uint16]:
                actual_dtype = torch.int64
            if expected_dtype in [torch.uint64, torch.uint32, torch.uint16]:
                expected_dtype = torch.int64
            # Promote types to a common dtype
            dtype = torch.promote_types(actual_dtype, expected_dtype)
            actual = actual.to(dtype)
            expected = expected.to(dtype)

        # Ensure both tensors have the same layout
        if actual.layout != expected.layout:
            # Convert tensors to dense format if they are not already strided
            actual = actual.to_dense() if actual.layout != torch.strided else actual
            expected = expected.to_dense() if expected.layout != torch.strided else expected

        return actual, expected
    def _compare_values(self, actual: torch.Tensor, expected: torch.Tensor) -> None:
        # 根据输入的 actual 张量的特性选择合适的比较函数
        if actual.is_quantized:
            compare_fn = self._compare_quantized_values
        elif actual.is_sparse:
            compare_fn = self._compare_sparse_coo_values
        elif actual.layout in {
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        }:
            compare_fn = self._compare_sparse_compressed_values
        else:
            compare_fn = self._compare_regular_values_close

        # 使用选定的比较函数进行张量比较
        compare_fn(
            actual, expected, rtol=self.rtol, atol=self.atol, equal_nan=self.equal_nan
        )

    def _compare_quantized_values(
        self,
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        rtol: float,
        atol: float,
        equal_nan: bool,
    ) -> None:
        """Compares quantized tensors by comparing the :meth:`~torch.Tensor.dequantize`'d variants for closeness.

        .. note::

            A detailed discussion about why only the dequantized variant is checked for closeness rather than checking
            the individual quantization parameters for closeness and the integer representation for equality can be
            found in https://github.com/pytorch/pytorch/issues/68548.
        """
        # 使用张量的 dequantize 方法获取浮点数表示，然后进行比较
        return self._compare_regular_values_close(
            actual.dequantize(),
            expected.dequantize(),
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            identifier=lambda default_identifier: f"Quantized {default_identifier.lower()}",
        )

    def _compare_sparse_coo_values(
        self,
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        rtol: float,
        atol: float,
        equal_nan: bool,
    ) -> None:
        # 稀疏张量的比较函数，考虑稀疏 COO (Coordinate list) 格式的特性
        # 实现细节可能涉及根据稀疏张量的具体结构进行比较
    ) -> None:
        """Compares sparse COO tensors by comparing

        - the number of sparse dimensions,
        - the number of non-zero elements (nnz) for equality,
        - the indices for equality, and
        - the values for closeness.
        """
        # 比较稀疏 COO 张量的各个方面：稀疏维度数量、非零元素数量(nnz)、索引和值的接近程度

        # 检查实际稀疏 COO 张量的稀疏维度数量是否与期望的相同
        if actual.sparse_dim() != expected.sparse_dim():
            self._fail(
                AssertionError,
                (
                    f"The number of sparse dimensions in sparse COO tensors does not match: "
                    f"{actual.sparse_dim()} != {expected.sparse_dim()}"
                ),
            )

        # 检查实际稀疏 COO 张量的非零元素数量是否与期望的相同
        if actual._nnz() != expected._nnz():
            self._fail(
                AssertionError,
                (
                    f"The number of specified values in sparse COO tensors does not match: "
                    f"{actual._nnz()} != {expected._nnz()}"
                ),
            )

        # 比较稀疏 COO 张量的索引是否相等
        self._compare_regular_values_equal(
            actual._indices(),
            expected._indices(),
            identifier="Sparse COO indices",
        )

        # 比较稀疏 COO 张量的值的接近程度
        self._compare_regular_values_close(
            actual._values(),
            expected._values(),
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            identifier="Sparse COO values",
        )

    def _compare_sparse_compressed_values(
        self,
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        rtol: float,
        atol: float,
        equal_nan: bool,
    ) -> None:
        """
        Compares sparse compressed tensors by comparing

        - the number of non-zero elements (nnz) for equality,
        - the plain indices for equality,
        - the compressed indices for equality, and
        - the values for closeness.
        """
        # 根据实际稀疏张量的布局选择相应的格式名称、压缩索引方法和普通索引方法
        format_name, compressed_indices_method, plain_indices_method = {
            torch.sparse_csr: (
                "CSR",
                torch.Tensor.crow_indices,
                torch.Tensor.col_indices,
            ),
            torch.sparse_csc: (
                "CSC",
                torch.Tensor.ccol_indices,
                torch.Tensor.row_indices,
            ),
            torch.sparse_bsr: (
                "BSR",
                torch.Tensor.crow_indices,
                torch.Tensor.col_indices,
            ),
            torch.sparse_bsc: (
                "BSC",
                torch.Tensor.ccol_indices,
                torch.Tensor.row_indices,
            ),
        }[actual.layout]

        # 检查实际和期望稀疏张量的非零元素数量是否相等
        if actual._nnz() != expected._nnz():
            self._fail(
                AssertionError,
                (
                    f"The number of specified values in sparse {format_name} tensors does not match: "
                    f"{actual._nnz()} != {expected._nnz()}"
                ),
            )

        # 在CSR/CSC/BSR/BSC稀疏格式中，压缩和普通索引可以是`torch.int32`或`torch.int64`类型。虽然单个张量内强制使用相同的dtype，
        # 但两个张量之间可以不同。因此，我们需要将它们转换为相同的dtype，否则比较将失败。
        actual_compressed_indices = compressed_indices_method(actual)
        expected_compressed_indices = compressed_indices_method(expected)
        indices_dtype = torch.promote_types(
            actual_compressed_indices.dtype, expected_compressed_indices.dtype
        )

        # 对比压缩索引方法的结果是否相等
        self._compare_regular_values_equal(
            actual_compressed_indices.to(indices_dtype),
            expected_compressed_indices.to(indices_dtype),
            identifier=f"Sparse {format_name} {compressed_indices_method.__name__}",
        )
        # 对比普通索引方法的结果是否相等
        self._compare_regular_values_equal(
            plain_indices_method(actual).to(indices_dtype),
            plain_indices_method(expected).to(indices_dtype),
            identifier=f"Sparse {format_name} {plain_indices_method.__name__}",
        )
        # 对比稀疏张量的值是否接近
        self._compare_regular_values_close(
            actual.values(),
            expected.values(),
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            identifier=f"Sparse {format_name} values",
        )

    def _compare_regular_values_equal(
        self,
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        equal_nan: bool = False,
        identifier: Optional[Union[str, Callable[[str], str]]] = None,
    ) -> None:
        """
        确认两个张量的值是否相等。
        """
        self._compare_regular_values_close(
            actual, expected, rtol=0, atol=0, equal_nan=equal_nan, identifier=identifier
        )

    def _compare_regular_values_close(
        self,
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        rtol: float,
        atol: float,
        equal_nan: bool,
        identifier: Optional[Union[str, Callable[[str], str]]] = None,
    ) -> None:
        """
        检查两个张量的值是否在指定的容差范围内接近。
        """
        matches = torch.isclose(
            actual, expected, rtol=rtol, atol=atol, equal_nan=equal_nan
        )
        if torch.all(matches):
            return

        if actual.shape == torch.Size([]):
            msg = make_scalar_mismatch_msg(
                actual.item(),
                expected.item(),
                rtol=rtol,
                atol=atol,
                identifier=identifier,
            )
        else:
            msg = make_tensor_mismatch_msg(
                actual, expected, matches, rtol=rtol, atol=atol, identifier=identifier
            )
        self._fail(AssertionError, msg)

    def extra_repr(self) -> Sequence[str]:
        """
        返回用于表示对象的额外信息的字符串序列。
        """
        return (
            "rtol",
            "atol",
            "equal_nan",
            "check_device",
            "check_dtype",
            "check_layout",
            "check_stride",
        )
def originate_pairs(
    actual: Any,
    expected: Any,
    *,
    pair_types: Sequence[Type[Pair]],
    sequence_types: Tuple[Type, ...] = (collections.abc.Sequence,),
    mapping_types: Tuple[Type, ...] = (collections.abc.Mapping,),
    id: Tuple[Any, ...] = (),
    **options: Any,
) -> List[Pair]:
    """Originates pairs from the individual inputs.

    ``actual`` and ``expected`` can be possibly nested :class:`~collections.abc.Sequence`'s or
    :class:`~collections.abc.Mapping`'s. In this case the pairs are originated by recursing through them.

    Args:
        actual (Any): Actual input.
        expected (Any): Expected input.
        pair_types (Sequence[Type[Pair]]): Sequence of pair types that will be tried to construct with the inputs.
            First successful pair will be used.
        sequence_types (Tuple[Type, ...]): Optional types treated as sequences that will be checked elementwise.
        mapping_types (Tuple[Type, ...]): Optional types treated as mappings that will be checked elementwise.
        id (Tuple[Any, ...]): Optional id of a pair that will be included in an error message.
        **options (Any): Options passed to each pair during construction.

    Raises:
        ErrorMeta: With :class`AssertionError`, if the inputs are :class:`~collections.abc.Sequence`'s, but their
            length does not match.
        ErrorMeta: With :class`AssertionError`, if the inputs are :class:`~collections.abc.Mapping`'s, but their set of
            keys do not match.
        ErrorMeta: With :class`TypeError`, if no pair is able to handle the inputs.
        ErrorMeta: With any expected exception that happens during the construction of a pair.

    Returns:
        (List[Pair]): Originated pairs.
    """
    # We explicitly exclude str's here since they are self-referential and would cause an infinite recursion loop:
    # "a" == "a"[0][0]...
    如果实际输入和期望输入都是序列类型（不包括字符串），则检查它们的长度是否一致
    if (
        isinstance(actual, sequence_types)
        and not isinstance(actual, str)
        and isinstance(expected, sequence_types)
        and not isinstance(expected, str)
    ):
        # 计算实际输入和期望输入的长度
        actual_len = len(actual)
        expected_len = len(expected)
        # 如果长度不一致，则抛出错误
        if actual_len != expected_len:
            raise ErrorMeta(
                AssertionError,
                f"The length of the sequences mismatch: {actual_len} != {expected_len}",
                id=id,
            )

        # 初始化空的 pairs 列表
        pairs = []
        # 遍历序列的每个元素，并递归调用 originate_pairs 来构建 pairs 列表
        for idx in range(actual_len):
            pairs.extend(
                originate_pairs(
                    actual[idx],
                    expected[idx],
                    pair_types=pair_types,
                    sequence_types=sequence_types,
                    mapping_types=mapping_types,
                    id=(*id, idx),
                    **options,
                )
            )
        # 返回构建好的 pairs 列表
        return pairs
    elif isinstance(actual, mapping_types) and isinstance(expected, mapping_types):
        # 检查实际值和期望值是否都属于映射类型
        actual_keys = set(actual.keys())
        # 获取实际映射的所有键集合
        expected_keys = set(expected.keys())
        # 获取期望映射的所有键集合
        if actual_keys != expected_keys:
            # 如果实际映射的键集合和期望映射的键集合不相等，说明键不匹配
            missing_keys = expected_keys - actual_keys
            # 计算缺失的键
            additional_keys = actual_keys - expected_keys
            # 计算多余的键
            raise ErrorMeta(
                AssertionError,
                (
                    f"The keys of the mappings do not match:\n"
                    f"Missing keys in the actual mapping: {sorted(missing_keys)}\n"
                    f"Additional keys in the actual mapping: {sorted(additional_keys)}"
                ),
                id=id,
            )
            # 抛出错误，提示映射的键不匹配

        keys: Collection = actual_keys
        # 将实际映射的键集合赋值给变量 keys
        # 由于在第一次失败后就中止，我们尝试保持确定性
        with contextlib.suppress(Exception):
            # 忽略任何异常
            keys = sorted(keys)
            # 对键进行排序

        pairs = []
        # 初始化空列表用于存储键值对
        for key in keys:
            # 遍历排序后的键列表
            pairs.extend(
                originate_pairs(
                    actual[key],
                    expected[key],
                    pair_types=pair_types,
                    sequence_types=sequence_types,
                    mapping_types=mapping_types,
                    id=(*id, key),
                    **options,
                )
            )
            # 调用 originate_pairs 函数生成键值对，并添加到 pairs 列表中
        return pairs
        # 返回生成的所有键值对列表
    else:
        # 对于每种比较类型，尝试执行以下操作
        for pair_type in pair_types:
            try:
                # 尝试使用当前的比较类型创建比较对象，传入实际值、期望值和其他选项
                return [pair_type(actual, expected, id=id, **options)]
            # 如果在创建过程中抛出 `UnsupportedInputs` 异常，表示当前比较类型无法处理这些输入，
            # 因此继续尝试下一个比较类型。
            except UnsupportedInputs:
                continue
            # 如果在创建过程中抛出 `ErrorMeta` 异常，这是有序终止的方法，直接重新抛出异常。
            # 这个分支是为了避免与下面的异常捕获混淆。
            except ErrorMeta:
                raise
            # 如果在创建过程中抛出其他异常，这是意外情况，会提供更多关于发生情况的信息。
            # 如果适用，这种异常应该被预期在未来的处理中。
            except Exception as error:
                raise RuntimeError(
                    # 创建异常信息，指出在哪里以及使用哪些值发生了异常
                    f"Originating a {pair_type.__name__}() at item {''.join(str([item]) for item in id)} with\n\n"
                    f"{type(actual).__name__}(): {actual}\n\n"
                    f"and\n\n"
                    f"{type(expected).__name__}(): {expected}\n\n"
                    f"resulted in the unexpected exception above. "
                    f"If you are a user and see this message during normal operation "
                    "please file an issue at https://github.com/pytorch/pytorch/issues. "
                    "If you are a developer and working on the comparison functions, "
                    "please except the previous error and raise an expressive `ErrorMeta` instead."
                ) from error
        else:
            # 如果所有比较类型都无法处理输入，抛出 `ErrorMeta` 异常，指明没有适合处理这种类型输入的比较对。
            raise ErrorMeta(
                TypeError,
                f"No comparison pair was able to handle inputs of type {type(actual)} and {type(expected)}.",
                id=id,
            )
def not_close_error_metas(
    actual: Any,
    expected: Any,
    *,
    pair_types: Sequence[Type[Pair]] = (ObjectPair,),
    sequence_types: Tuple[Type, ...] = (collections.abc.Sequence,),
    mapping_types: Tuple[Type, ...] = (collections.abc.Mapping,),
    **options: Any,
) -> List[ErrorMeta]:
    """Asserts that inputs are equal.

    ``actual`` and ``expected`` can be possibly nested :class:`~collections.abc.Sequence`'s or
    :class:`~collections.abc.Mapping`'s. In this case the comparison happens elementwise by recursing through them.

    Args:
        actual (Any): Actual input.
        expected (Any): Expected input.
        pair_types (Sequence[Type[Pair]]): Sequence of :class:`Pair` types that will be tried to construct with the
            inputs. First successful pair will be used. Defaults to only using :class:`ObjectPair`.
        sequence_types (Tuple[Type, ...]): Optional types treated as sequences that will be checked elementwise.
        mapping_types (Tuple[Type, ...]): Optional types treated as mappings that will be checked elementwise.
        **options (Any): Options passed to each pair during construction.
    """
    # Hide this function from `pytest`'s traceback
    __tracebackhide__ = True

    try:
        # Attempt to create pairs from actual and expected inputs
        pairs = originate_pairs(
            actual,
            expected,
            pair_types=pair_types,
            sequence_types=sequence_types,
            mapping_types=mapping_types,
            **options,
        )
    except ErrorMeta as error_meta:
        # Reraise as an error to hide internal traceback
        raise error_meta.to_error() from None  # noqa: RSE102

    error_metas: List[ErrorMeta] = []
    # Iterate over each pair and compare them
    for pair in pairs:
        try:
            # Attempt to compare the current pair
            pair.compare()
        except ErrorMeta as error_meta:
            # If comparison raises ErrorMeta, append it to error_metas
            error_metas.append(error_meta)
        except Exception as error:
            # Raise a RuntimeError for unexpected exceptions during comparison
            raise RuntimeError(
                f"Comparing\n\n"
                f"{pair}\n\n"
                f"resulted in the unexpected exception above. "
                f"If you are a user and see this message during normal operation "
                "please file an issue at https://github.com/pytorch/pytorch/issues. "
                "If you are a developer and working on the comparison functions, "
                "please except the previous error and raise an expressive `ErrorMeta` instead."
            ) from error

    # Note on error meta cycles and memory leak
    # ErrorMeta objects in this list capture
    # tracebacks that refer to the frame of this function.
    # The local variable `error_metas` refers to the error meta
    # objects, creating a reference cycle. Frames in the traceback
    # would not get freed until cycle collection, leaking cuda memory in tests.
    # 创建一个列表 error_metas，其中包含一个元素，这个元素是传入的 error_metas 对象
    error_metas = [error_metas]
    # 从列表中弹出并返回唯一的元素，从而移除对 error_metas 对象的引用
    return error_metas.pop()
# 定义一个断言函数 assert_close，用于比较两个值 actual 和 expected 是否接近
def assert_close(
    actual: Any,  # 参数 actual：表示实际值，可以是任何类型
    expected: Any,  # 参数 expected：表示期望值，可以是任何类型
    *,  # 后续参数为关键字参数，需要使用关键字调用
    allow_subclasses: bool = True,  # 是否允许 actual 和 expected 的子类相互比较，默认为 True
    rtol: Optional[float] = None,  # 相对误差阈值，默认为 None
    atol: Optional[float] = None,  # 绝对误差阈值，默认为 None
    equal_nan: bool = False,  # 是否认为 NaN 相等，默认为 False
    check_device: bool = True,  # 是否检查张量的设备，默认为 True
    check_dtype: bool = True,  # 是否检查张量的数据类型，默认为 True
    check_layout: bool = True,  # 是否检查张量的布局，默认为 True
    check_stride: bool = False,  # 是否检查张量的步幅，默认为 False
    msg: Optional[Union[str, Callable[[str], str]]] = None,  # 断言失败时的错误消息，可以是字符串或回调函数
):
    r"""Asserts that ``actual`` and ``expected`` are close.

    如果 ``actual`` 和 ``expected`` 是跨步、非量化、实值且有限的张量，则认为它们接近，条件为：

    .. math::

        \lvert \text{actual} - \text{expected} \rvert \le \texttt{atol} + \texttt{rtol} \cdot \lvert \text{expected} \rvert

    非有限值（如 ``-inf`` 和 ``inf``）只有在它们完全相等时才被认为接近。如果 ``equal_nan`` 为 ``True``，则认为 NaN 互相等同。

    此外，它们还需要满足以下条件才被认为接近：

    - :attr:`~torch.Tensor.device` 相同（如果 ``check_device`` 为 ``True``），
    - ``dtype`` 相同（如果 ``check_dtype`` 为 ``True``），
    - ``layout`` 相同（如果 ``check_layout`` 为 ``True``），
    - 步幅相同（如果 ``check_stride`` 为 ``True``）。

    如果 ``actual`` 或 ``expected`` 是元张量，则只会进行属性检查。

    如果 ``actual`` 和 ``expected`` 是稀疏张量（COO、CSR、CSC、BSR 或 BSC 布局），则会分别检查它们的跨步成员。
    其中，对于 COO 布局，总是检查索引 ``indices``；对于 CSR 和 BSR 布局，总是检查 ``crow_indices`` 和 ``col_indices``；
    对于 CSC 和 BSC 布局，总是检查 ``ccol_indices`` 和 ``row_indices``。至于值的接近性，按上述定义检查。

    如果 ``actual`` 和 ``expected`` 是量化的，它们被认为接近当且仅当它们具有相同的 :meth:`~torch.Tensor.qscheme`，
    并且根据上述定义， :meth:`~torch.Tensor.dequantize` 的结果接近。

    ``actual`` 和 ``expected`` 可以是 :class:`~torch.Tensor` 或任何类似张量或标量的结构，可通过 :func:`torch.as_tensor` 构造 :class:`torch.Tensor`。
    除了 Python 标量外，输入类型必须直接相关。此外，如果它们是 :class:`~collections.abc.Sequence` 或 :class:`~collections.abc.Mapping`，
    则认为它们接近，如果它们的结构匹配且所有元素按上述定义接近。

    .. note::

        Python 标量是类型关系要求的例外，因为它们的 :func:`type`，即 :class:`int`、 :class:`float` 和 :class:`complex`，
        等效于张量类似的 ``dtype``。因此，不同类型的 Python 标量可以进行比较，但需要 ``check_dtype=False``。

    """
    Args:
        actual (Any): 实际输入。
        expected (Any): 预期输入。
        allow_subclasses (bool): 如果为 ``True``（默认），则允许直接相关类型的输入（除了Python标量）。否则需要类型完全相等。
        rtol (Optional[float]): 相对容差。如果指定，则必须同时指定 ``atol``。如果省略，将根据 :attr:`~torch.Tensor.dtype` 选择默认值。
        atol (Optional[float]): 绝对容差。如果指定，则必须同时指定 ``rtol``。如果省略，将根据 :attr:`~torch.Tensor.dtype` 选择默认值。
        equal_nan (Union[bool, str]): 如果为 ``True``，则两个 ``NaN`` 值将被视为相等。
        check_device (bool): 如果为 ``True``（默认），断言对应的张量位于相同的 :attr:`~torch.Tensor.device` 上。如果禁用此检查，将比较前将位于不同 :attr:`~torch.Tensor.device` 的张量移动到 CPU。
        check_dtype (bool): 如果为 ``True``（默认），断言对应的张量具有相同的 ``dtype``。如果禁用此检查，将根据 :func:`torch.promote_types` 将具有不同 ``dtype`` 的张量提升到公共 ``dtype`` 后再比较。
        check_layout (bool): 如果为 ``True``（默认），断言对应的张量具有相同的 ``layout``。如果禁用此检查，将比较前将具有不同 ``layout`` 的张量转换为步进张量。
        check_stride (bool): 如果为 ``True`` 并且对应的张量是步进张量，则断言它们具有相同的步幅。
        msg (Optional[Union[str, Callable[[str], str]]]): 可选的错误消息，在比较过程中发生失败时使用。也可以作为可调用对象传递，此时将使用生成的消息调用它，并应返回新消息。
    Raises:
        ValueError: 如果无法从输入构造 :class:`torch.Tensor`。
        ValueError: 如果只指定了 ``rtol`` 或 ``atol`` 中的一个。
        AssertionError: 如果相应的输入不是 Python 标量并且没有直接关联。
        AssertionError: 如果 ``allow_subclasses`` 为 ``False``，但相应的输入不是 Python 标量且具有不同的类型。
        AssertionError: 如果输入是 :class:`~collections.abc.Sequence`，但它们的长度不匹配。
        AssertionError: 如果输入是 :class:`~collections.abc.Mapping`，但它们的键集不匹配。
        AssertionError: 如果相应的张量的 :attr:`~torch.Tensor.shape` 不相同。
        AssertionError: 如果 ``check_layout`` 为 ``True``，但相应的张量的 :attr:`~torch.Tensor.layout` 不相同。
        AssertionError: 如果只有其中一个相应的张量是量化的。
        AssertionError: 如果相应的张量是量化的，但具有不同的 :meth:`~torch.Tensor.qscheme`。
        AssertionError: 如果 ``check_device`` 为 ``True``，但相应的张量不在相同的 :attr:`~torch.Tensor.device` 上。
        AssertionError: 如果 ``check_dtype`` 为 ``True``，但相应的张量的 ``dtype`` 不相同。
        AssertionError: 如果 ``check_stride`` 为 ``True``，但相应的步幅张量的步幅不相同。
        AssertionError: 如果相应张量的值根据上述定义不接近。

    The following table displays the default ``rtol`` and ``atol`` for different ``dtype``'s. In case of mismatching
    ``dtype``'s, the maximum of both tolerances is used.

    +---------------------------+------------+----------+
    | ``dtype``                 | ``rtol``   | ``atol`` |
    +===========================+============+==========+
    | :attr:`~torch.float16`    | ``1e-3``   | ``1e-5`` |
    +---------------------------+------------+----------+
    | :attr:`~torch.bfloat16`   | ``1.6e-2`` | ``1e-5`` |
    +---------------------------+------------+----------+
    | :attr:`~torch.float32`    | ``1.3e-6`` | ``1e-5`` |
    +---------------------------+------------+----------+
    | :attr:`~torch.float64`    | ``1e-7``   | ``1e-7`` |
    +---------------------------+------------+----------+
    | :attr:`~torch.complex32`  | ``1e-3``   | ``1e-5`` |
    +---------------------------+------------+----------+
    | :attr:`~torch.complex64`  | ``1.3e-6`` | ``1e-5`` |
    +---------------------------+------------+----------+
    | :attr:`~torch.complex128` | ``1e-7``   | ``1e-7`` |
    +---------------------------+------------+----------+
    | :attr:`~torch.quint8`     | ``1.3e-6`` | ``1e-5`` |
    +---------------------------+------------+----------+
    | :attr:`~torch.quint2x4`   | ``1.3e-6`` | ``1e-5`` |
    +---------------------------+------------+----------+
    # Hide this function from `pytest`'s traceback
    __tracebackhide__ = True
    
    # 使用 `torch.testing.assert_close` 进行接近性检查，返回不接近的错误元数据列表
    error_metas = not_close_error_metas(
        actual,  # 实际值
        expected,  # 期望值
        pair_types=(  # 允许比较的数据对类型
            NonePair,  # None 类型对
            BooleanPair,  # 布尔类型对
            NumberPair,  # 数字类型对
            TensorLikePair,  # 张量类似类型对
        ),
        allow_subclasses=allow_subclasses,  # 是否允许子类
        rtol=rtol,  # 相对容差
        atol=atol,  # 绝对容差
        equal_nan=equal_nan,  # 是否相等 NaN
        check_device=check_device,  # 检查设备类型
        check_dtype=check_dtype,  # 检查数据类型
        check_layout=check_layout,  # 检查布局
        check_stride=check_stride,  # 检查步幅
        msg=msg,  # 错误消息
    )
    
    if error_metas:
        # TODO: 将所有的错误元数据组合成一个 AssertionError
        raise error_metas[0].to_error(msg)
# 标记函数为过时，提供替代方案和详细升级指南的警告信息
@deprecated(
    "`torch.testing.assert_allclose()` is deprecated since 1.12 and will be removed in a future release. "
    "Please use `torch.testing.assert_close()` instead. "
    "You can find detailed upgrade instructions in https://github.com/pytorch/pytorch/issues/61844.",
    category=FutureWarning,
)
# 定义函数 assert_allclose，用于比较两个 torch.Tensor 的近似相等性
def assert_allclose(
    actual: Any,
    expected: Any,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    equal_nan: bool = True,
    msg: str = "",
) -> None:
    """
    .. warning::

       :func:`torch.testing.assert_allclose` is deprecated since ``1.12`` and will be removed in a future release.
       Please use :func:`torch.testing.assert_close` instead. You can find detailed upgrade instructions
       `here <https://github.com/pytorch/pytorch/issues/61844>`_.
    """
    # 如果 actual 不是 torch.Tensor，则将其转换为 torch.Tensor
    if not isinstance(actual, torch.Tensor):
        actual = torch.tensor(actual)
    # 如果 expected 不是 torch.Tensor，则将其转换为 torch.Tensor，并使用 actual 的数据类型
    if not isinstance(expected, torch.Tensor):
        expected = torch.tensor(expected, dtype=actual.dtype)

    # 如果用户未指定 rtol 和 atol，则根据默认精度设置它们
    if rtol is None and atol is None:
        rtol, atol = default_tolerances(
            actual,
            expected,
            dtype_precisions={
                torch.float16: (1e-3, 1e-3),
                torch.float32: (1e-4, 1e-5),
                torch.float64: (1e-5, 1e-8),
            },
        )

    # 使用 torch.testing.assert_close 检查两个 tensor 是否在指定的容差范围内近似相等
    torch.testing.assert_close(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        check_device=True,
        check_dtype=False,
        check_stride=False,
        msg=msg or None,
    )
```