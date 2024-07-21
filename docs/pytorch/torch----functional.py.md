# `.\pytorch\torch\functional.py`

```
# mypy: allow-untyped-defs
# 引入 itertools 模块，提供了用于构建迭代器的函数
import itertools
# 引入 operator 模块，提供了对内置操作符的函数实现
import operator
# 引入类型提示模块
from typing import Any, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union

# 引入 PyTorch 库
import torch
# 引入 torch.nn.functional 模块，提供了神经网络函数的实现
import torch.nn.functional as F
# 从 torch 模块中引入 _VF（PyTorch 函数的核心部分）
from torch import _VF, Tensor
# 从 torch._C 模块中引入 _add_docstr 函数
from torch._C import _add_docstr
# 从 torch._jit_internal 模块中引入 _overload 函数作为 overload
from torch._jit_internal import _overload as overload, boolean_dispatch
# 从 torch._lowrank 模块中引入 pca_lowrank 和 svd_lowrank 函数
from torch._lowrank import pca_lowrank, svd_lowrank
# 从 torch.overrides 模块中引入一系列函数
from torch.overrides import (
    handle_torch_function,
    has_torch_function,
    has_torch_function_unary,
    has_torch_function_variadic,
)

# 定义模块的公开接口
__all__ = [
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "align_tensors",
    "broadcast_shapes",
    "broadcast_tensors",
    "cartesian_prod",
    "block_diag",
    "cdist",
    "chain_matmul",
    "einsum",
    "istft",
    "lu",
    "norm",
    "meshgrid",
    "pca_lowrank",
    "split",
    "stft",
    "svd_lowrank",
    "tensordot",
    "unique",
    "unique_consecutive",
    "unravel_index",
]

# 定义 broadcast_tensors 函数
def broadcast_tensors(*tensors):
    r"""broadcast_tensors(*tensors) -> List of Tensors

    Broadcasts the given tensors according to :ref:`broadcasting-semantics`.

    Args:
        *tensors: any number of tensors of the same type

    .. warning::

        More than one element of a broadcasted tensor may refer to a single
        memory location. As a result, in-place operations (especially ones that
        are vectorized) may result in incorrect behavior. If you need to write
        to the tensors, please clone them first.

    Example::

        >>> x = torch.arange(3).view(1, 3)
        >>> y = torch.arange(2).view(2, 1)
        >>> a, b = torch.broadcast_tensors(x, y)
        >>> a.size()
        torch.Size([2, 3])
        >>> a
        tensor([[0, 1, 2],
                [0, 1, 2]])
    """
    # 如果输入的 tensors 有 torch 函数的重载机制，则使用 torch 函数处理
    if has_torch_function(tensors):
        return handle_torch_function(broadcast_tensors, tensors, *tensors)
    # 否则调用 _VF.broadcast_tensors 函数进行广播操作
    return _VF.broadcast_tensors(tensors)  # type: ignore[attr-defined]


# 定义 broadcast_shapes 函数
def broadcast_shapes(*shapes):
    r"""broadcast_shapes(*shapes) -> Size

    Similar to :func:`broadcast_tensors` but for shapes.

    This is equivalent to
    ``torch.broadcast_tensors(*map(torch.empty, shapes))[0].shape``
    but avoids the need create to intermediate tensors. This is useful for
    broadcasting tensors of common batch shape but different rightmost shape,
    e.g. to broadcast mean vectors with covariance matrices.

    Example::

        >>> torch.broadcast_shapes((2,), (3, 1), (1, 1, 1))
        torch.Size([1, 3, 2])

    Args:
        \*shapes (torch.Size): Shapes of tensors.

    Returns:
        shape (torch.Size): A shape compatible with all input shapes.

    Raises:
        RuntimeError: If shapes are incompatible.
    """
    # This wrapper exists to support variadic args.
    # TODO Move this to C++ once the jit has better support for torch.Size.
    # 如果当前没有进行 Torch JIT 追踪
    if not torch.jit.is_tracing():
        # 初始化最大长度为 0
        max_len = 0
        # 遍历 shapes 中的每一个形状
        for shape in shapes:
            # 如果形状是整数或 Torch 符号整数
            if isinstance(shape, (int, torch.SymInt)):
                # 如果 max_len 小于 1，则将其设为 1
                if max_len < 1:
                    max_len = 1
            # 如果形状是元组或列表
            elif isinstance(shape, (tuple, list)):
                # 获取形状的长度
                s = len(shape)
                # 如果 max_len 小于当前形状的长度，则更新 max_len
                if max_len < s:
                    max_len = s
        # 使用长度为 max_len 的列表初始化结果
        result = [1] * max_len

        # 导入符号形状保护函数
        from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

        # 再次遍历 shapes 中的每一个形状
        for shape in shapes:
            # 如果形状是整数或 Torch 符号整数，转为单元素元组
            if isinstance(shape, (int, torch.SymInt)):
                shape = (shape,)
            # 如果形状是元组或列表
            if isinstance(shape, (tuple, list)):
                # 逆序遍历形状的每一个维度
                for i in range(-1, -1 - len(shape), -1):
                    # 如果形状中的维度小于 0，则抛出运行时错误
                    if shape[i] < 0:
                        raise RuntimeError(
                            f"Trying to create tensor with negative dimension ({shape[i]}): ({shape[i]})"
                        )
                    # NB: result 初始化为 1，因此这实际上是一个等于 1 的测试
                    # 如果 guard_size_oblivious 函数保护 shape[i] 等于 1 或者与 result[i] 相等，则继续
                    if guard_size_oblivious(shape[i] == 1) or guard_size_oblivious(
                        shape[i] == result[i]
                    ):
                        continue
                    # 如果 result[i] 不等于 1，则抛出形状不匹配的运行时错误
                    if result[i] != 1:
                        raise RuntimeError(
                            "Shape mismatch: objects cannot be broadcast to a single shape"
                        )
                    # 将 result[i] 更新为 shape[i]
                    result[i] = shape[i]
            else:
                # 如果形状不是整数、元组或列表，则抛出运行时错误
                raise RuntimeError(
                    "Input shapes should be of type ints, a tuple of ints, or a list of ints, got ",
                    shape,
                )
        # 返回 torch.Size 对象，其维度为 result
        return torch.Size(result)
    else:
        # 当 Torch JIT 正在追踪时，由于上面的实现，torch.jit.trace 会硬编码大小，导致后续的重播失败
        # 使用 torch.no_grad() 上下文管理器
        with torch.no_grad():
            # 创建一个在 CPU 上的零标量
            scalar = torch.zeros((), device="cpu")
            # 根据 shapes 创建张量列表
            tensors = [scalar.expand(shape) for shape in shapes]
            # 广播张量以匹配形状
            tensors = broadcast_tensors(*tensors)
            # 返回第一个张量的形状
            return tensors[0].shape
# 定义一个函数 einsum，用于根据爱因斯坦求和约定计算张量的乘积和求和
def einsum(*args: Any) -> Tensor:
    r"""einsum(equation, *operands) -> Tensor

    Sums the product of the elements of the input :attr:`operands` along dimensions specified using a notation
    based on the Einstein summation convention.

    Einsum allows computing many common multi-dimensional linear algebraic array operations by representing them
    in a short-hand format based on the Einstein summation convention, given by :attr:`equation`. The details of
    this format are described below, but the general idea is to label every dimension of the input :attr:`operands`
    with some subscript and define which subscripts are part of the output. The output is then computed by summing
    the product of the elements of the :attr:`operands` along the dimensions whose subscripts are not part of the
    output. For example, matrix multiplication can be computed using einsum as `torch.einsum("ij,jk->ik", A, B)`.
    Here, j is the summation subscript and i and k the output subscripts (see section below for more details on why).
    """

    # 如果输入的 tensor 具有 torch 函数的重载，使用 handle_torch_function 处理
    if has_torch_function_unary(tensor):
        return handle_torch_function(
            split, (tensor,), tensor, split_size_or_sections, dim=dim
        )
    
    # 调用 tensor 对象的 split 方法，根据给定的 split_size_or_sections 和 dim 进行张量分割
    # 这里是因为根据输入的 split_size_or_sections 参数的不同类型，调用了不同的 ATen 函数来执行分割操作
    return tensor.split(split_size_or_sections, dim)
    # 定义 Equation 类，描述了一个特定的数学运算规则
    # The :attr:`equation` string specifies the subscripts (letters in `[a-zA-Z]`) for each dimension of
    # the input :attr:`operands` in the same order as the dimensions, separating subscripts for each operand by a
    # comma (','), e.g. `'ij,jk'` specify subscripts for two 2D operands. The dimensions labeled with the same subscript
    # must be broadcastable, that is, their size must either match or be `1`. The exception is if a subscript is
    # repeated for the same input operand, in which case the dimensions labeled with this subscript for this operand
    # must match in size and the operand will be replaced by its diagonal along these dimensions. The subscripts that
    # appear exactly once in the :attr:`equation` will be part of the output, sorted in increasing alphabetical order.
    # The output is computed by multiplying the input :attr:`operands` element-wise, with their dimensions aligned based
    # on the subscripts, and then summing out the dimensions whose subscripts are not part of the output.
    # Optionally, the output subscripts can be explicitly defined by adding an arrow ('->') at the end of the equation
    # followed by the subscripts for the output. For instance, the following equation computes the transpose of a
    # matrix multiplication: 'ij,jk->ki'. The output subscripts must appear at least once for some input operand and
    # at most once for the output.
    # Ellipsis ('...') can be used in place of subscripts to broadcast the dimensions covered by the ellipsis.
    # Each input operand may contain at most one ellipsis which will cover the dimensions not covered by subscripts,
    # e.g. for an input operand with 5 dimensions, the ellipsis in the equation `'ab...c'` cover the third and fourth
    # dimensions. The ellipsis does not need to cover the same number of dimensions across the :attr:`operands` but the
    # 'shape' of the ellipsis (the size of the dimensions covered by them) must broadcast together. If the output is not
    # explicitly defined with the arrow ('->') notation, the ellipsis will come first in the output (left-most dimensions),
    # before the subscript labels that appear exactly once for the input operands. e.g. the following equation implements
    # batch matrix multiplication `'...ij,...jk'`.
    # A few final notes: the equation may contain whitespaces between the different elements (subscripts, ellipsis,
    # arrow and comma) but something like `'. . .'` is not valid. An empty string `''` is valid for scalar operands.
    # .. note::
    # ``torch.einsum`` handles ellipsis ('...') differently from NumPy in that it allows dimensions
    # covered by the ellipsis to be summed over, that is, ellipsis are not required to be part of the output.
    class Equation:
        pass
    # 导入 torch 库，用于张量操作
    import torch
    
    # 定义函数 einsum_custom，用于执行自定义的 Einstein 求和操作
    def einsum_custom(equation, operands):
        """
        .. note::
    
            This function uses opt_einsum (https://optimized-einsum.readthedocs.io/en/stable/) to speed up computation or to
            consume less memory by optimizing contraction order. This optimization occurs when there are at least three
            inputs, since the order does not matter otherwise. Note that finding _the_ optimal path is an NP-hard problem,
            thus, opt_einsum relies on different heuristics to achieve near-optimal results. If opt_einsum is not available,
            the default order is to contract from left to right.
    
            To bypass this default behavior, add the following line to disable the usage of opt_einsum and skip path
            calculation: `torch.backends.opt_einsum.enabled = False`
    
            To specify which strategy you'd like for opt_einsum to compute the contraction path, add the following line:
            `torch.backends.opt_einsum.strategy = 'auto'`. The default strategy is 'auto', and we also support 'greedy' and
            'optimal'. Disclaimer that the runtime of 'optimal' is factorial in the number of inputs! See more details in
            the opt_einsum documentation (https://optimized-einsum.readthedocs.io/en/stable/path_finding.html).
    
        .. note::
    
            As of PyTorch 1.10 :func:`torch.einsum` also supports the sublist format (see examples below). In this format,
            subscripts for each operand are specified by sublists, list of integers in the range [0, 52). These sublists
            follow their operands, and an extra sublist can appear at the end of the input to specify the output's
            subscripts., e.g. `torch.einsum(op1, sublist1, op2, sublist2, ..., [subslist_out])`. Python's `Ellipsis` object
            may be provided in a sublist to enable broadcasting as described in the Equation section above.
    
        Args:
            equation (str): The subscripts for the Einstein summation.
            operands (List[Tensor]): The tensors to compute the Einstein summation of.
        """
    
        # 使用 torch.einsum 执行 Einstein 求和操作，根据给定的方程式和操作数
        result = torch.einsum(equation, *operands)
    
        # 返回求和结果
        return result
    """
    # 引入 torch 的 opt_einsum 后端模块，用于优化 Einstein 求和符号的计算
    import torch.backends.opt_einsum as opt_einsum

    # 这个函数是为了支持可变参数而存在的包装器。
    # 如果参数少于2个，抛出数值错误异常，提示至少需要指定方程字符串和一个操作数，或者至少一个操作数及其子脚本列表
    if len(args) < 2:
        raise ValueError(
            "einsum(): must specify the equation string and at least one operand, "
            "or at least one operand and its subscripts list"
        )

    # 初始化方程式字符串和操作数为空
    equation = None
    operands = None
    ```
    # 检查第一个参数是否为 torch.Tensor 类型
    if isinstance(args[0], torch.Tensor):
        # 将子脚本列表格式转换为方程字符串格式，
        # 子脚本列表交替出现操作数及其下标，最后可能有一个输出子脚本列表（详见文档）
        # 通过从子脚本列表创建方程字符串，并将输入操作数分组成 tensorlist（List[Tensor]）
        def parse_subscript(n: int) -> str:
            # 处理特殊的省略符号 Ellipsis
            if n == Ellipsis:
                return "..."
            # 处理大写字母 A-Z 的下标
            if n >= 0 and n < 26:
                return chr(ord("A") + n)
            # 处理小写字母 a-z 的下标
            if n >= 26 and n < 52:
                return chr(ord("a") + n - 26)
            # 抛出异常，如果下标不在有效范围 [0, 52) 内
            raise ValueError(
                "einsum(): subscript in subscript list is not within the valid range [0, 52)"
            )

        # 解析输入操作数的子脚本
        equation = ",".join("".join(parse_subscript(s) for s in l) for l in args[1::2])

        # 解析可选的输出子脚本（当参数数量为奇数时提供）
        if len(args) % 2 == 1:
            equation += "->" + "".join(parse_subscript(s) for s in args[-1])
            operands = args[:-1:2]
        else:
            operands = args[::2]
    else:
        # 如果第一个参数不是 torch.Tensor 类型，则假定第一个参数是方程字符串
        equation = args[0]
        operands = args[1:]

    # 检查是否存在 torch function
    if has_torch_function(operands):
        # 处理 torch function 的情况
        return handle_torch_function(einsum, operands, equation, *operands)

    # 如果操作数的数量为 1 并且第一个操作数是列表或元组
    if len(operands) == 1 and isinstance(operands[0], (list, tuple)):
        # 旧接口：将操作数作为一个列表参数传递
        _operands = operands[0]
        # 递归调用 einsum，以处理包含有 torch function 的操作数的情况
        return einsum(equation, *_operands)

    # 如果操作数的数量小于等于 2 或者 opt_einsum 已禁用
    if len(operands) <= 2 or not opt_einsum.enabled:
        # 处理不需要优化的情况，或者用户已禁用 opt_einsum
        return _VF.einsum(equation, operands)  # type: ignore[attr-defined]

    path = None
    # 如果 opt_einsum 可用
    if opt_einsum.is_available():
        _opt_einsum = opt_einsum.get_opt_einsum()
        # 获取最佳路径并展开以便传递给 C++
        tupled_path = _opt_einsum.contract_path(
            equation, *operands, optimize=opt_einsum.strategy
        )[0]
        path = [item for pair in tupled_path for item in pair]
    return _VF.einsum(equation, operands, path=path)  # type: ignore[attr-defined]
# 如果 TYPE_CHECKING 为真，则定义 meshgrid 函数以支持可变参数。
if TYPE_CHECKING:
    # JIT 不理解 Union 类型，因此只为 mypy 添加类型注解。
    # 根据输入的张量和索引方式生成网格点。
    def meshgrid(
        *tensors: Union[Tensor, List[Tensor]], indexing: Optional[str] = None
    ) -> Tuple[Tensor, ...]:
        return _meshgrid(*tensors, indexing=indexing)

else:

# 如果 TYPE_CHECKING 为假，则定义 _meshgrid 函数。
def _meshgrid(*tensors, indexing: Optional[str]):
    # 如果输入张量存在 torch function，则调用 torch function 处理。
    if has_torch_function(tensors):
        return handle_torch_function(meshgrid, tensors, *tensors, indexing=indexing)
    # 如果只有一个张量且为列表或元组，则将其作为输入张量。
    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
        tensors = tensors[0]  # type: ignore[assignment]

    # 继续允许调用旧方法，该方法不接受 indexing 关键字参数，用于向前兼容。
    #
    # 在发布后两周后删除此功能。
    kwargs = {} if indexing is None else {"indexing": indexing}
    # 调用 _VF 模块的 meshgrid 函数，传入张量和 kwargs 参数。
    return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]


# 定义 stft 函数，进行短时傅里叶变换（STFT）。
def stft(
    input: Tensor,
    n_fft: int,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: Optional[Tensor] = None,
    center: bool = True,
    pad_mode: str = "reflect",
    normalized: bool = False,
    onesided: Optional[bool] = None,
    return_complex: Optional[bool] = None,
) -> Tensor:
    # 短时傅里叶变换（STFT）函数。

    # 从版本 1.8.0 开始，对于实数输入，必须显式给出 return_complex 参数。
    # 对于实数输入，强烈推荐使用 return_complex=True，因为在将来的 pytorch 发布中，此函数将只返回复数张量。
    #
    # 使用 torch.view_as_real 可以用于恢复具有额外最后维度的实数张量，表示实部和虚部。
    #
    # 从版本 2.1 开始，如果未指定 window 参数，将提供警告。在未来版本中，将需要此属性。
    # 当前未提供 window 参数时，默认使用矩形窗口，可能导致不良伪影。考虑使用锥形窗口，例如 torch.hann_window。
    #
    # STFT 计算输入的短重叠窗口的傅里叶变换，从而提供随时间变化的信号频率分量。
    # 此函数的接口设计模仿（但并非完全兼容）librosa 的 stft 函数。
    #
    # 忽略可选的批处理维度，此方法计算如下表达式：
    # X[\omega, m] = \sum_{k = 0}^{\text{win\_length-1}}%
    # \text{window}[k]\ \text{input}[m \times \text{hop\_length} + k]\ %
    # \exp\left(- j \frac{2 \pi \cdot \omega k}{\text{n\_fft}}\right),

    # 返回类型为 Tensor
    where :math:`m` is the index of the sliding window, and :math:`\omega` is
    the frequency :math:`0 \leq \omega < \text{n\_fft}` for ``onesided=False``,
    or :math:`0 \leq \omega < \lfloor \text{n\_fft} / 2 \rfloor + 1` for ``onesided=True``.

    * :attr:`input` must be either a 1-D time sequence or a 2-D batch of time
      sequences.
    * If :attr:`hop_length` is ``None`` (default), it is treated as equal to
      ``floor(n_fft / 4)``.
    * If :attr:`win_length` is ``None`` (default), it is treated as equal to
      :attr:`n_fft`.
    * :attr:`window` can be a 1-D tensor of size :attr:`win_length`, e.g., from
      :meth:`torch.hann_window`. If :attr:`window` is ``None`` (default), it is
      treated as if having :math:`1` everywhere in the window. If
      :math:`\text{win\_length} < \text{n\_fft}`, :attr:`window` will be padded on
      both sides to length :attr:`n_fft` before being applied.
    * If :attr:`center` is ``True`` (default), :attr:`input` will be padded on
      both sides so that the :math:`t`-th frame is centered at time
      :math:`t \times \text{hop\_length}`. Otherwise, the :math:`t`-th frame
      begins at time  :math:`t \times \text{hop\_length}`.
    * :attr:`pad_mode` determines the padding method used on :attr:`input` when
      :attr:`center` is ``True``. See :meth:`torch.nn.functional.pad` for
      all available options. Default is ``"reflect"``.
    * If :attr:`onesided` is ``True`` (default for real input), only values for
      :math:`\omega` in :math:`\left[0, 1, 2, \dots, \left\lfloor
      \frac{\text{n\_fft}}{2} \right\rfloor + 1\right]` are returned because
      the real-to-complex Fourier transform satisfies the conjugate symmetry,
      i.e., :math:`X[m, \omega] = X[m, \text{n\_fft} - \omega]^*`.
      Note if the input or window tensors are complex, then :attr:`onesided`
      output is not possible.
    * If :attr:`normalized` is ``True`` (default is ``False``), the function
      returns the normalized STFT results, i.e., multiplied by :math:`(\text{frame\_length})^{-0.5}`.
    * If :attr:`return_complex` is ``True`` (default if input is complex), the
      return is a ``input.dim() + 1`` dimensional complex tensor. If ``False``,
      the output is a ``input.dim() + 2`` dimensional real tensor where the last
      dimension represents the real and imaginary components.

    Returns either a complex tensor of size :math:`(* \times N \times T)` if
    :attr:`return_complex` is true, or a real tensor of size :math:`(* \times N
    \times T \times 2)`. Where :math:`*` is the optional batch size of
    :attr:`input`, :math:`N` is the number of frequencies where STFT is applied
    and :math:`T` is the total number of frames used.

    .. warning::
      This function changed signature at version 0.4.1. Calling with the
      previous signature may cause error or return incorrect result.
    """
    Args:
        input (Tensor): 输入张量，形状为 `(B?, L)`，其中 `B?` 是可选的批次维度
        n_fft (int): 傅里叶变换的大小
        hop_length (int, optional): 相邻滑动窗口帧之间的距离。默认为 ``None`` （视为等于 ``floor(n_fft / 4)``）
        win_length (int, optional): 窗口帧和STFT滤波器的大小。默认为 ``None`` （视为等于 :attr:`n_fft`）
        window (Tensor, optional): 可选的窗口函数。必须是1维且 `<= n_fft` 的形状
            默认为 ``None`` （视为全1的窗口）
        center (bool, optional): 是否在两侧填充 :attr:`input`，使第 `t` 帧在时间 `t \times \text{hop\_length}` 处居中。默认为 ``True``
        pad_mode (str, optional): 当 :attr:`center` 为 ``True`` 时控制填充方法。默认为 ``"reflect"``
        normalized (bool, optional): 控制是否返回归一化的STFT结果。默认为 ``False``
        onesided (bool, optional): 控制是否只返回一半结果以避免实输入的冗余。对于实数 :attr:`input` 和 :attr:`window`，默认为 ``True``，否则为 ``False``
        return_complex (bool, optional): 是否返回复数张量，或者为实数张量添加额外的最后一维表示实部和虚部。

            .. versionchanged:: 2.0
               ``return_complex`` 现在是实输入的必选参数，默认转换为 ``True``。

            .. deprecated:: 2.0
               ``return_complex=False`` 已弃用，请改用 ``return_complex=True``
               请注意，对输出调用 :func:`torch.view_as_real` 将恢复弃用的输出格式。

    Returns:
        Tensor: 包含STFT结果的张量，形状为 `(B?, N, T, C?)`，其中
           - `B?` 是输入的可选批次维度。
           - `N` 是频率样本数，对于 `onesided=True` 为 `(n_fft // 2) + 1`，否则为 `n_fft`。
           - `T` 是帧数，对于 `center=True`，为 `1 + L // hop_length`，否则为 `1 + (L - n_fft) // hop_length`。
           - `C?` 是实部和虚部的可选长度为2的维度，在 `return_complex=False` 时存在。

    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            stft,
            (input,),
            input,
            n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            normalized=normalized,
            onesided=onesided,
            return_complex=return_complex,
        )
    # NOTE: Do not edit. This code will be removed once the forward-compatibility
    #       period is over for PR #73432
    # 如果需要居中数据，调整输入张量的形状和填充以进行处理
    if center:
        # 获取输入张量的维度数量
        signal_dim = input.dim()
        # 创建一个扩展形状的列表，以便在输入张量维度小于3时进行填充
        extended_shape = [1] * (3 - signal_dim) + list(input.size())
        # 计算填充量，将输入张量用指定填充模式填充
        pad = int(n_fft // 2)
        input = F.pad(input.view(extended_shape), [pad, pad], pad_mode)
        # 恢复原始的输入张量形状
        input = input.view(input.shape[-signal_dim:])
    # 调用底层的 VF 模块中的 stft 函数，进行短时傅里叶变换
    return _VF.stft(  # type: ignore[attr-defined]
        input,
        n_fft,
        hop_length,
        win_length,
        window,
        normalized,
        onesided,
        return_complex,
    )
istft = _add_docstr(
    torch.istft,
    "istft(input, n_fft, hop_length=None, win_length=None, window=None, center=True, "
    "normalized=False, onesided=None, length=None, return_complex=False) -> Tensor:\n"
    r"""
Inverse short time Fourier Transform. This is expected to be the inverse of :func:`~torch.stft`.

.. warning::
    From version 2.1, a warning will be provided if a :attr:`window` is
    not specified. In a future release, this attribute will be required.
    Please provide the same window used in the stft call.

It has the same parameters (+ additional optional parameter of :attr:`length`) and it should return the
least squares estimation of the original signal. The algorithm will check using the NOLA condition (
nonzero overlap).

Important consideration in the parameters :attr:`window` and :attr:`center` so that the envelope
created by the summation of all the windows is never zero at certain point in time. Specifically,
:math:`\sum_{t=-\infty}^{\infty} |w|^2[n-t\times hop\_length] \cancel{=} 0`.

Since :func:`~torch.stft` discards elements at the end of the signal if they do not fit in a frame,
``istft`` may return a shorter signal than the original signal (can occur if :attr:`center` is False
since the signal isn't padded). If `length` is given in the arguments and is longer than expected,
``istft`` will pad zeros to the end of the returned signal.

If :attr:`center` is ``True``, then there will be padding e.g. ``'constant'``, ``'reflect'``, etc.
Left padding can be trimmed off exactly because they can be calculated but right padding cannot be
calculated without additional information.

Example: Suppose the last window is:
``[17, 18, 0, 0, 0]`` vs ``[18, 0, 0, 0, 0]``

The :attr:`n_fft`, :attr:`hop_length`, :attr:`win_length` are all the same which prevents the calculation
of right padding. These additional values could be zeros or a reflection of the signal so providing
:attr:`length` could be useful. If :attr:`length` is ``None`` then padding will be aggressively removed
(some loss of signal).

[1] D. W. Griffin and J. S. Lim, "Signal estimation from modified short-time Fourier transform,"
IEEE Trans. ASSP, vol.32, no.2, pp.236-243, Apr. 1984.

Args:
    input (Tensor): The input tensor. Expected to be in the format of :func:`~torch.stft`,
        output. That is a complex tensor of shape `(B?, N, T)` where

        - `B?` is an optional batch dimension
        - `N` is the number of frequency samples, `(n_fft // 2) + 1`
          for onesided input, or otherwise `n_fft`.
        - `T` is the number of frames, `1 + length // hop_length` for centered stft,
          or `1 + (length - n_fft) // hop_length` otherwise.

        .. versionchanged:: 2.0
            Real datatype inputs are no longer supported. Input must now have a
            complex datatype, as returned by ``stft(..., return_complex=True)``.
    n_fft (int): Size of Fourier transform
"""
)
    hop_length (Optional[int]): 滑动窗口相邻帧之间的距离。
        (默认值: ``n_fft // 4``)
    win_length (Optional[int]): 窗口帧的大小和STFT滤波器的大小。
        (默认值: ``n_fft``)
    window (Optional[torch.Tensor]): 可选的窗口函数。
        必须是1维的且长度 `<= n_fft`。
        (默认值: ``torch.ones(win_length)``)
    center (bool): 输入信号是否在两侧填充，使得第 `t` 个帧在时间 `t \times \text{hop\_length}` 处居中。
        (默认值: ``True``)
    normalized (bool): STFT是否已归一化。
        (默认值: ``False``)
    onesided (Optional[bool]): STFT是否是单边的。
        (默认值: 如果输入大小中 `n_fft != fft_size` 则为 ``True``)
    length (Optional[int]): 要修剪信号的长度（即原始信号的长度）。
        对于居中的STFT，默认为 `(T - 1) * hop_length`，否则为 `n_fft + (T - 1) * hop_length`，其中 `T` 是输入帧的数量。
    return_complex (Optional[bool]): 输出是否应该是复数，或者输入应被假定为来自实信号和窗口。
        注意这与 ``onesided=True`` 不兼容。
        (默认值: ``False``)
if TYPE_CHECKING:
    # 如果是类型检查阶段，_unique_impl_out 类型为 Any，表示返回任意类型的值
    _unique_impl_out = Any
else:
    # 如果不是类型检查阶段，_unique_impl_out 类型为 Tuple[Tensor, Tensor, Tensor]
    # 表示返回三个 Tensor 类型的值，分别对应 unique 结果、inverse 结果和 counts 结果
    _unique_impl_out = Tuple[Tensor, Tensor, Tensor]


def _unique_impl(
    input: Tensor,
    sorted: bool = True,
    return_inverse: bool = False,
    return_counts: bool = False,
    dim: Optional[int] = None,
) -> _unique_impl_out:
    r"""unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None) -> Tuple[Tensor, Tensor, Tensor]

    返回输入张量的唯一元素。

    .. note:: 与 :func:`torch.unique_consecutive` 不同，此函数还消除了非连续的重复值。

    .. note:: 当前在 CUDA 实现和 CPU 实现中，`torch.unique` 总是在开始时对张量进行排序，
        而不管 `sort` 参数如何。排序可能会很慢，因此如果输入张量已经排序，建议使用
        :func:`torch.unique_consecutive`，它避免了排序。

    Args:
        input (Tensor): 输入张量
        sorted (bool): 是否在返回输出之前对唯一元素进行升序排序。
        return_inverse (bool): 是否还返回原始输入中元素在返回的唯一列表中的索引。
        return_counts (bool): 是否还返回每个唯一元素的计数。
        dim (int, optional): 操作的维度。如果为 ``None``，则返回扁平化输入的唯一值。否则，
            给定维度的每个张量被视为要应用唯一操作的元素之一。详见示例以获取更多细节。默认为 ``None``

    Returns:
        Tuple[Tensor, Tensor, Tensor]: 返回三个张量，分别是唯一值、inverse 结果和 counts 结果
    """
    """
    Returns:
        (Tensor, Tensor (optional), Tensor (optional)): 返回一个张量或张量元组，包含以下内容：

            - **output** (*Tensor*): 唯一标量元素的输出列表。
            - **inverse_indices** (*Tensor*): （可选）如果
              :attr:`return_inverse` 为 True，则会返回一个额外的张量（与输入形状相同），
              表示原始输入中元素在输出中的索引；否则，此函数将只返回一个张量。
            - **counts** (*Tensor*): （可选）如果
              :attr:`return_counts` 为 True，则会返回一个额外的张量（与输出或指定维度的输出大小相同），
              表示每个唯一值或张量的出现次数。

    """
    # 如果输入具有 torch 函数的定义
    if has_torch_function_unary(input):
        return handle_torch_function(
            unique,
            (input,),
            input,
            sorted=sorted,
            return_inverse=return_inverse,
            return_counts=return_counts,
            dim=dim,
        )

    # 如果指定了维度 dim
    if dim is not None:
        # 使用 _VF.unique_dim 处理
        output, inverse_indices, counts = _VF.unique_dim(
            input,
            dim,
            sorted=sorted,
            return_inverse=return_inverse,
            return_counts=return_counts,
        )
    else:
        # 使用 torch._unique2 处理
        output, inverse_indices, counts = torch._unique2(
            input,
            sorted=sorted,
            return_inverse=return_inverse,
            return_counts=return_counts,
        )
    # 返回处理后的结果
    return output, inverse_indices, counts
# 定义一个内部函数 `_unique_consecutive_impl`，用于消除连续组中除第一个元素外的所有等效元素

    r"""Eliminates all but the first element from every consecutive group of equivalent elements.
    
    .. note:: 该函数与 `torch.unique` 不同，它仅消除连续的重复值。其语义类似于 C++ 中的 `std::unique`。

    Args:
        input (Tensor): 输入张量
        return_inverse (bool): 是否返回原始输入中元素在返回的唯一列表中的索引
        return_counts (bool): 是否返回每个唯一元素的出现次数
        dim (int): 应用唯一操作的维度。如果为 ``None``，则返回扁平化输入的唯一值。默认为 ``None``

    Returns:
        (Tensor, Tensor (optional), Tensor (optional)): 包含一个或多个张量的元组，其中

            - **output** (*Tensor*): 唯一标量元素的输出列表
            - **inverse_indices** (*Tensor*): (可选) 如果 :attr:`return_inverse` 为 True，则返回一个额外的张量（与输入形状相同），表示原始输入中的元素在输出中的映射位置；否则，该函数将只返回一个张量。
            - **counts** (*Tensor*): (可选) 如果 :attr:`return_counts` 为 True，则返回一个额外的张量（与输出或输出大小（如果指定了 dim）相同的形状），表示每个唯一值或张量的出现次数。

    Example::

        >>> x = torch.tensor([1, 1, 2, 2, 3, 1, 1, 2])
        >>> output = torch.unique_consecutive(x)
        >>> output
        tensor([1, 2, 3, 1, 2])

        >>> output, inverse_indices = torch.unique_consecutive(x, return_inverse=True)
        >>> output
        tensor([1, 2, 3, 1, 2])
        >>> inverse_indices
        tensor([0, 0, 1, 1, 2, 3, 3, 4])

        >>> output, counts = torch.unique_consecutive(x, return_counts=True)
        >>> output
        tensor([1, 2, 3, 1, 2])
        >>> counts
        tensor([2, 2, 1, 2, 1])
    """
    # 检查输入是否具有 torch 函数的一元处理方法
    if has_torch_function_unary(input):
        # 处理 torch 函数调用
        return handle_torch_function(
            unique_consecutive,
            (input,),
            input,
            return_inverse=return_inverse,
            return_counts=return_counts,
            dim=dim,
        )
    # 调用底层的 C++ 函数 _VF.unique_consecutive，获取输出、反向索引和计数
    output, inverse_indices, counts = _VF.unique_consecutive(
        input, return_inverse=return_inverse, return_counts=return_counts, dim=dim
    )  # type: ignore[attr-defined]
    # 返回输出、反向索引和计数
    return output, inverse_indices, counts


def _return_counts(
    input,
    sorted=True,
    return_inverse=False,
    return_counts=False,
    dim=None,
):
    # 检查输入张量是否有自定义的 PyTorch 函数，如果有则调用 _unique_impl 处理输入
    if has_torch_function_unary(input):
        # 调用 _unique_impl 处理输入张量，返回唯一值、排序结果、反向索引和计数（根据参数选择）
        return _unique_impl(input, sorted, return_inverse, return_counts, dim)
    
    # 否则，调用 _unique_impl 处理输入张量，返回唯一值和计数（根据参数选择），忽略返回的反向索引
    output, _, counts = _unique_impl(input, sorted, return_inverse, return_counts, dim)
    # 返回处理结果中的唯一值和计数
    return output, counts
# 根据输入的张量 input，根据指定的参数进行唯一化操作，返回唯一化后的结果张量 output
# 还根据 return_inverse 和 return_counts 参数返回其他附加信息
def _return_output(
    input,
    sorted=True,
    return_inverse=False,
    return_counts=False,
    dim=None,
):
    # type: (Tensor, bool, bool, bool, Optional[int]) -> Tensor

    # 如果输入的张量有 Torch 函数的重载版本，则调用 _unique_impl 函数处理唯一化
    if has_torch_function_unary(input):
        return _unique_impl(input, sorted, return_inverse, return_counts, dim)

    # 否则调用 _unique_impl 函数处理唯一化，获取输出结果张量 output
    output, _, _ = _unique_impl(input, sorted, return_inverse, return_counts, dim)
    return output


# 根据输入的张量 input，根据指定的参数进行唯一化操作，并返回唯一值及其在原张量中的索引信息
def _return_inverse(
    input,
    sorted=True,
    return_inverse=False,
    return_counts=False,
    dim=None,
):
    # type: (Tensor, bool, bool, bool, Optional[int]) -> Tuple[Tensor, Tensor]

    # 如果输入的张量有 Torch 函数的重载版本，则调用 _unique_impl 函数处理唯一化
    if has_torch_function_unary(input):
        return _unique_impl(input, sorted, return_inverse, return_counts, dim)

    # 否则调用 _unique_impl 函数处理唯一化，获取唯一值、逆序索引和计数信息
    output, inverse_indices, _ = _unique_impl(
        input, sorted, return_inverse, return_counts, dim
    )
    return output, inverse_indices


# 创建一个根据 return_counts 参数分派的函数，用于唯一化操作的结果处理
_return_inverse_false = boolean_dispatch(
    arg_name="return_counts",
    arg_index=3,
    default=False,
    if_true=_return_counts,
    if_false=_return_output,
    module_name=__name__,
    func_name="unique",
)

# 创建一个根据 return_counts 参数分派的函数，用于唯一化操作的结果处理
_return_inverse_true = boolean_dispatch(
    arg_name="return_counts",
    arg_index=3,
    default=False,
    if_true=_unique_impl,
    if_false=_return_inverse,
    module_name=__name__,
    func_name="unique",
)

# unique 函数根据 return_inverse 参数的真假，选择合适的处理函数进行唯一化操作
# 并使用 boolean_dispatch 函数动态分派，解析 TorchScript 中的输出类型
unique = boolean_dispatch(
    arg_name="return_inverse",
    arg_index=2,
    default=False,
    if_true=_return_inverse_true,
    if_false=_return_inverse_false,
    module_name=__name__,
    func_name="unique",
)
unique.__doc__ = _unique_impl.__doc__


# 根据输入的张量 input，根据指定的参数进行连续唯一化操作，并返回唯一值及其计数信息
def _consecutive_return_counts(
    input,
    return_inverse=False,
    return_counts=False,
    dim=None,
):
    # type: (Tensor, bool, bool, Optional[int]) -> Tuple[Tensor, Tensor]

    # 如果输入的张量有 Torch 函数的重载版本，则调用 _unique_consecutive_impl 函数处理连续唯一化
    if has_torch_function_unary(input):
        return _unique_consecutive_impl(input, return_inverse, return_counts, dim)

    # 否则调用 _unique_consecutive_impl 函数处理连续唯一化，获取唯一值及其计数信息
    output, _, counts = _unique_consecutive_impl(
        input, return_inverse, return_counts, dim
    )
    return output, counts


# 根据输入的张量 input，根据指定的参数进行连续唯一化操作，并返回唯一化后的结果张量
def _consecutive_return_output(
    input,
    return_inverse=False,
    return_counts=False,
    dim=None,
):
    # type: (Tensor, bool, bool, Optional[int]) -> Tensor

    # 如果输入的张量有 Torch 函数的重载版本，则调用 _unique_consecutive_impl 函数处理连续唯一化
    if has_torch_function_unary(input):
        return _unique_consecutive_impl(input, return_inverse, return_counts, dim)

    # 否则调用 _unique_consecutive_impl 函数处理连续唯一化，获取唯一化后的结果张量
    output, _, _ = _unique_consecutive_impl(input, return_inverse, return_counts, dim)
    return output


# 根据输入的张量 input，根据指定的参数进行连续唯一化操作，并返回唯一值及其在原张量中的逆序索引信息
def _consecutive_return_inverse(
    input,
    return_inverse=False,
    return_counts=False,
    dim=None,
):
    # type: (Tensor, bool, bool, Optional[int]) -> Tuple[Tensor, Tensor]

    # 如果输入的张量有 Torch 函数的重载版本，则调用 _unique_consecutive_impl 函数处理连续唯一化
    if has_torch_function_unary(input):
        return _unique_consecutive_impl(input, return_inverse, return_counts, dim)
    # 调用_unique_consecutive_impl函数进行处理，返回三个结果赋值给output, inverse_indices, _
    output, inverse_indices, _ = _unique_consecutive_impl(
        # 输入参数input是要处理的数据
        input,
        # 如果return_inverse为True，则返回输入数据在输出中的索引
        return_inverse,
        # 如果return_counts为True，则返回每个唯一元素的出现次数
        return_counts,
        # dim是指定维度，用于在输入中找到唯一的连续元素
        dim
    )
    # 返回处理后的结果output和inverse_indices
    return output, inverse_indices
# 使用布尔分发函数创建一个变量，根据给定参数和模块信息决定其值
_consecutive_return_inverse_false = boolean_dispatch(
    arg_name="return_counts",   # 参数名为return_counts
    arg_index=1,                # 参数在参数列表中的索引为1
    default=False,              # 默认取值为False
    if_true=_consecutive_return_counts,    # 如果return_counts为True，使用_consecutive_return_counts函数处理
    if_false=_consecutive_return_output,   # 如果return_counts为False，使用_consecutive_return_output函数处理
    module_name=__name__,       # 当前模块的名称
    func_name="unique_consecutive",     # 函数名为unique_consecutive
)

# 使用布尔分发函数创建另一个变量，参数和处理逻辑与上述类似
_consecutive_return_inverse_true = boolean_dispatch(
    arg_name="return_counts",   # 参数名为return_counts
    arg_index=1,                # 参数在参数列表中的索引为1
    default=False,              # 默认取值为False
    if_true=_unique_consecutive_impl,   # 如果return_counts为True，使用_unique_consecutive_impl函数处理
    if_false=_consecutive_return_inverse,   # 如果return_counts为False，使用_consecutive_return_inverse函数处理
    module_name=__name__,       # 当前模块的名称
    func_name="unique_consecutive",     # 函数名为unique_consecutive
)

# 定义函数unique_consecutive，根据参数return_inverse的不同值选择不同的处理函数
unique_consecutive = boolean_dispatch(
    arg_name="return_inverse",  # 参数名为return_inverse
    arg_index=2,                # 参数在参数列表中的索引为2
    default=False,              # 默认取值为False
    if_true=_consecutive_return_inverse_true,    # 如果return_inverse为True，使用_consecutive_return_inverse_true函数处理
    if_false=_consecutive_return_inverse_false,  # 如果return_inverse为False，使用_consecutive_return_inverse_false函数处理
    module_name=__name__,       # 当前模块的名称
    func_name="unique_consecutive",     # 函数名为unique_consecutive
)
# 将unique_consecutive函数的文档字符串设为_unique_consecutive_impl函数的文档字符串
unique_consecutive.__doc__ = _unique_consecutive_impl.__doc__

if TYPE_CHECKING:
    pass
    # 由于类型注解会破坏JIT重载，目前无法在此处使用良好的类型注解。因此暂时保留未类型化的mypy注释。
else:
    # 使用装饰器@overload为函数tensordot定义多个重载，根据不同的参数类型选择不同的处理逻辑
    @overload
    def tensordot(
        a,
        b,
        dims: int = 2,
        out: Optional[torch.Tensor] = None,
    ):
        pass

    @overload
    def tensordot(  # noqa: F811
        a,
        b,
        dims: Tuple[List[int], List[int]],
        out: Optional[torch.Tensor] = None,
    ):
        pass

    @overload
    def tensordot(  # noqa: F811
        a,
        b,
        dims: List[List[int]],
        out: Optional[torch.Tensor] = None,
    ):
        pass

    @overload
    def tensordot(  # noqa: F811
        a,
        b,
        dims: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ):
        pass

# 定义函数tensordot，实现两个张量的多维度收缩运算
def tensordot(  # noqa: F811
    a,
    b,
    dims=2,                     # dims参数默认为2
    out: Optional[torch.Tensor] = None,   # out参数为可选的torch.Tensor类型
):
    r"""Returns a contraction of a and b over multiple dimensions.

    :attr:`tensordot` implements a generalized matrix product.

    Args:
      a (Tensor): Left tensor to contract
      b (Tensor): Right tensor to contract
      dims (int or Tuple[List[int], List[int]] or List[List[int]] containing two lists or Tensor): number of dimensions to
         contract or explicit lists of dimensions for :attr:`a` and
         :attr:`b` respectively

    When called with a non-negative integer argument :attr:`dims` = :math:`d`, and
    the number of dimensions of :attr:`a` and :attr:`b` is :math:`m` and :math:`n`,
    respectively, :func:`~torch.tensordot` computes

    .. math::
        r_{i_0,...,i_{m-d}, i_d,...,i_n}
          = \sum_{k_0,...,k_{d-1}} a_{i_0,...,i_{m-d},k_0,...,k_{d-1}} \times b_{k_0,...,k_{d-1}, i_d,...,i_n}.

    When called with :attr:`dims` of the list form, the given dimensions will be contracted
    in place of the last :math:`d` of :attr:`a` and the first :math:`d` of :math:`b`. The sizes
    """
    """
    If either `a` or `b` has a torch function defined for variadic arguments,
    dispatches the call to `handle_torch_function`.

    If `dims` is not one of the expected types (tuple, list, torch.Tensor, int, torch.SymInt),
    raises a RuntimeError with an appropriate error message.

    Initializes `dims_a` and `dims_b` as empty lists.

    If `dims` is a tuple or list, assigns `dims_a` and `dims_b` accordingly.

    If `dims` is a torch.Tensor, interprets it:
    - If it has more than one element, assigns `dims_a` and `dims_b` based on its contents.
    - If it has a single element, interprets it as a scalar and assigns `dims_a` and `dims_b` accordingly.

    If `dims` is an int or torch.SymInt, interprets it:
    - If it is negative, raises a RuntimeError.
    - Assigns `dims_a` and `dims_b` as ranges based on its value.

    Validates `dims` against the dimensions of `a` and `b` to ensure they are within valid ranges.

    If `out` is None, calls `_VF.tensordot(a, b, dims_a, dims_b)` and returns the result.
    If `out` is provided, calls `_VF.tensordot(a, b, dims_a, dims_b, out=out)` and returns the result.

    Parameters:
    - a (Tensor): First tensor operand for tensordot operation.
    - b (Tensor): Second tensor operand for tensordot operation.
    - dims (tuple, list, int, torch.Tensor, torch.SymInt): Dimensions over which to contract tensors.
    - out (Tensor, optional): Output tensor to store the result (default: None).

    Returns:
    - Tensor: Result tensor after performing tensordot operation on `a` and `b` over specified dimensions.
    """
def cdist(x1, x2, p=2.0, compute_mode="use_mm_for_euclid_dist_if_necessary"):
    # type: (Tensor, Tensor, float, str) -> (Tensor)
    r"""Computes batched the p-norm distance between each pair of the two collections of row vectors.

    Args:
        x1 (Tensor): First input tensor containing row vectors.
        x2 (Tensor): Second input tensor containing row vectors.
        p (float, optional): The norm degree. Default: 2.0.
        compute_mode (str, optional): Unused argument. Default: "use_mm_for_euclid_dist_if_necessary".

    Returns:
        Tensor: A tensor of shape (N, M) where N is the number of vectors in x1 and M is the number of vectors in x2.
        Each element (i, j) contains the p-norm distance between x1[i] and x2[j].

    Note:
        This function computes the pairwise p-norm distances between vectors in x1 and x2. 
        The distance between two vectors u and v is defined as:
        \( \|u - v\|_p = \left( \sum_i |u_i - v_i|^p \right)^{1/p} \)

    Example::
    
        >>> import torch
        >>> x1 = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        >>> x2 = torch.tensor([[1.0, 0.0], [1.0, 2.0]])
        >>> torch.cdist(x1, x2, p=2.0)
        tensor([[1.0000, 2.2361],
                [1.4142, 1.0000]])
    """
    # This wrapper exists to support variadic args.
    if has_torch_function(x1, x2):
        return handle_torch_function(cdist, x1, x2, p, compute_mode)
    return torch._C._VariableFunctions.cdist(x1, x2, p)  # type: ignore[attr-defined]
    """
    Args:
        x1 (Tensor): 输入张量，形状为 :math:`B \times P \times M`.
        x2 (Tensor): 输入张量，形状为 :math:`B \times R \times M`.
        p: 计算每对向量之间的 p-范数距离，取值范围为 :math:`\in [0, \infty]`.
        compute_mode:
            'use_mm_for_euclid_dist_if_necessary' - 如果 P > 25 或 R > 25，则使用矩阵乘法方法计算欧氏距离（p = 2）
            'use_mm_for_euclid_dist' - 总是使用矩阵乘法方法计算欧氏距离（p = 2）
            'donot_use_mm_for_euclid_dist' - 永远不使用矩阵乘法方法计算欧氏距离（p = 2）
            默认值为 use_mm_for_euclid_dist_if_necessary.

    如果 x1 的形状为 :math:`B \times P \times M`，x2 的形状为 :math:`B \times R \times M`，则输出形状为 :math:`B \times P \times R`。

    当 :math:`p \in (0, \infty)` 时，此函数等效于 `scipy.spatial.distance.cdist(input,'minkowski', p=p)`。
    当 :math:`p = 0` 时，等效于 `scipy.spatial.distance.cdist(input, 'hamming') * M`。
    当 :math:`p = \infty` 时，最接近的 scipy 函数是 `scipy.spatial.distance.cdist(xn, lambda x, y: np.abs(x - y).max())`。

    Example:

        >>> a = torch.tensor([[0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059]])
        >>> a
        tensor([[ 0.9041,  0.0196],
                [-0.3108, -2.4423],
                [-0.4821,  1.0590]])
        >>> b = torch.tensor([[-2.1763, -0.4713], [-0.6986,  1.3702]])
        >>> b
        tensor([[-2.1763, -0.4713],
                [-0.6986,  1.3702]])
        >>> torch.cdist(a, b, p=2)
        tensor([[3.1193, 2.0959],
                [2.7138, 3.8322],
                [2.2830, 0.3791]])
    """
    # 如果输入的张量 x1 和 x2 具有 torch 函数的变量特性，则调用处理 torch 函数的方法
    if has_torch_function_variadic(x1, x2):
        return handle_torch_function(
            cdist, (x1, x2), x1, x2, p=p, compute_mode=compute_mode
        )
    # 根据 compute_mode 的值选择相应的计算模式
    if compute_mode == "use_mm_for_euclid_dist_if_necessary":
        return _VF.cdist(x1, x2, p, None)  # type: ignore[attr-defined]
    elif compute_mode == "use_mm_for_euclid_dist":
        return _VF.cdist(x1, x2, p, 1)  # type: ignore[attr-defined]
    elif compute_mode == "donot_use_mm_for_euclid_dist":
        return _VF.cdist(x1, x2, p, 2)  # type: ignore[attr-defined]
    else:
        # 如果 compute_mode 不是预期的值，则抛出 ValueError 异常
        raise ValueError(f"{compute_mode} is not a valid value for compute_mode")
# This function ensures that each input tensor has at least one dimension.
# If the input is a single tensor, it is converted to a tuple for processing.
def atleast_1d(*tensors):
    # This block checks if any of the tensors have a Torch function defined for them.
    if has_torch_function(tensors):
        # If Torch function exists, it handles the function call and returns the result.
        return handle_torch_function(atleast_1d, tensors, *tensors)
    # If only one tensor is provided, convert it into a tuple for uniform processing.
    if len(tensors) == 1:
        tensors = tensors[0]
    # Call the underlying function in Torch's C++ backend to ensure at least 1-dimensional view.
    return _VF.atleast_1d(tensors)  # type: ignore[attr-defined]


# This function ensures that each input tensor has at least two dimensions.
# If the input is a single tensor, it is converted to a tuple for processing.
def atleast_2d(*tensors):
    # This block checks if any of the tensors have a Torch function defined for them.
    if has_torch_function(tensors):
        # If Torch function exists, it handles the function call and returns the result.
        return handle_torch_function(atleast_2d, tensors, *tensors)
    # If only one tensor is provided, convert it into a tuple for uniform processing.
    if len(tensors) == 1:
        tensors = tensors[0]
    # Call the underlying function in Torch's C++ backend to ensure at least 2-dimensional view.
    return _VF.atleast_2d(tensors)  # type: ignore[attr-defined]


# This function ensures that each input tensor has at least three dimensions.
# If the input is a single tensor, it is converted to a tuple for processing.
def atleast_3d(*tensors):
    # This block checks if any of the tensors have a Torch function defined for them.
    if has_torch_function(tensors):
        # If Torch function exists, it handles the function call and returns the result.
        return handle_torch_function(atleast_3d, tensors, *tensors)
    # If only one tensor is provided, convert it into a tuple for uniform processing.
    if len(tensors) == 1:
        tensors = tensors[0]
    # Call the underlying function in Torch's C++ backend to ensure at least 3-dimensional view.
    return _VF.atleast_3d(tensors)  # type: ignore[attr-defined]
    # 这个函数是为了支持多参数输入而存在的包装器
    if has_torch_function(tensors):
        # 如果输入的张量有torch函数支持，则调用处理torch函数的方法，返回处理结果
        return handle_torch_function(atleast_3d, tensors, *tensors)
    # 如果输入的张量数量为1，则将其解包为单个张量
    if len(tensors) == 1:
        tensors = tensors[0]
    # 调用底层的PyTorch函数实现至少是3维的张量
    return _VF.atleast_3d(tensors)  # type: ignore[attr-defined]
if TYPE_CHECKING:
    pass
    # 如果在类型检查模式下，略过以下代码块
    # 无法通过改名 norm() 为 _norm_impl() 的方式来避免破坏 JIT 的重载。
    # 因此，暂时在 mypy 中保留未类型化的版本。
    #    def norm(input: Tensor,
    #             p: Optional[Union[str, Number]] = "fro",
    #             dim: Optional[Union[int, List[int]]] = None,
    #             keepdim: bool = False,
    #             out: Optional[Tensor] = None,
    #             dtype: _dtype = None) -> Tensor:
    #        return _norm_impl(input, p, dim, keepdim, out, dtype)
else:
    # 在非类型检查模式下，定义 norm 函数的类型重载

    # TODO: 当 https://github.com/pytorch/pytorch/issues/33782 问题解决后，将 dim 类型标注为 BroadcastingList
    @overload
    def norm(
        input,
        p="fro",
        dim=None,
        keepdim=False,
        out=None,
        dtype=None,
    ):
        # type: (Tensor, str, Optional[List[int]], bool, Optional[Tensor], Optional[int]) -> Tensor
        pass

    @overload
    def norm(  # noqa: F811
        input,
        p="fro",
        dim=None,
        keepdim=False,
        out=None,
        dtype=None,
    ):
        # type: (Tensor, Optional[number], Optional[List[int]], bool, Optional[Tensor], Optional[int]) -> Tensor
        pass

    @overload
    def norm(  # noqa: F811
        input,
        p="fro",
        dim=None,
        keepdim=False,
        out=None,
        dtype=None,
    ):
        # type: (Tensor, Optional[number], Optional[int], bool, Optional[Tensor], Optional[int]) -> Tensor
        pass

    @overload
    def norm(  # noqa: F811
        input,
        p="fro",
        dim=None,
        keepdim=False,
        out=None,
        dtype=None,
    ):
        # type: (Tensor, str, Optional[int], bool, Optional[Tensor], Optional[int]) -> Tensor
        pass

# 定义 norm 函数，用于计算给定张量的矩阵范数或向量范数
def norm(
    input,
    p: Optional[Union[float, str]] = "fro",
    dim=None,
    keepdim=False,
    out=None,
    dtype=None,
):
    r"""Returns the matrix norm or vector norm of a given tensor.

    .. warning::

        torch.norm is deprecated and may be removed in a future PyTorch release.
        Its documentation and behavior may be incorrect, and it is no longer
        actively maintained.

        Use :func:`torch.linalg.vector_norm` when computing vector norms and
        :func:`torch.linalg.matrix_norm` when computing matrix norms.
        For a function with a similar behavior as this one see :func:`torch.linalg.norm`.
        Note, however, the signature for these functions is slightly different than the
        signature for ``torch.norm``.
    """
    Args:
        input (Tensor): 输入张量。其数据类型必须是浮点类型或复数类型。对于复数输入，使用每个元素的绝对值来计算范数。如果输入是复数且未指定 `dtype` 或 `out`，结果的数据类型将是相应的浮点类型（例如，如果 `input` 是 complexfloat，则为 float）。

        p (int, float, inf, -inf, 'fro', 'nuc', optional): 范数的阶数。默认为 `'fro'`。可以计算以下范数：

            ======  ==============  ==========================
            ord     矩阵范数         向量范数
            ======  ==============  ==========================
            'fro'   弗罗贝尼乌斯范数  --
            'nuc'   核范数          --
            Number  --              sum(abs(x)**ord)**(1./ord)
            ======  ==============  ==========================

            向量范数可以跨任意数量的维度计算。输入的对应维度被展平为一个维度，并在展平的维度上计算范数。

            在所有情况下，除非 `dim` 是三个或更多维的列表，否则弗罗贝尼乌斯范数产生与 `p=2` 相同的结果，此时弗罗贝尼乌斯范数会抛出错误。

            核范数只能在精确地两个维度上计算。

        dim (int, tuple of ints, list of ints, optional):
            指定要沿着其计算范数的 :attr:`input` 的维度或维度。如果 :attr:`dim` 是 `None`，则将在 :attr:`input` 的所有维度上计算范数。如果由 :attr:`p` 指示的范数类型不支持指定数量的维度，则会发生错误。

        keepdim (bool, optional): 输出张量是否保留 :attr:`dim`。如果 :attr:`dim` = `None` 并且 :attr:`out` = `None`，则忽略此参数。默认为 `False`。

        out (Tensor, optional): 输出张量。如果 :attr:`dim` = `None` 并且 :attr:`out` = `None`，则忽略此参数。

        dtype (:class:`torch.dtype`, optional): 返回张量的期望数据类型。如果指定，则在执行操作时将输入张量转换为 :attr:`dtype`。默认为 `None`。

    .. note::
        即使 `p='fro'` 支持任意数量的维度，弗罗贝尼乌斯范数的真正数学定义仅适用于具有确切两个维度的张量。使用 `ord='fro'` 的 :func:`torch.linalg.matrix_norm` 符合数学定义，因为它只能应用于确切的两个维度。
    # 如果输入具有 torch 函数的一元操作，调用处理 torch 函数的方法
    if has_torch_function_unary(input):
        return handle_torch_function(
            norm, (input,), input, p=p, dim=dim, keepdim=keepdim, out=out, dtype=dtype
        )

    # 注意：下面的重复代码和奇怪的 Python 语法是为了兼容 TorchScript。
    #      若要看到更简洁的实现，请参考 `_refs/__init__.py` 中相关的函数。

    # 对于 MPS 或稀疏张量，我们不执行下面的操作
    if input.layout == torch.strided and input.device.type in (
        "cpu",
        "cuda",
        "meta",
        torch.utils.backend_registration._privateuse1_backend_name,
    ):
        # 检查是否指定了维度参数
        if dim is not None:
            # 如果 dim 是整数或 torch.SymInt 类型，则将其转换为列表形式
            if isinstance(dim, (int, torch.SymInt)):
                _dim = [dim]
            else:
                _dim = dim
        else:
            # 如果未指定 dim，则将 _dim 设置为 None
            _dim = None  # type: ignore[assignment]

        # 检查 p 是否为字符串类型
        if isinstance(p, str):
            # 如果 p 是 "fro" 并且满足以下条件之一：
            # - dim 未指定
            # - dim 是整数或 torch.SymInt 类型
            # - dim 的长度小于等于 2
            if p == "fro" and (
                dim is None or isinstance(dim, (int, torch.SymInt)) or len(dim) <= 2
            ):
                # 如果 out 未指定，则计算 input 的 Frobenius 范数
                if out is None:
                    return torch.linalg.vector_norm(
                        input, 2, _dim, keepdim, dtype=dtype
                    )
                else:
                    # 如果 out 已指定，则计算 input 的 Frobenius 范数并将结果存储到 out 中
                    return torch.linalg.vector_norm(
                        input, 2, _dim, keepdim, dtype=dtype, out=out
                    )

            # 如果 p 是字符串，但不是 "fro"，则抛出错误
            # 以下代码实现了 nuclear norm 的调用或者带有特定参数调用 matrix_norm，这可能会导致错误
            if _dim is None:
                _dim = list(range(input.ndim))
            if out is None:
                # 如果 out 未指定，则调用 matrix_norm 计算 input 的矩阵范数
                return torch.linalg.matrix_norm(input, p, _dim, keepdim, dtype=dtype)
            else:
                # 如果 out 已指定，则计算 input 的矩阵范数并将结果存储到 out 中
                return torch.linalg.matrix_norm(
                    input, p, _dim, keepdim, dtype=dtype, out=out
                )
        else:
            # 如果 p 不是字符串类型，则默认 _p 为 2.0（即默认为 L2 范数）
            _p = 2.0 if p is None else p
            if out is None:
                # 如果 out 未指定，则计算 input 的向量范数
                return torch.linalg.vector_norm(input, _p, _dim, keepdim, dtype=dtype)
            else:
                # 如果 out 已指定，则计算 input 的向量范数并将结果存储到 out 中
                return torch.linalg.vector_norm(
                    input, _p, _dim, keepdim, dtype=dtype, out=out
                )

    ndim = input.dim()

    # 捕获默认情况
    if dim is None and out is None and dtype is None and p is not None:
        # 如果 p 是字符串 "fro"，则计算 input 的 Frobenius 范数
        if isinstance(p, str):
            if p == "fro":
                return _VF.frobenius_norm(input, dim=(), keepdim=keepdim)
        # 如果 p 不是字符串类型，则将 _dim 设置为 input 的所有维度列表
        if not isinstance(p, str):
            _dim = list(range(ndim))
            # 调用 _VF.norm 计算 input 的范数，并指定维度为 _dim
            return _VF.norm(input, p, dim=_dim, keepdim=keepdim)  # type: ignore[attr-defined]

    # TODO: 当 https://github.com/pytorch/pytorch/issues/33782 问题解决后
    # 移除对 dim 为整数的重载，替换为 BroadcastingList1
    # 并移除接下来四行代码，将 _dim 替换为 dim
    if dim is not None:
        # 如果 dim 不为 None，则根据其类型确定 _dim
        if isinstance(dim, (int, torch.SymInt)):
            _dim = [dim]
        else:
            _dim = dim
    else:
        # 如果 dim 为 None，则将 _dim 设置为 None
        _dim = None  # type: ignore[assignment]
    # 检查参数 p 是否为字符串类型
    if isinstance(p, str):
        # 如果 p 是字符串 "fro"
        if p == "fro":
            # 如果 dtype 参数不为 None，则抛出数值错误
            if dtype is not None:
                raise ValueError("dtype argument is not supported in frobenius norm")

            # 如果 _dim 为 None，则设为所有维度的列表
            if _dim is None:
                _dim = list(range(ndim))

            # 如果 out 参数为 None，则调用 _VF.frobenius_norm 函数并返回结果
            if out is None:
                return _VF.frobenius_norm(input, _dim, keepdim=keepdim)  # type: ignore[arg-type]
            else:
                # 否则，调用 _VF.frobenius_norm 函数并返回结果，带有输出参数 out
                return _VF.frobenius_norm(input, _dim, keepdim=keepdim, out=out)  # type: ignore[arg-type]
        
        # 如果 p 是字符串 "nuc"
        elif p == "nuc":
            # 如果 dtype 参数不为 None，则抛出数值错误
            if dtype is not None:
                raise ValueError("dtype argument is not supported in nuclear norm")

            # 如果 _dim 为 None，则设为保持维度的标志 keepdim
            if _dim is None:
                if out is None:
                    # 如果 out 参数为 None，则调用 _VF.nuclear_norm 函数并返回结果
                    return _VF.nuclear_norm(input, keepdim=keepdim)  # type: ignore[arg-type]
                else:
                    # 否则，调用 _VF.nuclear_norm 函数并返回结果，带有输出参数 out
                    return _VF.nuclear_norm(input, keepdim=keepdim, out=out)  # type: ignore[arg-type]
            else:
                if out is None:
                    # 如果 out 参数为 None，则调用 _VF.nuclear_norm 函数并返回结果，指定维度 _dim
                    return _VF.nuclear_norm(input, _dim, keepdim=keepdim)  # type: ignore[arg-type]
                else:
                    # 否则，调用 _VF.nuclear_norm 函数并返回结果，指定维度 _dim 和输出参数 out
                    return _VF.nuclear_norm(input, _dim, keepdim=keepdim, out=out)  # type: ignore[arg-type]
        
        # 如果 p 不是 "fro" 或 "nuc"，抛出运行时错误，提示有效的字符串只能是 'fro' 或 'nuc'
        raise RuntimeError(f"only valid string values are 'fro' and 'nuc', found {p}")

    # 如果 p 不是字符串类型
    else:
        # 如果 _dim 为 None，则设为所有维度的列表
        if _dim is None:
            _dim = list(range(ndim))

        # 如果 out 参数为 None
        if out is None:
            # 如果 dtype 参数为 None，则调用 _VF.norm 函数并返回结果
            if dtype is None:
                return _VF.norm(input, p, _dim, keepdim=keepdim)  # type: ignore[attr-defined]
            else:
                # 否则，调用 _VF.norm 函数并返回结果，带有 dtype 参数
                return _VF.norm(input, p, _dim, keepdim=keepdim, dtype=dtype)  # type: ignore[attr-defined]
        else:
            # 如果 dtype 参数为 None，则调用 _VF.norm 函数并返回结果，带有 out 参数
            if dtype is None:
                return _VF.norm(input, p, _dim, keepdim=keepdim, out=out)  # type: ignore[attr-defined]
            else:
                # 否则，调用 _VF.norm 函数并返回结果，带有 dtype 和 out 参数
                return _VF.norm(input, p, _dim, keepdim=keepdim, dtype=dtype, out=out)  # type: ignore[attr-defined]
def unravel_index(
    indices: Tensor,
    shape: Union[int, Sequence[int], torch.Size],
) -> Tuple[Tensor, ...]:
    r"""Converts a tensor of flat indices into a tuple of coordinate tensors that
    index into an arbitrary tensor of the specified shape.

    Args:
        indices (Tensor): An integer tensor containing indices into the
            flattened version of an arbitrary tensor of shape :attr:`shape`.
            All elements must be in the range ``[0, prod(shape) - 1]``.

        shape (int, sequence of ints, or torch.Size): The shape of the arbitrary
            tensor. All elements must be non-negative.

    Returns:
        tuple of Tensors: Each ``i``-th tensor in the output corresponds with
        dimension ``i`` of :attr:`shape`. Each tensor has the same shape as
        ``indices`` and contains one index into dimension ``i`` for each of the
        flat indices given by ``indices``.

    Example::

        >>> import torch
        >>> torch.unravel_index(torch.tensor(4), (3, 2))
        (tensor(2),
         tensor(0))

        >>> torch.unravel_index(torch.tensor([4, 1]), (3, 2))
        (tensor([2, 0]),
         tensor([0, 1]))

        >>> torch.unravel_index(torch.tensor([0, 1, 2, 3, 4, 5]), (3, 2))
        (tensor([0, 0, 1, 1, 2, 2]),
         tensor([0, 1, 0, 1, 0, 1]))

        >>> torch.unravel_index(torch.tensor([1234, 5678]), (10, 10, 10, 10))
        (tensor([1, 5]),
         tensor([2, 6]),
         tensor([3, 7]),
         tensor([4, 8]))

        >>> torch.unravel_index(torch.tensor([[1234], [5678]]), (10, 10, 10, 10))
        (tensor([[1], [5]]),
         tensor([[2], [6]]),
         tensor([[3], [7]]),
         tensor([[4], [8]]))

        >>> torch.unravel_index(torch.tensor([[1234], [5678]]), (100, 100))
        (tensor([[12], [56]]),
         tensor([[34], [78]]))

    """
    # Check if indices support torch function
    if has_torch_function_unary(indices):
        return handle_torch_function(unravel_index, (indices,), indices, shape=shape)
    
    # Compute the unraveling of indices into coordinate tensors
    res_tensor = _unravel_index(indices, shape)
    
    # Return the coordinate tensors unbound along the last dimension
    return res_tensor.unbind(-1)


def _unravel_index(indices: Tensor, shape: Union[int, Sequence[int]]) -> Tensor:
    # Validate indices data type
    torch._check_type(
        not indices.is_complex()
        and not indices.is_floating_point()
        and not indices.dtype == torch.bool,
        lambda: f"expected 'indices' to be integer dtype, but got {indices.dtype}",
    )
    
    # Validate shape data type
    torch._check_type(
        isinstance(shape, (int, torch.SymInt, Sequence)),
        lambda: f"expected 'shape' to be int or sequence of ints, but got {type(shape)}",
    )
    
    # Convert shape into torch.Size if it's a single integer or torch.SymInt
    if isinstance(shape, (int, torch.SymInt)):
        shape = torch.Size([shape])
    else:
        # Validate each dimension in shape sequence
        for dim in shape:
            torch._check_type(
                isinstance(dim, (int, torch.SymInt)),
                lambda: f"expected 'shape' sequence to only contain ints, but got {type(dim)}",
            )
        shape = torch.Size(shape)

    # Return the validated shape
    return shape
    # 使用 torch._check_value 函数检查 shape 中的每个维度是否都大于等于零
    torch._check_value(
        all(dim >= 0 for dim in shape),
        lambda: f"'shape' cannot have negative values, but got {tuple(shape)}",
    )
    
    # 计算一个系数列表 coefs，用于进行索引计算
    coefs = list(
        reversed(  # 将计算结果反转，以匹配后续索引计算的顺序
            list(
                itertools.accumulate(  # 通过累积函数计算每个维度对应的系数
                    reversed(shape[1:] + torch.Size([1])), func=operator.mul
                )
            )
        )
    )
    
    # 对输入的 indices 进行操作，先进行整数除法再取模，以得到最终的索引结果
    return indices.unsqueeze(-1).floor_divide(
        torch.tensor(coefs, device=indices.device, dtype=torch.int64)
    ) % torch.tensor(shape, device=indices.device, dtype=torch.int64)
def chain_matmul(*matrices, out=None):
    r"""Returns the matrix product of the :math:`N` 2-D tensors. This product is efficiently computed
    using the matrix chain order algorithm which selects the order in which incurs the lowest cost in terms
    of arithmetic operations (`[CLRS]`_). Note that since this is a function to compute the product, :math:`N`
    needs to be greater than or equal to 2; if equal to 2 then a trivial matrix-matrix product is returned.
    If :math:`N` is 1, then this is a no-op - the original matrix is returned as is.

    .. warning::

        :func:`torch.chain_matmul` is deprecated and will be removed in a future PyTorch release.
        Use :func:`torch.linalg.multi_dot` instead, which accepts a list of two or more tensors
        rather than multiple arguments.

    Args:
        matrices (Tensors...): a sequence of 2 or more 2-D tensors whose product is to be determined.
        out (Tensor, optional): the output tensor. Ignored if :attr:`out` = ``None``.

    Returns:
        Tensor: if the :math:`i^{th}` tensor was of dimensions :math:`p_{i} \times p_{i + 1}`, then the product
        would be of dimensions :math:`p_{1} \times p_{N + 1}`.

    Example::

        >>> # xdoctest: +SKIP
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> a = torch.randn(3, 4)
        >>> b = torch.randn(4, 5)
        >>> c = torch.randn(5, 6)
        >>> d = torch.randn(6, 7)
        >>> # will raise a deprecation warning
        >>> torch.chain_matmul(a, b, c, d)
        tensor([[ -2.3375,  -3.9790,  -4.1119,  -6.6577,   9.5609, -11.5095,  -3.2614],
                [ 21.4038,   3.3378,  -8.4982,  -5.2457, -10.2561,  -2.4684,   2.7163],
                [ -0.9647,  -5.8917,  -2.3213,  -5.2284,  12.8615, -12.2816,  -2.5095]])

    .. _`[CLRS]`: https://mitpress.mit.edu/books/introduction-algorithms-third-edition
    """
    # This wrapper exists to support variadic args.
    # 检查是否有 torch 函数的重载，如果有则调用处理函数
    if has_torch_function(matrices):
        return handle_torch_function(chain_matmul, matrices, *matrices)

    # 如果没有提供输出张量，直接调用 Torch 内部的 chain_matmul 函数进行计算
    if out is None:
        return _VF.chain_matmul(matrices)  # type: ignore[attr-defined]
    else:
        # 如果提供了输出张量，则将其传递给 Torch 内部的 chain_matmul 函数进行计算
        return _VF.chain_matmul(matrices, out=out)  # type: ignore[attr-defined]


def _lu_impl(A, pivot=True, get_infos=False, out=None):
    # type: (Tensor, bool, bool, Any) -> Tuple[Tensor, Tensor, Tensor]
    r"""Computes the LU factorization of a matrix or batches of matrices
    :attr:`A`. Returns a tuple containing the LU factorization and
    pivots of :attr:`A`.  Pivoting is done if :attr:`pivot` is set to
    ``True``.
    ```
    # 计算矩阵 A 或批次矩阵的 LU 分解，返回包含 LU 分解和枢轴的元组
    r"""Computes the LU factorization of a matrix or batches of matrices
    :attr:`A`. Returns a tuple containing the LU factorization and
    pivots of :attr:`A`.  Pivoting is done if :attr:`pivot` is set to
    ``True``.
    # .. warning:: 标注警告信息开始，说明以下内容已被废弃，建议使用新的函数进行替代
    # :func:`torch.lu` is deprecated in favor of :func:`torch.linalg.lu_factor`
    # and :func:`torch.linalg.lu_factor_ex`. :func:`torch.lu` will be removed in a
    # future PyTorch release.
    # ``LU, pivots, info = torch.lu(A, compute_pivots)`` should be replaced with
    # .. code:: python
    # 替换建议示例代码段开始，示例如何使用新的函数进行替代
    # LU, pivots = torch.linalg.lu_factor(A, compute_pivots)
    # ``LU, pivots, info = torch.lu(A, compute_pivots, get_infos=True)`` should be replaced with
    # .. code:: python
    # 替换建议示例代码段开始，示例如何使用新的函数进行替代
    # LU, pivots, info = torch.linalg.lu_factor_ex(A, compute_pivots)
    
    # .. note:: 标注注释信息开始，提供额外说明和注意事项
    # * The returned permutation matrix for every matrix in the batch is
    #   represented by a 1-indexed vector of size ``min(A.shape[-2], A.shape[-1])``.
    #   ``pivots[i] == j`` represents that in the ``i``-th step of the algorithm,
    #   the ``i``-th row was permuted with the ``j-1``-th row.
    #   返回的置换矩阵表示每个批次中的每个矩阵，大小为 ``min(A.shape[-2], A.shape[-1])`` 的 1 索引向量。
    #   ``pivots[i] == j`` 表示在算法的第 ``i`` 步中，第 ``i`` 行与第 ``j-1`` 行进行了置换。
    # * LU factorization with :attr:`pivot` = ``False`` is not available
    #   for CPU, and attempting to do so will throw an error. However,
    #   LU factorization with :attr:`pivot` = ``False`` is available for
    #   CUDA.
    #   使用 :attr:`pivot` = ``False`` 的 LU 分解对于 CPU 不可用，尝试这样做会引发错误。然而，对于 CUDA，可以使用 :attr:`pivot` = ``False`` 进行 LU 分解。
    # * This function does not check if the factorization was successful
    #   or not if :attr:`get_infos` is ``True`` since the status of the
    #   factorization is present in the third element of the return tuple.
    #   如果 :attr:`get_infos` 为 ``True``，此函数不会检查分解是否成功，因为分解的状态包含在返回元组的第三个元素中。
    # * In the case of batches of square matrices with size less or equal
    #   to 32 on a CUDA device, the LU factorization is repeated for
    #   singular matrices due to the bug in the MAGMA library
    #   (see magma issue 13).
    #   在 CUDA 设备上，对于大小小于或等于 32 的方阵批次，由于 MAGMA 库中的 bug，会对奇异矩阵重复进行 LU 分解（参见 magma issue 13）。
    # * ``L``, ``U``, and ``P`` can be derived using :func:`torch.lu_unpack`.
    #   可以使用 :func:`torch.lu_unpack` 导出 ``L``, ``U`` 和 ``P``。
    
    # .. warning:: 标注警告信息开始，提供函数的梯度仅在 :attr:`A` 为全秩时才是有限的
    # The gradients of this function will only be finite when :attr:`A` is full rank.
    # This is because the LU decomposition is just differentiable at full rank matrices.
    # Furthermore, if :attr:`A` is close to not being full rank,
    # the gradient will be numerically unstable as it depends on the computation of :math:`L^{-1}` and :math:`U^{-1}`.
    # 该函数的梯度仅在 :attr:`A` 为全秩时才是有限的，因为 LU 分解仅在全秩矩阵上是可微的。
    # 此外，如果 :attr:`A` 接近非满秩，梯度将因为依赖于 :math:`L^{-1}` 和 :math:`U^{-1}` 的计算而数值不稳定。
    
    # Args: 标注参数部分开始，列出函数的输入参数及其说明
    # A (Tensor): the tensor to factor of size :math:`(*, m, n)`
    # pivot (bool, optional): controls whether pivoting is done. Default: ``True``
    # get_infos (bool, optional): if set to ``True``, returns an info IntTensor.
    #                             Default: ``False``
    # out (tuple, optional): optional output tuple. If :attr:`get_infos` is ``True``,
    #                        then the elements in the tuple are Tensor, IntTensor,
    #                        and IntTensor. If :attr:`get_infos` is ``False``, then the
    #                        elements in the tuple are Tensor, IntTensor. Default: ``None``
    """
    Returns:
        (Tensor, IntTensor, IntTensor (optional)): A tuple of tensors containing

            - **factorization** (*Tensor*): the factorization of size :math:`(*, m, n)`

            - **pivots** (*IntTensor*): the pivots of size :math:`(*, \text{min}(m, n))`.
              ``pivots`` stores all the intermediate transpositions of rows.
              The final permutation ``perm`` could be reconstructed by
              applying ``swap(perm[i], perm[pivots[i] - 1])`` for ``i = 0, ..., pivots.size(-1) - 1``,
              where ``perm`` is initially the identity permutation of :math:`m` elements
              (essentially this is what :func:`torch.lu_unpack` is doing).

            - **infos** (*IntTensor*, *optional*): if :attr:`get_infos` is ``True``, this is a tensor of
              size :math:`(*)` where non-zero values indicate whether factorization for the matrix or
              each minibatch has succeeded or failed

    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_LAPACK)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> A = torch.randn(2, 3, 3)
        >>> A_LU, pivots = torch.lu(A)
        >>> A_LU
        tensor([[[ 1.3506,  2.5558, -0.0816],
                 [ 0.1684,  1.1551,  0.1940],
                 [ 0.1193,  0.6189, -0.5497]],

                [[ 0.4526,  1.2526, -0.3285],
                 [-0.7988,  0.7175, -0.9701],
                 [ 0.2634, -0.9255, -0.3459]]])
        >>> pivots
        tensor([[ 3,  3,  3],
                [ 3,  3,  3]], dtype=torch.int32)
        >>> A_LU, pivots, info = torch.lu(A, get_infos=True)
        >>> if info.nonzero().size(0) == 0:
        ...     print('LU factorization succeeded for all samples!')
        LU factorization succeeded for all samples!
    """
    # 如果 get_infos 为 True，则不需要检查错误；否则需要检查错误
    return torch._lu_with_info(A, pivot=pivot, check_errors=(not get_infos))
if TYPE_CHECKING:
    _ListOrSeq = Sequence[Tensor]
else:
    _ListOrSeq = List[Tensor]


# 如果在类型检查模式下，定义_ListOrSeq为Sequence[Tensor]，否则为List[Tensor]
if TYPE_CHECKING:
    _ListOrSeq = Sequence[Tensor]
else:
    _ListOrSeq = List[Tensor]



def _check_list_size(out_len: int, get_infos: bool, out: _ListOrSeq) -> None:
    get_infos_int = 1 if get_infos else 0
    if out_len - get_infos_int != 2:
        raise TypeError(
            f"expected tuple of {2 + int(get_infos)} elements but got {out_len}"
        )
    if not isinstance(out, (tuple, list)):
        raise TypeError(
            f"argument 'out' must be tuple of Tensors, not {type(out).__name__}"
        )


# 检查列表大小的函数，确保与预期的长度相符
def _check_list_size(out_len: int, get_infos: bool, out: _ListOrSeq) -> None:
    # 根据是否有get_infos来决定增加的元素个数
    get_infos_int = 1 if get_infos else 0
    # 如果输出的长度不符合预期的2 + get_infos，则抛出类型错误
    if out_len - get_infos_int != 2:
        raise TypeError(
            f"expected tuple of {2 + int(get_infos)} elements but got {out_len}"
        )
    # 如果输出不是tuple或list类型，则抛出类型错误
    if not isinstance(out, (tuple, list)):
        raise TypeError(
            f"argument 'out' must be tuple of Tensors, not {type(out).__name__}"
        )



def _lu_with_infos(A, pivot=True, get_infos=False, out=None):
    # type: (Tensor, bool, bool, Optional[Tuple[Tensor, Tensor, Tensor]]) -> Tuple[Tensor, Tensor, Tensor]
    if has_torch_function_unary(A):
        return handle_torch_function(
            lu, (A,), A, pivot=pivot, get_infos=get_infos, out=out
        )
    result = _lu_impl(A, pivot, get_infos, out)
    if out is not None:
        _check_list_size(len(out), get_infos, out)
        for i in range(len(out)):
            out[i].resize_as_(result[i]).copy_(result[i])
        return out
    else:
        return result  # A_LU, pivots, infos


# 执行带有信息的LU分解
def _lu_with_infos(A, pivot=True, get_infos=False, out=None):
    # 如果A支持torch函数操作，则使用handle_torch_function处理
    if has_torch_function_unary(A):
        return handle_torch_function(
            lu, (A,), A, pivot=pivot, get_infos=get_infos, out=out
        )
    # 否则调用_lu_impl函数执行LU分解
    result = _lu_impl(A, pivot, get_infos, out)
    # 如果指定了输出out
    if out is not None:
        # 检查输出的长度是否符合预期
        _check_list_size(len(out), get_infos, out)
        # 将结果复制到输出的张量中
        for i in range(len(out)):
            out[i].resize_as_(result[i]).copy_(result[i])
        return out
    else:
        return result  # 返回LU分解的结果：A_LU, pivots, infos



def _lu_no_infos(A, pivot=True, get_infos=False, out=None):
    # type: (Tensor, bool, bool, Optional[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]
    # need to check for torch_function here so that we exit if
    if has_torch_function_unary(A):
        return handle_torch_function(
            lu, (A,), A, pivot=pivot, get_infos=get_infos, out=out
        )
    result = _lu_impl(A, pivot, get_infos, out)
    if out is not None:
        _check_list_size(len(out), get_infos, out)
        for i in range(len(out)):
            out[i].resize_as_(result[i]).copy_(result[i])
        return out
    else:
        return result[0], result[1]  # A_LU, pivots


# 执行不带信息的LU分解
def _lu_no_infos(A, pivot=True, get_infos=False, out=None):
    # 如果A支持torch函数操作，则使用handle_torch_function处理
    if has_torch_function_unary(A):
        return handle_torch_function(
            lu, (A,), A, pivot=pivot, get_infos=get_infos, out=out
        )
    # 否则调用_lu_impl函数执行LU分解
    result = _lu_impl(A, pivot, get_infos, out)
    # 如果指定了输出out
    if out is not None:
        # 检查输出的长度是否符合预期
        _check_list_size(len(out), get_infos, out)
        # 将结果复制到输出的张量中
        for i in range(len(out)):
            out[i].resize_as_(result[i]).copy_(result[i])
        return out
    else:
        return result[0], result[1]  # 返回LU分解的结果：A_LU, pivots



# The return type of lu depends on `get_infos`, so in order to resolve the output type
# of lu in TorchScript we need to statically know the value of `get_infos`
lu = boolean_dispatch(
    arg_name="get_infos",
    arg_index=2,
    default=False,
    if_true=_lu_with_infos,
    if_false=_lu_no_infos,
    module_name=__name__,
    func_name="lu",
)
lu.__doc__ = _lu_impl.__doc__


# 根据get_infos参数动态分发LU分解函数，并设置文档说明
lu = boolean_dispatch(
    arg_name="get_infos",
    arg_index=2,
    default=False,
    if_true=_lu_with_infos,
    if_false=_lu_no_infos,
    module_name=__name__,
    func_name="lu",
)
# 将lu函数的文档说明设置为_lu_impl函数的文档说明
lu.__doc__ = _lu_impl.__doc__



def align_tensors(*tensors):
    raise RuntimeError("`align_tensors` not yet implemented.")


# 抛出运行时错误，表示函数align_tensors尚未实现
def align_tensors(*tensors):
    raise RuntimeError("`align_tensors` not yet implemented.")
```