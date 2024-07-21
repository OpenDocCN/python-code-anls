# `.\pytorch\torch\signal\windows\windows.py`

```
# mypy: allow-untyped-defs
# 引入类型定义模块
from typing import Optional, Iterable

# 引入 PyTorch 库
import torch
# 引入数学库中的平方根函数
from math import sqrt

# 从 PyTorch 中引入 Tensor 类型
from torch import Tensor
# 从内部模块中引入函数
from torch._torch_docs import factory_common_args, parse_kwargs, merge_dicts

# 指定导出的函数名列表
__all__ = [
    'bartlett',      # 巴特利特窗函数
    'blackman',      # 布莱克曼窗函数
    'cosine',        # 余弦窗函数
    'exponential',   # 指数窗函数
    'gaussian',      # 高斯窗函数
    'general_cosine',# 一般余弦窗函数
    'general_hamming',# 一般汉明窗函数
    'hamming',       # 汉明窗函数
    'hann',          # 汉宁窗函数
    'kaiser',        # 凯泽窗函数
    'nuttall',       # 纽特尔窗函数
]

# 定义窗函数的共同参数
window_common_args = merge_dicts(
    parse_kwargs(
        """
    M (int): 窗口的长度。
        也就是返回窗口的数据点数目。
    sym (bool, optional): 如果为 `False`，返回适用于频谱分析的周期性窗口。
        如果为 `True`，返回适用于滤波器设计的对称窗口。默认为 `True`。
"""
    ),
    factory_common_args,
    {
        "normalization": "窗口被归一化为1（最大值为1）。但是如果 :attr:`M` 是偶数且 :attr:`sym` 是 `True`，那么1不会出现。",
    }
)


def _add_docstr(*args):
    r"""为给定的装饰函数添加文档字符串。

    在文档字符串需要字符串插值（例如使用 str.format()）时特别有用。
    注意：如果文档字符串不需要字符串插值，应当使用传统的文档字符串写法，不要使用此函数。

    Args:
        args (str): 插入到文档字符串中的字符串。
    """

    def decorator(o):
        o.__doc__ = "".join(args)
        return o

    return decorator


def _window_function_checks(function_name: str, M: int, dtype: torch.dtype, layout: torch.layout) -> None:
    r"""为所有定义的窗口函数执行常见检查。
    在计算任何窗口之前应调用此函数。

    Args:
        function_name (str): 窗口函数的名称。
        M (int): 窗口的长度。
        dtype (:class:`torch.dtype`): 返回张量的期望数据类型。
        layout (:class:`torch.layout`): 返回张量的期望布局。
    """
    if M < 0:
        raise ValueError(f'{function_name} 需要非负的窗口长度，得到 M={M}')
    if layout is not torch.strided:
        raise ValueError(f'{function_name} 仅支持步幅张量，得到：{layout}')
    if dtype not in [torch.float32, torch.float64]:
        raise ValueError(f'{function_name} 期望 float32 或 float64 数据类型，得到：{dtype}')


@_add_docstr(
    r"""
计算具有指数波形的窗口。
也称为泊松窗口。

指数窗口定义如下：

.. math::
    w_n = \exp{\left(-\frac{|n - c|}{\tau}\right)}

其中 `c` 是窗口的中心。
    """,
    r"""

{normalization}

Args:
    {M}

Keyword args:
    center (float, optional): 窗口的中心位置。
        默认为 `M / 2`，如果 `sym` 是 `False`，否则为 `(M - 1) / 2`。

"""
)
    tau (float, optional): the decay value.
        Tau is generally associated with a percentage, that means, that the value should
        vary within the interval (0, 100]. If tau is 100, it is considered the uniform window.
        Default: 1.0.
    {sym}
        Symbol placeholder used for parameter substitution.
    {dtype}
        Placeholder for specifying the data type of the tensor.
    {layout}
        Placeholder for specifying the memory layout of the tensor (e.g., 'strided').
    {device}
        Placeholder for specifying the device (e.g., 'cpu', 'cuda') on which the tensor resides.
    {requires_grad}
        Placeholder indicating whether gradients need to be computed for this tensor.
@_add_docstr(
    r"""
Computes a window with a simple cosine waveform, following the same implementation as SciPy.
This window is also known as the sine window.

The cosine window is defined as follows:

.. math::
    w_n = \sin\left(\frac{\pi (n + 0.5)}{M}\right)

This formula differs from the typical cosine window formula by incorporating a 0.5 term in the numerator,
which shifts the sample positions. This adjustment results in a window that starts and ends with non-zero values.

""",
    r"""
Computes a window using an exponential decay function.

Args:
    M (int): The number of points in the output window. It determines the length of the window.
    center (float, optional): The position of the peak of the window, if None it is at (M - 1) / 2.
    tau (float, optional): The decay factor. A smaller tau results in faster decay.
    sym (bool, optional): Whether to generate a symmetric window. Defaults to True.
    dtype (torch.dtype, optional): The desired data type of the output tensor.
    layout (torch.layout, optional): The desired layout of the output tensor.
    device (torch.device, optional): The desired device of the output tensor.
    requires_grad (bool, optional): Whether the output tensor requires gradient computation.

Raises:
    ValueError: If tau is non-positive or if sym is True and center is not None.
    RuntimeError: If M is 0.

Returns:
    Tensor: A 1-D tensor representing the exponential window.

Examples::

    >>> # Generates a symmetric exponential window of size 10 and with a decay value of 1.0.
    >>> # The center will be at (M - 1) / 2, where M is 10.
    >>> torch.signal.windows.exponential(10)
    tensor([0.0111, 0.0302, 0.0821, 0.2231, 0.6065, 0.6065, 0.2231, 0.0821, 0.0302, 0.0111])

    >>> # Generates a periodic exponential window and decay factor equal to .5
    >>> torch.signal.windows.exponential(10, sym=False, tau=.5)
    tensor([4.5400e-05, 3.3546e-04, 2.4788e-03, 1.8316e-02, 1.3534e-01, 1.0000e+00, 1.3534e-01, 1.8316e-02, 2.4788e-03, 3.3546e-04])
""".format(
        **window_common_args
    ),
)
def exponential(
        M: int,
        *,
        center: Optional[float] = None,
        tau: float = 1.0,
        sym: bool = True,
        dtype: Optional[torch.dtype] = None,
        layout: torch.layout = torch.strided,
        device: Optional[torch.device] = None,
        requires_grad: bool = False
) -> Tensor:
    if dtype is None:
        dtype = torch.get_default_dtype()

    _window_function_checks('exponential', M, dtype, layout)

    if tau <= 0:
        raise ValueError(f'Tau must be positive, got: {tau} instead.')

    if sym and center is not None:
        raise ValueError('Center must be None for symmetric windows')

    if M == 0:
        return torch.empty((0,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    if center is None:
        center = (M if not sym and M > 1 else M - 1) / 2.0

    constant = 1 / tau

    k = torch.linspace(start=-center * constant,
                       end=(-center + (M - 1)) * constant,
                       steps=M,
                       dtype=dtype,
                       layout=layout,
                       device=device,
                       requires_grad=requires_grad)

    return torch.exp(-torch.abs(k))
# 定义一个计算余弦窗口的函数
def cosine(
        M: int,
        *,
        sym: bool = True,  # 是否对称，默认为True
        dtype: Optional[torch.dtype] = None,  # 数据类型，默认为None
        layout: torch.layout = torch.strided,  # 张量布局，默认为torch.strided
        device: Optional[torch.device] = None,  # 设备类型，默认为None
        requires_grad: bool = False  # 是否需要梯度，默认为False
) -> Tensor:  # 函数返回值为Tensor类型

    if dtype is None:
        dtype = torch.get_default_dtype()  # 如果未指定数据类型，则使用默认数据类型

    _window_function_checks('cosine', M, dtype, layout)  # 调用辅助函数检查窗口函数参数

    if M == 0:
        # 如果M为0，返回一个空张量
        return torch.empty((0,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    start = 0.5  # 起始值为0.5
    constant = torch.pi / (M + 1 if not sym and M > 1 else M)  # 计算常数项，根据sym参数决定是否对称

    k = torch.linspace(start=start * constant,  # 在起始和结束值之间生成均匀间隔的数值
                       end=(start + (M - 1)) * constant,
                       steps=M,
                       dtype=dtype,
                       layout=layout,
                       device=device,
                       requires_grad=requires_grad)

    return torch.sin(k)  # 返回生成的余弦窗口


@_add_docstr(
    r"""
    Computes a window with a gaussian waveform.

    The gaussian window is defined as follows:

    .. math::
        w_n = \exp{\left(-\left(\frac{n}{2\sigma}\right)^2\right)}
    """,
    r"""
    {normalization}

    Args:
        {M}

    Keyword args:
        std (float, optional): the standard deviation of the gaussian. It controls how narrow or wide the window is.
            Default: 1.0.
        {sym}
        {dtype}
        {layout}
        {device}
        {requires_grad}

    Examples::

        >>> # Generates a symmetric gaussian window with a standard deviation of 1.0.
        >>> torch.signal.windows.gaussian(10)
        tensor([4.0065e-05, 2.1875e-03, 4.3937e-02, 3.2465e-01, 8.8250e-01, 8.8250e-01, 3.2465e-01, 4.3937e-02, 2.1875e-03, 4.0065e-05])

        >>> # Generates a periodic gaussian window and standard deviation equal to 0.9.
        >>> torch.signal.windows.gaussian(10, sym=False,std=0.9)
        tensor([1.9858e-07, 5.1365e-05, 3.8659e-03, 8.4658e-02, 5.3941e-01, 1.0000e+00, 5.3941e-01, 8.4658e-02, 3.8659e-03, 5.1365e-05])
    """.format(
        **window_common_args,
    ),
)
# 定义一个计算高斯窗口的函数，带有详细的文档字符串
def gaussian(
        M: int,
        *,
        std: float = 1.0,  # 高斯窗口的标准差，默认为1.0
        sym: bool = True,  # 是否对称，默认为True
        dtype: Optional[torch.dtype] = None,  # 数据类型，默认为None
        layout: torch.layout = torch.strided,  # 张量布局，默认为torch.strided
        device: Optional[torch.device] = None,  # 设备类型，默认为None
        requires_grad: bool = False  # 是否需要梯度，默认为False
) -> Tensor:  # 函数返回值为Tensor类型

    if dtype is None:
        dtype = torch.get_default_dtype()  # 如果未指定数据类型，则使用默认数据类型

    _window_function_checks('gaussian', M, dtype, layout)  # 调用辅助函数检查窗口函数参数

    if std <= 0:
        raise ValueError(f'Standard deviation must be positive, got: {std} instead.')  # 如果标准差小于等于0，抛出值错误异常

    if M == 0:
        # 如果M为0，返回一个空张量
        return torch.empty((0,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    start = -(M if not sym and M > 1 else M - 1) / 2.0  # 计算起始值，根据sym参数决定是否对称

    constant = 1 / (std * sqrt(2))  # 计算常数项，根据标准差控制高斯窗口的宽度

    k = torch.linspace(start=start * constant,  # 在起始和结束值之间生成均匀间隔的数值
                       end=(start + (M - 1)) * constant,
                       steps=M,
                       dtype=dtype,
                       layout=layout,
                       device=device,
                       requires_grad=requires_grad)
    # 计算输入张量 k 的每个元素的指数函数，即 e^(-k^2)
    return torch.exp(-k ** 2)
# 添加文档字符串到函数 kaiser，描述其计算 Kaiser 窗口的功能和公式
@_add_docstr(
    r"""
Computes the Kaiser window.

The Kaiser window is defined as follows:

.. math::
    w_n = I_0 \left( \beta \sqrt{1 - \left( {\frac{n - N/2}{N/2}} \right) ^2 } \right) / I_0( \beta )

where ``I_0`` is the zeroth order modified Bessel function of the first kind (see :func:`torch.special.i0`), and
``N = M - 1 if sym else M``.
    """,
    r"""

{normalization}

Args:
    {M}  # M 是窗口长度

Keyword args:
    beta (float, optional): shape parameter for the window. Must be non-negative. Default: 12.0  # beta 是窗口的形状参数，必须是非负数，默认为12.0
    {sym}  # sym 是一个布尔值，指示是否生成对称的窗口
    {dtype}  # dtype 是返回张量的数据类型
    {layout}  # layout 是返回张量的布局类型
    {device}  # device 是返回张量所在的设备
    {requires_grad}  # requires_grad 是一个布尔值，指示张量是否需要梯度

Examples::

    >>> # 生成标准差为1.0的对称高斯窗口。
    >>> torch.signal.windows.kaiser(5)
    tensor([4.0065e-05, 2.1875e-03, 4.3937e-02, 3.2465e-01, 8.8250e-01, 8.8250e-01, 3.2465e-01, 4.3937e-02, 2.1875e-03, 4.0065e-05])
    >>> # 生成周期高斯窗口，标准差为0.9。
    >>> torch.signal.windows.kaiser(5, sym=False, std=0.9)
    tensor([1.9858e-07, 5.1365e-05, 3.8659e-03, 8.4658e-02, 5.3941e-01, 1.0000e+00, 5.3941e-01, 8.4658e-02, 3.8659e-03, 5.1365e-05])
""".format(
        **window_common_args,
    ),
)
def kaiser(
        M: int,
        *,
        beta: float = 12.0,
        sym: bool = True,
        dtype: Optional[torch.dtype] = None,
        layout: torch.layout = torch.strided,
        device: Optional[torch.device] = None,
        requires_grad: bool = False
) -> Tensor:
    # 如果 dtype 未指定，使用默认数据类型
    if dtype is None:
        dtype = torch.get_default_dtype()

    # 检查窗口函数的参数和布局
    _window_function_checks('kaiser', M, dtype, layout)

    # 如果 beta 小于0，抛出 ValueError 异常
    if beta < 0:
        raise ValueError(f'beta must be non-negative, got: {beta} instead.')

    # 如果 M 等于0，返回一个空张量
    if M == 0:
        return torch.empty((0,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    # 如果 M 等于1，返回一个全为1的张量
    if M == 1:
        return torch.ones((1,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    # 将 beta 转换为指定的数据类型，避免产生 NaN
    beta = torch.tensor(beta, dtype=dtype, device=device)

    # 计算 Kaiser 窗口的起始和结束值，确保不会超出 beta 的范围
    start = -beta
    constant = 2.0 * beta / (M if not sym else M - 1)
    end = torch.minimum(beta, start + (M - 1) * constant)

    # 生成从 start 到 end 等间隔的 M 个数值
    k = torch.linspace(start=start,
                       end=end,
                       steps=M,
                       dtype=dtype,
                       layout=layout,
                       device=device,
                       requires_grad=requires_grad)

    # 计算 Kaiser 窗口的值并返回
    return torch.i0(torch.sqrt(beta * beta - torch.pow(k, 2))) / torch.i0(beta)


# 添加文档字符串到函数 Hamming，描述其计算 Hamming 窗口的功能和公式
@_add_docstr(
    r"""
Computes the Hamming window.

The Hamming window is defined as follows:

.. math::
    w_n = \alpha - \beta\ \cos \left( \frac{2 \pi n}{M - 1} \right)
    """,
    r"""

{normalization}

Arguments:
    {M}  # M 是窗口长度

Keyword args:
    {sym}  # sym 是一个布尔值，指示是否生成对称的窗口
    alpha (float, optional): The coefficient :math:`\alpha` in the equation above.  # alpha 是上述公式中的系数 α
    beta (float, optional): The coefficient :math:`\beta` in the equation above.  # beta 是上述公式中的系数 β
    {dtype}  # dtype 是返回张量的数据类型
    {layout}  # layout 是返回张量的布局类型
    {device}  # device 是返回张量所在的设备
    {requires_grad}  # requires_grad 是一个布尔值，指示张量是否需要梯度

Examples::
    # 生成一个对称的汉明窗口。
    torch.signal.windows.hamming(10)
    # 返回一个张量，表示长度为 10 的对称汉明窗口的值，依次为 [0.0800, 0.1876, 0.4601, 0.7700, 0.9723, 0.9723, 0.7700, 0.4601, 0.1876, 0.0800]
    
    # 生成一个周期性的汉明窗口。
    torch.signal.windows.hamming(10, sym=False)
    # 返回一个张量，表示长度为 10 的周期性汉明窗口的值，依次为 [0.0800, 0.1679, 0.3979, 0.6821, 0.9121, 1.0000, 0.9121, 0.6821, 0.3979, 0.1679]
@_add_docstr(
    r"""
Computes the Hann window.

The Hann window is defined as follows:

.. math::
    w_n = \frac{1}{2}\ \left[1 - \cos \left( \frac{2 \pi n}{M - 1} \right)\right] =
    \sin^2 \left( \frac{\pi n}{M - 1} \right)
    """,
    r"""
对于给定的 M 值，计算汉宁窗口。

{normalization}

Arguments:
    {M}  - 窗口长度 M

Keyword args:
    {sym}  - 是否对称，默认为 True
    {dtype}  - 数据类型，默认为 None
    {layout}  - 张量布局，默认为 torch.strided
    {device}  - 设备类型，默认为 None
    {requires_grad}  - 是否需要梯度，默认为 False

Examples::

    >>> # 生成对称的汉宁窗口。
    >>> torch.signal.windows.hann(10)
    tensor([0.0000, 0.1170, 0.4132, 0.7500, 0.9698, 0.9698, 0.7500, 0.4132, 0.1170, 0.0000])

    >>> # 生成周期的汉宁窗口。
    >>> torch.signal.windows.hann(10, sym=False)
    tensor([0.0000, 0.0955, 0.3455, 0.6545, 0.9045, 1.0000, 0.9045, 0.6545, 0.3455, 0.0955])
""".format(
        **window_common_args
    ),
)
def hann(M: int,
         *,
         sym: bool = True,
         dtype: Optional[torch.dtype] = None,
         layout: torch.layout = torch.strided,
         device: Optional[torch.device] = None,
         requires_grad: bool = False) -> Tensor:
    """
    计算汉宁窗口函数。

    Parameters:
        M (int): 窗口长度
        sym (bool, optional): 是否对称，默认为 True
        dtype (torch.dtype, optional): 数据类型，默认为 None
        layout (torch.layout): 张量布局，默认为 torch.strided
        device (torch.device, optional): 设备类型，默认为 None
        requires_grad (bool): 是否需要梯度，默认为 False

    Returns:
        Tensor: 汉宁窗口的张量表示

    Raises:
        None

    Notes:
        This function computes the Hann window function for a given length M.
        The Hann window is defined as a raised cosine window, which reduces spectral leakage.

    Examples:
        >>> # Generates a symmetric Hann window.
        >>> torch.signal.windows.hann(10)
        tensor([0.0000, 0.1170, 0.4132, 0.7500, 0.9698, 0.9698, 0.7500, 0.4132, 0.1170, 0.0000])

        >>> # Generates a periodic Hann window.
        >>> torch.signal.windows.hann(10, sym=False)
        tensor([0.0000, 0.0955, 0.3455, 0.6545, 0.9045, 1.0000, 0.9045, 0.6545, 0.3455, 0.0955])
    """
    return general_hamming(M,
                           alpha=0.5,
                           sym=sym,
                           dtype=dtype,
                           layout=layout,
                           device=device,
                           requires_grad=requires_grad)


@_add_docstr(
    r"""
Computes the Blackman window.

The Blackman window is defined as follows:

.. math::
    w_n = 0.42 - 0.5 \cos \left( \frac{2 \pi n}{M - 1} \right) + 0.08 \cos \left( \frac{4 \pi n}{M - 1} \right)
    """,
    r"""
对于给定的 M 值，计算布莱克曼窗口。

{normalization}

Arguments:
    {M}  - 窗口长度 M

Keyword args:
    {sym}  - 是否对称，默认为 True
    {dtype}  - 数据类型，默认为 None
    {layout}  - 张量布局，默认为 torch.strided
    {device}  - 设备类型，默认为 None
    {requires_grad}  - 是否需要梯度，默认为 False

Examples::

    >>> # 生成对称的布莱克曼窗口。
    >>> torch.signal.windows.blackman(5)
    tensor([-1.4901e-08,  3.4000e-01,  1.0000e+00,  3.4000e-01, -1.4901e-08])

    >>> # 生成周期的布莱克曼窗口。
    >>> torch.signal.windows.blackman(5, sym=False)
    tensor([-1.4901e-08,  2.0077e-01,  8.4923e-01,  8.4923e-01,  2.0077e-01])
""".format(
        **window_common_args
    ),
)
def blackman(M: int,
             *,
             sym: bool = True,
             dtype: Optional[torch.dtype] = None,
             layout: torch.layout = torch.strided,
             device: Optional[torch.device] = None,
             requires_grad: bool = False) -> Tensor:
    """
    计算布莱克曼窗口函数。

    Parameters:
        M (int): 窗口长度
        sym (bool, optional): 是否对称，默认为 True
        dtype (torch.dtype, optional): 数据类型，默认为 None
        layout (torch.layout): 张量布局，默认为 torch.strided
        device (torch.device, optional): 设备类型，默认为 None
        requires_grad (bool): 是否需要梯度，默认为 False

    Returns:
        Tensor: 布莱克曼窗口的张量表示

    Raises:
        None

    Notes:
        This function computes the Blackman window function for a given length M.
        The Blackman window is designed to have a better side-lobe attenuation than the Hann window.

    Examples:
        >>> # Generates a symmetric Blackman window.
        >>> torch.signal.windows.blackman(5)
        tensor([-1.4901e-08,  3.4000e-01,  1.0000e+00,  3.4000e-01, -1.4901e-08])

        >>> # Generates a periodic Blackman window.
        >>> torch.signal.windows.blackman(5, sym=False)
        tensor([-1.4901e-08,  2.0077e-01,  8.4923e-01,  8.4923e-01,  2.0077e-01])
    """
    if dtype is None:
        dtype = torch.get_default_dtype()

    _window_function_checks('blackman', M, dtype, layout)
    # 调用 general_cosine 函数并返回其结果
    return general_cosine(M, a=[0.42, 0.5, 0.08], sym=sym, dtype=dtype, layout=layout, device=device,
                          requires_grad=requires_grad)
# 添加文档字符串和函数装饰器，定义了计算Bartlett窗口的函数
@_add_docstr(
    r"""
Computes the Bartlett window.

The Bartlett window is defined as follows:

.. math::
    w_n = 1 - \left| \frac{2n}{M - 1} - 1 \right| = \begin{cases}
        \frac{2n}{M - 1} & \text{if } 0 \leq n \leq \frac{M - 1}{2} \\
        2 - \frac{2n}{M - 1} & \text{if } \frac{M - 1}{2} < n < M \\ \end{cases}
    """,
    r"""
    
{normalization}

Arguments:
    {M}                # 窗口长度M，整数类型

Keyword args:
    {sym}              # 是否生成对称窗口，布尔类型
    {dtype}            # 数据类型，torch.dtype或None
    {layout}           # 张量布局，torch.layout
    {device}           # 设备类型，torch.device或None
    {requires_grad}    # 是否需要梯度计算，布尔类型

Examples::

    >>> # 生成对称的Bartlett窗口。
    >>> torch.signal.windows.bartlett(10)
    tensor([0.0000, 0.2222, 0.4444, 0.6667, 0.8889, 0.8889, 0.6667, 0.4444, 0.2222, 0.0000])

    >>> # 生成周期性的Bartlett窗口。
    >>> torch.signal.windows.bartlett(10, sym=False)
    tensor([0.0000, 0.2000, 0.4000, 0.6000, 0.8000, 1.0000, 0.8000, 0.6000, 0.4000, 0.2000])
""".format(
        **window_common_args  # 插入通用参数字典中的值
    ),
)
def bartlett(M: int,
             *,
             sym: bool = True,            # sym默认为True，表示对称窗口
             dtype: Optional[torch.dtype] = None,    # dtype默认为None，表示使用默认数据类型
             layout: torch.layout = torch.strided,   # layout默认为strided，表示张量布局
             device: Optional[torch.device] = None,  # device默认为None，表示使用默认设备
             requires_grad: bool = False) -> Tensor:  # requires_grad默认为False，表示不需要梯度计算
    if dtype is None:
        dtype = torch.get_default_dtype()  # 如果dtype为None，则使用默认数据类型

    _window_function_checks('bartlett', M, dtype, layout)  # 检查窗口函数的参数

    if M == 0:
        return torch.empty((0,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)  # 如果M为0，则返回一个空张量

    if M == 1:
        return torch.ones((1,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)  # 如果M为1，则返回一个全为1的张量

    start = -1
    constant = 2 / (M if not sym else M - 1)  # 计算常数系数，根据是否对称决定使用M或者M-1

    k = torch.linspace(start=start,
                       end=start + (M - 1) * constant,
                       steps=M,
                       dtype=dtype,
                       layout=layout,
                       device=device,
                       requires_grad=requires_grad)  # 使用torch.linspace生成等间隔序列k

    return 1 - torch.abs(k)  # 返回计算得到的Bartlett窗口序列


# 添加文档字符串和函数装饰器，定义了计算通用cosine窗口的函数
@_add_docstr(
    r"""
Computes the general cosine window.

The general cosine window is defined as follows:

.. math::
    w_n = \sum^{M-1}_{i=0} (-1)^i a_i \cos{ \left( \frac{2 \pi i n}{M - 1}\right)}
    """,
    r"""
    
{normalization}

Arguments:
    {M}                # 窗口长度M，整数类型

Keyword args:
    a (Iterable): the coefficients associated to each of the cosine functions.  # a为每个cosine函数关联的系数，迭代器类型
    {sym}              # 是否生成对称窗口，布尔类型
    {dtype}            # 数据类型，torch.dtype或None
    {layout}           # 张量布局，torch.layout
    {device}           # 设备类型，torch.device或None
    {requires_grad}    # 是否需要梯度计算，布尔类型

Examples::

    >>> # 生成对称的通用cosine窗口，使用3个系数。
    >>> torch.signal.windows.general_cosine(10, a=[0.46, 0.23, 0.31], sym=True)
    tensor([0.5400, 0.3376, 0.1288, 0.4200, 0.9136, 0.9136, 0.4200, 0.1288, 0.3376, 0.5400])

    >>> # 生成周期性的通用cosine窗口，使用2个系数。
    >>> torch.signal.windows.general_cosine(10, a=[0.5, 1 - 0.5], sym=False)
    tensor([0.0000, 0.0955, 0.3455, 0.6545, 0.9045, 1.0000, 0.9045, 0.6545, 0.3455, 0.0955])
""".format(
        **window_common_args  # 插入通用参数字典中的值
    ),
)
# 定义一个通用的余弦窗口函数，用于生成余弦窗口系数。
def general_cosine(M, *,
                   a: Iterable,  # 窗口系数，必须是可迭代对象
                   sym: bool = True,  # 是否对称，默认为True
                   dtype: Optional[torch.dtype] = None,  # 数据类型，默认为None
                   layout: torch.layout = torch.strided,  # 张量布局，默认为strided
                   device: Optional[torch.device] = None,  # 设备类型，默认为None
                   requires_grad: bool = False) -> Tensor:  # 是否需要梯度，默认为False
    if dtype is None:
        dtype = torch.get_default_dtype()  # 如果数据类型为None，则使用默认数据类型

    _window_function_checks('general_cosine', M, dtype, layout)  # 检查窗口函数的参数合法性

    if M == 0:
        return torch.empty((0,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)
        # 若 M 等于 0，则返回一个空的张量

    if M == 1:
        return torch.ones((1,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)
        # 若 M 等于 1，则返回一个全为1的张量

    if not isinstance(a, Iterable):
        raise TypeError("Coefficients must be a list/tuple")  # 如果系数不是可迭代对象，则抛出类型错误异常

    if not a:
        raise ValueError("Coefficients cannot be empty")  # 如果系数为空，则抛出值错误异常

    constant = 2 * torch.pi / (M if not sym else M - 1)  # 计算常数系数，根据是否对称选择不同的公式

    k = torch.linspace(start=0,
                       end=(M - 1) * constant,
                       steps=M,
                       dtype=dtype,
                       layout=layout,
                       device=device,
                       requires_grad=requires_grad)
    # 生成从0到(M-1)*constant的等间距的张量k

    a_i = torch.tensor([(-1) ** i * w for i, w in enumerate(a)], device=device, dtype=dtype, requires_grad=requires_grad)
    # 根据系数a生成张量a_i，其中包含了(-1)^i * w的计算

    i = torch.arange(a_i.shape[0], dtype=a_i.dtype, device=a_i.device, requires_grad=a_i.requires_grad)
    # 生成一个范围从0到a_i.shape[0]-1的整数张量i，用于后续计算

    return (a_i.unsqueeze(-1) * torch.cos(i.unsqueeze(-1) * k)).sum(0)
    # 返回根据系数a_i和张量k计算出的余弦窗口函数的张量结果


``````
@_add_docstr(
    r"""
Computes the general Hamming window.

The general Hamming window is defined as follows:

.. math::
    w_n = \alpha - (1 - \alpha) \cos{ \left( \frac{2 \pi n}{M-1} \right)}
    """,
    r"""
    
{normalization}
    
Arguments:
    {M}

Keyword args:
    alpha (float, optional): the window coefficient. Default: 0.54.
    {sym}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Examples::

    >>> # Generates a symmetric Hamming window with the general Hamming window.
    >>> torch.signal.windows.general_hamming(10, sym=True)
    tensor([0.0800, 0.1876, 0.4601, 0.7700, 0.9723, 0.9723, 0.7700, 0.4601, 0.1876, 0.0800])

    >>> # Generates a periodic Hann window with the general Hamming window.
    >>> torch.signal.windows.general_hamming(10, alpha=0.5, sym=False)
    tensor([0.0000, 0.0955, 0.3455, 0.6545, 0.9045, 1.0000, 0.9045, 0.6545, 0.3455, 0.0955])
""".format(
        **window_common_args
    ),
)
def general_hamming(M,
                    *,
                    alpha: float = 0.54,
                    sym: bool = True,
                    dtype: Optional[torch.dtype] = None,
                    layout: torch.layout = torch.strided,
                    device: Optional[torch.device] = None,
                    requires_grad: bool = False) -> Tensor:


注释：
    # 使用 general_cosine 函数计算结果并返回
    return general_cosine(M,
                          a=[alpha, 1. - alpha],  # 设置函数的参数 a，用于计算加权平均值
                          sym=sym,                # 设置函数的参数 sym，指定是否对称
                          dtype=dtype,            # 设置函数的参数 dtype，指定数据类型
                          layout=layout,          # 设置函数的参数 layout，指定张量的布局
                          device=device,          # 设置函数的参数 device，指定计算设备
                          requires_grad=requires_grad)  # 设置函数的参数 requires_grad，指定是否需要梯度计算
@_add_docstr(
    r"""
Computes the minimum 4-term Blackman-Harris window according to Nuttall.

.. math::
    w_n = 1 - 0.36358 \cos{(z_n)} + 0.48917 \cos{(2z_n)} - 0.13659 \cos{(3z_n)} + 0.01064 \cos{(4z_n)}

where ``z_n = 2 \pi n / M``.

{normalization}

Arguments:
    M : int
        Number of points in the output window. It defines the length of the window.
        
Keyword args:
    sym : bool, optional
        Whether the window is symmetric (True) or periodic (False). Defaults to True.
    dtype : torch.dtype, optional
        The desired data type of the tensor. If None, uses the default dtype.
    layout : torch.layout, optional
        The desired layout of the tensor. Defaults to torch.strided.
    device : torch.device, optional
        The desired device of the tensor. If None, uses the current device.
    requires_grad : bool, optional
        If autograd should record operations on the returned tensor. Defaults to False.

References::

    - A. Nuttall, "Some windows with very good sidelobe behavior,"
      IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 29, no. 1, pp. 84-91,
      Feb 1981. https://doi.org/10.1109/TASSP.1981.1163506

    - Heinzel G. et al., "Spectrum and spectral density estimation by the Discrete Fourier transform (DFT),
      including a comprehensive list of window functions and some new flat-top windows",
      February 15, 2002 https://holometer.fnal.gov/GH_FFT.pdf

Examples::

    >>> # Generates a symmetric Nutall window.
    >>> torch.signal.windows.general_hamming(5, sym=True)
    tensor([3.6280e-04, 2.2698e-01, 1.0000e+00, 2.2698e-01, 3.6280e-04])

    >>> # Generates a periodic Nuttall window.
    >>> torch.signal.windows.general_hamming(5, sym=False)
    tensor([3.6280e-04, 1.1052e-01, 7.9826e-01, 7.9826e-01, 1.1052e-01])
""".format(
        **window_common_args
    ),
)
def nuttall(
        M: int,
        *,
        sym: bool = True,
        dtype: Optional[torch.dtype] = None,
        layout: torch.layout = torch.strided,
        device: Optional[torch.device] = None,
        requires_grad: bool = False
) -> Tensor:
    """
    Computes the Nuttall window using the general_cosine function.

    Parameters:
    - M : int
        Number of points in the output window. It defines the length of the window.

    Keyword arguments:
    - sym : bool, optional
        Whether the window is symmetric (True) or periodic (False). Defaults to True.
    - dtype : torch.dtype, optional
        The desired data type of the tensor. If None, uses the default dtype.
    - layout : torch.layout, optional
        The desired layout of the tensor. Defaults to torch.strided.
    - device : torch.device, optional
        The desired device of the tensor. If None, uses the current device.
    - requires_grad : bool, optional
        If autograd should record operations on the returned tensor. Defaults to False.

    Returns:
    - Tensor
        A tensor representing the computed Nuttall window.

    References:
    - A. Nuttall, "Some windows with very good sidelobe behavior,"
      IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 29, no. 1, pp. 84-91,
      Feb 1981. https://doi.org/10.1109/TASSP.1981.1163506
    - Heinzel G. et al., "Spectrum and spectral density estimation by the Discrete Fourier transform (DFT),
      including a comprehensive list of window functions and some new flat-top windows",
      February 15, 2002 https://holometer.fnal.gov/GH_FFT.pdf
    """
    return general_cosine(M,
                          a=[0.3635819, 0.4891775, 0.1365995, 0.0106411],
                          sym=sym,
                          dtype=dtype,
                          layout=layout,
                          device=device,
                          requires_grad=requires_grad)
```