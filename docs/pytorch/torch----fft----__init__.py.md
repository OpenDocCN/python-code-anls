# `.\pytorch\torch\fft\__init__.py`

```
# 导入 sys 模块，用于系统相关操作
import sys

# 导入 torch 库
import torch
# 从 torch._C 模块导入 _add_docstr 和 _fft 对象，这些对象在类型检查时会被忽略
from torch._C import _add_docstr, _fft  # type: ignore[attr-defined]
# 从 torch._torch_docs 模块导入 factory_common_args 和 common_args
from torch._torch_docs import factory_common_args, common_args

# 定义 __all__ 列表，包含公开的函数和类名称
__all__ = ['fft', 'ifft', 'fft2', 'ifft2', 'fftn', 'ifftn',
           'rfft', 'irfft', 'rfft2', 'irfft2', 'rfftn', 'irfftn',
           'hfft', 'ihfft', 'fftfreq', 'rfftfreq', 'fftshift', 'ifftshift',
           'Tensor']

# 将 torch.Tensor 赋值给 Tensor 变量
Tensor = torch.Tensor

# 定义 fft 函数，并为其添加文档字符串
fft = _add_docstr(_fft.fft_fft, r"""
fft(input, n=None, dim=-1, norm=None, *, out=None) -> Tensor

Computes the one dimensional discrete Fourier transform of :attr:`input`.

Note:
    The Fourier domain representation of any real signal satisfies the
    Hermitian property: `X[i] = conj(X[-i])`. This function always returns both
    the positive and negative frequency terms even though, for real inputs, the
    negative frequencies are redundant. :func:`~torch.fft.rfft` returns the
    more compact one-sided representation where only the positive frequencies
    are returned.

Note:
    Supports torch.half and torch.chalf on CUDA with GPU Architecture SM53 or greater.
    However it only supports powers of 2 signal length in every transformed dimension.

Args:
    input (Tensor): the input tensor
    n (int, optional): Signal length. If given, the input will either be zero-padded
        or trimmed to this length before computing the FFT.
    dim (int, optional): The dimension along which to take the one dimensional FFT.
    norm (str, optional): Normalization mode. For the forward transform
        (:func:`~torch.fft.fft`), these correspond to:

        * ``"forward"`` - normalize by ``1/n``
        * ``"backward"`` - no normalization
        * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the FFT orthonormal)

        Calling the backward transform (:func:`~torch.fft.ifft`) with the same
        normalization mode will apply an overall normalization of ``1/n`` between
        the two transforms. This is required to make :func:`~torch.fft.ifft`
        the exact inverse.

        Default is ``"backward"`` (no normalization).

Keyword args:
    {out}

Example:

    >>> t = torch.arange(4)
    >>> t
    tensor([0, 1, 2, 3])
    >>> torch.fft.fft(t)
    tensor([ 6.+0.j, -2.+2.j, -2.+0.j, -2.-2.j])

    >>> t = torch.tensor([0.+1.j, 2.+3.j, 4.+5.j, 6.+7.j])
    >>> torch.fft.fft(t)
    tensor([12.+16.j, -8.+0.j, -4.-4.j,  0.-8.j])
""".format(**common_args))

# 定义 ifft 函数，并为其添加文档字符串
ifft = _add_docstr(_fft.fft_ifft, r"""
ifft(input, n=None, dim=-1, norm=None, *, out=None) -> Tensor

Computes the one dimensional inverse discrete Fourier transform of :attr:`input`.

Note:
    Supports torch.half and torch.chalf on CUDA with GPU Architecture SM53 or greater.
    However it only supports powers of 2 signal length in every transformed dimension.

Args:
    input (Tensor): the input tensor
    # n (int, optional): 信号长度。如果提供了此参数，在计算逆FFT之前，输入信号将被零填充或修剪到这个长度。
    # dim (int, optional): 执行一维逆FFT的维度。
    # norm (str, optional): 规范化模式。对于逆变换（torch.fft.ifft），这些选项是：

    # * "forward" - 不进行规范化
    # * "backward" - 通过 ``1/n`` 进行规范化
    # * "ortho" - 通过 ``1/sqrt(n)`` 进行规范化（使得IFFT为正交归一化）

    # 使用相同的规范化模式调用正向变换（torch.fft.fft）将在两个变换之间应用总体规范化 ``1/n``。这是使得torch.fft.ifft成为精确的逆变换所必需的。

    # 默认为 ``"backward"`` （通过 ``1/n`` 进行规范化）。
fft2 = _add_docstr(_fft.fft_fft2, r"""
fft2(input, s=None, dim=(-2, -1), norm=None, *, out=None) -> Tensor

Computes the 2 dimensional discrete Fourier transform of :attr:`input`.
Equivalent to :func:`~torch.fft.fftn` but FFTs only the last two dimensions by default.

Note:
    The Fourier domain representation of any real signal satisfies the
    Hermitian property: ``X[i, j] = conj(X[-i, -j])``. This
    function always returns all positive and negative frequency terms even
    though, for real inputs, half of these values are redundant.
    :func:`~torch.fft.rfft2` returns the more compact one-sided representation
    where only the positive frequencies of the last dimension are returned.

Note:
    Supports torch.half and torch.chalf on CUDA with GPU Architecture SM53 or greater.
    However it only supports powers of 2 signal length in every transformed dimensions.

Args:
    input (Tensor): the input tensor
    s (Tuple[int], optional): Signal size in the transformed dimensions.
        If given, each dimension ``dim[i]`` will either be zero-padded or
        trimmed to the length ``s[i]`` before computing the FFT.
        If a length ``-1`` is specified, no padding is done in that dimension.
        Default: ``s = [input.size(d) for d in dim]``
    dim (Tuple[int], optional): Dimensions to be transformed.
        Default: last two dimensions.
    norm (str, optional): Normalization mode. For the forward transform
        (:func:`~torch.fft.fft2`), these correspond to:

        * ``"forward"`` - normalize by ``1/n``
        * ``"backward"`` - no normalization
        * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the FFT orthonormal)

        Where ``n = prod(s)`` is the logical FFT size.
        Calling the backward transform (:func:`~torch.fft.ifft2`) with the same
        normalization mode will apply an overall normalization of ``1/n``
        between the two transforms. This is required to make
        :func:`~torch.fft.ifft2` the exact inverse.

        Default is ``"backward"`` (no normalization).

Keyword args:
    {out}

Example:

    >>> x = torch.rand(10, 10, dtype=torch.complex64)
    >>> fft2 = torch.fft.fft2(x)

    The discrete Fourier transform is separable, so :func:`~torch.fft.fft2`
    here is equivalent to two one-dimensional :func:`~torch.fft.fft` calls:

    >>> two_ffts = torch.fft.fft(torch.fft.fft(x, dim=0), dim=1)
    >>> torch.testing.assert_close(fft2, two_ffts, check_stride=False)

""".format(**common_args))



ifft2 = _add_docstr(_fft.fft_ifft2, r"""
ifft2(input, s=None, dim=(-2, -1), norm=None, *, out=None) -> Tensor

Computes the 2 dimensional inverse discrete Fourier transform of :attr:`input`.
Equivalent to :func:`~torch.fft.ifftn` but IFFTs only the last two dimensions by default.

Note:
    The inverse Fourier transform of a real signal is also real-valued.
    If the input tensor is purely real, the output tensor will be purely real.

Args:
    input (Tensor): the input tensor
    s (Tuple[int], optional): Signal size in the transformed dimensions.
        If given, each dimension ``dim[i]`` will either be zero-padded or
        trimmed to the length ``s[i]`` before computing the IFFT.
        If a length ``-1`` is specified, no padding is done in that dimension.
        Default: ``s = [input.size(d) for d in dim]``
    dim (Tuple[int], optional): Dimensions to be transformed.
        Default: last two dimensions.
    norm (str, optional): Normalization mode. For the forward transform
        (:func:`~torch.fft.fft2`), these correspond to:

        * ``"forward"`` - normalize by ``1/n``
        * ``"backward"`` - no normalization
        * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the FFT orthonormal)

        Where ``n = prod(s)`` is the logical FFT size.
        Calling the backward transform (:func:`~torch.fft.ifft2`) with the same
        normalization mode will apply an overall normalization of ``1/n``
        between the two transforms. This is required to make
        :func:`~torch.fft.ifft2` the exact inverse.

        Default is ``"backward"`` (no normalization).

Keyword args:
    {out}

Example:

    >>> x = torch.rand(10, 10, dtype=torch.complex64)
    >>> ifft2 = torch.fft.ifft2(x)

    The inverse discrete Fourier transform is separable, so :func:`~torch.fft.ifft2`
    here is equivalent to two one-dimensional :func:`~torch.fft.ifft` calls:

    >>> two_iffts = torch.fft.ifft(torch.fft.ifft(x, dim=0), dim=1)
    >>> torch.testing.assert_close(ifft2, two_iffts, check_stride=False)

""".format(**common_args))
    Supports torch.half and torch.chalf on CUDA with GPU Architecture SM53 or greater.
    仅在支持的 GPU 架构（SM53 或更高版本）上，支持 torch.half 和 torch.chalf 数据类型。
    However it only supports powers of 2 signal length in every transformed dimensions.
    然而，仅支持每个转换维度中长度为 2 的幂的信号长度。
fftn = _add_docstr(_fft.fft_fftn, r"""
fftn(input, s=None, dim=None, norm=None, *, out=None) -> Tensor

计算输入张量 :attr:`input` 的 N 维离散傅里叶变换。

Note:
    任何实信号的傅里叶域表示满足共轭对称性质：``X[i_1, ..., i_n] = conj(X[-i_1, ..., -i_n])``。
    即使对于实输入，一半的值是冗余的，本函数总是返回所有正负频率的项。
    :func:`~torch.fft.rfftn` 返回更紧凑的单边表示，仅返回最后一维度的正频率。

Note:
    在支持的情况下，对于 CUDA 上的 torch.half 和 torch.chalf，需要 GPU 架构 SM53 或更高。
    然而，它仅支持每个转换维度的长度为 2 的幂。

Args:
    input (Tensor): 输入张量
    s (Tuple[int], optional): 转换维度中的信号大小。
        如果给定，每个维度 ``dim[i]`` 在计算 FFT 前将被零填充或截断到长度 ``s[i]``。
        如果指定长度为 ``-1``，则在该维度不进行填充。
        默认值为 ``s = [input.size(d) for d in dim]``
    dim (Tuple[int], optional): 要进行变换的维度。
        默认为所有维度，或者如果给定了 :attr:`s`，则为最后 ``len(s)`` 维度。
    norm (str, optional): 归一化模式。对于反向变换（:func:`~torch.fft.ifft2`），这些对应于：

        * ``"forward"`` - 不归一化
        * ``"backward"`` - 归一化为 ``1/n``
        * ``"ortho"`` - 归一化为 ``1/sqrt(n)``（使得 IFFT 正交归一化）

        其中 ``n = prod(s)`` 是逻辑 IFFT 大小。
        使用相同的归一化模式调用正向变换（:func:`~torch.fft.fft2`）将在两个变换之间应用总体归一化 ``1/n``。
        这是确保 :func:`~torch.fft.ifft2` 的确切逆变换所必需的。

        默认值为 ``"backward"``（归一化为 ``1/n``）。

Keyword args:
    {out}

Example:

    >>> x = torch.rand(10, 10, dtype=torch.complex64)
    >>> ifft2 = torch.fft.ifft2(x)

    离散傅里叶变换是可分离的，因此在这里 :func:`~torch.fft.ifft2`
    等效于两个一维 :func:`~torch.fft.ifft` 调用：

    >>> two_iffts = torch.fft.ifft(torch.fft.ifft(x, dim=0), dim=1)
    >>> torch.testing.assert_close(ifft2, two_iffts, check_stride=False)

""".format(**common_args))
    norm (str, optional): Normalization mode. For the forward transform
        (:func:`~torch.fft.fftn`), these correspond to:

        * ``"forward"`` - normalize by ``1/n``
        * ``"backward"`` - no normalization
        * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the FFT orthonormal)

        Where ``n = prod(s)`` is the logical FFT size.
        范数（str，可选）：归一化模式。对于正向变换
        （:func:`~torch.fft.fftn`），这些对应于：

        * ``"forward"`` - 通过 ``1/n`` 进行归一化
        * ``"backward"`` - 没有归一化
        * ``"ortho"`` - 通过 ``1/sqrt(n)`` 进行归一化（使FFT正交化）

        其中 ``n = prod(s)`` 是逻辑FFT大小。
        Calling the backward transform (:func:`~torch.fft.ifftn`) with the same
        normalization mode will apply an overall normalization of ``1/n``
        between the two transforms. This is required to make
        :func:`~torch.fft.ifftn` the exact inverse.
        调用相同的归一化模式进行反向变换（:func:`~torch.fft.ifftn`）将在两个变换之间施加一个总体的归一化 ``1/n``。
        这是确保 :func:`~torch.fft.ifftn` 成为精确反变换所必需的。

        Default is ``"backward"`` (no normalization).
        默认是 ``"backward"``（没有归一化）。
# 定义函数 rfft，用于计算实数输入的一维傅里叶变换
rfft = _add_docstr(_fft.fft_rfft, r"""
rfft(input, n=None, dim=-1, norm=None, *, out=None) -> Tensor

Computes the one dimensional Fourier transform of real-valued :attr:`input`.

The FFT of a real signal is Hermitian-symmetric, ``X[i] = conj(X[-i])`` so
the output contains only the positive frequencies below the Nyquist frequency.
To compute the full output, use :func:`~torch.fft.fft`

Note:
    # 在支持的情况下，支持在CUDA上使用 torch.half（SM53或更高版本的GPU架构）。

Args:
    input (Tensor): 输入张量
    n (int, optional): 变换的长度，如果提供，将用零填充或修剪每个维度到长度 `n`
    dim (int, optional): 要变换的维度，默认为最后一个维度
    norm (str, optional): 规范化模式
        - `"backward"`: 默认选项，通过 `1/n` 进行归一化
    out (Tensor, optional): 输出张量，可选

Keyword args:
    {out}

Example:

    >>> x = torch.rand(10, 10, dtype=torch.float32)
    >>> rfft = torch.fft.rfft(x)

    实数信号的傅里叶变换是共轭对称的，因此输出只包含 Nyquist 频率以下的正频率。
    若要计算完整输出，请使用 :func:`~torch.fft.fft`。

    >>> full_fft = torch.fft.fft(x)
    >>> torch.testing.assert_close(rfft, full_fft[..., :rfft.size(-1)])

""".format(**common_args))
    However it only supports powers of 2 signal length in every transformed dimension.


注释：


# 然而，它只支持每个转换维度中长度为2的幂的信号长度。
irfft = _add_docstr(_fft.fft_irfft, r"""
irfft(input, n=None, dim=-1, norm=None, *, out=None) -> Tensor

计算 :func:`~torch.fft.rfft` 的逆变换。

:attr:`input` 在傅里叶域中被解释为单边埃尔米特信号，由 :func:`~torch.fft.rfft` 生成。根据埃尔米特性质，输出将是实值。

Note:
    为了满足埃尔米特性质，一些输入频率必须是实值。在这些情况下，虚部将被忽略。
    例如，任何零频率项中的虚部不能在实输出中表示，因此将始终被忽略。

Note:
    正确解释埃尔米特输入取决于原始数据的长度，由 :attr:`n` 给出。这是因为每个输入形状可能对应于奇数或偶数长度的信号。
    默认情况下，信号被假定为偶数长度，奇数信号将无法完全回传。因此，建议始终传递信号长度 :attr:`n`。

Note:
    支持在具有GPU架构SM53或更高版本的CUDA上的 torch.half 和 torch.chalf。
    但是，在每个转换维度中，仅支持2的幂信号长度。使用默认参数时，转换维度的大小应为 (2^n + 1)，因为参数 `n` 默认为偶数，输出大小为 2 * (transformed_dim_size - 1)。

Args:
    input (Tensor): 表示半埃尔米特信号的输入张量
""")
    n (int, optional): 输出信号的长度。这决定了输出信号的长度。如果提供了此参数，在计算实部逆FFT之前，输入信号将被零填充或修剪到指定长度。
        默认情况下，长度为偶数：``n=2*(input.size(dim) - 1)``.
    dim (int, optional): 执行一维实部逆FFT的维度。
    norm (str, optional): 归一化模式。对于反向变换（:func:`~torch.fft.irfft`），这些模式对应：

        * ``"forward"`` - 无归一化
        * ``"backward"`` - 归一化因子为 ``1/n``
        * ``"ortho"`` - 归一化因子为 ``1/sqrt(n)`` （使得实部逆FFT为正交归一化）

        使用相同的归一化模式调用前向变换（:func:`~torch.fft.rfft`）将在两个变换之间应用总体归一化因子 ``1/n``。这是为了使得 :func:`~torch.fft.irfft`
        成为精确的逆变换。

        默认为 ``"backward"`` （归一化因子为 ``1/n``）。
rfft2 = _add_docstr(_fft.fft_rfft2, r"""
rfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None) -> Tensor

Computes the 2-dimensional discrete Fourier transform of real :attr:`input`.
Equivalent to :func:`~torch.fft.rfftn` but FFTs only the last two dimensions by default.

The FFT of a real signal is Hermitian-symmetric, ``X[i, j] = conj(X[-i, -j])``,
so the full :func:`~torch.fft.fft2` output contains redundant information.
:func:`~torch.fft.rfft2` instead omits the negative frequencies in the last
dimension.

Note:
    Supports torch.half on CUDA with GPU Architecture SM53 or greater.
    However it only supports powers of 2 signal length in every transformed dimensions.

Args:
    input (Tensor): the input tensor
    s (Tuple[int], optional): Signal size in the transformed dimensions.
        If given, each dimension ``dim[i]`` will either be zero-padded or
        trimmed to the length ``s[i]`` before computing the real FFT.
        If a length ``-1`` is specified, no padding is done in that dimension.
        Default: ``s = [input.size(d) for d in dim]``
    dim (Tuple[int], optional): Dimensions to be transformed.
        Default: last two dimensions.
    norm (str, optional): Normalization mode. For the forward transform
        (:func:`~torch.fft.rfft2`), these correspond to:

        * ``"forward"`` - normalize by ``1/n``
        * ``"backward"`` - no normalization
        * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the real FFT orthonormal)

        Where ``n = prod(s)`` is the logical FFT size.
        Calling the backward transform (:func:`~torch.fft.irfft2`) with the same
        normalization mode will apply an overall normalization of ``1/n`` between
        the two transforms. This is required to make :func:`~torch.fft.irfft2`
        the exact inverse.

        Default is ``"backward"`` (no normalization).

Keyword args:
    {out}  # Placeholder for additional keyword arguments provided by the user.

Example:

    >>> t = torch.rand(10, 10)
    # Compute the 2D real FFT of tensor t
    >>> rfft2 = torch.fft.rfft2(t)
    # Check the size of the resulting tensor
    >>> rfft2.size()
    torch.Size([10, 6])

    # Compare against the full output from torch.fft.fft2, focusing on frequencies up to the Nyquist limit
    >>> fft2 = torch.fft.fft2(t)
    >>> torch.testing.assert_close(fft2[..., :6], rfft2, check_stride=False)
    # 离散傅立叶变换是可分离的，因此这里的 torch.fft.rfft2 等效于 torch.fft.fft 和 torch.fft.rfft 的组合：
    
    # 对输入张量 t 在 dim=1 维度上进行实部傅立叶变换（Real Fast Fourier Transform，RFFT）
    rfft_result = torch.fft.rfft(t, dim=1)
    
    # 对上一步得到的结果在 dim=0 维度上进行完整的傅立叶变换
    two_ffts = torch.fft.fft(rfft_result, dim=0)
    
    # 使用测试工具检查 rfft2 和两次傅立叶变换的结果是否接近，不检查张量的步幅
    torch.testing.assert_close(rfft2, two_ffts, check_stride=False)
""".format(**common_args))

irfft2 = _add_docstr(_fft.fft_irfft2, r"""
irfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None) -> Tensor

Computes the inverse of :func:`~torch.fft.rfft2`.
Equivalent to :func:`~torch.fft.irfftn` but IFFTs only the last two dimensions by default.

:attr:`input` is interpreted as a one-sided Hermitian signal in the Fourier
domain, as produced by :func:`~torch.fft.rfft2`. By the Hermitian property, the
output will be real-valued.

Note:
    Some input frequencies must be real-valued to satisfy the Hermitian
    property. In these cases the imaginary component will be ignored.
    For example, any imaginary component in the zero-frequency term cannot
    be represented in a real output and so will always be ignored.

Note:
    The correct interpretation of the Hermitian input depends on the length of
    the original data, as given by :attr:`s`. This is because each input shape
    could correspond to either an odd or even length signal. By default, the
    signal is assumed to be even length and odd signals will not round-trip
    properly. So, it is recommended to always pass the signal shape :attr:`s`.

Note:
    Supports torch.half and torch.chalf on CUDA with GPU Architecture SM53 or greater.
    However it only supports powers of 2 signal length in every transformed dimensions.
    With default arguments, the size of last dimension should be (2^n + 1) as argument
    `s` defaults to even output size = 2 * (last_dim_size - 1)

Args:
    input (Tensor): the input tensor
    s (Tuple[int], optional): Signal size in the transformed dimensions.
        If given, each dimension ``dim[i]`` will either be zero-padded or
        trimmed to the length ``s[i]`` before computing the real FFT.
        If a length ``-1`` is specified, no padding is done in that dimension.
        Defaults to even output in the last dimension:
        ``s[-1] = 2*(input.size(dim[-1]) - 1)``.
    dim (Tuple[int], optional): Dimensions to be transformed.
        The last dimension must be the half-Hermitian compressed dimension.
        Default: last two dimensions.
    norm (str, optional): Normalization mode. For the backward transform
        (:func:`~torch.fft.irfft2`), these correspond to:

        * ``"forward"`` - no normalization
        * ``"backward"`` - normalize by ``1/n``
        * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the real IFFT orthonormal)

        Where ``n = prod(s)`` is the logical IFFT size.
        Calling the forward transform (:func:`~torch.fft.rfft2`) with the same
        normalization mode will apply an overall normalization of ``1/n`` between
        the two transforms. This is required to make :func:`~torch.fft.irfft2`
        the exact inverse.

        Default is ``"backward"`` (normalize by ``1/n``).

Keyword args:
    {out}  # Placeholder for additional keyword arguments documentation

Example:

    >>> t = torch.rand(10, 9)
    >>> T = torch.fft.rfft2(t)

    Without specifying the output length to :func:`~torch.fft.irfft2`, the output
    will not round-trip properly because the input is odd-length in the last
    dimension:

    >>> torch.fft.irfft2(T).size()
    torch.Size([10, 8])

    因此，如果输入在最后一个维度上的长度是奇数，将无法正确进行往返变换。

    So, it is recommended to always pass the signal shape :attr:`s`.

    因此，建议始终传递信号的形状 :attr:`s`。

    >>> roundtrip = torch.fft.irfft2(T, t.size())
    >>> roundtrip.size()
    torch.Size([10, 9])

    进行往返变换时，传递信号形状可以避免此类问题。

    >>> torch.testing.assert_close(roundtrip, t, check_stride=False)

    使用断言确保往返变换后的结果与原始信号在数值上接近，忽略检查步幅。
""".format(**common_args))

rfftn = _add_docstr(_fft.fft_rfftn, r"""
rfftn(input, s=None, dim=None, norm=None, *, out=None) -> Tensor

计算实数输入 :attr:`input` 的 N 维离散傅里叶变换。

实信号的傅里叶变换具有共轭对称性，即 ``X[i_1, ..., i_n] = conj(X[-i_1, ..., -i_n])``，
因此完整的 :func:`~torch.fft.fftn` 输出包含冗余信息。
:func:`~torch.fft.rfftn` 在最后一个维度上省略了负频率。

Note:
    在支持的条件下，如 CUDA 上的 GPU 架构 SM53 或更高版本，支持 torch.half 类型。
    然而，每个转换维度的信号长度只支持 2 的幂次方。

Args:
    input (Tensor): 输入张量
    s (Tuple[int], optional): 转换维度中的信号大小。
        如果给定，则每个维度 ``dim[i]`` 在计算实数 FFT 前将被零填充或修剪到长度 ``s[i]``。
        如果指定长度为 ``-1``，则在该维度上不进行填充。
        默认值：``s = [input.size(d) for d in dim]``
    dim (Tuple[int], optional): 要进行变换的维度。
        默认值：所有维度，或者如果给定了 :attr:`s`，则为最后 ``len(s)`` 个维度。
    norm (str, optional): 归一化模式。对于正向变换（:func:`~torch.fft.rfftn`），这些选项为：

        * ``"forward"`` - 通过 ``1/n`` 进行归一化
        * ``"backward"`` - 不进行归一化
        * ``"ortho"`` - 通过 ``1/sqrt(n)`` 进行归一化（使得实数 FFT 正交归一化）

        其中 ``n = prod(s)`` 是逻辑 FFT 大小。
        使用相同归一化模式调用反向变换（:func:`~torch.fft.irfftn`）将在两个变换之间应用整体归一化 ``1/n``。
        这是使得 :func:`~torch.fft.irfftn` 成为确切逆变换所必需的。

        默认值是 ``"backward"``（无归一化）。

Keyword args:
    {out}

Example:

    >>> t = torch.rand(10, 10)
    >>> rfftn = torch.fft.rfftn(t)
    >>> rfftn.size()
    torch.Size([10, 6])

    与 :func:`~torch.fft.fftn` 的完整输出进行比较，我们得到了所有低于奈奎斯特频率的元素。

    >>> fftn = torch.fft.fftn(t)
    >>> torch.testing.assert_close(fftn[..., :6], rfftn, check_stride=False)

    离散傅里叶变换是可分离的，因此这里的 :func:`~torch.fft.rfftn` 等效于 :func:`~torch.fft.fft` 和
    :func:`~torch.fft.rfft` 的组合：

    >>> two_ffts = torch.fft.fft(torch.fft.rfft(t, dim=1), dim=0)
    >>> torch.testing.assert_close(rfftn, two_ffts, check_stride=False)

""".format(**common_args))

irfftn = _add_docstr(_fft.fft_irfftn, r"""
irfftn(input, s=None, dim=None, norm=None, *, out=None) -> Tensor

计算 :func:`~torch.fft.rfftn` 的逆变换。

:attr:`input` 在傅里叶域中被解释为单边的 Hermitian 信号，由 :func:`~torch.fft.rfftn` 生成。
根据 Hermitian 特性，输出将是实值的。

Note:
    Some input frequencies must be real-valued to satisfy the Hermitian
    property. In these cases the imaginary component will be ignored.
    For example, any imaginary component in the zero-frequency term cannot
    be represented in a real output and so will always be ignored.
hfft = _add_docstr(_fft.fft_hfft, r"""
hfft(input, n=None, dim=-1, norm=None, *, out=None) -> Tensor

Computes the one dimensional discrete Fourier transform of a Hermitian
symmetric :attr:`input` signal.

Note:
    :func:`~torch.fft.hfft`/:func:`~torch.fft.ihfft` are analogous to
    the real-to-complex and complex-to-real transform pair of :func:`~torch.fft.rfft`/:func:`~torch.fft.irfft`.

Args:
    input (Tensor): the input tensor containing Hermitian symmetric data.
    n (int, optional): Number of elements in the transformed dimension. If not provided,
        the size will be inferred from the input tensor size.
    dim (int, optional): Dimension along which the transform is applied. Default is the last dimension (-1).
    norm (str, optional): Normalization mode for the transform. Can be one of:
        - ``None`` or ``"backward"``: No normalization.
        - ``"ortho"``: Normalize by ``1/sqrt(n)`` (making the transform orthonormal).
    out (Tensor, optional): Output tensor to store the result.

Returns:
    Tensor: the Fourier transform of the input signal.

Example:

    >>> t = torch.tensor([1.0, 2.0, 3.0, 4.0])
    >>> T = torch.fft.hfft(t)
    >>> T
    tensor([10.,  0., -2.])

    >>> roundtrip = torch.fft.ihfft(T, t.size())
    >>> roundtrip
    tensor([1., 2., 3., 4.])

""".format(**common_args))
    :func:`~torch.fft.rfft`/:func:`~torch.fft.irfft`. The real FFT expects
    a real signal in the time-domain and gives a Hermitian symmetry in the
    frequency-domain. The Hermitian FFT is the opposite; Hermitian symmetric in
    the time-domain and real-valued in the frequency-domain. For this reason,
    special care needs to be taken with the length argument :attr:`n`, in the
    same way as with :func:`~torch.fft.irfft`.
Note:
    因为在时域中信号是埃尔米特对称的，所以在频域中结果将是实数。需要注意的是，为了满足埃尔米特性质，某些输入频率必须是实数值。在这些情况下，虚部将被忽略。例如，任何在 `input[0]` 中的虚部将导致一个或多个复频率项，这些无法表示为实数输出，因此始终将被忽略。

Note:
    正确解释埃尔米特输入依赖于原始数据的长度，由 `n` 给出。这是因为每个输入形状可能对应于奇数或偶数长度的信号。默认情况下，信号被假定为偶数长度，奇数信号将无法完全往返。因此，建议始终传递信号长度 `n`。

Note:
    支持在 CUDA 上使用 torch.half 和 torch.chalf，但需要 GPU 架构为 SM53 或更高。然而，每个转换维度中的信号长度必须是 2 的幂。使用默认参数时，转换维度的大小应为 (2^n + 1)，因为参数 `n` 默认为偶数输出大小的一半：`2 * (transformed_dim_size - 1)`

Args:
    input (Tensor): 表示半埃尔米特信号的输入张量
    n (int, optional): 输出信号长度。这决定了实数输出的长度。如果给定，计算埃尔米特FFT之前将对输入进行零填充或修剪到这个长度。默认为偶数输出：`n=2*(input.size(dim) - 1)`
    dim (int, optional): 进行一维埃尔米特FFT的维度
    norm (str, optional): 归一化模式。对于正向变换（:func:`~torch.fft.hfft`），这些对应于：

        * ``"forward"`` - 归一化为 ``1/n``
        * ``"backward"`` - 不进行归一化
        * ``"ortho"`` - 归一化为 ``1/sqrt(n)``（使埃尔米特FFT正交化）

        使用相同的归一化模式调用反向变换（:func:`~torch.fft.ihfft`）将在两个变换之间应用总体归一化 ``1/n``。这是使 :func:`~torch.fft.ihfft` 成为精确逆变换所必需的。

        默认为 ``"backward"``（无归一化）。

Keyword args:
    {out}

Example:

    将实值频率信号带入时间域将产生埃尔米特对称的输出：

    >>> t = torch.linspace(0, 1, 5)
    >>> t
    tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
    >>> T = torch.fft.ifft(t)
    >>> T
    tensor([ 0.5000-0.0000j, -0.1250-0.1720j, -0.1250-0.0406j, -0.1250+0.0406j,
            -0.1250+0.1720j])

    注意 `T[1] == T[-1].conj()` 和 `T[2] == T[-2].conj()` 是多余的。因此，我们可以计算不考虑负频率的正向变换：

    >>> torch.fft.hfft(T[:3], n=5)
    tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
    # 类似于 `torch.fft.irfft` 函数，为了恢复偶数长度的输出，必须指定输出长度。
    torch.fft.hfft(T[:3])
    # 对输入的前三个元素执行 Hermitian FFT（HFFT），返回频谱的上半部分。
    # 输出是一个张量，包含频谱的实部和虚部的组合。
    # 示例中的输出是：tensor([0.1250, 0.2809, 0.6250, 0.9691])
""".format(**common_args))

# 将 _fft.fft_ihfft 函数添加文档字符串
ihfft = _add_docstr(_fft.fft_ihfft, r"""
ihfft(input, n=None, dim=-1, norm=None, *, out=None) -> Tensor

计算 :func:`~torch.fft.hfft` 的逆变换。

:attr:`input` 必须是实数信号，在 Fourier 域中解释。
实数信号的逆 FFT 是 Hermitian 对称的，``X[i] = conj(X[-i])``。
:func:`~torch.fft.ihfft` 以只包含 Nyquist 以下正频率的单边形式表示。
要计算完整的输出，请使用 :func:`~torch.fft.ifft`。

注意：
    在支持的情况下，CUDA 上的 torch.half 需要 GPU 架构 SM53 或更高版本。
    然而，每个转换维度中只支持 2 的幂长度的信号。

Args:
    input (Tensor): 实数输入张量
    n (int, optional): 信号长度。如果给定，将在计算 Hermitian IFFT 前对输入进行零填充或修剪到此长度。
    dim (int, optional): 进行一维 Hermitian IFFT 的维度。
    norm (str, optional): 归一化模式。对于反向变换（:func:`~torch.fft.ihfft`），这些对应于：

        * ``"forward"`` - 无归一化
        * ``"backward"`` - 归一化为 ``1/n``
        * ``"ortho"`` - 归一化为 ``1/sqrt(n)`` （使 IFFT 正交归一化）

        使用相同归一化模式调用前向变换（:func:`~torch.fft.hfft`）将在两个变换之间应用整体归一化为 ``1/n``。
        这是确保 :func:`~torch.fft.ihfft` 的精确逆变换所必需的。

        默认为 ``"backward"`` （归一化为 ``1/n``）。

Keyword args:
    {out}

Example:

    >>> t = torch.arange(5)
    >>> t
    tensor([0, 1, 2, 3, 4])
    >>> torch.fft.ihfft(t)
    tensor([ 2.0000-0.0000j, -0.5000-0.6882j, -0.5000-0.1625j])

    与 :func:`~torch.fft.ifft` 的完整输出进行比较：

    >>> torch.fft.ifft(t)
    tensor([ 2.0000-0.0000j, -0.5000-0.6882j, -0.5000-0.1625j, -0.5000+0.1625j,
            -0.5000+0.6882j])
""".format(**common_args))

# 将 _fft.fft_hfft2 函数添加文档字符串
hfft2 = _add_docstr(_fft.fft_hfft2, r"""
hfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None) -> Tensor

计算 Hermitian 对称 :attr:`input` 信号的二维离散 Fourier 变换。
等价于 :func:`~torch.fft.hfftn`，但默认只转换最后两个维度。

:attr:`input` 在时间域中被解释为单边 Hermitian 信号。
根据 Hermitian 属性，傅立叶变换将是实数值。

注意：
    在支持的情况下，CUDA 上的 torch.half 和 torch.chalf 需要 GPU 架构 SM53 或更高版本。
    然而，每个转换维度中只支持 2 的幂长度的信号。
    默认情况下，最后一个维度的大小应为 (2^n + 1)，因为参数 `s` 默认为偶数输出大小 = 2 * (last_dim_size - 1)

Args:
    input (Tensor): 输入张量
    # 定义函数 hfft2，实现二维半埃尔米特傅里叶变换（Hermitian FFT）。
    def hfft2(input, s=None, dim=(-2, -1), norm='backward'):
        # 如果给定了信号大小 s，则在变换的维度上进行零填充或修剪，以便计算半埃尔米特傅里叶变换。
        # 如果某维度指定为 -1，则在该维度上不进行填充。
        # 默认情况下，在最后一个维度上输出为偶数：s[-1] = 2 * (input.size(dim[-1]) - 1)。
        
        # 如果未指定变换的维度 dim，则默认为最后两个维度。
        
        # norm 参数用于指定归一化模式：
        # - "forward" 表示通过 1/n 进行归一化
        # - "backward" 表示不进行归一化
        # - "ortho" 表示通过 1/sqrt(n) 进行归一化，使得半埃尔米特傅里叶变换正交归一化。
        # 其中，n = prod(s) 是逻辑 FFT 尺寸的乘积。
        # 使用相同的归一化模式调用反变换函数 torch.fft.ihfft2 时，会在两次变换之间应用整体的 1/n 归一化。
        # 这是为了确保 torch.fft.ihfft2 是确切的逆变换。

        # 默认情况下，norm 参数为 "backward"（不进行归一化）。
ihfft2 = _add_docstr(_fft.fft_ihfft2, r"""
ihfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None) -> Tensor

计算实数输入的二维逆离散傅里叶变换（IDFT）。与 :func:`~torch.fft.ihfftn` 等效，但默认仅转换最后两个维度。

Note:
    在支持的 CUDA GPU 架构 SM53 及更高版本上，支持 torch.half。
    然而，每个转换维度的信号长度必须是 2 的幂。

Args:
    input (Tensor): 输入张量
    s (Tuple[int], optional): 转换维度的信号大小。
        如果给定，每个维度 ``dim[i]`` 将在计算埃尔米特逆FFT之前填充或修剪到长度 ``s[i]``。
        如果指定长度为 ``-1``，则在该维度上不进行填充。
        默认值为 ``s = [input.size(d) for d in dim]``
    dim (Tuple[int], optional): 要进行变换的维度。
        默认为最后两个维度。
    norm (str, optional): 标准化模式。对于反向变换
        (:func:`~torch.fft.ihfft2`)，这些对应于：

        * ``"forward"`` - 无标准化
        * ``"backward"`` - 按 ``1/n`` 标准化
        * ``"ortho"`` - 按 ``1/sqrt(n)`` 标准化（使埃尔米特逆FFT正交规范化）

        其中 ``n = prod(s)`` 是逻辑IFFT大小。
        使用相同标准化模式调用前向变换 (:func:`~torch.fft.hfft2`) 将在两个变换之间应用 ``1/n`` 的整体标准化。
        这是使 :func:`~torch.fft.ihfft2` 成为精确逆的必要条件。

        默认值为 ``"backward"``（按 ``1/n`` 标准化）。

Keyword args:
    {out}

Example:

    >>> T = torch.rand(10, 10)
    >>> t = torch.fft.ihfft2(T)
    >>> t.size()
    torch.Size([10, 6])

    与 :func:`~torch.fft.ifft2` 的完整输出相比，埃尔米特时域信号仅占一半空间。

    >>> fftn = torch.fft.ifft2(t)
    >>> torch.allclose(fftn[..., :6], T)
    True

    离散傅里叶变换是可分离的，因此这里的 :func:`~torch.fft.ihfft2`
    等效于 :func:`~torch.fft.ifft` 和 :func:`~torch.fft.ihfft` 的组合：

    >>> two_ffts = torch.fft.ifft(torch.fft.ihfft(t, dim=1), dim=0)
    >>> torch.allclose(t, two_ffts)
    # 返回布尔值 True
    True
# 使用 format 方法将 common_args 中的参数应用到字符串中
""".format(**common_args))

# 定义 hfftn 函数并添加文档字符串
hfftn = _add_docstr(_fft.fft_hfftn, r"""
hfftn(input, s=None, dim=None, norm=None, *, out=None) -> Tensor

计算具有 Hermitian 对称性的 n 维离散 Fourier 变换，输入为 Hermitian 对称信号。

参数:
    input (Tensor): 输入张量
    s (Tuple[int], 可选): 在变换的维度中的信号大小。如果给定，则每个维度 `dim[i]` 在计算实数 FFT 之前将被零填充或修剪到长度 `s[i]`。如果指定长度 `-1`，则在该维度上不进行填充。
        默认为最后一个维度上的偶数输出：
        `s[-1] = 2*(input.size(dim[-1]) - 1)`。
    dim (Tuple[int], 可选): 要变换的维度。最后一个维度必须是压缩的半-Hermitian 维度。默认为所有维度，或者如果给定了 `s`，则为最后 `len(s)` 个维度。

注意:
    :func:`~torch.fft.hfftn` / :func:`~torch.fft.ihfftn` 类似于 :func:`~torch.fft.rfftn` / :func:`~torch.fft.irfftn`。实数 FFT 预期时间域中的实数信号，并在频域中具有 Hermitian 对称性。Hermitian FFT 则相反；在时间域中具有 Hermitian 对称性，并在频域中是实数值。因此，需要特别注意形状参数 `s`，与 :func:`~torch.fft.irfftn` 类似。

注意:
    一些输入频率必须是实数值，以满足 Hermitian 属性。在这些情况下，将忽略虚部分量。例如，零频率项中的任何虚部分量都无法在实数输出中表示，因此将始终被忽略。

注意:
    正确解释 Hermitian 输入取决于原始数据的长度，由 `s` 给出。这是因为每个输入形状可能对应于奇数或偶数长度的信号。默认情况下，信号被假定为偶数长度，奇数信号将无法正确往返传输。建议始终传递信号形状 `s`。

注意:
    在支持的情况下，torch.half 和 torch.chalf 在带有 GPU 架构 SM53 或更高版本的 CUDA 上也受支持。然而，每个转换维度中仅支持 2 的幂次信号长度。使用默认参数，最后一个维度的大小应为 (2^n + 1) 作为参数 `s` 默认为偶数输出大小 = 2 * (last_dim_size - 1)

返回:
    Tensor: 变换后的张量
""")
    norm (str, optional): Normalization mode. For the forward transform
        (:func:`~torch.fft.hfftn`), these correspond to:

        * ``"forward"`` - normalize by ``1/n``
            正向变换的归一化方式，通过 ``1/n`` 进行归一化
        * ``"backward"`` - no normalization
            反向变换的归一化方式，无归一化
        * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the Hermitian FFT orthonormal)
            正交变换的归一化方式，通过 ``1/sqrt(n)`` 进行归一化，使得Hermitian FFT正交

        Where ``n = prod(s)`` is the logical FFT size.
        Calling the backward transform (:func:`~torch.fft.ihfftn`) with the same
        normalization mode will apply an overall normalization of ``1/n`` between
        the two transforms. This is required to make :func:`~torch.fft.ihfftn`
        the exact inverse.

        Default is ``"backward"`` (no normalization).
            默认值为 ``"backward"`` （无归一化）
ihfftn = _add_docstr(_fft.fft_ihfftn, r"""
ihfftn(input, s=None, dim=None, norm=None, *, out=None) -> Tensor

计算实数值输入的 N 维逆离散傅里叶变换。

:attr:`input` 必须是一个实值信号，在傅里叶域中进行解释。
一个实信号的 n 维逆傅里叶变换是共轭对称的，
``X[i, j, ...] = conj(X[-i, -j, ...])``。 :func:`~torch.fft.ihfftn` 表示
这种形式，其中仅包括最后信号维度中小于奈奎斯特频率的正频率。要计算完整的输出，请使用 :func:`~torch.fft.ifftn`。

注意:
    在支持 CUDA 的 GPU 架构 SM53 或更高版本上支持 torch.half。
    但它仅支持每个转换维度的长度为 2 的幂。

Args:
    input (Tensor): 输入张量
    s (Tuple[int], 可选): 在转换的维度中的信号大小。
        如果给出，每个维度 ``dim[i]`` 将在计算共轭对称逆傅里叶变换之前进行零填充或修剪到长度 ``s[i]``。
        如果指定长度为 ``-1``，则在该维度中不进行填充。
        默认值: ``s = [input.size(d) for d in dim]``
    dim (Tuple[int], 可选): 要转换的维度。
        默认为所有维度，或者如果给定了 :attr:`s`，则为最后 ``len(s)`` 个维度。
    norm (str, 可选): 归一化模式。对于反向变换（:func:`~torch.fft.ihfftn`），这些对应于：

        * ``"forward"`` - 不进行归一化
        * ``"backward"`` - 通过 ``1/n`` 进行归一化
        * ``"ortho"`` - 通过 ``1/sqrt(n)`` 进行归一化（使共轭对称逆傅里叶变换正交规范化）

        其中 ``n = prod(s)`` 是逻辑逆傅里叶变换的大小。
        使用相同的归一化模式调用正向变换（:func:`~torch.fft.hfftn`）将在两个变换之间应用总体归一化 ``1/n``。
        这是使 :func:`~torch.fft.ihfftn` 成为确切逆变换所必需的。

        默认为 ``"backward"``（通过 ``1/n`` 归一化）。

Keyword args:
    {out}

Example:

    >>> T = torch.rand(10, 10)
    >>> ihfftn = torch.fft.ihfftn(T)
    >>> ihfftn.size()
    torch.Size([10, 6])

    与 :func:`~torch.fft.ifftn` 的完整输出进行比较，我们得到了所有小于奈奎斯特频率的元素。


""")
    # 对输入张量 t 进行多维逆傅里叶变换（IFFT）
    ifftn = torch.fft.ifftn(t)
    
    # 检查逆傅里叶变换结果的前 6 个元素是否与 ihfftn 张量的对应元素在误差范围内相等
    torch.allclose(ifftn[..., :6], ihfftn)
    True
    
    # 由于离散傅里叶变换是可分的，因此这里的 torch.fft.ihfftn 等效于 torch.fft.ihfft 和 torch.fft.ifft 的组合：
    
    # 先在第一个维度上应用逆 Hermitean FFT（torch.fft.ihfft），然后在第零维度上应用逆 FFT（torch.fft.ifft）
    two_iffts = torch.fft.ifft(torch.fft.ihfft(t, dim=1), dim=0)
    
    # 检查两种方法得到的结果是否在误差范围内相等
    torch.allclose(ihfftn, two_iffts)
    True
""".format(**common_args))

# 添加文档字符串并返回带格式化参数的结果
fftfreq = _add_docstr(_fft.fft_fftfreq, r"""
fftfreq(n, d=1.0, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Computes the discrete Fourier Transform sample frequencies for a signal of size :attr:`n`.

Note:
    By convention, :func:`~torch.fft.fft` returns positive frequency terms
    first, followed by the negative frequencies in reverse order, so that
    ``f[-i]`` for all :math:`0 < i \leq n/2`` in Python gives the negative
    frequency terms. For an FFT of length :attr:`n` and with inputs spaced in
    length unit :attr:`d`, the frequencies are::

        f = [0, 1, ..., (n - 1) // 2, -(n // 2), ..., -1] / (d * n)

Note:
    For even lengths, the Nyquist frequency at ``f[n/2]`` can be thought of as
    either negative or positive. :func:`~torch.fft.fftfreq` follows NumPy's
    convention of taking it to be negative.

Args:
    n (int): the FFT length
    d (float, optional): The sampling length scale.
        The spacing between individual samples of the FFT input.
        The default assumes unit spacing, dividing that result by the actual
        spacing gives the result in physical frequency units.

Keyword Args:
    {out} (optional): Output tensor.
    {dtype} (optional): Data type specification.
    {layout} (optional): Layout of the tensor.
    {device} (optional): Device where the tensor is stored.
    {requires_grad} (optional): Whether to compute gradients.

Example:

    >>> torch.fft.fftfreq(5)
    tensor([ 0.0000,  0.2000,  0.4000, -0.4000, -0.2000])

    For even input, we can see the Nyquist frequency at ``f[2]`` is given as
    negative:

    >>> torch.fft.fftfreq(4)
    tensor([ 0.0000,  0.2500, -0.5000, -0.2500])
""".format(**factory_common_args))

# 添加文档字符串并返回带格式化参数的结果
rfftfreq = _add_docstr(_fft.fft_rfftfreq, r"""
rfftfreq(n, d=1.0, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Computes the sample frequencies for :func:`~torch.fft.rfft` with a signal of size :attr:`n`.

Note:
    :func:`~torch.fft.rfft` returns Hermitian one-sided output, so only the
    positive frequency terms are returned. For a real FFT of length :attr:`n`
    and with inputs spaced in length unit :attr:`d`, the frequencies are::

        f = torch.arange((n + 1) // 2) / (d * n)

Note:
    For even lengths, the Nyquist frequency at ``f[n/2]`` can be thought of as
    either negative or positive. Unlike :func:`~torch.fft.fftfreq`,
    :func:`~torch.fft.rfftfreq` always returns it as positive.

Args:
    n (int): the real FFT length
    d (float, optional): The sampling length scale.
        The spacing between individual samples of the FFT input.
        The default assumes unit spacing, dividing that result by the actual
        spacing gives the result in physical frequency units.

Keyword Args:
    {out} (optional): Output tensor.
    {dtype} (optional): Data type specification.
    {layout} (optional): Layout of the tensor.
    {device} (optional): Device where the tensor is stored.
    {requires_grad} (optional): Whether to compute gradients.

Example:

    >>> torch.fft.rfftfreq(5)
    tensor([0.0000, 0.2000, 0.4000])

    >>> torch.fft.rfftfreq(4)
    tensor([0.0000, 0.2500, 0.5000])

    Compared to the output from :func:`~torch.fft.fftfreq`, we see that the
    Nyquist frequency at ``f[2]`` has changed sign:
""")
    # 使用 PyTorch 中的 fftfreq 函数计算长度为 4 的 FFT（快速傅里叶变换）频率
    >>> torch.fft.fftfreq(4)
    # 返回一个张量（tensor），包含长度为 4 的 FFT 的频率分量
    tensor([ 0.0000,  0.2500, -0.5000, -0.2500])
""".format(**factory_common_args))

# 添加文档字符串到 fft_fftshift 函数
fftshift = _add_docstr(_fft.fft_fftshift, r"""
fftshift(input, dim=None) -> Tensor

Reorders n-dimensional FFT data, as provided by :func:`~torch.fft.fftn`, to have
negative frequency terms first.

This performs a periodic shift of n-dimensional data such that the origin
``(0, ..., 0)`` is moved to the center of the tensor. Specifically, to
``input.shape[dim] // 2`` in each selected dimension.

Note:
    By convention, the FFT returns positive frequency terms first, followed by
    the negative frequencies in reverse order, so that ``f[-i]`` for all
    :math:`0 < i \leq n/2` in Python gives the negative frequency terms.
    :func:`~torch.fft.fftshift` rearranges all frequencies into ascending order
    from negative to positive with the zero-frequency term in the center.

Note:
    For even lengths, the Nyquist frequency at ``f[n/2]`` can be thought of as
    either negative or positive. :func:`~torch.fft.fftshift` always puts the
    Nyquist term at the 0-index. This is the same convention used by
    :func:`~torch.fft.fftfreq`.

Args:
    input (Tensor): the tensor in FFT order
    dim (int, Tuple[int], optional): The dimensions to rearrange.
        Only dimensions specified here will be rearranged, any other dimensions
        will be left in their original order.
        Default: All dimensions of :attr:`input`.

Example:

    >>> f = torch.fft.fftfreq(4)
    >>> f
    tensor([ 0.0000,  0.2500, -0.5000, -0.2500])

    >>> torch.fft.fftshift(f)
    tensor([-0.5000, -0.2500,  0.0000,  0.2500])

    Also notice that the Nyquist frequency term at ``f[2]`` was moved to the
    beginning of the tensor.

    This also works for multi-dimensional transforms:

    >>> x = torch.fft.fftfreq(5, d=1/5) + 0.1 * torch.fft.fftfreq(5, d=1/5).unsqueeze(1)
    >>> x
    tensor([[ 0.0000,  1.0000,  2.0000, -2.0000, -1.0000],
            [ 0.1000,  1.1000,  2.1000, -1.9000, -0.9000],
            [ 0.2000,  1.2000,  2.2000, -1.8000, -0.8000],
            [-0.2000,  0.8000,  1.8000, -2.2000, -1.2000],
            [-0.1000,  0.9000,  1.9000, -2.1000, -1.1000]])

    >>> torch.fft.fftshift(x)
    tensor([[-2.2000, -1.2000, -0.2000,  0.8000,  1.8000],
            [-2.1000, -1.1000, -0.1000,  0.9000,  1.9000],
            [-2.0000, -1.0000,  0.0000,  1.0000,  2.0000],
            [-1.9000, -0.9000,  0.1000,  1.1000,  2.1000],
            [-1.8000, -0.8000,  0.2000,  1.2000,  2.2000]])

    :func:`~torch.fft.fftshift` can also be useful for spatial data. If our
    data is defined on a centered grid (``[-(N//2), (N-1)//2]``) then we can
    use the standard FFT defined on an uncentered grid (``[0, N)``) by first
    applying an :func:`~torch.fft.ifftshift`.

    >>> x_centered = torch.arange(-5, 5)
    >>> x_uncentered = torch.fft.ifftshift(x_centered)
    >>> fft_uncentered = torch.fft.fft(x_uncentered)

    Similarly, we can convert the frequency domain components to centered
""")
    # 对 fft_uncentered 执行 Fourier 变换的中心化，即应用 fftshift 函数
    >>> fft_centered = torch.fft.fftshift(fft_uncentered)

    # 反向变换，从中心化的 Fourier 空间回到中心化的空间数据，需要按照相反的顺序应用逆向的位移：
    # 首先对 fft_centered 执行逆 fftshift，然后逆傅里叶变换 ifft，最后再次逆 fftshift
    >>> x_centered_2 = torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(fft_centered)))
    # 使用 torch.testing.assert_close 检查 x_centered_2 和 x_centered 是否接近，不检查步幅
    >>> torch.testing.assert_close(x_centered.to(torch.complex64), x_centered_2, check_stride=False)
# 定义 ifftshift 变量，将 _fft.fft_ifftshift 函数包装为带有文档字符串的函数
ifftshift = _add_docstr(_fft.fft_ifftshift, r"""
ifftshift(input, dim=None) -> Tensor

Inverse of :func:`~torch.fft.fftshift`.

Args:
    input (Tensor): 要进行 FFT 逆序变换的张量
    dim (int, Tuple[int], optional): 要重新排列的维度。
        只有在这里指定的维度会被重新排列，其它维度保持原始顺序。
        默认值：:attr:`input` 的所有维度。

Example:

    >>> f = torch.fft.fftfreq(5)
    >>> f
    tensor([ 0.0000,  0.2000,  0.4000, -0.4000, -0.2000])

    通过 :func:`~torch.fft.fftshift` 和 :func:`~torch.fft.ifftshift` 的往返操作可以得到相同的结果：

    >>> shifted = torch.fft.fftshift(f)
    >>> torch.fft.ifftshift(shifted)
    tensor([ 0.0000,  0.2000,  0.4000, -0.4000, -0.2000])

""")
```