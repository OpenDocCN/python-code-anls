# `D:\src\scipysrc\scipy\scipy\fft\_realtransforms.py`

```
# 从 _basic 模块中导入 _dispatch 函数
from ._basic import _dispatch
# 导入 Dispatchable 类，用于创建可调度对象
from scipy._lib.uarray import Dispatchable
# 导入 numpy 库并使用 np 别名
import numpy as np

# 定义公开的函数列表
__all__ = ['dct', 'idct', 'dst', 'idst', 'dctn', 'idctn', 'dstn', 'idstn']

# 使用 _dispatch 装饰器，用于分派函数调用
@_dispatch
def dctn(x, type=2, s=None, axes=None, norm=None, overwrite_x=False,
         workers=None, *, orthogonalize=None):
    """
    Return multidimensional Discrete Cosine Transform along the specified axes.

    Parameters
    ----------
    x : array_like
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DCT (see Notes). Default type is 2.
    s : int or array_like of ints or None, optional
        The shape of the result. If both `s` and `axes` (see below) are None,
        `s` is ``x.shape``; if `s` is None but `axes` is not None, then `s` is
        ``numpy.take(x.shape, axes, axis=0)``.
        If ``s[i] > x.shape[i]``, the ith dimension of the input is padded with zeros.
        If ``s[i] < x.shape[i]``, the ith dimension of the input is truncated to length
        ``s[i]``.
        If any element of `s` is -1, the size of the corresponding dimension of
        `x` is used.
    axes : int or array_like of ints or None, optional
        Axes over which the DCT is computed. If not given, the last ``len(s)``
        axes are used, or all axes if `s` is also not specified.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see Notes). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    orthogonalize : bool, optional
        Whether to use the orthogonalized DCT variant (see Notes).
        Defaults to ``True`` when ``norm="ortho"`` and ``False`` otherwise.

        .. versionadded:: 1.8.0

    Returns
    -------
    y : ndarray of real
        The transformed input array.

    See Also
    --------
    idctn : Inverse multidimensional DCT

    Notes
    -----
    For full details of the DCT types and normalization modes, as well as
    references, see `dct`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.fft import dctn, idctn
    >>> rng = np.random.default_rng()
    >>> y = rng.standard_normal((16, 16))
    >>> np.allclose(y, idctn(dctn(y)))
    True

    """
    # 返回一个 Dispatchable 对象，其中包含 x 和 np.ndarray 类型的信息
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def idctn(x, type=2, s=None, axes=None, norm=None, overwrite_x=False,
          workers=None, orthogonalize=None):
    """
    Return multidimensional Inverse Discrete Cosine Transform along the specified axes.

    Parameters
    ----------
    x : array_like
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DCT (see Notes). Default type is 2.
    s : int or array_like of ints or None, optional
        The shape of the result. If both `s` and `axes` (see below) are None,
        `s` is ``x.shape``; if `s` is None but `axes` is not None, then `s` is
        ``numpy.take(x.shape, axes, axis=0)``.
        If ``s[i] > x.shape[i]``, the ith dimension of the input is padded with zeros.
        If ``s[i] < x.shape[i]``, the ith dimension of the input is truncated to length
        ``s[i]``.
        If any element of `s` is -1, the size of the corresponding dimension of
        `x` is used.
    axes : int or array_like of ints or None, optional
        Axes over which the DCT is computed. If not given, the last ``len(s)``
        axes are used, or all axes if `s` is also not specified.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see Notes). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    orthogonalize : bool, optional
        Whether to use the orthogonalized DCT variant (see Notes).

        .. versionadded:: 1.8.0

    Returns
    -------
    y : ndarray of real
        The transformed input array.
    """
    # 定义函数返回值的类型为 Dispatchable，可以是 x 或者 np.ndarray 类型
    return (Dispatchable(x, np.ndarray),)
# 声明一个函数，该函数被装饰器 @_dispatch 装饰
def dstn(x, type=2, s=None, axes=None, norm=None, overwrite_x=False,
         workers=None, orthogonalize=None):
    """
    Return multidimensional Discrete Sine Transform along the specified axes.

    Parameters
    ----------
    x : array_like
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DST (see Notes). Default type is 2.
    s : int or array_like of ints or None, optional
        The shape of the result.  If both `s` and `axes` (see below) are None,
        `s` is ``x.shape``; if `s` is None but `axes` is not None, then `s` is
        ``numpy.take(x.shape, axes, axis=0)``.
        If ``s[i] > x.shape[i]``, the ith dimension of the input is padded with zeros.
        If ``s[i] < x.shape[i]``, the ith dimension of the input is truncated to length
        ``s[i]``.
        If any element of `shape` is -1, the size of the corresponding dimension
        of `x` is used.
    axes : int or array_like of ints or None, optional
        Axes over which the DST is computed. If not given, the last ``len(s)``
        axes are used, or all axes if `s` is also not specified.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see Notes). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    orthogonalize : bool, optional
        Whether to use the orthogonalized DST variant (see Notes).
        Defaults to ``True`` when ``norm="ortho"`` and ``False`` otherwise.

        .. versionadded:: 1.8.0

    Returns
    -------
    y : ndarray of real
        The transformed input array.

    See Also
    --------
    idstn : Inverse multidimensional DST

    Notes
    -----
    For full details of the DST types and normalization modes, as well as
    references, see `dst`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.fft import dstn, idstn
    >>> rng = np.random.default_rng()
    >>> y = rng.standard_normal((16, 16))
    >>> np.allclose(y, idstn(dstn(y)))
    True

    """
    # 返回一个元组，包含一个 Dispatchable 对象和 np.ndarray 类型
    return (Dispatchable(x, np.ndarray),)
    # 定义函数的参数 `s`，表示结果的形状，可以是整数、整数数组或者 None
    # 如果 `s` 和 `axes`（见下文）都是 None，则 `s` 是 `x.shape`
    # 如果 `s` 是 None 而 `axes` 不是 None，则 `s` 是 `numpy.take(x.shape, axes, axis=0)`
    # 如果 `s[i] > x.shape[i]`，则输入的第 i 维度用零填充
    # 如果 `s[i] < x.shape[i]`，则输入的第 i 维度被截断为长度 `s[i]`
    # 如果 `s` 中任何元素为 -1，则使用 `x` 对应维度的大小
    s : int or array_like of ints or None, optional

    # 定义函数的参数 `axes`，表示计算 IDST 的轴。如果未给出，则使用最后 `len(s)` 个轴，或者如果 `s` 也未指定，则使用所有轴
    axes : int or array_like of ints or None, optional

    # 定义函数的参数 `norm`，表示标准化模式（详见备注）。默认为 "backward"
    norm : {"backward", "ortho", "forward"}, optional

    # 定义函数的参数 `overwrite_x`，如果为 True，则 `x` 的内容可以被销毁；默认为 False
    overwrite_x : bool, optional

    # 定义函数的参数 `workers`，用于并行计算的最大工作线程数。如果为负数，则从 `os.cpu_count()` 循环回绕
    # 更多细节参见 `scipy.fft.fft` 函数
    workers : int, optional

    # 定义函数的参数 `orthogonalize`，是否使用正交化的 IDST 变体（详见备注）
    # 当 `norm="ortho"` 时默认为 `True`，否则默认为 `False`
    orthogonalize : bool, optional

    # 返回值说明：返回一个实数 ndarray，表示变换后的输入数组
    Returns
    -------
    y : ndarray of real

    # 查看也可以参考 `dstn` 函数，这是一个多维度 DST 的相关函数
    See Also
    --------
    dstn : multidimensional DST

    # 备注部分：关于 IDST 类型和标准化模式的完整详情及参考，请参见 `idst` 文档
    Notes
    -----
    For full details of the IDST types and normalization modes, as well as
    references, see `idst`.

    # 示例部分：展示了如何使用 IDST 和 IDSTN 函数的示例
    Examples
    --------
    >>> import numpy as np
    >>> from scipy.fft import dstn, idstn
    >>> rng = np.random.default_rng()
    >>> y = rng.standard_normal((16, 16))
    >>> np.allclose(y, idstn(dstn(y)))
    True

    """
    # 返回一个包含 `x` 和 `np.ndarray` 的 Dispatchable 对象
    return (Dispatchable(x, np.ndarray),)
# 定义一个装饰器函数，用于处理 dct 函数的分派
@_dispatch
# 定义 dct 函数，计算任意类型序列 x 的离散余弦变换（DCT）
def dct(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False, workers=None,
        orthogonalize=None):
    r"""Return the Discrete Cosine Transform of arbitrary type sequence x.

    Parameters
    ----------
    x : array_like
        输入数组。
    type : {1, 2, 3, 4}, optional
        DCT 的类型（见备注）。默认为 2。
    n : int, optional
        变换的长度。如果 ``n < x.shape[axis]``，则截断 `x`。如果 ``n > x.shape[axis]``，则用零填充。
        默认为 ``n = x.shape[axis]``。
    axis : int, optional
        计算 DCT 的轴向；默认在最后一个轴上计算（即 ``axis=-1``）。
    norm : {"backward", "ortho", "forward"}, optional
        归一化模式（见备注）。默认为 "backward"。
    overwrite_x : bool, optional
        如果为 True，则可以破坏 `x` 的内容；默认为 False。
    workers : int, optional
        用于并行计算的最大工作线程数。如果为负数，则从 ``os.cpu_count()`` 循环。
        更多细节请参见 :func:`~scipy.fft.fft`。
    orthogonalize : bool, optional
        是否使用正交化的 DCT 变体（见备注）。
        当 ``norm="ortho"`` 时，默认为 ``True``；否则为 ``False``。

        .. versionadded:: 1.8.0

    Returns
    -------
    y : ndarray of real
        转换后的输入数组。

    See Also
    --------
    idct : 逆 DCT

    Notes
    -----
    对于单维数组 ``x``，``dct(x, norm='ortho')`` 等同于 MATLAB 的 ``dct(x)``。

    .. warning:: 对于 ``type in {1, 2, 3}``，当 ``norm="ortho"`` 时会破坏与直接傅里叶变换的直接对应关系。
                 要恢复它，必须指定 ``orthogonalize=False``。

    对于 ``norm="ortho"``，在正向和反向 `dct` 中都会应用相同的整体因子进行缩放。
    默认情况下，变换也被正交化，对于类型 1、2 和 3，这意味着变换定义被修改以使 DCT 矩阵正交化（见下文）。

    对于 ``norm="backward"``，`dct` 上没有缩放，而 `idct` 上缩放因子为 ``1/N``，其中 ``N`` 是 DCT 的 "逻辑" 大小。
    对于 ``norm="forward"``，``1/N`` 归一化应用于前向 `dct`，而 `idct` 没有归一化。

    理论上有 8 种 DCT 类型，但在 SciPy 中只实现了前 4 种。

    **Type I**

    DCT-I 有几种定义；我们使用以下定义（对于 ``norm="backward"``）

    .. math::

       y_k = x_0 + (-1)^k x_{N-1} + 2 \sum_{n=1}^{N-2} x_n \cos\left(
       \frac{\pi k n}{N-1} \right)

    如果 ``orthogonalize=True``，则 ``x[0]`` 和 ``x[N-1]`` 会乘以一个缩放因子 :math:`\sqrt{2}`，
    而 ``y[0]`` 和 ``y[N-1]`` 则会被除以该因子。
    return (Dispatchable(x, np.ndarray),)


注释：


# 返回一个包含 Dispatchable 对象的元组，其中 x 是第一个参数，np.ndarray 是第二个参数
return (Dispatchable(x, np.ndarray),)


这段代码定义了一个函数返回语句，返回一个包含一个 Dispatchable 对象的元组。这个对象的构造函数接受两个参数：x 和 np.ndarray。
# 标记为分发函数的装饰器，用于定义多态函数分派规则
@_dispatch
# 定义反离散余弦变换的函数
def idct(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False,
         workers=None, orthogonalize=None):
    """
    Return the Inverse Discrete Cosine Transform of an arbitrary type sequence.

    Parameters
    ----------
    x : array_like
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DCT (see Notes). Default type is 2.
    n : int, optional
        Length of the transform.  If ``n < x.shape[axis]``, `x` is
        truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The
        default results in ``n = x.shape[axis]``.
    axis : int, optional
        Axis along which the idct is computed; the default is over the
        last axis (i.e., ``axis=-1``).
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see Notes). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    orthogonalize : bool, optional
        Whether to use the orthogonalized IDCT variant (see Notes).
        Defaults to ``True`` when ``norm="ortho"`` and ``False`` otherwise.

        .. versionadded:: 1.8.0

    Returns
    -------
    idct : ndarray of real
        The transformed input array.

    See Also
    --------
    dct : Forward DCT

    Notes
    -----
    For a single dimension array `x`, ``idct(x, norm='ortho')`` is equal to
    MATLAB ``idct(x)``.

    .. warning:: For ``type in {1, 2, 3}``, ``norm="ortho"`` breaks the direct
                 correspondence with the inverse direct Fourier transform. To
                 recover it you must specify ``orthogonalize=False``.

    For ``norm="ortho"`` both the `dct` and `idct` are scaled by the same
    overall factor in both directions. By default, the transform is also
    orthogonalized which for types 1, 2 and 3 means the transform definition is
    modified to give orthogonality of the IDCT matrix (see `dct` for the full
    definitions).

    'The' IDCT is the IDCT-II, which is the same as the normalized DCT-III.

    The IDCT is equivalent to a normal DCT except for the normalization and
    type. DCT type 1 and 4 are their own inverse and DCTs 2 and 3 are each
    other's inverses.

    Examples
    --------
    The Type 1 DCT is equivalent to the DFT for real, even-symmetrical
    inputs. The output is also real and even-symmetrical. Half of the IFFT
    input is used to generate half of the IFFT output:

    >>> from scipy.fft import ifft, idct
    >>> import numpy as np
    >>> ifft(np.array([ 30.,  -8.,   6.,  -2.,   6.,  -8.])).real
    array([  4.,   3.,   5.,  10.,   5.,   3.])
    >>> idct(np.array([ 30.,  -8.,   6.,  -2.]), 1)
    array([  4.,   3.,   5.,  10.])

    """
    # 返回一个 Dispatchable 对象，用于处理输入的多态性
    return (Dispatchable(x, np.ndarray),)
# 定义函数 dst，计算任意类型序列 x 的离散正弦变换（DST）

r"""
Return the Discrete Sine Transform of arbitrary type sequence x.

Parameters
----------
x : array_like
    输入数组。
type : {1, 2, 3, 4}, optional
    DST 的类型（参见 Notes）。默认为 2。
n : int, optional
    变换的长度。如果 ``n < x.shape[axis]``，则截断 `x`。如果 ``n > x.shape[axis]``，则对 `x` 进行零填充。默认为 ``n = x.shape[axis]``。
axis : int, optional
    计算 DST 的轴；默认是最后一个轴（即 ``axis=-1``）。
norm : {"backward", "ortho", "forward"}, optional
    标准化模式（参见 Notes）。默认为 "backward"。
overwrite_x : bool, optional
    如果为 True，则可以销毁 `x` 的内容；默认为 False。
workers : int, optional
    用于并行计算的最大工作线程数。如果为负数，则从 ``os.cpu_count()`` 循环回来。详细信息请参阅 :func:`~scipy.fft.fft`。
orthogonalize : bool, optional
    是否使用正交化的 DST 变体（参见 Notes）。当 ``norm="ortho"`` 时，默认为 ``True``，否则为 ``False``。

    .. versionadded:: 1.8.0

Returns
-------
dst : ndarray of reals
    转换后的输入数组。

See Also
--------
idst : 逆 DST

Notes
-----
.. warning:: 对于 ``type in {2, 3}``，``norm="ortho"`` 会破坏与直接傅里叶变换的直接对应关系。要恢复它，必须指定 ``orthogonalize=False``。

对于 ``norm="ortho"``，`dst` 和 `idst` 在两个方向上都被相同的整体因子缩放。默认情况下，变换也是正交化的，对于类型 2 和 3，这意味着变换定义被修改以使 DST 矩阵正交化（见下文）。

对于 ``norm="backward"``，`dst` 没有缩放，而 `idst` 被缩放为 ``1/N``，其中 ``N`` 是 DST 的“逻辑”大小。

理论上，有 8 种 DST 的类型，针对不同的偶数/奇数边界条件和边界偏移 [1]_，SciPy 中只实现了前 4 种。

**Type I**

DST-I 有几种定义；我们使用以下定义来处理 ``norm="backward"`` 的情况。DST-I 假设输入在 :math:`n=-1` 和 :math:`n=N` 处是奇数。

.. math::

    y_k = 2 \sum_{n=0}^{N-1} x_n \sin\left(\frac{\pi(k+1)(n+1)}{N+1}\right)

注意，DST-I 仅支持输入大小 > 1。
（非标准化的）DST-I 是其自身的逆，乘以因子 :math:`2(N+1)`。
正交化的 DST-I 恰好是其自身的逆。

在这里，``orthogonalize`` 对 DST-I 没有影响，因为 DST-I 矩阵已经是正交的，乘以一个 ``2N`` 的比例因子。

**Type II**
    # 返回一个包含 Dispatchable 对象和 np.ndarray 的元组
    return (Dispatchable(x, np.ndarray),)
# 使用装饰器 @_dispatch 将函数 idst 包装成一个可分派的函数
@_dispatch
# 定义函数 idst，计算任意类型序列的反离散正弦变换（IDST）
def idst(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False,
         workers=None, orthogonalize=None):
    """
    Return the Inverse Discrete Sine Transform of an arbitrary type sequence.

    Parameters
    ----------
    x : array_like
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DST (see Notes). Default type is 2.
    n : int, optional
        Length of the transform. If ``n < x.shape[axis]``, `x` is
        truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The
        default results in ``n = x.shape[axis]``.
    axis : int, optional
        Axis along which the idst is computed; the default is over the
        last axis (i.e., ``axis=-1``).
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see Notes). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    orthogonalize : bool, optional
        Whether to use the orthogonalized IDST variant (see Notes).
        Defaults to ``True`` when ``norm="ortho"`` and ``False`` otherwise.

        .. versionadded:: 1.8.0

    Returns
    -------
    idst : ndarray of real
        The transformed input array.

    See Also
    --------
    dst : Forward DST

    Notes
    -----
    .. warning:: For ``type in {2, 3}``, ``norm="ortho"`` breaks the direct
                 correspondence with the inverse direct Fourier transform.

    For ``norm="ortho"`` both the `dst` and `idst` are scaled by the same
    overall factor in both directions. By default, the transform is also
    orthogonalized which for types 2 and 3 means the transform definition is
    modified to give orthogonality of the DST matrix (see `dst` for the full
    definitions).

    'The' IDST is the IDST-II, which is the same as the normalized DST-III.

    The IDST is equivalent to a normal DST except for the normalization and
    type. DST type 1 and 4 are their own inverse and DSTs 2 and 3 are each
    other's inverses.

    """
    # 返回一个元组，包含输入数组 x 和 np.ndarray 的 Dispatchable 对象
    return (Dispatchable(x, np.ndarray),)
```