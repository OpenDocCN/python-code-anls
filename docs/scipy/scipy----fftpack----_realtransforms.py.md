# `D:\src\scipysrc\scipy\scipy\fftpack\_realtransforms.py`

```
"""
Real spectrum transforms (DCT, DST, MDCT)
"""

__all__ = ['dct', 'idct', 'dst', 'idst', 'dctn', 'idctn', 'dstn', 'idstn']

from scipy.fft import _pocketfft  # 导入私有模块 _pocketfft
from ._helper import _good_shape  # 导入 _good_shape 函数

# 定义逆变换类型映射字典
_inverse_typemap = {1: 1, 2: 3, 3: 2, 4: 4}


def dctn(x, type=2, shape=None, axes=None, norm=None, overwrite_x=False):
    """
    Return multidimensional Discrete Cosine Transform along the specified axes.

    Parameters
    ----------
    x : array_like
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DCT (see Notes). Default type is 2.
    shape : int or array_like of ints or None, optional
        The shape of the result. If both `shape` and `axes` (see below) are
        None, `shape` is ``x.shape``; if `shape` is None but `axes` is
        not None, then `shape` is ``numpy.take(x.shape, axes, axis=0)``.
        If ``shape[i] > x.shape[i]``, the ith dimension is padded with zeros.
        If ``shape[i] < x.shape[i]``, the ith dimension is truncated to
        length ``shape[i]``.
        If any element of `shape` is -1, the size of the corresponding
        dimension of `x` is used.
    axes : int or array_like of ints or None, optional
        Axes along which the DCT is computed.
        The default is over all axes.
    norm : {None, 'ortho'}, optional
        Normalization mode (see Notes). Default is None.
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.

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
    >>> from scipy.fftpack import dctn, idctn
    >>> rng = np.random.default_rng()
    >>> y = rng.standard_normal((16, 16))
    >>> np.allclose(y, idctn(dctn(y, norm='ortho'), norm='ortho'))
    True

    """
    # 根据输入和参数计算出最终的 shape
    shape = _good_shape(x, shape, axes)
    # 调用底层的 FFT 库进行多维 DCT 变换
    return _pocketfft.dctn(x, type, shape, axes, norm, overwrite_x)


def idctn(x, type=2, shape=None, axes=None, norm=None, overwrite_x=False):
    """
    Return multidimensional Inverse Discrete Cosine Transform along the specified axes.

    Parameters
    ----------
    x : array_like
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DCT (see Notes). Default type is 2.
    # `shape` 控制结果的形状，可以是整数、整数数组或者 None。如果 `shape` 和 `axes`（见下文）都是 None，则结果形状为 `x.shape`；
    # 如果 `shape` 是 None 但 `axes` 不是 None，则 `shape` 为 `numpy.take(x.shape, axes, axis=0)`。
    # 如果 `shape[i] > x.shape[i]`，则第 i 维将用零填充；如果 `shape[i] < x.shape[i]`，则第 i 维将被截断为 `shape[i]` 的长度。
    # 如果 `shape` 中的任何元素为 -1，则使用 `x` 对应维度的大小。
    shape : int or array_like of ints or None, optional
    
    # `axes` 控制进行 IDCT 变换的轴。默认是对所有轴进行变换。
    axes : int or array_like of ints or None, optional
    
    # 规范化模式，控制变换的归一化方式（见注释）。默认为 None。
    norm : {None, 'ortho'}, optional
    
    # 如果为 True，则可以销毁 `x` 的内容；默认为 False。
    overwrite_x : bool, optional
    
    # 返回变换后的实数数组 `y`。
    Returns
    -------
    y : ndarray of real
    
    # 查看 `idctn` 的文档，了解更多关于 IDCT 类型和规范化模式的详细信息及参考资料。
    See Also
    --------
    dctn : 多维 DCT
    
    # 关于 IDCT 类型和规范化模式的完整详情，请参阅 `idct`。
    Notes
    -----
    
    # 示例：使用 `numpy` 和 `scipy.fftpack` 库计算 IDCT 变换，确保逆变换满足正变换的逆过程。
    Examples
    --------
    >>> import numpy as np
    >>> from scipy.fftpack import dctn, idctn
    >>> rng = np.random.default_rng()
    >>> y = rng.standard_normal((16, 16))
    >>> np.allclose(y, idctn(dctn(y, norm='ortho'), norm='ortho'))
    True
    """
    # 确定变换类型，根据 `_inverse_typemap` 中的映射找到对应的类型
    type = _inverse_typemap[type]
    
    # 根据输入 `x`、`shape` 和 `axes` 确定有效的形状 `shape`
    shape = _good_shape(x, shape, axes)
    
    # 调用 `_pocketfft.dctn` 执行多维 IDCT 变换，返回结果
    return _pocketfft.dctn(x, type, shape, axes, norm, overwrite_x)
# 返回沿指定轴进行多维离散正弦变换的函数
def dstn(x, type=2, shape=None, axes=None, norm=None, overwrite_x=False):
    """
    Return multidimensional Discrete Sine Transform along the specified axes.

    Parameters
    ----------
    x : array_like
        输入数组。
    type : {1, 2, 3, 4}, optional
        DST 的类型（参见备注）。默认为类型 2。
    shape : int or array_like of ints or None, optional
        结果的形状。如果 `shape` 和 `axes` 都是 None，则 `shape` 是 `x.shape`；
        如果 `shape` 是 None 但 `axes` 不是 None，则 `shape` 是 `numpy.take(x.shape, axes, axis=0)`。
        如果 `shape[i] > x.shape[i]`，则第 i 维用零填充。
        如果 `shape[i] < x.shape[i]`，则第 i 维被截断为长度 `shape[i]`。
        如果 `shape` 中的任何元素是 -1，则使用 `x` 的相应维度的大小。
    axes : int or array_like of ints or None, optional
        进行 DST 的轴。
        默认情况下是所有轴。
    norm : {None, 'ortho'}, optional
        标准化模式（参见备注）。默认为 None。
    overwrite_x : bool, optional
        如果为 True，则可以破坏 `x` 的内容；默认为 False。

    Returns
    -------
    y : ndarray of real
        变换后的输入数组。

    See Also
    --------
    idstn : 反向多维 DST

    Notes
    -----
    有关 DST 类型和标准化模式的详细信息以及参考资料，请参阅 `dst`。

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.fftpack import dstn, idstn
    >>> rng = np.random.default_rng()
    >>> y = rng.standard_normal((16, 16))
    >>> np.allclose(y, idstn(dstn(y, norm='ortho'), norm='ortho'))
    True

    """
    # 根据输入数组 `x`、指定参数调用底层的 `dstn` 函数
    shape = _good_shape(x, shape, axes)
    return _pocketfft.dstn(x, type, shape, axes, norm, overwrite_x)


def idstn(x, type=2, shape=None, axes=None, norm=None, overwrite_x=False):
    """
    Return multidimensional Inverse Discrete Sine Transform along the specified axes.

    Parameters
    ----------
    x : array_like
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the IDST (see Notes). Default type is 2.
    shape : int or array_like of ints or None, optional
        The shape of the result.  If both `shape` and `axes` (see below) are
        None, `shape` is ``x.shape``; if `shape` is None but `axes` is
        not None, then `shape` is ``numpy.take(x.shape, axes, axis=0)``.
        If ``shape[i] > x.shape[i]``, the ith dimension is padded with zeros.
        If ``shape[i] < x.shape[i]``, the ith dimension is truncated to
        length ``shape[i]``.
        If any element of `shape` is -1, the size of the corresponding
        dimension of `x` is used.
    axes : int or array_like of ints or None, optional
        Axes along which the IDST is computed.
        The default is over all axes.

    """
    # 返回沿指定轴进行多维逆离散正弦变换的函数
    pass  # 该函数目前尚未实现，因此使用 pass 语句占位
    # 确定变换类型，将字符串类型映射为对应的整数类型
    type = _inverse_typemap[type]
    # 确定有效的数据形状，根据输入数据 `x`、指定的形状和轴
    shape = _good_shape(x, shape, axes)
    # 调用底层 FFT 实现的 DSTN 变换函数，返回变换后的结果
    return _pocketfft.dstn(x, type, shape, axes, norm, overwrite_x)
# 定义一个函数 dct，用于计算任意类型序列 x 的离散余弦变换（DCT）

r"""
返回任意类型序列 x 的离散余弦变换。

Parameters
----------
x : array_like
    输入数组。
type : {1, 2, 3, 4}, optional
    DCT 的类型（参见注释）。默认为 2。
n : int, optional
    变换的长度。如果 ``n < x.shape[axis]``，则截断 `x`；如果 ``n > x.shape[axis]``，则用零填充 `x`。
    默认情况下，``n = x.shape[axis]``。
axis : int, optional
    计算 DCT 的轴；默认为最后一个轴（即 ``axis=-1``）。
norm : {None, 'ortho'}, optional
    归一化模式（参见注释）。默认为 None。
overwrite_x : bool, optional
    如果为 True，`x` 的内容可以被破坏；默认为 False。

Returns
-------
y : ndarray of real
    变换后的数组。

See Also
--------
idct : 逆离散余弦变换

Notes
-----
对于单维数组 ``x``，``dct(x, norm='ortho')`` 等同于 MATLAB 中的 ``dct(x)``。

理论上有 8 种 DCT 类型，但 scipy 中只实现了前 4 种。

**Type I**

DCT-I 有多种定义，我们使用以下定义（对于 ``norm=None``）

.. math::

   y_k = x_0 + (-1)^k x_{N-1} + 2 \sum_{n=1}^{N-2} x_n \cos\left(
   \frac{\pi k n}{N-1} \right)

如果 ``norm='ortho'``，则 ``x[0]`` 和 ``x[N-1]`` 乘以缩放因子 :math:`\sqrt{2}`，并且 ``y[k]`` 乘以缩放因子 ``f``

.. math::

    f = \begin{cases}
     \frac{1}{2}\sqrt{\frac{1}{N-1}} & \text{if }k=0\text{ or }N-1, \\
     \frac{1}{2}\sqrt{\frac{2}{N-1}} & \text{otherwise} \end{cases}

**Type II**

DCT-II 有多种定义，我们使用以下定义（对于 ``norm=None``）

.. math::

   y_k = 2 \sum_{n=0}^{N-1} x_n \cos\left(\frac{\pi k(2n+1)}{2N} \right)

如果 ``norm='ortho'``，则 ``y[k]`` 乘以缩放因子 ``f``

.. math::
   f = \begin{cases}
   \sqrt{\frac{1}{4N}} & \text{if }k=0, \\
   \sqrt{\frac{1}{2N}} & \text{otherwise} \end{cases}

**Type III**

DCT-III 有多种定义，我们使用以下定义（对于 ``norm=None``）

.. math::

   y_k = x_0 + 2 \sum_{n=1}^{N-1} x_n \cos\left(\frac{\pi(2k+1)n}{2N}\right)

或者，对于 ``norm='ortho'``

.. math::

   y_k = \frac{x_0}{\sqrt{N}} + \sqrt{\frac{2}{N}} \sum_{n=1}^{N-1} x_n
   \cos\left(\frac{\pi(2k+1)n}{2N}\right)

"""
    # 返回使用 PocketFFT 库计算的离散余弦变换 (DCT)。
    # 这里的 DCT 是针对给定类型的变换（例如类型 IV），类型 IV 的具体定义如下所示。
    # 使用 PocketFFT 库计算 DCT，传递给函数的参数包括输入数组 x，变换类型 type，长度 n，轴 axis，归一化选项 norm，以及是否覆盖输入数组的 overwrite_x 参数。
    # 返回变换后的结果。
    return _pocketfft.dct(x, type, n, axis, norm, overwrite_x)
# 返回指定类型的逆离散余弦变换的结果
def idct(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False):
    # 将指定的类型转换为其逆映射，以便在内部使用正确的类型进行计算
    type = _inverse_typemap[type]
    # 调用底层库执行离散余弦变换（DCT）的逆变换，返回变换后的数组
    return _pocketfft.dct(x, type, n, axis, norm, overwrite_x)


def dst(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False):
    r"""
    返回任意类型序列 x 的离散正弦变换（DST）。

    Parameters
    ----------
    x : array_like
        输入数组。
    type : {1, 2, 3, 4}, optional
        DST 的类型（见注释）。默认为类型 2。
    n : int, optional
        变换的长度。如果 ``n < x.shape[axis]``，则对 `x` 进行截断。
        如果 ``n > x.shape[axis]``，则对 `x` 进行零填充。默认结果为 ``n = x.shape[axis]``。
    axis : int, optional
        计算 DST 的轴；默认是最后一个轴（即 ``axis=-1``）。
    norm : {None, 'ortho'}, optional
        标准化模式（见注释）。默认为 None。
    overwrite_x : bool, optional
        如果为 True，则可以销毁 `x` 的内容；默认为 False。

    Returns
    -------
    dst : ndarray of reals
        变换后的输入数组。

    See Also
    --------
    ```
    # 返回离散正弦变换（DST）的逆变换结果，使用快速傅里叶变换（FFT）算法进行计算
    
    """
        --------
        idst : Inverse DST
    
        Notes
        -----
        For a single dimension array ``x``.
    
        There are, theoretically, 8 types of the DST for different combinations of
        even/odd boundary conditions and boundary off sets [1]_, only the first
        4 types are implemented in scipy.
    
        **Type I**
    
        There are several definitions of the DST-I; we use the following
        for ``norm=None``. DST-I assumes the input is odd around `n=-1` and `n=N`.
    
        .. math::
    
            y_k = 2 \sum_{n=0}^{N-1} x_n \sin\left(\frac{\pi(k+1)(n+1)}{N+1}\right)
    
        Note that the DST-I is only supported for input size > 1.
        The (unnormalized) DST-I is its own inverse, up to a factor ``2(N+1)``.
        The orthonormalized DST-I is exactly its own inverse.
    
        **Type II**
    
        There are several definitions of the DST-II; we use the following for
        ``norm=None``. DST-II assumes the input is odd around `n=-1/2` and
        `n=N-1/2`; the output is odd around :math:`k=-1` and even around `k=N-1`
    
        .. math::
    
            y_k = 2 \sum_{n=0}^{N-1} x_n \sin\left(\frac{\pi(k+1)(2n+1)}{2N}\right)
    
        if ``norm='ortho'``, ``y[k]`` is multiplied by a scaling factor ``f``
    
        .. math::
    
            f = \begin{cases}
            \sqrt{\frac{1}{4N}} & \text{if }k = 0, \\
            \sqrt{\frac{1}{2N}} & \text{otherwise} \end{cases}
    
        **Type III**
    
        There are several definitions of the DST-III, we use the following (for
        ``norm=None``). DST-III assumes the input is odd around `n=-1` and even
        around `n=N-1`
    
        .. math::
    
            y_k = (-1)^k x_{N-1} + 2 \sum_{n=0}^{N-2} x_n \sin\left(
            \frac{\pi(2k+1)(n+1)}{2N}\right)
    
        The (unnormalized) DST-III is the inverse of the (unnormalized) DST-II, up
        to a factor ``2N``. The orthonormalized DST-III is exactly the inverse of the
        orthonormalized DST-II.
    
        .. versionadded:: 0.11.0
    
        **Type IV**
    
        There are several definitions of the DST-IV, we use the following (for
        ``norm=None``). DST-IV assumes the input is odd around `n=-0.5` and even
        around `n=N-0.5`
    
        .. math::
    
            y_k = 2 \sum_{n=0}^{N-1} x_n \sin\left(\frac{\pi(2k+1)(2n+1)}{4N}\right)
    
        The (unnormalized) DST-IV is its own inverse, up to a factor ``2N``. The
        orthonormalized DST-IV is exactly its own inverse.
    
        .. versionadded:: 1.2.0
           Support for DST-IV.
    
        References
        ----------
        .. [1] Wikipedia, "Discrete sine transform",
               https://en.wikipedia.org/wiki/Discrete_sine_transform
    
        """
        # 使用 _pocketfft 模块的 dst 函数进行逆正弦变换计算
        return _pocketfft.dst(x, type, n, axis, norm, overwrite_x)
# 返回一个任意类型序列的逆离散正弦变换（IDST）结果

def idst(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False):
    """
    返回一个任意类型序列的逆离散正弦变换（IDST）结果。

    Parameters
    ----------
    x : array_like
        输入数组。
    type : {1, 2, 3, 4}, optional
        DST 的类型（参见注释）。默认为 type 2。
    n : int, optional
        变换的长度。如果 ``n < x.shape[axis]``，则截断 `x`。如果 ``n > x.shape[axis]``，则在 `x` 后面补零。
        默认结果为 ``n = x.shape[axis]``。
    axis : int, optional
        计算 IDST 的轴；默认在最后一个轴上进行计算（即 ``axis=-1``）。
    norm : {None, 'ortho'}, optional
        归一化模式（参见注释）。默认为 None。
    overwrite_x : bool, optional
        如果为 True，则可以销毁 `x` 的内容；默认为 False。

    Returns
    -------
    idst : ndarray of real
        变换后的实数数组。

    See Also
    --------
    dst : 正向 DST

    Notes
    -----
    'The' IDST 是类型为 2 的 IDST，与类型为 3 的 DST 相同。

    类型为 1 的 IDST 是类型为 1 的 DST，类型为 2 的 IDST 是类型为 3 的 DST，
    类型为 3 的 IDST 是类型为 2 的 DST。关于这些类型的定义，请参见 `dst`。

    .. versionadded:: 0.11.0

    """
    # 将 type 转换为相应的逆变换类型
    type = _inverse_typemap[type]
    # 调用内部函数 _pocketfft.dst 执行 DST 的逆变换，并返回结果
    return _pocketfft.dst(x, type, n, axis, norm, overwrite_x)
```