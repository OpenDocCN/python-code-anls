# `D:\src\scipysrc\scipy\scipy\signal\_upfirdn.py`

```
# 导入 NumPy 库，用于支持数值计算操作
import numpy as np

# 导入自定义模块中的特定函数和枚举类型
from ._upfirdn_apply import _output_len, _apply, mode_enum

# 定义可以公开访问的模块成员列表
__all__ = ['upfirdn', '_output_len']

# 定义可用的上采样和下采样模式列表
_upfirdn_modes = [
    'constant', 'wrap', 'edge', 'smooth', 'symmetric', 'reflect',
    'antisymmetric', 'antireflect', 'line',
]

# 定义一个函数，用于对滤波器系数进行填充和转置
def _pad_h(h, up):
    """Store coefficients in a transposed, flipped arrangement.

    For example, suppose upRate is 3, and the
    input number of coefficients is 10, represented as h[0], ..., h[9].

    Then the internal buffer will look like this::

       h[9], h[6], h[3], h[0],   // flipped phase 0 coefs
       0,    h[7], h[4], h[1],   // flipped phase 1 coefs (zero-padded)
       0,    h[8], h[5], h[2],   // flipped phase 2 coefs (zero-padded)

    """
    # 计算填充后的系数数组长度，确保可以整除上采样倍率
    h_padlen = len(h) + (-len(h) % up)
    # 创建一个全零数组，将滤波器系数复制到正确的位置，并进行逆序排列
    h_full = np.zeros(h_padlen, h.dtype)
    h_full[:len(h)] = h
    h_full = h_full.reshape(-1, up).T[:, ::-1].ravel()
    return h_full

# 检查给定的模式字符串是否有效，并返回其对应的枚举值
def _check_mode(mode):
    mode = mode.lower()
    enum = mode_enum(mode)
    return enum

# 定义一个辅助类，用于协助进行重采样操作
class _UpFIRDn:
    """Helper for resampling."""
    # 初始化方法，接受参数 h, x_dtype, up, down
    def __init__(self, h, x_dtype, up, down):
        # 将 h 转换为 NumPy 数组
        h = np.asarray(h)
        # 检查 h 是否为一维数组且长度不为零，否则抛出数值错误
        if h.ndim != 1 or h.size == 0:
            raise ValueError('h must be 1-D with non-zero length')
        # 确定输出类型为 h.dtype, x_dtype, np.float32 三者中的结果类型
        self._output_type = np.result_type(h.dtype, x_dtype, np.float32)
        # 将 h 转换为指定的输出类型的 NumPy 数组
        h = np.asarray(h, self._output_type)
        # 将 up 和 down 转换为整数
        self._up = int(up)
        self._down = int(down)
        # 如果 up 或 down 小于 1，则抛出数值错误
        if self._up < 1 or self._down < 1:
            raise ValueError('Both up and down must be >= 1')
        # 对 h 应用 _pad_h 函数，进行转置和翻转，为滤波做准备
        self._h_trans_flip = _pad_h(h, self._up)
        # 将转置和翻转后的 h 转换为连续的 NumPy 数组
        self._h_trans_flip = np.ascontiguousarray(self._h_trans_flip)
        # 记录原始 h 的长度
        self._h_len_orig = len(h)

    # 将准备好的滤波器应用于指定轴的 N 维信号 x
    def apply_filter(self, x, axis=-1, mode='constant', cval=0):
        """Apply the prepared filter to the specified axis of N-D signal x."""
        # 计算输出长度
        output_len = _output_len(self._h_len_orig, x.shape[axis],
                                 self._up, self._down)
        # 明确使用 np.int64 类型的 output_shape 避免在 32 位平台上分配大数组时出现溢出错误
        output_shape = np.asarray(x.shape, dtype=np.int64)
        # 根据指定轴和输出长度更新输出形状
        output_shape[axis] = output_len
        # 创建一个指定形状和数据类型的全零数组作为输出
        out = np.zeros(output_shape, dtype=self._output_type, order='C')
        # 确定有效的轴索引，确保在 x 的维数范围内
        axis = axis % x.ndim
        # 检查并规范化 mode 参数
        mode = _check_mode(mode)
        # 调用 _apply 函数，将滤波器应用于输入信号 x，结果存储在 out 中
        _apply(np.asarray(x, self._output_type),
               self._h_trans_flip, out,
               self._up, self._down, axis, mode, cval)
        # 返回滤波后的输出数组
        return out
# 定义了一个函数 upfirdn，用于进行上采样、有限脉冲响应（FIR）滤波和下采样操作
def upfirdn(h, x, up=1, down=1, axis=-1, mode='constant', cval=0):
    """Upsample, FIR filter, and downsample.

    Parameters
    ----------
    h : array_like
        1-D FIR (finite-impulse response) filter coefficients.
        FIR（有限脉冲响应）滤波器的系数，以数组形式给出。
    x : array_like
        Input signal array.
        输入信号的数组。
    up : int, optional
        Upsampling rate. Default is 1.
        上采样倍率，默认为1。
    down : int, optional
        Downsampling rate. Default is 1.
        下采样倍率，默认为1。
    axis : int, optional
        The axis of the input data array along which to apply the
        linear filter. The filter is applied to each subarray along
        this axis. Default is -1.
        应用线性滤波器的输入数据数组的轴。滤波器应用于沿此轴的每个子数组。默认为-1。
    mode : str, optional
        The signal extension mode to use. The set
        ``{"constant", "symmetric", "reflect", "edge", "wrap"}`` correspond to
        modes provided by `numpy.pad`. ``"smooth"`` implements a smooth
        extension by extending based on the slope of the last 2 points at each
        end of the array. ``"antireflect"`` and ``"antisymmetric"`` are
        anti-symmetric versions of ``"reflect"`` and ``"symmetric"``. The mode
        `"line"` extends the signal based on a linear trend defined by the
        first and last points along the ``axis``.
        信号扩展模式。设置 ``{"constant", "symmetric", "reflect", "edge", "wrap"}``
        对应于 `numpy.pad` 提供的模式。``"smooth"`` 根据数组末端最后2个点的斜率进行平滑扩展。
        ``"antireflect"`` 和 ``"antisymmetric"`` 是 ``"reflect"`` 和 ``"symmetric"`` 的反对称版本。
        模式 `"line"` 基于沿 ``axis`` 定义的线性趋势扩展信号。
        .. versionadded:: 1.4.0
    cval : float, optional
        The constant value to use when ``mode == "constant"``.
        当 ``mode == "constant"`` 时使用的常数值。
        .. versionadded:: 1.4.0

    Returns
    -------
    y : ndarray
        The output signal array. Dimensions will be the same as `x` except
        for along `axis`, which will change size according to the `h`,
        `up`,  and `down` parameters.
        输出信号数组。维度与 `x` 相同，除了沿 `axis` 的尺寸会根据 `h`、`up` 和 `down` 参数改变。

    Notes
    -----
    The algorithm is an implementation of the block diagram shown on page 129
    of the Vaidyanathan text [1]_ (Figure 4.3-8d).
    算法是Vaidyanathan文本第129页所示块图的实现（图4.3-8d）。

    The direct approach of upsampling by factor of P with zero insertion,
    FIR filtering of length ``N``, and downsampling by factor of Q is
    O(N*Q) per output sample. The polyphase implementation used here is
    O(N/P).
    直接的上采样因子为P，零插入，长度为``N`` 的FIR滤波器，以及下采样因子为Q的方法
    每个输出样本是O(N*Q)。此处使用的多相实现是O(N/P)。

    .. versionadded:: 0.18

    References
    ----------
    .. [1] P. P. Vaidyanathan, Multirate Systems and Filter Banks,
           Prentice Hall, 1993.
           参考文献：
           [1] P. P. Vaidyanathan，多速率系统和滤波器组，Prentice Hall，1993年。

    Examples
    --------
    Simple operations:

    >>> import numpy as np
    >>> from scipy.signal import upfirdn
    >>> upfirdn([1, 1, 1], [1, 1, 1])   # FIR filter
    array([ 1.,  2.,  3.,  2.,  1.])
    >>> upfirdn([1], [1, 2, 3], 3)  # upsampling with zeros insertion
    array([ 1.,  0.,  0.,  2.,  0.,  0.,  3.])
    >>> upfirdn([1, 1, 1], [1, 2, 3], 3)  # upsampling with sample-and-hold
    array([ 1.,  1.,  1.,  2.,  2.,  2.,  3.,  3.,  3.])
    >>> upfirdn([.5, 1, .5], [1, 1, 1], 2)  # linear interpolation
    array([ 0.5,  1. ,  1. ,  1. ,  1. ,  1. ,  0.5])
    >>> upfirdn([1], np.arange(10), 1, 3)  # decimation by 3
    array([ 0.,  3.,  6.,  9.])
    >>> upfirdn([.5, 1, .5], np.arange(10), 2, 3)  # linear interp, rate 2/3
    array([ 0. ,  1. ,  2.5,  4. ,  5.5,  7. ,  8.5])

    Apply a single filter to multiple signals:

    >>> x = np.reshape(np.arange(8), (4, 2))
    >>> x
    # 创建一个二维数组 `x`，包含整数值
    array([[0, 1],
           [2, 3],
           [4, 5],
           [6, 7]])
    
    Apply along the last dimension of ``x``:
    # 定义一个滤波器 `h`
    >>> h = [1, 1]
    # 使用 `upfirdn` 函数对数组 `x` 进行处理，增加每个元素的复制次数为2倍
    >>> upfirdn(h, x, 2)
    array([[ 0.,  0.,  1.,  1.],
           [ 2.,  2.,  3.,  3.],
           [ 4.,  4.,  5.,  5.],
           [ 6.,  6.,  7.,  7.]])
    
    Apply along the 0th dimension of ``x``:
    # 使用 `upfirdn` 函数对数组 `x` 进行处理，沿着第0维度（行）进行操作，增加每个行的复制次数为2倍
    >>> upfirdn(h, x, 2, axis=0)
    array([[ 0.,  1.],
           [ 0.,  1.],
           [ 2.,  3.],
           [ 2.,  3.],
           [ 4.,  5.],
           [ 4.,  5.],
           [ 6.,  7.],
           [ 6.,  7.]])
    """
    # 将输入参数 `x` 转换为 NumPy 数组
    x = np.asarray(x)
    # 创建一个 `_UpFIRDn` 类的实例 `ufd`，用给定的滤波器 `h`、数据类型和上下采样倍数初始化
    ufd = _UpFIRDn(h, x.dtype, up, down)
    # 执行滤波器的应用操作，根据指定的轴 (`axis`)、模式 (`mode`) 和常数值 (`cval`) 进行处理
    # 这与使用 `np.apply_along_axis` 函数等效，但速度更快
    return ufd.apply_filter(x, axis, mode, cval)
```