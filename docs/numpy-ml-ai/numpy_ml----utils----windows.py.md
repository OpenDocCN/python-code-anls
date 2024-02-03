# `numpy-ml\numpy_ml\utils\windows.py`

```py
import numpy as np  # 导入 NumPy 库


def blackman_harris(window_len, symmetric=False):
    """
    The Blackman-Harris window.

    Notes
    -----
    The Blackman-Harris window is an instance of the more general class of
    cosine-sum windows where `K=3`. Additional coefficients extend the Hamming
    window to further minimize the magnitude of the nearest side-lobe in the
    frequency response.

    .. math::
        \\text{bh}(n) = a_0 - a_1 \cos\left(\\frac{2 \pi n}{N}\\right) +
            a_2 \cos\left(\\frac{4 \pi n }{N}\\right) -
                a_3 \cos\left(\\frac{6 \pi n}{N}\\right)

    where `N` = `window_len` - 1, :math:`a_0` = 0.35875, :math:`a_1` = 0.48829,
    :math:`a_2` = 0.14128, and :math:`a_3` = 0.01168.

    Parameters
    ----------
    window_len : int
        The length of the window in samples. Should be equal to the
        `frame_width` if applying to a windowed signal.
    symmetric : bool
        If False, create a 'periodic' window that can be used in with an FFT /
        in spectral analysis.  If True, generate a symmetric window that can be
        used in, e.g., filter design. Default is False.

    Returns
    -------
    window : :py:class:`ndarray <numpy.ndarray>` of shape `(window_len,)`
        The window
    """
    return generalized_cosine(  # 调用 generalized_cosine 函数
        window_len, [0.35875, 0.48829, 0.14128, 0.01168], symmetric
    )


def hamming(window_len, symmetric=False):
    """
    The Hamming window.

    Notes
    -----
    The Hamming window is an instance of the more general class of cosine-sum
    windows where `K=1` and :math:`a_0 = 0.54`. Coefficients selected to
    minimize the magnitude of the nearest side-lobe in the frequency response.

    .. math::

        \\text{hamming}(n) = 0.54 -
            0.46 \cos\left(\\frac{2 \pi n}{\\text{window_len} - 1}\\right)

    Parameters
    ----------
    window_len : int
        The length of the window in samples. Should be equal to the
        `frame_width` if applying to a windowed signal.
    symmetric : bool
        # 定义一个布尔型参数 symmetric，用于指定是否生成对称窗口
        If False, create a 'periodic' window that can be used in with an FFT /
        in spectral analysis.  If True, generate a symmetric window that can be
        used in, e.g., filter design. Default is False.
        # 如果 symmetric 为 False，则创建一个周期性窗口，可用于 FFT 或频谱分析；如果为 True，则生成对称窗口，可用于滤波器设计，默认为 False

    Returns
    -------
    window : :py:class:`ndarray <numpy.ndarray>` of shape `(window_len,)`
        # 返回值为一个形状为 (window_len,) 的 ndarray 类型的窗口数组
        The window
    """
    # 调用 generalized_cosine 函数，传入窗口长度和对应的参数列表 [0.54, 1 - 0.54]，以及 symmetric 参数
    return generalized_cosine(window_len, [0.54, 1 - 0.54], symmetric)
# 定义汉宁窗口函数
def hann(window_len, symmetric=False):
    """
    The Hann window.

    Notes
    -----
    汉宁窗口是余弦和窗口的一个特例，其中 `K=1` 和 :math:`a_0` = 0.5。与 Hamming 窗口不同，汉宁窗口的端点接触到零。

    .. math::

        \\text{hann}(n) = 0.5 - 0.5 \cos\left(\\frac{2 \pi n}{\\text{window_len} - 1}\\right)

    Parameters
    ----------
    window_len : int
        窗口的长度（以样本为单位）。如果应用于窗口信号，则应等于 `frame_width`。
    symmetric : bool
        如果为 False，则创建一个可以在 FFT / 频谱分析中使用的“周期性”窗口。如果为 True，则生成一个可以在滤波器设计等方面使用的对称窗口。默认为 False。

    Returns
    -------
    window : :py:class:`ndarray <numpy.ndarray>` of shape `(window_len,)`
        窗口
    """
    return generalized_cosine(window_len, [0.5, 0.5], symmetric)


# 定义广义余弦窗口函数
def generalized_cosine(window_len, coefs, symmetric=False):
    """
    The generalized cosine family of window functions.

    Notes
    -----
    广义余弦窗口是余弦项的简单加权和。

    对于 :math:`n \in \{0, \ldots, \\text{window_len} \}`:

    .. math::

        \\text{GCW}(n) = \sum_{k=0}^K (-1)^k a_k \cos\left(\\frac{2 \pi k n}{\\text{window_len}}\\right)

    Parameters
    ----------
    window_len : int
        窗口的长度（以样本为单位）。如果应用于窗口信号，则应等于 `frame_width`。
    coefs: list of floats
        :math:`a_k` 系数值
    symmetric : bool
        如果为 False，则创建一个可以在 FFT / 频谱分析中使用的“周期性”窗口。如果为 True，则生成一个可以在滤波器设计等方面使用的对称窗口。默认为 False.

    Returns
    -------

    """
    window : :py:class:`ndarray <numpy.ndarray>` of shape `(window_len,)`
        The window
    """
    # 如果不是对称窗口，窗口长度加1
    window_len += 1 if not symmetric else 0
    # 生成等间距的数组，范围为 -π 到 π，长度为 window_len
    entries = np.linspace(-np.pi, np.pi, window_len)  # (-1)^k * 2pi*n / window_len
    # 根据给定的系数生成窗口
    window = np.sum([ak * np.cos(k * entries) for k, ak in enumerate(coefs)], axis=0)
    # 如果不是对称窗口，去掉最后一个元素
    return window[:-1] if not symmetric else window
# 定义一个 WindowInitializer 类
class WindowInitializer:
    # 定义 __call__ 方法，用于初始化窗口函数
    def __call__(self, window):
        # 如果窗口函数为 hamming，则返回 hamming 函数
        if window == "hamming":
            return hamming
        # 如果窗口函数为 blackman_harris，则返回 blackman_harris 函数
        elif window == "blackman_harris":
            return blackman_harris
        # 如果窗口函数为 hann，则返回 hann 函数
        elif window == "hann":
            return hann
        # 如果窗口函数为 generalized_cosine，则返回 generalized_cosine 函数
        elif window == "generalized_cosine":
            return generalized_cosine
        # 如果窗口函数不在以上几种情况中，则抛出 NotImplementedError 异常
        else:
            raise NotImplementedError("{}".format(window))
```