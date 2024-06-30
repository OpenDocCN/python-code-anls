# `D:\src\scipysrc\scipy\scipy\misc\_common.py`

```
"""
Functions which are common and require SciPy Base and Level 1 SciPy
(special, linalg)
"""

# 从 scipy._lib.deprecation 模块导入 _deprecated 函数
# 从 scipy._lib._finite_differences 模块导入 _central_diff_weights 和 _derivative 函数
# 从 numpy 模块导入 array, frombuffer, load 函数
from scipy._lib.deprecation import _deprecated
from scipy._lib._finite_differences import _central_diff_weights, _derivative
from numpy import array, frombuffer, load

# 定义模块的公开接口，即可以从当前模块导入的函数列表
__all__ = ['central_diff_weights', 'derivative', 'ascent', 'face',
           'electrocardiogram']

# 使用 _deprecated 装饰器标记 central_diff_weights 函数已废弃
@_deprecated(msg="scipy.misc.central_diff_weights is deprecated in "
                 "SciPy v1.10.0; and will be completely removed in "
                 "SciPy v1.12.0. You may consider using "
                 "findiff: https://github.com/maroba/findiff or "
                 "numdifftools: https://github.com/pbrod/numdifftools")
def central_diff_weights(Np, ndiv=1):
    """
    Return weights for an Np-point central derivative.

    Assumes equally-spaced function points.

    If weights are in the vector w, then
    derivative is w[0] * f(x-ho*dx) + ... + w[-1] * f(x+h0*dx)

    .. deprecated:: 1.10.0
        `central_diff_weights` has been deprecated from
        `scipy.misc.central_diff_weights` in SciPy 1.10.0 and
        it will be completely removed in SciPy 1.12.0.
        You may consider using
        findiff: https://github.com/maroba/findiff or
        numdifftools: https://github.com/pbrod/numdifftools

    Parameters
    ----------
    Np : int
        Number of points for the central derivative.
    ndiv : int, optional
        Number of divisions. Default is 1.

    Returns
    -------
    w : ndarray
        Weights for an Np-point central derivative. Its size is `Np`.

    Notes
    -----
    Can be inaccurate for a large number of points.

    Examples
    --------
    We can calculate a derivative value of a function.

    >>> from scipy.misc import central_diff_weights
    >>> def f(x):
    ...     return 2 * x**2 + 3
    >>> x = 3.0 # derivative point
    >>> h = 0.1 # differential step
    >>> Np = 3 # point number for central derivative
    >>> weights = central_diff_weights(Np) # weights for first derivative
    >>> vals = [f(x + (i - Np/2) * h) for i in range(Np)]
    >>> sum(w * v for (w, v) in zip(weights, vals))/h
    11.79999999999998

    This value is close to the analytical solution:
    f'(x) = 4x, so f'(3) = 12

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Finite_difference

    """
    return _central_diff_weights(Np, ndiv)

# 使用 _deprecated 装饰器标记 derivative 函数已废弃
@_deprecated(msg="scipy.misc.derivative is deprecated in "
                 "SciPy v1.10.0; and will be completely removed in "
                 "SciPy v1.12.0. You may consider using "
                 "findiff: https://github.com/maroba/findiff or "
                 "numdifftools: https://github.com/pbrod/numdifftools")
def derivative(func, x0, dx=1.0, n=1, args=(), order=3):
    """
    Find the nth derivative of a function at a point.

    Given a function, use a central difference formula with spacing `dx` to
    compute the nth derivative at `x0`.
    """
    # 返回调用 _derivative 函数的结果
    return _derivative(func, x0, dx=dx, n=n, args=args, order=order)
    .. deprecated:: 1.10.0
        `derivative` has been deprecated from `scipy.misc.derivative`
        in SciPy 1.10.0 and it will be completely removed in SciPy 1.12.0.
        You may consider using
        findiff: https://github.com/maroba/findiff or
        numdifftools: https://github.com/pbrod/numdifftools


    Parameters
    ----------
    func : function
        Input function.
    x0 : float
        The point at which the nth derivative is found.
    dx : float, optional
        Spacing.
    n : int, optional
        Order of the derivative. Default is 1.
    args : tuple, optional
        Arguments
    order : int, optional
        Number of points to use, must be odd.


    Notes
    -----
    Decreasing the step size too small can result in round-off error.


    Examples
    --------
    >>> from scipy.misc import derivative
    >>> def f(x):
    ...     return x**3 + x**2
    >>> derivative(f, 1.0, dx=1e-6)
    4.9999999999217337

    """
    return _derivative(func, x0, dx, n, args, order)
# 使用装饰器标记函数已被废弃，提供相关信息作为警告消息
@_deprecated(msg="scipy.misc.ascent has been deprecated in SciPy v1.10.0;"
                 " and will be completely removed in SciPy v1.12.0. "
                 "Dataset methods have moved into the scipy.datasets "
                 "module. Use scipy.datasets.ascent instead.")
def ascent():
    """
    获取一个 8 位灰度图像，大小为 512 x 512，用于演示和测试

    此图像来源于 http://www.public-domain-image.com/people-public-domain-images-pictures/
    的 accent-to-the-top.jpg

    .. deprecated:: 1.10.0
        `ascent` 在 SciPy 1.10.0 中已被废弃，并将在 SciPy 1.12.0 中完全移除。
        数据集方法已移至 `scipy.datasets` 模块。请使用 `scipy.datasets.ascent`。

    Parameters
    ----------
    None

    Returns
    -------
    ascent : ndarray
       用于测试和演示的方便图像

    Examples
    --------
    >>> import scipy.misc
    >>> ascent = scipy.misc.ascent()
    >>> ascent.shape
    (512, 512)
    >>> ascent.max()
    255

    >>> import matplotlib.pyplot as plt
    >>> plt.gray()
    >>> plt.imshow(ascent)
    >>> plt.show()

    """
    import pickle  # 导入 pickle 模块，用于反序列化数据
    import os  # 导入 os 模块，用于操作系统相关功能
    fname = os.path.join(os.path.dirname(__file__), 'ascent.dat')  # 构建文件路径
    with open(fname, 'rb') as f:  # 打开文件，以二进制读取模式
        ascent = array(pickle.load(f))  # 从文件中加载数据并转换成数组
    return ascent  # 返回加载的图像数据


# 使用装饰器标记函数已被废弃，提供相关信息作为警告消息
@_deprecated(msg="scipy.misc.face has been deprecated in SciPy v1.10.0; "
                 "and will be completely removed in SciPy v1.12.0. "
                 "Dataset methods have moved into the scipy.datasets "
                 "module. Use scipy.datasets.face instead.")
def face(gray=False):
    """
    获取一张尺寸为 1024 x 768 的浣熊脸色彩图像。

    图像来源于 http://www.public-domain-image.com 的 raccoon-procyon-lotor.jpg

    .. deprecated:: 1.10.0
        `face` 在 SciPy 1.10.0 中已被废弃，并将在 SciPy 1.12.0 中完全移除。
        数据集方法已移至 `scipy.datasets` 模块。请使用 `scipy.datasets.face`。

    Parameters
    ----------
    gray : bool, optional
        如果为 True，则返回 8 位灰度图像；否则返回彩色图像

    Returns
    -------
    face : ndarray
        浣熊脸图像数组

    Examples
    --------
    >>> import scipy.misc
    >>> face = scipy.misc.face()
    >>> face.shape
    (768, 1024, 3)
    >>> face.max()
    255
    >>> face.dtype
    dtype('uint8')

    >>> import matplotlib.pyplot as plt
    >>> plt.gray()
    >>> plt.imshow(face)
    >>> plt.show()

    """
    import bz2  # 导入 bz2 模块，用于解压缩数据
    import os  # 导入 os 模块，用于操作系统相关功能
    with open(os.path.join(os.path.dirname(__file__), 'face.dat'), 'rb') as f:  # 打开文件，以二进制读取模式
        rawdata = f.read()  # 读取文件内容
    data = bz2.decompress(rawdata)  # 解压缩读取的数据
    face = frombuffer(data, dtype='uint8')  # 将解压缩后的数据转换为 uint8 数据类型的数组
    face.shape = (768, 1024, 3)  # 设置数组形状为 768x1024x3
    # 如果 gray 参数为 True，则将彩色图像转换为灰度图像
    if gray is True:
        # 使用加权平均方法将彩色图像转换为灰度图像
        face = (0.21 * face[:,:,0]
                + 0.71 * face[:,:,1]
                + 0.07 * face[:,:,2]).astype('uint8')
    
    # 返回处理后的图像（可能是彩色图像或灰度图像，取决于 gray 参数）
    return face
# 使用 @_deprecated 装饰器标记的函数，指示该函数已被弃用并提供了一条相关的消息说明
@_deprecated(msg="scipy.misc.electrocardiogram has been "
                 "deprecated in SciPy v1.10.0; and will "
                 "be completely removed in SciPy v1.12.0. "
                 "Dataset methods have moved into the scipy.datasets "
                 "module. Use scipy.datasets.electrocardiogram instead.")
# electrocardiogram 函数用于加载一个一维信号的示例——心电图 (ECG)
def electrocardiogram():
    """
    Load an electrocardiogram as an example for a 1-D signal.

    The returned signal is a 5 minute long electrocardiogram (ECG), a medical
    recording of the heart's electrical activity, sampled at 360 Hz.

    .. deprecated:: 1.10.0
        `electrocardiogram` has been deprecated from
        `scipy.misc.electrocardiogram` in SciPy 1.10.0 and it will be
        completely removed in SciPy 1.12.0.
        Dataset methods have moved into the `scipy.datasets` module.
        Use `scipy.datasets.electrocardiogram` instead.

    Returns
    -------
    ecg : ndarray
        The electrocardiogram in millivolt (mV) sampled at 360 Hz.

    Notes
    -----
    The provided signal is an excerpt (19:35 to 24:35) from the `record 208`_
    (lead MLII) provided by the MIT-BIH Arrhythmia Database [1]_ on
    PhysioNet [2]_. The excerpt includes noise induced artifacts, typical
    heartbeats as well as pathological changes.

    .. _record 208: https://physionet.org/physiobank/database/html/mitdbdir/records.htm#208

    .. versionadded:: 1.1.0

    References
    ----------
    .. [1] Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database.
           IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001).
           (PMID: 11446209); :doi:`10.13026/C2F305`
    .. [2] Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh,
           Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank,
           PhysioToolkit, and PhysioNet: Components of a New Research Resource
           for Complex Physiologic Signals. Circulation 101(23):e215-e220;
           :doi:`10.1161/01.CIR.101.23.e215`

    Examples
    --------
    >>> from scipy.misc import electrocardiogram
    >>> ecg = electrocardiogram()
    >>> ecg
    array([-0.245, -0.215, -0.185, ..., -0.405, -0.395, -0.385])
    >>> ecg.shape, ecg.mean(), ecg.std()
    ((108000,), -0.16510875, 0.5992473991177294)

    As stated the signal features several areas with a different morphology.
    E.g., the first few seconds show the electrical activity of a heart in
    normal sinus rhythm as seen below.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> fs = 360
    >>> time = np.arange(ecg.size) / fs
    >>> plt.plot(time, ecg)
    >>> plt.xlabel("time in s")
    >>> plt.ylabel("ECG in mV")
    >>> plt.xlim(9, 10.2)
    >>> plt.ylim(-1, 1.5)
    >>> plt.show()

    After second 16, however, the first premature ventricular contractions, also
    called extrasystoles, appear. These have a different morphology compared to
    typical heartbeats. The difference can easily be observed in the following
    plot.
    """
    # 导入必要的绘图库
    >>> plt.plot(time, ecg)
    # 设置 x 轴标签
    >>> plt.xlabel("time in s")
    # 设置 y 轴标签
    >>> plt.ylabel("ECG in mV")
    # 设置 x 轴的显示范围
    >>> plt.xlim(46.5, 50)
    # 设置 y 轴的显示范围
    >>> plt.ylim(-2, 1.5)
    # 显示绘制的图形
    >>> plt.show()

    # 在记录中有几个点存在大干扰，例如：
    >>> plt.plot(time, ecg)
    # 设置 x 轴标签
    >>> plt.xlabel("time in s")
    # 设置 y 轴标签
    >>> plt.ylabel("ECG in mV")
    # 设置 x 轴的显示范围
    >>> plt.xlim(207, 215)
    # 设置 y 轴的显示范围
    >>> plt.ylim(-2, 3.5)
    # 显示绘制的图形
    >>> plt.show()

    # 最后，分析功率谱表明大部分生物信号由低频组成。在 60 Hz 处，可以清楚地观察到由电网引起的噪声。
    
    # 导入必要的库来计算功率谱
    >>> from scipy.signal import welch
    # 计算信号的功率谱密度估计
    >>> f, Pxx = welch(ecg, fs=fs, nperseg=2048, scaling="spectrum")
    # 绘制半对数坐标下的功率谱图
    >>> plt.semilogy(f, Pxx)
    # 设置 x 轴标签
    >>> plt.xlabel("Frequency in Hz")
    # 设置 y 轴标签
    >>> plt.ylabel("Power spectrum of the ECG in mV**2")
    # 设置 x 轴的显示范围为频率范围的起始和结束
    >>> plt.xlim(f[[0, -1]])
    # 显示绘制的图形
    >>> plt.show()
    
    # 加载并处理原始数据文件中的 ECG 信号
    import os
    # 构建数据文件的完整路径
    file_path = os.path.join(os.path.dirname(__file__), "ecg.dat")
    # 使用特定方法加载数据文件
    with load(file_path) as file:
        # 将加载的原始 ECG 数据转换为整数类型（np.uint16 -> int）
        ecg = file["ecg"].astype(int)
    # 将原始 ADC 输出转换为毫伏单位：(ecg - adc_zero) / adc_gain
    ecg = (ecg - 1024) / 200.0
    # 返回处理后的 ECG 信号数据
    return ecg
```