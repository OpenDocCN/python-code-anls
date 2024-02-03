# `numpy-ml\numpy_ml\preprocessing\dsp.py`

```
# 导入必要的库
import numpy as np
from numpy.lib.stride_tricks import as_strided

# 导入自定义的窗口初始化器
from ..utils.windows import WindowInitializer

#######################################################################
#                          Signal Resampling                          #
#######################################################################

# 批量对每个图像（或类似网格状的2D信号）进行重采样到指定维度
def batch_resample(X, new_dim, mode="bilinear"):
    """
    Resample each image (or similar grid-based 2D signal) in a batch to
    `new_dim` using the specified resampling strategy.

    Parameters
    ----------
    X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_channels)`
        An input image volume
    new_dim : 2-tuple of `(out_rows, out_cols)`
        The dimension to resample each image to
    mode : {'bilinear', 'neighbor'}
        The resampling strategy to employ. Default is 'bilinear'.

    Returns
    -------
    resampled : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, out_rows, out_cols, in_channels)`
        The resampled image volume.
    """
    # 根据不同的重采样策略选择插值函数
    if mode == "bilinear":
        interpolate = bilinear_interpolate
    elif mode == "neighbor":
        interpolate = nn_interpolate_2D
    else:
        raise NotImplementedError("Unrecognized resampling mode: {}".format(mode))

    out_rows, out_cols = new_dim
    n_ex, in_rows, in_cols, n_in = X.shape

    # 计算重采样的坐标
    x = np.tile(np.linspace(0, in_cols - 2, out_cols), out_rows)
    y = np.repeat(np.linspace(0, in_rows - 2, out_rows), out_cols)

    # 对每个图像进行重采样
    resampled = []
    for i in range(n_ex):
        r = interpolate(X[i, ...], x, y)
        r = r.reshape(out_rows, out_cols, n_in)
        resampled.append(r)
    return np.dstack(resampled)


# 使用最近邻插值策略估计在`X`中坐标(x, y)处的像素值
def nn_interpolate_2D(X, x, y):
    """
    Estimates of the pixel values at the coordinates (x, y) in `X` using a
    nearest neighbor interpolation strategy.

    Notes
    -----
    # 假设当前的 `X` 中的条目反映了一个二维整数网格上等间距采样的样本

    Parameters
    ----------
    X : :py:class:`ndarray <numpy.ndarray>` of shape `(in_rows, in_cols, in_channels)`
        一个沿着 `in_rows` 行 `in_cols` 列的网格采样的输入图像。
    x : list of length `k`
        我们希望生成样本的 x 坐标列表
    y : list of length `k`
        我们希望生成样本的 y 坐标列表

    Returns
    -------
    samples : :py:class:`ndarray <numpy.ndarray>` of shape `(k, in_channels)`
        每个 (x,y) 坐标的样本，通过最近邻插值计算得到
    """
    # 将 x 和 y 坐标四舍五入到最近的整数
    nx, ny = np.around(x), np.around(y)
    # 将四舍五入后的坐标限制在合适的范围内，并转换为整数类型
    nx = np.clip(nx, 0, X.shape[1] - 1).astype(int)
    ny = np.clip(ny, 0, X.shape[0] - 1).astype(int)
    # 返回根据最近邻插值计算得到的样本
    return X[ny, nx, :]
def nn_interpolate_1D(X, t):
    """
    Estimates of the signal values at `X[t]` using a nearest neighbor
    interpolation strategy.

    Parameters
    ----------
    X : :py:class:`ndarray <numpy.ndarray>` of shape `(in_length, in_channels)`
        An input image sampled along an integer `in_length`
    t : list of length `k`
        A list of coordinates for the samples we wish to generate

    Returns
    -------
    samples : :py:class:`ndarray <numpy.ndarray>` of shape `(k, in_channels)`
        The samples for each (x,y) coordinate computed via nearest neighbor
        interpolation
    """
    # 对 t 进行四舍五入，并限制在合适的范围内，转换为整数
    nt = np.clip(np.around(t), 0, X.shape[0] - 1).astype(int)
    return X[nt, :]


def bilinear_interpolate(X, x, y):
    """
    Estimates of the pixel values at the coordinates (x, y) in `X` via bilinear
    interpolation.

    Notes
    -----
    Assumes the current entries in X reflect equally-spaced
    samples from a 2D integer grid.

    Modified from https://bit.ly/2NMb1Dr

    Parameters
    ----------
    X : :py:class:`ndarray <numpy.ndarray>` of shape `(in_rows, in_cols, in_channels)`
        An input image sampled along a grid of `in_rows` by `in_cols`.
    x : list of length `k`
        A list of x-coordinates for the samples we wish to generate
    y : list of length `k`
        A list of y-coordinates for the samples we wish to generate

    Returns
    -------
    samples : list of length `(k, in_channels)`
        The samples for each (x,y) coordinate computed via bilinear
        interpolation
    """
    # 向下取整得到 x0 和 y0 的整数部分
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    # 计算 x1 和 y1
    x1 = x0 + 1
    y1 = y0 + 1

    # 限制 x0, y0, x1, y1 在合适的范围内
    x0 = np.clip(x0, 0, X.shape[1] - 1)
    y0 = np.clip(y0, 0, X.shape[0] - 1)
    x1 = np.clip(x1, 0, X.shape[1] - 1)
    y1 = np.clip(y1, 0, X.shape[0] - 1)

    # 计算插值所需的四个像素值
    Ia = X[y0, x0, :].T
    Ib = X[y1, x0, :].T
    Ic = X[y0, x1, :].T
    Id = X[y1, x1, :].T

    # 计算插值权重
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    # 计算第四个顶点的权重
    wd = (x - x0) * (y - y0)
    
    # 返回根据权重计算得到的插值结果
    return (Ia * wa).T + (Ib * wb).T + (Ic * wc).T + (Id * wd).T
# 定义一个函数，实现一维离散余弦变换（DCT-II），默认为正交的
def DCT(frame, orthonormal=True):
    """
    A naive :math:`O(N^2)` implementation of the 1D discrete cosine transform-II
    (DCT-II).

    Notes
    -----
    For a signal :math:`\mathbf{x} = [x_1, \ldots, x_N]` consisting of `N`
    samples, the `k` th DCT coefficient, :math:`c_k`, is

    .. math::

        c_k = 2 \sum_{n=0}^{N-1} x_n \cos(\pi k (2 n + 1) / (2 N))

    where `k` ranges from :math:`0, \ldots, N-1`.

    The DCT is highly similar to the DFT -- whereas in a DFT the basis
    functions are sinusoids, in a DCT they are restricted solely to cosines. A
    signal's DCT representation tends to have more of its energy concentrated
    in a smaller number of coefficients when compared to the DFT, and is thus
    commonly used for signal compression. [1]

    .. [1] Smoother signals can be accurately approximated using fewer DFT / DCT
       coefficients, resulting in a higher compression ratio. The DCT naturally
       yields a continuous extension at the signal boundaries due its use of
       even basis functions (cosine). This in turn produces a smoother
       extension in comparison to DFT or DCT approximations, resulting in a
       higher compression.

    Parameters
    ----------
    frame : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
        A signal frame consisting of N samples
    orthonormal : bool
        Scale to ensure the coefficient vector is orthonormal. Default is True.

    Returns
    -------
    dct : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
        The discrete cosine transform of the samples in `frame`.
    """
    # 获取信号帧的长度
    N = len(frame)
    # 创建一个与输入信号帧相同形状的全零数组
    out = np.zeros_like(frame)
    # 遍历范围为 N 的循环
    for k in range(N):
        # 遍历帧中的每个元素，返回索引和元素值
        for (n, xn) in enumerate(frame):
            # 计算离散余弦变换的公式
            out[k] += xn * np.cos(np.pi * k * (2 * n + 1) / (2 * N))
        # 根据 k 的值计算缩放因子
        scale = np.sqrt(1 / (4 * N)) if k == 0 else np.sqrt(1 / (2 * N))
        # 根据是否正交归一化来调整输出值
        out[k] *= 2 * scale if orthonormal else 2
    # 返回变换后的结果
    return out
def __DCT2(frame):
    """Currently broken"""
    # 计算窗口长度
    N = len(frame)  # window length

    # 创建一个长度为N的一维数组k
    k = np.arange(N, dtype=float)
    # 创建一个N*N的二维数组F，用于计算DCT
    F = k.reshape(1, -1) * k.reshape(-1, 1)
    # 创建一个N*N的二维数组K，用于计算DCT
    K = np.divide(F, k, out=np.zeros_like(F), where=F != 0)

    # 计算DCT的余弦部分
    FC = np.cos(F * np.pi / N + K * np.pi / 2 * N)
    # 返回DCT结果
    return 2 * (FC @ frame)


def DFT(frame, positive_only=True):
    """
    A naive :math:`O(N^2)` implementation of the 1D discrete Fourier transform (DFT).

    Notes
    -----
    The Fourier transform decomposes a signal into a linear combination of
    sinusoids (ie., basis elements in the space of continuous periodic
    functions).  For a sequence :math:`\mathbf{x} = [x_1, \ldots, x_N]` of N
    evenly spaced samples, the `k` th DFT coefficient is given by:

    .. math::

        c_k = \sum_{n=0}^{N-1} x_n \exp(-2 \pi i k n / N)

    where `i` is the imaginary unit, `k` is an index ranging from `0, ..., N-1`,
    and :math:`X_k` is the complex coefficient representing the phase
    (imaginary part) and amplitude (real part) of the `k` th sinusoid in the
    DFT spectrum. The frequency of the `k` th sinusoid is :math:`(k 2 \pi / N)`
    radians per sample.

    When applied to a real-valued input, the negative frequency terms are the
    complex conjugates of the positive-frequency terms and the overall spectrum
    is symmetric (excluding the first index, which contains the zero-frequency
    / intercept term).

    Parameters
    ----------
    frame : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
        A signal frame consisting of N samples
    positive_only : bool
        Whether to only return the coefficients for the positive frequency
        terms. Default is True.

    Returns
    -------
    spectrum : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)` or `(N // 2 + 1,)` if `real_only`
        The coefficients of the frequency spectrum for `frame`, including
        imaginary components.
    """
    # 计算窗口长度
    N = len(frame)  # window length
    # 创建一个 N x N 的矩阵，矩阵元素为基向量 i 和时间步长 j 的系数
    F = np.arange(N).reshape(1, -1) * np.arange(N).reshape(-1, 1)
    # 计算每个元素的值为 exp(-j * 2 * pi * i / N)
    F = np.exp(F * (-1j * 2 * np.pi / N))

    # vdot 只能操作向量（而不是 ndarrays），因此需要显式循环遍历 F 中的每个基向量
    # 计算每个基向量与给定帧 frame 的内积，得到频谱
    spectrum = np.array([np.vdot(f, frame) for f in F])
    # 如果 positive_only 为 True，则返回频谱的前一半（包括中间值），否则返回整个频谱
    return spectrum[: (N // 2) + 1] if positive_only else spectrum
# 计算具有 N 个系数的 DFT 的频率 bin 中心
def dft_bins(N, fs=44000, positive_only=True):
    # 如果只返回正频率项的频率 bin
    if positive_only:
        freq_bins = np.linspace(0, fs / 2, 1 + N // 2, endpoint=True)
    else:
        # 计算负频率项的频率 bin
        l, r = (1 + (N - 1) / 2, (1 - N) / 2) if N % 2 else (N / 2, -N / 2)
        freq_bins = np.r_[np.arange(l), np.arange(r, 0)] * fs / N
    return freq_bins


# 计算每个帧的幅度谱（即 DFT 谱的绝对值）
def magnitude_spectrum(frames):
    # 对于 frames 中的每个帧，计算其幅度谱
    return np.vstack([np.abs(DFT(frame, positive_only=True)) for frame in frames])


# 计算信号的功率谱，假设每个帧仅包含实值
def power_spectrum(frames, scale=False):
    # 对于以帧表示的信号，计算功率谱
    # 功率谱简单地是幅度谱的平方，可能会按 FFT bins 的数量进行缩放。它衡量信号的能量在频域上的分布
    # 定义函数参数和返回值的说明
    Parameters
    ----------
    frames : :py:class:`ndarray <numpy.ndarray>` of shape `(M, N)`
        A sequence of `M` frames each consisting of `N` samples
    scale : bool
        Whether the scale by the number of DFT bins. Default is False.

    Returns
    -------
    power_spec : :py:class:`ndarray <numpy.ndarray>` of shape `(M, N // 2 + 1)`
        The power spectrum for each frame in `frames`. Only includes the
        coefficients for the positive spectrum frequencies.
    """
    # 根据是否需要缩放计算缩放因子
    scaler = frames.shape[1] // 2 + 1 if scale else 1
    # 返回每个帧的功率谱，只包括正频谱频率的系数
    return (1 / scaler) * magnitude_spectrum(frames) ** 2
# 预处理工具函数，用于将一维信号 x 转换为帧宽度为 frame_width、步长为 stride 的重叠窗口
def to_frames(x, frame_width, stride, writeable=False):
    """
    Convert a 1D signal x into overlapping windows of width `frame_width` using
    a hop length of `stride`.

    Notes
    -----
    如果 ``(len(x) - frame_width) % stride != 0``，则 x 中的一些样本将被丢弃。具体来说::

        n_dropped_frames = len(x) - frame_width - stride * (n_frames - 1)

    其中::

        n_frames = (len(x) - frame_width) // stride + 1

    该方法使用低级别的步长操作来避免创建 `x` 的额外副本。缺点是如果 ``writeable`=True``,
    修改 `frame` 输出可能导致意外行为:

        >>> out = to_frames(np.arange(6), 5, 1)
        >>> out
        array([[0, 1, 2, 3, 4],
               [1, 2, 3, 4, 5]])
        >>> out[0, 1] = 99
        >>> out
        array([[ 0, 99,  2,  3,  4],
               [99,  2,  3,  4,  5]])

    Parameters
    ----------
    x : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
        A 1D signal consisting of N samples
    frame_width : int
        The width of a single frame window in samples
    stride : int
        The hop size / number of samples advanced between consecutive frames
    writeable : bool
        If set to False, the returned array will be readonly. Otherwise it will
        be writable if `x` was. It is advisable to set this to False whenever
        possible to avoid unexpected behavior (see NB 2 above). Default is False.

    Returns
    -------
    frame: :py:class:`ndarray <numpy.ndarray>` of shape `(n_frames, frame_width)`
        The collection of overlapping frames stacked into a matrix
    """
    assert x.ndim == 1
    assert stride >= 1
    assert len(x) >= frame_width
    # 获取数组 x 中每个元素的大小（以比特为单位）
    byte = x.itemsize
    # 计算可以生成的帧数，根据数组 x 的长度、帧宽度和步长计算
    n_frames = (len(x) - frame_width) // stride + 1
    # 使用 as_strided 函数创建一个新的数组视图
    return as_strided(
        x,
        shape=(n_frames, frame_width),  # 新数组的形状为 (帧数, 帧宽度)
        strides=(byte * stride, byte),  # 新数组的步长
        writeable=writeable,  # 指定新数组是否可写
    )
# 自相关一个一维信号 `x` 与自身。

# 1维自相关的第 `k` 项是

# .. math::

#     a_k = \sum_n x_{n + k} x_n

# 注意 这是一个朴素的 :math:`O(N^2)` 实现。对于一个更快的 :math:`O(N \log N)` 方法，可以参考 [1]。

# 参考资料
# ----------
# .. [1] https://en.wikipedia.org/wiki/Autocorrelation#Efficient%computation

# 参数
# ----------
# x : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
#     由 N 个样本组成的1维信号

# 返回
# -------
# auto : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
#     `x` 与自身的自相关
def autocorrelate1D(x):
    N = len(x)
    auto = np.zeros(N)
    for k in range(N):
        for n in range(N - k):
            auto[k] += x[n + k] * x[n]
    return auto


#######################################################################
#                               Filters                               #
#######################################################################


# 预加重，增加高频带的幅度 + 减少低频带的幅度。

# 预加重滤波是（曾经是？）语音处理中常见的变换，其中高频率在信号消除歧义时更有用。

# .. math::

#     \\text{preemphasis}( x_t ) = x_t - \\alpha x_{t-1}

# 参数
# ----------
# x : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
#     由 `N` 个样本组成的1维信号
# alpha : float in [0, 1)
#     预加重系数。值为0表示无滤波

# 返回
# -------
# out : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
#     滤波后的信号
def preemphasis(x, alpha):
    return np.concatenate([x[:1], x[1:] - alpha * x[:-1])


# 倒谱提升
    # 在梅尔频率域中应用简单的正弦滤波器
    
    # 备注
    # 倒谱提升有助于平滑频谱包络并减弱较高MFCC系数的幅度，同时保持其他系数不变。滤波器函数为：
    # lifter( x_n ) = x_n * (1 + (D * sin(pi * n / D) / 2))
    
    # 参数
    # mfccs: 形状为(G, C)的Mel倒谱系数矩阵。行对应帧，列对应倒谱系数
    # D: 在[0, +∞]范围内的整数。滤波器系数。0表示无滤波，较大的值表示更大程度的平滑
    
    # 返回
    # out: 形状为(G, C)的倒谱提升后的MFCC系数
    def sinusoidal_filter(mfccs, D):
        # 如果D为0，则返回原始MFCC系数
        if D == 0:
            return mfccs
        # 生成n，范围为MFCC系数的列数
        n = np.arange(mfccs.shape[1])
        # 返回倒谱提升后的MFCC系数
        return mfccs * (1 + (D / 2) * np.sin(np.pi * n / D))
# 计算信号 `x` 的 Mel 频谱图
def mel_spectrogram(
    x,
    window_duration=0.025,  # 每个帧/窗口的持续时间（秒）。默认为0.025。
    stride_duration=0.01,  # 连续窗口之间的跳跃持续时间（秒）。默认为0.01。
    mean_normalize=True,  # 是否从最终滤波器值中减去系数均值以提高信噪比。默认为True。
    window="hamming",  # 在FFT之前应用的窗函数。默认为'hamming'。
    n_filters=20,  # 包含在滤波器组中的 Mel 滤波器数量。默认为20。
    center=True,  # 信号的第 `k` 帧是否应该在索引 `x[k * stride_len]` 处*开始*（center = False）或在 `x[k * stride_len]` 处*居中*（center = True）。默认为False。
    alpha=0.95,  # 预加重滤波器的系数。值为0表示无滤波。默认为0.95。
    fs=44000,  # 信号的采样率/频率。默认为44000。
):
    """
    将 Mel 滤波器组应用于信号 `x` 的功率谱。

    Notes
    -----
    Mel 频谱图是对帧化和窗口化信号的功率谱在 Mel 滤波器组提供的基础上的投影。

    Parameters
    ----------
    x : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
        由 N 个样本组成的 1D 信号
    window_duration : float
        每个帧/窗口的持续时间（秒）。默认为0.025。
    stride_duration : float
        连续窗口之间的跳跃持续时间（秒）。默认为0.01。
    mean_normalize : bool
        是否从最终滤波器值中减去系数均值以提高信噪比。默认为True。
    window : {'hamming', 'hann', 'blackman_harris'}
        在FFT之前应用于信号的窗函数。默认为'hamming'。
    n_filters : int
        包含在滤波器组中的 Mel 滤波器数量。默认为20。
    center : bool
        信号的第 `k` 帧是否应该在索引 `x[k * stride_len]` 处*开始*（center = False）或在 `x[k * stride_len]` 处*居中*（center = True）。默认为False。
    alpha : float in [0, 1)
        预加重滤波器的系数。值为0表示无滤波。默认为0.95。
    fs : int
        信号的采样率/频率。默认为44000。

    Returns
    -------
    filter_energies : :py:class:`ndarray <numpy.ndarray>` of shape `(G, n_filters)`
        Mel 滤波器组中每个滤波器的（可能是均值归一化的）功率（即 Mel 频谱图）。行对应帧，列对应滤波器
    """
    # 每帧信号的总能量
    energy_per_frame : :py:class:`ndarray <numpy.ndarray>` of shape `(G,)`
        The total energy in each frame of the signal
    """
    # 机器精度
    eps = np.finfo(float).eps
    # 初始化窗口函数
    window_fn = WindowInitializer()(window)

    # 计算步长
    stride = round(stride_duration * fs)
    # 计算帧宽度
    frame_width = round(window_duration * fs)
    N = frame_width

    # 对原始信号应用预加重滤波器
    x = preemphasis(x, alpha)

    # 将信号转换为重叠帧并应用窗口函数
    x = np.pad(x, N // 2, "reflect") if center else x
    frames = to_frames(x, frame_width, stride, fs)

    # 生成窗口矩阵
    window = np.tile(window_fn(frame_width), (frames.shape[0], 1))
    frames = frames * window

    # 计算功率谱
    power_spec = power_spectrum(frames)
    energy_per_frame = np.sum(power_spec, axis=1)
    energy_per_frame[energy_per_frame == 0] = eps

    # 计算 Mel 滤波器组中每个滤波器的功率
    fbank = mel_filterbank(N, n_filters=n_filters, fs=fs)
    filter_energies = power_spec @ fbank.T
    filter_energies -= np.mean(filter_energies, axis=0) if mean_normalize else 0
    filter_energies[filter_energies == 0] = eps
    return filter_energies, energy_per_frame
# 计算信号的 Mel 频率倒谱系数（MFCC）

def mfcc(
    x,
    fs=44000,
    n_mfccs=13,
    alpha=0.95,
    center=True,
    n_filters=20,
    window="hann",
    normalize=True,
    lifter_coef=22,
    stride_duration=0.01,
    window_duration=0.025,
    replace_intercept=True,
):
    """
    计算信号的 Mel 频率倒谱系数（MFCC）。

    注意
    -----
    计算 MFCC 特征分为以下几个阶段：

        1. 将信号转换为重叠帧并应用窗口函数
        2. 计算每帧的功率谱
        3. 将 Mel 滤波器组应用于功率谱以获得 Mel 滤波器组功率
        4. 对每帧的 Mel 滤波器组功率取对数
        5. 对对数滤波器组能量进行离散余弦变换（DCT），并仅保留前 k 个系数以进一步降低维度

    MFCC 是在 HMM-GMM 自动语音识别（ASR）系统的背景下开发的，可用于提供某种程度上与说话者/音高无关的音素表示。

    参数
    ----------
    x : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
        由 N 个样本组成的 1D 信号
    fs : int
        信号的采样率/频率。默认值为 44000。
    n_mfccs : int
        要返回的倒谱系数的数量（包括截距系数）。默认值为 13。
    alpha : float in [0, 1)
        预加重系数。值为 0 表示无滤波。默认值为 0.95。
    center : bool
        信号的第 k 帧是否应该 *从* 索引 ``x[k * stride_len]`` 开始（center = False），
        还是 *居中* 在 ``x[k * stride_len]`` 处（center = True）。默认值为 True。
    # 设置用于 Mel 滤波器组的滤波器数量，默认为 20
    n_filters : int
        The number of filters to include in the Mel filterbank. Default is 20.
    # 是否对 MFCC 值进行均值归一化，默认为 True
    normalize : bool
        Whether to mean-normalize the MFCC values. Default is True.
    # 倒谱滤波器系数，0 表示不进行滤波，较大的值表示更多的平滑处理，默认为 22
    lifter_coef : int in :math:[0, + \infty]`
        The cepstral filter coefficient. 0 corresponds to no filtering, larger
        values correspond to greater amounts of smoothing. Default is 22.
    # 应用于信号在进行 DFT 之前的窗函数，默认为 'hann'
    window : {'hamming', 'hann', 'blackman_harris'}
        The windowing function to apply to the signal before taking the DFT.
        Default is 'hann'.
    # 连续窗口之间的跳跃持续时间（以秒为单位），默认为 0.01
    stride_duration : float
        The duration of the hop between consecutive windows (in seconds).
        Default is 0.01.
    # 每个帧/窗口的持续时间（以秒为单位），默认为 0.025
    window_duration : float
        The duration of each frame / window (in seconds). Default is 0.025.
    # 是否用总帧能量的对数替换第一个 MFCC 系数（截距项），默认为 True
    replace_intercept : bool
        Replace the first MFCC coefficient (the intercept term) with the
        log of the total frame energy instead. Default is True.

    Returns
    -------
    # Mel 频率倒谱系数的矩阵，行对应帧，列对应倒谱系数
    mfccs : :py:class:`ndarray <numpy.ndarray>` of shape `(G, C)`
        Matrix of Mel-frequency cepstral coefficients. Rows correspond to
        frames, columns to cepstral coefficients
    """
    # 将（帧化 + 窗函数处理后的）`x` 的功率谱映射到 Mel 频率刻度上
    filter_energies, frame_energies = mel_spectrogram(
        x=x,
        fs=fs,
        alpha=alpha,
        center=center,
        window=window,
        n_filters=n_filters,
        mean_normalize=False,
        window_duration=window_duration,
        stride_duration=stride_duration,
    )

    # 计算滤波器能量的对数值
    log_energies = 10 * np.log10(filter_energies)

    # 对对数 Mel 系数执行 DCT 以进一步降低数据维度
    # 早期的 DCT 系数将捕获大部分数据，允许我们丢弃 > n_mfccs 的系数
    mfccs = np.array([DCT(frame) for frame in log_energies])[:, :n_mfccs]

    # 对 MFCC 应用倒谱提升
    mfccs = cepstral_lifter(mfccs, D=lifter_coef)
    # 如果需要对 MFCC 进行归一化，则减去每列的均值
    mfccs -= np.mean(mfccs, axis=0) if normalize else 0

    # 如果需要替换截距项
    if replace_intercept:
        # 第0个 MFCC 系数不提供关于频谱的信息；
        # 用帧能量的对数替换它，得到更有信息量的特征
        mfccs[:, 0] = np.log(frame_energies)
    # 返回处理后的 MFCC 特征
    return mfccs
# 将 mel 频率表示的信号转换为 Hz 频率表示
def mel2hz(mel, formula="htk"):
    # 检查输入的 formula 参数是否合法
    fstr = "formula must be either 'htk' or 'slaney' but got '{}'"
    assert formula in ["htk", "slaney"], fstr.format(formula)
    
    # 根据 formula 参数选择不同的转换公式
    if formula == "htk":
        return 700 * (10 ** (mel / 2595) - 1)
    
    # 如果 formula 参数不是 'htk'，则抛出未实现的错误
    raise NotImplementedError("slaney")


# 将 Hz 频率表示的信号转换为 mel 频率表示
def hz2mel(hz, formula="htk"):
    # 检查输入的 formula 参数是否合法
    fstr = "formula must be either 'htk' or 'slaney' but got '{}'"
    assert formula in ["htk", "slaney"], fstr.format(formula)

    # 根据 formula 参数选择不同的转换公式
    if formula == "htk":
        return 2595 * np.log10(1 + hz / 700)
    
    # 如果 formula 参数不是 'htk'，则抛出未实现的错误
    raise NotImplementedError("slaney")


# 生成 mel 滤波器组
def mel_filterbank(
    N, n_filters=20, fs=44000, min_freq=0, max_freq=None, normalize=True
):
    # 计算 Mel 滤波器组并返回相应的转换矩阵

    # Mel 比例是一种感知比例，旨在模拟人耳的工作方式。在 Mel 比例上，被听众认为在感知/心理距离上相等的音高在 Mel 比例上具有相等的距离。实际上，这对应于在低频率具有更高分辨率，在高频率（> 500 Hz）具有较低分辨率的比例。

    # Mel 滤波器组中的每个滤波器都是三角形的，在其中心具有响应为 1，在两侧具有线性衰减，直到达到下一个相邻滤波器的中心频率。

    # 此实现基于（出色的）LibROSA软件包中的代码。

    # 参考资料
    # McFee 等人（2015年）。"librosa: Python中的音频和音乐信号分析"，*第14届Python科学会议论文集*
    # https://librosa.github.io

    # 参数
    # N : int
    #     DFT 布尔数
    # n_filters : int
    #     要包含在滤波器组中的 Mel 滤波器数量。默认为 20。
    # min_freq : int
    #     最小滤波器频率（以 Hz 为单位）。默认为 0。
    # max_freq : int
    #     最大滤波器频率（以 Hz 为单位）。默认为 0。
    # fs : int
    #     信号的采样率/频率。默认为 44000。
    # normalize : bool
    #     如果为 True，则按其在 Mel 空间中的面积缩放 Mel 滤波器权重。默认为 True。

    # 返回
    # fbank : :py:class:`ndarray <numpy.ndarray>` of shape `(n_filters, N // 2 + 1)`
    #     Mel 滤波器组转换矩阵。行对应滤波器，列对应 DFT 布尔数。
    """
    # 如果 max_freq 为 None，则将其设置为 fs 的一半
    max_freq = fs / 2 if max_freq is None else max_freq
    # 将最小频率和最大频率转换为 Mel 单位
    min_mel, max_mel = hz2mel(min_freq), hz2mel(max_freq)

    # 创建一个形状为 (n_filters, N // 2 + 1) 的全零数组
    fbank = np.zeros((n_filters, N // 2 + 1))

    # 在 Mel 比例上均匀分布的值，转换回 Hz 单位
    # 根据最小和最大梅尔频率以及滤波器数量计算梅尔频率的范围
    mel_bins = mel2hz(np.linspace(min_mel, max_mel, n_filters + 2))

    # 计算DFT频率区间的中心
    hz_bins = dft_bins(N, fs)

    # 计算相邻梅尔频率之间的间距
    mel_spacing = np.diff(mel_bins)

    # 计算梅尔频率和DFT频率之间的差值
    ramps = mel_bins.reshape(-1, 1) - hz_bins.reshape(1, -1)
    for i in range(n_filters):
        # 计算跨越频率区间左右两侧的滤波器值...
        left = -ramps[i] / mel_spacing[i]
        right = ramps[i + 2] / mel_spacing[i + 1]

        # ...并在它们穿过x轴时将它们设为零
        fbank[i] = np.maximum(0, np.minimum(left, right))

    # 如果需要进行归一化
    if normalize:
        # 计算能量归一化系数
        energy_norm = 2.0 / (mel_bins[2 : n_filters + 2] - mel_bins[:n_filters])
        # 对滤波器组进行能量归一化
        fbank *= energy_norm[:, np.newaxis]

    # 返回滤波器组
    return fbank
```