# `D:\src\scipysrc\scipy\scipy\signal\__init__.py`

```
"""
=======================================
Signal processing (:mod:`scipy.signal`)
=======================================

Convolution
===========

.. autosummary::
   :toctree: generated/

   convolve           -- N-D convolution.
                        多维卷积函数。
   correlate          -- N-D correlation.
                        多维相关函数。
   fftconvolve        -- N-D convolution using the FFT.
                        使用 FFT 进行多维卷积。
   oaconvolve         -- N-D convolution using the overlap-add method.
                        使用重叠-添加方法进行多维卷积。
   convolve2d         -- 2-D convolution (more options).
                        二维卷积函数（更多选项）。
   correlate2d        -- 2-D correlation (more options).
                        二维相关函数（更多选项）。
   sepfir2d           -- Convolve with a 2-D separable FIR filter.
                        使用二维可分离 FIR 滤波器进行卷积。
   choose_conv_method -- Chooses faster of FFT and direct convolution methods.
                        选择更快的 FFT 或直接卷积方法。
   correlation_lags   -- Determines lag indices for 1D cross-correlation.
                        确定一维交叉相关的滞后索引。

B-splines
=========

.. autosummary::
   :toctree: generated/

   gauss_spline   -- Gaussian approximation to the B-spline basis function.
                     B-样条基函数的高斯近似。
   cspline1d      -- Coefficients for 1-D cubic (3rd order) B-spline.
                     一维三次（3阶）B-样条的系数。
   qspline1d      -- Coefficients for 1-D quadratic (2nd order) B-spline.
                     一维二次（2阶）B-样条的系数。
   cspline2d      -- Coefficients for 2-D cubic (3rd order) B-spline.
                     二维三次（3阶）B-样条的系数。
   qspline2d      -- Coefficients for 2-D quadratic (2nd order) B-spline.
                     二维二次（2阶）B-样条的系数。
   cspline1d_eval -- Evaluate a cubic spline at the given points.
                     在给定点处评估三次样条。
   qspline1d_eval -- Evaluate a quadratic spline at the given points.
                     在给定点处评估二次样条。
   spline_filter  -- Smoothing spline (cubic) filtering of a rank-2 array.
                     对二维数组进行平滑样条（三次）滤波。

Filtering
=========

.. autosummary::
   :toctree: generated/

   order_filter  -- N-D order filter.
                    多维次序滤波器。
   medfilt       -- N-D median filter.
                    多维中值滤波器。
   medfilt2d     -- 2-D median filter (faster).
                    二维中值滤波器（更快）。
   wiener        -- N-D Wiener filter.
                    多维维纳滤波器。

   symiirorder1  -- 2nd-order IIR filter (cascade of first-order systems).
                    二阶IIR滤波器（一级系统级联）。
   symiirorder2  -- 4th-order IIR filter (cascade of second-order systems).
                    四阶IIR滤波器（二级系统级联）。
   lfilter       -- 1-D FIR and IIR digital linear filtering.
                    一维FIR和IIR数字线性滤波器。
   lfiltic       -- Construct initial conditions for `lfilter`.
                    为`lfilter`构造初始条件。
   lfilter_zi    -- Compute an initial state zi for the lfilter function that
                    corresponds to the steady state of the step response.
                    计算`lfilter`函数的初始状态zi，对应于阶跃响应的稳态。
   filtfilt      -- A forward-backward filter.
                    前向-后向滤波器。

   deconvolve    -- 1-D deconvolution using lfilter.
                    使用`lfilter`进行一维反卷积。

   sosfilt       -- 1-D IIR digital linear filtering using
                    a second-order sections filter representation.
                    使用二阶节段滤波器表示的一维IIR数字线性滤波器。
   sosfilt_zi    -- Compute an initial state zi for the sosfilt function that
                    corresponds to the steady state of the step response.
                    计算`lfilter`函数的初始状态zi，对应于阶跃响应的稳态。
   sosfiltfilt   -- A forward-backward filter for second-order sections.
                    二阶节段的前向-后向滤波器。
   hilbert       -- Compute 1-D analytic signal, using the Hilbert transform.
                    使用Hilbert变换计算一维解析信号。
   hilbert2      -- Compute 2-D analytic signal, using the Hilbert transform.
                    使用Hilbert变换计算二维解析信号。

   decimate      -- Downsample a signal.
                    对信号进行下采样。
   detrend       -- Remove linear and/or constant trends from data.
                    从数据中去除线性和/或常数趋势。
   resample      -- Resample using Fourier method.
                    使用傅里叶方法重新采样。
   resample_poly -- Resample using polyphase filtering method.
                    使用多相滤波方法重新采样。
   upfirdn       -- Upsample, apply FIR filter, downsample.
                    上采样，应用FIR滤波器，下采样。

"""
# 滤波器设计

# bilinear函数
# 使用双线性变换从模拟滤波器设计数字滤波器。
bilinear      -- Digital filter from an analog filter using
                -- the bilinear transform.

# bilinear_zpk函数
# 使用双线性变换从模拟滤波器设计数字滤波器（针对零极点增益）。
bilinear_zpk  -- Digital filter from an analog filter using
                -- the bilinear transform.

# findfreqs函数
# 查找用于计算滤波器响应的频率数组。
findfreqs     -- Find array of frequencies for computing filter response.

# firls函数
# 使用最小二乘误差最小化设计FIR滤波器。
firls         -- FIR filter design using least-squares error minimization.

# firwin函数
# 使用窗函数设计FIR滤波器，指定通带和阻带的频率响应。
firwin        -- Windowed FIR filter design, with frequency response
                -- defined as pass and stop bands.

# firwin2函数
# 使用窗函数设计FIR滤波器，指定任意频率响应。
firwin2       -- Windowed FIR filter design, with arbitrary frequency
                -- response.

# freqs函数
# 从传递函数系数计算模拟滤波器的频率响应。
freqs         -- Analog filter frequency response from TF coefficients.

# freqs_zpk函数
# 从零极点增益系数计算模拟滤波器的频率响应。
freqs_zpk     -- Analog filter frequency response from ZPK coefficients.

# freqz函数
# 从传递函数系数计算数字滤波器的频率响应。
freqz         -- Digital filter frequency response from TF coefficients.

# freqz_zpk函数
# 从零极点增益系数计算数字滤波器的频率响应。
freqz_zpk     -- Digital filter frequency response from ZPK coefficients.

# sosfreqz函数
# 为SOS格式的滤波器计算数字滤波器的频率响应。
sosfreqz      -- Digital filter frequency response for SOS format filter.

# gammatone函数
# 设计FIR和IIR的gamma通带滤波器。
gammatone     -- FIR and IIR gammatone filter design.

# group_delay函数
# 计算数字滤波器的群延迟。
group_delay   -- Digital filter group delay.

# iirdesign函数
# 给定频带和增益，设计IIR滤波器。
iirdesign     -- IIR filter design given bands and gains.

# iirfilter函数
# 给定阶数和临界频率，设计IIR滤波器。
iirfilter     -- IIR filter design given order and critical frequencies.

# kaiser_atten函数
# 计算Kaiser FIR滤波器的衰减，给定阶数和过渡区域的宽度。
kaiser_atten  -- Compute the attenuation of a Kaiser FIR filter, given
                -- the number of taps and the transition width at
                -- discontinuities in the frequency response.

# kaiser_beta函数
# 计算Kaiser参数beta，给定期望的FIR滤波器衰减。
kaiser_beta   -- Compute the Kaiser parameter beta, given the desired
                -- FIR filter attenuation.

# kaiserord函数
# 设计Kaiser窗口以限制波动和过渡区域的宽度。
kaiserord     -- Design a Kaiser window to limit ripple and width of
                -- transition region.

# minimum_phase函数
# 将线性相位FIR滤波器转换为最小相位。
minimum_phase -- Convert a linear phase FIR filter to minimum phase.

# savgol_coeffs函数
# 计算Savitzky-Golay滤波器的FIR滤波器系数。
savgol_coeffs -- Compute the FIR filter coefficients for a Savitzky-Golay
                -- filter.

# remez函数
# 最优FIR滤波器设计。
remez         -- Optimal FIR filter design.

# unique_roots函数
# 唯一根及其重数。
unique_roots  -- Unique roots and their multiplicities.

# residue函数
# b(s) / a(s)的部分分式展开。
residue       -- Partial fraction expansion of b(s) / a(s).

# residuez函数
# b(z) / a(z)的部分分式展开。
residuez      -- Partial fraction expansion of b(z) / a(z).

# invres函数
# 模拟滤波器的逆部分分式展开。
invres        -- Inverse partial fraction expansion for analog filter.

# invresz函数
# 数字滤波器的逆部分分式展开。
invresz       -- Inverse partial fraction expansion for digital filter.

# BadCoefficients类
# 关于滤波器系数不良条件的警告。
BadCoefficients  -- Warning on badly conditioned filter coefficients.

# 较低级别的滤波器设计函数
# （以下未给出详细注释，可类似上述进行注释）
# 检查状态空间矩阵，确保其为二阶
abcd_normalize -- Check state-space matrices and ensure they are rank-2.

# 带阻带通滤波器的目标函数，用于最小化阶数
band_stop_obj  -- Band Stop Objective Function for order minimization.

# 返回贝塞尔滤波器的模拟原型的(z,p,k)表示
besselap       -- Return (z,p,k) for analog prototype of Bessel filter.

# 返回巴特沃斯滤波器的模拟原型的(z,p,k)表示
buttap         -- Return (z,p,k) for analog prototype of Butterworth filter.

# 返回第一类切比雪夫滤波器的模拟原型的(z,p,k)表示
cheb1ap        -- Return (z,p,k) for type I Chebyshev filter.

# 返回第二类切比雪夫滤波器的模拟原型的(z,p,k)表示
cheb2ap        -- Return (z,p,k) for type II Chebyshev filter.

# 返回椭圆滤波器的模拟原型的(z,p,k)表示
ellipap        -- Return (z,p,k) for analog prototype of elliptic filter.

# 将低通滤波器原型转换为带通滤波器
lp2bp          -- Transform a lowpass filter prototype to a bandpass filter.

# 将低通滤波器原型转换为带通滤波器(zpk形式)
lp2bp_zpk      -- Transform a lowpass filter prototype to a bandpass filter.

# 将低通滤波器原型转换为带阻滤波器
lp2bs          -- Transform a lowpass filter prototype to a bandstop filter.

# 将低通滤波器原型转换为带阻滤波器(zpk形式)
lp2bs_zpk      -- Transform a lowpass filter prototype to a bandstop filter.

# 将低通滤波器原型转换为高通滤波器
lp2hp          -- Transform a lowpass filter prototype to a highpass filter.

# 将低通滤波器原型转换为高通滤波器(zpk形式)
lp2hp_zpk      -- Transform a lowpass filter prototype to a highpass filter.

# 将低通滤波器原型转换为低通滤波器
lp2lp          -- Transform a lowpass filter prototype to a lowpass filter.

# 将低通滤波器原型转换为低通滤波器(zpk形式)
lp2lp_zpk      -- Transform a lowpass filter prototype to a lowpass filter.

# 归一化传递函数的多项式表示
normalize      -- Normalize polynomial representation of a transfer function.
# 导入所需的库和模块
.. autosummary::
   :toctree: generated/

   dlti             -- Discrete-time linear time invariant system base class.
   StateSpace       -- Linear time invariant system in state space form.
   TransferFunction -- Linear time invariant system in transfer function form.
   ZerosPolesGain   -- Linear time invariant system in zeros, poles, gain form.
   dlsim            -- Simulation of output to a discrete-time linear system.
   dimpulse         -- Impulse response of a discrete-time LTI system.
   dstep            -- Step response of a discrete-time LTI system.
   dfreqresp        -- Frequency response of a discrete-time LTI system.
   dbode            -- Bode magnitude and phase data (discrete-time LTI).

# LTI 表示
===================

# 自动汇总
.. autosummary::
   :toctree: generated/

   tf2zpk        -- Transfer function to zero-pole-gain.
   tf2sos        -- Transfer function to second-order sections.
   tf2ss         -- Transfer function to state-space.
   zpk2tf        -- Zero-pole-gain to transfer function.
   zpk2sos       -- Zero-pole-gain to second-order sections.
   zpk2ss        -- Zero-pole-gain to state-space.
   ss2tf         -- State-pace to transfer function.
   ss2zpk        -- State-space to pole-zero-gain.
   sos2zpk       -- Second-order sections to zero-pole-gain.
   sos2tf        -- Second-order sections to transfer function.
   cont2discrete -- Continuous-time to discrete-time LTI conversion.
   place_poles   -- Pole placement.

# 波形
=========

# 自动汇总
.. autosummary::
   :toctree: generated/

   chirp        -- Frequency swept cosine signal, with several freq functions.
   gausspulse   -- Gaussian modulated sinusoid.
   max_len_seq  -- Maximum length sequence.
   sawtooth     -- Periodic sawtooth.
   square       -- Square wave.
   sweep_poly   -- Frequency swept cosine signal; freq is arbitrary polynomial.
   unit_impulse -- Discrete unit impulse.

# 窗口函数
================

# 对于窗口函数，请查看 `scipy.signal.windows` 命名空间。

# 在 `scipy.signal` 命名空间中，有一个方便的函数来获取这些窗口函数的名称：
.. autosummary::
   :toctree: generated/

   get_window -- Return a window of a given length and type.

# 峰值查找
============

# 自动汇总
.. autosummary::
   :toctree: generated/

   argrelmin        -- Calculate the relative minima of data.
   argrelmax        -- Calculate the relative maxima of data.
   argrelextrema    -- Calculate the relative extrema of data.
   find_peaks       -- Find a subset of peaks inside a signal.
   find_peaks_cwt   -- Find peaks in a 1-D array with wavelet transformation.
   peak_prominences -- Calculate the prominence of each peak in a signal.
   peak_widths      -- Calculate the width of each peak in a signal.

# 频谱分析
=================
# 导入_scipy.signal命名空间中的功能模块和函数

from . import _sigtools, windows
# 导入内部模块_sigtools和windows，这些模块提供了信号处理和窗口函数的实现

from ._waveforms import *
# 导入_waveforms模块中的所有内容，用于波形生成

from ._max_len_seq import max_len_seq
# 导入_max_len_seq模块中的max_len_seq函数，用于生成最大长度序列

from ._upfirdn import upfirdn
# 导入_upfirdn模块中的upfirdn函数，用于进行上采样和下采样的滤波处理

from ._spline import (
    sepfir2d
)
# 导入_spline模块中的sepfir2d函数，用于二维分离型FIR滤波器的实现

from ._splines import *
# 导入_splines模块中的所有内容，这些内容包含了样条插值和拟合的函数

from ._bsplines import *
# 导入_bsplines模块中的所有内容，这些内容包含了B样条函数和相关计算

from ._filter_design import *
# 导入_filter_design模块中的所有内容，这些内容包含了滤波器设计的函数和工具

from ._fir_filter_design import *
# 导入_fir_filter_design模块中的所有内容，这些内容包含了FIR滤波器的设计和计算

from ._ltisys import *
# 导入_ltisys模块中的所有内容，这些内容包含了线性时不变系统的描述和计算

from ._lti_conversion import *
# 导入_lti_conversion模块中的所有内容，这些内容包含了LTI系统的不同表示之间的转换函数

from ._signaltools import *
# 导入_signaltools模块中的所有内容，这些内容包含了信号处理工具函数和滤波器操作

from ._savitzky_golay import savgol_coeffs, savgol_filter
# 导入_savitzky_golay模块中的savgol_coeffs和savgol_filter函数，用于Savitzky-Golay滤波器

from ._spectral_py import *
# 导入_spectral_py模块中的所有内容，这些内容包含了频谱分析和相关算法的实现

from ._short_time_fft import *
# 导入_short_time_fft模块中的所有内容，这些内容包含了短时傅里叶变换和逆变换的函数

from ._peak_finding import *
# 导入_peak_finding模块中的所有内容，这些内容包含了峰值检测和相关算法的实现

from ._czt import *
# 导入_czt模块中的所有内容，这些内容包含了Chirp Z-Transform和Zoom FFT的算法实现

from .windows import get_window  # keep this one in signal namespace
# 导入windows模块中的get_window函数，用于获取窗口函数，保留在signal命名空间中使用

# Deprecated namespaces, to be removed in v2.0.0
# 弃用的命名空间，在v2.0.0版本中将被移除
from . import (
    bsplines, filter_design, fir_filter_design, lti_conversion, ltisys,
    spectral, signaltools, waveforms, wavelets, spline
)

__all__ = [
    s for s in dir() if not s.startswith("_")
]
# 定义模块中公开的所有非下划线开头的对象名称列表，以便于from module import * 的使用

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
# 导入_pytesttester类并实例化为test对象，用于模块的单元测试

del PytestTester
# 删除_pytesttester类，以确保不会影响到模块的使用
```