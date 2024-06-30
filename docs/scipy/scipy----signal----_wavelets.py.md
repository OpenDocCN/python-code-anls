# `D:\src\scipysrc\scipy\scipy\signal\_wavelets.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
from scipy.signal import convolve  # 从 SciPy 库中导入 convolve 函数，用于信号处理中的卷积运算


def _ricker(points, a):
    A = 2 / (np.sqrt(3 * a) * (np.pi**0.25))  # 计算 Ricker 小波函数的系数 A
    wsq = a**2  # 计算参数 a 的平方
    vec = np.arange(0, points) - (points - 1.0) / 2  # 创建一个以中心对称的向量 vec
    xsq = vec**2  # 计算 vec 中每个元素的平方
    mod = (1 - xsq / wsq)  # 计算 Ricker 小波的调制部分
    gauss = np.exp(-xsq / (2 * wsq))  # 计算 Ricker 小波的高斯部分
    total = A * mod * gauss  # 计算完整的 Ricker 小波函数
    return total  # 返回计算结果


def _cwt(data, wavelet, widths, dtype=None, **kwargs):
    # 确定输出的数据类型
    if dtype is None:
        # 如果未指定 dtype，则根据波形函数的返回类型来确定
        if np.asarray(wavelet(1, widths[0], **kwargs)).dtype.char in 'FDG':
            dtype = np.complex128  # 复数类型
        else:
            dtype = np.float64  # 浮点数类型

    output = np.empty((len(widths), len(data)), dtype=dtype)  # 创建输出数组，用于存储连续小波变换的结果
    for ind, width in enumerate(widths):
        N = np.min([10 * width, len(data)])  # 计算窗口长度 N
        wavelet_data = np.conj(wavelet(N, width, **kwargs)[::-1])  # 生成逆时针反转的复共轭小波数据
        output[ind] = convolve(data, wavelet_data, mode='same')  # 对输入数据进行卷积运算，得到连续小波变换的输出
    return output  # 返回连续小波变换的结果数组
```