# `D:\src\scipysrc\scipy\scipy\signal\tests\test_wavelets.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
from numpy.testing import assert_array_equal, assert_array_almost_equal  # 导入用于数组比较的测试工具

import scipy.signal._wavelets as wavelets  # 导入 SciPy 中的波形处理模块中的波形函数


class TestWavelets:
    def test_ricker(self):
        # 测试 ricker 波形生成函数
        w = wavelets._ricker(1.0, 1)
        expected = 2 / (np.sqrt(3 * 1.0) * (np.pi ** 0.25))
        assert_array_equal(w, expected)  # 检查生成的波形与期望值是否相等

        lengths = [5, 11, 15, 51, 101]
        for length in lengths:
            # 测试不同长度的 ricker 波形生成函数
            w = wavelets._ricker(length, 1.0)
            assert len(w) == length  # 检查生成的波形长度是否正确
            max_loc = np.argmax(w)
            assert max_loc == (length // 2)  # 检查波形的峰值位置是否在中心

        points = 100
        w = wavelets._ricker(points, 2.0)
        half_vec = np.arange(0, points // 2)
        # 检查波形是否对称
        assert_array_almost_equal(w[half_vec], w[-(half_vec + 1)])

        # 检查 ricker 波形的零点
        aas = [5, 10, 15, 20, 30]
        points = 99
        for a in aas:
            w = wavelets._ricker(points, a)
            vec = np.arange(0, points) - (points - 1.0) / 2
            exp_zero1 = np.argmin(np.abs(vec - a))
            exp_zero2 = np.argmin(np.abs(vec + a))
            assert_array_almost_equal(w[exp_zero1], 0)  # 检查波形的零点
            assert_array_almost_equal(w[exp_zero2], 0)  # 检查波形的零点

    def test_cwt(self):
        widths = [1.0]
        def delta_wavelet(s, t):
            return np.array([1])
        len_data = 100
        test_data = np.sin(np.pi * np.arange(0, len_data) / 10.0)

        # 测试 delta 函数输入时的连续小波变换结果
        cwt_dat = wavelets._cwt(test_data, delta_wavelet, widths)
        assert cwt_dat.shape == (len(widths), len_data)
        assert_array_almost_equal(test_data, cwt_dat.flatten())  # 检查连续小波变换的输出是否与原始数据一致

        # 检查不同宽度下的连续小波变换结果的形状
        widths = [1, 3, 4, 5, 10]
        cwt_dat = wavelets._cwt(test_data, wavelets._ricker, widths)
        assert cwt_dat.shape == (len(widths), len_data)

        widths = [len_data * 10]
        # 注意：这里的波形函数定义并不完全正确，但对于这个测试是可以接受的
        def flat_wavelet(l, w):
            return np.full(w, 1 / w)
        cwt_dat = wavelets._cwt(test_data, flat_wavelet, widths)
        assert_array_almost_equal(cwt_dat, np.mean(test_data))  # 检查连续小波变换的平均结果是否正确
```