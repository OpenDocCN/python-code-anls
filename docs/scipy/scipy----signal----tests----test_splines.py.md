# `D:\src\scipysrc\scipy\scipy\signal\tests\test_splines.py`

```
# pylint: disable=missing-docstring
# 导入必要的库
import numpy as np
from numpy import array
from numpy.testing import assert_allclose, assert_raises
import pytest

# 导入对称IIR滤波器函数
from scipy.signal._spline import (
    symiirorder1_ic, symiirorder2_ic_fwd, symiirorder2_ic_bwd)
from scipy.signal._splines import symiirorder1, symiirorder2


def _compute_symiirorder2_bwd_hs(k, cs, rsq, omega):
    # 计算 cs 的平方
    cssq = cs * cs
    # 取 k 的绝对值
    k = np.abs(k)
    # 计算 rsq 的 k/2 次方
    rsupk = np.power(rsq, k / 2.0)

    # 计算 c0 的值，这是一个滤波器的系数
    c0 = (cssq * (1.0 + rsq) / (1.0 - rsq) /
          (1 - 2 * rsq * np.cos(2 * omega) + rsq * rsq))
    # 计算 gamma 的值，这也是一个滤波器的系数
    gamma = (1.0 - rsq) / (1.0 + rsq) / np.tan(omega)
    # 返回滤波器的输出值
    return c0 * rsupk * (np.cos(omega * k) + gamma * np.sin(omega * k))


class TestSymIIR:
    @pytest.mark.parametrize(
        'dtype', [np.float32, np.float64, np.complex64, np.complex128])
    @pytest.mark.parametrize('precision', [-1.0, 0.7, 0.5, 0.25, 0.0075])
    def test_symiir1_ic(self, dtype, precision):
        # 根据 precision 的值调整 c_precision 的设定
        c_precision = precision
        if precision <= 0.0 or precision > 1.0:
            # 如果 precision 超出范围，则根据 dtype 调整 c_precision 的值
            if dtype in {np.float32, np.complex64}:
                c_precision = 1e-6
            else:
                c_precision = 1e-11

        # 对于一阶IIR滤波器的对称初始条件:
        # x[0] + z1 * \sum{k = 0}^{n - 1} x[k] * z1^k

        # 检查低通滤波器的初始条件
        # 使用系数 b = 0.85 对阶跃信号进行检查。初始条件是一个几何级数:
        # 1 + b * \sum_{k = 0}^{n - 1} u[k] b^k.

        # 找到初始条件对应的索引 n，使得 b**n < precision，这对应于 ceil(log(precision) / log(b))
        # 计算几何级数直到 n，可以使用部分和公式进行计算: (1 - b**n) / (1 - b)
        # 这是因为输入是一个阶跃信号。
        b = 0.85
        n_exp = int(np.ceil(np.log(c_precision) / np.log(b)))
        expected = np.asarray([[(1 - b ** n_exp) / (1 - b)]], dtype=dtype)
        expected = 1 + b * expected

        # 创建一个大小为 n + 1 的阶跃信号
        x = np.ones(n_exp + 1, dtype=dtype)
        # 使用 assert_allclose 来检查对称IIR一阶滤波器的初始条件
        assert_allclose(symiirorder1_ic(x, b, precision), expected,
                        atol=2e-6, rtol=2e-7)

        # 检查基于指数衰减信号的条件，基数为 2。
        # 同样的条件适用，因为 0.5^n * 0.85^n 的乘积仍然是一个几何级数
        b_d = array(b, dtype=dtype)
        expected = np.asarray(
            [[(1 - (0.5 * b_d) ** n_exp) / (1 - (0.5 * b_d))]], dtype=dtype)
        expected = 1 + b_d * expected

        # 创建一个指数衰减信号，大小为 n + 1
        x = 2 ** -np.arange(n_exp + 1, dtype=dtype)
        # 使用 assert_allclose 来检查对称IIR一阶滤波器的初始条件
        assert_allclose(symiirorder1_ic(x, b, precision), expected,
                        atol=2e-6, rtol=2e-7)
    # 定义一个测试函数，用于测试 symiirorder1_ic 函数的异常情况
    def test_symiir1_ic_fails(self):
        # 测试当 \sum_{n = 1}^{n} b^n > eps 时，symiirorder1_ic 函数会失败
        b = 0.85
        # 创建一个长度为 100 的单位步信号
        x = np.ones(100, dtype=np.float64)

        # 计算几何级数的闭式解
        precision = 1 / (1 - b)
        # 断言调用 symiirorder1_ic 函数会抛出 ValueError 异常
        assert_raises(ValueError, symiirorder1_ic, x, b, precision)

        # 测试当 |z1| >= 1 时，symiirorder1_ic 函数会失败
        assert_raises(ValueError, symiirorder1_ic, x, 1.0, -1)
        assert_raises(ValueError, symiirorder1_ic, x, 2.0, -1)

    @pytest.mark.parametrize(
        'dtype', [np.float32, np.float64, np.complex64, np.complex128])
    @pytest.mark.parametrize('precision', [-1.0, 0.7, 0.5, 0.25, 0.0075])
    # 定义一个测试函数，用于测试一阶符号递归IIR滤波器
    def test_symiir1(self, dtype, precision):
        # 将精度存储在本地变量c_precision中
        c_precision = precision
        # 如果精度小于等于0.0或大于1.0，则根据数据类型设置默认精度
        if precision <= 0.0 or precision > 1.0:
            if dtype in {np.float32, np.complex64}:
                c_precision = 1e-6
            else:
                c_precision = 1e-11

        # 测试一个以c0 = 0.15和z1 = 0.85为参数的低通滤波器
        # 使用200个样本的单位阶跃响应
        c0 = 0.15
        z1 = 0.85
        n = 200
        signal = np.ones(n, dtype=dtype)

        # 计算初始条件，详见test_symiir1_ic中的详细解释
        n_exp = int(np.ceil(np.log(c_precision) / np.log(z1)))
        initial = np.asarray((1 - z1 ** n_exp) / (1 - z1), dtype=dtype)
        initial = 1 + z1 * initial

        # 正向传播
        # 系统1 / (1 - z1 * z^-1)对单位阶跃响应的传递函数，
        # 其中初始条件为y0是 1 / (1 - z1 * z^-1) * (z^-1 / (1 - z^-1) + y0)

        # 解决给定表达式的逆Z变换得到：
        # y[n] = y0 * z1**n * u[n] +
        #        -z1 / (1 - z1) * z1**(k - 1) * u[k - 1] +
        #        1 / (1 - z1) * u[k - 1]
        # 这里d是Kronecker delta函数，u是单位阶跃函数

        # y0 * z1**n * u[n]
        pos = np.arange(n, dtype=dtype)
        comp1 = initial * z1**pos

        # -z1 / (1 - z1) * z1**(k - 1) * u[k - 1]
        comp2 = np.zeros(n, dtype=dtype)
        comp2[1:] = -z1 / (1 - z1) * z1**pos[:-1]

        # 1 / (1 - z1) * u[k - 1]
        comp3 = np.zeros(n, dtype=dtype)
        comp3[1:] = 1 / (1 - z1)

        # 预期的正向传播输出
        expected_fwd = comp1 + comp2 + comp3

        # 反向条件
        sym_cond = -c0 / (z1 - 1.0) * expected_fwd[-1]

        # 反向传播
        # 正向结果的传递函数等于正向系统乘以c0 / (1 - z1 * z)

        # 计算完整表达式的闭合形式是困难的，
        # 结果将从差分方程迭代计算得出
        exp_out = np.zeros(n, dtype=dtype)
        exp_out[0] = sym_cond

        for i in range(1, n):
            exp_out[i] = c0 * expected_fwd[n - 1 - i] + z1 * exp_out[i - 1]

        exp_out = exp_out[::-1]

        # 调用symiirorder1函数计算输出，并验证是否接近预期输出
        out = symiirorder1(signal, c0, z1, precision)
        assert_allclose(out, exp_out, atol=4e-6, rtol=6e-7)

    # 参数化测试，测试不同数据类型和精度下的符号递归IIR滤波器
    @pytest.mark.parametrize(
        'dtype', [np.float32, np.float64])
    @pytest.mark.parametrize('precision', [-1.0, 0.7, 0.5, 0.25, 0.0075])
    # 定义一个测试函数，测试二阶对称低通滤波器的初始前向条件
    def test_symiir2_initial_fwd(self, dtype, precision):
        # 将精度值存储在局部变量 c_precision 中，并根据条件进行调整
        c_precision = precision
        if precision <= 0.0 or precision > 1.0:
            # 如果精度不在合理范围内，根据数据类型设置默认精度值
            if dtype in {np.float32, np.complex64}:
                c_precision = 1e-6
            else:
                c_precision = 1e-11

        # 计算一个二阶对称低通滤波器的初始条件，其中 r = 0.5，omega = pi / 3，输入为单位阶跃响应
        r = np.asarray(0.5, dtype=dtype)
        omega = np.asarray(np.pi / 3.0, dtype=dtype)
        cs = 1 - 2 * r * np.cos(omega) + r**2

        # 初始化条件的索引 n 的上界和下界计算
        # 上界：log(精度 / sin(omega)) / log(精度)
        ub = np.ceil(np.log(c_precision / np.sin(omega)) / np.log(c_precision))
        # 下界：ceil(pi / omega) - 2
        lb = np.ceil(np.pi / omega) - 2
        # 选择较小的值作为指数 n_exp
        n_exp = min(ub, lb)

        # 计算二阶滤波器前向初始条件的第一个表达式
        fwd_initial_1 = (
            cs +
            2 * r * np.cos(omega) -
            r**2 -
            r**(n_exp + 2) * np.sin(omega * (n_exp + 3)) / np.sin(omega) +
            r**(n_exp + 3) * np.sin(omega * (n_exp + 2)) / np.sin(omega))

        # 计算二阶滤波器前向初始条件的第二个表达式
        ub = np.ceil(np.log(c_precision / np.sin(omega)) / np.log(c_precision))
        lb = np.ceil(np.pi / omega) - 3
        n_exp = min(ub, lb)
        fwd_initial_2 = (
            cs + cs * 2 * r * np.cos(omega) +
            (r**2 * np.sin(3 * omega) -
             r**3 * np.sin(2 * omega) -
             r**(n_exp + 3) * np.sin(omega * (n_exp + 4)) +
             r**(n_exp + 4) * np.sin(omega * (n_exp + 3))) / np.sin(omega))

        # 期望的输出，将前向初始条件组成一个行向量
        expected = np.r_[fwd_initial_1, fwd_initial_2][None, :]

        # 生成一个长度为 100 的信号向量，数据类型为 dtype
        n = 100
        signal = np.ones(n, dtype=dtype)

        # 调用 symiirorder2_ic_fwd 函数计算输出，并使用 assert_allclose 进行断言
        out = symiirorder2_ic_fwd(signal, r, omega, precision)
        assert_allclose(out, expected, atol=4e-6, rtol=6e-7)
    # 定义一个测试函数，用于测试二阶对称IIR滤波器的初始反向条件
    def test_symiir2_initial_bwd(self, dtype, precision):
        # 将参数 precision 赋值给局部变量 c_precision
        c_precision = precision
        # 如果 precision 小于等于0或者大于1，则根据 dtype 设置 c_precision
        if precision <= 0.0 or precision > 1.0:
            if dtype in {np.float32, np.complex64}:
                c_precision = 1e-6
            else:
                c_precision = 1e-11

        # 初始化 r 和 omega 数组
        r = np.asarray(0.5, dtype=dtype)
        omega = np.asarray(np.pi / 3.0, dtype=dtype)
        # 计算二阶滤波器的参数
        cs = 1 - 2 * r * np.cos(omega) + r * r
        a2 = 2 * r * np.cos(omega)
        a3 = -r * r

        # 设置信号长度为 n
        n = 100
        signal = np.ones(n, dtype=dtype)

        # 计算初始正向条件
        ic = symiirorder2_ic_fwd(signal, r, omega, precision)
        # 初始化输出数组 out，长度为 n+2
        out = np.zeros(n + 2, dtype=dtype)
        out[:2] = ic[0]

        # 应用正向系统 cs / (1 - a2 * z^-1 - a3 * z^-2))
        for i in range(2, n + 2):
            out[i] = cs * signal[i - 2] + a2 * out[i - 1] + a3 * out[i - 2]

        # 计算反向初始条件
        ic2 = np.zeros(2, dtype=dtype)
        idx = np.arange(n)

        # 计算对称IIR滤波器的反向初始条件
        diff = (_compute_symiirorder2_bwd_hs(idx, cs, r * r, omega) +
                _compute_symiirorder2_bwd_hs(idx + 1, cs, r * r, omega))
        ic2_0_all = np.cumsum(diff * out[:1:-1])
        pos = np.where(diff ** 2 < c_precision)[0]
        ic2[0] = ic2_0_all[pos[0]]

        diff = (_compute_symiirorder2_bwd_hs(idx - 1, cs, r * r, omega) +
                _compute_symiirorder2_bwd_hs(idx + 2, cs, r * r, omega))
        ic2_1_all = np.cumsum(diff * out[:1:-1])
        pos = np.where(diff ** 2 < c_precision)[0]
        ic2[1] = ic2_1_all[pos[0]]

        # 计算反向初始条件，作为对比用于断言
        out_ic = symiirorder2_ic_bwd(out, r, omega, precision)[0]
        # 使用断言检查计算出的反向初始条件与预期是否相近
        assert_allclose(out_ic, ic2, atol=4e-6, rtol=6e-7)

    # 标记测试参数化，设置 dtype 和 precision 的多组测试参数
    @pytest.mark.parametrize(
        'dtype', [np.float32, np.float64])
    @pytest.mark.parametrize('precision', [-1.0, 0.7, 0.5, 0.25, 0.0075])
    # 定义测试二阶对称IIR滤波器函数
    def test_symiir2(self, dtype, precision):
        # 初始化 r 和 omega 数组
        r = np.asarray(0.5, dtype=dtype)
        omega = np.asarray(np.pi / 3.0, dtype=dtype)
        # 计算二阶滤波器的参数
        cs = 1 - 2 * r * np.cos(omega) + r * r
        a2 = 2 * r * np.cos(omega)
        a3 = -r * r

        # 设置信号长度为 n
        n = 100
        signal = np.ones(n, dtype=dtype)

        # 计算初始正向条件
        ic = symiirorder2_ic_fwd(signal, r, omega, precision)
        # 初始化输出数组 out1，长度为 n+2
        out1 = np.zeros(n + 2, dtype=dtype)
        out1[:2] = ic[0]

        # 应用正向系统 cs / (1 - a2 * z^-1 - a3 * z^-2))
        for i in range(2, n + 2):
            out1[i] = cs * signal[i - 2] + a2 * out1[i - 1] + a3 * out1[i - 2]

        # 计算反向初始条件
        ic2 = symiirorder2_ic_bwd(out1, r, omega, precision)[0]

        # 应用反向系统 cs / (1 - a2 * z - a3 * z^2))，逆序计算
        exp = np.empty(n, dtype=dtype)
        exp[-2:] = ic2[::-1]

        for i in range(n - 3, -1, -1):
            exp[i] = cs * out1[i] + a2 * exp[i + 1] + a3 * exp[i + 2]

        # 调用二阶对称IIR滤波器函数，计算输出
        out = symiirorder2(signal, r, omega, precision)
        # 使用断言检查计算出的输出与预期是否相近
        assert_allclose(out, exp, atol=4e-6, rtol=6e-7)
    # 定义一个测试函数，用于测试对称IIR滤波器对整数输入的行为
    def test_symiir1_integer_input(self):
        # 创建一个包含整数值的 NumPy 数组，奇数索引位置为 -1，偶数索引位置为 1
        s = np.where(np.arange(100) % 2, -1, 1)
        # 将 s 转换为浮点数类型，并调用 symiirorder1 函数计算期望输出
        expected = symiirorder1(s.astype(float), 0.5, 0.5)
        # 调用 symiirorder1 函数对整数数组 s 进行计算
        out = symiirorder1(s, 0.5, 0.5)
        # 使用 assert_allclose 函数断言实际输出 out 与期望输出 expected 接近
        assert_allclose(out, expected)
    
    # 定义另一个测试函数，用于测试对称IIR滤波器对整数输入的行为
    def test_symiir2_integer_input(self):
        # 创建一个包含整数值的 NumPy 数组，奇数索引位置为 -1，偶数索引位置为 1
        s = np.where(np.arange(100) % 2, -1, 1)
        # 将 s 转换为浮点数类型，并调用 symiirorder2 函数计算期望输出
        expected = symiirorder2(s.astype(float), 0.5, np.pi / 3.0)
        # 调用 symiirorder2 函数对整数数组 s 进行计算
        out = symiirorder2(s, 0.5, np.pi / 3.0)
        # 使用 assert_allclose 函数断言实际输出 out 与期望输出 expected 接近
        assert_allclose(out, expected)
```