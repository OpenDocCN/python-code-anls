# `D:\src\scipysrc\scipy\scipy\signal\tests\test_czt.py`

```
# 导入pytest模块，用于运行单元测试
# 从numpy.testing中导入assert_allclose函数，用于比较数组是否接近
# 从scipy.fft中导入fft函数，用于进行快速傅里叶变换
# 从scipy.signal中导入czt, zoom_fft, czt_points, CZT, ZoomFFT等函数和类
import pytest
from numpy.testing import assert_allclose
from scipy.fft import fft
from scipy.signal import (czt, zoom_fft, czt_points, CZT, ZoomFFT)
import numpy as np


def check_czt(x):
    # 检查czt函数是否等效于普通fft函数
    y = fft(x)
    y1 = czt(x)
    # 使用assert_allclose函数比较y1和y的结果是否在给定的相对和绝对误差范围内接近
    assert_allclose(y1, y, rtol=1e-13)

    # 检查插值czt函数是否等效于普通fft函数
    y = fft(x, 100*len(x))
    y1 = czt(x, 100*len(x))
    assert_allclose(y1, y, rtol=1e-12)


def check_zoom_fft(x):
    # 检查zoom_fft函数是否等效于普通fft函数
    y = fft(x)
    y1 = zoom_fft(x, [0, 2-2./len(y)], endpoint=True)
    assert_allclose(y1, y, rtol=1e-11, atol=1e-14)
    y1 = zoom_fft(x, [0, 2])
    assert_allclose(y1, y, rtol=1e-11, atol=1e-14)

    # 测试fn参数为标量时的情况
    y1 = zoom_fft(x, 2-2./len(y), endpoint=True)
    assert_allclose(y1, y, rtol=1e-11, atol=1e-14)
    y1 = zoom_fft(x, 2)
    assert_allclose(y1, y, rtol=1e-11, atol=1e-14)

    # 检查带有过采样的zoom_fft函数是否等效于使用零填充的fft函数
    over = 10
    yover = fft(x, over*len(x))
    y2 = zoom_fft(x, [0, 2-2./len(yover)], m=len(yover), endpoint=True)
    assert_allclose(y2, yover, rtol=1e-12, atol=1e-10)
    y2 = zoom_fft(x, [0, 2], m=len(yover))
    assert_allclose(y2, yover, rtol=1e-12, atol=1e-10)

    # 检查zoom_fft函数在子范围上的工作是否正确
    w = np.linspace(0, 2-2./len(x), len(x))
    f1, f2 = w[3], w[6]
    y3 = zoom_fft(x, [f1, f2], m=3*over+1, endpoint=True)
    idx3 = slice(3*over, 6*over+1)
    assert_allclose(y3, yover[idx3], rtol=1e-13)


def test_1D():
    # 测试1维变换的情况

    np.random.seed(0)  # 设置随机数种子以保证结果可重复

    # 随机信号
    lengths = np.random.randint(8, 200, 20)
    np.append(lengths, 1)
    for length in lengths:
        x = np.random.random(length)
        check_zoom_fft(x)
        check_czt(x)

    # 高斯信号
    t = np.linspace(-2, 2, 128)
    x = np.exp(-t**2/0.01)
    check_zoom_fft(x)

    # 线性信号
    x = [1, 2, 3, 4, 5, 6, 7]
    check_zoom_fft(x)

    # 检查接近2的幂次方的情况
    check_zoom_fft(range(126-31))
    check_zoom_fft(range(127-31))
    check_zoom_fft(range(128-31))
    check_zoom_fft(range(129-31))
    check_zoom_fft(range(130-31))

    # 检查对n维数组输入的变换
    x = np.reshape(np.arange(3*2*28), (3, 2, 28))
    y1 = zoom_fft(x, [0, 2-2./28])
    y2 = zoom_fft(x[2, 0, :], [0, 2-2./28])
    assert_allclose(y1[2, 0], y2, rtol=1e-13, atol=1e-12)

    y1 = zoom_fft(x, [0, 2], endpoint=False)
    y2 = zoom_fft(x[2, 0, :], [0, 2], endpoint=False)
    assert_allclose(y1[2, 0], y2, rtol=1e-13, atol=1e-12)

    # 随机信号（非测试条件）
    x = np.random.rand(101)
    check_zoom_fft(x)

    # 尖峰信号
    t = np.linspace(0, 1, 128)
    x = np.sin(2*np.pi*t*5)+np.sin(2*np.pi*t*13)
    check_zoom_fft(x)

    # 正弦信号
    x = np.zeros(100, dtype=complex)
    # 将索引为 1、5 和 21 的元素赋值为 1，假设 x 是一个数组或列表
    x[[1, 5, 21]] = 1
    # 调用 check_zoom_fft 函数对数组 x 进行检查和处理
    check_zoom_fft(x)
    
    # 在数组 x 中添加复数部分，复数部分为虚数单位乘以从 0 到 0.5 均匀分布的数组
    x += 1j*np.linspace(0, 0.5, x.shape[0])
    # 再次调用 check_zoom_fft 函数对更新后的数组 x 进行检查和处理
    check_zoom_fft(x)
def test_large_prime_lengths():
    np.random.seed(0)  # 设置随机种子为0，使得随机数生成具有确定性
    for N in (101, 1009, 10007):
        x = np.random.rand(N)  # 生成长度为N的随机数组x
        y = fft(x)  # 对x进行快速傅里叶变换
        y1 = czt(x)  # 对x进行Chirp Z变换
        assert_allclose(y, y1, rtol=1e-12)  # 断言y和y1在相对误差1e-12内相等


@pytest.mark.slow
def test_czt_vs_fft():
    np.random.seed(123)  # 设置随机种子为123
    random_lengths = np.random.exponential(100000, size=10).astype('int')  # 生成指数分布的随机长度数组
    for n in random_lengths:
        a = np.random.randn(n)  # 生成长度为n的正态分布随机数组a
        assert_allclose(czt(a), fft(a), rtol=1e-11)  # 断言Chirp Z变换和快速傅里叶变换在相对误差1e-11内相等


def test_empty_input():
    with pytest.raises(ValueError, match='Invalid number of CZT'):  # 断言抛出ValueError异常，异常消息包含'Invalid number of CZT'
        czt([])  # 对空输入进行Chirp Z变换
    with pytest.raises(ValueError, match='Invalid number of CZT'):  # 断言抛出ValueError异常，异常消息包含'Invalid number of CZT'
        zoom_fft([], 0.5)  # 对空输入和缩放因子0.5进行快速傅里叶变换


def test_0_rank_input():
    with pytest.raises(IndexError, match='tuple index out of range'):  # 断言抛出IndexError异常，异常消息包含'tuple index out of range'
        czt(5)  # 对标量5进行Chirp Z变换
    with pytest.raises(IndexError, match='tuple index out of range'):  # 断言抛出IndexError异常，异常消息包含'tuple index out of range'
        zoom_fft(5, 0.5)  # 对标量5和缩放因子0.5进行快速傅里叶变换


@pytest.mark.parametrize('impulse', ([0, 0, 1], [0, 0, 1, 0, 0],
                                     np.concatenate((np.array([0, 0, 1]),
                                                     np.zeros(100)))))
@pytest.mark.parametrize('m', (1, 3, 5, 8, 101, 1021))
@pytest.mark.parametrize('a', (1, 2, 0.5, 1.1))
# 用于测试远离单位圆的步骤，但不会因数值误差而发生爆炸
@pytest.mark.parametrize('w', (None, 0.98534 + 0.17055j))
def test_czt_math(impulse, m, w, a):
    # 冲激响应的Z变换在所有点上都是1
    assert_allclose(czt(impulse[2:], m=m, w=w, a=a),
                    np.ones(m), rtol=1e-10)  # 断言Chirp Z变换结果和全1数组在相对误差1e-10内相等

    # 延迟冲激响应的Z变换是z**-1
    assert_allclose(czt(impulse[1:], m=m, w=w, a=a),
                    czt_points(m=m, w=w, a=a)**-1, rtol=1e-10)  # 断言Chirp Z变换结果和Chirp Z点集的倒数在相对误差1e-10内相等

    # 两个延迟冲激响应的Z变换是z**-2
    assert_allclose(czt(impulse, m=m, w=w, a=a),
                    czt_points(m=m, w=w, a=a)**-2, rtol=1e-10)  # 断言Chirp Z变换结果和Chirp Z点集的平方的倒数在相对误差1e-10内相等


def test_int_args():
    # 整数参数`a`会产生全为0的结果
    assert_allclose(abs(czt([0, 1], m=10, a=2)), 0.5*np.ones(10), rtol=1e-15)  # 断言Chirp Z变换的绝对值结果和0.5乘以全1数组在相对误差1e-15内相等
    assert_allclose(czt_points(11, w=2), 1/(2**np.arange(11)), rtol=1e-30)  # 断言Chirp Z点集结果和1除以2的n次方数组在相对误差1e-30内相等


def test_czt_points():
    for N in (1, 2, 3, 8, 11, 100, 101, 10007):
        assert_allclose(czt_points(N), np.exp(2j*np.pi*np.arange(N)/N),
                        rtol=1e-30)  # 断言Chirp Z点集结果和指数函数在相对误差1e-30内相等

    assert_allclose(czt_points(7, w=1), np.ones(7), rtol=1e-30)  # 断言Chirp Z点集结果和全1数组在相对误差1e-30内相等
    assert_allclose(czt_points(11, w=2.), 1/(2**np.arange(11)), rtol=1e-30)  # 断言Chirp Z点集结果和1除以2的n次方数组在相对误差1e-30内相等

    func = CZT(12, m=11, w=2., a=1)
    assert_allclose(func.points(), 1/(2**np.arange(11)), rtol=1e-30)  # 断言Chirp Z点集对象的结果和1除以2的n次方数组在相对误差1e-30内相等


@pytest.mark.parametrize('cls, args', [(CZT, (100,)), (ZoomFFT, (100, 0.2))])
def test_CZT_size_mismatch(cls, args):
    # 数据大小与函数预期大小不匹配
    myfunc = cls(*args)
    with pytest.raises(ValueError, match='CZT defined for'):
        myfunc(np.arange(5))  # 断言调用的函数抛出ValueError异常，异常消息包含'CZT defined for'


def test_invalid_range():
    with pytest.raises(ValueError, match='2-length sequence'):  # 断言抛出ValueError异常，异常消息包含'2-length sequence'
        ZoomFFT(100, [1, 2, 3])  # 创建ZoomFFT对象时传入长度为3的列表作为参数
# 测试函数，用于测试 czt_points 函数在输入 m 时是否抛出 ValueError 异常
def test_czt_points_errors(m):
    # 使用 pytest 的 pytest.raises 断言捕获 ValueError 异常，并匹配错误消息 'Invalid number of CZT'
    with pytest.raises(ValueError, match='Invalid number of CZT'):
        czt_points(m)


# 参数化测试函数，测试在不同的 size 值下是否会抛出 ValueError 异常
@pytest.mark.parametrize('size', [0, -5, 3.5, 4.0])
def test_nonsense_size(size):
    # Numpy 和 Scipy 的 fft() 对于输出大小为 0 会抛出 ValueError 异常，因此我们也会如此处理
    # 使用 pytest.raises 断言捕获 ValueError 异常，并匹配错误消息 'Invalid number of CZT'
    with pytest.raises(ValueError, match='Invalid number of CZT'):
        CZT(size, 3)
    with pytest.raises(ValueError, match='Invalid number of CZT'):
        ZoomFFT(size, 0.2, 3)
    with pytest.raises(ValueError, match='Invalid number of CZT'):
        CZT(3, size)
    with pytest.raises(ValueError, match='Invalid number of CZT'):
        ZoomFFT(3, 0.2, size)
    with pytest.raises(ValueError, match='Invalid number of CZT'):
        czt([1, 2, 3], size)
    with pytest.raises(ValueError, match='Invalid number of CZT'):
        zoom_fft([1, 2, 3], 0.2, size)
```