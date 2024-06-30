# `D:\src\scipysrc\scipy\scipy\fft\tests\test_multithreading.py`

```
# 导入需要的库和模块
from scipy import fft  # 导入 scipy 中的 FFT 相关函数
import numpy as np  # 导入 NumPy 库，用于科学计算
import pytest  # 导入 pytest 测试框架
from numpy.testing import assert_allclose  # 导入 NumPy 的测试函数 assert_allclose
import multiprocessing  # 导入 multiprocessing 模块，用于多进程支持
import os  # 导入 os 模块，用于获取系统信息和执行系统命令


@pytest.fixture(scope='module')
def x():
    return np.random.randn(512, 128)  # 返回一个随机生成的 512x128 的 NumPy 数组作为测试数据


@pytest.mark.parametrize("func", [
    fft.fft, fft.ifft, fft.fft2, fft.ifft2, fft.fftn, fft.ifftn,  # 定义 FFT 相关函数的参数化测试
    fft.rfft, fft.irfft, fft.rfft2, fft.irfft2, fft.rfftn, fft.irfftn,
    fft.hfft, fft.ihfft, fft.hfft2, fft.ihfft2, fft.hfftn, fft.ihfftn,
    fft.dct, fft.idct, fft.dctn, fft.idctn,
    fft.dst, fft.idst, fft.dstn, fft.idstn,
])
@pytest.mark.parametrize("workers", [2, -1])
def test_threaded_same(x, func, workers):
    expected = func(x, workers=1)  # 调用 FFT 函数，使用单线程计算预期结果
    actual = func(x, workers=workers)  # 调用 FFT 函数，使用参数化的线程数计算实际结果
    assert_allclose(actual, expected)  # 使用 NumPy 的 assert_allclose 函数检查计算结果是否接近


def _mt_fft(x):
    return fft.fft(x, workers=2)  # 在单独的函数中调用 FFT 函数，使用两个线程进行计算


@pytest.mark.slow
def test_mixed_threads_processes(x):
    # 测试在 fork 前后 FFT 线程池的安全性

    expect = fft.fft(x, workers=2)  # 使用两个线程计算 FFT 的预期结果

    with multiprocessing.Pool(2) as p:
        res = p.map(_mt_fft, [x for _ in range(4)])  # 使用多进程 Pool 调用 _mt_fft 函数

    for r in res:
        assert_allclose(r, expect)  # 检查多进程计算的结果是否与预期接近

    fft.fft(x, workers=2)  # 再次调用 FFT 函数，使用两个线程进行计算


def test_invalid_workers(x):
    cpus = os.cpu_count()  # 获取系统 CPU 核心数

    fft.ifft([1], workers=-cpus)  # 调用 FFT 的逆变换函数，使用负数核心数作为参数

    with pytest.raises(ValueError, match='workers must not be zero'):
        fft.fft(x, workers=0)  # 使用零作为参数调用 FFT 函数，期待抛出 ValueError 异常

    with pytest.raises(ValueError, match='workers value out of range'):
        fft.ifft(x, workers=-cpus-1)  # 使用超出范围的负数作为参数调用 FFT 函数，期待抛出 ValueError 异常


def test_set_get_workers():
    cpus = os.cpu_count()  # 获取系统 CPU 核心数
    assert fft.get_workers() == 1  # 断言当前 FFT 函数的默认线程数为 1

    with fft.set_workers(4):  # 设置 FFT 函数的线程数为 4
        assert fft.get_workers() == 4  # 检查设置后 FFT 函数的线程数是否为 4

        with fft.set_workers(-1):  # 使用 -1 设置 FFT 函数的线程数
            assert fft.get_workers() == cpus  # 检查是否自动设置为系统 CPU 核心数

        assert fft.get_workers() == 4  # 确保内部设置不影响外部设置

    assert fft.get_workers() == 1  # 确保退出上下文后 FFT 函数的线程数恢复为默认值 1

    with fft.set_workers(-cpus):  # 使用负的 CPU 核心数设置 FFT 函数的线程数
        assert fft.get_workers() == 1  # 确保不允许使用负数线程数


def test_set_workers_invalid():

    with pytest.raises(ValueError, match='workers must not be zero'):
        with fft.set_workers(0):  # 使用零作为参数设置 FFT 函数的线程数，期待抛出 ValueError 异常
            pass

    with pytest.raises(ValueError, match='workers value out of range'):
        with fft.set_workers(-os.cpu_count()-1):  # 使用超出范围的负数作为参数设置 FFT 函数的线程数，期待抛出 ValueError 异常
            pass
```