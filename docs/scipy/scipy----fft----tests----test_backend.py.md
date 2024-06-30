# `D:\src\scipysrc\scipy\scipy\fft\tests\test_backend.py`

```
# 导入所需模块和函数
from functools import partial  # 导入partial函数，用于创建带有部分参数的新函数

import numpy as np  # 导入NumPy库并简称为np
import scipy.fft  # 导入SciPy的FFT模块
from scipy.fft import _fftlog, _pocketfft, set_backend  # 导入SciPy的内部FFT相关模块和函数
from scipy.fft.tests import mock_backend  # 导入用于测试的模拟后端

from numpy.testing import assert_allclose, assert_equal  # 导入NumPy测试模块中的测试函数
import pytest  # 导入pytest模块用于单元测试

# 定义要测试的FFT相关函数的名称
fnames = ('fft', 'fft2', 'fftn',
          'ifft', 'ifft2', 'ifftn',
          'rfft', 'rfft2', 'rfftn',
          'irfft', 'irfft2', 'irfftn',
          'dct', 'idct', 'dctn', 'idctn',
          'dst', 'idst', 'dstn', 'idstn',
          'fht', 'ifht')

# 使用NumPy中对应的FFT函数
np_funcs = (np.fft.fft, np.fft.fft2, np.fft.fftn,
            np.fft.ifft, np.fft.ifft2, np.fft.ifftn,
            np.fft.rfft, np.fft.rfft2, np.fft.rfftn,
            np.fft.irfft, np.fft.irfft2, np.fft.irfftn,
            np.fft.hfft, _pocketfft.hfft2, _pocketfft.hfftn,  # 使用_pocketfft中的hfft2和hfftn函数，因为NumPy中没有
            np.fft.ihfft, _pocketfft.ihfft2, _pocketfft.ihfftn,
            _pocketfft.dct, _pocketfft.idct, _pocketfft.dctn, _pocketfft.idctn,
            _pocketfft.dst, _pocketfft.idst, _pocketfft.dstn, _pocketfft.idstn,
            partial(_fftlog.fht, dln=2, mu=0.5),  # 使用_fftlog中的fht函数，设置参数dln和mu
            partial(_fftlog.ifht, dln=2, mu=0.5))  # 使用_fftlog中的ifht函数，设置参数dln和mu

# 使用SciPy中对应的FFT函数
funcs = (scipy.fft.fft, scipy.fft.fft2, scipy.fft.fftn,
         scipy.fft.ifft, scipy.fft.ifft2, scipy.fft.ifftn,
         scipy.fft.rfft, scipy.fft.rfft2, scipy.fft.rfftn,
         scipy.fft.irfft, scipy.fft.irfft2, scipy.fft.irfftn,
         scipy.fft.hfft, scipy.fft.hfft2, scipy.fft.hfftn,
         scipy.fft.ihfft, scipy.fft.ihfft2, scipy.fft.ihfftn,
         scipy.fft.dct, scipy.fft.idct, scipy.fft.dctn, scipy.fft.idctn,
         scipy.fft.dst, scipy.fft.idst, scipy.fft.dstn, scipy.fft.idstn,
         partial(scipy.fft.fht, dln=2, mu=0.5),  # 使用scipy.fft中的fht函数，设置参数dln和mu
         partial(scipy.fft.ifht, dln=2, mu=0.5))  # 使用scipy.fft中的ifht函数，设置参数dln和mu

# 使用模拟后端的相关函数
mocks = (mock_backend.fft, mock_backend.fft2, mock_backend.fftn,
         mock_backend.ifft, mock_backend.ifft2, mock_backend.ifftn,
         mock_backend.rfft, mock_backend.rfft2, mock_backend.rfftn,
         mock_backend.irfft, mock_backend.irfft2, mock_backend.irfftn,
         mock_backend.hfft, mock_backend.hfft2, mock_backend.hfftn,
         mock_backend.ihfft, mock_backend.ihfft2, mock_backend.ihfftn,
         mock_backend.dct, mock_backend.idct,
         mock_backend.dctn, mock_backend.idctn,
         mock_backend.dst, mock_backend.idst,
         mock_backend.dstn, mock_backend.idstn,
         mock_backend.fht, mock_backend.ifht)

# 使用pytest的参数化测试，分别测试每个函数
@pytest.mark.parametrize("func, np_func, mock", zip(funcs, np_funcs, mocks))
def test_backend_call(func, np_func, mock):
    x = np.arange(20).reshape((10,2))  # 创建一个20个元素的数组，并重塑为10行2列
    answer = np_func(x.astype(np.float64))  # 使用NumPy的对应函数计算期望值
    assert_allclose(func(x), answer, atol=1e-10)  # 断言函数输出与期望值接近

    with set_backend(mock_backend, only=True):  # 设置使用模拟后端
        mock.number_calls = 0  # 重置模拟函数调用次数计数器
        y = func(x)  # 调用函数使用模拟后端
        assert_equal(y, mock.return_value)  # 断言函数输出与模拟返回值相等
        assert_equal(mock.number_calls, 1)  # 断言模拟函数被调用次数为1

    assert_allclose(func(x), answer, atol=1e-10)  # 再次断言函数输出与期望值接近
# 定义用于测试的 FFT 相关函数的元组，包括标准的和模拟的实现
plan_funcs = (scipy.fft.fft, scipy.fft.fft2, scipy.fft.fftn,
              scipy.fft.ifft, scipy.fft.ifft2, scipy.fft.ifftn,
              scipy.fft.rfft, scipy.fft.rfft2, scipy.fft.rfftn,
              scipy.fft.irfft, scipy.fft.irfft2, scipy.fft.irfftn,
              scipy.fft.hfft, scipy.fft.hfft2, scipy.fft.hfftn,
              scipy.fft.ihfft, scipy.fft.ihfft2, scipy.fft.ihfftn)

plan_mocks = (mock_backend.fft, mock_backend.fft2, mock_backend.fftn,
              mock_backend.ifft, mock_backend.ifft2, mock_backend.ifftn,
              mock_backend.rfft, mock_backend.rfft2, mock_backend.rfftn,
              mock_backend.irfft, mock_backend.irfft2, mock_backend.irfftn,
              mock_backend.hfft, mock_backend.hfft2, mock_backend.hfftn,
              mock_backend.ihfft, mock_backend.ihfft2, mock_backend.ihfftn)


# 使用 pytest 的参数化测试来逐一测试每个 FFT 函数及其模拟实现
@pytest.mark.parametrize("func, mock", zip(plan_funcs, plan_mocks))
def test_backend_plan(func, mock):
    # 创建一个包含 20 个元素的数组，并将其变形为 (10, 2) 的矩阵
    x = np.arange(20).reshape((10, 2))

    # 测试在指定了不支持的 plan 参数时，是否会抛出 NotImplementedError 异常
    with pytest.raises(NotImplementedError, match='precomputed plan'):
        func(x, plan='foo')

    # 使用 mock_backend 设置 FFT 函数的模拟实现，并执行相同的 plan 参数测试
    with set_backend(mock_backend, only=True):
        # 初始化模拟函数调用次数为 0
        mock.number_calls = 0
        # 调用 FFT 函数，使用不支持的 plan 参数，并检查返回值是否符合预期
        y = func(x, plan='foo')
        assert_equal(y, mock.return_value)
        # 检查模拟函数的调用次数是否增加到 1
        assert_equal(mock.number_calls, 1)
        # 检查最后一次调用模拟函数时使用的 plan 参数是否为 'foo'
        assert_equal(mock.last_args[1]['plan'], 'foo')
```