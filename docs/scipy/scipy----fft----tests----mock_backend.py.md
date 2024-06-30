# `D:\src\scipysrc\scipy\scipy\fft\tests\mock_backend.py`

```
# 导入必要的库：numpy 和 scipy 中的 fft 模块
import numpy as np
import scipy.fft

# 定义一个名为 _MockFunction 的类，用于模拟函数调用和返回值
class _MockFunction:
    def __init__(self, return_value = None):
        self.number_calls = 0  # 记录函数被调用的次数
        self.return_value = return_value  # 设定函数的返回值
        self.last_args = ([], {})  # 记录函数最后一次调用时的参数

    def __call__(self, *args, **kwargs):
        self.number_calls += 1  # 增加函数调用计数
        self.last_args = (args, kwargs)  # 记录当前函数调用的参数
        return self.return_value  # 返回预设的返回值


# 创建多个 _MockFunction 的实例，模拟不同函数的返回值
fft = _MockFunction(np.random.random(10))
fft2 = _MockFunction(np.random.random(10))
fftn = _MockFunction(np.random.random(10))

ifft = _MockFunction(np.random.random(10))
ifft2 = _MockFunction(np.random.random(10))
ifftn = _MockFunction(np.random.random(10))

rfft = _MockFunction(np.random.random(10))
rfft2 = _MockFunction(np.random.random(10))
rfftn = _MockFunction(np.random.random(10))

irfft = _MockFunction(np.random.random(10))
irfft2 = _MockFunction(np.random.random(10))
irfftn = _MockFunction(np.random.random(10))

hfft = _MockFunction(np.random.random(10))
hfft2 = _MockFunction(np.random.random(10))
hfftn = _MockFunction(np.random.random(10))

ihfft = _MockFunction(np.random.random(10))
ihfft2 = _MockFunction(np.random.random(10))
ihfftn = _MockFunction(np.random.random(10))

dct = _MockFunction(np.random.random(10))
idct = _MockFunction(np.random.random(10))
dctn = _MockFunction(np.random.random(10))
idctn = _MockFunction(np.random.random(10))

dst = _MockFunction(np.random.random(10))
idst = _MockFunction(np.random.random(10))
dstn = _MockFunction(np.random.random(10))
idstn = _MockFunction(np.random.random(10))

fht = _MockFunction(np.random.random(10))
ifht = _MockFunction(np.random.random(10))

# 设定模块的域名标识
__ua_domain__ = "numpy.scipy.fft"

# 创建一个字典 _implements，将 scipy.fft 模块中的函数映射到对应的 _MockFunction 实例
_implements = {
    scipy.fft.fft: fft,
    scipy.fft.fft2: fft2,
    scipy.fft.fftn: fftn,
    scipy.fft.ifft: ifft,
    scipy.fft.ifft2: ifft2,
    scipy.fft.ifftn: ifftn,
    scipy.fft.rfft: rfft,
    scipy.fft.rfft2: rfft2,
    scipy.fft.rfftn: rfftn,
    scipy.fft.irfft: irfft,
    scipy.fft.irfft2: irfft2,
    scipy.fft.irfftn: irfftn,
    scipy.fft.hfft: hfft,
    scipy.fft.hfft2: hfft2,
    scipy.fft.hfftn: hfftn,
    scipy.fft.ihfft: ihfft,
    scipy.fft.ihfft2: ihfft2,
    scipy.fft.ihfftn: ihfftn,
    scipy.fft.dct: dct,
    scipy.fft.idct: idct,
    scipy.fft.dctn: dctn,
    scipy.fft.idctn: idctn,
    scipy.fft.dst: dst,
    scipy.fft.idst: idst,
    scipy.fft.dstn: dstn,
    scipy.fft.idstn: idstn,
    scipy.fft.fht: fht,
    scipy.fft.ifht: ifht
}

# 定义 __ua_function__ 函数，用于根据传入的方法调用对应的 _MockFunction 实例来模拟函数调用
def __ua_function__(method, args, kwargs):
    fn = _implements.get(method)  # 获取对应 method 的 _MockFunction 实例
    return (fn(*args, **kwargs) if fn is not None  # 如果找到对应的 _MockFunction，则调用它
            else NotImplemented)  # 否则返回 NotImplemented
```