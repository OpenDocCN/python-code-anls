# `D:\src\scipysrc\numpy\numpy\fft\__init__.pyi`

```py
# 导入 numpy 库中的 PytestTester 类
from numpy._pytesttester import PytestTester

# 导入 numpy.fft._pocketfft 模块中的 FFT 相关函数
from numpy.fft._pocketfft import (
    fft as fft,           # 快速傅里叶变换（FFT）
    ifft as ifft,         # 快速傅里叶逆变换（IFFT）
    rfft as rfft,         # 实数输入的快速傅里叶变换
    irfft as irfft,       # 实数输入的快速傅里叶逆变换
    hfft as hfft,         # Hermitian 傅里叶变换
    ihfft as ihfft,       # Hermitian 傅里叶逆变换
    rfftn as rfftn,       # 多维实数输入的快速傅里叶变换
    irfftn as irfftn,     # 多维实数输入的快速傅里叶逆变换
    rfft2 as rfft2,       # 2D 实数输入的快速傅里叶变换
    irfft2 as irfft2,     # 2D 实数输入的快速傅里叶逆变换
    fft2 as fft2,         # 2D 快速傅里叶变换
    ifft2 as ifft2,       # 2D 快速傅里叶逆变换
    fftn as fftn,         # 多维快速傅里叶变换
    ifftn as ifftn,       # 多维快速傅里叶逆变换
)

# 导入 numpy.fft._helper 模块中的辅助函数
from numpy.fft._helper import (
    fftshift as fftshift,     # 频率域数据的移位操作
    ifftshift as ifftshift,   # 逆移位操作
    fftfreq as fftfreq,       # 返回 FFT 输出频率
    rfftfreq as rfftfreq,     # 实数输入的 FFT 输出频率
)

# 定义模块中公开的函数和类的名称列表
__all__: list[str]

# 定义 PytestTester 类的实例 test
test: PytestTester
```