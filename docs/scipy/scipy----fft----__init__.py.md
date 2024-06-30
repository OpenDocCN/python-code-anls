# `D:\src\scipysrc\scipy\scipy\fft\__init__.py`

```
# 导入模块中的各种离散傅立叶变换函数，包括一维、二维和多维的变换
from ._basic import (
    fft, ifft, fft2, ifft2, fftn, ifftn,
    rfft, irfft, rfft2, irfft2, rfftn, irfftn,
    hfft, ihfft, hfft2, ihfft2, hfftn, ihfftn)

# 导入模块中的各种实数序列的傅立叶变换函数，包括一维、多维的变换
from ._realtransforms import dct, idct, dst, idst, dctn, idctn, dstn, idstn

# 导入模块中的快速汉克尔变换函数
from ._fftlog import fht, ifht, fhtoffset

# 导入模块中的辅助函数，包括频率计算、频谱移位等
from ._helper import (
    next_fast_len, prev_fast_len, fftfreq,
    rfftfreq, fftshift, ifftshift)

# 导入模块中的后端控制函数，用于设置和管理傅立叶变换的后端
from ._backend import (set_backend, skip_backend, set_global_backend,
                       register_backend)
# 导入从 _pocketfft.helper 模块中的 set_workers 和 get_workers 函数
from ._pocketfft.helper import set_workers, get_workers

# 定义 __all__ 列表，指定了该模块公开的函数和类名
__all__ = [
    'fft', 'ifft', 'fft2', 'ifft2', 'fftn', 'ifftn',
    'rfft', 'irfft', 'rfft2', 'irfft2', 'rfftn', 'irfftn',
    'hfft', 'ihfft', 'hfft2', 'ihfft2', 'hfftn', 'ihfftn',
    'fftfreq', 'rfftfreq', 'fftshift', 'ifftshift',
    'next_fast_len', 'prev_fast_len',
    'dct', 'idct', 'dst', 'idst', 'dctn', 'idctn', 'dstn', 'idstn',
    'fht', 'ifht',
    'fhtoffset',
    'set_backend', 'skip_backend', 'set_global_backend', 'register_backend',
    'get_workers', 'set_workers']

# 导入 PytestTester 类来进行测试，使用当前模块的名称作为参数
from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)

# 删除当前作用域中的 PytestTester 类的引用，以避免在模块中保留不必要的对象引用
del PytestTester
```