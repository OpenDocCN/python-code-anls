# `D:\src\scipysrc\scipy\scipy\fft\_realtransforms_backend.py`

```
# 从 scipy._lib._array_api 模块导入 array_namespace 函数
from scipy._lib._array_api import array_namespace
# 导入 numpy 库，并用 np 别名表示
import numpy as np
# 从当前目录中导入 _pocketfft 模块
from . import _pocketfft

# 定义公开的函数和变量列表
__all__ = ['dct', 'idct', 'dst', 'idst', 'dctn', 'idctn', 'dstn', 'idstn']

# 定义一个内部函数，用于执行 PocketFFT 提供的变换函数
def _execute(pocketfft_func, x, type, s, axes, norm, 
             overwrite_x, workers, orthogonalize):
    # 使用 array_namespace 函数获得 x 对象所属的命名空间
    xp = array_namespace(x)
    # 将 x 转换为 numpy 数组
    x = np.asarray(x)
    # 调用 PocketFFT 提供的变换函数进行变换
    y = pocketfft_func(x, type, s, axes, norm,
                       overwrite_x=overwrite_x, workers=workers,
                       orthogonalize=orthogonalize)
    # 将结果 y 转换为 xp 命名空间的数组并返回
    return xp.asarray(y)

# 定义多维 DCT 变换函数，调用 _execute 函数执行变换
def dctn(x, type=2, s=None, axes=None, norm=None,
         overwrite_x=False, workers=None, *, orthogonalize=None):
    return _execute(_pocketfft.dctn, x, type, s, axes, norm, 
                    overwrite_x, workers, orthogonalize)

# 定义多维 IDCT 变换函数，调用 _execute 函数执行变换
def idctn(x, type=2, s=None, axes=None, norm=None,
          overwrite_x=False, workers=None, *, orthogonalize=None):
    return _execute(_pocketfft.idctn, x, type, s, axes, norm, 
                    overwrite_x, workers, orthogonalize)

# 定义多维 DST 变换函数，调用 _execute 函数执行变换
def dstn(x, type=2, s=None, axes=None, norm=None,
         overwrite_x=False, workers=None, orthogonalize=None):
    return _execute(_pocketfft.dstn, x, type, s, axes, norm, 
                    overwrite_x, workers, orthogonalize)

# 定义多维 IDST 变换函数，调用 _execute 函数执行变换
def idstn(x, type=2, s=None, axes=None, norm=None,
          overwrite_x=False, workers=None, *, orthogonalize=None):
    return _execute(_pocketfft.idstn, x, type, s, axes, norm, 
                    overwrite_x, workers, orthogonalize)

# 定义一维 DCT 变换函数，调用 _execute 函数执行变换
def dct(x, type=2, n=None, axis=-1, norm=None,
        overwrite_x=False, workers=None, orthogonalize=None):
    return _execute(_pocketfft.dct, x, type, n, axis, norm, 
                    overwrite_x, workers, orthogonalize)

# 定义一维 IDCT 变换函数，调用 _execute 函数执行变换
def idct(x, type=2, n=None, axis=-1, norm=None,
         overwrite_x=False, workers=None, orthogonalize=None):
    return _execute(_pocketfft.idct, x, type, n, axis, norm, 
                    overwrite_x, workers, orthogonalize)

# 定义一维 DST 变换函数，调用 _execute 函数执行变换
def dst(x, type=2, n=None, axis=-1, norm=None,
        overwrite_x=False, workers=None, orthogonalize=None):
    return _execute(_pocketfft.dst, x, type, n, axis, norm, 
                    overwrite_x, workers, orthogonalize)

# 定义一维 IDST 变换函数，调用 _execute 函数执行变换
def idst(x, type=2, n=None, axis=-1, norm=None,
         overwrite_x=False, workers=None, orthogonalize=None):
    return _execute(_pocketfft.idst, x, type, n, axis, norm, 
                    overwrite_x, workers, orthogonalize)
```