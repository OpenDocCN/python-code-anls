# `D:\src\scipysrc\scipy\scipy\optimize\cython_optimize.pxd`

```
# Public Cython API declarations
#
# See doc/source/dev/contributor/public_cython_api.rst for guidelines

# 导入需要的 Cython 函数声明，这些函数提供了公共的 Cython API
# 这里的 cimport 语句提供了对旧版 ABI 的支持，修改它可能导致 ABI 的向前兼容性中断
# (gh-11793)，因此我们保持原样不进行进一步的 cimport 修改
from scipy.optimize.cython_optimize._zeros cimport (
    brentq, brenth, ridder, bisect, zeros_full_output)
```