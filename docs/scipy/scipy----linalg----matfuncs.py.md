# `D:\src\scipysrc\scipy\scipy\linalg\matfuncs.py`

```
# 导入 _sub_module_deprecation 函数，用于处理子模块的废弃警告
from scipy._lib.deprecation import _sub_module_deprecation

# 定义 __all__ 列表，包含了本模块中公开的所有函数名
__all__ = [  # noqa: F822
    'expm', 'cosm', 'sinm', 'tanm', 'coshm', 'sinhm',
    'tanhm', 'logm', 'funm', 'signm', 'sqrtm',
    'expm_frechet', 'expm_cond', 'fractional_matrix_power',
    'khatri_rao', 'norm', 'solve', 'inv', 'svd', 'schur', 'rsf2csf'
]

# 定义 __dir__ 函数，返回模块中所有公开的函数和变量名
def __dir__():
    return __all__

# 定义 __getattr__ 函数，处理当访问未定义的属性时的行为
def __getattr__(name):
    # 调用 _sub_module_deprecation 函数，传入相关参数，用于发出废弃警告
    return _sub_module_deprecation(sub_package="linalg", module="matfuncs",
                                   private_modules=["_matfuncs"], all=__all__,
                                   attribute=name)
```