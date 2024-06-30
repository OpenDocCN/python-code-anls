# `D:\src\scipysrc\scipy\scipy\linalg\special_matrices.py`

```
# 导入 _sub_module_deprecation 函数，用于处理子模块弃用警告
# 导入警告声明：本文件不适合公开使用，将在 SciPy v2.0.0 中删除
from scipy._lib.deprecation import _sub_module_deprecation

# 定义 __all__ 列表，包含特殊矩阵操作的函数名，用于限制模块的导出内容
__all__ = [
    'toeplitz', 'circulant', 'hankel',
    'hadamard', 'leslie', 'kron', 'block_diag', 'companion',
    'helmert', 'hilbert', 'invhilbert', 'pascal', 'invpascal', 'dft',
    'fiedler', 'fiedler_companion', 'convolution_matrix'
]


# 定义 __dir__() 函数，返回模块中 __all__ 列表中的所有名称，用于模块自省
def __dir__():
    return __all__


# 定义 __getattr__(name) 函数，用于在模块中动态获取属性
def __getattr__(name):
    # 调用 _sub_module_deprecation 函数，发出子模块弃用警告，并指定相关参数
    return _sub_module_deprecation(sub_package="linalg", module="special_matrices",
                                   private_modules=["_special_matrices"], all=__all__,
                                   attribute=name)
```