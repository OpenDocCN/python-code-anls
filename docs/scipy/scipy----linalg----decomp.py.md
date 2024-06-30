# `D:\src\scipysrc\scipy\scipy\linalg\decomp.py`

```
# 导入_scipy._lib.deprecation模块中的_sub_module_deprecation函数，用于处理子模块的弃用警告
from scipy._lib.deprecation import _sub_module_deprecation

# 定义__all__列表，指定了模块中可以被导入的公共接口名称
__all__ = [
    'eig', 'eigvals', 'eigh', 'eigvalsh',  # 特征值和特征向量计算函数
    'eig_banded', 'eigvals_banded',        # 带状矩阵特征值计算函数
    'eigh_tridiagonal', 'eigvalsh_tridiagonal',  # 三对角矩阵特征值计算函数
    'hessenberg', 'cdf2rdf',               # 矩阵分解和转换函数
    'LinAlgError', 'norm', 'get_lapack_funcs'   # 线性代数异常、范数计算和LAPACK函数获取函数
]


def __dir__():
    return __all__  # 返回模块中所有公共接口的列表


def __getattr__(name):
    # 返回_sub_module_deprecation函数的调用结果，用于处理模块、子模块和属性的弃用警告
    return _sub_module_deprecation(sub_package="linalg", module="decomp",
                                   private_modules=["_decomp"], all=__all__,
                                   attribute=name)
```