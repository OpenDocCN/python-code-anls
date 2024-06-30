# `D:\src\scipysrc\scipy\scipy\sparse\csc.py`

```
# 导入从SciPy的私有库_scipy._lib.deprecation中导入_sub_module_deprecation函数，
# 该函数用于处理子模块的弃用警告和转向。
from scipy._lib.deprecation import _sub_module_deprecation

# 设置导出的模块列表，用于定义当前模块中公开的函数和类的名称。
__all__ = [  # noqa: F822
    'csc_matrix',    # 稀疏压缩列矩阵的类
    'csc_tocsr',     # 将稀疏压缩列矩阵转换为压缩行矩阵的函数
    'expandptr',     # 未详细说明，可能用于扩展指针的功能
    'isspmatrix_csc',# 判断是否为稀疏压缩列矩阵的函数
    'spmatrix',      # 稀疏矩阵的基类
    'upcast',        # 类型上升转换的函数
]

# 定义__dir__()函数，该函数定义了当前模块中的公开接口列表。
def __dir__():
    return __all__

# 定义__getattr__(name)函数，用于动态获取模块中的属性，通过_sub_module_deprecation函数
# 处理稀疏矩阵模块的属性访问，生成相应的弃用警告信息。
def __getattr__(name):
    return _sub_module_deprecation(sub_package="sparse", module="csc",
                                   private_modules=["_csc"], all=__all__,
                                   attribute=name)
```