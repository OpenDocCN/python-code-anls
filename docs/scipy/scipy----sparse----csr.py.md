# `D:\src\scipysrc\scipy\scipy\sparse\csr.py`

```
# 导入 `scipy._lib.deprecation` 模块中的 `_sub_module_deprecation` 函数，
# 用于处理子模块的弃用警告和提示信息。
from scipy._lib.deprecation import _sub_module_deprecation

# 定义一个列表 `__all__`，包含了本模块中需要导出的公共接口函数和类名。
# 此处使用 `noqa: F822` 来忽略 Flake8 检查中的 `undefined name` 错误。
__all__ = [
    'csr_count_blocks',    # 导出函数 `csr_count_blocks`
    'csr_matrix',          # 导出类 `csr_matrix`
    'csr_tobsr',           # 导出函数 `csr_tobsr`
    'csr_tocsc',           # 导出函数 `csr_tocsc`
    'get_csr_submatrix',   # 导出函数 `get_csr_submatrix`
    'isspmatrix_csr',      # 导出函数 `isspmatrix_csr`
    'spmatrix',            # 导出类 `spmatrix`
    'upcast',              # 导出函数 `upcast`
]

# 定义一个特殊的 `__dir__()` 函数，用于返回模块的所有公共接口的列表 `__all__`。
def __dir__():
    return __all__

# 定义一个特殊的 `__getattr__()` 函数，用于处理模块中未定义的属性访问。
# 此函数调用 `_sub_module_deprecation()` 函数，生成有关废弃警告的信息，
# 指示使用者在稀疏矩阵模块中应该使用 `sparse` 子模块的功能。
def __getattr__(name):
    return _sub_module_deprecation(sub_package="sparse", module="csr",
                                   private_modules=["_csr"], all=__all__,
                                   attribute=name)
```