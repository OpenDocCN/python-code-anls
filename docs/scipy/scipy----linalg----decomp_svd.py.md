# `D:\src\scipysrc\scipy\scipy\linalg\decomp_svd.py`

```
# 导入从 scipy._lib.deprecation 模块中导入 _sub_module_deprecation 函数
# 该文件不是公共使用的，将在 SciPy v2.0.0 中删除
# 使用 `scipy.linalg` 命名空间来导入以下列出的函数

from scipy._lib.deprecation import _sub_module_deprecation

# 定义一个全局变量 __all__，用于指定模块中公开的函数名列表，禁止检查缺少文档字符串 (noqa: F822)
__all__ = [
    'svd', 'svdvals', 'diagsvd', 'orth', 'subspace_angles', 'null_space',
    'LinAlgError', 'get_lapack_funcs'
]

# 定义一个特殊方法 __dir__()，返回模块中公开的函数名列表 __all__
def __dir__():
    return __all__

# 定义一个特殊方法 __getattr__()，在获取未定义的属性时调用
# 调用 _sub_module_deprecation 函数，标记子模块 'linalg' 中 'decomp_svd' 模块及其私有模块 '_decomp_svd'
# 使用 __all__ 列表中的名称，进行模块级别的废弃警告
def __getattr__(name):
    return _sub_module_deprecation(sub_package="linalg", module="decomp_svd",
                                   private_modules=["_decomp_svd"], all=__all__,
                                   attribute=name)
```