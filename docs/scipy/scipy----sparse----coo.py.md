# `D:\src\scipysrc\scipy\scipy\sparse\coo.py`

```
# 本文件不适用于公共使用，并且将在 SciPy v2.0.0 中删除。
# 使用 `scipy.sparse` 命名空间来导入下面列出的函数。

# 从 `scipy._lib.deprecation` 模块中导入 `_sub_module_deprecation` 函数
from scipy._lib.deprecation import _sub_module_deprecation

# 定义一个公共变量 `__all__`，包含了可以从当前模块导入的所有公共符号
__all__ = [  # noqa: F822
    'SparseEfficiencyWarning',  # 稀疏矩阵效率警告类
    'check_reshape_kwargs',     # 检查重塑参数的函数
    'check_shape',              # 检查形状的函数
    'coo_matrix',               # COO 稀疏矩阵的类
    'coo_matvec',               # COO 矩阵向量乘法函数
    'coo_tocsr',                # 将 COO 矩阵转换为 CSR 格式的函数
    'coo_todense',              # 将 COO 矩阵转换为稠密矩阵的函数
    'downcast_intp_index',      # 将整数类型的索引向下转换为较小的整数类型的函数
    'getdata',                  # 获取稀疏矩阵数据的函数
    'getdtype',                 # 获取稀疏矩阵数据类型的函数
    'isshape',                  # 检查是否为合法形状的函数
    'isspmatrix_coo',           # 检查是否为 COO 格式稀疏矩阵的函数
    'operator',                 # 运算符模块
    'spmatrix',                 # 稀疏矩阵的基类
    'to_native',                # 将对象转换为其本地表示形式的函数
    'upcast',                   # 向上转换数据类型的函数
    'upcast_char',              # 向上转换字符类型的函数
    'warn',                     # 发出警告的函数
]

# 定义一个特殊的 `__dir__()` 函数，返回当前模块中定义的公共符号列表 `__all__`
def __dir__():
    return __all__

# 定义一个特殊的 `__getattr__(name)` 函数，用于在当前模块中动态获取属性
def __getattr__(name):
    # 调用 `_sub_module_deprecation` 函数，生成一个关于稀疏子模块的过时警告
    return _sub_module_deprecation(sub_package="sparse", module="coo",
                                   private_modules=["_coo"], all=__all__,
                                   attribute=name)
```