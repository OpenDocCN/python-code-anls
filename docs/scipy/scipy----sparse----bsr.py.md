# `D:\src\scipysrc\scipy\scipy\sparse\bsr.py`

```
# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.sparse` namespace for importing the functions
# included below.

# 从 scipy._lib.deprecation 模块导入 _sub_module_deprecation 函数
from scipy._lib.deprecation import _sub_module_deprecation

# 定义一个包含以下函数名称的列表，用于指定模块的公开接口
__all__ = [  # noqa: F822
    'bsr_matmat',        # 稀疏矩阵 BSR 格式的矩阵乘法
    'bsr_matrix',         # 创建稀疏矩阵 BSR 格式的对象
    'bsr_matvec',         # 稀疏矩阵 BSR 格式的矩阵向量乘法
    'bsr_matvecs',        # 稀疏矩阵 BSR 格式的批量矩阵向量乘法
    'bsr_sort_indices',   # 对稀疏矩阵 BSR 格式的索引进行排序
    'bsr_tocsr',          # 将稀疏矩阵 BSR 格式转换为 CSR 格式
    'bsr_transpose',      # 对稀疏矩阵 BSR 格式进行转置
    'check_shape',        # 检查矩阵形状的函数
    'csr_matmat_maxnnz',  # CSR 格式矩阵乘法中的最大非零元素数
    'getdata',            # 获取稀疏矩阵数据的函数
    'getdtype',           # 获取稀疏矩阵数据类型的函数
    'isshape',            # 检查矩阵形状的函数
    'isspmatrix_bsr',     # 检查对象是否为稀疏矩阵 BSR 格式的函数
    'spmatrix',           # 稀疏矩阵的基类
    'to_native',          # 将对象转换为本地类型的函数
    'upcast',             # 向上转换数据类型的函数
    'warn',               # 发出警告消息的函数
]

# 定义一个特殊的函数 __dir__()，返回当前模块的公开接口列表 __all__
def __dir__():
    return __all__

# 定义一个特殊的函数 __getattr__()，用于动态获取模块属性
def __getattr__(name):
    # 调用 _sub_module_deprecation 函数，发出关于稀疏矩阵 BSR 格式过时使用的警告消息
    return _sub_module_deprecation(sub_package="sparse", module="bsr",
                                   private_modules=["_bsr"], all=__all__,
                                   attribute=name)
```