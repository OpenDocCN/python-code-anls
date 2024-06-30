# `D:\src\scipysrc\scipy\scipy\sparse\dok.py`

```
# 引入 `_sub_module_deprecation` 函数，用于处理子模块的废弃警告和迁移提示
from scipy._lib.deprecation import _sub_module_deprecation

# 定义导出的所有符号列表，这些符号将会被公开
__all__ = [  # noqa: F822
    'IndexMixin',       # 混合索引类
    'check_shape',      # 检查形状函数
    'dok_matrix',       # 字典偏移坐标矩阵类
    'getdtype',         # 获取数据类型函数
    'isdense',          # 判断是否为稠密矩阵的函数
    'isintlike',        # 判断是否为整数类型的函数
    'isscalarlike',     # 判断是否为标量类型的函数
    'isshape',          # 判断是否为形状的函数
    'isspmatrix_dok',   # 判断是否为字典偏移坐标矩阵的函数
    'itertools',        # Python 标准库中的迭代工具模块
    'spmatrix',         # 稀疏矩阵基类
    'upcast',           # 类型提升函数
    'upcast_scalar',    # 标量类型提升函数
]

# 自定义 __dir__() 函数，使得在调用 dir() 函数时返回定义的所有符号列表
def __dir__():
    return __all__

# 自定义 __getattr__(name) 函数，用于当访问不存在的属性时触发
def __getattr__(name):
    return _sub_module_deprecation(sub_package="sparse", module="dok",
                                   private_modules=["_dok"], all=__all__,
                                   attribute=name)
```