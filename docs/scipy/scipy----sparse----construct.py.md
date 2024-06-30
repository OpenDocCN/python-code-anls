# `D:\src\scipysrc\scipy\scipy\sparse\construct.py`

```
# This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.sparse` namespace for importing the functions
# included below.

# 从 scipy._lib.deprecation 模块中导入 _sub_module_deprecation 函数
from scipy._lib.deprecation import _sub_module_deprecation

# 定义 __all__ 变量，包含了当前模块中要暴露的所有函数和类名
__all__ = [  # noqa: F822
    'block_diag',        # 将函数 block_diag 加入到 __all__ 列表中
    'bmat',              # 将函数 bmat 加入到 __all__ 列表中
    'bsr_matrix',        # 将类 bsr_matrix 加入到 __all__ 列表中
    'check_random_state',# 将函数 check_random_state 加入到 __all__ 列表中
    'coo_matrix',        # 将类 coo_matrix 加入到 __all__ 列表中
    'csc_matrix',        # 将类 csc_matrix 加入到 __all__ 列表中
    'csr_hstack',        # 将函数 csr_hstack 加入到 __all__ 列表中
    'csr_matrix',        # 将类 csr_matrix 加入到 __all__ 列表中
    'dia_matrix',        # 将类 dia_matrix 加入到 __all__ 列表中
    'diags',             # 将函数 diags 加入到 __all__ 列表中
    'eye',               # 将函数 eye 加入到 __all__ 列表中
    'get_index_dtype',   # 将函数 get_index_dtype 加入到 __all__ 列表中
    'hstack',            # 将函数 hstack 加入到 __all__ 列表中
    'identity',          # 将函数 identity 加入到 __all__ 列表中
    'isscalarlike',      # 将函数 isscalarlike 加入到 __all__ 列表中
    'issparse',          # 将函数 issparse 加入到 __all__ 列表中
    'kron',              # 将函数 kron 加入到 __all__ 列表中
    'kronsum',           # 将函数 kronsum 加入到 __all__ 列表中
    'numbers',           # 将函数 numbers 加入到 __all__ 列表中
    'rand',              # 将函数 rand 加入到 __all__ 列表中
    'random',            # 将模块 random 加入到 __all__ 列表中
    'rng_integers',      # 将函数 rng_integers 加入到 __all__ 列表中
    'spdiags',           # 将函数 spdiags 加入到 __all__ 列表中
    'upcast',            # 将函数 upcast 加入到 __all__ 列表中
    'vstack',            # 将函数 vstack 加入到 __all__ 列表中
]

# 定义 __dir__ 函数，返回当前模块中要公开的所有函数和类名
def __dir__():
    return __all__

# 定义 __getattr__ 函数，用于处理动态属性访问，当访问的属性不存在时会调用 _sub_module_deprecation 函数
def __getattr__(name):
    return _sub_module_deprecation(sub_package="sparse", module="construct",
                                   private_modules=["_construct"], all=__all__,
                                   attribute=name)
```