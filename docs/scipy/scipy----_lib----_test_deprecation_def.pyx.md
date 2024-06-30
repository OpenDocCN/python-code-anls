# `D:\src\scipysrc\scipy\scipy\_lib\_test_deprecation_def.pyx`

```
# 定义一个公共的 Cython 函数 `foo`，声明不会抛出异常，返回整数 1
cdef public int foo() noexcept:
    return 1

# 定义一个公共的 Cython 函数 `foo_deprecated`，声明不会抛出异常，返回整数 1
cdef public int foo_deprecated() noexcept:
    return 1

# 从 `scipy._lib.deprecation` 模块中导入 `deprecate_cython_api` 函数
from scipy._lib.deprecation import deprecate_cython_api

# 从 `scipy._lib._test_deprecation_def` 模块中导入 `mod` 对象
import scipy._lib._test_deprecation_def as mod

# 对 `mod` 模块中名为 `foo_deprecated` 的函数进行废弃处理，指定新名称为 `foo`，提供废弃信息
deprecate_cython_api(mod, "foo_deprecated", new_name="foo",
                     message="Deprecated in Scipy 42.0.0")

# 删除不再需要的 `deprecate_cython_api` 和 `mod` 引用，释放内存
del deprecate_cython_api, mod
```