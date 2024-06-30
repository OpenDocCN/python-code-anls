# `D:\src\scipysrc\sympy\sympy\utilities\_compilation\__init__.py`

```
""" This sub-module is private, i.e. external code should not depend on it.

These functions are used by tests run as part of continuous integration.
Once the implementation is mature (it should support the major
platforms: Windows, OS X & Linux) it may become official API which
 may be relied upon by downstream libraries. Until then API may break
without prior notice.

TODO:
- (optionally) clean up after tempfile.mkdtemp()
- cross-platform testing
- caching of compiler choice and intermediate files

"""

# 导入编译、链接和导入字符串相关的函数
from .compilation import compile_link_import_strings, compile_run_strings
# 导入检查 Fortran、C 和 C++ 编译器是否存在的函数
from .availability import has_fortran, has_c, has_cxx

# 模块中公开的函数和变量名列表
__all__ = [
    'compile_link_import_strings', 'compile_run_strings',
    'has_fortran', 'has_c', 'has_cxx',
]
```