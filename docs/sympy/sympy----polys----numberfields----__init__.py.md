# `D:\src\scipysrc\sympy\sympy\polys\numberfields\__init__.py`

```
"""Defines modules and functions related to computational algebraic field theory."""

# 指定模块中导出的公共接口
__all__ = [
    'minpoly', 'minimal_polynomial',  # 导出的函数：minpoly 和 minimal_polynomial

    'field_isomorphism', 'primitive_element', 'to_number_field',  # 导出的函数：field_isomorphism、primitive_element 和 to_number_field

    'isolate',  # 导出的函数：isolate

    'round_two',  # 导出的函数：round_two

    'prime_decomp', 'prime_valuation',  # 导出的函数：prime_decomp 和 prime_valuation

    'galois_group',  # 导出的函数：galois_group
]

# 从 minpoly 模块中导入 minpoly 和 minimal_polynomial 函数
from .minpoly import minpoly, minimal_polynomial

# 从 subfield 模块中导入 field_isomorphism、primitive_element 和 to_number_field 函数
from .subfield import field_isomorphism, primitive_element, to_number_field

# 从 utilities 模块中导入 isolate 函数
from .utilities import isolate

# 从 basis 模块中导入 round_two 函数
from .basis import round_two

# 从 primes 模块中导入 prime_decomp 和 prime_valuation 函数
from .primes import prime_decomp, prime_valuation

# 从 galoisgroups 模块中导入 galois_group 函数
from .galoisgroups import galois_group
```