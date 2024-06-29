# `.\numpy\numpy\_core\umath.py`

```py
"""
Create the numpy._core.umath namespace for backward compatibility. In v1.16
the multiarray and umath c-extension modules were merged into a single
_multiarray_umath extension module. So we replicate the old namespace
by importing from the extension module.

"""

# 导入整个 numpy 包
import numpy
# 从当前包中导入 _multiarray_umath 模块
from . import _multiarray_umath
# 导入 _multiarray_umath 模块中的所有内容，忽略 F403 错误
from ._multiarray_umath import *  # noqa: F403

# 这些导入是为了保持向后兼容性，不要更改它们。问题 gh-11862
# _ones_like 是半公开的，故意未添加到 __all__ 中
# 从 _multiarray_umath 模块中导入特定函数和变量
from ._multiarray_umath import (
    _UFUNC_API, _add_newdoc_ufunc, _ones_like, _get_extobj_dict, _make_extobj,
    _extobj_contextvar)

# 这些导入是为了 strip 和 replace 实现
# 从 _multiarray_umath 模块中导入更多的函数
from ._multiarray_umath import (
    _replace, _strip_whitespace, _lstrip_whitespace, _rstrip_whitespace,
    _strip_chars, _lstrip_chars, _rstrip_chars, _expandtabs_length,
    _expandtabs, _center, _ljust, _rjust, _zfill, _partition, _partition_index,
    _rpartition, _rpartition_index)

# 定义 __all__ 列表，包含需要公开的函数和变量名
__all__ = [
    'absolute', 'add',
    'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan', 'arctan2', 'arctanh',
    'bitwise_and', 'bitwise_or', 'bitwise_xor', 'cbrt', 'ceil', 'conj',
    'conjugate', 'copysign', 'cos', 'cosh', 'bitwise_count', 'deg2rad',
    'degrees', 'divide', 'divmod', 'e', 'equal', 'euler_gamma', 'exp', 'exp2',
    'expm1', 'fabs', 'floor', 'floor_divide', 'float_power', 'fmax', 'fmin',
    'fmod', 'frexp', 'frompyfunc', 'gcd', 'greater', 'greater_equal',
    'heaviside', 'hypot', 'invert', 'isfinite', 'isinf', 'isnan', 'isnat',
    'lcm', 'ldexp', 'left_shift', 'less', 'less_equal', 'log', 'log10',
    'log1p', 'log2', 'logaddexp', 'logaddexp2', 'logical_and', 'logical_not',
    'logical_or', 'logical_xor', 'maximum', 'minimum', 'mod', 'modf',
    'multiply', 'negative', 'nextafter', 'not_equal', 'pi', 'positive',
    'power', 'rad2deg', 'radians', 'reciprocal', 'remainder', 'right_shift',
    'rint', 'sign', 'signbit', 'sin', 'sinh', 'spacing', 'sqrt', 'square',
    'subtract', 'tan', 'tanh', 'true_divide', 'trunc']
```