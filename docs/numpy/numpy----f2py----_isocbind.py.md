# `.\numpy\numpy\f2py\_isocbind.py`

```py
"""
ISO_C_BINDING maps for f2py2e.
Only required declarations/macros/functions will be used.

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""

# 定义 ISO_C_BINDING 映射字典，用于将 C 类型映射到对应的 Fortran 类型
iso_c_binding_map = {
    'integer': {
        'c_int': 'int',                   # 将 'c_int' 映射到 'int'
        'c_short': 'short',               # 将 'c_short' 映射到 'short'
        'c_long': 'long',                 # 将 'c_long' 映射到 'long'
        'c_long_long': 'long_long',       # 将 'c_long_long' 映射到 'long_long'
        'c_signed_char': 'signed_char',   # 将 'c_signed_char' 映射到 'signed_char'
        'c_size_t': 'unsigned',           # 将 'c_size_t' 映射到 'unsigned'
        'c_int8_t': 'signed_char',        # 将 'c_int8_t' 映射到 'signed_char'
        'c_int16_t': 'short',             # 将 'c_int16_t' 映射到 'short'
        'c_int32_t': 'int',               # 将 'c_int32_t' 映射到 'int'
        'c_int64_t': 'long_long',         # 将 'c_int64_t' 映射到 'long_long'
        'c_int_least8_t': 'signed_char',  # 将 'c_int_least8_t' 映射到 'signed_char'
        'c_int_least16_t': 'short',       # 将 'c_int_least16_t' 映射到 'short'
        'c_int_least32_t': 'int',         # 将 'c_int_least32_t' 映射到 'int'
        'c_int_least64_t': 'long_long',   # 将 'c_int_least64_t' 映射到 'long_long'
        'c_int_fast8_t': 'signed_char',   # 将 'c_int_fast8_t' 映射到 'signed_char'
        'c_int_fast16_t': 'short',        # 将 'c_int_fast16_t' 映射到 'short'
        'c_int_fast32_t': 'int',          # 将 'c_int_fast32_t' 映射到 'int'
        'c_int_fast64_t': 'long_long',    # 将 'c_int_fast64_t' 映射到 'long_long'
        'c_intmax_t': 'long_long',        # 将 'c_intmax_t' 映射到 'long_long'
        'c_intptr_t': 'long',             # 将 'c_intptr_t' 映射到 'long'
        'c_ptrdiff_t': 'long',            # 将 'c_ptrdiff_t' 映射到 'long'
    },
    'real': {
        'c_float': 'float',               # 将 'c_float' 映射到 'float'
        'c_double': 'double',             # 将 'c_double' 映射到 'double'
        'c_long_double': 'long_double'    # 将 'c_long_double' 映射到 'long_double'
    },
    'complex': {
        'c_float_complex': 'complex_float',         # 将 'c_float_complex' 映射到 'complex_float'
        'c_double_complex': 'complex_double',       # 将 'c_double_complex' 映射到 'complex_double'
        'c_long_double_complex': 'complex_long_double'  # 将 'c_long_double_complex' 映射到 'complex_long_double'
    },
    'logical': {
        'c_bool': 'unsigned_char'         # 将 'c_bool' 映射到 'unsigned_char'
    },
    'character': {
        'c_char': 'char'                  # 将 'c_char' 映射到 'char'
    }
}

# TODO: See gh-25229

# 定义空字典 isoc_c2pycode_map 和 isoc_c2py_map，用于后续的映射操作
isoc_c2pycode_map = {}
iso_c2py_map = {}

# 定义空字典 isoc_kindmap，用于将 C 类型映射到对应的 Fortran 类型
isoc_kindmap = {}

# 遍历 iso_c_binding_map 中的每个 Fortran 类型和其对应的 C 类型字典
for fortran_type, c_type_dict in iso_c_binding_map.items():
    # 遍历每个 C 类型及其对应的 Fortran 类型
    for c_type in c_type_dict.keys():
        # 将当前的 C 类型映射到对应的 Fortran 类型
        isoc_kindmap[c_type] = fortran_type
```