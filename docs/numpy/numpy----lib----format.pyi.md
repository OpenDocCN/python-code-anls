# `.\numpy\numpy\lib\format.pyi`

```py
# 导入必要的类型声明
from typing import Any, Literal, Final

# 定义 __all__ 变量，用于声明模块的公开接口
__all__: list[str]

# 定义预期的键集合，使用 Final 标注表示为不可变的常量
EXPECTED_KEYS: Final[set[str]]

# 定义魔术前缀，使用 Final 标注表示为不可变的常量
MAGIC_PREFIX: Final[bytes]

# 定义魔术数的长度，使用 Literal 标注指明为整数 8
MAGIC_LEN: Literal[8]

# 定义数组的对齐方式，使用 Literal 标注指明为整数 64
ARRAY_ALIGN: Literal[64]

# 定义缓冲区的大小，使用 Literal 标注指明为整数 262144，等同于 2 的 18 次方
BUFFER_SIZE: Literal[262144]  # 2**18

# 定义函数 magic，参数为 major 和 minor，但函数体未提供
def magic(major, minor): ...

# 定义函数 read_magic，参数为 fp，但函数体未提供
def read_magic(fp): ...

# 定义函数 dtype_to_descr，参数为 dtype，但函数体未提供
def dtype_to_descr(dtype): ...

# 定义函数 descr_to_dtype，参数为 descr，但函数体未提供
def descr_to_dtype(descr): ...

# 定义函数 header_data_from_array_1_0，参数为 array，但函数体未提供
def header_data_from_array_1_0(array): ...

# 定义函数 write_array_header_1_0，参数为 fp 和 d，但函数体未提供
def write_array_header_1_0(fp, d): ...

# 定义函数 write_array_header_2_0，参数为 fp 和 d，但函数体未提供
def write_array_header_2_0(fp, d): ...

# 定义函数 read_array_header_1_0，参数为 fp，但函数体未提供
def read_array_header_1_0(fp): ...

# 定义函数 read_array_header_2_0，参数为 fp，但函数体未提供
def read_array_header_2_0(fp): ...

# 定义函数 write_array，参数包括 fp, array, version, allow_pickle 和 pickle_kwargs，但函数体未提供
def write_array(fp, array, version=..., allow_pickle=..., pickle_kwargs=...): ...

# 定义函数 read_array，参数包括 fp, allow_pickle 和 pickle_kwargs，但函数体未提供
def read_array(fp, allow_pickle=..., pickle_kwargs=...): ...

# 定义函数 open_memmap，参数包括 filename, mode, dtype, shape, fortran_order 和 version，但函数体未提供
def open_memmap(filename, mode=..., dtype=..., shape=..., fortran_order=..., version=...): ...
```