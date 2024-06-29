# `.\numpy\numpy\_utils\_convertions.py`

```
"""
A set of methods retained from np.compat module that
are still used across codebase.
"""

# 定义导出的方法名列表，这些方法在代码库中仍然被使用
__all__ = ["asunicode", "asbytes"]

# 定义将字节流转换为 Unicode 字符串的方法
def asunicode(s):
    # 如果输入是字节流，则用 Latin-1 解码为 Unicode 字符串
    if isinstance(s, bytes):
        return s.decode('latin1')
    # 如果输入不是字节流，直接转换为字符串并返回
    return str(s)

# 定义将输入转换为字节流的方法
def asbytes(s):
    # 如果输入已经是字节流，直接返回
    if isinstance(s, bytes):
        return s
    # 如果输入不是字节流，将其转换为 Latin-1 编码的字节流并返回
    return str(s).encode('latin1')
```