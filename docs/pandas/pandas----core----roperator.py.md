# `D:\src\scipysrc\pandas\pandas\core\roperator.py`

```
"""
stdlib operator 模块中没有提供反向操作。
定义这些函数而不是使用 lambda 允许我们通过名称引用它们。
"""

# 导入标准库中的 operator 模块
import operator

# 定义 radd 函数，实现右操作数加左操作数
def radd(left, right):
    return right + left

# 定义 rsub 函数，实现右操作数减左操作数
def rsub(left, right):
    return right - left

# 定义 rmul 函数，实现右操作数乘左操作数
def rmul(left, right):
    return right * left

# 定义 rdiv 函数，实现右操作数除以左操作数
def rdiv(left, right):
    return right / left

# 定义 rtruediv 函数，实现右操作数真除以左操作数
def rtruediv(left, right):
    return right / left

# 定义 rfloordiv 函数，实现右操作数整除左操作数
def rfloordiv(left, right):
    return right // left

# 定义 rmod 函数，实现右操作数模左操作数的操作
def rmod(left, right):
    # 检查 right 是否为字符串，因为 % 是字符串的格式化操作；这会导致 TypeError
    # 否则执行模运算操作
    if isinstance(right, str):
        typ = type(left).__name__
        raise TypeError(f"{typ} cannot perform the operation mod")
    
    return right % left

# 定义 rdivmod 函数，实现右操作数除以左操作数的商和余数
def rdivmod(left, right):
    return divmod(right, left)

# 定义 rpow 函数，实现右操作数的左操作数次方
def rpow(left, right):
    return right**left

# 定义 rand_ 函数，实现右操作数与左操作数按位与
def rand_(left, right):
    return operator.and_(right, left)

# 定义 ror_ 函数，实现右操作数与左操作数按位或
def ror_(left, right):
    return operator.or_(right, left)

# 定义 rxor 函数，实现右操作数与左操作数按位异或
def rxor(left, right):
    return operator.xor(right, left)
```