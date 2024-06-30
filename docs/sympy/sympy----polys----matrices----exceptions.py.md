# `D:\src\scipysrc\sympy\sympy\polys\matrices\exceptions.py`

```
"""
Module to define exceptions to be used in sympy.polys.matrices modules and
classes.

Ideally all exceptions raised in these modules would be defined and documented
here and not e.g. imported from matrices. Also ideally generic exceptions like
ValueError/TypeError would not be raised anywhere.

"""

# 定义一个基础类，表示由 DomainMatrix 引发的错误
class DMError(Exception):
    """Base class for errors raised by DomainMatrix"""
    pass

# 表示由于输入错误导致的异常
class DMBadInputError(DMError):
    """list of lists is inconsistent with shape"""
    pass

# 表示域不匹配导致的异常
class DMDomainError(DMError):
    """domains do not match"""
    pass

# 表示域不是一个字段导致的异常
class DMNotAField(DMDomainError):
    """domain is not a field"""
    pass

# 表示混合稠密/稀疏矩阵格式不受支持导致的异常
class DMFormatError(DMError):
    """mixed dense/sparse not supported"""
    pass

# 表示矩阵不可逆导致的异常
class DMNonInvertibleMatrixError(DMError):
    """The matrix in not invertible"""
    pass

# 表示矩阵的秩不符合预期导致的异常
class DMRankError(DMError):
    """matrix does not have expected rank"""
    pass

# 表示矩阵形状不一致导致的异常
class DMShapeError(DMError):
    """shapes are inconsistent"""
    pass

# 表示矩阵不是方阵导致的异常
class DMNonSquareMatrixError(DMShapeError):
    """The matrix is not square"""
    pass

# 表示传递的值无效导致的异常
class DMValueError(DMError):
    """The value passed is invalid"""
    pass

# 模块导出的异常类列表，用于控制导出的符号
__all__ = [
    'DMError', 'DMBadInputError', 'DMDomainError', 'DMFormatError',
    'DMRankError', 'DMShapeError', 'DMNotAField',
    'DMNonInvertibleMatrixError', 'DMNonSquareMatrixError', 'DMValueError'
]
```