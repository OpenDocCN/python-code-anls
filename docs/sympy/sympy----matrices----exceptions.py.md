# `D:\src\scipysrc\sympy\sympy\matrices\exceptions.py`

```
"""
Exceptions raised by the matrix module.
"""

# 定义一个自定义异常类 MatrixError，继承自内置的 Exception 类
class MatrixError(Exception):
    pass

# 定义一个异常类 ShapeError，继承自内置的 ValueError 类和自定义的 MatrixError 类
class ShapeError(ValueError, MatrixError):
    """Wrong matrix shape"""
    pass

# 定义一个异常类 NonSquareMatrixError，继承自 ShapeError 类
class NonSquareMatrixError(ShapeError):
    pass

# 定义一个异常类 NonInvertibleMatrixError，继承自内置的 ValueError 类和自定义的 MatrixError 类
class NonInvertibleMatrixError(ValueError, MatrixError):
    """The matrix in not invertible (division by multidimensional zero error)."""
    pass

# 定义一个异常类 NonPositiveDefiniteMatrixError，继承自内置的 ValueError 类和自定义的 MatrixError 类
class NonPositiveDefiniteMatrixError(ValueError, MatrixError):
    """The matrix is not a positive-definite matrix."""
    pass
```