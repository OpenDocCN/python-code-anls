# `D:\src\scipysrc\sympy\sympy\matrices\expressions\sets.py`

```
from sympy.core.assumptions import check_assumptions
from sympy.core.logic import fuzzy_and
from sympy.core.sympify import _sympify
from sympy.matrices.kind import MatrixKind
from sympy.sets.sets import Set, SetKind
from sympy.core.kind import NumberKind
from .matexpr import MatrixExpr

class MatrixSet(Set):
    """
    MatrixSet represents the set of matrices with ``shape = (n, m)`` over the
    given set.

    Examples
    ========

    >>> from sympy.matrices import MatrixSet
    >>> from sympy import S, I, Matrix
    >>> M = MatrixSet(2, 2, set=S.Reals)
    >>> X = Matrix([[1, 2], [3, 4]])
    >>> X in M
    True
    >>> X = Matrix([[1, 2], [I, 4]])
    >>> X in M
    False

    """
    is_empty = False  # 设置类属性，表示此类不为空

    def __new__(cls, n, m, set):
        n, m, set = _sympify(n), _sympify(m), _sympify(set)
        cls._check_dim(n)  # 检查维度 n 是否合法
        cls._check_dim(m)  # 检查维度 m 是否合法
        if not isinstance(set, Set):
            raise TypeError("{} should be an instance of Set.".format(set))
        return Set.__new__(cls, n, m, set)

    @property
    def shape(self):
        return self.args[:2]  # 返回矩阵集合的形状 (n, m)

    @property
    def set(self):
        return self.args[2]  # 返回矩阵集合的基础集合对象

    def _contains(self, other):
        if not isinstance(other, MatrixExpr):
            raise TypeError("{} should be an instance of MatrixExpr.".format(other))
        if other.shape != self.shape:  # 检查矩阵形状是否与集合要求相符
            are_symbolic = any(_sympify(x).is_Symbol for x in other.shape + self.shape)
            if are_symbolic:
                return None  # 如果形状包含符号，则无法确定包含关系
            return False  # 否则返回不包含
        return fuzzy_and(self.set.contains(x) for x in other)  # 使用模糊逻辑检查每个元素是否属于基础集合

    @classmethod
    def _check_dim(cls, dim):
        """Helper function to check invalid matrix dimensions"""
        ok = not dim.is_Float and check_assumptions(
            dim, integer=True, nonnegative=True)  # 检查维度是否为非负整数
        if ok is False:
            raise ValueError(
                "The dimension specification {} should be "
                "a nonnegative integer.".format(dim))  # 如果维度不合法，则引发值错误异常

    def _kind(self):
        return SetKind(MatrixKind(NumberKind))  # 返回矩阵集合的类型信息
```