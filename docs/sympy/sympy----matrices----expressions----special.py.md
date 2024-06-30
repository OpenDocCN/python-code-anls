# `D:\src\scipysrc\sympy\sympy\matrices\expressions\special.py`

```
# 导入模块中的函数和类
from sympy.assumptions.ask import ask, Q
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.sympify import _sympify
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.exceptions import NonInvertibleMatrixError
from .matexpr import MatrixExpr  # 导入相对路径中的 matexpr 模块中的 MatrixExpr 类


class ZeroMatrix(MatrixExpr):
    """The Matrix Zero 0 - additive identity

    Examples
    ========

    >>> from sympy import MatrixSymbol, ZeroMatrix
    >>> A = MatrixSymbol('A', 3, 5)
    >>> Z = ZeroMatrix(3, 5)
    >>> A + Z
    A
    >>> Z*A.T
    0
    """
    is_ZeroMatrix = True  # 设置类属性 is_ZeroMatrix 为 True，表示这是一个 ZeroMatrix 实例

    def __new__(cls, m, n):
        m, n = _sympify(m), _sympify(n)  # 将 m 和 n 转换为 SymPy 对象
        cls._check_dim(m)  # 调用类方法 _check_dim 检查 m 的维度
        cls._check_dim(n)  # 调用类方法 _check_dim 检查 n 的维度

        return super().__new__(cls, m, n)  # 调用父类 MatrixExpr 的 __new__ 方法创建实例

    @property
    def shape(self):
        return (self.args[0], self.args[1])  # 返回矩阵的形状 (m, n)

    def _eval_power(self, exp):
        # exp = -1, 0, 1 在此阶段已经处理过
        if (exp < 0) == True:  # 如果指数 exp 小于 0
            raise NonInvertibleMatrixError("Matrix det == 0; not invertible")  # 抛出非可逆矩阵异常
        return self  # 返回当前矩阵对象本身

    def _eval_transpose(self):
        return ZeroMatrix(self.cols, self.rows)  # 返回当前矩阵的转置矩阵，即行列互换

    def _eval_adjoint(self):
        return ZeroMatrix(self.cols, self.rows)  # 返回当前矩阵的伴随矩阵，即转置矩阵

    def _eval_trace(self):
        return S.Zero  # 返回零标量，表示矩阵的迹为零

    def _eval_determinant(self):
        return S.Zero  # 返回零标量，表示矩阵的行列式为零

    def _eval_inverse(self):
        raise NonInvertibleMatrixError("Matrix det == 0; not invertible.")  # 抛出非可逆矩阵异常

    def _eval_as_real_imag(self):
        return (self, self)  # 返回自身作为实部和虚部的元组

    def _eval_conjugate(self):
        return self  # 返回当前矩阵的共轭，对于零矩阵而言仍是零矩阵

    def _entry(self, i, j, **kwargs):
        return S.Zero  # 返回零标量，表示矩阵的第 (i, j) 元素为零


class GenericZeroMatrix(ZeroMatrix):
    """
    A zero matrix without a specified shape

    This exists primarily so MatAdd() with no arguments can return something
    meaningful.
    """
    def __new__(cls):
        # 使用 super(ZeroMatrix, cls) 而不是 super(GenericZeroMatrix, cls)
        # 因为 ZeroMatrix.__new__ 没有相同的签名
        return super(ZeroMatrix, cls).__new__(cls)

    @property
    def rows(self):
        raise TypeError("GenericZeroMatrix does not have a specified shape")  # 抛出类型错误，因为没有指定形状

    @property
    def cols(self):
        raise TypeError("GenericZeroMatrix does not have a specified shape")  # 抛出类型错误，因为没有指定形状

    @property
    def shape(self):
        raise TypeError("GenericZeroMatrix does not have a specified shape")  # 抛出类型错误，因为没有指定形状

    def __eq__(self, other):
        return isinstance(other, GenericZeroMatrix)  # 判断是否与另一个 GenericZeroMatrix 实例相等

    def __ne__(self, other):
        return not (self == other)  # 判断是否与另一个 GenericZeroMatrix 实例不相等

    def __hash__(self):
        return super().__hash__()  # 返回基类的哈希值
        

class Identity(MatrixExpr):
    """The Matrix Identity I - multiplicative identity

    Examples
    ========

    >>> from sympy import Identity, MatrixSymbol
    >>> A = MatrixSymbol('A', 3, 5)
    >>> I = Identity(3)
    >>> I*A
    A
    """

    is_Identity = True  # 设置类属性 is_Identity 为 True，表示这是一个 Identity 实例
    # 定义一个特殊方法 __new__，用于创建新的对象实例
    def __new__(cls, n):
        # 将输入参数 n 转换为符号表达式（sympy 的对象）
        n = _sympify(n)
        # 调用类方法 _check_dim 来检查维度是否有效
        cls._check_dim(n)

        # 调用父类的 __new__ 方法创建对象实例并返回
        return super().__new__(cls, n)

    # 定义一个属性方法 rows，用于返回对象的行数
    @property
    def rows(self):
        return self.args[0]

    # 定义一个属性方法 cols，用于返回对象的列数
    @property
    def cols(self):
        return self.args[0]

    # 定义一个属性方法 shape，返回对象的形状，这里是一个方阵，行数和列数相同
    @property
    def shape(self):
        return (self.args[0], self.args[0])

    # 定义一个属性方法 is_square，判断对象是否为方阵，始终返回 True
    @property
    def is_square(self):
        return True

    # 定义一个方法 _eval_transpose，返回对象的转置，对于方阵来说是自身
    def _eval_transpose(self):
        return self

    # 定义一个方法 _eval_trace，返回对象的迹（对角元素之和），对于方阵来说是行数
    def _eval_trace(self):
        return self.rows

    # 定义一个方法 _eval_inverse，返回对象的逆矩阵，对于单位矩阵来说是自身
    def _eval_inverse(self):
        return self

    # 定义一个方法 _eval_as_real_imag，返回对象的实部和虚部，对于单位矩阵来说实部是自身，虚部是零矩阵
    def _eval_as_real_imag(self):
        return (self, ZeroMatrix(*self.shape))

    # 定义一个方法 _eval_conjugate，返回对象的共轭，对于单位矩阵来说是自身
    def _eval_conjugate(self):
        return self

    # 定义一个方法 _eval_adjoint，返回对象的伴随矩阵，对于单位矩阵来说是自身
    def _eval_adjoint(self):
        return self

    # 定义一个方法 _entry，返回矩阵中第 i 行第 j 列的元素值
    def _entry(self, i, j, **kwargs):
        # 判断 i 是否等于 j，如果是返回 1，否则返回 0
        eq = Eq(i, j)
        if eq is S.true:
            return S.One
        elif eq is S.false:
            return S.Zero
        # 如果 i 不等于 j，则返回克罗内克 δ 函数，用于描述对角元素为 1，非对角元素为 0 的情况
        return KroneckerDelta(i, j, (0, self.cols-1))

    # 定义一个方法 _eval_determinant，返回对象的行列式，对于单位矩阵来说是 1
    def _eval_determinant(self):
        return S.One

    # 定义一个方法 _eval_power，返回对象的乘幂结果，对于单位矩阵来说是自身
    def _eval_power(self, exp):
        return self
class GenericIdentity(Identity):
    """
    An identity matrix without a specified shape

    This exists primarily so MatMul() with no arguments can return something
    meaningful.
    """

    def __new__(cls):
        # 使用 super(Identity, cls) 而不是 super(GenericIdentity, cls)，因为
        # Identity.__new__ 的签名不同
        return super(Identity, cls).__new__(cls)

    @property
    def rows(self):
        # 抛出异常，因为 GenericIdentity 没有指定的形状
        raise TypeError("GenericIdentity does not have a specified shape")

    @property
    def cols(self):
        # 抛出异常，因为 GenericIdentity 没有指定的形状
        raise TypeError("GenericIdentity does not have a specified shape")

    @property
    def shape(self):
        # 抛出异常，因为 GenericIdentity 没有指定的形状
        raise TypeError("GenericIdentity does not have a specified shape")

    @property
    def is_square(self):
        # 返回 True，表示这是一个方阵
        return True

    # 避免调用 .shape 的 Matrix.__eq__
    def __eq__(self, other):
        # 返回是否是 GenericIdentity 类的实例
        return isinstance(other, GenericIdentity)

    def __ne__(self, other):
        # 返回是否不是 GenericIdentity 类的实例
        return not (self == other)

    def __hash__(self):
        # 调用父类的 __hash__ 方法
        return super().__hash__()


class OneMatrix(MatrixExpr):
    """
    Matrix whose all entries are ones.
    """

    def __new__(cls, m, n, evaluate=False):
        # 将 m 和 n 转换为 SymPy 表达式
        m, n = _sympify(m), _sympify(n)
        # 检查维度是否有效
        cls._check_dim(m)
        cls._check_dim(n)

        if evaluate:
            # 检查是否是 1x1 矩阵
            condition = Eq(m, 1) & Eq(n, 1)
            if condition == True:
                # 返回一个 1x1 的 Identity 矩阵
                return Identity(1)

        # 调用父类的 __new__ 方法创建对象
        obj = super().__new__(cls, m, n)
        return obj

    @property
    def shape(self):
        # 返回矩阵的形状
        return self._args

    @property
    def is_Identity(self):
        # 返回是否是 1x1 矩阵
        return self._is_1x1() == True

    def as_explicit(self):
        # 导入 ImmutableDenseMatrix 并返回一个全为 1 的矩阵
        from sympy.matrices.immutable import ImmutableDenseMatrix
        return ImmutableDenseMatrix.ones(*self.shape)

    def doit(self, **hints):
        # 如果 hints 中的 'deep' 是 True，则递归地调用 doit 方法
        args = self.args
        if hints.get('deep', True):
            args = [a.doit(**hints) for a in args]
        return self.func(*args, evaluate=True)

    def _eval_power(self, exp):
        # 处理指数为 -1, 0, 1 的情况
        if self._is_1x1() == True:
            # 返回一个 1x1 的 Identity 矩阵
            return Identity(1)
        if (exp < 0) == True:
            # 抛出异常，因为矩阵的行列式为 0，不可逆
            raise NonInvertibleMatrixError("Matrix det == 0; not invertible")
        if ask(Q.integer(exp)):
            # 如果指数是整数，则返回相应的幂次方结果
            return self.shape[0] ** (exp - 1) * OneMatrix(*self.shape)
        # 调用父类的 _eval_power 方法
        return super()._eval_power(exp)

    def _eval_transpose(self):
        # 返回矩阵的转置
        return OneMatrix(self.cols, self.rows)

    def _eval_adjoint(self):
        # 返回矩阵的伴随矩阵
        return OneMatrix(self.cols, self.rows)

    def _eval_trace(self):
        # 返回矩阵的迹（元素的和）
        return S.One * self.rows

    def _is_1x1(self):
        """Returns true if the matrix is known to be 1x1"""
        # 获取矩阵的形状
        shape = self.shape
        # 返回是否是 1x1 矩阵
        return Eq(shape[0], 1) & Eq(shape[1], 1)
    # 计算矩阵的行列式值
    def _eval_determinant(self):
        # 检查是否为 1x1 矩阵
        condition = self._is_1x1()
        if condition == True:
            # 如果是 1x1 矩阵，返回 1
            return S.One
        elif condition == False:
            # 如果不是 1x1 矩阵，返回 0
            return S.Zero
        else:
            # 导入矩阵行列式计算类，并返回计算结果
            from sympy.matrices.expressions.determinant import Determinant
            return Determinant(self)

    # 计算矩阵的逆
    def _eval_inverse(self):
        # 检查是否为 1x1 矩阵
        condition = self._is_1x1()
        if condition == True:
            # 如果是 1x1 矩阵，返回单位矩阵
            return Identity(1)
        elif condition == False:
            # 如果不是 1x1 矩阵，抛出异常，说明矩阵不可逆
            raise NonInvertibleMatrixError("Matrix det == 0; not invertible.")
        else:
            # 导入矩阵逆计算类，并返回计算结果
            from .inverse import Inverse
            return Inverse(self)

    # 将矩阵转换为实部和虚部的形式
    def _eval_as_real_imag(self):
        # 返回矩阵本身和与其形状相同的零矩阵
        return (self, ZeroMatrix(*self.shape))

    # 返回矩阵的共轭
    def _eval_conjugate(self):
        # 返回矩阵本身，因为实矩阵的共轭就是其自身
        return self

    # 返回矩阵指定位置的元素
    def _entry(self, i, j, **kwargs):
        # 对于任意位置 (i, j)，返回单位元素 1
        return S.One
```