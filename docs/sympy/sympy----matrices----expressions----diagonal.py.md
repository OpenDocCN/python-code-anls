# `D:\src\scipysrc\sympy\sympy\matrices\expressions\diagonal.py`

```
# 导入_sympify函数，用于将输入转换为Sympy表达式
from sympy.core.sympify import _sympify

# 导入MatrixExpr类，用于创建矩阵表达式
from sympy.matrices.expressions import MatrixExpr

# 导入S、Eq、Ge类，用于处理Sympy中的符号、等式和不等式
from sympy.core import S, Eq, Ge

# 导入Mul类，用于处理Sympy中的乘法运算
from sympy.core.mul import Mul

# 导入KroneckerDelta函数，用于表示Kronecker Delta符号
from sympy.functions.special.tensor_functions import KroneckerDelta

# 定义DiagonalMatrix类，继承自MatrixExpr，表示对角矩阵
class DiagonalMatrix(MatrixExpr):
    """DiagonalMatrix(M) will create a matrix expression that
    behaves as though all off-diagonal elements,
    `M[i, j]` where `i != j`, are zero.

    Examples
    ========

    >>> from sympy import MatrixSymbol, DiagonalMatrix, Symbol
    >>> n = Symbol('n', integer=True)
    >>> m = Symbol('m', integer=True)
    >>> D = DiagonalMatrix(MatrixSymbol('x', 2, 3))
    >>> D[1, 2]
    0
    >>> D[1, 1]
    x[1, 1]

    The length of the diagonal -- the lesser of the two dimensions of `M` --
    is accessed through the `diagonal_length` property:

    >>> D.diagonal_length
    2
    >>> DiagonalMatrix(MatrixSymbol('x', n + 1, n)).diagonal_length
    n

    When one of the dimensions is symbolic the other will be treated as
    though it is smaller:

    >>> tall = DiagonalMatrix(MatrixSymbol('x', n, 3))
    >>> tall.diagonal_length
    3
    >>> tall[10, 1]
    0

    When the size of the diagonal is not known, a value of None will
    be returned:

    >>> DiagonalMatrix(MatrixSymbol('x', n, m)).diagonal_length is None
    True

    """
    # 获取arg属性，返回当前对象的第一个参数，即原始矩阵表达式M
    arg = property(lambda self: self.args[0])

    # 获取shape属性，返回当前对象的形状，即原始矩阵M的形状
    shape = property(lambda self: self.arg.shape)  # type:ignore

    # 定义diagonal_length属性，返回对角线长度
    @property
    def diagonal_length(self):
        r, c = self.shape
        # 如果行数和列数都是整数，则取较小的值作为对角线长度
        if r.is_Integer and c.is_Integer:
            m = min(r, c)
        # 如果其中一维是整数，另一维不是，则取整数维作为对角线长度
        elif r.is_Integer and not c.is_Integer:
            m = r
        elif c.is_Integer and not r.is_Integer:
            m = c
        # 如果行数和列数相等，则取任意维度作为对角线长度
        elif r == c:
            m = r
        # 如果无法确定对角线长度，则返回None
        else:
            try:
                m = min(r, c)
            except TypeError:
                m = None
        return m

    # 定义_entry方法，返回对角矩阵的元素
    def _entry(self, i, j, **kwargs):
        # 如果对角线长度不为None且索引超出对角线长度，则返回0
        if self.diagonal_length is not None:
            if Ge(i, self.diagonal_length) is S.true:
                return S.Zero
            elif Ge(j, self.diagonal_length) is S.true:
                return S.Zero
        # 如果i等于j，则返回原始矩阵M的对应对角线元素
        eq = Eq(i, j)
        if eq is S.true:
            return self.arg[i, i]
        # 如果i不等于j，则返回0
        elif eq is S.false:
            return S.Zero
        # 否则返回原始矩阵M的对应非对角线元素乘以Kronecker Delta符号
        return self.arg[i, j]*KroneckerDelta(i, j)


# 定义DiagonalOf类，继承自MatrixExpr，表示矩阵的对角线
class DiagonalOf(MatrixExpr):
    """DiagonalOf(M) will create a matrix expression that
    is equivalent to the diagonal of `M`, represented as
    a single column matrix.

    Examples
    ========

    >>> from sympy import MatrixSymbol, DiagonalOf, Symbol
    >>> n = Symbol('n', integer=True)
    >>> m = Symbol('m', integer=True)
    >>> x = MatrixSymbol('x', 2, 3)
    >>> diag = DiagonalOf(x)
    >>> diag.shape
    (2, 1)

    The diagonal can be addressed like a matrix or vector and will
    return the corresponding element of the original matrix:

    >>> diag[1, 0] == diag[1] == x[1, 1]
    True

    """
    # 获取arg属性，返回当前对象的第一个参数，即原始矩阵表达式M
    arg = property(lambda self: self.args[0])

    # 获取shape属性，返回当前对象的形状，即原始矩阵M的形状
    shape = property(lambda self: (self.arg.shape[0], 1))

    # 定义_entry方法，返回对角线元素
    def _entry(self, i, j, **kwargs):
        # 返回原始矩阵M对应位置的元素
        return self.arg[i, i]
    """
    arg = property(lambda self: self.args[0])
    """

    # 定义一个属性 `arg`，返回对象的第一个参数
    arg = property(lambda self: self.args[0])

    """
    @property
    def shape(self):
        r, c = self.arg.shape
        if r.is_Integer and c.is_Integer:
            m = min(r, c)
        elif r.is_Integer and not c.is_Integer:
            m = r
        elif c.is_Integer and not r.is_Integer:
            m = c
        elif r == c:
            m = r
        else:
            try:
                m = min(r, c)
            except TypeError:
                m = None
        return m, S.One
    """

    # 定义一个 `shape` 属性，返回对象的形状信息
    @property
    def shape(self):
        r, c = self.arg.shape
        if r.is_Integer and c.is_Integer:
            m = min(r, c)  # 取较小的维度值
        elif r.is_Integer and not c.is_Integer:
            m = r  # 若只有一个维度是整数，取该维度值
        elif c.is_Integer and not r.is_Integer:
            m = c  # 若只有一个维度是整数，取该维度值
        elif r == c:
            m = r  # 若两个维度相等，取任意一个
        else:
            try:
                m = min(r, c)  # 尝试取较小的维度值
            except TypeError:
                m = None  # 出现类型错误时返回 None
        return m, S.One  # 返回形状信息，第二维度默认为 1

    """
    @property
    def diagonal_length(self):
        return self.shape[0]
    """

    # 定义一个 `diagonal_length` 属性，返回对象的对角线长度
    @property
    def diagonal_length(self):
        return self.shape[0]

    """
    def _entry(self, i, j, **kwargs):
        return self.arg._entry(i, i, **kwargs)
    """

    # 定义 `_entry` 方法，返回对象的特定元素
    def _entry(self, i, j, **kwargs):
        return self.arg._entry(i, i, **kwargs)
class DiagMatrix(MatrixExpr):
    """
    Turn a vector into a diagonal matrix.
    """

    # 构造函数，将向量转换为对角矩阵表达式
    def __new__(cls, vector):
        # 将输入的向量转换为符号表达式
        vector = _sympify(vector)
        # 创建一个新的矩阵表达式对象
        obj = MatrixExpr.__new__(cls, vector)
        # 获取向量的形状信息
        shape = vector.shape
        # 确定矩阵的维度
        dim = shape[1] if shape[0] == 1 else shape[0]
        # 如果向量是列向量，则设置属性 _iscolumn 为 True
        if vector.shape[0] != 1:
            obj._iscolumn = True
        else:
            obj._iscolumn = False
        # 设置矩阵的形状属性为 (dim, dim)
        obj._shape = (dim, dim)
        # 设置矩阵的原始向量属性
        obj._vector = vector
        return obj

    @property
    # 返回矩阵的形状
    def shape(self):
        return self._shape

    # 计算矩阵的元素（i, j）处的值
    def _entry(self, i, j, **kwargs):
        # 如果是列向量，则获取第 i 行元素
        if self._iscolumn:
            result = self._vector._entry(i, 0, **kwargs)
        else:
            result = self._vector._entry(0, j, **kwargs)
        # 如果 i 不等于 j，则乘以 KroneckerDelta(i, j)
        if i != j:
            result *= KroneckerDelta(i, j)
        return result

    # 矩阵的转置操作
    def _eval_transpose(self):
        return self

    # 将对角矩阵转换为显式矩阵
    def as_explicit(self):
        from sympy.matrices.dense import diag
        return diag(*list(self._vector.as_explicit()))

    # 执行对角矩阵的操作
    def doit(self, **hints):
        from sympy.assumptions import ask, Q
        from sympy.matrices.expressions.matmul import MatMul
        from sympy.matrices.expressions.transpose import Transpose
        from sympy.matrices.dense import eye
        from sympy.matrices.matrixbase import MatrixBase
        vector = self._vector
        # 处理对角矩阵特殊情况（如形状为 (1, 1) 和单位矩阵等）
        if ask(Q.diagonal(vector)):
            return vector
        # 处理矩阵基类 MatrixBase 的情况
        if isinstance(vector, MatrixBase):
            # 创建一个单位矩阵，维度为 vector 的最大形状
            ret = eye(max(vector.shape))
            # 填充对角元素为 vector 的元素
            for i in range(ret.shape[0]):
                ret[i, i] = vector[i]
            return type(vector)(ret)
        # 处理矩阵乘法表达式 MatMul 的情况
        if vector.is_MatMul:
            # 提取矩阵和标量
            matrices = [arg for arg in vector.args if arg.is_Matrix]
            scalars = [arg for arg in vector.args if arg not in matrices]
            if scalars:
                # 计算标量乘积和矩阵乘积的对角矩阵形式
                return Mul.fromiter(scalars) * DiagMatrix(MatMul.fromiter(matrices).doit()).doit()
        # 处理转置操作的情况
        if isinstance(vector, Transpose):
            vector = vector.arg
        # 返回处理后的对角矩阵
        return DiagMatrix(vector)


# 函数：对给定的向量进行对角化处理并返回结果
def diagonalize_vector(vector):
    return DiagMatrix(vector).doit()
```