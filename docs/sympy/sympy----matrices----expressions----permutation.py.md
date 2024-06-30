# `D:\src\scipysrc\sympy\sympy\matrices\expressions\permutation.py`

```
# 导入 sympy.core 模块的 S 符号
from sympy.core import S
# 导入 sympy.core.sympify 模块的 _sympify 函数
from sympy.core.sympify import _sympify
# 导入 sympy.functions 模块的 KroneckerDelta 函数
from sympy.functions import KroneckerDelta

# 导入当前包中的 matexpr 模块的 MatrixExpr 类
from .matexpr import MatrixExpr
# 导入当前包中的 special 模块的 ZeroMatrix, Identity, OneMatrix 类
from .special import ZeroMatrix, Identity, OneMatrix

# 定义 PermutationMatrix 类，继承自 MatrixExpr 类
class PermutationMatrix(MatrixExpr):
    """A Permutation Matrix

    Parameters
    ==========

    perm : Permutation
        The permutation the matrix uses.

        The size of the permutation determines the matrix size.

        See the documentation of
        :class:`sympy.combinatorics.permutations.Permutation` for
        the further information of how to create a permutation object.

    Examples
    ========

    >>> from sympy import Matrix, PermutationMatrix
    >>> from sympy.combinatorics import Permutation

    Creating a permutation matrix:

    >>> p = Permutation(1, 2, 0)
    >>> P = PermutationMatrix(p)
    >>> P = P.as_explicit()
    >>> P
    Matrix([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]])

    Permuting a matrix row and column:

    >>> M = Matrix([0, 1, 2])
    >>> Matrix(P*M)
    Matrix([
    [1],
    [2],
    [0]])

    >>> Matrix(M.T*P)
    Matrix([[2, 0, 1]])

    See Also
    ========

    sympy.combinatorics.permutations.Permutation
    """

    # 构造函数，接受一个 perm 参数，类型为 Permutation
    def __new__(cls, perm):
        # 导入 sympy.combinatorics.permutations 模块的 Permutation 类
        from sympy.combinatorics.permutations import Permutation

        # 将 perm 转换为 SymPy 表达式
        perm = _sympify(perm)
        # 检查 perm 是否为 Permutation 类型的实例
        if not isinstance(perm, Permutation):
            # 如果不是，抛出 ValueError 异常
            raise ValueError(
                "{} must be a SymPy Permutation instance.".format(perm))

        # 调用父类的构造方法，返回一个新的 PermutationMatrix 对象
        return super().__new__(cls, perm)

    # 返回矩阵的形状属性
    @property
    def shape(self):
        # 获取 Permutation 对象的大小，并返回一个元组表示矩阵的形状
        size = self.args[0].size
        return (size, size)

    # 返回矩阵是否为单位矩阵的布尔属性
    @property
    def is_Identity(self):
        return self.args[0].is_Identity

    # 执行矩阵的操作，如果矩阵是单位矩阵，则返回 Identity 类型的对象
    def doit(self, **hints):
        if self.is_Identity:
            return Identity(self.rows)
        return self

    # 返回矩阵的第 (i, j) 位置的元素
    def _entry(self, i, j, **kwargs):
        perm = self.args[0]
        # 使用 KroneckerDelta 函数实现矩阵元素的表示
        return KroneckerDelta(perm.apply(i), j)

    # 返回矩阵的 exp 次幂
    def _eval_power(self, exp):
        # 调用 PermutationMatrix 类的构造方法，计算 perm 的 exp 次幂后再进行 doit 操作
        return PermutationMatrix(self.args[0] ** exp).doit()

    # 返回矩阵的逆矩阵
    def _eval_inverse(self):
        # 调用 PermutationMatrix 类的构造方法，计算 perm 的逆矩阵
        return PermutationMatrix(self.args[0] ** -1)

    # 返回矩阵的转置矩阵或伴随矩阵
    _eval_transpose = _eval_adjoint = _eval_inverse

    # 返回矩阵的行列式值
    def _eval_determinant(self):
        sign = self.args[0].signature()
        # 根据置换的符号值返回对应的 SymPy S 符号
        if sign == 1:
            return S.One
        elif sign == -1:
            return S.NegativeOne
        # 如果符号不为 1 或 -1，则抛出 NotImplementedError 异常
        raise NotImplementedError
    def _eval_rewrite_as_BlockDiagMatrix(self, *args, **kwargs):
        # 导入必要的模块和类
        from sympy.combinatorics.permutations import Permutation
        from .blockmatrix import BlockDiagMatrix

        # 获取第一个参数作为置换对象
        perm = self.args[0]
        # 获取置换的完整循环形式
        full_cyclic_form = perm.full_cyclic_form

        # 用于存储分解后的循环集合
        cycles_picks = []

        # Stage 1. 将循环分解为可块状化的形式。
        a, b, c = 0, 0, 0  # 初始化变量
        flag = False  # 标志位，用于控制状态
        for cycle in full_cyclic_form:
            l = len(cycle)  # 循环长度
            m = max(cycle)  # 最大元素值

            if not flag:
                if m + 1 > a + l:
                    flag = True
                    temp = [cycle]  # 开始新的块
                    b = m
                    c = l
                else:
                    cycles_picks.append([cycle])
                    a += l

            else:
                if m > b:
                    if m + 1 == a + c + l:
                        temp.append(cycle)
                        cycles_picks.append(temp)
                        flag = False
                        a = m + 1
                    else:
                        b = m
                        temp.append(cycle)
                        c += l
                else:
                    if b + 1 == a + c + l:
                        temp.append(cycle)
                        cycles_picks.append(temp)
                        flag = False
                        a = b + 1
                    else:
                        temp.append(cycle)
                        c += l

        # Stage 2. 标准化每个分解后的循环并构建矩阵。
        p = 0  # 初始化偏移量
        args = []  # 存储块对角矩阵的参数列表
        for pick in cycles_picks:
            new_cycles = []
            l = 0
            for cycle in pick:
                new_cycle = [i - p for i in cycle]  # 标准化循环
                new_cycles.append(new_cycle)
                l += len(cycle)
            p += l  # 更新偏移量
            perm = Permutation(new_cycles)
            mat = PermutationMatrix(perm)
            args.append(mat)  # 添加块对角矩阵到参数列表

        return BlockDiagMatrix(*args)  # 返回构建好的块对角矩阵对象
class MatrixPermute(MatrixExpr):
    r"""Symbolic representation for permuting matrix rows or columns.

    Parameters
    ==========

    perm : Permutation, PermutationMatrix
        The permutation to use for permuting the matrix.
        The permutation can be resized to the suitable one,

    axis : 0 or 1
        The axis to permute alongside.
        If `0`, it will permute the matrix rows.
        If `1`, it will permute the matrix columns.

    Notes
    =====

    This follows the same notation used in
    :meth:`sympy.matrices.matrixbase.MatrixBase.permute`.

    Examples
    ========

    >>> from sympy import Matrix, MatrixPermute
    >>> from sympy.combinatorics import Permutation

    Permuting the matrix rows:

    >>> p = Permutation(1, 2, 0)
    >>> A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> B = MatrixPermute(A, p, axis=0)
    >>> B.as_explicit()
    Matrix([
    [4, 5, 6],
    [7, 8, 9],
    [1, 2, 3]])

    Permuting the matrix columns:

    >>> B = MatrixPermute(A, p, axis=1)
    >>> B.as_explicit()
    Matrix([
    [2, 3, 1],
    [5, 6, 4],
    [8, 9, 7]])

    See Also
    ========

    sympy.matrices.matrixbase.MatrixBase.permute
    """
    
    # 定义一个新的类方法，用于创建 MatrixPermute 实例
    def __new__(cls, mat, perm, axis=S.Zero):
        # 导入 Permutation 类
        from sympy.combinatorics.permutations import Permutation
        
        # 将 mat 转换为 SymPy 表达式
        mat = _sympify(mat)
        # 检查 mat 是否为 SymPy 矩阵实例，否则引发错误
        if not mat.is_Matrix:
            raise ValueError(
                "{} must be a SymPy matrix instance.".format(perm))
        
        # 将 perm 转换为 SymPy 表达式
        perm = _sympify(perm)
        # 如果 perm 是 PermutationMatrix 实例，获取其参数
        if isinstance(perm, PermutationMatrix):
            perm = perm.args[0]
        
        # 检查 perm 是否为 Permutation 类型，否则引发错误
        if not isinstance(perm, Permutation):
            raise ValueError(
                "{} must be a SymPy Permutation or a PermutationMatrix " \
                "instance".format(perm))
        
        # 将 axis 转换为 SymPy 表达式
        axis = _sympify(axis)
        # 检查 axis 是否为 0 或 1，否则引发错误
        if axis not in (0, 1):
            raise ValueError("The axis must be 0 or 1.")
        
        # 获取矩阵 mat 在指定 axis 上的尺寸
        mat_size = mat.shape[axis]
        # 如果矩阵尺寸与置换 perm 的大小不匹配，尝试调整 perm 的大小
        if mat_size != perm.size:
            try:
                perm = perm.resize(mat_size)
            except ValueError:
                raise ValueError(
                    "Size does not match between the permutation {} "
                    "and the matrix {} threaded over the axis {} "
                    "and cannot be converted."
                    .format(perm, mat, axis))
        
        # 调用父类的 __new__ 方法创建 MatrixPermute 实例并返回
        return super().__new__(cls, mat, perm, axis)
    def doit(self, deep=True, **hints):
        # 从 self.args 中获取矩阵 mat、置换 perm 和轴 axis
        mat, perm, axis = self.args

        # 如果 deep 参数为 True，则递归地对 mat 和 perm 调用 doit 方法
        if deep:
            mat = mat.doit(deep=deep, **hints)
            perm = perm.doit(deep=deep, **hints)

        # 如果 perm 是单位置换，直接返回 mat
        if perm.is_Identity:
            return mat

        # 如果 mat 是单位矩阵，并且 axis 是零，则返回置换矩阵 PermutationMatrix(perm)
        # 如果 axis 是一，则返回逆置换矩阵 PermutationMatrix(perm**-1)
        if mat.is_Identity:
            if axis is S.Zero:
                return PermutationMatrix(perm)
            elif axis is S.One:
                return PermutationMatrix(perm**-1)

        # 如果 mat 是 ZeroMatrix 或 OneMatrix，则直接返回 mat
        if isinstance(mat, (ZeroMatrix, OneMatrix)):
            return mat

        # 如果 mat 是 MatrixPermute 类型，并且其第三个参数等于 axis，则返回重新排列后的 MatrixPermute 对象
        if isinstance(mat, MatrixPermute) and mat.args[2] == axis:
            return MatrixPermute(mat.args[0], perm * mat.args[1], axis)

        # 默认情况下返回 self 对象
        return self

    @property
    def shape(self):
        # 返回第一个元素 mat 的 shape 属性
        return self.args[0].shape

    def _entry(self, i, j, **kwargs):
        # 从 self.args 中获取矩阵 mat、置换 perm 和轴 axis
        mat, perm, axis = self.args

        # 如果 axis 等于 0，则返回经过置换 perm 后的 mat[i, j]
        if axis == 0:
            return mat[perm.apply(i), j]
        # 如果 axis 等于 1，则返回 mat[i, perm.apply(j)]
        elif axis == 1:
            return mat[i, perm.apply(j)]

    def _eval_rewrite_as_MatMul(self, *args, **kwargs):
        # 导入 MatMul 类
        from .matmul import MatMul

        # 从 self.args 中获取矩阵 mat、置换 perm 和轴 axis
        mat, perm, axis = self.args

        # 获取 deep 参数，默认为 True
        deep = kwargs.get("deep", True)

        # 如果 deep 参数为 True，则将 mat 重写为 MatMul 类型
        if deep:
            mat = mat.rewrite(MatMul)

        # 如果 axis 等于 0，则返回 MatMul(PermutationMatrix(perm), mat)
        # 如果 axis 等于 1，则返回 MatMul(mat, PermutationMatrix(perm**-1))
        if axis == 0:
            return MatMul(PermutationMatrix(perm), mat)
        elif axis == 1:
            return MatMul(mat, PermutationMatrix(perm**-1))
```