# `D:\src\scipysrc\sympy\sympy\matrices\expressions\blockmatrix.py`

```
# 从 sympy.assumptions.ask 模块中导入 Q 和 ask 函数，用于符号表达式的问题和假设
from sympy.assumptions.ask import (Q, ask)
# 从 sympy.core 模块中导入 Basic, Add, Mul, S 等核心符号和运算
from sympy.core import Basic, Add, Mul, S
# 从 sympy.core.sympify 模块中导入 _sympify 函数，用于将对象转换为 SymPy 表达式
from sympy.core.sympify import _sympify
# 从 sympy.functions 模块中导入 adjoint 函数，用于计算伴随矩阵
from sympy.functions import adjoint
# 从 sympy.functions.elementary.complexes 模块中导入 re 和 im 函数，用于实部和虚部的提取
from sympy.functions.elementary.complexes import re, im
# 从 sympy.strategies 模块中导入 typed, exhaust, condition, do_one, unpack 等策略函数
from sympy.strategies import typed, exhaust, condition, do_one, unpack
# 从 sympy.strategies.traverse 模块中导入 bottom_up 函数，用于从底向上遍历表达式
from sympy.strategies.traverse import bottom_up
# 从 sympy.utilities.iterables 模块中导入 is_sequence 和 sift 函数，用于序列操作
from sympy.utilities.iterables import is_sequence, sift
# 从 sympy.utilities.misc 模块中导入 filldedent 函数，用于填充和移除文本缩进
from sympy.utilities.misc import filldedent

# 从 sympy.matrices 模块中导入 Matrix 和 ShapeError 异常
from sympy.matrices import Matrix, ShapeError
# 从 sympy.matrices.exceptions 模块中导入 NonInvertibleMatrixError 异常
from sympy.matrices.exceptions import NonInvertibleMatrixError
# 从 sympy.matrices.expressions.determinant 模块中导入 det 和 Determinant 类，用于计算行列式
from sympy.matrices.expressions.determinant import det, Determinant
# 从 sympy.matrices.expressions.inverse 模块中导入 Inverse 类，用于矩阵求逆运算
from sympy.matrices.expressions.inverse import Inverse
# 从 sympy.matrices.expressions.matadd 模块中导入 MatAdd 类，用于矩阵加法表达式
from sympy.matrices.expressions.matadd import MatAdd
# 从 sympy.matrices.expressions.matexpr 模块中导入 MatrixExpr 和 MatrixElement 类
from sympy.matrices.expressions.matexpr import MatrixExpr, MatrixElement
# 从 sympy.matrices.expressions.matmul 模块中导入 MatMul 类，用于矩阵乘法表达式
from sympy.matrices.expressions.matmul import MatMul
# 从 sympy.matrices.expressions.matpow 模块中导入 MatPow 类，用于矩阵幂运算表达式
from sympy.matrices.expressions.matpow import MatPow
# 从 sympy.matrices.expressions.slice 模块中导入 MatrixSlice 类，用于矩阵切片表达式
from sympy.matrices.expressions.slice import MatrixSlice
# 从 sympy.matrices.expressions.special 模块中导入 ZeroMatrix 和 Identity 类，分别表示零矩阵和单位矩阵
from sympy.matrices.expressions.special import ZeroMatrix, Identity
# 从 sympy.matrices.expressions.trace 模块中导入 trace 函数，用于计算矩阵的迹
from sympy.matrices.expressions.trace import trace
# 从 sympy.matrices.expressions.transpose 模块中导入 Transpose 和 transpose 函数，用于矩阵的转置操作
from sympy.matrices.expressions.transpose import Transpose, transpose

# 定义一个名为 BlockMatrix 的类，继承自 MatrixExpr 类
class BlockMatrix(MatrixExpr):
    """A BlockMatrix is a Matrix comprised of other matrices.

    The submatrices are stored in a SymPy Matrix object but accessed as part of
    a Matrix Expression

    >>> from sympy import (MatrixSymbol, BlockMatrix, symbols,
    ...     Identity, ZeroMatrix, block_collapse)
    >>> n,m,l = symbols('n m l')
    >>> X = MatrixSymbol('X', n, n)
    >>> Y = MatrixSymbol('Y', m, m)
    >>> Z = MatrixSymbol('Z', n, m)
    >>> B = BlockMatrix([[X, Z], [ZeroMatrix(m,n), Y]])
    >>> print(B)
    Matrix([
    [X, Z],
    [0, Y]])

    >>> C = BlockMatrix([[Identity(n), Z]])
    >>> print(C)
    Matrix([[I, Z]])

    >>> print(block_collapse(C*B))
    Matrix([[X, Z + Z*Y]])

    Some matrices might be comprised of rows of blocks with
    the matrices in each row having the same height and the
    rows all having the same total number of columns but
    not having the same number of columns for each matrix
    in each row. In this case, the matrix is not a block
    matrix and should be instantiated by Matrix.

    >>> from sympy import ones, Matrix
    >>> dat = [
    ... [ones(3,2), ones(3,3)*2],
    ... [ones(2,3)*3, ones(2,2)*4]]
    ...
    >>> BlockMatrix(dat)
    Traceback (most recent call last):
    ...
    ValueError:
    Although this matrix is comprised of blocks, the blocks do not fill
    the matrix in a size-symmetric fashion. To create a full matrix from
    these arguments, pass them directly to Matrix.
    >>> Matrix(dat)
    Matrix([
    [1, 1, 2, 2, 2],
    [1, 1, 2, 2, 2],
    [1, 1, 2, 2, 2],
    [3, 3, 3, 4, 4],
    [3, 3, 3, 4, 4]])

    See Also
    ========
    sympy.matrices.matrixbase.MatrixBase.irregular
    """
    # 重载构造方法，用于创建新的不可变密集矩阵对象
    def __new__(cls, *args, **kwargs):
        # 导入必要的模块和类
        from sympy.matrices.immutable import ImmutableDenseMatrix
        # 定义一个函数，用于检测对象是否为矩阵
        isMat = lambda i: getattr(i, 'is_Matrix', False)
        
        # 检查参数数量及类型是否符合预期
        if len(args) != 1 or \
                not is_sequence(args[0]) or \
                len({isMat(r) for r in args[0]}) != 1:
            raise ValueError(filldedent('''
                expecting a sequence of 1 or more rows
                containing Matrices.'''))
        
        # 将第一个参数作为行向量存储
        rows = args[0] if args else []
        
        # 如果不是单个矩阵，则尝试将其转化为矩阵的列表形式
        if not isMat(rows):
            if rows and isMat(rows[0]):
                rows = [rows]  # 如果rows不是列表或为空，则构造一个列表
            # 进行正则性检查
            # 每行中矩阵数量相同
            blocky = ok = len({len(r) for r in rows}) == 1
            if ok:
                # 每行中每个矩阵的行数相同
                for r in rows:
                    ok = len({i.rows for i in r}) == 1
                    if not ok:
                        break
                blocky = ok
                if ok:
                    # 每列中每个矩阵的列数相同
                    for c in range(len(rows[0])):
                        ok = len({rows[i][c].cols
                            for i in range(len(rows))}) == 1
                        if not ok:
                            break
            
            # 如果正则性检查未通过，则引发异常
            if not ok:
                # 每行中总列数相同
                ok = len({
                    sum(i.cols for i in r) for r in rows}) == 1
                if blocky and ok:
                    raise ValueError(filldedent('''
                        Although this matrix is comprised of blocks,
                        the blocks do not fill the matrix in a
                        size-symmetric fashion. To create a full matrix
                        from these arguments, pass them directly to
                        Matrix.'''))
                raise ValueError(filldedent('''
                    When there are not the same number of rows in each
                    row's matrices or there are not the same number of
                    total columns in each row, the matrix is not a
                    block matrix. If this matrix is known to consist of
                    blocks fully filling a 2-D space then see
                    Matrix.irregular.'''))
        
        # 创建不可变密集矩阵对象
        mat = ImmutableDenseMatrix(rows, evaluate=False)
        # 调用基类构造方法创建对象
        obj = Basic.__new__(cls, mat)
        return obj

    @property
    # 返回当前矩阵的形状，即行数和列数的元组
    def shape(self):
        numrows = numcols = 0
        M = self.blocks
        
        # 计算所有矩阵行数的总和
        for i in range(M.shape[0]):
            numrows += M[i, 0].shape[0]
        
        # 计算所有矩阵列数的总和
        for i in range(M.shape[1]):
            numcols += M[0, i].shape[1]
        
        # 返回形状的元组
        return (numrows, numcols)

    @property
    # 返回当前矩阵块的形状，即行数和列数的元组
    def blockshape(self):
        return self.blocks.shape

    @property
    # 返回当前矩阵的块对象
    def blocks(self):
        return self.args[0]

    @property
    # 返回每行块的行数列表
    def rowblocksizes(self):
        return [self.blocks[i, 0].rows for i in range(self.blockshape[0])]

    @property
    # 返回每列块的列数列表
    def colblocksizes(self):
        return [self.blocks[0, j].cols for j in range(self.blockshape[1])]
    # 返回一个列表，包含每个列块的列数
    def colblocksizes(self):
        return [self.blocks[0, i].cols for i in range(self.blockshape[1])]

    # 检查当前 BlockMatrix 是否结构上等同于另一个 BlockMatrix
    def structurally_equal(self, other):
        return (isinstance(other, BlockMatrix)
            and self.shape == other.shape
            and self.blockshape == other.blockshape
            and self.rowblocksizes == other.rowblocksizes
            and self.colblocksizes == other.colblocksizes)

    # 如果当前 BlockMatrix 与另一个 BlockMatrix 的列块大小相等，则返回它们的乘积
    def _blockmul(self, other):
        if (isinstance(other, BlockMatrix) and
                self.colblocksizes == other.rowblocksizes):
            return BlockMatrix(self.blocks * other.blocks)

        # 否则返回普通矩阵乘积
        return self * other

    # 如果当前 BlockMatrix 与另一个 BlockMatrix 结构相同，则返回它们的加法结果
    def _blockadd(self, other):
        if (isinstance(other, BlockMatrix)
                and self.structurally_equal(other)):
            return BlockMatrix(self.blocks + other.blocks)

        # 否则返回普通矩阵加法结果
        return self + other

    # 对当前 BlockMatrix 执行转置操作
    def _eval_transpose(self):
        # 对每个单独的矩阵执行转置操作
        matrices = [transpose(matrix) for matrix in self.blocks]
        # 创建一个新的 Matrix 对象
        M = Matrix(self.blockshape[0], self.blockshape[1], matrices)
        # 对整体的块结构进行转置
        M = M.transpose()
        return BlockMatrix(M)

    # 对当前 BlockMatrix 执行共轭转置操作
    def _eval_adjoint(self):
        # 对每个单独的矩阵执行共轭转置操作
        matrices = [adjoint(matrix) for matrix in self.blocks]
        # 创建一个新的 Matrix 对象
        M = Matrix(self.blockshape[0], self.blockshape[1], matrices)
        # 对整体的块结构进行转置
        M = M.transpose()
        return BlockMatrix(M)

    # 如果行块大小与列块大小相等，则计算当前 BlockMatrix 的迹
    def _eval_trace(self):
        if self.rowblocksizes == self.colblocksizes:
            # 提取对角块并计算每个块的迹，然后求和
            blocks = [self.blocks[i, i] for i in range(self.blockshape[0])]
            return Add(*[trace(block) for block in blocks])

    # 计算当前 BlockMatrix 的行列式
    def _eval_determinant(self):
        if self.blockshape == (1, 1):
            # 对于 1x1 的块矩阵，直接计算其行列式
            return det(self.blocks[0, 0])
        if self.blockshape == (2, 2):
            # 对于 2x2 的块矩阵，按照块矩阵的形式提取块 A、B、C、D，并根据条件计算行列式
            [[A, B],
             [C, D]] = self.blocks.tolist()
            if ask(Q.invertible(A)):
                return det(A) * det(D - C * A.I * B)
            elif ask(Q.invertible(D)):
                return det(D) * det(A - B * D.I * C)
        # 对于其他情况，返回当前 BlockMatrix 的行列式
        return Determinant(self)

    # 将当前 BlockMatrix 拆分为实部和虚部的两个 BlockMatrix
    def _eval_as_real_imag(self):
        # 计算每个矩阵的实部和虚部
        real_matrices = [re(matrix) for matrix in self.blocks]
        real_matrices = Matrix(self.blockshape[0], self.blockshape[1], real_matrices)

        im_matrices = [im(matrix) for matrix in self.blocks]
        im_matrices = Matrix(self.blockshape[0], self.blockshape[1], im_matrices)

        return (BlockMatrix(real_matrices), BlockMatrix(im_matrices))

    # 对当前 BlockMatrix 求关于变量 x 的导数
    def _eval_derivative(self, x):
        return BlockMatrix(self.blocks.diff(x))
    def transpose(self):
        """
        Return transpose of matrix.

        Examples
        ========

        >>> from sympy import MatrixSymbol, BlockMatrix, ZeroMatrix
        >>> from sympy.abc import m, n

        # 创建一个 n x n 的符号矩阵 X
        X = MatrixSymbol('X', n, n)
        # 创建一个 m x m 的符号矩阵 Y
        Y = MatrixSymbol('Y', m, m)
        # 创建一个 n x m 的符号矩阵 Z
        Z = MatrixSymbol('Z', n, m)
        # 创建一个块矩阵 B，由四个子矩阵组成：左上角为 X，右上角为 Z，左下角为 m x n 的零矩阵，右下角为 Y
        B = BlockMatrix([[X, Z], [ZeroMatrix(m,n), Y]])
        
        # 对块矩阵 B 进行转置操作，得到新的矩阵
        B.transpose()
        Matrix([
        [X.T,  0],
        [Z.T, Y.T]])

        # 用下划线获取上一个语句的结果（转置后的矩阵），再次对其进行转置操作
        _.transpose()
        Matrix([
        [X, Z],
        [0, Y]])
        """
        # 调用对象的内部方法 _eval_transpose() 实现矩阵转置，返回转置后的矩阵对象
        return self._eval_transpose()
    # 返回 BlockMatrix 的指定矩阵的 Schur 补
    def schur(self, mat='A', generalized=False):
        """Return the Schur Complement of the 2x2 BlockMatrix
        
        Parameters
        ==========
        
        mat : String, optional
            The matrix with respect to which the
            Schur Complement is calculated. 'A' is
            used by default
        
        generalized : bool, optional
            If True, returns the generalized Schur
            Component which uses Moore-Penrose Inverse
        
        Examples
        ========
        
        >>> from sympy import symbols, MatrixSymbol, BlockMatrix
        >>> m, n = symbols('m n')
        >>> A = MatrixSymbol('A', n, n)
        >>> B = MatrixSymbol('B', n, m)
        >>> C = MatrixSymbol('C', m, n)
        >>> D = MatrixSymbol('D', m, m)
        >>> X = BlockMatrix([[A, B], [C, D]])
        
        The default Schur Complement is evaluated with "A"
        
        >>> X.schur()
        -C*A**(-1)*B + D
        >>> X.schur('D')
        A - B*D**(-1)*C
        
        Schur complement with non-invertible matrices is not
        defined. Instead, the generalized Schur complement can
        be calculated which uses the Moore-Penrose Inverse. To
        achieve this, `generalized` must be set to `True`
        
        >>> X.schur('B', generalized=True)
        C - D*(B.T*B)**(-1)*B.T*A
        >>> X.schur('C', generalized=True)
        -A*(C.T*C)**(-1)*C.T*D + B
        
        Returns
        =======
        
        M : Matrix
            The Schur Complement Matrix
        
        Raises
        ======
        
        ShapeError
            If the block matrix is not a 2x2 matrix
        
        NonInvertibleMatrixError
            If given matrix is non-invertible
        
        References
        ==========
        
        .. [1] Wikipedia Article on Schur Component : https://en.wikipedia.org/wiki/Schur_complement
        
        See Also
        ========
        
        sympy.matrices.matrixbase.MatrixBase.pinv
        """
        
        # 检查是否为 2x2 块矩阵
        if self.blockshape == (2, 2):
            # 解析块矩阵为 A, B, C, D 四个子矩阵
            [[A, B],
             [C, D]] = self.blocks.tolist()
            d = {'A': A, 'B': B, 'C': C, 'D': D}
            try:
                # 计算矩阵 mat 的逆或广义逆
                inv = (d[mat].T * d[mat]).inv() * d[mat].T if generalized else d[mat].inv()
                # 根据给定的 mat 返回相应的 Schur 补
                if mat == 'A':
                    return D - C * inv * B
                elif mat == 'B':
                    return C - D * inv * A
                elif mat == 'C':
                    return B - A * inv * D
                elif mat == 'D':
                    return A - B * inv * C
                # 如果没有子矩阵是方阵，则返回自身
                return self
            except NonInvertibleMatrixError:
                # 抛出非可逆矩阵错误，建议设置 generalized=True 使用广义逆计算
                raise NonInvertibleMatrixError('The given matrix is not invertible. Please set generalized=True \
            to compute the generalized Schur Complement which uses Moore-Penrose Inverse')
        else:
            # 如果不是 2x2 块矩阵，则抛出形状错误
            raise ShapeError('Schur Complement can only be calculated for 2x2 block matrices')
    def LDUdecomposition(self):
        """Returns the Block LDU decomposition of
        a 2x2 Block Matrix
        
        返回2x2块矩阵的块LDU分解结果

        Returns
        =======

        (L, D, U) : Matrices
            L : Lower Diagonal Matrix
                下三角矩阵 L
            D : Diagonal Matrix
                对角矩阵 D
            U : Upper Diagonal Matrix
                上三角矩阵 U

        Examples
        ========

        >>> from sympy import symbols, MatrixSymbol, BlockMatrix, block_collapse
        >>> m, n = symbols('m n')
        >>> A = MatrixSymbol('A', n, n)
        >>> B = MatrixSymbol('B', n, m)
        >>> C = MatrixSymbol('C', m, n)
        >>> D = MatrixSymbol('D', m, m)
        >>> X = BlockMatrix([[A, B], [C, D]])
        >>> L, D, U = X.LDUdecomposition()
        >>> block_collapse(L*D*U)
        Matrix([
        [A, B],
        [C, D]])

        Raises
        ======

        ShapeError
            If the block matrix is not a 2x2 matrix
            如果块矩阵不是2x2矩阵

        NonInvertibleMatrixError
            If the matrix "A" is non-invertible
            如果矩阵 "A" 不可逆

        See Also
        ========
        sympy.matrices.expressions.blockmatrix.BlockMatrix.UDLdecomposition
        sympy.matrices.expressions.blockmatrix.BlockMatrix.LUdecomposition
        参见 sympy.matrices.expressions.blockmatrix.BlockMatrix.UDLdecomposition 和 sympy.matrices.expressions.blockmatrix.BlockMatrix.LUdecomposition
        """
        if self.blockshape == (2,2):
            [[A, B],
             [C, D]] = self.blocks.tolist()
            try:
                AI = A.I
            except NonInvertibleMatrixError:
                raise NonInvertibleMatrixError('Block LDU decomposition cannot be calculated when\
                    "A" is singular')
            Ip = Identity(B.shape[0])
            Iq = Identity(B.shape[1])
            Z = ZeroMatrix(*B.shape)
            L = BlockMatrix([[Ip, Z], [C*AI, Iq]])
            D = BlockDiagMatrix(A, self.schur())
            U = BlockMatrix([[Ip, AI*B],[Z.T, Iq]])
            return L, D, U
        else:
            raise ShapeError("Block LDU decomposition is supported only for 2x2 block matrices")
    def UDLdecomposition(self):
        """Returns the Block UDL decomposition of
        a 2x2 Block Matrix
        
        返回一个2x2块矩阵的块UDL分解

        Returns
        =======

        (U, D, L) : Matrices
            U : Upper Diagonal Matrix
                上对角矩阵
            D : Diagonal Matrix
                对角矩阵
            L : Lower Diagonal Matrix
                下对角矩阵

        Examples
        ========

        >>> from sympy import symbols, MatrixSymbol, BlockMatrix, block_collapse
        >>> m, n = symbols('m n')
        >>> A = MatrixSymbol('A', n, n)
        >>> B = MatrixSymbol('B', n, m)
        >>> C = MatrixSymbol('C', m, n)
        >>> D = MatrixSymbol('D', m, m)
        >>> X = BlockMatrix([[A, B], [C, D]])
        >>> U, D, L = X.UDLdecomposition()
        >>> block_collapse(U*D*L)
        Matrix([
        [A, B],
        [C, D]])

        Raises
        ======

        ShapeError
            If the block matrix is not a 2x2 matrix
            如果块矩阵不是2x2矩阵

        NonInvertibleMatrixError
            If the matrix "D" is non-invertible
            如果矩阵"D"不可逆

        See Also
        ========
        sympy.matrices.expressions.blockmatrix.BlockMatrix.LDUdecomposition
        sympy.matrices.expressions.blockmatrix.BlockMatrix.LUdecomposition
        """
        if self.blockshape == (2,2):
            # 将块矩阵解析成子矩阵A, B, C, D
            [[A, B],
             [C, D]] = self.blocks.tolist()
            try:
                # 尝试计算D的逆DI
                DI = D.I
            except NonInvertibleMatrixError:
                # 若D不可逆，则抛出异常
                raise NonInvertibleMatrixError('Block UDL decomposition cannot be calculated when\
                    "D" is singular')
            # 构造单位矩阵Ip, Iq，零矩阵Z
            Ip = Identity(A.shape[0])
            Iq = Identity(B.shape[1])
            Z = ZeroMatrix(*B.shape)
            # 构造块矩阵U, D, L
            U = BlockMatrix([[Ip, B*DI], [Z.T, Iq]])
            D = BlockDiagMatrix(self.schur('D'), D)
            L = BlockMatrix([[Ip, Z],[DI*C, Iq]])
            return U, D, L
        else:
            # 若不是2x2块矩阵，则抛出形状错误异常
            raise ShapeError("Block UDL decomposition is supported only for 2x2 block matrices")
    # 实现块状 LU 分解的方法，适用于 2x2 块状矩阵

    def LUdecomposition(self):
        """Returns the Block LU decomposition of
        a 2x2 Block Matrix

        Returns
        =======
        (L, U) : Matrices
            L : Lower Diagonal Matrix
            U : Upper Diagonal Matrix

        Examples
        ========
        >>> from sympy import symbols, MatrixSymbol, BlockMatrix, block_collapse
        >>> m, n = symbols('m n')
        >>> A = MatrixSymbol('A', n, n)
        >>> B = MatrixSymbol('B', n, m)
        >>> C = MatrixSymbol('C', m, n)
        >>> D = MatrixSymbol('D', m, m)
        >>> X = BlockMatrix([[A, B], [C, D]])
        >>> L, U = X.LUdecomposition()
        >>> block_collapse(L*U)
        Matrix([
        [A, B],
        [C, D]])

        Raises
        ======
        ShapeError
            If the block matrix is not a 2x2 matrix

        NonInvertibleMatrixError
            If the matrix "A" is non-invertible

        See Also
        ========
        sympy.matrices.expressions.blockmatrix.BlockMatrix.UDLdecomposition
        sympy.matrices.expressions.blockmatrix.BlockMatrix.LDUdecomposition
        """
        # 检查块矩阵是否为 2x2
        if self.blockshape == (2,2):
            [[A, B],
             [C, D]] = self.blocks.tolist()
            try:
                # 计算 A 的平方根和逆矩阵
                A = A**S.Half
                AI = A.I
            except NonInvertibleMatrixError:
                # 如果 A 是奇异矩阵，则无法进行块状 LU 分解
                raise NonInvertibleMatrixError('Block LU decomposition cannot be calculated when\
                    "A" is singular')
            # 构造零矩阵 Z 和 Schur 平方根 Q
            Z = ZeroMatrix(*B.shape)
            Q = self.schur()**S.Half
            # 构造 L 和 U 矩阵
            L = BlockMatrix([[A, Z], [C*AI, Q]])
            U = BlockMatrix([[A, AI*B],[Z.T, Q]])
            return L, U
        else:
            # 如果不是 2x2 块矩阵，抛出形状错误异常
            raise ShapeError("Block LU decomposition is supported only for 2x2 block matrices")

    def _entry(self, i, j, **kwargs):
        # 查找行条目
        orig_i, orig_j = i, j
        # 遍历行块和行条目数
        for row_block, numrows in enumerate(self.rowblocksizes):
            cmp = i < numrows
            if cmp == True:
                break
            elif cmp == False:
                i -= numrows
            elif row_block < self.blockshape[0] - 1:
                # 无法确定是哪个块并且不是最后一个，返回未求值的矩阵元素
                return MatrixElement(self, orig_i, orig_j)
        # 遍历列块和列条目数
        for col_block, numcols in enumerate(self.colblocksizes):
            cmp = j < numcols
            if cmp == True:
                break
            elif cmp == False:
                j -= numcols
            elif col_block < self.blockshape[1] - 1:
                return MatrixElement(self, orig_i, orig_j)
        # 返回指定位置的矩阵块元素
        return self.blocks[row_block, col_block][i, j]

    @property
    # 检查是否为单位矩阵，即检查矩阵是否为方阵且对角线元素为单位块，非对角线元素为零块
    def is_Identity(self):
        # 检查矩阵是否为方阵
        if self.blockshape[0] != self.blockshape[1]:
            return False
        # 遍历矩阵的每个块
        for i in range(self.blockshape[0]):
            for j in range(self.blockshape[1]):
                # 如果在对角线上且块不是单位块，则不是单位矩阵
                if i == j and not self.blocks[i, j].is_Identity:
                    return False
                # 如果不在对角线上且块不是零块，则不是单位矩阵
                if i != j and not self.blocks[i, j].is_ZeroMatrix:
                    return False
        # 若所有条件满足，则是单位矩阵
        return True

    @property
    # 检查结构上是否对称，即行块大小是否等于列块大小
    def is_structurally_symmetric(self):
        return self.rowblocksizes == self.colblocksizes

    # 比较两个 BlockMatrix 对象是否相等
    def equals(self, other):
        # 如果是同一个对象，则相等
        if self == other:
            return True
        # 如果是 BlockMatrix 类型且块矩阵相等，则相等
        if isinstance(other, BlockMatrix) and self.blocks == other.blocks:
            return True
        # 否则调用父类的 equals 方法进行比较
        return super().equals(other)
class BlockDiagMatrix(BlockMatrix):
    """A sparse matrix with block matrices along its diagonals

    Examples
    ========

    >>> from sympy import MatrixSymbol, BlockDiagMatrix, symbols
    >>> n, m, l = symbols('n m l')
    >>> X = MatrixSymbol('X', n, n)
    >>> Y = MatrixSymbol('Y', m, m)
    >>> BlockDiagMatrix(X, Y)
    Matrix([
    [X, 0],
    [0, Y]])

    Notes
    =====

    If you want to get the individual diagonal blocks, use
    :meth:`get_diag_blocks`.

    See Also
    ========

    sympy.matrices.dense.diag
    """

    def __new__(cls, *mats):
        # 创建一个新的 BlockDiagMatrix 实例，确保所有输入的矩阵被 sympify 处理
        return Basic.__new__(BlockDiagMatrix, *[_sympify(m) for m in mats])

    @property
    def diag(self):
        # 返回 BlockDiagMatrix 的参数列表，即对角块矩阵的列表
        return self.args

    @property
    def blocks(self):
        # 导入必要的模块并获取参数列表
        from sympy.matrices.immutable import ImmutableDenseMatrix
        mats = self.args
        # 构造一个数据列表，其中对角块和非对角块都正确放置
        data = [[mats[i] if i == j else ZeroMatrix(mats[i].rows, mats[j].cols)
                 for j in range(len(mats))]
                for i in range(len(mats))]
        # 返回一个 ImmutableDenseMatrix 对象，表示整个 BlockDiagMatrix
        return ImmutableDenseMatrix(data, evaluate=False)

    @property
    def shape(self):
        # 返回 BlockDiagMatrix 的形状，即所有块的行数总和和列数总和的元组
        return (sum(block.rows for block in self.args),
                sum(block.cols for block in self.args))

    @property
    def blockshape(self):
        # 返回 BlockDiagMatrix 的块形状，即块的行数和列数的元组
        n = len(self.args)
        return (n, n)

    @property
    def rowblocksizes(self):
        # 返回 BlockDiagMatrix 中每个块的行数组成的列表
        return [block.rows for block in self.args]

    @property
    def colblocksizes(self):
        # 返回 BlockDiagMatrix 中每个块的列数组成的列表
        return [block.cols for block in self.args]

    def _all_square_blocks(self):
        """Returns true if all blocks are square"""
        # 检查 BlockDiagMatrix 中所有块是否都是方阵
        return all(mat.is_square for mat in self.args)

    def _eval_determinant(self):
        # 如果所有块都是方阵，返回所有块的行列式的乘积；否则返回零
        if self._all_square_blocks():
            return Mul(*[det(mat) for mat in self.args])
        # 至少有一个非方阵的块，则整个矩阵的行列式为零
        return S.Zero

    def _eval_inverse(self, expand='ignored'):
        # 如果所有块都是方阵，返回每个块的逆组成的 BlockDiagMatrix；否则引发异常
        if self._all_square_blocks():
            return BlockDiagMatrix(*[mat.inverse() for mat in self.args])
        # 至少有一个非方阵的块，则无法求逆，引发异常
        raise NonInvertibleMatrixError('Matrix det == 0; not invertible.')

    def _eval_transpose(self):
        # 返回 BlockDiagMatrix 的转置，即每个块的转置组成的 BlockDiagMatrix
        return BlockDiagMatrix(*[mat.transpose() for mat in self.args])

    def _blockmul(self, other):
        # 如果 other 是 BlockDiagMatrix 并且列块大小与 self 的行块大小相同，则返回两个矩阵的块乘积
        # 否则调用父类的 _blockmul 方法处理乘积
        if (isinstance(other, BlockDiagMatrix) and
                self.colblocksizes == other.rowblocksizes):
            return BlockDiagMatrix(*[a*b for a, b in zip(self.args, other.args)])
        else:
            return BlockMatrix._blockmul(self, other)
    # 定义一个方法，用于将当前对象与另一个块对角矩阵相加
    def _blockadd(self, other):
        # 检查另一个对象是否为块对角矩阵，并且其块形状、行块大小和列块大小与当前对象相同
        if (isinstance(other, BlockDiagMatrix) and
                self.blockshape == other.blockshape and
                self.rowblocksizes == other.rowblocksizes and
                self.colblocksizes == other.colblocksizes):
            # 返回一个新的块对角矩阵，其对角块是当前对象和另一个对象对应位置对角块的和
            return BlockDiagMatrix(*[a + b for a, b in zip(self.args, other.args)])
        else:
            # 如果条件不满足，调用父类 BlockMatrix 的 _blockadd 方法进行处理
            return BlockMatrix._blockadd(self, other)

    def get_diag_blocks(self):
        """返回矩阵的对角块列表。

        示例
        ========

        >>> from sympy import BlockDiagMatrix, Matrix

        >>> A = Matrix([[1, 2], [3, 4]])
        >>> B = Matrix([[5, 6], [7, 8]])
        >>> M = BlockDiagMatrix(A, B)

        如何从块对角矩阵中获取对角块：

        >>> diag_blocks = M.get_diag_blocks()
        >>> diag_blocks[0]
        Matrix([
        [1, 2],
        [3, 4]])
        >>> diag_blocks[1]
        Matrix([
        [5, 6],
        [7, 8]])
        """
        # 返回当前块对角矩阵对象的参数列表，即其对角块的列表
        return self.args
def block_collapse(expr):
    """Evaluates a block matrix expression

    >>> from sympy import MatrixSymbol, BlockMatrix, symbols, Identity, ZeroMatrix, block_collapse
    >>> n,m,l = symbols('n m l')
    >>> X = MatrixSymbol('X', n, n)
    >>> Y = MatrixSymbol('Y', m, m)
    >>> Z = MatrixSymbol('Z', n, m)
    >>> B = BlockMatrix([[X, Z], [ZeroMatrix(m, n), Y]])
    >>> print(B)
    Matrix([
    [X, Z],
    [0, Y]])

    >>> C = BlockMatrix([[Identity(n), Z]])
    >>> print(C)
    Matrix([[I, Z]])

    >>> print(block_collapse(C*B))
    Matrix([[X, Z + Z*Y]])
    """
    from sympy.strategies.util import expr_fns

    # 定义一个检查表达式是否包含 BlockMatrix 的函数
    hasbm = lambda expr: isinstance(expr, MatrixExpr) and expr.has(BlockMatrix)

    # 条件化的重写规则，根据表达式类型执行不同的操作
    conditioned_rl = condition(
        hasbm,
        typed(
            {MatAdd: do_one(bc_matadd, bc_block_plus_ident),
             MatMul: do_one(bc_matmul, bc_dist),
             MatPow: bc_matmul,
             Transpose: bc_transpose,
             Inverse: bc_inverse,
             BlockMatrix: do_one(bc_unpack, deblock)}
        )
    )

    # 定义一个底向上的递归规则
    rule = exhaust(
        bottom_up(
            exhaust(conditioned_rl),
            fns=expr_fns
        )
    )

    # 应用重写规则到输入的表达式
    result = rule(expr)
    # 检查结果是否有 'doit' 方法，若有则调用，否则直接返回结果
    doit = getattr(result, 'doit', None)
    if doit is not None:
        return doit()
    else:
        return result


def bc_unpack(expr):
    # 如果表达式的块形状为 (1, 1)，则返回该块
    if expr.blockshape == (1, 1):
        return expr.blocks[0, 0]
    return expr


def bc_matadd(expr):
    # 将表达式中的 BlockMatrix 参数提取出来并相加
    args = sift(expr.args, lambda M: isinstance(M, BlockMatrix))
    blocks = args[True]
    if not blocks:
        return expr

    nonblocks = args[False]
    block = blocks[0]
    for b in blocks[1:]:
        block = block._blockadd(b)
    if nonblocks:
        return MatAdd(*nonblocks) + block
    else:
        return block


def bc_block_plus_ident(expr):
    # 将表达式中的 Identity 项与 BlockMatrix 相乘并合并
    idents = [arg for arg in expr.args if arg.is_Identity]
    if not idents:
        return expr

    blocks = [arg for arg in expr.args if isinstance(arg, BlockMatrix)]
    if (blocks and all(b.structurally_equal(blocks[0]) for b in blocks)
               and blocks[0].is_structurally_symmetric):
        block_id = BlockDiagMatrix(*[Identity(k)
                                        for k in blocks[0].rowblocksizes])
        rest = [arg for arg in expr.args if not arg.is_Identity and not isinstance(arg, BlockMatrix)]
        return MatAdd(block_id * len(idents), *blocks, *rest).doit()

    return expr


def bc_dist(expr):
    """ Turn  a*[X, Y] into [a*X, a*Y] """
    factor, mat = expr.as_coeff_mmul()
    if factor == 1:
        return expr

    unpacked = unpack(mat)

    if isinstance(unpacked, BlockDiagMatrix):
        B = unpacked.diag
        new_B = [factor * mat for mat in B]
        return BlockDiagMatrix(*new_B)
    elif isinstance(unpacked, BlockMatrix):
        B = unpacked.blocks
        new_B = [
            [factor * B[i, j] for j in range(B.cols)] for i in range(B.rows)]
        return BlockMatrix(new_B)
    return expr
    # 检查表达式是否为 MatPow 类型
    if isinstance(expr, MatPow):
        # 如果指数部分是整数且大于0
        if expr.args[1].is_Integer and expr.args[1] > 0:
            # 初始化变量：系数为1，矩阵列表包含多个相同的基础矩阵
            factor, matrices = 1, [expr.args[0]] * expr.args[1]
        else:
            # 如果不满足条件，直接返回原始表达式
            return expr
    else:
        # 尝试将表达式转换为系数与矩阵列表的形式
        factor, matrices = expr.as_coeff_matrices()

    # 初始化索引
    i = 0
    # 遍历矩阵列表中的相邻矩阵对
    while (i+1 < len(matrices)):
        A, B = matrices[i:i+2]
        # 如果 A 和 B 均为 BlockMatrix 类型
        if isinstance(A, BlockMatrix) and isinstance(B, BlockMatrix):
            # 将 A 和 B 进行块乘操作，并将结果替换索引为 i 的位置的矩阵
            matrices[i] = A._blockmul(B)
            # 移除索引为 i+1 的矩阵
            matrices.pop(i+1)
        # 如果 A 是 BlockMatrix 类型而 B 不是
        elif isinstance(A, BlockMatrix):
            # 将 A 与 B 封装成单元素块矩阵后进行块乘操作
            matrices[i] = A._blockmul(BlockMatrix([[B]]))
            # 移除索引为 i+1 的矩阵
            matrices.pop(i+1)
        # 如果 B 是 BlockMatrix 类型而 A 不是
        elif isinstance(B, BlockMatrix):
            # 将 A 与 B 封装成单元素块矩阵后进行块乘操作
            matrices[i] = BlockMatrix([[A]])._blockmul(B)
            # 移除索引为 i+1 的矩阵
            matrices.pop(i+1)
        else:
            # 如果 A 和 B 都不是 BlockMatrix 类型，增加索引 i
            i += 1

    # 构造新的 MatMul 对象，将系数与处理过的矩阵列表作为参数，最后进行求值
    return MatMul(factor, *matrices).doit()
# 对表达式进行转置操作
def bc_transpose(expr):
    # 将表达式的块状折叠成简化形式
    collapse = block_collapse(expr.arg)
    # 调用折叠后对象的转置方法
    return collapse._eval_transpose()


# 对表达式进行逆操作
def bc_inverse(expr):
    # 如果表达式的参数是 BlockDiagMatrix 类型，则直接返回其逆矩阵
    if isinstance(expr.arg, BlockDiagMatrix):
        return expr.inverse()

    # 否则，调用 blockinverse_1x1 函数处理
    expr2 = blockinverse_1x1(expr)
    # 如果经过处理后的表达式与原表达式不同，则返回处理后的结果
    if expr != expr2:
        return expr2
    # 否则，继续调用 blockinverse_2x2 处理更复杂的情况
    return blockinverse_2x2(Inverse(reblock_2x2(expr.arg)))


# 处理 1x1 块矩阵的逆运算
def blockinverse_1x1(expr):
    # 如果表达式的参数是 BlockMatrix 类型且块形状为 (1, 1)，则返回其逆矩阵的 BlockMatrix 包装
    if isinstance(expr.arg, BlockMatrix) and expr.arg.blockshape == (1, 1):
        mat = Matrix([[expr.arg.blocks[0].inverse()]])
        return BlockMatrix(mat)
    # 否则，直接返回原表达式
    return expr


# 处理 2x2 块矩阵的逆运算
def blockinverse_2x2(expr):
    # 如果表达式的参数是 BlockMatrix 类型且块形状为 (2, 2)
    if isinstance(expr.arg, BlockMatrix) and expr.arg.blockshape == (2, 2):
        # 将块矩阵分解为 A, B, C, D 四个分块
        [[A, B],
         [C, D]] = expr.arg.blocks.tolist()

        # 选择适合的 2x2 块矩阵逆运算公式
        formula = _choose_2x2_inversion_formula(A, B, C, D)
        # 根据不同公式进行计算
        if formula != None:
            MI = expr.arg.schur(formula).I
        if formula == 'A':
            AI = A.I
            return BlockMatrix([[AI + AI * B * MI * C * AI, -AI * B * MI], [-MI * C * AI, MI]])
        if formula == 'B':
            BI = B.I
            return BlockMatrix([[-MI * D * BI, MI], [BI + BI * A * MI * D * BI, -BI * A * MI]])
        if formula == 'C':
            CI = C.I
            return BlockMatrix([[-CI * D * MI, CI + CI * D * MI * A * CI], [MI, -MI * A * CI]])
        if formula == 'D':
            DI = D.I
            return BlockMatrix([[MI, -MI * B * DI], [-DI * C * MI, DI + DI * C * MI * B * DI]])

    # 如果不符合 2x2 块矩阵的条件，直接返回原表达式
    return expr


def _choose_2x2_inversion_formula(A, B, C, D):
    """
    假设 [[A, B], [C, D]] 形成一个有效的方块矩阵，找到最适合的经典 2x2 块矩阵逆运算公式。

    返回 'A', 'B', 'C', 'D' 以表示涉及参数逆运算的算法，如果矩阵不能使用任何公式进行逆运算，则返回 None。
    """
    # 尝试找到已知可逆矩阵，注意此处不考虑目前的 Schur 补
    A_inv = ask(Q.invertible(A))
    if A_inv == True:
        return 'A'
    B_inv = ask(Q.invertible(B))
    if B_inv == True:
        return 'B'
    C_inv = ask(Q.invertible(C))
    if C_inv == True:
        return 'C'
    D_inv = ask(Q.invertible(D))
    if D_inv == True:
        return 'D'
    # 否则，尝试找到未知非可逆矩阵
    if A_inv != False:
        return 'A'
    if B_inv != False:
        return 'B'
    if C_inv != False:
        return 'C'
    if D_inv != False:
        return 'D'
    # 如果找不到任何可逆矩阵，则返回 None
    return None


# 扁平化 BlockMatrix 的 BlockMatrix
def deblock(B):
    """ 将 BlockMatrix 中的 BlockMatrix 扁平化 """
    # 如果 B 不是 BlockMatrix 类型或者 B.blocks 中不含 BlockMatrix，则直接返回 B
    if not isinstance(B, BlockMatrix) or not B.blocks.has(BlockMatrix):
        return B
    # 否则，将所有元素转换为 BlockMatrix
    wrap = lambda x: x if isinstance(x, BlockMatrix) else BlockMatrix([[x]])
    bb = B.blocks.applyfunc(wrap)  # 所有元素都是块
    return bb
    # 尝试执行以下代码块，处理可能出现的 ShapeError 异常
    try:
        # 创建一个 Matrix 对象 MM，初始化为 0，行数不变，列数为所有 bb 第一列子块的总和，空列表作为数据
        MM = Matrix(0, sum(bb[0, i].blocks.shape[1] for i in range(bb.shape[1])), [])
        
        # 遍历 bb 的每一行
        for row in range(0, bb.shape[0]):
            # 从 bb 中取出每行的第一列子块，创建一个 Matrix 对象 M
            M = Matrix(bb[row, 0].blocks)
            # 遍历当前行的其余列
            for col in range(1, bb.shape[1]):
                # 将当前列的子块与 M 连接起来，更新 M
                M = M.row_join(bb[row, col].blocks)
            # 将 M 与 MM 进行列连接，更新 MM
            MM = MM.col_join(M)
        
        # 将 MM 封装成 BlockMatrix 对象并返回
        return BlockMatrix(MM)
    
    # 如果出现 ShapeError 异常，则执行以下代码块
    except ShapeError:
        # 返回 B 对象（这个 B 变量应该是在函数外部定义的）
        return B
# 将给定的 BlockMatrix 重新分块为具有2x2块的块矩阵，以保证矩阵仍然可以使用经典的2x2块求逆公式进行求逆。
def reblock_2x2(expr):
    # 如果表达式不是 BlockMatrix 类型，或者所有块的维度都小于等于2，则直接返回原始表达式
    if not isinstance(expr, BlockMatrix) or not all(d > 2 for d in expr.blockshape):
        return expr

    BM = BlockMatrix  # 为了简洁性起见
    rowblocks, colblocks = expr.blockshape  # 获取块矩阵的行和列块数
    blocks = expr.blocks  # 获取块矩阵的块

    # 尝试将行分割为 i 和列分割为 j
    for i in range(1, rowblocks):
        for j in range(1, colblocks):
            # 解包块矩阵，分别获取左上(A), 右上(B), 左下(C), 右下(D)四个块
            A = bc_unpack(BM(blocks[:i, :j]))
            B = bc_unpack(BM(blocks[:i, j:]))
            C = bc_unpack(BM(blocks[i:, :j]))
            D = bc_unpack(BM(blocks[i:, j:]))

            # 选择适当的2x2块求逆公式
            formula = _choose_2x2_inversion_formula(A, B, C, D)
            if formula is not None:
                # 返回新的2x2块矩阵
                return BlockMatrix([[A, B], [C, D]])

    # 如果所有尝试都不可行，则默认分割左上角
    return BM([[blocks[0, 0], BM(blocks[0, 1:])],
               [BM(blocks[1:, 0]), BM(blocks[1:, 1:])]])


# 将大小序列转换为低-高对
def bounds(sizes):
    """
    Convert sequence of numbers into pairs of low-high pairs

    >>> from sympy.matrices.expressions.blockmatrix import bounds
    >>> bounds((1, 10, 50))
    [(0, 1), (1, 11), (11, 61)]
    """
    low = 0
    rv = []  # 结果列表
    for size in sizes:
        rv.append((low, low + size))  # 添加低-高对到结果列表
        low += size  # 更新低值
    return rv  # 返回结果列表


# 将矩阵表达式切割成块
def blockcut(expr, rowsizes, colsizes):
    """
    Cut a matrix expression into Blocks

    >>> from sympy import ImmutableMatrix, blockcut
    >>> M = ImmutableMatrix(4, 4, range(16))
    >>> B = blockcut(M, (1, 3), (1, 3))
    >>> type(B).__name__
    'BlockMatrix'
    >>> ImmutableMatrix(B.blocks[0, 1])
    Matrix([[1, 2, 3]])
    """

    # 计算行和列的边界
    rowbounds = bounds(rowsizes)
    colbounds = bounds(colsizes)

    # 使用行和列的边界切割矩阵表达式，返回 BlockMatrix 对象
    return BlockMatrix([[MatrixSlice(expr, rowbound, colbound)
                         for colbound in colbounds]
                         for rowbound in rowbounds])
```