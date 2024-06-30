# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_matfuncs.py`

```
"""
Sparse matrix functions
"""

#
# Authors: Travis Oliphant, March 2002
#          Anthony Scopatz, August 2012 (Sparse Updates)
#          Jake Vanderplas, August 2012 (Sparse Updates)
#

__all__ = ['expm', 'inv', 'matrix_power']

import numpy as np
from scipy.linalg._basic import solve, solve_triangular

from scipy.sparse._base import issparse
from scipy.sparse.linalg import spsolve
from scipy.sparse._sputils import is_pydata_spmatrix, isintlike

import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse.linalg._interface import LinearOperator
from scipy.sparse._construct import eye

from ._expm_multiply import _ident_like, _exact_1_norm as _onenorm


UPPER_TRIANGULAR = 'upper_triangular'


def inv(A):
    """
    Compute the inverse of a sparse matrix

    Parameters
    ----------
    A : (M, M) sparse matrix
        square matrix to be inverted

    Returns
    -------
    Ainv : (M, M) sparse matrix
        inverse of `A`

    Notes
    -----
    This computes the sparse inverse of `A`. If the inverse of `A` is expected
    to be non-sparse, it will likely be faster to convert `A` to dense and use
    `scipy.linalg.inv`.

    Examples
    --------
    >>> from scipy.sparse import csc_matrix
    >>> from scipy.sparse.linalg import inv
    >>> A = csc_matrix([[1., 0.], [1., 2.]])
    >>> Ainv = inv(A)
    >>> Ainv
    <Compressed Sparse Column sparse matrix of dtype 'float64'
        with 3 stored elements and shape (2, 2)>
    >>> A.dot(Ainv)
    <Compressed Sparse Column sparse matrix of dtype 'float64'
        with 2 stored elements and shape (2, 2)>
    >>> A.dot(Ainv).toarray()
    array([[ 1.,  0.],
           [ 0.,  1.]])

    .. versionadded:: 0.12.0

    """
    # Check input
    if not (scipy.sparse.issparse(A) or is_pydata_spmatrix(A)):
        raise TypeError('Input must be a sparse matrix')

    # Use sparse direct solver to solve "AX = I" accurately
    I = _ident_like(A)
    Ainv = spsolve(A, I)
    return Ainv


def _onenorm_matrix_power_nnm(A, p):
    """
    Compute the 1-norm of a non-negative integer power of a non-negative matrix.

    Parameters
    ----------
    A : a square ndarray or matrix or sparse matrix
        Input matrix with non-negative entries.
    p : non-negative integer
        The power to which the matrix is to be raised.

    Returns
    -------
    out : float
        The 1-norm of the matrix power p of A.

    """
    # Check input
    if int(p) != p or p < 0:
        raise ValueError('expected non-negative integer p')
    p = int(p)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be like a square matrix')

    # Explicitly make a column vector so that this works when A is a
    # numpy matrix (in addition to ndarray and sparse matrix).
    v = np.ones((A.shape[0], 1), dtype=float)
    M = A.T
    for i in range(p):
        v = M.dot(v)
    return np.max(v)


def _is_upper_triangular(A):
    # This function could possibly be of wider interest.
    """
    Check if a matrix is upper triangular.

    Parameters
    ----------
    A : ndarray or sparse matrix
        Input matrix.

    Returns
    -------
    bool
        True if `A` is upper triangular, False otherwise.
    """
    # 如果输入的矩阵 A 是稀疏矩阵
    if issparse(A):
        # 提取稀疏矩阵 A 的下三角部分
        lower_part = scipy.sparse.tril(A, -1)
        # 检查结构上三角性，然后必要时检查偶然的上三角性。
        return lower_part.nnz == 0 or lower_part.count_nonzero() == 0
    # 如果输入的矩阵 A 是 PyData 的稀疏矩阵
    elif is_pydata_spmatrix(A):
        # 导入 sparse 模块
        import sparse
        # 提取稀疏矩阵 A 的下三角部分
        lower_part = sparse.tril(A, -1)
        # 检查下三角部分是否全部为零
        return lower_part.nnz == 0
    else:
        # 对于普通的 NumPy 数组或矩阵 A，检查其非主对角线上是否存在非零元素
        return not np.tril(A, -1).any()
# 定义一个能处理稀疏和结构化矩阵的矩阵乘法函数
def _smart_matrix_product(A, B, alpha=None, structure=None):
    """
    A matrix product that knows about sparse and structured matrices.

    Parameters
    ----------
    A : 2d ndarray
        First matrix.
    B : 2d ndarray
        Second matrix.
    alpha : float
        The matrix product will be scaled by this constant.
    structure : str, optional
        A string describing the structure of both matrices `A` and `B`.
        Only `upper_triangular` is currently supported.

    Returns
    -------
    M : 2d ndarray
        Matrix product of A and B.

    """
    # 检查 A 和 B 是否为二维矩阵
    if len(A.shape) != 2:
        raise ValueError('expected A to be a rectangular matrix')
    if len(B.shape) != 2:
        raise ValueError('expected B to be a rectangular matrix')
    
    # 初始化 BLAS 函数
    f = None
    # 如果结构为 UPPER_TRIANGULAR，且 A 和 B 都不是稀疏矩阵或 PyData 稀疏矩阵
    if structure == UPPER_TRIANGULAR:
        if (not issparse(A) and not issparse(B)
                and not is_pydata_spmatrix(A) and not is_pydata_spmatrix(B)):
            # 获取 trmm BLAS 函数
            f, = scipy.linalg.get_blas_funcs(('trmm',), (A, B))
    
    # 如果 f 存在，则使用 BLAS 函数计算
    if f is not None:
        # 如果 alpha 未指定，默认为 1.0
        if alpha is None:
            alpha = 1.
        out = f(alpha, A, B)  # 使用 BLAS 函数计算矩阵乘积
    else:
        # 如果没有可用的 BLAS 函数，则根据 alpha 的值计算矩阵乘积
        if alpha is None:
            out = A.dot(B)
        else:
            out = alpha * A.dot(B)
    
    # 返回计算结果
    return out


class MatrixPowerOperator(LinearOperator):
    """
    Represents a matrix power operator extending LinearOperator.

    Attributes
    ----------
    _A : ndarray
        The square matrix A.
    _p : int
        The power to which matrix A is raised.
    _structure : str or None
        Describes the structure of matrix A.
    dtype : dtype
        Data type of matrix A.
    ndim : int
        Number of dimensions of matrix A.
    shape : tuple
        Shape of matrix A.

    Methods
    -------
    _matvec(x)
        Computes matrix-vector product of A with vector x.
    _rmatvec(x)
        Computes reverse matrix-vector product of A with vector x.
    _matmat(X)
        Computes matrix-matrix product of A with matrix X.
    T
        Property that returns the transpose of the MatrixPowerOperator.

    """
    
    def __init__(self, A, p, structure=None):
        # 检查 A 是否为方阵
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError('expected A to be like a square matrix')
        # 检查 p 是否为非负整数
        if p < 0:
            raise ValueError('expected p to be a non-negative integer')
        
        # 初始化对象属性
        self._A = A
        self._p = p
        self._structure = structure
        self.dtype = A.dtype
        self.ndim = A.ndim
        self.shape = A.shape
    
    def _matvec(self, x):
        # 计算 A 的 p 次幂与向量 x 的乘积
        for i in range(self._p):
            x = self._A.dot(x)
        return x
    
    def _rmatvec(self, x):
        # 计算 A 转置的 p 次幂与向量 x 的乘积
        A_T = self._A.T
        x = x.ravel()
        for i in range(self._p):
            x = A_T.dot(x)
        return x
    
    def _matmat(self, X):
        # 计算 A 的 p 次幂与矩阵 X 的乘积
        for i in range(self._p):
            X = _smart_matrix_product(self._A, X, structure=self._structure)
        return X
    
    @property
    def T(self):
        # 返回 MatrixPowerOperator 对象的转置
        return MatrixPowerOperator(self._A.T, self._p)


class ProductOperator(LinearOperator):
    """
    Represents a product operator for multiple square matrices extending LinearOperator.

    Notes
    -----
    For now, this is limited to products of multiple square matrices.

    """
    # 初始化方法，接受任意数量的位置参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 从关键字参数中获取结构（structure）参数，若不存在则为None
        self._structure = kwargs.get('structure', None)
        
        # 验证所有位置参数是否为二维方阵，并且维度相同
        for A in args:
            if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
                raise ValueError(
                        'For now, the ProductOperator implementation is '
                        'limited to the product of multiple square matrices.')
        
        # 如果存在位置参数
        if args:
            # 获取第一个位置参数的行数
            n = args[0].shape[0]
            
            # 验证所有位置参数的维度与第一个参数的行数相同
            for A in args:
                for d in A.shape:
                    if d != n:
                        raise ValueError(
                                'The square matrices of the ProductOperator '
                                'must all have the same shape.')
            
            # 设置对象的形状属性为 (n, n)
            self.shape = (n, n)
            # 设置对象的维度属性为形状元组的长度
            self.ndim = len(self.shape)
        
        # 设置对象的数据类型属性为所有参数的结果数据类型
        self.dtype = np.result_type(*[x.dtype for x in args])
        
        # 设置对象的操作序列为位置参数的元组
        self._operator_sequence = args

    # 矩阵-向量乘法方法，按逆序应用操作序列中的矩阵
    def _matvec(self, x):
        for A in reversed(self._operator_sequence):
            x = A.dot(x)
        return x

    # 向量-矩阵乘法方法，按顺序应用操作序列中的矩阵的转置
    def _rmatvec(self, x):
        # 将输入向量展平为一维数组
        x = x.ravel()
        # 按顺序应用操作序列中的矩阵的转置
        for A in self._operator_sequence:
            x = A.T.dot(x)
        return x

    # 矩阵-矩阵乘法方法，按逆序应用操作序列中的矩阵
    def _matmat(self, X):
        for A in reversed(self._operator_sequence):
            # 使用智能矩阵乘积函数对 X 和 A 进行乘积，结构由 self._structure 指定
            X = _smart_matrix_product(A, X, structure=self._structure)
        return X

    # 转置属性，返回操作序列中所有矩阵的转置构成的 ProductOperator 对象
    @property
    def T(self):
        # 生成操作序列中所有矩阵的转置列表
        T_args = [A.T for A in reversed(self._operator_sequence)]
        # 使用转置列表创建新的 ProductOperator 对象并返回
        return ProductOperator(*T_args)
# 有效估计矩阵 A 的 p 次幂的 1-范数
def _onenormest_matrix_power(A, p,
        t=2, itmax=5, compute_v=False, compute_w=False, structure=None):
    """
    Efficiently estimate the 1-norm of A^p.

    Parameters
    ----------
    A : ndarray
        Matrix whose 1-norm of a power is to be computed.
    p : int
        Non-negative integer power.
    t : int, optional
        A positive parameter controlling the tradeoff between
        accuracy versus time and memory usage.
        Larger values take longer and use more memory
        but give more accurate output.
    itmax : int, optional
        Use at most this many iterations.
    compute_v : bool, optional
        Request a norm-maximizing linear operator input vector if True.
    compute_w : bool, optional
        Request a norm-maximizing linear operator output vector if True.
    structure : NoneType, optional
        Not used in this function.

    Returns
    -------
    est : float
        An underestimate of the 1-norm of the sparse matrix A^p.
    v : ndarray, optional
        The vector such that ||Av||_1 == est*||v||_1.
        It can be thought of as an input to the linear operator
        that gives an output with particularly large norm.
    w : ndarray, optional
        The vector Av which has relatively large 1-norm.
        It can be thought of as an output of the linear operator
        that is relatively large in norm compared to the input.

    """
    # 调用 scipy.sparse.linalg.onenormest 函数计算稀疏矩阵 A^p 的 1-范数估计
    return scipy.sparse.linalg.onenormest(
            MatrixPowerOperator(A, p, structure=structure))


# 有效估计线性操作数序列的矩阵乘积的 1-范数
def _onenormest_product(operator_seq,
        t=2, itmax=5, compute_v=False, compute_w=False, structure=None):
    """
    Efficiently estimate the 1-norm of the matrix product of the args.

    Parameters
    ----------
    operator_seq : linear operator sequence
        Matrices whose 1-norm of product is to be computed.
    t : int, optional
        A positive parameter controlling the tradeoff between
        accuracy versus time and memory usage.
        Larger values take longer and use more memory
        but give more accurate output.
    itmax : int, optional
        Use at most this many iterations.
    compute_v : bool, optional
        Request a norm-maximizing linear operator input vector if True.
    compute_w : bool, optional
        Request a norm-maximizing linear operator output vector if True.
    structure : str, optional
        A string describing the structure of all operators.
        Only `upper_triangular` is currently supported.

    Returns
    -------
    est : float
        An underestimate of the 1-norm of the sparse matrix product of operator_seq.
    v : ndarray, optional
        The vector such that ||Av||_1 == est*||v||_1.
        It can be thought of as an input to the linear operator
        that gives an output with particularly large norm.
    w : ndarray, optional
        The vector Av which has relatively large 1-norm.
        It can be thought of as an output of the linear operator
        that is relatively large in norm compared to the input.

    """
    # 使用 SciPy 中的 sparse 模块计算给定线性算子序列的一范数估计值
    return scipy.sparse.linalg.onenormest(
            # 创建一个 ProductOperator 对象，传入操作符序列和结构参数
            ProductOperator(*operator_seq, structure=structure))
    def d10_tight(self):
        """
        Property method to compute and return the tight 10th root of the one-norm of A10.

        If not already computed, computes and caches the exact one-norm of A10 raised to the power of 1/10.

        Returns
        -------
        float
            The exact 10th root of the one-norm of A10.
        """
        if self._d10_exact is None:
            self._d10_exact = _onenorm(self.A10)**(1/10.)
        return self._d10_exact
    # 如果 self._d10_exact 属性为 None，则计算其值并赋给 self._d10_exact
    if self._d10_exact is None:
        self._d10_exact = _onenorm(self.A10)**(1/10.)
    # 返回属性 self._d10_exact 的值
    return self._d10_exact

@property
def d4_loose(self):
    # 如果使用精确的单范数计算，则返回 self.d4_tight 的值
    if self.use_exact_onenorm:
        return self.d4_tight
    # 如果 self._d4_exact 属性不为 None，则返回其值
    if self._d4_exact is not None:
        return self._d4_exact
    else:
        # 如果 self._d4_approx 属性为 None，则计算其值
        if self._d4_approx is None:
            self._d4_approx = _onenormest_matrix_power(self.A2, 2,
                    structure=self.structure)**(1/4.)
        # 返回属性 self._d4_approx 的值
        return self._d4_approx

@property
def d6_loose(self):
    # 如果使用精确的单范数计算，则返回 self.d6_tight 的值
    if self.use_exact_onenorm:
        return self.d6_tight
    # 如果 self._d6_exact 属性不为 None，则返回其值
    if self._d6_exact is not None:
        return self._d6_exact
    else:
        # 如果 self._d6_approx 属性为 None，则计算其值
        if self._d6_approx is None:
            self._d6_approx = _onenormest_matrix_power(self.A2, 3,
                    structure=self.structure)**(1/6.)
        # 返回属性 self._d6_approx 的值
        return self._d6_approx

@property
def d8_loose(self):
    # 如果使用精确的单范数计算，则返回 self.d8_tight 的值
    if self.use_exact_onenorm:
        return self.d8_tight
    # 如果 self._d8_exact 属性不为 None，则返回其值
    if self._d8_exact is not None:
        return self._d8_exact
    else:
        # 如果 self._d8_approx 属性为 None，则计算其值
        if self._d8_approx is None:
            self._d8_approx = _onenormest_matrix_power(self.A4, 2,
                    structure=self.structure)**(1/8.)
        # 返回属性 self._d8_approx 的值
        return self._d8_approx

@property
def d10_loose(self):
    # 如果使用精确的单范数计算，则返回 self.d10_tight 的值
    if self.use_exact_onenorm:
        return self.d10_tight
    # 如果 self._d10_exact 属性不为 None，则返回其值
    if self._d10_exact is not None:
        return self._d10_exact
    else:
        # 如果 self._d10_approx 属性为 None，则计算其值
        if self._d10_approx is None:
            self._d10_approx = _onenormest_product((self.A4, self.A6),
                    structure=self.structure)**(1/10.)
        # 返回属性 self._d10_approx 的值
        return self._d10_approx

def pade3(self):
    b = (120., 60., 12., 1.)
    # 计算 U 和 V 的值并返回
    U = _smart_matrix_product(self.A,
            b[3]*self.A2 + b[1]*self.ident,
            structure=self.structure)
    V = b[2]*self.A2 + b[0]*self.ident
    return U, V

def pade5(self):
    b = (30240., 15120., 3360., 420., 30., 1.)
    # 计算 U 和 V 的值并返回
    U = _smart_matrix_product(self.A,
            b[5]*self.A4 + b[3]*self.A2 + b[1]*self.ident,
            structure=self.structure)
    V = b[4]*self.A4 + b[2]*self.A2 + b[0]*self.ident
    return U, V

def pade7(self):
    b = (17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.)
    # 计算 U 和 V 的值并返回
    U = _smart_matrix_product(self.A,
            b[7]*self.A6 + b[5]*self.A4 + b[3]*self.A2 + b[1]*self.ident,
            structure=self.structure)
    V = b[6]*self.A6 + b[4]*self.A4 + b[2]*self.A2 + b[0]*self.ident
    return U, V
    # 定义一个名为 pade9 的方法，用于计算九阶 Pade 近似
    def pade9(self):
        # 定义 Pade 近似的系数 b
        b = (17643225600., 8821612800., 2075673600., 302702400., 30270240.,
                2162160., 110880., 3960., 90., 1.)
        # 计算 U 矩阵，通过智能矩阵乘法 _smart_matrix_product 计算得到
        U = _smart_matrix_product(self.A,
                (b[9]*self.A8 + b[7]*self.A6 + b[5]*self.A4 +
                    b[3]*self.A2 + b[1]*self.ident),
                structure=self.structure)
        # 计算 V 矩阵
        V = (b[8]*self.A8 + b[6]*self.A6 + b[4]*self.A4 +
                b[2]*self.A2 + b[0]*self.ident)
        # 返回计算得到的 U 和 V
        return U, V

    # 定义一个名为 pade13_scaled 的方法，用于计算带缩放参数的十三阶 Pade 近似
    def pade13_scaled(self, s):
        # 定义带缩放参数 s 的 Pade 近似的系数 b
        b = (64764752532480000., 32382376266240000., 7771770303897600.,
                1187353796428800., 129060195264000., 10559470521600.,
                670442572800., 33522128640., 1323241920., 40840800., 960960.,
                16380., 182., 1.)
        # 根据缩放参数 s 计算 B, B2, B4, B6
        B = self.A * 2**-s
        B2 = self.A2 * 2**(-2*s)
        B4 = self.A4 * 2**(-4*s)
        B6 = self.A6 * 2**(-6*s)
        # 计算 U2 矩阵
        U2 = _smart_matrix_product(B6,
                b[13]*B6 + b[11]*B4 + b[9]*B2,
                structure=self.structure)
        # 计算 U 矩阵
        U = _smart_matrix_product(B,
                (U2 + b[7]*B6 + b[5]*B4 +
                    b[3]*B2 + b[1]*self.ident),
                structure=self.structure)
        # 计算 V2 矩阵
        V2 = _smart_matrix_product(B6,
                b[12]*B6 + b[10]*B4 + b[8]*B2,
                structure=self.structure)
        # 计算 V 矩阵
        V = V2 + b[6]*B6 + b[4]*B4 + b[2]*B2 + b[0]*self.ident
        # 返回计算得到的 U 和 V
        return U, V
def expm(A):
    """
    Compute the matrix exponential using Pade approximation.

    Parameters
    ----------
    A : (M,M) array_like or sparse matrix
        2D Array or Matrix (sparse or dense) to be exponentiated

    Returns
    -------
    expA : (M,M) ndarray
        Matrix exponential of `A`

    Notes
    -----
    This is algorithm (6.1) which is a simplification of algorithm (5.1).

    .. versionadded:: 0.12.0

    References
    ----------
    .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2009)
           "A New Scaling and Squaring Algorithm for the Matrix Exponential."
           SIAM Journal on Matrix Analysis and Applications.
           31 (3). pp. 970-989. ISSN 1095-7162

    Examples
    --------
    >>> from scipy.sparse import csc_matrix
    >>> from scipy.sparse.linalg import expm
    >>> A = csc_matrix([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    >>> A.toarray()
    array([[1, 0, 0],
           [0, 2, 0],
           [0, 0, 3]], dtype=int64)
    >>> Aexp = expm(A)
    >>> Aexp
    <Compressed Sparse Column sparse matrix of dtype 'float64'
        with 3 stored elements and shape (3, 3)>
    >>> Aexp.toarray()
    array([[  2.71828183,   0.        ,   0.        ],
           [  0.        ,   7.3890561 ,   0.        ],
           [  0.        ,   0.        ,  20.08553692]])
    """
    return _expm(A, use_exact_onenorm='auto')


def _expm(A, use_exact_onenorm):
    """
    Core of expm, separated to allow testing exact and approximate
    algorithms.

    Parameters
    ----------
    A : ndarray or sparse matrix
        2D Array or Matrix (sparse or dense) to be exponentiated
    use_exact_onenorm : str
        Strategy for determining if exact one-norm should be used

    Returns
    -------
    expA : ndarray or sparse matrix
        Matrix exponential of `A`

    Raises
    ------
    ValueError
        If `A` is not a square matrix

    Notes
    -----
    This function handles both dense and sparse matrices. It calculates
    the matrix exponential using a scaling and squaring approach.

    References
    ----------
    Algorithmic details are based on Awad H. Al-Mohy and Nicholas J. Higham's
    paper "A New Scaling and Squaring Algorithm for the Matrix Exponential."

    Examples
    --------
    >>> A = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    >>> _expm(A, use_exact_onenorm='auto')
    array([[  2.71828183,   0.        ,   0.        ],
           [  0.        ,   7.3890561 ,   0.        ],
           [  0.        ,   0.        ,  20.08553692]])

    """
    # Avoid indiscriminate asarray() to allow sparse or other strange arrays.
    if isinstance(A, (list, tuple, np.matrix)):
        A = np.asarray(A)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected a square matrix')

    # gracefully handle size-0 input,
    # carefully handling sparse scenario
    if A.shape == (0, 0):
        out = np.zeros([0, 0], dtype=A.dtype)
        if issparse(A) or is_pydata_spmatrix(A):
            return A.__class__(out)
        return out

    # Trivial case
    if A.shape == (1, 1):
        out = [[np.exp(A[0, 0])]]

        # Avoid indiscriminate casting to ndarray to
        # allow for sparse or other strange arrays
        if issparse(A) or is_pydata_spmatrix(A):
            return A.__class__(out)

        return np.array(out)

    # Ensure input is of float type, to avoid integer overflows etc.
    if ((isinstance(A, np.ndarray) or issparse(A) or is_pydata_spmatrix(A))
            and not np.issubdtype(A.dtype, np.inexact)):
        A = A.astype(float)

    # Detect upper triangularity.
    structure = UPPER_TRIANGULAR if _is_upper_triangular(A) else None

    if use_exact_onenorm == "auto":
        # Hardcode a matrix order threshold for exact vs. estimated one-norms.
        use_exact_onenorm = A.shape[0] < 200

    # Track functions of A to help compute the matrix exponential.
    h = _ExpmPadeHelper(
            A, structure=structure, use_exact_onenorm=use_exact_onenorm)

    # Try Pade order 3.
    # 计算 eta_1，选择最大的 d4_loose 和 d6_loose
    eta_1 = max(h.d4_loose, h.d6_loose)
    
    # 如果 eta_1 小于指定阈值且 A 的 3 阶椭圆函数为零
    if eta_1 < 1.495585217958292e-002 and _ell(h.A, 3) == 0:
        # 调用 Pade 近似阶数为 3 的方法
        U, V = h.pade3()
        # 返回 Pade 近似求解结果
        return _solve_P_Q(U, V, structure=structure)

    # 尝试 Pade 近似阶数为 5
    eta_2 = max(h.d4_tight, h.d6_loose)
    
    # 如果 eta_2 小于指定阈值且 A 的 5 阶椭圆函数为零
    if eta_2 < 2.539398330063230e-001 and _ell(h.A, 5) == 0:
        # 调用 Pade 近似阶数为 5 的方法
        U, V = h.pade5()
        # 返回 Pade 近似求解结果
        return _solve_P_Q(U, V, structure=structure)

    # 尝试 Pade 近似阶数为 7 或 9
    eta_3 = max(h.d6_tight, h.d8_loose)
    
    # 如果 eta_3 小于指定阈值且 A 的 7 阶椭圆函数为零
    if eta_3 < 9.504178996162932e-001 and _ell(h.A, 7) == 0:
        # 调用 Pade 近似阶数为 7 的方法
        U, V = h.pade7()
        # 返回 Pade 近似求解结果
        return _solve_P_Q(U, V, structure=structure)
    
    # 如果 eta_3 小于指定阈值且 A 的 9 阶椭圆函数为零
    if eta_3 < 2.097847961257068e+000 and _ell(h.A, 9) == 0:
        # 调用 Pade 近似阶数为 9 的方法
        U, V = h.pade9()
        # 返回 Pade 近似求解结果
        return _solve_P_Q(U, V, structure=structure)

    # 使用 Pade 近似阶数为 13
    eta_4 = max(h.d8_loose, h.d10_loose)
    eta_5 = min(eta_3, eta_4)
    theta_13 = 4.25

    # 选择最小的 s>=0，使得 2**(-s) * eta_5 <= theta_13
    if eta_5 == 0:
        # 特殊情况，当 eta_5 为零时
        s = 0
    else:
        # 计算 s，确保条件满足
        s = max(int(np.ceil(np.log2(eta_5 / theta_13))), 0)
    # 计算 s，作为输入矩阵的阶数，再加上 A 的 s 阶椭圆函数
    s = s + _ell(2**-s * h.A, 13)
    # 调用 Pade 近似阶数为 13 的方法，并进行缩放
    U, V = h.pade13_scaled(s)
    # 使用 P-Q 解算法求解结果 X
    X = _solve_P_Q(U, V, structure=structure)
    if structure == UPPER_TRIANGULAR:
        # 如果结构是上三角形，调用代码片段 2.1
        X = _fragment_2_1(X, h.A, s)
    else:
        # 否则，通过重复平方得到 r_13(A)^(2^s)
        for i in range(s):
            X = X.dot(X)
    # 返回最终结果 X
    return X
# 定义一个辅助函数用于计算 Pade 近似的 P 和 Q
def _solve_P_Q(U, V, structure=None):
    """
    A helper function for expm_2009.

    Parameters
    ----------
    U : ndarray
        Pade 近似的分子。
    V : ndarray
        Pade 近似的分母。
    structure : str, optional
        描述矩阵 U 和 V 结构的字符串。
        目前仅支持 'upper_triangular'。

    Notes
    -----
    参数 `structure` 受到 theano 和 cvxopt 函数类似参数的启发。
    """
    # 计算 P = U + V
    P = U + V
    # 计算 Q = -U + V
    Q = -U + V
    # 如果 U 是稀疏矩阵或者 pydata_spmatrix，则使用 spsolve 求解线性方程组 Qx = P
    if issparse(U) or is_pydata_spmatrix(U):
        return spsolve(Q, P)
    # 如果 structure 为 None，则使用 solve 求解线性方程组 Qx = P
    elif structure is None:
        return solve(Q, P)
    # 如果 structure 为 'upper_triangular'，则使用 solve_triangular 求解上三角线性方程组 Qx = P
    elif structure == UPPER_TRIANGULAR:
        return solve_triangular(Q, P)
    else:
        # 抛出错误，指出不支持的矩阵结构
        raise ValueError('unsupported matrix structure: ' + str(structure))


def _exp_sinch(a, x):
    """
    Stably evaluate exp(a)*sinh(x)/x

    Notes
    -----
    当 x 较小时，使用六阶泰勒展开稳定地计算 exp(a)*sinh(x)/x。
    这里“较小”的界限是相对误差小于 1e-14 的点。
    当 x 较大时，直接计算 sinh(x) / x。

    该策略参考了 Spallation Neutron Source 文档中的建议，可以通过谷歌搜索获得。
    http://www.ornl.gov/~t6p/resources/xal/javadoc/gov/sns/tools/math/ElementaryFunction.html
    我们选择了截止点和 Horner 形式的评估，没有特定的参考。

    需要注意的是 scipy.special 中目前没有实现 sinch 函数，而“工程师”的定义中实现了 sinc 函数。
    sinc 函数的实现包括一个与“数学家”的版本不同的 π 缩放因子。

    """
    # 如果 x 很小，则使用六阶泰勒展开
    if abs(x) < 0.0135:
        x2 = x*x
        return np.exp(a) * (1 + (x2/6.)*(1 + (x2/20.)*(1 + (x2/42.))))
    else:
        # 否则计算 (exp(a+x) - exp(a-x)) / (2*x)
        return (np.exp(a + x) - np.exp(a - x)) / (2*x)


def _eq_10_42(lam_1, lam_2, t_12):
    """
    Equation (10.42) of Functions of Matrices: Theory and Computation.

    Notes
    -----
    这是 expm_2009 中 _fragment_2_1 的辅助函数。
    方程 (10.42) 在《Functions of Matrices: Theory and Computation》中。
    第 251 页的 Schur 算法章节中有详细解释。
    特别地，10.4.3 节介绍了 Schur-Parlett 算法。
    expm([[lam_1, t_12], [0, lam_1])
    =
    [[exp(lam_1), t_12*exp((lam_1 + lam_2)/2)*sinch((lam_1 - lam_2)/2)],
    [0, exp(lam_2)]
    """

    # 普通公式 t_12 * (exp(lam_2) - exp(lam_2)) / (lam_2 - lam_1)
    # 显然受到 Higham 的教材中的取消影响。
    # 一个良好的 sinch 实现，定义为 sinh(x)/x，显然能避免这种取消影响。
    a = 0.5 * (lam_1 + lam_2)
    b = 0.5 * (lam_1 - lam_2)
    return t_12 * _exp_sinch(a, b)


def _fragment_2_1(X, T, s):
    """
    A helper function for expm_2009.

    Notes
    -----
    X 参数会就地修改，但这种修改与
    # Form X = r_m(2^-s T)
    # 创建一个矩阵 X，其元素为 r_m(2^-s T)，r_m 表示某种矩阵函数。
    n = X.shape[0]
    # 复制 T 矩阵的对角线，并展平成一维数组
    diag_T = np.ravel(T.diagonal().copy())
    
    # Replace diag(X) by exp(2^-s diag(T)).
    # 将 X 的对角线替换为 exp(2^-s diag(T))
    scale = 2 ** -s
    exp_diag = np.exp(scale * diag_T)
    for k in range(n):
        X[k, k] = exp_diag[k]
    
    # Perform matrix exponentiation X = X^2, starting from s-1 down to 0.
    # 从 s-1 到 0，对 X 进行矩阵指数运算 X = X^2。
    for i in range(s-1, -1, -1):
        X = X.dot(X)
    
        # Replace diag(X) by exp(2^-i diag(T)).
        # 将 X 的对角线替换为 exp(2^-i diag(T))
        scale = 2 ** -i
        exp_diag = np.exp(scale * diag_T)
        for k in range(n):
            X[k, k] = exp_diag[k]
    
        # Replace (first) superdiagonal of X by explicit formula
        # 根据作者2008年教材中第10.42节的公式，将 X 的第一个超对角线替换为 exp(2^-i T) 的明确公式。
        for k in range(n-1):
            lam_1 = scale * diag_T[k]
            lam_2 = scale * diag_T[k+1]
            t_12 = scale * T[k, k+1]
            # 使用 _eq_10_42 函数计算值
            value = _eq_10_42(lam_1, lam_2, t_12)
            X[k, k+1] = value
    
    # Return the updated X matrix.
    # 返回更新后的矩阵 X。
    return X
def matrix_power(A, power):
    """
    Raise a square matrix to the integer power, `power`.

    For non-negative integers, ``A**power`` is computed using repeated
    matrix multiplications. Negative integers are not supported. 

    Parameters
    ----------
    A : (M, M) square sparse array or matrix
        sparse array that will be raised to power `power`
    power : int
        Exponent used to raise sparse array `A`

    Returns
    -------
    A**power : (M, M) sparse array or matrix
        The output matrix will be the same shape as A, and will preserve
        the class of A, but the format of the output may be changed.
    
    Notes
    -----
    This uses a recursive implementation of the matrix power. For computing
    the matrix power using a reasonably large `power`, this may be less efficient
    than computing the product directly, using A @ A @ ... @ A.
    This is contingent upon the number of nonzero entries in the matrix. 

    .. versionadded:: 1.12.0

    Examples
    --------
    >>> from scipy import sparse
    >>> A = sparse.csc_array([[0,1,0],[1,0,1],[0,1,0]])
    >>> A.todense()
    array([[0, 1, 0],
           [1, 0, 1],
           [0, 1, 0]])
    >>> (A @ A).todense()
    array([[1, 0, 1],
           [0, 2, 0],
           [1, 0, 1]])
    >>> A2 = sparse.linalg.matrix_power(A, 2)
    >>> A2.todense()
    array([[1, 0, 1],
           [0, 2, 0],
           [1, 0, 1]])
    >>> A4 = sparse.linalg.matrix_power(A, 4)
    >>> A4.todense()
    array([[2, 0, 2],
           [0, 4, 0],
           [2, 0, 2]])

    """
    M, N = A.shape
    # 检查 A 是否为方阵
    if M != N:
        raise TypeError('sparse matrix is not square')
    # 如果参数 power 被判断为类似整数类型的值，则执行以下逻辑
    if isintlike(power):
        # 将 power 转换为整数类型
        power = int(power)
        # 如果 power 小于 0，则抛出数值错误异常
        if power < 0:
            raise ValueError('exponent must be >= 0')

        # 如果 power 等于 0，则返回一个 MxM 的单位矩阵，数据类型与 A 的数据类型相同
        if power == 0:
            return eye(M, dtype=A.dtype)

        # 如果 power 等于 1，则返回 A 的一个副本
        if power == 1:
            return A.copy()

        # 计算 A 的 power // 2 次幂的矩阵
        tmp = matrix_power(A, power // 2)
        # 如果 power 是奇数，则返回 A @ tmp @ tmp
        if power % 2:
            return A @ tmp @ tmp
        # 如果 power 是偶数，则返回 tmp @ tmp
        else:
            return tmp @ tmp
    else:
        # 如果 power 不是整数类型的值，则抛出数值错误异常
        raise ValueError("exponent must be an integer")
```