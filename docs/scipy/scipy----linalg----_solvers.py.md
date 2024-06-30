# `D:\src\scipysrc\scipy\scipy\linalg\_solvers.py`

```
"""Matrix equation solver routines"""
# 作者：Jeffrey Armstrong <jeff@approximatrix.com>
# 日期：2012年2月24日

# 修改者：Chad Fulton <ChadFulton@gmail.com>
# 日期：2014年6月19日

# 修改者：Ilhan Polat <ilhanpolat@gmail.com>
# 日期：2016年9月13日

# 导入警告模块
import warnings
# 导入 NumPy 库并重命名为 np
import numpy as np
# 从 NumPy 的线性代数模块中导入特定函数
from numpy.linalg import inv, LinAlgError, norm, cond, svd

# 从内部模块导入特定函数和类
from ._basic import solve, solve_triangular, matrix_balance
from .lapack import get_lapack_funcs
from ._decomp_schur import schur
from ._decomp_lu import lu
from ._decomp_qr import qr
from ._decomp_qz import ordqz
from ._decomp import _asarray_validated
from ._special_matrices import kron, block_diag

# 定义可以从当前模块导入的公共接口
__all__ = ['solve_sylvester',
           'solve_continuous_lyapunov', 'solve_discrete_lyapunov',
           'solve_lyapunov',
           'solve_continuous_are', 'solve_discrete_are']

# 定义解 Sylvester 方程的函数
def solve_sylvester(a, b, q):
    """
    Computes a solution (X) to the Sylvester equation :math:`AX + XB = Q`.

    Parameters
    ----------
    a : (M, M) array_like
        Leading matrix of the Sylvester equation
    b : (N, N) array_like
        Trailing matrix of the Sylvester equation
    q : (M, N) array_like
        Right-hand side

    Returns
    -------
    x : (M, N) ndarray
        The solution to the Sylvester equation.

    Raises
    ------
    LinAlgError
        If solution was not found

    Notes
    -----
    Computes a solution to the Sylvester matrix equation via the Bartels-
    Stewart algorithm. The A and B matrices first undergo Schur
    decompositions. The resulting matrices are used to construct an
    alternative Sylvester equation (``RY + YS^T = F``) where the R and S
    matrices are in quasi-triangular form (or, when R, S or F are complex,
    triangular form). The simplified equation is then solved using
    ``*TRSYL`` from LAPACK directly.

    .. versionadded:: 0.11.0

    Examples
    --------
    Given `a`, `b`, and `q` solve for `x`:

    >>> import numpy as np
    >>> from scipy import linalg
    >>> a = np.array([[-3, -2, 0], [-1, -1, 3], [3, -5, -1]])
    >>> b = np.array([[1]])
    >>> q = np.array([[1],[2],[3]])
    >>> x = linalg.solve_sylvester(a, b, q)
    >>> x
    array([[ 0.0625],
           [-0.5625],
           [ 0.6875]])
    >>> np.allclose(a.dot(x) + x.dot(b), q)
    True

    """

    # 计算矩阵 a 的 Schur 分解形式
    r, u = schur(a, output='real')

    # 计算矩阵 b 的 Schur 分解形式
    s, v = schur(b.conj().transpose(), output='real')

    # 构造 f = u'*q*v
    f = np.dot(np.dot(u.conj().transpose(), q), v)

    # 调用 Sylvester 方程求解器
    trsyl, = get_lapack_funcs(('trsyl',), (r, s, f))
    if trsyl is None:
        raise RuntimeError('LAPACK implementation does not contain a proper '
                           'Sylvester equation solver (TRSYL)')
    y, scale, info = trsyl(r, s, f, tranb='C')

    y = scale*y
    # 如果 info 小于 0，则抛出线性代数错误异常，说明出现了非法值
    if info < 0:
        raise LinAlgError("Illegal value encountered in "
                          "the %d term" % (-info,))
    # 返回矩阵乘积 u * y * v 的共轭转置
    return np.dot(np.dot(u, y), v.conj().transpose())
# 解决连续 Lyapunov 方程：AX + XA^H = Q
# 使用 Bartels-Stewart 算法找到 X

def solve_continuous_lyapunov(a, q):
    """
    Solves the continuous Lyapunov equation :math:`AX + XA^H = Q`.

    Uses the Bartels-Stewart algorithm to find :math:`X`.

    Parameters
    ----------
    a : array_like
        A square matrix

    q : array_like
        Right-hand side square matrix

    Returns
    -------
    x : ndarray
        Solution to the continuous Lyapunov equation

    See Also
    --------
    solve_discrete_lyapunov : computes the solution to the discrete-time
        Lyapunov equation
    solve_sylvester : computes the solution to the Sylvester equation

    Notes
    -----
    The continuous Lyapunov equation is a special form of the Sylvester
    equation, hence this solver relies on LAPACK routine ?TRSYL.

    .. versionadded:: 0.11.0

    Examples
    --------
    Given `a` and `q` solve for `x`:

    >>> import numpy as np
    >>> from scipy import linalg
    >>> a = np.array([[-3, -2, 0], [-1, -1, 0], [0, -5, -1]])
    >>> b = np.array([2, 4, -1])
    >>> q = np.eye(3)
    >>> x = linalg.solve_continuous_lyapunov(a, q)
    >>> x
    array([[ -0.75  ,   0.875 ,  -3.75  ],
           [  0.875 ,  -1.375 ,   5.3125],
           [ -3.75  ,   5.3125, -27.0625]])
    >>> np.allclose(a.dot(x) + x.dot(a.T), q)
    True
    """

    # 将 a 和 q 转换为至少是二维的数组，并验证是否有限
    a = np.atleast_2d(_asarray_validated(a, check_finite=True))
    q = np.atleast_2d(_asarray_validated(q, check_finite=True))

    # 初始化 r_or_c 为 float 类型
    r_or_c = float

    # 确定 a 和 q 中是否有复数元素，并设置 r_or_c 的类型
    for ind, _ in enumerate((a, q)):
        if np.iscomplexobj(_):
            r_or_c = complex

        # 检查矩阵是否是方阵
        if not np.equal(*_.shape):
            raise ValueError("Matrix {} should be square.".format("aq"[ind]))

    # 检查 a 和 q 的形状是否相同
    if a.shape != q.shape:
        raise ValueError("Matrix a and q should have the same shape.")

    # 计算矩阵 a 的 Schur 分解形式
    r, u = schur(a, output='real')

    # 构造 f = u'*q*u
    f = u.conj().T.dot(q.dot(u))

    # 调用 Sylvester 方程的求解器
    trsyl = get_lapack_funcs('trsyl', (r, f))

    # 设置数据类型字符串
    dtype_string = 'T' if r_or_c == float else 'C'

    # 调用 LAPACK 函数 trsyl 解决方程
    y, scale, info = trsyl(r, r, f, tranb=dtype_string)

    # 处理 LAPACK 函数的返回信息
    if info < 0:
        raise ValueError('?TRSYL exited with the internal error '
                         f'"illegal value in argument number {-info}.". See '
                         'LAPACK documentation for the ?TRSYL error codes.')
    elif info == 1:
        warnings.warn('Input "a" has an eigenvalue pair whose sum is '
                      'very close to or exactly zero. The solution is '
                      'obtained via perturbing the coefficients.',
                      RuntimeWarning, stacklevel=2)

    # 乘以比例因子 scale
    y *= scale

    # 计算最终结果 x = u * y * u^H
    return u.dot(y).dot(u.conj().T)


# 为了向后兼容，保留旧名称
solve_lyapunov = solve_continuous_lyapunov


def _solve_discrete_lyapunov_direct(a, q):
    """
    Solves the discrete Lyapunov equation directly.

    This function is called by the `solve_discrete_lyapunov` function with
    """
    """
    `method=direct`. It is not supposed to be called directly.
    """

    # 使用 kron 函数计算矩阵 a 与其共轭的 Kronecker 乘积，赋值给 lhs
    lhs = kron(a, a.conj())
    # 构造一个单位矩阵，减去 lhs，得到一个新的矩阵 lhs
    lhs = np.eye(lhs.shape[0]) - lhs
    # 解线性方程 lhs * x = q.flatten()，求解 x
    x = solve(lhs, q.flatten())

    # 将 x 重新调整为原始形状 q.shape，并返回
    return np.reshape(x, q.shape)
# 将输入的矩阵 `a` 和 `q` 转换为 NumPy 数组
a = np.asarray(a)
q = np.asarray(q)
    # 如果未指定求解方法，则根据矩阵大小自动选择方法
    if method is None:
        # 如果矩阵 a 的行数大于等于 10，则选择双线性插值法
        if a.shape[0] >= 10:
            method = 'bilinear'
        else:
            # 否则选择直接法
            method = 'direct'

    # 将方法名称转换为小写
    meth = method.lower()

    # 根据选择的方法求解离散 Lyapunov 方程
    if meth == 'direct':
        # 使用直接法求解离散 Lyapunov 方程
        x = _solve_discrete_lyapunov_direct(a, q)
    elif meth == 'bilinear':
        # 使用双线性插值法求解离散 Lyapunov 方程
        x = _solve_discrete_lyapunov_bilinear(a, q)
    else:
        # 若方法不是 'direct' 或 'bilinear'，则引发值错误异常
        raise ValueError('Unknown solver %s' % method)

    # 返回求解得到的结果 x
    return x
# 解决连续时间代数Riccati方程（CARE）

# CARE方程定义如下：
# X A + A^H X - X B R^{-1} B^H X + Q = 0

# 当存在解时的限制条件包括：
# * A的所有特征值在右半平面，应可控。
# * 相关的哈密顿Pencil（见Notes部分），其特征值应远离虚轴。

# 如果e或s不是None，则解决广义版本的CARE：
# E^HXA + A^HXE - (E^HXB + S) R^{-1} (B^HXE + S^H) + Q = 0
# 当省略时，假定e为单位矩阵，s为与a和b兼容的零矩阵。

def solve_continuous_are(a, b, q, r, e=None, s=None, balanced=True):
    r"""
    Solves the continuous-time algebraic Riccati equation (CARE).

    The CARE is defined as

    .. math::

          X A + A^H X - X B R^{-1} B^H X + Q = 0

    The limitations for a solution to exist are :

        * All eigenvalues of :math:`A` on the right half plane, should be
          controllable.

        * The associated hamiltonian pencil (See Notes), should have
          eigenvalues sufficiently away from the imaginary axis.

    Moreover, if ``e`` or ``s`` is not precisely ``None``, then the
    generalized version of CARE

    .. math::

          E^HXA + A^HXE - (E^HXB + S) R^{-1} (B^HXE + S^H) + Q = 0

    is solved. When omitted, ``e`` is assumed to be the identity and ``s``
    is assumed to be the zero matrix with sizes compatible with ``a`` and
    ``b``, respectively.

    Parameters
    ----------
    a : (M, M) array_like
        Square matrix
    b : (M, N) array_like
        Input
    q : (M, M) array_like
        Input
    r : (N, N) array_like
        Nonsingular square matrix
    e : (M, M) array_like, optional
        Nonsingular square matrix
    s : (M, N) array_like, optional
        Input
    balanced : bool, optional
        The boolean that indicates whether a balancing step is performed
        on the data. The default is set to True.

    Returns
    -------
    x : (M, M) ndarray
        Solution to the continuous-time algebraic Riccati equation.

    Raises
    ------
    LinAlgError
        For cases where the stable subspace of the pencil could not be
        isolated. See Notes section and the references for details.

    See Also
    --------
    solve_discrete_are : Solves the discrete-time algebraic Riccati equation

    Notes
    -----
    The equation is solved by forming the extended hamiltonian matrix pencil,
    as described in [1]_, :math:`H - \lambda J` given by the block matrices ::

        [ A    0    B ]             [ E   0    0 ]
        [-Q  -A^H  -S ] - \lambda * [ 0  E^H   0 ]
        [ S^H B^H   R ]             [ 0   0    0 ]

    and using a QZ decomposition method.

    In this algorithm, the fail conditions are linked to the symmetry
    of the product :math:`U_2 U_1^{-1}` and condition number of
    :math:`U_1`. Here, :math:`U` is the 2m-by-m matrix that holds the
    eigenvectors spanning the stable subspace with 2-m rows and partitioned
    into two m-row matrices. See [1]_ and [2]_ for more details.

    In order to improve the QZ decomposition accuracy, the pencil goes
    through a balancing step where the sum of absolute values of
    :math:`H` and :math:`J` entries (after removing the diagonal entries of
    the sum) is balanced following the recipe given in [3]_.

    .. versionadded:: 0.11.0

    References
    ----------
    """
    # Validate input arguments
    # 调用 _are_validate_args 函数验证输入参数的正确性，并获取返回值
    a, b, q, r, e, s, m, n, r_or_c, gen_are = _are_validate_args(
                                                     a, b, q, r, e, s, 'care')

    # 创建一个空的 H 矩阵，大小为 (2*m+n) x (2*m+n)，数据类型为 r_or_c
    H = np.empty((2*m+n, 2*m+n), dtype=r_or_c)
    # 将矩阵 a 复制到 H 的左上角 m x m 部分
    H[:m, :m] = a
    # 将 H 的左上角 m x m 部分右边的 m x m 部分设置为零矩阵
    H[:m, m:2*m] = 0.
    # 将 b 矩阵复制到 H 的左上角 m x m 部分右边的 m x n 部分
    H[:m, 2*m:] = b
    # 将 H 的左下角 m x m 部分设置为 -q
    H[m:2*m, :m] = -q
    # 将 H 的左下角 m x m 部分右边的 m x m 部分设置为 -a 的共轭转置
    H[m:2*m, m:2*m] = -a.conj().T
    # 将 H 的左下角 m x m 部分右边的 m x n 部分设置为零矩阵或者 -s，如果 s 不为 None 的话
    H[m:2*m, 2*m:] = 0. if s is None else -s
    # 将 H 的右下角 m x n 部分的上面 m x m 部分设置为零矩阵或者 s 的共轭转置，如果 s 不为 None 的话
    H[2*m:, :m] = 0. if s is None else s.conj().T
    # 将 H 的右下角 m x n 部分的上面 m x m 部分右边的 m x n 部分设置为 b 的共轭转置
    H[2*m:, m:2*m] = b.conj().T
    # 将 H 的右下角 m x n 部分设置为 r
    H[2*m:, 2*m:] = r

    # 如果 gen_are 为 True，并且 e 不为 None，则根据 gen_are 的设置选择块对角矩阵 J
    if gen_are and e is not None:
        J = block_diag(e, e.conj().T, np.zeros_like(r, dtype=r_or_c))
    else:
        # 否则选择块对角矩阵 J，其中对角块为 2m x 2m 的单位矩阵
        J = block_diag(np.eye(2*m), np.zeros_like(r, dtype=r_or_c))

    # 如果 balanced 为 True，则根据 Ref.3 进行矩阵平衡操作
    if balanced:
        # 计算 M 矩阵，M 的元素为 H 和 J 元素绝对值的和
        M = np.abs(H) + np.abs(J)
        # 将 M 的对角线元素设置为零
        np.fill_diagonal(M, 0.)
        # 对 M 进行平衡操作，获取平衡后的矩阵及其对角元素
        _, (sca, _) = matrix_balance(M, separate=1, permute=0)
        # 如果 sca 不全等于 1，则根据 Benner 的方法调整 H 和 J
        if not np.allclose(sca, np.ones_like(sca)):
            # 计算 sca 的对数
            sca = np.log2(sca)
            # 计算 s 向量，通过 sca 计算得到
            s = np.round((sca[m:2*m] - sca[:m])/2)
            # 计算 elwisescale 矩阵，元素为 sca 向量的广播乘积
            elwisescale = sca[:, None] * np.reciprocal(sca)
            # 对 H 和 J 进行元素级别的乘法操作
            H *= elwisescale
            J *= elwisescale

    # 将 H 的右侧 n 列进行 QR 分解，返回正交矩阵 q 和 上三角矩阵 r
    q, r = qr(H[:, -n:])
    # 将 H 进行缩减至 2m x 2m 的大小，按照 Ref.1 中的方法进行处理
    H = q[:, n:].conj().T.dot(H[:, :2*m])
    # 将 J 缩减至 2m x 2m 的大小，按照 Ref.1 中的方法进行处理
    J = q[:2*m, n:].conj().T.dot(J[:2*m, :2*m])

    # 根据 r_or_c 的类型决定输出类型的字符串
    out_str = 'real' if r_or_c == float else 'complex'
    ```
    # 调用 ordqz 函数，返回结果的第六个元素 u
    _, _, _, _, _, u = ordqz(H, J, sort='lhp', overwrite_a=True,
                             overwrite_b=True, check_finite=False,
                             output=out_str)

    # 如果存在非空矩阵 e，则对稳定子空间基础的相关部分进行 QR 分解
    if e is not None:
        u, _ = qr(np.vstack((e.dot(u[:m, :m]), u[m:, :m])))
    
    # 提取 u 的左上角 m × m 部分和右上角 m × m 部分
    u00 = u[:m, :m]
    u10 = u[m:, :m]

    # 对 u00 进行 LU 分解
    up, ul, uu = lu(u00)
    
    # 检查 uu 的条件数的倒数是否小于机器精度的阈值
    if 1/cond(uu) < np.spacing(1.):
        raise LinAlgError('Failed to find a finite solution.')

    # 利用三角结构解方程
    x = solve_triangular(ul.conj().T,
                         solve_triangular(uu.conj().T,
                                          u10.conj().T,
                                          lower=True),
                         unit_diagonal=True,
                         ).conj().T.dot(up.conj().T)
    
    # 如果 balanced 参数为真，则乘以对角阵 sca[:m, None] 和 sca[:m]
    if balanced:
        x *= sca[:m, None] * sca[:m]

    # 检查对称性偏差，参考文献 [2] 中 Thm.5 的第 3 条证明
    u_sym = u00.conj().T.dot(u10)
    n_u_sym = norm(u_sym, 1)
    u_sym = u_sym - u_sym.conj().T
    # 计算对称性阈值，为 1000*eps 或 0.1*n_u_sym 中的较大值
    sym_threshold = np.max([np.spacing(1000.), 0.1*n_u_sym])

    # 如果对称性偏差大于阈值，则抛出线性代数错误
    if norm(u_sym, 1) > sym_threshold:
        raise LinAlgError('The associated Hamiltonian pencil has eigenvalues '
                          'too close to the imaginary axis')

    # 返回修正后的对称解 x
    return (x + x.conj().T)/2
# 定义解离散时间代数Riccati方程（DARE）的函数
def solve_discrete_are(a, b, q, r, e=None, s=None, balanced=True):
    """
    Solves the discrete-time algebraic Riccati equation (DARE).

    The DARE is defined as

    .. math::

          A^H X A - X - (A^H X B) (R + B^H X B)^{-1} (B^H X A) + Q = 0

    The limitations for a solution to exist are :

        * All eigenvalues of :math:`A` outside the unit disc, should be
          controllable.

        * The associated symplectic pencil (See Notes), should have
          eigenvalues sufficiently away from the unit circle.

    Moreover, if ``e`` and ``s`` are not both precisely ``None``, then the
    generalized version of DARE

    .. math::

          A^H X A - E^H X E - (A^H X B + S) (R + B^H X B)^{-1} (B^H X A + S^H) + Q = 0

    is solved. When omitted, ``e`` is assumed to be the identity and ``s``
    is assumed to be the zero matrix.

    Parameters
    ----------
    a : (M, M) array_like
        Square matrix
    b : (M, N) array_like
        Input
    q : (M, M) array_like
        Input
    r : (N, N) array_like
        Square matrix
    e : (M, M) array_like, optional
        Nonsingular square matrix
    s : (M, N) array_like, optional
        Input
    balanced : bool
        The boolean that indicates whether a balancing step is performed
        on the data. The default is set to True.

    Returns
    -------
    x : (M, M) ndarray
        Solution to the discrete algebraic Riccati equation.

    Raises
    ------
    LinAlgError
        For cases where the stable subspace of the pencil could not be
        isolated. See Notes section and the references for details.

    See Also
    --------
    solve_continuous_are : Solves the continuous algebraic Riccati equation

    Notes
    -----
    The equation is solved by forming the extended symplectic matrix pencil,
    as described in [1]_, :math:`H - \lambda J` given by the block matrices ::

           [  A   0   B ]             [ E   0   B ]
           [ -Q  E^H -S ] - \lambda * [ 0  A^H  0 ]
           [ S^H  0   R ]             [ 0 -B^H  0 ]

    and using a QZ decomposition method.

    In this algorithm, the fail conditions are linked to the symmetry
    of the product :math:`U_2 U_1^{-1}` and condition number of
    :math:`U_1`. Here, :math:`U` is the 2m-by-m matrix that holds the
    eigenvectors spanning the stable subspace with 2-m rows and partitioned
    into two m-row matrices. See [1]_ and [2]_ for more details.

    In order to improve the QZ decomposition accuracy, the pencil goes
    through a balancing step where the sum of absolute values of
    :math:`H` and :math:`J` rows/cols (after removing the diagonal entries)
    is balanced following the recipe given in [3]_. If the data has small
    numerical noise, balancing may amplify their effects and some clean up
    is required.

    .. versionadded:: 0.11.0

    References
    ----------
    """
    """
    Validate input arguments and prepare for solving Discrete Algebraic Riccati Equation (DARE).

    Parameters
    ----------
    a : array_like
        State matrix of the system.
    b : array_like
        Input matrix of the system.
    q : array_like
        Cost matrix.
    r : array_like
        Control weighting matrix.
    e : array_like, optional
        If provided, overrides the default state space coupling matrix.
    s : array_like, optional
        If provided, overrides the default state space output matrix.
    m : int
        Size of the state matrix `a`.
    n : int
        Number of columns in `b`.
    r_or_c : {'real', 'complex'}, optional
        Type of the matrices (real or complex).
    gen_are : bool
        Whether to use the generalized algebraic Riccati equation solver.

    Returns
    -------
    a : array_like
        Validated state matrix `a`.
    b : array_like
        Validated input matrix `b`.
    q : array_like
        Validated cost matrix `q`.
    r : array_like
        Validated control weighting matrix `r`.
    e : array_like, optional
        Validated state space coupling matrix `e`.
    s : array_like, optional
        Validated state space output matrix `s`.
    m : int
        Size of the state matrix `a`.
    n : int
        Number of columns in `b`.
    r_or_c : {'real', 'complex'}
        Type of the matrices (real or complex).
    gen_are : bool
        Whether to use the generalized algebraic Riccati equation solver.

    Notes
    -----
    This function ensures all input arguments are correctly formatted and prepares
    them for solving the Discrete Algebraic Riccati Equation (DARE).
    """

    # Validate input arguments using a helper function
    a, b, q, r, e, s, m, n, r_or_c, gen_are = _are_validate_args(
                                                     a, b, q, r, e, s, 'dare')

    # Form the matrix pencil
    H = np.zeros((2*m+n, 2*m+n), dtype=r_or_c)
    H[:m, :m] = a
    H[:m, 2*m:] = b
    H[m:2*m, :m] = -q
    H[m:2*m, m:2*m] = np.eye(m) if e is None else e.conj().T
    H[m:2*m, 2*m:] = 0. if s is None else -s
    H[2*m:, :m] = 0. if s is None else s.conj().T
    H[2*m:, 2*m:] = r

    # Initialize J matrix for the pencil
    J = np.zeros_like(H, dtype=r_or_c)
    J[:m, :m] = np.eye(m) if e is None else e
    J[m:2*m, m:2*m] = a.conj().T
    J[2*m:, m:2*m] = -b.conj().T

    # Perform balancing if specified
    if balanced:
        # Calculate the scaling matrix M
        M = np.abs(H) + np.abs(J)
        np.fill_diagonal(M, 0.)
        # Balance the matrices using the matrix_balance function
        _, (sca, _) = matrix_balance(M, separate=1, permute=0)
        
        # Check if scaling matrix needs adjustment
        if not np.allclose(sca, np.ones_like(sca)):
            # Adjust scaling using logarithms and rounding
            sca = np.log2(sca)
            s = np.round((sca[m:2*m] - sca[:m])/2)
            sca = 2 ** np.r_[s, -s, sca[2*m:]]
            elwisescale = sca[:, None] * np.reciprocal(sca)
            H *= elwisescale
            J *= elwisescale

    # Deflate the pencil by the R column
    q_of_qr, _ = qr(H[:, -n:])
    H = q_of_qr[:, n:].conj().T.dot(H[:, :2*m])
    J = q_of_qr[:, n:].conj().T.dot(J[:, :2*m])

    # Determine the output type for QZ
    out_str = 'real' if r_or_c == float else 'complex'
    # 调用 ordqz 函数，并将返回的结果赋值给变量 u
    _, _, _, _, _, u = ordqz(H, J, sort='iuc',
                             overwrite_a=True,
                             overwrite_b=True,
                             check_finite=False,
                             output=out_str)

    # 如果存在矩阵 e，则将 u 的部分进行 QR 分解
    if e is not None:
        u, _ = qr(np.vstack((e.dot(u[:m, :m]), u[m:, :m])))
    
    # 提取稳定子空间基础的相关部分
    u00 = u[:m, :m]   # 取 u 的左上角 m×m 部分
    u10 = u[m:, :m]   # 取 u 的第 m 行到最后一行，左上角 m×m 部分

    # 检查 u00 的条件并通过 LU 分解解决方程
    up, ul, uu = lu(u00)
    
    # 检查 uu 的条件数倒数是否小于机器精度，若是则抛出异常
    if 1/cond(uu) < np.spacing(1.):
        raise LinAlgError('Failed to find a finite solution.')

    # 利用三角结构求解方程
    x = solve_triangular(ul.conj().T,
                         solve_triangular(uu.conj().T,
                                          u10.conj().T,
                                          lower=True),
                         unit_diagonal=True,
                         ).conj().T.dot(up.conj().T)
    
    # 如果需要平衡，则乘以平衡矩阵 sca 的左上角部分
    if balanced:
        x *= sca[:m, None] * sca[:m]

    # 检查是否符合对称性的阈值条件
    # 参考 [2] 中定理 5 第 3 条的证明
    u_sym = u00.conj().T.dot(u10)   # 计算 u00 的共轭转置与 u10 的乘积
    n_u_sym = norm(u_sym, 1)        # 计算 u_sym 的 L1 范数
    u_sym = u_sym - u_sym.conj().T   # 计算 u_sym 与其共轭转置的差
    sym_threshold = np.max([np.spacing(1000.), 0.1*n_u_sym])   # 计算对称性阈值

    # 如果 u_sym 的 L1 范数超过阈值，则抛出异常
    if norm(u_sym, 1) > sym_threshold:
        raise LinAlgError('The associated symplectic pencil has eigenvalues '
                          'too close to the unit circle')

    # 返回对称的 x
    return (x + x.conj().T)/2
def _are_validate_args(a, b, q, r, e, s, eq_type='care'):
    """
    A helper function to validate the arguments supplied to the
    Riccati equation solvers. Any discrepancy found in the input
    matrices leads to a ``ValueError`` exception.

    Essentially, it performs:

        - a check whether the input is free of NaN and Infs
        - a pass for the data through ``numpy.atleast_2d()``
        - squareness check of the relevant arrays
        - shape consistency check of the arrays
        - singularity check of the relevant arrays
        - symmetricity check of the relevant matrices
        - a check whether the regular or the generalized version is asked.

    This function is used by ``solve_continuous_are`` and
    ``solve_discrete_are``.

    Parameters
    ----------
    a, b, q, r, e, s : array_like
        Input data
    eq_type : str
        Accepted arguments are 'care' and 'dare'.

    Returns
    -------
    a, b, q, r, e, s : ndarray
        Regularized input data
    m, n : int
        shape of the problem
    r_or_c : type
        Data type of the problem, returns float or complex
    gen_or_not : bool
        Type of the equation, True for generalized and False for regular ARE.

    """

    # Check if eq_type is valid
    if eq_type.lower() not in ("dare", "care"):
        raise ValueError("Equation type unknown. "
                         "Only 'care' and 'dare' is understood")

    # Regularize input arrays to at least 2 dimensions
    a = np.atleast_2d(_asarray_validated(a, check_finite=True))
    b = np.atleast_2d(_asarray_validated(b, check_finite=True))
    q = np.atleast_2d(_asarray_validated(q, check_finite=True))
    r = np.atleast_2d(_asarray_validated(r, check_finite=True))

    # Determine whether r_or_c should be float or complex based on input b
    r_or_c = complex if np.iscomplexobj(b) else float

    # Ensure matrices a, q, r are square
    for ind, mat in enumerate((a, q, r)):
        if np.iscomplexobj(mat):
            r_or_c = complex
        if not np.equal(*mat.shape):
            raise ValueError("Matrix {} should be square.".format("aqr"[ind]))

    # Check shape consistency between matrices a, b, q, r
    m, n = b.shape
    if m != a.shape[0]:
        raise ValueError("Matrix a and b should have the same number of rows.")
    if m != q.shape[0]:
        raise ValueError("Matrix a and q should have the same shape.")
    if n != r.shape[0]:
        raise ValueError("Matrix b and r should have the same number of cols.")

    # Check if matrices q, r are symmetric/hermitian
    for ind, mat in enumerate((q, r)):
        if norm(mat - mat.conj().T, 1) > np.spacing(norm(mat, 1)) * 100:
            raise ValueError("Matrix {} should be symmetric/hermitian."
                             "".format("qr"[ind]))

    # Check for singularity of matrix r in continuous-time ARE (care)
    if eq_type == 'care':
        min_sv = svd(r, compute_uv=False)[-1]
        if min_sv == 0. or min_sv < np.spacing(1.) * norm(r, 1):
            raise ValueError('Matrix r is numerically singular.')
    # 检查是否需要处理一般情况，即是否有省略参数
    # 执行延迟形状检查等操作。
    generalized_case = e is not None or s is not None

    if generalized_case:
        if e is not None:
            # 将 e 转换为至少二维数组，并确保其元素是有限的
            e = np.atleast_2d(_asarray_validated(e, check_finite=True))
            # 检查 e 是否是方阵
            if not np.equal(*e.shape):
                raise ValueError("Matrix e should be square.")
            # 检查 a 和 e 是否具有相同的形状
            if m != e.shape[0]:
                raise ValueError("Matrix a and e should have the same shape.")
            # 使用奇异值分解检查 e 的数值条件
            # numpy.linalg.cond 不会检查精确的零值，并发出运行时警告。因此以下是手动检查。
            min_sv = svd(e, compute_uv=False)[-1]
            if min_sv == 0. or min_sv < np.spacing(1.) * norm(e, 1):
                raise ValueError('Matrix e is numerically singular.')
            # 如果 e 是复数对象，则 r_or_c 被设为 complex
            if np.iscomplexobj(e):
                r_or_c = complex
        if s is not None:
            # 将 s 转换为至少二维数组，并确保其元素是有限的
            s = np.atleast_2d(_asarray_validated(s, check_finite=True))
            # 检查 b 和 s 是否具有相同的形状
            if s.shape != b.shape:
                raise ValueError("Matrix b and s should have the same shape.")
            # 如果 s 是复数对象，则 r_or_c 被设为 complex
            if np.iscomplexobj(s):
                r_or_c = complex

    # 返回所有参数和标志，包括 a, b, q, r, e, s, m, n, r_or_c, generalized_case
    return a, b, q, r, e, s, m, n, r_or_c, generalized_case
```