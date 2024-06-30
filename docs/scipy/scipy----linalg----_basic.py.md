# `D:\src\scipysrc\scipy\scipy\linalg\_basic.py`

```
# 导入警告模块中的warn函数
from warnings import warn
# 导入product函数，用于生成笛卡尔积的迭代器
from itertools import product
# 导入numpy库，并将其重命名为np
import numpy as np
# 从numpy中导入atleast_1d和atleast_2d函数
from numpy import atleast_1d, atleast_2d
# 从当前包中导入lapack模块中的get_lapack_funcs和_compute_lwork函数
from .lapack import get_lapack_funcs, _compute_lwork
# 从当前包中导入_misc模块中的LinAlgError、_datacopied和LinAlgWarning
from ._misc import LinAlgError, _datacopied, LinAlgWarning
# 从当前包中导入_decomp模块中的_asarray_validated
from ._decomp import _asarray_validated
# 从当前包中导入_decomp_svd模块
from . import _decomp, _decomp_svd
# 从当前包中导入_solve_toeplitz模块中的levinson函数
from ._solve_toeplitz import levinson
# 从当前包中导入_cythonized_array_utils模块中的find_det_from_lu函数
from ._cythonized_array_utils import find_det_from_lu

# __all__列表定义，指定模块的公开接口
__all__ = ['solve', 'solve_triangular', 'solveh_banded', 'solve_banded',
           'solve_toeplitz', 'solve_circulant', 'inv', 'det', 'lstsq',
           'pinv', 'pinvh', 'matrix_balance', 'matmul_toeplitz']

# numpy中用于类型转换检查的设施对于小尺寸的数组来说速度太慢，并且消耗了时间预算。
# 这里设置一个预先计算的numpy.can_cast()表的字典容器。
# 可以用来快速确定一个dtype可以被转换为LAPACK兼容的类型，即'float32, float64, complex64, complex128'。
# 然后可以通过"casting_dict[arr.dtype.char]"进行检查。
lapack_cast_dict = {x: ''.join([y for y in 'fdFD' if np.can_cast(x, y)])
                    for x in np.typecodes['All']}


def _solve_check(n, info, lamch=None, rcond=None):
    """ 
    在解决阶段的不同步骤中检查参数的有效性。
    
    参数：
    - n: 解的大小
    - info: LAPACK返回的信息值
    - lamch: 函数，用于获取机器精度相关的值
    - rcond: 条件数的阈值
    
    如果info小于0，抛出值错误，指出LAPACK中的非法值。
    如果info大于0，抛出线性代数错误，指出矩阵是奇异的。
    如果lamch不为None，则获取机器精度相关值，并比较rcond与其大小，若rcond小于E，则发出警告。
    """
    if info < 0:
        raise ValueError(f'LAPACK reported an illegal value in {-info}-th argument.')
    elif 0 < info:
        raise LinAlgError('Matrix is singular.')

    if lamch is None:
        return
    E = lamch('E')
    if rcond < E:
        warn(f'Ill-conditioned matrix (rcond={rcond:.6g}): '
             'result may not be accurate.',
             LinAlgWarning, stacklevel=3)


def solve(a, b, lower=False, overwrite_a=False,
          overwrite_b=False, check_finite=True, assume_a='gen',
          transposed=False):
    """
    解决线性方程组a @ x == b，其中a为方阵。

    如果数据矩阵已知是特定类型，则通过提供相应的字符串给“assume_a”键选择专用解算器。
    可用的选项有

    ===================  ========
     通用矩阵             'gen'
     对称矩阵             'sym'
     Hermite矩阵         'her'
     正定矩阵             'pos'
    ===================  ========

    如果省略，“'gen'”是默认结构。

    数组的数据类型定义了调用哪个解算器，不管其值如何。
    换句话说，即使复数数组条目具有精确的零虚部，也将基于数组的数据类型调用复数解算器。

    参数
    ----------
    a : (N, N) array_like
        输入方阵数据
    b : (N, NRHS) array_like
        右手边的输入数据。

    """
    """
    Flags for 1-D or N-D right-hand side
    标志变量，用于表示右侧向量b是1维还是多维

    a1 = atleast_2d(_asarray_validated(a, check_finite=check_finite))
    将输入的矩阵a转换为至少是二维的数组，确保输入合法性和有限性检查

    b1 = atleast_1d(_asarray_validated(b, check_finite=check_finite))
    将输入的向量b转换为至少是一维的数组，确保输入合法性和有限性检查

    n = a1.shape[0]
    获取矩阵a1的行数，即矩阵的大小

    overwrite_a = overwrite_a or _datacopied(a1, a)
    如果overwrite_a为True或者a1是通过复制数据得到的，则overwrite_a保持True

    overwrite_b = overwrite_b or _datacopied(b1, b)
    如果overwrite_b为True或者b1是通过复制数据得到的，则overwrite_b保持True

    if a1.shape[0] != a1.shape[1]:
        如果矩阵a1的行数不等于列数，抛出值错误异常
        raise ValueError('Input a needs to be a square matrix.')

    if n != b1.shape[0]:
        如果矩阵a1的行数不等于向量b1的行数，抛出值错误异常
        raise ValueError('Input b has to have same number of rows as '
                         'input a')

    # accommodate empty arrays
    适应空数组的情况，暂无具体实现
    """
    # 如果 b1 的大小为 0，需要解决一个特殊情况
    if b1.size == 0:
        # 解决一个特殊情况，生成一个单位矩阵，再根据 b1 的数据类型确定其数据类型，并返回一个与 b1 形状相同的空数组
        dt = solve(np.eye(2, dtype=a1.dtype), np.ones(2, dtype=b1.dtype)).dtype
        return np.empty_like(b1, dtype=dt)

    # 将 1 维数组 b1 规范化为 2 维数组
    if b1.ndim == 1:
        # 如果数组 b1 是 1 维且 n 等于 1，将其转换为行向量；否则将其转换为列向量
        if n == 1:
            b1 = b1[None, :]
        else:
            b1 = b1[:, None]
        b_is_1D = True

    # 检查 assume_a 是否属于 ('gen', 'sym', 'her', 'pos') 中的一种，如果不是则引发 ValueError 异常
    if assume_a not in ('gen', 'sym', 'her', 'pos'):
        raise ValueError(f'{assume_a} is not a recognized matrix structure')

    # 对于实矩阵，将其描述为 "symmetric"，而不是 "hermitian"
    # （因为 LAPACK 无法处理实数域下的 Hermitian 矩阵）
    if assume_a == 'her' and not np.iscomplexobj(a1):
        assume_a = 'sym'

    # 获取正确的 lamch 函数
    # 根据 a1 的数据类型字符判断是单精度还是双精度，并获取对应的 LAPACK 函数 lamch
    if a1.dtype.char in 'fF':  # 单精度
        lamch = get_lapack_funcs('lamch', dtype='f')
    else:  # 双精度
        lamch = get_lapack_funcs('lamch', dtype='d')

    # 获取 lange 函数，用于计算矩阵的范数
    lange = get_lapack_funcs('lange', (a1,))

    # 根据 transposed 变量的值确定 trans 和 norm 的取值
    if transposed:
        trans = 1
        norm = 'I'
        # 如果 a1 是复数对象，当前不支持解决 a^T x = b 或 a^H x = b 的情况
        if np.iscomplexobj(a1):
            raise NotImplementedError('scipy.linalg.solve can currently '
                                      'not solve a^T x = b or a^H x = b '
                                      'for complex matrices.')
    else:
        trans = 0
        norm = '1'

    # 计算矩阵 a1 的范数 anorm
    anorm = lange(norm, a1)

    # 如果 assume_a 为 'gen'，执行广义求解 'gesv' 的情况
    if assume_a == 'gen':
        # 获取 LAPACK 函数 gecon, getrf, getrs
        gecon, getrf, getrs = get_lapack_funcs(('gecon', 'getrf', 'getrs'),
                                               (a1, b1))
        # 对 a1 进行 LU 分解
        lu, ipvt, info = getrf(a1, overwrite_a=overwrite_a)
        _solve_check(n, info)
        # 求解线性方程组 getrs(lu, ipvt, b1, trans=trans, overwrite_b=overwrite_b)
        x, info = getrs(lu, ipvt, b1,
                        trans=trans, overwrite_b=overwrite_b)
        _solve_check(n, info)
        # 计算条件数 rcond
        rcond, info = gecon(lu, anorm, norm=norm)
    
    # 如果 assume_a 为 'her'，执行 Hermitian 矩阵求解 'hesv' 的情况
    elif assume_a == 'her':
        # 获取 LAPACK 函数 hecon, hesv, hesv_lw
        hecon, hesv, hesv_lw = get_lapack_funcs(('hecon', 'hesv',
                                                 'hesv_lwork'), (a1, b1))
        # 计算需要的工作空间大小 lwork
        lwork = _compute_lwork(hesv_lw, n, lower)
        # 对 Hermitian 矩阵 a1 进行求解 hesv(a1, b1, lwork=lwork, lower=lower, overwrite_a=overwrite_a, overwrite_b=overwrite_b)
        lu, ipvt, x, info = hesv(a1, b1, lwork=lwork,
                                 lower=lower,
                                 overwrite_a=overwrite_a,
                                 overwrite_b=overwrite_b)
        _solve_check(n, info)
        # 计算条件数 rcond
        rcond, info = hecon(lu, ipvt, anorm)
    
    # 如果 assume_a 为 'sym'，执行对称矩阵求解 'sysv' 的情况
    # 如果 assume_a 参数为 'sym'，表示解向量系数矩阵是对称的
    elif assume_a == 'sym':
        # 获取 LAPACK 函数 sycon、sysv、sysv_lwork 对象
        sycon, sysv, sysv_lw = get_lapack_funcs(('sycon', 'sysv',
                                                 'sysv_lwork'), (a1, b1))
        # 计算所需的工作空间大小
        lwork = _compute_lwork(sysv_lw, n, lower)
        # 调用 sysv 函数求解对称矩阵方程组
        lu, ipvt, x, info = sysv(a1, b1, lwork=lwork,
                                 lower=lower,
                                 overwrite_a=overwrite_a,
                                 overwrite_b=overwrite_b)
        # 检查求解是否成功
        _solve_check(n, info)
        # 计算条件数 rcond
        rcond, info = sycon(lu, ipvt, anorm)
    # 正定矩阵情况 'posv'
    else:
        # 获取 LAPACK 函数 pocon、posv 对象
        pocon, posv = get_lapack_funcs(('pocon', 'posv'),
                                       (a1, b1))
        # 调用 posv 函数求解正定矩阵方程组
        lu, x, info = posv(a1, b1, lower=lower,
                           overwrite_a=overwrite_a,
                           overwrite_b=overwrite_b)
        # 检查求解是否成功
        _solve_check(n, info)
        # 计算条件数 rcond
        rcond, info = pocon(lu, anorm)

    # 最终的求解结果检查
    _solve_check(n, info, lamch, rcond)

    # 如果解向量 x 是一维数组，则展平为一维
    if b_is_1D:
        x = x.ravel()

    # 返回求解得到的解向量 x
    return x
# 解决方程 a x = b，假设 a 是一个三角矩阵
def solve_triangular(a, b, trans=0, lower=False, unit_diagonal=False,
                     overwrite_b=False, check_finite=True):
    """
    Solve the equation ``a x = b`` for `x`, assuming a is a triangular matrix.

    Parameters
    ----------
    a : (M, M) array_like
        A triangular matrix
    b : (M,) or (M, N) array_like
        Right-hand side matrix in ``a x = b``
    lower : bool, optional
        Use only data contained in the lower triangle of `a`.
        Default is to use upper triangle.
    trans : {0, 1, 2, 'N', 'T', 'C'}, optional
        Type of system to solve:

        ========  =========
        trans     system
        ========  =========
        0 or 'N'  a x  = b
        1 or 'T'  a^T x = b
        2 or 'C'  a^H x = b
        ========  =========
    unit_diagonal : bool, optional
        If True, diagonal elements of `a` are assumed to be 1 and
        will not be referenced.
    overwrite_b : bool, optional
        Allow overwriting data in `b` (may enhance performance)
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    x : (M,) or (M, N) ndarray
        Solution to the system ``a x = b``.  Shape of return matches `b`.

    Raises
    ------
    LinAlgError
        If `a` is singular

    Notes
    -----
    .. versionadded:: 0.9.0

    Examples
    --------
    Solve the lower triangular system a x = b, where::

             [3  0  0  0]       [4]
        a =  [2  1  0  0]   b = [2]
             [1  0  1  0]       [4]
             [1  1  1  1]       [2]

    >>> import numpy as np
    >>> from scipy.linalg import solve_triangular
    >>> a = np.array([[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])
    >>> b = np.array([4, 2, 4, 2])
    >>> x = solve_triangular(a, b, lower=True)
    >>> x
    array([ 1.33333333, -0.66666667,  2.66666667, -1.33333333])
    >>> a.dot(x)  # Check the result
    array([ 4.,  2.,  4.,  2.])

    """

    # 将输入参数转换为合适的数组形式，检查是否包含无限值或 NaN
    a1 = _asarray_validated(a, check_finite=check_finite)
    b1 = _asarray_validated(b, check_finite=check_finite)

    # 检查矩阵 a 是否为方阵
    if len(a1.shape) != 2 or a1.shape[0] != a1.shape[1]:
        raise ValueError('expected square matrix')

    # 检查矩阵 a 和向量 b 的形状是否兼容
    if a1.shape[0] != b1.shape[0]:
        raise ValueError(f'shapes of a {a1.shape} and b {b1.shape} are incompatible')

    # 如果 b 是空数组，返回一个与 b 形状相同的空数组
    if b1.size == 0:
        dt_nonempty = solve_triangular(
            np.eye(2, dtype=a1.dtype), np.ones(2, dtype=b1.dtype)
        ).dtype
        return np.empty_like(b1, dtype=dt_nonempty)

    # 根据 trans 参数确定解的类型
    trans = {'N': 0, 'T': 1, 'C': 2}.get(trans, trans)
    # 获取 LAPACK 函数中的 trtrs 函数，用于解三角线性系统
    trtrs, = get_lapack_funcs(('trtrs',), (a1, b1))
    # 检查数组 a1 是否是列优先存储 (Fortran 风格) 或者 trans 变量是否为 2
    if a1.flags.f_contiguous or trans == 2:
        # 调用 trtrs 函数解决线性方程组，使用 a1 和 b1，可能覆盖 b1 的值
        x, info = trtrs(a1, b1, overwrite_b=overwrite_b, lower=lower,
                        trans=trans, unitdiag=unit_diagonal)
    else:
        # 因为 trtrs 函数期望 Fortran 排序，所以解决转置系统
        x, info = trtrs(a1.T, b1, overwrite_b=overwrite_b, lower=not lower,
                        trans=not trans, unitdiag=unit_diagonal)

    # 如果求解成功，返回解 x
    if info == 0:
        return x
    # 如果 info 大于 0，说明矩阵奇异，无法解决，抛出异常
    if info > 0:
        raise LinAlgError("singular matrix: resolution failed at diagonal %d" %
                          (info-1))
    # 如果 info 小于 0，说明 trtrs 函数参数出现问题，抛出值错误异常
    raise ValueError('illegal value in %dth argument of internal trtrs' %
                     (-info))
# 解决带状矩阵方程 a x = b，其中 a 是带状矩阵。

def solve_banded(l_and_u, ab, b, overwrite_ab=False, overwrite_b=False,
                 check_finite=True):
    """
    Solve the equation a x = b for x, assuming a is banded matrix.

    The matrix a is stored in `ab` using the matrix diagonal ordered form::

        ab[u + i - j, j] == a[i,j]

    Example of `ab` (shape of a is (6,6), `u` =1, `l` =2)::

        *    a01  a12  a23  a34  a45
        a00  a11  a22  a33  a44  a55
        a10  a21  a32  a43  a54   *
        a20  a31  a42  a53   *    *

    Parameters
    ----------
    (l, u) : (integer, integer)
        Number of non-zero lower and upper diagonals
    ab : (`l` + `u` + 1, M) array_like
        Banded matrix
    b : (M,) or (M, K) array_like
        Right-hand side
    overwrite_ab : bool, optional
        Discard data in `ab` (may enhance performance)
    overwrite_b : bool, optional
        Discard data in `b` (may enhance performance)
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    x : (M,) or (M, K) ndarray
        The solution to the system a x = b. Returned shape depends on the
        shape of `b`.

    Examples
    --------
    Solve the banded system a x = b, where::

            [5  2 -1  0  0]       [0]
            [1  4  2 -1  0]       [1]
        a = [0  1  3  2 -1]   b = [2]
            [0  0  1  2  2]       [2]
            [0  0  0  1  1]       [3]

    There is one nonzero diagonal below the main diagonal (l = 1), and
    two above (u = 2). The diagonal banded form of the matrix is::

             [*  * -1 -1 -1]
        ab = [*  2  2  2  2]
             [5  4  3  2  1]
             [1  1  1  1  *]

    >>> import numpy as np
    >>> from scipy.linalg import solve_banded
    >>> ab = np.array([[0,  0, -1, -1, -1],
    ...                [0,  2,  2,  2,  2],
    ...                [5,  4,  3,  2,  1],
    ...                [1,  1,  1,  1,  0]])
    >>> b = np.array([0, 1, 2, 2, 3])
    >>> x = solve_banded((1, 2), ab, b)
    >>> x
    array([-2.37288136,  3.93220339, -4.        ,  4.3559322 , -1.3559322 ])

    """

    # 将输入的带状矩阵 ab 和右侧向量 b 转换为浮点数数组
    a1 = _asarray_validated(ab, check_finite=check_finite, as_inexact=True)
    b1 = _asarray_validated(b, check_finite=check_finite, as_inexact=True)

    # 检查矩阵和向量的形状是否兼容
    if a1.shape[-1] != b1.shape[0]:
        raise ValueError("shapes of ab and b are not compatible.")

    # 检查输入的带状矩阵的下限和上限对角线数是否与 ab 的行数相符
    (nlower, nupper) = l_and_u
    if nlower + nupper + 1 != a1.shape[0]:
        raise ValueError("invalid values for the number of lower and upper "
                         "diagonals: l+u+1 (%d) does not equal ab.shape[0] "
                         "(%d)" % (nlower + nupper + 1, ab.shape[0]))

    # accommodate empty arrays
    # 如果 b1 的大小为 0，则根据 solve 函数解决一个单位矩阵和全为 1 的向量，获取数据类型，并返回一个与 b1 类型相同但未初始化的数组
    if b1.size == 0:
        dt = solve(np.eye(1, dtype=a1.dtype), np.ones(1, dtype=b1.dtype)).dtype
        return np.empty_like(b1, dtype=dt)
    
    # 如果 overwrite_b 为真或者 _datacopied(b1, b) 返回真，则将 overwrite_b 设置为真
    overwrite_b = overwrite_b or _datacopied(b1, b)
    
    # 如果 a1 的最后一个维度的长度为 1
    if a1.shape[-1] == 1:
        # 根据 overwrite_b 的值创建 b2 数组，若 overwrite_b 为假，则不复制 b1 的内容
        b2 = np.array(b1, copy=(not overwrite_b))
        # 将 b2 中的每个元素除以 a1 的第一行第一列的值
        b2 /= a1[1, 0]
        return b2
    
    # 如果 nlower 和 nupper 均为 1
    if nlower == nupper == 1:
        # 检查是否需要重写 overwrite_ab，并获取 Lapack 函数 gtsv
        overwrite_ab = overwrite_ab or _datacopied(a1, ab)
        gtsv, = get_lapack_funcs(('gtsv',), (a1, b1))
        # 提取 a1 矩阵的第一行后的所有元素作为 du
        du = a1[0, 1:]
        # 提取 a1 矩阵的第二行所有元素作为 d
        d = a1[1, :]
        # 提取 a1 矩阵的第三行除最后一个元素外的所有元素作为 dl
        dl = a1[2, :-1]
        # 调用 gtsv 函数解决三对角线方程组，并返回解 x、信息 info
        du2, d, du, x, info = gtsv(dl, d, du, b1, overwrite_ab, overwrite_ab,
                                   overwrite_ab, overwrite_b)
    else:
        # 获取 Lapack 函数 gbsv
        gbsv, = get_lapack_funcs(('gbsv',), (a1, b1))
        # 创建一个零填充的矩阵 a2，其形状为 (2*nlower + nupper + 1, a1.shape[1])
        a2 = np.zeros((2*nlower + nupper + 1, a1.shape[1]), dtype=gbsv.dtype)
        # 将 a1 的内容复制到 a2 的适当位置
        a2[nlower:, :] = a1
        # 调用 gbsv 函数解决一般带状线性方程组，并返回解 x、信息 info
        lu, piv, x, info = gbsv(nlower, nupper, a2, b1, overwrite_ab=True,
                                overwrite_b=overwrite_b)
    
    # 如果返回的信息 info 为 0，则返回解 x
    if info == 0:
        return x
    # 如果返回的信息 info 大于 0，则抛出线性代数错误，指示矩阵奇异
    if info > 0:
        raise LinAlgError("singular matrix")
    # 如果返回的信息 info 小于 0，则抛出数值错误，指示内部 gbsv/gtsv 函数的参数中有非法值
    raise ValueError('illegal value in %d-th argument of internal '
                     'gbsv/gtsv' % -info)
# 解决带状矩阵方程 a x = b，其中 a 是 Hermitian 正定的带状矩阵。
def solveh_banded(ab, b, overwrite_ab=False, overwrite_b=False, lower=False,
                  check_finite=True):
    """
    使用 Thomas 算法解方程 a x = b。该算法比标准的 LU 分解更高效，但只适用于 Hermitian 正定矩阵。

    矩阵 `a` 存储在 `ab` 中，可以是以下列顺序的上对角线或者下对角线形式：

        ab[u + i - j, j] == a[i,j]        (如果是上对角线形式; i <= j)
        ab[    i - j, j] == a[i,j]        (如果是下对角线形式; i >= j)

    示例 `ab`（矩阵 `a` 的形状为 (6, 6)，上对角线数量 `u` = 2）：

        上对角线形式：
        *   *   a02 a13 a24 a35
        *   a01 a12 a23 a34 a45
        a00 a11 a22 a33 a44 a55

        下对角线形式：
        a00 a11 a22 a33 a44 a55
        a10 a21 a32 a43 a54 *
        a20 a31 a42 a53 *   *

    标记为 * 的单元格未被使用。

    Parameters
    ----------
    ab : (``u`` + 1, M) array_like
        带状矩阵
    b : (M,) or (M, K) array_like
        右侧向量
    overwrite_ab : bool, optional
        是否丢弃 `ab` 中的数据（可能提升性能）
    overwrite_b : bool, optional
        是否丢弃 `b` 中的数据（可能提升性能）
    lower : bool, optional
        矩阵是否以下对角线形式存储（默认为上对角线形式）
    check_finite : bool, optional
        是否检查输入矩阵只包含有限数值。禁用此选项可能提升性能，但若输入包含无穷或 NaN 可能会导致问题（崩溃、无限循环）。

    Returns
    -------
    x : (M,) or (M, K) ndarray
        方程组 `a x = b` 的解。返回的形状与 `b` 的形状一致。

    Notes
    -----
    若矩阵 `a` 不是正定的，则应使用 `solve_banded` 解算器。

    Examples
    --------
    解带状系统 ``A x = b``，其中::

            [ 4  2 -1  0  0  0]       [1]
            [ 2  5  2 -1  0  0]       [2]
        A = [-1  2  6  2 -1  0]   b = [2]
            [ 0 -1  2  7  2 -1]       [3]
            [ 0  0 -1  2  8  2]       [3]
            [ 0  0  0 -1  2  9]       [3]

    >>> import numpy as np
    >>> from scipy.linalg import solveh_banded

    `ab` 包含主对角线及其下方的非零对角线元素。在此示例中使用下对角线形式：

    >>> ab = np.array([[ 4,  5,  6,  7, 8, 9],
    ...                [ 2,  2,  2,  2, 2, 0],
    ...                [-1, -1, -1, -1, 0, 0]])
    >>> b = np.array([1, 2, 2, 3, 3, 3])
    >>> x = solveh_banded(ab, b, lower=True)
    >>> x
    array([ 0.03431373,  0.45938375,  0.05602241,  0.47759104,  0.17577031,
            0.34733894])
    """
    """
    Solve the Hermitian banded system ``H x = b``, where::
    
            [ 8   2-1j   0     0  ]        [ 1  ]
        H = [2+1j  5     1j    0  ]    b = [1+1j]
            [ 0   -1j    9   -2-1j]        [1-2j]
            [ 0    0   -2+1j   6  ]        [ 0  ]
    
    In this example, we put the upper diagonals in the array ``hb``:
    
    >>> hb = np.array([[0, 2-1j, 1j, -2-1j],
    ...                [8,  5,    9,   6  ]])
    >>> b = np.array([1, 1+1j, 1-2j, 0])
    >>> x = solveh_banded(hb, b)
    >>> x
    array([ 0.07318536-0.02939412j,  0.11877624+0.17696461j,
            0.10077984-0.23035393j, -0.00479904-0.09358128j])
    
    """
    
    # 将输入参数转换为有效的数组形式，进行有限性检查
    a1 = _asarray_validated(ab, check_finite=check_finite)
    b1 = _asarray_validated(b, check_finite=check_finite)
    
    # 验证形状是否匹配
    if a1.shape[-1] != b1.shape[0]:
        raise ValueError("shapes of ab and b are not compatible.")
    
    # 处理空数组情况
    if b1.size == 0:
        # 当数组为空时，使用单位矩阵解决方程，返回与b1相同形状的空数组
        dt = solve(np.eye(1, dtype=a1.dtype), np.ones(1, dtype=b1.dtype)).dtype
        return np.empty_like(b1, dtype=dt)
    
    # 检查是否复制了输入数组b1和ab
    overwrite_b = overwrite_b or _datacopied(b1, b)
    overwrite_ab = overwrite_ab or _datacopied(a1, ab)
    
    # 根据a1的行数选择合适的LAPACK函数
    if a1.shape[0] == 2:
        # 对于二阶方阵a1，使用ptsv函数进行求解
        ptsv, = get_lapack_funcs(('ptsv',), (a1, b1))
        if lower:
            # 如果lower=True，取a1的第一行实部为d，取a1的第二行除最后一个元素外的共轭为e
            d = a1[0, :].real
            e = a1[1, :-1]
        else:
            # 如果lower=False，取a1的第二行实部为d，取a1的第一行从第二个元素开始的共轭为e
            d = a1[1, :].real
            e = a1[0, 1:].conj()
        # 调用ptsv函数求解带状线性方程组
        d, du, x, info = ptsv(d, e, b1, overwrite_ab, overwrite_ab,
                              overwrite_b)
    else:
        # 对于其他情况，使用pbsv函数进行求解
        pbsv, = get_lapack_funcs(('pbsv',), (a1, b1))
        # 调用pbsv函数求解带状线性方程组
        c, x, info = pbsv(a1, b1, lower=lower, overwrite_ab=overwrite_ab,
                          overwrite_b=overwrite_b)
    
    # 检查求解过程中的信息代码，抛出相应的异常
    if info > 0:
        raise LinAlgError("%dth leading minor not positive definite" % info)
    if info < 0:
        raise ValueError('illegal value in %dth argument of internal '
                         'pbsv' % -info)
    
    # 返回解向量x
    return x
# 解决 Toeplitz 系统的函数，使用 Levinson 递归算法
def solve_toeplitz(c_or_cr, b, check_finite=True):
    """Solve a Toeplitz system using Levinson Recursion

    The Toeplitz matrix has constant diagonals, with c as its first column
    and r as its first row. If r is not given, ``r == conjugate(c)`` is
    assumed.

    Parameters
    ----------
    c_or_cr : array_like or tuple of (array_like, array_like)
        The vector ``c``, or a tuple of arrays (``c``, ``r``). Whatever the
        actual shape of ``c``, it will be converted to a 1-D array. If not
        supplied, ``r = conjugate(c)`` is assumed; in this case, if c[0] is
        real, the Toeplitz matrix is Hermitian. r[0] is ignored; the first row
        of the Toeplitz matrix is ``[c[0], r[1:]]``. Whatever the actual shape
        of ``r``, it will be converted to a 1-D array.
    b : (M,) or (M, K) array_like
        Right-hand side in ``T x = b``.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (result entirely NaNs) if the inputs do contain infinities or NaNs.

    Returns
    -------
    x : (M,) or (M, K) ndarray
        The solution to the system ``T x = b``. Shape of return matches shape
        of `b`.

    See Also
    --------
    toeplitz : Toeplitz matrix

    Notes
    -----
    The solution is computed using Levinson-Durbin recursion, which is faster
    than generic least-squares methods, but can be less numerically stable.

    Examples
    --------
    Solve the Toeplitz system T x = b, where::

            [ 1 -1 -2 -3]       [1]
        T = [ 3  1 -1 -2]   b = [2]
            [ 6  3  1 -1]       [2]
            [10  6  3  1]       [5]

    To specify the Toeplitz matrix, only the first column and the first
    row are needed.

    >>> import numpy as np
    >>> c = np.array([1, 3, 6, 10])    # First column of T
    >>> r = np.array([1, -1, -2, -3])  # First row of T
    >>> b = np.array([1, 2, 2, 5])

    >>> from scipy.linalg import solve_toeplitz, toeplitz
    >>> x = solve_toeplitz((c, r), b)
    >>> x
    array([ 1.66666667, -1.        , -2.66666667,  2.33333333])

    Check the result by creating the full Toeplitz matrix and
    multiplying it by `x`.  We should get `b`.

    >>> T = toeplitz(c, r)
    >>> T.dot(x)
    array([ 1.,  2.,  2.,  5.])

    """
    # 如果这个算法的数值稳定性有问题，未来的开发者可能会考虑实现其他 O(N^2) 的 Toeplitz 求解器，
    # 如 GKO (https://www.jstor.org/stable/2153371) 或 Bareiss.

    # 验证参数，获取 c, r, b 的正确格式，同时保留 b 的形状
    r, c, b, dtype, b_shape = _validate_args_for_toeplitz_ops(
        c_or_cr, b, check_finite, keep_b_shape=True)

    # 处理空数组情况
    if b.size == 0:
        return np.empty_like(b)

    # 构造一个包含 r[1:] 的反向复制和 c 的一维值数组，作为矩阵中使用的值
    vals = np.concatenate((r[-1:0:-1], c))
    # 如果 b 参数为 None，则抛出值错误异常，指示 `b` 是必需的参数
    if b is None:
        raise ValueError('illegal value, `b` is a required argument')
    
    # 如果 b 是一维数组（向量）
    if b.ndim == 1:
        # 调用 levinson 函数，计算线性预测系数，返回结果存入 x，忽略第二个返回值
        x, _ = levinson(vals, np.ascontiguousarray(b))
    else:
        # 对于 b 是二维数组（矩阵）的情况
        # 使用列表推导式遍历 b 的列索引范围，对每列调用 levinson 函数，获取第一个返回值（线性预测系数），并堆叠为列向量
        x = np.column_stack([levinson(vals, np.ascontiguousarray(b[:, i]))[0]
                             for i in range(b.shape[1])])
        # 将 x 重塑为与 b 形状相同的数组
        x = x.reshape(*b_shape)
    
    # 返回计算得到的 x
    return x
# 定义函数，用于获取指定轴向的数组长度
def _get_axis_len(aname, a, axis):
    # 将轴号存储在局部变量 ax 中
    ax = axis
    # 如果轴号为负数，将其转换为对应的非负轴号
    if ax < 0:
        ax += a.ndim
    # 如果转换后的轴号在有效范围内，则返回该轴的长度
    if 0 <= ax < a.ndim:
        return a.shape[ax]
    # 否则，抛出值错误，提示轴号越界
    raise ValueError(f"'{aname}axis' entry is out of bounds")


# 定义函数，解决形如 C x = b 的线性方程组，其中 C 是循环矩阵
def solve_circulant(c, b, singular='raise', tol=None,
                    caxis=-1, baxis=0, outaxis=0):
    """Solve C x = b for x, where C is a circulant matrix.

    `C` is the circulant matrix associated with the vector `c`.

    The system is solved by doing division in Fourier space. The
    calculation is::

        x = ifft(fft(b) / fft(c))

    where `fft` and `ifft` are the fast Fourier transform and its inverse,
    respectively. For a large vector `c`, this is *much* faster than
    solving the system with the full circulant matrix.

    Parameters
    ----------
    c : array_like
        The coefficients of the circulant matrix.
    b : array_like
        Right-hand side matrix in ``a x = b``.
    singular : str, optional
        This argument controls how a near singular circulant matrix is
        handled.  If `singular` is "raise" and the circulant matrix is
        near singular, a `LinAlgError` is raised. If `singular` is
        "lstsq", the least squares solution is returned. Default is "raise".
    tol : float, optional
        If any eigenvalue of the circulant matrix has an absolute value
        that is less than or equal to `tol`, the matrix is considered to be
        near singular. If not given, `tol` is set to::

            tol = abs_eigs.max() * abs_eigs.size * np.finfo(np.float64).eps

        where `abs_eigs` is the array of absolute values of the eigenvalues
        of the circulant matrix.
    caxis : int
        When `c` has dimension greater than 1, it is viewed as a collection
        of circulant vectors. In this case, `caxis` is the axis of `c` that
        holds the vectors of circulant coefficients.
    baxis : int
        When `b` has dimension greater than 1, it is viewed as a collection
        of vectors. In this case, `baxis` is the axis of `b` that holds the
        right-hand side vectors.
    outaxis : int
        When `c` or `b` are multidimensional, the value returned by
        `solve_circulant` is multidimensional. In this case, `outaxis` is
        the axis of the result that holds the solution vectors.

    Returns
    -------
    x : ndarray
        Solution to the system ``C x = b``.

    Raises
    ------
    LinAlgError
        If the circulant matrix associated with `c` is near singular.

    See Also
    --------
    circulant : circulant matrix

    Notes
    -----
    For a 1-D vector `c` with length `m`, and an array `b`
    with shape ``(m, ...)``,

        solve_circulant(c, b)

    returns the same result as

        solve(circulant(c), b)

    where `solve` and `circulant` are from `scipy.linalg`.

    .. versionadded:: 0.16.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import solve_circulant, solve, circulant, lstsq
    """
    # 函数文档字符串结束，后面是函数实现
    pass
    c = np.atleast_1d(c)
    # 将输入的参数 c 转换为至少是一维的 numpy 数组

    nc = _get_axis_len("c", c, caxis)
    # 调用 _get_axis_len 函数获取参数 c 的指定轴上的长度，并将结果保存在 nc 变量中

    b = np.atleast_1d(b)
    # 将输入的参数 b 转换为至少是一维的 numpy 数组

    nb = _get_axis_len("b", b, baxis)
    # 调用 _get_axis_len 函数获取参数 b 的指定轴上的长度，并将结果保存在 nb 变量中

    if nc != nb:
        # 如果参数 c 和 b 的长度不相等，抛出 ValueError 异常
        raise ValueError(f'Shapes of c {c.shape} and b {b.shape} are incompatible')

    # accommodate empty arrays
    # 处理空数组的情况，此处可以用于处理长度为零的 c 和 b 数组的情况
    # 如果向量 b 的大小为 0，则需要解决一个特殊情况：
    if b.size == 0:
        # 使用 solve_circulant 函数求解一个循环矩阵的特征值，
        # 其中使用 np.arange(3, dtype=c.dtype) 和 np.ones(3, dtype=b.dtype) 来构造初始条件，
        # 然后获取其数据类型并赋给 dt。
        dt = solve_circulant(np.arange(3, dtype=c.dtype),
                             np.ones(3, dtype=b.dtype)).dtype
        # 返回一个与 b 相同形状的空数组，数据类型为 dt。
        return np.empty_like(b, dtype=dt)

    # 对输入矩阵 c 进行 FFT 变换，将轴 caxis 移动到最后一维，并计算其频谱。
    fc = np.fft.fft(np.moveaxis(c, caxis, -1), axis=-1)
    # 计算频谱的绝对值。
    abs_fc = np.abs(fc)

    # 如果没有指定 tol（容差），则使用与 np.linalg.matrix_rank 中使用的相同容差。
    if tol is None:
        # 计算容差 tol，这里使用的是最大频谱值乘以 nc 和 np.finfo(np.float64).eps 的乘积。
        tol = abs_fc.max(axis=-1) * nc * np.finfo(np.float64).eps
        # 如果 tol 的形状不是标量（即它不是单个值），则将其形状调整为 (n,) 形式。
        if tol.shape != ():
            tol.shape = tol.shape + (1,)
        else:
            # 否则，将 tol 至少调整为一个 1 维数组。
            tol = np.atleast_1d(tol)

    # 创建一个布尔掩码数组，指示 abs_fc 中哪些元素的值小于或等于对应位置的 tol。
    near_zeros = abs_fc <= tol
    # 检查是否存在接近奇异的情况，即 near_zeros 是否有任何 True 值。
    is_near_singular = np.any(near_zeros)

    # 如果发现接近奇异的情况：
    if is_near_singular:
        # 如果 singular 参数设置为 'raise'，则抛出线性代数错误。
        if singular == 'raise':
            raise LinAlgError("near singular circulant matrix.")
        else:
            # 否则，将 fc 中接近零的值替换为 1，以避免在下面的除法 fb/fc 中出现错误。
            fc[near_zeros] = 1

    # 对输入向量 b 进行 FFT 变换，将轴 baxis 移动到最后一维，并计算其频谱。
    fb = np.fft.fft(np.moveaxis(b, baxis, -1), axis=-1)

    # 计算除法 q = fb / fc，这里是频谱之间的逐元素除法。
    q = fb / fc

    # 如果存在接近奇异的情况：
    if is_near_singular:
        # 创建一个布尔掩码数组 mask，其形状与 b 相同，指示 fc 接近零的位置。
        mask = np.ones_like(b, dtype=bool) & near_zeros
        # 将 q 中 mask 为 True 的位置的值设为 0，以避免在除法 fb/fc 中出现问题。
        q[mask] = 0

    # 对 q 进行逆 FFT 变换，沿着最后一维计算逆变换得到 x。
    x = np.fft.ifft(q, axis=-1)

    # 如果输入矩阵 c 和向量 b 都不是复数类型，则将 x 转换为实部。
    if not (np.iscomplexobj(c) or np.iscomplexobj(b)):
        x = x.real

    # 如果 outaxis 不等于 -1，则将 x 中的最后一维移动到指定的 outaxis 位置。
    if outaxis != -1:
        x = np.moveaxis(x, -1, outaxis)

    # 返回计算结果 x。
    return x
# 矩阵求逆运算
def inv(a, overwrite_a=False, check_finite=True):
    """
    计算矩阵的逆。

    Parameters
    ----------
    a : array_like
        待求逆的方阵。
    overwrite_a : bool, optional
        是否丢弃 `a` 中的数据（可能提高性能）。默认为 False。
    check_finite : bool, optional
        是否检查输入矩阵是否仅包含有限数值。
        禁用此项可能提高性能，但如果输入包含无穷大或 NaN，则可能导致问题
        （崩溃、非终止）。

    Returns
    -------
    ainv : ndarray
        矩阵 `a` 的逆。

    Raises
    ------
    LinAlgError
        如果 `a` 是奇异的。
    ValueError
        如果 `a` 不是方阵，或不是二维的。

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import linalg
    >>> a = np.array([[1., 2.], [3., 4.]])
    >>> linalg.inv(a)
    array([[-2. ,  1. ],
           [ 1.5, -0.5]])
    >>> np.dot(a, linalg.inv(a))
    array([[ 1.,  0.],
           [ 0.,  1.]])

    """
    a1 = _asarray_validated(a, check_finite=check_finite)
    if len(a1.shape) != 2 or a1.shape[0] != a1.shape[1]:
        raise ValueError('expected square matrix')

    # 处理空的方阵情况
    if a1.size == 0:
        dt = inv(np.eye(2, dtype=a1.dtype)).dtype
        return np.empty_like(a1, dtype=dt)

    overwrite_a = overwrite_a or _datacopied(a1, a)
    getrf, getri, getri_lwork = get_lapack_funcs(('getrf', 'getri',
                                                  'getri_lwork'),
                                                 (a1,))
    lu, piv, info = getrf(a1, overwrite_a=overwrite_a)
    if info == 0:
        lwork = _compute_lwork(getri_lwork, a1.shape[0])

        # XXX: 以下一行修复了在对 500x500 矩阵求逆时奇怪的段错误（SEGFAULT）问题。
        # 这似乎是 LAPACK 中 getri 程序的一个错误，因为如果 lwork 很小
        # （使用 lwork[0] 而不是 lwork[1]），那么所有测试都通过。
        # 如果出现更多这样的段错误，需要进一步调查。
        lwork = int(1.01 * lwork)
        inv_a, info = getri(lu, piv, lwork=lwork, overwrite_lu=1)
    if info > 0:
        raise LinAlgError("singular matrix")
    if info < 0:
        raise ValueError('illegal value in %d-th argument of internal '
                         'getrf|getri' % -info)
    return inv_a


# 行列式
def det(a, overwrite_a=False, check_finite=True):
    """
    计算矩阵的行列式。

    行列式是一个标量，是与相关方阵系数相关的函数。奇异矩阵的行列式值为零。

    Parameters
    ----------
    a : (..., M, M) array_like
        输入数组，用于计算行列式。
    overwrite_a : bool, optional
        允许覆盖 `a` 中的数据（可能提升性能）。
        
    """
    # 我们的目标是获得一个可写的连续数组，以便传递给Cython

    # 首先，我们检查并创建数组。
    a1 = np.asarray_chkfinite(a) if check_finite else np.asarray(a)
    # 如果数组不是至少二维的，抛出数值错误
    if a1.ndim < 2:
        raise ValueError('The input array must be at least two-dimensional.')
    # 如果数组的最后两个维度不是方阵，抛出数值错误，并显示实际形状
    if a1.shape[-1] != a1.shape[-2]:
        raise ValueError('Last 2 dimensions of the array must be square'
                         f' but received shape {a1.shape}.')

    # 同时检查数组的数据类型是否兼容 LAPACK
    if a1.dtype.char not in 'fdFD':
        # 如果数据类型不能直接转换，查找是否有可以转换的类型
        dtype_char = lapack_cast_dict[a1.dtype.char]
        # 如果没有可转换的类型，抛出类型错误
        if not dtype_char:  # No casting possible
            raise TypeError(f'The dtype "{a1.dtype.name}" cannot be cast '
                            'to float(32, 64) or complex(64, 128).')

        # 将数组转换为指定的数据类型，这会创建一个副本，可以在其中进行修改
        a1 = a1.astype(dtype_char[0])
        overwrite_a = True

    # 如果数组的最小维度为0，则返回一个具有单位值的数组
    if min(*a1.shape) == 0:
        # 根据数据类型字符决定返回的数据类型
        dtyp = np.float64 if a1.dtype.char not in 'FD' else np.complex128
        # 如果数组是二维的，直接返回一个指定数据类型的单位值
        if a1.ndim == 2:
            return dtyp(1.0)
        else:
            # 如果数组不是二维，返回一个指定形状和数据类型的全1数组
            return np.ones(shape=a1.shape[:-2], dtype=dtyp)
    # Scalar case
    if a1.shape[-2:] == (1, 1):
        # 检查是否为标量情况，即最后两个维度为 (1, 1)
        # 可能是带有多余单例维度的 ndarray 或者单个元素
        if max(*a1.shape) > 1:
            # 去除多余的单例维度
            temp = np.squeeze(a1)
            # 如果数据类型为双精度或复双精度，直接返回
            if a1.dtype.char in 'dD':
                return temp
            else:
                # 否则将数据类型转换为双精度或复双精度后返回
                return (temp.astype('d') if a1.dtype.char == 'f' else
                        temp.astype('D'))
        else:
            # 如果是单个元素，根据数据类型返回对应的数值类型
            return (np.float64(a1.item()) if a1.dtype.char in 'fd' else
                    np.complex128(a1.item()))

    # Then check overwrite permission
    if not _datacopied(a1, a):  # "a"  still alive through "a1"
        # 检查是否需要复制数据到新的数组 a1
        if not overwrite_a:
            # 如果不允许覆盖，则复制数组 a1
            a1 = a1.copy(order='C')
        #  else: Do nothing we'll use "a" if possible
        # 否则，如果允许覆盖，则保持原样，可能会使用数组 a

    # Then layout checks, might happen that overwrite is allowed but original
    # array was read-only or non-C-contiguous.
    # 检查数组布局，可能出现的情况是允许覆盖，但原始数组只读或非 C 连续。
    if not (a1.flags['C_CONTIGUOUS'] and a1.flags['WRITEABLE']):
        # 如果数组不是 C 连续或不可写，则进行复制
        a1 = a1.copy(order='C')

    if a1.ndim == 2:
        # 如果数组是二维的，计算其行列式
        det = find_det_from_lu(a1)
        # 将浮点数或复数转换为 NumPy 标量并返回
        return (np.float64(det) if np.isrealobj(det) else np.complex128(det))

    # loop over the stacked array, and avoid overflows for single precision
    # Cf. np.linalg.det(np.diag([1e+38, 1e+38]).astype(np.float32))
    # 遍历堆叠的数组，并避免单精度溢出
    dtype_char = a1.dtype.char
    # 如果数据类型是单精度或复单精度，转换为双精度或复双精度
    if dtype_char in 'fF':
        dtype_char = 'd' if dtype_char.islower() else 'D'

    # 创建一个与 a1 形状[:-2] 相同的空数组
    det = np.empty(a1.shape[:-2], dtype=dtype_char)
    # 遍历数组的所有索引，并计算每个索引位置的行列式
    for ind in product(*[range(x) for x in a1.shape[:-2]]):
        det[ind] = find_det_from_lu(a1[ind])
    # 返回计算得到的行列式结果
    return det
# Linear Least Squares
def lstsq(a, b, cond=None, overwrite_a=False, overwrite_b=False,
          check_finite=True, lapack_driver=None):
    """
    Compute least-squares solution to equation Ax = b.

    Compute a vector x such that the 2-norm ``|b - A x|`` is minimized.

    Parameters
    ----------
    a : (M, N) array_like
        Left-hand side array
    b : (M,) or (M, K) array_like
        Right hand side array
    cond : float, optional
        Cutoff for 'small' singular values; used to determine effective
        rank of a. Singular values smaller than
        ``cond * largest_singular_value`` are considered zero.
    overwrite_a : bool, optional
        Discard data in `a` (may enhance performance). Default is False.
    overwrite_b : bool, optional
        Discard data in `b` (may enhance performance). Default is False.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    lapack_driver : str, optional
        Which LAPACK driver is used to solve the least-squares problem.
        Options are ``'gelsd'``, ``'gelsy'``, ``'gelss'``. Default
        (``'gelsd'``) is a good choice.  However, ``'gelsy'`` can be slightly
        faster on many problems.  ``'gelss'`` was used historically.  It is
        generally slow but uses less memory.

        .. versionadded:: 0.17.0

    Returns
    -------
    x : (N,) or (N, K) ndarray
        Least-squares solution.
    residues : (K,) ndarray or float
        Square of the 2-norm for each column in ``b - a x``, if ``M > N`` and
        ``rank(A) == n`` (returns a scalar if ``b`` is 1-D). Otherwise a
        (0,)-shaped array is returned.
    rank : int
        Effective rank of `a`.
    s : (min(M, N),) ndarray or None
        Singular values of `a`. The condition number of ``a`` is
        ``s[0] / s[-1]``.

    Raises
    ------
    LinAlgError
        If computation does not converge.

    ValueError
        When parameters are not compatible.

    See Also
    --------
    scipy.optimize.nnls : linear least squares with non-negativity constraint

    Notes
    -----
    When ``'gelsy'`` is used as a driver, `residues` is set to a (0,)-shaped
    array and `s` is always ``None``.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import lstsq
    >>> import matplotlib.pyplot as plt

    Suppose we have the following data:

    >>> x = np.array([1, 2.5, 3.5, 4, 5, 7, 8.5])
    >>> y = np.array([0.3, 1.1, 1.5, 2.0, 3.2, 6.6, 8.6])

    We want to fit a quadratic polynomial of the form ``y = a + b*x**2``
    to this data.  We first form the "design matrix" M, with a constant
    column of 1s and a column containing ``x**2``:

    >>> M = x[:, np.newaxis]**[0, 2]
    >>> M
    array([[  1.  ,   1.  ],  # 创建一个二维数组，表示线性方程组的系数矩阵 M
           [  1.  ,   6.25],
           [  1.  ,  12.25],
           [  1.  ,  16.  ],
           [  1.  ,  25.  ],
           [  1.  ,  49.  ],
           [  1.  ,  72.25]])

    We want to find the least-squares solution to ``M.dot(p) = y``,
    where ``p`` is a vector with length 2 that holds the parameters
    ``a`` and ``b``.

    >>> p, res, rnk, s = lstsq(M, y)
    >>> p
    array([ 0.20925829,  0.12013861])  # 解出的最小二乘解，即参数向量 p

    Plot the data and the fitted curve.

    >>> plt.plot(x, y, 'o', label='data')  # 绘制原始数据的散点图
    >>> xx = np.linspace(0, 9, 101)  # 在区间 [0, 9] 上生成 101 个等间距的点作为 x 值
    >>> yy = p[0] + p[1]*xx**2  # 使用最小二乘解 p 拟合出的二次曲线的 y 值
    >>> plt.plot(xx, yy, label='least squares fit, $y = a + bx^2$')  # 绘制拟合的二次曲线
    >>> plt.xlabel('x')  # x 轴标签
    >>> plt.ylabel('y')  # y 轴标签
    >>> plt.legend(framealpha=1, shadow=True)  # 添加图例，设置透明度和阴影
    >>> plt.grid(alpha=0.25)  # 显示网格，设置透明度
    >>> plt.show()  # 显示图形

    """
    a1 = _asarray_validated(a, check_finite=check_finite)  # 对数组 a 进行有效性验证，确保可以处理的数组格式
    b1 = _asarray_validated(b, check_finite=check_finite)  # 对数组 b 进行有效性验证，确保可以处理的数组格式
    if len(a1.shape) != 2:  # 检查数组 a1 是否为二维数组
        raise ValueError('Input array a should be 2D')  # 如果不是二维数组，则抛出 ValueError 异常
    m, n = a1.shape  # 获取数组 a1 的形状 (m, n)
    if len(b1.shape) == 2:  # 检查数组 b1 是否为二维数组
        nrhs = b1.shape[1]  # 如果是二维数组，获取其第二维的大小，表示右端向量的数量
    else:
        nrhs = 1  # 否则，右端向量的数量为 1
    if m != b1.shape[0]:  # 检查数组 a1 和 b1 的行数是否相同
        raise ValueError('Shape mismatch: a and b should have the same number'
                         f' of rows ({m} != {b1.shape[0]}).')  # 如果行数不相同，则抛出 ValueError 异常
    if m == 0 or n == 0:  # 如果问题的规模为零，会困扰 LAPACK
        x = np.zeros((n,) + b1.shape[1:], dtype=np.common_type(a1, b1))  # 创建一个零矩阵 x，与 b1 的形状相同
        if n == 0:
            residues = np.linalg.norm(b1, axis=0)**2  # 如果 n 为零，计算 b1 的范数的平方作为残差
        else:
            residues = np.empty((0,))  # 否则，残差为空数组
        return x, residues, 0, np.empty((0,))  # 返回零矩阵 x、残差、秩为 0 和空数组作为解

    driver = lapack_driver  # 获取 LAPACK 驱动程序
    if driver is None:
        driver = lstsq.default_lapack_driver  # 如果驱动程序为空，则使用默认的 LAPACK 驱动程序
    if driver not in ('gelsd', 'gelsy', 'gelss'):  # 检查驱动程序是否有效
        raise ValueError('LAPACK driver "%s" is not found' % driver)  # 如果驱动程序无效，则抛出 ValueError 异常

    lapack_func, lapack_lwork = get_lapack_funcs((driver,
                                                 '%s_lwork' % driver),
                                                 (a1, b1))  # 获取 LAPACK 函数和所需的工作空间大小

    real_data = True if (lapack_func.dtype.kind == 'f') else False  # 检查数据类型是否为实数类型

    if m < n:
        # 需要扩展 b 矩阵，因为它将被填充为一个较大的解矩阵
        if len(b1.shape) == 2:
            b2 = np.zeros((n, nrhs), dtype=lapack_func.dtype)  # 创建一个零矩阵 b2，与解矩阵相同的形状
            b2[:m, :] = b1  # 将 b1 的值复制到 b2 的前 m 行
        else:
            b2 = np.zeros(n, dtype=lapack_func.dtype)  # 创建一个零数组 b2，与解向量相同的形状
            b2[:m] = b1  # 将 b1 的值复制到 b2 的前 m 个元素
        b1 = b2  # 更新 b1 为扩展后的 b2

    overwrite_a = overwrite_a or _datacopied(a1, a)  # 检查是否需要覆盖数组 a1
    overwrite_b = overwrite_b or _datacopied(b1, b)  # 检查是否需要覆盖数组 b1

    if cond is None:
        cond = np.finfo(lapack_func.dtype).eps  # 如果条件数为 None，则设置为 LAPACK 函数数据类型的机器精度
    # 如果驱动程序是 'gelss' 或 'gelsd'，则执行以下操作
    if driver in ('gelss', 'gelsd'):
        # 如果驱动程序是 'gelss'
        if driver == 'gelss':
            # 计算所需的工作空间大小
            lwork = _compute_lwork(lapack_lwork, m, n, nrhs, cond)
            # 调用 LAPACK 函数进行最小二乘求解，返回结果 v, x, s, rank, work, info
            v, x, s, rank, work, info = lapack_func(a1, b1, cond, lwork,
                                                    overwrite_a=overwrite_a,
                                                    overwrite_b=overwrite_b)

        # 如果驱动程序是 'gelsd'
        elif driver == 'gelsd':
            # 如果是实数数据
            if real_data:
                # 计算所需的工作空间大小和整型工作数组
                lwork, iwork = _compute_lwork(lapack_lwork, m, n, nrhs, cond)
                # 调用 LAPACK 函数进行最小二乘求解，返回结果 x, s, rank, info
                x, s, rank, info = lapack_func(a1, b1, lwork,
                                               iwork, cond, False, False)
            else:  # 复数数据
                # 计算所需的工作空间大小、实数工作数组和整型工作数组
                lwork, rwork, iwork = _compute_lwork(lapack_lwork, m, n,
                                                     nrhs, cond)
                # 调用 LAPACK 函数进行最小二乘求解，返回结果 x, s, rank, info
                x, s, rank, info = lapack_func(a1, b1, lwork, rwork, iwork,
                                               cond, False, False)
        
        # 如果 LAPACK 函数返回值 info 大于 0，表示奇异值分解没有收敛
        if info > 0:
            raise LinAlgError("SVD did not converge in Linear Least Squares")
        # 如果 LAPACK 函数返回值 info 小于 0，表示参数传递给 LAPACK 函数的第 (-info) 个参数出现问题
        if info < 0:
            raise ValueError('illegal value in %d-th argument of internal %s'
                             % (-info, lapack_driver))
        
        # 初始化残差数组
        resids = np.asarray([], dtype=x.dtype)
        # 如果 m 大于 n
        if m > n:
            # 截取 x 的前 n 行
            x1 = x[:n]
            # 如果秩等于 n，计算残差
            if rank == n:
                resids = np.sum(np.abs(x[n:])**2, axis=0)
            x = x1
        
        # 返回结果 x, resids, rank, s
        return x, resids, rank, s

    # 如果驱动程序是 'gelsy'
    elif driver == 'gelsy':
        # 计算所需的工作空间大小
        lwork = _compute_lwork(lapack_lwork, m, n, nrhs, cond)
        # 初始化整型数组 jptv，用于 gelsy 函数
        jptv = np.zeros((a1.shape[1], 1), dtype=np.int32)
        # 调用 LAPACK 函数进行最小二乘求解，返回结果 v, x, j, rank, info
        v, x, j, rank, info = lapack_func(a1, b1, jptv, cond,
                                          lwork, False, False)
        
        # 如果 LAPACK 函数返回值 info 小于 0，表示参数传递给 LAPACK 函数的第 (-info) 个参数出现问题
        if info < 0:
            raise ValueError("illegal value in %d-th argument of internal "
                             "gelsy" % -info)
        
        # 如果 m 大于 n
        if m > n:
            # 截取 x 的前 n 行
            x1 = x[:n]
            x = x1
        
        # 返回结果 x, 空的数组，秩，None
        return x, np.array([], x.dtype), rank, None
# 设置最小二乘法求解中使用的 LAPACK 驱动程序为 'gelsd'
lstsq.default_lapack_driver = 'gelsd'

# 定义计算矩阵的伪逆的函数
def pinv(a, *, atol=None, rtol=None, return_rank=False, check_finite=True):
    """
    Compute the (Moore-Penrose) pseudo-inverse of a matrix.

    Calculate a generalized inverse of a matrix using its
    singular-value decomposition ``U @ S @ V`` in the economy mode and picking
    up only the columns/rows that are associated with significant singular
    values.

    If ``s`` is the maximum singular value of ``a``, then the
    significance cut-off value is determined by ``atol + rtol * s``. Any
    singular value below this value is assumed insignificant.

    Parameters
    ----------
    a : (M, N) array_like
        Matrix to be pseudo-inverted.
    atol : float, optional
        Absolute threshold term, default value is 0.

        .. versionadded:: 1.7.0

    rtol : float, optional
        Relative threshold term, default value is ``max(M, N) * eps`` where
        ``eps`` is the machine precision value of the datatype of ``a``.

        .. versionadded:: 1.7.0

    return_rank : bool, optional
        If True, return the effective rank of the matrix.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    B : (N, M) ndarray
        The pseudo-inverse of matrix `a`.
    rank : int
        The effective rank of the matrix. Returned if `return_rank` is True.

    Raises
    ------
    LinAlgError
        If SVD computation does not converge.

    See Also
    --------
    pinvh : Moore-Penrose pseudoinverse of a hermitian matrix.

    Notes
    -----
    If ``A`` is invertible then the Moore-Penrose pseudoinverse is exactly
    the inverse of ``A`` [1]_. If ``A`` is not invertible then the
    Moore-Penrose pseudoinverse computes the ``x`` solution to ``Ax = b`` such
    that ``||Ax - b||`` is minimized [1]_.

    References
    ----------
    .. [1] Penrose, R. (1956). On best approximate solutions of linear matrix
           equations. Mathematical Proceedings of the Cambridge Philosophical
           Society, 52(1), 17-19. doi:10.1017/S0305004100030929

    Examples
    --------

    Given an ``m x n`` matrix ``A`` and an ``n x m`` matrix ``B`` the four
    Moore-Penrose conditions are:

    1. ``ABA = A`` (``B`` is a generalized inverse of ``A``),
    2. ``BAB = B`` (``A`` is a generalized inverse of ``B``),
    3. ``(AB)* = AB`` (``AB`` is hermitian),
    4. ``(BA)* = BA`` (``BA`` is hermitian) [1]_.

    Here, ``A*`` denotes the conjugate transpose. The Moore-Penrose
    pseudoinverse is a unique ``B`` that satisfies all four of these
    conditions and exists for any ``A``. Note that, unlike the standard
    matrix inverse, ``A`` does not have to be a square matrix or have
    linearly independent columns/rows.
    """
    # 实现矩阵 `a` 的奇异值分解（SVD），并计算其伪逆
    # 使用 SVD 的经济模式，并且只选择与显著奇异值相关的列/行
    # 如果 `s` 是矩阵 `a` 的最大奇异值，则显著性截止值由 `atol + rtol * s` 决定
    # 任何低于此值的奇异值被认为是不显著的
    pass  # 实现部分未提供，故使用 pass 表示函数未完成
    """
    Given a matrix `a`, compute its Moore-Penrose pseudoinverse using the
    singular value decomposition (SVD) method and optionally return the rank.

    Parameters:
    - a : array_like
        Input matrix to compute pseudoinverse for.
    - atol : float, optional
        Absolute tolerance. Defaults to zero.
    - rtol : float, optional
        Relative tolerance. Defaults to machine epsilon scaled by the matrix size.
    - check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers. Defaults to False.
    - return_rank : bool, optional
        Whether to return the computed rank alongside the pseudoinverse matrix. Defaults to False.

    Returns:
    - B : ndarray
        Computed pseudoinverse of matrix `a`.
    - rank : int, optional
        Computed rank of matrix `a`, returned only if `return_rank=True`.

    Raises:
    - ValueError
        If either `atol` or `rtol` is negative.

    Notes:
    - This function computes the pseudoinverse using SVD, ensuring conditions similar to
      the true pseudoinverse hold true, such as A @ B @ A ≈ A and B @ A @ B ≈ B.
    """

    # Validate and convert `a` to an array if needed, ensuring finite values
    a = _asarray_validated(a, check_finite=check_finite)

    # Perform singular value decomposition (SVD) on `a`
    u, s, vh = _decomp_svd.svd(a, full_matrices=False, check_finite=False)

    # Determine the dtype character of `u`, typically 'f' for float
    t = u.dtype.char.lower()

    # Compute the maximum singular value
    maxS = np.max(s, initial=0.)

    # Set `atol` to zero if not provided, otherwise use the provided value
    atol = 0. if atol is None else atol

    # Calculate `rtol` based on the matrix size and machine epsilon if not provided
    rtol = max(a.shape) * np.finfo(t).eps if (rtol is None) else rtol

    # Ensure `atol` and `rtol` are non-negative; raise ValueError if either is negative
    if (atol < 0.) or (rtol < 0.):
        raise ValueError("atol and rtol values must be positive.")

    # Compute the tolerance value `val` based on `atol`, `maxS`, and `rtol`
    val = atol + maxS * rtol

    # Calculate the rank of `a` based on the singular values
    rank = np.sum(s > val)

    # Reduce `u` and adjust `B` accordingly to compute the pseudoinverse
    u = u[:, :rank]
    u /= s[:rank]
    B = (u @ vh[:rank]).conj().T

    # Return pseudoinverse `B` and optionally computed rank
    if return_rank:
        return B, rank
    else:
        return B
def pinvh(a, atol=None, rtol=None, lower=True, return_rank=False,
          check_finite=True):
    """
    Compute the (Moore-Penrose) pseudo-inverse of a Hermitian matrix.

    Calculate a generalized inverse of a complex Hermitian/real symmetric
    matrix using its eigenvalue decomposition and including all eigenvalues
    with 'large' absolute value.

    Parameters
    ----------
    a : (N, N) array_like
        Real symmetric or complex hermetian matrix to be pseudo-inverted

    atol : float, optional
        Absolute threshold term, default value is 0.

        .. versionadded:: 1.7.0

    rtol : float, optional
        Relative threshold term, default value is ``N * eps`` where
        ``eps`` is the machine precision value of the datatype of ``a``.

        .. versionadded:: 1.7.0

    lower : bool, optional
        Whether the pertinent array data is taken from the lower or upper
        triangle of `a`. (Default: lower)
    return_rank : bool, optional
        If True, return the effective rank of the matrix.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    B : (N, N) ndarray
        The pseudo-inverse of matrix `a`.
    rank : int
        The effective rank of the matrix.  Returned if `return_rank` is True.

    Raises
    ------
    LinAlgError
        If eigenvalue algorithm does not converge.

    See Also
    --------
    pinv : Moore-Penrose pseudoinverse of a matrix.

    Examples
    --------

    For a more detailed example see `pinv`.

    >>> import numpy as np
    >>> from scipy.linalg import pinvh
    >>> rng = np.random.default_rng()
    >>> a = rng.standard_normal((9, 6))
    >>> a = np.dot(a, a.T)
    >>> B = pinvh(a)
    >>> np.allclose(a, a @ B @ a)
    True
    >>> np.allclose(B, B @ a @ B)
    True

    """
    # 将输入矩阵转换为数组，并验证其有效性
    a = _asarray_validated(a, check_finite=check_finite)
    # 对输入矩阵进行特征值分解，返回特征值s和特征向量u
    s, u = _decomp.eigh(a, lower=lower, check_finite=False, driver='ev')
    # 确定数据类型字符表示
    t = u.dtype.char.lower()
    # 计算特征值s的最大绝对值
    maxS = np.max(np.abs(s), initial=0.)

    # 设置默认的绝对阈值atol和相对阈值rtol
    atol = 0. if atol is None else atol
    rtol = max(a.shape) * np.finfo(t).eps if (rtol is None) else rtol

    # 检查阈值是否为非负数
    if (atol < 0.) or (rtol < 0.):
        raise ValueError("atol and rtol values must be positive.")

    # 计算有效阈值val
    val = atol + maxS * rtol
    # 筛选出大于阈值的特征值
    above_cutoff = (abs(s) > val)

    # 计算伪逆矩阵B的对角线部分
    psigma_diag = 1.0 / s[above_cutoff]
    # 选取特征向量u的对应部分
    u = u[:, above_cutoff]

    # 计算伪逆矩阵B
    B = (u * psigma_diag) @ u.conj().T

    # 如果需要返回矩阵的有效秩，则返回伪逆矩阵B和秩的长度
    if return_rank:
        return B, len(psigma_diag)
    else:
        return B
    a similarity transformation such that the magnitude variation of the
    matrix entries is reflected to the scaling matrices.

    Moreover, if enabled, the matrix is first permuted to isolate the upper
    triangular parts of the matrix and, again if scaling is also enabled,
    only the remaining subblocks are subjected to scaling.

    The balanced matrix satisfies the following equality

    .. math::

                        B = T^{-1} A T

    The scaling coefficients are approximated to the nearest power of 2
    to avoid round-off errors.

    Parameters
    ----------
    A : (n, n) array_like
        Square data matrix for the balancing.
    permute : bool, optional
        The selector to define whether permutation of A is also performed
        prior to scaling.
    scale : bool, optional
        The selector to turn on and off the scaling. If False, the matrix
        will not be scaled.
    separate : bool, optional
        This switches from returning a full matrix of the transformation
        to a tuple of two separate 1-D permutation and scaling arrays.
    overwrite_a : bool, optional
        This is passed to xGEBAL directly. Essentially, overwrites the result
        to the data. It might increase the space efficiency. See LAPACK manual
        for details. This is False by default.

    Returns
    -------
    B : (n, n) ndarray
        Balanced matrix
    T : (n, n) ndarray
        A possibly permuted diagonal matrix whose nonzero entries are
        integer powers of 2 to avoid numerical truncation errors.
    scale, perm : (n,) ndarray
        If ``separate`` keyword is set to True then instead of the array
        ``T`` above, the scaling and the permutation vectors are given
        separately as a tuple without allocating the full array ``T``.

    Notes
    -----
    This algorithm is particularly useful for eigenvalue and matrix
    decompositions and in many cases it is already called by various
    LAPACK routines.

    The algorithm is based on the well-known technique of [1]_ and has
    been modified to account for special cases. See [2]_ for details
    which have been implemented since LAPACK v3.5.0. Before this version
    there are corner cases where balancing can actually worsen the
    conditioning. See [3]_ for such examples.

    The code is a wrapper around LAPACK's xGEBAL routine family for matrix
    balancing.

    .. versionadded:: 0.19.0

    References
    ----------
    .. [1] B.N. Parlett and C. Reinsch, "Balancing a Matrix for
       Calculation of Eigenvalues and Eigenvectors", Numerische Mathematik,
       Vol.13(4), 1969, :doi:`10.1007/BF02165404`
    .. [2] R. James, J. Langou, B.R. Lowery, "On matrix balancing and
       eigenvector computation", 2014, :arxiv:`1401.5766`
    .. [3] D.S. Watkins. A case where balancing is harmful.
       Electron. Trans. Numer. Anal, Vol.23, 2006.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import linalg
    # 创建一个 NumPy 数组，包含指定的数值
    >>> x = np.array([[1,2,0], [9,1,0.01], [1,2,10*np.pi]])
    
    # 使用 matrix_balance 函数对数组 x 进行矩阵平衡处理，并返回平衡后的矩阵 y 和尺度数组 permscale
    >>> y, permscale = linalg.matrix_balance(x)
    
    # 计算数组 x 每列绝对值之和除以每行绝对值之和的结果，返回一个数组
    >>> np.abs(x).sum(axis=0) / np.abs(x).sum(axis=1)
    array([ 3.66666667,  0.4995005 ,  0.91312162])
    
    # 计算平衡后的数组 y 每列绝对值之和除以每行绝对值之和的结果，返回一个数组
    >>> np.abs(y).sum(axis=0) / np.abs(y).sum(axis=1)
    array([ 1.2       ,  1.27041742,  0.92658316])  # 结果可能会有所变化
    
    # 显示平衡尺度数组 permscale 的值，表示平衡矩阵的缩放比例
    >>> permscale
    array([[  0.5,   0. ,  0. ],  # 结果可能会有所变化
           [  0. ,   1. ,  0. ],
           [  0. ,   0. ,  1. ]])
    
    """
    
    # 将输入数组 A 至少转换为二维数组，并通过 _asarray_validated 函数验证其有效性
    A = np.atleast_2d(_asarray_validated(A, check_finite=True))
    
    # 检查数组 A 是否为方阵，如果不是则引发 ValueError 异常
    if not np.equal(*A.shape):
        raise ValueError('The data matrix for balancing should be square.')
    
    # 处理空数组情况
    if A.size == 0:
        # 对单位矩阵调用 matrix_balance 函数，获取尺度数组 b_n 和 t_n
        b_n, t_n = matrix_balance(np.eye(2, dtype=A.dtype))
        # 创建与输入数组 A 相同形状的空数组 B
        B = np.empty_like(A, dtype=b_n.dtype)
        # 如果 separate 参数为真，则返回 B 和尺度数组及排列的元组
        if separate:
            scaling = np.ones_like(A, shape=len(A))
            perm = np.arange(len(A))
            return B, (scaling, perm)
        # 否则返回 B 和一个与 t_n 类型相同的空数组
        return B, np.empty_like(A, dtype=t_n.dtype)
    
    # 获取 LAPACK 库中 gebal 函数，用于矩阵平衡操作
    gebal = get_lapack_funcs(('gebal'), (A,))
    # 调用 gebal 函数对数组 A 进行平衡处理，并返回平衡后的结果 B、低位和高位索引 lo、hi、尺度数组 ps、信息标志 info
    B, lo, hi, ps, info = gebal(A, scale=scale, permute=permute,
                                overwrite_a=overwrite_a)
    
    # 如果 info 小于 0，说明 gebal 函数发生错误，根据错误代码引发 ValueError 异常
    if info < 0:
        raise ValueError('xGEBAL exited with the internal error '
                         f'"illegal value in argument number {-info}.". See '
                         'LAPACK documentation for the xGEBAL error codes.')
    
    # 创建与尺度数组 ps 类型相同的全为 1 的数组 scaling
    scaling = np.ones_like(ps, dtype=float)
    # 将 ps 中指定范围的值赋给 scaling，这些值用于矩阵的缩放操作
    scaling[lo:hi+1] = ps[lo:hi+1]
    
    # 将 ps 数组转换为整数类型，并调整索引，以便进行 LAPACK 风格的排列
    ps = ps.astype(int, copy=False) - 1
    n = A.shape[0]
    # 创建一个从 0 到 n-1 的排列数组 perm
    perm = np.arange(n)
    
    # 根据 LAPACK 的排列规则，调整排列 perm，使得顺序符合 n --> hi，然后 0 --> lo
    if hi < n:
        for ind, x in enumerate(ps[hi+1:][::-1], 1):
            if n-ind == x:
                continue
            perm[[x, n-ind]] = perm[[n-ind, x]]
    
    if lo > 0:
        for ind, x in enumerate(ps[:lo]):
            if ind == x:
                continue
            perm[[x, ind]] = perm[[ind, x]]
    
    # 如果 separate 参数为真，则返回 B 和一个包含尺度数组 scaling 和排列 perm 的元组
    if separate:
        return B, (scaling, perm)
    
    # 创建一个与 perm 相同形状的空数组 iperm，用于存储 perm 的逆排列
    iperm = np.empty_like(perm)
    iperm[perm] = np.arange(n)
    
    # 返回经过尺度矩阵缩放后的矩阵 B，以及通过 iperm 对角线索引得到的结果数组
    return B, np.diag(scaling)[iperm, :]
# 验证和格式化 toeplitz 函数的输入参数
def _validate_args_for_toeplitz_ops(c_or_cr, b, check_finite, keep_b_shape,
                                    enforce_square=True):
    """Validate arguments and format inputs for toeplitz functions

    Parameters
    ----------
    c_or_cr : array_like or tuple of (array_like, array_like)
        输入参数 ``c`` 或者元组 ``(c, r)``。不管 ``c`` 的实际形状如何，都将被转换成一维数组。
        如果未提供，则假定 ``r = conjugate(c)``；在这种情况下，如果 c[0] 是实数，Toeplitz 矩阵是 Hermite 的。
        r[0] 被忽略；Toeplitz 矩阵的第一行是 ``[c[0], r[1:]]``。无论 ``r`` 的实际形状如何，都将被转换成一维数组。
    b : (M,) or (M, K) array_like
        ``T x = b`` 中的右侧向量。
    check_finite : bool
        是否检查输入矩阵是否只包含有限数字。禁用可能会提高性能，但如果输入包含无穷大或 NaN，可能会导致问题（结果全部为 NaN）。
    keep_b_shape : bool
        是否将维度为 (M,) 的 b 转换为 (M, 1) 维度的矩阵。
    enforce_square : bool, optional
        如果为 True（默认），验证 Toeplitz 矩阵是否是方阵。

    Returns
    -------
    r : array
        Toeplitz 矩阵的第一行对应的一维数组。
    c: array
        Toeplitz 矩阵的第一列对应的一维数组。
    b: array
        验证后的 (M,), (M, 1) 或 (M, K) 维度的数组，对应于 ``b``。
    dtype: numpy 数据类型
        ``dtype`` 存储 ``r``, ``c`` 和 ``b`` 的数据类型。如果 ``r``, ``c`` 或 ``b`` 中有复数，则为 ``np.complex128``，否则为 ``np.float``。
    b_shape: tuple
        通过 ``_asarray_validated`` 处理后的 ``b`` 的形状。

    """

    # 如果 c_or_cr 是元组，则分别处理 c 和 r
    if isinstance(c_or_cr, tuple):
        c, r = c_or_cr
        c = _asarray_validated(c, check_finite=check_finite).ravel()  # 将 c 转换为一维数组
        r = _asarray_validated(r, check_finite=check_finite).ravel()  # 将 r 转换为一维数组
    else:
        c = _asarray_validated(c_or_cr, check_finite=check_finite).ravel()  # 将 c_or_cr 转换为一维数组
        r = c.conjugate()  # r 是 c 的共轭

    # 检查 b 是否为 None
    if b is None:
        raise ValueError('`b` must be an array, not None.')

    b = _asarray_validated(b, check_finite=check_finite)  # 验证并转换 b 为数组
    b_shape = b.shape  # 记录 b 的形状

    is_not_square = r.shape[0] != c.shape[0]  # 检查是否为非方阵
    if (enforce_square and is_not_square) or b.shape[0] != r.shape[0]:
        raise ValueError('Incompatible dimensions.')  # 抛出不兼容维度的异常

    is_cmplx = np.iscomplexobj(r) or np.iscomplexobj(c) or np.iscomplexobj(b)  # 检查 r, c, b 是否有复数
    dtype = np.complex128 if is_cmplx else np.float64  # 根据是否复数确定数据类型为 complex128 或 float64
    r, c, b = (np.asarray(i, dtype=dtype) for i in (r, c, b))  # 将 r, c, b 转换为指定数据类型的数组

    # 如果 b 是一维数组且不保持原始形状，将其转换为 (M, 1) 的二维数组
    if b.ndim == 1 and not keep_b_shape:
        b = b.reshape(-1, 1)
    # 如果 b 不是一维数组，则根据其大小转换为相应的二维数组
    elif b.ndim != 1:
        b = b.reshape(b.shape[0], -1 if b.size > 0 else 0)

    return r, c, b, dtype, b_shape  # 返回验证后的 r, c, b，数据类型和 b 的形状
# 定义一个函数，使用 FFT 实现高效的 Toeplitz 矩阵与稠密矩阵的乘法

def matmul_toeplitz(c_or_cr, x, check_finite=False, workers=None):
    """Efficient Toeplitz Matrix-Matrix Multiplication using FFT

    This function returns the matrix multiplication between a Toeplitz
    matrix and a dense matrix.

    The Toeplitz matrix has constant diagonals, with c as its first column
    and r as its first row. If r is not given, ``r == conjugate(c)`` is
    assumed.

    Parameters
    ----------
    c_or_cr : array_like or tuple of (array_like, array_like)
        The vector ``c``, or a tuple of arrays (``c``, ``r``). Whatever the
        actual shape of ``c``, it will be converted to a 1-D array. If not
        supplied, ``r = conjugate(c)`` is assumed; in this case, if c[0] is
        real, the Toeplitz matrix is Hermitian. r[0] is ignored; the first row
        of the Toeplitz matrix is ``[c[0], r[1:]]``. Whatever the actual shape
        of ``r``, it will be converted to a 1-D array.
        
    x : (M,) or (M, K) array_like
        Matrix with which to multiply.
        
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (result entirely NaNs) if the inputs do contain infinities or NaNs.
        
    workers : int, optional
        To pass to scipy.fft.fft and ifft. Maximum number of workers to use
        for parallel computation. If negative, the value wraps around from
        ``os.cpu_count()``. See scipy.fft.fft for more details.
        
    Returns
    -------
    T @ x : (M,) or (M, K) ndarray
        The result of the matrix multiplication ``T @ x``. Shape of return
        matches shape of `x`.

    See Also
    --------
    toeplitz : Toeplitz matrix
    solve_toeplitz : Solve a Toeplitz system using Levinson Recursion

    Notes
    -----
    The Toeplitz matrix is embedded in a circulant matrix and the FFT is used
    to efficiently calculate the matrix-matrix product.

    Because the computation is based on the FFT, integer inputs will
    result in floating point outputs.  This is unlike NumPy's `matmul`,
    which preserves the data type of the input.

    This is partly based on the implementation that can be found in [1]_,
    licensed under the MIT license. More information about the method can be
    found in reference [2]_. References [3]_ and [4]_ have more reference
    implementations in Python.

    .. versionadded:: 1.6.0

    References
    ----------
    .. [1] Jacob R Gardner, Geoff Pleiss, David Bindel, Kilian
       Q Weinberger, Andrew Gordon Wilson, "GPyTorch: Blackbox Matrix-Matrix
       Gaussian Process Inference with GPU Acceleration" with contributions
       from Max Balandat and Ruihan Wu. Available online:
       https://github.com/cornellius-gp/gpytorch
    """
    # 导入需要的模块和函数：fft, ifft, rfft, irfft
    from ..fft import fft, ifft, rfft, irfft

    # 对输入参数进行验证和处理，返回经过验证后的 c, r, x, dtype, x_shape
    r, c, x, dtype, x_shape = _validate_args_for_toeplitz_ops(
        c_or_cr, x, check_finite, keep_b_shape=False, enforce_square=False)
    
    # 获取矩阵 x 的形状信息
    n, m = x.shape

    # 获取 Toeplitz 矩阵 T 的行数和列数
    T_nrows = len(c)
    T_ncols = len(r)
    
    # 计算嵌入列 embedded_col 的长度，即 Toeplitz 矩阵的总列数
    p = T_nrows + T_ncols - 1  # 等同于 len(embedded_col)
    
    # 根据 x 的形状确定返回结果的形状
    return_shape = (T_nrows,) if len(x_shape) == 1 else (T_nrows, m)

    # 如果输入矩阵 x 是空的，则返回一个与其形状相同的空数组
    if x.size == 0:
        return np.empty_like(x, shape=return_shape)

    # 将向量 c 和反向向量 r（除了第一个元素外的倒序）拼接成嵌入列 embedded_col
    embedded_col = np.concatenate((c, r[-1:0:-1]))

    # 如果嵌入列或矩阵 x 包含复数，则进行傅里叶变换
    if np.iscomplexobj(embedded_col) or np.iscomplexobj(x):
        # 对嵌入列进行傅里叶变换
        fft_mat = fft(embedded_col, axis=0, workers=workers).reshape(-1, 1)
        # 对矩阵 x 进行零填充后的傅里叶变换
        fft_x = fft(x, n=p, axis=0, workers=workers)

        # 计算傅里叶变换后的矩阵乘积，再进行逆傅里叶变换，得到乘积结果的前 T_nrows 行
        mat_times_x = ifft(fft_mat * fft_x, axis=0,
                           workers=workers)[:T_nrows, :]
    else:
        # 处理真实输入数据，使用rfft函数可以提高速度
        # 对嵌入列进行实数快速傅里叶变换（Real FFT），沿着列轴进行计算，使用多线程加速（workers参数）
        fft_mat = rfft(embedded_col, axis=0, workers=workers).reshape(-1, 1)
        
        # 对输入数据x进行实数快速傅里叶变换（Real FFT），计算n点FFT，沿着列轴进行计算，使用多线程加速（workers参数）
        fft_x = rfft(x, n=p, axis=0, workers=workers)

        # 计算傅里叶变换后的矩阵fft_mat与输入数据fft_x的点乘，并进行逆傅里叶变换
        # 结果保留前T_nrows行，列数为p，使用多线程加速（workers参数）
        mat_times_x = irfft(fft_mat * fft_x, axis=0, workers=workers, n=p)[:T_nrows, :]

    # 返回重塑后的mat_times_x，形状与return_shape一致
    return mat_times_x.reshape(*return_shape)
```