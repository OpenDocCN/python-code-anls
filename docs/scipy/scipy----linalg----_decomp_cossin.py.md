# `D:\src\scipysrc\scipy\scipy\linalg\_decomp_cossin.py`

```
# 导入所需的模块和函数
from collections.abc import Iterable  # 导入Iterable类用于判断是否可迭代
import numpy as np  # 导入NumPy库并使用别名np

# 导入Scipy中的特定函数和类
from scipy._lib._util import _asarray_validated  # 导入_asarray_validated函数
from scipy.linalg import block_diag, LinAlgError  # 导入block_diag函数和LinAlgError异常
from .lapack import _compute_lwork, get_lapack_funcs  # 从lapack模块导入_compute_lwork和get_lapack_funcs函数

__all__ = ['cossin']  # 设置模块的公开接口，只包括cossin函数

def cossin(X, p=None, q=None, separate=False,
           swap_sign=False, compute_u=True, compute_vh=True):
    """
    Compute the cosine-sine (CS) decomposition of an orthogonal/unitary matrix.

    X is an ``(m, m)`` orthogonal/unitary matrix, partitioned as the following
    where upper left block has the shape of ``(p, q)``::

                                   ┌                   ┐
                                   │ I  0  0 │ 0  0  0 │
        ┌           ┐   ┌         ┐│ 0  C  0 │ 0 -S  0 │┌         ┐*
        │ X11 │ X12 │   │ U1 │    ││ 0  0  0 │ 0  0 -I ││ V1 │    │
        │ ────┼──── │ = │────┼────││─────────┼─────────││────┼────│
        │ X21 │ X22 │   │    │ U2 ││ 0  0  0 │ I  0  0 ││    │ V2 │
        └           ┘   └         ┘│ 0  S  0 │ 0  C  0 │└         ┘
                                   │ 0  0  I │ 0  0  0 │
                                   └                   ┘

    ``U1``, ``U2``, ``V1``, ``V2`` are square orthogonal/unitary matrices of
    dimensions ``(p,p)``, ``(m-p,m-p)``, ``(q,q)``, and ``(m-q,m-q)``
    respectively, and ``C`` and ``S`` are ``(r, r)`` nonnegative diagonal
    matrices satisfying ``C^2 + S^2 = I`` where ``r = min(p, m-p, q, m-q)``.

    Moreover, the rank of the identity matrices are ``min(p, q) - r``,
    ``min(p, m - q) - r``, ``min(m - p, q) - r``, and ``min(m - p, m - q) - r``
    respectively.

    X can be supplied either by itself and block specifications p, q or its
    subblocks in an iterable from which the shapes would be derived. See the
    examples below.

    Parameters
    ----------
    X : array_like, iterable
        complex unitary or real orthogonal matrix to be decomposed, or iterable
        of subblocks ``X11``, ``X12``, ``X21``, ``X22``, when ``p``, ``q`` are
        omitted.
    p : int, optional
        Number of rows of the upper left block ``X11``, used only when X is
        given as an array.
    q : int, optional
        Number of columns of the upper left block ``X11``, used only when X is
        given as an array.
    separate : bool, optional
        if ``True``, the low level components are returned instead of the
        matrix factors, i.e. ``(u1,u2)``, ``theta``, ``(v1h,v2h)`` instead of
        ``u``, ``cs``, ``vh``.
    swap_sign : bool, optional
        if ``True``, the ``-S``, ``-I`` block will be the bottom left,
        otherwise (by default) they will be in the upper right block.
    compute_u : bool, optional
        if ``False``, ``u`` won't be computed and an empty array is returned.
    compute_vh : bool, optional
        if ``False``, ``vh`` won't be computed and an empty array is returned.

    Returns
    -------
    u : ndarray
        当 ``compute_u=True`` 时，包含由块 ``U1`` (``p`` x ``p``) 和 ``U2`` (``m-p`` x ``m-p``) 组成的块对角正交/酉矩阵。如果 ``separate=True``，则包含 ``(U1, U2)`` 的元组。
    cs : ndarray
        具有上述结构的余弦-正弦因子。
        如果 ``separate=True``，则包含包含角度（弧度制）的 ``theta`` 数组。
    vh : ndarray
        当 ``compute_vh=True`` 时，包含由块 ``V1H`` (``q`` x ``q``) 和 ``V2H`` (``m-q`` x ``m-q``) 组成的块对角正交/酉矩阵。如果 ``separate=True``，则包含 ``(V1H, V2H)`` 的元组。

    References
    ----------
    .. [1] Brian D. Sutton. Computing the complete CS decomposition. Numer.
           Algorithms, 50(1):33-65, 2009.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import cossin
    >>> from scipy.stats import unitary_group
    >>> x = unitary_group.rvs(4)
    >>> u, cs, vdh = cossin(x, p=2, q=2)
    >>> np.allclose(x, u @ cs @ vdh)
    True

    Same can be entered via subblocks without the need of ``p`` and ``q``. Also
    let's skip the computation of ``u``

    >>> ue, cs, vdh = cossin((x[:2, :2], x[:2, 2:], x[2:, :2], x[2:, 2:]),
    ...                      compute_u=False)
    >>> print(ue)
    []
    >>> np.allclose(x, u @ cs @ vdh)
    True

    """

    # 如果 p 或 q 存在，则进行以下操作
    if p or q:
        # 将 p 和 q 转换为整数，如果它们为 None 则设为 1
        p = 1 if p is None else int(p)
        q = 1 if q is None else int(q)
        # 将 X 转换为一个验证过的 ndarray，确保其元素有限
        X = _asarray_validated(X, check_finite=True)
        # 检查 X 是否是方阵
        if not np.equal(*X.shape):
            raise ValueError("Cosine Sine decomposition only supports square"
                             f" matrices, got {X.shape}")
        m = X.shape[0]
        # 检查 p 和 q 的有效性
        if p >= m or p <= 0:
            raise ValueError(f"invalid p={p}, 0<p<{X.shape[0]} must hold")
        if q >= m or q <= 0:
            raise ValueError(f"invalid q={q}, 0<q<{X.shape[0]} must hold")

        # 按照 p 和 q 的值分割 X 成四个子块
        x11, x12, x21, x22 = X[:p, :q], X[:p, q:], X[p:, :q], X[p:, q:]
    # 如果 p 和 q 均为 None，而 X 不是可迭代对象，则引发错误
    elif not isinstance(X, Iterable):
        raise ValueError("When p and q are None, X must be an Iterable"
                         " containing the subblocks of X")
    else:
        # 如果 p 和 q 都为 None，则 X 应该包含四个数组
        if len(X) != 4:
            raise ValueError("When p and q are None, exactly four arrays"
                             f" should be in X, got {len(X)}")

        # 将每个数组至少转换为二维数组
        x11, x12, x21, x22 = (np.atleast_2d(x) for x in X)

        # 检查每个块是否为空
        for name, block in zip(["x11", "x12", "x21", "x22"],
                               [x11, x12, x21, x22]):
            if block.shape[1] == 0:
                raise ValueError(f"{name} can't be empty")

        # 获取 x11 和 x22 的维度
        p, q = x11.shape
        mmp, mmq = x22.shape

        # 检查 x12 和 x21 的维度是否符合预期
        if x12.shape != (p, mmq):
            raise ValueError(f"Invalid x12 dimensions: desired {(p, mmq)}, "
                             f"got {x12.shape}")

        if x21.shape != (mmp, q):
            raise ValueError(f"Invalid x21 dimensions: desired {(mmp, q)}, "
                             f"got {x21.shape}")

        # 检查子块是否形成一个方阵
        if p + mmp != q + mmq:
            raise ValueError("The subblocks have compatible sizes but "
                             "don't form a square array (instead they form a"
                             " {}x{} array). This might be due to missing "
                             "p, q arguments.".format(p + mmp, q + mmq))

        # 计算 m 的值
        m = p + mmp

    # 检查是否存在复数类型的输入
    cplx = any([np.iscomplexobj(x) for x in [x11, x12, x21, x22]])

    # 确定要使用的 LAPACK 驱动程序
    driver = "uncsd" if cplx else "orcsd"

    # 获取 LAPACK 函数 csd 和 csd_lwork
    csd, csd_lwork = get_lapack_funcs([driver, driver + "_lwork"],
                                      [x11, x12, x21, x22])

    # 计算所需的工作空间大小
    lwork = _compute_lwork(csd_lwork, m=m, p=p, q=q)

    # 根据是否为复数，设置工作空间参数
    lwork_args = ({'lwork': lwork[0], 'lrwork': lwork[1]} if cplx else
                  {'lwork': lwork})

    # 调用 LAPACK 函数 csd，获取返回的结果
    *_, theta, u1, u2, v1h, v2h, info = csd(x11=x11, x12=x12, x21=x21, x22=x22,
                                            compute_u1=compute_u,
                                            compute_u2=compute_u,
                                            compute_v1t=compute_vh,
                                            compute_v2t=compute_vh,
                                            trans=False, signs=swap_sign,
                                            **lwork_args)

    # 获取使用的 LAPACK 方法名称
    method_name = csd.typecode + driver

    # 检查返回的信息是否小于 0，如果是，则引发异常
    if info < 0:
        raise ValueError(f'illegal value in argument {-info} '
                         f'of internal {method_name}')

    # 检查返回的信息是否大于 0，如果是，则引发线性代数错误异常
    if info > 0:
        raise LinAlgError(f"{method_name} did not converge: {info}")

    # 如果需要分离结果，则返回相应的元组
    if separate:
        return (u1, u2), theta, (v1h, v2h)

    # 否则，构造 U 和 VDH 矩阵
    U = block_diag(u1, u2)
    VDH = block_diag(v1h, v2h)

    # 构造中间因子 CS
    c = np.diag(np.cos(theta))
    s = np.diag(np.sin(theta))
    r = min(p, q, m - p, m - q)
    n11 = min(p, q) - r
    n12 = min(p, m - q) - r
    n21 = min(m - p, q) - r
    n22 = min(m - p, m - q) - r
    Id = np.eye(np.max([n11, n12, n21, n22, r]), dtype=theta.dtype)
    CS = np.zeros((m, m), dtype=theta.dtype)

    # 填充 CS 的对角块
    CS[:n11, :n11] = Id[:n11, :n11]

    # 设置索引以填充 CS 的其余部分
    xs = n11 + r
    xe = n11 + r + n12
    ys = n11 + n21 + n22 + 2 * r
    ye = n11 + n21 + n22 + 2 * r + n12
    # 在 CS 矩阵的指定区域设置对角矩阵或其相反数，取决于 swap_sign 的布尔值
    CS[xs: xe, ys:ye] = Id[:n12, :n12] if swap_sign else -Id[:n12, :n12]

    # 设置新的索引范围用于 CS 矩阵的下一个区域
    xs = p + n22 + r
    xe = p + n22 + r + + n21
    ys = n11 + r
    ye = n11 + r + n21
    # 在 CS 矩阵的指定区域设置对角矩阵的相反数或其本身，取决于 swap_sign 的布尔值
    CS[xs:xe, ys:ye] = -Id[:n21, :n21] if swap_sign else Id[:n21, :n21]

    # 在 CS 矩阵的指定区域设置对角矩阵
    CS[p:p + n22, q:q + n22] = Id[:n22, :n22]

    # 在 CS 矩阵的指定区域设置常数矩阵 c
    CS[n11:n11 + r, n11:n11 + r] = c

    # 在 CS 矩阵的指定区域设置常数矩阵 c
    CS[p + n22:p + n22 + r, r + n21 + n22:2 * r + n21 + n22] = c

    # 设置新的索引范围用于 CS 矩阵的下一个区域
    xs = n11
    xe = n11 + r
    ys = n11 + n21 + n22 + r
    ye = n11 + n21 + n22 + 2 * r
    # 在 CS 矩阵的指定区域设置正负号取决于 swap_sign 的矩阵 s
    CS[xs:xe, ys:ye] = s if swap_sign else -s

    # 在 CS 矩阵的指定区域设置正负号取决于 swap_sign 的矩阵 s
    CS[p + n22:p + n22 + r, n11:n11 + r] = -s if swap_sign else s

    # 返回计算结果 U, CS, VDH
    return U, CS, VDH
```