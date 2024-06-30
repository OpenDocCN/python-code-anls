# `D:\src\scipysrc\sympy\sympy\physics\matrices.py`

```
"""Known matrices related to physics"""

# 导入复数单位 I 和矩阵类 Matrix
from sympy.core.numbers import I
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.utilities.decorator import deprecated


def msigma(i):
    r"""Returns a Pauli matrix `\sigma_i` with `i=1,2,3`.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Pauli_matrices

    Examples
    ========

    >>> from sympy.physics.matrices import msigma
    >>> msigma(1)
    Matrix([
    [0, 1],
    [1, 0]])
    """
    # 根据给定的 Pauli 矩阵索引生成相应的矩阵
    if i == 1:
        mat = (
            (0, 1),
            (1, 0)
        )
    elif i == 2:
        mat = (
            (0, -I),
            (I, 0)
        )
    elif i == 3:
        mat = (
            (1, 0),
            (0, -1)
        )
    else:
        # 如果索引超出范围，抛出异常
        raise IndexError("Invalid Pauli index")
    return Matrix(mat)


def pat_matrix(m, dx, dy, dz):
    """Returns the Parallel Axis Theorem matrix to translate the inertia
    matrix a distance of `(dx, dy, dz)` for a body of mass m.

    Examples
    ========

    To translate a body having a mass of 2 units a distance of 1 unit along
    the `x`-axis we get:

    >>> from sympy.physics.matrices import pat_matrix
    >>> pat_matrix(2, 1, 0, 0)
    Matrix([
    [0, 0, 0],
    [0, 2, 0],
    [0, 0, 2]])

    """
    # 计算平行轴定理的惯性矩阵的平移版本
    dxdy = -dx*dy
    dydz = -dy*dz
    dzdx = -dz*dx
    dxdx = dx**2
    dydy = dy**2
    dzdz = dz**2
    mat = ((dydy + dzdz, dxdy, dzdx),
           (dxdy, dxdx + dzdz, dydz),
           (dzdx, dydz, dydy + dxdx))
    return m*Matrix(mat)


def mgamma(mu, lower=False):
    r"""Returns a Dirac gamma matrix `\gamma^\mu` in the standard
    (Dirac) representation.

    Explanation
    ===========

    If you want `\gamma_\mu`, use ``gamma(mu, True)``.

    We use a convention:

    `\gamma^5 = i \cdot \gamma^0 \cdot \gamma^1 \cdot \gamma^2 \cdot \gamma^3`

    `\gamma_5 = i \cdot \gamma_0 \cdot \gamma_1 \cdot \gamma_2 \cdot \gamma_3 = - \gamma^5`

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gamma_matrices

    Examples
    ========

    >>> from sympy.physics.matrices import mgamma
    >>> mgamma(1)
    Matrix([
    [ 0,  0, 0, 1],
    [ 0,  0, 1, 0],
    [ 0, -1, 0, 0],
    [-1,  0, 0, 0]])
    """
    # 根据给定的 Dirac 矩阵索引生成相应的矩阵
    if mu not in (0, 1, 2, 3, 5):
        raise IndexError("Invalid Dirac index")
    if mu == 0:
        mat = (
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, -1, 0),
            (0, 0, 0, -1)
        )
    elif mu == 1:
        mat = (
            (0, 0, 0, 1),
            (0, 0, 1, 0),
            (0, -1, 0, 0),
            (-1, 0, 0, 0)
        )
    elif mu == 2:
        mat = (
            (0, 0, 0, -I),
            (0, 0, I, 0),
            (0, I, 0, 0),
            (-I, 0, 0, 0)
        )
    elif mu == 3:
        mat = (
            (0, 0, 1, 0),
            (0, 0, 0, -1),
            (-1, 0, 0, 0),
            (0, 1, 0, 0)
        )
    elif mu == 5:
        mat = (
            (0, 0, 1, 0),    # 如果 mu 等于 5，则创建一个特定的4x4矩阵，用于后续操作
            (0, 0, 0, 1),    # 这个矩阵是一个元组，表示一个特定的数学转换
            (1, 0, 0, 0),    # 其中每个元组代表矩阵的一行
            (0, 1, 0, 0)
        )
    m = Matrix(mat)        # 使用上述定义的矩阵 mat 创建一个 Matrix 对象 m
    if lower:
        if mu in (1, 2, 3, 5):  # 如果 lower 为真且 mu 的值是 1、2、3 或 5 中的一个
            m = -m             # 则将矩阵 m 取负
    return m                # 返回处理后的矩阵 m
# 定义 Minkowski 张量，使用量子场论中的约定（+,-,-,-）
minkowski_tensor = Matrix( (
    (1, 0, 0, 0),    # 第一行：(+, 0, 0, 0)
    (0, -1, 0, 0),   # 第二行：(0, -, 0, 0)
    (0, 0, -1, 0),   # 第三行：(0, 0, -, 0)
    (0, 0, 0, -1)    # 第四行：(0, 0, 0, -)
)

@deprecated(
    """
    sympy.physics.matrices.mdft 方法已弃用。请使用 sympy.DFT(n).as_explicit() 替代。
    """,
    deprecated_since_version="1.9",
    active_deprecations_target="deprecated-physics-mdft",
)
# mdft 函数，用于返回 DFT(n) 的可变版本
def mdft(n):
    r"""
    .. deprecated:: 1.9

       使用 sympy.matrices.expressions.fourier 中的 DFT。

       要获得与 ``mdft(n)`` 相同的行为，请使用 ``DFT(n).as_explicit()``。
    """
    from sympy.matrices.expressions.fourier import DFT
    return DFT(n).as_mutable()
```