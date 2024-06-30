# `D:\src\scipysrc\sympy\sympy\combinatorics\generators.py`

```
# 从 sympy 库中导入排列相关的类和函数
from sympy.combinatorics.permutations import Permutation
# 从 sympy 库中导入符号类 symbols
from sympy.core.symbol import symbols
# 从 sympy 库中导入矩阵类 Matrix
from sympy.matrices import Matrix
# 从 sympy 库中导入可迭代工具函数 variations 和 rotate_left
from sympy.utilities.iterables import variations, rotate_left


def symmetric(n):
    """
    Generates the symmetric group of order n, Sn.

    Examples
    ========

    >>> from sympy.combinatorics.generators import symmetric
    >>> list(symmetric(3))
    [(2), (1 2), (2)(0 1), (0 1 2), (0 2 1), (0 2)]
    """
    # 使用 variations 函数生成指定长度为 n 的排列，并将每个排列转换为 Permutation 对象
    yield from (Permutation(perm) for perm in variations(range(n), n))


def cyclic(n):
    """
    Generates the cyclic group of order n, Cn.

    Examples
    ========

    >>> from sympy.combinatorics.generators import cyclic
    >>> list(cyclic(5))
    [(4), (0 1 2 3 4), (0 2 4 1 3),
     (0 3 1 4 2), (0 4 3 2 1)]

    See Also
    ========

    dihedral
    """
    # 生成初始序列 gen 为 0 到 n-1 的列表
    gen = list(range(n))
    # 循环生成 n 个排列，每次将 gen 向左旋转一位
    for i in range(n):
        yield Permutation(gen)
        gen = rotate_left(gen, 1)


def alternating(n):
    """
    Generates the alternating group of order n, An.

    Examples
    ========

    >>> from sympy.combinatorics.generators import alternating
    >>> list(alternating(3))
    [(2), (0 1 2), (0 2 1)]
    """
    # 遍历长度为 n 的所有排列
    for perm in variations(range(n), n):
        # 将当前排列 perm 转换为 Permutation 对象
        p = Permutation(perm)
        # 如果排列 p 是偶排列，则生成该排列
        if p.is_even:
            yield p


def dihedral(n):
    """
    Generates the dihedral group of order 2n, Dn.

    The result is given as a subgroup of Sn, except for the special cases n=1
    (the group S2) and n=2 (the Klein 4-group) where that's not possible
    and embeddings in S2 and S4 respectively are given.

    Examples
    ========

    >>> from sympy.combinatorics.generators import dihedral
    >>> list(dihedral(3))
    [(2), (0 2), (0 1 2), (1 2), (0 2 1), (2)(0 1)]

    See Also
    ========

    cyclic
    """
    # 如果 n 等于 1，生成 S2 的两个排列
    if n == 1:
        yield Permutation([0, 1])
        yield Permutation([1, 0])
    # 如果 n 等于 2，生成 Klein 四群 S4 的四个排列
    elif n == 2:
        yield Permutation([0, 1, 2, 3])
        yield Permutation([1, 0, 3, 2])
        yield Permutation([2, 3, 0, 1])
        yield Permutation([3, 2, 1, 0])
    else:
        # 生成初始序列 gen 为 0 到 n-1 的列表
        gen = list(range(n))
        # 循环生成 2n 个排列，每次生成 gen 和 gen 的反向排列，并将 gen 向左旋转一位
        for i in range(n):
            yield Permutation(gen)
            yield Permutation(gen[::-1])
            gen = rotate_left(gen, 1)


def rubik_cube_generators():
    """Return the permutations of the 3x3 Rubik's cube, see
    https://www.gap-system.org/Doc/Examples/rubik.html
    """
    # 创建一个多维列表，表示若干个排列的元组
    a = [
        [(1, 3, 8, 6), (2, 5, 7, 4), (9, 33, 25, 17), (10, 34, 26, 18), (11, 35, 27, 19)],
        [(9, 11, 16, 14), (10, 13, 15, 12), (1, 17, 41, 40), (4, 20, 44, 37), (6, 22, 46, 35)],
        [(17, 19, 24, 22), (18, 21, 23, 20), (6, 25, 43, 16), (7, 28, 42, 13), (8, 30, 41, 11)],
        [(25, 27, 32, 30), (26, 29, 31, 28), (3, 38, 43, 19), (5, 36, 45, 21), (8, 33, 48, 24)],
        [(33, 35, 40, 38), (34, 37, 39, 36), (3, 9, 46, 32), (2, 12, 47, 29), (1, 14, 48, 27)],
        [(41, 43, 48, 46), (42, 45, 47, 44), (14, 22, 30, 38), (15, 23, 31, 39), (16, 24, 32, 40)]
    ]
    # 返回一个列表，其中每个元素是一个 Permutation 对象，表示对应的排列
    return [Permutation([[i - 1 for i in xi] for xi in x], size=48) for x in a]
    def rubik(n):
        """Return permutations for an nxn Rubik's cube.

        Permutations returned are for rotation of each of the slice
        from the face up to the last face for each of the 3 sides (in this order):
        front, right and bottom. Hence, the first n - 1 permutations are for the
        slices from the front.
        """

        if n < 2:
            raise ValueError('dimension of cube must be > 1')

        # 1-based reference to rows and columns in Matrix
        def getr(f, i):
            return faces[f].col(n - i)

        def getl(f, i):
            return faces[f].col(i - 1)

        def getu(f, i):
            return faces[f].row(i - 1)

        def getd(f, i):
            return faces[f].row(n - i)

        def setr(f, i, s):
            faces[f][:, n - i] = Matrix(n, 1, s)

        def setl(f, i, s):
            faces[f][:, i - 1] = Matrix(n, 1, s)

        def setu(f, i, s):
            faces[f][i - 1, :] = Matrix(1, n, s)

        def setd(f, i, s):
            faces[f][n - i, :] = Matrix(1, n, s)

        # motion of a single face
        def cw(F, r=1):
            for _ in range(r):
                face = faces[F]
                rv = []
                for c in range(n):
                    for r in range(n - 1, -1, -1):
                        rv.append(face[r, c])
                faces[F] = Matrix(n, n, rv)

        def ccw(F):
            cw(F, 3)

        # motion of plane i from the F side;
        # fcw(0) moves the F face, fcw(1) moves the plane
        # just behind the front face, etc...
        def fcw(i, r=1):
            for _ in range(r):
                if i == 0:
                    cw(F)
                i += 1
                temp = getr(L, i)
                setr(L, i, list(getu(D, i)))
                setu(D, i, list(reversed(getl(R, i))))
                setl(R, i, list(getd(U, i)))
                setd(U, i, list(reversed(temp)))
                i -= 1

        def fccw(i):
            fcw(i, 3)

        # motion of the entire cube from the F side
        def FCW(r=1):
            for _ in range(r):
                cw(F)
                ccw(B)
                cw(U)
                t = faces[U]
                cw(L)
                faces[U] = faces[L]
                cw(D)
                faces[L] = faces[D]
                cw(R)
                faces[D] = faces[R]
                faces[R] = t

        def FCCW():
            FCW(3)

        # motion of the entire cube from the U side
        def UCW(r=1):
            for _ in range(r):
                cw(U)
                ccw(D)
                t = faces[F]
                faces[F] = faces[R]
                faces[R] = faces[B]
                faces[B] = faces[L]
                faces[L] = t

        def UCCW():
            UCW(3)

        # defining the permutations for the cube

        U, F, R, B, L, D = names = symbols('U, F, R, B, L, D')

        # the faces are represented by nxn matrices
        faces = {}
        count = 0
        for fi in range(6):
            f = []
            for a in range(n**2):
                f.append(count)
                count += 1
            faces[names[fi]] = Matrix(n, n, f)

        # this will either return the value of the current permutation
        # (show != 1) or else append the permutation to the group, g
    def perm(show=0):
        # 定义一个函数 perm，用于生成置换列表
        p = []
        for f in names:
            p.extend(faces[f])
        if show:
            return p
        g.append(Permutation(p))

    g = []  # 用于存储群的置换的容器
    I = list(range(6*n**2))  # 用于检查的恒等置换

    # 定义对应于顺时针旋转平面的置换
    # 到达从该方向上的倒数第二个平面；通过不包括最后一个平面，保持立方体的方向。
    
    # F 层
    for i in range(n - 1):
        fcw(i)  # 顺时针旋转第 i 层 F 层
        perm()  # 生成当前置换
        fccw(i)  # 恢复原状
    assert perm(1) == I  # 断言当前置换与恒等置换相等

    # R 层
    # 将 R 层移到最前面
    UCW()
    for i in range(n - 1):
        fcw(i)  # 顺时针旋转第 i 层 F 层
        UCCW()  # 将 R 层放回原位
        perm()  # 记录当前置换
        UCW()  # 恢复原状，将面移回最前面
        fccw(i)  # 将 F 层恢复原状
    UCCW()  # 恢复 R 层到原位
    assert perm(1) == I  # 断言当前置换与恒等置换相等

    # D 层
    # 将底部移到最上面
    FCW()
    UCCW()
    FCCW()
    for i in range(n - 1):
        fcw(i)  # 顺时针旋转第 i 层 F 层
        FCW()  # 将底部放回原位
        UCW()
        FCCW()  # 将 F 层恢复原状
        perm()  # 记录当前置换
        FCW()  # 恢复底部到原位
        UCCW()
        FCCW()  # 顺时针旋转第 i 层 F 层
        fccw(i)  # 将 F 层恢复原状
    FCW()  # 将底部放回原位
    UCW()
    FCCW()
    assert perm(1) == I  # 断言当前置换与恒等置换相等

    return g  # 返回存储所有置换的列表
```