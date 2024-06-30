# `D:\src\scipysrc\sympy\sympy\combinatorics\polyhedron.py`

```
    def __init__(self, corners, faces=None, pgroup=None):
        """
        Initialize a Polyhedron object.

        Parameters
        ==========

        corners : iterable
            Vertices or corners of the polyhedron.
        faces : iterable, optional
            Faces of the polyhedron.
        pgroup : PermutationGroup, optional
            Permutation group associated with the polyhedron.

        Examples
        ========

        >>> from sympy.combinatorics.polyhedron import tetrahedron
        >>> tetrahedron.corners
        (0, 1, 2, 3)
        >>> tetrahedron.size
        4
        >>> tetrahedron.faces
        Traceback (most recent call last):
        ...
        AttributeError: 'Polyhedron' object has no attribute '_faces'
        >>> tetrahedron.pgroup
        Traceback (most recent call last):
        ...
        AttributeError: 'Polyhedron' object has no attribute '_pgroup'

        Notes
        =====

        This constructor initializes a Polyhedron object with corners,
        optionally faces, and optionally a permutation group.

        """
        self._corners = Tuple(*corners)
        self._faces = faces
        self._pgroup = pgroup
    # 定义一个方法来计算多面体的边
    def edges(self):
        """
        给定多面体的面，我们可以得到其边。

        Examples
        ========

        >>> from sympy.combinatorics import Polyhedron
        >>> from sympy.abc import a, b, c
        >>> corners = (a, b, c)
        >>> faces = [(0, 1, 2)]
        >>> Polyhedron(corners, faces).edges
        {(0, 1), (0, 2), (1, 2)}

        """
        # 如果尚未计算过边的集合
        if self._edges is None:
            # 初始化一个空集合用于存储边
            output = set()
            # 遍历每一个面
            for face in self.faces:
                # 遍历每个面的顶点
                for i in range(len(face)):
                    # 计算当前顶点与前一个顶点组成的边，并保证顶点顺序一致
                    edge = tuple(sorted([face[i], face[i - 1]]))
                    # 将边加入到输出集合中
                    output.add(edge)
            # 使用得到的边集合创建一个有限集对象，并存储在 self._edges 中
            self._edges = FiniteSet(*output)
        # 返回存储在 self._edges 中的边集合
        return self._edges
    def rotate(self, perm):
        """
        Apply a permutation to the polyhedron *in place*. The permutation
        may be given as a Permutation instance or an integer indicating
        which permutation from pgroup of the Polyhedron should be
        applied.

        This is an operation that is analogous to rotation about
        an axis by a fixed increment.

        Notes
        =====

        When a Permutation is applied, no check is done to see if that
        is a valid permutation for the Polyhedron. For example, a cube
        could be given a permutation which effectively swaps only 2
        vertices. A valid permutation (that rotates the object in a
        physical way) will be obtained if one only uses
        permutations from the ``pgroup`` of the Polyhedron. On the other
        hand, allowing arbitrary rotations (applications of permutations)
        gives a way to follow named elements rather than indices since
        Polyhedron allows vertices to be named while Permutation works
        only with indices.

        Examples
        ========

        >>> from sympy.combinatorics import Polyhedron, Permutation
        >>> from sympy.combinatorics.polyhedron import cube
        >>> cube = cube.copy()
        >>> cube.corners
        (0, 1, 2, 3, 4, 5, 6, 7)
        >>> cube.rotate(0)
        >>> cube.corners
        (1, 2, 3, 0, 5, 6, 7, 4)

        A non-physical "rotation" that is not prohibited by this method:

        >>> cube.reset()
        >>> cube.rotate(Permutation([[1, 2]], size=8))
        >>> cube.corners
        (0, 2, 1, 3, 4, 5, 6, 7)

        Polyhedron can be used to follow elements of set that are
        identified by letters instead of integers:

        >>> shadow = h5 = Polyhedron(list('abcde'))
        >>> p = Permutation([3, 0, 1, 2, 4])
        >>> h5.rotate(p)
        >>> h5.corners
        (d, a, b, c, e)
        >>> _ == shadow.corners
        True
        >>> copy = h5.copy()
        >>> h5.rotate(p)
        >>> h5.corners == copy.corners
        False
        """
        # Check if perm is not a Permutation instance, convert it to a Permutation from the pgroup
        if not isinstance(perm, Perm):
            perm = self.pgroup[perm]
            # and we know it's valid
        else:
            # Verify that the size of perm matches the size of the Polyhedron
            if perm.size != self.size:
                raise ValueError('Polyhedron and Permutation sizes differ.')
        # Obtain the permutation in array form
        a = perm.array_form
        # Apply the permutation to the corners of the Polyhedron
        corners = [self.corners[a[i]] for i in range(len(self.corners))]
        # Update the corners of the Polyhedron with the new permutation
        self._corners = tuple(corners)

    def reset(self):
        """Return corners to their original positions.

        Examples
        ========

        >>> from sympy.combinatorics.polyhedron import tetrahedron as T
        >>> T = T.copy()
        >>> T.corners
        (0, 1, 2, 3)
        >>> T.rotate(0)
        >>> T.corners
        (0, 2, 3, 1)
        >>> T.reset()
        >>> T.corners
        (0, 1, 2, 3)
        """
        # Reset the corners of the Polyhedron to their original positions
        self._corners = self.args[0]
# 定义一个函数 _pgroup_calcs，用于计算多面体的置换群和面定义
def _pgroup_calcs():
    """Return the permutation groups for each of the polyhedra and the face
    definitions: tetrahedron, cube, octahedron, dodecahedron, icosahedron,
    tetrahedron_faces, cube_faces, octahedron_faces, dodecahedron_faces,
    icosahedron_faces

    Explanation
    ===========

    (This author did not find and did not know of a better way to do it though
    there likely is such a way.)

    Although only 2 permutations are needed for a polyhedron in order to
    generate all the possible orientations, a group of permutations is
    provided instead. A set of permutations is called a "group" if::

    a*b = c (for any pair of permutations in the group, a and b, their
    product, c, is in the group)

    a*(b*c) = (a*b)*c (for any 3 permutations in the group associativity holds)

    there is an identity permutation, I, such that I*a = a*I for all elements
    in the group

    a*b = I (the inverse of each permutation is also in the group)

    None of the polyhedron groups defined follow these definitions of a group.
    Instead, they are selected to contain those permutations whose powers
    alone will construct all orientations of the polyhedron, i.e. for
    permutations ``a``, ``b``, etc... in the group, ``a, a**2, ..., a**o_a``,
    ``b, b**2, ..., b**o_b``, etc... (where ``o_i`` is the order of
    permutation ``i``) generate all permutations of the polyhedron instead of
    mixed products like ``a*b``, ``a*b**2``, etc....

    Note that for a polyhedron with n vertices, the valid permutations of the
    vertices exclude those that do not maintain its faces. e.g. the
    permutation BCDE of a square's four corners, ABCD, is a valid
    permutation while CBDE is not (because this would twist the square).

    Examples
    ========

    The is_group checks for: closure, the presence of the Identity permutation,
    and the presence of the inverse for each of the elements in the group. This
    confirms that none of the polyhedra are true groups:

    >>> from sympy.combinatorics.polyhedron import (
    ... tetrahedron, cube, octahedron, dodecahedron, icosahedron)
    ...
    >>> polyhedra = (tetrahedron, cube, octahedron, dodecahedron, icosahedron)
    >>> [h.pgroup.is_group for h in polyhedra]
    ...
    [True, True, True, True, True]

    Although tests in polyhedron's test suite check that powers of the
    permutations in the groups generate all permutations of the vertices
    of the polyhedron, here we also demonstrate the powers of the given
    permutations create a complete group for the tetrahedron:

    >>> from sympy.combinatorics import Permutation, PermutationGroup
    >>> for h in polyhedra[:1]:
    ...     G = h.pgroup
    ...     perms = set()
    ...     for g in G:
    ...         for e in range(g.order()):
    ...             p = tuple((g**e).array_form)
    ...             perms.add(p)
    ...
    ...     perms = [Permutation(p) for p in perms]

    """
    # 确保给定的置换群 perms 是一个有效的群
    assert PermutationGroup(perms).is_group

    # 在执行上述操作的同时，测试套件确认每个置换应用后所有面都存在。

    # 参考资料
    # [1] https://dogschool.tripod.com/trianglegroup.html

    """
    将一个多面体的双面体群作用于指定的顺序面集合，返回结果作为置换群的列表。

    Parameters
    ----------
    polyh : Polyhedron
        外部多面体对象，用于定义双面体的作用。
    ordered_faces : list
        多面体面的顺序列表。
    pgroup : list of Perm
        置换群，每个置换表示一个操作作用于外部多面体的双面体。

    Returns
    -------
    list of Perm
        经过双面体群作用后的置换群列表。
    """
    def _pgroup_of_double(polyh, ordered_faces, pgroup):
        n = len(ordered_faces[0])
        
        # 为了追踪外部多面体的面，建立面到双面体顶点的映射
        fmap = dict(zip(ordered_faces, range(len(ordered_faces))))
        
        # 展平面列表，以便后续处理
        flat_faces = flatten(ordered_faces)
        
        new_pgroup = []
        for p in pgroup:
            h = polyh.copy()
            h.rotate(p)
            c = h.corners
            
            # 根据顺序面重新排列顶点，以确保枚举面时顶点的正确顺序
            reorder = unflatten([c[j] for j in flat_faces], n)
            
            # 将顶点排列为规范形式
            reorder = [tuple(map(as_int, minlex(f, directed=False))) for f in reorder]
            
            # 将面映射到顶点：结果列表是所寻求的双面体的置换
            new_pgroup.append(Perm([fmap[f] for f in reorder]))
        
        return new_pgroup

    tetrahedron_faces = [
        (0, 1, 2), (0, 2, 3), (0, 3, 1),  # 上部三面
        (1, 2, 3),  # 底部面
    ]

    # 顺时针从顶部看
    _t_pgroup = [
        Perm([[1, 2, 3], [0]]),  # 从顶部顺时针
        Perm([[0, 1, 2], [3]]),  # 从前面顺时针
        Perm([[0, 3, 2], [1]]),  # 从右后方面顺时针
        Perm([[0, 3, 1], [2]]),  # 从左后方面顺时针
        Perm([[0, 1], [2, 3]]),  # 穿过前左边缘
        Perm([[0, 2], [1, 3]]),  # 穿过前右边缘
        Perm([[0, 3], [1, 2]]),  # 穿过后边缘
    ]

    # 创建一个四面体对象
    tetrahedron = Polyhedron(
        range(4),
        tetrahedron_faces,
        _t_pgroup)

    cube_faces = [
        (0, 1, 2, 3),  # 上部面
        (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (0, 3, 7, 4),  # 中部四个面
        (4, 5, 6, 7),  # 底部面
    ]

    # U, D, F, B, L, R = 上，下，前，后，左，右
    # 定义魔方的每个面的旋转置换组合
    _c_pgroup = [Perm(p) for p in
        [
        [1, 2, 3, 0, 5, 6, 7, 4],  # 从顶部顺时针旋转，U面
        [4, 0, 3, 7, 5, 1, 2, 6],  # 从前面顺时针旋转，F面
        [4, 5, 1, 0, 7, 6, 2, 3],  # 从右面顺时针旋转，R面

        [1, 0, 4, 5, 2, 3, 7, 6],  # 通过UF棱顺时针旋转
        [6, 2, 1, 5, 7, 3, 0, 4],  # 通过UR棱顺时针旋转
        [6, 7, 3, 2, 5, 4, 0, 1],  # 通过UB棱顺时针旋转
        [3, 7, 4, 0, 2, 6, 5, 1],  # 通过UL棱顺时针旋转
        [4, 7, 6, 5, 0, 3, 2, 1],  # 通过FL棱顺时针旋转
        [6, 5, 4, 7, 2, 1, 0, 3],  # 通过FR棱顺时针旋转

        [0, 3, 7, 4, 1, 2, 6, 5],  # 通过UFL角顺时针旋转
        [5, 1, 0, 4, 6, 2, 3, 7],  # 通过UFR角顺时针旋转
        [5, 6, 2, 1, 4, 7, 3, 0],  # 通过UBR角顺时针旋转
        [7, 4, 0, 3, 6, 5, 1, 2],  # 通过UBL角顺时针旋转
        ]]

    # 创建魔方的多面体对象，包括顶点、面和旋转置换组合
    cube = Polyhedron(
        range(8),
        cube_faces,
        _c_pgroup)

    # 定义八面体的面列表
    octahedron_faces = [
        (0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 1, 4),  # 顶部4个面
        (1, 2, 5), (2, 3, 5), (3, 4, 5), (1, 4, 5),  # 底部4个面
    ]

    # 创建八面体的多面体对象，包括顶点、面和特定的旋转置换组合
    octahedron = Polyhedron(
        range(6),
        octahedron_faces,
        _pgroup_of_double(cube, cube_faces, _c_pgroup))

    # 定义十二面体的面列表
    dodecahedron_faces = [
        (0, 1, 2, 3, 4),  # 顶部
        (0, 1, 6, 10, 5), (1, 2, 7, 11, 6), (2, 3, 8, 12, 7),  # 上部5个面
        (3, 4, 9, 13, 8), (0, 4, 9, 14, 5),
        (5, 10, 16, 15, 14), (6, 10, 16, 17, 11), (7, 11, 17, 18,
          12),  # 下部5个面
        (8, 12, 18, 19, 13), (9, 13, 19, 15, 14),
        (15, 16, 17, 18, 19)  # 底部
    ]

    # 将字符串转换为置换对象
    def _string_to_perm(s):
        rv = [Perm(range(20))]
        p = None
        for si in s:
            if si not in '01':
                count = int(si) - 1
            else:
                count = 1
                if si == '0':
                    p = _f0
                elif si == '1':
                    p = _f1
            rv.extend([p]*count)
        return Perm.rmul(*rv)

    # 定义顶部面的顺时针旋转置换
    _f0 = Perm([
        1, 2, 3, 4, 0, 6, 7, 8, 9, 5, 11,
        12, 13, 14, 10, 16, 17, 18, 19, 15])
    # 定义前面面的顺时针旋转置换
    _f1 = Perm([
        5, 0, 4, 9, 14, 10, 1, 3, 13, 15,
        6, 2, 8, 19, 16, 17, 11, 7, 12, 18])
    # 下面的字符串，如0104，代表F0*F1*F0**4的简写，包括剩余的4个面旋转、15个棱的置换和
    # 10个顶点的旋转。
    _dodeca_pgroup = [_f0, _f1] + [_string_to_perm(s) for s in '''
    0104 140 014 0410
    010 1403 03104 04103 102
    120 1304 01303 021302 03130
    0412041 041204103 04120410 041204104 041204102
    10 01 1402 0140 04102 0412 1204 1302 0130 03120'''.strip().split()]

    # 创建十二面体的多面体对象，包括顶点、面和旋转置换组合
    dodecahedron = Polyhedron(
        range(20),
        dodecahedron_faces,
        _dodeca_pgroup)

    # 定义二十面体的面列表
    icosahedron_faces = [
        (0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5), (0, 1, 5),
        (1, 6, 7), (1, 2, 7), (2, 7, 8), (2, 3, 8), (3, 8, 9),
        (3, 4, 9), (4, 9, 10), (4, 5, 10), (5, 6, 10), (1, 5, 6),
        (6, 7, 11), (7, 8, 11), (8, 9, 11), (9, 10, 11), (6, 10, 11)]
    # 创建一个五面体对象（Tetrahedron），使用指定的顶点索引和面列表
    tetrahedron = Polyhedron(
        range(4),  # 使用范围为 0 到 3 的顶点索引
        tetrahedron_faces,  # 使用预定义的五面体面列表
        _pgroup_of_double(  # 使用给定的函数生成顶点群
            tetrahedron,  # 使用五面体作为基础形状
            tetrahedron_faces,  # 五面体的面列表
            _tetra_pgroup))  # 使用特定函数生成五面体的顶点群
    
    # 创建一个立方体对象（Cube），使用指定的顶点索引和面列表
    cube = Polyhedron(
        range(8),  # 使用范围为 0 到 7 的顶点索引
        cube_faces,  # 使用预定义的立方体面列表
        _pgroup_of_double(  # 使用给定的函数生成顶点群
            cube,  # 使用立方体作为基础形状
            cube_faces,  # 立方体的面列表
            _cube_pgroup))  # 使用特定函数生成立方体的顶点群
    
    # 创建一个八面体对象（Octahedron），使用指定的顶点索引和面列表
    octahedron = Polyhedron(
        range(6),  # 使用范围为 0 到 5 的顶点索引
        octahedron_faces,  # 使用预定义的八面体面列表
        _pgroup_of_double(  # 使用给定的函数生成顶点群
            octahedron,  # 使用八面体作为基础形状
            octahedron_faces,  # 八面体的面列表
            _octa_pgroup))  # 使用特定函数生成八面体的顶点群
    
    # 创建一个十二面体对象（Dodecahedron），使用指定的顶点索引和面列表
    dodecahedron = Polyhedron(
        range(20),  # 使用范围为 0 到 19 的顶点索引
        dodecahedron_faces,  # 使用预定义的十二面体面列表
        _pgroup_of_double(  # 使用给定的函数生成顶点群
            dodecahedron,  # 使用十二面体作为基础形状
            dodecahedron_faces,  # 十二面体的面列表
            _dodeca_pgroup))  # 使用特定函数生成十二面体的顶点群
    
    # 创建一个二十面体对象（Icosahedron），使用指定的顶点索引和面列表
    icosahedron = Polyhedron(
        range(12),  # 使用范围为 0 到 11 的顶点索引
        icosahedron_faces,  # 使用预定义的二十面体面列表
        _pgroup_of_double(  # 使用给定的函数生成顶点群
            dodecahedron,  # 使用十二面体作为基础形状
            dodecahedron_faces,  # 十二面体的面列表
            _dodeca_pgroup))  # 使用特定函数生成十二面体的顶点群
    
    # 返回创建的各种多面体对象及其对应的面列表
    return (tetrahedron, cube, octahedron, dodecahedron, icosahedron,
        tetrahedron_faces, cube_faces, octahedron_faces,
        dodecahedron_faces, icosahedron_faces)
# -----------------------------------------------------------------------
#   Standard Polyhedron groups
#
#   These are generated using _pgroup_calcs() above. However to save
#   import time we encode them explicitly here.
# -----------------------------------------------------------------------

# 创建四面体的多面体对象，定义顶点、面和面的顶点索引
tetrahedron = Polyhedron(
    Tuple(0, 1, 2, 3),  # 顶点索引
    Tuple(  # 面的顶点索引
        Tuple(0, 1, 2),
        Tuple(0, 2, 3),
        Tuple(0, 1, 3),
        Tuple(1, 2, 3)),
    Tuple(  # 面的旋转置换
        Perm(1, 2, 3),
        Perm(3)(0, 1, 2),
        Perm(0, 3, 2),
        Perm(0, 3, 1),
        Perm(0, 1)(2, 3),
        Perm(0, 2)(1, 3),
        Perm(0, 3)(1, 2)
    ))

# 创建立方体的多面体对象，定义顶点、面和面的顶点索引
cube = Polyhedron(
    Tuple(0, 1, 2, 3, 4, 5, 6, 7),  # 顶点索引
    Tuple(  # 面的顶点索引
        Tuple(0, 1, 2, 3),
        Tuple(0, 1, 5, 4),
        Tuple(1, 2, 6, 5),
        Tuple(2, 3, 7, 6),
        Tuple(0, 3, 7, 4),
        Tuple(4, 5, 6, 7)),
    Tuple(  # 面的旋转置换
        Perm(0, 1, 2, 3)(4, 5, 6, 7),
        Perm(0, 4, 5, 1)(2, 3, 7, 6),
        Perm(0, 4, 7, 3)(1, 5, 6, 2),
        Perm(0, 1)(2, 4)(3, 5)(6, 7),
        Perm(0, 6)(1, 2)(3, 5)(4, 7),
        Perm(0, 6)(1, 7)(2, 3)(4, 5),
        Perm(0, 3)(1, 7)(2, 4)(5, 6),
        Perm(0, 4)(1, 7)(2, 6)(3, 5),
        Perm(0, 6)(1, 5)(2, 4)(3, 7),
        Perm(1, 3, 4)(2, 7, 5),
        Perm(7)(0, 5, 2)(3, 4, 6),
        Perm(0, 5, 7)(1, 6, 3),
        Perm(0, 7, 2)(1, 4, 6)))

# 创建八面体的多面体对象，定义顶点、面和面的顶点索引
octahedron = Polyhedron(
    Tuple(0, 1, 2, 3, 4, 5),  # 顶点索引
    Tuple(  # 面的顶点索引
        Tuple(0, 1, 2),
        Tuple(0, 2, 3),
        Tuple(0, 3, 4),
        Tuple(0, 1, 4),
        Tuple(1, 2, 5),
        Tuple(2, 3, 5),
        Tuple(3, 4, 5),
        Tuple(1, 4, 5)),
    Tuple(  # 面的旋转置换
        Perm(5)(1, 2, 3, 4),
        Perm(0, 4, 5, 2),
        Perm(0, 1, 5, 3),
        Perm(0, 1)(2, 4)(3, 5),
        Perm(0, 2)(1, 3)(4, 5),
        Perm(0, 3)(1, 5)(2, 4),
        Perm(0, 4)(1, 3)(2, 5),
        Perm(0, 5)(1, 4)(2, 3),
        Perm(0, 5)(1, 2)(3, 4),
        Perm(0, 4, 1)(2, 3, 5),
        Perm(0, 1, 2)(3, 4, 5),
        Perm(0, 2, 3)(1, 5, 4),
        Perm(0, 4, 3)(1, 5, 2)))

# 创建十二面体的多面体对象，定义顶点、面和面的顶点索引
dodecahedron = Polyhedron(
    Tuple(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19),  # 顶点索引
    Tuple(  # 面的顶点索引
        Tuple(0, 1, 2, 3, 4),
        Tuple(0, 1, 6, 10, 5),
        Tuple(1, 2, 7, 11, 6),
        Tuple(2, 3, 8, 12, 7),
        Tuple(3, 4, 9, 13, 8),
        Tuple(0, 4, 9, 14, 5),
        Tuple(5, 10, 16, 15, 14),
        Tuple(6, 10, 16, 17, 11),
        Tuple(7, 11, 17, 18, 12),
        Tuple(8, 12, 18, 19, 13),
        Tuple(9, 13, 19, 15, 14),
        Tuple(15, 16, 17, 18, 19)),  # 面的顶点索引
    # 面的旋转置换
    Tuple(
        Perm(5)(1, 2, 3, 4),
        Perm(0, 4, 5, 2),
        Perm(0, 1, 5, 3),
        Perm(0, 1)(2, 4)(3, 5),
        Perm(0, 2)(1, 3)(4, 5),
        Perm(0, 3)(1, 5)(2, 4),
        Perm(0, 4)(1, 3)(2, 5),
        Perm(0, 5)(1, 4)(2, 3),
        Perm(0, 5)(1, 2)(3, 4),
        Perm(0, 4, 1)(2, 3, 5),
        Perm(0, 1, 2)(3, 4, 5),
        Perm(0, 2, 3)(1, 5, 4),
        Perm(0, 4, 3)(1, 5, 2)))
    # 创建一个包含多个 Perm 对象的 Tuple 对象
    Tuple(
        # 第一个 Perm 对象，每个 Perm 对象由多个参数构成
        Perm(0, 1, 2, 3, 4)(5, 6, 7, 8, 9)(10, 11, 12, 13, 14)(15, 16, 17, 18, 19),
        # 第二个 Perm 对象
        Perm(0, 5, 10, 6, 1)(2, 4, 14, 16, 11)(3, 9, 15, 17, 7)(8, 13, 19, 18, 12),
        # 第三个 Perm 对象
        Perm(0, 10, 17, 12, 3)(1, 6, 11, 7, 2)(4, 5, 16, 18, 8)(9, 14, 15, 19, 13),
        # 第四个 Perm 对象
        Perm(0, 6, 17, 19, 9)(1, 11, 18, 13, 4)(2, 7, 12, 8, 3)(5, 10, 16, 15, 14),
        # 第五个 Perm 对象
        Perm(0, 2, 12, 19, 14)(1, 7, 18, 15, 5)(3, 8, 13, 9, 4)(6, 11, 17, 16, 10),
        # 第六个 Perm 对象
        Perm(0, 4, 9, 14, 5)(1, 3, 13, 15, 10)(2, 8, 19, 16, 6)(7, 12, 18, 17, 11),
        # 第七个 Perm 对象
        Perm(0, 1)(2, 5)(3, 10)(4, 6)(7, 14)(8, 16)(9, 11)(12, 15)(13, 17)(18, 19),
        # 第八个 Perm 对象
        Perm(0, 7)(1, 2)(3, 6)(4, 11)(5, 12)(8, 10)(9, 17)(13, 16)(14, 18)(15, 19),
        # 第九个 Perm 对象
        Perm(0, 12)(1, 8)(2, 3)(4, 7)(5, 18)(6, 13)(9, 11)(10, 19)(14, 17)(15, 16),
        # 第十个 Perm 对象
        Perm(0, 8)(1, 13)(2, 9)(3, 4)(5, 12)(6, 19)(7, 14)(10, 18)(11, 15)(16, 17),
        # 第十一个 Perm 对象
        Perm(0, 4)(1, 9)(2, 14)(3, 5)(6, 13)(7, 15)(8, 10)(11, 19)(12, 16)(17, 18),
        # 第十二个 Perm 对象
        Perm(0, 5)(1, 14)(2, 15)(3, 16)(4, 10)(6, 9)(7, 19)(8, 17)(11, 13)(12, 18),
        # 第十三个 Perm 对象
        Perm(0, 11)(1, 6)(2, 10)(3, 16)(4, 17)(5, 7)(8, 15)(9, 18)(12, 14)(13, 19),
        # 第十四个 Perm 对象
        Perm(0, 18)(1, 12)(2, 7)(3, 11)(4, 17)(5, 19)(6, 8)(9, 16)(10, 13)(14, 15),
        # 第十五个 Perm 对象
        Perm(0, 18)(1, 19)(2, 13)(3, 8)(4, 12)(5, 17)(6, 15)(7, 9)(10, 16)(11, 14),
        # 第十六个 Perm 对象
        Perm(0, 13)(1, 19)(2, 15)(3, 14)(4, 9)(5, 8)(6, 18)(7, 16)(10, 12)(11, 17),
        # 第十七个 Perm 对象
        Perm(0, 16)(1, 15)(2, 19)(3, 18)(4, 17)(5, 10)(6, 14)(7, 13)(8, 12)(9, 11),
        # 第十八个 Perm 对象
        Perm(0, 18)(1, 17)(2, 16)(3, 15)(4, 19)(5, 12)(6, 11)(7, 10)(8, 14)(9, 13),
        # 第十九个 Perm 对象
        Perm(0, 15)(1, 19)(2, 18)(3, 17)(4, 16)(5, 14)(6, 13)(7, 12)(8, 11)(9, 10),
        # 第二十个 Perm 对象
        Perm(0, 17)(1, 16)(2, 15)(3, 19)(4, 18)(5, 11)(6, 10)(7, 14)(8, 13)(9, 12),
        # 第二十一个 Perm 对象
        Perm(0, 19)(1, 18)(2, 17)(3, 16)(4, 15)(5, 13)(6, 12)(7, 11)(8, 10)(9, 14),
        # 第二十二个 Perm 对象
        Perm(1, 4, 5)(2, 9, 10)(3, 14, 6)(7, 13, 16)(8, 15, 11)(12, 19, 17),
        # 第二十三个 Perm 对象
        Perm(19)(0, 6, 2)(3, 5, 11)(4, 10, 7)(8, 14, 17)(9, 16, 12)(13, 15, 18),
        # 第二十四个 Perm 对象
        Perm(0, 11, 8)(1, 7, 3)(4, 6, 12)(5, 17, 13)(9, 10, 18)(14, 16, 19),
        # 第二十五个 Perm 对象
        Perm(0, 7, 13)(1, 12, 9)(2, 8, 4)(5, 11, 19)(6, 18, 14)(10, 17, 15),
        # 第二十六个 Perm 对象
        Perm(0, 3, 9)(1, 8, 14)(2, 13, 5)(6, 12, 15)(7, 19, 10)(11, 18, 16),
        # 第二十七个 Perm 对象
        Perm(0, 14, 10)(1, 9, 16)(2, 13, 17)(3, 19, 11)(4, 15, 6)(7, 8, 18),
        # 第二十八个 Perm 对象
        Perm(0, 16, 7)(1, 10, 11)(2, 5, 17)(3, 14, 18)(4, 15, 12)(8, 9, 19),
        # 第二十九个 Perm 对象
        Perm(0, 16, 13)(1, 17, 8)(2, 11, 12)(3, 6, 18)(4, 10, 19)(5, 15, 9),
        # 第三十个 Perm 对象
        Perm(0, 11, 15)(1, 17, 14)(2, 18, 9)(3, 12, 13)(4, 7, 19)(5, 6, 16),
        # 第三十一个 Perm 对象
        Perm(0, 8, 15)(1, 12, 16)(2, 18, 10)(3, 19, 5)(4, 13, 14)(6, 7, 17)
    )
# 定义一个正20面体对象，并指定其顶点的索引
icosahedron = Polyhedron(
    Tuple(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
    Tuple(
        Tuple(0, 1, 2),
        Tuple(0, 2, 3),
        Tuple(0, 3, 4),
        Tuple(0, 4, 5),
        Tuple(0, 1, 5),
        Tuple(1, 6, 7),
        Tuple(1, 2, 7),
        Tuple(2, 7, 8),
        Tuple(2, 3, 8),
        Tuple(3, 8, 9),
        Tuple(3, 4, 9),
        Tuple(4, 9, 10),
        Tuple(4, 5, 10),
        Tuple(5, 6, 10),
        Tuple(1, 5, 6),
        Tuple(6, 7, 11),
        Tuple(7, 8, 11),
        Tuple(8, 9, 11),
        Tuple(9, 10, 11),
        Tuple(6, 10, 11)),
    Tuple(
        Perm(11)(1, 2, 3, 4, 5)(6, 7, 8, 9, 10),    # 顶点置换群的置换，用于定义面的顺序
        Perm(0, 5, 6, 7, 2)(3, 4, 10, 11, 8),
        Perm(0, 1, 7, 8, 3)(4, 5, 6, 11, 9),
        Perm(0, 2, 8, 9, 4)(1, 7, 11, 10, 5),
        Perm(0, 3, 9, 10, 5)(1, 2, 8, 11, 6),
        Perm(0, 4, 10, 6, 1)(2, 3, 9, 11, 7),
        Perm(0, 1)(2, 5)(3, 6)(4, 7)(8, 10)(9, 11),
        Perm(0, 2)(1, 3)(4, 7)(5, 8)(6, 9)(10, 11),
        Perm(0, 3)(1, 9)(2, 4)(5, 8)(6, 11)(7, 10),
        Perm(0, 4)(1, 9)(2, 10)(3, 5)(6, 8)(7, 11),
        Perm(0, 5)(1, 4)(2, 10)(3, 6)(7, 9)(8, 11),
        Perm(0, 6)(1, 5)(2, 10)(3, 11)(4, 7)(8, 9),
        Perm(0, 7)(1, 2)(3, 6)(4, 11)(5, 8)(9, 10),
        Perm(0, 8)(1, 9)(2, 3)(4, 7)(5, 11)(6, 10),
        Perm(0, 9)(1, 11)(2, 10)(3, 4)(5, 8)(6, 7),
        Perm(0, 10)(1, 9)(2, 11)(3, 6)(4, 5)(7, 8),
        Perm(0, 11)(1, 6)(2, 10)(3, 9)(4, 8)(5, 7),
        Perm(0, 11)(1, 8)(2, 7)(3, 6)(4, 10)(5, 9),
        Perm(0, 11)(1, 10)(2, 9)(3, 8)(4, 7)(5, 6),
        Perm(0, 11)(1, 7)(2, 6)(3, 10)(4, 9)(5, 8),
        Perm(0, 11)(1, 9)(2, 8)(3, 7)(4, 6)(5, 10),
        Perm(0, 5, 1)(2, 4, 6)(3, 10, 7)(8, 9, 11),
        Perm(0, 1, 2)(3, 5, 7)(4, 6, 8)(9, 10, 11),
        Perm(0, 2, 3)(1, 8, 4)(5, 7, 9)(6, 11, 10),
        Perm(0, 3, 4)(1, 8, 10)(2, 9, 5)(6, 7, 11),
        Perm(0, 4, 5)(1, 3, 10)(2, 9, 6)(7, 8, 11),
        Perm(0, 10, 7)(1, 5, 6)(2, 4, 11)(3, 9, 8),
        Perm(0, 6, 8)(1, 7, 2)(3, 5, 11)(4, 10, 9),
        Perm(0, 7, 9)(1, 11, 4)(2, 8, 3)(5, 6, 10),
        Perm(0, 8, 10)(1, 7, 6)(2, 11, 5)(3, 9, 4),
        Perm(0, 9, 6)(1, 3, 11)(2, 8, 7)(4, 10, 5)))
# 获取正20面体的面并将其转换为元组列表
icosahedron_faces = [tuple(arg) for arg in icosahedron.faces]
```