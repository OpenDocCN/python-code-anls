# `D:\src\scipysrc\sympy\sympy\combinatorics\tests\test_polyhedron.py`

```
# 导入符号操作所需的模块和函数
from sympy.core.symbol import symbols
# 导入有限集合的表示
from sympy.sets.sets import FiniteSet
# 导入多面体相关的类和函数
from sympy.combinatorics.polyhedron import (Polyhedron,
    tetrahedron, cube as square, octahedron, dodecahedron, icosahedron,
    cube_faces)
# 导入排列和置换群相关的类和函数
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.perm_groups import PermutationGroup
# 导入测试框架中的异常抛出函数
from sympy.testing.pytest import raises

# 设置全局变量 rmul 为置换对象的乘法运算
rmul = Permutation.rmul

# 定义测试函数 test_polyhedron
def test_polyhedron():
    # 测试用例：确保使用不合法的参数创建多面体时会抛出 ValueError 异常
    raises(ValueError, lambda: Polyhedron(list('ab'),
        pgroup=[Permutation([0])]))
    
    # 定义置换群 pgroup，包含多个置换对象
    pgroup = [Permutation([[0, 7, 2, 5], [6, 1, 4, 3]]),
              Permutation([[0, 7, 1, 6], [5, 2, 4, 3]]),
              Permutation([[3, 6, 0, 5], [4, 1, 7, 2]]),
              Permutation([[7, 4, 5], [1, 3, 0], [2], [6]]),
              Permutation([[1, 3, 2], [7, 6, 5], [4], [0]]),
              Permutation([[4, 7, 6], [2, 0, 3], [1], [5]]),
              Permutation([[1, 2, 0], [4, 5, 6], [3], [7]]),
              Permutation([[4, 2], [0, 6], [3, 7], [1, 5]]),
              Permutation([[3, 5], [7, 1], [2, 6], [0, 4]]),
              Permutation([[2, 5], [1, 6], [0, 4], [3, 7]]),
              Permutation([[4, 3], [7, 0], [5, 1], [6, 2]]),
              Permutation([[4, 1], [0, 5], [6, 2], [7, 3]]),
              Permutation([[7, 2], [3, 6], [0, 4], [1, 5]]),
              Permutation([0, 1, 2, 3, 4, 5, 6, 7])]
    
    # 定义多面体的角点符号列表
    corners = tuple(symbols('A:H'))
    # 使用预定义的立方体面来定义面集合
    faces = cube_faces
    # 创建立方体对象 cube，使用角点、面集合和置换群 pgroup 初始化
    cube = Polyhedron(corners, faces, pgroup)

    # 断言：验证立方体的边集合与预期的有限集合相匹配
    assert cube.edges == FiniteSet(*(
        (0, 1), (6, 7), (1, 2), (5, 6), (0, 3), (2, 3),
        (4, 7), (4, 5), (3, 7), (1, 5), (0, 4), (2, 6)))

    # 循环遍历置换群的前三个置换，每个都进行 180 度的面旋转
    for i in range(3):
        cube.rotate(cube.pgroup[i]**2)

    # 断言：验证立方体的角点与初始化时的角点符号列表相同
    assert cube.corners == corners

    # 循环遍历置换群的第四个到第七个置换，每个都进行 240 度的角点轴向旋转
    for i in range(3, 7):
        cube.rotate(cube.pgroup[i]**2)

    # 断言：验证立方体的角点与初始化时的角点符号列表相同
    assert cube.corners == corners
    
    # 对立方体应用单个置换，并断言立方体的角点与初始化时的角点符号列表不相同
    cube.rotate(1)
    assert cube.corners != corners
    
    # 断言：验证立方体的数组形式与预期的列表形式相同
    assert cube.array_form == [7, 6, 4, 5, 3, 2, 0, 1]
    
    # 断言：验证立方体的循环形式与预期的列表形式相同
    assert cube.cyclic_form == [[0, 7, 1, 6], [2, 4, 3, 5]]
    
    # 重置立方体对象的状态
    cube.reset()
    
    # 断言：验证重置后立方体的角点与初始化时的角点符号列表相同
    assert cube.corners == corners
    def check(h, size, rpt, target):
        # 断言多面体的特性：顶点数加面数减去边数等于2
        assert len(h.faces) + len(h.vertices) - len(h.edges) == 2
        # 断言多面体的大小与给定的大小相等
        assert h.size == size

        # 用于存储所有可能的排列组合结果
        got = set()
        for p in h.pgroup:
            # 检查每个排列组合是否能够还原到原始状态
            P = h.copy()  # 创建多面体的副本
            hit = P.corners  # 记录当前多面体的角点
            for i in range(rpt):
                P.rotate(p)  # 使用排列 p 进行旋转操作
                if P.corners == hit:  # 如果旋转后角点不变，则退出循环
                    break
            else:
                print('error in permutation', p.array_form)  # 如果无法还原，则输出错误信息
            for i in range(rpt):
                P.rotate(p)  # 再次使用排列 p 进行旋转操作
                got.add(tuple(P.corners))  # 将当前多面体的角点元组添加到集合中
                c = P.corners
                f = [[c[i] for i in f] for f in P.faces]  # 重新构建多面体的面信息
                assert h.faces == Polyhedron(c, f).faces  # 断言多面体的面与重建的面信息相等
        assert len(got) == target  # 断言集合中元素的数量与目标数量相等
        assert PermutationGroup([Permutation(g) for g in got]).is_group  # 断言集合中的元素组成的置换群是一个群

    # 遍历五种多面体及其相关参数，依次调用 check 函数进行检查
    for h, size, rpt, target in zip(
        (tetrahedron, square, octahedron, dodecahedron, icosahedron),  # 五种多面体对象
        (4, 8, 6, 20, 12),  # 对应的顶点数
        (3, 4, 4, 5, 5),  # 对应的旋转次数
        (12, 24, 24, 60, 60)):  # 对应的目标排列组合数量
        check(h, size, rpt, target)  # 调用 check 函数进行检查
# 定义一个测试函数，用于验证多面体对象和面列表的对称性质
def test_pgroups():
    # 从 sympy.combinatorics.polyhedron 导入所需的对象和面列表
    from sympy.combinatorics.polyhedron import (cube, tetrahedron_faces,
            octahedron_faces, dodecahedron_faces, icosahedron_faces)
    # 导入用于计算多面体对象和面列表的函数 _pgroup_calcs
    from sympy.combinatorics.polyhedron import _pgroup_calcs
    # 调用 _pgroup_calcs 函数，获取计算得到的多面体对象和面列表
    (tetrahedron2, cube2, octahedron2, dodecahedron2, icosahedron2,
     tetrahedron_faces2, cube_faces2, octahedron_faces2,
     dodecahedron_faces2, icosahedron_faces2) = _pgroup_calcs()

    # 断言每个多面体对象与其对应计算得到的对象相等
    assert tetrahedron == tetrahedron2
    assert cube == cube2
    assert octahedron == octahedron2
    assert dodecahedron == dodecahedron2
    assert icosahedron == icosahedron2
    # 断言每个多面体的面列表，经过排序后应与对应计算得到的面列表排序后相等
    assert sorted(map(sorted, tetrahedron_faces)) == sorted(map(sorted, tetrahedron_faces2))
    assert sorted(cube_faces) == sorted(cube_faces2)
    assert sorted(octahedron_faces) == sorted(octahedron_faces2)
    assert sorted(dodecahedron_faces) == sorted(dodecahedron_faces2)
    assert sorted(icosahedron_faces) == sorted(icosahedron_faces2)
```