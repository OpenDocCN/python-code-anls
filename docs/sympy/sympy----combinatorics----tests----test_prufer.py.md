# `D:\src\scipysrc\sympy\sympy\combinatorics\tests\test_prufer.py`

```
# 从 sympy.combinatorics.prufer 导入 Prufer 类
from sympy.combinatorics.prufer import Prufer
# 从 sympy.testing.pytest 导入 raises 函数
from sympy.testing.pytest import raises


# 定义测试函数 test_prufer
def test_prufer():
    # 测试 Prufer 类的构造函数，接受边列表和节点数，节点数为可选参数
    assert Prufer([[0, 1], [0, 2], [0, 3], [0, 4]], 5).nodes == 5
    assert Prufer([[0, 1], [0, 2], [0, 3], [0, 4]]).nodes == 5

    # 创建 Prufer 对象 a，并进行断言验证其属性
    a = Prufer([[0, 1], [0, 2], [0, 3], [0, 4]])
    assert a.rank == 0
    assert a.nodes == 5
    assert a.prufer_repr == [0, 0, 0]

    # 创建 Prufer 对象 a，使用不同的边列表，并进行断言验证其属性
    a = Prufer([[2, 4], [1, 4], [1, 3], [0, 5], [0, 4]])
    assert a.rank == 924
    assert a.nodes == 6
    assert a.tree_repr == [[2, 4], [1, 4], [1, 3], [0, 5], [0, 4]]
    assert a.prufer_repr == [4, 1, 4, 0]

    # 使用 Prufer 类的静态方法 edges，验证其返回的边列表和节点数
    assert Prufer.edges([0, 1, 2, 3], [1, 4, 5], [1, 4, 6]) == \
        ([[0, 1], [1, 2], [1, 4], [2, 3], [4, 5], [4, 6]], 7)
    assert Prufer([0]*4).size == Prufer([6]*4).size == 1296

    # 测试 Prufer 类接受可迭代对象作为参数，但将其转换为列表列表形式
    tree = [(0, 1), (1, 5), (0, 3), (0, 2), (2, 6), (4, 7), (2, 4)]
    tree_lists = [list(t) for t in tree]
    assert Prufer(tree).tree_repr == tree_lists
    assert sorted(Prufer(set(tree)).tree_repr) == sorted(tree_lists)

    # 使用 lambda 函数和 raises 函数测试 Prufer 类对于缺少必要节点的边列表会引发 ValueError
    raises(ValueError, lambda: Prufer([[1, 2], [3, 4]]))  # 0 is missing
    raises(ValueError, lambda: Prufer([[2, 3], [3, 4]]))  # 0, 1 are missing
    assert Prufer(*Prufer.edges([1, 2], [3, 4])).prufer_repr == [1, 3]
    raises(ValueError, lambda: Prufer.edges(
        [1, 3], [3, 4]))  # a broken tree but edges doesn't care
    raises(ValueError, lambda: Prufer.edges([1, 2], [5, 6]))
    raises(ValueError, lambda: Prufer([[]]))

    # 创建 Prufer 对象 a，并进行断言验证其下一个对象 b 的属性
    a = Prufer([[0, 1], [0, 2], [0, 3]])
    b = a.next()
    assert b.tree_repr == [[0, 2], [0, 1], [1, 3]]
    assert b.rank == 1


# 定义测试函数 test_round_trip
def test_round_trip():
    # 定义内部函数 doit，测试 Prufer 类在边列表和节点数下的互相转换和属性验证
    def doit(t, b):
        e, n = Prufer.edges(*t)
        t = Prufer(e, n)
        a = sorted(t.tree_repr)
        b = [i - 1 for i in b]
        assert t.prufer_repr == b
        assert sorted(Prufer(b).tree_repr) == a
        assert Prufer.unrank(t.rank, n).prufer_repr == b

    # 执行多个 doit 函数调用，验证 Prufer 类在不同输入下的正确性
    doit([[1, 2]], [])
    doit([[2, 1, 3]], [1])
    doit([[1, 3, 2]], [3])
    doit([[1, 2, 3]], [2])
    doit([[2, 1, 4], [1, 3]], [1, 1])
    doit([[3, 2, 1, 4]], [2, 1])
    doit([[3, 2, 1], [2, 4]], [2, 2])
    doit([[1, 3, 2, 4]], [3, 2])
    doit([[1, 4, 2, 3]], [4, 2])
    doit([[3, 1, 4, 2]], [4, 1])
    doit([[4, 2, 1, 3]], [1, 2])
    doit([[1, 2, 4, 3]], [2, 4])
    doit([[1, 3, 4, 2]], [3, 4])
    doit([[2, 4, 1], [4, 3]], [4, 4])
    doit([[1, 2, 3, 4]], [2, 3])
    doit([[2, 3, 1], [3, 4]], [3, 3])
    doit([[1, 4, 3, 2]], [4, 3])
    doit([[2, 1, 4, 3]], [1, 4])
    doit([[2, 1, 3, 4]], [1, 3])
    doit([[6, 2, 1, 4], [1, 3, 5, 8], [3, 7]], [1, 2, 1, 3, 3, 5])
```