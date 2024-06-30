# `D:\src\scipysrc\scipy\scipy\sparse\tests\test_dok.py`

```
# 导入 pytest 库，用于编写和运行测试
import pytest
# 导入 numpy 库，并将其重命名为 np，用于处理数值计算和数组操作
import numpy as np
# 从 numpy.testing 模块导入 assert_equal 函数，用于测试 numpy 数组的相等性
from numpy.testing import assert_equal
# 导入 scipy 库，并从中稀疏矩阵模块（sparse）中导入 dok_array 和 dok_matrix 类
import scipy as sp
from scipy.sparse import dok_array, dok_matrix

# 定义测试前的准备工作，使用 pytest.fixture 装饰器
@pytest.fixture
def d():
    return {(0, 1): 1, (0, 2): 2}

# 定义测试前的准备工作，使用 pytest.fixture 装饰器
@pytest.fixture
def A():
    return np.array([[0, 1, 2], [0, 0, 0], [0, 0, 0]])

# 定义参数化测试，使用 pytest.fixture 装饰器，参数为 dok_array 或 dok_matrix
@pytest.fixture(params=[dok_array, dok_matrix])
def Asp(request):
    # 根据参数创建 dok_array 或 dok_matrix 对象，并初始化
    A = request.param((3, 3))
    A[(0, 1)] = 1
    A[(0, 2)] = 2
    yield A

# 注意：__iter__ 和比较运算符对 DOK 格式类似于 ndarray，而不是字典。
# DOK 格式的 __iter__ 和相关比较运算符 reversed、or、ror、ior 在 dok_matrix 中与字典相似，在 dok_array 中会引发异常。
# DOK 格式的所有其它字典方法在 dok_matrix 中表现为字典方法（带有额外的检查）。

# 开始测试
################

# 测试 dict 方法在 dok_array 和 dok_matrix 中的覆盖情况
def test_dict_methods_covered(d, Asp):
    # 获取 d 和 Asp 的方法集合，去除 "__class_getitem__" 方法后进行比较
    d_methods = set(dir(d)) - {"__class_getitem__"}
    asp_methods = set(dir(Asp))
    assert d_methods < asp_methods

# 测试 clear 方法在 d 和 Asp 中的作用
def test_clear(d, Asp):
    # 检查 d 和 Asp 的 items 是否相等
    assert d.items() == Asp.items()
    # 清空 d 和 Asp
    d.clear()
    Asp.clear()
    # 再次检查清空后的 d 和 Asp 的 items 是否相等
    assert d.items() == Asp.items()

# 测试 copy 方法在 d 和 Asp 中的作用
def test_copy(d, Asp):
    # 检查 d 和 Asp 的 items 是否相等
    assert d.items() == Asp.items()
    # 复制 d 和 Asp
    dd = d.copy()
    asp = Asp.copy()
    # 检查复制后的 dd 和 asp 的 items 是否相等
    assert dd.items() == asp.items()
    assert asp.items() == Asp.items()
    # 修改 asp 的元素
    asp[(0, 1)] = 3
    # 检查修改后，Asp 中对应元素的值是否为 1
    assert Asp[(0, 1)] == 1

# 测试 fromkeys 方法在 dok_array 中的作用，使用默认值
def test_fromkeys_default():
    # 使用默认值测试 fromkeys 方法
    edges = [(0, 2), (1, 0), (2, 1)]
    Xdok = dok_array.fromkeys(edges)
    X = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    assert_equal(Xdok.toarray(), X)

# 测试 fromkeys 方法在 dok_array 中的作用，使用指定位置的值
def test_fromkeys_positional():
    # 使用指定位置的值测试 fromkeys 方法
    edges = [(0, 2), (1, 0), (2, 1)]
    Xdok = dok_array.fromkeys(edges, -1)
    X = [[0, 0, -1], [-1, 0, 0], [0, -1, 0]]
    assert_equal(Xdok.toarray(), X)

# 测试 fromkeys 方法在 dok_array 中的作用，使用迭代器
def test_fromkeys_iterator():
    # 使用迭代器测试 fromkeys 方法
    it = ((a, a % 2) for a in range(4))
    Xdok = dok_array.fromkeys(it)
    X = [[1, 0], [0, 1], [1, 0], [0, 1]]
    assert_equal(Xdok.toarray(), X)

# 测试 get 方法在 d 和 Asp 中的作用
def test_get(d, Asp):
    # 检查 get 方法在 d 和 Asp 中的行为
    assert Asp.get((0, 1)) == d.get((0, 1))
    assert Asp.get((0, 0), 99) == d.get((0, 0), 99)
    # 检查 get 方法在超出边界时是否抛出 IndexError 异常
    with pytest.raises(IndexError, match="out of bounds"):
        Asp.get((0, 4), 99)

# 测试 items 方法在 d 和 Asp 中的作用
def test_items(d, Asp):
    # 检查 items 方法在 d 和 Asp 中的行为
    assert Asp.items() == d.items()

# 测试 keys 方法在 d 和 Asp 中的作用
def test_keys(d, Asp):
    # 检查 keys 方法在 d 和 Asp 中的行为
    assert Asp.keys() == d.keys()

# 测试 pop 方法在 d 和 Asp 中的作用
def test_pop(d, Asp):
    # 检查 pop 方法在 d 和 Asp 中的行为
    assert d.pop((0, 1)) == 1
    assert Asp.pop((0, 1)) == 1
    # 检查 pop 方法后 d 和 Asp 的 items 是否相等
    assert d.items() == Asp.items()

    # 测试 pop 方法在找不到键时的行为
    assert Asp.pop((22, 21), None) is None
    assert Asp.pop((22, 21), "other") == "other"
    # 检查 pop 方法在找不到键且没有默认返回值时是否引发 KeyError 异常
    with pytest.raises(KeyError, match="(22, 21)"):
        Asp.pop((22, 21))
    # 检查 pop 方法在提供了不期望的关键字参数时是否引发 TypeError 异常
    with pytest.raises(TypeError, match="got an unexpected keyword argument"):
        Asp.pop((22, 21), default=5)

# 测试 popitem 方法在 d 和 Asp 中的作用
def test_popitem(d, Asp):
    # 检查 popitem 方法在 d 和 Asp 中的行为
    assert d.popitem() == Asp.popitem()
    # 检查 popitem 方法后 d 和 Asp 的 items 是否相等
    assert d.items() == Asp.items()

# 测试 setdefault 方法在 d 和 Asp 中的作用
def test_setdefault(d, Asp):
    # 检查 setdefault 方法在 d 和 Asp 中的行为
    assert Asp.setdefault((0, 1), 4) == 1
    assert Asp.setdefault((2, 2), 4) == 4
    # 添加新元素后，检查 d 和 Asp 的 items 是否相等
    d.setdefault((0, 1), 4)
    d.setdefault((2, 2), 4)
    assert d.items() == Asp.items()

# 测试 update 方法在 Asp 中的行为，期望引发 NotImplementedError 异常
def test_update(d, Asp):
    with pytest.raises(NotImplementedError):
        Asp.update(Asp)

# 测试 values 方法在 d 和 Asp 中的作用
def test_values(d, Asp):
    # 注释: 注意字典的值（values）在比较时的特殊行为：d={1: 1}; d.values() == d.values() 返回 False
    # 使用 list(d.values()) 将其转换为列表可以使它们可以进行比较。
    assert list(Asp.values()) == list(d.values())
# 检验特殊方法 __getitem__ 的功能
def test_dunder_getitem(d, Asp):
    # 断言使用 Asp[(0, 1)] 与 d[(0, 1)] 相等
    assert Asp[(0, 1)] == d[(0, 1)]

# 检验特殊方法 __setitem__ 的功能
def test_dunder_setitem(d, Asp):
    # 设置 Asp[(1, 1)] 和 d[(1, 1)] 的值为 5
    Asp[(1, 1)] = 5
    d[(1, 1)] = 5
    # 断言 d.items() 和 Asp.items() 相等
    assert d.items() == Asp.items()

# 检验特殊方法 __delitem__ 的功能
def test_dunder_delitem(d, Asp):
    # 删除 Asp[(0, 1)] 和 d[(0, 1)]
    del Asp[(0, 1)]
    del d[(0, 1)]
    # 断言 d.items() 和 Asp.items() 相等
    assert d.items() == Asp.items()

# 检验特殊方法 __contains__ 的功能
def test_dunder_contains(d, Asp):
    # 断言 (0, 1) 是否在 d 和 Asp 中
    assert ((0, 1) in d) == ((0, 1) in Asp)
    assert ((0, 0) in d) == ((0, 0) in Asp)

# 检验特殊方法 __len__ 的功能
def test_dunder_len(d, Asp):
    # 断言 d 的长度与 Asp 的长度相等
    assert len(d) == len(Asp)

# 检验特殊方法 __reversed__ 的功能
def test_dunder_reversed(d, Asp):
    # 如果 Asp 是 dok_array 类型，则期望抛出 TypeError 异常
    if isinstance(Asp, dok_array):
        with pytest.raises(TypeError):
            list(reversed(Asp))
    else:
        # 断言反转后的 Asp 和 d 相等
        list(reversed(Asp)) == list(reversed(d))

# 检验特殊方法 __ior__ 的功能
def test_dunder_ior(d, Asp):
    # 如果 Asp 是 dok_array 类型，则期望抛出 TypeError 异常
    if isinstance(Asp, dok_array):
        with pytest.raises(TypeError):
            Asp |= Asp
    else:
        # 创建一个新字典 dd，并与 Asp 合并
        dd = {(0, 0): 5}
        Asp |= dd
        assert Asp[(0, 0)] == 5
        d |= dd
        assert d.items() == Asp.items()
        dd |= Asp
        assert dd.items() == Asp.items()

# 检验特殊方法 __or__ 的功能
def test_dunder_or(d, Asp):
    # 如果 Asp 是 dok_array 类型，则期望抛出 TypeError 异常
    if isinstance(Asp, dok_array):
        with pytest.raises(TypeError):
            Asp | Asp
    else:
        # 断言 d | d 和 Asp | d 相等，以及 d | d 和 Asp | Asp 相等
        assert d | d == Asp | d
        assert d | d == Asp | Asp

# 检验特殊方法 __ror__ 的功能
def test_dunder_ror(d, Asp):
    # 如果 Asp 是 dok_array 类型，则期望抛出 TypeError 异常
    if isinstance(Asp, dok_array):
        with pytest.raises(TypeError):
            Asp | Asp
        with pytest.raises(TypeError):
            d | Asp
    else:
        # 断言 Asp.__ror__(d) 和 Asp.__ror__(Asp) 相等，以及 d.__ror__(d) 和 Asp.__ror__(d) 相等
        assert Asp.__ror__(d) == Asp.__ror__(Asp)
        assert d.__ror__(d) == Asp.__ror__(d)
        assert d | Asp

# 检验特殊方法 __eq__ 的功能
def test_dunder_eq(A, Asp):
    # 使用 np.testing.suppress_warnings 忽略 SparseEfficiencyWarning
    with np.testing.suppress_warnings() as sup:
        sup.filter(sp.sparse.SparseEfficiencyWarning)
        # 断言 (Asp == Asp).toarray().all() 和 (A == Asp).all() 成立
        assert (Asp == Asp).toarray().all()
        assert (A == Asp).all()

# 检验特殊方法 __ne__ 的功能
def test_dunder_ne(A, Asp):
    # 断言 (Asp != Asp).toarray().any() 和 (A != Asp).any() 均为 False
    assert not (Asp != Asp).toarray().any()
    assert not (A != Asp).any()

# 检验特殊方法 __lt__ 的功能
def test_dunder_lt(A, Asp):
    # 断言 (Asp < Asp).toarray().any() 和 (A < Asp).any() 均为 False
    assert not (Asp < Asp).toarray().any()
    assert not (A < Asp).any()

# 检验特殊方法 __gt__ 的功能
def test_dunder_gt(A, Asp):
    # 断言 (Asp > Asp).toarray().any() 和 (A > Asp).any() 均为 False
    assert not (Asp > Asp).toarray().any()
    assert not (A > Asp).any()

# 检验特殊方法 __le__ 的功能
def test_dunder_le(A, Asp):
    # 使用 np.testing.suppress_warnings 忽略 SparseEfficiencyWarning
    with np.testing.suppress_warnings() as sup:
        sup.filter(sp.sparse.SparseEfficiencyWarning)
        # 断言 (Asp <= Asp).toarray().all() 和 (A <= Asp).all() 成立
        assert (Asp <= Asp).toarray().all()
        assert (A <= Asp).all()

# 检验特殊方法 __ge__ 的功能
def test_dunder_ge(A, Asp):
    # 使用 np.testing.suppress_warnings 忽略 SparseEfficiencyWarning
    with np.testing.suppress_warnings() as sup:
        sup.filter(sp.sparse.SparseEfficiencyWarning)
        # 断言 (Asp >= Asp).toarray().all() 和 (A >= Asp).all() 成立
        assert (Asp >= Asp).toarray().all()
        assert (A >= Asp).all()

# 检验特殊方法 __iter__ 的功能
def test_dunder_iter(A, Asp):
    # 断言 A 和 Asp 中的每个元素是否相等
    assert all((a == asp).all() for a, asp in zip(A, Asp))
```