# `D:\src\scipysrc\sympy\sympy\utilities\tests\test_iterables.py`

```
# 导入所需模块和函数
from textwrap import dedent
from itertools import islice, product

# 导入 sympy 的核心类和函数
from sympy.core.basic import Basic
from sympy.core.numbers import Integer
from sympy.core.sorting import ordered
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.combinatorial.factorials import factorial
from sympy.matrices.dense import Matrix
from sympy.combinatorics import RGS_enum, RGS_unrank, Permutation
from sympy.utilities.iterables import (
    _partition, _set_partitions, binary_partitions, bracelets, capture,
    cartes, common_prefix, common_suffix, connected_components, dict_merge,
    filter_symbols, flatten, generate_bell, generate_derangements,
    generate_involutions, generate_oriented_forest, group, has_dups, ibin,
    iproduct, kbins, minlex, multiset, multiset_combinations,
    multiset_partitions, multiset_permutations, necklaces, numbered_symbols,
    partitions, permutations, postfixes,
    prefixes, reshape, rotate_left, rotate_right, runs, sift,
    strongly_connected_components, subsets, take, topological_sort, unflatten,
    uniq, variations, ordered_partitions, rotations, is_palindromic, iterable,
    NotIterable, multiset_derangements, signed_permutations,
    sequence_partitions, sequence_partitions_empty)
from sympy.utilities.enumerative import (
    factoring_visitor, multiset_partitions_taocp )

# 导入 sympy 的全局单例 S 和测试函数
from sympy.core.singleton import S
from sympy.testing.pytest import raises, warns_deprecated_sympy

# 定义符号变量
w, x, y, z = symbols('w,x,y,z')

# 测试函数：检验 deprecated_iterables() 函数的行为
def test_deprecated_iterables():
    # 使用 warns_deprecated_sympy() 检查 ordered() 函数的警告信息
    from sympy.utilities.iterables import default_sort_key, ordered
    with warns_deprecated_sympy():
        assert list(ordered([y, x])) == [x, y]
    # 使用 warns_deprecated_sympy() 检查使用 default_sort_key 排序的警告信息
    with warns_deprecated_sympy():
        assert sorted([y, x], key=default_sort_key) == [x, y]

# 测试函数：检验 is_palindromic() 函数的行为
def test_is_palindromic():
    # 测试空字符串是否为回文
    assert is_palindromic('')
    # 测试单个字符是否为回文
    assert is_palindromic('x')
    # 测试对称字符串是否为回文
    assert is_palindromic('xx')
    # 测试非回文字符串
    assert not is_palindromic('xy')
    # 测试带有指定偏移量的回文性检测
    assert is_palindromic('xxyzzyx', 1)
    # 测试不带指定偏移量的非回文字符串
    assert not is_palindromic('xxyzzyx', 2)
    # 测试带有指定偏移量的回文性检测
    assert is_palindromic('xxyzyx', 2, 2 + 3)

# 测试函数：检验 flatten() 函数的行为
def test_flatten():
    # 测试对包含元组的列表进行展平操作
    assert flatten((1, (1,))) == [1, 1]
    # 测试对包含符号变量的列表进行展平操作
    assert flatten((x, (x,))) == [x, x]

    # 定义包含子列表的列表 ls
    ls = [[(-2, -1), (1, 2)], [(0, 0)]]

    # 测试不同层级的展平操作
    assert flatten(ls, levels=0) == ls
    assert flatten(ls, levels=1) == [(-2, -1), (1, 2), (0, 0)]
    assert flatten(ls, levels=2) == [-2, -1, 1, 2, 0, 0]
    assert flatten(ls, levels=3) == [-2, -1, 1, 2, 0, 0]

    # 测试不合法的 levels 参数抛出异常
    raises(ValueError, lambda: flatten(ls, levels=-1))

    # 定义一个自定义的 Basic 类
    class MyOp(Basic):
        pass

    # 测试对包含自定义类实例的列表进行展平操作
    assert flatten([MyOp(x, y), z]) == [MyOp(x, y), z]
    # 测试指定类进行展平操作
    assert flatten([MyOp(x, y), z], cls=MyOp) == [x, y, z]

    # 测试对集合进行展平操作
    assert flatten({1, 11, 2}) == list({1, 11, 2})

# 测试函数：检验 iproduct() 函数的行为
def test_iproduct():
    # 测试空迭代器的笛卡尔积
    assert list(iproduct()) == [()]
    # 测试空列表的笛卡尔积
    assert list(iproduct([])) == []
    # 确保 iproduct 函数生成的结果与预期的元组列表相等
    assert list(iproduct([1,2,3])) == [(1,),(2,),(3,)]
    
    # 确保 iproduct 函数生成的结果与预期的排序后的元组列表相等
    assert sorted(iproduct([1, 2], [3, 4, 5])) == [
        (1,3),(1,4),(1,5),(2,3),(2,4),(2,5)]
    
    # 确保 iproduct 函数生成的结果与预期的排序后的元组列表相等
    assert sorted(iproduct([0,1],[0,1],[0,1])) == [
        (0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)]
    
    # 确保 iproduct 函数生成的结果是可迭代的
    assert iterable(iproduct(S.Integers)) is True
    
    # 确保 iproduct 函数生成的结果是可迭代的
    assert iterable(iproduct(S.Integers, S.Integers)) is True
    
    # 确保元组 (3,) 存在于 iproduct 函数生成的结果中
    assert (3,) in iproduct(S.Integers)
    
    # 确保元组 (4, 5) 存在于 iproduct 函数生成的结果中
    assert (4, 5) in iproduct(S.Integers, S.Integers)
    
    # 确保元组 (1, 2, 3) 存在于 iproduct 函数生成的结果中
    assert (1, 2, 3) in iproduct(S.Integers, S.Integers, S.Integers)
    
    # 生成包含1000个元组的集合，元组来自于 S.Integers 的三个迭代器的笛卡尔积的前1000项
    triples = set(islice(iproduct(S.Integers, S.Integers, S.Integers), 1000))
    for n1, n2, n3 in triples:
        # 确保每个元组中的元素 n1、n2、n3 都是整数类型 Integer
        assert isinstance(n1, Integer)
        assert isinstance(n2, Integer)
        assert isinstance(n3, Integer)
    
    # 遍历 (-2, -1, 0, 1, 2) 三个集合的笛卡尔积，确保每个元组 t 都存在于 iproduct 函数生成的结果中
    for t in set(product(*([range(-2, 3)]*3))):
        assert t in iproduct(S.Integers, S.Integers, S.Integers)
def test_group():
    # 测试空列表的情况，应返回空列表
    assert group([]) == []
    # 测试空列表的情况，但指定 multiple=False 参数，应返回空列表
    assert group([], multiple=False) == []

    # 测试只有一个元素的列表，应返回包含单个子列表的列表
    assert group([1]) == [[1]]
    # 测试只有一个元素的列表，但指定 multiple=False 参数，应返回包含单个元组的列表
    assert group([1], multiple=False) == [(1, 1)]

    # 测试有两个相同元素的列表，应返回包含一个子列表的列表
    assert group([1, 1]) == [[1, 1]]
    # 测试有两个相同元素的列表，但指定 multiple=False 参数，应返回包含一个元组的列表
    assert group([1, 1], multiple=False) == [(1, 2)]

    # 测试有三个相同元素的列表，应返回包含一个子列表的列表
    assert group([1, 1, 1]) == [[1, 1, 1]]
    # 测试有三个相同元素的列表，但指定 multiple=False 参数，应返回包含一个元组的列表
    assert group([1, 1, 1], multiple=False) == [(1, 3)]

    # 测试包含不同元素的列表，应返回每个元素作为单独子列表的列表
    assert group([1, 2, 1]) == [[1], [2], [1]]
    # 测试包含不同元素的列表，但指定 multiple=False 参数，应返回每个元素及其出现次数的元组列表
    assert group([1, 2, 1], multiple=False) == [(1, 1), (2, 1), (1, 1)]

    # 测试包含多个不同元素的列表，应返回按元素连续出现分组的列表
    assert group([1, 1, 2, 2, 2, 1, 3, 3]) == [[1, 1], [2, 2, 2], [1], [3, 3]]
    # 测试包含多个不同元素的列表，但指定 multiple=False 参数，应返回每个元素及其出现次数的元组列表
    assert group([1, 1, 2, 2, 2, 1, 3, 3], multiple=False) == [(1, 2),
                 (2, 3), (1, 1), (3, 2)]


def test_subsets():
    # combinations
    # 测试生成空集的情况
    assert list(subsets([1, 2, 3], 0)) == [()]
    # 测试生成包含一个元素的所有组合
    assert list(subsets([1, 2, 3], 1)) == [(1,), (2,), (3,)]
    # 测试生成包含两个元素的所有组合
    assert list(subsets([1, 2, 3], 2)) == [(1, 2), (1, 3), (2, 3)]
    # 测试生成包含三个元素的所有组合
    assert list(subsets([1, 2, 3], 3)) == [(1, 2, 3)]
    
    l = list(range(4))
    # 测试生成空集的情况，允许元素重复
    assert list(subsets(l, 0, repetition=True)) == [()]
    # 测试生成包含一个元素的所有组合，允许元素重复
    assert list(subsets(l, 1, repetition=True)) == [(0,), (1,), (2,), (3,)]
    # 测试生成包含两个元素的所有组合，允许元素重复
    assert list(subsets(l, 2, repetition=True)) == [(0, 0), (0, 1), (0, 2),
                                                    (0, 3), (1, 1), (1, 2),
                                                    (1, 3), (2, 2), (2, 3),
                                                    (3, 3)]
    # 测试生成包含三个元素的所有组合，允许元素重复
    assert list(subsets(l, 3, repetition=True)) == [(0, 0, 0), (0, 0, 1),
                                                    (0, 0, 2), (0, 0, 3),
                                                    (0, 1, 1), (0, 1, 2),
                                                    (0, 1, 3), (0, 2, 2),
                                                    (0, 2, 3), (0, 3, 3),
                                                    (1, 1, 1), (1, 1, 2),
                                                    (1, 1, 3), (1, 2, 2),
                                                    (1, 2, 3), (1, 3, 3),
                                                    (2, 2, 2), (2, 2, 3),
                                                    (2, 3, 3), (3, 3, 3)]
    # 测试生成包含四个元素的所有组合，允许元素重复
    assert len(list(subsets(l, 4, repetition=True))) == 35

    # 测试生成不允许元素重复的情况下，无法生成指定长度的组合
    assert list(subsets(l[:2], 3, repetition=False)) == []
    # 测试生成不允许元素重复的情况下，生成指定长度的组合
    assert list(subsets(l[:2], 3, repetition=True)) == [(0, 0, 0),
                                                        (0, 0, 1),
                                                        (0, 1, 1),
                                                        (1, 1, 1)]
    # 测试生成包含重复元素的所有组合
    assert list(subsets([1, 2], repetition=True)) == \
        [(), (1,), (2,), (1, 1), (1, 2), (2, 2)]
    # 测试生成不包含重复元素的所有组合
    assert list(subsets([1, 2], repetition=False)) == \
        [(), (1,), (2,), (1, 2)]
    # 测试生成包含两个元素的所有组合
    assert list(subsets([1, 2, 3], 2)) == \
        [(1, 2), (1, 3), (2, 3)]
    # 测试生成包含两个元素的所有组合，允许元素重复
    assert list(subsets([1, 2, 3], 2, repetition=True)) == \
        [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]


def test_variations():
    # permutations
    l = list(range(4))
    # 断言：使用不重复的方式生成长度为 0 的列表的排列组合，预期结果是一个空元组列表
    assert list(variations(l, 0, repetition=False)) == [()]

    # 断言：使用不重复的方式生成长度为 1 的列表的排列组合，预期结果是包含四个元组的列表
    assert list(variations(l, 1, repetition=False)) == [(0,), (1,), (2,), (3,)]

    # 断言：使用不重复的方式生成长度为 2 的列表的排列组合，预期结果是包含十二个元组的列表
    assert list(variations(l, 2, repetition=False)) == [(0, 1), (0, 2), (0, 3),
                                                       (1, 0), (1, 2), (1, 3),
                                                       (2, 0), (2, 1), (2, 3),
                                                       (3, 0), (3, 1), (3, 2)]

    # 断言：使用不重复的方式生成长度为 3 的列表的排列组合，预期结果是包含二十四个元组的列表
    assert list(variations(l, 3, repetition=False)) == [(0, 1, 2), (0, 1, 3), (0, 2, 1), (0, 2, 3),
                                                       (0, 3, 1), (0, 3, 2), (1, 0, 2), (1, 0, 3),
                                                       (1, 2, 0), (1, 2, 3), (1, 3, 0), (1, 3, 2),
                                                       (2, 0, 1), (2, 0, 3), (2, 1, 0), (2, 1, 3),
                                                       (2, 3, 0), (2, 3, 1), (3, 0, 1), (3, 0, 2),
                                                       (3, 1, 0), (3, 1, 2), (3, 2, 0), (3, 2, 1)]

    # 断言：使用重复的方式生成长度为 0 的列表的排列组合，预期结果是一个空元组列表
    assert list(variations(l, 0, repetition=True)) == [()]

    # 断言：使用重复的方式生成长度为 1 的列表的排列组合，预期结果是包含四个元组的列表
    assert list(variations(l, 1, repetition=True)) == [(0,), (1,), (2,), (3,)]

    # 断言：使用重复的方式生成长度为 2 的列表的排列组合，预期结果是包含十六个元组的列表，代码使用了换行来保持可读性
    assert list(variations(l, 2, repetition=True)) == [(0, 0), (0, 1), (0, 2),
                                                       (0, 3), (1, 0), (1, 1),
                                                       (1, 2), (1, 3), (2, 0),
                                                       (2, 1), (2, 2), (2, 3),
                                                       (3, 0), (3, 1), (3, 2),
                                                       (3, 3)]

    # 断言：使用重复的方式生成长度为 3 的列表的排列组合，预期结果是包含 64 个元组的列表
    assert len(list(variations(l, 3, repetition=True))) == 64

    # 断言：使用重复的方式生成长度为 4 的列表的排列组合，预期结果是包含 256 个元组的列表
    assert len(list(variations(l, 4, repetition=True))) == 256

    # 断言：使用不重复的方式生成长度为 3 的列表的排列组合，但列表长度小于 3，预期结果是一个空列表
    assert list(variations(l[:2], 3, repetition=False)) == []

    # 断言：使用重复的方式生成长度为 3 的列表的排列组合，但列表长度小于 3，预期结果是包含八个元组的列表
    assert list(variations(l[:2], 3, repetition=True)) == [
        (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
        (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)
    ]
def test_cartes():
    # 测试 cartesian product 函数 cartes
    assert list(cartes([1, 2], [3, 4, 5])) == \
        [(1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)]
    # 空列表的 cartesian product 应该返回包含一个空元组的列表
    assert list(cartes()) == [()]
    # 只有一个元素的列表的 cartesian product
    assert list(cartes('a')) == [('a',)]
    # 重复两次的单个元素的列表的 cartesian product
    assert list(cartes('a', repeat=2)) == [('a', 'a')]
    # 列表元素为范围内的整数的 cartesian product
    assert list(cartes(list(range(2)))) == [(0,), (1,)]


def test_filter_symbols():
    # 测试 filter_symbols 函数
    s = numbered_symbols()
    # 从 numbered_symbols 生成的符号序列 s 中过滤出指定符号后的前三个结果
    filtered = filter_symbols(s, symbols("x0 x2 x3"))
    assert take(filtered, 3) == list(symbols("x1 x4 x5"))


def test_numbered_symbols():
    # 测试 numbered_symbols 函数
    s = numbered_symbols(cls=Dummy)
    # 检查 numbered_symbols 生成的第一个符号是否为 Dummy 类的实例
    assert isinstance(next(s), Dummy)
    # 测试指定前缀和排除特定符号后的 numbered_symbols 生成结果
    assert next(numbered_symbols('C', start=1, exclude=[symbols('C1')])) == \
        symbols('C2')


def test_sift():
    # 测试 sift 函数
    assert sift(list(range(5)), lambda _: _ % 2) == {1: [1, 3], 0: [0, 2, 4]}
    # 使用符号 x 和 y 测试 sift 函数
    assert sift([x, y], lambda _: _.has(x)) == {False: [y], True: [x]}
    # 使用符号 S.One 测试 sift 函数
    assert sift([S.One], lambda _: _.has(x)) == {False: [1]}
    # 使用二进制标志测试 sift 函数
    assert sift([0, 1, 2, 3], lambda x: x % 2, binary=True) == (
        [1, 3], [0, 2])
    # 使用另一种条件测试 sift 函数
    assert sift([0, 1, 2, 3], lambda x: x % 3 == 1, binary=True) == (
        [1], [0, 2, 3])
    # 测试当二进制标志与不符合条件时，sift 函数抛出异常
    raises(ValueError, lambda:
        sift([0, 1, 2, 3], lambda x: x % 3, binary=True))


def test_take():
    # 测试 take 函数
    X = numbered_symbols()
    # 测试从 numbered_symbols 生成的序列 X 中取出前五个符号
    assert take(X, 5) == list(symbols('x0:5'))
    # 再次测试从 numbered_symbols 生成的序列 X 中取出接下来的五个符号
    assert take(X, 5) == list(symbols('x5:10'))
    # 测试从普通列表中取出前五个元素
    assert take([1, 2, 3, 4, 5], 5) == [1, 2, 3, 4, 5]


def test_dict_merge():
    # 测试 dict_merge 函数
    assert dict_merge({}, {1: x, y: z}) == {1: x, y: z}
    assert dict_merge({1: x, y: z}, {}) == {1: x, y: z}

    assert dict_merge({2: z}, {1: x, y: z}) == {1: x, 2: z, y: z}
    assert dict_merge({1: x, y: z}, {2: z}) == {1: x, 2: z, y: z}

    assert dict_merge({1: y, 2: z}, {1: x, y: z}) == {1: x, 2: z, y: z}
    assert dict_merge({1: x, y: z}, {1: y, 2: z}) == {1: y, 2: z, y: z}


def test_prefixes():
    # 测试 prefixes 函数
    assert list(prefixes([])) == []
    assert list(prefixes([1])) == [[1]]
    assert list(prefixes([1, 2])) == [[1], [1, 2]]

    assert list(prefixes([1, 2, 3, 4, 5])) == \
        [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]]


def test_postfixes():
    # 测试 postfixes 函数
    assert list(postfixes([])) == []
    assert list(postfixes([1])) == [[1]]
    assert list(postfixes([1, 2])) == [[2], [1, 2]]

    assert list(postfixes([1, 2, 3, 4, 5])) == \
        [[5], [4, 5], [3, 4, 5], [2, 3, 4, 5], [1, 2, 3, 4, 5]]


def test_topological_sort():
    V = [2, 3, 5, 7, 8, 9, 10, 11]
    E = [(7, 11), (7, 8), (5, 11),
         (3, 8), (3, 10), (11, 2),
         (11, 9), (11, 10), (8, 9)]
    
    # 测试 topological_sort 函数
    assert topological_sort((V, E)) == [3, 5, 7, 8, 11, 2, 9, 10]
    # 测试带有自定义排序键的 topological_sort 函数
    assert topological_sort((V, E), key=lambda v: -v) == \
        [7, 5, 11, 3, 10, 8, 9, 2]
    # 测试 topological_sort 函数在存在环时抛出异常
    raises(ValueError, lambda: topological_sort((V, E + [(10, 7)])))


def test_strongly_connected_components():
    # 测试 strongly_connected_components 函数
    assert strongly_connected_components(([], [])) == []
    assert strongly_connected_components(([1, 2, 3], [])) == [[1], [2], [3]]

    V = [1, 2, 3]
    E = [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1)]
    # 调用函数并断言其返回值为强连通分量的列表，期望返回 [[1, 2, 3]]
    assert strongly_connected_components((V, E)) == [[1, 2, 3]]
    
    # 定义顶点集合 V 和边集合 E
    V = [1, 2, 3, 4]
    E = [(1, 2), (2, 3), (3, 2), (3, 4)]
    # 断言函数返回值为强连通分量的列表，期望返回 [[4], [2, 3], [1]]
    assert strongly_connected_components((V, E)) == [[4], [2, 3], [1]]
    
    # 重新定义顶点集合 V 和边集合 E
    V = [1, 2, 3, 4]
    E = [(1, 2), (2, 1), (3, 4), (4, 3)]
    # 断言函数返回值为强连通分量的列表，期望返回 [[1, 2], [3, 4]]
    assert strongly_connected_components((V, E)) == [[1, 2], [3, 4]]
# 定义测试函数 test_connected_components，用于测试 connected_components 函数的不同输入情况
def test_connected_components():
    # 测试空图的情况，期望返回空列表
    assert connected_components(([], [])) == []
    # 测试单个顶点的情况，期望返回每个顶点单独作为一个连通分量的列表
    assert connected_components(([1, 2, 3], [])) == [[1], [2], [3]]

    # 测试一个连通图的情况，期望返回所有顶点作为一个连通分量的列表
    V = [1, 2, 3]
    E = [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1)]
    assert connected_components((V, E)) == [[1, 2, 3]]

    # 测试多个连通分量的情况，期望返回每个连通分量作为一个列表的列表
    V = [1, 2, 3, 4]
    E = [(1, 2), (2, 3), (3, 2), (3, 4)]
    assert connected_components((V, E)) == [[1, 2, 3, 4]]

    # 测试不全连通的情况，期望返回多个列表表示各自的连通分量
    V = [1, 2, 3, 4]
    E = [(1, 2), (3, 4)]
    assert connected_components((V, E)) == [[1, 2], [3, 4]]


# 定义测试函数 test_rotate，用于测试 rotate_left 和 rotate_right 函数的不同输入情况
def test_rotate():
    # 测试向左旋转数组的情况，期望返回旋转后的数组
    A = [0, 1, 2, 3, 4]
    assert rotate_left(A, 2) == [2, 3, 4, 0, 1]
    
    # 测试向右旋转数组的情况，期望返回旋转后的数组
    assert rotate_right(A, 1) == [4, 0, 1, 2, 3]
    
    # 测试空数组的情况，期望返回空数组
    A = []
    B = rotate_right(A, 1)
    assert B == []
    
    # 测试旋转后不影响原数组的情况
    B.append(1)
    assert A == []
    
    # 再次测试空数组的情况，期望返回空数组
    B = rotate_left(A, 1)
    assert B == []
    
    # 测试旋转后不影响原数组的情况
    B.append(1)
    assert A == []


# 定义测试函数 test_multiset_partitions，用于测试 multiset_partitions 函数的不同输入情况
def test_multiset_partitions():
    # 测试将一个数组分割为包含单个子数组的列表的情况
    A = [0, 1, 2, 3, 4]
    assert list(multiset_partitions(A, 5)) == [[[0], [1], [2], [3], [4]]]
    
    # 测试将一个数组分割为包含多个子数组的列表的情况
    assert len(list(multiset_partitions(A, 4))) == 10
    assert len(list(multiset_partitions(A, 3))) == 25
    
    # 测试包含重复元素的数组的分割情况
    assert list(multiset_partitions([1, 1, 1, 2, 2], 2)) == [
        [[1, 1, 1, 2], [2]], [[1, 1, 1], [2, 2]], [[1, 1, 2, 2], [1]],
        [[1, 1, 2], [1, 2]], [[1, 1], [1, 2, 2]]]

    assert list(multiset_partitions([1, 1, 2, 2], 2)) == [
        [[1, 1, 2], [2]], [[1, 1], [2, 2]], [[1, 2, 2], [1]],
        [[1, 2], [1, 2]]]

    assert list(multiset_partitions([1, 2, 3, 4], 2)) == [
        [[1, 2, 3], [4]], [[1, 2, 4], [3]], [[1, 2], [3, 4]],
        [[1, 3, 4], [2]], [[1, 3], [2, 4]], [[1, 4], [2, 3]],
        [[1], [2, 3, 4]]]

    assert list(multiset_partitions([1, 2, 2], 2)) == [
        [[1, 2], [2]], [[1], [2, 2]]]

    # 测试整数作为输入的情况
    assert list(multiset_partitions(3)) == [
        [[0, 1, 2]], [[0, 1], [2]], [[0, 2], [1]], [[0], [1, 2]],
        [[0], [1], [2]]]
    assert list(multiset_partitions(3, 2)) == [
        [[0, 1], [2]], [[0, 2], [1]], [[0], [1, 2]]]

    # 测试包含相同元素的列表的情况
    assert list(multiset_partitions([1] * 3, 2)) == [[[1], [1, 1]]]
    assert list(multiset_partitions([1] * 3)) == [
        [[1, 1, 1]], [[1], [1, 1]], [[1], [1], [1]]]

    # 测试排序后结果不变的情况
    a = [3, 2, 1]
    assert list(multiset_partitions(a)) == \
        list(multiset_partitions(sorted(a)))

    # 测试无法分割的情况
    assert list(multiset_partitions(a, 5)) == []

    # 测试分成单个子数组的情况
    assert list(multiset_partitions(a, 1)) == [[[1, 2, 3]]]

    # 测试增加额外元素后无法分割的情况
    assert list(multiset_partitions(a + [4], 5)) == []
    assert list(multiset_partitions(a + [4], 1)) == [[[1, 2, 3, 4]]]

    # 测试整数作为输入的情况
    assert list(multiset_partitions(2, 5)) == []
    assert list(multiset_partitions(2, 1)) == [[[0, 1]]]

    # 测试字符串作为输入的情况
    assert list(multiset_partitions('a')) == [[['a']]]
    assert list(multiset_partitions('a', 2)) == []
    assert list(multiset_partitions('ab')) == [[['a', 'b']], [['a'], ['b']]]
    assert list(multiset_partitions('ab', 1)) == [[['a', 'b']]]
    assert list(multiset_partitions('aaa', 1)) == [['aaa']]
    assert list(multiset_partitions([1, 1], 1)) == [[[1, 1]]]
    ans = [('mpsyy',), ('mpsy', 'y'), ('mps', 'yy'), ('mps', 'y', 'y'),
           ('mpyy', 's'), ('mpy', 'sy'), ('mpy', 's', 'y'), ('mp', 'syy'),
           ('mp', 'sy', 'y'), ('mp', 's', 'yy'), ('mp', 's', 'y', 'y'),
           ('msyy', 'p'), ('msy', 'py'), ('msy', 'p', 'y'), ('ms', 'pyy'),
           ('ms', 'py', 'y'), ('ms', 'p', 'yy'), ('ms', 'p', 'y', 'y'),
           ('myy', 'ps'), ('myy', 'p', 's'), ('my', 'psy'), ('my', 'ps', 'y'),
           ('my', 'py', 's'), ('my', 'p', 'sy'), ('my', 'p', 's', 'y'),
           ('m', 'psyy'), ('m', 'psy', 'y'), ('m', 'ps', 'yy'),
           ('m', 'ps', 'y', 'y'), ('m', 'pyy', 's'), ('m', 'py', 'sy'),
           ('m', 'py', 's', 'y'), ('m', 'p', 'syy'),
           ('m', 'p', 'sy', 'y'), ('m', 'p', 's', 'yy'),
           ('m', 'p', 's', 'y', 'y')]
    # 定义预期的答案，这是一个包含多个元组的列表，每个元组代表一个字符串的分割结果

    assert [tuple("".join(part) for part in p)
                for p in multiset_partitions('sympy')] == ans
    # 使用 multiset_partitions 函数对字符串 'sympy' 进行多重集分割，将分割结果中的每个部分连接成字符串，并与预期的答案 ans 进行断言比较

    factorings = [[24], [8, 3], [12, 2], [4, 6], [4, 2, 3],
                  [6, 2, 2], [2, 2, 2, 3]]
    # 定义预期的因子分解结果，每个子列表代表一个整数的因子分解

    assert [factoring_visitor(p, [2,3]) for
                p in multiset_partitions_taocp([3, 1])] == factorings
    # 使用 multiset_partitions_taocp 函数对列表 [3, 1] 进行多重集分割，对每个分割结果调用 factoring_visitor 函数，然后与预期的因子分解结果 factorings 进行断言比较
def test_multiset_combinations():
    # 预期的输出列表
    ans = ['iii', 'iim', 'iip', 'iis', 'imp', 'ims', 'ipp', 'ips',
           'iss', 'mpp', 'mps', 'mss', 'pps', 'pss', 'sss']
    # 断言：将多重集合 'mississippi' 中长度为 3 的组合转换为字符串列表，并与预期输出比较
    assert [''.join(i) for i in
            list(multiset_combinations('mississippi', 3))] == ans
    # 创建多重集合 M，包含 'mississippi'
    M = multiset('mississippi')
    # 断言：将多重集合 M 中长度为 3 的组合转换为字符串列表，并与预期输出比较
    assert [''.join(i) for i in
            list(multiset_combinations(M, 3))] == ans
    # 断言：尝试从多重集合 M 中获取长度为 30 的组合，预期为空列表
    assert [''.join(i) for i in multiset_combinations(M, 30)] == []
    # 断言：尝试对包含两个子列表的列表获取长度为 2 的组合，预期返回原列表
    assert list(multiset_combinations([[1], [2, 3]], 2)) == [[[1], [2, 3]]]
    # 断言：尝试从单字符 'a' 获取长度为 3 的组合，预期返回空列表
    assert len(list(multiset_combinations('a', 3))) == 0
    # 断言：尝试从单字符 'a' 获取长度为 0 的组合，预期返回包含一个空列表的列表
    assert len(list(multiset_combinations('a', 0))) == 1
    # 断言：尝试从 'abc' 获取长度为 1 的组合，预期返回包含三个子列表的列表
    assert list(multiset_combinations('abc', 1)) == [['a'], ['b'], ['c']]
    # 断言：对不合法的输入（包含负值的字典）进行测试，预期引发 ValueError 异常
    raises(ValueError, lambda: list(multiset_combinations({0: 3, 1: -1}, 2)))


def test_multiset_permutations():
    # 预期的输出列表
    ans = ['abby', 'abyb', 'aybb', 'baby', 'bayb', 'bbay', 'bbya', 'byab',
           'byba', 'yabb', 'ybab', 'ybba']
    # 断言：将多重集合 'baby' 中的排列转换为字符串列表，并与预期输出比较
    assert [''.join(i) for i in multiset_permutations('baby')] == ans
    # 断言：将多重集合 M 中的排列转换为字符串列表，并与预期输出比较
    assert [''.join(i) for i in multiset_permutations(multiset('baby'))] == ans
    # 断言：从多重集合 [0, 0, 0] 中获取长度为 2 的排列，预期返回包含一个子列表的列表
    assert list(multiset_permutations([0, 0, 0], 2)) == [[0, 0]]
    # 断言：从多重集合 [0, 2, 1] 中获取长度为 2 的排列，预期返回包含多个子列表的列表
    assert list(multiset_permutations([0, 2, 1], 2)) == [
        [0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]]
    # 断言：尝试从单字符 'a' 获取长度为 0 的排列，预期返回包含一个空列表的列表
    assert len(list(multiset_permutations('a', 0))) == 1
    # 断言：尝试从单字符 'a' 获取长度为 3 的排列，预期返回空列表
    assert len(list(multiset_permutations('a', 3))) == 0
    # 对空集合、空字典、空字符串等进行排列，预期返回包含一个空列表的列表
    for nul in ([], {}, ''):
        assert list(multiset_permutations(nul)) == [[]]
    # 断言：尝试从空集合获取长度为 0 的排列，预期返回包含一个空列表的列表
    assert list(multiset_permutations(nul, 0)) == [[]]
    # 断言：尝试从空集合获取长度为 1 的排列，预期返回空列表
    assert list(multiset_permutations(nul, 1)) == []
    # 断言：尝试从空集合获取长度为 -1 的排列，预期返回空列表
    assert list(multiset_permutations(nul, -1)) == []

    def test():
        # 对于长度从 1 到 6 的范围进行遍历
        for i in range(1, 7):
            print(i)
            # 打印每个长度的排列
            for p in multiset_permutations([0, 0, 1, 0, 1], i):
                print(p)
    # 断言：捕获 test 函数的输出，与预期的格式化字符串进行比较
    assert capture(lambda: test()) == dedent('''\
        1
        [0]
        [1]
        2
        [0, 0]
        [0, 1]
        [1, 0]
        [1, 1]
        3
        [0, 0, 0]
        [0, 0, 1]
        [0, 1, 0]
        [0, 1, 1]
        [1, 0, 0]
        [1, 0, 1]
        [1, 1, 0]
        4
        [0, 0, 0, 1]
        [0, 0, 1, 0]
        [0, 0, 1, 1]
        [0, 1, 0, 0]
        [0, 1, 0, 1]
        [0, 1, 1, 0]
        [1, 0, 0, 0]
        [1, 0, 0, 1]
        [1, 0, 1, 0]
        [1, 1, 0, 0]
        5
        [0, 0, 0, 1, 1]
        [0, 0, 1, 0, 1]
        [0, 0, 1, 1, 0]
        [0, 1, 0, 0, 1]
        [0, 1, 0, 1, 0]
        [0, 1, 1, 0, 0]
        [1, 0, 0, 0, 1]
        [1, 0, 0, 1, 0]
        [1, 0, 1, 0, 0]
        [1, 1, 0, 0, 0]
        6\n''')
    # 断言：尝试对不合法的输入（包含负值的字典）进行测试，预期引发 ValueError 异常
    raises(ValueError, lambda: list(multiset_permutations({0: 3, 1: -1})))

def test_partitions():
    # 预期的输出列表
    ans = [[{}], [(0, {})]]
    # 对于每个测试案例，验证 partitions 函数的输出是否与预期的答案 ans[i] 相同
    for i in range(2):
        assert list(partitions(0, size=i)) == ans[i]
        assert list(partitions(1, 0, size=i)) == ans[i]
        assert list(partitions(6, 2, 2, size=i)) == ans[i]
        assert list(partitions(6, 2, None, size=i)) != ans[i]
        assert list(partitions(6, None, 2, size=i)) != ans[i]
        assert list(partitions(6, 2, 0, size=i)) == ans[i]

    # 验证 partitions 函数在给定 k 值时的输出是否与预期的分割方案相同
    assert list(partitions(6, k=2)) == [
        {2: 3}, {1: 2, 2: 2}, {1: 4, 2: 1}, {1: 6}]

    # 验证 partitions 函数在给定 k 值时的输出是否与预期的分割方案相同
    assert list(partitions(6, k=3)) == [
        {3: 2}, {1: 1, 2: 1, 3: 1}, {1: 3, 3: 1}, {2: 3}, {1: 2, 2: 2},
        {1: 4, 2: 1}, {1: 6}]

    # 验证 partitions 函数在给定 k 和 m 值时的输出是否与预期的分割方案相同
    assert list(partitions(8, k=4, m=3)) == [
        {4: 2}, {1: 1, 3: 1, 4: 1}, {2: 2, 4: 1}, {2: 1, 3: 2}] == [
        i for i in partitions(8, k=4, m=3) if all(k <= 4 for k in i)
        and sum(i.values()) <=3]

    # 验证 partitions 函数在给定 m 值时的输出是否与预期的分割方案相同
    assert list(partitions(S(3), m=2)) == [
        {3: 1}, {1: 1, 2: 1}]

    # 验证 partitions 函数在给定 k 值时的输出是否与预期的分割方案相同
    assert list(partitions(4, k=3)) == [
        {1: 1, 3: 1}, {2: 2}, {1: 2, 2: 1}, {1: 4}] == [
        i for i in partitions(4) if all(k <= 3 for k in i)]

    # 一致性检查：验证 _partitions 和 RGS_unrank 的输出是否一致
    # 这是对两个函数的健全性测试，也验证了两者输出的分割总数是否相同
    # （来自 pkrathmann2 的建议）
    for n in range(2, 6):
        i  = 0
        # 遍历 _set_partitions 函数返回的分割方式和对应的 RGS_unrank 函数输出进行比较
        for m, q  in _set_partitions(n):
            assert  q == RGS_unrank(i, n)
            i += 1
        # 确保枚举的分割数量与 RGS_enum 函数返回的数量一致
        assert i == RGS_enum(n)
# 测试二进制分割函数的功能
def test_binary_partitions():
    # 断言二进制分割函数对于输入 10 返回的结果是否与预期相同
    assert [i[:] for i in binary_partitions(10)] == [[8, 2], [8, 1, 1],
        [4, 4, 2], [4, 4, 1, 1], [4, 2, 2, 2], [4, 2, 2, 1, 1],
        [4, 2, 1, 1, 1, 1], [4, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2],
        [2, 2, 2, 2, 1, 1], [2, 2, 2, 1, 1, 1, 1], [2, 2, 1, 1, 1, 1, 1, 1],
        [2, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    
    # 断言二进制分割函数对于输入 16 返回的结果长度是否为 36
    assert len([j[:] for j in binary_partitions(16)]) == 36


# 测试 Bell 排列生成函数的功能
def test_bell_perm():
    # 断言生成 Bell 排列对于给定范围内的长度是否与阶乘值相匹配
    assert [len(set(generate_bell(i))) for i in range(1, 7)] == [
        factorial(i) for i in range(1, 7)]
    
    # 断言生成 Bell 排列对于特定输入的结果是否与预期一致
    assert list(generate_bell(3)) == [
        (0, 1, 2), (0, 2, 1), (2, 0, 1), (2, 1, 0), (1, 2, 0), (1, 0, 2)]
    
    # 对于每个 n 在指定范围内，验证生成的 Bell 排列和 Trotter-Johnson 排列是否一致
    for n in range(1, 5):
        p = Permutation(range(n))
        b = generate_bell(n)
        for bi in b:
            assert bi == tuple(p.array_form)
            p = p.next_trotterjohnson()
    
    # 断言当输入为 0 时，生成 Bell 排列会引发 ValueError 异常
    raises(ValueError, lambda: list(generate_bell(0)))  # XXX is this consistent with other permutation algorithms?


# 测试不动点排列生成函数的功能
def test_involutions():
    # 验证生成的不动点排列列表的长度是否与预期相同
    lengths = [1, 2, 4, 10, 26, 76]
    for n, N in enumerate(lengths):
        i = list(generate_involutions(n + 1))
        assert len(i) == N
        # 验证不动点排列的平方是否都相同
        assert len({Permutation(j)**2 for j in i}) == 1


# 测试错位排列生成函数的功能
def test_derangements():
    # 断言生成的错位排列列表的长度是否为 265
    assert len(list(generate_derangements(list(range(6))))) == 265
    
    # 断言生成的错位排列对于字符串 'abcde' 的结果是否与预期一致
    assert ''.join(''.join(i) for i in generate_derangements('abcde')) == (
    'badecbaecdbcaedbcdeabceadbdaecbdeacbdecabeacdbedacbedcacabedcadebcaebd'
    'cdaebcdbeacdeabcdebaceabdcebadcedabcedbadabecdaebcdaecbdcaebdcbeadceab'
    'dcebadeabcdeacbdebacdebcaeabcdeadbceadcbecabdecbadecdabecdbaedabcedacb'
    'edbacedbca')
    
    # 断言生成的错位排列对于列表 [0, 1, 2, 3] 的结果是否与预期一致
    assert list(generate_derangements([0, 1, 2, 3])) == [
        [1, 0, 3, 2], [1, 2, 3, 0], [1, 3, 0, 2], [2, 0, 3, 1],
        [2, 3, 0, 1], [2, 3, 1, 0], [3, 0, 1, 2], [3, 2, 0, 1], [3, 2, 1, 0]]
    
    # 断言生成的错位排列对于列表 [0, 1, 2, 2] 的结果是否与预期一致
    assert list(generate_derangements([0, 1, 2, 2])) == [
        [2, 2, 0, 1], [2, 2, 1, 0]]
    
    # 断言生成的错位排列对于字符串 'ba' 的结果是否与预期一致
    assert list(generate_derangements('ba')) == [list('ab')]
    
    # 测试多重集错位排列函数 multiset_derangements 的功能
    D = multiset_derangements
    assert list(D('abb')) == []
    assert [''.join(i) for i in D('ab')] == ['ba']
    assert [''.join(i) for i in D('abc')] == ['bca', 'cab']
    assert [''.join(i) for i in D('aabb')] == ['bbaa']
    assert [''.join(i) for i in D('aabbcccc')] == [
        'ccccaabb', 'ccccabab', 'ccccabba', 'ccccbaab', 'ccccbaba',
        'ccccbbaa']
    assert [''.join(i) for i in D('aabbccc')] == [
        'cccabba', 'cccabab', 'cccaabb', 'ccacbba', 'ccacbab',
        'ccacabb', 'cbccbaa', 'cbccaba', 'cbccaab', 'bcccbaa',
        'bcccaba', 'bcccaab']
    assert [''.join(i) for i in D('books')] == ['kbsoo', 'ksboo',
        'sbkoo', 'skboo', 'oksbo', 'oskbo', 'okbso', 'obkso', 'oskob',
        'oksob', 'osbok', 'obsok']
    # 断言语句，用于确保调用 generate_derangements 函数生成的结果与预期结果列表相同
    assert list(generate_derangements([[3], [2], [2], [1]])) == [
        # 预期的第一个结果：[[2], [1], [3], [2]]
        [[2], [1], [3], [2]],
        # 预期的第二个结果：[[2], [3], [1], [2]]
        [[2], [3], [1], [2]]
    ]
# 定义测试函数 `test_necklaces`
def test_necklaces():
    # 定义内部函数 `count`，用于计算项链的个数
    def count(n, k, f):
        return len(list(necklaces(n, k, f)))
    
    # 初始化空列表 `m`，用于存储测试结果
    m = []
    # 循环遍历范围为 [1, 8) 的整数
    for i in range(1, 8):
        # 向列表 `m` 中添加元组，元组包含不同计数情况的项链数量
        m.append((
            i, count(i, 2, 0), count(i, 2, 1), count(i, 3, 1)))
    
    # 断言列表 `m` 的矩阵形式等于给定的矩阵
    assert Matrix(m) == Matrix([
        [1,   2,   2,   3],
        [2,   3,   3,   6],
        [3,   4,   4,  10],
        [4,   6,   6,  21],
        [5,   8,   8,  39],
        [6,  14,  13,  92],
        [7,  20,  18, 198]])


# 定义测试函数 `test_bracelets`
def test_bracelets():
    # 生成项链的列表 `bc`
    bc = list(bracelets(2, 4))
    # 断言生成的项链列表 `bc` 的矩阵形式等于给定的矩阵
    assert Matrix(bc) == Matrix([
        [0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 1],
        [1, 2],
        [1, 3],
        [2, 2],
        [2, 3],
        [3, 3]
        ])
    # 再次生成项链的列表 `bc`
    bc = list(bracelets(4, 2))
    # 断言生成的项链列表 `bc` 的矩阵形式等于给定的矩阵
    assert Matrix(bc) == Matrix([
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        [0, 1, 1, 1],
        [1, 1, 1, 1]
    ])


# 定义测试函数 `test_generate_oriented_forest`
def test_generate_oriented_forest():
    # 断言生成的有向森林列表与给定的列表相等
    assert list(generate_oriented_forest(5)) == [[0, 1, 2, 3, 4],
        [0, 1, 2, 3, 3], [0, 1, 2, 3, 2], [0, 1, 2, 3, 1], [0, 1, 2, 3, 0],
        [0, 1, 2, 2, 2], [0, 1, 2, 2, 1], [0, 1, 2, 2, 0], [0, 1, 2, 1, 2],
        [0, 1, 2, 1, 1], [0, 1, 2, 1, 0], [0, 1, 2, 0, 1], [0, 1, 2, 0, 0],
        [0, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 1, 1, 0, 1], [0, 1, 1, 0, 0],
        [0, 1, 0, 1, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]]
    # 断言生成的有向森林列表的长度为给定的值
    assert len(list(generate_oriented_forest(10))) == 1842


# 定义测试函数 `test_unflatten`
def test_unflatten():
    # 创建包含 0 到 9 的列表 `r`
    r = list(range(10))
    # 断言 `unflatten(r)` 返回的结果与预期的结果相等
    assert unflatten(r) == list(zip(r[::2], r[1::2]))
    # 断言 `unflatten(r, 5)` 返回的结果与预期的结果相等
    assert unflatten(r, 5) == [tuple(r[:5]), tuple(r[5:])]
    # 断言 `unflatten(list(range(10)), 3)` 抛出 ValueError 异常
    raises(ValueError, lambda: unflatten(list(range(10)), 3))
    # 断言 `unflatten(list(range(10)), -2)` 抛出 ValueError 异常
    raises(ValueError, lambda: unflatten(list(range(10)), -2))


# 定义测试函数 `test_common_prefix_suffix`
def test_common_prefix_suffix():
    # 断言空列表与 `[1]` 的公共前缀为空列表
    assert common_prefix([], [1]) == []
    # 断言 `[0, 1, 2]` 与其自身的公共前缀为 `[0, 1, 2]`
    assert common_prefix(list(range(3))) == [0, 1, 2]
    # 断言 `[0, 1, 2]` 与 `[0, 1, 2, 3]` 的公共前缀为 `[0, 1, 2]`
    assert common_prefix(list(range(3)), list(range(4))) == [0, 1, 2]
    # 断言 `[1, 2, 3]` 与 `[1, 2, 5]` 的公共前缀为 `[1, 2]`
    assert common_prefix([1, 2, 3], [1, 2, 5]) == [1, 2]
    # 断言 `[1, 2, 3]` 与 `[1, 3, 5]` 的公共前缀为 `[1]`
    assert common_prefix([1, 2, 3], [1, 3, 5]) == [1]

    # 断言空列表与 `[1]` 的公共后缀为空列表
    assert common_suffix([], [1]) == []
    # 断言 `[0, 1, 2]` 与其自身的公共后缀为 `[0, 1, 2]`
    assert common_suffix(list(range(3))) == [0, 1, 2]
    # 断言 `[0, 1, 2]` 与 `[0, 1, 2]` 的公共后缀为 `[0, 1, 2]`
    assert common_suffix(list(range(3)), list(range(3))) == [0, 1, 2]
    # 断言 `[0, 1, 2]` 与 `[0, 1, 2, 3]` 的公共后缀为空列表
    assert common_suffix(list(range(3)), list(range(4))) == []
    # 断言 `[1, 2, 3]` 与 `[9, 2, 3]` 的公共后缀为 `[2, 3]`
    assert common_suffix([1, 2, 3], [9, 2, 3]) == [2, 3]
    # 断言 `[1, 2, 3]` 与 `[9, 7, 3]` 的公共后缀为 `[3]`
    assert common_suffix([1, 2, 3], [9, 7, 3]) == [3]


# 定义测试函数 `test_minlex`
def test_minlex():
    # 断言对 `[1, 2, 0]` 进行最小字典序排序后的结果为 `(0, 1, 2)`
    assert minlex([1, 2, 0]) == (0, 1, 2)
    # 断言对 `(1, 2, 0)` 进行最小字典序排序后的结果为 `(0, 1, 2)`
    assert minlex((1, 2, 0)) == (0, 1, 2)
    # 断言对 `(1, 0,
    # 定义一个包含两个元素的列表和一个元组，用于测试 ordered 函数的排序功能
    seq, keys = [[[1, 2, 1], [0, 3, 1], [1, 1, 3], [2], [1]],
                 (lambda x: len(x), lambda x: sum(x))]
    # 断言使用 ordered 函数对 seq 列表进行排序后的结果是否符合预期
    assert list(ordered(seq, keys, default=False, warn=False)) == \
        [[1], [2], [1, 2, 1], [0, 3, 1], [1, 1, 3]]
    # 使用 ordered 函数时，当 warn 参数为 True 时，应该引发 ValueError 异常
    raises(ValueError, lambda:
           list(ordered(seq, keys, default=False, warn=True)))
# 定义一个函数 test_runs，用于测试 runs 函数的各种情况
def test_runs():
    # 测试空列表的情况，期望返回空列表
    assert runs([]) == []
    # 测试包含单个元素的列表，期望返回包含单个子列表的列表
    assert runs([1]) == [[1]]
    # 测试连续相同元素的情况，期望每个子列表包含相同元素
    assert runs([1, 1]) == [[1], [1]]
    # 测试连续不同元素的情况，期望每个子列表按照元素相邻性划分
    assert runs([1, 1, 2]) == [[1], [1, 2]]
    # 测试不连续但有序的元素，期望根据顺序正确划分子列表
    assert runs([1, 2, 1]) == [[1, 2], [1]]
    # 测试无序元素的情况，期望每个元素单独成为一个子列表
    assert runs([2, 1, 1]) == [[2], [1], [1]]
    # 导入 lt 操作符，用于后续比较函数的运行测试
    from operator import lt
    # 测试使用自定义的比较函数 lt 对列表进行划分的情况
    assert runs([2, 1, 1], lt) == [[2, 1], [1]]


# 定义一个函数 test_reshape，用于测试 reshape 函数的各种情况
def test_reshape():
    # 创建一个序列 seq，包含数字 1 到 8
    seq = list(range(1, 9))
    # 测试将序列 seq 重新整形成包含两个子列表的列表的情况
    assert reshape(seq, [4]) == \
        [[1, 2, 3, 4], [5, 6, 7, 8]]
    # 测试将序列 seq 重新整形成包含两个元组的列表的情况
    assert reshape(seq, (4,)) == \
        [(1, 2, 3, 4), (5, 6, 7, 8)]
    # 测试将序列 seq 重新整形成包含两个元组的列表的情况（通过元组传递维度）
    assert reshape(seq, (2, 2)) == \
        [(1, 2, 3, 4), (5, 6, 7, 8)]
    # 测试将序列 seq 重新整形成包含包含列表的元组的列表的情况
    assert reshape(seq, (2, [2])) == \
        [(1, 2, [3, 4]), (5, 6, [7, 8])]
    # 测试将序列 seq 重新整形成包含包含元组和列表的元组的列表的情况
    assert reshape(seq, ((2,), [2])) == \
        [((1, 2), [3, 4]), ((5, 6), [7, 8])]
    # 测试将序列 seq 重新整形成包含包含列表和元组的列表的情况
    assert reshape(seq, (1, [2], 1)) == \
        [(1, [2, 3], 4), (5, [6, 7], 8)]
    # 测试将序列 seq 重新整形成包含包含列表的元组的元组的情况
    assert reshape(tuple(seq), ([[1], 1, (2,)],)) == \
        (([[1], 2, (3, 4)],), ([[5], 6, (7, 8)],))
    # 测试将序列 seq 重新整形成包含包含列表的元组的元组的情况（通过元组传递维度）
    assert reshape(tuple(seq), ([1], 1, (2,))) == \
        (([1], 2, (3, 4)), ([5], 6, (7, 8)))
    # 测试将序列 seq 重新整形成包含包含数字、集合和元组的列表的情况
    assert reshape(list(range(12)), [2, [3], {2}, (1, (3,), 1)]) == \
        [[0, 1, [2, 3, 4], {5, 6}, (7, (8, 9, 10), 11)]]
    # 测试 reshape 函数对不合法维度的异常处理，期望抛出 ValueError 异常
    raises(ValueError, lambda: reshape([0, 1], [-1]))
    raises(ValueError, lambda: reshape([0, 1], [3]))


# 定义一个函数 test_uniq，用于测试 uniq 函数的各种情况
def test_uniq():
    # 测试对 partitions(4) 生成的所有排列进行去重的情况
    assert list(uniq(p for p in partitions(4))) == \
        [{4: 1}, {1: 1, 3: 1}, {2: 2}, {1: 2, 2: 1}, {1: 4}]
    # 测试对 range(5) 生成的数字取模结果进行去重的情况
    assert list(uniq(x % 2 for x in range(5))) == [0, 1]
    # 测试对单个字符 'a' 进行去重的情况
    assert list(uniq('a')) == ['a']
    # 测试对字符串 'ababc' 进行去重的情况
    assert list(uniq('ababc')) == list('abc')
    # 测试对包含列表的列表进行去重的情况
    assert list(uniq([[1], [2, 1], [1]])) == [[1], [2, 1]]
    # 测试对包含不同类型元素的列表进行去重的情况
    assert list(uniq(permutations(i for i in [[1], 2, 2]))) == \
        [([1], 2, 2), (2, [1], 2), (2, 2, [1])]
    # 测试对包含数字和列表的混合列表进行去重的情况
    assert list(uniq([2, 3, 2, 4, [2], [1], [2], [3], [1]])) == \
        [2, 3, 4, [2], [1], [3]]
    # 定义一个列表 f，包含数字 1
    f = [1]
    # 测试对包含 f 列表的去重操作中出现的 RuntimeError 异常
    raises(RuntimeError, lambda: [f.remove(i) for i in uniq(f)])
    # 重新定义列表 f，包含包含数字 1 的列表
    f = [[1]]
    # 测试对包含 f 列表的去重操作中出现的 RuntimeError 异常
    raises(RuntimeError, lambda: [f.remove(i) for i in uniq(f)])


# 定义一个函数 test_kbins，用于测试 kbins 函数的各种情况
def test_kbins():
    # 测试将字符串 '1123' 划分成长度为 2 的有序子序列的情况
    assert len(list(kbins('1123', 2, ordered=1))) == 24
    # 测试将字符串 '1123' 划分成长度为 2 的有序子序列的情况
    assert len(list(kbins('1123', 2, ordered=11))) == 36
    # 测试将字符串 '1123' 划分成长度为 2 的有序子序列的情况
    assert len(list(kbins('1123', 2, ordered=10))) == 10
    # 测试将字符串 '1123' 划分成长度为 2 的无序子序列的情况
    assert len(list(kbins('1123', 2, ordered=0))) == 5
    # 测试将字符串 '1123' 划分成长度为 2 的无序子序列的情况
    assert len(list(kbins('1123', 2, ordered=None))) == 3

    # 定义一个内部函数 test1，用于测试不同 ordered 值下 kbins 函数的输出
    def test1():
        for orderedval in [None, 0, 1, 10, 11]:
            print('ordered =', ordered
    # 断言测试函数 test1() 的输出是否与预期相符
    assert capture(lambda : test1()) == dedent('''\
        ordered = None
            [[0], [0, 1]]
            [[0, 0], [1]]
        ordered = 0
            [[0, 0], [1]]
            [[0, 1], [0]]
        ordered = 1
            [[0], [0, 1]]
            [[0], [1, 0]]
            [[1], [0, 0]]
        ordered = 10
            [[0, 0], [1]]
            [[1], [0, 0]]
            [[0, 1], [0]]
            [[0], [0, 1]]
        ordered = 11
            [[0], [0, 1]]
            [[0, 0], [1]]
            [[0], [1, 0]]
            [[0, 1], [0]]
            [[1], [0, 0]]
            [[1, 0], [0]]\n''')

    # 定义测试函数 test2()
    def test2():
        # 对于 orderedval 取值分别为 None, 0, 1, 10, 11
        for orderedval in [None, 0, 1, 10, 11]:
            # 打印当前 orderedval 的取值
            print('ordered =', orderedval)
            # 遍历 kbins() 函数返回的结果列表，参数为 list(range(3))，分成2组，使用 ordered=orderedval
            for p in kbins(list(range(3)), 2, ordered=orderedval):
                # 打印每个分组 p
                print('   ', p)
    # 断言捕获测试函数 test2() 的输出是否与预期相符
    assert capture(lambda : test2()) == dedent('''\
        ordered = None
            [[0], [1, 2]]
            [[0, 1], [2]]
        ordered = 0
            [[0, 1], [2]]
            [[0, 2], [1]]
            [[0], [1, 2]]
        ordered = 1
            [[0], [1, 2]]
            [[0], [2, 1]]
            [[1], [0, 2]]
            [[1], [2, 0]]
            [[2], [0, 1]]
            [[2], [1, 0]]
        ordered = 10
            [[0, 1], [2]]
            [[2], [0, 1]]
            [[0, 2], [1]]
            [[1], [0, 2]]
            [[0], [1, 2]]
            [[1, 2], [0]]
        ordered = 11
            [[0], [1, 2]]
            [[0, 1], [2]]
            [[0], [2, 1]]
            [[0, 2], [1]]
            [[1], [0, 2]]
            [[1, 0], [2]]
            [[1], [2, 0]]
            [[1, 2], [0]]
            [[2], [0, 1]]
            [[2, 0], [1]]
            [[2], [1, 0]]
            [[2, 1], [0]]\n''')
# 定义测试函数 test_has_dups，用于测试 has_dups 函数的不同输入情况
def test_has_dups():
    # 断言空集合没有重复项，应返回 False
    assert has_dups(set()) is False
    # 断言列表 [0, 1, 2] 没有重复项，应返回 False
    assert has_dups(list(range(3))) is False
    # 断言列表 [1, 2, 1] 存在重复项，应返回 True
    assert has_dups([1, 2, 1]) is True
    # 断言包含嵌套列表 [[1], [1]]，其中存在重复项，应返回 True
    assert has_dups([[1], [1]]) is True
    # 断言包含嵌套列表 [[1], [2]]，其中不存在重复项，应返回 False
    assert has_dups([[1], [2]]) is False


# 定义测试函数 test__partition，用于测试 _partition 函数的不同输入情况
def test__partition():
    # 断言 _partition('abcde', [1, 0, 1, 2, 0]) 的结果为 [['b', 'e'], ['a', 'c'], ['d']]
    assert _partition('abcde', [1, 0, 1, 2, 0]) == [
        ['b', 'e'], ['a', 'c'], ['d']]
    # 断言 _partition('abcde', [1, 0, 1, 2, 0], 3) 的结果同上
    assert _partition('abcde', [1, 0, 1, 2, 0], 3) == [
        ['b', 'e'], ['a', 'c'], ['d']]
    # 定义 output 为元组 (3, [1, 0, 1, 2, 0])
    output = (3, [1, 0, 1, 2, 0])
    # 断言 _partition('abcde', *output) 的结果同上
    assert _partition('abcde', *output) == [['b', 'e'], ['a', 'c'], ['d']]


# 定义测试函数 test_ordered_partitions，用于测试 ordered_partitions 函数的不同输入情况
def test_ordered_partitions():
    # 导入 sympy 库中的 nT 函数
    from sympy.functions.combinatorial.numbers import nT
    # 将 ordered_partitions 函数赋给 f
    f = ordered_partitions
    # 断言 list(f(0, 1)) 的结果为 [[]]
    assert list(f(0, 1)) == [[]]
    # 断言 list(f(1, 0)) 的结果为 [[]]
    assert list(f(1, 0)) == [[]]
    # 对于范围在 1 到 6 的每个值 i
    for i in range(1, 7):
        # 对于 j 为 None 和从 1 到 i 的每个值的情况
        for j in [None] + list(range(1, i)):
            # 断言使用 nT(i, j) 计算的结果
            assert (
                sum(1 for p in f(i, j, 1)) ==
                sum(1 for p in f(i, j, 0)) ==
                nT(i, j))


# 定义测试函数 test_rotations，用于测试 rotations 函数的不同输入情况
def test_rotations():
    # 断言 list(rotations('ab')) 的结果为 [['a', 'b'], ['b', 'a']]
    assert list(rotations('ab')) == [['a', 'b'], ['b', 'a']]
    # 断言 list(rotations(range(3))) 的结果为 [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    assert list(rotations(range(3))) == [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    # 断言 list(rotations(range(3), dir=-1)) 的结果为 [[0, 1, 2], [2, 0, 1], [1, 2, 0]]
    assert list(rotations(range(3), dir=-1)) == [[0, 1, 2], [2, 0, 1], [1, 2, 0]]


# 定义测试函数 test_ibin，用于测试 ibin 函数的不同输入情况
def test_ibin():
    # 断言 ibin(3) 的结果为 [1, 1]
    assert ibin(3) == [1, 1]
    # 断言 ibin(3, 3) 的结果为 [0, 1, 1]
    assert ibin(3, 3) == [0, 1, 1]
    # 断言 ibin(3, str=True) 的结果为 '11'
    assert ibin(3, str=True) == '11'
    # 断言 ibin(3, 3, str=True) 的结果为 '011'
    assert ibin(3, 3, str=True) == '011'
    # 断言 list(ibin(2, 'all')) 的结果为 [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert list(ibin(2, 'all')) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    # 断言 list(ibin(2, '', str=True)) 的结果为 ['00', '01', '10', '11']
    assert list(ibin(2, '', str=True)) == ['00', '01', '10', '11']
    # 断言调用 ibin(-0.5) 会引发 ValueError 异常
    raises(ValueError, lambda: ibin(-.5))
    # 断言调用 ibin(2, 1) 会引发 ValueError 异常
    raises(ValueError, lambda: ibin(2, 1))


# 定义测试函数 test_iterable，用于测试 iterable 函数的不同输入情况
def test_iterable():
    # 断言 iterable(0) 的结果为 False
    assert iterable(0) is False
    # 断言 iterable(1) 的结果为 False
    assert iterable(1) is False
    # 断言 iterable(None) 的结果为 False
    assert iterable(None) is False

    # 定义 Test1 类，继承自 NotIterable
    class Test1(NotIterable):
        pass

    # 断言 iterable(Test1()) 的结果为 False
    assert iterable(Test1()) is False

    # 定义 Test2 类，继承自 NotIterable，且 _iterable = True
    class Test2(NotIterable):
        _iterable = True

    # 断言 iterable(Test2()) 的结果为 True
    assert iterable(Test2()) is True

    # 定义 Test3 类
    class Test3:
        pass

    # 断言 iterable(Test3()) 的结果为 False
    assert iterable(Test3()) is False

    # 定义 Test4 类，其中 _iterable = True
    class Test4:
        _iterable = True

    # 断言 iterable(Test4()) 的结果为 True
    assert iterable(Test4()) is True

    # 定义 Test5 类，实现 __iter__ 方法
    class Test5:
        def __iter__(self):
            yield 1

    # 断言 iterable(Test5()) 的结果为 True
    assert iterable(Test5()) is True

    # 定义 Test6 类，继承自 Test5，且 _iterable = False
    class Test6(Test5):
        _iterable = False

    # 断言 iterable(Test6()) 的结果为 False
    assert iterable(Test6()) is False


# 定义测试函数 test_sequence_partitions，用于测试 sequence_partitions 函数的不同输入情况
def test_sequence_partitions():
    # 断言 list(sequence_partitions([1], 1)) 的结果为 [[[1]]]
    assert list(sequence_partitions([1], 1)) == [[[1]]]
    # 断言 list(sequence_partitions([1, 2], 1)) 的结果为 [[[1, 2]]]
    assert list(sequence_partitions([1, 2], 1)) == [[[1, 2]]]
    # 断言 list(sequence_partitions([1, 2], 2)) 的结果为 [[[1], [2]]]
    assert list(sequence_partitions([1, 2], 2)) == [[[1], [2]]]
    # 断言 list(sequence_partitions([1, 2, 3], 1)) 的结果为 [[[1, 2, 3]]]
    assert list(sequence_partitions([1,
    # 使用 assert 语句来验证函数 sequence_partitions_empty 的返回结果是否符合预期
    assert list(sequence_partitions_empty([], 2)) == [[[], []]]
    assert list(sequence_partitions_empty([], 3)) == [[[], [], []]]
    assert list(sequence_partitions_empty([1], 1)) == [[[1]]]
    assert list(sequence_partitions_empty([1], 2)) == [[[], [1]], [[1], []]]
    assert list(sequence_partitions_empty([1], 3)) == \
        [[[], [], [1]], [[], [1], []], [[1], [], []]]
    assert list(sequence_partitions_empty([1, 2], 1)) == [[[1, 2]]]
    assert list(sequence_partitions_empty([1, 2], 2)) == \
        [[[], [1, 2]], [[1], [2]], [[1, 2], []]]
    assert list(sequence_partitions_empty([1, 2], 3)) == [
        [[], [], [1, 2]], [[], [1], [2]], [[], [1, 2], []],
        [[1], [], [2]], [[1], [2], []], [[1, 2], [], []]
    ]
    assert list(sequence_partitions_empty([1, 2, 3], 1)) == [[[1, 2, 3]]]
    assert list(sequence_partitions_empty([1, 2, 3], 2)) == \
        [[[], [1, 2, 3]], [[1], [2, 3]], [[1, 2], [3]], [[1, 2, 3], []]]
    assert list(sequence_partitions_empty([1, 2, 3], 3)) == [
        [[], [], [1, 2, 3]], [[], [1], [2, 3]],
        [[], [1, 2], [3]], [[], [1, 2, 3], []],
        [[1], [], [2, 3]], [[1], [2], [3]],
        [[1], [2, 3], []], [[1, 2], [], [3]],
        [[1, 2], [3], []], [[1, 2, 3], [], []]
    ]

    # 针对异常情况的验证
    assert list(sequence_partitions([], 0)) == []
    assert list(sequence_partitions([1], 0)) == []
    assert list(sequence_partitions([1, 2], 0)) == []
# 定义测试函数，用于测试 signed_permutations 函数的功能
def test_signed_permutations():
    # 预期的答案列表，包含各种符号组合的元组
    ans = [(0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1),
           (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
           (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0)]
    # 断言：调用 signed_permutations 函数生成的结果列表应与预期答案 ans 相同
    assert list(signed_permutations((0, 1, 1))) == ans
    # 断言：调用 signed_permutations 函数生成的结果列表应与预期答案 ans 相同
    assert list(signed_permutations((1, 0, 1))) == ans
    # 断言：调用 signed_permutations 函数生成的结果列表应与预期答案 ans 相同
    assert list(signed_permutations((1, 1, 0))) == ans
```