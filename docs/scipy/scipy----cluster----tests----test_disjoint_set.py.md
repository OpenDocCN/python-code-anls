# `D:\src\scipysrc\scipy\scipy\cluster\tests\test_disjoint_set.py`

```
import pytest  # 导入 pytest 库，用于编写和运行测试
from pytest import raises as assert_raises  # 导入 pytest 库中的 raises 函数并重命名为 assert_raises
import numpy as np  # 导入 NumPy 库并重命名为 np，用于数值计算
from scipy.cluster.hierarchy import DisjointSet  # 导入 SciPy 库中的 DisjointSet 类，用于并查集操作
import string  # 导入 Python 内置的 string 模块，用于处理字符串


def generate_random_token():
    k = len(string.ascii_letters)  # 获取字母表长度
    tokens = list(np.arange(k, dtype=int))  # 创建一个包含整数的列表
    tokens += list(np.arange(k, dtype=float))  # 将浮点数列表拼接到 tokens
    tokens += list(string.ascii_letters)  # 将字母列表拼接到 tokens
    tokens += [None for i in range(k)]  # 添加 k 个 None 到 tokens 中
    tokens = np.array(tokens, dtype=object)  # 将 tokens 转换为 NumPy 数组，类型为 object
    rng = np.random.RandomState(seed=0)  # 创建随机数生成器对象 rng，种子为 0

    while 1:
        size = rng.randint(1, 3)  # 生成一个 1 到 2 之间的随机整数 size
        element = rng.choice(tokens, size)  # 从 tokens 中随机选择 size 个元素
        if size == 1:
            yield element[0]  # 如果 size 为 1，则生成单个元素
        else:
            yield tuple(element)  # 如果 size 大于 1，则生成元组


def get_elements(n):
    # 使用 generate_random_token 生成元素，返回包含 n 个元素的列表
    elements = {}
    for element in generate_random_token():
        if element not in elements:
            elements[element] = len(elements)
            if len(elements) >= n:
                break
    return list(elements.keys())  # 返回元素列表的键（即元素）


def test_init():
    n = 10
    elements = get_elements(n)  # 获取 n 个随机生成的元素
    dis = DisjointSet(elements)  # 创建并查集对象 dis，初始化时包含 elements 中的元素
    assert dis.n_subsets == n  # 断言并查集中的子集数量为 n
    assert list(dis) == elements  # 断言并查集的所有元素与 elements 相同


def test_len():
    n = 10
    elements = get_elements(n)  # 获取 n 个随机生成的元素
    dis = DisjointSet(elements)  # 创建并查集对象 dis，初始化时包含 elements 中的元素
    assert len(dis) == n  # 断言并查集的长度为 n

    dis.add("dummy")  # 向并查集中添加一个新元素 "dummy"
    assert len(dis) == n + 1  # 断言并查集的长度为 n+1


@pytest.mark.parametrize("n", [10, 100])
def test_contains(n):
    elements = get_elements(n)  # 获取 n 个随机生成的元素
    dis = DisjointSet(elements)  # 创建并查集对象 dis，初始化时包含 elements 中的元素
    for x in elements:
        assert x in dis  # 断言元素 x 在并查集中

    assert "dummy" not in dis  # 断言 "dummy" 不在并查集中


@pytest.mark.parametrize("n", [10, 100])
def test_add(n):
    elements = get_elements(n)  # 获取 n 个随机生成的元素
    dis1 = DisjointSet(elements)  # 创建并查集对象 dis1，初始化时包含 elements 中的元素

    dis2 = DisjointSet()  # 创建一个空的并查集对象 dis2
    for i, x in enumerate(elements):
        dis2.add(x)  # 向 dis2 中添加元素 x
        assert len(dis2) == i + 1  # 断言 dis2 的长度为 i+1

        # 再次添加相同的元素 x，测试幂等性
        dis2.add(x)
        assert len(dis2) == i + 1  # 断言 dis2 的长度仍为 i+1

    assert list(dis1) == list(dis2)  # 断言 dis1 和 dis2 中的元素相同


def test_element_not_present():
    elements = get_elements(n=10)  # 获取 10 个随机生成的元素
    dis = DisjointSet(elements)  # 创建并查集对象 dis，初始化时包含 elements 中的元素

    with assert_raises(KeyError):
        dis["dummy"]  # 断言访问不存在的元素会引发 KeyError

    with assert_raises(KeyError):
        dis.merge(elements[0], "dummy")  # 断言合并一个不存在的元素会引发 KeyError

    with assert_raises(KeyError):
        dis.connected(elements[0], "dummy")  # 断言检查两个不存在的元素是否连接会引发 KeyError


@pytest.mark.parametrize("direction", ["forwards", "backwards"])
@pytest.mark.parametrize("n", [10, 100])
def test_linear_union_sequence(n, direction):
    elements = get_elements(n)  # 获取 n 个随机生成的元素
    dis = DisjointSet(elements)  # 创建并查集对象 dis，初始化时包含 elements 中的元素
    assert elements == list(dis)  # 断言并查集中的元素与 elements 相同

    indices = list(range(n - 1))
    if direction == "backwards":
        indices = indices[::-1]

    for it, i in enumerate(indices):
        assert not dis.connected(elements[i], elements[i + 1])  # 断言元素 i 和元素 i+1 未连接
        assert dis.merge(elements[i], elements[i + 1])  # 合并元素 i 和元素 i+1，并断言合并成功
        assert dis.connected(elements[i], elements[i + 1])  # 断言元素 i 和元素 i+1 已连接
        assert dis.n_subsets == n - 1 - it  # 断言并查集中子集的数量减少了 1

    roots = [dis[i] for i in elements]
    if direction == "forwards":
        assert all(elements[0] == r for r in roots)  # 断言所有元素的根为第一个元素
    else:
        assert all(elements[-2] == r for r in roots)  # 断言所有元素的根为倒数第二个元素
    # 断言：验证 `dis.merge` 函数返回值为假值（False、None、空字符串等）。
    # `dis.merge(elements[0], elements[-1])` 尝试合并列表 `elements` 的第一个和最后一个元素。
    assert not dis.merge(elements[0], elements[-1])
# 使用 pytest 模块的 mark.parametrize 装饰器，为 test_self_unions 函数参数化，参数为 10 和 100
@pytest.mark.parametrize("n", [10, 100])
def test_self_unions(n):
    # 获取 n 个元素
    elements = get_elements(n)
    # 创建并初始化并查集对象
    dis = DisjointSet(elements)

    # 遍历所有元素
    for x in elements:
        # 检查每个元素是否与自身连接
        assert dis.connected(x, x)
        # 尝试将每个元素与自身合并，预期不会合并成功
        assert not dis.merge(x, x)
        # 再次检查每个元素是否与自身连接
        assert dis.connected(x, x)
    
    # 断言并查集的子集数量与元素数量相同
    assert dis.n_subsets == len(elements)

    # 断言元素列表与并查集迭代器生成的列表相同
    assert elements == list(dis)
    # 获取并查集中每个元素的根节点列表
    roots = [dis[x] for x in elements]
    # 断言元素列表与根节点列表相同
    assert elements == roots


# 使用 pytest 模块的 mark.parametrize 装饰器，为 test_equal_size_ordering 函数参数化，参数为 "ab" 和 "ba" 以及 10 和 100
@pytest.mark.parametrize("order", ["ab", "ba"])
@pytest.mark.parametrize("n", [10, 100])
def test_equal_size_ordering(n, order):
    # 获取 n 个元素
    elements = get_elements(n)
    # 创建并初始化并查集对象
    dis = DisjointSet(elements)

    # 使用随机数生成器创建随机状态对象
    rng = np.random.RandomState(seed=0)
    # 生成索引数组并随机打乱
    indices = np.arange(n)
    rng.shuffle(indices)

    # 遍历索引数组的一半
    for i in range(0, len(indices), 2):
        # 获取要合并的两个元素 a 和 b
        a, b = elements[indices[i]], elements[indices[i + 1]]
        # 根据 order 参数决定以 a 和 b 的顺序合并或者以 b 和 a 的顺序合并
        if order == "ab":
            assert dis.merge(a, b)
        else:
            assert dis.merge(b, a)

        # 确定合并后 a 和 b 的根节点应为较小索引对应的元素
        expected = elements[min(indices[i], indices[i + 1])]
        assert dis[a] == expected
        assert dis[b] == expected


# 使用 pytest 模块的 mark.parametrize 装饰器，为 test_binary_tree 函数参数化，参数为 5 和 10
@pytest.mark.parametrize("kmax", [5, 10])
def test_binary_tree(kmax):
    # 计算元素数量
    n = 2**kmax
    # 获取 n 个元素
    elements = get_elements(n)
    # 创建并初始化并查集对象
    dis = DisjointSet(elements)
    # 使用随机数生成器创建随机状态对象
    rng = np.random.RandomState(seed=0)

    # 遍历 k 的指数幂
    for k in 2**np.arange(kmax):
        # 遍历元素的一半
        for i in range(0, n, 2 * k):
            # 从每个块中随机选择两个元素
            r1, r2 = rng.randint(0, k, size=2)
            a, b = elements[i + r1], elements[i + k + r2]
            # 断言 a 和 b 不连接
            assert not dis.connected(a, b)
            # 尝试将 a 和 b 合并，预期会成功合并
            assert dis.merge(a, b)
            # 断言合并后 a 和 b 连接
            assert dis.connected(a, b)

        # 断言并查集迭代器生成的列表与元素列表相同
        assert elements == list(dis)
        # 获取并查集中每个元素的根节点列表
        roots = [dis[i] for i in elements]
        # 计算预期的根节点列表
        expected_indices = np.arange(n) - np.arange(n) % (2 * k)
        expected = [elements[i] for i in expected_indices]
        # 断言根节点列表与预期列表相同
        assert roots == expected


# 使用 pytest 模块的 mark.parametrize 装饰器，为 test_subsets 函数参数化，参数为 10 和 100
@pytest.mark.parametrize("n", [10, 100])
def test_subsets(n):
    # 获取 n 个元素
    elements = get_elements(n)
    # 创建并初始化并查集对象
    dis = DisjointSet(elements)

    # 使用随机数生成器创建随机状态对象
    rng = np.random.RandomState(seed=0)
    # 遍历生成的随机索引对
    for i, j in rng.randint(0, n, (n, 2)):
        # 获取元素 x 和 y
        x = elements[i]
        y = elements[j]

        # 计算预期的 x 元素所属的子集
        expected = {element for element in dis if {dis[element]} == {dis[x]}}
        # 断言并查集中 x 元素所属的子集大小与预期相同
        assert dis.subset_size(x) == len(dis.subset(x))
        # 断言并查集中 x 元素所属的子集与预期相同
        assert expected == dis.subset(x)

        # 计算预期的每个根节点及其对应的子集
        expected = {dis[element]: set() for element in dis}
        for element in dis:
            expected[dis[element]].add(element)
        expected = list(expected.values())
        # 断言并查集中的所有子集与预期相同
        assert expected == dis.subsets()

        # 合并 x 和 y 元素
        dis.merge(x, y)
        # 断言并查集中 x 和 y 元素所属的子集相同
        assert dis.subset(x) == dis.subset(y)
```