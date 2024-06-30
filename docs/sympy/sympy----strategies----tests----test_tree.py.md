# `D:\src\scipysrc\sympy\sympy\strategies\tests\test_tree.py`

```
from sympy.strategies.tree import treeapply, greedy, allresults, brute  # 导入必要的函数和类
from functools import partial, reduce  # 导入 functools 库中的 partial 和 reduce 函数


def inc(x):
    return x + 1  # 返回参数 x 加 1 的结果


def dec(x):
    return x - 1  # 返回参数 x 减 1 的结果


def double(x):
    return 2 * x  # 返回参数 x 的两倍


def square(x):
    return x**2  # 返回参数 x 的平方


def add(*args):
    return sum(args)  # 返回所有参数的和


def mul(*args):
    return reduce(lambda a, b: a * b, args, 1)  # 返回所有参数的乘积，使用 reduce 函数实现


def test_treeapply():
    tree = ([3, 3], [4, 1], 2)  # 定义一个测试树结构
    assert treeapply(tree, {list: min, tuple: max}) == 3  # 应用 treeapply 函数，对树中的列表使用 min 函数，对元组使用 max 函数，返回结果应为 3
    assert treeapply(tree, {list: add, tuple: mul}) == 60  # 应用 treeapply 函数，对树中的列表使用 add 函数，对元组使用 mul 函数，返回结果应为 60


def test_treeapply_leaf():
    assert treeapply(3, {}, leaf=lambda x: x**2) == 9  # 应用 treeapply 函数，输入为 3，leaf 函数为对输入平方的 lambda 函数，返回结果应为 9
    tree = ([3, 3], [4, 1], 2)  # 定义一个测试树结构
    treep1 = ([4, 4], [5, 2], 3)  # 定义一个测试树结构
    assert treeapply(tree, {list: min, tuple: max}, leaf=lambda x: x + 1) == \
           treeapply(treep1, {list: min, tuple: max})  # 应用 treeapply 函数，对树中的列表使用 min 函数，对元组使用 max 函数，leaf 函数为对输入加 1 的 lambda 函数，返回结果与 treep1 的应用结果相同


def test_treeapply_strategies():
    from sympy.strategies import chain, minimize  # 导入 chain 和 minimize 函数
    join = {list: chain, tuple: minimize}  # 定义一个字典，列表使用 chain 函数，元组使用 minimize 函数

    assert treeapply(inc, join) == inc  # 应用 treeapply 函数，使用 join 字典，返回结果应为 inc 函数本身
    assert treeapply((inc, dec), join)(5) == minimize(inc, dec)(5)  # 应用 treeapply 函数，对树中的元组使用 join 字典，返回结果应与 minimize(inc, dec)(5) 相同
    assert treeapply([inc, dec], join)(5) == chain(inc, dec)(5)  # 应用 treeapply 函数，对树中的列表使用 join 字典，返回结果应与 chain(inc, dec)(5) 相同
    tree = (inc, [dec, double])  # 定义一个测试树结构，包含 inc 函数和包含 dec 和 double 函数的列表
    assert treeapply(tree, join)(5) == 6  # 应用 treeapply 函数，使用 join 字典，返回结果应为 6
    assert treeapply(tree, join)(1) == 0  # 应用 treeapply 函数，使用 join 字典，返回结果应为 0

    maximize = partial(minimize, objective=lambda x: -x)  # 使用 partial 函数创建 maximize 函数，对 objective 函数取负数
    join = {list: chain, tuple: maximize}  # 更新 join 字典，列表使用 chain 函数，元组使用 maximize 函数
    fn = treeapply(tree, join)  # 应用 treeapply 函数，使用更新后的 join 字典，返回结果赋给 fn
    assert fn(4) == 6  # fn(4) 的结果应为 6，最大值来自于 dec 然后 double
    assert fn(1) == 2  # fn(1) 的结果应为 2，最大值来自于 inc


def test_greedy():
    tree = [inc, (dec, double)]  # 定义一个测试树结构，包含 inc 函数和包含 dec 和 double 函数的元组

    fn = greedy(tree, objective=lambda x: -x)  # 使用 greedy 函数，目标函数为对输入取负数
    assert fn(4) == 6  # fn(4) 的结果应为 6，最大值来自于 dec 然后 double
    assert fn(1) == 2  # fn(1) 的结果应为 2，最大值来自于 inc

    tree = [inc, dec, [inc, dec, [(inc, inc), (dec, dec)]]]  # 定义一个测试树结构
    lowest = greedy(tree)  # 使用 greedy 函数，默认目标为最小值
    assert lowest(10) == 8  # lowest(10) 的结果应为 8

    highest = greedy(tree, objective=lambda x: -x)  # 使用 greedy 函数，目标函数为对输入取负数
    assert highest(10) == 12  # highest(10) 的结果应为 12


def test_allresults():
    # square = lambda x: x**2  # 定义一个 lambda 函数 square，计算参数 x 的平方

    assert set(allresults(inc)(3)) == {inc(3)}  # 应用 allresults 函数，输入为 inc 函数，返回结果应为集合 {inc(3)}
    assert set(allresults([inc, dec])(3)) == {2, 4}  # 应用 allresults 函数，输入为包含 inc 和 dec 函数的列表，返回结果应为集合 {2, 4}
    assert set(allresults((inc, dec))(3)) == {3}  # 应用 allresults 函数，输入为包含 inc 和 dec 函数的元组，返回结果应为集合 {3}
    assert set(allresults([inc, (dec, double)])(4)) == {5, 6}  # 应用 allresults 函数，输入为包含 inc 函数和包含 dec 和 double 函数的列表，返回结果应为集合 {5, 6}


def test_brute():
    tree = ([inc, dec], square)  # 定义一个测试树结构，包含 inc 和 dec 函数的列表以及 square 函数
    fn = brute(tree, lambda x: -x)  # 使用 brute 函数，目标函数为对输入取负数

    assert fn(2) == (2 + 1)**2  # fn(2) 的结果应为 3 的平方
    assert fn(-2) == (-2 - 1)**2  # fn(-2) 的结果应为 -3 的平方

    assert brute(inc)(1) == 2  # 应用 brute 函数，输入为 inc 函数，返回结果应为 2
```