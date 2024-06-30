# `D:\src\scipysrc\sympy\sympy\core\tests\test_traversal.py`

```
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import symbols
from sympy.core.singleton import S
from sympy.core.function import expand, Function
from sympy.core.numbers import I
from sympy.integrals.integrals import Integral
from sympy.polys.polytools import factor
from sympy.core.traversal import preorder_traversal, use, postorder_traversal, iterargs, iterfreeargs
from sympy.functions.elementary.piecewise import ExprCondPair, Piecewise
from sympy.testing.pytest import warns_deprecated_sympy
from sympy.utilities.iterables import capture

# 创建几个 Basic 类的实例
b1 = Basic()
b2 = Basic(b1)
b3 = Basic(b2)
b21 = Basic(b2, b1)

# 定义测试函数 test_preorder_traversal，测试 preorder_traversal 函数
def test_preorder_traversal():
    # 创建一个表达式 expr，包含 b21 和 b3 两个 Basic 类的实例
    expr = Basic(b21, b3)
    # 断言预期的 preorder_traversal 结果与实际结果相符
    assert list(preorder_traversal(expr)) == [expr, b21, b2, b1, b1, b3, b2, b1]
    # 断言对元组的 preorder_traversal 结果与实际结果相符
    assert list(preorder_traversal(('abc', ('d', 'ef')))) == [('abc', ('d', 'ef')), 'abc', ('d', 'ef'), 'd', 'ef']

    # 创建空列表 result，并进行遍历
    result = []
    pt = preorder_traversal(expr)
    for i in pt:
        result.append(i)
        # 如果当前遍历到 b2，则调用 pt.skip() 跳过 b2 的子节点遍历
        if i == b2:
            pt.skip()
    # 断言遍历结果与预期结果相符
    assert result == [expr, b21, b2, b1, b3, b2]

    # 定义符号变量 w, x, y, z
    w, x, y, z = symbols('w:z')
    # 创建表达式 expr
    expr = z + w*(x + y)
    # 使用默认排序键对表达式进行 preorder_traversal
    assert list(preorder_traversal([expr], keys=default_sort_key)) == [[w*(x + y) + z], w*(x + y) + z, z, w*(x + y), w, x + y, x, y]
    # 使用默认键对表达式进行 preorder_traversal
    assert list(preorder_traversal((x + y)*z, keys=True)) == [z*(x + y), z, x + y, x, y]


# 定义测试函数 test_use，测试 use 函数
def test_use():
    # 定义符号变量 x, y
    x, y = symbols('x y')

    # 断言 use 函数的使用情况
    assert use(0, expand) == 0

    # 定义表达式 f
    f = (x + y)**2*x + 1

    # 使用 expand 函数，level=0 的情况
    assert use(f, expand, level=0) == x**3 + 2*x**2*y + x*y**2 + 1
    # 使用 expand 函数，level=1 的情况
    assert use(f, expand, level=1) == x**3 + 2*x**2*y + x*y**2 + 1
    # 使用 expand 函数，level=2 的情况
    assert use(f, expand, level=2) == 1 + x*(2*x*y + x**2 + y**2)
    # 使用 expand 函数，level=3 的情况
    assert use(f, expand, level=3) == (x + y)**2*x + 1

    # 定义表达式 f
    f = (x**2 + 1)**2 - 1
    kwargs = {'gaussian': True}

    # 使用 factor 函数，level=0 的情况
    assert use(f, factor, level=0, kwargs=kwargs) == x**2*(x**2 + 2)
    # 使用 factor 函数，level=1 的情况
    assert use(f, factor, level=1, kwargs=kwargs) == (x + I)**2*(x - I)**2 - 1
    # 使用 factor 函数，level=2 的情况
    assert use(f, factor, level=2, kwargs=kwargs) == (x + I)**2*(x - I)**2 - 1
    # 使用 factor 函数，level=3 的情况
    assert use(f, factor, level=3, kwargs=kwargs) == (x**2 + 1)**2 - 1


# 定义测试函数 test_postorder_traversal，测试 postorder_traversal 函数
def test_postorder_traversal():
    # 定义符号变量 x, y, z, w
    x, y, z, w = symbols('x y z w')
    # 创建表达式 expr
    expr = z + w*(x + y)
    expected = [z, w, x, y, x + y, w*(x + y), w*(x + y) + z]
    # 使用默认排序键对表达式进行 postorder_traversal
    assert list(postorder_traversal(expr, keys=default_sort_key)) == expected
    # 使用默认键对表达式进行 postorder_traversal
    assert list(postorder_traversal(expr, keys=True)) == expected

    # 创建 Piecewise 表达式 expr
    expr = Piecewise((x, x < 1), (x**2, True))
    expected = [
        x, 1, x, x < 1, ExprCondPair(x, x < 1),
        2, x, x**2, S.true,
        ExprCondPair(x**2, True), Piecewise((x, x < 1), (x**2, True))
    ]
    # 使用默认排序键对表达式进行 postorder_traversal
    assert list(postorder_traversal(expr, keys=default_sort_key)) == expected
    # 使用默认排序键对表达式进行 postorder_traversal
    assert list(postorder_traversal([expr], keys=default_sort_key)) == expected + [[expr]]
    # 断言：验证积分表达式在后序遍历中的正确顺序
    assert list(postorder_traversal(Integral(x**2, (x, 0, 1)),
        keys=default_sort_key)) == [
            2, x, x**2, 0, 1, x, Tuple(x, 0, 1),
            Integral(x**2, Tuple(x, 0, 1))
        ]
    
    # 断言：验证元组嵌套结构在后序遍历中的正确顺序
    assert list(postorder_traversal(('abc', ('d', 'ef')))) == [
        'abc', 'd', 'ef', ('d', 'ef'), ('abc', ('d', 'ef'))
    ]
# 测试函数，用于测试 iterargs 函数的行为
def test_iterargs():
    # 创建一个函数符号 'f'
    f = Function('f')
    # 创建一个符号变量 'x'
    x = symbols('x')
    # 断言 iterfreeargs 函数对 Integral(f(x), (f(x), 1)) 的输出结果
    assert list(iterfreeargs(Integral(f(x), (f(x), 1)))) == [
        Integral(f(x), (f(x), 1)), 1]
    # 断言 iterargs 函数对 Integral(f(x), (f(x), 1)) 的输出结果
    assert list(iterargs(Integral(f(x), (f(x), 1)))) == [
        Integral(f(x), (f(x), 1)), f(x), (f(x), 1), x, f(x), 1, x]

# 测试函数，用于测试即将弃用的导入项
def test_deprecated_imports():
    # 创建一个符号变量 'x'
    x = symbols('x')

    # 捕捉 sympy 库的弃用警告
    with warns_deprecated_sympy():
        # 从 sympy.core.basic 模块导入 preorder_traversal 函数并使用
        from sympy.core.basic import preorder_traversal
        preorder_traversal(x)
    with warns_deprecated_sympy():
        # 从 sympy.simplify.simplify 模块导入 bottom_up 函数并使用
        from sympy.simplify.simplify import bottom_up
        bottom_up(x, lambda x: x)
    with warns_deprecated_sympy():
        # 从 sympy.simplify.simplify 模块导入 walk 函数并使用
        from sympy.simplify.simplify import walk
        walk(x, lambda x: x)
    with warns_deprecated_sympy():
        # 从 sympy.simplify.traversaltools 模块导入 use 函数并使用
        from sympy.simplify.traversaltools import use
        use(x, lambda x: x)
    with warns_deprecated_sympy():
        # 从 sympy.utilities.iterables 模块导入 postorder_traversal 函数并使用
        from sympy.utilities.iterables import postorder_traversal
        postorder_traversal(x)
    with warns_deprecated_sympy():
        # 从 sympy.utilities.iterables 模块导入 interactive_traversal 函数并使用
        from sympy.utilities.iterables import interactive_traversal
        capture(lambda: interactive_traversal(x))
```