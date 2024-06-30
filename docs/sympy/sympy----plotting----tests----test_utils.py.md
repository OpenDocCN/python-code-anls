# `D:\src\scipysrc\sympy\sympy\plotting\tests\test_utils.py`

```
from pytest import raises
from sympy import (
    symbols, Expr, Tuple, Integer, cos, solveset, FiniteSet, ImageSet)
from sympy.plotting.utils import (
    _create_ranges, _plot_sympify, extract_solution)
from sympy.physics.mechanics import ReferenceFrame, Vector as MechVector
from sympy.vector import CoordSys3D, Vector

def test_plot_sympify():
    x, y = symbols("x, y")

    # 参数已经是 sympify 后的表达式，直接返回
    args = x + y
    r = _plot_sympify(args)
    assert r == args

    # 其中一个参数需要进行 sympify
    args = (x + y, 1)
    r = _plot_sympify(args)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert isinstance(r[0], Expr)
    assert isinstance(r[1], Integer)

    # 字符串和字典不应进行 sympify
    args = (x + y, (x, 0, 1), "str", 1, {1: 1, 2: 2.0})
    r = _plot_sympify(args)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 5
    assert isinstance(r[0], Expr)
    assert isinstance(r[1], Tuple)
    assert isinstance(r[2], str)
    assert isinstance(r[3], Integer)
    assert isinstance(r[4], dict) and isinstance(r[4][1], int) and isinstance(r[4][2], float)

    # 包含字符串的嵌套参数
    args = ((x + y, (y, 0, 1), "a"), (x + 1, (x, 0, 1), "$f_{1}$"))
    r = _plot_sympify(args)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert isinstance(r[0], Tuple)
    assert isinstance(r[0][1], Tuple)
    assert isinstance(r[0][1][1], Integer)
    assert isinstance(r[0][2], str)
    assert isinstance(r[1], Tuple)
    assert isinstance(r[1][1], Tuple)
    assert isinstance(r[1][1][1], Integer)
    assert isinstance(r[1][2], str)

    # 来自 sympy.physics.mechanics 模块的向量不进行 sympify
    # 来自 sympy.vector 模块的向量进行 sympify
    # 在两种情况下都不应引发错误
    R = ReferenceFrame("R")
    v1 = 2 * R.x + R.y
    C = CoordSys3D("C")
    v2 = 2 * C.i + C.j
    args = (v1, v2)
    r = _plot_sympify(args)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert isinstance(v1, MechVector)
    assert isinstance(v2, Vector)


def test_create_ranges():
    x, y = symbols("x, y")

    # 用户未提供任何范围 -> 返回默认范围
    r = _create_ranges({x}, [], 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 1
    assert isinstance(r[0], (Tuple, tuple))
    assert r[0] == (x, -10, 10)

    # 用户提供不足的范围 -> 创建默认范围
    r = _create_ranges({x, y}, [], 2)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert isinstance(r[0], (Tuple, tuple))
    assert isinstance(r[1], (Tuple, tuple))
    assert r[0] == (x, -10, 10) or (y, -10, 10)
    assert r[1] == (y, -10, 10) or (x, -10, 10)
    assert r[0] != r[1]

    # 用户提供的范围不足 -> 创建默认范围
    r = _create_ranges(
        {x, y},
        [
            (x, 0, 1),
        ],
        2,
    )
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert isinstance(r[0], (Tuple, tuple))
    # 断言检查第二个元素是否为元组类型
    assert isinstance(r[1], (Tuple, tuple))
    # 断言检查第一个元素是否为指定元组格式
    assert r[0] == (x, 0, 1) or (y, -10, 10)
    # 断言检查第二个元素是否为指定元组格式
    assert r[1] == (y, -10, 10) or (x, 0, 1)
    # 断言检查第一个元素和第二个元素不相等

    # 抛出 ValueError 异常，检查是否有过多的自由符号，使用空列表作为已知范围
    raises(ValueError, lambda: _create_ranges({x, y}, [], 1))
    # 抛出 ValueError 异常，检查是否有过多的自由符号，使用指定的已知范围
    raises(ValueError, lambda: _create_ranges({x, y}, [(x, 0, 5), (y, 0, 1)], 1))
# 定义一个测试函数，用于测试 extract_solution 函数的功能
def test_extract_solution():
    # 创建一个符号变量 x
    x = symbols("x")

    # 求解方程 cos(10 * x)，返回一个解集对象
    sol = solveset(cos(10 * x))
    # 断言解集对象包含 ImageSet 类型的对象
    assert sol.has(ImageSet)
    
    # 调用 extract_solution 函数处理解集 sol，返回结果 res
    res = extract_solution(sol)
    # 断言结果 res 的长度为 20
    assert len(res) == 20
    # 断言结果 res 的类型为 FiniteSet
    assert isinstance(res, FiniteSet)

    # 再次调用 extract_solution 函数，传入额外的参数 20
    res = extract_solution(sol, 20)
    # 断言结果 res 的长度为 40
    assert len(res) == 40
    # 断言结果 res 的类型为 FiniteSet
    assert isinstance(res, FiniteSet)
```