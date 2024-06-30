# `D:\src\scipysrc\sympy\sympy\printing\tests\test_dot.py`

```
from sympy.printing.dot import (purestr, styleof, attrprint, dotnode,
        dotedges, dotprint)
# 导入必要的函数和类从 sympy.printing.dot 模块

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.numbers import (Float, Integer)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
# 导入必要的类和函数从 sympy.core 模块

from sympy.printing.repr import srepr
# 导入 srepr 函数从 sympy.printing.repr 模块

from sympy.abc import x
# 导入符号 x 从 sympy.abc 模块

# 定义测试函数 test_purestr，测试 purestr 函数的功能
def test_purestr():
    assert purestr(Symbol('x')) == "Symbol('x')"
    assert purestr(Basic(S(1), S(2))) == "Basic(Integer(1), Integer(2))"
    assert purestr(Float(2)) == "Float('2.0', precision=53)"

    assert purestr(Symbol('x'), with_args=True) == ("Symbol('x')", ())
    assert purestr(Basic(S(1), S(2)), with_args=True) == \
            ('Basic(Integer(1), Integer(2))', ('Integer(1)', 'Integer(2)'))
    assert purestr(Float(2), with_args=True) == \
        ("Float('2.0', precision=53)", ())

# 定义测试函数 test_styleof，测试 styleof 函数的功能
def test_styleof():
    styles = [(Basic, {'color': 'blue', 'shape': 'ellipse'}),
              (Expr,  {'color': 'black'})]
    assert styleof(Basic(S(1)), styles) == {'color': 'blue', 'shape': 'ellipse'}

    assert styleof(x + 1, styles) == {'color': 'black', 'shape': 'ellipse'}

# 定义测试函数 test_attrprint，测试 attrprint 函数的功能
def test_attrprint():
    assert attrprint({'color': 'blue', 'shape': 'ellipse'}) == \
           '"color"="blue", "shape"="ellipse"'

# 定义测试函数 test_dotnode，测试 dotnode 函数的功能
def test_dotnode():

    assert dotnode(x, repeat=False) == \
        '"Symbol(\'x\')" ["color"="black", "label"="x", "shape"="ellipse"];'
    assert dotnode(x+2, repeat=False) == \
        '"Add(Integer(2), Symbol(\'x\'))" ' \
        '["color"="black", "label"="Add", "shape"="ellipse"];', \
        dotnode(x+2,repeat=0)

    assert dotnode(x + x**2, repeat=False) == \
        '"Add(Symbol(\'x\'), Pow(Symbol(\'x\'), Integer(2)))" ' \
        '["color"="black", "label"="Add", "shape"="ellipse"];'
    assert dotnode(x + x**2, repeat=True) == \
        '"Add(Symbol(\'x\'), Pow(Symbol(\'x\'), Integer(2)))_()" ' \
        '["color"="black", "label"="Add", "shape"="ellipse"];'

# 定义测试函数 test_dotedges，测试 dotedges 函数的功能
def test_dotedges():
    assert sorted(dotedges(x+2, repeat=False)) == [
        '"Add(Integer(2), Symbol(\'x\'))" -> "Integer(2)";',
        '"Add(Integer(2), Symbol(\'x\'))" -> "Symbol(\'x\')";'
    ]
    assert sorted(dotedges(x + 2, repeat=True)) == [
        '"Add(Integer(2), Symbol(\'x\'))_()" -> "Integer(2)_(0,)";',
        '"Add(Integer(2), Symbol(\'x\'))_()" -> "Symbol(\'x\')_(1,)";'
    ]

# 定义测试函数 test_dotprint，测试 dotprint 函数的功能
def test_dotprint():
    text = dotprint(x+2, repeat=False)
    assert all(e in text for e in dotedges(x+2, repeat=False))
    assert all(
        n in text for n in [dotnode(expr, repeat=False)
        for expr in (x, Integer(2), x+2)])
    assert 'digraph' in text

    text = dotprint(x+x**2, repeat=False)
    assert all(e in text for e in dotedges(x+x**2, repeat=False))
    assert all(
        n in text for n in [dotnode(expr, repeat=False)
        for expr in (x, Integer(2), x**2)])
    assert 'digraph' in text

    text = dotprint(x+x**2, repeat=True)
    assert all(e in text for e in dotedges(x+x**2, repeat=True))
    # 断言：验证所有表达式的节点都存在于文本中
    assert all(
        n in text for n in [dotnode(expr, pos=())
        for expr in [x + x**2]])
    
    # 生成文本：生成一个带有重复节点的点图文本
    text = dotprint(x**x, repeat=True)
    
    # 断言：验证所有边存在于文本中
    assert all(e in text for e in dotedges(x**x, repeat=True))
    
    # 断言：验证特定位置的节点存在于文本中
    assert all(
        n in text for n in [dotnode(x, pos=(0,)), dotnode(x, pos=(1,))])
    
    # 断言：验证文本中包含 'digraph' 关键字
    assert 'digraph' in text
def test_dotprint_depth():
    # 调用 dotprint 函数，对表达式 3*x+2 进行深度为 1 的 DOT 图形生成
    text = dotprint(3*x+2, depth=1)
    # 断言生成的 DOT 文本中包含节点 3*x+2
    assert dotnode(3*x+2) in text
    # 断言生成的 DOT 文本中不包含节点 x
    assert dotnode(x) not in text
    # 调用 dotprint 函数，对表达式 3*x+2 进行默认深度的 DOT 图形生成
    text = dotprint(3*x+2)
    # 断言生成的 DOT 文本中不包含 "depth" 关键字
    assert "depth" not in text

def test_Matrix_and_non_basics():
    # 导入需要的模块
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    # 定义符号变量 n
    n = Symbol('n')
    # 断言对 MatrixSymbol('X', n, n) 的 DOT 图形生成结果符合预期格式
    assert dotprint(MatrixSymbol('X', n, n)) == \
"""digraph{

# Graph style
"ordering"="out"
"rankdir"="TD"

#########
# Nodes #
#########

"MatrixSymbol(Str('X'), Symbol('n'), Symbol('n'))_()" ["color"="black", "label"="MatrixSymbol", "shape"="ellipse"];
"Str('X')_(0,)" ["color"="blue", "label"="X", "shape"="ellipse"];
"Symbol('n')_(1,)" ["color"="black", "label"="n", "shape"="ellipse"];
"Symbol('n')_(2,)" ["color"="black", "label"="n", "shape"="ellipse"];

#########
# Edges #
#########

"MatrixSymbol(Str('X'), Symbol('n'), Symbol('n'))_()" -> "Str('X')_(0,)";
"MatrixSymbol(Str('X'), Symbol('n'), Symbol('n'))_()" -> "Symbol('n')_(1,)";
"MatrixSymbol(Str('X'), Symbol('n'), Symbol('n'))_()" -> "Symbol('n')_(2,)";
}"""


def test_labelfunc():
    # 调用 dotprint 函数，对表达式 x+2 进行 DOT 图形生成，节点标签使用 srepr 函数
    text = dotprint(x + 2, labelfunc=srepr)
    # 断言生成的 DOT 文本中包含 "Symbol('x')" 字符串
    assert "Symbol('x')" in text
    # 断言生成的 DOT 文本中包含 "Integer(2)" 字符串
    assert "Integer(2)" in text


def test_commutative():
    # 定义符号变量 x 和 y，且它们是非交换的
    x, y = symbols('x y', commutative=False)
    # 断言 x + y 和 y + x 生成的 DOT 图形相同
    assert dotprint(x + y) == dotprint(y + x)
    # 断言 x*y 和 y*x 生成的 DOT 图形不相同
    assert dotprint(x*y) != dotprint(y*x)
```