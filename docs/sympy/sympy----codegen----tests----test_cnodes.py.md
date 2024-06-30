# `D:\src\scipysrc\sympy\sympy\codegen\tests\test_cnodes.py`

```
# 导入必要的符号和代码生成工具
from sympy.core.symbol import symbols
from sympy.printing.codeprinter import ccode
from sympy.codegen.ast import Declaration, Variable, float64, int64, String, CodeBlock
from sympy.codegen.cnodes import (
    alignof, CommaOperator, goto, Label, PreDecrement, PostDecrement, PreIncrement, PostIncrement,
    sizeof, union, struct
)

# 定义符号变量 x 和 y
x, y = symbols('x y')

# 测试 alignof 函数
def test_alignof():
    ax = alignof(x)
    # 断言 alignof(x) 的 C 代码表示为 'alignof(x)'
    assert ccode(ax) == 'alignof(x)'
    # 断言 ax 对象的函数与其原始函数一致
    assert ax.func(*ax.args) == ax

# 测试 CommaOperator 类
def test_CommaOperator():
    # 创建一个 CommaOperator 对象，包含 PreIncrement(x) 和 2*x
    expr = CommaOperator(PreIncrement(x), 2*x)
    # 断言生成的 C 代码为 '(++(x), 2*x)'
    assert ccode(expr) == '(++(x), 2*x)'
    # 断言 expr 对象的函数与其原始函数一致
    assert expr.func(*expr.args) == expr

# 测试 goto 和 Label 函数
def test_goto_Label():
    s = 'early_exit'
    # 创建一个跳转到 'early_exit' 的 goto 对象
    g = goto(s)
    # 断言 g 对象的函数与其原始函数一致
    assert g.func(*g.args) == g
    # 断言 g 与另一个跳转到 'foobar' 的 goto 对象不相等
    assert g != goto('foobar')
    # 断言生成的 C 代码为 'goto early_exit'
    assert ccode(g) == 'goto early_exit'

    # 创建一个名为 'early_exit' 的 Label 对象
    l1 = Label(s)
    # 断言生成的 C 代码为 'early_exit:'
    assert ccode(l1) == 'early_exit:'
    # 断言 l1 与另一个名为 'early_exit' 的 Label 对象相等
    assert l1 == Label('early_exit')
    # 断言 l1 与名为 'foobar' 的 Label 对象不相等
    assert l1 != Label('foobar')

    # 创建一个名为 'early_exit' 的 Label 对象，带有预定义的 body
    body = [PreIncrement(x)]
    l2 = Label(s, body)
    # 断言 l2 的名称为 String("early_exit")
    assert l2.name == String("early_exit")
    # 断言 l2 的 body 包含 PreIncrement(x) 操作
    assert l2.body == CodeBlock(PreIncrement(x))
    # 断言生成的 C 代码为如下格式:
    # "early_exit:\n"
    # "++(x);"
    assert ccode(l2) == ("early_exit:\n"
        "++(x);")

    # 创建一个名为 'early_exit' 的 Label 对象，带有多个预定义的 body
    body = [PreIncrement(x), PreDecrement(y)]
    l2 = Label(s, body)
    # 断言 l2 的名称为 String("early_exit")
    assert l2.name == String("early_exit")
    # 断言 l2 的 body 包含 PreIncrement(x) 和 PreDecrement(y) 操作
    assert l2.body == CodeBlock(PreIncrement(x), PreDecrement(y))
    # 断言生成的 C 代码为如下格式:
    # "early_exit:\n"
    # "{\n   ++(x);\n   --(y);\n}"
    assert ccode(l2) == ("early_exit:\n"
        "{\n   ++(x);\n   --(y);\n}")

# 测试 PreDecrement 函数
def test_PreDecrement():
    p = PreDecrement(x)
    # 断言 p 对象的函数与其原始函数一致
    assert p.func(*p.args) == p
    # 断言生成的 C 代码为 '--(x)'
    assert ccode(p) == '--(x)'

# 测试 PostDecrement 函数
def test_PostDecrement():
    p = PostDecrement(x)
    # 断言 p 对象的函数与其原始函数一致
    assert p.func(*p.args) == p
    # 断言生成的 C 代码为 '(x)--'
    assert ccode(p) == '(x)--'

# 测试 PreIncrement 函数
def test_PreIncrement():
    p = PreIncrement(x)
    # 断言 p 对象的函数与其原始函数一致
    assert p.func(*p.args) == p
    # 断言生成的 C 代码为 '++(x)'
    assert ccode(p) == '++(x)'

# 测试 PostIncrement 函数
def test_PostIncrement():
    p = PostIncrement(x)
    # 断言 p 对象的函数与其原始函数一致
    assert p.func(*p.args) == p
    # 断言生成的 C 代码为 '(x)++'
    assert ccode(p) == '(x)++'

# 测试 sizeof 函数
def test_sizeof():
    typename = 'unsigned int'
    # 创建一个 sizeof(typename) 的对象
    sz = sizeof(typename)
    # 断言生成的 C 代码为 'sizeof(unsigned int)'
    assert ccode(sz) == 'sizeof(%s)' % typename
    # 断言 sz 对象的函数与其原始函数一致
    assert sz.func(*sz.args) == sz
    # 断言 sz 不是原子对象
    assert not sz.is_Atom
    # 断言 sz 的原子包含 'unsigned int' 和 'sizeof'
    assert sz.atoms() == {String('unsigned int'), String('sizeof')}

# 测试 struct 函数
def test_struct():
    # 创建两个变量，类型为 float64
    vx, vy = Variable(x, type=float64), Variable(y, type=float64)
    # 创建一个名为 'vec2' 的 struct 对象，包含两个变量 vx 和 vy
    s = struct('vec2', [vx, vy])
    # 断言 s 对象的函数与其原始函数一致
    assert s.func(*s.args) == s
    # 断言 s 与另一个包含相同 vx 和 vy 的 struct 对象相等
    assert s == struct('vec2', (vx, vy))
    # 断言 s 与包含不同顺序的 struct 对象不相等
    assert s != struct('vec2', (vy, vx))
    # 断言 s 的名称为 'vec2'
    assert str(s.name) == 'vec2'
    # 断言 s 包含两个声明
    assert len(s.declarations) == 2
    # 断言 s 的所有声明都是 Declaration 类型
    assert all(isinstance(arg, Declaration) for arg in s.declarations)
    # 断言生成的 C 代码为如下格式:
    # "struct vec2 {\n"
    # "   double x;\n"
    # "   double y;\n"
    # "}"
    assert ccode(s) == (
        "struct vec2 {\n"
        "   double x;\n"
        "   double y;\n"
        "}")

# 测试 union 函数
def test_union():
    # 创建两个变量，一个类型为 float64，一个类型为 int64
    vx, vy = Variable(x, type=float64), Variable(y, type=int64)
    # 创建一个名为 'dualuse' 的 union 对象，包含两个变量 vx 和 vy
    u = union('dualuse', [vx, vy])
    # 断言 u 对象的函数与其原始函数一致
    assert u.func(*u.args) == u
    # 断言 u 与另一个包含相同 vx 和 vy 的 union 对象相等
    assert u == union('dualuse', (vx, vy))
    # 断言 u 的名称为 'dualuse'
    assert str(u.name) == 'dualuse'
    # 断言 u 包含两个声明
    assert len(u.declarations) == 2
    # 断言 u 的所有声明都是 Declaration 类型
    assert all(isinstance(arg, Declaration) for arg in u.declarations)
    # 断言语句，用于检查条件是否为真，若为假则触发 AssertionError，并输出相应信息
    assert ccode(u) == (
        "union dualuse {\n"  # 定义一个联合体 dualuse，可以同时用于存储 double 类型或 int64_t 类型的数据
        "   double x;\n"      # 在联合体中定义一个 double 类型的变量 x
        "   int64_t y;\n"     # 在联合体中定义一个 int64_t 类型的变量 y
        "}")                  # 结束联合体定义
```