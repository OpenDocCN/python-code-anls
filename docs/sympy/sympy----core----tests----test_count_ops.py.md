# `D:\src\scipysrc\sympy\sympy\core\tests\test_count_ops.py`

```
# 导入符号数学库中的具体类和函数
from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.function import (Derivative, Function, count_ops)
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import (Eq, Rel)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import (And, Equivalent, ITE, Implies, Nand,
    Nor, Not, Or, Xor)
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.core.containers import Tuple

# 定义符号变量 x, y, z 和 a, b, c
x, y, z = symbols('x,y,z')
a, b, c = symbols('a,b,c')

# 定义测试函数 test_count_ops_non_visual
def test_count_ops_non_visual():
    # 定义内部函数 count，用于计算操作数数量，visual 参数为 False
    def count(val):
        return count_ops(val, visual=False)
    
    # 断言操作数为 0
    assert count(x) == 0
    # 断言操作数不是 S.Zero
    assert count(x) is not S.Zero
    # 断言操作数为 1
    assert count(x + y) == 1
    # 断言操作数不是 S.One
    assert count(x + y) is not S.One
    # 断言操作数为 4
    assert count(x + y*x + 2*y) == 4
    # 断言操作数为 1，注意字典的计数方式
    assert count({x + y: x}) == 1
    # 断言操作数不是 S.One
    assert count({x + y: S(2) + x}) is not S.One
    # 断言操作数为 1
    assert count(x < y) == 1
    # 断言操作数为 1
    assert count(Or(x,y)) == 1
    # 断言操作数为 1
    assert count(And(x,y)) == 1
    # 断言操作数为 1
    assert count(Not(x)) == 1
    # 断言操作数为 2
    assert count(Nor(x,y)) == 2
    # 断言操作数为 2
    assert count(Nand(x,y)) == 2
    # 断言操作数为 1
    assert count(Xor(x,y)) == 1
    # 断言操作数为 1
    assert count(Implies(x,y)) == 1
    # 断言操作数为 1
    assert count(Equivalent(x,y)) == 1
    # 断言操作数为 1
    assert count(ITE(x,y,z)) == 1
    # 断言操作数为 0，特殊情况下的 ITE
    assert count(ITE(True,x,y)) == 0

# 定义测试函数 test_count_ops_visual
def test_count_ops_visual():
    # 定义符号常量，用于视觉操作数计算
    ADD, MUL, POW, SIN, COS, EXP, AND, D, G, M = symbols(
        'Add Mul Pow sin cos exp And Derivative Integral Sum'.upper())
    DIV, SUB, NEG = symbols('DIV SUB NEG')
    LT, LE, GT, GE, EQ, NE = symbols('LT LE GT GE EQ NE')
    NOT, OR, AND, XOR, IMPLIES, EQUIVALENT, _ITE, BASIC, TUPLE = symbols(
        'Not Or And Xor Implies Equivalent ITE Basic Tuple'.upper())

    # 定义内部函数 count，用于计算视觉操作数数量，visual 参数为 True
    def count(val):
        return count_ops(val, visual=True)

    # 以下为各种断言，验证不同数学表达式的视觉操作数计算结果
    assert count(7) is S.Zero
    assert count(S(7)) is S.Zero
    assert count(-1) == NEG
    assert count(-2) == NEG
    assert count(S(2)/3) == DIV
    assert count(Rational(2, 3)) == DIV
    assert count(pi/3) == DIV
    assert count(-pi/3) == DIV + NEG
    assert count(I - 1) == SUB
    assert count(1 - I) == SUB
    assert count(1 - 2*I) == SUB + MUL

    assert count(x) is S.Zero
    assert count(-x) == NEG
    assert count(-2*x/3) == NEG + DIV + MUL
    assert count(Rational(-2, 3)*x) == NEG + DIV + MUL
    assert count(1/x) == DIV
    assert count(1/(x*y)) == DIV + MUL
    assert count(-1/x) == NEG + DIV
    assert count(-2/x) == NEG + DIV
    assert count(x/y) == DIV
    assert count(-x/y) == NEG + DIV

    assert count(x**2) == POW
    assert count(-x**2) == POW + NEG
    assert count(-2*x**2) == POW + MUL + NEG

    assert count(x + pi/3) == ADD + DIV
    assert count(x + S.One/3) == ADD + DIV
    assert count(x + Rational(1, 3)) == ADD + DIV
    assert count(x + y) == ADD
    assert count(x - y) == SUB
    assert count(y - x) == SUB
    # 断言语句：计算表达式 `-1/(x - y)` 中操作的数量，应等于 DIV + NEG + SUB
    assert count(-1/(x - y)) == DIV + NEG + SUB
    
    # 断言语句：计算表达式 `-1/(y - x)` 中操作的数量，应等于 DIV + NEG + SUB
    assert count(-1/(y - x)) == DIV + NEG + SUB
    
    # 断言语句：计算表达式 `1 + x**y` 中操作的数量，应等于 ADD + POW
    assert count(1 + x**y) == ADD + POW
    
    # 断言语句：计算表达式 `1 + x + y` 中操作的数量，应等于 2*ADD
    assert count(1 + x + y) == 2*ADD
    
    # 断言语句：计算表达式 `1 + x + y + z` 中操作的数量，应等于 3*ADD
    assert count(1 + x + y + z) == 3*ADD
    
    # 断言语句：计算表达式 `1 + x**y + 2*x*y + y**2` 中操作的数量，应等于 3*ADD + 2*POW + 2*MUL
    assert count(1 + x**y + 2*x*y + y**2) == 3*ADD + 2*POW + 2*MUL
    
    # 断言语句：计算表达式 `2*z + y + x + 1` 中操作的数量，应等于 3*ADD + MUL
    assert count(2*z + y + x + 1) == 3*ADD + MUL
    
    # 断言语句：计算表达式 `2*z + y**17 + x + 1` 中操作的数量，应等于 3*ADD + MUL + POW
    assert count(2*z + y**17 + x + 1) == 3*ADD + MUL + POW
    
    # 断言语句：计算表达式 `2*z + y**17 + x + sin(x)` 中操作的数量，应等于 3*ADD + POW + MUL + SIN
    assert count(2*z + y**17 + x + sin(x)) == 3*ADD + POW + MUL + SIN
    
    # 断言语句：计算表达式 `2*z + y**17 + x + sin(x**2)` 中操作的数量，应等于 3*ADD + MUL + 2*POW + SIN
    assert count(2*z + y**17 + x + sin(x**2)) == 3*ADD + MUL + 2*POW + SIN
    
    # 断言语句：计算表达式 `2*z + y**17 + x + sin(x**2) + exp(cos(x))` 中操作的数量，应等于 4*ADD + MUL + 2*POW + EXP + COS + SIN
    assert count(2*z + y**17 + x + sin(x**2) + exp(cos(x))) == 4*ADD + MUL + 2*POW + EXP + COS + SIN
    
    # 断言语句：计算 Derivative(x, x) 操作的数量，应等于 D
    assert count(Derivative(x, x)) == D
    
    # 断言语句：计算表达式 Integral(x, x) + 2*x/(1 + x) 中操作的数量，应等于 G + DIV + MUL + 2*ADD
    assert count(Integral(x, x) + 2*x/(1 + x)) == G + DIV + MUL + 2*ADD
    
    # 断言语句：计算表达式 Sum(x, (x, 1, x + 1)) + 2*x/(1 + x) 中操作的数量，应等于 M + DIV + MUL + 3*ADD
    assert count(Sum(x, (x, 1, x + 1)) + 2*x/(1 + x)) == M + DIV + MUL + 3*ADD
    
    # 断言语句：计算 Basic() 操作的数量，应等于 S.Zero
    assert count(Basic()) is S.Zero
    
    # 断言语句：计算表达式 {x + 1: sin(x)} 中操作的数量，应等于 ADD + SIN
    assert count({x + 1: sin(x)}) == ADD + SIN
    
    # 断言语句：计算表达式 [x + 1, sin(x) + y, None] 中操作的数量，应等于 ADD + SIN + ADD
    assert count([x + 1, sin(x) + y, None]) == ADD + SIN + ADD
    
    # 断言语句：计算表达式 {x + 1: sin(x), y: cos(x) + 1} 中操作的数量，应等于 SIN + COS + 2*ADD
    assert count({x + 1: sin(x), y: cos(x) + 1}) == SIN + COS + 2*ADD
    
    # 断言语句：计算空字典 {} 的操作数量，应等于 S.Zero
    assert count({}) is S.Zero
    
    # 断言语句：计算表达式 [x + 1, sin(x)*y, None] 中操作的数量，应等于 SIN + ADD + MUL
    assert count([x + 1, sin(x)*y, None]) == SIN + ADD + MUL
    
    # 断言语句：计算空列表 [] 的操作数量，应等于 S.Zero
    assert count([]) is S.Zero
    
    # 断言语句：计算 Basic() 操作的数量，应等于 0
    assert count(Basic()) == 0
    
    # 断言语句：计算表达式 Basic(Basic(), Basic(x, x+y)) 中操作的数量，应等于 ADD + 2*BASIC
    assert count(Basic(Basic(),Basic(x,x+y))) == ADD + 2*BASIC
    
    # 断言语句：计算表达式 Basic(x, x + y) 中操作的数量，应等于 ADD + BASIC
    assert count(Basic(x, x + y)) == ADD + BASIC
    
    # 断言语句：计算表达式 [Rel(x, y, op) for op in '< <= > >= == <> !='.split()] 中操作的数量列表
    assert [count(Rel(x, y, op)) for op in '< <= > >= == <> !='.split()] == [LT, LE, GT, GE, EQ, NE, NE]
    
    # 断言语句：计算表达式 Or(x, y) 中操作的数量，应等于 OR
    assert count(Or(x, y)) == OR
    
    # 断言语句：计算表达式 And(x, y) 中操作的数量，应等于 AND
    assert count(And(x, y)) == AND
    
    # 断言语句：计算表达式 Or(x, Or(y, And(z, a))) 中操作的数量，应等于 AND + OR
    assert count(Or(x, Or(y, And(z, a)))) == AND + OR
    
    # 断言语句：计算表达式 Nor(x, y) 中操作的数量，应等于 NOT + OR
    assert count(Nor(x, y)) == NOT + OR
    
    # 断言语句：计算表达式 Nand(x, y) 中操作的数量，应等于 NOT + AND
    assert count(Nand(x, y)) == NOT + AND
    
    # 断言语句：计算表达式 Xor(x, y) 中操作的数量，应等于 XOR
    assert count(Xor(x, y)) == XOR
    
    # 断言语句：计算表达式 Implies(x, y) 中操作的数量，应等于 IMPLIES
    assert count(Implies(x, y)) == IMPLIES
    
    # 断言语句：计算表达式 Equivalent(x, y) 中操作的数量，应等于 EQUIVALENT
    assert count(Equivalent(x, y)) == EQUIVALENT
    
    # 断言语句：计算表达式 ITE(x, y, z) 中操作的数量，应等于 _ITE
    assert count(ITE(x, y, z)) == _ITE
    
    # 断言语句：计算表达式 [Or(x, y), And(x, y), Basic(x + y)] 中操作的数量，应等于 ADD + AND + BASIC + OR
    assert count([Or(x, y), And(x, y), Basic(x + y)]) == ADD + AND +
# 定义一个测试函数，用于测试问题编号为 9324 的问题
def test_issue_9324():
    # 定义一个内部函数 count，用于计算表达式的操作数数量，不显示可视化结果
    def count(val):
        return count_ops(val, visual=False)

    # 创建一个符号矩阵 M，大小为 10x10
    M = MatrixSymbol('M', 10, 10)
    # 断言：矩阵 M 中索引为 (0, 0) 的元素操作数数量为 0
    assert count(M[0, 0]) == 0
    # 断言：表达式 2 * M[0, 0] + M[5, 7] 的操作数数量为 2
    assert count(2 * M[0, 0] + M[5, 7]) == 2

    # 创建两个符号矩阵 P 和 Q，大小均为 3x3
    P = MatrixSymbol('P', 3, 3)
    Q = MatrixSymbol('Q', 3, 3)
    # 断言：矩阵 P 和 Q 的加法操作的操作数数量为 1
    assert count(P + Q) == 1

    # 创建两个整数符号 m 和 n
    m = Symbol('m', integer=True)
    n = Symbol('n', integer=True)
    # 根据 m 和 n 创建一个符号矩阵 M，大小为 (m+n) x (m*m)
    M = MatrixSymbol('M', m + n, m * m)
    # 断言：矩阵 M 中索引为 (0, 1) 的元素操作数数量为 2
    assert count(M[0, 1]) == 2


# 定义一个测试函数，用于测试问题编号为 21532 的问题
def test_issue_21532():
    # 创建两个函数 f 和 g
    f = Function('f')
    g = Function('g')
    # 创建两个符号 FUNC_F 和 FUNC_G，用于表示函数 f 和 g 的操作数数量
    FUNC_F, FUNC_G = symbols('FUNC_F, FUNC_G')
    # 断言：函数 f 在变量 x 上的操作数数量等于 FUNC_F
    assert f(x).count_ops(visual=True) == FUNC_F
    # 断言：函数 g 在变量 x 上的操作数数量等于 FUNC_G
    assert g(x).count_ops(visual=True) == FUNC_G
```