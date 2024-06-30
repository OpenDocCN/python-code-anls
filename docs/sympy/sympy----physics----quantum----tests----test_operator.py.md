# `D:\src\scipysrc\sympy\sympy\physics\quantum\tests\test_operator.py`

```
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.mul import Mul
from sympy.core.numbers import (Integer, pi)
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.trigonometric import sin
from sympy.physics.quantum.qexpr import QExpr
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.hilbert import HilbertSpace
from sympy.physics.quantum.operator import (Operator, UnitaryOperator,
                                            HermitianOperator, OuterProduct,
                                            DifferentialOperator,
                                            IdentityOperator)
from sympy.physics.quantum.state import Ket, Bra, Wavefunction
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.spin import JzKet, JzBra
from sympy.physics.quantum.trace import Tr
from sympy.matrices import eye


class CustomKet(Ket):
    @classmethod
    def default_args(cls):
        return ("t",)


class CustomOp(HermitianOperator):
    @classmethod
    def default_args(cls):
        return ("T",)


t_ket = CustomKet()  # 创建一个自定义的量子态对象 CustomKet 实例
t_op = CustomOp()  # 创建一个自定义的厄米算符对象 CustomOp 实例


def test_operator():
    A = Operator('A')  # 创建一个名为 'A' 的算符对象 Operator 实例
    B = Operator('B')  # 创建一个名为 'B' 的算符对象 Operator 实例
    C = Operator('C')  # 创建一个名为 'C' 的算符对象 Operator 实例

    assert isinstance(A, Operator)  # 确认 A 是 Operator 类的实例
    assert isinstance(A, QExpr)  # 确认 A 是 QExpr 类的实例

    assert A.label == (Symbol('A'),)  # 确认 A 的标签为 ('A',)
    assert A.is_commutative is False  # 确认 A 不是可交换的
    assert A.hilbert_space == HilbertSpace()  # 确认 A 的希尔伯特空间为默认空间 HilbertSpace()

    assert A*B != B*A  # 确认 A 和 B 不可交换

    assert (A*(B + C)).expand() == A*B + A*C  # 确认算符展开的分配性质
    assert ((A + B)**2).expand() == A**2 + A*B + B*A + B**2  # 确认算符展开的平方性质

    assert t_op.label[0] == Symbol(t_op.default_args()[0])  # 确认自定义厄米算符的默认参数和标签符号相符

    assert Operator() == Operator("O")  # 确认默认算符和指定名字算符等效
    assert A*IdentityOperator() == A  # 确认算符乘以单位算符等效于自身


def test_operator_inv():
    A = Operator('A')  # 创建一个名为 'A' 的算符对象 Operator 实例
    assert A*A.inv() == 1  # 确认算符 A 乘以其逆算符等于单位算符
    assert A.inv()*A == 1  # 确认算符 A 逆算符乘以 A 等于单位算符


def test_hermitian():
    H = HermitianOperator('H')  # 创建一个名为 'H' 的厄米算符对象 HermitianOperator 实例

    assert isinstance(H, HermitianOperator)  # 确认 H 是厄米算符类的实例
    assert isinstance(H, Operator)  # 确认 H 是算符类的实例

    assert Dagger(H) == H  # 确认 H 的厄米共轭等于自身
    assert H.inv() != H  # 确认 H 的逆算符不等于自身
    assert H.is_commutative is False  # 确认 H 不是可交换的
    assert Dagger(H).is_commutative is False  # 确认 H 的厄米共轭也不是可交换的


def test_unitary():
    U = UnitaryOperator('U')  # 创建一个名为 'U' 的酉算符对象 UnitaryOperator 实例

    assert isinstance(U, UnitaryOperator)  # 确认 U 是酉算符类的实例
    assert isinstance(U, Operator)  # 确认 U 是算符类的实例

    assert U.inv() == Dagger(U)  # 确认 U 的逆算符等于其厄米共轭
    assert U*Dagger(U) == 1  # 确认 U 乘以其厄米共轭等于单位算符
    assert Dagger(U)*U == 1  # 确认 U 的厄米共轭乘以 U 等于单位算符
    assert U.is_commutative is False  # 确认 U 不是可交换的
    assert Dagger(U).is_commutative is False  # 确认 U 的厄米共轭也不是可交换的


def test_identity():
    I = IdentityOperator()  # 创建一个单位算符对象 IdentityOperator 实例
    O = Operator('O')  # 创建一个名为 'O' 的算符对象 Operator 实例
    x = Symbol("x")  # 创建一个符号对象 x

    assert isinstance(I, IdentityOperator)  # 确认 I 是单位算符类的实例
    assert isinstance(I, Operator)  # 确认 I 是算符类的实例

    assert I * O == O  # 确认单位算符乘以任意算符等效于算符本身
    assert O * I == O  # 确认任意算符乘以单位算符等效于算符本身
    assert I * Dagger(O) == Dagger(O)  # 确认单位算符乘以算符的厄米共轭等效于厄米共轭
    assert Dagger(O) * I == Dagger(O)  # 确认算符的厄米共轭乘以单位算符等效于厄米共轭
    assert isinstance(I * I, IdentityOperator)  # 确认单位算符乘以自身仍然是单位算符
    assert isinstance(3 * I, Mul)  # 确认单位算符乘以数值得到乘法表达式
    assert isinstance(I * x, Mul)  # 确认单位算符乘以符号得到乘法表达式
    assert I.inv() == I  # 确认单位算符的逆等于自身
    assert Dagger(I) == I  # 确认单位算符的厄米共轭等于自身
    assert qapply(I * O) == O  # 确认算符作用于单位算符乘以任意算符等效于算符本身
    assert qapply(O * I) == O  # 确认算符作用于任意算符乘以单位算符等效于算符本身
    # 对于给定的整数列表 [2, 3, 5]，依次进行以下操作：
    for n in [2, 3, 5]:
        # 使用 IdentityOperator 类创建对象，并计算其表示（representation），
        # 然后断言其结果等于单位矩阵（eye(n) 表示大小为 n 的单位矩阵）。
        assert represent(IdentityOperator(n)) == eye(n)
# 定义一个名为 test_outer_product 的测试函数
def test_outer_product():
    # 创建一个名为 k 的 Ket 对象
    k = Ket('k')
    # 创建一个名为 b 的 Bra 对象
    b = Bra('b')
    # 创建一个 k 和 b 的外积对象 op
    op = OuterProduct(k, b)

    # 断言 op 是 OuterProduct 类的实例
    assert isinstance(op, OuterProduct)
    # 断言 op 是 Operator 类的实例
    assert isinstance(op, Operator)

    # 断言 op 的 ket 属性等于 k
    assert op.ket == k
    # 断言 op 的 bra 属性等于 b
    assert op.bra == b
    # 断言 op 的 label 属性等于 (k, b)
    assert op.label == (k, b)
    # 断言 op 的 is_commutative 属性为 False
    assert op.is_commutative is False

    # 将 op 重新赋值为 k*b 的乘积
    op = k*b

    # 断言 op 是 OuterProduct 类的实例
    assert isinstance(op, OuterProduct)
    # 断言 op 是 Operator 类的实例
    assert isinstance(op, Operator)

    # 断言 op 的 ket 属性等于 k
    assert op.ket == k
    # 断言 op 的 bra 属性等于 b
    assert op.bra == b
    # 断言 op 的 label 属性等于 (k, b)
    assert op.label == (k, b)
    # 断言 op 的 is_commutative 属性为 False
    assert op.is_commutative is False

    # 将 op 重新赋值为 2*k*b 的乘积
    op = 2*k*b

    # 断言 op 等于 Mul(Integer(2), k, b)
    assert op == Mul(Integer(2), k, b)

    # 将 op 重新赋值为 2*(k*b) 的乘积
    op = 2*(k*b)

    # 断言 op 等于 Mul(Integer(2), OuterProduct(k, b))
    assert op == Mul(Integer(2), OuterProduct(k, b))

    # 断言 Dagger(k*b) 等于 OuterProduct(Dagger(b), Dagger(k))
    assert Dagger(k*b) == OuterProduct(Dagger(b), Dagger(k))
    # 断言 Dagger(k*b).is_commutative 属性为 False
    assert Dagger(k*b).is_commutative is False

    # 测试 _eval_trace 方法
    assert Tr(OuterProduct(JzKet(1, 1), JzBra(1, 1))).doit() == 1

    # 测试缩放后的 kets 和 bras
    assert OuterProduct(2 * k, b) == 2 * OuterProduct(k, b)
    assert OuterProduct(k, 2 * b) == 2 * OuterProduct(k, b)

    # 测试 kets 和 bras 的求和
    k1, k2 = Ket('k1'), Ket('k2')
    b1, b2 = Bra('b1'), Bra('b2')
    assert (OuterProduct(k1 + k2, b1) ==
            OuterProduct(k1, b1) + OuterProduct(k2, b1))
    assert (OuterProduct(k1, b1 + b2) ==
            OuterProduct(k1, b1) + OuterProduct(k1, b2))
    assert (OuterProduct(1 * k1 + 2 * k2, 3 * b1 + 4 * b2) ==
            3 * OuterProduct(k1, b1) +
            4 * OuterProduct(k1, b2) +
            6 * OuterProduct(k2, b1) +
            8 * OuterProduct(k2, b2))


# 定义一个测试 operator dagger 的函数
def test_operator_dagger():
    # 创建名为 A 和 B 的 Operator 对象
    A = Operator('A')
    B = Operator('B')
    # 断言 Dagger(A*B) 等于 Dagger(B)*Dagger(A)
    assert Dagger(A*B) == Dagger(B)*Dagger(A)
    # 断言 Dagger(A + B) 等于 Dagger(A) + Dagger(B)
    assert Dagger(A + B) == Dagger(A) + Dagger(B)
    # 断言 Dagger(A**2) 等于 Dagger(A)**2
    assert Dagger(A**2) == Dagger(A)**2


# 定义一个测试 differential operator 的函数
def test_differential_operator():
    # 创建符号 x 和函数 f
    x = Symbol('x')
    f = Function('f')
    # 创建 DifferentialOperator 对象 d
    d = DifferentialOperator(Derivative(f(x), x), f(x))
    # 创建 Wavefunction 对象 g
    g = Wavefunction(x**2, x)
    # 断言 qapply(d*g) 等于 Wavefunction(2*x, x)
    assert qapply(d*g) == Wavefunction(2*x, x)
    # 断言 d 的 expr 属性等于 Derivative(f(x), x)
    assert d.expr == Derivative(f(x), x)
    # 断言 d 的 function 属性等于 f(x)
    assert d.function == f(x)
    # 断言 d 的 variables 属性是元组 (x,)
    assert d.variables == (x,)
    # 断言 diff(d, x) 等于 DifferentialOperator(Derivative(f(x), x, 2), f(x))
    assert diff(d, x) == DifferentialOperator(Derivative(f(x), x, 2), f(x))

    # 重置 d 为具有二阶导数的 DifferentialOperator 对象
    d = DifferentialOperator(Derivative(f(x), x, 2), f(x))
    # 重置 g 为 Wavefunction 对象
    g = Wavefunction(x**3, x)
    # 断言 qapply(d*g) 等于 Wavefunction(6*x, x)
    assert qapply(d*g) == Wavefunction(6*x, x)
    # 断言 d 的 expr 属性等于 Derivative(f(x), x, 2)
    assert d.expr == Derivative(f(x), x, 2)
    # 断言 d 的 function 属性等于 f(x)
    assert d.function == f(x)
    # 断言 d 的 variables 属性是元组 (x,)
    assert d.variables == (x,)
    # 断言 diff(d, x) 等于 DifferentialOperator(Derivative(f(x), x, 3), f(x))
    assert diff(d, x) == DifferentialOperator(Derivative(f(x), x, 3), f(x))

    # 重置 d 为具有 1/x*Derivative(f(x), x) 的 DifferentialOperator 对象
    d = DifferentialOperator(1/x*Derivative(f(x), x), f(x))
    # 断言 d 的 expr 属性等于 1/x*Derivative(f(x), x)
    assert d.expr == 1/x*Derivative(f(x), x)
    # 断言 d 的 function 属性等于 f(x)
    assert d.function == f(x)
    # 断言 d 的 variables 属性是元组 (x,)
    assert d.variables == (x,)
    # 断言 diff(d, x) 等于 DifferentialOperator(Derivative(1/x*Derivative(f(x), x), x), f(x))
    assert diff(d, x) == \
        DifferentialOperator(Derivative(1/x*Derivative(f(x), x), x), f(x))
    # 断言 qapply(d*g) 等于 Wavefunction(3*x, x)
    assert qapply(d*g) == Wavefunction(3*x, x)

    # 创建符号 y
    y = Symbol('y')
    # 创建二维笛卡尔 Laplacian 的 DifferentialOperator 对象 d
    d = DifferentialOperator(Derivative(f(x, y), x, 2) +
                             Derivative(f(x, y), y, 2), f(x, y))
    # 创建 Wavefunction 对象 w
    w = Wavefunction(x**3*y**2 + y**3*x**2, x, y)
    # 确认导数操作对象的表达式等于二阶偏导数相加
    assert d.expr == Derivative(f(x, y), x, 2) + Derivative(f(x, y), y, 2)
    
    # 确认导数操作对象的函数等于原始函数 f(x, y)
    assert d.function == f(x, y)
    
    # 确认导数操作对象的变量是元组 (x, y)
    assert d.variables == (x, y)
    
    # 确认对导数操作对象在 x 方向的偏导数操作结果
    assert diff(d, x) == \
        DifferentialOperator(Derivative(d.expr, x), f(x, y))
    
    # 确认对导数操作对象在 y 方向的偏导数操作结果
    assert diff(d, y) == \
        DifferentialOperator(Derivative(d.expr, y), f(x, y))
    
    # 确认对 d*w 的量子应用结果
    assert qapply(d*w) == Wavefunction(2*x**3 + 6*x*y**2 + 6*x**2*y + 2*y**3,
                                       x, y)
    
    # 2维极坐标系下的拉普拉斯算子（th 为 theta）
    # 定义极坐标变量 r 和 th
    r, th = symbols('r th')
    
    # 创建拉普拉斯算子对象 d，包含径向和角向的二阶偏导数
    d = DifferentialOperator(1/r*Derivative(r*Derivative(f(r, th), r), r) +
                             1/(r**2)*Derivative(f(r, th), th, 2), f(r, th))
    
    # 创建波函数 w，描述极坐标系中的波函数形式
    w = Wavefunction(r**2*sin(th), r, (th, 0, pi))
    
    # 确认拉普拉斯算子对象 d 的表达式
    assert d.expr == \
        1/r*Derivative(r*Derivative(f(r, th), r), r) + \
        1/(r**2)*Derivative(f(r, th), th, 2)
    
    # 确认拉普拉斯算子对象 d 的函数是 f(r, th)
    assert d.function == f(r, th)
    
    # 确认拉普拉斯算子对象 d 的变量是元组 (r, th)
    assert d.variables == (r, th)
    
    # 确认对拉普拉斯算子对象 d 在 r 方向的偏导数操作结果
    assert diff(d, r) == \
        DifferentialOperator(Derivative(d.expr, r), f(r, th))
    
    # 确认对拉普拉斯算子对象 d 在 th 方向的偏导数操作结果
    assert diff(d, th) == \
        DifferentialOperator(Derivative(d.expr, th), f(r, th))
    
    # 确认对 d*w 的量子应用结果
    assert qapply(d*w) == Wavefunction(3*sin(th), r, (th, 0, pi))
# 定义一个名为 test_eval_power 的测试函数
def test_eval_power():
    # 从 sympy.core 模块导入 Pow 类
    from sympy.core import Pow
    # 从 sympy.core.expr 模块导入 unchanged 函数
    from sympy.core.expr import unchanged
    # 创建名为 O 的算符对象
    O = Operator('O')
    # 创建名为 U 的单位算符对象
    U = UnitaryOperator('U')
    # 创建名为 H 的厄米算符对象
    H = HermitianOperator('H')
    # 断言 O 的逆等于 O.inv()，这与文档测试一致
    assert O**-1 == O.inv()  # same as doc test
    # 断言 U 的逆等于 U.inv()
    assert U**-1 == U.inv()
    # 断言 H 的逆等于 H.inv()
    assert H**-1 == H.inv()
    # 创建一个符号 x，使其具有可交换属性
    x = symbols("x", commutative=True)
    # 断言 Pow(H, x) 调用 unchanged(Pow, H, x)，验证 Pow(H, x)=="X^n"
    assert unchanged(Pow, H, x)
    # 断言 H 的 x 次方等于 Pow(H, x)
    assert H**x == Pow(H, x)
    # 断言 Pow(H, x) 调用 Pow(H, x, evaluate=False)，只是简单检查
    assert Pow(H, x) == Pow(H, x, evaluate=False)
    # 从 sympy.physics.quantum.gate 模块导入 XGate 类
    from sympy.physics.quantum.gate import XGate
    # 创建一个 XGate 对象 X，索引为 0，该对象既是厄米又是单位的门
    X = XGate(0)
    # 断言 Pow(X, x) 调用 unchanged(Pow, X, x)，验证 Pow(X, x)=="X^x"
    assert unchanged(Pow, X, x)
    # 断言 X 的 x 次方等于 Pow(X, x)
    assert X**x == Pow(X, x)
    # 断言 Pow(X, x, evaluate=False) 等于 Pow(X, x)，只是简单检查
    assert Pow(X, x, evaluate=False) == Pow(X, x)
    # 创建一个符号 n，使其为偶数整数
    n = symbols("n", integer=True, even=True)
    # 断言 X 的 n 次方等于 1
    assert X**n == 1
    # 创建一个符号 n，使其为奇数整数
    n = symbols("n", integer=True, odd=True)
    # 断言 X 的 n 次方等于 X
    assert X**n == X
    # 创建一个符号 n，使其为整数
    n = symbols("n", integer=True)
    # 断言 Pow(X, n) 调用 unchanged(Pow, X, n)，验证 Pow(X, n)=="X^n"
    assert unchanged(Pow, X, n)
    # 断言 X 的 n 次方等于 Pow(X, n)
    assert X**n == Pow(X, n)
    # 断言 Pow(X, n, evaluate=False) 等于 Pow(X, n)，只是简单检查
    assert Pow(X, n, evaluate=False) == Pow(X, n)
    # 断言 X 的 4 次方等于 1
    assert X**4 == 1
    # 断言 X 的 7 次方等于 X
    assert X**7 == X
```