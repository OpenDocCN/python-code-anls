# `D:\src\scipysrc\sympy\sympy\physics\quantum\tests\test_state.py`

```
# 导入SymPy库中的特定模块和函数

from sympy.core.add import Add
from sympy.core.function import diff
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer, Rational, oo, pi)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.testing.pytest import raises

# 导入量子力学相关的SymPy库模块

from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.qexpr import QExpr
from sympy.physics.quantum.state import (
    Ket, Bra, TimeDepKet, TimeDepBra,
    KetBase, BraBase, StateBase, Wavefunction,
    OrthogonalKet, OrthogonalBra
)
from sympy.physics.quantum.hilbert import HilbertSpace

# 定义符号变量 x, y, t
x, y, t = symbols('x,y,t')

# 自定义子类 CustomKet 继承自 Ket
class CustomKet(Ket):
    @classmethod
    def default_args(self):
        return ("test",)

# 自定义子类 CustomKetMultipleLabels 继承自 Ket
class CustomKetMultipleLabels(Ket):
    @classmethod
    def default_args(self):
        return ("r", "theta", "phi")

# 自定义子类 CustomTimeDepKet 继承自 TimeDepKet
class CustomTimeDepKet(TimeDepKet):
    @classmethod
    def default_args(self):
        return ("test", "t")

# 自定义子类 CustomTimeDepKetMultipleLabels 继承自 TimeDepKet
class CustomTimeDepKetMultipleLabels(TimeDepKet):
    @classmethod
    def default_args(self):
        return ("r", "theta", "phi", "t")

# 测试函数 test_ket()
def test_ket():
    # 创建一个标签为 '0' 的 Ket 对象 k
    k = Ket('0')

    # 断言 k 是 Ket 类的实例
    assert isinstance(k, Ket)
    # 断言 k 是 KetBase 类的实例
    assert isinstance(k, KetBase)
    # 断言 k 是 StateBase 类的实例
    assert isinstance(k, StateBase)
    # 断言 k 是 QExpr 类的实例
    assert isinstance(k, QExpr)

    # 断言 k 的标签是 (Symbol('0'),)
    assert k.label == (Symbol('0'),)
    # 断言 k 的希尔伯特空间是 HilbertSpace()
    assert k.hilbert_space == HilbertSpace()
    # 断言 k 不可交换
    assert k.is_commutative is False

    # 创建一个标签为 'pi' 的 Ket 对象 k
    k = Ket('pi')
    # 断言 k 的标签是 (Symbol('pi'),)
    assert k.label == (Symbol('pi'),)

    # 创建一个带有变量 x 和 y 的 Ket 对象 k
    k = Ket(x, y)
    # 断言 k 的标签是 (x, y)
    assert k.label == (x, y)
    # 断言 k 的希尔伯特空间是 HilbertSpace()
    assert k.hilbert_space == HilbertSpace()
    # 断言 k 不可交换
    assert k.is_commutative is False

    # 断言 k 的对偶类是 Bra
    assert k.dual_class() == Bra
    # 断言 k 的对偶是 Bra(x, y)
    assert k.dual == Bra(x, y)
    # 断言 k 在变量 x 上代换为 y 后是 Ket(y, y)
    assert k.subs(x, y) == Ket(y, y)

    # 创建一个自定义的 CustomKet 对象 k
    k = CustomKet()
    # 断言 k 等于 CustomKet("test")
    assert k == CustomKet("test")

    # 创建一个自定义的 CustomKetMultipleLabels 对象 k
    k = CustomKetMultipleLabels()
    # 断言 k 等于 CustomKetMultipleLabels("r", "theta", "phi")
    assert k == CustomKetMultipleLabels("r", "theta", "phi")

    # 断言 Ket() 等于 Ket('psi')
    assert Ket() == Ket('psi')

# 测试函数 test_bra()
def test_bra():
    # 创建一个标签为 '0' 的 Bra 对象 b
    b = Bra('0')

    # 断言 b 是 Bra 类的实例
    assert isinstance(b, Bra)
    # 断言 b 是 BraBase 类的实例
    assert isinstance(b, BraBase)
    # 断言 b 是 StateBase 类的实例
    assert isinstance(b, StateBase)
    # 断言 b 是 QExpr 类的实例
    assert isinstance(b, QExpr)

    # 断言 b 的标签是 (Symbol('0'),)
    assert b.label == (Symbol('0'),)
    # 断言 b 的希尔伯特空间是 HilbertSpace()
    assert b.hilbert_space == HilbertSpace()
    # 断言 b 不可交换
    assert b.is_commutative is False

    # 创建一个标签为 'pi' 的 Bra 对象 b
    b = Bra('pi')
    # 断言 b 的标签是 (Symbol('pi'),)
    assert b.label == (Symbol('pi'),)

    # 创建一个带有变量 x 和 y 的 Bra 对象 b
    b = Bra(x, y)
    # 断言 b 的标签是 (x, y)
    assert b.label == (x, y)
    # 断言 b 的希尔伯特空间是 HilbertSpace()
    assert b.hilbert_space == HilbertSpace()
    # 断言 b 不可交换
    assert b.is_commutative is False

    # 断言 b 的对偶类是 Ket
    assert b.dual_class() == Ket
    # 断言 b 的对偶是 Ket(x, y)
    assert b.dual == Ket(x, y)
    # 断言 b 在变量 x 上代换为 y 后是 Bra(y, y)
    assert b.subs(x, y) == Bra(y, y)

    # 断言 Bra() 等于 Bra('psi')

# 测试函数 test_ops()
def test_ops():
    # 创建标签为 0 的 Ket 对象 k0
    k0 = Ket(0)
    # 创建标签为 1 的 Ket 对象 k1
    k1 = Ket(1)
    # 创建表达式 k = 2*I*k0 - (x/sqrt(2))*k1
    k = 2*I*k0 - (x/sqrt(2))*k1
    # 断言 k 等于 Add(Mul(2, I, k0), Mul(Rational(-1, 2), x, Pow(2, S.Half), k1))
    assert k == Add(Mul(2, I, k0),
        Mul(Rational(-1, 2), x, Pow(2, S.Half), k1))
`
def test_time_dep_ket():
    # 创建一个时间依赖的态向量，时间参数为 t
    k = TimeDepKet(0, t)

    # 确认 k 是 TimeDepKet 类的实例
    assert isinstance(k, TimeDepKet)
    # 确认 k 是 KetBase 类的实例
    assert isinstance(k, KetBase)
    # 确认 k 是 StateBase 类的实例
    assert isinstance(k, StateBase)
    # 确认 k 是 QExpr 类的实例
    assert isinstance(k, QExpr)

    # 确认 k 的 label 属性等于 (Integer(0),)
    assert k.label == (Integer(0),)
    # 确认 k 的 args 属性等于 (Integer(0), t)
    assert k.args == (Integer(0), t)
    # 确认 k 的 time 属性等于 t
    assert k.time == t

    # 确认 k 的对偶类是 TimeDepBra
    assert k.dual_class() == TimeDepBra
    # 确认 k 的对偶是 TimeDepBra(0, t)
    assert k.dual == TimeDepBra(0, t)

    # 确认将 t 替换为 2 后，k 的结果是 TimeDepKet(0, 2)
    assert k.subs(t, 2) == TimeDepKet(0, 2)

    # 创建一个新的时间依赖的态向量，x 为 label，0.5 为时间参数
    k = TimeDepKet(x, 0.5)
    # 确认 k 的 label 属性等于 (x,)
    assert k.label == (x,)
    # 确认 k 的 args 属性等于 (x, sympify(0.5))
    assert k.args == (x, sympify(0.5))

    # 创建一个自定义时间依赖的态向量，默认 label 为 "test"，时间参数为 "t"
    k = CustomTimeDepKet()
    # 确认 k 的 label 属性等于 (Symbol("test"),)
    assert k.label == (Symbol("test"),)
    # 确认 k 的 time 属性等于 Symbol("t")
    assert k.time == Symbol("t")
    # 确认 k 等于 CustomTimeDepKet("test", "t")
    assert k == CustomTimeDepKet("test", "t")

    # 创建一个自定义时间依赖的态向量，多个 label，时间参数为 "t"
    k = CustomTimeDepKetMultipleLabels()
    # 确认 k 的 label 属性等于 (Symbol("r"), Symbol("theta"), Symbol("phi"))
    assert k.label == (Symbol("r"), Symbol("theta"), Symbol("phi"))
    # 确认 k 的 time 属性等于 Symbol("t")
    assert k.time == Symbol("t")
    # 确认 k 等于 CustomTimeDepKetMultipleLabels("r", "theta", "phi", "t")
    assert k == CustomTimeDepKetMultipleLabels("r", "theta", "phi", "t")

    # 确认 TimeDepKet() 等于 TimeDepKet("psi", "t")
    assert TimeDepKet() == TimeDepKet("psi", "t")


def test_time_dep_bra():
    # 创建一个时间依赖的bra，时间参数为 t
    b = TimeDepBra(0, t)

    # 确认 b 是 TimeDepBra 类的实例
    assert isinstance(b, TimeDepBra)
    # 确认 b 是 BraBase 类的实例
    assert isinstance(b, BraBase)
    # 确认 b 是 StateBase 类的实例
    assert isinstance(b, StateBase)
    # 确认 b 是 QExpr 类的实例
    assert isinstance(b, QExpr)

    # 确认 b 的 label 属性等于 (Integer(0),)
    assert b.label == (Integer(0),)
    # 确认 b 的 args 属性等于 (Integer(0), t)
    assert b.args == (Integer(0), t)
    # 确认 b 的 time 属性等于 t
    assert b.time == t

    # 确认 b 的对偶类是 TimeDepKet
    assert b.dual_class() == TimeDepKet
    # 确认 b 的对偶是 TimeDepKet(0, t)
    assert b.dual == TimeDepKet(0, t)

    # 创建一个新的时间依赖的bra，x 为 label，0.5 为时间参数
    k = TimeDepBra(x, 0.5)
    # 确认 k 的 label 属性等于 (x,)
    assert k.label == (x,)
    # 确认 k 的 args 属性等于 (x, sympify(0.5))
    assert k.args == (x, sympify(0.5))

    # 确认 TimeDepBra() 等于 TimeDepBra("psi", "t")
    assert TimeDepBra() == TimeDepBra("psi", "t")


def test_bra_ket_dagger():
    # 定义一个复数符号 x
    x = symbols('x', complex=True)
    # 创建一个 Ket 对象 k，label 为 'k'
    k = Ket('k')
    # 创建一个 Bra 对象 b，label 为 'b'
    b = Bra('b')
    # 确认 Dagger(k) 等于 Bra('k')
    assert Dagger(k) == Bra('k')
    # 确认 Dagger(b) 等于 Ket('b')
    assert Dagger(b) == Ket('b')
    # 确认 Dagger(k).is_commutative 为 False
    assert Dagger(k).is_commutative is False

    # 创建一个 Ket 对象 k2，label 为 'k2'
    k2 = Ket('k2')
    # 定义表达式 e = 2*I*k + x*k2
    e = 2*I*k + x*k2
    # 确认 Dagger(e) 等于 conjugate(x)*Dagger(k2) - 2*I*Dagger(k)
    assert Dagger(e) == conjugate(x)*Dagger(k2) - 2*I*Dagger(k)


def test_wavefunction():
    # 定义实数符号 x 和 y
    x, y = symbols('x y', real=True)
    # 定义正实数符号 L
    L = symbols('L', positive=True)
    # 定义正整数符号 n
    n = symbols('n', integer=True, positive=True)

    # 创建一个 Wavefunction 对象 f，表达式为 x**2，变量为 x
    f = Wavefunction(x**2, x)
    # 计算 f 的概率密度
    p = f.prob()
    # 获取 f 的积分范围限制
    lims = f.limits

    # 确认 f.is_normalized 为 False
    assert f.is_normalized is False
    # 确认 f.norm 为无穷大
    assert f.norm is oo
    # 确认 f(10) 的值为 100
    assert f(10) == 100
    # 确认 p(10) 的值为 10000
    assert p(10) == 10000
    # 确认 lims[x] 等于 (-oo, oo)
    assert lims[x] == (-oo, oo)
    # 确认 f 对 x 的一阶导数等于 Wavefunction(2*x, x)
    assert diff(f, x) == Wavefunction(2*x, x)
    # 确认调用 f.normalize() 会抛出 NotImplementedError 异常
    raises(NotImplementedError, lambda: f.normalize())
    # 确认 f 的共轭态向量等于 Wavefunction(conjugate(f.expr), x)
    assert conjugate(f) == Wavefunction(conjugate(f.expr), x)
    # 确认 f 的共轭态向量等于 Dagger(f)
    assert conjugate(f) == Dagger(f)

    # 创建一个 Wavefunction 对象 g，表达式为 x**2*y + y**2*x，变量为 (x, 0, 1) 和 (y, 0, 2)
    g = Wavefunction(x**2*y + y**2*x, (x, 0, 1), (y, 0, 2))
    # 获取 g 的积分范围限制
    lims_g = g.limits

    # 确认 lims_g[x] 等于 (0, 1)
    assert lims_g[x] == (0, 1)
    # 确认 lims_g[y] 等于 (0, 2)
    assert lims_g[y] == (0, 2)
    # 确认 g.is_normalized 为 False
    assert g.is_normalized is False
    # 确认 g.norm 等于 sqrt(42)/3
    assert g.norm == sqrt(42)/3
    # 确认 g(2, 4) 的值为 0
    assert g(2, 4) == 0

# 定义测试函数，用于测试 TimeDepKet 类
def test_time_dep_ket():
    # 创建一个 TimeDepKet 对象 k，参数为整数 0 和变量 t
    k = TimeDepKet(0, t)

    # 断言 k 是 TimeDepKet 类的实例
    assert isinstance(k, TimeDepKet)
    # 断言 k 是 KetBase 类的实例
    assert isinstance(k, KetBase)
    # 断言 k 是 StateBase 类的实例
    assert isinstance(k, StateBase)
    # 断言 k 是 QExpr 类的实例
    assert isinstance(k, QExpr)

    # 断言 k 的标签为元组 (Integer(0),)
    assert k.label == (Integer(0),)
    # 断言 k 的参数为元组 (Integer(0), t)
    assert k.args == (Integer(0), t)
    # 断言 k 的时间参数为 t
    assert k.time == t

    # 断言 k 的对偶类为 TimeDepBra
    assert k.dual_class() == TimeDepBra
    # 断言 k 的对偶对象为 TimeDepBra(0, t)
    assert k.dual == TimeDepBra(0, t)

    # 对 k 调用 subs 方法，将 t 替换为整数 2，预期返回 TimeDepKet(0, 2)
    assert k.subs(t, 2) == TimeDepKet(0, 2)

    # 创建一个新的 TimeDepKet 对象 k，参数为符号 x 和浮点数 0.5
    k = TimeDepKet(x, 0.5)
    # 断言 k 的标签为元组 (x,)
    assert k.label == (x,)
    # 断言 k 的参数为元组 (x, sympify(0.5))
    assert k.args == (x, sympify(0.5))

    # 创建一个自定义的 CustomTimeDepKet 对象 k
    k = CustomTimeDepKet()
    # 断言 k 的标签为元组 (Symbol("test"),)
    assert k.label == (Symbol("test"),)
    # 断言 k 的时间参数为符号 "t"
    assert k.time == Symbol("t")
    # 断言 k 与 CustomTimeDepKet("test", "t") 相等
    assert k == CustomTimeDepKet("test", "t")

    # 创建一个自定义的 CustomTimeDepKetMultipleLabels 对象 k
    k = CustomTimeDepKetMultipleLabels()
    # 断言 k 的标签为元组 (Symbol("r"), Symbol("theta"), Symbol("phi"))
    assert k.label == (Symbol("r"), Symbol("theta"), Symbol("phi"))
    # 断言 k 的时间参数为符号 "t"
    assert k.time == Symbol("t")
    # 断言 k 与 CustomTimeDepKetMultipleLabels("r", "theta", "phi", "t") 相等
    assert k == CustomTimeDepKetMultipleLabels("r", "theta", "phi", "t")

    # 断言 TimeDepKet() 等价于 TimeDepKet("psi", "t")
    assert TimeDepKet() == TimeDepKet("psi", "t")


# 定义测试函数，用于测试 TimeDepBra 类
def test_time_dep_bra():
   ```python
# 定义测试函数，用于测试 TimeDepKet 类
def test_time_dep_ket():
    # 创建一个 TimeDepKet 对象 k，参数为整数 0 和变量 t
    k = TimeDepKet(0, t)

    # 断言 k 是 TimeDepKet 类的实例
    assert isinstance(k, TimeDepKet)
    # 断言 k 是 KetBase 类的实例
    assert isinstance(k, KetBase)
    # 断言 k 是 StateBase 类的实例
    assert isinstance(k, StateBase)
    # 断言 k 是 QExpr 类的实例
    assert isinstance(k, QExpr)

    # 断言 k 的标签为元组 (Integer(0),)
    assert k.label == (Integer(0),)
    # 断言 k 的参数为元组 (Integer(0), t)
    assert k.args == (Integer(0), t)
    # 断言 k 的时间参数为 t
    assert k.time == t

    # 断言 k 的对偶类为 TimeDepBra
    assert k.dual_class() == TimeDepBra
    # 断言 k 的对偶对象为 TimeDepBra(0, t)
    assert k.dual == TimeDepBra(0, t)

    # 对 k 调用 subs 方法，将 t 替换为整数 2，预期返回 TimeDepKet(0, 2)
    assert k.subs(t, 2) == TimeDepKet(0, 2)

    # 创建一个新的 TimeDepKet 对象 k，参数为符号 x 和浮点数 0.5
    k = TimeDepKet(x, 0.5)
    # 断言 k 的标签为元组 (x,)
    assert k.label == (x,)
    # 断言 k 的参数为元组 (x, sympify(0.5))
    assert k.args == (x, sympify(0.5))

    # 创建一个自定义的 CustomTimeDepKet 对象 k
    k = CustomTimeDepKet()
    # 断言 k 的标签为元组 (Symbol("test"),)
    assert k.label == (Symbol("test"),)
    # 断言 k 的时间参数为符号 "t"
    assert k.time == Symbol("t")
    # 断言 k 与 CustomTimeDepKet("test", "t") 相等
    assert k == CustomTimeDepKet("test", "t")

    # 创建一个自定义的 CustomTimeDepKetMultipleLabels 对象 k
    k = CustomTimeDepKetMultipleLabels()
    # 断言 k 的标签为元组 (Symbol("r"), Symbol("theta"), Symbol("phi"))
    assert k.label == (Symbol("r"), Symbol("theta"), Symbol("phi"))
    # 断言 k 的时间参数为符号 "t"
    assert k.time == Symbol("t")
    # 断言 k 与 CustomTimeDepKetMultipleLabels("r", "theta", "phi", "t") 相等
    assert k == CustomTimeDepKetMultipleLabels("r", "theta", "phi", "t")

    # 断言 TimeDepKet() 等价于 TimeDepKet("psi", "t")
    assert TimeDepKet() == TimeDepKet("psi", "t")


# 定义测试函数，用于测试 TimeDepBra 类
def test_time_dep_bra():
    # 创建一个 TimeDepBra 对象 b，参数为整数 0 和变量 t
    b = TimeDepBra(0, t)

    # 断言 b 是 TimeDepBra 类的实例
    assert isinstance(b, TimeDepBra)
    # 断言 b 是 BraBase 类的实例
    assert isinstance(b, BraBase)
    # 断言 b 是 StateBase 类的实例
    assert isinstance(b, StateBase)
    # 断言 b 是 QExpr 类的实例
    assert isinstance(b, QExpr)

    # 断言 b 的标签为元组 (Integer(0),)
    assert b.label == (Integer(0),)
    # 断言 b 的参数为元组 (Integer(0), t)
    assert b.args == (Integer(0), t)
    # 断言 b 的时间参数为 t
    assert b.time == t

    # 断言 b 的对偶类为 TimeDepKet
    assert b.dual_class() == TimeDepKet
    # 断言 b 的对偶对象为 TimeDepKet(0, t)
    assert b.dual == TimeDepKet(0, t)

    # 创建一个新的 TimeDepBra 对象 k，参数为符号 x 和浮点数 0.5
    k = TimeDepBra(x, 0.5)
    # 断言 k 的标签为元组 (x,)
    assert k.label == (x,)
    # 断言 k 的参数为元组 (x, sympify(0.5))
    assert k.args == (x, sympify(0.5))

    # 断言 TimeDepBra() 等价于 TimeDepBra("psi", "t")
    assert TimeDepBra() == TimeDepBra("psi", "t")


# 定义测试函数，用于测试 Bra 和 Ket 的共轭操作
def test_bra_ket_dagger():
    # 定义一个复数符号 x
    x = symbols('x', complex=True)
    # 创建一个 Ket 对象 k
    k = Ket('k')
    # 创建一个 Bra 对象 b
    b = Bra('b')
    # 断言 Dagger(k) 等价于 Bra('k')
    assert Dagger(k) == Bra('k')
    # 断言 Dagger(b) 等价于 Ket('b')
    assert Dagger(b) == Ket('b')
    # 断言 Dagger(k) 的交换性为 False
    assert Dagger(k).is_commutative is False

    # 创建一个新的 Ket 对象 k2
    k2 = Ket('k2')
    # 定义表达式 e
    e = 2*I*k + x*k2
    # 断言 Dagger(e) 等价于 conjugate(x)*Dagger(k2) - 2*I*Dagger(k)
    assert Dagger(e) == conjugate(x)*Dagger(k2) - 2*I*Dagger(k)


# 定义测试函数，用于测试 Wavefunction 类
def test_wavefunction():
    # 定义实数符号 x 和 y
    x, y = symbols('x y', real=True)
    # 定义正实数符号 L
    L = symbols('L', positive=True)
    # 定义正整数符号 n
    n = symbols('n', integer=True, positive=True)

    # 创建 Wavefunction 对象 f，参数为 x**2 和 x
    f = Wavefunction(x**2, x)
    # 调用 f 的 prob 方法，赋值给 p
    p = f.prob()
    # 获取 f 的 limits 属性，赋值给 lims
    lims = f.limits

    # 断言 f 是否归一化为 False
    assert f.is_normalized is False
    # 断言 f 的归一化常数为无穷大
    assert f.norm is oo
    # 断言 f 在 x=10 处的值为 100
    assert f(10) == 100
    # 断言 p 在 x=10 处的值为 10000
    assert p(10) == 10000
    # 断言 lims[x] 的取值范围为 (-oo, oo)
    assert lims[x] == (-oo, oo)
    # 断言 f 对 x 的一阶导数为 Wavefunction(2*x, x)
    assert diff(f, x) == Wavefunction(2*x, x)
    # 断言调用 f 的 normalize 方法会引发 NotImplementedError 异常
    raises(NotImplementedError, lambda: f.normalize())
    # 断言 f 的共轭等于 Wavefunction(conjugate(f.expr), x)
    assert conjugate(f) == Wavefunction(conjugate(f.expr), x)
    # 断言 f 的共轭等于 Dagger(f)
    assert conjugate(f) == Dagger(f)

    # 创建 Wavefunction 对象 g，参数为 x**2*y + y**2*x 和 (x, 0, 1), (y, 0, 2)
    g = Wavefunction(x**2*y + y**2*x, (x, 0, 1), (y, 0, 2))
    # 获取 g 的 limits 属性，赋值给 lims_g
    lims_g = g.limits

    # 断言 lims_g[x] 的取值范围为 (0, 1)
    assert lims_g[x] == (0, 1)
    # 断言 lims_g
    # 创建一个包含正弦函数的波函数对象 `piab`
    piab = Wavefunction(sin(n*pi*x/L), (x, 0, L))
    
    # 断言：波函数对象 `piab` 的归一化常数是否等于 sqrt(L/2)
    assert piab.norm == sqrt(L/2)
    
    # 断言：在 x = L + 1 处，波函数对象 `piab` 的取值是否为 0
    assert piab(L + 1) == 0
    
    # 断言：在 x = 0.5 处，波函数对象 `piab` 的取值是否等于 sin(0.5*n*pi/L)
    assert piab(0.5) == sin(0.5*n*pi/L)
    
    # 断言：在 x = 0.5、n = 1、L = 1 的条件下，波函数对象 `piab` 的取值是否等于 sin(0.5*pi)
    assert piab(0.5, n=1, L=1) == sin(0.5*pi)
    
    # 断言：对波函数对象 `piab` 进行归一化后，是否得到指定的归一化结果
    assert piab.normalize() == Wavefunction(sqrt(2)/sqrt(L)*sin(n*pi*x/L), (x, 0, L))
    
    # 断言：波函数对象 `piab` 的共轭是否等于其表达式的共轭，并构成一个新的波函数对象
    assert conjugate(piab) == Wavefunction(conjugate(piab.expr), (x, 0, L))
    
    # 断言：波函数对象 `piab` 的共轭是否等于其 Dagger（Hermitian 共轭）
    assert conjugate(piab) == Dagger(piab)
    
    # 创建一个包含 x^2 函数的波函数对象 `k`
    k = Wavefunction(x**2, 'x')
    
    # 断言：波函数对象 `k` 的第一个变量是否是符号对象 `Symbol`
    assert type(k.variables[0]) == Symbol
# 定义测试函数，用于验证正交态的性质
def test_orthogonal_states():
    # 创建正交态的内积对象，乘积中使用相同的参数 x
    braket = OrthogonalBra(x) * OrthogonalKet(x)
    # 断言内积计算结果为 1
    assert braket.doit() == 1

    # 创建正交态的内积对象，其中使用参数 x 和 x+1
    braket = OrthogonalBra(x) * OrthogonalKet(x+1)
    # 断言内积计算结果为 0，因为正交态的内积应为 0
    assert braket.doit() == 0

    # 创建正交态的内积对象，分别使用参数 x 和 y
    braket = OrthogonalBra(x) * OrthogonalKet(y)
    # 断言内积计算结果应为原始内积对象，因为 x 和 y 可能不正交
    assert braket.doit() == braket
```