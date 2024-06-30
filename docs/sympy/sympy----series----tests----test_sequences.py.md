# `D:\src\scipysrc\sympy\sympy\series\tests\test_sequences.py`

```
from sympy.core.containers import Tuple  # 导入 Tuple 类
from sympy.core.function import Function  # 导入 Function 类
from sympy.core.numbers import oo, Rational  # 导入 oo 和 Rational
from sympy.core.singleton import S  # 导入 S 单例
from sympy.core.symbol import symbols, Symbol  # 导入 symbols 和 Symbol
from sympy.functions.combinatorial.numbers import tribonacci, fibonacci  # 导入 tribonacci 和 fibonacci 函数
from sympy.functions.elementary.exponential import exp  # 导入 exp 函数
from sympy.functions.elementary.miscellaneous import sqrt  # 导入 sqrt 函数
from sympy.functions.elementary.trigonometric import cos, sin  # 导入 cos 和 sin 函数
from sympy.series import EmptySequence  # 导入 EmptySequence 类
from sympy.series.sequences import (SeqMul, SeqAdd, SeqPer, SeqFormula,  # 导入不同的序列类
    sequence)
from sympy.sets.sets import Interval  # 导入 Interval 类
from sympy.tensor.indexed import Indexed, Idx  # 导入 Indexed 和 Idx 类
from sympy.series.sequences import SeqExpr, SeqExprOp, RecursiveSeq  # 导入 SeqExpr 等序列类
from sympy.testing.pytest import raises, slow  # 导入 raises 和 slow 函数

x, y, z = symbols('x y z')  # 创建符号变量 x, y, z
n, m = symbols('n m')  # 创建符号变量 n, m

def test_EmptySequence():
    # 测试 S.EmptySequence 是否等于 EmptySequence
    assert S.EmptySequence is EmptySequence

    # 测试 S.EmptySequence 的 interval 是否为 S.EmptySet
    assert S.EmptySequence.interval is S.EmptySet
    # 测试 S.EmptySequence 的 length 是否为 S.Zero
    assert S.EmptySequence.length is S.Zero

    # 测试将 S.EmptySequence 转换为列表是否为空列表
    assert list(S.EmptySequence) == []


def test_SeqExpr():
    # 创建 SeqExpr 对象 s，使用 Tuple(1, n, y) 作为 gen，Tuple(x, 0, 10) 作为 interval
    s = SeqExpr(Tuple(1, n, y), Tuple(x, 0, 10))

    assert isinstance(s, SeqExpr)  # 断言 s 是 SeqExpr 类的实例
    assert s.gen == (1, n, y)  # 断言 s 的 gen 属性等于 (1, n, y)
    assert s.interval == Interval(0, 10)  # 断言 s 的 interval 属性等于 Interval(0, 10)
    assert s.start == 0  # 断言 s 的 start 属性为 0
    assert s.stop == 10  # 断言 s 的 stop 属性为 10
    assert s.length == 11  # 断言 s 的 length 属性为 11
    assert s.variables == (x,)  # 断言 s 的 variables 属性为 (x,)

    # 测试另一个 SeqExpr 对象，使用 Tuple(1, 2, 3) 作为 gen，Tuple(x, 0, oo) 作为 interval，其 length 是否为 oo
    assert SeqExpr(Tuple(1, 2, 3), Tuple(x, 0, oo)).length is oo


def test_SeqPer():
    # 创建 SeqPer 对象 s，使用 Tuple(1, n, 3) 作为 periodical，Tuple(x, 0, 5) 作为 interval
    s = SeqPer((1, n, 3), (x, 0, 5))

    assert isinstance(s, SeqPer)  # 断言 s 是 SeqPer 类的实例
    assert s.periodical == Tuple(1, n, 3)  # 断言 s 的 periodical 属性等于 Tuple(1, n, 3)
    assert s.period == 3  # 断言 s 的 period 属性为 3
    assert s.coeff(3) == 1  # 断言 s 的 coeff(3) 方法返回 1
    assert s.free_symbols == {n}  # 断言 s 的 free_symbols 属性为 {n}

    # 断言将 s 转换为列表后是否与预期列表相同
    assert list(s) == [1, n, 3, 1, n, 3]
    assert s[:] == [1, n, 3, 1, n, 3]
    assert SeqPer((1, n, 3), (x, -oo, 0))[0:6] == [1, n, 3, 1, n, 3]

    # 测试异常情况下的行为
    raises(ValueError, lambda: SeqPer((1, 2, 3), (0, 1, 2)))
    raises(ValueError, lambda: SeqPer((1, 2, 3), (x, -oo, oo)))
    raises(ValueError, lambda: SeqPer(n**2, (0, oo)))

    # 测试不同参数组合下的行为
    assert SeqPer((n, n**2, n**3), (m, 0, oo))[:6] == [n, n**2, n**3, n, n**2, n**3]
    assert SeqPer((n, n**2, n**3), (n, 0, oo))[:6] == [0, 1, 8, 3, 16, 125]
    assert SeqPer((n, m), (n, 0, oo))[:6] == [0, m, 2, m, 4, m]


def test_SeqFormula():
    # 创建 SeqFormula 对象 s，使用 n**2 作为 formula，(n, 0, 5) 作为 interval
    s = SeqFormula(n**2, (n, 0, 5))

    assert isinstance(s, SeqFormula)  # 断言 s 是 SeqFormula 类的实例
    assert s.formula == n**2  # 断言 s 的 formula 属性等于 n**2
    assert s.coeff(3) == 9  # 断言 s 的 coeff(3) 方法返回 9

    # 断言将 s 转换为列表后是否与预期列表相同
    assert list(s) == [i**2 for i in range(6)]
    assert s[:] == [i**2 for i in range(6)]
    assert SeqFormula(n**2, (n, -oo, 0))[0:6] == [i**2 for i in range(6)]

    # 测试相同 formula 不同 interval 的行为
    assert SeqFormula(n**2, (0, oo)) == SeqFormula(n**2, (n, 0, oo))

    # 测试 subs 方法的行为
    assert SeqFormula(n**2, (0, m)).subs(m, x) == SeqFormula(n**2, (0, x))
    assert SeqFormula(m*n**2, (n, 0, oo)).subs(m, x) == SeqFormula(x*n**2, (n, 0, oo))

    # 测试异常情况下的行为
    raises(ValueError, lambda: SeqFormula(n**2, (0, 1, 2)))
    raises(ValueError, lambda: SeqFormula(n**2, (n, -oo, oo)))


这段代码是对 SymPy 库中序列相关类的测试代码，包括 EmptySequence、SeqExpr、SeqPer 和 SeqFormula 等。注释详细解释了每个测试函数中的操作和断言。
    # 使用 raises 函数测试是否会引发 ValueError 异常，lambda 函数用于传递要测试的表达式
    raises(ValueError, lambda: SeqFormula(m*n**2, (0, oo)))

    # 创建一个 SeqFormula 对象，表达式为 x*(y**2 + z)，其中 z 在范围 1 到 100
    seq = SeqFormula(x*(y**2 + z), (z, 1, 100))
    # 断言展开后的序列与预期的 SeqFormula 对象相等
    assert seq.expand() == SeqFormula(x*y**2 + x*z, (z, 1, 100))
    
    # 将表达式 sin(x*(y**2 + z)) 创建为 SeqFormula 对象
    seq = SeqFormula(sin(x*(y**2 + z)), (z, 1, 100))
    # 断言开启三角函数展开后的序列与预期的 SeqFormula 对象相等
    assert seq.expand(trig=True) == SeqFormula(sin(x*y**2)*cos(x*z) + sin(x*z)*cos(x*y**2), (z, 1, 100))
    # 断言未开启三角函数展开后的序列与预期的 SeqFormula 对象相等
    assert seq.expand() == SeqFormula(sin(x*y**2 + x*z), (z, 1, 100))
    # 断言关闭三角函数展开后的序列与预期的 SeqFormula 对象相等
    assert seq.expand(trig=False) == SeqFormula(sin(x*y**2 + x*z), (z, 1, 100))
    
    # 将表达式 exp(x*(y**2 + z)) 创建为 SeqFormula 对象
    seq = SeqFormula(exp(x*(y**2 + z)), (z, 1, 100))
    # 断言展开后的指数函数与预期的 SeqFormula 对象相等
    assert seq.expand() == SeqFormula(exp(x*y**2)*exp(x*z), (z, 1, 100))
    # 断言未开启指数展开后的序列与预期的 SeqFormula 对象相等
    assert seq.expand(power_exp=False) == SeqFormula(exp(x*y**2 + x*z), (z, 1, 100))
    # 断言同时关闭乘法展开和指数展开后的序列与预期的 SeqFormula 对象相等
    assert seq.expand(mul=False, power_exp=False) == SeqFormula(exp(x*(y**2 + z)), (z, 1, 100))
def test_sequence():
    # 创建 SeqFormula 对象，表示序列 n**2 在 n 从 0 到 5 的范围内的表达式
    form = SeqFormula(n**2, (n, 0, 5))
    # 创建 SeqPer 对象，表示序列 (1, 2, 3) 在 n 从 0 到 5 的范围内的周期序列
    per = SeqPer((1, 2, 3), (n, 0, 5))
    # 创建 SeqFormula 对象，表示序列 n**2 的一般形式
    inter = SeqFormula(n**2)

    # 断言检查 sequence 函数的返回值是否符合预期
    assert sequence(n**2, (n, 0, 5)) == form
    assert sequence((1, 2, 3), (n, 0, 5)) == per
    assert sequence(n**2) == inter


def test_SeqExprOp():
    # 创建 SeqFormula 对象，表示序列 n**2 在 n 从 0 到 10 的范围内的表达式
    form = SeqFormula(n**2, (n, 0, 10))
    # 创建 SeqPer 对象，表示序列 (1, 2, 3) 在 m 从 5 到 10 的范围内的周期序列
    per = SeqPer((1, 2, 3), (m, 5, 10))

    # 创建 SeqExprOp 对象，表示在 form 和 per 之间的序列表达式操作
    s = SeqExprOp(form, per)
    # 断言检查 SeqExprOp 对象的各个属性是否符合预期
    assert s.gen == (n**2, (1, 2, 3))
    assert s.interval == Interval(5, 10)
    assert s.start == 5
    assert s.stop == 10
    assert s.length == 6
    assert s.variables == (n, m)


def test_SeqAdd():
    # 创建 SeqPer 对象，表示序列 (1, 2, 3) 在 n 从 0 到 ∞ 的范围内的周期序列
    per = SeqPer((1, 2, 3), (n, 0, oo))
    # 创建 SeqFormula 对象，表示序列 n**2 的一般形式
    form = SeqFormula(n**2)

    # 创建 SeqPer 对象，表示序列 (1, 2) 在 n 从 1 到 5 的范围内的周期序列
    per_bou = SeqPer((1, 2), (n, 1, 5))
    # 创建 SeqFormula 对象，表示序列 n**2 在 (6, 10) 范围内的表达式
    form_bou = SeqFormula(n**2, (6, 10))
    # 创建 SeqFormula 对象，表示序列 n**2 在 (1, 5) 范围内的表达式
    form_bou2 = SeqFormula(n**2, (1, 5))

    # 断言检查 SeqAdd 的各种情况下的返回值是否符合预期
    assert SeqAdd() == S.EmptySequence
    assert SeqAdd(S.EmptySequence) == S.EmptySequence
    assert SeqAdd(per) == per
    assert SeqAdd(per, S.EmptySequence) == per
    assert SeqAdd(per_bou, form_bou) == S.EmptySequence

    # 创建 SeqAdd 对象，并且不进行计算（evaluate=False）
    s = SeqAdd(per_bou, form_bou2, evaluate=False)
    # 断言检查 SeqAdd 对象的 args 属性和索引访问的结果是否符合预期
    assert s.args == (form_bou2, per_bou)
    assert s[:] == [2, 6, 10, 18, 26]
    assert list(s) == [2, 6, 10, 18, 26]

    # 断言检查 SeqAdd(per, per_bou, evaluate=False) 返回的对象是否是 SeqAdd 类型
    assert isinstance(SeqAdd(per, per_bou, evaluate=False), SeqAdd)

    # 创建 SeqPer 对象，并且断言检查其类型和值是否符合预期
    s1 = SeqAdd(per, per_bou)
    assert isinstance(s1, SeqPer)
    assert s1 == SeqPer((2, 4, 4, 3, 3, 5), (n, 1, 5))
    
    # 创建 SeqFormula 对象，并且断言检查其类型和值是否符合预期
    s2 = SeqAdd(form, form_bou)
    assert isinstance(s2, SeqFormula)
    assert s2 == SeqFormula(2*n**2, (6, 10))

    # 断言检查 SeqAdd 各种组合情况下的返回值是否符合预期
    assert SeqAdd(form, form_bou, per) == \
        SeqAdd(per, SeqFormula(2*n**2, (6, 10)))
    assert SeqAdd(form, SeqAdd(form_bou, per)) == \
        SeqAdd(per, SeqFormula(2*n**2, (6, 10)))
    assert SeqAdd(per, SeqAdd(form, form_bou), evaluate=False) == \
        SeqAdd(per, SeqFormula(2*n**2, (6, 10)))

    # 断言检查 SeqAdd(SeqPer((1, 2), (n, 0, oo)), SeqPer((1, 2), (m, 0, oo))) 的结果是否符合预期
    assert SeqAdd(SeqPer((1, 2), (n, 0, oo)), SeqPer((1, 2), (m, 0, oo))) == \
        SeqPer((2, 4), (n, 0, oo))


def test_SeqMul():
    # 创建 SeqPer 对象，表示序列 (1, 2, 3) 在 n 从 0 到 ∞ 的范围内的周期序列
    per = SeqPer((1, 2, 3), (n, 0, oo))
    # 创建 SeqFormula 对象，表示序列 n**2 的一般形式
    form = SeqFormula(n**2)

    # 创建 SeqPer 对象，表示序列 (1, 2) 在 n 从 1 到 5 的范围内的周期序列
    per_bou = SeqPer((1, 2), (n, 1, 5))
    # 创建 SeqFormula 对象，表示序列 n**2 在 (n, 6, 10) 范围内的表达式
    form_bou = SeqFormula(n**2, (n, 6, 10))
    # 创建 SeqFormula 对象，表示序列 n**2 在 (1, 5) 范围内的表达式
    form_bou2 = SeqFormula(n**2, (1, 5))

    # 断言检查 SeqMul 的各种情况下的返回值是否符合预期
    assert SeqMul() == S.EmptySequence
    assert SeqMul(S.EmptySequence) == S.EmptySequence
    assert SeqMul(per) == per
    assert SeqMul(per, S.EmptySequence) == S.EmptySequence
    assert SeqMul(per_bou, form_bou) == S.EmptySequence

    # 创建 SeqMul 对象，并且不进行计算（evaluate=False）
    s = SeqMul(per_bou, form_bou2, evaluate=False)
    # 断言检查 SeqMul 对象的 args 属性和索引访问的结果是否符合预期
    assert s.args == (form_bou2, per_bou)
    assert s[:] == [1, 8, 9, 32, 25]
    assert list(s) == [1, 8, 9, 32, 25]

    # 断言检查 SeqMul(per, per_bou, evaluate=False) 返回的对象是否是 SeqMul 类型
    assert isinstance(SeqMul(per, per_bou, evaluate=False), SeqMul)

    # 创建 SeqPer 对象，并且断言检查其类型和值是否符合预期
    s1 = SeqMul(per, per_bou)
    assert isinstance(s1, SeqPer)
    assert s1 == SeqPer((1, 4, 3, 2, 2, 6), (n, 1, 5))
    
    # 创建 SeqFormula 对象，并且断言检查其类型和值是否符合预期
    s2 = SeqMul(form, form_bou)
    assert isinstance(s2, SeqFormula)
    assert s2 == SeqFormula(n**4, (6, 10))

    # 断言检查 SeqMul 各种组合情况下的返回值是否
    # 断言：验证两个表达式是否相等，表达式包括三个乘积
    assert SeqMul(form, SeqMul(form_bou, per)) == \
        SeqMul(per, SeqFormula(n**4, (6, 10)))
    
    # 断言：验证两个表达式是否相等，表达式包括四个乘积，其中第二个乘积具有禁用评估标志
    assert SeqMul(per, SeqMul(form, form_bou2,
                              evaluate=False), evaluate=False) == \
        SeqMul(form, per, form_bou2, evaluate=False)
    
    # 断言：验证两个序列乘积是否相等，每个序列包括一个周期序列，其起始索引从0到正无穷，步长为1
    assert SeqMul(SeqPer((1, 2), (n, 0, oo)), SeqPer((1, 2), (n, 0, oo))) == \
        SeqPer((1, 4), (n, 0, oo))
# 定义一个测试函数，用于测试 SeqPer 类的加法操作
def test_add():
    # 创建一个等差序列 SeqPer，表示 (1, 2) 到 (n, 0, oo)
    per = SeqPer((1, 2), (n, 0, oo))
    # 创建一个序列公式 SeqFormula，表示 n**2
    form = SeqFormula(n**2)

    # 断言 SeqPer 加上另一个 SeqPer((2, 3)) 结果等于 SeqPer((3, 5), (n, 0, oo))
    assert per + (SeqPer((2, 3))) == SeqPer((3, 5), (n, 0, oo))
    # 断言 SeqFormula 加上另一个 SeqFormula(n**3) 结果等于 SeqFormula(n**2 + n**3)
    assert form + SeqFormula(n**3) == SeqFormula(n**2 + n**3)

    # 断言 SeqPer 加上 SeqFormula 结果等于 SeqAdd(per, form)
    assert per + form == SeqAdd(per, form)

    # 断言加法操作中，如果操作数类型不匹配会引发 TypeError 异常
    raises(TypeError, lambda: per + n)
    raises(TypeError, lambda: n + per)


# 定义一个测试函数，用于测试 SeqPer 类的减法操作
def test_sub():
    # 创建一个等差序列 SeqPer，表示 (1, 2) 到 (n, 0, oo)
    per = SeqPer((1, 2), (n, 0, oo))
    # 创建一个序列公式 SeqFormula，表示 n**2
    form = SeqFormula(n**2)

    # 断言 SeqPer 减去另一个 SeqPer((2, 3)) 结果等于 SeqPer((-1, -1), (n, 0, oo))
    assert per - (SeqPer((2, 3))) == SeqPer((-1, -1), (n, 0, oo))
    # 断言 SeqFormula 减去另一个 SeqFormula(n**3) 结果等于 SeqFormula(n**2 - n**3)
    assert form - (SeqFormula(n**3)) == SeqFormula(n**2 - n**3)

    # 断言 SeqPer 减去 SeqFormula 结果等于 SeqAdd(per, -form)
    assert per - form == SeqAdd(per, -form)

    # 断言减法操作中，如果操作数类型不匹配会引发 TypeError 异常
    raises(TypeError, lambda: per - n)
    raises(TypeError, lambda: n - per)


# 定义一个测试函数，用于测试 SeqPer 类的乘法和系数乘法操作
def test_mul__coeff_mul():
    # 断言 SeqPer 等差序列乘以系数 2 的结果
    assert SeqPer((1, 2), (n, 0, oo)).coeff_mul(2) == SeqPer((2, 4), (n, 0, oo))
    # 断言 SeqFormula 序列公式乘以系数 2 的结果
    assert SeqFormula(n**2).coeff_mul(2) == SeqFormula(2*n**2)
    # 断言空序列乘以任何系数仍为空序列
    assert S.EmptySequence.coeff_mul(100) == S.EmptySequence

    # 断言 SeqPer 等差序列乘以另一个 SeqPer((2, 3)) 的结果
    assert SeqPer((1, 2), (n, 0, oo)) * (SeqPer((2, 3))) == SeqPer((2, 6), (n, 0, oo))
    # 断言 SeqFormula 序列公式乘以另一个 SeqFormula(n**3) 的结果
    assert SeqFormula(n**2) * SeqFormula(n**3) == SeqFormula(n**5)

    # 断言空序列与 SeqFormula 或 SeqPer 相乘的结果仍为空序列
    assert S.EmptySequence * SeqFormula(n**2) == S.EmptySequence
    assert SeqFormula(n**2) * S.EmptySequence == S.EmptySequence

    # 断言乘法操作中，如果操作数类型不匹配会引发 TypeError 异常
    raises(TypeError, lambda: sequence(n**2) * n)
    raises(TypeError, lambda: n * sequence(n**2))


# 定义一个测试函数，用于测试 SeqPer 类的取负操作
def test_neg():
    # 断言取负操作 SeqPer((1, -2), (n, 0, oo)) 的结果
    assert -SeqPer((1, -2), (n, 0, oo)) == SeqPer((-1, 2), (n, 0, oo))
    # 断言取负操作 SeqFormula(n**2) 的结果
    assert -SeqFormula(n**2) == SeqFormula(-n**2)


# 定义一个测试函数，用于测试 SeqPer 和 SeqFormula 类的组合操作
def test_operations():
    # 创建两个等差序列 SeqPer 和它们的乘积、序列公式及其组合
    per = SeqPer((1, 2), (n, 0, oo))
    per2 = SeqPer((2, 4), (n, 0, oo))
    form = SeqFormula(n**2)
    form2 = SeqFormula(n**3)

    # 断言等差序列与序列公式及其组合的结果
    assert per + form + form2 == SeqAdd(per, form, form2)
    assert per + form - form2 == SeqAdd(per, form, -form2)
    assert per + form - S.EmptySequence == SeqAdd(per, form)
    assert per + per2 + form == SeqAdd(SeqPer((3, 6), (n, 0, oo)), form)
    assert S.EmptySequence - per == -per
    assert form + form == SeqFormula(2*n**2)

    # 断言等差序列和序列公式的乘积及其组合的结果
    assert per * form * form2 == SeqMul(per, form, form2)
    assert form * form == SeqFormula(n**4)
    assert form * -form == SeqFormula(-n**4)

    # 断言序列公式乘以等差序列及其组合的结果
    assert form * (per + form2) == SeqMul(form, SeqAdd(per, form2))
    assert form * (per + per) == SeqMul(form, per2)

    # 断言乘法操作中，如果操作数类型不匹配会引发 TypeError 异常
    assert form.coeff_mul(m) == SeqFormula(m*n**2, (n, 0, oo))
    assert per.coeff_mul(m) == SeqPer((m, 2*m), (n, 0, oo))


# 定义一个测试函数，用于测试序列的索引限制操作
def test_Idx_limits():
    # 创建一个指标符号 i 和相应的 Indexed 对象 r
    i = symbols('i', cls=Idx)
    r = Indexed('r', i)

    # 断言 SeqFormula 序列公式在指标范围 (i, 0, 5) 下的索引操作结果
    assert SeqFormula(r, (i, 0, 5))[:] == [r.subs(i, j) for j in range(6)]
    # 断言 SeqPer 等差序列在指标范围 (i, 0, 5) 下的索引操作结果
    assert SeqPer((1, 2), (i, 0, 5))[:] == [1, 2, 1, 2, 1, 2]


# 定义一个慢速测试函数，用于测试序列的线性递推查找操作
@slow
def test_find_linear_recurrence():
    # 断言具有指定数列和指标范围的序列的线性递推查找结果
    assert sequence((0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55), (n, 0, 10)).find_linear_recurrence(11) == [1, 1]
    assert sequence((1, 2, 4, 7, 28, 128, 582, 2745, 13021, 61699, 292521, 1387138), (n, 0, 11)).find_linear_recurrence(12) == [5, -2, 6, -11]
    assert sequence(x*n**3+y*n, (n, 0, oo)).find_linear_recurrence(10) == [4, -6, 4, -1]
``
    # 断言：计算 x**n 的序列，找到其前 21 项的线性递推关系，预期结果是 [x]
    assert sequence(x**n, (n,0,20)).find_linear_recurrence(21) == [x]
    
    # 断言：计算 (1,2,3) 的序列，找到其前 10 项的线性递推关系，预期结果是 [0, 0, 1]
    assert sequence((1,2,3)).find_linear_recurrence(10, 5) == [0, 0, 1]
    
    # 断言：计算 Fibonacci 形式的序列，带有斐波那契数的通项公式，找到前 10 项的线性递推关系，预期结果是 [1, 1]
    assert sequence(((1 + sqrt(5))/2)**n + (-(1 + sqrt(5))/2)**(-n)).find_linear_recurrence(10) == [1, 1]
    
    # 断言：计算由 x 和 y 构成的序列，其中包含黄金分割数的递推形式，找到前 10 项的线性递推关系，预期结果是 [1, 1]
    assert sequence(x*((1 + sqrt(5))/2)**n + y*(-(1 + sqrt(5))/2)**(-n), (n,0,oo)).find_linear_recurrence(10) == [1, 1]
    
    # 断言：计算序列 (1,2,3,4,6)，在 n 从 0 到 4 的范围内，找到前 5 项的线性递推关系，预期结果是空列表 []
    assert sequence((1,2,3,4,6),(n, 0, 4)).find_linear_recurrence(5) == []
    
    # 断言：计算序列 (2,3,4,5,6,79)，在 n 从 0 到 5 的范围内，找到前 6 项的线性递推关系，预期结果是空列表 []，且没有通项公式
    assert sequence((2,3,4,5,6,79),(n, 0, 5)).find_linear_recurrence(6,gfvar=x) == ([], None)
    
    # 断言：计算序列 (2,3,4,5,8,30)，在 n 从 0 到 5 的范围内，找到前 6 项的线性递推关系，预期结果是 [19/2, -20, 27/2]，
    # 并给出其通项公式 (-31*x**2 + 32*x - 4)/(27*x**3 - 40*x**2 + 19*x -2)
    assert sequence((2,3,4,5,8,30),(n, 0, 5)).find_linear_recurrence(6,gfvar=x) == ([Rational(19, 2), -20, Rational(27, 2)],
        (-31*x**2 + 32*x - 4)/(27*x**3 - 40*x**2 + 19*x -2))
    
    # 断言：计算斐波那契数列的序列，找到前 30 项的线性递推关系，预期结果是 [1, 1]，并给出其通项公式 -x/(x**2 + x - 1)
    assert sequence(fibonacci(n)).find_linear_recurrence(30,gfvar=x) == ([1, 1], -x/(x**2 + x - 1))
    
    # 断言：计算三阶斐波那契数列的序列，找到前 30 项的线性递推关系，预期结果是 [1, 1, 1]，并给出其通项公式 -x/(x**3 + x**2 + x - 1)
    assert sequence(tribonacci(n)).find_linear_recurrence(30,gfvar=x) == ([1, 1, 1], -x/(x**3 + x**2 + x - 1))
# 定义测试函数 test_RecursiveSeq
def test_RecursiveSeq():
    # 创建一个符号函数 y
    y = Function('y')
    # 创建一个符号变量 n
    n = Symbol('n')
    # 使用 RecursiveSeq 类创建一个递归序列 fib，
    # 其递推关系为 y(n - 1) + y(n - 2)，当前项表示为 y(n)，起始值为 [0, 1]
    fib = RecursiveSeq(y(n - 1) + y(n - 2), y(n), n, [0, 1])
    # 断言序列 fib 的第三项系数为 2
    assert fib.coeff(3) == 2
```