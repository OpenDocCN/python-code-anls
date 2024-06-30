# `D:\src\scipysrc\sympy\sympy\assumptions\tests\test_query.py`

```
# 导入从 sympy.abc 模块中的符号 t, w, x, y, z, n, k, m, p, i
from sympy.abc import t, w, x, y, z, n, k, m, p, i
# 导入 sympy.assumptions 模块中的相关函数和类
from sympy.assumptions import (ask, AssumptionsContext, Q, register_handler,
        remove_handler)
# 导入 sympy.assumptions.assume 模块中的相关函数和类
from sympy.assumptions.assume import assuming, global_assumptions, Predicate
# 导入 sympy.assumptions.cnf 模块中的相关函数和类
from sympy.assumptions.cnf import CNF, Literal
# 导入 sympy.assumptions.facts 模块中的相关函数和类
from sympy.assumptions.facts import (single_fact_lookup,
    get_known_facts, generate_known_facts_dict, get_known_facts_keys)
# 导入 sympy.assumptions.handlers 模块中的相关函数和类
from sympy.assumptions.handlers import AskHandler
# 导入 sympy.assumptions.ask_generated 模块中的相关函数
from sympy.assumptions.ask_generated import (get_all_known_facts,
    get_known_facts_dict)
# 导入 sympy.core.add 模块中的 Add 类
from sympy.core.add import Add
# 导入 sympy.core.numbers 模块中的数值类型 Integer, Rational, 等
from sympy.core.numbers import (I, Integer, Rational, oo, zoo, pi)
# 导入 sympy.core.singleton 模块中的 S 单例
from sympy.core.singleton import S
# 导入 sympy.core.power 模块中的 Pow 类
from sympy.core.power import Pow
# 导入 sympy.core.symbol 模块中的 Str, symbols, Symbol 类
from sympy.core.symbol import Str, symbols, Symbol
# 导入 sympy.functions.combinatorial.factorials 模块中的 factorial 函数
from sympy.functions.combinatorial.factorials import factorial
# 导入 sympy.functions.elementary.complexes 模块中的复数相关函数
from sympy.functions.elementary.complexes import (Abs, im, re, sign)
# 导入 sympy.functions.elementary.exponential 模块中的指数和对数函数
from sympy.functions.elementary.exponential import (exp, log)
# 导入 sympy.functions.elementary.miscellaneous 模块中的杂项函数
from sympy.functions.elementary.miscellaneous import sqrt
# 导入 sympy.functions.elementary.trigonometric 模块中的三角函数
from sympy.functions.elementary.trigonometric import (
    acos, acot, asin, atan, cos, cot, sin, tan)
# 导入 sympy.logic.boolalg 模块中的逻辑运算函数和类
from sympy.logic.boolalg import Equivalent, Implies, Xor, And, to_cnf
# 导入 sympy.matrices 模块中的矩阵类 Matrix, SparseMatrix
from sympy.matrices import Matrix, SparseMatrix
# 导入 sympy.testing.pytest 模块中的测试装饰器和函数
from sympy.testing.pytest import (XFAIL, slow, raises, warns_deprecated_sympy,
    _both_exp_pow)
# 导入 Python 内置的 math 模块
import math


# 定义测试函数 test_int_1，用于测试整数 z = 1 的多个数学性质
def test_int_1():
    # 设定 z = 1
    z = 1
    # 断言 z 是否是可交换的（commutative）
    assert ask(Q.commutative(z)) is True
    # 断言 z 是否是整数
    assert ask(Q.integer(z)) is True
    # 断言 z 是否是有理数
    assert ask(Q.rational(z)) is True
    # 断言 z 是否是实数
    assert ask(Q.real(z)) is True
    # 断言 z 是否是复数
    assert ask(Q.complex(z)) is True
    # 断言 z 是否是无理数
    assert ask(Q.irrational(z)) is False
    # 断言 z 是否是虚数
    assert ask(Q.imaginary(z)) is False
    # 断言 z 是否是正数
    assert ask(Q.positive(z)) is True
    # 断言 z 是否是负数
    assert ask(Q.negative(z)) is False
    # 断言 z 是否是偶数
    assert ask(Q.even(z)) is False
    # 断言 z 是否是奇数
    assert ask(Q.odd(z)) is True
    # 断言 z 是否是有限数
    assert ask(Q.finite(z)) is True
    # 断言 z 是否是素数
    assert ask(Q.prime(z)) is False
    # 断言 z 是否是合数
    assert ask(Q.composite(z)) is False
    # 断言 z 是否是厄米（hermitian）的
    assert ask(Q.hermitian(z)) is True
    # 断言 z 是否是反厄米（antihermitian）的
    assert ask(Q.antihermitian(z)) is False


# 定义测试函数 test_int_11，用于测试整数 z = 11 的多个数学性质
def test_int_11():
    # 设定 z = 11
    z = 11
    # 断言 z 是否是可交换的（commutative）
    assert ask(Q.commutative(z)) is True
    # 断言 z 是否是整数
    assert ask(Q.integer(z)) is True
    # 断言 z 是否是有理数
    assert ask(Q.rational(z)) is True
    # 断言 z 是否是实数
    assert ask(Q.real(z)) is True
    # 断言 z 是否是复数
    assert ask(Q.complex(z)) is True
    # 断言 z 是否是无理数
    assert ask(Q.irrational(z)) is False
    # 断言 z 是否是虚数
    assert ask(Q.imaginary(z)) is False
    # 断言 z 是否是正数
    assert ask(Q.positive(z)) is True
    # 断言 z 是否是负数
    assert ask(Q.negative(z)) is False
    # 断言 z 是否是偶数
    assert ask(Q.even(z)) is False
    # 断言 z 是否是奇数
    assert ask(Q.odd(z)) is True
    # 断言 z 是否是有限数
    assert ask(Q.finite(z)) is True
    # 断言 z 是否是素数
    assert ask(Q.prime(z)) is True
    # 断言 z 是否是合数
    assert ask(Q.composite(z)) is False
    # 断言 z 是否是厄米（hermitian）的
    assert ask(Q.hermitian(z)) is True
    # 断言 z 是否是反厄米（antihermitian）的
    assert ask(Q.antihermitian(z)) is False


# 定义测试函数 test_int_12，用于测试整数 z = 12 的多个数学性质
def test_int_12():
    # 设定 z = 12
    z = 12
    # 断言 z 是否是可交换的（commutative）
    assert ask(Q.commutative(z)) is True
    # 断言 z 是否是整数
    assert ask(Q.integer(z)) is True
    # 断言 z 是否是有理数
    assert ask(Q.rational(z)) is True
    # 断言 z 是否是实数
    assert ask(Q.real(z)) is True
    # 断言 z 是否是复数
    assert ask(Q.complex(z)) is True
    # 断言 z 是否是无理数
    assert ask(Q.irrational(z)) is False
    # 断言 z 是否是虚数
    assert ask(Q.imaginary(z)) is False
    # 断言 z 是否是正数
    assert ask(Q.positive(z)) is True
    # 断言 z 是否是负数
    assert ask(Q.negative(z)) is False
    # 断言检查给定的数 z 是否为偶数，预期结果应为 True
    assert ask(Q.even(z)) is True
    # 断言检查给定的数 z 是否为奇数，预期结果应为 False
    assert ask(Q.odd(z)) is False
    # 断言检查给定的数 z 是否为有限数，预期结果应为 True
    assert ask(Q.finite(z)) is True
    # 断言检查给定的数 z 是否为质数，预期结果应为 False
    assert ask(Q.prime(z)) is False
    # 断言检查给定的数 z 是否为合数，预期结果应为 True
    assert ask(Q.composite(z)) is True
    # 断言检查给定的数 z 是否为厄米矩阵，预期结果应为 True
    assert ask(Q.hermitian(z)) is True
    # 断言检查给定的数 z 是否为反厄米矩阵，预期结果应为 False
    assert ask(Q.antihermitian(z)) is False
# 测试浮点数的属性
def test_float_1():
    z = 1.0
    # 检查是否是可交换的
    assert ask(Q.commutative(z)) is True
    # 检查是否是整数
    assert ask(Q.integer(z)) is None
    # 检查是否是有理数
    assert ask(Q.rational(z)) is None
    # 检查是否是实数
    assert ask(Q.real(z)) is True
    # 检查是否是复数
    assert ask(Q.complex(z)) is True
    # 检查是否是无理数
    assert ask(Q.irrational(z)) is None
    # 检查是否是虚数
    assert ask(Q.imaginary(z)) is False
    # 检查是否是正数
    assert ask(Q.positive(z)) is True
    # 检查是否是负数
    assert ask(Q.negative(z)) is False
    # 检查是否是偶数
    assert ask(Q.even(z)) is None
    # 检查是否是奇数
    assert ask(Q.odd(z)) is None
    # 检查是否是有限数
    assert ask(Q.finite(z)) is True
    # 检查是否是质数
    assert ask(Q.prime(z)) is None
    # 检查是否是合数
    assert ask(Q.composite(z)) is None
    # 检查是否是共轭对称的
    assert ask(Q.hermitian(z)) is True
    # 检查是否是反共轭对称的
    assert ask(Q.antihermitian(z)) is False

    z = 7.2123
    # 同上，测试另一个浮点数
    assert ask(Q.commutative(z)) is True
    assert ask(Q.integer(z)) is False
    assert ask(Q.rational(z)) is None
    assert ask(Q.real(z)) is True
    assert ask(Q.complex(z)) is True
    assert ask(Q.irrational(z)) is None
    assert ask(Q.imaginary(z)) is False
    assert ask(Q.positive(z)) is True
    assert ask(Q.negative(z)) is False
    assert ask(Q.even(z)) is False
    assert ask(Q.odd(z)) is False
    assert ask(Q.finite(z)) is True
    assert ask(Q.prime(z)) is False
    assert ask(Q.composite(z)) is False
    assert ask(Q.hermitian(z)) is True
    assert ask(Q.antihermitian(z)) is False

    # 测试 issue #12168，验证π是否为有理数
    assert ask(Q.rational(math.pi)) is None


# 测试整数0的属性
def test_zero_0():
    z = Integer(0)
    assert ask(Q.nonzero(z)) is False
    assert ask(Q.zero(z)) is True
    assert ask(Q.commutative(z)) is True
    assert ask(Q.integer(z)) is True
    assert ask(Q.rational(z)) is True
    assert ask(Q.real(z)) is True
    assert ask(Q.complex(z)) is True
    assert ask(Q.imaginary(z)) is False
    assert ask(Q.positive(z)) is False
    assert ask(Q.negative(z)) is False
    assert ask(Q.even(z)) is True
    assert ask(Q.odd(z)) is False
    assert ask(Q.finite(z)) is True
    assert ask(Q.prime(z)) is False
    assert ask(Q.composite(z)) is False
    assert ask(Q.hermitian(z)) is True
    assert ask(Q.antihermitian(z)) is True


# 测试整数-1的属性
def test_negativeone():
    z = Integer(-1)
    assert ask(Q.nonzero(z)) is True
    assert ask(Q.zero(z)) is False
    assert ask(Q.commutative(z)) is True
    assert ask(Q.integer(z)) is True
    assert ask(Q.rational(z)) is True
    assert ask(Q.real(z)) is True
    assert ask(Q.complex(z)) is True
    assert ask(Q.irrational(z)) is False
    assert ask(Q.imaginary(z)) is False
    assert ask(Q.positive(z)) is False
    assert ask(Q.negative(z)) is True
    assert ask(Q.even(z)) is False
    assert ask(Q.odd(z)) is True
    assert ask(Q.finite(z)) is True
    assert ask(Q.prime(z)) is False
    assert ask(Q.composite(z)) is False
    assert ask(Q.hermitian(z)) is True
    assert ask(Q.antihermitian(z)) is False


# 测试无穷大的属性
def test_infinity():
    assert ask(Q.commutative(oo)) is True
    assert ask(Q.integer(oo)) is False
    assert ask(Q.rational(oo)) is False
    assert ask(Q.algebraic(oo)) is False
    assert ask(Q.real(oo)) is False
    #`
# 检查问询 Q.extended_real(oo) 是否为 True
assert ask(Q.extended_real(oo)) is True
# 检查问询 Q.complex(oo) 是否为 False
assert ask(Q.complex(oo)) is False
# 检查问询 Q.irrational(oo) 是否为 False
assert ask(Q.irrational(oo)) is False
# 检查问询 Q.imaginary(oo) 是否为 False
assert ask(Q.imaginary(oo)) is False
# 检查问询 Q.positive(oo) 是否为 False
assert ask(Q.positive(oo)) is False
# 检查问询 Q.extended_positive(oo) 是否为 True
assert ask(Q.extended_positive(oo)) is True
# 检查问询 Q.negative(oo) 是否为 False
assert ask(Q.negative(oo)) is False
# 检查问询 Q.even(oo) 是否为 False
assert ask(Q.even(oo)) is False
# 检查问询 Q.odd(oo) 是否为 False
assert ask(Q.odd(oo)) is False
# 检查问询 Q.finite(oo) 是否为 False
assert ask(Q.finite(oo)) is False
# 检查问询 Q.infinite(oo) 是否为 True
assert ask(Q.infinite(oo)) is True
# 检查问询 Q.prime(oo) 是否为 False
assert ask(Q.prime(oo)) is False
# 检查问询 Q.composite(oo) 是否为 False
assert ask(Q.composite(oo)) is False
# 检查问询 Q.hermitian(oo) 是否为 False
assert ask(Q.hermitian(oo)) is False
# 检查问询 Q.antihermitian(oo) 是否为 False
assert ask(Q.antihermitian(oo)) is False
# 检查问询 Q.positive_infinite(oo) 是否为 True
assert ask(Q.positive_infinite(oo)) is True
# 检查问询 Q.negative_infinite(oo) 是否为 False
assert ask(Q.negative_infinite(oo)) is False
# 定义一个测试函数，用于测试特殊符号NegativeInfinity的性质
def test_neg_infinity():
    # 将变量mm设置为S.NegativeInfinity
    mm = S.NegativeInfinity
    # 断言S.NegativeInfinity满足Q.commutative属性
    assert ask(Q.commutative(mm)) is True
    # 断言S.NegativeInfinity不满足Q.integer属性
    assert ask(Q.integer(mm)) is False
    # 断言S.NegativeInfinity不满足Q.rational属性
    assert ask(Q.rational(mm)) is False
    # 断言S.NegativeInfinity不满足Q.algebraic属性
    assert ask(Q.algebraic(mm)) is False
    # 断言S.NegativeInfinity不满足Q.real属性
    assert ask(Q.real(mm)) is False
    # 断言S.NegativeInfinity满足Q.extended_real属性
    assert ask(Q.extended_real(mm)) is True
    # 断言S.NegativeInfinity不满足Q.complex属性
    assert ask(Q.complex(mm)) is False
    # 断言S.NegativeInfinity不满足Q.irrational属性
    assert ask(Q.irrational(mm)) is False
    # 断言S.NegativeInfinity不满足Q.imaginary属性
    assert ask(Q.imaginary(mm)) is False
    # 断言S.NegativeInfinity不满足Q.positive属性
    assert ask(Q.positive(mm)) is False
    # 断言S.NegativeInfinity不满足Q.negative属性
    assert ask(Q.negative(mm)) is False
    # 断言S.NegativeInfinity满足Q.extended_negative属性
    assert ask(Q.extended_negative(mm)) is True
    # 断言S.NegativeInfinity不满足Q.even属性
    assert ask(Q.even(mm)) is False
    # 断言S.NegativeInfinity不满足Q.odd属性
    assert ask(Q.odd(mm)) is False
    # 断言S.NegativeInfinity不满足Q.finite属性
    assert ask(Q.finite(mm)) is False
    # 断言S.NegativeInfinity满足Q.infinite属性
    assert ask(Q.infinite(oo)) is True
    # 断言S.NegativeInfinity不满足Q.prime属性
    assert ask(Q.prime(mm)) is False
    # 断言S.NegativeInfinity不满足Q.composite属性
    assert ask(Q.composite(mm)) is False
    # 断言S.NegativeInfinity不满足Q.hermitian属性
    assert ask(Q.hermitian(mm)) is False
    # 断言S.NegativeInfinity不满足Q.antihermitian属性
    assert ask(Q.antihermitian(mm)) is False
    # 断言S.NegativeInfinity不满足Q.positive_infinite属性
    assert ask(Q.positive_infinite(-oo)) is False
    # 断言S.NegativeInfinity满足Q.negative_infinite属性
    assert ask(Q.negative_infinite(-oo)) is True


# 定义一个测试函数，用于测试特殊符号zoo的性质
def test_complex_infinity():
    # 断言zoo满足Q.commutative属性
    assert ask(Q.commutative(zoo)) is True
    # 断言zoo不满足Q.integer属性
    assert ask(Q.integer(zoo)) is False
    # 断言zoo不满足Q.rational属性
    assert ask(Q.rational(zoo)) is False
    # 断言zoo不满足Q.algebraic属性
    assert ask(Q.algebraic(zoo)) is False
    # 断言zoo不满足Q.real属性
    assert ask(Q.real(zoo)) is False
    # 断言zoo不满足Q.extended_real属性
    assert ask(Q.extended_real(zoo)) is False
    # 断言zoo不满足Q.complex属性
    assert ask(Q.complex(zoo)) is False
    # 断言zoo不满足Q.irrational属性
    assert ask(Q.irrational(zoo)) is False
    # 断言zoo不满足Q.imaginary属性
    assert ask(Q.imaginary(zoo)) is False
    # 断言zoo不满足Q.positive属性
    assert ask(Q.positive(zoo)) is False
    # 断言zoo不满足Q.negative属性
    assert ask(Q.negative(zoo)) is False
    # 断言zoo不满足Q.zero属性
    assert ask(Q.zero(zoo)) is False
    # 断言zoo不满足Q.nonzero属性
    assert ask(Q.nonzero(zoo)) is False
    # 断言zoo不满足Q.even属性
    assert ask(Q.even(zoo)) is False
    # 断言zoo不满足Q.odd属性
    assert ask(Q.odd(zoo)) is False
    # 断言zoo不满足Q.finite属性
    assert ask(Q.finite(zoo)) is False
    # 断言zoo满足Q.infinite属性
    assert ask(Q.infinite(zoo)) is True
    # 断言zoo不满足Q.prime属性
    assert ask(Q.prime(zoo)) is False
    # 断言zoo不满足Q.composite属性
    assert ask(Q.composite(zoo)) is False
    # 断言zoo不满足Q.hermitian属性
    assert ask(Q.hermitian(zoo)) is False
    # 断言zoo不满足Q.antihermitian属性
    assert ask(Q.antihermitian(zoo)) is False
    # 断言zoo不满足Q.positive_infinite属性
    assert ask(Q.positive_infinite(zoo)) is False
    # 断言zoo不满足Q.negative_infinite属性
    assert ask(Q.negative_infinite(zoo)) is False


# 定义一个测试函数，用于测试特殊符号NaN的性质
def test_nan():
    # 将变量nan设置为S.NaN
    nan = S.NaN
    # 断言S.NaN满足Q.commutative属性
    assert ask(Q.commutative(nan)) is True
    # 断言S.NaN不满足Q.integer属性
    assert ask(Q.integer(nan)) is None
    # 断言S.NaN不满足Q.rational属性
    assert ask(Q.rational(nan)) is None
    # 断言S.NaN不满足Q.algebraic属性
    assert ask(Q.algebraic(nan)) is None
    # 断言S.NaN不满足Q.real属性
    assert ask(Q.real(nan)) is None
    # 断言S.NaN不满足Q.extended_real属性
    assert ask(Q.extended_real(nan)) is None
    # 断言S.NaN不满足Q.complex属性
    assert ask(Q.complex(nan)) is None
    # 断言S.NaN不满足Q.irrational属性
    assert ask(Q.irrational(nan)) is None
    # 断言S.NaN不满足Q.imaginary属性
    assert ask(Q.imaginary(nan)) is None
    # 断言S.NaN不满足Q.positive属性
    assert ask(Q.positive(nan)) is None
    # 断言S.NaN不满足Q.nonzero属性
    assert ask(Q.nonzero(nan)) is None
    # 断言S.NaN不满足Q.zero属性
    assert ask(Q.zero(nan)) is None
    # 断言S.NaN不满足Q.even属性
    assert ask(Q.even(nan)) is None
    # 断言S.NaN不满足Q.odd属性
    assert ask(Q.odd(nan)) is None
    # 断言S.NaN不满足Q.finite属性
    assert ask(Q.finite(nan)) is None
    # 断言S.NaN不满足Q.infinite属性
    assert ask(Q.infinite(nan)) is None
    # 断言S.NaN不满足Q.prime属性
    assert ask(Q.prime(nan)) is None
    # 断言S.NaN不满足Q.composite属性
    assert ask(Q.composite(nan)) is None
    # 断言S.NaN不满足Q.hermitian属性
    assert ask(Q.hermitian(nan)) is None
    # 断
    # 检查给定的有理数是否属于复数集合，返回布尔值
    assert ask(Q.complex(r)) is True
    
    # 检查给定的有理数是否为无理数，返回布尔值
    assert ask(Q.irrational(r)) is False
    
    # 检查给定的有理数是否为虚数，返回布尔值
    assert ask(Q.imaginary(r)) is False
    
    # 检查给定的有理数是否为正数，返回布尔值
    assert ask(Q.positive(r)) is True
    
    # 检查给定的有理数是否为负数，返回布尔值
    assert ask(Q.negative(r)) is False
    
    # 检查给定的有理数是否为偶数，返回布尔值
    assert ask(Q.even(r)) is False
    
    # 检查给定的有理数是否为奇数，返回布尔值
    assert ask(Q.odd(r)) is False
    
    # 检查给定的有理数是否为有限数，返回布尔值
    assert ask(Q.finite(r)) is True
    
    # 检查给定的有理数是否为质数，返回布尔值
    assert ask(Q.prime(r)) is False
    
    # 检查给定的有理数是否为合数，返回布尔值
    assert ask(Q.composite(r)) is False
    
    # 检查给定的有理数是否为厄米矩阵，返回布尔值
    assert ask(Q.hermitian(r)) is True
    
    # 检查给定的有理数是否为反厄米矩阵，返回布尔值
    assert ask(Q.antihermitian(r)) is False
    
    # 创建有理数对象 Rational(1, 4)
    r = Rational(1, 4)
    assert ask(Q.positive(r)) is True
    assert ask(Q.negative(r)) is False
    
    # 修改有理数对象为 Rational(5, 4)
    r = Rational(5, 4)
    assert ask(Q.negative(r)) is False
    assert ask(Q.positive(r)) is True
    
    # 修改有理数对象为 Rational(5, 3)
    r = Rational(5, 3)
    assert ask(Q.positive(r)) is True
    assert ask(Q.negative(r)) is False
    
    # 修改有理数对象为 Rational(-3, 4)
    r = Rational(-3, 4)
    assert ask(Q.positive(r)) is False
    assert ask(Q.negative(r)) is True
    
    # 修改有理数对象为 Rational(-1, 4)
    r = Rational(-1, 4)
    assert ask(Q.positive(r)) is False
    assert ask(Q.negative(r)) is True
    
    # 修改有理数对象为 Rational(-5, 4)
    r = Rational(-5, 4)
    assert ask(Q.negative(r)) is True
    assert ask(Q.positive(r)) is False
    
    # 修改有理数对象为 Rational(-5, 3)
    r = Rational(-5, 3)
    assert ask(Q.positive(r)) is False
    assert ask(Q.negative(r)) is True
def test_sqrt_2():
    # 计算平方根并赋值给变量 z
    z = sqrt(2)
    # 断言 z 是可交换的
    assert ask(Q.commutative(z)) is True
    # 断言 z 不是整数
    assert ask(Q.integer(z)) is False
    # 断言 z 不是有理数
    assert ask(Q.rational(z)) is False
    # 断言 z 是实数
    assert ask(Q.real(z)) is True
    # 断言 z 是复数
    assert ask(Q.complex(z)) is True
    # 断言 z 是无理数
    assert ask(Q.irrational(z)) is True
    # 断言 z 不是虚数
    assert ask(Q.imaginary(z)) is False
    # 断言 z 是正数
    assert ask(Q.positive(z)) is True
    # 断言 z 不是负数
    assert ask(Q.negative(z)) is False
    # 断言 z 不是偶数
    assert ask(Q.even(z)) is False
    # 断言 z 不是奇数
    assert ask(Q.odd(z)) is False
    # 断言 z 是有限数
    assert ask(Q.finite(z)) is True
    # 断言 z 不是质数
    assert ask(Q.prime(z)) is False
    # 断言 z 不是合数
    assert ask(Q.composite(z)) is False
    # 断言 z 是共轭对称的
    assert ask(Q.hermitian(z)) is True
    # 断言 z 不是反共轭对称的
    assert ask(Q.antihermitian(z)) is False

def test_pi():
    # 将 π 赋值给变量 z
    z = S.Pi
    # 断言 z 是可交换的
    assert ask(Q.commutative(z)) is True
    # 断言 z 不是整数
    assert ask(Q.integer(z)) is False
    # 断言 z 不是有理数
    assert ask(Q.rational(z)) is False
    # 断言 z 不是代数数
    assert ask(Q.algebraic(z)) is False
    # 断言 z 是实数
    assert ask(Q.real(z)) is True
    # 断言 z 是复数
    assert ask(Q.complex(z)) is True
    # 断言 z 是无理数
    assert ask(Q.irrational(z)) is True
    # 断言 z 不是虚数
    assert ask(Q.imaginary(z)) is False
    # 断言 z 是正数
    assert ask(Q.positive(z)) is True
    # 断言 z 不是负数
    assert ask(Q.negative(z)) is False
    # 断言 z 不是偶数
    assert ask(Q.even(z)) is False
    # 断言 z 不是奇数
    assert ask(Q.odd(z)) is False
    # 断言 z 是有限数
    assert ask(Q.finite(z)) is True
    # 断言 z 不是质数
    assert ask(Q.prime(z)) is False
    # 断言 z 不是合数
    assert ask(Q.composite(z)) is False
    # 断言 z 是共轭对称的
    assert ask(Q.hermitian(z)) is True
    # 断言 z 不是反共轭对称的
    assert ask(Q.antihermitian(z)) is False

    # 将 π + 1 赋值给变量 z
    z = S.Pi + 1
    # 断言 z 是可交换的
    assert ask(Q.commutative(z)) is True
    # 断言 z 不是整数
    assert ask(Q.integer(z)) is False
    # 断言 z 不是有理数
    assert ask(Q.rational(z)) is False
    # 断言 z 不是代数数
    assert ask(Q.algebraic(z)) is False
    # 断言 z 是实数
    assert ask(Q.real(z)) is True
    # 断言 z 是复数
    assert ask(Q.complex(z)) is True
    # 断言 z 是无理数
    assert ask(Q.irrational(z)) is True
    # 断言 z 不是虚数
    assert ask(Q.imaginary(z)) is False
    # 断言 z 是正数
    assert ask(Q.positive(z)) is True
    # 断言 z 不是负数
    assert ask(Q.negative(z)) is False
    # 断言 z 不是偶数
    assert ask(Q.even(z)) is False
    # 断言 z 不是奇数
    assert ask(Q.odd(z)) is False
    # 断言 z 是有限数
    assert ask(Q.finite(z)) is True
    # 断言 z 不是质数
    assert ask(Q.prime(z)) is False
    # 断言 z 不是合数
    assert ask(Q.composite(z)) is False
    # 断言 z 是共轭对称的
    assert ask(Q.hermitian(z)) is True
    # 断言 z 不是反共轭对称的
    assert ask(Q.antihermitian(z)) is False

    # 将 2π 赋值给变量 z
    z = 2*S.Pi
    # 断言 z 是可交换的
    assert ask(Q.commutative(z)) is True
    # 断言 z 不是整数
    assert ask(Q.integer(z)) is False
    # 断言 z 不是有理数
    assert ask(Q.rational(z)) is False
    # 断言 z 不是代数数
    assert ask(Q.algebraic(z)) is False
    # 断言 z 是实数
    assert ask(Q.real(z)) is True
    # 断言 z 是复数
    assert ask(Q.complex(z)) is True
    # 断言 z 是无理数
    assert ask(Q.irrational(z)) is True
    # 断言 z 不是虚数
    assert ask(Q.imaginary(z)) is False
    # 断言 z 是正数
    assert ask(Q.positive(z)) is True
    # 断言 z 不是负数
    assert ask(Q.negative(z)) is False
    # 断言 z 不是偶数
    assert ask(Q.even(z)) is False
    # 断言 z 不是奇数
    assert ask(Q.odd(z)) is False
    # 断言 z 是有限数
    assert ask(Q.finite(z)) is True
    # 断言 z 不是质数
    assert ask(Q.prime(z)) is False
    # 断言 z 不是合数
    assert ask(Q.composite(z)) is False
    # 断言 z 是共轭对称的
    assert ask(Q.hermitian(z)) is True
    # 断言 z 不是反共轭对称的
    assert ask(Q.antihermitian(z)) is False

    # 将 π^2 赋值给变量 z
    z = S.Pi ** 2
    # 断言 z 是可交换的
    assert ask(Q.commutative(z)) is True
    # 断言 z 不是整数
    assert ask(Q.integer(z)) is False
    # 断言 z 不是有理数
    assert ask(Q.rational(z)) is False
    # 断言 z 不是代数数
    assert ask(Q.algebraic(z)) is False
    # 断言 z 是实数
    assert ask(Q.real(z)) is True
    # 断言 z 是复数
    assert ask(Q.complex(z)) is True
    # 断言 z 是无理数
    assert ask(Q.irrational(z)) is True
    # 断言 z 不是虚数
    assert ask(Q.imaginary(z)) is False
    # 断言 z 是正数
    assert ask(Q.positive(z)) is True
    # 断言 z 不是负数
    assert ask(Q.negative(z)) is False
    # 断言 z 不是偶数
    assert ask(Q.even(z)) is False
    # 断言 z 不是奇数
    assert ask(Q.odd(z)) is False
    # 断言 z 是有限数
    assert ask(Q.finite(z)) is True
    # 断言 z 不是质数
    assert ask(Q.prime(z)) is False
    # 断言 z 不是合数
    assert ask(Q.composite(z)) is False
    # 断言 z 是厄米矩阵
    assert ask(Q.hermitian(z)) is True
    # 断言 z 不是反厄米矩阵
    assert ask(Q.antihermitian(z)) is False
    
    # 计算 z = (1 + π)^2
    z = (1 + S.Pi) ** 2
    # 断言 z 是可交换的
    assert ask(Q.commutative(z)) is True
    # 断言 z 不是整数
    assert ask(Q.integer(z)) is False
    # 断言 z 不是有理数
    assert ask(Q.rational(z)) is False
    # 断言 z 不是代数数
    assert ask(Q.algebraic(z)) is False
    # 断言 z 是实数
    assert ask(Q.real(z)) is True
    # 断言 z 是复数
    assert ask(Q.complex(z)) is True
    # 断言 z 是无理数
    assert ask(Q.irrational(z)) is True
    # 断言 z 不是虚数
    assert ask(Q.imaginary(z)) is False
    # 断言 z 是正数
    assert ask(Q.positive(z)) is True
    # 断言 z 不是负数
    assert ask(Q.negative(z)) is False
    # 断言 z 不是偶数
    assert ask(Q.even(z)) is False
    # 断言 z 不是奇数
    assert ask(Q.odd(z)) is False
    # 断言 z 是有限数
    assert ask(Q.finite(z)) is True
    # 断言 z 不是质数
    assert ask(Q.prime(z)) is False
    # 断言 z 不是合数
    assert ask(Q.composite(z)) is False
    # 断言 z 是厄米矩阵
    assert ask(Q.hermitian(z)) is True
    # 断言 z 不是反厄米矩阵
    assert ask(Q.antihermitian(z)) is False
# 定义函数 test_E，用于测试表达式 S.Exp1 的性质
def test_E():
    # 获取 S.Exp1 的值并进行下列断言
    z = S.Exp1
    # 断言 S.Exp1 是可交换的
    assert ask(Q.commutative(z)) is True
    # 断言 S.Exp1 不是整数
    assert ask(Q.integer(z)) is False
    # 断言 S.Exp1 不是有理数
    assert ask(Q.rational(z)) is False
    # 断言 S.Exp1 不是代数数
    assert ask(Q.algebraic(z)) is False
    # 断言 S.Exp1 是实数
    assert ask(Q.real(z)) is True
    # 断言 S.Exp1 是复数
    assert ask(Q.complex(z)) is True
    # 断言 S.Exp1 是无理数
    assert ask(Q.irrational(z)) is True
    # 断言 S.Exp1 不是虚数
    assert ask(Q.imaginary(z)) is False
    # 断言 S.Exp1 是正数
    assert ask(Q.positive(z)) is True
    # 断言 S.Exp1 不是负数
    assert ask(Q.negative(z)) is False
    # 断言 S.Exp1 不是偶数
    assert ask(Q.even(z)) is False
    # 断言 S.Exp1 不是奇数
    assert ask(Q.odd(z)) is False
    # 断言 S.Exp1 是有限数
    assert ask(Q.finite(z)) is True
    # 断言 S.Exp1 不是素数
    assert ask(Q.prime(z)) is False
    # 断言 S.Exp1 不是合数
    assert ask(Q.composite(z)) is False
    # 断言 S.Exp1 是厄米矩阵
    assert ask(Q.hermitian(z)) is True
    # 断言 S.Exp1 不是反厄米矩阵
    assert ask(Q.antihermitian(z)) is False


# 定义函数 test_GoldenRatio，用于测试黄金比例 S.GoldenRatio 的性质
def test_GoldenRatio():
    # 获取黄金比例 S.GoldenRatio 的值并进行下列断言
    z = S.GoldenRatio
    # 断言 S.GoldenRatio 是可交换的
    assert ask(Q.commutative(z)) is True
    # 断言 S.GoldenRatio 不是整数
    assert ask(Q.integer(z)) is False
    # 断言 S.GoldenRatio 不是有理数
    assert ask(Q.rational(z)) is False
    # 断言 S.GoldenRatio 是代数数
    assert ask(Q.algebraic(z)) is True
    # 断言 S.GoldenRatio 是实数
    assert ask(Q.real(z)) is True
    # 断言 S.GoldenRatio 是复数
    assert ask(Q.complex(z)) is True
    # 断言 S.GoldenRatio 是无理数
    assert ask(Q.irrational(z)) is True
    # 断言 S.GoldenRatio 不是虚数
    assert ask(Q.imaginary(z)) is False
    # 断言 S.GoldenRatio 是正数
    assert ask(Q.positive(z)) is True
    # 断言 S.GoldenRatio 不是负数
    assert ask(Q.negative(z)) is False
    # 断言 S.GoldenRatio 不是偶数
    assert ask(Q.even(z)) is False
    # 断言 S.GoldenRatio 不是奇数
    assert ask(Q.odd(z)) is False
    # 断言 S.GoldenRatio 是有限数
    assert ask(Q.finite(z)) is True
    # 断言 S.GoldenRatio 不是素数
    assert ask(Q.prime(z)) is False
    # 断言 S.GoldenRatio 不是合数
    assert ask(Q.composite(z)) is False
    # 断言 S.GoldenRatio 是厄米矩阵
    assert ask(Q.hermitian(z)) is True
    # 断言 S.GoldenRatio 不是反厄米矩阵
    assert ask(Q.antihermitian(z)) is False


# 定义函数 test_TribonacciConstant，用于测试第三类斐波那契常数 S.TribonacciConstant 的性质
def test_TribonacciConstant():
    # 获取第三类斐波那契常数 S.TribonacciConstant 的值并进行下列断言
    z = S.TribonacciConstant
    # 断言 S.TribonacciConstant 是可交换的
    assert ask(Q.commutative(z)) is True
    # 断言 S.TribonacciConstant 不是整数
    assert ask(Q.integer(z)) is False
    # 断言 S.TribonacciConstant 不是有理数
    assert ask(Q.rational(z)) is False
    # 断言 S.TribonacciConstant 是代数数
    assert ask(Q.algebraic(z)) is True
    # 断言 S.TribonacciConstant 是实数
    assert ask(Q.real(z)) is True
    # 断言 S.TribonacciConstant 是复数
    assert ask(Q.complex(z)) is True
    # 断言 S.TribonacciConstant 是无理数
    assert ask(Q.irrational(z)) is True
    # 断言 S.TribonacciConstant 不是虚数
    assert ask(Q.imaginary(z)) is False
    # 断言 S.TribonacciConstant 是正数
    assert ask(Q.positive(z)) is True
    # 断言 S.TribonacciConstant 不是负数
    assert ask(Q.negative(z)) is False
    # 断言 S.TribonacciConstant 不是偶数
    assert ask(Q.even(z)) is False
    # 断言 S.TribonacciConstant 不是奇数
    assert ask(Q.odd(z)) is False
    # 断言 S.TribonacciConstant 是有限数
    assert ask(Q.finite(z)) is True
    # 断言 S.TribonacciConstant 不是素数
    assert ask(Q.prime(z)) is False
    # 断言 S.TribonacciConstant 不是合数
    assert ask(Q.composite(z)) is False
    # 断言 S.TribonacciConstant 是厄米矩阵
    assert ask(Q.hermitian(z)) is True
    # 断言 S.TribonacciConstant 不是反厄米矩阵
    assert ask(Q.antihermitian(z)) is False


# 定义函数 test_I，用于测试虚数单位 I 和复数 1 + I 的性质
def test_I():
    # 获取虚数单位 I 的值并进行下列断言
    z = I
    # 断言虚数单位 I 是可交换的
    assert ask(Q.commutative(z)) is True
    # 断言虚数单位 I 不是整数
    assert ask(Q.integer(z)) is False
    # 断言虚数单位 I 不是有理数
    assert ask(Q.rational(z)) is False
    # 断言虚数单位 I 是代数数
    assert ask(Q.algebraic(z)) is True
    # 断言虚数单位 I 不是实数
    assert ask(Q.real(z)) is False
    # 断言虚数单位 I 是复数
    assert ask(Q.complex(z)) is True
    # 断言虚数单位 I 不是无理数
    assert ask(Q.irrational(z)) is False
    # 断言虚数单位 I 是虚数
    assert ask(Q.imaginary(z)) is True
    # 断言虚数单位 I 不是正数
    assert ask(Q.positive(z)) is False
    # 断言虚数单位 I 不是负数
    assert ask(Q.negative(z)) is False
    # 断言虚数单位 I 不是偶数
    assert ask(Q.even(z)) is False
    # 断言虚数单位 I 不是奇数
    assert ask(Q.odd(z)) is False
    # 断言虚数单位 I 是有限数
    assert ask(Q.finite(z)) is True
    # 断言虚数单位 I 不是素数
    assert ask(Q.prime(z)) is False
    # 断言虚数单位 I 不是合数
    assert ask(Q.composite(z)) is False
    # 断言虚数单位 I 不是厄米矩阵
    assert ask(Q.hermitian(z)) is False
    # 断言虚数单位 I 是
    # 对给定复数 z 进行多个属性的逻辑断言

    assert ask(Q.complex(z)) is True
    # 检查 z 是否为复数

    assert ask(Q.irrational(z)) is False
    # 检查 z 是否为无理数

    assert ask(Q.imaginary(z)) is False
    # 检查 z 是否为虚数

    assert ask(Q.positive(z)) is False
    # 检查 z 是否为正数

    assert ask(Q.negative(z)) is False
    # 检查 z 是否为负数

    assert ask(Q.even(z)) is False
    # 检查 z 是否为偶数

    assert ask(Q.odd(z)) is False
    # 检查 z 是否为奇数

    assert ask(Q.finite(z)) is True
    # 检查 z 是否为有限数

    assert ask(Q.prime(z)) is False
    # 检查 z 是否为素数

    assert ask(Q.composite(z)) is False
    # 检查 z 是否为合数

    assert ask(Q.hermitian(z)) is False
    # 检查 z 是否为埃尔米特矩阵

    assert ask(Q.antihermitian(z)) is False
    # 检查 z 是否为反埃尔米特矩阵

    z = I*(1 + I)
    # 重新赋值 z

    assert ask(Q.commutative(z)) is True
    # 检查 z 是否为可交换的

    assert ask(Q.integer(z)) is False
    # 检查 z 是否为整数

    assert ask(Q.rational(z)) is False
    # 检查 z 是否为有理数

    assert ask(Q.algebraic(z)) is True
    # 检查 z 是否为代数数

    assert ask(Q.real(z)) is False
    # 检查 z 是否为实数

    assert ask(Q.complex(z)) is True
    # 检查 z 是否为复数

    assert ask(Q.irrational(z)) is False
    # 检查 z 是否为无理数

    assert ask(Q.imaginary(z)) is False
    # 检查 z 是否为虚数

    assert ask(Q.positive(z)) is False
    # 检查 z 是否为正数

    assert ask(Q.negative(z)) is False
    # 检查 z 是否为负数

    assert ask(Q.even(z)) is False
    # 检查 z 是否为偶数

    assert ask(Q.odd(z)) is False
    # 检查 z 是否为奇数

    assert ask(Q.finite(z)) is True
    # 检查 z 是否为有限数

    assert ask(Q.prime(z)) is False
    # 检查 z 是否为素数

    assert ask(Q.composite(z)) is False
    # 检查 z 是否为合数

    assert ask(Q.hermitian(z)) is False
    # 检查 z 是否为埃尔米特矩阵

    assert ask(Q.antihermitian(z)) is False
    # 检查 z 是否为反埃尔米特矩阵

    z = I**(I)
    # 重新赋值 z

    assert ask(Q.imaginary(z)) is False
    # 检查 z 是否为虚数

    assert ask(Q.real(z)) is True
    # 检查 z 是否为实数

    z = (-I)**(I)
    # 重新赋值 z

    assert ask(Q.imaginary(z)) is False
    # 检查 z 是否为虚数

    assert ask(Q.real(z)) is True
    # 检查 z 是否为实数

    z = (3*I)**(I)
    # 重新赋值 z

    assert ask(Q.imaginary(z)) is False
    # 检查 z 是否为虚数

    assert ask(Q.real(z)) is False
    # 检查 z 是否为实数

    z = (1)**(I)
    # 重新赋值 z

    assert ask(Q.imaginary(z)) is False
    # 检查 z 是否为虚数

    assert ask(Q.real(z)) is True
    # 检查 z 是否为实数

    z = (-1)**(I)
    # 重新赋值 z

    assert ask(Q.imaginary(z)) is False
    # 检查 z 是否为虚数

    assert ask(Q.real(z)) is True
    # 检查 z 是否为实数

    z = (1+I)**(I)
    # 重新赋值 z

    assert ask(Q.imaginary(z)) is False
    # 检查 z 是否为虚数

    assert ask(Q.real(z)) is False
    # 检查 z 是否为实数

    z = (I)**(I+3)
    # 重新赋值 z

    assert ask(Q.imaginary(z)) is True
    # 检查 z 是否为虚数

    assert ask(Q.real(z)) is False
    # 检查 z 是否为实数

    z = (I)**(I+2)
    # 重新赋值 z

    assert ask(Q.imaginary(z)) is False
    # 检查 z 是否为虚数

    assert ask(Q.real(z)) is True
    # 检查 z 是否为实数

    z = (I)**(2)
    # 重新赋值 z

    assert ask(Q.imaginary(z)) is False
    # 检查 z 是否为虚数

    assert ask(Q.real(z)) is True
    # 检查 z 是否为实数

    z = (I)**(3)
    # 重新赋值 z

    assert ask(Q.imaginary(z)) is True
    # 检查 z 是否为虚数

    assert ask(Q.real(z)) is False
    # 检查 z 是否为实数

    z = (3)**(I)
    # 重新赋值 z

    assert ask(Q.imaginary(z)) is False
    # 检查 z 是否为虚数

    assert ask(Q.real(z)) is False
    # 检查 z 是否为实数

    z = (I)**(0)
    # 重新赋值 z

    assert ask(Q.imaginary(z)) is False
    # 检查 z 是否为虚数

    assert ask(Q.real(z)) is True
    # 检查 z 是否为实数
# 定义一个测试函数 test_bounded
def test_bounded():
    # 创建符号变量 x, y, z
    x, y, z = symbols('x,y,z')
    # 计算 x + y 并赋值给 a
    a = x + y
    # 将 a 的参数赋值给 x, y
    x, y = a.args
    # 断言 a 是否是有限的且 y 是正无穷大的
    assert ask(Q.finite(a), Q.positive_infinite(y)) is None
    # 断言 x 是否是有限的
    assert ask(Q.finite(x)) is None
    # 断言 x 是否是有限的且 x 自身也是有限的
    assert ask(Q.finite(x), Q.finite(x)) is True
    # 断言 x 是否是有限的且 y 也是有限的
    assert ask(Q.finite(x), Q.finite(y)) is None
    # 断言 x 是否是有限的且 x 是复数
    assert ask(Q.finite(x), Q.complex(x)) is True
    # 断言 x 是否是有限的且 x 是扩展实数集中的元素
    assert ask(Q.finite(x), Q.extended_real(x)) is None

    # 断言 x+1 是否是有限的
    assert ask(Q.finite(x + 1)) is None
    # 断言 x+1 是否是有限的且 x 是有限的
    assert ask(Q.finite(x + 1), Q.finite(x)) is True
    # 重新计算 a = x + y 并将其参数重新赋值给 x, y
    a = x + y
    x, y = a.args
    # 断言 a 是否是有限的且 x, y 都是有限的
    assert ask(Q.finite(a), Q.finite(x) & Q.finite(y)) is True
    # 断言 a 是否是有限的且 x 是正数且 y 是有限的
    assert ask(Q.finite(a), Q.positive(x) & Q.finite(y)) is True
    # 断言 a 是否是有限的且 x, y 都是有限的且 x 是正数
    assert ask(Q.finite(a), Q.finite(x) & Q.positive(y)) is True
    # 断言 a 是否是有限的且 x, y 都是正数
    assert ask(Q.finite(a), Q.positive(x) & Q.positive(y)) is True
    # 断言 a 是否是有限的且 x 是正数且 y 是有限的且 y 不是正数
    assert ask(Q.finite(a), Q.positive(x) & Q.finite(y)
        & ~Q.positive(y)) is True
    # 断言 a 是否是有限的且 x, y 都是有限的且 x 不是正数且 y 是正数
    assert ask(Q.finite(a), Q.finite(x) & ~Q.positive(x)
        & Q.positive(y)) is True
    # 断言 a 是否是有限的且 x, y 都是有限的且 x 不是正数且 y 不是正数
    assert ask(Q.finite(a), Q.finite(x) & Q.finite(y) & ~Q.positive(x)
        & ~Q.positive(y)) is True
    # 断言 a 是否是有限的且 x 是有限的且 y 不是有限的
    assert ask(Q.finite(a), Q.finite(x) & ~Q.finite(y)) is False
    # 断言 a 是否是有限的且 x 是正数且 y 不是有限的
    assert ask(Q.finite(a), Q.positive(x) & ~Q.finite(y)) is False
    # 断言 a 是否是有限的且 x 是有限的且 y 是正无穷大的
    assert ask(Q.finite(a), Q.finite(x)
        & Q.positive_infinite(y)) is False
    # 断言 a 是否是有限的且 x 是正数且 y 是正无穷大的
    assert ask(Q.finite(a), Q.positive(x)
        & Q.positive_infinite(y)) is False
    # 断言 a 是否是有限的且 x 是正数且 y 不是有限的且 y 不是正数
    assert ask(Q.finite(a), Q.positive(x) & ~Q.finite(y)
        & ~Q.positive(y)) is False
    # 断言 a 是否是有限的且 x 是有限的且 x 不是正数且 y 是正无穷大的
    assert ask(Q.finite(a), Q.finite(x) & ~Q.positive(x)
        & Q.positive_infinite(y)) is False
    # 断言 a 是否是有限的且 x 是有限的且 x 不是正数且 y 不是有限的且 y 不是正数
    assert ask(Q.finite(a), Q.finite(x) & ~Q.positive(x) & ~Q.finite(y)
        & ~Q.positive(y)) is False
    # 断言 a 是否是有限的且 x 是有限的且 y 是扩展正数集中的元素
    assert ask(Q.finite(a), Q.finite(x) & Q.extended_positive(y)) is None
    # 断言 a 是否是有限的且 x 是正数且 y 是扩展正数集中的元素
    assert ask(Q.finite(a), Q.positive(x) & Q.extended_positive(y)) is None
    # 断言 a 是否是有限的且 x 是正数且 y 不是正数
    assert ask(Q.finite(a), Q.positive(x) & ~Q.positive(y)) is None
    # 断言 a 是否是有限的且 x 是有限的且 x 不是正数且 y 是扩展正数集中的元素
    assert ask(Q.finite(a), Q.finite(x) & ~Q.positive(x)
        & Q.extended_positive(y)) is None
    # 断言 a 是否是有限的且 x 是有限的且 x 不是正数且 y 不是正数
    assert ask(Q.finite(a), Q.finite(x) & ~Q.positive(x)
        & ~Q.positive(y)) is None
    # 断言 a 是否是有限的且 x, y 都不是有限的
    assert ask(Q.finite(a), ~Q.finite(x) & ~Q.finite(y)) is None
    # 断言 a 是否是有限的且 x 是正无穷大的且 y 不是有限的
    assert ask(Q.finite(a), Q.positive_infinite(x)
        & ~Q.finite(y)) is None
    # 断言 a 是否是有限的且 x 不是有限的且 y 是正无穷大的
    assert ask(Q.finite(a), ~Q.finite(x)
        & Q.positive_infinite(y)) is None
    # 断言 a 是否是有限的且 x 是正无穷大的且 y 是正无穷大的
    assert ask(Q.finite(a), Q.positive_infinite(x) & Q.positive_infinite(y)) is False
    # 断言 a 是否是有限的且 x 是正无穷大的且 y 不是有限的且 y 不是扩展正数集中的元素
    assert ask(Q.finite(a), Q.positive_infinite(x) & ~Q.finite(y)
        & ~Q.extended_positive(y)) is None
    # 断言 a 是否是有限的且 x 不是有限的且 x 不是扩展正数集中的元素且 y 是正无穷大的
    assert ask(Q.finite(a), ~Q.finite(x) & ~Q.extended_positive(x)
        & Q.positive_infinite(y)) is None
    # 断言 a 是否是有限的且 x 不是有限的且 y 不是有限的且 x 不是扩展正数集中的元素且 y 不是扩展正数集中的元素
    assert ask(Q.finite(a), ~Q.finite(x) & ~Q.finite(y)
        & ~Q.extended_positive(x) & ~Q.extended_positive(y)) is False
    # 断言 a 是否是有限的且 y 不是有限的
    assert ask(Q.finite(a), ~Q.finite(y)) is None
    # 断言：对于给定的 a，要求在以下条件下成立：a 是有限的，x 是扩展正数，y 不是有限的。
    assert ask(Q.finite(a), Q.extended_positive(x) & ~Q.finite(y)) is None
    # 断言：对于给定的 a，要求在以下条件下成立：a 是有限的，y 是正无穷。
    assert ask(Q.finite(a), Q.positive_infinite(y)) is None
    # 断言：对于给定的 a，要求在以下条件下成立：a 是有限的，x 是扩展正数，y 是正无穷。
    assert ask(Q.finite(a), Q.extended_positive(x) & Q.positive_infinite(y)) is False
    # 断言：对于给定的 a，要求在以下条件下成立：a 是有限的，x 是扩展正数，y 不是有限的，y 不是扩展正数。
    assert ask(Q.finite(a), Q.extended_positive(x) & ~Q.finite(y) & ~Q.extended_positive(y)) is None
    # 断言：对于给定的 a，要求在以下条件下成立：a 是有限的，x 不是扩展正数，y 是正无穷。
    assert ask(Q.finite(a), ~Q.extended_positive(x) & Q.positive_infinite(y)) is None
    # 断言：对于给定的 a，要求在以下条件下成立：a 是有限的，x 不是扩展正数，y 不是有限的，y 不是扩展正数。
    assert ask(Q.finite(a), ~Q.extended_positive(x) & ~Q.finite(y) & ~Q.extended_positive(y)) is False
    # 断言：对于给定的 a，要求在以下条件下成立：a 是有限的，无其他条件。
    assert ask(Q.finite(a)) is None
    # 断言：对于给定的 a，要求在以下条件下成立：a 是有限的，x 是扩展正数。
    assert ask(Q.finite(a), Q.extended_positive(x)) is None
    # 断言：对于给定的 a，要求在以下条件下成立：a 是有限的，y 是扩展正数。
    assert ask(Q.finite(a), Q.extended_positive(y)) is None
    # 断言：对于给定的 a，要求在以下条件下成立：a 是有限的，x 是扩展正数，y 是扩展正数。
    assert ask(Q.finite(a), Q.extended_positive(x) & Q.extended_positive(y)) is None
    # 断言：对于给定的 a，要求在以下条件下成立：a 是有限的，x 是扩展正数，y 不是扩展正数。
    assert ask(Q.finite(a), Q.extended_positive(x) & ~Q.extended_positive(y)) is None
    # 断言：对于给定的 a，要求在以下条件下成立：a 是有限的，x 不是扩展正数，y 是扩展正数。
    assert ask(Q.finite(a), ~Q.extended_positive(x) & Q.extended_positive(y)) is None
    # 断言：对于给定的 a，要求在以下条件下成立：a 是有限的，x 不是扩展正数，y 不是扩展正数。
    assert ask(Q.finite(a), ~Q.extended_positive(x) & ~Q.extended_positive(y)) is None

    # 定义符号变量 x, y, z
    x, y, z = symbols('x,y,z')
    # 定义符号表达式 a
    a = x + y + z
    # 解构 a 的参数为 x, y, z
    x, y, z = a.args
    # 断言：对于给定的 a，要求在以下条件下成立：a 是有限的，x 是负数，y 是负数，z 是负数。
    assert ask(Q.finite(a), Q.negative(x) & Q.negative(y) & Q.negative(z)) is True
    # 断言：对于给定的 a，要求在以下条件下成立：a 是有限的，x 是负数，y 是负数，z 是有限的。
    assert ask(Q.finite(a), Q.negative(x) & Q.negative(y) & Q.finite(z)) is True
    # 断言：对于给定的 a，要求在以下条件下成立：a 是有限的，x 是负数，y 是负数，z 是正数。
    assert ask(Q.finite(a), Q.negative(x) & Q.negative(y) & Q.positive(z)) is True
    # 断言：对于给定的 a，要求在以下条件下成立：a 是有限的，x 是负数，y 是负数，z 是负无穷。
    assert ask(Q.finite(a), Q.negative(x) & Q.negative(y) & Q.negative_infinite(z)) is False
    # 断言：对于给定的 a，要求在以下条件下成立：a 是有限的，x 是负数，y 是负数，z 不是有限的。
    assert ask(Q.finite(a), Q.negative(x) & Q.negative(y) & ~Q.finite(z)) is False
    # 断言：对于给定的 a，要求在以下条件下成立：a 是有限的，x 是负数，y 是负数，z 是正无穷。
    assert ask(Q.finite(a), Q.negative(x) & Q.negative(y) & Q.positive_infinite(z)) is False
    # 断言：对于给定的 a，要求在以下条件下成立：a 是有限的，x 是负数，y 是负数，z 是扩展负数。
    assert ask(Q.finite(a), Q.negative(x) & Q.negative(y) & Q.extended_negative(z)) is None
    # 断言：对于给定的 a，要求在以下条件下成立：a 是有限的，x 是负数，y 是负数。
    assert ask(Q.finite(a), Q.negative(x) & Q.negative(y)) is None
    # 断言：对于给定的 a，要求在以下条件下成立：a 是有限的，x 是负数，y 是负数，z 是扩展正数。
    assert ask(Q.finite(a), Q.negative(x) & Q.negative(y) & Q.extended_positive(z)) is None
    # 断言：对于给定的 a，要求在以下条件下成立：a 是有限的，x 是负数，y 是有限的，z 是有限的。
    assert ask(Q.finite(a), Q.negative(x) & Q.finite(y) & Q.finite(z)) is True
    # 断言：对于给定的 a，要求在以下条件下成立：a 是有限的，x 是负数，y 是有限的，z 是正数。
    assert ask(Q.finite(a), Q.negative(x) & Q.finite(y) & Q.positive(z)) is True
    # 断言：对于给定的 a，要求在以下条件下成立：a 是有限的，x 是负数，y 是有限的，z 是负无穷。
    assert ask(Q.finite(a), Q.negative(x) & Q.finite(y) & Q.negative_infinite(z)) is False
    # 断言：对于给定的 a，要求在以下条件下成立：a 是有限的，x 是负数，y 是有限的，z 不是有限的。
    assert ask(Q.finite(a), Q.negative(x) & Q.finite(y) & ~Q.finite(z)) is False
    # 断言：对于给定的 a，要求在以下条件下成立：a 是有限的，x 是负数，y 是有限的，z 是正无穷。
    assert ask(Q.finite(a), Q.negative(x) & Q.finite(y) & Q.positive_infinite(z)) is False
    # 断言：对于给定的 a，要求在以下条件下成立：a 是有限的，x 是负数，y 是有限的，z 是扩展负数。
    assert ask(Q.finite(a), Q.negative(x) & Q.finite(y
    # 断言语句，验证逻辑表达式是否为 False
    assert ask(Q.finite(a), Q.negative(x) & Q.positive(y)
        & ~Q.finite(z)) is False
    # 断言语句，验证逻辑表达式是否为 False
    assert ask(Q.finite(a), Q.negative(x) & Q.positive(y)
        & Q.positive_infinite(z)) is False
    # 断言语句，验证逻辑表达式是否为 None
    assert ask(Q.finite(a), Q.negative(x) & Q.positive(y)
        & Q.extended_negative(z)) is None
    # 断言语句，验证逻辑表达式是否为 None
    assert ask(Q.finite(a), Q.negative(x) & Q.extended_positive(y)
        & Q.finite(y)) is None
    # 断言语句，验证逻辑表达式是否为 None
    assert ask(Q.finite(a), Q.negative(x) & Q.positive(y)
        & Q.extended_positive(z)) is None
    # 断言语句，验证逻辑表达式是否为 False
    assert ask(Q.finite(a), Q.negative(x) & Q.negative_infinite(y)
        & Q.negative_infinite(z)) is False
    # 断言语句，验证逻辑表达式是否为 None
    assert ask(Q.finite(a), Q.negative(x) & Q.negative_infinite(y)
        & ~Q.finite(z)) is None
    # 断言语句，验证逻辑表达式是否为 None
    assert ask(Q.finite(a), Q.negative(x) & Q.negative_infinite(y)
        & Q.positive_infinite(z)) is None
    # 断言语句，验证逻辑表达式是否为 False
    assert ask(Q.finite(a), Q.negative(x) & Q.negative_infinite(y)
        & Q.extended_negative(z)) is False
    # 断言语句，验证逻辑表达式是否为 None
    assert ask(Q.finite(a), Q.negative(x)
        & Q.negative_infinite(y)) is None
    # 断言语句，验证逻辑表达式是否为 None
    assert ask(Q.finite(a), Q.negative(x) & Q.negative_infinite(y)
        & Q.extended_positive(z)) is None
    # 断言语句，验证逻辑表达式是否为 None
    assert ask(Q.finite(a), Q.negative(x) & ~Q.finite(y)
        & ~Q.finite(z)) is None
    # 断言语句，验证逻辑表达式是否为 None
    assert ask(Q.finite(a), Q.negative(x) & ~Q.finite(y)
        & Q.positive_infinite(z)) is None
    # 断言语句，验证逻辑表达式是否为 None
    assert ask(Q.finite(a), Q.negative(x) & ~Q.finite(y)
        & Q.extended_negative(z)) is None
    # 断言语句，验证逻辑表达式是否为 None
    assert ask(Q.finite(a), Q.negative(x) & ~Q.finite(y)) is None
    # 断言语句，验证逻辑表达式是否为 None
    assert ask(Q.finite(a), Q.negative(x) & ~Q.finite(y)
        & Q.extended_positive(z)) is None
    # 断言语句，验证逻辑表达式是否为 False
    assert ask(Q.finite(a), Q.negative(x) & Q.positive_infinite(y)
        & Q.positive_infinite(z)) is False
    # 断言语句，验证逻辑表达式是否为 None
    assert ask(Q.finite(a), Q.negative(x) & Q.positive_infinite(y)
        & Q.negative_infinite(z)) is None
    # 断言语句，验证逻辑表达式是否为 None
    assert ask(Q.finite(a), Q.negative(x) &
         Q.positive_infinite(y)) is None
    # 断言语句，验证逻辑表达式是否为 False
    assert ask(Q.finite(a), Q.negative(x) & Q.positive_infinite(y)
        & Q.extended_positive(z)) is False
    # 断言语句，验证逻辑表达式是否为 None
    assert ask(Q.finite(a), Q.negative(x) & Q.extended_negative(y)
        & Q.extended_negative(z)) is None
    # 断言语句，验证逻辑表达式是否为 None
    assert ask(Q.finite(a), Q.negative(x)
        & Q.extended_negative(y)) is None
    # 断言语句，验证逻辑表达式是否为 None
    assert ask(Q.finite(a), Q.negative(x) & Q.extended_negative(y)
        & Q.extended_positive(z)) is None
    # 断言语句，验证逻辑表达式是否为 None
    assert ask(Q.finite(a), Q.negative(x)) is None
    # 断言语句，验证逻辑表达式是否为 None
    assert ask(Q.finite(a), Q.negative(x)
        & Q.extended_positive(z)) is None
    # 断言语句，验证逻辑表达式是否为 None
    assert ask(Q.finite(a), Q.negative(x) & Q.extended_positive(y)
        & Q.extended_positive(z)) is None
    # 断言语句，验证逻辑表达式是否为 True
    assert ask(Q.finite(a), Q.finite(x) & Q.finite(y)
        & Q.finite(z)) is True
    # 断言语句，验证逻辑表达式是否为 True
    assert ask(Q.finite(a), Q.finite(x) & Q.finite(y)
        & Q.positive(z)) is True
    # 断言语句，验证逻辑表达式是否为 False
    assert ask(Q.finite(a), Q.finite(x) & Q.finite(y)
        & Q.negative_infinite(z)) is False
    # 断言语句，验证逻辑表达式是否为 False
    assert ask(Q.finite(a), Q.finite(x) & Q.finite(y)
        & ~Q.finite(z)) is False
    # 断言语句，验证逻辑表达式是否为 False
    assert ask(Q.finite(a), Q.finite(x) & Q.finite(y)
        & Q.positive_infinite(z)) is False
    # 确保对于给定的 Q.finite(a) 条件，如果同时满足 Q.finite(x) 和 Q.finite(y) 和 Q.extended_negative(z)，返回 None
    assert ask(Q.finite(a), Q.finite(x) & Q.finite(y)
        & Q.extended_negative(z)) is None

    # 确保对于给定的 Q.finite(a) 条件，如果同时满足 Q.finite(x) 和 Q.finite(y)，返回 None
    assert ask(Q.finite(a), Q.finite(x) & Q.finite(y)) is None

    # 确保对于给定的 Q.finite(a) 条件，如果同时满足 Q.finite(x) 和 Q.finite(y) 和 Q.extended_positive(z)，返回 None
    assert ask(Q.finite(a), Q.finite(x) & Q.finite(y)
        & Q.extended_positive(z)) is None

    # 确保对于给定的 Q.finite(a) 条件，如果同时满足 Q.finite(x) 和 Q.positive(y) 和 Q.positive(z)，返回 True
    assert ask(Q.finite(a), Q.finite(x) & Q.positive(y)
        & Q.positive(z)) is True

    # 确保对于给定的 Q.finite(a) 条件，如果同时满足 Q.finite(x) 和 Q.positive(y) 和 Q.negative_infinite(z)，返回 False
    assert ask(Q.finite(a), Q.finite(x) & Q.positive(y)
        & Q.negative_infinite(z)) is False

    # 确保对于给定的 Q.finite(a) 条件，如果同时满足 Q.finite(x) 和 Q.positive(y) 和 ~Q.finite(z)，返回 False
    assert ask(Q.finite(a), Q.finite(x) & Q.positive(y)
        & ~Q.finite(z)) is False

    # 确保对于给定的 Q.finite(a) 条件，如果同时满足 Q.finite(x) 和 Q.positive(y) 和 Q.positive_infinite(z)，返回 False
    assert ask(Q.finite(a), Q.finite(x) & Q.positive(y)
        & Q.positive_infinite(z)) is False

    # 确保对于给定的 Q.finite(a) 条件，如果同时满足 Q.finite(x) 和 Q.positive(y) 和 Q.extended_negative(z)，返回 None
    assert ask(Q.finite(a), Q.finite(x) & Q.positive(y)
        & Q.extended_negative(z)) is None

    # 确保对于给定的 Q.finite(a) 条件，如果同时满足 Q.finite(x) 和 Q.positive(y)，返回 None
    assert ask(Q.finite(a), Q.finite(x) & Q.positive(y)) is None

    # 确保对于给定的 Q.finite(a) 条件，如果同时满足 Q.finite(x) 和 Q.positive(y) 和 Q.extended_positive(z)，返回 None
    assert ask(Q.finite(a), Q.finite(x) & Q.positive(y)
        & Q.extended_positive(z)) is None

    # 确保对于给定的 Q.finite(a) 条件，如果同时满足 Q.finite(x) 和 Q.negative_infinite(y) 和 Q.negative_infinite(z)，返回 False
    assert ask(Q.finite(a), Q.finite(x) & Q.negative_infinite(y)
        & Q.negative_infinite(z)) is False

    # 确保对于给定的 Q.finite(a) 条件，如果同时满足 Q.finite(x) 和 Q.negative_infinite(y) 和 ~Q.finite(z)，返回 None
    assert ask(Q.finite(a), Q.finite(x) & Q.negative_infinite(y)
        & ~Q.finite(z)) is None

    # 确保对于给定的 Q.finite(a) 条件，如果同时满足 Q.finite(x) 和 Q.negative_infinite(y) 和 Q.positive_infinite(z)，返回 None
    assert ask(Q.finite(a), Q.finite(x) & Q.negative_infinite(y)
        & Q.positive_infinite(z)) is None

    # 确保对于给定的 Q.finite(a) 条件，如果同时满足 Q.finite(x) 和 Q.negative_infinite(y) 和 Q.extended_negative(z)，返回 False
    assert ask(Q.finite(a), Q.finite(x) & Q.negative_infinite(y)
        & Q.extended_negative(z)) is False

    # 确保对于给定的 Q.finite(a) 条件，如果同时满足 Q.finite(x) 和 Q.negative_infinite(y)，返回 None
    assert ask(Q.finite(a), Q.finite(x)
        & Q.negative_infinite(y)) is None

    # 确保对于给定的 Q.finite(a) 条件，如果同时满足 Q.finite(x) 和 Q.negative_infinite(y) 和 Q.extended_positive(z)，返回 None
    assert ask(Q.finite(a), Q.finite(x) & Q.negative_infinite(y)
        & Q.extended_positive(z)) is None

    # 确保对于给定的 Q.finite(a) 条件，如果同时满足 Q.finite(x) 和 ~Q.finite(y) 和 ~Q.finite(z)，返回 None
    assert ask(Q.finite(a), Q.finite(x) & ~Q.finite(y)
        & ~Q.finite(z)) is None

    # 确保对于给定的 Q.finite(a) 条件，如果同时满足 Q.finite(x) 和 ~Q.finite(y) 和 Q.positive_infinite(z)，返回 None
    assert ask(Q.finite(a), Q.finite(x) & ~Q.finite(y)
        & Q.positive_infinite(z)) is None

    # 确保对于给定的 Q.finite(a) 条件，如果同时满足 Q.finite(x) 和 ~Q.finite(y) 和 Q.extended_negative(z)，返回 None
    assert ask(Q.finite(a), Q.finite(x) & ~Q.finite(y)
        & Q.extended_negative(z)) is None

    # 确保对于给定的 Q.finite(a) 条件，如果同时满足 Q.finite(x) 和 ~Q.finite(y)，返回 None
    assert ask(Q.finite(a), Q.finite(x) & ~Q.finite(y)) is None

    # 确保对于给定的 Q.finite(a) 条件，如果同时满足 Q.finite(x) 和 ~Q.finite(y) 和 Q.extended_positive(z)，返回 None
    assert ask(Q.finite(a), Q.finite(x) & ~Q.finite(y)
        & Q.extended_positive(z)) is None

    # 确保对于给定的 Q.finite(a) 条件，如果同时满足 Q.finite(x) 和 Q.positive_infinite(y) 和 Q.positive_infinite(z)，返回 False
    assert ask(Q.finite(a), Q.finite(x) & Q.positive_infinite(y)
        & Q.positive_infinite(z)) is False

    # 确保对于给定的 Q.finite(a) 条件，如果同时满足 Q.finite(x) 和 Q.positive_infinite(y) 和 Q.extended_negative(z)，返回 None
    assert ask(Q.finite(a), Q.finite(x)
        & Q.positive_infinite(y)) is None

    # 确保对于给定的 Q.finite(a) 条件，如果同时满足 Q.finite(x) 和 Q.positive_infinite(y) 和 Q.extended_positive(z)，返回 False
    assert ask(Q.finite(a), Q.finite(x) & Q.positive_infinite(y)
        & Q.extended_positive(z)) is False

    # 确保对于给定的 Q.finite(a) 条件，如果同时满足 Q.finite(x) 和 Q.extended_negative(y) 和 Q.extended_negative(z)，返回 None
    assert ask(Q.finite(a), Q.finite(x)
        & Q.extended_negative(y)) is None

    # 确保对于给定的 Q.finite(a) 条件，如果同时满足 Q.finite(x) 和 Q.extended_negative(y) 和 Q.extended_positive(z)，返回 None
    assert ask(Q.finite(a), Q.finite(x) & Q.extended_negative(y)
        & Q.extended_positive(z)) is None

    # 确保对于给定的 Q.finite(a) 条件，如果同时满足 Q.finite(x)，返回 None
    assert ask(Q.finite(a), Q.finite(x)) is None

    # 确保
    # 断言：如果 a 是有限的，并且 x 和 y 都是正数，并且 z 是负无穷，则返回 False
    assert ask(Q.finite(a), Q.positive(x) & Q.positive(y)
        & Q.negative_infinite(z)) is False
    
    # 断言：如果 a 是有限的，并且 x 和 y 都是正数，并且 z 不是有限的，则返回 False
    assert ask(Q.finite(a), Q.positive(x) & Q.positive(y)
        & ~Q.finite(z)) is False
    
    # 断言：如果 a 是有限的，并且 x 和 y 都是正数，并且 z 是正无穷，则返回 False
    assert ask(Q.finite(a), Q.positive(x) & Q.positive(y)
        & Q.positive_infinite(z)) is False
    
    # 断言：如果 a 是有限的，并且 x 和 y 都是正数，并且 z 是扩展负数，则返回 None
    assert ask(Q.finite(a), Q.positive(x) & Q.positive(y)
        & Q.extended_negative(z)) is None
    
    # 断言：如果 a 是有限的，并且 x 和 y 都是正数，则返回 None
    assert ask(Q.finite(a), Q.positive(x) & Q.positive(y)) is None
    
    # 断言：如果 a 是有限的，并且 x 是正数，并且 y 是扩展正数，则返回 None
    assert ask(Q.finite(a), Q.positive(x) & Q.positive(y)
        & Q.extended_positive(z)) is None
    
    # 断言：如果 a 是有限的，并且 x 是正数，并且 y 和 z 都是负无穷，则返回 False
    assert ask(Q.finite(a), Q.positive(x) & Q.negative_infinite(y)
        & Q.negative_infinite(z)) is False
    
    # 断言：如果 a 是有限的，并且 x 是正数，并且 y 是负无穷，而 z 不是有限的，则返回 None
    assert ask(Q.finite(a), Q.positive(x) & Q.negative_infinite(y)
        & ~Q.finite(z)) is None
    
    # 断言：如果 a 是有限的，并且 x 是正数，并且 y 是负无穷，而 z 是正无穷，则返回 None
    assert ask(Q.finite(a), Q.positive(x) & Q.negative_infinite(y)
        & Q.positive_infinite(z)) is None
    
    # 断言：如果 a 是有限的，并且 x 是正数，并且 y 是负无穷，而 z 是扩展负数，则返回 False
    assert ask(Q.finite(a), Q.positive(x) & Q.negative_infinite(y)
        & Q.extended_negative(z)) is False
    
    # 断言：如果 a 是有限的，并且 x 是正数，并且 y 是负无穷，则返回 None
    assert ask(Q.finite(a), Q.positive(x)
        & Q.negative_infinite(y)) is None
    
    # 断言：如果 a 是有限的，并且 x 是正数，并且 y 是负无穷，而 z 是扩展正数，则返回 None
    assert ask(Q.finite(a), Q.positive(x) & Q.negative_infinite(y)
        & Q.extended_positive(z)) is None
    
    # 断言：如果 a 是有限的，并且 x 是正数，并且 y 和 z 都不是有限的，则返回 None
    assert ask(Q.finite(a), Q.positive(x) & ~Q.finite(y)
        & ~Q.finite(z)) is None
    
    # 断言：如果 a 是有限的，并且 x 是正数，并且 y 不是有限的，而 z 是正无穷，则返回 None
    assert ask(Q.finite(a), Q.positive(x) & ~Q.finite(y)
        & Q.positive_infinite(z)) is None
    
    # 断言：如果 a 是有限的，并且 x 是正数，并且 y 不是有限的，而 z 是扩展负数，则返回 None
    assert ask(Q.finite(a), Q.positive(x) & ~Q.finite(y)
        & Q.extended_negative(z)) is None
    
    # 断言：如果 a 是有限的，并且 x 是正数，并且 y 不是有限的，则返回 None
    assert ask(Q.finite(a), Q.positive(x) & ~Q.finite(y)) is None
    
    # 断言：如果 a 是有限的，并且 x 是正数，并且 y 不是有限的，而 z 是扩展正数，则返回 None
    assert ask(Q.finite(a), Q.positive(x) & ~Q.finite(y)
        & Q.extended_positive(z)) is None
    
    # 断言：如果 a 是有限的，并且 x 是正数，并且 y 是正无穷，而 z 也是正无穷，则返回 False
    assert ask(Q.finite(a), Q.positive(x) & Q.positive_infinite(y)
        & Q.positive_infinite(z)) is False
    
    # 断言：如果 a 是有限的，并且 x 是正数，并且 y 是正无穷，而 z 是扩展负数，则返回 None
    assert ask(Q.finite(a), Q.positive(x) & Q.positive_infinite(y)
        & Q.extended_negative(z)) is None
    
    # 断言：如果 a 是有限的，并且 x 是正数，并且 y 是正无穷，则返回 None
    assert ask(Q.finite(a), Q.positive(x)
        & Q.positive_infinite(y)) is None
    
    # 断言：如果 a 是有限的，并且 x 是正数，并且 y 是正无穷，而 z 是扩展正数，则返回 False
    assert ask(Q.finite(a), Q.positive(x) & Q.positive_infinite(y)
        & Q.extended_positive(z)) is False
    
    # 断言：如果 a 是有限的，并且 x 是正数，并且 y 是扩展负数，而 z 也是扩展负数，则返回 None
    assert ask(Q.finite(a), Q.positive(x) & Q.extended_negative(y)
        & Q.extended_negative(z)) is None
    
    # 断言：如果 a 是有限的，并且 x 是正数，并且 y 是扩展负数，则返回 None
    assert ask(Q.finite(a), Q.positive(x)
        & Q.extended_negative(y)) is None
    
    # 断言：如果 a 是有限的，并且 x 是正数，并且 y 是扩展负数，而 z 是扩展正数，则返回 None
    assert ask(Q.finite(a), Q.positive(x) & Q.extended_negative(y)
        & Q.extended_positive(z)) is None
    
    # 断言：如果 a 是有限的，并且 x 是正数，则返回 None
    assert ask(Q.finite(a), Q.positive(x)) is None
    
    # 断言：如果 a 是有限的，并且 x 是正数，而 z 是扩展正数，则返回 None
    assert ask(Q.finite(a), Q.positive(x)
        & Q.extended_positive(z)) is None
    
    # 断言：如果 a 是有限的，并且 x 是正数，并且 y 和 z 都是扩展正数，则返回 None
    assert ask(Q.finite(a), Q.positive(x) & Q.extended_positive(y)
        & Q.extended_positive(z)) is None
    
    # 断言：如果 a 是有限的，并且 x 是负无穷，y 是负无穷，z 也是负无穷，则返回 False
    assert ask(Q.finite(a), Q.negative_infinite(x)
        & Q.negative_infinite(y) & Q.negative_infinite(z)) is False
    
    # 断言：如果 a 是有限的，并且 x 是负无穷，y 是负无穷，而 z 不是有限的，则返回 None
    assert ask(Q.finite(a), Q.negative_infinite(x)
        & Q.negative_infinite(y) & ~Q.finite(z)) is None
    
    # 断言：如果 a 是有限的，并且 x 是负无穷，y 是负无穷，而 z 是正无穷，则返回 None
    assert ask(Q.finite(a), Q.negative_infinite(x)
        & Q.negative_infinite(y)& Q.positive_infinite(z)) is None
    # 断言语句，检查给定条件是否成立，如果条件不成立则抛出异常或者返回错误结果
    
    assert ask(Q.finite(a), Q.negative_infinite(x)
        & Q.negative_infinite(y) & Q.extended_negative(z)) is False
    # 检查条件：a 是有限数，x 是负无穷，y 是负无穷，z 是扩展负数，期望结果为 False
    
    assert ask(Q.finite(a), Q.negative_infinite(x)
        & Q.negative_infinite(y)) is None
    # 检查条件：a 是有限数，x 是负无穷，y 是负无穷，期望结果为 None
    
    assert ask(Q.finite(a), Q.negative_infinite(x)
        & Q.negative_infinite(y) & Q.extended_positive(z)) is None
    # 检查条件：a 是有限数，x 是负无穷，y 是负无穷，z 是扩展正数，期望结果为 None
    
    assert ask(Q.finite(a), Q.negative_infinite(x)
        & ~Q.finite(y) & ~Q.finite(z)) is None
    # 检查条件：a 是有限数，x 是负无穷，y 不是有限数，z 不是有限数，期望结果为 None
    
    assert ask(Q.finite(a), Q.negative_infinite(x)
        & ~Q.finite(y) & Q.positive_infinite(z)) is None
    # 检查条件：a 是有限数，x 是负无穷，y 不是有限数，z 是正无穷，期望结果为 None
    
    assert ask(Q.finite(a), Q.negative_infinite(x)
        & ~Q.finite(y) & Q.extended_negative(z)) is None
    # 检查条件：a 是有限数，x 是负无穷，y 不是有限数，z 是扩展负数，期望结果为 None
    
    assert ask(Q.finite(a), Q.negative_infinite(x)
        & ~Q.finite(y)) is None
    # 检查条件：a 是有限数，x 是负无穷，y 不是有限数，期望结果为 None
    
    assert ask(Q.finite(a), Q.negative_infinite(x)
        & ~Q.finite(y) & Q.extended_positive(z)) is None
    # 检查条件：a 是有限数，x 是负无穷，y 不是有限数，z 是扩展正数，期望结果为 None
    
    assert ask(Q.finite(a), Q.negative_infinite(x)
        & Q.positive_infinite(y) & Q.positive_infinite(z)) is None
    # 检查条件：a 是有限数，x 是负无穷，y 是正无穷，z 是正无穷，期望结果为 None
    
    assert ask(Q.finite(a), Q.negative_infinite(x)
        & Q.positive_infinite(y) & Q.extended_negative(z)) is None
    # 检查条件：a 是有限数，x 是负无穷，y 是正无穷，z 是扩展负数，期望结果为 None
    
    assert ask(Q.finite(a), Q.negative_infinite(x)
        & Q.positive_infinite(y)) is None
    # 检查条件：a 是有限数，x 是负无穷，y 是正无穷，期望结果为 None
    
    assert ask(Q.finite(a), Q.negative_infinite(x)
        & Q.positive_infinite(y) & Q.extended_positive(z)) is None
    # 检查条件：a 是有限数，x 是负无穷，y 是正无穷，z 是扩展正数，期望结果为 None
    
    assert ask(Q.finite(a), Q.negative_infinite(x)
        & Q.extended_negative(y) & Q.extended_negative(z)) is False
    # 检查条件：a 是有限数，x 是负无穷，y 是扩展负数，z 是扩展负数，期望结果为 False
    
    assert ask(Q.finite(a), Q.negative_infinite(x)
        & Q.extended_negative(y)) is None
    # 检查条件：a 是有限数，x 是负无穷，y 是扩展负数，期望结果为 None
    
    assert ask(Q.finite(a), Q.negative_infinite(x)
        & Q.extended_negative(y) & Q.extended_positive(z)) is None
    # 检查条件：a 是有限数，x 是负无穷，y 是扩展负数，z 是扩展正数，期望结果为 None
    
    assert ask(Q.finite(a), Q.negative_infinite(x)) is None
    # 检查条件：a 是有限数，x 是负无穷，期望结果为 None
    
    assert ask(Q.finite(a), Q.negative_infinite(x)
        & Q.extended_positive(z)) is None
    # 检查条件：a 是有限数，x 是负无穷，z 是扩展正数，期望结果为 None
    
    assert ask(Q.finite(a), Q.negative_infinite(x)
        & Q.extended_positive(y) & Q.extended_positive(z)) is None
    # 检查条件：a 是有限数，x 是负无穷，y 是扩展正数，z 是扩展正数，期望结果为 None
    
    assert ask(Q.finite(a), ~Q.finite(x) & ~Q.finite(y)
        & ~Q.finite(z)) is None
    # 检查条件：a 是有限数，x 不是有限数，y 不是有限数，z 不是有限数，期望结果为 None
    
    assert ask(Q.finite(a), ~Q.finite(x) & Q.positive_infinite(z)
        & ~Q.finite(z)) is None
    # 检查条件：a 是有限数，x 不是有限数，z 是正无穷且不是有限数，期望结果为 None
    
    assert ask(Q.finite(a), ~Q.finite(x) & ~Q.finite(y)
        & Q.extended_negative(z)) is None
    # 检查条件：a 是有限数，x 不是有限数，y 不是有限数，z 是扩展负数，期望结果为 None
    
    assert ask(Q.finite(a), ~Q.finite(x) & ~Q.finite(y)) is None
    # 检查条件：a 是有限数，x 不是有限数，y 不是有限数，期望结果为 None
    
    assert ask(Q.finite(a), ~Q.finite(x) & ~Q.finite(y)
        & Q.extended_positive(z)) is None
    # 检查条件：a 是有限数，x 不是有限数，y 不是有限数，z 是扩展正数，期望结果为 None
    
    assert ask(Q.finite(a), ~Q.finite(x) & Q.positive_infinite(y)
        & Q.positive_infinite(z)) is None
    # 检查条件：a 是有限数，x 不是有限数，y 是正无穷，z 是正无穷，期望结果为 None
    
    assert ask(Q.finite(a), ~Q.finite(x) & Q.positive_infinite(y)
        & Q.extended_negative(z)) is None
    # 检查条件：a 是有限数，x 不是有限数，y 是正无穷，z 是扩展负数，期望结果为 None
    
    assert ask(Q.finite(a), ~Q.finite(x)
        & Q.positive_infinite(y)) is None
    # 检查条件：a 是有限数，x 不是有限数，y 是正无穷，期望结果为 None
    
    assert ask(Q.finite(a), ~Q.finite(x) & Q.positive_infinite(y)
        & Q.extended_positive(z)) is None
    # 检查条件：a 是有限数，x 不是有限数，y 是正无穷，z 是扩展正数，期望结果为 None
    
    assert ask(Q.finite(a), ~Q.finite(x) & Q.extended_negative(y)
        & Q.extended_negative(z)) is None
    # 检查条件：a 是有限数，x 不是有限数，y 是扩展负数，z 是扩展负数，期望结果为 None
    
    assert ask(Q.finite(a), ~Q.finite(x)
        & Q.extended_negative(y)) is None
    # 检查条件：a 是有限数，x 不
    # 断言检查：如果 a 是有限的，同时 x 不是有限的且 y 是扩展负数，z 是扩展正数，则结果应为 None
    assert ask(Q.finite(a), ~Q.finite(x) & Q.extended_negative(y) & Q.extended_positive(z)) is None
    
    # 断言检查：如果 a 是有限的，同时 x 不是有限的，则结果应为 None
    assert ask(Q.finite(a), ~Q.finite(x)) is None
    
    # 断言检查：如果 a 是有限的，同时 x 不是有限的且 z 是扩展正数，则结果应为 None
    assert ask(Q.finite(a), ~Q.finite(x) & Q.extended_positive(z)) is None
    
    # 断言检查：如果 a 是有限的，同时 x 不是有限的且 y、z 都是扩展正数，则结果应为 None
    assert ask(Q.finite(a), ~Q.finite(x) & Q.extended_positive(y) & Q.extended_positive(z)) is None
    
    # 断言检查：如果 a 是有限的，同时 x 是正无穷大，y 是正无穷大，z 是正无穷大，则结果应为 False
    assert ask(Q.finite(a), Q.positive_infinite(x) & Q.positive_infinite(y) & Q.positive_infinite(z)) is False
    
    # 断言检查：如果 a 是有限的，同时 x 是正无穷大，y 是正无穷大，z 是扩展负数，则结果应为 None
    assert ask(Q.finite(a), Q.positive_infinite(x) & Q.positive_infinite(y) & Q.extended_negative(z)) is None
    
    # 断言检查：如果 a 是有限的，同时 x 是正无穷大，y 是正无穷大，则结果应为 None
    assert ask(Q.finite(a), Q.positive_infinite(x) & Q.positive_infinite(y)) is None
    
    # 断言检查：如果 a 是有限的，同时 x 是正无穷大，y 是正无穷大，z 是扩展正数，则结果应为 False
    assert ask(Q.finite(a), Q.positive_infinite(x) & Q.positive_infinite(y) & Q.extended_positive(z)) is False
    
    # 断言检查：如果 a 是有限的，同时 x 是正无穷大，y 是扩展负数，z 是扩展负数，则结果应为 None
    assert ask(Q.finite(a), Q.positive_infinite(x) & Q.extended_negative(y) & Q.extended_negative(z)) is None
    
    # 断言检查：如果 a 是有限的，同时 x 是正无穷大，y 是扩展负数，则结果应为 None
    assert ask(Q.finite(a), Q.positive_infinite(x) & Q.extended_negative(y)) is None
    
    # 断言检查：如果 a 是有限的，同时 x 是正无穷大，y 是扩展负数，z 是扩展正数，则结果应为 None
    assert ask(Q.finite(a), Q.positive_infinite(x) & Q.extended_negative(y) & Q.extended_positive(z)) is None
    
    # 断言检查：如果 a 是有限的，同时 x 是正无穷大，则结果应为 None
    assert ask(Q.finite(a), Q.positive_infinite(x)) is None
    
    # 断言检查：如果 a 是有限的，同时 x 是正无穷大，z 是扩展正数，则结果应为 None
    assert ask(Q.finite(a), Q.positive_infinite(x) & Q.extended_positive(z)) is None
    
    # 断言检查：如果 a 是有限的，同时 x 是正无穷大，y 是扩展正数，z 是扩展正数，则结果应为 False
    assert ask(Q.finite(a), Q.positive_infinite(x) & Q.extended_positive(y) & Q.extended_positive(z)) is False
    
    # 断言检查：如果 a 是有限的，同时 x 是扩展负数，y 是扩展负数，z 是扩展负数，则结果应为 None
    assert ask(Q.finite(a), Q.extended_negative(x) & Q.extended_negative(y) & Q.extended_negative(z)) is None
    
    # 断言检查：如果 a 是有限的，同时 x 是扩展负数，y 是扩展负数，则结果应为 None
    assert ask(Q.finite(a), Q.extended_negative(x) & Q.extended_negative(y)) is None
    
    # 断言检查：如果 a 是有限的，同时 x 是扩展负数，y 是扩展负数，z 是扩展正数，则结果应为 None
    assert ask(Q.finite(a), Q.extended_negative(x) & Q.extended_negative(y) & Q.extended_positive(z)) is None
    
    # 断言检查：如果 a 是有限的，同时 x 是扩展负数，则结果应为 None
    assert ask(Q.finite(a), Q.extended_negative(x)) is None
    
    # 断言检查：如果 a 是有限的，同时 x 是扩展负数，z 是扩展正数，则结果应为 None
    assert ask(Q.finite(a), Q.extended_negative(x) & Q.extended_positive(z)) is None
    
    # 断言检查：如果 a 是有限的，同时 x 是扩展负数，y 是扩展正数，z 是扩展正数，则结果应为 None
    assert ask(Q.finite(a), Q.extended_negative(x) & Q.extended_positive(y) & Q.extended_positive(z)) is None
    
    # 断言检查：如果 a 是有限的，则结果应为 None
    assert ask(Q.finite(a)) is None
    
    # 断言检查：如果 a 是有限的，同时 z 是扩展正数，则结果应为 None
    assert ask(Q.finite(a), Q.extended_positive(z)) is None
    
    # 断言检查：如果 a 是有限的，同时 y 和 z 都是扩展正数，则结果应为 None
    assert ask(Q.finite(a), Q.extended_positive(y) & Q.extended_positive(z)) is None
    
    # 断言检查：如果 a 是有限的，同时 x、y、z 都是扩展正数，则结果应为 None
    assert ask(Q.finite(a), Q.extended_positive(x) & Q.extended_positive(y) & Q.extended_positive(z)) is None
    
    # 断言检查：如果 2*x 是有限的，则结果应为 None
    assert ask(Q.finite(2*x)) is None
    
    # 断言检查：如果 2*x 是有限的，且 x 是有限的，则结果应为 True
    assert ask(Q.finite(2*x), Q.finite(x)) is True
    
    # 定义符号变量 x, y, z，并将 a 设为 x*y
    x, y, z = symbols('x,y,z')
    a = x*y
    
    # 将 a 的参数分别设为 x, y
    x, y = a.args
    
    # 断言检查：如果 a 是有限的，且 x, y 都是有限的，则结果应为 True
    assert ask(Q.finite(a), Q.finite(x) & Q.finite(y)) is True
    
    # 断言检查：如果 a 是有限的，且 x 是有限的而 y 不是有限的，则结果应为 False
    assert ask(Q.finite(a), Q.finite(x) & ~Q.finite(y)) is False
    
    # 断言检查：如果 a 是有限的，且 x 是有限的，则结果应为 None
    assert ask(Q.finite(a), Q.finite(x)) is None
    
    # 断言检查：如果 a 是有限的，且 x 不是有限的而 y 是有限的，则结果应为 False
    assert ask(Q.finite(a), ~Q.finite(x) & Q.finite(y)) is False
    
    # 断言检查：如果 a 是有限的，且 x, y 都不是有限的，则结果应为 False
    assert ask(Q.finite(a), ~Q.finite(x) & ~Q.finite(y)) is False
    
    # 断言检查：如果 a 是有限的，且 x 不是有限的，则结果应为 None
    assert ask(Q.finite(a), ~Q.finite(x)) is None
    
    # 断言检查：如果 a 是有限的，
    # 确保 a 是有限的，同时 x, y, z 也是有限的，返回 True
    assert ask(Q.finite(a), Q.finite(x) & Q.finite(y)
        & Q.finite(z)) is True

    # 确保 a 是有限的，x, y 是有限的，但 z 是无限的，返回 False
    assert ask(Q.finite(a), Q.finite(x) & Q.finite(y)
        & ~Q.finite(z)) is False

    # 确保 a 是有限的，同时 x, y 都是有限的，但不关心 z，返回 None
    assert ask(Q.finite(a), Q.finite(x) & Q.finite(y)) is None

    # 确保 a 是有限的，x 是有限的，y 是无限的，z 是有限的，返回 False
    assert ask(Q.finite(a), Q.finite(x) & ~Q.finite(y)
        & Q.finite(z)) is False

    # 确保 a 是有限的，x, z 是有限的，y 是无限的，返回 False
    assert ask(Q.finite(a), Q.finite(x) & ~Q.finite(y)
        & ~Q.finite(z)) is False

    # 确保 a 是有限的，x 是有限的，y 是无限的，不关心 z，返回 None
    assert ask(Q.finite(a), Q.finite(x) & ~Q.finite(y)) is None

    # 确保 a 是有限的，同时 x, z 是有限的，不关心 y，返回 None
    assert ask(Q.finite(a), Q.finite(x) & Q.finite(z)) is None

    # 确保 a 是有限的，x 是有限的，z 是无限的，返回 None
    assert ask(Q.finite(a), Q.finite(x) & ~Q.finite(z)) is None

    # 确保 a 是有限的，x 是有限的，不关心 y, z，返回 None
    assert ask(Q.finite(a), Q.finite(x)) is None

    # 确保 a 是有限的，x 是无限的，y, z 都是有限的，返回 False
    assert ask(Q.finite(a), ~Q.finite(x) & Q.finite(y)
        & Q.finite(z)) is False

    # 确保 a 是有限的，x 是无限的，y 是有限的，z 是无限的，返回 False
    assert ask(Q.finite(a), ~Q.finite(x) & Q.finite(y)
        & ~Q.finite(z)) is False

    # 确保 a 是有限的，x 是无限的，y 是有限的，不关心 z，返回 None
    assert ask(Q.finite(a), ~Q.finite(x) & Q.finite(y)) is None

    # 确保 a 是有限的，x, y 都是无限的，z 是有限的，返回 False
    assert ask(Q.finite(a), ~Q.finite(x) & ~Q.finite(y)
        & Q.finite(z)) is False

    # 确保 a 是有限的，x, y, z 都是无限的，返回 False
    assert ask(Q.finite(a), ~Q.finite(x) & ~Q.finite(y)
        & ~Q.finite(z)) is False

    # 确保 a 是有限的，x, y 都是无限的，不关心 z，返回 None
    assert ask(Q.finite(a), ~Q.finite(x) & ~Q.finite(y)) is None

    # 确保 a 是有限的，x 是无限的，z 是有限的，返回 None
    assert ask(Q.finite(a), ~Q.finite(x) & Q.finite(z)) is None

    # 确保 a 是有限的，x 是无限的，z 是无限的，返回 None
    assert ask(Q.finite(a), ~Q.finite(x) & ~Q.finite(z)) is None

    # 确保 a 是有限的，x 是无限的，同时 x, y, z 都是扩展非零的，返回 None
    assert ask(Q.finite(a), ~Q.finite(z) & Q.extended_nonzero(x)
        & Q.extended_nonzero(y) & Q.extended_nonzero(z)) is None

    # 确保 a 是有限的，2^x 是有限的，x 是无限的，y 是扩展非零的，z 是无限的，返回 False
    assert ask(Q.finite(a), Q.extended_nonzero(x) & ~Q.finite(y)
        & Q.extended_nonzero(y) & ~Q.finite(z)
        & Q.extended_nonzero(z)) is False

    # 为符号 x, y, z 创建符号变量
    x, y, z = symbols('x,y,z')

    # 确保 x^2 是有限的，返回 None
    assert ask(Q.finite(x**2)) is None

    # 确保 2^x 是有限的，返回 None
    assert ask(Q.finite(2**x)) is None

    # 确保 2^x 是有限的，并且 x 是有限的，返回 True
    assert ask(Q.finite(2**x), Q.finite(x)) is True

    # 确保 x^x 是有限的，返回 None
    assert ask(Q.finite(x**x)) is None

    # 确保 (1/2)^x 是有限的，返回 None
    assert ask(Q.finite(S.Half ** x)) is None

    # 确保 (1/2)^x 是有限的，并且 x 是扩展正数的，返回 True
    assert ask(Q.finite(S.Half ** x), Q.extended_positive(x)) is True

    # 确保 (1/2)^x 是有限的，并且 x 是扩展负数的，返回 None
    assert ask(Q.finite(S.Half ** x), Q.extended_negative(x)) is None

    # 确保 2^x 是有限的，并且 x 是扩展负数的，返回 True
    assert ask(Q.finite(2**x), Q.extended_negative(x)) is True

    # 确保 sqrt(x) 是有限的，返回 None
    assert ask(Q.finite(sqrt(x))) is None

    # 确保 2^x 是有限的，但 x 是无限的，返回 False
    assert ask(Q.finite(2**x), ~Q.finite(x)) is False

    # 确保 x^2 是有限的，但 x 是无限的，返回 False
    assert ask(Q.finite(x**2), ~Q.finite(x)) is False

    # 确保 sign(x) 是有限的，返回 True
    assert ask(Q.finite(sign(x))) is True

    # 确保 sign(x) 是有限的，并且 x 是无限的，返回 True
    assert ask(Q.finite(sign(x)), ~Q.finite(x)) is True

    # 确保 log(x) 是有限的，返回 None
    assert ask(Q.finite(log(x))) is None

    # 确保 log(x) 是有限的，并且 x 是有限的，返回 None
    assert ask(Q.finite(log(x)), Q.finite(x)) is None

    # 确保 log(x) 是有限的，并且 x 不是零，返回 True
    assert ask(Q.finite(log(x)), ~Q.zero(x)) is True
    # 断言：对于对数函数log(x)，在有限域条件下，x不可能是无限的，因此返回False
    assert ask(Q.finite(log(x)), Q.infinite(x)) is False
    # 断言：对于对数函数log(x)，在有限域条件下，x不可能是零，因此返回False
    assert ask(Q.finite(log(x)), Q.zero(x)) is False
    # 断言：对于指数函数exp(x)，在有限域条件下，x不一定是有限的，因此返回None
    assert ask(Q.finite(exp(x))) is None
    # 断言：对于指数函数exp(x)，在有限域条件下，x是有限的，则返回True
    assert ask(Q.finite(exp(x)), Q.finite(x)) is True
    # 断言：对于指数函数exp(2)，在有限域条件下，2是有限的，因此返回True
    assert ask(Q.finite(exp(2))) is True

    # 三角函数部分
    # 断言：对于正弦函数sin(x)，在有限域条件下，x是有限的，则返回True
    assert ask(Q.finite(sin(x))) is True
    # 断言：对于正弦函数sin(x)，在非有限域条件下，即x可能是无限的，则返回True
    assert ask(Q.finite(sin(x)), ~Q.finite(x)) is True
    # 断言：对于余弦函数cos(x)，在有限域条件下，x是有限的，则返回True
    assert ask(Q.finite(cos(x))) is True
    # 断言：对于余弦函数cos(x)，在非有限域条件下，即x可能是无限的，则返回True
    assert ask(Q.finite(cos(x)), ~Q.finite(x)) is True
    # 断言：对于2*sin(x)，在有限域条件下，sin(x)是有限的，则返回True
    assert ask(Q.finite(2*sin(x))) is True
    # 断言：对于sin(x)**2，在有限域条件下，sin(x)的平方是有限的，则返回True
    assert ask(Q.finite(sin(x)**2)) is True
    # 断言：对于cos(x)**2，在有限域条件下，cos(x)的平方是有限的，则返回True
    assert ask(Q.finite(cos(x)**2)) is True
    # 断言：对于cos(x) + sin(x)，在有限域条件下，cos(x)和sin(x)的和是有限的，则返回True
    assert ask(Q.finite(cos(x) + sin(x))) is True
@XFAIL
# 标记该测试函数为预期失败，即测试预期会失败
def test_bounded_xfail():
    """We need to support relations in ask for this to work"""
    # 断言sin(x)**x是有限的
    assert ask(Q.finite(sin(x)**x)) is True
    # 断言cos(x)**x是有限的
    assert ask(Q.finite(cos(x)**x)) is True


def test_commutative():
    """By default objects are Q.commutative that is why it returns True
    for both key=True and key=False"""
    # 断言x是可交换的
    assert ask(Q.commutative(x)) is True
    # 断言x是可交换的，而~Q.commutative(x)是假的
    assert ask(Q.commutative(x), ~Q.commutative(x)) is False
    # 断言x是可交换的，且x是复数的
    assert ask(Q.commutative(x), Q.complex(x)) is True
    # 断言x是可交换的，且x是虚数的
    assert ask(Q.commutative(x), Q.imaginary(x)) is True
    # 断言x是可交换的，且x是实数的
    assert ask(Q.commutative(x), Q.real(x)) is True
    # 断言x是可交换的，且x是正数的
    assert ask(Q.commutative(x), Q.positive(x)) is True
    # 断言x是可交换的，而~Q.commutative(y)是真的
    assert ask(Q.commutative(x), ~Q.commutative(y)) is True

    # 断言2*x是可交换的
    assert ask(Q.commutative(2*x)) is True
    # 断言2*x是可交换的，而~Q.commutative(x)是假的
    assert ask(Q.commutative(2*x), ~Q.commutative(x)) is False

    # 断言x+1是可交换的
    assert ask(Q.commutative(x + 1)) is True
    # 断言x+1是可交换的，而~Q.commutative(x)是假的
    assert ask(Q.commutative(x + 1), ~Q.commutative(x)) is False

    # 断言x**2是可交换的
    assert ask(Q.commutative(x**2)) is True
    # 断言x**2是可交换的，而~Q.commutative(x)是假的
    assert ask(Q.commutative(x**2), ~Q.commutative(x)) is False

    # 断言log(x)是可交换的
    assert ask(Q.commutative(log(x))) is True


@_both_exp_pow
# 标记这个测试函数将同时测试指数和幂的性质
def test_complex():
    # 断言x是复数的结果是不确定的
    assert ask(Q.complex(x)) is None
    # 断言x是复数的，且x是复数的
    assert ask(Q.complex(x), Q.complex(x)) is True
    # 断言x是复数的，且y是复数的结果是不确定的
    assert ask(Q.complex(x), Q.complex(y)) is None
    # 断言x是复数的，而~Q.complex(x)是假的
    assert ask(Q.complex(x), ~Q.complex(x)) is False
    # 断言x是复数的，且x是实数的
    assert ask(Q.complex(x), Q.real(x)) is True
    # 断言x是复数的，而~Q.real(x)是不确定的
    assert ask(Q.complex(x), ~Q.real(x)) is None
    # 断言x是复数的，且x是有理数的
    assert ask(Q.complex(x), Q.rational(x)) is True
    # 断言x是复数的，且x是无理数的
    assert ask(Q.complex(x), Q.irrational(x)) is True
    # 断言x是复数的，且x是正数的
    assert ask(Q.complex(x), Q.positive(x)) is True
    # 断言x是复数的，且x是虚数的
    assert ask(Q.complex(x), Q.imaginary(x)) is True
    # 断言x是复数的，且x是代数数的
    assert ask(Q.complex(x), Q.algebraic(x)) is True

    # 断言x+1是复数的
    assert ask(Q.complex(x + 1), Q.complex(x)) is True
    # 断言x+1是实数的
    assert ask(Q.complex(x + 1), Q.real(x)) is True
    # 断言x+1是有理数的
    assert ask(Q.complex(x + 1), Q.rational(x)) is True
    # 断言x+1是无理数的
    assert ask(Q.complex(x + 1), Q.irrational(x)) is True
    # 断言x+1是虚数的
    assert ask(Q.complex(x + 1), Q.imaginary(x)) is True
    # 断言x+1是整数的
    assert ask(Q.complex(x + 1), Q.integer(x)) is True
    # 断言x+1是偶数的
    assert ask(Q.complex(x + 1), Q.even(x)) is True
    # 断言x+1是奇数的
    assert ask(Q.complex(x + 1), Q.odd(x)) is True
    # 断言x+y是复数的，且x和y都是复数的
    assert ask(Q.complex(x + y), Q.complex(x) & Q.complex(y)) is True
    # 断言x+y是实数的，且x是实数且y是虚数的
    assert ask(Q.complex(x + y), Q.real(x) & Q.imaginary(y)) is True

    # 断言2*x+1是复数的
    assert ask(Q.complex(2*x + 1), Q.complex(x)) is True
    # 断言2*x+1是实数的
    assert ask(Q.complex(2*x + 1), Q.real(x)) is True
    # 断言2*x+1是正数的
    assert ask(Q.complex(2*x + 1), Q.positive(x)) is True
    # 断言2*x+1是有理数的
    assert ask(Q.complex(2*x + 1), Q.rational(x)) is True
    # 断言2*x+1是无理数的
    assert ask(Q.complex(2*x + 1), Q.irrational(x)) is True
    # 断言2*x+1是虚数的
    assert ask(Q.complex(2*x + 1), Q.imaginary(x)) is True
    # 断言2*x+1是整数的
    assert ask(Q.complex(2*x + 1), Q.integer(x)) is True
    # 断言2*x+1是偶数的
    assert ask(Q.complex(2*x + 1), Q.even(x)) is True
    # 断言2*x+1是奇数的
    assert ask(Q.complex(2*x + 1), Q.odd(x)) is True

    # 断言x**2是复数的
    assert ask(Q.complex(x**2), Q.complex(x)) is True
    # 断言x**2是实数的
    assert ask(Q.complex(x**2), Q.real(x)) is True
    # 断言x**2是正数的
    assert ask(Q.complex(x**2), Q.positive(x)) is True
    # 检查 x**2 是否属于复数集合
    assert ask(Q.complex(x**2), Q.rational(x)) is True
    # 检查 x**2 是否属于有理数集合
    assert ask(Q.complex(x**2), Q.irrational(x)) is True
    # 检查 x**2 是否属于虚数集合
    assert ask(Q.complex(x**2), Q.imaginary(x)) is True
    # 检查 x**2 是否属于整数集合
    assert ask(Q.complex(x**2), Q.integer(x)) is True
    # 检查 x**2 是否属于偶数集合
    assert ask(Q.complex(x**2), Q.even(x)) is True
    # 检查 x**2 是否属于奇数集合
    assert ask(Q.complex(x**2), Q.odd(x)) is True

    # 检查 2**x 是否属于复数集合
    assert ask(Q.complex(2**x), Q.complex(x)) is True
    # 检查 2**x 是否属于实数集合
    assert ask(Q.complex(2**x), Q.real(x)) is True
    # 检查 2**x 是否属于正数集合
    assert ask(Q.complex(2**x), Q.positive(x)) is True
    # 检查 2**x 是否属于有理数集合
    assert ask(Q.complex(2**x), Q.rational(x)) is True
    # 检查 2**x 是否属于无理数集合
    assert ask(Q.complex(2**x), Q.irrational(x)) is True
    # 检查 2**x 是否属于虚数集合
    assert ask(Q.complex(2**x), Q.imaginary(x)) is True
    # 检查 2**x 是否属于整数集合
    assert ask(Q.complex(2**x), Q.integer(x)) is True
    # 检查 2**x 是否属于偶数集合
    assert ask(Q.complex(2**x), Q.even(x)) is True
    # 检查 2**x 是否属于奇数集合
    assert ask(Q.complex(2**x), Q.odd(x)) is True
    # 检查 x**y 是否同时属于复数集合中的 x 和 y
    assert ask(Q.complex(x**y), Q.complex(x) & Q.complex(y)) is True

    # 正弦函数表达式是否属于复数集合
    assert ask(Q.complex(sin(x))) is True
    # 复合正弦表达式是否属于复数集合
    assert ask(Q.complex(sin(2*x + 1))) is True
    # 余弦函数表达式是否属于复数集合
    assert ask(Q.complex(cos(x))) is True
    # 复合余弦表达式是否属于复数集合
    assert ask(Q.complex(cos(2*x + 1))) is True

    # 指数函数表达式是否属于复数集合
    assert ask(Q.complex(exp(x))) is True
    assert ask(Q.complex(exp(x))) is True  # 这行似乎是重复的，可能是笔误

    # 绝对值函数是否属于复数集合
    assert ask(Q.complex(Abs(x))) is True
    # 实部函数是否属于复数集合
    assert ask(Q.complex(re(x))) is True
    # 虚部函数是否属于复数集合
    assert ask(Q.complex(im(x))) is True
# 定义一个测试函数，用于测试偶数相关的查询
def test_even_query():
    # 断言查询是否偶数，应该返回 None
    assert ask(Q.even(x)) is None
    # 断言查询是否偶数且为整数，应该返回 None
    assert ask(Q.even(x), Q.integer(x)) is None
    # 断言查询是否偶数且不是整数，应该返回 False
    assert ask(Q.even(x), ~Q.integer(x)) is False
    # 断言查询是否偶数且为有理数，应该返回 None
    assert ask(Q.even(x), Q.rational(x)) is None
    # 断言查询是否偶数且为正数，应该返回 None
    assert ask(Q.even(x), Q.positive(x)) is None

    # 断言查询是否偶数，应该返回 None
    assert ask(Q.even(2*x)) is None
    # 断言查询是否偶数且为整数，应该返回 True
    assert ask(Q.even(2*x), Q.integer(x)) is True
    # 断言查询是否偶数且为偶数，应该返回 True
    assert ask(Q.even(2*x), Q.even(x)) is True
    # 断言查询是否偶数且为无理数，应该返回 False
    assert ask(Q.even(2*x), Q.irrational(x)) is False
    # 断言查询是否偶数且为奇数，应该返回 True
    assert ask(Q.even(2*x), Q.odd(x)) is True
    # 断言查询是否偶数且不是整数，应该返回 None
    assert ask(Q.even(2*x), ~Q.integer(x)) is None
    # 断言查询是否偶数且为整数，应该返回 None
    assert ask(Q.even(3*x), Q.integer(x)) is None
    # 断言查询是否偶数且为偶数，应该返回 True
    assert ask(Q.even(3*x), Q.even(x)) is True
    # 断言查询是否偶数且为奇数，应该返回 False
    assert ask(Q.even(3*x), Q.odd(x)) is False

    # 断言查询是否偶数且为奇数，应该返回 True
    assert ask(Q.even(x + 1), Q.odd(x)) is True
    # 断言查询是否偶数且为偶数，应该返回 False
    assert ask(Q.even(x + 1), Q.even(x)) is False
    # 断言查询是否偶数且为奇数，应该返回 False
    assert ask(Q.even(x + 2), Q.odd(x)) is False
    # 断言查询是否偶数且为偶数，应该返回 True
    assert ask(Q.even(x + 2), Q.even(x)) is True
    # 断言查询是否偶数且为奇数，应该返回 True
    assert ask(Q.even(7 - x), Q.odd(x)) is True
    # 断言查询是否偶数且为奇数，应该返回 True
    assert ask(Q.even(7 + x), Q.odd(x)) is True
    # 断言查询是否偶数且为奇数且为奇数，应该返回 True
    assert ask(Q.even(x + y), Q.odd(x) & Q.odd(y)) is True
    # 断言查询是否偶数且为奇数且为偶数，应该返回 False
    assert ask(Q.even(x + y), Q.odd(x) & Q.even(y)) is False
    # 断言查询是否偶数且为偶数且为偶数，应该返回 True
    assert ask(Q.even(x + y), Q.even(x) & Q.even(y)) is True

    # 断言查询是否偶数且为偶数乘以 2 再加 1，应该返回 False
    assert ask(Q.even(2*x + 1), Q.integer(x)) is False
    # 断言查询是否偶数且为偶数乘积，且为有理数乘积，应该返回 None
    assert ask(Q.even(2*x*y), Q.rational(x) & Q.rational(x)) is None
    # 断言查询是否偶数且为偶数乘积，且为无理数乘积，应该返回 None
    assert ask(Q.even(2*x*y), Q.irrational(x) & Q.irrational(x)) is None

    # 断言查询是否偶数且为奇数相加乘积，且 x 和 y 为奇数，应该返回 True
    assert ask(Q.even(x + y + z), Q.odd(x) & Q.odd(y) & Q.even(z)) is True
    # 断言查询是否偶数且为奇数相加乘积再加 t，且 x 和 y 为奇数，t 为整数，应该返回 None
    assert ask(Q.even(x + y + z + t),
               Q.odd(x) & Q.odd(y) & Q.even(z) & Q.integer(t)) is None

    # 断言查询是否偶数且为 x 绝对值偶数，应该返回 True
    assert ask(Q.even(Abs(x)), Q.even(x)) is True
    # 断言查询是否偶数且为 x 绝对值不为偶数，应该返回 None
    assert ask(Q.even(Abs(x)), ~Q.even(x)) is None
    # 断言查询是否偶数且为 x 实部偶数，应该返回 True
    assert ask(Q.even(re(x)), Q.even(x)) is True
    # 断言查询是否偶数且为 x 实部不为偶数，应该返回 None
    assert ask(Q.even(re(x)), ~Q.even(x)) is None
    # 断言查询是否偶数且为 x 虚部偶数，应该返回 True
    assert ask(Q.even(im(x)), Q.even(x)) is True
    # 断言查询是否偶数且为 x 实部为实数，应该返回 True
    assert ask(Q.even(im(x)), Q.real(x)) is True

    # 断言查询是否偶数且为 (-1)^n，n 为整数，应该返回 False
    assert ask(Q.even((-1)**n), Q.integer(n)) is False

    # 断言查询是否偶数且为 k 的平方，k 为偶数，应该返回 True
    assert ask(Q.even(k**2), Q.even(k)) is True
    # 断言查询是否偶数且为 n 的平方，n 为奇数，应该返回 False
    assert ask(Q.even(n**2), Q.odd(n)) is False
    # 断言查询是否偶数且为 2 的 k 次方，k 为偶数，应该返回 None
    assert ask(Q.even(2**k), Q.even(k)) is None
    # 断言查询是否偶数且为 x 的平方，应该返回 None
    assert ask(Q.even(x**2)) is None

    # 断言查询是否偶数且为 k 的 m 次方，k 为偶数，m 为整数且非负，应该返回 None
    assert ask(Q.even(k**m), Q.even(k) & Q.integer(m) & ~Q.negative(m)) is None
    # 断言查询是否偶数且为 n 的 m 次方，n 为奇数，m 为整数且非负，应该返回 False
    assert ask(Q.even(n**m), Q.odd(n) & Q.integer(m) & ~Q.negative(m)) is False

    # 断言查询是否偶数且为 k 的 p 次方，k 为偶数，p 为整数且为正数，应该返回 True
    assert ask(Q.even(k**p), Q.even(k) & Q.integer(p) & Q.positive(p)) is True
    # 断言查询是否偶数且为 n 的 p 次方，n 为奇数，p 为整数且为正数，应该返回 False
    # 使用断言来验证一个条件，确保满足给定的约束条件
    assert ask(Q.even(x*(x + y)), Q.integer(x) & Q.even(y)) is None
@XFAIL
# 标记为预期失败的测试函数
def test_evenness_in_ternary_integer_product_with_odd():
    # 测试奇偶性推断是否独立于项的顺序。
    # 在测试时，项的顺序取决于SymPy的符号顺序，因此我们通过修改符号名称来尝试强制使用不同的顺序。
    assert ask(Q.even(x*y*(y + z)), Q.integer(x) & Q.integer(y) & Q.odd(z)) is True
    assert ask(Q.even(y*x*(x + z)), Q.integer(x) & Q.integer(y) & Q.odd(z)) is True


# 测试三元整数乘积中的偶数性
def test_evenness_in_ternary_integer_product_with_even():
    assert ask(Q.even(x*y*(y + z)), Q.integer(x) & Q.integer(y) & Q.even(z)) is None


# 测试扩展实数
def test_extended_real():
    assert ask(Q.extended_real(x), Q.positive_infinite(x)) is True
    assert ask(Q.extended_real(x), Q.positive(x)) is True
    assert ask(Q.extended_real(x), Q.zero(x)) is True
    assert ask(Q.extended_real(x), Q.negative(x)) is True
    assert ask(Q.extended_real(x), Q.negative_infinite(x)) is True

    assert ask(Q.extended_real(-x), Q.positive(x)) is True
    assert ask(Q.extended_real(-x), Q.negative(x)) is True

    assert ask(Q.extended_real(x + S.Infinity), Q.real(x)) is True

    assert ask(Q.extended_real(x), Q.infinite(x)) is None


# 在指数和幂方面进行测试
@_both_exp_pow
def test_rational():
    assert ask(Q.rational(x), Q.integer(x)) is True
    assert ask(Q.rational(x), Q.irrational(x)) is False
    assert ask(Q.rational(x), Q.real(x)) is None
    assert ask(Q.rational(x), Q.positive(x)) is None
    assert ask(Q.rational(x), Q.negative(x)) is None
    assert ask(Q.rational(x), Q.nonzero(x)) is None
    assert ask(Q.rational(x), ~Q.algebraic(x)) is False

    assert ask(Q.rational(2*x), Q.rational(x)) is True
    assert ask(Q.rational(2*x), Q.integer(x)) is True
    assert ask(Q.rational(2*x), Q.even(x)) is True
    assert ask(Q.rational(2*x), Q.odd(x)) is True
    assert ask(Q.rational(2*x), Q.irrational(x)) is False

    assert ask(Q.rational(x/2), Q.rational(x)) is True
    assert ask(Q.rational(x/2), Q.integer(x)) is True
    assert ask(Q.rational(x/2), Q.even(x)) is True
    assert ask(Q.rational(x/2), Q.odd(x)) is True
    assert ask(Q.rational(x/2), Q.irrational(x)) is False

    assert ask(Q.rational(1/x), Q.rational(x)) is True
    assert ask(Q.rational(1/x), Q.integer(x)) is True
    assert ask(Q.rational(1/x), Q.even(x)) is True
    assert ask(Q.rational(1/x), Q.odd(x)) is True
    assert ask(Q.rational(1/x), Q.irrational(x)) is False

    assert ask(Q.rational(2/x), Q.rational(x)) is True
    assert ask(Q.rational(2/x), Q.integer(x)) is True
    assert ask(Q.rational(2/x), Q.even(x)) is True
    assert ask(Q.rational(2/x), Q.odd(x)) is True
    assert ask(Q.rational(2/x), Q.irrational(x)) is False

    assert ask(Q.rational(x), ~Q.algebraic(x)) is False

    # 使用多个符号进行测试
    assert ask(Q.rational(x*y), Q.irrational(x) & Q.irrational(y)) is None
    assert ask(Q.rational(y/x), Q.rational(x) & Q.rational(y)) is True
    assert ask(Q.rational(y/x), Q.integer(x) & Q.rational(y)) is True
    # 检查 y/x 是否为有理数，并且 x 是偶数时返回 True
    assert ask(Q.rational(y/x), Q.even(x) & Q.rational(y)) is True
    # 检查 y/x 是否为有理数，并且 x 是奇数时返回 True
    assert ask(Q.rational(y/x), Q.odd(x) & Q.rational(y)) is True
    # 检查 y/x 是否为有理数，并且 x 是无理数时返回 False
    assert ask(Q.rational(y/x), Q.irrational(x) & Q.rational(y)) is False
    
    # 对于指数函数 exp, 正弦函数 sin, 正切函数 tan, 反正弦函数 asin, 反正切函数 atan, 余弦函数 cos
    for f in [exp, sin, tan, asin, atan, cos]:
        # 检查 f(7) 是否为有理数，应返回 False
        assert ask(Q.rational(f(7))) is False
        # 检查 f(7, evaluate=False) 是否为有理数，应返回 False
        assert ask(Q.rational(f(7, evaluate=False))) is False
        # 检查 f(0, evaluate=False) 是否为有理数，应返回 True
        assert ask(Q.rational(f(0, evaluate=False))) is True
        # 检查 f(x) 是否为有理数，无法确定时返回 None
        assert ask(Q.rational(f(x)), Q.rational(x)) is None
        # 检查 f(x) 是否为有理数，并且 x 不为零时返回 False
        assert ask(Q.rational(f(x)), Q.rational(x) & Q.nonzero(x)) is False
    
    # 对于对数函数 log, 反余弦函数 acos
    for g in [log, acos]:
        # 检查 g(7) 是否为有理数，应返回 False
        assert ask(Q.rational(g(7))) is False
        # 检查 g(7, evaluate=False) 是否为有理数，应返回 False
        assert ask(Q.rational(g(7, evaluate=False))) is False
        # 检查 g(1, evaluate=False) 是否为有理数，应返回 True
        assert ask(Q.rational(g(1, evaluate=False))) is True
        # 检查 g(x) 是否为有理数，无法确定时返回 None
        assert ask(Q.rational(g(x)), Q.rational(x)) is None
        # 检查 g(x) 是否为有理数，并且 x-1 是有理数时返回 False
        assert ask(Q.rational(g(x)), Q.rational(x) & Q.nonzero(x - 1)) is False
    
    # 对于余切函数 cot, 反余切函数 acot
    for h in [cot, acot]:
        # 检查 h(7) 是否为有理数，应返回 False
        assert ask(Q.rational(h(7))) is False
        # 检查 h(7, evaluate=False) 是否为有理数，应返回 False
        assert ask(Q.rational(h(7, evaluate=False))) is False
        # 检查 h(x) 是否为有理数，应返回 False
        assert ask(Q.rational(h(x)), Q.rational(x)) is False
# 定义测试函数 test_hermitian，用于测试符号表达式是否符合 Hermite 性质
def test_hermitian():
    # 断言测试表达式 x 是否为 Hermite 的
    assert ask(Q.hermitian(x)) is None
    # 断言测试表达式 x 是否为 Hermite 和反-Hermite 的
    assert ask(Q.hermitian(x), Q.antihermitian(x)) is None
    # 断言测试表达式 x 是否为 Hermite 和虚数的
    assert ask(Q.hermitian(x), Q.imaginary(x)) is False
    # 断言测试表达式 x 是否为 Hermite 和素数的
    assert ask(Q.hermitian(x), Q.prime(x)) is True
    # 断言测试表达式 x 是否为 Hermite 和实数的
    assert ask(Q.hermitian(x), Q.real(x)) is True
    # 断言测试表达式 x 是否为 Hermite 和零的
    assert ask(Q.hermitian(x), Q.zero(x)) is True

    # 断言测试表达式 x + 1 是否为 Hermite 和反-Hermite 的
    assert ask(Q.hermitian(x + 1), Q.antihermitian(x)) is None
    # 断言测试表达式 x + 1 是否为 Hermite 和复数的
    assert ask(Q.hermitian(x + 1), Q.complex(x)) is None
    # 断言测试表达式 x + 1 是否为 Hermite 的
    assert ask(Q.hermitian(x + 1), Q.hermitian(x)) is True
    # 断言测试表达式 x + 1 是否为 Hermite 和虚数的
    assert ask(Q.hermitian(x + 1), Q.imaginary(x)) is False
    # 断言测试表达式 x + 1 是否为 Hermite 和实数的
    assert ask(Q.hermitian(x + 1), Q.real(x)) is True
    # 断言测试表达式 x + I 是否为 Hermite 和反-Hermite 的
    assert ask(Q.hermitian(x + I), Q.antihermitian(x)) is None
    # 断言测试表达式 x + I 是否为 Hermite 和复数的
    assert ask(Q.hermitian(x + I), Q.complex(x)) is None
    # 断言测试表达式 x + I 是否为 Hermite 的
    assert ask(Q.hermitian(x + I), Q.hermitian(x)) is False
    # 断言测试表达式 x + I 是否为 Hermite 和虚数的
    assert ask(Q.hermitian(x + I), Q.imaginary(x)) is None
    # 断言测试表达式 x + I 是否为 Hermite 和实数的
    assert ask(Q.hermitian(x + I), Q.real(x)) is False
    # 断言测试表达式 x + y 是否为 Hermite 和反-Hermite 的
    assert ask(Q.hermitian(x + y), Q.antihermitian(x) & Q.antihermitian(y)) is None
    # 断言测试表达式 x + y 是否为 Hermite 和反-Hermite & 复数的
    assert ask(Q.hermitian(x + y), Q.antihermitian(x) & Q.complex(y)) is None
    # 断言测试表达式 x + y 是否为 Hermite 和反-Hermite & Hermite 的
    assert ask(Q.hermitian(x + y), Q.antihermitian(x) & Q.hermitian(y)) is None
    # 断言测试表达式 x + y 是否为 Hermite 和反-Hermite & 虚数的
    assert ask(Q.hermitian(x + y), Q.antihermitian(x) & Q.imaginary(y)) is None
    # 断言测试表达式 x + y 是否为 Hermite 和反-Hermite & 实数的
    assert ask(Q.hermitian(x + y), Q.antihermitian(x) & Q.real(y)) is None
    # 断言测试表达式 x + y 是否为 Hermite 和 Hermite & 复数的
    assert ask(Q.hermitian(x + y), Q.hermitian(x) & Q.complex(y)) is None
    # 断言测试表达式 x + y 是否为 Hermite 和 Hermite 的
    assert ask(Q.hermitian(x + y), Q.hermitian(x) & Q.hermitian(y)) is True
    # 断言测试表达式 x + y 是否为 Hermite 和 Hermite & 虚数的
    assert ask(Q.hermitian(x + y), Q.hermitian(x) & Q.imaginary(y)) is False
    # 断言测试表达式 x + y 是否为 Hermite 和 Hermite & 实数的
    assert ask(Q.hermitian(x + y), Q.hermitian(x) & Q.real(y)) is True
    # 断言测试表达式 x + y 是否为 Hermite 和虚数 & 复数的
    assert ask(Q.hermitian(x + y), Q.imaginary(x) & Q.complex(y)) is None
    # 断言测试表达式 x + y 是否为 Hermite 和虚数 & 虚数的
    assert ask(Q.hermitian(x + y), Q.imaginary(x) & Q.imaginary(y)) is None
    # 断言测试表达式 x + y 是否为 Hermite 和虚数 & 实数的
    assert ask(Q.hermitian(x + y), Q.imaginary(x) & Q.real(y)) is False
    # 断言测试表达式 x + y 是否为 Hermite 和实数 & 复数的
    assert ask(Q.hermitian(x + y), Q.real(x) & Q.complex(y)) is None
    # 断言测试表达式 x + y 是否为 Hermite 和实数 & 实数的
    assert ask(Q.hermitian(x + y), Q.real(x) & Q.real(y)) is True

    # 断言测试表达式 I*x 是否为 Hermite 和反-Hermite 的
    assert ask(Q.hermitian(I*x), Q.antihermitian(x)) is True
    # 断言测试表达式 I*x 是否为 Hermite 和复数的
    assert ask(Q.hermitian(I*x), Q.complex(x)) is None
    # 断言测试表达式 I*x 是否为 Hermite 的
    assert ask(Q.hermitian(I*x), Q.hermitian(x)) is False
    # 断言测试表达式 I*x 是否为 Hermite 和虚数的
    assert ask(Q.hermitian(I*x), Q.imaginary(x)) is True
    # 断言测试表达式 I*x 是否为 Hermite 和实数的
    assert ask(Q.hermitian(I*x), Q.real(x)) is False
    # 断言测试表达式 x*y 是否为 Hermite 和 Hermite & 实数的
    assert ask(Q.hermitian(x*y), Q.hermitian(x) & Q.real(y)) is True

    # 断言测试表达式 x + y + z 是否为 Hermite 和实数 & 实数 & 实数的
    assert ask(Q.hermitian(x + y + z), Q.real(x) & Q.real(y) & Q.real(z)) is True
    # 断言测试表达式 x + y + z 是否为 Hermite 和实数 & 实数 & 虚数的
    assert ask(Q.hermitian(x + y + z), Q.real(x) & Q.real(y) & Q.imaginary(z)) is False
    # 断言测试表达式 x + y + z 是否为 Hermite 和虚数 & 虚数 & 虚数的
    assert ask(Q.hermitian(x + y + z), Q.real(x) & Q.imaginary(y) & Q.imaginary(z)) is None
    # 断言测试表达式 x + y + z 是否为 Hermite 和虚数 & 虚数 & 虚数的
    assert ask(Q.hermitian(x + y + z), Q.imaginary(x) & Q.imaginary(y) & Q.imaginary(z)) is None

    # 断言测试表达式 x 是否为反-Hermite 的
    assert ask(Q.antihermitian(x)) is None
    # 断言测试表达式 x 是否为反-Hermite 和实数的
    assert ask(Q.antihermitian(x), Q.real(x)) is False
    # 断言测试表达式 x 是否为反-Hermite 和素数的
    assert ask(Q.antihermitian(x), Q.prime(x)) is False

    # 断言测试表达式 x + 1 是否为反-Hermite 和反-Hermite 的
    assert ask(Q.antihermitian(x + 1), Q.antihermitian(x)) is False
    # 断言：询问是否 x + 1 是反厄米的，给定 x 是复数
    assert ask(Q.antihermitian(x + 1), Q.complex(x)) is None
    # 断言：询问是否 x + 1 是反厄米的，给定 x 是厄米的
    assert ask(Q.antihermitian(x + 1), Q.hermitian(x)) is None
    # 断言：询问是否 x + 1 是反厄米的，给定 x 是虚数
    assert ask(Q.antihermitian(x + 1), Q.imaginary(x)) is False
    # 断言：询问是否 x + 1 是反厄米的，给定 x 是实数
    assert ask(Q.antihermitian(x + 1), Q.real(x)) is None
    # 断言：询问是否 x + i 是反厄米的，给定 x 是反厄米的
    assert ask(Q.antihermitian(x + I), Q.antihermitian(x)) is True
    # 断言：询问是否 x + i 是反厄米的，给定 x 是复数
    assert ask(Q.antihermitian(x + I), Q.complex(x)) is None
    # 断言：询问是否 x + i 是反厄米的，给定 x 是厄米的
    assert ask(Q.antihermitian(x + I), Q.hermitian(x)) is None
    # 断言：询问是否 x + i 是反厄米的，给定 x 是虚数
    assert ask(Q.antihermitian(x + I), Q.imaginary(x)) is True
    # 断言：询问是否 x + i 是反厄米的，给定 x 是实数
    assert ask(Q.antihermitian(x + I), Q.real(x)) is False
    # 断言：询问是否 x 是反厄米的，给定 x 是零
    assert ask(Q.antihermitian(x), Q.zero(x)) is True

    # 断言：询问是否 x + y 是反厄米的，给定 x 和 y 都是反厄米的
    assert ask(
        Q.antihermitian(x + y), Q.antihermitian(x) & Q.antihermitian(y)
    ) is True
    # 断言：询问是否 x + y 是反厄米的，给定 x 是反厄米的，y 是复数
    assert ask(
        Q.antihermitian(x + y), Q.antihermitian(x) & Q.complex(y)) is None
    # 断言：询问是否 x + y 是反厄米的，给定 x 是反厄米的，y 是厄米的
    assert ask(
        Q.antihermitian(x + y), Q.antihermitian(x) & Q.hermitian(y)) is None
    # 断言：询问是否 x + y 是反厄米的，给定 x 是反厄米的，y 是虚数
    assert ask(
        Q.antihermitian(x + y), Q.antihermitian(x) & Q.imaginary(y)) is True
    # 断言：询问是否 x + y 是反厄米的，给定 x 是反厄米的，y 是实数
    assert ask(Q.antihermitian(x + y), Q.antihermitian(x) & Q.real(y)
        ) is False
    # 断言：询问是否 x + y 是反厄米的，给定 x 是厄米的，y 是复数
    assert ask(Q.antihermitian(x + y), Q.hermitian(x) & Q.complex(y)) is None
    # 断言：询问是否 x + y 是反厄米的，给定 x 和 y 都是厄米的
    assert ask(Q.antihermitian(x + y), Q.hermitian(x) & Q.hermitian(y)
        ) is None
    # 断言：询问是否 x + y 是反厄米的，给定 x 是厄米的，y 是虚数
    assert ask(
        Q.antihermitian(x + y), Q.hermitian(x) & Q.imaginary(y)) is None
    # 断言：询问是否 x + y 是反厄米的，给定 x 是厄米的，y 是实数
    assert ask(Q.antihermitian(x + y), Q.hermitian(x) & Q.real(y)) is None
    # 断言：询问是否 x + y 是反厄米的，给定 x 是虚数，y 是复数
    assert ask(Q.antihermitian(x + y), Q.imaginary(x) & Q.complex(y)) is None
    # 断言：询问是否 x + y 是反厄米的，给定 x 和 y 都是虚数
    assert ask(Q.antihermitian(x + y), Q.imaginary(x) & Q.imaginary(y)) is True
    # 断言：询问是否 x + y 是反厄米的，给定 x 是虚数，y 是实数
    assert ask(Q.antihermitian(x + y), Q.imaginary(x) & Q.real(y)) is False
    # 断言：询问是否 x + y 是反厄米的，给定 x 是实数，y 是复数
    assert ask(Q.antihermitian(x + y), Q.real(x) & Q.complex(y)) is None
    # 断言：询问是否 x + y 是反厄米的，给定 x 和 y 都是实数
    assert ask(Q.antihermitian(x + y), Q.real(x) & Q.real(y)) is None

    # 断言：询问是否 i*x 是反厄米的，给定 x 是实数
    assert ask(Q.antihermitian(I*x), Q.real(x)) is True
    # 断言：询问是否 i*x 是反厄米的，给定 x 是反厄米的
    assert ask(Q.antihermitian(I*x), Q.antihermitian(x)) is False
    # 断言：询问是否 i*x 是反厄米的，给定 x 是复数
    assert ask(Q.antihermitian(I*x), Q.complex(x)) is None
    # 断言：询问是否 x*y 是反厄米的，给定 x 是反厄米的，y 是实数
    assert ask(Q.antihermitian(x*y), Q.antihermitian(x) & Q.real(y)) is True

    # 断言：询问是否 x + y + z 是反厄米的，给定 x、y、z 都是实数
    assert ask(Q.antihermitian(x + y + z),
        Q.real(x) & Q.real(y) & Q.real(z)) is None
    # 断言：询问是否 x + y + z 是反厄米的，给定 x、y 是实数，z 是虚数
    assert ask(Q.antihermitian(x + y + z),
        Q.real(x) & Q.real(y) & Q.imaginary(z)) is None
    # 断言：询问是否 x + y + z 是反厄米的，给定 x 是实数，y、z 都是虚数
    assert ask(Q.antihermitian(x + y + z),
        Q.real(x) & Q.imaginary(y) & Q.imaginary(z)) is False
    # 断言：询问是否 x + y + z 是反厄米的，给定 x、y、z 都是虚数
    assert ask(Q.antihermitian(x + y + z),
        Q.imaginary(x) & Q.imaginary(y) & Q.imaginary(z)) is True
# 定义装饰器函数，用于同时处理指数和幂运算
@_both_exp_pow
# 测试函数，用于检验关于虚部的逻辑推断
def test_imaginary():
    # 断言：检查是否询问关于 x 是否有虚部，预期返回 None
    assert ask(Q.imaginary(x)) is None
    # 断言：检查如果 x 是实数，则 x 没有虚部，预期返回 False
    assert ask(Q.imaginary(x), Q.real(x)) is False
    # 断言：检查如果 x 是质数，则 x 没有虚部，预期返回 False
    assert ask(Q.imaginary(x), Q.prime(x)) is False

    # 下面一系列断言类似地检查不同复杂表达式对虚部的推断结果

    assert ask(Q.imaginary(x + 1), Q.real(x)) is False
    assert ask(Q.imaginary(x + 1), Q.imaginary(x)) is False
    assert ask(Q.imaginary(x + I), Q.real(x)) is False
    assert ask(Q.imaginary(x + I), Q.imaginary(x)) is True
    assert ask(Q.imaginary(x + y), Q.imaginary(x) & Q.imaginary(y)) is True
    assert ask(Q.imaginary(x + y), Q.real(x) & Q.real(y)) is False
    assert ask(Q.imaginary(x + y), Q.imaginary(x) & Q.real(y)) is False
    assert ask(Q.imaginary(x + y), Q.complex(x) & Q.real(y)) is None
    assert ask(Q.imaginary(x + y + z), Q.real(x) & Q.real(y) & Q.real(z)) is False
    assert ask(Q.imaginary(x + y + z), Q.real(x) & Q.real(y) & Q.imaginary(z)) is None
    assert ask(Q.imaginary(x + y + z), Q.real(x) & Q.imaginary(y) & Q.imaginary(z)) is False

    assert ask(Q.imaginary(I*x), Q.real(x)) is True
    assert ask(Q.imaginary(I*x), Q.imaginary(x)) is False
    assert ask(Q.imaginary(I*x), Q.complex(x)) is None
    assert ask(Q.imaginary(x*y), Q.imaginary(x) & Q.real(y)) is True
    assert ask(Q.imaginary(x*y), Q.real(x) & Q.real(y)) is False

    assert ask(Q.imaginary(I**x), Q.negative(x)) is None
    assert ask(Q.imaginary(I**x), Q.positive(x)) is None
    assert ask(Q.imaginary(I**x), Q.even(x)) is False
    assert ask(Q.imaginary(I**x), Q.odd(x)) is True
    assert ask(Q.imaginary(I**x), Q.imaginary(x)) is False
    assert ask(Q.imaginary((2*I)**x), Q.imaginary(x)) is False
    assert ask(Q.imaginary(x**0), Q.imaginary(x)) is False
    assert ask(Q.imaginary(x**y), Q.imaginary(x) & Q.imaginary(y)) is None
    assert ask(Q.imaginary(x**y), Q.imaginary(x) & Q.real(y)) is None
    assert ask(Q.imaginary(x**y), Q.real(x) & Q.imaginary(y)) is None
    assert ask(Q.imaginary(x**y), Q.real(x) & Q.real(y)) is None
    assert ask(Q.imaginary(x**y), Q.imaginary(x) & Q.integer(y)) is None
    assert ask(Q.imaginary(x**y), Q.imaginary(y) & Q.integer(x)) is None
    assert ask(Q.imaginary(x**y), Q.imaginary(x) & Q.odd(y)) is True
    assert ask(Q.imaginary(x**y), Q.imaginary(x) & Q.rational(y)) is None
    assert ask(Q.imaginary(x**y), Q.imaginary(x) & Q.even(y)) is False

    assert ask(Q.imaginary(x**y), Q.real(x) & Q.integer(y)) is False
    assert ask(Q.imaginary(x**y), Q.positive(x) & Q.real(y)) is False
    assert ask(Q.imaginary(x**y), Q.negative(x) & Q.real(y)) is None
    assert ask(Q.imaginary(x**y), Q.negative(x) & Q.real(y) & ~Q.rational(y)) is False
    assert ask(Q.imaginary(x**y), Q.integer(x) & Q.imaginary(y)) is None
    assert ask(Q.imaginary(x**y), Q.negative(x) & Q.rational(y) & Q.integer(2*y)) is True
    assert ask(Q.imaginary(x**y), Q.negative(x) & Q.rational(y) & ~Q.integer(2*y)) is False
    assert ask(Q.imaginary(x**y), Q.negative(x) & Q.rational(y)) is None
    # 检查 x ** y 是否具有虚部，前提条件是 x 是实数、y 是有理数且不是整数的两倍
    assert ask(Q.imaginary(x**y), Q.real(x) & Q.rational(y) & ~Q.integer(2*y)) is False
    # 检查 x ** y 是否具有虚部，前提条件是 x 是实数、y 是有理数且是整数的两倍
    assert ask(Q.imaginary(x**y), Q.real(x) & Q.rational(y) & Q.integer(2*y)) is None

    # 对数函数
    # 检查 log(I) 是否具有虚部，预期结果为 True
    assert ask(Q.imaginary(log(I))) is True
    # 检查 log(2*I) 是否具有虚部，预期结果为 False
    assert ask(Q.imaginary(log(2*I))) is False
    # 检查 log(I + 1) 是否具有虚部，预期结果为 False
    assert ask(Q.imaginary(log(I + 1))) is False
    # 检查 log(x) 是否具有虚部，前提条件是 x 是复数
    assert ask(Q.imaginary(log(x)), Q.complex(x)) is None
    # 检查 log(x) 是否具有虚部，前提条件是 x 自身具有虚部
    assert ask(Q.imaginary(log(x)), Q.imaginary(x)) is None
    # 检查 log(x) 是否具有虚部，前提条件是 x 是正实数
    assert ask(Q.imaginary(log(x)), Q.positive(x)) is False
    # 检查 log(exp(x)) 是否具有虚部，前提条件是 x 是复数
    assert ask(Q.imaginary(log(exp(x))), Q.complex(x)) is None
    # 检查 log(exp(x)) 是否具有虚部，前提条件是 x 自身具有虚部
    assert ask(Q.imaginary(log(exp(x))), Q.imaginary(x)) is None  # zoo/I/a+I*b
    # 检查 log(exp(I)) 是否具有虚部，预期结果为 True
    assert ask(Q.imaginary(log(exp(I)))) is True

    # 指数函数
    # 检查 exp(x)**x 是否具有虚部，前提条件是 x 自身具有虚部
    assert ask(Q.imaginary(exp(x)**x), Q.imaginary(x)) is False
    # 构造复杂的指数表达式，检查其是否具有虚部，前提条件是 x 是偶数
    eq = Pow(exp(pi*I*x, evaluate=False), x, evaluate=False)
    assert ask(Q.imaginary(eq), Q.even(x)) is False
    # 构造复杂的指数表达式，检查其是否具有虚部，前提条件是 x 是奇数
    eq = Pow(exp(pi*I*x/2, evaluate=False), x, evaluate=False)
    assert ask(Q.imaginary(eq), Q.odd(x)) is True
    # 检查 exp(3*I*pi*x)**x 是否具有虚部，前提条件是 x 是整数
    assert ask(Q.imaginary(exp(3*I*pi*x)**x), Q.integer(x)) is False
    # 检查 exp(2*pi*I) 是否具有虚部，预期结果为 False
    assert ask(Q.imaginary(exp(2*pi*I, evaluate=False))) is False
    # 检查 exp(pi*I/2) 是否具有虚部，预期结果为 True
    assert ask(Q.imaginary(exp(pi*I/2, evaluate=False))) is True

    # issue 7886
    # 检查 Pow(x, 1/4) 是否具有虚部，前提条件是 x 是实数且不是负数
    assert ask(Q.imaginary(Pow(x, Rational(1, 4))), Q.real(x) & Q.negative(x)) is False
# 定义一个测试整数性质的函数
def test_integer():
    # 断言整数属性的问题求解结果为 None
    assert ask(Q.integer(x)) is None
    # 断言两个整数属性的问题求解结果为 True
    assert ask(Q.integer(x), Q.integer(x)) is True
    # 断言整数属性与非整数属性的问题求解结果为 False
    assert ask(Q.integer(x), ~Q.integer(x)) is False
    # 断言整数属性与非实数属性的问题求解结果为 False
    assert ask(Q.integer(x), ~Q.real(x)) is False
    # 断言整数属性与非正数属性的问题求解结果为 None
    assert ask(Q.integer(x), ~Q.positive(x)) is None
    # 断言整数属性与偶数或奇数属性的问题求解结果为 True
    assert ask(Q.integer(x), Q.even(x) | Q.odd(x)) is True

    # 断言对 2*x 的整数属性的问题求解结果为 True
    assert ask(Q.integer(2*x), Q.integer(x)) is True
    # 断言对 2*x 的整数属性和偶数属性的问题求解结果为 True
    assert ask(Q.integer(2*x), Q.even(x)) is True
    # 断言对 2*x 的整数属性和素数属性的问题求解结果为 True
    assert ask(Q.integer(2*x), Q.prime(x)) is True
    # 断言对 2*x 的整数属性和有理数属性的问题求解结果为 None
    assert ask(Q.integer(2*x), Q.rational(x)) is None
    # 断言对 2*x 的整数属性和实数属性的问题求解结果为 None
    assert ask(Q.integer(2*x), Q.real(x)) is None
    # 断言对 sqrt(2)*x 的整数属性的问题求解结果为 False
    assert ask(Q.integer(sqrt(2)*x), Q.integer(x)) is False
    # 断言对 sqrt(2)*x 的整数属性和无理数属性的问题求解结果为 None
    assert ask(Q.integer(sqrt(2)*x), Q.irrational(x)) is None

    # 断言对 x/2 的整数属性和奇数属性的问题求解结果为 False
    assert ask(Q.integer(x/2), Q.odd(x)) is False
    # 断言对 x/2 的整数属性和偶数属性的问题求解结果为 True
    assert ask(Q.integer(x/2), Q.even(x)) is True
    # 断言对 x/3 的整数属性和奇数属性的问题求解结果为 None
    assert ask(Q.integer(x/3), Q.odd(x)) is None
    # 断言对 x/3 的整数属性和偶数属性的问题求解结果为 None
    assert ask(Q.integer(x/3), Q.even(x)) is None


# 定义一个测试负数性质的函数
def test_negative():
    # 断言负数属性的问题求解结果为 True
    assert ask(Q.negative(x), Q.negative(x)) is True
    # 断言负数属性与正数属性的问题求解结果为 False
    assert ask(Q.negative(x), Q.positive(x)) is False
    # 断言负数属性与非实数属性的问题求解结果为 False
    assert ask(Q.negative(x), ~Q.real(x)) is False
    # 断言负数属性与素数属性的问题求解结果为 False
    assert ask(Q.negative(x), Q.prime(x)) is False
    # 断言负数属性与非素数属性的问题求解结果为 None
    assert ask(Q.negative(x), ~Q.prime(x)) is None

    # 断言负数属性对 -x 的问题求解结果为 True
    assert ask(Q.negative(-x), Q.positive(x)) is True
    # 断言负数属性对 -x 的问题求解结果为 None
    assert ask(Q.negative(-x), ~Q.positive(x)) is None
    # 断言负数属性对 -x 的问题求解结果为 False
    assert ask(Q.negative(-x), Q.negative(x)) is False
    # 断言负数属性对 -x 的问题求解结果为 True
    assert ask(Q.negative(-x), Q.positive(x)) is True

    # 断言负数属性对 x - 1 的问题求解结果为 True
    assert ask(Q.negative(x - 1), Q.negative(x)) is True
    # 断言负数属性对 x + y 的问题求解结果为 None
    assert ask(Q.negative(x + y)) is None
    # 断言负数属性对 x + y 的问题求解结果为 None
    assert ask(Q.negative(x + y), Q.negative(x)) is None
    # 断言负数属性对 x + y 的问题求解结果为 True
    assert ask(Q.negative(x + y), Q.negative(x) & Q.negative(y)) is True
    # 断言负数属性对 x + y 的问题求解结果为 True
    assert ask(Q.negative(x + y), Q.negative(x) & Q.nonpositive(y)) is True
    # 断言负数属性对 2 + I 的问题求解结果为 False
    assert ask(Q.negative(2 + I)) is False
    # 断言复数属性对 cos(I)**2 + sin(I)**2 - 1 的问题求解结果为 None
    # 虽然这个表达式可能为 False，但它代表了不精确计算为零的情况
    assert ask(Q.negative(cos(I)**2 + sin(I)**2 - 1)) is None
    # 断言负数属性对 -I + I*(cos(2)**2 + sin(2)**2) 的问题求解结果为 None
    assert ask(Q.negative(-I + I*(cos(2)**2 + sin(2)**2))) is None

    # 断言负数属性对 x**2 的问题求解结果为 None
    assert ask(Q.negative(x**2)) is None
    # 断言负数属性对 x**2 的问题求解结果和实数属性的问题求解结果为 False
    assert ask(Q.negative(x**2), Q.real(x)) is False
    # 断言负数属性对 x**1.4 的问题求解结果和实数属性的问题求解结果为 None
    assert ask(Q.negative(x**1.4), Q.real(x)) is None

    # 断言负数属性对 x**I 的问题求解结果和正数属性的问题求解结果为 None
    assert ask(Q.negative(x**I), Q.positive(x)) is None

    # 断言负数属性对 x*y 的问题求解结果为 None
    assert ask(Q.negative(x*y)) is None
    # 断言负数属性对 x*y 的问题求解结果和正数属性与正数属性的问题求解结果为 False
    assert ask(Q.negative(x*y), Q.positive(x) & Q.positive(y)) is False
    # 断言负数属性对 x*y 的问题求解结果和正数属性与负数属性的问题求解结果为 True
    assert ask(Q.negative(x*y), Q.positive(x) & Q.negative(y)) is True
    # 断言负数属性对 x*y 的问题求解结果和复数属性与复数属性的问题求解结果为 None
    assert ask(Q.negative(x*y), Q.complex(x) & Q.complex(y)) is None

    # 断言负数属性对 x**y 的问题求解结果为 None
    assert ask(Q.negative(x**y)) is None
    # 断言负数属性对 x**y 的问题求解结果和负数属性与偶数属性的问题求解结果为 False
    assert ask(Q.negative(x**y), Q.negative(x) & Q.even(y)) is False
    # 断言负数属性对 x**y 的问题求解结果和负数属性与奇数属性的问题求解结果为 True
    assert ask(Q.negative(x**y), Q.negative(x) & Q.odd(y)) is True
    # 断言负数属性对 x**y 的问题求解结果和正数属性与整数属性的问题求解结果为 False
    assert ask(Q.negative(x**y), Q.positive(x) & Q.integer(y)) is False

    # 断言负数属性对 Abs(x) 的问题求
    # 使用 ask 函数来检查关于 x 的条件
    assert ask(Q.nonzero(x), Q.negative(x) | Q.positive(x)) is True

    # 检查 x + y 是否为非零值的条件
    assert ask(Q.nonzero(x + y)) is None
    # 检查 x + y 是否为正数 x 和 y 同时为正数的条件
    assert ask(Q.nonzero(x + y), Q.positive(x) & Q.positive(y)) is True
    # 检查 x + y 是否为正数 x 为正数且 y 为负数的条件
    assert ask(Q.nonzero(x + y), Q.positive(x) & Q.negative(y)) is None
    # 检查 x + y 是否为正数 x 和 y 同时为负数的条件
    assert ask(Q.nonzero(x + y), Q.negative(x) & Q.negative(y)) is True

    # 检查 2*x 是否为非零值的条件
    assert ask(Q.nonzero(2*x)) is None
    # 检查 2*x 是否为非零值 x 为正数的条件
    assert ask(Q.nonzero(2*x), Q.positive(x)) is True
    # 检查 2*x 是否为非零值 x 为负数的条件
    assert ask(Q.nonzero(2*x), Q.negative(x)) is True
    # 检查 x*y 是否为非零值 x 为非零值的条件
    assert ask(Q.nonzero(x*y), Q.nonzero(x)) is None
    # 检查 x*y 是否为非零值 x 和 y 同时为非零值的条件
    assert ask(Q.nonzero(x*y), Q.nonzero(x) & Q.nonzero(y)) is True

    # 检查 x**y 是否为非零值 x 为非零值的条件
    assert ask(Q.nonzero(x**y), Q.nonzero(x)) is True

    # 检查 Abs(x) 是否为非零值的条件
    assert ask(Q.nonzero(Abs(x))) is None
    # 检查 Abs(x) 是否为非零值 x 为非零值的条件
    assert ask(Q.nonzero(Abs(x)), Q.nonzero(x)) is True

    # 检查 log(exp(2*I)) 是否为零的条件
    assert ask(Q.nonzero(log(exp(2*I)))) is False
    # 虽然这个表达式可能为 False，但它代表了不精确计算结果为零的情况
    assert ask(Q.nonzero(cos(1)**2 + sin(1)**2 - 1)) is None
# 定义测试函数 test_zero，用于测试 Q.zero(x) 查询
def test_zero():
    # 断言 Q.zero(x) 的返回值为 None
    assert ask(Q.zero(x)) is None
    # 断言 Q.zero(x) & Q.real(x) 的返回值为 None
    assert ask(Q.zero(x), Q.real(x)) is None
    # 断言 Q.zero(x) & Q.positive(x) 的返回值为 False
    assert ask(Q.zero(x), Q.positive(x)) is False
    # 断言 Q.zero(x) & Q.negative(x) 的返回值为 False
    assert ask(Q.zero(x), Q.negative(x)) is False
    # 断言 Q.zero(x) & (Q.negative(x) | Q.positive(x)) 的返回值为 False
    assert ask(Q.zero(x), Q.negative(x) | Q.positive(x)) is False

    # 断言 Q.zero(x) & (Q.nonnegative(x) & Q.nonpositive(x)) 的返回值为 True
    assert ask(Q.zero(x), Q.nonnegative(x) & Q.nonpositive(x)) is True

    # 断言 Q.zero(x + y) 的返回值为 None
    assert ask(Q.zero(x + y)) is None
    # 断言 Q.zero(x + y) & (Q.positive(x) & Q.positive(y)) 的返回值为 False
    assert ask(Q.zero(x + y), Q.positive(x) & Q.positive(y)) is False
    # 断言 Q.zero(x + y) & (Q.positive(x) & Q.negative(y)) 的返回值为 None
    assert ask(Q.zero(x + y), Q.positive(x) & Q.negative(y)) is None
    # 断言 Q.zero(x + y) & (Q.negative(x) & Q.negative(y)) 的返回值为 False
    assert ask(Q.zero(x + y), Q.negative(x) & Q.negative(y)) is False

    # 断言 Q.zero(2*x) 的返回值为 None
    assert ask(Q.zero(2*x)) is None
    # 断言 Q.zero(2*x) & Q.positive(x) 的返回值为 False
    assert ask(Q.zero(2*x), Q.positive(x)) is False
    # 断言 Q.zero(2*x) & Q.negative(x) 的返回值为 False
    assert ask(Q.zero(2*x), Q.negative(x)) is False
    # 断言 Q.zero(x*y) & Q.nonzero(x) 的返回值为 None
    assert ask(Q.zero(x*y), Q.nonzero(x)) is None

    # 断言 Q.zero(Abs(x)) 的返回值为 None
    assert ask(Q.zero(Abs(x))) is None
    # 断言 Q.zero(Abs(x)) & Q.zero(x) 的返回值为 True
    assert ask(Q.zero(Abs(x)), Q.zero(x)) is True

    # 断言 Q.integer(x) & Q.zero(x) 的返回值为 True
    assert ask(Q.integer(x), Q.zero(x)) is True
    # 断言 Q.even(x) & Q.zero(x) 的返回值为 True
    assert ask(Q.even(x), Q.zero(x)) is True
    # 断言 Q.odd(x) & Q.zero(x) 的返回值为 False
    assert ask(Q.odd(x), Q.zero(x)) is False
    # 断言 Q.zero(x) & Q.even(x) 的返回值为 None
    assert ask(Q.zero(x), Q.even(x)) is None
    # 断言 Q.zero(x) & Q.odd(x) 的返回值为 False
    assert ask(Q.zero(x), Q.odd(x)) is False
    # 断言 Q.zero(x) | Q.zero(y) & Q.zero(x*y) 的返回值为 True
    assert ask(Q.zero(x) | Q.zero(y), Q.zero(x*y)) is True


# 定义测试函数 test_odd_query，用于测试 Q.odd(x) 查询
def test_odd_query():
    # 断言 Q.odd(x) 的返回值为 None
    assert ask(Q.odd(x)) is None
    # 断言 Q.odd(x) & Q.odd(x) 的返回值为 True
    assert ask(Q.odd(x), Q.odd(x)) is True
    # 断言 Q.odd(x) & Q.integer(x) 的返回值为 None
    assert ask(Q.odd(x), Q.integer(x)) is None
    # 断言 Q.odd(x) & ~Q.integer(x) 的返回值为 False
    assert ask(Q.odd(x), ~Q.integer(x)) is False
    # 断言 Q.odd(x) & Q.rational(x) 的返回值为 None
    assert ask(Q.odd(x), Q.rational(x)) is None
    # 断言 Q.odd(x) & Q.positive(x) 的返回值为 None
    assert ask(Q.odd(x), Q.positive(x)) is None

    # 断言 Q.odd(-x) & Q.odd(x) 的返回值为 True
    assert ask(Q.odd(-x), Q.odd(x)) is True

    # 断言 Q.odd(2*x) 的返回值为 None
    assert ask(Q.odd(2*x)) is None
    # 断言 Q.odd(2*x) & Q.integer(x) 的返回值为 False
    assert ask(Q.odd(2*x), Q.integer(x)) is False
    # 断言 Q.odd(2*x) & Q.odd(x) 的返回值为 False
    assert ask(Q.odd(2*x), Q.odd(x)) is False
    # 断言 Q.odd(2*x) & Q.irrational(x) 的返回值为 False
    assert ask(Q.odd(2*x), Q.irrational(x)) is False
    # 断言 Q.odd(2*x) & ~Q.integer(x) 的返回值为 None
    assert ask(Q.odd(2*x), ~Q.integer(x)) is None
    # 断言 Q.odd(3*x) & Q.integer(x) 的返回值为 None
    assert ask(Q.odd(3*x), Q.integer(x)) is None

    # 断言 Q.odd(x/3) & Q.odd(x) 的返回值为 None
    assert ask(Q.odd(x/3), Q.odd(x)) is None
    # 断言 Q.odd(x/3) & Q.even(x) 的返回值为 None
    assert ask(Q.odd(x/3), Q.even(x)) is None

    # 断言 Q.odd(x + 1) & Q.even(x) 的返回值为 True
    assert ask(Q.odd(x + 1), Q.even(x)) is True
    # 断言 Q.odd(x + 2) & Q.even(x) 的返回值为 False
    assert ask(Q.odd(x + 2), Q.even(x)) is False
    # 断言 Q.odd(x + 2) & Q.odd(x) 的返回值为 True
    assert ask(Q.odd(x + 2), Q.odd(x)) is True
    # 断言 Q.odd(3 - x) & Q.odd(x) 的返回值为 False
    assert ask(Q.odd(3 - x), Q.odd(x)) is False
    # 断言 Q.odd(3 - x) & Q.even(x) 的返回值为 True
    assert ask(Q.odd(3 - x), Q.even(x)) is True
    # 断言 Q.odd(3 + x) & Q.odd(x) 的返回值为 False
    assert ask(Q.odd(3 + x), Q.odd(x)) is False
    # 断言 Q.odd(3 + x) & Q.even(x) 的返回值为 True
    assert ask(Q.odd(3 + x), Q.even(x)) is True
    # 断言 Q.odd(x + y) & (Q.odd(x) & Q.odd(y)) 的返回值为 False
    assert ask(Q.odd(x + y), Q.odd(x) & Q.odd(y)) is False
    # 断言 Q.odd(x + y) & (Q.odd(x) & Q.even(y)) 的返回值为 True
    assert ask(Q.odd(x + y), Q.odd(x) & Q.even(y)) is True
    # 断言 Q.odd(x - y) & (Q.even(x) & Q.odd(y)) 的返回值为 True
    assert ask(Q.odd(x - y), Q.even(x) & Q.odd(y)) is True
    # 断言 Q.odd(x - y) & (Q.odd(x) & Q.odd(y)) 的返回值为 False
    assert ask(Q.odd(x - y), Q.odd(x) & Q.odd(y)) is False

    # 断言 Q.odd(x + y + z) & (Q.odd(x) & Q.odd(y) & Q.even(z)) 的返回
    # 断言：检查表达式 Q.odd(2*x*y)，Q.rational(x) & Q.rational(x) 是否为真
    assert ask(Q.odd(2*x*y), Q.rational(x) & Q.rational(x)) is None

    # 断言：检查表达式 Q.odd(2*x*y)，Q.irrational(x) & Q.irrational(x) 是否为真
    assert ask(Q.odd(2*x*y), Q.irrational(x) & Q.irrational(x)) is None

    # 断言：检查表达式 Q.odd(Abs(x)) 是否为真
    assert ask(Q.odd(Abs(x)), Q.odd(x)) is True

    # 断言：检查表达式 Q.odd((-1)**n)，Q.integer(n) 是否为真
    assert ask(Q.odd((-1)**n), Q.integer(n)) is True

    # 断言：检查表达式 Q.odd(k**2)，Q.even(k) 是否为假
    assert ask(Q.odd(k**2), Q.even(k)) is False

    # 断言：检查表达式 Q.odd(n**2)，Q.odd(n) 是否为真
    assert ask(Q.odd(n**2), Q.odd(n)) is True

    # 断言：检查表达式 Q.odd(3**k)，Q.even(k) 是否为 None
    assert ask(Q.odd(3**k), Q.even(k)) is None

    # 断言：检查表达式 Q.odd(k**m)，Q.even(k) & Q.integer(m) & ~Q.negative(m) 是否为 None
    assert ask(Q.odd(k**m), Q.even(k) & Q.integer(m) & ~Q.negative(m)) is None

    # 断言：检查表达式 Q.odd(n**m)，Q.odd(n) & Q.integer(m) & ~Q.negative(m) 是否为真
    assert ask(Q.odd(n**m), Q.odd(n) & Q.integer(m) & ~Q.negative(m)) is True

    # 断言：检查表达式 Q.odd(k**p)，Q.even(k) & Q.integer(p) & Q.positive(p) 是否为假
    assert ask(Q.odd(k**p), Q.even(k) & Q.integer(p) & Q.positive(p)) is False

    # 断言：检查表达式 Q.odd(n**p)，Q.odd(n) & Q.integer(p) & Q.positive(p) 是否为真
    assert ask(Q.odd(n**p), Q.odd(n) & Q.integer(p) & Q.positive(p)) is True

    # 断言：检查表达式 Q.odd(m**k)，Q.even(k) & Q.integer(m) & ~Q.negative(m) 是否为 None
    assert ask(Q.odd(m**k), Q.even(k) & Q.integer(m) & ~Q.negative(m)) is None

    # 断言：检查表达式 Q.odd(p**k)，Q.even(k) & Q.integer(p) & Q.positive(p) 是否为 None
    assert ask(Q.odd(p**k), Q.even(k) & Q.integer(p) & Q.positive(p)) is None

    # 断言：检查表达式 Q.odd(m**n)，Q.odd(n) & Q.integer(m) & ~Q.negative(m) 是否为 None
    assert ask(Q.odd(m**n), Q.odd(n) & Q.integer(m) & ~Q.negative(m)) is None

    # 断言：检查表达式 Q.odd(p**n)，Q.odd(n) & Q.integer(p) & Q.positive(p) 是否为 None
    assert ask(Q.odd(p**n), Q.odd(n) & Q.integer(p) & Q.positive(p)) is None

    # 断言：检查表达式 Q.odd(k**x)，Q.even(k) 是否为 None
    assert ask(Q.odd(k**x), Q.even(k)) is None

    # 断言：检查表达式 Q.odd(n**x)，Q.odd(n) 是否为 None
    assert ask(Q.odd(n**x), Q.odd(n)) is None

    # 断言：检查表达式 Q.odd(x*y)，Q.integer(x) & Q.integer(y) 是否为 None
    assert ask(Q.odd(x*y), Q.integer(x) & Q.integer(y)) is None

    # 断言：检查表达式 Q.odd(x*x)，Q.integer(x) 是否为 None
    assert ask(Q.odd(x*x), Q.integer(x)) is None

    # 断言：检查表达式 Q.odd(x*(x + y))，Q.integer(x) & Q.odd(y) 是否为假
    assert ask(Q.odd(x*(x + y)), Q.integer(x) & Q.odd(y)) is False

    # 断言：检查表达式 Q.odd(x*(x + y))，Q.integer(x) & Q.even(y) 是否为 None
    assert ask(Q.odd(x*(x + y)), Q.integer(x) & Q.even(y)) is None
@XFAIL
# 标记为 XFAIL 的测试函数，预期该测试会失败

def test_oddness_in_ternary_integer_product_with_odd():
    # 测试整数乘积中的奇偶性推理是否与术语的顺序无关。
    # 在测试时，术语的顺序取决于SymPy的符号顺序，因此我们尝试通过修改符号名称来强制使用不同的顺序。
    assert ask(Q.odd(x*y*(y + z)), Q.integer(x) & Q.integer(y) & Q.odd(z)) is False
    assert ask(Q.odd(y*x*(x + z)), Q.integer(x) & Q.integer(y) & Q.odd(z)) is False


def test_oddness_in_ternary_integer_product_with_even():
    # 测试整数乘积中的奇偶性推理与 z 是偶数的情况。
    assert ask(Q.odd(x*y*(y + z)), Q.integer(x) & Q.integer(y) & Q.even(z)) is None


def test_prime():
    # 测试素数的性质
    assert ask(Q.prime(x), Q.prime(x)) is True
    assert ask(Q.prime(x), ~Q.prime(x)) is False
    assert ask(Q.prime(x), Q.integer(x)) is None
    assert ask(Q.prime(x), ~Q.integer(x)) is False

    assert ask(Q.prime(2*x), Q.integer(x)) is None
    assert ask(Q.prime(x*y)) is None
    assert ask(Q.prime(x*y), Q.prime(x)) is None
    assert ask(Q.prime(x*y), Q.integer(x) & Q.integer(y)) is None
    assert ask(Q.prime(4*x), Q.integer(x)) is False
    assert ask(Q.prime(4*x)) is None

    assert ask(Q.prime(x**2), Q.integer(x)) is False
    assert ask(Q.prime(x**2), Q.prime(x)) is False
    assert ask(Q.prime(x**y), Q.integer(x) & Q.integer(y)) is False


@_both_exp_pow
# 使用装饰器 `_both_exp_pow` 标记的测试函数

def test_positive():
    assert ask(Q.positive(cos(I) ** 2 + sin(I) ** 2 - 1)) is None
    assert ask(Q.positive(x), Q.positive(x)) is True
    assert ask(Q.positive(x), Q.negative(x)) is False
    assert ask(Q.positive(x), Q.nonzero(x)) is None

    assert ask(Q.positive(-x), Q.positive(x)) is False
    assert ask(Q.positive(-x), Q.negative(x)) is True

    assert ask(Q.positive(x + y), Q.positive(x) & Q.positive(y)) is True
    assert ask(Q.positive(x + y), Q.positive(x) & Q.nonnegative(y)) is True
    assert ask(Q.positive(x + y), Q.positive(x) & Q.negative(y)) is None
    assert ask(Q.positive(x + y), Q.positive(x) & Q.imaginary(y)) is False

    assert ask(Q.positive(2*x), Q.positive(x)) is True
    assumptions = Q.positive(x) & Q.negative(y) & Q.negative(z) & Q.positive(w)
    assert ask(Q.positive(x*y*z)) is None
    assert ask(Q.positive(x*y*z), assumptions) is True
    assert ask(Q.positive(-x*y*z), assumptions) is False

    assert ask(Q.positive(x**I), Q.positive(x)) is None

    assert ask(Q.positive(x**2), Q.positive(x)) is True
    assert ask(Q.positive(x**2), Q.negative(x)) is True
    assert ask(Q.positive(x**3), Q.negative(x)) is False
    assert ask(Q.positive(1/(1 + x**2)), Q.real(x)) is True
    assert ask(Q.positive(2**I)) is False
    assert ask(Q.positive(2 + I)) is False
    # 虽然这可能是 False，但它代表了精度不足以计算为零的表达式。
    assert ask(Q.positive(cos(I)**2 + sin(I)**2 - 1)) is None
    assert ask(Q.positive(-I + I*(cos(2)**2 + sin(2)**2))) is None

    # 指数函数
    assert ask(Q.positive(exp(x)), Q.real(x)) is True
    assert ask(~Q.negative(exp(x)), Q.real(x)) is True
    # 检查表达式 x + exp(x) 在 x 为实数时是否为正数，预期结果为 None
    assert ask(Q.positive(x + exp(x)), Q.real(x)) is None

    # 检查表达式 exp(x) 在 x 为虚数时是否为正数，预期结果为 None
    assert ask(Q.positive(exp(x)), Q.imaginary(x)) is None

    # 检查表达式 exp(2*pi*I, evaluate=False) 在 x 为虚数时是否为正数，预期结果为 True
    assert ask(Q.positive(exp(2*pi*I, evaluate=False)), Q.imaginary(x)) is True

    # 检查表达式 exp(pi*I, evaluate=False) 在 x 为虚数时是否为负数，预期结果为 True
    assert ask(Q.negative(exp(pi*I, evaluate=False)), Q.imaginary(x)) is True

    # 检查表达式 exp(x*pi*I) 在 x 为偶数时是否为正数，预期结果为 True
    assert ask(Q.positive(exp(x*pi*I)), Q.even(x)) is True

    # 检查表达式 exp(x*pi*I) 在 x 为奇数时是否为正数，预期结果为 False
    assert ask(Q.positive(exp(x*pi*I)), Q.odd(x)) is False

    # 检查表达式 exp(x*pi*I) 在 x 为实数时是否为正数，预期结果为 None
    assert ask(Q.positive(exp(x*pi*I)), Q.real(x)) is None

    # 对数函数
    # 检查表达式 log(x) 在 x 为虚数时是否为正数，预期结果为 False
    assert ask(Q.positive(log(x)), Q.imaginary(x)) is False

    # 检查表达式 log(x) 在 x 为负数时是否为正数，预期结果为 False
    assert ask(Q.positive(log(x)), Q.negative(x)) is False

    # 检查表达式 log(x) 在 x 为正数时是否为正数，预期结果为 None
    assert ask(Q.positive(log(x)), Q.positive(x)) is None

    # 检查表达式 log(x + 2) 在 x 为正数时是否为正数，预期结果为 True
    assert ask(Q.positive(log(x + 2)), Q.positive(x)) is True

    # 阶乘函数
    # 检查表达式 factorial(x) 在 x 为整数且为正数时是否为正数
    assert ask(Q.positive(factorial(x)), Q.integer(x) & Q.positive(x))

    # 检查表达式 factorial(x) 在 x 为整数时是否为正数，预期结果为 None
    assert ask(Q.positive(factorial(x)), Q.integer(x)) is None

    # 绝对值函数
    # 检查表达式 Abs(x) 是否为正数，预期结果为 None（因为 Abs(0) = 0）
    assert ask(Q.positive(Abs(x))) is None

    # 检查表达式 Abs(x) 在 x 为正数时是否为正数，预期结果为 True
    assert ask(Q.positive(Abs(x)), Q.positive(x)) is True
def test_nonpositive():
    # 检查是否小于等于零
    assert ask(Q.nonpositive(-1))
    # 检查是否小于等于零
    assert ask(Q.nonpositive(0))
    # 检查是否小于等于零，应返回 False
    assert ask(Q.nonpositive(1)) is False
    # 检查是否非正数，应返回 True
    assert ask(~Q.positive(x), Q.nonpositive(x))
    # 检查是否非正数，应返回 False
    assert ask(Q.nonpositive(x), Q.positive(x)) is False
    # 检查是否非正数，应返回 False（sqrt(-1) 不是实数）
    assert ask(Q.nonpositive(sqrt(-1))) is False
    # 检查是否非正数，应返回 False（虚部非零）
    assert ask(Q.nonpositive(x), Q.imaginary(x)) is False


def test_nonnegative():
    # 检查是否非负数，应返回 False
    assert ask(Q.nonnegative(-1)) is False
    # 检查是否非负数
    assert ask(Q.nonnegative(0))
    # 检查是否非负数
    assert ask(Q.nonnegative(1))
    # 检查是否非负数，应返回 True
    assert ask(~Q.negative(x), Q.nonnegative(x))
    # 检查是否非负数，应返回 False
    assert ask(Q.nonnegative(x), Q.negative(x)) is False
    # 检查是否非负数，应返回 False（sqrt(-1) 不是实数）
    assert ask(Q.nonnegative(sqrt(-1))) is False
    # 检查是否非负数，应返回 False（虚部非零）
    assert ask(Q.nonnegative(x), Q.imaginary(x)) is False


def test_real_basic():
    # 检查是否为实数，应返回 None（x 未指定）
    assert ask(Q.real(x)) is None
    # 检查是否为实数，应返回 True
    assert ask(Q.real(x), Q.real(x)) is True
    # 检查是否为实数，应返回 True（x 非零）
    assert ask(Q.real(x), Q.nonzero(x)) is True
    # 检查是否为实数，应返回 True（x 为正数）
    assert ask(Q.real(x), Q.positive(x)) is True
    # 检查是否为实数，应返回 True（x 为负数）
    assert ask(Q.real(x), Q.negative(x)) is True
    # 检查是否为实数，应返回 True（x 为整数）
    assert ask(Q.real(x), Q.integer(x)) is True
    # 检查是否为实数，应返回 True（x 为偶数）
    assert ask(Q.real(x), Q.even(x)) is True
    # 检查是否为实数，应返回 True（x 为素数）
    assert ask(Q.real(x), Q.prime(x)) is True

    # 检查是否为实数，应返回 True（x/sqrt(2) 为实数）
    assert ask(Q.real(x/sqrt(2)), Q.real(x)) is True
    # 检查是否为实数，应返回 False（x/sqrt(-2) 为虚数）
    assert ask(Q.real(x/sqrt(-2)), Q.real(x)) is False

    # 检查是否为实数，应返回 True（x + 1 为实数）
    assert ask(Q.real(x + 1), Q.real(x)) is True
    # 检查是否为实数，应返回 False（x + I 为虚数）
    assert ask(Q.real(x + I), Q.real(x)) is False
    # 检查是否为复数，应返回 None（x + I 为复数）
    assert ask(Q.real(x + I), Q.complex(x)) is None

    # 检查是否为实数，应返回 True（2*x 为实数）
    assert ask(Q.real(2*x), Q.real(x)) is True
    # 检查是否为实数，应返回 False（I*x 为虚数）
    assert ask(Q.real(I*x), Q.real(x)) is False
    # 检查是否为虚数，应返回 True（I*x 为虚数）
    assert ask(Q.real(I*x), Q.imaginary(x)) is True
    # 检查是否为复数，应返回 None（I*x 为复数）
    assert ask(Q.real(I*x), Q.complex(x)) is None


def test_real_pow():
    # 检查是否为实数，应返回 True（x**2 为实数）
    assert ask(Q.real(x**2), Q.real(x)) is True
    # 检查是否为实数，应返回 False（sqrt(x) 为虚数，当 x 为负数时）
    assert ask(Q.real(sqrt(x)), Q.negative(x)) is False
    # 检查是否为实数，应返回 True（x**y 为实数，当 x 为实数且 y 为整数时）
    assert ask(Q.real(x**y), Q.real(x) & Q.integer(y)) is True
    # 检查是否为实数，应返回 None（x**y 为实数，当 x 和 y 都为实数时）
    assert ask(Q.real(x**y), Q.real(x) & Q.real(y)) is None
    # 检查是否为实数，应返回 True（x**y 为实数，当 x 为正数且 y 为实数时）
    assert ask(Q.real(x**y), Q.positive(x) & Q.real(y)) is True
    # 检查是否为实数，应返回 None（x**y 可能为 I**I 或 (2*I)**I，即虚数的虚数次幂）
    assert ask(Q.real(x**y), Q.imaginary(x) & Q.imaginary(y)) is None
    # 检查是否为实数，应返回 None（x**y 可能为 I**1 或 I**0，即虚数的整数次幂）
    assert ask(Q.real(x**y), Q.imaginary(x) & Q.real(y)) is None
    # 检查是否为实数，应返回 None（x**y 可能为 exp(2*pi*I) 或 2**I，即实数的虚数次幂）
    assert ask(Q.real(x**y), Q.real(x) & Q.imaginary(y)) is None
    # 检查是否为实数，应返回 True（x**0 为实数，即任何数的零次幂为实数）
    assert ask(Q.real(x**0), Q.imaginary(x)) is True
    # 检查是否为实数，应返回 True（x**y 为实数，当 x 为实数且 y 为整数时）
    assert ask(Q.real(x**y), Q.real(x) & Q.integer(y)) is True
    # 检查是否为实数，应返回 True（x**y 为实数，当 x 为正数且 y 为实数时）
    assert ask(Q.real(x**y), Q.positive(x) & Q.real(y)) is True
    # 检查是否为实数，应返回 None（x**y 为实数，当 x 为实数且 y 为有理数时）
    assert ask(Q.real(x**y), Q.real(x) & Q.rational(y)) is None
    # 检查是否为实数，应返回 None（x**y 可能为 I**1 或 I**0，即虚数的整数次幂）
    assert ask(Q.real(x**y), Q.imaginary(x) & Q.integer(y)) is None
    # 检查是否为实数，应返回 False（x**y 为虚数，当 x 为虚数且 y 为奇数时）
    assert ask(Q.real(x**y), Q.imaginary(x) & Q.odd(y)) is False
    # 检查是否为实数，应返回 True（x**y 为实数，当 x 为虚数且 y 为偶数时）
    assert ask(Q.real(x**y), Q.imaginary(x) & Q.even(y)) is True
    # 检查是否为实数，应返回 True（x**(y/z) 为实数，当 x 为正数、y/z 为有理数、z 为偶数时）
    assert ask(Q.real(x**(y/z)), Q.real(x) & Q.real(y/z) & Q.rational(y
    # 断言：检查表达式 x**(y/z) 是否为实数，同时满足 x 是实数、y/z 是实数以及 x 是负数的条件，期望结果为 False
    assert ask(Q.real(x**(y/z)), Q.real(x) & Q.real(y/z) & Q.negative(x)) is False
    
    # 断言：检查表达式 (-I)**i 是否为实数，同时满足 i 是虚数的条件，期望结果为 True
    assert ask(Q.real((-I)**i), Q.imaginary(i)) is True
    
    # 断言：检查表达式 I**i 是否为实数，同时满足 i 是虚数的条件，期望结果为 True
    assert ask(Q.real(I**i), Q.imaginary(i)) is True
    
    # 断言：检查表达式 i**i 是否为实数，同时满足 i 是虚数的条件，预期结果不确定（None），因为 i 可能为 2*I
    assert ask(Q.real(i**i), Q.imaginary(i)) is None  # i might be 2*I
    
    # 断言：检查表达式 x**i 是否为实数，同时满足 i 是虚数的条件，预期结果不确定（None），因为 x 可能为 0
    assert ask(Q.real(x**i), Q.imaginary(i)) is None  # x could be 0
    
    # 断言：检查表达式 x**(I*pi/log(x)) 是否为实数，同时满足 x 是实数的条件，期望结果为 True
    assert ask(Q.real(x**(I*pi/log(x))), Q.real(x)) is True
# 定义一个装饰器函数，用于同时处理指数和幂的测试函数
@_both_exp_pow
def test_real_functions():
    # 正弦和余弦函数的实部检查
    assert ask(Q.real(sin(x))) is None
    assert ask(Q.real(cos(x))) is None
    # 确认 x 是实数时，正弦和余弦函数的实部为真
    assert ask(Q.real(sin(x)), Q.real(x)) is True
    assert ask(Q.real(cos(x)), Q.real(x)) is True

    # 指数函数的实部检查
    assert ask(Q.real(exp(x))) is None
    assert ask(Q.real(exp(x)), Q.real(x)) is True
    assert ask(Q.real(x + exp(x)), Q.real(x)) is True
    assert ask(Q.real(exp(2*pi*I, evaluate=False))) is True
    assert ask(Q.real(exp(pi*I, evaluate=False))) is True
    assert ask(Q.real(exp(pi*I/2, evaluate=False))) is False

    # 对数函数的实部检查
    assert ask(Q.real(log(I))) is False
    assert ask(Q.real(log(2*I))) is False
    assert ask(Q.real(log(I + 1))) is False
    assert ask(Q.real(log(x)), Q.complex(x)) is None
    assert ask(Q.real(log(x)), Q.imaginary(x)) is False
    assert ask(Q.real(log(exp(x))), Q.imaginary(x)) is None  # exp(2*pi*I) is 1, log(exp(pi*I)) is pi*I (disregarding periodicity)
    assert ask(Q.real(log(exp(x))), Q.complex(x)) is None
    eq = Pow(exp(2*pi*I*x, evaluate=False), x, evaluate=False)
    assert ask(Q.real(eq), Q.integer(x)) is True
    assert ask(Q.real(exp(x)**x), Q.imaginary(x)) is True
    assert ask(Q.real(exp(x)**x), Q.complex(x)) is None

    # 实部检查 Q.complexes
    assert ask(Q.real(re(x))) is True
    assert ask(Q.real(im(x))) is True


def test_matrix():
    # Hermite矩阵的性质检查
    assert ask(Q.hermitian(Matrix([[2, 2 + I, 4], [2 - I, 3, I], [4, -I, 1]]))) == True
    assert ask(Q.hermitian(Matrix([[2, 2 + I, 4], [2 + I, 3, I], [4, -I, 1]]))) == False
    z = symbols('z', complex=True)
    assert ask(Q.hermitian(Matrix([[2, 2 + I, z], [2 - I, 3, I], [4, -I, 1]]))) == None
    assert ask(Q.hermitian(SparseMatrix(((25, 15, -5), (15, 18, 0), (-5, 0, 11))))) == True
    assert ask(Q.hermitian(SparseMatrix(((25, 15, -5), (15, I, 0), (-5, 0, 11))))) == False
    assert ask(Q.hermitian(SparseMatrix(((25, 15, -5), (15, z, 0), (-5, 0, 11))))) == None

    # 反-Hermite矩阵的性质检查
    A = Matrix([[0, -2 - I, 0], [2 - I, 0, -I], [0, -I, 0]])
    B = Matrix([[-I, 2 + I, 0], [-2 + I, 0, 2 + I], [0, -2 + I, -I]])
    assert ask(Q.antihermitian(A)) is True
    assert ask(Q.antihermitian(B)) is True
    assert ask(Q.antihermitian(A**2)) is False
    C = (B**3)
    C.simplify()
    assert ask(Q.antihermitian(C)) is True
    _A = Matrix([[0, -2 - I, 0], [z, 0, -I], [0, -I, 0]])
    assert ask(Q.antihermitian(_A)) is None


@_both_exp_pow
def test_algebraic():
    # 代数数的性质检查
    assert ask(Q.algebraic(x)) is None

    assert ask(Q.algebraic(I)) is True
    assert ask(Q.algebraic(2*I)) is True
    assert ask(Q.algebraic(I/3)) is True

    assert ask(Q.algebraic(sqrt(7))) is True
    assert ask(Q.algebraic(2*sqrt(7))) is True
    assert ask(Q.algebraic(sqrt(7)/3)) is True

    assert ask(Q.algebraic(I*sqrt(3))) is True
    assert ask(Q.algebraic(sqrt(1 + I*sqrt(3)))) is True

    assert ask(Q.algebraic(1 + I*sqrt(3)**Rational(17, 31))) is True
    # 检查表达式是否是代数的，应该返回 False
    assert ask(Q.algebraic(1 + I*sqrt(3)**(17/pi))) is False
    
    # 对于每个数学函数 f，检查其应用于参数 7 的结果是否是代数的，应该返回 False
    for f in [exp, sin, tan, asin, atan, cos]:
        assert ask(Q.algebraic(f(7))) is False
        # 检查 f(7, evaluate=False) 是否是代数的，应该返回 False
        assert ask(Q.algebraic(f(7, evaluate=False))) is False
        # 检查 f(0, evaluate=False) 是否是代数的，应该返回 True
        assert ask(Q.algebraic(f(0, evaluate=False))) is True
        # 检查 f(x) 是否是代数的，但不提供 x 的特定属性，应该返回 None
        assert ask(Q.algebraic(f(x)), Q.algebraic(x)) is None
        # 检查 f(x) 是否是代数的，并且 x 非零，应该返回 False
        assert ask(Q.algebraic(f(x)), Q.algebraic(x) & Q.nonzero(x)) is False
    
    # 对于每个数学函数 g，检查其应用于参数 7 的结果是否是代数的，应该返回 False
    for g in [log, acos]:
        assert ask(Q.algebraic(g(7))) is False
        # 检查 g(7, evaluate=False) 是否是代数的，应该返回 False
        assert ask(Q.algebraic(g(7, evaluate=False))) is False
        # 检查 g(1, evaluate=False) 是否是代数的，应该返回 True
        assert ask(Q.algebraic(g(1, evaluate=False))) is True
        # 检查 g(x) 是否是代数的，但不提供 x 的特定属性，应该返回 None
        assert ask(Q.algebraic(g(x)), Q.algebraic(x)) is None
        # 检查 g(x) 是否是代数的，并且 x - 1 非零，应该返回 False
        assert ask(Q.algebraic(g(x)), Q.algebraic(x) & Q.nonzero(x - 1)) is False
    
    # 对于每个数学函数 h，检查其应用于参数 7 的结果是否是代数的，应该返回 False
    for h in [cot, acot]:
        assert ask(Q.algebraic(h(7))) is False
        # 检查 h(7, evaluate=False) 是否是代数的，应该返回 False
        assert ask(Q.algebraic(h(7, evaluate=False))) is False
        # 检查 h(x) 是否是代数的，应该返回 False
        assert ask(Q.algebraic(h(x)), Q.algebraic(x)) is False
    
    # 检查 sqrt(sin(7)) 是否是代数的，应该返回 False
    assert ask(Q.algebraic(sqrt(sin(7)))) is False
    # 检查 sqrt(y + I*sqrt(7)) 是否是代数的，没有足够信息来确定，应返回 None
    assert ask(Q.algebraic(sqrt(y + I*sqrt(7)))) is None
    
    # 检查 2.47 是否是代数的，应该返回 True
    assert ask(Q.algebraic(2.47)) is True
    
    # 检查 Q.transcendental(x) 是否是 Q.algebraic(x) 的补集，应该返回 False
    assert ask(Q.algebraic(x), Q.transcendental(x)) is False
    # 检查 Q.algebraic(x) 是否是 Q.transcendental(x) 的补集，应该返回 False
    assert ask(Q.transcendental(x), Q.algebraic(x)) is False
# 测试全局假设条件的函数
def test_global():
    # 断言没有关于 x 的整数假设
    assert ask(Q.integer(x)) is None
    # 向全局假设添加关于 x 是整数的假设
    global_assumptions.add(Q.integer(x))
    # 断言现在 x 是整数
    assert ask(Q.integer(x)) is True
    # 清空全局假设
    global_assumptions.clear()
    # 再次断言没有关于 x 的整数假设
    assert ask(Q.integer(x)) is None


# 测试自定义假设上下文的函数
def test_custom_context():
    # 断言没有关于 x 的整数假设
    assert ask(Q.integer(x)) is None
    # 创建一个本地的假设上下文
    local_context = AssumptionsContext()
    # 向本地假设上下文添加关于 x 是整数的假设
    local_context.add(Q.integer(x))
    # 在本地假设上下文中询问 x 是否是整数，应该为 True
    assert ask(Q.integer(x), context=local_context) is True
    # 在全局假设上下文中再次询问，应该为 None
    assert ask(Q.integer(x)) is None


# 测试假设中函数的函数
def test_functions_in_assumptions():
    # 测试负数的 x 是否为实数且正数，应为 False
    assert ask(Q.negative(x), Q.real(x) >> Q.positive(x)) is False
    # 测试负数的 x 是否等价于 x 是实数且正数，应为 False
    assert ask(Q.negative(x), Equivalent(Q.real(x), Q.positive(x))) is False
    # 测试负数的 x 是否异或 x 是实数，应为 False
    assert ask(Q.negative(x), Xor(Q.real(x), Q.negative(x))) is False


# 测试复合询问的函数
def test_composite_ask():
    # 断言 x 是负整数，应用实数推出正数的假设，应为 False
    assert ask(Q.negative(x) & Q.integer(x),
        assumptions=Q.real(x) >> Q.positive(x)) is False


# 测试复合命题的函数
def test_composite_proposition():
    # 断言 True，应为 True
    assert ask(True) is True
    # 断言 False，应为 False
    assert ask(False) is False
    # 断言 x 不是负数且是正数，应为 True
    assert ask(~Q.negative(x), Q.positive(x)) is True
    # 断言 x 不是实数且是可交换的，应为 None
    assert ask(~Q.real(x), Q.commutative(x)) is None
    # 断言 x 是负整数且是整数，应为 False
    assert ask(Q.negative(x) & Q.integer(x), Q.positive(x)) is False
    # 断言 x 是负整数且是整数，应为 None
    assert ask(Q.negative(x) & Q.integer(x)) is None
    # 断言 x 是实数或者是整数，应为 True
    assert ask(Q.real(x) | Q.integer(x), Q.positive(x)) is True
    # 断言 x 是实数或者是整数，应为 None
    assert ask(Q.real(x) | Q.integer(x)) is None
    # 断言 x 是实数推出是正数，且 x 是负数，应为 False
    assert ask(Q.real(x) >> Q.positive(x), Q.negative(x)) is False
    # 断言当实数推出正数（不评估）时，且 x 是负数，应为 False
    assert ask(Implies(
        Q.real(x), Q.positive(x), evaluate=False), Q.negative(x)) is False
    # 断言当实数推出正数（不评估）时，应为 None
    assert ask(Implies(Q.real(x), Q.positive(x), evaluate=False)) is None
    # 断言整数 x 等价于 x 是偶数，且 x 是偶数，应为 True
    assert ask(Equivalent(Q.integer(x), Q.even(x)), Q.even(x)) is True
    # 断言整数 x 等价于 x 是偶数，应为 None
    assert ask(Equivalent(Q.integer(x), Q.even(x))) is None
    # 断言正数 x 等价于 x 是整数，且 x 是整数，应为 None
    assert ask(Equivalent(Q.positive(x), Q.integer(x)), Q.integer(x)) is None
    # 断言 x 是实数或者是整数，且 x 是实数或者是整数，应为 True
    assert ask(Q.real(x) | Q.integer(x), Q.real(x) | Q.integer(x)) is True


# 测试重言式的函数
def test_tautology():
    # 断言 x 是实数或者不是实数，应为 True
    assert ask(Q.real(x) | ~Q.real(x)) is True
    # 断言 x 是实数且不是实数，应为 False
    assert ask(Q.real(x) & ~Q.real(x)) is False


# 测试复合假设的函数
def test_composite_assumptions():
    # 断言 x 是实数，且 x 是实数且 y 也是实数，应为 True
    assert ask(Q.real(x), Q.real(x) & Q.real(y)) is True
    # 断言 x 是正数，且 x 是正数或者 y 是正数，应为 None
    assert ask(Q.positive(x), Q.positive(x) | Q.positive(y)) is None
    # 断言 x 是正数，且 x 是实数推出 y 是正数，应为 None
    assert ask(Q.positive(x), Q.real(x) >> Q.positive(y)) is None
    # 断言 x 是实数，且不是（x 是实数推出 y 是实数），应为 True
    assert ask(Q.real(x), ~(Q.real(x) >> Q.real(y))) is True


# 测试键的可扩展性的函数
def test_key_extensibility():
    """test that you can add keys to the ask system at runtime"""
    # 确保键未定义
    raises(AttributeError, lambda: ask(Q.my_key(x)))

    # 旧的处理程序系统
    class MyAskHandler(AskHandler):
        @staticmethod
        def Symbol(expr, assumptions):
            return True
    try:
        # 使用警告：即将弃用 sympy()
        with warns_deprecated_sympy():
            # 注册 'my_key' 使用 MyAskHandler 处理程序
            register_handler('my_key', MyAskHandler)
        with warns_deprecated_sympy():
            # 询问 Q.my_key(x)，应为 True
            assert ask(Q.my_key(x)) is True
        with warns_deprecated_sympy():
            # 询问 Q.my_key(x + 1)，应为 None
            assert ask(Q.my_key(x + 1)) is None
    ```
    # 最终执行块，处理异常时的清理工作
    finally:
        # 在这里我们需要禁用栈级别的测试，因为这会导致从两个不同的地方触发警告
        # 使用 warns_deprecated_sympy() 上下文管理器来处理过时警告
        with warns_deprecated_sympy():
            # 移除名为 'my_key' 的处理器 MyAskHandler
            remove_handler('my_key', MyAskHandler)
        # 删除 Q.my_key 属性
        del Q.my_key

    # 断言 Q.my_key(x) 引发 AttributeError 异常
    raises(AttributeError, lambda: ask(Q.my_key(x)))

    # 新的处理器系统
    class MyPredicate(Predicate):
        pass
    
    try:
        # 将 Q.my_key 设置为 MyPredicate 的实例
        Q.my_key = MyPredicate()
        # 为 Q.my_key 注册 Symbol 类型的处理函数
        @Q.my_key.register(Symbol)
        def _(expr, assumptions):
            return True
        # 断言调用 ask(Q.my_key(x)) 返回 True
        assert ask(Q.my_key(x)) is True
        # 断言调用 ask(Q.my_key(x+1)) 返回 None
        assert ask(Q.my_key(x+1)) is None
    finally:
        # 最终执行块，清理 Q.my_key 属性
        del Q.my_key
    
    # 断言 Q.my_key(x) 引发 AttributeError 异常
    raises(AttributeError, lambda: ask(Q.my_key(x)))
def test_type_extensibility():
    """测试能否在运行时向询问系统添加新类型"""
    from sympy.core import Basic  # 导入 SymPy 的 Basic 类

    class MyType(Basic):  # 定义一个继承自 Basic 的新类型 MyType
        pass

    @Q.prime.register(MyType)
    def _(expr, assumptions):  # 注册 MyType 类型的 Q.prime 方法
        return True  # 返回 True

    assert ask(Q.prime(MyType())) is True  # 断言询问 MyType 类型的 Q.prime 结果为 True


def test_single_fact_lookup():
    known_facts = And(Implies(Q.integer, Q.rational),  # 定义已知事实
                      Implies(Q.rational, Q.real),
                      Implies(Q.real, Q.complex))
    known_facts_keys = {Q.integer, Q.rational, Q.real, Q.complex}  # 创建已知事实的键集合

    known_facts_cnf = to_cnf(known_facts)  # 转换已知事实为合取范式
    mapping = single_fact_lookup(known_facts_keys, known_facts_cnf)  # 进行单一事实查找

    assert mapping[Q.rational] == {Q.real, Q.rational, Q.complex}  # 断言 Q.rational 的映射结果


def test_generate_known_facts_dict():
    known_facts = And(Implies(Q.integer(x), Q.rational(x)),  # 定义已知事实
                      Implies(Q.rational(x), Q.real(x)),
                      Implies(Q.real(x), Q.complex(x)))
    known_facts_keys = {Q.integer(x), Q.rational(x), Q.real(x), Q.complex(x)}  # 创建已知事实的键集合

    assert generate_known_facts_dict(known_facts_keys, known_facts) == \
        {Q.complex: ({Q.complex}, set()),  # 断言生成的已知事实字典
         Q.integer: ({Q.complex, Q.integer, Q.rational, Q.real}, set()),
         Q.rational: ({Q.complex, Q.rational, Q.real}, set()),
         Q.real: ({Q.complex, Q.real}, set())}


@slow
def test_known_facts_consistent():
    """测试 ask_generated.py 是否为最新版本"""
    x = Symbol('x')  # 创建符号变量 x
    fact = get_known_facts(x)  # 获取关于 x 的已知事实
    cnf = CNF.to_CNF(fact)  # 将事实转换为合取范式
    clauses = set()
    clauses.update(frozenset(Literal(lit.arg.function, lit.is_Not) for lit in sorted(cl, key=str)) for cl in cnf.clauses)  # 更新 CNF 子句集合
    assert get_all_known_facts() == clauses  # 断言所有已知事实与子句相同
    keys = [pred(x) for pred in get_known_facts_keys()]  # 获取已知事实的键列表
    mapping = generate_known_facts_dict(keys, fact)  # 生成已知事实的字典映射
    assert get_known_facts_dict() == mapping  # 断言已知事实字典与生成的映射相同


def test_Add_queries():
    assert ask(Q.prime(12345678901234567890 + (cos(1)**2 + sin(1)**2))) is True  # 断言询问是否为质数
    assert ask(Q.even(Add(S(2), S(2), evaluate=0))) is True  # 断言询问是否为偶数
    assert ask(Q.prime(Add(S(2), S(2), evaluate=0))) is False  # 断言询问是否为质数
    assert ask(Q.integer(Add(S(2), S(2), evaluate=0))) is True  # 断言询问是否为整数


def test_positive_assuming():
    with assuming(Q.positive(x + 1)):  # 假设 x+1 是正数
        assert not ask(Q.positive(x))  # 断言询问 x 是否为正数为 False


def test_issue_5421():
    raises(TypeError, lambda: ask(pi/log(x), Q.real))  # 断言询问 pi/log(x) 是否为实数时引发 TypeError


def test_issue_3906():
    raises(TypeError, lambda: ask(Q.positive))  # 断言询问 Q.positive 是否引发 TypeError


def test_issue_5833():
    assert ask(Q.positive(log(x)**2), Q.positive(x)) is None  # 断言询问 log(x)**2 是否为正数 x 时结果为 None
    assert ask(~Q.negative(log(x)**2), Q.positive(x)) is True  # 断言询问 log(x)**2 是否为非负数 x  时结果为 True


def test_issue_6732():
    raises(ValueError, lambda: ask(Q.positive(x), Q.positive(x) & Q.negative(x)))  # 断言询问 Q.positive(x) 和 Q.negative(x) 的组合是否引发 ValueError
    raises(ValueError, lambda: ask(Q.negative(x), Q.positive(x) & Q.negative(x)))  # 断言询问 Q.negative(x) 和 Q.positive(x) 的组合是否引发 ValueError


def test_issue_7246():
    assert ask(Q.positive(atan(p)), Q.positive(p)) is True  # 断言询问 atan(p) 是否为正数 p 时结果为 True
    assert ask(Q.positive(atan(p)), Q.negative(p)) is False  # 断言询问 atan(p) 是否为负数 p 时结果为 False
    # 断言是否满足 Q.positive(atan(p)) 且 Q.zero(p) 的条件为 False
    assert ask(Q.positive(atan(p)), Q.zero(p)) is False
    # 断言是否满足 Q.positive(atan(x)) 的条件为 None
    assert ask(Q.positive(atan(x))) is None

    # 断言是否满足 Q.positive(asin(p)) 且 Q.positive(p) 的条件为 None
    assert ask(Q.positive(asin(p)), Q.positive(p)) is None
    # 断言是否满足 Q.positive(asin(p)) 且 Q.zero(p) 的条件为 None
    assert ask(Q.positive(asin(p)), Q.zero(p)) is None
    # 断言是否满足 Q.positive(asin(Rational(1, 7))) 的条件为 True
    assert ask(Q.positive(asin(Rational(1, 7)))) is True
    # 断言是否满足 Q.positive(asin(x)) 且 Q.positive(x) & Q.nonpositive(x - 1) 的条件为 True
    assert ask(Q.positive(asin(x)), Q.positive(x) & Q.nonpositive(x - 1)) is True
    # 断言是否满足 Q.positive(asin(x)) 且 Q.negative(x) & Q.nonnegative(x + 1) 的条件为 False
    assert ask(Q.positive(asin(x)), Q.negative(x) & Q.nonnegative(x + 1)) is False

    # 断言是否满足 Q.positive(acos(p)) 且 Q.positive(p) 的条件为 None
    assert ask(Q.positive(acos(p)), Q.positive(p)) is None
    # 断言是否满足 Q.positive(acos(Rational(1, 7))) 的条件为 True
    assert ask(Q.positive(acos(Rational(1, 7)))) is True
    # 断言是否满足 Q.positive(acos(x)) 且 Q.nonnegative(x + 1) & Q.nonpositive(x - 1) 的条件为 True
    assert ask(Q.positive(acos(x)), Q.nonnegative(x + 1) & Q.nonpositive(x - 1)) is True
    # 断言是否满足 Q.positive(acos(x)) 且 Q.nonnegative(x - 1) 的条件为 None
    assert ask(Q.positive(acos(x)), Q.nonnegative(x - 1)) is None

    # 断言是否满足 Q.positive(acot(x)) 且 Q.positive(x) 的条件为 True
    assert ask(Q.positive(acot(x)), Q.positive(x)) is True
    # 断言是否满足 Q.positive(acot(x)) 且 Q.real(x) 的条件为 True
    assert ask(Q.positive(acot(x)), Q.real(x)) is True
    # 断言是否满足 Q.positive(acot(x)) 且 Q.imaginary(x) 的条件为 False
    assert ask(Q.positive(acot(x)), Q.imaginary(x)) is False
    # 断言是否满足 Q.positive(acot(x)) 的条件为 None
    assert ask(Q.positive(acot(x))) is None
@XFAIL
# 标记该测试为预期失败的测试

def test_issue_7246_failing():
    # 将此测试移到 test_issue_7246 一旦新的假设模块得到改进
    # 使用问询函数询问 acos(x) 是否在 x 等于 0 时为正
    assert ask(Q.positive(acos(x)), Q.zero(x)) is True


def test_check_old_assumption():
    # 测试各种数学假设

    x = symbols('x', real=True)
    # 检查 x 是否为实数
    assert ask(Q.real(x)) is True
    # 检查 x 是否为虚数
    assert ask(Q.imaginary(x)) is False
    # 检查 x 是否为复数
    assert ask(Q.complex(x)) is True

    x = symbols('x', imaginary=True)
    # 检查 x 是否为实数
    assert ask(Q.real(x)) is False
    # 检查 x 是否为虚数
    assert ask(Q.imaginary(x)) is True
    # 检查 x 是否为复数
    assert ask(Q.complex(x)) is True

    x = symbols('x', complex=True)
    # 检查 x 是否为实数（返回 None 表示无法确定）
    assert ask(Q.real(x)) is None
    # 检查 x 是否为复数
    assert ask(Q.complex(x)) is True

    x = symbols('x', positive=True)
    # 检查 x 是否为正数
    assert ask(Q.positive(x)) is True
    # 检查 x 是否为负数
    assert ask(Q.negative(x)) is False
    # 检查 x 是否为实数
    assert ask(Q.real(x)) is True

    x = symbols('x', commutative=False)
    # 检查 x 是否为可交换的（非交换）
    assert ask(Q.commutative(x)) is False

    x = symbols('x', negative=True)
    # 检查 x 是否为正数
    assert ask(Q.positive(x)) is False
    # 检查 x 是否为负数
    assert ask(Q.negative(x)) is True

    x = symbols('x', nonnegative=True)
    # 检查 x 是否为负数
    assert ask(Q.negative(x)) is False
    # 检查 x 是否为正数（返回 None 表示无法确定）
    assert ask(Q.positive(x)) is None
    # 检查 x 是否为零（返回 None 表示无法确定）
    assert ask(Q.zero(x)) is None

    x = symbols('x', finite=True)
    # 检查 x 是否为有限数
    assert ask(Q.finite(x)) is True

    x = symbols('x', prime=True)
    # 检查 x 是否为质数
    assert ask(Q.prime(x)) is True
    # 检查 x 是否为合数
    assert ask(Q.composite(x)) is False

    x = symbols('x', composite=True)
    # 检查 x 是否为质数
    assert ask(Q.prime(x)) is False
    # 检查 x 是否为合数
    assert ask(Q.composite(x)) is True

    x = symbols('x', even=True)
    # 检查 x 是否为偶数
    assert ask(Q.even(x)) is True
    # 检查 x 是否为奇数
    assert ask(Q.odd(x)) is False

    x = symbols('x', odd=True)
    # 检查 x 是否为偶数
    assert ask(Q.even(x)) is False
    # 检查 x 是否为奇数
    assert ask(Q.odd(x)) is True

    x = symbols('x', nonzero=True)
    # 检查 x 是否为非零数
    assert ask(Q.nonzero(x)) is True
    # 检查 x 是否为零
    assert ask(Q.zero(x)) is False

    x = symbols('x', zero=True)
    # 检查 x 是否为零
    assert ask(Q.zero(x)) is True

    x = symbols('x', integer=True)
    # 检查 x 是否为整数
    assert ask(Q.integer(x)) is True

    x = symbols('x', rational=True)
    # 检查 x 是否为有理数
    assert ask(Q.rational(x)) is True
    # 检查 x 是否为无理数
    assert ask(Q.irrational(x)) is False

    x = symbols('x', irrational=True)
    # 检查 x 是否为无理数
    assert ask(Q.irrational(x)) is True
    # 检查 x 是否为有理数
    assert ask(Q.rational(x)) is False


def test_issue_9636():
    # 检查浮点数的数学属性（均返回 None 表示无法确定）
    assert ask(Q.integer(1.0)) is None
    assert ask(Q.prime(3.0)) is None
    assert ask(Q.composite(4.0)) is None
    assert ask(Q.even(2.0)) is None
    assert ask(Q.odd(3.0)) is None


def test_autosimp_used_to_fail():
    # 见问题 #9807
    # 检查复数幂次方的实部和虚部（均返回 None 表示无法确定）
    assert ask(Q.imaginary(0**I)) is None
    assert ask(Q.imaginary(0**(-I))) is None
    assert ask(Q.real(0**I)) is None
    assert ask(Q.real(0**(-I))) is None


def test_custom_AskHandler():
    from sympy.logic.boolalg import conjuncts

    # 旧的处理程序系统
    class MersenneHandler(AskHandler):
        @staticmethod
        def Integer(expr, assumptions):
            # 如果 log(expr + 1, 2) 是整数，则返回 True
            if ask(Q.integer(log(expr + 1, 2))):
                return True
        @staticmethod
        def Symbol(expr, assumptions):
            # 如果表达式在假设的合取列表中，则返回 True
            if expr in conjuncts(assumptions):
                return True
    # 使用 warns_deprecated_sympy 上下文管理器注册 'mersenne' 处理程序为 MersenneHandler
    with warns_deprecated_sympy():
        register_handler('mersenne', MersenneHandler)

    # 创建一个整数符号 n
    n = Symbol('n', integer=True)

    # 在 warns_deprecated_sympy 上下文中检查 Q.mersenne(7) 的真假，并断言其为真
    with warns_deprecated_sympy():
        assert ask(Q.mersenne(7))

    # 在 warns_deprecated_sympy 上下文中检查 Q.mersenne(n) 的真假，并断言其为真
    with warns_deprecated_sympy():
        assert ask(Q.mersenne(n), Q.mersenne(n))

    # 删除 Q.mersenne，进入 finally 块
    finally:
        del Q.mersenne

    # 新的处理程序系统
    class MersennePredicate(Predicate):
        pass

    # 将 MersennePredicate 实例赋值给 Q.mersenne
    try:
        Q.mersenne = MersennePredicate()

        # 注册整数类型的处理函数，检查表达式和假设，如果满足条件返回 True
        @Q.mersenne.register(Integer)
        def _(expr, assumptions):
            if ask(Q.integer(log(expr + 1, 2))):
                return True

        # 注册符号类型的处理函数，检查表达式是否在假设的逻辑连词中，如果满足条件返回 True
        @Q.mersenne.register(Symbol)
        def _(expr, assumptions):
            if expr in conjuncts(assumptions):
                return True

        # 在 asserts 中使用 Q.mersenne(7) 进行询问，并断言其为真
        assert ask(Q.mersenne(7))

        # 在 asserts 中使用 Q.mersenne(n) 进行询问，并断言其为真
        assert ask(Q.mersenne(n), Q.mersenne(n))

    # 最终处理，删除 Q.mersenne
    finally:
        del Q.mersenne
# 定义测试函数，用于测试多元谓词功能
def test_polyadic_predicate():

    # 定义一个继承自 Predicate 的 SexyPredicate 类
    class SexyPredicate(Predicate):
        pass
    
    # 尝试设置 Q 对象的 sexyprime 属性为 SexyPredicate 的实例
    try:
        Q.sexyprime = SexyPredicate()

        # 注册接受两个 Integer 类型参数的谓词函数
        @Q.sexyprime.register(Integer, Integer)
        def _(int1, int2, assumptions):
            # 将参数排序
            args = sorted([int1, int2])
            # 如果所有参数都是质数（根据 assumptions），且它们之差为 6，则返回 True
            if not all(ask(Q.prime(a), assumptions) for a in args):
                return False
            return args[1] - args[0] == 6

        # 注册接受三个 Integer 类型参数的谓词函数
        @Q.sexyprime.register(Integer, Integer, Integer)
        def _(int1, int2, int3, assumptions):
            # 将参数排序
            args = sorted([int1, int2, int3])
            # 如果所有参数都是质数（根据 assumptions），且第二个参数与第一个参数之差为 6，第三个参数与第二个参数之差也为 6，则返回 True
            if not all(ask(Q.prime(a), assumptions) for a in args):
                return False
            return args[2] - args[1] == 6 and args[1] - args[0] == 6

        # 断言验证 Q.sexyprime(5, 11) 返回 True
        assert ask(Q.sexyprime(5, 11))
        # 断言验证 Q.sexyprime(7, 13, 19) 返回 True
        assert ask(Q.sexyprime(7, 13, 19))
    
    # 最终清除 Q 对象的 sexyprime 属性
    finally:
        del Q.sexyprime


# 定义测试函数，用于验证 Predicate 类的 handler 属性是否唯一
def test_Predicate_handler_is_unique():

    # 断言未定义的谓词 'mypredicate' 的 handler 为 None
    assert Predicate('mypredicate').handler is None

    # 定义一个继承自 Predicate 的 MyPredicate 类
    # 断言两个不同实例 mp1 和 mp2 的 handler 属性相同
    class MyPredicate(Predicate):
        pass
    mp1 = MyPredicate(Str('mp1'))
    mp2 = MyPredicate(Str('mp2'))
    assert mp1.handler is mp2.handler


# 定义测试函数，用于验证关系判断的功能
def test_relational():
    # 断言当 x 等于 0 时，Q.zero(x) 返回 True，Q.eq(x, 0) 也返回 True
    assert ask(Q.eq(x, 0), Q.zero(x))
    # 断言当 x 等于 0 时，Q.nonzero(x) 返回 False，Q.eq(x, 0) 返回 True
    assert not ask(Q.eq(x, 0), Q.nonzero(x))
    # 断言当 x 不等于 0 时，Q.zero(x) 返回 False，Q.ne(x, 0) 返回 True
    assert not ask(Q.ne(x, 0), Q.zero(x))
    # 断言当 x 不等于 0 时，Q.nonzero(x) 返回 True，Q.ne(x, 0) 返回 True
    assert ask(Q.ne(x, 0), Q.nonzero(x))


# 定义测试函数，用于验证问题 #25221 的特定情况
def test_issue_25221():
    # 断言 Q.transcendental(x) 与 (Q.algebraic(x) 或 Q.positive(y, y)) 的或逻辑结果为 None
    assert ask(Q.transcendental(x), Q.algebraic(x) | Q.positive(y, y)) is None
    # 断言 Q.transcendental(x) 与 (Q.algebraic(x) 或 (0 > y)) 的或逻辑结果为 None
    assert ask(Q.transcendental(x), Q.algebraic(x) | (0 > y)) is None
    # 断言 Q.transcendental(x) 与 (Q.algebraic(x) 或 Q.gt(0, y)) 的或逻辑结果为 None
    assert ask(Q.transcendental(x), Q.algebraic(x) | Q.gt(0, y)) is None
```