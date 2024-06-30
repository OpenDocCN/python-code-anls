# `D:\src\scipysrc\sympy\sympy\core\tests\test_assumptions.py`

```
# 从 sympy.core.mod 模块中导入 Mod 类
from sympy.core.mod import Mod
# 从 sympy.core.numbers 模块中导入 I, oo, pi 三个常数
from sympy.core.numbers import (I, oo, pi)
# 从 sympy.functions.combinatorial.factorials 模块中导入 factorial 函数
from sympy.functions.combinatorial.factorials import factorial
# 从 sympy.functions.elementary.exponential 模块中导入 exp, log 函数
from sympy.functions.elementary.exponential import (exp, log)
# 从 sympy.functions.elementary.miscellaneous 模块中导入 sqrt 函数
from sympy.functions.elementary.miscellaneous import sqrt
# 从 sympy.functions.elementary.trigonometric 模块中导入 asin, sin 函数
from sympy.functions.elementary.trigonometric import (asin, sin)
# 从 sympy.simplify.simplify 模块中导入 simplify 函数
from sympy.simplify.simplify import simplify
# 从 sympy.core 模块中导入 Symbol, S, Rational, Integer, Dummy, Wild, Pow 等类或函数
from sympy.core import Symbol, S, Rational, Integer, Dummy, Wild, Pow
# 从 sympy.core.assumptions 模块中导入多个相关函数和类
from sympy.core.assumptions import (assumptions, check_assumptions,
    failing_assumptions, common_assumptions, _generate_assumption_rules,
    _load_pre_generated_assumption_rules)
# 从 sympy.core.facts 模块中导入 InconsistentAssumptions 异常类
from sympy.core.facts import InconsistentAssumptions
# 从 sympy.core.random 模块中导入 seed 函数
from sympy.core.random import seed
# 从 sympy.combinatorics 模块中导入 Permutation 类
from sympy.combinatorics import Permutation
# 从 sympy.combinatorics.perm_groups 模块中导入 PermutationGroup 类
from sympy.combinatorics.perm_groups import PermutationGroup

# 从 sympy.testing.pytest 模块中导入 raises, XFAIL 函数或装饰器
from sympy.testing.pytest import raises, XFAIL


# 定义测试函数 test_symbol_unset
def test_symbol_unset():
    # 创建一个名为 x 的符号对象，设置其为实数和整数类型
    x = Symbol('x', real=True, integer=True)
    # 断言 x 是否为实数
    assert x.is_real is True
    # 断言 x 是否为整数
    assert x.is_integer is True
    # 断言 x 是否为虚数
    assert x.is_imaginary is False
    # 断言 x 是否为非整数
    assert x.is_noninteger is False
    # 断言 x 是否为数字
    assert x.is_number is False


# 定义测试函数 test_zero
def test_zero():
    # 创建一个值为 0 的整数对象 z
    z = Integer(0)
    # 断言 z 是否为可交换的
    assert z.is_commutative is True
    # 断言 z 是否为整数
    assert z.is_integer is True
    # 断言 z 是否为有理数
    assert z.is_rational is True
    # 断言 z 是否为代数数
    assert z.is_algebraic is True
    # 断言 z 是否为超越数
    assert z.is_transcendental is False
    # 断言 z 是否为实数
    assert z.is_real is True
    # 断言 z 是否为复数
    assert z.is_complex is True
    # 断言 z 是否为非整数
    assert z.is_noninteger is False
    # 断言 z 是否为无理数
    assert z.is_irrational is False
    # 断言 z 是否为虚数
    assert z.is_imaginary is False
    # 断言 z 是否为正数
    assert z.is_positive is False
    # 断言 z 是否为负数
    assert z.is_negative is False
    # 断言 z 是否为非正数
    assert z.is_nonpositive is True
    # 断言 z 是否为非负数
    assert z.is_nonnegative is True
    # 断言 z 是否为偶数
    assert z.is_even is True
    # 断言 z 是否为奇数
    assert z.is_odd is False
    # 断言 z 是否为有限数
    assert z.is_finite is True
    # 断言 z 是否为无限数
    assert z.is_infinite is False
    # 断言 z 是否可比较
    assert z.is_comparable is True
    # 断言 z 是否为质数
    assert z.is_prime is False
    # 断言 z 是否为合数
    assert z.is_composite is False
    # 断言 z 是否为数字
    assert z.is_number is True


# 定义测试函数 test_one
def test_one():
    # 创建一个值为 1 的整数对象 z
    z = Integer(1)
    # 断言 z 是否为可交换的
    assert z.is_commutative is True
    # 断言 z 是否为整数
    assert z.is_integer is True
    # 断言 z 是否为有理数
    assert z.is_rational is True
    # 断言 z 是否为代数数
    assert z.is_algebraic is True
    # 断言 z 是否为超越数
    assert z.is_transcendental is False
    # 断言 z 是否为实数
    assert z.is_real is True
    # 断言 z 是否为复数
    assert z.is_complex is True
    # 断言 z 是否为非整数
    assert z.is_noninteger is False
    # 断言 z 是否为无理数
    assert z.is_irrational is False
    # 断言 z 是否为虚数
    assert z.is_imaginary is False
    # 断言 z 是否为正数
    assert z.is_positive is True
    # 断言 z 是否为负数
    assert z.is_negative is False
    # 断言 z 是否为非正数
    assert z.is_nonpositive is False
    # 断言 z 是否为非负数
    assert z.is_nonnegative is True
    # 断言 z 是否为偶数
    assert z.is_even is False
    # 断言 z 是否为奇数
    assert z.is_odd is True
    # 断言 z 是否为有限数
    assert z.is_finite is True
    # 断言 z 是否为无限数
    assert z.is_infinite is False
    # 断言 z 是否可比较
    assert z.is_comparable is True
    # 断言 z 是否为质数
    assert z.is_prime is False
    # 断言 z 是否为合数
    assert z.is_composite is False  # issue 8807


# 定义测试函数 test_negativeone
def test_negativeone():
    # 创建一个值为 -1 的整数对象 z
    z = Integer(-1)
    # 断言 z 是否为可交换的
    assert z.is_commutative is True
    # 断言 z 是否为整数
    assert z.is_integer is True
    # 断言 z 是否为有理数
    assert z.is_rational is True
    # 断言 z 是否为代数数
    assert z.is_algebraic is True
    # 断言 z 是否为超越数
    assert z.is_transcendental is False
    # 断言 z 是否为实数
    assert z.is_real is True
    # 断言 z 是否为复数
    assert z.is_complex is True
    # 断言 z 是否为非整数
    assert z.is_noninteger is False
    # 断言：验证复数 z 是否为非无理数
    assert z.is_irrational is False
    # 断言：验证复数 z 是否为非虚数
    assert z.is_imaginary is False
    # 断言：验证复数 z 是否为非正数
    assert z.is_positive is False
    # 断言：验证复数 z 是否为负数
    assert z.is_negative is True
    # 断言：验证复数 z 是否为非正数
    assert z.is_nonpositive is True
    # 断言：验证复数 z 是否为非负数
    assert z.is_nonnegative is False
    # 断言：验证复数 z 是否为奇数
    assert z.is_even is False
    # 断言：验证复数 z 是否为偶数
    assert z.is_odd is True
    # 断言：验证复数 z 是否为有限数
    assert z.is_finite is True
    # 断言：验证复数 z 是否为无穷数
    assert z.is_infinite is False
    # 断言：验证复数 z 是否可以比较大小
    assert z.is_comparable is True
    # 断言：验证复数 z 是否为质数
    assert z.is_prime is False
    # 断言：验证复数 z 是否为合数
    assert z.is_composite is False
    # 断言：验证复数 z 是否为数字
    assert z.is_number is True
# 定义测试函数 test_infinity，测试正无穷大 S.Infinity 的属性
def test_infinity():
    # 将正无穷大赋值给变量 oo
    oo = S.Infinity

    # 断言正无穷大是可交换的
    assert oo.is_commutative is True
    # 断言正无穷大不是整数
    assert oo.is_integer is False
    # 断言正无穷大不是有理数
    assert oo.is_rational is False
    # 断言正无穷大不是代数数
    assert oo.is_algebraic is False
    # 断言正无穷大不是超越数
    assert oo.is_transcendental is False
    # 断言正无穷大是扩展实数
    assert oo.is_extended_real is True
    # 断言正无穷大不是实数
    assert oo.is_real is False
    # 断言正无穷大不是复数
    assert oo.is_complex is False
    # 断言正无穷大是非整数
    assert oo.is_noninteger is True
    # 断言正无穷大不是无理数
    assert oo.is_irrational is False
    # 断言正无穷大不是虚数
    assert oo.is_imaginary is False
    # 断言正无穷大是非零
    assert oo.is_nonzero is False
    # 断言正无穷大不是正数
    assert oo.is_positive is False
    # 断言正无穷大不是负数
    assert oo.is_negative is False
    # 断言正无穷大不是非正数
    assert oo.is_nonpositive is False
    # 断言正无穷大不是非负数
    assert oo.is_nonnegative is False
    # 断言正无穷大是扩展非零
    assert oo.is_extended_nonzero is True
    # 断言正无穷大是扩展正数
    assert oo.is_extended_positive is True
    # 断言正无穷大不是扩展负数
    assert oo.is_extended_negative is False
    # 断言正无穷大不是扩展非正数
    assert oo.is_extended_nonpositive is False
    # 断言正无穷大是扩展非负数
    assert oo.is_extended_nonnegative is True
    # 断言正无穷大不是偶数
    assert oo.is_even is False
    # 断言正无穷大不是奇数
    assert oo.is_odd is False
    # 断言正无穷大不是有限数
    assert oo.is_finite is False
    # 断言正无穷大是无穷大
    assert oo.is_infinite is True
    # 断言正无穷大是可比较的
    assert oo.is_comparable is True
    # 断言正无穷大不是素数
    assert oo.is_prime is False
    # 断言正无穷大不是合数
    assert oo.is_composite is False
    # 断言正无穷大是数值
    assert oo.is_number is True


# 定义测试函数 test_neg_infinity，测试负无穷大 S.NegativeInfinity 的属性
def test_neg_infinity():
    # 将负无穷大赋值给变量 mm
    mm = S.NegativeInfinity

    # 断言负无穷大是可交换的
    assert mm.is_commutative is True
    # 断言负无穷大不是整数
    assert mm.is_integer is False
    # 断言负无穷大不是有理数
    assert mm.is_rational is False
    # 断言负无穷大不是代数数
    assert mm.is_algebraic is False
    # 断言负无穷大不是超越数
    assert mm.is_transcendental is False
    # 断言负无穷大是扩展实数
    assert mm.is_extended_real is True
    # 断言负无穷大不是实数
    assert mm.is_real is False
    # 断言负无穷大不是复数
    assert mm.is_complex is False
    # 断言负无穷大是非整数
    assert mm.is_noninteger is True
    # 断言负无穷大不是无理数
    assert mm.is_irrational is False
    # 断言负无穷大不是虚数
    assert mm.is_imaginary is False
    # 断言负无穷大是非零
    assert mm.is_nonzero is False
    # 断言负无穷大不是正数
    assert mm.is_positive is False
    # 断言负无穷大不是负数
    assert mm.is_negative is False
    # 断言负无穷大不是非正数
    assert mm.is_nonpositive is False
    # 断言负无穷大不是非负数
    assert mm.is_nonnegative is False
    # 断言负无穷大是扩展非零
    assert mm.is_extended_nonzero is True
    # 断言负无穷大不是扩展正数
    assert mm.is_extended_positive is False
    # 断言负无穷大是扩展负数
    assert mm.is_extended_negative is True
    # 断言负无穷大是扩展非正数
    assert mm.is_extended_nonpositive is True
    # 断言负无穷大不是扩展非负数
    assert mm.is_extended_nonnegative is False
    # 断言负无穷大不是偶数
    assert mm.is_even is False
    # 断言负无穷大不是奇数
    assert mm.is_odd is False
    # 断言负无穷大不是有限数
    assert mm.is_finite is False
    # 断言负无穷大是无穷大
    assert mm.is_infinite is True
    # 断言负无穷大是可比较的
    assert mm.is_comparable is True
    # 断言负无穷大不是素数
    assert mm.is_prime is False
    # 断言负无穷大不是合数
    assert mm.is_composite is False
    # 断言负无穷大是数值
    assert mm.is_number is True


# 定义测试函数 test_zoo，测试复杂无穷大 S.ComplexInfinity 的属性
def test_zoo():
    # 将复杂无穷大赋值给变量 zoo
    zoo = S.ComplexInfinity
    # 断言复杂无穷大不是复数
    assert zoo.is_complex is False
    # 断言复杂无穷大不是实数
    assert zoo.is_real is False
    # 断言复杂无穷大不是素数
    assert zoo.is_prime is False


# 定义测试函数 test_nan，测试 NaN S.NaN 的属性
def test_nan():
    # 将 NaN 赋值给变量 nan
    nan = S.NaN

    # 断言 NaN 是可交换的
    assert nan.is_commutative is True
    # 断言 NaN 的整数性质为 None
    assert nan.is_integer is None
    # 断言 NaN 的有理数性质为 None
    assert nan.is_rational is None
    # 断言 NaN 的代数性质为 None
    assert nan.is_algebraic is None
    # 断言 NaN 的超越性质为 None
    assert nan.is_transcendental is None
    # 断言 NaN 的实数性质为 None
    assert nan.is_real is None
    # 断言 NaN 的复数性质为 None
    assert nan.is_complex is None
    # 断言 NaN 的非整数性质为 None
    assert nan.is_noninteger is None
    # 断言 NaN 的无理数性质为 None
    assert nan.is_irrational is None
    # 断言 NaN 的虚数性质为 None
    assert nan.is_imaginary is None
    # 断言 NaN 的正数性质为 None
    assert nan.is_positive is None
    # 断言 NaN 的
    # 断言：确保 NaN 对象的有限性为 None
    assert nan.is_finite is None
    # 断言：确保 NaN 对象的无限性为 None
    assert nan.is_infinite is None
    # 断言：确保 NaN 对象不可比较性为 False
    assert nan.is_comparable is False
    # 断言：确保 NaN 对象的质数性质为 None
    assert nan.is_prime is None
    # 断言：确保 NaN 对象的合数性质为 None
    assert nan.is_composite is None
    # 断言：确保 NaN 对象的数值性质为 True
    assert nan.is_number is True
# 测试正有理数 Rational 类的功能

def test_pos_rational():
    # 创建有理数 3/4
    r = Rational(3, 4)
    # 断言这个有理数是可交换的（True）
    assert r.is_commutative is True
    # 断言这个有理数不是整数（False）
    assert r.is_integer is False
    # 断言这个有理数是有理数（True）
    assert r.is_rational is True
    # 断言这个有理数是代数数（True）
    assert r.is_algebraic is True
    # 断言这个有理数不是超越数（False）
    assert r.is_transcendental is False
    # 断言这个有理数是实数（True）
    assert r.is_real is True
    # 断言这个有理数是复数（True）
    assert r.is_complex is True
    # 断言这个有理数是非整数（True）
    assert r.is_noninteger is True
    # 断言这个有理数不是无理数（False）
    assert r.is_irrational is False
    # 断言这个有理数不是虚数（False）
    assert r.is_imaginary is False
    # 断言这个有理数是正数（True）
    assert r.is_positive is True
    # 断言这个有理数不是负数（False）
    assert r.is_negative is False
    # 断言这个有理数不是非正数（False）
    assert r.is_nonpositive is False
    # 断言这个有理数是非负数（True）
    assert r.is_nonnegative is True
    # 断言这个有理数不是偶数（False）
    assert r.is_even is False
    # 断言这个有理数不是奇数（False）
    assert r.is_odd is False
    # 断言这个有理数是有限数（True）
    assert r.is_finite is True
    # 断言这个有理数不是无限数（False）
    assert r.is_infinite is False
    # 断言这个有理数是可比较的（True）
    assert r.is_comparable is True
    # 断言这个有理数不是质数（False）
    assert r.is_prime is False
    # 断言这个有理数不是合数（False）
    assert r.is_composite is False

    # 创建有理数 1/4
    r = Rational(1, 4)
    # 断言这个有理数不是非正数（False）
    assert r.is_nonpositive is False
    # 断言这个有理数是正数（True）
    assert r.is_positive is True
    # 断言这个有理数不是负数（False）
    assert r.is_negative is False
    # 断言这个有理数是非负数（True）
    assert r.is_nonnegative is True

    # 创建有理数 5/4
    r = Rational(5, 4)
    # 断言这个有理数不是负数（False）
    assert r.is_negative is False
    # 断言这个有理数是正数（True）
    assert r.is_positive is True
    # 断言这个有理数不是非正数（False）
    assert r.is_nonpositive is False
    # 断言这个有理数是非负数（True）
    assert r.is_nonnegative is True

    # 创建有理数 5/3
    r = Rational(5, 3)
    # 断言这个有理数是非负数（True）
    assert r.is_nonnegative is True
    # 断言这个有理数是正数（True）
    assert r.is_positive is True
    # 断言这个有理数不是负数（False）
    assert r.is_negative is False
    # 断言这个有理数不是非正数（False）
    assert r.is_nonpositive is False


# 测试负有理数 Rational 类的功能

def test_neg_rational():
    # 创建负有理数 -3/4
    r = Rational(-3, 4)
    # 断言这个有理数不是正数（False）
    assert r.is_positive is False
    # 断言这个有理数是非正数（True）
    assert r.is_nonpositive is True
    # 断言这个有理数是负数（True）
    assert r.is_negative is True
    # 断言这个有理数不是非负数（False）
    assert r.is_nonnegative is False

    # 创建负有理数 -1/4
    r = Rational(-1, 4)
    # 断言这个有理数是非正数（True）
    assert r.is_nonpositive is True
    # 断言这个有理数不是正数（False）
    assert r.is_positive is False
    # 断言这个有理数是负数（True）
    assert r.is_negative is True
    # 断言这个有理数不是非负数（False）
    assert r.is_nonnegative is False

    # 创建负有理数 -5/4
    r = Rational(-5, 4)
    # 断言这个有理数是负数（True）
    assert r.is_negative is True
    # 断言这个有理数不是正数（False）
    assert r.is_positive is False
    # 断言这个有理数是非正数（True）
    assert r.is_nonpositive is True
    # 断言这个有理数不是非负数（False）
    assert r.is_nonnegative is False

    # 创建负有理数 -5/3
    r = Rational(-5, 3)
    # 断言这个有理数不是非负数（False）
    assert r.is_nonnegative is False
    # 断言这个有理数不是正数（False）
    assert r.is_positive is False
    # 断言这个有理数是负数（True）
    assert r.is_negative is True
    # 断言这个有理数是非正数（True）
    assert r.is_nonpositive is True


# 测试 Pi 常数的属性

def test_pi():
    # 获取 Pi 常数
    z = S.Pi
    # 断言这个常数是可交换的（True）
    assert z.is_commutative is True
    # 断言这个常数不是整数（False）
    assert z.is_integer is False
    # 断言这个常数不是有理数（False）
    assert z.is_rational is False
    # 断言这个常数不是代数数（False）
    assert z.is_algebraic is False
    # 断言这个常数是超越数（True）
    assert z.is_transcendental is True
    # 断言这个常数是实数（True）
    assert z.is_real is True
    # 断言这个常数是复数（True）
    assert z.is_complex is True
    # 断言这个常数是非整数（True）
    assert z.is_noninteger is True
    # 断言这个常数是无理数（True）
    assert z.is_irrational is True
    # 断言这个常数不是虚数（False）
    assert z.is_imaginary is False
    # 断言这个常数是正数（True）
    assert z.is_positive is True
    # 断言这个常数不是负数（False）
    assert z.is_negative is False
    # 断言这个常数不是非正数（False）
    assert z.is_nonpositive is False
    # 断言这个常数是非负数（True）
    assert z.is_nonnegative is True
    # 断言这个常数不是偶数（False）
    assert z.is_even is False
    # 断言这个常数不是奇数（False）
    assert z.is_odd is False
    # 断言这个常数是有限数（True）
    assert z.is_finite is True
    # 断言这个常数不是无限数（False）
    assert z.is_infinite is False
    # 断言这个常数是可比较的（True）
    assert z.is_comparable is True
    # 断言这个常数不是质数（False）
    assert z.is_prime is False
    # 断言这个常数不
    # 断言：检查数 z 是否为无理数（irrational）
    assert z.is_irrational is True
    
    # 断言：检查数 z 是否为虚数（imaginary）
    assert z.is_imaginary is False
    
    # 断言：检查数 z 是否为正数（positive）
    assert z.is_positive is True
    
    # 断言：检查数 z 是否为负数（negative）
    assert z.is_negative is False
    
    # 断言：检查数 z 是否为非正数（nonpositive）
    assert z.is_nonpositive is False
    
    # 断言：检查数 z 是否为非负数（nonnegative）
    assert z.is_nonnegative is True
    
    # 断言：检查数 z 是否为偶数（even）
    assert z.is_even is False
    
    # 断言：检查数 z 是否为奇数（odd）
    assert z.is_odd is False
    
    # 断言：检查数 z 是否为有限数（finite）
    assert z.is_finite is True
    
    # 断言：检查数 z 是否为无限数（infinite）
    assert z.is_infinite is False
    
    # 断言：检查数 z 是否可比较（comparable）
    assert z.is_comparable is True
    
    # 断言：检查数 z 是否为质数（prime）
    assert z.is_prime is False
    
    # 断言：检查数 z 是否为合数（composite）
    assert z.is_composite is False
# 定义测试函数 test_I，用于测试 ImaginaryUnit 对象的属性
def test_I():
    # 获取虚数单位对象 S.ImaginaryUnit
    z = S.ImaginaryUnit
    # 断言虚数单位对象具有交换性
    assert z.is_commutative is True
    # 断言虚数单位对象不是整数
    assert z.is_integer is False
    # 断言虚数单位对象不是有理数
    assert z.is_rational is False
    # 断言虚数单位对象是代数数
    assert z.is_algebraic is True
    # 断言虚数单位对象不是超越数
    assert z.is_transcendental is False
    # 断言虚数单位对象不是实数
    assert z.is_real is False
    # 断言虚数单位对象是复数
    assert z.is_complex is True
    # 断言虚数单位对象不是非整数
    assert z.is_noninteger is False
    # 断言虚数单位对象不是无理数
    assert z.is_irrational is False
    # 断言虚数单位对象是虚数
    assert z.is_imaginary is True
    # 断言虚数单位对象不是正数
    assert z.is_positive is False
    # 断言虚数单位对象不是负数
    assert z.is_negative is False
    # 断言虚数单位对象不是非正数
    assert z.is_nonpositive is False
    # 断言虚数单位对象不是非负数
    assert z.is_nonnegative is False
    # 断言虚数单位对象不是偶数
    assert z.is_even is False
    # 断言虚数单位对象不是奇数
    assert z.is_odd is False
    # 断言虚数单位对象是有限数
    assert z.is_finite is True
    # 断言虚数单位对象不是无限数
    assert z.is_infinite is False
    # 断言虚数单位对象不可比较
    assert z.is_comparable is False
    # 断言虚数单位对象不是素数
    assert z.is_prime is False
    # 断言虚数单位对象不是合数
    assert z.is_composite is False


# 定义测试函数 test_symbol_real_false，测试 real=False 的 Symbol 对象属性
def test_symbol_real_false():
    # 创建一个名为 'a'，real=False 的符号对象
    a = Symbol('a', real=False)

    # 断言该符号对象不是实数
    assert a.is_real is False
    # 断言该符号对象不是整数
    assert a.is_integer is False
    # 断言该符号对象不是零
    assert a.is_zero is False

    # 断言该符号对象不是负数
    assert a.is_negative is False
    # 断言该符号对象不是正数
    assert a.is_positive is False
    # 断言该符号对象不是非负数
    assert a.is_nonnegative is False
    # 断言该符号对象不是非正数
    assert a.is_nonpositive is False
    # 断言该符号对象不是非零数
    assert a.is_nonzero is False

    # 断言扩展负数属性为 None
    assert a.is_extended_negative is None
    # 断言扩展正数属性为 None
    assert a.is_extended_positive is None
    # 断言扩展非负数属性为 None
    assert a.is_extended_nonnegative is None
    # 断言扩展非正数属性为 None
    assert a.is_extended_nonpositive is None
    # 断言扩展非零数属性为 None
    assert a.is_extended_nonzero is None


# 定义测试函数 test_symbol_extended_real_false，测试 extended_real=False 的 Symbol 对象属性
def test_symbol_extended_real_false():
    # 创建一个名为 'a'，extended_real=False 的符号对象
    a = Symbol('a', extended_real=False)

    # 断言该符号对象不是实数
    assert a.is_real is False
    # 断言该符号对象不是整数
    assert a.is_integer is False
    # 断言该符号对象不是零
    assert a.is_zero is False

    # 断言该符号对象不是负数
    assert a.is_negative is False
    # 断言该符号对象不是正数
    assert a.is_positive is False
    # 断言该符号对象不是非负数
    assert a.is_nonnegative is False
    # 断言该符号对象不是非正数
    assert a.is_nonpositive is False
    # 断言该符号对象不是非零数
    assert a.is_nonzero is False

    # 断言扩展负数属性为 False
    assert a.is_extended_negative is False
    # 断言扩展正数属性为 False
    assert a.is_extended_positive is False
    # 断言扩展非负数属性为 False
    assert a.is_extended_nonnegative is False
    # 断言扩展非正数属性为 False
    assert a.is_extended_nonpositive is False
    # 断言扩展非零数属性为 False
    assert a.is_extended_nonzero is False


# 定义测试函数 test_symbol_imaginary，测试 imaginary=True 的 Symbol 对象属性
def test_symbol_imaginary():
    # 创建一个名为 'a'，imaginary=True 的符号对象
    a = Symbol('a', imaginary=True)

    # 断言该符号对象不是实数
    assert a.is_real is False
    # 断言该符号对象不是整数
    assert a.is_integer is False
    # 断言该符号对象不是负数
    assert a.is_negative is False
    # 断言该符号对象不是正数
    assert a.is_positive is False
    # 断言该符号对象不是非负数
    assert a.is_nonnegative is False
    # 断言该符号对象不是非正数
    assert a.is_nonpositive is False
    # 断言该符号对象不是零
    assert a.is_zero is False
    # 断言该符号对象不是非零数（因为非零数应为实数）
    assert a.is_nonzero is False


# 定义测试函数 test_symbol_zero，测试 zero=True 的 Symbol 对象属性
def test_symbol_zero():
    # 创建一个名为 'x'，zero=True 的符号对象
    x = Symbol('x', zero=True)

    # 断言该符号对象不是正数
    assert x.is_positive is False
    # 断言该符号对象是非正数
    assert x.is_nonpositive
    # 断言该符号对象不是负数
    assert x.is_negative is False
    # 断言该符号对象是非负数
    assert x.is_nonnegative
    # 断言该符号对象是零
    assert x.is_zero is True
    # 断言该符号对象不是非零数
    assert x.is_nonzero is False
    # 断言该符号对象是有限数
    assert x.is_finite is True


# 定义测试函数 test_symbol_positive，测试 positive=True 的 Symbol 对象属性
def test_symbol_positive():
    # 创建一个名为 'x'，positive=True 的符号对象
    x = Symbol('x', positive=True)

    # 断言该符号对象是正数
    assert x.is_positive is True
    # 断言该符号对象不是非正数
    assert x.is_nonpositive is False
    # 断言该符号对象不是负数
    assert x.is_negative is False
    # 断言该符号对象是非负数
    assert x.is_nonnegative is True
    # 断言该符号对象不是零
    assert x.is_zero is False
    # 断言该符号对象是非零数
    assert x.is_nonzero is True


# 定义测试函数 test_neg_symbol_positive，测试负值为 positive=True 的 Symbol 对象属性
def test_neg_symbol_positive():
    # 创建一个名为 '-x'，positive=True 的符号对象
    x = -Symbol('x', positive=True)
    # 断言 x 的正性为假
    assert x.is_positive is False
    # 断言 x 的非正性为真
    assert x.is_nonpositive is True
    # 断言 x 的负性为真
    assert x.is_negative is True
    # 断言 x 的非负性为假
    assert x.is_nonnegative is False
    # 断言 x 为零的性质为假
    assert x.is_zero is False
    # 断言 x 的非零性为真
    assert x.is_nonzero is True
# 定义一个测试函数，用于测试符号的非正性质
def test_symbol_nonpositive():
    # 创建一个名为 'x' 的符号，并设定其为非正
    x = Symbol('x', nonpositive=True)
    # 断言 'x' 是正数为 False
    assert x.is_positive is False
    # 断言 'x' 是非正数为 True
    assert x.is_nonpositive is True
    # 断言 'x' 是负数为 None
    assert x.is_negative is None
    # 断言 'x' 是非负数为 None
    assert x.is_nonnegative is None
    # 断言 'x' 是零为 None
    assert x.is_zero is None
    # 断言 'x' 是非零数为 None


# 定义一个测试函数，用于测试负号作用于非正符号的情况
def test_neg_symbol_nonpositive():
    # 创建一个名为 'x' 的符号，并设定其为非正
    x = -Symbol('x', nonpositive=True)
    # 断言负 'x' 不是正数为 None
    assert x.is_positive is None
    # 断言负 'x' 不是非正数为 None
    assert x.is_nonpositive is None
    # 断言负 'x' 是负数为 False
    assert x.is_negative is False
    # 断言负 'x' 是非负数为 True
    assert x.is_nonnegative is True
    # 断言负 'x' 是零为 None
    assert x.is_zero is None
    # 断言负 'x' 是非零数为 None


# 定义一个测试函数，用于测试符号的误判为正数的情况
def test_symbol_falsepositive():
    # 创建一个名为 'x' 的符号，并设定其为非正数
    x = Symbol('x', positive=False)
    # 断言 'x' 是正数为 False
    assert x.is_positive is False
    # 断言 'x' 不是非正数为 None
    assert x.is_nonpositive is None
    # 断言 'x' 是负数为 None
    assert x.is_negative is None
    # 断言 'x' 是非负数为 None
    assert x.is_nonnegative is None
    # 断言 'x' 是零为 None
    assert x.is_zero is None
    # 断言 'x' 是非零数为 None


# 定义一个测试函数，用于测试误判为正数的符号乘法
def test_symbol_falsepositive_mul():
    # 用于测试拉取请求 9379
    # Mul._eval_is_positive 中明确处理 arg.is_positive=False 的情况
    x = 2*Symbol('x', positive=False)
    # 断言 2*'x' 是正数为 False，之前是 None
    assert x.is_positive is False
    # 断言 2*'x' 不是非正数为 None
    assert x.is_nonpositive is None
    # 断言 2*'x' 是负数为 None
    assert x.is_negative is None
    # 断言 2*'x' 是非负数为 None
    assert x.is_nonnegative is None
    # 断言 2*'x' 是零为 None
    assert x.is_zero is None
    # 断言 2*'x' 是非零数为 None


@XFAIL
def test_symbol_infinitereal_mul():
    # 测试无限实数符号乘法
    ix = Symbol('ix', infinite=True, extended_real=True)
    # 断言 -'ix' 是扩展正数为 None


# 定义一个测试函数，用于测试负号作用于误判为正数的符号的情况
def test_neg_symbol_falsepositive():
    # 创建一个名为 'x' 的符号，并设定其为非正数
    x = -Symbol('x', positive=False)
    # 断言负 'x' 是正数为 None
    assert x.is_positive is None
    # 断言负 'x' 不是非正数为 None
    assert x.is_nonpositive is None
    # 断言负 'x' 是负数为 False
    assert x.is_negative is False
    # 断言负 'x' 是非负数为 None
    assert x.is_nonnegative is None
    # 断言负 'x' 是零为 None
    assert x.is_zero is None
    # 断言负 'x' 是非零数为 None


# 定义一个测试函数，用于测试负号作用于误判为负数的符号的情况
def test_neg_symbol_falsenegative():
    # 用于测试拉取请求 9379
    # Mul._eval_is_positive 中明确处理 arg.is_negative=False 的情况
    x = -Symbol('x', negative=False)
    # 断言负 'x' 是正数为 False，之前是 None
    assert x.is_positive is False
    # 断言负 'x' 不是非正数为 None
    assert x.is_nonpositive is None
    # 断言负 'x' 是负数为 None
    assert x.is_negative is None
    # 断言负 'x' 是非负数为 None
    assert x.is_nonnegative is None
    # 断言负 'x' 是零为 None
    assert x.is_zero is None
    # 断言负 'x' 是非零数为 None


# 定义一个测试函数，用于测试误判为正数的实数符号的情况
def test_symbol_falsepositive_real():
    # 创建一个名为 'x' 的符号，并设定其为非正数和实数
    x = Symbol('x', positive=False, real=True)
    # 断言 'x' 是正数为 False
    assert x.is_positive is False
    # 断言 'x' 是非正数为 True
    assert x.is_nonpositive is True
    # 断言 'x' 是负数为 None
    assert x.is_negative is None
    # 断言 'x' 是非负数为 None
    assert x.is_nonnegative is None
    # 断言 'x' 是零为 None
    assert x.is_zero is None
    # 断言 'x' 是非零数为 None


# 定义一个测试函数，用于测试负号作用于误判为正数的实数符号的情况
def test_neg_symbol_falsepositive_real():
    # 创建一个名为 'x' 的符号，并设定其为非正数和实数
    x = -Symbol('x', positive=False, real=True)
    # 断言负 'x' 是正数为 None
    assert x.is_positive is None
    # 断言负 'x' 不是非正数为 None
    assert x.is_nonpositive is None
    # 断言负 'x' 是负数为 False
    assert x.is_negative is False
    # 断言负 'x' 是非负数为 True
    assert x.is_nonnegative is True
    # 断言负 'x' 是零为 None
    assert x.is_zero is None
    # 断言负 'x' 是非零数为 None


# 定义一个测试函数，用于测试误判为非非负数的符号的情况
def test_symbol_falsenonnegative():
    # 创建一个名为 'x' 的符号，并设定其为非非负数
    x = Symbol('x', nonnegative=False)
    # 断言 'x' 是正数为 False
    assert x.is_positive is False
    # 断言 'x' 不是非正数为 None
    assert x.is_nonpositive is None
    # 断言 'x' 是负数为 None
    assert x.is_negative is None
    # 断言 'x' 是非负数为 False
    assert x.is_nonnegative is False
    # 断言 'x' 是零为 False
    assert x.is_zero is False
    # 断言 'x' 是非零数为 None


@XFAIL
def test_neg_symbol_falsenonnegative():
    # 测试负号作用于误判为非非负数的符号的情况
    # 创建一个负值的符号变量 'x'，并指定它可以为非负数
    x = -Symbol('x', nonnegative=False)
    
    # 断言：检查符号变量 x 是否为正数，预期结果是 None
    assert x.is_positive is None
    
    # 断言：检查符号变量 x 是否为非正数，预期结果是 False
    assert x.is_nonpositive is False  # 当前返回值为 None
    
    # 断言：检查符号变量 x 是否为负数，预期结果是 False
    assert x.is_negative is False  # 当前返回值为 None
    
    # 断言：检查符号变量 x 是否为非负数，预期结果是 None
    assert x.is_nonnegative is None
    
    # 断言：检查符号变量 x 是否为零，预期结果是 False
    assert x.is_zero is False  # 当前返回值为 None
    
    # 断言：检查符号变量 x 是否为非零，预期结果是 True
    assert x.is_nonzero is True  # 当前返回值为 None
# 定义测试函数，测试带有特定属性的符号的行为
def test_symbol_falsenonnegative_real():
    # 创建一个名为 'x' 的符号，其非负和实数属性为假
    x = Symbol('x', nonnegative=False, real=True)
    # 断言 x 不是正数
    assert x.is_positive is False
    # 断言 x 是非正数
    assert x.is_nonpositive is True
    # 断言 x 是负数
    assert x.is_negative is True
    # 断言 x 不是非负数
    assert x.is_nonnegative is False
    # 断言 x 不是零
    assert x.is_zero is False
    # 断言 x 是非零数
    assert x.is_nonzero is True


# 定义测试函数，测试带有特定属性的负数符号的行为
def test_neg_symbol_falsenonnegative_real():
    # 创建一个名为 'x' 的符号，其非负和实数属性为假，然后取其相反数
    x = -Symbol('x', nonnegative=False, real=True)
    # 断言 x 是正数
    assert x.is_positive is True
    # 断言 x 不是非正数
    assert x.is_nonpositive is False
    # 断言 x 不是负数
    assert x.is_negative is False
    # 断言 x 是非负数
    assert x.is_nonnegative is True
    # 断言 x 不是零
    assert x.is_zero is False
    # 断言 x 是非零数
    assert x.is_nonzero is True


# 定义测试函数，测试 S 类中数字的素数属性
def test_prime():
    # 断言 -1 不是素数
    assert S.NegativeOne.is_prime is False
    # 断言 -2 不是素数
    assert S(-2).is_prime is False
    # 断言 -4 不是素数
    assert S(-4).is_prime is False
    # 断言 0 不是素数
    assert S.Zero.is_prime is False
    # 断言 1 不是素数
    assert S.One.is_prime is False
    # 断言 2 是素数
    assert S(2).is_prime is True
    # 断言 17 是素数
    assert S(17).is_prime is True
    # 断言 4 不是素数
    assert S(4).is_prime is False


# 定义测试函数，测试 S 类中数字的合数属性
def test_composite():
    # 断言 -1 不是合数
    assert S.NegativeOne.is_composite is False
    # 断言 -2 不是合数
    assert S(-2).is_composite is False
    # 断言 -4 不是合数
    assert S(-4).is_composite is False
    # 断言 0 不是合数
    assert S.Zero.is_composite is False
    # 断言 2 不是合数
    assert S(2).is_composite is False
    # 断言 17 不是合数
    assert S(17).is_composite is False
    # 断言 4 是合数
    assert S(4).is_composite is True
    # 创建一个具有整数、正数和非素数属性的虚拟符号 x
    x = Dummy(integer=True, positive=True, prime=False)
    # 断言 x 的合数属性为 None（可能是 1）
    assert x.is_composite is None
    # 断言 (x + 1) 的合数属性为 None
    assert (x + 1).is_composite is None
    # 创建一个具有正数、偶数和非素数属性的虚拟符号 x
    x = Dummy(positive=True, even=True, prime=False)
    # 断言 x 是整数
    assert x.is_integer is True
    # 断言 x 是合数
    assert x.is_composite is True


# 定义测试函数，测试带有素数属性的符号 x 的行为
def test_prime_symbol():
    # 创建一个名为 'x' 的符号，其素数属性为真
    x = Symbol('x', prime=True)
    # 断言 x 是素数
    assert x.is_prime is True
    # 断言 x 是整数
    assert x.is_integer is True
    # 断言 x 是正数
    assert x.is_positive is True
    # 断言 x 不是负数
    assert x.is_negative is False
    # 断言 x 不是非正数
    assert x.is_nonpositive is False
    # 断言 x 是非负数
    assert x.is_nonnegative is True

    # 创建一个名为 'x' 的符号，其素数属性为假
    x = Symbol('x', prime=False)
    # 断言 x 不是素数
    assert x.is_prime is False
    # 断言 x 的整数属性为 None
    assert x.is_integer is None
    # 断言 x 的正数属性为 None
    assert x.is_positive is None
    # 断言 x 的负数属性为 None
    assert x.is_negative is None
    # 断言 x 的非正数属性为 None
    assert x.is_nonpositive is None
    # 断言 x 的非负数属性为 None
    assert x.is_nonnegative is None


# 定义测试函数，测试带有非交换属性的符号 x 的行为
def test_symbol_noncommutative():
    # 创建一个名为 'x' 的符号，其可交换属性为假
    x = Symbol('x', commutative=True)
    # 断言 x 的复数属性为 None

    # 创建一个名为 'x' 的符号，其可交换属性为假
    x = Symbol('x', commutative=False)
    # 断言 x 不是整数
    assert x.is_integer is False
    # 断言 x 不是有理数
    assert x.is_rational is False
    # 断言 x 不是代数数
    assert x.is_algebraic is False
    # 断言 x 不是无理数
    assert x.is_irrational is False
    # 断言 x 不是实数
    assert x.is_real is False
    # 断言 x 不是复数
    assert x.is_complex is False


# 定义测试函数，测试带有其它属性的符号 x 的行为
def test_other_symbol():
    # 创建一个名为 'x' 的符号，其整数属性为真
    x = Symbol('x', integer=True)
    # 断言 x 是整数
    assert x.is_integer is True
    # 断言 x 是实数
    assert x.is_real is True
    # 断言 x 是有限数
    assert x.is_finite is True

    # 创建一个名为 'x' 的符号，其整数和非负属性为真
    x = Symbol('x', integer=True, nonnegative=True)
    # 断言 x 是整数
    assert x.is_integer is True
    # 断言 x 是非负数
    assert x.is_nonnegative is True
    # 断言 x 是负数
    assert x.is_negative is False
    # 断言 x 的正数属性为 None
    assert x.is_positive is None
    # 断言 x 是有限数
    assert x.is_finite is True

    # 创建一个名为 'x' 的符号，其整数和非正属性为真
    x = Symbol('x', integer=True, nonpositive=True)
    # 断言 x 是整数
    assert x.is_integer is True
    # 断言 x 是非正数
    assert x.is_nonpositive is True
    # 断言 x 是负数
    assert x.is_positive is False
    # 断言 x 的负数属性为 None
    assert x.is_negative is None
    # 断言 x 是有限数
    assert x.is_finite is True

    # 创建一个名为 'x' 的符号，其奇数属性为真
    x = Symbol('x', odd=True)
    # 断言 x 是奇数
    assert x.is_odd is True
    # 断言 x 不是偶数
    assert x.is_even is False
    # 断言 x 是否为整数
    assert x.is_integer is True
    # 断言 x 是否为有限数
    assert x.is_finite is True

    # 创建一个名为 x 的符号，并指定其为偶数
    x = Symbol('x', odd=False)
    # 断言 x 是否为奇数
    assert x.is_odd is False
    # 断言 x 是否为偶数
    assert x.is_even is None
    # 断言 x 是否为整数
    assert x.is_integer is None
    # 断言 x 是否为有限数
    assert x.is_finite is None

    # 创建一个名为 x 的符号，并指定其为偶数
    x = Symbol('x', even=True)
    # 断言 x 是否为偶数
    assert x.is_even is True
    # 断言 x 是否为奇数
    assert x.is_odd is False
    # 断言 x 是否为整数
    assert x.is_integer is True
    # 断言 x 是否为有限数
    assert x.is_finite is True

    # 创建一个名为 x 的符号，并指定其为奇数
    x = Symbol('x', even=False)
    # 断言 x 是否为偶数
    assert x.is_even is False
    # 断言 x 是否为奇数
    assert x.is_odd is None
    # 断言 x 是否为整数
    assert x.is_integer is None
    # 断言 x 是否为有限数
    assert x.is_finite is None

    # 创建一个名为 x 的符号，并指定其为整数且非负
    x = Symbol('x', integer=True, nonnegative=True)
    # 断言 x 是否为整数
    assert x.is_integer is True
    # 断言 x 是否为非负数
    assert x.is_nonnegative is True
    # 断言 x 是否为有限数
    assert x.is_finite is True

    # 创建一个名为 x 的符号，并指定其为整数且非正
    x = Symbol('x', integer=True, nonpositive=True)
    # 断言 x 是否为整数
    assert x.is_integer is True
    # 断言 x 是否为非正数
    assert x.is_nonpositive is True
    # 断言 x 是否为有限数
    assert x.is_finite is True

    # 创建一个名为 x 的符号，并指定其为有理数
    x = Symbol('x', rational=True)
    # 断言 x 是否为实数
    assert x.is_real is True
    # 断言 x 是否为有限数
    assert x.is_finite is True

    # 创建一个名为 x 的符号，并指定其为非有理数
    x = Symbol('x', rational=False)
    # 断言 x 是否为实数
    assert x.is_real is None
    # 断言 x 是否为有限数
    assert x.is_finite is None

    # 创建一个名为 x 的符号，并指定其为无理数
    x = Symbol('x', irrational=True)
    # 断言 x 是否为实数
    assert x.is_real is True
    # 断言 x 是否为有限数
    assert x.is_finite is True

    # 创建一个名为 x 的符号，并指定其为非无理数
    x = Symbol('x', irrational=False)
    # 断言 x 是否为实数
    assert x.is_real is None
    # 断言 x 是否为有限数
    assert x.is_finite is None

    # 使用 raises 检查设置 x.is_real 属性是否会引发 AttributeError 异常
    with raises(AttributeError):
        x.is_real = False

    # 创建一个名为 x 的符号，并指定其为代数数
    x = Symbol('x', algebraic=True)
    # 断言 x 是否为超越数
    assert x.is_transcendental is False

    # 创建一个名为 x 的符号，并指定其为超越数
    x = Symbol('x', transcendental=True)
    # 断言 x 是否为代数数
    assert x.is_algebraic is False
    # 断言 x 是否为有理数
    assert x.is_rational is False
    # 断言 x 是否为整数
    assert x.is_integer is False
def test_evaluate_false():
    # 导入所需模块和符号变量
    from sympy.core.parameters import evaluate
    from sympy.abc import x, h
    # 定义表达式 f = 2^(x^7)
    f = 2**x**7
    # 使用 evaluate(False) 上下文
    with evaluate(False):
        # 将表达式 f 中的 x 替换为 x+h，生成新的表达式 fh
        fh = f.xreplace({x: x+h})
        # 断言 fh.exp 是否为有理数
        assert fh.exp.is_rational is None


def test_issue_3825():
    """catch: hash instability"""
    # 定义符号变量 x 和 y
    x = Symbol("x")
    y = Symbol("y")
    # 创建两个表达式 a1 和 a2，它们语义上相等但结构不同
    a1 = x + y
    a2 = y + x
    # 检查 a2 是否可比较
    a2.is_comparable

    # 计算表达式 a1 和 a2 的哈希值
    h1 = hash(a1)
    h2 = hash(a2)
    # 断言两者哈希值相等
    assert h1 == h2


def test_issue_4822():
    # 定义复数 z
    z = (-1)**Rational(1, 3)*(1 - I*sqrt(3))
    # 断言 z 是否为实数，期望结果为 True 或 None
    assert z.is_real in [True, None]


def test_hash_vs_typeinfo():
    """seemingly different typeinfo, but in fact equal"""

    # 定义两个语义上相等但类型信息不同的符号变量 x1 和 x2
    x1 = Symbol('x', even=True)
    x2 = Symbol('x', integer=True, odd=False)

    # 断言 x1 和 x2 的哈希值相等
    assert hash(x1) == hash(x2)
    # 断言 x1 和 x2 是相等的
    assert x1 == x2


def test_hash_vs_typeinfo_2():
    """different typeinfo should mean !eq"""
    # 定义符号变量 x 和具有不同类型信息的符号变量 x1
    x = Symbol('x')
    x1 = Symbol('x', even=True)

    # 断言 x 和 x1 不相等
    assert x != x1
    # 断言 x 和 x1 的哈希值不相等（极低概率下可能会失败）
    assert hash(x) != hash(x1)


def test_hash_vs_eq():
    """catch: different hash for equal objects"""
    # 创建表达式 a = 1 + S.Pi，确保不将其折叠成一个 Number 实例
    a = 1 + S.Pi    # important: do not fold it into a Number instance
    ha = hash(a)  # 计算表达式 a 的哈希值（应为 Add/Mul/... 以触发 bug）

    a.is_positive   # 使用 .evalf() 推断其为正数
    assert a.is_positive is True

    # 确保哈希值保持不变
    assert ha == hash(a)

    # 创建表达式 b，对 a 进行展开（在三角函数中）
    b = a.expand(trig=True)
    hb = hash(b)

    # 断言 a 和 b 是相等的
    assert a == b
    # 断言 a 和 b 的哈希值相等
    assert ha == hb


def test_Add_is_pos_neg():
    # 这些测试覆盖了核心中其他测试未覆盖的行
    # 定义各种类型的符号变量和表达式，并进行相关断言测试


def test_Add_is_imaginary():
    # 定义一个虚拟的符号变量 nn
    nn = Dummy(nonnegative=True)
    # 断言检查表达式 (I*nn + I).is_imaginary 是否为真
    assert (I*nn + I).is_imaginary  # issue 8046, 17
# 定义一个测试函数，用于测试符号操作中的代数性质
def test_Add_is_algebraic():
    # 创建代数符号 'a' 和 'b'
    a = Symbol('a', algebraic=True)
    b = Symbol('a', algebraic=True)
    # 创建非代数符号 'na' 和 'nb'
    na = Symbol('na', algebraic=False)
    nb = Symbol('nb', algebraic=False)
    # 创建符号 'x'
    x = Symbol('x')
    # 断言代数表达式 'a + b' 是否为代数
    assert (a + b).is_algebraic
    # 断言非代数表达式 'na + nb' 是否为代数（应返回 None）
    assert (na + nb).is_algebraic is None
    # 断言 'a + na' 是否为代数（应返回 False）
    assert (a + na).is_algebraic is False
    # 断言 'a + x' 是否为代数（应返回 None）
    assert (a + x).is_algebraic is None
    # 断言 'na + x' 是否为代数（应返回 None）
    assert (na + x).is_algebraic is None


# 定义测试函数，测试乘法操作中的代数性质
def test_Mul_is_algebraic():
    # 创建代数符号 'a' 和 'b'
    a = Symbol('a', algebraic=True)
    b = Symbol('b', algebraic=True)
    # 创建非代数符号 'na' 和 'nb'
    na = Symbol('na', algebraic=False)
    nb = Symbol('nb', algebraic=False)
    # 创建代数符号 'an'，且为非零
    an = Symbol('an', algebraic=True, nonzero=True)
    # 创建符号 'x'
    x = Symbol('x')
    # 断言代数表达式 'a * b' 是否为代数
    assert (a*b).is_algebraic is True
    # 断言非代数表达式 'na * nb' 是否为代数（应返回 None）
    assert (na*nb).is_algebraic is None
    # 断言 'a * na' 是否为代数（应返回 None）
    assert (a*na).is_algebraic is None
    # 断言 'an * na' 是否为代数（应返回 False）
    assert (an*na).is_algebraic is False
    # 断言 'a * x' 是否为代数（应返回 None）
    assert (a*x).is_algebraic is None
    # 断言 'na * x' 是否为代数（应返回 None）
    assert (na*x).is_algebraic is None


# 定义测试函数，测试幂次操作中的代数性质
def test_Pow_is_algebraic():
    # 创建代数符号 'e'
    e = Symbol('e', algebraic=True)

    # 断言幂次运算 '1**e' 是否为代数
    assert Pow(1, e, evaluate=False).is_algebraic
    # 断言幂次运算 '0**e' 是否为代数
    assert Pow(0, e, evaluate=False).is_algebraic

    # 创建代数符号 'a' 和 'azf'，以及非代数符号 'na'、有理数 'r' 和符号 'x'
    a = Symbol('a', algebraic=True)
    azf = Symbol('azf', algebraic=True, zero=False)
    na = Symbol('na', algebraic=False)
    ia = Symbol('ia', algebraic=True, irrational=True)
    ib = Symbol('ib', algebraic=True, irrational=True)
    r = Symbol('r', rational=True)
    x = Symbol('x')

    # 断言幂次运算 'a**2' 是否为代数
    assert (a**2).is_algebraic is True
    # 断言幂次运算 'a**r' 是否为代数（应返回 None）
    assert (a**r).is_algebraic is None
    # 断言幂次运算 'azf**r' 是否为代数
    assert (azf**r).is_algebraic is True
    # 断言幂次运算 'a**x' 是否为代数（应返回 None）
    assert (a**x).is_algebraic is None
    # 断言幂次运算 'na**r' 是否为代数（应返回 None）
    assert (na**r).is_algebraic is None
    # 断言幂次运算 'ia**r' 是否为代数
    assert (ia**r).is_algebraic is True
    # 断言幂次运算 'ia**ib' 是否为代数（应返回 False）
    assert (ia**ib).is_algebraic is False

    # 断言幂次运算 'a**e' 是否为代数（应返回 None）
    assert (a**e).is_algebraic is None

    # 断言 Gelfond-Schneider 常数幂次运算 '2**sqrt(2)' 是否为代数（应返回 False）
    assert Pow(2, sqrt(2), evaluate=False).is_algebraic is False

    # 断言黄金比例幂次运算 'GoldenRatio**sqrt(3)' 是否为代数（应返回 False）
    assert Pow(S.GoldenRatio, sqrt(3), evaluate=False).is_algebraic is False

    # issue 8649
    # 创建实数且超越数 't' 和整数 'n'
    t = Symbol('t', real=True, transcendental=True)
    n = Symbol('n', integer=True)
    # 断言幂次运算 't**n' 是否为代数（应返回 None）
    assert (t**n).is_algebraic is None
    # 断言幂次运算 't**n' 是否为整数（应返回 None）
    assert (t**n).is_integer is None

    # 断言幂次运算 'pi**3' 是否为代数（应返回 False）
    assert (pi**3).is_algebraic is False
    # 创建零值符号 'r'
    r = Symbol('r', zero=True)
    # 断言幂次运算 'pi**r' 是否为代数
    assert (pi**r).is_algebraic is True


# 定义测试函数，测试乘法操作中的质数和合数性质
def test_Mul_is_prime_composite():
    # 创建正整数符号 'x' 和 'y'
    x = Symbol('x', positive=True, integer=True)
    y = Symbol('y', positive=True, integer=True)
    # 断言乘法表达式 'x * y' 是否为质数（应返回 None）
    assert (x*y).is_prime is None
    # 断言乘法表达式 '(x+1)*(y+1)' 是否为质数（应返回 False）
    assert ((x+1)*(y+1)).is_prime is False
    # 断言乘法表达式 '(x+1)*(y+1)' 是否为合数
    assert ((x+1)*(y+1)).is_composite is True

    # 创建正数符号 'x'
    x = Symbol('x', positive=True)
    # 断言乘法表达式 '(x+1)*(y+1)' 是否为质数（应返回 None）
    assert ((x+1)*(y+1)).is_prime is None
    # 断言乘法表达式 '(x+1)*(y+1)' 是否为合数（应返回 None）
    assert ((x+1)*(y+1)).is_composite is None


# 定义测试函数，测试幂次操作中的正负性质
def test_Pow_is_pos_neg():
    # 创建实数符号 'z' 和 'w'
    z = Symbol('z', real=True)
    w = Symbol('w', nonpositive=True)

    # 断言幂次运算 '(-1)**2' 是否为正数
    assert (S.NegativeOne**S(2)).is_positive is True
    # 断言幂次运算 '1**z' 是否为正数
    assert (S.One**z).is_positive is True
    # 断言幂次运算 '(-1)**3' 是否为正数
    assert (S.NegativeOne**S(3)).is_positive is False
    # 断言幂次运算 '0**0' 是否为正数（0**0 为 1）
    assert (S.Zero**S.Zero).is_positive is True
    # 断言幂次运算 'w**
    # 定义符号变量 p，其值可以为零
    p = Symbol('p', zero=True)
    # 定义符号变量 q，其值不为零且为实数
    q = Symbol('q', zero=False, real=True)
    # 定义符号变量 j，其值不为零且为偶数
    j = Symbol('j', zero=False, even=True)
    # 定义符号变量 x 和 y，它们的值可以为零
    x = Symbol('x', zero=True)
    y = Symbol('y', zero=True)
    
    # 断言，检查表达式 (p**q).is_positive 是否为 False
    assert (p**q).is_positive is False
    # 断言，检查表达式 (p**q).is_negative 是否为 False
    assert (p**q).is_negative is False
    # 断言，检查表达式 (p**j).is_positive 是否为 False
    assert (p**j).is_positive is False
    # 断言，检查表达式 (x**y).is_positive 是否为 True，这里是 0**0 的情况
    assert (x**y).is_positive is True
    # 断言，检查表达式 (x**y).is_negative 是否为 False
    assert (x**y).is_negative is False
# 定义一个测试函数，测试幂运算的数学性质
def test_Pow_is_prime_composite():
    # 定义符号变量 x 和 y，要求它们为正整数
    x = Symbol('x', positive=True, integer=True)
    y = Symbol('y', positive=True, integer=True)
    # 检查 x**y 是否为质数，预期返回 None
    assert (x**y).is_prime is None
    # 检查 x**(y+1) 是否为质数，预期返回 False
    assert ( x**(y+1) ).is_prime is False
    # 检查 x**(y+1) 是否为合数，预期返回 None
    assert ( x**(y+1) ).is_composite is None
    # 检查 (x+1)**(y+1) 是否为合数，预期返回 True
    assert ( (x+1)**(y+1) ).is_composite is True
    # 检查 (-x-1)**(2*y) 是否为合数，预期返回 True
    assert ( (-x-1)**(2*y) ).is_composite is True

    # 重新定义符号变量 x，仅要求其为正数，y 的类型保持不变
    x = Symbol('x', positive=True)
    # 检查 x**y 是否为质数，预期返回 None
    assert (x**y).is_prime is None


# 定义一个测试函数，测试乘法运算的有限性和无限性性质
def test_Mul_is_infinite():
    # 定义符号变量 x，f，i，分别具有不同的性质
    x = Symbol('x')
    f = Symbol('f', finite=True)
    i = Symbol('i', infinite=True)
    z = Dummy(zero=True)
    nzf = Dummy(finite=True, zero=False)
    from sympy.core.mul import Mul
    # 检查 x*f 是否为有限的，预期返回 None
    assert (x*f).is_finite is None
    # 检查 x*i 是否为有限的，预期返回 None
    assert (x*i).is_finite is None
    # 检查 f*i 是否为有限的，预期返回 None
    assert (f*i).is_finite is None
    # 检查 x*f*i 是否为有限的，预期返回 None
    assert (x*f*i).is_finite is None
    # 检查 z*i 是否为有限的，预期返回 None
    assert (z*i).is_finite is None
    # 检查 nzf*i 是否为有限的，预期返回 False
    assert (nzf*i).is_finite is False
    # 检查 z*f 是否为有限的，预期返回 True
    assert (z*f).is_finite is True
    # 检查 Mul(0, f, evaluate=False) 是否为有限的，预期返回 True
    assert Mul(0, f, evaluate=False).is_finite is True
    # 检查 Mul(0, i, evaluate=False) 是否为有限的，预期返回 None
    assert Mul(0, i, evaluate=False).is_finite is None

    # 检查 x*f 是否为无限的，预期返回 None
    assert (x*f).is_infinite is None
    # 检查 x*i 是否为无限的，预期返回 None
    assert (x*i).is_infinite is None
    # 检查 f*i 是否为无限的，预期返回 None
    assert (f*i).is_infinite is None
    # 检查 x*f*i 是否为无限的，预期返回 None
    assert (x*f*i).is_infinite is None
    # 检查 z*i 是否为无限的，预期返回 True
    assert (z*i).is_infinite is True
    # 检查 nzf*i 是否为无限的，预期返回 True
    assert (nzf*i).is_infinite is True
    # 检查 z*f 是否为无限的，预期返回 False
    assert (z*f).is_infinite is False
    # 检查 Mul(0, f, evaluate=False) 是否为无限的，预期返回 False
    assert Mul(0, f, evaluate=False).is_infinite is False
    # 检查 Mul(0, i, evaluate=False) 是否为无限的，预期返回 True
    assert Mul(0, i, evaluate=False).is_infinite is True


# 定义一个测试函数，测试加法运算的有限性和无限性性质
def test_Add_is_infinite():
    # 定义符号变量 x，f，i，i2，分别具有不同的性质
    x = Symbol('x')
    f = Symbol('f', finite=True)
    i = Symbol('i', infinite=True)
    i2 = Symbol('i2', infinite=True)
    z = Dummy(zero=True)
    nzf = Dummy(finite=True, zero=False)
    from sympy.core.add import Add
    # 检查 x+f 是否为有限的，预期返回 None
    assert (x+f).is_finite is None
    # 检查 x+i 是否为有限的，预期返回 None
    assert (x+i).is_finite is None
    # 检查 f+i 是否为有限的，预期返回 False
    assert (f+i).is_finite is False
    # 检查 x+f+i 是否为有限的，预期返回 None
    assert (x+f+i).is_finite is None
    # 检查 z+i 是否为有限的，预期返回 False
    assert (z+i).is_finite is False
    # 检查 nzf+i 是否为有限的，预期返回 False
    assert (nzf+i).is_finite is False
    # 检查 z+f 是否为有限的，预期返回 True
    assert (z+f).is_finite is True
    # 检查 i+i2 是否为有限的，预期返回 None
    assert (i+i2).is_finite is None
    # 检查 Add(0, f, evaluate=False) 是否为有限的，预期返回 True
    assert Add(0, f, evaluate=False).is_finite is True
    # 检查 Add(0, i, evaluate=False) 是否为有限的，预期返回 False
    assert Add(0, i, evaluate=False).is_finite is False

    # 检查 x+f 是否为无限的，预期返回 None
    assert (x+f).is_infinite is None
    # 检查 x+i 是否为无限的，预期返回 None
    assert (x+i).is_infinite is None
    # 检查 f+i 是否为无限的，预期返回 True
    assert (f+i).is_infinite is True
    # 检查 x+f+i 是否为无限的，预期返回 None
    assert (x+f+i).is_infinite is None
    # 检查 z+i 是否为无限的，预期返回 True
    assert (z+i).is_infinite is True
    # 检查 nzf+i 是否为无限的，预期返回 True
    assert (nzf+i).is_infinite is True
    # 检查 z+f 是否为无限的，预期返回 False
    assert (z+f).is_infinite is False
    # 检查 i+i2 是否为无限的，预期返回 None
    assert (i+i2).is_infinite is None
    # 检查 Add(0, f, evaluate=False) 是否为无限的，预期返回 False
    assert Add(0, f, evaluate=False).is_infinite is False
    # 检查 Add(0, i, evaluate=False) 是否为无限的，预期返回 True
    assert Add(0, i, evaluate=False).is_infinite is True


# 定义一个测试函数，测试特殊函数的有理性质
def test_special_is_rational():
    # 定义符号变量 i，i2，ni，r，rn，nr，x，具有不同的数学性质
    i = Symbol('i', integer=True)
    i2 = Symbol('i2', integer=True)
    ni = Symbol('ni', integer=True, nonzero=True)
    r = Symbol('r', rational=True)
    rn = Symbol('r', rational=True, nonzero=True)
    nr = Symbol('nr', irrational=True)
    x = Symbol('x')
    # 检查 sqrt(3) 是否为有理数，预期返回 False
    assert
    # 检查指数函数应用到给定符号 rn 上，验证其是否为有理数，预期为 False
    assert exp(rn).is_rational is False
    
    # 检查指数函数应用到给定符号 x 上，验证其是否为有理数，预期为 None
    assert exp(x).is_rational is None
    
    # 检查对数函数应用到 log(3) 上，禁用自动求值，验证其是否为有理数，预期为 True
    assert exp(log(3), evaluate=False).is_rational is True
    
    # 检查对数函数应用到 exp(3) 上，禁用自动求值，验证其是否为有理数，预期为 True
    assert log(exp(3), evaluate=False).is_rational is True
    
    # 检查对数函数应用到 log(3) 上，验证其是否为有理数，预期为 False
    assert log(3).is_rational is False
    
    # 检查对数函数应用到 ni + 1 上，验证其是否为有理数，预期为 False
    assert log(ni + 1).is_rational is False
    
    # 检查对数函数应用到 rn + 1 上，验证其是否为有理数，预期为 False
    assert log(rn + 1).is_rational is False
    
    # 检查对数函数应用到符号 x 上，验证其是否为有理数，预期为 None
    assert log(x).is_rational is None
    
    # 检查 sqrt(3) + sqrt(5) 的和，验证其是否为有理数，预期为 None
    assert (sqrt(3) + sqrt(5)).is_rational is None
    
    # 检查 sqrt(3) + S.Pi 的和，验证其是否为有理数，预期为 False
    assert (sqrt(3) + S.Pi).is_rational is False
    
    # 检查 x 的 i 次幂，验证其是否为有理数，预期为 None
    assert (x**i).is_rational is None
    
    # 检查 i 的 i 次幂，验证其是否为有理数，预期为 True
    assert (i**i).is_rational is True
    
    # 检查 i2 的 i 次幂，验证其是否为有理数，预期为 None
    assert (i**i2).is_rational is None
    
    # 检查 r 的 i 次幂，验证其是否为有理数，预期为 None
    assert (r**i).is_rational is None
    
    # 检查 r 的 r 次幂，验证其是否为有理数，预期为 None
    assert (r**r).is_rational is None
    
    # 检查 r 的 x 次幂，验证其是否为有理数，预期为 None
    assert (r**x).is_rational is None
    
    # 检查 nr 的 i 次幂，验证其是否为有理数，预期为 None，此为问题编号 8598
    assert (nr**i).is_rational is None
    
    # 检查 nr 的 Symbol('z', zero=True) 次幂，验证其是否为有理数，预期为 True
    assert (nr**Symbol('z', zero=True)).is_rational
    
    # 检查 sin(1) 的值，验证其是否为有理数，预期为 False
    assert sin(1).is_rational is False
    
    # 检查 sin(ni) 的值，验证其是否为有理数，预期为 False
    assert sin(ni).is_rational is False
    
    # 检查 sin(rn) 的值，验证其是否为有理数，预期为 False
    assert sin(rn).is_rational is False
    
    # 检查 sin(x) 的值，验证其是否为有理数，预期为 None
    assert sin(x).is_rational is None
    
    # 检查 asin(r) 的值，验证其是否为有理数，预期为 False
    assert asin(r).is_rational is False
    
    # 检查 sin(asin(3)) 的值，禁用自动求值，验证其是否为有理数，预期为 True
    assert sin(asin(3), evaluate=False).is_rational is True
@XFAIL
# 标记为预期失败的测试函数，用于标识测试案例中的特定问题或缺陷

def test_issue_6275():
    # 创建一个符号变量 x
    x = Symbol('x')
    # 断言 x*0 的类型是否与 0*S.Infinity 的类型相同
    # 注释指出这种情况类似于 x/x => 1，即使当 x = 0 时实际上是 nan
    assert isinstance(x*0, type(0*S.Infinity))
    # 如果 0*S.Infinity 等于 S.NaN
    if 0*S.Infinity is S.NaN:
        # 创建一个符号变量 b，其 finite 属性设置为 None
        b = Symbol('b', finite=None)
        # 断言 b*0 的 is_zero 属性为 None
        assert (b*0).is_zero is None


def test_sanitize_assumptions():
    # issue 6666
    # 对于每个类 cls 在 Symbol、Dummy 和 Wild 中
    for cls in (Symbol, Dummy, Wild):
        # 创建一个名为 x 的符号变量，设置 real=1, positive=0
        x = cls('x', real=1, positive=0)
        # 断言 x 的 is_real 属性为 True
        assert x.is_real is True
        # 断言 x 的 is_positive 属性为 False
        assert x.is_positive is False
        # 创建一个空字符串符号，设置 real=True, positive=None
        # 断言其 is_positive 属性为 None
        assert cls('', real=True, positive=None).is_positive is None
        # 调用 Symbol._sanitize 函数，期望引发 ValueError 异常
        raises(ValueError, lambda: cls('', commutative=None))
    # 调用 Symbol._sanitize 函数，期望引发 ValueError 异常
    raises(ValueError, lambda: Symbol._sanitize({"commutative": None}))


def test_special_assumptions():
    e = -3 - sqrt(5) + (-sqrt(10)/2 - sqrt(2)/2)**2
    # 断言简化表达式 e < 0 的结果为 S.false
    assert simplify(e < 0) is S.false
    # 断言简化表达式 e > 0 的结果为 S.false
    assert simplify(e > 0) is S.false
    # 断言 e == 0 的结果为 False
    assert (e == 0) is False  # it's not a literal 0
    # 断言 e.equals(0) 的结果为 True
    assert e.equals(0) is True


def test_inconsistent():
    # cf. issues 5795 and 5545
    # 断言创建一个名为 x 的符号变量，real=True, commutative=False
    # 期望引发 InconsistentAssumptions 异常
    raises(InconsistentAssumptions, lambda: Symbol('x', real=True,
           commutative=False))


def test_issue_6631():
    # 断言 (-1)**(I) 的 is_real 属性为 True
    assert ((-1)**(I)).is_real is True
    # 断言 (-1)**(I*2) 的 is_real 属性为 True
    assert ((-1)**(I*2)).is_real is True
    # 断言 (-1)**(I/2) 的 is_real 属性为 True
    assert ((-1)**(I/2)).is_real is True
    # 断言 (-1)**(I*S.Pi) 的 is_real 属性为 True
    assert ((-1)**(I*S.Pi)).is_real is True
    # 断言 (I**(I + 2)) 的 is_real 属性为 True
    assert (I**(I + 2)).is_real is True


def test_issue_2730():
    # 断言 (1/(1 + I)) 的 is_real 属性为 False
    assert (1/(1 + I)).is_real is False


def test_issue_4149():
    # 断言 (3 + I) 的 is_complex 属性为 True
    assert (3 + I).is_complex
    # 断言 (3 + I) 的 is_imaginary 属性为 False
    assert (3 + I).is_imaginary is False
    # 断言 (3*I + S.Pi*I) 的 is_imaginary 属性为 True
    assert (3*I + S.Pi*I).is_imaginary
    # 创建一个名为 y 的符号变量，real=True
    y = Symbol('y', real=True)
    # 断言 (3*I + S.Pi*I + y*I) 的 is_imaginary 属性为 None
    assert (3*I + S.Pi*I + y*I).is_imaginary is None
    # 创建一个名为 p 的符号变量，positive=True
    p = Symbol('p', positive=True)
    # 断言 (3*I + S.Pi*I + p*I) 的 is_imaginary 属性为 True
    assert (3*I + S.Pi*I + p*I).is_imaginary
    # 创建一个名为 n 的符号变量，negative=True
    n = Symbol('n', negative=True)
    # 断言 (-3*I - S.Pi*I + n*I) 的 is_imaginary 属性为 True
    assert (-3*I - S.Pi*I + n*I).is_imaginary

    # 创建一个名为 i 的符号变量，imaginary=True
    i = Symbol('i', imaginary=True)
    # 断言 [(i**a).is_imaginary for a in range(4)] 的结果为 [False, True, False, True]
    assert ([(i**a).is_imaginary for a in range(4)] ==
            [False, True, False, True])

    # 创建一个表达式 e，包含复数部分
    e = S("-sqrt(3)*I/2 + 0.866025403784439*I")
    # 断言 e 的 is_real 属性为 False
    assert e.is_real is False
    # 断言 e 的 is_imaginary 属性为 True


def test_issue_2920():
    # 创建一个名为 n 的符号变量，negative=True
    n = Symbol('n', negative=True)
    # 断言 sqrt(n).is_imaginary 的结果为 True
    assert sqrt(n).is_imaginary


def test_issue_7899():
    # 创建一个名为 x 的符号变量，real=True
    x = Symbol('x', real=True)
    # 断言 (I*x).is_real 的结果为 None
    assert (I*x).is_real is None
    # 断言 ((x - I)*(x - 1)).is_zero 的结果为 None
    assert ((x - I)*(x - 1)).is_zero is None
    # 断言 ((x - I)*(x - 1)).is_real 的结果为 None


@XFAIL
# 标记为预期失败的测试函数，用于标识测试案例中的特定问题或缺陷

def test_issue_7993():
    # 创建一个 Dummy 对象 x，设置 integer=True
    x = Dummy(integer=True)
    # 创建一个 Dummy 对象 y，设置 noninteger=True
    y = Dummy(noninteger=True)
    # 断言 (x - y).is_zero 的结果为 False


def test_issue_8075():
    # 断言创建一个 Dummy 对象，设置 zero=True, finite=False
    # 期望引发 InconsistentAssumptions 异常
    raises(InconsistentAssumptions, lambda: Dummy(zero=True, finite=False))
    # 断言创建一个 Dummy 对象，设置 zero=True, infinite=True
    # 期望引发 InconsistentAssumptions 异常
    raises(InconsistentAssumptions, lambda: Dummy(zero=True, infinite=True))


def test_issue_8642():
    # 创建一个名为 x 的符号变量，real=True, integer=False
    x = Symbol('x', real=True, integer=False)
    # 断言 (x*2).is_integer 的结果为 None，输出 (x*2).is_integer 的值
    assert (x*2).is_integer is None, (x*2).is_integer


def test_issues_8632_8633_8638_8675_8992():
    # 创建一个 Dummy 对象 p，设置 integer=True, positive=True
    p = Dummy(integer=True, positive=True)
    # 创建一个 Dummy 对象 nn，设置 integer=True, nonnegative=True
    nn = Dummy(integer=True, nonnegative=True)
    # 断言 p - S.Half 大于零
    assert (p - S.Half).is_positive
    # 断言 p - 1 非负
    assert (p - 1).is_nonnegative
    # 断言 nn + 1 大于零
    assert (nn + 1).is_positive
    # 断言 -p + 1 非正
    assert (-p + 1).is_nonpositive
    # 断言 -nn - 1 小于零
    assert (-nn - 1).is_negative
    # 创建一个 prime 的虚拟变量，其值为素数
    prime = Dummy(prime=True)
    # 断言 prime - 2 非负
    assert (prime - 2).is_nonnegative
    # 断言 prime - 3 非负为 None
    assert (prime - 3).is_nonnegative is None
    # 创建一个 even 的虚拟变量，其为正偶数
    even = Dummy(positive=True, even=True)
    # 断言 even - 2 非负
    assert (even - 2).is_nonnegative

    # 创建一个 positive 的虚拟变量 p
    p = Dummy(positive=True)
    # 断言 p / (p + 1) - 1 小于零
    assert (p/(p + 1) - 1).is_negative
    # 断言 (p + 2)**3 - S.Half 大于零
    assert ((p + 2)**3 - S.Half).is_positive
    # 创建一个 negative 的虚拟变量 n
    n = Dummy(negative=True)
    # 断言 n - 3 非正
    assert (n - 3).is_nonpositive
def test_issue_9115_9150():
    # 创建一个整数非负的虚拟变量 n
    n = Dummy('n', integer=True, nonnegative=True)
    # 断言 n 的阶乘大于等于 1
    assert (factorial(n) >= 1) == True
    # 断言 n 的阶乘小于 1
    assert (factorial(n) < 1) == False

    # 断言 n+1 的阶乘的偶数性质为 None
    assert factorial(n + 1).is_even is None
    # 断言 n+2 的阶乘的偶数性质为 True
    assert factorial(n + 2).is_even is True
    # 断言 n+2 的阶乘大于等于 2
    assert factorial(n + 2) >= 2


def test_issue_9165():
    # 创建一个可以表示为零的符号变量 z
    z = Symbol('z', zero=True)
    # 创建一个无限但有限的符号变量 f
    f = Symbol('f', finite=False)
    # 断言 0 除以 z 的结果为 S.NaN
    assert 0/z is S.NaN
    # 断言 0 乘以 1/z 的结果为 S.NaN
    assert 0*(1/z) is S.NaN
    # 断言 0 乘以 f 的结果为 S.NaN
    assert 0*f is S.NaN


def test_issue_10024():
    # 创建一个虚拟变量 x
    x = Dummy('x')
    # 断言 x 对 2*pi 取模的结果为零
    assert Mod(x, 2*pi).is_zero is None


def test_issue_10302():
    # 创建实数符号变量 x 和 r
    x = Symbol('x')
    r = Symbol('r', real=True)
    # 计算复数 u 和复数 i
    u = -(3*2**pi)**(1/pi) + 2*3**(1/pi)
    i = u + u*I

    # 断言 i 是实数的结果为 None（未简化情况下应失败）
    assert i.is_real is None
    # 断言 u + i 的结果为零的结果为 None
    assert (u + i).is_zero is None
    # 断言 1 + i 的结果为零为 False
    assert (1 + i).is_zero is False

    # 创建一个零的虚拟变量 a
    a = Dummy('a', zero=True)
    # 断言 a + I 的结果为零为 False
    assert (a + I).is_zero is False
    # 断言 a + r*I 的结果为零的结果为 None
    assert (a + r*I).is_zero is None
    # 断言 a + I 是虚数的结果为 True
    assert (a + I).is_imaginary
    # 断言 a + x + I 是虚数的结果为 None
    assert (a + x + I).is_imaginary is None
    # 断言 a + r*I + I 是虚数的结果为 None
    assert (a + r*I + I).is_imaginary is None


def test_complex_reciprocal_imaginary():
    # 断言 1 / (4 + 3*I) 是虚数的结果为 False
    assert (1 / (4 + 3*I)).is_imaginary is False


def test_issue_16313():
    # 创建一个不是扩展实数的符号变量 x，以及两个实数变量 k 和 l
    x = Symbol('x', extended_real=False)
    k = Symbol('k', real=True)
    l = Symbol('l', real=True, zero=False)
    # 断言 -x 不是实数的结果为 False
    assert (-x).is_real is False
    # 断言 k*x 是实数的结果为 None（k 也可能为零）
    assert (k*x).is_real is None
    # 断言 l*x 不是实数的结果为 False
    assert (l*x).is_real is False
    # 断言 l*x*x 是实数的结果为 None（因为 x*x 可能是实数）
    assert (l*x*x).is_real is None
    # 断言 -x 是正数的结果为 False
    assert (-x).is_positive is False


def test_issue_16579():
    # 创建一个扩展实数但有限的符号变量 x，以及一个扩展实数但无限的符号变量 y
    x = Symbol('x', extended_real=True, infinite=False)
    y = Symbol('y', extended_real=True, finite=False)
    # 断言 x 是有限的结果为 True
    assert x.is_finite is True
    # 断言 y 是无限的结果为 True
    assert y.is_infinite is True

    # 随着 PR 16978 的更新，复数现在意味着有限
    c = Symbol('c', complex=True)
    # 断言 c 是有限的结果为 True
    assert c.is_finite is True
    # 使用 lambda 表达式来检查符号的矛盾假设
    raises(InconsistentAssumptions, lambda: Dummy(complex=True, finite=False))

    # 现在无限意味着非有限
    nf = Symbol('nf', finite=False)
    # 断言 nf 是无限的结果为 True
    assert nf.is_infinite is True


def test_issue_17556():
    # 创建一个虚数无穷的符号变量 z
    z = I*oo
    # 断言 z 是虚数的结果为 False
    assert z.is_imaginary is False
    # 断言 z 是有限的结果为 False
    assert z.is_finite is False


def test_issue_21651():
    # 创建一个正整数的整数变量 k
    k = Symbol('k', positive=True, integer=True)
    # 计算表达式 exp
    exp = 2*2**(-k)
    # 断言 exp 是整数的结果为 None
    assert exp.is_integer is None


def test_assumptions_copy():
    # 断言对于符号变量 'x'，给定的假设为 {'commutative': True}
    assert assumptions(Symbol('x'), {"commutative": True}
        ) == {'commutative': True}
    # 断言对于符号变量 'x'，给定的假设为 ['integer']，返回空字典
    assert assumptions(Symbol('x'), ['integer']) == {}
    # 断言对于符号变量 'x'，给定的假设为 ['commutative']，返回 {'commutative': True}
    assert assumptions(Symbol('x'), ['commutative']
        ) == {'commutative': True}
    # 断言对于符号变量 'x'，默认假设为 {'commutative': True}
    assert assumptions(Symbol('x')) == {'commutative': True}
    # 断言对于整数 1，其默认假设为 {'positive'}
    assert assumptions(1)['positive']
    # 断言测试假设条件函数 assumptions(3 + I) 的返回结果是否符合预期
    assert assumptions(3 + I) == {
        # 是否是代数的
        'algebraic': True,
        # 是否是可交换的
        'commutative': True,
        # 是否是复数
        'complex': True,
        # 是否是复合数
        'composite': False,
        # 是否是偶数
        'even': False,
        # 是否是扩展负数
        'extended_negative': False,
        # 是否是扩展非负数
        'extended_nonnegative': False,
        # 是否是扩展非正数
        'extended_nonpositive': False,
        # 是否是扩展非零数
        'extended_nonzero': False,
        # 是否是扩展正数
        'extended_positive': False,
        # 是否是扩展实数
        'extended_real': False,
        # 是否是有限的
        'finite': True,
        # 是否是虚数
        'imaginary': False,
        # 是否是无限的
        'infinite': False,
        # 是否是整数
        'integer': False,
        # 是否是无理数
        'irrational': False,
        # 是否是负数
        'negative': False,
        # 是否是非整数
        'noninteger': False,
        # 是否是非负数
        'nonnegative': False,
        # 是否是非正数
        'nonpositive': False,
        # 是否是非零数
        'nonzero': False,
        # 是否是奇数
        'odd': False,
        # 是否是正数
        'positive': False,
        # 是否是质数
        'prime': False,
        # 是否是有理数
        'rational': False,
        # 是否是实数
        'real': False,
        # 是否是超越数
        'transcendental': False,
        # 是否是零
        'zero': False
    }
# 检查符号表达式是否满足给定的假设条件
def test_check_assumptions():
    # 断言假设条件为 (1, 0) 时返回 False
    assert check_assumptions(1, 0) is False
    # 创建一个正数符号变量 x
    x = Symbol('x', positive=True)
    # 断言假设条件为 (1, x) 时返回 True
    assert check_assumptions(1, x) is True
    # 断言假设条件为 (1, 1) 时返回 True
    assert check_assumptions(1, 1) is True
    # 断言假设条件为 (-1, 1) 时返回 False
    assert check_assumptions(-1, 1) is False
    # 创建一个整数符号变量 i
    i = Symbol('i', integer=True)
    # 断言假设条件为 (i, 1) 时返回 None，因为 i 的正负情况未知
    assert check_assumptions(i, 1) is None
    # 断言假设条件为 (Dummy(integer=None), integer=True) 时返回 None
    assert check_assumptions(Dummy(integer=None), integer=True) is None
    # 断言假设条件为 (Dummy(integer=None), integer=False) 时返回 None
    assert check_assumptions(Dummy(integer=None), integer=False) is None
    # 断言假设条件为 (Dummy(integer=False), integer=True) 时返回 False
    assert check_assumptions(Dummy(integer=False), integer=True) is False
    # 断言假设条件为 (Dummy(integer=True), integer=False) 时返回 False
    assert check_assumptions(Dummy(integer=True), integer=False) is False
    # 断言假设条件为 (Dummy(integer=False), integer=None) 时返回 True
    assert check_assumptions(Dummy(integer=False), integer=None) is True
    # 检查传入正数假设的表达式是否引发 ValueError 异常
    raises(ValueError, lambda: check_assumptions(2*x, x, positive=True))


# 测试符号表达式的假设失败情况
def test_failing_assumptions():
    # 创建一个正数符号变量 x 和一个一般符号变量 y
    x = Symbol('x', positive=True)
    y = Symbol('y')
    # 断言对 6*x + y 的假设检查结果符合预期
    assert failing_assumptions(6*x + y, **x.assumptions0) == {
        'real': None, 'imaginary': None, 'complex': None, 'hermitian': None,
        'positive': None, 'nonpositive': None, 'nonnegative': None, 'nonzero': None,
        'negative': None, 'zero': None, 'extended_real': None, 'finite': None,
        'infinite': None, 'extended_negative': None, 'extended_nonnegative': None,
        'extended_nonpositive': None, 'extended_nonzero': None,
        'extended_positive': None }


# 测试常见假设条件的函数
def test_common_assumptions():
    # 断言对于输入列表 [0, 1, 2] 的常见假设结果符合预期
    assert common_assumptions([0, 1, 2]
        ) == {'algebraic': True, 'irrational': False, 'hermitian':
        True, 'extended_real': True, 'real': True, 'extended_negative':
        False, 'extended_nonnegative': True, 'integer': True,
        'rational': True, 'imaginary': False, 'complex': True,
        'commutative': True,'noninteger': False, 'composite': False,
        'infinite': False, 'nonnegative': True, 'finite': True,
        'transcendental': False,'negative': False}
    # 断言对于输入列表 [0, 1, 2] 和检查的属性 'positive' 和 'integer' 的常见假设结果符合预期
    assert common_assumptions([0, 1, 2], 'positive integer'.split()
        ) == {'integer': True}
    # 断言对于空列表的常见假设结果为空字典
    assert common_assumptions([0, 1, 2], []) == {}
    # 断言对于空列表和检查的属性 'integer' 的常见假设结果为空字典
    assert common_assumptions([], ['integer']) == {}
    # 断言对于只含一个元素 0 和检查的属性 'integer' 的常见假设结果为 {'integer': True}
    assert common_assumptions([0], ['integer']) == {'integer': True}


# 测试预生成的假设规则是否有效
def test_pre_generated_assumption_rules_are_valid():
    # 检查预生成的假设规则与新生成的假设规则是否匹配
    # 如果此检查失败，考虑更新假设规则
    # 参考 sympy.core.assumptions._generate_assumption_rules 查看更多信息
    pre_generated_assumptions = _load_pre_generated_assumption_rules()
    generated_assumptions = _generate_assumption_rules()
    assert pre_generated_assumptions._to_python() == generated_assumptions._to_python(), "pre-generated assumptions are invalid, see sympy.core.assumptions._generate_assumption_rules"


# 测试询问随机重排的功能
def test_ask_shuffle():
    # 创建一个排列群组 grp
    grp = PermutationGroup(Permutation(1, 0, 2), Permutation(2, 1, 3))

    # 使用相同的随机数种子检查不同调用下的随机结果是否一致
    seed(123)
    first = grp.random()
    seed(123)
    simplify(I)
    second = grp.random()
    seed(123)
    simplify(-I)
    # 调用 grp 对象的 random() 方法，返回一个随机数
    third = grp.random()
    
    # 使用断言检查三个变量 first、second 和 third 是否完全相等
    assert first == second == third
```