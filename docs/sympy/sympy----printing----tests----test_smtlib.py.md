# `D:\src\scipysrc\sympy\sympy\printing\tests\test_smtlib.py`

```
# 导入必要的库和模块
import contextlib  # 提供上下文管理器的模块，用于管理资源
import itertools  # 提供迭代工具的模块
import re  # 提供正则表达式操作的模块
import typing  # 提供类型提示的模块
from enum import Enum  # 导入枚举类型的基类
from typing import Callable  # 引入Callable类型提示

import sympy  # 导入sympy库，用于符号计算
from sympy import Add, Implies, sqrt  # 导入sympy中的符号操作和数学函数
from sympy.core import Mul, Pow  # 导入sympy中的核心符号操作
from sympy.core import (S, pi, symbols, Function, Rational, Integer,  # 导入sympy中的常数和符号
                        Symbol, Eq, Ne, Le, Lt, Gt, Ge)
from sympy.functions import Piecewise, exp, sin, cos  # 导入sympy中的特殊函数
from sympy.assumptions.ask import Q  # 导入sympy中的假设判断
from sympy.printing.smtlib import smtlib_code  # 导入sympy中的SMT-LIB打印功能
from sympy.testing.pytest import raises, Failed  # 导入sympy中的测试工具

x, y, z = symbols('x,y,z')  # 创建符号变量x, y, z

# 定义枚举_W，包含了三个正则表达式对象，用于匹配不同类型的警告信息
class _W(Enum):
    DEFAULTING_TO_FLOAT = re.compile("Could not infer type of `.+`. Defaulting to float.", re.I)
    WILL_NOT_DECLARE = re.compile("Non-Symbol/Function `.+` will not be declared.", re.I)
    WILL_NOT_ASSERT = re.compile("Non-Boolean expression `.+` will not be asserted. Converting to SMTLib verbatim.", re.I)

# 定义上下文管理器函数_check_warns，用于检查警告信息
@contextlib.contextmanager
def _check_warns(expected: typing.Iterable[_W]):
    warns: typing.List[str] = []  # 初始化警告列表
    log_warn = warns.append  # 定义警告信息追加函数
    yield log_warn  # 执行上下文管理器

    errors = []  # 初始化错误列表
    # 检查警告信息是否符合期望
    for i, (w, e) in enumerate(itertools.zip_longest(warns, expected)):
        if not e:
            errors += [f"[{i}] Received unexpected warning `{w}`."]
        elif not w:
            errors += [f"[{i}] Did not receive expected warning `{e.name}`."]
        elif not e.value.match(w):
            errors += [f"[{i}] Warning `{w}` does not match expected {e.name}."]

    if errors:  # 如果存在错误信息，则抛出异常
        raise Failed('\n'.join(errors))

# 测试函数test_Integer，测试整数的SMT-LIB代码生成和警告
def test_Integer():
    with _check_warns([_W.WILL_NOT_ASSERT] * 2) as w:
        assert smtlib_code(Integer(67), log_warn=w) == "67"  # 生成整数67的SMT-LIB代码
        assert smtlib_code(Integer(-1), log_warn=w) == "-1"  # 生成整数-1的SMT-LIB代码
    with _check_warns([]) as w:
        assert smtlib_code(Integer(67)) == "67"  # 生成整数67的SMT-LIB代码（无警告）
        assert smtlib_code(Integer(-1)) == "-1"  # 生成整数-1的SMT-LIB代码（无警告）

# 测试函数test_Rational，测试有理数的SMT-LIB代码生成和警告
def test_Rational():
    with _check_warns([_W.WILL_NOT_ASSERT] * 4) as w:
        assert smtlib_code(Rational(3, 7), log_warn=w) == "(/ 3 7)"  # 生成有理数3/7的SMT-LIB代码
        assert smtlib_code(Rational(18, 9), log_warn=w) == "2"  # 生成有理数18/9的SMT-LIB代码
        assert smtlib_code(Rational(3, -7), log_warn=w) == "(/ -3 7)"  # 生成有理数3/-7的SMT-LIB代码
        assert smtlib_code(Rational(-3, -7), log_warn=w) == "(/ 3 7)"  # 生成有理数-3/-7的SMT-LIB代码

    with _check_warns([_W.DEFAULTING_TO_FLOAT, _W.WILL_NOT_ASSERT] * 2) as w:
        assert smtlib_code(x + Rational(3, 7), auto_declare=False, log_warn=w) == "(+ (/ 3 7) x)"  # 生成表达式x + 3/7的SMT-LIB代码
        assert smtlib_code(Rational(3, 7) * x, log_warn=w) == "(declare-const x Real)\n" \
                                                              "(* (/ 3 7) x)"  # 生成表达式3/7 * x的SMT-LIB代码

# 测试函数test_Relational未完整显示，略去其余部分以保持代码块的长度合适
    # 使用 _check_warns 上下文管理器，生成包含默认警告信息的列表 w
    with _check_warns([_W.DEFAULTING_TO_FLOAT] * 12) as w:
        # 断言语句，验证 smtlib_code 函数处理等式 Eq(x, y) 的结果是否等于 "(assert (= x y))"
        assert smtlib_code(Eq(x, y), auto_declare=False, log_warn=w) == "(assert (= x y))"
        # 断言语句，验证 smtlib_code 函数处理不等式 Ne(x, y) 的结果是否等于 "(assert (not (= x y)))"
        assert smtlib_code(Ne(x, y), auto_declare=False, log_warn=w) == "(assert (not (= x y)))"
        # 断言语句，验证 smtlib_code 函数处理小于等于 Le(x, y) 的结果是否等于 "(assert (<= x y))"
        assert smtlib_code(Le(x, y), auto_declare=False, log_warn=w) == "(assert (<= x y))"
        # 断言语句，验证 smtlib_code 函数处理小于 Lt(x, y) 的结果是否等于 "(assert (< x y))"
        assert smtlib_code(Lt(x, y), auto_declare=False, log_warn=w) == "(assert (< x y))"
        # 断言语句，验证 smtlib_code 函数处理大于 Gt(x, y) 的结果是否等于 "(assert (> x y))"
        assert smtlib_code(Gt(x, y), auto_declare=False, log_warn=w) == "(assert (> x y))"
        # 断言语句，验证 smtlib_code 函数处理大于等于 Ge(x, y) 的结果是否等于 "(assert (>= x y))"
        assert smtlib_code(Ge(x, y), auto_declare=False, log_warn=w) == "(assert (>= x y))"
# 定义测试函数 test_AppliedBinaryRelation，用于测试二元关系运算符的行为
def test_AppliedBinaryRelation():
    # 使用 _check_warns 上下文管理器来捕获特定警告
    with _check_warns([_W.DEFAULTING_TO_FLOAT] * 12) as w:
        # 断言等式运算的 SMT-LIB 表示正确
        assert smtlib_code(Q.eq(x, y), auto_declare=False, log_warn=w) == "(assert (= x y))"
        # 断言不等运算的 SMT-LIB 表示正确
        assert smtlib_code(Q.ne(x, y), auto_declare=False, log_warn=w) == "(assert (not (= x y)))"
        # 断言小于运算的 SMT-LIB 表示正确
        assert smtlib_code(Q.lt(x, y), auto_declare=False, log_warn=w) == "(assert (< x y))"
        # 断言小于等于运算的 SMT-LIB 表示正确
        assert smtlib_code(Q.le(x, y), auto_declare=False, log_warn=w) == "(assert (<= x y))"
        # 断言大于运算的 SMT-LIB 表示正确
        assert smtlib_code(Q.gt(x, y), auto_declare=False, log_warn=w) == "(assert (> x y))"
        # 断言大于等于运算的 SMT-LIB 表示正确
        assert smtlib_code(Q.ge(x, y), auto_declare=False, log_warn=w) == "(assert (>= x y))"

    # 断言对复杂运算符调用抛出 ValueError 异常
    raises(ValueError, lambda: smtlib_code(Q.complex(x), log_warn=w))


# 定义测试函数 test_AppliedPredicate，用于测试谓词函数的行为
def test_AppliedPredicate():
    # 使用 _check_warns 上下文管理器来捕获特定警告
    with _check_warns([_W.DEFAULTING_TO_FLOAT] * 6) as w:
        # 断言正数判断的 SMT-LIB 表示正确
        assert smtlib_code(Q.positive(x), auto_declare=False, log_warn=w) == "(assert (> x 0))"
        # 断言负数判断的 SMT-LIB 表示正确
        assert smtlib_code(Q.negative(x), auto_declare=False, log_warn=w) == "(assert (< x 0))"
        # 断言零判断的 SMT-LIB 表示正确
        assert smtlib_code(Q.zero(x), auto_declare=False, log_warn=w) == "(assert (= x 0))"
        # 断言非正数判断的 SMT-LIB 表示正确
        assert smtlib_code(Q.nonpositive(x), auto_declare=False, log_warn=w) == "(assert (<= x 0))"
        # 断言非负数判断的 SMT-LIB 表示正确
        assert smtlib_code(Q.nonnegative(x), auto_declare=False, log_warn=w) == "(assert (>= x 0))"
        # 断言非零判断的 SMT-LIB 表示正确
        assert smtlib_code(Q.nonzero(x), auto_declare=False, log_warn=w) == "(assert (not (= x 0)))"


# 定义测试函数 test_Function，用于测试函数处理的行为
def test_Function():
    # 使用 _check_warns 上下文管理器来捕获特定警告
    with _check_warns([_W.DEFAULTING_TO_FLOAT, _W.WILL_NOT_ASSERT]) as w:
        # 断言函数运算的 SMT-LIB 表示正确
        assert smtlib_code(sin(x) ** cos(x), auto_declare=False, log_warn=w) == "(pow (sin x) (cos x))"

    with _check_warns([_W.WILL_NOT_ASSERT]) as w:
        # 断言绝对值运算的 SMT-LIB 表示正确，并且声明变量和函数
        assert smtlib_code(
            abs(x),
            symbol_table={x: int, y: bool},
            known_types={int: "INTEGER_TYPE"},
            known_functions={sympy.Abs: "ABSOLUTE_VALUE_OF"},
            log_warn=w
        ) == "(declare-const x INTEGER_TYPE)\n" \
             "(ABSOLUTE_VALUE_OF x)"

    # 创建自定义函数对象，并测试其在 SMT-LIB 表示中的行为
    my_fun1 = Function('f1')
    with _check_warns([_W.WILL_NOT_ASSERT]) as w:
        assert smtlib_code(
            my_fun1(x),
            symbol_table={my_fun1: Callable[[bool], float]},
            log_warn=w
        ) == "(declare-const x Bool)\n" \
             "(declare-fun f1 (Bool) Real)\n" \
             "(f1 x)"
    # 使用 _check_warns 函数来检查和处理警告信息，w 是用于记录警告的上下文管理器
    with _check_warns([]) as w:
        # 调用 smtlib_code 函数生成 SMT-LIB 代码，传入 my_fun1(x) 的结果作为参数
        # 使用 symbol_table 参数指定 my_fun1 函数的类型签名，log_warn 参数用于记录警告
        assert smtlib_code(
            my_fun1(x),
            symbol_table={my_fun1: Callable[[bool], bool]},
            log_warn=w
        ) == "(declare-const x Bool)\n" \
             "(declare-fun f1 (Bool) Bool)\n" \
             "(assert (f1 x))"

        # 再次调用 smtlib_code 函数，传入 Eq(my_fun1(x, z), y) 的结果作为参数
        # 使用 symbol_table 参数指定 my_fun1 函数的类型签名，log_warn 参数用于记录警告
        assert smtlib_code(
            Eq(my_fun1(x, z), y),
            symbol_table={my_fun1: Callable[[int, bool], bool]},
            log_warn=w
        ) == "(declare-const x Int)\n" \
             "(declare-const y Bool)\n" \
             "(declare-const z Bool)\n" \
             "(declare-fun f1 (Int Bool) Bool)\n" \
             "(assert (= (f1 x z) y))"

        # 第三次调用 smtlib_code 函数，传入 Eq(my_fun1(x, z), y) 的结果作为参数
        # 使用 symbol_table 参数指定 my_fun1 函数的类型签名，known_functions 参数指定已知函数名映射
        # log_warn 参数用于记录警告
        assert smtlib_code(
            Eq(my_fun1(x, z), y),
            symbol_table={my_fun1: Callable[[int, bool], bool]},
            known_functions={my_fun1: "MY_KNOWN_FUN", Eq: '=='},
            log_warn=w
        ) == "(declare-const x Int)\n" \
             "(declare-const y Bool)\n" \
             "(declare-const z Bool)\n" \
             "(assert (== (MY_KNOWN_FUN x z) y))"

    # 使用 _check_warns 函数来检查和处理警告信息，w 是用于记录警告的上下文管理器
    with _check_warns([_W.DEFAULTING_TO_FLOAT] * 3) as w:
        # 最后一次调用 smtlib_code 函数，传入 Eq(my_fun1(x, z), y) 的结果作为参数
        # 使用 known_functions 参数指定已知函数名映射，log_warn 参数用于记录警告
        assert smtlib_code(
            Eq(my_fun1(x, z), y),
            known_functions={my_fun1: "MY_KNOWN_FUN", Eq: '=='},
            log_warn=w
        ) == "(declare-const x Real)\n" \
             "(declare-const y Real)\n" \
             "(declare-const z Real)\n" \
             "(assert (== (MY_KNOWN_FUN x z) y))"
# 定义一个名为 test_Pow 的测试函数
def test_Pow():
    # 使用 _check_warns 上下文管理器检查警告列表，验证 smtlib_code 的输出是否符合预期
    with _check_warns([_W.DEFAULTING_TO_FLOAT, _W.WILL_NOT_ASSERT]) as w:
        # 断言 smtlib_code 的输出是否等于预期的字符串 "(pow x 3)"
        assert smtlib_code(x ** 3, auto_declare=False, log_warn=w) == "(pow x 3)"

    # 再次使用 _check_warns 上下文管理器检查警告列表，验证 smtlib_code 的输出是否符合预期
    with _check_warns([_W.DEFAULTING_TO_FLOAT, _W.DEFAULTING_TO_FLOAT, _W.WILL_NOT_ASSERT]) as w:
        # 断言 smtlib_code 的输出是否等于预期的字符串 "(pow x (pow y 3))"
        assert smtlib_code(x ** (y ** 3), auto_declare=False, log_warn=w) == "(pow x (pow y 3))"

    # 使用 _check_warns 上下文管理器检查警告列表，验证 smtlib_code 的输出是否符合预期
    with _check_warns([_W.DEFAULTING_TO_FLOAT, _W.WILL_NOT_ASSERT]) as w:
        # 断言 smtlib_code 的输出是否等于预期的字符串 '(pow x (/ 2 3))'
        assert smtlib_code(x ** Rational(2, 3), auto_declare=False, log_warn=w) == '(pow x (/ 2 3))'

        # 创建整数符号 a 和实数符号 b，以及符号 c
        a = Symbol('a', integer=True)
        b = Symbol('b', real=True)
        c = Symbol('c')

        # 定义一个匿名函数 g(x)，其返回值为 2 * x
        def g(x): return 2 * x

        # 计算表达式 expr 的值
        # 若 x=1, y=2，则 expr=1 / (2 * a * 3.5) ** (a - b ** a) / (a ** 2 + b)
        expr = 1 / (g(a) * 3.5) ** (a - b ** a) / (a ** 2 + b)

    # 使用 _check_warns 上下文管理器检查空的警告列表，验证 smtlib_code 的输出是否符合预期
    with _check_warns([]) as w:
        # 断言 smtlib_code 的输出是否等于预期的字符串
        assert smtlib_code(
            [
                Eq(a < 2, c),
                Eq(b > a, c),
                c & True,
                Eq(expr, 2 + Rational(1, 3))
            ],
            log_warn=w
        ) == '(declare-const a Int)\n' \
             '(declare-const b Real)\n' \
             '(declare-const c Bool)\n' \
             '(assert (= (< a 2) c))\n' \
             '(assert (= (> b a) c))\n' \
             '(assert c)\n' \
             '(assert (= ' \
             '(* (pow (* 7.0 a) (+ (pow b a) (* -1 a))) (pow (+ b (pow a 2)) -1)) ' \
             '(/ 7 3)' \
             '))'

    # 再次使用 _check_warns 上下文管理器检查警告列表，验证 smtlib_code 的输出是否符合预期
    with _check_warns([_W.DEFAULTING_TO_FLOAT, _W.WILL_NOT_ASSERT]) as w:
        # 断言 smtlib_code 的输出是否等于预期的字符串
        assert smtlib_code(
            Mul(-2, c, Pow(Mul(b, b, evaluate=False), -1, evaluate=False), evaluate=False),
            log_warn=w
        ) == '(declare-const b Real)\n' \
             '(declare-const c Real)\n' \
             '(* -2 c (pow (* b b) -1))'


# 定义一个名为 test_basic_ops 的测试函数
def test_basic_ops():
    # 使用 _check_warns 上下文管理器检查警告列表，验证 smtlib_code 的输出是否符合预期
    with _check_warns([_W.DEFAULTING_TO_FLOAT, _W.DEFAULTING_TO_FLOAT, _W.WILL_NOT_ASSERT]) as w:
        # 断言 smtlib_code 的输出是否等于预期的字符串 "(* x y)"
        assert smtlib_code(x * y, auto_declare=False, log_warn=w) == "(* x y)"

    # 再次使用 _check_warns 上下文管理器检查警告列表，验证 smtlib_code 的输出是否符合预期
    with _check_warns([_W.DEFAULTING_TO_FLOAT, _W.DEFAULTING_TO_FLOAT, _W.WILL_NOT_ASSERT]) as w:
        # 断言 smtlib_code 的输出是否等于预期的字符串 "(+ x y)"
        assert smtlib_code(x + y, auto_declare=False, log_warn=w) == "(+ x y)"

    # 注释掉的部分是待实现的代码重写，当前输出为 '(+ x (* -1 y))'
    # todo: implement re-write, currently does '(+ x (* -1 y))' instead
    # assert smtlib_code(x - y, auto_declare=False, log_warn=w) == "(- x y)"

    # 使用 _check_warns 上下文管理器检查警告列表，验证 smtlib_code 的输出是否符合预期
    with _check_warns([_W.DEFAULTING_TO_FLOAT, _W.WILL_NOT_ASSERT]) as w:
        # 断言 smtlib_code 的输出是否等于预期的字符串 "(* -1 x)"
        assert smtlib_code(-x, auto_declare=False, log_warn=w) == "(* -1 x)"


# 定义一个名为 test_quantifier_extensions 的测试函数
def test_quantifier_extensions():
    # 导入需要使用的模块和函数
    from sympy.logic.boolalg import Boolean
    from sympy import Interval, Tuple, sympify

    # 开始一个 For-all 量化器类的示例
    # 定义 ForAll 类，继承自 Boolean 类
    class ForAll(Boolean):
        
        # 生成 SMT-LIB 格式的量词逻辑表达式
        def _smtlib(self, printer):
            # 生成绑定符号声明的表达式列表
            bound_symbol_declarations = [
                printer._s_expr(sym.name, [
                    printer._known_types[printer.symbol_table[sym]],
                    Interval(start, end)
                ]) for sym, start, end in self.limits
            ]
            # 返回量词逻辑表达式
            return printer._s_expr('forall', [
                printer._s_expr('', bound_symbol_declarations),
                self.function
            ])

        # 返回绑定的符号集合
        @property
        def bound_symbols(self):
            return {s for s, _, _ in self.limits}

        # 返回自由符号集合
        @property
        def free_symbols(self):
            # 获取已绑定符号名称的集合
            bound_symbol_names = {s.name for s in self.bound_symbols}
            # 返回不在绑定符号名称集合中的自由符号集合
            return {
                s for s in self.function.free_symbols
                if s.name not in bound_symbol_names
            }

        # 构造函数，根据参数创建 ForAll 对象
        def __new__(cls, *args):
            # 提取限制条件
            limits = [sympify(a) for a in args if isinstance(a, (tuple, Tuple))]
            # 提取布尔函数
            function = [sympify(a) for a in args if isinstance(a, Boolean)]
            # 断言限制条件和布尔函数的总数等于参数总数
            assert len(limits) + len(function) == len(args)
            # 断言只有一个布尔函数
            assert len(function) == 1
            # 获取布尔函数
            function = function[0]

            # 如果布尔函数是 ForAll 类的实例，则合并限制条件并返回新的 ForAll 对象
            if isinstance(function, ForAll): return ForAll.__new__(
                ForAll, *(limits + function.limits), function.function
            )
            # 否则创建新的 Boolean 对象并返回
            inst = Boolean.__new__(cls)
            inst._args = tuple(limits + [function])
            inst.limits = limits
            inst.function = function
            return inst

    # For-All 量词类示例结束

    # 创建函数 f
    f = Function('f')

    # 使用 _check_warns 上下文，检查警告 _W.DEFAULTING_TO_FLOAT
    with _check_warns([_W.DEFAULTING_TO_FLOAT]) as w:
        # 断言 SMT-LIB 代码与预期输出相符
        assert smtlib_code(
            ForAll((x, -42, +21), Eq(f(x), f(x))),
            symbol_table={f: Callable[[float], float]},
            log_warn=w
        ) == '(assert (forall ( (x Real [-42, 21])) true))'

    # 使用 _check_warns 上下文，检查多个警告 _W.DEFAULTING_TO_FLOAT
    with _check_warns([_W.DEFAULTING_TO_FLOAT] * 2) as w:
        # 断言 SMT-LIB 代码与预期输出相符
        assert smtlib_code(
            ForAll(
                (x, -42, +21), (y, -100, 3),
                Implies(Eq(x, y), Eq(f(x), f(y)))
            ),
            symbol_table={f: Callable[[float], float]},
            log_warn=w
        ) == '(declare-fun f (Real) Real)\n' \
             '(assert (' \
             'forall ( (x Real [-42, 21]) (y Real [-100, 3])) ' \
             '(=> (= x y) (= (f x) (f y)))' \
             '))'

    # 创建整数符号 a、实数符号 b、符号 c
    a = Symbol('a', integer=True)
    b = Symbol('b', real=True)
    c = Symbol('c')

    # 使用 _check_warns 上下文，不检查任何警告
    with _check_warns([]) as w:
        # 断言 SMT-LIB 代码与预期输出相符
        assert smtlib_code(
            ForAll(
                (a, 2, 100), ForAll(
                    (b, 2, 100),
                    Implies(a < b, sqrt(a) < b) | c
                )),
            log_warn=w
        ) == '(declare-const c Bool)\n' \
             '(assert (forall ( (a Int [2, 100]) (b Real [2, 100])) ' \
             '(or c (=> (< a b) (< (pow a (/ 1 2)) b)))' \
             '))'
# 定义一个测试函数，测试混合数字和符号的情况
def test_mix_number_mult_symbols():
    # 运行时检查警告信息，并期望不会触发某些警告
    with _check_warns([_W.WILL_NOT_ASSERT]) as w:
        # 调用函数 smtlib_code，生成 SMT-LIB 格式的代码
        assert smtlib_code(
            1 / pi,  # 以 pi 的倒数作为输入
            known_constants={pi: "MY_PI"},  # 已知常数 pi 的别名为 MY_PI
            log_warn=w  # 将警告信息传递给 log_warn 参数
        ) == '(pow MY_PI -1)'  # 预期生成的 SMT-LIB 代码

    with _check_warns([_W.WILL_NOT_ASSERT]) as w:
        # 调用函数 smtlib_code，生成 SMT-LIB 格式的代码
        assert smtlib_code(
            [
                Eq(pi, 3.14, evaluate=False),  # 添加一个方程式 pi = 3.14
                1 / pi,  # 以 pi 的倒数作为输入
            ],
            known_constants={pi: "MY_PI"},  # 已知常数 pi 的别名为 MY_PI
            log_warn=w  # 将警告信息传递给 log_warn 参数
        ) == '(assert (= MY_PI 3.14))\n' \
             '(pow MY_PI -1)'  # 预期生成的 SMT-LIB 代码

    with _check_warns([_W.WILL_NOT_ASSERT]) as w:
        # 调用函数 smtlib_code，生成 SMT-LIB 格式的代码
        assert smtlib_code(
            Add(S.Zero, S.One, S.NegativeOne, S.Half,
                S.Exp1, S.Pi, S.GoldenRatio, evaluate=False),  # 添加多个表达式
            known_constants={  # 已知常数及其别名
                S.Pi: 'p', S.GoldenRatio: 'g',
                S.Exp1: 'e'
            },
            known_functions={  # 已知函数及其别名
                Add: 'plus',
                exp: 'exp'
            },
            precision=3,  # 设置精度为 3
            log_warn=w  # 将警告信息传递给 log_warn 参数
        ) == '(plus 0 1 -1 (/ 1 2) (exp 1) p g)'  # 预期生成的 SMT-LIB 代码

    with _check_warns([_W.WILL_NOT_ASSERT]) as w:
        # 调用函数 smtlib_code，生成 SMT-LIB 格式的代码
        assert smtlib_code(
            Add(S.Zero, S.One, S.NegativeOne, S.Half,
                S.Exp1, S.Pi, S.GoldenRatio, evaluate=False),  # 添加多个表达式
            known_constants={S.Pi: 'p'},  # 已知常数 pi 的别名为 p
            known_functions={  # 已知函数及其别名
                Add: 'plus',
                exp: 'exp'
            },
            precision=3,  # 设置精度为 3
            log_warn=w  # 将警告信息传递给 log_warn 参数
        ) == '(plus 0 1 -1 (/ 1 2) (exp 1) p 1.62)'  # 预期生成的 SMT-LIB 代码

    with _check_warns([_W.WILL_NOT_ASSERT]) as w:
        # 调用函数 smtlib_code，生成 SMT-LIB 格式的代码
        assert smtlib_code(
            Add(S.Zero, S.One, S.NegativeOne, S.Half,
                S.Exp1, S.Pi, S.GoldenRatio, evaluate=False),  # 添加多个表达式
            known_functions={Add: 'plus'},  # 已知函数 Add 的别名为 plus
            precision=3,  # 设置精度为 3
            log_warn=w  # 将警告信息传递给 log_warn 参数
        ) == '(plus 0 1 -1 (/ 1 2) 2.72 3.14 1.62)'  # 预期生成的 SMT-LIB 代码

    with _check_warns([_W.WILL_NOT_ASSERT]) as w:
        # 调用函数 smtlib_code，生成 SMT-LIB 格式的代码
        assert smtlib_code(
            Add(S.Zero, S.One, S.NegativeOne, S.Half,
                S.Exp1, S.Pi, S.GoldenRatio, evaluate=False),  # 添加多个表达式
            known_constants={S.Exp1: 'e'},  # 已知常数 e 的别名为 e
            known_functions={Add: 'plus'},  # 已知函数 Add 的别名为 plus
            precision=3,  # 设置精度为 3
            log_warn=w  # 将警告信息传递给 log_warn 参数
        ) == '(plus 0 1 -1 (/ 1 2) e 3.14 1.62)'  # 预期生成的 SMT-LIB 代码
    # 使用 _check_warns 函数检查警告，并将结果赋值给变量 w
    with _check_warns([]) as w:
        # 断言条件：生成 x & y 的 SMT-LIB 代码，并检查是否与预期字符串相等
        assert smtlib_code(x & y, log_warn=w) == '(declare-const x Bool)\n' \
                                                 '(declare-const y Bool)\n' \
                                                 '(assert (and x y))'
        # 断言条件：生成 x | y 的 SMT-LIB 代码，并检查是否与预期字符串相等
        assert smtlib_code(x | y, log_warn=w) == '(declare-const x Bool)\n' \
                                                 '(declare-const y Bool)\n' \
                                                 '(assert (or x y))'
        # 断言条件：生成 ~x 的 SMT-LIB 代码，并检查是否与预期字符串相等
        assert smtlib_code(~x, log_warn=w) == '(declare-const x Bool)\n' \
                                              '(assert (not x))'
        # 断言条件：生成 x & y & z 的 SMT-LIB 代码，并检查是否与预期字符串相等
        assert smtlib_code(x & y & z, log_warn=w) == '(declare-const x Bool)\n' \
                                                     '(declare-const y Bool)\n' \
                                                     '(declare-const z Bool)\n' \
                                                     '(assert (and x y z))'

    # 使用 _check_warns 函数检查警告，并将特定警告 _W.DEFAULTING_TO_FLOAT 添加到变量 w
    with _check_warns([_W.DEFAULTING_TO_FLOAT]) as w:
        # 断言条件：生成 (x & ~y) | (z > 3) 的 SMT-LIB 代码，并检查是否与预期字符串相等
        assert smtlib_code((x & ~y) | (z > 3), log_warn=w) == '(declare-const x Bool)\n' \
                                                              '(declare-const y Bool)\n' \
                                                              '(declare-const z Real)\n' \
                                                              '(assert (or (> z 3) (and x (not y))))'

    # 创建函数 f, g, h，并使用 _check_warns 函数检查警告，并将特定警告 _W.DEFAULTING_TO_FLOAT 添加到变量 w
    f = Function('f')
    g = Function('g')
    h = Function('h')
    with _check_warns([_W.DEFAULTING_TO_FLOAT]) as w:
        # 断言条件：生成 [Gt(f(x), y), Lt(y, g(z))] 的 SMT-LIB 代码，并检查是否与预期字符串相等
        # 同时提供符号表 symbol_table 用于描述函数的签名
        assert smtlib_code(
            [Gt(f(x), y),
             Lt(y, g(z))],
            symbol_table={
                f: Callable[[bool], int], g: Callable[[bool], int],
            }, log_warn=w
        ) == '(declare-const x Bool)\n' \
             '(declare-const y Real)\n' \
             '(declare-const z Bool)\n' \
             '(declare-fun f (Bool) Int)\n' \
             '(declare-fun g (Bool) Int)\n' \
             '(assert (> (f x) y))\n' \
             '(assert (< y (g z)))'

    # 使用 _check_warns 函数检查警告，并将结果赋值给变量 w
    with _check_warns([]) as w:
        # 断言条件：生成 [Eq(f(x), y), Lt(y, g(z))] 的 SMT-LIB 代码，并检查是否与预期字符串相等
        # 同时提供符号表 symbol_table 用于描述函数的签名
        assert smtlib_code(
            [Eq(f(x), y),
             Lt(y, g(z))],
            symbol_table={
                f: Callable[[bool], int], g: Callable[[bool], int],
            }, log_warn=w
        ) == '(declare-const x Bool)\n' \
             '(declare-const y Int)\n' \
             '(declare-const z Bool)\n' \
             '(declare-fun f (Bool) Int)\n' \
             '(declare-fun g (Bool) Int)\n' \
             '(assert (= (f x) y))\n' \
             '(assert (< y (g z)))'
    # 使用 _check_warns 上下文管理器来检查警告列表
    with _check_warns([]) as w:
        # 断言 smtlib_code 函数的返回结果
        assert smtlib_code(
            # 提供一个包含三个等式的列表
            [Eq(f(x), y),  # 表示 f(x) == y
             Eq(g(f(x)), z),  # 表示 g(f(x)) == z
             Eq(h(g(f(x))), x)],  # 表示 h(g(f(x))) == x
            # 提供符号表，映射函数到其类型的可调用类型
            symbol_table={
                f: Callable[[float], int],  # f 是从 Real 到 Int 的函数
                g: Callable[[int], bool],   # g 是从 Int 到 Bool 的函数
                h: Callable[[bool], float]  # h 是从 Bool 到 Real 的函数
            },
            log_warn=w  # 将警告日志传递给 smtlib_code 函数
        ) == '(declare-const x Real)\n' \  # 断言结果字符串的第一行
             '(declare-const y Int)\n' \   # 断言结果字符串的第二行
             '(declare-const z Bool)\n' \  # 断言结果字符串的第三行
             '(declare-fun f (Real) Int)\n' \  # 断言结果字符串的第四行
             '(declare-fun g (Int) Bool)\n' \  # 断言结果字符串的第五行
             '(declare-fun h (Bool) Real)\n' \  # 断言结果字符串的第六行
             '(assert (= (f x) y))\n' \   # 断言结果字符串的第七行
             '(assert (= (g (f x)) z))\n' \  # 断言结果字符串的第八行
             '(assert (= (h (g (f x))) x))'  # 断言结果字符串的最后一行
# todo: make smtlib_code support arrays
# def test_containers():
#     assert julia_code([1, 2, 3, [4, 5, [6, 7]], 8, [9, 10], 11]) == \
#            "Any[1, 2, 3, Any[4, 5, Any[6, 7]], 8, Any[9, 10], 11]"
#     assert julia_code((1, 2, (3, 4))) == "(1, 2, (3, 4))"
#     assert julia_code([1]) == "Any[1]"
#     assert julia_code((1,)) == "(1,)"
#     assert julia_code(Tuple(*[1, 2, 3])) == "(1, 2, 3)"
#     assert julia_code((1, x * y, (3, x ** 2))) == "(1, x .* y, (3, x .^ 2))"
#     # scalar, matrix, empty matrix and empty list
#     assert julia_code((1, eye(3), Matrix(0, 0, []), [])) == "(1, [1 0 0;\n0 1 0;\n0 0 1], zeros(0, 0), Any[])"

# todo: make smtlib_code support arrays / matrices ?
# def test_smtlib_matrix_assign_to():
#     A = Matrix([[1, 2, 3]])
#     assert smtlib_code(A, assign_to='a') == "a = [1 2 3]"
#     A = Matrix([[1, 2], [3, 4]])
#     assert smtlib_code(A, assign_to='A') == "A = [1 2;\n3 4]"

# def test_julia_matrix_1x1():
#     A = Matrix([[3]])
#     B = MatrixSymbol('B', 1, 1)

def test_smtlib_piecewise():
    # 检查警告信息，确保默认转换为浮点数和不会断言
    with _check_warns([_W.DEFAULTING_TO_FLOAT, _W.WILL_NOT_ASSERT]) as w:
        # 测试 Piecewise 函数的 SMT-LIB 代码生成
        assert smtlib_code(
            Piecewise((x, x < 1),
                      (x ** 2, True)),
            auto_declare=False,
            log_warn=w
        ) == '(ite (< x 1) x (pow x 2))'

    with _check_warns([_W.DEFAULTING_TO_FLOAT, _W.WILL_NOT_ASSERT]) as w:
        # 测试多条件的 Piecewise 函数的 SMT-LIB 代码生成
        assert smtlib_code(
            Piecewise((x ** 2, x < 1),
                      (x ** 3, x < 2),
                      (x ** 4, x < 3),
                      (x ** 5, True)),
            auto_declare=False,
            log_warn=w
        ) == '(ite (< x 1) (pow x 2) ' \
             '(ite (< x 2) (pow x 3) ' \
             '(ite (< x 3) (pow x 4) ' \
             '(pow x 5))))'

    # 检查 Piecewise 函数没有 True（默认）条件时的错误
    expr = Piecewise((x, x < 1), (x ** 2, x > 1), (sin(x), x > 0))
    with _check_warns([_W.DEFAULTING_TO_FLOAT, _W.WILL_NOT_ASSERT]) as w:
        raises(AssertionError, lambda: smtlib_code(expr, log_warn=w))


def test_smtlib_piecewise_times_const():
    pw = Piecewise((x, x < 1), (x ** 2, True))
    with _check_warns([_W.DEFAULTING_TO_FLOAT, _W.WILL_NOT_ASSERT]) as w:
        # 测试常数乘以 Piecewise 函数的 SMT-LIB 代码生成
        assert smtlib_code(2 * pw, log_warn=w) == '(declare-const x Real)\n(* 2 (ite (< x 1) x (pow x 2)))'
    with _check_warns([_W.DEFAULTING_TO_FLOAT, _W.WILL_NOT_ASSERT]) as w:
        # 测试变量除以 Piecewise 函数的 SMT-LIB 代码生成
        assert smtlib_code(pw / x, log_warn=w) == '(declare-const x Real)\n(* (pow x -1) (ite (< x 1) x (pow x 2)))'
    with _check_warns([_W.DEFAULTING_TO_FLOAT, _W.DEFAULTING_TO_FLOAT, _W.WILL_NOT_ASSERT]) as w:
        # 测试两个变量相乘再除以 Piecewise 函数的 SMT-LIB 代码生成
        assert smtlib_code(pw / (x * y), log_warn=w) == '(declare-const x Real)\n(declare-const y Real)\n(* (pow x -1) (pow y -1) (ite (< x 1) x (pow x 2)))'
    with _check_warns([_W.DEFAULTING_TO_FLOAT, _W.WILL_NOT_ASSERT]) as w:
        # 测试常数除以 Piecewise 函数的 SMT-LIB 代码生成
        assert smtlib_code(pw / 3, log_warn=w) == '(declare-const x Real)\n(* (/ 1 3) (ite (< x 1) x (pow x 2)))'
# 定义一个测试函数，用于测试生成 SMT-LIB 布尔表达式的代码生成器
def test_smtlib_boolean():
    # 测试生成 true 的 SMT-LIB 表达式，不自动添加断言，验证输出是否为 'true'
    with _check_warns([]) as w:
        assert smtlib_code(True, auto_assert=False, log_warn=w) == 'true'
    # 测试生成 true 的 SMT-LIB 断言表达式，验证输出是否为 '(assert true)'
    assert smtlib_code(True, log_warn=w) == '(assert true)'
    # 使用 SymPy 的 S.true 测试生成 true 的 SMT-LIB 断言表达式，验证输出是否为 '(assert true)'
    assert smtlib_code(S.true, log_warn=w) == '(assert true)'
    # 使用 SymPy 的 S.false 测试生成 false 的 SMT-LIB 断言表达式，验证输出是否为 '(assert false)'
    assert smtlib_code(S.false, log_warn=w) == '(assert false)'
    # 测试生成 false 的 SMT-LIB 断言表达式，验证输出是否为 '(assert false)'
    assert smtlib_code(False, log_warn=w) == '(assert false)'
    # 测试生成 false 的 SMT-LIB 表达式，不自动添加断言，验证输出是否为 'false'
    assert smtlib_code(False, auto_assert=False, log_warn=w) == 'false'


# 定义一个测试函数，用于测试在某些情况下代码生成器不支持的功能
def test_not_supported():
    # 创建一个函数 f(x)，并使用 Callable 类型的符号表定义
    f = Function('f')
    # 测试当生成 f(x).diff(x) 的 SMT-LIB 表达式时，预期引发 KeyError 异常，
    # 并捕获默认情况下会转换为浮点数和不会生成断言的警告
    with _check_warns([_W.DEFAULTING_TO_FLOAT, _W.WILL_NOT_ASSERT]) as w:
        raises(KeyError, lambda: smtlib_code(f(x).diff(x), symbol_table={f: Callable[[float], float]}, log_warn=w))
    # 测试当生成 SymPy 的 ComplexInfinity 的 SMT-LIB 表达式时，预期引发 KeyError 异常，
    # 并捕获不会生成断言的警告
    with _check_warns([_W.WILL_NOT_ASSERT]) as w:
        raises(KeyError, lambda: smtlib_code(S.ComplexInfinity, log_warn=w))


# 定义一个测试函数，用于测试生成浮点数的 SMT-LIB 表达式的代码生成器
def test_Float():
    # 测试生成浮点数 0.0 的 SMT-LIB 表达式，验证输出是否为 "0.0"
    assert smtlib_code(0.0) == "0.0"
    # 测试生成浮点数 0.000000000000000003 的 SMT-LIB 表达式，
    # 验证输出是否为 '(* 3.0 (pow 10 -18))'
    assert smtlib_code(0.000000000000000003) == '(* 3.0 (pow 10 -18))'
    # 测试生成浮点数 5.3 的 SMT-LIB 表达式，验证输出是否为 "5.3"
    assert smtlib_code(5.3) == "5.3"
```