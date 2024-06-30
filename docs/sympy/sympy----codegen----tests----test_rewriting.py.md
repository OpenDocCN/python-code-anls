# `D:\src\scipysrc\sympy\sympy\codegen\tests\test_rewriting.py`

```
# 导入临时文件模块
import tempfile
# 导入 SymPy 中的数学常数 pi 和有理数类型 Rational
from sympy.core.numbers import pi, Rational
# 导入 SymPy 中的幂操作 Pow 类
from sympy.core.power import Pow
# 导入 SymPy 中的单例对象 S
from sympy.core.singleton import S
# 导入 SymPy 中的符号变量 Symbol
from sympy.core.symbol import Symbol
# 导入 SymPy 中的复数函数 Abs
from sympy.functions.elementary.complexes import Abs
# 导入 SymPy 中的指数函数 exp 和对数函数 log
from sympy.functions.elementary.exponential import (exp, log)
# 导入 SymPy 中的三角函数 cos, sin, sinc
from sympy.functions.elementary.trigonometric import (cos, sin, sinc)
# 导入 SymPy 中矩阵表达式相关的 MatrixSymbol
from sympy.matrices.expressions.matexpr import MatrixSymbol
# 导入 SymPy 中的假设处理模块 assuming 和 Q
from sympy.assumptions import assuming, Q
# 导入 SymPy 中的外部模块导入函数 import_module
from sympy.external import import_module
# 导入 SymPy 中的代码打印器 ccode
from sympy.printing.codeprinter import ccode
# 导入 SymPy 中的矩阵操作节点 MatrixSolve
from sympy.codegen.matrix_nodes import MatrixSolve
# 导入 SymPy 中的 C 语言函数相关的函数 log2, exp2, expm1, log1p
from sympy.codegen.cfunctions import log2, exp2, expm1, log1p
# 导入 SymPy 中的 NumPy 节点相关的函数 logaddexp, logaddexp2
from sympy.codegen.numpy_nodes import logaddexp, logaddexp2
# 导入 SymPy 中的 SciPy 节点相关的函数 cosm1, powm1
from sympy.codegen.scipy_nodes import cosm1, powm1
# 导入 SymPy 中的重写优化相关的函数和类 optimize, 各种优化函数（略）
from sympy.codegen.rewriting import (
    optimize, cosm1_opt, log2_opt, exp2_opt, expm1_opt, log1p_opt, powm1_opt, optims_c99,
    create_expand_pow_optimization, matinv_opt, logaddexp_opt, logaddexp2_opt,
    optims_numpy, optims_scipy, sinc_opts, FuncMinusOneOptim
)
# 导入 SymPy 中的测试相关函数 XFAIL, skip
from sympy.testing.pytest import XFAIL, skip
# 导入 SymPy 中的 lambdify 函数
from sympy.utilities import lambdify
# 导入 SymPy 中的编译和链接字符串的函数 compile_link_import_strings, 检查 C 编译器的函数 has_c
from sympy.utilities._compilation import compile_link_import_strings, has_c
# 导入 SymPy 中的可能会失败的测试标记函数 may_xfail
from sympy.utilities._compilation.util import may_xfail

# 使用 import_module 函数导入 Cython, NumPy, SciPy 模块
cython = import_module('cython')
numpy = import_module('numpy')
scipy = import_module('scipy')


# 定义测试函数 test_log2_opt
def test_log2_opt():
    # 创建符号变量 x
    x = Symbol('x')
    # 定义表达式 expr1
    expr1 = 7*log(3*x + 5)/(log(2))
    # 对 expr1 进行 log2 优化
    opt1 = optimize(expr1, [log2_opt])
    # 断言优化后的表达式 opt1
    assert opt1 == 7*log2(3*x + 5)
    # 断言 opt1 被重写为 log
    assert opt1.rewrite(log) == expr1

    # 定义表达式 expr2
    expr2 = 3*log(5*x + 7)/(13*log(2))
    # 对 expr2 进行 log2 优化
    opt2 = optimize(expr2, [log2_opt])
    # 断言优化后的表达式 opt2
    assert opt2 == 3*log2(5*x + 7)/13
    # 断言 opt2 被重写为 log
    assert opt2.rewrite(log) == expr2

    # 定义表达式 expr3
    expr3 = log(x)/log(2)
    # 对 expr3 进行 log2 优化
    opt3 = optimize(expr3, [log2_opt])
    # 断言优化后的表达式 opt3
    assert opt3 == log2(x)
    # 断言 opt3 被重写为 log
    assert opt3.rewrite(log) == expr3

    # 定义表达式 expr4
    expr4 = log(x)/log(2) + log(x+1)
    # 对 expr4 进行 log2 优化
    opt4 = optimize(expr4, [log2_opt])
    # 断言优化后的表达式 opt4
    assert opt4 == log2(x) + log(2)*log2(x+1)
    # 断言 opt4 被重写为 log
    assert opt4.rewrite(log) == expr4

    # 定义表达式 expr5
    expr5 = log(17)
    # 对 expr5 进行 log2 优化
    opt5 = optimize(expr5, [log2_opt])
    # 断言优化后的表达式 opt5
    assert opt5 == expr5

    # 定义表达式 expr6
    expr6 = log(x + 3)/log(2)
    # 对 expr6 进行 log2 优化
    opt6 = optimize(expr6, [log2_opt])
    # 断言优化后的表达式 opt6
    assert str(opt6) == 'log2(x + 3)'
    # 断言 opt6 被重写为 log
    assert opt6.rewrite(log) == expr6


# 定义测试函数 test_exp2_opt
def test_exp2_opt():
    # 创建符号变量 x
    x = Symbol('x')
    # 定义表达式 expr1
    expr1 = 1 + 2**x
    # 对 expr1 进行 exp2 优化
    opt1 = optimize(expr1, [exp2_opt])
    # 断言优化后的表达式 opt1
    assert opt1 == 1 + exp2(x)
    # 断言 opt1 被重写为 Pow
    assert opt1.rewrite(Pow) == expr1

    # 定义表达式 expr2
    expr2 = 1 + 3**x
    # 断言 expr2 与其进行 exp2 优化后的结果相等
    assert expr2 == optimize(expr2, [exp2_opt])


# 定义测试函数 test_expm1_opt
def test_expm1_opt():
    # 创建符号变量 x
    x = Symbol('x')

    # 定义表达式 expr1
    expr1 = exp(x) - 1
    # 对 expr1 进行 expm1 优化
    opt1 = optimize(expr1, [expm1_opt])
    # 断言 expm1(x) 减去优化后的表达式 opt1 等于零
    assert expm1(x) - opt1 == 0
    # 断言 opt1 被重写为 exp
    assert opt1.rewrite(exp) == expr1

    # 定义表达式 expr2
    expr2 = 3*exp(x) - 3
    # 对 expr2 进行 expm1 优化
    opt2 = optimize(expr2, [expm1_opt])
    # 断言 3*expm1(x) 等于优化后的表达式 opt2
    assert 3*expm1(x) == opt2
    # 断言 opt2 被重写为 exp
    assert opt2.rewrite(exp) == expr2

    # 定义表达式 expr3
    expr3 = 3*exp(x) - 5
    # 对 expr3 进行 expm1 优化
    opt3 = optimize(expr3, [expm1_opt])
    # 断言 3*expm1(x) 减去 2 等于优化后的表达式 opt3
    assert 3*expm1(x) - 2 == opt3
    # 断言 opt3 被重写为 exp
    assert opt3.rewrite(exp) ==
    # 确保表达式 expr3 通过给定的优化器 optimize 函数处理后与其自身相等
    assert expr3 == optimize(expr3, [expm1_opt_non_opportunistic])
    # 确保表达式 expr1 通过给定的优化器 optimize 函数处理后与 opt1 相等
    assert opt1 == optimize(expr1, [expm1_opt_non_opportunistic])
    # 确保表达式 expr2 通过给定的优化器 optimize 函数处理后与 opt2 相等
    assert opt2 == optimize(expr2, [expm1_opt_non_opportunistic])

    # 定义表达式 expr4
    expr4 = 3*exp(x) + log(x) - 3
    # 使用优化器 expm1_opt 处理 expr4，并将结果赋给 opt4
    opt4 = optimize(expr4, [expm1_opt])
    # 确保优化后的表达式 opt4 等于原始表达式的期望形式
    assert 3*expm1(x) + log(x) == opt4
    # 确保 opt4 可以通过 rewrite(exp) 方法恢复为原始表达式 expr4
    assert opt4.rewrite(exp) == expr4

    # 定义表达式 expr5
    expr5 = 3*exp(2*x) - 3
    # 使用优化器 expm1_opt 处理 expr5，并将结果赋给 opt5
    opt5 = optimize(expr5, [expm1_opt])
    # 确保优化后的表达式 opt5 等于原始表达式的期望形式
    assert 3*expm1(2*x) == opt5
    # 确保 opt5 可以通过 rewrite(exp) 方法恢复为原始表达式 expr5
    assert opt5.rewrite(exp) == expr5

    # 定义表达式 expr6
    expr6 = (2*exp(x) + 1)/(exp(x) + 1) + 1
    # 使用优化器 expm1_opt 处理 expr6，并将结果赋给 opt6
    opt6 = optimize(expr6, [expm1_opt])
    # 确保 opt6 的操作数数量不大于 expr6 的操作数数量
    assert opt6.count_ops() <= expr6.count_ops()

    # 定义函数 ev(e)，将变量 x 替换为 3 后，对表达式 e 求值
    def ev(e):
        return e.subs(x, 3).evalf()
    # 确保对于表达式 expr6 和其优化后的形式 opt6，在 x=3 时的数值接近
    assert abs(ev(expr6) - ev(opt6)) < 1e-15

    # 创建符号变量 y
    y = Symbol('y')
    # 定义表达式 expr7
    expr7 = (2*exp(x) - 1)/(1 - exp(y)) - 1/(1-exp(y))
    # 使用优化器 expm1_opt 处理 expr7，并将结果赋给 opt7
    opt7 = optimize(expr7, [expm1_opt])
    # 确保优化后的表达式 opt7 等于原始表达式的期望形式
    assert -2*expm1(x)/expm1(y) == opt7
    # 确保 opt7 通过 rewrite(exp) 方法后与 expr7 的差异因式分解为 0
    assert (opt7.rewrite(exp) - expr7).factor() == 0

    # 定义表达式 expr8
    expr8 = (1+exp(x))**2 - 4
    # 使用优化器 expm1_opt 处理 expr8，并将结果赋给 opt8
    opt8 = optimize(expr8, [expm1_opt])
    # 定义两个可能的目标形式 tgt8a 和 tgt8b，通过比较其 rewrite(exp) 后的差异因式分解为 0 来确定优选版本
    tgt8a = (exp(x) + 3)*expm1(x)
    tgt8b = 2*expm1(x) + expm1(2*x)
    # 确保 tgt8a 和 tgt8b 的 rewrite(exp) 后的差异因式分解为 0，表明二者等价
    assert (tgt8a - tgt8b).rewrite(exp).factor() == 0
    # 确保 opt8 是 tgt8a 或 tgt8b 中的一个
    assert opt8 in (tgt8a, tgt8b)
    # 确保 opt8 通过 rewrite(exp) 方法后与 expr8 的差异因式分解为 0
    assert (opt8.rewrite(exp) - expr8).factor() == 0

    # 定义表达式 expr9
    expr9 = sin(expr8)
    # 使用优化器 expm1_opt 处理 expr9，并将结果赋给 opt9
    opt9 = optimize(expr9, [expm1_opt])
    # 定义两个可能的目标形式 tgt9a 和 tgt9b，确保 opt9 是其中一个
    tgt9a = sin(tgt8a)
    tgt9b = sin(tgt8b)
    assert opt9 in (tgt9a, tgt9b)
    # 确保 opt9 和 expr9 经 rewrite(exp) 后的形式相等，并且差异因式分解为 0
    assert (opt9.rewrite(exp) - expr9.rewrite(exp)).factor().is_zero
def test_expm1_two_exp_terms():
    # 定义符号变量 x 和 y
    x, y = map(Symbol, 'x y'.split())
    # 构建表达式 exp(x) + exp(y) - 2
    expr1 = exp(x) + exp(y) - 2
    # 对表达式进行优化，使用 expm1_opt 优化器
    opt1 = optimize(expr1, [expm1_opt])
    # 断言优化后的结果等于 expm1(x) + expm1(y)
    assert opt1 == expm1(x) + expm1(y)


def test_cosm1_opt():
    # 定义符号变量 x
    x = Symbol('x')

    # 表达式1: cos(x) - 1
    expr1 = cos(x) - 1
    # 对表达式1进行优化，使用 cosm1_opt 优化器
    opt1 = optimize(expr1, [cosm1_opt])
    # 断言优化后的结果等于 cosm1(x) - opt1 == 0
    assert cosm1(x) - opt1 == 0
    # 断言优化后的结果等于原始表达式 expr1
    assert opt1.rewrite(cos) == expr1

    # 表达式2: 3*cos(x) - 3
    expr2 = 3*cos(x) - 3
    # 对表达式2进行优化，使用 cosm1_opt 优化器
    opt2 = optimize(expr2, [cosm1_opt])
    # 断言优化后的结果等于 3*cosm1(x)
    assert 3*cosm1(x) == opt2
    # 断言优化后的结果等于原始表达式 expr2
    assert opt2.rewrite(cos) == expr2

    # 表达式3: 3*cos(x) - 5
    expr3 = 3*cos(x) - 5
    # 对表达式3进行优化，使用 cosm1_opt 优化器
    opt3 = optimize(expr3, [cosm1_opt])
    # 断言优化后的结果等于 3*cosm1(x) - 2
    assert 3*cosm1(x) - 2 == opt3
    # 断言优化后的结果等于原始表达式 expr3
    assert opt3.rewrite(cos) == expr3

    # 创建非机会性的 cosm1_opt 优化器
    cosm1_opt_non_opportunistic = FuncMinusOneOptim(cos, cosm1, opportunistic=False)
    # 断言原始表达式 expr3 等于使用非机会性优化器优化的结果
    assert expr3 == optimize(expr3, [cosm1_opt_non_opportunistic])
    # 断言表达式1使用非机会性优化器优化后等于 opt1
    assert opt1 == optimize(expr1, [cosm1_opt_non_opportunistic])
    # 断言表达式2使用非机会性优化器优化后等于 opt2
    assert opt2 == optimize(expr2, [cosm1_opt_non_opportunistic])

    # 表达式4: 3*cos(x) + log(x) - 3
    expr4 = 3*cos(x) + log(x) - 3
    # 对表达式4进行优化，使用 cosm1_opt 优化器
    opt4 = optimize(expr4, [cosm1_opt])
    # 断言优化后的结果等于 3*cosm1(x) + log(x)
    assert 3*cosm1(x) + log(x) == opt4
    # 断言优化后的结果等于原始表达式 expr4
    assert opt4.rewrite(cos) == expr4

    # 表达式5: 3*cos(2*x) - 3
    expr5 = 3*cos(2*x) - 3
    # 对表达式5进行优化，使用 cosm1_opt 优化器
    opt5 = optimize(expr5, [cosm1_opt])
    # 断言优化后的结果等于 3*cosm1(2*x)
    assert 3*cosm1(2*x) == opt5
    # 断言优化后的结果等于原始表达式 expr5
    assert opt5.rewrite(cos) == expr5

    # 表达式6: 2 - 2*cos(x)
    expr6 = 2 - 2*cos(x)
    # 对表达式6进行优化，使用 cosm1_opt 优化器
    opt6 = optimize(expr6, [cosm1_opt])
    # 断言优化后的结果等于 -2*cosm1(x)
    assert -2*cosm1(x) == opt6
    # 断言优化后的结果等于原始表达式 expr6
    assert opt6.rewrite(cos) == expr6


def test_cosm1_two_cos_terms():
    # 定义符号变量 x 和 y
    x, y = map(Symbol, 'x y'.split())
    # 构建表达式 cos(x) + cos(y) - 2
    expr1 = cos(x) + cos(y) - 2
    # 对表达式进行优化，使用 cosm1_opt 优化器
    opt1 = optimize(expr1, [cosm1_opt])
    # 断言优化后的结果等于 cosm1(x) + cosm1(y)
    assert opt1 == cosm1(x) + cosm1(y)


def test_expm1_cosm1_mixed():
    # 定义符号变量 x
    x = Symbol('x')
    # 构建表达式 exp(x) + cos(x) - 2
    expr1 = exp(x) + cos(x) - 2
    # 对表达式进行优化，使用 expm1_opt 和 cosm1_opt 优化器
    opt1 = optimize(expr1, [expm1_opt, cosm1_opt])
    # 断言优化后的结果等于 cosm1(x) + expm1(x)
    assert opt1 == cosm1(x) + expm1(x)


def _check_num_lambdify(expr, opt, val_subs, approx_ref, lambdify_kw=None, poorness=1e10):
    """ poorness=1e10 signifies that `expr` loses precision of at least ten decimal digits. """
    # 计算数值参考值
    num_ref = expr.subs(val_subs).evalf()
    # 计算机器精度的估计
    eps = numpy.finfo(numpy.float64).eps
    # 断言数值参考值与近似参考值的误差小于参考值乘以机器精度
    assert abs(num_ref - approx_ref) < approx_ref*eps
    # 使用 lambdify 函数创建数值函数 f1
    f1 = lambdify(list(val_subs.keys()), opt, **(lambdify_kw or {}))
    args_float = tuple(map(float, val_subs.values()))
    # 计算数值误差 num_err1
    num_err1 = abs(f1(*args_float) - approx_ref)
    # 断言数值误差 num_err1 小于数值参考值乘以机器精度的绝对值
    assert num_err1 < abs(num_ref*eps)
    # 使用 lambdify 函数创建数值函数 f2
    f2 = lambdify(list(val_subs.keys()), expr, **(lambdify_kw or {}))
    # 计算数值误差 num_err2
    num_err2 = abs(f2(*args_float) - approx_ref)
    # 断言数值误差 num_err2 大于数值参考值乘以机器精度的绝对值乘以 poorness
    assert num_err2 > abs(num_ref*eps*poorness)   # this only ensures that the *test* works as intended


def test_cosm1_apart():
    # 定义符号变量 x
    x = Symbol('x')

    # 表达式1: 1/cos(x) - 1
    expr1 = 1/cos(x) - 1
    # 对表达式1进行优化，使用 cosm1_opt 优化器
    opt1 = optimize(expr1, [cosm1_opt])
    # 断言优化后的结果等于 -cosm1(x)/cos(x)
    assert opt1 == -cosm1(x)/cos(x)
    if scipy:
        # 如果有 scipy 模块，进行数值验证
        _check_num_lambdify(expr1, opt1, {x: S(10)**-30}, 5e-61, lambdify_kw={"modules": 'scipy'})

    # 表达式2: 2/cos(x
    # 对表达式 expr3 进行优化，使用 cosm1_opt 作为优化选项
    opt3 = optimize(expr3, [cosm1_opt])

    # 断言优化后的结果 opt3 应为 -pi*cosm1(3*x)/cos(3*x)
    assert opt3 == -pi*cosm1(3*x)/cos(3*x)

    # 如果 scipy 模块可用，则执行下面的代码块
    if scipy:
        # 使用 lambdify_kw={"modules": 'scipy'} 参数将表达式转换为数值函数，并进行数值检查
        _check_num_lambdify(expr3, opt3, {x: S(10)**-30/3}, float(5e-61*pi), lambdify_kw={"modules": 'scipy'})
# 定义函数test_powm1，用于测试优化函数对表达式 x**y - 1 的处理效果
def test_powm1():
    # 将字符串 "xy" 中的字符映射为符号对象，并存储在元组 args 中
    args = x, y = map(Symbol, "xy")

    # 定义表达式 x**y - 1
    expr1 = x**y - 1

    # 对 expr1 应用 optimize 函数，使用 powm1_opt 作为优化器
    opt1 = optimize(expr1, [powm1_opt])

    # 断言优化后的结果 opt1 等于 powm1(x, y)
    assert opt1 == powm1(x, y)

    # 遍历参数 args，断言 expr1 对每个参数的导数等于 opt1 对每个参数的导数
    for arg in args:
        assert expr1.diff(arg) == opt1.diff(arg)

    # 如果 scipy 存在，并且版本大于等于 (1, 10, 0)
    if scipy and tuple(map(int, scipy.version.version.split('.')[:3])) >= (1, 10, 0):
        # 定义替换字典 subs1_a，用于 lambdify 函数的测试
        subs1_a = {x: Rational(*(1.0+1e-13).as_integer_ratio()), y: pi}
        ref1_f64_a = 3.139081648208105e-13
        # 调用 _check_num_lambdify 函数，检查 lambdify 结果是否符合预期
        _check_num_lambdify(expr1, opt1, subs1_a, ref1_f64_a, lambdify_kw={"modules": 'scipy'}, poorness=10**11)

        # 定义替换字典 subs1_b，用于 lambdify 函数的测试
        subs1_b = {x: pi, y: Rational(*(1e-10).as_integer_ratio())}
        ref1_f64_b = 1.1447298859149205e-10
        # 调用 _check_num_lambdify 函数，检查 lambdify 结果是否符合预期
        _check_num_lambdify(expr1, opt1, subs1_b, ref1_f64_b, lambdify_kw={"modules": 'scipy'}, poorness=10**9)


# 定义函数 test_log1p_opt，用于测试优化函数对 log1p 的处理效果
def test_log1p_opt():
    # 定义符号对象 x
    x = Symbol('x')

    # 定义表达式 log(x + 1)
    expr1 = log(x + 1)

    # 对 expr1 应用 optimize 函数，使用 log1p_opt 作为优化器
    opt1 = optimize(expr1, [log1p_opt])

    # 断言 log1p(x) - opt1 等于 0
    assert log1p(x) - opt1 == 0

    # 断言 opt1 通过 rewrite(log) 等于 expr1
    assert opt1.rewrite(log) == expr1

    # 定义表达式 log(3*x + 3)
    expr2 = log(3*x + 3)

    # 对 expr2 应用 optimize 函数，使用 log1p_opt 作为优化器
    opt2 = optimize(expr2, [log1p_opt])

    # 断言 log1p(x) + log(3) 等于 opt2
    assert log1p(x) + log(3) == opt2

    # 断言 (opt2 通过 rewrite(log) - expr2).simplify() 等于 0
    assert (opt2.rewrite(log) - expr2).simplify() == 0

    # 定义表达式 log(2*x + 1)
    expr3 = log(2*x + 1)

    # 对 expr3 应用 optimize 函数，使用 log1p_opt 作为优化器
    opt3 = optimize(expr3, [log1p_opt])

    # 断言 log1p(2*x) - opt3 等于 0
    assert log1p(2*x) - opt3 == 0

    # 断言 opt3 通过 rewrite(log) 等于 expr3
    assert opt3.rewrite(log) == expr3

    # 定义表达式 log(x+3)
    expr4 = log(x+3)

    # 对 expr4 应用 optimize 函数，使用 log1p_opt 作为优化器
    opt4 = optimize(expr4, [log1p_opt])

    # 断言 opt4 转换为字符串等于 'log(x + 3)'
    assert str(opt4) == 'log(x + 3)'


# 定义函数 test_optims_c99，用于测试优化函数对 C99 标准函数的处理效果
def test_optims_c99():
    # 定义符号对象 x
    x = Symbol('x')

    # 定义复合表达式 expr1
    expr1 = 2**x + log(x)/log(2) + log(x + 1) + exp(x) - 1

    # 对 expr1 应用 optimize 函数，使用 optims_c99 作为优化器，并简化结果
    opt1 = optimize(expr1, optims_c99).simplify()

    # 断言 opt1 等于 exp2(x) + log2(x) + log1p(x) + expm1(x)
    assert opt1 == exp2(x) + log2(x) + log1p(x) + expm1(x)

    # 断言 opt1 通过 rewrite(exp).rewrite(log).rewrite(Pow) 等于 expr1
    assert opt1.rewrite(exp).rewrite(log).rewrite(Pow) == expr1

    # 定义表达式 expr2
    expr2 = log(x)/log(2) + log(x + 1)

    # 对 expr2 应用 optimize 函数，使用 optims_c99 作为优化器
    opt2 = optimize(expr2, optims_c99)

    # 断言 opt2 等于 log2(x) + log1p(x)
    assert opt2 == log2(x) + log1p(x)

    # 断言 opt2 通过 rewrite(log) 等于 expr2
    assert opt2.rewrite(log) == expr2

    # 定义表达式 expr3
    expr3 = log(x)/log(2) + log(17*x + 17)

    # 对 expr3 应用 optimize 函数，使用 optims_c99 作为优化器
    opt3 = optimize(expr3, optims_c99)

    # 计算 delta3，opt3 与期望值 (log2(x) + log(17) + log1p(x)) 的差值
    delta3 = opt3 - (log2(x) + log(17) + log1p(x))

    # 断言 delta3 等于 0
    assert delta3 == 0

    # 断言 (opt3 通过 rewrite(log) - expr3).simplify() 等于 0
    assert (opt3.rewrite(log) - expr3).simplify() == 0

    # 定义表达式 expr4
    expr4 = 2**x + 3*log(5*x + 7)/(13*log(2)) + 11*exp(x) - 11 + log(17*x + 17)

    # 对 expr4 应用 optimize 函数，使用 optims_c99 作为优化器，并简化结果
    opt4 = optimize(expr4, optims_c99).simplify()

    # 计算 delta4，opt4 与期望值的差值
    delta4 = opt4 - (exp2(x) + 3*log2(5*x + 7)/13 + 11*expm1(x) + log(17) + log1p(x))

    # 断言 delta4 等于 0
    assert delta4 == 0

    # 断言 (opt4 通过 rewrite(exp).rewrite(log).rewrite(Pow) - expr4).simplify() 等于 0
    assert (opt4.rewrite(exp).rewrite(log).rewrite(Pow) - expr4).simplify() == 0

    # 定义表达式 expr5
    expr5 = 3*exp(2*x) - 3

    # 对 expr5 应用 optimize 函数，使用 optims_c99 作为优化器
    opt5 = optimize(expr5, optims_c99)

    # 计算 delta5，opt5 与期望值的差值
    delta5 = opt5 - 3*expm1(2
    # 使用 lambda 表达式定义一个匿名函数 cc，该函数接受一个参数 x，并对其进行优化处理后再转换为 C 代码
    cc = lambda x: ccode(
        optimize(x, [create_expand_pow_optimization(4)]))

    # 创建符号变量 x
    x = Symbol('x')

    # 断言优化后 x 的四次方表达式转换为 'x*x*x*x'
    assert cc(x**4) == 'x*x*x*x'

    # 断言优化后 x 的四次方加 x 的二次方表达式转换为 'x*x + x*x*x*x'
    assert cc(x**4 + x**2) == 'x*x + x*x*x*x'

    # 断言优化后 x 的五次方加 x 的四次方表达式转换为 'pow(x, 5) + x*x*x*x'
    assert cc(x**5 + x**4) == 'pow(x, 5) + x*x*x*x'

    # 断言优化后 sin(x) 的四次方表达式转换为 'pow(sin(x), 4)'
    assert cc(sin(x)**4) == 'pow(sin(x), 4)'

    # 断言优化后 x 的负四次方表达式转换为 '1.0/(x*x*x*x)'
    # 这里是解决 GitHub 问题 15335
    assert cc(x**(-4)) == '1.0/(x*x*x*x)'

    # 断言优化后 x 的负五次方表达式转换为 'pow(x, -5)'
    assert cc(x**(-5)) == 'pow(x, -5)'

    # 断言优化后 x 的负四次方表达式转换为 '-(x*x*x*x)'
    assert cc(-x**4) == '-(x*x*x*x)'

    # 创建一个整数类型的符号变量 i
    i = Symbol('i', integer=True)

    # 断言优化后 x 的 i 次方减去 x 的二次方表达式转换为 'pow(x, i) - (x*x)'
    assert cc(x**i - x**2) == 'pow(x, i) - (x*x)'

    # 创建一个实数类型的符号变量 y
    y = Symbol('y', real=True)

    # 断言优化后 exp(y^4) 的绝对值表达式转换为 "exp(y*y*y*y)"
    assert cc(Abs(exp(y**4))) == "exp(y*y*y*y)"

    # 这里是解决 GitHub 问题 20753
    # 使用 lambda 表达式定义一个新的匿名函数 cc2，该函数接受一个参数 x，并对其进行特定条件的优化处理后再转换为 C 代码
    cc2 = lambda x: ccode(optimize(x, [create_expand_pow_optimization(
        4, base_req=lambda b: b.is_Function)]))

    # 断言优化后 x 的三次方加 sin(x) 的三次方表达式转换为 "pow(x, 3) + sin(x)*sin(x)*sin(x)"
    assert cc2(x**3 + sin(x)**3) == "pow(x, 3) + sin(x)*sin(x)*sin(x)"
def test_matsolve():
    n = Symbol('n', integer=True)  # 声明一个整数符号变量 n
    A = MatrixSymbol('A', n, n)     # 声明一个 n x n 的矩阵符号 A
    x = MatrixSymbol('x', n, 1)     # 声明一个 n x 1 的矩阵符号 x

    with assuming(Q.fullrank(A)):   # 假设 A 是满秩矩阵
        assert optimize(A**(-1) * x, [matinv_opt]) == MatrixSolve(A, x)  # 优化矩阵求逆操作，应等价于矩阵求解操作
        assert optimize(A**(-1) * x + x, [matinv_opt]) == MatrixSolve(A, x) + x  # 优化矩阵求逆后加上 x，应等价于矩阵求解后加上 x


def test_logaddexp_opt():
    x, y = map(Symbol, 'x y'.split())  # 声明符号变量 x 和 y
    expr1 = log(exp(x) + exp(y))       # 构造表达式 log(exp(x) + exp(y))
    opt1 = optimize(expr1, [logaddexp_opt])  # 优化表达式，应用 logaddexp 优化
    assert logaddexp(x, y) - opt1 == 0   # 断言优化后的结果应与 logaddexp(x, y) 相等
    assert logaddexp(y, x) - opt1 == 0   # 断言优化后的结果应与 logaddexp(y, x) 相等
    assert opt1.rewrite(log) == expr1    # 断言优化后的结果应与原始表达式相等（用 log 重写的形式）


def test_logaddexp2_opt():
    x, y = map(Symbol, 'x y'.split())  # 声明符号变量 x 和 y
    expr1 = log(2**x + 2**y)/log(2)    # 构造表达式 log(2**x + 2**y) / log(2)
    opt1 = optimize(expr1, [logaddexp2_opt])  # 优化表达式，应用 logaddexp2 优化
    assert logaddexp2(x, y) - opt1 == 0   # 断言优化后的结果应与 logaddexp2(x, y) 相等
    assert logaddexp2(y, x) - opt1 == 0   # 断言优化后的结果应与 logaddexp2(y, x) 相等
    assert opt1.rewrite(log) == expr1    # 断言优化后的结果应与原始表达式相等（用 log 重写的形式）


def test_sinc_opts():
    def check(d):
        for k, v in d.items():
            assert optimize(k, sinc_opts) == v   # 断言优化给定的表达式 k 使用 sinc_opts 后的结果应等于 v

    x = Symbol('x')   # 声明符号变量 x
    check({
        sin(x)/x       : sinc(x),        # 检查 sin(x)/x 是否能优化成 sinc(x)
        sin(2*x)/(2*x) : sinc(2*x),      # 检查 sin(2*x)/(2*x) 是否能优化成 sinc(2*x)
        sin(3*x)/x     : 3*sinc(3*x),    # 检查 sin(3*x)/x 是否能优化成 3*sinc(3*x)
        x*sin(x)       : x*sin(x)       # 此项不应有优化
    })

    y = Symbol('y')   # 声明符号变量 y
    check({
        sin(x*y)/(x*y)       : sinc(x*y),            # 检查 sin(x*y)/(x*y) 是否能优化成 sinc(x*y)
        y*sin(x/y)/x         : sinc(x/y),            # 检查 y*sin(x/y)/x 是否能优化成 sinc(x/y)
        sin(sin(x))/sin(x)   : sinc(sin(x)),         # 检查 sin(sin(x))/sin(x) 是否能优化成 sinc(sin(x))
        sin(3*sin(x))/sin(x) : 3*sinc(3*sin(x)),     # 检查 sin(3*sin(x))/sin(x) 是否能优化成 3*sinc(3*sin(x))
        sin(x)/y             : sin(x)/y              # 此项不应有优化
    })


def test_optims_numpy():
    def check(d):
        for k, v in d.items():
            assert optimize(k, optims_numpy) == v   # 断言优化给定的表达式 k 使用 optims_numpy 后的结果应等于 v

    x = Symbol('x')   # 声明符号变量 x
    check({
        sin(2*x)/(2*x) + exp(2*x) - 1: sinc(2*x) + expm1(2*x),  # 检查表达式 sin(2*x)/(2*x) + exp(2*x) - 1 能否优化成 sinc(2*x) + expm1(2*x)
        log(x+3)/log(2) + log(x**2 + 1): log1p(x**2) + log2(x+3)  # 检查表达式 log(x+3)/log(2) + log(x**2 + 1) 能否优化成 log1p(x**2) + log2(x+3)
    })


@XFAIL  # room for improvement, ideally this test case should pass.
def test_optims_numpy_TODO():
    def check(d):
        for k, v in d.items():
            assert optimize(k, optims_numpy) == v   # 断言优化给定的表达式 k 使用 optims_numpy 后的结果应等于 v

    x, y = map(Symbol, 'x y'.split())   # 声明符号变量 x 和 y
    check({
        log(x*y)*sin(x*y)*log(x*y+1)/(log(2)*x*y): log2(x*y)*sinc(x*y)*log1p(x*y),  # 检查表达式 log(x*y)*sin(x*y)*log(x*y+1)/(log(2)*x*y) 能否优化成 log2(x*y)*sinc(x*y)*log1p(x*y)
        exp(x*sin(y)/y) - 1: expm1(x*sinc(y))   # 检查表达式 exp(x*sin(y)/y) - 1 能否优化成 expm1(x*sinc(y))
    })


@may_xfail
def test_compiled_ccode_with_rewriting():
    if not cython:
        skip("cython not installed.")
    if not has_c():
        skip("No C compiler found.")

    x = Symbol('x')   # 声明符号变量 x
    about_two = 2**(58/S(117))*3**(97/S(117))*5**(4/S(39))*7**(92/S(117))/S(30)*pi   # 计算 about_two 的近似值
    # about_two: 1.999999999999581826
    unchanged = 2*exp(x) - about_two   # 构造一个未改变的表达式
    xval = S(10)**-11   # 设置 x 的值
    ref = unchanged.subs(x, xval).n(19)  # 计算未改变的表达式在 x = xval 处的值，精确到 19 位数字

    rewritten = optimize(2*exp(x) - about_two, [expm1_opt])  # 优化表达式，应用 expm1_opt 优化

    # Unfortunately, we need to call ``.n()`` on our expressions before we hand them
    # to ``ccode``, and we need to request a large number of significant digits.
    # In this test, results converged for double precision when the following number
    # of significant digits were chosen:
    NUMBER_OF_DIGITS = 25   # TODO: this should ideally be automatically handled.
# 定义一个未改变的函数，接受一个参数 x 并返回未改变的数值
double func_unchanged(double x) {
    return %(unchanged)s;
}
# 定义一个重写的函数，接受一个参数 x 并返回重写后的数值
double func_rewritten(double x) {
    return %(rewritten)s;
}
''' % {"unchanged": ccode(unchanged.n(NUMBER_OF_DIGITS)),
           "rewritten": ccode(rewritten.n(NUMBER_OF_DIGITS))}
    # 使用 ccode 函数将 unchanged 和 rewritten 的值以 C 代码的格式插入到字符串中

func_pyx = '''
#cython: language_level=3
cdef extern double func_unchanged(double)
cdef extern double func_rewritten(double)
def py_unchanged(x):
    return func_unchanged(x)
def py_rewritten(x):
    return func_rewritten(x)
'''
    # 声明使用 Cython 的语法，指定语言级别为 3
    # 使用 cdef extern 声明外部函数 func_unchanged 和 func_rewritten
    # 定义两个 Python 函数 py_unchanged 和 py_rewritten，调用对应的外部函数

with tempfile.TemporaryDirectory() as folder:
    # 使用临时文件夹 folder 执行以下操作
    mod, info = compile_link_import_strings(
        [('func.c', func_c), ('_func.pyx', func_pyx)],
        build_dir=folder, compile_kwargs={"std": 'c99'}
    )
    # 编译和链接两个字符串：func_c 和 func_pyx，分别保存为 'func.c' 和 '_func.pyx'
    # 使用指定的编译选项 {"std": 'c99'} 编译
    err_rewritten = abs(mod.py_rewritten(1e-11) - ref)
    # 计算重写后函数调用的误差
    err_unchanged = abs(mod.py_unchanged(1e-11) - ref)
    # 计算未改变函数调用的误差
    assert 1e-27 < err_rewritten < 1e-25  # highly accurate.
    # 断言重写后函数的误差在 1e-27 和 1e-25 之间，表现非常精确
    assert 1e-19 < err_unchanged < 1e-16  # quite poor.
    # 断言未改变函数的误差在 1e-19 和 1e-16 之间，表现相对较差

# 上述容差的确定方式如下：
# >>> no_opt = unchanged.subs(x, xval.evalf()).evalf()
# >>> with_opt = rewritten.n(25).subs(x, 1e-11).evalf()
# >>> with_opt - ref, no_opt - ref
# (1.1536301877952077e-26, 1.6547074214222335e-18)
# 使用 sympy 进行计算，对未优化和优化后的表达式进行求值，并与参考值 ref 进行比较
```