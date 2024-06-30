# `D:\src\scipysrc\sympy\sympy\printing\tests\test_cupy.py`

```
# 导入所需的模块和函数
from sympy.concrete.summations import Sum  # 导入 Sum 类，用于表示和
from sympy.functions.elementary.exponential import log  # 导入对数函数 log
from sympy.functions.elementary.miscellaneous import sqrt  # 导入平方根函数 sqrt
from sympy.utilities.lambdify import lambdify  # 导入 lambdify 函数，用于将 SymPy 表达式转换为可执行的函数
from sympy.abc import x, i, a, b  # 导入符号 x, i, a, b
from sympy.codegen.numpy_nodes import logaddexp  # 导入对数加法函数 logaddexp
from sympy.printing.numpy import CuPyPrinter, _cupy_known_constants, _cupy_known_functions  # 导入 CuPyPrinter 和相关常数、函数

from sympy.testing.pytest import skip, raises  # 导入测试所需的 skip 和 raises 函数
from sympy.external import import_module  # 导入 import_module 函数，用于动态导入模块

cp = import_module('cupy')  # 尝试导入 CuPy 库，若未安装，则 cp 为 None

def test_cupy_print():
    prntr = CuPyPrinter()  # 创建 CuPyPrinter 实例 prntr
    assert prntr.doprint(logaddexp(a, b)) == 'cupy.logaddexp(a, b)'  # 测试 logaddexp 函数的打印输出
    assert prntr.doprint(sqrt(x)) == 'cupy.sqrt(x)'  # 测试 sqrt 函数的打印输出
    assert prntr.doprint(log(x)) == 'cupy.log(x)'  # 测试 log 函数的打印输出
    assert prntr.doprint("acos(x)") == 'cupy.arccos(x)'  # 测试字符串 "acos(x)" 的打印输出
    assert prntr.doprint("exp(x)") == 'cupy.exp(x)'  # 测试字符串 "exp(x)" 的打印输出
    assert prntr.doprint("Abs(x)") == 'abs(x)'  # 测试字符串 "Abs(x)" 的打印输出

def test_not_cupy_print():
    prntr = CuPyPrinter()  # 创建 CuPyPrinter 实例 prntr
    with raises(NotImplementedError):  # 确保抛出 NotImplementedError 异常
        prntr.doprint("abcd(x)")  # 对未实现的字符串 "abcd(x)" 进行打印输出

def test_cupy_sum():
    if not cp:
        skip("CuPy not installed")  # 如果 CuPy 未安装，则跳过测试

    s = Sum(x ** i, (i, a, b))  # 创建符号求和表达式 x ** i，其中 i 的范围是 a 到 b
    f = lambdify((a, b, x), s, 'cupy')  # 将符号表达式 s 转换为 CuPy 函数 f

    a_, b_ = 0, 10  # 设置求和范围的下限和上限
    x_ = cp.linspace(-1, +1, 10)  # 创建 CuPy 数组 x_
    assert cp.allclose(f(a_, b_, x_), sum(x_ ** i_ for i_ in range(a_, b_ + 1)))  # 测试 CuPy 函数 f 的输出是否与预期一致

    s = Sum(i * x, (i, a, b))  # 创建符号求和表达式 i * x，其中 i 的范围是 a 到 b
    f = lambdify((a, b, x), s, 'numpy')  # 将符号表达式 s 转换为 NumPy 函数 f

    a_, b_ = 0, 10  # 再次设置求和范围的下限和上限
    x_ = cp.linspace(-1, +1, 10)  # 创建 CuPy 数组 x_
    assert cp.allclose(f(a_, b_, x_), sum(i_ * x_ for i_ in range(a_, b_ + 1)))  # 测试 NumPy 函数 f 的输出是否与预期一致

def test_cupy_known_funcs_consts():
    assert _cupy_known_constants['NaN'] == 'cupy.nan'  # 检查 _cupy_known_constants 中 NaN 的映射
    assert _cupy_known_constants['EulerGamma'] == 'cupy.euler_gamma'  # 检查 _cupy_known_constants 中 EulerGamma 的映射

    assert _cupy_known_functions['acos'] == 'cupy.arccos'  # 检查 _cupy_known_functions 中 acos 的映射
    assert _cupy_known_functions['log'] == 'cupy.log'  # 检查 _cupy_known_functions 中 log 的映射

def test_cupy_print_methods():
    prntr = CuPyPrinter()  # 创建 CuPyPrinter 实例 prntr
    assert hasattr(prntr, '_print_acos')  # 检查 prntr 是否具有 _print_acos 方法
    assert hasattr(prntr, '_print_log')  # 检查 prntr 是否具有 _print_log 方法
```