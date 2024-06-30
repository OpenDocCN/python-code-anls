# `D:\src\scipysrc\sympy\sympy\codegen\tests\test_approximations.py`

```
import math  # 导入 math 模块，提供数学函数支持
from sympy.core.symbol import symbols  # 导入 sympy 库中的 symbols 函数，用于定义符号变量
from sympy.functions.elementary.exponential import exp  # 导入 sympy 库中的 exp 函数，指数函数
from sympy.codegen.rewriting import optimize  # 导入 sympy 库中的 optimize 函数，用于优化表达式
from sympy.codegen.approximations import SumApprox, SeriesApprox  # 导入 sympy 库中的 SumApprox 和 SeriesApprox 类，用于近似求和和级数展开


def test_SumApprox_trivial():
    x = symbols('x')  # 创建符号变量 x
    expr1 = 1 + x  # 定义表达式 1 + x
    sum_approx = SumApprox(bounds={x: (-1e-20, 1e-20)}, reltol=1e-16)  # 创建 SumApprox 近似求和对象
    apx1 = optimize(expr1, [sum_approx])  # 对表达式进行优化近似求和
    assert apx1 - 1 == 0  # 断言优化结果与预期值相等


def test_SumApprox_monotone_terms():
    x, y, z = symbols('x y z')  # 创建符号变量 x, y, z
    expr1 = exp(z)*(x**2 + y**2 + 1)  # 定义包含指数和二次项的表达式
    bnds1 = {x: (0, 1e-3), y: (100, 1000)}  # 定义变量 x 和 y 的取值范围
    sum_approx_m2 = SumApprox(bounds=bnds1, reltol=1e-2)  # 创建 SumApprox 近似求和对象，相对误差容忍度为 1e-2
    sum_approx_m5 = SumApprox(bounds=bnds1, reltol=1e-5)  # 创建 SumApprox 近似求和对象，相对误差容忍度为 1e-5
    sum_approx_m11 = SumApprox(bounds=bnds1, reltol=1e-11)  # 创建 SumApprox 近似求和对象，相对误差容忍度为 1e-11
    assert (optimize(expr1, [sum_approx_m2])/exp(z) - (y**2)).simplify() == 0  # 断言优化结果与预期值相等
    assert (optimize(expr1, [sum_approx_m5])/exp(z) - (y**2 + 1)).simplify() == 0  # 断言优化结果与预期值相等
    assert (optimize(expr1, [sum_approx_m11])/exp(z) - (y**2 + 1 + x**2)).simplify() == 0  # 断言优化结果与预期值相等


def test_SeriesApprox_trivial():
    x, z = symbols('x z')  # 创建符号变量 x, z
    for factor in [1, exp(z)]:  # 遍历因子列表
        x = symbols('x')  # 重新定义符号变量 x
        expr1 = exp(x)*factor  # 定义包含指数和因子的表达式
        bnds1 = {x: (-1, 1)}  # 定义变量 x 的取值范围
        series_approx_50 = SeriesApprox(bounds=bnds1, reltol=0.50)  # 创建 SeriesApprox 级数展开对象，相对误差容忍度为 0.50
        series_approx_10 = SeriesApprox(bounds=bnds1, reltol=0.10)  # 创建 SeriesApprox 级数展开对象，相对误差容忍度为 0.10
        series_approx_05 = SeriesApprox(bounds=bnds1, reltol=0.05)  # 创建 SeriesApprox 级数展开对象，相对误差容忍度为 0.05
        c = (bnds1[x][1] + bnds1[x][0])/2  # 计算变量 x 的中间值
        f0 = math.exp(c)  # 计算指数函数在中间值处的值

        ref_50 = f0 + x + x**2/2  # 计算参考值 ref_50
        ref_10 = f0 + x + x**2/2 + x**3/6  # 计算参考值 ref_10
        ref_05 = f0 + x + x**2/2 + x**3/6 + x**4/24  # 计算参考值 ref_05

        res_50 = optimize(expr1, [series_approx_50])  # 对表达式进行优化级数展开
        res_10 = optimize(expr1, [series_approx_10])  # 对表达式进行优化级数展开
        res_05 = optimize(expr1, [series_approx_05])  # 对表达式进行优化级数展开

        assert (res_50/factor - ref_50).simplify() == 0  # 断言优化结果与预期值相等
        assert (res_10/factor - ref_10).simplify() == 0  # 断言优化结果与预期值相等
        assert (res_05/factor - ref_05).simplify() == 0  # 断言优化结果与预期值相等

        max_ord3 = SeriesApprox(bounds=bnds1, reltol=0.05, max_order=3)  # 创建最大阶数为 3 的 SeriesApprox 级数展开对象
        assert optimize(expr1, [max_ord3]) == expr1  # 断言优化结果与原始表达式相等
```