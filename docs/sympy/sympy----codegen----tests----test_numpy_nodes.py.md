# `D:\src\scipysrc\sympy\sympy\codegen\tests\test_numpy_nodes.py`

```
# 导入需要的模块和函数
from itertools import product
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import (exp, log)
from sympy.printing.repr import srepr
from sympy.codegen.numpy_nodes import logaddexp, logaddexp2

# 定义符号变量 x, y, z
x, y, z = symbols('x y z')

# 定义测试函数 test_logaddexp
def test_logaddexp():
    # 计算 logaddexp(x, y)
    lae_xy = logaddexp(x, y)
    # 参考值 ref_xy = log(exp(x) + exp(y))
    ref_xy = log(exp(x) + exp(y))
    # 对于每个变量 wrt 和每个导数阶数 deriv_order 的组合进行迭代
    for wrt, deriv_order in product([x, y, z], range(3)):
        # 断言 lae_xy 对 wrt 变量的 deriv_order 阶导数与 ref_xy 相同
        assert (
            lae_xy.diff(wrt, deriv_order) -
            ref_xy.diff(wrt, deriv_order)
        ).rewrite(log).simplify() == 0

    # 定义常量
    one_third_e = 1*exp(1)/3
    two_thirds_e = 2*exp(1)/3
    # 计算 one_third_e 和 two_thirds_e 的自然对数
    logThirdE = log(one_third_e)
    logTwoThirdsE = log(two_thirds_e)
    # 计算 logaddexp(logThirdE, logTwoThirdsE)
    lae_sum_to_e = logaddexp(logThirdE, logTwoThirdsE)
    # 断言对数恒等式 logaddexp(logThirdE, logTwoThirdsE) == 1
    assert lae_sum_to_e.rewrite(log) == 1
    # 断言化简后的结果为 1
    assert lae_sum_to_e.simplify() == 1
    # 计算 logaddexp(2, 3)
    was = logaddexp(2, 3)
    # 断言 srepr(was) 等于 srepr(was.simplify())
    assert srepr(was) == srepr(was.simplify())  # 不能使用 2, 3 进行化简


# 定义测试函数 test_logaddexp2
def test_logaddexp2():
    # 计算 logaddexp2(x, y)
    lae2_xy = logaddexp2(x, y)
    # 参考值 ref2_xy = log(2**x + 2**y)/log(2)
    ref2_xy = log(2**x + 2**y)/log(2)
    # 对于每个变量 wrt 和每个导数阶数 deriv_order 的组合进行迭代
    for wrt, deriv_order in product([x, y, z], range(3)):
        # 断言 lae2_xy 对 wrt 变量的 deriv_order 阶导数与 ref2_xy 相同
        assert (
            lae2_xy.diff(wrt, deriv_order) -
            ref2_xy.diff(wrt, deriv_order)
        ).rewrite(log).cancel() == 0

    # 定义函数 lb(x) 返回 log(x)/log(2)
    def lb(x):
        return log(x)/log(2)

    # 定义常量
    two_thirds = S.One*2/3
    four_thirds = 2*two_thirds
    # 计算 lb(two_thirds) 和 lb(four_thirds) 的值
    lbTwoThirds = lb(two_thirds)
    lbFourThirds = lb(four_thirds)
    # 计算 logaddexp2(lbTwoThirds, lbFourThirds)
    lae2_sum_to_2 = logaddexp2(lbTwoThirds, lbFourThirds)
    # 断言对数恒等式 logaddexp2(lbTwoThirds, lbFourThirds) == 1
    assert lae2_sum_to_2.rewrite(log) == 1
    # 断言化简后的结果为 1
    assert lae2_sum_to_2.simplify() == 1
    # 计算 logaddexp2(x, y)
    was = logaddexp2(x, y)
    # 断言 srepr(was) 等于 srepr(was.simplify())
    assert srepr(was) == srepr(was.simplify())  # 不能使用 x, y 进行化简
```