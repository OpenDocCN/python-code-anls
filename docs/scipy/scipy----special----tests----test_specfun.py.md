# `D:\src\scipysrc\scipy\scipy\special\tests\test_specfun.py`

```
"""
Various made-up tests to hit different branches of the code in specfun.c
"""

import numpy as np  # 导入 NumPy 库，用于数值计算
from numpy.testing import assert_allclose  # 导入 NumPy 测试模块，用于数值比较断言
from scipy import special  # 导入 SciPy 的 special 模块，用于特殊数学函数的计算


def test_cva2_cv0_branches():
    # 调用特殊数学函数 mathieu_cem，返回结果 res 和 resp
    res, resp = special.mathieu_cem([40, 129], [13, 14], [30, 45])
    # 使用 assert_allclose 断言 res 数组的值接近给定的参考值数组
    assert_allclose(res, np.array([-0.3741211, 0.74441928]))
    # 使用 assert_allclose 断言 resp 数组的值接近给定的参考值数组
    assert_allclose(resp, np.array([-37.02872758, -86.13549877]))

    # 调用特殊数学函数 mathieu_sem，返回结果 res 和 resp
    res, resp = special.mathieu_sem([40, 129], [13, 14], [30, 45])
    # 使用 assert_allclose 断言 res 数组的值接近给定的参考值数组
    assert_allclose(res, np.array([0.92955551, 0.66771207]))
    # 使用 assert_allclose 断言 resp 数组的值接近给定的参考值数组
    assert_allclose(resp, np.array([-14.91073448, 96.02954185]))


def test_chgm_branches():
    # 调用 eval_genlaguerre 函数，计算结果并赋值给 res
    res = special.eval_genlaguerre(-3.2, 3, 2.5)
    # 使用 assert_allclose 断言 res 的值接近给定的参考值
    assert_allclose(res, -0.7077721935779854)


def test_hygfz_branches():
    """(z == 1.0) && (c-a-b > 0.0)"""
    # 调用 hyp2f1 函数，计算结果并赋值给 res
    res = special.hyp2f1(1.5, 2.5, 4.5, 1.+0.j)
    # 使用 assert_allclose 断言 res 的实部接近给定的参考实部
    assert_allclose(res, 10.30835089459151+0j)
    """(cabs(z+1) < eps) && (fabs(c-a+b - 1.0) < eps)"""
    # 调用 hyp2f1 函数，计算结果并赋值给 res
    res = special.hyp2f1(5+5e-16, 2, 2, -1.0 + 5e-16j)
    # 使用 assert_allclose 断言 res 的实部和虚部接近给定的参考值
    assert_allclose(res, 0.031249999999999986+3.9062499999999994e-17j)


def test_pro_rad1():
    # https://github.com/scipy/scipy/issues/21058
    # Reference values taken from WolframAlpha
    # SpheroidalS1(1, 1, 30, 1.1) 的参考值来自 WolframAlpha
    # SpheroidalS1Prime(1, 1, 30, 1.1) 的参考值来自 WolframAlpha
    # 调用 pro_rad1 函数，计算结果并赋值给 res
    res = special.pro_rad1(1, 1, 30, 1.1)
    # 使用 assert_allclose 断言 res 的各个返回值接近给定的参考值，相对误差容忍度为 2e-5
    assert_allclose(res, (0.009657872296166435, 3.253369651472877), rtol=2e-5)
```