# `D:\src\scipysrc\scipy\scipy\stats\tests\test_relative_risk.py`

```
# 导入 pytest 库，用于测试框架
# 导入 numpy 库，并将其命名为 np
# 从 numpy.testing 模块中导入 assert_allclose 和 assert_equal 函数
# 从 scipy.stats.contingency 模块中导入 relative_risk 函数
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy.stats.contingency import relative_risk

# 测试相对风险计算的函数，包括结果为 0、inf 或 nan 的边缘情况
@pytest.mark.parametrize(
    'exposed_cases, exposed_total, control_cases, control_total, expected_rr',
    [(1, 4, 3, 8, 0.25 / 0.375),    # 相对风险不为 0, inf 或 nan 的一般情况
     (0, 10, 5, 20, 0),            # 相对风险为 0 的情况
     (0, 10, 0, 20, np.nan),       # 相对风险为 nan 的情况
     (5, 15, 0, 20, np.inf)]       # 相对风险为 inf 的情况
)
def test_relative_risk(exposed_cases, exposed_total,
                       control_cases, control_total, expected_rr):
    # 调用 relative_risk 函数计算相对风险
    result = relative_risk(exposed_cases, exposed_total,
                           control_cases, control_total)
    # 使用 assert_allclose 函数断言计算结果与预期相对风险的接近程度
    assert_allclose(result.relative_risk, expected_rr, rtol=1e-13)


# 测试相对风险和置信区间计算
def test_relative_risk_confidence_interval():
    # 调用 relative_risk 函数计算相对风险
    result = relative_risk(exposed_cases=16, exposed_total=128,
                           control_cases=24, control_total=256)
    rr = result.relative_risk
    # 获取相对风险 rr
    # 计算置信区间 ci
    ci = result.confidence_interval(confidence_level=0.95)
    # 以下是使用 R 和 epitools 包计算相应值的对应代码
    # 断言相对风险 rr 等于 4/3
    assert_allclose(rr, 4/3)
    # 使用 assert_allclose 函数断言置信区间的下界和上界接近预期值
    assert_allclose((ci.low, ci.high), (0.7347317, 2.419628), rtol=5e-7)


# 测试置信水平为 0 的相对风险置信区间计算
def test_relative_risk_ci_conflevel0():
    # 调用 relative_risk 函数计算相对风险
    result = relative_risk(exposed_cases=4, exposed_total=12,
                           control_cases=5, control_total=30)
    rr = result.relative_risk
    # 使用 assert_allclose 函数断言相对风险 rr 等于 2.0
    assert_allclose(rr, 2.0, rtol=1e-14)
    # 计算置信区间，置信水平为 0
    ci = result.confidence_interval(0)
    # 使用 assert_allclose 函数断言置信区间的下界和上界等于 2.0
    assert_allclose((ci.low, ci.high), (2.0, 2.0), rtol=1e-12)


# 测试置信水平为 1 的相对风险置信区间计算
def test_relative_risk_ci_conflevel1():
    # 调用 relative_risk 函数计算相对风险
    result = relative_risk(exposed_cases=4, exposed_total=12,
                           control_cases=5, control_total=30)
    # 计算置信区间，置信水平为 1
    ci = result.confidence_interval(1)
    # 使用 assert_equal 函数断言置信区间的下界和上界等于 (0, inf)
    assert_equal((ci.low, ci.high), (0, np.inf))


# 测试相对风险计算的边缘情况：exposed_cases 和 control_cases 均为 0
def test_relative_risk_ci_edge_cases_00():
    # 调用 relative_risk 函数计算相对风险
    result = relative_risk(exposed_cases=0, exposed_total=12,
                           control_cases=0, control_total=30)
    # 使用 assert_equal 函数断言相对风险为 np.nan
    assert_equal(result.relative_risk, np.nan)
    # 计算置信区间
    ci = result.confidence_interval()
    # 使用 assert_equal 函数断言置信区间的下界和上界均为 np.nan
    assert_equal((ci.low, ci.high), (np.nan, np.nan))


# 测试相对风险计算的边缘情况：exposed_cases 为 0，control_cases 为 1
def test_relative_risk_ci_edge_cases_01():
    # 调用 relative_risk 函数计算相对风险
    result = relative_risk(exposed_cases=0, exposed_total=12,
                           control_cases=1, control_total=30)
    # 使用 assert_equal 函数断言相对风险为 0
    assert_equal(result.relative_risk, 0)
    # 计算置信区间
    ci = result.confidence_interval()
    # 使用 assert_equal 函数断言置信区间的下界为 0.0，上界为 np.nan
    assert_equal((ci.low, ci.high), (0.0, np.nan))


# 测试相对风险计算的边缘情况：exposed_cases 为 1，control_cases 为 0
def test_relative_risk_ci_edge_cases_10():
    # 调用 relative_risk 函数计算相对风险
    result = relative_risk(exposed_cases=1, exposed_total=12,
                           control_cases=0, control_total=30)
    # 断言结果的相对风险为无穷大
    assert_equal(result.relative_risk, np.inf)
    # 计算结果的置信区间
    ci = result.confidence_interval()
    # 断言置信区间的下界和上界分别为 NaN 和无穷大
    assert_equal((ci.low, ci.high), (np.nan, np.inf))
# 使用 pytest 的 parametrize 装饰器，为 test_relative_risk_bad_value 函数定义多组参数组合进行测试
@pytest.mark.parametrize('ec, et, cc, ct', [(0, 0, 10, 20),
                                            (-1, 10, 1, 5),
                                            (1, 10, 0, 0),
                                            (1, 10, -1, 4)])
# 定义测试函数 test_relative_risk_bad_value，测试相对风险函数对于不合法值的处理
def test_relative_risk_bad_value(ec, et, cc, ct):
    # 使用 pytest 的 raises 函数验证是否抛出 ValueError 异常，并匹配指定的错误信息
    with pytest.raises(ValueError, match="must be an integer not less than"):
        relative_risk(ec, et, cc, ct)


# 定义测试函数 test_relative_risk_bad_type，测试相对风险函数对于不合法类型的处理
def test_relative_risk_bad_type():
    # 使用 pytest 的 raises 函数验证是否抛出 TypeError 异常，并匹配指定的错误信息
    with pytest.raises(TypeError, match="must be an integer"):
        relative_risk(1, 10, 2.0, 40)
```