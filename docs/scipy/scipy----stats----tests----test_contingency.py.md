# `D:\src\scipysrc\scipy\scipy\stats\tests\test_contingency.py`

```
# 导入必要的库和模块
import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
                           assert_array_almost_equal, assert_approx_equal,
                           assert_allclose)
import pytest
from pytest import raises as assert_raises
from scipy.special import xlogy
from scipy.stats.contingency import (margins, expected_freq,
                                     chi2_contingency, association)

# 定义测试函数 test_margins，用于测试 margins 函数
def test_margins():
    # 测试数组 a = np.array([1])
    a = np.array([1])
    # 调用 margins 函数计算边际
    m = margins(a)
    # 断言边际数组 m 的长度为 1
    assert_equal(len(m), 1)
    # 获取边际数组 m 的第一个元素 m0
    m0 = m[0]
    # 断言 m0 与预期的 np.array([1]) 相等
    assert_array_equal(m0, np.array([1]))

    # 测试二维数组 a = np.array([[1]])
    a = np.array([[1]])
    # 调用 margins 函数计算边际 m0, m1
    m0, m1 = margins(a)
    expected0 = np.array([[1]])
    expected1 = np.array([[1]])
    # 断言 m0 和 m1 与预期的数组 expected0, expected1 相等
    assert_array_equal(m0, expected0)
    assert_array_equal(m1, expected1)

    # 测试二维数组 a = np.arange(12).reshape(2, 6)
    a = np.arange(12).reshape(2, 6)
    # 调用 margins 函数计算边际 m0, m1
    m0, m1 = margins(a)
    expected0 = np.array([[15], [51]])
    expected1 = np.array([[6, 8, 10, 12, 14, 16]])
    # 断言 m0 和 m1 与预期的数组 expected0, expected1 相等
    assert_array_equal(m0, expected0)
    assert_array_equal(m1, expected1)

    # 测试三维数组 a = np.arange(24).reshape(2, 3, 4)
    a = np.arange(24).reshape(2, 3, 4)
    # 调用 margins 函数计算边际 m0, m1, m2
    m0, m1, m2 = margins(a)
    expected0 = np.array([[[66]], [[210]]])
    expected1 = np.array([[[60], [92], [124]]])
    expected2 = np.array([[[60, 66, 72, 78]]])
    # 断言 m0, m1, m2 与预期的数组 expected0, expected1, expected2 相等
    assert_array_equal(m0, expected0)
    assert_array_equal(m1, expected1)
    assert_array_equal(m2, expected2)


# 定义测试函数 test_expected_freq，用于测试 expected_freq 函数
def test_expected_freq():
    # 测试一维数组 [1]
    assert_array_equal(expected_freq([1]), np.array([1.0]))

    # 测试三维数组 observed
    observed = np.array([[[2, 0], [0, 2]], [[0, 2], [2, 0]], [[1, 1], [1, 1]]])
    e = expected_freq(observed)
    # 断言计算得到的期望频率 e 与全一数组 np.ones_like(observed) 相等
    assert_array_equal(e, np.ones_like(observed))

    # 测试二维数组 observed
    observed = np.array([[10, 10, 20], [20, 20, 20]])
    e = expected_freq(observed)
    correct = np.array([[12., 12., 16.], [18., 18., 24.]])
    # 断言计算得到的期望频率 e 与预期的数组 correct 相等
    assert_array_almost_equal(e, correct)


# 定义测试函数 test_chi2_contingency_trivial，测试 chi2_contingency 函数的简单情况
def test_chi2_contingency_trivial():
    # 测试一个简单的情况
    obs = np.array([[1, 2], [1, 2]])
    # 调用 chi2_contingency 函数计算卡方值 chi2, p 值 p, 自由度 dof 和期望数组 expected
    chi2, p, dof, expected = chi2_contingency(obs, correction=False)
    # 断言计算得到的 chi2, p, dof 和期望数组 expected 与预期相等
    assert_equal(chi2, 0.0)
    assert_equal(p, 1.0)
    assert_equal(dof, 1)
    assert_array_equal(obs, expected)

    # 测试一个更简单的情况：一维数据
    obs = np.array([1, 2, 3])
    # 调用 chi2_contingency 函数计算卡方值 chi2, p 值 p, 自由度 dof 和期望数组 expected
    chi2, p, dof, expected = chi2_contingency(obs, correction=False)
    # 断言计算得到的 chi2, p, dof 和期望数组 expected 与预期相等
    assert_equal(chi2, 0.0)
    assert_equal(p, 1.0)
    assert_equal(dof, 0)
    assert_array_equal(obs, expected)


# 定义测试函数 test_chi2_contingency_R，测试 chi2_contingency 函数与 R 计算结果的一致性
def test_chi2_contingency_R():
    # 一些使用 R 独立计算的测试案例

    # Rcode = \
    # """
    # # Data vector.
    # data <- c(
    #   12, 34, 23,     4,  47,  11,
    #   35, 31, 11,    34,  10,  18,
    #   12, 32,  9,    18,  13,  19,
    #   12, 12, 14,     9,  33,  25
    #   )
    #
    # # Create factor tags:r=rows, c=columns, t=tiers
    # r <- factor(gl(4, 2*3, 2*3*4, labels=c("r1", "r2", "r3", "r4")))
    # c <- factor(gl(3, 1,   2*3*4, labels=c("c1", "c2", "c3")))
    # t <- factor(gl(2, 3,   2*3*4, labels=c("t1", "t2")))
    #
    # # 3-way Chi squared test of independence
    # s = summary(xtabs(data~r+c+t))
    # """

    # 这里没有直接编写 Python 代码，而是提供了 R 代码的注释
    # 该部分的测试需要通过 R 代码的输出来验证与 Python 计算的一致性
    # 定义观察到的频数数据（observed frequencies），包含两个3维数组
    obs = np.array(
        [[[12, 34, 23],       # 第一个3维数组的第一个平面
          [35, 31, 11],       # 第一个3维数组的第二个平面
          [12, 32, 9],        # 第一个3维数组的第三个平面
          [12, 12, 14]],      # 第一个3维数组的第四个平面
         [[4, 47, 11],        # 第二个3维数组的第一个平面
          [34, 10, 18],       # 第二个3维数组的第二个平面
          [18, 13, 19],       # 第二个3维数组的第三个平面
          [9, 33, 25]]])      # 第二个3维数组的第四个平面
    # 执行卡方独立性检验，计算卡方值、p值、自由度和期望频数
    chi2, p, dof, expected = chi2_contingency(obs)
    # 断言卡方值近似于预期值102.17，精度为5位有效数字
    assert_approx_equal(chi2, 102.17, significant=5)
    # 断言p值近似于预期值3.514e-14，精度为4位有效数字
    assert_approx_equal(p, 3.514e-14, significant=4)
    # 断言自由度等于预期值17
    assert_equal(dof, 17)

    # 定义观察到的频数数据（observed frequencies），包含两个4维数组
    obs = np.array(
        [[[[12, 17],           # 第一个4维数组的第一个平面
           [11, 16]],          # 第一个4维数组的第二个平面
          [[11, 12],           # 第一个4维数组的第三个平面
           [15, 16]]],         # 第一个4维数组的第四个平面
         [[[23, 15],           # 第二个4维数组的第一个平面
           [30, 22]],          # 第二个4维数组的第二个平面
          [[14, 17],           # 第二个4维数组的第三个平面
           [15, 16]]]])        # 第二个4维数组的第四个平面
    # 执行卡方独立性检验，计算卡方值、p值、自由度和期望频数
    chi2, p, dof, expected = chi2_contingency(obs)
    # 断言卡方值近似于预期值8.758，精度为4位有效数字
    assert_approx_equal(chi2, 8.758, significant=4)
    # 断言p值近似于预期值0.6442，精度为4位有效数字
    assert_approx_equal(p, 0.6442, significant=4)
    # 断言自由度等于预期值11
    assert_equal(dof, 11)
def test_chi2_contingency_g():
    # 创建一个2x2的numpy数组作为观察频率
    c = np.array([[15, 60], [15, 90]])
    # 使用log-likelihood作为lambda参数计算卡方统计量g，不进行校正
    g, p, dof, e = chi2_contingency(c, lambda_='log-likelihood',
                                    correction=False)
    # 断言卡方统计量g与2 * xlogy(c, c/e).sum()的近似相等
    assert_allclose(g, 2*xlogy(c, c/e).sum())

    # 使用log-likelihood作为lambda参数计算卡方统计量g，进行校正
    g, p, dof, e = chi2_contingency(c, lambda_='log-likelihood',
                                    correction=True)
    # 创建校正后的观察频率数组
    c_corr = c + np.array([[-0.5, 0.5], [0.5, -0.5]])
    # 断言校正后的卡方统计量g与2 * xlogy(c_corr, c_corr/e).sum()的近似相等
    assert_allclose(g, 2*xlogy(c_corr, c_corr/e).sum())

    # 创建一个2x3的numpy数组作为观察频率
    c = np.array([[10, 12, 10], [12, 10, 10]])
    # 使用log-likelihood作为lambda参数计算卡方统计量g
    g, p, dof, e = chi2_contingency(c, lambda_='log-likelihood')
    # 断言卡方统计量g与2 * xlogy(c, c/e).sum()的近似相等
    assert_allclose(g, 2*xlogy(c, c/e).sum())


def test_chi2_contingency_bad_args():
    # 测试输入异常情况是否会引发ValueError异常

    # 在观察频率数组中包含负值
    obs = np.array([[-1, 10], [1, 2]])
    assert_raises(ValueError, chi2_contingency, obs)

    # 观察频率数组中包含零将导致期望频率数组中出现零
    obs = np.array([[0, 1], [0, 1]])
    assert_raises(ValueError, chi2_contingency, obs)

    # 退化情况：观察数组的大小为0
    obs = np.empty((0, 8))
    assert_raises(ValueError, chi2_contingency, obs)


def test_chi2_contingency_yates_gh13875():
    # 验证Yates连续性校正的幅度不应超过统计量的观察值和期望值之差；参见gh-13875
    observed = np.array([[1573, 3], [4, 0]])
    p = chi2_contingency(observed)[1]
    # 断言p值与1的近似相等，相对误差不超过1e-12
    assert_allclose(p, 1, rtol=1e-12)


@pytest.mark.parametrize("correction", [False, True])
def test_result(correction):
    # 创建一个2x2的numpy数组作为观察频率
    obs = np.array([[1, 2], [1, 2]])
    # 进行卡方独立性检验，根据correction参数是否进行Yates校正
    res = chi2_contingency(obs, correction=correction)
    # 断言返回的结果元组中的(statistic, pvalue, dof, expected_freq)与res相等
    assert_equal((res.statistic, res.pvalue, res.dof, res.expected_freq), res)


def test_bad_association_args():
    # 无效的测试统计量
    assert_raises(ValueError, association, [[1, 2], [3, 4]], "X")
    # 无效的数组形状
    assert_raises(ValueError, association, [[[1, 2]], [[3, 4]]], "cramer")
    # chi2_contingency异常
    assert_raises(ValueError, association, [[-1, 10], [1, 2]], 'cramer')
    # 无效的数组项数据类型
    assert_raises(ValueError, association,
                  np.array([[1, 2], ["dd", 4]], dtype=object), 'cramer')


@pytest.mark.parametrize('stat, expected',
                         [('cramer', 0.09222412010290792),
                          ('tschuprow', 0.0775509319944633),
                          ('pearson', 0.12932925727138758)])
def test_assoc(stat, expected):
    # 创建一个3x5的numpy数组作为观察频率
    obs1 = np.array([[12, 13, 14, 15, 16],
                     [17, 16, 18, 19, 11],
                     [9, 15, 14, 12, 11]])
    # 使用指定的方法计算关联性，并断言结果与期望值的近似相等
    a = association(observed=obs1, method=stat)
    assert_allclose(a, expected)
```