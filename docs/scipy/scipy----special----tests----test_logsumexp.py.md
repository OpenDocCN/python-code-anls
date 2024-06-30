# `D:\src\scipysrc\scipy\scipy\special\tests\test_logsumexp.py`

```
# 导入 numpy 库，并将其重命名为 np
import numpy as np
# 从 numpy.testing 模块中导入多个断言函数
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
                           assert_array_almost_equal, assert_)

# 从 scipy.special 模块中导入 logsumexp 和 softmax 函数
from scipy.special import logsumexp, softmax


def test_logsumexp():
    # 测试空数组的情况
    a = []
    desired = -np.inf
    assert_equal(logsumexp(a), desired)

    # 测试 logsumexp() 函数对大输入的处理是否正确
    a = np.arange(200)
    desired = np.log(np.sum(np.exp(a)))
    assert_almost_equal(logsumexp(a), desired)

    # 现在测试包含大数值的情况
    b = [1000, 1000]
    desired = 1000.0 + np.log(2.0)
    assert_almost_equal(logsumexp(b), desired)

    n = 1000
    b = np.full(n, 10000, dtype='float64')
    desired = 10000.0 + np.log(n)
    assert_almost_equal(logsumexp(b), desired)

    x = np.array([1e-40] * 1000000)
    logx = np.log(x)

    X = np.vstack([x, x])
    logX = np.vstack([logx, logx])
    # 断言对数求和的结果是否等于原始数据的和
    assert_array_almost_equal(np.exp(logsumexp(logX)), X.sum())
    assert_array_almost_equal(np.exp(logsumexp(logX, axis=0)), X.sum(axis=0))
    assert_array_almost_equal(np.exp(logsumexp(logX, axis=1)), X.sum(axis=1))

    # 正确处理特殊值的情况
    assert_equal(logsumexp(np.inf), np.inf)
    assert_equal(logsumexp(-np.inf), -np.inf)
    assert_equal(logsumexp(np.nan), np.nan)
    assert_equal(logsumexp([-np.inf, -np.inf]), -np.inf)

    # 处理轴上数值差异较大的数组
    assert_array_almost_equal(logsumexp([[1e10, 1e-10],
                                         [-1e10, -np.inf]], axis=-1),
                              [1e10, -1e10])

    # 测试保持维度的情况
    assert_array_almost_equal(logsumexp([[1e10, 1e-10],
                                         [-1e10, -np.inf]],
                                        axis=-1,
                                        keepdims=True),
                              [[1e10], [-1e10]])

    # 测试多轴的情况
    assert_array_almost_equal(logsumexp([[1e10, 1e-10],
                                         [-1e10, -np.inf]],
                                        axis=(-1,-2)),
                              1e10)


def test_logsumexp_b():
    a = np.arange(200)
    b = np.arange(200, 0, -1)
    desired = np.log(np.sum(b*np.exp(a)))
    assert_almost_equal(logsumexp(a, b=b), desired)

    a = [1000, 1000]
    b = [1.2, 1.2]
    desired = 1000 + np.log(2 * 1.2)
    assert_almost_equal(logsumexp(a, b=b), desired)

    x = np.array([1e-40] * 100000)
    b = np.linspace(1, 1000, 100000)
    logx = np.log(x)

    X = np.vstack((x, x))
    logX = np.vstack((logx, logx))
    B = np.vstack((b, b))
    # 断言带有权重的对数求和结果是否等于加权和
    assert_array_almost_equal(np.exp(logsumexp(logX, b=B)), (B * X).sum())
    assert_array_almost_equal(np.exp(logsumexp(logX, b=B, axis=0)),
                                (B * X).sum(axis=0))
    assert_array_almost_equal(np.exp(logsumexp(logX, b=B, axis=1)),
                                (B * X).sum(axis=1))


def test_logsumexp_sign():
    # 测试具有不同符号的数组
    a = [1,1,1]
    b = [1,-1,-1]
    # 调用logsumexp函数计算两个数a和b的指数对数和，并返回结果r和符号s
    r, s = logsumexp(a, b=b, return_sign=True)
    # 断言r几乎等于1，即检查r的值是否接近1
    assert_almost_equal(r, 1)
    # 断言s等于-1，即检查s的值是否等于-1
    assert_equal(s, -1)
# 定义一个测试函数，用于测试 logsumexp 函数在返回符号和零值时的行为
def test_logsumexp_sign_zero():
    # 定义两个数组 a 和 b
    a = [1, 1]
    b = [1, -1]

    # 调用 logsumexp 函数，返回结果 r 和 s，要求返回符号
    r, s = logsumexp(a, b=b, return_sign=True)
    # 断言 r 不是有限的（即为无穷大或无穷小）
    assert_(not np.isfinite(r))
    # 断言 r 不是 NaN
    assert_(not np.isnan(r))
    # 断言 r 小于 0
    assert_(r < 0)
    # 断言 s 等于 0
    assert_equal(s, 0)


# 定义一个测试函数，用于测试 logsumexp 函数在返回形状和符号时的行为
def test_logsumexp_sign_shape():
    # 创建一个形状为 (1,2,3,4) 的全为 1 的数组 a
    a = np.ones((1, 2, 3, 4))
    # 创建一个与 a 相同形状的全为 1 的数组 b
    b = np.ones_like(a)

    # 调用 logsumexp 函数，指定在第二个轴上求和，返回结果 r 和 s，要求返回符号
    r, s = logsumexp(a, axis=2, b=b, return_sign=True)

    # 断言 r 和 s 的形状相同，且为 (1,2,4)
    assert_equal(r.shape, s.shape)
    assert_equal(r.shape, (1, 2, 4))

    # 再次调用 logsumexp 函数，在第一和第三个轴上求和，返回结果 r 和 s，要求返回符号
    r, s = logsumexp(a, axis=(1, 3), b=b, return_sign=True)

    # 断言 r 和 s 的形状相同，且为 (1,3)
    assert_equal(r.shape, s.shape)
    assert_equal(r.shape, (1, 3))


# 定义一个测试函数，用于测试 logsumexp 函数在复数及其符号返回时的行为
def test_logsumexp_complex_sign():
    # 创建一个复数数组 a
    a = np.array([1 + 1j, 2 - 1j, -2 + 3j])

    # 调用 logsumexp 函数，返回结果 r 和 s，要求返回符号
    r, s = logsumexp(a, return_sign=True)

    # 计算期望的指数和及符号
    expected_sumexp = np.exp(a).sum()
    expected_sign = expected_sumexp / abs(expected_sumexp)

    # 断言 s 的值接近期望的符号
    assert_allclose(s, expected_sign)
    # 断言 s 乘以 exp(r) 的结果接近期望的指数和
    assert_allclose(s * np.exp(r), expected_sumexp)


# 定义一个测试函数，用于测试 logsumexp 函数在返回形状时的行为
def test_logsumexp_shape():
    # 创建一个形状为 (1,2,3,4) 的全为 1 的数组 a
    a = np.ones((1, 2, 3, 4))
    # 创建一个与 a 相同形状的全为 1 的数组 b
    b = np.ones_like(a)

    # 调用 logsumexp 函数，在第二个轴上求和，返回结果 r
    r = logsumexp(a, axis=2, b=b)
    # 断言 r 的形状为 (1,2,4)
    assert_equal(r.shape, (1, 2, 4))

    # 再次调用 logsumexp 函数，在第一和第三个轴上求和，返回结果 r
    r = logsumexp(a, axis=(1, 3), b=b)
    # 断言 r 的形状为 (1,3)
    assert_equal(r.shape, (1, 3))


# 定义一个测试函数，用于测试 logsumexp 函数在 b 包含零值时的行为
def test_logsumexp_b_zero():
    # 定义一个数组 a 包含两个元素 [1,10000]
    a = [1, 10000]
    # 定义一个数组 b 包含两个元素 [1,0]
    b = [1, 0]

    # 断言 logsumexp(a, b=b) 函数的结果接近 1
    assert_almost_equal(logsumexp(a, b=b), 1)


# 定义一个测试函数，用于测试 logsumexp 函数在 b 的形状和 a 不匹配时的行为
def test_logsumexp_b_shape():
    # 创建一个形状为 (4,1,2,1) 的全为 0 的数组 a
    a = np.zeros((4, 1, 2, 1))
    # 创建一个形状为 (3,1,5) 的全为 1 的数组 b
    b = np.ones((3, 1, 5))

    # 调用 logsumexp 函数，忽略返回结果
    logsumexp(a, b=b)


# 定义一个测试函数，用于测试 softmax 函数在不同情况下的行为
def test_softmax_fixtures():
    # 断言 softmax([1000, 0, 0, 0]) 的结果接近 [1, 0, 0, 0]
    assert_allclose(softmax([1000, 0, 0, 0]), np.array([1, 0, 0, 0]), rtol=1e-13)
    # 断言 softmax([1, 1]) 的结果接近 [.5, .5]
    assert_allclose(softmax([1, 1]), np.array([.5, .5]), rtol=1e-13)
    # 断言 softmax([0, 1]) 的结果接近 [1, e/(1+e)]
    assert_allclose(softmax([0, 1]), np.array([1, np.e])/(1 + np.e), rtol=1e-13)

    # 创建一个数组 x，其值为 [0, 1, 2, 3]
    x = np.arange(4)
    # 预期的 softmax 结果，通过 mpmath 计算后转换为浮点数
    expected = np.array([0.03205860328008499,
                         0.08714431874203256,
                         0.23688281808991013,
                         0.6439142598879722])

    # 断言 softmax(x) 的结果接近预期的 expected
    assert_allclose(softmax(x), expected, rtol=1e-13)

    # 平移性质测试：如果所有值都增加相同量，softmax 结果不变
    assert_allclose(softmax(x + 100), expected, rtol=1e-13)

    # 当 axis=None 时，softmax 操作作用在整个数组上，并保持形状不变
    assert_allclose(softmax(x.reshape(2, 2)), expected.reshape(2, 2), rtol=1e-13)


# 定义一个测试函数，用于测试 softmax 函数在多轴操作时的行为
def test_softmax_multi_axes():
    # 断言在 axis=0 上对 [[1000, 0], [1000, 0]] 求 softmax 结果接近 [[.5, .5], [.5, .5]]
    assert_allclose(softmax([[1000, 0], [1000, 0]], axis=0), np.array([[.5, .5], [.5, .5]]), rtol=1e-13)
    # 断言在 axis=1 上对 [[1000, 0], [1000, 0]] 求 softmax 结果接近 [[1, 0], [1, 0]]
    assert_allclose(softmax([[1000, 0], [1000, 0]], axis=1), np.array([[1, 0], [1, 0]]), rtol=1e-13)

    # 创建一个数组 x，其值为 [[-25, 0, 25, 50], [1, 325, 749, 750]]
    x = np.array([[-25, 0, 25, 50],
                  [1, 325, 749, 750]])

    # 通过 mpmath 计算预期的 softmax 结果，并转换为浮点数
    expected = np.array
    # 创建一个预期的 NumPy 数组，包含指定的浮点数值
    expected = np.array([[2.678636961770877e-33,
                          1.9287498479371314e-22,
                          1.3887943864771144e-11,
                          0.999999999986112],
                         [0.0,
                          1.9444526359919372e-185,
                          0.2689414213699951,
                          0.7310585786300048]])
    
    # 使用 assert_allclose 函数检查 softmax 函数在给定轴上的输出是否与预期的数组非常接近
    assert_allclose(softmax(x, axis=1), expected, rtol=1e-13)
    
    # 使用 assert_allclose 函数检查 softmax 函数在 x 的转置上的输出是否与预期的数组转置非常接近
    assert_allclose(softmax(x.T, axis=0), expected.T, rtol=1e-13)
    
    # 创建一个 3 维的输入 x3d，通过将 x 重新整形成 (2, 2, 2) 的形状
    x3d = x.reshape(2, 2, 2)
    
    # 使用 assert_allclose 函数检查 softmax 函数在给定轴 (1, 2) 上的输出是否与预期的数组重新整形成 (2, 2, 2) 形状非常接近
    assert_allclose(softmax(x3d, axis=(1, 2)), expected.reshape(2, 2, 2),
                    rtol=1e-13)
```