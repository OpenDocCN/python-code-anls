# `D:\src\scipysrc\seaborn\tests\test_algorithms.py`

```
# 导入必要的库
import numpy as np

# 导入 pytest 相关模块
import pytest
from numpy.testing import assert_array_equal

# 导入 seaborn 库中的 algorithms 模块
from seaborn import algorithms as algo

# 定义 pytest 的 fixture，用于生成随机数种子
@pytest.fixture
def random():
    np.random.seed(sum(map(ord, "test_algorithms")))

# 测试函数，验证 bootstrap 函数在简单情况下的正确性
def test_bootstrap(random):
    """Test that bootstrapping gives the right answer in dumb cases."""
    # 创建一个包含 10 个 1 的数组
    a_ones = np.ones(10)
    n_boot = 5
    # 进行 bootstrap 操作，并验证结果是否与预期一致
    out1 = algo.bootstrap(a_ones, n_boot=n_boot)
    assert_array_equal(out1, np.ones(n_boot))
    # 使用指定的函数（这里是 np.median）进行 bootstrap，并验证结果
    out2 = algo.bootstrap(a_ones, n_boot=n_boot, func=np.median)
    assert_array_equal(out2, np.ones(n_boot))

# 测试函数，验证 bootstrap 结果的形状是否正确
def test_bootstrap_length(random):
    """Test that we get a bootstrap array of the right shape."""
    # 创建一个包含 1000 个正态分布随机数的数组
    a_norm = np.random.randn(1000)
    # 进行 bootstrap 操作，并验证输出数组的长度是否正确
    out = algo.bootstrap(a_norm)
    assert len(out) == 10000

    n_boot = 100
    out = algo.bootstrap(a_norm, n_boot=n_boot)
    assert len(out) == n_boot

# 测试函数，验证 bootstrap 结果是否在正确的范围内
def test_bootstrap_range(random):
    """Test that bootstrapping a random array stays within the right range."""
    # 创建一个包含 1000 个正态分布随机数的数组
    a_norm = np.random.randn(1000)
    amin, amax = a_norm.min(), a_norm.max()
    # 进行 bootstrap 操作，并验证输出结果是否在原始数据的最小值和最大值范围内
    out = algo.bootstrap(a_norm)
    assert amin <= out.min()
    assert amax >= out.max()

# 测试函数，验证 bootstrap 函数是否支持多个输入数组
def test_bootstrap_multiarg(random):
    """Test that bootstrap works with multiple input arrays."""
    # 创建两个输入数组 x 和 y
    x = np.vstack([[1, 10] for i in range(10)])
    y = np.vstack([[5, 5] for i in range(10)])

    # 定义一个函数，用于处理多个输入数组，并返回合并后的结果
    def f(x, y):
        return np.vstack((x, y)).max(axis=0)

    # 进行多输入数组的 bootstrap 操作，并验证结果是否与预期一致
    out_actual = algo.bootstrap(x, y, n_boot=2, func=f)
    out_wanted = np.array([[5, 10], [5, 10]])
    assert_array_equal(out_actual, out_wanted)

# 测试函数，验证 bootstrap 函数的 axis 参数是否正确处理
def test_bootstrap_axis(random):
    """Test axis kwarg to bootstrap function."""
    # 创建一个形状为 (10, 20) 的随机数组
    x = np.random.randn(10, 20)
    n_boot = 100

    # 使用默认参数进行 bootstrap，并验证输出结果的形状是否正确
    out_default = algo.bootstrap(x, n_boot=n_boot)
    assert out_default.shape == (n_boot,)

    # 使用指定 axis 参数进行 bootstrap，并验证输出结果的形状是否正确
    out_axis = algo.bootstrap(x, n_boot=n_boot, axis=0)
    assert out_axis.shape, (n_boot, x.shape[1])

# 测试函数，验证设置随机数种子后能够获得可重现的重抽样结果
def test_bootstrap_seed(random):
    """Test that we can get reproducible resamples by seeding the RNG."""
    # 创建一个包含 50 个正态分布随机数的数组
    data = np.random.randn(50)
    seed = 42
    # 使用指定种子进行 bootstrap 操作，并验证两次操作的结果是否一致
    boots1 = algo.bootstrap(data, seed=seed)
    boots2 = algo.bootstrap(data, seed=seed)
    assert_array_equal(boots1, boots2)

# 测试函数，验证 bootstrap 在线性回归模型拟合中的应用
def test_bootstrap_ols(random):
    """Test bootstrap of OLS model fit."""
    # 定义一个用于执行 OLS 拟合的函数
    def ols_fit(X, y):
        XtXinv = np.linalg.inv(np.dot(X.T, X))
        return XtXinv.dot(X.T).dot(y)

    # 创建输入特征矩阵 X 和目标值数组 y
    X = np.column_stack((np.random.randn(50, 4), np.ones(50)))
    w = [2, 4, 0, 3, 5]
    y_noisy = np.dot(X, w) + np.random.randn(50) * 20
    y_lownoise = np.dot(X, w) + np.random.randn(50)

    n_boot = 500
    # 对含噪声和低噪声的目标值进行 bootstrap 拟合，并验证输出结果的形状是否正确
    w_boot_noisy = algo.bootstrap(X, y_noisy,
                                  n_boot=n_boot,
                                  func=ols_fit)
    w_boot_lownoise = algo.bootstrap(X, y_lownoise,
                                     n_boot=n_boot,
                                     func=ols_fit)

    assert w_boot_noisy.shape == (n_boot, 5)
    assert w_boot_lownoise.shape == (n_boot, 5)
    # 断言：验证 w_boot_noisy 的标准差大于 w_boot_lownoise 的标准差
    assert w_boot_noisy.std() > w_boot_lownoise.std()
# 测试函数，验证通过单位ID进行自助法时结果是否合理
def test_bootstrap_units(random):
    # 生成包含50个随机数的数据
    data = np.random.randn(50)
    # 创建包含重复单位ID的数组，每个ID重复5次
    ids = np.repeat(range(10), 5)
    # 生成一个长度为10的正态分布随机数数组作为误差
    bwerr = np.random.normal(0, 2, 10)
    # 根据单位ID选择相应的误差值，构造带有误差的数据
    bwerr = bwerr[ids]
    # 将原始数据与误差相加，生成带有误差的数据
    data_rm = data + bwerr
    # 设定随机种子
    seed = 77

    # 进行自助法计算，不使用单位ID
    boots_orig = algo.bootstrap(data_rm, seed=seed)
    # 进行自助法计算，使用单位ID
    boots_rm = algo.bootstrap(data_rm, units=ids, seed=seed)
    # 断言使用单位ID的情况下标准差大于不使用单位ID的情况
    assert boots_rm.std() > boots_orig.std()


# 测试函数，验证当传入的参数长度不同时是否引发 ValueError 异常
def test_bootstrap_arglength():
    with pytest.raises(ValueError):
        # 调用自助法函数，传入长度分别为5和10的两个数组
        algo.bootstrap(np.arange(5), np.arange(10))


# 测试函数，验证命名的 numpy 方法与 numpy 函数的结果是否一致
def test_bootstrap_string_func():
    # 生成包含100个随机数的数组
    x = np.random.randn(100)

    # 使用自助法计算均值，传入字符串形式的函数名和 numpy 函数形式
    res_a = algo.bootstrap(x, func="mean", seed=0)
    res_b = algo.bootstrap(x, func=np.mean, seed=0)
    assert np.array_equal(res_a, res_b)

    # 使用自助法计算标准差，传入字符串形式的函数名和 numpy 函数形式
    res_a = algo.bootstrap(x, func="std", seed=0)
    res_b = algo.bootstrap(x, func=np.std, seed=0)
    assert np.array_equal(res_a, res_b)

    # 断言传入不存在的方法名时会引发 AttributeError 异常
    with pytest.raises(AttributeError):
        algo.bootstrap(x, func="not_a_method_name")


# 测试函数，验证自助法的可重现性
def test_bootstrap_reproducibility(random):
    # 生成包含50个随机数的数组
    data = np.random.randn(50)
    # 使用相同的种子进行两次自助法计算，断言结果数组相等
    boots1 = algo.bootstrap(data, seed=100)
    boots2 = algo.bootstrap(data, seed=100)
    assert_array_equal(boots1, boots2)

    # 使用自定义的随机状态对象作为种子，再次验证自助法的结果是否一致
    random_state1 = np.random.RandomState(200)
    boots1 = algo.bootstrap(data, seed=random_state1)
    random_state2 = np.random.RandomState(200)
    boots2 = algo.bootstrap(data, seed=random_state2)
    assert_array_equal(boots1, boots2)

    # 测试使用已弃用的 random_seed 参数时是否会发出警告
    with pytest.warns(UserWarning):
        boots1 = algo.bootstrap(data, random_seed=100)
        boots2 = algo.bootstrap(data, random_seed=100)
        assert_array_equal(boots1, boots2)


# 测试函数，验证自助法函数在处理包含 NaN 的数据时是否能正确处理
def test_nanaware_func_auto(random):
    # 生成包含10个正态分布随机数的数组，第一个数设为 NaN
    x = np.random.normal(size=10)
    x[0] = np.nan
    # 使用自助法计算均值，断言结果中不含 NaN
    boots = algo.bootstrap(x, func="mean")
    assert not np.isnan(boots).any()


# 测试函数，验证自助法函数在处理包含 NaN 的数据时是否能正确发出警告
def test_nanaware_func_warning(random):
    # 生成包含10个正态分布随机数的数组，第一个数设为 NaN
    x = np.random.normal(size=10)
    x[0] = np.nan
    # 使用自助法计算极差，断言结果中含有 NaN，并且会发出 UserWarning 警告
    with pytest.warns(UserWarning, match="Data contain nans but"):
        boots = algo.bootstrap(x, func="ptp")
    assert np.isnan(boots).any()
```