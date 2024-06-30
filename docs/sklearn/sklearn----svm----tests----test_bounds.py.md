# `D:\src\scipysrc\scikit-learn\sklearn\svm\tests\test_bounds.py`

```
# 导入必要的库
import numpy as np
import pytest
from scipy import stats

# 导入机器学习相关模块
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm._bounds import l1_min_c
from sklearn.svm._newrand import bounded_rand_int_wrap, set_seed_wrap
from sklearn.utils.fixes import CSR_CONTAINERS

# 定义一个密集型数据集
dense_X = [[-1, 0], [0, 1], [1, 1], [1, 1]]

# 定义两种不同的分类标签
Y1 = [0, 1, 1, 1]
Y2 = [2, 1, 0, 0]

# 使用pytest的parametrize装饰器来定义多组测试参数
@pytest.mark.parametrize("X_container", CSR_CONTAINERS + [np.array])
@pytest.mark.parametrize("loss", ["squared_hinge", "log"])
@pytest.mark.parametrize("Y_label", ["two-classes", "multi-class"])
@pytest.mark.parametrize("intercept_label", ["no-intercept", "fit-intercept"])
def test_l1_min_c(X_container, loss, Y_label, intercept_label):
    # 定义两种不同的标签对应的标签值
    Ys = {"two-classes": Y1, "multi-class": Y2}
    # 定义两种不同的截距设置
    intercepts = {
        "no-intercept": {"fit_intercept": False},
        "fit-intercept": {"fit_intercept": True, "intercept_scaling": 10},
    }

    # 创建X数据对象
    X = X_container(dense_X)
    # 根据Y_label选择对应的标签数据
    Y = Ys[Y_label]
    # 根据intercept_label选择对应的截距参数
    intercept_params = intercepts[intercept_label]
    # 调用check_l1_min_c函数进行测试
    check_l1_min_c(X, Y, loss, **intercept_params)


# 定义一个函数，用于检查给定条件下的最小C值
def check_l1_min_c(X, y, loss, fit_intercept=True, intercept_scaling=1.0):
    # 计算满足L1正则化的最小C值
    min_c = l1_min_c(
        X,
        y,
        loss=loss,
        fit_intercept=fit_intercept,
        intercept_scaling=intercept_scaling,
    )

    # 根据不同的损失函数类型，选择相应的分类器对象
    clf = {
        "log": LogisticRegression(penalty="l1", solver="liblinear"),
        "squared_hinge": LinearSVC(loss="squared_hinge", penalty="l1", dual=False),
    }[loss]

    # 设置分类器的截距和截距缩放参数
    clf.fit_intercept = fit_intercept
    clf.intercept_scaling = intercept_scaling

    # 将最小的C值设置给分类器
    clf.C = min_c
    # 使用数据进行分类器训练
    clf.fit(X, y)
    # 断言分类器的所有系数都为零
    assert (np.asarray(clf.coef_) == 0).all()
    # 断言分类器的截距都为零
    assert (np.asarray(clf.intercept_) == 0).all()

    # 将略微大于最小C值的值设置给分类器
    clf.C = min_c * 1.01
    # 再次使用数据进行分类器训练
    clf.fit(X, y)
    # 断言分类器的系数中至少有一个不为零，或者截距不为零
    assert (np.asarray(clf.coef_) != 0).any() or (np.asarray(clf.intercept_) != 0).any()


# 定义一个测试函数，用于测试异常情况下的最小C值计算
def test_ill_posed_min_c():
    X = [[0, 0], [0, 0]]
    y = [0, 1]
    # 使用pytest的raises断言检查是否抛出异常
    with pytest.raises(ValueError):
        l1_min_c(X, y)


# 定义一个测试函数，用于测试随机数生成函数的默认行为
def test_newrand_default():
    """Test that bounded_rand_int_wrap without seeding respects the range

    Note this test should pass either if executed alone, or in conjunctions
    with other tests that call set_seed explicit in any order: it checks
    invariants on the RNG instead of specific values.
    """
    # 生成一组随机数，并断言它们在指定范围内
    generated = [bounded_rand_int_wrap(100) for _ in range(10)]
    assert all(0 <= x < 100 for x in generated)
    # 断言生成的随机数不全相等
    assert not all(x == generated[0] for x in generated)


# 使用pytest的parametrize装饰器定义多组参数化测试
@pytest.mark.parametrize("seed, expected", [(0, 54), (_MAX_UNSIGNED_INT, 9)])
def test_newrand_set_seed(seed, expected):
    """Test that `set_seed` produces deterministic results"""
    # 设置随机数种子，并生成随机数
    set_seed_wrap(seed)
    generated = bounded_rand_int_wrap(100)
    # 断言生成的随机数符合预期值
    assert generated == expected


# 使用pytest的parametrize装饰器定义多组参数化测试
@pytest.mark.parametrize("seed", [-1, _MAX_UNSIGNED_INT + 1])
def test_newrand_set_seed_overflow(seed):
    """Test that `set_seed_wrap` is defined for unsigned 32bits ints"""
    # 测试无符号32位整数范围外的种子值
    # 使用 pytest 框架的 raises 方法来检测是否会抛出 OverflowError 异常
    with pytest.raises(OverflowError):
        # 调用 set_seed_wrap 函数，并尝试设置给定的种子值 seed
        set_seed_wrap(seed)
@pytest.mark.parametrize("range_, n_pts", [(_MAX_UNSIGNED_INT, 10000), (100, 25)])
def test_newrand_bounded_rand_int(range_, n_pts):
    """Test that `bounded_rand_int` follows a uniform distribution"""
    # XXX: this test is very seed sensitive: either it is wrong (too strict?)
    # or the wrapped RNG is not uniform enough, at least on some platforms.
    # 设置随机数种子为固定值，确保测试结果可重现性
    set_seed_wrap(42)
    # 循环次数
    n_iter = 100
    # 存储 KS 检验的 p-value
    ks_pvals = []
    # 创建一个指定范围的均匀分布
    uniform_dist = stats.uniform(loc=0, scale=range_)
    
    # 执行多次采样以减少异常采样的机会
    for _ in range(n_iter):
        # 使用确定性的随机采样
        sample = [bounded_rand_int_wrap(range_) for _ in range(n_pts)]
        # 对采样数据进行 KS 检验
        res = stats.kstest(sample, uniform_dist.cdf)
        # 将检验结果的 p-value 存入列表
        ks_pvals.append(res.pvalue)
    
    # 假设检验的原假设 = 样本来自均匀分布
    # 在原假设下，p-value 应该是均匀分布的，并不集中在较低的值
    # （这可能看起来反直觉，但有多篇文献支持）
    # 因此我们可以进行两项检查：

    # (1) 检查 p-value 的均匀性
    uniform_p_vals_dist = stats.uniform(loc=0, scale=1)
    res_pvals = stats.kstest(ks_pvals, uniform_p_vals_dist.cdf)
    assert res_pvals.pvalue > 0.05, (
        "Null hypothesis rejected: generated random numbers are not uniform."
        " Details: the (meta) p-value of the test of uniform distribution"
        f" of p-values is {res_pvals.pvalue} which is not > 0.05"
    )

    # (2) (safety belt) 检查 90% 的 p-value 是否大于 0.05
    min_10pct_pval = np.percentile(ks_pvals, q=10)
    # p-value 的下 10% 分位数小于等于 0.05 表示拒绝原假设，即样本不来自均匀分布
    assert min_10pct_pval > 0.05, (
        "Null hypothesis rejected: generated random numbers are not uniform. "
        f"Details: lower 10th quantile p-value of {min_10pct_pval} not > 0.05."
    )


@pytest.mark.parametrize("range_", [-1, _MAX_UNSIGNED_INT + 1])
def test_newrand_bounded_rand_int_limits(range_):
    """Test that `bounded_rand_int_wrap` is defined for unsigned 32bits ints"""
    # 使用 pytest.raises 检测 OverflowError 异常
    with pytest.raises(OverflowError):
        bounded_rand_int_wrap(range_)


注释完成，代码块完整且符合规范。
```