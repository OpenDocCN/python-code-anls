# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\tests\test_least_angle.py`

```
# 导入警告模块，用于处理警告信息
import warnings

# 导入科学计算相关库
import numpy as np
import pytest
from scipy import linalg

# 导入机器学习相关库
from sklearn import datasets, linear_model
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
    Lars,
    LarsCV,
    LassoLars,
    LassoLarsCV,
    LassoLarsIC,
    lars_path,
)
from sklearn.linear_model._least_angle import _lars_path_residues
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import (
    TempMemmap,
    assert_allclose,
    assert_array_almost_equal,
    ignore_warnings,
)

# 使用糖尿病数据集作为示例数据
diabetes = datasets.load_diabetes()
X, y = diabetes.data, diabetes.target

# 计算特征矩阵 X 的转置与自身的乘积，得到 Gram 矩阵
G = np.dot(X.T, X)
# 计算 X 的转置与目标向量 y 的乘积
Xy = np.dot(X.T, y)
# 样本数量
n_samples = y.size


def test_simple():
    # Lars 方法的原则是保持协方差相关性并递减

    # 同时测试详细输出
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    try:
        sys.stdout = StringIO()

        # 调用 lars_path 方法，使用 "lar" 方法，详细输出级别为 10
        _, _, coef_path_ = linear_model.lars_path(X, y, method="lar", verbose=10)

        sys.stdout = old_stdout

        # 遍历 coef_path_ 的转置，每次取出一个 coef_
        for i, coef_ in enumerate(coef_path_.T):
            # 计算残差
            res = y - np.dot(X, coef_)
            # 计算协方差
            cov = np.dot(X.T, res)
            # 计算最大的绝对协方差
            C = np.max(abs(cov))
            eps = 1e-3
            # 计算绝对协方差大于 C-eps 的数量
            ocur = len(cov[C - eps < abs(cov)])
            # 如果 i 小于 X 的列数，assert 发生的次数应为 i+1 次
            if i < X.shape[1]:
                assert ocur == i + 1
            else:
                # 否则，assert 发生的次数应为 X 的列数
                assert ocur == X.shape[1]
    finally:
        sys.stdout = old_stdout


def test_simple_precomputed():
    # 使用预计算的 Gram 矩阵进行相同的操作

    # 调用 lars_path 方法，使用预计算的 Gram 矩阵
    _, _, coef_path_ = linear_model.lars_path(X, y, Gram=G, method="lar")

    # 遍历 coef_path_ 的转置，每次取出一个 coef_
    for i, coef_ in enumerate(coef_path_.T):
        # 计算残差
        res = y - np.dot(X, coef_)
        # 计算协方差
        cov = np.dot(X.T, res)
        # 计算最大的绝对协方差
        C = np.max(abs(cov))
        eps = 1e-3
        # 计算绝对协方差大于 C-eps 的数量
        ocur = len(cov[C - eps < abs(cov)])
        # 如果 i 小于 X 的列数，assert 发生的次数应为 i+1 次
        if i < X.shape[1]:
            assert ocur == i + 1
        else:
            # 否则，assert 发生的次数应为 X 的列数
            assert ocur == X.shape[1]


def _assert_same_lars_path_result(output1, output2):
    # 断言两个 lars_path 的输出结果相同
    assert len(output1) == len(output2)
    for o1, o2 in zip(output1, output2):
        assert_allclose(o1, o2)


@pytest.mark.parametrize("method", ["lar", "lasso"])
@pytest.mark.parametrize("return_path", [True, False])
def test_lars_path_gram_equivalent(method, return_path):
    # 测试 lars_path 和 lars_path_gram 的等效性

    _assert_same_lars_path_result(
        # 调用 lars_path_gram 方法
        linear_model.lars_path_gram(
            Xy=Xy, Gram=G, n_samples=n_samples, method=method, return_path=return_path
        ),
        # 调用 lars_path 方法
        linear_model.lars_path(X, y, Gram=G, method=method, return_path=return_path),
    )


def test_x_none_gram_none_raises_value_error():
    # 测试当 X 和 Gram 都为 None 时，lars_path 方法是否会抛出 ValueError 异常
    Xy = np.dot(X.T, y)
    # 使用 pytest 的上下文管理器来验证是否会抛出 ValueError 异常，并检查异常消息是否包含特定字符串 "X and Gram cannot both be unspecified"
    with pytest.raises(ValueError, match="X and Gram cannot both be unspecified"):
        # 调用 linear_model 模块的 lars_path 函数，并传入参数
        linear_model.lars_path(None, y, Gram=None, Xy=Xy)
def test_all_precomputed():
    # Test that lars_path with precomputed Gram and Xy gives the right answer
    
    # 计算 Gram 矩阵，即 X 的转置乘以 X
    G = np.dot(X.T, X)
    
    # 计算 Xy，即 X 的转置乘以 y
    Xy = np.dot(X.T, y)
    
    # 对于 "lar" 和 "lasso" 两种方法，分别调用 lars_path 函数
    for method in "lar", "lasso":
        # 调用 lars_path 函数，使用默认的 Gram 和 Xy 参数
        output = linear_model.lars_path(X, y, method=method)
        
        # 调用 lars_path 函数，使用预先计算的 Gram 和 Xy 参数
        output_pre = linear_model.lars_path(X, y, Gram=G, Xy=Xy, method=method)
        
        # 对比两种调用结果的每一项
        for expected, got in zip(output, output_pre):
            assert_array_almost_equal(expected, got)


@pytest.mark.filterwarnings("ignore: `rcond` parameter will change")
# 忽略 numpy 的警告信息
def test_lars_lstsq():
    # Test that Lars gives least square solution at the end
    # of the path
    
    # 使用未标准化的数据集 X1
    X1 = 3 * X
    
    # 创建 LassoLars 对象 clf，并拟合数据
    clf = linear_model.LassoLars(alpha=0.0)
    clf.fit(X1, y)
    
    # 使用 np.linalg.lstsq 计算 X1 和 y 的最小二乘解
    coef_lstsq = np.linalg.lstsq(X1, y, rcond=None)[0]
    
    # 断言拟合后的系数与最小二乘解的系数相等
    assert_array_almost_equal(clf.coef_, coef_lstsq)


@pytest.mark.filterwarnings("ignore:`rcond` parameter will change")
# 忽略 numpy 的警告信息
def test_lasso_gives_lstsq_solution():
    # Test that Lars Lasso gives least square solution at the end
    # of the path
    
    # 调用 lars_path 函数，使用 "lasso" 方法
    _, _, coef_path_ = linear_model.lars_path(X, y, method="lasso")
    
    # 使用 np.linalg.lstsq 计算 X 和 y 的最小二乘解
    coef_lstsq = np.linalg.lstsq(X, y)[0]
    
    # 断言最小二乘解的系数与 lars_path 函数返回的系数路径的最后一列相等
    assert_array_almost_equal(coef_lstsq, coef_path_[:, -1])


def test_collinearity():
    # Check that lars_path is robust to collinearity in input
    
    # 创建具有共线性的输入数据 X 和目标变量 y
    X = np.array([[3.0, 3.0, 1.0], [2.0, 2.0, 0.0], [1.0, 1.0, 0]])
    y = np.array([1.0, 0.0, 0])
    
    # 使用 ignore_warnings 装饰器调用 lars_path 函数
    f = ignore_warnings
    _, _, coef_path_ = f(linear_model.lars_path)(X, y, alpha_min=0.01)
    
    # 断言系数路径中不含 NaN 值
    assert not np.isnan(coef_path_).any()
    
    # 计算预测值与实际值的残差
    residual = np.dot(X, coef_path_[:, -1]) - y
    
    # 断言残差的平方和小于 1.0
    assert (residual**2).sum() < 1.0

    # 生成随机数种子
    rng = np.random.RandomState(0)
    
    # 创建具有随机数据的输入数据集 X 和零目标变量 y
    n_samples = 10
    X = rng.rand(n_samples, 5)
    y = np.zeros(n_samples)
    
    # 调用 lars_path 函数，使用自动计算的 Gram 矩阵
    _, _, coef_path_ = linear_model.lars_path(
        X,
        y,
        Gram="auto",
        copy_X=False,
        copy_Gram=False,
        alpha_min=0.0,
        method="lasso",
        verbose=0,
        max_iter=500,
    )
    
    # 断言系数路径与全零数组相等
    assert_array_almost_equal(coef_path_, np.zeros_like(coef_path_))


def test_no_path():
    # Test that the ``return_path=False`` option returns the correct output
    
    # 调用 lars_path 函数，使用 "lar" 方法
    alphas_, _, coef_path_ = linear_model.lars_path(X, y, method="lar")
    
    # 调用 lars_path 函数，使用 "lar" 方法和 return_path=False 参数
    alpha_, _, coef = linear_model.lars_path(X, y, method="lar", return_path=False)
    
    # 断言返回的系数与系数路径的最后一列相等
    assert_array_almost_equal(coef, coef_path_[:, -1])
    
    # 断言返回的 alpha 值等于 alphas_ 中的最后一个值
    assert alpha_ == alphas_[-1]


def test_no_path_precomputed():
    # Test that the ``return_path=False`` option with Gram remains correct
    
    # 调用 lars_path 函数，使用 "lar" 方法和预先计算的 Gram 参数
    alphas_, _, coef_path_ = linear_model.lars_path(X, y, method="lar", Gram=G)
    
    # 调用 lars_path 函数，使用 "lar" 方法、预先计算的 Gram 参数和 return_path=False 参数
    alpha_, _, coef = linear_model.lars_path(
        X, y, method="lar", Gram=G, return_path=False
    )
    
    # 断言返回的系数与系数路径的最后一列相等
    assert_array_almost_equal(coef, coef_path_[:, -1])
    
    # 断言返回的 alpha 值等于 alphas_ 中的最后一个值
    assert alpha_ == alphas_[-1]


def test_no_path_all_precomputed():
    # Test that the ``return_path=False`` option with Gram and Xy remains
    # correct
    
    # 使用 diabetes 数据集的扩展数据 X 和目标变量 y
    X, y = 3 * diabetes.data, diabetes.target
    # 计算矩阵 X 的转置与自身的乘积，得到 Gram 矩阵 G
    G = np.dot(X.T, X)
    # 计算矩阵 X 的转置与向量 y 的乘积，得到 Xy 向量
    Xy = np.dot(X.T, y)
    # 使用 LARS（最小角回归）路径算法计算 Lasso 回归的路径
    # 返回 alphas_（正则化参数）、_（相关信息）、coef_path_（系数路径）
    alphas_, _, coef_path_ = linear_model.lars_path(
        X, y, method="lasso", Xy=Xy, Gram=G, alpha_min=0.9
    )
    # 使用 LARS（最小角回归）路径算法计算 Lasso 回归的路径
    # 返回 alpha_（最终选择的正则化参数）、_（相关信息）、coef（最终的系数）
    alpha_, _, coef = linear_model.lars_path(
        X, y, method="lasso", Gram=G, Xy=Xy, alpha_min=0.9, return_path=False
    )

    # 断言最终计算得到的系数 coef 与路径中最后一步的系数路径 coef_path_ 的近似相等
    assert_array_almost_equal(coef, coef_path_[:, -1])
    # 断言最终选择的正则化参数 alpha_ 等于 alphas_ 中的最后一个值
    assert alpha_ == alphas_[-1]
@pytest.mark.parametrize(
    "classifier", [linear_model.Lars, linear_model.LarsCV, linear_model.LassoLarsIC]
)
def test_lars_precompute(classifier):
    # 对于不同的预计算值进行检查
    G = np.dot(X.T, X)  # 计算输入数据的转置与自身的乘积，得到Gram矩阵

    clf = classifier(precompute=G)  # 使用给定的预计算Gram矩阵创建分类器对象
    output_1 = ignore_warnings(clf.fit)(X, y).coef_  # 忽略警告并拟合数据，获取系数向量
    for precompute in [True, False, "auto", None]:
        clf = classifier(precompute=precompute)  # 使用不同的预计算选项创建分类器对象
        output_2 = clf.fit(X, y).coef_  # 拟合数据并获取系数向量
        assert_array_almost_equal(output_1, output_2, decimal=8)  # 断言两个系数向量近似相等


def test_singular_matrix():
    # 测试输入为奇异矩阵的情况
    X1 = np.array([[1, 1.0], [1.0, 1.0]])
    y1 = np.array([1, 1])
    _, _, coef_path = linear_model.lars_path(X1, y1)  # 使用输入数据计算LARS路径
    assert_array_almost_equal(coef_path.T, [[0, 0], [1, 0]])  # 断言计算出的路径系数与预期值近似相等


def test_rank_deficient_design():
    # 一致性测试，检查LARS Lasso如何处理秩不足的输入数据（即特征数小于秩）
    y = [5, 0, 5]
    for X in ([[5, 0], [0, 5], [10, 10]], [[10, 10, 0], [1e-32, 0, 0], [0, 0, 1]]):
        # 为了能够使用系数计算目标函数，需要关闭归一化
        lars = linear_model.LassoLars(0.1)
        coef_lars_ = lars.fit(X, y).coef_  # 使用LARS Lasso拟合数据并获取系数
        obj_lars = 1.0 / (2.0 * 3.0) * linalg.norm(
            y - np.dot(X, coef_lars_)
        ) ** 2 + 0.1 * linalg.norm(coef_lars_, 1)  # 计算LARS Lasso的目标函数值
        coord_descent = linear_model.Lasso(0.1, tol=1e-6)
        coef_cd_ = coord_descent.fit(X, y).coef_  # 使用坐标下降Lasso拟合数据并获取系数
        obj_cd = (1.0 / (2.0 * 3.0)) * linalg.norm(
            y - np.dot(X, coef_cd_)
        ) ** 2 + 0.1 * linalg.norm(coef_cd_, 1)  # 计算坐标下降Lasso的目标函数值
        assert obj_lars < obj_cd * (1.0 + 1e-8)  # 断言LARS Lasso的目标函数值小于坐标下降Lasso的目标函数值


def test_lasso_lars_vs_lasso_cd():
    # 测试LassoLars和坐标下降Lasso是否给出相同的结果
    X = 3 * diabetes.data  # 数据放大3倍

    alphas, _, lasso_path = linear_model.lars_path(X, y, method="lasso")  # 使用LARS路径方法计算Lasso路径
    lasso_cd = linear_model.Lasso(fit_intercept=False, tol=1e-8)
    for c, a in zip(lasso_path.T, alphas):
        if a == 0:
            continue
        lasso_cd.alpha = a
        lasso_cd.fit(X, y)  # 使用坐标下降Lasso拟合数据
        error = linalg.norm(c - lasso_cd.coef_)  # 计算Lasso路径系数与坐标下降Lasso系数的误差
        assert error < 0.01  # 断言误差小于0.01

    # 类似的测试，使用分类器
    for alpha in np.linspace(1e-2, 1 - 1e-2, 20):
        clf1 = linear_model.LassoLars(alpha=alpha).fit(X, y)  # 使用LassoLars拟合数据
        clf2 = linear_model.Lasso(alpha=alpha, tol=1e-8).fit(X, y)  # 使用坐标下降Lasso拟合数据
        err = linalg.norm(clf1.coef_ - clf2.coef_)  # 计算两个模型的系数之间的误差
        assert err < 1e-3  # 断言误差小于1e-3

    # 同样的测试，使用归一化后的数据
    X = diabetes.data
    X = X - X.sum(axis=0)
    X /= np.linalg.norm(X, axis=0)
    alphas, _, lasso_path = linear_model.lars_path(X, y, method="lasso")  # 使用LARS路径方法计算Lasso路径
    lasso_cd = linear_model.Lasso(fit_intercept=False, tol=1e-8)
    for c, a in zip(lasso_path.T, alphas):
        if a == 0:
            continue
        lasso_cd.alpha = a
        lasso_cd.fit(X, y)  # 使用坐标下降Lasso拟合数据
        error = linalg.norm(c - lasso_cd.coef_)  # 计算Lasso路径系数与坐标下降Lasso系数的误差
        assert error < 0.01  # 断言误差小于0.01
def test_lasso_lars_vs_lasso_cd_early_stopping():
    # Test that LassoLars and Lasso using coordinate descent give the
    # same results when early stopping is used.
    # (test : before, in the middle, and in the last part of the path)
    alphas_min = [10, 0.9, 1e-4]

    X = diabetes.data  # 使用糖尿病数据集中的数据 X

    for alpha_min in alphas_min:
        # 计算 LassoLars 和 Lasso 的路径
        alphas, _, lasso_path = linear_model.lars_path(
            X, y, method="lasso", alpha_min=alpha_min
        )
        # 创建 Lasso 对象并使用坐标下降法拟合数据
        lasso_cd = linear_model.Lasso(fit_intercept=False, tol=1e-8)
        lasso_cd.alpha = alphas[-1]  # 设置 Lasso 对象的 alpha 参数
        lasso_cd.fit(X, y)  # 拟合数据
        # 计算最后一步的路径误差
        error = linalg.norm(lasso_path[:, -1] - lasso_cd.coef_)
        # 断言路径误差小于 0.01
        assert error < 0.01

    # 在规范化后进行相同的测试
    X = diabetes.data - diabetes.data.sum(axis=0)
    X /= np.linalg.norm(X, axis=0)

    for alpha_min in alphas_min:
        # 计算规范化后的 LassoLars 和 Lasso 的路径
        alphas, _, lasso_path = linear_model.lars_path(
            X, y, method="lasso", alpha_min=alpha_min
        )
        # 创建 Lasso 对象并使用坐标下降法拟合数据
        lasso_cd = linear_model.Lasso(tol=1e-8)
        lasso_cd.alpha = alphas[-1]  # 设置 Lasso 对象的 alpha 参数
        lasso_cd.fit(X, y)  # 拟合数据
        # 计算最后一步的路径误差
        error = linalg.norm(lasso_path[:, -1] - lasso_cd.coef_)
        # 断言路径误差小于 0.01
        assert error < 0.01


def test_lasso_lars_path_length():
    # Test that the path length of the LassoLars is right
    lasso = linear_model.LassoLars()
    lasso.fit(X, y)  # 使用数据 X, y 拟合 LassoLars 模型
    lasso2 = linear_model.LassoLars(alpha=lasso.alphas_[2])
    lasso2.fit(X, y)  # 使用数据 X, y 拟合带有特定 alpha 的 LassoLars 模型
    # 断言前三个 alpha 值序列相等
    assert_array_almost_equal(lasso.alphas_[:3], lasso2.alphas_)
    # 同时检查 alpha 序列是递减的
    assert np.all(np.diff(lasso.alphas_) < 0)


def test_lasso_lars_vs_lasso_cd_ill_conditioned():
    # Test lasso lars on a very ill-conditioned design, and check that
    # it does not blow up, and stays somewhat close to a solution given
    # by the coordinate descent solver
    # Also test that lasso_path (using lars_path output style) gives
    # the same result as lars_path and previous lasso output style
    # under these conditions.
    rng = np.random.RandomState(42)

    # Generate data
    n, m = 70, 100
    k = 5
    X = rng.randn(n, m)
    w = np.zeros((m, 1))
    i = np.arange(0, m)
    rng.shuffle(i)
    supp = i[:k]
    w[supp] = np.sign(rng.randn(k, 1)) * (rng.rand(k, 1) + 1)
    y = np.dot(X, w)
    sigma = 0.2
    y += sigma * rng.rand(*y.shape)
    y = y.squeeze()
    lars_alphas, _, lars_coef = linear_model.lars_path(X, y, method="lasso")

    _, lasso_coef2, _ = linear_model.lasso_path(X, y, alphas=lars_alphas, tol=1e-6)

    # 断言 LassoLars 和 Lasso 使用相同 alpha 路径得到的系数近似相等
    assert_array_almost_equal(lars_coef, lasso_coef2, decimal=1)


def test_lasso_lars_vs_lasso_cd_ill_conditioned2():
    # Create an ill-conditioned situation in which the LARS has to go
    # far in the path to converge, and check that LARS and coordinate
    # descent give the same answers
    # Note it used to be the case that Lars had to use the drop for good
    # strategy for this but this is no longer the case with the
    # equality_tolerance checks
    X = [[1e20, 1e20, 0], [-1e-32, 0, 0], [1, 1, 1]]
    y = [10, 10, 1]
    alpha = 0.0001

    # 定义目标函数，用于评估线性回归模型的拟合效果及稀疏性
    def objective_function(coef):
        # 计算平方误差的均值，并添加 L1 正则化项
        return 1.0 / (2.0 * len(X)) * linalg.norm(
            y - np.dot(X, coef)
        ) ** 2 + alpha * linalg.norm(coef, 1)

    # 使用 LassoLars 线性回归模型，设置正则化参数 alpha
    lars = linear_model.LassoLars(alpha=alpha)
    warning_message = "Regressors in active set degenerate."
    # 使用 pytest 来捕获 ConvergenceWarning 警告信息
    with pytest.warns(ConvergenceWarning, match=warning_message):
        # 使用 LassoLars 拟合数据 X, y
        lars.fit(X, y)
    # 获取 LassoLars 模型的系数
    lars_coef_ = lars.coef_
    # 计算 LassoLars 模型的目标函数值
    lars_obj = objective_function(lars_coef_)

    # 使用 Lasso 线性回归模型，设置正则化参数 alpha 和收敛阈值 tol
    coord_descent = linear_model.Lasso(alpha=alpha, tol=1e-4)
    # 使用坐标下降法拟合数据 X, y，并获取其系数
    cd_coef_ = coord_descent.fit(X, y).coef_
    # 计算坐标下降法拟合结果的目标函数值
    cd_obj = objective_function(cd_coef_)

    # 断言：确保 LassoLars 的目标函数值要小于坐标下降法的目标函数值乘以一个小的增量
    assert lars_obj < cd_obj * (1.0 + 1e-8)
def test_lars_add_features():
    # 确保在必要时至少添加一些特征
    # 使用 6d2b4c 进行测试
    # 创建 Hilbert 矩阵
    n = 5
    H = 1.0 / (np.arange(1, n + 1) + np.arange(n)[:, np.newaxis])
    # 使用 Lars 算法拟合数据，不包括截距项
    clf = linear_model.Lars(fit_intercept=False).fit(H, np.arange(n))
    # 断言所有的系数都是有限的
    assert np.all(np.isfinite(clf.coef_))


def test_lars_n_nonzero_coefs(verbose=False):
    # 创建具有指定非零系数个数的 Lars 模型
    lars = linear_model.Lars(n_nonzero_coefs=6, verbose=verbose)
    # 使用数据 X 和目标 y 拟合模型
    lars.fit(X, y)
    # 断言非零系数的数量等于 6
    assert len(lars.coef_.nonzero()[0]) == 6
    # Lars 模型路径的长度应为 6 + 1，表示从 6 个非零系数到 0 的路径
    assert len(lars.alphas_) == 7


@ignore_warnings
def test_multitarget():
    # 确保接收多维目标 y 的估计器表现正确
    Y = np.vstack([y, y**2]).T
    n_targets = Y.shape[1]
    estimators = [
        linear_model.LassoLars(),
        linear_model.Lars(),
        # 用于 gh-1615 的回归测试
        linear_model.LassoLars(fit_intercept=False),
        linear_model.Lars(fit_intercept=False),
    ]

    for estimator in estimators:
        # 使用数据 X 和目标 Y 拟合估计器
        estimator.fit(X, Y)
        # 预测目标值
        Y_pred = estimator.predict(X)
        # 提取模型属性
        alphas, active, coef, path = (
            estimator.alphas_,
            estimator.active_,
            estimator.coef_,
            estimator.coef_path_,
        )
        for k in range(n_targets):
            # 使用数据 X 和第 k 列目标 Y[:, k] 拟合估计器
            estimator.fit(X, Y[:, k])
            y_pred = estimator.predict(X)
            # 断言各项属性准确度
            assert_array_almost_equal(alphas[k], estimator.alphas_)
            assert_array_almost_equal(active[k], estimator.active_)
            assert_array_almost_equal(coef[k], estimator.coef_)
            assert_array_almost_equal(path[k], estimator.coef_path_)
            assert_array_almost_equal(Y_pred[:, k], y_pred)


def test_lars_cv():
    # 通过检查随着样本数量增加，LassoLarsCV 对象的最优 alpha 值增加来测试
    # 这一属性实际上并不一般保证，在给定数据集和选择的步骤下才成立
    old_alpha = 0
    lars_cv = linear_model.LassoLarsCV()
    for length in (400, 200, 100):
        X = diabetes.data[:length]
        y = diabetes.target[:length]
        lars_cv.fit(X, y)
        np.testing.assert_array_less(old_alpha, lars_cv.alpha_)
        old_alpha = lars_cv.alpha_
    # 断言 LassoLarsCV 没有 n_nonzero_coefs 属性
    assert not hasattr(lars_cv, "n_nonzero_coefs")


def test_lars_cv_max_iter(recwarn):
    warnings.simplefilter("always")
    with np.errstate(divide="raise", invalid="raise"):
        X = diabetes.data
        y = diabetes.target
        rng = np.random.RandomState(42)
        x = rng.randn(len(y))
        X = diabetes.data
        X = np.c_[X, x, x]  # 添加相关特征
        X = StandardScaler().fit_transform(X)
        # 创建 LassoLarsCV 对象，设定最大迭代次数和交叉验证折数
        lars_cv = linear_model.LassoLarsCV(max_iter=5, cv=5)
        lars_cv.fit(X, y)

    # 检查总体上没有警告产生，特别是没有 ConvergenceWarning
    # 将警告的字符串表示形式转换为列表，以便在发生 AssertionError 时获得更详细的错误信息。
    recorded_warnings = [str(w) for w in recwarn]
    # 断言确保记录的警告数量为零，否则引发 AssertionError。
    assert len(recorded_warnings) == 0
# 测试 LassoLarsIC 对象的功能，验证以下内容：
# - 选择了一些好的特征。
# - alpha_bic > alpha_aic。
# - n_nonzero_bic < n_nonzero_aic。
def test_lasso_lars_ic():
    # 创建 LassoLarsIC 对象，使用 BIC 准则
    lars_bic = linear_model.LassoLarsIC("bic")
    # 创建 LassoLarsIC 对象，使用 AIC 准则
    lars_aic = linear_model.LassoLarsIC("aic")
    # 设置随机种子
    rng = np.random.RandomState(42)
    # 获取糖尿病数据集的特征
    X = diabetes.data
    # 向数据集添加 5 个不好的特征
    X = np.c_[X, rng.randn(X.shape[0], 5)]
    # 对数据进行标准化
    X = StandardScaler().fit_transform(X)
    # 使用 lars_bic 拟合数据
    lars_bic.fit(X, y)
    # 使用 lars_aic 拟合数据
    lars_aic.fit(X, y)
    # 获取 lars_bic 的非零系数索引
    nonzero_bic = np.where(lars_bic.coef_)[0]
    # 获取 lars_aic 的非零系数索引
    nonzero_aic = np.where(lars_aic.coef_)[0]
    # 断言 alpha_bic 大于 alpha_aic
    assert lars_bic.alpha_ > lars_aic.alpha_
    # 断言 lars_bic 的非零系数个数小于 lars_aic 的非零系数个数
    assert len(nonzero_bic) < len(nonzero_aic)
    # 断言最大的非零系数索引小于糖尿病数据集特征的数量
    assert np.max(nonzero_bic) < diabetes.data.shape[1]


def test_lars_path_readonly_data():
    # 在处理大型输入时使用自动内存映射，折叠数据以只读模式存储
    # 这是一个非回归测试，用于检查：https://github.com/scikit-learn/scikit-learn/issues/4597
    # 对数据进行分割
    splitted_data = train_test_split(X, y, random_state=42)
    # 使用 TempMemmap 上下文管理器来处理分割后的数据
    with TempMemmap(splitted_data) as (X_train, X_test, y_train, y_test):
        # 使用 _lars_path_residues 函数处理数据，确保不复制数据（copy=False）
        _lars_path_residues(X_train, y_train, X_test, y_test, copy=False)


def test_lars_path_positive_constraint():
    # 这是对 lars_path 方法中 positive 参数的主要测试
    # 估计器类别仅仅是利用了这个函数

    # 我们在糖尿病数据集上进行测试

    # 确保在 positive=False 时获得负系数，并且在 positive=True 时获得所有正系数
    err_msg = "Positive constraint not supported for 'lar' coding method."
    # 使用 pytest 断言检查是否会引发 ValueError 异常，并匹配特定的错误消息
    with pytest.raises(ValueError, match=err_msg):
        # 使用 'lar' 方法和 positive=True 参数调用 lars_path 函数
        linear_model.lars_path(
            diabetes["data"], diabetes["target"], method="lar", positive=True
        )

    # 设置方法为 'lasso'
    method = "lasso"
    # 调用 lars_path 函数，返回路径和系数
    _, _, coefs = linear_model.lars_path(
        X, y, return_path=True, method=method, positive=False
    )
    # 断言系数的最小值小于 0
    assert coefs.min() < 0

    # 再次调用 lars_path 函数，返回路径和系数
    _, _, coefs = linear_model.lars_path(
        X, y, return_path=True, method=method, positive=True
    )
    # 断言系数的最小值大于等于 0
    assert coefs.min() >= 0


# 现在我们将测试所有估计器类别的 positive 选项

# 默认参数设置
default_parameter = {"fit_intercept": False}

# 估计器类别与参数映射
estimator_parameter_map = {
    "LassoLars": {"alpha": 0.1},
    "LassoLarsCV": {},
    "LassoLarsIC": {},
}


def test_estimatorclasses_positive_constraint():
    # 测试所有估计器类别的 positive 选项传递性
    default_parameter = {"fit_intercept": False}

    estimator_parameter_map = {
        "LassoLars": {"alpha": 0.1},
        "LassoLarsCV": {},
        "LassoLarsIC": {},
    }
    # 遍历 estimator_parameter_map 字典中的每个估计器名称
    for estname in estimator_parameter_map:
        # 复制默认参数字典到 params 变量
        params = default_parameter.copy()
        # 更新 params 字典，使用 estimator_parameter_map 中对应估计器名称的参数
        params.update(estimator_parameter_map[estname])
        # 根据 estname 从 linear_model 模块中获取对应的估计器类，并初始化该估计器对象
        estimator = getattr(linear_model, estname)(positive=False, **params)
        # 使用 X 和 y 数据对该估计器对象进行拟合
        estimator.fit(X, y)
        # 断言估计器的系数中最小值小于0，即所有系数都是负数
        assert estimator.coef_.min() < 0
        # 根据 estname 从 linear_model 模块中获取对应的估计器类，并初始化该估计器对象（positive=True）
        estimator = getattr(linear_model, estname)(positive=True, **params)
        # 使用 X 和 y 数据对该估计器对象进行拟合
        estimator.fit(X, y)
        # 断言估计器的系数中最小值大于或等于0，即所有系数都是非负数
        assert min(estimator.coef_) >= 0
# 测试当使用正向选项时，LassoLars 和使用坐标下降的 Lasso 是否给出相同的结果
def test_lasso_lars_vs_lasso_cd_positive():
    # 不规范化的数据
    X = 3 * diabetes.data

    # 使用 LassoLars 方法计算 Lasso 路径，包括正向选项
    alphas, _, lasso_path = linear_model.lars_path(X, y, method="lasso", positive=True)
    
    # 创建 Lasso 对象，使用坐标下降法，包括正向选项
    lasso_cd = linear_model.Lasso(fit_intercept=False, tol=1e-8, positive=True)
    
    # 遍历 Lasso 路径中的系数和对应的 alpha 值
    for c, a in zip(lasso_path.T, alphas):
        if a == 0:
            continue
        # 设置 Lasso 对象的 alpha 值并拟合数据
        lasso_cd.alpha = a
        lasso_cd.fit(X, y)
        # 计算系数之间的误差范数，断言误差小于 0.01
        error = linalg.norm(c - lasso_cd.coef_)
        assert error < 0.01

    # 在比较系数值时，选择了一定范围的 alpha 值，此范围比未使用正向选项时更为受限
    # 原因是 Lars-Lasso 算法对于较小的 alpha 值不会收敛到最小二乘解，详见 Efron 等人 2004 年的 'Least Angle Regression'
    # 在 Lars-Lasso 算法达到最小 alpha 值时，系数通常是一致的，但在此之后可能会出现差异。
    # 参考 https://gist.github.com/michigraber/7e7d7c75eca694c7a6ff

    # 对于一系列 alpha 值进行比较，确保 LassoLars 和 Lasso 实现的系数差异小于 1e-3
    for alpha in np.linspace(6e-1, 1 - 1e-2, 20):
        clf1 = linear_model.LassoLars(
            fit_intercept=False, alpha=alpha, positive=True
        ).fit(X, y)
        clf2 = linear_model.Lasso(
            fit_intercept=False, alpha=alpha, tol=1e-8, positive=True
        ).fit(X, y)
        err = linalg.norm(clf1.coef_ - clf2.coef_)
        assert err < 1e-3

    # 规范化的数据
    X = diabetes.data - diabetes.data.sum(axis=0)
    X /= np.linalg.norm(X, axis=0)
    
    # 使用 LassoLars 方法计算 Lasso 路径，包括正向选项
    alphas, _, lasso_path = linear_model.lars_path(X, y, method="lasso", positive=True)
    # 创建 Lasso 对象，使用坐标下降法，包括正向选项
    lasso_cd = linear_model.Lasso(fit_intercept=False, tol=1e-8, positive=True)
    
    # 遍历 Lasso 路径中的系数和对应的 alpha 值，不包括 alpha=0 的情况
    for c, a in zip(lasso_path.T[:-1], alphas[:-1]):
        # 设置 Lasso 对象的 alpha 值并拟合数据
        lasso_cd.alpha = a
        lasso_cd.fit(X, y)
        # 计算系数之间的误差范数，断言误差小于 0.01
        error = linalg.norm(c - lasso_cd.coef_)
        assert error < 0.01


# 测试 sklearn 的 LassoLars 实现与 R 中的 LassoLars 实现（lars 库）在 fit_intercept=False 时是否一致
def test_lasso_lars_vs_R_implementation():
    # 生成在 bug 报告 7778 中使用的数据
    y = np.array([-6.45006793, -3.51251449, -8.52445396, 6.12277822, -19.42109366])
    x = np.array(
        [
            [0.47299829, 0, 0, 0, 0],
            [0.08239882, 0.85784863, 0, 0, 0],
            [0.30114139, -0.07501577, 0.80895216, 0, 0],
            [-0.01460346, -0.1015233, 0.0407278, 0.80338378, 0],
            [-0.69363927, 0.06754067, 0.18064514, -0.0803561, 0.40427291],
        ]
    )

    X = x.T

    # R 中的结果是使用以下代码获取的：
    #
    # 导入库：lars（该行注释原始为R语言的注释，但在Python中不起作用）
    # 创建 LassoLars 模型对象，设定参数 alpha=0 表示使用 Lasso 模型，fit_intercept=False 表示不拟合截距
    model_lasso_lars = linear_model.LassoLars(alpha=0, fit_intercept=False)
    # 使用模型对象对数据集 X 和目标变量 y 进行拟合
    model_lasso_lars.fit(X, y)
    # 从拟合的模型中获取系数路径（即稀疏解的系数）
    skl_betas = model_lasso_lars.coef_path_
    
    # 使用 numpy 创建一个预期的系数矩阵 r
    r = np.array(
        [
            [0, 0, 0, 0, 0, -79.810362809499026, -83.528788732782829, -83.777653739190711, -83.784156932888934, -84.033390591756657],
            [0, 0, 0, 0, -0.476624256777266, 0, 0, 0, 0, 0.025219751009936],
            [0, -3.577397088285891, -4.702795355871871, -7.016748621359461, -7.614898471899412, -0.336938391359179, 0, 0, 0.001213370600853, 0.048162321585148],
            [0, 0, 0, 2.231558436628169, 2.723267514525966, 2.811549786389614, 2.813766976061531, 2.817462468949557, 2.817368178703816, 2.816221090636795],
            [0, 0, -1.218422599914637, -3.457726183014808, -4.021304522060710, -45.827461592423745, -47.776608869312305, -47.911561610746404, -47.914845922736234, -48.039562334265717],
        ]
    )
    
    # 断言：验证 skl_betas 与预期的系数矩阵 r 几乎相等，精度为 12 位小数
    assert_array_almost_equal(r, skl_betas, decimal=12)
# 使用 pytest 的 mark 来参数化测试函数，分别测试 copy_X 为 True 和 False 时的行为
@pytest.mark.parametrize("copy_X", [True, False])
def test_lasso_lars_copyX_behaviour(copy_X):
    """
    Test that user input regarding copy_X is not being overridden (it was until
    at least version 0.21)

    """
    # 创建 LassoLarsIC 实例，根据传入的 copy_X 参数决定是否复制输入数据
    lasso_lars = LassoLarsIC(copy_X=copy_X, precompute=False)
    # 使用随机种子生成 X 数据，形状为 (100, 5)，并复制一份作为 X_copy
    rng = np.random.RandomState(0)
    X = rng.normal(0, 1, (100, 5))
    X_copy = X.copy()
    # 从 X 中选择第二列作为目标 y
    y = X[:, 2]
    # 使用 fit 方法拟合模型
    lasso_lars.fit(X, y)
    # 断言确保 copy_X 参数是否按预期工作，即 X 是否与 X_copy 相等
    assert copy_X == np.array_equal(X, X_copy)


# 使用 pytest 的 mark 来参数化测试函数，分别测试 copy_X 为 True 和 False 时的行为
@pytest.mark.parametrize("copy_X", [True, False])
def test_lasso_lars_fit_copyX_behaviour(copy_X):
    """
    Test that user input to .fit for copy_X overrides default __init__ value

    """
    # 创建 LassoLarsIC 实例，预计算参数 precompute 设置为 False
    lasso_lars = LassoLarsIC(precompute=False)
    # 使用随机种子生成 X 数据，形状为 (100, 5)，并复制一份作为 X_copy
    rng = np.random.RandomState(0)
    X = rng.normal(0, 1, (100, 5))
    X_copy = X.copy()
    # 从 X 中选择第二列作为目标 y
    y = X[:, 2]
    # 使用 fit 方法拟合模型，并传入 copy_X 参数以覆盖默认的初始化值
    lasso_lars.fit(X, y, copy_X=copy_X)
    # 断言确保 copy_X 参数是否按预期工作，即 X 是否与 X_copy 相等
    assert copy_X == np.array_equal(X, X_copy)


def test_lars_with_jitter(est):
    # Test that a small amount of jitter helps stability,
    # using example provided in issue #2746

    # 使用一个特定的例子来测试小量的抖动是否有助于稳定性
    X = np.array([[0.0, 0.0, 0.0, -1.0, 0.0], [0.0, -1.0, 0.0, 0.0, 0.0]])
    y = [-2.5, -2.5]
    expected_coef = [0, 2.5, 0, 2.5, 0]

    # 将 fit_intercept 设置为 False，因为目标是常量且我们希望检查 coef 的值
    est.set_params(fit_intercept=False)
    # 克隆模型并设置抖动参数以及随机种子
    est_jitter = clone(est).set_params(jitter=10e-8, random_state=0)

    # 分别使用原始模型和添加抖动的模型来拟合数据
    est.fit(X, y)
    est_jitter.fit(X, y)

    # 断言确保添加抖动后的模型的 coef 与预期的 coef 接近，且均方差大于 0.1
    assert np.mean((est.coef_ - est_jitter.coef_) ** 2) > 0.1
    np.testing.assert_allclose(est_jitter.coef_, expected_coef, rtol=1e-3)


def test_X_none_gram_not_none():
    # 使用 pytest 来确保在 Gram 不为空时，X 不能为 None
    with pytest.raises(ValueError, match="X cannot be None if Gram is not None"):
        lars_path(X=None, y=np.array([1]), Gram=True)


def test_copy_X_with_auto_gram():
    # 针对 issue #17789 进行的非回归测试，确保 `copy_X=True` 和 Gram='auto' 不会覆盖 X
    rng = np.random.RandomState(42)
    X = rng.rand(6, 6)
    y = rng.rand(6)

    # 保存原始的 X 数据
    X_before = X.copy()
    # 调用 linear_model.lars_path 来测试
    linear_model.lars_path(X, y, Gram="auto", copy_X=True, method="lasso")
    # 断言检查 X 是否有变化
    assert_allclose(X, X_before)


# 使用 pytest 的 mark 参数化来测试不同的 LARS 模型及参数类型
@pytest.mark.parametrize(
    "LARS, has_coef_path, args",
    (
        (Lars, True, {}),
        (LassoLars, True, {}),
        (LassoLarsIC, False, {}),
        (LarsCV, True, {}),
        # max_iter=5 是为了避免 ConvergenceWarning
        (LassoLarsCV, True, {"max_iter": 5}),
    ),
)
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_lars_dtype_match(LARS, has_coef_path, args, dtype):
    # 该测试确保 fit 方法保留输入数据的 dtype
    rng = np.random.RandomState(0)
    X = rng.rand(20, 6).astype(dtype)
    y = rng.rand(20).astype(dtype)

    # 创建 LARS 模型，并拟合数据
    model = LARS(**args)
    model.fit(X, y)
    # 断言检查 coef_ 的 dtype 是否与输入的 dtype 一致
    assert model.coef_.dtype == dtype
    if has_coef_path:
        assert model.coef_path_.dtype == dtype
    assert model.intercept_.dtype == dtype
@pytest.mark.parametrize(
    "LARS, has_coef_path, args",
    (
        (Lars, True, {}),
        (LassoLars, True, {}),
        (LassoLarsIC, False, {}),
        (LarsCV, True, {}),
        # max_iter=5 is for avoiding ConvergenceWarning
        (LassoLarsCV, True, {"max_iter": 5}),
    ),
)
def test_lars_numeric_consistency(LARS, has_coef_path, args):
    # 测试确保在 float32 和 float64 训练系数之间的数值一致性
    rtol = 1e-5  # 相对误差容限
    atol = 1e-5  # 绝对误差容限

    rng = np.random.RandomState(0)
    X_64 = rng.rand(10, 6)  # 创建一个 10x6 的随机数组 X_64
    y_64 = rng.rand(10)     # 创建一个长度为 10 的随机数组 y_64

    model_64 = LARS(**args).fit(X_64, y_64)  # 使用 float64 数据拟合 LARS 模型
    model_32 = LARS(**args).fit(X_64.astype(np.float32), y_64.astype(np.float32))  # 使用 float32 数据拟合 LARS 模型

    assert_allclose(model_64.coef_, model_32.coef_, rtol=rtol, atol=atol)  # 断言两个模型的系数近似相等
    if has_coef_path:
        assert_allclose(model_64.coef_path_, model_32.coef_path_, rtol=rtol, atol=atol)  # 如果模型有系数路径信息，则断言路径近似相等
    assert_allclose(model_64.intercept_, model_32.intercept_, rtol=rtol, atol=atol)  # 断言两个模型的截距近似相等


@pytest.mark.parametrize("criterion", ["aic", "bic"])
def test_lassolarsic_alpha_selection(criterion):
    """检查我们是否正确计算了 AIC 和 BIC 分数。

    在这个测试中，我们复现了 Zou 等人在 LassoLarsIC 中参考文献 [1] 的图 2 示例。
    在这个示例中，应该选择 7 个特征。
    """
    model = make_pipeline(StandardScaler(), LassoLarsIC(criterion=criterion))
    model.fit(X, y)

    best_alpha_selected = np.argmin(model[-1].criterion_)
    assert best_alpha_selected == 7


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_lassolarsic_noise_variance(fit_intercept):
    """检查当 `n_samples` < `n_features` 时的行为，并且需要提供噪声方差。"""
    rng = np.random.RandomState(0)
    X, y = datasets.make_regression(
        n_samples=10, n_features=11 - fit_intercept, random_state=rng
    )

    model = make_pipeline(StandardScaler(), LassoLarsIC(fit_intercept=fit_intercept))

    err_msg = (
        "You are using LassoLarsIC in the case where the number of samples is smaller"
        " than the number of features"
    )
    with pytest.raises(ValueError, match=err_msg):
        model.fit(X, y)

    model.set_params(lassolarsic__noise_variance=1.0)
    model.fit(X, y).predict(X)
```