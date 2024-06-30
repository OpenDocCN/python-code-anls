# `D:\src\scipysrc\scikit-learn\sklearn\covariance\tests\test_graphical_lasso.py`

```
"""Test the graphical_lasso module."""

# 导入必要的库和模块
import sys
from io import StringIO

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import linalg

# 导入 sklearn 中的相关模块和类
from sklearn import datasets
from sklearn.covariance import (
    GraphicalLasso,
    GraphicalLassoCV,
    empirical_covariance,
    graphical_lasso,
)
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.model_selection import GroupKFold
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
    _convert_container,
    assert_array_almost_equal,
    assert_array_less,
)


# 定义测试函数 test_graphical_lassos，测试图形套索求解器
def test_graphical_lassos(random_state=1):
    """Test the graphical lasso solvers.

    This checks is unstable for some random seeds where the covariance found with "cd"
    and "lars" solvers are different (4 cases / 100 tries).
    """
    # 生成稀疏多元正态分布的样本数据
    dim = 20
    n_samples = 100
    random_state = check_random_state(random_state)
    prec = make_sparse_spd_matrix(dim, alpha=0.95, random_state=random_state)
    cov = linalg.inv(prec)
    X = random_state.multivariate_normal(np.zeros(dim), cov, size=n_samples)
    emp_cov = empirical_covariance(X)

    # 遍历不同的 alpha 值
    for alpha in (0.0, 0.1, 0.25):
        covs = dict()
        icovs = dict()
        # 遍历不同的方法： "cd" 和 "lars"
        for method in ("cd", "lars"):
            # 调用图形套索函数进行计算，并返回相关的协方差和精确协方差
            cov_, icov_, costs = graphical_lasso(
                emp_cov, return_costs=True, alpha=alpha, mode=method
            )
            covs[method] = cov_
            icovs[method] = icov_
            costs, dual_gap = np.array(costs).T
            # 检查成本是否总是减少（如果 alpha == 0，则不一定成立）
            if not alpha == 0:
                # 使用 1e-12，因为成本可能恰好为 0
                assert_array_less(np.diff(costs), 1e-12)
        # 检查两种方法得到的结果是否相似
        assert_allclose(covs["cd"], covs["lars"], atol=5e-4)
        assert_allclose(icovs["cd"], icovs["lars"], atol=5e-4)

    # 对评估器进行烟雾测试
    model = GraphicalLasso(alpha=0.25).fit(X)
    model.score(X)
    assert_array_almost_equal(model.covariance_, covs["cd"], decimal=4)
    assert_array_almost_equal(model.covariance_, covs["lars"], decimal=4)

    # 对于中心化的矩阵，可以选择 assume_centered 为 True 或 False
    # 检查对中心化数据确实返回相同结果
    Z = X - X.mean(0)
    precs = list()
    for assume_centered in (False, True):
        prec_ = GraphicalLasso(assume_centered=assume_centered).fit(Z).precision_
        precs.append(prec_)
    assert_array_almost_equal(precs[0], precs[1])


# 定义测试函数 test_graphical_lasso_when_alpha_equals_0，测试 alpha=0 时的早期返回条件
def test_graphical_lasso_when_alpha_equals_0():
    """Test graphical_lasso's early return condition when alpha=0."""
    # 生成随机数据
    X = np.random.randn(100, 10)
    emp_cov = empirical_covariance(X, assume_centered=True)

    # 使用 alpha=0，协方差矩阵为预先计算的情况下拟合模型
    model = GraphicalLasso(alpha=0, covariance="precomputed").fit(emp_cov)
    # 检查精确协方差是否与其逆矩阵非常接近
    assert_allclose(model.precision_, np.linalg.inv(emp_cov))
    # 调用图形拉索函数计算精度矩阵，但忽略返回的第一个结果（使用下划线命名）。返回的第二个结果存储在变量 precision 中。
    _, precision = graphical_lasso(emp_cov, alpha=0)
    
    # 使用 NumPy 的 assert_allclose 函数检查 precision 是否接近于经验协方差矩阵 emp_cov 的逆矩阵。
    assert_allclose(precision, np.linalg.inv(emp_cov))
@pytest.mark.parametrize("mode", ["cd", "lars"])
def test_graphical_lasso_n_iter(mode):
    # 生成一个具有5000个样本和20个特征的分类数据集，并设置随机种子为0
    X, _ = datasets.make_classification(n_samples=5_000, n_features=20, random_state=0)
    # 计算数据集X的经验协方差矩阵
    emp_cov = empirical_covariance(X)

    # 调用graphical_lasso函数，传入emp_cov作为协方差估计值，alpha为0.2，选择的优化模式为mode，
    # 最大迭代次数为2，并要求返回迭代次数n_iter
    _, _, n_iter = graphical_lasso(
        emp_cov, 0.2, mode=mode, max_iter=2, return_n_iter=True
    )
    # 断言迭代次数n_iter为2
    assert n_iter == 2


def test_graphical_lasso_iris():
    # 从R的glasso包中硬编码的解决方案，针对alpha=1.0（需要将penalize.diagonal设置为FALSE）
    cov_R = np.array(
        [
            [0.68112222, 0.0000000, 0.265820, 0.02464314],
            [0.00000000, 0.1887129, 0.000000, 0.00000000],
            [0.26582000, 0.0000000, 3.095503, 0.28697200],
            [0.02464314, 0.0000000, 0.286972, 0.57713289],
        ]
    )
    icov_R = np.array(
        [
            [1.5190747, 0.000000, -0.1304475, 0.0000000],
            [0.0000000, 5.299055, 0.0000000, 0.0000000],
            [-0.1304475, 0.000000, 0.3498624, -0.1683946],
            [0.0000000, 0.000000, -0.1683946, 1.8164353],
        ]
    )
    # 加载鸢尾花数据集，并获取其数据部分X
    X = datasets.load_iris().data
    # 计算数据集X的经验协方差矩阵
    emp_cov = empirical_covariance(X)
    # 对于两种方法"cd"和"lars"，分别调用graphical_lasso函数，传入emp_cov作为协方差估计值，
    # alpha为1.0，要求不返回代价，并设置优化模式为method
    for method in ("cd", "lars"):
        cov, icov = graphical_lasso(emp_cov, alpha=1.0, return_costs=False, mode=method)
        # 断言计算出的协方差矩阵cov与预期的cov_R非常接近
        assert_array_almost_equal(cov, cov_R)
        # 断言计算出的精确协方差矩阵icov与预期的icov_R非常接近
        assert_array_almost_equal(icov, icov_R)


def test_graph_lasso_2D():
    # 从Python skggm包中硬编码的解决方案，通过调用`quic(emp_cov, lam=.1, tol=1e-8)`获得
    cov_skggm = np.array([[3.09550269, 1.186972], [1.186972, 0.57713289]])

    icov_skggm = np.array([[1.52836773, -3.14334831], [-3.14334831, 8.19753385]])
    # 加载鸢尾花数据集中的第3和第4列特征，构成数据集X
    X = datasets.load_iris().data[:, 2:]
    # 计算数据集X的经验协方差矩阵
    emp_cov = empirical_covariance(X)
    # 对于两种方法"cd"和"lars"，分别调用graphical_lasso函数，传入emp_cov作为协方差估计值，
    # alpha为0.1，要求不返回代价，并设置优化模式为method
    for method in ("cd", "lars"):
        cov, icov = graphical_lasso(emp_cov, alpha=0.1, return_costs=False, mode=method)
        # 断言计算出的协方差矩阵cov与预期的cov_skggm非常接近
        assert_array_almost_equal(cov, cov_skggm)
        # 断言计算出的精确协方差矩阵icov与预期的icov_skggm非常接近
        assert_array_almost_equal(icov, icov_skggm)


def test_graphical_lasso_iris_singular():
    # 选择样本索引以测试秩亏的情况
    # 需要选择样本，确保所有的方差都不为零
    indices = np.arange(10, 13)

    # 从R的glasso包中硬编码的解决方案，针对alpha=0.01
    cov_R = np.array(
        [
            [0.08, 0.056666662595, 0.00229729713223, 0.00153153142149],
            [0.056666662595, 0.082222222222, 0.00333333333333, 0.00222222222222],
            [0.002297297132, 0.003333333333, 0.00666666666667, 0.00009009009009],
            [0.001531531421, 0.002222222222, 0.00009009009009, 0.00222222222222],
        ]
    )
    icov_R = np.array(
        [
            [24.42244057, -16.831679593, 0.0, 0.0],
            [-16.83168201, 24.351841681, -6.206896552, -12.5],
            [0.0, -6.206896171, 153.103448276, 0.0],
            [0.0, -12.499999143, 0.0, 462.5],
        ]
    )
    # 加载鸢尾花数据集中的部分行，构成数据集X
    X = datasets.load_iris().data[indices, :]
    # 计算数据集X的经验协方差矩阵
    emp_cov = empirical_covariance(X)
    # 对于每种方法（"cd" 和 "lars"），分别执行图形化拉索方法，估计协方差和精确逆协方差矩阵
    for method in ("cd", "lars"):
        cov, icov = graphical_lasso(
            emp_cov, alpha=0.01, return_costs=False, mode=method
        )
        # 断言计算得到的协方差矩阵与参考值 cov_R 几乎相等（小数点后保留五位）
        assert_array_almost_equal(cov, cov_R, decimal=5)
        # 断言计算得到的精确逆协方差矩阵与参考值 icov_R 几乎相等（小数点后保留五位）
        assert_array_almost_equal(icov, icov_R, decimal=5)
def test_graphical_lasso_cv(random_state=1):
    # 从稀疏多变量正态分布中抽样数据
    dim = 5
    n_samples = 6
    random_state = check_random_state(random_state)
    # 生成稀疏正定对角矩阵
    prec = make_sparse_spd_matrix(dim, alpha=0.96, random_state=random_state)
    # 计算协方差矩阵
    cov = linalg.inv(prec)
    # 生成多元正态分布数据
    X = random_state.multivariate_normal(np.zeros(dim), cov, size=n_samples)
    # 捕获标准输出，用于测试详细模式
    orig_stdout = sys.stdout
    try:
        sys.stdout = StringIO()
        # 设置详细模式非常高，以便并行过程在标准输出中打印信息
        GraphicalLassoCV(verbose=100, alphas=5, tol=1e-1).fit(X)
    finally:
        sys.stdout = orig_stdout


@pytest.mark.parametrize("alphas_container_type", ["list", "tuple", "array"])
def test_graphical_lasso_cv_alphas_iterable(alphas_container_type):
    """验证可以将类数组传递给 `alphas`。

    针对非回归的测试:
    https://github.com/scikit-learn/scikit-learn/issues/22489
    """
    true_cov = np.array(
        [
            [0.8, 0.0, 0.2, 0.0],
            [0.0, 0.4, 0.0, 0.0],
            [0.2, 0.0, 0.3, 0.1],
            [0.0, 0.0, 0.1, 0.7],
        ]
    )
    rng = np.random.RandomState(0)
    X = rng.multivariate_normal(mean=[0, 0, 0, 0], cov=true_cov, size=200)
    # 将 `alphas` 转换为指定类型的容器
    alphas = _convert_container([0.02, 0.03], alphas_container_type)
    GraphicalLassoCV(alphas=alphas, tol=1e-1, n_jobs=1).fit(X)


@pytest.mark.parametrize(
    "alphas,err_type,err_msg",
    [
        ([-0.02, 0.03], ValueError, "must be > 0"),
        ([0, 0.03], ValueError, "must be > 0"),
        (["not_number", 0.03], TypeError, "must be an instance of float"),
    ],
)
def test_graphical_lasso_cv_alphas_invalid_array(alphas, err_type, err_msg):
    """检查如果向 `alphas` 传递了包含(0, inf]范围外值的类数组，会引发 ValueError。
    如果传递了字符串，会引发 TypeError。
    """
    true_cov = np.array(
        [
            [0.8, 0.0, 0.2, 0.0],
            [0.0, 0.4, 0.0, 0.0],
            [0.2, 0.0, 0.3, 0.1],
            [0.0, 0.0, 0.1, 0.7],
        ]
    )
    rng = np.random.RandomState(0)
    X = rng.multivariate_normal(mean=[0, 0, 0, 0], cov=true_cov, size=200)

    # 使用 pytest 断言检查是否引发了指定类型的异常，并匹配相应的错误消息
    with pytest.raises(err_type, match=err_msg):
        GraphicalLassoCV(alphas=alphas, tol=1e-1, n_jobs=1).fit(X)


def test_graphical_lasso_cv_scores():
    splits = 4
    n_alphas = 5
    n_refinements = 3
    true_cov = np.array(
        [
            [0.8, 0.0, 0.2, 0.0],
            [0.0, 0.4, 0.0, 0.0],
            [0.2, 0.0, 0.3, 0.1],
            [0.0, 0.0, 0.1, 0.7],
        ]
    )
    rng = np.random.RandomState(0)
    X = rng.multivariate_normal(mean=[0, 0, 0, 0], cov=true_cov, size=200)
    # 使用交叉验证和网格搜索参数优化的图形套索方法
    cov = GraphicalLassoCV(cv=splits, alphas=n_alphas, n_refinements=n_refinements).fit(
        X
    )

    # 断言图形套索交叉验证分数的准确性
    _assert_graphical_lasso_cv_scores(
        cov=cov,
        n_splits=splits,
        n_refinements=n_refinements,
        n_alphas=n_alphas,
    )
@pytest.mark.usefixtures("enable_slep006")
def test_graphical_lasso_cv_scores_with_routing(global_random_seed):
    """Check that `GraphicalLassoCV` internally dispatches metadata to
    the splitter.
    """
    # 定义交叉验证的折数和超参数数量
    splits = 5
    n_alphas = 5
    n_refinements = 3
    # 真实协方差矩阵
    true_cov = np.array(
        [
            [0.8, 0.0, 0.2, 0.0],
            [0.0, 0.4, 0.0, 0.0],
            [0.2, 0.0, 0.3, 0.1],
            [0.0, 0.0, 0.1, 0.7],
        ]
    )
    # 使用全局随机种子生成多元正态分布样本
    rng = np.random.RandomState(global_random_seed)
    X = rng.multivariate_normal(mean=[0, 0, 0, 0], cov=true_cov, size=300)
    n_samples = X.shape[0]
    # 生成随机的分组信息
    groups = rng.randint(0, 5, n_samples)
    params = {"groups": groups}
    # 创建 GroupKFold 交叉验证对象并设置分组请求
    cv = GroupKFold(n_splits=splits)
    cv.set_split_request(groups=True)

    # 使用 GraphicalLassoCV 进行拟合
    cov = GraphicalLassoCV(cv=cv, alphas=n_alphas, n_refinements=n_refinements).fit(
        X, **params
    )

    # 调用断言函数，验证交叉验证结果
    _assert_graphical_lasso_cv_scores(
        cov=cov,
        n_splits=splits,
        n_refinements=n_refinements,
        n_alphas=n_alphas,
    )


def _assert_graphical_lasso_cv_scores(cov, n_splits, n_refinements, n_alphas):
    # 获取交叉验证结果
    cv_results = cov.cv_results_
    # 计算总的 alpha 数量
    total_alphas = n_refinements * n_alphas + 1
    keys = ["alphas"]
    split_keys = [f"split{i}_test_score" for i in range(n_splits)]
    # 验证结果中是否包含特定键，并检查其长度是否正确
    for key in keys + split_keys:
        assert key in cv_results
        assert len(cv_results[key]) == total_alphas

    # 将测试分数转换为 NumPy 数组
    cv_scores = np.asarray([cov.cv_results_[key] for key in split_keys])
    # 计算预期的平均值和标准差
    expected_mean = cv_scores.mean(axis=0)
    expected_std = cv_scores.std(axis=0)

    # 使用 assert_allclose 检查均值和标准差是否符合预期
    assert_allclose(cov.cv_results_["mean_test_score"], expected_mean)
    assert_allclose(cov.cv_results_["std_test_score"], expected_std)
```