# `D:\src\scipysrc\scikit-learn\sklearn\tests\test_discriminant_analysis.py`

```
# 导入警告模块，用于管理警告信息
import warnings

# 导入 NumPy 库并使用别名 np
import numpy as np

# 导入 pytest 测试框架
import pytest

# 从 SciPy 库中导入 linalg 模块
from scipy import linalg

# 从 scikit-learn 库中导入 KMeans 聚类算法
from sklearn.cluster import KMeans

# 从 scikit-learn 库中导入协方差估计方法：LedoitWolf、ShrunkCovariance、ledoit_wolf
from sklearn.covariance import LedoitWolf, ShrunkCovariance, ledoit_wolf

# 从 scikit-learn 库中导入数据生成函数 make_blobs
from sklearn.datasets import make_blobs

# 从 scikit-learn 库中导入线性判别分析和二次判别分析方法
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
    _cov,
)

# 从 scikit-learn 库中导入数据标准化模块
from sklearn.preprocessing import StandardScaler

# 从 scikit-learn 库中导入随机数检查工具
from sklearn.utils import check_random_state

# 从 scikit-learn 库中导入用于测试的函数和类
from sklearn.utils._testing import (
    _convert_container,
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)

# 从 scikit-learn 库中导入修复相关工具
from sklearn.utils.fixes import _IS_WASM

# Data is just 6 separable points in the plane
# 定义一个包含 6 个可分离点的数据集 X 和相应的标签 y
X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]], dtype="f")
y = np.array([1, 1, 1, 2, 2, 2])
y3 = np.array([1, 1, 2, 2, 3, 3])

# Degenerate data with only one feature (still should be separable)
# 另一个数据集 X1，只包含一个特征，但应仍然可分离
X1 = np.array(
    [[-2], [-1], [-1], [1], [1], [2]],
    dtype="f",
)

# Data is just 9 separable points in the plane
# 定义一个包含 9 个可分离点的数据集 X6 和相应的标签 y6
X6 = np.array(
    [[0, 0], [-2, -2], [-2, -1], [-1, -1], [-1, -2], [1, 3], [1, 2], [2, 1], [2, 2]]
)
y6 = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2])
y7 = np.array([1, 2, 3, 2, 3, 1, 2, 3, 1])

# Degenerate data with 1 feature (still should be separable)
# 另一个数据集 X7，只包含一个特征，但应仍然可分离
X7 = np.array([[-3], [-2], [-1], [-1], [0], [1], [1], [2], [3]])

# Data that has zero variance in one dimension and needs regularization
# 另一个数据集 X2，其中一个维度具有零方差，需要正则化处理
X2 = np.array(
    [[-3, 0], [-2, 0], [-1, 0], [-1, 0], [0, 0], [1, 0], [1, 0], [2, 0], [3, 0]]
)

# One element class
# 只包含一个类别的标签 y4
y4 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 2])

# Data with less samples in a class than n_features
# 另一个数据集 X5，其中某些类别的样本数小于特征数
X5 = np.c_[np.arange(8), np.zeros((8, 3))]
y5 = np.array([0, 0, 0, 0, 0, 1, 1, 1])

# 定义一个求解器收缩参数列表
solver_shrinkage = [
    ("svd", None),
    ("lsqr", None),
    ("eigen", None),
    ("lsqr", "auto"),
    ("lsqr", 0),
    ("lsqr", 0.43),
    ("eigen", "auto"),
    ("eigen", 0),
    ("eigen", 0.43),
]


def test_lda_predict():
    # 测试 LDA 分类器的预测能力
    # 这个测试检查 LDA 是否实现了拟合和预测，并对简单的测试数据返回正确的结果。
    # 对于每个测试用例，从 solver_shrinkage 中获取 solver 和 shrinkage 参数
    for test_case in solver_shrinkage:
        solver, shrinkage = test_case
        
        # 使用给定的 solver 和 shrinkage 参数创建 LinearDiscriminantAnalysis 分类器对象
        clf = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
        
        # 使用分类器拟合训练数据 X，并对 X 进行预测
        y_pred = clf.fit(X, y).predict(X)
        
        # 断言预测结果与真实标签 y 相等，如果不等则抛出异常，附带 solver 的信息
        assert_array_equal(y_pred, y, "solver %s" % solver)

        # 对于 1D 数据 X1，测试分类器的拟合和预测
        y_pred1 = clf.fit(X1, y).predict(X1)
        
        # 断言预测结果与真实标签 y 相等，如果不等则抛出异常，附带 solver 的信息
        assert_array_equal(y_pred1, y, "solver %s" % solver)

        # 测试分类器对于概率估计的功能
        y_proba_pred1 = clf.predict_proba(X1)
        
        # 断言预测的类别概率大于0.5的结果加1与真实标签 y 相等，如果不等则抛出异常，附带 solver 的信息
        assert_array_equal((y_proba_pred1[:, 1] > 0.5) + 1, y, "solver %s" % solver)
        
        # 获取对数概率预测
        y_log_proba_pred1 = clf.predict_log_proba(X1)
        
        # 断言对数概率的指数与概率预测结果相等，允许的相对和绝对误差分别为 1e-6，如果不满足则抛出异常，附带 solver 的信息
        assert_allclose(
            np.exp(y_log_proba_pred1),
            y_proba_pred1,
            rtol=1e-6,
            atol=1e-6,
            err_msg="solver %s" % solver,
        )

        # 主要测试提交 2f34950 -- 先验的重用
        y_pred3 = clf.fit(X, y3).predict(X)
        
        # 断言分类器不能完全正确地分离 y3，即预测结果与 y3 有不同的部分，如果全相等则抛出异常，附带 solver 的信息
        assert np.any(y_pred3 != y3), "solver %s" % solver

    # 使用 solver="svd", shrinkage="auto" 创建 LDA 分类器对象
    clf = LinearDiscriminantAnalysis(solver="svd", shrinkage="auto")
    
    # 断言使用该配置拟合数据 X 时会抛出 NotImplementedError 异常
    with pytest.raises(NotImplementedError):
        clf.fit(X, y)

    # 使用 solver="lsqr", shrinkage=0.1 和特定的 covariance_estimator 创建 LDA 分类器对象
    clf = LinearDiscriminantAnalysis(
        solver="lsqr", shrinkage=0.1, covariance_estimator=ShrunkCovariance()
    )
    
    # 断言当 covariance_estimator 和 shrinkage 参数同时设置时会抛出 ValueError 异常，附带相应的错误信息
    with pytest.raises(
        ValueError,
        match=(
            "covariance_estimator and shrinkage "
            "parameters are not None. "
            "Only one of the two can be set."
        ),
    ):
        clf.fit(X, y)

    # 使用不支持的 solver="svd" 和 covariance_estimator 创建 LDA 分类器对象
    clf = LinearDiscriminantAnalysis(solver="svd", covariance_estimator=LedoitWolf())
    
    # 断言当尝试使用不支持的 covariance estimator 时会抛出 ValueError 异常，附带相应的错误信息
    with pytest.raises(
        ValueError, match="covariance estimator is not supported with svd"
    ):
        clf.fit(X, y)

    # 使用不支持的 covariance estimator 创建 LDA 分类器对象
    clf = LinearDiscriminantAnalysis(
        solver="lsqr", covariance_estimator=KMeans(n_clusters=2, n_init="auto")
    )
    
    # 断言当使用不支持的 covariance estimator 时会抛出 ValueError 异常
    with pytest.raises(ValueError):
        clf.fit(X, y)
@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("solver", ["svd", "lsqr", "eigen"])
def test_lda_predict_proba(solver, n_classes):
    def generate_dataset(n_samples, centers, covariances, random_state=None):
        """Generate a multivariate normal data given some centers and
        covariances"""
        # 使用给定的中心点和协方差生成多变量正态分布的数据集
        rng = check_random_state(random_state)
        # 堆叠生成多个中心点和协方差对应的数据集
        X = np.vstack(
            [
                rng.multivariate_normal(mean, cov, size=n_samples // len(centers))
                for mean, cov in zip(centers, covariances)
            ]
        )
        # 创建对应的标签数据
        y = np.hstack(
            [[clazz] * (n_samples // len(centers)) for clazz in range(len(centers))]
        )
        return X, y

    # 设置数据集的中心点和协方差
    blob_centers = np.array([[0, 0], [-10, 40], [-30, 30]])[:n_classes]
    blob_stds = np.array([[[10, 10], [10, 100]]] * len(blob_centers))
    # 生成数据集
    X, y = generate_dataset(
        n_samples=90000, centers=blob_centers, covariances=blob_stds, random_state=42
    )
    # 使用线性判别分析拟合数据
    lda = LinearDiscriminantAnalysis(
        solver=solver, store_covariance=True, shrinkage=None
    ).fit(X, y)
    # 检查拟合后的均值和协方差矩阵是否与生成数据时使用的接近
    assert_allclose(lda.means_, blob_centers, atol=1e-1)
    assert_allclose(lda.covariance_, blob_stds[0], atol=1)

    # 实现计算概率的方法，参考《统计学习的要素》第4.4.5节
    precision = linalg.inv(blob_stds[0])
    alpha_k = []
    alpha_k_0 = []
    for clazz in range(len(blob_centers) - 1):
        # 计算 alpha_k
        alpha_k.append(
            np.dot(precision, (blob_centers[clazz] - blob_centers[-1])[:, np.newaxis])
        )
        # 计算 alpha_k_0
        alpha_k_0.append(
            np.dot(
                -0.5 * (blob_centers[clazz] + blob_centers[-1])[np.newaxis, :],
                alpha_k[-1],
            )
        )

    # 设置样本
    sample = np.array([[-22, 22]])

    def discriminant_func(sample, coef, intercept, clazz):
        # 计算判别函数的指数部分
        return np.exp(intercept[clazz] + np.dot(sample, coef[clazz])).item()

    # 计算概率
    prob = np.array(
        [
            float(
                discriminant_func(sample, alpha_k, alpha_k_0, clazz)
                / (
                    1
                    + sum(
                        [
                            discriminant_func(sample, alpha_k, alpha_k_0, clazz)
                            for clazz in range(n_classes - 1)
                        ]
                    )
                )
            )
            for clazz in range(n_classes - 1)
        ]
    )

    # 计算参考概率
    prob_ref = 1 - np.sum(prob)

    # 检查计算出的概率的一致性
    # 所有概率应该加起来等于一
    prob_ref_2 = float(
        1
        / (
            1
            + sum(
                [
                    discriminant_func(sample, alpha_k, alpha_k_0, clazz)
                    for clazz in range(n_classes - 1)
                ]
            )
        )
    )
    # 使用断言检查 prob_ref 是否与 prob_ref_2 近似相等
    assert prob_ref == pytest.approx(prob_ref_2)
    
    # 使用断言检查 LDA 模型预测的样本概率是否接近理论概率
    assert_allclose(
        # 预测 LDA 模型对样本的概率分布
        lda.predict_proba(sample),
        # 将 prob 和 prob_ref 合并后，作为一个新的数组与预测结果比较
        np.hstack([prob, prob_ref])[np.newaxis],
        # 设置绝对容差为 1e-2
        atol=1e-2
    )
def test_lda_priors():
    # 测试先验概率（负先验概率）
    priors = np.array([0.5, -0.5])  # 设置先验概率数组，包含一个负值
    clf = LinearDiscriminantAnalysis(priors=priors)  # 使用设置的先验概率初始化线性判别分析器
    msg = "priors must be non-negative"  # 错误消息

    with pytest.raises(ValueError, match=msg):  # 检查是否抛出值错误，并匹配错误消息
        clf.fit(X, y)

    # 测试以列表形式传递的先验概率是否被正确处理（运行以查看是否失败）
    clf = LinearDiscriminantAnalysis(priors=[0.5, 0.5])  # 使用列表形式的先验概率初始化线性判别分析器
    clf.fit(X, y)

    # 测试先验概率总和始终为1
    priors = np.array([0.5, 0.6])  # 设置先验概率数组
    prior_norm = np.array([0.45, 0.55])  # 正则化后的先验概率数组
    clf = LinearDiscriminantAnalysis(priors=priors)  # 使用设置的先验概率初始化线性判别分析器

    with pytest.warns(UserWarning):  # 检查是否发出用户警告
        clf.fit(X, y)

    assert_array_almost_equal(clf.priors_, prior_norm, 2)  # 断言先验概率数组是否与正则化后的数组几乎相等


def test_lda_coefs():
    # 测试求解器的系数是否大致相同
    n_features = 2  # 特征数
    n_classes = 2  # 类别数
    n_samples = 1000  # 样本数
    X, y = make_blobs(
        n_samples=n_samples, n_features=n_features, centers=n_classes, random_state=11
    )  # 生成用于测试的数据集

    clf_lda_svd = LinearDiscriminantAnalysis(solver="svd")  # 使用SVD求解器初始化线性判别分析器
    clf_lda_lsqr = LinearDiscriminantAnalysis(solver="lsqr")  # 使用LSQR求解器初始化线性判别分析器
    clf_lda_eigen = LinearDiscriminantAnalysis(solver="eigen")  # 使用EIGEN求解器初始化线性判别分析器

    clf_lda_svd.fit(X, y)  # 使用SVD求解器拟合数据
    clf_lda_lsqr.fit(X, y)  # 使用LSQR求解器拟合数据
    clf_lda_eigen.fit(X, y)  # 使用EIGEN求解器拟合数据

    assert_array_almost_equal(clf_lda_svd.coef_, clf_lda_lsqr.coef_, 1)  # 断言SVD和LSQR求解器得到的系数是否几乎相等
    assert_array_almost_equal(clf_lda_svd.coef_, clf_lda_eigen.coef_, 1)  # 断言SVD和EIGEN求解器得到的系数是否几乎相等
    assert_array_almost_equal(clf_lda_eigen.coef_, clf_lda_lsqr.coef_, 1)  # 断言EIGEN和LSQR求解器得到的系数是否几乎相等


def test_lda_transform():
    # 测试LDA变换
    clf = LinearDiscriminantAnalysis(solver="svd", n_components=1)  # 使用SVD求解器和指定的成分数初始化线性判别分析器
    X_transformed = clf.fit(X, y).transform(X)  # 拟合数据并进行变换
    assert X_transformed.shape[1] == 1  # 断言变换后的数据维度是否为1
    clf = LinearDiscriminantAnalysis(solver="eigen", n_components=1)  # 使用EIGEN求解器和指定的成分数初始化线性判别分析器
    X_transformed = clf.fit(X, y).transform(X)  # 拟合数据并进行变换
    assert X_transformed.shape[1] == 1  # 断言变换后的数据维度是否为1

    clf = LinearDiscriminantAnalysis(solver="lsqr", n_components=1)  # 使用LSQR求解器和指定的成分数初始化线性判别分析器
    clf.fit(X, y)  # 拟合数据

    msg = "transform not implemented for 'lsqr'"  # 错误消息

    with pytest.raises(NotImplementedError, match=msg):  # 检查是否抛出未实现错误，并匹配错误消息
        clf.transform(X)


def test_lda_explained_variance_ratio():
    # 测试归一化特征向量值之和是否等于1
    # 同时测试由EIGEN求解器形成的解释方差比率是否与由SVD求解器形成的解释方差比率相同

    state = np.random.RandomState(0)  # 设置随机种子
    X = state.normal(loc=0, scale=100, size=(40, 20))  # 生成服从正态分布的数据
    y = state.randint(0, 3, size=(40,))  # 生成随机整数类别标签

    clf_lda_eigen = LinearDiscriminantAnalysis(solver="eigen")  # 使用EIGEN求解器初始化线性判别分析器
    clf_lda_eigen.fit(X, y)  # 拟合数据

    assert_almost_equal(clf_lda_eigen.explained_variance_ratio_.sum(), 1.0, 3)  # 断言解释方差比率之和是否接近1
    assert clf_lda_eigen.explained_variance_ratio_.shape == (2,)  # 断言解释方差比率数组的长度是否为2

    clf_lda_svd = LinearDiscriminantAnalysis(solver="svd")  # 使用SVD求解器初始化线性判别分析器
    clf_lda_svd.fit(X, y)  # 拟合数据

    assert_almost_equal(clf_lda_svd.explained_variance_ratio_.sum(), 1.0, 3)  # 断言解释方差比率之和是否接近1
    assert clf_lda_svd.explained_variance_ratio_.shape == (2,)  # 断言解释方差比率数组的长度是否为2
    ), "Unexpected length for explained_variance_ratio_"

# 检查两个变量的长度是否相等，如果不相等则触发断言异常，并输出指定的错误消息。


    assert_array_almost_equal(
        clf_lda_svd.explained_variance_ratio_, clf_lda_eigen.explained_variance_ratio_
    )

# 使用 `assert_array_almost_equal` 函数比较 `clf_lda_svd` 和 `clf_lda_eigen` 的 `explained_variance_ratio_` 属性，确保它们几乎相等。
def test_lda_orthogonality():
    # 安排四个类别的均值，呈风筝形状分布
    # 较长的距离应转换为第一个主成分，
    # 较短的距离应转换为第二个主成分。
    means = np.array([[0, 0, -1], [0, 2, 0], [0, -2, 0], [0, 0, 5]])

    # 构造完全对称的分布，以便 LDA 可以精确估计均值。
    scatter = np.array(
        [
            [0.1, 0, 0],
            [-0.1, 0, 0],
            [0, 0.1, 0],
            [0, -0.1, 0],
            [0, 0, 0.1],
            [0, 0, -0.1],
        ]
    )

    # 将均值和散点加起来，形成输入矩阵 X
    X = (means[:, np.newaxis, :] + scatter[np.newaxis, :, :]).reshape((-1, 3))
    # 生成对应的类别标签 y
    y = np.repeat(np.arange(means.shape[0]), scatter.shape[0])

    # 拟合 LDA 并且转换均值
    clf = LinearDiscriminantAnalysis(solver="svd").fit(X, y)
    means_transformed = clf.transform(means)

    d1 = means_transformed[3] - means_transformed[0]
    d2 = means_transformed[2] - means_transformed[1]
    d1 /= np.sqrt(np.sum(d1**2))
    d2 /= np.sqrt(np.sum(d2**2))

    # 转换后的类内协方差应为单位矩阵
    assert_almost_equal(np.cov(clf.transform(scatter).T), np.eye(2))

    # 类别 0 和 3 的均值应该位于第一个主成分上
    assert_almost_equal(np.abs(np.dot(d1[:2], [1, 0])), 1.0)

    # 类别 1 和 2 的均值应该位于第二个主成分上
    assert_almost_equal(np.abs(np.dot(d2[:2], [0, 1])), 1.0)


def test_lda_scaling():
    # 测试不同缩放特征是否能够正确分类
    n = 100
    rng = np.random.RandomState(1234)
    # 使用均匀分布生成特征，以确保类别之间没有重叠
    x1 = rng.uniform(-1, 1, (n, 3)) + [-10, 0, 0]
    x2 = rng.uniform(-1, 1, (n, 3)) + [10, 0, 0]
    x = np.vstack((x1, x2)) * [1, 100, 10000]
    y = [-1] * n + [1] * n

    for solver in ("svd", "lsqr", "eigen"):
        clf = LinearDiscriminantAnalysis(solver=solver)
        # 应该能够完美地分离数据
        assert clf.fit(x, y).score(x, y) == 1.0, "using covariance: %s" % solver


def test_lda_store_covariance():
    # 测试 'lsqr' 和 'eigen' 解算器的情况
    # 'store_covariance' 对 'lsqr' 和 'eigen' 解算器没有影响
    for solver in ("lsqr", "eigen"):
        clf = LinearDiscriminantAnalysis(solver=solver).fit(X6, y6)
        assert hasattr(clf, "covariance_")

        # 测试实际属性：
        clf = LinearDiscriminantAnalysis(solver=solver, store_covariance=True).fit(
            X6, y6
        )
        assert hasattr(clf, "covariance_")

        assert_array_almost_equal(
            clf.covariance_, np.array([[0.422222, 0.088889], [0.088889, 0.533333]])
        )

    # 对于 SVD 解算器，默认情况下不设置 'covariance_' 属性
    clf = LinearDiscriminantAnalysis(solver="svd").fit(X6, y6)
    assert not hasattr(clf, "covariance_")

    # 测试实际属性：
    # 使用线性判别分析模型进行训练，并设置存储协方差矩阵为真
    clf = LinearDiscriminantAnalysis(solver=solver, store_covariance=True).fit(X6, y6)
    # 断言确保分类器具有属性 "covariance_"
    assert hasattr(clf, "covariance_")
    
    # 断言确保分类器的协方差矩阵与给定的数值数组几乎相等
    assert_array_almost_equal(
        clf.covariance_, np.array([[0.422222, 0.088889], [0.088889, 0.533333]])
    )
@pytest.mark.parametrize("seed", range(10))
# 使用pytest的parametrize装饰器，为test_lda_shrinkage函数提供了10个不同的seed参数进行参数化测试
def test_lda_shrinkage(seed):
    # 测试收缩协方差估计器和收缩参数的行为是否一致
    rng = np.random.RandomState(seed)
    # 创建一个特定种子的随机数生成器实例
    X = rng.rand(100, 10)
    # 生成一个100行10列的随机数矩阵
    y = rng.randint(3, size=(100))
    # 生成一个包含100个元素的随机整数数组，取值范围为0到2
    c1 = LinearDiscriminantAnalysis(store_covariance=True, shrinkage=0.5, solver="lsqr")
    # 创建线性判别分析对象c1，使用收缩参数0.5，solver为"lsqr"
    c2 = LinearDiscriminantAnalysis(
        store_covariance=True,
        covariance_estimator=ShrunkCovariance(shrinkage=0.5),
        solver="lsqr",
    )
    # 创建线性判别分析对象c2，使用自定义的ShrunkCovariance类作为协方差估计器，收缩参数为0.5，solver为"lsqr"
    c1.fit(X, y)
    # 使用数据X和标签y对模型c1进行拟合
    c2.fit(X, y)
    # 使用数据X和标签y对模型c2进行拟合
    assert_allclose(c1.means_, c2.means_)
    # 检查两个模型的均值是否近似相等
    assert_allclose(c1.covariance_, c2.covariance_)
    # 检查两个模型的协方差是否近似相等


def test_lda_ledoitwolf():
    # 当收缩参数为"auto"时，当前实现使用ledoitwolf方法估计协方差矩阵，在数据标准化后进行。这里检查是否确实如此。
    class StandardizedLedoitWolf:
        def fit(self, X):
            sc = StandardScaler()  # standardize features
            # 创建标准化器对象sc，对特征进行标准化处理
            X_sc = sc.fit_transform(X)
            # 对输入数据X进行标准化转换
            s = ledoit_wolf(X_sc)[0]
            # 使用ledoit_wolf方法估计标准化后数据的协方差矩阵
            s = sc.scale_[:, np.newaxis] * s * sc.scale_[np.newaxis, :]
            # 重新缩放协方差矩阵
            self.covariance_ = s

    rng = np.random.RandomState(0)
    # 创建一个特定种子的随机数生成器实例
    X = rng.rand(100, 10)
    # 生成一个100行10列的随机数矩阵
    y = rng.randint(3, size=(100,))
    # 生成一个包含100个元素的随机整数数组，取值范围为0到2
    c1 = LinearDiscriminantAnalysis(
        store_covariance=True, shrinkage="auto", solver="lsqr"
    )
    # 创建线性判别分析对象c1，使用收缩参数为"auto"，solver为"lsqr"
    c2 = LinearDiscriminantAnalysis(
        store_covariance=True,
        covariance_estimator=StandardizedLedoitWolf(),
        solver="lsqr",
    )
    # 创建线性判别分析对象c2，使用自定义的StandardizedLedoitWolf类作为协方差估计器，solver为"lsqr"
    c1.fit(X, y)
    # 使用数据X和标签y对模型c1进行拟合
    c2.fit(X, y)
    # 使用数据X和标签y对模型c2进行拟合
    assert_allclose(c1.means_, c2.means_)
    # 检查两个模型的均值是否近似相等
    assert_allclose(c1.covariance_, c2.covariance_)
    # 检查两个模型的协方差是否近似相等


@pytest.mark.parametrize("n_features", [3, 5])
@pytest.mark.parametrize("n_classes", [5, 3])
# 使用pytest的parametrize装饰器，为test_lda_dimension_warning函数提供不同的n_features和n_classes参数进行参数化测试
def test_lda_dimension_warning(n_classes, n_features):
    rng = check_random_state(0)
    # 创建一个特定种子的随机数生成器实例
    n_samples = 10
    # 设置样本数量为10
    X = rng.randn(n_samples, n_features)
    # 生成一个服从标准正态分布的n_samples行、n_features列的随机数矩阵
    y = np.tile(range(n_classes), n_samples // n_classes + 1)[:n_samples]
    # 生成n_samples个标签，由range(n_classes)重复直到满足n_samples的数量要求，然后截断多余部分

    max_components = min(n_features, n_classes - 1)
    # 计算最大允许的成分数量，为n_features和n_classes-1的最小值

    for n_components in [max_components - 1, None, max_components]:
        # 遍历不同的成分数量进行测试

        lda = LinearDiscriminantAnalysis(n_components=n_components)
        # 创建线性判别分析对象lda，设置成分数量为n_components
        lda.fit(X, y)
        # 使用数据X和标签y对模型lda进行拟合

    for n_components in [max_components + 1, max(n_features, n_classes - 1) + 1]:
        # 遍历不同的成分数量进行测试

        lda = LinearDiscriminantAnalysis(n_components=n_components)
        # 创建线性判别分析对象lda，设置成分数量为n_components
        msg = "n_components cannot be larger than "
        # 设置错误消息内容
        with pytest.raises(ValueError, match=msg):
            # 使用pytest的raises断言检查是否抛出特定的ValueError异常，并匹配指定的错误消息
            lda.fit(X, y)
            # 使用数据X和标签y对模型lda进行拟合
    # 定义一个包含四个元组的列表，每个元组包含两个元素：第一个是 np.float32 类型，第二个是 np.float32 类型
    [
        (np.float32, np.float32),
        # 下一个元组，第一个元素是 np.float64 类型，第二个元素也是 np.float64 类型
        (np.float64, np.float64),
        # 再下一个元组，第一个元素是 np.int32 类型，第二个元素是 np.float64 类型
        (np.int32, np.float64),
        # 最后一个元组，第一个元素是 np.int64 类型，第二个元素是 np.float64 类型
        (np.int64, np.float64),
    ],
# 定义测试函数，用于验证线性判别分析器的数据类型匹配
def test_lda_dtype_match(data_type, expected_type):
    # 遍历解决器和收缩参数组合
    for solver, shrinkage in solver_shrinkage:
        # 创建线性判别分析器对象
        clf = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
        # 使用指定数据类型拟合数据
        clf.fit(X.astype(data_type), y.astype(data_type))
        # 断言线性判别分析器的系数数据类型符合预期类型
        assert clf.coef_.dtype == expected_type


# 定义测试函数，用于验证不同浮点数精度（float32和float64）下的线性判别分析器结果一致性
def test_lda_numeric_consistency_float32_float64():
    # 遍历解决器和收缩参数组合
    for solver, shrinkage in solver_shrinkage:
        # 创建float32精度的线性判别分析器对象并拟合数据
        clf_32 = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
        clf_32.fit(X.astype(np.float32), y.astype(np.float32))
        # 创建float64精度的线性判别分析器对象并拟合数据
        clf_64 = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
        clf_64.fit(X.astype(np.float64), y.astype(np.float64))

        # 检查两种精度下的系数值的一致性
        rtol = 1e-6
        assert_allclose(clf_32.coef_, clf_64.coef_, rtol=rtol)


# 定义测试函数，验证二次判别分析器在不同条件下的行为
def test_qda():
    # 创建二次判别分析器对象进行分类
    clf = QuadraticDiscriminantAnalysis()
    y_pred = clf.fit(X6, y6).predict(X6)
    # 断言预测结果与实际标签一致
    assert_array_equal(y_pred, y6)

    # 使用1维数据进行拟合和预测，验证其正常工作
    y_pred1 = clf.fit(X7, y6).predict(X7)
    assert_array_equal(y_pred1, y6)

    # 测试概率估计功能
    y_proba_pred1 = clf.predict_proba(X7)
    assert_array_equal((y_proba_pred1[:, 1] > 0.5) + 1, y6)
    y_log_proba_pred1 = clf.predict_log_proba(X7)
    assert_array_almost_equal(np.exp(y_log_proba_pred1), y_proba_pred1, 8)

    y_pred3 = clf.fit(X6, y7).predict(X6)
    # 二次判别分析器不应能够完全分开这些类
    assert np.any(y_pred3 != y7)

    # 类别应至少包含2个元素
    with pytest.raises(ValueError):
        clf.fit(X6, y4)


# 定义测试函数，验证带先验概率的二次判别分析器行为
def test_qda_priors():
    # 创建二次判别分析器对象进行分类
    clf = QuadraticDiscriminantAnalysis()
    y_pred = clf.fit(X6, y6).predict(X6)
    n_pos = np.sum(y_pred == 2)

    neg = 1e-10
    # 使用特定的先验概率创建二次判别分析器对象进行分类
    clf = QuadraticDiscriminantAnalysis(priors=np.array([neg, 1 - neg]))
    y_pred = clf.fit(X6, y6).predict(X6)
    n_pos2 = np.sum(y_pred == 2)

    # 断言在改变先验概率后，正类的数量有所增加
    assert n_pos2 > n_pos


# 使用参数化测试验证先验概率输入类型的多样性
@pytest.mark.parametrize("priors_type", ["list", "tuple", "array"])
def test_qda_prior_type(priors_type):
    """验证先验概率接受类数组类型。"""
    priors = [0.5, 0.5]
    # 将先验概率转换为指定的容器类型并拟合数据
    clf = QuadraticDiscriminantAnalysis(
        priors=_convert_container([0.5, 0.5], priors_type)
    ).fit(X6, y6)
    # 断言先验概率被正确转换为numpy数组
    assert isinstance(clf.priors_, np.ndarray)
    assert_array_equal(clf.priors_, priors)


# 定义测试函数，验证改变未拟合先验概率不会改变已拟合模型的行为
def test_qda_prior_copy():
    """验证在未拟合数据的情况下改变先验概率不会影响已拟合模型的先验概率。"""
    priors = np.array([0.5, 0.5])
    qda = QuadraticDiscriminantAnalysis(priors=priors).fit(X, y)

    # 预期以下行为
    assert_array_equal(qda.priors_, qda.priors)

    # 改变未拟合数据的先验概率不应该改变已拟合模型的先验概率
    priors[0] = 0.2
    assert qda.priors_[0] != qda.priors[0]


# 定义测试函数，验证是否存储协方差矩阵的选项
def test_qda_store_covariance():
    # 默认情况下不设置covariances_属性
    clf = QuadraticDiscriminantAnalysis().fit(X6, y6)
    # 确保分类器 clf 没有属性 "covariance_"
    assert not hasattr(clf, "covariance_")
    
    # 测试设置了 store_covariance=True 的 QuadraticDiscriminantAnalysis 分类器
    clf = QuadraticDiscriminantAnalysis(store_covariance=True).fit(X6, y6)
    # 确保分类器 clf 现在有属性 "covariance_"
    assert hasattr(clf, "covariance_")
    
    # 检查第一个类别的协方差矩阵是否接近预期值
    assert_array_almost_equal(clf.covariance_[0], np.array([[0.7, 0.45], [0.45, 0.7]]))
    
    # 检查第二个类别的协方差矩阵是否接近预期值
    assert_array_almost_equal(
        clf.covariance_[1],
        np.array([[0.33333333, -0.33333333], [-0.33333333, 0.66666667]]),
    )
@pytest.mark.xfail(
    _IS_WASM,
    reason=(
        "no floating point exceptions, see"
        " https://github.com/numpy/numpy/pull/21895#issuecomment-1311525881"
    ),
)
# 标记为预期失败的测试用例，当在 WASM 环境下时会失败，原因是不支持浮点异常
def test_qda_regularization():
    # 默认情况下 reg_param=0. 会在存在常数变量时导致问题。

    # 在没有正则化的情况下拟合具有常数变量的数据会触发 LinAlgError。
    msg = r"The covariance matrix of class .+ is not full rank"
    clf = QuadraticDiscriminantAnalysis()
    with pytest.warns(linalg.LinAlgWarning, match=msg):
        y_pred = clf.fit(X2, y6)

    y_pred = clf.predict(X2)
    assert np.any(y_pred != y6)

    # 添加一点正则化可以修复拟合时的错误。
    clf = QuadraticDiscriminantAnalysis(reg_param=0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
    clf.fit(X2, y6)
    y_pred = clf.predict(X2)
    assert_array_equal(y_pred, y6)

    # 对于 n_samples_in_a_class < n_features 的情况，也应出现 LinAlgWarning。
    clf = QuadraticDiscriminantAnalysis()
    with pytest.warns(linalg.LinAlgWarning, match=msg):
        clf.fit(X5, y5)

    # 即使进行了正则化，错误仍然会持续存在。
    clf = QuadraticDiscriminantAnalysis(reg_param=0.3)
    with pytest.warns(linalg.LinAlgWarning, match=msg):
        clf.fit(X5, y5)


def test_covariance():
    x, y = make_blobs(n_samples=100, n_features=5, centers=1, random_state=42)

    # 使特征之间产生相关性
    x = np.dot(x, np.arange(x.shape[1] ** 2).reshape(x.shape[1], x.shape[1]))

    # 计算经验协方差
    c_e = _cov(x, "empirical")
    assert_almost_equal(c_e, c_e.T)

    # 计算自动协方差
    c_s = _cov(x, "auto")
    assert_almost_equal(c_s, c_s.T)


@pytest.mark.parametrize("solver", ["svd", "lsqr", "eigen"])
# 测试当样本数等于类别数时会引发 ValueError 的情况
def test_raises_value_error_on_same_number_of_classes_and_samples(solver):
    """
    Tests that if the number of samples equals the number
    of classes, a ValueError is raised.
    """
    X = np.array([[0.5, 0.6], [0.6, 0.5]])
    y = np.array(["a", "b"])
    clf = LinearDiscriminantAnalysis(solver=solver)
    with pytest.raises(ValueError, match="The number of samples must be more"):
        clf.fit(X, y)


def test_get_feature_names_out():
    """Check get_feature_names_out uses class name as prefix."""

    # 拟合线性判别分析模型
    est = LinearDiscriminantAnalysis().fit(X, y)
    # 获取输出特征名
    names_out = est.get_feature_names_out()

    class_name_lower = "LinearDiscriminantAnalysis".lower()
    # 生成预期的输出特征名数组
    expected_names_out = np.array(
        [
            f"{class_name_lower}{i}"
            for i in range(est.explained_variance_ratio_.shape[0])
        ],
        dtype=object,
    )
    assert_array_equal(names_out, expected_names_out)
```