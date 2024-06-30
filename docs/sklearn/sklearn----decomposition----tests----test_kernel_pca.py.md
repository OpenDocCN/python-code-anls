# `D:\src\scipysrc\scikit-learn\sklearn\decomposition\tests\test_kernel_pca.py`

```
import warnings  # 导入警告模块

import numpy as np  # 导入NumPy库
import pytest  # 导入pytest测试框架

import sklearn  # 导入scikit-learn库
from sklearn.datasets import load_iris, make_blobs, make_circles  # 从scikit-learn导入数据集生成和加载函数
from sklearn.decomposition import PCA, KernelPCA  # 导入PCA和KernelPCA降维模块
from sklearn.exceptions import NotFittedError  # 导入未拟合错误异常类
from sklearn.linear_model import Perceptron  # 导入感知器模型
from sklearn.metrics.pairwise import rbf_kernel  # 导入径向基函数核计算函数
from sklearn.model_selection import GridSearchCV  # 导入网格搜索交叉验证模块
from sklearn.pipeline import Pipeline  # 导入Pipeline构建模块
from sklearn.preprocessing import StandardScaler  # 导入数据标准化模块
from sklearn.utils._testing import (  # 导入用于测试的辅助函数
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
)
from sklearn.utils.fixes import CSR_CONTAINERS  # 导入用于修复的CSR容器
from sklearn.utils.validation import _check_psd_eigenvalues  # 导入检查半正定特征值函数


def test_kernel_pca():
    """Nominal test for all solvers and all known kernels + a custom one

    It tests
     - that fit_transform is equivalent to fit+transform
     - that the shapes of transforms and inverse transforms are correct
    """
    rng = np.random.RandomState(0)  # 创建随机数种子

    X_fit = rng.random_sample((5, 4))  # 生成适合拟合的随机数据集
    X_pred = rng.random_sample((2, 4))  # 生成预测的随机数据集

    def histogram(x, y, **kwargs):
        # Histogram kernel implemented as a callable.
        assert kwargs == {}  # 断言核参数为空
        return np.minimum(x, y).sum()  # 返回直方图核的计算结果

    for eigen_solver in ("auto", "dense", "arpack", "randomized"):
        for kernel in ("linear", "rbf", "poly", histogram):
            # histogram kernel produces singular matrix inside linalg.solve
            # XXX use a least-squares approximation?
            inv = not callable(kernel)  # 如果核不是可调用的，则设置为True

            # transform fit data
            kpca = KernelPCA(
                4, kernel=kernel, eigen_solver=eigen_solver, fit_inverse_transform=inv
            )  # 创建KernelPCA对象
            X_fit_transformed = kpca.fit_transform(X_fit)  # 拟合并转换拟合数据
            X_fit_transformed2 = kpca.fit(X_fit).transform(X_fit)  # 分开拟合和转换数据
            assert_array_almost_equal(
                np.abs(X_fit_transformed), np.abs(X_fit_transformed2)
            )  # 断言两种转换方法的结果几乎相等

            # non-regression test: previously, gamma would be 0 by default,
            # forcing all eigenvalues to 0 under the poly kernel
            assert X_fit_transformed.size != 0  # 断言转换后的数据不为空

            # transform new data
            X_pred_transformed = kpca.transform(X_pred)  # 转换新数据
            assert X_pred_transformed.shape[1] == X_fit_transformed.shape[1]  # 断言转换后的数据形状正确

            # inverse transform
            if inv:
                X_pred2 = kpca.inverse_transform(X_pred_transformed)  # 对转换后的数据进行逆转换
                assert X_pred2.shape == X_pred.shape  # 断言逆转换后数据形状与原始预测数据相同


def test_kernel_pca_invalid_parameters():
    """Check that kPCA raises an error if the parameters are invalid

    Tests fitting inverse transform with a precomputed kernel raises a
    ValueError.
    """
    estimator = KernelPCA(
        n_components=10, fit_inverse_transform=True, kernel="precomputed"
    )  # 创建使用预计算核的KernelPCA对象
    err_ms = "Cannot fit_inverse_transform with a precomputed kernel"
    with pytest.raises(ValueError, match=err_ms):
        estimator.fit(np.random.randn(10, 10))  # 断言预测引发值错误异常
# 测试 Kernel PCA 在原始训练数组变异时的稳健性

def test_kernel_pca_consistent_transform():
    """Check robustness to mutations in the original training array

    Test that after fitting a kPCA model, it stays independent of any
    mutation of the values of the original data object by relying on an
    internal copy.
    """
    # 使用种子值 0 初始化随机状态对象
    state = np.random.RandomState(0)
    # 创建一个 10x10 的随机数组 X
    X = state.rand(10, 10)
    # 使用 KernelPCA 拟合数组 X，并生成变换后的数据 transformed1
    kpca = KernelPCA(random_state=state).fit(X)
    transformed1 = kpca.transform(X)

    # 复制数组 X 为 X_copy
    X_copy = X.copy()
    # 将 X 的第一列所有元素替换为 666
    X[:, 0] = 666
    # 使用复制的 X_copy 进行变换，生成 transformed2
    transformed2 = kpca.transform(X_copy)
    # 检查 transformed1 和 transformed2 是否近似相等
    assert_array_almost_equal(transformed1, transformed2)


# 测试 Kernel PCA 是否产生确定性输出

def test_kernel_pca_deterministic_output():
    """Test that Kernel PCA produces deterministic output

    Tests that the same inputs and random state produce the same output.
    """
    # 使用种子值 0 初始化随机状态对象 rng
    rng = np.random.RandomState(0)
    # 创建一个 10x10 的随机数组 X
    X = rng.rand(10, 10)
    # 使用 arpack 和 dense 两种方法进行特征值求解
    eigen_solver = ("arpack", "dense")

    for solver in eigen_solver:
        # 初始化 transformed_X 为 20x2 的零数组
        transformed_X = np.zeros((20, 2))
        for i in range(20):
            # 使用不同的 solver 和相同的种子值 rng 进行 KernelPCA 变换
            kpca = KernelPCA(n_components=2, eigen_solver=solver, random_state=rng)
            transformed_X[i, :] = kpca.fit_transform(X)[0]
        # 检查 transformed_X 是否近似相等
        assert_allclose(transformed_X, np.tile(transformed_X[0, :], 20).reshape(20, 2))


# 使用稀疏数据输入测试 Kernel PCA 的工作情况

@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_kernel_pca_sparse(csr_container):
    """Test that kPCA works on a sparse data input.

    Same test as ``test_kernel_pca except inverse_transform`` since it's not
    implemented for sparse matrices.
    """
    # 使用种子值 0 初始化随机状态对象 rng
    rng = np.random.RandomState(0)
    # 创建稀疏矩阵 X_fit 和 X_pred
    X_fit = csr_container(rng.random_sample((5, 4)))
    X_pred = csr_container(rng.random_sample((2, 4)))

    for eigen_solver in ("auto", "arpack", "randomized"):
        for kernel in ("linear", "rbf", "poly"):
            # 对 fit 数据进行变换
            kpca = KernelPCA(
                4,
                kernel=kernel,
                eigen_solver=eigen_solver,
                fit_inverse_transform=False,
                random_state=0,
            )
            # 使用不同的方法进行 fit_transform，并比较结果
            X_fit_transformed = kpca.fit_transform(X_fit)
            X_fit_transformed2 = kpca.fit(X_fit).transform(X_fit)
            assert_array_almost_equal(
                np.abs(X_fit_transformed), np.abs(X_fit_transformed2)
            )

            # 对新数据进行变换
            X_pred_transformed = kpca.transform(X_pred)
            assert X_pred_transformed.shape[1] == X_fit_transformed.shape[1]

            # 逆变换：稀疏矩阵不支持逆变换
            # XXX: 是否应该在此处引发另一种异常？例如：NotImplementedError。
            with pytest.raises(NotFittedError):
                kpca.inverse_transform(X_pred_transformed)


# 测试使用线性核的 Kernel PCA 是否对所有求解器都等价于 PCA

@pytest.mark.parametrize("solver", ["auto", "dense", "arpack", "randomized"])
@pytest.mark.parametrize("n_features", [4, 10])
def test_kernel_pca_linear_kernel(solver, n_features):
    """Test that kPCA with linear kernel is equivalent to PCA for all solvers.

    This test checks if using a linear kernel in Kernel PCA is equivalent
    to using PCA across different solvers.
    """
    # 使用线性核的 KernelPCA 应该产生与 PCA 相同的输出。
    """
    # 创建一个随机数生成器对象，种子为0，确保结果可复现
    rng = np.random.RandomState(0)
    # 生成一个形状为 (5, n_features) 的随机样本作为拟合数据
    X_fit = rng.random_sample((5, n_features))
    # 生成一个形状为 (2, n_features) 的随机样本作为预测数据
    X_pred = rng.random_sample((2, n_features))

    # 对于线性核，KernelPCA 应当找到与 PCA 相同的投影，除了方向可能相反
    # 仅拟合前四个主成分：第五个主成分接近零特征值，因此可能由于舍入误差被修剪
    n_comps = 3 if solver == "arpack" else 4
    # 断言 KernelPCA 和 PCA 的变换输出几乎相等
    assert_array_almost_equal(
        np.abs(KernelPCA(n_comps, eigen_solver=solver).fit(X_fit).transform(X_pred)),
        np.abs(
            PCA(n_comps, svd_solver=solver if solver != "dense" else "full")
            .fit(X_fit)
            .transform(X_pred)
        ),
    )
# 测试 `n_components` 参数在投影中的正确应用

# 使用随机数生成器创建一个5行4列的随机数矩阵作为训练数据
rng = np.random.RandomState(0)
X_fit = rng.random_sample((5, 4))
# 使用随机数生成器创建一个2行4列的随机数矩阵作为预测数据
X_pred = rng.random_sample((2, 4))

# 遍历三种特征值求解器和多种组件数值
for eigen_solver in ("dense", "arpack", "randomized"):
    for c in [1, 2, 4]:
        # 创建一个 KernelPCA 对象，设置 n_components 和 eigen_solver 参数
        kpca = KernelPCA(n_components=c, eigen_solver=eigen_solver)
        # 计算并返回使用训练数据拟合后的数据的形状，并对预测数据进行转换
        shape = kpca.fit(X_fit).transform(X_pred).shape

        # 断言转换后的形状与预期相符
        assert shape == (2, c)


# 检查 `remove_zero_eig` 参数是否正确工作

# 创建一个包含浮点数的数组 X
X = np.array([[1 - 1e-30, 1], [1, 1], [1, 1 - 1e-20]])

# 使用默认参数创建 KernelPCA 对象，即 n_components=None，默认情况下 remove_zero_eig=True
kpca = KernelPCA()
# 对 X 进行拟合和转换，并断言转换后的形状
Xt = kpca.fit_transform(X)
assert Xt.shape == (3, 0)

# 创建一个指定 n_components=2 的 KernelPCA 对象
kpca = KernelPCA(n_components=2)
# 对 X 进行拟合和转换，并断言转换后的形状
Xt = kpca.fit_transform(X)
assert Xt.shape == (3, 2)

# 创建一个指定 n_components=2 和 remove_zero_eig=True 的 KernelPCA 对象
kpca = KernelPCA(n_components=2, remove_zero_eig=True)
# 对 X 进行拟合和转换，并断言转换后的形状
Xt = kpca.fit_transform(X)
assert Xt.shape == (3, 0)


# 针对问题 #12141 (PR #12143) 的非回归测试

# 创建一个2行2列的数组作为训练数据
X_fit = np.array([[1, 1], [0, 0]])

# 确保即使在所有 NumPy 警告开启的情况下，没有除以零的警告
with warnings.catch_warnings():
    # 可能会出现有关内核条件不良的警告，但不应出现除以零的警告
    warnings.simplefilter("error", RuntimeWarning)
    with np.errstate(all="warn"):
        # 创建一个 KernelPCA 对象，设置 n_components、remove_zero_eig 和 eigen_solver 参数
        k = KernelPCA(n_components=2, remove_zero_eig=False, eigen_solver="dense")
        # 先拟合再转换
        A = k.fit(X_fit).transform(X_fit)
        # 同时执行拟合和转换
        B = k.fit_transform(X_fit)
        # 比较两者的结果
        assert_array_almost_equal(np.abs(A), np.abs(B))


# 测试 kPCA 在预先计算的核函数下的工作性能，适用于所有求解器

# 使用随机数生成器创建一个5行4列的随机数矩阵作为训练数据
rng = np.random.RandomState(0)
X_fit = rng.random_sample((5, 4))
# 使用随机数生成器创建一个2行4列的随机数矩阵作为预测数据
X_pred = rng.random_sample((2, 4))
    # 遍历三种不同的特征值求解器类型：dense（稠密求解器）、arpack（稀疏矩阵求解器）、randomized（随机化求解器）
    for eigen_solver in ("dense", "arpack", "randomized"):
        # 使用 KernelPCA 进行降维，设置目标维度为 4，指定特征值求解器类型和随机种子
        X_kpca = (
            KernelPCA(4, eigen_solver=eigen_solver, random_state=0)
            .fit(X_fit)  # 在训练集上拟合 KernelPCA 模型
            .transform(X_pred)  # 对预测集进行数据转换
        )

        # 使用 KernelPCA 进行降维，设置目标维度为 4，指定特征值求解器类型、核函数为预计算类型和随机种子
        X_kpca2 = (
            KernelPCA(
                4, eigen_solver=eigen_solver, kernel="precomputed", random_state=0
            )
            .fit(np.dot(X_fit, X_fit.T))  # 在训练集上拟合 KernelPCA 模型，使用预计算的核矩阵
            .transform(np.dot(X_pred, X_fit.T))  # 对预测集使用预计算的核矩阵进行数据转换
        )

        # 使用 KernelPCA 进行降维，设置目标维度为 4，指定特征值求解器类型、核函数为预计算类型和随机种子，直接进行拟合和转换
        X_kpca_train = KernelPCA(
            4, eigen_solver=eigen_solver, kernel="precomputed", random_state=0
        ).fit_transform(np.dot(X_fit, X_fit.T))  # 在训练集上拟合 KernelPCA 模型，并对训练集进行数据转换

        # 使用 KernelPCA 进行降维，设置目标维度为 4，指定特征值求解器类型、核函数为预计算类型和随机种子
        X_kpca_train2 = (
            KernelPCA(
                4, eigen_solver=eigen_solver, kernel="precomputed", random_state=0
            )
            .fit(np.dot(X_fit, X_fit.T))  # 在训练集上拟合 KernelPCA 模型，使用预计算的核矩阵
            .transform(np.dot(X_fit, X_fit.T))  # 对训练集使用预计算的核矩阵进行数据转换
        )

        # 断言：验证 X_kpca 和 X_kpca2 的绝对值近似相等
        assert_array_almost_equal(np.abs(X_kpca), np.abs(X_kpca2))

        # 断言：验证 X_kpca_train 和 X_kpca_train2 的绝对值近似相等
        assert_array_almost_equal(np.abs(X_kpca_train), np.abs(X_kpca_train2))
@pytest.mark.parametrize("solver", ["auto", "dense", "arpack", "randomized"])
def test_kernel_pca_precomputed_non_symmetric(solver):
    """Check that the kernel centerer works.

    Tests that a non symmetric precomputed kernel is actually accepted
    because the kernel centerer does its job correctly.
    """

    # a non symmetric gram matrix
    K = [[1, 2], [3, 40]]
    # 创建一个 KernelPCA 对象，使用给定的求解器和参数进行初始化
    kpca = KernelPCA(
        kernel="precomputed", eigen_solver=solver, n_components=1, random_state=0
    )
    # 使用给定的 Gram 矩阵 K 来拟合 KernelPCA 模型
    kpca.fit(K)  # no error

    # same test with centered kernel
    Kc = [[9, -9], [-9, 9]]
    # 创建另一个 KernelPCA 对象，使用相同的参数，但应用于中心化的 Gram 矩阵 Kc
    kpca_c = KernelPCA(
        kernel="precomputed", eigen_solver=solver, n_components=1, random_state=0
    )
    # 使用中心化的 Gram 矩阵 Kc 来拟合 KernelPCA 模型
    kpca_c.fit(Kc)

    # comparison between the non-centered and centered versions
    # 比较未中心化和中心化版本的特征向量
    assert_array_equal(kpca.eigenvectors_, kpca_c.eigenvectors_)
    # 比较未中心化和中心化版本的特征值
    assert_array_equal(kpca.eigenvalues_, kpca_c.eigenvalues_)


def test_gridsearch_pipeline():
    """Check that kPCA works as expected in a grid search pipeline

    Test if we can do a grid-search to find parameters to separate
    circles with a perceptron model.
    """
    # 生成用于分类的圆形数据集
    X, y = make_circles(n_samples=400, factor=0.3, noise=0.05, random_state=0)
    # 创建一个 KernelPCA 对象，使用 RBF 核并指定 2 个主成分
    kpca = KernelPCA(kernel="rbf", n_components=2)
    # 创建一个包含 KernelPCA 和 Perceptron 的管道
    pipeline = Pipeline([("kernel_pca", kpca), ("Perceptron", Perceptron(max_iter=5))])
    # 定义网格搜索的参数网格，用于寻找最佳参数组合
    param_grid = dict(kernel_pca__gamma=2.0 ** np.arange(-2, 2))
    # 创建 GridSearchCV 对象，用于在管道上执行网格搜索
    grid_search = GridSearchCV(pipeline, cv=3, param_grid=param_grid)
    # 在数据集 X, y 上执行网格搜索
    grid_search.fit(X, y)
    # 断言网格搜索的最佳得分为 1
    assert grid_search.best_score_ == 1


def test_gridsearch_pipeline_precomputed():
    """Check that kPCA works as expected in a grid search pipeline (2)

    Test if we can do a grid-search to find parameters to separate
    circles with a perceptron model. This test uses a precomputed kernel.
    """
    # 生成用于分类的圆形数据集
    X, y = make_circles(n_samples=400, factor=0.3, noise=0.05, random_state=0)
    # 创建一个 KernelPCA 对象，使用预计算的核并指定 2 个主成分
    kpca = KernelPCA(kernel="precomputed", n_components=2)
    # 创建一个包含 KernelPCA 和 Perceptron 的管道
    pipeline = Pipeline([("kernel_pca", kpca), ("Perceptron", Perceptron(max_iter=5))])
    # 定义网格搜索的参数网格，用于寻找最佳参数组合
    param_grid = dict(Perceptron__max_iter=np.arange(1, 5))
    # 创建预计算核的数据矩阵
    X_kernel = rbf_kernel(X, gamma=2.0)
    # 创建 GridSearchCV 对象，用于在管道上执行网格搜索
    grid_search = GridSearchCV(pipeline, cv=3, param_grid=param_grid)
    # 在预计算的核数据 X_kernel 上执行网格搜索
    grid_search.fit(X_kernel, y)
    # 断言网格搜索的最佳得分为 1
    assert grid_search.best_score_ == 1


def test_nested_circles():
    """Check that kPCA projects in a space where nested circles are separable

    Tests that 2D nested circles become separable with a perceptron when
    projected in the first 2 kPCA using an RBF kernel, while raw samples
    are not directly separable in the original space.
    """
    # 生成用于分类的圆形数据集
    X, y = make_circles(n_samples=400, factor=0.3, noise=0.05, random_state=0)

    # 2D nested circles are not linearly separable
    # 使用感知机对原始数据 X, y 进行训练，并计算训练得分
    train_score = Perceptron(max_iter=5).fit(X, y).score(X, y)
    # 断言训练得分小于 0.8
    assert train_score < 0.8

    # Project the circles data into the first 2 components of a RBF Kernel
    # PCA model.
    # 将圆形数据集投影到 RBF Kernel PCA 模型的前两个主成分上。
    # 注意，gamma 值是依赖于数据的。如果这个测试断言失败，
    # 使用径向基函数 (RBF) 作为核函数进行核主成分分析 (Kernel PCA)，设置为提取2个主成分
    kpca = KernelPCA(
        kernel="rbf", n_components=2, fit_inverse_transform=True, gamma=2.0
    )
    # 对输入数据 X 进行核主成分分析，得到降维后的结果 X_kpca
    X_kpca = kpca.fit_transform(X)

    # 在降维后的空间中，数据是完全线性可分的
    # 使用感知机 (Perceptron) 对降维后的数据 X_kpca 进行训练，并计算训练集的准确率
    train_score = Perceptron(max_iter=5).fit(X_kpca, y).score(X_kpca, y)
    # 断言训练集的准确率为 1.0，即完美分类
    assert train_score == 1.0
def test_kernel_conditioning():
    """Check that ``_check_psd_eigenvalues`` is correctly called in kPCA

    Non-regression test for issue #12140 (PR #12145).
    """

    # create a pathological X leading to small non-zero eigenvalue
    # 创建一个导致有小的非零特征值的病态矩阵 X
    X = [[5, 1], [5 + 1e-8, 1e-8], [5 + 1e-8, 0]]
    # 使用线性核，设置为 2 个主成分，同时拟合逆变换
    kpca = KernelPCA(kernel="linear", n_components=2, fit_inverse_transform=True)
    # 对数据 X 进行核主成分分析拟合
    kpca.fit(X)

    # check that the small non-zero eigenvalue was correctly set to zero
    # 检查小的非零特征值是否被正确设置为零
    assert kpca.eigenvalues_.min() == 0
    # 检查所有特征值是否等于经过 _check_psd_eigenvalues 处理后的特征值
    assert np.all(kpca.eigenvalues_ == _check_psd_eigenvalues(kpca.eigenvalues_))


@pytest.mark.parametrize("solver", ["auto", "dense", "arpack", "randomized"])
def test_precomputed_kernel_not_psd(solver):
    """Check how KernelPCA works with non-PSD kernels depending on n_components

    Tests for all methods what happens with a non PSD gram matrix (this
    can happen in an isomap scenario, or with custom kernel functions, or
    maybe with ill-posed datasets).

    When ``n_component`` is large enough to capture a negative eigenvalue, an
    error should be raised. Otherwise, KernelPCA should run without error
    since the negative eigenvalues are not selected.
    """

    # a non PSD kernel with large eigenvalues, already centered
    # it was captured from an isomap call and multiplied by 100 for compacity
    # 一个具有大特征值的非半正定核矩阵，已经被居中处理，并且为了紧凑性乘以了 100
    K = [
        [4.48, -1.0, 8.07, 2.33, 2.33, 2.33, -5.76, -12.78],
        [-1.0, -6.48, 4.5, -1.24, -1.24, -1.24, -0.81, 7.49],
        [8.07, 4.5, 15.48, 2.09, 2.09, 2.09, -11.1, -23.23],
        [2.33, -1.24, 2.09, 4.0, -3.65, -3.65, 1.02, -0.9],
        [2.33, -1.24, 2.09, -3.65, 4.0, -3.65, 1.02, -0.9],
        [2.33, -1.24, 2.09, -3.65, -3.65, 4.0, 1.02, -0.9],
        [-5.76, -0.81, -11.1, 1.02, 1.02, 1.02, 4.86, 9.75],
        [-12.78, 7.49, -23.23, -0.9, -0.9, -0.9, 9.75, 21.46],
    ]
    # this gram matrix has 5 positive eigenvalues and 3 negative ones
    # 这个 Gram 矩阵有 5 个正特征值和 3 个负特征值
    # [ 52.72,   7.65,   7.65,   5.02,   0.  ,  -0.  ,  -6.13, -15.11]

    # 1. ask for enough components to get a significant negative one
    # 请求足够多的主成分以获取一个显著的负特征值
    kpca = KernelPCA(kernel="precomputed", eigen_solver=solver, n_components=7)
    # 确保会抛出适当的错误
    with pytest.raises(ValueError, match="There are significant negative eigenvalues"):
        # 对核矩阵 K 进行核主成分分析拟合
        kpca.fit(K)

    # 2. ask for a small enough n_components to get only positive ones
    # 请求足够小的 n_components 以获取只有正特征值
    kpca = KernelPCA(kernel="precomputed", eigen_solver=solver, n_components=2)
    if solver == "randomized":
        # 随机化方法在此问题上仍与其他方法不一致
        # 因为它基于最大的两个模块选择特征值，而不是基于最大的两个值
        # 至少我们可以确保抛出错误而不是返回错误的特征值
        with pytest.raises(
            ValueError, match="There are significant negative eigenvalues"
        ):
            kpca.fit(K)
    else:
        # 一般情况：确保正常工作
        kpca.fit(K)
@pytest.mark.parametrize("n_components", [4, 10, 20])
def test_kernel_pca_solvers_equivalence(n_components):
    """Check that 'dense' 'arpack' & 'randomized' solvers give similar results"""

    # Generate random data
    n_train, n_test = 1_000, 100
    X, _ = make_circles(
        n_samples=(n_train + n_test), factor=0.3, noise=0.05, random_state=0
    )
    X_fit, X_pred = X[:n_train, :], X[n_train:, :]

    # reference (full)
    ref_pred = (
        KernelPCA(n_components, eigen_solver="dense", random_state=0)
        .fit(X_fit)
        .transform(X_pred)
    )

    # arpack
    a_pred = (
        KernelPCA(n_components, eigen_solver="arpack", random_state=0)
        .fit(X_fit)
        .transform(X_pred)
    )
    # check that the result is still correct despite the approx
    assert_array_almost_equal(np.abs(a_pred), np.abs(ref_pred))

    # randomized
    r_pred = (
        KernelPCA(n_components, eigen_solver="randomized", random_state=0)
        .fit(X_fit)
        .transform(X_pred)
    )
    # check that the result is still correct despite the approximation
    assert_array_almost_equal(np.abs(r_pred), np.abs(ref_pred))


def test_kernel_pca_inverse_transform_reconstruction():
    """Test if the reconstruction is a good approximation.

    Note that in general it is not possible to get an arbitrarily good
    reconstruction because of kernel centering that does not
    preserve all the information of the original data.
    """
    X, *_ = make_blobs(n_samples=100, n_features=4, random_state=0)

    kpca = KernelPCA(
        n_components=20, kernel="rbf", fit_inverse_transform=True, alpha=1e-3
    )
    X_trans = kpca.fit_transform(X)
    X_reconst = kpca.inverse_transform(X_trans)
    assert np.linalg.norm(X - X_reconst) / np.linalg.norm(X) < 1e-1


def test_kernel_pca_raise_not_fitted_error():
    X = np.random.randn(15).reshape(5, 3)
    kpca = KernelPCA()
    kpca.fit(X)
    with pytest.raises(NotFittedError):
        kpca.inverse_transform(X)


def test_32_64_decomposition_shape():
    """Test that the decomposition is similar for 32 and 64 bits data

    Non regression test for
    https://github.com/scikit-learn/scikit-learn/issues/18146
    """
    X, y = make_blobs(
        n_samples=30, centers=[[0, 0, 0], [1, 1, 1]], random_state=0, cluster_std=0.1
    )
    X = StandardScaler().fit_transform(X)
    X -= X.min()

    # Compare the shapes (corresponds to the number of non-zero eigenvalues)
    kpca = KernelPCA()
    assert kpca.fit_transform(X).shape == kpca.fit_transform(X.astype(np.float32)).shape


def test_kernel_pca_feature_names_out():
    """Check feature names out for KernelPCA."""
    X, *_ = make_blobs(n_samples=100, n_features=4, random_state=0)
    kpca = KernelPCA(n_components=2).fit(X)

    names = kpca.get_feature_names_out()
    assert_array_equal([f"kernelpca{i}" for i in range(2)], names)


def test_kernel_pca_inverse_correct_gamma():
    """Check that gamma is set correctly when not provided."""
    Non-regression test for #26280
    """
    # 创建一个随机数生成器，种子为0
    rng = np.random.RandomState(0)
    # 生成一个 5x4 的随机数组
    X = rng.random_sample((5, 4))

    # 定义参数字典
    kwargs = {
        "n_components": 2,                      # PCA 组件数量为 2
        "random_state": rng,                    # 随机数生成器
        "fit_inverse_transform": True,          # 拟合逆变换开启
        "kernel": "rbf",                        # 核函数选择为 RBF
    }

    # 计算预期的 gamma 值
    expected_gamma = 1 / X.shape[1]

    # 创建两个 KernelPCA 实例，分别使用不同的 gamma 值进行拟合
    kpca1 = KernelPCA(gamma=None, **kwargs).fit(X)
    kpca2 = KernelPCA(gamma=expected_gamma, **kwargs).fit(X)

    # 断言两个实例的 gamma 值应该与预期相等
    assert kpca1.gamma_ == expected_gamma
    assert kpca2.gamma_ == expected_gamma

    # 对原始数据进行 PCA 变换和逆变换，并断言两次逆变换结果的近似性
    X1_recon = kpca1.inverse_transform(kpca1.transform(X))
    X2_recon = kpca2.inverse_transform(kpca1.transform(X))

    assert_allclose(X1_recon, X2_recon)
# 定义测试函数 `test_kernel_pca_pandas_output()`，用于验证 KernelPCA 在使用 arpack 求解器时与 pandas 输出一起工作正常。

"""Check that KernelPCA works with pandas output when the solver is arpack.

Non-regression test for:
https://github.com/scikit-learn/scikit-learn/issues/27579
"""
# 导入 pytest 库，如果找不到 pandas 库则跳过测试
pytest.importorskip("pandas")

# 载入鸢尾花数据集，以 DataFrame 格式加载特征矩阵 X
X, _ = load_iris(as_frame=True, return_X_y=True)

# 设置 sklearn 的上下文环境，将转换后的输出格式设置为 pandas
with sklearn.config_context(transform_output="pandas"):
    # 使用 KernelPCA 模型，设定主成分数为 2，使用 arpack 求解器进行拟合和转换
    KernelPCA(n_components=2, eigen_solver="arpack").fit_transform(X)
```