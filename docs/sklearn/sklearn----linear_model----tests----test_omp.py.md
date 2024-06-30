# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\tests\test_omp.py`

```
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入必要的库
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于单元测试

# 导入 scikit-learn 的相关模块和函数
from sklearn.datasets import make_sparse_coded_signal  # 导入生成稀疏编码信号的函数
from sklearn.linear_model import (  # 导入线性模型中的不同方法
    LinearRegression,
    OrthogonalMatchingPursuit,
    OrthogonalMatchingPursuitCV,
    orthogonal_mp,
    orthogonal_mp_gram,
)
from sklearn.utils import check_random_state  # 导入用于检查随机状态的函数
from sklearn.utils._testing import (  # 导入用于单元测试的辅助函数
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
    ignore_warnings,
)

# 定义数据集的维度和参数
n_samples, n_features, n_nonzero_coefs, n_targets = 25, 35, 5, 3

# 生成稀疏编码信号数据集
y, X, gamma = make_sparse_coded_signal(
    n_samples=n_targets,
    n_components=n_features,
    n_features=n_samples,
    n_nonzero_coefs=n_nonzero_coefs,
    random_state=0,
)

# 调整数据的维度
y, X, gamma = y.T, X.T, gamma.T

# 为了测试，将 X 的值扩大 10 倍，使其不再是单位范数
X *= 10
y *= 10

# 计算 Gram 矩阵 G 和 X 的转置乘以 y
G, Xy = np.dot(X.T, X), np.dot(X.T, y)

# 此时 X 的维度为 (n_samples, n_features)，y 的维度为 (n_samples, 3)
    # 使用 pytest 的上下文管理器来检查是否引发了 ValueError 异常
    with pytest.raises(ValueError):
        # 调用 orthogonal_mp 函数，并传入位置参数 positional_params 和关键字参数 keyword_params
        orthogonal_mp(*positional_params, **keyword_params)
def test_perfect_signal_recovery():
    # 从 gamma 的第一列中找出非零元素的索引
    (idx,) = gamma[:, 0].nonzero()
    # 使用 orthogonal_mp 函数恢复信号 gamma_rec
    gamma_rec = orthogonal_mp(X, y[:, 0], n_nonzero_coefs=5)
    # 使用 orthogonal_mp_gram 函数恢复信号 gamma_gram
    gamma_gram = orthogonal_mp_gram(G, Xy[:, 0], n_nonzero_coefs=5)
    # 检查 gamma_rec 中非零元素的索引是否与 idx 相同
    assert_array_equal(idx, np.flatnonzero(gamma_rec))
    # 检查 gamma_gram 中非零元素的索引是否与 idx 相同
    assert_array_equal(idx, np.flatnonzero(gamma_gram))
    # 检查 gamma 的第一列与 gamma_rec 的值是否几乎相等，精度为小数点后两位
    assert_array_almost_equal(gamma[:, 0], gamma_rec, decimal=2)
    # 检查 gamma 的第一列与 gamma_gram 的值是否几乎相等，精度为小数点后两位
    assert_array_almost_equal(gamma[:, 0], gamma_gram, decimal=2)


def test_orthogonal_mp_gram_readonly():
    # 针对 https://github.com/scikit-learn/scikit-learn/issues/5956 的非回归测试
    (idx,) = gamma[:, 0].nonzero()
    # 复制 G 并设置为只读
    G_readonly = G.copy()
    G_readonly.setflags(write=False)
    # 复制 Xy 并设置为只读
    Xy_readonly = Xy.copy()
    Xy_readonly.setflags(write=False)
    # 使用 orthogonal_mp_gram 函数恢复信号 gamma_gram，不复制 Gram 矩阵和 Xy
    gamma_gram = orthogonal_mp_gram(
        G_readonly, Xy_readonly[:, 0], n_nonzero_coefs=5, copy_Gram=False, copy_Xy=False
    )
    # 检查 gamma_gram 中非零元素的索引是否与 idx 相同
    assert_array_equal(idx, np.flatnonzero(gamma_gram))
    # 检查 gamma 的第一列与 gamma_gram 的值是否几乎相等，精度为小数点后两位
    assert_array_almost_equal(gamma[:, 0], gamma_gram, decimal=2)


def test_estimator():
    # 创建 OrthogonalMatchingPursuit 对象 omp
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
    # 使用 X 和 y 的第一列拟合 omp
    omp.fit(X, y[:, 0])
    # 检查 omp 的 coef_ 属性形状是否为 (n_features,)
    assert omp.coef_.shape == (n_features,)
    # 检查 omp 的 intercept_ 属性形状是否为 ()
    assert omp.intercept_.shape == ()
    # 检查 omp 的 coef_ 中非零元素个数是否不超过 n_nonzero_coefs
    assert np.count_nonzero(omp.coef_) <= n_nonzero_coefs

    # 使用 X 和 y 拟合 omp
    omp.fit(X, y)
    # 检查 omp 的 coef_ 属性形状是否为 (n_targets, n_features)
    assert omp.coef_.shape == (n_targets, n_features)
    # 检查 omp 的 intercept_ 属性形状是否为 (n_targets,)
    assert omp.intercept_.shape == (n_targets,)
    # 检查 omp 的 coef_ 中非零元素个数是否不超过 n_targets * n_nonzero_coefs

    # 复制 coef_ 的第一行到 coef_normalized
    coef_normalized = omp.coef_[0].copy()
    # 设置 fit_intercept=True 后重新拟合 omp
    omp.set_params(fit_intercept=True)
    omp.fit(X, y[:, 0])
    # 检查 coef_normalized 是否与 omp 的 coef_ 几乎相等
    assert_array_almost_equal(coef_normalized, omp.coef_)

    # 设置 fit_intercept=False 后重新拟合 omp
    omp.set_params(fit_intercept=False)
    omp.fit(X, y[:, 0])
    # 检查 omp 的 coef_ 中非零元素个数是否不超过 n_nonzero_coefs
    assert np.count_nonzero(omp.coef_) <= n_nonzero_coefs
    # 检查 omp 的 coef_ 属性形状是否为 (n_features,)
    assert omp.coef_.shape == (n_features,)
    # 检查 omp 的 intercept_ 是否为 0
    assert omp.intercept_ == 0

    # 使用 X 和 y 拟合 omp
    omp.fit(X, y)
    # 检查 omp 的 coef_ 属性形状是否为 (n_targets, n_features)
    assert omp.coef_.shape == (n_targets, n_features)
    # 检查 omp 的 intercept_ 是否为 0
    assert omp.intercept_ == 0
    # 检查 omp 的 coef_ 中非零元素个数是否不超过 n_targets * n_nonzero_coefs


def test_estimator_n_nonzero_coefs():
    """检查当 `tol` 被设置和未被设置时 `n_nonzero_coefs_` 是否正确。"""
    # 创建 OrthogonalMatchingPursuit 对象 omp
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
    # 使用 X 和 y 的第一列拟合 omp
    omp.fit(X, y[:, 0])
    # 检查 omp 的 n_nonzero_coefs_ 是否等于 n_nonzero_coefs

    # 创建带有 tol 参数的 OrthogonalMatchingPursuit 对象 omp
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs, tol=0.5)
    # 使用 X 和 y 的第一列拟合 omp
    omp.fit(X, y[:, 0])
    # 检查 omp 的 n_nonzero_coefs_ 是否为 None


def test_identical_regressors():
    # 复制 X 到 newX
    newX = X.copy()
    # 将 newX 的第二列设为 newX 的第一列
    newX[:, 1] = newX[:, 0]
    # 创建全零的 gamma 向量
    gamma = np.zeros(n_features)
    # 将 gamma 的前两个元素设为 1.0
    gamma[0] = gamma[1] = 1.0
    # 通过 newX 和 gamma 计算得到 newy
    newy = np.dot(newX, gamma)
    # 设置警告信息
    warning_message = (
        "Orthogonal matching pursuit ended prematurely "
        "due to linear dependence in the dictionary. "
        "The requested precision might not have been met."
    )
    # 检查是否触发 RuntimeWarning 并且警告信息匹配 warning_message
    with pytest.warns(RuntimeWarning, match=warning_message):
        orthogonal_mp(newX, newy, n_nonzero_coefs=2)


def test_swapped_regressors():
    # 创建全零的 gamma 向量
    gamma = np.zeros(n_features)
    # 设置 gamma 向量的第 21 个元素为 1.0
    gamma[21] = 1.0
    # 设置 gamma 向量的第 0 个元素为 0.5
    gamma[0] = 0.5
    # 计算新的响应变量 new_y，通过矩阵乘法 X 和 gamma
    new_y = np.dot(X, gamma)
    # 计算新的特征矩阵与响应变量的乘积 new_Xy，使用 X 的转置与 new_y 的乘积
    new_Xy = np.dot(X.T, new_y)
    # 使用正交匹配追踪算法计算估计的稀疏系数 gamma_hat，限制非零系数个数为 2
    gamma_hat = orthogonal_mp(X, new_y, n_nonzero_coefs=2)
    # 使用正交匹配追踪算法计算基于 Gram 矩阵 G 的估计稀疏系数 gamma_hat_gram，限制非零系数个数为 2
    gamma_hat_gram = orthogonal_mp_gram(G, new_Xy, n_nonzero_coefs=2)
    # 断言 gamma_hat 中非零元素的索引为 [0, 21]
    assert_array_equal(np.flatnonzero(gamma_hat), [0, 21])
    # 断言 gamma_hat_gram 中非零元素的索引为 [0, 21]
    assert_array_equal(np.flatnonzero(gamma_hat_gram), [0, 21])
def test_no_atoms():
    # 创建一个与 y 维度相同的全零数组
    y_empty = np.zeros_like(y)
    # 计算 X 的转置与 y_empty 的乘积
    Xy_empty = np.dot(X.T, y_empty)
    # 使用 orthogonal_mp 函数计算稀疏表示，忽略警告
    gamma_empty = ignore_warnings(orthogonal_mp)(X, y_empty, n_nonzero_coefs=1)
    # 使用 orthogonal_mp 函数计算基于 Gram 矩阵的稀疏表示，忽略警告
    gamma_empty_gram = ignore_warnings(orthogonal_mp)(G, Xy_empty, n_nonzero_coefs=1)
    # 断言 gamma_empty 全为零
    assert np.all(gamma_empty == 0)
    # 断言 gamma_empty_gram 全为零
    assert np.all(gamma_empty_gram == 0)


def test_omp_path():
    # 使用 orthogonal_mp 函数获取路径信息
    path = orthogonal_mp(X, y, n_nonzero_coefs=5, return_path=True)
    # 使用 orthogonal_mp 函数获取最终结果
    last = orthogonal_mp(X, y, n_nonzero_coefs=5, return_path=False)
    # 断言路径的形状为 (n_features, n_targets, 5)
    assert path.shape == (n_features, n_targets, 5)
    # 检查路径的最后一步与直接结果的近似性
    assert_array_almost_equal(path[:, :, -1], last)
    # 使用 orthogonal_mp_gram 函数获取路径信息
    path = orthogonal_mp_gram(G, Xy, n_nonzero_coefs=5, return_path=True)
    # 使用 orthogonal_mp_gram 函数获取最终结果
    last = orthogonal_mp_gram(G, Xy, n_nonzero_coefs=5, return_path=False)
    # 断言路径的形状为 (n_features, n_targets, 5)
    assert path.shape == (n_features, n_targets, 5)
    # 检查路径的最后一步与直接结果的近似性
    assert_array_almost_equal(path[:, :, -1], last)


def test_omp_return_path_prop_with_gram():
    # 使用 orthogonal_mp 函数获取路径信息，并预先计算相关数据
    path = orthogonal_mp(X, y, n_nonzero_coefs=5, return_path=True, precompute=True)
    # 使用 orthogonal_mp 函数获取最终结果，并预先计算相关数据
    last = orthogonal_mp(X, y, n_nonzero_coefs=5, return_path=False, precompute=True)
    # 断言路径的形状为 (n_features, n_targets, 5)
    assert path.shape == (n_features, n_targets, 5)
    # 检查路径的最后一步与直接结果的近似性
    assert_array_almost_equal(path[:, :, -1], last)


def test_omp_cv():
    # 取 y 的第一列
    y_ = y[:, 0]
    # 取 gamma 的第一列
    gamma_ = gamma[:, 0]
    # 初始化交叉验证的正交匹配追踪模型，不包括截距项，最大迭代次数为 10
    ompcv = OrthogonalMatchingPursuitCV(fit_intercept=False, max_iter=10)
    # 拟合模型
    ompcv.fit(X, y_)
    # 断言非零系数的数量与预期相符
    assert ompcv.n_nonzero_coefs_ == n_nonzero_coefs
    # 检查预测的系数与 gamma_ 的近似性
    assert_array_almost_equal(ompcv.coef_, gamma_)
    # 初始化正交匹配追踪模型，不包括截距项，非零系数数量由交叉验证模型决定
    omp = OrthogonalMatchingPursuit(
        fit_intercept=False, n_nonzero_coefs=ompcv.n_nonzero_coefs_
    )
    # 拟合模型
    omp.fit(X, y_)
    # 检查预测的系数与交叉验证模型的系数近似性
    assert_array_almost_equal(ompcv.coef_, omp.coef_)


def test_omp_reaches_least_squares():
    # 使用简单的小型数据进行验证；这是一个健全性检查，但是 OMP 可能会提前停止
    rng = check_random_state(0)
    n_samples, n_features = (10, 8)
    n_targets = 3
    X = rng.randn(n_samples, n_features)
    Y = rng.randn(n_samples, n_targets)
    # 初始化正交匹配追踪模型，非零系数数量设定为特征数量
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_features)
    # 初始化最小二乘回归模型
    lstsq = LinearRegression()
    # 拟合正交匹配追踪模型
    omp.fit(X, Y)
    # 拟合最小二乘回归模型
    lstsq.fit(X, Y)
    # 检查正交匹配追踪模型的系数与最小二乘回归模型的系数近似性
    assert_array_almost_equal(omp.coef_, lstsq.coef_)


@pytest.mark.parametrize("data_type", (np.float32, np.float64))
def test_omp_gram_dtype_match(data_type):
    # 验证输入数据类型和输出数据类型匹配
    coef = orthogonal_mp_gram(
        G.astype(data_type), Xy.astype(data_type), n_nonzero_coefs=5
    )
    # 断言系数的数据类型与预期匹配
    assert coef.dtype == data_type


def test_omp_gram_numerical_consistency():
    # 验证 np.float32 和 np.float64 之间的数值一致性
    coef_32 = orthogonal_mp_gram(
        G.astype(np.float32), Xy.astype(np.float32), n_nonzero_coefs=5
    )
    coef_64 = orthogonal_mp_gram(
        G.astype(np.float32), Xy.astype(np.float64), n_nonzero_coefs=5
    )
    # 检查所有元素的接近性
    assert_allclose(coef_32, coef_64)
```