# `D:\src\scipysrc\scikit-learn\sklearn\covariance\tests\test_robust_covariance.py`

```
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入 itertools 模块，用于迭代工具函数
import itertools

# 导入 numpy 库，并使用别名 np
import numpy as np

# 导入 pytest 库，用于编写和运行测试用例
import pytest

# 从 sklearn 库中导入 datasets 模块
from sklearn import datasets

# 从 sklearn.covariance 中导入 MinCovDet, empirical_covariance, fast_mcd 类
from sklearn.covariance import MinCovDet, empirical_covariance, fast_mcd

# 从 sklearn.utils._testing 中导入 assert_array_almost_equal 函数
from sklearn.utils._testing import assert_array_almost_equal

# 从 sklearn.datasets 加载鸢尾花数据集的特征数据到 X 变量中
X = datasets.load_iris().data

# 从 X 中取出第一列数据，存储到 X_1d 变量中
X_1d = X[:, 0]

# 获取 X 的样本数和特征数，分别存储到 n_samples 和 n_features 变量中
n_samples, n_features = X.shape


# 定义测试函数 test_mcd，测试 FastMCD 算法的实现
def test_mcd(global_random_seed):
    # 测试用例1：小数据集，无异常值（随机独立的正态分布数据）
    launch_mcd_on_dataset(100, 5, 0, 0.02, 0.1, 75, global_random_seed)
    # 测试用例2：包含异常值的数据集（中等污染）
    launch_mcd_on_dataset(100, 5, 20, 0.3, 0.3, 65, global_random_seed)
    # 测试用例3：包含异常值的数据集（强烈污染）
    launch_mcd_on_dataset(100, 5, 40, 0.1, 0.1, 50, global_random_seed)

    # 中等大小的数据集
    launch_mcd_on_dataset(1000, 5, 450, 0.1, 0.1, 540, global_random_seed)

    # 大型数据集
    launch_mcd_on_dataset(1700, 5, 800, 0.1, 0.1, 870, global_random_seed)

    # 一维数据集
    launch_mcd_on_dataset(500, 1, 100, 0.02, 0.02, 350, global_random_seed)


# 定义测试函数 test_fast_mcd_on_invalid_input，测试 fast_mcd 函数对无效输入的反应
def test_fast_mcd_on_invalid_input():
    # 创建一个长度为 100 的数组 X
    X = np.arange(100)
    # 定义期望的错误信息
    msg = "Expected 2D array, got 1D array instead"
    # 使用 pytest 来检查是否引发 ValueError，并匹配期望的错误信息
    with pytest.raises(ValueError, match=msg):
        fast_mcd(X)


# 定义测试函数 test_mcd_class_on_invalid_input，测试 MinCovDet 类在无效输入时的反应
def test_mcd_class_on_invalid_input():
    # 创建一个长度为 100 的数组 X
    X = np.arange(100)
    # 创建一个 MinCovDet 的对象 mcd
    mcd = MinCovDet()
    # 定义期望的错误信息
    msg = "Expected 2D array, got 1D array instead"
    # 使用 pytest 来检查是否引发 ValueError，并匹配期望的错误信息
    with pytest.raises(ValueError, match=msg):
        mcd.fit(X)


# 定义函数 launch_mcd_on_dataset，对数据集执行 MCD 算法的测试
def launch_mcd_on_dataset(
    n_samples, n_features, n_outliers, tol_loc, tol_cov, tol_support, seed
):
    # 创建随机数生成器 rand_gen，并设置随机种子为 seed
    rand_gen = np.random.RandomState(seed)
    # 生成一个 n_samples 行 n_features 列的随机数据矩阵 data
    data = rand_gen.randn(n_samples, n_features)
    # 添加一些异常值
    outliers_index = rand_gen.permutation(n_samples)[:n_outliers]
    outliers_offset = 10.0 * (rand_gen.randint(2, size=(n_outliers, n_features)) - 0.5)
    data[outliers_index] += outliers_offset
    # 创建一个布尔掩码数组，表示正常值和异常值
    inliers_mask = np.ones(n_samples).astype(bool)
    inliers_mask[outliers_index] = False

    # 提取出正常数据
    pure_data = data[inliers_mask]

    # 通过拟合对象计算 MCD
    mcd_fit = MinCovDet(random_state=seed).fit(data)
    T = mcd_fit.location_  # 得到位置估计值
    S = mcd_fit.covariance_  # 得到协方差矩阵估计值
    H = mcd_fit.support_  # 得到支持值

    # 与从正常值学习的估计进行比较
    error_location = np.mean((pure_data.mean(0) - T) ** 2)
    assert error_location < tol_loc
    error_cov = np.mean((empirical_covariance(pure_data) - S) ** 2)
    assert error_cov < tol_cov
    assert np.sum(H) >= tol_support
    assert_array_almost_equal(mcd_fit.mahalanobis(data), mcd_fit.dist_)


# 定义测试函数 test_mcd_issue1127，检查当 X.shape = (3, 1) 时代码不会中断
def test_mcd_issue1127():
    # 创建一个符合正态分布的 3 行 1 列的数据矩阵 X
    rnd = np.random.RandomState(0)
    X = rnd.normal(size=(3, 1))
    # 创建一个 MinCovDet 的对象 mcd
    mcd = MinCovDet()
    # 对数据进行拟合
    mcd.fit(X)


# 定义测试函数 test_mcd_issue3367，检查当协方差矩阵奇异时 MCD 是否能完成
def test_mcd_issue3367(global_random_seed):
    # 留待实现，这里只给出了函数定义，但未给出具体的测试内容
    pass
    # 使用全局随机种子初始化随机数生成器
    rand_gen = np.random.RandomState(global_random_seed)

    # 创建一个包含10个值，范围在-5到5之间的数列，并转换为列表形式
    data_values = np.linspace(-5, 5, 10).tolist()

    # 获取上述数列所有可能的笛卡尔积，形成所有可能的坐标对
    data = np.array(list(itertools.product(data_values, data_values)))

    # 添加一个全为零的第三列，使得数据成为一个平面内的点集，
    # 这样协方差矩阵将会是奇异的
    data = np.hstack((data, np.zeros((data.shape[0], 1))))

    # 如果协方差矩阵是奇异的，下面的代码将会引发异常。此外，因为我们有XYZ坐标点，
    # 这些坐标点的主成分（特征向量）直接与点的几何形状相关联。因为这是一个平面，
    # 我们应该能够测试最小特征值对应的特征向量是否是平面的法向量，具体来说是[0, 0, 1]，
    # 因为所有点都在XY平面上（正如上面设置的那样）。为了做到这一点，首先应该执行：
    #
    #     evals, evecs = np.linalg.eigh(mcd_fit.covariance_)
    #     normal = evecs[:, np.argmin(evals)]
    #
    # 然后我们需要断言我们的 `normal` 是否等于 [0, 0, 1]。
    # 需要注意的是，由于浮点数误差，最好是先相减，然后再比较一些小的容差（例如 1e-12）。
    MinCovDet(random_state=rand_gen).fit(data)
# 定义一个测试函数，用于检查 MCD 在支持数据的协方差为零时是否返回 ValueError，并包含信息性的错误消息。
def test_mcd_support_covariance_is_zero():
    # 创建两个包含浮点数的 NumPy 数组，分别作为支持数据 X_1 和 X_2
    X_1 = np.array([0.5, 0.1, 0.1, 0.1, 0.957, 0.1, 0.1, 0.1, 0.4285, 0.1])
    X_1 = X_1.reshape(-1, 1)  # 调整 X_1 的形状为列向量
    X_2 = np.array([0.5, 0.3, 0.3, 0.3, 0.957, 0.3, 0.3, 0.3, 0.4285, 0.3])
    X_2 = X_2.reshape(-1, 1)  # 调整 X_2 的形状为列向量
    # 定义错误消息，指示支持数据的协方差矩阵为零时应尝试增加 support_fraction
    msg = (
        "The covariance matrix of the support data is equal to 0, try to "
        "increase support_fraction"
    )
    # 对 X_1 和 X_2 进行迭代
    for X in [X_1, X_2]:
        # 使用 pytest 检查是否引发 ValueError，且错误消息匹配 msg
        with pytest.raises(ValueError, match=msg):
            MinCovDet().fit(X)


# 定义一个测试函数，用于检查在 c_step 过程中是否会引发警告，如果观察到行列式（determinant）增加。
# 理论上，行列式的序列应该是递减的。增加的行列式可能是由于条件糟糕的协方差矩阵导致的，进而导致精度矩阵不佳。
def test_mcd_increasing_det_warning(global_random_seed):
    # 创建一个包含浮点数的二维列表 X，作为输入数据
    X = [
        [5.1, 3.5, 1.4, 0.2],
        [4.9, 3.0, 1.4, 0.2],
        [4.7, 3.2, 1.3, 0.2],
        [4.6, 3.1, 1.5, 0.2],
        [5.0, 3.6, 1.4, 0.2],
        [4.6, 3.4, 1.4, 0.3],
        [5.0, 3.4, 1.5, 0.2],
        [4.4, 2.9, 1.4, 0.2],
        [4.9, 3.1, 1.5, 0.1],
        [5.4, 3.7, 1.5, 0.2],
        [4.8, 3.4, 1.6, 0.2],
        [4.8, 3.0, 1.4, 0.1],
        [4.3, 3.0, 1.1, 0.1],
        [5.1, 3.5, 1.4, 0.3],
        [5.7, 3.8, 1.7, 0.3],
        [5.4, 3.4, 1.7, 0.2],
        [4.6, 3.6, 1.0, 0.2],
        [5.0, 3.0, 1.6, 0.2],
        [5.2, 3.5, 1.5, 0.2],
    ]
    # 创建一个 MinCovDet 对象 mcd，设置支持分数为 0.5，并使用全局随机种子
    mcd = MinCovDet(support_fraction=0.5, random_state=global_random_seed)
    # 定义警告消息，指示行列式增加的情况
    warn_msg = "Determinant has increased"
    # 使用 pytest 检查是否引发 RuntimeWarning，且警告消息匹配 warn_msg
    with pytest.warns(RuntimeWarning, match=warn_msg):
        mcd.fit(X)
```