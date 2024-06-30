# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\tests\test_ransac.py`

```
# 导入所需的库和模块
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from sklearn.datasets import make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
    LinearRegression,
    OrthogonalMatchingPursuit,
    RANSACRegressor,
    Ridge,
)
from sklearn.linear_model._ransac import _dynamic_max_trials
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_allclose
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS

# 生成线性回归数据集
X = np.arange(-200, 200)
y = 0.2 * X + 20
data = np.column_stack([X, y])

# 添加一些错误的数据点
rng = np.random.RandomState(1000)
outliers = np.unique(rng.randint(len(X), size=200))
data[outliers, :] += 50 + rng.rand(len(outliers), 2) * 10

# 重新定义 X 和 y
X = data[:, 0][:, np.newaxis]
y = data[:, 1]

# 测试 RANSACRegressor 的 inliers 和 outliers
def test_ransac_inliers_outliers():
    estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(
        estimator, min_samples=2, residual_threshold=5, random_state=0
    )

    # 对损坏数据估计参数
    ransac_estimator.fit(X, y)

    # 真实的内点掩码
    ref_inlier_mask = np.ones_like(ransac_estimator.inlier_mask_).astype(np.bool_)
    ref_inlier_mask[outliers] = False

    assert_array_equal(ransac_estimator.inlier_mask_, ref_inlier_mask)


# 测试 RANSACRegressor 中的数据有效性检查
def test_ransac_is_data_valid():
    def is_data_valid(X, y):
        assert X.shape[0] == 2
        assert y.shape[0] == 2
        return False

    rng = np.random.RandomState(0)
    X = rng.rand(10, 2)
    y = rng.rand(10, 1)

    estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(
        estimator,
        min_samples=2,
        residual_threshold=5,
        is_data_valid=is_data_valid,
        random_state=0,
    )
    with pytest.raises(ValueError):
        ransac_estimator.fit(X, y)


# 测试 RANSACRegressor 中的模型有效性检查
def test_ransac_is_model_valid():
    def is_model_valid(estimator, X, y):
        assert X.shape[0] == 2
        assert y.shape[0] == 2
        return False

    estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(
        estimator,
        min_samples=2,
        residual_threshold=5,
        is_model_valid=is_model_valid,
        random_state=0,
    )
    with pytest.raises(ValueError):
        ransac_estimator.fit(X, y)


# 测试 RANSACRegressor 中的最大试验次数限制
def test_ransac_max_trials():
    estimator = LinearRegression()

    # 当 max_trials=0 时应该抛出 ValueError
    ransac_estimator = RANSACRegressor(
        estimator,
        min_samples=2,
        residual_threshold=5,
        max_trials=0,
        random_state=0,
    )
    with pytest.raises(ValueError):
        ransac_estimator.fit(X, y)

    # 计算动态的最大试验次数
    max_trials = _dynamic_max_trials(len(X) - len(outliers), X.shape[0], 2, 1 - 1e-9)
    ransac_estimator = RANSACRegressor(estimator, min_samples=2)
    # 循环执行50次，每次设置不同的随机状态并使用 RANSAC 估计器拟合数据
    for i in range(50):
        # 设置 RANSAC 估计器的参数：最小样本数为2，随机状态为当前循环的索引值 i
        ransac_estimator.set_params(min_samples=2, random_state=i)
        # 使用当前参数拟合数据集 X 和对应的目标值 y
        ransac_estimator.fit(X, y)
        # 断言 RANSAC 估计器的试验次数小于最大试验次数加一
        assert ransac_estimator.n_trials_ < max_trials + 1
# 定义一个测试函数，用于测试 RANSACRegressor 的 stop_n_inliers 参数
def test_ransac_stop_n_inliers():
    # 创建一个线性回归模型作为基础评估器
    estimator = LinearRegression()
    # 创建 RANSACRegressor 实例，设置最小样本数为 2，残差阈值为 5，停止条件为达到 2 个内点，随机状态为 0
    ransac_estimator = RANSACRegressor(
        estimator,
        min_samples=2,
        residual_threshold=5,
        stop_n_inliers=2,
        random_state=0,
    )
    # 对数据集 X, y 进行拟合
    ransac_estimator.fit(X, y)

    # 断言 RANSAC 模型的试验次数为 1
    assert ransac_estimator.n_trials_ == 1


# 定义一个测试函数，用于测试 RANSACRegressor 的 stop_score 参数
def test_ransac_stop_score():
    # 创建一个线性回归模型作为基础评估器
    estimator = LinearRegression()
    # 创建 RANSACRegressor 实例，设置最小样本数为 2，残差阈值为 5，停止条件为达到分数为 0，随机状态为 0
    ransac_estimator = RANSACRegressor(
        estimator,
        min_samples=2,
        residual_threshold=5,
        stop_score=0,
        random_state=0,
    )
    # 对数据集 X, y 进行拟合
    ransac_estimator.fit(X, y)

    # 断言 RANSAC 模型的试验次数为 1
    assert ransac_estimator.n_trials_ == 1


# 定义一个测试函数，用于测试 RANSACRegressor 的 score 方法
def test_ransac_score():
    # 创建一个一维的数据集 X，100 个样本
    X = np.arange(100)[:, None]
    # 创建一个一维的标签 y，100 个样本，大部分为 0，但第一个和第二个样本分别为 1 和 100
    y = np.zeros((100,))
    y[0] = 1
    y[1] = 100

    # 创建一个线性回归模型作为基础评估器
    estimator = LinearRegression()
    # 创建 RANSACRegressor 实例，设置最小样本数为 2，残差阈值为 0.5，随机状态为 0
    ransac_estimator = RANSACRegressor(
        estimator, min_samples=2, residual_threshold=0.5, random_state=0
    )
    # 对数据集 X, y 进行拟合
    ransac_estimator.fit(X, y)

    # 断言 RANSAC 模型对 X[2:] 的预测分数为 1
    assert ransac_estimator.score(X[2:], y[2:]) == 1
    # 断言 RANSAC 模型对 X[:2] 的预测分数小于 1
    assert ransac_estimator.score(X[:2], y[:2]) < 1


# 定义一个测试函数，用于测试 RANSACRegressor 的 predict 方法
def test_ransac_predict():
    # 创建一个一维的数据集 X，100 个样本
    X = np.arange(100)[:, None]
    # 创建一个一维的标签 y，100 个样本，大部分为 0，但第一个和第二个样本分别为 1 和 100
    y = np.zeros((100,))
    y[0] = 1
    y[1] = 100

    # 创建一个线性回归模型作为基础评估器
    estimator = LinearRegression()
    # 创建 RANSACRegressor 实例，设置最小样本数为 2，残差阈值为 0.5，随机状态为 0
    ransac_estimator = RANSACRegressor(
        estimator, min_samples=2, residual_threshold=0.5, random_state=0
    )
    # 对数据集 X, y 进行拟合
    ransac_estimator.fit(X, y)

    # 断言 RANSAC 模型对 X 的预测结果与 np.zeros(100) 相等
    assert_array_equal(ransac_estimator.predict(X), np.zeros(100))


# 定义一个测试函数，用于测试 RANSACRegressor 在没有有效数据时的行为
def test_ransac_no_valid_data():
    # 定义一个函数，判断数据集是否有效，总是返回 False
    def is_data_valid(X, y):
        return False

    # 创建一个线性回归模型作为基础评估器
    estimator = LinearRegression()
    # 创建 RANSACRegressor 实例，设置数据有效性判断函数为 is_data_valid，最大试验次数为 5
    ransac_estimator = RANSACRegressor(
        estimator, is_data_valid=is_data_valid, max_trials=5
    )

    # 期望抛出 ValueError 异常，消息为 "RANSAC could not find a valid consensus set"
    msg = "RANSAC could not find a valid consensus set"
    with pytest.raises(ValueError, match=msg):
        # 尝试对数据集 X, y 进行拟合
        ransac_estimator.fit(X, y)
    # 断言 RANSAC 模型中无内点跳过次数为 0
    assert ransac_estimator.n_skips_no_inliers_ == 0
    # 断言 RANSAC 模型中无效数据跳过次数为 5
    assert ransac_estimator.n_skips_invalid_data_ == 5
    # 断言 RANSAC 模型中无效模型跳过次数为 0
    assert ransac_estimator.n_skips_invalid_model_ == 0


# 定义一个测试函数，用于测试 RANSACRegressor 在没有有效模型时的行为
def test_ransac_no_valid_model():
    # 定义一个函数，判断模型是否有效，总是返回 False
    def is_model_valid(estimator, X, y):
        return False

    # 创建一个线性回归模型作为基础评估器
    estimator = LinearRegression()
    # 创建 RANSACRegressor 实例，设置模型有效性判断函数为 is_model_valid，最大试验次数为 5
    ransac_estimator = RANSACRegressor(
        estimator, is_model_valid=is_model_valid, max_trials=5
    )

    # 期望抛出 ValueError 异常，消息为 "RANSAC could not find a valid consensus set"
    msg = "RANSAC could not find a valid consensus set"
    with pytest.raises(ValueError, match=msg):
        # 尝试对数据集 X, y 进行拟合
        ransac_estimator.fit(X, y)
    # 断言 RANSAC 模型中无内点跳过次数为 0
    assert ransac_estimator.n_skips_no_inliers_ == 0
    # 断言 RANSAC 模型中无效数据跳过次数为 0
    assert ransac_estimator.n_skips_invalid_data_ == 0
    # 断言 RANSAC 模型中无效模型跳过次数为 5
    assert ransac_estimator.n_skips_invalid_model_ == 5


# 定义一个测试函数，用于测试 RANSACRegressor 在超过最大跳过次数时的行为
def test_ransac_exceed_max_skips():
    # 定义一个函数，判断数据集是否有效，总是返回 False
    def is_data_valid(X, y):
        return False

    # 创建一个线性回归模型作为基础评估器
    estimator = LinearRegression()
    # 创建 RANSACRegressor 实例，设置数据有效性判断函数为 is_data_valid，最大试验次数为 5，最大跳过次数为 3
    ransac_estimator = RANSACRegressor(
        estimator, is_data_valid=is_data_valid, max_trials=5, max_skips=3
    )

    # 期望抛出 ValueError 异常，消息为 "RANSAC skipped more iterations than `max_skips`"
    msg = "RANSAC skipped more iterations than `max_skips`"
    with pytest.raises(ValueError, match=msg):
        # 尝试对数据集 X, y 进行拟合
        ransac_estimator.fit(X, y)
    # 断言 RANSAC 模型中无内点跳过
    # 断言：验证 RANSAC 估计器中无效模型跳过次数为 0
    assert ransac_estimator.n_skips_invalid_model_ == 0
# 定义一个测试函数，用于测试 RANSAC 在超过最大跳过次数时是否发出警告
def test_ransac_warn_exceed_max_skips():
    # 声明全局变量 cause_skip，用于控制数据是否有效的标志
    global cause_skip
    # 初始化 cause_skip 为 False
    cause_skip = False

    # 定义一个内部函数，用于检查数据是否有效
    def is_data_valid(X, y):
        # 引用全局变量 cause_skip
        global cause_skip
        # 如果 cause_skip 为 False，则设为 True 并返回 True，表示数据有效
        if not cause_skip:
            cause_skip = True
            return True
        else:
            # 如果 cause_skip 已经为 True，则返回 False，表示数据无效
            return False

    # 创建一个线性回归的估计器
    estimator = LinearRegression()
    # 创建一个 RANSAC 回归器，设置最大跳过次数为 3，最大试验次数为 5，并指定数据有效性检查函数
    ransac_estimator = RANSACRegressor(
        estimator, is_data_valid=is_data_valid, max_skips=3, max_trials=5
    )
    # 设置警告消息内容
    warning_message = (
        "RANSAC found a valid consensus set but exited "
        "early due to skipping more iterations than "
        "`max_skips`. See estimator attributes for "
        "diagnostics."
    )
    # 使用 pytest 的 warn 断言来捕获 ConvergenceWarning 类型的警告，并匹配指定的警告消息内容
    with pytest.warns(ConvergenceWarning, match=warning_message):
        # 对数据进行拟合
        ransac_estimator.fit(X, y)
    # 断言 RANSAC 回归器中没有跳过的样本数为 0
    assert ransac_estimator.n_skips_no_inliers_ == 0
    # 断言 RANSAC 回归器中无效数据的跳过次数为 4
    assert ransac_estimator.n_skips_invalid_data_ == 4
    # 断言 RANSAC 回归器中无效模型的跳过次数为 0
    assert ransac_estimator.n_skips_invalid_model_ == 0


# 使用 pytest 的参数化功能，对不同的稀疏数据容器进行测试
@pytest.mark.parametrize(
    "sparse_container", COO_CONTAINERS + CSR_CONTAINERS + CSC_CONTAINERS
)
def test_ransac_sparse(sparse_container):
    # 根据指定的稀疏数据容器类型创建稀疏矩阵 X_sparse
    X_sparse = sparse_container(X)

    # 创建一个线性回归的估计器
    estimator = LinearRegression()
    # 创建一个 RANSAC 回归器，设置最小样本数为 2，残差阈值为 5，随机种子为 0
    ransac_estimator = RANSACRegressor(
        estimator, min_samples=2, residual_threshold=5, random_state=0
    )
    # 使用稀疏矩阵进行拟合
    ransac_estimator.fit(X_sparse, y)

    # 创建参考的内点掩码，将异常值对应的掩码设置为 False
    ref_inlier_mask = np.ones_like(ransac_estimator.inlier_mask_).astype(np.bool_)
    ref_inlier_mask[outliers] = False

    # 断言 RANSAC 回归器中的内点掩码与参考内点掩码相等
    assert_array_equal(ransac_estimator.inlier_mask_, ref_inlier_mask)


# 测试当估计器为 None 时的 RANSAC 回归器行为
def test_ransac_none_estimator():
    # 创建一个线性回归的估计器
    estimator = LinearRegression()

    # 创建一个正常的 RANSAC 回归器，设置最小样本数为 2，残差阈值为 5，随机种子为 0
    ransac_estimator = RANSACRegressor(
        estimator, min_samples=2, residual_threshold=5, random_state=0
    )
    # 创建一个使用 None 作为估计器的 RANSAC 回归器，设置最小样本数为 2，残差阈值为 5，随机种子为 0
    ransac_none_estimator = RANSACRegressor(
        None, min_samples=2, residual_threshold=5, random_state=0
    )

    # 对正常的 RANSAC 回归器进行拟合
    ransac_estimator.fit(X, y)
    # 对使用 None 作为估计器的 RANSAC 回归器进行拟合
    ransac_none_estimator.fit(X, y)

    # 断言两个 RANSAC 回归器预测结果的数组几乎相等
    assert_array_almost_equal(
        ransac_estimator.predict(X), ransac_none_estimator.predict(X)
    )


# 测试不同设置下的 RANSAC 回归器行为
def test_ransac_min_n_samples():
    # 创建一个线性回归的估计器
    estimator = LinearRegression()

    # 创建多个不同设置的 RANSAC 回归器，并分别进行拟合
    ransac_estimator1 = RANSACRegressor(
        estimator, min_samples=2, residual_threshold=5, random_state=0
    )
    ransac_estimator2 = RANSACRegressor(
        estimator,
        min_samples=2.0 / X.shape[0],
        residual_threshold=5,
        random_state=0,
    )
    ransac_estimator5 = RANSACRegressor(
        estimator, min_samples=2, residual_threshold=5, random_state=0
    )
    ransac_estimator6 = RANSACRegressor(estimator, residual_threshold=5, random_state=0)
    ransac_estimator7 = RANSACRegressor(
        estimator, min_samples=X.shape[0] + 1, residual_threshold=5, random_state=0
    )
    # GH #19390
    ransac_estimator8 = RANSACRegressor(
        Ridge(), min_samples=None, residual_threshold=5, random_state=0
    )

    # 分别对这些 RANSAC 回归器进行拟合
    ransac_estimator1.fit(X, y)
    ransac_estimator2.fit(X, y)
    ransac_estimator5.fit(X, y)
    ransac_estimator6.fit(X, y)

    # 断言不同设置下的 RANSAC 回归器预测结果的数组几乎相等
    assert_array_almost_equal(
        ransac_estimator1.predict(X), ransac_estimator2.predict(X)
    )
    # 使用 NumPy 的函数来比较 RANSAC 估计器的预测结果，断言它们几乎相等
    assert_array_almost_equal(
        ransac_estimator1.predict(X), ransac_estimator5.predict(X)
    )
    
    # 使用 NumPy 的函数来比较 RANSAC 估计器的预测结果，断言它们几乎相等
    assert_array_almost_equal(
        ransac_estimator1.predict(X), ransac_estimator6.predict(X)
    )
    
    # 使用 pytest 来检查是否会抛出 ValueError 异常，预期代码段会抛出异常
    with pytest.raises(ValueError):
        ransac_estimator7.fit(X, y)
    
    # 使用 pytest 来检查是否会抛出 ValueError 异常，并且异常信息需要与指定的错误消息匹配
    err_msg = "`min_samples` needs to be explicitly set"
    with pytest.raises(ValueError, match=err_msg):
        ransac_estimator8.fit(X, y)
def test_ransac_multi_dimensional_targets():
    # 创建一个线性回归估计器
    estimator = LinearRegression()
    # 创建 RANSAC 回归器，使用上述估计器，最小样本数为2，残差阈值为5，随机种子为0
    ransac_estimator = RANSACRegressor(
        estimator, min_samples=2, residual_threshold=5, random_state=0
    )

    # 将目标值 y 按列堆叠成 3 维数组
    yyy = np.column_stack([y, y, y])

    # 使用 RANSAC 估计器拟合数据 X 和多维目标值 yyy 的参数
    ransac_estimator.fit(X, yyy)

    # 参考值：地面真实的内点掩码
    ref_inlier_mask = np.ones_like(ransac_estimator.inlier_mask_).astype(np.bool_)
    ref_inlier_mask[outliers] = False

    # 断言 RANSAC 估计器的内点掩码与参考内点掩码相等
    assert_array_equal(ransac_estimator.inlier_mask_, ref_inlier_mask)


def test_ransac_residual_loss():
    # 定义多维损失函数，计算真实值与预测值之间的绝对值之和
    def loss_multi1(y_true, y_pred):
        return np.sum(np.abs(y_true - y_pred), axis=1)

    # 定义多维损失函数，计算真实值与预测值之间的平方和
    def loss_multi2(y_true, y_pred):
        return np.sum((y_true - y_pred) ** 2, axis=1)

    # 定义单维损失函数，计算真实值与预测值之间的绝对差
    def loss_mono(y_true, y_pred):
        return np.abs(y_true - y_pred)

    # 将目标值 y 按列堆叠成 3 维数组
    yyy = np.column_stack([y, y, y])

    # 创建线性回归估计器
    estimator = LinearRegression()
    # 创建 RANSAC 回归器，使用上述估计器，最小样本数为2，残差阈值为5，随机种子为0
    ransac_estimator0 = RANSACRegressor(
        estimator, min_samples=2, residual_threshold=5, random_state=0
    )
    # 创建带 loss_multi1 损失函数的 RANSAC 回归器
    ransac_estimator1 = RANSACRegressor(
        estimator,
        min_samples=2,
        residual_threshold=5,
        random_state=0,
        loss=loss_multi1,
    )
    # 创建带 loss_multi2 损失函数的 RANSAC 回归器
    ransac_estimator2 = RANSACRegressor(
        estimator,
        min_samples=2,
        residual_threshold=5,
        random_state=0,
        loss=loss_multi2,
    )

    # 多维情况下，分别使用不同的 RANSAC 回归器拟合数据 X 和目标值 yyy
    ransac_estimator0.fit(X, yyy)
    ransac_estimator1.fit(X, yyy)
    ransac_estimator2.fit(X, yyy)
    # 断言预测结果在多维情况下各自的一致性
    assert_array_almost_equal(
        ransac_estimator0.predict(X), ransac_estimator1.predict(X)
    )
    assert_array_almost_equal(
        ransac_estimator0.predict(X), ransac_estimator2.predict(X)
    )

    # 单维情况下，分别使用不同的 RANSAC 回归器拟合数据 X 和目标值 y
    ransac_estimator0.fit(X, y)
    # 修改 ransac_estimator2 的损失函数为 loss_mono
    ransac_estimator2.loss = loss_mono
    ransac_estimator2.fit(X, y)
    # 断言预测结果在单维情况下的一致性
    assert_array_almost_equal(
        ransac_estimator0.predict(X), ransac_estimator2.predict(X)
    )
    # 创建另一个 RANSAC 回归器，使用默认的 squared_error 损失函数
    ransac_estimator3 = RANSACRegressor(
        estimator,
        min_samples=2,
        residual_threshold=5,
        random_state=0,
        loss="squared_error",
    )
    ransac_estimator3.fit(X, y)
    # 断言预测结果在不同损失函数下的一致性
    assert_array_almost_equal(
        ransac_estimator0.predict(X), ransac_estimator2.predict(X)
    )


def test_ransac_default_residual_threshold():
    # 创建线性回归估计器
    estimator = LinearRegression()
    # 创建 RANSAC 回归器，使用上述估计器，最小样本数为2，随机种子为0
    ransac_estimator = RANSACRegressor(estimator, min_samples=2, random_state=0)

    # 使用 RANSAC 估计器拟合数据 X 和目标值 y
    ransac_estimator.fit(X, y)

    # 参考值：地面真实的内点掩码
    ref_inlier_mask = np.ones_like(ransac_estimator.inlier_mask_).astype(np.bool_)
    ref_inlier_mask[outliers] = False

    # 断言 RANSAC 估计器的内点掩码与参考内点掩码相等
    assert_array_equal(ransac_estimator.inlier_mask_, ref_inlier_mask)


def test_ransac_dynamic_max_trials():
    # 手工计算的数字，在《Multiple View Geometry in Computer Vision》第二版第119页表格4.3中确认
    # e = 0%, min_samples = X
    # 当误差率 e 为 0% 时，调用 _dynamic_max_trials 函数，期望返回值为 1
    assert _dynamic_max_trials(100, 100, 2, 0.99) == 1

    # e = 5%, min_samples = 2
    # 当误差率 e 为 5% 时，调用 _dynamic_max_trials 函数，期望返回值为 2
    assert _dynamic_max_trials(95, 100, 2, 0.99) == 2
    # e = 10%, min_samples = 2
    # 当误差率 e 为 10% 时，调用 _dynamic_max_trials 函数，期望返回值为 3
    assert _dynamic_max_trials(90, 100, 2, 0.99) == 3
    # e = 30%, min_samples = 2
    # 当误差率 e 为 30% 时，调用 _dynamic_max_trials 函数，期望返回值为 7
    assert _dynamic_max_trials(70, 100, 2, 0.99) == 7
    # e = 50%, min_samples = 2
    # 当误差率 e 为 50% 时，调用 _dynamic_max_trials 函数，期望返回值为 17
    assert _dynamic_max_trials(50, 100, 2, 0.99) == 17

    # e = 5%, min_samples = 8
    # 当误差率 e 为 5% 且最小样本数 min_samples 为 8 时，调用 _dynamic_max_trials 函数，期望返回值为 5
    assert _dynamic_max_trials(95, 100, 8, 0.99) == 5
    # e = 10%, min_samples = 8
    # 当误差率 e 为 10% 且最小样本数 min_samples 为 8 时，调用 _dynamic_max_trials 函数，期望返回值为 9
    assert _dynamic_max_trials(90, 100, 8, 0.99) == 9
    # e = 30%, min_samples = 8
    # 当误差率 e 为 30% 且最小样本数 min_samples 为 8 时，调用 _dynamic_max_trials 函数，期望返回值为 78
    assert _dynamic_max_trials(70, 100, 8, 0.99) == 78
    # e = 50%, min_samples = 8
    # 当误差率 e 为 50% 且最小样本数 min_samples 为 8 时，调用 _dynamic_max_trials 函数，期望返回值为 1177
    assert _dynamic_max_trials(50, 100, 8, 0.99) == 1177

    # e = 0%, min_samples = 10
    # 当误差率 e 为 0% 且最小样本数 min_samples 为 10 时，调用 _dynamic_max_trials 函数，期望返回值为 0
    assert _dynamic_max_trials(1, 100, 10, 0) == 0
    # 当误差率 e 为 1，即完全可信任，且最小样本数 min_samples 为 10 时，调用 _dynamic_max_trials 函数，期望返回值为正无穷
    assert _dynamic_max_trials(1, 100, 10, 1) == float("inf")
def test_ransac_fit_sample_weight():
    # 创建 RANSAC 回归估计器对象，设定随机状态为0
    ransac_estimator = RANSACRegressor(random_state=0)
    # 获取 y 的样本数
    n_samples = y.shape[0]
    # 创建权重数组，所有权重初始化为1
    weights = np.ones(n_samples)
    # 使用样本权重 weights 拟合 RANSAC 模型到数据集 X, y
    ransac_estimator.fit(X, y, sample_weight=weights)
    # 检查结果的一致性
    assert ransac_estimator.inlier_mask_.shape[0] == n_samples

    # 创建一个与 ransac_estimator.inlier_mask_ 大小相同的全为 True 的参考掩码
    ref_inlier_mask = np.ones_like(ransac_estimator.inlier_mask_).astype(np.bool_)
    # 将异常值处的掩码置为 False
    ref_inlier_mask[outliers] = False
    # 检查掩码的正确性
    assert_array_equal(ransac_estimator.inlier_mask_, ref_inlier_mask)

    # 检查 fit(X)  = fit([X1, X2, X3],sample_weight = [n1, n2, n3])，其中
    #   X = X1 重复 n1 次，X2 重复 n2 次，依此类推
    random_state = check_random_state(0)
    X_ = random_state.randint(0, 200, [10, 1])
    y_ = np.ndarray.flatten(0.2 * X_ + 2)
    sample_weight = random_state.randint(0, 10, 10)
    outlier_X = random_state.randint(0, 1000, [1, 1])
    outlier_weight = random_state.randint(0, 10, 1)
    outlier_y = random_state.randint(-1000, 0, 1)

    # 将 X_ 和 outlier_X 重复对应次数，组成 X_flat
    X_flat = np.append(
        np.repeat(X_, sample_weight, axis=0),
        np.repeat(outlier_X, outlier_weight, axis=0),
        axis=0,
    )
    # 将 y_ 和 outlier_y 重复对应次数，组成 y_flat
    y_flat = np.ndarray.flatten(
        np.append(
            np.repeat(y_, sample_weight, axis=0),
            np.repeat(outlier_y, outlier_weight, axis=0),
            axis=0,
        )
    )
    # 使用样本权重拟合 RANSAC 模型到扁平化的数据集 X_flat, y_flat
    ransac_estimator.fit(X_flat, y_flat)
    # 获取参考系数 ref_coef_
    ref_coef_ = ransac_estimator.estimator_.coef_

    # 将 outlier_weight 加入 sample_weight
    sample_weight = np.append(sample_weight, outlier_weight)
    # 将 outlier_X 加入 X_
    X_ = np.append(X_, outlier_X, axis=0)
    # 将 outlier_y 加入 y_
    y_ = np.append(y_, outlier_y)
    # 使用更新的样本权重拟合 RANSAC 模型到更新的数据集 X_, y_
    ransac_estimator.fit(X_, y_, sample_weight=sample_weight)

    # 检查拟合后的系数是否一致
    assert_allclose(ransac_estimator.estimator_.coef_, ref_coef_)

    # 检查如果 estimator.fit 不支持 sample_weight，是否会引发错误
    estimator = OrthogonalMatchingPursuit()
    ransac_estimator = RANSACRegressor(estimator, min_samples=10)

    err_msg = f"{estimator.__class__.__name__} does not support sample_weight."
    # 使用 pytest 检查是否会引发预期的 ValueError 错误
    with pytest.raises(ValueError, match=err_msg):
        ransac_estimator.fit(X, y, sample_weight=weights)


def test_ransac_final_model_fit_sample_weight():
    # 创建随机生成的回归数据集 X, y
    X, y = make_regression(n_samples=1000, random_state=10)
    # 生成随机数发生器 rng
    rng = check_random_state(42)
    # 随机生成样本权重 sample_weight，并标准化
    sample_weight = rng.randint(1, 4, size=y.shape[0])
    sample_weight = sample_weight / sample_weight.sum()
    # 创建 RANSAC 回归对象，设定随机状态为0
    ransac = RANSACRegressor(random_state=0)
    # 使用样本权重拟合 RANSAC 模型到数据集 X, y
    ransac.fit(X, y, sample_weight=sample_weight)

    # 创建线性回归模型对象 final_model
    final_model = LinearRegression()
    # 获取内点掩码 mask_samples
    mask_samples = ransac.inlier_mask_
    # 使用样本权重拟合最终模型到内点数据集 X[mask_samples], y[mask_samples]
    final_model.fit(
        X[mask_samples], y[mask_samples], sample_weight=sample_weight[mask_samples]
    )

    # 检查 RANSAC 模型估计器的系数与最终模型的系数是否接近
    assert_allclose(ransac.estimator_.coef_, final_model.coef_, atol=1e-12)


def test_perfect_horizontal_line():
    """检查是否可以拟合所有样本均为内点的情况下的线性模型。
    非回归测试:
    https://github.com/scikit-learn/scikit-learn/issues/19497
    """
    # 创建包含100个样本的单变量数组 X
    X = np.arange(100)[:, None]
    # 创建全零的目标值数组 y
    y = np.zeros((100,))

    # 创建线性回归模型对象 estimator
    estimator = LinearRegression()
    # 使用 RANSAC 算法创建一个回归器，使用给定的估计器作为基础模型，并设置随机种子为 0
    ransac_estimator = RANSACRegressor(estimator, random_state=0)
    # 使用给定的训练数据 X 和目标值 y 来拟合 RANSAC 回归器
    ransac_estimator.fit(X, y)
    
    # 断言检查：验证 RANSAC 回归器中基础模型的系数是否接近于 0.0
    assert_allclose(ransac_estimator.estimator_.coef_, 0.0)
    # 断言检查：验证 RANSAC 回归器中基础模型的截距是否接近于 0.0
    assert_allclose(ransac_estimator.estimator_.intercept_, 0.0)
```