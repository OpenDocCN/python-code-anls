# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\tests\test_huber.py`

```
# 导入必要的库和模块
import numpy as np
import pytest
from scipy import optimize

# 导入用于生成数据和模型的函数和类
from sklearn.datasets import make_regression
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge, SGDRegressor
from sklearn.linear_model._huber import _huber_loss_and_gradient
from sklearn.utils._testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)
from sklearn.utils.fixes import CSR_CONTAINERS

# 创建带有异常值的回归数据集的函数
def make_regression_with_outliers(n_samples=50, n_features=20):
    rng = np.random.RandomState(0)
    # 使用 make_regression 生成带有异常值的数据集，将 10% 的样本替换为噪声
    X, y = make_regression(
        n_samples=n_samples, n_features=n_features, random_state=0, noise=0.05
    )

    # 替换 10% 的样本为噪声
    num_noise = int(0.1 * n_samples)
    random_samples = rng.randint(0, n_samples, num_noise)
    X[random_samples, :] = 2.0 * rng.normal(0, 1, (num_noise, X.shape[1]))
    return X, y

# 测试 HuberRegressor 在大 epsilon 下与 LinearRegression 的匹配性
def test_huber_equals_lr_for_high_epsilon():
    # 生成带有异常值的回归数据集
    X, y = make_regression_with_outliers()
    
    # 使用 LinearRegression 拟合数据
    lr = LinearRegression()
    lr.fit(X, y)
    
    # 使用 HuberRegressor 拟合数据，设置大的 epsilon
    huber = HuberRegressor(epsilon=1e3, alpha=0.0)
    huber.fit(X, y)
    
    # 断言 HuberRegressor 的系数与 LinearRegression 的系数接近
    assert_almost_equal(huber.coef_, lr.coef_, 3)
    assert_almost_equal(huber.intercept_, lr.intercept_, 2)

# 测试 HuberRegressor 的最大迭代次数设置
def test_huber_max_iter():
    # 生成带有异常值的回归数据集
    X, y = make_regression_with_outliers()
    
    # 使用 HuberRegressor，设置最大迭代次数为 1
    huber = HuberRegressor(max_iter=1)
    huber.fit(X, y)
    
    # 断言实际迭代次数等于设定的最大迭代次数
    assert huber.n_iter_ == huber.max_iter

# 测试 _huber_loss_and_gradient 函数计算的梯度是否正确
def test_huber_gradient():
    rng = np.random.RandomState(1)
    # 生成带有异常值的回归数据集
    X, y = make_regression_with_outliers()
    
    # 生成样本权重
    sample_weight = rng.randint(1, 3, (y.shape[0]))

    def loss_func(x, *args):
        return _huber_loss_and_gradient(x, *args)[0]

    def grad_func(x, *args):
        return _huber_loss_and_gradient(x, *args)[1]

    # 使用 optimize.check_grad 检查梯度是否相等
    for _ in range(5):
        for n_features in [X.shape[1] + 1, X.shape[1] + 2]:
            w = rng.randn(n_features)
            w[-1] = np.abs(w[-1])
            grad_same = optimize.check_grad(
                loss_func, grad_func, w, X, y, 0.01, 0.1, sample_weight
            )
            # 断言梯度的差异小于给定的阈值
            assert_almost_equal(grad_same, 1e-6, 4)

# 使用参数化测试来测试 HuberRegressor 中的样本权重实现
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_huber_sample_weights(csr_container):
    # 生成带有异常值的回归数据集
    X, y = make_regression_with_outliers()
    
    # 使用 HuberRegressor 拟合数据
    huber = HuberRegressor()
    huber.fit(X, y)
    huber_coef = huber.coef_
    huber_intercept = huber.intercept_

    # 在使用 assert_array_almost_equal 比较之前，重新缩放系数
    # 确保比较的小数位数在一定程度上不敏感
    # 计算系数的振幅，用于数据的规模和正则化参数
    scale = max(np.mean(np.abs(huber.coef_)), np.mean(np.abs(huber.intercept_)))

    # 使用默认的样本权重拟合 HuberRegressor 模型
    huber.fit(X, y, sample_weight=np.ones(y.shape[0]))
    # 检查拟合后的系数是否在特定缩放下近似相等
    assert_array_almost_equal(huber.coef_ / scale, huber_coef / scale)
    assert_array_almost_equal(huber.intercept_ / scale, huber_intercept / scale)

    # 创建带有异常值的合成回归数据集
    X, y = make_regression_with_outliers(n_samples=5, n_features=20)
    # 创建一个新的输入数据集，包含重复的异常值
    X_new = np.vstack((X, np.vstack((X[1], X[1], X[3]))))
    # 创建对应的目标值，包含额外的异常值
    y_new = np.concatenate((y, [y[1]], [y[1]], [y[3]]))
    # 使用带异常值的数据集拟合 HuberRegressor 模型
    huber.fit(X_new, y_new)
    # 保存拟合后的系数和截距
    huber_coef = huber.coef_
    huber_intercept = huber.intercept_
    # 创建自定义样本权重向量，其中第1和第3个样本的权重分别为3和2
    sample_weight = np.ones(X.shape[0])
    sample_weight[1] = 3
    sample_weight[3] = 2
    # 使用自定义样本权重拟合 HuberRegressor 模型
    huber.fit(X, y, sample_weight=sample_weight)

    # 检查拟合后的系数是否在特定缩放下近似相等
    assert_array_almost_equal(huber.coef_ / scale, huber_coef / scale)
    assert_array_almost_equal(huber.intercept_ / scale, huber_intercept / scale)

    # 使用稀疏数据的实现测试带样本权重的 HuberRegressor
    X_csr = csr_container(X)
    huber_sparse = HuberRegressor()
    huber_sparse.fit(X_csr, y, sample_weight=sample_weight)
    # 检查稀疏数据拟合后的系数是否在特定缩放下近似相等
    assert_array_almost_equal(huber_sparse.coef_ / scale, huber_coef / scale)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
# 使用 pytest 的参数化装饰器，允许对 test_huber_sparse 函数进行多组参数化测试
def test_huber_sparse(csr_container):
    # 生成带异常值的回归数据集
    X, y = make_regression_with_outliers()
    # 创建 Huber 回归器对象
    huber = HuberRegressor(alpha=0.1)
    # 使用回归数据集拟合 Huber 回归器
    huber.fit(X, y)

    # 将原始特征矩阵 X 转换成稀疏矩阵格式
    X_csr = csr_container(X)
    # 创建另一个 Huber 回归器对象，用于稀疏矩阵格式的拟合
    huber_sparse = HuberRegressor(alpha=0.1)
    # 使用稀疏特征矩阵拟合 Huber 回归器
    huber_sparse.fit(X_csr, y)
    # 断言稀疏拟合与原始拟合的系数近似相等
    assert_array_almost_equal(huber_sparse.coef_, huber.coef_)
    # 断言稀疏拟合与原始拟合的异常值索引数组相等
    assert_array_equal(huber.outliers_, huber_sparse.outliers_)


def test_huber_scaling_invariant():
    # 测试异常值过滤在特征缩放下的不变性
    X, y = make_regression_with_outliers()
    # 创建不包含截距项的 Huber 回归器对象，设置 alpha 为 0.0
    huber = HuberRegressor(fit_intercept=False, alpha=0.0)
    # 使用回归数据集拟合 Huber 回归器
    huber.fit(X, y)
    # 获取第一次拟合的异常值掩码
    n_outliers_mask_1 = huber.outliers_
    # 断言第一次拟合的异常值掩码不全为 True
    assert not np.all(n_outliers_mask_1)

    # 使用原始数据的两倍进行拟合，获取第二次拟合的异常值掩码
    huber.fit(X, 2.0 * y)
    n_outliers_mask_2 = huber.outliers_
    # 断言第二次拟合的异常值掩码与第一次相同
    assert_array_equal(n_outliers_mask_2, n_outliers_mask_1)

    # 使用特征数据的两倍进行拟合，获取第三次拟合的异常值掩码
    huber.fit(2.0 * X, 2.0 * y)
    n_outliers_mask_3 = huber.outliers_
    # 断言第三次拟合的异常值掩码与第一次相同
    assert_array_equal(n_outliers_mask_3, n_outliers_mask_1)


def test_huber_and_sgd_same_results():
    # 测试 Huber 回归器与 SGD 回归器在相同参数下是否收敛到相同的系数

    # 生成带异常值的回归数据集，包含两个特征
    X, y = make_regression_with_outliers(n_samples=10, n_features=2)

    # 创建 Huber 回归器对象，不包含截距项，alpha 为 0.0，epsilon 为 1.35
    huber = HuberRegressor(fit_intercept=False, alpha=0.0, epsilon=1.35)
    # 使用回归数据集拟合 Huber 回归器
    huber.fit(X, y)
    # 将特征数据按照 scale_ 缩放系数进行缩放
    X_scale = X / huber.scale_
    y_scale = y / huber.scale_
    # 使用缩放后的数据拟合 Huber 回归器
    huber.fit(X_scale, y_scale)
    # 断言 scale_ 缩放系数近似为 1.0，精度为 0.001
    assert_almost_equal(huber.scale_, 1.0, 3)

    # 创建 SGD 回归器对象，loss 为 "huber"，设置了多个参数
    sgdreg = SGDRegressor(
        alpha=0.0,
        loss="huber",
        shuffle=True,
        random_state=0,
        max_iter=10000,
        fit_intercept=False,
        epsilon=1.35,
        tol=None,
    )
    # 使用缩放后的数据拟合 SGD 回归器
    sgdreg.fit(X_scale, y_scale)
    # 断言 Huber 回归器与 SGD 回归器的系数近似相等，精度为 1
    assert_array_almost_equal(huber.coef_, sgdreg.coef_, 1)


def test_huber_warm_start():
    # 测试 Huber 回归器的 warm_start 参数

    # 生成带异常值的回归数据集
    X, y = make_regression_with_outliers()
    # 创建 Huber 回归器对象，设置 alpha 为 1.0，max_iter 为 10000，启用 warm_start，tol 为 0.1
    huber_warm = HuberRegressor(alpha=1.0, max_iter=10000, warm_start=True, tol=1e-1)

    # 使用回归数据集拟合 Huber 回归器
    huber_warm.fit(X, y)
    # 复制当前的系数
    huber_warm_coef = huber_warm.coef_.copy()
    # 再次使用相同数据集拟合 Huber 回归器
    huber_warm.fit(X, y)

    # 断言两次拟合的系数几乎相等，精度为 1
    assert_array_almost_equal(huber_warm.coef_, huber_warm_coef, 1)

    # 断言迭代次数为 0
    assert huber_warm.n_iter_ == 0


def test_huber_better_r2_score():
    # 测试 Huber 回归器相比于普通回归器在非异常值上的 R^2 分数更好

    # 生成带异常值的回归数据集
    X, y = make_regression_with_outliers()
    # 创建 Huber 回归器对象，设置 alpha 为 0.01
    huber = HuberRegressor(alpha=0.01)
    # 使用回归数据集拟合 Huber 回归器
    huber.fit(X, y)
    # 计算线性损失
    linear_loss = np.dot(X, huber.coef_) + huber.intercept_ - y
    # 创建异常值掩码
    mask = np.abs(linear_loss) < huber.epsilon * huber.scale_
    # 计算在非异常值上的 R^2 分数
    huber_score = huber.score(X[mask], y[mask])
    huber_outlier_score = huber.score(X[~mask], y[~mask])

    # 使用 Ridge 回归器对象拟合回归数据集
    ridge = Ridge(alpha=0.01)
    ridge.fit(X, y)
    # 使用岭回归模型计算在数据集 X[mask] 和 y[mask] 上的得分
    ridge_score = ridge.score(X[mask], y[mask])
    # 使用岭回归模型计算在数据集 X[~mask] 和 y[~mask] 上的得分，这里的~mask表示逆向掩码（即非掩码）
    ridge_outlier_score = ridge.score(X[~mask], y[~mask])
    # 断言：Huber 回归模型的得分应该优于岭回归模型的得分
    assert huber_score > ridge_score

    # 断言：岭回归模型在异常值上的表现应该优于 Huber 回归模型在异常值上的表现
    assert ridge_outlier_score > huber_outlier_score
# 定义一个函数用于测试 HuberRegressor 在布尔数据下的行为
def test_huber_bool():
    # 创建一个回归测试数据集，包括 200 个样本，每个样本有两个特征，噪声为 4.0，随机种子为 0
    X, y = make_regression(n_samples=200, n_features=2, noise=4.0, random_state=0)
    # 将原始特征数据 X 转换为布尔类型（True/False），新的数据集命名为 X_bool
    X_bool = X > 0
    # 使用 HuberRegressor 拟合布尔类型的特征数据 X_bool 和目标数据 y
    HuberRegressor().fit(X_bool, y)
```