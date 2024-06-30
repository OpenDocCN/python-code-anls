# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\tests\test_sparse_coordinate_descent.py`

```
# 导入必要的库和模块
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose

# 导入 sklearn 中相关的模型和函数
from sklearn.datasets import make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV
from sklearn.utils._testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    create_memmap_backed_data,
    ignore_warnings,
)
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, LIL_CONTAINERS

# 定义测试函数：测试稀疏系数属性
def test_sparse_coef():
    # 创建 ElasticNet 模型实例
    clf = ElasticNet()
    # 设置模型的系数属性
    clf.coef_ = [1, 2, 3]

    # 断言稀疏系数属性确实是稀疏矩阵
    assert sp.issparse(clf.sparse_coef_)
    # 断言稀疏系数属性的密集表示与原系数相同
    assert clf.sparse_coef_.toarray().tolist()[0] == clf.coef_


# 使用参数化测试来测试 Lasso 模型处理零数据的能力
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_lasso_zero(csc_container):
    # 创建稀疏矩阵实例 X
    X = csc_container((3, 1))
    # 设置目标值 y
    y = [0, 0, 0]
    # 设置测试数据 T
    T = np.array([[1], [2], [3]])
    # 创建 Lasso 模型实例并拟合数据
    clf = Lasso().fit(X, y)
    # 预测测试数据 T 的结果
    pred = clf.predict(T)
    # 断言模型的系数为零
    assert_array_almost_equal(clf.coef_, [0])
    # 断言预测结果与预期的零向量相等
    assert_array_almost_equal(pred, [0, 0, 0])
    # 断言对偶间隙为零
    assert_almost_equal(clf.dual_gap_, 0)


# 使用参数化测试来测试 ElasticNet 模型处理列表输入数据的能力
@pytest.mark.parametrize("with_sample_weight", [True, False])
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_enet_toy_list_input(with_sample_weight, csc_container):
    # 创建列表输入数据 X
    X = np.array([[-1], [0], [1]])
    X = csc_container(X)
    # 创建目标值 Y
    Y = [-1, 0, 1]  # just a straight line
    # 创建测试数据 T
    T = np.array([[2], [3], [4]])  # test sample
    # 根据是否使用样本权重创建样本权重数组
    if with_sample_weight:
        sw = np.array([2.0, 2, 2])
    else:
        sw = None

    # 创建 alpha=0, l1_ratio=1.0 的 ElasticNet 模型实例
    clf = ElasticNet(alpha=0, l1_ratio=1.0)
    # 忽略关于 alpha=0 的警告，继续拟合模型
    ignore_warnings(clf.fit)(X, Y, sample_weight=sw)
    # 预测测试数据 T 的结果
    pred = clf.predict(T)
    # 断言模型的系数与预期的值相等
    assert_array_almost_equal(clf.coef_, [1])
    # 断言预测结果与预期的结果相等
    assert_array_almost_equal(pred, [2, 3, 4])
    # 断言对偶间隙为零
    assert_almost_equal(clf.dual_gap_, 0)

    # 创建 alpha=0.5, l1_ratio=0.3 的 ElasticNet 模型实例
    clf = ElasticNet(alpha=0.5, l1_ratio=0.3)
    clf.fit(X, Y, sample_weight=sw)
    pred = clf.predict(T)
    # 断言模型的系数与预期的值相等（精确到小数点后三位）
    assert_array_almost_equal(clf.coef_, [0.50819], decimal=3)
    # 断言预测结果与预期的结果相等（精确到小数点后三位）
    assert_array_almost_equal(pred, [1.0163, 1.5245, 2.0327], decimal=3)
    # 断言对偶间隙为零
    assert_almost_equal(clf.dual_gap_, 0)

    # 创建 alpha=0.5, l1_ratio=0.5 的 ElasticNet 模型实例
    clf = ElasticNet(alpha=0.5, l1_ratio=0.5)
    clf.fit(X, Y, sample_weight=sw)
    pred = clf.predict(T)
    # 断言模型的系数与预期的值相等（精确到小数点后三位）
    assert_array_almost_equal(clf.coef_, [0.45454], 3)
    # 断言预测结果与预期的结果相等（精确到小数点后三位）
    assert_array_almost_equal(pred, [0.9090, 1.3636, 1.8181], 3)
    # 断言对偶间隙为零
    assert_almost_equal(clf.dual_gap_, 0)


# 使用参数化测试来测试 ElasticNet 模型处理显式稀疏输入数据的能力
@pytest.mark.parametrize("lil_container", LIL_CONTAINERS)
def test_enet_toy_explicit_sparse_input(lil_container):
    # 测试 ElasticNet 模型对于不同 alpha 和 l1_ratio 的表现，使用稀疏输入 X
    f = ignore_warnings
    # 创建稀疏矩阵实例 X
    X = lil_container((3, 1))
    X[0, 0] = -1
    # X[1, 0] = 0
    X[2, 0] = 1
    Y = [-1, 0, 1]  # Y represents the target values for linear regression (identity function)

    # test samples
    T = lil_container((3, 1))  # Create a container for test samples with shape (3, 1)
    T[0, 0] = 2  # Assign values to the first test sample
    T[1, 0] = 3  # Assign values to the second test sample
    T[2, 0] = 4  # Assign values to the third test sample

    # this should be the same as lasso
    clf = ElasticNet(alpha=0, l1_ratio=1.0)  # Initialize ElasticNet with no regularization (alpha=0) and pure L1 penalty (l1_ratio=1.0)
    f(clf.fit)(X, Y)  # Fit the classifier on training data X and target values Y using the identity function
    pred = clf.predict(T)  # Predict using the fitted model on test samples T
    assert_array_almost_equal(clf.coef_, [1])  # Assert that the coefficients are almost equal to [1]
    assert_array_almost_equal(pred, [2, 3, 4])  # Assert that predictions are almost equal to [2, 3, 4]
    assert_almost_equal(clf.dual_gap_, 0)  # Assert that the duality gap is almost zero

    clf = ElasticNet(alpha=0.5, l1_ratio=0.3)  # Initialize ElasticNet with alpha=0.5 and l1_ratio=0.3
    clf.fit(X, Y)  # Fit the classifier on training data X and target values Y
    pred = clf.predict(T)  # Predict using the fitted model on test samples T
    assert_array_almost_equal(clf.coef_, [0.50819], decimal=3)  # Assert that coefficients are almost equal to [0.50819]
    assert_array_almost_equal(pred, [1.0163, 1.5245, 2.0327], decimal=3)  # Assert that predictions are almost equal to [1.0163, 1.5245, 2.0327]
    assert_almost_equal(clf.dual_gap_, 0)  # Assert that the duality gap is almost zero

    clf = ElasticNet(alpha=0.5, l1_ratio=0.5)  # Initialize ElasticNet with alpha=0.5 and l1_ratio=0.5
    clf.fit(X, Y)  # Fit the classifier on training data X and target values Y
    pred = clf.predict(T)  # Predict using the fitted model on test samples T
    assert_array_almost_equal(clf.coef_, [0.45454], 3)  # Assert that coefficients are almost equal to [0.45454]
    assert_array_almost_equal(pred, [0.9090, 1.3636, 1.8181], 3)  # Assert that predictions are almost equal to [0.9090, 1.3636, 1.8181]
    assert_almost_equal(clf.dual_gap_, 0)  # Assert that the duality gap is almost zero
# 定义生成稀疏数据的函数，用于创建稀疏矩阵数据
def make_sparse_data(
    sparse_container,  # 稀疏矩阵容器，接受稀疏矩阵作为输入
    n_samples=100,  # 样本数，默认为100
    n_features=100,  # 特征数，默认为100
    n_informative=10,  # 信息特征数，默认为10
    seed=42,  # 随机种子，默认为42
    positive=False,  # 是否强制为正数，默认为False
    n_targets=1,  # 目标数，默认为1
):
    random_state = np.random.RandomState(seed)

    # 构建一个特征较多、样本较少的线性回归问题，模型存在问题

    # 生成一个基准模型
    w = random_state.randn(n_features, n_targets)
    w[n_informative:] = 0.0  # 只有前面几个特征对模型有影响
    if positive:
        w = np.abs(w)

    X = random_state.randn(n_samples, n_features)
    rnd = random_state.uniform(size=(n_samples, n_features))
    X[rnd > 0.5] = 0.0  # 输入信号中有50%的零值

    # 生成训练数据的真实标签
    y = np.dot(X, w)
    X = sparse_container(X)  # 调用稀疏矩阵容器处理输入矩阵
    if n_targets == 1:
        y = np.ravel(y)
    return X, y


# 使用pytest.mark.parametrize装饰器对稀疏弹性网络的测试进行参数化
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
@pytest.mark.parametrize(
    "alpha, fit_intercept, positive",
    [(0.1, False, False), (0.1, True, False), (1e-3, False, True), (1e-3, True, True)],
)
def test_sparse_enet_not_as_toy_dataset(csc_container, alpha, fit_intercept, positive):
    n_samples, n_features, max_iter = 100, 100, 1000
    n_informative = 10

    X, y = make_sparse_data(
        csc_container, n_samples, n_features, n_informative, positive=positive
    )

    X_train, X_test = X[n_samples // 2 :], X[: n_samples // 2]
    y_train, y_test = y[n_samples // 2 :], y[: n_samples // 2]

    # 初始化稀疏弹性网络分类器
    s_clf = ElasticNet(
        alpha=alpha,
        l1_ratio=0.8,
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        tol=1e-7,
        positive=positive,
        warm_start=True,
    )
    s_clf.fit(X_train, y_train)  # 使用训练数据拟合模型

    # 断言模型的对偶间隙（dual_gap）接近0，精确到小数点后4位
    assert_almost_equal(s_clf.dual_gap_, 0, 4)
    assert s_clf.score(X_test, y_test) > 0.85  # 断言模型在测试数据上的得分大于0.85

    # 检查稀疏弹性网络收敛性与稠密版本相同
    d_clf = ElasticNet(
        alpha=alpha,
        l1_ratio=0.8,
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        tol=1e-7,
        positive=positive,
        warm_start=True,
    )
    d_clf.fit(X_train.toarray(), y_train)  # 使用稠密的训练数据拟合模型

    assert_almost_equal(d_clf.dual_gap_, 0, 4)
    assert d_clf.score(X_test, y_test) > 0.85  # 断言稠密模型在测试数据上的得分大于0.85

    assert_almost_equal(s_clf.coef_, d_clf.coef_, 5)  # 断言稀疏和稠密模型系数接近，精确到小数点后5位
    assert_almost_equal(s_clf.intercept_, d_clf.intercept_, 5)  # 断言稀疏和稠密模型截距接近，精确到小数点后5位

    # 检查模型系数是否稀疏
    assert np.sum(s_clf.coef_ != 0.0) < 2 * n_informative


# 使用pytest.mark.parametrize装饰器对稀疏Lasso回归的测试进行参数化
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_sparse_lasso_not_as_toy_dataset(csc_container):
    n_samples = 100
    max_iter = 1000
    n_informative = 10
    X, y = make_sparse_data(
        csc_container, n_samples=n_samples, n_informative=n_informative
    )

    X_train, X_test = X[n_samples // 2 :], X[: n_samples // 2]
    y_train, y_test = y[n_samples // 2 :], y[: n_samples // 2]

    # 初始化稀疏Lasso回归模型
    s_clf = Lasso(alpha=0.1, fit_intercept=False, max_iter=max_iter, tol=1e-7)
    s_clf.fit(X_train, y_train)
    
    # 断言稀疏Lasso模型的对偶间隙（dual_gap）接近0，精确到小数点后4位
    assert_almost_equal(s_clf.dual_gap_, 0, 4)
    # 断言模型在测试集上的准确率大于0.85
    assert s_clf.score(X_test, y_test) > 0.85

    # 创建一个 Lasso 回归模型，设置 alpha=0.1，不计算截距，最大迭代次数为 max_iter，收敛阈值为1e-7
    d_clf = Lasso(alpha=0.1, fit_intercept=False, max_iter=max_iter, tol=1e-7)
    # 使用稀疏矩阵 X_train.toarray() 和 y_train 拟合模型
    d_clf.fit(X_train.toarray(), y_train)
    # 断言稀疏模型的对偶间隙（dual_gap_）接近于0，精度为小数点后4位
    assert_almost_equal(d_clf.dual_gap_, 0, 4)
    # 断言稀疏模型在测试集上的准确率大于0.85
    assert d_clf.score(X_test, y_test) > 0.85

    # 断言稀疏模型的系数是稀疏的，即非零系数的数量等于预设的信息量特征数量 n_informative
    assert np.sum(s_clf.coef_ != 0.0) == n_informative


这段代码主要用于测试稀疏模型的性能和特征选择效果，通过断言来验证模型在测试集上的准确率、对偶间隙是否接近于0以及系数是否稀疏。
# 使用 pytest 的 parametrize 装饰器来多次运行此测试函数，每次提供不同的 csc_container 参数
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_enet_multitarget(csc_container):
    # 指定目标变量的数量为 3，创建稀疏数据集 X 和目标变量 y
    n_targets = 3
    X, y = make_sparse_data(csc_container, n_targets=n_targets)

    # 创建 ElasticNet 回归模型的实例，设置 alpha=0.01，precompute=False
    estimator = ElasticNet(alpha=0.01, precompute=False)
    # XXX: 当 precompute 不为 False 时存在 bug！
    estimator.fit(X, y)
    # 获取拟合后的系数、截距和对偶间隙
    coef, intercept, dual_gap = (
        estimator.coef_,
        estimator.intercept_,
        estimator.dual_gap_,
    )

    # 对每个目标变量 k 进行单独的拟合和断言检查
    for k in range(n_targets):
        estimator.fit(X, y[:, k])
        assert_array_almost_equal(coef[k, :], estimator.coef_)
        assert_array_almost_equal(intercept[k], estimator.intercept_)
        assert_array_almost_equal(dual_gap[k], estimator.dual_gap_)


# 使用 pytest 的 parametrize 装饰器来多次运行此测试函数，每次提供不同的 csc_container 参数
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_path_parameters(csc_container):
    # 创建稀疏数据集 X 和目标变量 y
    X, y = make_sparse_data(csc_container)
    # 设置 max_iter=50, n_alphas=10，创建 ElasticNetCV 模型的实例
    max_iter = 50
    n_alphas = 10
    clf = ElasticNetCV(
        n_alphas=n_alphas,
        eps=1e-3,
        max_iter=max_iter,
        l1_ratio=0.5,
        fit_intercept=False,
    )
    # 忽略警告，拟合模型
    ignore_warnings(clf.fit)(X, y)  # new params
    # 断言 l1_ratio 被设置为 0.5
    assert_almost_equal(0.5, clf.l1_ratio)
    # 断言 n_alphas 的值与设置一致
    assert n_alphas == clf.n_alphas
    # 断言 alphas_ 的长度与 n_alphas 一致
    assert n_alphas == len(clf.alphas_)
    # 获取稀疏数据的 mse_path_
    sparse_mse_path = clf.mse_path_
    # 将 X 转换为密集数据后，再次拟合模型，比较 mse_path_
    ignore_warnings(clf.fit)(X.toarray(), y)  # compare with dense data
    assert_almost_equal(clf.mse_path_, sparse_mse_path)


# 使用 pytest 的 parametrize 装饰器来多次运行此测试函数，每次提供不同的 Model 和其他参数
@pytest.mark.parametrize("Model", [Lasso, ElasticNet, LassoCV, ElasticNetCV])
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("n_samples, n_features", [(24, 6), (6, 24)])
@pytest.mark.parametrize("with_sample_weight", [True, False])
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_sparse_dense_equality(
    Model, fit_intercept, n_samples, n_features, with_sample_weight, csc_container
):
    # 创建回归数据集 X 和目标变量 y
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        effective_rank=n_features // 2,
        n_informative=n_features // 2,
        bias=4 * fit_intercept,
        noise=1,
        random_state=42,
    )
    # 如果设置了样本权重，则生成样本权重向量 sw
    if with_sample_weight:
        sw = np.abs(np.random.RandomState(42).normal(scale=10, size=y.shape))
    else:
        sw = None
    # 将 X 转换为 csc_container 格式
    Xs = csc_container(X)
    # 设置模型参数
    params = {"fit_intercept": fit_intercept}
    # 创建 Model 模型的密集数据版本和稀疏数据版本，分别进行拟合
    reg_dense = Model(**params).fit(X, y, sample_weight=sw)
    reg_sparse = Model(**params).fit(Xs, y, sample_weight=sw)
    # 如果设置了 fit_intercept，则断言截距相等，并且通过权重平衡性断言预测值和目标变量的平均值相等
    if fit_intercept:
        assert reg_sparse.intercept_ == pytest.approx(reg_dense.intercept_)
        # balance property
        assert np.average(reg_sparse.predict(X), weights=sw) == pytest.approx(
            np.average(y, weights=sw)
        )
    # 断言稀疏数据和密集数据版本的系数数组相近
    assert_allclose(reg_sparse.coef_, reg_dense.coef_)


# 使用 pytest 的 parametrize 装饰器来多次运行此测试函数，每次提供不同的 csc_container 参数
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_same_output_sparse_dense_lasso_and_enet_cv(csc_container):
    # 创建稀疏数据集 X 和目标变量 y
    X, y = make_sparse_data(csc_container, n_samples=40, n_features=10)
    # 创建 ElasticNetCV 模型的实例
    clfs = ElasticNetCV(max_iter=100)
    # 在稀疏数据上拟合模型
    clfs.fit(X, y)
    # 创建另一个 ElasticNetCV 模型的实例，用于后续比较
    clfd = ElasticNetCV(max_iter=100)
    # 使用训练数据 X 和标签 y 来拟合岭回归模型
    clfd.fit(X.toarray(), y)
    # 断言验证交叉验证后的岭回归模型的超参数 alpha 和当前模型 clfd 的 alpha 值近似相等，精确到小数点后第7位
    assert_almost_equal(clfs.alpha_, clfd.alpha_, 7)
    # 断言验证交叉验证后的岭回归模型的截距 intercept 和当前模型 clfd 的截距值近似相等，精确到小数点后第7位
    assert_almost_equal(clfs.intercept_, clfd.intercept_, 7)
    # 断言验证交叉验证后的岭回归模型的均方误差路径 mse_path 和当前模型 clfd 的均方误差路径近似相等
    assert_array_almost_equal(clfs.mse_path_, clfd.mse_path_)
    # 断言验证交叉验证后的岭回归模型的 alpha 路径 alphas 和当前模型 clfd 的 alpha 路径近似相等
    assert_array_almost_equal(clfs.alphas_, clfd.alphas_)

    # 创建 LassoCV 模型 clfs，设定最大迭代次数为 100，交叉验证折数为 4
    clfs = LassoCV(max_iter=100, cv=4)
    # 使用训练数据 X 和标签 y 来拟合 LassoCV 模型
    clfs.fit(X, y)
    # 创建 LassoCV 模型 clfd，设定最大迭代次数为 100，交叉验证折数为 4
    clfd = LassoCV(max_iter=100, cv=4)
    # 使用训练数据 X 转换为稀疏矩阵的形式和标签 y 来拟合 LassoCV 模型
    clfd.fit(X.toarray(), y)
    # 断言验证交叉验证后的 LassoCV 模型的超参数 alpha 和当前模型 clfd 的 alpha 值近似相等，精确到小数点后第7位
    assert_almost_equal(clfs.alpha_, clfd.alpha_, 7)
    # 断言验证交叉验证后的 LassoCV 模型的截距 intercept 和当前模型 clfd 的截距值近似相等，精确到小数点后第7位
    assert_almost_equal(clfs.intercept_, clfd.intercept_, 7)
    # 断言验证交叉验证后的 LassoCV 模型的均方误差路径 mse_path 和当前模型 clfd 的均方误差路径近似相等
    assert_array_almost_equal(clfs.mse_path_, clfd.mse_path_)
    # 断言验证交叉验证后的 LassoCV 模型的 alpha 路径 alphas 和当前模型 clfd 的 alpha 路径近似相等
    assert_array_almost_equal(clfs.alphas_, clfd.alphas_)
# 使用参数化测试框架pytest.mark.parametrize，对COO_CONTAINERS中的每个容器执行以下测试函数
@pytest.mark.parametrize("coo_container", COO_CONTAINERS)
def test_same_multiple_output_sparse_dense(coo_container):
    # 创建ElasticNet回归模型实例
    l = ElasticNet()
    # 创建稠密特征矩阵X
    X = [
        [0, 1, 2, 3, 4],
        [0, 2, 5, 8, 11],
        [9, 10, 11, 12, 13],
        [10, 11, 12, 13, 14],
    ]
    # 创建稠密标签矩阵y
    y = [
        [1, 2, 3, 4, 5],
        [1, 3, 6, 9, 12],
        [10, 11, 12, 13, 14],
        [11, 12, 13, 14, 15],
    ]
    # 使用X和y训练ElasticNet模型
    l.fit(X, y)
    # 创建样本数据，并将其转换为NumPy数组并重塑为二维数组
    sample = np.array([1, 2, 3, 4, 5]).reshape(1, -1)
    # 使用训练后的模型预测稠密样本的结果
    predict_dense = l.predict(sample)

    # 创建另一个ElasticNet回归模型实例
    l_sp = ElasticNet()
    # 使用coo_container将稠密特征矩阵X转换为稀疏表示
    X_sp = coo_container(X)
    # 使用X_sp和y训练ElasticNet模型
    l_sp.fit(X_sp, y)
    # 使用coo_container将样本数据转换为稀疏表示
    sample_sparse = coo_container(sample)
    # 使用训练后的稀疏模型预测稀疏样本的结果
    predict_sparse = l_sp.predict(sample_sparse)

    # 断言稀疏预测结果与稠密预测结果的近似性
    assert_array_almost_equal(predict_sparse, predict_dense)


# 使用参数化测试框架pytest.mark.parametrize，对CSC_CONTAINERS中的每个容器执行以下测试函数
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_sparse_enet_coordinate_descent(csc_container):
    """Test that a warning is issued if model does not converge"""
    # 创建Lasso回归模型实例，设置最大迭代次数为2
    clf = Lasso(max_iter=2)
    n_samples = 5
    n_features = 2
    # 使用csc_container创建稀疏特征矩阵X，并将其乘以1e50
    X = csc_container((n_samples, n_features)) * 1e50
    y = np.ones(n_samples)
    # 设置警告消息内容
    warning_message = (
        "Objective did not converge. You might want "
        "to increase the number of iterations."
    )
    # 使用pytest.warns断言捕获到ConvergenceWarning警告，并匹配warning_message内容
    with pytest.warns(ConvergenceWarning, match=warning_message):
        # 使用X和y训练Lasso模型
        clf.fit(X, y)


# 使用参数化测试框架pytest.mark.parametrize，对copy_X参数为True和False执行以下测试函数
@pytest.mark.parametrize("copy_X", (True, False))
def test_sparse_read_only_buffer(copy_X):
    """Test that sparse coordinate descent works for read-only buffers"""
    # 创建随机数生成器
    rng = np.random.RandomState(0)

    # 创建ElasticNet回归模型实例，设置alpha参数、copy_X参数和随机数种子
    clf = ElasticNet(alpha=0.1, copy_X=copy_X, random_state=rng)
    # 使用稀疏矩阵生成器sp.random创建100x20的稀疏特征矩阵X，格式为"csc"，设置随机数种子
    X = sp.random(100, 20, format="csc", random_state=rng)

    # 将X.data设置为只读
    X.data = create_memmap_backed_data(X.data)

    # 创建随机目标值向量y
    y = rng.rand(100)
    # 使用X和y训练ElasticNet模型
    clf.fit(X, y)
```