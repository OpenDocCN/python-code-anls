# `D:\src\scipysrc\scikit-learn\sklearn\tests\test_naive_bayes.py`

```
import re  # 导入正则表达式模块
import warnings  # 导入警告模块

import numpy as np  # 导入NumPy库，并使用别名np
import pytest  # 导入pytest测试框架
from scipy.special import logsumexp  # 从SciPy库中导入logsumexp函数

from sklearn.datasets import load_digits, load_iris  # 导入加载数据集的函数
from sklearn.model_selection import cross_val_score, train_test_split  # 导入交叉验证和数据集分割函数
from sklearn.naive_bayes import (  # 导入朴素贝叶斯模型
    BernoulliNB,
    CategoricalNB,
    ComplementNB,
    GaussianNB,
    MultinomialNB,
)
from sklearn.utils._testing import (  # 导入用于测试的辅助函数
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)
from sklearn.utils.fixes import CSR_CONTAINERS  # 导入用于稀疏矩阵容器的修复函数

DISCRETE_NAIVE_BAYES_CLASSES = [BernoulliNB, CategoricalNB, ComplementNB, MultinomialNB]  # 定义离散型贝叶斯模型类列表
ALL_NAIVE_BAYES_CLASSES = DISCRETE_NAIVE_BAYES_CLASSES + [GaussianNB]  # 定义所有贝叶斯模型类列表，包括高斯朴素贝叶斯

msg = "The default value for `force_alpha` will change"  # 定义警告消息
pytestmark = pytest.mark.filterwarnings(f"ignore:{msg}:FutureWarning")  # 忽略特定警告消息的pytest标记

# Data is just 6 separable points in the plane
X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])  # 定义特征矩阵X，包含6个二维点
y = np.array([1, 1, 1, 2, 2, 2])  # 定义目标变量y，包含6个类标号


def get_random_normal_x_binary_y(global_random_seed):
    # A bit more random tests
    rng = np.random.RandomState(global_random_seed)  # 使用全局随机种子创建随机数生成器对象rng
    X1 = rng.normal(size=(10, 3))  # 生成一个10x3的正态分布随机数矩阵X1
    y1 = (rng.normal(size=10) > 0).astype(int)  # 生成一个10个元素的正态分布随机数向量，并转换为布尔类型后转为整数，作为y1
    return X1, y1  # 返回生成的随机数据


def get_random_integer_x_three_classes_y(global_random_seed):
    # Data is 6 random integer points in a 100 dimensional space classified to
    # three classes.
    rng = np.random.RandomState(global_random_seed)  # 使用全局随机种子创建随机数生成器对象rng
    X2 = rng.randint(5, size=(6, 100))  # 生成一个6x100的随机整数矩阵X2，元素范围在0到4之间
    y2 = np.array([1, 1, 2, 2, 3, 3])  # 创建一个包含6个类标号的目标向量y2
    return X2, y2  # 返回生成的随机数据


def test_gnb():
    # Gaussian Naive Bayes classification.
    # This checks that GaussianNB implements fit and predict and returns
    # correct values for a simple toy dataset.

    clf = GaussianNB()  # 创建高斯朴素贝叶斯分类器对象clf
    y_pred = clf.fit(X, y).predict(X)  # 使用X和y进行模型拟合并预测，得到预测结果y_pred
    assert_array_equal(y_pred, y)  # 断言预测结果y_pred与真实标签y相等

    y_pred_proba = clf.predict_proba(X)  # 获取预测概率
    y_pred_log_proba = clf.predict_log_proba(X)  # 获取预测对数概率
    assert_array_almost_equal(np.log(y_pred_proba), y_pred_log_proba, 8)  # 断言对数概率与预测概率的对数值近似相等，精度为8位小数

    # Test whether label mismatch between target y and classes raises
    # an Error
    # FIXME Remove this test once the more general partial_fit tests are merged
    with pytest.raises(
        ValueError, match="The target label.* in y do not exist in the initial classes"
    ):
        GaussianNB().partial_fit(X, y, classes=[0, 1])  # 使用部分拟合方法对数据进行拟合，预期引发值错误异常


def test_gnb_prior(global_random_seed):
    # Test whether class priors are properly set.
    clf = GaussianNB().fit(X, y)  # 使用X和y拟合高斯朴素贝叶斯分类器对象clf
    assert_array_almost_equal(np.array([3, 3]) / 6.0, clf.class_prior_, 8)  # 断言类先验概率是否正确设置
    X1, y1 = get_random_normal_x_binary_y(global_random_seed)  # 使用全局随机种子生成随机正态分布数据X1和二进制类标号y1
    clf = GaussianNB().fit(X1, y1)  # 使用X1和y1拟合高斯朴素贝叶斯分类器对象clf
    # Check that the class priors sum to 1
    assert_array_almost_equal(clf.class_prior_.sum(), 1)  # 断言类先验概率之和是否为1


def test_gnb_sample_weight(global_random_seed):
    """Test whether sample weights are properly used in GNB."""
    # Sample weights all being 1 should not change results
    sw = np.ones(6)  # 创建一个长度为6，元素值全为1的样本权重向量sw
    clf = GaussianNB().fit(X, y)  # 使用X和y拟合高斯朴素贝叶斯分类器对象clf
    clf_sw = GaussianNB().fit(X, y, sw)  # 使用带权重的X、y和sw拟合高斯朴素贝叶斯分类器对象clf_sw
    # 对比两个分类器的参数 theta_ 和 var_ 是否几乎相等
    assert_array_almost_equal(clf.theta_, clf_sw.theta_)
    assert_array_almost_equal(clf.var_, clf_sw.var_)

    # 使用一半的样本权重进行两次拟合，期望结果与全权重一次拟合相同
    rng = np.random.RandomState(global_random_seed)

    # 创建随机数生成器，种子为全局随机种子
    sw = rng.rand(y.shape[0])
    
    # 使用 GaussianNB 拟合数据 X, y，使用样本权重 sw
    clf1 = GaussianNB().fit(X, y, sample_weight=sw)
    
    # 使用部分拟合方法，使用一半样本权重 sw / 2
    clf2 = GaussianNB().partial_fit(X, y, classes=[1, 2], sample_weight=sw / 2)
    clf2.partial_fit(X, y, sample_weight=sw / 2)

    # 对比两个分类器 clf1 和 clf2 的参数 theta_ 和 var_ 是否几乎相等
    assert_array_almost_equal(clf1.theta_, clf2.theta_)
    assert_array_almost_equal(clf1.var_, clf2.var_)

    # 检查重复的样本条目和相应增加的样本权重是否会产生相同的结果
    # 从 X 中随机选择 20 个索引，计算相应的样本权重
    ind = rng.randint(0, X.shape[0], 20)
    sample_weight = np.bincount(ind, minlength=X.shape[0])

    # 使用部分数据 X[ind], y[ind] 拟合分类器 clf_dupl
    clf_dupl = GaussianNB().fit(X[ind], y[ind])
    
    # 使用完整样本权重 sample_weight 拟合分类器 clf_sw
    clf_sw = GaussianNB().fit(X, y, sample_weight)

    # 对比分类器 clf_dupl 和 clf_sw 的参数 theta_ 和 var_ 是否几乎相等
    assert_array_almost_equal(clf_dupl.theta_, clf_sw.theta_)
    assert_array_almost_equal(clf_dupl.var_, clf_sw.var_)

    # 非回归测试，用于验证 gh-24140 中存在单一类别时是否会出现除以零的情况
    sample_weight = (y == 1).astype(np.float64)
    
    # 使用样本权重 sample_weight 拟合分类器 clf
    clf = GaussianNB().fit(X, y, sample_weight=sample_weight)
# 测试贝叶斯高斯分类器是否会在负先验的情况下引发错误
def test_gnb_neg_priors():
    # 使用负数先验创建一个高斯朴素贝叶斯分类器对象
    clf = GaussianNB(priors=np.array([-1.0, 2.0]))

    # 定义错误信息
    msg = "Priors must be non-negative"
    # 使用 pytest 检测是否会引发 ValueError，并验证错误信息是否匹配
    with pytest.raises(ValueError, match=msg):
        clf.fit(X, y)


# 测试类先验是否被正确使用覆盖
def test_gnb_priors():
    # 创建一个使用指定先验概率的高斯朴素贝叶斯分类器对象，并进行拟合
    clf = GaussianNB(priors=np.array([0.3, 0.7])).fit(X, y)
    # 检查预测的概率是否准确
    assert_array_almost_equal(
        clf.predict_proba([[-0.1, -0.1]]),
        np.array([[0.825303662161683, 0.174696337838317]]),
        8,
    )
    # 检查类先验是否被正确设置
    assert_array_almost_equal(clf.class_prior_, np.array([0.3, 0.7]))


# 测试类先验之和是否正确验证
def test_gnb_priors_sum_isclose():
    # 创建一个示例数据集 X 和指定的类先验概率 priors
    X = np.array(
        [
            [-1, -1],
            [-2, -1],
            [-3, -2],
            [-4, -5],
            [-5, -4],
            [1, 1],
            [2, 1],
            [3, 2],
            [4, 4],
            [5, 5],
        ]
    )
    priors = np.array([0.08, 0.14, 0.03, 0.16, 0.11, 0.16, 0.07, 0.14, 0.11, 0.0])
    Y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # 创建一个使用指定类先验概率的高斯朴素贝叶斯分类器对象
    clf = GaussianNB(priors=priors)
    # 执行拟合操作，验证是否能成功进行拟合
    clf.fit(X, Y)


# 测试在先验数目与类数目不匹配时是否会引发错误
def test_gnb_wrong_nb_priors():
    # 创建一个使用不匹配类数目的先验概率数组的高斯朴素贝叶斯分类器对象
    clf = GaussianNB(priors=np.array([0.25, 0.25, 0.25, 0.25]))

    # 定义错误信息
    msg = "Number of priors must match number of classes"
    # 使用 pytest 检测是否会引发 ValueError，并验证错误信息是否匹配
    with pytest.raises(ValueError, match=msg):
        clf.fit(X, y)


# 测试当先验概率之和大于一时是否会引发错误
def test_gnb_prior_greater_one():
    # 创建一个使用总和大于一的先验概率数组的高斯朴素贝叶斯分类器对象
    clf = GaussianNB(priors=np.array([2.0, 1.0]))

    # 定义错误信息
    msg = "The sum of the priors should be 1"
    # 使用 pytest 检测是否会引发 ValueError，并验证错误信息是否匹配
    with pytest.raises(ValueError, match=msg):
        clf.fit(X, y)


# 测试当类先验明显偏向一个类时是否能得到良好的预测
def test_gnb_prior_large_bias():
    # 创建一个使用先验概率明显偏向某一类的高斯朴素贝叶斯分类器对象，并进行拟合
    clf = GaussianNB(priors=np.array([0.01, 0.99]))
    clf.fit(X, y)
    # 检查预测结果是否正确
    assert clf.predict([[-0.1, -0.1]]) == np.array([2])


# 测试当部分拟合调用时不带任何数据时是否能正常运行
def test_gnb_check_update_with_no_data():
    # 创建一个空数组和所需的初始值
    prev_points = 100
    mean = 0.0
    var = 1.0
    x_empty = np.empty((0, X.shape[1]))
    # 调用函数以更新均值和方差，验证是否保持不变
    tmean, tvar = GaussianNB._update_mean_variance(prev_points, mean, var, x_empty)
    assert tmean == mean
    assert tvar == var


# 测试部分拟合方法是否正常工作
def test_gnb_partial_fit():
    # 创建两个高斯朴素贝叶斯分类器对象，分别进行完全拟合和部分拟合
    clf = GaussianNB().fit(X, y)
    clf_pf = GaussianNB().partial_fit(X, y, np.unique(y))
    # 检查拟合后的参数是否一致
    assert_array_almost_equal(clf.theta_, clf_pf.theta_)
    assert_array_almost_equal(clf.var_, clf_pf.var_)
    assert_array_almost_equal(clf.class_prior_, clf_pf.class_prior_)

    # 进行另一轮部分拟合，验证拟合结果是否依然一致
    clf_pf2 = GaussianNB().partial_fit(X[0::2, :], y[0::2], np.unique(y))
    clf_pf2.partial_fit(X[1::2], y[1::2])
    assert_array_almost_equal(clf.theta_, clf_pf2.theta_)
    assert_array_almost_equal(clf.var_, clf_pf2.var_)
    # 断言：验证两个分类器的类先验概率数组几乎相等
    assert_array_almost_equal(clf.class_prior_, clf_pf2.class_prior_)
# 测试高斯朴素贝叶斯在数据缩放下的尺度不变性
def test_gnb_naive_bayes_scale_invariance():
    # 载入鸢尾花数据集
    iris = load_iris()
    X, y = iris.data, iris.target
    # 对于不同的数据缩放因子 f，使用高斯朴素贝叶斯进行拟合和预测
    labels = [GaussianNB().fit(f * X, y).predict(f * X) for f in [1e-10, 1, 1e10]]
    # 断言不同缩放因子下的预测结果应相同
    assert_array_equal(labels[0], labels[1])
    assert_array_equal(labels[1], labels[2])


# 使用参数化测试，测试离散型朴素贝叶斯模型的类先验设置
@pytest.mark.parametrize("DiscreteNaiveBayes", DISCRETE_NAIVE_BAYES_CLASSES)
def test_discretenb_prior(DiscreteNaiveBayes, global_random_seed):
    # 获取随机生成的三类整数特征和标签
    X2, y2 = get_random_integer_x_three_classes_y(global_random_seed)
    # 使用给定的离散型朴素贝叶斯模型进行拟合
    clf = DiscreteNaiveBayes().fit(X2, y2)
    # 断言类先验的对数概率向量是否正确设置
    assert_array_almost_equal(
        np.log(np.array([2, 2, 2]) / 6.0), clf.class_log_prior_, 8
    )


# 使用参数化测试，测试离散型朴素贝叶斯模型的部分拟合功能
@pytest.mark.parametrize("DiscreteNaiveBayes", DISCRETE_NAIVE_BAYES_CLASSES)
def test_discretenb_partial_fit(DiscreteNaiveBayes):
    # 创建两个离散型朴素贝叶斯分类器实例，分别进行完全拟合和部分拟合
    clf1 = DiscreteNaiveBayes()
    clf1.fit([[0, 1], [1, 0], [1, 1]], [0, 1, 1])

    clf2 = DiscreteNaiveBayes()
    clf2.partial_fit([[0, 1], [1, 0], [1, 1]], [0, 1, 1], classes=[0, 1])
    # 断言两个分类器的类计数是否相等
    assert_array_equal(clf1.class_count_, clf2.class_count_)
    # 对于 CategoricalNB，还需检查类别计数是否相等
    if DiscreteNaiveBayes is CategoricalNB:
        for i in range(len(clf1.category_count_)):
            assert_array_equal(clf1.category_count_[i], clf2.category_count_[i])
    else:
        # 对于其他模型，检查特征计数是否相等
        assert_array_equal(clf1.feature_count_, clf2.feature_count_)

    # 创建第三个离散型朴素贝叶斯分类器实例，进行多次部分拟合
    clf3 = DiscreteNaiveBayes()
    clf3.partial_fit([[0, 1]], [0], classes=[0, 1])
    clf3.partial_fit([[1, 0]], [1])
    clf3.partial_fit([[1, 1]], [1])
    # 断言三个分类器的类计数是否相等
    assert_array_equal(clf1.class_count_, clf3.class_count_)
    # 对于 CategoricalNB，还需检查类别计数矩阵的形状和总数是否相等
    if DiscreteNaiveBayes is CategoricalNB:
        for i in range(len(clf1.category_count_)):
            assert_array_equal(
                clf1.category_count_[i].shape, clf3.category_count_[i].shape
            )
            assert_array_equal(
                np.sum(clf1.category_count_[i], axis=1),
                np.sum(clf3.category_count_[i], axis=1),
            )
        # 对于特定类别，检查其在不同类别中出现的次数
        assert_array_equal(clf1.category_count_[0][0], np.array([1, 0]))
        assert_array_equal(clf1.category_count_[0][1], np.array([0, 2]))
        assert_array_equal(clf1.category_count_[1][0], np.array([0, 1]))
        assert_array_equal(clf1.category_count_[1][1], np.array([1, 1]))
    else:
        # 使用断言检查两个分类器的特征计数是否相等
        assert_array_equal(clf1.feature_count_, clf3.feature_count_)
@pytest.mark.parametrize("NaiveBayes", ALL_NAIVE_BAYES_CLASSES)
def test_NB_partial_fit_no_first_classes(NaiveBayes, global_random_seed):
    # 使用 pytest 的 parametrize 装饰器来多次运行此测试函数，参数为 ALL_NAIVE_BAYES_CLASSES 中的每个 NaiveBayes 类
    # 测试 NaiveBayes 模型在不提供 classes 参数的情况下是否会引发 ValueError 异常

    # 准备测试数据 X2, y2，从全局随机种子中获取
    X2, y2 = get_random_integer_x_three_classes_y(global_random_seed)

    # 验证在调用 partial_fit 的第一次时，必须提供 classes 参数，否则应该抛出 ValueError 异常
    with pytest.raises(
        ValueError, match="classes must be passed on the first call to partial_fit."
    ):
        NaiveBayes().partial_fit(X2, y2)

    # 检查连续调用 partial_fit 时 classes 参数的一致性
    clf = NaiveBayes()
    clf.partial_fit(X2, y2, classes=np.unique(y2))
    with pytest.raises(
        ValueError, match="is not the same as on last call to partial_fit"
    ):
        clf.partial_fit(X2, y2, classes=np.arange(42))


def test_discretenb_predict_proba():
    # 测试离散型朴素贝叶斯模型的类别概率预测

    # 下面的 100 区分伯努利和多项式分布。
    # FIXME: 编写一个测试来展示这一点。
    X_bernoulli = [[1, 100, 0], [0, 1, 0], [0, 100, 1]]
    X_multinomial = [[0, 1], [1, 3], [4, 0]]

    # 测试二元情况（输出为1维）
    y = [0, 0, 2]  # 2 是二元情况的回归测试，02e673
    for DiscreteNaiveBayes, X in zip(
        [BernoulliNB, MultinomialNB], [X_bernoulli, X_multinomial]
    ):
        clf = DiscreteNaiveBayes().fit(X, y)
        assert clf.predict(X[-1:]) == 2
        assert clf.predict_proba([X[0]]).shape == (1, 2)
        assert_array_almost_equal(
            clf.predict_proba(X[:2]).sum(axis=1), np.array([1.0, 1.0]), 6
        )

    # 测试多类情况（输出为2维，必须总和为1）
    y = [0, 1, 2]
    for DiscreteNaiveBayes, X in zip(
        [BernoulliNB, MultinomialNB], [X_bernoulli, X_multinomial]
    ):
        clf = DiscreteNaiveBayes().fit(X, y)
        assert clf.predict_proba(X[0:1]).shape == (1, 3)
        assert clf.predict_proba(X[:2]).shape == (2, 3)
        assert_almost_equal(np.sum(clf.predict_proba([X[1]])), 1)
        assert_almost_equal(np.sum(clf.predict_proba([X[-1]])), 1)
        assert_almost_equal(np.sum(np.exp(clf.class_log_prior_)), 1)


@pytest.mark.parametrize("DiscreteNaiveBayes", DISCRETE_NAIVE_BAYES_CLASSES)
def test_discretenb_uniform_prior(DiscreteNaiveBayes):
    # 测试离散型朴素贝叶斯模型在 fit_prior=False 且 class_prior=None 时是否拟合出均匀的先验概率分布

    clf = DiscreteNaiveBayes()
    clf.set_params(fit_prior=False)
    clf.fit([[0], [0], [1]], [0, 0, 1])
    prior = np.exp(clf.class_log_prior_)
    assert_array_almost_equal(prior, np.array([0.5, 0.5]))


@pytest.mark.parametrize("DiscreteNaiveBayes", DISCRETE_NAIVE_BAYES_CLASSES)
def test_discretenb_provide_prior(DiscreteNaiveBayes):
    # 测试离散型朴素贝叶斯模型是否能够使用提供的先验概率分布

    clf = DiscreteNaiveBayes(class_prior=[0.5, 0.5])
    clf.fit([[0], [0], [1]], [0, 0, 1])
    prior = np.exp(clf.class_log_prior_)
    assert_array_almost_equal(prior, np.array([0.5, 0.5]))

    # 当提供的先验概率数量与类别数量不一致时，应该抛出异常消息
    msg = "Number of priors must match number of classes"
    # 使用 pytest 框架来测试是否抛出 ValueError 异常，并验证异常消息是否与给定的 msg 变量匹配
    with pytest.raises(ValueError, match=msg):
        # 调用 clf 对象的 fit 方法，传入数据 [[0], [1], [2]] 和标签 [0, 1, 2]
        clf.fit([[0], [1], [2]], [0, 1, 2])
    
    # 设置变量 msg 为字符串 "is not the same as on last call to partial_fit"
    msg = "is not the same as on last call to partial_fit"
    # 使用 pytest 框架来测试是否抛出 ValueError 异常，并验证异常消息是否与给定的 msg 变量匹配
    with pytest.raises(ValueError, match=msg):
        # 调用 clf 对象的 partial_fit 方法，传入数据 [[0], [1]] 和标签 [0, 1]，以及 classes 参数为 [0, 1, 1]
        clf.partial_fit([[0], [1]], [0, 1], classes=[0, 1, 1])
@pytest.mark.parametrize("DiscreteNaiveBayes", DISCRETE_NAIVE_BAYES_CLASSES)
# 使用参数化测试，对不同的离散朴素贝叶斯类进行测试
def test_discretenb_provide_prior_with_partial_fit(DiscreteNaiveBayes):
    # 测试离散贝叶斯类在使用 partial_fit 方法时是否使用提供的先验信息

    iris = load_iris()
    iris_data1, iris_data2, iris_target1, iris_target2 = train_test_split(
        iris.data, iris.target, test_size=0.4, random_state=415
    )

    for prior in [None, [0.3, 0.3, 0.4]]:
        # 使用不同的先验信息初始化完整和部分拟合的离散贝叶斯分类器
        clf_full = DiscreteNaiveBayes(class_prior=prior)
        clf_full.fit(iris.data, iris.target)
        clf_partial = DiscreteNaiveBayes(class_prior=prior)
        clf_partial.partial_fit(iris_data1, iris_target1, classes=[0, 1, 2])
        clf_partial.partial_fit(iris_data2, iris_target2)
        # 检查完整拟合和部分拟合后的类别对数先验是否近似相等
        assert_array_almost_equal(
            clf_full.class_log_prior_, clf_partial.class_log_prior_
        )


@pytest.mark.parametrize("DiscreteNaiveBayes", DISCRETE_NAIVE_BAYES_CLASSES)
# 使用参数化测试，对不同的离散朴素贝叶斯类进行测试
def test_discretenb_sample_weight_multiclass(DiscreteNaiveBayes):
    # 检查在拟合时样本权重的形状一致性

    X = [
        [0, 0, 1],
        [0, 1, 1],
        [0, 1, 1],
        [1, 0, 0],
    ]
    y = [0, 0, 1, 2]
    sample_weight = np.array([1, 1, 2, 2], dtype=np.float64)
    sample_weight /= sample_weight.sum()
    # 使用样本权重拟合离散贝叶斯分类器，并验证预测结果
    clf = DiscreteNaiveBayes().fit(X, y, sample_weight=sample_weight)
    assert_array_equal(clf.predict(X), [0, 1, 1, 2])

    # 使用 partial_fit 方法检查样本权重
    clf = DiscreteNaiveBayes()
    clf.partial_fit(X[:2], y[:2], classes=[0, 1, 2], sample_weight=sample_weight[:2])
    clf.partial_fit(X[2:3], y[2:3], sample_weight=sample_weight[2:3])
    clf.partial_fit(X[3:], y[3:], sample_weight=sample_weight[3:])
    # 验证部分拟合后的预测结果是否正确
    assert_array_equal(clf.predict(X), [0, 1, 1, 2])


@pytest.mark.parametrize("DiscreteNaiveBayes", DISCRETE_NAIVE_BAYES_CLASSES)
@pytest.mark.parametrize("use_partial_fit", [False, True])
@pytest.mark.parametrize("train_on_single_class_y", [False, True])
def test_discretenb_degenerate_one_class_case(
    DiscreteNaiveBayes,
    use_partial_fit,
    train_on_single_class_y,
):
    # 大多数离散朴素贝叶斯分类器的数组属性应该具有与类别数相等的第一轴长度。
    # 例外包括：ComplementNB.feature_all_, CategoricalNB.n_categories_。
    # 在二元问题和训练集中只有一个类别的退化情况下，使用 `fit` 或 `partial_fit` 确认这一点。
    # 用于处理退化单类情况的非回归测试：
    # https://github.com/scikit-learn/scikit-learn/issues/18974

    X = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    y = [1, 1, 2]
    if train_on_single_class_y:
        X = X[:-1]
        y = y[:-1]
    classes = sorted(list(set(y)))
    num_classes = len(classes)

    clf = DiscreteNaiveBayes()
    if use_partial_fit:
        # 使用部分拟合方法拟合离散贝叶斯分类器
        clf.partial_fit(X, y, classes=classes)
    else:
        # 使用完整拟合方法拟合离散贝叶斯分类器
        clf.fit(X, y)
    # 验证预测结果是否正确
    assert clf.predict(X[:1]) == y[0]
    `
    # 检查属性的第一维长度是否符合预期
    attribute_names = [
        "classes_",           # 类别标签数组
        "class_count_",       # 每个类别的样本数量数组
        "class_log_prior_",   # 类别的对数先验概率数组
        "feature_count_",     # 每个类别中每个特征的样本数量数组
        "feature_log_prob_",  # 每个类别中每个特征的对数概率数组
    ]
    
    # 遍历属性名列表
    for attribute_name in attribute_names:
        # 获取分类器clf对象的对应属性值，若属性不存在则为None
        attribute = getattr(clf, attribute_name, None)
        # 如果属性为None，跳过继续下一个属性的检查
        if attribute is None:
            # CategoricalNB分类器没有feature_count_属性，因此跳过检查
            continue
        # 如果属性是numpy数组
        if isinstance(attribute, np.ndarray):
            # 断言数组的第一维长度是否等于类别的数量num_classes
            assert attribute.shape[0] == num_classes
        else:
            # 如果属性不是numpy数组，则认为是列表，如CategoricalNB的feature_log_prob_是数组的列表
            # 遍历列表中的每个数组元素
            for element in attribute:
                # 断言数组元素的第一维长度是否等于类别的数量num_classes
                assert element.shape[0] == num_classes
# 使用 pytest 模块的 parametrize 装饰器，定义两个参数化测试参数：kind 和 csr_container
@pytest.mark.parametrize("kind", ("dense", "sparse"))
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_mnnb(kind, global_random_seed, csr_container):
    # 测试多项式朴素贝叶斯分类器
    # 检查 MultinomialNB 是否实现了 fit 和 predict 方法，并对简单的玩具数据集返回正确的值

    # 调用函数生成随机整数型 X2 和三类别标签 y2
    X2, y2 = get_random_integer_x_three_classes_y(global_random_seed)

    # 根据 kind 参数选择 X 的类型（dense 或 sparse）
    if kind == "dense":
        X = X2
    elif kind == "sparse":
        X = csr_container(X2)

    # 检查对学习集的预测能力
    clf = MultinomialNB()

    # 检查是否在数据中传递了负值
    msg = "Negative values in data passed to"
    with pytest.raises(ValueError, match=msg):
        clf.fit(-X, y2)
    
    # 训练分类器并预测结果
    y_pred = clf.fit(X, y2).predict(X)

    # 检查预测结果是否与实际标签 y2 相等
    assert_array_equal(y_pred, y2)

    # 验证 np.log(clf.predict_proba(X)) 是否与 clf.predict_log_proba(X) 给出相同的结果
    y_pred_proba = clf.predict_proba(X)
    y_pred_log_proba = clf.predict_log_proba(X)
    assert_array_almost_equal(np.log(y_pred_proba), y_pred_log_proba, 8)

    # 检查增量拟合是否产生相同的结果
    clf2 = MultinomialNB()
    clf2.partial_fit(X[:2], y2[:2], classes=np.unique(y2))
    clf2.partial_fit(X[2:5], y2[2:5])
    clf2.partial_fit(X[5:], y2[5:])

    y_pred2 = clf2.predict(X)
    assert_array_equal(y_pred2, y2)

    y_pred_proba2 = clf2.predict_proba(X)
    y_pred_log_proba2 = clf2.predict_log_proba(X)
    assert_array_almost_equal(np.log(y_pred_proba2), y_pred_log_proba2, 8)
    assert_array_almost_equal(y_pred_proba2, y_pred_proba)
    assert_array_almost_equal(y_pred_log_proba2, y_pred_log_proba)

    # 对整体数据进行增量拟合应该与一次性拟合相同
    clf3 = MultinomialNB()
    clf3.partial_fit(X, y2, classes=np.unique(y2))

    y_pred3 = clf3.predict(X)
    assert_array_equal(y_pred3, y2)
    y_pred_proba3 = clf3.predict_proba(X)
    y_pred_log_proba3 = clf3.predict_log_proba(X)
    assert_array_almost_equal(np.log(y_pred_proba3), y_pred_log_proba3, 8)
    assert_array_almost_equal(y_pred_proba3, y_pred_proba)
    assert_array_almost_equal(y_pred_log_proba3, y_pred_log_proba)


def test_mnb_prior_unobserved_targets():
    # 测试未观察到的目标类别的先验平滑化

    # 创建玩具训练数据
    X = np.array([[0, 1], [1, 0]])
    y = np.array([0, 1])

    clf = MultinomialNB()

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)

        # 对部分数据进行拟合，设置类别为 [0, 1, 2]
        clf.partial_fit(X, y, classes=[0, 1, 2])

    assert clf.predict([[0, 1]]) == 0
    assert clf.predict([[1, 0]]) == 1
    assert clf.predict([[1, 1]]) == 0

    # 添加一个具有先前未观察到的类别的训练示例
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)

        clf.partial_fit([[1, 1]], [2])

    assert clf.predict([[0, 1]]) == 0
    assert clf.predict([[1, 0]]) == 1
    assert clf.predict([[1, 1]]) == 2
    # 测试 BernoulliNB 在 alpha=1.0 时是否与 Manning, Raghavan, and Schuetze 的《Introduction to Information Retrieval》书中的玩具示例给出的值相同：
    # https://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-model-1.html

    # 训练数据点如下：
    # Chinese Beijing Chinese (class: China)
    # Chinese Chinese Shanghai (class: China)
    # Chinese Macao (class: China)
    # Tokyo Japan Chinese (class: Japan)

    # 特征包括：Beijing, Chinese, Japan, Macao, Shanghai, 和 Tokyo
    X = np.array(
        [[1, 1, 0, 0, 0, 0], [0, 1, 0, 0, 1, 0], [0, 1, 0, 1, 0, 0], [0, 1, 1, 0, 0, 1]]
    )

    # 类别为 China (0) 和 Japan (1)
    Y = np.array([0, 0, 0, 1])

    # 使用 alpha=1.0 拟合 BernoulliNB 模型
    clf = BernoulliNB(alpha=1.0)
    clf.fit(X, Y)

    # 检查类别的先验概率是否正确
    class_prior = np.array([0.75, 0.25])
    assert_array_almost_equal(np.exp(clf.class_log_prior_), class_prior)

    # 检查特征概率是否正确
    feature_prob = np.array(
        [
            [0.4, 0.8, 0.2, 0.4, 0.4, 0.2],
            [1 / 3.0, 2 / 3.0, 2 / 3.0, 1 / 3.0, 1 / 3.0, 2 / 3.0],
        ]
    )
    assert_array_almost_equal(np.exp(clf.feature_log_prob_), feature_prob)

    # 测试数据点为：
    # Chinese Chinese Chinese Tokyo Japan
    X_test = np.array([[0, 1, 1, 0, 0, 1]])

    # 检查预测概率是否正确
    unnorm_predict_proba = np.array([[0.005183999999999999, 0.02194787379972565]])
    predict_proba = unnorm_predict_proba / np.sum(unnorm_predict_proba)
    assert_array_almost_equal(clf.predict_proba(X_test), predict_proba)
def test_bnb_feature_log_prob():
    # Test for issue #4268.
    # Tests that the feature log prob value computed by BernoulliNB when
    # alpha=1.0 is equal to the expression given in Manning, Raghavan,
    # and Schuetze's "Introduction to Information Retrieval" book:
    # http://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-model-1.html

    # 创建一个包含特征数据的 NumPy 数组 X
    X = np.array([[0, 0, 0], [1, 1, 0], [0, 1, 0], [1, 0, 1], [0, 1, 0]])

    # 创建一个包含类标签的 NumPy 数组 Y
    Y = np.array([0, 0, 1, 2, 2])

    # 使用 alpha=1.0 实例化 BernoulliNB 分类器对象 clf
    clf = BernoulliNB(alpha=1.0)

    # 使用 X 和 Y 进行拟合，训练 BernoulliNB 分类器
    clf.fit(X, Y)

    # 手动计算 (log) 概率的分子和分母，用于构成 P(特征存在 | 类别)
    num = np.log(clf.feature_count_ + 1.0)
    denom = np.tile(np.log(clf.class_count_ + 2.0), (X.shape[1], 1)).T

    # 检查手动估计值与分类器计算的特征 log 概率是否相匹配
    assert_array_almost_equal(clf.feature_log_prob_, (num - denom))


def test_cnb():
    # Tests ComplementNB when alpha=1.0 for the toy example in Manning,
    # Raghavan, and Schuetze's "Introduction to Information Retrieval" book:
    # https://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-model-1.html

    # 训练数据点如下：
    # Chinese Beijing Chinese (class: China)
    # Chinese Chinese Shanghai (class: China)
    # Chinese Macao (class: China)
    # Tokyo Japan Chinese (class: Japan)

    # 特征包括 Beijing, Chinese, Japan, Macao, Shanghai, 和 Tokyo.
    X = np.array(
        [[1, 1, 0, 0, 0, 0], [0, 1, 0, 0, 1, 0], [0, 1, 0, 1, 0, 0], [0, 1, 1, 0, 0, 1]]
    )

    # 类别标签为 China (0), Japan (1).
    Y = np.array([0, 0, 0, 1])

    # 检查权重是否正确。参见 Rennie et al. (2003) 的表 4 的步骤 4-6。
    theta = np.array(
        [
            [
                (0 + 1) / (3 + 6),
                (1 + 1) / (3 + 6),
                (1 + 1) / (3 + 6),
                (0 + 1) / (3 + 6),
                (0 + 1) / (3 + 6),
                (1 + 1) / (3 + 6),
            ],
            [
                (1 + 1) / (6 + 6),
                (3 + 1) / (6 + 6),
                (0 + 1) / (6 + 6),
                (1 + 1) / (6 + 6),
                (1 + 1) / (6 + 6),
                (0 + 1) / (6 + 6),
            ],
        ]
    )

    # 创建全零的权重和归一化后的权重数组
    weights = np.zeros(theta.shape)
    normed_weights = np.zeros(theta.shape)

    # 计算权重和归一化后的权重
    for i in range(2):
        weights[i] = -np.log(theta[i])
        normed_weights[i] = weights[i] / weights[i].sum()

    # 实例化 ComplementNB 分类器对象 clf，使用 alpha=1.0
    clf = ComplementNB(alpha=1.0)

    # 验证输入数据非负
    msg = re.escape("Negative values in data passed to ComplementNB (input X)")
    with pytest.raises(ValueError, match=msg):
        clf.fit(-X, Y)

    # 使用 X 和 Y 进行拟合，训练 ComplementNB 分类器
    clf.fit(X, Y)

    # 检查特征计数和权重是否正确
    feature_count = np.array([[1, 3, 0, 1, 1, 0], [0, 1, 1, 0, 0, 1]])
    assert_array_equal(clf.feature_count_, feature_count)
    class_count = np.array([3, 1])
    assert_array_equal(clf.class_count_, class_count)
    feature_all = np.array([1, 4, 1, 1, 1, 1])
    # 断言：验证分类器的特征数组是否与预期数组相等
    assert_array_equal(clf.feature_all_, feature_all)
    
    # 断言：验证分类器的特征对数概率数组是否与预期权重数组几乎相等
    assert_array_almost_equal(clf.feature_log_prob_, weights)
    
    # 创建ComplementNB分类器对象，使用alpha=1.0进行补充朴素贝叶斯训练，启用归一化
    clf = ComplementNB(alpha=1.0, norm=True)
    clf.fit(X, Y)
    
    # 断言：验证分类器训练后的特征对数概率数组是否与归一化后的预期权重数组几乎相等
    assert_array_almost_equal(clf.feature_log_prob_, normed_weights)
def test_categoricalnb(global_random_seed):
    # Check the ability to predict the training set.
    # 创建一个CategoricalNB分类器的实例
    clf = CategoricalNB()
    # 调用函数获取随机生成的整数特征和三类标签
    X2, y2 = get_random_integer_x_three_classes_y(global_random_seed)

    # 在训练集上进行拟合和预测，并断言预测结果与真实标签相等
    y_pred = clf.fit(X2, y2).predict(X2)
    assert_array_equal(y_pred, y2)

    # 准备一个新的特征矩阵X3和标签y3，进行CategoricalNB的实例化
    X3 = np.array([[1, 4], [2, 5]])
    y3 = np.array([1, 2])
    clf = CategoricalNB(alpha=1, fit_prior=False)

    # 使用X3和y3进行拟合
    clf.fit(X3, y3)
    # 断言clf对象中存储的类别数目等于预期值
    assert_array_equal(clf.n_categories_, np.array([3, 6]))

    # 检查输入特征矩阵X包含负数时是否会引发异常
    X = np.array([[0, -1]])
    y = np.array([1])
    error_msg = re.escape("Negative values in data passed to CategoricalNB (input X)")
    # 使用pytest检查是否会引发值错误，并匹配预期的错误消息
    with pytest.raises(ValueError, match=error_msg):
        clf.predict(X)
    with pytest.raises(ValueError, match=error_msg):
        clf.fit(X, y)

    # 测试alpha参数的影响
    X3_test = np.array([[2, 5]])
    # alpha=1会将所有类别的计数增加1，从而影响最终概率的计算
    bayes_numerator = np.array([[1 / 3 * 1 / 3, 2 / 3 * 2 / 3]])
    bayes_denominator = bayes_numerator.sum()
    assert_array_almost_equal(
        clf.predict_proba(X3_test), bayes_numerator / bayes_denominator
    )

    # 断言category_count_属性统计了所有特征的类别数目
    assert len(clf.category_count_) == X3.shape[1]

    # 检查sample_weight参数的影响
    X = np.array([[0, 0], [0, 1], [0, 0], [1, 1]])
    y = np.array([1, 1, 2, 2])
    clf = CategoricalNB(alpha=1, fit_prior=False)
    clf.fit(X, y)
    assert_array_equal(clf.predict(np.array([[0, 0]])), np.array([1]))
    assert_array_equal(clf.n_categories_, np.array([2, 2]))

    for factor in [1.0, 0.3, 5, 0.0001]:
        X = np.array([[0, 0], [0, 1], [0, 0], [1, 1]])
        y = np.array([1, 1, 2, 2])
        sample_weight = np.array([1, 1, 10, 0.1]) * factor
        clf = CategoricalNB(alpha=1, fit_prior=False)
        clf.fit(X, y, sample_weight=sample_weight)
        assert_array_equal(clf.predict(np.array([[0, 0]])), np.array([2]))
        assert_array_equal(clf.n_categories_, np.array([2, 2]))


@pytest.mark.parametrize(
    "min_categories, exp_X1_count, exp_X2_count, new_X, exp_n_categories_",
    [
        # check min_categories with int > observed categories
        (
            3,
            np.array([[2, 0, 0], [1, 1, 0]]),
            np.array([[1, 1, 0], [1, 1, 0]]),
            np.array([[0, 2]]),
            np.array([3, 3]),
        ),
        # check with list input
        (
            [3, 4],
            np.array([[2, 0, 0], [1, 1, 0]]),
            np.array([[1, 1, 0, 0], [1, 1, 0, 0]]),
            np.array([[0, 3]]),
            np.array([3, 4]),
        ),
        # check min_categories with min less than actual
        (
            [
                1,
                np.array([[2, 0], [1, 1]]),
                np.array([[1, 1], [1, 1]]),
                np.array([[0, 1]]),
                np.array([2, 2]),
            ]
        ),
    ],
)
def test_categoricalnb_with_min_categories(
    # 定义变量 min_categories，exp_X1_count，exp_X2_count，new_X，exp_n_categories_
# Import necessary libraries and modules
import numpy as np
from sklearn.naive_bayes import CategoricalNB, BernoulliNB, MultinomialNB
from sklearn.utils.testing import assert_array_equal, assert_array_almost_equal
from scipy.sparse import csr_matrix
import pytest

# Test case for CategoricalNB with specific data
def test_categoricalnb_fit():
    # Define input data and expected results
    X_n_categories = np.array([[0, 0], [0, 1], [0, 0], [1, 1]])
    y_n_categories = np.array([1, 1, 2, 2])
    expected_prediction = np.array([1])

    # Initialize CategoricalNB classifier
    clf = CategoricalNB(alpha=1, fit_prior=False, min_categories=min_categories)
    
    # Fit the classifier with the input data
    clf.fit(X_n_categories, y_n_categories)
    
    # Retrieve category counts for X1 and X2
    X1_count, X2_count = clf.category_count_
    
    # Assert the category counts match expected values
    assert_array_equal(X1_count, exp_X1_count)
    assert_array_equal(X2_count, exp_X2_count)
    
    # Predict using new_X and assert the prediction matches expected_prediction
    predictions = clf.predict(new_X)
    assert_array_equal(predictions, expected_prediction)
    
    # Assert the number of categories in the classifier matches exp_n_categories_
    assert_array_equal(clf.n_categories_, exp_n_categories_)

# Parameterized test for handling errors with 'min_categories' input
@pytest.mark.parametrize(
    "min_categories, error_msg",
    [
        ([[3, 2], [2, 4]], "'min_categories' should have shape"),
    ],
)
def test_categoricalnb_min_categories_errors(min_categories, error_msg):
    # Define input data
    X = np.array([[0, 0], [0, 1], [0, 0], [1, 1]])
    y = np.array([1, 1, 2, 2])

    # Initialize CategoricalNB classifier with invalid 'min_categories'
    clf = CategoricalNB(alpha=1, fit_prior=False, min_categories=min_categories)
    
    # Check that ValueError is raised with the expected error message
    with pytest.raises(ValueError, match=error_msg):
        clf.fit(X, y)

# Test case for checking alpha parameter behavior in different Naive Bayes classifiers
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_alpha(csr_container):
    # Define input data
    X = np.array([[1, 0], [1, 1]])
    y = np.array([0, 1])
    
    # Warn message for alpha too small
    msg = "alpha too small will result in numeric errors, setting alpha = 1.0e-10"
    
    # Test BernoulliNB with alpha=0
    nb = BernoulliNB(alpha=0.0, force_alpha=False)
    with pytest.warns(UserWarning, match=msg):
        nb.partial_fit(X, y, classes=[0, 1])
    assert_array_almost_equal(nb.predict_proba(X), prob)

    # Test MultinomialNB with alpha=0
    nb = MultinomialNB(alpha=0.0, force_alpha=False)
    with pytest.warns(UserWarning, match=msg):
        nb.partial_fit(X, y, classes=[0, 1])
    assert_array_almost_equal(nb.predict_proba(X), prob)

    # Test CategoricalNB with alpha=0
    nb = CategoricalNB(alpha=0.0, force_alpha=False)
    with pytest.warns(UserWarning, match=msg):
        nb.fit(X, y)
    assert_array_almost_equal(nb.predict_proba(X), prob)

    # Test sparse X input for BernoulliNB
    X = csr_container(X)
    nb = BernoulliNB(alpha=0.0, force_alpha=False)
    with pytest.warns(UserWarning, match=msg):
        nb.fit(X, y)
    assert_array_almost_equal(nb.predict_proba(X), prob)

    # Test sparse X input for MultinomialNB
    nb = MultinomialNB(alpha=0.0, force_alpha=False)
    with pytest.warns(UserWarning, match=msg):
        nb.fit(X, y)
    assert_array_almost_equal(nb.predict_proba(X), prob)

# Test case for alpha parameter as a vector in MultinomialNB
def test_alpha_vector():
    # Define input data
    X = np.array([[1, 0], [1, 1]])
    y = np.array([0, 1])
    
    # Setting alpha as an array with same length as number of features
    alpha = np.array([1, 2])
    nb = MultinomialNB(alpha=alpha, force_alpha=False)
    
    # Partial fit the classifier with input data
    nb.partial_fit(X, y, classes=[0, 1])
    # 测试特征概率使用伪计数（alpha）
    feature_prob = np.array([[1 / 2, 1 / 2], [2 / 5, 3 / 5]])
    # 断言多项式朴素贝叶斯模型的特征对数概率与预期的对数概率是否几乎相等
    assert_array_almost_equal(nb.feature_log_prob_, np.log(feature_prob))

    # 测试预测
    prob = np.array([[5 / 9, 4 / 9], [25 / 49, 24 / 49]])
    # 断言多项式朴素贝叶斯模型对给定数据 X 的预测概率与预期的概率是否几乎相等
    assert_array_almost_equal(nb.predict_proba(X), prob)

    # 测试 alpha 是否非负
    alpha = np.array([1.0, -0.1])
    # 创建多项式朴素贝叶斯模型对象，并指定 alpha 参数，force_alpha 参数设为 False
    m_nb = MultinomialNB(alpha=alpha, force_alpha=False)
    expected_msg = "All values in alpha must be greater than 0."
    # 使用 pytest 断言，验证 fit 方法是否会引发 ValueError 异常，并检查异常消息是否与预期相匹配
    with pytest.raises(ValueError, match=expected_msg):
        m_nb.fit(X, y)

    # 测试过小的伪计数是否被替换
    ALPHA_MIN = 1e-10
    alpha = np.array([ALPHA_MIN / 2, 0.5])
    # 创建多项式朴素贝叶斯模型对象，并指定 alpha 参数，force_alpha 参数设为 False
    m_nb = MultinomialNB(alpha=alpha, force_alpha=False)
    # 部分拟合模型，使用给定的类别列表 [0, 1]
    m_nb.partial_fit(X, y, classes=[0, 1])
    # 断言检查 _check_alpha 方法返回的 alpha 数组是否与预期数组几乎相等，使用 decimal 参数精确到小数点后 12 位
    assert_array_almost_equal(m_nb._check_alpha(), [ALPHA_MIN, 0.5], decimal=12)

    # 测试正确的维度
    alpha = np.array([1.0, 2.0, 3.0])
    # 创建多项式朴素贝叶斯模型对象，并指定 alpha 参数，force_alpha 参数设为 False
    m_nb = MultinomialNB(alpha=alpha, force_alpha=False)
    expected_msg = "When alpha is an array, it should contains `n_features`"
    # 使用 pytest 断言，验证 fit 方法是否会引发 ValueError 异常，并检查异常消息是否与预期相匹配
    with pytest.raises(ValueError, match=expected_msg):
        m_nb.fit(X, y)
def test_check_accuracy_on_digits():
    # 非回归测试，确保对朴素贝叶斯模型的进一步重构/优化不会损害对稍微非线性可分数据集的性能
    X, y = load_digits(return_X_y=True)
    # 创建一个布尔数组，指示类标签是否为3或8
    binary_3v8 = np.logical_or(y == 3, y == 8)
    # 提取出类标签为3或8的样本和对应的类标签
    X_3v8, y_3v8 = X[binary_3v8], y[binary_3v8]

    # 多项式朴素贝叶斯
    scores = cross_val_score(MultinomialNB(alpha=10), X, y, cv=10)
    assert scores.mean() > 0.86

    scores = cross_val_score(MultinomialNB(alpha=10), X_3v8, y_3v8, cv=10)
    assert scores.mean() > 0.94

    # 伯努利朴素贝叶斯
    scores = cross_val_score(BernoulliNB(alpha=10), X > 4, y, cv=10)
    assert scores.mean() > 0.83

    scores = cross_val_score(BernoulliNB(alpha=10), X_3v8 > 4, y_3v8, cv=10)
    assert scores.mean() > 0.92

    # 高斯朴素贝叶斯
    scores = cross_val_score(GaussianNB(), X, y, cv=10)
    assert scores.mean() > 0.77

    scores = cross_val_score(GaussianNB(var_smoothing=0.1), X, y, cv=10)
    assert scores.mean() > 0.89

    scores = cross_val_score(GaussianNB(), X_3v8, y_3v8, cv=10)
    assert scores.mean() > 0.86


def test_check_alpha():
    """如果 alpha < _ALPHA_MIN 并且 force_alpha 为 True，则使用提供的 alpha 值。

    非回归测试，用于验证：
    https://github.com/scikit-learn/scikit-learn/issues/10772
    """
    _ALPHA_MIN = 1e-10
    b = BernoulliNB(alpha=0, force_alpha=True)
    # 断言 BernoulliNB 类的 _check_alpha 方法返回 0
    assert b._check_alpha() == 0

    alphas = np.array([0.0, 1.0])

    b = BernoulliNB(alpha=alphas, force_alpha=True)
    # 我们手动设置 `n_features_in_`，以避免 `_check_alpha` 报错
    b.n_features_in_ = alphas.shape[0]
    # 断言 BernoulliNB 类的 _check_alpha 方法返回预期的 alpha 数组
    assert_array_equal(b._check_alpha(), alphas)

    msg = (
        "alpha too small will result in numeric errors, setting alpha = %.1e"
        % _ALPHA_MIN
    )
    b = BernoulliNB(alpha=0, force_alpha=False)
    # 使用 pytest 的 warns 方法检查是否发出 UserWarning 并匹配特定消息
    with pytest.warns(UserWarning, match=msg):
        assert b._check_alpha() == _ALPHA_MIN

    b = BernoulliNB(alpha=0, force_alpha=False)
    # 再次使用 pytest 的 warns 方法检查是否发出 UserWarning 并匹配特定消息
    with pytest.warns(UserWarning, match=msg):
        assert b._check_alpha() == _ALPHA_MIN

    b = BernoulliNB(alpha=alphas, force_alpha=False)
    # 我们手动设置 `n_features_in_`，以避免 `_check_alpha` 报错
    b.n_features_in_ = alphas.shape[0]
    # 再次使用 pytest 的 warns 方法检查是否发出 UserWarning 并匹配特定消息
    with pytest.warns(UserWarning, match=msg):
        assert_array_equal(b._check_alpha(), np.array([_ALPHA_MIN, 1.0]))


@pytest.mark.parametrize("Estimator", ALL_NAIVE_BAYES_CLASSES)
def test_predict_joint_proba(Estimator, global_random_seed):
    # 获取一个随机整数特征和三类别标签的数据集
    X2, y2 = get_random_integer_x_three_classes_y(global_random_seed)
    # 使用 Estimator 训练模型
    est = Estimator().fit(X2, y2)
    # 预测联合对数概率
    jll = est.predict_joint_log_proba(X2)
    # 计算每个样本的对数概率和
    log_prob_x = logsumexp(jll, axis=1)
    # 计算条件对数概率
    log_prob_x_y = jll - np.atleast_2d(log_prob_x).T
    # 断言预测的对数概率与计算的条件对数概率接近
    assert_allclose(est.predict_log_proba(X2), log_prob_x_y)
```