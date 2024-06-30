# `D:\src\scipysrc\scikit-learn\sklearn\feature_selection\tests\test_rfe.py`

```
"""
Testing Recursive feature elimination
"""

# 从 operator 模块中导入 attrgetter 函数
from operator import attrgetter

# 导入必要的库和模块
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal

# 导入 sklearn 中的各种类和函数
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import CCA, PLSCanonical, PLSRegression
from sklearn.datasets import load_iris, make_classification, make_friedman1
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import get_scorer, make_scorer, zero_one_loss
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR, LinearSVR
from sklearn.utils import check_random_state
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.fixes import CSR_CONTAINERS

# 定义一个虚拟的分类器类，用于测试递归特征消除
class MockClassifier:
    """
    Dummy classifier to test recursive feature elimination
    """

    def __init__(self, foo_param=0):
        self.foo_param = foo_param

    # 训练分类器，确保输入的特征和标签长度相同，并设定 coef_ 属性为全为 1 的数组
    def fit(self, X, y):
        assert len(X) == len(y)
        self.coef_ = np.ones(X.shape[1], dtype=np.float64)
        return self

    # 预测函数，返回输入特征的行数
    def predict(self, T):
        return T.shape[0]

    # 预测概率与预测函数相同
    predict_proba = predict
    decision_function = predict
    transform = predict

    # 返回一个固定的分数 0.0
    def score(self, X=None, y=None):
        return 0.0

    # 返回类的参数
    def get_params(self, deep=True):
        return {"foo_param": self.foo_param}

    # 设置类的参数
    def set_params(self, **params):
        return self

    # 返回更多标签信息，允许 NaN 值
    def _more_tags(self):
        return {"allow_nan": True}


# 测试函数，验证递归特征消除的重要性
def test_rfe_features_importance():
    generator = check_random_state(0)
    iris = load_iris()
    # 添加一些不相关的特征。设置随机种子确保这些不相关特征始终不相关。
    X = np.c_[iris.data, generator.normal(size=(len(iris.data), 6))]
    y = iris.target

    # 使用随机森林分类器作为估计器，设定树的数量和深度
    clf = RandomForestClassifier(n_estimators=20, random_state=generator, max_depth=2)
    # 初始化递归特征消除对象，设定要选择的特征数和步长
    rfe = RFE(estimator=clf, n_features_to_select=4, step=0.1)
    # 在数据集上训练递归特征消除
    rfe.fit(X, y)
    # 检查排名的长度是否与特征数相同
    assert len(rfe.ranking_) == X.shape[1]

    # 使用支持向量机分类器作为估计器
    clf_svc = SVC(kernel="linear")
    rfe_svc = RFE(estimator=clf_svc, n_features_to_select=4, step=0.1)
    # 在数据集上训练递归特征消除
    rfe_svc.fit(X, y)

    # 检查支持向量是否相同
    assert_array_equal(rfe.get_support(), rfe_svc.get_support())


# 使用参数化测试函数，测试递归特征消除在不同 CSR 容器上的表现
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_rfe(csr_container):
    generator = check_random_state(0)
    iris = load_iris()
    # 添加一些不相关的特征。设置随机种子确保这些不相关特征始终不相关。
    X = np.c_[iris.data, generator.normal(size=(len(iris.data), 6))]
    X_sparse = csr_container(X)
    y = iris.target

    # 稠密模型
    clf = SVC(kernel="linear")
    # 使用递归特征消除（Recursive Feature Elimination，RFE）选择器，设定估计器为clf，选择保留4个特征，步长为0.1
    rfe = RFE(estimator=clf, n_features_to_select=4, step=0.1)
    # 在给定的特征和标签上拟合RFE选择器
    rfe.fit(X, y)
    # 使用RFE选择的特征转换原始特征集X
    X_r = rfe.transform(X)
    # 使用经过特征选择后的特征训练分类器clf
    clf.fit(X_r, y)
    # 断言RFE选择器的排名数量与原始特征集X的特征数相等
    assert len(rfe.ranking_) == X.shape[1]

    # 创建一个稀疏模型分类器
    clf_sparse = SVC(kernel="linear")
    # 使用稀疏模型分类器进行RFE特征选择，选择保留4个特征，步长为0.1
    rfe_sparse = RFE(estimator=clf_sparse, n_features_to_select=4, step=0.1)
    # 在稀疏输入特征集上拟合RFE选择器
    rfe_sparse.fit(X_sparse, y)
    # 使用RFE选择的特征转换稀疏输入特征集X_sparse
    X_r_sparse = rfe_sparse.transform(X_sparse)

    # 断言经过特征选择后的特征集X_r的形状与鸢尾花数据集iris.data的形状相等
    assert X_r.shape == iris.data.shape
    # 断言经过特征选择后的前10个样本X_r与鸢尾花数据集iris.data的前10个样本相近（几乎相等）
    assert_array_almost_equal(X_r[:10], iris.data[:10])

    # 断言RFE选择器预测的输出与分类器clf在鸢尾花数据集上的预测输出几乎相等
    assert_array_almost_equal(rfe.predict(X), clf.predict(iris.data))
    # 断言RFE选择器在给定特征集X上的评分与分类器clf在鸢尾花数据集上的评分相等
    assert rfe.score(X, y) == clf.score(iris.data, iris.target)
    # 断言经过特征选择后的特征集X_r与稀疏转换后的特征集X_r_sparse的稀疏表示几乎相等
    assert_array_almost_equal(X_r, X_r_sparse.toarray())
def test_RFE_fit_score_params():
    # 确保 RFE 将元数据传递给基础估计器的 fit 和 score 方法

    # 定义一个测试估计器类，继承自 BaseEstimator 和 ClassifierMixin
    class TestEstimator(BaseEstimator, ClassifierMixin):
        # 定义 fit 方法，接受输入 X, y 和可选参数 prop
        def fit(self, X, y, prop=None):
            # 如果 prop 为 None，则抛出 ValueError 异常
            if prop is None:
                raise ValueError("fit: prop cannot be None")
            # 使用线性核初始化 SVC 模型并拟合输入数据 X, y
            self.svc_ = SVC(kernel="linear").fit(X, y)
            # 将 coef_ 设置为拟合后的 SVC 的系数
            self.coef_ = self.svc_.coef_
            return self

        # 定义 score 方法，接受输入 X, y 和可选参数 prop
        def score(self, X, y, prop=None):
            # 如果 prop 为 None，则抛出 ValueError 异常
            if prop is None:
                raise ValueError("score: prop cannot be None")
            # 返回 SVC 模型在输入数据 X, y 上的评分结果
            return self.svc_.score(X, y)

    # 载入鸢尾花数据集 X, y
    X, y = load_iris(return_X_y=True)

    # 使用 pytest 检查是否能捕获到 ValueError 异常，且异常信息为 "fit: prop cannot be None"
    with pytest.raises(ValueError, match="fit: prop cannot be None"):
        RFE(estimator=TestEstimator()).fit(X, y)

    # 使用 pytest 检查是否能捕获到 ValueError 异常，且异常信息为 "score: prop cannot be None"
    with pytest.raises(ValueError, match="score: prop cannot be None"):
        RFE(estimator=TestEstimator()).fit(X, y, prop="foo").score(X, y)

    # 使用 RFE 进行特征选择，然后进行评分，传入 prop="foo"
    RFE(estimator=TestEstimator()).fit(X, y, prop="foo").score(X, y, prop="foo")


def test_rfe_percent_n_features():
    # 测试结果是否一致

    # 使用随机状态 0 初始化生成器
    generator = check_random_state(0)
    # 载入鸢尾花数据集
    iris = load_iris()

    # 添加一些无关特征，确保无关特征始终不相关
    X = np.c_[iris.data, generator.normal(size=(len(iris.data), 6))]
    y = iris.target

    # 数据集中有 10 个特征，我们选择 40% 的特征
    clf = SVC(kernel="linear")
    rfe_num = RFE(estimator=clf, n_features_to_select=4, step=0.1)
    rfe_num.fit(X, y)

    rfe_perc = RFE(estimator=clf, n_features_to_select=0.4, step=0.1)
    rfe_perc.fit(X, y)

    # 断言两种选择方式的特征排名是否一致
    assert_array_equal(rfe_perc.ranking_, rfe_num.ranking_)
    # 断言两种选择方式的支持向量是否一致
    assert_array_equal(rfe_perc.support_, rfe_num.support_)


def test_rfe_mockclassifier():
    # 测试 MockClassifier

    # 使用随机状态 0 初始化生成器
    generator = check_random_state(0)
    # 载入鸢尾花数据集
    iris = load_iris()

    # 添加一些无关特征，确保无关特征始终不相关
    X = np.c_[iris.data, generator.normal(size=(len(iris.data), 6))]
    y = iris.target

    # 使用 MockClassifier 进行拟合
    clf = MockClassifier()
    rfe = RFE(estimator=clf, n_features_to_select=4, step=0.1)
    rfe.fit(X, y)

    # 对 X 进行转换
    X_r = rfe.transform(X)

    # 使用 clf 拟合转换后的数据
    clf.fit(X_r, y)

    # 断言特征排名的长度是否与原始数据集的特征数相等
    assert len(rfe.ranking_) == X.shape[1]
    # 断言转换后的数据形状是否与原始数据集 X 的形状相等
    assert X_r.shape == iris.data.shape


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_rfecv(csr_container):
    # 测试 RFECV

    # 使用随机状态 0 初始化生成器
    generator = check_random_state(0)
    # 载入鸢尾花数据集
    iris = load_iris()

    # 添加一些无关特征，确保无关特征始终不相关
    X = np.c_[iris.data, generator.normal(size=(len(iris.data), 6))]
    y = list(iris.target)  # regression test: list should be supported

    # 使用 SVC 进行评估，步长为 1
    rfecv = RFECV(estimator=SVC(kernel="linear"), step=1)
    rfecv.fit(X, y)

    # 对于 rfecv.cv_results_ 中的每个键，断言其长度是否与 X 的特征数相等
    for key in rfecv.cv_results_.keys():
        assert len(rfecv.cv_results_[key]) == X.shape[1]
    # 确保 RFECV 对象的特征排名数与输入数据的特征数相同
    assert len(rfecv.ranking_) == X.shape[1]
    # 使用 RFECV 对象对输入数据 X 进行特征选择，得到特征选择后的数据 X_r
    X_r = rfecv.transform(X)

    # 断言特征选择后的数据 X_r 与鸢尾花数据集的原始数据相等
    # 说明所有噪声变量已被过滤掉
    assert_array_equal(X_r, iris.data)

    # 在稀疏矩阵上执行相同操作
    rfecv_sparse = RFECV(estimator=SVC(kernel="linear"), step=1)
    # 将稠密矩阵 X 转换为稀疏矩阵 X_sparse
    X_sparse = csr_container(X)
    # 使用稀疏矩阵 X_sparse 对 RFECV 进行拟合
    rfecv_sparse.fit(X_sparse, y)
    # 对稀疏矩阵 X_sparse 进行特征选择，得到特征选择后的稀疏矩阵 X_r_sparse
    X_r_sparse = rfecv_sparse.transform(X_sparse)
    # 断言稀疏矩阵 X_r_sparse 转换为稠密数组后与鸢尾花数据集的原始数据相等
    assert_array_equal(X_r_sparse.toarray(), iris.data)

    # 使用自定义损失函数进行测试
    scoring = make_scorer(zero_one_loss, greater_is_better=False)
    # 创建 RFECV 对象，使用自定义损失函数进行评分
    rfecv = RFECV(estimator=SVC(kernel="linear"), step=1, scoring=scoring)
    # 忽略警告并拟合 RFECV 对象
    ignore_warnings(rfecv.fit)(X, y)
    # 对输入数据 X 进行特征选择，得到特征选择后的数据 X_r
    X_r = rfecv.transform(X)
    # 断言特征选择后的数据 X_r 与鸢尾花数据集的原始数据相等
    assert_array_equal(X_r, iris.data)

    # 使用预设的评分器进行测试
    scorer = get_scorer("accuracy")
    # 创建 RFECV 对象，使用预设的评分器进行评分
    rfecv = RFECV(estimator=SVC(kernel="linear"), step=1, scoring=scorer)
    # 对输入数据 X 进行拟合
    rfecv.fit(X, y)
    # 对输入数据 X 进行特征选择，得到特征选择后的数据 X_r
    X_r = rfecv.transform(X)
    # 断言特征选择后的数据 X_r 与鸢尾花数据集的原始数据相等
    assert_array_equal(X_r, iris.data)

    # 测试 cv_results_ 的修复情况
    def test_scorer(estimator, X, y):
        return 1.0

    # 创建 RFECV 对象，使用自定义评分函数 test_scorer 进行评分
    rfecv = RFECV(estimator=SVC(kernel="linear"), step=1, scoring=test_scorer)
    # 对输入数据 X 进行拟合
    rfecv.fit(X, y)

    # 断言在交叉验证得分相等的情况下，RFECV 返回具有最大交叉验证得分的最少特征数
    assert rfecv.n_features_ == 1

    # 使用 step=2 进行相同的前两个测试
    rfecv = RFECV(estimator=SVC(kernel="linear"), step=2)
    # 对输入数据 X 进行拟合
    rfecv.fit(X, y)

    # 断言 cv_results_ 中各键对应的值的长度为 6
    for key in rfecv.cv_results_.keys():
        assert len(rfecv.cv_results_[key]) == 6

    # 确保 RFECV 对象的特征排名数与输入数据的特征数相同
    assert len(rfecv.ranking_) == X.shape[1]
    # 使用 RFECV 对象对输入数据 X 进行特征选择，得到特征选择后的数据 X_r
    X_r = rfecv.transform(X)
    # 断言特征选择后的数据 X_r 与鸢尾花数据集的原始数据相等
    assert_array_equal(X_r, iris.data)

    # 在稀疏矩阵上执行相同的操作
    rfecv_sparse = RFECV(estimator=SVC(kernel="linear"), step=2)
    # 将稠密矩阵 X 转换为稀疏矩阵 X_sparse
    X_sparse = csr_container(X)
    # 使用稀疏矩阵 X_sparse 对 RFECV 进行拟合
    rfecv_sparse.fit(X_sparse, y)
    # 对稀疏矩阵 X_sparse 进行特征选择，得到特征选择后的稀疏矩阵 X_r_sparse
    X_r_sparse = rfecv_sparse.transform(X_sparse)
    # 断言稀疏矩阵 X_r_sparse 转换为稠密数组后与鸢尾花数据集的原始数据相等
    assert_array_equal(X_r_sparse.toarray(), iris.data)

    # 验证 steps < 1 不会导致错误
    rfecv_sparse = RFECV(estimator=SVC(kernel="linear"), step=0.2)
    # 将稠密矩阵 X 转换为稀疏矩阵 X_sparse
    X_sparse = csr_container(X)
    # 使用稀疏矩阵 X_sparse 对 RFECV 进行拟合
    rfecv_sparse.fit(X_sparse, y)
    # 对稀疏矩阵 X_sparse 进行特征选择，得到特征选择后的稀疏矩阵 X_r_sparse
    X_r_sparse = rfecv_sparse.transform(X_sparse)
    # 断言稀疏矩阵 X_r_sparse 转换为稠密数组后与鸢尾花数据集的原始数据相等
    assert_array_equal(X_r_sparse.toarray(), iris.data)
# 定义一个测试函数，用于测试 RFECV 与 MockClassifier 的组合
def test_rfecv_mockclassifier():
    # 创建随机数生成器
    generator = check_random_state(0)
    # 载入鸢尾花数据集
    iris = load_iris()
    # 构建特征矩阵 X，包括原始特征和随机生成的6列数据
    X = np.c_[iris.data, generator.normal(size=(len(iris.data), 6))]
    # 构建目标向量 y，将 iris.target 转换为列表格式
    y = list(iris.target)  # 回归测试：应支持列表格式的 y

    # 使用 MockClassifier 作为评估器进行 RFECV 特征选择
    rfecv = RFECV(estimator=MockClassifier(), step=1)
    rfecv.fit(X, y)

    # 针对 rfecv.cv_results_ 的每个键进行断言，确保结果长度与特征数量相同
    for key in rfecv.cv_results_.keys():
        assert len(rfecv.cv_results_[key]) == X.shape[1]

    # 断言 rfecv.ranking_ 的长度与特征数量相同
    assert len(rfecv.ranking_) == X.shape[1]


# 定义测试函数，验证 RFECV 的 verbose 输出是否大于零
def test_rfecv_verbose_output():
    # 重定向 sys.stdout 到 StringIO 对象，以捕获输出
    import sys
    from io import StringIO

    sys.stdout = StringIO()

    # 创建随机数生成器
    generator = check_random_state(0)
    # 载入鸢尾花数据集
    iris = load_iris()
    # 构建特征矩阵 X，包括原始特征和随机生成的6列数据
    X = np.c_[iris.data, generator.normal(size=(len(iris.data), 6))]
    # 构建目标向量 y，将 iris.target 转换为列表格式
    y = list(iris.target)

    # 使用 SVC(kernel="linear") 作为评估器进行 RFECV 特征选择，verbose 设置为 1
    rfecv = RFECV(estimator=SVC(kernel="linear"), step=1, verbose=1)
    rfecv.fit(X, y)

    # 恢复标准输出，获取 verbose 输出
    verbose_output = sys.stdout
    verbose_output.seek(0)
    # 断言 verbose 输出的第一行长度大于零，表示有输出产生
    assert len(verbose_output.readline()) > 0


# 定义测试函数，验证 RFECV 的 cv_results_ 大小是否符合预期
def test_rfecv_cv_results_size(global_random_seed):
    # 创建随机数生成器
    generator = check_random_state(global_random_seed)
    # 载入鸢尾花数据集
    iris = load_iris()
    # 构建特征矩阵 X，包括原始特征和随机生成的6列数据
    X = np.c_[iris.data, generator.normal(size=(len(iris.data), 6))]
    # 构建目标向量 y，将 iris.target 转换为列表格式
    y = list(iris.target)  # 回归测试：应支持列表格式的 y

    # 非回归测试，验证不同 step 和 min_features_to_select 组合的 RFECV 特征选择结果
    for step, min_features_to_select in [[2, 1], [2, 2], [3, 3]]:
        rfecv = RFECV(
            estimator=MockClassifier(),
            step=step,
            min_features_to_select=min_features_to_select,
        )
        rfecv.fit(X, y)

        # 计算预期的 score_len
        score_len = np.ceil((X.shape[1] - min_features_to_select) / step) + 1

        # 针对 rfecv.cv_results_ 的每个键进行断言，确保结果长度与预期的 score_len 相同
        for key in rfecv.cv_results_.keys():
            assert len(rfecv.cv_results_[key]) == score_len

        # 断言 rfecv.ranking_ 的长度与特征数量相同
        assert len(rfecv.ranking_) == X.shape[1]
        # 断言 rfecv.n_features_ 大于等于 min_features_to_select
        assert rfecv.n_features_ >= min_features_to_select


# 定义测试函数，验证 RFE 使用 SVC(kernel="linear") 评估器的标签
def test_rfe_estimator_tags():
    # 创建 RFE 对象，使用 SVC(kernel="linear") 作为评估器
    rfe = RFE(SVC(kernel="linear"))
    # 断言 rfe._estimator_type 为 "classifier"
    assert rfe._estimator_type == "classifier"
    # 确保交叉验证是分层的，使用鸢尾花数据集进行测试
    iris = load_iris()
    score = cross_val_score(rfe, iris.data, iris.target)
    # 断言交叉验证的最小得分大于 0.7
    assert score.min() > 0.7


# 定义测试函数，验证 RFE 使用 SVR(kernel="linear") 评估器的最小步长
def test_rfe_min_step(global_random_seed):
    n_features = 10
    # 生成 Friedman1 数据集
    X, y = make_friedman1(
        n_samples=50, n_features=n_features, random_state=global_random_seed
    )
    n_samples, n_features = X.shape
    # 创建 SVR(kernel="linear") 评估器
    estimator = SVR(kernel="linear")

    # 测试当 floor(step * n_features) <= 0 时的 RFE 特征选择
    selector = RFE(estimator, step=0.01)
    sel = selector.fit(X, y)
    # 断言选定的特征数等于 n_features 的一半
    assert sel.support_.sum() == n_features // 2

    # 测试当 step 在 (0,1) 之间且 floor(step * n_features) > 0 时的 RFE 特征选择
    selector = RFE(estimator, step=0.20)
    sel = selector.fit(X, y)
    # 断言选定的特征数等于 n_features 的一半
    assert sel.support_.sum() == n_features // 2

    # 测试当 step 为整数时的 RFE 特征选择
    selector = RFE(estimator, step=5)
    sel = selector.fit(X, y)
    # 断言选定的特征数等于 n_features 的一半
    assert sel.support_.sum() == n_features // 2
def test_number_of_subsets_of_features(global_random_seed):
    # In RFE, 'number_of_subsets_of_features'
    # = the number of iterations in '_fit'
    # = max(ranking_)
    # = 1 + (n_features + step - n_features_to_select - 1) // step
    # After optimization #4534, this number
    # = 1 + np.ceil((n_features - n_features_to_select) / float(step))
    # This test case is to test their equivalence, refer to #4534 and #3824

    # 定义计算公式1的函数，计算迭代次数
    def formula1(n_features, n_features_to_select, step):
        return 1 + ((n_features + step - n_features_to_select - 1) // step)

    # 定义计算公式2的函数，计算迭代次数
    def formula2(n_features, n_features_to_select, step):
        return 1 + np.ceil((n_features - n_features_to_select) / float(step))

    # RFE测试
    # Case 1, n_features - n_features_to_select 可以被 step 整除
    # Case 2, n_features - n_features_to_select 不能被 step 整除
    n_features_list = [11, 11]
    n_features_to_select_list = [3, 3]
    step_list = [2, 3]
    for n_features, n_features_to_select, step in zip(
        n_features_list, n_features_to_select_list, step_list
    ):
        generator = check_random_state(global_random_seed)
        X = generator.normal(size=(100, n_features))
        y = generator.rand(100).round()
        rfe = RFE(
            estimator=SVC(kernel="linear"),
            n_features_to_select=n_features_to_select,
            step=step,
        )
        rfe.fit(X, y)
        # 这个数值也等于 ranking_ 的最大值
        assert np.max(rfe.ranking_) == formula1(n_features, n_features_to_select, step)
        assert np.max(rfe.ranking_) == formula2(n_features, n_features_to_select, step)

    # In RFECV, 'fit' calls 'RFE._fit'
    # 'number_of_subsets_of_features' of RFE
    # = the size of each score in 'cv_results_' of RFECV
    # = the number of iterations of the for loop before optimization #4534

    # RFECV测试, n_features_to_select = 1
    # Case 1, n_features - 1 可以被 step 整除
    # Case 2, n_features - 1 不能被 step 整除
    n_features_to_select = 1
    n_features_list = [11, 10]
    step_list = [2, 2]
    for n_features, step in zip(n_features_list, step_list):
        generator = check_random_state(global_random_seed)
        X = generator.normal(size=(100, n_features))
        y = generator.rand(100).round()
        rfecv = RFECV(estimator=SVC(kernel="linear"), step=step)
        rfecv.fit(X, y)

        # 遍历 rfecv.cv_results_ 的所有键
        for key in rfecv.cv_results_.keys():
            # 断言每个 score 在 'cv_results_' 中的长度等于 formula1 的计算结果
            assert len(rfecv.cv_results_[key]) == formula1(
                n_features, n_features_to_select, step
            )
            # 断言每个 score 在 'cv_results_' 中的长度等于 formula2 的计算结果
            assert len(rfecv.cv_results_[key]) == formula2(
                n_features, n_features_to_select, step
            )
    # 获取特征选择后的排名列表
    rfecv_ranking = rfecv.ranking_
    
    # 获取特征选择交叉验证结果的字典
    rfecv_cv_results_ = rfecv.cv_results_
    
    # 设置特征选择过程中并行运行的作业数为2
    rfecv.set_params(n_jobs=2)
    # 使用给定的特征矩阵 X 和目标向量 y 进行特征选择拟合
    rfecv.fit(X, y)
    
    # 断言特征选择后的排名列表与之前获取的 rfecv_ranking 相等
    assert_array_almost_equal(rfecv.ranking_, rfecv_ranking)
    
    # 断言特征选择交叉验证结果的键集合与之前获取的 rfecv_cv_results_ 的键集合相等
    assert rfecv_cv_results_.keys() == rfecv.cv_results_.keys()
    
    # 遍历特征选择交叉验证结果的键集合，断言每个键对应的值近似等于 rfecv.cv_results_ 中对应键的值
    for key in rfecv_cv_results_.keys():
        assert rfecv_cv_results_[key] == pytest.approx(rfecv.cv_results_[key])
    
    
    这段代码用于描述了一个特征选择的过程，其中 `rfecv` 是一个递归特征消除选择器的实例，通过交叉验证来评估特征的重要性，并使用并行计算来加速运算。
def test_rfe_cv_groups():
    # 使用随机状态生成器初始化
    generator = check_random_state(0)
    # 载入鸢尾花数据集
    iris = load_iris()
    # 定义组数
    number_groups = 4
    # 创建组标签，根据目标向量长度均匀分布的分割点向下取整
    groups = np.floor(np.linspace(0, number_groups, len(iris.target)))
    # 获取特征数据和二分类目标
    X = iris.data
    y = (iris.target > 0).astype(int)

    # 初始化 RFECV 对象
    est_groups = RFECV(
        estimator=RandomForestClassifier(random_state=generator),
        step=1,
        scoring="accuracy",
        cv=GroupKFold(n_splits=2),
    )
    # 在数据上拟合 RFECV 选择器
    est_groups.fit(X, y, groups=groups)
    # 断言所选特征数量大于零
    assert est_groups.n_features_ > 0


@pytest.mark.parametrize(
    "importance_getter", [attrgetter("regressor_.coef_"), "regressor_.coef_"]
)
@pytest.mark.parametrize("selector, expected_n_features", [(RFE, 5), (RFECV, 4)])
def test_rfe_wrapped_estimator(importance_getter, selector, expected_n_features):
    # 非回归测试，用于验证指定 GitHub 问题
    X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
    estimator = LinearSVR(random_state=0)

    # 构建带变换目标的回归估计器
    log_estimator = TransformedTargetRegressor(
        regressor=estimator, func=np.log, inverse_func=np.exp
    )

    # 初始化 RFE 或 RFECV 选择器
    selector = selector(log_estimator, importance_getter=importance_getter)
    # 在数据上拟合选择器
    sel = selector.fit(X, y)
    # 断言支持的特征数量等于预期的特征数量
    assert sel.support_.sum() == expected_n_features


@pytest.mark.parametrize(
    "importance_getter, err_type",
    [
        ("auto", ValueError),
        ("random", AttributeError),
        (lambda x: x.importance, AttributeError),
    ],
)
@pytest.mark.parametrize("Selector", [RFE, RFECV])
def test_rfe_importance_getter_validation(importance_getter, err_type, Selector):
    X, y = make_friedman1(n_samples=50, n_features=10, random_state=42)
    estimator = LinearSVR()
    log_estimator = TransformedTargetRegressor(
        regressor=estimator, func=np.log, inverse_func=np.exp
    )

    # 验证重要性获取器的有效性
    with pytest.raises(err_type):
        model = Selector(log_estimator, importance_getter=importance_getter)
        model.fit(X, y)


@pytest.mark.parametrize("cv", [None, 5])
def test_rfe_allow_nan_inf_in_x(cv):
    iris = load_iris()
    X = iris.data
    y = iris.target

    # 向 X 添加 NaN 和 Inf 值
    X[0][0] = np.nan
    X[0][1] = np.inf

    clf = MockClassifier()
    if cv is not None:
        rfe = RFECV(estimator=clf, cv=cv)
    else:
        rfe = RFE(estimator=clf)
    # 在数据上拟合 RFE 选择器
    rfe.fit(X, y)
    # 对数据进行转换
    rfe.transform(X)


def test_w_pipeline_2d_coef_():
    # 创建管道对象，包含标准化和逻辑回归
    pipeline = make_pipeline(StandardScaler(), LogisticRegression())

    # 载入鸢尾花数据集
    data, y = load_iris(return_X_y=True)
    # 初始化 RFE 选择器，选择特征数为 2
    sfm = RFE(
        pipeline,
        n_features_to_select=2,
        importance_getter="named_steps.logisticregression.coef_",
    )

    # 在数据上拟合 RFE 选择器
    sfm.fit(data, y)
    # 断言转换后数据的特征数量等于 2
    assert sfm.transform(data).shape[1] == 2


def test_rfecv_std_and_mean(global_random_seed):
    # 使用全局随机种子初始化生成器
    generator = check_random_state(global_random_seed)
    # 载入鸢尾花数据集并增加随机正态分布的特征
    iris = load_iris()
    X = np.c_[iris.data, generator.normal(size=(len(iris.data), 6))]
    y = iris.target

    # 初始化 RFECV 选择器，使用线性核的 SVC 作为估计器
    rfecv = RFECV(estimator=SVC(kernel="linear"))
    # 在数据上拟合 RFECV 选择器
    rfecv.fit(X, y)
    # 根据 RFECV 交叉验证结果的键值列表，筛选出包含 "split" 的键名
    split_keys = [key for key in rfecv.cv_results_.keys() if "split" in key]
    
    # 使用筛选出的键名从 RFECV 交叉验证结果中获取对应的值，形成数组
    cv_scores = np.asarray([rfecv.cv_results_[key] for key in split_keys])
    
    # 计算交叉验证分数的均值，沿着数组的轴0进行计算
    expected_mean = np.mean(cv_scores, axis=0)
    
    # 计算交叉验证分数的标准差，沿着数组的轴0进行计算
    expected_std = np.std(cv_scores, axis=0)
    
    # 使用 numpy.testing.assert_allclose 函数断言 RFECV 结果中的平均测试分数与预期均值的近似性
    assert_allclose(rfecv.cv_results_["mean_test_score"], expected_mean)
    
    # 使用 numpy.testing.assert_allclose 函数断言 RFECV 结果中的测试分数标准差与预期标准差的近似性
    assert_allclose(rfecv.cv_results_["std_test_score"], expected_std)
@pytest.mark.parametrize(
    ["min_features_to_select", "n_features", "step", "cv_results_n_features"],
    [  # 参数化测试用例，定义了多组输入参数和对应的期望输出
        [1, 4, 1, np.array([1, 2, 3, 4])],
        [1, 5, 1, np.array([1, 2, 3, 4, 5])],
        [1, 4, 2, np.array([1, 2, 4])],
        [1, 5, 2, np.array([1, 3, 5])],
        [1, 4, 3, np.array([1, 4])],
        [1, 5, 3, np.array([1, 2, 5])],
        [1, 4, 4, np.array([1, 4])],
        [1, 5, 4, np.array([1, 5])],
        [4, 4, 2, np.array([4])],
        [4, 5, 1, np.array([4, 5])],
        [4, 5, 2, np.array([4, 5])],
    ],
)
def test_rfecv_cv_results_n_features(
    min_features_to_select,
    n_features,
    step,
    cv_results_n_features,
):
    """Test case for verifying RFECV behavior.

    This test checks if the RFECV estimator correctly selects features based on
    the given parameters and compares the result with expected `cv_results_n_features`.
    """
    X, y = make_classification(
        n_samples=20, n_features=n_features, n_informative=n_features, n_redundant=0
    )
    # 创建 RFECV 对象，使用线性核的 SVC 作为基础评估器
    rfecv = RFECV(
        estimator=SVC(kernel="linear"),
        step=step,
        min_features_to_select=min_features_to_select,
    )
    rfecv.fit(X, y)
    # 断言 RFECV 的交叉验证结果中选出的特征数量符合预期
    assert_array_equal(rfecv.cv_results_["n_features"], cv_results_n_features)
    # 断言 RFECV 的所有交叉验证结果中的值的长度一致
    assert all(
        len(value) == len(rfecv.cv_results_["n_features"])
        for value in rfecv.cv_results_.values()
    )


@pytest.mark.parametrize("ClsRFE", [RFE, RFECV])
def test_multioutput(ClsRFE):
    """Test case for multi-output scenarios with RFE or RFECV.

    This test verifies that the RFE or RFECV estimator works correctly when
    handling multi-output data.
    """
    X = np.random.normal(size=(10, 3))
    y = np.random.randint(2, size=(10, 2))
    clf = RandomForestClassifier(n_estimators=5)
    rfe_test = ClsRFE(clf)
    rfe_test.fit(X, y)


@pytest.mark.parametrize("ClsRFE", [RFE, RFECV])
def test_pipeline_with_nans(ClsRFE):
    """Check that RFE works with pipelines that accept NaNs.

    Non-regression test specifically for GitHub issue 21743.
    """
    X, y = load_iris(return_X_y=True)
    X[0, 0] = np.nan

    # 创建包含多个步骤的 Pipeline 对象，处理 NaN 值
    pipe = make_pipeline(
        SimpleImputer(),
        StandardScaler(),
        LogisticRegression(),
    )

    # 使用 ClsRFE 对象处理 Pipeline
    fs = ClsRFE(
        estimator=pipe,
        importance_getter="named_steps.logisticregression.coef_",
    )
    fs.fit(X, y)


@pytest.mark.parametrize("ClsRFE", [RFE, RFECV])
@pytest.mark.parametrize("PLSEstimator", [CCA, PLSCanonical, PLSRegression])
def test_rfe_pls(ClsRFE, PLSEstimator):
    """Check the behavior of RFE with PLS estimators.

    Non-regression test specifically for GitHub issue 12410.
    """
    X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
    estimator = PLSEstimator(n_components=1)
    selector = ClsRFE(estimator, step=1).fit(X, y)
    # 断言选择器的得分是否大于 0.5
    assert selector.score(X, y) > 0.5


def test_rfe_estimator_attribute_error():
    """Check that we raise the proper AttributeError when the estimator
    does not implement the `decision_function` method.

    Non-regression test specifically for GitHub issue 28108.
    """
    iris = load_iris()

    # 当估算器（estimator）不实现 `decision_function` 方法时，应该引发 AttributeError
    rfe = RFE(estimator=LinearRegression())
    # 定义外部错误信息，指示'RFE'对象没有'decision_function'属性
    outer_msg = "This 'RFE' has no attribute 'decision_function'"
    # 定义内部错误信息，指示'LinearRegression'对象没有'decision_function'属性
    inner_msg = "'LinearRegression' object has no attribute 'decision_function'"
    # 使用 pytest 模块检查是否抛出 AttributeError 异常，并匹配外部错误信息
    with pytest.raises(AttributeError, match=outer_msg) as exec_info:
        # 对 RFE 模型使用 fit 方法，并尝试调用 decision_function 方法
        rfe.fit(iris.data, iris.target).decision_function(iris.data)
    # 断言异常信息的原因是 AttributeError
    assert isinstance(exec_info.value.__cause__, AttributeError)
    # 断言内部错误信息存在于异常信息字符串中
    assert inner_msg in str(exec_info.value.__cause__)
@pytest.mark.parametrize(
    "ClsRFE, param", [(RFE, "n_features_to_select"), (RFECV, "min_features_to_select")]
)
def test_rfe_n_features_to_select_warning(ClsRFE, param):
    """检查当尝试使用大于传递给fit方法的X变量中存在的特征数量的n_features_to_select属性初始化RFE对象时，是否会引发正确的警告。"""
    # 使用make_classification生成具有20个特征的数据集X和相应的标签y
    X, y = make_classification(n_features=20, random_state=0)

    # 使用pytest.warns检查是否会引发UserWarning，并验证警告消息中包含期望的参数和特征数量信息
    with pytest.warns(UserWarning, match=f"{param}=21 > n_features=20"):
        # 创建RFE或RFECV对象，其n_features_to_select或min_features_to_select属性大于X变量中的特征数量
        clsrfe = ClsRFE(estimator=LogisticRegression(), **{param: 21})
        # 对数据集X进行拟合
        clsrfe.fit(X, y)
```