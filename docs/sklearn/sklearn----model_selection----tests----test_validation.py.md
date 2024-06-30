# `D:\src\scipysrc\scikit-learn\sklearn\model_selection\tests\test_validation.py`

```
# 导入所需的模块和类
"""Test the validation module"""

import os  # 导入操作系统接口模块
import re  # 导入正则表达式模块
import sys  # 导入系统模块
import tempfile  # 导入临时文件模块
import warnings  # 导入警告模块
from functools import partial  # 导入偏函数模块
from io import StringIO  # 导入字符串IO模块
from time import sleep  # 导入时间模块中的睡眠函数

import numpy as np  # 导入NumPy库并重命名为np
import pytest  # 导入pytest测试框架
from scipy.sparse import issparse  # 导入判断稀疏矩阵的函数

from sklearn.base import BaseEstimator, clone  # 从sklearn库中导入基础类和克隆函数
from sklearn.cluster import KMeans  # 导入K均值聚类算法
from sklearn.datasets import (  # 从sklearn的数据集模块导入多个数据集生成函数
    load_diabetes,
    load_digits,
    load_iris,
    make_classification,
    make_multilabel_classification,
    make_regression,
)
from sklearn.ensemble import RandomForestClassifier  # 导入随机森林分类器
from sklearn.exceptions import FitFailedWarning  # 导入拟合失败警告
from sklearn.impute import SimpleImputer  # 导入简单填充模块
from sklearn.linear_model import (  # 从sklearn的线性模型模块导入多个线性模型
    LogisticRegression,
    PassiveAggressiveClassifier,
    Ridge,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.metrics import (  # 从sklearn的指标模块导入多个评估指标
    accuracy_score,
    check_scoring,
    confusion_matrix,
    explained_variance_score,
    make_scorer,
    mean_squared_error,
    precision_recall_fscore_support,
    precision_score,
    r2_score,
)
from sklearn.metrics._scorer import _MultimetricScorer  # 导入多指标评分器
from sklearn.model_selection import (  # 导入模型选择模块中的多个函数和类
    GridSearchCV,
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    LeaveOneGroupOut,
    LeaveOneOut,
    LeavePGroupsOut,
    ShuffleSplit,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    cross_validate,
    learning_curve,
    permutation_test_score,
    validation_curve,
)
from sklearn.model_selection._validation import (  # 导入验证模块中的多个函数
    _check_is_permutation,
    _fit_and_score,
    _score,
)
from sklearn.model_selection.tests.common import OneTimeSplitter  # 导入测试共用模块
from sklearn.model_selection.tests.test_search import FailingClassifier  # 导入失败分类器
from sklearn.multiclass import OneVsRestClassifier  # 导入一对多分类器
from sklearn.neighbors import KNeighborsClassifier  # 导入K近邻分类器
from sklearn.neural_network import MLPRegressor  # 导入多层感知机回归器
from sklearn.pipeline import Pipeline  # 导入管道模块
from sklearn.preprocessing import LabelEncoder, scale  # 导入标签编码器和缩放函数
from sklearn.svm import SVC, LinearSVC  # 导入支持向量分类器和线性支持向量分类器
from sklearn.tests.metadata_routing_common import (  # 导入测试元数据常用模块中的多个类和函数
    ConsumingClassifier,
    ConsumingScorer,
    ConsumingSplitter,
    _Registry,
    check_recorded_metadata,
)
from sklearn.utils import shuffle  # 导入随机打乱函数
from sklearn.utils._mocking import CheckingClassifier, MockDataFrame  # 导入检查分类器和模拟数据帧
from sklearn.utils._testing import (  # 导入测试工具中的多个断言函数
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)
from sklearn.utils.fixes import COO_CONTAINERS, CSR_CONTAINERS  # 导入修复模块中的COO和CSR容器
from sklearn.utils.validation import _num_samples  # 导入验证模块中的样本数量函数


class MockImprovingEstimator(BaseEstimator):
    """Dummy classifier to test the learning curve"""

    def __init__(self, n_max_train_sizes):
        self.n_max_train_sizes = n_max_train_sizes  # 初始化最大训练集大小
        self.train_sizes = 0  # 初始化训练集大小为0
        self.X_subset = None  # 初始化特征子集为空

    def fit(self, X_subset, y_subset=None):
        self.X_subset = X_subset  # 设置特征子集
        self.train_sizes = X_subset.shape[0]  # 计算特征子集的样本数量作为训练集大小
        return self  # 返回当前实例

    def predict(self, X):
        raise NotImplementedError  # 抛出未实现错误，要求子类实现该方法
    # 计算模型得分
    def score(self, X=None, Y=None):
        # 如果传入的数据是训练数据
        if self._is_training_data(X):
            # 计算训练得分，根据比例调整基础得分（2.0），使训练得分变差（从2降到1）
            return 2.0 - float(self.train_sizes) / self.n_max_train_sizes
        else:
            # 计算测试得分，基础得分（1.0）除以最大训练数据量，以此作为测试得分
            return float(self.train_sizes) / self.n_max_train_sizes
    
    # 判断是否为训练数据的私有方法
    def _is_training_data(self, X):
        # 返回传入的数据 X 是否与对象中存储的训练数据 X_subset 相同
        return X is self.X_subset
class MockIncrementalImprovingEstimator(MockImprovingEstimator):
    """Dummy classifier that provides partial_fit"""

    def __init__(self, n_max_train_sizes, expected_fit_params=None):
        # 调用父类的初始化方法，设置最大训练大小
        super().__init__(n_max_train_sizes)
        # 初始化属性
        self.x = None
        self.expected_fit_params = expected_fit_params

    def _is_training_data(self, X):
        # 检查 self.x 是否在 X 中
        return self.x in X

    def partial_fit(self, X, y=None, **params):
        # 增加已训练样本数量
        self.train_sizes += X.shape[0]
        # 设置 self.x 为 X 的第一个元素
        self.x = X[0]
        # 检查是否存在期望的拟合参数
        if self.expected_fit_params:
            missing = set(self.expected_fit_params) - set(params)
            if missing:
                # 抛出断言错误，显示缺失的拟合参数
                raise AssertionError(
                    f"Expected fit parameter(s) {list(missing)} not seen."
                )
            for key, value in params.items():
                # 检查每个拟合参数的样本数是否正确
                if key in self.expected_fit_params and _num_samples(
                    value
                ) != _num_samples(X):
                    # 如果样本数不匹配，抛出断言错误
                    raise AssertionError(
                        f"Fit parameter {key} has length {_num_samples(value)}"
                        f"; expected {_num_samples(X)}."
                    )


class MockEstimatorWithParameter(BaseEstimator):
    """Dummy classifier to test the validation curve"""

    def __init__(self, param=0.5):
        # 初始化属性
        self.X_subset = None
        self.param = param

    def fit(self, X_subset, y_subset):
        # 设置训练子集并记录其大小
        self.X_subset = X_subset
        self.train_sizes = X_subset.shape[0]
        return self

    def predict(self, X):
        # 抛出未实现错误，要求子类实现该方法
        raise NotImplementedError

    def score(self, X=None, y=None):
        # 根据是否为训练数据返回得分
        return self.param if self._is_training_data(X) else 1 - self.param

    def _is_training_data(self, X):
        # 检查输入数据是否为训练子集
        return X is self.X_subset


class MockEstimatorWithSingleFitCallAllowed(MockEstimatorWithParameter):
    """Dummy classifier that disallows repeated calls of fit method"""

    def fit(self, X_subset, y_subset):
        # 检查 fit 方法是否已被调用过
        assert not hasattr(self, "fit_called_"), "fit is called the second time"
        self.fit_called_ = True
        return super().fit(X_subset, y_subset)

    def predict(self, X):
        # 抛出未实现错误，要求子类实现该方法
        raise NotImplementedError


class MockClassifier:
    """Dummy classifier to test the cross-validation"""

    def __init__(self, a=0, allow_nd=False):
        # 初始化属性
        self.a = a
        self.allow_nd = allow_nd

    def fit(
        self,
        X,
        Y=None,
        sample_weight=None,
        class_prior=None,
        sparse_sample_weight=None,
        sparse_param=None,
        dummy_int=None,
        dummy_str=None,
        dummy_obj=None,
        callback=None,
    ):
        # fit 方法用于训练分类器，接受多个参数
        pass
    ):
        """
        The dummy arguments are to test that this fit function can
        accept non-array arguments through cross-validation, such as:
            - int
            - str (this is actually array-like)
            - object
            - function
        """
        # 将传入的参数赋给对象的属性
        self.dummy_int = dummy_int
        self.dummy_str = dummy_str
        self.dummy_obj = dummy_obj
        # 如果存在回调函数，则调用回调函数并传入当前对象
        if callback is not None:
            callback(self)

        # 如果允许多维数据，将X重塑为二维数组
        if self.allow_nd:
            X = X.reshape(len(X), -1)
        # 如果X的维度大于等于3且不允许多维数据，则引发数值错误
        if X.ndim >= 3 and not self.allow_nd:
            raise ValueError("X cannot be d")
        # 如果存在样本权重，则确保样本权重的形状与X的第一维度相同
        if sample_weight is not None:
            assert sample_weight.shape[0] == X.shape[0], (
                "MockClassifier extra fit_param "
                "sample_weight.shape[0] is {0}, should be {1}".format(
                    sample_weight.shape[0], X.shape[0]
                )
            )
        # 如果存在类先验概率，则确保类先验概率的形状与y的唯一值数量相同
        if class_prior is not None:
            assert class_prior.shape[0] == len(np.unique(y)), (
                "MockClassifier extra fit_param class_prior.shape[0]"
                " is {0}, should be {1}".format(class_prior.shape[0], len(np.unique(y)))
            )
        # 如果存在稀疏样本权重，则确保稀疏样本权重的形状与X的第一维度相同
        if sparse_sample_weight is not None:
            fmt = (
                "MockClassifier extra fit_param sparse_sample_weight"
                ".shape[0] is {0}, should be {1}"
            )
            assert sparse_sample_weight.shape[0] == X.shape[0], fmt.format(
                sparse_sample_weight.shape[0], X.shape[0]
            )
        # 如果存在稀疏参数，则确保稀疏参数的形状与P的形状相同
        if sparse_param is not None:
            fmt = (
                "MockClassifier extra fit_param sparse_param.shape "
                "is ({0}, {1}), should be ({2}, {3})"
            )
            assert sparse_param.shape == P.shape, fmt.format(
                sparse_param.shape[0],
                sparse_param.shape[1],
                P.shape[0],
                P.shape[1],
            )
        # 返回当前对象自身
        return self

    def predict(self, T):
        # 如果允许多维数据，将T重塑为二维数组
        if self.allow_nd:
            T = T.reshape(len(T), -1)
        # 返回T的第一列作为预测结果
        return T[:, 0]

    def predict_proba(self, T):
        # 返回T作为预测概率结果
        return T

    def score(self, X=None, Y=None):
        # 返回一个得分，计算公式为 1 / (1 + |self.a|)
        return 1.0 / (1 + np.abs(self.a))

    def get_params(self, deep=False):
        # 返回一个包含当前对象参数的字典
        return {"a": self.a, "allow_nd": self.allow_nd}
# XXX: use 2D array, since 1D X is being detected as a single sample in
# check_consistent_length
X = np.ones((10, 2))  # 创建一个包含10行2列的二维数组，所有元素初始化为1
y = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])  # 创建一个包含10个元素的一维数组，表示类别标签
# The number of samples per class needs to be > n_splits,
# for StratifiedKFold(n_splits=3)
y2 = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 3])  # 创建一个包含10个元素的一维数组，表示类别标签
P = np.eye(5)  # 创建一个5x5的单位矩阵


@pytest.mark.parametrize("coo_container", COO_CONTAINERS)
def test_cross_val_score(coo_container):
    clf = MockClassifier()  # 创建一个模拟分类器对象
    X_sparse = coo_container(X)  # 将稠密矩阵X转换为稀疏矩阵格式

    for a in range(-10, 10):  # 对a从-10到9进行迭代
        clf.a = a  # 设置分类器的参数a为当前迭代值

        # Smoke test
        scores = cross_val_score(clf, X, y2)  # 进行交叉验证评分
        assert_array_equal(scores, clf.score(X, y2))  # 检查交叉验证评分与单次评分的一致性

        # test with multioutput y
        multioutput_y = np.column_stack([y2, y2[::-1]])  # 创建一个多输出格式的标签数组
        scores = cross_val_score(clf, X_sparse, multioutput_y)  # 使用稀疏矩阵和多输出标签进行交叉验证评分
        assert_array_equal(scores, clf.score(X_sparse, multioutput_y))  # 检查交叉验证评分与单次评分的一致性

        scores = cross_val_score(clf, X_sparse, y2)  # 使用稀疏矩阵和单一标签进行交叉验证评分
        assert_array_equal(scores, clf.score(X_sparse, y2))  # 检查交叉验证评分与单次评分的一致性

        # test with multioutput y
        scores = cross_val_score(clf, X_sparse, multioutput_y)  # 再次使用多输出标签进行交叉验证评分
        assert_array_equal(scores, clf.score(X_sparse, multioutput_y))  # 检查交叉验证评分与单次评分的一致性

    # test with X and y as list
    list_check = lambda x: isinstance(x, list)  # 定义一个检查函数，判断输入是否为列表
    clf = CheckingClassifier(check_X=list_check)  # 创建一个检查输入特征的分类器对象
    scores = cross_val_score(clf, X.tolist(), y2.tolist(), cv=3)  # 使用列表形式的输入特征进行交叉验证评分

    clf = CheckingClassifier(check_y=list_check)  # 创建一个检查标签的分类器对象
    scores = cross_val_score(clf, X, y2.tolist(), cv=3)  # 使用列表形式的标签进行交叉验证评分

    # test with 3d X and
    X_3d = X[:, :, np.newaxis]  # 将二维矩阵X扩展为三维矩阵
    clf = MockClassifier(allow_nd=True)  # 创建一个允许多维特征输入的分类器对象
    scores = cross_val_score(clf, X_3d, y2)  # 使用三维矩阵进行交叉验证评分

    clf = MockClassifier(allow_nd=False)  # 创建一个不允许多维特征输入的分类器对象
    with pytest.raises(ValueError):  # 检查是否会引发值错误异常
        cross_val_score(clf, X_3d, y2, error_score="raise")  # 使用不允许的分类器进行交叉验证评分，期望引发异常


def test_cross_validate_many_jobs():
    # regression test for #12154: cv='warn' with n_jobs>1 trigger a copy of
    # the parameters leading to a failure in check_cv due to cv is 'warn'
    # instead of cv == 'warn'.
    X, y = load_iris(return_X_y=True)  # 加载鸢尾花数据集的特征和标签
    clf = SVC(gamma="auto")  # 创建一个支持向量分类器对象
    grid = GridSearchCV(clf, param_grid={"C": [1, 10]})  # 创建一个带有参数网格的网格搜索对象
    cross_validate(grid, X, y, n_jobs=2)  # 使用多线程进行交叉验证评分


def test_cross_validate_invalid_scoring_param():
    X, y = make_classification(random_state=0)  # 创建一个随机分类数据集
    estimator = MockClassifier()  # 创建一个模拟分类器对象

    # Test the errors
    error_message_regexp = ".*must be unique strings.*"

    # List/tuple of callables should raise a message advising users to use
    # dict of names to callables mapping
    with pytest.raises(ValueError, match=error_message_regexp):
        cross_validate(
            estimator,
            X,
            y,
            scoring=(make_scorer(precision_score), make_scorer(accuracy_score)),
        )  # 使用可调用对象列表进行交叉验证评分，期望引发值错误异常
    with pytest.raises(ValueError, match=error_message_regexp):
        cross_validate(estimator, X, y, scoring=(make_scorer(precision_score),))  # 使用单个可调用对象进行交叉验证评分，期望引发值错误异常

    # So should empty lists/tuples
    with pytest.raises(ValueError, match=error_message_regexp + "Empty list.*"):
        cross_validate(estimator, X, y, scoring=())  # 使用空的评分列表进行交叉验证评分，期望引发值错误异常
    # 使用 pytest 的 raises 断言来检查是否抛出 ValueError 异常，并验证异常消息是否包含特定的正则表达式
    with pytest.raises(ValueError, match=error_message_regexp + "Duplicate.*"):
        cross_validate(estimator, X, y, scoring=("f1_micro", "f1_micro"))

    # 使用 pytest 的 raises 断言来检查是否抛出 ValueError 异常，并验证异常消息是否包含特定的正则表达式
    # scoring 参数传入了一个嵌套的列表，期望抛出异常并包含特定的错误消息正则表达式
    with pytest.raises(ValueError, match=error_message_regexp):
        cross_validate(estimator, X, y, scoring=[[make_scorer(precision_score)]])

    # 使用 pytest 的 raises 断言来检查是否抛出 ValueError 异常，并验证异常消息是否包含特定的字符串
    # scoring 参数传入了一个空的字典，期望抛出异常并包含特定的错误消息
    with pytest.raises(ValueError, match="An empty dict"):
        cross_validate(estimator, X, y, scoring=(dict()))

    # 创建一个多类别的评分器
    multiclass_scorer = make_scorer(precision_recall_fscore_support)

    # 使用 pytest 的 warns 断言来检查是否发出 UserWarning 警告，并验证警告消息是否包含特定的字符串
    # scoring 参数传入了一个返回多个值的多类别评分器，期望发出警告并包含特定的警告消息
    warning_message = (
        "Scoring failed. The score on this train-test "
        f"partition for these parameters will be set to {np.nan}. "
        "Details: \n"
    )
    with pytest.warns(UserWarning, match=warning_message):
        cross_validate(estimator, X, y, scoring=multiclass_scorer)

    # 使用 pytest 的 warns 断言来检查是否发出 UserWarning 警告，并验证警告消息是否包含特定的字符串
    # scoring 参数传入了一个字典，其中键为 "foo"，值为多类别评分器，期望发出警告并包含特定的警告消息
    with pytest.warns(UserWarning, match=warning_message):
        cross_validate(estimator, X, y, scoring={"foo": multiclass_scorer})
def test_cross_validate_nested_estimator():
    # 非回归测试，确保嵌套估算器以列表形式正确返回
    # https://github.com/scikit-learn/scikit-learn/pull/17745
    # 载入鸢尾花数据集，返回特征矩阵 X 和标签向量 y
    (X, y) = load_iris(return_X_y=True)
    
    # 创建管道对象，包括填充器和模拟分类器
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer()),
            ("classifier", MockClassifier()),
        ]
    )
    
    # 使用交叉验证评估管道对象，返回结果字典
    results = cross_validate(pipeline, X, y, return_estimator=True)
    # 获取估算器列表
    estimators = results["estimator"]

    # 断言估算器是一个列表
    assert isinstance(estimators, list)
    # 断言估算器列表中的每个元素都是管道对象
    assert all(isinstance(estimator, Pipeline) for estimator in estimators)


@pytest.mark.parametrize("use_sparse", [False, True])
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_cross_validate(use_sparse: bool, csr_container):
    # 计算训练和测试的均方误差（MSE）和决定系数（R^2）分数
    # 使用 KFold 交叉验证对象
    cv = KFold()

    # 回归任务
    X_reg, y_reg = make_regression(n_samples=30, random_state=0)
    reg = Ridge(random_state=0)

    # 分类任务
    X_clf, y_clf = make_classification(n_samples=30, random_state=0)
    clf = SVC(kernel="linear", random_state=0)

    # 如果需要稀疏矩阵，将数据转换为稀疏格式
    if use_sparse:
        X_reg = csr_container(X_reg)
        X_clf = csr_container(X_clf)

    # 对于每个任务（回归和分类），执行以下操作
    for X, y, est in ((X_reg, y_reg, reg), (X_clf, y_clf, clf)):
        # 克隆估算器并在训练集上拟合
        # 计算均方误差和决定系数的得分
        mse_scorer = check_scoring(est, scoring="neg_mean_squared_error")
        r2_scorer = check_scoring(est, scoring="r2")
        train_mse_scores = []
        test_mse_scores = []
        train_r2_scores = []
        test_r2_scores = []
        fitted_estimators = []

        # 使用交叉验证拆分数据集并进行评估
        for train, test in cv.split(X, y):
            est = clone(est).fit(X[train], y[train])
            train_mse_scores.append(mse_scorer(est, X[train], y[train]))
            train_r2_scores.append(r2_scorer(est, X[train], y[train]))
            test_mse_scores.append(mse_scorer(est, X[test], y[test]))
            test_r2_scores.append(r2_scorer(est, X[test], y[test]))
            fitted_estimators.append(est)

        # 转换为 NumPy 数组以便进一步处理
        train_mse_scores = np.array(train_mse_scores)
        test_mse_scores = np.array(test_mse_scores)
        train_r2_scores = np.array(train_r2_scores)
        test_r2_scores = np.array(test_r2_scores)
        fitted_estimators = np.array(fitted_estimators)

        # 将所有得分和拟合的估算器组合成一个元组
        scores = (
            train_mse_scores,
            test_mse_scores,
            train_r2_scores,
            test_r2_scores,
            fitted_estimators,
        )

        # 通过检查单一度量函数来确保测试不会受到数据切片引起的统计波动的影响
        check_cross_validate_single_metric(est, X, y, scores, cv)
        # 通过检查多重度量函数来确保测试的完整性
        check_cross_validate_multi_metric(est, X, y, scores, cv)


def check_cross_validate_single_metric(clf, X, y, scores, cv):
    # 解压缩得分元组
    (
        train_mse_scores,
        test_mse_scores,
        train_r2_scores,
        test_r2_scores,
        fitted_estimators,
    ) = scores
    # 测试单一指标评估，当 scoring 是字符串或单一列表时
    for return_train_score, dict_len in ((True, 4), (False, 3)):
        # 单一指标作为字符串传递
        if return_train_score:
            # 使用交叉验证评估模型，返回训练集上的均方误差字典
            mse_scores_dict = cross_validate(
                clf,
                X,
                y,
                scoring="neg_mean_squared_error",
                return_train_score=True,
                cv=cv,
            )
            # 断言训练集均方误差分数与预期值相近
            assert_array_almost_equal(mse_scores_dict["train_score"], train_mse_scores)
        else:
            # 使用交叉验证评估模型，返回测试集上的均方误差字典
            mse_scores_dict = cross_validate(
                clf,
                X,
                y,
                scoring="neg_mean_squared_error",
                return_train_score=False,
                cv=cv,
            )
        # 断言返回结果是字典类型
        assert isinstance(mse_scores_dict, dict)
        # 断言字典长度符合预期
        assert len(mse_scores_dict) == dict_len
        # 断言测试集均方误差分数与预期值相近
        assert_array_almost_equal(mse_scores_dict["test_score"], test_mse_scores)

        # 单一指标作为列表传递
        if return_train_score:
            # 默认情况下必须为 True - 已弃用
            r2_scores_dict = cross_validate(
                clf, X, y, scoring=["r2"], return_train_score=True, cv=cv
            )
            # 断言训练集 R^2 分数与预期值相近
            assert_array_almost_equal(r2_scores_dict["train_r2"], train_r2_scores, True)
        else:
            # 使用交叉验证评估模型，返回测试集上的 R^2 字典
            r2_scores_dict = cross_validate(
                clf, X, y, scoring=["r2"], return_train_score=False, cv=cv
            )
        # 断言返回结果是字典类型
        assert isinstance(r2_scores_dict, dict)
        # 断言字典长度符合预期
        assert len(r2_scores_dict) == dict_len
        # 断言测试集 R^2 分数与预期值相近
        assert_array_almost_equal(r2_scores_dict["test_r2"], test_r2_scores)

    # 测试 return_estimator 选项
    # 使用交叉验证评估模型，返回包含估计器的字典，其中包括每个折叠的训练后的估计器
    mse_scores_dict = cross_validate(
        clf, X, y, scoring="neg_mean_squared_error", return_estimator=True, cv=cv
    )
    for k, est in enumerate(mse_scores_dict["estimator"]):
        est_coef = est.coef_.copy()
        if issparse(est_coef):
            est_coef = est_coef.toarray()

        fitted_est_coef = fitted_estimators[k].coef_.copy()
        if issparse(fitted_est_coef):
            fitted_est_coef = fitted_est_coef.toarray()

        # 断言估计器的系数与预期值相近
        assert_almost_equal(est_coef, fitted_est_coef)
        # 断言估计器的截距与预期值相近
        assert_almost_equal(est.intercept_, fitted_estimators[k].intercept_)
# 定义一个函数用于多指标交叉验证检查
def check_cross_validate_multi_metric(clf, X, y, scores, cv):
    # 将得分(scores)解包为训练集和测试集的均方误差分数、R^2分数以及已拟合的估算器
    (
        train_mse_scores,
        test_mse_scores,
        train_r2_scores,
        test_r2_scores,
        fitted_estimators,
    ) = scores

    # 定义一个自定义评分器函数，用于计算 R^2 和负均方误差
    def custom_scorer(clf, X, y):
        y_pred = clf.predict(X)
        return {
            "r2": r2_score(y, y_pred),
            "neg_mean_squared_error": -mean_squared_error(y, y_pred),
        }

    # 定义多种评分方式：元组、字典和自定义评分器
    all_scoring = (
        ("r2", "neg_mean_squared_error"),
        {
            "r2": make_scorer(r2_score),
            "neg_mean_squared_error": "neg_mean_squared_error",
        },
        custom_scorer,
    )

    # 定义没有训练集相关键的集合
    keys_sans_train = {
        "test_r2",
        "test_neg_mean_squared_error",
        "fit_time",
        "score_time",
    }
    # 含有训练集相关键的集合
    keys_with_train = keys_sans_train.union(
        {"train_r2", "train_neg_mean_squared_error"}
    )

    # 遍历是否返回训练集分数的选项和评分方式的组合
    for return_train_score in (True, False):
        for scoring in all_scoring:
            if return_train_score:
                # 如果要求返回训练集分数，则使用交叉验证函数获取交叉验证结果
                cv_results = cross_validate(
                    clf, X, y, scoring=scoring, return_train_score=True, cv=cv
                )
                # 断言训练集R^2分数和均方误差与预期相近
                assert_array_almost_equal(cv_results["train_r2"], train_r2_scores)
                assert_array_almost_equal(
                    cv_results["train_neg_mean_squared_error"], train_mse_scores
                )
            else:
                # 如果不要求返回训练集分数，则使用交叉验证函数获取交叉验证结果
                cv_results = cross_validate(
                    clf, X, y, scoring=scoring, return_train_score=False, cv=cv
                )
            # 断言交叉验证结果是一个字典类型
            assert isinstance(cv_results, dict)
            # 断言交叉验证结果的键与期望的键集合相匹配
            assert set(cv_results.keys()) == (
                keys_with_train if return_train_score else keys_sans_train
            )
            # 断言测试集R^2分数和均方误差与预期相近
            assert_array_almost_equal(cv_results["test_r2"], test_r2_scores)
            assert_array_almost_equal(
                cv_results["test_neg_mean_squared_error"], test_mse_scores
            )

            # 确保所有的数组都是 np.ndarray 类型
            assert type(cv_results["test_r2"]) == np.ndarray
            assert type(cv_results["test_neg_mean_squared_error"]) == np.ndarray
            assert type(cv_results["fit_time"]) == np.ndarray
            assert type(cv_results["score_time"]) == np.ndarray

            # 确保所有时间都在合理范围内
            assert np.all(cv_results["fit_time"] >= 0)
            assert np.all(cv_results["fit_time"] < 10)
            assert np.all(cv_results["score_time"] >= 0)
            assert np.all(cv_results["score_time"] < 10)


# 定义一个函数用于测试交叉验证分数和预测分组
def test_cross_val_score_predict_groups():
    # 生成一个分类样本集
    X, y = make_classification(n_samples=20, n_classes=2, random_state=0)
    
    # 初始化一个支持向量分类器
    clf = SVC(kernel="linear")
    # 定义一组用于组别交叉验证的交叉验证策略对象列表
    group_cvs = [
        LeaveOneGroupOut(),      # 每次留下一个组进行验证的交叉验证策略
        LeavePGroupsOut(2),      # 每次留下指定数量的组进行验证的交叉验证策略
        GroupKFold(),            # 根据组标签进行 k 折交叉验证的策略
        GroupShuffleSplit(),     # 根据组标签进行分组随机划分的交叉验证策略
    ]
    # 定义一个错误消息，用于验证 'groups' 参数不应为 None 的情况
    error_message = "The 'groups' parameter should not be None."
    # 对每个交叉验证策略对象进行循环
    for cv in group_cvs:
        # 使用 pytest 的断言检查 cross_val_score 函数是否会引发 ValueError 异常，并匹配特定的错误消息
        with pytest.raises(ValueError, match=error_message):
            cross_val_score(estimator=clf, X=X, y=y, cv=cv)
        # 使用 pytest 的断言检查 cross_val_predict 函数是否会引发 ValueError 异常，并匹配特定的错误消息
        with pytest.raises(ValueError, match=error_message):
            cross_val_predict(estimator=clf, X=X, y=y, cv=cv)
@pytest.mark.filterwarnings("ignore: Using or importing the ABCs from")
# 定义一个测试函数，用于验证 cross_val_score 不会破坏 pandas 的 dataframe
def test_cross_val_score_pandas():
    types = [(MockDataFrame, MockDataFrame)]
    try:
        from pandas import DataFrame, Series
        types.append((Series, DataFrame))
    except ImportError:
        pass
    # 遍历数据类型列表
    for TargetType, InputFeatureType in types:
        # X 是 dataframe，y 是 series
        X_df, y_ser = InputFeatureType(X), TargetType(y2)
        # 定义检查函数用于确认类型
        check_df = lambda x: isinstance(x, InputFeatureType)
        check_series = lambda x: isinstance(x, TargetType)
        # 创建检查器分类器对象
        clf = CheckingClassifier(check_X=check_df, check_y=check_series)
        # 使用交叉验证评估分类器
        cross_val_score(clf, X_df, y_ser, cv=3)


def test_cross_val_score_mask():
    # 测试 cross_val_score 能够处理布尔掩码
    svm = SVC(kernel="linear")
    iris = load_iris()
    X, y = iris.data, iris.target
    # 使用 KFold 分割数据
    kfold = KFold(5)
    # 使用索引数组存储评分结果
    scores_indices = cross_val_score(svm, X, y, cv=kfold)
    # 再次创建 KFold 对象
    kfold = KFold(5)
    cv_masks = []
    # 遍历分割后的训练集和测试集
    for train, test in kfold.split(X, y):
        # 创建布尔掩码
        mask_train = np.zeros(len(y), dtype=bool)
        mask_test = np.zeros(len(y), dtype=bool)
        mask_train[train] = 1
        mask_test[test] = 1
        cv_masks.append((train, test))
    # 使用布尔掩码评估分类器
    scores_masks = cross_val_score(svm, X, y, cv=cv_masks)
    # 断言索引评分和掩码评分结果一致
    assert_array_equal(scores_indices, scores_masks)


def test_cross_val_score_precomputed():
    # 测试使用预计算内核的 SVM
    svm = SVC(kernel="precomputed")
    iris = load_iris()
    X, y = iris.data, iris.target
    # 计算线性内核
    linear_kernel = np.dot(X, X.T)
    # 使用预计算内核评分
    score_precomputed = cross_val_score(svm, linear_kernel, y)
    # 创建普通线性 SVM
    svm = SVC(kernel="linear")
    # 使用普通线性内核评分
    score_linear = cross_val_score(svm, X, y)
    # 断言预计算内核和普通线性内核的评分接近
    assert_array_almost_equal(score_precomputed, score_linear)

    # 使用可调用内核进行测试
    svm = SVC(kernel=lambda x, y: np.dot(x, y.T))
    score_callable = cross_val_score(svm, X, y)
    # 断言预计算内核和可调用内核的评分接近
    assert_array_almost_equal(score_precomputed, score_callable)

    # 针对非方阵 X 抛出 ValueError
    svm = SVC(kernel="precomputed")
    with pytest.raises(ValueError):
        cross_val_score(svm, X, y)

    # 测试当预计算内核不是类数组或稀疏格式时抛出 ValueError
    with pytest.raises(ValueError):
        cross_val_score(svm, linear_kernel.tolist(), y)


@pytest.mark.parametrize("coo_container", COO_CONTAINERS)
# 测试 cross_val_score 的 fit_params 参数
def test_cross_val_score_fit_params(coo_container):
    clf = MockClassifier()
    n_samples = X.shape[0]
    n_classes = len(np.unique(y))

    # 创建稀疏权重和稀疏投影矩阵
    W_sparse = coo_container(
        (np.array([1]), (np.array([1]), np.array([0]))), shape=(10, 1)
    )
    P_sparse = coo_container(np.eye(5))

    DUMMY_INT = 42
    DUMMY_STR = "42"
    DUMMY_OBJ = object()
    # 定义一个函数，用于检查分类器的参数是否正确传递
    def assert_fit_params(clf):
        # 断言分类器的 dummy_int 参数与预期值 DUMMY_INT 相等
        assert clf.dummy_int == DUMMY_INT
        # 断言分类器的 dummy_str 参数与预期值 DUMMY_STR 相等
        assert clf.dummy_str == DUMMY_STR
        # 断言分类器的 dummy_obj 参数与预期值 DUMMY_OBJ 相等
        assert clf.dummy_obj == DUMMY_OBJ

    # 定义一个字典，包含了用于拟合模型的参数
    fit_params = {
        "sample_weight": np.ones(n_samples),  # 设置样本权重为全1数组
        "class_prior": np.full(n_classes, 1.0 / n_classes),  # 设置类先验概率为均匀分布
        "sparse_sample_weight": W_sparse,  # 设置稀疏样本权重
        "sparse_param": P_sparse,  # 设置稀疏参数
        "dummy_int": DUMMY_INT,  # 设置虚拟整数参数
        "dummy_str": DUMMY_STR,  # 设置虚拟字符串参数
        "dummy_obj": DUMMY_OBJ,  # 设置虚拟对象参数
        "callback": assert_fit_params,  # 设置回调函数为 assert_fit_params
    }
    # 使用交叉验证评估分类器 clf 在数据集 X, y 上的性能，并传递拟合参数 fit_params
    cross_val_score(clf, X, y, params=fit_params)
# 定义一个测试函数，用于测试带有自定义评分函数的交叉验证
def test_cross_val_score_score_func():
    # 创建一个模拟分类器对象
    clf = MockClassifier()
    # 初始化一个空列表，用于存储评分函数被调用时的参数
    _score_func_args = []

    # 定义一个评分函数，用于将参数保存到 _score_func_args 列表中并返回固定的分数 1.0
    def score_func(y_test, y_predict):
        _score_func_args.append((y_test, y_predict))
        return 1.0

    # 使用 `warnings.catch_warnings` 上下文管理器捕获警告
    with warnings.catch_warnings(record=True):
        # 使用 `make_scorer` 函数创建一个评分器，将评分函数作为参数传入
        scoring = make_scorer(score_func)
        # 进行交叉验证，计算每个折叠的评分，返回一个包含分数的数组
        score = cross_val_score(clf, X, y, scoring=scoring, cv=3)
    # 断言交叉验证得到的分数数组与预期的 [1.0, 1.0, 1.0] 相等
    assert_array_equal(score, [1.0, 1.0, 1.0])
    # 断言评分函数被调用的次数为 3 次（对应 cv=3）
    assert len(_score_func_args) == 3


# 定义一个测试函数，用于测试带有不同评分方法的交叉验证（分类任务）
def test_cross_val_score_with_score_func_classification():
    # 加载鸢尾花数据集
    iris = load_iris()
    # 创建一个支持向量分类器对象，使用线性核
    clf = SVC(kernel="linear")

    # 默认评分（应为准确率）
    scores = cross_val_score(clf, iris.data, iris.target)
    # 断言默认评分结果与预期的 [0.97, 1.0, 0.97, 0.97, 1.0] 数组在小数点后两位上相等
    assert_array_almost_equal(scores, [0.97, 1.0, 0.97, 0.97, 1.0], 2)

    # 正确分类评分（即零一损失评分），应与默认评估器评分相同
    zo_scores = cross_val_score(clf, iris.data, iris.target, scoring="accuracy")
    # 断言零一损失评分结果与预期的 [0.97, 1.0, 0.97, 0.97, 1.0] 数组在小数点后两位上相等
    assert_array_almost_equal(zo_scores, [0.97, 1.0, 0.97, 0.97, 1.0], 2)

    # F1 分数（由于类别平衡，因此应与零一损失评分相等）
    f1_scores = cross_val_score(clf, iris.data, iris.target, scoring="f1_weighted")
    # 断言 F1 分数结果与预期的 [0.97, 1.0, 0.97, 0.97, 1.0] 数组在小数点后两位上相等
    assert_array_almost_equal(f1_scores, [0.97, 1.0, 0.97, 0.97, 1.0], 2)


# 定义一个测试函数，用于测试带有不同评分方法的交叉验证（回归任务）
def test_cross_val_score_with_score_func_regression():
    # 生成回归任务的样本数据
    X, y = make_regression(n_samples=30, n_features=20, n_informative=5, random_state=0)
    # 创建一个岭回归对象
    reg = Ridge()

    # 默认岭回归评分
    scores = cross_val_score(reg, X, y)
    # 断言默认评分结果与预期的 [0.94, 0.97, 0.97, 0.99, 0.92] 数组在小数点后两位上相等
    assert_array_almost_equal(scores, [0.94, 0.97, 0.97, 0.99, 0.92], 2)

    # R2 分数（决定系数），应与默认评估器评分相同
    r2_scores = cross_val_score(reg, X, y, scoring="r2")
    # 断言 R2 分数结果与预期的 [0.94, 0.97, 0.97, 0.99, 0.92] 数组在小数点后两位上相等
    assert_array_almost_equal(r2_scores, [0.94, 0.97, 0.97, 0.99, 0.92], 2)

    # 均方误差（损失函数，因此分数是负数）
    neg_mse_scores = cross_val_score(reg, X, y, scoring="neg_mean_squared_error")
    expected_neg_mse = np.array([-763.07, -553.16, -274.38, -273.26, -1681.99])
    # 断言负均方误差分数结果与预期的数组在小数点后两位上相等
    assert_array_almost_equal(neg_mse_scores, expected_neg_mse, 2)

    # 解释方差
    scoring = make_scorer(explained_variance_score)
    ev_scores = cross_val_score(reg, X, y, scoring=scoring)
    # 断言解释方差分数结果与预期的 [0.94, 0.97, 0.97, 0.99, 0.92] 数组在小数点后两位上相等
    assert_array_almost_equal(ev_scores, [0.94, 0.97, 0.97, 0.99, 0.92], 2)


# 使用 pytest 的参数化装饰器，测试置换测试分数的函数
@pytest.mark.parametrize("coo_container", COO_CONTAINERS)
def test_permutation_score(coo_container):
    # 加载鸢尾花数据集
    iris = load_iris()
    X = iris.data
    # 将数据稀疏化
    X_sparse = coo_container(X)
    y = iris.target
    # 创建一个支持向量机对象，使用线性核
    svm = SVC(kernel="linear")
    # 使用分层 k 折交叉验证
    cv = StratifiedKFold(2)

    # 进行置换测试得分计算
    score, scores, pvalue = permutation_test_score(
        svm, X, y, n_permutations=30, cv=cv, scoring="accuracy"
    )
    # 断言得分大于 0.9
    assert score > 0.9
    # 断言 p 值几乎为 0.0，精确到小数点后一位
    assert_almost_equal(pvalue, 0.0, 1)
    # 使用置换检验计算模型在数据集上的得分和 p 值
    score_group, _, pvalue_group = permutation_test_score(
        svm,  # 使用的支持向量机模型
        X,    # 特征数据集
        y,    # 标签数据集
        n_permutations=30,  # 置换次数
        cv=cv,  # 交叉验证策略
        scoring="accuracy",  # 使用准确率作为评分指标
        groups=np.ones(y.size),  # 样本分组（本例中所有样本分为一组）
        random_state=0,  # 随机种子
    )
    # 断言检查置换检验得到的分数和 p 值与预期相符
    assert score_group == score
    assert pvalue_group == pvalue

    # 使用稀疏表示检查是否能得到相同的结果
    svm_sparse = SVC(kernel="linear")  # 使用线性核的支持向量机
    cv_sparse = StratifiedKFold(2)  # 分层 K 折交叉验证策略
    score_group, _, pvalue_group = permutation_test_score(
        svm_sparse,  # 稀疏表示下的支持向量机模型
        X_sparse,    # 稀疏表示的特征数据集
        y,           # 标签数据集
        n_permutations=30,  # 置换次数
        cv=cv_sparse,  # 交叉验证策略
        scoring="accuracy",  # 使用准确率作为评分指标
        groups=np.ones(y.size),  # 样本分组（本例中所有样本分为一组）
        random_state=0,  # 随机种子
    )

    # 断言检查稀疏表示下的置换检验结果与预期相符
    assert score_group == score
    assert pvalue_group == pvalue

    # 使用自定义评分对象进行测试
    def custom_score(y_true, y_pred):
        return ((y_true == y_pred).sum() - (y_true != y_pred).sum()) / y_true.shape[0]

    scorer = make_scorer(custom_score)  # 创建自定义评分对象
    score, _, pvalue = permutation_test_score(
        svm,  # 使用的支持向量机模型
        X,    # 特征数据集
        y,    # 标签数据集
        n_permutations=100,  # 置换次数
        scoring=scorer,  # 使用自定义评分对象进行评分
        cv=cv,  # 交叉验证策略
        random_state=0,  # 随机种子
    )
    # 断言检查自定义评分下的置换检验结果与预期相符
    assert_almost_equal(score, 0.93, 2)
    assert_almost_equal(pvalue, 0.01, 3)

    # 设置随机的 y 值
    y = np.mod(np.arange(len(y)), 3)  # y 值取余操作

    score, scores, pvalue = permutation_test_score(
        svm,  # 使用的支持向量机模型
        X,    # 特征数据集
        y,    # 标签数据集（已修改为随机值）
        n_permutations=30,  # 置换次数
        cv=cv,  # 交叉验证策略
        scoring="accuracy"  # 使用准确率作为评分指标
    )

    # 断言检查随机 y 值下的置换检验结果符合预期
    assert score < 0.5
    assert pvalue > 0.2
# 测试函数，用于验证 permutation_test_score 是否允许包含 NaN 值的输入数据
def test_permutation_test_score_allow_nans():
    # 创建一个包含 NaN 值的浮点型数组 X，形状为 (10, 20)
    X = np.arange(200, dtype=np.float64).reshape(10, -1)
    X[2, :] = np.nan  # 将第二行所有元素设置为 NaN
    y = np.repeat([0, 1], X.shape[0] / 2)  # 创建目标值 y，重复 [0, 1] 各5次
    # 创建 Pipeline 对象 p，包含两个步骤：使用均值策略填充 NaN 的简单填充器和 MockClassifier 分类器
    p = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean", missing_values=np.nan)),
            ("classifier", MockClassifier()),
        ]
    )
    # 调用 permutation_test_score 函数，测试 Pipeline p 对象在输入数据 X 和 y 上的表现


# 测试函数，用于验证 permutation_test_score 是否正确处理 fit_params 参数
def test_permutation_test_score_fit_params():
    # 创建一个 10x10 的数组 X，包含 0 到 99 的整数
    X = np.arange(100).reshape(10, 10)
    y = np.array([0] * 5 + [1] * 5)  # 创建目标值 y，前半部分为 0，后半部分为 1
    # 创建一个 CheckingClassifier 对象 clf，期望 sample_weight 参数被传递
    clf = CheckingClassifier(expected_sample_weight=True)

    # 定义错误信息，用于测试异常情况：期望 sample_weight 被传递，但未被传递时抛出 AssertionError 异常
    err_msg = r"Expected sample_weight to be passed"
    with pytest.raises(AssertionError, match=err_msg):
        # 调用 permutation_test_score 函数，测试是否能捕获并抛出预期的 AssertionError 异常
        permutation_test_score(clf, X, y)

    # 定义错误信息，用于测试异常情况：sample_weight 参数形状为 (1,)，期望形状为 (8,)
    err_msg = r"sample_weight.shape == \(1,\), expected \(8,\)!"
    with pytest.raises(ValueError, match=err_msg):
        # 调用 permutation_test_score 函数，测试是否能捕获并抛出预期的 ValueError 异常
        permutation_test_score(clf, X, y, fit_params={"sample_weight": np.ones(1)})
    
    # 调用 permutation_test_score 函数，测试传递 sample_weight 参数形状为 (10,) 的情况
    permutation_test_score(clf, X, y, fit_params={"sample_weight": np.ones(10)})


# 测试函数，用于验证 cross_val_score 是否允许包含 NaN 值的输入数据
def test_cross_val_score_allow_nans():
    # 创建一个包含 NaN 值的浮点型数组 X，形状为 (10, 20)
    X = np.arange(200, dtype=np.float64).reshape(10, -1)
    X[2, :] = np.nan  # 将第二行所有元素设置为 NaN
    y = np.repeat([0, 1], X.shape[0] / 2)  # 创建目标值 y，重复 [0, 1] 各5次
    # 创建 Pipeline 对象 p，包含两个步骤：使用均值策略填充 NaN 的简单填充器和 MockClassifier 分类器
    p = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean", missing_values=np.nan)),
            ("classifier", MockClassifier()),
        ]
    )
    # 调用 cross_val_score 函数，测试 Pipeline p 对象在输入数据 X 和 y 上的表现


# 测试函数，用于验证 cross_val_score 是否正确处理多标签数据
def test_cross_val_score_multilabel():
    # 创建一个二维数组 X，包含十个样本，每个样本有两个特征
    X = np.array(
        [
            [-3, 4],
            [2, 4],
            [3, 3],
            [0, 2],
            [-3, 1],
            [-2, 1],
            [0, 0],
            [-2, -1],
            [-1, -2],
            [1, -2],
        ]
    )
    # 创建一个二维数组 y，包含十个样本，每个样本有两个标签
    y = np.array(
        [[1, 1], [0, 1], [0, 1], [0, 1], [1, 1], [0, 1], [1, 0], [1, 1], [1, 0], [0, 0]]
    )
    # 创建 KNeighborsClassifier 分类器对象 clf，设定邻居数为 1
    clf = KNeighborsClassifier(n_neighbors=1)
    # 创建三个不同的评分器：scoring_micro、scoring_macro 和 scoring_samples，用于评估分类器性能
    scoring_micro = make_scorer(precision_score, average="micro")
    scoring_macro = make_scorer(precision_score, average="macro")
    scoring_samples = make_scorer(precision_score, average="samples")
    # 分别调用 cross_val_score 函数，测试分类器在多标签数据 X 和 y 上使用不同评分器的表现
    score_micro = cross_val_score(clf, X, y, scoring=scoring_micro)
    score_macro = cross_val_score(clf, X, y, scoring=scoring_macro)
    score_samples = cross_val_score(clf, X, y, scoring=scoring_samples)
    # 断言评分结果与预期结果的接近程度
    assert_almost_equal(score_micro, [1, 1 / 2, 3 / 4, 1 / 2, 1 / 3])
    assert_almost_equal(score_macro, [1, 1 / 2, 3 / 4, 1 / 2, 1 / 4])
    assert_almost_equal(score_samples, [1, 1 / 2, 3 / 4, 1 / 2, 1 / 4])


# 使用 COO_CONTAINERS 参数化的方式测试 cross_val_predict 函数
@pytest.mark.parametrize("coo_container", COO_CONTAINERS)
def test_cross_val_predict(coo_container):
    # 加载糖尿病数据集，返回特征 X 和目标值 y
    X, y = load_diabetes(return_X_y=True)
    # 创建 KFold 交叉验证对象 cv
    cv = KFold()
    # 创建 Ridge 回归器对象 est
    est = Ridge()

    # 使用 for 循环手动实现交叉验证预测 preds2，应与 cross_val_predict 的结果相同
    preds2 = np.zeros_like(y)
    for train, test in cv.split(X, y):
        est.fit(X[train], y[train])  # 在训练集上拟合回归器
        preds2[test] = est.predict(X[test])  # 在测试集上进行预测

    # 调用 cross_val_predict 函数，获取 Ridge 回归器在数据集上的交叉验证预测结果 preds
    preds = cross_val_predict(est, X, y, cv=cv)
    # 断言预测结果和另一个预测结果几乎相等
    assert_array_almost_equal(preds, preds2)

    # 使用交叉验证预测器预测结果，并断言预测数量与实际标签数量相等
    preds = cross_val_predict(est, X, y)
    assert len(preds) == len(y)

    # 使用 LeaveOneOut 交叉验证方法预测结果，并断言预测数量与实际标签数量相等
    cv = LeaveOneOut()
    preds = cross_val_predict(est, X, y, cv=cv)
    assert len(preds) == len(y)

    # 复制特征矩阵 X，将大于中位数的元素乘以自身，转换为稀疏矩阵格式，然后进行交叉验证预测
    Xsp = X.copy()
    Xsp *= Xsp > np.median(Xsp)
    Xsp = coo_container(Xsp)
    preds = cross_val_predict(est, Xsp, y)
    # 断言预测数量与实际标签数量几乎相等
    assert_array_almost_equal(len(preds), len(y))

    # 使用 KMeans 聚类器进行交叉验证预测，并断言预测数量与实际标签数量相等
    preds = cross_val_predict(KMeans(n_init="auto"), X)
    assert len(preds) == len(y)

    # 定义一个错误的交叉验证类 BadCV，其中每次划分包含不匹配的标签索引
    class BadCV:
        def split(self, X, y=None, groups=None):
            for i in range(4):
                yield np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7, 8])

    # 使用 BadCV 进行交叉验证预测，预期引发 ValueError 异常
    with pytest.raises(ValueError):
        cross_val_predict(est, X, y, cv=BadCV())

    # 加载鸢尾花数据集，并设置警告消息内容
    X, y = load_iris(return_X_y=True)
    warning_message = (
        r"Number of classes in training fold \(2\) does "
        r"not match total number of classes \(3\). "
        "Results may not be appropriate for your use case."
    )

    # 使用 LogisticRegression 进行交叉验证预测，预期引发 RuntimeWarning 警告
    with pytest.warns(RuntimeWarning, match=warning_message):
        cross_val_predict(
            LogisticRegression(solver="liblinear"),
            X,
            y,
            method="predict_proba",
            cv=KFold(2),
        )
# 定义一个用于测试交叉验证预测决策函数形状的函数
def test_cross_val_predict_decision_function_shape():
    # 创建一个二元分类的样本集 X 和标签 y，包含 50 个样本，使用随机种子 0
    X, y = make_classification(n_classes=2, n_samples=50, random_state=0)

    # 使用交叉验证预测决策函数，采用逻辑回归分类器（solver="liblinear"）
    preds = cross_val_predict(
        LogisticRegression(solver="liblinear"), X, y, method="decision_function"
    )
    # 断言预测结果的形状应为 (50,)
    assert preds.shape == (50,)

    # 载入鸢尾花数据集，返回特征矩阵 X 和标签 y
    X, y = load_iris(return_X_y=True)

    # 使用交叉验证预测决策函数，同样采用逻辑回归分类器（solver="liblinear"）
    preds = cross_val_predict(
        LogisticRegression(solver="liblinear"), X, y, method="decision_function"
    )
    # 断言预测结果的形状应为 (150, 3)
    assert preds.shape == (150, 3)

    # 这段代码特别测试了在二元分类中使用决策函数的不平衡拆分情况。
    # 这仅适用于可以在单个类别上拟合的分类器。
    # 准备数据，只使用前 100 个样本和对应的标签
    X = X[:100]
    y = y[:100]
    # 预期的错误消息，用于捕获值错误的异常，匹配特定的错误信息
    error_message = (
        "Only 1 class/es in training fold,"
        " but 2 in overall dataset. This"
        " is not supported for decision_function"
        " with imbalanced folds. To fix "
        "this, use a cross-validation technique "
        "resulting in properly stratified folds"
    )
    # 使用 pytest 检查是否会引发 ValueError 异常，并匹配特定的错误消息
    with pytest.raises(ValueError, match=error_message):
        cross_val_predict(
            RidgeClassifier(), X, y, method="decision_function", cv=KFold(2)
        )

    # 载入手写数字数据集，返回特征矩阵 X 和标签 y
    X, y = load_digits(return_X_y=True)
    # 创建一个线性核的支持向量机分类器，决策函数采用 "ovo" 形式
    est = SVC(kernel="linear", decision_function_shape="ovo")

    # 使用交叉验证预测决策函数
    preds = cross_val_predict(est, X, y, method="decision_function")
    # 断言预测结果的形状应为 (1797, 45)
    assert preds.shape == (1797, 45)

    # 根据标签排序索引重新排列特征矩阵 X 和标签 y
    ind = np.argsort(y)
    X, y = X[ind], y[ind]
    # 预期的错误消息正则表达式，用于捕获值错误的异常，匹配特定的错误信息
    error_message_regexp = (
        r"Output shape \(599L?, 21L?\) of "
        "decision_function does not match number of "
        r"classes \(7\) in fold. Irregular "
        "decision_function .*"
    )
    # 使用 pytest 检查是否会引发 ValueError 异常，并匹配特定的错误消息
    with pytest.raises(ValueError, match=error_message_regexp):
        cross_val_predict(est, X, y, cv=KFold(n_splits=3), method="decision_function")


# 定义一个用于测试交叉验证预测概率预测形状的函数
def test_cross_val_predict_predict_proba_shape():
    # 创建一个二元分类的样本集 X 和标签 y，包含 50 个样本，使用随机种子 0
    X, y = make_classification(n_classes=2, n_samples=50, random_state=0)

    # 使用交叉验证预测概率预测，采用逻辑回归分类器（solver="liblinear"）
    preds = cross_val_predict(
        LogisticRegression(solver="liblinear"), X, y, method="predict_proba"
    )
    # 断言预测结果的形状应为 (50, 2)
    assert preds.shape == (50, 2)

    # 载入鸢尾花数据集，返回特征矩阵 X 和标签 y
    X, y = load_iris(return_X_y=True)

    # 使用交叉验证预测概率预测，同样采用逻辑回归分类器（solver="liblinear"）
    preds = cross_val_predict(
        LogisticRegression(solver="liblinear"), X, y, method="predict_proba"
    )
    # 断言预测结果的形状应为 (150, 3)
    assert preds.shape == (150, 3)


# 定义一个用于测试交叉验证预测对数概率预测形状的函数
def test_cross_val_predict_predict_log_proba_shape():
    # 创建一个二元分类的样本集 X 和标签 y，包含 50 个样本，使用随机种子 0
    X, y = make_classification(n_classes=2, n_samples=50, random_state=0)

    # 使用交叉验证预测对数概率预测，采用逻辑回归分类器（solver="liblinear"）
    preds = cross_val_predict(
        LogisticRegression(solver="liblinear"), X, y, method="predict_log_proba"
    )
    # 断言预测结果的形状应为 (50, 2)
    assert preds.shape == (50, 2)

    # 载入鸢尾花数据集，返回特征矩阵 X 和标签 y
    X, y = load_iris(return_X_y=True)

    # 使用交叉验证预测对数概率预测，同样采用逻辑回归分类器（solver="liblinear"）
    preds = cross_val_predict(
        LogisticRegression(solver="liblinear"), X, y, method="predict_log_proba"
    )
    # 断言预测结果的形状应为 (150, 3)
    assert preds.shape == (150, 3)


# 使用参数化测试装饰器，用不同的 COO 容器测试交叉验证预测输入类型的函数
@pytest.mark.parametrize("coo_container", COO_CONTAINERS)
def test_cross_val_predict_input_types(coo_container):
    # 载入鸢尾花数据集
    iris = load_iris()
    X, y = iris.data, iris.target
    # 将特征矩阵 X 转换为指定的 COO 容器类型
    X_sparse = coo_container(X)
    # 创建一个多输出标签 y 的数组，包含原始标签和反转后的标签
    multioutput_y = np.column_stack([y, y[::-1]])
    # 创建一个 Ridge 回归器对象，禁用截距项，使用随机种子 0
    clf = Ridge(fit_intercept=False, random_state=0)
    
    # 使用 3 折交叉验证进行预测，确保每个类别至少有 3 个样本
    # 烟雾测试
    predictions = cross_val_predict(clf, X, y)
    # 断言预测结果的形状为 (150,)
    assert predictions.shape == (150,)

    # 使用多输出的 y 进行测试
    predictions = cross_val_predict(clf, X_sparse, multioutput_y)
    # 断言预测结果的形状为 (150, 2)
    assert predictions.shape == (150, 2)

    # 使用稀疏矩阵 X 和普通 y 进行测试
    predictions = cross_val_predict(clf, X_sparse, y)
    # 断言预测结果的形状为 (150,)
    assert_array_equal(predictions.shape, (150,))

    # 再次测试多输出的 y
    predictions = cross_val_predict(clf, X_sparse, multioutput_y)
    # 断言预测结果的形状为 (150, 2)
    assert_array_equal(predictions.shape, (150, 2))

    # 将 X 和 y 转换为列表进行测试
    list_check = lambda x: isinstance(x, list)
    # 使用自定义检查 X 是否为列表的分类器
    clf = CheckingClassifier(check_X=list_check)
    predictions = cross_val_predict(clf, X.tolist(), y.tolist())

    # 使用自定义检查 y 是否为列表的分类器
    clf = CheckingClassifier(check_y=list_check)
    predictions = cross_val_predict(clf, X, y.tolist())

    # 使用列表形式的 X 和 y，并指定非空方法进行决策函数预测
    predictions = cross_val_predict(
        LogisticRegression(solver="liblinear"),
        X.tolist(),
        y.tolist(),
        method="decision_function",
    )
    predictions = cross_val_predict(
        LogisticRegression(solver="liblinear"),
        X,
        y.tolist(),
        method="decision_function",
    )

    # 使用三维的 X 进行测试
    X_3d = X[:, :, np.newaxis]
    # 检查 X 是否为三维的函数
    check_3d = lambda x: x.ndim == 3
    # 使用自定义检查 X 是否为三维的分类器
    clf = CheckingClassifier(check_X=check_3d)
    predictions = cross_val_predict(clf, X_3d, y)
    # 断言预测结果的形状为 (150,)
    assert_array_equal(predictions.shape, (150,))
@pytest.mark.filterwarnings("ignore: Using or importing the ABCs from")
# 忽略 Python 3.7 中 pandas 通过 matplotlib 引发的警告 :-/
def test_cross_val_predict_pandas():
    # 检查 cross_val_score 是否不会破坏 pandas dataframe
    types = [(MockDataFrame, MockDataFrame)]
    try:
        from pandas import DataFrame, Series

        types.append((Series, DataFrame))
    except ImportError:
        pass
    for TargetType, InputFeatureType in types:
        # X 是 dataframe，y 是 series
        X_df, y_ser = InputFeatureType(X), TargetType(y2)
        # 检查函数，用于确认参数类型
        check_df = lambda x: isinstance(x, InputFeatureType)
        check_series = lambda x: isinstance(x, TargetType)
        clf = CheckingClassifier(check_X=check_df, check_y=check_series)
        cross_val_predict(clf, X_df, y_ser, cv=3)


def test_cross_val_predict_unbalanced():
    X, y = make_classification(
        n_samples=100,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        random_state=1,
    )
    # 将第一个样本的类别改为新的类别
    y[0] = 2
    clf = LogisticRegression(random_state=1, solver="liblinear")
    cv = StratifiedKFold(n_splits=2)
    train, test = list(cv.split(X, y))
    yhat_proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")
    # 确认进一步断言的合理性检查
    assert y[test[0]][0] == 2
    assert np.all(yhat_proba[test[0]][:, 2] == 0)
    assert np.all(yhat_proba[test[0]][:, 0:1] > 0)
    assert np.all(yhat_proba[test[1]] > 0)
    assert_array_almost_equal(yhat_proba.sum(axis=1), np.ones(y.shape), decimal=12)


def test_cross_val_predict_y_none():
    # 确保当 y 为 None 时 cross_val_predict 能正常工作
    mock_classifier = MockClassifier()
    rng = np.random.RandomState(42)
    X = rng.rand(100, 10)
    y_hat = cross_val_predict(mock_classifier, X, y=None, cv=5, method="predict")
    assert_allclose(X[:, 0], y_hat)
    y_hat_proba = cross_val_predict(
        mock_classifier, X, y=None, cv=5, method="predict_proba"
    )
    assert_allclose(X, y_hat_proba)


@pytest.mark.parametrize("coo_container", COO_CONTAINERS)
def test_cross_val_score_sparse_fit_params(coo_container):
    iris = load_iris()
    X, y = iris.data, iris.target
    clf = MockClassifier()
    fit_params = {"sparse_sample_weight": coo_container(np.eye(X.shape[0]))}
    a = cross_val_score(clf, X, y, params=fit_params, cv=3)
    assert_array_equal(a, np.ones(3))


def test_learning_curve():
    n_samples = 30
    n_splits = 3
    X, y = make_classification(
        n_samples=n_samples,
        n_features=1,
        n_informative=1,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=0,
    )
    estimator = MockImprovingEstimator(n_samples * ((n_splits - 1) / n_splits))
    # 循环两次，分别设置 shuffle_train 为 False 和 True
    for shuffle_train in [False, True]:
        # 捕获所有警告，记录在变量 w 中
        with warnings.catch_warnings(record=True) as w:
            # 调用 learning_curve 函数进行学习曲线计算
            (
                train_sizes,            # 训练集大小
                train_scores,           # 训练集得分
                test_scores,            # 测试集得分
                fit_times,              # 拟合时间
                score_times,            # 评分时间
            ) = learning_curve(
                estimator,              # 拟合器对象
                X,                      # 输入特征 X
                y,                      # 输出目标 y
                cv=KFold(n_splits=n_splits),   # K 折交叉验证对象
                train_sizes=np.linspace(0.1, 1.0, 10),   # 训练集大小比例数组
                shuffle=shuffle_train,  # 是否打乱数据
                return_times=True,      # 返回拟合和评分时间
            )
        # 如果有警告产生，抛出 RuntimeError 异常
        if len(w) > 0:
            raise RuntimeError("Unexpected warning: %r" % w[0].message)
        # 断言各数组的形状是否符合预期
        assert train_scores.shape == (10, 3)
        assert test_scores.shape == (10, 3)
        assert fit_times.shape == (10, 3)
        assert score_times.shape == (10, 3)
        # 断言训练集大小数组与期望值数组相等
        assert_array_equal(train_sizes, np.linspace(2, 20, 10))
        # 断言训练集平均得分与期望值数组相近
        assert_array_almost_equal(train_scores.mean(axis=1), np.linspace(1.9, 1.0, 10))
        # 断言测试集平均得分与期望值数组相近
        assert_array_almost_equal(test_scores.mean(axis=1), np.linspace(0.1, 1.0, 10))

        # 由于拟合和评分时间依赖硬件，不能使用 assert_array_almost_equal 进行断言
        assert fit_times.dtype == "float64"
        assert score_times.dtype == "float64"

        # 测试一个自定义的交叉验证分离器，只能迭代一次
        with warnings.catch_warnings(record=True) as w:
            # 使用 OneTimeSplitter 进行学习曲线计算
            train_sizes2, train_scores2, test_scores2 = learning_curve(
                estimator,              # 拟合器对象
                X,                      # 输入特征 X
                y,                      # 输出目标 y
                cv=OneTimeSplitter(n_splits=n_splits, n_samples=n_samples),  # 自定义分离器
                train_sizes=np.linspace(0.1, 1.0, 10),   # 训练集大小比例数组
                shuffle=shuffle_train,  # 是否打乱数据
            )
        # 如果有警告产生，抛出 RuntimeError 异常
        if len(w) > 0:
            raise RuntimeError("Unexpected warning: %r" % w[0].message)
        # 断言两个学习曲线的训练得分和测试得分数组几乎相等
        assert_array_almost_equal(train_scores2, train_scores)
        assert_array_almost_equal(test_scores2, test_scores)
# 测试学习曲线函数，用于无监督学习场景
def test_learning_curve_unsupervised():
    # 创建一个具有特定特征的分类数据集，但不返回目标值
    X, _ = make_classification(
        n_samples=30,
        n_features=1,
        n_informative=1,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=0,
    )
    # 创建一个模拟的改进估计器对象，它可以处理20个训练步骤
    estimator = MockImprovingEstimator(20)
    # 计算学习曲线的训练样本大小、训练分数和测试分数
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y=None, cv=3, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    # 断言训练样本大小的数组与预期的线性空间相等
    assert_array_equal(train_sizes, np.linspace(2, 20, 10))
    # 断言训练分数的平均值数组与预期的线性空间近似相等
    assert_array_almost_equal(train_scores.mean(axis=1), np.linspace(1.9, 1.0, 10))
    # 断言测试分数的平均值数组与预期的线性空间近似相等
    assert_array_almost_equal(test_scores.mean(axis=1), np.linspace(0.1, 1.0, 10))


# 测试详细模式下的学习曲线函数
def test_learning_curve_verbose():
    # 创建一个具有特定特征和目标值的分类数据集
    X, y = make_classification(
        n_samples=30,
        n_features=1,
        n_informative=1,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=0,
    )
    # 创建一个模拟的改进估计器对象，它可以处理20个训练步骤
    estimator = MockImprovingEstimator(20)

    # 重定向标准输出到字符串缓冲区
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        # 执行学习曲线计算，输出详细信息
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=3, verbose=1
        )
    finally:
        # 恢复标准输出，获取缓冲区的内容
        out = sys.stdout.getvalue()
        sys.stdout.close()
        sys.stdout = old_stdout

    # 断言输出中包含学习曲线的标识字符串
    assert "[learning_curve]" in out


# 测试增量学习不可能的情况下的学习曲线函数
def test_learning_curve_incremental_learning_not_possible():
    # 创建一个小规模的分类数据集
    X, y = make_classification(
        n_samples=2,
        n_features=1,
        n_informative=1,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=0,
    )
    # 创建一个不支持增量学习的模拟改进估计器对象
    estimator = MockImprovingEstimator(1)
    # 断言调用学习曲线函数时会引发值错误异常
    with pytest.raises(ValueError):
        learning_curve(estimator, X, y, exploit_incremental_learning=True)


# 测试增量学习情况下的学习曲线函数
def test_learning_curve_incremental_learning():
    # 创建一个具有特定特征和目标值的分类数据集
    X, y = make_classification(
        n_samples=30,
        n_features=1,
        n_informative=1,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=0,
    )
    # 创建一个模拟的增量改进估计器对象，它可以处理20个训练步骤
    estimator = MockIncrementalImprovingEstimator(20)
    # 对于每种训练洗牌方式，执行增量学习曲线计算
    for shuffle_train in [False, True]:
        train_sizes, train_scores, test_scores = learning_curve(
            estimator,
            X,
            y,
            cv=3,
            exploit_incremental_learning=True,
            train_sizes=np.linspace(0.1, 1.0, 10),
            shuffle=shuffle_train,
        )
        # 断言训练样本大小的数组与预期的线性空间相等
        assert_array_equal(train_sizes, np.linspace(2, 20, 10))
        # 断言训练分数的平均值数组与预期的线性空间近似相等
        assert_array_almost_equal(train_scores.mean(axis=1), np.linspace(1.9, 1.0, 10))
        # 断言测试分数的平均值数组与预期的线性空间近似相等
        assert_array_almost_equal(test_scores.mean(axis=1), np.linspace(0.1, 1.0, 10))


# 测试无监督学习场景下的增量学习曲线函数
def test_learning_curve_incremental_learning_unsupervised():
    # 创建一个具有特定特征的分类数据集，但不返回目标值
    X, _ = make_classification(
        n_samples=30,
        n_features=1,
        n_informative=1,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=0,
    )
    # 创建一个模拟的增量改进估计器对象，它可以处理20个训练步骤
    estimator = MockIncrementalImprovingEstimator(20)
    # 使用 learning_curve 函数生成学习曲线数据
    train_sizes, train_scores, test_scores = learning_curve(
        estimator,                           # 机器学习模型估计器，用于拟合数据
        X,                                   # 特征数据集
        y=None,                              # 目标数据集，此处为无监督学习
        cv=3,                                # 交叉验证折数
        exploit_incremental_learning=True,   # 是否利用增量学习的优化方式
        train_sizes=np.linspace(0.1, 1.0, 10), # 训练样本集的大小，逐步增加从10%到100%
    )
    
    # 检查训练样本大小数组是否与预期的线性空间范围相等
    assert_array_equal(train_sizes, np.linspace(2, 20, 10))
    
    # 检查训练分数的均值是否接近于从1.9到1.0的线性空间范围
    assert_array_almost_equal(train_scores.mean(axis=1), np.linspace(1.9, 1.0, 10))
    
    # 检查测试分数的均值是否接近于从0.1到1.0的线性空间范围
    assert_array_almost_equal(test_scores.mean(axis=1), np.linspace(0.1, 1.0, 10))
# 测试批量学习和增量学习的结果是否相等
def test_learning_curve_batch_and_incremental_learning_are_equal():
    # 生成具有特定特征的分类数据集
    X, y = make_classification(
        n_samples=30,
        n_features=1,
        n_informative=1,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=0,
    )
    # 定义训练集大小的数组
    train_sizes = np.linspace(0.2, 1.0, 5)
    # 创建被动攻击分类器的实例
    estimator = PassiveAggressiveClassifier(max_iter=1, tol=None, shuffle=False)

    # 执行增量学习曲线分析
    train_sizes_inc, train_scores_inc, test_scores_inc = learning_curve(
        estimator,
        X,
        y,
        train_sizes=train_sizes,
        cv=3,
        exploit_incremental_learning=True,
    )
    # 执行批量学习曲线分析
    train_sizes_batch, train_scores_batch, test_scores_batch = learning_curve(
        estimator,
        X,
        y,
        cv=3,
        train_sizes=train_sizes,
        exploit_incremental_learning=False,
    )

    # 断言增量学习和批量学习的训练集大小数组相等
    assert_array_equal(train_sizes_inc, train_sizes_batch)
    # 断言增量学习和批量学习的平均训练分数数组近似相等
    assert_array_almost_equal(
        train_scores_inc.mean(axis=1), train_scores_batch.mean(axis=1)
    )
    # 断言增量学习和批量学习的平均测试分数数组近似相等
    assert_array_almost_equal(
        test_scores_inc.mean(axis=1), test_scores_batch.mean(axis=1)
    )


# 测试学习曲线在样本范围超出边界时是否引发异常
def test_learning_curve_n_sample_range_out_of_bounds():
    # 生成具有特定特征的分类数据集
    X, y = make_classification(
        n_samples=30,
        n_features=1,
        n_informative=1,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=0,
    )
    # 创建改进估计器的模拟实例
    estimator = MockImprovingEstimator(20)
    # 使用 pytest 断言引发 ValueError 异常
    with pytest.raises(ValueError):
        learning_curve(estimator, X, y, cv=3, train_sizes=[0, 1])
    with pytest.raises(ValueError):
        learning_curve(estimator, X, y, cv=3, train_sizes=[0.0, 1.0])
    with pytest.raises(ValueError):
        learning_curve(estimator, X, y, cv=3, train_sizes=[0.1, 1.1])
    with pytest.raises(ValueError):
        learning_curve(estimator, X, y, cv=3, train_sizes=[0, 20])
    with pytest.raises(ValueError):
        learning_curve(estimator, X, y, cv=3, train_sizes=[1, 21])


# 测试学习曲线在删除重复样本大小时的行为
def test_learning_curve_remove_duplicate_sample_sizes():
    # 生成具有特定特征的分类数据集
    X, y = make_classification(
        n_samples=3,
        n_features=1,
        n_informative=1,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=0,
    )
    # 创建改进估计器的模拟实例
    estimator = MockImprovingEstimator(2)
    # 设置警告消息内容
    warning_message = (
        "Removed duplicate entries from 'train_sizes'. Number of ticks "
        "will be less than the size of 'train_sizes': 2 instead of 3."
    )
    # 使用 pytest 断言捕获 RuntimeWarning，并匹配警告消息内容
    with pytest.warns(RuntimeWarning, match=warning_message):
        train_sizes, _, _ = learning_curve(
            estimator, X, y, cv=3, train_sizes=np.linspace(0.33, 1.0, 3)
        )
    # 断言处理后的训练集大小数组与预期值相等
    assert_array_equal(train_sizes, [1, 2])


# 测试带布尔索引的学习曲线行为
def test_learning_curve_with_boolean_indices():
    # 生成具有特定特征的分类数据集
    X, y = make_classification(
        n_samples=30,
        n_features=1,
        n_informative=1,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=0,
    )
    # 创建改进估计器的模拟实例
    estimator = MockImprovingEstimator(20)
    # 创建 KFold 交叉验证对象
    cv = KFold(n_splits=3)
    # 使用学习曲线函数 `learning_curve` 计算训练集大小、训练分数和测试分数
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    # 断言训练集大小与指定的线性空间均匀分布相等，用于验证学习曲线的训练集大小
    assert_array_equal(train_sizes, np.linspace(2, 20, 10))
    
    # 断言训练分数的均值按行（每个训练集大小）近似等于指定的线性空间均匀分布，用于验证学习曲线的训练分数
    assert_array_almost_equal(train_scores.mean(axis=1), np.linspace(1.9, 1.0, 10))
    
    # 断言测试分数的均值按行（每个训练集大小）近似等于指定的线性空间均匀分布，用于验证学习曲线的测试分数
    assert_array_almost_equal(test_scores.mean(axis=1), np.linspace(0.1, 1.0, 10))
def test_learning_curve_with_shuffle():
    # Following test case was designed this way to verify the code
    # changes made in pull request: #7506.
    
    # 构造输入特征矩阵 X
    X = np.array(
        [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [11, 12],
            [13, 14],
            [15, 16],
            [17, 18],
            [19, 20],
            [7, 8],
            [9, 10],
            [11, 12],
            [13, 14],
            [15, 16],
            [17, 18],
        ]
    )
    
    # 构造目标变量 y
    y = np.array([1, 1, 1, 2, 3, 4, 1, 1, 2, 3, 4, 1, 2, 3, 4])
    
    # 构造分组标识 groups
    groups = np.array([1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 4, 4, 4, 4])
    
    # 初始化一个 PassiveAggressiveClassifier 分类器对象
    # 设置最大迭代次数为 5，tol 参数为 None，shuffle 参数为 False
    estimator = PassiveAggressiveClassifier(max_iter=5, tol=None, shuffle=False)

    # 使用 GroupKFold 进行交叉验证，将数据集分为 2 折
    cv = GroupKFold(n_splits=2)
    
    # 调用 learning_curve 函数进行学习曲线分析
    train_sizes_batch, train_scores_batch, test_scores_batch = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=1,
        train_sizes=np.linspace(0.3, 1.0, 3),
        groups=groups,
        shuffle=True,
        random_state=2,
    )
    
    # 断言训练集分数的平均值
    assert_array_almost_equal(
        train_scores_batch.mean(axis=1), np.array([0.75, 0.3, 0.36111111])
    )
    
    # 断言测试集分数的平均值
    assert_array_almost_equal(
        test_scores_batch.mean(axis=1), np.array([0.36111111, 0.25, 0.25])
    )
    
    # 使用 pytest 检测学习曲线函数对错误的处理
    with pytest.raises(ValueError):
        learning_curve(
            estimator,
            X,
            y,
            cv=cv,
            n_jobs=1,
            train_sizes=np.linspace(0.3, 1.0, 3),
            groups=groups,
            error_score="raise",
        )

    # 调用 learning_curve 函数进行增量学习曲线分析
    train_sizes_inc, train_scores_inc, test_scores_inc = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=1,
        train_sizes=np.linspace(0.3, 1.0, 3),
        groups=groups,
        shuffle=True,
        random_state=2,
        exploit_incremental_learning=True,
    )
    
    # 断言增量学习曲线的训练集分数与普通学习曲线的训练集分数的平均值相等
    assert_array_almost_equal(
        train_scores_inc.mean(axis=1), train_scores_batch.mean(axis=1)
    )
    
    # 断言增量学习曲线的测试集分数与普通学习曲线的测试集分数的平均值相等
    assert_array_almost_equal(
        test_scores_inc.mean(axis=1), test_scores_batch.mean(axis=1)
    )


def test_learning_curve_params():
    # 构造输入特征矩阵 X
    X = np.arange(100).reshape(10, 10)
    
    # 构造目标变量 y
    y = np.array([0] * 5 + [1] * 5)
    
    # 初始化一个 CheckingClassifier 分类器对象
    # 设置 expected_sample_weight 参数为 True
    clf = CheckingClassifier(expected_sample_weight=True)

    # 使用 pytest 检测学习曲线函数对错误的处理
    err_msg = r"Expected sample_weight to be passed"
    with pytest.raises(AssertionError, match=err_msg):
        learning_curve(clf, X, y, error_score="raise")

    # 使用 pytest 检测学习曲线函数对错误的处理
    err_msg = r"sample_weight.shape == \(1,\), expected \(2,\)!"
    with pytest.raises(ValueError, match=err_msg):
        learning_curve(
            clf, X, y, error_score="raise", params={"sample_weight": np.ones(1)}
        )
    
    # 调用 learning_curve 函数进行学习曲线分析
    learning_curve(
        clf, X, y, error_score="raise", params={"sample_weight": np.ones(10)}
    )
    # 使用 make_classification 函数生成一个二分类问题的数据集，包括30个样本和1个特征
    X, y = make_classification(
        n_samples=30,
        n_features=1,
        n_informative=1,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=0,
    )
    
    # 创建一个 MockIncrementalImprovingEstimator 实例，设置参数为 20 和 ["sample_weight"]
    estimator = MockIncrementalImprovingEstimator(20, ["sample_weight"])
    
    # 定义错误消息，用于检查是否抛出预期的 AssertionError 异常
    err_msg = r"Expected fit parameter\(s\) \['sample_weight'\] not seen."
    
    # 使用 pytest.raises 检查是否抛出 AssertionError，并匹配指定的错误消息
    with pytest.raises(AssertionError, match=err_msg):
        # 调用 learning_curve 函数，验证是否抛出预期异常
        learning_curve(
            estimator,
            X,
            y,
            cv=3,
            exploit_incremental_learning=True,
            train_sizes=np.linspace(0.1, 1.0, 10),
            error_score="raise",
        )
    
    # 修改错误消息内容，用于检查另一种异常情况
    err_msg = "Fit parameter sample_weight has length 3; expected"
    
    # 再次使用 pytest.raises 检查是否抛出 AssertionError，并匹配新的错误消息
    with pytest.raises(AssertionError, match=err_msg):
        # 再次调用 learning_curve 函数，验证是否抛出预期异常，这次传入了一个带有 sample_weight 参数的 params 字典
        learning_curve(
            estimator,
            X,
            y,
            cv=3,
            exploit_incremental_learning=True,
            train_sizes=np.linspace(0.1, 1.0, 10),
            error_score="raise",
            params={"sample_weight": np.ones(3)},
        )
    
    # 最后一次调用 learning_curve 函数，这次传入了符合预期的 sample_weight 参数长度为 2
    learning_curve(
        estimator,
        X,
        y,
        cv=3,
        exploit_incremental_learning=True,
        train_sizes=np.linspace(0.1, 1.0, 10),
        error_score="raise",
        params={"sample_weight": np.ones(2)},
    )
# 定义一个测试函数，用于验证 validation_curve 函数的行为是否符合预期
def test_validation_curve():
    # 生成一个简单的分类数据集 X 和 y
    X, y = make_classification(
        n_samples=2,              # 样本数量为 2
        n_features=1,             # 特征数量为 1
        n_informative=1,          # 有信息特征数量为 1
        n_redundant=0,            # 冗余特征数量为 0
        n_classes=2,              # 类别数量为 2
        n_clusters_per_class=1,   # 每个类别中簇的数量为 1
        random_state=0,           # 随机种子设为 0
    )
    # 设置参数范围为从 0 到 1 的均匀分布的 10 个数值
    param_range = np.linspace(0, 1, 10)
    # 捕获所有警告
    with warnings.catch_warnings(record=True) as w:
        # 执行 validation_curve 函数，返回训练分数和测试分数
        train_scores, test_scores = validation_curve(
            MockEstimatorWithParameter(),  # 使用 MockEstimatorWithParameter 模拟估计器
            X,                             # 输入数据 X
            y,                             # 输出数据 y
            param_name="param",             # 参数名称为 "param"
            param_range=param_range,        # 参数范围为之前定义的 param_range
            cv=2,                          # 使用 2 折交叉验证
        )
    # 如果捕获到任何警告，则抛出 RuntimeError 异常
    if len(w) > 0:
        raise RuntimeError("Unexpected warning: %r" % w[0].message)

    # 断言训练分数的均值沿轴 1 的近似值等于 param_range
    assert_array_almost_equal(train_scores.mean(axis=1), param_range)
    # 断言测试分数的均值沿轴 1 的近似值等于 1 减去 param_range
    assert_array_almost_equal(test_scores.mean(axis=1), 1 - param_range)


# 定义测试函数，用于验证 validation_curve 函数在克隆估计器上的行为
def test_validation_curve_clone_estimator():
    # 生成一个简单的分类数据集 X 和 y
    X, y = make_classification(
        n_samples=2,              # 样本数量为 2
        n_features=1,             # 特征数量为 1
        n_informative=1,          # 有信息特征数量为 1
        n_redundant=0,            # 冗余特征数量为 0
        n_classes=2,              # 类别数量为 2
        n_clusters_per_class=1,   # 每个类别中簇的数量为 1
        random_state=0,           # 随机种子设为 0
    )
    # 设置参数范围为从 1 到 0 的均匀分布的 10 个数值
    param_range = np.linspace(1, 0, 10)
    # 调用 validation_curve 函数，但不关心返回值
    _, _ = validation_curve(
        MockEstimatorWithSingleFitCallAllowed(),  # 使用 MockEstimatorWithSingleFitCallAllowed 模拟估计器
        X,                                       # 输入数据 X
        y,                                       # 输出数据 y
        param_name="param",                       # 参数名称为 "param"
        param_range=param_range,                  # 参数范围为之前定义的 param_range
        cv=2,                                     # 使用 2 折交叉验证
    )


# 定义测试函数，用于验证 validation_curve 函数在交叉验证分割的一致性上的行为
def test_validation_curve_cv_splits_consistency():
    n_samples = 100
    n_splits = 5
    # 生成一个复杂的分类数据集 X 和 y
    X, y = make_classification(n_samples=100, random_state=0)
    
    # 使用自定义的 OneTimeSplitter 对象进行验证曲线
    scores1 = validation_curve(
        SVC(kernel="linear", random_state=0),   # 使用线性核的支持向量分类器
        X,                                     # 输入数据 X
        y,                                     # 输出数据 y
        param_name="C",                        # 参数名称为 "C"
        param_range=[0.1, 0.1, 0.2, 0.2],       # 参数范围为指定的列表
        cv=OneTimeSplitter(n_splits=n_splits, n_samples=n_samples),  # 自定义的分割器对象
    )
    # 断言 scores1 的分数在参数设置 1 和 2 上的一致性
    assert_array_almost_equal(*np.vsplit(np.hstack(scores1)[(0, 2, 1, 3), :], 2))

    # 使用 KFold 对象进行验证曲线
    scores2 = validation_curve(
        SVC(kernel="linear", random_state=0),   # 使用线性核的支持向量分类器
        X,                                     # 输入数据 X
        y,                                     # 输出数据 y
        param_name="C",                        # 参数名称为 "C"
        param_range=[0.1, 0.1, 0.2, 0.2],       # 参数范围为指定的列表
        cv=KFold(n_splits=n_splits, shuffle=True),  # 使用 K 折交叉验证，打乱顺序
    )
    # 断言 scores2 的第一个和第二个参数设置的分数一致性
    assert_array_almost_equal(*np.vsplit(np.hstack(scores2)[(0, 2, 1, 3), :], 2))

    # 再次使用 KFold 对象进行验证曲线
    scores3 = validation_curve(
        SVC(kernel="linear", random_state=0),   # 使用线性核的支持向量分类器
        X,                                     # 输入数据 X
        y,                                     # 输出数据 y
        param_name="C",                        # 参数名称为 "C"
        param_range=[0.1, 0.1, 0.2, 0.2],       # 参数范围为指定的列表
        cv=KFold(n_splits=n_splits),           # 使用 K 折交叉验证
    )
    # 断言 scores3 和 scores1 的分数一致性
    assert_array_almost_equal(np.array(scores3), np.array(scores1))


# 定义测试函数，用于验证 validation_curve 函数在拟合参数上的行为
def test_validation_curve_fit_params():
    # 创建一个简单的数据集 X 和 y
    X = np.arange(100).reshape(10, 10)
    y = np.array([0] * 5 + [1] * 5)
    # 创建一个CheckingClassifier对象clf，该对象期望接收样本权重作为参数
    clf = CheckingClassifier(expected_sample_weight=True)
    
    # 设置错误消息，用于断言检查
    err_msg = r"Expected sample_weight to be passed"
    
    # 使用pytest的raises方法，检查是否会引发AssertionError异常，并匹配错误消息
    with pytest.raises(AssertionError, match=err_msg):
        # 执行validation_curve函数，验证是否会抛出断言错误异常
        validation_curve(
            clf,
            X,
            y,
            param_name="foo_param",
            param_range=[1, 2, 3],
            error_score="raise",
        )
    
    # 设置错误消息，用于值错误检查
    err_msg = r"sample_weight.shape == \(1,\), expected \(8,\)!"
    
    # 使用pytest的raises方法，检查是否会引发ValueError异常，并匹配错误消息
    with pytest.raises(ValueError, match=err_msg):
        # 执行validation_curve函数，验证是否会抛出值错误异常
        validation_curve(
            clf,
            X,
            y,
            param_name="foo_param",
            param_range=[1, 2, 3],
            error_score="raise",
            fit_params={"sample_weight": np.ones(1)},
        )
    
    # 执行validation_curve函数，使用样本权重为10的np.ones数组，进行验证曲线计算
    validation_curve(
        clf,
        X,
        y,
        param_name="foo_param",
        param_range=[1, 2, 3],
        error_score="raise",
        fit_params={"sample_weight": np.ones(10)},
    )
# 定义测试函数，用于检查是否排列
def test_check_is_permutation():
    # 创建随机数生成器，种子为0
    rng = np.random.RandomState(0)
    # 创建包含100个元素的数组 p，顺序为0到99
    p = np.arange(100)
    # 使用随机数生成器打乱数组 p 的顺序
    rng.shuffle(p)
    # 断言：检查函数 _check_is_permutation 是否正确判断 p 是排列
    assert _check_is_permutation(p, 100)
    # 断言：检查函数 _check_is_permutation 是否正确判断从 p 中删除第23个元素后不是排列
    assert not _check_is_permutation(np.delete(p, 23), 100)

    # 修改 p 的第一个元素为 23
    p[0] = 23
    # 断言：检查函数 _check_is_permutation 是否能正确判断 p 不是排列
    assert not _check_is_permutation(p, 100)

    # 检查是否能捕获额外重复索引的情况
    # 断言：检查函数 _check_is_permutation 是否能正确判断包含额外重复索引的数组不是排列
    assert not _check_is_permutation(np.hstack((p, 0)), 100)


# 使用参数化测试框架 pytest.mark.parametrize 对 CSR_CONTAINERS 中的每个容器执行测试
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_cross_val_predict_sparse_prediction(csr_container):
    # 检查 cross_val_predict 是否对稀疏输入和密集输入给出相同的结果
    # 生成多标签分类数据 X 和 y
    X, y = make_multilabel_classification(
        n_classes=2,
        n_labels=1,
        allow_unlabeled=False,
        return_indicator=True,
        random_state=1,
    )
    # 将 X 和 y 转换为稀疏表示
    X_sparse = csr_container(X)
    y_sparse = csr_container(y)
    # 创建一个使用线性核的 OneVsRestClassifier 分类器
    classif = OneVsRestClassifier(SVC(kernel="linear"))
    # 对 X 和 y 进行 10 折交叉验证预测
    preds = cross_val_predict(classif, X, y, cv=10)
    # 对稀疏表示的 X_sparse 和 y_sparse 进行 10 折交叉验证预测
    preds_sparse = cross_val_predict(classif, X_sparse, y_sparse, cv=10)
    # 将稀疏表示的预测结果转换为密集数组
    preds_sparse = preds_sparse.toarray()
    # 断言：检查稀疏表示和密集表示的预测结果是否几乎相等
    assert_array_almost_equal(preds_sparse, preds)


# 辅助函数，用于测试二分类情况下的 cross_val_predict
def check_cross_val_predict_binary(est, X, y, method):
    """Helper for tests of cross_val_predict with binary classification"""
    # 创建 KFold 交叉验证对象，将数据分成 3 折，不打乱数据
    cv = KFold(n_splits=3, shuffle=False)

    # 生成预期输出
    if y.ndim == 1:
        exp_shape = (len(X),) if method == "decision_function" else (len(X), 2)
    else:
        exp_shape = y.shape
    expected_predictions = np.zeros(exp_shape)
    # 遍历交叉验证的训练集和测试集
    for train, test in cv.split(X, y):
        # 复制分类器并在训练集上拟合
        est = clone(est).fit(X[train], y[train])
        # 使用指定方法对测试集进行预测，并将结果存入 expected_predictions
        expected_predictions[test] = getattr(est, method)(X[test])

    # 检查不同表示形式的 y 的实际输出
    for tg in [y, y + 1, y - 2, y.astype("str")]:
        # 断言：检查 cross_val_predict 的预测结果是否与预期输出几乎相等
        assert_allclose(
            cross_val_predict(est, X, tg, method=method, cv=cv), expected_predictions
        )


# 辅助函数，用于测试多类别分类情况下的 cross_val_predict
def check_cross_val_predict_multiclass(est, X, y, method):
    """Helper for tests of cross_val_predict with multiclass classification"""
    # 创建 KFold 交叉验证对象，将数据分成 3 折，不打乱数据
    cv = KFold(n_splits=3, shuffle=False)

    # 生成预期输出
    float_min = np.finfo(np.float64).min
    default_values = {
        "decision_function": float_min,
        "predict_log_proba": float_min,
        "predict_proba": 0,
    }
    expected_predictions = np.full(
        (len(X), len(set(y))), default_values[method], dtype=np.float64
    )
    _, y_enc = np.unique(y, return_inverse=True)
    # 遍历交叉验证的训练集和测试集
    for train, test in cv.split(X, y_enc):
        # 复制分类器并在训练集上拟合
        est = clone(est).fit(X[train], y_enc[train])
        # 使用指定方法对测试集进行预测，并将结果存入 expected_predictions
        fold_preds = getattr(est, method)(X[test])
        i_cols_fit = np.unique(y_enc[train])
        expected_predictions[np.ix_(test, i_cols_fit)] = fold_preds

    # 检查不同表示形式的 y 的实际输出
    for tg in [y, y + 1, y - 2, y.astype("str")]:
        # 断言：检查 cross_val_predict 的预测结果是否与预期输出几乎相等
        assert_allclose(
            cross_val_predict(est, X, tg, method=method, cv=cv), expected_predictions
        )
    """检查 cross_val_predict 对于二维目标输出的结果，
    使用提供每个类别预测作为列表的评估器。
    """
    # 使用 KFold 创建交叉验证对象，分成 3 折，不打乱顺序
    cv = KFold(n_splits=3, shuffle=False)

    # 创建空数组，用于存储输出结果，大小正确
    float_min = np.finfo(np.float64).min  # 获取浮点数最小值
    default_values = {
        "decision_function": float_min,  # 决策函数的默认值设为最小浮点数
        "predict_log_proba": float_min,  # 对数概率预测的默认值设为最小浮点数
        "predict_proba": 0,  # 概率预测的默认值设为0
    }
    n_targets = y.shape[1]  # 目标变量的数量
    expected_preds = []  # 用于存储预期预测结果的列表
    for i_col in range(n_targets):
        n_classes_in_label = len(set(y[:, i_col]))  # 目标变量每列中类别的数量
        if n_classes_in_label == 2 and method == "decision_function":
            exp_shape = (len(X),)  # 如果类别数为2且方法为决策函数，预期形状为(len(X),)
        else:
            exp_shape = (len(X), n_classes_in_label)  # 否则预期形状为(len(X), 类别数)
        expected_preds.append(
            np.full(exp_shape, default_values[method], dtype=np.float64)
        )  # 使用默认值填充预期形状的数组，存入 expected_preds

    # 生成预期输出
    y_enc_cols = [
        np.unique(y[:, i], return_inverse=True)[1][:, np.newaxis]
        for i in range(y.shape[1])
    ]  # 对每列目标变量进行编码转换
    y_enc = np.concatenate(y_enc_cols, axis=1)  # 拼接编码后的结果
    for train, test in cv.split(X, y_enc):
        est = clone(est).fit(X[train], y_enc[train])  # 克隆评估器，拟合训练数据
        fold_preds = getattr(est, method)(X[test])  # 调用评估器的指定方法进行预测
        for i_col in range(n_targets):
            fold_cols = np.unique(y_enc[train][:, i_col])  # 获取训练数据中每列目标变量的唯一值
            if expected_preds[i_col].ndim == 1:
                # 如果是决策函数且类别数小于等于2
                expected_preds[i_col][test] = fold_preds[i_col]  # 直接赋值预测结果
            else:
                idx = np.ix_(test, fold_cols)  # 使用索引函数生成索引
                expected_preds[i_col][idx] = fold_preds[i_col]  # 赋值预测结果到指定位置

    # 检查多种表示的实际输出 y
    for tg in [y, y + 1, y - 2, y.astype("str")]:
        cv_predict_output = cross_val_predict(est, X, tg, method=method, cv=cv)  # 使用交叉验证预测输出
        assert len(cv_predict_output) == len(expected_preds)  # 断言预测输出长度与预期结果长度相等
        for i in range(len(cv_predict_output)):
            assert_allclose(cv_predict_output[i], expected_preds[i])  # 断言预测输出与预期结果的近似性
# 定义一个函数，用于检查二元分类器在不同方法下的交叉验证预测
def check_cross_val_predict_with_method_binary(est):
    # 使用make_classification生成二元分类数据集X和y，random_state设为0确保可重复性
    X, y = make_classification(n_classes=2, random_state=0)
    # 遍历三种方法："decision_function", "predict_proba", "predict_log_proba"
    for method in ["decision_function", "predict_proba", "predict_log_proba"]:
        # 调用check_cross_val_predict_binary函数，对给定的estimator、数据集X、y和方法method进行交叉验证预测
        check_cross_val_predict_binary(est, X, y, method)


# 定义一个函数，用于检查多类分类器在不同方法下的交叉验证预测
def check_cross_val_predict_with_method_multiclass(est):
    # 载入鸢尾花数据集
    iris = load_iris()
    X, y = iris.data, iris.target
    # 对数据集X和y进行随机重排，random_state设为0确保可重复性
    X, y = shuffle(X, y, random_state=0)
    # 遍历三种方法："decision_function", "predict_proba", "predict_log_proba"
    for method in ["decision_function", "predict_proba", "predict_log_proba"]:
        # 调用check_cross_val_predict_multiclass函数，对给定的estimator、数据集X、y和方法method进行交叉验证预测
        check_cross_val_predict_multiclass(est, X, y, method)


# 定义一个函数，用于测试不同分类器方法下的交叉验证预测
def test_cross_val_predict_with_method():
    # 测试二元分类器方法下的交叉验证预测
    check_cross_val_predict_with_method_binary(LogisticRegression(solver="liblinear"))
    # 测试多类分类器方法下的交叉验证预测
    check_cross_val_predict_with_method_multiclass(LogisticRegression(solver="liblinear"))


# 定义一个函数，用于测试cross_val_predict函数在检查estimator方法（如predict_proba）之前是否进行拟合
def test_cross_val_predict_method_checking():
    # Regression test for issue #9639. Tests that cross_val_predict does not
    # check estimator methods (e.g. predict_proba) before fitting
    # 载入鸢尾花数据集
    iris = load_iris()
    X, y = iris.data, iris.target
    # 对数据集X和y进行随机重排，random_state设为0确保可重复性
    X, y = shuffle(X, y, random_state=0)
    # 遍历三种方法："decision_function", "predict_proba", "predict_log_proba"
    for method in ["decision_function", "predict_proba", "predict_log_proba"]:
        # 创建SGDClassifier，loss设为"log_loss"，random_state设为2
        est = SGDClassifier(loss="log_loss", random_state=2)
        # 调用check_cross_val_predict_multiclass函数，对给定的estimator、数据集X、y和方法method进行交叉验证预测
        check_cross_val_predict_multiclass(est, X, y, method)


# 定义一个函数，用于测试GridSearchCV结合不同分类器方法下的交叉验证预测
def test_gridsearchcv_cross_val_predict_with_method():
    # 载入鸢尾花数据集
    iris = load_iris()
    X, y = iris.data, iris.target
    # 对数据集X和y进行随机重排，random_state设为0确保可重复性
    X, y = shuffle(X, y, random_state=0)
    # 创建LogisticRegression分类器，solver设为"liblinear"，random_state设为42
    est = GridSearchCV(
        LogisticRegression(random_state=42, solver="liblinear"), {"C": [0.1, 1]}, cv=2
    )
    # 遍历三种方法："decision_function", "predict_proba", "predict_log_proba"
    for method in ["decision_function", "predict_proba", "predict_log_proba"]:
        # 调用check_cross_val_predict_multiclass函数，对给定的estimator、数据集X、y和方法method进行交叉验证预测
        check_cross_val_predict_multiclass(est, X, y, method)


# 定义一个函数，用于测试多标签分类中OvR策略下的交叉验证预测
def test_cross_val_predict_with_method_multilabel_ovr():
    # OVR does multilabel predictions, but only arrays of
    # binary indicator columns. The output of predict_proba
    # is a 2D array with shape (n_samples, n_classes).
    # 定义样本数n_samp、类别数n_classes
    n_samp = 100
    n_classes = 4
    # 创建多标签分类数据集X和y
    X, y = make_multilabel_classification(
        n_samples=n_samp, n_labels=3, n_classes=n_classes, n_features=5, random_state=42
    )
    # 创建OneVsRestClassifier，内部使用LogisticRegression分类器，solver设为"liblinear"，random_state设为0
    est = OneVsRestClassifier(LogisticRegression(solver="liblinear", random_state=0))
    # 遍历两种方法："predict_proba", "decision_function"
    for method in ["predict_proba", "decision_function"]:
        # 调用check_cross_val_predict_binary函数，对给定的estimator、数据集X、y和方法method进行交叉验证预测
        check_cross_val_predict_binary(est, X, y, method=method)


# 定义一个继承RandomForestClassifier的类，用于测试决策函数方法
class RFWithDecisionFunction(RandomForestClassifier):
    # None of the current multioutput-multiclass estimators have
    # decision function methods. Create a mock decision function
    # to test the cross_val_predict function's handling of this case.
    # 定义一个decision_function方法，用于模拟多输出多类分类器的决策函数
    def decision_function(self, X):
        # 使用predict_proba进行预测
        probs = self.predict_proba(X)
        msg = "This helper should only be used on multioutput-multiclass tasks"
        # 断言probs是一个列表
        assert isinstance(probs, list), msg
        # 如果probs的第二维长度为2，则取其最后一列；否则保持不变
        probs = [p[:, -1] if p.shape[1] == 2 else p for p in probs]
        # 返回处理后的probs
        return probs
# 定义一个函数，用于测试带有多标签随机森林的交叉验证预测
def test_cross_val_predict_with_method_multilabel_rf():
    # RandomForest 允许每个标签中有多个类别。
    # predict_proba 的输出是每个单独标签的 predict_proba 输出的列表。
    n_classes = 4
    # 生成一个多标签分类数据集，包括100个样本，3个标签，每个标签有n_classes个类别，5个特征，随机种子为42
    X, y = make_multilabel_classification(
        n_samples=100, n_labels=3, n_classes=n_classes, n_features=5, random_state=42
    )
    # 将第一列中的三个类别放入第一列
    y[:, 0] += y[:, 1]
    # 遍历不同的方法：["predict_proba", "predict_log_proba", "decision_function"]
    for method in ["predict_proba", "predict_log_proba", "decision_function"]:
        # 创建一个具有决策函数的随机森林估计器，使用5个估计器，随机种子为0
        est = RFWithDecisionFunction(n_estimators=5, random_state=0)
        # 忽略 "RuntimeWarning: divide by zero encountered in log" 警告
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 执行多标签交叉验证预测检查
            check_cross_val_predict_multilabel(est, X, y, method=method)


# 定义一个函数，用于测试带有稀有类别的多类别问题的交叉验证预测
def test_cross_val_predict_with_method_rare_class():
    # 测试一个多类别问题，其中一个类别在一个CV训练集中将缺失。
    rng = np.random.RandomState(0)
    # 生成一个14行10列的正态分布矩阵作为特征数据
    X = rng.normal(0, 1, size=(14, 10))
    # 创建一个包含了一个类别缺失的多类别标签数组
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 3])
    # 创建一个逻辑回归估计器，使用 "liblinear" 求解器
    est = LogisticRegression(solver="liblinear")
    # 遍历不同的方法：["predict_proba", "predict_log_proba", "decision_function"]
    for method in ["predict_proba", "predict_log_proba", "decision_function"]:
        # 忽略有关某个类别示例太少的警告
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 执行多类别交叉验证预测检查
            check_cross_val_predict_multiclass(est, X, y, method)


# 定义一个函数，用于测试带有稀有类别和多标签随机森林的交叉验证预测
def test_cross_val_predict_with_method_multilabel_rf_rare_class():
    # RandomForest 允许标签中的任何内容。
    # predict_proba 的输出是每个单独标签的 predict_proba 输出的列表。
    # 在这个测试中，第一个标签中有一个只有一个示例的类别。
    # 我们将有一个CV折叠，其中训练数据不包括它。
    rng = np.random.RandomState(0)
    # 生成一个5行10列的正态分布矩阵作为特征数据
    X = rng.normal(0, 1, size=(5, 10))
    # 创建一个包含多标签的多标签标签数组
    y = np.array([[0, 0], [1, 1], [2, 1], [0, 1], [1, 0]])
    # 遍历不同的方法：["predict_proba", "predict_log_proba"]
    for method in ["predict_proba", "predict_log_proba"]:
        # 创建一个具有决策函数的随机森林估计器，使用5个估计器，随机种子为0
        est = RFWithDecisionFunction(n_estimators=5, random_state=0)
        # 忽略 "RuntimeWarning: divide by zero encountered in log" 警告
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 执行多标签交叉验证预测检查
            check_cross_val_predict_multilabel(est, X, y, method=method)


# 定义一个函数，用于获取预期的预测结果
def get_expected_predictions(X, y, cv, classes, est, method):
    # 创建一个空的预期预测数组，形状为 [y的长度, 类别数]
    expected_predictions = np.zeros([len(y), classes])
    # 获取估计器中的方法函数（如 predict_proba、predict_log_proba 或 decision_function）
    func = getattr(est, method)

    # 遍历交叉验证的每一折
    for train, test in cv.split(X, y):
        # 使用训练集训练估计器
        est.fit(X[train], y[train])
        # 获取测试集上的预期预测
        expected_predictions_ = func(X[test])
        # 为了避免二维索引，根据方法选择适当的空数组
        if method == "predict_proba":
            exp_pred_test = np.zeros((len(test), classes))
        else:
            exp_pred_test = np.full(
                (len(test), classes), np.finfo(expected_predictions.dtype).min
            )
        # 将预期预测放入正确的位置
        exp_pred_test[:, est.classes_] = expected_predictions_
        expected_predictions[test] = exp_pred_test
    # 返回函数中的预期预测结果
    return expected_predictions
# 定义测试函数 test_cross_val_predict_class_subset
def test_cross_val_predict_class_subset():
    # 创建包含数字 0 到 199 的数组 X，重塑为 100 行 2 列的数组
    X = np.arange(200).reshape(100, 2)
    # 创建包含 0 到 9 的数组 y，以便表示每个样本所属的类别
    y = np.array([x // 10 for x in range(100)])
    # 类别总数设定为 10
    classes = 10

    # 创建 KFold 对象，将数据集分为 3 折
    kfold3 = KFold(n_splits=3)
    # 创建 KFold 对象，将数据集分为 4 折
    kfold4 = KFold(n_splits=4)

    # 创建 LabelEncoder 对象
    le = LabelEncoder()

    # 定义需要测试的方法列表
    methods = ["decision_function", "predict_proba", "predict_log_proba"]
    # 遍历每种方法
    for method in methods:
        # 创建逻辑回归分类器对象 est，指定求解器为 "liblinear"
        est = LogisticRegression(solver="liblinear")

        # 使用交叉验证预测方法 method 进行交叉验证，使用 3 折交叉验证
        predictions = cross_val_predict(est, X, y, method=method, cv=kfold3)
        # 获得预期的预测结果
        expected_predictions = get_expected_predictions(
            X, y, kfold3, classes, est, method
        )
        # 断言预测结果准确性
        assert_array_almost_equal(expected_predictions, predictions)

        # 使用交叉验证预测方法 method 进行交叉验证，使用 4 折交叉验证
        predictions = cross_val_predict(est, X, y, method=method, cv=kfold4)
        # 获得预期的预测结果
        expected_predictions = get_expected_predictions(
            X, y, kfold4, classes, est, method
        )
        # 断言预测结果准确性
        assert_array_almost_equal(expected_predictions, predictions)

        # 打乱标签 y 的顺序
        y = shuffle(np.repeat(range(10), 10), random_state=0)
        # 使用交叉验证预测方法 method 进行交叉验证，使用 3 折交叉验证
        predictions = cross_val_predict(est, X, y, method=method, cv=kfold3)
        # 对标签 y 进行编码
        y = le.fit_transform(y)
        # 获得预期的预测结果
        expected_predictions = get_expected_predictions(
            X, y, kfold3, classes, est, method
        )
        # 断言预测结果准确性
        assert_array_almost_equal(expected_predictions, predictions)


# 定义测试函数 test_score_memmap
def test_score_memmap():
    # 确保接受 memmap 类型的标量分数
    iris = load_iris()
    X, y = iris.data, iris.target
    # 创建 MockClassifier 对象 clf
    clf = MockClassifier()
    # 创建临时文件对象 tf，并写入数据
    tf = tempfile.NamedTemporaryFile(mode="wb", delete=False)
    tf.write(b"Hello world!!!!!")
    tf.close()
    # 创建 memmap 数组 scores，与临时文件关联，数据类型为 np.float64
    scores = np.memmap(tf.name, dtype=np.float64)
    # 创建 memmap 标量 score，与临时文件关联，数据类型为 np.float64
    score = np.memmap(tf.name, shape=(), mode="r", dtype=np.float64)
    try:
        # 使用 score 作为评分函数进行交叉验证
        cross_val_score(clf, X, y, scoring=lambda est, X, y: score)
        # 使用 scores 作为评分函数进行交叉验证，预期引发 ValueError 异常
        with pytest.raises(ValueError):
            cross_val_score(clf, X, y, scoring=lambda est, X, y: scores)
    finally:
        # 尽最大努力在删除 Windows 下的后备文件之前释放 mmap 文件句柄
        scores, score = None, None
        for _ in range(3):
            try:
                # 尝试删除临时文件 tf 的文件名
                os.unlink(tf.name)
                break
            except OSError:
                # 如果删除失败，则等待 1.0 秒钟
                sleep(1.0)


# 标记 pytest.filterwarnings，忽略特定警告消息
@pytest.mark.filterwarnings("ignore: Using or importing the ABCs from")
# 定义测试函数 test_permutation_test_score_pandas
def test_permutation_test_score_pandas():
    # 检查 permutation_test_score 是否不会破坏 pandas 数据帧
    types = [(MockDataFrame, MockDataFrame)]
    try:
        # 尝试导入 pandas 的 DataFrame 和 Series
        from pandas import DataFrame, Series

        # 向 types 列表添加 Series 和 DataFrame 类型的元组
        types.append((Series, DataFrame))
    except ImportError:
        # 如果导入失败，则忽略
        pass
    # 对 types 列表中的每个元组进行迭代，每个元组的元素分别赋值给 TargetType 和 InputFeatureType
    for TargetType, InputFeatureType in types:
        # 从 sklearn.datasets 加载鸢尾花数据集 iris
        iris = load_iris()
        # 将 iris 数据集中的特征数据 X 和标签数据 y 分别赋值给 X 和 y
        X, y = iris.data, iris.target
        # 使用 InputFeatureType 对 X 进行转换，将结果赋值给 X_df
        X_df, y_ser = InputFeatureType(X), TargetType(y)
        # 定义用于检查 DataFrame 的函数 check_df 和检查 Series 的函数 check_series
        check_df = lambda x: isinstance(x, InputFeatureType)
        check_series = lambda x: isinstance(x, TargetType)
        # 创建 CheckingClassifier 类的实例 clf，传入 check_df 和 check_series 作为参数
        clf = CheckingClassifier(check_X=check_df, check_y=check_series)
        # 对 clf 进行排列测试评分，使用 X_df 作为特征数据，y_ser 作为标签数据
        permutation_test_score(clf, X_df, y_ser)
def test_fit_and_score_failing():
    # 创建一个故意失败的分类器
    failing_clf = FailingClassifier(FailingClassifier.FAILING_PARAMETER)
    # dummy X data
    X = np.arange(1, 10)
    train, test = np.arange(0, 5), np.arange(5, 9)
    fit_and_score_args = dict(
        estimator=failing_clf,
        X=X,
        y=None,
        scorer=dict(),
        train=train,
        test=test,
        verbose=0,
        parameters=None,
        fit_params=None,
        score_params=None,
    )
    # 设置error_score为"raise"，触发警告消息
    fit_and_score_args["error_score"] = "raise"
    # 检查是否引发异常，默认error_score='raise'
    with pytest.raises(ValueError, match="Failing classifier failed as required"):
        _fit_and_score(**fit_and_score_args)

    assert failing_clf.score() == 0.0  # FailingClassifier覆盖率


def test_fit_and_score_working():
    X, y = make_classification(n_samples=30, random_state=0)
    clf = SVC(kernel="linear", random_state=0)
    train, test = next(ShuffleSplit().split(X))
    # 测试return_parameters选项
    fit_and_score_args = dict(
        estimator=clf,
        X=X,
        y=y,
        scorer=dict(),
        train=train,
        test=test,
        verbose=0,
        parameters={"max_iter": 100, "tol": 0.1},
        fit_params=None,
        score_params=None,
        return_parameters=True,
    )
    result = _fit_and_score(**fit_and_score_args)
    assert result["parameters"] == fit_and_score_args["parameters"]


class DataDependentFailingClassifier(BaseEstimator):
    def __init__(self, max_x_value=None):
        self.max_x_value = max_x_value

    def fit(self, X, y=None):
        num_values_too_high = (X > self.max_x_value).sum()
        if num_values_too_high:
            raise ValueError(
                f"Classifier fit failed with {num_values_too_high} values too high"
            )

    def score(self, X=None, Y=None):
        return 0.0


@pytest.mark.parametrize("error_score", [np.nan, 0])
def test_cross_validate_some_failing_fits_warning(error_score):
    # 创建一个故意失败的分类器
    failing_clf = DataDependentFailingClassifier(max_x_value=8)
    # dummy X data
    X = np.arange(1, 10)
    y = np.ones(9)
    # passing error score to trigger the warning message
    cross_validate_args = [failing_clf, X, y]
    cross_validate_kwargs = {"cv": 3, "error_score": error_score}
    # 检查警告消息类型是否符合预期

    individual_fit_error_message = (
        "ValueError: Classifier fit failed with 1 values too high"
    )
    warning_message = re.compile(
        (
            "2 fits failed.+total of 3.+The score on these"
            " train-test partitions for these parameters will be set to"
            f" {cross_validate_kwargs['error_score']}.+{individual_fit_error_message}"
        ),
        flags=re.DOTALL,
    )
    # 使用 pytest 模块中的 `warns` 函数来捕获特定类型的警告，确保会触发 FitFailedWarning 类型的警告，并且警告消息与指定的 warning_message 匹配
    with pytest.warns(FitFailedWarning, match=warning_message):
        # 执行交叉验证的函数调用，传入参数列表 `cross_validate_args` 和关键字参数 `cross_validate_kwargs`
        cross_validate(*cross_validate_args, **cross_validate_kwargs)
# 使用 pytest 的 parametrize 装饰器，定义多组参数化测试用例，每个测试用例都会单独运行一次
@pytest.mark.parametrize("error_score", [np.nan, 0])
# 定义一个测试函数，测试所有失败的拟合是否会引发错误
def test_cross_validate_all_failing_fits_error(error_score):
    # 创建一个故意会失败的分类器实例
    failing_clf = FailingClassifier(FailingClassifier.FAILING_PARAMETER)
    # 创建虚拟的 X 数据
    X = np.arange(1, 10)
    # 创建虚拟的 y 数据，全为 1
    y = np.ones(9)

    # 将参数化的参数放入列表中
    cross_validate_args = [failing_clf, X, y]
    # 将参数化的关键字参数放入字典中
    cross_validate_kwargs = {"cv": 7, "error_score": error_score}

    # 定义单个拟合失败时的错误消息
    individual_fit_error_message = "ValueError: Failing classifier failed as required"
    # 编译正则表达式，用于匹配包含特定错误消息的完整错误信息
    error_message = re.compile(
        (
            "All the 7 fits failed.+your model is misconfigured.+"
            f"{individual_fit_error_message}"
        ),
        flags=re.DOTALL,
    )

    # 使用 pytest 的上下文管理器，检查是否引发预期的 ValueError 异常，并且其错误信息匹配预定义的正则表达式
    with pytest.raises(ValueError, match=error_message):
        cross_validate(*cross_validate_args, **cross_validate_kwargs)


# 定义一个私有函数 _failing_scorer，用于引发 ValueError 异常
def _failing_scorer(estimator, X, y, error_msg):
    raise ValueError(error_msg)


# 使用 pytest 的 filterwarnings 装饰器，忽略特定警告信息
@pytest.mark.filterwarnings("ignore:lbfgs failed to converge")
# 使用 pytest 的 parametrize 装饰器，定义多组参数化测试用例
@pytest.mark.parametrize("error_score", [np.nan, 0, "raise"])
# 定义一个测试函数，测试在 cross_val_score 中使用失败的评分器
def test_cross_val_score_failing_scorer(error_score):
    # 检查在 cross_val_score 中评分过程中是否能够处理评分器失败，并可选地用 error_score 替换
    X, y = load_iris(return_X_y=True)
    # 创建逻辑回归分类器的实例，并拟合数据
    clf = LogisticRegression(max_iter=5).fit(X, y)

    # 定义用于失败评分器的错误消息
    error_msg = "This scorer is supposed to fail!!!"
    # 创建一个部分函数，将失败的评分器与特定错误消息绑定在一起
    failing_scorer = partial(_failing_scorer, error_msg=error_msg)

    # 根据 error_score 的不同取值，使用 pytest 的上下文管理器进行不同的测试
    if error_score == "raise":
        # 检查是否引发预期的 ValueError 异常，并且其错误消息匹配预定义的消息
        with pytest.raises(ValueError, match=error_msg):
            cross_val_score(
                clf, X, y, cv=3, scoring=failing_scorer, error_score=error_score
            )
    else:
        # 创建警告消息，用于指示评分过程失败，并且将分数设置为 error_score
        warning_msg = (
            "Scoring failed. The score on this train-test partition for "
            f"these parameters will be set to {error_score}"
        )
        # 检查是否引发 UserWarning 警告，并且其消息匹配预定义的警告消息
        with pytest.warns(UserWarning, match=warning_msg):
            # 进行交叉验证评分，检查其结果是否接近于预期的 error_score
            scores = cross_val_score(
                clf, X, y, cv=3, scoring=failing_scorer, error_score=error_score
            )
            assert_allclose(scores, error_score)


# 使用 pytest 的 filterwarnings 装饰器，忽略特定警告信息
@pytest.mark.filterwarnings("ignore:lbfgs failed to converge")
# 使用 pytest 的 parametrize 装饰器，定义多组参数化测试用例
@pytest.mark.parametrize("error_score", [np.nan, 0, "raise"])
# 使用 pytest 的 parametrize 装饰器，定义多组参数化测试用例
@pytest.mark.parametrize("return_train_score", [True, False])
# 使用 pytest 的 parametrize 装饰器，定义多组参数化测试用例
@pytest.mark.parametrize("with_multimetric", [False, True])
# 定义一个测试函数，测试在 cross_validate 中使用失败的评分器
def test_cross_validate_failing_scorer(
    error_score, return_train_score, with_multimetric
):
    # 检查在 cross_validate 中评分过程中是否能够处理评分器失败，并可选地用 error_score 替换
    # 在多指标的情况下，还检查了一个评分器正常，其他评分器失败时的结果
    X, y = load_iris(return_X_y=True)
    # 创建逻辑回归分类器的实例，并拟合数据
    clf = LogisticRegression(max_iter=5).fit(X, y)

    # 定义用于失败评分器的错误消息
    error_msg = "This scorer is supposed to fail!!!"
    # 创建一个部分函数，将失败的评分器与特定错误消息绑定在一起
    failing_scorer = partial(_failing_scorer, error_msg=error_msg)
    # 如果需要多指标评估
    if with_multimetric:
        # 创建一个评分器，用于处理成功的评分（均方误差）
        non_failing_scorer = make_scorer(mean_squared_error)
        # 设置多个评分指标，其中包括一个失败的评分器和一个非失败的评分器
        scoring = {
            "score_1": failing_scorer,
            "score_2": non_failing_scorer,
            "score_3": failing_scorer,
        }
    else:
        # 否则，只使用失败的评分器
        scoring = failing_scorer

    # 如果错误分数设置为 "raise"
    if error_score == "raise":
        # 使用 pytest 检查是否会抛出 ValueError，并匹配指定的错误消息
        with pytest.raises(ValueError, match=error_msg):
            # 进行交叉验证，验证给定分类器 clf 在数据集 X 和标签 y 上的性能
            cross_validate(
                clf,
                X,
                y,
                cv=3,
                scoring=scoring,
                return_train_score=return_train_score,
                error_score=error_score,
            )
    else:
        # 否则，生成一个警告消息，指示评分失败，并设置错误分数
        warning_msg = (
            "Scoring failed. The score on this train-test partition for "
            f"these parameters will be set to {error_score}"
        )
        # 使用 pytest 检查是否会发出 UserWarning，并匹配警告消息
        with pytest.warns(UserWarning, match=warning_msg):
            # 进行交叉验证，验证给定分类器 clf 在数据集 X 和标签 y 上的性能
            results = cross_validate(
                clf,
                X,
                y,
                cv=3,
                scoring=scoring,
                return_train_score=return_train_score,
                error_score=error_score,
            )
            # 针对结果中的每一个键
            for key in results:
                # 如果键包含 "_score"
                if "_score" in key:
                    # 如果键包含 "_score_2"，检查应为非失败的评分器的测试得分
                    if "_score_2" in key:
                        for i in results[key]:
                            # 断言每个得分为 float 类型
                            assert isinstance(i, float)
                    else:
                        # 否则，检查所有指定为 `error_score` 的评分器的测试得分
                        assert_allclose(results[key], error_score)
# 定义一个函数，接受三个参数 i, j, k，总是返回固定的浮点数 3.4213
def three_params_scorer(i, j, k):
    return 3.4213


# 使用 pytest 的 parametrize 装饰器为 test_fit_and_score_verbosity 函数提供多组参数化输入
@pytest.mark.parametrize(
    # 定义参数化的参数：train_score, scorer, verbose, split_prg, cdt_prg, expected
    "train_score, scorer, verbose, split_prg, cdt_prg, expected",
    [
        (
            # 第一组参数化输入
            False,
            three_params_scorer,  # scorer 参数为 three_params_scorer 函数
            2,  # verbose 参数为 2
            (1, 3),  # split_prg 参数为元组 (1, 3)
            (0, 1),  # cdt_prg 参数为元组 (0, 1)
            r"\[CV\] END ...................................................."
            r" total time=   0.\ds",  # expected 参数为正则表达式字符串
        ),
        (
            # 第二组参数化输入
            True,
            _MultimetricScorer(
                scorers={"sc1": three_params_scorer, "sc2": three_params_scorer}
            ),  # scorer 参数为 _MultimetricScorer 对象，包含两个名为 sc1 和 sc2 的评分器
            3,  # verbose 参数为 3
            (1, 3),  # split_prg 参数为元组 (1, 3)
            (0, 1),  # cdt_prg 参数为元组 (0, 1)
            r"\[CV 2/3\] END  sc1: \(train=3.421, test=3.421\) sc2: "
            r"\(train=3.421, test=3.421\) total time=   0.\ds",  # expected 参数为正则表达式字符串
        ),
        (
            # 第三组参数化输入
            False,
            _MultimetricScorer(
                scorers={"sc1": three_params_scorer, "sc2": three_params_scorer}
            ),  # scorer 参数为 _MultimetricScorer 对象，包含两个名为 sc1 和 sc2 的评分器
            10,  # verbose 参数为 10
            (1, 3),  # split_prg 参数为元组 (1, 3)
            (0, 1),  # cdt_prg 参数为元组 (0, 1)
            r"\[CV 2/3; 1/1\] END ....... sc1: \(test=3.421\) sc2: \(test=3.421\)"
            r" total time=   0.\ds",  # expected 参数为正则表达式字符串
        ),
    ],
)
# 定义测试函数 test_fit_and_score_verbosity，接受多个参数化输入
def test_fit_and_score_verbosity(
    capsys, train_score, scorer, verbose, split_prg, cdt_prg, expected
):
    # 使用 make_classification 生成数据集 X, y
    X, y = make_classification(n_samples=30, random_state=0)
    # 使用线性核的 SVC 分类器
    clf = SVC(kernel="linear", random_state=0)
    # 生成一个 ShuffleSplit 对象并获取一组训练集和测试集的索引
    train, test = next(ShuffleSplit().split(X))

    # 准备 fit_and_score 函数的参数字典
    fit_and_score_args = dict(
        estimator=clf,
        X=X,
        y=y,
        scorer=scorer,
        train=train,
        test=test,
        verbose=verbose,
        parameters=None,
        fit_params=None,
        score_params=None,
        return_train_score=train_score,
        split_progress=split_prg,
        candidate_progress=cdt_prg,
    )
    # 调用 _fit_and_score 函数，捕获输出到 capsys
    _fit_and_score(**fit_and_score_args)
    out, _ = capsys.readouterr()
    # 将捕获的输出按行分割
    outlines = out.split("\n")
    # 如果输出行数大于 2，则断言第二行与 expected 匹配，否则断言第一行与 expected 匹配
    if len(outlines) > 2:
        assert re.match(expected, outlines[1])
    else:
        assert re.match(expected, outlines[0])


# 定义测试函数 test_score
def test_score():
    error_message = "scoring must return a number, got None"

    # 定义一个返回 None 的评分器函数 two_params_scorer
    def two_params_scorer(estimator, X_test):
        return None

    # 使用 pytest.raises 断言捕获 ValueError 异常，异常消息需匹配 error_message
    with pytest.raises(ValueError, match=error_message):
        _score(
            estimator=None,
            X_test=None,
            y_test=None,
            scorer=two_params_scorer,
            score_params=None,
            error_score=np.nan,
        )


# 定义测试函数 test_callable_multimetric_confusion_matrix_cross_validate
def test_callable_multimetric_confusion_matrix_cross_validate():
    # 定义一个自定义评分器函数 custom_scorer，返回混淆矩阵的各项指标组成的字典
    def custom_scorer(clf, X, y):
        y_pred = clf.predict(X)
        cm = confusion_matrix(y, y_pred)
        return {"tn": cm[0, 0], "fp": cm[0, 1], "fn": cm[1, 0], "tp": cm[1, 1]}

    # 生成一个包含 40 个样本和 4 个特征的数据集 X, y
    X, y = make_classification(n_samples=40, n_features=4, random_state=42)
    # 使用线性核的 LinearSVC 分类器
    est = LinearSVC(random_state=42)
    # 使用交叉验证评估分类器，返回交叉验证结果字典 cv_results
    cv_results = cross_validate(est, X, y, cv=5, scoring=custom_scorer)

    # 断言交叉验证结果中包含指定的评分名称
    score_names = ["tn", "fp", "fn", "tp"]
    for name in score_names:
        assert "test_{}".format(name) in cv_results
# 检查具有 partial_fit 支持的回归器的学习曲线
def test_learning_curve_partial_fit_regressors():
    """Check that regressors with partial_fit is supported.

    Non-regression test for #22981.
    """
    # 生成一个随机数据集 X 和 y
    X, y = make_regression(random_state=42)

    # 使用 MLPRegressor 调用 learning_curve 函数，开启增量学习模式，cv 设置为 2
    # 这里不会引发错误
    learning_curve(MLPRegressor(), X, y, exploit_incremental_learning=True, cv=2)


# 检查 `learning_curve` 中一些拟合失败的情况是否会引发必要的警告
def test_learning_curve_some_failing_fits_warning(global_random_seed):
    """Checks for fit failures in `learning_curve` and raises the required warning"""

    # 创建一个分类数据集 X 和 y
    X, y = make_classification(
        n_samples=30,
        n_classes=3,
        n_informative=6,
        shuffle=False,
        random_state=global_random_seed,
    )
    # 对目标进行排序，以触发 SVC 在前两次拆分时的错误，因为一个类别只有一个实例
    sorted_idx = np.argsort(y)
    X, y = X[sorted_idx], y[sorted_idx]

    # 创建一个 SVM 分类器
    svc = SVC()
    warning_message = "10 fits failed out of a total of 25"

    # 使用 pytest 来捕获 FitFailedWarning，确保拟合失败时会引发警告
    with pytest.warns(FitFailedWarning, match=warning_message):
        # 调用 learning_curve 函数，检查拟合失败的情况
        _, train_score, test_score, *_ = learning_curve(
            svc, X, y, cv=5, error_score=np.nan
        )

    # 前两次拆分应该导致警告，因此得到 np.nan 的分数
    for idx in range(2):
        assert np.isnan(train_score[idx]).all()
        assert np.isnan(test_score[idx]).all()

    # 后面的拆分应该没有警告，确保没有 np.nan 的分数
    for idx in range(2, train_score.shape[0]):
        assert not np.isnan(train_score[idx]).any()
        assert not np.isnan(test_score[idx]).any()


# 检查 `cross_validate` 中 `return_indices` 参数的行为
def test_cross_validate_return_indices(global_random_seed):
    """Check the behaviour of `return_indices` in `cross_validate`."""
    # 加载鸢尾花数据集 X 和 y
    X, y = load_iris(return_X_y=True)
    # 对特征进行缩放以获得更好的收敛性能
    X = scale(X)
    estimator = LogisticRegression()

    # 创建一个 KFold 交叉验证对象
    cv = KFold(n_splits=3, shuffle=True, random_state=global_random_seed)

    # 测试 `cross_validate` 函数，确保返回结果中不包含 "indices"
    cv_results = cross_validate(estimator, X, y, cv=cv, n_jobs=2, return_indices=False)
    assert "indices" not in cv_results

    # 再次测试 `cross_validate` 函数，确保返回结果中包含 "indices"
    cv_results = cross_validate(estimator, X, y, cv=cv, n_jobs=2, return_indices=True)
    assert "indices" in cv_results
    train_indices = cv_results["indices"]["train"]
    test_indices = cv_results["indices"]["test"]

    # 确保训练集和测试集的索引数量正确
    assert len(train_indices) == cv.n_splits
    assert len(test_indices) == cv.n_splits

    # 确保每个拆分的训练集和测试集索引的大小正确
    assert_array_equal([indices.size for indices in train_indices], 100)
    assert_array_equal([indices.size for indices in test_indices], 50)

    # 对比每个拆分的预期训练集和测试集索引
    for split_idx, (expected_train_idx, expected_test_idx) in enumerate(cv.split(X, y)):
        assert_array_equal(train_indices[split_idx], expected_train_idx)
        assert_array_equal(test_indices[split_idx], expected_test_idx)


# 用于测试交叉验证和学习曲线中元数据路由的函数
@pytest.mark.parametrize("func", [cross_validate, cross_val_predict, learning_curve])
def test_fit_param_deprecation(func):
    """Check that we warn about deprecating `fit_params`."""
    # 使用 pytest 来检查是否有 FutureWarning 警告，匹配指定的警告信息
    with pytest.warns(FutureWarning, match="`fit_params` is deprecated"):
        # 调用 func 函数，传入 estimator 参数为 ConsumingClassifier() 的实例，
        # X 参数为当前作用域中的 X，y 参数为当前作用域中的 y，
        # cv 参数设为 2，fit_params 参数设为空字典 {}。
        func(estimator=ConsumingClassifier(), X=X, y=y, cv=2, fit_params={})
    
    # 使用 pytest 来检查是否有 ValueError 异常抛出，匹配指定的异常信息
    with pytest.raises(
        ValueError, match="`params` and `fit_params` cannot both be provided"
    ):
        # 调用 func 函数，传入 estimator 参数为 ConsumingClassifier() 的实例，
        # X 参数为当前作用域中的 X，y 参数为当前作用域中的 y，
        # fit_params 参数设为空字典 {}，params 参数也设为空字典 {}。
        func(estimator=ConsumingClassifier(), X=X, y=y, fit_params={}, params={})
# 使用 pytest.mark.usefixtures 装饰器指定测试用例依赖的 fixture "enable_slep006"
# 使用 pytest.mark.parametrize 装饰器，参数化测试函数 func，传入函数列表 [cross_validate, cross_val_score, cross_val_predict, learning_curve]
def test_groups_with_routing_validation(func):
    """Check that we raise an error if `groups` are passed to the cv method instead
    of `params` when metadata routing is enabled.
    """
    # 使用 pytest.raises 断言捕获 ValueError 异常，匹配异常消息中包含 "`groups` can only be passed if" 的部分
    with pytest.raises(ValueError, match="`groups` can only be passed if"):
        # 调用传入的 func 函数，传入参数字典和额外的 groups=[] 参数
        func(
            estimator=ConsumingClassifier(),
            X=X,
            y=y,
            groups=[],  # 尝试传入空的 groups 列表作为参数
        )


# 使用 pytest.mark.usefixtures 装饰器指定测试用例依赖的 fixture "enable_slep006"
# 使用 pytest.mark.parametrize 装饰器，参数化测试函数 func，传入函数列表 [cross_validate, cross_val_score, cross_val_predict, learning_curve]
def test_passed_unrequested_metadata(func):
    """Check that we raise an error when passing metadata that is not
    requested."""
    # 使用 re.escape 转义字符串 "but are not explicitly set as requested or not requested" 作为异常消息的匹配模式
    err_msg = re.escape("but are not explicitly set as requested or not requested")
    # 使用 pytest.raises 断言捕获 ValueError 异常，匹配异常消息与预期 err_msg 相符的部分
    with pytest.raises(ValueError, match=err_msg):
        # 调用传入的 func 函数，传入参数字典和额外的 params=dict(metadata=[]) 参数
        func(
            estimator=ConsumingClassifier(),
            X=X,
            y=y,
            params=dict(metadata=[]),  # 尝试传入空的 metadata 列表作为参数
        )


# 使用 pytest.mark.usefixtures 装饰器指定测试用例依赖的 fixture "enable_slep006"
# 使用 pytest.mark.parametrize 装饰器，参数化测试函数 func，传入函数列表 [cross_validate, cross_val_score, cross_val_predict, learning_curve]
def test_validation_functions_routing(func):
    """Check that the respective cv method is properly dispatching the metadata
    to the consumer."""
    # 创建 _Registry 实例作为 scorer_registry
    scorer_registry = _Registry()
    # 创建 ConsumingScorer 实例，设置 score_request 参数为 sample_weight="score_weights", metadata="score_metadata"
    scorer = ConsumingScorer(registry=scorer_registry).set_score_request(
        sample_weight="score_weights", metadata="score_metadata"
    )
    # 创建 _Registry 实例作为 splitter_registry
    splitter_registry = _Registry()
    # 创建 ConsumingSplitter 实例，设置 split_request 参数为 groups="split_groups", metadata="split_metadata"
    splitter = ConsumingSplitter(registry=splitter_registry).set_split_request(
        groups="split_groups", metadata="split_metadata"
    )
    # 创建 _Registry 实例作为 estimator_registry
    estimator_registry = _Registry()
    # 创建 ConsumingClassifier 实例，设置 fit_request 参数为 sample_weight="fit_sample_weight", metadata="fit_metadata"
    estimator = ConsumingClassifier(registry=estimator_registry).set_fit_request(
        sample_weight="fit_sample_weight", metadata="fit_metadata"
    )

    # 计算样本数并设置随机数生成器
    n_samples = _num_samples(X)
    rng = np.random.RandomState(0)
    # 生成随机的 score_weights、score_metadata、split_groups、split_metadata、fit_sample_weight、fit_metadata
    score_weights = rng.rand(n_samples)
    score_metadata = rng.rand(n_samples)
    split_groups = rng.randint(0, 3, n_samples)
    split_metadata = rng.rand(n_samples)
    fit_sample_weight = rng.rand(n_samples)
    fit_metadata = rng.rand(n_samples)

    # 定义额外参数字典，根据不同的 func 函数设置不同的 scoring 参数
    extra_params = {
        cross_validate: dict(scoring=dict(my_scorer=scorer, accuracy="accuracy")),
        cross_val_score: dict(scoring=scorer),  # cross_val_score 和 learning_curve 不支持多个 scorer
        learning_curve: dict(scoring=scorer),
        cross_val_predict: dict(),  # cross_val_predict 不需要 scorer
    }

    # 定义 params 字典，包含 split_groups、split_metadata、fit_sample_weight、fit_metadata
    params = dict(
        split_groups=split_groups,
        split_metadata=split_metadata,
        fit_sample_weight=fit_sample_weight,
        fit_metadata=fit_metadata,
    )

    # 如果 func 不是 cross_val_predict，则更新 params 字典添加 score_weights 和 score_metadata
    if func is not cross_val_predict:
        params.update(
            score_weights=score_weights,
            score_metadata=score_metadata,
        )
    # 调用函数func，并传入以下参数：estimator, X, y, cv=splitter, 以及来自extra_params[func]的额外参数，还有params
    func(
        estimator,
        X=X,
        y=y,
        cv=splitter,
        **extra_params[func],
        params=params,
    )

    # 如果func不是cross_val_predict函数，则执行以下断言，确保scorer_registry非空
    if func is not cross_val_predict:
        assert len(scorer_registry)
    
    # 遍历scorer_registry中的每个_scorer，对每个_scorer执行metadata的记录检查
    for _scorer in scorer_registry:
        check_recorded_metadata(
            obj=_scorer,
            method="score",
            parent=func.__name__,
            split_params=("sample_weight", "metadata"),
            sample_weight=score_weights,
            metadata=score_metadata,
        )

    # 确保splitter_registry非空
    assert len(splitter_registry)
    
    # 遍历splitter_registry中的每个_splitter，对每个_splitter执行metadata的记录检查
    for _splitter in splitter_registry:
        check_recorded_metadata(
            obj=_splitter,
            method="split",
            parent=func.__name__,
            groups=split_groups,
            metadata=split_metadata,
        )

    # 确保estimator_registry非空
    assert len(estimator_registry)
    
    # 遍历estimator_registry中的每个_estimator，对每个_estimator执行metadata的记录检查
    for _estimator in estimator_registry:
        check_recorded_metadata(
            obj=_estimator,
            method="fit",
            parent=func.__name__,
            split_params=("sample_weight", "metadata"),
            sample_weight=fit_sample_weight,
            metadata=fit_metadata,
        )
@pytest.mark.usefixtures("enable_slep006")
# 使用 pytest 的标记，用来指示测试函数需要依赖名为 "enable_slep006" 的 fixture

def test_learning_curve_exploit_incremental_learning_routing():
    """Test that learning_curve routes metadata to the estimator correctly while
    partial_fitting it with `exploit_incremental_learning=True`."""
    # 测试 learning_curve 在设置 `exploit_incremental_learning=True` 时，正确将元数据传递给估计器

    n_samples = _num_samples(X)
    # 获取样本数量，_num_samples 函数的实际作用未知，可能用于获取样本数目

    rng = np.random.RandomState(0)
    # 创建随机数生成器 rng，种子为 0，确保结果可复现

    fit_sample_weight = rng.rand(n_samples)
    # 用 rng 生成 n_samples 个随机数作为拟合时的样本权重

    fit_metadata = rng.rand(n_samples)
    # 用 rng 生成 n_samples 个随机数作为拟合时的元数据

    estimator_registry = _Registry()
    # 创建估计器注册表实例

    estimator = ConsumingClassifier(
        registry=estimator_registry
    ).set_partial_fit_request(
        sample_weight="fit_sample_weight", metadata="fit_metadata"
    )
    # 创建 ConsumingClassifier 实例，设置其部分拟合请求的样本权重和元数据

    learning_curve(
        estimator,
        X=X,
        y=y,
        cv=ConsumingSplitter(),
        exploit_incremental_learning=True,
        params=dict(fit_sample_weight=fit_sample_weight, fit_metadata=fit_metadata),
    )
    # 调用 learning_curve 函数进行学习曲线计算，传入估计器、特征 X、目标变量 y、交叉验证对象 ConsumingSplitter、设置 `exploit_incremental_learning=True`，以及拟合参数字典

    assert len(estimator_registry)
    # 断言估计器注册表中有记录，即确保至少有一个估计器被注册

    for _estimator in estimator_registry:
        check_recorded_metadata(
            obj=_estimator,
            method="partial_fit",
            parent="learning_curve",
            split_params=("sample_weight", "metadata"),
            sample_weight=fit_sample_weight,
            metadata=fit_metadata,
        )
    # 遍历估计器注册表中的每个估计器，调用 check_recorded_metadata 函数检查记录的元数据：
    # - obj: 当前估计器实例
    # - method: 被调用的方法名为 "partial_fit"
    # - parent: 调用该方法的父函数为 "learning_curve"
    # - split_params: 拆分参数包括 "sample_weight" 和 "metadata"
    # - sample_weight: 拟合时使用的样本权重
    # - metadata: 拟合时使用的元数据

# End of metadata routing tests
# =============================
```