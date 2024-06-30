# `D:\src\scipysrc\scikit-learn\sklearn\model_selection\tests\test_search.py`

```
# 引入必要的库和模块
import pickle  # 导入pickle模块，用于对象序列化和反序列化
import re  # 导入re模块，用于正则表达式操作
import sys  # 导入sys模块，提供对Python解释器的访问
import warnings  # 导入warnings模块，用于管理警告
from collections.abc import Iterable, Sized  # 从collections.abc模块导入Iterable和Sized抽象基类
from functools import partial  # 导入functools模块中的partial函数，用于部分函数应用
from io import StringIO  # 导入StringIO类，用于在内存中操作文本数据
from itertools import chain, product  # 导入itertools模块中的chain和product函数，用于迭代工具
from types import GeneratorType  # 从types模块导入GeneratorType类型，用于生成器类型的检查

import numpy as np  # 导入NumPy库，并简写为np，用于数值计算
import pytest  # 导入pytest库，用于编写和运行测试用例
from scipy.stats import bernoulli, expon, uniform  # 从scipy.stats模块导入统计分布函数

# 导入Scikit-Learn库中的相关模块和类
from sklearn import config_context  # 导入config_context，用于配置上下文环境
from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier  # 从sklearn.base模块导入基础类和函数
from sklearn.cluster import KMeans  # 导入KMeans类，用于k均值聚类
from sklearn.compose import ColumnTransformer  # 导入ColumnTransformer类，用于数据转换管道
from sklearn.datasets import (
    make_blobs,  # 导入make_blobs函数，用于生成聚类测试数据
    make_classification,  # 导入make_classification函数，用于生成分类测试数据
    make_multilabel_classification,  # 导入make_multilabel_classification函数，用于生成多标签分类测试数据
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # 导入LinearDiscriminantAnalysis类，用于线性判别分析
from sklearn.dummy import DummyClassifier  # 导入DummyClassifier类，用于生成虚拟分类器
from sklearn.ensemble import HistGradientBoostingClassifier  # 导入HistGradientBoostingClassifier类，用于直方图梯度提升分类器
from sklearn.exceptions import FitFailedWarning  # 导入FitFailedWarning异常类，用于拟合失败警告
from sklearn.experimental import enable_halving_search_cv  # 导入enable_halving_search_cv，用于启用搜索交叉验证
from sklearn.feature_extraction.text import TfidfVectorizer  # 导入TfidfVectorizer类，用于文本特征提取
from sklearn.impute import SimpleImputer  # 导入SimpleImputer类，用于填充缺失值
from sklearn.linear_model import (
    LinearRegression,  # 导入LinearRegression类，用于线性回归
    LogisticRegression,  # 导入LogisticRegression类，用于逻辑回归
    Ridge,  # 导入Ridge类，用于岭回归
    SGDClassifier,  # 导入SGDClassifier类，用于随机梯度下降分类器
)
from sklearn.metrics import (
    accuracy_score,  # 导入accuracy_score函数，用于计算准确率
    confusion_matrix,  # 导入confusion_matrix函数，用于计算混淆矩阵
    f1_score,  # 导入f1_score函数，用于计算F1分数
    make_scorer,  # 导入make_scorer函数，用于生成评分函数
    r2_score,  # 导入r2_score函数，用于计算R²分数
    recall_score,  # 导入recall_score函数，用于计算召回率
    roc_auc_score,  # 导入roc_auc_score函数，用于计算ROC曲线下面积
)
from sklearn.metrics.pairwise import euclidean_distances  # 导入euclidean_distances函数，用于计算欧氏距离
from sklearn.model_selection import (
    GridSearchCV,  # 导入GridSearchCV类，用于网格搜索交叉验证
    GroupKFold,  # 导入GroupKFold类，用于分组k折交叉验证
    GroupShuffleSplit,  # 导入GroupShuffleSplit类，用于分组随机打乱划分
    HalvingGridSearchCV,  # 导入HalvingGridSearchCV类，用于半分搜索交叉验证
    KFold,  # 导入KFold类，用于k折交叉验证
    LeaveOneGroupOut,  # 导入LeaveOneGroupOut类，用于留一分组交叉验证
    LeavePGroupsOut,  # 导入LeavePGroupsOut类，用于留P分组交叉验证
    ParameterGrid,  # 导入ParameterGrid类，用于参数网格
    ParameterSampler,  # 导入ParameterSampler类，用于参数采样
    RandomizedSearchCV,  # 导入RandomizedSearchCV类，用于随机搜索交叉验证
    StratifiedKFold,  # 导入StratifiedKFold类，用于分层k折交叉验证
    StratifiedShuffleSplit,  # 导入StratifiedShuffleSplit类，用于分层随机打乱划分
    train_test_split,  # 导入train_test_split函数，用于训练集测试集划分
)
from sklearn.model_selection._search import BaseSearchCV  # 导入BaseSearchCV类，用于基础搜索交叉验证
from sklearn.model_selection.tests.common import OneTimeSplitter  # 导入OneTimeSplitter类，用于一次性分割器
from sklearn.naive_bayes import ComplementNB  # 导入ComplementNB类，用于补充朴素贝叶斯分类器
from sklearn.neighbors import KernelDensity, KNeighborsClassifier, LocalOutlierFactor  # 导入各种邻居方法和局部异常因子
from sklearn.pipeline import Pipeline  # 导入Pipeline类，用于管道
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler  # 导入数据预处理类
from sklearn.svm import SVC, LinearSVC  # 导入SVC和LinearSVC类，用于支持向量机
from sklearn.tests.metadata_routing_common import (
    ConsumingScorer,  # 导入ConsumingScorer类，用于消耗评分器
    _Registry,  # 导入_Registry类，用于注册表
    check_recorded_metadata,  # 导入check_recorded_metadata函数，用于检查记录的元数据
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # 导入决策树分类器和回归器
from sklearn.utils._array_api import yield_namespace_device_dtype_combinations  # 导入yield_namespace_device_dtype_combinations函数，用于生成器
from sklearn.utils._mocking import CheckingClassifier, MockDataFrame  # 导入CheckingClassifier和MockDataFrame类，用于模拟测试
from sklearn.utils._testing import (
    MinimalClassifier,  # 导入MinimalClassifier类，用于最小分类器测试
    MinimalRegressor,  # 导入MinimalRegressor类，用于最小回归器测试
    MinimalTransformer,  # 导入MinimalTransformer类，用于最小转换器测试
    _array_api_for_tests,  # 导入_array_api_for_tests函数，用于测试数组API
    assert_allclose,  # 导入assert_allclose函数，用于全面比较数组或标量
    assert_almost_equal,  # 导入assert_almost_equal函数，用于比较数字的近似相等性
    assert_array_almost_equal,  # 导入assert_array_almost_equal函数，用于比较数组的近似相等性
    assert_array_equal,  # 导入assert_array_equal函数，用于比较数组的相等性
    ignore_warnings,  # 导入ignore_warnings函数，用于忽略警告
)
from sklearn.utils.fixes import CSR_CONTAINERS  # 导入CSR_CONTAINERS常量，用于压
    # 初始化方法，设置对象的初始状态，可选参数为 foo_param，默认为 0
    def __init__(self, foo_param=0):
        # 将参数 foo_param 存储在对象的属性中
        self.foo_param = foo_param

    # 拟合方法，用于训练模型，要求输入的 X 和 Y 的长度必须相等
    def fit(self, X, Y):
        # 断言输入的 X 和 Y 的长度相同
        assert len(X) == len(Y)
        # 计算 Y 中唯一值，作为模型的类别
        self.classes_ = np.unique(Y)
        # 返回对象本身，用于方法链
        return self

    # 预测方法，返回输入 T 的行数，即样本数
    def predict(self, T):
        return T.shape[0]

    # 转换方法，将输入 X 和对象的 foo_param 相加后返回
    def transform(self, X):
        return X + self.foo_param

    # 反向转换方法，将输入 X 减去对象的 foo_param 后返回
    def inverse_transform(self, X):
        return X - self.foo_param

    # 预测概率方法，与 predict 方法相同
    predict_proba = predict
    # 预测对数概率方法，与 predict 方法相同
    predict_log_proba = predict
    # 决策函数方法，与 predict 方法相同
    decision_function = predict

    # 打分方法，根据对象的 foo_param 返回分数，大于 1 返回 1.0，否则返回 0.0
    def score(self, X=None, Y=None):
        if self.foo_param > 1:
            score = 1.0
        else:
            score = 0.0
        return score

    # 获取对象的参数，返回包含 foo_param 的字典
    def get_params(self, deep=False):
        return {"foo_param": self.foo_param}

    # 设置对象的参数，接受一个名为 foo_param 的参数，并将其设置为对象的 foo_param 属性
    def set_params(self, **params):
        self.foo_param = params["foo_param"]
        return self
class LinearSVCNoScore(LinearSVC):
    """A LinearSVC classifier that has no score method."""

    @property
    def score(self):
        # 如果尝试访问 score 属性，抛出 AttributeError
        raise AttributeError


X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])


def assert_grid_iter_equals_getitem(grid):
    # 断言迭代器和索引访问结果相同
    assert list(grid) == [grid[i] for i in range(len(grid))]


@pytest.mark.parametrize("klass", [ParameterGrid, partial(ParameterSampler, n_iter=10)])
@pytest.mark.parametrize(
    "input, error_type, error_message",
    [
        (0, TypeError, r"Parameter .* a dict or a list, got: 0 of type int"),
        ([{"foo": [0]}, 0], TypeError, r"Parameter .* is not a dict \(0\)"),
        (
            {"foo": 0},
            TypeError,
            r"Parameter (grid|distribution) for parameter 'foo' (is not|needs to be) "
            r"(a list or a numpy array|iterable or a distribution).*",
        ),
    ],
)
def test_validate_parameter_input(klass, input, error_type, error_message):
    # 使用 pytest.raises 检查是否抛出指定类型和消息的异常
    with pytest.raises(error_type, match=error_message):
        klass(input)


def test_parameter_grid():
    # 测试 ParameterGrid 的基本特性
    params1 = {"foo": [1, 2, 3]}
    grid1 = ParameterGrid(params1)
    assert isinstance(grid1, Iterable)
    assert isinstance(grid1, Sized)
    assert len(grid1) == 3
    assert_grid_iter_equals_getitem(grid1)

    params2 = {"foo": [4, 2], "bar": ["ham", "spam", "eggs"]}
    grid2 = ParameterGrid(params2)
    assert len(grid2) == 6

    # 循环以确保可以多次迭代网格
    for i in range(2):
        # tuple + chain 将 {"a": 1, "b": 2} 转换为 ("a", 1, "b", 2)
        points = set(tuple(chain(*(sorted(p.items())))) for p in grid2)
        assert points == set(
            ("bar", x, "foo", y) for x, y in product(params2["bar"], params2["foo"])
        )
    assert_grid_iter_equals_getitem(grid2)

    # 特殊情况：空网格（获取默认估计器设置）
    empty = ParameterGrid({})
    assert len(empty) == 1
    assert list(empty) == [{}]
    assert_grid_iter_equals_getitem(empty)
    with pytest.raises(IndexError):
        empty[1]

    has_empty = ParameterGrid([{"C": [1, 10]}, {}, {"C": [0.5]}])
    assert len(has_empty) == 4
    assert list(has_empty) == [{"C": 1}, {"C": 10}, {}, {"C": 0.5}]
    assert_grid_iter_equals_getitem(has_empty)


def test_grid_search():
    # 测试最佳估计器是否包含正确的 foo_param 值
    clf = MockClassifier()
    grid_search = GridSearchCV(clf, {"foo_param": [1, 2, 3]}, cv=3, verbose=3)
    # 确保在参数平局时选择最小的参数
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    grid_search.fit(X, y)
    sys.stdout = old_stdout
    assert grid_search.best_estimator_.foo_param == 2

    assert_array_equal(grid_search.cv_results_["param_foo_param"].data, [1, 2, 3])

    # Smoke test the score etc:
    grid_search.score(X, y)
    grid_search.predict_proba(X)
    grid_search.decision_function(X)
    # 对输入数据集 X 进行转换，使用网格搜索对象 grid_search 的 transform 方法
    grid_search.transform(X)
    
    # 测试评分时的异常处理
    # 将网格搜索对象 grid_search 的评分方法设置为 "sklearn"
    grid_search.scoring = "sklearn"
    # 使用 pytest 的 pytest.raises 方法验证是否会抛出 ValueError 异常
    with pytest.raises(ValueError):
        # 在网格搜索对象上调用 fit 方法，并传入输入数据集 X 和标签数据集 y
        grid_search.fit(X, y)
# 定义一个测试函数，用于验证网格搜索管道步骤的行为
def test_grid_search_pipeline_steps():
    # 创建一个管道，包含一个线性回归器作为估算器
    pipe = Pipeline([("regressor", LinearRegression())])
    # 定义参数网格，包含两种回归器类型：线性回归和岭回归
    param_grid = {"regressor": [LinearRegression(), Ridge()]}
    # 创建一个网格搜索对象，使用2折交叉验证
    grid_search = GridSearchCV(pipe, param_grid, cv=2)
    # 对数据进行拟合
    grid_search.fit(X, y)
    # 获取回归器的参数列表
    regressor_results = grid_search.cv_results_["param_regressor"]
    # 第一个回归器应当是线性回归类型
    assert isinstance(regressor_results[0], LinearRegression)
    # 第二个回归器应当是岭回归类型
    assert isinstance(regressor_results[1], Ridge)
    # 验证第一个回归器没有属性 'coef_'，即不应当有系数属性
    assert not hasattr(regressor_results[0], "coef_")
    # 验证第二个回归器没有属性 'coef_'，即不应当有系数属性
    assert not hasattr(regressor_results[1], "coef_")
    # 验证第一个回归器不是最佳估算器的同一个实例
    assert regressor_results[0] is not grid_search.best_estimator_
    # 验证第二个回归器不是最佳估算器的同一个实例
    assert regressor_results[1] is not grid_search.best_estimator_
    # 检查我们没有修改传递的参数网格
    assert not hasattr(param_grid["regressor"][0], "coef_")
    # 检查我们没有修改传递的参数网格
    assert not hasattr(param_grid["regressor"][1], "coef_")


# 使用参数化测试装饰器，定义一个测试函数，用于测试带有特定参数的搜索器对象
@pytest.mark.parametrize("SearchCV", [GridSearchCV, RandomizedSearchCV])
def test_SearchCV_with_fit_params(SearchCV):
    # 创建一个10x10的数组作为特征矩阵 X，创建一个长度为10的二元数组作为目标变量 y
    X = np.arange(100).reshape(10, 10)
    y = np.array([0] * 5 + [1] * 5)
    # 创建一个CheckingClassifier对象，期望的拟合参数为 ["spam", "eggs"]
    clf = CheckingClassifier(expected_fit_params=["spam", "eggs"])
    # 创建一个搜索器对象，使用指定的分类器和参数网格，2折交叉验证，并且如果出错则抛出异常
    searcher = SearchCV(clf, {"foo_param": [1, 2, 3]}, cv=2, error_score="raise")

    # 当缺少参数 'eggs' 时，验证是否会抛出断言错误
    err_msg = r"Expected fit parameter\(s\) \['eggs'\] not seen."
    with pytest.raises(AssertionError, match=err_msg):
        searcher.fit(X, y, spam=np.ones(10))

    # 当参数 'spam' 的长度为1时，验证是否会抛出断言错误
    err_msg = "Fit parameter spam has length 1; expected"
    with pytest.raises(AssertionError, match=err_msg):
        searcher.fit(X, y, spam=np.ones(1), eggs=np.zeros(10))
    # 使用正常的参数进行拟合
    searcher.fit(X, y, spam=np.ones(10), eggs=np.zeros(10))


# 使用忽略警告装饰器，定义一个测试函数，用于测试不带评分功能的网格搜索
@ignore_warnings
def test_grid_search_no_score():
    # 创建一个LinearSVC分类器对象
    clf = LinearSVC(random_state=0)
    # 创建一个包含2个中心的样本集 X 和相应的目标向量 y
    X, y = make_blobs(random_state=0, centers=2)
    # 定义一组可能的C参数值
    Cs = [0.1, 1, 10]
    # 创建一个没有评分函数的网格搜索对象，使用"accuracy"评分
    clf_no_score = LinearSVCNoScore(random_state=0)
    grid_search = GridSearchCV(clf, {"C": Cs}, scoring="accuracy")
    grid_search.fit(X, y)

    grid_search_no_score = GridSearchCV(clf_no_score, {"C": Cs}, scoring="accuracy")
    # 对没有评分函数的网格搜索对象进行基本测试
    grid_search_no_score.fit(X, y)

    # 检查最佳参数是否相等
    assert grid_search_no_score.best_params_ == grid_search.best_params_
    # 检查能否调用评分函数，并且返回正确的结果
    assert grid_search.score(X, y) == grid_search_no_score.score(X, y)

    # 当不提供评分函数时，应当引发TypeError异常
    grid_search_no_score = GridSearchCV(clf_no_score, {"C": Cs})
    with pytest.raises(TypeError, match="no scoring"):
        grid_search_no_score.fit([[1]])


# 定义一个测试函数，用于测试网格搜索的评分方法
def test_grid_search_score_method():
    # 创建一个样本数为100，2类别，flip_y=0.2的分类样本集 X 和相应的目标向量 y
    X, y = make_classification(n_samples=100, n_classes=2, flip_y=0.2, random_state=0)
    # 创建一个LinearSVC分类器对象
    clf = LinearSVC(random_state=0)
    # 定义一个包含单个C参数的网格
    grid = {"C": [0.1]}
    # 使用 GridSearchCV 进行参数网格搜索，不指定评分方法（默认使用 estimator 的评分方法）
    search_no_scoring = GridSearchCV(clf, grid, scoring=None).fit(X, y)
    
    # 使用 GridSearchCV 进行参数网格搜索，指定评分方法为准确率（accuracy）
    search_accuracy = GridSearchCV(clf, grid, scoring="accuracy").fit(X, y)
    
    # 使用 GridSearchCV 进行参数网格搜索，使用 LinearSVCNoScore 作为 estimator，指定评分方法为 ROC 曲线下面积（roc_auc）
    search_no_score_method_auc = GridSearchCV(
        LinearSVCNoScore(), grid, scoring="roc_auc"
    ).fit(X, y)
    
    # 使用 GridSearchCV 进行参数网格搜索，指定评分方法为 ROC 曲线下面积（roc_auc）
    search_auc = GridSearchCV(clf, grid, scoring="roc_auc").fit(X, y)

    # 检查警告仅在行为改变的情况下出现：
    # estimator 需要实现 score 方法来与 scoring 参数竞争
    score_no_scoring = search_no_scoring.score(X, y)
    score_accuracy = search_accuracy.score(X, y)
    score_no_score_auc = search_no_score_method_auc.score(X, y)
    score_auc = search_auc.score(X, y)

    # 确保测试是合理的
    assert score_auc < 1.0
    assert score_accuracy < 1.0
    assert score_auc != score_accuracy

    # 检验准确率分数与无评分方法的结果是否几乎相等
    assert_almost_equal(score_accuracy, score_no_scoring)
    
    # 检验 ROC AUC 分数与无评分方法的结果是否几乎相等
    assert_almost_equal(score_auc, score_no_score_auc)
def test_grid_search_groups():
    # Check if ValueError (when groups is None) propagates to GridSearchCV
    # And also check if groups is correctly passed to the cv object

    # Create a random number generator with a fixed seed
    rng = np.random.RandomState(0)

    # Generate synthetic data: 15 samples, 2 classes
    X, y = make_classification(n_samples=15, n_classes=2, random_state=0)

    # Generate random groups for the samples
    groups = rng.randint(0, 3, 15)

    # Initialize a linear SVM classifier
    clf = LinearSVC(random_state=0)

    # Define a grid with a single parameter C
    grid = {"C": [1]}

    # Define different cross-validation strategies for grouped data
    group_cvs = [
        LeaveOneGroupOut(),
        LeavePGroupsOut(2),
        GroupKFold(n_splits=3),
        GroupShuffleSplit(),
    ]

    # Error message to check for ValueError
    error_msg = "The 'groups' parameter should not be None."

    # Iterate over each cross-validation strategy
    for cv in group_cvs:
        # Create a GridSearchCV object with the classifier, grid, and cv strategy
        gs = GridSearchCV(clf, grid, cv=cv)
        # Check if ValueError with specified message is raised during fitting without groups
        with pytest.raises(ValueError, match=error_msg):
            gs.fit(X, y)
        # Fit the GridSearchCV with groups specified
        gs.fit(X, y, groups=groups)

    # Define cross-validation strategies that do not use groups
    non_group_cvs = [StratifiedKFold(), StratifiedShuffleSplit()]

    # Iterate over each non-group cross-validation strategy
    for cv in non_group_cvs:
        # Create a GridSearchCV object with the classifier and cv strategy
        gs = GridSearchCV(clf, grid, cv=cv)
        # Fit the GridSearchCV without groups specified (should not raise error)
        gs.fit(X, y)


def test_classes__property():
    # Test that classes_ property matches best_estimator_.classes_

    # Create synthetic data: X is a 10x10 array, y has 5 samples of class 0 and 5 samples of class 1
    X = np.arange(100).reshape(10, 10)
    y = np.array([0] * 5 + [1] * 5)

    # Values of parameter C to test
    Cs = [0.1, 1, 10]

    # Perform grid search with LinearSVC on the data
    grid_search = GridSearchCV(LinearSVC(random_state=0), {"C": Cs})
    grid_search.fit(X, y)

    # Assert that classes_ attribute of best_estimator_ matches grid_search.classes_
    assert_array_equal(grid_search.best_estimator_.classes_, grid_search.classes_)

    # Test that regressors like Ridge do not have a classes_ attribute
    grid_search = GridSearchCV(Ridge(), {"alpha": [1.0, 2.0]})
    grid_search.fit(X, y)
    assert not hasattr(grid_search, "classes_")

    # Test that grid searcher has no classes_ attribute before fitting
    grid_search = GridSearchCV(LinearSVC(random_state=0), {"C": Cs})
    assert not hasattr(grid_search, "classes_")

    # Test that grid searcher has no classes_ attribute when refit is False
    grid_search = GridSearchCV(LinearSVC(random_state=0), {"C": Cs}, refit=False)
    grid_search.fit(X, y)
    assert not hasattr(grid_search, "classes_")


def test_trivial_cv_results_attr():
    # Test search over a "grid" with only one point.

    # MockClassifier for testing purposes
    clf = MockClassifier()

    # GridSearchCV with one parameter and 3-fold cross-validation
    grid_search = GridSearchCV(clf, {"foo_param": [1]}, cv=3)
    grid_search.fit(X, y)

    # Assert that grid_search has cv_results_ attribute
    assert hasattr(grid_search, "cv_results_")

    # RandomizedSearchCV with one parameter and 3-fold cross-validation
    random_search = RandomizedSearchCV(clf, {"foo_param": [0]}, n_iter=1, cv=3)
    random_search.fit(X, y)

    # Assert that random_search has cv_results_ attribute
    assert hasattr(random_search, "cv_results_")


def test_no_refit():
    # Test that GSCV can be used for model selection alone without refitting

    # MockClassifier for testing purposes
    clf = MockClassifier()
    # 针对两种不同的评分配置进行网格搜索
    for scoring in [None, ["accuracy", "precision"]]:
        # 创建一个网格搜索对象，使用指定的分类器 clf，参数字典 {"foo_param": [1, 2, 3]}，
        # refit=False 表示不在找到最佳参数后重新拟合，cv=3 表示使用 3 折交叉验证
        grid_search = GridSearchCV(clf, {"foo_param": [1, 2, 3]}, refit=False, cv=3)
        # 在给定数据集 X, y 上执行网格搜索
        grid_search.fit(X, y)
        
        # 断言以下条件：
        # - grid_search 没有 best_estimator_ 属性
        # - grid_search 有 best_index_ 属性
        # - grid_search 有 best_params_ 属性
        assert (
            not hasattr(grid_search, "best_estimator_")
            and hasattr(grid_search, "best_index_")
            and hasattr(grid_search, "best_params_")
        )

        # 确保 predict/transform 等函数会引发有意义的错误消息
        for fn_name in (
            "predict",
            "predict_proba",
            "predict_log_proba",
            "transform",
            "inverse_transform",
        ):
            outer_msg = f"has no attribute '{fn_name}'"
            inner_msg = (
                f"`refit=False`. {fn_name} is available only after "
                "refitting on the best parameters"
            )
            # 使用 pytest 来检查是否抛出 AttributeError 异常，异常消息应包含 outer_msg
            with pytest.raises(AttributeError, match=outer_msg) as exec_info:
                getattr(grid_search, fn_name)(X)

            # 断言异常的原因是 AttributeError
            assert isinstance(exec_info.value.__cause__, AttributeError)
            # 断言异常消息中包含 inner_msg
            assert inner_msg in str(exec_info.value.__cause__)

    # 测试当 refit 参数设置为无效值时是否会引发适当的错误消息
    error_msg = (
        "For multi-metric scoring, the parameter refit must be set to a scorer key"
    )
    for refit in [True, "recall", "accuracy"]:
        # 使用 pytest 来检查是否抛出 ValueError 异常，异常消息应匹配 error_msg
        with pytest.raises(ValueError, match=error_msg):
            # 创建一个 GridSearchCV 对象，clf 和一个空参数字典，以及指定的 refit 参数和多指标评分
            GridSearchCV(
                clf, {}, refit=refit, scoring={"acc": "accuracy", "prec": "precision"}
            ).fit(X, y)
def test_grid_search_error():
    # 测试网格搜索是否能够捕获数据长度不同的错误
    X_, y_ = make_classification(n_samples=200, n_features=100, random_state=0)

    # 初始化线性支持向量机分类器
    clf = LinearSVC()

    # 创建网格搜索交叉验证对象，尝试不同的参数C：[0.1, 1.0]
    cv = GridSearchCV(clf, {"C": [0.1, 1.0]})

    # 使用断言检测是否会抛出 ValueError 异常
    with pytest.raises(ValueError):
        # 对数据的前180个样本进行拟合
        cv.fit(X_[:180], y_)


def test_grid_search_one_grid_point():
    # 创建数据集
    X_, y_ = make_classification(n_samples=200, n_features=100, random_state=0)

    # 定义参数字典
    param_dict = {"C": [1.0], "kernel": ["rbf"], "gamma": [0.1]}

    # 初始化支持向量机分类器
    clf = SVC(gamma="auto")

    # 创建网格搜索交叉验证对象，参数来自param_dict
    cv = GridSearchCV(clf, param_dict)

    # 对整个数据集进行拟合
    cv.fit(X_, y_)

    # 用特定参数再次拟合支持向量机分类器
    clf = SVC(C=1.0, kernel="rbf", gamma=0.1)
    clf.fit(X_, y_)

    # 断言两个数组是否相等
    assert_array_equal(clf.dual_coef_, cv.best_estimator_.dual_coef_)


def test_grid_search_when_param_grid_includes_range():
    # 测试最佳估计器是否包含正确的foo_param值
    clf = MockClassifier()

    # 使用网格搜索交叉验证对象，测试参数foo_param的取值范围为1到3
    grid_search = GridSearchCV(clf, {"foo_param": range(1, 4)}, cv=3)
    grid_search.fit(X, y)

    # 断言最佳估计器的foo_param值是否等于2
    assert grid_search.best_estimator_.foo_param == 2


def test_grid_search_bad_param_grid():
    # 创建小样本数据集
    X, y = make_classification(n_samples=10, n_features=5, random_state=0)

    # 定义参数字典，这里意图导致类型错误
    param_dict = {"C": 1}

    # 初始化支持向量机分类器
    clf = SVC(gamma="auto")

    # 准备错误消息
    error_msg = re.escape(
        "Parameter grid for parameter 'C' needs to be a list or "
        "a numpy array, but got 1 (of type int) instead. Single "
        "values need to be wrapped in a list with one element."
    )

    # 创建网格搜索交叉验证对象，检测是否会抛出预期的类型错误异常
    search = GridSearchCV(clf, param_dict)
    with pytest.raises(TypeError, match=error_msg):
        search.fit(X, y)

    # 定义参数字典，这里意图导致值错误
    param_dict = {"C": []}

    # 准备错误消息
    error_msg = re.escape(
        "Parameter grid for parameter 'C' need to be a non-empty sequence, got: []"
    )

    # 创建网格搜索交叉验证对象，检测是否会抛出预期的值错误异常
    search = GridSearchCV(clf, param_dict)
    with pytest.raises(ValueError, match=error_msg):
        search.fit(X, y)

    # 定义参数字典，这里意图导致类型错误
    param_dict = {"C": "1,2,3"}

    # 准备错误消息
    error_msg = re.escape(
        "Parameter grid for parameter 'C' needs to be a list or a numpy array, "
        "but got '1,2,3' (of type str) instead. Single values need to be "
        "wrapped in a list with one element."
    )

    # 创建网格搜索交叉验证对象，检测是否会抛出预期的类型错误异常
    search = GridSearchCV(clf, param_dict)
    with pytest.raises(TypeError, match=error_msg):
        search.fit(X, y)

    # 定义参数字典，这里意图导致值错误
    param_dict = {"C": np.ones((3, 2))}

    # 创建网格搜索交叉验证对象，检测是否会抛出预期的值错误异常
    search = GridSearchCV(clf, param_dict)
    with pytest.raises(ValueError):
        search.fit(X, y)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_grid_search_sparse(csr_container):
    # 测试网格搜索是否能够处理稠密和稀疏矩阵
    X_, y_ = make_classification(n_samples=200, n_features=100, random_state=0)

    # 初始化线性支持向量机分类器
    clf = LinearSVC()

    # 创建网格搜索交叉验证对象，尝试不同的参数C：[0.1, 1.0]
    cv = GridSearchCV(clf, {"C": [0.1, 1.0]})

    # 对数据的前180个样本进行拟合
    cv.fit(X_[:180], y_[:180])

    # 使用最佳估计器进行预测
    y_pred = cv.predict(X_[180:])

    # 获取最佳模型的参数C
    C = cv.best_estimator_.C

    # 将数据转换为csr格式
    X_ = csr_container(X_)

    # 初始化线性支持向量机分类器
    clf = LinearSVC()

    # 创建网格搜索交叉验证对象，尝试不同的参数C：[0.1, 1.0]
    cv = GridSearchCV(clf, {"C": [0.1, 1.0]})

    # 对数据的前180个样本进行拟合，将稠密数据转换为coo格式
    cv.fit(X_[:180].tocoo(), y_[:180])
    # 使用交叉验证模型 `cv` 对输入数据 `X_[180:]` 进行预测，得到预测结果 `y_pred2`
    y_pred2 = cv.predict(X_[180:])
    
    # 从交叉验证模型 `cv` 的最佳估计器中获取参数 `C` 的值
    C2 = cv.best_estimator_.C
    
    # 断言确保前后两次预测结果 `y_pred` 和 `y_pred2` 的平均值大于等于 0.9
    assert np.mean(y_pred == y_pred2) >= 0.9
    
    # 断言确保外部传入的参数 `C` 与 `cv` 模型中最佳参数 `C2` 相等
    assert C == C2
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_grid_search_sparse_scoring(csr_container):
    # 创建用于测试的样本数据集
    X_, y_ = make_classification(n_samples=200, n_features=100, random_state=0)

    # 初始化线性支持向量机分类器
    clf = LinearSVC()

    # 使用网格搜索交叉验证来选择最优的超参数 C，评分指标为 F1 分数
    cv = GridSearchCV(clf, {"C": [0.1, 1.0]}, scoring="f1")

    # 在部分数据集上拟合模型
    cv.fit(X_[:180], y_[:180])

    # 对剩余数据集进行预测
    y_pred = cv.predict(X_[180:])

    # 获取最优模型的超参数 C
    C = cv.best_estimator_.C

    # 使用 csr_container 转换输入数据 X_
    X_ = csr_container(X_)

    # 重新初始化线性支持向量机分类器
    clf = LinearSVC()

    # 再次进行网格搜索交叉验证，选择最优的超参数 C，评分指标为 F1 分数
    cv = GridSearchCV(clf, {"C": [0.1, 1.0]}, scoring="f1")

    # 在转换后的部分数据集上拟合模型
    cv.fit(X_[:180], y_[:180])

    # 对剩余数据集进行预测
    y_pred2 = cv.predict(X_[180:])

    # 获取最优模型的超参数 C
    C2 = cv.best_estimator_.C

    # 断言两次预测结果相同
    assert_array_equal(y_pred, y_pred2)

    # 断言两次最优超参数 C 相同
    assert C == C2

    # Smoke test the score
    # np.testing.assert_allclose(f1_score(cv.predict(X_[:180]), y[:180]),
    #                            cv.score(X_[:180], y[:180]))

    # test loss where greater is worse
    # 定义一个 F1 分数的损失函数，使得分数越小越好
    def f1_loss(y_true_, y_pred_):
        return -f1_score(y_true_, y_pred_)

    # 创建一个使得 F1Loss 越小越好的评分器
    F1Loss = make_scorer(f1_loss, greater_is_better=False)

    # 使用 F1Loss 作为评分指标进行网格搜索交叉验证，选择最优的超参数 C
    cv = GridSearchCV(clf, {"C": [0.1, 1.0]}, scoring=F1Loss)

    # 在部分数据集上拟合模型
    cv.fit(X_[:180], y_[:180])

    # 对剩余数据集进行预测
    y_pred3 = cv.predict(X_[180:])

    # 获取最优模型的超参数 C
    C3 = cv.best_estimator_.C

    # 断言第一次和最后一次选择的最优超参数 C 相同
    assert C == C3

    # 断言第一次和最后一次预测结果相同
    assert_array_equal(y_pred, y_pred3)


def test_grid_search_precomputed_kernel():
    # 测试当输入特征以预计算核矩阵的形式给出时，网格搜索工作正常
    X_, y_ = make_classification(n_samples=200, n_features=100, random_state=0)

    # 计算线性核对应的训练核矩阵
    K_train = np.dot(X_[:180], X_[:180].T)
    y_train = y_[:180]

    # 初始化支持向量机分类器，核函数为预计算核
    clf = SVC(kernel="precomputed")

    # 使用网格搜索交叉验证来选择最优的超参数 C
    cv = GridSearchCV(clf, {"C": [0.1, 1.0]})

    # 在训练核矩阵上拟合模型
    cv.fit(K_train, y_train)

    # 断言最佳得分不小于 0
    assert cv.best_score_ >= 0

    # 计算测试核矩阵
    K_test = np.dot(X_[180:], X_[:180].T)
    y_test = y_[180:]

    # 对测试核矩阵进行预测
    y_pred = cv.predict(K_test)

    # 断言预测精度不小于 0
    assert np.mean(y_pred == y_test) >= 0

    # test error is raised when the precomputed kernel is not array-like
    # or sparse
    # 当预计算核不是类数组或稀疏矩阵时，验证应该抛出 ValueError 错误
    with pytest.raises(ValueError):
        cv.fit(K_train.tolist(), y_train)


def test_grid_search_precomputed_kernel_error_nonsquare():
    # 测试当训练核矩阵是非方阵时，网格搜索返回错误
    K_train = np.zeros((10, 20))
    y_train = np.ones((10,))

    # 初始化支持向量机分类器，核函数为预计算核
    clf = SVC(kernel="precomputed")
    cv = GridSearchCV(clf, {"C": [0.1, 1.0]})

    # 断言应该抛出 ValueError 错误
    with pytest.raises(ValueError):
        cv.fit(K_train, y_train)


class BrokenClassifier(BaseEstimator):
    """Broken classifier that cannot be fit twice"""

    def __init__(self, parameter=None):
        self.parameter = parameter

    def fit(self, X, y):
        # 确保模型只能被拟合一次
        assert not hasattr(self, "has_been_fit_")
        self.has_been_fit_ = True

    def predict(self, X):
        # 预测返回全零数组
        return np.zeros(X.shape[0])


@ignore_warnings
def test_refit():
    # 重现对重拟合的 bug
    # 模拟对一个无法重复拟合的损坏分类器进行重拟合；以前在稀疏 SVM 中会出问题。
    X = np.arange(100).reshape(10, 10)
    # 创建一个包含10个元素的 NumPy 数组，前5个元素为 0，后5个元素为 1
    y = np.array([0] * 5 + [1] * 5)
    
    # 使用 GridSearchCV 进行网格搜索交叉验证，寻找最佳参数
    clf = GridSearchCV(
        # 使用 BrokenClassifier 作为待优化的分类器
        BrokenClassifier(),
        # 参数网格包含一个字典，其中 parameter 参数可选值为 0 或 1
        [{"parameter": [0, 1]}],
        # 使用 precision 作为评分指标，优化时选择精确度最高的参数
        scoring="precision",
        # 设置 refit=True，使用找到的最佳参数重新拟合整个数据集
        refit=True
    )
    # 对输入数据集 X 和目标变量 y 进行拟合和优化
    clf.fit(X, y)
# 测试 `refit=callable` 的功能，该选项增加了对识别“最佳”估计器的灵活性。
def test_refit_callable():
    """
    Test refit=callable, which adds flexibility in identifying the
    "best" estimator.
    """

    # 定义一个测试 `refit=callable` 接口的虚拟函数
    def refit_callable(cv_results):
        """
        A dummy function tests `refit=callable` interface.
        Return the index of a model that has the least
        `mean_test_score`.
        """
        # 使用 `refit=True` 来拟合一个虚拟的分类器以获取 `clf.cv_results_` 中的键列表。
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        clf = GridSearchCV(
            LinearSVC(random_state=42),
            {"C": [0.01, 0.1, 1]},
            scoring="precision",
            refit=True,
        )
        clf.fit(X, y)
        # 确保对于这个虚拟分类器，`best_index_ != 0`
        assert clf.best_index_ != 0

        # 断言每个键都与 `cv_results` 中的键匹配
        for key in clf.cv_results_.keys():
            assert key in cv_results

        # 返回具有最小 `mean_test_score` 的模型的索引
        return cv_results["mean_test_score"].argmin()

    # 创建虚拟数据集
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    # 使用 `refit=callable` 来创建一个 GridSearchCV 对象
    clf = GridSearchCV(
        LinearSVC(random_state=42),
        {"C": [0.01, 0.1, 1]},
        scoring="precision",
        refit=refit_callable,
    )
    clf.fit(X, y)

    # 确保 `best_index_ == 0`
    assert clf.best_index_ == 0
    # 确保在使用 `refit=callable` 时 `best_score_` 被禁用
    assert not hasattr(clf, "best_score_")


# 测试实现在 `best_index_` 返回无效结果时的错误处理
def test_refit_callable_invalid_type():
    """
    Test implementation catches the errors when 'best_index_' returns an
    invalid result.
    """

    # 定义一个测试当返回的 `best_index_` 不是整数时的虚拟函数
    def refit_callable_invalid_type(cv_results):
        """
        A dummy function tests when returned 'best_index_' is not integer.
        """
        return None

    # 创建虚拟数据集
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)

    # 使用 `refit=callable_invalid_type` 来创建一个 GridSearchCV 对象
    clf = GridSearchCV(
        LinearSVC(random_state=42),
        {"C": [0.1, 1]},
        scoring="precision",
        refit=refit_callable_invalid_type,
    )
    # 确保抛出预期的 TypeError 异常
    with pytest.raises(TypeError, match="best_index_ returned is not an integer"):
        clf.fit(X, y)


# 使用参数化测试，测试当 `best_index_` 返回超出范围结果时的错误处理
@pytest.mark.parametrize("out_bound_value", [-1, 2])
@pytest.mark.parametrize("search_cv", [RandomizedSearchCV, GridSearchCV])
def test_refit_callable_out_bound(out_bound_value, search_cv):
    """
    Test implementation catches the errors when 'best_index_' returns an
    out of bound result.
    """

    # 定义一个测试当返回的 `best_index_` 超出范围时的虚拟函数
    def refit_callable_out_bound(cv_results):
        """
        A dummy function tests when returned 'best_index_' is out of bounds.
        """
        return out_bound_value

    # 创建虚拟数据集
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)

    # 使用参数化的 `search_cv` 类来创建一个 GridSearchCV 或 RandomizedSearchCV 对象
    clf = search_cv(
        LinearSVC(random_state=42),
        {"C": [0.1, 1]},
        scoring="precision",
        refit=refit_callable_out_bound,
    )
    # 确保抛出预期的 IndexError 异常
    with pytest.raises(IndexError, match="best_index_ index out of range"):
        clf.fit(X, y)


# 测试 `refit=callable` 在多指标评估设置中的应用
def test_refit_callable_multi_metric():
    """
    Test refit=callable in multiple metric evaluation setting
    """
    # 定义一个函数 refit_callable，用于 GridSearchCV 的 refit 参数，返回具有最小 mean_test_prec 的模型索引
    def refit_callable(cv_results):
        """
        A dummy function tests `refit=callable` interface.
        Return the index of a model that has the least
        `mean_test_prec`.
        """
        # 断言确保 cv_results 包含 mean_test_prec 字段
        assert "mean_test_prec" in cv_results
        # 返回 mean_test_prec 最小值对应的索引
        return cv_results["mean_test_prec"].argmin()

    # 生成一个简单的分类数据集 X 和对应的标签 y
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    # 定义评分指标，包括 Accuracy 和 prec（precision）
    scoring = {"Accuracy": make_scorer(accuracy_score), "prec": "precision"}
    # 创建一个 GridSearchCV 对象 clf，使用 LinearSVC 模型，调优参数 C=[0.01, 0.1, 1]，指定 scoring 和 refit=refit_callable
    clf = GridSearchCV(
        LinearSVC(random_state=42),
        {"C": [0.01, 0.1, 1]},
        scoring=scoring,
        refit=refit_callable,
    )
    # 在数据集 X, y 上拟合 GridSearchCV 对象 clf
    clf.fit(X, y)

    # 断言确保最佳模型索引为 0
    assert clf.best_index_ == 0
    # 确保在使用 refit=callable 时，best_score_ 属性未启用
    assert not hasattr(clf, "best_score_")
def test_gridsearch_nd():
    # Pass X as a 4-dimensional array
    X_4d = np.arange(10 * 5 * 3 * 2).reshape(10, 5, 3, 2)
    # Create y as a 3-dimensional array
    y_3d = np.arange(10 * 7 * 11).reshape(10, 7, 11)

    def check_X(x):
        return x.shape[1:] == (5, 3, 2)

    def check_y(x):
        return x.shape[1:] == (7, 11)

    # Initialize CheckingClassifier with custom check functions
    clf = CheckingClassifier(
        check_X=check_X,
        check_y=check_y,
        methods_to_check=["fit"],
    )
    # Perform grid search using GridSearchCV
    grid_search = GridSearchCV(clf, {"foo_param": [1, 2, 3]})
    # Fit the grid search with X_4d and y_3d, score with original X and y
    grid_search.fit(X_4d, y_3d).score(X, y)
    # Assert that grid_search has 'cv_results_' attribute
    assert hasattr(grid_search, "cv_results_")


def test_X_as_list():
    # Pass X as list in GridSearchCV
    X = np.arange(100).reshape(10, 10)
    y = np.array([0] * 5 + [1] * 5)

    # Initialize CheckingClassifier with lambda function checking if X is a list
    clf = CheckingClassifier(
        check_X=lambda x: isinstance(x, list),
        methods_to_check=["fit"],
    )
    # Define KFold cross-validator
    cv = KFold(n_splits=3)
    # Perform grid search using GridSearchCV with X converted to list
    grid_search = GridSearchCV(clf, {"foo_param": [1, 2, 3]}, cv=cv)
    # Fit the grid search with X converted to list and original y, score with original X and y
    grid_search.fit(X.tolist(), y).score(X, y)
    # Assert that grid_search has 'cv_results_' attribute
    assert hasattr(grid_search, "cv_results_")


def test_y_as_list():
    # Pass y as list in GridSearchCV
    X = np.arange(100).reshape(10, 10)
    y = np.array([0] * 5 + [1] * 5)

    # Initialize CheckingClassifier with lambda function checking if y is a list
    clf = CheckingClassifier(
        check_y=lambda x: isinstance(x, list),
        methods_to_check=["fit"],
    )
    # Define KFold cross-validator
    cv = KFold(n_splits=3)
    # Perform grid search using GridSearchCV with original X and y converted to list
    grid_search = GridSearchCV(clf, {"foo_param": [1, 2, 3]}, cv=cv)
    # Fit the grid search with original X and y converted to list, score with original X and y
    grid_search.fit(X, y.tolist()).score(X, y)
    # Assert that grid_search has 'cv_results_' attribute
    assert hasattr(grid_search, "cv_results_")


@ignore_warnings
def test_pandas_input():
    # check cross_val_score doesn't destroy pandas dataframe
    types = [(MockDataFrame, MockDataFrame)]
    try:
        from pandas import DataFrame, Series

        types.append((DataFrame, Series))
    except ImportError:
        pass

    X = np.arange(100).reshape(10, 10)
    y = np.array([0] * 5 + [1] * 5)

    for InputFeatureType, TargetType in types:
        # X dataframe, y series
        X_df, y_ser = InputFeatureType(X), TargetType(y)

        def check_df(x):
            return isinstance(x, InputFeatureType)

        def check_series(x):
            return isinstance(x, TargetType)

        # Initialize CheckingClassifier with custom check functions
        clf = CheckingClassifier(check_X=check_df, check_y=check_series)

        # Perform grid search using GridSearchCV
        grid_search = GridSearchCV(clf, {"foo_param": [1, 2, 3]})
        # Fit the grid search with X_df and y_ser, score with X_df and y_ser
        grid_search.fit(X_df, y_ser).score(X_df, y_ser)
        # Predict with X_df
        grid_search.predict(X_df)
        # Assert that grid_search has 'cv_results_' attribute
        assert hasattr(grid_search, "cv_results_")


def test_unsupervised_grid_search():
    # test grid-search with unsupervised estimator
    X, y = make_blobs(n_samples=50, random_state=0)
    km = KMeans(random_state=0, init="random", n_init=1)

    # Multi-metric evaluation unsupervised
    scoring = ["adjusted_rand_score", "fowlkes_mallows_score"]
    # 对于每个评分指标 refit 在 ["adjusted_rand_score", "fowlkes_mallows_score"] 中，
    # 创建一个 GridSearchCV 对象，用于寻找最优的聚类数。
    for refit in ["adjusted_rand_score", "fowlkes_mallows_score"]:
        grid_search = GridSearchCV(
            km, param_grid=dict(n_clusters=[2, 3, 4]), scoring=scoring, refit=refit
        )
        # 在给定数据集 X 和标签 y 上进行网格搜索和拟合
        grid_search.fit(X, y)
        # 断言找到的最佳参数中聚类数为 3
        assert grid_search.best_params_["n_clusters"] == 3

    # 在无监督学习场景下，使用单一评分指标 "fowlkes_mallows_score" 进行网格搜索
    grid_search = GridSearchCV(
        km, param_grid=dict(n_clusters=[2, 3, 4]), scoring="fowlkes_mallows_score"
    )
    # 在给定数据集 X 和标签 y 上进行网格搜索和拟合
    grid_search.fit(X, y)
    # 断言找到的最佳参数中聚类数为 3
    assert grid_search.best_params_["n_clusters"] == 3

    # 在没有指定评分指标和标签 y 的情况下，进行网格搜索
    grid_search = GridSearchCV(km, param_grid=dict(n_clusters=[2, 3, 4]))
    # 在给定数据集 X 上进行无监督的网格搜索和拟合
    grid_search.fit(X)
    # 断言找到的最佳参数中聚类数为 4
    assert grid_search.best_params_["n_clusters"] == 4
# 定义一个测试函数，用于测试没有预测功能的估算器的网格搜索
def test_gridsearch_no_predict():
    # 定义一个自定义的评分函数，根据估算器的带宽参数返回不同的分数
    def custom_scoring(estimator, X):
        return 42 if estimator.bandwidth == 0.1 else 0

    # 生成一个样本数据集 X，不使用对应的标签 _
    X, _ = make_blobs(cluster_std=0.1, random_state=1, centers=[[0, 1], [1, 0], [0, 0]])
    
    # 创建一个网格搜索对象，使用 KernelDensity 作为估算器，搜索带宽参数
    search = GridSearchCV(
        KernelDensity(),  # 使用默认的 KernelDensity 作为估算器
        param_grid=dict(bandwidth=[0.01, 0.1, 1]),  # 定义带宽参数的搜索范围
        scoring=custom_scoring,  # 指定自定义的评分函数
    )
    
    # 在样本数据集上执行网格搜索
    search.fit(X)
    
    # 断言最佳带宽参数应为 0.1
    assert search.best_params_["bandwidth"] == 0.1
    # 断言最佳分数应为 42
    assert search.best_score_ == 42


# 定义一个测试函数，用于测试参数采样器的基本属性
def test_param_sampler():
    # 定义参数分布字典，包括 kernel 和 C 参数
    param_distributions = {"kernel": ["rbf", "linear"], "C": uniform(0, 1)}
    
    # 创建参数采样器对象，从参数分布中生成 10 组参数样本
    sampler = ParameterSampler(
        param_distributions=param_distributions,  # 参数分布定义
        n_iter=10,  # 生成的样本数
        random_state=0  # 随机种子
    )
    
    # 生成样本列表
    samples = [x for x in sampler]
    
    # 断言生成的样本数量应为 10
    assert len(samples) == 10
    
    # 对每个样本进行断言，kernel 参数应为 "rbf" 或 "linear"，C 参数应在 [0, 1] 范围内
    for sample in samples:
        assert sample["kernel"] in ["rbf", "linear"]
        assert 0 <= sample["C"] <= 1
    
    # 测试多次调用采样器生成的参数是否一致
    param_distributions = {"C": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    sampler = ParameterSampler(
        param_distributions=param_distributions,  # 参数分布定义
        n_iter=3,  # 生成的样本数
        random_state=0  # 随机种子
    )
    
    # 断言两次生成的参数列表是否相同
    assert [x for x in sampler] == [x for x in sampler]
    
    # 使用 uniform 分布的参数采样器，生成 10 组参数样本
    param_distributions = {"C": uniform(0, 1)}
    sampler = ParameterSampler(
        param_distributions=param_distributions,  # 参数分布定义
        n_iter=10,  # 生成的样本数
        random_state=0  # 随机种子
    )
    
    # 断言两次生成的参数列表是否相同
    assert [x for x in sampler] == [x for x in sampler]


# 定义一个函数，用于检查搜索结果的数组类型是否正确
def check_cv_results_array_types(
    search, param_keys, score_keys, expected_cv_results_kinds
):
    # 获取搜索结果的 cv_results 属性
    cv_results = search.cv_results_
    
    # 断言所有参数键的值都是 np.ma.MaskedArray 类型
    assert all(isinstance(cv_results[param], np.ma.MaskedArray) for param in param_keys)
    
    # 断言参数键的类型符合预期的 cv_results 类型
    assert {
        key: cv_results[key].dtype.kind for key in param_keys
    } == expected_cv_results_kinds
    
    # 断言所有分数键的值不是 np.ma.MaskedArray 类型
    assert not any(isinstance(cv_results[key], np.ma.MaskedArray) for key in score_keys)
    
    # 断言所有分数键的类型为 np.float64，但不包括以 "rank" 开头的键
    assert all(
        cv_results[key].dtype == np.float64
        for key in score_keys
        if not key.startswith("rank")
    )
    
    # 如果搜索是多指标的，则对每个评分键进行检查
    scorer_keys = search.scorer_.keys() if search.multimetric_ else ["score"]
    for key in scorer_keys:
        # 断言 "rank_test_%s" 键的类型为 np.int32
        assert cv_results["rank_test_%s" % key].dtype == np.int32


# 定义一个函数，用于检查搜索结果的键是否完整
def check_cv_results_keys(cv_results, param_keys, score_keys, n_cand, extra_keys=()):
    # 检查 cv_results 是否包含所有必需的结果键
    all_keys = param_keys + score_keys + extra_keys
    assert_array_equal(sorted(cv_results.keys()), sorted(all_keys + ("params",)))
    
    # 断言每个参数键和分数键的数组形状为 (n_cand,)
    assert all(cv_results[key].shape == (n_cand,) for key in param_keys + score_keys)


# 定义一个测试函数，用于测试网格搜索的 cv_results 结果
def test_grid_search_cv_results():
    # 生成一个分类数据集 X 和对应的标签 y
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)

    # 定义网格搜索中参数网格的数量
    n_grid_points = 6
    # 定义一个包含参数网格的列表，每个元素是一个字典，表示不同的参数组合
    params = [
        dict(
            kernel=[
                "rbf",
            ],
            C=[1, 10],
            gamma=[0.1, 1],
        ),
        dict(
            kernel=[
                "poly",
            ],
            degree=[1, 2],
        ),
    ]

    # 定义参数键的元组，用于检索交叉验证结果的参数信息
    param_keys = ("param_C", "param_degree", "param_gamma", "param_kernel")

    # 定义评分键的元组，包含了不同的交叉验证评分和时间信息
    score_keys = (
        "mean_test_score",
        "mean_train_score",
        "rank_test_score",
        "split0_test_score",
        "split1_test_score",
        "split2_test_score",
        "split0_train_score",
        "split1_train_score",
        "split2_train_score",
        "std_test_score",
        "std_train_score",
        "mean_fit_time",
        "std_fit_time",
        "mean_score_time",
        "std_score_time",
    )

    # 设置候选参数的数量为网格点的数量
    n_candidates = n_grid_points

    # 使用GridSearchCV进行参数网格搜索，使用SVC作为基础分类器，3折交叉验证
    search = GridSearchCV(SVC(), cv=3, param_grid=params, return_train_score=True)
    search.fit(X, y)
    # 获取交叉验证结果
    cv_results = search.cv_results_

    # 检查评分和时间是否在合理范围内
    assert all(cv_results["rank_test_score"] >= 1)
    assert (all(cv_results[k] >= 0) for k in score_keys if k != "rank_test_score")
    assert (
        all(cv_results[k] <= 1)
        for k in score_keys
        if "time" not in k and k != "rank_test_score"
    )

    # 检查cv_results的结构是否符合预期
    expected_cv_results_kinds = {
        "param_C": "i",
        "param_degree": "i",
        "param_gamma": "f",
        "param_kernel": "O",
    }
    check_cv_results_array_types(
        search, param_keys, score_keys, expected_cv_results_kinds
    )
    
    # 检查cv_results中是否包含期望的参数键和评分键，以及候选参数的数量
    check_cv_results_keys(cv_results, param_keys, score_keys, n_candidates)

    # 再次获取cv_results，用于进行遮蔽检查
    cv_results = search.cv_results_

    # 检查poly核函数的结果是否符合预期
    poly_results = [
        (
            cv_results["param_C"].mask[i]
            and cv_results["param_gamma"].mask[i]
            and not cv_results["param_degree"].mask[i]
        )
        for i in range(n_candidates)
        if cv_results["param_kernel"][i] == "poly"
    ]
    assert all(poly_results)
    assert len(poly_results) == 2

    # 检查rbf核函数的结果是否符合预期
    rbf_results = [
        (
            not cv_results["param_C"].mask[i]
            and not cv_results["param_gamma"].mask[i]
            and cv_results["param_degree"].mask[i]
        )
        for i in range(n_candidates)
        if cv_results["param_kernel"][i] == "rbf"
    ]
    assert all(rbf_results)
    assert len(rbf_results) == 4
def test_random_search_cv_results():
    # 生成一个包含50个样本和4个特征的分类数据集，使用固定的随机种子以保证结果的可重复性
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)

    # 设置随机搜索的迭代次数
    n_search_iter = 30

    # 定义参数空间，包括两组参数：一组用于rbf核函数的SVC，另一组用于多项式核函数的SVC
    params = [
        {"kernel": ["rbf"], "C": expon(scale=10), "gamma": expon(scale=0.1)},
        {"kernel": ["poly"], "degree": [2, 3]},
    ]

    # 定义参数键的顺序
    param_keys = ("param_C", "param_degree", "param_gamma", "param_kernel")

    # 定义分数键的顺序，用于评估模型性能
    score_keys = (
        "mean_test_score",
        "mean_train_score",
        "rank_test_score",
        "split0_test_score",
        "split1_test_score",
        "split2_test_score",
        "split0_train_score",
        "split1_train_score",
        "split2_train_score",
        "std_test_score",
        "std_train_score",
        "mean_fit_time",
        "std_fit_time",
        "mean_score_time",
        "std_score_time",
    )

    # 候选模型的数量等于搜索迭代次数
    n_candidates = n_search_iter

    # 创建随机搜索对象，使用SVC作为基础模型
    search = RandomizedSearchCV(
        SVC(),
        n_iter=n_search_iter,
        cv=3,
        param_distributions=params,
        return_train_score=True,
    )

    # 在数据集上执行随机搜索
    search.fit(X, y)
    cv_results = search.cv_results_

    # 检查交叉验证结果的结构
    expected_cv_results_kinds = {
        "param_C": "f",          # C参数应为浮点数
        "param_degree": "i",     # degree参数应为整数
        "param_gamma": "f",      # gamma参数应为浮点数
        "param_kernel": "O",     # kernel参数为对象类型（字符串）
    }
    check_cv_results_array_types(
        search, param_keys, score_keys, expected_cv_results_kinds
    )

    # 检查交叉验证结果的键是否符合预期，并验证候选模型的数量
    check_cv_results_keys(cv_results, param_keys, score_keys, n_candidates)

    # 断言：对于poly核函数，param_C和param_gamma应该存在，而param_degree不应该存在
    assert all(
        (
            cv_results["param_C"].mask[i]
            and cv_results["param_gamma"].mask[i]
            and not cv_results["param_degree"].mask[i]
        )
        for i in range(n_candidates)
        if cv_results["param_kernel"][i] == "poly"
    )

    # 断言：对于rbf核函数，param_C和param_gamma不应该存在，而param_degree应该存在
    assert all(
        (
            not cv_results["param_C"].mask[i]
            and not cv_results["param_gamma"].mask[i]
            and cv_results["param_degree"].mask[i]
        )
        for i in range(n_candidates)
        if cv_results["param_kernel"][i] == "rbf"
    )


@pytest.mark.parametrize(
    "SearchCV, specialized_params",
    [
        (GridSearchCV, {"param_grid": {"C": [1, 10]}}),  # 使用GridSearchCV，指定C参数为1和10
        (RandomizedSearchCV, {"param_distributions": {"C": [1, 10]}, "n_iter": 2}),  # 使用RandomizedSearchCV，指定C参数分布为1和10，迭代次数为2
    ],
)
def test_search_default_iid(SearchCV, specialized_params):
    # 测试IID参数 TODO: 显然，这个测试做了其他事情???
    
    # 创建包含四个中心点的带有噪声的二维数据集
    X, y = make_blobs(
        centers=[[0, 0], [1, 0], [0, 1], [1, 1]],
        random_state=0,
        cluster_std=0.1,
        shuffle=False,
        n_samples=80,
    )

    # 将数据集划分为不符合iid的两个fold
    mask = np.ones(X.shape[0], dtype=bool)
    mask[np.where(y == 1)[0][::2]] = 0
    mask[np.where(y == 2)[0][::2]] = 0

    # 创建CV拆分
    cv = [[mask, ~mask], [~mask, mask]]
    # 定义通用的参数字典，包括使用默认的 SVC 评估器、交叉验证对象 cv 和返回训练分数
    common_params = {"estimator": SVC(), "cv": cv, "return_train_score": True}
    # 使用 common_params 和 specialized_params 创建一个 SearchCV 对象
    search = SearchCV(**common_params, **specialized_params)
    # 对数据集 X 和标签 y 进行搜索和拟合
    search.fit(X, y)

    # 提取测试集的交叉验证分数数组，分数来源于每个分割的第一个结果
    test_cv_scores = np.array(
        [
            search.cv_results_["split%d_test_score" % s][0]
            for s in range(search.n_splits_)
        ]
    )
    # 提取测试集的平均分数
    test_mean = search.cv_results_["mean_test_score"][0]
    # 提取测试集的标准差分数
    test_std = search.cv_results_["std_test_score"][0]

    # 提取训练集的交叉验证分数数组，分数来源于每个分割的第一个结果
    train_cv_scores = np.array(
        [
            search.cv_results_["split%d_train_score" % s][0]
            for s in range(search.n_splits_)
        ]
    )
    # 提取训练集的平均分数
    train_mean = search.cv_results_["mean_train_score"][0]
    # 提取训练集的标准差分数
    train_std = search.cv_results_["std_train_score"][0]

    # 断言参数 C 的第一个值为 1
    assert search.cv_results_["param_C"][0] == 1
    # 断言测试集的交叉验证分数与预期值接近
    assert_allclose(test_cv_scores, [1, 1.0 / 3.0])
    # 断言训练集的交叉验证分数与预期值接近
    assert_allclose(train_cv_scores, [1, 1])
    # 断言测试集的平均分数与测试集交叉验证分数的平均值接近
    assert test_mean == pytest.approx(np.mean(test_cv_scores))
    # 断言测试集的标准差与测试集交叉验证分数的标准差接近
    assert test_std == pytest.approx(np.std(test_cv_scores))

    # 断言训练集的平均分数与预期值接近，不考虑是否 i.i.d.
    assert train_mean == pytest.approx(1)
    # 断言训练集的标准差与预期值接近，不考虑是否 i.i.d.
    assert train_std == pytest.approx(0)
def compare_cv_results_multimetric_with_single(search_multi, search_acc, search_rec):
    """比较多指标的交叉验证结果与从单指标网格/随机搜索中得到的多个单指标结果的集成。

    Args:
        search_multi: 多指标搜索的GridSearchCV或RandomizedSearchCV对象
        search_acc: 用于精度评分的单指标搜索的GridSearchCV或RandomizedSearchCV对象
        search_rec: 用于召回率评分的单指标搜索的GridSearchCV或RandomizedSearchCV对象

    Returns:
        None
    """

    assert search_multi.multimetric_  # 断言多指标搜索确实是多指标的
    assert_array_equal(sorted(search_multi.scorer_), ("accuracy", "recall"))  # 检查多指标的评分器确保包括精度和召回率

    # 获取多指标搜索的交叉验证结果
    cv_results_multi = search_multi.cv_results_

    # 从单指标的精度搜索结果中创建对应的精度结果
    cv_results_acc_rec = {
        re.sub("_score$", "_accuracy", k): v for k, v in search_acc.cv_results_.items()
    }

    # 从单指标的召回率搜索结果中创建对应的召回率结果，并与精度结果合并
    cv_results_acc_rec.update(
        {re.sub("_score$", "_recall", k): v for k, v in search_rec.cv_results_.items()}
    )

    # 检查评分和时间是否合理，同时检查键是否存在
    # 使用断言确保以下条件全部成立：
    # 对于 "mean_score_time", "std_score_time", "mean_fit_time", "std_fit_time" 这几个键，
    # 所有的值都应该小于等于 1。
    assert all(
        (
            np.all(cv_results_multi[k] <= 1)
            for k in (
                "mean_score_time",
                "std_score_time",
                "mean_fit_time",
                "std_fit_time",
            )
        )
    )
    
    # 比较多指标和单指标网格搜索结果中除时间相关键之外的所有键。
    # np.testing.assert_equal 对两个 cv_results 字典进行深度嵌套比较。
    np.testing.assert_equal(
        # 从 cv_results_multi 中选择那些不以 "_time" 结尾的键值对
        {k: v for k, v in cv_results_multi.items() if not k.endswith("_time")},
        # 从 cv_results_acc_rec 中选择那些不以 "_time" 结尾的键值对
        {k: v for k, v in cv_results_acc_rec.items() if not k.endswith("_time")},
    )
# 比较使用不同 refit 设置的多指标搜索方法与单指标搜索方法的差异
def compare_refit_methods_when_refit_with_acc(search_multi, search_acc, refit):
    """Compare refit multi-metric search methods with single metric methods"""
    # 断言多指标搜索对象的 refit 属性与给定的 refit 参数相同
    assert search_acc.refit == refit
    if refit:
        # 如果 refit 为 True，则断言多指标搜索对象的 refit 属性应为 "accuracy"
        assert search_multi.refit == "accuracy"
    else:
        # 如果 refit 为 False，则断言多指标搜索对象的 refit 属性应为 False
        assert not search_multi.refit
        return  # 没有 refit 时无法进行预测或评分

    # 创建一个包含 100 个样本和 4 个特征的数据集
    X, y = make_blobs(n_samples=100, n_features=4, random_state=42)
    # 对于每种方法（预测、概率预测、对数概率预测），断言多指标搜索对象与单指标搜索对象的结果几乎相等
    for method in ("predict", "predict_proba", "predict_log_proba"):
        assert_almost_equal(
            getattr(search_multi, method)(X), getattr(search_acc, method)(X)
        )
    # 断言多指标搜索对象与单指标搜索对象的评分结果几乎相等
    assert_almost_equal(search_multi.score(X, y), search_acc.score(X, y))
    # 断言多指标搜索对象与单指标搜索对象的最佳索引、最佳分数和最佳参数相等
    for key in ("best_index_", "best_score_", "best_params_"):
        assert getattr(search_multi, key) == getattr(search_acc, key)


@pytest.mark.parametrize(
    "search_cv",
    [
        RandomizedSearchCV(
            estimator=DecisionTreeClassifier(),
            param_distributions={"max_depth": [5, 10]},
        ),
        GridSearchCV(
            estimator=DecisionTreeClassifier(), param_grid={"max_depth": [5, 10]}
        ),
    ],
)
def test_search_cv_score_samples_error(search_cv):
    X, y = make_blobs(n_samples=100, n_features=4, random_state=42)
    search_cv.fit(X, y)

    # 确保当基础估计器不实现 `score_samples` 方法时报错
    outer_msg = f"'{search_cv.__class__.__name__}' has no attribute 'score_samples'"
    inner_msg = "'DecisionTreeClassifier' object has no attribute 'score_samples'"

    # 使用 pytest 的 assertRaises 来捕获 AttributeError 异常，确保其内部原因是 inner_msg 所描述的错误
    with pytest.raises(AttributeError, match=outer_msg) as exec_info:
        search_cv.score_samples(X)
    assert isinstance(exec_info.value.__cause__, AttributeError)
    assert inner_msg == str(exec_info.value.__cause__)


@pytest.mark.parametrize(
    "search_cv",
    [
        RandomizedSearchCV(
            estimator=LocalOutlierFactor(novelty=True),
            param_distributions={"n_neighbors": [5, 10]},
            scoring="precision",
        ),
        GridSearchCV(
            estimator=LocalOutlierFactor(novelty=True),
            param_grid={"n_neighbors": [5, 10]},
            scoring="precision",
        ),
    ],
)
def test_search_cv_score_samples_method(search_cv):
    # 设置参数
    rng = np.random.RandomState(42)
    n_samples = 300
    outliers_fraction = 0.15
    n_outliers = int(outliers_fraction * n_samples)
    n_inliers = n_samples - n_outliers

    # 创建数据集
    X = make_blobs(
        n_samples=n_inliers,
        n_features=2,
        centers=[[0, 0], [0, 0]],
        cluster_std=0.5,
        random_state=0,
    )[0]
    # 添加一些噪点
    X = np.concatenate([X, rng.uniform(low=-6, high=6, size=(n_outliers, 2))], axis=0)

    # 为了使用 `search_cv` 对估计器进行评分，定义标签 `y_true`
    y_true = np.array([1] * n_samples)
    y_true[-n_outliers:] = -1

    # 在数据上进行拟合
    search_cv.fit(X, y_true)
    # 验证独立的估计器与 *SearchCV 获得的结果是否一致
    # 使用 assert_allclose 函数检查两个结果是否在可接受的误差范围内相等，
    # 第一个参数是调用 *SearchCV 的 score_samples 方法得到的结果，
    # 第二个参数是从 *SearchCV 的最佳估计器（best_estimator_）调用 score_samples 方法得到的结果。
    assert_allclose(
        search_cv.score_samples(X), search_cv.best_estimator_.score_samples(X)
    )
def test_search_cv_results_rank_tie_breaking():
    X, y = make_blobs(n_samples=50, random_state=42)

    # 定义参数网格，包含几个接近的 C 值，可能会导致模型得分相似并且排名并列
    param_grid = {"C": [1, 1.001, 0.001]}

    # 创建网格搜索对象
    grid_search = GridSearchCV(SVC(), param_grid=param_grid, return_train_score=True)
    # 创建随机搜索对象
    random_search = RandomizedSearchCV(
        SVC(), n_iter=3, param_distributions=param_grid, return_train_score=True
    )

    # 对于每种搜索方法（网格搜索和随机搜索），执行拟合和评估
    for search in (grid_search, random_search):
        search.fit(X, y)
        # 获取交叉验证结果
        cv_results = search.cv_results_
        
        # 检查排名并列的情况 -
        # 检查第一和第二候选模型的平均测试分数是否相等
        assert_almost_equal(
            cv_results["mean_test_score"][0], cv_results["mean_test_score"][1]
        )
        # 检查第一和第二候选模型的平均训练分数是否相等
        assert_almost_equal(
            cv_results["mean_train_score"][0], cv_results["mean_train_score"][1]
        )
        # 检查第二和第三候选模型的平均测试分数是否不相等
        assert not np.allclose(
            cv_results["mean_test_score"][1], cv_results["mean_test_score"][2]
        )
        # 检查第二和第三候选模型的平均训练分数是否不相等
        assert not np.allclose(
            cv_results["mean_train_score"][1], cv_results["mean_train_score"][2]
        )
        # 应该给排名相同的候选模型分配最小的排名
        assert_almost_equal(search.cv_results_["rank_test_score"], [1, 1, 3])


def test_search_cv_results_none_param():
    X, y = [[1], [2], [3], [4], [5]], [0, 0, 0, 0, 1]
    estimators = (DecisionTreeRegressor(), DecisionTreeClassifier())
    est_parameters = {"random_state": [0, None]}
    cv = KFold()

    # 对于每个估算器，执行网格搜索
    for est in estimators:
        grid_search = GridSearchCV(
            est,
            est_parameters,
            cv=cv,
        ).fit(X, y)
        # 检查参数 random_state 的值是否与预期一致
        assert_array_equal(grid_search.cv_results_["param_random_state"], [0, None])


@ignore_warnings()
def test_search_cv_timing():
    svc = LinearSVC(random_state=0)

    X = [
        [
            1,
        ],
        [
            2,
        ],
        [
            3,
        ],
        [
            4,
        ],
    ]
    y = [0, 1, 1, 0]

    # 创建线性支持向量机的网格搜索和随机搜索对象
    gs = GridSearchCV(svc, {"C": [0, 1]}, cv=2, error_score=0)
    rs = RandomizedSearchCV(svc, {"C": [0, 1]}, cv=2, error_score=0, n_iter=2)

    # 对于每种搜索方法（网格搜索和随机搜索），执行拟合和评估
    for search in (gs, rs):
        search.fit(X, y)
        # 检查拟合时间和评分时间是否在合理范围内
        for key in ["mean_fit_time", "std_fit_time"]:
            # 注意：在 Windows 下，time.time 的精度不足以保证拟合/评分时间不为零
            assert np.all(search.cv_results_[key] >= 0)
            assert np.all(search.cv_results_[key] < 1)

        for key in ["mean_score_time", "std_score_time"]:
            assert search.cv_results_[key][1] >= 0
            assert search.cv_results_[key][0] == 0.0
            assert np.all(search.cv_results_[key] < 1)

        # 检查是否有 refit_time_ 属性，并且其类型为 float
        assert hasattr(search, "refit_time_")
        assert isinstance(search.refit_time_, float)
        assert search.refit_time_ >= 0
# 测试正确的网格搜索结果分数
def test_grid_search_correct_score_results():
    # 设置交叉验证的折数
    n_splits = 3
    # 初始化线性支持向量机分类器
    clf = LinearSVC(random_state=0)
    # 生成示例数据集 X, y
    X, y = make_blobs(random_state=0, centers=2)
    # 设置待搜索的超参数列表
    Cs = [0.1, 1, 10]
    # 遍历评分指标列表
    for score in ["f1", "roc_auc"]:
        # 创建网格搜索对象，传入分类器、超参数字典、评分指标和交叉验证折数
        grid_search = GridSearchCV(clf, {"C": Cs}, scoring=score, cv=n_splits)
        # 执行网格搜索，获取交叉验证结果
        cv_results = grid_search.fit(X, y).cv_results_

        # 检查交叉验证结果中是否包含预期的键
        result_keys = list(cv_results.keys())
        expected_keys = ("mean_test_score", "rank_test_score") + tuple(
            "split%d_test_score" % cv_i for cv_i in range(n_splits)
        )
        assert all(np.isin(expected_keys, result_keys))

        # 初始化分层 K 折交叉验证对象
        cv = StratifiedKFold(n_splits=n_splits)
        # 获取网格搜索的折数
        n_splits = grid_search.n_splits_
        # 遍历每个候选参数和对应的 C 值
        for candidate_i, C in enumerate(Cs):
            # 设置当前分类器的 C 参数
            clf.set_params(C=C)
            # 提取当前候选参数的交叉验证分数数组
            cv_scores = np.array(
                [
                    grid_search.cv_results_["split%d_test_score" % s][candidate_i]
                    for s in range(n_splits)
                ]
            )
            # 遍历每一折的训练集和测试集
            for i, (train, test) in enumerate(cv.split(X, y)):
                # 训练分类器
                clf.fit(X[train], y[train])
                # 根据评分指标计算正确的分数
                if score == "f1":
                    correct_score = f1_score(y[test], clf.predict(X[test]))
                elif score == "roc_auc":
                    dec = clf.decision_function(X[test])
                    correct_score = roc_auc_score(y[test], dec)
                # 断言正确的分数与交叉验证得分一致
                assert_almost_equal(correct_score, cv_scores[i])


# 测试模型的序列化和反序列化
def test_pickle():
    # 创建模拟分类器对象
    clf = MockClassifier()
    # 创建网格搜索对象，传入模拟分类器、参数字典、重拟、交叉验证折数
    grid_search = GridSearchCV(clf, {"foo_param": [1, 2, 3]}, refit=True, cv=3)
    # 执行网格搜索
    grid_search.fit(X, y)
    # 序列化并反序列化网格搜索对象
    grid_search_pickled = pickle.loads(pickle.dumps(grid_search))
    # 断言序列化前后的预测结果一致
    assert_array_almost_equal(grid_search.predict(X), grid_search_pickled.predict(X))

    # 创建随机搜索对象，传入模拟分类器、参数字典、重拟、迭代次数、交叉验证折数
    random_search = RandomizedSearchCV(
        clf, {"foo_param": [1, 2, 3]}, refit=True, n_iter=3, cv=3
    )
    # 执行随机搜索
    random_search.fit(X, y)
    # 序列化并反序列化随机搜索对象
    random_search_pickled = pickle.loads(pickle.dumps(random_search))
    # 断言序列化前后的预测结果一致
    assert_array_almost_equal(
        random_search.predict(X), random_search_pickled.predict(X)
    )


# 测试多输出估计器的网格搜索
def test_grid_search_with_multioutput_data():
    # 生成多标签分类示例数据集 X, y
    X, y = make_multilabel_classification(return_indicator=True, random_state=0)

    # 设置决策树参数字典
    est_parameters = {"max_depth": [1, 2, 3, 4]}
    # 创建 K 折交叉验证对象
    cv = KFold()

    # 初始化决策树回归器和分类器
    estimators = [
        DecisionTreeRegressor(random_state=0),
        DecisionTreeClassifier(random_state=0),
    ]

    # 测试网格搜索交叉验证
    # 对于每个估算器进行网格搜索
    for est in estimators:
        # 创建一个网格搜索对象，使用指定的估算器和参数字典，以及交叉验证策略
        grid_search = GridSearchCV(est, est_parameters, cv=cv)
        # 在给定数据集上拟合网格搜索对象
        grid_search.fit(X, y)
        # 获取最佳参数的结果
        res_params = grid_search.cv_results_["params"]
        # 遍历每个候选参数组合
        for cand_i in range(len(res_params)):
            # 设置估算器的参数为当前候选参数组合
            est.set_params(**res_params[cand_i])

            # 使用交叉验证策略划分数据集，获取训练集和测试集索引
            for i, (train, test) in enumerate(cv.split(X, y)):
                # 在训练集上拟合估算器
                est.fit(X[train], y[train])
                # 计算在测试集上的预测准确度得分
                correct_score = est.score(X[test], y[test])
                # 断言预测准确度得分与交叉验证结果中对应的得分接近
                assert_almost_equal(
                    correct_score,
                    grid_search.cv_results_["split%d_test_score" % i][cand_i],
                )

    # 使用随机搜索对每个估算器进行测试
    for est in estimators:
        # 创建一个随机搜索对象，使用指定的估算器和参数字典，以及交叉验证策略和迭代次数
        random_search = RandomizedSearchCV(est, est_parameters, cv=cv, n_iter=3)
        # 在给定数据集上拟合随机搜索对象
        random_search.fit(X, y)
        # 获取最佳参数的结果
        res_params = random_search.cv_results_["params"]
        # 遍历每个候选参数组合
        for cand_i in range(len(res_params)):
            # 设置估算器的参数为当前候选参数组合
            est.set_params(**res_params[cand_i])

            # 使用交叉验证策略划分数据集，获取训练集和测试集索引
            for i, (train, test) in enumerate(cv.split(X, y)):
                # 在训练集上拟合估算器
                est.fit(X[train], y[train])
                # 计算在测试集上的预测准确度得分
                correct_score = est.score(X[test], y[test])
                # 断言预测准确度得分与随机搜索结果中对应的得分接近
                assert_almost_equal(
                    correct_score,
                    random_search.cv_results_["split%d_test_score" % i][cand_i],
                )
def test_predict_proba_disabled():
    # 测试在估计器上禁用 predict_proba 的情况
    X = np.arange(20).reshape(5, -1)  # 创建一个20个元素的数组，重新塑形为5行
    y = [0, 0, 1, 1, 1]  # 创建一个包含5个元素的数组
    clf = SVC(probability=False)  # 初始化一个支持向量机分类器，禁用概率估计
    gs = GridSearchCV(clf, {}, cv=2).fit(X, y)  # 创建一个网格搜索交叉验证对象，拟合数据
    assert not hasattr(gs, "predict_proba")  # 断言，确保 gs 对象没有 predict_proba 属性


def test_grid_search_allows_nans():
    # 测试在 GridSearchCV 中使用 SimpleImputer 处理缺失值
    X = np.arange(20, dtype=np.float64).reshape(5, -1)  # 创建一个包含20个浮点数的数组，重新塑形为5行
    X[2, :] = np.nan  # 将第二行所有元素设为 NaN
    y = [0, 0, 1, 1, 1]  # 创建一个包含5个元素的数组
    p = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean", missing_values=np.nan)),  # 使用平均值策略的简单填充器处理 NaN
            ("classifier", MockClassifier()),  # 使用模拟分类器
        ]
    )
    GridSearchCV(p, {"classifier__foo_param": [1, 2, 3]}, cv=2).fit(X, y)  # 创建网格搜索交叉验证对象，拟合数据


class FailingClassifier(BaseEstimator):
    """在 fit() 方法中引发 ValueError 的分类器"""

    FAILING_PARAMETER = 2

    def __init__(self, parameter=None):
        self.parameter = parameter

    def fit(self, X, y=None):
        if self.parameter == FailingClassifier.FAILING_PARAMETER:
            raise ValueError("Failing classifier failed as required")  # 如果参数等于 FAILING_PARAMETER，引发 ValueError

    def predict(self, X):
        return np.zeros(X.shape[0])  # 返回一个长度与 X 行数相同的全零数组

    def score(self, X=None, Y=None):
        return 0.0  # 返回固定的得分值


def test_grid_search_failing_classifier():
    # 使用 on_error != 'raise' 的 GridSearchCV
    # 确保在适当的情况下引发警告并重置得分。

    X, y = make_classification(n_samples=20, n_features=10, random_state=0)  # 生成用于分类的合成数据集

    clf = FailingClassifier()  # 创建一个失败的分类器实例

    # refit=False，因为我们只想检查是否会捕获到单独折叠引起的拟合错误并引发警告。
    # 如果 refit 被执行，那么在 refit 时将引发异常，GridSearchCV 不会捕获并会导致测试错误。
    gs = GridSearchCV(
        clf,
        [{"parameter": [0, 1, 2]}],  # 参数网格
        scoring="accuracy",  # 使用准确率作为评分标准
        refit=False,  # 不进行 refit
        error_score=0.0,  # 错误得分设置为 0.0
    )

    warning_message = re.compile(
        "5 fits failed.+total of 15.+The score on these"
        r" train-test partitions for these parameters will be set to 0\.0.+"
        "5 fits failed with the following error.+ValueError.+Failing classifier failed"
        " as required",
        flags=re.DOTALL,
    )
    with pytest.warns(FitFailedWarning, match=warning_message):  # 断言，确保引发 FitFailedWarning 警告，并匹配特定的警告消息
        gs.fit(X, y)  # 拟合数据集
    n_candidates = len(gs.cv_results_["params"])  # 获取参数组合的数量

    # 确保对于预期失败的拟合，网格得分被设置为零。
    def get_cand_scores(i):
        return np.array(
            [gs.cv_results_["split%d_test_score" % s][i] for s in range(gs.n_splits_)]
        )

    assert all(
        (
            np.all(get_cand_scores(cand_i) == 0.0)
            for cand_i in range(n_candidates)
            if gs.cv_results_["param_parameter"][cand_i]
            == FailingClassifier.FAILING_PARAMETER
        )
    )
    # 创建一个GridSearchCV对象，用于参数搜索和交叉验证
    gs = GridSearchCV(
        clf,  # 使用的分类器对象
        [{"parameter": [0, 1, 2]}],  # 参数字典列表，包含参数值[0, 1, 2]
        scoring="accuracy",  # 使用"accuracy"作为评分标准
        refit=False,  # 不重新拟合最佳模型
        error_score=float("nan"),  # 错误分数设为NaN
    )
    # 创建一个正则表达式模式，用于匹配特定的警告消息
    warning_message = re.compile(
        "5 fits failed.+total of 15.+The score on these"
        r" train-test partitions for these parameters will be set to nan.+"
        "5 fits failed with the following error.+ValueError.+Failing classifier failed"
        " as required",
        flags=re.DOTALL,
    )
    # 使用pytest的warns方法，检查是否生成了指定的警告消息
    with pytest.warns(FitFailedWarning, match=warning_message):
        gs.fit(X, y)  # 对数据进行参数搜索和拟合
    n_candidates = len(gs.cv_results_["params"])  # 获取候选参数的数量
    # 断言所有失败的候选参数的分数均为NaN
    assert all(
        np.all(np.isnan(get_cand_scores(cand_i)))
        for cand_i in range(n_candidates)
        if gs.cv_results_["param_parameter"][cand_i]
        == FailingClassifier.FAILING_PARAMETER
    )

    ranks = gs.cv_results_["rank_test_score"]

    # 断言排名最高的两个估计器排名不超过2（即排名较低）
    assert ranks[0] <= 2 and ranks[1] <= 2
    # 断言失败的估计器具有最高的排名（排名为3）
    assert ranks[clf.FAILING_PARAMETER] == 3
    # 断言最佳索引不是失败参数所对应的索引
    assert gs.best_index_ != clf.FAILING_PARAMETER
# 测试用例：当所有的分类器拟合都失败时的网格搜索测试
def test_grid_search_classifier_all_fits_fail():
    # 生成一个具有20个样本和10个特征的分类数据集
    X, y = make_classification(n_samples=20, n_features=10, random_state=0)

    # 创建一个失败的分类器实例
    clf = FailingClassifier()

    # 创建一个网格搜索对象，用于测试所有拟合失败的情况
    gs = GridSearchCV(
        clf,
        [{"parameter": [FailingClassifier.FAILING_PARAMETER] * 3}],  # 参数网格，指定失败参数的三个副本
        error_score=0.0,  # 出错时返回的分数
    )

    # 设置一个正则表达式，用于匹配预期的警告消息
    warning_message = re.compile(
        (
            "All the 15 fits failed.+15 fits failed with the following"
            " error.+ValueError.+Failing classifier failed as required"
        ),
        flags=re.DOTALL,
    )
    # 使用 pytest 检查是否抛出 ValueError 异常，并匹配警告消息
    with pytest.raises(ValueError, match=warning_message):
        gs.fit(X, y)


# 测试用例：网格搜索时处理失败分类器抛出异常的情况
def test_grid_search_failing_classifier_raise():
    # 使用指定种子生成一个具有20个样本和10个特征的分类数据集
    X, y = make_classification(n_samples=20, n_features=10, random_state=0)

    # 创建一个失败的分类器实例
    clf = FailingClassifier()

    # 创建一个网格搜索对象，配置为在出错时抛出异常
    gs = GridSearchCV(
        clf,
        [{"parameter": [0, 1, 2]}],  # 参数网格，包含几个可能的参数
        scoring="accuracy",  # 使用准确率作为评分标准
        refit=False,  # 不重新拟合最佳模型，用于测试网格搜索的行为
        error_score="raise",  # 出错时抛出异常
    )

    # 使用 pytest 检查是否抛出 ValueError 异常
    with pytest.raises(ValueError):
        gs.fit(X, y)


# 测试用例：参数抽样器如果 n_iter 大于总参数空间时会引发警告
def test_parameters_sampler_replacement():
    # 定义一个参数字典列表
    params = [
        {"first": [0, 1], "second": ["a", "b", "c"]},
        {"third": ["two", "values"]},
    ]
    # 创建一个参数抽样器对象，设置 n_iter=9
    sampler = ParameterSampler(params, n_iter=9)
    n_iter = 9
    grid_size = 8
    # 预期的警告消息，根据参数空间大小和 n_iter 的比较
    expected_warning = (
        "The total space of parameters %d is smaller "
        "than n_iter=%d. Running %d iterations. For "
        "exhaustive searches, use GridSearchCV." % (grid_size, n_iter, grid_size)
    )
    # 使用 pytest 检查是否发出 UserWarning 并匹配预期的警告消息
    with pytest.warns(UserWarning, match=expected_warning):
        list(sampler)

    # 如果 n_iter 等于参数空间大小，抽样器应退化为 GridSearchCV
    sampler = ParameterSampler(params, n_iter=8)
    samples = list(sampler)
    assert len(samples) == 8  # 确保生成的样本数为 8
    for values in ParameterGrid(params):
        assert values in samples  # 确保每个参数组合都在生成的样本中
    assert len(ParameterSampler(params, n_iter=1000)) == 8  # 对比大量迭代的情况

    # 测试在大网格中无替换地抽样
    params = {"a": range(10), "b": range(10), "c": range(10)}
    sampler = ParameterSampler(params, n_iter=99, random_state=42)
    samples = list(sampler)
    assert len(samples) == 99  # 确保生成的样本数为 99
    # 将参数组合哈希化，确保所有的哈希值都是唯一的
    hashable_samples = ["a%db%dc%d" % (p["a"], p["b"], p["c"]) for p in samples]
    assert len(set(hashable_samples)) == 99  # 确保所有的哈希值都是唯一的

    # 测试在 Bernoulli 分布中不替换地抽样，以防止进入无限循环
    params_distribution = {"first": bernoulli(0.5), "second": ["a", "b", "c"]}
    sampler = ParameterSampler(params_distribution, n_iter=7)
    samples = list(sampler)
    assert len(samples) == 7  # 确保生成的样本数为 7


# 测试用例：确保当 loss 参数作为 param_grid 中的一个参数时，predict_proba 能够正常工作
def test_stochastic_gradient_loss_param():
    param_grid = {
        "loss": ["log_loss"],  # 指定 loss 参数为 "log_loss"
    }
    X = np.arange(24).reshape(6, -1)  # 创建一个6x4的数组作为特征矩阵
    y = [0, 0, 0, 1, 1, 1]  # 创建相应的目标标签数组
    # 使用 GridSearchCV 对象进行参数网格搜索，使用 SGDClassifier 作为基础估计器，损失函数为'hinge'。
    # param_grid 定义了参数网格，cv=3 表示使用 3 折交叉验证。
    clf = GridSearchCV(
        estimator=SGDClassifier(loss="hinge"), param_grid=param_grid, cv=3
    )

    # 断言：当估计器未拟合时，不应具有 'predict_proba' 属性，因为损失函数为 'hinge'。
    assert not hasattr(clf, "predict_proba")
    
    # 对 clf 进行拟合，使用输入数据 X 和标签 y。
    clf.fit(X, y)
    
    # 调用 clf.predict_proba(X) 会引发 AttributeError，因为损失函数为 'hinge'，不支持概率预测。
    clf.predict_proba(X)
    
    # 同样，调用 clf.predict_log_proba(X) 也会引发 AttributeError，因为损失函数为 'hinge'。
    clf.predict_log_proba(X)

    # 确保在 param_grid 中设置 loss=['hinge'] 时，GridSearchCV 创建的 clf 仍然不具有 'predict_proba' 属性。
    param_grid = {
        "loss": ["hinge"],
    }
    clf = GridSearchCV(
        estimator=SGDClassifier(loss="hinge"), param_grid=param_grid, cv=3
    )
    
    # 断言：在这种情况下，clf 不应具有 'predict_proba' 属性。
    assert not hasattr(clf, "predict_proba")
    
    # 对 clf 进行拟合，使用输入数据 X 和标签 y。
    clf.fit(X, y)
    
    # 再次断言：拟合后，clf 仍然不应具有 'predict_proba' 属性。
    assert not hasattr(clf, "predict_proba")
def test_search_train_scores_set_to_false():
    # 创建一个包含 0 到 5 的二维数组作为特征矩阵 X
    X = np.arange(6).reshape(6, -1)
    # 创建包含类标签的列表 y，标签分别为 0 和 1
    y = [0, 0, 0, 1, 1, 1]
    # 创建一个 LinearSVC 分类器对象 clf
    clf = LinearSVC(random_state=0)

    # 创建一个 GridSearchCV 对象 gs，用于交叉验证和参数优化
    gs = GridSearchCV(clf, param_grid={"C": [0.1, 0.2]}, cv=3)
    # 在特征矩阵 X 和类标签 y 上执行 GridSearchCV 运算
    gs.fit(X, y)


def test_grid_search_cv_splits_consistency():
    # 检查是否接受一次性可迭代对象作为 cv 参数
    n_samples = 100
    n_splits = 5
    # 生成一个分类数据集 X 和对应的类标签 y
    X, y = make_classification(n_samples=n_samples, random_state=0)

    # 创建 GridSearchCV 对象 gs，使用自定义的 OneTimeSplitter 作为交叉验证策略
    gs = GridSearchCV(
        LinearSVC(random_state=0),
        param_grid={"C": [0.1, 0.2, 0.3]},
        cv=OneTimeSplitter(n_splits=n_splits, n_samples=n_samples),
        return_train_score=True,
    )
    gs.fit(X, y)

    # 创建 GridSearchCV 对象 gs2，使用 KFold 作为交叉验证策略
    gs2 = GridSearchCV(
        LinearSVC(random_state=0),
        param_grid={"C": [0.1, 0.2, 0.3]},
        cv=KFold(n_splits=n_splits),
        return_train_score=True,
    )
    gs2.fit(X, y)

    # 创建 GridSearchCV 对象 gs3，使用带有 shuffle 的 KFold 生成器作为交叉验证策略
    assert isinstance(
        KFold(n_splits=n_splits, shuffle=True, random_state=0).split(X, y),
        GeneratorType,
    )
    gs3 = GridSearchCV(
        LinearSVC(random_state=0),
        param_grid={"C": [0.1, 0.2, 0.3]},
        cv=KFold(n_splits=n_splits, shuffle=True, random_state=0).split(X, y),
        return_train_score=True,
    )
    gs3.fit(X, y)

    # 创建 GridSearchCV 对象 gs4，使用带有 shuffle 的 KFold 作为交叉验证策略
    gs4 = GridSearchCV(
        LinearSVC(random_state=0),
        param_grid={"C": [0.1, 0.2, 0.3]},
        cv=KFold(n_splits=n_splits, shuffle=True, random_state=0),
        return_train_score=True,
    )
    gs4.fit(X, y)

    # 定义一个函数 _pop_time_keys，用于从 cv_results 中移除时间相关的键
    def _pop_time_keys(cv_results):
        for key in (
            "mean_fit_time",
            "std_fit_time",
            "mean_score_time",
            "std_score_time",
        ):
            cv_results.pop(key)
        return cv_results

    # 检查生成器是否作为 cv 参数支持，并确保分割结果的一致性
    np.testing.assert_equal(
        _pop_time_keys(gs3.cv_results_), _pop_time_keys(gs4.cv_results_)
    )

    # OneTimeSplitter 是一个不可重入的交叉验证器，每次调用只能生成一次分割
    # 如果 GridSearchCV.fit 中对每个参数设置只调用一次 cv.split，则不会为第二次和后续的 cv.split 调用评估
    # 这是一个检查，以确保 cv.split 没有针对每个参数设置调用多次的测试
    np.testing.assert_equal(
        {k: v for k, v in gs.cv_results_.items() if not k.endswith("_time")},
        {k: v for k, v in gs2.cv_results_.items() if not k.endswith("_time")},
    )

    # 检查参数设置之间的分割一致性
    gs = GridSearchCV(
        LinearSVC(random_state=0),
        param_grid={"C": [0.1, 0.1, 0.2, 0.2]},
        cv=KFold(n_splits=n_splits, shuffle=True),
        return_train_score=True,
    )
    gs.fit(X, y)

    # 由于前两个参数设置 (C=0.1) 和后两个参数设置 (C=0.2) 相同，测试分数和训练分数也必须相同
    # 对于每个评分类型（"train" 和 "test"），循环执行以下操作
    for score_type in ("train", "test"):
        # 创建一个空字典，用于存储每个参数设置的分数列表
        per_param_scores = {}
        # 对参数索引范围内的每个参数执行以下操作
        for param_i in range(4):
            # 从网格搜索结果中获取特定参数设置下的每次交叉验证的分数
            # 使用字符串格式化从 'split0_train_score' 到 'split4_train_score' 或 'split0_test_score' 到 'split4_test_score'
            per_param_scores[param_i] = [
                gs.cv_results_["split%d_%s_score" % (s, score_type)][param_i]
                for s in range(5)
            ]
    
        # 断言每个参数索引对应的分数列表近似相等
        assert_array_almost_equal(per_param_scores[0], per_param_scores[1])
        assert_array_almost_equal(per_param_scores[2], per_param_scores[3])
def test_transform_inverse_transform_round_trip():
    # 创建一个 MockClassifier 实例
    clf = MockClassifier()
    # 创建一个 GridSearchCV 对象，指定参数字典和交叉验证折数，设置详细日志级别为3
    grid_search = GridSearchCV(clf, {"foo_param": [1, 2, 3]}, cv=3, verbose=3)

    # 使用网格搜索拟合数据集
    grid_search.fit(X, y)
    # 对 X 应用 transform 方法，并对结果应用 inverse_transform 方法，得到 X_round_trip
    X_round_trip = grid_search.inverse_transform(grid_search.transform(X))
    # 断言 X 与 X_round_trip 在数组的所有元素上是否相等
    assert_array_equal(X, X_round_trip)


def test_custom_run_search():
    def check_results(results, gscv):
        # 获取 GridSearchCV 的交叉验证结果
        exp_results = gscv.cv_results_
        # 断言结果字典的键是否一致
        assert sorted(results.keys()) == sorted(exp_results)
        for k in results:
            if not k.endswith("_time"):
                # 对 results[k] 转换成 numpy 数组
                results[k] = np.asanyarray(results[k])
                if results[k].dtype.kind == "O":
                    # 对象数组情况下，检查所有元素是否相等
                    assert_array_equal(
                        exp_results[k], results[k], err_msg="Checking " + k
                    )
                else:
                    # 数值数组情况下，使用 allclose 检查近似相等性
                    assert_allclose(exp_results[k], results[k], err_msg="Checking " + k)

    def fit_grid(param_grid):
        # 使用 GridSearchCV 对象拟合数据集并返回结果
        return GridSearchCV(clf, param_grid, return_train_score=True).fit(X, y)

    class CustomSearchCV(BaseSearchCV):
        def __init__(self, estimator, **kwargs):
            super().__init__(estimator, **kwargs)

        def _run_search(self, evaluate):
            # 评估函数 evaluate 执行并返回结果
            results = evaluate([{"max_depth": 1}, {"max_depth": 2}])
            # 检查结果与指定参数的网格搜索结果是否相等
            check_results(results, fit_grid({"max_depth": [1, 2]}))
            results = evaluate([{"min_samples_split": 5}, {"min_samples_split": 10}])
            # 检查结果与指定参数的网格搜索结果是否相等
            check_results(
                results,
                fit_grid([{"max_depth": [1, 2]}, {"min_samples_split": [5, 10]}]),
            )

    # 创建一个 DecisionTreeRegressor 实例作为分类器
    clf = DecisionTreeRegressor(random_state=0)
    # 生成分类数据集 X 和标签 y
    X, y = make_classification(n_samples=100, n_informative=4, random_state=0)
    # 创建 CustomSearchCV 对象并拟合数据集
    mycv = CustomSearchCV(clf, return_train_score=True).fit(X, y)
    # 使用 GridSearchCV 对象拟合数据集并返回结果
    gscv = fit_grid([{"max_depth": [1, 2]}, {"min_samples_split": [5, 10]}])

    # 获取 CustomSearchCV 的交叉验证结果
    results = mycv.cv_results_
    # 检查结果是否与 gscv 对象的结果相等
    check_results(results, gscv)
    # 遍历 gscv 对象的所有属性
    for attr in dir(gscv):
        if (
            attr[0].islower()
            and attr[-1:] == "_"
            and attr
            not in {
                "cv_results_",
                "best_estimator_",
                "refit_time_",
                "classes_",
                "scorer_",
            }
        ):
            # 断言 gscv 和 mycv 对象的相应属性是否相等
            assert getattr(gscv, attr) == getattr(mycv, attr), (
                "Attribute %s not equal" % attr
            )


def test__custom_fit_no_run_search():
    class NoRunSearchSearchCV(BaseSearchCV):
        def __init__(self, estimator, **kwargs):
            super().__init__(estimator, **kwargs)

        def fit(self, X, y=None, groups=None, **fit_params):
            # 直接返回自身对象
            return self

    # 不应该引发任何异常
    NoRunSearchSearchCV(SVC()).fit(X, y)

    class BadSearchCV(BaseSearchCV):
        def __init__(self, estimator, **kwargs):
            super().__init__(estimator, **kwargs)
    # 使用 pytest 来测试是否会引发 NotImplementedError 异常，并验证异常消息是否包含 "_run_search not implemented."
    with pytest.raises(NotImplementedError, match="_run_search not implemented."):
        # 调用 BadSearchCV 类的 fit 方法，传入一个 SVC() 实例作为参数，以及数据集 X 和标签 y
        BadSearchCV(SVC()).fit(X, y)
def test_empty_cv_iterator_error():
    # Use global X, y

    # create cv using KFold with 3 splits on X data
    cv = KFold(n_splits=3).split(X)

    # consume all iterations of cv iterator, which empties it; this triggers an expected ValueError
    [u for u in cv]
    # cv is empty now

    train_size = 100
    # perform randomized search cross-validation with Ridge regression, expecting a ValueError
    ridge = RandomizedSearchCV(Ridge(), {"alpha": [1e-3, 1e-2, 1e-1]}, cv=cv, n_jobs=4)

    # assert that the fit operation raises a ValueError with specific error message
    with pytest.raises(
        ValueError,
        match=(
            "No fits were performed. "
            "Was the CV iterator empty\\? "
            "Were there no candidates\\?"
        ),
    ):
        ridge.fit(X[:train_size], y[:train_size])


def test_random_search_bad_cv():
    # Use global X, y

    class BrokenKFold(KFold):
        # override get_n_splits method to always return 1 split
        def get_n_splits(self, *args, **kw):
            return 1

    # create a custom KFold subclass with broken behavior
    cv = BrokenKFold(n_splits=3)

    train_size = 100
    # perform randomized search cross-validation with Ridge regression, expecting a ValueError
    ridge = RandomizedSearchCV(Ridge(), {"alpha": [1e-3, 1e-2, 1e-1]}, cv=cv, n_jobs=4)

    # assert that the fit operation raises a ValueError with specific error message
    with pytest.raises(
        ValueError,
        match=(
            "cv.split and cv.get_n_splits returned "
            "inconsistent results. Expected \\d+ "
            "splits, got \\d+"
        ),
    ):
        ridge.fit(X[:train_size], y[:train_size])


@pytest.mark.parametrize("return_train_score", [False, True])
@pytest.mark.parametrize(
    "SearchCV, specialized_params",
    [
        (GridSearchCV, {"param_grid": {"max_depth": [2, 3, 5, 8]}}),
        (
            RandomizedSearchCV,
            {"param_distributions": {"max_depth": [2, 3, 5, 8]}, "n_iter": 4},
        ),
    ],
)
def test_searchcv_raise_warning_with_non_finite_score(
    SearchCV, specialized_params, return_train_score
):
    # Non-regression test for specific GitHub issue
    # Ensure a UserWarning is raised when non-finite scores occur in SearchCV

    # Generate synthetic classification data
    X, y = make_classification(n_classes=2, random_state=0)

    class FailingScorer:
        """Custom scorer that intermittently returns NaN."""

        def __init__(self):
            self.n_counts = 0

        def __call__(self, estimator, X, y):
            self.n_counts += 1
            if self.n_counts % 5 == 0:
                return np.nan
            return 1

    # Instantiate SearchCV with DecisionTreeClassifier and custom failing scorer
    grid = SearchCV(
        DecisionTreeClassifier(),
        scoring=FailingScorer(),
        cv=3,
        return_train_score=return_train_score,
        **specialized_params,
    )

    # Assert that a UserWarning is raised during fitting
    with pytest.warns(UserWarning) as warn_msg:
        grid.fit(X, y)

    # Verify warning messages contain expected content related to non-finite scores
    set_with_warning = ["test", "train"] if return_train_score else ["test"]
    assert len(warn_msg) == len(set_with_warning)
    for msg, dataset in zip(warn_msg, set_with_warning):
        assert f"One or more of the {dataset} scores are non-finite" in str(msg.message)

    # Ensure all non-finite scores are equally ranked last
    last_rank = grid.cv_results_["rank_test_score"].max()
    non_finite_mask = np.isnan(grid.cv_results_["mean_test_score"])
    # 使用断言检查非有限分数的排名是否与上一次排名相同
    assert_array_equal(grid.cv_results_["rank_test_score"][non_finite_mask], last_rank)
    # 所有有限分数应该比非有限分数排名更靠前
    assert np.all(grid.cv_results_["rank_test_score"][~non_finite_mask] < last_rank)
def test_callable_multimetric_confusion_matrix():
    # 测试可调用函数与多指标混淆矩阵，将正确的名称和指标插入搜索交叉验证对象中

    def custom_scorer(clf, X, y):
        # 自定义评分函数，接受分类器、特征数据 X 和标签数据 y 作为参数
        y_pred = clf.predict(X)
        # 使用分类器预测数据 X，得到预测结果 y_pred
        cm = confusion_matrix(y, y_pred)
        # 计算真实标签 y 和预测标签 y_pred 的混淆矩阵
        return {"tn": cm[0, 0], "fp": cm[0, 1], "fn": cm[1, 0], "tp": cm[1, 1]}
        # 返回包含混淆矩阵中四个指标（真负例、假正例、假负例、真正例）的字典

    X, y = make_classification(n_samples=40, n_features=4, random_state=42)
    # 生成用于测试的分类数据 X 和标签数据 y
    est = LinearSVC(random_state=42)
    # 初始化线性支持向量机分类器 est
    search = GridSearchCV(est, {"C": [0.1, 1]}, scoring=custom_scorer, refit="fp")
    # 创建网格搜索交叉验证对象 search，使用 est 分类器，参数为 {"C": [0.1, 1]}，评分函数为 custom_scorer，refit 参数设置为 "fp"

    search.fit(X, y)
    # 在数据集 X, y 上进行网格搜索

    score_names = ["tn", "fp", "fn", "tp"]
    # 定义用于验证的指标名称列表

    for name in score_names:
        assert "mean_test_{}".format(name) in search.cv_results_
        # 验证搜索结果中是否包含每个指标名称的平均测试结果

    y_pred = search.predict(X)
    # 使用搜索得到的最佳模型预测数据集 X 的标签
    cm = confusion_matrix(y, y_pred)
    # 计算最终的混淆矩阵
    assert search.score(X, y) == pytest.approx(cm[0, 1])
    # 验证使用搜索对象的 score 方法计算得到的分数是否与混淆矩阵中的假正例对应的值相近


def test_callable_multimetric_same_as_list_of_strings():
    # 测试可调用函数与字符串列表指标相同的效果

    def custom_scorer(est, X, y):
        # 自定义评分函数，接受分类器 est、特征数据 X 和标签数据 y 作为参数
        y_pred = est.predict(X)
        # 使用分类器 est 预测数据 X，得到预测结果 y_pred
        return {
            "recall": recall_score(y, y_pred),
            "accuracy": accuracy_score(y, y_pred),
        }
        # 返回包含召回率和准确率指标的字典

    X, y = make_classification(n_samples=40, n_features=4, random_state=42)
    # 生成用于测试的分类数据 X 和标签数据 y
    est = LinearSVC(random_state=42)
    # 初始化线性支持向量机分类器 est

    search_callable = GridSearchCV(
        est, {"C": [0.1, 1]}, scoring=custom_scorer, refit="recall"
    )
    # 创建网格搜索交叉验证对象 search_callable，使用 est 分类器，参数为 {"C": [0.1, 1]}，评分函数为 custom_scorer，refit 参数设置为 "recall"
    search_str = GridSearchCV(
        est, {"C": [0.1, 1]}, scoring=["recall", "accuracy"], refit="recall"
    )
    # 创建网格搜索交叉验证对象 search_str，使用 est 分类器，参数为 {"C": [0.1, 1]}，评分函数为 ["recall", "accuracy"]，refit 参数设置为 "recall"

    search_callable.fit(X, y)
    # 在数据集 X, y 上进行网格搜索
    search_str.fit(X, y)
    # 在数据集 X, y 上进行网格搜索

    assert search_callable.best_score_ == pytest.approx(search_str.best_score_)
    # 验证 search_callable 的最佳分数是否与 search_str 的最佳分数近似相等
    assert search_callable.best_index_ == search_str.best_index_
    # 验证 search_callable 的最佳索引是否与 search_str 的最佳索引相等
    assert search_callable.score(X, y) == pytest.approx(search_str.score(X, y))
    # 验证 search_callable 在数据集 X, y 上的得分是否与 search_str 的得分近似相等


def test_callable_single_metric_same_as_single_string():
    # 测试可调用函数评分器与使用单个字符串评分效果相同

    def custom_scorer(est, X, y):
        # 自定义评分函数，接受分类器 est、特征数据 X 和标签数据 y 作为参数
        y_pred = est.predict(X)
        # 使用分类器 est 预测数据 X，得到预测结果 y_pred
        return recall_score(y, y_pred)
        # 返回模型在数据集 X, y 上的召回率指标

    X, y = make_classification(n_samples=40, n_features=4, random_state=42)
    # 生成用于测试的分类数据 X 和标签数据 y
    est = LinearSVC(random_state=42)
    # 初始化线性支持向量机分类器 est

    search_callable = GridSearchCV(
        est, {"C": [0.1, 1]}, scoring=custom_scorer, refit=True
    )
    # 创建网格搜索交叉验证对象 search_callable，使用 est 分类器，参数为 {"C": [0.1, 1]}，评分函数为 custom_scorer，refit 参数设置为 True
    search_str = GridSearchCV(est, {"C": [0.1, 1]}, scoring="recall", refit="recall")
    # 创建网格搜索交叉验证对象 search_str，使用 est 分类器，参数为 {"C": [0.1, 1]}，评分函数为 "recall"，refit 参数设置为 "recall"
    search_list_str = GridSearchCV(
        est, {"C": [0.1, 1]}, scoring=["recall"], refit="recall"
    )
    # 创建网格搜索交叉验证对象 search_list_str，使用 est 分类器，参数为 {"C": [0.1, 1]}，评分函数为 ["recall"]，refit 参数设置为 "recall"

    search_callable.fit(X, y)
    # 在数据集 X, y 上进行网格搜索
    search_str.fit(X, y)
    # 在数据集 X, y 上进行网格搜索
    search_list_str.fit(X, y)
    # 在数据集 X, y 上进行网格搜索

    assert search_callable.best_score_ == pytest.approx(search_str.best_score_)
    # 验证 search_callable 的最佳分数是否与 search_str 的最佳分数近似相等
    assert search_callable.best_index_ == search_str.best_index_
    # 验证 search_callable 的最佳索引是否与 search_str 的最佳索引相等
    assert search_callable.score(X, y) == pytest.approx(search_str.score(X, y))
    # 验证 search_callable 在数据集 X, y 上的得分是否与 search_str 的得分近似相等

    assert search_list_str.best_score_ == pytest.approx(search_str.best_score_)
    # 验证 search_list_str 的最佳分数是否与 search_str 的最佳分数近似相等
    assert search_list_str.best_index_ == search_str.best_index_
    # 验证 search_list_str 的最佳索引是否与 search_str 的最佳索引相等
    assert search_list_str.score(X, y) == pytest.approx(search_str.score(X, y))
    # 验证 search_list_str 在数据集 X, y 上的得分是否与 search_str 的得分近似相等
# 测试可调用的多度量错误，当可调用的评分器未返回带有 `refit` 键的字典时会引发异常。
def test_callable_multimetric_error_on_invalid_key():
    # 定义一个错误的评分器，它返回一个不包含 `refit` 键的字典
    def bad_scorer(est, X, y):
        return {"bad_name": 1}

    # 创建一个样本数据集
    X, y = make_classification(n_samples=40, n_features=4, random_state=42)
    # 创建一个 GridSearchCV 对象，使用 LinearSVC 分类器，评分器为 bad_scorer，refit 参数设置为 "good_name"
    clf = GridSearchCV(
        LinearSVC(random_state=42),
        {"C": [0.1, 1]},
        scoring=bad_scorer,
        refit="good_name",
    )

    # 准备错误消息，用于匹配 pytest 抛出的 ValueError 异常信息
    msg = (
        "For multi-metric scoring, the parameter refit must be set to a "
        "scorer key or a callable to refit"
    )
    # 使用 pytest 检查是否会抛出指定消息的 ValueError 异常
    with pytest.raises(ValueError, match=msg):
        clf.fit(X, y)


# 测试可调用的多度量错误，当评估器在使用错误分数时警告
def test_callable_multimetric_error_failing_clf():
    # 定义一个自定义的评分器，返回一个带有 "acc" 键的字典
    def custom_scorer(est, X, y):
        return {"acc": 1}

    # 创建一个样本数据集
    X, y = make_classification(n_samples=20, n_features=10, random_state=0)

    # 创建一个 FailingClassifier 对象
    clf = FailingClassifier()
    # 创建一个 GridSearchCV 对象，使用 FailingClassifier 分类器，评分器为 custom_scorer，refit 参数设置为 False，error_score 设置为 0.1
    gs = GridSearchCV(
        clf,
        [{"parameter": [0, 1, 2]}],
        scoring=custom_scorer,
        refit=False,
        error_score=0.1,
    )

    # 准备警告消息的正则表达式，用于匹配 pytest 抛出的 FitFailedWarning 警告信息
    warning_message = re.compile(
        "5 fits failed.+total of 15.+The score on these"
        r" train-test partitions for these parameters will be set to 0\.1",
        flags=re.DOTALL,
    )
    # 使用 pytest 检查是否会抛出指定警告信息的 FitFailedWarning 警告
    with pytest.warns(FitFailedWarning, match=warning_message):
        gs.fit(X, y)

    # 断言检查 GridSearchCV 对象的交叉验证结果中的平均测试准确率是否符合预期
    assert_allclose(gs.cv_results_["mean_test_acc"], [1, 1, 0.1])


# 测试可调用的多度量错误，当所有评估器都无法拟合时，警告并引发异常
def test_callable_multimetric_clf_all_fits_fail():
    # 定义一个自定义的评分器，返回一个带有 "acc" 键的字典
    def custom_scorer(est, X, y):
        return {"acc": 1}

    # 创建一个样本数据集
    X, y = make_classification(n_samples=20, n_features=10, random_state=0)

    # 创建一个 FailingClassifier 对象
    clf = FailingClassifier()

    # 创建一个 GridSearchCV 对象，使用 FailingClassifier 分类器，评分器为 custom_scorer，refit 参数设置为 False，error_score 设置为 0.1
    gs = GridSearchCV(
        clf,
        [{"parameter": [FailingClassifier.FAILING_PARAMETER] * 3}],
        scoring=custom_scorer,
        refit=False,
        error_score=0.1,
    )

    # 准备错误消息的正则表达式，用于匹配 pytest 抛出的 ValueError 异常信息
    individual_fit_error_message = "ValueError: Failing classifier failed as required"
    error_message = re.compile(
        (
            "All the 15 fits failed.+your model is misconfigured.+"
            f"{individual_fit_error_message}"
        ),
        flags=re.DOTALL,
    )

    # 使用 pytest 检查是否会抛出指定错误信息的 ValueError 异常
    with pytest.raises(ValueError, match=error_message):
        gs.fit(X, y)


# 测试 n_features_in_ 属性是否正确委托给最佳评估器
def test_n_features_in():
    # 定义样本数据集的特征数
    n_features = 4
    # 创建一个样本数据集
    X, y = make_classification(n_features=n_features)
    # 创建一个 HistGradientBoostingClassifier 分类器
    gbdt = HistGradientBoostingClassifier()
    # 定义参数网格
    param_grid = {"max_iter": [3, 4]}
    # 创建 GridSearchCV 和 RandomizedSearchCV 对象
    gs = GridSearchCV(gbdt, param_grid)
    rs = RandomizedSearchCV(gbdt, param_grid, n_iter=1)
    
    # 断言检查 GridSearchCV 和 RandomizedSearchCV 对象是否没有 n_features_in_ 属性
    assert not hasattr(gs, "n_features_in_")
    assert not hasattr(rs, "n_features_in_")
    
    # 分别对 GridSearchCV 和 RandomizedSearchCV 对象进行拟合
    gs.fit(X, y)
    rs.fit(X, y)
    
    # 断言检查 GridSearchCV 和 RandomizedSearchCV 对象的 n_features_in_ 属性是否等于预期的特征数
    assert gs.n_features_in_ == n_features
    assert rs.n_features_in_ == n_features


# 使用 pytest 的参数化装饰器，测试 SearchCV 对象是否正确委托 pairwise 属性给基础评估器
@pytest.mark.parametrize("pairwise", [True, False])
def test_search_cv_pairwise_property_delegated_to_base_estimator(pairwise):
    """
    Test implementation of BaseSearchCV has the pairwise tag
    which matches the pairwise tag of its estimator.
    This test make sure pairwise tag is delegated to the base estimator.

    Non-regression test for issue #13920.
    """

    # 定义一个测试用的自定义估算器类 TestEstimator，继承自 BaseEstimator
    class TestEstimator(BaseEstimator):
        # 覆盖 _more_tags 方法，返回一个字典，包含 "pairwise" 键，其值为 pairwise 变量的值
        def _more_tags(self):
            return {"pairwise": pairwise}

    # 创建 TestEstimator 类的实例 est
    est = TestEstimator()

    # 提示信息，用于断言检查 BaseSearchCV 的 pairwise 标签必须与估算器匹配
    attr_message = "BaseSearchCV pairwise tag must match estimator"

    # 创建一个 GridSearchCV 的实例 cv，传入 est 和参数字典 {"n_neighbors": [10]}
    cv = GridSearchCV(est, {"n_neighbors": [10]})

    # 断言检查 pairwise 变量与 cv 实例的 _get_tags() 方法返回的 "pairwise" 标签是否相等，
    # 如果不相等，抛出 AssertionError，提示信息为 attr_message
    assert pairwise == cv._get_tags()["pairwise"], attr_message
def test_search_cv__pairwise_property_delegated_to_base_estimator():
    """
    Test implementation of BaseSearchCV has the pairwise property
    which matches the pairwise tag of its estimator.
    This test make sure pairwise tag is delegated to the base estimator.

    Non-regression test for issue #13920.
    """

    # 定义一个自定义的估计器类，用于测试pairwise属性的委托行为
    class EstimatorPairwise(BaseEstimator):
        def __init__(self, pairwise=True):
            self.pairwise = pairwise

        # 返回更多的标签信息，包括pairwise属性
        def _more_tags(self):
            return {"pairwise": self.pairwise}

    # 创建一个EstimatorPairwise类的实例
    est = EstimatorPairwise()
    # 设置错误消息
    attr_message = "BaseSearchCV _pairwise property must match estimator"

    # 遍历测试pairwise属性的不同设置
    for _pairwise_setting in [True, False]:
        # 设置估计器的pairwise属性
        est.set_params(pairwise=_pairwise_setting)
        # 创建GridSearchCV对象，以验证_pairwise属性是否与估计器匹配
        cv = GridSearchCV(est, {"n_neighbors": [10]})
        # 断言GridSearchCV对象的_pairwise属性与估计器的设置是否一致
        assert _pairwise_setting == cv._get_tags()["pairwise"], attr_message


def test_search_cv_pairwise_property_equivalence_of_precomputed():
    """
    Test implementation of BaseSearchCV has the pairwise tag
    which matches the pairwise tag of its estimator.
    This test ensures the equivalence of 'precomputed'.

    Non-regression test for issue #13920.
    """
    # 创建一些样本数据和参数
    n_samples = 50
    n_splits = 2
    X, y = make_classification(n_samples=n_samples, random_state=0)
    grid_params = {"n_neighbors": [10]}

    # 默认使用欧几里得距离作为度量方式（minkowski p = 2）
    clf = KNeighborsClassifier()
    # 创建GridSearchCV对象，使用默认的度量方式
    cv = GridSearchCV(clf, grid_params, cv=n_splits)
    # 对数据进行拟合
    cv.fit(X, y)
    # 获取原始预测结果
    preds_original = cv.predict(X)

    # 预先计算欧几里得距离矩阵，以验证pairwise是否有效工作
    X_precomputed = euclidean_distances(X)
    # 创建KNeighborsClassifier对象，使用预先计算的距离矩阵作为度量方式
    clf = KNeighborsClassifier(metric="precomputed")
    # 创建GridSearchCV对象，使用预先计算的距离矩阵
    cv = GridSearchCV(clf, grid_params, cv=n_splits)
    # 对数据进行拟合
    cv.fit(X_precomputed, y)
    # 获取预测结果
    preds_precomputed = cv.predict(X_precomputed)

    # 断言两种方式的预测结果是否一致
    attr_message = "GridSearchCV not identical with precomputed metric"
    assert (preds_original == preds_precomputed).all(), attr_message


@pytest.mark.parametrize(
    "SearchCV, param_search",
    [(GridSearchCV, {"a": [0.1, 0.01]}), (RandomizedSearchCV, {"a": uniform(1, 3)})],
)
def test_scalar_fit_param(SearchCV, param_search):
    """
    unofficially sanctioned tolerance for scalar values in fit_params
    non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/15805
    """
    # 创建一个测试用的估计器类，继承自ClassifierMixin和BaseEstimator
    class TestEstimator(ClassifierMixin, BaseEstimator):
        def __init__(self, a=None):
            self.a = a

        # 实现拟合方法，接受参数X，y和r
        def fit(self, X, y, r=None):
            self.r_ = r

        # 实现预测方法，返回全零数组
        def predict(self, X):
            return np.zeros(shape=(len(X)))

    # 创建一个SearchCV对象，使用TestEstimator和param_search参数
    model = SearchCV(TestEstimator(), param_search)
    # 生成一些样本数据
    X, y = make_classification(random_state=42)
    # 对模型进行拟合，传入额外的r参数
    model.fit(X, y, r=42)
    # 断言最佳估计器的r_属性是否等于42
    assert model.best_estimator_.r_ == 42


@pytest.mark.parametrize(
    "SearchCV, param_search",
    [
        (GridSearchCV, {"alpha": [0.1, 0.01]}),
        (RandomizedSearchCV, {"alpha": uniform(0.01, 0.1)}),
    ],
)
def test_scalar_fit_param_compat(SearchCV, param_search):
    # This test is currently incomplete and lacks a complete implementation.
    # Additional code is required to fully implement the test.
    # 此测试目前尚不完整，并缺乏完整的实现。
    # 需要额外的代码来完全实现该测试。
    # 检查在 `fit_params` 中是否支持标量值，例如 LightGBM
    # 虽然它们不完全遵守 scikit-learn API 的契约，但我们不希望在没有明确弃用周期和 API 建议的情况下中断对其的支持。
    # 这里进行非回归测试，用于：https://github.com/scikit-learn/scikit-learn/issues/15805

    # 划分训练集和验证集，返回的是 X_train, X_valid, y_train, y_valid
    X_train, X_valid, y_train, y_valid = train_test_split(
        *make_classification(random_state=42), random_state=42
    )

    # 定义一个继承自 SGDClassifier 的类 _FitParamClassifier
    class _FitParamClassifier(SGDClassifier):
        # 重写 fit 方法
        def fit(
            self,
            X,
            y,
            sample_weight=None,
            tuple_of_arrays=None,
            scalar_param=None,
            callable_param=None,
        ):
            # 调用父类的 fit 方法
            super().fit(X, y, sample_weight=sample_weight)
            # 断言 scalar_param 必须大于 0
            assert scalar_param > 0
            # 断言 callable_param 必须是可调用对象
            assert callable(callable_param)

            # 断言 tuple_of_arrays 应该保持为元组形式
            assert isinstance(tuple_of_arrays, tuple)
            # 断言 tuple_of_arrays 的第一个元素是二维数组
            assert tuple_of_arrays[0].ndim == 2
            # 断言 tuple_of_arrays 的第二个元素是一维数组
            assert tuple_of_arrays[1].ndim == 1

            # 返回当前实例对象
            return self

    # 定义一个空的函数 _fit_param_callable
    def _fit_param_callable():
        pass

    # 创建一个 SearchCV 对象 model，使用 _FitParamClassifier 类作为估计器，传入 param_search 参数
    model = SearchCV(_FitParamClassifier(), param_search)

    # 注意：`fit_params` 应该依赖于数据（例如 `sample_weight`），但以下参数并非如此。
    # 但这种滥用在流行的第三方库中很常见，我们应该容忍这种行为，暂时不要在没有遵循适当的弃用周期的情况下中断对其的支持。
    # 传入 fit_params 字典作为关键字参数，用于 model.fit 方法
    fit_params = {
        "tuple_of_arrays": (X_valid, y_valid),  # 将验证集作为元组传入
        "callable_param": _fit_param_callable,  # 将可调用函数传入
        "scalar_param": 42,  # 将标量参数传入
    }
    model.fit(X_train, y_train, **fit_params)
# FIXME: 替换此测试为完整的 `check_estimator` 一旦我们只有 API 检查时。
# 在测试中忽略特定警告信息。
@pytest.mark.filterwarnings("ignore:The total space of parameters 4 is")

# 参数化测试，使用 GridSearchCV 和 RandomizedSearchCV 作为 SearchCV 参数，
# MinimalRegressor 和 MinimalClassifier 作为 Predictor 参数。
@pytest.mark.parametrize("SearchCV", [GridSearchCV, RandomizedSearchCV])
@pytest.mark.parametrize("Predictor", [MinimalRegressor, MinimalClassifier])
def test_search_cv_using_minimal_compatible_estimator(SearchCV, Predictor):
    # 检查第三方库可以在不继承 BaseEstimator 的情况下运行测试。
    rng = np.random.RandomState(0)
    X, y = rng.randn(25, 2), np.array([0] * 5 + [1] * 20)

    # 创建 Pipeline 模型，包括 MinimalTransformer 和给定的 Predictor。
    model = Pipeline(
        [("transformer", MinimalTransformer()), ("predictor", Predictor())]
    )

    # 设置参数字典，用于搜索最佳参数。
    params = {
        "transformer__param": [1, 10],
        "predictor__parama": [1, 10],
    }
    
    # 使用 SearchCV 对象进行模型拟合，设置错误分数策略为 "raise"。
    search = SearchCV(model, params, error_score="raise")
    search.fit(X, y)

    # 断言最佳参数的键与预期参数的键相同。
    assert search.best_params_.keys() == params.keys()

    # 预测并检查分类器或回归器的表现。
    y_pred = search.predict(X)
    if is_classifier(search):
        assert_array_equal(y_pred, 1)
        assert search.score(X, y) == pytest.approx(accuracy_score(y, y_pred))
    else:
        assert_allclose(y_pred, y.mean())
        assert search.score(X, y) == pytest.approx(r2_score(y, y_pred))


# 参数化测试，检查 SearchCV 对象在 verbose > 2 时展示单一度量的分数。
# 针对 issue #19658 的非回归测试。
@pytest.mark.parametrize("return_train_score", [True, False])
def test_search_cv_verbose_3(capsys, return_train_score):
    X, y = make_classification(n_samples=100, n_classes=2, flip_y=0.2, random_state=0)
    clf = LinearSVC(random_state=0)
    grid = {"C": [0.1]}

    # 创建 GridSearchCV 对象，设置 verbose=3 和其他参数。
    GridSearchCV(
        clf,
        grid,
        scoring="accuracy",
        verbose=3,
        cv=3,
        return_train_score=return_train_score,
    ).fit(X, y)
    captured = capsys.readouterr().out

    # 根据 return_train_score 的不同，匹配输出中的分数信息。
    if return_train_score:
        match = re.findall(r"score=\(train=[\d\.]+, test=[\d.]+\)", captured)
    else:
        match = re.findall(r"score=[\d\.]+", captured)
    assert len(match) == 3


# 参数化测试，验证 SearchCV 对象不会改变参数网格中给定的对象。
@pytest.mark.parametrize(
    "SearchCV, param_search",
    [
        (GridSearchCV, "param_grid"),
        (RandomizedSearchCV, "param_distributions"),
        (HalvingGridSearchCV, "param_grid"),
    ],
)
def test_search_estimator_param(SearchCV, param_search):
    # 创建分类数据集。
    X, y = make_classification(random_state=42)

    # 设置参数字典，包含 LinearSVC 分类器。
    params = {"clf": [LinearSVC()], "clf__C": [0.01]}
    orig_C = params["clf"][0].C

    # 创建 Pipeline，包括 MinimalTransformer 和 None 的分类器。
    pipe = Pipeline([("trs", MinimalTransformer()), ("clf", None)])

    # 设置参数网格搜索字典。
    param_grid_search = {param_search: params}

    # 使用 SearchCV 对象进行模型拟合，设置 refit=True 和其他参数。
    gs = SearchCV(pipe, refit=True, cv=2, scoring="accuracy", **param_grid_search).fit(
        X, y
    )

    # 断言原始参数中的对象未被改变。
    assert params["clf"][0].C == orig_C
    # 断言 GridSearchCV 正确设置了步骤的参数。
    assert gs.best_estimator_.named_steps["clf"].C == 0.01
    parameter_grid = {
        "vect__ngram_range": ((1, 1), (1, 2)),  # 定义参数网格：选择使用单个词或两个词组成的文本片段
        "vect__norm": ("l1", "l2"),  # 定义参数网格：选择使用L1或L2范数进行特征向量归一化
    }
    pipeline = Pipeline(
        [
            ("vect", TfidfVectorizer()),  # 文本特征提取器：使用TF-IDF向量化器
            ("clf", ComplementNB()),  # 分类器：使用Complement Naive Bayes分类器
        ]
    )
    random_search = RandomizedSearchCV(
        estimator=pipeline,  # 使用pipeline作为估计器
        param_distributions=parameter_grid,  # 使用定义好的参数网格
        n_iter=3,  # 进行3次迭代的随机搜索
        random_state=0,  # 设定随机数种子以复现结果
        n_jobs=2,  # 指定并行工作的CPU核心数
        verbose=1,  # 打印详细信息，输出进度和结果
        cv=3,  # 使用3折交叉验证
    )
    data_train = ["one", "two", "three", "four", "five"]  # 训练数据集
    data_target = [0, 0, 1, 0, 1]  # 目标分类标签
    random_search.fit(data_train, data_target)  # 在训练数据上拟合随机搜索对象
    result = random_search.cv_results_["param_vect__ngram_range"]  # 获取最佳参数的ngram_range结果
    expected_data = np.empty(3, dtype=object)  # 创建一个空的NumPy数组，用于存储预期的数据
    expected_data[:] = [(1, 2), (1, 2), (1, 1)]  # 填充预期数据数组
    np.testing.assert_array_equal(result.data, expected_data)  # 断言：验证结果与预期数据的一致性
# 定义一个测试函数，用于测试不同的 HTML 表示形式对 GridSearchCV 的影响
def test_search_html_repr():
    # 使用 make_classification 生成样本数据 X 和 y
    X, y = make_classification(random_state=42)

    # 创建一个数据处理流水线，包括数据标准化和分类器
    pipeline = Pipeline([("scale", StandardScaler()), ("clf", DummyClassifier())])
    
    # 定义分类器参数网格
    param_grid = {"clf": [DummyClassifier(), LogisticRegression()]}

    # 创建 GridSearchCV 对象，不进行拟合，用于显示原始流水线
    search_cv = GridSearchCV(pipeline, param_grid=param_grid, refit=False)
    # 使用上下文配置，设置显示为图表
    with config_context(display="diagram"):
        # 获取 GridSearchCV 的 HTML 表示形式
        repr_html = search_cv._repr_html_()
        # 断言 DummyClassifier 的 HTML 表示形式在结果中
        assert "<pre>DummyClassifier()</pre>" in repr_html

    # 拟合 GridSearchCV 对象，refit=False 时显示原始流水线
    search_cv.fit(X, y)
    with config_context(display="diagram"):
        repr_html = search_cv._repr_html_()
        assert "<pre>DummyClassifier()</pre>" in repr_html

    # 拟合 GridSearchCV 对象，refit=True 时显示最佳估计器
    search_cv = GridSearchCV(pipeline, param_grid=param_grid, refit=True)
    search_cv.fit(X, y)
    with config_context(display="diagram"):
        repr_html = search_cv._repr_html_()
        assert "<pre>DummyClassifier()</pre>" not in repr_html
        assert "<pre>LogisticRegression()</pre>" in repr_html


# TODO(1.7): remove this test
# 使用 GridSearchCV 或 RandomizedSearchCV 进行参数化测试
@pytest.mark.parametrize("SearchCV", [GridSearchCV, RandomizedSearchCV])
def test_inverse_transform_Xt_deprecation(SearchCV):
    # 创建 MockClassifier 对象
    clf = MockClassifier()
    # 创建 SearchCV 对象，用于模拟参数搜索
    search = SearchCV(clf, {"foo_param": [1, 2, 3]}, cv=3, verbose=3)

    # 使用搜索对象拟合并转换数据 X
    X2 = search.fit(X, y).transform(X)

    # 断言调用 inverse_transform() 没有提供必需的位置参数会引发 TypeError
    with pytest.raises(TypeError, match="Missing required positional argument"):
        search.inverse_transform()

    # 断言使用 X 和 Xt 同时调用 inverse_transform() 会引发 TypeError
    with pytest.raises(TypeError, match="Cannot use both X and Xt. Use X only"):
        search.inverse_transform(X=X2, Xt=X2)

    # 使用 warnings 捕获器，设置警告抛出为错误
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("error")
        # 调用 inverse_transform()，捕获 FutureWarning 警告
        search.inverse_transform(X2)

    # 断言捕获到 FutureWarning 警告，提示 Xt 已在 1.5 版本中更名为 X
    with pytest.warns(FutureWarning, match="Xt was renamed X in version 1.5"):
        search.inverse_transform(Xt=X2)


# Metadata Routing Tests
# ======================

# 使用 slep006 功能启用装饰器
@pytest.mark.usefixtures("enable_slep006")
@pytest.mark.parametrize(
    "SearchCV, param_search",
    [
        # 测试 GridSearchCV 和 RandomizedSearchCV 对于多个度量时的元数据传递
        (GridSearchCV, "param_grid"),
        (RandomizedSearchCV, "param_distributions"),
    ],
)
def test_multi_metric_search_forwards_metadata(SearchCV, param_search):
    """Test that *SearchCV forwards metadata correctly when passed multiple metrics."""
    # 生成样本数据 X 和 y
    X, y = make_classification(random_state=42)
    # 获取样本数量
    n_samples = _num_samples(X)
    # 创建随机数生成器对象
    rng = np.random.RandomState(0)
    # 创建得分权重数组，用于模拟元数据
    score_weights = rng.rand(n_samples)
    # 创建得分元数据数组
    score_metadata = rng.rand(n_samples)

    # 创建 LinearSVC 分类器对象
    est = LinearSVC()
    # 创建参数网格搜索字典
    param_grid_search = {param_search: {"C": [1]}}

    # 创建得分注册表对象
    scorer_registry = _Registry()
    # 创建消耗得分器对象，设置得分请求的样本权重和元数据
    scorer = ConsumingScorer(registry=scorer_registry).set_score_request(
        sample_weight="score_weights", metadata="score_metadata"
    )
    # 创建得分字典，包括自定义得分器和 accuracy
    scoring = dict(my_scorer=scorer, accuracy="accuracy")
    # 使用给定的估算器 `est` 进行交叉验证搜索，以优化 "accuracy" 评分，参数通过 `param_grid_search` 指定
    SearchCV(est, refit="accuracy", cv=2, scoring=scoring, **param_grid_search).fit(
        X, y, score_weights=score_weights, score_metadata=score_metadata
    )
    # 确保评分器注册表 `scorer_registry` 非空
    assert len(scorer_registry)
    # 遍历评分器注册表 `scorer_registry`
    for _scorer in scorer_registry:
        # 检查每个评分器的记录元数据
        check_recorded_metadata(
            obj=_scorer,
            method="score",
            parent="_score",
            split_params=("sample_weight", "metadata"),
            sample_weight=score_weights,
            metadata=score_metadata,
        )
@pytest.mark.parametrize(
    "SearchCV, param_search",
    [  # 使用 pytest.mark.parametrize 来定义参数化测试，包含不同的搜索器和参数搜索方式
        (GridSearchCV, "param_grid"),  # 使用 GridSearchCV 进行参数网格搜索，参数在 param_grid 中
        (RandomizedSearchCV, "param_distributions"),  # 使用 RandomizedSearchCV 进行随机搜索，参数在 param_distributions 中
        (HalvingGridSearchCV, "param_grid"),  # 使用 HalvingGridSearchCV 进行半格搜索，参数在 param_grid 中
    ],
)
def test_score_rejects_params_with_no_routing_enabled(SearchCV, param_search):
    """*SearchCV should reject **params when metadata routing is not enabled
    since this is added only when routing is enabled."""
    X, y = make_classification(random_state=42)  # 创建分类数据集 X 和 y
    est = LinearSVC()  # 创建线性支持向量分类器 est
    param_grid_search = {param_search: {"C": [1]}}  # 根据 param_search 创建参数网格字典

    gs = SearchCV(est, cv=2, **param_grid_search).fit(X, y)  # 使用 SearchCV 进行模型拟合

    with pytest.raises(ValueError, match="is only supported if"):  # 断言捕获到 ValueError 异常，并匹配特定错误信息
        gs.score(X, y, metadata=1)


# End of Metadata Routing Tests
# =============================


def test_cv_results_dtype_issue_29074():
    """Non-regression test for https://github.com/scikit-learn/scikit-learn/issues/29074"""

    class MetaEstimator(BaseEstimator, ClassifierMixin):
        def __init__(
            self,
            base_clf,
            parameter1=None,
            parameter2=None,
            parameter3=None,
            parameter4=None,
        ):
            self.base_clf = base_clf  # 初始化 MetaEstimator 类，接受基础分类器 base_clf 和多个可选参数
            self.parameter1 = parameter1
            self.parameter2 = parameter2
            self.parameter3 = parameter3
            self.parameter4 = parameter4

        def fit(self, X, y=None):
            self.base_clf.fit(X, y)  # 调用基础分类器的拟合方法
            return self

        def score(self, X, y):
            return self.base_clf.score(X, y)  # 返回基础分类器的评分结果

    # Values of param_grid are such that np.result_type gives slightly
    # different errors, in particular ValueError and TypeError
    param_grid = {
        "parameter1": [None, {"option": "A"}, {"option": "B"}],  # 定义参数网格，包含多种不同的参数组合
        "parameter2": [None, [1, 2]],  # 定义第二个参数的可能取值，包括 None 和列表 [1, 2]
        "parameter3": [{"a": 1}],  # 定义第三个参数的可能取值，包括字典 {"a": 1}
        "parameter4": ["str1", "str2"],  # 定义第四个参数的可能取值，包括字符串 "str1" 和 "str2"
    }
    grid_search = GridSearchCV(
        estimator=MetaEstimator(LogisticRegression()),  # 创建一个使用 MetaEstimator 的 GridSearchCV 对象
        param_grid=param_grid,  # 将定义的参数网格传递给 GridSearchCV 对象
        cv=3,  # 设置交叉验证的折数为 3
    )

    X, y = make_blobs(random_state=0)  # 创建数据集 X 和 y
    grid_search.fit(X, y)  # 使用 GridSearchCV 对象拟合数据集

    for param in param_grid:
        assert grid_search.cv_results_[f"param_{param}"].dtype == object


def test_search_with_estimators_issue_29157():
    """Check cv_results_ for estimators with a `dtype` parameter, e.g. OneHotEncoder."""
    pd = pytest.importorskip("pandas")  # 导入 pandas 库，如果不存在则跳过测试
    df = pd.DataFrame(  # 创建一个 pandas DataFrame
        {
            "numeric_1": [1, 2, 3, 4, 5],  # 包含数值列 numeric_1
            "object_1": ["a", "a", "a", "a", "a"],  # 包含分类列 object_1
            "target": [1.0, 4.1, 2.0, 3.0, 1.0],  # 包含目标列 target
        }
    )
    X = df.drop("target", axis=1)  # 从 DataFrame 中获取特征 X
    y = df["target"]  # 获取目标变量 y

    enc = ColumnTransformer(
        [("enc", OneHotEncoder(sparse_output=False), ["object_1"])],  # 创建 ColumnTransformer 对象，包含 OneHotEncoder 转换器
        remainder="passthrough",  # 设置未指定列保持不变
    )
    pipe = Pipeline(
        [
            ("enc", enc),  # 将 ColumnTransformer 对象作为管道的一个步骤
            ("regressor", LinearRegression()),  # 添加线性回归器作为管道的另一个步骤
        ]
    )
    grid_params = {
        "enc__enc": [
            OneHotEncoder(sparse_output=False),  # 定义 OneHotEncoder 转换器的参数取值
            OrdinalEncoder(),  # 定义 OrdinalEncoder 转换器的参数取值
        ]
    }
    # 创建一个网格搜索对象，用于对管道（pipeline）进行参数优化
    grid_search = GridSearchCV(pipe, grid_params, cv=2)
    # 使用网格搜索对象拟合数据，寻找最佳参数组合
    grid_search.fit(X, y)
    # 断言确保网格搜索结果中参数"enc__enc"的数据类型为对象类型（dtype == object）
    assert grid_search.cv_results_["param_enc__enc"].dtype == object
# 使用 pytest 的参数化装饰器，为测试函数提供多组参数化输入
@pytest.mark.parametrize(
    "array_namespace, device, dtype", yield_namespace_device_dtype_combinations()
)
# 参数化另一个参数，选择不同的搜索算法类（GridSearchCV 或 RandomizedSearchCV）
@pytest.mark.parametrize("SearchCV", [GridSearchCV, RandomizedSearchCV])
# 定义测试函数，测试数组 API 的搜索分类器功能
def test_array_api_search_cv_classifier(SearchCV, array_namespace, device, dtype):
    # 获取特定数组 API（如 numpy 或 cupy）的接口
    xp = _array_api_for_tests(array_namespace, device)

    # 创建一个 10x10 的二维数组 X
    X = np.arange(100).reshape((10, 10))
    # 将 X 转换为指定的数据类型
    X_np = X.astype(dtype)
    # 将 X_np 转换为 xp 对象（例如 numpy 数组或 cupy 数组）
    X_xp = xp.asarray(X_np, device=device)

    # 创建一个分类标签数组 y_np，总共有 10 个元素，前 5 个是 0，后 5 个是 1
    y_np = np.array([0] * 5 + [1] * 5)
    # 将 y_np 转换为 xp 对象
    y_xp = xp.asarray(y_np, device=device)

    # 在配置上下文中启用数组 API 的调度
    with config_context(array_api_dispatch=True):
        # 创建一个搜索器对象 SearchCV，使用线性判别分析作为评估器
        searcher = SearchCV(
            LinearDiscriminantAnalysis(),
            # 设置参数网格，尝试不同的 tol 值
            {"tol": [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]},
            # 设置交叉验证折数为 2
            cv=2,
        )
        # 使用搜索器对象拟合数据 X_xp 和标签 y_xp
        searcher.fit(X_xp, y_xp)
        # 使用搜索器对象评估模型在给定数据 X_xp 和标签 y_xp 上的得分
        searcher.score(X_xp, y_xp)
```