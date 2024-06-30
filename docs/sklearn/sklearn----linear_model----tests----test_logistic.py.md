# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\tests\test_logistic.py`

```
# 导入必要的库和模块
import itertools  # 提供高效的迭代器工具
import os  # 提供与操作系统交互的功能
import warnings  # 控制警告的显示
from functools import partial  # 创建偏函数的工具

import numpy as np  # 数值计算库
import pytest  # 测试框架
from numpy.testing import (  # NumPy的测试工具
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)
from scipy import sparse  # 稀疏矩阵和相关算法

from sklearn import config_context  # 控制全局配置的上下文管理器
from sklearn.base import clone  # 克隆估算器的基类
from sklearn.datasets import load_iris, make_classification  # 加载示例数据集
from sklearn.exceptions import ConvergenceWarning  # 收敛警告异常类
from sklearn.linear_model import SGDClassifier  # 随机梯度下降分类器
from sklearn.linear_model._logistic import (  # 逻辑回归及其交叉验证版本的模块
    LogisticRegression as LogisticRegressionDefault,
)
from sklearn.linear_model._logistic import (
    LogisticRegressionCV as LogisticRegressionCVDefault,
)
from sklearn.linear_model._logistic import (  # 逻辑回归的路径和评分路径
    _log_reg_scoring_path,
    _logistic_regression_path,
)
from sklearn.metrics import get_scorer, log_loss  # 获取评分器和对数损失函数
from sklearn.model_selection import (  # 模型选择工具
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.multiclass import OneVsRestClassifier  # 多类分类器
from sklearn.preprocessing import (  # 数据预处理工具
    LabelEncoder,
    StandardScaler,
    scale,
)
from sklearn.svm import l1_min_c  # 计算最小 L1 正则化系数
from sklearn.utils import (  # 各种工具函数
    compute_class_weight,
    shuffle,
)
from sklearn.utils._testing import ignore_warnings, skip_if_no_parallel  # 测试时的忽略和并行支持
from sklearn.utils.fixes import _IS_32BIT, COO_CONTAINERS, CSR_CONTAINERS  # 兼容性修复工具

# 设定 pytest 的标记以过滤特定警告
pytestmark = pytest.mark.filterwarnings(
    "error::sklearn.exceptions.ConvergenceWarning:sklearn.*"
)

# 设置随机种子以防止收敛警告
LogisticRegression = partial(LogisticRegressionDefault, random_state=0)
LogisticRegressionCV = partial(LogisticRegressionCVDefault, random_state=0)

# 可用的求解器类型
SOLVERS = ("lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga")

# 示例数据集 X 和标签 Y1、Y2
X = [[-1, 0], [0, 1], [1, 1]]
Y1 = [0, 1, 1]
Y2 = [2, 1, 0]

# 加载鸢尾花数据集作为示例
iris = load_iris()


def check_predictions(clf, X, y):
    """检查模型是否能够适应分类数据"""
    n_samples = len(y)  # 样本数量
    classes = np.unique(y)  # 唯一类别
    n_classes = classes.shape[0]  # 类别数量

    # 使用分类器拟合数据并进行预测
    predicted = clf.fit(X, y).predict(X)
    assert_array_equal(clf.classes_, classes)  # 断言分类器的类别与实际类别一致

    # 断言预测结果的形状和实际标签一致
    assert predicted.shape == (n_samples,)
    assert_array_equal(predicted, y)  # 断言预测结果与实际标签一致

    # 获取预测概率
    probabilities = clf.predict_proba(X)
    # 断言预测概率的形状正确，并且每行概率之和为1
    assert probabilities.shape == (n_samples, n_classes)
    assert_array_almost_equal(probabilities.sum(axis=1), np.ones(n_samples))
    assert_array_equal(probabilities.argmax(axis=1), y)  # 断言最大概率的索引与实际标签一致


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_predict_2_classes(csr_container):
    """对于二分类数据集进行简单的预测测试"""
    # 使用 LogisticRegression 检查预测结果
    check_predictions(LogisticRegression(random_state=0), X, Y1)
    check_predictions(LogisticRegression(random_state=0), csr_container(X), Y1)

    # 使用不同的超参数检查预测结果
    check_predictions(LogisticRegression(C=100, random_state=0), X, Y1)
    check_predictions(LogisticRegression(C=100, random_state=0), csr_container(X), Y1)

    # 使用不同的参数设置检查预测结果
    check_predictions(LogisticRegression(fit_intercept=False, random_state=0), X, Y1)
    # 使用 Logistic 回归模型进行预测检查，fit_intercept 参数设置为 False，随机种子为 0
    check_predictions(
        LogisticRegression(fit_intercept=False, random_state=0), csr_container(X), Y1
    )
def test_logistic_cv_mock_scorer():
    # 定义一个模拟评分器类，用于测试
    class MockScorer:
        def __init__(self):
            self.calls = 0  # 初始化调用次数
            self.scores = [0.1, 0.4, 0.8, 0.5]  # 预设的评分列表

        def __call__(self, model, X, y, sample_weight=None):
            # 模拟评分器的调用过程，返回预设的评分并更新调用次数
            score = self.scores[self.calls % len(self.scores)]
            self.calls += 1
            return score

    mock_scorer = MockScorer()  # 实例化模拟评分器对象
    Cs = [1, 2, 3, 4]  # 模型参数的备选列表
    cv = 2  # 交叉验证的折数

    lr = LogisticRegressionCV(Cs=Cs, scoring=mock_scorer, cv=cv)  # 创建逻辑回归交叉验证对象
    X, y = make_classification(random_state=0)  # 生成一个样本数据集
    lr.fit(X, y)  # 使用样本数据集进行拟合

    # 断言：Cs[2] 对应的参数值具有最高的评分 (0.8)
    assert lr.C_[0] == Cs[2]

    # 断言：评分器被调用了 8 次 (cv * len(Cs))
    assert mock_scorer.calls == cv * len(Cs)

    # 重置模拟评分器的调用次数
    mock_scorer.calls = 0
    custom_score = lr.score(X, lr.predict(X))

    # 断言：自定义评分等于评分器的第一个评分 (0.1)
    assert custom_score == mock_scorer.scores[0]
    # 断言：模拟评分器的调用次数为 1
    assert mock_scorer.calls == 1


@skip_if_no_parallel
def test_lr_liblinear_warning():
    n_samples, n_features = iris.data.shape  # 获取样本数据的形状信息
    target = iris.target_names[iris.target]  # 获取目标变量的名称列表

    lr = LogisticRegression(solver="liblinear", n_jobs=2)  # 创建逻辑回归对象
    warning_message = (
        "'n_jobs' > 1 does not have any effect when"
        " 'solver' is set to 'liblinear'. Got 'n_jobs'"
        " = 2."
    )
    with pytest.warns(UserWarning, match=warning_message):
        lr.fit(iris.data, target)  # 使用样本数据和目标变量进行拟合


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_predict_3_classes(csr_container):
    check_predictions(LogisticRegression(C=10), X, Y2)  # 检查逻辑回归模型对数据的预测结果
    check_predictions(LogisticRegression(C=10), csr_container(X), Y2)  # 检查逻辑回归模型对数据的预测结果


# TODO(1.7): remove filterwarnings after the deprecation of multi_class
@pytest.mark.filterwarnings("ignore:.*'multi_class' was deprecated.*:FutureWarning")
@pytest.mark.parametrize(
    "clf",
    [
        LogisticRegression(C=len(iris.data), solver="liblinear", multi_class="ovr"),
        LogisticRegression(C=len(iris.data), solver="lbfgs"),
        LogisticRegression(C=len(iris.data), solver="newton-cg"),
        LogisticRegression(
            C=len(iris.data), solver="sag", tol=1e-2, multi_class="ovr", random_state=42
        ),
        LogisticRegression(
            C=len(iris.data),
            solver="saga",
            tol=1e-2,
            multi_class="ovr",
            random_state=42,
        ),
        LogisticRegression(
            C=len(iris.data), solver="newton-cholesky", multi_class="ovr"
        ),
    ],
)
def test_predict_iris(clf):
    """Test logistic regression with the iris dataset.

    Test that both multinomial and OvR solvers handle multiclass data correctly and
    give good accuracy score (>0.95) for the training data.
    """
    n_samples, n_features = iris.data.shape  # 获取样本数据的形状信息
    target = iris.target_names[iris.target]  # 获取目标变量的名称列表

    if clf.solver == "lbfgs":
        # lbfgs 在默认情况下对鸢尾花数据的收敛存在问题，因此使用忽略警告过滤器
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf.fit(iris.data, target)  # 使用样本数据和目标变量进行拟合
    else:
        # 如果分类器是非概率型的，使用 fit 方法拟合数据
        clf.fit(iris.data, target)
    # 断言目标变量的唯一值与分类器的类别属性相等
    assert_array_equal(np.unique(target), clf.classes_)

    # 使用分类器进行预测
    pred = clf.predict(iris.data)
    # 断言预测的准确率大于 0.95
    assert np.mean(pred == target) > 0.95

    # 使用分类器预测样本属于各个类别的概率
    probabilities = clf.predict_proba(iris.data)
    # 断言每个样本属于各个类别的概率之和等于 1
    assert_allclose(probabilities.sum(axis=1), np.ones(n_samples))

    # 将预测的概率最高的类别转换为类别名称
    pred = iris.target_names[probabilities.argmax(axis=1)]
    # 断言转换后的预测准确率大于 0.95
    assert np.mean(pred == target) > 0.95
# TODO(1.7): remove filterwarnings after the deprecation of multi_class
@pytest.mark.filterwarnings("ignore:.*'multi_class' was deprecated.*:FutureWarning")
# 使用pytest的参数化测试标记，测试不同的LR模型：LogisticRegression和LogisticRegressionCV
@pytest.mark.parametrize("LR", [LogisticRegression, LogisticRegressionCV])
def test_check_solver_option(LR):
    # 加载鸢尾花数据集的特征和标签
    X, y = iris.data, iris.target

    # 遍历支持'multinomial'多类别后端的求解器，只支持'liblinear'和'newton-cholesky'
    for solver in ["liblinear", "newton-cholesky"]:
        # 准备错误消息
        msg = f"Solver {solver} does not support a multinomial backend."
        # 创建LR模型，设置求解器为当前solver，多类别设置为'multinomial'
        lr = LR(solver=solver, multi_class="multinomial")
        # 断言抛出值错误，并匹配特定消息
        with pytest.raises(ValueError, match=msg):
            lr.fit(X, y)

    # 遍历除了'liblinear'和'saga'之外的所有求解器
    for solver in ["lbfgs", "newton-cg", "newton-cholesky", "sag"]:
        # 准备错误消息
        msg = "Solver %s supports only 'l2' or None penalties," % solver
        # 创建LR模型，设置求解器为当前solver，惩罚项设置为'l1'，多类别设置为'ovr'
        lr = LR(solver=solver, penalty="l1", multi_class="ovr")
        # 断言抛出值错误，并匹配特定消息
        with pytest.raises(ValueError, match=msg):
            lr.fit(X, y)

    # 遍历所有求解器，准备错误消息
    for solver in ["lbfgs", "newton-cg", "newton-cholesky", "sag", "saga"]:
        msg = "Solver %s supports only dual=False, got dual=True" % solver
        # 创建LR模型，设置求解器为当前solver，双重性设置为True，多类别设置为'ovr'
        lr = LR(solver=solver, dual=True, multi_class="ovr")
        # 断言抛出值错误，并匹配特定消息
        with pytest.raises(ValueError, match=msg):
            lr.fit(X, y)

    # 只有'saga'求解器支持elasticnet惩罚项，测试liblinear求解器
    for solver in ["liblinear"]:
        msg = f"Only 'saga' solver supports elasticnet penalty, got solver={solver}."
        # 创建LR模型，设置求解器为当前solver，惩罚项设置为'elasticnet'
        lr = LR(solver=solver, penalty="elasticnet")
        # 断言抛出值错误，并匹配特定消息
        with pytest.raises(ValueError, match=msg):
            lr.fit(X, y)

    # 对于liblinear求解器，不支持penalty='none'，LogisticRegressionCV根本不支持penalty='none'
    if LR is LogisticRegression:
        msg = "penalty=None is not supported for the liblinear solver"
        # 创建LR模型，设置惩罚项为None，求解器为'liblinear'
        lr = LR(penalty=None, solver="liblinear")
        # 断言抛出值错误，并匹配特定消息
        with pytest.raises(ValueError, match=msg):
            lr.fit(X, y)


@pytest.mark.parametrize("LR", [LogisticRegression, LogisticRegressionCV])
def test_elasticnet_l1_ratio_err_helpful(LR):
    # 检查在penalty="elasticnet"但未指定l1_ratio时是否引发了信息丰富的错误消息
    model = LR(penalty="elasticnet", solver="saga")
    # 断言抛出值错误，并匹配包含'l1_ratio'的消息
    with pytest.raises(ValueError, match=r".*l1_ratio.*"):
        model.fit(np.array([[1, 2], [3, 4]]), np.array([0, 1]))


# TODO(1.7): remove whole test with deprecation of multi_class
@pytest.mark.filterwarnings("ignore:.*'multi_class' was deprecated.*:FutureWarning")
# 使用pytest的参数化测试标记，测试多分类逻辑回归对二元问题的表现
@pytest.mark.parametrize("solver", ["lbfgs", "newton-cg", "sag", "saga"])
def test_multinomial_binary(solver):
    # 在二元问题上测试多分类逻辑回归
    # 将鸢尾花数据集的目标变量转换为二元
    target = (iris.target > 0).astype(np.intp)
    target = np.array(["setosa", "not-setosa"])[target]

    # 创建LogisticRegression模型，设置求解器为当前solver，多类别设置为'multinomial'
    clf = LogisticRegression(
        solver=solver, multi_class="multinomial", random_state=42, max_iter=2000
    )
    # 拟合模型
    clf.fit(iris.data, target)
    # 断言分类器的系数形状是否为 (1, 特征数)，用于验证分类器是否正确初始化
    assert clf.coef_.shape == (1, iris.data.shape[1])
    # 断言分类器的截距形状是否为 (1,)，用于验证分类器是否正确初始化
    assert clf.intercept_.shape == (1,)
    # 使用 iris 数据集进行预测，并断言预测结果与真实标签相等，用于验证分类器是否正确训练
    assert_array_equal(clf.predict(iris.data), target)

    # 使用 LogisticRegression 构造一个多分类逻辑回归模型
    mlr = LogisticRegression(
        solver=solver,  # 指定求解器
        multi_class="multinomial",  # 指定多分类策略为 multinomial
        random_state=42,  # 设定随机数种子
        fit_intercept=False  # 不使用截距项进行拟合
    )
    # 使用 iris 数据集训练多分类逻辑回归模型
    mlr.fit(iris.data, target)
    # 使用分类器的类别信息，根据预测的对数概率得出预测类别
    pred = clf.classes_[np.argmax(clf.predict_log_proba(iris.data), axis=1)]
    # 断言预测准确率是否大于 90%，用于验证模型训练效果
    assert np.mean(pred == target) > 0.9
# TODO(1.7): remove filterwarnings after the deprecation of multi_class
# Maybe even remove this whole test as correctness of multinomial loss is tested
# elsewhere.
# 使用 pytest 的标记来忽略特定的警告信息，该警告与 'multi_class' 参数的废弃相关。
@pytest.mark.filterwarnings("ignore:.*'multi_class' was deprecated.*:FutureWarning")
def test_multinomial_binary_probabilities(global_random_seed):
    # Test multinomial LR gives expected probabilities based on the
    # decision function, for a binary problem.
    # 创建一个随机生成的二分类数据集
    X, y = make_classification(random_state=global_random_seed)
    # 初始化逻辑回归分类器，设置 multi_class 参数为 "multinomial"，solver 为 "saga"，tolerance 为 1e-3
    # 并设置随机种子
    clf = LogisticRegression(
        multi_class="multinomial",
        solver="saga",
        tol=1e-3,
        random_state=global_random_seed,
    )
    # 在数据集上拟合分类器
    clf.fit(X, y)

    # 获取分类器的决策函数和预测的概率
    decision = clf.decision_function(X)
    proba = clf.predict_proba(X)

    # 计算预期的概率，针对类别 1 的概率
    expected_proba_class_1 = np.exp(decision) / (np.exp(decision) + np.exp(-decision))
    expected_proba = np.c_[1 - expected_proba_class_1, expected_proba_class_1]

    # 断言分类器预测的概率与预期的概率非常接近
    assert_almost_equal(proba, expected_proba)


@pytest.mark.parametrize("coo_container", COO_CONTAINERS)
def test_sparsify(coo_container):
    # Test sparsify and densify members.
    # 获取数据集的样本数和特征数
    n_samples, n_features = iris.data.shape
    # 获取目标标签名称
    target = iris.target_names[iris.target]
    # 对数据进行标准化处理
    X = scale(iris.data)
    # 初始化逻辑回归分类器，并在训练集上拟合分类器
    clf = LogisticRegression(random_state=0).fit(X, target)

    # 获取在密集数据上的决策函数值
    pred_d_d = clf.decision_function(X)

    # 将分类器的系数矩阵转换为稀疏格式
    clf.sparsify()
    assert sparse.issparse(clf.coef_)
    # 获取在稀疏数据上的决策函数值
    pred_s_d = clf.decision_function(X)

    # 创建一个 COO 格式的数据容器
    sp_data = coo_container(X)
    # 获取在稀疏数据上的决策函数值
    pred_s_s = clf.decision_function(sp_data)

    # 将分类器的系数矩阵转换为密集格式
    clf.densify()
    # 获取在密集数据上的决策函数值
    pred_d_s = clf.decision_function(sp_data)

    # 断言不同数据格式下的决策函数值应当非常接近
    assert_array_almost_equal(pred_d_d, pred_s_d)
    assert_array_almost_equal(pred_d_d, pred_s_s)
    assert_array_almost_equal(pred_d_d, pred_d_s)


def test_inconsistent_input():
    # Test that an exception is raised on inconsistent input
    # 创建一个随机数生成器
    rng = np.random.RandomState(0)
    # 创建一个随机的数据矩阵
    X_ = rng.random_sample((5, 10))
    y_ = np.ones(X_.shape[0])
    y_[0] = 0

    # 初始化逻辑回归分类器
    clf = LogisticRegression(random_state=0)

    # 为训练数据设置错误的维度
    y_wrong = y_[:-1]

    # 使用 pytest 断言异常被正确地抛出
    with pytest.raises(ValueError):
        clf.fit(X, y_wrong)

    # 为测试数据设置错误的维度
    with pytest.raises(ValueError):
        clf.fit(X_, y_).predict(rng.random_sample((3, 12)))


def test_write_parameters():
    # Test that we can write to coef_ and intercept_
    # 初始化逻辑回归分类器
    clf = LogisticRegression(random_state=0)
    # 在训练集上拟合分类器
    clf.fit(X, Y1)
    # 将分类器的系数矩阵和截距向量都置为0
    clf.coef_[:] = 0
    clf.intercept_[:] = 0
    # 断言分类器的决策函数值全为0
    assert_array_almost_equal(clf.decision_function(X), 0)


def test_nan():
    # Test proper NaN handling.
    # Regression test for Issue #252: fit used to go into an infinite loop.
    # 创建一个包含 NaN 值的数据集
    Xnan = np.array(X, dtype=np.float64)
    Xnan[0, 1] = np.nan
    # 初始化逻辑回归分类器
    logistic = LogisticRegression(random_state=0)

    # 使用 pytest 断言异常被正确地抛出
    with pytest.raises(ValueError):
        logistic.fit(Xnan, Y1)


def test_consistency_path():
    # Test that the path algorithm is consistent
    # 创建一个随机数生成器
    rng = np.random.RandomState(0)
    # 生成两个正态分布的数据集，组合成一个包含正类和负类的数据集
    X = np.concatenate((rng.randn(100, 2) + [1, 1], rng.randn(100, 2)))
    y = [1] * 100 + [-1] * 100
    # 创建一个包含10个元素的对数空间数组，范围从1到10000
    Cs = np.logspace(0, 4, 10)

    # 将函数 ignore_warnings 赋值给变量 f
    # 由于 LIBLINEAR 程序会对截距进行惩罚，因此不能使用 fit_intercept=True 进行测试
    for solver in ["sag", "saga"]:
        # 使用 ignore_warnings 函数调用 _logistic_regression_path 函数，获取系数、Cs 值和消息
        coefs, Cs, _ = f(_logistic_regression_path)(
            X,
            y,
            Cs=Cs,
            fit_intercept=False,
            tol=1e-5,
            solver=solver,
            max_iter=1000,
            random_state=0,
        )
        # 针对每个 Cs 值进行迭代
        for i, C in enumerate(Cs):
            # 创建 LogisticRegression 对象 lr，使用指定的参数进行初始化
            lr = LogisticRegression(
                C=C,
                fit_intercept=False,
                tol=1e-5,
                solver=solver,
                random_state=0,
                max_iter=1000,
            )
            # 使用训练数据 X, y 进行拟合
            lr.fit(X, y)
            # 获取 LogisticRegression 模型的系数
            lr_coef = lr.coef_.ravel()
            # 断言 lr_coef 和预期的 coefs[i] 数组几乎相等，精度为小数点后四位
            assert_array_almost_equal(
                lr_coef, coefs[i], decimal=4, err_msg="with solver = %s" % solver
            )

    # 针对 fit_intercept=True 的情况进行测试
    for solver in ("lbfgs", "newton-cg", "newton-cholesky", "liblinear", "sag", "saga"):
        # 设置 Cs 数组只包含一个元素 1000
        Cs = [1e3]
        # 使用 ignore_warnings 函数调用 _logistic_regression_path 函数，获取系数、Cs 值和消息
        coefs, Cs, _ = f(_logistic_regression_path)(
            X,
            y,
            Cs=Cs,
            tol=1e-6,
            solver=solver,
            intercept_scaling=10000.0,
            random_state=0,
        )
        # 创建 LogisticRegression 对象 lr，使用指定的参数进行初始化
        lr = LogisticRegression(
            C=Cs[0],
            tol=1e-6,
            intercept_scaling=10000.0,
            random_state=0,
            solver=solver,
        )
        # 使用训练数据 X, y 进行拟合
        lr.fit(X, y)
        # 获取 LogisticRegression 模型的系数和截距并连接成一个数组 lr_coef
        lr_coef = np.concatenate([lr.coef_.ravel(), lr.intercept_])
        # 断言 lr_coef 和预期的 coefs[0] 数组几乎相等，精度为小数点后四位
        assert_array_almost_equal(
            lr_coef, coefs[0], decimal=4, err_msg="with solver = %s" % solver
        )
def test_logistic_regression_path_convergence_fail():
    # 设置随机数生成器，种子为0
    rng = np.random.RandomState(0)
    # 生成特征矩阵X，包括两个分布：一个偏移量为[1, 1]的正态分布和一个标准正态分布
    X = np.concatenate((rng.randn(100, 2) + [1, 1], rng.randn(100, 2)))
    # 生成标签向量y，前100个为1，后100个为-1
    y = [1] * 100 + [-1] * 100
    # 设定正则化参数列表Cs为[1000.0]
    Cs = [1e3]

    # 使用pytest检查收敛警告
    with pytest.warns(ConvergenceWarning) as record:
        # 调用_logistic_regression_path函数，测试其收敛性
        _logistic_regression_path(
            X, y, Cs=Cs, tol=0.0, max_iter=1, random_state=0, verbose=0
        )

    # 确认仅有一条警告记录
    assert len(record) == 1
    # 获取警告消息
    warn_msg = record[0].message.args[0]
    # 确认警告消息包含"lbfgs failed to converge"
    assert "lbfgs failed to converge" in warn_msg
    # 确认警告消息包含"Increase the number of iterations"
    assert "Increase the number of iterations" in warn_msg
    # 确认警告消息包含"scale the data"
    assert "scale the data" in warn_msg
    # 确认警告消息包含"linear_model.html#logistic-regression"
    assert "linear_model.html#logistic-regression" in warn_msg


def test_liblinear_dual_random_state():
    # 对于liblinear求解器，仅在dual=True时random_state才相关
    X, y = make_classification(n_samples=20, random_state=0)
    # 初始化LogisticRegression对象lr1
    lr1 = LogisticRegression(
        random_state=0,
        dual=True,
        tol=1e-3,
        solver="liblinear",
    )
    # 使用数据拟合lr1
    lr1.fit(X, y)
    # 初始化LogisticRegression对象lr2
    lr2 = LogisticRegression(
        random_state=0,
        dual=True,
        tol=1e-3,
        solver="liblinear",
    )
    # 使用数据拟合lr2
    lr2.fit(X, y)
    # 初始化LogisticRegression对象lr3
    lr3 = LogisticRegression(
        random_state=8,
        dual=True,
        tol=1e-3,
        solver="liblinear",
    )
    # 使用数据拟合lr3
    lr3.fit(X, y)

    # 确认相同随机状态下，lr1和lr2的系数几乎相等
    assert_array_almost_equal(lr1.coef_, lr2.coef_)
    # 对于不同随机状态，lr1和lr3的系数不应该几乎相等
    msg = "Arrays are not almost equal to 6 decimals"
    with pytest.raises(AssertionError, match=msg):
        assert_array_almost_equal(lr1.coef_, lr3.coef_)


def test_logistic_cv():
    # 测试LogisticRegressionCV对象
    n_samples, n_features = 50, 5
    rng = np.random.RandomState(0)
    # 生成随机特征矩阵X_ref
    X_ref = rng.randn(n_samples, n_features)
    # 生成随机标签向量y
    y = np.sign(X_ref.dot(5 * rng.randn(n_features)))
    # 数据标准化
    X_ref -= X_ref.mean()
    X_ref /= X_ref.std()
    # 初始化LogisticRegressionCV对象lr_cv
    lr_cv = LogisticRegressionCV(
        Cs=[1.0], fit_intercept=False, solver="liblinear", cv=3
    )
    # 使用数据拟合lr_cv
    lr_cv.fit(X_ref, y)
    # 初始化LogisticRegression对象lr
    lr = LogisticRegression(C=1.0, fit_intercept=False, solver="liblinear")
    # 使用数据拟合lr
    lr.fit(X_ref, y)

    # 确认lr和lr_cv的系数几乎相等
    assert_array_almost_equal(lr.coef_, lr_cv.coef_)
    # 确认lr_cv的系数形状为(1, n_features)
    assert_array_equal(lr_cv.coef_.shape, (1, n_features))
    # 确认lr_cv的类别为[-1, 1]
    assert_array_equal(lr_cv.classes_, [-1, 1])
    # 确认lr_cv的类别数为2
    assert len(lr_cv.classes_) == 2

    # 确认lr_cv的系数路径形状为(1, 3, 1, n_features)
    coefs_paths = np.asarray(list(lr_cv.coefs_paths_.values()))
    assert_array_equal(coefs_paths.shape, (1, 3, 1, n_features))
    # 确认lr_cv的正则化参数Cs形状为(1,)
    assert_array_equal(lr_cv.Cs_.shape, (1,))
    # 确认lr_cv的分数形状为(1, 3, 1)
    scores = np.asarray(list(lr_cv.scores_.values()))
    assert_array_equal(scores.shape, (1, 3, 1))
    [
        # 元组列表的第一个元组，包含指标名称"accuracy"和空列表作为后续内容的占位符
        ("accuracy", [""]),
        # 元组列表的第二个元组，包含指标名称"precision"和列表["_macro", "_weighted"]，表示精度指标的宏平均和加权平均
        ("precision", ["_macro", "_weighted"]),
        # 不需要测试微平均，因为对于F1、精度和召回率，它与准确率相同
        # 参考：https://github.com/scikit-learn/scikit-learn/pull/11578#discussion_r203250062
        ("f1", ["_macro", "_weighted"]),
        # 元组列表的第四个元组，包含指标名称"neg_log_loss"和空列表作为后续内容的占位符
        ("neg_log_loss", [""]),
        # 元组列表的第五个元组，包含指标名称"recall"和列表["_macro", "_weighted"]，表示召回率指标的宏平均和加权平均
        ("recall", ["_macro", "_weighted"]),
    ],
# 定义测试函数，用于验证 LogisticRegressionCV 在使用多项式评分时计算交叉验证分数的正确性
def test_logistic_cv_multinomial_score(scoring, multiclass_agg_list):
    # 创建一个分类数据集，包括100个样本，3个类别，6个信息特征
    X, y = make_classification(
        n_samples=100, random_state=0, n_classes=3, n_informative=6
    )
    # 将数据集分为训练集和测试集
    train, test = np.arange(80), np.arange(80, 100)
    # 初始化 LogisticRegression 对象
    lr = LogisticRegression(C=1.0)
    # 使用 lbfgs 作为求解器，以支持多项式
    params = lr.get_params()
    # 删除 params 中的 "C", "n_jobs", "warm_start" 参数，以备后续设置在 _log_reg_scoring_path 中
    for key in ["C", "n_jobs", "warm_start"]:
        del params[key]
    # 使用训练集训练 LogisticRegression 模型
    lr.fit(X[train], y[train])
    # 对于每种多分类汇总方式，进行以下操作
    for averaging in multiclass_agg_list:
        # 获取指定评分方式的评分器
        scorer = get_scorer(scoring + averaging)
        # 断言通过 _log_reg_scoring_path 计算出的交叉验证分数与评分器在测试集上的评分结果近似相等
        assert_array_almost_equal(
            _log_reg_scoring_path(
                X,
                y,
                train,
                test,
                Cs=[1.0],
                scoring=scorer,
                pos_class=None,
                max_squared_sum=None,
                sample_weight=None,
                score_params=None,
                **(params | {"multi_class": "multinomial"}),
            )[2][0],
            scorer(lr, X[test], y[test]),
        )


# 测试 LogisticRegression(CV) 使用字符串标签的情况
def test_multinomial_logistic_regression_string_inputs():
    # 定义数据集维度
    n_samples, n_features, n_classes = 50, 5, 3
    # 创建分类数据集
    X_ref, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=3,
        random_state=0,
    )
    # 将数值标签转换为字符串标签
    y_str = LabelEncoder().fit(["bar", "baz", "foo"]).inverse_transform(y)
    # 对数值标签进行调整，使其在集合 (-1, 0, 1) 中
    y = np.array(y) - 1
    # 初始化 LogisticRegression 对象和 LogisticRegressionCV 对象，分别处理数值标签和字符串标签
    lr = LogisticRegression()
    lr_cv = LogisticRegressionCV(Cs=3)
    lr_str = LogisticRegression()
    lr_cv_str = LogisticRegressionCV(Cs=3)

    # 分别使用数值标签和字符串标签训练模型
    lr.fit(X_ref, y)
    lr_cv.fit(X_ref, y)
    lr_str.fit(X_ref, y_str)
    lr_cv_str.fit(X_ref, y_str)

    # 断言数值标签和字符串标签模型的系数近似相等
    assert_array_almost_equal(lr.coef_, lr_str.coef_)
    # 断言字符串标签模型的类别顺序与 ["bar", "baz", "foo"] 相同
    assert sorted(lr_str.classes_) == ["bar", "baz", "foo"]
    # 断言 LogisticRegressionCV 对象在数值标签和字符串标签上的系数近似相等
    assert_array_almost_equal(lr_cv.coef_, lr_cv_str.coef_)
    # 断言字符串标签模型的类别顺序与 ["bar", "baz", "foo"] 相同
    assert sorted(lr_str.classes_) == ["bar", "baz", "foo"]
    # 断言 LogisticRegressionCV 对象的类别顺序与 ["bar", "baz", "foo"] 相同
    assert sorted(lr_cv_str.classes_) == ["bar", "baz", "foo"]

    # 断言字符串标签预测结果的唯一值排序与 ["bar", "baz", "foo"] 相同
    assert sorted(np.unique(lr_str.predict(X_ref))) == ["bar", "baz", "foo"]
    # 断言 LogisticRegressionCV 对象预测结果的唯一值排序与 ["bar", "baz", "foo"] 相同
    assert sorted(np.unique(lr_cv_str.predict(X_ref))) == ["bar", "baz", "foo"]

    # 确保可以给字符串标签的 LogisticRegressionCV 对象设置类别权重
    lr_cv_str = LogisticRegression(class_weight={"bar": 1, "baz": 2, "foo": 0}).fit(
        X_ref, y_str
    )
    # 断言 LogisticRegressionCV 对象预测结果的唯一值排序与 ["bar", "baz"] 相同
    assert sorted(np.unique(lr_cv_str.predict(X_ref))) == ["bar", "baz"]
    # 使用 make_classification 函数生成一个包含 50 个样本和 5 个特征的分类数据集，设置随机数种子为 0
    X, y = make_classification(n_samples=50, n_features=5, random_state=0)
    
    # 将 X 中小于 1.0 的值设为 0.0
    X[X < 1.0] = 0.0
    
    # 将处理后的稀疏矩阵 X 转换为 CSR 格式的容器
    csr = csr_container(X)
    
    # 初始化一个 LogisticRegressionCV 分类器
    clf = LogisticRegressionCV()
    
    # 使用 X 和 y 训练 LogisticRegressionCV 分类器
    clf.fit(X, y)
    
    # 初始化另一个 LogisticRegressionCV 分类器
    clfs = LogisticRegressionCV()
    
    # 使用 csr（CSR 格式的容器）和 y 训练 LogisticRegressionCV 分类器
    clfs.fit(csr, y)
    
    # 断言稀疏矩阵和普通矩阵训练出来的模型参数（coef_ 和 intercept_）几乎相等
    assert_array_almost_equal(clfs.coef_, clf.coef_)
    assert_array_almost_equal(clfs.intercept_, clf.intercept_)
    
    # 断言稀疏矩阵和普通矩阵训练出来的模型正则化参数 C 相等
    assert clfs.C_ == clf.C_
# TODO(1.7): 在多类别模式（'multi_class'）被弃用后移除过滤警告
# 最好在测试完后删除整个测试。
@pytest.mark.filterwarnings("ignore:.*'multi_class' was deprecated.*:FutureWarning")
def test_ovr_multinomial_iris():
    # 测试使用鸢尾花数据集验证 OvR 和 multinomial 的正确性。
    train, target = iris.data, iris.target
    n_samples, n_features = train.shape

    # 使用分层 k 折交叉验证（StratifiedKFold）的索引（这里是基于鸢尾花数据集的精细化分类，即在合并类别 0 和 1 之前进行分层）来为 clf 和 clf1 提供预计算的折叠
    n_cv = 2
    cv = StratifiedKFold(n_cv)
    precomputed_folds = list(cv.split(train, target))

    # 在原始数据集上训练 clf，其中类别 0 和 1 是分开的
    clf = LogisticRegressionCV(cv=precomputed_folds, multi_class="ovr")
    clf.fit(train, target)

    # 将类别 0 和 1 合并后，在修改后的数据集上训练 clf1
    clf1 = LogisticRegressionCV(cv=precomputed_folds, multi_class="ovr")
    target_copy = target.copy()
    target_copy[target_copy == 0] = 1
    clf1.fit(train, target_copy)

    # 确保 OvR 对类别 2 的学习结果是相同的，无论类别 0 和 1 是否分开
    assert_allclose(clf.scores_[2], clf1.scores_[2])
    assert_allclose(clf.intercept_[2:], clf1.intercept_)
    assert_allclose(clf.coef_[2][np.newaxis, :], clf1.coef_)

    # 测试各种属性的形状。
    assert clf.coef_.shape == (3, n_features)
    assert_array_equal(clf.classes_, [0, 1, 2])
    coefs_paths = np.asarray(list(clf.coefs_paths_.values()))
    assert coefs_paths.shape == (3, n_cv, 10, n_features + 1)
    assert clf.Cs_.shape == (10,)
    scores = np.asarray(list(clf.scores_.values()))
    assert scores.shape == (3, n_cv, 10)

    # 测试鸢尾花数据集上，multinomial 比 OvR 提供更好准确度的情况。
    for solver in ["lbfgs", "newton-cg", "sag", "saga"]:
        max_iter = 500 if solver in ["sag", "saga"] else 30
        clf_multi = LogisticRegressionCV(
            solver=solver,
            max_iter=max_iter,
            random_state=42,
            tol=1e-3 if solver in ["sag", "saga"] else 1e-2,
            cv=2,
        )
        if solver == "lbfgs":
            # lbfgs 需要进行缩放以避免收敛警告
            train = scale(train)

        clf_multi.fit(train, target)
        multi_score = clf_multi.score(train, target)
        ovr_score = clf.score(train, target)
        assert multi_score > ovr_score

        # 测试 LogisticRegressionCV 的属性
        assert clf.coef_.shape == clf_multi.coef_.shape
        assert_array_equal(clf_multi.classes_, [0, 1, 2])
        coefs_paths = np.asarray(list(clf_multi.coefs_paths_.values()))
        assert coefs_paths.shape == (3, n_cv, 10, n_features + 1)
        assert clf_multi.Cs_.shape == (10,)
        scores = np.asarray(list(clf_multi.scores_.values()))
        assert scores.shape == (3, n_cv, 10)


def test_logistic_regression_solvers():
    """Test solvers converge to the same result."""
    # 使用 make_classification 函数生成一个具有10个特征和5个信息特征的分类数据集，设置随机种子为0
    X, y = make_classification(n_features=10, n_informative=5, random_state=0)

    # 定义回归器的参数字典，包括 fit_intercept=False 和 random_state=42
    params = dict(fit_intercept=False, random_state=42)

    # 创建包含不同求解器的逻辑回归模型，并在训练集 X, y 上拟合每个模型
    regressors = {
        solver: LogisticRegression(solver=solver, **params).fit(X, y)
        for solver in SOLVERS
    }

    # 遍历所有可能的求解器组合，比较它们的系数是否接近，精度为小数点后3位
    for solver_1, solver_2 in itertools.combinations(regressors, r=2):
        assert_array_almost_equal(
            regressors[solver_1].coef_, regressors[solver_2].coef_, decimal=3
        )
# 测试多类问题时，确保不同求解器收敛到相同结果
def test_logistic_regression_solvers_multiclass():
    # 创建具有多类的分类数据集，包括特征和目标变量
    X, y = make_classification(
        n_samples=20, n_features=20, n_informative=10, n_classes=3, random_state=0
    )
    # 设置收敛容差
    tol = 1e-8
    # 定义模型参数，包括拟合截距、容差和随机种子
    params = dict(fit_intercept=False, tol=tol, random_state=42)

    # 针对特定求解器覆盖最大迭代次数，以确保适当的收敛
    solver_max_iter = {"sag": 10_000, "saga": 10_000}

    # 创建逻辑回归模型的字典，针对除了["liblinear", "newton-cholesky"]之外的求解器
    regressors = {
        solver: LogisticRegression(
            solver=solver, max_iter=solver_max_iter.get(solver, 100), **params
        ).fit(X, y)
        for solver in set(SOLVERS) - set(["liblinear", "newton-cholesky"])
    }

    # 对每对不同求解器的组合进行比较，确保其系数接近
    for solver_1, solver_2 in itertools.combinations(regressors, r=2):
        assert_allclose(
            regressors[solver_1].coef_,
            regressors[solver_2].coef_,
            rtol=5e-3 if solver_2 == "saga" else 1e-3,
            err_msg=f"{solver_1} vs {solver_2}",
        )


# 参数化测试，测试逻辑回归的类权重对于LogisticRegressionCV的影响
@pytest.mark.parametrize("weight", [{0: 0.1, 1: 0.2}, {0: 0.1, 1: 0.2, 2: 0.5}])
@pytest.mark.parametrize("class_weight", ["weight", "balanced"])
def test_logistic_regressioncv_class_weights(weight, class_weight, global_random_seed):
    """Test class_weight for LogisticRegressionCV."""
    # 确定权重的类数
    n_classes = len(weight)
    # 如果类权重设置为"weight"，则使用给定的权重
    if class_weight == "weight":
        class_weight = weight

    # 创建具有指定特征和类别数的分类数据集
    X, y = make_classification(
        n_samples=30,
        n_features=3,
        n_repeated=0,
        n_informative=3,
        n_redundant=0,
        n_classes=n_classes,
        random_state=global_random_seed,
    )
    # 设置模型参数，包括正则化系数Cs、拟合截距、类别权重和容差
    params = dict(
        Cs=1,
        fit_intercept=False,
        class_weight=class_weight,
        tol=1e-8,
    )
    # 创建使用lbfgs求解器的LogisticRegressionCV对象
    clf_lbfgs = LogisticRegressionCV(solver="lbfgs", **params)

    # 对lbfgs求解器进行训练，忽略潜在的收敛警告
    with ignore_warnings(category=ConvergenceWarning):
        clf_lbfgs.fit(X, y)

    # 对于除了["lbfgs", "liblinear", "newton-cholesky"]之外的求解器进行比较
    for solver in set(SOLVERS) - set(["lbfgs", "liblinear", "newton-cholesky"]):
        # 创建使用不同求解器的LogisticRegressionCV对象
        clf = LogisticRegressionCV(solver=solver, **params)
        # 针对"sag"和"saga"求解器，调整参数以确保适当的收敛
        if solver in ("sag", "saga"):
            clf.set_params(
                tol=1e-18, max_iter=10000, random_state=global_random_seed + 1
            )
        # 训练模型
        clf.fit(X, y)

        # 比较不同求解器的系数是否接近lbfgs求解器的系数
        assert_allclose(
            clf.coef_, clf_lbfgs.coef_, rtol=1e-3, err_msg=f"{solver} vs lbfgs"
        )


# 测试逻辑回归模型的样本权重对模型的影响
def test_logistic_regression_sample_weights():
    # 创建具有指定特征和类别数的分类数据集
    X, y = make_classification(
        n_samples=20, n_features=5, n_informative=3, n_classes=2, random_state=0
    )
    # 样本权重设置为目标变量加一
    sample_weight = y + 1
    # 对于每种逻辑回归模型（LogisticRegression, LogisticRegressionCV），分别进行以下测试
    for LR in [LogisticRegression, LogisticRegressionCV]:
        kw = {"random_state": 42, "fit_intercept": False}
        if LR is LogisticRegressionCV:
            # 如果 LR 是 LogisticRegressionCV，则更新参数字典 kw
            kw.update({"Cs": 3, "cv": 3})

        # 测试传递 sample_weight 参数为全为1时，与不传 sample_weight 参数效果是否一致（默认为 None）
        for solver in ["lbfgs", "liblinear"]:
            # 创建两个逻辑回归模型实例，分别使用 sample_weight 为 None 和全为1的方式进行训练
            clf_sw_none = LR(solver=solver, **kw)
            clf_sw_ones = LR(solver=solver, **kw)
            clf_sw_none.fit(X, y)
            clf_sw_ones.fit(X, y, sample_weight=np.ones(y.shape[0]))
            # 断言两个模型的 coef_ 是否接近，相对误差为 1e-4
            assert_allclose(clf_sw_none.coef_, clf_sw_ones.coef_, rtol=1e-4)

        # 测试 lbfgs、newton-cg、newton-cholesky 和 'sag' 这几种求解器下 sample_weight 的效果是否一致
        clf_sw_lbfgs = LR(**kw, tol=1e-5)
        clf_sw_lbfgs.fit(X, y, sample_weight=sample_weight)
        for solver in set(SOLVERS) - set(["lbfgs"]):
            # 根据不同的求解器创建逻辑回归模型实例 clf_sw
            clf_sw = LR(solver=solver, tol=1e-10 if solver == "sag" else 1e-5, **kw)
            # 忽略由于数据集小而产生的收敛警告（针对 sag 求解器）
            with ignore_warnings():
                clf_sw.fit(X, y, sample_weight=sample_weight)
            # 断言不同求解器下的 coef_ 是否接近，相对误差为 1e-4
            assert_allclose(clf_sw_lbfgs.coef_, clf_sw.coef_, rtol=1e-4)

        # 测试传递 class_weight 参数为 {0: 1, 1: 2} 时，与传递 class_weight 参数为 {0: 1, 1: 1} 且相应调整 sample_weight 为 2 的效果是否一致
        for solver in ["lbfgs", "liblinear"]:
            # 创建两个逻辑回归模型实例，分别使用不同的 class_weight 和 sample_weight 进行训练
            clf_cw_12 = LR(solver=solver, class_weight={0: 1, 1: 2}, **kw)
            clf_cw_12.fit(X, y)
            clf_sw_12 = LR(solver=solver, **kw)
            clf_sw_12.fit(X, y, sample_weight=sample_weight)
            # 断言两个模型的 coef_ 是否接近，相对误差为 1e-4
            assert_allclose(clf_cw_12.coef_, clf_sw_12.coef_, rtol=1e-4)

    # 对于使用 l1 惩罚和 l2 惩罚并且 dual=True 的情况进行测试，因为修补后的 liblinear 代码有所不同
    clf_cw = LogisticRegression(
        solver="liblinear",
        fit_intercept=False,
        class_weight={0: 1, 1: 2},
        penalty="l1",
        tol=1e-5,
        random_state=42,
    )
    clf_cw.fit(X, y)
    clf_sw = LogisticRegression(
        solver="liblinear",
        fit_intercept=False,
        penalty="l1",
        tol=1e-5,
        random_state=42,
    )
    clf_sw.fit(X, y, sample_weight)
    # 断言使用 l1 惩罚时的 coef_ 是否接近，精度为小数点后四位
    assert_array_almost_equal(clf_cw.coef_, clf_sw.coef_, decimal=4)

    clf_cw = LogisticRegression(
        solver="liblinear",
        fit_intercept=False,
        class_weight={0: 1, 1: 2},
        penalty="l2",
        dual=True,
        random_state=42,
    )
    clf_cw.fit(X, y)
    clf_sw = LogisticRegression(
        solver="liblinear",
        fit_intercept=False,
        penalty="l2",
        dual=True,
        random_state=42,
    )
    clf_sw.fit(X, y, sample_weight)
    # 断言使用 l2 惩罚和 dual=True 时的 coef_ 是否接近，精度为小数点后四位
    assert_array_almost_equal(clf_cw.coef_, clf_sw.coef_, decimal=4)
# 定义一个辅助函数，用于计算类权重字典而不是数组形式
def _compute_class_weight_dictionary(y):
    # 获取类别列表
    classes = np.unique(y)
    # 计算平衡的类权重数组
    class_weight = compute_class_weight("balanced", classes=classes, y=y)
    # 将类别和对应的权重组成字典
    class_weight_dict = dict(zip(classes, class_weight))
    # 返回类权重字典
    return class_weight_dict


# 参数化测试函数，测试逻辑回归的类权重功能
@pytest.mark.parametrize("csr_container", [lambda x: x] + CSR_CONTAINERS)
def test_logistic_regression_class_weights(csr_container):
    # 缩放数据以避免使用 lbfgs 求解器时的收敛警告
    X_iris = scale(iris.data)
    
    # 多项式情况：移除类别 0 的 90% 数据
    X = X_iris[45:, :]
    X = csr_container(X)
    y = iris.target[45:]
    
    # 计算类别权重字典
    class_weight_dict = _compute_class_weight_dictionary(y)

    # 对于不包含 "liblinear" 和 "newton-cholesky" 的求解器集合
    for solver in set(SOLVERS) - set(["liblinear", "newton-cholesky"]):
        params = dict(solver=solver, max_iter=1000)
        
        # 创建使用平衡类权重的逻辑回归分类器
        clf1 = LogisticRegression(class_weight="balanced", **params)
        # 创建使用计算得的类权重字典的逻辑回归分类器
        clf2 = LogisticRegression(class_weight=class_weight_dict, **params)
        
        # 拟合分类器
        clf1.fit(X, y)
        clf2.fit(X, y)
        
        # 断言类别数为 3
        assert len(clf1.classes_) == 3
        
        # 断言权重系数在相对误差容差为 1e-4 的情况下近似相等
        assert_allclose(clf1.coef_, clf2.coef_, rtol=1e-4)
        
        # 与适当的样本权重相同
        sw = np.ones(X.shape[0])
        for c in clf1.classes_:
            sw[y == c] *= class_weight_dict[c]
        
        # 创建使用样本权重的逻辑回归分类器
        clf3 = LogisticRegression(**params).fit(X, y, sample_weight=sw)
        
        # 断言权重系数在相对误差容差为 1e-4 的情况下近似相等
        assert_allclose(clf3.coef_, clf2.coef_, rtol=1e-4)

    # 二分类情况：移除类别 0 的 90% 数据和类别 2 的 100% 数据
    X = X_iris[45:100, :]
    y = iris.target[45:100]
    
    # 重新计算类别权重字典
    class_weight_dict = _compute_class_weight_dictionary(y)

    # 对于所有求解器的情况
    for solver in SOLVERS:
        params = dict(solver=solver, max_iter=1000)
        
        # 创建使用平衡类权重的逻辑回归分类器
        clf1 = LogisticRegression(class_weight="balanced", **params)
        # 创建使用计算得的类权重字典的逻辑回归分类器
        clf2 = LogisticRegression(class_weight=class_weight_dict, **params)
        
        # 拟合分类器
        clf1.fit(X, y)
        clf2.fit(X, y)
        
        # 断言权重系数在小数点后 6 位的情况下近似相等
        assert_array_almost_equal(clf1.coef_, clf2.coef_, decimal=6)


# 测试逻辑回归的多项式选项
def test_logistic_regression_multinomial():
    # 测试逻辑回归中的多项式选项

    # 创建具有一些基本属性的逻辑回归
    n_samples, n_features, n_classes = 50, 20, 3
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        n_classes=n_classes,
        random_state=0,
    )

    # 标准化数据，均值为零
    X = StandardScaler(with_mean=False).fit_transform(X)

    # 使用 lbfgs 求解器作为参考
    solver = "lbfgs"
    ref_i = LogisticRegression(solver=solver, tol=1e-6)
    ref_w = LogisticRegression(solver=solver, fit_intercept=False, tol=1e-6)
    
    # 拟合参考分类器
    ref_i.fit(X, y)
    ref_w.fit(X, y)
    
    # 断言权重系数的形状符合预期
    assert ref_i.coef_.shape == (n_classes, n_features)
    assert ref_w.coef_.shape == (n_classes, n_features)
    # 对于每种求解器，创建一个 LogisticRegression 分类器 clf_i，带有指定的参数
    # solver 表示求解器类型，random_state 设置随机种子，max_iter 设置最大迭代次数，tol 设置收敛阈值
    clf_i = LogisticRegression(
        solver=solver,
        random_state=42,
        max_iter=2000,
        tol=1e-7,
    )
    # 创建另一个 LogisticRegression 分类器 clf_w，同样带有指定参数，但 fit_intercept 设置为 False
    clf_w = LogisticRegression(
        solver=solver,
        random_state=42,
        max_iter=2000,
        tol=1e-7,
        fit_intercept=False,
    )
    
    # 使用 clf_i 和 clf_w 分别拟合输入数据 X 和标签 y
    clf_i.fit(X, y)
    clf_w.fit(X, y)
    
    # 断言 clf_i 和 clf_w 的系数形状与类数和特征数匹配
    assert clf_i.coef_.shape == (n_classes, n_features)
    assert clf_w.coef_.shape == (n_classes, n_features)

    # 比较 lbfgs 求解器与其他求解器的解，通过 assert_allclose 近似判断是否相等
    assert_allclose(ref_i.coef_, clf_i.coef_, rtol=1e-3)
    assert_allclose(ref_w.coef_, clf_w.coef_, rtol=1e-2)
    assert_allclose(ref_i.intercept_, clf_i.intercept_, rtol=1e-3)

# 对于 lbfgs, newton-cg, sag, saga 求解器进行循环测试
for solver in ["lbfgs", "newton-cg", "sag", "saga"]:
    # 创建 LogisticRegressionCV 分类器 clf_path，带有指定参数
    clf_path = LogisticRegressionCV(
        solver=solver, max_iter=2000, tol=1e-6, Cs=[1.0]
    )
    # 使用 clf_path 拟合输入数据 X 和标签 y
    clf_path.fit(X, y)
    
    # 断言 clf_path 的系数和截距与 ref_i 的系数和截距近似相等
    assert_allclose(clf_path.coef_, ref_i.coef_, rtol=1e-2)
    assert_allclose(clf_path.intercept_, ref_i.intercept_, rtol=1e-2)
def test_liblinear_decision_function_zero():
    # 测试当决策函数值为零时的负预测。
    # 当决策函数值为零时，Liblinear 会预测为正类。这个测试用例用于验证我们不会做同样的预测。
    # 参见问题：https://github.com/scikit-learn/scikit-learn/issues/3600
    # 以及 PR：https://github.com/scikit-learn/scikit-learn/pull/3623

    # 生成一个样本为5，特征为5的分类数据集
    X, y = make_classification(n_samples=5, n_features=5, random_state=0)
    # 初始化 LogisticRegression 分类器，使用 liblinear 求解器
    clf = LogisticRegression(fit_intercept=False, solver="liblinear")
    # 在数据集上进行拟合
    clf.fit(X, y)

    # 创建一个决策函数值为零的虚拟数据
    X = np.zeros((5, 5))
    # 断言预测结果与零向量相等
    assert_array_equal(clf.predict(X), np.zeros(5))


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_liblinear_logregcv_sparse(csr_container):
    # 测试 LogRegCV 在使用 solver='liblinear' 时对稀疏矩阵的工作情况

    # 生成一个样本为10，特征为5的分类数据集
    X, y = make_classification(n_samples=10, n_features=5, random_state=0)
    # 初始化 LogisticRegressionCV 分类器，使用 liblinear 求解器
    clf = LogisticRegressionCV(solver="liblinear")
    # 在 CSR 容器上拟合模型
    clf.fit(csr_container(X), y)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_saga_sparse(csr_container):
    # 测试 LogRegCV 在使用 solver='saga' 时对稀疏矩阵的工作情况

    # 生成一个样本为10，特征为5的分类数据集
    X, y = make_classification(n_samples=10, n_features=5, random_state=0)
    # 初始化 LogisticRegressionCV 分类器，使用 saga 求解器和指定的公差
    clf = LogisticRegressionCV(solver="saga", tol=1e-2)
    # 在 CSR 容器上拟合模型
    clf.fit(csr_container(X), y)


def test_logreg_intercept_scaling_zero():
    # 测试当 fit_intercept 为 False 时，intercept_scaling 被忽略的情况

    # 初始化 LogisticRegression 分类器，fit_intercept 设置为 False
    clf = LogisticRegression(fit_intercept=False)
    # 在数据集上拟合模型
    clf.fit(X, Y1)
    # 断言截距为零
    assert clf.intercept_ == 0.0


def test_logreg_l1():
    # 因为 liblinear 对截距进行惩罚而 saga 不会，所以我们不拟合截距以便能够比较两个模型在收敛时的系数。

    rng = np.random.RandomState(42)
    n_samples = 50
    # 生成一个样本为50，特征为20的分类数据集
    X, y = make_classification(n_samples=n_samples, n_features=20, random_state=0)
    # 生成噪声数据和常量数据
    X_noise = rng.normal(size=(n_samples, 3))
    X_constant = np.ones(shape=(n_samples, 2))
    # 拼接数据集
    X = np.concatenate((X, X_noise, X_constant), axis=1)

    # 初始化使用 l1 惩罚的 LogisticRegression 分类器，使用 liblinear 求解器，fit_intercept 设置为 False
    lr_liblinear = LogisticRegression(
        penalty="l1",
        C=1.0,
        solver="liblinear",
        fit_intercept=False,
        tol=1e-10,
    )
    # 在数据集上拟合模型
    lr_liblinear.fit(X, y)

    # 初始化使用 l1 惩罚的 LogisticRegression 分类器，使用 saga 求解器，fit_intercept 设置为 False，最大迭代次数设置为1000，公差设置为1e-10
    lr_saga = LogisticRegression(
        penalty="l1",
        C=1.0,
        solver="saga",
        fit_intercept=False,
        max_iter=1000,
        tol=1e-10,
    )
    # 在数据集上拟合模型
    lr_saga.fit(X, y)
    # 断言两个模型在收敛时系数近似相等
    assert_array_almost_equal(lr_saga.coef_, lr_liblinear.coef_)

    # 噪声和常量特征应该通过 l1 惩罚被正则化为零
    assert_array_almost_equal(lr_liblinear.coef_[0, -5:], np.zeros(5))
    assert_array_almost_equal(lr_saga.coef_[0, -5:], np.zeros(5))


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_logreg_l1_sparse_data(csr_container):
    # 因为 liblinear 对截距进行惩罚而 saga 不会，所以我们不
    # 使用种子为42的随机数生成器创建一个 RandomState 实例
    rng = np.random.RandomState(42)
    # 设定样本数量为50
    n_samples = 50
    # 生成具有20个特征的分类数据集
    X, y = make_classification(n_samples=n_samples, n_features=20, random_state=0)
    # 生成一个均值为0，标准差为0.1的正态分布噪声数据，形状为(n_samples, 3)
    X_noise = rng.normal(scale=0.1, size=(n_samples, 3))
    # 创建一个形状为(n_samples, 2)的全零数组
    X_constant = np.zeros(shape=(n_samples, 2))
    # 将原始特征数据 X、噪声数据 X_noise 和全零数组 X_constant 进行拼接
    X = np.concatenate((X, X_noise, X_constant), axis=1)
    # 将小于1的值设置为0
    X[X < 1] = 0
    # 将数据转换为稀疏矩阵格式
    X = csr_container(X)

    # 使用 liblinear solver 的 LogisticRegression 模型，penalty 为 l1 正则化，不包含截距项
    lr_liblinear = LogisticRegression(
        penalty="l1",
        C=1.0,
        solver="liblinear",
        fit_intercept=False,
        tol=1e-10,
    )
    # 对数据进行拟合
    lr_liblinear.fit(X, y)

    # 使用 saga solver 的 LogisticRegression 模型，penalty 为 l1 正则化，不包含截距项，最大迭代次数为1000
    lr_saga = LogisticRegression(
        penalty="l1",
        C=1.0,
        solver="saga",
        fit_intercept=False,
        max_iter=1000,
        tol=1e-10,
    )
    # 对数据进行拟合
    lr_saga.fit(X, y)
    # 检查两个模型的系数是否几乎相等
    assert_array_almost_equal(lr_saga.coef_, lr_liblinear.coef_)
    # 检查 l1 正则化应该将噪声和常数特征的系数都约束为零
    assert_array_almost_equal(lr_liblinear.coef_[0, -5:], np.zeros(5))
    assert_array_almost_equal(lr_saga.coef_[0, -5:], np.zeros(5))

    # 检查在稀疏和密集数据上求解是否产生相同的结果
    lr_saga_dense = LogisticRegression(
        penalty="l1",
        C=1.0,
        solver="saga",
        fit_intercept=False,
        max_iter=1000,
        tol=1e-10,
    )
    # 对稀疏格式数据进行拟合
    lr_saga_dense.fit(X.toarray(), y)
    # 检查两个 saga solver 模型的系数是否几乎相等
    assert_array_almost_equal(lr_saga.coef_, lr_saga_dense.coef_)
# 使用 pytest.mark.parametrize 装饰器指定测试参数 random_seed 为 42
# 和 penalty 分别为 "l1" 和 "l2"
@pytest.mark.parametrize("random_seed", [42])
@pytest.mark.parametrize("penalty", ["l1", "l2"])
# 定义测试函数 test_logistic_regression_cv_refit，测试当 refit=True 时，
# logistic regression cv 使用 saga solver 是否能收敛到与固定正则化参数 logistic regression 相同的解。
def test_logistic_regression_cv_refit(random_seed, penalty):
    # 创建用于测试的特征矩阵 X 和目标向量 y，通过 make_classification 生成
    X, y = make_classification(n_samples=100, n_features=20, random_state=random_seed)
    
    # 定义共享的参数字典 common_params，包括 solver="saga"、penalty=penalty、
    # random_state=random_seed、max_iter=1000、tol=1e-12
    common_params = dict(
        solver="saga",
        penalty=penalty,
        random_state=random_seed,
        max_iter=1000,
        tol=1e-12,
    )
    
    # 创建 LogisticRegressionCV 模型 lr_cv，使用 Cs=[1.0] 和 refit=True
    # 其他参数从 common_params 中获取
    lr_cv = LogisticRegressionCV(Cs=[1.0], refit=True, **common_params)
    # 在训练数据 X, y 上拟合 lr_cv 模型
    lr_cv.fit(X, y)
    
    # 创建 LogisticRegression 模型 lr，使用 C=1.0 和 common_params 中的其他参数
    lr = LogisticRegression(C=1.0, **common_params)
    # 在训练数据 X, y 上拟合 lr 模型
    lr.fit(X, y)
    
    # 断言 lr_cv 模型的 coef_（系数）与 lr 模型的 coef_ 相近
    assert_array_almost_equal(lr_cv.coef_, lr.coef_)


# 定义测试函数 test_logreg_predict_proba_multinomial
def test_logreg_predict_proba_multinomial():
    # 创建用于测试的特征矩阵 X 和目标向量 y，通过 make_classification 生成
    X, y = make_classification(
        n_samples=10, n_features=20, random_state=0, n_classes=3, n_informative=10
    )

    # 创建 LogisticRegression 模型 clf_multi，使用 solver="lbfgs"
    clf_multi = LogisticRegression(solver="lbfgs")
    # 在训练数据 X, y 上拟合 clf_multi 模型
    clf_multi.fit(X, y)
    
    # 计算 clf_multi 模型的预测概率，并计算使用真实熵损失函数的损失 clf_multi_loss
    clf_multi_loss = log_loss(y, clf_multi.predict_proba(X))
    
    # 创建 OneVsRestClassifier 模型 clf_ovr，内部使用 LogisticRegression 模型，solver="lbfgs"
    clf_ovr = OneVsRestClassifier(LogisticRegression(solver="lbfgs"))
    # 在训练数据 X, y 上拟合 clf_ovr 模型
    clf_ovr.fit(X, y)
    
    # 计算 clf_ovr 模型的预测概率，并计算使用真实熵损失函数的损失 clf_ovr_loss
    clf_ovr_loss = log_loss(y, clf_ovr.predict_proba(X))
    
    # 断言 clf_ovr_loss 大于 clf_multi_loss，验证使用真实熵损失函数的损失较小
    assert clf_ovr_loss > clf_multi_loss
    
    # 重新计算 clf_multi_loss，使用 clf_multi 模型的预测概率和 _predict_proba_lr 函数的损失
    clf_multi_loss = log_loss(y, clf_multi.predict_proba(X))
    clf_wrong_loss = log_loss(y, clf_multi._predict_proba_lr(X))
    
    # 断言 clf_wrong_loss 大于 clf_multi_loss，验证使用 softmax 函数的损失较小
    assert clf_wrong_loss > clf_multi_loss


# 标记待办事项，版本为 1.7，待移除对 multi_class 警告过滤器的使用
@pytest.mark.filterwarnings("ignore:.*'multi_class' was deprecated.*:FutureWarning")
# 使用 pytest.mark.parametrize 装饰器指定测试参数 max_iter 为 np.arange(1, 5)
# multi_class 为 "ovr" 和 "multinomial"
# solver, message 参数化多个测试条件，对应不同的求解器和信息消息
@pytest.mark.parametrize("max_iter", np.arange(1, 5))
@pytest.mark.parametrize("multi_class", ["ovr", "multinomial"])
@pytest.mark.parametrize(
    "solver, message",
    [
        (
            "newton-cg",
            "newton-cg failed to converge.* Increase the number of iterations.",
        ),
        (
            "liblinear",
            "Liblinear failed to converge, increase the number of iterations.",
        ),
        ("sag", "The max_iter was reached which means the coef_ did not converge"),
        ("saga", "The max_iter was reached which means the coef_ did not converge"),
        ("lbfgs", "lbfgs failed to converge"),
        ("newton-cholesky", "Newton solver did not converge after [0-9]* iterations"),
    ],
)
# 测试最大迭代次数是否被达到
def test_max_iter(max_iter, multi_class, solver, message):
    # 加载鸢尾花数据集及其标签，复制标签以便修改
    X, y_bin = iris.data, iris.target.copy()
    y_bin[y_bin == 2] = 0

    # 如果使用的求解器不支持"multinomial"多分类，则跳过测试
    if solver in ("liblinear", "newton-cholesky") and multi_class == "multinomial":
        pytest.skip("'multinomial' is not supported by liblinear and newton-cholesky")
    # 如果使用的是"newton-cholesky"求解器，并且最大迭代次数大于1，则跳过测试
    if solver == "newton-cholesky" and max_iter > 1:
        pytest.skip("solver newton-cholesky might converge very fast")

    # 创建逻辑回归模型对象
    lr = LogisticRegression(
        max_iter=max_iter,
        tol=1e-15,
        multi_class=multi_class,
        random_state=0,
        solver=solver,
    )
    # 检查是否会产生收敛警告，符合预期的警告消息
    with pytest.warns(ConvergenceWarning, match=message):
        lr.fit(X, y_bin)

    # 断言迭代次数是否与最大迭代次数相等
    assert lr.n_iter_[0] == max_iter


# TODO(1.7): 在多类别的废弃后移除 filterwarnings
@pytest.mark.filterwarnings("ignore:.*'multi_class' was deprecated.*:FutureWarning")
@pytest.mark.parametrize("solver", SOLVERS)
def test_n_iter(solver):
    # 测试 self.n_iter_ 是否具有正确的格式
    X, y = iris.data, iris.target
    if solver == "lbfgs":
        # lbfgs 需要进行缩放以避免收敛警告
        X = scale(X)

    # 确定类别数目
    n_classes = np.unique(y).shape[0]
    assert n_classes == 3

    # 创建二分类子问题
    y_bin = y.copy()
    y_bin[y_bin == 2] = 0

    n_Cs = 4
    n_cv_fold = 2

    # 二分类情况
    clf = LogisticRegression(tol=1e-2, C=1.0, solver=solver, random_state=42)
    clf.fit(X, y_bin)
    assert clf.n_iter_.shape == (1,)

    # 使用交叉验证的逻辑回归模型
    clf_cv = LogisticRegressionCV(
        tol=1e-2, solver=solver, Cs=n_Cs, cv=n_cv_fold, random_state=42
    )
    clf_cv.fit(X, y_bin)
    assert clf_cv.n_iter_.shape == (1, n_cv_fold, n_Cs)

    # OvR 多类别情况
    clf.set_params(multi_class="ovr").fit(X, y)
    assert clf.n_iter_.shape == (n_classes,)

    clf_cv.set_params(multi_class="ovr").fit(X, y)
    assert clf_cv.n_iter_.shape == (n_classes, n_cv_fold, n_Cs)

    # multinomial 多类别情况
    if solver in ("liblinear", "newton-cholesky"):
        # 这些求解器仅支持一对多的多类别分类
        return

    # 当使用 multinomial 目标函数时，一次性解决所有类别的优化问题
    clf.set_params(multi_class="multinomial").fit(X, y)
    assert clf.n_iter_.shape == (1,)

    clf_cv.set_params(multi_class="multinomial").fit(X, y)
    assert clf_cv.n_iter_.shape == (1, n_cv_fold, n_Cs)


@pytest.mark.parametrize(
    "solver", sorted(set(SOLVERS) - set(["liblinear", "newton-cholesky"]))
)
@pytest.mark.parametrize("warm_start", (True, False))
@pytest.mark.parametrize("fit_intercept", (True, False))
def test_warm_start(solver, warm_start, fit_intercept):
    # 在相同数据上进行第二次拟合，1 次迭代时使用温暖启动应产生几乎相同的结果，
    # 没有温暖启动时应产生完全不同的结果。liblinear 求解器不支持温暖启动。
    # 从iris数据集中分别获取特征数据X和目标数据y
    X, y = iris.data, iris.target
    
    # 初始化LogisticRegression分类器，设置参数和随机种子
    clf = LogisticRegression(
        tol=1e-4,                      # 设置收敛阈值
        warm_start=warm_start,         # 是否启用热启动
        solver=solver,                 # 指定求解器类型
        random_state=42,               # 设置随机种子
        fit_intercept=fit_intercept,   # 是否拟合截距
    )
    
    # 忽略收敛警告，训练分类器
    with ignore_warnings(category=ConvergenceWarning):
        clf.fit(X, y)                  # 使用数据X和标签y进行模型拟合
        coef_1 = clf.coef_             # 获取第一次拟合后的系数
    
        clf.max_iter = 1               # 将最大迭代次数设置为1
        clf.fit(X, y)                  # 重新拟合分类器
    
    # 计算两次拟合系数之间的绝对差异
    cum_diff = np.sum(np.abs(coef_1 - clf.coef_))
    
    # 构建错误消息，描述与solver、fit_intercept和warm_start相关的热启动问题
    msg = (
        f"Warm starting issue with solver {solver}"  # 描述solver的热启动问题
        f"with {fit_intercept=} and {warm_start=}"   # 包含fit_intercept和warm_start的设置
    )
    
    # 如果启用了热启动，则断言绝对差异小于2.0；否则，断言绝对差异大于2.0
    if warm_start:
        assert 2.0 > cum_diff, msg
    else:
        assert cum_diff > 2.0, msg
# 使用参数化测试，依次对每个 csr_container 运行测试
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_saga_vs_liblinear(csr_container):
    # 载入鸢尾花数据集
    iris = load_iris()
    X, y = iris.data, iris.target
    # 扩展数据集使得数据 X 重复三次
    X = np.concatenate([X] * 3)
    # 扩展目标标签使得数据 y 重复三次
    y = np.concatenate([y] * 3)

    # 选择二分类数据
    X_bin = X[y <= 1]
    y_bin = y[y <= 1] * 2 - 1

    # 创建稀疏矩阵数据集
    X_sparse, y_sparse = make_classification(
        n_samples=50, n_features=20, random_state=0
    )
    # 将稀疏矩阵数据集转换为特定的格式（由 csr_container 指定的格式）
    X_sparse = csr_container(X_sparse)

    # 遍历二分类数据和稀疏数据的组合
    for X, y in ((X_bin, y_bin), (X_sparse, y_sparse)):
        # 遍历正则化惩罚项的选择
        for penalty in ["l1", "l2"]:
            n_samples = X.shape[0]
            # 遍历不同的 alpha 参数（对数空间中的三个值）
            for alpha in np.logspace(-1, 1, 3):
                # 使用 saga solver 的逻辑回归模型
                saga = LogisticRegression(
                    C=1.0 / (n_samples * alpha),
                    solver="saga",
                    max_iter=200,
                    fit_intercept=False,
                    penalty=penalty,
                    random_state=0,
                    tol=1e-6,
                )

                # 使用 liblinear solver 的逻辑回归模型
                liblinear = LogisticRegression(
                    C=1.0 / (n_samples * alpha),
                    solver="liblinear",
                    max_iter=200,
                    fit_intercept=False,
                    penalty=penalty,
                    random_state=0,
                    tol=1e-6,
                )

                # 在当前数据上拟合模型
                saga.fit(X, y)
                liblinear.fit(X, y)
                # 断言两种 solver 的系数数组近似相等
                assert_array_almost_equal(saga.coef_, liblinear.coef_, 3)


# TODO(1.7): 移除多类别警告后的 filterwarnings
@pytest.mark.filterwarnings("ignore:.*'multi_class' was deprecated.*:FutureWarning")
# 参数化测试多类别的选择
@pytest.mark.parametrize("multi_class", ["ovr", "multinomial"])
# 参数化测试 solver 的选择
@pytest.mark.parametrize(
    "solver", ["liblinear", "newton-cg", "newton-cholesky", "saga"]
)
# 参数化测试是否拟合截距项
@pytest.mark.parametrize("fit_intercept", [False, True])
# 参数化测试 csr_container 的选择
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_dtype_match(solver, multi_class, fit_intercept, csr_container):
    # 测试确保 np.float32 输入数据在可能的情况下不会转换为 np.float64，并且无论输入格式如何，输出都大致相同。

    # 如果 solver 是 liblinear 并且 multi_class 是 multinomial，则跳过测试
    if solver in ("liblinear", "newton-cholesky") and multi_class == "multinomial":
        pytest.skip(f"Solver={solver} does not support multinomial logistic.")

    # 根据 solver 选择不同的输出数据类型
    out32_type = np.float64 if solver == "liblinear" else np.float32

    # 使用 np.float32 类型创建输入数据和标签
    X_32 = np.array(X).astype(np.float32)
    y_32 = np.array(Y1).astype(np.float32)
    # 使用 np.float64 类型创建输入数据和标签
    X_64 = np.array(X).astype(np.float64)
    y_64 = np.array(Y1).astype(np.float64)
    # 使用 csr_container 将输入数据转换为稀疏格式，并指定数据类型为 np.float32 或 np.float64
    X_sparse_32 = csr_container(X, dtype=np.float32)
    X_sparse_64 = csr_container(X, dtype=np.float64)
    # solver 的容忍度设置为 5e-4
    solver_tol = 5e-4

    # 创建逻辑回归模板对象
    lr_templ = LogisticRegression(
        solver=solver,
        multi_class=multi_class,
        random_state=42,
        tol=solver_tol,
        fit_intercept=fit_intercept,
    )

    # 检查 np.float32 类型的一致性
    lr_32 = clone(lr_templ)
    lr_32.fit(X_32, y_32)
    # 确保 lr_32.coef_ 的数据类型与 out32_type 相匹配
    assert lr_32.coef_.dtype == out32_type

    # 使用 lr_templ 的克隆对象创建 lr_32_sparse，用于稀疏数据的逻辑回归拟合
    lr_32_sparse = clone(lr_templ)
    lr_32_sparse.fit(X_sparse_32, y_32)
    # 确保 lr_32_sparse.coef_ 的数据类型与 out32_type 相匹配
    assert lr_32_sparse.coef_.dtype == out32_type

    # 使用 lr_templ 的克隆对象创建 lr_64，用于全精度数据的逻辑回归拟合
    lr_64 = clone(lr_templ)
    lr_64.fit(X_64, y_64)
    # 确保 lr_64.coef_ 的数据类型为 np.float64
    assert lr_64.coef_.dtype == np.float64

    # 使用 lr_templ 的克隆对象创建 lr_64_sparse，用于稀疏数据的全精度逻辑回归拟合
    lr_64_sparse = clone(lr_templ)
    lr_64_sparse.fit(X_sparse_64, y_64)
    # 确保 lr_64_sparse.coef_ 的数据类型为 np.float64
    assert lr_64_sparse.coef_.dtype == np.float64

    # solver_tol 限制了损失梯度的范数
    # dw ~= inv(H)*grad ==> |dw| ~= |inv(H)| * solver_tol，其中 H 为 Hessian 矩阵
    #
    # 参考链接：https://github.com/scikit-learn/scikit-learn/pull/13645
    #
    # 对于 Z = np.hstack((np.ones((3,1)), np.array(X))) 的情况
    # In [8]: np.linalg.norm(np.diag([0,2,2]) + np.linalg.inv((Z.T @ Z)/4))
    # Out[8]: 1.7193336918135917

    # 乘以 2 来获取球的直径
    atol = 2 * 1.72 * solver_tol
    if os.name == "nt" and _IS_32BIT:
        # FIXME：在 Windows 平台下，且为 32 位时的特殊处理
        atol = 1e-2

    # 确保 lr_32.coef_ 与 lr_64.coef_ 的数值精度在给定的公差范围内一致
    assert_allclose(lr_32.coef_, lr_64.coef_.astype(np.float32), atol=atol)

    if solver == "saga" and fit_intercept:
        # FIXME：对于 "saga" 求解器在稀疏数据上默认的 tol 和 max_iter 参数可能导致截距拟合不准确的问题
        atol = 1e-1

    # 确保 lr_32.coef_ 与 lr_32_sparse.coef_ 的数值精度在给定的公差范围内一致
    assert_allclose(lr_32.coef_, lr_32_sparse.coef_, atol=atol)
    # 确保 lr_64.coef_ 与 lr_64_sparse.coef_ 的数值精度在给定的公差范围内一致
    assert_allclose(lr_64.coef_, lr_64_sparse.coef_, atol=atol)
def test_warm_start_converge_LR():
    # 测试逻辑回归在热启动下是否收敛，使用 multi_class='multinomial'。非回归测试，用于解决问题 #10836

    # 创建一个随机数生成器，种子为0
    rng = np.random.RandomState(0)
    # 生成数据集 X，其中包含两个部分：一个均值为 [1, 1] 的正态分布数据和另一个随机正态分布数据
    X = np.concatenate((rng.randn(100, 2) + [1, 1], rng.randn(100, 2)))
    # 创建标签 y，前100个为1，后100个为-1
    y = np.array([1] * 100 + [-1] * 100)
    
    # 创建逻辑回归对象，不使用热启动，solver 为 "sag"，种子为0
    lr_no_ws = LogisticRegression(solver="sag", warm_start=False, random_state=0)
    # 创建逻辑回归对象，使用热启动，solver 为 "sag"，种子为0
    lr_ws = LogisticRegression(solver="sag", warm_start=True, random_state=0)

    # 训练并预测无热启动的逻辑回归，计算损失
    ultimate
    # 创建一个逻辑回归模型，使用L1正则化，指定正则化强度C，使用"saga"求解器，设置随机种子为0，容忍度为0.01
    l1_clf = LogisticRegression(
        penalty="l1", C=C, solver="saga", random_state=0, tol=1e-2
    )
    
    # 创建一个逻辑回归模型，使用L2正则化，指定正则化强度C，使用"saga"求解器，设置随机种子为0，容忍度为0.01
    l2_clf = LogisticRegression(
        penalty="l2", C=C, solver="saga", random_state=0, tol=1e-2
    )
    
    # 对三个分类器进行迭代训练：网格搜索优化后的模型(gs)，L1正则化的模型(l1_clf)，L2正则化的模型(l2_clf)
    for clf in (gs, l1_clf, l2_clf):
        clf.fit(X_train, y_train)
    
    # 断言网格搜索优化后的模型(gs)在测试集上的准确率大于等于L1正则化模型(l1_clf)在测试集上的准确率
    assert gs.score(X_test, y_test) >= l1_clf.score(X_test, y_test)
    
    # 断言网格搜索优化后的模型(gs)在测试集上的准确率大于等于L2正则化模型(l2_clf)在测试集上的准确率
    assert gs.score(X_test, y_test) >= l2_clf.score(X_test, y_test)
# 使用 pytest 的参数化功能，定义测试函数 test_LogisticRegression_elastic_net_objective，用于测试逻辑回归的弹性网络目标
# 参数 C 取自 np.logspace(-3, 2, 4)，l1_ratio 取值为 [0.1, 0.5, 0.9]
@pytest.mark.parametrize("C", np.logspace(-3, 2, 4))
@pytest.mark.parametrize("l1_ratio", [0.1, 0.5, 0.9])
def test_LogisticRegression_elastic_net_objective(C, l1_ratio):
    # 检查使用与目标匹配的惩罚训练是否会导致更低的目标值。
    # 在这里，我们训练一个带有 l2 和 elasticnet 惩罚的逻辑回归，并计算 elasticnet 目标。
    # a 的目标值应该大于 b 的目标值（两个目标值都是凸的）。
    
    # 创建具有特定特征的分类数据集
    X, y = make_classification(
        n_samples=1000,
        n_classes=2,
        n_features=20,
        n_informative=10,
        n_redundant=0,
        n_repeated=0,
        random_state=0,
    )
    # 缩放特征数据
    X = scale(X)

    # 创建使用 elasticnet 惩罚的逻辑回归模型
    lr_enet = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        random_state=0,
        C=C,
        l1_ratio=l1_ratio,
        fit_intercept=False,
    )
    # 创建使用 l2 惩罚的逻辑回归模型
    lr_l2 = LogisticRegression(
        penalty="l2", solver="saga", random_state=0, C=C, fit_intercept=False
    )
    # 分别用数据集 X, y 训练两个逻辑回归模型
    lr_enet.fit(X, y)
    lr_l2.fit(X, y)

    # 定义计算弹性网络目标值的函数
    def enet_objective(lr):
        coef = lr.coef_.ravel()  # 模型系数展平为一维数组
        obj = C * log_loss(y, lr.predict_proba(X))  # 计算交叉熵损失
        obj += l1_ratio * np.sum(np.abs(coef))  # 添加 l1 正则化项
        obj += (1.0 - l1_ratio) * 0.5 * np.dot(coef, coef)  # 添加 l2 正则化项
        return obj

    # 断言：弹性网络模型的目标值应该小于 l2 惩罚模型的目标值
    assert enet_objective(lr_enet) < enet_objective(lr_l2)


# 使用 pytest 的参数化功能，定义测试函数 test_LogisticRegressionCV_GridSearchCV_elastic_net
# 参数 n_classes 取值为 (2, 3)
@pytest.mark.parametrize("n_classes", (2, 3))
def test_LogisticRegressionCV_GridSearchCV_elastic_net(n_classes):
    # 确保 LogisticRegressionCV 在 penalty 为 elasticnet 时给出与 GridSearchCV 相同的最佳参数（l1_ratio 和 C）

    # 创建具有特定特征的分类数据集
    X, y = make_classification(
        n_samples=100, n_classes=n_classes, n_informative=3, random_state=0
    )

    # 定义交叉验证策略
    cv = StratifiedKFold(5)

    # 定义 l1_ratio 和 C 的取值范围
    l1_ratios = np.linspace(0, 1, 3)
    Cs = np.logspace(-4, 4, 3)

    # 创建使用 elasticnet 惩罚的 LogisticRegressionCV 模型
    lrcv = LogisticRegressionCV(
        penalty="elasticnet",
        Cs=Cs,
        solver="saga",
        cv=cv,
        l1_ratios=l1_ratios,
        random_state=0,
        tol=1e-2,
    )
    # 在数据集 X, y 上训练 LogisticRegressionCV 模型
    lrcv.fit(X, y)

    # 定义 GridSearchCV 的参数网格
    param_grid = {"C": Cs, "l1_ratio": l1_ratios}
    # 创建使用 elasticnet 惩罚的 LogisticRegression 模型
    lr = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        random_state=0,
        tol=1e-2,
    )
    # 创建 GridSearchCV 对象
    gs = GridSearchCV(lr, param_grid, cv=cv)
    # 在数据集 X, y 上进行网格搜索
    gs.fit(X, y)

    # 断言：GridSearchCV 找到的最佳 l1_ratio 应该与 LogisticRegressionCV 中的一致
    assert gs.best_params_["l1_ratio"] == lrcv.l1_ratio_[0]
    # 断言：GridSearchCV 找到的最佳 C 应该与 LogisticRegressionCV 中的一致
    assert gs.best_params_["C"] == lrcv.C_[0]


# 使用 pytest 的参数化功能，定义测试函数 test_LogisticRegressionCV_GridSearchCV_elastic_net_ovr
# 不处理 'multi_class' 被弃用后的警告
@pytest.mark.filterwarnings("ignore:.*'multi_class' was deprecated.*:FutureWarning")
def test_LogisticRegressionCV_GridSearchCV_elastic_net_ovr():
    # 确保 LogisticRegressionCV 在 penalty 为 elasticnet 且 multiclass 为 ovr 时，
    # 给出与 GridSearchCV 相同的最佳参数（l1 和 C）。我们不能像前一个测试那样比较 best_params，
    # 因为 multi_class='ovr' 的 LogisticRegressionCV 每个类别都有一个 C 和一个 l1 参数，
    # 而 LogisticRegression 会共享

    # 创建具有特定特征的分类数据集
    X, y = make_classification(
        n_samples=100, n_classes=n_classes, n_informative=3, random_state=0
    )

    # 定义交叉验证策略
    cv = StratifiedKFold(5)

    # 创建使用 elasticnet 惩罚的 LogisticRegressionCV 模型
    lrcv = LogisticRegressionCV(
        penalty="elasticnet",
        Cs=Cs,
        solver="saga",
        cv=cv,
        l1_ratios=l1_ratios,
        random_state=0,
        tol=1e-2,
    )
    # 在数据集 X, y 上训练 LogisticRegressionCV 模型
    lrcv.fit(X, y)

    # 定义 GridSearchCV 的参数网格
    param_grid = {"C": Cs, "l1_ratio": l1_ratios}
    # 创建使用 elasticnet 惩罚的 LogisticRegression 模型
    lr = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        random_state=0,
        tol=1e-2,
    )
    # 创建 GridSearchCV 对象
    gs = GridSearchCV(lr, param_grid, cv=cv)
    # 在数据集 X, y 上进行网格搜索
    gs.fit(X, y)

    # 断言：GridSearchCV 找到的最佳 l1_ratio 应该与 LogisticRegressionCV 中的一致
    assert gs.best_params_["l1_ratio"] == lrcv.l1_ratio_[0]
    # 断言：GridSearchCV 找到的最佳 C 应该与 LogisticRegressionCV 中的一致
    assert gs.best_params_["C"] == lrcv.C_[0]
    # 生成一个具有3个类别的分类数据集，包含100个样本和3个信息特征，使用随机种子0
    X, y = make_classification(
        n_samples=100, n_classes=3, n_informative=3, random_state=0
    )
    # 将数据集划分为训练集和测试集，使用相同的随机种子0
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # 创建一个分层交叉验证对象，将数据分成5折
    cv = StratifiedKFold(5)

    # 定义用于逻辑回归交叉验证的l1_ratio参数值
    l1_ratios = np.linspace(0, 1, 3)
    # 定义用于逻辑回归交叉验证的正则化参数C值
    Cs = np.logspace(-4, 4, 3)

    # 创建一个带有交叉验证的弹性网络逻辑回归CV对象，使用"saga"求解器，多分类方式为一对多，设置随机种子0，容忍度为0.01
    lrcv = LogisticRegressionCV(
        penalty="elasticnet",
        Cs=Cs,
        solver="saga",
        cv=cv,
        l1_ratios=l1_ratios,
        random_state=0,
        multi_class="ovr",
        tol=1e-2,
    )
    # 对训练集进行拟合
    lrcv.fit(X_train, y_train)

    # 定义逻辑回归的超参数网格，包括C和l1_ratio
    param_grid = {"C": Cs, "l1_ratio": l1_ratios}
    # 创建一个带有网格搜索交叉验证的弹性网络逻辑回归对象，使用"saga"求解器，多分类方式为一对多，设置随机种子0，容忍度为0.01
    lr = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        random_state=0,
        multi_class="ovr",
        tol=1e-2,
    )
    # 创建一个带有网格搜索交叉验证的GridSearchCV对象，对训练集进行拟合
    gs = GridSearchCV(lr, param_grid, cv=cv)
    gs.fit(X_train, y_train)

    # 检查训练集的预测结果和测试集的预测结果是否至少有80%相同
    assert (lrcv.predict(X_train) == gs.predict(X_train)).mean() >= 0.8
    assert (lrcv.predict(X_test) == gs.predict(X_test)).mean() >= 0.8
# TODO(1.7): remove filterwarnings after the deprecation of multi_class
# 在多类别分类器 multi_class 被弃用后移除警告
@pytest.mark.filterwarnings("ignore:.*'multi_class' was deprecated.*:FutureWarning")
# 使用参数化测试，测试 LogisticRegressionCV 当 refit=False 时的属性形状
@pytest.mark.parametrize("penalty", ("l2", "elasticnet"))
@pytest.mark.parametrize("multi_class", ("ovr", "multinomial", "auto"))
def test_LogisticRegressionCV_no_refit(penalty, multi_class):
    # Test LogisticRegressionCV attribute shapes when refit is False

    n_classes = 3
    n_features = 20
    # 生成一个分类数据集
    X, y = make_classification(
        n_samples=200,
        n_classes=n_classes,
        n_informative=n_classes,
        n_features=n_features,
        random_state=0,
    )

    # 定义正则化参数范围
    Cs = np.logspace(-4, 4, 3)
    if penalty == "elasticnet":
        # 如果正则化为 elasticnet，则定义 l1_ratio 参数范围
        l1_ratios = np.linspace(0, 1, 2)
    else:
        l1_ratios = None

    # 创建 LogisticRegressionCV 对象
    lrcv = LogisticRegressionCV(
        penalty=penalty,
        Cs=Cs,
        solver="saga",
        l1_ratios=l1_ratios,
        random_state=0,
        multi_class=multi_class,
        tol=1e-2,
        refit=False,
    )
    # 拟合模型
    lrcv.fit(X, y)
    # 断言模型属性的形状是否符合预期
    assert lrcv.C_.shape == (n_classes,)
    assert lrcv.l1_ratio_.shape == (n_classes,)
    assert lrcv.coef_.shape == (n_classes, n_features)


# TODO(1.7): remove filterwarnings after the deprecation of multi_class
# 在多类别分类器 multi_class 被弃用后移除警告
# 移除 multi_class 并修改预期的 n_iter_.shape 的第一个元素从 n_classes 到 1（根据文档字符串）
@pytest.mark.filterwarnings("ignore:.*'multi_class' was deprecated.*:FutureWarning")
def test_LogisticRegressionCV_elasticnet_attribute_shapes():
    # Make sure the shapes of scores_ and coefs_paths_ attributes are correct
    # when using elasticnet (added one dimension for l1_ratios)

    n_classes = 3
    n_features = 20
    # 生成一个分类数据集
    X, y = make_classification(
        n_samples=200,
        n_classes=n_classes,
        n_informative=n_classes,
        n_features=n_features,
        random_state=0,
    )

    # 定义正则化参数范围和 l1_ratio 参数范围
    Cs = np.logspace(-4, 4, 3)
    l1_ratios = np.linspace(0, 1, 2)

    n_folds = 2
    # 创建 LogisticRegressionCV 对象
    lrcv = LogisticRegressionCV(
        penalty="elasticnet",
        Cs=Cs,
        solver="saga",
        cv=n_folds,
        l1_ratios=l1_ratios,
        multi_class="ovr",
        random_state=0,
        tol=1e-2,
    )
    # 拟合模型
    lrcv.fit(X, y)
    # 将 coefs_paths_ 属性转换为数组并断言其形状是否符合预期
    coefs_paths = np.asarray(list(lrcv.coefs_paths_.values()))
    assert coefs_paths.shape == (
        n_classes,
        n_folds,
        Cs.size,
        l1_ratios.size,
        n_features + 1,
    )
    # 将 scores_ 属性转换为数组并断言其形状是否符合预期
    scores = np.asarray(list(lrcv.scores_.values()))
    assert scores.shape == (n_classes, n_folds, Cs.size, l1_ratios.size)

    # 断言 n_iter_ 属性的形状是否符合预期
    assert lrcv.n_iter_.shape == (n_classes, n_folds, Cs.size, l1_ratios.size)


def test_l1_ratio_non_elasticnet():
    # 定义警告消息，当 penalty='l1' 时，l1_ratio 参数仅在 penalty 为 'elasticnet' 时使用
    msg = (
        r"l1_ratio parameter is only used when penalty is"
        r" 'elasticnet'. Got \(penalty=l1\)"
    )
    # 使用 pytest 的 warn 方法捕获 UserWarning，并匹配特定消息
    with pytest.warns(UserWarning, match=msg):
        # 使用 LogisticRegression 创建对象，当 penalty='l1' 时，调用 saga solver 和 l1_ratio=0.5
        LogisticRegression(penalty="l1", solver="saga", l1_ratio=0.5).fit(X, Y1)


@pytest.mark.parametrize("C", np.logspace(-3, 2, 4))
@pytest.mark.parametrize("l1_ratio", [0.1, 0.5, 0.9])
# 使用pytest.mark.parametrize装饰器，为测试函数test_elastic_net_versus_sgd参数化l1_ratio变量，分别使用0.1、0.5、0.9。
def test_elastic_net_versus_sgd(C, l1_ratio):
    # Compare elasticnet penalty in LogisticRegression() and SGD(loss='log')
    # 比较LogisticRegression()和SGD(loss='log')中的elasticnet惩罚。

    n_samples = 500
    # 创建包含500个样本的数据集
    X, y = make_classification(
        n_samples=n_samples,
        n_classes=2,
        n_features=5,
        n_informative=5,
        n_redundant=0,
        n_repeated=0,
        random_state=1,
    )
    # 生成一个分类数据集X和对应的标签y，其中包含5个特征，5个信息特征，没有冗余或重复特征。

    X = scale(X)
    # 对数据集X进行标准化处理。

    sgd = SGDClassifier(
        penalty="elasticnet",
        random_state=1,
        fit_intercept=False,
        tol=None,
        max_iter=2000,
        l1_ratio=l1_ratio,
        alpha=1.0 / C / n_samples,
        loss="log_loss",
    )
    # 创建SGDClassifier对象，使用elasticnet惩罚，配置其参数。

    log = LogisticRegression(
        penalty="elasticnet",
        random_state=1,
        fit_intercept=False,
        tol=1e-5,
        max_iter=1000,
        l1_ratio=l1_ratio,
        C=C,
        solver="saga",
    )
    # 创建LogisticRegression对象，使用elasticnet惩罚，配置其参数，使用saga求解器。

    sgd.fit(X, y)
    # 使用SGD模型拟合数据集X和标签y。

    log.fit(X, y)
    # 使用LogisticRegression模型拟合数据集X和标签y。

    assert_array_almost_equal(sgd.coef_, log.coef_, decimal=1)
    # 断言SGD模型和LogisticRegression模型的系数几乎相等，精确到小数点后一位。


def test_logistic_regression_path_coefs_multinomial():
    # Make sure that the returned coefs by logistic_regression_path when
    # multi_class='multinomial' don't override each other (used to be a
    # bug).
    # 确保当multi_class='multinomial'时，logistic_regression_path返回的系数不会互相覆盖（这曾是一个bug）。

    X, y = make_classification(
        n_samples=200,
        n_classes=3,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=0,
        n_features=2,
    )
    # 生成一个分类数据集X和对应的标签y，包含200个样本，3个类别，2个信息特征，没有冗余特征，每个类别有一个聚类中心。

    Cs = [0.00001, 1, 10000]
    # 设置正则化强度C的值列表。

    coefs, _, _ = _logistic_regression_path(
        X,
        y,
        penalty="l1",
        Cs=Cs,
        solver="saga",
        random_state=0,
        multi_class="multinomial",
    )
    # 使用_logistic_regression_path函数计算多类别分类时的系数路径，采用l1惩罚，使用saga求解器。

    with pytest.raises(AssertionError):
        assert_array_almost_equal(coefs[0], coefs[1], decimal=1)
    # 使用断言，验证第一个和第二个类别的系数几乎相等，精确到小数点后一位。
    with pytest.raises(AssertionError):
        assert_array_almost_equal(coefs[0], coefs[2], decimal=1)
    # 使用断言，验证第一个和第三个类别的系数几乎相等，精确到小数点后一位。
    with pytest.raises(AssertionError):
        assert_array_almost_equal(coefs[1], coefs[2], decimal=1)
    # 使用断言，验证第二个和第三个类别的系数几乎相等，精确到小数点后一位。


# TODO(1.7): remove filterwarnings after the deprecation of multi_class
# 在multi_class被弃用后，删除filterwarnings。
@pytest.mark.filterwarnings("ignore:.*'multi_class' was deprecated.*:FutureWarning")
@pytest.mark.parametrize(
    "est",
    [
        LogisticRegression(random_state=0, max_iter=500),
        LogisticRegressionCV(random_state=0, cv=3, Cs=3, tol=1e-3, max_iter=500),
    ],
    ids=lambda x: x.__class__.__name__,
)
@pytest.mark.parametrize("solver", SOLVERS)
# 使用pytest.mark.parametrize装饰器，为测试函数test_logistic_regression_multi_class_auto参数化est和solver变量。
def test_logistic_regression_multi_class_auto(est, solver):
    # check multi_class='auto' => multi_class='ovr'
    # iff binary y or liblinear or newton-cholesky
    # 检查当multi_class='auto'时，是否等价于multi_class='ovr'，当y为二进制或者使用liblinear或newton-cholesky求解器时。

    def fit(X, y, **kw):
        return clone(est).set_params(**kw).fit(X, y)

    # 定义一个fit函数，用于拟合数据。

    scaled_data = scale(iris.data)
    # 对鸢尾花数据集进行标准化处理。
    X = scaled_data[::10]
    X2 = scaled_data[1::10]
    y_multi = iris.target[::10]
    y_bin = y_multi == 0
    # 从标准化后的数据集中选择样本和标签，设置多类别y和二进制y。

    est_auto_bin = fit(X, y_bin, multi_class="auto", solver=solver)
    # 使用fit函数拟合多类别y和multi_class='auto'的模型，使用给定的求解器。
    est_ovr_bin = fit(X, y_bin, multi_class="ovr", solver=solver)
    # 使用fit函数拟合二进制y和multi_class='ovr'的模型，使用给定的求解器。

    assert_allclose(est_auto_bin.coef_, est_ovr_bin.coef_)
    # 断言多类别模型和二进制模型的系数几乎相等。
    # 对两个估算器的预测概率进行断言，验证它们的输出是否全部接近
    assert_allclose(est_auto_bin.predict_proba(X2), est_ovr_bin.predict_proba(X2))
    
    # 使用自动选择多类别分类方法训练估算器，并指定求解器
    est_auto_multi = fit(X, y_multi, multi_class="auto", solver=solver)
    
    # 如果求解器在"liblinear"或"newton-cholesky"中，则使用一对多方法训练估算器
    if solver in ("liblinear", "newton-cholesky"):
        est_ovr_multi = fit(X, y_multi, multi_class="ovr", solver=solver)
        # 断言自动选择的多类别估算器系数与一对多估算器系数是否全部接近
        assert_allclose(est_auto_multi.coef_, est_ovr_multi.coef_)
        # 断言自动选择的多类别估算器在给定数据上的预测概率与一对多估算器的预测概率是否全部接近
        assert_allclose(
            est_auto_multi.predict_proba(X2), est_ovr_multi.predict_proba(X2)
        )
    else:
        # 否则，使用多项式方法训练估算器
        est_multi_multi = fit(X, y_multi, multi_class="multinomial", solver=solver)
        # 断言自动选择的多类别估算器系数与多项式估算器系数是否全部接近
        assert_allclose(est_auto_multi.coef_, est_multi_multi.coef_)
        # 断言自动选择的多类别估算器在给定数据上的预测概率与多项式估算器的预测概率是否全部接近
        assert_allclose(
            est_auto_multi.predict_proba(X2), est_multi_multi.predict_proba(X2)
        )
    
        # 确保多类别参数设置为'ovr'与'multinomial'时是不同的
        assert not np.allclose(
            est_auto_bin.coef_,
            fit(X, y_bin, multi_class="multinomial", solver=solver).coef_,
        )
        assert not np.allclose(
            est_auto_bin.coef_,
            fit(X, y_multi, multi_class="multinomial", solver=solver).coef_,
        )
@pytest.mark.parametrize("solver", sorted(set(SOLVERS) - set(["liblinear"])))
# 使用参数化测试，对不包含 "liblinear" 的 SOLVERS 集合中的每个 solver 进行测试
def test_penalty_none(solver):
    # 确保当 penalty=None 且 C 被设置为非默认值时会触发警告
    # 确保设置 penalty=None 等同于设置 C=np.inf 且使用 l2 penalty
    X, y = make_classification(n_samples=1000, n_redundant=0, random_state=0)

    msg = "Setting penalty=None will ignore the C"
    lr = LogisticRegression(penalty=None, solver=solver, C=4)
    with pytest.warns(UserWarning, match=msg):
        lr.fit(X, y)

    lr_none = LogisticRegression(penalty=None, solver=solver, random_state=0)
    lr_l2_C_inf = LogisticRegression(
        penalty="l2", C=np.inf, solver=solver, random_state=0
    )
    pred_none = lr_none.fit(X, y).predict(X)
    pred_l2_C_inf = lr_l2_C_inf.fit(X, y).predict(X)
    assert_array_equal(pred_none, pred_l2_C_inf)


@pytest.mark.parametrize(
    "params",
    [
        {"penalty": "l1", "dual": False, "tol": 1e-6, "max_iter": 1000},
        {"penalty": "l2", "dual": True, "tol": 1e-12, "max_iter": 1000},
        {"penalty": "l2", "dual": False, "tol": 1e-12, "max_iter": 1000},
    ],
)
# 检查在 liblinear solver 中所有可能的情况下是否支持 sample_weight：
# l1-primal, l2-primal, l2-dual
def test_logisticregression_liblinear_sample_weight(params):
    X = np.array(
        [
            [1, 3],
            [1, 3],
            [1, 3],
            [1, 3],
            [2, 1],
            [2, 1],
            [2, 1],
            [2, 1],
            [3, 3],
            [3, 3],
            [3, 3],
            [3, 3],
            [4, 1],
            [4, 1],
            [4, 1],
            [4, 1],
        ],
        dtype=np.dtype("float"),
    )
    y = np.array(
        [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.dtype("int")
    )

    X2 = np.vstack([X, X])
    y2 = np.hstack([y, 3 - y])
    sample_weight = np.ones(shape=len(y) * 2)
    sample_weight[len(y) :] = 0
    X2, y2, sample_weight = shuffle(X2, y2, sample_weight, random_state=0)

    base_clf = LogisticRegression(solver="liblinear", random_state=42)
    base_clf.set_params(**params)
    clf_no_weight = clone(base_clf).fit(X, y)
    clf_with_weight = clone(base_clf).fit(X2, y2, sample_weight=sample_weight)

    for method in ("predict", "predict_proba", "decision_function"):
        X_clf_no_weight = getattr(clf_no_weight, method)(X)
        X_clf_with_weight = getattr(clf_with_weight, method)(X)
        assert_allclose(X_clf_no_weight, X_clf_with_weight)


def test_scores_attribute_layout_elasticnet():
    # 针对问题＃14955 的非回归测试
    # 当 penalty 是 elastic net 时，scores_ 属性的形状应为 (n_classes, n_Cs, n_l1_ratios)
    # 确保第二维对应于 Cs，第三维对应于 l1_ratios
    X, y = make_classification(n_samples=1000, random_state=0)
    cv = StratifiedKFold(n_splits=5)
    # 定义 L1 正则化的比例列表
    l1_ratios = [0.1, 0.9]
    # 定义正则化参数 C 的列表
    Cs = [0.1, 1, 10]
    
    # 创建带有交叉验证的 LogisticRegressionCV 对象
    lrcv = LogisticRegressionCV(
        penalty="elasticnet",  # 使用弹性网络作为惩罚项
        solver="saga",          # 使用 SAGA 求解器
        l1_ratios=l1_ratios,    # 指定 L1 正则化比例的列表
        Cs=Cs,                  # 指定正则化参数 C 的列表
        cv=cv,                  # 使用指定的交叉验证策略
        random_state=0,         # 随机数种子，保证可重现性
        max_iter=250,           # 最大迭代次数
        tol=1e-3,               # 收敛阈值
    )
    # 使用训练数据 X 和标签 y 训练模型
    lrcv.fit(X, y)
    
    # 计算在每个 C 和 l1_ratio 下的平均得分
    avg_scores_lrcv = lrcv.scores_[1].mean(axis=0)  # 对交叉验证的结果进行均值计算
    
    # 遍历正则化参数 C 的列表
    for i, C in enumerate(Cs):
        # 遍历 L1 正则化比例的列表
        for j, l1_ratio in enumerate(l1_ratios):
            # 创建 LogisticRegression 对象，用于交叉验证
            lr = LogisticRegression(
                penalty="elasticnet",  # 使用弹性网络作为惩罚项
                solver="saga",          # 使用 SAGA 求解器
                C=C,                    # 设置正则化参数 C
                l1_ratio=l1_ratio,      # 设置 L1 正则化比例
                random_state=0,         # 随机数种子，保证可重现性
                max_iter=250,           # 最大迭代次数
                tol=1e-3,               # 收敛阈值
            )
    
            # 计算当前参数下的交叉验证平均得分
            avg_score_lr = cross_val_score(lr, X, y, cv=cv).mean()
            
            # 断言当前参数下的交叉验证得分与 lrcv 计算的平均得分相近
            assert avg_scores_lrcv[i, j] == pytest.approx(avg_score_lr)
# 使用 pytest.mark.parametrize 装饰器为 test_multinomial_identifiability_on_iris 函数创建参数化测试
@pytest.mark.parametrize("fit_intercept", [False, True])
def test_multinomial_identifiability_on_iris(fit_intercept):
    """Test that the multinomial classification is identifiable.

    A multinomial with c classes can be modeled with
    probability_k = exp(X@coef_k) / sum(exp(X@coef_l), l=1..c) for k=1..c.
    This is not identifiable, unless one chooses a further constraint.
    According to [1], the maximum of the L2 penalized likelihood automatically
    satisfies the symmetric constraint:
    sum(coef_k, k=1..c) = 0

    Further details can be found in [2].

    Reference
    ---------
    .. [1] :doi:`Zhu, Ji and Trevor J. Hastie. "Classification of gene microarrays by
           penalized logistic regression". Biostatistics 5 3 (2004): 427-43.
           <10.1093/biostatistics/kxg046>`

    .. [2] :arxiv:`Noah Simon and Jerome Friedman and Trevor Hastie. (2013)
           "A Blockwise Descent Algorithm for Group-penalized Multiresponse and
           Multinomial Regression". <1311.6529>`
    """

    # 获取 iris 数据集的样本数量和特征数量
    n_samples, n_features = iris.data.shape
    # 获取 iris 数据集的目标类别名称
    target = iris.target_names[iris.target]

    # 使用 LogisticRegression 进行分类器的初始化设定
    clf = LogisticRegression(
        C=len(iris.data),  # 正则化强度参数设定为数据集的样本数量
        solver="lbfgs",    # 指定求解器为 L-BFGS
        fit_intercept=fit_intercept,  # 是否拟合截距项，根据参数化传入的值确定
    )

    # 对输入数据进行标准化以提高收敛速度
    X_scaled = scale(iris.data)
    # 使用标准化后的数据拟合分类器
    clf.fit(X_scaled, target)

    # axis=0 表示在类别上求和
    # 检查分类器的系数是否满足约束条件：对所有类别的系数求和应该接近于 0
    assert_allclose(clf.coef_.sum(axis=0), 0, atol=1e-10)
    # 如果拟合包括截距项，则检查截距项的总和是否接近于 0
    if fit_intercept:
        clf.intercept_.sum(axis=0) == pytest.approx(0, abs=1e-15)


# TODO(1.7): remove filterwarnings after the deprecation of multi_class
# 使用 pytest.mark.filterwarnings 装饰器忽略关于 'multi_class' 参数被弃用的警告
@pytest.mark.filterwarnings("ignore:.*'multi_class' was deprecated.*:FutureWarning")
# 使用 pytest.mark.parametrize 装饰器为 test_sample_weight_not_modified 函数创建参数化测试
@pytest.mark.parametrize("multi_class", ["ovr", "multinomial", "auto"])
# 使用 pytest.mark.parametrize 装饰器为 test_sample_weight_not_modified 函数创建参数化测试
@pytest.mark.parametrize("class_weight", [{0: 1.0, 1: 10.0, 2: 1.0}, "balanced"])
def test_sample_weight_not_modified(multi_class, class_weight):
    # 载入 iris 数据集并返回特征矩阵 X 和目标向量 y
    X, y = load_iris(return_X_y=True)
    # 计算特征数量
    n_features = len(X)
    # 创建样本权重向量 W，初始化为全1
    W = np.ones(n_features)
    # 将权重向量的前一半设置为2
    W[: n_features // 2] = 2

    # 复制期望的权重向量
    expected = W.copy()

    # 使用 LogisticRegression 进行分类器的初始化设定
    clf = LogisticRegression(
        random_state=0,  # 设定随机数种子
        class_weight=class_weight,  # 设定类别权重
        max_iter=200,  # 设定最大迭代次数
        multi_class=multi_class  # 设定多分类策略
    )
    # 使用给定的样本权重向量拟合分类器
    clf.fit(X, y, sample_weight=W)
    # 检查拟合后的样本权重向量是否与期望的权重向量一致
    assert_allclose(expected, W)


# 使用 pytest.mark.parametrize 装饰器为 test_large_sparse_matrix 函数创建参数化测试
@pytest.mark.parametrize("solver", SOLVERS)
# 使用 pytest.mark.parametrize 装饰器为 test_large_sparse_matrix 函数创建参数化测试
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_large_sparse_matrix(solver, global_random_seed, csr_container):
    # Solvers either accept large sparse matrices, or raise helpful error.
    # Non-regression test for pull-request #21093.

    # 生成稀疏矩阵，使用 int64 类型的索引
    X = csr_container(sparse.rand(20, 10, random_state=global_random_seed))
    for attr in ["indices", "indptr"]:
        setattr(X, attr, getattr(X, attr).astype("int64"))
    # 使用全局随机种子生成目标向量 y
    rng = np.random.RandomState(global_random_seed)
    y = rng.randint(2, size=X.shape[0])
    # 如果 solver 参数在列表 ["liblinear", "sag", "saga"] 中
    if solver in ["liblinear", "sag", "saga"]:
        # 设置错误消息，用于匹配 pytest.raises 抛出的 ValueError 异常
        msg = "Only sparse matrices with 32-bit integer indices"
        # 使用 pytest 模块检查是否会抛出 ValueError 异常，并验证错误消息是否匹配
        with pytest.raises(ValueError, match=msg):
            # 尝试使用 LogisticRegression 模型拟合数据 X, y，使用指定的 solver
            LogisticRegression(solver=solver).fit(X, y)
    else:
        # 如果 solver 参数不在列表中，则直接使用 LogisticRegression 模型拟合数据 X, y
        LogisticRegression(solver=solver).fit(X, y)
def test_single_feature_newton_cg():
    # 测试单个特征和截距情况下，Newton-CG 是否正常工作。
    # 针对问题 #23605 的非回归测试。

    # 创建包含单个特征的特征矩阵 X 和对应的标签 y
    X = np.array([[0.5, 0.65, 1.1, 1.25, 0.8, 0.54, 0.95, 0.7]]).T
    y = np.array([1, 1, 0, 0, 1, 1, 0, 1])
    # 断言特征矩阵 X 的列数为 1
    assert X.shape[1] == 1
    # 使用 LogisticRegression 模型，solver 设为 "newton-cg"，fit_intercept 设为 True 进行拟合
    LogisticRegression(solver="newton-cg", fit_intercept=True).fit(X, y)


def test_liblinear_not_stuck():
    # 非回归测试，解决 https://github.com/scikit-learn/scikit-learn/issues/18264
    # 复制 iris 数据集，进行预处理以移除类别为 2 的样本
    X = iris.data.copy()
    y = iris.target.copy()
    X = X[y != 2]
    y = y[y != 2]
    # 使用 StandardScaler 进行数据标准化处理
    X_prep = StandardScaler().fit_transform(X)

    # 计算 C 值，并设置 LogisticRegression 模型的参数
    C = l1_min_c(X, y, loss="log") * 10 ** (10 / 29)
    clf = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        tol=1e-6,
        max_iter=100,
        intercept_scaling=10000.0,
        random_state=0,
        C=C,
    )

    # 测试拟合过程不会引发 ConvergenceWarning
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        clf.fit(X_prep, y)


@pytest.mark.usefixtures("enable_slep006")
def test_lr_cv_scores_differ_when_sample_weight_is_requested():
    """测试当请求 sample_weight 时，确保 LogisticRegressionCV.fit 和 LogisticRegressionCV.score 正确传递给评分器。
    通过比较请求了 sample_weight 和未请求 sample_weight 时的得分差异来验证。
    """
    rng = np.random.RandomState(10)
    X, y = make_classification(n_samples=10, random_state=rng)
    X_t, y_t = make_classification(n_samples=10, random_state=rng)
    sample_weight = np.ones(len(y))
    sample_weight[: len(y) // 2] = 2
    kwargs = {"sample_weight": sample_weight}

    scorer1 = get_scorer("accuracy")
    lr_cv1 = LogisticRegressionCV(scoring=scorer1)
    lr_cv1.fit(X, y, **kwargs)

    scorer2 = get_scorer("accuracy")
    scorer2.set_score_request(sample_weight=True)
    lr_cv2 = LogisticRegressionCV(scoring=scorer2)
    lr_cv2.fit(X, y, **kwargs)

    # 断言两种情况下得分不全相等
    assert not np.allclose(lr_cv1.scores_[1], lr_cv2.scores_[1])

    # 测试使用两种情况下的得分
    score_1 = lr_cv1.score(X_t, y_t, **kwargs)
    score_2 = lr_cv2.score(X_t, y_t, **kwargs)

    # 断言两种情况下得分不全相等
    assert not np.allclose(score_1, score_2)


def test_lr_cv_scores_without_enabling_metadata_routing():
    """测试即使在 enable_metadata_routing=False 的情况下，sample_weight 也能正确传递给 LogisticRegressionCV.fit 和 LogisticRegressionCV.score。
    """
    rng = np.random.RandomState(10)
    X, y = make_classification(n_samples=10, random_state=rng)
    X_t, y_t = make_classification(n_samples=10, random_state=rng)
    sample_weight = np.ones(len(y))
    sample_weight[: len(y) // 2] = 2
    kwargs = {"sample_weight": sample_weight}

    # 使用 enable_metadata_routing=False 的上下文环境
    with config_context(enable_metadata_routing=False):
        scorer1 = get_scorer("accuracy")
        lr_cv1 = LogisticRegressionCV(scoring=scorer1)
        lr_cv1.fit(X, y, **kwargs)
        # 获取得分，验证是否正常工作
        score_1 = lr_cv1.score(X_t, y_t, **kwargs)
    # 在启用元数据路由的配置上下文中执行以下操作
    with config_context(enable_metadata_routing=True):
        # 获取一个针对"accuracy"指标的评分器对象
        scorer2 = get_scorer("accuracy")
        # 设置评分请求，使用样本权重
        scorer2.set_score_request(sample_weight=True)
        # 使用交叉验证的逻辑回归模型，使用指定的评分器进行评分
        lr_cv2 = LogisticRegressionCV(scoring=scorer2)
        # 使用给定的数据 X 和标签 y 训练逻辑回归模型，传递任何额外的关键字参数
        lr_cv2.fit(X, y, **kwargs)
        # 使用测试数据集 X_t 和 y_t 对模型进行评分，传递任何额外的关键字参数
        score_2 = lr_cv2.score(X_t, y_t, **kwargs)

    # 断言验证第一个逻辑回归模型和第二个逻辑回归模型的第一个分数是否接近
    assert_allclose(lr_cv1.scores_[1], lr_cv2.scores_[1])
    # 断言验证第一个逻辑回归模型的分数与第二个逻辑回归模型的分数是否接近
    assert_allclose(score_1, score_2)
# 使用 pytest 的 parametrize 装饰器，参数化测试函数 test_zero_max_iter，对 SOLVERS 列表中的每个 solver 执行测试
@pytest.mark.parametrize("solver", SOLVERS)
def test_zero_max_iter(solver):
    # 确保在 LogisticRegression 初始化后（在第一次权重更新之前）可以检查其状态
    X, y = load_iris(return_X_y=True)
    y = y == 2
    # 忽略 ConvergenceWarning 类别的警告
    with ignore_warnings(category=ConvergenceWarning):
        # 使用 solver 和 max_iter=0 初始化 LogisticRegression，并拟合数据 X, y
        clf = LogisticRegression(solver=solver, max_iter=0).fit(X, y)
    if solver not in ["saga", "sag"]:
        # 对于非 "saga" 和 "sag" 的 solver，验证 clf 的 n_iter_ 属性是否为 0
        assert clf.n_iter_ == 0

    if solver != "lbfgs":
        # 对于非 "lbfgs" 的 solver，验证 clf 的 coef_ 属性是否为全零数组
        assert_allclose(clf.coef_, np.zeros_like(clf.coef_))
        # 验证 clf 的 decision_function(X) 是否等于 clf.intercept_ 的全值填充数组
        assert_allclose(
            clf.decision_function(X),
            np.full(shape=X.shape[0], fill_value=clf.intercept_),
        )
        # 验证 clf 的 predict_proba(X) 是否为形状为 (X.shape[0], 2) 的全 0.5 填充数组
        assert_allclose(
            clf.predict_proba(X),
            np.full(shape=(X.shape[0], 2), fill_value=0.5),
        )
    # 验证 clf 在数据集 X, y 上的得分是否小于 0.7
    assert clf.score(X, y) < 0.7


def test_passing_params_without_enabling_metadata_routing():
    """Test that the right error message is raised when metadata params
    are passed while not supported when `enable_metadata_routing=False`."""
    # 创建一个数据集 X, y
    X, y = make_classification(n_samples=10, random_state=0)
    # 初始化 LogisticRegressionCV 对象
    lr_cv = LogisticRegressionCV()
    # 预期的错误消息
    msg = "is only supported if enable_metadata_routing=True"

    # 在 enable_metadata_routing=False 的上下文中
    with config_context(enable_metadata_routing=False):
        params = {"extra_param": 1.0}

        # 验证在调用 lr_cv.fit(X, y, **params) 时是否会引发 ValueError，且错误消息匹配 msg
        with pytest.raises(ValueError, match=msg):
            lr_cv.fit(X, y, **params)

        # 验证在调用 lr_cv.score(X, y, **params) 时是否会引发 ValueError，且错误消息匹配 msg
        with pytest.raises(ValueError, match=msg):
            lr_cv.score(X, y, **params)


# TODO(1.7): remove
def test_multi_class_deprecated():
    """Check `multi_class` parameter deprecated."""
    # 创建多类别分类数据集 X, y
    X, y = make_classification(n_classes=3, n_samples=50, n_informative=6)
    # 初始化 LogisticRegression 对象，使用 multi_class="ovr"，并测试 FutureWarning 警告信息
    lr = LogisticRegression(multi_class="ovr")
    msg = "'multi_class' was deprecated"
    with pytest.warns(FutureWarning, match=msg):
        lr.fit(X, y)

    # 初始化 LogisticRegressionCV 对象，使用 multi_class="ovr"，并测试 FutureWarning 警告信息
    lrCV = LogisticRegressionCV(multi_class="ovr")
    with pytest.warns(FutureWarning, match=msg):
        lrCV.fit(X, y)

    # 对于二分类数据集 X, y
    X, y = make_classification(n_classes=2, n_samples=50, n_informative=6)
    # 初始化 LogisticRegression 对象，使用 multi_class="multinomial"，并测试 FutureWarning 警告信息
    lr = LogisticRegression(multi_class="multinomial")
    msg = "'multi_class' was deprecated.*binary problems"
    with pytest.warns(FutureWarning, match=msg):
        lr.fit(X, y)

    # 初始化 LogisticRegressionCV 对象，使用 multi_class="multinomial"，并测试 FutureWarning 警告信息
    lrCV = LogisticRegressionCV(multi_class="multinomial")
    with pytest.warns(FutureWarning, match=msg):
        lrCV.fit(X, y)
```