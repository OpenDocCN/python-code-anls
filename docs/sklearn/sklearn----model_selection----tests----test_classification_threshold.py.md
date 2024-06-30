# `D:\src\scipysrc\scikit-learn\sklearn\model_selection\tests\test_classification_threshold.py`

```
# 导入必要的库和模块
import numpy as np  # 导入 NumPy 库并使用别名 np
import pytest  # 导入 pytest 库

from sklearn.base import BaseEstimator, ClassifierMixin, clone  # 导入基类和分类器相关的类和函数
from sklearn.datasets import (  # 导入数据集生成和加载相关的函数和类
    load_breast_cancer,
    load_iris,
    make_classification,
    make_multilabel_classification,
)
from sklearn.dummy import DummyClassifier  # 导入虚拟分类器
from sklearn.ensemble import GradientBoostingClassifier  # 导入梯度提升分类器
from sklearn.exceptions import NotFittedError  # 导入未拟合错误类
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归模型
from sklearn.metrics import (  # 导入评估指标相关函数
    balanced_accuracy_score,
    f1_score,
    fbeta_score,
    make_scorer,
    recall_score,
)
from sklearn.model_selection import (  # 导入模型选择相关函数和类
    FixedThresholdClassifier,
    StratifiedShuffleSplit,
    TunedThresholdClassifierCV,
)
from sklearn.model_selection._classification_threshold import (  # 导入分类阈值相关函数和类
    _CurveScorer,
    _fit_and_score_over_thresholds,
)
from sklearn.pipeline import make_pipeline  # 导入创建管道的函数
from sklearn.preprocessing import StandardScaler  # 导入标准化处理器
from sklearn.svm import SVC  # 导入支持向量机模型
from sklearn.tree import DecisionTreeClassifier  # 导入决策树分类器
from sklearn.utils._mocking import CheckingClassifier  # 导入检查分类器
from sklearn.utils._testing import (  # 导入测试工具函数
    _convert_container,
    assert_allclose,
    assert_array_equal,
)


def test_curve_scorer():
    """Check the behaviour of the `_CurveScorer` class."""
    # 生成一个分类数据集 X 和标签 y
    X, y = make_classification(random_state=0)
    # 使用逻辑回归模型拟合数据集 X 和标签 y
    estimator = LogisticRegression().fit(X, y)
    # 创建 _CurveScorer 类实例，用于评估曲线分数
    curve_scorer = _CurveScorer(
        balanced_accuracy_score,  # 使用平衡精度评估指标
        sign=1,  # 分数的符号为正
        response_method="predict_proba",  # 使用预测概率响应方法
        thresholds=10,  # 使用10个阈值
        kwargs={},  # 额外参数为空字典
    )
    # 调用 curve_scorer 对象来评估估算器在数据集 X 和 y 上的分数和阈值
    scores, thresholds = curve_scorer(estimator, X, y)

    # 断言阈值数组的形状与分数数组相同
    assert thresholds.shape == scores.shape
    # 检查阈值是否为接近0和1的概率值
    assert 0 <= thresholds.min() <= 0.01
    assert 0.99 <= thresholds.max() <= 1
    # 平衡精度分数应在0.5到1之间（未调整时）
    assert 0.5 <= scores.min() <= 1

    # 检查传递给评分器的 kwargs 是否起作用
    curve_scorer = _CurveScorer(
        balanced_accuracy_score,  # 使用平衡精度评估指标
        sign=1,  # 分数的符号为正
        response_method="predict_proba",  # 使用预测概率响应方法
        thresholds=10,  # 使用10个阈值
        kwargs={"adjusted": True},  # 使用调整后的参数
    )
    # 再次调用 curve_scorer 对象来评估估算器在数据集 X 和 y 上的分数和阈值
    scores, thresholds = curve_scorer(estimator, X, y)

    # 平衡精度分数应在0到0.5之间（调整后）
    assert 0 <= scores.min() <= 0.5

    # 检查在处理 'neg_*' 分数时是否可以反转分数的符号
    curve_scorer = _CurveScorer(
        balanced_accuracy_score,  # 使用平衡精度评估指标
        sign=-1,  # 分数的符号为负
        response_method="predict_proba",  # 使用预测概率响应方法
        thresholds=10,  # 使用10个阈值
        kwargs={"adjusted": True},  # 使用调整后的参数
    )
    # 再次调用 curve_scorer 对象来评估估算器在数据集 X 和 y 上的分数和阈值
    scores, thresholds = curve_scorer(estimator, X, y)

    # 断言所有分数都小于等于0
    assert all(scores <= 0)


def test_curve_scorer_pos_label(global_random_seed):
    """Check that we propagate properly the `pos_label` parameter to the scorer."""
    n_samples = 30
    # 使用 `make_classification` 函数生成分类数据集 `X` 和目标变量 `y`
    X, y = make_classification(
        n_samples=n_samples, weights=[0.9, 0.1], random_state=global_random_seed
    )

    # 使用 LogisticRegression 模型拟合数据集
    estimator = LogisticRegression().fit(X, y)

    # 创建 `_CurveScorer` 实例，用于计算召回率曲线得分
    curve_scorer = _CurveScorer(
        recall_score,
        sign=1,
        response_method="predict_proba",
        thresholds=10,
        kwargs={"pos_label": 1},
    )
    # 计算 `pos_label=1` 的召回率曲线得分及对应的阈值
    scores_pos_label_1, thresholds_pos_label_1 = curve_scorer(estimator, X, y)

    # 创建 `_CurveScorer` 实例，用于计算召回率曲线得分，`pos_label` 为 0
    curve_scorer = _CurveScorer(
        recall_score,
        sign=1,
        response_method="predict_proba",
        thresholds=10,
        kwargs={"pos_label": 0},
    )
    # 计算 `pos_label=0` 的召回率曲线得分及对应的阈值
    scores_pos_label_0, thresholds_pos_label_0 = curve_scorer(estimator, X, y)

    # 断言 `pos_label=1` 和 `pos_label=0` 的阈值数组不全等
    assert not (thresholds_pos_label_1 == thresholds_pos_label_0).all()

    # 断言 `pos_label=0` 和 `pos_label=1` 的阈值范围分别近似于 `predict_proba` 函数返回的概率列的最小和最大值
    y_pred = estimator.predict_proba(X)
    assert thresholds_pos_label_0.min() == pytest.approx(y_pred.min(axis=0)[0])
    assert thresholds_pos_label_0.max() == pytest.approx(y_pred.max(axis=0)[0])
    assert thresholds_pos_label_1.min() == pytest.approx(y_pred.min(axis=0)[1])
    assert thresholds_pos_label_1.max() == pytest.approx(y_pred.max(axis=0)[1])

    # 断言 `pos_label=0` 的召回率得分范围在 (0.0, 1.0) 之间，并且应小于 `pos_label=1` 的对应得分范围
    assert 0.0 < scores_pos_label_0.min() < scores_pos_label_1.min()
    assert scores_pos_label_0.max() == pytest.approx(1.0)
    assert scores_pos_label_1.max() == pytest.approx(1.0)
# 定义测试函数，用于验证 `_fit_and_score_over_thresholds` 对不同的曲线评分器返回按升序排列的阈值
def test_fit_and_score_over_thresholds_curve_scorers():
    """Check that `_fit_and_score_over_thresholds` returns thresholds in ascending order
    for the different accepted curve scorers."""
    
    # 生成一个具有100个样本的分类数据集
    X, y = make_classification(n_samples=100, random_state=0)
    
    # 定义训练集索引和验证集索引
    train_idx, val_idx = np.arange(50), np.arange(50, 100)
    
    # 创建逻辑回归分类器对象
    classifier = LogisticRegression()

    # 创建曲线评分器 `_CurveScorer` 对象，使用平衡精度评分函数，返回预测概率，设定10个阈值
    curve_scorer = _CurveScorer(
        score_func=balanced_accuracy_score,
        sign=1,
        response_method="predict_proba",
        thresholds=10,
        kwargs={},
    )
    
    # 调用 `_fit_and_score_over_thresholds` 函数，计算分数和阈值
    scores, thresholds = _fit_and_score_over_thresholds(
        classifier,
        X,
        y,
        fit_params={},
        train_idx=train_idx,
        val_idx=val_idx,
        curve_scorer=curve_scorer,
        score_params={},
    )

    # 断言确保阈值按升序排列
    assert np.all(thresholds[:-1] <= thresholds[1:])
    # 断言确保分数是 numpy 数组
    assert isinstance(scores, np.ndarray)
    # 断言确保所有分数在 [0, 1] 范围内
    assert np.logical_and(scores >= 0, scores <= 1).all()


# 定义测试函数，验证预训练分类器的行为
def test_fit_and_score_over_thresholds_prefit():
    """Check the behaviour with a prefit classifier."""
    
    # 生成一个具有100个样本的分类数据集
    X, y = make_classification(n_samples=100, random_state=0)

    # 对决策树分类器进行预训练，并设定训练集索引为 None，表示分类器已预训练
    train_idx, val_idx = None, np.arange(50, 100)
    classifier = DecisionTreeClassifier(random_state=0).fit(X, y)
    
    # 断言确保分类器完全记忆了整个数据集，验证集上的预测分数应接近1.0
    assert classifier.score(X[val_idx], y[val_idx]) == pytest.approx(1.0)

    # 创建曲线评分器 `_CurveScorer` 对象，使用平衡精度评分函数，返回预测概率，设定2个阈值
    curve_scorer = _CurveScorer(
        score_func=balanced_accuracy_score,
        sign=1,
        response_method="predict_proba",
        thresholds=2,
        kwargs={},
    )
    
    # 调用 `_fit_and_score_over_thresholds` 函数，计算分数和阈值
    scores, thresholds = _fit_and_score_over_thresholds(
        classifier,
        X,
        y,
        fit_params={},
        train_idx=train_idx,
        val_idx=val_idx,
        curve_scorer=curve_scorer,
        score_params={},
    )
    
    # 断言确保阈值按升序排列
    assert np.all(thresholds[:-1] <= thresholds[1:])
    # 断言确保分数与预期值接近
    assert_allclose(scores, [0.5, 1.0])


# 使用 `pytest.mark.usefixtures` 标记的测试函数，检查样本权重是否正确分发到分类器的拟合和评分过程中
@pytest.mark.usefixtures("enable_slep006")
def test_fit_and_score_over_thresholds_sample_weight():
    """Check that we dispatch the sample-weight to fit and score the classifier."""
    
    # 加载鸢尾花数据集，返回特征矩阵 X 和目标向量 y
    X, y = load_iris(return_X_y=True)
    X, y = X[:100], y[:100]  # 只保留两个类别的数据集

    # 创建一个数据集，将类别 #0 的样本重复两次
    X_repeated, y_repeated = np.vstack([X, X[y == 0]]), np.hstack([y, y[y == 0]])
    
    # 创建一个与重复数据集等价的样本权重向量
    sample_weight = np.ones_like(y)
    sample_weight[:50] *= 2

    # 创建逻辑回归分类器对象
    classifier = LogisticRegression()
    
    # 定义重复数据集的训练集索引和验证集索引
    train_repeated_idx = np.arange(X_repeated.shape[0])
    val_repeated_idx = np.arange(X_repeated.shape[0])
    
    # 创建曲线评分器 `_CurveScorer` 对象，使用平衡精度评分函数，返回预测概率，设定10个阈值
    curve_scorer = _CurveScorer(
        score_func=balanced_accuracy_score,
        sign=1,
        response_method="predict_proba",
        thresholds=10,
        kwargs={},
    )
    # 使用自定义函数 `_fit_and_score_over_thresholds` 对分类器进行拟合和评分，并返回得分和阈值
    scores_repeated, thresholds_repeated = _fit_and_score_over_thresholds(
        classifier,  # 要拟合和评分的分类器对象
        X_repeated,  # 重复数据集的特征向量
        y_repeated,  # 重复数据集的目标值
        fit_params={},  # 拟合参数，这里为空字典
        train_idx=train_repeated_idx,  # 用于训练的索引集合（重复数据）
        val_idx=val_repeated_idx,  # 用于验证的索引集合（重复数据）
        curve_scorer=curve_scorer,  # 曲线评分器对象，用于计算评分曲线
        score_params={},  # 评分参数，这里为空字典
    )
    
    # 创建全局训练索引和验证索引，索引范围是数据集 X 的行数
    train_idx, val_idx = np.arange(X.shape[0]), np.arange(X.shape[0])
    
    # 使用 `_fit_and_score_over_thresholds` 对分类器进行拟合和评分，并返回得分和阈值
    scores, thresholds = _fit_and_score_over_thresholds(
        classifier.set_fit_request(sample_weight=True),  # 设置了样本权重的分类器对象
        X,  # 数据集的特征向量
        y,  # 数据集的目标值
        fit_params={"sample_weight": sample_weight},  # 拟合参数，包括样本权重
        train_idx=train_idx,  # 用于训练的索引集合
        val_idx=val_idx,  # 用于验证的索引集合
        curve_scorer=curve_scorer.set_score_request(sample_weight=True),  # 设置了样本权重的评分曲线评分器对象
        score_params={"sample_weight": sample_weight},  # 评分参数，包括样本权重
    )
    
    # 断言两次评分操作得到的阈值和得分数组应该非常接近，否则会触发 AssertionError
    assert_allclose(thresholds_repeated, thresholds)
    assert_allclose(scores_repeated, scores)
# 使用 pytest.mark.usefixtures 注释来声明使用 enable_slep006 fixture
@pytest.mark.usefixtures("enable_slep006")
# 使用 pytest.mark.parametrize 注释来定义 test_fit_and_score_over_thresholds_fit_params 函数的参数化测试，测试参数为 fit_params_type，取值为 "list" 和 "array"
@pytest.mark.parametrize("fit_params_type", ["list", "array"])
def test_fit_and_score_over_thresholds_fit_params(fit_params_type):
    """Check that we pass `fit_params` to the classifier when calling `fit`."""
    # 生成用于测试的随机分类数据 X 和 y
    X, y = make_classification(n_samples=100, random_state=0)
    # 创建 fit_params 字典，包含两个键值对，键分别为 "a" 和 "b"，值通过 _convert_container 函数转换为 fit_params_type 类型的 y
    fit_params = {
        "a": _convert_container(y, fit_params_type),
        "b": _convert_container(y, fit_params_type),
    }

    # 创建 CheckingClassifier 实例，设置期望的 fit 参数为 ["a", "b"]，随机状态为 0
    classifier = CheckingClassifier(expected_fit_params=["a", "b"], random_state=0)
    # 调用 classifier 的 set_fit_request 方法，设置参数 a=True, b=True
    classifier.set_fit_request(a=True, b=True)
    # 创建训练集索引 train_idx 和验证集索引 val_idx，均为 0 到 49 和 50 到 99 的数组
    train_idx, val_idx = np.arange(50), np.arange(50, 100)

    # 创建 _CurveScorer 实例，使用 balanced_accuracy_score 作为评分函数，sign 设为 1，response_method 设为 "predict_proba"，thresholds 设为 10，kwargs 设为空字典
    curve_scorer = _CurveScorer(
        score_func=balanced_accuracy_score,
        sign=1,
        response_method="predict_proba",
        thresholds=10,
        kwargs={},
    )
    # 调用 _fit_and_score_over_thresholds 函数，传入 classifier、X、y、fit_params、train_idx、val_idx、curve_scorer 和 score_params 参数
    _fit_and_score_over_thresholds(
        classifier,
        X,
        y,
        fit_params=fit_params,
        train_idx=train_idx,
        val_idx=val_idx,
        curve_scorer=curve_scorer,
        score_params={},
    )


# 使用 pytest.mark.parametrize 注释来定义 test_tuned_threshold_classifier_no_binary 函数的参数化测试，测试参数为 data，包含两个数据集：make_classification 和 make_multilabel_classification 生成的数据
@pytest.mark.parametrize(
    "data",
    [
        make_classification(n_classes=3, n_clusters_per_class=1, random_state=0),
        make_multilabel_classification(random_state=0),
    ],
)
def test_tuned_threshold_classifier_no_binary(data):
    """Check that we raise an informative error message for non-binary problem."""
    # 定义错误信息 err_msg，指明只支持二元分类
    err_msg = "Only binary classification is supported."
    # 使用 pytest.raises 断言，期望捕获 ValueError 异常，且异常消息匹配 err_msg
    with pytest.raises(ValueError, match=err_msg):
        # 创建 TunedThresholdClassifierCV 实例，使用 LogisticRegression 作为基础估计器，调用 fit 方法传入 data 参数
        TunedThresholdClassifierCV(LogisticRegression()).fit(*data)


# 使用 pytest.mark.parametrize 注释来定义 test_tuned_threshold_classifier_conflict_cv_refit 函数的参数化测试，测试参数为 params, err_type, err_msg，包含三个测试参数组合
@pytest.mark.parametrize(
    "params, err_type, err_msg",
    [
        (
            {"cv": "prefit", "refit": True},
            ValueError,
            "When cv='prefit', refit cannot be True.",
        ),
        (
            {"cv": 10, "refit": False},
            ValueError,
            "When cv has several folds, refit cannot be False.",
        ),
        (
            {"cv": "prefit", "refit": False},
            NotFittedError,
            "`estimator` must be fitted.",
        ),
    ],
)
def test_tuned_threshold_classifier_conflict_cv_refit(params, err_type, err_msg):
    """Check that we raise an informative error message when `cv` and `refit`
    cannot be used together.
    """
    # 生成用于测试的随机分类数据 X 和 y
    X, y = make_classification(n_samples=100, random_state=0)
    # 使用 pytest.raises 断言，期望捕获 err_type 类型的异常，且异常消息匹配 err_msg
    with pytest.raises(err_type, match=err_msg):
        # 创建 TunedThresholdClassifierCV 实例，使用 LogisticRegression 作为基础估计器，传入 params 参数，调用 fit 方法传入 X 和 y 参数
        TunedThresholdClassifierCV(LogisticRegression(), **params).fit(X, y)


# 使用 pytest.mark.parametrize 注释来定义 test_threshold_classifier_estimator_response_methods 函数的参数化测试，测试参数为 ThresholdClassifier, estimator, response_method，分别测试三个分类器和三种响应方法的组合
@pytest.mark.parametrize(
    "estimator",
    [LogisticRegression(), SVC(), GradientBoostingClassifier(n_estimators=4)],
)
@pytest.mark.parametrize(
    "response_method", ["predict_proba", "predict_log_proba", "decision_function"]
)
@pytest.mark.parametrize(
    "ThresholdClassifier", [FixedThresholdClassifier, TunedThresholdClassifierCV]
)
def test_threshold_classifier_estimator_response_methods(
    ThresholdClassifier, estimator, response_method
):
    """Check that `TunedThresholdClassifierCV` exposes the same response methods as the
    chosen estimator.
    """
    X, y = make_classification(n_samples=100, random_state=0)

# 生成一个随机的分类数据集，包括100个样本。X 是特征数据，y 是对应的标签。


    model = ThresholdClassifier(estimator=estimator)

# 使用 ThresholdClassifier 类创建一个模型对象 model，该类的 estimator 参数是传入的外部参数 estimator。


    assert hasattr(model, response_method) == hasattr(estimator, response_method)

# 断言模型对象 model 是否具有与外部参数 estimator 相同的 response_method 方法。


    model.fit(X, y)

# 使用生成的数据集 X 和 y 对模型进行训练。


    assert hasattr(model, response_method) == hasattr(estimator, response_method)

# 再次断言模型对象 model 是否具有与外部参数 estimator 相同的 response_method 方法。


    if hasattr(model, response_method):

# 如果模型对象 model 具有指定的 response_method 方法，则执行以下代码块。


        y_pred_cutoff = getattr(model, response_method)(X)

# 使用模型对象 model 的 response_method 方法对数据集 X 进行预测，得到预测结果 y_pred_cutoff。


        y_pred_underlying_estimator = getattr(model.estimator_, response_method)(X)

# 使用模型对象 model 的 estimator_ 属性（该属性是通过 ThresholdClassifier 类初始化时传入的 estimator 参数）的 response_method 方法进行预测，得到预测结果 y_pred_underlying_estimator。


        assert_allclose(y_pred_cutoff, y_pred_underlying_estimator)

# 断言两种预测结果 y_pred_cutoff 和 y_pred_underlying_estimator 的接近程度，使用 assert_allclose 函数。
@pytest.mark.parametrize(
    "response_method", ["auto", "decision_function", "predict_proba"]
)
# 使用pytest的parametrize装饰器，为response_method参数提供三个不同的测试值

@pytest.mark.parametrize(
    "metric",
    [
        make_scorer(balanced_accuracy_score),
        make_scorer(f1_score, pos_label="cancer"),
    ],
)
# 使用pytest的parametrize装饰器，为metric参数提供两个不同的测试值：平衡准确率和带有指定正例标签的F1分数计算器

def test_tuned_threshold_classifier_with_string_targets(response_method, metric):
    """Check that targets represented by str are properly managed.
    Also, check with several metrics to be sure that `pos_label` is properly
    dispatched.
    """
    # 加载乳腺癌数据集
    X, y = load_breast_cancer(return_X_y=True)
    
    # 对数据进行标准化并训练逻辑回归模型
    lr = make_pipeline(StandardScaler(), LogisticRegression()).fit(X, y)
    
    # 使用TunedThresholdClassifierCV拟合模型，使用给定的metric作为评分器
    model_fbeta_1 = TunedThresholdClassifierCV(
        estimator=lr, scoring=make_scorer(fbeta_score, beta=1)
    ).fit(X, y)
    
    # 使用不同的beta值拟合模型
    model_fbeta_2 = TunedThresholdClassifierCV(
        estimator=lr, scoring=make_scorer(fbeta_score, beta=2)
    ).fit(X, y)
    
    # 使用默认参数拟合模型，使用F1分数作为评分器
    model_f1 = TunedThresholdClassifierCV(
        estimator=lr, scoring=make_scorer(f1_score)
    ).fit(X, y)

    # 断言模型使用beta=1时的最佳阈值近似等于使用F1分数时的最佳阈值
    assert model_fbeta_1.best_threshold_ == pytest.approx(model_f1.best_threshold_)
    
    # 断言使用beta=1和beta=2时的最佳阈值不同
    assert model_fbeta_1.best_threshold_ != pytest.approx(model_fbeta_2.best_threshold_)
    # 创建一个包含类别标签的 NumPy 数组，类型为 object
    classes = np.array(["cancer", "healthy"], dtype=object)
    # 将 y 中的整数索引映射为相应的类别标签
    y = classes[y]
    # 使用 TunedThresholdClassifierCV 模型，通过交叉验证来调整阈值
    model = TunedThresholdClassifierCV(
        # 使用管道，包括数据标准化和逻辑回归模型
        estimator=make_pipeline(StandardScaler(), LogisticRegression()),
        # 指定评分指标
        scoring=metric,
        # 指定响应方法
        response_method=response_method,
        # 设定阈值数量
        thresholds=100,
    ).fit(X, y)  # 将模型与数据 X 和标签 y 拟合
    # 确保模型学习的类别顺序与给定的 classes 数组排序后一致
    assert_array_equal(model.classes_, np.sort(classes))
    # 对数据 X 进行预测
    y_pred = model.predict(X)
    # 确保预测的类别与给定的 classes 数组排序后一致
    assert_array_equal(np.unique(y_pred), np.sort(classes))
# 标记此测试函数使用 `enable_slep006` fixture
@pytest.mark.usefixtures("enable_slep006")
# 参数化测试函数，测试 `with_sample_weight` 参数为 True 和 False 两种情况
@pytest.mark.parametrize("with_sample_weight", [True, False])
# 定义测试函数，测试调整阈值分类器在 refit 参数下的行为
def test_tuned_threshold_classifier_refit(with_sample_weight, global_random_seed):
    """Check the behaviour of the `refit` parameter."""
    # 使用全局随机种子创建随机数生成器
    rng = np.random.RandomState(global_random_seed)
    # 生成分类数据 X 和 y
    X, y = make_classification(n_samples=100, random_state=0)
    
    # 根据 with_sample_weight 参数决定是否生成样本权重
    if with_sample_weight:
        sample_weight = rng.randn(X.shape[0])
        sample_weight = np.abs(sample_weight, out=sample_weight)
    else:
        sample_weight = None

    # 检查当 refit=True 时，`estimator_` 是否在完整数据集上进行拟合
    estimator = LogisticRegression().set_fit_request(sample_weight=True)
    model = TunedThresholdClassifierCV(estimator, refit=True).fit(
        X, y, sample_weight=sample_weight
    )

    assert model.estimator_ is not estimator
    estimator.fit(X, y, sample_weight=sample_weight)
    assert_allclose(model.estimator_.coef_, estimator.coef_)
    assert_allclose(model.estimator_.intercept_, estimator.intercept_)

    # 检查当 refit=False 和 cv="prefit" 时，`estimator_` 是否保持不变
    estimator = LogisticRegression().set_fit_request(sample_weight=True)
    estimator.fit(X, y, sample_weight=sample_weight)
    coef = estimator.coef_.copy()
    model = TunedThresholdClassifierCV(estimator, cv="prefit", refit=False).fit(
        X, y, sample_weight=sample_weight
    )

    assert model.estimator_ is estimator
    assert_allclose(model.estimator_.coef_, coef)

    # 检查在给定交叉验证的训练拆分上训练 `estimator_`
    estimator = LogisticRegression().set_fit_request(sample_weight=True)
    cv = [
        (np.arange(50), np.arange(50, 100)),
    ]  # 单一拆分
    model = TunedThresholdClassifierCV(estimator, cv=cv, refit=False).fit(
        X, y, sample_weight=sample_weight
    )

    assert model.estimator_ is not estimator
    # 如果有样本权重，使用交叉验证的训练集进行样本权重的获取
    if with_sample_weight:
        sw_train = sample_weight[cv[0][0]]
    else:
        sw_train = None
    estimator.fit(X[cv[0][0]], y[cv[0][0]], sample_weight=sw_train)
    assert_allclose(model.estimator_.coef_, estimator.coef_)



# 标记此测试函数使用 `enable_slep006` fixture
@pytest.mark.usefixtures("enable_slep006")
# 参数化测试函数，测试 `fit_params_type` 参数为 "list" 和 "array" 两种情况
@pytest.mark.parametrize("fit_params_type", ["list", "array"])
# 定义测试函数，检查在调用 `fit` 时将 `fit_params` 传递给分类器
def test_tuned_threshold_classifier_fit_params(fit_params_type):
    """Check that we pass `fit_params` to the classifier when calling `fit`."""
    # 生成分类数据 X 和 y
    X, y = make_classification(n_samples=100, random_state=0)
    # 根据 fit_params_type 生成 fit_params 字典
    fit_params = {
        "a": _convert_container(y, fit_params_type),
        "b": _convert_container(y, fit_params_type),
    }

    # 创建预期要求 fit_params 的 CheckingClassifier 对象
    classifier = CheckingClassifier(expected_fit_params=["a", "b"], random_state=0)
    classifier.set_fit_request(a=True, b=True)
    # 使用 TunedThresholdClassifierCV 对象进行拟合
    model = TunedThresholdClassifierCV(classifier)
    model.fit(X, y, **fit_params)



# 标记此测试函数使用 `enable_slep006` fixture
@pytest.mark.usefixtures("enable_slep006")
# 定义测试函数，检查在删除部分样本后将 `fit_params` 传递给分类器
def test_tuned_threshold_classifier_cv_zeros_sample_weights_equivalence():
    """Check that passing removing some sample from the dataset `X` is
    equivalent to passing a `sample_weight` with a factor 0."""
    # 从 sklearn.datasets 中加载鸢尾花数据集，返回特征矩阵 X 和目标向量 y
    X, y = load_iris(return_X_y=True)
    # 使用 StandardScaler 对数据进行标准化，以避免收敛问题
    X = StandardScaler().fit_transform(X)
    # 仅使用两个类别，并选择样本，使得二折交叉验证拆分后等效于使用 `sample_weight` 为 0
    X = np.vstack((X[:40], X[50:90]))
    y = np.hstack((y[:40], y[50:90]))
    # 创建一个与 y 维度相同的全零数组作为样本权重
    sample_weight = np.zeros_like(y)
    # 每隔一个元素设置为 1，即每隔一个样本的权重为 1，其余为 0
    sample_weight[::2] = 1

    # 创建一个 LogisticRegression 的估计器对象，并设定 fit 请求为使用样本权重
    estimator = LogisticRegression().set_fit_request(sample_weight=True)
    # 使用 TunedThresholdClassifierCV 对估计器进行调优，使用 2 折交叉验证
    model_without_weights = TunedThresholdClassifierCV(estimator, cv=2)
    # 克隆未使用样本权重的模型
    model_with_weights = clone(model_without_weights)

    # 使用带样本权重的数据拟合模型
    model_with_weights.fit(X, y, sample_weight=sample_weight)
    # 使用不带样本权重的数据拟合模型
    model_without_weights.fit(X[::2], y[::2])

    # 检查带权重模型和不带权重模型的系数是否接近
    assert_allclose(
        model_with_weights.estimator_.coef_, model_without_weights.estimator_.coef_
    )

    # 对带权重模型和不带权重模型分别进行预测概率计算
    y_pred_with_weights = model_with_weights.predict_proba(X)
    y_pred_without_weights = model_without_weights.predict_proba(X)
    # 检查两种情况下的预测概率是否接近
    assert_allclose(y_pred_with_weights, y_pred_without_weights)
# 测试调整过的阈值分类器，验证可以将数组传递给 `thresholds`，并在内部作为候选阈值使用
def test_tuned_threshold_classifier_thresholds_array():
    X, y = make_classification(random_state=0)  # 生成一个分类数据集
    estimator = LogisticRegression()  # 创建逻辑回归分类器
    thresholds = np.linspace(0, 1, 11)  # 创建一个包含 11 个均匀间隔值的阈值数组
    # 使用自定义阈值数组进行交叉验证调整分类器
    tuned_model = TunedThresholdClassifierCV(
        estimator,
        thresholds=thresholds,
        response_method="predict_proba",
        store_cv_results=True,
    ).fit(X, y)
    # 断言调整后的模型的交叉验证结果中的阈值与传入的阈值数组相近
    assert_allclose(tuned_model.cv_results_["thresholds"], thresholds)


# 使用 pytest 参数化装饰器标记，测试调整过的阈值分类器的 `store_cv_results` 功能
def test_tuned_threshold_classifier_store_cv_results(store_cv_results):
    X, y = make_classification(random_state=0)  # 生成一个分类数据集
    estimator = LogisticRegression()  # 创建逻辑回归分类器
    # 根据参数化装饰器提供的 store_cv_results 值，调整分类器并进行拟合
    tuned_model = TunedThresholdClassifierCV(
        estimator, store_cv_results=store_cv_results
    ).fit(X, y)
    if store_cv_results:
        # 断言如果 store_cv_results=True，分类器对象应该有 `cv_results_` 属性
        assert hasattr(tuned_model, "cv_results_")
    else:
        # 断言如果 store_cv_results=False，分类器对象不应该有 `cv_results_` 属性
        assert not hasattr(tuned_model, "cv_results_")


# 测试当 `cv` 参数设置为浮点数时的行为
def test_tuned_threshold_classifier_cv_float():
    X, y = make_classification(random_state=0)  # 生成一个分类数据集

    # 当 `refit=False` 且 `cv` 是浮点数时的情况：
    # 底层估计器将在由 ShuffleSplit 给出的训练集上进行拟合。检查是否得到相同的模型系数。
    test_size = 0.3
    estimator = LogisticRegression()  # 创建逻辑回归分类器
    tuned_model = TunedThresholdClassifierCV(
        estimator, cv=test_size, refit=False, random_state=0
    ).fit(X, y)

    # 使用 StratifiedShuffleSplit 分割数据集
    cv = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
    train_idx, val_idx = next(cv.split(X, y))
    cloned_estimator = clone(estimator).fit(X[train_idx], y[train_idx])

    # 断言调整后的模型的估计器系数与克隆估计器的系数相近
    assert_allclose(tuned_model.estimator_.coef_, cloned_estimator.coef_)

    # 当 `refit=True` 时，底层估计器在整个数据集上进行拟合
    tuned_model.set_params(refit=True).fit(X, y)
    cloned_estimator = clone(estimator).fit(X, y)

    # 再次断言调整后的模型的估计器系数与克隆估计器的系数相近
    assert_allclose(tuned_model.estimator_.coef_, cloned_estimator.coef_)


# 测试调整过的阈值分类器，当底层分类器返回常量概率时是否能够正确抛出 ValueError
def test_tuned_threshold_classifier_error_constant_predictor():
    X, y = make_classification(random_state=0)  # 生成一个分类数据集
    estimator = DummyClassifier(strategy="constant", constant=1)  # 创建一个常数预测的伪分类器
    # 使用自定义阈值分类器，检查是否能捕获到底层分类器返回常量预测的错误
    tuned_model = TunedThresholdClassifierCV(estimator, response_method="predict_proba")
    err_msg = "The provided estimator makes constant predictions"
    with pytest.raises(ValueError, match=err_msg):
        tuned_model.fit(X, y)


# 使用 pytest 参数化装饰器，测试固定阈值分类器的等效性，默认的 `response_method`
@pytest.mark.parametrize(
    "response_method", ["auto", "predict_proba", "decision_function"]
)
def test_fixed_threshold_classifier_equivalence_default(response_method):
    """Check that `FixedThresholdClassifier` has the same behaviour as the vanilla
    classifier.
    """
    # 生成一个随机的分类数据集 X 和 y
    X, y = make_classification(random_state=0)
    # 使用逻辑回归模型作为基础分类器，并拟合数据
    classifier = LogisticRegression().fit(X, y)
    # 使用 clone 函数复制基础分类器，并传入 FixedThresholdClassifier 中
    classifier_default_threshold = FixedThresholdClassifier(
        estimator=clone(classifier), response_method=response_method
    )
    # 使用复制的分类器拟合数据
    classifier_default_threshold.fit(X, y)

    # 模拟响应方法，根据 response_method 设置阈值进行预测
    if response_method in ("auto", "predict_proba"):
        # 如果 response_method 是 "auto" 或 "predict_proba"，使用 predict_proba 方法获取概率预测值
        y_score = classifier_default_threshold.predict_proba(X)[:, 1]
        threshold = 0.5
    else:  # 如果 response_method 是 "decision_function"
        # 使用 decision_function 方法获取决策函数值
        y_score = classifier_default_threshold.decision_function(X)
        threshold = 0.0

    # 根据阈值将预测概率或决策函数值转换为二元预测结果
    y_pred_lr = (y_score >= threshold).astype(int)
    # 使用 assert_allclose 函数验证 FixedThresholdClassifier 的预测结果与逻辑回归模型的预测结果是否接近
    assert_allclose(classifier_default_threshold.predict(X), y_pred_lr)
@pytest.mark.parametrize(
    "response_method, threshold", [("predict_proba", 0.7), ("decision_function", 2.0)]
)
@pytest.mark.parametrize("pos_label", [0, 1])
def test_fixed_threshold_classifier(response_method, threshold, pos_label):
    """检查固定阈值分类器的行为，验证对预测结果的阈值应用与响应方法输出的预测结果相同。"""
    # 创建一个随机的分类数据集
    X, y = make_classification(n_samples=50, random_state=0)
    # 使用逻辑回归拟合数据
    logistic_regression = LogisticRegression().fit(X, y)
    # 创建固定阈值分类器对象
    model = FixedThresholdClassifier(
        estimator=clone(logistic_regression),
        threshold=threshold,
        response_method=response_method,
        pos_label=pos_label,
    ).fit(X, y)

    # 检查底层估计器是否相同
    assert_allclose(model.estimator_.coef_, logistic_regression.coef_)

    # 模拟响应方法，应考虑`pos_label`参数
    if response_method == "predict_proba":
        y_score = model.predict_proba(X)[:, pos_label]
    else:  # response_method == "decision_function"
        y_score = model.decision_function(X)
        y_score = y_score if pos_label == 1 else -y_score

    # 创建一个从布尔值到类标签的映射
    map_to_label = np.array([0, 1]) if pos_label == 1 else np.array([1, 0])
    y_pred_lr = map_to_label[(y_score >= threshold).astype(int)]
    assert_allclose(model.predict(X), y_pred_lr)

    # 验证其他方法的输出是否与逻辑回归模型相同
    for method in ("predict_proba", "predict_log_proba", "decision_function"):
        assert_allclose(
            getattr(model, method)(X), getattr(logistic_regression, method)(X)
        )
        assert_allclose(
            getattr(model.estimator_, method)(X),
            getattr(logistic_regression, method)(X),
        )


@pytest.mark.usefixtures("enable_slep006")
def test_fixed_threshold_classifier_metadata_routing():
    """检查元数据路由功能是否正常工作。"""
    X, y = make_classification(random_state=0)
    sample_weight = np.ones_like(y)
    sample_weight[::2] = 2
    # 创建一个支持样本权重的逻辑回归分类器
    classifier = LogisticRegression().set_fit_request(sample_weight=True)
    classifier.fit(X, y, sample_weight=sample_weight)
    # 使用默认阈值创建固定阈值分类器对象
    classifier_default_threshold = FixedThresholdClassifier(estimator=clone(classifier))
    classifier_default_threshold.fit(X, y, sample_weight=sample_weight)
    # 验证估计器的系数是否相同
    assert_allclose(classifier_default_threshold.estimator_.coef_, classifier.coef_)


class ClassifierLoggingFit(ClassifierMixin, BaseEstimator):
    """记录`fit`调用次数的分类器。"""

    def __init__(self, fit_calls=0):
        self.fit_calls = fit_calls

    def fit(self, X, y, **fit_params):
        self.fit_calls += 1
        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        return np.ones((X.shape[0], 2), np.float64)  # pragma: nocover


def test_fixed_threshold_classifier_prefit():
    """检查带有`prefit`参数的`FixedThresholdClassifier`的行为。"""
    X, y = make_classification(random_state=0)
    # 创建一个 ClassifierLoggingFit 的实例作为 estimator
    estimator = ClassifierLoggingFit()
    # 使用 prefit=True 创建 FixedThresholdClassifier 的实例 model，并将 estimator 作为参数传入
    model = FixedThresholdClassifier(estimator=estimator, prefit=True)
    # 使用 pytest 的 raises 方法检查是否抛出 NotFittedError 异常
    with pytest.raises(NotFittedError):
        # 对 model 执行 fit 操作，期望抛出 NotFittedError 异常
        model.fit(X, y)

    # 检查在 prefit=True 时，确保不会克隆分类器
    # 先对 estimator 进行 fit 操作
    estimator.fit(X, y)
    # 再对 model 进行 fit 操作
    model.fit(X, y)
    # 断言 estimator 的 fit_calls 属性为 1
    assert estimator.fit_calls == 1
    # 断言 model 的 estimator_ 属性引用的是 estimator 对象本身
    assert model.estimator_ is estimator

    # 检查在 prefit=False 时，确保会克隆分类器
    # 使用 prefit=False 创建 FixedThresholdClassifier 的实例 model，并将 estimator 作为参数传入
    estimator = ClassifierLoggingFit()
    model = FixedThresholdClassifier(estimator=estimator, prefit=False)
    # 对 model 执行 fit 操作
    model.fit(X, y)
    # 断言 estimator 的 fit_calls 属性为 0
    assert estimator.fit_calls == 0
    # 断言 model 的 estimator_ 的 fit_calls 属性为 1，说明已经进行了一次 fit 操作
    assert model.estimator_.fit_calls == 1
    # 断言 model 的 estimator_ 不是同一个对象实例，而是克隆出来的新对象
    assert model.estimator_ is not estimator
```