# `D:\src\scipysrc\scikit-learn\sklearn\tests\test_pipeline.py`

```
"""
Test the pipeline module.
"""

# 导入所需的库和模块
import itertools  # 导入 itertools 库，用于高效循环和迭代操作
import re  # 导入 re 库，用于正则表达式操作
import shutil  # 导入 shutil 库，用于高级文件操作
import time  # 导入 time 库，用于时间相关功能
import warnings  # 导入 warnings 库，用于警告处理
from tempfile import mkdtemp  # 从 tempfile 模块中导入 mkdtemp 函数，用于创建临时目录

import joblib  # 导入 joblib 库，用于高效的对象序列化和反序列化
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试

# 导入 sklearn 库中的各个模块和类
from sklearn.base import BaseEstimator, TransformerMixin, clone, is_classifier
from sklearn.cluster import KMeans  # 导入 KMeans 聚类算法
from sklearn.datasets import load_iris  # 导入 load_iris 函数，用于加载鸢尾花数据集
from sklearn.decomposition import PCA, TruncatedSVD  # 导入 PCA 和 TruncatedSVD 降维算法
from sklearn.dummy import DummyRegressor  # 导入 DummyRegressor 类
from sklearn.ensemble import (
    HistGradientBoostingClassifier,  # 导入 HistGradientBoostingClassifier 类
    RandomForestClassifier,  # 导入 RandomForestClassifier 类
    RandomTreesEmbedding,  # 导入 RandomTreesEmbedding 类
)
from sklearn.exceptions import NotFittedError, UnsetMetadataPassedError  # 导入异常类
from sklearn.feature_extraction.text import CountVectorizer  # 导入 CountVectorizer 类
from sklearn.feature_selection import SelectKBest, f_classif  # 导入特征选择相关函数和类
from sklearn.impute import SimpleImputer  # 导入 SimpleImputer 类
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression  # 导入线性模型类
from sklearn.metrics import accuracy_score, r2_score  # 导入评估指标函数
from sklearn.model_selection import train_test_split  # 导入 train_test_split 函数
from sklearn.neighbors import LocalOutlierFactor  # 导入 LocalOutlierFactor 类
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline, make_union  # 导入 Pipeline 相关类和函数
from sklearn.preprocessing import FunctionTransformer, StandardScaler  # 导入数据预处理相关类
from sklearn.svm import SVC  # 导入支持向量机分类器类
from sklearn.tests.metadata_routing_common import (  # 导入测试相关的自定义模块和函数
    ConsumingNoFitTransformTransformer,
    ConsumingTransformer,
    _Registry,
    check_recorded_metadata,
)
from sklearn.utils._metadata_requests import COMPOSITE_METHODS, METHODS  # 导入元数据请求相关内容
from sklearn.utils._testing import (  # 导入测试相关的函数和类
    MinimalClassifier,
    MinimalRegressor,
    MinimalTransformer,
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
)
from sklearn.utils.fixes import CSR_CONTAINERS  # 导入 CSR_CONTAINERS 修复相关内容
from sklearn.utils.validation import check_is_fitted  # 导入 check_is_fitted 函数

# 加载鸢尾花数据集
iris = load_iris()

# JUNK_FOOD_DOCS 列表，包含文本片段作为测试数据
JUNK_FOOD_DOCS = (
    "the pizza pizza beer copyright",
    "the pizza burger beer copyright",
    "the the pizza beer beer copyright",
    "the burger beer beer copyright",
    "the coke burger coke copyright",
    "the coke burger burger",
)


class NoFit:
    """Small class to test parameter dispatching."""

    def __init__(self, a=None, b=None):
        self.a = a
        self.b = b


class NoTrans(NoFit):
    def fit(self, X, y):
        return self

    def get_params(self, deep=False):
        return {"a": self.a, "b": self.b}

    def set_params(self, **params):
        self.a = params["a"]
        return self


class NoInvTransf(NoTrans):
    def transform(self, X):
        return X


class Transf(NoInvTransf):
    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X


class TransfFitParams(Transf):
    def fit(self, X, y, **fit_params):
        self.fit_params = fit_params
        return self


class Mult(BaseEstimator):
    def __init__(self, mult=1):
        self.mult = mult

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.asarray(X) * self.mult

    def inverse_transform(self, X):
        return np.asarray(X) / self.mult
    # 定义一个方法 predict，用于对输入的数据 X 进行预测
    def predict(self, X):
        # 将输入的 X 转换为 NumPy 数组，然后每个元素乘以对象的 mult 属性，然后按行求和
        return (np.asarray(X) * self.mult).sum(axis=1)

    # 定义 predict_proba、predict_log_proba、decision_function 这三个方法，它们都指向 predict 方法
    predict_proba = predict_log_proba = decision_function = predict

    # 定义一个方法 score，用于计算输入数据 X 的总和，y 参数未被使用
    def score(self, X, y=None):
        return np.sum(X)
# 定义一个自定义的参数估计器类 FitParamT，继承自 BaseEstimator
class FitParamT(BaseEstimator):
    
    # 构造函数，初始化一个标志位 successful，表示是否成功
    def __init__(self):
        self.successful = False

    # 训练方法，接受特征 X 和标签 y，以及一个可选参数 should_succeed，用来设置成功标志位
    def fit(self, X, y, should_succeed=False):
        self.successful = should_succeed

    # 预测方法，直接返回成功标志位
    def predict(self, X):
        return self.successful

    # 组合方法，先调用 fit 方法进行训练，然后调用 predict 方法进行预测并返回结果
    def fit_predict(self, X, y, should_succeed=False):
        self.fit(X, y, should_succeed=should_succeed)
        return self.predict(X)

    # 打分方法，接受特征 X、标签 y 和样本权重 sample_weight，如果有权重，则将特征加权后求和返回
    def score(self, X, y=None, sample_weight=None):
        if sample_weight is not None:
            X = X * sample_weight
        return np.sum(X)


# 定义一个 DummyTransf 类，继承自 Transf，用来存储特征列的均值
class DummyTransf(Transf):
    
    # 训练方法，计算特征 X 按列的均值，并存储在 means_ 属性中；记录当前时间戳在 timestamp_ 属性中
    def fit(self, X, y):
        self.means_ = np.mean(X, axis=0)
        self.timestamp_ = time.time()
        return self


# 定义一个 DummyEstimatorParams 类，继承自 BaseEstimator，用于模拟带参数的分类器
class DummyEstimatorParams(BaseEstimator):
    
    # 训练方法，简单返回实例本身
    def fit(self, X, y):
        return self

    # 预测方法，接受特征 X 和一个可选参数 got_attribute，用来设置实例的 got_attribute 属性，并返回实例本身
    def predict(self, X, got_attribute=False):
        self.got_attribute = got_attribute
        return self

    # 预测概率方法，与 predict 方法类似，设置 got_attribute 属性，并返回实例本身
    def predict_proba(self, X, got_attribute=False):
        self.got_attribute = got_attribute
        return self

    # 预测对数概率方法，设置 got_attribute 属性，并返回实例本身
    def predict_log_proba(self, X, got_attribute=False):
        self.got_attribute = got_attribute
        return self


# 定义一个测试函数 test_pipeline_invalid_parameters，用于测试 pipeline 中的不合法参数情况
def test_pipeline_invalid_parameters():
    
    # 测试 pipeline 初始化参数不合法的情况，在 fit 方法中会抛出 TypeError 异常
    pipeline = Pipeline([(1, 1)])
    with pytest.raises(TypeError):
        pipeline.fit([[1]], [1])

    # 测试 pipeline 中的最后一个步骤不实现 fit 方法或者不是 'passthrough' 字符串的情况，会抛出 TypeError 异常
    msg = (
        "Last step of Pipeline should implement fit "
        "or be the string 'passthrough'"
        ".*NoFit.*"
    )
    pipeline = Pipeline([("clf", NoFit())])
    with pytest.raises(TypeError, match=msg):
        pipeline.fit([[1]], [1])

    # 烟雾测试，使用只有一个估计器的 pipeline
    clf = NoTrans()
    pipe = Pipeline([("svc", clf)])
    assert pipe.get_params(deep=True) == dict(
        svc__a=None, svc__b=None, svc=clf, **pipe.get_params(deep=False)
    )

    # 检查参数是否被正确设置
    pipe.set_params(svc__a=0.1)
    assert clf.a == 0.1
    assert clf.b is None
    # 烟雾测试 pipeline 的字符串表示形式
    repr(pipe)

    # 使用两个对象测试
    clf = SVC()
    filter1 = SelectKBest(f_classif)
    pipe = Pipeline([("anova", filter1), ("svc", clf)])

    # 检查估计器在构建 pipeline 时是否被克隆
    assert pipe.named_steps["anova"] is filter1
    assert pipe.named_steps["svc"] is clf

    # 检查不能使用非转换器进行拟合的情况，抛出 TypeError 异常，因为 NoTrans 实现了 fit 方法但没有 transform 方法
    msg = "All intermediate steps should be transformers.*\\bNoTrans\\b.*"
    pipeline = Pipeline([("t", NoTrans()), ("svc", clf)])
    with pytest.raises(TypeError, match=msg):
        pipeline.fit([[1]], [1])

    # 检查参数是否被正确设置
    pipe.set_params(svc__C=0.1)
    # 断言 clf.C 的值为 0.1，用于确保参数设置正确
    assert clf.C == 0.1

    # 对管道进行 repr 的烟雾测试，通常用于检查对象的字符串表示是否正常
    repr(pipe)

    # 当设置参数名错误时，检查参数是否未设置
    msg = re.escape(
        "Invalid parameter 'C' for estimator SelectKBest(). Valid parameters are: ['k',"
        " 'score_func']."
    )
    # 使用 pytest 检查是否抛出 ValueError 异常，并匹配预期的错误消息
    with pytest.raises(ValueError, match=msg):
        pipe.set_params(anova__C=0.1)

    # 测试克隆功能，确保克隆后的对象与原对象不是同一个引用
    pipe2 = clone(pipe)
    assert pipe.named_steps["svc"] is not pipe2.named_steps["svc"]

    # 检查除了估计器之外的参数是否相同
    params = pipe.get_params(deep=True)
    params2 = pipe2.get_params(deep=True)

    # 删除不属于估计器的参数
    for x in pipe.get_params(deep=False):
        params.pop(x)

    for x in pipe2.get_params(deep=False):
        params2.pop(x)

    params.pop("svc")
    params.pop("anova")
    params2.pop("svc")
    params2.pop("anova")

    # 断言剩余的参数字典是否完全相同
    assert params == params2
def test_pipeline_init_tuple():
    # Pipeline accepts steps as tuple
    
    # 创建一个输入数组 X
    X = np.array([[1, 2]])
    # 创建一个 Pipeline 对象，其中包含两个步骤，每个步骤由元组表示
    pipe = Pipeline((("transf", Transf()), ("clf", FitParamT())))
    # 对 Pipeline 对象进行拟合，不使用目标值 y
    pipe.fit(X, y=None)
    # 对拟合后的 Pipeline 对象进行评分
    pipe.score(X)

    # 设置 Pipeline 中第一个步骤的参数 transf 为 "passthrough"
    pipe.set_params(transf="passthrough")
    # 再次对 Pipeline 对象进行拟合
    pipe.fit(X, y=None)
    # 再次对拟合后的 Pipeline 对象进行评分
    pipe.score(X)


def test_pipeline_methods_anova():
    # Test the various methods of the pipeline (anova).
    
    # 载入经典的 Iris 数据集的特征 X 和目标值 y
    X = iris.data
    y = iris.target
    # 创建一个 Pipeline 对象，包含两个步骤：Anova 和 LogisticRegression
    clf = LogisticRegression()
    filter1 = SelectKBest(f_classif, k=2)
    pipe = Pipeline([("anova", filter1), ("logistic", clf)])
    # 对 Pipeline 对象进行拟合
    pipe.fit(X, y)
    # 使用拟合后的 Pipeline 对象进行预测
    pipe.predict(X)
    # 使用拟合后的 Pipeline 对象进行预测概率估计
    pipe.predict_proba(X)
    # 使用拟合后的 Pipeline 对象进行预测对数概率估计
    pipe.predict_log_proba(X)
    # 使用拟合后的 Pipeline 对象计算得分
    pipe.score(X, y)


def test_pipeline_fit_params():
    # Test that the pipeline can take fit parameters
    
    # 创建一个 Pipeline 对象，包含两个步骤：Transf 和 FitParamT
    pipe = Pipeline([("transf", Transf()), ("clf", FitParamT())])
    # 使用 fit 方法拟合 Pipeline 对象，传入 clf__should_succeed 参数
    pipe.fit(X=None, y=None, clf__should_succeed=True)
    # 断言分类器应返回 True
    assert pipe.predict(None)
    # 断言转换器的参数未被更改
    assert pipe.named_steps["transf"].a is None
    assert pipe.named_steps["transf"].b is None
    # 断言使用无效参数会引发 TypeError 错误
    msg = re.escape("fit() got an unexpected keyword argument 'bad'")
    with pytest.raises(TypeError, match=msg):
        pipe.fit(None, None, clf__bad=True)


def test_pipeline_sample_weight_supported():
    # Pipeline should pass sample_weight
    
    # 创建一个输入数组 X
    X = np.array([[1, 2]])
    # 创建一个 Pipeline 对象，包含两个步骤：Transf 和 FitParamT
    pipe = Pipeline([("transf", Transf()), ("clf", FitParamT())])
    # 使用 fit 方法拟合 Pipeline 对象
    pipe.fit(X, y=None)
    # 断言 Pipeline 对象的得分为 3
    assert pipe.score(X) == 3
    assert pipe.score(X, y=None) == 3
    assert pipe.score(X, y=None, sample_weight=None) == 3
    # 断言使用样本权重参数后的得分为 8
    assert pipe.score(X, sample_weight=np.array([2, 3])) == 8


def test_pipeline_sample_weight_unsupported():
    # When sample_weight is None it shouldn't be passed
    
    # 创建一个输入数组 X
    X = np.array([[1, 2]])
    # 创建一个 Pipeline 对象，包含两个步骤：Transf 和 Mult
    pipe = Pipeline([("transf", Transf()), ("clf", Mult())])
    # 使用 fit 方法拟合 Pipeline 对象
    pipe.fit(X, y=None)
    # 断言 Pipeline 对象的得分为 3
    assert pipe.score(X) == 3
    assert pipe.score(X, sample_weight=None) == 3

    # 断言在使用无效参数时会引发 TypeError 错误
    msg = re.escape("score() got an unexpected keyword argument 'sample_weight'")
    with pytest.raises(TypeError, match=msg):
        pipe.score(X, sample_weight=np.array([2, 3]))


def test_pipeline_raise_set_params_error():
    # Test pipeline raises set params error message for nested models.
    
    # 创建一个 Pipeline 对象，包含一个步骤 LinearRegression
    pipe = Pipeline([("cls", LinearRegression())])

    # 预期的错误消息
    error_msg = re.escape(
        "Invalid parameter 'fake' for estimator Pipeline(steps=[('cls',"
        " LinearRegression())]). Valid parameters are: ['memory', 'steps',"
        " 'verbose']."
    )
    # 使用 set_params 方法设置无效参数 'fake'
    with pytest.raises(ValueError, match=error_msg):
        pipe.set_params(fake="nope")

    # 对复合参数的无效外部参数名称，预期的错误消息与上述相同
    with pytest.raises(ValueError, match=error_msg):
        pipe.set_params(fake__estimator="nope")

    # 对内部参数的预期错误消息
    # 将错误消息中的特殊字符转义，确保它可以用作正则表达式的匹配模式
    error_msg = re.escape(
        "Invalid parameter 'invalid_param' for estimator LinearRegression(). Valid"
        " parameters are: ['copy_X', 'fit_intercept', 'n_jobs', 'positive']."
    )
    # 使用 pytest 模块的 raises 方法来断言抛出 ValueError 异常，并匹配预期的错误消息
    with pytest.raises(ValueError, match=error_msg):
        # 尝试设置管道 pipe 中类别 cls__invalid_param 的参数为 "nope"
        pipe.set_params(cls__invalid_param="nope")
def test_pipeline_methods_pca_svm():
    # Test the various methods of the pipeline (pca + svm).
    X = iris.data
    y = iris.target
    # Test with PCA + SVC
    clf = SVC(probability=True, random_state=0)  # 使用SVC分类器，设置probability为True，random_state为0
    pca = PCA(svd_solver="full", n_components="mle", whiten=True)  # 使用PCA降维，设置svd_solver为"full"，n_components为"mle"，whiten为True
    pipe = Pipeline([("pca", pca), ("svc", clf)])  # 创建Pipeline，包含PCA和SVC步骤
    pipe.fit(X, y)  # 在数据集上拟合Pipeline
    pipe.predict(X)  # 对数据集进行预测
    pipe.predict_proba(X)  # 对数据集进行概率预测
    pipe.predict_log_proba(X)  # 对数据集进行对数概率预测
    pipe.score(X, y)  # 计算Pipeline在数据集上的准确率


def test_pipeline_score_samples_pca_lof():
    X = iris.data
    # Test that the score_samples method is implemented on a pipeline.
    # Test that the score_samples method on pipeline yields same results as
    # applying transform and score_samples steps separately.
    pca = PCA(svd_solver="full", n_components="mle", whiten=True)  # 使用PCA降维，设置svd_solver为"full"，n_components为"mle"，whiten为True
    lof = LocalOutlierFactor(novelty=True)  # 使用LocalOutlierFactor检测异常点，设置novelty为True
    pipe = Pipeline([("pca", pca), ("lof", lof)])  # 创建Pipeline，包含PCA和LOF步骤
    pipe.fit(X)  # 在数据集上拟合Pipeline
    # Check the shapes
    assert pipe.score_samples(X).shape == (X.shape[0],)  # 检查score_samples方法返回结果的形状是否正确
    # Check the values
    lof.fit(pca.fit_transform(X))  # 先对数据进行PCA处理，然后在降维后的数据上拟合LOF模型
    assert_allclose(pipe.score_samples(X), lof.score_samples(pca.transform(X)))  # 检查Pipeline和独立应用transform和score_samples步骤的结果是否接近


def test_score_samples_on_pipeline_without_score_samples():
    X = np.array([[1], [2]])
    y = np.array([1, 2])
    # Test that a pipeline does not have score_samples method when the final
    # step of the pipeline does not have score_samples defined.
    pipe = make_pipeline(LogisticRegression())  # 创建Pipeline，仅包含LogisticRegression步骤
    pipe.fit(X, y)  # 在数据集上拟合Pipeline

    inner_msg = "'LogisticRegression' object has no attribute 'score_samples'"
    outer_msg = "'Pipeline' has no attribute 'score_samples'"
    with pytest.raises(AttributeError, match=outer_msg) as exec_info:  # 使用pytest检查是否抛出AttributeError异常
        pipe.score_samples(X)  # 调用score_samples方法

    assert isinstance(exec_info.value.__cause__, AttributeError)  # 检查异常的原因是否是AttributeError
    assert inner_msg in str(exec_info.value.__cause__)  # 检查异常消息是否包含指定信息


def test_pipeline_methods_preprocessing_svm():
    # Test the various methods of the pipeline (preprocessing + svm).
    X = iris.data
    y = iris.target
    n_samples = X.shape[0]
    n_classes = len(np.unique(y))
    scaler = StandardScaler()  # 使用StandardScaler进行特征缩放
    pca = PCA(n_components=2, svd_solver="randomized", whiten=True)  # 使用PCA降维，设置n_components为2，svd_solver为"randomized"，whiten为True
    clf = SVC(probability=True, random_state=0, decision_function_shape="ovr")  # 使用SVC分类器，设置probability为True，random_state为0，decision_function_shape为"ovr"

    for preprocessing in [scaler, pca]:
        pipe = Pipeline([("preprocess", preprocessing), ("svc", clf)])  # 创建Pipeline，包含预处理步骤和SVC步骤
        pipe.fit(X, y)  # 在数据集上拟合Pipeline

        # check shapes of various prediction functions
        predict = pipe.predict(X)  # 预测结果
        assert predict.shape == (n_samples,)  # 检查预测结果的形状是否正确

        proba = pipe.predict_proba(X)  # 概率预测结果
        assert proba.shape == (n_samples, n_classes)  # 检查概率预测结果的形状是否正确

        log_proba = pipe.predict_log_proba(X)  # 对数概率预测结果
        assert log_proba.shape == (n_samples, n_classes)  # 检查对数概率预测结果的形状是否正确

        decision_function = pipe.decision_function(X)  # 决策函数值
        assert decision_function.shape == (n_samples, n_classes)  # 检查决策函数值的形状是否正确

        pipe.score(X, y)  # 计算Pipeline在数据集上的准确率


def test_fit_predict_on_pipeline():
    # test that the fit_predict method is implemented on a pipeline
    # test that the fit_predict on pipeline yields same results as applying
    # 创建一个标准化（Scaler）对象，用于数据标准化操作
    scaler = StandardScaler()
    # 创建一个 KMeans 聚类对象，设置随机种子为0，n_init 参数设为 "auto"
    km = KMeans(random_state=0, n_init="auto")
    
    # 由于 Pipeline 在构造时不会克隆估算器（estimator），
    # 因此它必须有自己的估算器
    scaler_for_pipeline = StandardScaler()
    km_for_pipeline = KMeans(random_state=0, n_init="auto")

    # 首先单独执行数据标准化和聚类步骤
    scaled = scaler.fit_transform(iris.data)
    # 对标准化后的数据执行 KMeans 聚类，并获取预测结果
    separate_pred = km.fit_predict(scaled)

    # 使用 Pipeline 将标准化和聚类合并为一个步骤
    pipe = Pipeline([("scaler", scaler_for_pipeline), ("Kmeans", km_for_pipeline)])
    # 对数据执行 Pipeline 中的标准化和聚类操作，并获取预测结果
    pipeline_pred = pipe.fit_predict(iris.data)

    # 断言 Pipeline 和单独执行步骤的预测结果应该几乎相等
    assert_array_almost_equal(pipeline_pred, separate_pred)
def test_fit_predict_on_pipeline_without_fit_predict():
    # 测试管道中当最终步骤没有定义 fit_predict 方法时，管道不具有 fit_predict 方法
    scaler = StandardScaler()  # 创建一个 StandardScaler 对象
    pca = PCA(svd_solver="full")  # 创建一个 PCA 对象，指定 svd_solver 为 "full"
    pipe = Pipeline([("scaler", scaler), ("pca", pca)])  # 创建管道对象，包括 scaler 和 pca 两个步骤

    outer_msg = "'Pipeline' has no attribute 'fit_predict'"
    inner_msg = "'PCA' object has no attribute 'fit_predict'"
    with pytest.raises(AttributeError, match=outer_msg) as exec_info:
        getattr(pipe, "fit_predict")  # 尝试获取管道的 fit_predict 方法
    assert isinstance(exec_info.value.__cause__, AttributeError)
    assert inner_msg in str(exec_info.value.__cause__)


def test_fit_predict_with_intermediate_fit_params():
    # 测试当调用 fit_predict 时，Pipeline 是否将 fit_params 传递给中间步骤
    pipe = Pipeline([("transf", TransfFitParams()), ("clf", FitParamT())])  # 创建管道对象，包括 transf 和 clf 两个步骤
    pipe.fit_predict(
        X=None, y=None, transf__should_get_this=True, clf__should_succeed=True
    )  # 调用管道的 fit_predict 方法，并传递 fit_params 给中间步骤
    assert pipe.named_steps["transf"].fit_params["should_get_this"]
    assert pipe.named_steps["clf"].successful
    assert "should_succeed" not in pipe.named_steps["transf"].fit_params


@pytest.mark.parametrize(
    "method_name", ["predict", "predict_proba", "predict_log_proba"]
)
def test_predict_methods_with_predict_params(method_name):
    # 测试当调用 predict_* 方法时，Pipeline 是否将 predict_* 参数传递给最终估计器
    pipe = Pipeline([("transf", Transf()), ("clf", DummyEstimatorParams())])  # 创建管道对象，包括 transf 和 clf 两个步骤
    pipe.fit(None, None)  # 对管道进行拟合
    method = getattr(pipe, method_name)  # 获取管道对象的指定预测方法（predict/predict_proba/predict_log_proba）
    method(X=None, got_attribute=True)  # 调用指定的预测方法，并传递参数

    assert pipe.named_steps["clf"].got_attribute


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_feature_union(csr_container):
    # 特征联合的基本合理性检查
    X = iris.data  # 加载数据集特征
    X -= X.mean(axis=0)  # 中心化数据
    y = iris.target  # 加载数据集标签
    svd = TruncatedSVD(n_components=2, random_state=0)  # 创建 SVD 对象
    select = SelectKBest(k=1)  # 创建 SelectKBest 对象
    fs = FeatureUnion([("svd", svd), ("select", select)])  # 创建特征联合对象，包括 svd 和 select 两个步骤
    fs.fit(X, y)  # 对特征联合对象进行拟合
    X_transformed = fs.transform(X)  # 对数据集进行转换
    assert X_transformed.shape == (X.shape[0], 3)  # 断言转换后的形状是否正确

    # 检查是否符合预期
    assert_array_almost_equal(X_transformed[:, :-1], svd.fit_transform(X))
    assert_array_equal(X_transformed[:, -1], select.fit_transform(X, y).ravel())

    # 测试稀疏输入是否有效
    # 使用不同的 svd 对象以控制 random_state 流
    fs = FeatureUnion([("svd", svd), ("select", select)])
    X_sp = csr_container(X)
    X_sp_transformed = fs.fit_transform(X_sp, y)
    assert_array_almost_equal(X_transformed, X_sp_transformed.toarray())

    # 测试克隆
    fs2 = clone(fs)
    assert fs.transformer_list[0][1] is not fs2.transformer_list[0][1]

    # 测试设置参数
    fs.set_params(select__k=2)
    assert fs.fit_transform(X, y).shape == (X.shape[0], 4)

    # 测试它是否适用于缺少 fit_transform 的转换器
    # 创建 FeatureUnion 对象，将三个转换器（Transf、svd、select）组合起来
    fs = FeatureUnion([("mock", Transf()), ("svd", svd), ("select", select)])
    # 对输入数据 X 和目标数据 y 进行联合拟合和转换
    X_transformed = fs.fit_transform(X, y)
    # 断言转换后的数据形状是否符合预期
    assert X_transformed.shape == (X.shape[0], 8)

    # 测试当某些元素不支持 transform 时是否会出现错误
    msg = "All estimators should implement fit and transform.*\\bNoTrans\\b"
    # 创建 FeatureUnion 对象，包含两个转换器（Transf、NoTrans）
    fs = FeatureUnion([("transform", Transf()), ("no_transform", NoTrans())])
    # 使用 pytest 检查是否会抛出 TypeError 异常，并匹配特定的错误消息
    with pytest.raises(TypeError, match=msg):
        fs.fit(X)

    # 测试初始化是否接受元组作为参数
    # 创建 FeatureUnion 对象，使用元组形式指定转换器（svd、select）
    fs = FeatureUnion((("svd", svd), ("select", select)))
    # 对输入数据 X 和目标数据 y 进行拟合
    fs.fit(X, y)
def test_feature_union_named_transformers():
    """Check the behaviour of `named_transformers` attribute."""
    # 创建 Transf 和 NoInvTransf 的实例
    transf = Transf()
    noinvtransf = NoInvTransf()
    # 使用 FeatureUnion 组合两个转换器
    fs = FeatureUnion([("transf", transf), ("noinvtransf", noinvtransf)])
    # 断言 named_transformers 字典中的键值对是否正确
    assert fs.named_transformers["transf"] == transf
    assert fs.named_transformers["noinvtransf"] == noinvtransf

    # 测试 named 属性
    assert fs.named_transformers.transf == transf
    assert fs.named_transformers.noinvtransf == noinvtransf


def test_make_union():
    pca = PCA(svd_solver="full")
    mock = Transf()
    # 使用 make_union 创建一个包含 PCA 和 Transf 的 FeatureUnion
    fu = make_union(pca, mock)
    # 获取 transformer_list 中的名称和转换器
    names, transformers = zip(*fu.transformer_list)
    # 断言名称是否正确
    assert names == ("pca", "transf")
    # 断言转换器是否正确
    assert transformers == (pca, mock)


def test_make_union_kwargs():
    pca = PCA(svd_solver="full")
    mock = Transf()
    # 使用 make_union 创建一个包含 PCA 和 Transf 的 FeatureUnion，并设置 n_jobs=3
    fu = make_union(pca, mock, n_jobs=3)
    # 断言 transformer_list 是否与未设置 n_jobs 的相同
    assert fu.transformer_list == make_union(pca, mock).transformer_list
    # 断言 n_jobs 是否为 3
    assert 3 == fu.n_jobs

    # 无效的关键字参数应该引发错误消息
    msg = re.escape(
        "make_union() got an unexpected keyword argument 'transformer_weights'"
    )
    with pytest.raises(TypeError, match=msg):
        make_union(pca, mock, transformer_weights={"pca": 10, "Transf": 1})


def test_pipeline_transform():
    # 测试 pipeline 是否能够正常使用最后一个转换器
    # 以及测试 pipeline.transform 和 pipeline.inverse_transform
    X = iris.data
    pca = PCA(n_components=2, svd_solver="full")
    pipeline = Pipeline([("pca", pca)])

    # 测试 transform 和 fit_transform：
    X_trans = pipeline.fit(X).transform(X)
    X_trans2 = pipeline.fit_transform(X)
    X_trans3 = pca.fit_transform(X)
    assert_array_almost_equal(X_trans, X_trans2)
    assert_array_almost_equal(X_trans, X_trans3)

    X_back = pipeline.inverse_transform(X_trans)
    X_back2 = pca.inverse_transform(X_trans)
    assert_array_almost_equal(X_back, X_back2)


def test_pipeline_fit_transform():
    # 测试 pipeline 是否能够处理缺少 fit_transform 方法的转换器
    X = iris.data
    y = iris.target
    transf = Transf()
    pipeline = Pipeline([("mock", transf)])

    # 测试 fit_transform：
    X_trans = pipeline.fit_transform(X, y)
    X_trans2 = transf.fit(X, y).transform(X)
    assert_array_almost_equal(X_trans, X_trans2)


@pytest.mark.parametrize(
    "start, end", [(0, 1), (0, 2), (1, 2), (1, 3), (None, 1), (1, None), (None, None)]
)
def test_pipeline_slice(start, end):
    # 创建包含 Transf1, Transf2 和 FitParamT 的 Pipeline
    # 同时设置 memory 和 verbose 参数
    pipe = Pipeline(
        [("transf1", Transf()), ("transf2", Transf()), ("clf", FitParamT())],
        memory="123",
        verbose=True,
    )
    # 对 Pipeline 进行切片操作
    pipe_slice = pipe[start:end]
    # 测试返回的对象是否为 Pipeline 类型
    assert isinstance(pipe_slice, Pipeline)
    # 测试切片后的步骤是否正确
    assert pipe_slice.steps == pipe.steps[start:end]
    # 测试 named_steps 属性
    assert (
        list(pipe_slice.named_steps.items())
        == list(pipe.named_steps.items())[start:end]
    )
    # 测试剩余的参数是否与原来的相同
    # 获取管道对象的参数，不包括子管道中的参数
    pipe_params = pipe.get_params(deep=False)
    # 获取子管道对象的参数，不包括其子管道中的参数
    pipe_slice_params = pipe_slice.get_params(deep=False)
    # 删除管道参数字典中的 "steps" 键值对
    del pipe_params["steps"]
    # 删除子管道参数字典中的 "steps" 键值对
    del pipe_slice_params["steps"]
    # 断言管道参数字典与子管道参数字典相等
    assert pipe_params == pipe_slice_params
    # 测试异常情况，确认是否抛出 ValueError 异常，并匹配指定的错误消息
    msg = "Pipeline slicing only supports a step of 1"
    with pytest.raises(ValueError, match=msg):
        # 对管道进行切片操作，确认仅支持步长为 1 的情况
        pipe[start:end:-1]
def test_pipeline_index():
    # 创建转换器对象
    transf = Transf()
    # 创建分类器对象
    clf = FitParamT()
    # 创建管道对象，将转换器和分类器作为步骤
    pipe = Pipeline([("transf", transf), ("clf", clf)])
    # 验证通过索引获取第一个步骤
    assert pipe[0] == transf
    # 验证通过名称获取第一个步骤
    assert pipe["transf"] == transf
    # 验证通过索引获取最后一个步骤
    assert pipe[-1] == clf
    # 验证通过名称获取最后一个步骤
    assert pipe["clf"] == clf

    # 如果超出索引范围应该引发错误
    with pytest.raises(IndexError):
        pipe[3]

    # 如果使用错误的元素名称进行索引应该引发错误
    with pytest.raises(KeyError):
        pipe["foobar"]


def test_set_pipeline_steps():
    # 创建两个转换器对象
    transf1 = Transf()
    transf2 = Transf()
    # 创建管道对象，包含一个转换器步骤
    pipeline = Pipeline([("mock", transf1)])
    # 验证初始步骤名称与第一个转换器对象相同
    assert pipeline.named_steps["mock"] is transf1

    # 直接设置步骤属性
    pipeline.steps = [("mock2", transf2)]
    # 验证原始步骤名称不再存在，并且新名称对应于第二个转换器对象
    assert "mock" not in pipeline.named_steps
    assert pipeline.named_steps["mock2"] is transf2
    assert [("mock2", transf2)] == pipeline.steps

    # 使用 set_params 方法设置步骤
    pipeline.set_params(steps=[("mock", transf1)])
    assert [("mock", transf1)] == pipeline.steps

    # 使用 set_params 方法替换单个步骤
    pipeline.set_params(mock=transf2)
    assert [("mock", transf2)] == pipeline.steps

    # 使用无效数据调用 set_params
    pipeline.set_params(steps=[("junk", ())])
    msg = re.escape(
        "Last step of Pipeline should implement fit or be the string 'passthrough'."
    )
    with pytest.raises(TypeError, match=msg):
        pipeline.fit([[1]], [1])

    msg = "This 'Pipeline' has no attribute 'fit_transform'"
    with pytest.raises(AttributeError, match=msg):
        pipeline.fit_transform([[1]], [1])


def test_pipeline_named_steps():
    # 创建转换器和乘法器对象
    transf = Transf()
    mult2 = Mult(mult=2)
    # 创建管道对象，包含两个步骤
    pipeline = Pipeline([("mock", transf), ("mult", mult2)])

    # 测试通过 named_steps 属性访问步骤
    assert "mock" in pipeline.named_steps
    assert "mock2" not in pipeline.named_steps
    assert pipeline.named_steps.mock is transf
    assert pipeline.named_steps.mult is mult2

    # 测试在步骤名称冲突时 named_steps 的行为
    pipeline = Pipeline([("values", transf), ("mult", mult2)])
    assert pipeline.named_steps.values is not transf
    assert pipeline.named_steps.mult is mult2


@pytest.mark.parametrize("passthrough", [None, "passthrough"])
def test_pipeline_correctly_adjusts_steps(passthrough):
    # 创建数据和多个乘法器对象
    X = np.array([[1]])
    y = np.array([1])
    mult2 = Mult(mult=2)
    mult3 = Mult(mult=3)
    mult5 = Mult(mult=5)

    # 创建管道对象，包含多个步骤
    pipeline = Pipeline(
        [("m2", mult2), ("bad", passthrough), ("m3", mult3), ("m5", mult5)]
    )

    # 对管道对象进行拟合操作
    pipeline.fit(X, y)
    # 预期的步骤名称顺序
    expected_names = ["m2", "bad", "m3", "m5"]
    actual_names = [name for name, _ in pipeline.steps]
    assert expected_names == actual_names


@pytest.mark.parametrize("passthrough", [None, "passthrough"])
def test_set_pipeline_step_passthrough(passthrough):
    # 创建数据和多个乘法器对象
    X = np.array([[1]])
    y = np.array([1])
    mult2 = Mult(mult=2)
    mult3 = Mult(mult=3)
    mult5 = Mult(mult=5)

    def make():
        # 创建管道对象，包含多个乘法器步骤
        return Pipeline([("m2", mult2), ("m3", mult3), ("last", mult5)])
    # 创建管道对象
    pipeline = make()

    # 计算表达式 2 * 3 * 5 并赋值给 exp
    exp = 2 * 3 * 5
    # 断言管道进行拟合和转换后的结果是否与预期数组相等
    assert_array_equal([[exp]], pipeline.fit_transform(X, y))
    # 断言管道拟合后进行预测的结果是否与预期数组相等
    assert_array_equal([exp], pipeline.fit(X).predict(X))
    # 断言通过反向转换后的结果是否与原始数据 X 相等
    assert_array_equal(X, pipeline.inverse_transform([[exp]]))

    # 设置管道中 m3 步骤为 passthrough
    pipeline.set_params(m3=passthrough)
    # 计算表达式 2 * 5 并赋值给 exp
    exp = 2 * 5
    # 断言管道进行拟合和转换后的结果是否与预期数组相等
    assert_array_equal([[exp]], pipeline.fit_transform(X, y))
    # 断言管道拟合后进行预测的结果是否与预期数组相等
    assert_array_equal([exp], pipeline.fit(X).predict(X))
    # 断言通过反向转换后的结果是否与原始数据 X 相等
    assert_array_equal(X, pipeline.inverse_transform([[exp]]))
    # 断言获取管道当前参数是否与预期字典相等
    assert pipeline.get_params(deep=True) == {
        "steps": pipeline.steps,
        "m2": mult2,
        "m3": passthrough,
        "last": mult5,
        "memory": None,
        "m2__mult": 2,
        "last__mult": 5,
        "verbose": False,
    }

    # 设置管道中 m2 步骤为 passthrough
    pipeline.set_params(m2=passthrough)
    # 计算表达式 5 并赋值给 exp
    exp = 5
    # 断言管道进行拟合和转换后的结果是否与预期数组相等
    assert_array_equal([[exp]], pipeline.fit_transform(X, y))
    # 断言管道拟合后进行预测的结果是否与预期数组相等
    assert_array_equal([exp], pipeline.fit(X).predict(X))
    # 断言通过反向转换后的结果是否与原始数据 X 相等
    assert_array_equal(X, pipeline.inverse_transform([[exp]]))

    # 对于其他方法，确保在 None 的情况下没有 AttributeError：
    other_methods = [
        "predict_proba",
        "predict_log_proba",
        "decision_function",
        "transform",
        "score",
    ]
    for method in other_methods:
        # 调用管道对象的各种方法，确保没有 AttributeError
        getattr(pipeline, method)(X)

    # 设置管道中 m2 步骤为 mult2
    pipeline.set_params(m2=mult2)
    # 计算表达式 2 * 5 并赋值给 exp
    exp = 2 * 5
    # 断言管道进行拟合和转换后的结果是否与预期数组相等
    assert_array_equal([[exp]], pipeline.fit_transform(X, y))
    # 断言管道拟合后进行预测的结果是否与预期数组相等
    assert_array_equal([exp], pipeline.fit(X).predict(X))
    # 断言通过反向转换后的结果是否与原始数据 X 相等
    assert_array_equal(X, pipeline.inverse_transform([[exp]]))

    # 创建新的管道对象
    pipeline = make()
    # 设置管道中 last 步骤为 passthrough
    pipeline.set_params(last=passthrough)
    # 计算表达式 6 并赋值给 exp
    exp = 6
    # 断言管道拟合和转换后的结果是否与预期数组相等
    assert_array_equal([[exp]], pipeline.fit(X, y).transform(X))
    # 断言管道进行拟合和转换后的结果是否与预期数组相等
    assert_array_equal([[exp]], pipeline.fit_transform(X, y))
    # 断言通过反向转换后的结果是否与原始数据 X 相等
    assert_array_equal(X, pipeline.inverse_transform([[exp]]))

    # 定义内部错误消息和外部错误消息
    inner_msg = "'str' object has no attribute 'predict'"
    outer_msg = "This 'Pipeline' has no attribute 'predict'"
    # 使用 pytest 的上下文管理器检查是否抛出了预期的 AttributeError
    with pytest.raises(AttributeError, match=outer_msg) as exec_info:
        getattr(pipeline, "predict")
    # 断言异常的原因是 AttributeError
    assert isinstance(exec_info.value.__cause__, AttributeError)
    # 断言内部错误消息包含在异常信息中
    assert inner_msg in str(exec_info.value.__cause__)

    # 检查构造时的 'passthrough' 步骤
    exp = 2 * 5
    # 创建包含多个步骤的管道对象，并进行相关断言
    pipeline = Pipeline([("m2", mult2), ("m3", passthrough), ("last", mult5)])
    assert_array_equal([[exp]], pipeline.fit_transform(X, y))
    assert_array_equal([exp], pipeline.fit(X).predict(X))
    assert_array_equal(X, pipeline.inverse_transform([[exp]]))
def test_pipeline_ducktyping():
    # 测试使用不同类型的转换器构建管道对象

    # 第一个管道，使用 Mult(5) 转换器
    pipeline = make_pipeline(Mult(5))
    pipeline.predict  # 预测方法（未完整调用，可能有错误）
    pipeline.transform  # 转换方法（未完整调用，可能有错误）
    pipeline.inverse_transform  # 逆转换方法（未完整调用，可能有错误）

    # 第二个管道，使用 Transf() 转换器
    pipeline = make_pipeline(Transf())
    assert not hasattr(pipeline, "predict")  # 断言没有预测方法
    pipeline.transform  # 转换方法（未完整调用，可能有错误）
    pipeline.inverse_transform  # 逆转换方法（未完整调用，可能有错误）

    # 第三个管道，使用 "passthrough" 转换器
    pipeline = make_pipeline("passthrough")
    assert pipeline.steps[0] == ("passthrough", "passthrough")  # 断言第一步是 ("passthrough", "passthrough")
    assert not hasattr(pipeline, "predict")  # 断言没有预测方法
    pipeline.transform  # 转换方法（未完整调用，可能有错误）
    pipeline.inverse_transform  # 逆转换方法（未完整调用，可能有错误）

    # 第四个管道，使用 Transf() 和 NoInvTransf() 转换器
    pipeline = make_pipeline(Transf(), NoInvTransf())
    assert not hasattr(pipeline, "predict")  # 断言没有预测方法
    pipeline.transform  # 转换方法（未完整调用，可能有错误）
    assert not hasattr(pipeline, "inverse_transform")  # 断言没有逆转换方法

    # 第五个管道，使用 NoInvTransf() 和 Transf() 转换器
    pipeline = make_pipeline(NoInvTransf(), Transf())
    assert not hasattr(pipeline, "predict")  # 断言没有预测方法
    pipeline.transform  # 转换方法（未完整调用，可能有错误）
    assert not hasattr(pipeline, "inverse_transform")  # 断言没有逆转换方法


def test_make_pipeline():
    # 测试 make_pipeline 函数

    t1 = Transf()
    t2 = Transf()

    # 测试使用 t1 和 t2 构建的管道对象
    pipe = make_pipeline(t1, t2)
    assert isinstance(pipe, Pipeline)  # 断言管道对象是 Pipeline 类型
    assert pipe.steps[0][0] == "transf-1"  # 断言第一步名称为 "transf-1"
    assert pipe.steps[1][0] == "transf-2"  # 断言第二步名称为 "transf-2"

    # 测试使用 t1、t2 和 FitParamT() 构建的管道对象
    pipe = make_pipeline(t1, t2, FitParamT())
    assert isinstance(pipe, Pipeline)  # 断言管道对象是 Pipeline 类型
    assert pipe.steps[0][0] == "transf-1"  # 断言第一步名称为 "transf-1"
    assert pipe.steps[1][0] == "transf-2"  # 断言第二步名称为 "transf-2"
    assert pipe.steps[2][0] == "fitparamt"  # 断言第三步名称为 "fitparamt"


def test_feature_union_weights():
    # 测试带有转换器权重的特征合并

    # 使用 iris 数据集
    X = iris.data
    y = iris.target

    # 创建 PCA 和 SelectKBest 转换器
    pca = PCA(n_components=2, svd_solver="randomized", random_state=0)
    select = SelectKBest(k=1)

    # 使用 FeatureUnion 进行特征合并，设置 pca 的权重为 10
    fs = FeatureUnion(
        [("pca", pca), ("select", select)], transformer_weights={"pca": 10}
    )
    fs.fit(X, y)
    X_transformed = fs.transform(X)

    # 再次使用 FeatureUnion 进行特征合并，设置 pca 的权重为 10
    fs = FeatureUnion(
        [("pca", pca), ("select", select)], transformer_weights={"pca": 10}
    )
    X_fit_transformed = fs.fit_transform(X, y)

    # 使用 FeatureUnion 进行特征合并，设置 mock 的权重为 10（mock 没有 fit_transform 方法）
    fs = FeatureUnion(
        [("mock", Transf()), ("pca", pca), ("select", select)],
        transformer_weights={"mock": 10},
    )
    X_fit_transformed_wo_method = fs.fit_transform(X, y)

    # 检查预期结果

    # 使用不同的 pca 对象来控制 random_state 流
    assert_array_almost_equal(X_transformed[:, :-1], 10 * pca.fit_transform(X))
    assert_array_equal(X_transformed[:, -1], select.fit_transform(X, y).ravel())
    assert_array_almost_equal(X_fit_transformed[:, :-1], 10 * pca.fit_transform(X))
    assert_array_equal(X_fit_transformed[:, -1], select.fit_transform(X, y).ravel())
    assert X_fit_transformed_wo_method.shape == (X.shape[0], 7)


def test_feature_union_parallel():
    # 测试 FeatureUnion 的 n_jobs 参数

    X = JUNK_FOOD_DOCS

    # 使用 CountVectorizer 进行单词和字符分析的特征合并
    fs = FeatureUnion(
        [
            ("words", CountVectorizer(analyzer="word")),
            ("chars", CountVectorizer(analyzer="char")),
        ]
    )
    # 创建并行特征联合对象 `fs_parallel`，包括单词级别和字符级别的特征计数器
    fs_parallel = FeatureUnion(
        [
            ("words", CountVectorizer(analyzer="word")),
            ("chars", CountVectorizer(analyzer="char")),
        ],
        n_jobs=2,  # 使用2个并行作业来加速特征提取
    )

    # 创建另一个并行特征联合对象 `fs_parallel2`，与 `fs_parallel` 相同
    fs_parallel2 = FeatureUnion(
        [
            ("words", CountVectorizer(analyzer="word")),
            ("chars", CountVectorizer(analyzer="char")),
        ],
        n_jobs=2,  # 同样使用2个并行作业
    )

    # 使用普通特征联合对象 `fs` 对数据 `X` 进行拟合
    fs.fit(X)
    # 对数据 `X` 进行转换得到 `X_transformed`
    X_transformed = fs.transform(X)
    # 断言转换后的数据行数与原始数据行数相等
    assert X_transformed.shape[0] == len(X)

    # 使用并行特征联合对象 `fs_parallel` 对数据 `X` 进行拟合
    fs_parallel.fit(X)
    # 对数据 `X` 进行并行转换得到 `X_transformed_parallel`
    X_transformed_parallel = fs_parallel.transform(X)
    # 断言并行转换后的数据形状与普通转换后的数据形状相等
    assert X_transformed.shape == X_transformed_parallel.shape
    # 断言并行转换后的稀疏数组与普通转换后的稀疏数组相等
    assert_array_equal(X_transformed.toarray(), X_transformed_parallel.toarray())

    # 使用 `fit_transform` 方法进行并行转换，并将结果与普通转换结果相等进行断言
    X_transformed_parallel2 = fs_parallel2.fit_transform(X)
    assert_array_equal(X_transformed.toarray(), X_transformed_parallel2.toarray())

    # 断言在调用 `fit_transform` 后，转换器应该保持拟合状态
    X_transformed_parallel2 = fs_parallel2.transform(X)
    assert_array_equal(X_transformed.toarray(), X_transformed_parallel2.toarray())
# 测试特征联合（FeatureUnion）对象的特征名相关功能
def test_feature_union_feature_names():
    # 创建一个基于单词分析的计数向量化器
    word_vect = CountVectorizer(analyzer="word")
    # 创建一个基于字符 n-gram 的计数向量化器，n 取值范围为 (3, 3)
    char_vect = CountVectorizer(analyzer="char_wb", ngram_range=(3, 3))
    # 创建特征联合对象，包括字符 n-gram 和单词分析的向量化器
    ft = FeatureUnion([("chars", char_vect), ("words", word_vect)])
    # 在样本数据 JUNK_FOOD_DOCS 上拟合特征联合对象
    ft.fit(JUNK_FOOD_DOCS)
    # 获取特征联合对象的输出特征名称列表
    feature_names = ft.get_feature_names_out()
    # 遍历每个特征名，确保其包含 "chars__" 或 "words__"
    for feat in feature_names:
        assert "chars__" in feat or "words__" in feat
    # 断言特征名称列表的长度为 35
    assert len(feature_names) == 35

    # 使用自定义转换器 Transf 创建特征联合对象的测试
    ft = FeatureUnion([("tr1", Transf())]).fit([[1]])

    # 断言特征联合对象调用 get_feature_names_out() 方法时会抛出属性错误异常，错误消息被转义为正则表达式
    msg = re.escape(
        "Transformer tr1 (type Transf) does not provide get_feature_names_out"
    )
    with pytest.raises(AttributeError, match=msg):
        ft.get_feature_names_out()


# 测试管道对象的 classes_ 属性
def test_classes_property():
    # 加载鸢尾花数据集的特征 X 和目标 y
    X = iris.data
    y = iris.target

    # 创建包含特征选择和线性回归的管道对象 reg
    reg = make_pipeline(SelectKBest(k=1), LinearRegression())
    reg.fit(X, y)
    # 断言管道对象 reg 不包含属性 classes_
    with pytest.raises(AttributeError):
        getattr(reg, "classes_")

    # 创建包含特征选择和逻辑回归的管道对象 clf
    clf = make_pipeline(SelectKBest(k=1), LogisticRegression(random_state=0))
    with pytest.raises(AttributeError):
        getattr(clf, "classes_")
    clf.fit(X, y)
    # 断言管道对象 clf 的 classes_ 属性与目标 y 的唯一值相等
    assert_array_equal(clf.classes_, np.unique(y))


# 测试设置特征联合对象的步骤
def test_set_feature_union_steps():
    # 创建自定义转换器对象 Mult，分别使用不同的乘法因子
    mult2 = Mult(2)
    mult3 = Mult(3)
    mult5 = Mult(5)

    # 定义每个 Mult 对象的 get_feature_names_out 方法为 lambda 表达式，返回相应特征名称
    mult3.get_feature_names_out = lambda input_features: ["x3"]
    mult2.get_feature_names_out = lambda input_features: ["x2"]
    mult5.get_feature_names_out = lambda input_features: ["x5"]

    # 创建特征联合对象 ft，包含 m2 和 m3 两个 Mult 转换器
    ft = FeatureUnion([("m2", mult2), ("m3", mult3)])
    # 断言特征联合对象 ft 在转换样本数据 [[1]] 后的输出与预期相等
    assert_array_equal([[2, 3]], ft.transform(np.asarray([[1]])))
    # 断言特征联合对象 ft 的 get_feature_names_out 方法输出的特征名列表与预期相等
    assert_array_equal(["m2__x2", "m3__x3"], ft.get_feature_names_out())

    # 直接设置特征联合对象 ft 的 transformer_list 属性，替换为只包含 m5 的 Mult 转换器
    ft.transformer_list = [("m5", mult5)]
    # 断言特征联合对象 ft 在转换样本数据 [[1]] 后的输出与预期相等
    assert_array_equal([[5]], ft.transform(np.asarray([[1]])))
    # 断言特征联合对象 ft 的 get_feature_names_out 方法输出的特征名列表与预期相等
    assert_array_equal(["m5__x5"], ft.get_feature_names_out())

    # 使用 set_params 方法替换特征联合对象 ft 的 transformer_list 属性为 [("mock", mult3)]
    ft.set_params(transformer_list=[("mock", mult3)])
    # 断言特征联合对象 ft 在转换样本数据 [[1]] 后的输出与预期相等
    assert_array_equal([[3]], ft.transform(np.asarray([[1]])))
    # 断言特征联合对象 ft 的 get_feature_names_out 方法输出的特征名列表与预期相等
    assert_array_equal(["mock__x3"], ft.get_feature_names_out())

    # 使用 set_params 方法替换特征联合对象 ft 的 mock 属性为 mult5
    ft.set_params(mock=mult5)
    # 断言特征联合对象 ft 在转换样本数据 [[1]] 后的输出与预期相等
    assert_array_equal([[5]], ft.transform(np.asarray([[1]])))
    # 断言特征联合对象 ft 的 get_feature_names_out 方法输出的特征名列表与预期相等
    assert_array_equal(["mock__x5"], ft.get_feature_names_out())


# 测试设置特征联合对象步骤删除的情况
def test_set_feature_union_step_drop():
    # 创建自定义转换器对象 Mult，分别使用不同的乘法因子
    mult2 = Mult(2)
    mult3 = Mult(3)

    # 定义每个 Mult 对象的 get_feature_names_out 方法为 lambda 表达式，返回相应特征名称
    mult2.get_feature_names_out = lambda input_features: ["x2"]
    mult3.get_feature_names_out = lambda input_features: ["x3"]

    # 创建特征联合对象 ft，包含 m2 和 m3 两个 Mult 转换器
    X = np.asarray([[1]])
    ft = FeatureUnion([("m2", mult2), ("m3", mult3)])
    # 断言特征联合对象 ft 在拟合和转换样本数据 X 后的输出与预期相等
    assert_array_equal([[2, 3]], ft.fit(X).transform(X))
    assert_array_equal([[2, 3]], ft.fit_transform(X))
    # 断言特征联合对象 ft 的 get_feature_names_out 方法输出的特征名列表与预期相等
    assert_array_equal(["m2__x2", "m3__x3"], ft.get_feature_names_out())

    # 使用 set_params 方法将特征联合对象 ft 的 m2 步骤设置为 "drop"
    ft.set_params(m2="drop")
    # 断言特征联合对象 ft 在拟合和转换样本数据 X 后的输出与预期相等
    assert_array_equal([[3]], ft.fit(X).transform(X))
    assert_array_equal([[3]], ft.fit_transform(X))
    # 断言特征联合对象 ft 的 get_feature_names_out 方法输出的特征名列表与预期相等
    assert_array_equal(["m3__x3"], ft.get_feature_names_out())

    # 使用 set_params 方法将特征联合对象 ft 的 m3 步骤设置为 "drop"
    ft.set_params(m3="drop")
    # 断言特征联合对象 ft 在拟合和转换样本数据 X 后的输出与预期相等
    assert_array_equal([[]], ft.fit(X).transform(X))
    # 使用 assert_array_equal 断言函数验证 ft.fit_transform(X) 返回的结果与预期的空数组[[]]相等
    assert_array_equal([[]], ft.fit_transform(X))
    
    # 使用 assert_array_equal 断言函数验证 ft.get_feature_names_out() 返回的结果与预期的空列表[]相等
    assert_array_equal([], ft.get_feature_names_out())
    
    # 检查我们可以重新设置参数的情况
    # 使用 ft.set_params(m3=mult3) 设置 FeatureUnion 对象 ft 的参数 m3 为 mult3
    ft.set_params(m3=mult3)
    # 使用 assert_array_equal 断言函数验证 ft.fit(X).transform(X) 返回的结果与预期的数组[[3]]相等
    assert_array_equal([[3]], ft.fit(X).transform(X))
    
    # 在构造时检查 'drop' 步骤
    # 使用 FeatureUnion 创建对象 ft，包含两个转换器: ("m2", "drop") 和 ("m3", mult3)
    ft = FeatureUnion([("m2", "drop"), ("m3", mult3)])
    # 使用 assert_array_equal 断言函数验证 ft.fit(X).transform(X) 返回的结果与预期的数组[[3]]相等
    assert_array_equal([[3]], ft.fit(X).transform(X))
    # 使用 assert_array_equal 断言函数验证 ft.fit_transform(X) 返回的结果与预期的数组[[3]]相等
    assert_array_equal([[3]], ft.fit_transform(X))
    # 使用 assert_array_equal 断言函数验证 ft.get_feature_names_out() 返回的结果包含预期的特征名称["m3__x3"]
    assert_array_equal(["m3__x3"], ft.get_feature_names_out())
# 定义测试函数，用于验证设置转换器为 `"passthrough"` 的行为
def test_set_feature_union_passthrough():
    """Check the behaviour of setting a transformer to `"passthrough"`."""
    # 创建两个乘法转换器实例，分别乘以2和3
    mult2 = Mult(2)
    mult3 = Mult(3)

    # 重写乘法转换器的 get_feature_names_out 方法，返回指定的特征名称
    mult2.get_feature_names_out = lambda input_features: ["x2"]
    mult3.get_feature_names_out = lambda input_features: ["x3"]

    # 创建一个包含两个转换器的 FeatureUnion 实例
    X = np.asarray([[1]])
    ft = FeatureUnion([("m2", mult2), ("m3", mult3)])

    # 断言转换后的特征数组与预期相等
    assert_array_equal([[2, 3]], ft.fit(X).transform(X))
    assert_array_equal([[2, 3]], ft.fit_transform(X))
    assert_array_equal(["m2__x2", "m3__x3"], ft.get_feature_names_out())

    # 设置 m2 转换器为 "passthrough"，断言转换后的特征数组与预期相等
    ft.set_params(m2="passthrough")
    assert_array_equal([[1, 3]], ft.fit(X).transform(X))
    assert_array_equal([[1, 3]], ft.fit_transform(X))
    assert_array_equal(["m2__myfeat", "m3__x3"], ft.get_feature_names_out(["myfeat"]))

    # 设置 m3 转换器为 "passthrough"，断言转换后的特征数组与预期相等
    ft.set_params(m3="passthrough")
    assert_array_equal([[1, 1]], ft.fit(X).transform(X))
    assert_array_equal([[1, 1]], ft.fit_transform(X))
    assert_array_equal(["m2__myfeat", "m3__myfeat"], ft.get_feature_names_out(["myfeat"]))

    # 检查可以重新设置回转换器实例
    ft.set_params(m3=mult3)
    assert_array_equal([[1, 3]], ft.fit(X).transform(X))
    assert_array_equal([[1, 3]], ft.fit_transform(X))
    assert_array_equal(["m2__myfeat", "m3__x3"], ft.get_feature_names_out(["myfeat"]))

    # 检查构造时使用 'passthrough' 的步骤
    ft = FeatureUnion([("m2", "passthrough"), ("m3", mult3)])
    assert_array_equal([[1, 3]], ft.fit(X).transform(X))
    assert_array_equal([[1, 3]], ft.fit_transform(X))
    assert_array_equal(["m2__myfeat", "m3__x3"], ft.get_feature_names_out(["myfeat"]))

    # 使用 iris 数据集的 Principal Component Analysis (PCA) 进行特征合并
    X = iris.data
    columns = X.shape[1]
    pca = PCA(n_components=2, svd_solver="randomized", random_state=0)

    ft = FeatureUnion([("passthrough", "passthrough"), ("pca", pca)])

    # 断言转换后的特征数组与预期相等
    assert_array_equal(X, ft.fit(X).transform(X)[:, :columns])
    assert_array_equal(X, ft.fit_transform(X)[:, :columns])

    # 断言获取特征名称与预期相等
    assert_array_equal(
        [
            "passthrough__f0",
            "passthrough__f1",
            "passthrough__f2",
            "passthrough__f3",
            "pca__pca0",
            "pca__pca1",
        ],
        ft.get_feature_names_out(["f0", "f1", "f2", "f3"]),
    )

    # 将 PCA 设置为 "passthrough"，断言转换后的特征数组与预期相等
    ft.set_params(pca="passthrough")
    X_ft = ft.fit(X).transform(X)
    assert_array_equal(X_ft, np.hstack([X, X]))
    X_ft = ft.fit_transform(X)
    assert_array_equal(X_ft, np.hstack([X, X]))

    # 断言获取特征名称与预期相等
    assert_array_equal(
        [
            "passthrough__f0",
            "passthrough__f1",
            "passthrough__f2",
            "passthrough__f3",
            "pca__f0",
            "pca__f1",
            "pca__f2",
            "pca__f3",
        ],
        ft.get_feature_names_out(["f0", "f1", "f2", "f3"]),
    )

    # 将 'passthrough' 设置为 PCA，断言转换后的特征数组与预期相等
    ft.set_params(passthrough=pca)
    assert_array_equal(X, ft.fit(X).transform(X)[:, -columns:])
    # 使用 assert_array_equal 函数检查 X 是否与 ft.fit_transform(X)[:, -columns:] 相等
    assert_array_equal(X, ft.fit_transform(X)[:, -columns:])
    
    # 使用 assert_array_equal 函数检查生成的特征名列表是否与预期列表相等
    assert_array_equal(
        [
            "passthrough__pca0",
            "passthrough__pca1",
            "pca__f0",
            "pca__f1",
            "pca__f2",
            "pca__f3",
        ],
        ft.get_feature_names_out(["f0", "f1", "f2", "f3"]),
    )
    
    # 创建 FeatureUnion 对象 ft，包含两个转换器：一个是 "passthrough"，一个是 pca
    # 并设置 "passthrough" 转换器的权重为 2
    ft = FeatureUnion(
        [("passthrough", "passthrough"), ("pca", pca)],
        transformer_weights={"passthrough": 2},
    )
    
    # 使用 assert_array_equal 函数检查 X 的两倍是否与 ft.fit(X).transform(X)[:, :columns] 相等
    assert_array_equal(X * 2, ft.fit(X).transform(X)[:, :columns])
    
    # 使用 assert_array_equal 函数检查 X 的两倍是否与 ft.fit_transform(X)[:, :columns] 相等
    assert_array_equal(X * 2, ft.fit_transform(X)[:, :columns])
    
    # 使用 assert_array_equal 函数检查生成的特征名列表是否与预期列表相等
    assert_array_equal(
        [
            "passthrough__f0",
            "passthrough__f1",
            "passthrough__f2",
            "passthrough__f3",
            "pca__pca0",
            "pca__pca1",
        ],
        ft.get_feature_names_out(["f0", "f1", "f2", "f3"]),
    )
def test_feature_union_passthrough_get_feature_names_out_true():
    """Check feature_names_out for verbose_feature_names_out=True (default)"""
    # 使用 iris 数据集作为输入特征矩阵 X
    X = iris.data
    # 创建 PCA 模型，设置参数：2个主成分，随机化方法为 randomized，随机种子为 0
    pca = PCA(n_components=2, svd_solver="randomized", random_state=0)

    # 创建 FeatureUnion 对象，包含两个转换器："pca" 和 "passthrough"
    ft = FeatureUnion([("pca", pca), ("passthrough", "passthrough")])
    # 对输入数据 X 进行拟合
    ft.fit(X)
    # 断言获取的输出特征名称列表是否符合预期
    assert_array_equal(
        [
            "pca__pca0",
            "pca__pca1",
            "passthrough__x0",
            "passthrough__x1",
            "passthrough__x2",
            "passthrough__x3",
        ],
        ft.get_feature_names_out(),
    )


def test_feature_union_passthrough_get_feature_names_out_false():
    """Check feature_names_out for verbose_feature_names_out=False"""
    # 使用 iris 数据集作为输入特征矩阵 X
    X = iris.data
    # 创建 PCA 模型，设置参数：2个主成分，随机化方法为 randomized，随机种子为 0
    pca = PCA(n_components=2, svd_solver="randomized", random_state=0)

    # 创建 FeatureUnion 对象，包含两个转换器："pca" 和 "passthrough"，并设定 verbose_feature_names_out=False
    ft = FeatureUnion(
        [("pca", pca), ("passthrough", "passthrough")], verbose_feature_names_out=False
    )
    # 对输入数据 X 进行拟合
    ft.fit(X)
    # 断言获取的输出特征名称列表是否符合预期
    assert_array_equal(
        [
            "pca0",
            "pca1",
            "x0",
            "x1",
            "x2",
            "x3",
        ],
        ft.get_feature_names_out(),
    )


def test_feature_union_passthrough_get_feature_names_out_false_errors():
    """Check get_feature_names_out and non-verbose names and colliding names."""
    # 导入 pandas 库，如果不存在则跳过测试
    pd = pytest.importorskip("pandas")
    # 创建一个简单的 DataFrame X，包含两行数据，列名为 ["a", "b"]
    X = pd.DataFrame([[1, 2], [2, 3]], columns=["a", "b"])

    # 创建 FunctionTransformer 对象 select_a，通过 lambda 函数选择特征 "a"，并设定输出特征名为 "a"
    select_a = FunctionTransformer(
        lambda X: X[["a"]], feature_names_out=lambda self, _: np.asarray(["a"])
    )
    # 创建 FeatureUnion 对象 union，包含两个转换器："t1" 对应 StandardScaler()，"t2" 对应 select_a
    union = FeatureUnion(
        [("t1", StandardScaler()), ("t2", select_a)],
        verbose_feature_names_out=False,
    )
    # 对输入数据 X 进行拟合
    union.fit(X)

    # 设置预期的错误消息
    msg = re.escape(
        "Output feature names: ['a'] are not unique. "
        "Please set verbose_feature_names_out=True to add prefixes to feature names"
    )

    # 使用断言检查是否抛出预期的 ValueError 异常，并且异常消息符合预期
    with pytest.raises(ValueError, match=msg):
        union.get_feature_names_out()


def test_feature_union_passthrough_get_feature_names_out_false_errors_overlap_over_5():
    """Check get_feature_names_out with non-verbose names and >= 5 colliding names."""
    # 导入 pandas 库，如果不存在则跳过测试
    pd = pytest.importorskip("pandas")
    # 创建一个 DataFrame X，包含一行数据，列名为 ["f0", "f1", ..., "f9"]
    X = pd.DataFrame([list(range(10))], columns=[f"f{i}" for i in range(10)])

    # 创建 FeatureUnion 对象 union，包含两个转换器："t1" 和 "t2"，均为 "passthrough"
    union = FeatureUnion(
        [("t1", "passthrough"), ("t2", "passthrough")],
        verbose_feature_names_out=False,
    )

    # 对输入数据 X 进行拟合
    union.fit(X)

    # 设置预期的错误消息
    msg = re.escape(
        "Output feature names: ['f0', 'f1', 'f2', 'f3', 'f4', ...] "
        "are not unique. Please set verbose_feature_names_out=True to add prefixes to"
        " feature names"
    )

    # 使用断言检查是否抛出预期的 ValueError 异常，并且异常消息符合预期
    with pytest.raises(ValueError, match=msg):
        union.get_feature_names_out()


def test_step_name_validation():
    error_message_1 = r"Estimator names must not contain __: got \['a__q'\]"
    error_message_2 = r"Names provided are not unique: \['a', 'a'\]"
    error_message_3 = r"Estimator names conflict with constructor arguments: \['%s'\]"
    # 创建一个不合法的步骤列表，包含一个带有 "__" 的名称和重复的名称
    bad_steps1 = [("a__q", Mult(2)), ("b", Mult(3))]
    # 定义一个包含多个无效步骤的列表，每个元素是一个元组，包含类和参数的组合
    bad_steps2 = [("a", Mult(2)), ("a", Mult(3))]
    # 遍历包含类和参数名称的列表
    for cls, param in [(Pipeline, "steps"), (FeatureUnion, "transformer_list")]:
        # 在构造过程中验证无效步骤，尽管这违反了scikit-learn的惯例
        # 创建一个新的包含无效步骤的列表，其中一个步骤的参数名与当前循环的param相同
        bad_steps3 = [("a", Mult(2)), (param, Mult(3))]
        # 遍历包含不同无效步骤列表和相应错误消息的元组
        for bad_steps, message in [
            (bad_steps1, error_message_1),
            (bad_steps2, error_message_2),
            (bad_steps3, error_message_3 % param),
        ]:
            # 使用pytest模块的raises方法验证在构造过程中抛出ValueError异常
            with pytest.raises(ValueError, match=message):
                cls(**{param: bad_steps}).fit([[1]], [1])

            # 使用setattr方法设置一个参数为单个有效步骤的est对象，并验证在setattr后设置无效步骤会抛出ValueError异常
            est = cls(**{param: [("a", Mult(1))]})
            setattr(est, param, bad_steps)
            with pytest.raises(ValueError, match=message):
                est.fit([[1]], [1])

            # 使用set_params方法设置一个参数为单个有效步骤的est对象，并验证在set_params后设置无效步骤会抛出ValueError异常
            est = cls(**{param: [("a", Mult(1))]})
            est.set_params(**{param: bad_steps})
            with pytest.raises(ValueError, match=message):
                est.fit([[1]], [1])
# 定义测试函数 test_set_params_nested_pipeline，用于测试设置嵌套管道的参数
def test_set_params_nested_pipeline():
    # 创建管道估计器，其中包含一个内部管道 'a'，内部管道 'a' 包含一个伪回归器 'b'
    estimator = Pipeline([("a", Pipeline([("b", DummyRegressor())]))])
    # 设置 'a__b' 内部管道的参数 'alpha' 为 0.001，并将 'a__b' 替换为一个 Lasso 回归器
    estimator.set_params(a__b__alpha=0.001, a__b=Lasso())
    # 更新 'a' 管道的步骤，将 'b' 替换为逻辑回归器，并设置 'a__b' 的参数 'C' 为 5
    estimator.set_params(a__steps=[("b", LogisticRegression())], a__b__C=5)


# 定义测试函数 test_pipeline_memory，用于测试管道的内存使用情况
def test_pipeline_memory():
    # 加载鸢尾花数据集的特征数据 X 和目标数据 y
    X = iris.data
    y = iris.target
    # 创建临时目录作为缓存目录
    cachedir = mkdtemp()
    try:
        # 设置内存缓存，用于存储中间结果以提高运行速度，输出详细日志
        memory = joblib.Memory(location=cachedir, verbose=10)
        # 创建支持概率估计的支持向量分类器
        clf = SVC(probability=True, random_state=0)
        # 创建虚拟的转换器
        transf = DummyTransf()
        # 创建包含转换器和分类器的管道
        pipe = Pipeline([("transf", clone(transf)), ("svc", clf)])
        # 创建带有内存缓存的管道
        cached_pipe = Pipeline([("transf", transf), ("svc", clf)], memory=memory)

        # 在第一次拟合时对转换器进行记忆化处理
        cached_pipe.fit(X, y)
        pipe.fit(X, y)
        # 获取缓存管道中转换器的时间戳
        ts = cached_pipe.named_steps["transf"].timestamp_
        # 检查缓存管道和普通管道产生相同的预测结果
        assert_array_equal(pipe.predict(X), cached_pipe.predict(X))
        assert_array_equal(pipe.predict_proba(X), cached_pipe.predict_proba(X))
        assert_array_equal(pipe.predict_log_proba(X), cached_pipe.predict_log_proba(X))
        assert_array_equal(pipe.score(X, y), cached_pipe.score(X, y))
        assert_array_equal(
            pipe.named_steps["transf"].means_, cached_pipe.named_steps["transf"].means_
        )
        assert not hasattr(transf, "means_")
        # 第二次拟合时检查是否从缓存中读取
        cached_pipe.fit(X, y)
        # 再次检查缓存管道和普通管道产生相同的预测结果
        assert_array_equal(pipe.predict(X), cached_pipe.predict(X))
        assert_array_equal(pipe.predict_proba(X), cached_pipe.predict_proba(X))
        assert_array_equal(pipe.predict_log_proba(X), cached_pipe.predict_log_proba(X))
        assert_array_equal(pipe.score(X, y), cached_pipe.score(X, y))
        assert_array_equal(
            pipe.named_steps["transf"].means_, cached_pipe.named_steps["transf"].means_
        )
        assert ts == cached_pipe.named_steps["transf"].timestamp_
        # 使用克隆的估计器创建新的管道
        # 检查即使更改步骤名称也不影响缓存命中
        clf_2 = SVC(probability=True, random_state=0)
        transf_2 = DummyTransf()
        cached_pipe_2 = Pipeline(
            [("transf_2", transf_2), ("svc", clf_2)], memory=memory
        )
        cached_pipe_2.fit(X, y)

        # 再次检查缓存管道和普通管道产生相同的预测结果
        assert_array_equal(pipe.predict(X), cached_pipe_2.predict(X))
        assert_array_equal(pipe.predict_proba(X), cached_pipe_2.predict_proba(X))
        assert_array_equal(
            pipe.predict_log_proba(X), cached_pipe_2.predict_log_proba(X)
        )
        assert_array_equal(pipe.score(X, y), cached_pipe_2.score(X, y))
        assert_array_equal(
            pipe.named_steps["transf"].means_,
            cached_pipe_2.named_steps["transf_2"].means_,
        )
        assert ts == cached_pipe_2.named_steps["transf_2"].timestamp_
    finally:
        # 最终清理缓存目录
        shutil.rmtree(cachedir)
# 定义一个测试函数，用于测试具有内存缓存的机器学习管道
def test_make_pipeline_memory():
    # 创建临时缓存目录
    cachedir = mkdtemp()
    # 创建带有内存缓存的joblib Memory对象
    memory = joblib.Memory(location=cachedir, verbose=10)
    # 创建包含DummyTransf(), SVC()两个步骤的机器学习管道，并指定内存为memory
    pipeline = make_pipeline(DummyTransf(), SVC(), memory=memory)
    # 断言管道的内存是之前创建的memory对象
    assert pipeline.memory is memory
    # 创建不带内存缓存的机器学习管道
    pipeline = make_pipeline(DummyTransf(), SVC())
    # 断言管道的内存为None，即没有使用内存缓存
    assert pipeline.memory is None
    # 断言管道步骤的数量为2
    assert len(pipeline) == 2

    # 删除临时缓存目录
    shutil.rmtree(cachedir)


# 自定义的特征名保存器类，继承自BaseEstimator
class FeatureNameSaver(BaseEstimator):
    # 实现fit方法，用于设置特征名
    def fit(self, X, y=None):
        self._check_feature_names(X, reset=True)
        return self

    # 实现transform方法，简单返回输入数据X
    def transform(self, X, y=None):
        return X

    # 实现get_feature_names_out方法，返回输入的特征名
    def get_feature_names_out(self, input_features=None):
        return input_features


# 测试特征名通过流传递的情况
def test_features_names_passthrough():
    """Check pipeline.get_feature_names_out with passthrough"""
    # 创建Pipeline对象，包含特征名保存器、直接流传递、LogisticRegression分类器三个步骤
    pipe = Pipeline(
        steps=[
            ("names", FeatureNameSaver()),
            ("pass", "passthrough"),
            ("clf", LogisticRegression()),
        ]
    )
    # 载入鸢尾花数据集
    iris = load_iris()
    # 对Pipeline进行训练
    pipe.fit(iris.data, iris.target)
    # 断言输出的特征名与输入的特征名相等
    assert_array_equal(
        pipe[:-1].get_feature_names_out(iris.feature_names), iris.feature_names
    )


# 测试CountVectorizer向量化器的特征名输出
def test_feature_names_count_vectorizer():
    """Check pipeline.get_feature_names_out with vectorizers"""
    # 创建Pipeline对象，包含CountVectorizer和LogisticRegression分类器两个步骤
    pipe = Pipeline(steps=[("vect", CountVectorizer()), ("clf", LogisticRegression())])
    # 定义JUNK_FOOD_DOCS的标签，是否包含"pizza"
    y = ["pizza" in x for x in JUNK_FOOD_DOCS]
    # 对Pipeline进行训练
    pipe.fit(JUNK_FOOD_DOCS, y)
    # 断言输出的特征名列表与预期列表相等
    assert_array_equal(
        pipe[:-1].get_feature_names_out(),
        ["beer", "burger", "coke", "copyright", "pizza", "the"],
    )
    # 再次断言输出的特征名列表与预期列表相等，输入参数无效
    assert_array_equal(
        pipe[:-1].get_feature_names_out("nonsense_is_ignored"),
        ["beer", "burger", "coke", "copyright", "pizza", "the"],
    )


# 测试管道特征名输出时，当转换器未定义get_feature_names_out方法时抛出错误
def test_pipeline_feature_names_out_error_without_definition():
    """Check that error is raised when a transformer does not define
    `get_feature_names_out`."""
    # 创建Pipeline对象，仅包含一个未定义get_feature_names_out方法的转换器NoTrans
    pipe = Pipeline(steps=[("notrans", NoTrans())])
    # 载入鸢尾花数据集
    iris = load_iris()
    # 对Pipeline进行训练
    pipe.fit(iris.data, iris.target)

    # 断言捕获到AttributeError异常，且异常消息包含指定字符串
    msg = "does not provide get_feature_names_out"
    with pytest.raises(AttributeError, match=msg):
        pipe.get_feature_names_out()


# 测试管道参数错误时的异常处理
def test_pipeline_param_error():
    # 创建包含LogisticRegression分类器的Pipeline对象
    clf = make_pipeline(LogisticRegression())
    # 断言调用Pipeline.fit方法时，传入sample_weight参数会抛出ValueError异常
    with pytest.raises(
        ValueError, match="Pipeline.fit does not accept the sample_weight parameter"
    ):
        clf.fit([[0], [0]], [0, 1], sample_weight=[1, 1])


# 定义用于测试的参数网格，参数是一个元组列表
parameter_grid_test_verbose = (
    (est, pattern, method)
    # 继续填充
    # 使用 itertools.product 生成器，生成多个元组 (est, pattern) 和 method 的组合
    for (est, pattern), method in itertools.product(
        [
            (
                Pipeline([("transf", Transf()), ("clf", FitParamT())]),
                r"\[Pipeline\].*\(step 1 of 2\) Processing transf.* total=.*\n"
                r"\[Pipeline\].*\(step 2 of 2\) Processing clf.* total=.*\n$",
            ),
            (
                Pipeline([("transf", Transf()), ("noop", None), ("clf", FitParamT())]),
                r"\[Pipeline\].*\(step 1 of 3\) Processing transf.* total=.*\n"
                r"\[Pipeline\].*\(step 2 of 3\) Processing noop.* total=.*\n"
                r"\[Pipeline\].*\(step 3 of 3\) Processing clf.* total=.*\n$",
            ),
            (
                Pipeline(
                    [
                        ("transf", Transf()),
                        ("noop", "passthrough"),
                        ("clf", FitParamT()),
                    ]
                ),
                r"\[Pipeline\].*\(step 1 of 3\) Processing transf.* total=.*\n"
                r"\[Pipeline\].*\(step 2 of 3\) Processing noop.* total=.*\n"
                r"\[Pipeline\].*\(step 3 of 3\) Processing clf.* total=.*\n$",
            ),
            (
                Pipeline([("transf", Transf()), ("clf", None)]),
                r"\[Pipeline\].*\(step 1 of 2\) Processing transf.* total=.*\n"
                r"\[Pipeline\].*\(step 2 of 2\) Processing clf.* total=.*\n$",
            ),
            (
                Pipeline([("transf", None), ("mult", Mult())]),
                r"\[Pipeline\].*\(step 1 of 2\) Processing transf.* total=.*\n"
                r"\[Pipeline\].*\(step 2 of 2\) Processing mult.* total=.*\n$",
            ),
            (
                Pipeline([("transf", "passthrough"), ("mult", Mult())]),
                r"\[Pipeline\].*\(step 1 of 2\) Processing transf.* total=.*\n"
                r"\[Pipeline\].*\(step 2 of 2\) Processing mult.* total=.*\n$",
            ),
            (
                FeatureUnion([("mult1", Mult()), ("mult2", Mult())]),
                r"\[FeatureUnion\].*\(step 1 of 2\) Processing mult1.* total=.*\n"
                r"\[FeatureUnion\].*\(step 2 of 2\) Processing mult2.* total=.*\n$",
            ),
            (
                FeatureUnion([("mult1", "drop"), ("mult2", Mult()), ("mult3", "drop")]),
                r"\[FeatureUnion\].*\(step 1 of 1\) Processing mult2.* total=.*\n$",
            ),
        ],
        ["fit", "fit_transform", "fit_predict"],  # method 列表包含适用的方法名称
    )
    # 选择具有 method 方法的 est，并且不符合特定条件的 est
    if hasattr(est, method)
    and not (
        method == "fit_transform"  # 如果 method 是 fit_transform
        and hasattr(est, "steps")  # 并且 est 具有 steps 属性
        and isinstance(est.steps[-1][1], FitParamT)  # 并且 est 的最后一个步骤是 FitParamT 类型
    )
# 使用 pytest.mark.parametrize 装饰器指定参数化测试的参数，这些参数来自于 parameter_grid_test_verbose
@pytest.mark.parametrize("est, pattern, method", parameter_grid_test_verbose)
def test_verbose(est, method, pattern, capsys):
    # 获取 est 对象中指定方法的可调用函数对象
    func = getattr(est, method)

    # 定义输入特征 X 和目标值 y
    X = [[1, 2, 3], [4, 5, 6]]
    y = [[7], [8]]

    # 设置估计器 est 的 verbose 参数为 False
    est.set_params(verbose=False)
    # 调用指定方法 func 执行估计器的训练过程，捕获标准输出和错误
    func(X, y)
    # 断言在 verbose=False 时，标准输出没有内容
    assert not capsys.readouterr().out, "Got output for verbose=False"

    # 设置估计器 est 的 verbose 参数为 True
    est.set_params(verbose=True)
    # 再次调用指定方法 func 执行估计器的训练过程，捕获标准输出和错误
    func(X, y)
    # 使用正则表达式匹配模式 pattern 来断言标准输出的内容
    assert re.match(pattern, capsys.readouterr().out)


def test_n_features_in_pipeline():
    # 确保管道能够委托 n_features_in 属性给其第一个步骤

    # 定义输入特征 X 和目标值 y
    X = [[1, 2], [3, 4], [5, 6]]
    y = [0, 1, 2]

    # 创建标准化器和梯度增强决策树分类器
    ss = StandardScaler()
    gbdt = HistGradientBoostingClassifier()
    # 创建管道，依次使用标准化器和梯度增强决策树分类器
    pipe = make_pipeline(ss, gbdt)
    # 断言管道对象没有 n_features_in_ 属性
    assert not hasattr(pipe, "n_features_in_")
    # 对管道进行拟合操作
    pipe.fit(X, y)
    # 断言管道的 n_features_in_ 属性等于标准化器的 n_features_in_ 属性，且都为 2
    assert pipe.n_features_in_ == ss.n_features_in_ == 2

    # 如果第一个步骤具有 n_features_in_ 属性，则管道也会具有该属性，即使尚未拟合
    ss = StandardScaler()
    gbdt = HistGradientBoostingClassifier()
    # 创建新的管道，依次使用标准化器和梯度增强决策树分类器
    pipe = make_pipeline(ss, gbdt)
    # 对标准化器进行拟合操作
    ss.fit(X, y)
    # 断言管道的 n_features_in_ 属性等于标准化器的 n_features_in_ 属性，且都为 2
    assert pipe.n_features_in_ == ss.n_features_in_ == 2
    # 断言梯度增强决策树分类器没有 n_features_in_ 属性
    assert not hasattr(gbdt, "n_features_in_")


def test_n_features_in_feature_union():
    # 确保特征合并能够委托 n_features_in 属性给其第一个转换器

    # 定义输入特征 X 和目标值 y
    X = [[1, 2], [3, 4], [5, 6]]
    y = [0, 1, 2]

    # 创建标准化器
    ss = StandardScaler()
    # 创建特征合并对象，其中包含标准化器
    fu = make_union(ss)
    # 断言特征合并对象没有 n_features_in_ 属性
    assert not hasattr(fu, "n_features_in_")
    # 对特征合并对象进行拟合操作
    fu.fit(X, y)
    # 断言特征合并对象的 n_features_in_ 属性等于标准化器的 n_features_in_ 属性，且都为 2
    assert fu.n_features_in_ == ss.n_features_in_ == 2

    # 如果第一个步骤具有 n_features_in_ 属性，则特征合并对象也会具有该属性，即使尚未拟合
    ss = StandardScaler()
    # 创建新的特征合并对象，其中包含标准化器
    fu = make_union(ss)
    # 对标准化器进行拟合操作
    ss.fit(X, y)
    # 断言特征合并对象的 n_features_in_ 属性等于标准化器的 n_features_in_ 属性，且都为 2
    assert fu.n_features_in_ == ss.n_features_in_ == 2


def test_feature_union_fit_params():
    # 测试问题回归：#15117

    # 定义一个虚拟的转换器类，用于测试
    class DummyTransformer(TransformerMixin, BaseEstimator):
        def fit(self, X, y=None, **fit_params):
            # 如果传入的 fit_params 不等于 {"a": 0}，则抛出 ValueError
            if fit_params != {"a": 0}:
                raise ValueError
            return self

        def transform(self, X, y=None):
            return X

    # 使用 iris 数据集
    X, y = iris.data, iris.target
    # 创建特征合并对象，包含两个虚拟的 DummyTransformer 转换器
    t = FeatureUnion([("dummy0", DummyTransformer()), ("dummy1", DummyTransformer())])
    # 断言对特征合并对象的拟合操作会抛出 ValueError 异常，因为没有传入正确的 fit_params
    with pytest.raises(ValueError):
        t.fit(X, y)

    # 同样的测试，使用 fit_transform 方法
    with pytest.raises(ValueError):
        t.fit_transform(X, y)

    # 使用正确的 fit_params 对特征合并对象进行拟合操作，不应该抛出异常
    t.fit(X, y, a=0)
    t.fit_transform(X, y, a=0)


def test_feature_union_fit_params_without_fit_transform():
    # 当 SLEP6 未启用时，测试正确传递元数据给不实现 `fit_transform` 方法的底层转换器

    # 定义一个虚拟的转换器类，继承自 ConsumingNoFitTransformTransformer 类
    class DummyTransformer(ConsumingNoFitTransformTransformer):
        def fit(self, X, y=None, **fit_params):
            # 如果传入的 fit_params 不等于 {"metadata": 1}，则抛出 ValueError
            if fit_params != {"metadata": 1}:
                raise ValueError
            return self

    # 使用 iris 数据集
    X, y = iris.data, iris.target
    # 创建一个 FeatureUnion 对象 t，用于将多个转换器的输出合并成单个特征空间
    t = FeatureUnion(
        [
            ("nofittransform0", DummyTransformer()),  # 使用 DummyTransformer 创建一个无需拟合和转换的转换器
            ("nofittransform1", DummyTransformer()),  # 使用 DummyTransformer 创建另一个无需拟合和转换的转换器
        ]
    )
    
    # 使用 pytest 模块中的 pytest.raises 来断言是否抛出特定的 ValueError 异常
    with pytest.raises(ValueError):
        # 调用 t 的 fit_transform 方法，传入 X, y 和 metadata=0，期望抛出 ValueError 异常
        t.fit_transform(X, y, metadata=0)
    
    # 调用 t 的 fit_transform 方法，传入 X, y 和 metadata=1，进行拟合和转换操作
    t.fit_transform(X, y, metadata=1)
def test_pipeline_missing_values_leniency():
    # 检查管道是否将缺失值验证委托给底层的转换器和预测器。
    X, y = iris.data, iris.target
    # 创建一个与 X 形状相同的布尔掩码，其中大约10%的值被设置为 NaN
    mask = np.random.choice([1, 0], X.shape, p=[0.1, 0.9]).astype(bool)
    X[mask] = np.nan
    # 创建管道，包括简单填充器和逻辑回归模型
    pipe = make_pipeline(SimpleImputer(), LogisticRegression())
    # 断言管道拟合后的得分大于0.4
    assert pipe.fit(X, y).score(X, y) > 0.4


def test_feature_union_warns_unknown_transformer_weight():
    # 当 transformer_weights 包含 transformer_list 中不存在的键时，向用户发出警告
    X = [[1, 2], [3, 4], [5, 6]]
    y = [0, 1, 2]

    transformer_list = [("transf", Transf())]
    # 使用不正确的名称创建转换器权重字典
    weights = {"transformer": 1}
    # 期望的警告信息
    expected_msg = (
        'Attempting to weight transformer "transformer", '
        "but it is not present in transformer_list."
    )
    # 创建特征联合对象，指定转换器列表和转换器权重
    union = FeatureUnion(transformer_list, transformer_weights=weights)
    # 使用 pytest 检查是否引发预期的 ValueError 异常并匹配特定消息
    with pytest.raises(ValueError, match=expected_msg):
        union.fit(X, y)


@pytest.mark.parametrize("passthrough", [None, "passthrough"])
def test_pipeline_get_tags_none(passthrough):
    # 检查当第一个转换器为 None 或 'passthrough' 时，标签是否正确设置
    # 非回归测试，用于修复问题：https://github.com/scikit-learn/scikit-learn/issues/18815
    pipe = make_pipeline(passthrough, SVC())
    # 断言 _get_tags()["pairwise"] 返回 False
    assert not pipe._get_tags()["pairwise"]


# FIXME: 一旦我们有仅 API 检查的完整 `check_estimator`，替换此测试。
@pytest.mark.parametrize("Predictor", [MinimalRegressor, MinimalClassifier])
def test_search_cv_using_minimal_compatible_estimator(Predictor):
    # 检查第三方库的估计器是否可以作为管道的一部分，并通过网格搜索进行调整，而无需继承 BaseEstimator。
    rng = np.random.RandomState(0)
    X, y = rng.randn(25, 2), np.array([0] * 5 + [1] * 20)

    model = Pipeline(
        [("transformer", MinimalTransformer()), ("predictor", Predictor())]
    )
    model.fit(X, y)

    y_pred = model.predict(X)
    if is_classifier(model):
        assert_array_equal(y_pred, 1)
        assert model.score(X, y) == pytest.approx(accuracy_score(y, y_pred))
    else:
        assert_allclose(y_pred, y.mean())
        assert model.score(X, y) == pytest.approx(r2_score(y, y_pred))


def test_pipeline_check_if_fitted():
    class Estimator(BaseEstimator):
        def fit(self, X, y):
            self.fitted_ = True
            return self

    pipeline = Pipeline([("clf", Estimator())])
    # 使用 pytest 检查是否引发 NotFittedError
    with pytest.raises(NotFittedError):
        check_is_fitted(pipeline)
    pipeline.fit(iris.data, iris.target)
    # 确保管道已经拟合
    check_is_fitted(pipeline)


def test_feature_union_check_if_fitted():
    """检查 __sklearn_is_fitted__ 是否正确定义。"""

    X = [[1, 2], [3, 4], [5, 6]]
    y = [0, 1, 2]

    union = FeatureUnion([("clf", MinimalTransformer())])
    # 使用 pytest 检查是否引发 NotFittedError
    with pytest.raises(NotFittedError):
        check_is_fitted(union)

    union.fit(X, y)
    # 检查 union 对象是否已经被拟合（fitted）
    check_is_fitted(union)

    # 创建一个 FeatureUnion 对象，其中包含一个名为 "pass" 的 passthrough 转换器
    # passthrough 转换器是无状态的
    union = FeatureUnion([("pass", "passthrough")])
    
    # 再次检查 union 对象是否已经被拟合
    check_is_fitted(union)

    # 创建一个 FeatureUnion 对象，包含一个自定义的转换器 "clf" 和一个名为 "pass" 的 passthrough 转换器
    union = FeatureUnion([("clf", MinimalTransformer()), ("pass", "passthrough")])
    
    # 使用 pytest 来验证，如果 union 对象未被拟合，会抛出 NotFittedError 异常
    with pytest.raises(NotFittedError):
        check_is_fitted(union)

    # 使用给定的训练数据 X 和标签 y 对 union 对象进行拟合
    union.fit(X, y)
    
    # 再次检查 union 对象是否已经被拟合
    check_is_fitted(union)
# 检查流水线是否正确传递特征名称。
# 非回归测试用例，针对问题 #21349。
def test_pipeline_get_feature_names_out_passes_names_through():
    X, y = iris.data, iris.target

    # 定义一个继承自 StandardScaler 的类 AddPrefixStandardScalar
    class AddPrefixStandardScalar(StandardScaler):
        # 重写 get_feature_names_out 方法
        def get_feature_names_out(self, input_features=None):
            # 调用父类方法获取特征名称
            names = super().get_feature_names_out(input_features=input_features)
            # 添加前缀 "my_prefix_" 并返回 numpy 数组
            return np.asarray([f"my_prefix_{name}" for name in names], dtype=object)

    # 创建流水线，包括 AddPrefixStandardScalar 和 StandardScaler
    pipe = make_pipeline(AddPrefixStandardScalar(), StandardScaler())
    # 在数据上拟合流水线
    pipe.fit(X, y)

    # 获取输入数据的特征名称
    input_names = iris.feature_names
    # 获取流水线处理后的特征名称输出
    feature_names_out = pipe.get_feature_names_out(input_names)

    # 断言流水线输出的特征名称是否符合预期
    assert_array_equal(feature_names_out, [f"my_prefix_{name}" for name in input_names])


# 测试流水线的 set_output 方法与特征名称的集成
def test_pipeline_set_output_integration():
    pytest.importorskip("pandas")

    # 加载鸢尾花数据集的特征和目标值，返回 DataFrame 形式
    X, y = load_iris(as_frame=True, return_X_y=True)

    # 创建流水线，包括 StandardScaler 和 LogisticRegression
    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    # 设置流水线的输出格式为 pandas
    pipe.set_output(transform="pandas")
    # 在数据上拟合流水线
    pipe.fit(X, y)

    # 获取流水线前部分处理后的特征名称
    feature_names_in_ = pipe[:-1].get_feature_names_out()
    # 获取 LogisticRegression 模型输入的特征名称
    log_reg_feature_names = pipe[-1].feature_names_in_

    # 断言流水线前部分处理后的特征名称与 LogisticRegression 模型输入的特征名称是否一致
    assert_array_equal(feature_names_in_, log_reg_feature_names)


# 测试 FeatureUnion 与 set_output API 的集成
def test_feature_union_set_output():
    pd = pytest.importorskip("pandas")

    # 加载鸢尾花数据集的特征和目标值，返回 DataFrame 形式
    X, _ = load_iris(as_frame=True, return_X_y=True)
    X_train, X_test = train_test_split(X, random_state=0)

    # 创建包括 StandardScaler 和 PCA 的特征合并对象
    union = FeatureUnion([("scalar", StandardScaler()), ("pca", PCA())])
    # 设置特征合并对象的输出格式为 pandas
    union.set_output(transform="pandas")
    # 在训练集上拟合特征合并对象
    union.fit(X_train)

    # 对测试集进行数据转换
    X_trans = union.transform(X_test)

    # 断言转换后的数据类型为 pandas DataFrame
    assert isinstance(X_trans, pd.DataFrame)
    # 断言转换后的特征列名与特征合并对象的输出特征名称一致
    assert_array_equal(X_trans.columns, union.get_feature_names_out())
    # 断言转换后的数据索引与测试集索引一致
    assert_array_equal(X_trans.index, X_test.index)


# 检查 FeatureUnion.__getitem__ 返回预期结果
def test_feature_union_getitem():
    scalar = StandardScaler()
    pca = PCA()

    # 创建包括标准缩放器、PCA、传递操作和丢弃操作的特征合并对象
    union = FeatureUnion([
        ("scalar", scalar),
        ("pca", pca),
        ("pass", "passthrough"),
        ("drop_me", "drop"),
    ])

    # 断言获取特征合并对象中指定部分的正确性
    assert union["scalar"] is scalar
    assert union["pca"] is pca
    assert union["pass"] == "passthrough"
    assert union["drop_me"] == "drop"


# 参数化测试：当 __getitem__ 获取非字符串输入时，引发错误
@pytest.mark.parametrize("key", [0, slice(0, 2)])
def test_feature_union_getitem_error(key):
    # 创建包括标准缩放器和 PCA 的特征合并对象
    union = FeatureUnion([("scalar", StandardScaler()), ("pca", PCA())])

    # 断言获取非字符串键时，引发 KeyError 异常
    msg = "Only string keys are supported"
    with pytest.raises(KeyError, match=msg):
        union[key]


# 确保特征合并对象具有 `.feature_names_in_` 属性（如果 `X` 具有 `columns` 属性）
# 针对问题 #24754 的测试
def test_feature_union_feature_names_in_():
    pytest.importorskip("pandas")

    # 加载鸢尾花数据集的特征和目标值，返回 DataFrame 形式
    X, _ = load_iris(as_frame=True, return_X_y=True)

    # 特征合并对象应当在具有 `columns` 属性的情况下具有 `feature_names_in_` 属性
    # 创建一个标准化的转换器对象
    scaler = StandardScaler()
    # 使用给定数据 X 来拟合标准化转换器
    scaler.fit(X)
    # 创建一个特征联合对象，包含一个标准化的转换器
    union = FeatureUnion([("scale", scaler)])
    # 断言特征联合对象具有名为 feature_names_in_ 的属性
    assert hasattr(union, "feature_names_in_")
    # 断言数据 X 的列名与特征联合对象的 feature_names_in_ 属性相等
    assert_array_equal(X.columns, union.feature_names_in_)
    # 断言标准化转换器的 feature_names_in_ 属性与特征联合对象的相等
    assert_array_equal(scaler.feature_names_in_, union.feature_names_in_)

    # 使用 pandas.DataFrame 进行拟合
    union = FeatureUnion([("pass", "passthrough")])
    union.fit(X)
    # 断言特征联合对象具有名为 feature_names_in_ 的属性
    assert hasattr(union, "feature_names_in_")
    # 断言数据 X 的列名与特征联合对象的 feature_names_in_ 属性相等
    assert_array_equal(X.columns, union.feature_names_in_)

    # 使用 numpy 数组进行拟合
    X_array = X.to_numpy()
    union = FeatureUnion([("pass", "passthrough")])
    union.fit(X_array)
    # 断言特征联合对象不具有名为 feature_names_in_ 的属性
    assert not hasattr(union, "feature_names_in_")
# TODO(1.7): remove this test
# 定义一个测试函数，用于测试 Pipeline 对象的逆变换方法的弃用情况
def test_pipeline_inverse_transform_Xt_deprecation():
    # 创建一个 10x5 的随机正态分布数组作为输入数据 X
    X = np.random.RandomState(0).normal(size=(10, 5))
    # 创建一个 Pipeline 对象，包含一个 PCA 步骤，将数据 X 进行降维到 2 维
    pipe = Pipeline([("pca", PCA(n_components=2))])
    X = pipe.fit_transform(X)  # 对数据 X 进行拟合并转换

    # 测试在没有指定参数的情况下调用逆变换方法是否会抛出 TypeError 异常
    with pytest.raises(TypeError, match="Missing required positional argument"):
        pipe.inverse_transform()

    # 测试同时指定 X 和 Xt 参数时是否会抛出 TypeError 异常
    with pytest.raises(TypeError, match="Cannot use both X and Xt. Use X only"):
        pipe.inverse_transform(X=X, Xt=X)

    # 测试在捕获所有警告的情况下调用逆变换方法是否会产生警告
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("error")
        pipe.inverse_transform(X)

    # 测试在捕获 FutureWarning 警告的情况下调用逆变换方法是否会产生警告
    with pytest.warns(FutureWarning, match="Xt was renamed X in version 1.5"):
        pipe.inverse_transform(Xt=X)


# Test that metadata is routed correctly for pipelines and FeatureUnion
# =====================================================================


class SimpleEstimator(BaseEstimator):
    # This class is used in this section for testing routing in the pipeline.
    # This class should have every set_{method}_request
    # 以下是为测试管道中路由而定义的简单估算器类，包含多个用于测试的方法
    def fit(self, X, y, sample_weight=None, prop=None):
        assert sample_weight is not None
        assert prop is not None
        return self

    def fit_transform(self, X, y, sample_weight=None, prop=None):
        assert sample_weight is not None
        assert prop is not None
        return X + 1

    def fit_predict(self, X, y, sample_weight=None, prop=None):
        assert sample_weight is not None
        assert prop is not None
        return np.ones(len(X))

    def predict(self, X, sample_weight=None, prop=None):
        assert sample_weight is not None
        assert prop is not None
        return np.ones(len(X))

    def predict_proba(self, X, sample_weight=None, prop=None):
        assert sample_weight is not None
        assert prop is not None
        return np.ones(len(X))

    def predict_log_proba(self, X, sample_weight=None, prop=None):
        assert sample_weight is not None
        assert prop is not None
        return np.zeros(len(X))

    def decision_function(self, X, sample_weight=None, prop=None):
        assert sample_weight is not None
        assert prop is not None
        return np.ones(len(X))

    def score(self, X, y, sample_weight=None, prop=None):
        assert sample_weight is not None
        assert prop is not None
        return 1

    def transform(self, X, sample_weight=None, prop=None):
        assert sample_weight is not None
        assert prop is not None
        return X + 1

    def inverse_transform(self, X, sample_weight=None, prop=None):
        assert sample_weight is not None
        assert prop is not None
        return X - 1


@pytest.mark.usefixtures("enable_slep006")
# split and partial_fit not relevant for pipelines
# 参数化测试函数，用于测试管道中的元数据路由
@pytest.mark.parametrize("method", sorted(set(METHODS) - {"split", "partial_fit"}))
def test_metadata_routing_for_pipeline(method):
    """Test that metadata is routed correctly for pipelines."""
    def set_request(est, method, **kwarg):
        """为给定方法设置请求。

        如果给定方法是一个复合方法，为组成它的所有方法设置相同的请求。
        """
        if method in COMPOSITE_METHODS:  # 检查方法是否为复合方法
            methods = COMPOSITE_METHODS[method]  # 获取复合方法对应的所有方法列表
        else:
            methods = [method]  # 如果不是复合方法，则将方法放入列表中

        for method in methods:
            getattr(est, f"set_{method}_request")(**kwarg)  # 动态调用 est 对象的对应方法设置请求
        return est  # 返回设置后的 est 对象

    X, y = np.array([[1]]), np.array([1])
    sample_weight, prop, metadata = [1], "a", "b"

    # 测试元数据在请求管道时的正确路由
    est = SimpleEstimator()  # 创建 SimpleEstimator 实例
    est = set_request(est, method, sample_weight=True, prop=True)  # 设置请求到 est 实例中
    est = set_request(est, "fit", sample_weight=True, prop=True)  # 设置 "fit" 方法的请求到 est 实例中
    trs = (
        ConsumingTransformer()
        .set_fit_request(sample_weight=True, metadata=True)  # 设置消耗型转换器的 "fit" 请求
        .set_transform_request(sample_weight=True, metadata=True)  # 设置消耗型转换器的 "transform" 请求
        .set_inverse_transform_request(sample_weight=True, metadata=True)  # 设置消耗型转换器的 "inverse_transform" 请求
    )
    pipeline = Pipeline([("trs", trs), ("estimator", est)])  # 创建管道，包含消耗型转换器和估计器

    if "fit" not in method:  # 如果方法不包含 "fit"
        pipeline = pipeline.fit(X, y, sample_weight=sample_weight, prop=prop)  # 对管道进行拟合

    try:
        getattr(pipeline, method)(
            X, y, sample_weight=sample_weight, prop=prop, metadata=metadata
        )  # 调用管道的方法，并传递参数
    except TypeError:
        # 有些方法不接受 y 参数
        getattr(pipeline, method)(
            X, sample_weight=sample_weight, prop=prop, metadata=metadata
        )  # 调用管道的方法，并不传递 y 参数

    # 确保转换器已接收到元数据
    # 对于转换器，始终只调用 `fit` 和 `transform` 方法。
    check_recorded_metadata(
        obj=trs,
        method="fit",
        parent="fit",
        sample_weight=sample_weight,
        metadata=metadata,
    )  # 检查转换器是否记录了正确的元数据（"fit" 方法）
    check_recorded_metadata(
        obj=trs,
        method="transform",
        parent="transform",
        sample_weight=sample_weight,
        metadata=metadata,
    )  # 检查转换器是否记录了正确的元数据（"transform" 方法）
@pytest.mark.usefixtures("enable_slep006")
# 在管道中，split 和 partial_fit 方法不相关
# 使用 sorted 是为了让 `pytest -nX` 能够正常工作。如果不排序，不同的工作进程会以不同的顺序收集测试，导致测试失败。
# 从 METHODS 中排除 "split" 和 "partial_fit" 方法，其余方法作为参数传递给测试函数
@pytest.mark.parametrize("method", sorted(set(METHODS) - {"split", "partial_fit"}))
def test_metadata_routing_error_for_pipeline(method):
    """Test that metadata is not routed for pipelines when not requested."""
    X, y = [[1]], [1]
    sample_weight, prop = [1], "a"
    est = SimpleEstimator()
    # 在这里没有设置 sample_weight 请求，保持其为 None
    pipeline = Pipeline([("estimator", est)])
    error_message = (
        "[sample_weight, prop] are passed but are not explicitly set as requested"
        f" or not requested for SimpleEstimator.{method}"
    )
    with pytest.raises(ValueError, match=re.escape(error_message)):
        try:
            # 将 X, y 作为第一第二位置参数传递
            getattr(pipeline, method)(X, y, sample_weight=sample_weight, prop=prop)
        except TypeError:
            # 并非所有方法都接受 y（比如 `predict`），因此这里只传递 X 作为位置参数
            getattr(pipeline, method)(X, sample_weight=sample_weight, prop=prop)


@pytest.mark.parametrize(
    "method", ["decision_function", "transform", "inverse_transform"]
)
def test_routing_passed_metadata_not_supported(method):
    """Test that the right error message is raised when metadata is passed while
    not supported when `enable_metadata_routing=False`."""

    pipe = Pipeline([("estimator", SimpleEstimator())])

    with pytest.raises(
        ValueError, match="is only supported if enable_metadata_routing=True"
    ):
        getattr(pipe, method)([[1]], sample_weight=[1], prop="a")


@pytest.mark.usefixtures("enable_slep006")
def test_pipeline_with_estimator_with_len():
    """Test that pipeline works with estimators that have a `__len__` method."""
    pipe = Pipeline(
        [("trs", RandomTreesEmbedding()), ("estimator", RandomForestClassifier())]
    )
    pipe.fit([[1]], [1])
    pipe.predict([[1]])


@pytest.mark.usefixtures("enable_slep006")
@pytest.mark.parametrize("last_step", [None, "passthrough"])
def test_pipeline_with_no_last_step(last_step):
    """Test that the pipeline works when there is not last step.

    It should just ignore and pass through the data on transform.
    """
    pipe = Pipeline([("trs", FunctionTransformer()), ("estimator", last_step)])
    assert pipe.fit([[1]], [1]).transform([[1], [2], [3]]) == [[1], [2], [3]]


@pytest.mark.usefixtures("enable_slep006")
def test_feature_union_metadata_routing_error():
    """Test that the right error is raised when metadata is not requested."""
    X = np.array([[0, 1], [2, 2], [4, 6]])
    y = [1, 2, 3]
    sample_weight, metadata = [1, 1, 1], "a"

    # 测试缺少 set_fit_request 的情况
    feature_union = FeatureUnion([("sub_transformer", ConsumingTransformer())])
    # 创建错误消息，说明 sample_weight 和 metadata 被传递但未按要求设置
    error_message = (
        "[sample_weight, metadata] are passed but are not explicitly set as requested"
        f" or not requested for {ConsumingTransformer.__name__}.fit"
    )

    # 使用 pytest 检查是否引发了 UnsetMetadataPassedError 异常，匹配特定的错误消息
    with pytest.raises(UnsetMetadataPassedError, match=re.escape(error_message)):
        # 在 feature_union 上调用 fit 方法，传递 X, y, sample_weight 和 metadata 参数
        feature_union.fit(X, y, sample_weight=sample_weight, metadata=metadata)

    # 创建 FeatureUnion 对象 feature_union，包含一个子转换器 "sub_transformer"
    feature_union = FeatureUnion(
        [
            (
                "sub_transformer",
                # 创建 ConsumingTransformer 实例，并调用 set_fit_request 方法设置 sample_weight 和 metadata 请求为 True
                ConsumingTransformer().set_fit_request(
                    sample_weight=True, metadata=True
                ),
            )
        ]
    )

    # 创建错误消息，说明 sample_weight 和 metadata 被传递但未按要求设置
    error_message = (
        "[sample_weight, metadata] are passed but are not explicitly set as requested "
        f"or not requested for {ConsumingTransformer.__name__}.transform"
    )

    # 使用 pytest 检查是否引发了 UnsetMetadataPassedError 异常，匹配特定的错误消息
    with pytest.raises(UnsetMetadataPassedError, match=re.escape(error_message)):
        # 在 feature_union 上调用 fit 和 transform 方法，传递 X, sample_weight 和 metadata 参数
        feature_union.fit(
            X, y, sample_weight=sample_weight, metadata=metadata
        ).transform(X, sample_weight=sample_weight, metadata=metadata)
@pytest.mark.usefixtures("enable_slep006")
# 使用装饰器标记测试，启用 SLEP006 功能
def test_feature_union_get_metadata_routing_without_fit():
    """Test that get_metadata_routing() works regardless of the Child's
    consumption of any metadata."""
    # 创建 FeatureUnion 实例，包含一个子转换器 ConsumingTransformer()
    feature_union = FeatureUnion([("sub_transformer", ConsumingTransformer())])
    # 调用 FeatureUnion 的 get_metadata_routing() 方法进行测试
    feature_union.get_metadata_routing()


@pytest.mark.usefixtures("enable_slep006")
# 使用装饰器标记测试，启用 SLEP006 功能
@pytest.mark.parametrize(
    "transformer", [ConsumingTransformer, ConsumingNoFitTransformTransformer]
)
# 参数化测试函数，测试不同的转换器
def test_feature_union_metadata_routing(transformer):
    """Test that metadata is routed correctly for FeatureUnion."""
    # 创建示例数据 X, y
    X = np.array([[0, 1], [2, 2], [4, 6]])
    y = [1, 2, 3]
    sample_weight, metadata = [1, 1, 1], "a"

    # 创建 FeatureUnion 实例，包含多个子转换器
    feature_union = FeatureUnion(
        [
            (
                "sub_trans1",
                transformer(registry=_Registry())
                .set_fit_request(sample_weight=True, metadata=True)
                .set_transform_request(sample_weight=True, metadata=True),
            ),
            (
                "sub_trans2",
                transformer(registry=_Registry())
                .set_fit_request(sample_weight=True, metadata=True)
                .set_transform_request(sample_weight=True, metadata=True),
            ),
        ]
    )

    # 调用 feature_union 的 fit 方法，传入参数 sample_weight 和 metadata
    kwargs = {"sample_weight": sample_weight, "metadata": metadata}
    feature_union.fit(X, y, **kwargs)
    # 调用 feature_union 的 fit_transform 方法，传入参数 sample_weight 和 metadata
    feature_union.fit_transform(X, y, **kwargs)
    # 调用 feature_union 的 fit 方法后立即调用 transform 方法，传入参数 sample_weight 和 metadata
    feature_union.fit(X, y, **kwargs).transform(X, **kwargs)

    # 遍历 feature_union 的 transformer_list
    for transformer in feature_union.transformer_list:
        # 访问 transformer 的第二个元素，即子转换器
        # 通过 transformer[1] 访问子转换器，获取其 registry 属性
        registry = transformer[1].registry
        # 断言 registry 的长度不为零
        assert len(registry)
        # 遍历 registry 中的每个子转换器 sub_trans
        for sub_trans in registry:
            # 调用 check_recorded_metadata 函数，验证记录的元数据
            check_recorded_metadata(
                obj=sub_trans,
                method="fit",
                parent="fit",
                **kwargs,
            )


# End of routing tests
# ====================
```