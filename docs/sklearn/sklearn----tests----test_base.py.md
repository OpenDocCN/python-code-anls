# `D:\src\scipysrc\scikit-learn\sklearn\tests\test_base.py`

```
# 导入必要的库和模块
import pickle  # 导入pickle模块，用于对象序列化和反序列化
import re  # 导入re模块，用于正则表达式操作
import warnings  # 导入warnings模块，用于警告管理

import numpy as np  # 导入NumPy库，并使用np作为别名
import pytest  # 导入pytest测试框架
import scipy.sparse as sp  # 导入SciPy稀疏矩阵模块，并使用sp作为别名
from numpy.testing import assert_allclose  # 从NumPy测试模块中导入assert_allclose函数

import sklearn  # 导入scikit-learn机器学习库
from sklearn import config_context, datasets  # 导入scikit-learn的config_context和datasets模块
from sklearn.base import (  # 从scikit-learn的base模块中导入以下类和函数
    BaseEstimator,  # 基础估计器类
    OutlierMixin,  # 异常值混合类
    TransformerMixin,  # 转换器混合类
    clone,  # 克隆函数
    is_classifier,  # 判断是否是分类器的函数
    is_clusterer,  # 判断是否是聚类器的函数
    is_regressor,  # 判断是否是回归器的函数
)
from sklearn.cluster import KMeans  # 导入KMeans聚类算法
from sklearn.decomposition import PCA  # 导入PCA主成分分析算法
from sklearn.exceptions import InconsistentVersionWarning  # 导入版本不一致警告类
from sklearn.model_selection import GridSearchCV  # 导入网格搜索交叉验证类
from sklearn.pipeline import Pipeline  # 导入管道类
from sklearn.preprocessing import StandardScaler  # 导入标准化处理类
from sklearn.svm import SVC, SVR  # 导入支持向量分类和回归算法类
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # 导入决策树分类和回归算法类
from sklearn.utils._mocking import MockDataFrame  # 导入模拟数据帧类
from sklearn.utils._set_output import _get_output_config  # 导入获取输出配置函数
from sklearn.utils._testing import (  # 导入测试工具函数
    _convert_container,  # 转换容器的函数
    assert_array_equal,  # 断言两个数组是否相等的函数
    assert_no_warnings,  # 断言没有警告的函数
    ignore_warnings,  # 忽略警告的函数
)


#############################################################################
# 几个测试类
class MyEstimator(BaseEstimator):
    """自定义估计器类，继承自BaseEstimator"""

    def __init__(self, l1=0, empty=None):
        self.l1 = l1  # 初始化属性l1
        self.empty = empty  # 初始化属性empty


class K(BaseEstimator):
    """K类，继承自BaseEstimator"""

    def __init__(self, c=None, d=None):
        self.c = c  # 初始化属性c
        self.d = d  # 初始化属性d


class T(BaseEstimator):
    """T类，继承自BaseEstimator"""

    def __init__(self, a=None, b=None):
        self.a = a  # 初始化属性a
        self.b = b  # 初始化属性b


class NaNTag(BaseEstimator):
    """NaNTag类，继承自BaseEstimator"""

    def _more_tags(self):
        return {"allow_nan": True}  # 返回包含"allow_nan"键的字典，允许NaN


class NoNaNTag(BaseEstimator):
    """NoNaNTag类，继承自BaseEstimator"""

    def _more_tags(self):
        return {"allow_nan": False}  # 返回包含"allow_nan"键的字典，不允许NaN


class OverrideTag(NaNTag):
    """OverrideTag类，继承自NaNTag"""

    def _more_tags(self):
        return {"allow_nan": False}  # 返回包含"allow_nan"键的字典，不允许NaN


class DiamondOverwriteTag(NaNTag, NoNaNTag):
    """DiamondOverwriteTag类，多重继承自NaNTag和NoNaNTag"""

    def _more_tags(self):
        return dict()  # 返回一个空字典，不包含任何标签


class InheritDiamondOverwriteTag(DiamondOverwriteTag):
    """InheritDiamondOverwriteTag类，继承自DiamondOverwriteTag"""

    pass  # 未定义新的方法或属性，继承父类的行为


class ModifyInitParams(BaseEstimator):
    """ModifyInitParams类，继承自BaseEstimator"""

    """已弃用的行为。
    具有类型转换的相同参数。
    不满足 a is a
    """

    def __init__(self, a=np.array([0])):
        self.a = a.copy()  # 初始化属性a，使用a的副本


class Buggy(BaseEstimator):
    """Buggy类，继承自BaseEstimator"""

    "一个有缺陷的估计器，未正确设置其参数。"

    def __init__(self, a=None):
        self.a = 1  # 初始化属性a为1，存在参数设置错误


class NoEstimator:
    """NoEstimator类，不继承自BaseEstimator"""

    def __init__(self):
        pass

    def fit(self, X=None, y=None):
        return self  # 返回自身实例

    def predict(self, X=None):
        return None  # 返回None


class VargEstimator(BaseEstimator):
    """VargEstimator类，继承自BaseEstimator"""

    """scikit-learn估计器不应该使用不定参数。"""

    def __init__(self, *vargs):
        pass  # 未定义任何初始化行为


#############################################################################
# 测试部分


def test_clone():
    """测试clone函数的功能。

    检查clone函数是否能创建正确的深拷贝。
    我们创建一个估计器，对其原始状态进行拷贝
    （在本例中，即当前估计器的当前状态），
    并检查所得到的拷贝是否是正确的深拷贝。
    """
    # 从 sklearn 库中导入特征选择模块 SelectFpr 和特征评估函数 f_classif
    from sklearn.feature_selection import SelectFpr, f_classif
    
    # 创建 SelectFpr 特征选择器对象，使用 f_classif 作为评估函数，设置 alpha 值为 0.1
    selector = SelectFpr(f_classif, alpha=0.1)
    
    # 使用 clone 函数复制 selector 对象，生成一个新的特征选择器对象 new_selector
    new_selector = clone(selector)
    
    # 断言语句，验证 selector 和 new_selector 是不同的对象
    assert selector is not new_selector
    
    # 断言语句，验证 selector 和 new_selector 的参数相同
    assert selector.get_params() == new_selector.get_params()
    
    # 创建一个新的 SelectFpr 特征选择器对象 selector，使用 f_classif 作为评估函数，
    # alpha 值为一个形状为 (10, 2) 的全零数组 np.zeros((10, 2))
    selector = SelectFpr(f_classif, alpha=np.zeros((10, 2)))
    
    # 使用 clone 函数复制 selector 对象，生成一个新的特征选择器对象 new_selector
    new_selector = clone(selector)
    
    # 断言语句，验证 selector 和 new_selector 是不同的对象
    assert selector is not new_selector
def test_clone_2():
    # Tests that clone doesn't copy everything.
    # We first create an estimator, give it an own attribute, and
    # make a copy of its original state. Then we check that the copy doesn't
    # have the specific attribute we manually added to the initial estimator.
    
    from sklearn.feature_selection import SelectFpr, f_classif

    selector = SelectFpr(f_classif, alpha=0.1)  # 创建一个 SelectFpr 特征选择器对象
    selector.own_attribute = "test"  # 给 selector 对象添加自定义属性
    new_selector = clone(selector)  # 克隆 selector 对象
    assert not hasattr(new_selector, "own_attribute")  # 断言克隆后的对象没有 own_attribute 属性


def test_clone_buggy():
    # Check that clone raises an error on buggy estimators.
    
    buggy = Buggy()  # 创建 Buggy 类的实例对象
    buggy.a = 2  # 给 buggy 对象添加属性 a
    with pytest.raises(RuntimeError):
        clone(buggy)  # 应当抛出 RuntimeError 异常

    no_estimator = NoEstimator()  # 创建 NoEstimator 类的实例对象
    with pytest.raises(TypeError):
        clone(no_estimator)  # 应当抛出 TypeError 异常

    varg_est = VargEstimator()  # 创建 VargEstimator 类的实例对象
    with pytest.raises(RuntimeError):
        clone(varg_est)  # 应当抛出 RuntimeError 异常

    est = ModifyInitParams()  # 创建 ModifyInitParams 类的实例对象
    with pytest.raises(RuntimeError):
        clone(est)  # 应当抛出 RuntimeError 异常


def test_clone_empty_array():
    # Regression test for cloning estimators with empty arrays
    
    clf = MyEstimator(empty=np.array([]))  # 创建 MyEstimator 类的实例对象 clf，使用空数组作为参数
    clf2 = clone(clf)  # 克隆 clf 对象
    assert_array_equal(clf.empty, clf2.empty)  # 断言 clf 和 clf2 的 empty 属性相等

    clf = MyEstimator(empty=sp.csr_matrix(np.array([[0]])))  # 创建 MyEstimator 类的实例对象 clf，使用稀疏矩阵作为参数
    clf2 = clone(clf)  # 克隆 clf 对象
    assert_array_equal(clf.empty.data, clf2.empty.data)  # 断言 clf 和 clf2 的 empty 属性数据相等


def test_clone_nan():
    # Regression test for cloning estimators with default parameter as np.nan
    
    clf = MyEstimator(empty=np.nan)  # 创建 MyEstimator 类的实例对象 clf，使用 np.nan 作为参数
    clf2 = clone(clf)  # 克隆 clf 对象

    assert clf.empty is clf2.empty  # 断言 clf 和 clf2 的 empty 属性是同一个对象


def test_clone_dict():
    # test that clone creates a clone of a dict
    
    orig = {"a": MyEstimator()}  # 创建包含 MyEstimator 类实例的字典 orig
    cloned = clone(orig)  # 克隆 orig 字典
    assert orig["a"] is not cloned["a"]  # 断言 orig 和 cloned 的 "a" 键对应的值不是同一个对象


def test_clone_sparse_matrices():
    sparse_matrix_classes = [
        cls
        for name in dir(sp)
        if name.endswith("_matrix") and type(cls := getattr(sp, name)) is type
    ]

    for cls in sparse_matrix_classes:
        sparse_matrix = cls(np.eye(5))  # 根据当前的稀疏矩阵类创建一个 5x5 的单位矩阵对象 sparse_matrix
        clf = MyEstimator(empty=sparse_matrix)  # 创建 MyEstimator 类的实例对象 clf，使用稀疏矩阵作为参数
        clf_cloned = clone(clf)  # 克隆 clf 对象
        assert clf.empty.__class__ is clf_cloned.empty.__class__  # 断言 clf 和 clf_cloned 的 empty 属性类相同
        assert_array_equal(clf.empty.toarray(), clf_cloned.empty.toarray())  # 断言 clf 和 clf_cloned 的 empty 属性数组相等


def test_clone_estimator_types():
    # Check that clone works for parameters that are types rather than
    # instances
    
    clf = MyEstimator(empty=MyEstimator)  # 创建 MyEstimator 类的实例对象 clf，使用 MyEstimator 类作为参数
    clf2 = clone(clf)  # 克隆 clf 对象

    assert clf.empty is clf2.empty  # 断言 clf 和 clf2 的 empty 属性是同一个对象


def test_clone_class_rather_than_instance():
    # Check that clone raises expected error message when
    # cloning class rather than instance
    
    msg = "You should provide an instance of scikit-learn estimator"
    with pytest.raises(TypeError, match=msg):
        clone(MyEstimator)  # 应当抛出 TypeError 异常，匹配给定的错误消息


def test_repr():
    # Smoke test the repr of the base estimator.
    
    my_estimator = MyEstimator()  # 创建 MyEstimator 类的实例对象 my_estimator
    repr(my_estimator)  # 调用 repr 函数以测试基础评估器的表示形式
    test = T(K(), K())
    assert repr(test) == "T(a=K(), b=K())"  # 断言 test 对象的表示形式符合预期

    some_est = T(a=["long_params"] * 1000)
    assert len(repr(some_est)) == 485  # 断言 repr(some_est) 的长度为 485


def test_str():
    pass  # 该函数暂未实现，留空
    # 创建一个名为 my_estimator 的 MyEstimator 实例
    my_estimator = MyEstimator()
    # 调用 my_estimator 对象的 str 方法，将其转换为字符串表示形式
    str(my_estimator)
def test_get_params():
    # 创建一个测试对象test，使用K()作为第一个参数，K作为第二个参数
    test = T(K(), K)
    
    # 断言"a__d"在深度参数为True时出现在test的参数中
    assert "a__d" in test.get_params(deep=True)
    
    # 断言"a__d"在深度参数为False时不出现在test的参数中
    assert "a__d" not in test.get_params(deep=False)
    
    # 设置参数a__d为2
    test.set_params(a__d=2)
    
    # 断言test.a.d的值为2
    assert test.a.d == 2
    
    # 使用pytest检查设置不存在的参数a__a=2时是否会引发ValueError异常
    with pytest.raises(ValueError):
        test.set_params(a__a=2)


@pytest.mark.parametrize(
    "estimator, expected_result",
    [
        # 测试各种estimator是否正确判断为分类器
        (SVC(), True),
        (GridSearchCV(SVC(), {"C": [0.1, 1]}), True),
        (Pipeline([("svc", SVC())]), True),
        (Pipeline([("svc_cv", GridSearchCV(SVC(), {"C": [0.1, 1]}))]), True),
        (SVR(), False),
        (GridSearchCV(SVR(), {"C": [0.1, 1]}), False),
        (Pipeline([("svr", SVR())]), False),
        (Pipeline([("svr_cv", GridSearchCV(SVR(), {"C": [0.1, 1]}))]), False),
    ],
)
def test_is_classifier(estimator, expected_result):
    # 断言is_classifier函数对于给定的estimator返回expected_result
    assert is_classifier(estimator) == expected_result


@pytest.mark.parametrize(
    "estimator, expected_result",
    [
        # 测试各种estimator是否正确判断为回归器
        (SVR(), True),
        (GridSearchCV(SVR(), {"C": [0.1, 1]}), True),
        (Pipeline([("svr", SVR())]), True),
        (Pipeline([("svr_cv", GridSearchCV(SVR(), {"C": [0.1, 1]}))]), True),
        (SVC(), False),
        (GridSearchCV(SVC(), {"C": [0.1, 1]}), False),
        (Pipeline([("svc", SVC())]), False),
        (Pipeline([("svc_cv", GridSearchCV(SVC(), {"C": [0.1, 1]}))]), False),
    ],
)
def test_is_regressor(estimator, expected_result):
    # 断言is_regressor函数对于给定的estimator返回expected_result
    assert is_regressor(estimator) == expected_result


@pytest.mark.parametrize(
    "estimator, expected_result",
    [
        # 测试各种estimator是否正确判断为聚类器
        (KMeans(), True),
        (GridSearchCV(KMeans(), {"n_clusters": [3, 8]}), True),
        (Pipeline([("km", KMeans())]), True),
        (Pipeline([("km_cv", GridSearchCV(KMeans(), {"n_clusters": [3, 8]}))]), True),
        (SVC(), False),
        (GridSearchCV(SVC(), {"C": [0.1, 1]}), False),
        (Pipeline([("svc", SVC())]), False),
        (Pipeline([("svc_cv", GridSearchCV(SVC(), {"C": [0.1, 1]}))]), False),
    ],
)
def test_is_clusterer(estimator, expected_result):
    # 断言is_clusterer函数对于给定的estimator返回expected_result
    assert is_clusterer(estimator) == expected_result


def test_set_params():
    # 测试嵌套估计器参数设置的情况
    clf = Pipeline([("svc", SVC())])
    
    # 断言尝试设置svc中不存在的参数svc__stupid_param=True时，会引发ValueError异常
    with pytest.raises(ValueError):
        clf.set_params(svc__stupid_param=True)
    
    # 断言尝试设置Pipeline中不存在的参数svm__stupid_param=True时，会引发ValueError异常
    with pytest.raises(ValueError):
        clf.set_params(svm__stupid_param=True)

    # 当前代码段未捕获Pipeline中的项是否为估计器的异常
    # bad_pipeline = Pipeline([("bad", NoEstimator())])
    # assert_raises(AttributeError, bad_pipeline.set_params,
    #               bad__stupid_param=True)


def test_set_params_passes_all_parameters():
    # 确保所有参数一起传递给嵌套估计器的set_params函数。用于验证＃9944的回归测试
    pass  # 本测试目前没有具体实现，只是占位符
    class TestDecisionTree(DecisionTreeClassifier):
        # 定义一个测试用的决策树类，继承自DecisionTreeClassifier
    
        def set_params(self, **kwargs):
            # 设置对象参数的方法，接受关键字参数
    
            super().set_params(**kwargs)
            # 调用父类的set_params方法，传递所有接收到的关键字参数
    
            # 在测试范围内验证期望的关键字参数是否正确
            assert kwargs == expected_kwargs
    
            return self
            # 返回当前对象自身
    
    expected_kwargs = {"max_depth": 5, "min_samples_leaf": 2}
    # 预期的关键字参数字典，用于验证设置参数的正确性
    
    for est in [
        Pipeline([("estimator", TestDecisionTree())]),
        GridSearchCV(TestDecisionTree(), {}),
    ]:
        # 对于列表中的每个est对象
    
        est.set_params(estimator__max_depth=5, estimator__min_samples_leaf=2)
        # 调用est对象的set_params方法，设置estimator对象的max_depth和min_samples_leaf参数
def test_set_params_updates_valid_params():
    # 创建一个使用 DecisionTreeClassifier 作为基础估计器的 GridSearchCV 对象
    gscv = GridSearchCV(DecisionTreeClassifier(), {})
    # 设置 GridSearchCV 对象的估计器为 SVC，并尝试设置其 C 参数为 42.0
    gscv.set_params(estimator=SVC(), estimator__C=42.0)
    # 断言设置后的估计器 gscv.estimator 的 C 参数是否为 42.0
    assert gscv.estimator.C == 42.0


@pytest.mark.parametrize(
    "tree,dataset",
    [
        (
            DecisionTreeClassifier(max_depth=2, random_state=0),
            datasets.make_classification(random_state=0),
        ),
        (
            DecisionTreeRegressor(max_depth=2, random_state=0),
            datasets.make_regression(random_state=0),
        ),
    ],
)
def test_score_sample_weight(tree, dataset):
    rng = np.random.RandomState(0)
    # 检查使用样本权重和不使用样本权重时得分是否不同
    X, y = dataset

    # 使用给定数据集 (X, y) 对决策树 tree 进行拟合
    tree.fit(X, y)
    # 生成随机样本权重
    sample_weight = rng.randint(1, 10, size=len(y))
    # 计算未加权和加权后的得分
    score_unweighted = tree.score(X, y)
    score_weighted = tree.score(X, y, sample_weight=sample_weight)
    msg = "Unweighted and weighted scores are unexpectedly equal"
    # 断言未加权和加权得分不相等
    assert score_unweighted != score_weighted, msg


def test_clone_pandas_dataframe():
    class DummyEstimator(TransformerMixin, BaseEstimator):
        """This is a dummy class for generating numerical features

        This feature extractor extracts numerical features from pandas data
        frame.

        Parameters
        ----------

        df: pandas data frame
            The pandas data frame parameter.

        Notes
        -----
        """

        def __init__(self, df=None, scalar_param=1):
            self.df = df
            self.scalar_param = scalar_param

        def fit(self, X, y=None):
            pass

        def transform(self, X):
            pass

    # 创建一个 MockDataFrame 对象 df，并使用其初始化 DummyEstimator 对象 e
    d = np.arange(10)
    df = MockDataFrame(d)
    e = DummyEstimator(df, scalar_param=1)
    # 克隆 DummyEstimator 对象 e
    cloned_e = clone(e)

    # 断言克隆后的对象 cloned_e 的 df 属性与原对象 e 的 df 属性相等
    assert (e.df == cloned_e.df).values.all()
    # 断言克隆后的对象 cloned_e 的 scalar_param 属性与原对象 e 的 scalar_param 属性相等
    assert e.scalar_param == cloned_e.scalar_param


def test_clone_protocol():
    """Checks that clone works with `__sklearn_clone__` protocol."""

    class FrozenEstimator(BaseEstimator):
        def __init__(self, fitted_estimator):
            self.fitted_estimator = fitted_estimator

        def __getattr__(self, name):
            return getattr(self.fitted_estimator, name)

        def __sklearn_clone__(self):
            return self

        def fit(self, *args, **kwargs):
            return self

        def fit_transform(self, *args, **kwargs):
            return self.fitted_estimator.transform(*args, **kwargs)

    # 创建一个 PCA 对象并拟合数据 X
    X = np.array([[-1, -1], [-2, -1], [-3, -2]])
    pca = PCA().fit(X)
    components = pca.components_

    # 使用 PCA 对象初始化 FrozenEstimator 对象 frozen_pca
    frozen_pca = FrozenEstimator(pca)
    # 断言 frozen_pca 的 components_ 属性与原 PCA 对象 pca 的 components_ 属性相等
    assert_allclose(frozen_pca.components_, components)

    # 调用 PCA 对象的方法，如 get_feature_names_out，依然有效
    assert_array_equal(frozen_pca.get_feature_names_out(), pca.get_feature_names_out())
    # 在新数据上进行拟合不会改变 `components_` 的值
    X_new = np.asarray([[-1, 2], [3, 4], [1, 2]])
    # 对新数据进行拟合
    frozen_pca.fit(X_new)
    # 断言 `frozen_pca.components_` 与预期的 `components` 值相等
    assert_allclose(frozen_pca.components_, components)

    # `fit_transform` 方法不会改变状态
    frozen_pca.fit_transform(X_new)
    # 再次断言 `frozen_pca.components_` 与预期的 `components` 值相等
    assert_allclose(frozen_pca.components_, components)

    # 克隆估计器不会产生任何操作
    clone_frozen_pca = clone(frozen_pca)
    # 断言克隆后的对象与原始对象是同一个对象
    assert clone_frozen_pca is frozen_pca
    # 再次断言克隆后的 `clone_frozen_pca.components_` 与预期的 `components` 值相等
    assert_allclose(clone_frozen_pca.components_, components)
def test_pickle_version_warning_is_not_raised_with_matching_version():
    # 加载鸢尾花数据集
    iris = datasets.load_iris()
    # 使用决策树分类器拟合数据集
    tree = DecisionTreeClassifier().fit(iris.data, iris.target)
    # 对拟合好的决策树进行序列化
    tree_pickle = pickle.dumps(tree)
    # 断言序列化结果中包含 "_sklearn_version" 字段
    assert b"_sklearn_version" in tree_pickle
    # 通过 pickle 反序列化恢复决策树分类器
    tree_restored = assert_no_warnings(pickle.loads, tree_pickle)

    # 测试恢复后的决策树分类器是否可以进行预测
    score_of_original = tree.score(iris.data, iris.target)
    score_of_restored = tree_restored.score(iris.data, iris.target)
    # 断言原始和恢复后的决策树分类器预测得分相同
    assert score_of_original == score_of_restored


class TreeBadVersion(DecisionTreeClassifier):
    def __getstate__(self):
        # 返回包含自身属性以及 "_sklearn_version" 属性的字典
        return dict(self.__dict__.items(), _sklearn_version="something")


# 提示消息模板，用于在不同版本的情况下提供警告
pickle_error_message = (
    "Trying to unpickle estimator {estimator} from "
    "version {old_version} when using version "
    "{current_version}. This might "
    "lead to breaking code or invalid results. "
    "Use at your own risk."
)


def test_pickle_version_warning_is_issued_upon_different_version():
    # 加载鸢尾花数据集
    iris = datasets.load_iris()
    # 使用 TreeBadVersion 拟合数据集
    tree = TreeBadVersion().fit(iris.data, iris.target)
    # 对拟合好的 TreeBadVersion 进行序列化
    tree_pickle_other = pickle.dumps(tree)
    # 构造警告消息，指出不同版本情况下的潜在问题
    message = pickle_error_message.format(
        estimator="TreeBadVersion",
        old_version="something",
        current_version=sklearn.__version__,
    )
    # 使用 pytest 的 warn 函数检查是否发出 UserWarning 警告，并匹配警告消息
    with pytest.warns(UserWarning, match=message) as warning_record:
        pickle.loads(tree_pickle_other)

    # 获取警告消息
    message = warning_record.list[0].message
    # 断言警告消息的类型为 InconsistentVersionWarning
    assert isinstance(message, InconsistentVersionWarning)
    # 断言警告消息中的分类器名称为 "TreeBadVersion"
    assert message.estimator_name == "TreeBadVersion"
    # 断言警告消息中的原始 sklearn 版本为 "something"
    assert message.original_sklearn_version == "something"
    # 断言警告消息中的当前 sklearn 版本为当前版本
    assert message.current_sklearn_version == sklearn.__version__


class TreeNoVersion(DecisionTreeClassifier):
    def __getstate__(self):
        # 返回当前对象的 __dict__，不包含任何版本信息
        return self.__dict__


def test_pickle_version_warning_is_issued_when_no_version_info_in_pickle():
    # 加载鸢尾花数据集
    iris = datasets.load_iris()
    # 使用 TreeNoVersion 拟合数据集，TreeNoVersion 没有 __getstate__ 方法，类似于 0.18 版本之前的行为
    tree = TreeNoVersion().fit(iris.data, iris.target)

    # 对拟合好的 TreeNoVersion 进行序列化
    tree_pickle_noversion = pickle.dumps(tree)
    # 断言序列化结果中不包含 "_sklearn_version" 字段
    assert b"_sklearn_version" not in tree_pickle_noversion

    # 构造警告消息，指出使用没有版本信息的序列化对象可能会导致问题
    message = pickle_error_message.format(
        estimator="TreeNoVersion",
        old_version="pre-0.18",
        current_version=sklearn.__version__,
    )
    # 使用 pytest 的 warn 函数检查是否发出 UserWarning 警告，并匹配警告消息
    with pytest.warns(UserWarning, match=message):
        pickle.loads(tree_pickle_noversion)


def test_pickle_version_no_warning_is_issued_with_non_sklearn_estimator():
    # 加载鸢尾花数据集
    iris = datasets.load_iris()
    # 使用 TreeNoVersion 拟合数据集
    tree = TreeNoVersion().fit(iris.data, iris.target)
    # 对拟合好的 TreeNoVersion 进行序列化
    tree_pickle_noversion = pickle.dumps(tree)
    try:
        # 备份 TreeNoVersion 的 __module__ 属性
        module_backup = TreeNoVersion.__module__
        # 将 TreeNoVersion 的 __module__ 属性设置为 "notsklearn"
        TreeNoVersion.__module__ = "notsklearn"
        # 断言在反序列化时不会发出警告
        assert_no_warnings(pickle.loads, tree_pickle_noversion)
    finally:
        # 恢复 TreeNoVersion 的 __module__ 属性
        TreeNoVersion.__module__ = module_backup


class DontPickleAttributeMixin:
    # 这里是类的定义，没有具体的代码需要注释
    # 返回对象的状态字典，用于对象的序列化
    def __getstate__(self):
        # 复制对象的字典表示，并添加一个不需要被序列化的属性设置为 None
        data = self.__dict__.copy()
        data["_attribute_not_pickled"] = None
        return data

    # 设置对象的状态，用于对象的反序列化
    def __setstate__(self, state):
        # 在状态字典中添加一个标记，表示对象已经被恢复
        state["_restored"] = True
        # 更新对象的字典表示，以反序列化的状态信息
        self.__dict__.update(state)
class MultiInheritanceEstimator(DontPickleAttributeMixin, BaseEstimator):
    # 多重继承的估算器类，混合了不可pickle属性的Mixin和基础估算器类

    def __init__(self, attribute_pickled=5):
        # 初始化方法，设置可pickle的属性和不可pickle的属性
        self.attribute_pickled = attribute_pickled
        self._attribute_not_pickled = None


def test_pickling_when_getstate_is_overwritten_by_mixin():
    # 测试当由Mixin重写__getstate__时的pickle行为

    estimator = MultiInheritanceEstimator()
    # 创建MultiInheritanceEstimator实例

    estimator._attribute_not_pickled = "this attribute should not be pickled"
    # 设置不可pickle属性的值为字符串

    serialized = pickle.dumps(estimator)
    # 序列化估算器对象

    estimator_restored = pickle.loads(serialized)
    # 反序列化恢复估算器对象

    assert estimator_restored.attribute_pickled == 5
    # 断言恢复后的可pickle属性值为5
    assert estimator_restored._attribute_not_pickled is None
    # 断言恢复后的不可pickle属性值为None
    assert estimator_restored._restored


def test_pickling_when_getstate_is_overwritten_by_mixin_outside_of_sklearn():
    # 测试在sklearn之外，由Mixin重写__getstate__的pickle行为

    try:
        estimator = MultiInheritanceEstimator()
        # 创建MultiInheritanceEstimator实例

        text = "this attribute should not be pickled"
        estimator._attribute_not_pickled = text
        # 设置不可pickle属性的值为字符串

        old_mod = type(estimator).__module__
        type(estimator).__module__ = "notsklearn"
        # 临时修改估算器对象的模块名称为"notsklearn"

        serialized = estimator.__getstate__()
        # 调用__getstate__方法进行序列化

        assert serialized == {"_attribute_not_pickled": None, "attribute_pickled": 5}
        # 断言序列化后的数据字典符合预期

        serialized["attribute_pickled"] = 4
        # 修改序列化后的数据字典中可pickle属性的值

        estimator.__setstate__(serialized)
        # 调用__setstate__方法进行反序列化恢复

        assert estimator.attribute_pickled == 4
        # 断言恢复后的可pickle属性值为4
        assert estimator._restored
        # 断言恢复后的状态标记为已恢复
    finally:
        type(estimator).__module__ = old_mod
        # 恢复估算器对象的原始模块名称


class SingleInheritanceEstimator(BaseEstimator):
    # 单继承的估算器类，继承自基础估算器类

    def __init__(self, attribute_pickled=5):
        # 初始化方法，设置可pickle的属性和不可pickle的属性
        self.attribute_pickled = attribute_pickled
        self._attribute_not_pickled = None

    def __getstate__(self):
        # 自定义__getstate__方法，返回序列化数据字典
        data = self.__dict__.copy()
        data["_attribute_not_pickled"] = None
        return data


@ignore_warnings(category=(UserWarning))
def test_pickling_works_when_getstate_is_overwritten_in_the_child_class():
    # 测试当子类中重写__getstate__方法时的pickle行为

    estimator = SingleInheritanceEstimator()
    # 创建SingleInheritanceEstimator实例

    estimator._attribute_not_pickled = "this attribute should not be pickled"
    # 设置不可pickle属性的值为字符串

    serialized = pickle.dumps(estimator)
    # 序列化估算器对象

    estimator_restored = pickle.loads(serialized)
    # 反序列化恢复估算器对象

    assert estimator_restored.attribute_pickled == 5
    # 断言恢复后的可pickle属性值为5
    assert estimator_restored._attribute_not_pickled is None
    # 断言恢复后的不可pickle属性值为None


def test_tag_inheritance():
    # 测试继承中标签的变更不被允许

    nan_tag_est = NaNTag()
    # 创建NaNTag实例

    no_nan_tag_est = NoNaNTag()
    # 创建NoNaNTag实例

    assert nan_tag_est._get_tags()["allow_nan"]
    # 断言NaNTag实例的标签中包含"allow_nan"

    assert not no_nan_tag_est._get_tags()["allow_nan"]
    # 断言NoNaNTag实例的标签中不包含"allow_nan"

    redefine_tags_est = OverrideTag()
    # 创建OverrideTag实例

    assert not redefine_tags_est._get_tags()["allow_nan"]
    # 断言OverrideTag实例的标签中不包含"allow_nan"

    diamond_tag_est = DiamondOverwriteTag()
    # 创建DiamondOverwriteTag实例

    assert diamond_tag_est._get_tags()["allow_nan"]
    # 断言DiamondOverwriteTag实例的标签中包含"allow_nan"

    inherit_diamond_tag_est = InheritDiamondOverwriteTag()
    # 创建InheritDiamondOverwriteTag实例

    assert inherit_diamond_tag_est._get_tags()["allow_nan"]
    # 断言InheritDiamondOverwriteTag实例的标签中包含"allow_nan"


def test_raises_on_get_params_non_attribute():
    # 测试在获取非属性参数时引发异常的情况

    class MyEstimator(BaseEstimator):
        # 定义MyEstimator类，继承自基础估算器类

        def __init__(self, param=5):
            # 初始化方法，参数param默认值为5
            pass

        def fit(self, X, y=None):
            # fit方法，用于拟合模型，返回self
            return self

    est = MyEstimator()
    # 创建MyEstimator实例

    msg = "'MyEstimator' object has no attribute 'param'"
    # 预期的异常消息
    # 使用 pytest 模块的 raises 函数来检查是否抛出指定类型的异常，并且异常消息要与给定的正则表达式模式匹配
    with pytest.raises(AttributeError, match=msg):
        # 调用 est 对象的 get_params 方法
        est.get_params()
# 检查显示配置标志是否控制 JSON 输出
def test_repr_mimebundle_():
    # 创建决策树分类器对象
    tree = DecisionTreeClassifier()
    # 调用对象的 _repr_mimebundle_ 方法，获取其输出
    output = tree._repr_mimebundle_()
    # 断言输出中包含 "text/plain"
    assert "text/plain" in output
    # 断言输出中包含 "text/html"
    assert "text/html" in output

    # 修改显示配置为 "text"，检查输出
    with config_context(display="text"):
        # 调用对象的 _repr_mimebundle_ 方法，获取其输出
        output = tree._repr_mimebundle_()
        # 断言输出中包含 "text/plain"
        assert "text/plain" in output
        # 断言输出中不包含 "text/html"
        assert "text/html" not in output


# 检查显示配置标志是否控制 HTML 输出
def test_repr_html_wraps():
    # 创建决策树分类器对象
    tree = DecisionTreeClassifier()

    # 调用对象的 _repr_html_ 方法，获取其输出
    output = tree._repr_html_()
    # 断言输出中包含 "<style>"
    assert "<style>" in output

    # 修改显示配置为 "text"，验证 `_repr_html_` 方法在此情况下抛出异常
    with config_context(display="text"):
        msg = "_repr_html_ is only defined when"
        # 断言调用对象的 _repr_html_ 方法会引发 AttributeError 异常，并且异常信息符合预期
        with pytest.raises(AttributeError, match=msg):
            output = tree._repr_html_()


# 检查 `_check_n_features` 方法在 reset=False 时验证数据
def test_n_features_in_validation():
    # 创建自定义估计器对象
    est = MyEstimator()
    X_train = [[1, 2, 3], [4, 5, 6]]
    # 调用对象的 _check_n_features 方法，重置 n_features_in_
    est._check_n_features(X_train, reset=True)

    # 断言对象的 n_features_in_ 属性值为 3
    assert est.n_features_in_ == 3

    # 准备错误的消息字符串
    msg = "X does not contain any features, but MyEstimator is expecting 3 features"
    # 断言调用对象的 _check_n_features 方法会引发 ValueError 异常，并且异常信息符合预期
    with pytest.raises(ValueError, match=msg):
        est._check_n_features("invalid X", reset=False)


# 检查 `_check_n_features` 方法在未定义 n_features_in_ 时不验证数据
def test_n_features_in_no_validation():
    # 创建自定义估计器对象
    est = MyEstimator()
    # 调用对象的 _check_n_features 方法，但不重置 n_features_in_
    est._check_n_features("invalid X", reset=True)

    # 断言对象没有属性 n_features_in_
    assert not hasattr(est, "n_features_in_")

    # 调用对象的 _check_n_features 方法，不会引发异常
    est._check_n_features("invalid X", reset=False)


# 检查 `_validate_data` 方法是否记录 feature_names_in_ 的情况
def test_feature_names_in():
    # 导入 pandas 库，如不存在则跳过此测试
    pd = pytest.importorskip("pandas")
    # 加载鸢尾花数据集
    iris = datasets.load_iris()
    X_np = iris.data
    # 创建数据框 df，并使用鸢尾花数据集的特征名作为列名
    df = pd.DataFrame(X_np, columns=iris.feature_names)

    # 定义一个无操作的转换器类
    class NoOpTransformer(TransformerMixin, BaseEstimator):
        def fit(self, X, y=None):
            # 验证数据，并保存 feature_names_in_
            self._validate_data(X)
            return self

        def transform(self, X):
            # 在不重置的情况下验证数据
            self._validate_data(X, reset=False)
            return X

    # 在数据框上拟合转换器，保存 feature_names_in_
    trans = NoOpTransformer().fit(df)
    # 断言 trans 对象的 feature_names_in_ 属性值与 df.columns 相同
    assert_array_equal(trans.feature_names_in_, df.columns)

    # 再次拟合，但这次使用 ndarray，不会保留之前的 feature_names_in_ 属性
    trans.fit(X_np)
    # 断言 trans 对象没有属性 feature_names_in_
    assert not hasattr(trans, "feature_names_in_")

    # 再次在数据框上拟合转换器
    trans.fit(df)
    # 准备错误的消息字符串
    msg = "The feature names should match those that were passed"
    # 断言调用 trans 对象的 transform 方法会引发 ValueError 异常，并且异常信息符合预期
    df_bad = pd.DataFrame(X_np, columns=iris.feature_names[::-1])
    with pytest.raises(ValueError, match=msg):
        trans.transform(df_bad)

    # 在数据框上拟合转换器，但尝试在转换 ndarray 时会引发警告
    msg = (
        "X does not have valid feature names, but NoOpTransformer was "
        "fitted with feature names"
    )
    with pytest.warns(UserWarning, match=msg):
        trans.transform(X_np)

    # 在 ndarray 上拟合转换器，但尝试在转换数据框时会引发警告
    msg = "X has feature names, but NoOpTransformer was fitted without feature names"
    # 定义警告信息字符串，指示特征名已存在但 NoOpTransformer 没有适配特征名
    trans = NoOpTransformer().fit(X_np)
    # 使用 NoOpTransformer 对象拟合 X_np 数据
    with pytest.warns(UserWarning, match=msg):
        # 使用 pytest 检测警告，期望匹配给定的警告信息字符串
        trans.transform(df)
        # 对 df 进行转换操作，捕获到匹配的 UserWarning 警告

    # fit on dataframe with all integer feature names works without warning
    # 在所有整数特征名的数据框上拟合，不会产生警告
    df_int_names = pd.DataFrame(X_np)
    trans = NoOpTransformer()
    with warnings.catch_warnings():
        # 使用 warnings 模块捕获警告
        warnings.simplefilter("error", UserWarning)
        # 设置简单过滤器，将 UserWarning 转换为错误
        trans.fit(df_int_names)
        # 对 df_int_names 进行拟合操作

    # fit on dataframe with no feature names or all integer feature names
    # 在没有特征名或所有整数特征名的数据框上拟合，不会在转换时产生警告
    Xs = [X_np, df_int_names]
    for X in Xs:
        with warnings.catch_warnings():
            # 使用 warnings 模块捕获警告
            warnings.simplefilter("error", UserWarning)
            # 设置简单过滤器，将 UserWarning 转换为错误
            trans.transform(X)
            # 对 X 进行转换操作

    # fit on dataframe with feature names that are mixed raises an error:
    # 在包含混合特征名的数据框上拟合，会引发错误
    df_mixed = pd.DataFrame(X_np, columns=["a", "b", 1, 2])
    trans = NoOpTransformer()
    msg = re.escape(
        "Feature names are only supported if all input features have string names, "
        "but your input has ['int', 'str'] as feature name / column name types. "
        "If you want feature names to be stored and validated, you must convert "
        "them all to strings, by using X.columns = X.columns.astype(str) for "
        "example. Otherwise you can remove feature / column names from your input "
        "data, or convert them all to a non-string data type."
    )
    with pytest.raises(TypeError, match=msg):
        # 使用 pytest 检测预期的 TypeError 异常，并匹配指定的错误消息
        trans.fit(df_mixed)
        # 对 df_mixed 进行拟合操作，预期引发 TypeError 异常

    # transform on feature names that are mixed also raises:
    # 在包含混合特征名的数据上进行转换，同样会引发错误
    with pytest.raises(TypeError, match=msg):
        # 使用 pytest 检测预期的 TypeError 异常，并匹配指定的错误消息
        trans.transform(df_mixed)
        # 对 df_mixed 进行转换操作，预期引发 TypeError 异常
def test_validate_data_cast_to_ndarray():
    """Check cast_to_ndarray option of _validate_data."""

    # 导入 pytest 库，如果不存在则跳过测试
    pd = pytest.importorskip("pandas")
    # 加载鸢尾花数据集
    iris = datasets.load_iris()
    # 创建包含数据的 DataFrame
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    # 创建包含目标值的 Series
    y = pd.Series(iris.target)

    # 定义一个空的转换器类，继承自 TransformerMixin 和 BaseEstimator
    class NoOpTransformer(TransformerMixin, BaseEstimator):
        pass

    # 创建一个 NoOpTransformer 实例
    no_op = NoOpTransformer()

    # 测试 _validate_data 方法，将 DataFrame 转换为 ndarray
    X_np_out = no_op._validate_data(df, cast_to_ndarray=True)
    assert isinstance(X_np_out, np.ndarray)
    assert_allclose(X_np_out, df.to_numpy())

    # 测试 _validate_data 方法，不进行任何转换，直接返回 DataFrame
    X_df_out = no_op._validate_data(df, cast_to_ndarray=False)
    assert X_df_out is df

    # 测试 _validate_data 方法，将 Series 转换为 ndarray
    y_np_out = no_op._validate_data(y=y, cast_to_ndarray=True)
    assert isinstance(y_np_out, np.ndarray)
    assert_allclose(y_np_out, y.to_numpy())

    # 测试 _validate_data 方法，不进行任何转换，直接返回 Series
    y_series_out = no_op._validate_data(y=y, cast_to_ndarray=False)
    assert y_series_out is y

    # 测试 _validate_data 方法，同时转换 X 和 y 为 ndarray
    X_np_out, y_np_out = no_op._validate_data(df, y, cast_to_ndarray=True)
    assert isinstance(X_np_out, np.ndarray)
    assert_allclose(X_np_out, df.to_numpy())
    assert isinstance(y_np_out, np.ndarray)
    assert_allclose(y_np_out, y.to_numpy())

    # 测试 _validate_data 方法，不进行任何转换，直接返回 X 和 y
    X_df_out, y_series_out = no_op._validate_data(df, y, cast_to_ndarray=False)
    assert X_df_out is df
    assert y_series_out is y

    # 测试 _validate_data 方法，捕获预期的 ValueError 异常
    msg = "Validation should be done on X, y or both."
    with pytest.raises(ValueError, match=msg):
        no_op._validate_data()


def test_clone_keeps_output_config():
    """Check that clone keeps the set_output config."""

    # 创建一个标准缩放器，并设置输出为 pandas DataFrame
    ss = StandardScaler().set_output(transform="pandas")
    # 获取当前输出配置
    config = _get_output_config("transform", ss)

    # 克隆标准缩放器实例
    ss_clone = clone(ss)
    # 获取克隆后的输出配置
    config_clone = _get_output_config("transform", ss_clone)
    # 断言克隆后的配置与原配置相同
    assert config == config_clone


class _Empty:
    pass


class EmptyEstimator(_Empty, BaseEstimator):
    pass


@pytest.mark.parametrize("estimator", [BaseEstimator(), EmptyEstimator()])
def test_estimator_empty_instance_dict(estimator):
    """Check that ``__getstate__`` returns an empty ``dict`` with an empty
    instance.

    Python 3.11+ changed behaviour by returning ``None`` instead of raising an
    ``AttributeError``. Non-regression test for gh-25188.
    """
    # 获取估算器的状态字典
    state = estimator.__getstate__()
    # 预期的状态字典
    expected = {"_sklearn_version": sklearn.__version__}
    # 断言获取的状态字典与预期的状态字典相同
    assert state == expected

    # 不应该引发异常
    pickle.loads(pickle.dumps(BaseEstimator()))


def test_estimator_getstate_using_slots_error_message():
    """Using a `BaseEstimator` with `__slots__` is not supported."""

    # 定义一个带有 __slots__ 的类
    class WithSlots:
        __slots__ = ("x",)

    # 创建一个 Estimator 类，继承自 BaseEstimator 和 WithSlots
    class Estimator(BaseEstimator, WithSlots):
        pass

    # 预期的错误消息
    msg = (
        "You cannot use `__slots__` in objects inheriting from "
        "`sklearn.base.BaseEstimator`"
    )

    # 使用 pytest 捕获预期的 TypeError 异常，并检查错误消息是否匹配预期
    with pytest.raises(TypeError, match=msg):
        Estimator().__getstate__()

    # 使用 pickle 捕获预期的 TypeError 异常，并检查错误消息是否匹配预期
    with pytest.raises(TypeError, match=msg):
        pickle.dumps(Estimator())


@pytest.mark.parametrize(
    "constructor_name, minversion",
    [
        ("dataframe", "1.5.0"),   # 元组: 第一个元素是模块名 "dataframe"，第二个元素是版本号 "1.5.0"
        ("pyarrow", "12.0.0"),    # 元组: 第一个元素是模块名 "pyarrow"，第二个元素是版本号 "12.0.0"
        ("polars", "0.20.23"),    # 元组: 第一个元素是模块名 "polars"，第二个元素是版本号 "0.20.23"
    ],
def test_dataframe_protocol(constructor_name, minversion):
    """Uses the dataframe exchange protocol to get feature names."""
    # 示例：创建一个包含数据的列表
    data = [[1, 4, 2], [3, 3, 6]]
    # 示例：指定数据列的名称
    columns = ["col_0", "col_1", "col_2"]
    # 示例：调用_convert_container函数，将数据转换为指定的数据容器对象
    df = _convert_container(
        data, constructor_name, columns_name=columns, minversion=minversion
    )

    # 示例：定义一个空操作的转换器类，实现了TransformerMixin和BaseEstimator接口
    class NoOpTransformer(TransformerMixin, BaseEstimator):
        # 示例：实现fit方法，对输入的数据进行验证
        def fit(self, X, y=None):
            self._validate_data(X)
            return self

        # 示例：实现transform方法，对输入的数据进行验证（不重置）
        def transform(self, X):
            return self._validate_data(X, reset=False)

    # 示例：创建NoOpTransformer的实例
    no_op = NoOpTransformer()
    # 示例：使用df来拟合（训练）转换器
    no_op.fit(df)
    # 示例：验证转换器的输出特征名与预期的列名相等
    assert_array_equal(no_op.feature_names_in_, columns)
    # 示例：使用转换器对df进行转换操作，获取输出X_out
    X_out = no_op.transform(df)

    # 示例：如果constructor_name不是"pyarrow"，则进行数据近似相等性断言
    if constructor_name != "pyarrow":
        assert_allclose(df, X_out)

    # 示例：定义一组错误的特征名
    bad_names = ["a", "b", "c"]
    # 示例：使用错误的特征名创建另一个转换器df_bad
    df_bad = _convert_container(data, constructor_name, columns_name=bad_names)
    # 示例：使用pytest.raises断言，验证对df_bad进行转换时抛出值错误异常
    with pytest.raises(ValueError, match="The feature names should match"):
        no_op.transform(df_bad)


@pytest.mark.usefixtures("enable_slep006")
def test_transformer_fit_transform_with_metadata_in_transform():
    """Test that having a transformer with metadata for transform raises a
    warning when calling fit_transform."""

    # 示例：定义一个自定义转换器类，实现了BaseEstimator和TransformerMixin接口
    class CustomTransformer(BaseEstimator, TransformerMixin):
        # 示例：实现fit方法，用于拟合（训练）数据
        def fit(self, X, y=None, prop=None):
            return self

        # 示例：实现transform方法，用于转换数据
        def transform(self, X, prop=None):
            return X

    # 示例：调用fit_transform时传递元数据prop=True，应该触发UserWarning警告
    with pytest.warns(UserWarning, match="`transform` method which consumes metadata"):
        CustomTransformer().set_transform_request(prop=True).fit_transform(
            [[1]], [1], prop=1
        )

    # 示例：不传递可能被transform方法消耗的元数据prop=True，不应该触发警告
    with warnings.catch_warnings(record=True) as record:
        CustomTransformer().set_transform_request(prop=True).fit_transform([[1]], [1])
        assert len(record) == 0


@pytest.mark.usefixtures("enable_slep006")
def test_outlier_mixin_fit_predict_with_metadata_in_predict():
    """Test that having an OutlierMixin with metadata for predict raises a
    warning when calling fit_predict."""

    # 示例：定义一个自定义异常检测器类，实现了BaseEstimator和OutlierMixin接口
    class CustomOutlierDetector(BaseEstimator, OutlierMixin):
        # 示例：实现fit方法，用于拟合（训练）数据
        def fit(self, X, y=None, prop=None):
            return self

        # 示例：实现predict方法，用于预测数据
        def predict(self, X, prop=None):
            return X

    # 示例：调用fit_predict时传递元数据prop=True，应该触发UserWarning警告
    with pytest.warns(UserWarning, match="`predict` method which consumes metadata"):
        CustomOutlierDetector().set_predict_request(prop=True).fit_predict(
            [[1]], [1], prop=1
        )

    # 示例：不传递可能被predict方法消耗的元数据prop=True，不应该触发警告
    with warnings.catch_warnings(record=True) as record:
        CustomOutlierDetector().set_predict_request(prop=True).fit_predict([[1]], [1])
    # 不会触发警告
    # 使用 catch_warnings() 上下文管理器捕获警告信息
    with warnings.catch_warnings(record=True) as record:
        # 创建 CustomOutlierDetector 实例，设置预测请求属性为真，拟合并预测数据
        CustomOutlierDetector().set_predict_request(prop=True).fit_predict([[1]], [1])
        # 断言捕获的警告信息长度为0
        assert len(record) == 0
```