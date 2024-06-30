# `D:\src\scipysrc\scikit-learn\sklearn\tests\test_metaestimators.py`

```
"""Common tests for metaestimators"""

# 导入必要的模块和函数
import functools  # 导入 functools 模块，用于创建高阶函数
from inspect import signature  # 导入 signature 函数，用于获取函数的参数签名信息

import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试用例

from sklearn.base import BaseEstimator, is_regressor  # 导入基础估计器和判断是否回归器的函数
from sklearn.datasets import make_classification  # 导入用于生成分类数据集的函数
from sklearn.ensemble import BaggingClassifier  # 导入 Bagging 分类器
from sklearn.exceptions import NotFittedError  # 导入未拟合错误
from sklearn.feature_extraction.text import TfidfVectorizer  # 导入用于文本特征提取的 TF-IDF 向量化器
from sklearn.feature_selection import RFE, RFECV  # 导入递归特征消除和交叉验证递归特征消除
from sklearn.linear_model import LogisticRegression, Ridge  # 导入逻辑回归和岭回归模型
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV  # 导入网格搜索交叉验证和随机搜索交叉验证
from sklearn.pipeline import Pipeline, make_pipeline  # 导入管道和创建管道的函数
from sklearn.preprocessing import MaxAbsScaler, StandardScaler  # 导入最大绝对值缩放和标准缩放器
from sklearn.semi_supervised import SelfTrainingClassifier  # 导入自训练分类器
from sklearn.utils import all_estimators  # 导入所有可用估计器
from sklearn.utils._testing import set_random_state  # 导入设置随机状态的函数
from sklearn.utils.estimator_checks import (
    _enforce_estimator_tags_X,  # 导入强制估计器特性的函数（针对 X）
    _enforce_estimator_tags_y,  # 导入强制估计器特性的函数（针对 y）
)
from sklearn.utils.validation import check_is_fitted  # 导入检查模型是否已拟合的函数


class DelegatorData:
    def __init__(
        self,
        name,
        construct,
        skip_methods=(),
        fit_args=make_classification(random_state=0),
    ):
        # 初始化 DelegatorData 类的实例
        self.name = name  # 设置实例的名称属性
        self.construct = construct  # 设置实例的构造函数属性
        self.fit_args = fit_args  # 设置实例的拟合参数属性
        self.skip_methods = skip_methods  # 设置实例的跳过方法属性


DELEGATING_METAESTIMATORS = [
    DelegatorData("Pipeline", lambda est: Pipeline([("est", est)])),  # 创建 Pipeline 的 DelegatorData 实例
    DelegatorData(
        "GridSearchCV",
        lambda est: GridSearchCV(est, param_grid={"param": [5]}, cv=2),  # 创建 GridSearchCV 的 DelegatorData 实例
        skip_methods=["score"],  # 指定跳过的方法
    ),
    DelegatorData(
        "RandomizedSearchCV",
        lambda est: RandomizedSearchCV(
            est, param_distributions={"param": [5]}, cv=2, n_iter=1  # 创建 RandomizedSearchCV 的 DelegatorData 实例
        ),
        skip_methods=["score"],  # 指定跳过的方法
    ),
    DelegatorData("RFE", RFE, skip_methods=["transform", "inverse_transform"]),  # 创建 RFE 的 DelegatorData 实例
    DelegatorData("RFECV", RFECV, skip_methods=["transform", "inverse_transform"]),  # 创建 RFECV 的 DelegatorData 实例
    DelegatorData(
        "BaggingClassifier",
        BaggingClassifier,
        skip_methods=[
            "transform",
            "inverse_transform",
            "score",
            "predict_proba",
            "predict_log_proba",
            "predict",
        ],  # 创建 BaggingClassifier 的 DelegatorData 实例，并指定跳过的方法
    ),
    DelegatorData(
        "SelfTrainingClassifier",
        lambda est: SelfTrainingClassifier(est),  # 创建 SelfTrainingClassifier 的 DelegatorData 实例
        skip_methods=["transform", "inverse_transform", "predict_proba"],  # 指定跳过的方法
    ),
]


def test_metaestimator_delegation():
    # 测试元估计器是否正确委托子估计器的方法
    def hides(method):
        @property
        def wrapper(obj):
            if obj.hidden_method == method.__name__:
                raise AttributeError("%r is hidden" % obj.hidden_method)
            return functools.partial(method, obj)

        return wrapper
    class SubEstimator(BaseEstimator):
        # 定义一个子估计器类，继承自BaseEstimator

        def __init__(self, param=1, hidden_method=None):
            # 构造函数，初始化实例变量param和hidden_method
            self.param = param
            self.hidden_method = hidden_method

        def fit(self, X, y=None, *args, **kwargs):
            # 拟合方法，设置coef_为从0到X.shape[1]的数组，classes_为空列表，然后返回True
            self.coef_ = np.arange(X.shape[1])
            self.classes_ = []
            return True

        def _check_fit(self):
            # 检查是否已经拟合的内部方法
            check_is_fitted(self)

        @hides
        def inverse_transform(self, X, *args, **kwargs):
            # 调用_check_fit方法检查是否已经拟合，然后返回输入的X
            self._check_fit()
            return X

        @hides
        def transform(self, X, *args, **kwargs):
            # 调用_check_fit方法检查是否已经拟合，然后返回输入的X
            self._check_fit()
            return X

        @hides
        def predict(self, X, *args, **kwargs):
            # 调用_check_fit方法检查是否已经拟合，然后返回一个与X.shape[0]大小相同的全1数组
            self._check_fit()
            return np.ones(X.shape[0])

        @hides
        def predict_proba(self, X, *args, **kwargs):
            # 调用_check_fit方法检查是否已经拟合，然后返回一个与X.shape[0]大小相同的全1数组
            self._check_fit()
            return np.ones(X.shape[0])

        @hides
        def predict_log_proba(self, X, *args, **kwargs):
            # 调用_check_fit方法检查是否已经拟合，然后返回一个与X.shape[0]大小相同的全1数组
            self._check_fit()
            return np.ones(X.shape[0])

        @hides
        def decision_function(self, X, *args, **kwargs):
            # 调用_check_fit方法检查是否已经拟合，然后返回一个与X.shape[0]大小相同的全1数组
            self._check_fit()
            return np.ones(X.shape[0])

        @hides
        def score(self, X, y, *args, **kwargs):
            # 调用_check_fit方法检查是否已经拟合，然后返回1.0
            self._check_fit()
            return 1.0

    methods = [
        k
        for k in SubEstimator.__dict__.keys()  # 遍历SubEstimator类的所有属性名
        if not k.startswith("_") and not k.startswith("fit")  # 筛选出不以"_"和"fit"开头的属性名
    ]
    methods.sort()  # 对筛选出的属性名进行排序
    # 遍历 DELEGATING_METAESTIMATORS 列表中的每个元素
    for delegator_data in DELEGATING_METAESTIMATORS:
        # 创建子估计器实例
        delegate = SubEstimator()
        # 使用 delegator_data 中的 construct 方法构建 delegator
        delegator = delegator_data.construct(delegate)
        # 遍历 methods 列表中的每个方法名
        for method in methods:
            # 如果当前方法名在 delegator_data 的 skip_methods 列表中，则跳过本次循环
            if method in delegator_data.skip_methods:
                continue
            # 断言 delegate 对象具有当前方法
            assert hasattr(delegate, method)
            # 断言 delegator 对象具有当前方法；如果不具备，则抛出异常
            assert hasattr(
                delegator, method
            ), "%s does not have method %r when its delegate does" % (
                delegator_data.name,
                method,
            )
            # 对于方法名为 "score" 的情况，验证在调用 fit 前调用 delegator 的 score 方法会引发 NotFittedError
            if method == "score":
                with pytest.raises(NotFittedError):
                    getattr(delegator, method)(
                        delegator_data.fit_args[0], delegator_data.fit_args[1]
                    )
            else:
                # 对于其他方法，验证在调用 fit 前调用 delegator 的方法会引发 NotFittedError
                with pytest.raises(NotFittedError):
                    getattr(delegator, method)(delegator_data.fit_args[0])

        # 使用 delegator_data 的 fit_args 参数调用 delegator 的 fit 方法
        delegator.fit(*delegator_data.fit_args)

        # 再次遍历 methods 列表中的每个方法名
        for method in methods:
            # 如果当前方法名在 delegator_data 的 skip_methods 列表中，则跳过本次循环
            if method in delegator_data.skip_methods:
                continue
            # 对于方法名为 "score" 的情况，验证调用 delegator 的 score 方法是否正常执行
            if method == "score":
                getattr(delegator, method)(
                    delegator_data.fit_args[0], delegator_data.fit_args[1]
                )
            else:
                # 对于其他方法，验证调用 delegator 的方法是否正常执行
                getattr(delegator, method)(delegator_data.fit_args[0])

        # 再次遍历 methods 列表中的每个方法名
        for method in methods:
            # 如果当前方法名在 delegator_data 的 skip_methods 列表中，则跳过本次循环
            if method in delegator_data.skip_methods:
                continue
            # 使用 hidden_method 参数创建新的 delegate 实例
            delegate = SubEstimator(hidden_method=method)
            # 使用 delegator_data 的 construct 方法构建新的 delegator
            delegator = delegator_data.construct(delegate)
            # 断言 delegate 对象不具有当前方法
            assert not hasattr(delegate, method)
            # 断言 delegator 对象不具有当前方法；如果具备，则抛出异常
            assert not hasattr(
                delegator, method
            ), "%s has method %r when its delegate does not" % (
                delegator_data.name,
                method,
            )
def _generate_meta_estimator_instances_with_pipeline():
    """Generate instances of meta-estimators fed with a pipeline

    Are considered meta-estimators all estimators accepting one of "estimator",
    "base_estimator" or "estimators".
    """
    # 遍历所有的估算器
    for _, Estimator in sorted(all_estimators()):
        # 获取估算器的参数签名
        sig = set(signature(Estimator).parameters)

        # 如果参数中包含"estimator"、"base_estimator"或者"regressor"
        if "estimator" in sig or "base_estimator" in sig or "regressor" in sig:
            # 如果是回归器类型的估算器
            if is_regressor(Estimator):
                # 创建一个管道，包括TfidfVectorizer和Ridge回归器
                estimator = make_pipeline(TfidfVectorizer(), Ridge())
                # 设置参数网格
                param_grid = {"ridge__alpha": [0.1, 1.0]}
            else:
                # 创建一个管道，包括TfidfVectorizer和LogisticRegression分类器
                estimator = make_pipeline(TfidfVectorizer(), LogisticRegression())
                # 设置参数网格
                param_grid = {"logisticregression__C": [0.1, 1.0]}

            # 如果参数中包含"param_grid"或者"param_distributions"
            if "param_grid" in sig or "param_distributions" in sig:
                # 对于SearchCV类型的估算器，添加额外的参数
                extra_params = {"n_iter": 2} if "n_iter" in sig else {}
                # 生成估算器实例
                yield Estimator(estimator, param_grid, **extra_params)
            else:
                # 对于普通的估算器，生成估算器实例
                yield Estimator(estimator)

        # 如果参数中包含"transformer_list"
        elif "transformer_list" in sig:
            # 创建特征联合器的转换器列表
            transformer_list = [
                ("trans1", make_pipeline(TfidfVectorizer(), MaxAbsScaler())),
                ("trans2", make_pipeline(TfidfVectorizer(), StandardScaler(with_mean=False))),
            ]
            # 生成估算器实例
            yield Estimator(transformer_list)

        # 如果参数中包含"estimators"
        elif "estimators" in sig:
            # 如果是回归器类型的估算器
            if is_regressor(Estimator):
                # 创建回归器列表
                estimator = [
                    ("est1", make_pipeline(TfidfVectorizer(), Ridge(alpha=0.1))),
                    ("est2", make_pipeline(TfidfVectorizer(), Ridge(alpha=1))),
                ]
            else:
                # 创建分类器列表
                estimator = [
                    ("est1", make_pipeline(TfidfVectorizer(), LogisticRegression(C=0.1))),
                    ("est2", make_pipeline(TfidfVectorizer(), LogisticRegression(C=1))),
                ]
            # 生成估算器实例
            yield Estimator(estimator)

        else:
            # 如果不符合上述条件，继续下一个循环
            continue


# TODO: remove data validation for the following estimators
# They should be able to work on any data and delegate data validation to
# their inner estimator(s).
DATA_VALIDATION_META_ESTIMATORS_TO_IGNORE = [
    "AdaBoostClassifier",
    "AdaBoostRegressor",
    "BaggingClassifier",
    "BaggingRegressor",
    "ClassifierChain",  # data validation is necessary
    "IterativeImputer",
    "OneVsOneClassifier",  # input validation can't be avoided
    "RANSACRegressor",
    "RFE",
    "RFECV",
    "RegressorChain",  # data validation is necessary
    "SelfTrainingClassifier",
    "SequentialFeatureSelector",  # not applicable (2D data mandatory)
]

# 生成包含管道的元估算器实例列表，用于数据验证的元估算器应被忽略
DATA_VALIDATION_META_ESTIMATORS = [
    est
    for est in _generate_meta_estimator_instances_with_pipeline()
    # 将符合条件的元估算器添加到列表中
    if est.__name__ in DATA_VALIDATION_META_ESTIMATORS_TO_IGNORE
]
    # 如果 est 对象的类名不在 DATA_VALIDATION_META_ESTIMATORS_TO_IGNORE 列表中
# 导入必要的模块或函数
import pytest
import numpy as np

# 定义一个辅助函数，用于获取估算器的类名作为标识符
def _get_meta_estimator_id(estimator):
    return estimator.__class__.__name__

# 使用pytest的parametrize装饰器，为测试函数test_meta_estimators_delegate_data_validation参数化测试用例
@pytest.mark.parametrize(
    "estimator", DATA_VALIDATION_META_ESTIMATORS, ids=_get_meta_estimator_id
)
# 定义测试函数，测试元估计器是否将数据验证委托给内部估计器(s)
def test_meta_estimators_delegate_data_validation(estimator):
    # 检查元估计器是否将数据验证委托给内部估计器(s)
    rng = np.random.RandomState(0)
    # 设置随机状态，确保可重现性
    set_random_state(estimator)

    # 生成包含30个样本的随机数据
    n_samples = 30
    X = rng.choice(np.array(["aa", "bb", "cc"], dtype=object), size=n_samples)

    # 根据估计器类型生成对应的随机目标值y
    if is_regressor(estimator):
        y = rng.normal(size=n_samples)
    else:
        y = rng.randint(3, size=n_samples)

    # 将X和y转换为列表，以确保它们在类数组数据上也能正常工作
    X = _enforce_estimator_tags_X(estimator, X).tolist()
    y = _enforce_estimator_tags_y(estimator, y).tolist()

    # 调用fit方法，应该不会因为数据验证问题而引发异常，因为X是作为基本估计器传递给元估计器的流水线的有效输入数据结构。
    estimator.fit(X, y)

    # 断言n_features_in_属性不应该被定义，因为数据不是表格数据。
    assert not hasattr(estimator, "n_features_in_")
```