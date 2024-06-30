# `D:\src\scipysrc\scikit-learn\sklearn\tests\metadata_routing_common.py`

```
# 导入inspect模块，用于获取函数调用栈信息
import inspect
# 导入defaultdict类，用于创建默认字典
from collections import defaultdict
# 导入partial函数，用于创建偏函数
from functools import partial

# 导入numpy库，并重命名为np
import numpy as np
# 从numpy.testing模块导入assert_array_equal函数，用于比较数组是否相等
from numpy.testing import assert_array_equal

# 从sklearn.base模块导入多个Mixin类和函数
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MetaEstimatorMixin,
    RegressorMixin,
    TransformerMixin,
    clone,
)
# 从sklearn.metrics._scorer模块导入_Scorer类和mean_squared_error函数
from sklearn.metrics._scorer import _Scorer, mean_squared_error
# 从sklearn.model_selection模块导入BaseCrossValidator类
from sklearn.model_selection import BaseCrossValidator
# 从sklearn.model_selection._split模块导入GroupsConsumerMixin类
from sklearn.model_selection._split import GroupsConsumerMixin
# 从sklearn.utils._metadata_requests模块导入SIMPLE_METHODS常量
from sklearn.utils._metadata_requests import (
    SIMPLE_METHODS,
)
# 从sklearn.utils.metadata_routing模块导入MetadataRouter类、MethodMapping类和process_routing函数
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    process_routing,
)
# 从sklearn.utils.multiclass模块导入_check_partial_fit_first_call函数
from sklearn.utils.multiclass import _check_partial_fit_first_call


def record_metadata(obj, record_default=True, **kwargs):
    """Utility function to store passed metadata to a method of obj.

    If record_default is False, kwargs whose values are "default" are skipped.
    This is so that checks on keyword arguments whose default was not changed
    are skipped.

    """
    # 获取调用栈信息
    stack = inspect.stack()
    # 获取调用当前函数的函数名
    callee = stack[1].function
    # 获取调用当前函数的上一层函数名（调用者的函数名）
    caller = stack[2].function
    # 如果对象没有属性"_records"，则创建一个默认字典作为其属性
    if not hasattr(obj, "_records"):
        obj._records = defaultdict(lambda: defaultdict(list))
    # 如果record_default为False，则过滤掉值为"default"的kwargs
    if not record_default:
        kwargs = {
            key: val
            for key, val in kwargs.items()
            if not isinstance(val, str) or (val != "default")
        }
    # 将传递的kwargs记录到对象的_records属性中对应的callee和caller下
    obj._records[callee][caller].append(kwargs)


def check_recorded_metadata(obj, method, parent, split_params=tuple(), **kwargs):
    """Check whether the expected metadata is passed to the object's method.

    Parameters
    ----------
    obj : estimator object
        sub-estimator to check routed params for
    method : str
        sub-estimator's method where metadata is routed to, or otherwise in
        the context of metadata routing referred to as 'callee'
    parent : str
        the parent method which should have called `method`, or otherwise in
        the context of metadata routing referred to as 'caller'
    split_params : tuple, default=empty
        specifies any parameters which are to be checked as being a subset
        of the original values
    **kwargs : dict
        passed metadata
    """
    # 获取对象的_records属性中method和parent下的所有记录
    all_records = (
        getattr(obj, "_records", dict()).get(method, dict()).get(parent, list())
    )
    # 对于所有的记录 `record` 在 `all_records` 中循环处理
    for record in all_records:
        # 首先检查传递的元数据名称与预期名称是否相同，预期的名称作为 `record` 的键存储
        assert set(kwargs.keys()) == set(
            record.keys()
        ), f"Expected {kwargs.keys()} vs {record.keys()}"
        
        # 遍历传递进来的关键字参数 `kwargs`，包括键 `key` 和对应的值 `value`
        for key, value in kwargs.items():
            # 获取记录中键 `key` 对应的值 `recorded_value`
            recorded_value = record[key]
            
            # 如果 `key` 存在于 `split_params` 中并且 `recorded_value` 不为 `None`
            if key in split_params and recorded_value is not None:
                # 使用 NumPy 的 `np.isin` 函数检查 `recorded_value` 是否是 `value` 的子集
                assert np.isin(recorded_value, value).all()
            else:
                # 否则，如果 `recorded_value` 是 NumPy 数组
                if isinstance(recorded_value, np.ndarray):
                    # 使用 `assert_array_equal` 函数比较 `recorded_value` 和 `value`
                    assert_array_equal(recorded_value, value)
                else:
                    # 否则，直接比较 `recorded_value` 和 `value`
                    assert (
                        recorded_value is value
                    ), f"Expected {recorded_value} vs {value}. Method: {method}"
# 使用 functools.partial 函数，创建一个新的函数 record_metadata_not_default，这个新函数是 record_metadata 的部分应用，其中 record_default 参数被设置为 False
record_metadata_not_default = partial(record_metadata, record_default=False)


def assert_request_is_empty(metadata_request, exclude=None):
    """Check if a metadata request dict is empty.

    One can exclude a method or a list of methods from the check using the
    ``exclude`` parameter. If metadata_request is a MetadataRouter, then
    ``exclude`` can be of the form ``{"object" : [method, ...]}``.
    """
    # 如果 metadata_request 是 MetadataRouter 的实例，则递归地检查其子路由器的空请求情况，可以根据 exclude 参数排除特定方法或方法列表
    if isinstance(metadata_request, MetadataRouter):
        for name, route_mapping in metadata_request:
            if exclude is not None and name in exclude:
                _exclude = exclude[name]
            else:
                _exclude = None
            assert_request_is_empty(route_mapping.router, exclude=_exclude)
        return

    # 如果 exclude 为 None，则将其设为空列表
    exclude = [] if exclude is None else exclude
    # 遍历 SIMPLE_METHODS 列表中的方法，检查是否在 exclude 中，如果不在则检查该方法的请求是否为空
    for method in SIMPLE_METHODS:
        if method in exclude:
            continue
        mmr = getattr(metadata_request, method)
        # 收集非空的请求属性并断言其为空
        props = [
            prop
            for prop, alias in mmr.requests.items()
            if isinstance(alias, str) or alias is not None
        ]
        assert not props


def assert_request_equal(request, dictionary):
    # 遍历字典中的每个方法及其请求，检查请求是否与字典中的请求相等
    for method, requests in dictionary.items():
        mmr = getattr(request, method)
        assert mmr.requests == requests

    # 检查是否有未在字典中的方法，它们的请求应为空
    empty_methods = [method for method in SIMPLE_METHODS if method not in dictionary]
    for method in empty_methods:
        assert not len(getattr(request, method).requests)


class _Registry(list):
    """A specialized list to reference sub-estimators.

    This list helps to reference sub-estimators which are not necessarily stored
    on the metaestimator. Overrides __deepcopy__ to ensure deepcopy and copy
    return the same instance of the list.
    """

    # 重写 __deepcopy__ 方法，使得深拷贝和浅拷贝返回同一个实例
    def __deepcopy__(self, memo):
        return self

    # 重写 __copy__ 方法，使得浅拷贝返回同一个实例
    def __copy__(self):
        return self


class ConsumingRegressor(RegressorMixin, BaseEstimator):
    """A regressor consuming metadata.

    Parameters
    ----------
    registry : list, default=None
        If a list, the estimator will append itself to the list in order to have
        a reference to the estimator later on. Since that reference is not
        required in all tests, registration can be skipped by leaving this value
        as None.
    """

    # 初始化方法，接受一个 registry 参数，用于存储对当前估算器的引用
    def __init__(self, registry=None):
        self.registry = registry

    # 部分拟合方法，用于将当前估算器的引用添加到 registry 中，并调用 record_metadata_not_default 函数
    def partial_fit(self, X, y, sample_weight="default", metadata="default"):
        if self.registry is not None:
            self.registry.append(self)

        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )
        return self
    # 定义一个方法用于模型训练，将自身注册到模型注册表中（如果存在）
    def fit(self, X, y, sample_weight="default", metadata="default"):
        # 如果模型注册表不为空，则将当前模型实例添加到注册表中
        if self.registry is not None:
            self.registry.append(self)

        # 记录非默认的元数据和样本权重
        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )

        # 返回当前模型实例
        return self

    # 定义一个方法用于模型预测
    def predict(self, X, y=None, sample_weight="default", metadata="default"):
        # 记录非默认的元数据和样本权重
        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )

        # 返回一个全为零的预测结果数组，其长度与输入数据 X 的行数相同
        return np.zeros(shape=(len(X),))

    # 定义一个方法用于评估模型得分
    def score(self, X, y, sample_weight="default", metadata="default"):
        # 记录非默认的元数据和样本权重
        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )

        # 返回一个默认得分值 1，因为该方法的具体实现并未提供更多细节
        return 1
class NonConsumingClassifier(ClassifierMixin, BaseEstimator):
    """A classifier which accepts no metadata on any method."""

    def __init__(self, alpha=0.0):
        self.alpha = alpha

    def fit(self, X, y):
        # 获取类别信息并保存到实例变量
        self.classes_ = np.unique(y)
        return self

    def partial_fit(self, X, y, classes=None):
        # 返回自身实例，不做任何操作
        return self

    def decision_function(self, X):
        # 调用预测方法返回预测结果
        return self.predict(X)

    def predict(self, X):
        # 创建一个空的预测结果数组
        y_pred = np.empty(shape=(len(X),))
        # 将前一半设为类别 0，后一半设为类别 1
        y_pred[: len(X) // 2] = 0
        y_pred[len(X) // 2 :] = 1
        return y_pred


class NonConsumingRegressor(RegressorMixin, BaseEstimator):
    """A classifier which accepts no metadata on any method."""

    def fit(self, X, y):
        # 返回自身实例，不做任何操作
        return self

    def partial_fit(self, X, y):
        # 返回自身实例，不做任何操作
        return self

    def predict(self, X):
        # 返回一个全部为 1 的预测结果数组
        return np.ones(len(X))  # pragma: no cover


class ConsumingClassifier(ClassifierMixin, BaseEstimator):
    """A classifier consuming metadata.

    Parameters
    ----------
    registry : list, default=None
        If a list, the estimator will append itself to the list in order to have
        a reference to the estimator later on. Since that reference is not
        required in all tests, registration can be skipped by leaving this value
        as None.

    alpha : float, default=0
        This parameter is only used to test the ``*SearchCV`` objects, and
        doesn't do anything.
    """

    def __init__(self, registry=None, alpha=0.0):
        # 初始化 alpha 和 registry 实例变量
        self.alpha = alpha
        self.registry = registry

    def partial_fit(
        self, X, y, classes=None, sample_weight="default", metadata="default"
    ):
        # 如果 registry 不为 None，则将自身添加到 registry 中
        if self.registry is not None:
            self.registry.append(self)

        # 记录非默认的元数据
        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )
        # 检查部分拟合的第一次调用
        _check_partial_fit_first_call(self, classes)
        return self

    def fit(self, X, y, sample_weight="default", metadata="default"):
        # 如果 registry 不为 None，则将自身添加到 registry 中
        if self.registry is not None:
            self.registry.append(self)

        # 记录非默认的元数据
        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )

        # 获取类别信息并保存到实例变量
        self.classes_ = np.unique(y)
        return self

    def predict(self, X, sample_weight="default", metadata="default"):
        # 记录非默认的元数据
        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )
        # 创建一个空的预测结果数组
        y_score = np.empty(shape=(len(X),), dtype="int8")
        # 将后一半设为类别 0，前一半设为类别 1
        y_score[len(X) // 2 :] = 0
        y_score[: len(X) // 2] = 1
        return y_score

    def predict_proba(self, X, sample_weight="default", metadata="default"):
        # 记录非默认的元数据
        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )
        # 创建一个空的概率预测结果数组
        y_proba = np.empty(shape=(len(X), 2))
        # 将前一半的概率设为 [1.0, 0.0]，后一半的概率设为 [0.0, 1.0]
        y_proba[: len(X) // 2, :] = np.asarray([1.0, 0.0])
        y_proba[len(X) // 2 :, :] = np.asarray([0.0, 1.0])
        return y_proba
    # 定义一个方法用于预测对数概率，接受输入参数 X、样本权重和元数据，默认情况下使用 "default"
    def predict_log_proba(self, X, sample_weight="default", metadata="default"):
        pass  # pragma: no cover  # 该方法暂时不实现，不会被测试覆盖到

        # 当需要时取消注释以下代码
        # 调用记录非默认元数据的函数，传入当前对象及相关参数
        # record_metadata_not_default(
        #     self, sample_weight=sample_weight, metadata=metadata
        # )
        # 返回一个形状为 (样本数, 2) 的零数组
        # return np.zeros(shape=(len(X), 2))

    # 定义一个方法用于决策函数计算，接受输入参数 X、样本权重和元数据，默认情况下使用 "default"
    def decision_function(self, X, sample_weight="default", metadata="default"):
        # 调用记录非默认元数据的函数，传入当前对象及相关参数
        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )
        # 创建一个形状为 (样本数,) 的空数组 y_score
        y_score = np.empty(shape=(len(X),))
        # 将后半部分样本的得分设为 0
        y_score[len(X) // 2 :] = 0
        # 将前半部分样本的得分设为 1
        y_score[: len(X) // 2] = 1
        # 返回计算得到的分数数组
        return y_score

    # 当需要时取消注释以下代码
    # 定义一个方法用于计算评分，接受输入参数 X、y、样本权重和元数据，默认情况下使用 "default"
    # def score(self, X, y, sample_weight="default", metadata="default"):
    #     # 调用记录非默认元数据的函数，传入当前对象及相关参数
    #     record_metadata_not_default(
    #         self, sample_weight=sample_weight, metadata=metadata
    #     )
    #     # 返回固定的评分值 1
    #     return 1
class ConsumingTransformer(TransformerMixin, BaseEstimator):
    """A transformer which accepts metadata on fit and transform.

    Parameters
    ----------
    registry : list, default=None
        If a list, the estimator will append itself to the list in order to have
        a reference to the estimator later on. Since that reference is not
        required in all tests, registration can be skipped by leaving this value
        as None.
    """

    def __init__(self, registry=None):
        # 初始化方法，设置注册表属性
        self.registry = registry

    def fit(self, X, y=None, sample_weight="default", metadata="default"):
        # 如果注册表不为 None，则将自身添加到注册表中
        if self.registry is not None:
            self.registry.append(self)

        # 调用记录非默认元数据的函数，记录样本权重和元数据
        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )
        return self

    def transform(self, X, sample_weight="default", metadata="default"):
        # 调用记录非默认元数据的函数，记录样本权重和元数据
        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )
        # 返回转换后的数据，这里简单地返回输入数据加一
        return X + 1

    def fit_transform(self, X, y, sample_weight="default", metadata="default"):
        # 实现 fit_transform 方法，因为 TransformerMixin 的 fit_transform 方法
        # 不会将元数据传递给 transform 方法，但是这里希望 transform 方法接收
        # sample_weight 和 metadata。
        
        # 调用记录非默认元数据的函数，记录样本权重和元数据
        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )
        # 调用自身的 fit 方法，再调用 transform 方法，返回结果
        return self.fit(X, y, sample_weight=sample_weight, metadata=metadata).transform(
            X, sample_weight=sample_weight, metadata=metadata
        )

    def inverse_transform(self, X, sample_weight=None, metadata=None):
        # 调用记录非默认元数据的函数，记录样本权重和元数据
        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )
        # 返回逆转换后的数据，这里简单地返回输入数据减一
        return X - 1


class ConsumingNoFitTransformTransformer(BaseEstimator):
    """A metadata consuming transformer that doesn't inherit from
    TransformerMixin, and thus doesn't implement `fit_transform`. Note that
    TransformerMixin's `fit_transform` doesn't route metadata to `transform`."""

    def __init__(self, registry=None):
        # 初始化方法，设置注册表属性
        self.registry = registry

    def fit(self, X, y=None, sample_weight=None, metadata=None):
        # 如果注册表不为 None，则将自身添加到注册表中
        if self.registry is not None:
            self.registry.append(self)

        # 调用记录元数据的函数，记录样本权重和元数据
        record_metadata(self, sample_weight=sample_weight, metadata=metadata)

        return self

    def transform(self, X, sample_weight=None, metadata=None):
        # 调用记录元数据的函数，记录样本权重和元数据
        record_metadata(self, sample_weight=sample_weight, metadata=metadata)
        # 返回转换后的数据
        return X


class ConsumingScorer(_Scorer):
    def __init__(self, registry=None):
        # 调用父类的初始化方法，设置评分函数、符号、空字典参数和响应方法
        super().__init__(
            score_func=mean_squared_error, sign=1, kwargs={}, response_method="predict"
        )
        # 初始化方法，设置注册表属性
        self.registry = registry
    # 定义一个方法 `_score`，接受多个参数：`method_caller`, `clf`, `X`, `y` 和任意关键字参数 `kwargs`
    def _score(self, method_caller, clf, X, y, **kwargs):
        # 如果注册表不为 None，则将当前对象 `self` 添加到注册表中
        if self.registry is not None:
            self.registry.append(self)

        # 调用 `record_metadata_not_default` 函数，传入当前对象 `self` 和所有关键字参数 `kwargs`
        record_metadata_not_default(self, **kwargs)

        # 获取关键字参数 `sample_weight` 的值，如果不存在则默认为 `None`
        sample_weight = kwargs.get("sample_weight", None)
        
        # 调用父类的 `_score` 方法，传入 `method_caller`, `clf`, `X`, `y` 和 `sample_weight` 参数，
        # 返回其结果
        return super()._score(method_caller, clf, X, y, sample_weight=sample_weight)
# 定义一个自定义的交叉验证器类，同时继承了GroupsConsumerMixin和BaseCrossValidator
class ConsumingSplitter(GroupsConsumerMixin, BaseCrossValidator):
    
    # 初始化方法，接受一个可选的registry参数
    def __init__(self, registry=None):
        self.registry = registry

    # 实现split方法，用于生成训练集和测试集的索引
    def split(self, X, y=None, groups="default", metadata="default"):
        # 如果registry不为None，则将当前对象添加到registry中
        if self.registry is not None:
            self.registry.append(self)

        # 调用record_metadata_not_default函数记录元数据，传入groups和metadata参数
        record_metadata_not_default(self, groups=groups, metadata=metadata)

        # 计算分割索引为数据长度的一半
        split_index = len(X) // 2
        # 创建训练集索引列表，从0到split_index
        train_indices = list(range(0, split_index))
        # 创建测试集索引列表，从split_index到数据长度
        test_indices = list(range(split_index, len(X)))
        # 使用yield关键字生成测试集索引和训练集索引的迭代器
        yield test_indices, train_indices
        yield train_indices, test_indices

    # 实现get_n_splits方法，返回数据集被分割的次数，这里是2
    def get_n_splits(self, X=None, y=None, groups=None, metadata=None):
        return 2

    # 实现_iter_test_indices方法，生成测试集的索引迭代器
    def _iter_test_indices(self, X=None, y=None, groups=None):
        # 计算分割索引为数据长度的一半
        split_index = len(X) // 2
        # 创建训练集索引列表，从0到split_index
        train_indices = list(range(0, split_index))
        # 创建测试集索引列表，从split_index到数据长度
        test_indices = list(range(split_index, len(X)))
        # 使用yield关键字生成测试集索引的迭代器
        yield test_indices
        # 使用yield关键字生成训练集索引的迭代器
        yield train_indices


# 定义一个元回归器类，同时继承了MetaEstimatorMixin、RegressorMixin和BaseEstimator
class MetaRegressor(MetaEstimatorMixin, RegressorMixin, BaseEstimator):
    """A meta-regressor which is only a router."""

    # 初始化方法，接受一个estimator参数作为输入
    def __init__(self, estimator):
        self.estimator = estimator

    # 实现fit方法，用于拟合模型
    def fit(self, X, y, **fit_params):
        # 调用process_routing函数处理路由，传入self、"fit"和fit_params
        params = process_routing(self, "fit", **fit_params)
        # 使用clone函数复制estimator对象，并调用其fit方法拟合数据
        self.estimator_ = clone(self.estimator).fit(X, y, **params.estimator.fit)

    # 实现get_metadata_routing方法，返回一个元数据路由对象
    def get_metadata_routing(self):
        # 创建一个MetadataRouter对象，设置其所有者为当前类名
        router = MetadataRouter(owner=self.__class__.__name__).add(
            # 添加estimator参数和方法映射到路由器中
            estimator=self.estimator,
            method_mapping=MethodMapping().add(caller="fit", callee="fit"),
        )
        return router


# 定义一个带权重的元回归器类，同时继承了MetaEstimatorMixin、RegressorMixin和BaseEstimator
class WeightedMetaRegressor(MetaEstimatorMixin, RegressorMixin, BaseEstimator):
    """A meta-regressor which is also a consumer."""

    # 初始化方法，接受一个estimator和一个可选的registry参数
    def __init__(self, estimator, registry=None):
        self.estimator = estimator
        self.registry = registry

    # 实现fit方法，用于拟合带权重的数据
    def fit(self, X, y, sample_weight=None, **fit_params):
        # 如果registry不为None，则将当前对象添加到registry中
        if self.registry is not None:
            self.registry.append(self)

        # 调用record_metadata函数记录元数据，传入self和sample_weight参数
        record_metadata(self, sample_weight=sample_weight)
        # 调用process_routing函数处理路由，传入self、"fit"、sample_weight和fit_params
        params = process_routing(self, "fit", sample_weight=sample_weight, **fit_params)
        # 使用clone函数复制estimator对象，并调用其fit方法拟合数据
        self.estimator_ = clone(self.estimator).fit(X, y, **params.estimator.fit)
        # 返回self对象
        return self

    # 实现predict方法，用于预测数据
    def predict(self, X, **predict_params):
        # 调用process_routing函数处理路由，传入self、"predict"和predict_params
        params = process_routing(self, "predict", **predict_params)
        # 调用estimator_对象的predict方法进行预测，并返回结果
        return self.estimator_.predict(X, **params.estimator.predict)

    # 实现get_metadata_routing方法，返回一个元数据路由对象
    def get_metadata_routing(self):
        # 创建一个MetadataRouter对象，设置其所有者为当前类名
        router = (
            MetadataRouter(owner=self.__class__.__name__)
            # 添加self请求到路由器中
            .add_self_request(self)
            # 添加estimator参数和方法映射到路由器中
            .add(
                estimator=self.estimator,
                method_mapping=MethodMapping()
                .add(caller="fit", callee="fit")
                .add(caller="predict", callee="predict"),
            )
        )
        return router


# 定义一个带权重的元分类器类，同时继承了MetaEstimatorMixin、ClassifierMixin和BaseEstimator
class WeightedMetaClassifier(MetaEstimatorMixin, ClassifierMixin, BaseEstimator):
    """A meta-estimator which also consumes sample_weight itself in ``fit``."""

    # 初始化方法，接受一个estimator和一个可选的registry参数
    def __init__(self, estimator, registry=None):
        self.estimator = estimator
        self.registry = registry
    # 初始化方法，接受一个估计器和一个注册表作为参数，并将它们存储在实例变量中
    def __init__(self, estimator, registry=None):
        self.estimator = estimator
        self.registry = registry

    # 拟合方法，用于训练模型
    def fit(self, X, y, sample_weight=None, **kwargs):
        # 如果注册表不为空，则将当前对象添加到注册表中
        if self.registry is not None:
            self.registry.append(self)

        # 记录元数据，包括样本权重
        record_metadata(self, sample_weight=sample_weight)

        # 处理路由，确定估计器的参数
        params = process_routing(self, "fit", sample_weight=sample_weight, **kwargs)

        # 克隆估计器对象，并使用处理后的参数拟合模型
        self.estimator_ = clone(self.estimator).fit(X, y, **params.estimator.fit)

        # 返回当前对象的引用
        return self

    # 获取元数据路由信息的方法
    def get_metadata_routing(self):
        # 创建一个元数据路由对象，设置其所有者为当前类名
        router = (
            MetadataRouter(owner=self.__class__.__name__)
            .add_self_request(self)  # 添加当前对象的请求
            .add(
                estimator=self.estimator,  # 添加估计器对象
                method_mapping=MethodMapping().add(caller="fit", callee="fit"),  # 指定调用者和被调用者的方法映射
            )
        )
        # 返回创建的路由对象
        return router
class MetaTransformer(MetaEstimatorMixin, TransformerMixin, BaseEstimator):
    """A simple meta-transformer."""

    def __init__(self, transformer):
        # 初始化方法，接收一个转换器对象并存储在实例变量中
        self.transformer = transformer

    def fit(self, X, y=None, **fit_params):
        # 处理路由，根据给定参数处理fit方法的路由信息
        params = process_routing(self, "fit", **fit_params)
        # 克隆转换器对象并调用其fit方法进行拟合，存储在实例变量中
        self.transformer_ = clone(self.transformer).fit(X, y, **params.transformer.fit)
        # 返回自身实例
        return self

    def transform(self, X, y=None, **transform_params):
        # 处理路由，根据给定参数处理transform方法的路由信息
        params = process_routing(self, "transform", **transform_params)
        # 调用已拟合的转换器对象的transform方法进行数据转换
        return self.transformer_.transform(X, **params.transformer.transform)

    def get_metadata_routing(self):
        # 返回元数据路由对象，配置了转换器和方法映射的信息
        return MetadataRouter(owner=self.__class__.__name__).add(
            transformer=self.transformer,
            method_mapping=MethodMapping()
            .add(caller="fit", callee="fit")
            .add(caller="transform", callee="transform"),
        )
```