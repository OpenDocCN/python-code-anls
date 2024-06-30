# `D:\src\scipysrc\scikit-learn\sklearn\tests\test_metadata_routing.py`

```
"""
Metadata Routing Utility Tests
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入正则表达式模块
import re

# 导入numpy库并命名为np
import numpy as np

# 导入pytest测试框架
import pytest

# 导入sklearn库中的配置上下文管理器
from sklearn import config_context

# 导入sklearn库中的基本估计器和克隆函数
from sklearn.base import (
    BaseEstimator,
    clone,
)

# 导入sklearn库中的异常处理模块
from sklearn.exceptions import UnsetMetadataPassedError

# 导入sklearn库中的线性回归模型
from sklearn.linear_model import LinearRegression

# 导入sklearn库中的Pipeline类
from sklearn.pipeline import Pipeline

# 导入sklearn库中的元数据路由测试工具
from sklearn.tests.metadata_routing_common import (
    ConsumingClassifier,
    ConsumingRegressor,
    ConsumingTransformer,
    MetaRegressor,
    MetaTransformer,
    NonConsumingClassifier,
    WeightedMetaClassifier,
    WeightedMetaRegressor,
    _Registry,
    assert_request_equal,
    assert_request_is_empty,
    check_recorded_metadata,
)

# 导入sklearn库中的元数据路由相关工具
from sklearn.utils import metadata_routing
from sklearn.utils._metadata_requests import (
    COMPOSITE_METHODS,
    METHODS,
    SIMPLE_METHODS,
    MethodMetadataRequest,
    MethodPair,
    _MetadataRequester,
    request_is_alias,
    request_is_valid,
)

# 导入sklearn库中的元数据请求和路由相关工具
from sklearn.utils.metadata_routing import (
    MetadataRequest,
    MetadataRouter,
    MethodMapping,
    _RoutingNotSupportedMixin,
    get_routing_for_object,
    process_routing,
)

# 导入sklearn库中的验证工具
from sklearn.utils.validation import check_is_fitted

# 设置随机数生成器
rng = np.random.RandomState(42)

# 定义样本数量和特征数量
N, M = 100, 4

# 生成随机样本数据
X = rng.rand(N, M)

# 生成随机标签数据
y = rng.randint(0, 2, size=N)

# 生成随机分组信息
my_groups = rng.randint(0, 10, size=N)

# 生成随机权重数据
my_weights = rng.rand(N)

# 生成另一组随机权重数据
my_other_weights = rng.rand(N)


@pytest.fixture(autouse=True)
def enable_slep006():
    """Enable SLEP006 for all tests."""
    # 使用配置上下文管理器开启元数据路由支持，并在整个测试过程中有效
    with config_context(enable_metadata_routing=True):
        # 使用yield语句将控制权交给测试函数
        yield


class SimplePipeline(BaseEstimator):
    """A very simple pipeline, assuming the last step is always a predictor.

    Parameters
    ----------
    steps : iterable of objects
        An iterable of transformers with the last step being a predictor.
    """

    def __init__(self, steps):
        # 初始化管道对象，接收一个步骤列表
        self.steps = steps

    def fit(self, X, y, **fit_params):
        # 处理fit方法的元数据请求参数
        params = process_routing(self, "fit", **fit_params)
        
        # 初始化变量，保存处理后的步骤列表
        self.steps_ = []
        
        # 初始数据转换
        X_transformed = X
        
        # 迭代所有步骤，除了最后一个预测器步骤
        for i, step in enumerate(self.steps[:-1]):
            # 克隆并拟合当前步骤的变换器
            transformer = clone(step).fit(
                X_transformed, y, **params.get(f"step_{i}").fit
            )
            # 将拟合后的变换器保存到步骤列表中
            self.steps_.append(transformer)
            # 应用当前变换器对数据进行转换
            X_transformed = transformer.transform(
                X_transformed, **params.get(f"step_{i}").transform
            )

        # 克隆并拟合最后一个预测器步骤
        self.steps_.append(
            clone(self.steps[-1]).fit(X_transformed, y, **params.predictor.fit)
        )
        # 返回拟合后的管道对象
        return self

    def predict(self, X, **predict_params):
        # 检查管道对象是否已拟合
        check_is_fitted(self)
        # 初始数据转换
        X_transformed = X
        
        # 处理predict方法的元数据请求参数
        params = process_routing(self, "predict", **predict_params)
        
        # 应用所有变换器对数据进行转换
        for i, step in enumerate(self.steps_[:-1]):
            X_transformed = step.transform(X, **params.get(f"step_{i}").transform)

        # 使用最后一个预测器步骤进行预测
        return self.steps_[-1].predict(X_transformed, **params.predictor.predict)
    # 获取元数据路由器对象的方法
    def get_metadata_routing(self):
        # 创建一个元数据路由器对象，设置其所有者为当前类的名称
        router = MetadataRouter(owner=self.__class__.__name__)
        
        # 遍历处理管道中除最后一个步骤外的所有步骤
        for i, step in enumerate(self.steps[:-1]):
            # 为每个步骤创建映射关系，命名为 step_i，使用 MethodMapping 对象管理方法映射
            router.add(
                **{f"step_{i}": step},
                method_mapping=MethodMapping()
                .add(caller="fit", callee="fit")  # 添加 fit 方法的调用映射
                .add(caller="fit", callee="transform")  # 添加 fit 方法到 transform 方法的映射
                .add(caller="predict", callee="transform"),  # 添加 predict 方法到 transform 方法的映射
            )
        
        # 处理管道中的最后一个步骤，添加预测器，并设置其方法映射
        router.add(
            predictor=self.steps[-1],
            method_mapping=MethodMapping()
            .add(caller="fit", callee="fit")  # 添加 fit 方法的调用映射
            .add(caller="predict", callee="predict"),  # 添加 predict 方法的调用映射
        )
        
        # 返回配置好的元数据路由器对象
        return router
# 测试断言请求是否为空的函数
def test_assert_request_is_empty():
    # 创建一个 MetadataRequest 对象，指定所有者为 "test"
    requests = MetadataRequest(owner="test")
    # 断言该请求对象为空
    assert_request_is_empty(requests)

    # 向 fit 方法添加一个请求参数 "foo"，别名为 None
    # 这应该仍然有效，因为 None 是默认值
    assert_request_is_empty(requests)

    # 向 fit 方法添加一个请求参数 "bar"，别名为 "value"
    # 应该会引发 AssertionError，因为请求对象不再为空
    with pytest.raises(AssertionError):
        assert_request_is_empty(requests)

    # 可以排除一个方法（fit）
    assert_request_is_empty(requests, exclude="fit")

    # 向 score 方法添加一个请求参数 "carrot"，别名为 True
    # 应该会引发 AssertionError，因为排除 'fit' 方法不足以避免异常
    with pytest.raises(AssertionError):
        assert_request_is_empty(requests, exclude="fit")

    # 排除 fit 和 score 方法将避免异常
    assert_request_is_empty(requests, exclude=["fit", "score"])

    # 测试路由是否为空
    assert_request_is_empty(
        MetadataRouter(owner="test")
        .add_self_request(WeightedMetaRegressor(estimator=None))
        .add(
            estimator=ConsumingRegressor(),
            method_mapping=MethodMapping().add(caller="fit", callee="fit"),
        )
    )


# 参数化测试：检查估计器是否在 fit 后将自身放入注册表中
@pytest.mark.parametrize(
    "estimator",
    [
        ConsumingClassifier(registry=_Registry()),
        ConsumingRegressor(registry=_Registry()),
        ConsumingTransformer(registry=_Registry()),
        WeightedMetaClassifier(estimator=ConsumingClassifier(), registry=_Registry()),
        WeightedMetaRegressor(estimator=ConsumingRegressor(), registry=_Registry()),
    ],
)
def test_estimator_puts_self_in_registry(estimator):
    """Check that an estimator puts itself in the registry upon fit."""
    # 对估计器进行 fit 操作
    estimator.fit(X, y)
    # 断言估计器已经在注册表中
    assert estimator in estimator.registry


# 参数化测试：检查请求类型是否为别名
@pytest.mark.parametrize(
    "val, res",
    [
        (False, False),
        (True, False),
        (None, False),
        ("$UNUSED$", False),
        ("$WARN$", False),
        ("invalid-input", False),
        ("valid_arg", True),
    ],
)
def test_request_type_is_alias(val, res):
    # 测试 request_is_alias 函数
    assert request_is_alias(val) == res


# 参数化测试：检查请求类型是否有效
@pytest.mark.parametrize(
    "val, res",
    [
        (False, True),
        (True, True),
        (None, True),
        ("$UNUSED$", True),
        ("$WARN$", True),
        ("invalid-input", False),
        ("alias_arg", False),
    ],
)
def test_request_type_is_valid(val, res):
    # 测试 request_is_valid 函数
    assert request_is_valid(val) == res


# 测试默认请求
def test_default_requests():
    # 定义一个 OddEstimator 类，继承自 BaseEstimator
    class OddEstimator(BaseEstimator):
        # 设置不同的默认请求
        __metadata_request__fit = {
            "sample_weight": True
        }  # type: ignore

    # 获取 OddEstimator 对象的路由信息
    odd_request = get_routing_for_object(OddEstimator())
    # 断言 fit 方法的请求为 {"sample_weight": True}
    assert odd_request.fit.requests == {"sample_weight": True}

    # 检查其他测试估计器
    # 断言 NonConsumingClassifier 的 fit 方法请求为空
    assert not len(get_routing_for_object(NonConsumingClassifier()).fit.requests)
    # 断言 NonConsumingClassifier 的元数据路由为空
    assert_request_is_empty(NonConsumingClassifier().get_metadata_routing())
    # 使用自定义的 ConsumingTransformer 类创建 trs_request，获取其请求对象
    trs_request = get_routing_for_object(ConsumingTransformer())
    # 断言 trs_request 的 fit 请求包含指定的键和值
    assert trs_request.fit.requests == {
        "sample_weight": None,
        "metadata": None,
    }
    # 断言 trs_request 的 transform 请求包含指定的键和值
    assert trs_request.transform.requests == {"metadata": None, "sample_weight": None}
    # 断言 trs_request 是空的，即不包含任何请求
    assert_request_is_empty(trs_request)

    # 使用自定义的 ConsumingClassifier 类创建 est_request，获取其请求对象
    est_request = get_routing_for_object(ConsumingClassifier())
    # 断言 est_request 的 fit 请求包含指定的键和值
    assert est_request.fit.requests == {
        "sample_weight": None,
        "metadata": None,
    }
    # 断言 est_request 是空的，即不包含任何请求
    assert_request_is_empty(est_request)
# 定义一个测试函数，用于测试默认请求是否正确覆盖，不受类名ASCII顺序影响。
# 这是一个非回归测试，用于检查 https://github.com/scikit-learn/scikit-learn/issues/28430 的问题是否修复。
def test_default_request_override():

    # 定义一个名为 Base 的类，继承自 BaseEstimator
    class Base(BaseEstimator):
        # 定义一个特殊的类属性 __metadata_request__split，设定为 {"groups": True}
        __metadata_request__split = {"groups": True}

    # 定义一个名为 class_1 的类，继承自 Base
    class class_1(Base):
        # 覆盖父类的 __metadata_request__split 属性，设定为 {"groups": "sample_domain"}
        __metadata_request__split = {"groups": "sample_domain"}

    # 定义一个名为 Class_1 的类，继承自 Base
    class Class_1(Base):
        # 覆盖父类的 __metadata_request__split 属性，设定为 {"groups": "sample_domain"}
        __metadata_request__split = {"groups": "sample_domain"}

    # 断言 class_1()._get_metadata_request() 的返回结果是否等于 {"split": {"groups": "sample_domain"}}
    assert_request_equal(
        class_1()._get_metadata_request(), {"split": {"groups": "sample_domain"}}
    )

    # 断言 Class_1()._get_metadata_request() 的返回结果是否等于 {"split": {"groups": "sample_domain"}}
    assert_request_equal(
        Class_1()._get_metadata_request(), {"split": {"groups": "sample_domain"}}
    )


# 定义一个测试函数，用于测试处理路由中的无效方法
def test_process_routing_invalid_method():
    # 使用 pytest.raises 断言捕获 TypeError 异常，匹配错误信息 "Can only route and process input"
    with pytest.raises(TypeError, match="Can only route and process input"):
        # 调用 process_routing 函数，传入一个 ConsumingClassifier 实例、"invalid_method" 字符串以及 groups=my_groups 参数
        process_routing(ConsumingClassifier(), "invalid_method", groups=my_groups)


# 定义一个测试函数，用于测试处理路由中的无效对象
def test_process_routing_invalid_object():
    # 定义一个名为 InvalidObject 的简单类
    class InvalidObject:
        pass

    # 使用 pytest.raises 断言捕获 AttributeError 异常，匹配错误信息 "either implement the routing method"
    with pytest.raises(AttributeError, match="either implement the routing method"):
        # 调用 process_routing 函数，传入一个 InvalidObject 实例、"fit" 字符串以及 groups=my_groups 参数
        process_routing(InvalidObject(), "fit", groups=my_groups)


# 使用 pytest.mark.parametrize 注解，定义一个参数化测试函数，测试处理路由中空参数获取默认行为
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("default", [None, "default", []])
def test_process_routing_empty_params_get_with_default(method, default):
    # 定义一个空字典 empty_params
    empty_params = {}
    # 调用 process_routing 函数，传入一个 ConsumingClassifier 实例、"fit" 字符串以及空字典 empty_params
    routed_params = process_routing(ConsumingClassifier(), "fit", **empty_params)

    # 断言 routed_params[method] 的类型为 dict
    params_for_method = routed_params[method]
    assert isinstance(params_for_method, dict)
    # 断言 params_for_method 的键集合与 METHODS 的集合相等
    assert set(params_for_method.keys()) == set(METHODS)

    # 断言使用 process_routing 的 get 方法获取 method 的默认参数，与 params_for_method 相等
    default_params_for_method = routed_params.get(method, default=default)
    assert default_params_for_method == params_for_method


# 定义一个测试函数，测试简单的元数据路由
def test_simple_metadata_routing():
    # 测试元数据是否被正确路由

    # 实例化一个 WeightedMetaClassifier 对象 clf，使用 NonConsumingClassifier 作为估计器
    clf = WeightedMetaClassifier(estimator=NonConsumingClassifier())
    # 调用 clf.fit(X, y) 方法

    # 实例化一个 WeightedMetaClassifier 对象 clf，使用 NonConsumingClassifier 作为估计器，传入 sample_weight=my_weights 参数
    clf = WeightedMetaClassifier(estimator=NonConsumingClassifier())
    # 调用 clf.fit(X, y, sample_weight=my_weights) 方法

    # 实例化一个 WeightedMetaClassifier 对象 clf，使用 ConsumingClassifier 作为估计器
    clf = WeightedMetaClassifier(estimator=ConsumingClassifier())
    # 使用 pytest.raises 断言捕获 ValueError 异常，匹配预期错误信息
    err_message = (
        "[sample_weight] are passed but are not explicitly set as requested or"
        " not requested for ConsumingClassifier.fit"
    )
    with pytest.raises(ValueError, match=re.escape(err_message)):
        # 调用 clf.fit(X, y, sample_weight=my_weights) 方法
        clf.fit(X, y, sample_weight=my_weights)

    # 明确指定估计器不需要元数据，预期错误消失
    # 创建一个 WeightedMetaClassifier 实例，使用一个定制的 ConsumingClassifier 作为其基础模型，
    # 并设置其不需要 sample_weight 的情况。
    clf = WeightedMetaClassifier(
        estimator=ConsumingClassifier().set_fit_request(sample_weight=False)
    )
    # 调用 fit 方法拟合模型，传入特征数据 X、目标变量 y 和样本权重 my_weights，
    # 这里不会引发异常，因为 WeightedMetaClassifier 本身会消耗 sample_weight。
    clf.fit(X, y, sample_weight=my_weights)
    # 检查记录的元数据，验证 fit 方法是否正确记录了元数据，
    # 包括方法名为 "fit"，父级为 "fit"。
    check_recorded_metadata(clf.estimator_, method="fit", parent="fit")

    # 创建另一个 WeightedMetaClassifier 实例，使用 ConsumingClassifier 作为其基础模型，
    # 并设置其需要 sample_weight 的情况。
    clf = WeightedMetaClassifier(
        estimator=ConsumingClassifier().set_fit_request(sample_weight=True)
    )
    # 调用 fit 方法拟合模型，传入特征数据 X、目标变量 y 和样本权重 my_weights，
    # 这里不会引发异常，因为 WeightedMetaClassifier 本身会消耗 sample_weight。
    clf.fit(X, y, sample_weight=my_weights)
    # 检查记录的元数据，验证 fit 方法是否正确记录了元数据，
    # 包括方法名为 "fit"，父级为 "fit"，并且传递了 sample_weight=my_weights。
    check_recorded_metadata(
        clf.estimator_, method="fit", parent="fit", sample_weight=my_weights
    )

    # 创建另一个 WeightedMetaClassifier 实例，使用 ConsumingClassifier 作为其基础模型，
    # 并设置其使用 sample_weight 的别名 "alternative_weight"。
    clf = WeightedMetaClassifier(
        estimator=ConsumingClassifier().set_fit_request(
            sample_weight="alternative_weight"
        )
    )
    # 调用 fit 方法拟合模型，传入特征数据 X、目标变量 y 和样本权重 my_weights，
    # 这里不会引发异常，因为 WeightedMetaClassifier 本身会消耗 sample_weight。
    # 根据别名传递样本权重，实际效果与传递 sample_weight=my_weights 相同。
    clf.fit(X, y, alternative_weight=my_weights)
    # 检查记录的元数据，验证 fit 方法是否正确记录了元数据，
    # 包括方法名为 "fit"，父级为 "fit"，并且传递了 sample_weight=my_weights。
    check_recorded_metadata(
        clf.estimator_, method="fit", parent="fit", sample_weight=my_weights
    )
def test_nested_routing():
    # 检查元数据在嵌套路由情况下的路由情况。
    pipeline = SimplePipeline(
        [
            MetaTransformer(
                transformer=ConsumingTransformer()
                .set_fit_request(metadata=True, sample_weight=False)
                .set_transform_request(sample_weight=True, metadata=False)
            ),
            WeightedMetaRegressor(
                estimator=ConsumingRegressor()
                .set_fit_request(sample_weight="inner_weights", metadata=False)
                .set_predict_request(sample_weight=False)
            ).set_fit_request(sample_weight="outer_weights"),
        ]
    )
    w1, w2, w3 = [1], [2], [3]
    # 对流水线进行拟合操作，传入数据集(X, y)，元数据(my_groups)，样本权重(w1)，外部权重(outer_weights)，内部权重(inner_weights)
    pipeline.fit(
        X, y, metadata=my_groups, sample_weight=w1, outer_weights=w2, inner_weights=w3
    )
    # 检查记录的元数据，应用在流水线第一个步骤的转换器上，方法为"fit"，父级为"fit"，传入元数据my_groups
    check_recorded_metadata(
        pipeline.steps_[0].transformer_,
        method="fit",
        parent="fit",
        metadata=my_groups,
    )
    # 检查记录的元数据，应用在流水线第一个步骤的转换器上，方法为"transform"，父级为"fit"，传入样本权重w1
    check_recorded_metadata(
        pipeline.steps_[0].transformer_,
        method="transform",
        parent="fit",
        sample_weight=w1,
    )
    # 检查记录的元数据，应用在流水线第二个步骤上，方法为"fit"，父级为"fit"，传入样本权重w2
    check_recorded_metadata(
        pipeline.steps_[1], method="fit", parent="fit", sample_weight=w2
    )
    # 检查记录的元数据，应用在流水线第二个步骤的估计器上，方法为"fit"，父级为"fit"，传入样本权重w3
    check_recorded_metadata(
        pipeline.steps_[1].estimator_, method="fit", parent="fit", sample_weight=w3
    )

    # 对流水线进行预测操作，传入数据集X，样本权重w3
    pipeline.predict(X, sample_weight=w3)
    # 检查记录的元数据，应用在流水线第一个步骤的转换器上，方法为"transform"，父级为"fit"，传入样本权重w3
    check_recorded_metadata(
        pipeline.steps_[0].transformer_,
        method="transform",
        parent="fit",
        sample_weight=w3,
    )


def test_nested_routing_conflict():
    # 检查是否在键冲突时引发错误
    pipeline = SimplePipeline(
        [
            MetaTransformer(
                transformer=ConsumingTransformer()
                .set_fit_request(metadata=True, sample_weight=False)
                .set_transform_request(sample_weight=True)
            ),
            WeightedMetaRegressor(
                estimator=ConsumingRegressor().set_fit_request(sample_weight=True)
            ).set_fit_request(sample_weight="outer_weights"),
        ]
    )
    w1, w2 = [1], [2]
    # 使用pytest检查是否引发ValueError，并匹配特定错误消息
    with pytest.raises(
        ValueError,
        match=(
            re.escape(
                "In WeightedMetaRegressor, there is a conflict on sample_weight between"
                " what is requested for this estimator and what is requested by its"
                " children. You can resolve this conflict by using an alias for the"
                " child estimator(s) requested metadata."
            )
        ),
    ):
        pipeline.fit(X, y, metadata=my_groups, sample_weight=w1, outer_weights=w2)


def test_invalid_metadata():
    # 检查传递错误的元数据是否会引发错误
    trs = MetaTransformer(
        transformer=ConsumingTransformer().set_transform_request(sample_weight=True)
    )
    # 使用pytest检查是否引发TypeError，并匹配特定错误消息
    with pytest.raises(
        TypeError,
        match=(re.escape("transform got unexpected argument(s) {'other_param'}")),
        # 这里注释未完整，仅为示范，应该继续完成这行注释
    ):
        # 使用 `MetaTransformer` 初始化一个实例 `trs`，其中包含一个 `ConsumingTransformer` 的实例作为 `transformer` 参数，
        # 并设置 `sample_weight=False` 作为不请求的转换参数。
        trs.fit(X, y).transform(X, other_param=my_weights)

    # 当传递一个元数据参数给任何估算器时，该参数未被请求，应当引发异常。
    trs = MetaTransformer(
        transformer=ConsumingTransformer().set_transform_request(sample_weight=False)
    )
    with pytest.raises(
        TypeError,
        match=(re.escape("transform got unexpected argument(s) {'sample_weight'}")),
    ):
        # 使用 `trs` 对象调用 `fit` 方法，并尝试使用 `transform` 方法传递 `sample_weight=my_weights` 参数，
        # 应当引发 `TypeError` 异常，并匹配给定的错误消息字符串。
        trs.fit(X, y).transform(X, sample_weight=my_weights)
def test_get_metadata_routing():
    # 定义一个测试类 TestDefaultsBadMethodName，继承自 _MetadataRequester
    class TestDefaultsBadMethodName(_MetadataRequester):
        # 定义属性 __metadata_request__fit，用于 fit 方法的默认元数据请求
        __metadata_request__fit = {
            "sample_weight": None,
            "my_param": None,
        }
        # 定义属性 __metadata_request__score，用于 score 方法的默认元数据请求
        __metadata_request__score = {
            "sample_weight": None,
            "my_param": True,
            "my_other_param": None,
        }
        # 定义属性 __metadata_request__other_method，预期会因为方法名不正确而引发错误
        # 实际上是不正确的方法名，会导致 AttributeError
        __metadata_request__other_method = {"my_param": True}

    # 定义一个测试类 TestDefaults，继承自 _MetadataRequester
    class TestDefaults(_MetadataRequester):
        # 定义属性 __metadata_request__fit，用于 fit 方法的默认元数据请求
        __metadata_request__fit = {
            "sample_weight": None,
            "my_other_param": None,
        }
        # 定义属性 __metadata_request__score，用于 score 方法的默认元数据请求
        __metadata_request__score = {
            "sample_weight": None,
            "my_param": True,
            "my_other_param": None,
        }
        # 定义属性 __metadata_request__predict，用于 predict 方法的默认元数据请求
        __metadata_request__predict = {"my_param": True}

    # 使用 pytest 的 assertRaises 来检测 AttributeError 异常是否被正确引发
    with pytest.raises(
        AttributeError, match="'MetadataRequest' object has no attribute 'other_method'"
    ):
        # 实例化 TestDefaultsBadMethodName 类，并调用其 get_metadata_routing 方法
        TestDefaultsBadMethodName().get_metadata_routing()

    # 定义预期的元数据路由字典 expected，包含了不同方法对应的元数据请求
    expected = {
        "score": {
            "my_param": True,
            "my_other_param": None,
            "sample_weight": None,
        },
        "fit": {
            "my_other_param": None,
            "sample_weight": None,
        },
        "predict": {"my_param": True},
    }
    # 使用 assert_request_equal 函数检查 TestDefaults 类的 get_metadata_routing 方法返回值是否与 expected 相等
    assert_request_equal(TestDefaults().get_metadata_routing(), expected)

    # 实例化 TestDefaults 类，并使用 set_score_request 方法设置特定参数后再调用 get_metadata_routing 方法
    est = TestDefaults().set_score_request(my_param="other_param")
    # 更新预期的元数据路由字典 expected，以反映设置参数后的预期值
    expected = {
        "score": {
            "my_param": "other_param",
            "my_other_param": None,
            "sample_weight": None,
        },
        "fit": {
            "my_other_param": None,
            "sample_weight": None,
        },
        "predict": {"my_param": True},
    }
    # 使用 assert_request_equal 函数检查设置参数后的 get_metadata_routing 方法返回值是否与 updated_expected 相等
    assert_request_equal(est.get_metadata_routing(), expected)

    # 实例化 TestDefaults 类，并使用 set_fit_request 方法设置特定参数后再调用 get_metadata_routing 方法
    est = TestDefaults().set_fit_request(sample_weight=True)
    # 更新预期的元数据路由字典 expected，以反映设置参数后的预期值
    expected = {
        "score": {
            "my_param": True,
            "my_other_param": None,
            "sample_weight": None,
        },
        "fit": {
            "my_other_param": None,
            "sample_weight": True,
        },
        "predict": {"my_param": True},
    }
    # 使用 assert_request_equal 函数检查设置参数后的 get_metadata_routing 方法返回值是否与 updated_expected 相等
    assert_request_equal(est.get_metadata_routing(), expected)


def test_setting_default_requests():
    # 定义一个空的测试用例字典 test_cases
    test_cases = dict()

    # 定义一个显式请求的测试类 ExplicitRequest，继承自 BaseEstimator
    class ExplicitRequest(BaseEstimator):
        # 定义属性 __metadata_request__fit，用于 fit 方法的默认元数据请求
        # 要求包含 'prop'，但 fit 方法本身不接受 'prop'
        __metadata_request__fit = {"prop": None}

        # 定义 fit 方法，接受 X, y 和其他关键字参数 kwargs
        def fit(self, X, y, **kwargs):
            return self

    # 将 ExplicitRequest 类及其预期的元数据请求添加到 test_cases 字典中
    test_cases[ExplicitRequest] = {"prop": None}

    # 定义一个显式请求覆盖的测试类 ExplicitRequestOverwrite，继承自 BaseEstimator
    class ExplicitRequestOverwrite(BaseEstimator):
        # 定义属性 __metadata_request__fit，用于 fit 方法的默认元数据请求
        # 要求包含 'prop'，并将默认请求值从 None 改为 True
        __metadata_request__fit = {"prop": True}

        # 定义 fit 方法，接受 X, y 和 prop 参数，以及其他关键字参数 kwargs
        def fit(self, X, y, prop=None, **kwargs):
            return self
    # 在测试用例字典中添加一个显式请求覆盖的条目，请求的属性为True
    test_cases[ExplicitRequestOverwrite] = {"prop": True}

    class ImplicitRequest(BaseEstimator):
        # BaseEstimator 的子类，定义了 `fit` 方法，该方法请求 `prop` 属性，
        # 使用默认值 None
        def fit(self, X, y, prop=None, **kwargs):
            return self

    # 在测试用例字典中添加一个隐式请求的条目，请求的属性为None
    test_cases[ImplicitRequest] = {"prop": None}

    class ImplicitRequestRemoval(BaseEstimator):
        # 定义了一个类级别的元数据 `__metadata_request__fit`，
        # 指示在 `fit` 方法中请求 `prop` 属性，但请求应标记为未使用
        __metadata_request__fit = {"prop": metadata_routing.UNUSED}

        # 定义了 `fit` 方法，接受 `X`, `y`, `prop` 和其他关键字参数
        def fit(self, X, y, prop=None, **kwargs):
            return self

    # 在测试用例字典中添加一个请求移除的条目，没有任何请求的属性
    test_cases[ImplicitRequestRemoval] = {}

    # 对于测试用例字典中的每个类和其请求，
    # 确保调用 `get_routing_for_object(Klass()).fit.requests` 返回正确的请求
    for Klass, requests in test_cases.items():
        assert get_routing_for_object(Klass()).fit.requests == requests
        # 确保调用 `assert_request_is_empty` 方法检查元数据路由，排除 `fit` 方法
        assert_request_is_empty(Klass().get_metadata_routing(), exclude="fit")
        # 调用每个类的 `fit` 方法以提高覆盖率
        Klass().fit(None, None)  # for coverage
# 测试在使用 UNUSED 删除不存在的元数据时是否会引发异常
def test_removing_non_existing_param_raises():

    class InvalidRequestRemoval(BaseEstimator):
        # `fit` 方法（在这个类或其父类中）要求 `prop`，但我们不希望它被请求
        __metadata_request__fit = {"prop": metadata_routing.UNUSED}

        # 定义 fit 方法，接受 X、y 参数以及其他关键字参数
        def fit(self, X, y, **kwargs):
            return self

    # 使用 pytest 检查是否会引发 ValueError 异常，异常信息中包含 "Trying to remove parameter"
    with pytest.raises(ValueError, match="Trying to remove parameter"):
        InvalidRequestRemoval().get_metadata_routing()


# 测试 MethodMetadataRequest 类的行为
def test_method_metadata_request():
    # 创建 MethodMetadataRequest 实例，所有请求均与拥有者 "test" 和方法 "fit" 相关
    mmr = MethodMetadataRequest(owner="test", method="fit")

    # 使用 pytest 检查是否会引发 ValueError 异常，异常信息中包含 "The alias you're setting for"
    with pytest.raises(ValueError, match="The alias you're setting for"):
        mmr.add_request(param="foo", alias=1.4)

    # 添加请求，参数为 "foo"，别名为 None
    mmr.add_request(param="foo", alias=None)
    assert mmr.requests == {"foo": None}

    # 添加请求，参数为 "foo"，别名为 False
    mmr.add_request(param="foo", alias=False)
    assert mmr.requests == {"foo": False}

    # 添加请求，参数为 "foo"，别名为 True
    mmr.add_request(param="foo", alias=True)
    assert mmr.requests == {"foo": True}

    # 添加请求，参数为 "foo"，别名为 "foo"
    mmr.add_request(param="foo", alias="foo")
    assert mmr.requests == {"foo": True}

    # 添加请求，参数为 "foo"，别名为 "bar"
    mmr.add_request(param="foo", alias="bar")
    assert mmr.requests == {"foo": "bar"}

    # 断言获取参数名时，返回集合 {"foo"}，不返回别名
    assert mmr._get_param_names(return_alias=False) == {"foo"}

    # 断言获取参数名时，返回集合 {"bar"}，返回别名
    assert mmr._get_param_names(return_alias=True) == {"bar"}


# 测试获取对象的路由
def test_get_routing_for_object():
    # 定义一个消费者类 Consumer，其 `fit` 方法请求 `prop` 元数据
    class Consumer(BaseEstimator):
        __metadata_request__fit = {"prop": None}

    # 断言获取空请求对象时，返回空集合
    assert_request_is_empty(get_routing_for_object(None))
    assert_request_is_empty(get_routing_for_object(object()))

    # 创建 MetadataRequest 实例 mr，拥有者为 "test"
    mr = MetadataRequest(owner="test")

    # 向 mr 实例的 `fit` 方法添加请求，参数为 "foo"，别名为 "bar"
    mr.fit.add_request(param="foo", alias="bar")

    # 获取 mr 实例的路由对象
    mr_factory = get_routing_for_object(mr)

    # 断言获取空请求对象时，返回空集合，不包括 "fit" 方法
    assert_request_is_empty(mr_factory, exclude="fit")

    # 断言 mr_factory 的 `fit` 方法请求为 {"foo": "bar"}
    assert mr_factory.fit.requests == {"foo": "bar"}

    # 获取 Consumer 类的路由对象 mr
    mr = get_routing_for_object(Consumer())

    # 断言获取空请求对象时，返回空集合，不包括 "fit" 方法
    assert_request_is_empty(mr, exclude="fit")

    # 断言 mr 的 `fit` 方法请求为 {"prop": None}
    assert mr.fit.requests == {"prop": None}


# 测试 MetadataRequest 类的 consumes 方法
def test_metadata_request_consumes_method():
    # 创建 MetadataRouter 实例 request，拥有者为 "test"
    request = MetadataRouter(owner="test")

    # 断言 consumes 方法对于 "fit" 方法和参数 {"foo"} 返回空集合
    assert request.consumes(method="fit", params={"foo"}) == set()

    # 创建 MetadataRequest 实例 request，拥有者为 "test"
    request = MetadataRequest(owner="test")

    # 向 request 的 `fit` 方法添加请求，参数为 "foo"，别名为 True
    request.fit.add_request(param="foo", alias=True)

    # 断言 consumes 方法对于 "fit" 方法和参数 {"foo"} 返回 {"foo"}
    assert request.consumes(method="fit", params={"foo"}) == {"foo"}

    # 创建 MetadataRequest 实例 request，拥有者为 "test"
    request = MetadataRequest(owner="test")

    # 向 request 的 `fit` 方法添加请求，参数为 "foo"，别名为 "bar"
    request.fit.add_request(param="foo", alias="bar")

    # 断言 consumes 方法对于 "fit" 方法和参数 {"bar", "foo"} 返回 {"bar"}
    assert request.consumes(method="fit", params={"bar", "foo"}) == {"bar"}


# 测试 MetadataRouter 类的 consumes 方法
def test_metadata_router_consumes_method():
    # 由于 `set_fit_request` 在收集测试时不可用，因此在此处而非参数化测试中测试

    # 定义一个消费者类 Consumer，其 `fit` 方法请求 `prop` 元数据
    class Consumer(BaseEstimator):
        __metadata_request__fit = {"prop": None}

    # 断言消费者类 Consumer 的路由对象为空请求对象，不包括 "fit" 方法
    assert_request_is_empty(get_routing_for_object(None))
    assert_request_is_empty(get_routing_for_object(object()))

    # 获取 Consumer 类的路由对象 mr
    mr = get_routing_for_object(Consumer())

    # 断言 mr 的路由对象为空请求对象，不包括 "fit" 方法
    assert_request_is_empty(mr, exclude="fit")

    # 断言 mr 的 `fit` 方法请求为 {"prop": None}
    assert mr.fit.requests == {"prop": None}
    cases = [
        (  # 创建一个包含元组的列表，每个元组包含以下内容：
            WeightedMetaRegressor(  # 使用 WeightedMetaRegressor 类创建对象
                estimator=ConsumingRegressor().set_fit_request(sample_weight=True)  # 创建 ConsumingRegressor 实例，并设置 fit 请求的参数为 sample_weight=True
            ),
            {"sample_weight"},  # 预期 fit 方法的输入参数为 "sample_weight"
            {"sample_weight"},  # 预期 fit 方法的输出参数为 "sample_weight"
        ),
        (
            WeightedMetaRegressor(  # 使用 WeightedMetaRegressor 类创建对象
                estimator=ConsumingRegressor().set_fit_request(
                    sample_weight="my_weights"  # 创建 ConsumingRegressor 实例，并设置 fit 请求的参数为 sample_weight="my_weights"
                )
            ),
            {"my_weights", "sample_weight"},  # 预期 fit 方法的输入参数为 "my_weights" 和 "sample_weight"
            {"my_weights"},  # 预期 fit 方法的输出参数为 "my_weights"
        ),
    ]

    for obj, input, output in cases:  # 对 cases 列表进行迭代，每次迭代解包一个元组到 obj, input, output
        assert obj.get_metadata_routing().consumes(method="fit", params=input) == output  # 断言调用 obj 的 get_metadata_routing 方法，检查其 fit 方法的输入参数是否等于 output
# 定义测试函数，用于测试 MetaEstimator 的警告情况
def test_metaestimator_warnings():
    # 定义一个带有警告的 WeightedMetaRegressorWarn 类，指定在 fit 方法中的 sample_weight 参数会发出警告
    class WeightedMetaRegressorWarn(WeightedMetaRegressor):
        __metadata_request__fit = {"sample_weight": metadata_routing.WARN}

    # 使用 pytest 的 warns 上下文，检测是否会发出 UserWarning，匹配指定的警告信息
    with pytest.warns(
        UserWarning, match="Support for .* has recently been added to this class"
    ):
        # 创建 WeightedMetaRegressorWarn 的实例，设置 estimator 并调用 fit 方法，传入 X、y 和 sample_weight 参数
        WeightedMetaRegressorWarn(
            estimator=LinearRegression().set_fit_request(sample_weight=False)
        ).fit(X, y, sample_weight=my_weights)


# 定义测试函数，用于测试 Estimator 的警告情况
def test_estimator_warnings():
    # 定义一个带有警告的 ConsumingRegressorWarn 类，指定在 fit 方法中的 sample_weight 参数会发出警告
    class ConsumingRegressorWarn(ConsumingRegressor):
        __metadata_request__fit = {"sample_weight": metadata_routing.WARN}

    # 使用 pytest 的 warns 上下文，检测是否会发出 UserWarning，匹配指定的警告信息
    with pytest.warns(
        UserWarning, match="Support for .* has recently been added to this class"
    ):
        # 创建 MetaRegressor 的实例，设置 estimator 并调用 fit 方法，传入 X、y 和 sample_weight 参数
        MetaRegressor(estimator=ConsumingRegressorWarn()).fit(
            X, y, sample_weight=my_weights
        )


# 使用 pytest 的 parametrize 装饰器，对字符串表示进行测试
@pytest.mark.parametrize(
    "obj, string",
    [
        (
            # 创建 MethodMetadataRequest 的实例，设置 owner 和 method，并添加请求参数 foo 和其别名 bar
            MethodMetadataRequest(owner="test", method="fit").add_request(
                param="foo", alias="bar"
            ),
            "{'foo': 'bar'}",
        ),
        (
            # 创建 MetadataRequest 的实例，没有额外的请求参数
            MetadataRequest(owner="test"),
            "{}",
        ),
        (
            # 创建 MetadataRouter 的实例，设置 estimator 和 method_mapping，包括调用者为 predict 的方法映射
            MetadataRouter(owner="test").add(
                estimator=ConsumingRegressor(),
                method_mapping=MethodMapping().add(caller="predict", callee="predict"),
            ),
            (
                "{'estimator': {'mapping': [{'caller': 'predict', 'callee':"
                " 'predict'}], 'router': {'fit': {'sample_weight': None, 'metadata':"
                " None}, 'partial_fit': {'sample_weight': None, 'metadata': None},"
                " 'predict': {'sample_weight': None, 'metadata': None}, 'score':"
                " {'sample_weight': None, 'metadata': None}}}}"
            ),
        ),
    ],
)
# 定义测试函数，验证对象的字符串表示是否正确
def test_string_representations(obj, string):
    assert str(obj) == string


# 使用 pytest 的 parametrize 装饰器，对验证方法参数有效性的测试进行参数化
@pytest.mark.parametrize(
    "obj, method, inputs, err_cls, err_msg",
    [
        (
            # 创建 MethodMapping 的实例，调用 add 方法，传入错误的 callee 参数值
            MethodMapping(),
            "add",
            {"caller": "fit", "callee": "invalid"},
            ValueError,
            "Given callee",
        ),
        (
            # 创建 MethodMapping 的实例，调用 add 方法，传入错误的 caller 参数值
            MethodMapping(),
            "add",
            {"caller": "invalid", "callee": "fit"},
            ValueError,
            "Given caller",
        ),
        (
            # 创建 MetadataRouter 的实例，调用 add_self_request 方法，传入不正确的 obj 参数值
            MetadataRouter(owner="test"),
            "add_self_request",
            {"obj": MetadataRouter(owner="test")},
            ValueError,
            "Given `obj` is neither a `MetadataRequest` nor does it implement",
        ),
        (
            # 创建 ConsumingClassifier 的实例，调用 set_fit_request 方法，传入未预期的参数
            ConsumingClassifier(),
            "set_fit_request",
            {"invalid": True},
            TypeError,
            "Unexpected args",
        ),
    ],
)
# 定义测试函数，验证方法参数的有效性
def test_validations(obj, method, inputs, err_cls, err_msg):
    # 使用 pytest 的 raises 上下文，检测是否会抛出指定的错误类型和错误信息
    with pytest.raises(err_cls, match=err_msg):
        getattr(obj, method)(**inputs)
    # 创建 MethodMapping 对象，并依次添加映射关系 ("fit", "transform") 和 ("fit", "fit")
    mm = (
        MethodMapping()
        .add(caller="fit", callee="transform")  # 添加方法映射关系：caller="fit" -> callee="transform"
        .add(caller="fit", callee="fit")       # 再次添加方法映射关系：caller="fit" -> callee="fit"
    )

    # 将 MethodMapping 对象转换为列表
    mm_list = list(mm)
    # 断言第一个元素是否为 ("fit", "transform")
    assert mm_list[0] == ("fit", "transform")
    # 断言第二个元素是否为 ("fit", "fit")
    assert mm_list[1] == ("fit", "fit")

    # 创建新的 MethodMapping 对象 mm
    mm = MethodMapping()
    # 遍历 METHODS 列表中的每个方法，并将每个方法添加为其自身的映射关系
    for method in METHODS:
        mm.add(caller=method, callee=method)  # 添加方法映射关系：caller=method -> callee=method
        # 断言每个方法对应的 MethodPair(method, method) 在 mm._routes 中存在
        assert MethodPair(method, method) in mm._routes
    # 断言 mm._routes 中映射关系的数量与 METHODS 列表的长度相等
    assert len(mm._routes) == len(METHODS)

    # 创建新的 MethodMapping 对象 mm，并添加映射关系 ("score", "score")
    mm = MethodMapping().add(caller="score", callee="score")
    # 断言 mm 对象的字符串表示形式是否为 "[{'caller': 'score', 'callee': 'score'}]"
    assert repr(mm) == "[{'caller': 'score', 'callee': 'score'}]"
def test_metadatarouter_add_self_request():
    # 创建一个 MetadataRequest 对象，指定所有者为 "nested"
    request = MetadataRequest(owner="nested")
    # 向请求对象的 fit 属性添加一个请求，参数为 "param"，别名为 True
    request.fit.add_request(param="param", alias=True)
    # 创建一个 MetadataRouter 对象，指定所有者为 "test"，并将 request 添加为自身请求
    router = MetadataRouter(owner="test").add_self_request(request)
    # 断言确保 router 的 _self_request 与 request 的字符串表示相同
    assert str(router._self_request) == str(request)
    # 断言确保 router 的 _self_request 不是 request 对象本身，而是其副本
    assert router._self_request is not request

    # 可以将一个 estimator 添加为自身请求
    est = ConsumingRegressor().set_fit_request(sample_weight="my_weights")
    # 创建一个 MetadataRouter 对象，指定所有者为 "test"，并将 est 添加为自身请求
    router = MetadataRouter(owner="test").add_self_request(obj=est)
    # 断言确保 router 的 _self_request 与 est.get_metadata_routing() 的字符串表示相同
    assert str(router._self_request) == str(est.get_metadata_routing())
    # 断言确保 router 的 _self_request 不是 est.get_metadata_routing() 对象本身，而是其副本
    assert router._self_request is not est.get_metadata_routing()

    # 将一个 consumer+router 添加为自身请求应仅添加 consumer 部分
    est = WeightedMetaRegressor(
        estimator=ConsumingRegressor().set_fit_request(sample_weight="nested_weights")
    )
    # 创建一个 MetadataRouter 对象，指定所有者为 "test"，并将 est 添加为自身请求
    router = MetadataRouter(owner="test").add_self_request(obj=est)
    # _get_metadata_request() 返回请求的 consumer 部分
    assert str(router._self_request) == str(est._get_metadata_request())
    # get_metadata_routing() 返回完整的请求集合，包括 consumer 和 router
    assert str(router._self_request) != str(est.get_metadata_routing())
    # 断言确保 router 的 _self_request 不是 est._get_metadata_request() 对象本身，而是其副本
    assert router._self_request is not est._get_metadata_request()


def test_metadata_routing_add():
    # 使用字符串 "method_mapping" 添加一个 estimator
    router = MetadataRouter(owner="test").add(
        est=ConsumingRegressor().set_fit_request(sample_weight="weights"),
        method_mapping=MethodMapping().add(caller="fit", callee="fit"),
    )
    # 断言确保 router 的字符串表示符合预期格式
    assert (
        str(router)
        == "{'est': {'mapping': [{'caller': 'fit', 'callee': 'fit'}], 'router': {'fit':"
        " {'sample_weight': 'weights', 'metadata': None}, 'partial_fit':"
        " {'sample_weight': None, 'metadata': None}, 'predict': {'sample_weight':"
        " None, 'metadata': None}, 'score': {'sample_weight': None, 'metadata':"
        " None}}}}"
    )

    # 使用 MethodMapping 的实例添加一个 estimator
    router = MetadataRouter(owner="test").add(
        method_mapping=MethodMapping().add(caller="fit", callee="score"),
        est=ConsumingRegressor().set_score_request(sample_weight=True),
    )
    # 断言确保 router 的字符串表示符合预期格式
    assert (
        str(router)
        == "{'est': {'mapping': [{'caller': 'fit', 'callee': 'score'}], 'router':"
        " {'fit': {'sample_weight': None, 'metadata': None}, 'partial_fit':"
        " {'sample_weight': None, 'metadata': None}, 'predict': {'sample_weight':"
        " None, 'metadata': None}, 'score': {'sample_weight': True, 'metadata':"
        " None}}}}"
    )
    # 创建一个 MetadataRouter 对象，设置所有者为 "test"
    router = (
        MetadataRouter(owner="test")
        # 添加自请求配置，使用 WeightedMetaRegressor 作为估计器，设置拟合请求中的样本权重为 "self_weights"
        .add_self_request(
            WeightedMetaRegressor(estimator=ConsumingRegressor()).set_fit_request(
                sample_weight="self_weights"
            )
        )
        # 添加一般配置，使用 ConsumingTransformer 作为转换器，设置拟合请求中的样本权重为 "transform_weights"，
        # 并设置方法映射为将 "fit" 方法映射到自身的 "fit" 方法
        .add(
            trs=ConsumingTransformer().set_fit_request(
                sample_weight="transform_weights"
            ),
            method_mapping=MethodMapping().add(caller="fit", callee="fit"),
        )
    )

    # 断言 MetadataRouter 对象的字符串表示符合预期值
    assert (
        str(router)
        == "{'$self_request': {'fit': {'sample_weight': 'self_weights'}, 'score':"
        " {'sample_weight': None}}, 'trs': {'mapping': [{'caller': 'fit', 'callee':"
        " 'fit'}], 'router': {'fit': {'sample_weight': 'transform_weights',"
        " 'metadata': None}, 'transform': {'sample_weight': None, 'metadata': None},"
        " 'inverse_transform': {'sample_weight': None, 'metadata': None}}}}"
    )

    # 断言调用 MetadataRouter 对象的 _get_param_names 方法，返回包含指定参数名称的集合，
    # 方法为 "fit"，同时返回别名（如果有的话），不忽略自请求配置
    assert router._get_param_names(
        method="fit", return_alias=True, ignore_self_request=False
    ) == {"transform_weights", "metadata", "self_weights"}

    # 断言调用 MetadataRouter 对象的 _get_param_names 方法，返回包含指定参数名称的集合，
    # 方法为 "fit"，返回原始名称而不是别名，不忽略自请求配置
    assert router._get_param_names(
        method="fit", return_alias=False, ignore_self_request=False
    ) == {"sample_weight", "metadata", "transform_weights"}

    # 断言调用 MetadataRouter 对象的 _get_param_names 方法，返回包含指定参数名称的集合，
    # 方法为 "fit"，返回原始名称而不是别名，并且忽略自请求配置
    assert router._get_param_names(
        method="fit", return_alias=False, ignore_self_request=True
    ) == {"metadata", "transform_weights"}

    # 断言调用 MetadataRouter 对象的 _get_param_names 方法，返回包含指定参数名称的集合，
    # 方法为 "fit"，返回别名（如果有的话），并且忽略自请求配置
    assert router._get_param_names(
        method="fit", return_alias=True, ignore_self_request=True
    ) == router._get_param_names(
        method="fit", return_alias=False, ignore_self_request=True
    )
def test_method_generation():
    # Test if all required request methods are generated.

    # TODO: these test classes can be moved to sklearn.utils._testing once we
    # have a better idea of what the commonly used classes are.

    # 定义一个简单的估计器类，继承自BaseEstimator
    class SimpleEstimator(BaseEstimator):
        # This class should have no set_{method}_request
        # 定义fit方法，无需生成set_{method}_request
        def fit(self, X, y):
            pass  # pragma: no cover

        # 定义fit_transform方法，无需生成set_{method}_request
        def fit_transform(self, X, y):
            pass  # pragma: no cover

        # 定义fit_predict方法，无需生成set_{method}_request
        def fit_predict(self, X, y):
            pass  # pragma: no cover

        # 定义partial_fit方法，无需生成set_{method}_request
        def partial_fit(self, X, y):
            pass  # pragma: no cover

        # 定义predict方法，无需生成set_{method}_request
        def predict(self, X):
            pass  # pragma: no cover

        # 定义predict_proba方法，无需生成set_{method}_request
        def predict_proba(self, X):
            pass  # pragma: no cover

        # 定义predict_log_proba方法，无需生成set_{method}_request
        def predict_log_proba(self, X):
            pass  # pragma: no cover

        # 定义decision_function方法，无需生成set_{method}_request
        def decision_function(self, X):
            pass  # pragma: no cover

        # 定义score方法，无需生成set_{method}_request
        def score(self, X, y):
            pass  # pragma: no cover

        # 定义split方法，无需生成set_{method}_request
        def split(self, X, y=None):
            pass  # pragma: no cover

        # 定义transform方法，无需生成set_{method}_request
        def transform(self, X):
            pass  # pragma: no cover

        # 定义inverse_transform方法，无需生成set_{method}_request
        def inverse_transform(self, X):
            pass  # pragma: no cover

    # 遍历METHODS列表中的每个方法名
    for method in METHODS:
        # 断言SimpleEstimator类中没有名为set_{method}_request的属性
        assert not hasattr(SimpleEstimator(), f"set_{method}_request")

    # 重新定义SimpleEstimator类，这次需要生成每个set_{method}_request方法
    class SimpleEstimator(BaseEstimator):
        # This class should have every set_{method}_request
        # 定义fit方法，需要生成set_{method}_request
        def fit(self, X, y, sample_weight=None):
            pass  # pragma: no cover

        # 定义fit_transform方法，需要生成set_{method}_request
        def fit_transform(self, X, y, sample_weight=None):
            pass  # pragma: no cover

        # 定义fit_predict方法，需要生成set_{method}_request
        def fit_predict(self, X, y, sample_weight=None):
            pass  # pragma: no cover

        # 定义partial_fit方法，需要生成set_{method}_request
        def partial_fit(self, X, y, sample_weight=None):
            pass  # pragma: no cover

        # 定义predict方法，需要生成set_{method}_request
        def predict(self, X, sample_weight=None):
            pass  # pragma: no cover

        # 定义predict_proba方法，需要生成set_{method}_request
        def predict_proba(self, X, sample_weight=None):
            pass  # pragma: no cover

        # 定义predict_log_proba方法，需要生成set_{method}_request
        def predict_log_proba(self, X, sample_weight=None):
            pass  # pragma: no cover

        # 定义decision_function方法，需要生成set_{method}_request
        def decision_function(self, X, sample_weight=None):
            pass  # pragma: no cover

        # 定义score方法，需要生成set_{method}_request
        def score(self, X, y, sample_weight=None):
            pass  # pragma: no cover

        # 定义split方法，需要生成set_{method}_request
        def split(self, X, y=None, sample_weight=None):
            pass  # pragma: no cover

        # 定义transform方法，需要生成set_{method}_request
        def transform(self, X, sample_weight=None):
            pass  # pragma: no cover

        # 定义inverse_transform方法，需要生成set_{method}_request
        def inverse_transform(self, X, sample_weight=None):
            pass  # pragma: no cover

    # composite methods shouldn't have a corresponding set method.
    # 遍历COMPOSITE_METHODS列表中的每个方法名
    for method in COMPOSITE_METHODS:
        # 断言SimpleEstimator类中没有名为set_{method}_request的属性
        assert not hasattr(SimpleEstimator(), f"set_{method}_request")

    # simple methods should have a corresponding set method.
    # 遍历SIMPLE_METHODS列表中的每个方法名
    for method in SIMPLE_METHODS:
        # 断言SimpleEstimator类中有名为set_{method}_request的属性
        assert hasattr(SimpleEstimator(), f"set_{method}_request")


def test_composite_methods():
    # Test the behavior and the values of methods (composite methods) whose
    # request values are a union of requests by other methods (simple methods).
    # fit_transform and fit_predict are the only composite methods we have in
    # scikit-learn.
    class SimpleEstimator(BaseEstimator):
        # This class should have every set_{method}_request

        # 定义一个空的fit方法，用于训练模型，但实际未实现任何功能
        def fit(self, X, y, foo=None, bar=None):
            pass  # pragma: no cover

        # 定义一个空的predict方法，用于预测，但实际未实现任何功能
        def predict(self, X, foo=None, bar=None):
            pass  # pragma: no cover

        # 定义一个空的transform方法，用于转换数据，但实际未实现任何功能
        def transform(self, X, other_param=None):
            pass  # pragma: no cover

    # 创建SimpleEstimator类的实例对象
    est = SimpleEstimator()

    # Since no request is set for fit or predict or transform, the request for
    # fit_transform and fit_predict should also be empty.
    # 断言fit_transform和fit_predict的请求应该为空字典
    assert est.get_metadata_routing().fit_transform.requests == {
        "bar": None,
        "foo": None,
        "other_param": None,
    }

    # 断言fit_predict的请求应该为空字典
    assert est.get_metadata_routing().fit_predict.requests == {"bar": None, "foo": None}

    # setting the request on only one of them should raise an error
    # 在其中一个方法上设置请求应该引发错误
    est.set_fit_request(foo=True, bar="test")
    with pytest.raises(ValueError, match="Conflicting metadata requests for"):
        est.get_metadata_routing().fit_predict

    # setting the request on the other one should fail if not the same as the
    # first method
    # 如果不与第一个方法相同，则在另一个方法上设置请求应该失败
    est.set_predict_request(bar=True)
    with pytest.raises(ValueError, match="Conflicting metadata requests for"):
        est.get_metadata_routing().fit_predict

    # now the requests are consistent and getting the requests for fit_predict
    # shouldn't raise.
    # 现在请求是一致的，获取fit_predict的请求不应该引发异常
    est.set_predict_request(foo=True, bar="test")
    est.get_metadata_routing().fit_predict

    # setting the request for a none-overlapping parameter would merge them
    # together.
    # 设置一个不重叠的参数请求应该会合并它们
    est.set_transform_request(other_param=True)

    # 断言fit_transform的请求应该包含合并后的请求参数
    assert est.get_metadata_routing().fit_transform.requests == {
        "bar": "test",
        "foo": True,
        "other_param": True,
    }
# 当特性标志禁用时，测试确保调用 set_{method}_requests 会引发错误。
def test_no_feature_flag_raises_error():
    # 使用上下文管理器设置元数据路由为禁用状态
    with config_context(enable_metadata_routing=False):
        # 断言调用 ConsumingClassifier().set_fit_request(sample_weight=True) 会抛出 RuntimeError，并匹配指定错误信息
        with pytest.raises(RuntimeError, match="This method is only available"):
            ConsumingClassifier().set_fit_request(sample_weight=True)


# 测试当未请求元数据时，将 None 作为元数据传递不会引发异常
def test_none_metadata_passed():
    # 使用 MetaRegressor 进行拟合，传入的 sample_weight 为 None
    MetaRegressor(estimator=ConsumingRegressor()).fit(X, y, sample_weight=None)


# 测试当未传递元数据时，meta-estimator 不支持元数据路由的情况
# 用于验证 https://github.com/scikit-learn/scikit-learn/issues/28246 的非回归测试
def test_no_metadata_always_works():
    # 定义一个不支持元数据路由的 Estimator 类
    class Estimator(_RoutingNotSupportedMixin, BaseEstimator):
        def fit(self, X, y, metadata=None):
            return self

    # 传入 Estimator() 作为 estimator 进行拟合，不传递元数据
    MetaRegressor(estimator=Estimator()).fit(X, y)
    # 断言传递了元数据时会引发 NotImplementedError，并匹配指定错误信息
    with pytest.raises(
        NotImplementedError, match="Estimator has not implemented metadata routing yet."
    ):
        MetaRegressor(estimator=Estimator()).fit(X, y, metadata=my_groups)


# 测试当未设置 set_{method}_request 的情况下，UnsetMetadataPassedError 能够引发正确的错误消息
def test_unsetmetadatapassederror_correct():
    # 使用 WeightedMetaClassifier 包装 ConsumingClassifier 进行拟合
    weighted_meta = WeightedMetaClassifier(estimator=ConsumingClassifier())
    # 构建一个简单的管道 SimplePipeline 包含 weighted_meta
    pipe = SimplePipeline([weighted_meta])
    # 准备匹配的错误消息
    msg = re.escape(
        "[metadata] are passed but are not explicitly set as requested or not requested"
        " for ConsumingClassifier.fit, which is used within WeightedMetaClassifier.fit."
        " Call `ConsumingClassifier.set_fit_request({metadata}=True/False)` for each"
        " metadata you want to request/ignore."
    )

    # 断言调用 pipe.fit(X, y, metadata="blah") 会引发 UnsetMetadataPassedError，并匹配指定错误信息
    with pytest.raises(UnsetMetadataPassedError, match=msg):
        pipe.fit(X, y, metadata="blah")


# 测试当未设置 set_{method}_request 的情况下，UnsetMetadataPassedError 能够引发正确的错误消息
def test_unsetmetadatapassederror_correct_for_composite_methods():
    # 实例化 ConsumingTransformer
    consuming_transformer = ConsumingTransformer()
    # 构建一个管道 Pipeline，包含 "consuming_transformer"
    pipe = Pipeline([("consuming_transformer", consuming_transformer)])

    # 准备匹配的错误消息
    msg = re.escape(
        "[metadata] are passed but are not explicitly set as requested or not requested"
        " for ConsumingTransformer.fit_transform, which is used within"
        " Pipeline.fit_transform. Call"
        " `ConsumingTransformer.set_fit_request({metadata}=True/False)"
        ".set_transform_request({metadata}=True/False)`"
        " for each metadata you want to request/ignore."
    )

    # 断言调用 pipe.fit_transform(X, y, metadata="blah") 会引发 UnsetMetadataPassedError，并匹配指定错误信息
    with pytest.raises(UnsetMetadataPassedError, match=msg):
        pipe.fit_transform(X, y, metadata="blah")


# 测试未绑定的 set 方法是否有效
def test_unbound_set_methods_work():
    """Tests that if the set_{method}_request is unbound, it still works.

    Also test that passing positional arguments to the set_{method}_request fails
    with the right TypeError message.

    Non-regression test for https://github.com/scikit-learn/scikit-learn/issues/28632
    """

    # 定义一个示例类 A，继承自 BaseEstimator
    class A(BaseEstimator):
        # 实现 fit 方法，接受 X、y 和可选的 sample_weight 参数，并返回实例本身
        def fit(self, X, y, sample_weight=None):
            return self

    # 创建一个用于匹配 TypeError 的错误消息模板
    error_message = re.escape(
        "set_fit_request() takes 0 positional arguments but 1 were given"
    )

    # 在使描述符方法解绑之前测试传递位置参数的错误
    # 使用 pytest 断言来检查是否会抛出预期的 TypeError 异常
    with pytest.raises(TypeError, match=error_message):
        A().set_fit_request(True)

    # 以下操作某种方式会解绑描述符方法，导致 `instance` 参数变为 None，
    # 而 `self` 作为位置参数传递给描述符方法。
    A.set_fit_request = A.set_fit_request

    # 这里应该像往常一样通过测试
    # 调用 set_fit_request 方法，传递 sample_weight=True 参数
    A().set_fit_request(sample_weight=True)

    # 在使描述符方法解绑之后再次测试传递位置参数的错误
    with pytest.raises(TypeError, match=error_message):
        A().set_fit_request(True)
```