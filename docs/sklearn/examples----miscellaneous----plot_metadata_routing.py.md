# `D:\src\scipysrc\scikit-learn\examples\miscellaneous\plot_metadata_routing.py`

```
# 导入警告模块
import warnings
# 导入漂亮打印模块
from pprint import pprint

# 导入NumPy库
import numpy as np

# 导入scikit-learn库的配置设置函数
from sklearn import set_config
# 导入scikit-learn库的基础类
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MetaEstimatorMixin,
    RegressorMixin,
    TransformerMixin,
    clone,
)
# 导入scikit-learn库的线性回归模型
from sklearn.linear_model import LinearRegression
# 导入scikit-learn库的元数据路由相关工具
from sklearn.utils import metadata_routing
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    get_routing_for_object,
    process_routing,
)
# 导入scikit-learn库的验证函数
from sklearn.utils.validation import check_is_fitted

# 设定随机数生成的样本数和特征数
n_samples, n_features = 100, 4
# 使用随机数种子初始化随机数生成器
rng = np.random.RandomState(42)
# 生成随机的样本数据和标签数据
X = rng.rand(n_samples, n_features)
y = rng.randint(0, 2, size=n_samples)
# 创建随机的分组数据
my_groups = rng.randint(0, 10, size=n_samples)
# 创建随机的样本权重数据
my_weights = rng.rand(n_samples)
my_other_weights = rng.rand(n_samples)

# %%
# 元数据路由功能只有在显式启用时才可用：
set_config(enable_metadata_routing=True)


# %%
# 这个实用函数是一个虚拟函数，用于检查是否传递了元数据：
def check_metadata(obj, **kwargs):
    for key, value in kwargs.items():
        if value is not None:
            # 打印接收到的元数据名称、长度以及对象类名
            print(
                f"Received {key} of length = {len(value)} in {obj.__class__.__name__}."
            )
        else:
            # 打印元数据名称为空的提示信息和对象类名
            print(f"{key} is None in {obj.__class__.__name__}.")


# %%
# 一个用于漂亮打印对象元数据路由信息的实用函数：
def print_routing(obj):
    # 使用漂亮打印函数打印对象的元数据路由序列化信息
    pprint(obj.get_metadata_routing()._serialize())


# %%
# 消费者估计器
# -------------------
# 这里演示了如何将一个估计器作为消费者来暴露所需的API，以支持元数据路由。想象一个简单的分类器，
# 在其“fit”方法中接受“sample_weight”作为元数据，在其“predict”方法中接受“groups”。
class ExampleClassifier(ClassifierMixin, BaseEstimator):
    # 定义一个示例分类器类，继承自ClassifierMixin和BaseEstimator

    def fit(self, X, y, sample_weight=None):
        # 拟合方法，检查元数据
        check_metadata(self, sample_weight=sample_weight)
        # 所有分类器在拟合后需要暴露classes_属性
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X, groups=None):
        # 预测方法，检查元数据
        check_metadata(self, groups=groups)
        # 返回固定值1，一个简单的分类器
        return np.ones(len(X))


# %%
# 上述的分类器现在已经具备了消费元数据的所有必要条件。这是通过:class:`~base.BaseEstimator`中的一些魔法实现的。
# 现在以上述类暴露了三个方法：``set_fit_request``, ``set_predict_request``，和 ``get_metadata_routing``。
# 除此之外，还有一个``set_score_request``用于``sample_weight``，这是因为:class:`~base.ClassifierMixin`实现了接受``sample_weight``的``score``方法。
# 对于继承自:class:`~base.RegressorMixin`的回归器，也是一样的情况。
#
# 默认情况下，不会请求任何元数据，可以看到如下输出：

print_routing(ExampleClassifier())

# %%
# 上述输出意味着``ExampleClassifier``不需要``sample_weight``和``groups``元数据，如果给定了这些元数据，路由器应该抛出错误，
# 因为用户并没有显式设置它们是否需要。对于``score``方法中的``sample_weight``也是同样的情况，它是从:class:`~base.ClassifierMixin`继承而来。
# 若要显式设置这些元数据的请求值，可以使用这些方法：

est = (
    ExampleClassifier()
    .set_fit_request(sample_weight=False)
    .set_predict_request(groups=True)
    .set_score_request(sample_weight=False)
)
print_routing(est)

# %%
# .. note ::
#     请注意，只要上述的分类器没有在元分类器中使用，用户不需要为元数据设置任何请求，设置的值会被忽略，因为消费者不验证或路由给定的元数据。
#     对上述分类器的简单使用会按预期工作。

est = ExampleClassifier()
est.fit(X, y, sample_weight=my_weights)
est.predict(X[:3, :], groups=my_groups)

# %%
# 元数据路由的元分类器
# ----------------------
# 现在，我们展示如何设计一个作为路由器的元分类器。作为一个简化的示例，这里是一个元分类器，除了路由元数据外，几乎什么都不做。

class MetaClassifier(MetaEstimatorMixin, ClassifierMixin, BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator
    # 定义获取元估计器路由的方法
    def get_metadata_routing(self):
        # 创建一个 MetadataRouter 实例，用于定义元估计器的路由
        router = MetadataRouter(owner=self.__class__.__name__).add(
            # 将元估计器的方法映射添加到路由中
            estimator=self.estimator,
            method_mapping=MethodMapping()
            .add(caller="fit", callee="fit")
            .add(caller="predict", callee="predict")
            .add(caller="score", callee="score"),
        )
        return router

    # 实现拟合方法，对于元估计器而言，这里需要验证给定的元数据
    def fit(self, X, y, **fit_params):
        # 获取对象的元数据路由副本
        request_router = get_routing_for_object(self)
        # 验证给定的元数据，针对父类方法 'fit'
        request_router.validate_metadata(params=fit_params, method="fit")
        # 根据元数据路由信息将参数路由到底层估计器的 'fit' 方法
        routed_params = request_router.route_params(params=fit_params, caller="fit")

        # 调用克隆的子估计器的 'fit' 方法，并将类属性赋予元估计器
        self.estimator_ = clone(self.estimator).fit(X, y, **routed_params.estimator.fit)
        self.classes_ = self.estimator_.classes_
        return self

    # 实现预测方法，类似于 'fit' 方法，需要验证元数据并准备输入给底层 'predict' 方法
    def predict(self, X, **predict_params):
        # 检查元估计器是否已经拟合
        check_is_fitted(self)
        # 获取对象的元数据路由副本
        request_router = get_routing_for_object(self)
        # 验证给定的元数据，针对方法 'predict'
        request_router.validate_metadata(params=predict_params, method="predict")
        # 根据元数据路由信息将参数路由到底层估计器的 'predict' 方法
        routed_params = request_router.route_params(
            params=predict_params, caller="predict"
        )
        # 调用底层估计器的 'predict' 方法进行预测
        return self.estimator_.predict(X, **routed_params.estimator.predict)
# %%
# 上面的代码示例展示了如何使用 MetaClassifier 类和相关的方法来进行元估计器的配置和调用。
#
# 首先，创建了一个 MetaClassifier 实例 meta_est，其中 estimator 参数使用了 ExampleClassifier 的一个实例，并调用了其 set_fit_request 方法设置了 sample_weight=True。
# 这个配置指定了在调用 meta_est.fit(X, y, sample_weight=my_weights) 时，将 sample_weight 参数传递给底层的 ExampleClassifier 实例的 fit 方法。
meta_est = MetaClassifier(
    estimator=ExampleClassifier().set_fit_request(sample_weight=True)
)
meta_est.fit(X, y, sample_weight=my_weights)

# %%
# 注意，在上面的示例中，我们通过 ExampleClassifier 调用了 check_metadata() 实用函数来验证 sample_weight 是否正确传递。
# 如果没有像下面的示例那样正确传递，会打印出 sample_weight 的值为 None：
meta_est.fit(X, y)

# %%
# 如果我们传递了一个未知的元数据参数，会引发 TypeError 错误：
try:
    meta_est.fit(X, y, test=my_weights)
except TypeError as e:
    print(e)

# %%
# 如果传递了一个并未显式请求的元数据参数，会引发 ValueError 错误：
try:
    meta_est.fit(X, y, sample_weight=my_weights).predict(X, groups=my_groups)
except ValueError as e:
    print(e)

# %%
# 另外，如果明确将某个元数据参数设置为不请求，但仍然提供了该参数，会引发 TypeError 错误：
meta_est = MetaClassifier(
    estimator=ExampleClassifier()
    .set_fit_request(sample_weight=True)
    .set_predict_request(groups=False)
)
try:
    meta_est.fit(X, y, sample_weight=my_weights).predict(X[:3, :], groups=my_groups)
except TypeError as e:
    print(e)

# %%
# 还有一个概念是**别名元数据**，即当一个估计器请求一个具有与默认变量名不同的变量名的元数据时。
# 例如，在一个管道设置中，有两个估计器，一个可能请求 sample_weight1，另一个请求 sample_weight2。
# 这并不改变估计器期望的内容，只是告诉元估计器如何将提供的元数据映射到所需的内容。
# 在这个例子中，我们将 aliased_sample_weight 传递给 meta_estimator，但是 meta_estimator 理解 aliased_sample_weight 是 sample_weight 的别名，并将其作为 sample_weight 传递给底层的估计器：
meta_est = MetaClassifier(
    estimator=ExampleClassifier().set_fit_request(sample_weight="aliased_sample_weight")
)
meta_est.fit(X, y, aliased_sample_weight=my_weights)

# %%
# Passing ``sample_weight`` here will fail since it is requested with an
# alias and ``sample_weight`` with that name is not requested:
try:
    # 尝试使用 `meta_est` 的 `fit` 方法来拟合数据 `X` 和目标 `y`，并传入样本权重 `my_weights`
    meta_est.fit(X, y, sample_weight=my_weights)
except TypeError as e:
    # 如果发生 `TypeError` 异常，打印异常信息
    print(e)

# %%
# This leads us to the ``get_metadata_routing``. The way routing works in
# scikit-learn is that consumers request what they need, and routers pass that
# along. Additionally, a router exposes what it requires itself so that it can
# be used inside another router, e.g. a pipeline inside a grid search object.
# The output of the ``get_metadata_routing`` which is a dictionary
# representation of a :class:`~utils.metadata_routing.MetadataRouter`, includes
# the complete tree of requested metadata by all nested objects and their
# corresponding method routings, i.e. which method of a sub-estimator is used
# in which method of a meta-estimator:
# 打印 `meta_est` 的元数据路由信息
print_routing(meta_est)

# %%
# As you can see, the only metadata requested for method ``fit`` is
# ``"sample_weight"`` with ``"aliased_sample_weight"`` as the alias. The
# ``~utils.metadata_routing.MetadataRouter`` class enables us to easily create
# the routing object which would create the output we need for our
# ``get_metadata_routing``.
#
# In order to understand how aliases work in meta-estimators, imagine our
# meta-estimator inside another one:
# 创建一个新的元分类器 `meta_meta_est`，并使用 `my_weights` 作为 `aliased_sample_weight` 传递给其 `fit` 方法
meta_meta_est = MetaClassifier(estimator=meta_est).fit(
    X, y, aliased_sample_weight=my_weights
)

# %%
# In the above example, this is how the ``fit`` method of `meta_meta_est`
# will call their sub-estimator's ``fit`` methods::
#
#     # user feeds `my_weights` as `aliased_sample_weight` into `meta_meta_est`:
#     meta_meta_est.fit(X, y, aliased_sample_weight=my_weights):
#         ...
#
#         # the first sub-estimator (`meta_est`) expects `aliased_sample_weight`
#         self.estimator_.fit(X, y, aliased_sample_weight=aliased_sample_weight):
#             ...
#
#             # the second sub-estimator (`est`) expects `sample_weight`
#             self.estimator_.fit(X, y, sample_weight=aliased_sample_weight):
#                 ...
# 描述了如何通过元估计器 `meta_meta_est` 的 `fit` 方法传递样本权重到其子估计器的 `fit` 方法

# %%
# Consuming and routing Meta-Estimator
# ------------------------------------
# For a slightly more complex example, consider a meta-estimator that routes
# metadata to an underlying estimator as before, but it also uses some metadata
# in its own methods. This meta-estimator is a consumer and a router at the
# same time. Implementing one is very similar to what we had before, but with a
# few tweaks.
#
# 创建一个消费和路由元估计器 `RouterConsumerClassifier`，该估计器通过 `MetaEstimatorMixin`、`ClassifierMixin` 和 `BaseEstimator` 实现元估计器功能
class RouterConsumerClassifier(MetaEstimatorMixin, ClassifierMixin, BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator
    # 获取当前对象的元数据路由器
    def get_metadata_routing(self):
        # 创建一个MetadataRouter对象，使用当前类名作为所有者
        router = (
            MetadataRouter(owner=self.__class__.__name__)
            # 添加当前对象自身的元数据路由请求值，用于在元估计器中使用
            .add_self_request(self)
            # 添加元数据路由请求值，用于在子估计器中使用
            .add(
                estimator=self.estimator,
                method_mapping=MethodMapping()
                # 添加方法映射，将调用者为"fit"的方法路由到"fit"方法
                .add(caller="fit", callee="fit")
                # 添加方法映射，将调用者为"predict"的方法路由到"predict"方法
                .add(caller="predict", callee="predict")
                # 添加方法映射，将调用者为"score"的方法路由到"score"方法
                .add(caller="score", callee="score"),
            )
        )
        return router

    # 因为这里使用了`sample_weight`参数，应该在方法签名中明确定义。所有其他仅路由的元数据将作为`**fit_params`传递：
    def fit(self, X, y, sample_weight, **fit_params):
        # 如果估计器为None，则抛出值错误异常
        if self.estimator is None:
            raise ValueError("estimator cannot be None!")

        # 检查并验证元数据，其中包括`sample_weight`
        check_metadata(self, sample_weight=sample_weight)

        # 如果`sample_weight`不为None，则将其添加到`fit_params`字典中
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight

        # 获取当前对象的路由器
        request_router = get_routing_for_object(self)
        # 验证给定的元数据参数，方法为"fit"
        request_router.validate_metadata(params=fit_params, method="fit")
        # 路由参数并调用估计器的"fit"方法，使用路由后的参数
        routed_params = request_router.route_params(params=fit_params, caller="fit")
        self.estimator_ = clone(self.estimator).fit(X, y, **routed_params.estimator.fit)
        # 设置类别属性为估计器的类别属性
        self.classes_ = self.estimator_.classes_
        return self

    # 预测方法，接受输入X以及可选的预测参数
    def predict(self, X, **predict_params):
        # 检查对象是否已拟合
        check_is_fitted(self)
        # 获取当前对象的路由器
        request_router = get_routing_for_object(self)
        # 验证给定的预测参数元数据，方法为"predict"
        request_router.validate_metadata(params=predict_params, method="predict")
        # 路由参数并调用估计器的"predict"方法，使用路由后的参数
        routed_params = request_router.route_params(
            params=predict_params, caller="predict"
        )
        # 返回估计器的预测结果
        return self.estimator_.predict(X, **routed_params.estimator.predict)
# %%
# 上述元估计器与我们之前的元估计器的关键区别在于显式接受 ``sample_weight`` 参数并在 ``fit`` 方法中将其包含在 ``fit_params`` 中。
# 由于 ``sample_weight`` 是一个显式参数，我们可以确保该方法中存在 ``set_fit_request(sample_weight=...)``。该元估计器既是消费者，也是 ``sample_weight`` 的路由器。

# %%
# 在 ``get_metadata_routing`` 方法中，我们使用 ``add_self_request`` 将 ``self`` 添加到路由中，表明此估计器既消费 ``sample_weight`` 也是路由器。
# 这还会在路由信息中添加一个 ``$self_request`` 键。现在让我们看一些示例：

# %%
# - 没有请求元数据
meta_est = RouterConsumerClassifier(estimator=ExampleClassifier())
print_routing(meta_est)

# %%
# - 子估计器请求 ``sample_weight``
meta_est = RouterConsumerClassifier(
    estimator=ExampleClassifier().set_fit_request(sample_weight=True)
)
print_routing(meta_est)

# %%
# - 元估计器请求 ``sample_weight``
meta_est = RouterConsumerClassifier(estimator=ExampleClassifier()).set_fit_request(
    sample_weight=True
)
print_routing(meta_est)

# %%
# 注意上述请求元数据表示中的差异。
#
# - 我们还可以为元估计器和子估计器的 ``fit`` 方法传递不同的值来别名元数据：

meta_est = RouterConsumerClassifier(
    estimator=ExampleClassifier().set_fit_request(sample_weight="clf_sample_weight"),
).set_fit_request(sample_weight="meta_clf_sample_weight")
print_routing(meta_est)

# %%
# 然而，元估计器的 ``fit`` 方法只需要子估计器的别名，它将自己的样本权重视为 ``sample_weight``，
# 因为它不验证和路由自己所需的元数据：
meta_est.fit(X, y, sample_weight=my_weights, clf_sample_weight=my_other_weights)

# %%
# - 仅在子估计器上使用别名：
#
# 当我们不希望元估计器使用元数据，而子估计器应该使用时，这是有用的。
meta_est = RouterConsumerClassifier(
    estimator=ExampleClassifier().set_fit_request(sample_weight="aliased_sample_weight")
)
print_routing(meta_est)

# %%
# 元估计器不能使用 ``aliased_sample_weight``，因为它期望作为 ``sample_weight`` 传递。
# 即使在元估计器上设置了 ``set_fit_request(sample_weight=True)`` 也是如此。

# %%
# 简单管道
# ---------------
# 一个稍微复杂的用例是类似于 :class:`~pipeline.Pipeline` 的元估计器。这是一个元估计器，接受一个转换器和一个分类器。
# 在调用其 ``fit`` 方法时，它会先对转换器进行 ``fit`` 和 ``transform``，然后在转换后的数据上运行分类器。
# 在 ``predict`` 时，它会先对转换器进行 ``transform``，然后在转换后的新数据上使用分类器的 ``predict`` 方法进行预测。

class SimplePipeline(ClassifierMixin, BaseEstimator):
    # 初始化方法，接受一个transformer和一个classifier作为参数
    def __init__(self, transformer, classifier):
        # 将传入的transformer赋值给实例变量self.transformer
        self.transformer = transformer
        # 将传入的classifier赋值给实例变量self.classifier
        self.classifier = classifier

    # 获取元数据路由对象的方法
    def get_metadata_routing(self):
        # 创建一个MetadataRouter对象，其所有者为当前类的类名
        router = (
            MetadataRouter(owner=self.__class__.__name__)
            # 添加transformer的路由配置
            .add(
                transformer=self.transformer,
                method_mapping=MethodMapping()
                # 将元数据路由到transformer的`fit`和`transform`方法，
                # 反映了在SimplePipeline中如何调用transformer的这两个方法
                .add(caller="fit", callee="fit")
                .add(caller="fit", callee="transform")
                .add(caller="predict", callee="transform"),
            )
            # 添加classifier的路由配置
            .add(
                classifier=self.classifier,
                method_mapping=MethodMapping()
                # 将元数据路由到classifier的`fit`和`predict`方法
                .add(caller="fit", callee="fit")
                .add(caller="predict", callee="predict"),
            )
        )
        # 返回配置好的路由对象
        return router

    # 拟合方法，用于拟合transformer和classifier
    def fit(self, X, y, **fit_params):
        # 处理路由参数，获取处理后的参数
        routed_params = process_routing(self, "fit", **fit_params)

        # 克隆并拟合transformer，将拟合后的对象保存在self.transformer_中
        self.transformer_ = clone(self.transformer).fit(
            X, y, **routed_params.transformer.fit
        )
        # 使用拟合后的transformer对象对X进行变换，得到变换后的数据X_transformed
        X_transformed = self.transformer_.transform(
            X, **routed_params.transformer.transform
        )

        # 克隆并拟合classifier，将拟合后的对象保存在self.classifier_中
        self.classifier_ = clone(self.classifier).fit(
            X_transformed, y, **routed_params.classifier.fit
        )
        # 返回实例本身，用于方法链式调用
        return self

    # 预测方法，用于对输入数据X进行预测
    def predict(self, X, **predict_params):
        # 处理路由参数，获取处理后的参数
        routed_params = process_routing(self, "predict", **predict_params)

        # 使用拟合后的transformer对象对X进行变换，得到变换后的数据X_transformed
        X_transformed = self.transformer_.transform(
            X, **routed_params.transformer.transform
        )
        # 使用拟合后的classifier对象对变换后的数据X_transformed进行预测，返回预测结果
        return self.classifier_.predict(
            X_transformed, **routed_params.classifier.predict
        )
# %%
# Note the usage of :class:`~utils.metadata_routing.MethodMapping` to
# declare which methods of the child estimator (callee) are used in which
# methods of the meta estimator (caller). As you can see, `SimplePipeline` uses
# the transformer's ``transform`` and ``fit`` methods in ``fit``, and its
# ``transform`` method in ``predict``, and that's what you see implemented in
# the routing structure of the pipeline class.
#
# Another difference in the above example with the previous ones is the usage
# of :func:`~utils.metadata_routing.process_routing`, which processes the input
# parameters, does the required validation, and returns the `routed_params`
# which we had created in previous examples. This reduces the boilerplate code
# a developer needs to write in each meta-estimator's method. Developers are
# strongly recommended to use this function unless there is a good reason
# against it.
#
# In order to test the above pipeline, let's add an example transformer.

class ExampleTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y, sample_weight=None):
        # Validate metadata for fitting process
        check_metadata(self, sample_weight=sample_weight)
        return self

    def transform(self, X, groups=None):
        # Validate metadata for transformation process
        check_metadata(self, groups=groups)
        return X

    def fit_transform(self, X, y, sample_weight=None, groups=None):
        # Combined fit and transform, ensuring metadata validity
        return self.fit(X, y, sample_weight).transform(X, groups)


# %%
# Note that in the above example, we have implemented ``fit_transform`` which
# calls ``fit`` and ``transform`` with the appropriate metadata. This is only
# required if ``transform`` accepts metadata, since the default ``fit_transform``
# implementation in :class:`~base.TransformerMixin` doesn't pass metadata to
# ``transform``.
#
# Now we can test our pipeline, and see if metadata is correctly passed around.
# This example uses our `SimplePipeline`, our `ExampleTransformer`, and our
# `RouterConsumerClassifier` which uses our `ExampleClassifier`.

pipe = SimplePipeline(
    transformer=ExampleTransformer()
    # Configure transformer to accept sample_weight during fit
    .set_fit_request(sample_weight=True)
    # Configure transformer to accept groups during transform
    .set_transform_request(groups=True),
    classifier=RouterConsumerClassifier(
        estimator=ExampleClassifier()
        # Configure sub-estimator to accept sample_weight during fit
        .set_fit_request(sample_weight=True)
        # Configure sub-estimator to not accept groups during predict
        .set_predict_request(groups=False),
    )
    # Configure meta-estimator to accept sample_weight during fit
    .set_fit_request(sample_weight=True),
)

# Fit the pipeline with specified weights and groups, then predict on a subset of data
pipe.fit(X, y, sample_weight=my_weights, groups=my_groups).predict(
    X[:3], groups=my_groups
)

# %%
# Deprecation / Default Value Change
# ----------------------------------
# In this section we show how one should handle the case where a router becomes
# also a consumer, especially when it consumes the same metadata as its
# 定义一个元估计器类，继承自MetaEstimatorMixin、RegressorMixin和BaseEstimator
class MetaRegressor(MetaEstimatorMixin, RegressorMixin, BaseEstimator):
    
    # 初始化方法，接受一个估计器对象作为参数
    def __init__(self, estimator):
        self.estimator = estimator

    # 拟合方法，接受输入数据X和目标y，以及可选的拟合参数
    def fit(self, X, y, **fit_params):
        # 处理路由参数，根据当前对象和方法名"fit"进行处理
        routed_params = process_routing(self, "fit", **fit_params)
        # 克隆估计器对象并拟合数据X和y，使用路由参数中的估计器拟合参数
        self.estimator_ = clone(self.estimator).fit(X, y, **routed_params.estimator.fit)

    # 获取元数据路由方法
    def get_metadata_routing(self):
        # 创建一个元数据路由对象，设置其所有者为当前类名
        router = MetadataRouter(owner=self.__class__.__name__).add(
            # 添加估计器和方法映射，调用者为"fit"，被调用者为"fit"
            estimator=self.estimator,
            method_mapping=MethodMapping().add(caller="fit", callee="fit"),
        )
        return router


# %%
# 根据上述说明，如果`my_weights`不应作为`sample_weight`传递给`MetaRegressor`，这是有效的用法：
reg = MetaRegressor(estimator=LinearRegression().set_fit_request(sample_weight=True))
reg.fit(X, y, sample_weight=my_weights)


# %%
# 现在假设我们进一步开发``MetaRegressor``，它现在还会*消耗*``sample_weight``：
class WeightedMetaRegressor(MetaEstimatorMixin, RegressorMixin, BaseEstimator):
    # 显示警告以提醒用户显式设置值为`.set_{method}_request(sample_weight={boolean})`
    __metadata_request__fit = {"sample_weight": metadata_routing.WARN}

    # 初始化方法，接受一个估计器对象作为参数
    def __init__(self, estimator):
        self.estimator = estimator

    # 拟合方法，接受输入数据X和目标y，以及可选的sample_weight和拟合参数
    def fit(self, X, y, sample_weight=None, **fit_params):
        # 处理路由参数，根据当前对象、方法名"fit"以及sample_weight进行处理
        routed_params = process_routing(
            self, "fit", sample_weight=sample_weight, **fit_params
        )
        # 检查元数据，确保sample_weight设置正确
        check_metadata(self, sample_weight=sample_weight)
        # 克隆估计器对象并拟合数据X和y，使用路由参数中的估计器拟合参数
        self.estimator_ = clone(self.estimator).fit(X, y, **routed_params.estimator.fit)

    # 获取元数据路由方法
    def get_metadata_routing(self):
        # 创建一个元数据路由对象，设置其所有者为当前类名
        router = (
            MetadataRouter(owner=self.__class__.__name__)
            .add_self_request(self)
            .add(
                # 添加估计器和方法映射，调用者为"fit"，被调用者为"fit"
                estimator=self.estimator,
                method_mapping=MethodMapping().add(caller="fit", callee="fit"),
            )
        )
        return router


# %%
# 上述实现与``MetaRegressor``几乎相同，并且由于在``__metadata_request__fit``中定义了默认请求值，因此在拟合时会引发警告。
with warnings.catch_warnings(record=True) as record:
    # 创建一个带有警告的WeightedMetaRegressor对象，使用LinearRegression设置fit请求为sample_weight=False
    WeightedMetaRegressor(
        estimator=LinearRegression().set_fit_request(sample_weight=False)
    ).fit(X, y, sample_weight=my_weights)
# 打印记录中的每个警告消息
for w in record:
    print(w.message)


# %%
# 当一个估计器消耗它以前没有消耗的元数据时，可以使用以下模式向用户发出警告。
class ExampleRegressor(RegressorMixin, BaseEstimator):
    # 在拟合过程中显示警告，提醒用户元数据的使用
    __metadata_request__fit = {"sample_weight": metadata_routing.WARN}
    # 定义一个类方法 `fit`，用于模型训练，接受输入数据 `X` 和标签 `y`
    def fit(self, X, y, sample_weight=None):
        # 调用 `check_metadata` 函数，验证模型元数据，可选参数 `sample_weight` 用于加权
        check_metadata(self, sample_weight=sample_weight)
        # 返回模型实例本身，表示训练完成后返回的模型对象
        return self

    # 定义一个类方法 `predict`，用于模型预测，接受输入数据 `X`
    def predict(self, X):
        # 返回一个全零数组，形状为输入数据 `X` 的长度
        return np.zeros(shape=(len(X)))
# 使用 `warnings` 模块捕获警告信息，并记录在 `record` 中
with warnings.catch_warnings(record=True) as record:
    # 创建 MetaRegressor 对象，使用 ExampleRegressor 作为估算器，调用 fit 方法进行拟合
    MetaRegressor(estimator=ExampleRegressor()).fit(X, y, sample_weight=my_weights)

# 遍历捕获到的所有警告信息并打印出警告的消息
for w in record:
    print(w.message)

# %%
# 最后，禁用元数据路由的配置标志：
set_config(enable_metadata_routing=False)

# %%
# 第三方开发和 scikit-learn 依赖
# ---------------------------------------------------
#
# 如上所示，类之间通过 :class:`~utils.metadata_routing.MetadataRequest` 和
# :class:`~utils.metadata_routing.MetadataRouter` 进行信息交互。强烈建议不这样做，
# 但是如果你严格要求一个兼容 scikit-learn 的估算器而不依赖于 scikit-learn 包，
# 则可以考虑打包与元数据路由相关的工具。如果满足以下所有条件，你无需修改代码：
#
# - 你的估算器继承自 :class:`~base.BaseEstimator`
# - 你的估算器方法（例如 `fit`）中消耗的参数在方法签名中明确定义，而不是使用 `*args` 或 `*kwargs`。
# - 你的估算器不将任何元数据路由到底层对象，即它不是一个 *router*。
```