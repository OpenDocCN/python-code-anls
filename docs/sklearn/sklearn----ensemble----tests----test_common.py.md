# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\tests\test_common.py`

```
import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于单元测试

from sklearn.base import ClassifierMixin, clone, is_classifier  # 导入ClassifierMixin类和相关函数
from sklearn.datasets import (  # 导入数据集加载函数
    load_diabetes,
    load_iris,
    make_classification,
    make_regression,
)
from sklearn.ensemble import (  # 导入集成学习模型类
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.impute import SimpleImputer  # 导入数据缺失值处理类
from sklearn.linear_model import LinearRegression, LogisticRegression  # 导入线性回归和逻辑回归模型
from sklearn.pipeline import make_pipeline  # 导入构建管道的函数
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR  # 导入支持向量机模型

X, y = load_iris(return_X_y=True)  # 加载鸢尾花数据集作为示例特征和标签

X_r, y_r = load_diabetes(return_X_y=True)  # 加载糖尿病数据集作为示例特征和标签


@pytest.mark.parametrize(  # 使用pytest的参数化测试装饰器定义多组参数化测试
    "X, y, estimator",  # 参数化变量名及其对应的参数
    [
        (
            *make_classification(n_samples=10),  # 生成二分类样本数据
            StackingClassifier(  # 创建堆叠分类器
                estimators=[
                    ("lr", LogisticRegression()),  # 逻辑回归作为基础分类器
                    ("svm", LinearSVC()),  # 线性支持向量分类器作为基础分类器
                    ("rf", RandomForestClassifier(n_estimators=5, max_depth=3)),  # 随机森林分类器作为基础分类器
                ],
                cv=2,  # 交叉验证折数
            ),
        ),
        (
            *make_classification(n_samples=10),  # 生成二分类样本数据
            VotingClassifier(  # 创建投票分类器
                estimators=[
                    ("lr", LogisticRegression()),  # 逻辑回归作为基础分类器
                    ("svm", LinearSVC()),  # 线性支持向量分类器作为基础分类器
                    ("rf", RandomForestClassifier(n_estimators=5, max_depth=3)),  # 随机森林分类器作为基础分类器
                ]
            ),
        ),
        (
            *make_regression(n_samples=10),  # 生成回归样本数据
            StackingRegressor(  # 创建堆叠回归器
                estimators=[
                    ("lr", LinearRegression()),  # 线性回归作为基础回归器
                    ("svm", LinearSVR()),  # 线性支持向量回归作为基础回归器
                    ("rf", RandomForestRegressor(n_estimators=5, max_depth=3)),  # 随机森林回归器作为基础回归器
                ],
                cv=2,  # 交叉验证折数
            ),
        ),
        (
            *make_regression(n_samples=10),  # 生成回归样本数据
            VotingRegressor(  # 创建投票回归器
                estimators=[
                    ("lr", LinearRegression()),  # 线性回归作为基础回归器
                    ("svm", LinearSVR()),  # 线性支持向量回归作为基础回归器
                    ("rf", RandomForestRegressor(n_estimators=5, max_depth=3)),  # 随机森林回归器作为基础回归器
                ]
            ),
        ),
    ],
    ids=[  # 参数化测试的标识符列表
        "stacking-classifier",
        "voting-classifier",
        "stacking-regressor",
        "voting-regressor",
    ],
)
def test_ensemble_heterogeneous_estimators_behavior(X, y, estimator):
    # 检查`estimators`、`estimators_`、`named_estimators`、`named_estimators_`
    # 在所有集成类中以及使用`set_params()`时的一致性行为。

    # 在拟合之前的检查
    assert "svm" in estimator.named_estimators  # 检查是否在命名的基础估计器中包含"svm"
    assert estimator.named_estimators.svm is estimator.estimators[1][1]  # 检查命名的基础估计器"svm"是否与估计器列表中的相同
    assert estimator.named_estimators.svm is estimator.named_estimators["svm"]  # 检查命名的基础估计器"svm"是否与命名字典中的相同

    # 检查拟合后的属性
    estimator.fit(X, y)  # 拟合模型
    assert len(estimator.named_estimators) == 3  # 检查命名的基础估计器的数量是否为3
    assert len(estimator.named_estimators_) == 3  # 检查拟合后的命名的基础估计器的数量是否为3
    assert sorted(list(estimator.named_estimators_.keys())) == sorted(  # 检查命名的基础估计器的键是否为["lr", "svm", "rf"]
        ["lr", "svm", "rf"]
    )
    # 检查 set_params() 方法是否会添加新的属性
    estimator_new_params = clone(estimator)
    # 根据 estimator 的类型选择 SVM 分类器或回归器
    svm_estimator = SVC() if is_classifier(estimator) else SVR()
    # 设置 estimator_new_params 的 svm 参数，并训练模型
    estimator_new_params.set_params(svm=svm_estimator).fit(X, y)
    # 断言 estimator_new_params 中没有名为 "svm" 的属性
    assert not hasattr(estimator_new_params, "svm")
    # 断言 estimator_new_params 中 lr 子估计器的参数与原 estimator 中相同
    assert (
        estimator_new_params.named_estimators.lr.get_params()
        == estimator.named_estimators.lr.get_params()
    )
    # 断言 estimator_new_params 中 rf 子估计器的参数与原 estimator 中相同
    assert (
        estimator_new_params.named_estimators.rf.get_params()
        == estimator.named_estimators.rf.get_params()
    )

    # 检查设置和删除估计器时的行为
    estimator_dropped = clone(estimator)
    # 设置 estimator_dropped 的 svm 参数为 "drop"
    estimator_dropped.set_params(svm="drop")
    # 训练 estimator_dropped 模型
    estimator_dropped.fit(X, y)
    # 断言 estimator_dropped 中有三个命名估计器
    assert len(estimator_dropped.named_estimators) == 3
    # 断言 estimator_dropped 中 svm 的值为 "drop"
    assert estimator_dropped.named_estimators.svm == "drop"
    # 断言 estimator_dropped 中的命名估计器数量为三
    assert len(estimator_dropped.named_estimators_) == 3
    # 断言 estimator_dropped 中的命名估计器的键与预期的相同
    assert sorted(list(estimator_dropped.named_estimators_.keys())) == sorted(
        ["lr", "svm", "rf"]
    )
    # 遍历 estimator_dropped 的命名估计器，并检查其是否与原 estimator 中的 svm 子估计器类型不同
    for sub_est in estimator_dropped.named_estimators_:
        # 检查子估计器的对应关系是否正确
        assert not isinstance(sub_est, type(estimator.named_estimators.svm))

    # 检查是否可以设置基础分类器的参数
    # 设置 svm 子估计器的 C 参数为 10.0
    estimator.set_params(svm__C=10.0)
    # 设置 rf 子估计器的 max_depth 参数为 5
    estimator.set_params(rf__max_depth=5)
    # 断言 estimator 中 svm 子估计器的 C 参数与其参数设置相同
    assert (
        estimator.get_params()["svm__C"]
        == estimator.get_params()["svm"].get_params()["C"]
    )
    # 断言 estimator 中 rf 子估计器的 max_depth 参数与其参数设置相同
    assert (
        estimator.get_params()["rf__max_depth"]
        == estimator.get_params()["rf"].get_params()["max_depth"]
    )
# 使用 pytest.mark.parametrize 装饰器为测试函数 test_ensemble_heterogeneous_estimators_type 添加参数化测试
@pytest.mark.parametrize(
    "Ensemble",
    [VotingClassifier, StackingRegressor, VotingRegressor],
)
def test_ensemble_heterogeneous_estimators_type(Ensemble):
    # 检查集成器在验证过程中是否会失败，如果底层估计器不是相同类型（即分类器或回归器）
    # StackingClassifier 可以有一个底层回归器，因此不进行检查
    if issubclass(Ensemble, ClassifierMixin):
        X, y = make_classification(n_samples=10)
        estimators = [("lr", LinearRegression())]
        ensemble_type = "classifier"
    else:
        X, y = make_regression(n_samples=10)
        estimators = [("lr", LogisticRegression())]
        ensemble_type = "regressor"
    ensemble = Ensemble(estimators=estimators)

    err_msg = "should be a {}".format(ensemble_type)
    # 使用 pytest.raises 检查是否会引发 ValueError 异常，并匹配错误消息
    with pytest.raises(ValueError, match=err_msg):
        ensemble.fit(X, y)


# 使用 pytest.mark.parametrize 装饰器为测试函数 test_ensemble_heterogeneous_estimators_name_validation 添加参数化测试
@pytest.mark.parametrize(
    "X, y, Ensemble",
    [
        (*make_classification(n_samples=10), StackingClassifier),
        (*make_classification(n_samples=10), VotingClassifier),
        (*make_regression(n_samples=10), StackingRegressor),
        (*make_regression(n_samples=10), VotingRegressor),
    ],
)
def test_ensemble_heterogeneous_estimators_name_validation(X, y, Ensemble):
    # 当名称包含双下划线时引发错误
    if issubclass(Ensemble, ClassifierMixin):
        estimators = [("lr__", LogisticRegression())]
    else:
        estimators = [("lr__", LinearRegression())]
    ensemble = Ensemble(estimators=estimators)

    err_msg = r"Estimator names must not contain __: got \['lr__'\]"
    # 使用 pytest.raises 检查是否会引发 ValueError 异常，并匹配错误消息
    with pytest.raises(ValueError, match=err_msg):
        ensemble.fit(X, y)

    # 当名称不唯一时引发错误
    if issubclass(Ensemble, ClassifierMixin):
        estimators = [("lr", LogisticRegression()), ("lr", LogisticRegression())]
    else:
        estimators = [("lr", LinearRegression()), ("lr", LinearRegression())]
    ensemble = Ensemble(estimators=estimators)

    err_msg = r"Names provided are not unique: \['lr', 'lr'\]"
    # 使用 pytest.raises 检查是否会引发 ValueError 异常，并匹配错误消息
    with pytest.raises(ValueError, match=err_msg):
        ensemble.fit(X, y)

    # 当名称与参数冲突时引发错误
    if issubclass(Ensemble, ClassifierMixin):
        estimators = [("estimators", LogisticRegression())]
    else:
        estimators = [("estimators", LinearRegression())]
    ensemble = Ensemble(estimators=estimators)

    err_msg = "Estimator names conflict with constructor arguments"
    # 使用 pytest.raises 检查是否会引发 ValueError 异常，并匹配错误消息
    with pytest.raises(ValueError, match=err_msg):
        ensemble.fit(X, y)
    # 创建一个包含四个元组的列表，每个元组包含以下内容：
    # 1. 通过 make_classification 生成的分类问题数据集，包括样本和特征
    # 2. 一个 StackingClassifier 实例，包含一个 LogisticRegression 作为基础分类器
    # 3. 通过 make_classification 生成的另一个分类问题数据集
    # 4. 一个 VotingClassifier 实例，包含一个 LogisticRegression 作为基础分类器

    [
        (
            *make_classification(n_samples=10),
            StackingClassifier(estimators=[("lr", LogisticRegression())]),
        ),
        (
            *make_classification(n_samples=10),
            VotingClassifier(estimators=[("lr", LogisticRegression())]),
        ),
        (
            *make_regression(n_samples=10),
            StackingRegressor(estimators=[("lr", LinearRegression())]),
        ),
        (
            *make_regression(n_samples=10),
            VotingRegressor(estimators=[("lr", LinearRegression())]),
        ),
    ],

    # 提供用于标识每个元组的名称列表，对应于前面列表中的四个分类器/回归器组合
    ids=[
        "stacking-classifier",   # 第一个元组的标识符，用于识别堆叠分类器
        "voting-classifier",     # 第二个元组的标识符，用于识别投票分类器
        "stacking-regressor",    # 第三个元组的标识符，用于识别堆叠回归器
        "voting-regressor",      # 第四个元组的标识符，用于识别投票回归器
    ],
# 定义一个测试函数，用于测试异构集成学习中所有估计器都被丢弃时是否能正确抛出异常。
def test_ensemble_heterogeneous_estimators_all_dropped(X, y, estimator):
    # 设置估计器的参数，将逻辑回归估计器设为"drop"
    estimator.set_params(lr="drop")
    # 使用 pytest 检查是否引发 ValueError 异常，并匹配指定的错误消息。
    with pytest.raises(ValueError, match="All estimators are dropped."):
        estimator.fit(X, y)


# 使用 pytest 的 parametrize 装饰器，定义多组参数化测试数据集和估计器类型
@pytest.mark.parametrize(
    "Ensemble, Estimator, X, y",
    [
        (StackingClassifier, LogisticRegression, X, y),  # 测试堆叠分类器
        (StackingRegressor, LinearRegression, X_r, y_r),  # 测试堆叠回归器
        (VotingClassifier, LogisticRegression, X, y),  # 测试投票分类器
        (VotingRegressor, LinearRegression, X_r, y_r),  # 测试投票回归器
    ],
)
# FIXME: 一旦能够构造元估计器实例，应该将此测试移到 `estimator_checks` 中。
# 定义一个测试函数，用于验证异构集成学习中对缺失值的支持情况。
def test_heterogeneous_ensemble_support_missing_values(Ensemble, Estimator, X, y):
    # 检查投票和堆叠预测器是否将缺失值验证委托给底层估计器。
    X = X.copy()
    # 随机生成掩码数组，模拟缺失值的存在
    mask = np.random.choice([1, 0], X.shape, p=[0.1, 0.9]).astype(bool)
    X[mask] = np.nan
    # 创建管道，使用简单填充器填充缺失值，并添加指定类型的估计器
    pipe = make_pipeline(SimpleImputer(), Estimator())
    # 创建异构集成对象，包含两个管道作为估计器
    ensemble = Ensemble(estimators=[("pipe1", pipe), ("pipe2", pipe)])
    # 对数据进行拟合和评分
    ensemble.fit(X, y).score(X, y)
```