# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\tests\test_base.py`

```
"""
Testing for the base module (sklearn.ensemble.base).
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 从 collections 模块中导入 OrderedDict 类
from collections import OrderedDict

# 导入 numpy 库，并将其命名为 np
import numpy as np

# 从 sklearn.datasets 中导入 load_iris 函数
from sklearn.datasets import load_iris

# 从 sklearn.discriminant_analysis 中导入 LinearDiscriminantAnalysis 类
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 从 sklearn.ensemble 中导入 BaggingClassifier 类
from sklearn.ensemble import BaggingClassifier

# 从 sklearn.ensemble._base 中导入 _set_random_states 函数
from sklearn.ensemble._base import _set_random_states

# 从 sklearn.feature_selection 中导入 SelectFromModel 类
from sklearn.feature_selection import SelectFromModel

# 从 sklearn.linear_model 中导入 Perceptron 类
from sklearn.linear_model import Perceptron

# 从 sklearn.pipeline 中导入 Pipeline 类
from sklearn.pipeline import Pipeline

# 定义测试函数 test_base
def test_base():
    # 创建 BaggingClassifier 实例 ensemble，使用 Perceptron 作为基评估器，设置 n_estimators 为 3
    ensemble = BaggingClassifier(
        estimator=Perceptron(random_state=None), n_estimators=3
    )

    # 载入鸢尾花数据集 iris
    iris = load_iris()

    # 使用 ensemble 拟合 iris 数据集
    ensemble.fit(iris.data, iris.target)

    # 清空 ensemble.estimators_ 列表并手动创建评估器
    ensemble.estimators_ = []

    # 调用 _make_estimator 方法，未指定 random_state
    ensemble._make_estimator()

    # 创建一个随机状态为 3 的 np.random.RandomState 实例
    random_state = np.random.RandomState(3)

    # 使用指定的 random_state 创建一个评估器
    ensemble._make_estimator(random_state=random_state)

    # 再次使用相同的 random_state 创建一个评估器
    ensemble._make_estimator(random_state=random_state)

    # 调用 _make_estimator 方法，设置 append=False
    ensemble._make_estimator(append=False)

    # 断言 ensemble 中评估器的数量为 3
    assert 3 == len(ensemble)

    # 断言 ensemble.estimators_ 中评估器的数量为 3
    assert 3 == len(ensemble.estimators_)

    # 断言 ensemble 的第一个评估器是 Perceptron 类型
    assert isinstance(ensemble[0], Perceptron)

    # 断言 ensemble 的第一个评估器的 random_state 为 None
    assert ensemble[0].random_state is None

    # 断言 ensemble 的第二个评估器的 random_state 是整数类型
    assert isinstance(ensemble[1].random_state, int)

    # 断言 ensemble 的第三个评估器的 random_state 是整数类型
    assert isinstance(ensemble[2].random_state, int)

    # 断言 ensemble 的第二个评估器和第三个评估器的 random_state 不相等
    assert ensemble[1].random_state != ensemble[2].random_state

    # 创建一个使用 np.int32(3) 设置 n_estimators 的 BaggingClassifier 实例 np_int_ensemble
    np_int_ensemble = BaggingClassifier(
        estimator=Perceptron(), n_estimators=np.int32(3)
    )

    # 使用 np_int_ensemble 拟合 iris 数据集
    np_int_ensemble.fit(iris.data, iris.target)


# 定义测试函数 test_set_random_states
def test_set_random_states():
    # 对 LinearDiscriminantAnalysis 实例调用 _set_random_states 方法，设置 random_state 为 17
    _set_random_states(LinearDiscriminantAnalysis(), random_state=17)

    # 创建一个 random_state 为 None 的 Perceptron 实例 clf1
    clf1 = Perceptron(random_state=None)

    # 断言 clf1 的 random_state 为 None
    assert clf1.random_state is None

    # 对 clf1 调用 _set_random_states 方法，设置 random_state 为 None
    # 断言 clf1 的 random_state 已经被设置为整数类型
    _set_random_states(clf1, None)
    assert isinstance(clf1.random_state, int)

    # 对 clf1 调用 _set_random_states 方法，设置 random_state 为 3
    # 断言 clf1 的 random_state 已经被设置为整数类型
    _set_random_states(clf1, 3)
    assert isinstance(clf1.random_state, int)

    # 创建一个 random_state 为 None 的 Perceptron 实例 clf2
    clf2 = Perceptron(random_state=None)

    # 对 clf2 调用 _set_random_states 方法，设置 random_state 为 3
    # 断言 clf1 和 clf2 的 random_state 相等
    _set_random_states(clf2, 3)
    assert clf1.random_state == clf2.random_state

    # 定义一个函数 make_steps，返回一个包含两个步骤的列表
    def make_steps():
        return [
            ("sel", SelectFromModel(Perceptron(random_state=None))),
            ("clf", Perceptron(random_state=None)),
        ]

    # 使用 make_steps 创建 Pipeline 实例 est1
    est1 = Pipeline(make_steps())

    # 对 est1 调用 _set_random_states 方法，设置 random_state 为 3
    # 断言 est1 中第一个步骤的评估器（即 SelectFromModel 中的 Perceptron）的 random_state 是整数类型
    assert isinstance(est1.steps[0][1].estimator.random_state, int)

    # 断言 est1 中第二个步骤的评估器（即 Pipeline 中的第二个 Perceptron）的 random_state 是整数类型
    assert isinstance(est1.steps[1][1].random_state, int)

    # 断言 est1 中第一个步骤的评估器的 random_state 不等于 est1 中第二个步骤的评估器的 random_state
    assert (
        est1.get_params()["sel__estimator__random_state"]
        != est1.get_params()["clf__random_state"]
    )
    class AlphaParamPipeline(Pipeline):
        # 定义 AlphaParamPipeline 类，继承自 Pipeline 类
        def get_params(self, *args, **kwargs):
            # 重写 get_params 方法，接受任意位置参数和关键字参数
            params = Pipeline.get_params(self, *args, **kwargs).items()
            # 调用父类 Pipeline 的 get_params 方法，并将其结果转换为字典项
            return OrderedDict(sorted(params))
            # 返回排序后的有序字典

    class RevParamPipeline(Pipeline):
        # 定义 RevParamPipeline 类，继承自 Pipeline 类
        def get_params(self, *args, **kwargs):
            # 重写 get_params 方法，接受任意位置参数和关键字参数
            params = Pipeline.get_params(self, *args, **kwargs).items()
            # 调用父类 Pipeline 的 get_params 方法，并将其结果转换为字典项
            return OrderedDict(sorted(params, reverse=True))
            # 返回排序后的逆序有序字典

    for cls in [AlphaParamPipeline, RevParamPipeline]:
        # 对于 AlphaParamPipeline 和 RevParamPipeline 类进行迭代
        est2 = cls(make_steps())
        # 创建一个 cls 类的实例 est2，使用 make_steps() 函数返回的步骤作为参数
        _set_random_states(est2, 3)
        # 调用 _set_random_states 函数，将 est2 实例和数字 3 作为参数传入
        assert (
            est1.get_params()["sel__estimator__random_state"]
            == est2.get_params()["sel__estimator__random_state"]
        )
        # 断言检查 est1 实例和 est2 实例的 sel__estimator__random_state 参数是否相同
        assert (
            est1.get_params()["clf__random_state"]
            == est2.get_params()["clf__random_state"]
        )
        # 断言检查 est1 实例和 est2 实例的 clf__random_state 参数是否相同
```