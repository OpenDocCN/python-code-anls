# `D:\src\scipysrc\scikit-learn\sklearn\covariance\tests\test_elliptic_envelope.py`

```
"""
Testing for Elliptic Envelope algorithm (sklearn.covariance.elliptic_envelope).
"""

import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于测试

from sklearn.covariance import EllipticEnvelope  # 导入EllipticEnvelope异常检测模型
from sklearn.exceptions import NotFittedError  # 导入NotFittedError异常类
from sklearn.utils._testing import (  # 导入测试辅助函数
    assert_almost_equal,  # 检查数值近似相等
    assert_array_almost_equal,  # 检查数组近似相等
    assert_array_equal,  # 检查数组完全相等
)


def test_elliptic_envelope(global_random_seed):
    rnd = np.random.RandomState(global_random_seed)  # 使用全局随机种子创建随机数生成器
    X = rnd.randn(100, 10)  # 生成100x10的随机数数组
    clf = EllipticEnvelope(contamination=0.1)  # 创建EllipticEnvelope对象，设置污染度为0.1
    with pytest.raises(NotFittedError):  # 断言捕获到NotFittedError异常
        clf.predict(X)  # 调用未拟合的模型进行预测
    with pytest.raises(NotFittedError):  # 断言捕获到NotFittedError异常
        clf.decision_function(X)  # 调用未拟合的模型进行决策函数计算
    clf.fit(X)  # 对数据进行拟合
    y_pred = clf.predict(X)  # 对数据进行预测
    scores = clf.score_samples(X)  # 计算数据样本的得分
    decisions = clf.decision_function(X)  # 计算数据样本的决策函数值

    assert_array_almost_equal(scores, -clf.mahalanobis(X))  # 断言scores与mahalanobis距离的负值近似相等
    assert_array_almost_equal(clf.mahalanobis(X), clf.dist_)  # 断言mahalanobis距离与dist_属性近似相等
    assert_almost_equal(
        clf.score(X, np.ones(100)), (100 - y_pred[y_pred == -1].size) / 100.0
    )  # 断言模型得分与真实标签1的比例近似相等
    assert sum(y_pred == -1) == sum(decisions < 0)  # 断言预测为异常的数量与决策函数小于0的数量相等


def test_score_samples():
    X_train = [[1, 1], [1, 2], [2, 1]]  # 训练数据
    clf1 = EllipticEnvelope(contamination=0.2).fit(X_train)  # 创建并拟合EllipticEnvelope对象，设置污染度为0.2
    clf2 = EllipticEnvelope().fit(X_train)  # 创建并拟合EllipticEnvelope对象，使用默认参数

    assert_array_equal(
        clf1.score_samples([[2.0, 2.0]]),
        clf1.decision_function([[2.0, 2.0]]) + clf1.offset_,
    )  # 断言两种方法计算的样本得分数组完全相等
    assert_array_equal(
        clf2.score_samples([[2.0, 2.0]]),
        clf2.decision_function([[2.0, 2.0]]) + clf2.offset_,
    )  # 断言两种方法计算的样本得分数组完全相等
    assert_array_equal(
        clf1.score_samples([[2.0, 2.0]]), clf2.score_samples([[2.0, 2.0]])
    )  # 断言两种方法计算的样本得分数组完全相等
```