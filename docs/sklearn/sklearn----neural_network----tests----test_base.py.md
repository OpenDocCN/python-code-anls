# `D:\src\scipysrc\scikit-learn\sklearn\neural_network\tests\test_base.py`

```
import numpy as np
import pytest

# 从 sklearn.neural_network._base 模块中导入 binary_log_loss 和 log_loss 函数


def test_binary_log_loss_1_prob_finite():
    # 定义真实标签 y_true 和预测概率 y_prob 的 numpy 数组
    y_true = np.array([[0, 0, 1]]).T
    y_prob = np.array([[0.9, 1.0, 1.0]]).T

    # 计算二元交叉熵损失
    loss = binary_log_loss(y_true, y_prob)
    # 断言损失值为有限数值
    assert np.isfinite(loss)


@pytest.mark.parametrize(
    "y_true, y_prob",
    [
        (
            np.array([[1, 0, 0], [0, 1, 0]]),
            np.array([[0.0, 1.0, 0.0], [0.9, 0.05, 0.05]]),
        ),
        (np.array([[0, 0, 1]]).T, np.array([[0.9, 1.0, 1.0]]).T),
    ],
)
def test_log_loss_1_prob_finite(y_true, y_prob):
    # 定义真实标签 y_true 和预测概率 y_prob 的参数化测试数据

    # 计算多类别交叉熵损失
    loss = log_loss(y_true, y_prob)
    # 断言损失值为有限数值
    assert np.isfinite(loss)
```