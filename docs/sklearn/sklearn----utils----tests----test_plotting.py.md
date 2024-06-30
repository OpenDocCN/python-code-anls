# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_plotting.py`

```
# 导入所需的库
import numpy as np
import pytest

# 从 sklearn.utils._plotting 中导入特定函数
from sklearn.utils._plotting import _interval_max_min_ratio, _validate_score_name


# 定义一个空的函数 metric，用于测试覆盖率时不被覆盖
def metric():
    pass  # pragma: no cover


# 定义一个空的函数 neg_metric，用于测试覆盖率时不被覆盖
def neg_metric():
    pass  # pragma: no cover


# 使用 pytest.mark.parametrize 装饰器定义多组参数化测试
@pytest.mark.parametrize(
    "score_name, scoring, negate_score, expected_score_name",
    [
        ("accuracy", None, False, "accuracy"),  # 不改变名称
        (None, "accuracy", False, "Accuracy"),  # 将名称首字母大写
        (None, "accuracy", True, "Negative accuracy"),  # 添加 "Negative"
        (None, "neg_mean_absolute_error", False, "Negative mean absolute error"),  # 移除 "neg_"
        (None, "neg_mean_absolute_error", True, "Mean absolute error"),  # 移除 "neg_"
        ("MAE", "neg_mean_absolute_error", True, "MAE"),  # 保持 score_name 不变
        (None, None, False, "Score"),  # 默认名称
        (None, None, True, "Negative score"),  # 默认名称但是取反
        ("Some metric", metric, False, "Some metric"),  # 不改变名称
        ("Some metric", metric, True, "Some metric"),  # 不改变名称
        (None, metric, False, "Metric"),  # 默认名称
        (None, metric, True, "Negative metric"),  # 默认名称但是取反
        ("Some metric", neg_metric, False, "Some metric"),  # 不改变名称
        ("Some metric", neg_metric, True, "Some metric"),  # 不改变名称
        (None, neg_metric, False, "Negative metric"),  # 默认名称
        (None, neg_metric, True, "Metric"),  # 默认名称但是取反
    ],
)
def test_validate_score_name(score_name, scoring, negate_score, expected_score_name):
    """检查返回正确的评分名称。"""
    assert (
        _validate_score_name(score_name, scoring, negate_score) == expected_score_name
    )


# 在以下测试中，我们检查参数值区间的最大到最小比率
# 以决定在常见参数值范围上使用 5. 作为线性和对数刻度之间的决策阈值的良好启发。
@pytest.mark.parametrize(
    "data, lower_bound, upper_bound",
    [
        # 这样的范围可以使用对数或线性刻度清晰显示。
        (np.geomspace(0.1, 1, 5), 5, 6),
        # 在负对数刻度上检查比率仍然为正。
        (-np.geomspace(0.1, 1, 10), 7, 8),
        # 均匀间隔的参数值导致比率为 1。
        (np.linspace(0, 1, 5), 0.9, 1.1),
        # 这不是完全在对数刻度上均匀间隔，但为了可视化效果，我们会将其视为对数刻度。
        ([1, 2, 5, 10, 20, 50], 20, 40),
    ],
)
def test_inverval_max_min_ratio(data, lower_bound, upper_bound):
    assert lower_bound < _interval_max_min_ratio(data) < upper_bound
```