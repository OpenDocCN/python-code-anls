# `D:\src\scipysrc\scikit-learn\sklearn\_loss\tests\test_link.py`

```
# 导入所需的库和模块
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

# 从sklearn._loss.link模块导入特定的类和函数
from sklearn._loss.link import (
    _LINKS,
    HalfLogitLink,
    Interval,
    MultinomialLogit,
    _inclusive_low_high,
)

# 从_LINKS字典中提取链接函数列表
LINK_FUNCTIONS = list(_LINKS.values())


def test_interval_raises():
    """测试当区间的上界大于下界时引发值错误的情况。"""
    with pytest.raises(
        ValueError, match="One must have low <= high; got low=1, high=0."
    ):
        Interval(1, 0, False, False)


@pytest.mark.parametrize(
    "interval",
    [
        Interval(0, 1, False, False),
        Interval(0, 1, False, True),
        Interval(0, 1, True, False),
        Interval(0, 1, True, True),
        Interval(-np.inf, np.inf, False, False),
        Interval(-np.inf, np.inf, False, True),
        Interval(-np.inf, np.inf, True, False),
        Interval(-np.inf, np.inf, True, True),
        Interval(-10, -1, False, False),
        Interval(-10, -1, False, True),
        Interval(-10, -1, True, False),
        Interval(-10, -1, True, True),
    ],
)
def test_is_in_range(interval):
    """确保区间的下界和上界总在区间内，用于linspace。"""
    # 获取区间的包容性低界和高界
    low, high = _inclusive_low_high(interval)

    # 生成一个包含10个元素的均匀分布的数组
    x = np.linspace(low, high, num=10)
    assert interval.includes(x)

    # x 包含下界
    assert interval.includes(np.r_[x, interval.low]) == interval.low_inclusive

    # x 包含上界
    assert interval.includes(np.r_[x, interval.high]) == interval.high_inclusive

    # x 同时包含下界和上界
    assert interval.includes(np.r_[x, interval.low, interval.high]) == (
        interval.low_inclusive and interval.high_inclusive
    )


@pytest.mark.parametrize("link", LINK_FUNCTIONS)
def test_link_inverse_identity(link, global_random_seed):
    """测试链接函数的逆函数应用后得到原始值的恒等性。"""
    # 使用全局随机种子创建随机数生成器
    rng = np.random.RandomState(global_random_seed)
    # 创建特定的链接函数实例
    link = link()
    n_samples, n_classes = 100, None
    # 由于在类LogitLink中，对于大正数x，术语expit(x)非常接近1，因此存在精度损失的情况，
    # 因此限制`raw_prediction`的值在-20到20之间。
    if link.is_multiclass:
        n_classes = 10
        raw_prediction = rng.uniform(low=-20, high=20, size=(n_samples, n_classes))
        if isinstance(link, MultinomialLogit):
            raw_prediction = link.symmetrize_raw_prediction(raw_prediction)
    elif isinstance(link, HalfLogitLink):
        raw_prediction = rng.uniform(low=-10, high=10, size=(n_samples))
    else:
        raw_prediction = rng.uniform(low=-20, high=20, size=(n_samples))

    # 断言链接函数应用逆函数后结果与原始输入值接近
    assert_allclose(link.link(link.inverse(raw_prediction)), raw_prediction)
    # 对逆函数应用链接函数后，结果应接近原始预测值
    y_pred = link.inverse(raw_prediction)
    assert_allclose(link.inverse(link.link(y_pred)), y_pred)


@pytest.mark.parametrize("link", LINK_FUNCTIONS)
def test_link_out_argument(link):
    """测试out参数被分配结果的情况。"""
    # 使用固定的随机种子创建随机数生成器
    rng = np.random.RandomState(42)
    # 创建特定的链接函数实例
    link = link()
    # 设置样本数和类数变量，初始类数为 None
    n_samples, n_classes = 100, None
    
    # 如果链接对象是多类别的，则设定类数为 10
    if link.is_multiclass:
        n_classes = 10
        # 生成服从正态分布的随机预测值矩阵，形状为 (n_samples, n_classes)
        raw_prediction = rng.normal(loc=0, scale=10, size=(n_samples, n_classes))
        # 如果链接对象是多项逻辑回归，对预测值进行对称化处理
        if isinstance(link, MultinomialLogit):
            raw_prediction = link.symmetrize_raw_prediction(raw_prediction)
    else:
        # 对于非多类别情况，生成在区间 [-10, 10] 内均匀分布的随机预测值矩阵，形状为 (n_samples,)
        raw_prediction = rng.uniform(low=-10, high=10, size=(n_samples,))
    
    # 使用链接对象的逆函数将原始预测值转换为预测标签，存储在 y_pred 中
    y_pred = link.inverse(raw_prediction, out=None)
    
    # 创建一个与 raw_prediction 形状相同的空数组 out
    out = np.empty_like(raw_prediction)
    
    # 再次使用链接对象的逆函数将 raw_prediction 转换为预测标签，存储在 y_pred_2 中，并将结果存储在 out 中
    y_pred_2 = link.inverse(raw_prediction, out=out)
    
    # 检查 y_pred 和 out 数组是否近似相等
    assert_allclose(y_pred, out)
    
    # 检查 out 和 y_pred_2 数组是否完全相等
    assert_array_equal(out, y_pred_2)
    
    # 检查 out 和 y_pred_2 是否共享内存
    assert np.shares_memory(out, y_pred_2)
    
    # 重新创建一个与 y_pred 形状相同的空数组 out
    out = np.empty_like(y_pred)
    
    # 使用链接对象的链接函数将预测标签 y_pred 转换为原始预测值，存储在 raw_prediction_2 中，并将结果存储在 out 中
    raw_prediction_2 = link.link(y_pred, out=out)
    
    # 检查 raw_prediction 和 out 数组是否近似相等
    assert_allclose(raw_prediction, out)
    
    # 检查 out 和 raw_prediction_2 数组是否完全相等
    assert_array_equal(out, raw_prediction_2)
    
    # 检查 out 和 raw_prediction_2 是否共享内存
    assert np.shares_memory(out, raw_prediction_2)
```