# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_tags.py`

```
import pytest  # 导入 pytest 测试框架

from sklearn.base import BaseEstimator  # 导入基类 BaseEstimator
from sklearn.utils._tags import (  # 从 sklearn.utils._tags 中导入以下模块
    _DEFAULT_TAGS,  # 默认标签集合
    _safe_tags,  # 安全获取标签的函数
)


class NoTagsEstimator:  # 定义一个没有标签的估算器类
    pass


class MoreTagsEstimator:  # 定义一个具有更多标签的估算器类
    def _more_tags(self):
        return {"allow_nan": True}  # 返回一个包含 allow_nan 标签的字典


@pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器来定义测试参数
    "estimator, err_msg",  # 参数化测试的参数列表和错误消息
    [
        (BaseEstimator(), "The key xxx is not defined in _get_tags"),  # 测试基类估算器的错误情况
        (NoTagsEstimator(), "The key xxx is not defined in _DEFAULT_TAGS"),  # 测试没有标签估算器的错误情况
    ],
)
def test_safe_tags_error(estimator, err_msg):
    # 检查在不明确情况下 safe_tags 函数是否引发错误。
    with pytest.raises(ValueError, match=err_msg):  # 使用 pytest 的断言检查是否引发 ValueError 异常，并匹配错误消息
        _safe_tags(estimator, key="xxx")  # 调用 _safe_tags 函数，检查是否引发预期的错误


@pytest.mark.parametrize(  # 参数化装饰器定义第二个测试函数的参数
    "estimator, key, expected_results",  # 参数化测试的参数列表和期望结果
    [
        (NoTagsEstimator(), None, _DEFAULT_TAGS),  # 测试没有标签估算器的情况下的默认标签
        (NoTagsEstimator(), "allow_nan", _DEFAULT_TAGS["allow_nan"]),  # 测试没有标签估算器的 allow_nan 标签
        (MoreTagsEstimator(), None, {**_DEFAULT_TAGS, **{"allow_nan": True}}),  # 测试具有更多标签估算器的情况下的默认标签和额外的 allow_nan 标签
        (MoreTagsEstimator(), "allow_nan", True),  # 测试具有更多标签估算器的 allow_nan 标签
        (BaseEstimator(), None, _DEFAULT_TAGS),  # 测试基类估算器的默认标签
        (BaseEstimator(), "allow_nan", _DEFAULT_TAGS["allow_nan"]),  # 测试基类估算器的 allow_nan 标签
        (BaseEstimator(), "allow_nan", _DEFAULT_TAGS["allow_nan"]),  # 再次测试基类估算器的 allow_nan 标签，与上一行重复
    ],
)
def test_safe_tags_no_get_tags(estimator, key, expected_results):
    # 检查当估算器没有实现 _get_tags 方法时 _safe_tags 函数的行为
    assert _safe_tags(estimator, key=key) == expected_results  # 使用断言检查 _safe_tags 函数返回的结果是否与期望结果一致
```