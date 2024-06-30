# `D:\src\scipysrc\scikit-learn\sklearn\datasets\tests\test_california_housing.py`

```
"""Test the california_housing loader, if the data is available,
or if specifically requested via environment variable
(e.g. for CI jobs)."""

# 从 functools 模块导入 partial 函数，用于创建偏函数
from functools import partial

# 导入 pytest 测试框架
import pytest

# 从 sklearn.datasets.tests.test_common 中导入 check_return_X_y 函数
from sklearn.datasets.tests.test_common import check_return_X_y


# 测试函数，测试 fetch_california_housing_fxt 函数的返回结果
def test_fetch(fetch_california_housing_fxt):
    # 调用 fetch_california_housing_fxt 函数获取数据集
    data = fetch_california_housing_fxt()
    # 断言数据集的数据部分形状为 (20640, 8)
    assert (20640, 8) == data.data.shape
    # 断言数据集的目标部分形状为 (20640,)
    assert (20640,) == data.target.shape
    # 断言数据集描述信息以 ".. _california_housing_dataset:" 开头
    assert data.DESCR.startswith(".. _california_housing_dataset:")

    # 测试返回 return_X_y 选项
    # 创建 fetch_func 为 fetch_california_housing_fxt 的偏函数
    fetch_func = partial(fetch_california_housing_fxt)
    # 调用 check_return_X_y 函数验证数据集是否符合预期
    check_return_X_y(data, fetch_func)


# 测试函数，测试 fetch_california_housing_fxt 函数返回 DataFrame 的情况
def test_fetch_asframe(fetch_california_housing_fxt):
    # 尝试导入 pandas 库，如果不存在则跳过测试
    pd = pytest.importorskip("pandas")
    # 调用 fetch_california_housing_fxt 函数获取数据集，并设定 as_frame=True
    bunch = fetch_california_housing_fxt(as_frame=True)
    # 获取数据集的 DataFrame 形式
    frame = bunch.frame
    # 断言数据集对象含有 frame 属性
    assert hasattr(bunch, "frame") is True
    # 断言 DataFrame 的形状为 (20640, 9)
    assert frame.shape == (20640, 9)
    # 断言 bunch.data 是 pandas 的 DataFrame 类型
    assert isinstance(bunch.data, pd.DataFrame)
    # 断言 bunch.target 是 pandas 的 Series 类型
    assert isinstance(bunch.target, pd.Series)


# 测试函数，测试 pandas 依赖性消息
def test_pandas_dependency_message(fetch_california_housing_fxt, hide_available_pandas):
    # 检查 pandas 是否被延迟导入，并在缺少 pandas 时引发信息性错误消息
    expected_msg = "fetch_california_housing with as_frame=True requires pandas"
    with pytest.raises(ImportError, match=expected_msg):
        fetch_california_housing_fxt(as_frame=True)
```