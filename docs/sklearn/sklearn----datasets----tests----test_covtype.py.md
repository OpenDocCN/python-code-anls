# `D:\src\scipysrc\scikit-learn\sklearn\datasets\tests\test_covtype.py`

```
"""Test the covtype loader, if the data is available,
or if specifically requested via environment variable
(e.g. for CI jobs)."""

# 导入所需的库和模块
from functools import partial  # 导入partial函数，用于创建偏函数
import pytest  # 导入pytest测试框架

# 从sklearn.datasets.tests.test_common模块中导入check_return_X_y函数
from sklearn.datasets.tests.test_common import check_return_X_y


# 定义测试函数test_fetch，测试fetch_covtype_fxt函数的行为
def test_fetch(fetch_covtype_fxt, global_random_seed):
    # 调用fetch_covtype_fxt函数，获取数据集data1和data2，并确保乱序和随机种子一致
    data1 = fetch_covtype_fxt(shuffle=True, random_state=global_random_seed)
    data2 = fetch_covtype_fxt(shuffle=True, random_state=global_random_seed + 1)

    # 获取数据集的特征数据X1和X2，并进行形状断言
    X1, X2 = data1["data"], data2["data"]
    assert (581012, 54) == X1.shape
    assert X1.shape == X2.shape

    # 断言特征数据的总和相等
    assert X1.sum() == X2.sum()

    # 获取数据集的目标值y1和y2，并进行形状断言
    y1, y2 = data1["target"], data2["target"]
    assert (X1.shape[0],) == y1.shape
    assert (X1.shape[0],) == y2.shape

    # 检查数据集的描述信息是否以指定的前缀开头
    descr_prefix = ".. _covtype_dataset:"
    assert data1.DESCR.startswith(descr_prefix)
    assert data2.DESCR.startswith(descr_prefix)

    # 测试fetch_func偏函数返回数据的X_y选项
    fetch_func = partial(fetch_covtype_fxt)
    check_return_X_y(data1, fetch_func)


# 定义测试函数test_fetch_asframe，测试fetch_covtype_fxt函数的as_frame参数为True时的行为
def test_fetch_asframe(fetch_covtype_fxt):
    pytest.importorskip("pandas")  # 确保pandas库可用，否则跳过此测试

    # 调用fetch_covtype_fxt函数，获取数据集bunch，并进行相关断言
    bunch = fetch_covtype_fxt(as_frame=True)
    assert hasattr(bunch, "frame")
    frame = bunch.frame
    assert frame.shape == (581012, 55)
    assert bunch.data.shape == (581012, 54)
    assert bunch.target.shape == (581012,)

    # 检查数据框的列名是否包含预期的枚举名称
    column_names = set(frame.columns)
    assert set(f"Wilderness_Area_{i}" for i in range(4)) < column_names
    assert set(f"Soil_Type_{i}" for i in range(40)) < column_names


# 定义测试函数test_pandas_dependency_message，测试fetch_covtype_fxt函数的as_frame参数为True时缺少pandas库的行为
def test_pandas_dependency_message(fetch_covtype_fxt, hide_available_pandas):
    expected_msg = "fetch_covtype with as_frame=True requires pandas"
    # 断言调用fetch_covtype_fxt函数时，确实会抛出ImportError并包含特定的错误信息
    with pytest.raises(ImportError, match=expected_msg):
        fetch_covtype_fxt(as_frame=True)
```