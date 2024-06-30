# `D:\src\scipysrc\scikit-learn\sklearn\datasets\tests\test_olivetti_faces.py`

```
"""Test Olivetti faces fetcher, if the data is available,
or if specifically requested via environment variable
(e.g. for CI jobs)."""

# 导入所需的库
import numpy as np

# 导入测试所需的函数和类
from sklearn.datasets.tests.test_common import check_return_X_y
from sklearn.utils import Bunch
from sklearn.utils._testing import assert_array_equal


# 定义测试函数，用于测试 fetch_olivetti_faces_fxt 函数
def test_olivetti_faces(fetch_olivetti_faces_fxt):
    # 调用 fetch_olivetti_faces_fxt 函数获取数据集
    data = fetch_olivetti_faces_fxt(shuffle=True, random_state=0)

    # 断言返回的数据是 Bunch 类型
    assert isinstance(data, Bunch)

    # 断言数据集包含以下键值对应的属性
    for expected_keys in ("data", "images", "target", "DESCR"):
        assert expected_keys in data.keys()

    # 断言数据集中的 data 数组形状为 (400, 4096)
    assert data.data.shape == (400, 4096)

    # 断言数据集中的 images 数组形状为 (400, 64, 64)
    assert data.images.shape == (400, 64, 64)

    # 断言数据集中的 target 数组形状为 (400,)
    assert data.target.shape == (400)

    # 断言 target 数组中的值是从 0 到 39 的整数，并且没有重复
    assert_array_equal(np.unique(np.sort(data.target)), np.arange(40))

    # 断言数据集的 DESCR 字符串以指定的开头文本开头
    assert data.DESCR.startswith(".. _olivetti_faces_dataset:")

    # 测试返回 X 和 y 的选项
    check_return_X_y(data, fetch_olivetti_faces_fxt)
```