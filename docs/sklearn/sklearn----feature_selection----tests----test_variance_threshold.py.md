# `D:\src\scipysrc\scikit-learn\sklearn\feature_selection\tests\test_variance_threshold.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 库，用于编写和运行测试用例

from sklearn.feature_selection import VarianceThreshold  # 从 sklearn.feature_selection 模块导入 VarianceThreshold 类
from sklearn.utils._testing import assert_array_equal  # 从 sklearn.utils._testing 模块导入 assert_array_equal 函数
from sklearn.utils.fixes import BSR_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS  # 从 sklearn.utils.fixes 模块导入容器列表

data = [[0, 1, 2, 3, 4], [0, 2, 2, 3, 5], [1, 1, 2, 4, 0]]  # 定义一个二维列表 data

data2 = [[-0.13725701]] * 10  # 定义一个包含相同值的二维列表 data2，重复10次


@pytest.mark.parametrize(
    "sparse_container", [None] + BSR_CONTAINERS + CSC_CONTAINERS + CSR_CONTAINERS
)
def test_zero_variance(sparse_container):
    # 测试使用默认设置的 VarianceThreshold，即零方差情况
    X = data if sparse_container is None else sparse_container(data)
    sel = VarianceThreshold().fit(X)
    assert_array_equal([0, 1, 3, 4], sel.get_support(indices=True))


def test_zero_variance_value_error():
    # 测试使用默认设置的 VarianceThreshold，即零方差情况下的错误情况
    with pytest.raises(ValueError):
        VarianceThreshold().fit([[0, 1, 2, 3]])
    with pytest.raises(ValueError):
        VarianceThreshold().fit([[0, 1], [0, 1]])


@pytest.mark.parametrize("sparse_container", [None] + CSR_CONTAINERS)
def test_variance_threshold(sparse_container):
    # 测试使用自定义方差阈值的 VarianceThreshold
    X = data if sparse_container is None else sparse_container(data)
    X = VarianceThreshold(threshold=0.4).fit_transform(X)
    assert (len(data), 1) == X.shape


@pytest.mark.skipif(
    np.var(data2) == 0,
    reason=(
        "This test is not valid for this platform, "
        "as it relies on numerical instabilities."
    ),
)
@pytest.mark.parametrize(
    "sparse_container", [None] + BSR_CONTAINERS + CSC_CONTAINERS + CSR_CONTAINERS
)
def test_zero_variance_floating_point_error(sparse_container):
    # 测试 VarianceThreshold(0.0).fit 是否消除每个样本中具有相同值的特征，
    # 即使浮点数误差导致 np.var 不为 0
    # 参见 Issue #13691
    X = data2 if sparse_container is None else sparse_container(data2)
    msg = "No feature in X meets the variance threshold 0.00000"
    with pytest.raises(ValueError, match=msg):
        VarianceThreshold().fit(X)


@pytest.mark.parametrize(
    "sparse_container", [None] + BSR_CONTAINERS + CSC_CONTAINERS + CSR_CONTAINERS
)
def test_variance_nan(sparse_container):
    arr = np.array(data, dtype=np.float64)
    # 添加单个 NaN 值，该特征应该仍然包含在内
    arr[0, 0] = np.nan
    # 将特征中的所有值设置为 NaN，该特征应该被拒绝
    arr[:, 1] = np.nan

    X = arr if sparse_container is None else sparse_container(arr)
    sel = VarianceThreshold().fit(X)
    assert_array_equal([0, 3, 4], sel.get_support(indices=True))
```