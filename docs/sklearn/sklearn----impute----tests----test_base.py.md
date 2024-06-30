# `D:\src\scipysrc\scikit-learn\sklearn\impute\tests\test_base.py`

```
# 导入需要的库
import numpy as np
import pytest

# 导入相关的类和函数
from sklearn.impute._base import _BaseImputer
from sklearn.impute._iterative import _assign_where
from sklearn.utils._mask import _get_mask
from sklearn.utils._testing import _convert_container, assert_allclose

# 定义一个 pytest 的 fixture，生成随机数据并含有缺失值的数组 X
@pytest.fixture
def data():
    X = np.random.randn(10, 2)
    X[::2] = np.nan  # 每隔一行设置为 NaN，模拟缺失值
    return X

# 定义一个不需要 fit 的 Imputer 类，直接调用 transform
class NoFitIndicatorImputer(_BaseImputer):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self._concatenate_indicator(X, self._transform_indicator(X))

# 定义一个不需要 transform 的 Imputer 类，直接调用 fit
class NoTransformIndicatorImputer(_BaseImputer):
    def fit(self, X, y=None):
        mask = _get_mask(X, value_to_mask=np.nan)  # 生成 NaN 的掩码
        super()._fit_indicator(mask)  # 调用父类方法设置指示器
        return self

    def transform(self, X, y=None):
        return self._concatenate_indicator(X, None)

# 定义一个需要 fit 前预先计算掩码的 Imputer 类
class NoPrecomputedMaskFit(_BaseImputer):
    def fit(self, X, y=None):
        self._fit_indicator(X)  # 调用方法设置指示器
        return self

    def transform(self, X):
        return self._concatenate_indicator(X, self._transform_indicator(X))

# 定义一个需要 fit 时预先计算掩码的 Imputer 类
class NoPrecomputedMaskTransform(_BaseImputer):
    def fit(self, X, y=None):
        mask = _get_mask(X, value_to_mask=np.nan)  # 生成 NaN 的掩码
        self._fit_indicator(mask)  # 调用方法设置指示器
        return self

    def transform(self, X):
        return self._concatenate_indicator(X, self._transform_indicator(X))

# 测试用例：验证 NoFitIndicatorImputer 类在没有调用 _fit_indicator 的情况下抛出 ValueError
def test_base_imputer_not_fit(data):
    imputer = NoFitIndicatorImputer(add_indicator=True)
    err_msg = "Make sure to call _fit_indicator before _transform_indicator"
    with pytest.raises(ValueError, match=err_msg):
        imputer.fit(data).transform(data)
    with pytest.raises(ValueError, match=err_msg):
        imputer.fit_transform(data)

# 测试用例：验证 NoTransformIndicatorImputer 类在没有调用 _transform_indicator 的情况下抛出 ValueError
def test_base_imputer_not_transform(data):
    imputer = NoTransformIndicatorImputer(add_indicator=True)
    err_msg = (
        "Call _fit_indicator and _transform_indicator in the imputer implementation"
    )
    with pytest.raises(ValueError, match=err_msg):
        imputer.fit(data).transform(data)
    with pytest.raises(ValueError, match=err_msg):
        imputer.fit_transform(data)

# 测试用例：验证 NoPrecomputedMaskFit 类在 precomputed 设置为 True 但输入数据不是掩码时抛出 ValueError
def test_base_no_precomputed_mask_fit(data):
    imputer = NoPrecomputedMaskFit(add_indicator=True)
    err_msg = "precomputed is True but the input data is not a mask"
    with pytest.raises(ValueError, match=err_msg):
        imputer.fit(data)
    with pytest.raises(ValueError, match=err_msg):
        imputer.fit_transform(data)

# 测试用例：验证 NoPrecomputedMaskTransform 类在 precomputed 设置为 True 但输入数据不是掩码时抛出 ValueError
def test_base_no_precomputed_mask_transform(data):
    imputer = NoPrecomputedMaskTransform(add_indicator=True)
    err_msg = "precomputed is True but the input data is not a mask"
    imputer.fit(data)
    with pytest.raises(ValueError, match=err_msg):
        imputer.transform(data)
    with pytest.raises(ValueError, match=err_msg):
        imputer.fit_transform(data)

# 参数化测试：测试私有辅助函数 `_assign_where` 的行为
@pytest.mark.parametrize("X1_type", ["array", "dataframe"])
def test_assign_where(X1_type):
    """Check the behaviour of the private helpers `_assign_where`."""
    rng = np.random.RandomState(0)  # 使用种子 0 初始化随机数生成器
    # 定义样本数和特征数
    n_samples, n_features = 10, 5
    # 使用随机数生成 n_samples x n_features 大小的数据，并通过指定的构造器名转换成 X1
    X1 = _convert_container(rng.randn(n_samples, n_features), constructor_name=X1_type)
    # 使用随机数生成 n_samples x n_features 大小的数据，赋值给 X2
    X2 = rng.randn(n_samples, n_features)
    # 使用随机数生成 0 或 1 的整数数组，大小为 n_samples x n_features，并转换为布尔类型的掩码
    mask = rng.randint(0, 2, size=(n_samples, n_features)).astype(bool)

    # 使用掩码将 X2 的值赋给 X1，仅对应掩码为 True 的位置
    _assign_where(X1, X2, mask)

    # 如果 X1 的类型为 "dataframe"，则将其转换为 numpy 数组
    if X1_type == "dataframe":
        X1 = X1.to_numpy()
    # 断言 X1 中掩码为 True 的位置的值与 X2 中相应位置的值接近（在数值上近似相等）
    assert_allclose(X1[mask], X2[mask])
```