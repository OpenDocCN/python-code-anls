# `D:\src\scipysrc\scikit-learn\sklearn\cluster\tests\test_feature_agglomeration.py`

```
"""
Tests for sklearn.cluster._feature_agglomeration
"""

# Authors: Sergul Aydore 2017

# 导入警告模块
import warnings

# 导入NumPy库，并使用别名np
import numpy as np

# 导入pytest测试框架
import pytest

# 导入数组相等断言函数
from numpy.testing import assert_array_equal

# 导入特征聚合类和数据集生成函数
from sklearn.cluster import FeatureAgglomeration
from sklearn.datasets import make_blobs

# 导入数组近似相等断言函数
from sklearn.utils._testing import assert_array_almost_equal


# 定义特征聚合测试函数
def test_feature_agglomeration():
    # 设置簇的数量
    n_clusters = 1
    
    # 创建输入数据X，reshape成(1, 3)的形状
    X = np.array([0, 0, 1]).reshape(1, 3)  # (n_samples, n_features)

    # 创建使用平均池化函数的特征聚合对象
    agglo_mean = FeatureAgglomeration(n_clusters=n_clusters, pooling_func=np.mean)
    
    # 创建使用中位数池化函数的特征聚合对象
    agglo_median = FeatureAgglomeration(n_clusters=n_clusters, pooling_func=np.median)
    
    # 对数据X进行拟合
    agglo_mean.fit(X)
    agglo_median.fit(X)

    # 断言平均特征聚合的标签数量与预期相同
    assert np.size(np.unique(agglo_mean.labels_)) == n_clusters
    
    # 断言中位数特征聚合的标签数量与预期相同
    assert np.size(np.unique(agglo_median.labels_)) == n_clusters
    
    # 断言平均特征聚合的标签大小与输入数据X的特征数量相同
    assert np.size(agglo_mean.labels_) == X.shape[1]
    
    # 断言中位数特征聚合的标签大小与输入数据X的特征数量相同
    assert np.size(agglo_median.labels_) == X.shape[1]

    # 测试转换功能
    Xt_mean = agglo_mean.transform(X)
    Xt_median = agglo_median.transform(X)
    
    # 断言平均特征聚合后的形状与预期相同
    assert Xt_mean.shape[1] == n_clusters
    
    # 断言中位数特征聚合后的形状与预期相同
    assert Xt_median.shape[1] == n_clusters
    
    # 断言平均特征聚合后的结果与预期相等
    assert Xt_mean == np.array([1 / 3.0])
    
    # 断言中位数特征聚合后的结果与预期相等
    assert Xt_median == np.array([0.0])

    # 测试逆转换功能
    X_full_mean = agglo_mean.inverse_transform(Xt_mean)
    X_full_median = agglo_median.inverse_transform(Xt_median)
    
    # 断言平均特征聚合后的全数据唯一值数量与簇的数量相同
    assert np.unique(X_full_mean[0]).size == n_clusters
    
    # 断言中位数特征聚合后的全数据唯一值数量与簇的数量相同
    assert np.unique(X_full_median[0]).size == n_clusters

    # 断言平均特征聚合后的转换结果与预期相近
    assert_array_almost_equal(agglo_mean.transform(X_full_mean), Xt_mean)
    
    # 断言中位数特征聚合后的转换结果与预期相近
    assert_array_almost_equal(agglo_median.transform(X_full_median), Xt_median)


# 定义测试特征聚合函数的特征名称输出
def test_feature_agglomeration_feature_names_out():
    """Check `get_feature_names_out` for `FeatureAgglomeration`."""
    # 生成包含6个特征的数据集X
    X, _ = make_blobs(n_features=6, random_state=0)
    
    # 创建特征聚合对象
    agglo = FeatureAgglomeration(n_clusters=3)
    
    # 对数据X进行拟合
    agglo.fit(X)
    
    # 获取特征聚合后的输出特征名称
    n_clusters = agglo.n_clusters_
    names_out = agglo.get_feature_names_out()
    
    # 断言输出特征名称与预期格式相同
    assert_array_equal(
        [f"featureagglomeration{i}" for i in range(n_clusters)], names_out
    )


# TODO(1.7): remove this test
# 定义测试逆转换Xt的过时警告函数
def test_inverse_transform_Xt_deprecation():
    # 创建输入数据X，reshape成(1, 3)的形状
    X = np.array([0, 0, 1]).reshape(1, 3)  # (n_samples, n_features)

    # 创建特征聚合对象，使用平均池化函数
    est = FeatureAgglomeration(n_clusters=1, pooling_func=np.mean)
    
    # 对数据X进行拟合，并进行特征聚合转换
    est.fit(X)
    X = est.transform(X)

    # 断言调用逆转换函数时缺少必需的位置参数时引发TypeError异常
    with pytest.raises(TypeError, match="Missing required positional argument"):
        est.inverse_transform()

    # 断言同时使用X和Xt时引发TypeError异常
    with pytest.raises(TypeError, match="Cannot use both X and Xt. Use X only."):
        est.inverse_transform(X=X, Xt=X)

    # 捕获并断言警告类型为FutureWarning，匹配相应信息
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("error")
        est.inverse_transform(X)

    # 断言警告类型为FutureWarning，匹配相应信息
    with pytest.warns(FutureWarning, match="Xt was renamed X in version 1.5"):
        est.inverse_transform(Xt=X)
```