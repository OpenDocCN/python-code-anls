# `D:\src\scipysrc\scikit-learn\sklearn\impute\tests\test_knn.py`

```
# 导入需要的库
import numpy as np
import pytest

# 从 sklearn 库中导入配置上下文、KNNImputer、距离计算函数以及 KNeighborsRegressor 类
from sklearn import config_context
from sklearn.impute import KNNImputer
from sklearn.metrics.pairwise import nan_euclidean_distances, pairwise_distances
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils._testing import assert_allclose

# 使用 pytest.mark.parametrize 来定义参数化测试，测试不同的权重和邻居数量
@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("n_neighbors", range(1, 6))
def test_knn_imputer_shape(weights, n_neighbors):
    # 验证对于不同的权重和邻居数量，填充后矩阵的形状是否正确
    n_rows = 10  # 矩阵的行数
    n_cols = 2   # 矩阵的列数
    X = np.random.rand(n_rows, n_cols)  # 创建一个随机矩阵
    X[0, 0] = np.nan  # 将矩阵中第一个元素设为 NaN

    # 创建 KNNImputer 对象，进行填充
    imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
    X_imputed = imputer.fit_transform(X)  # 对矩阵进行填充
    assert X_imputed.shape == (n_rows, n_cols)  # 验证填充后矩阵的形状是否与原始矩阵相同

# 使用 pytest.mark.parametrize 来定义参数化测试，测试不同的 NaN 值
@pytest.mark.parametrize("na", [np.nan, -1])
def test_knn_imputer_default_with_invalid_input(na):
    # 测试使用默认值填充以及无效输入的情况

    # 测试矩阵中存在 inf 的情况
    X = np.array(
        [
            [np.inf, 1, 1, 2, na],
            [2, 1, 2, 2, 3],
            [3, 2, 3, 3, 8],
            [na, 6, 0, 5, 13],
            [na, 7, 0, 7, 8],
            [6, 6, 2, 5, 7],
        ]
    )
    # 预期会抛出 ValueError，并匹配特定的错误信息
    with pytest.raises(ValueError, match="Input X contains (infinity|NaN)"):
        KNNImputer(missing_values=na).fit(X)

    # 测试在 transform() 方法中传入含有 inf 的矩阵的情况
    X = np.array(
        [
            [np.inf, 1, 1, 2, na],
            [2, 1, 2, 2, 3],
            [3, 2, 3, 3, 8],
            [na, 6, 0, 5, 13],
            [na, 7, 0, 7, 8],
            [6, 6, 2, 5, 7],
        ]
    )
    # 使用已经 fit 过的 imputer 对象进行 transform，预期会抛出 ValueError
    X_fit = np.array(
        [
            [0, 1, 1, 2, na],
            [2, 1, 2, 2, 3],
            [3, 2, 3, 3, 8],
            [na, 6, 0, 5, 13],
            [na, 7, 0, 7, 8],
            [6, 6, 2, 5, 7],
        ]
    )
    imputer = KNNImputer(missing_values=na).fit(X_fit)
    with pytest.raises(ValueError, match="Input X contains (infinity|NaN)"):
        imputer.transform(X)

    # 测试在存在 NaN 的情况下使用 missing_values=0
    imputer = KNNImputer(missing_values=0, n_neighbors=2, weights="uniform")
    X = np.array(
        [
            [np.nan, 0, 0, 0, 5],
            [np.nan, 1, 0, np.nan, 3],
            [np.nan, 2, 0, 0, 0],
            [np.nan, 6, 0, 5, 13],
        ]
    )
    msg = "Input X contains NaN"
    # 预期会抛出 ValueError，并匹配特定的错误信息
    with pytest.raises(ValueError, match=msg):
        imputer.fit(X)

    X = np.array(
        [
            [0, 0],
            [np.nan, 2],
        ]
    )

@pytest.mark.parametrize("na", [np.nan, -1])
def test_knn_imputer_removes_all_na_features(na):
    # 测试 KNNImputer 对象移除所有 NaN 特征的情况
    X = np.array(
        [
            [1, 1, na, 1, 1, 1.0],
            [2, 3, na, 2, 2, 2],
            [3, 4, na, 3, 3, na],
            [6, 4, na, na, 6, 6],
        ]
    )
    knn = KNNImputer(missing_values=na, n_neighbors=2).fit(X)  # 创建并 fit KNNImputer 对象

    X_transform = knn.transform(X)  # 对矩阵进行 transform
    assert not np.isnan(X_transform).any()  # 验证 transform 后的矩阵中不再存在 NaN
    assert X_transform.shape == (4, 5)  # 验证 transform 后的矩阵形状是否符合预期
    # 创建一个 2x6 的 NumPy 数组，其值从 0 到 11，然后进行重新形状为 2x6 的操作
    X_test = np.arange(0, 12).reshape(2, 6)
    
    # 使用 knn 对象的 transform 方法对 X_test 进行变换操作，并将结果赋给 X_transform
    X_transform = knn.transform(X_test)
    
    # 使用 assert_allclose 函数断言 X_test 的第 0、1、3、4、5 列的值与 X_transform 的相应列值非常接近
    assert_allclose(X_test[:, [0, 1, 3, 4, 5]], X_transform)
# 使用 pytest 的标记参数化测试函数，测试 KNNImputer 对象处理不同缺失值的情况
@pytest.mark.parametrize("na", [np.nan, -1])
def test_knn_imputer_zero_nan_imputes_the_same(na):
    # 创建一个可填补的矩阵 X_zero，并设置不同的缺失值
    X_zero = np.array(
        [
            [1, 0, 1, 1, 1.0],
            [2, 2, 2, 2, 2],
            [3, 3, 3, 3, 0],
            [6, 6, 0, 6, 6],
        ]
    )

    # 创建一个带缺失值的矩阵 X_nan，使用参数化的缺失值 na
    X_nan = np.array(
        [
            [1, na, 1, 1, 1.0],
            [2, 2, 2, 2, 2],
            [3, 3, 3, 3, na],
            [6, 6, na, 6, 6],
        ]
    )

    # 创建填补后的参考矩阵 X_imputed
    X_imputed = np.array(
        [
            [1, 2.5, 1, 1, 1.0],
            [2, 2, 2, 2, 2],
            [3, 3, 3, 3, 1.5],
            [6, 6, 2.5, 6, 6],
        ]
    )

    # 创建使用零值作为缺失值的 KNNImputer 对象
    imputer_zero = KNNImputer(missing_values=0, n_neighbors=2, weights="uniform")

    # 创建使用 na 值作为缺失值的 KNNImputer 对象
    imputer_nan = KNNImputer(missing_values=na, n_neighbors=2, weights="uniform")

    # 断言 imputer_zero 对 X_zero 的转换结果与 X_imputed 相等
    assert_allclose(imputer_zero.fit_transform(X_zero), X_imputed)

    # 断言 imputer_zero 对 X_zero 和 imputer_nan 对 X_nan 的转换结果相等
    assert_allclose(
        imputer_zero.fit_transform(X_zero), imputer_nan.fit_transform(X_nan)
    )


# 使用 pytest 的标记参数化测试函数，测试 KNNImputer 对象的正确性
@pytest.mark.parametrize("na", [np.nan, -1])
def test_knn_imputer_verify(na):
    # 创建一个可填补的矩阵 X
    X = np.array(
        [
            [1, 0, 0, 1],
            [2, 1, 2, na],
            [3, 2, 3, na],
            [na, 4, 5, 5],
            [6, na, 6, 7],
            [8, 8, 8, 8],
            [16, 15, 18, 19],
        ]
    )

    # 创建填补后的参考矩阵 X_imputed
    X_imputed = np.array(
        [
            [1, 0, 0, 1],
            [2, 1, 2, 8],
            [3, 2, 3, 8],
            [4, 4, 5, 5],
            [6, 3, 6, 7],
            [8, 8, 8, 8],
            [16, 15, 18, 19],
        ]
    )

    # 创建 KNNImputer 对象，使用 na 作为缺失值
    imputer = KNNImputer(missing_values=na)

    # 断言 imputer 对 X 的转换结果与 X_imputed 相等
    assert_allclose(imputer.fit_transform(X), X_imputed)

    # 创建一个测试数据集 X，当邻居不足时，使用训练集的列均值进行填充
    X = np.array(
        [
            [1, 0, 0, na],
            [2, 1, 2, na],
            [3, 2, 3, na],
            [4, 4, 5, na],
            [6, 7, 6, na],
            [8, 8, 8, na],
            [20, 20, 20, 20],
            [22, 22, 22, 22],
        ]
    )

    # 计算 X_impute_value 作为 X 的列均值
    X_impute_value = (20 + 22) / 2

    # 创建填补后的参考矩阵 X_imputed
    X_imputed = np.array(
        [
            [1, 0, 0, X_impute_value],
            [2, 1, 2, X_impute_value],
            [3, 2, 3, X_impute_value],
            [4, 4, 5, X_impute_value],
            [6, 7, 6, X_impute_value],
            [8, 8, 8, X_impute_value],
            [20, 20, 20, 20],
            [22, 22, 22, 22],
        ]
    )

    # 创建 KNNImputer 对象，使用 na 作为缺失值
    imputer = KNNImputer(missing_values=na)

    # 断言 imputer 对 X 的转换结果与 X_imputed 相等
    assert_allclose(imputer.fit_transform(X), X_imputed)

    # 创建一个测试数据集 X 和 X1，验证 fit() 和 transform() 时数据不同的情况
    X = np.array([[0, 0], [na, 2], [4, 3], [5, 6], [7, 7], [9, 8], [11, 16]])
    X1 = np.array([[1, 0], [3, 2], [4, na]])

    # 计算 X_2_1 作为 X 的列均值
    X_2_1 = (0 + 3 + 6 + 7 + 8) / 5

    # 创建填补后的参考矩阵 X1_imputed
    X1_imputed = np.array([[1, 0], [3, 2], [4, X_2_1]])

    # 创建 KNNImputer 对象，使用 na 作为缺失值
    imputer = KNNImputer(missing_values=na)

    # 断言 imputer 对 X1 的转换结果与 X1_imputed 相等
    assert_allclose(imputer.fit(X).transform(X1), X1_imputed)
# 使用 pytest.mark.parametrize 装饰器定义测试函数，测试 KNNImputer 在不同参数下的表现
@pytest.mark.parametrize("na", [np.nan, -1])
def test_knn_imputer_one_n_neighbors(na):
    # 创建输入数据 X，包含缺失值 na
    X = np.array([[0, 0], [na, 2], [4, 3], [5, na], [7, 7], [na, 8], [14, 13]])

    # 期望的缺失值填充后的结果 X_imputed
    X_imputed = np.array([[0, 0], [4, 2], [4, 3], [5, 3], [7, 7], [7, 8], [14, 13]])

    # 创建 KNNImputer 实例，使用1个最近邻，并指定缺失值的类型 na
    imputer = KNNImputer(n_neighbors=1, missing_values=na)

    # 使用 assert_allclose 断言检查 imputer.fit_transform(X) 的输出是否与 X_imputed 接近
    assert_allclose(imputer.fit_transform(X), X_imputed)


@pytest.mark.parametrize("na", [np.nan, -1])
def test_knn_imputer_all_samples_are_neighbors(na):
    # 创建输入数据 X，包含缺失值 na
    X = np.array([[0, 0], [na, 2], [4, 3], [5, na], [7, 7], [na, 8], [14, 13]])

    # 期望的缺失值填充后的结果 X_imputed
    X_imputed = np.array(
        [[0, 0], [6.25, 2], [4, 3], [5, 5.75], [7, 7], [6.25, 8], [14, 13]]
    )

    # 计算邻居数量，并创建 KNNImputer 实例，使用对应的邻居数量和缺失值类型 na
    n_neighbors = X.shape[0] - 1
    imputer = KNNImputer(n_neighbors=n_neighbors, missing_values=na)

    # 使用 assert_allclose 断言检查 imputer.fit_transform(X) 的输出是否与 X_imputed 接近
    assert_allclose(imputer.fit_transform(X), X_imputed)

    # 对比使用 n_neighbors = X.shape[0] 的情况
    n_neighbors = X.shape[0]
    imputer_plus1 = KNNImputer(n_neighbors=n_neighbors, missing_values=na)
    assert_allclose(imputer_plus1.fit_transform(X), X_imputed)


@pytest.mark.parametrize("na", [np.nan, -1])
def test_knn_imputer_weight_uniform(na):
    # 创建输入数据 X，包含缺失值 na
    X = np.array([[0, 0], [na, 2], [4, 3], [5, 6], [7, 7], [9, 8], [11, 10]])

    # 使用 "uniform" 权重方式填充后的结果 X_imputed_uniform
    X_imputed_uniform = np.array(
        [[0, 0], [5, 2], [4, 3], [5, 6], [7, 7], [9, 8], [11, 10]]
    )

    # 创建 KNNImputer 实例，使用 "uniform" 权重方式和缺失值类型 na
    imputer = KNNImputer(weights="uniform", missing_values=na)

    # 使用 assert_allclose 断言检查 imputer.fit_transform(X) 的输出是否与 X_imputed_uniform 接近
    assert_allclose(imputer.fit_transform(X), X_imputed_uniform)

    # 使用自定义的权重函数 no_weight 进行填充
    def no_weight(dist):
        return None

    imputer = KNNImputer(weights=no_weight, missing_values=na)
    assert_allclose(imputer.fit_transform(X), X_imputed_uniform)

    # 使用自定义的 uniform_weight 权重函数进行填充
    def uniform_weight(dist):
        return np.ones_like(dist)

    imputer = KNNImputer(weights=uniform_weight, missing_values=na)
    assert_allclose(imputer.fit_transform(X), X_imputed_uniform)


@pytest.mark.parametrize("na", [np.nan, -1])
def test_knn_imputer_weight_distance(na):
    # 创建输入数据 X，包含缺失值 na
    X = np.array([[0, 0], [na, 2], [4, 3], [5, 6], [7, 7], [9, 8], [11, 10]])

    # 使用 "distance" 权重方式填充后的结果 X_imputed_distance1 和 X_imputed_distance2
    X_imputed_distance1 = np.array(
        [[0, 0], [manual_imputed_value, 2], [4, 3], [5, 6], [7, 7], [9, 8], [11, 10]]
    )
    X_imputed_distance2 = np.array(
        [[0, 0], [knn_imputed_value, 2], [4, 3], [5, 6], [7, 7], [9, 8], [11, 10]]
    )

    # 创建 KNNImputer 实例，使用 "distance" 权重方式和缺失值类型 na
    imputer = KNNImputer(weights="distance", missing_values=na)

    # 使用 assert_allclose 断言检查 imputer.fit_transform(X) 的输出是否与 X_imputed_distance1 接近
    assert_allclose(imputer.fit_transform(X), X_imputed_distance1)
    # 使用 KNNImputer 对象对数据 X 进行拟合并转换，使用默认参数
    assert_allclose(imputer.fit_transform(X), X_imputed_distance2)

    # 使用权重 "distance" 和邻居数为 2 进行测试
    X = np.array(
        [
            [na, 0, 0],
            [2, 1, 2],
            [3, 2, 3],
            [4, 5, 5],
        ]
    )

    # 计算缺失值为 nan 时的欧几里得距离
    dist_0_1 = np.sqrt((3 / 2) * ((1 - 0) ** 2 + (2 - 0) ** 2))
    dist_0_2 = np.sqrt((3 / 2) * ((2 - 0) ** 2 + (3 - 0) ** 2))
    # 使用加权平均计算插补值
    imputed_value = np.average([2, 3], weights=[1 / dist_0_1, 1 / dist_0_2])

    X_imputed = np.array(
        [
            [imputed_value, 0, 0],
            [2, 1, 2],
            [3, 2, 3],
            [4, 5, 5],
        ]
    )

    # 创建 KNNImputer 对象，设定权重为 "distance"，缺失值为 na
    imputer = KNNImputer(n_neighbors=2, weights="distance", missing_values=na)
    assert_allclose(imputer.fit_transform(X), X_imputed)

    # 测试不同的缺失模式
    X = np.array(
        [
            [1, 0, 0, 1],
            [0, na, 1, na],
            [1, 1, 1, na],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 1, 1],
            [10, 10, 10, 10],
        ]
    )

    # 计算带有缺失值的行之间的欧几里得距离
    dist = nan_euclidean_distances(X, missing_values=na)
    r1c1_nbor_dists = dist[1, [0, 2, 3, 4, 5]]
    r1c3_nbor_dists = dist[1, [0, 3, 4, 5, 6]]
    # 计算邻居的权重
    r1c1_nbor_wt = 1 / r1c1_nbor_dists
    r1c3_nbor_wt = 1 / r1c3_nbor_dists

    r2c3_nbor_dists = dist[2, [0, 3, 4, 5, 6]]
    r2c3_nbor_wt = 1 / r2c3_nbor_dists

    # 收集邻居的有效值
    col1_donor_values = np.ma.masked_invalid(X[[0, 2, 3, 4, 5], 1]).copy()
    col3_donor_values = np.ma.masked_invalid(X[[0, 3, 4, 5, 6], 3]).copy()

    # 计算最终的插补值
    r1c1_imp = np.ma.average(col1_donor_values, weights=r1c1_nbor_wt)
    r1c3_imp = np.ma.average(col3_donor_values, weights=r1c3_nbor_wt)
    r2c3_imp = np.ma.average(col3_donor_values, weights=r2c3_nbor_wt)

    X_imputed = np.array(
        [
            [1, 0, 0, 1],
            [0, r1c1_imp, 1, r1c3_imp],
            [1, 1, 1, r2c3_imp],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 1, 1],
            [10, 10, 10, 10],
        ]
    )

    # 创建 KNNImputer 对象，设定权重为 "distance"，缺失值为 na
    imputer = KNNImputer(weights="distance", missing_values=na)
    assert_allclose(imputer.fit_transform(X), X_imputed)

    X = np.array(
        [
            [0, 0, 0, na],
            [1, 1, 1, na],
            [2, 2, na, 2],
            [3, 3, 3, 3],
            [4, 4, 4, 4],
            [5, 5, 5, 5],
            [6, 6, 6, 6],
            [na, 7, 7, 7],
        ]
    )

    dist = pairwise_distances(
        X, metric="nan_euclidean", squared=False, missing_values=na
    )

    # 计算权重
    r0c3_w = 1.0 / dist[0, 2:-1]
    r1c3_w = 1.0 / dist[1, 2:-1]
    r2c2_w = 1.0 / dist[2, (0, 1, 3, 4, 5)]
    r7c0_w = 1.0 / dist[7, 2:7]

    # 计算加权平均值
    r0c3 = np.average(X[2:-1, -1], weights=r0c3_w)
    r1c3 = np.average(X[2:-1, -1], weights=r1c3_w)
    r2c2 = np.average(X[(0, 1, 3, 4, 5), 2], weights=r2c2_w)
    # 计算列 'r7c0' 的加权平均值，使用 X 数组的特定行范围和给定权重 'r7c0_w'
    r7c0 = np.average(X[2:7, 0], weights=r7c0_w)

    # 创建包含填充值的新数组 X_imputed
    X_imputed = np.array(
        [
            [0, 0, 0, r0c3],   # 第一行：前三列填充为 0，第四列填充为 r0c3
            [1, 1, 1, r1c3],   # 第二行：前三列填充为 1，第四列填充为 r1c3
            [2, 2, r2c2, 2],   # 第三行：前两列填充为 2，第三列填充为 r2c2，最后一列填充为 2
            [3, 3, 3, 3],      # 第四行：所有列填充为 3
            [4, 4, 4, 4],      # 第五行：所有列填充为 4
            [5, 5, 5, 5],      # 第六行：所有列填充为 5
            [6, 6, 6, 6],      # 第七行：所有列填充为 6
            [r7c0, 7, 7, 7],   # 第八行：第一列填充为 r7c0，其余列填充为 7
        ]
    )

    # 使用 KNNImputer 对象 imputer_comp_wt 基于距离加权的 KNN 方法填充 X 数组中的缺失值
    imputer_comp_wt = KNNImputer(missing_values=na, weights="distance")
    # 断言填充后的 X 数组与预期的 X_imputed 数组非常接近（使用 assert_allclose 函数）
    assert_allclose(imputer_comp_wt.fit_transform(X), X_imputed)
# 定义一个测试函数，用于测试KNNImputer类在自定义度量函数下的表现
def test_knn_imputer_callable_metric():
    # 定义一个可调用的度量函数，返回向量之间的l1范数
    def custom_callable(x, y, missing_values=np.nan, squared=False):
        # 将输入向量转换为带有NaN掩码的MaskedArray
        x = np.ma.array(x, mask=np.isnan(x))
        y = np.ma.array(y, mask=np.isnan(y))
        # 计算带有NaN的向量之间的绝对值和
        dist = np.nansum(np.abs(x - y))
        return dist

    # 创建一个包含NaN的二维数组作为测试数据集
    X = np.array([[4, 3, 3, np.nan], [6, 9, 6, 9], [4, 8, 6, 9], [np.nan, 9, 11, 10.0]])

    # 计算指定位置的缺失值的列均值
    X_0_3 = (9 + 9) / 2
    X_3_0 = (6 + 4) / 2
    # 构建用于比较的预期结果数组
    X_imputed = np.array(
        [[4, 3, 3, X_0_3], [6, 9, 6, 9], [4, 8, 6, 9], [X_3_0, 9, 11, 10.0]]
    )

    # 创建KNNImputer对象，使用2个最近邻和自定义度量函数进行缺失值插补
    imputer = KNNImputer(n_neighbors=2, metric=custom_callable)
    # 断言插补后的结果与预期结果相近
    assert_allclose(imputer.fit_transform(X), X_imputed)


# 使用参数化测试，测试KNNImputer类在简单示例下的表现
@pytest.mark.parametrize("working_memory", [None, 0])
@pytest.mark.parametrize("na", [-1, np.nan])
# 注意，设置working_memory=0以确保测试分块处理，即使对于小数据集也是如此。
# 但是，这应该会引发一个我们忽略的UserWarning。
@pytest.mark.filterwarnings("ignore:adhere to working_memory")
def test_knn_imputer_with_simple_example(na, working_memory):
    # 创建包含NaN的二维数组作为测试数据集
    X = np.array(
        [
            [0, na, 0, na],
            [1, 1, 1, na],
            [2, 2, na, 2],
            [3, 3, 3, 3],
            [4, 4, 4, 4],
            [5, 5, 5, 5],
            [6, 6, 6, 6],
            [na, 7, 7, 7],
        ]
    )

    # 计算指定位置的缺失值的列均值
    r0c1 = np.mean(X[1:6, 1])
    r0c3 = np.mean(X[2:-1, -1])
    r1c3 = np.mean(X[2:-1, -1])
    r2c2 = np.mean(X[[0, 1, 3, 4, 5], 2])
    r7c0 = np.mean(X[2:-1, 0])

    # 构建用于比较的预期结果数组
    X_imputed = np.array(
        [
            [0, r0c1, 0, r0c3],
            [1, 1, 1, r1c3],
            [2, 2, r2c2, 2],
            [3, 3, 3, 3],
            [4, 4, 4, 4],
            [5, 5, 5, 5],
            [6, 6, 6, 6],
            [r7c0, 7, 7, 7],
        ]
    )

    # 使用参数化设置和上下文管理器测试KNNImputer类在不同参数下的表现
    with config_context(working_memory=working_memory):
        imputer_comp = KNNImputer(missing_values=na)
        # 断言插补后的结果与预期结果相近
        assert_allclose(imputer_comp.fit_transform(X), X_imputed)


# 使用参数化测试，测试KNNImputer类在缺失足够的有效距离时的表现
@pytest.mark.parametrize("na", [-1, np.nan])
@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_knn_imputer_not_enough_valid_distances(na, weights):
    # 创建包含NaN的二维数组作为测试数据集
    X1 = np.array([[na, 11], [na, 1], [3, na]])
    X1_imputed = np.array([[3, 11], [3, 1], [3, 6]])

    # 创建KNNImputer对象，使用1个最近邻和不同的权重方式进行缺失值插补
    knn = KNNImputer(missing_values=na, n_neighbors=1, weights=weights)
    # 断言插补后的结果与预期结果相近
    assert_allclose(knn.fit_transform(X1), X1_imputed)

    # 创建包含NaN的二维数组作为测试数据集
    X2 = np.array([[4, na]])
    X2_imputed = np.array([[4, 6]])
    # 断言插补后的结果与预期结果相近
    assert_allclose(knn.transform(X2), X2_imputed)


# 使用参数化测试，测试KNNImputer类在存在NaN距离时的表现
@pytest.mark.parametrize("na", [-1, np.nan])
@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_knn_imputer_nan_distance(na, weights):
    # 创建包含NaN距离的训练数据集和测试数据集
    X1_train = np.array([[1, 1], [na, 2]])
    X1_test = np.array([[0, na]])
    X1_test_expected = np.array([[0, 1]])

    # 创建KNNImputer对象，使用2个最近邻和不同的权重方式进行缺失值插补
    knn1 = KNNImputer(n_neighbors=2, missing_values=na, weights=weights)
    knn1.fit(X1_train)
    # 使用 assert_allclose 函数检查 knn1 对象对 X1_test 的转换结果是否与 X1_test_expected 相近
    assert_allclose(knn1.transform(X1_test), X1_test_expected)

    # 定义 X2_train，包含缺失值 na，用于训练数据
    X2_train = np.array([[na, 1, 1], [2, na, 2], [3, 3, na]])
    # 定义 X2_test，包含缺失值 na，用于测试数据
    X2_test = np.array([[na, 0, na], [0, na, na], [na, na, 0]])
    # 定义 X2_test_expected，包含 X2_test 预期的填充结果
    X2_test_expected = np.array([[3, 0, 1], [0, 3, 2], [2, 1, 0]])

    # 创建 KNNImputer 对象 knn2，设置参数：邻居数为 2，缺失值为 na，权重为预设的 weights
    knn2 = KNNImputer(n_neighbors=2, missing_values=na, weights=weights)
    # 使用 X2_train 对象训练 knn2 对象
    knn2.fit(X2_train)
    # 使用 assert_allclose 函数检查 knn2 对象对 X2_test 的转换结果是否与 X2_test_expected 相近
    assert_allclose(knn2.transform(X2_test), X2_test_expected)
# 使用 pytest.mark.parametrize 装饰器为 test_knn_imputer_drops_all_nan_features 函数指定两个参数化的测试参数：na 分别为 -1 和 np.nan
@pytest.mark.parametrize("na", [-1, np.nan])
def test_knn_imputer_drops_all_nan_features(na):
    # 创建一个包含 NaN 值的 NumPy 数组 X1
    X1 = np.array([[na, 1], [na, 2]])
    # 使用 KNNImputer 创建 KNN 插值器 knn，设置 missing_values 参数为 na，n_neighbors 参数为 1
    knn = KNNImputer(missing_values=na, n_neighbors=1)
    # 定义期望的输出结果 X1_expected
    X1_expected = np.array([[1], [2]])
    # 使用 assert_allclose 函数验证 knn 对 X1 的拟合与期望值 X1_expected 的接近程度
    assert_allclose(knn.fit_transform(X1), X1_expected)

    # 创建另一个包含 NaN 值的 NumPy 数组 X2
    X2 = np.array([[1, 2], [3, na]])
    # 定义期望的输出结果 X2_expected
    X2_expected = np.array([[2], [1.5]])
    # 使用 assert_allclose 函数验证 knn 对 X2 的变换与期望值 X2_expected 的接近程度
    assert_allclose(knn.transform(X2), X2_expected)


# 使用 pytest.mark.parametrize 装饰器为 test_knn_imputer_distance_weighted_not_enough_neighbors 函数指定两个参数化的测试参数：working_memory 分别为 None 和 0；na 分别为 -1 和 np.nan
@pytest.mark.parametrize("working_memory", [None, 0])
@pytest.mark.parametrize("na", [-1, np.nan])
def test_knn_imputer_distance_weighted_not_enough_neighbors(na, working_memory):
    # 创建一个包含 NaN 值的 NumPy 数组 X
    X = np.array([[3, na], [2, na], [na, 4], [5, 6], [6, 8], [na, 5]])

    # 计算 X 中 NaN 值的欧氏距离，保存在 dist 变量中
    dist = pairwise_distances(
        X, metric="nan_euclidean", squared=False, missing_values=na
    )

    # 计算加权平均值 X_01, X_11, X_20, X_50，分别对应于 dist 中的不同索引位置
    X_01 = np.average(X[3:5, 1], weights=1 / dist[0, 3:5])
    X_11 = np.average(X[3:5, 1], weights=1 / dist[1, 3:5])
    X_20 = np.average(X[3:5, 0], weights=1 / dist[2, 3:5])
    X_50 = np.average(X[3:5, 0], weights=1 / dist[5, 3:5])

    # 定义期望的输出结果 X_expected
    X_expected = np.array([[3, X_01], [2, X_11], [X_20, 4], [5, 6], [6, 8], [X_50, 5]])

    # 使用 config_context 上下文管理器设定 working_memory 参数，然后进行以下测试：
    with config_context(working_memory=working_memory):
        # 使用 KNNImputer 创建 knn_3 插值器，设置 missing_values 参数为 na，n_neighbors 参数为 3，weights 参数为 "distance"
        knn_3 = KNNImputer(missing_values=na, n_neighbors=3, weights="distance")
        # 使用 assert_allclose 函数验证 knn_3 对 X 的拟合与期望值 X_expected 的接近程度
        assert_allclose(knn_3.fit_transform(X), X_expected)

        # 使用 KNNImputer 创建 knn_4 插值器，设置 missing_values 参数为 na，n_neighbors 参数为 4，weights 参数为 "distance"
        knn_4 = KNNImputer(missing_values=na, n_neighbors=4, weights="distance")
        # 使用 assert_allclose 函数验证 knn_4 对 X 的拟合与期望值 X_expected 的接近程度
        assert_allclose(knn_4.fit_transform(X), X_expected)


# 使用 pytest.mark.parametrize 装饰器为 test_knn_tags 函数指定两个参数化的测试参数：na 分别为 -1 和 np.nan；allow_nan 分别为 False 和 True
@pytest.mark.parametrize("na, allow_nan", [(-1, False), (np.nan, True)])
def test_knn_tags(na, allow_nan):
    # 使用 KNNImputer 创建 knn 插值器，设置 missing_values 参数为 na
    knn = KNNImputer(missing_values=na)
    # 使用 assert 语句验证 knn 对应的 _get_tags()["allow_nan"] 是否等于 allow_nan
    assert knn._get_tags()["allow_nan"] == allow_nan
```