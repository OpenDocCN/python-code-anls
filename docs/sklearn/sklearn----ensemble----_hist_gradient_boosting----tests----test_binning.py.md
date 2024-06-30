# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\_hist_gradient_boosting\tests\test_binning.py`

```
# 导入必要的库和模块
import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest，用于编写和运行测试
from numpy.testing import assert_allclose, assert_array_equal  # 导入NumPy测试工具

# 导入需要测试的函数和类
from sklearn.ensemble._hist_gradient_boosting.binning import (
    _BinMapper,
    _find_binning_thresholds,
    _map_to_bins,
)
from sklearn.ensemble._hist_gradient_boosting.common import (
    ALMOST_INF,
    X_BINNED_DTYPE,
    X_DTYPE,
)
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads  # 导入OpenMP线程效率相关函数

# 获取当前的OpenMP有效线程数
n_threads = _openmp_effective_n_threads()

# 生成测试数据
DATA = (
    np.random.RandomState(42)
    .normal(loc=[0, 10], scale=[1, 0.01], size=(int(1e6), 2))
    .astype(X_DTYPE)
)

# 测试函数：test_find_binning_thresholds_regular_data
def test_find_binning_thresholds_regular_data():
    # 生成线性间隔的测试数据
    data = np.linspace(0, 10, 1001)
    # 调用函数获取分箱阈值
    bin_thresholds = _find_binning_thresholds(data, max_bins=10)
    # 断言分箱阈值与预期值接近
    assert_allclose(bin_thresholds, [1, 2, 3, 4, 5, 6, 7, 8, 9])

    # 再次调用函数获取不同最大分箱数下的分箱阈值
    bin_thresholds = _find_binning_thresholds(data, max_bins=5)
    assert_allclose(bin_thresholds, [2, 4, 6, 8])

# 测试函数：test_find_binning_thresholds_small_regular_data
def test_find_binning_thresholds_small_regular_data():
    data = np.linspace(0, 10, 11)

    bin_thresholds = _find_binning_thresholds(data, max_bins=5)
    assert_allclose(bin_thresholds, [2, 4, 6, 8])

    bin_thresholds = _find_binning_thresholds(data, max_bins=10)
    assert_allclose(bin_thresholds, [1, 2, 3, 4, 5, 6, 7, 8, 9])

    bin_thresholds = _find_binning_thresholds(data, max_bins=11)
    assert_allclose(bin_thresholds, np.arange(10) + 0.5)

    bin_thresholds = _find_binning_thresholds(data, max_bins=255)
    assert_allclose(bin_thresholds, np.arange(10) + 0.5)

# 测试函数：test_find_binning_thresholds_random_data
def test_find_binning_thresholds_random_data():
    # 对随机生成的数据进行分箱阈值计算
    bin_thresholds = [
        _find_binning_thresholds(DATA[:, i], max_bins=255) for i in range(2)
    ]
    # 断言分箱阈值的形状和数据类型正确
    for i in range(len(bin_thresholds)):
        assert bin_thresholds[i].shape == (254,)  # 255 - 1
        assert bin_thresholds[i].dtype == DATA.dtype

    # 断言特定索引处的分箱阈值接近预期值
    assert_allclose(
        bin_thresholds[0][[64, 128, 192]], np.array([-0.7, 0.0, 0.7]), atol=1e-1
    )

    assert_allclose(
        bin_thresholds[1][[64, 128, 192]], np.array([9.99, 10.00, 10.01]), atol=1e-2
    )

# 测试函数：test_find_binning_thresholds_low_n_bins
def test_find_binning_thresholds_low_n_bins():
    bin_thresholds = [
        _find_binning_thresholds(DATA[:, i], max_bins=128) for i in range(2)
    ]
    for i in range(len(bin_thresholds)):
        assert bin_thresholds[i].shape == (127,)  # 128 - 1
        assert bin_thresholds[i].dtype == DATA.dtype

# 参数化测试函数：test_invalid_n_bins
@pytest.mark.parametrize("n_bins", (2, 257))
def test_invalid_n_bins(n_bins):
    # 准备错误信息
    err_msg = "n_bins={} should be no smaller than 3 and no larger than 256".format(
        n_bins
    )
    # 断言在参数化条件下抛出预期的错误
    with pytest.raises(ValueError, match=err_msg):
        _BinMapper(n_bins=n_bins).fit(DATA)

# 测试函数：test_bin_mapper_n_features_transform
def test_bin_mapper_n_features_transform():
    # 创建并拟合分箱映射器
    mapper = _BinMapper(n_bins=42, random_state=42).fit(DATA)
    # 准备错误信息
    err_msg = "This estimator was fitted with 2 features but 4 got passed"
    # 断言在特定条件下抛出预期的错误
    with pytest.raises(ValueError, match=err_msg):
        mapper.transform(np.repeat(DATA, 2, axis=1))

# 参数化测试函数：test_map_to_bins
@pytest.mark.parametrize("max_bins", [16, 128, 255])
def test_map_to_bins(max_bins):
    # 为每个特征列计算分箱的阈值，返回一个包含阈值数组的列表
    bin_thresholds = [
        _find_binning_thresholds(DATA[:, i], max_bins=max_bins) for i in range(2)
    ]
    # 创建一个与DATA相同形状的全零数组，但数据类型为X_BINNED_DTYPE，以F顺序存储
    binned = np.zeros_like(DATA, dtype=X_BINNED_DTYPE, order="F")
    # 创建一个长度为2的零数组，用于表示每个特征列是否为分类变量
    is_categorical = np.zeros(2, dtype=np.uint8)
    # 将最后一个分箱的索引设置为max_bins
    last_bin_idx = max_bins
    # 将DATA映射到分箱中，填充binned数组
    _map_to_bins(DATA, bin_thresholds, is_categorical, last_bin_idx, n_threads, binned)
    # 断言binned数组的形状与DATA相同
    assert binned.shape == DATA.shape
    # 断言binned数组的数据类型为np.uint8
    assert binned.dtype == np.uint8
    # 断言binned数组以F顺序存储
    assert binned.flags.f_contiguous

    # 计算每列中最小值的索引
    min_indices = DATA.argmin(axis=0)
    # 计算每列中最大值的索引
    max_indices = DATA.argmax(axis=0)

    # 对于每个特征索引和其最小索引，断言对应的binned值为0
    for feature_idx, min_idx in enumerate(min_indices):
        assert binned[min_idx, feature_idx] == 0
    # 对于每个特征索引和其最大索引，断言对应的binned值为max_bins - 1
    for feature_idx, max_idx in enumerate(max_indices):
        assert binned[max_idx, feature_idx] == max_bins - 1
# 使用 pytest 的装饰器标记参数化测试，测试用例将会以 max_bins 参数为 [5, 10, 42] 运行多次
@pytest.mark.parametrize("max_bins", [5, 10, 42])
def test_bin_mapper_random_data(max_bins):
    # 获取数据集的样本数和特征数
    n_samples, n_features = DATA.shape

    # 预期每个箱子中的样本数
    expected_count_per_bin = n_samples // max_bins
    # 容差，允许的偏差为预期每个箱子样本数的5%
    tol = int(0.05 * expected_count_per_bin)

    # max_bins 是非缺失值的箱子数目
    n_bins = max_bins + 1
    # 使用随机状态 42 初始化 _BinMapper 对象，并拟合数据
    mapper = _BinMapper(n_bins=n_bins, random_state=42).fit(DATA)
    # 将数据转换为分箱数据
    binned = mapper.transform(DATA)

    # 断言分箱后数据的形状应为 (样本数, 特征数)
    assert binned.shape == (n_samples, n_features)
    # 断言分箱后数据的类型应为 np.uint8
    assert binned.dtype == np.uint8
    # 断言每个特征的最小值应为 [0, 0]
    assert_array_equal(binned.min(axis=0), np.array([0, 0]))
    # 断言每个特征的最大值应为 [max_bins-1, max_bins-1]
    assert_array_equal(binned.max(axis=0), np.array([max_bins - 1, max_bins - 1]))
    # 断言 bin_thresholds_ 的长度应为特征数
    assert len(mapper.bin_thresholds_) == n_features
    # 对于每个特征的 bin_thresholds_，断言其形状为 (max_bins-1,)，数据类型与原始数据相同
    for bin_thresholds_feature in mapper.bin_thresholds_:
        assert bin_thresholds_feature.shape == (max_bins - 1,)
        assert bin_thresholds_feature.dtype == DATA.dtype
    # 断言非缺失值的箱子数目为 max_bins
    assert np.all(mapper.n_bins_non_missing_ == max_bins)

    # 检查分箱后数据在不同箱子中大致平衡
    for feature_idx in range(n_features):
        for bin_idx in range(max_bins):
            count = (binned[:, feature_idx] == bin_idx).sum()
            # 断言每个箱子中样本数与预期接近
            assert abs(count - expected_count_per_bin) < tol


# 使用 pytest 的装饰器标记参数化测试，测试用例将会以 (n_samples, max_bins) 参数为 [(5, 5), (5, 10), (5, 11), (42, 255)] 运行多次
@pytest.mark.parametrize("n_samples, max_bins", [(5, 5), (5, 10), (5, 11), (42, 255)])
def test_bin_mapper_small_random_data(n_samples, max_bins):
    # 生成随机正态分布数据，确保数据的唯一值数目与样本数相同
    data = np.random.RandomState(42).normal(size=n_samples).reshape(-1, 1)
    assert len(np.unique(data)) == n_samples

    # max_bins 是非缺失值的箱子数目
    n_bins = max_bins + 1
    # 使用随机状态 42 初始化 _BinMapper 对象
    mapper = _BinMapper(n_bins=n_bins, random_state=42)
    # 将数据拟合并转换为分箱数据
    binned = mapper.fit_transform(data)

    # 断言分箱后数据的形状与原始数据相同
    assert binned.shape == data.shape
    # 断言分箱后数据的类型应为 np.uint8
    assert binned.dtype == np.uint8
    # 断言分箱后数据的顺序应与原始数据排序后一致
    assert_array_equal(binned.ravel()[np.argsort(data.ravel())], np.arange(n_samples))


# 使用 pytest 的装饰器标记参数化测试，测试用例将会以 (max_bins, n_distinct, multiplier) 参数为 [(5, 5, 1), (5, 5, 3), (255, 12, 42)] 运行多次
@pytest.mark.parametrize(
    "max_bins, n_distinct, multiplier",
    [
        (5, 5, 1),
        (5, 5, 3),
        (255, 12, 42),
    ],
)
def test_bin_mapper_identity_repeated_values(max_bins, n_distinct, multiplier):
    # 创建包含重复数值的数组数据
    data = np.array(list(range(n_distinct)) * multiplier).reshape(-1, 1)
    # max_bins 是非缺失值的箱子数目
    n_bins = max_bins + 1
    # 使用 _BinMapper 进行拟合和转换
    binned = _BinMapper(n_bins=n_bins).fit_transform(data)
    # 断言原始数据与分箱后数据相等
    assert_array_equal(data, binned)


# 使用 pytest 的装饰器标记参数化测试，测试用例将会以 n_distinct 参数为 [2, 7, 42] 运行多次
@pytest.mark.parametrize("n_distinct", [2, 7, 42])
def test_bin_mapper_repeated_values_invariance(n_distinct):
    # 使用随机状态 42 生成不同数值的数组
    rng = np.random.RandomState(42)
    distinct_values = rng.normal(size=n_distinct)
    assert len(np.unique(distinct_values)) == n_distinct

    # 生成包含重复值的数据，并确保数据的唯一值与原始数据的排序后唯一值一致
    repeated_indices = rng.randint(low=0, high=n_distinct, size=1000)
    data = distinct_values[repeated_indices]
    rng.shuffle(data)
    assert_array_equal(np.unique(data), np.sort(distinct_values))

    # 将数据转换为二维数组形式
    data = data.reshape(-1, 1)

    # 初始化 _BinMapper 对象
    mapper_1 = _BinMapper(n_bins=n_distinct + 1)
    # 对数据进行拟合和转换
    binned_1 = mapper_1.fit_transform(data)
    # 断言分箱后每列数据的唯一值与 np.arange(n_distinct) 相等
    assert_array_equal(np.unique(binned_1[:, 0]), np.arange(n_distinct))
    # 使用更多的 bin 数量生成的映射器将产生相同的结果（相同的阈值）
    mapper_2 = _BinMapper(n_bins=min(256, n_distinct * 3) + 1)
    # 使用 mapper_2 对数据进行拟合和转换
    binned_2 = mapper_2.fit_transform(data)

    # 断言：验证第一个映射器和第二个映射器的第一个 bin 的阈值是否相等
    assert_allclose(mapper_1.bin_thresholds_[0], mapper_2.bin_thresholds_[0])
    # 断言：验证 binned_1 和 binned_2 是否完全相等
    assert_array_equal(binned_1, binned_2)
# 使用 pytest 的 @pytest.mark.parametrize 装饰器，为 test_bin_mapper_identity_small 函数定义多组参数化测试数据
@pytest.mark.parametrize(
    "max_bins, scale, offset",
    [
        (3, 2, -1),     # 第一组参数：max_bins=3, scale=2, offset=-1
        (42, 1, 0),     # 第二组参数：max_bins=42, scale=1, offset=0
        (255, 0.3, 42), # 第三组参数：max_bins=255, scale=0.3, offset=42
    ],
)
# 定义测试函数 test_bin_mapper_identity_small，用于验证 _BinMapper 的 fit_transform 方法在不同参数下的输出
def test_bin_mapper_identity_small(max_bins, scale, offset):
    # 生成测试数据，构造一个一维数组，并根据 scale 和 offset 进行线性变换
    data = np.arange(max_bins).reshape(-1, 1) * scale + offset
    # 计算 n_bins，为 max_bins 的值加1
    n_bins = max_bins + 1
    # 使用 _BinMapper 对象，调用其 fit_transform 方法处理数据
    binned = _BinMapper(n_bins=n_bins).fit_transform(data)
    # 断言处理后的结果与预期的一维数组相等
    assert_array_equal(binned, np.arange(max_bins).reshape(-1, 1))


# 使用 pytest 的 @pytest.mark.parametrize 装饰器，为 test_bin_mapper_idempotence 函数定义多组参数化测试数据
@pytest.mark.parametrize(
    "max_bins_small, max_bins_large",
    [
        (2, 2),
        (3, 3),
        (4, 4),
        (42, 42),
        (255, 255),
        (5, 17),
        (42, 255),
    ],
)
# 定义测试函数 test_bin_mapper_idempotence，验证 _BinMapper 的 fit_transform 方法在不同参数下的输出
def test_bin_mapper_idempotence(max_bins_small, max_bins_large):
    # 确保 max_bins_large 大于等于 max_bins_small
    assert max_bins_large >= max_bins_small
    # 生成随机数据，构造一个二维数组
    data = np.random.RandomState(42).normal(size=30000).reshape(-1, 1)
    # 创建两个 _BinMapper 对象，分别处理不同参数的数据
    mapper_small = _BinMapper(n_bins=max_bins_small + 1)
    mapper_large = _BinMapper(n_bins=max_bins_small + 1)
    # 分别对数据进行 fit_transform 处理
    binned_small = mapper_small.fit_transform(data)
    binned_large = mapper_large.fit_transform(binned_small)
    # 断言两次处理的结果相等
    assert_array_equal(binned_small, binned_large)


# 使用 pytest 的 @pytest.mark.parametrize 装饰器，为 test_n_bins_non_missing 函数定义多组参数化测试数据
@pytest.mark.parametrize("n_bins", [10, 100, 256])
@pytest.mark.parametrize("diff", [-5, 0, 5])
# 定义测试函数 test_n_bins_non_missing，验证 _BinMapper 对象处理不同 n_bins 和 diff 参数下的数据处理情况
def test_n_bins_non_missing(n_bins, diff):
    # 设置 n_unique_values，用于生成测试数据
    n_unique_values = n_bins + diff
    # 构造一个重复的列表，并转换为 numpy 二维数组
    X = list(range(n_unique_values)) * 2
    X = np.array(X).reshape(-1, 1)
    # 创建 _BinMapper 对象，并对数据进行 fit 处理
    mapper = _BinMapper(n_bins=n_bins).fit(X)
    # 断言 _BinMapper 对象的 n_bins_non_missing_ 属性与预期的最小值相等
    assert np.all(mapper.n_bins_non_missing_ == min(n_bins - 1, n_unique_values))


# 定义测试函数 test_subsample，验证 _BinMapper 对象在应用子采样时的处理情况
def test_subsample():
    # 创建两个 _BinMapper 对象，分别使用不同的子采样参数和随机种子处理数据集 DATA
    mapper_no_subsample = _BinMapper(subsample=None, random_state=0).fit(DATA)
    mapper_subsample = _BinMapper(subsample=256, random_state=0).fit(DATA)

    # 遍历数据集的特征，并断言两种处理方式得到的 bin_thresholds_ 在指定的相对误差范围内不相似
    for feature in range(DATA.shape[1]):
        assert not np.allclose(
            mapper_no_subsample.bin_thresholds_[feature],
            mapper_subsample.bin_thresholds_[feature],
            rtol=1e-4,
        )


# 使用 pytest 的 @pytest.mark.parametrize 装饰器，为 test_missing_values_support 函数定义多组参数化测试数据
@pytest.mark.parametrize(
    "n_bins, n_bins_non_missing, X_trans_expected",
    [
        (
            256,
            [4, 2, 2],
            [
                [0, 0, 0],  # 255 <=> missing value
                [255, 255, 0],
                [1, 0, 0],
                [255, 1, 1],
                [2, 1, 1],
                [3, 0, 0],
            ],
        ),
        (
            3,
            [2, 2, 2],
            [
                [0, 0, 0],  # 2 <=> missing value
                [2, 2, 0],
                [0, 0, 0],
                [2, 1, 1],
                [1, 1, 1],
                [1, 0, 0],
            ],
        ),
    ],
)
# 定义测试函数 test_missing_values_support，验证 _BinMapper 对象对含有缺失值的数据的处理情况
def test_missing_values_support(n_bins, n_bins_non_missing, X_trans_expected):
    # 检查含有缺失值的数据，确保 NaN 值映射到最后一个 bin，并验证 _BinMapper 对象的属性正确性
    pass
    # 定义包含 NaN 值的二维数组 X，每行代表一个样本，每列代表一个特征
    X = [
        [1, 1, 0],
        [np.nan, np.nan, 0],
        [2, 1, 0],
        [np.nan, 2, 1],
        [3, 2, 1],
        [4, 1, 0],
    ]
    
    # 将列表 X 转换为 NumPy 数组
    X = np.array(X)
    
    # 使用 _BinMapper 类创建 mapper 对象，并指定 n_bins 参数进行初始化
    mapper = _BinMapper(n_bins=n_bins)
    
    # 对 mapper 对象进行拟合，以便根据 X 的数据进行分箱处理
    mapper.fit(X)
    
    # 断言检查 mapper 对象中的 n_bins_non_missing_ 属性与给定的 n_bins_non_missing 是否相等
    assert_array_equal(mapper.n_bins_non_missing_, n_bins_non_missing)
    
    # 遍历 X 的每个特征索引
    for feature_idx in range(X.shape[1]):
        # 断言检查每个特征的分箱阈值数组长度是否等于 n_bins_non_missing 对应特征索引的值减 1
        assert (
            len(mapper.bin_thresholds_[feature_idx])
            == n_bins_non_missing[feature_idx] - 1
        )
    
    # 断言检查 mapper 对象中的 missing_values_bin_idx_ 属性是否等于 n_bins - 1
    assert mapper.missing_values_bin_idx_ == n_bins - 1
    
    # 使用 mapper 对象对 X 进行转换，得到转换后的 X_trans
    X_trans = mapper.transform(X)
    
    # 断言检查转换后的 X_trans 是否与预期的 X_trans_expected 数组相等
    assert_array_equal(X_trans, X_trans_expected)
def test_infinite_values():
    # Make sure infinite values are properly handled.
    # 创建一个 _BinMapper 的实例对象
    bin_mapper = _BinMapper()

    # 创建一个包含无穷大数值的 NumPy 数组 X
    X = np.array([-np.inf, 0, 1, np.inf]).reshape(-1, 1)

    # 使用 X 对 bin_mapper 进行拟合
    bin_mapper.fit(X)
    # 检查 bin_thresholds_ 第一个元素是否与预期接近
    assert_allclose(bin_mapper.bin_thresholds_[0], [-np.inf, 0.5, ALMOST_INF])
    # 检查 n_bins_non_missing_ 是否等于 [4]
    assert bin_mapper.n_bins_non_missing_ == [4]

    # 创建一个预期的经过分箱转换后的 X 数组
    expected_binned_X = np.array([0, 1, 2, 3]).reshape(-1, 1)
    # 检查 bin_mapper 对 X 的转换结果是否与预期相等
    assert_array_equal(bin_mapper.transform(X), expected_binned_X)


@pytest.mark.parametrize("n_bins", [15, 256])
def test_categorical_feature(n_bins):
    # Basic test for categorical features
    # we make sure that categories are mapped into [0, n_categories - 1] and
    # that nans are mapped to the last bin
    # 创建一个包含分类特征的 NumPy 数组 X，其中包括了 NaN 值
    X = np.array(
        [[4] * 500 + [1] * 3 + [10] * 4 + [0] * 4 + [13] + [7] * 5 + [np.nan] * 2],
        dtype=X_DTYPE,
    ).T
    # 获取已知的分类特征列表
    known_categories = [np.unique(X[~np.isnan(X)])]

    # 创建 _BinMapper 的实例对象，使用 fit 方法拟合 X
    bin_mapper = _BinMapper(
        n_bins=n_bins,
        is_categorical=np.array([True]),
        known_categories=known_categories,
    ).fit(X)
    # 检查 n_bins_non_missing_ 是否等于 [6]
    assert bin_mapper.n_bins_non_missing_ == [6]
    # 检查 bin_thresholds_ 第一个元素是否与预期相等
    assert_array_equal(bin_mapper.bin_thresholds_[0], [0, 1, 4, 7, 10, 13])

    # 创建一个新的 NumPy 数组 X，用于测试转换
    X = np.array([[0, 1, 4, np.nan, 7, 10, 13]], dtype=X_DTYPE).T
    # 创建一个预期的转换结果
    expected_trans = np.array([[0, 1, 2, n_bins - 1, 3, 4, 5]]).T
    # 检查 bin_mapper 对 X 的转换结果是否与预期相等
    assert_array_equal(bin_mapper.transform(X), expected_trans)

    # 创建另一个新的 NumPy 数组 X，用于测试转换
    X = np.array([[-4, -1, 100]], dtype=X_DTYPE).T
    # 创建一个预期的转换结果
    expected_trans = np.array([[n_bins - 1, n_bins - 1, 6]]).T
    # 检查 bin_mapper 对 X 的转换结果是否与预期相等
    assert_array_equal(bin_mapper.transform(X), expected_trans)


def test_categorical_feature_negative_missing():
    """Make sure bin mapper treats negative categories as missing values."""
    # 创建一个包含负数分类特征的 NumPy 数组 X，其中包括了 NaN 值
    X = np.array(
        [[4] * 500 + [1] * 3 + [5] * 10 + [-1] * 3 + [np.nan] * 4], dtype=X_DTYPE
    ).T
    # 创建 _BinMapper 的实例对象，使用 fit 方法拟合 X
    bin_mapper = _BinMapper(
        n_bins=4,
        is_categorical=np.array([True]),
        known_categories=[np.array([1, 4, 5], dtype=X_DTYPE)],
    ).fit(X)

    # 检查 n_bins_non_missing_ 是否等于 [3]
    assert bin_mapper.n_bins_non_missing_ == [3]

    # 创建一个新的 NumPy 数组 X，用于测试转换
    X = np.array([[-1, 1, 3, 5, np.nan]], dtype=X_DTYPE).T

    # 检查负数分类特征被视为缺失值，是否被映射到 missing_values_bin_idx_ 所指示的 bin（这里是 3）
    assert bin_mapper.missing_values_bin_idx_ == 3
    # 创建一个预期的转换结果
    expected_trans = np.array([[3, 0, 1, 2, 3]]).T
    # 检查 bin_mapper 对 X 的转换结果是否与预期相等
    assert_array_equal(bin_mapper.transform(X), expected_trans)


@pytest.mark.parametrize("n_bins", (128, 256))
def test_categorical_with_numerical_features(n_bins):
    # basic check for binmapper with mixed data
    # 创建两个包含数值特征的 NumPy 数组 X1 和 X2
    X1 = np.arange(10, 20).reshape(-1, 1)  # numerical
    X2 = np.arange(10, 15).reshape(-1, 1)  # categorical
    X2 = np.r_[X2, X2]
    X = np.c_[X1, X2]
    # 初始化已知类别列表，包括一个空值和X2列的唯一值数组
    known_categories = [None, np.unique(X2).astype(X_DTYPE)]
    
    # 创建_BinMapper对象，设定参数如下：
    # - n_bins为指定的箱数
    # - is_categorical是一个布尔数组，表示每列是否是分类变量（第一列不是，第二列是）
    # - known_categories为已知的类别信息，包含空值和X2列的唯一值数组
    bin_mapper = _BinMapper(
        n_bins=n_bins,
        is_categorical=np.array([False, True]),
        known_categories=known_categories,
    ).fit(X)
    
    # 检查非缺失值的箱数是否如预期，分别是10和5
    assert_array_equal(bin_mapper.n_bins_non_missing_, [10, 5])
    
    # 获取计算后的箱阈值
    bin_thresholds = bin_mapper.bin_thresholds_
    # 检查阈值的长度是否为2
    assert len(bin_thresholds) == 2
    # 检查第二个特征的阈值是否为从10到14的数组
    assert_array_equal(bin_thresholds[1], np.arange(10, 15))
    
    # 预期的X转换结果，将X转换为相应的箱索引
    expected_X_trans = [
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4],
        [5, 0],
        [6, 1],
        [7, 2],
        [8, 3],
        [9, 4],
    ]
    # 检查_X是否按预期被转换为expected_X_trans
    assert_array_equal(bin_mapper.transform(X), expected_X_trans)
def test_make_known_categories_bitsets():
    # 检查 make_known_categories_bitsets 的输出
    X = np.array(
        [[14, 2, 30], [30, 4, 70], [40, 10, 180], [40, 240, 180]], dtype=X_DTYPE
    )

    # 创建 _BinMapper 对象，指定参数
    bin_mapper = _BinMapper(
        n_bins=256,
        is_categorical=np.array([False, True, True]),
        known_categories=[None, X[:, 1], X[:, 2]],
    )
    # 对 X 进行拟合
    bin_mapper.fit(X)

    # 调用 make_known_categories_bitsets 方法生成已知类别的位集合和特征索引映射
    known_cat_bitsets, f_idx_map = bin_mapper.make_known_categories_bitsets()

    # 注意：非分类特征的值保持为 0
    expected_f_idx_map = np.array([0, 0, 1], dtype=np.uint8)
    assert_allclose(expected_f_idx_map, f_idx_map)

    # 预期的类别位集合，初始化为全 0 的数组
    expected_cat_bitset = np.zeros((2, 8), dtype=np.uint32)

    # 第一个分类特征: [2, 4, 10, 240]
    f_idx = 1
    mapped_f_idx = f_idx_map[f_idx]
    expected_cat_bitset[mapped_f_idx, 0] = 2**2 + 2**4 + 2**10
    # 240 = 32**7 + 16，因此在第 7 个数组的第 16 位设置为 1
    expected_cat_bitset[mapped_f_idx, 7] = 2**16

    # 第二个分类特征: [30, 70, 180]
    f_idx = 2
    mapped_f_idx = f_idx_map[f_idx]
    expected_cat_bitset[mapped_f_idx, 0] = 2**30
    expected_cat_bitset[mapped_f_idx, 2] = 2**6
    expected_cat_bitset[mapped_f_idx, 5] = 2**20

    # 断言已知类别的位集合与预期结果相近
    assert_allclose(expected_cat_bitset, known_cat_bitsets)


@pytest.mark.parametrize(
    "is_categorical, known_categories, match",
    [
        (np.array([True]), [None], "Known categories for feature 0 must be provided"),
        (
            np.array([False]),
            np.array([1, 2, 3]),
            "isn't marked as a categorical feature, but categories were passed",
        ),
    ],
)
def test_categorical_parameters(is_categorical, known_categories, match):
    # 测试 is_categorical 和 known_categories 参数的验证

    X = np.array([[1, 2, 3]], dtype=X_DTYPE)

    # 创建 _BinMapper 对象，指定 is_categorical 和 known_categories 参数
    bin_mapper = _BinMapper(
        is_categorical=is_categorical, known_categories=known_categories
    )
    # 使用 pytest 检查是否抛出 ValueError 异常，异常信息应匹配 match 字符串
    with pytest.raises(ValueError, match=match):
        bin_mapper.fit(X)
```