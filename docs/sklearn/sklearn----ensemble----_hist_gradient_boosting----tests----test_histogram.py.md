# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\_hist_gradient_boosting\tests\test_histogram.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 库，用于编写和运行测试
from numpy.testing import assert_allclose, assert_array_equal  # 导入 NumPy 测试工具函数

from sklearn.ensemble._hist_gradient_boosting.common import (
    G_H_DTYPE,  # 导入定义的数据类型 G_H_DTYPE
    HISTOGRAM_DTYPE,  # 导入定义的数据类型 HISTOGRAM_DTYPE
    X_BINNED_DTYPE,  # 导入定义的数据类型 X_BINNED_DTYPE
)
from sklearn.ensemble._hist_gradient_boosting.histogram import (
    _build_histogram,  # 导入构建直方图的函数 _build_histogram
    _build_histogram_naive,  # 导入朴素构建直方图的函数 _build_histogram_naive
    _build_histogram_no_hessian,  # 导入不考虑 Hessian 矩阵构建直方图的函数 _build_histogram_no_hessian
    _build_histogram_root,  # 导入根节点构建直方图的函数 _build_histogram_root
    _build_histogram_root_no_hessian,  # 导入根节点不考虑 Hessian 矩阵构建直方图的函数 _build_histogram_root_no_hessian
    _subtract_histograms,  # 导入直方图相减的函数 _subtract_histograms
)


@pytest.mark.parametrize("build_func", [_build_histogram_naive, _build_histogram])
def test_build_histogram(build_func):
    binned_feature = np.array([0, 2, 0, 1, 2, 0, 2, 1], dtype=X_BINNED_DTYPE)

    # Small sample_indices (below unrolling threshold)
    ordered_gradients = np.array([0, 1, 3], dtype=G_H_DTYPE)
    ordered_hessians = np.array([1, 1, 2], dtype=G_H_DTYPE)

    sample_indices = np.array([0, 2, 3], dtype=np.uint32)
    hist = np.zeros((1, 3), dtype=HISTOGRAM_DTYPE)

    # 调用指定的构建直方图函数，填充 hist 数组
    build_func(
        0, sample_indices, binned_feature, ordered_gradients, ordered_hessians, hist
    )

    hist = hist[0]
    # 断言直方图的计数部分
    assert_array_equal(hist["count"], [2, 1, 0])
    # 断言直方图的梯度和部分，要求数据在一个较小的误差范围内相等
    assert_allclose(hist["sum_gradients"], [1, 3, 0])
    # 断言直方图的 Hessian 矩阵和部分，要求数据在一个较小的误差范围内相等
    assert_allclose(hist["sum_hessians"], [2, 2, 0])

    # Larger sample_indices (above unrolling threshold)
    sample_indices = np.array([0, 2, 3, 6, 7], dtype=np.uint32)
    ordered_gradients = np.array([0, 1, 3, 0, 1], dtype=G_H_DTYPE)
    ordered_hessians = np.array([1, 1, 2, 1, 0], dtype=G_H_DTYPE)

    hist = np.zeros((1, 3), dtype=HISTOGRAM_DTYPE)

    # 再次调用指定的构建直方图函数，填充 hist 数组
    build_func(
        0, sample_indices, binned_feature, ordered_gradients, ordered_hessians, hist
    )

    hist = hist[0]
    # 断言直方图的计数部分
    assert_array_equal(hist["count"], [2, 2, 1])
    # 断言直方图的梯度和部分，要求数据在一个较小的误差范围内相等
    assert_allclose(hist["sum_gradients"], [1, 4, 0])
    # 断言直方图的 Hessian 矩阵和部分，要求数据在一个较小的误差范围内相等
    assert_allclose(hist["sum_hessians"], [2, 2, 1])


def test_histogram_sample_order_independence():
    # Make sure the order of the samples has no impact on the histogram
    # computations
    rng = np.random.RandomState(42)
    n_sub_samples = 100
    n_samples = 1000
    n_bins = 256

    binned_feature = rng.randint(0, n_bins - 1, size=n_samples, dtype=X_BINNED_DTYPE)
    sample_indices = rng.choice(
        np.arange(n_samples, dtype=np.uint32), n_sub_samples, replace=False
    )
    ordered_gradients = rng.randn(n_sub_samples).astype(G_H_DTYPE)

    hist_gc = np.zeros((1, n_bins), dtype=HISTOGRAM_DTYPE)
    _build_histogram_no_hessian(
        0, sample_indices, binned_feature, ordered_gradients, hist_gc
    )

    ordered_hessians = rng.exponential(size=n_sub_samples).astype(G_H_DTYPE)
    hist_ghc = np.zeros((1, n_bins), dtype=HISTOGRAM_DTYPE)

    _build_histogram(
        0, sample_indices, binned_feature, ordered_gradients, ordered_hessians, hist_ghc
    )

    permutation = rng.permutation(n_sub_samples)
    hist_gc_perm = np.zeros((1, n_bins), dtype=HISTOGRAM_DTYPE)
    # 调用函数 _build_histogram_no_hessian，构建不包含 Hessian 矩阵的直方图
    _build_histogram_no_hessian(
        0,  # 参数0：指定直方图索引
        sample_indices[permutation],  # 使用打乱顺序的样本索引
        binned_feature,  # 经过分箱处理的特征值
        ordered_gradients[permutation],  # 使用打乱顺序的梯度值
        hist_gc_perm,  # 直方图结果将存储在 hist_gc_perm 中
    )

    # 初始化一个全零数组，用于存储包含 Hessian 矩阵的直方图
    hist_ghc_perm = np.zeros((1, n_bins), dtype=HISTOGRAM_DTYPE)
    # 调用函数 _build_histogram，构建包含 Hessian 矩阵的直方图
    _build_histogram(
        0,  # 参数0：指定直方图索引
        sample_indices[permutation],  # 使用打乱顺序的样本索引
        binned_feature,  # 经过分箱处理的特征值
        ordered_gradients[
@pytest.mark.parametrize("constant_hessian", [True, False])
# 定义测试函数，使用参数化测试，constant_hessian 分别为 True 和 False
def test_unrolled_equivalent_to_naive(constant_hessian):
    # 确保不同的展开直方图计算与朴素方法给出相同的结果。
    
    # 设定随机数生成器
    rng = np.random.RandomState(42)
    
    # 样本数和直方图柱数
    n_samples = 10
    n_bins = 5
    
    # 生成样本索引，使用 uint32 类型
    sample_indices = np.arange(n_samples).astype(np.uint32)
    
    # 生成随机分箱特征，使用 uint8 类型
    binned_feature = rng.randint(0, n_bins - 1, size=n_samples, dtype=np.uint8)
    
    # 生成有序梯度，使用 G_H_DTYPE 类型
    ordered_gradients = rng.randn(n_samples).astype(G_H_DTYPE)
    
    # 根据 constant_hessian 的值生成有序 Hessian 矩阵，使用 G_H_DTYPE 类型
    if constant_hessian:
        ordered_hessians = np.ones(n_samples, dtype=G_H_DTYPE)
    else:
        ordered_hessians = rng.lognormal(size=n_samples).astype(G_H_DTYPE)

    # 初始化根直方图
    hist_gc_root = np.zeros((1, n_bins), dtype=HISTOGRAM_DTYPE)
    hist_ghc_root = np.zeros((1, n_bins), dtype=HISTOGRAM_DTYPE)
    
    # 初始化直方图
    hist_gc = np.zeros((1, n_bins), dtype=HISTOGRAM_DTYPE)
    hist_ghc = np.zeros((1, n_bins), dtype=HISTOGRAM_DTYPE)
    hist_naive = np.zeros((1, n_bins), dtype=HISTOGRAM_DTYPE)

    # 构建无 Hessian 的根直方图
    _build_histogram_root_no_hessian(0, binned_feature, ordered_gradients, hist_gc_root)
    
    # 构建有 Hessian 的根直方图
    _build_histogram_root(
        0, binned_feature, ordered_gradients, ordered_hessians, hist_ghc_root
    )
    
    # 构建无 Hessian 的直方图
    _build_histogram_no_hessian(
        0, sample_indices, binned_feature, ordered_gradients, hist_gc
    )
    
    # 构建有 Hessian 的直方图
    _build_histogram(
        0, sample_indices, binned_feature, ordered_gradients, ordered_hessians, hist_ghc
    )
    
    # 构建朴素方法的直方图
    _build_histogram_naive(
        0,
        sample_indices,
        binned_feature,
        ordered_gradients,
        ordered_hessians,
        hist_naive,
    )

    # 从多维数组变为一维数组，以便后续比较
    hist_naive = hist_naive[0]
    hist_gc_root = hist_gc_root[0]
    hist_ghc_root = hist_ghc_root[0]
    hist_gc = hist_gc[0]
    hist_ghc = hist_ghc[0]
    
    # 比较各种直方图的属性
    for hist in (hist_gc_root, hist_ghc_root, hist_gc, hist_ghc):
        assert_array_equal(hist["count"], hist_naive["count"])
        assert_allclose(hist["sum_gradients"], hist_naive["sum_gradients"])
    
    for hist in (hist_ghc_root, hist_ghc):
        assert_allclose(hist["sum_hessians"], hist_naive["sum_hessians"])
    
    for hist in (hist_gc_root, hist_gc):
        assert_array_equal(hist["sum_hessians"], np.zeros(n_bins))


@pytest.mark.parametrize("constant_hessian", [True, False])
# 定义测试函数，使用参数化测试，constant_hessian 分别为 True 和 False
def test_hist_subtraction(constant_hessian):
    # 确保直方图减法技巧给出与传统方法相同的结果。
    
    # 设定随机数生成器
    rng = np.random.RandomState(42)
    
    # 样本数和直方图柱数
    n_samples = 10
    n_bins = 5
    
    # 生成样本索引，使用 uint32 类型
    sample_indices = np.arange(n_samples).astype(np.uint32)
    
    # 生成随机分箱特征，使用 uint8 类型
    binned_feature = rng.randint(0, n_bins - 1, size=n_samples, dtype=np.uint8)
    
    # 生成有序梯度，使用 G_H_DTYPE 类型
    ordered_gradients = rng.randn(n_samples).astype(G_H_DTYPE)
    
    # 根据 constant_hessian 的值生成有序 Hessian 矩阵，使用 G_H_DTYPE 类型
    if constant_hessian:
        ordered_hessians = np.ones(n_samples, dtype=G_H_DTYPE)
    else:
        ordered_hessians = rng.lognormal(size=n_samples).astype(G_H_DTYPE)

    # 初始化父直方图
    hist_parent = np.zeros((1, n_bins), dtype=HISTOGRAM_DTYPE)
    # 如果常数 Hessian 为真，则调用没有 Hessian 的直方图构建函数
    _build_histogram_no_hessian(
        0, sample_indices, binned_feature, ordered_gradients, hist_parent
    )
else:
    # 否则，调用包含 Hessian 的直方图构建函数
    _build_histogram(
        0,
        sample_indices,
        binned_feature,
        ordered_gradients,
        ordered_hessians,
        hist_parent,
    )

# 生成一个布尔掩码数组，用于选择样本的左子集
mask = rng.randint(0, 2, n_samples).astype(bool)

# 根据掩码选择样本的左子集和对应的梯度与Hessian
sample_indices_left = sample_indices[mask]
ordered_gradients_left = ordered_gradients[mask]
ordered_hessians_left = ordered_hessians[mask]
# 初始化左子集的直方图
hist_left = np.zeros((1, n_bins), dtype=HISTOGRAM_DTYPE)

# 根据是否常数 Hessian 调用对应的直方图构建函数来填充左子集的直方图
if constant_hessian:
    _build_histogram_no_hessian(
        0, sample_indices_left, binned_feature, ordered_gradients_left, hist_left
    )
else:
    _build_histogram(
        0,
        sample_indices_left,
        binned_feature,
        ordered_gradients_left,
        ordered_hessians_left,
        hist_left,
    )

# 根据掩码选择样本的右子集和对应的梯度与Hessian
sample_indices_right = sample_indices[~mask]
ordered_gradients_right = ordered_gradients[~mask]
ordered_hessians_right = ordered_hessians[~mask]
# 初始化右子集的直方图
hist_right = np.zeros((1, n_bins), dtype=HISTOGRAM_DTYPE)

# 根据是否常数 Hessian 调用对应的直方图构建函数来填充右子集的直方图
if constant_hessian:
    _build_histogram_no_hessian(
        0, sample_indices_right, binned_feature, ordered_gradients_right, hist_right
    )
else:
    _build_histogram(
        0,
        sample_indices_right,
        binned_feature,
        ordered_gradients_right,
        ordered_hessians_right,
        hist_right,
    )

# 复制父节点的直方图到左子集和右子集的副本
hist_left_sub = np.copy(hist_parent)
hist_right_sub = np.copy(hist_parent)

# 在左右子集的直方图之间执行直方图减法操作
_subtract_histograms(0, n_bins, hist_left_sub, hist_right)
_subtract_histograms(0, n_bins, hist_right_sub, hist_left)

# 验证左右子集的直方图是否符合预期的数值误差范围
for key in ("count", "sum_hessians", "sum_gradients"):
    assert_allclose(hist_left[key], hist_left_sub[key], rtol=1e-6)
    assert_allclose(hist_right[key], hist_right_sub[key], rtol=1e-6)
```