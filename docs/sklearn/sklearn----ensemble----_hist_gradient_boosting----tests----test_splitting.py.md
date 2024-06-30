# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\_hist_gradient_boosting\tests\test_splitting.py`

```
# 导入所需的库和模块
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于测试框架
from numpy.testing import assert_array_equal  # 导入 NumPy 的数组比较函数

# 导入 sklearn 中的模块和类
from sklearn.ensemble._hist_gradient_boosting.common import (
    G_H_DTYPE,  # 导入 G_H_DTYPE 数据类型常量
    HISTOGRAM_DTYPE,  # 导入 HISTOGRAM_DTYPE 数据类型常量
    X_BINNED_DTYPE,  # 导入 X_BINNED_DTYPE 数据类型常量
    MonotonicConstraint,  # 导入 MonotonicConstraint 类
)
from sklearn.ensemble._hist_gradient_boosting.histogram import HistogramBuilder  # 导入直方图构建器类
from sklearn.ensemble._hist_gradient_boosting.splitting import (
    Splitter,  # 导入 Splitter 类
    compute_node_value,  # 导入 compute_node_value 函数
)
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads  # 导入多线程相关函数
from sklearn.utils._testing import skip_if_32bit  # 导入一个条件跳过测试的装饰器函数

# 获取有效的 OpenMP 线程数
n_threads = _openmp_effective_n_threads()

# 使用 pytest.mark.parametrize 装饰器，定义参数化测试函数 test_histogram_split
@pytest.mark.parametrize("n_bins", [3, 32, 256])
def test_histogram_split(n_bins):
    rng = np.random.RandomState(42)  # 创建随机数生成器对象 rng
    feature_idx = 0  # 设置特征索引为 0
    l2_regularization = 0  # L2 正则化参数设为 0
    min_hessian_to_split = 1e-3  # 分裂节点所需的最小 Hessian 值设为 0.001
    min_samples_leaf = 1  # 叶子节点的最小样本数设为 1
    min_gain_to_split = 0.0  # 分裂节点所需的最小增益设为 0.0

    # 创建一个 n_bins × 1 的 Fortran 连续存储的二维数组 X_binned
    X_binned = np.asfortranarray(
        rng.randint(0, n_bins - 1, size=(int(1e4), 1)), dtype=X_BINNED_DTYPE
    )

    # 提取出 X_binned 的第 feature_idx 列作为 binned_feature
    binned_feature = X_binned.T[feature_idx]

    # 创建一个 uint32 类型的数组 sample_indices，其长度与 binned_feature 相同
    sample_indices = np.arange(binned_feature.shape[0], dtype=np.uint32)

    # 创建一个与 binned_feature 大小相同的全为 1 的数组 ordered_hessians，数据类型为 G_H_DTYPE
    ordered_hessians = np.ones_like(binned_feature, dtype=G_H_DTYPE)

    # 将 ordered_hessians 赋值给 all_hessians
    all_hessians = ordered_hessians

    # 计算 all_hessians 的总和，存储在 sum_hessians 中
    sum_hessians = all_hessians.sum()

    # 初始化 hessians_are_constant 为 False，表示 Hessian 值不全相等
    hessians_are_constant = False
    # 对每个真实的二进制划分点进行迭代，范围是从1到n_bins-2
    for true_bin in range(1, n_bins - 2):
        # 对每个划分点分别考虑正负两种方向
        for sign in [-1, 1]:
            # 创建一个与 binned_feature 相同形状的数组，填充为 sign 值，数据类型为 G_H_DTYPE
            ordered_gradients = np.full_like(binned_feature, sign, dtype=G_H_DTYPE)
            # 将小于等于 true_bin 的部分乘以 -1，实现梯度方向的调整
            ordered_gradients[binned_feature <= true_bin] *= -1
            # 将所有梯度保存在 all_gradients 中
            all_gradients = ordered_gradients
            # 计算所有梯度的和
            sum_gradients = all_gradients.sum()

            # 使用 HistogramBuilder 类创建一个直方图构建器对象
            builder = HistogramBuilder(
                X_binned,
                n_bins,
                all_gradients,
                all_hessians,
                hessians_are_constant,
                n_threads,
            )
            # 创建一个长度为 X_binned.shape[1] 的整数数组，每个元素为 n_bins-1，表示非缺失值的 bin 数量
            n_bins_non_missing = np.array(
                [n_bins - 1] * X_binned.shape[1], dtype=np.uint32
            )
            # 创建一个长度为 X_binned.shape[1] 的布尔数组，每个元素为 False，表示没有缺失值
            has_missing_values = np.array([False] * X_binned.shape[1], dtype=np.uint8)
            # 创建一个长度为 X_binned.shape[1] 的整数数组，每个元素为 MonotonicConstraint.NO_CST，表示无单调性约束
            monotonic_cst = np.array(
                [MonotonicConstraint.NO_CST] * X_binned.shape[1], dtype=np.int8
            )
            # 创建一个与 monotonic_cst 形状相同的无符号整数数组，每个元素为 0，表示不是分类特征
            is_categorical = np.zeros_like(monotonic_cst, dtype=np.uint8)
            # 缺失值的 bin 索引为 n_bins - 1
            missing_values_bin_idx = n_bins - 1
            # 使用 Splitter 类创建一个分裂器对象
            splitter = Splitter(
                X_binned,
                n_bins_non_missing,
                missing_values_bin_idx,
                has_missing_values,
                is_categorical,
                monotonic_cst,
                l2_regularization,
                min_hessian_to_split,
                min_samples_leaf,
                min_gain_to_split,
                hessians_are_constant,
            )

            # 使用 builder 对象计算样本索引为 sample_indices 的直方图
            histograms = builder.compute_histograms_brute(sample_indices)
            # 计算节点的值，用于评估分裂的质量
            value = compute_node_value(
                sum_gradients, sum_hessians, -np.inf, np.inf, l2_regularization
            )
            # 在 splitter 对象中查找节点分裂信息
            split_info = splitter.find_node_split(
                sample_indices.shape[0], histograms, sum_gradients, sum_hessians, value
            )

            # 确保找到的分裂点 bin_idx 与当前迭代的 true_bin 一致
            assert split_info.bin_idx == true_bin
            # 确保分裂的增益大于等于 0
            assert split_info.gain >= 0
            # 确保分裂特征的索引与当前特征索引相同
            assert split_info.feature_idx == feature_idx
            # 确保左右子节点样本数之和等于总样本数
            assert (
                split_info.n_samples_left + split_info.n_samples_right
                == sample_indices.shape[0]
            )
            # 对于恒定的 Hessian 值为每个样本为 1
            assert split_info.n_samples_left == split_info.sum_hessian_left
# 标记为跳过如果是32位系统的测试函数装饰器，如果是32位系统则跳过执行这个测试
@skip_if_32bit
# 使用参数化测试，测试constant_hessian参数为True和False时的函数
@pytest.mark.parametrize("constant_hessian", [True, False])
# 定义测试函数，用于验证梯度和Hessian值在不同位置的一致性：
def test_gradient_and_hessian_sanity(constant_hessian):
    # 创建随机数生成器对象，种子为42
    rng = np.random.RandomState(42)

    # 定义测试数据的维度和规模
    n_bins = 10
    n_features = 20
    n_samples = 500

    # 定义模型参数
    l2_regularization = 0.0
    min_hessian_to_split = 1e-3
    min_samples_leaf = 1
    min_gain_to_split = 0.0

    # 生成随机的特征分箱数据，类型为X_BINNED_DTYPE
    X_binned = rng.randint(
        0, n_bins, size=(n_samples, n_features), dtype=X_BINNED_DTYPE
    )
    # 将特征分箱数据转换为Fortran风格数组
    X_binned = np.asfortranarray(X_binned)

    # 创建样本索引数组
    sample_indices = np.arange(n_samples, dtype=np.uint32)

    # 生成随机的梯度数据，类型为G_H_DTYPE，并计算所有梯度之和
    all_gradients = rng.randn(n_samples).astype(G_H_DTYPE)
    sum_gradients = all_gradients.sum()

    # 根据constant_hessian参数生成随机的Hessian数据，类型为G_H_DTYPE，并计算所有Hessian之和
    if constant_hessian:
        all_hessians = np.ones(1, dtype=G_H_DTYPE)
        sum_hessians = 1 * n_samples
    else:
        all_hessians = rng.lognormal(size=n_samples).astype(G_H_DTYPE)
        sum_hessians = all_hessians.sum()

    # 创建HistogramBuilder对象，用于计算直方图
    builder = HistogramBuilder(
        X_binned, n_bins, all_gradients, all_hessians, constant_hessian, n_threads
    )

    # 创建非缺失值分箱数数组和是否具有缺失值数组等
    n_bins_non_missing = np.array([n_bins - 1] * X_binned.shape[1], dtype=np.uint32)
    has_missing_values = np.array([False] * X_binned.shape[1], dtype=np.uint8)
    monotonic_cst = np.array(
        [MonotonicConstraint.NO_CST] * X_binned.shape[1], dtype=np.int8
    )
    is_categorical = np.zeros_like(monotonic_cst, dtype=np.uint8)
    missing_values_bin_idx = n_bins - 1

    # 创建Splitter对象，用于节点分裂操作
    splitter = Splitter(
        X_binned,
        n_bins_non_missing,
        missing_values_bin_idx,
        has_missing_values,
        is_categorical,
        monotonic_cst,
        l2_regularization,
        min_hessian_to_split,
        min_samples_leaf,
        min_gain_to_split,
        constant_hessian,
    )

    # 计算父节点的直方图
    hists_parent = builder.compute_histograms_brute(sample_indices)
    # 计算父节点的节点值
    value_parent = compute_node_value(
        sum_gradients, sum_hessians, -np.inf, np.inf, l2_regularization
    )
    # 查找节点分裂的最优位置
    si_parent = splitter.find_node_split(
        n_samples, hists_parent, sum_gradients, sum_hessians, value_parent
    )
    # 获取左右子节点的样本索引
    sample_indices_left, sample_indices_right, _ = splitter.split_indices(
        si_parent, sample_indices
    )

    # 计算左子节点的直方图
    hists_left = builder.compute_histograms_brute(sample_indices_left)
    # 计算左子节点的节点值
    value_left = compute_node_value(
        si_parent.sum_gradient_left,
        si_parent.sum_hessian_left,
        -np.inf,
        np.inf,
        l2_regularization,
    )
    # 计算右子节点的直方图
    hists_right = builder.compute_histograms_brute(sample_indices_right)
    # 计算右子节点的节点值，用于划分决策树节点
    value_right = compute_node_value(
        si_parent.sum_gradient_right,  # 右子节点的梯度总和
        si_parent.sum_hessian_right,   # 右子节点的Hessian矩阵总和
        -np.inf,                       # 节点值的下限
        np.inf,                        # 节点值的上限
        l2_regularization,             # L2正则化参数
    )
    # 找到左子节点的最佳分裂
    si_left = splitter.find_node_split(
        n_samples,                     # 样本数量
        hists_left,                    # 左子节点的直方图数据
        si_parent.sum_gradient_left,   # 左子节点的梯度总和
        si_parent.sum_hessian_left,    # 左子节点的Hessian矩阵总和
        value_left,                    # 左子节点的节点值
    )
    # 找到右子节点的最佳分裂
    si_right = splitter.find_node_split(
        n_samples,                     # 样本数量
        hists_right,                   # 右子节点的直方图数据
        si_parent.sum_gradient_right,  # 右子节点的梯度总和
        si_parent.sum_hessian_right,   # 右子节点的Hessian矩阵总和
        value_right,                   # 右子节点的节点值
    )

    # 确保 si.sum_gradient_left + si.sum_gradient_right 的值与期望值相等，同样适用于 hessians
    for si, indices in (
        (si_parent, sample_indices),         # 对于父节点，使用所有样本的索引
        (si_left, sample_indices_left),      # 对于左子节点，使用左子节点的样本索引
        (si_right, sample_indices_right),    # 对于右子节点，使用右子节点的样本索引
    ):
        gradient = si.sum_gradient_right + si.sum_gradient_left  # 计算梯度总和
        expected_gradient = all_gradients[indices].sum()         # 计算期望的梯度总和
        hessian = si.sum_hessian_right + si.sum_hessian_left     # 计算Hessian矩阵总和
        if constant_hessian:
            expected_hessian = indices.shape[0] * all_hessians[0]  # 如果Hessian矩阵是常数，则直接计算期望值
        else:
            expected_hessian = all_hessians[indices].sum()         # 否则，计算期望的Hessian矩阵总和

        assert np.isclose(gradient, expected_gradient)   # 断言梯度总和与期望梯度总和近似相等
        assert np.isclose(hessian, expected_hessian)     # 断言Hessian矩阵总和与期望Hessian矩阵总和近似相等

    # 确保所有特征的直方图中的梯度总和相同，并且等于其期望值
    hists_parent = np.asarray(hists_parent, dtype=HISTOGRAM_DTYPE)  # 转换父节点直方图数据为指定数据类型
    hists_left = np.asarray(hists_left, dtype=HISTOGRAM_DTYPE)      # 转换左子节点直方图数据为指定数据类型
    hists_right = np.asarray(hists_right, dtype=HISTOGRAM_DTYPE)    # 转换右子节点直方图数据为指定数据类型
    for hists, indices in (
        (hists_parent, sample_indices),         # 对于父节点的直方图数据，使用所有样本的索引
        (hists_left, sample_indices_left),      # 对于左子节点的直方图数据，使用左子节点的样本索引
        (hists_right, sample_indices_right),    # 对于右子节点的直方图数据，使用右子节点的样本索引
    ):
        # 注意：梯度和Hessian矩阵的形状是 (n_features,)，我们将其与标量进行比较。
        # 这样做的好处是还可以确保跨特征的所有条目都是相等的。
        gradients = hists["sum_gradients"].sum(axis=1)  # 计算每个特征的梯度总和
        expected_gradient = all_gradients[indices].sum()  # 计算期望的梯度总和
        hessians = hists["sum_hessians"].sum(axis=1)     # 计算每个特征的Hessian矩阵总和
        if constant_hessian:
            expected_hessian = 0.0  # 如果Hessian矩阵是常数，则期望的Hessian矩阵总和为0
        else:
            expected_hessian = all_hessians[indices].sum()  # 否则，计算期望的Hessian矩阵总和

        assert np.allclose(gradients, expected_gradient)   # 断言每个特征的梯度总和与期望梯度总和近似相等
        assert np.allclose(hessians, expected_hessian)     # 断言每个特征的Hessian矩阵总和与期望Hessian矩阵总和近似相等
def test_split_indices():
    # 检查 split_indices 返回正确的分割结果，并验证 splitter.partition 是否一致。
    rng = np.random.RandomState(421)  # 创建一个指定种子的随机数生成器

    n_bins = 5  # 定义 bin 的数量
    n_samples = 10  # 定义样本数量
    l2_regularization = 0.0  # L2 正则化参数
    min_hessian_to_split = 1e-3  # 分割所需的最小 Hessian 值
    min_samples_leaf = 1  # 叶子节点的最小样本数
    min_gain_to_split = 0.0  # 分割所需的最小增益

    # split 将在特征 1 和 bin 3 上发生
    X_binned = [
        [0, 0],
        [0, 3],
        [0, 4],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 4],
        [0, 0],
        [0, 4],
    ]
    X_binned = np.asfortranarray(X_binned, dtype=X_BINNED_DTYPE)  # 转换为 Fortran 风格的数组
    sample_indices = np.arange(n_samples, dtype=np.uint32)  # 创建一个样本索引数组
    all_gradients = rng.randn(n_samples).astype(G_H_DTYPE)  # 随机生成梯度值数组
    all_hessians = np.ones(1, dtype=G_H_DTYPE)  # 创建全为 1 的 Hessian 数组
    sum_gradients = all_gradients.sum()  # 计算梯度总和
    sum_hessians = 1 * n_samples  # 计算 Hessian 总和
    hessians_are_constant = True  # 指示 Hessian 是否常数

    # 创建 HistogramBuilder 对象，用于计算直方图
    builder = HistogramBuilder(
        X_binned, n_bins, all_gradients, all_hessians, hessians_are_constant, n_threads
    )
    n_bins_non_missing = np.array([n_bins] * X_binned.shape[1], dtype=np.uint32)  # 创建非缺失值 bin 数组
    has_missing_values = np.array([False] * X_binned.shape[1], dtype=np.uint8)  # 创建缺失值标记数组
    monotonic_cst = np.array(
        [MonotonicConstraint.NO_CST] * X_binned.shape[1], dtype=np.int8
    )  # 创建单调约束数组
    is_categorical = np.zeros_like(monotonic_cst, dtype=np.uint8)  # 创建分类特征标记数组
    missing_values_bin_idx = n_bins - 1  # 缺失值所在的 bin 索引
    # 创建 Splitter 对象，用于执行分割操作
    splitter = Splitter(
        X_binned,
        n_bins_non_missing,
        missing_values_bin_idx,
        has_missing_values,
        is_categorical,
        monotonic_cst,
        l2_regularization,
        min_hessian_to_split,
        min_samples_leaf,
        min_gain_to_split,
        hessians_are_constant,
    )

    assert np.all(sample_indices == splitter.partition)  # 断言样本索引与 splitter.partition 相等

    histograms = builder.compute_histograms_brute(sample_indices)  # 计算直方图
    value = compute_node_value(
        sum_gradients, sum_hessians, -np.inf, np.inf, l2_regularization
    )  # 计算节点的值
    si_root = splitter.find_node_split(
        n_samples, histograms, sum_gradients, sum_hessians, value
    )  # 找到节点的最佳分割

    # 对最佳分割的合理性进行检查
    assert si_root.feature_idx == 1  # 断言最佳分割的特征索引为 1
    assert si_root.bin_idx == 3  # 断言最佳分割的 bin 索引为 3

    samples_left, samples_right, position_right = splitter.split_indices(
        si_root, splitter.partition
    )  # 执行分割操作，获取左右子节点的样本索引
    assert set(samples_left) == set([0, 1, 3, 4, 5, 6, 8])  # 断言左子节点的样本索引正确
    assert set(samples_right) == set([2, 7, 9])  # 断言右子节点的样本索引正确

    assert list(samples_left) == list(splitter.partition[:position_right])  # 断言左子节点的样本索引列表正确
    assert list(samples_right) == list(splitter.partition[position_right:])  # 断言右子节点的样本索引列表正确

    # 检查分割后的样本索引大小与预期统计数一致性
    assert samples_left.shape[0] == si_root.n_samples_left  # 断言左子节点样本数正确
    assert samples_right.shape[0] == si_root.n_samples_right  # 断言右子节点样本数正确


def test_min_gain_to_split():
    # 尝试对纯节点进行分割（所有梯度值相等，Hessian 值也相等）
    # 这里需要添加进一步的测试代码
    # 使用种子值 42 初始化随机数生成器
    rng = np.random.RandomState(42)
    # L2 正则化参数设为 0
    l2_regularization = 0
    # 最小的 Hessian 值设为 0，表示节点分裂的最小条件之一
    min_hessian_to_split = 0
    # 叶子节点最小样本数设为 1
    min_samples_leaf = 1
    # 最小分裂增益设为 0.0，确保节点不会以 0 的增益值分裂
    min_gain_to_split = 0.0
    # 离散化时的箱子数量设为 255
    n_bins = 255
    # 样本数设为 100
    n_samples = 100
    
    # 使用随机数生成器生成一个 n_samples 行 1 列的随机整数矩阵，作为 X_binned
    X_binned = np.asfortranarray(
        rng.randint(0, n_bins, size=(n_samples, 1)), dtype=X_BINNED_DTYPE
    )
    
    # 从 X_binned 中取出第一列作为 binned_feature
    binned_feature = X_binned[:, 0]
    
    # 生成一个包含 n_samples 个元素的索引数组
    sample_indices = np.arange(n_samples, dtype=np.uint32)
    
    # 创建一个与 binned_feature 大小相同的全为 1 的数组，作为所有梯度的数组
    all_hessians = np.ones_like(binned_feature, dtype=G_H_DTYPE)
    all_gradients = np.ones_like(binned_feature, dtype=G_H_DTYPE)
    
    # 计算所有梯度的总和
    sum_gradients = all_gradients.sum()
    # 计算所有 Hessian 的总和
    sum_hessians = all_hessians.sum()
    
    # 表示所有的 Hessian 值是否都相等
    hessians_are_constant = False
    
    # 使用 HistogramBuilder 类构建直方图
    builder = HistogramBuilder(
        X_binned, n_bins, all_gradients, all_hessians, hessians_are_constant, n_threads
    )
    
    # 创建一个数组，每个元素为 n_bins - 1，表示非缺失值的箱子数量
    n_bins_non_missing = np.array([n_bins - 1] * X_binned.shape[1], dtype=np.uint32)
    
    # 创建一个数组，每个元素为 False，表示是否有缺失值
    has_missing_values = np.array([False] * X_binned.shape[1], dtype=np.uint8)
    
    # 创建一个数组，每个元素为 MonotonicConstraint.NO_CST，表示无单调性约束
    monotonic_cst = np.array(
        [MonotonicConstraint.NO_CST] * X_binned.shape[1], dtype=np.int8
    )
    
    # 创建一个与 monotonic_cst 大小相同的全零数组，表示特征是否是分类特征
    is_categorical = np.zeros_like(monotonic_cst, dtype=np.uint8)
    
    # 缺失值的箱子索引设为 n_bins - 1
    missing_values_bin_idx = n_bins - 1
    
    # 使用 Splitter 类初始化一个节点分裂器
    splitter = Splitter(
        X_binned,
        n_bins_non_missing,
        missing_values_bin_idx,
        has_missing_values,
        is_categorical,
        monotonic_cst,
        l2_regularization,
        min_hessian_to_split,
        min_samples_leaf,
        min_gain_to_split,
        hessians_are_constant,
    )
    
    # 计算直方图
    histograms = builder.compute_histograms_brute(sample_indices)
    
    # 计算节点的值
    value = compute_node_value(
        sum_gradients, sum_hessians, -np.inf, np.inf, l2_regularization
    )
    
    # 在给定直方图的情况下，找到节点的最佳分裂点
    split_info = splitter.find_node_split(
        n_samples, histograms, sum_gradients, sum_hessians, value
    )
    
    # 断言节点的分裂增益为 -1，用于验证算法的正确性
    assert split_info.gain == -1
@pytest.mark.parametrize(
    (
        "X_binned, all_gradients, has_missing_values, n_bins_non_missing, "
        " expected_split_on_nan, expected_bin_idx, expected_go_to_left"
    ),
    ],
)
def test_splitting_missing_values(
    X_binned,
    all_gradients,
    has_missing_values,
    n_bins_non_missing,
    expected_split_on_nan,
    expected_bin_idx,
    expected_go_to_left,
):
    # 确保缺失值得到正确支持。
    # 构建一个人工示例，使得在没有缺失值时最佳分割位于 bin_idx=3 的位置。
    # 然后引入缺失值，并且：
    #   - 确保选择的 bin 是正确的（find_best_bin()）：仍然是相同的分割，尽管 bin 的索引可能会改变
    #   - 确保缺失值被映射到正确的子节点（split_indices()）

    # 计算样本的数量和最大的 bin 编号
    n_bins = max(X_binned) + 1
    n_samples = len(X_binned)
    l2_regularization = 0.0
    min_hessian_to_split = 1e-3
    min_samples_leaf = 1
    min_gain_to_split = 0.0

    # 创建样本索引数组
    sample_indices = np.arange(n_samples, dtype=np.uint32)
    # 将 X_binned 转换为指定类型的数组，并且重新塑形为一列
    X_binned = np.array(X_binned, dtype=X_BINNED_DTYPE).reshape(-1, 1)
    X_binned = np.asfortranarray(X_binned)
    # 将 all_gradients 转换为指定类型的数组
    all_gradients = np.array(all_gradients, dtype=G_H_DTYPE)
    # 创建一个标志是否有缺失值的数组
    has_missing_values = np.array([has_missing_values], dtype=np.uint8)
    # 创建所有 hessians 值为 1 的数组
    all_hessians = np.ones(1, dtype=G_H_DTYPE)
    # 计算所有梯度的总和和 hessians 的总和
    sum_gradients = all_gradients.sum()
    sum_hessians = 1 * n_samples
    hessians_are_constant = True

    # 使用 HistogramBuilder 类创建直方图构建器对象
    builder = HistogramBuilder(
        X_binned, n_bins, all_gradients, all_hessians, hessians_are_constant, n_threads
    )

    # 创建非缺失值 bin 的数量的数组
    n_bins_non_missing = np.array([n_bins_non_missing], dtype=np.uint32)
    # 创建一个指示单调约束的数组
    monotonic_cst = np.array(
        [MonotonicConstraint.NO_CST] * X_binned.shape[1], dtype=np.int8
    )
    # 创建一个指示是否是分类特征的数组
    is_categorical = np.zeros_like(monotonic_cst, dtype=np.uint8)
    # 缺失值的 bin 索引为 n_bins - 1
    missing_values_bin_idx = n_bins - 1
    # 使用 Splitter 类创建分割器对象
    splitter = Splitter(
        X_binned,
        n_bins_non_missing,
        missing_values_bin_idx,
        has_missing_values,
        is_categorical,
        monotonic_cst,
        l2_regularization,
        min_hessian_to_split,
        min_samples_leaf,
        min_gain_to_split,
        hessians_are_constant,
    )

    # 计算直方图
    histograms = builder.compute_histograms_brute(sample_indices)
    # 计算节点的值
    value = compute_node_value(
        sum_gradients, sum_hessians, -np.inf, np.inf, l2_regularization
    )
    # 找到节点的分割信息
    split_info = splitter.find_node_split(
        n_samples, histograms, sum_gradients, sum_hessians, value
    )

    # 断言分割的 bin_idx 是否符合预期
    assert split_info.bin_idx == expected_bin_idx
    # 如果有缺失值，断言缺失值的处理方向是否符合预期
    if has_missing_values:
        assert split_info.missing_go_to_left == expected_go_to_left

    # 判断是否在缺失值上进行分割，并且检查其是否符合预期
    split_on_nan = split_info.bin_idx == n_bins_non_missing[0] - 1
    assert split_on_nan == expected_split_on_nan

    # 确保分割被正确计算。
    # 这也确保缺失值被正确分配到 split_indices() 中的正确子节点。
    # 使用分裂器对象和分裂信息，获取左右子样本的索引
    samples_left, samples_right, _ = splitter.split_indices(
        split_info, splitter.partition
    )

    if not expected_split_on_nan:
        # 当不按 NaN 分裂时，左右子样本的分裂应始终保持一致。
        assert set(samples_left) == set([0, 1, 2, 3])
        assert set(samples_right) == set([4, 5, 6, 7, 8, 9])
    else:
        # 当按 NaN 分裂时，具有缺失值的样本始终映射到右子节点。
        # 找到所有具有缺失值的样本的索引
        missing_samples_indices = np.flatnonzero(
            np.array(X_binned) == missing_values_bin_idx
        )
        # 找到所有没有缺失值的样本的索引
        non_missing_samples_indices = np.flatnonzero(
            np.array(X_binned) != missing_values_bin_idx
        )

        assert set(samples_right) == set(missing_samples_indices)
        assert set(samples_left) == set(non_missing_samples_indices)
@pytest.mark.parametrize(
    "X_binned, has_missing_values, n_bins_non_missing, ",
    [
        # one category
        ([0] * 20, False, 1),
        # all categories appear less than MIN_CAT_SUPPORT (hardcoded to 10)
        ([0] * 9 + [1] * 8, False, 2),
        # only one category appears more than MIN_CAT_SUPPORT
        ([0] * 12 + [1] * 8, False, 2),
        # missing values + category appear less than MIN_CAT_SUPPORT
        # 9 is missing
        ([0] * 9 + [1] * 8 + [9] * 4, True, 2),
        # no non-missing category
        ([9] * 11, True, 0),
    ],
)
def test_splitting_categorical_cat_smooth(
    X_binned, has_missing_values, n_bins_non_missing
):
    # Checks categorical splits are correct when the MIN_CAT_SUPPORT constraint
    # isn't respected: there are no splits

    # Calculate number of bins based on maximum value in X_binned
    n_bins = max(X_binned) + 1
    # Get number of samples
    n_samples = len(X_binned)
    # Convert X_binned to a 2D array of X_BINNED_DTYPE and transpose it
    X_binned = np.array([X_binned], dtype=X_BINNED_DTYPE).T
    # Convert X_binned to Fortran-contiguous array
    X_binned = np.asfortranarray(X_binned)

    # Initialize regularization parameters
    l2_regularization = 0.0
    min_hessian_to_split = 1e-3
    min_samples_leaf = 1
    min_gain_to_split = 0.0

    # Create an array of sample indices
    sample_indices = np.arange(n_samples, dtype=np.uint32)
    # Create an array of all_gradients filled with ones
    all_gradients = np.ones(n_samples, dtype=G_H_DTYPE)
    # Convert has_missing_values to a uint8 array
    has_missing_values = np.array([has_missing_values], dtype=np.uint8)
    # Create an array all_hessians filled with ones
    all_hessians = np.ones(1, dtype=G_H_DTYPE)
    # Calculate sum of gradients
    sum_gradients = all_gradients.sum()
    # Set sum_hessians to the number of samples
    sum_hessians = n_samples
    # Indicate if hessians are constant
    hessians_are_constant = True

    # Create a HistogramBuilder object
    builder = HistogramBuilder(
        X_binned, n_bins, all_gradients, all_hessians, hessians_are_constant, n_threads
    )

    # Convert n_bins_non_missing to a uint32 array
    n_bins_non_missing = np.array([n_bins_non_missing], dtype=np.uint32)
    # Create monotonic_cst array with NO_CST for each column in X_binned
    monotonic_cst = np.array(
        [MonotonicConstraint.NO_CST] * X_binned.shape[1], dtype=np.int8
    )
    # Create is_categorical array with ones for each column in X_binned
    is_categorical = np.ones_like(monotonic_cst, dtype=np.uint8)
    # Set missing_values_bin_idx to n_bins - 1
    missing_values_bin_idx = n_bins - 1

    # Create a Splitter object
    splitter = Splitter(
        X_binned,
        n_bins_non_missing,
        missing_values_bin_idx,
        has_missing_values,
        is_categorical,
        monotonic_cst,
        l2_regularization,
        min_hessian_to_split,
        min_samples_leaf,
        min_gain_to_split,
        hessians_are_constant,
    )

    # Compute histograms using brute-force method
    histograms = builder.compute_histograms_brute(sample_indices)
    # Compute node value
    value = compute_node_value(
        sum_gradients, sum_hessians, -np.inf, np.inf, l2_regularization
    )
    # Find node split
    split_info = splitter.find_node_split(
        n_samples, histograms, sum_gradients, sum_hessians, value
    )

    # Assert no split found
    assert split_info.gain == -1


def _assert_categories_equals_bitset(categories, bitset):
    # assert that the bitset exactly corresponds to the categories
    # bitset is assumed to be an array of 8 uint32 elements

    # Initialize expected_bitset with zeros
    expected_bitset = np.zeros(8, dtype=np.uint32)
    # Set bits in expected_bitset based on categories
    for cat in categories:
        idx = cat // 32
        shift = cat % 32
        expected_bitset[idx] |= 1 << shift

    # Assert equality between expected_bitset and bitset
    assert_array_equal(expected_bitset, bitset)


@pytest.mark.parametrize(
    (
        "X_binned, all_gradients, expected_categories_left, n_bins_non_missing,"
        "missing_values_bin_idx, has_missing_values, expected_missing_go_to_left"
    ),

这部分代码定义了一个包含多个字符串元素的元组。这些字符串表示一个包含多个元素的元组的元素。
def test_splitting_categorical_sanity(
    X_binned,  # 输入的离散化后的特征向量
    all_gradients,  # 所有样本的梯度数组
    expected_categories_left,  # 预期左子节点的类别集合
    n_bins_non_missing,  # 非缺失值的箱子数量
    missing_values_bin_idx,  # 缺失值所在箱子的索引
    has_missing_values,  # 是否存在缺失值的标志
    expected_missing_go_to_left,  # 缺失值是否归于左子节点的预期结果
):
    # Tests various combinations of categorical splits

    n_samples = len(X_binned)  # 样本数量
    n_bins = max(X_binned) + 1  # 离散化后的特征可能的最大值加一，即箱子的数量

    X_binned = np.array(X_binned, dtype=X_BINNED_DTYPE).reshape(-1, 1)  # 将离散化后的特征转换为指定数据类型的NumPy数组
    X_binned = np.asfortranarray(X_binned)  # 将数组转换为Fortran顺序（列优先）的数组

    l2_regularization = 0.0  # L2正则化参数设为0
    min_hessian_to_split = 1e-3  # 分裂节点所需的最小Hessian值
    min_samples_leaf = 1  # 叶子节点的最小样本数
    min_gain_to_split = 0.0  # 分裂节点所需的最小增益

    sample_indices = np.arange(n_samples, dtype=np.uint32)  # 创建包含所有样本索引的NumPy数组
    all_gradients = np.array(all_gradients, dtype=G_H_DTYPE)  # 将所有样本的梯度转换为指定数据类型的NumPy数组
    all_hessians = np.ones(1, dtype=G_H_DTYPE)  # 创建包含全1的Hessian数组
    has_missing_values = np.array([has_missing_values], dtype=np.uint8)  # 将是否存在缺失值的标志转换为NumPy数组
    sum_gradients = all_gradients.sum()  # 计算所有样本的梯度总和
    sum_hessians = n_samples  # Hessian总数设为样本数量
    hessians_are_constant = True  # Hessian是否恒定

    builder = HistogramBuilder(
        X_binned, n_bins, all_gradients, all_hessians, hessians_are_constant, n_threads
    )  # 创建直方图构建器对象

    n_bins_non_missing = np.array([n_bins_non_missing], dtype=np.uint32)  # 将非缺失值箱子数量转换为NumPy数组
    monotonic_cst = np.array(
        [MonotonicConstraint.NO_CST] * X_binned.shape[1], dtype=np.int8
    )  # 创建与特征数目相同的单调性约束数组
    is_categorical = np.ones_like(monotonic_cst, dtype=np.uint8)  # 创建与单调性约束数组相同形状的类别特征标志数组

    splitter = Splitter(
        X_binned,
        n_bins_non_missing,
        missing_values_bin_idx,
        has_missing_values,
        is_categorical,
        monotonic_cst,
        l2_regularization,
        min_hessian_to_split,
        min_samples_leaf,
        min_gain_to_split,
        hessians_are_constant,
    )  # 创建分裂器对象

    histograms = builder.compute_histograms_brute(sample_indices)  # 计算直方图

    value = compute_node_value(
        sum_gradients, sum_hessians, -np.inf, np.inf, l2_regularization
    )  # 计算节点的值
    split_info = splitter.find_node_split(
        n_samples, histograms, sum_gradients, sum_hessians, value
    )  # 查找节点的最佳分裂

    assert split_info.is_categorical  # 断言最佳分裂是分类特征
    assert split_info.gain > 0  # 断言分裂的增益大于0
    _assert_categories_equals_bitset(
        expected_categories_left, split_info.left_cat_bitset
    )  # 检查预期的左子节点类别是否与位集相等
    if has_missing_values:
        assert split_info.missing_go_to_left == expected_missing_go_to_left
    # 如果训练过程中不存在缺失值，则在生长器中稍后设置 missing_go_to_left 标志。

    # 确保样本被正确分裂
    samples_left, samples_right, _ = splitter.split_indices(
        split_info, splitter.partition
    )  # 使用分裂信息分裂样本索引

    left_mask = np.isin(X_binned.ravel(), expected_categories_left)  # 创建左子节点的样本掩码
    assert_array_equal(sample_indices[left_mask], samples_left)  # 断言左子节点的样本索引正确
    assert_array_equal(sample_indices[~left_mask], samples_right)  # 断言右子节点的样本索引正确
    # 设置最小分裂增益为0.0
    min_gain_to_split = 0.0

    # 创建包含所有样本索引的数组
    sample_indices = np.arange(n_samples, dtype=np.uint32)
    # 创建全为1的包含Hessian矩阵的数组，指定dtype为G_H_DTYPE
    all_hessians = np.ones(1, dtype=G_H_DTYPE)
    # 计算所有样本的总Hessian和
    sum_hessians = n_samples
    # 标记Hessian矩阵是否为常数
    hessians_are_constant = True

    # 初始化用于记录分裂特征的列表
    split_features = []

    # 循环确保至少在每个允许的特征（0, 3）上分裂一次
    for i in range(10):
        # 使用种子919 + i创建随机数生成器
        rng = np.random.RandomState(919 + i)
        # 生成n_samples x n_features的二进制化特征矩阵X_binned
        X_binned = np.asfortranarray(
            rng.randint(0, n_bins - 1, size=(n_samples, n_features)),
            dtype=X_BINNED_DTYPE,
        )
        # 将X_binned再次转换为Fortran风格的数组，确保数据类型为X_BINNED_DTYPE
        X_binned = np.asfortranarray(X_binned, dtype=X_BINNED_DTYPE)

        # 创建所有梯度的数组，其中特征1被设为非常重要
        all_gradients = (10 * X_binned[:, 1] + rng.randn(n_samples)).astype(G_H_DTYPE)
        # 计算所有梯度的总和
        sum_gradients = all_gradients.sum()

        # 创建HistogramBuilder对象，用于构建直方图
        builder = HistogramBuilder(
            X_binned,
            n_bins,
            all_gradients,
            all_hessians,
            hessians_are_constant,
            n_threads,
        )

        # 创建包含每个特征的非缺失值箱数的数组
        n_bins_non_missing = np.array([n_bins] * X_binned.shape[1], dtype=np.uint32)
        # 创建指示每个特征是否具有缺失值的数组
        has_missing_values = np.array([False] * X_binned.shape[1], dtype=np.uint8)
        # 创建表示每个特征的单调性约束的数组
        monotonic_cst = np.array(
            [MonotonicConstraint.NO_CST] * X_binned.shape[1], dtype=np.int8
        )
        # 创建表示每个特征是否为分类特征的数组
        is_categorical = np.zeros_like(monotonic_cst, dtype=np.uint8)
        # 设置缺失值的箱索引
        missing_values_bin_idx = n_bins - 1

        # 创建Splitter对象，用于执行分裂
        splitter = Splitter(
            X_binned,
            n_bins_non_missing,
            missing_values_bin_idx,
            has_missing_values,
            is_categorical,
            monotonic_cst,
            l2_regularization,
            min_hessian_to_split,
            min_samples_leaf,
            min_gain_to_split,
            hessians_are_constant,
        )

        # 断言样本索引分区与Splitter对象的分区相匹配
        assert np.all(sample_indices == splitter.partition)

        # 计算直方图
        histograms = builder.compute_histograms_brute(sample_indices)
        # 计算节点值
        value = compute_node_value(
            sum_gradients, sum_hessians, -np.inf, np.inf, l2_regularization
        )

        # 在所有特征允许的情况下，特征1应该被选为分裂特征
        si_root = splitter.find_node_split(
            n_samples,
            histograms,
            sum_gradients,
            sum_hessians,
            value,
            allowed_features=None,
        )
        assert si_root.feature_idx == 1

        # 仅允许特征0和特征3进行分裂
        si_root = splitter.find_node_split(
            n_samples,
            histograms,
            sum_gradients,
            sum_hessians,
            value,
            allowed_features=allowed_features,
        )
        # 记录分裂特征的索引
        split_features.append(si_root.feature_idx)
        # 断言分裂特征在允许的特征集合中
        assert si_root.feature_idx in allowed_features

    # 确保在约束设置中特征0和特征3被分裂
    assert set(allowed_features) == set(split_features)
@pytest.mark.parametrize("forbidden_features", [set(), {1, 3}])
def test_split_feature_fraction_per_split(forbidden_features):
    """Check that feature_fraction_per_split is respected.

    Because we set `n_features = 4` and `feature_fraction_per_split = 0.25`, it means
    that calling `splitter.find_node_split` will be allowed to select a split for a
    single completely random feature at each call. So if we iterate enough, we should
    cover all the allowed features, irrespective of the values of the gradients and
    Hessians of the objective.
    """
    # 定义测试中的特征数量
    n_features = 4
    # 根据禁止的特征集合计算允许的特征集合
    allowed_features = np.array(
        list(set(range(n_features)) - forbidden_features), dtype=np.uint32
    )
    # 设置直方图构建的参数
    n_bins = 5
    n_samples = 40
    l2_regularization = 0.0
    min_hessian_to_split = 1e-3
    min_samples_leaf = 1
    min_gain_to_split = 0.0
    # 随机数生成器
    rng = np.random.default_rng(42)

    # 创建样本索引数组
    sample_indices = np.arange(n_samples, dtype=np.uint32)
    # 随机生成梯度数组，并计算总梯度
    all_gradients = rng.uniform(low=0.5, high=1, size=n_samples).astype(G_H_DTYPE)
    sum_gradients = all_gradients.sum()
    # 创建全为1的Hessian数组，并计算总Hessian
    all_hessians = np.ones(1, dtype=G_H_DTYPE)
    sum_hessians = n_samples
    hessians_are_constant = True

    # 创建二维数组X_binned，表示特征的分箱情况
    X_binned = np.asfortranarray(
        rng.integers(low=0, high=n_bins - 1, size=(n_samples, n_features)),
        dtype=X_BINNED_DTYPE,
    )
    X_binned = np.asfortranarray(X_binned, dtype=X_BINNED_DTYPE)
    # 创建直方图构建器对象
    builder = HistogramBuilder(
        X_binned,
        n_bins,
        all_gradients,
        all_hessians,
        hessians_are_constant,
        n_threads,
    )
    # 计算直方图
    histograms = builder.compute_histograms_brute(sample_indices)
    # 计算节点值
    value = compute_node_value(
        sum_gradients, sum_hessians, -np.inf, np.inf, l2_regularization
    )
    # 创建非缺失值分箱数数组
    n_bins_non_missing = np.array([n_bins] * X_binned.shape[1], dtype=np.uint32)
    # 创建是否存在缺失值数组
    has_missing_values = np.array([False] * X_binned.shape[1], dtype=np.uint8)
    # 创建单调约束数组
    monotonic_cst = np.array(
        [MonotonicConstraint.NO_CST] * X_binned.shape[1], dtype=np.int8
    )
    # 创建是否为分类特征数组
    is_categorical = np.zeros_like(monotonic_cst, dtype=np.uint8)
    # 设置缺失值对应的分箱索引
    missing_values_bin_idx = n_bins - 1

    # 构建参数字典
    params = dict(
        X_binned=X_binned,
        n_bins_non_missing=n_bins_non_missing,
        missing_values_bin_idx=missing_values_bin_idx,
        has_missing_values=has_missing_values,
        is_categorical=is_categorical,
        monotonic_cst=monotonic_cst,
        l2_regularization=l2_regularization,
        min_hessian_to_split=min_hessian_to_split,
        min_samples_leaf=min_samples_leaf,
        min_gain_to_split=min_gain_to_split,
        hessians_are_constant=hessians_are_constant,
        rng=rng,
    )
    # 使用部分特征比例创建分裂器对象
    splitter_subsample = Splitter(
        feature_fraction_per_split=0.25,  # 这里是关键设置
        **params,
    )
    # 使用所有特征创建分裂器对象
    splitter_all_features = Splitter(feature_fraction_per_split=1.0, **params)

    # 断言分裂器的分区结果与样本索引相同
    assert np.all(sample_indices == splitter_subsample.partition)

    # 初始化分裂后的特征列表
    split_features_subsample = []
    split_features_all = []
    # 确保每个特征至少被划分一次的循环。
    # 这通过 split_features 被追踪并在最后进行检查。
    for i in range(20):
        # 在子抽样数据集上查找节点分割器，si_root 是找到的根节点分割器
        si_root = splitter_subsample.find_node_split(
            n_samples,
            histograms,
            sum_gradients,
            sum_hessians,
            value,
            allowed_features=allowed_features,
        )
        # 将找到的特征索引添加到子抽样数据集的分割特征列表中
        split_features_subsample.append(si_root.feature_idx)

        # 这第二个分割器是我们的“对照”。
        # 在全部特征数据集上查找节点分割器，si_root 是找到的根节点分割器
        si_root = splitter_all_features.find_node_split(
            n_samples,
            histograms,
            sum_gradients,
            sum_hessians,
            value,
            allowed_features=allowed_features,
        )
        # 将找到的特征索引添加到全部特征数据集的分割特征列表中
        split_features_all.append(si_root.feature_idx)

    # 确保所有特征都被划分。
    assert set(split_features_subsample) == set(allowed_features)

    # 确保我们的“对照”始终在相同的特征上进行划分。
    assert len(set(split_features_all)) == 1
```