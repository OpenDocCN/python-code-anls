# `D:\src\scipysrc\scikit-learn\sklearn\decomposition\tests\test_sparse_pca.py`

```
# 导入系统模块
import sys

# 导入需要使用的第三方库和模块
import numpy as np
import pytest
from numpy.testing import assert_array_equal

# 导入需要使用的scikit-learn相关模块
from sklearn.decomposition import PCA, MiniBatchSparsePCA, SparsePCA
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
    assert_allclose,
    assert_array_almost_equal,
    if_safe_multiprocessing_with_blas,
)
from sklearn.utils.extmath import svd_flip


# 生成模拟数据函数，用于生成符合指定要求的数据集
def generate_toy_data(n_components, n_samples, image_size, random_state=None):
    # 计算特征数量
    n_features = image_size[0] * image_size[1]

    # 初始化随机数生成器
    rng = check_random_state(random_state)
    
    # 生成随机的U和V矩阵
    U = rng.randn(n_samples, n_components)
    V = rng.randn(n_components, n_features)

    # 定义图像中心和尺寸
    centers = [(3, 3), (6, 7), (8, 1)]
    sz = [1, 2, 1]
    
    # 根据中心和尺寸生成图像数据
    for k in range(n_components):
        img = np.zeros(image_size)
        xmin, xmax = centers[k][0] - sz[k], centers[k][0] + sz[k]
        ymin, ymax = centers[k][1] - sz[k], centers[k][1] + sz[k]
        img[xmin:xmax][:, ymin:ymax] = 1.0
        V[k, :] = img.ravel()

    # 生成最终的Y矩阵，加入随机噪声
    Y = np.dot(U, V)
    Y += 0.1 * rng.randn(Y.shape[0], Y.shape[1])  # 添加噪声
    return Y, U, V


# 测试函数：验证SparsePCA的输出形状是否正确
def test_correct_shapes():
    rng = np.random.RandomState(0)
    X = rng.randn(12, 10)
    
    # 使用SparsePCA拟合数据
    spca = SparsePCA(n_components=8, random_state=rng)
    U = spca.fit_transform(X)
    
    # 验证成分矩阵和转换后的U的形状是否符合预期
    assert spca.components_.shape == (8, 10)
    assert U.shape == (12, 8)
    
    # 测试过度完备分解
    spca = SparsePCA(n_components=13, random_state=rng)
    U = spca.fit_transform(X)
    assert spca.components_.shape == (13, 10)
    assert U.shape == (12, 13)


# 测试函数：验证SparsePCA的拟合和转换过程
def test_fit_transform():
    alpha = 1
    rng = np.random.RandomState(0)
    
    # 生成模拟数据Y
    Y, _, _ = generate_toy_data(3, 10, (8, 8), random_state=rng)  # wide array
    
    # 使用不同的SparsePCA方法拟合数据
    spca_lars = SparsePCA(n_components=3, method="lars", alpha=alpha, random_state=0)
    spca_lars.fit(Y)

    # 使用CD方法验证结果是否相似
    spca_lasso = SparsePCA(n_components=3, method="cd", random_state=0, alpha=alpha)
    spca_lasso.fit(Y)
    
    # 验证两种方法得到的成分矩阵是否几乎相等
    assert_array_almost_equal(spca_lasso.components_, spca_lars.components_)


# 测试函数：验证SparsePCA在多核处理下的拟合和转换过程
@if_safe_multiprocessing_with_blas
def test_fit_transform_parallel():
    alpha = 1
    rng = np.random.RandomState(0)
    
    # 生成模拟数据Y
    Y, _, _ = generate_toy_data(3, 10, (8, 8), random_state=rng)  # wide array
    
    # 使用不同的SparsePCA方法拟合数据
    spca_lars = SparsePCA(n_components=3, method="lars", alpha=alpha, random_state=0)
    spca_lars.fit(Y)
    U1 = spca_lars.transform(Y)
    
    # 测试多CPU情况
    spca = SparsePCA(
        n_components=3, n_jobs=2, method="lars", alpha=alpha, random_state=0
    ).fit(Y)
    U2 = spca.transform(Y)
    
    # 验证结果是否完全非零，并且U1和U2是否几乎相等
    assert not np.all(spca_lars.components_ == 0)
    assert_array_almost_equal(U1, U2)


# 测试函数：验证SparsePCA处理全零特征时不返回NaN
def test_transform_nan():
    # 测试当样本中的所有特征都为零时，SparsePCA不会返回NaN
    rng = np.random.RandomState(0)
    # 使用 generate_toy_data 函数生成一个形状为 (3, 10) 的数据 Y，其中每个值在 (8, 8) 范围内，使用指定的随机种子 rng
    Y, _, _ = generate_toy_data(3, 10, (8, 8), random_state=rng)  # wide array
    
    # 将 Y 的第一列所有元素设置为 0
    Y[:, 0] = 0
    
    # 创建 SparsePCA 的估计器对象，设置要提取的主成分数量为 8
    estimator = SparsePCA(n_components=8)
    
    # 断言：确保 estimator.fit_transform(Y) 不包含任何 NaN 值
    assert not np.any(np.isnan(estimator.fit_transform(Y)))
# 定义一个测试函数，用于验证 SparsePCA 模型在高维数据上的拟合与变换
def test_fit_transform_tall():
    # 创建一个随机数生成器对象 rng
    rng = np.random.RandomState(0)
    # 生成一个高维的玩具数据集 Y，并且获取返回的第一个元素 Y（这里生成的数据集是高且窄的数组）
    Y, _, _ = generate_toy_data(3, 65, (8, 8), random_state=rng)  # tall array
    # 使用 SparsePCA 模型进行稀疏主成分分析，方法为 LARS
    spca_lars = SparsePCA(n_components=3, method="lars", random_state=rng)
    # 对 Y 应用拟合与变换操作，得到变换后的结果 U1
    U1 = spca_lars.fit_transform(Y)
    # 使用 SparsePCA 模型进行稀疏主成分分析，方法为 CD
    spca_lasso = SparsePCA(n_components=3, method="cd", random_state=rng)
    # 先对 Y 应用拟合操作，然后再进行变换操作，得到变换后的结果 U2
    U2 = spca_lasso.fit(Y).transform(Y)
    # 断言 U1 和 U2 的值几乎相等
    assert_array_almost_equal(U1, U2)


# 定义一个测试函数，用于验证 SparsePCA 模型的初始化
def test_initialization():
    # 创建一个随机数生成器对象 rng
    rng = np.random.RandomState(0)
    # 使用随机数生成器创建一个 5x3 的随机数矩阵 U_init
    U_init = rng.randn(5, 3)
    # 使用随机数生成器创建一个 3x4 的随机数矩阵 V_init
    V_init = rng.randn(3, 4)
    # 创建 SparsePCA 模型对象 model，设置参数：主成分数为 3，初始化矩阵分别为 U_init 和 V_init，最大迭代次数为 0
    model = SparsePCA(
        n_components=3, U_init=U_init, V_init=V_init, max_iter=0, random_state=rng
    )
    # 对随机生成的 5x4 的数据集应用拟合操作
    model.fit(rng.randn(5, 4))

    # 计算期望的主成分，对 V_init 进行标准化后赋给 expected_components
    expected_components = V_init / np.linalg.norm(V_init, axis=1, keepdims=True)
    expected_components = svd_flip(u=expected_components.T, v=None)[0].T
    # 断言 model 的主成分 components_ 与 expected_components 几乎相等
    assert_allclose(model.components_, expected_components)


# 定义一个测试函数，用于验证 MiniBatchSparsePCA 模型在数据维度上的正确性
def test_mini_batch_correct_shapes():
    # 创建一个随机数生成器对象 rng
    rng = np.random.RandomState(0)
    # 创建一个 12x10 的随机数矩阵 X
    X = rng.randn(12, 10)
    # 创建 MiniBatchSparsePCA 模型对象 pca，设置参数：主成分数为 8，最大迭代次数为 1
    pca = MiniBatchSparsePCA(n_components=8, max_iter=1, random_state=rng)
    # 对 X 应用拟合与变换操作，得到变换后的结果 U
    U = pca.fit_transform(X)
    # 断言 pca 的主成分 components_ 的形状为 (8, 10)
    assert pca.components_.shape == (8, 10)
    # 断言 U 的形状为 (12, 8)
    assert U.shape == (12, 8)
    # 测试超完备分解情况
    # 创建 MiniBatchSparsePCA 模型对象 pca，设置参数：主成分数为 13，最大迭代次数为 1
    pca = MiniBatchSparsePCA(n_components=13, max_iter=1, random_state=rng)
    # 对 X 应用拟合与变换操作，得到变换后的结果 U
    U = pca.fit_transform(X)
    # 断言 pca 的主成分 components_ 的形状为 (13, 10)
    assert pca.components_.shape == (13, 10)
    # 断言 U 的形状为 (12, 13)


# 定义一个测试函数，用于验证 MiniBatchSparsePCA 模型在宽数组上的拟合与变换
def test_mini_batch_fit_transform():
    # 设置 alpha 值为 1
    alpha = 1
    # 创建一个随机数生成器对象 rng
    rng = np.random.RandomState(0)
    # 生成一个宽且低维的玩具数据集 Y，并且获取返回的第一个元素 Y（这里生成的数据集是宽且矮的数组）
    Y, _, _ = generate_toy_data(3, 10, (8, 8), random_state=rng)  # wide array
    # 使用 MiniBatchSparsePCA 模型进行稀疏主成分分析，设置参数：主成分数为 3，alpha 值为 alpha
    spca_lars = MiniBatchSparsePCA(n_components=3, random_state=0, alpha=alpha).fit(Y)
    # 对 Y 应用变换操作，得到变换后的结果 U1
    U1 = spca_lars.transform(Y)
    
    # 测试多CPU情况
    if sys.platform == "win32":  # 如果是 Windows 系统（假设为单CPU处理）
        import joblib
        # 保存并修改 joblib 的并行模块设置
        _mp = joblib.parallel.multiprocessing
        joblib.parallel.multiprocessing = None
        try:
            # 创建 MiniBatchSparsePCA 模型对象 spca，设置参数：主成分数为 3，使用 2 个进程进行并行处理
            spca = MiniBatchSparsePCA(
                n_components=3, n_jobs=2, alpha=alpha, random_state=0
            )
            # 对 Y 应用拟合与变换操作，得到变换后的结果 U2
            U2 = spca.fit(Y).transform(Y)
        finally:
            # 恢复 joblib 的并行模块设置
            joblib.parallel.multiprocessing = _mp
    else:  # 如果可以有效地使用并行处理
        # 创建 MiniBatchSparsePCA 模型对象 spca，设置参数：主成分数为 3，使用 2 个进程进行并行处理
        spca = MiniBatchSparsePCA(n_components=3, n_jobs=2, alpha=alpha, random_state=0)
        # 对 Y 应用拟合与变换操作，得到变换后的结果 U2
        U2 = spca.fit(Y).transform(Y)
    
    # 断言 spca_lars 的主成分 components_ 不全为 0
    assert not np.all(spca_lars.components_ == 0)
    # 断言 U1 和 U2 的值几乎相等
    assert_array_almost_equal(U1, U2)
    # 测试 CD 方法是否给出类似的结果
    # 使用 MiniBatchSparsePCA 模型进行稀疏主成分分析，设置参数：主成分数为 3，方法为 CD，alpha 值为 alpha
    spca_lasso = MiniBatchSparsePCA(
        n_components=3, method="cd", alpha=alpha, random_state=0
    ).fit(Y)
    # 断言 spca_lasso 的主成分 components_ 与 spca_lars 的主成分 components_ 几乎相等
    assert_array_almost_equal(spca_lasso.components_, spca_lars.components_)


# 定义一个测试函数，用于验证 SparsePCA 模型在数据尺度上的拟合与变换
def test_scaling_fit_transform():
    # 设置 alpha 值为 1
    alpha = 1
    # 创建一个随机数生成器对象 rng
    rng = np.random.RandomState(0)
    # 生成一个高维的玩具数据集 Y，并且获取返回的第一个元素 Y
    Y, _, _ = generate_toy_data(3, 1000, (8, 8), random_state=rng)
    # 使用 SparsePCA 模型进行稀疏主成分分析
    # 使用 SparsePCA-LARS 模型对训练数据 Y 进行拟合并进行转换，返回转换后的结果
    results_train = spca_lars.fit_transform(Y)
    # 使用之前拟合好的 SparsePCA-LARS 模型对测试数据 Y 的前10个样本进行转换，返回转换后的结果
    results_test = spca_lars.transform(Y[:10])
    # 断言检查转换后的训练结果的第一个元素与测试结果的第一个元素是否在数值上非常接近
    assert_allclose(results_train[0], results_test[0])
# 定义测试函数，用于比较 PCA 和 SparsePCA 的结果
def test_pca_vs_spca():
    # 设置随机数生成器的种子
    rng = np.random.RandomState(0)
    # 生成一个大小为 (3, 1000) 的玩具数据集 Y，同时获取其均值和标准差
    Y, _, _ = generate_toy_data(3, 1000, (8, 8), random_state=rng)
    # 生成一个大小为 (3, 10) 的玩具数据集 Z，同时获取其均值和标准差
    Z, _, _ = generate_toy_data(3, 10, (8, 8), random_state=rng)
    # 创建 SparsePCA 对象，设定 alpha 和 ridge_alpha 为 0，要求 2 个主成分
    spca = SparsePCA(alpha=0, ridge_alpha=0, n_components=2)
    # 创建 PCA 对象，要求 2 个主成分
    pca = PCA(n_components=2)
    # 在数据集 Y 上拟合 PCA 模型
    pca.fit(Y)
    # 在数据集 Y 上拟合 SparsePCA 模型
    spca.fit(Y)
    # 对数据集 Z 进行 PCA 变换，并保存结果
    results_test_pca = pca.transform(Z)
    # 对数据集 Z 进行 SparsePCA 变换，并保存结果
    results_test_spca = spca.transform(Z)
    # 断言 SparsePCA 的成分与 PCA 的转置成分的绝对值近似为单位矩阵，允许误差为 1e-5
    assert_allclose(
        np.abs(spca.components_.dot(pca.components_.T)), np.eye(2), atol=1e-5
    )
    # 对 PCA 变换后的结果进行符号处理
    results_test_pca *= np.sign(results_test_pca[0, :])
    # 对 SparsePCA 变换后的结果进行符号处理
    results_test_spca *= np.sign(results_test_spca[0, :])
    # 断言经过符号处理后的 PCA 和 SparsePCA 变换结果近似相等
    assert_allclose(results_test_pca, results_test_spca)


# 使用参数化测试框架进行测试 SparsePCA 的主成分数量设定
@pytest.mark.parametrize("SPCA", [SparsePCA, MiniBatchSparsePCA])
@pytest.mark.parametrize("n_components", [None, 3])
def test_spca_n_components_(SPCA, n_components):
    # 设置随机数生成器的种子
    rng = np.random.RandomState(0)
    # 定义数据集的样本数量和特征数量
    n_samples, n_features = 12, 10
    # 生成服从标准正态分布的随机数据集 X
    X = rng.randn(n_samples, n_features)

    # 创建 SparsePCA 模型，根据参数 n_components 进行拟合
    model = SPCA(n_components=n_components).fit(X)

    # 如果 n_components 不为 None，则断言模型的主成分数与 n_components 相等
    if n_components is not None:
        assert model.n_components_ == n_components
    else:
        # 如果 n_components 为 None，则断言模型的主成分数等于特征数 n_features
        assert model.n_components_ == n_features


# 使用参数化测试框架测试 SparsePCA 的数据类型匹配
@pytest.mark.parametrize("SPCA", (SparsePCA, MiniBatchSparsePCA))
@pytest.mark.parametrize("method", ("lars", "cd"))
@pytest.mark.parametrize(
    "data_type, expected_type",
    (
        (np.float32, np.float32),
        (np.float64, np.float64),
        (np.int32, np.float64),
        (np.int64, np.float64),
    ),
)
def test_sparse_pca_dtype_match(SPCA, method, data_type, expected_type):
    # 验证 SparsePCA 模型输出矩阵的数据类型匹配
    n_samples, n_features, n_components = 12, 10, 3
    # 设置随机数生成器的种子
    rng = np.random.RandomState(0)
    # 生成随机数组成的输入矩阵，并将其类型转换为指定的 data_type
    input_array = rng.randn(n_samples, n_features).astype(data_type)
    # 创建 SparsePCA 模型，设定主成分数量和求解方法
    model = SPCA(n_components=n_components, method=method)
    # 在输入矩阵上进行拟合和变换
    transformed = model.fit_transform(input_array)

    # 断言变换后的结果的数据类型与预期类型 expected_type 相等
    assert transformed.dtype == expected_type
    # 断言模型的主成分矩阵的数据类型与预期类型 expected_type 相等
    assert model.components_.dtype == expected_type


# 使用参数化测试框架测试 SparsePCA 的数值一致性
@pytest.mark.parametrize("SPCA", (SparsePCA, MiniBatchSparsePCA))
@pytest.mark.parametrize("method", ("lars", "cd"))
def test_sparse_pca_numerical_consistency(SPCA, method):
    # 验证在 np.float32 和 np.float64 之间的数值一致性
    rtol = 1e-3
    alpha = 2
    n_samples, n_features, n_components = 12, 10, 3
    # 设置随机数生成器的种子
    rng = np.random.RandomState(0)
    # 生成随机数组成的输入矩阵
    input_array = rng.randn(n_samples, n_features)

    # 创建两个不同数据类型的 SparsePCA 模型
    model_32 = SPCA(
        n_components=n_components, alpha=alpha, method=method, random_state=0
    )
    transformed_32 = model_32.fit_transform(input_array.astype(np.float32))

    model_64 = SPCA(
        n_components=n_components, alpha=alpha, method=method, random_state=0
    )
    transformed_64 = model_64.fit_transform(input_array.astype(np.float64))

    # 断言两种数据类型下变换后的结果近似相等
    assert_allclose(transformed_64, transformed_32, rtol=rtol)
    # 断言两种数据类型下模型的主成分矩阵近似相等
    assert_allclose(model_64.components_, model_32.components_, rtol=rtol)


# 使用参数化测试框架测试 SparsePCA 的特征名称输出
@pytest.mark.parametrize("SPCA", [SparsePCA, MiniBatchSparsePCA])
def test_spca_feature_names_out(SPCA):
    # 此测试函数尚未实现具体的测试内容，留待后续补充
    # 为 *SparsePCA* 检查特征名称。
    # 使用种子值为 0 的随机状态创建 NumPy 的随机数生成器对象 rng
    rng = np.random.RandomState(0)
    # 设定样本数为 12，特征数为 10
    n_samples, n_features = 12, 10
    # 生成一个形状为 (12, 10) 的随机数矩阵 X
    X = rng.randn(n_samples, n_features)
    
    # 使用 SPCA 模型拟合数据 X，设置主成分数为 4
    model = SPCA(n_components=4).fit(X)
    # 获取拟合后模型的输出特征名称
    names = model.get_feature_names_out()
    
    # 获取 SPCA 类的名称并转换为小写
    estimator_name = SPCA.__name__.lower()
    # 断言生成的特征名称列表与预期的列表相等
    assert_array_equal([f"{estimator_name}{i}" for i in range(4)], names)
# TODO(1.6): remove in 1.6
# 为了版本1.6中的移除，这个函数需要被删除
def test_spca_max_iter_None_deprecation():
    """Check that we raise a warning for the deprecation of `max_iter=None`."""
    # 创建一个随机数生成器对象rng
    rng = np.random.RandomState(0)
    # 设定样本数和特征数
    n_samples, n_features = 12, 10
    # 生成服从标准正态分布的数据矩阵X
    X = rng.randn(n_samples, n_features)

    # 设定警告消息
    warn_msg = "`max_iter=None` is deprecated in version 1.4 and will be removed"
    # 使用pytest的warns函数检查是否触发FutureWarning，并匹配warn_msg
    with pytest.warns(FutureWarning, match=warn_msg):
        # 使用MiniBatchSparsePCA(max_iter=None)拟合数据X
        MiniBatchSparsePCA(max_iter=None).fit(X)


def test_spca_early_stopping(global_random_seed):
    """Check that `tol` and `max_no_improvement` act as early stopping."""
    # 创建一个随机数生成器对象rng
    rng = np.random.RandomState(global_random_seed)
    # 设定样本数和特征数
    n_samples, n_features = 50, 10
    # 生成服从标准正态分布的数据矩阵X
    X = rng.randn(n_samples, n_features)

    # 使用不同的tolerance值来强制一个模型早停
    # 创建使用较大tolerance的模型，用于测试早停
    model_early_stopped = MiniBatchSparsePCA(
        max_iter=100, tol=0.5, random_state=global_random_seed
    ).fit(X)
    # 创建使用较小tolerance的模型，用于对比不早停
    model_not_early_stopped = MiniBatchSparsePCA(
        max_iter=100, tol=1e-3, random_state=global_random_seed
    ).fit(X)
    # 断言早停模型的迭代次数小于未早停模型的迭代次数
    assert model_early_stopped.n_iter_ < model_not_early_stopped.n_iter_

    # 强制最大不改善次数为一个较大值，以检查其是否帮助早停
    model_early_stopped = MiniBatchSparsePCA(
        max_iter=100, tol=1e-6, max_no_improvement=2, random_state=global_random_seed
    ).fit(X)
    model_not_early_stopped = MiniBatchSparsePCA(
        max_iter=100, tol=1e-6, max_no_improvement=100, random_state=global_random_seed
    ).fit(X)
    # 断言早停模型的迭代次数小于未早停模型的迭代次数
    assert model_early_stopped.n_iter_ < model_not_early_stopped.n_iter_


def test_equivalence_components_pca_spca(global_random_seed):
    """Check the equivalence of the components found by PCA and SparsePCA.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/23932
    """
    # 创建一个随机数生成器对象rng
    rng = np.random.RandomState(global_random_seed)
    # 生成服从标准正态分布的数据矩阵X
    X = rng.randn(50, 4)

    # 设定主成分的数量
    n_components = 2
    # 使用PCA找到主成分
    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=0,
    ).fit(X)
    # 使用SparsePCA找到主成分
    spca = SparsePCA(
        n_components=n_components,
        method="lars",
        ridge_alpha=0,
        alpha=0,
        random_state=0,
    ).fit(X)

    # 断言两者找到的主成分数组非常接近
    assert_allclose(pca.components_, spca.components_)


def test_sparse_pca_inverse_transform():
    """Check that `inverse_transform` in `SparsePCA` and `PCA` are similar."""
    # 创建一个随机数生成器对象rng
    rng = np.random.RandomState(0)
    # 设定样本数和特征数
    n_samples, n_features = 10, 5
    # 生成服从标准正态分布的数据矩阵X
    X = rng.randn(n_samples, n_features)

    # 设定主成分的数量
    n_components = 2
    # 创建SparsePCA对象
    spca = SparsePCA(
        n_components=n_components, alpha=1e-12, ridge_alpha=1e-12, random_state=0
    )
    # 创建PCA对象
    pca = PCA(n_components=n_components, random_state=0)
    # 对数据X进行SparsePCA转换
    X_trans_spca = spca.fit_transform(X)
    # 对数据X进行PCA转换
    X_trans_pca = pca.fit_transform(X)
    # 断言SparsePCA和PCA的逆变换结果非常接近
    assert_allclose(
        spca.inverse_transform(X_trans_spca), pca.inverse_transform(X_trans_pca)
    )


@pytest.mark.parametrize("SPCA", [SparsePCA, MiniBatchSparsePCA])
def test_transform_inverse_transform_round_trip(SPCA):
    """Check the `transform` and `inverse_transform` round trip with no loss of
    information.
    """
    # 设置一个随机数生成器，并指定种子以保证结果的可复现性
    rng = np.random.RandomState(0)
    # 定义样本数和特征数
    n_samples, n_features = 10, 5
    # 生成一个随机的样本矩阵 X，服从标准正态分布
    X = rng.randn(n_samples, n_features)
    
    # 将主成分数量设置为特征数，创建一个 Sparse PCA 对象
    n_components = n_features
    spca = SPCA(
        n_components=n_components, alpha=1e-12, ridge_alpha=1e-12, random_state=0
    )
    # 对样本矩阵 X 进行稀疏主成分分析，并返回转换后的结果
    X_trans_spca = spca.fit_transform(X)
    # 使用逆变换恢复原始数据，并通过断言检查是否回到了原始数据 X
    assert_allclose(spca.inverse_transform(X_trans_spca), X)
```