# `D:\src\scipysrc\scikit-learn\sklearn\neighbors\tests\test_kde.py`

```
# 导入joblib库，用于模型持久化和加载
import joblib
# 导入NumPy库，用于科学计算操作
import numpy as np
# 导入pytest库，用于编写和运行测试
import pytest

# 从sklearn.datasets模块中导入make_blobs函数，用于生成聚类数据
from sklearn.datasets import make_blobs
# 导入sklearn中的异常类NotFittedError，用于指示模型未拟合的异常
from sklearn.exceptions import NotFittedError
# 从sklearn.model_selection模块中导入GridSearchCV类，用于网格搜索交叉验证
from sklearn.model_selection import GridSearchCV
# 从sklearn.neighbors模块中导入KDTree、KernelDensity和NearestNeighbors类
from sklearn.neighbors import KDTree, KernelDensity, NearestNeighbors
# 从sklearn.neighbors._ball_tree模块中导入kernel_norm函数
from sklearn.neighbors._ball_tree import kernel_norm
# 从sklearn.pipeline模块中导入make_pipeline函数，用于创建管道
from sklearn.pipeline import make_pipeline
# 从sklearn.preprocessing模块中导入StandardScaler类，用于数据标准化
from sklearn.preprocessing import StandardScaler
# 从sklearn.utils._testing模块中导入assert_allclose函数，用于断言所有元素近似相等

# 计算慢速核密度估计的函数定义
def compute_kernel_slow(Y, X, kernel, h):
    # 如果h是"scott"，使用Scott's规则计算带宽h
    if h == "scott":
        h = X.shape[0] ** (-1 / (X.shape[1] + 4))
    # 如果h是"silverman"，使用Silverman's规则计算带宽h
    elif h == "silverman":
        h = (X.shape[0] * (X.shape[1] + 2) / 4) ** (-1 / (X.shape[1] + 4))

    # 计算Y与X之间的距离矩阵d
    d = np.sqrt(((Y[:, None, :] - X) ** 2).sum(-1))
    # 计算核函数的归一化系数norm
    norm = kernel_norm(h, X.shape[1], kernel) / X.shape[0]

    # 根据指定的核函数类型计算核密度估计值
    if kernel == "gaussian":
        return norm * np.exp(-0.5 * (d * d) / (h * h)).sum(-1)
    elif kernel == "tophat":
        return norm * (d < h).sum(-1)
    elif kernel == "epanechnikov":
        return norm * ((1.0 - (d * d) / (h * h)) * (d < h)).sum(-1)
    elif kernel == "exponential":
        return norm * (np.exp(-d / h)).sum(-1)
    elif kernel == "linear":
        return norm * ((1 - d / h) * (d < h)).sum(-1)
    elif kernel == "cosine":
        return norm * (np.cos(0.5 * np.pi * d / h) * (d < h)).sum(-1)
    else:
        raise ValueError("kernel not recognized")

# 检查核密度估计结果的函数定义
def check_results(kernel, bandwidth, atol, rtol, X, Y, dens_true):
    # 创建KernelDensity对象kde，使用指定的核函数和带宽参数
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth, atol=atol, rtol=rtol)
    # 拟合模型并计算在样本Y上的分数
    log_dens = kde.fit(X).score_samples(Y)
    # 断言计算的核密度估计值与真实值dens_true近似相等
    assert_allclose(np.exp(log_dens), dens_true, atol=atol, rtol=max(1e-7, rtol))
    # 断言计算的样本概率密度与真实值dens_true的乘积近似相等
    assert_allclose(
        np.exp(kde.score(Y)), np.prod(dens_true), atol=atol, rtol=max(1e-7, rtol)
    )

# 使用参数化测试框架pytest.mark.parametrize定义的测试用例
@pytest.mark.parametrize(
    "kernel", ["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"]
)
@pytest.mark.parametrize("bandwidth", [0.01, 0.1, 1, "scott", "silverman"])
# 核密度估计函数的测试函数定义
def test_kernel_density(kernel, bandwidth):
    # 定义生成数据的样本数和特征数
    n_samples, n_features = (100, 3)

    # 初始化随机数生成器rng
    rng = np.random.RandomState(0)
    # 生成符合标准正态分布的样本数据X和Y
    X = rng.randn(n_samples, n_features)
    Y = rng.randn(n_samples, n_features)

    # 计算真实的核密度估计值dens_true
    dens_true = compute_kernel_slow(Y, X, kernel, bandwidth)

    # 针对不同的相对和绝对容差进行断言测试
    for rtol in [0, 1e-5]:
        for atol in [1e-6, 1e-2]:
            for breadth_first in (True, False):
                # 调用check_results函数检查核密度估计结果
                check_results(kernel, bandwidth, atol, rtol, X, Y, dens_true)

# 核密度估计的采样测试函数定义
def test_kernel_density_sampling(n_samples=100, n_features=3):
    # 初始化随机数生成器rng
    rng = np.random.RandomState(0)
    # 生成符合标准正态分布的样本数据X
    X = rng.randn(n_samples, n_features)

    # 指定带宽参数为0.2
    bandwidth = 0.2

# 代码块结束
    # 对于每个指定的核函数（高斯核或者矩形核），进行以下操作：
    for kernel in ["gaussian", "tophat"]:
        # 使用指定的带宽和核函数构建 KernelDensity 对象，并在数据集 X 上进行拟合
        kde = KernelDensity(bandwidth=bandwidth, kernel=kernel).fit(X)
        # 从拟合的概率密度函数中抽取 100 个样本
        samp = kde.sample(100)
        # 断言样本的形状与原始数据集 X 的形状相同
        assert X.shape == samp.shape
    
        # 检查抽取的样本是否在正确的范围内
        nbrs = NearestNeighbors(n_neighbors=1).fit(X)
        dist, ind = nbrs.kneighbors(X, return_distance=True)
    
        if kernel == "tophat":
            # 断言距离小于带宽，确保矩形核样本在指定带宽内
            assert np.all(dist < bandwidth)
        elif kernel == "gaussian":
            # 断言距离小于 5 倍带宽的标准差，确保高斯核样本在安全范围内
            # 对于 100 个样本来说，5 倍带宽的标准差是一个很安全的边界，但可能有极小的失败几率
            assert np.all(dist < 5 * bandwidth)
    
    # 检查不支持的核函数
    for kernel in ["epanechnikov", "exponential", "linear", "cosine"]:
        # 使用指定的带宽和核函数构建 KernelDensity 对象，并在数据集 X 上进行拟合
        kde = KernelDensity(bandwidth=bandwidth, kernel=kernel).fit(X)
        # 使用 pytest 来断言抽样该核函数时会引发 NotImplementedError 异常
        with pytest.raises(NotImplementedError):
            kde.sample(100)
    
    # 非回归测试：曾经返回一个标量的情况
    # 重新定义数据集 X，确保它是一个 4x1 的随机正态分布数据
    X = rng.randn(4, 1)
    # 使用高斯核函数构建 KernelDensity 对象，并在数据集 X 上进行拟合
    kde = KernelDensity(kernel="gaussian").fit(X)
    # 断言从拟合的概率密度函数中抽取一个样本的形状为 (1, 1)
    assert kde.sample().shape == (1, 1)
# 使用 pytest 的装饰器标记，参数化测试，测试不同的算法选项
@pytest.mark.parametrize("algorithm", ["auto", "ball_tree", "kd_tree"])
# 使用 pytest 的装饰器标记，参数化测试，测试不同的距离度量选项
@pytest.mark.parametrize(
    "metric", ["euclidean", "minkowski", "manhattan", "chebyshev", "haversine"]
)
def test_kde_algorithm_metric_choice(algorithm, metric):
    # 烟雾测试，用于不同的距离度量和算法选择
    rng = np.random.RandomState(0)
    # 生成随机数据，形状为 (10, 2)，用于 Haversine 距离
    X = rng.randn(10, 2)  # 2 features required for haversine dist.
    Y = rng.randn(10, 2)

    # 创建 KernelDensity 对象，根据给定的算法和距离度量
    kde = KernelDensity(algorithm=algorithm, metric=metric)

    # 如果算法选择了 "kd_tree" 并且距离度量不在 KDTree.valid_metrics 中
    if algorithm == "kd_tree" and metric not in KDTree.valid_metrics:
        # 预期引发 ValueError 异常，异常消息包含 "invalid metric"
        with pytest.raises(ValueError, match="invalid metric"):
            kde.fit(X)
    else:
        # 否则，正常拟合数据 X
        kde.fit(X)
        # 计算样本 Y 的密度评分
        y_dens = kde.score_samples(Y)
        # 断言密度评分的形状与样本 Y 的形状的第一维度相同
        assert y_dens.shape == Y.shape[:1]


def test_kde_score(n_samples=100, n_features=3):
    pass
    # FIXME
    # 以下两行代码存在问题，需要修复
    # rng = np.random.RandomState(0)
    # X = rng.random_sample((n_samples, n_features))
    # Y = rng.random_sample((n_samples, n_features))


def test_kde_sample_weights_error():
    # 创建 KernelDensity 对象
    kde = KernelDensity()
    # 使用随机数据拟合，预期引发 ValueError 异常
    with pytest.raises(ValueError):
        kde.fit(np.random.random((200, 10)), sample_weight=np.random.random((200, 10)))
    # 使用负权重的随机数据拟合，预期引发 ValueError 异常
    with pytest.raises(ValueError):
        kde.fit(np.random.random((200, 10)), sample_weight=-np.random.random(200))


def test_kde_pipeline_gridsearch():
    # 测试 KernelDensity 在流水线和网格搜索中的兼容性
    X, _ = make_blobs(cluster_std=0.1, random_state=1, centers=[[0, 1], [1, 0], [0, 0]])
    # 创建流水线，包括数据标准化和 KernelDensity 对象
    pipe1 = make_pipeline(
        StandardScaler(with_mean=False, with_std=False),
        KernelDensity(kernel="gaussian"),
    )
    # 设置网格搜索的参数字典，包括不同的 bandwidth 参数
    params = dict(kerneldensity__bandwidth=[0.001, 0.01, 0.1, 1, 10])
    # 创建 GridSearchCV 对象，使用流水线和参数字典进行网格搜索
    search = GridSearchCV(pipe1, param_grid=params)
    # 对数据 X 进行网格搜索拟合
    search.fit(X)
    # 断言最佳参数中的 bandwidth 值为 0.1
    assert search.best_params_["kerneldensity__bandwidth"] == 0.1


def test_kde_sample_weights():
    n_samples = 400
    size_test = 20
    # 创建一个全为 3.0 的权重数组，长度为 n_samples
    weights_neutral = np.full(n_samples, 3.0)
    for d in [1, 2, 10]:
        # 使用不同维度创建随机数生成器
        rng = np.random.RandomState(0)
        # 生成 n_samples 行 d 列的随机矩阵 X
        X = rng.rand(n_samples, d)
        # 计算样本权重，权重为 1 加上每行 X 元素之和乘以 10，转换为整数类型
        weights = 1 + (10 * X.sum(axis=1)).astype(np.int8)
        # 根据权重重复 X 的行，形成新的 X_repetitions
        X_repetitions = np.repeat(X, weights, axis=0)
        # 计算测试样本数，size_test 除以 d
        n_samples_test = size_test // d
        # 生成 n_samples_test 行 d 列的随机矩阵 test_points
        test_points = rng.rand(n_samples_test, d)
        # 遍历不同的算法和距离度量
        for algorithm in ["auto", "ball_tree", "kd_tree"]:
            for metric in ["euclidean", "minkowski", "manhattan", "chebyshev"]:
                # 如果算法不是 kd_tree 或者距离度量在 KDTree.valid_metrics 中
                if algorithm != "kd_tree" or metric in KDTree.valid_metrics:
                    # 创建 KernelDensity 对象 kde，使用指定的算法和距离度量
                    kde = KernelDensity(algorithm=algorithm, metric=metric)

                    # 测试添加常数样本权重是否不影响结果
                    kde.fit(X, sample_weight=weights_neutral)
                    # 计算使用测试点的分数
                    scores_const_weight = kde.score_samples(test_points)
                    # 生成使用随机种子的样本
                    sample_const_weight = kde.sample(random_state=1234)
                    # 重新用 X 拟合 kde，不使用样本权重
                    kde.fit(X)
                    # 计算不使用样本权重的测试点分数
                    scores_no_weight = kde.score_samples(test_points)
                    # 生成不使用样本权重的随机样本
                    sample_no_weight = kde.sample(random_state=1234)
                    # 断言两组分数非常接近
                    assert_allclose(scores_const_weight, scores_no_weight)
                    # 断言两组样本非常接近
                    assert_allclose(sample_const_weight, sample_no_weight)

                    # 测试样本权重和整数权重之间的等效性
                    kde.fit(X, sample_weight=weights)
                    # 计算使用权重的测试点分数
                    scores_weight = kde.score_samples(test_points)
                    # 生成使用权重的随机样本
                    sample_weight = kde.sample(random_state=1234)
                    # 用 X_repetitions 拟合 kde
                    kde.fit(X_repetitions)
                    # 计算使用 X_repetitions 的测试点分数
                    scores_ref_sampling = kde.score_samples(test_points)
                    # 生成使用 X_repetitions 的随机样本
                    sample_ref_sampling = kde.sample(random_state=1234)
                    # 断言两组分数非常接近
                    assert_allclose(scores_weight, scores_ref_sampling)
                    # 断言两组样本非常接近
                    assert_allclose(sample_weight, sample_ref_sampling)

                    # 测试样本权重是否有显著影响
                    diff = np.max(np.abs(scores_no_weight - scores_weight))
                    # 断言差异大于 0.001
                    assert diff > 0.001

                    # 测试相对于任意缩放的不变性
                    scale_factor = rng.rand()
                    # 使用缩放后的权重拟合 kde
                    kde.fit(X, sample_weight=(scale_factor * weights))
                    # 计算使用缩放后的权重的测试点分数
                    scores_scaled_weight = kde.score_samples(test_points)
                    # 断言两组分数非常接近
                    assert_allclose(scores_scaled_weight, scores_weight)
# 使用 pytest 的 parametrize 装饰器来定义多个参数化测试用例，其中 sample_weight 参数可以是 None 或包含浮点数列表
@pytest.mark.parametrize("sample_weight", [None, [0.1, 0.2, 0.3]])
def test_pickling(tmpdir, sample_weight):
    # 确保在序列化（pickling）前后预测结果保持一致。曾经因为未序列化 sample_weights 导致结果缺失信息的 bug。

    # 创建 KernelDensity 对象
    kde = KernelDensity()
    # 创建数据数组，将其变形为一列
    data = np.reshape([1.0, 2.0, 3.0], (-1, 1))
    # 使用给定的 sample_weight 参数拟合 KernelDensity 模型
    kde.fit(data, sample_weight=sample_weight)

    # 创建新的数据数组，同样将其变形为一列
    X = np.reshape([1.1, 2.1], (-1, 1))
    # 计算新数据 X 的分数（scores）
    scores = kde.score_samples(X)

    # 创建一个临时文件路径，用于保存序列化后的模型
    file_path = str(tmpdir.join("dump.pkl"))
    # 将 kde 对象序列化到文件中
    joblib.dump(kde, file_path)
    # 从文件中加载序列化后的 kde 对象
    kde = joblib.load(file_path)
    # 对新数据 X 再次计算分数（scores）
    scores_pickled = kde.score_samples(X)

    # 断言序列化前后的分数应该非常接近
    assert_allclose(scores, scores_pickled)


# 使用 pytest 的 parametrize 装饰器来定义多个参数化测试用例，其中 method 参数可以是 "score_samples" 或 "sample"
@pytest.mark.parametrize("method", ["score_samples", "sample"])
def test_check_is_fitted(method):
    # 检查未经拟合的估计器在调用 predict 方法时是否引发异常。
    # 未经拟合的估计器应该引发 NotFittedError 异常。
    rng = np.random.RandomState(0)
    # 创建随机数据 X
    X = rng.randn(10, 2)
    # 创建一个未经拟合的 KernelDensity 对象
    kde = KernelDensity()

    # 使用 pytest 的 raises 断言来检查是否引发 NotFittedError 异常
    with pytest.raises(NotFittedError):
        getattr(kde, method)(X)


# 使用 pytest 的 parametrize 装饰器来定义多个参数化测试用例，其中 bandwidth 参数可以是 "scott"、"silverman" 或浮点数
@pytest.mark.parametrize("bandwidth", ["scott", "silverman", 0.1])
def test_bandwidth(bandwidth):
    # 定义数据样本数和特征数
    n_samples, n_features = (100, 3)
    rng = np.random.RandomState(0)
    # 创建随机数据 X
    X = rng.randn(n_samples, n_features)
    # 创建一个带有指定 bandwidth 的 KernelDensity 对象，并拟合数据 X
    kde = KernelDensity(bandwidth=bandwidth).fit(X)
    # 从拟合的模型中抽取样本
    samp = kde.sample(100)
    # 计算拟合模型在数据 X 上的分数
    kde_sc = kde.score_samples(X)
    
    # 断言抽样数据 samp 的形状应该与 X 的形状相同
    assert X.shape == samp.shape
    # 断言拟合模型的分数形状应该为 (n_samples,)
    assert kde_sc.shape == (n_samples,)

    # 测试 self.bandwidth_ 属性是否具有预期的值
    if bandwidth == "scott":
        h = X.shape[0] ** (-1 / (X.shape[1] + 4))
    elif bandwidth == "silverman":
        h = (X.shape[0] * (X.shape[1] + 2) / 4) ** (-1 / (X.shape[1] + 4))
    else:
        h = bandwidth
    # 使用 pytest 的 approx 断言来检查 kde.bandwidth_ 属性是否接近预期值 h
    assert kde.bandwidth_ == pytest.approx(h)
```