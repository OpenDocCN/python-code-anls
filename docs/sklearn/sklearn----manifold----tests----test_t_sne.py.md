# `D:\src\scipysrc\scikit-learn\sklearn\manifold\tests\test_t_sne.py`

```
# 导入系统模块和字符串IO模块
import sys
from io import StringIO

# 导入第三方库：numpy、pytest、scipy.sparse
import numpy as np
import pytest
import scipy.sparse as sp

# 导入numpy测试工具和scipy优化模块
from numpy.testing import assert_allclose
from scipy.optimize import check_grad

# 导入scipy空间距离计算相关函数
from scipy.spatial.distance import pdist, squareform

# 导入sklearn相关模块
from sklearn import config_context
from sklearn.datasets import make_blobs
from sklearn.exceptions import EfficiencyWarning
from sklearn.manifold import (  # type: ignore
    TSNE,
    _barnes_hut_tsne,
)
from sklearn.manifold._t_sne import (
    _gradient_descent,
    _joint_probabilities,
    _joint_probabilities_nn,
    _kl_divergence,
    _kl_divergence_bh,
    trustworthiness,
)
from sklearn.manifold._utils import _binary_search_perplexity
from sklearn.metrics.pairwise import (
    cosine_distances,
    manhattan_distances,
    pairwise_distances,
)
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    ignore_warnings,
    skip_if_32bit,
)
from sklearn.utils.fixes import CSR_CONTAINERS, LIL_CONTAINERS

# 创建一个均匀分布的一维数组
x = np.linspace(0, 1, 10)

# 生成一个网格矩阵
xx, yy = np.meshgrid(x, x)

# 将网格矩阵中的点展平并组合成二维数组
X_2d_grid = np.hstack(
    [
        xx.ravel().reshape(-1, 1),
        yy.ravel().reshape(-1, 1),
    ]
)

# 定义测试函数：测试梯度下降的停止条件
def test_gradient_descent_stops():
    # 定义一个测试用的小梯度目标函数类
    class ObjectiveSmallGradient:
        def __init__(self):
            self.it = -1

        def __call__(self, _, compute_error=True):
            self.it += 1
            # 返回当前迭代的误差和梯度
            return (10 - self.it) / 10.0, np.array([1e-5])

    # 定义一个简单的平坦函数
    def flat_function(_, compute_error=True):
        # 返回固定的误差和梯度
        return 0.0, np.ones(1)

    # 测试梯度范数的停止条件
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        # 调用梯度下降函数
        _, error, it = _gradient_descent(
            ObjectiveSmallGradient(),
            np.zeros(1),
            0,
            max_iter=100,
            n_iter_without_progress=100,
            momentum=0.0,
            learning_rate=0.0,
            min_gain=0.0,
            min_grad_norm=1e-5,
            verbose=2,
        )
    finally:
        # 恢复标准输出并获取输出内容
        out = sys.stdout.getvalue()
        sys.stdout.close()
        sys.stdout = old_stdout

    # 断言：验证梯度下降结果符合预期
    assert error == 1.0
    assert it == 0
    assert "gradient norm" in out

    # 测试最大迭代次数但未有改进的停止条件
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        # 调用梯度下降函数
        _, error, it = _gradient_descent(
            flat_function,
            np.zeros(1),
            0,
            max_iter=100,
            n_iter_without_progress=10,
            momentum=0.0,
            learning_rate=0.0,
            min_gain=0.0,
            min_grad_norm=0.0,
            verbose=2,
        )
    finally:
        # 恢复标准输出并获取输出内容
        out = sys.stdout.getvalue()
        sys.stdout.close()
        sys.stdout = old_stdout

    # 断言：验证梯度下降结果符合预期
    assert error == 0.0
    assert it == 11
    assert "did not make any progress" in out
    # 最大迭代次数
    old_stdout = sys.stdout  # 保存原始的标准输出流
    sys.stdout = StringIO()  # 将标准输出重定向到一个字符串缓冲区
    try:
        # 调用梯度下降函数 `_gradient_descent` 进行优化
        _, error, it = _gradient_descent(
            ObjectiveSmallGradient(),  # 使用 ObjectiveSmallGradient 类作为优化的目标函数
            np.zeros(1),  # 初始化参数向量，这里是一个包含单个零元素的 NumPy 数组
            0,  # 初始步数（通常是第一步）
            max_iter=11,  # 最大迭代次数设定为 11
            n_iter_without_progress=100,  # 没有进展的最大迭代次数
            momentum=0.0,  # 动量参数设置为 0.0
            learning_rate=0.0,  # 学习率设置为 0.0
            min_gain=0.0,  # 最小增益阈值设定为 0.0
            min_grad_norm=0.0,  # 最小梯度范数设定为 0.0
            verbose=2,  # 详细输出级别设定为 2
        )
    finally:
        out = sys.stdout.getvalue()  # 获取从标准输出捕获的值（梯度下降过程中的输出）
        sys.stdout.close()  # 关闭重定向的标准输出流
        sys.stdout = old_stdout  # 恢复原始的标准输出流
    assert error == 0.0  # 断言误差为 0.0，即优化过程应当收敛
    assert it == 10  # 断言迭代次数为 10，确认达到了预期的迭代次数
    assert "Iteration 10" in out  # 断言输出中包含字符串 "Iteration 10"，用于确认迭代到第 10 步
def test_binary_search():
    # 测试二分查找是否找到具有所需困惑度的高斯分布。
    random_state = check_random_state(0)
    # 使用随机种子生成随机数据，形状为 (50, 5)
    data = random_state.randn(50, 5)
    # 计算数据点之间的距离，并将其转换为 np.float32 类型
    distances = pairwise_distances(data).astype(np.float32)
    # 所需的困惑度值
    desired_perplexity = 25.0
    # 使用二分查找算法计算困惑度 P
    P = _binary_search_perplexity(distances, desired_perplexity, verbose=0)
    # 将 P 中的每个元素与 np.finfo(np.double).eps 取最大值
    P = np.maximum(P, np.finfo(np.double).eps)
    # 计算平均困惑度
    mean_perplexity = np.mean(
        [np.exp(-np.sum(P[i] * np.log(P[i]))) for i in range(P.shape[0])]
    )
    # 断言平均困惑度与期望值接近，精度为小数点后三位
    assert_almost_equal(mean_perplexity, desired_perplexity, decimal=3)


def test_binary_search_underflow():
    # 测试二分查找是否找到具有所需困惑度的高斯分布。
    # 这是一个更具挑战性的情况，可能会在浮点精度上产生数值下溢 (见问题 #19471 和 PR #19472)。
    random_state = check_random_state(42)
    # 使用随机种子生成随机数据，形状为 (1, 90)，并转换为 np.float32 类型
    data = random_state.randn(1, 90).astype(np.float32) + 100
    # 所需的困惑度值
    desired_perplexity = 30.0
    # 使用二分查找算法计算困惑度 P
    P = _binary_search_perplexity(data, desired_perplexity, verbose=0)
    # 计算困惑度
    perplexity = 2 ** -np.nansum(P[0, 1:] * np.log2(P[0, 1:]))
    # 断言困惑度与期望值接近，精度为小数点后三位
    assert_almost_equal(perplexity, desired_perplexity, decimal=3)


def test_binary_search_neighbors():
    # 二分困惑度搜索的近似方法。
    # 当使用所有点作为邻居时，应与慢速方法大致相等。
    n_samples = 200
    desired_perplexity = 25.0
    random_state = check_random_state(0)
    # 使用随机种子生成随机数据，形状为 (n_samples, 2)，并转换为 np.float32 类型
    data = random_state.randn(n_samples, 2).astype(np.float32, copy=False)
    # 计算数据点之间的距离
    distances = pairwise_distances(data)
    # 使用二分查找算法计算困惑度 P1
    P1 = _binary_search_perplexity(distances, desired_perplexity, verbose=0)

    # 测试当使用所有邻居时结果是否相同
    n_neighbors = n_samples - 1
    nn = NearestNeighbors().fit(data)
    # 构建邻居距离图
    distance_graph = nn.kneighbors_graph(n_neighbors=n_neighbors, mode="distance")
    distances_nn = distance_graph.data.astype(np.float32, copy=False)
    distances_nn = distances_nn.reshape(n_samples, n_neighbors)
    # 使用二分查找算法计算困惑度 P2
    P2 = _binary_search_perplexity(distances_nn, desired_perplexity, verbose=0)

    indptr = distance_graph.indptr
    # 从 P1 中选择与每个数据点的邻居对应的困惑度值 P1_nn
    P1_nn = np.array(
        [
            P1[k, distance_graph.indices[indptr[k] : indptr[k + 1]]]
            for k in range(n_samples)
        ]
    )
    # 断言 P1_nn 与 P2 数组几乎相等，精度为小数点后四位
    assert_array_almost_equal(P1_nn, P2, decimal=4)

    # 测试当使用较少邻居时，最高的 P_ij 是否相同
    # 使用 numpy 的 linspace 函数生成从 150 到 n_samples - 1 的五个等间距的数值，作为 k 的取值
    for k in np.linspace(150, n_samples - 1, 5):
        # 将 k 转换为整数
        k = int(k)
        # 计算需要检查的条目数量，即 top 10 * k，在 k * k 条目中检查
        topn = k * 10  
        # 使用最近邻算法生成距离图，以距离模式返回距离矩阵
        distance_graph = nn.kneighbors_graph(n_neighbors=k, mode="distance")
        # 将距离数据转换为 np.float32 类型的数组，并保留原始数据（不进行拷贝）
        distances_nn = distance_graph.data.astype(np.float32, copy=False)
        # 将距离数据重新整形为 n_samples x k 的数组
        distances_nn = distances_nn.reshape(n_samples, k)
        # 使用二分搜索算法计算给定困惑度的 P2k 值，关闭详细输出
        P2k = _binary_search_perplexity(distances_nn, desired_perplexity, verbose=0)
        # 断言 P1_nn 和 P2 两者在小数点后两位精度上近似相等
        assert_array_almost_equal(P1_nn, P2, decimal=2)
        # 对 P1 中的值按降序排列，并取出前 topn 个值
        idx = np.argsort(P1.ravel())[::-1]
        P1top = P1.ravel()[idx][:topn]
        # 对 P2k 中的值按降序排列，并取出前 topn 个值
        idx = np.argsort(P2k.ravel())[::-1]
        P2top = P2k.ravel()[idx][:topn]
        # 断言 P1top 和 P2top 两者在小数点后两位精度上近似相等
        assert_array_almost_equal(P1top, P2top, decimal=2)
def test_binary_perplexity_stability():
    # Binary perplexity search should be stable.
    # 二进制困惑度搜索应该是稳定的。
    # The binary_search_perplexity had a bug wherein the P array
    # was uninitialized, leading to sporadically failing tests.
    # binary_search_perplexity 存在一个 bug，其中 P 数组未初始化，
    # 导致偶尔测试失败。
    n_neighbors = 10
    # 设定最近邻数为 10
    n_samples = 100
    # 设定样本数为 100
    random_state = check_random_state(0)
    # 使用种子 0 来初始化随机状态
    data = random_state.randn(n_samples, 5)
    # 生成符合正态分布的随机数据，形状为 (100, 5)
    nn = NearestNeighbors().fit(data)
    # 使用随机数据训练最近邻模型
    distance_graph = nn.kneighbors_graph(n_neighbors=n_neighbors, mode="distance")
    # 获取最近邻距离的稀疏图
    distances = distance_graph.data.astype(np.float32, copy=False)
    # 将稀疏图数据转换为浮点数数组
    distances = distances.reshape(n_samples, n_neighbors)
    # 重塑数据为 (100, 10) 的形状
    last_P = None
    # 初始化上一次的 P 为 None
    desired_perplexity = 3
    # 设定期望的困惑度为 3
    for _ in range(100):
        # 循环 100 次
        P = _binary_search_perplexity(distances.copy(), desired_perplexity, verbose=0)
        # 使用二进制搜索方法计算困惑度 P
        P1 = _joint_probabilities_nn(distance_graph, desired_perplexity, verbose=0)
        # 使用最近邻方法计算联合概率 P1
        # Convert the sparse matrix to a dense one for testing
        # 将稀疏矩阵转换为密集矩阵进行测试
        P1 = P1.toarray()
        if last_P is None:
            # 如果 last_P 是 None
            last_P = P
            last_P1 = P1
        else:
            # 否则进行数组近似相等断言
            assert_array_almost_equal(P, last_P, decimal=4)
            assert_array_almost_equal(P1, last_P1, decimal=4)


def test_gradient():
    # Test gradient of Kullback-Leibler divergence.
    # 测试 Kullback-Leibler 散度的梯度。
    random_state = check_random_state(0)
    # 使用种子 0 来初始化随机状态

    n_samples = 50
    # 设定样本数为 50
    n_features = 2
    # 设定特征数为 2
    n_components = 2
    # 设定组件数为 2
    alpha = 1.0
    # 设定 alpha 值为 1.0

    distances = random_state.randn(n_samples, n_features).astype(np.float32)
    # 生成符合正态分布的随机数据，形状为 (50, 2)，转换为浮点数数组
    distances = np.abs(distances.dot(distances.T))
    # 计算距离矩阵的平方和，并取绝对值
    np.fill_diagonal(distances, 0.0)
    # 将对角线元素填充为 0
    X_embedded = random_state.randn(n_samples, n_components).astype(np.float32)
    # 生成符合正态分布的随机嵌入数据，形状为 (50, 2)，转换为浮点数数组

    P = _joint_probabilities(distances, desired_perplexity=25.0, verbose=0)
    # 计算距离的困惑度为 25.0 时的联合概率矩阵 P

    def fun(params):
        return _kl_divergence(params, P, alpha, n_samples, n_components)[0]

    def grad(params):
        return _kl_divergence(params, P, alpha, n_samples, n_components)[1]

    assert_almost_equal(check_grad(fun, grad, X_embedded.ravel()), 0.0, decimal=5)


def test_trustworthiness():
    # Test trustworthiness score.
    # 测试可信度评分。
    random_state = check_random_state(0)
    # 使用种子 0 来初始化随机状态

    # Affine transformation
    # 仿射变换
    X = random_state.randn(100, 2)
    # 生成符合正态分布的随机数据，形状为 (100, 2)
    assert trustworthiness(X, 5.0 + X / 10.0) == 1.0
    # 断言仿射变换后的可信度评分为 1.0

    # Randomly shuffled
    # 随机打乱顺序
    X = np.arange(100).reshape(-1, 1)
    # 生成序列 0 到 99 的数组，并重塑为列向量
    X_embedded = X.copy()
    random_state.shuffle(X_embedded)
    # 随机打乱 X_embedded 的顺序
    assert trustworthiness(X, X_embedded) < 0.6
    # 断言打乱顺序后的可信度评分小于 0.6

    # Completely different
    # 完全不同的数据
    X = np.arange(5).reshape(-1, 1)
    # 生成序列 0 到 4 的数组，并重塑为列向量
    X_embedded = np.array([[0], [2], [4], [1], [3]])
    # 手动设置一个不同的嵌入
    assert_almost_equal(trustworthiness(X, X_embedded, n_neighbors=1), 0.2)


def test_trustworthiness_n_neighbors_error():
    """Raise an error when n_neighbors >= n_samples / 2.

    Non-regression test for #18567.
    """
    regex = "n_neighbors .+ should be less than .+"
    # 设置正则表达式，匹配 n_neighbors 错误信息
    rng = np.random.RandomState(42)
    # 使用种子 42 来初始化随机状态
    X = rng.rand(7, 4)
    # 生成形状为 (7, 4) 的随机数组
    X_embedded = rng.rand(7, 2)
    # 生成形状为 (7, 2) 的随机嵌入数组
    with pytest.raises(ValueError, match=regex):
        # 使用 pytest 断言捕获 ValueError 异常，并匹配错误信息
        trustworthiness(X, X_embedded, n_neighbors=5)
        # 调用 trustworthiness 函数并设定 n_neighbors 为 5

    trust = trustworthiness(X, X_embedded, n_neighbors=3)
    # 调用 trustworthiness 函数并设定 n_neighbors 为 3，计算可信度评分
    assert 0 <= trust <= 1
    # 断言可信度评分在 0 到 1 之间
# 使用 pytest 的参数化功能，为不同的方法(method)和初始化策略(init)组合执行测试
@pytest.mark.parametrize("method", ["exact", "barnes_hut"])
@pytest.mark.parametrize("init", ("random", "pca"))
def test_preserve_trustworthiness_approximately(method, init):
    # 设置随机种子为0，确保结果可重复
    random_state = check_random_state(0)
    # 定义数据集维度和样本数
    n_components = 2
    X = random_state.randn(50, n_components).astype(np.float32)
    # 初始化 t-SNE 模型，配置各项参数
    tsne = TSNE(
        n_components=n_components,
        init=init,
        random_state=0,
        method=method,
        max_iter=700,
        learning_rate="auto",
    )
    # 对数据进行降维处理
    X_embedded = tsne.fit_transform(X)
    # 计算降维后数据的可信度，并断言其大于0.85
    t = trustworthiness(X, X_embedded, n_neighbors=1)
    assert t > 0.85


def test_optimization_minimizes_kl_divergence():
    """t-SNE 应随着迭代次数增加，KL 散度值应逐渐降低。"""
    # 设置随机种子为0，确保结果可重复
    random_state = check_random_state(0)
    # 生成一个三维特征的样本集合
    X, _ = make_blobs(n_features=3, random_state=random_state)
    kl_divergences = []
    # 对不同的最大迭代次数进行测试
    for max_iter in [250, 300, 350]:
        # 初始化 t-SNE 模型，配置各项参数
        tsne = TSNE(
            n_components=2,
            init="random",
            perplexity=10,
            learning_rate=100.0,
            max_iter=max_iter,
            random_state=0,
        )
        # 对数据进行降维处理
        tsne.fit_transform(X)
        # 记录当前迭代下的 KL 散度值
        kl_divergences.append(tsne.kl_divergence_)
    # 断言 KL 散度值随迭代次数的增加而减小
    assert kl_divergences[1] <= kl_divergences[0]
    assert kl_divergences[2] <= kl_divergences[1]


@pytest.mark.parametrize("method", ["exact", "barnes_hut"])
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_fit_transform_csr_matrix(method, csr_container):
    # TODO: compare results on dense and sparse data as proposed in:
    # https://github.com/scikit-learn/scikit-learn/pull/23585#discussion_r968388186
    # X can be a sparse matrix.
    # 设置随机种子为0，确保结果可重复
    rng = check_random_state(0)
    # 生成一个二维样本集合
    X = rng.randn(50, 2)
    # 将部分数据置为稀疏值
    X[(rng.randint(0, 50, 25), rng.randint(0, 2, 25))] = 0.0
    # 将数据转换为稀疏矩阵格式
    X_csr = csr_container(X)
    # 初始化 t-SNE 模型，配置各项参数
    tsne = TSNE(
        n_components=2,
        init="random",
        perplexity=10,
        learning_rate=100.0,
        random_state=0,
        method=method,
        max_iter=750,
    )
    # 对稀疏矩阵进行降维处理
    X_embedded = tsne.fit_transform(X_csr)
    # 断言稀疏矩阵降维后的可信度接近1.0
    assert_allclose(trustworthiness(X_csr, X_embedded, n_neighbors=1), 1.0, rtol=1.1e-1)


def test_preserve_trustworthiness_approximately_with_precomputed_distances():
    # Nearest neighbors should be preserved approximately.
    # 设置随机种子为0，确保结果可重复
    random_state = check_random_state(0)
    for i in range(3):
        # 生成一个二维样本集合
        X = random_state.randn(80, 2)
        # 计算样本集合的欧氏距离的平方，并生成距离矩阵
        D = squareform(pdist(X), "sqeuclidean")
        # 初始化 t-SNE 模型，配置各项参数
        tsne = TSNE(
            n_components=2,
            perplexity=2,
            learning_rate=100.0,
            early_exaggeration=2.0,
            metric="precomputed",
            random_state=i,
            verbose=0,
            max_iter=500,
            init="random",
        )
        # 对预计算的距离矩阵进行降维处理
        X_embedded = tsne.fit_transform(D)
        # 计算降维后数据的可信度，并断言其大于0.95
        t = trustworthiness(D, X_embedded, n_neighbors=1, metric="precomputed")
        assert t > 0.95


def test_trustworthiness_not_euclidean_metric():
    # This test is not implemented yet.
    pass
    # 使用一个与 'euclidean' 和 'precomputed' 不同的度量标准来测试可信度
    random_state = check_random_state(0)
    # 生成一个 100x2 的随机数矩阵 X
    X = random_state.randn(100, 2)
    # 断言：使用余弦距离作为度量标准时，trustworthiness 应该等于
    # 使用 X 与自身计算余弦距离得到的距离矩阵与 X 作为预计算距离矩阵时的 trustworthiness
    assert trustworthiness(X, X, metric="cosine") == trustworthiness(
        pairwise_distances(X, metric="cosine"), X, metric="precomputed"
    )
@pytest.mark.parametrize(
    "method, retype",
    [
        ("exact", np.asarray),  # 使用精确方法和 np.asarray 函数
        ("barnes_hut", np.asarray),  # 使用 Barnes-Hut 方法和 np.asarray 函数
        *[("barnes_hut", csr_container) for csr_container in CSR_CONTAINERS],  # 使用 Barnes-Hut 方法和 CSR_CONTAINERS 中的每个容器
    ],
)
@pytest.mark.parametrize(
    "D, message_regex",
    [
        ([[0.0], [1.0]], ".* square distance matrix"),  # 使用 0.0 和 1.0 构建距离矩阵，匹配正则表达式 ".* square distance matrix"
        ([[0.0, -1.0], [1.0, 0.0]], ".* positive.*"),  # 使用 0.0, -1.0, 1.0, 0.0 构建距离矩阵，匹配正则表达式 ".* positive.*"
    ],
)
def test_bad_precomputed_distances(method, D, retype, message_regex):
    tsne = TSNE(
        metric="precomputed",  # 使用预先计算的距离度量
        method=method,  # 使用给定的方法
        init="random",  # 随机初始化
        random_state=42,  # 随机种子为 42
        perplexity=1,  # 困惑度为 1
    )
    with pytest.raises(ValueError, match=message_regex):  # 检测是否引发 ValueError，并匹配指定的消息正则表达式
        tsne.fit_transform(retype(D))  # 对给定的距离数据进行拟合转换


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_exact_no_precomputed_sparse(csr_container):
    tsne = TSNE(
        metric="precomputed",  # 使用预先计算的距离度量
        method="exact",  # 使用精确方法
        init="random",  # 随机初始化
        random_state=42,  # 随机种子为 42
        perplexity=1,  # 困惑度为 1
    )
    with pytest.raises(TypeError, match="sparse"):  # 检测是否引发 TypeError，并匹配字符串 "sparse"
        tsne.fit_transform(csr_container([[0, 5], [5, 0]]))  # 对给定的稀疏容器和距离数据进行拟合转换


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_high_perplexity_precomputed_sparse_distances(csr_container):
    # 困惑度应小于 50
    dist = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    bad_dist = csr_container(dist)
    tsne = TSNE(metric="precomputed", init="random", random_state=42, perplexity=1)
    msg = "3 neighbors per samples are required, but some samples have only 1"
    with pytest.raises(ValueError, match=msg):  # 检测是否引发 ValueError，并匹配指定的消息
        tsne.fit_transform(bad_dist)  # 对给定的不良距离数据进行拟合转换


@ignore_warnings(category=EfficiencyWarning)
@pytest.mark.parametrize("sparse_container", CSR_CONTAINERS + LIL_CONTAINERS)
def test_sparse_precomputed_distance(sparse_container):
    """确保 TSNE 对稀疏和密集矩阵的工作完全一致"""
    random_state = check_random_state(0)
    X = random_state.randn(100, 2)

    D_sparse = kneighbors_graph(X, n_neighbors=100, mode="distance", include_self=True)
    D = pairwise_distances(X)
    assert sp.issparse(D_sparse)  # 断言 D_sparse 是稀疏矩阵
    assert_almost_equal(D_sparse.toarray(), D)  # 断言 D_sparse 的稀疏表示与 D 的密集表示几乎相等

    tsne = TSNE(
        metric="precomputed",  # 使用预先计算的距离度量
        random_state=0,  # 随机种子为 0
        init="random",  # 随机初始化
        learning_rate="auto"  # 学习率自动调整
    )
    Xt_dense = tsne.fit_transform(D)  # 对密集距离矩阵进行拟合转换

    Xt_sparse = tsne.fit_transform(sparse_container(D_sparse))  # 对稀疏容器的距离矩阵进行拟合转换
    assert_almost_equal(Xt_dense, Xt_sparse)  # 断言密集和稀疏转换结果几乎相等


def test_non_positive_computed_distances():
    # 计算的距离矩阵必须是正数的。
    def metric(x, y):
        return -1  # 返回负数的度量

    # 即使结果平方后也应捕获到负数的计算距离
    tsne = TSNE(metric=metric, method="exact", perplexity=1)
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="All distances .*metric given.*"):  # 检测是否引发 ValueError，并匹配指定的消息
        tsne.fit_transform(X)  # 对给定数据进行拟合转换


def test_init_ndarray():
    # 使用 ndarray 初始化 TSNE 并测试拟合
    tsne = TSNE(init=np.zeros((100, 2)), learning_rate="auto")  # 使用全零 ndarray 初始化 TSNE
    X_embedded = tsne.fit_transform(np.ones((100, 5)))  # 对全一数据进行拟合转换
    # 使用 numpy 库创建一个形状为 (100, 2) 的零数组，并将其与 X_embedded 进行比较
    assert_array_equal(np.zeros((100, 2)), X_embedded)
def test_init_ndarray_precomputed():
    # 初始化 TSNE 对象，使用 ndarray 和度量方式 'precomputed'
    # 确保在 _fit 方法中不会引发 FutureWarning
    tsne = TSNE(
        init=np.zeros((100, 2)),  # 初始化为一个 100x2 的零矩阵
        metric="precomputed",  # 使用预先计算的距离矩阵作为度量方式
        learning_rate=50.0,
    )
    tsne.fit(np.zeros((100, 100)))  # 对一个100x100的零矩阵进行拟合操作


def test_pca_initialization_not_compatible_with_precomputed_kernel():
    # 预先计算的距离矩阵不能使用 PCA 初始化
    tsne = TSNE(metric="precomputed", init="pca", perplexity=1)
    with pytest.raises(
        ValueError,
        match='The parameter init="pca" cannot be used with metric="precomputed".',
    ):
        tsne.fit_transform(np.array([[0.0], [1.0]]))


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_pca_initialization_not_compatible_with_sparse_input(csr_container):
    # 稀疏输入矩阵不能使用 PCA 初始化
    tsne = TSNE(init="pca", learning_rate=100.0, perplexity=1)
    with pytest.raises(TypeError, match="PCA initialization.*"):
        tsne.fit_transform(csr_container([[0, 5], [5, 0]]))


def test_n_components_range():
    # barnes_hut 方法仅可用于 n_components <= 3
    tsne = TSNE(n_components=4, method="barnes_hut", perplexity=1)
    with pytest.raises(ValueError, match="'n_components' should be .*"):
        tsne.fit_transform(np.array([[0.0], [1.0]]))


def test_early_exaggeration_used():
    # 检查 ``early_exaggeration`` 参数是否有效
    random_state = check_random_state(0)
    n_components = 2
    methods = ["exact", "barnes_hut"]
    X = random_state.randn(25, n_components).astype(np.float32)
    for method in methods:
        tsne = TSNE(
            n_components=n_components,
            perplexity=1,
            learning_rate=100.0,
            init="pca",
            random_state=0,
            method=method,
            early_exaggeration=1.0,  # 设置 early_exaggeration 为 1.0
            max_iter=250,
        )
        X_embedded1 = tsne.fit_transform(X)
        tsne = TSNE(
            n_components=n_components,
            perplexity=1,
            learning_rate=100.0,
            init="pca",
            random_state=0,
            method=method,
            early_exaggeration=10.0,  # 设置 early_exaggeration 为 10.0
            max_iter=250,
        )
        X_embedded2 = tsne.fit_transform(X)

        assert not np.allclose(X_embedded1, X_embedded2)


def test_max_iter_used():
    # 检查 ``max_iter`` 参数是否有效
    random_state = check_random_state(0)
    n_components = 2
    methods = ["exact", "barnes_hut"]
    X = random_state.randn(25, n_components).astype(np.float32)
    # 对于给定的降维方法列表中的每一种方法，依次进行处理
    for method in methods:
        # 对于每个最大迭代次数进行迭代，分别为251和500
        for max_iter in [251, 500]:
            # 创建一个TSNE对象，设置其参数
            tsne = TSNE(
                n_components=n_components,       # 设置降维后的维度数目
                perplexity=1,                    # 困惑度参数
                learning_rate=0.5,               # 学习率
                init="random",                   # 初始化方式为随机
                random_state=0,                  # 随机数种子
                method=method,                   # 当前使用的降维方法
                early_exaggeration=1.0,          # 初始增强参数
                max_iter=max_iter,               # 最大迭代次数
            )
            # 使用数据X进行降维转换
            tsne.fit_transform(X)

            # 断言当前TSNE对象的实际迭代次数等于设定的最大迭代次数减一
            assert tsne.n_iter_ == max_iter - 1
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_answer_gradient_two_points(csr_container):
    # 使用参数化测试，循环每个 CSR 容器
    #
    # 这些测试和结果已经通过 LvdM 的参考实现进行了验证。
    pos_input = np.array([[1.0, 0.0], [0.0, 1.0]])
    # 输入的位置信息数组
    pos_output = np.array(
        [[-4.961291e-05, -1.072243e-04], [9.259460e-05, 2.702024e-04]]
    )
    # 预期的位置输出数组
    neighbors = np.array([[1], [0]])
    # 邻居索引数组
    grad_output = np.array(
        [[-2.37012478e-05, -6.29044398e-05], [2.37012478e-05, 6.29044398e-05]]
    )
    # 预期的梯度输出数组
    _run_answer_test(pos_input, pos_output, neighbors, grad_output, csr_container)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_answer_gradient_four_points(csr_container):
    # 四个点测试树结构的多层子节点。
    #
    # 这些测试和结果已经通过 LvdM 的参考实现进行了验证。
    pos_input = np.array([[1.0, 0.0], [0.0, 1.0], [5.0, 2.0], [7.3, 2.2]])
    # 输入的位置信息数组
    pos_output = np.array(
        [
            [6.080564e-05, -7.120823e-05],
            [-1.718945e-04, -4.000536e-05],
            [-2.271720e-04, 8.663310e-05],
            [-1.032577e-04, -3.582033e-05],
        ]
    )
    # 预期的位置输出数组
    neighbors = np.array([[1, 2, 3], [0, 2, 3], [1, 0, 3], [1, 2, 0]])
    # 邻居索引数组
    grad_output = np.array(
        [
            [5.81128448e-05, -7.78033454e-06],
            [-5.81526851e-05, 7.80976444e-06],
            [4.24275173e-08, -3.69569698e-08],
            [-2.58720939e-09, 7.52706374e-09],
        ]
    )
    # 预期的梯度输出数组
    _run_answer_test(pos_input, pos_output, neighbors, grad_output, csr_container)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_skip_num_points_gradient(csr_container):
    # 测试关键字参数 skip_num_points。
    #
    # skip_num_points 应该使 Barnes_hut 梯度在低于 skip_num_point 的索引下不计算。
    # 除了 skip_num_points=2 和前两个梯度行被设置为零外，这些数据点与 test_answer_gradient_four_points() 中的相同。
    pos_input = np.array([[1.0, 0.0], [0.0, 1.0], [5.0, 2.0], [7.3, 2.2]])
    # 输入的位置信息数组
    pos_output = np.array(
        [
            [6.080564e-05, -7.120823e-05],
            [-1.718945e-04, -4.000536e-05],
            [-2.271720e-04, 8.663310e-05],
            [-1.032577e-04, -3.582033e-05],
        ]
    )
    # 预期的位置输出数组
    neighbors = np.array([[1, 2, 3], [0, 2, 3], [1, 0, 3], [1, 2, 0]])
    # 邻居索引数组
    grad_output = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [4.24275173e-08, -3.69569698e-08],
            [-2.58720939e-09, 7.52706374e-09],
        ]
    )
    # 预期的梯度输出数组
    _run_answer_test(
        pos_input, pos_output, neighbors, grad_output, csr_container, False, 0.1, 2
    )


def _run_answer_test(
    pos_input,
    pos_output,
    neighbors,
    grad_output,
    csr_container,
    verbose=False,
    perplexity=0.1,
    skip_num_points=0,
):
    # 内部函数，运行答案测试
    # 计算输入位置的两两距离，并转换为32位浮点数
    distances = pairwise_distances(pos_input).astype(np.float32)
    
    # 将距离矩阵、困惑度和详细输出标志组成参数元组
    args = distances, perplexity, verbose
    
    # 将输出位置数组转换为32位浮点数
    pos_output = pos_output.astype(np.float32)
    
    # 将邻居索引数组转换为64位整数类型，同时确保不进行复制操作
    neighbors = neighbors.astype(np.int64, copy=False)
    
    # 计算输入概率分布的联合概率分布
    pij_input = _joint_probabilities(*args)
    
    # 将平方形式的联合概率分布转换为32位浮点数
    pij_input = squareform(pij_input).astype(np.float32)
    
    # 创建一个与输出位置数组形状相同的全零数组，数据类型为32位浮点数，用于存储梯度
    grad_bh = np.zeros(pos_output.shape, dtype=np.float32)
    
    # 使用csr_matrix容器封装联合概率分布，以便有效地处理稀疏矩阵
    P = csr_container(pij_input)
    
    # 将P中的数据索引转换为64位整数类型
    neighbors = P.indices.astype(np.int64)
    
    # 将P中的行指针转换为64位整数类型
    indptr = P.indptr.astype(np.int64)
    
    # 调用_barnes_hut_tsne模块的gradient函数，计算梯度
    _barnes_hut_tsne.gradient(
        P.data, pos_output, neighbors, indptr, grad_bh, 0.5, 2, 1, skip_num_points=0
    )
    
    # 断言输出的梯度数组与预期的grad_output数组在小数点后4位精度下近似相等
    assert_array_almost_equal(grad_bh, grad_output, decimal=4)
def test_verbose():
    # Verbose options write to stdout.
    # 设置随机种子为0，保证结果可复现性
    random_state = check_random_state(0)
    # 创建一个 t-SNE 对象，设置 verbose 等级为2，perplexity 为4
    tsne = TSNE(verbose=2, perplexity=4)
    # 生成一个 5x2 的随机数组 X
    X = random_state.randn(5, 2)

    # 保存当前的标准输出对象
    old_stdout = sys.stdout
    # 将标准输出重定向到一个内存中的字符串对象
    sys.stdout = StringIO()
    try:
        # 执行 t-SNE 的拟合和转换操作，并将输出捕获到内存中的字符串对象中
        tsne.fit_transform(X)
    finally:
        # 获取捕获的输出
        out = sys.stdout.getvalue()
        # 关闭捕获输出的字符串对象
        sys.stdout.close()
        # 恢复原来的标准输出对象
        sys.stdout = old_stdout

    # 断言捕获的输出中包含特定的关键词
    assert "[t-SNE]" in out
    assert "nearest neighbors..." in out
    assert "Computed conditional probabilities" in out
    assert "Mean sigma" in out
    assert "early exaggeration" in out


def test_chebyshev_metric():
    # t-SNE should allow metrics that cannot be squared (issue #3526).
    # 设置随机种子为0，保证结果可复现性
    random_state = check_random_state(0)
    # 创建一个 t-SNE 对象，使用切比雪夫距离作为度量方式，perplexity 为4
    tsne = TSNE(metric="chebyshev", perplexity=4)
    # 生成一个 5x2 的随机数组 X
    X = random_state.randn(5, 2)
    # 执行 t-SNE 的拟合和转换操作
    tsne.fit_transform(X)


def test_reduction_to_one_component():
    # t-SNE should allow reduction to one component (issue #4154).
    # 设置随机种子为0，保证结果可复现性
    random_state = check_random_state(0)
    # 创建一个 t-SNE 对象，设置降维到1个组件，perplexity 为4
    tsne = TSNE(n_components=1, perplexity=4)
    # 生成一个 5x2 的随机数组 X
    X = random_state.randn(5, 2)
    # 执行 t-SNE 的拟合操作，并获取嵌入结果
    X_embedded = tsne.fit(X).embedding_
    # 断言嵌入结果中的所有值都是有限的
    assert np.all(np.isfinite(X_embedded))


@pytest.mark.parametrize("method", ["barnes_hut", "exact"])
@pytest.mark.parametrize("dt", [np.float32, np.float64])
def test_64bit(method, dt):
    # Ensure 64bit arrays are handled correctly.
    # 设置随机种子为0，保证结果可复现性
    random_state = check_random_state(0)

    # 生成一个 10x2 的随机数组 X，数据类型为指定的 dt
    X = random_state.randn(10, 2).astype(dt, copy=False)
    # 创建一个 t-SNE 对象，设置 n_components 为2，perplexity 为2，学习率为100.0，
    # 随机种子为0，优化方法为 method 指定的方法，verbose 等级为0，最大迭代次数为300，
    # 初始化方式为随机初始化
    tsne = TSNE(
        n_components=2,
        perplexity=2,
        learning_rate=100.0,
        random_state=0,
        method=method,
        verbose=0,
        max_iter=300,
        init="random",
    )
    # 执行 t-SNE 的拟合和转换操作，并获取嵌入结果
    X_embedded = tsne.fit_transform(X)
    # 获取嵌入结果的数据类型
    effective_type = X_embedded.dtype

    # 断言嵌入结果的数据类型为 np.float32
    # 因为 t-SNE 的 Cython 实现只支持单精度，无论输入的数据类型如何，输出都是单精度
    assert effective_type == np.float32


@pytest.mark.parametrize("method", ["barnes_hut", "exact"])
def test_kl_divergence_not_nan(method):
    # Ensure kl_divergence_ is computed at last iteration
    # even though max_iter % n_iter_check != 0, i.e. 1003 % 50 != 0
    # 设置随机种子为0，保证结果可复现性
    random_state = check_random_state(0)

    # 生成一个 50x2 的随机数组 X
    X = random_state.randn(50, 2)
    # 创建一个 t-SNE 对象，设置 n_components 为2，perplexity 为2，学习率为100.0，
    # 随机种子为0，优化方法为 method 指定的方法，verbose 等级为0，最大迭代次数为503，
    # 初始化方式为随机初始化
    tsne = TSNE(
        n_components=2,
        perplexity=2,
        learning_rate=100.0,
        random_state=0,
        method=method,
        verbose=0,
        max_iter=503,
        init="random",
    )
    # 执行 t-SNE 的拟合和转换操作
    tsne.fit_transform(X)

    # 断言 kl_divergence_ 不是 NaN
    assert not np.isnan(tsne.kl_divergence_)


def test_barnes_hut_angle():
    # When Barnes-Hut's angle=0 this corresponds to the exact method.
    # 设置角度为0，即使用 Barnes-Hut 方法时等同于 exact 方法
    angle = 0.0
    perplexity = 10
    n_samples = 100
    # 对每个给定的成分数进行迭代，这里是2和3
    for n_components in [2, 3]:
        # 设置特征数量为5
        n_features = 5
        # 计算自由度，这里是成分数减去1.0
        degrees_of_freedom = float(n_components - 1.0)

        # 使用随机状态生成器检查随机状态，并生成随机数据
        random_state = check_random_state(0)
        data = random_state.randn(n_samples, n_features)
        
        # 计算数据点之间的距离
        distances = pairwise_distances(data)
        
        # 使用随机状态生成器生成参数
        params = random_state.randn(n_samples, n_components)
        
        # 计算联合概率分布P
        P = _joint_probabilities(distances, perplexity, verbose=0)
        
        # 计算精确的KL散度和其梯度
        kl_exact, grad_exact = _kl_divergence(
            params, P, degrees_of_freedom, n_samples, n_components
        )

        # 确定最近邻的数量为样本数减去1
        n_neighbors = n_samples - 1
        
        # 使用最近邻方法计算数据点之间的距离
        distances_csr = (
            NearestNeighbors()
            .fit(data)
            .kneighbors_graph(n_neighbors=n_neighbors, mode="distance")
        )
        
        # 计算基于最近邻的联合概率分布P_bh
        P_bh = _joint_probabilities_nn(distances_csr, perplexity, verbose=0)
        
        # 计算基于最近邻的KL散度和其梯度
        kl_bh, grad_bh = _kl_divergence_bh(
            params,
            P_bh,
            degrees_of_freedom,
            n_samples,
            n_components,
            angle=angle,
            skip_num_points=0,
            verbose=0,
        )

        # 将P转换为方阵形式
        P = squareform(P)
        # 将P_bh转换为密集数组形式
        P_bh = P_bh.toarray()
        
        # 断言P_bh与P在小数点后五位上几乎相等
        assert_array_almost_equal(P_bh, P, decimal=5)
        
        # 断言精确的KL散度与基于最近邻的KL散度在小数点后三位上几乎相等
        assert_almost_equal(kl_exact, kl_bh, decimal=3)
# 装饰器，用于跳过 32 位系统的测试函数
@skip_if_32bit
# 定义测试函数，测试 n_iter_without_progress 参数的影响
def test_n_iter_without_progress():
    # 使用固定的随机种子初始化随机状态
    random_state = check_random_state(0)
    # 创建一个随机生成的 100x10 的矩阵
    X = random_state.randn(100, 10)
    # 遍历两种方法进行 t-SNE 运算
    for method in ["barnes_hut", "exact"]:
        # 创建一个 t-SNE 实例，设置 n_iter_without_progress 为 -1，学习率为 1e8，随机种子为 0，方法为当前循环的 method
        tsne = TSNE(
            n_iter_without_progress=-1,
            verbose=2,
            learning_rate=1e8,
            random_state=0,
            method=method,
            max_iter=351,
            init="random",
        )
        # 设置 t-SNE 实例的 _N_ITER_CHECK 和 _EXPLORATION_MAX_ITER 属性
        tsne._N_ITER_CHECK = 1
        tsne._EXPLORATION_MAX_ITER = 0

        # 保存原来的标准输出流
        old_stdout = sys.stdout
        # 用 StringIO 创建一个临时的内存文件对象来捕获标准输出
        sys.stdout = StringIO()
        try:
            # 执行 t-SNE 的拟合转换操作
            tsne.fit_transform(X)
        finally:
            # 获取标准输出的内容
            out = sys.stdout.getvalue()
            # 关闭临时的标准输出流
            sys.stdout.close()
            # 恢复原来的标准输出流
            sys.stdout = old_stdout

        # 断言：输出中应该包含关于 n_iter_without_progress 的信息
        assert "did not make any progress during the last -1 episodes. Finished." in out


# 测试函数，确保 min_grad_norm 参数的正确使用
def test_min_grad_norm():
    # 使用固定的随机种子初始化随机状态
    random_state = check_random_state(0)
    # 创建一个随机生成的 100x2 的矩阵
    X = random_state.randn(100, 2)
    # 设置 min_grad_norm 参数
    min_grad_norm = 0.002
    # 创建一个 t-SNE 实例，设置 min_grad_norm 参数，verbose 为 2，随机种子为 0，方法为 "exact"
    tsne = TSNE(min_grad_norm=min_grad_norm, verbose=2, random_state=0, method="exact")

    # 保存原来的标准输出流
    old_stdout = sys.stdout
    # 用 StringIO 创建一个临时的内存文件对象来捕获标准输出
    sys.stdout = StringIO()
    try:
        # 执行 t-SNE 的拟合转换操作
        tsne.fit_transform(X)
    finally:
        # 获取标准输出的内容
        out = sys.stdout.getvalue()
        # 关闭临时的标准输出流
        sys.stdout.close()
        # 恢复原来的标准输出流
        sys.stdout = old_stdout

    # 将输出内容按行分割成列表
    lines_out = out.split("\n")

    # 提取 verbose 输出中的梯度范数值
    gradient_norm_values = []
    for line in lines_out:
        # 当遇到 "Finished" 字样时，停止提取
        if "Finished" in line:
            break

        # 查找行中的 "gradient norm" 开始位置
        start_grad_norm = line.find("gradient norm")
        if start_grad_norm >= 0:
            # 截取包含梯度范数信息的子串，并提取出梯度范数值
            line = line[start_grad_norm:]
            line = line.replace("gradient norm = ", "").split(" ")[0]
            gradient_norm_values.append(float(line))

    # 将梯度范数值转换为 NumPy 数组
    gradient_norm_values = np.array(gradient_norm_values)
    # 计算小于等于 min_grad_norm 的梯度范数值的数量
    n_smaller_gradient_norms = len(
        gradient_norm_values[gradient_norm_values <= min_grad_norm]
    )

    # 断言：小于等于 min_grad_norm 的梯度范数值最多只能出现一次，
    # 因为一旦梯度范数小于 min_grad_norm，优化过程将停止
    assert n_smaller_gradient_norms <= 1


# 测试函数，确保 accessible kl_divergence 的正确性
def test_accessible_kl_divergence():
    # 使用固定的随机种子初始化随机状态
    random_state = check_random_state(0)
    # 创建一个随机生成的 50x2 的矩阵
    X = random_state.randn(50, 2)
    # 创建一个 t-SNE 实例，设置 n_iter_without_progress 为 2，verbose 为 2，随机种子为 0，方法为 "exact"，最大迭代次数为 500
    tsne = TSNE(
        n_iter_without_progress=2,
        verbose=2,
        random_state=0,
        method="exact",
        max_iter=500,
    )

    # 保存原来的标准输出流
    old_stdout = sys.stdout
    # 用 StringIO 创建一个临时的内存文件对象来捕获标准输出
    sys.stdout = StringIO()
    try:
        # 执行 t-SNE 的拟合转换操作
        tsne.fit_transform(X)
    finally:
        # 获取标准输出的内容
        out = sys.stdout.getvalue()
        # 关闭临时的标准输出流
        sys.stdout.close()
        # 恢复原来的标准输出流
        sys.stdout = old_stdout
    # 从输出字符串中按行反向查找，直至找到包含"Iteration"的行
    for line in out.split("\n")[::-1]:
        # 如果当前行包含"Iteration"关键字
        if "Iteration" in line:
            # 使用"error = "作为分隔符，分割当前行，获取错误值部分
            _, _, error = line.partition("error = ")
            # 如果成功获取到错误值
            if error:
                # 再次分割错误值，使用","作为分隔符，获取最终的错误数值部分
                error, _, _ = error.partition(",")
                # 跳出循环，已经找到并提取了错误值
                break
    # 使用准确度断言函数，检查 t-SNE 模型的 kl_divergence 是否与提取的错误值几乎相等
    assert_almost_equal(tsne.kl_divergence_, float(error), decimal=5)
@pytest.mark.parametrize("method", ["barnes_hut", "exact"])
def test_uniform_grid(method):
    """测试确保 t-SNE 能够大致恢复一个均匀的二维网格
    
    由于 X_2d_grid 中点之间距离的并列性，对于 ``method='barnes_hut'``，
    由于数值精度问题，此测试在不同平台上的结果可能会有所不同。
    
    另外，由于 t-SNE 存在收敛到错误解的可能性（由于坏的初始化导致收敛到坏的局部最小值，
    优化问题是非凸的），为了避免测试经常失败，当收敛不够好时，我们会从最终点重新运行 t-SNE。
    """
    seeds = range(3)
    max_iter = 500
    for seed in seeds:
        tsne = TSNE(
            n_components=2,
            init="random",
            random_state=seed,
            perplexity=50,
            max_iter=max_iter,
            method=method,
            learning_rate="auto",
        )
        Y = tsne.fit_transform(X_2d_grid)

        try_name = "{}_{}".format(method, seed)
        try:
            assert_uniform_grid(Y, try_name)
        except AssertionError:
            # 如果测试第一次失败，使用 init=Y 重新运行以查看是否是由于坏的初始化导致的。
            # 注意，这也会运行一个 early_exaggeration 步骤。
            try_name += ":rerun"
            tsne.init = Y
            Y = tsne.fit_transform(X_2d_grid)
            assert_uniform_grid(Y, try_name)


def assert_uniform_grid(Y, try_name=None):
    # 确保结果嵌入导致大致均匀分布的点：最近邻的距离应该大于零且大致恒定。
    nn = NearestNeighbors(n_neighbors=1).fit(Y)
    dist_to_nn = nn.kneighbors(return_distance=True)[0].ravel()
    assert dist_to_nn.min() > 0.1

    smallest_to_mean = dist_to_nn.min() / np.mean(dist_to_nn)
    largest_to_mean = dist_to_nn.max() / np.mean(dist_to_nn)

    assert smallest_to_mean > 0.5, try_name
    assert largest_to_mean < 2, try_name


def test_bh_match_exact():
    # 检查 ``barnes_hut`` 方法在 ``angle = 0`` 且 ``perplexity > n_samples / 3`` 时
    # 是否与精确方法匹配。
    random_state = check_random_state(0)
    n_features = 10
    X = random_state.randn(30, n_features).astype(np.float32)
    X_embeddeds = {}
    max_iter = {}
    for method in ["exact", "barnes_hut"]:
        tsne = TSNE(
            n_components=2,
            method=method,
            learning_rate=1.0,
            init="random",
            random_state=0,
            max_iter=251,
            perplexity=29.5,
            angle=0,
        )
        # 关闭 early_exaggeration
        tsne._EXPLORATION_MAX_ITER = 0
        X_embeddeds[method] = tsne.fit_transform(X)
        max_iter[method] = tsne.n_iter_

    assert max_iter["exact"] == max_iter["barnes_hut"]
    assert_allclose(X_embeddeds["exact"], X_embeddeds["barnes_hut"], rtol=1e-4)
# 测试多线程与顺序执行下的 Barnes-Hut 梯度计算的一致性
def test_gradient_bh_multithread_match_sequential():
    # 设置特征数、样本数、降维后的组件数、自由度
    n_features = 10
    n_samples = 30
    n_components = 2
    degrees_of_freedom = 1

    # 设置角度和 perplexity
    angle = 3
    perplexity = 5

    # 使用随机状态生成指定维度的浮点型数据
    random_state = check_random_state(0)
    data = random_state.randn(n_samples, n_features).astype(np.float32)
    # 生成随机状态下指定维度的参数数组
    params = random_state.randn(n_samples, n_components)

    # 确定最近邻的数量
    n_neighbors = n_samples - 1
    # 计算数据的最近邻距离的稀疏矩阵
    distances_csr = (
        NearestNeighbors()
        .fit(data)
        .kneighbors_graph(n_neighbors=n_neighbors, mode="distance")
    )
    # 根据最近邻距离稀疏矩阵计算联合概率分布 P_bh
    P_bh = _joint_probabilities_nn(distances_csr, perplexity, verbose=0)
    # 计算顺序执行下的 KL 散度和梯度
    kl_sequential, grad_sequential = _kl_divergence_bh(
        params,
        P_bh,
        degrees_of_freedom,
        n_samples,
        n_components,
        angle=angle,
        skip_num_points=0,
        verbose=0,
        num_threads=1,
    )

    # 遍历多线程数 [2, 4]，计算多线程下的 KL 散度和梯度
    for num_threads in [2, 4]:
        kl_multithread, grad_multithread = _kl_divergence_bh(
            params,
            P_bh,
            degrees_of_freedom,
            n_samples,
            n_components,
            angle=angle,
            skip_num_points=0,
            verbose=0,
            num_threads=num_threads,
        )

        # 断言多线程计算的 KL 散度和梯度与顺序执行的结果接近
        assert_allclose(kl_multithread, kl_sequential, rtol=1e-6)
        assert_allclose(grad_multithread, grad_sequential)


@pytest.mark.parametrize(
    "metric, dist_func",
    [("manhattan", manhattan_distances), ("cosine", cosine_distances)],
)
@pytest.mark.parametrize("method", ["barnes_hut", "exact"])
def test_tsne_with_different_distance_metrics(metric, dist_func, method):
    """Make sure that TSNE works for different distance metrics"""

    if method == "barnes_hut" and metric == "manhattan":
        # 使用 Manhattan 距离计算距离会导致 T-SNE 收敛到不同的解，
        # 这在定性结果上应该不会有影响，但从数学角度来看可能无效。
        # TODO: 如果 `manhattan_distances` 被重构以重用与 NearestNeighbors 相同的 Cython 代码，
        # 可以重新启用这个测试。
        # 参考：
        # https://github.com/scikit-learn/scikit-learn/pull/23865/files#r925721573
        pytest.xfail(
            "Distance computations are different for method == 'barnes_hut' and metric"
            " == 'manhattan', but this is expected."
        )

    # 使用随机状态生成原始组件数和嵌入组件数的数据
    random_state = check_random_state(0)
    n_components_original = 3
    n_components_embedding = 2
    X = random_state.randn(50, n_components_original).astype(np.float32)
    # 使用 t-SNE 算法将输入数据 X 转换为降维后的表示，使用给定的距离度量(metric)、方法(method)、降维后的维度数(n_components_embedding)、随机种子(random_state)、最大迭代次数(max_iter)、初始化方式(init)和学习率(learning_rate)
    X_transformed_tsne = TSNE(
        metric=metric,
        method=method,
        n_components=n_components_embedding,
        random_state=0,
        max_iter=300,
        init="random",
        learning_rate="auto",
    ).fit_transform(X)
    
    # 使用预计算的距离矩阵作为距离度量(metric)，对输入数据 X 进行 t-SNE 转换，使用给定的方法(method)、降维后的维度数(n_components_embedding)、随机种子(random_state)、最大迭代次数(max_iter)、初始化方式(init)和学习率(learning_rate)
    X_transformed_tsne_precomputed = TSNE(
        metric="precomputed",
        method=method,
        n_components=n_components_embedding,
        random_state=0,
        max_iter=300,
        init="random",
        learning_rate="auto",
    ).fit_transform(dist_func(X))
    
    # 断言两个 t-SNE 转换的结果必须完全一致，用于验证 t-SNE 算法的正确性
    assert_array_equal(X_transformed_tsne, X_transformed_tsne_precomputed)
@pytest.mark.parametrize("method", ["exact", "barnes_hut"])
def test_tsne_n_jobs(method):
    """Make sure that the n_jobs parameter doesn't impact the output"""
    # 设置随机种子为0，确保结果可重复
    random_state = check_random_state(0)
    # 创建一个 30x10 的随机数据集
    n_features = 10
    X = random_state.randn(30, n_features)
    # 使用 n_jobs=1 计算 t-SNE 转换结果的参考值
    X_tr_ref = TSNE(
        n_components=2,
        method=method,
        perplexity=25.0,
        angle=0,
        n_jobs=1,
        random_state=0,
        init="random",
        learning_rate="auto",
    ).fit_transform(X)
    # 使用 n_jobs=2 计算 t-SNE 转换结果
    X_tr = TSNE(
        n_components=2,
        method=method,
        perplexity=25.0,
        angle=0,
        n_jobs=2,
        random_state=0,
        init="random",
        learning_rate="auto",
    ).fit_transform(X)

    # 断言两次计算结果的接近程度
    assert_allclose(X_tr_ref, X_tr)


def test_tsne_with_mahalanobis_distance():
    """Make sure that method_parameters works with mahalanobis distance."""
    # 设置随机种子为0，确保结果可重复
    random_state = check_random_state(0)
    n_samples, n_features = 300, 10
    # 创建一个 300x10 的随机数据集
    X = random_state.randn(n_samples, n_features)
    # 设置 t-SNE 的默认参数
    default_params = {
        "perplexity": 40,
        "max_iter": 250,
        "learning_rate": "auto",
        "init": "random",
        "n_components": 3,
        "random_state": 0,
    }

    # 使用 Mahalanobis 距离时，应该抛出 ValueError 异常
    tsne = TSNE(metric="mahalanobis", **default_params)
    msg = "Must provide either V or VI for Mahalanobis distance"
    with pytest.raises(ValueError, match=msg):
        tsne.fit_transform(X)

    # 使用预先计算的 Mahalanobis 距离矩阵进行 t-SNE 转换
    precomputed_X = squareform(pdist(X, metric="mahalanobis"), checks=True)
    X_trans_expected = TSNE(metric="precomputed", **default_params).fit_transform(
        precomputed_X
    )

    # 使用指定协方差矩阵 V 的 Mahalanobis 距离进行 t-SNE 转换
    X_trans = TSNE(
        metric="mahalanobis", metric_params={"V": np.cov(X.T)}, **default_params
    ).fit_transform(X)
    # 断言两次计算结果的接近程度
    assert_allclose(X_trans, X_trans_expected)


@pytest.mark.parametrize("perplexity", (20, 30))
def test_tsne_perplexity_validation(perplexity):
    """Make sure that perplexity > n_samples results in a ValueError"""

    # 设置随机种子为0，确保结果可重复
    random_state = check_random_state(0)
    # 创建一个 20x2 的随机数据集
    X = random_state.randn(20, 2)
    # 使用不合理的 perplexity 值时，应该抛出 ValueError 异常
    est = TSNE(
        learning_rate="auto",
        init="pca",
        perplexity=perplexity,
        random_state=random_state,
    )
    msg = "perplexity must be less than n_samples"
    with pytest.raises(ValueError, match=msg):
        est.fit_transform(X)


def test_tsne_works_with_pandas_output():
    """Make sure that TSNE works when the output is set to "pandas".

    Non-regression test for gh-25365.
    """
    # 确保 pandas 库存在，否则跳过测试
    pytest.importorskip("pandas")
    # 将输出转换为 pandas 格式后进行 t-SNE 计算
    with config_context(transform_output="pandas"):
        arr = np.arange(35 * 4).reshape(35, 4)
        TSNE(n_components=2).fit_transform(arr)


# TODO(1.7): remove
def test_tnse_n_iter_deprecated():
    """Check `n_iter` parameter deprecated."""
    # 设置随机种子为0，确保结果可重复
    random_state = check_random_state(0)
    # 创建一个 40x100 的随机数据集
    X = random_state.randn(40, 100)
    # 使用已弃用的 n_iter 参数应该发出 FutureWarning 警告
    tsne = TSNE(n_iter=250)
    msg = "'n_iter' was renamed to 'max_iter'"
    with pytest.warns(FutureWarning, match=msg):
        tsne.fit_transform(X)


# TODO(1.7): remove
# 定义测试函数，用于检查当 `n_iter` 和 `max_iter` 都被设置时是否会引发错误
def test_tnse_n_iter_max_iter_both_set():
    # 设置随机数种子为0，以便结果可重复
    random_state = check_random_state(0)
    # 生成一个 40x100 的随机数据矩阵 X
    X = random_state.randn(40, 100)
    # 创建一个 TSNE 对象，设置 `n_iter` 为 250，`max_iter` 为 500
    tsne = TSNE(n_iter=250, max_iter=500)
    # 设置错误消息字符串，用于匹配预期的异常信息
    msg = "Both 'n_iter' and 'max_iter' attributes were set"
    # 使用 pytest 模块的 `raises` 函数来检查是否会抛出 ValueError 异常，并验证异常消息是否匹配
    with pytest.raises(ValueError, match=msg):
        # 调用 TSNE 对象的 `fit_transform` 方法，预期会引发 ValueError 异常
        tsne.fit_transform(X)
```