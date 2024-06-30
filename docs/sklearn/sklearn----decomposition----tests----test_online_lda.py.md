# `D:\src\scipysrc\scikit-learn\sklearn\decomposition\tests\test_online_lda.py`

```
import sys
from io import StringIO

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.linalg import block_diag
from scipy.special import psi

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition._online_lda_fast import (
    _dirichlet_expectation_1d,
    _dirichlet_expectation_2d,
)
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    if_safe_multiprocessing_with_blas,
)
from sklearn.utils.fixes import CSR_CONTAINERS

# 定义一个函数，用于构建稀疏数组，接受一个 CSR 容器作为输入
def _build_sparse_array(csr_container):
    # 创建 3 个主题，每个主题有 3 个不同的单词
    # （每个单词只属于一个主题）
    n_components = 3
    block = np.full((3, 3), n_components, dtype=int)
    blocks = [block] * n_components
    X = block_diag(*blocks)
    X = csr_container(X)  # 将构建好的稀疏块转换为 CSR 格式
    return (n_components, X)

# 使用 pytest 标记定义的测试函数，参数化使用 CSR_CONTAINERS 中的每个容器
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_lda_default_prior_params(csr_container):
    # 测试 LDA 默认先验参数
    # 先验参数应为 `1 / 主题数`，并且 verbose 参数不应影响结果
    n_components, X = _build_sparse_array(csr_container)
    prior = 1.0 / n_components
    lda_1 = LatentDirichletAllocation(
        n_components=n_components,
        doc_topic_prior=prior,
        topic_word_prior=prior,
        random_state=0,
    )
    lda_2 = LatentDirichletAllocation(n_components=n_components, random_state=0)
    topic_distr_1 = lda_1.fit_transform(X)
    topic_distr_2 = lda_2.fit_transform(X)
    assert_almost_equal(topic_distr_1, topic_distr_2)

# 使用 pytest 标记定义的测试函数，参数化使用 CSR_CONTAINERS 中的每个容器
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_lda_fit_batch(csr_container):
    # 测试 LDA 批量学习 (`fit` 方法使用 'batch' 学习)
    rng = np.random.RandomState(0)
    n_components, X = _build_sparse_array(csr_container)
    lda = LatentDirichletAllocation(
        n_components=n_components,
        evaluate_every=1,
        learning_method="batch",
        random_state=rng,
    )
    lda.fit(X)

    correct_idx_grps = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    for component in lda.components_:
        # 找到每个 LDA 成分中的前 3 个单词
        top_idx = set(component.argsort()[-3:][::-1])
        assert tuple(sorted(top_idx)) in correct_idx_grps

# 使用 pytest 标记定义的测试函数，参数化使用 CSR_CONTAINERS 中的每个容器
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_lda_fit_online(csr_container):
    # 测试 LDA 在线学习 (`fit` 方法使用 'online' 学习)
    rng = np.random.RandomState(0)
    n_components, X = _build_sparse_array(csr_container)
    lda = LatentDirichletAllocation(
        n_components=n_components,
        learning_offset=10.0,
        evaluate_every=1,
        learning_method="online",
        random_state=rng,
    )
    lda.fit(X)

    correct_idx_grps = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    # 遍历 LDA 模型中的每个主题成分（即每个主题的词分布）
    for component in lda.components_:
        # 找到每个 LDA 主题成分中排名前三的词的索引
        top_idx = set(component.argsort()[-3:][::-1])
        # 断言：确保这些索引按顺序排列的元组存在于正确的索引组中
        assert tuple(sorted(top_idx)) in correct_idx_grps
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_lda_partial_fit(csr_container):
    # 使用参数化测试来对不同的 CSR 容器进行测试
    # 测试 LDA 的在线学习 (`partial_fit` 方法)，与 `test_lda_batch` 相同
    rng = np.random.RandomState(0)
    # 构建稀疏数组，并返回其主题数和稀疏矩阵 X
    n_components, X = _build_sparse_array(csr_container)
    # 初始化 LDA 模型，设置参数：主题数、学习偏移量、总样本数和随机状态
    lda = LatentDirichletAllocation(
        n_components=n_components,
        learning_offset=10.0,
        total_samples=100,
        random_state=rng,
    )
    # 对稀疏矩阵 X 进行三次部分拟合
    for i in range(3):
        lda.partial_fit(X)

    # 预期的正确主题索引组合
    correct_idx_grps = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    # 验证每个 LDA 组件中的前三个单词索引是否符合预期组合
    for c in lda.components_:
        top_idx = set(c.argsort()[-3:][::-1])
        assert tuple(sorted(top_idx)) in correct_idx_grps


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_lda_dense_input(csr_container):
    # 使用参数化测试来对不同的 CSR 容器进行测试
    # 测试使用密集输入的 LDA
    rng = np.random.RandomState(0)
    # 构建稀疏数组，并返回其主题数和稀疏矩阵 X
    n_components, X = _build_sparse_array(csr_container)
    # 初始化 LDA 模型，设置参数：主题数、学习方法为批量法和随机状态
    lda = LatentDirichletAllocation(
        n_components=n_components, learning_method="batch", random_state=rng
    )
    # 对 X 的密集表示进行拟合
    lda.fit(X.toarray())

    # 预期的正确主题索引组合
    correct_idx_grps = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    # 验证每个 LDA 组件中的前三个单词索引是否符合预期组合
    for component in lda.components_:
        # 找到每个 LDA 组件中前三个单词的索引
        top_idx = set(component.argsort()[-3:][::-1])
        assert tuple(sorted(top_idx)) in correct_idx_grps


def test_lda_transform():
    # 测试 LDA 的变换功能
    # 变换结果不能为负且应该被归一化
    rng = np.random.RandomState(0)
    # 创建一个随机整数矩阵 X
    X = rng.randint(5, size=(20, 10))
    n_components = 3
    # 初始化 LDA 模型，设置主题数和随机状态
    lda = LatentDirichletAllocation(n_components=n_components, random_state=rng)
    # 对 X 进行拟合并进行变换
    X_trans = lda.fit_transform(X)
    # 断言变换后的结果中是否存在大于零的值
    assert (X_trans > 0.0).any()
    # 断言每一行变换后的结果是否被归一化为接近1
    assert_array_almost_equal(np.sum(X_trans, axis=1), np.ones(X_trans.shape[0]))


@pytest.mark.parametrize("method", ("online", "batch"))
def test_lda_fit_transform(method):
    # 使用参数化测试来测试 LDA 的 fit_transform 和 transform 方法
    # fit_transform 和 transform 的结果应该一致
    rng = np.random.RandomState(0)
    # 创建一个随机整数矩阵 X
    X = rng.randint(10, size=(50, 20))
    # 初始化 LDA 模型，设置主题数、学习方法和随机状态
    lda = LatentDirichletAllocation(
        n_components=5, learning_method=method, random_state=rng
    )
    # 对 X 进行拟合并进行 fit_transform
    X_fit = lda.fit_transform(X)
    # 对 X 进行 transform
    X_trans = lda.transform(X)
    # 断言 fit_transform 和 transform 的结果是否近似相等，精度为四位小数
    assert_array_almost_equal(X_fit, X_trans, 4)


def test_lda_negative_input():
    # 测试传入具有稀疏负输入的稠密矩阵
    X = np.full((5, 10), -1.0)
    # 初始化 LDA 模型
    lda = LatentDirichletAllocation()
    # 使用 pytest 断言来捕获 ValueError 异常，并匹配特定正则表达式
    regex = r"^Negative values in data passed"
    with pytest.raises(ValueError, match=regex):
        lda.fit(X)


def test_lda_no_component_error():
    # 测试在调用 `perplexity` 方法之前未调用 `fit` 方法的情况
    rng = np.random.RandomState(0)
    # 创建一个随机整数矩阵 X
    X = rng.randint(4, size=(20, 10))
    # 初始化 LDA 模型
    lda = LatentDirichletAllocation()
    # 定义匹配的错误消息
    regex = (
        "This LatentDirichletAllocation instance is not fitted yet. "
        "Call 'fit' with appropriate arguments before using this "
        "estimator."
    )
    # 使用 pytest 断言来捕获 NotFittedError 异常，并匹配特定的错误消息
    with pytest.raises(NotFittedError, match=regex):
        lda.perplexity(X)


@if_safe_multiprocessing_with_blas
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
@pytest.mark.parametrize("method", ("online", "batch"))
def test_lda_multi_jobs(method, csr_container):
    # 从 CSR_CONTAINERS 参数化测试数据中获取稀疏数组 X 和主题数 n_components
    n_components, X = _build_sparse_array(csr_container)
    # 使用随机数种子 0 创建随机数生成器 rng
    rng = np.random.RandomState(0)
    # 创建 LatentDirichletAllocation 对象 lda，设置主题数、使用的 CPU 数量、学习方法、评估频率和随机数种子
    lda = LatentDirichletAllocation(
        n_components=n_components,
        n_jobs=2,
        learning_method=method,
        evaluate_every=1,
        random_state=rng,
    )
    # 对稀疏数组 X 进行模型拟合
    lda.fit(X)

    # 预期的正确主题索引组合
    correct_idx_grps = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    # 验证每个主题的前三个最高索引是否在预期的索引组合中
    for c in lda.components_:
        top_idx = set(c.argsort()[-3:][::-1])
        assert tuple(sorted(top_idx)) in correct_idx_grps


@if_safe_multiprocessing_with_blas
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_lda_partial_fit_multi_jobs(csr_container):
    # 从 CSR_CONTAINERS 参数化测试数据中获取稀疏数组 X 和主题数 n_components
    rng = np.random.RandomState(0)
    n_components, X = _build_sparse_array(csr_container)
    # 创建 LatentDirichletAllocation 对象 lda，设置主题数、使用的 CPU 数量、学习偏移量、总样本数和随机数种子
    lda = LatentDirichletAllocation(
        n_components=n_components,
        n_jobs=2,
        learning_offset=5.0,
        total_samples=30,
        random_state=rng,
    )
    # 使用 online 方法对稀疏数组 X 进行多次部分拟合
    for i in range(2):
        lda.partial_fit(X)

    # 预期的正确主题索引组合
    correct_idx_grps = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    # 验证每个主题的前三个最高索引是否在预期的索引组合中
    for c in lda.components_:
        top_idx = set(c.argsort()[-3:][::-1])
        assert tuple(sorted(top_idx)) in correct_idx_grps


def test_lda_preplexity_mismatch():
    # 测试 `perplexity` 方法中的维度不匹配情况
    rng = np.random.RandomState(0)
    # 随机生成主题数和样本数
    n_components = rng.randint(3, 6)
    n_samples = rng.randint(6, 10)
    # 创建随机稀疏矩阵 X
    X = np.random.randint(4, size=(n_samples, 10))
    # 创建 LatentDirichletAllocation 对象 lda，设置主题数、学习偏移量、总样本数和随机数种子
    lda = LatentDirichletAllocation(
        n_components=n_components,
        learning_offset=5.0,
        total_samples=20,
        random_state=rng,
    )
    # 对稀疏矩阵 X 进行模型拟合
    lda.fit(X)
    # 创建维度不匹配的样本数据 invalid_n_samples
    invalid_n_samples = rng.randint(4, size=(n_samples + 1, n_components))
    # 断言异常，确保在 `perplexity` 计算中捕获到预期的 ValueError
    with pytest.raises(ValueError, match=r"Number of samples"):
        lda._perplexity_precomp_distr(X, invalid_n_samples)
    # 创建维度不匹配的主题数数据 invalid_n_components
    invalid_n_components = rng.randint(4, size=(n_samples, n_components + 1))
    # 断言异常，确保在 `perplexity` 计算中捕获到预期的 ValueError
    with pytest.raises(ValueError, match=r"Number of topics"):
        lda._perplexity_precomp_distr(X, invalid_n_components)


@pytest.mark.parametrize("method", ("online", "batch"))
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_lda_perplexity(method, csr_container):
    # 测试批处理训练的 LDA perplexity
    # 每次迭代后 perplexity 应该更低
    # 从 CSR_CONTAINERS 参数化测试数据中获取稀疏数组 X 和主题数 n_components
    n_components, X = _build_sparse_array(csr_container)
    # 创建 LatentDirichletAllocation 对象 lda_1 和 lda_2，分别设置主题数、最大迭代次数、学习方法、总样本数和随机数种子
    lda_1 = LatentDirichletAllocation(
        n_components=n_components,
        max_iter=1,
        learning_method=method,
        total_samples=100,
        random_state=0,
    )
    lda_2 = LatentDirichletAllocation(
        n_components=n_components,
        max_iter=10,
        learning_method=method,
        total_samples=100,
        random_state=0,
    )
    # 对稀疏数组 X 进行模型拟合（只进行一次迭代）
    lda_1.fit(X)
    # 计算 LDA 模型 lda_1 在数据集 X 上的困惑度，不使用子采样
    perp_1 = lda_1.perplexity(X, sub_sampling=False)

    # 在数据集 X 上拟合 LDA 模型 lda_2
    lda_2.fit(X)
    # 计算 LDA 模型 lda_2 在数据集 X 上的困惑度，不使用子采样
    perp_2 = lda_2.perplexity(X, sub_sampling=False)
    # 断言：确保未使用子采样时，lda_1 的困惑度大于等于 lda_2 的困惑度
    assert perp_1 >= perp_2

    # 计算 LDA 模型 lda_1 在数据集 X 上的困惑度，使用子采样
    perp_1_subsampling = lda_1.perplexity(X, sub_sampling=True)
    # 计算 LDA 模型 lda_2 在数据集 X 上的困惑度，使用子采样
    perp_2_subsampling = lda_2.perplexity(X, sub_sampling=True)
    # 断言：确保使用子采样时，lda_1 的困惑度大于等于 lda_2 的困惑度
    assert perp_1_subsampling >= perp_2_subsampling
@pytest.mark.parametrize("method", ("online", "batch"))
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_lda_score(method, csr_container):
    # 测试批量训练下的 LDA 分数
    # 每次迭代后分数应该更高
    # 构建稀疏数组
    n_components, X = _build_sparse_array(csr_container)
    # 创建 LDA 模型，最大迭代次数为1，指定学习方法和随机种子
    lda_1 = LatentDirichletAllocation(
        n_components=n_components,
        max_iter=1,
        learning_method=method,
        total_samples=100,
        random_state=0,
    )
    # 创建另一个 LDA 模型，最大迭代次数为10，指定学习方法和随机种子
    lda_2 = LatentDirichletAllocation(
        n_components=n_components,
        max_iter=10,
        learning_method=method,
        total_samples=100,
        random_state=0,
    )
    # 对第一个模型进行拟合和转换
    lda_1.fit_transform(X)
    # 计算第一个模型的分数
    score_1 = lda_1.score(X)

    # 对第二个模型进行拟合和转换
    lda_2.fit_transform(X)
    # 计算第二个模型的分数
    score_2 = lda_2.score(X)
    # 断言第二个模型的分数应大于等于第一个模型的分数
    assert score_2 >= score_1


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_perplexity_input_format(csr_container):
    # 测试稀疏和稠密输入的 LDA 困惑度
    # 稠密和稀疏输入的困惑度应该相同
    # 构建稀疏数组
    n_components, X = _build_sparse_array(csr_container)
    # 创建 LDA 模型，最大迭代次数为1，使用批量学习方法和随机种子
    lda = LatentDirichletAllocation(
        n_components=n_components,
        max_iter=1,
        learning_method="batch",
        total_samples=100,
        random_state=0,
    )
    # 对模型进行拟合
    lda.fit(X)
    # 计算第一个输入的困惑度
    perp_1 = lda.perplexity(X)
    # 计算转换为稠密数组后的困惑度
    perp_2 = lda.perplexity(X.toarray())
    # 断言两种输入的困惑度应该近似相等
    assert_almost_equal(perp_1, perp_2)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_lda_score_perplexity(csr_container):
    # 测试 LDA 分数和困惑度之间的关系
    # 构建稀疏数组
    n_components, X = _build_sparse_array(csr_container)
    # 创建 LDA 模型，最大迭代次数为10，随机种子为0
    lda = LatentDirichletAllocation(
        n_components=n_components, max_iter=10, random_state=0
    )
    # 对模型进行拟合
    lda.fit(X)
    # 计算不进行子采样的困惑度
    perplexity_1 = lda.perplexity(X, sub_sampling=False)

    # 计算模型分数
    score = lda.score(X)
    # 计算根据分数计算的困惑度
    perplexity_2 = np.exp(-1.0 * (score / np.sum(X.data)))
    # 断言两种困惑度应该近似相等
    assert_almost_equal(perplexity_1, perplexity_2)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_lda_fit_perplexity(csr_container):
    # 测试拟合过程中计算的困惑度是否与 perplexity 方法返回的一致
    # 构建稀疏数组
    n_components, X = _build_sparse_array(csr_container)
    # 创建 LDA 模型，最大迭代次数为1，使用批量学习方法和随机种子
    lda = LatentDirichletAllocation(
        n_components=n_components,
        max_iter=1,
        learning_method="batch",
        random_state=0,
        evaluate_every=1,
    )
    # 对模型进行拟合
    lda.fit(X)

    # 拟合方法结束时计算的困惑度
    perplexity1 = lda.bound_

    # 计算在训练集上的困惑度
    perplexity2 = lda.perplexity(X)

    # 断言两者应该近似相等
    assert_almost_equal(perplexity1, perplexity2)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_lda_empty_docs(csr_container):
    """Test LDA on empty document (all-zero rows)."""
    # 测试 LDA 在空文档（全零行）上的表现
    Z = np.zeros((5, 4))
    # 对于列表中的每个对象 X，分别执行以下操作：
    for X in [Z, csr_container(Z)]:
        # 使用 LatentDirichletAllocation 模型拟合数据 X，最大迭代次数为 750
        lda = LatentDirichletAllocation(max_iter=750).fit(X)
        # 断言 lda 模型的每列和为 1，这里使用 assert_almost_equal 进行检查
        assert_almost_equal(
            lda.components_.sum(axis=0), np.ones(lda.components_.shape[1])
        )
# 定义函数，用于测试 Cython 版本的狄利克雷期望计算
def test_dirichlet_expectation():
    # 创建一个从 -100 到 10 的对数空间数组，包含 10000 个元素
    x = np.logspace(-100, 10, 10000)
    # 创建一个与 x 具有相同形状的空数组
    expectation = np.empty_like(x)
    # 调用 Cython 函数 _dirichlet_expectation_1d 计算狄利克雷期望
    _dirichlet_expectation_1d(x, 0, expectation)
    # 断言计算得到的期望与 np.exp(psi(x) - psi(np.sum(x))) 的值在给定的公差范围内接近
    assert_allclose(expectation, np.exp(psi(x) - psi(np.sum(x))), atol=1e-19)

    # 将 x 重新形状为 100x100 的数组
    x = x.reshape(100, 100)
    # 断言调用 Cython 函数 _dirichlet_expectation_2d 返回的结果与 psi(x) - psi(np.sum(x, axis=1)[:, np.newaxis]) 的值在给定的相对和绝对公差范围内接近
    assert_allclose(
        _dirichlet_expectation_2d(x),
        psi(x) - psi(np.sum(x, axis=1)[:, np.newaxis]),
        rtol=1e-11,
        atol=3e-9,
    )


# 定义函数，检查 LatentDirichletAllocation 对象的详细输出
def check_verbosity(
    verbose, evaluate_every, expected_lines, expected_perplexities, csr_container
):
    # 使用 _build_sparse_array 函数构建稀疏数组
    n_components, X = _build_sparse_array(csr_container)
    # 创建 LatentDirichletAllocation 对象 lda
    lda = LatentDirichletAllocation(
        n_components=n_components,
        max_iter=3,
        learning_method="batch",
        verbose=verbose,
        evaluate_every=evaluate_every,
        random_state=0,
    )
    # 重定向标准输出至 StringIO 对象 out
    out = StringIO()
    old_out, sys.stdout = sys.stdout, out
    try:
        # 调用 lda.fit(X) 进行模型拟合
        lda.fit(X)
    finally:
        # 恢复标准输出
        sys.stdout = old_out

    # 统计输出中的行数和包含 "perplexity" 字符串的次数
    n_lines = out.getvalue().count("\n")
    n_perplexity = out.getvalue().count("perplexity")
    # 断言统计结果与期望值相等
    assert expected_lines == n_lines
    assert expected_perplexities == n_perplexity


# 使用参数化测试框架，测试不同的 verbose 和 evaluate_every 组合
@pytest.mark.parametrize(
    "verbose,evaluate_every,expected_lines,expected_perplexities",
    [
        (False, 1, 0, 0),
        (False, 0, 0, 0),
        (True, 0, 3, 0),
        (True, 1, 3, 3),
        (True, 2, 3, 1),
    ],
)
# 参数化测试函数，测试不同的 csr_container 参数
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_verbosity(
    verbose, evaluate_every, expected_lines, expected_perplexities, csr_container
):
    # 调用 check_verbosity 函数进行测试
    check_verbosity(
        verbose, evaluate_every, expected_lines, expected_perplexities, csr_container
    )


# 使用参数化测试框架，测试 LatentDirichletAllocation 对象的特征名称输出
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_lda_feature_names_out(csr_container):
    """Check feature names out for LatentDirichletAllocation."""
    # 使用 _build_sparse_array 函数构建稀疏数组
    n_components, X = _build_sparse_array(csr_container)
    # 创建并拟合 LatentDirichletAllocation 对象 lda
    lda = LatentDirichletAllocation(n_components=n_components).fit(X)

    # 获取 lda 对象的特征名称输出
    names = lda.get_feature_names_out()
    # 断言特征名称与预期的格式匹配
    assert_array_equal(
        [f"latentdirichletallocation{i}" for i in range(n_components)], names
    )


# 使用参数化测试框架，测试 LatentDirichletAllocation 对象的数据类型匹配性
@pytest.mark.parametrize("learning_method", ("batch", "online"))
def test_lda_dtype_match(learning_method, global_dtype):
    """Check data type preservation of fitted attributes."""
    # 使用全局随机种子创建随机数生成器 rng
    rng = np.random.RandomState(0)
    # 生成一个具有指定数据类型的均匀分布随机数组 X
    X = rng.uniform(size=(20, 10)).astype(global_dtype, copy=False)

    # 创建 LatentDirichletAllocation 对象 lda
    lda = LatentDirichletAllocation(
        n_components=5, random_state=0, learning_method=learning_method
    )
    # 拟合 lda 对象
    lda.fit(X)
    # 断言 lda 对象的组件属性和 exp_dirichlet_component_ 属性的数据类型与全局数据类型匹配
    assert lda.components_.dtype == global_dtype
    assert lda.exp_dirichlet_component_.dtype == global_dtype


# 使用参数化测试框架，测试 LatentDirichletAllocation 对象的数值一致性
@pytest.mark.parametrize("learning_method", ("batch", "online"))
def test_lda_numerical_consistency(learning_method, global_random_seed):
    """Check numerical consistency between np.float32 and np.float64."""
    # 使用全局随机种子创建随机数生成器 rng
    rng = np.random.RandomState(global_random_seed)
    # 生成一个形状为 (20, 10) 的均匀分布随机数组 X64
    X64 = rng.uniform(size=(20, 10))
    # 将 X64 数组转换为 np.float32 类型的数组，并赋值给 X32
    X32 = X64.astype(np.float32)

    # 使用 LatentDirichletAllocation 模型对 X64 进行拟合，设置主题数为 5，随机种子为 global_random_seed，学习方法为 learning_method
    lda_64 = LatentDirichletAllocation(
        n_components=5, random_state=global_random_seed, learning_method=learning_method
    ).fit(X64)

    # 使用 LatentDirichletAllocation 模型对 X32 进行拟合，设置主题数为 5，随机种子为 global_random_seed，学习方法为 learning_method
    lda_32 = LatentDirichletAllocation(
        n_components=5, random_state=global_random_seed, learning_method=learning_method
    ).fit(X32)

    # 断言检查 lda_32 的主题组件与 lda_64 的主题组件是否近似相等
    assert_allclose(lda_32.components_, lda_64.components_)

    # 断言检查 lda_32 对 X32 的变换结果与 lda_64 对 X64 的变换结果是否近似相等
    assert_allclose(lda_32.transform(X32), lda_64.transform(X64))
```