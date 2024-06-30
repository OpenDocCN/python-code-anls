# `D:\src\scipysrc\scikit-learn\sklearn\feature_selection\tests\test_mutual_info.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于单元测试

from sklearn.datasets import make_classification, make_regression  # 导入数据生成函数
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression  # 导入互信息计算函数
from sklearn.feature_selection._mutual_info import _compute_mi  # 导入互信息计算核心函数
from sklearn.utils import check_random_state  # 导入随机状态检查函数
from sklearn.utils._testing import (  # 导入用于测试的工具函数
    assert_allclose,
    assert_array_equal,
)
from sklearn.utils.fixes import CSR_CONTAINERS  # 导入用于修复的函数库


def test_compute_mi_dd():
    # 在离散情况下，计算是直接的，并且可以在给定向量上手动完成。
    x = np.array([0, 1, 1, 0, 0])  # 第一个离散变量
    y = np.array([1, 0, 0, 0, 1])  # 第二个离散变量

    H_x = H_y = -(3 / 5) * np.log(3 / 5) - (2 / 5) * np.log(2 / 5)  # 计算边缘熵
    H_xy = -1 / 5 * np.log(1 / 5) - 2 / 5 * np.log(2 / 5) - 2 / 5 * np.log(2 / 5)  # 计算联合熵
    I_xy = H_x + H_y - H_xy  # 计算互信息

    assert_allclose(_compute_mi(x, y, x_discrete=True, y_discrete=True), I_xy)  # 断言互信息的计算结果与预期值相近


def test_compute_mi_cc(global_dtype):
    # 对于两个连续变量，一个好的方法是在双变量正态分布上进行测试，其中互信息是已知的。

    # 分布的均值，与互信息无关。
    mean = np.zeros(2)

    # 设置具有相关系数0.5的协方差矩阵。
    sigma_1 = 1
    sigma_2 = 10
    corr = 0.5
    cov = np.array(
        [
            [sigma_1**2, corr * sigma_1 * sigma_2],
            [corr * sigma_1 * sigma_2, sigma_2**2],
        ]
    )

    # 真实的理论互信息。
    I_theory = np.log(sigma_1) + np.log(sigma_2) - 0.5 * np.log(np.linalg.det(cov))

    rng = check_random_state(0)
    Z = rng.multivariate_normal(mean, cov, size=1000).astype(global_dtype, copy=False)

    x, y = Z[:, 0], Z[:, 1]

    # 理论值和计算值不会非常接近
    # 在此处我们使用较大的相对容差进行检查
    for n_neighbors in [3, 5, 7]:
        I_computed = _compute_mi(
            x, y, x_discrete=False, y_discrete=False, n_neighbors=n_neighbors
        )
        assert_allclose(I_computed, I_theory, rtol=1e-1)


def test_compute_mi_cd(global_dtype):
    # 要测试，定义如下的联合分布:
    # p(x, y) = p(x) p(y | x)
    # X ~ Bernoulli(p)
    # (Y | x = 0) ~ Uniform(-1, 1)
    # (Y | x = 1) ~ Uniform(0, 2)

    # 使用以下互信息公式:
    # I(X; Y) = H(Y) - H(Y | X)
    # 可以手动计算两个熵:
    # H(Y) = -(1-p)/2 * ln((1-p)/2) - p/2*log(p/2) - 1/2*log(1/2)
    # H(Y | X) = ln(2)

    # 现在我们需要实现从我们的分布中抽样，这可以很容易地使用条件分布逻辑来完成。

    n_samples = 1000
    rng = check_random_state(0)
    # 对于每个概率值 p 进行循环，计算随机变量 x
    for p in [0.3, 0.5, 0.7]:
        # 使用均匀分布生成大小为 n_samples 的布尔数组 x，表示 x > p 的情况
        x = rng.uniform(size=n_samples) > p

        # 使用全局数据类型 global_dtype 创建大小为 n_samples 的空数组 y
        y = np.empty(n_samples, global_dtype)
        
        # 根据 x 的值为 0 或非 0，分别生成随机数填充 y 数组的对应位置
        mask = x == 0
        y[mask] = rng.uniform(-1, 1, size=np.sum(mask))  # 当 x == 0 时，在 y 中填充 [-1, 1] 之间的随机数
        y[~mask] = rng.uniform(0, 2, size=np.sum(~mask))  # 当 x != 0 时，在 y 中填充 [0, 2] 之间的随机数

        # 计算理论上的互信息 I_theory
        I_theory = -0.5 * (
            (1 - p) * np.log(0.5 * (1 - p)) + p * np.log(0.5 * p) + np.log(0.5)
        ) - np.log(2)

        # 断言计算得到的互信息 I_computed 与理论值 I_theory 相似度在相对误差 1e-1 范围内
        for n_neighbors in [3, 5, 7]:
            I_computed = _compute_mi(
                x, y, x_discrete=True, y_discrete=False, n_neighbors=n_neighbors
            )
            assert_allclose(I_computed, I_theory, rtol=1e-1)
# 测试计算互信息时添加唯一标签不改变互信息的情况。
def test_compute_mi_cd_unique_label(global_dtype):
    # 设定样本数量为100
    n_samples = 100
    # 生成长度为n_samples的随机布尔数组x，值大于0.5为True，否则为False
    x = np.random.uniform(size=n_samples) > 0.5

    # 创建长度为n_samples的空数组y，数据类型为global_dtype
    y = np.empty(n_samples, global_dtype)
    # 根据x中为0的位置，将y对应位置填充为在[-1, 1]区间内的随机数
    mask = x == 0
    y[mask] = np.random.uniform(-1, 1, size=np.sum(mask))
    # 根据x中不为0的位置，将y对应位置填充为在[0, 2]区间内的随机数
    y[~mask] = np.random.uniform(0, 2, size=np.sum(~mask))

    # 计算x和y之间的互信息，其中x为离散，y为连续
    mi_1 = _compute_mi(x, y, x_discrete=True, y_discrete=False)

    # 在x末尾添加值为2的元素
    x = np.hstack((x, 2))
    # 在y末尾添加值为10的元素
    y = np.hstack((y, 10))
    # 重新计算扩展后x和y之间的互信息
    mi_2 = _compute_mi(x, y, x_discrete=True, y_discrete=False)

    # 断言两次计算得到的互信息应该接近
    assert_allclose(mi_1, mi_2)


# 测试特征排序是否符合互信息的期望
def test_mutual_info_classif_discrete(global_dtype):
    # 创建一个3x5的数组X，包含离散和连续特征
    X = np.array(
        [[0, 0, 0], [1, 1, 0], [2, 0, 1], [2, 0, 1], [2, 0, 1]], dtype=global_dtype
    )
    # 创建长度为5的目标数组y，包含离散标签
    y = np.array([0, 1, 2, 2, 1])

    # 使用互信息分类算法计算特征X对y的互信息，其中离散特征为True
    mi = mutual_info_classif(X, y, discrete_features=True)
    # 断言互信息排序后，最高互信息的特征索引为0，其次为2，最低为1
    assert_array_equal(np.argsort(-mi), np.array([0, 2, 1]))


def test_mutual_info_regression(global_dtype):
    # 从多元正态分布生成样本数据Z，其中变量之间有相关性
    T = np.array([[1, 0.5, 2, 1], [0, 1, 0.1, 0.0], [0, 0.1, 1, 0.1], [0, 0.1, 0.1, 1]])
    cov = T.dot(T.T)
    mean = np.zeros(4)

    # 使用随机状态生成器生成Z的1000个样本，数据类型为global_dtype
    rng = check_random_state(0)
    Z = rng.multivariate_normal(mean, cov, size=1000).astype(global_dtype, copy=False)
    # X为Z的除第一列外的所有列，y为Z的第一列
    X = Z[:, 1:]
    y = Z[:, 0]

    # 使用互信息回归算法计算特征X对y的互信息，验证输出数据类型是否为np.float64
    mi = mutual_info_regression(X, y, random_state=0)
    assert_array_equal(np.argsort(-mi), np.array([1, 2, 0]))
    # XXX: 应该修复mutual_info_regression以避免将float32输入向上转换为float64？
    assert mi.dtype == np.float64


def test_mutual_info_classif_mixed(global_dtype):
    # 目标y为离散值，有两个连续和一个离散特征的X数组
    rng = check_random_state(0)
    # 创建1000x3的随机数组X，包含两个连续特征和一个变换后的离散特征
    X = rng.rand(1000, 3).astype(global_dtype, copy=False)
    X[:, 1] += X[:, 0]
    # 创建y作为X第一列和第三列的线性组合的离散标签
    y = ((0.5 * X[:, 0] + X[:, 2]) > 0.5).astype(int)
    # 将X的第三列转换为二进制值
    X[:, 2] = X[:, 2] > 0.5

    # 使用互信息分类算法计算特征X对y的互信息，其中第三个特征为离散特征
    mi = mutual_info_classif(X, y, discrete_features=[2], n_neighbors=3, random_state=0)
    # 断言互信息排序后，最高互信息的特征索引为2，其次为0，最低为1
    assert_array_equal(np.argsort(-mi), [2, 0, 1])
    for n_neighbors in [5, 7, 9]:
        # 使用不同的n_neighbors重新计算互信息
        mi_nn = mutual_info_classif(
            X, y, discrete_features=[2], n_neighbors=n_neighbors, random_state=0
        )
        # 断言连续特征的互信息随着n_neighbors的增加而增加
        assert mi_nn[0] > mi[0]
        assert mi_nn[1] > mi[1]
        # 断言离散特征的互信息不受n_neighbors影响，应该保持不变
        assert mi_nn[2] == mi[2]
# 使用 pytest 的参数化装饰器来运行多个测试用例，其中 csr_container 是一个参数
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_mutual_info_options(global_dtype, csr_container):
    # 创建一个 NumPy 数组 X，包含特定的整数值
    X = np.array(
        [[0, 0, 0], [1, 1, 0], [2, 0, 1], [2, 0, 1], [2, 0, 1]], dtype=global_dtype
    )
    # 创建一个 NumPy 数组 y，包含特定的整数值
    y = np.array([0, 1, 2, 2, 1], dtype=global_dtype)
    # 使用 csr_container 函数将 X 转换为稀疏矩阵表示
    X_csr = csr_container(X)

    # 对于 mutual_info_regression 和 mutual_info_classif 中的每一个函数，执行以下测试
    for mutual_info in (mutual_info_regression, mutual_info_classif):
        # 测试当 discrete_features 参数为 False 时抛出 ValueError 异常
        with pytest.raises(ValueError):
            mutual_info(X_csr, y, discrete_features=False)
        # 测试当 discrete_features 参数为 "manual" 时抛出 ValueError 异常
        with pytest.raises(ValueError):
            mutual_info(X, y, discrete_features="manual")
        # 测试当 discrete_features 参数为列表 [True, False, True] 时抛出 ValueError 异常
        with pytest.raises(ValueError):
            mutual_info(X_csr, y, discrete_features=[True, False, True])
        # 测试当 discrete_features 参数为列表 [True, False, True, False] 时抛出 IndexError 异常
        with pytest.raises(IndexError):
            mutual_info(X, y, discrete_features=[True, False, True, False])
        # 测试当 discrete_features 参数为列表 [1, 4] 时抛出 IndexError 异常
        with pytest.raises(IndexError):
            mutual_info(X, y, discrete_features=[1, 4])

        # 计算不同参数设置下的互信息
        mi_1 = mutual_info(X, y, discrete_features="auto", random_state=0)
        mi_2 = mutual_info(X, y, discrete_features=False, random_state=0)
        mi_3 = mutual_info(X_csr, y, discrete_features="auto", random_state=0)
        mi_4 = mutual_info(X_csr, y, discrete_features=True, random_state=0)
        mi_5 = mutual_info(X, y, discrete_features=[True, False, True], random_state=0)
        mi_6 = mutual_info(X, y, discrete_features=[0, 2], random_state=0)

        # 断言不同设置下的互信息值是否接近
        assert_allclose(mi_1, mi_2)
        assert_allclose(mi_3, mi_4)
        assert_allclose(mi_5, mi_6)

        # 断言 mi_1 和 mi_3 的值不完全相等
        assert not np.allclose(mi_1, mi_3)


# 使用 pytest 的参数化装饰器来运行多个测试用例，其中 correlated 是一个参数
@pytest.mark.parametrize("correlated", [True, False])
def test_mutual_information_symmetry_classif_regression(correlated, global_random_seed):
    """Check that `mutual_info_classif` and `mutual_info_regression` are
    symmetric by switching the target `y` as `feature` in `X` and vice
    versa.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/23720
    """
    # 使用全局随机种子创建随机数生成器
    rng = np.random.RandomState(global_random_seed)
    n = 100
    d = rng.randint(10, size=n)

    if correlated:
        c = d.astype(np.float64)
    else:
        c = rng.normal(0, 1, size=n)

    # 计算使用 mutual_info_classif 的互信息
    mi_classif = mutual_info_classif(
        c[:, None], d, discrete_features=[False], random_state=global_random_seed
    )

    # 计算使用 mutual_info_regression 的互信息
    mi_regression = mutual_info_regression(
        d[:, None], c, discrete_features=[True], random_state=global_random_seed
    )

    # 断言 mutual_info_classif 和 mutual_info_regression 的结果是否近似相等
    assert mi_classif == pytest.approx(mi_regression)


def test_mutual_info_regression_X_int_dtype(global_random_seed):
    """Check that results agree when X is integer dtype and float dtype.

    Non-regression test for Issue #26696.
    """
    # 使用全局随机种子创建随机数生成器
    rng = np.random.RandomState(global_random_seed)
    X = rng.randint(100, size=(100, 10))
    X_float = X.astype(np.float64, copy=True)
    y = rng.randint(100, size=100)

    # 计算使用整数类型 X 的互信息
    expected = mutual_info_regression(X_float, y, random_state=global_random_seed)
    result = mutual_info_regression(X, y, random_state=global_random_seed)
    # 断言整数类型 X 和浮点类型 X 的互信息结果是否全部接近
    assert_allclose(result, expected)
# 使用 pytest.mark.parametrize 装饰器定义参数化测试函数，用于测试互信息函数在不同数据生成器下的行为
@pytest.mark.parametrize(
    "mutual_info_func, data_generator",
    [
        # 参数化测试的参数列表，包括互信息回归函数和回归数据生成器
        (mutual_info_regression, make_regression),
        # 包括互信息分类函数和分类数据生成器
        (mutual_info_classif, make_classification),
    ],
)
# 定义测试函数 test_mutual_info_n_jobs，用于检查不同 `n_jobs` 下结果的一致性
def test_mutual_info_n_jobs(global_random_seed, mutual_info_func, data_generator):
    """Check that results are consistent with different `n_jobs`."""
    # 使用指定的数据生成器生成数据 X, y
    X, y = data_generator(random_state=global_random_seed)
    # 分别使用单线程 (n_jobs=1) 和双线程 (n_jobs=2) 运行互信息函数，使用相同的随机种子
    single_job = mutual_info_func(X, y, random_state=global_random_seed, n_jobs=1)
    multi_job = mutual_info_func(X, y, random_state=global_random_seed, n_jobs=2)
    # 断言单线程和双线程运行的结果应该非常接近（使用 assert_allclose 进行比较）
    assert_allclose(single_job, multi_job)
```