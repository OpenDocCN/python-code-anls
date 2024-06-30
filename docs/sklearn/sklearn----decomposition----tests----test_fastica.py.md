# `D:\src\scipysrc\scikit-learn\sklearn\decomposition\tests\test_fastica.py`

```
"""
Test the fastica algorithm.
"""

import itertools  # 导入 itertools 模块，用于生成迭代器的函数
import os  # 导入 os 模块，提供与操作系统交互的功能
import warnings  # 导入 warnings 模块，用于管理警告信息的功能

import numpy as np  # 导入 NumPy 库，并使用 np 别名

import pytest  # 导入 pytest 库，用于编写和运行测试用例

from scipy import stats  # 从 SciPy 库中导入 stats 模块，提供统计函数和工具

from sklearn.decomposition import PCA, FastICA, fastica  # 从 scikit-learn 库的 decomposition 模块中导入 PCA 和 FastICA 类，以及 fastica 函数
from sklearn.decomposition._fastica import _gs_decorrelation  # 导入 _gs_decorrelation 函数，用于 Gram-Schmidt 正交化
from sklearn.exceptions import ConvergenceWarning  # 导入 ConvergenceWarning 异常类
from sklearn.utils._testing import assert_allclose  # 从 scikit-learn 的 utils 模块中导入 assert_allclose 函数，用于比较两个数组是否在一定容差范围内相等


def center_and_norm(x, axis=-1):
    """Centers and norms x **in place**

    Parameters
    -----------
    x: ndarray
        Array with an axis of observations (statistical units) measured on
        random variables.
    axis: int, optional
        Axis along which the mean and variance are calculated.
    """
    x = np.rollaxis(x, axis)  # 将指定轴移动到新位置
    x -= x.mean(axis=0)  # 减去均值使数据居中
    x /= x.std(axis=0)  # 除以标准差以进行归一化


def test_gs():
    # Test gram schmidt orthonormalization
    # generate a random orthogonal  matrix
    rng = np.random.RandomState(0)  # 使用种子为0创建随机数生成器
    W, _, _ = np.linalg.svd(rng.randn(10, 10))  # 生成一个随机正交矩阵
    w = rng.randn(10)  # 生成长度为10的随机向量
    _gs_decorrelation(w, W, 10)  # 对 w 进行 Gram-Schmidt 正交化
    assert (w**2).sum() < 1.0e-10  # 检查 w 是否已被正交化

    w = rng.randn(10)  # 重新生成长度为10的随机向量
    u = _gs_decorrelation(w, W, 5)  # 对 w 进行 Gram-Schmidt 正交化，返回结果保存在 u 中
    tmp = np.dot(u, W.T)  # 计算 u 与 W 转置的点积
    assert (tmp[:5] ** 2).sum() < 1.0e-10  # 检查结果的前5个元素是否已被正交化


def test_fastica_attributes_dtypes(global_dtype):
    rng = np.random.RandomState(0)  # 使用种子为0创建随机数生成器
    X = rng.random_sample((100, 10)).astype(global_dtype, copy=False)  # 创建一个100x10的随机数组，并将其类型转换为指定的全局数据类型
    fica = FastICA(
        n_components=5, max_iter=1000, whiten="unit-variance", random_state=0
    ).fit(X)  # 创建 FastICA 对象并拟合数据 X
    assert fica.components_.dtype == global_dtype  # 检查返回的成分的数据类型是否与指定的全局数据类型一致
    assert fica.mixing_.dtype == global_dtype  # 检查混合矩阵的数据类型是否与指定的全局数据类型一致
    assert fica.mean_.dtype == global_dtype  # 检查均值的数据类型是否与指定的全局数据类型一致
    assert fica.whitening_.dtype == global_dtype  # 检查白化矩阵的数据类型是否与指定的全局数据类型一致


def test_fastica_return_dtypes(global_dtype):
    rng = np.random.RandomState(0)  # 使用种子为0创建随机数生成器
    X = rng.random_sample((100, 10)).astype(global_dtype, copy=False)  # 创建一个100x10的随机数组，并将其类型转换为指定的全局数据类型
    k_, mixing_, s_ = fastica(
        X, max_iter=1000, whiten="unit-variance", random_state=rng
    )  # 应用 FastICA 算法，并返回结果
    assert k_.dtype == global_dtype  # 检查 k_ 的数据类型是否与指定的全局数据类型一致
    assert mixing_.dtype == global_dtype  # 检查混合矩阵的数据类型是否与指定的全局数据类型一致
    assert s_.dtype == global_dtype  # 检查分离的数据类型是否与指定的全局数据类型一致


@pytest.mark.parametrize("add_noise", [True, False])
def test_fastica_simple(add_noise, global_random_seed, global_dtype):
    if (
        global_random_seed == 20
        and global_dtype == np.float32
        and not add_noise
        and os.getenv("DISTRIB") == "ubuntu"
    ):
        pytest.xfail(
            "FastICA instability with Ubuntu Atlas build with float32 "
            "global_dtype. For more details, see "
            "https://github.com/scikit-learn/scikit-learn/issues/24131#issuecomment-1208091119"  # noqa
        )

    # Test the FastICA algorithm on very simple data.
    rng = np.random.RandomState(global_random_seed)  # 使用全局种子创建随机数生成器
    n_samples = 1000  # 样本数设为1000
    # Generate two sources:
    s1 = (2 * np.sin(np.linspace(0, 100, n_samples)) > 0) - 1  # 生成第一个源信号
    s2 = stats.t.rvs(1, size=n_samples, random_state=global_random_seed)  # 生成第二个源信号
    s = np.c_[s1, s2].T  # 将两个信号堆叠起来，并转置以使其与 FastICA 函数的预期格式匹配
    center_and_norm(s)  # 对信号进行居中和归一化处理
    s = s.astype(global_dtype)  # 将信号类型转换为指定的全局数据类型
    s1, s2 = s  # 分离处理后的信号

    # Mixing angle
    phi = 0.6  # 混合角度设为0.6
    # 创建一个混合矩阵，根据给定的角度 phi
    mixing = np.array([[np.cos(phi), np.sin(phi)], [np.sin(phi), -np.cos(phi)]])
    # 将混合矩阵的数据类型转换为全局设定的数据类型
    mixing = mixing.astype(global_dtype)
    # 使用混合矩阵对信号矩阵 s 进行线性变换，得到混合后的信号矩阵 m
    m = np.dot(mixing, s)

    # 如果需要添加噪音，对混合后的信号矩阵 m 添加高斯噪音
    if add_noise:
        m += 0.1 * rng.randn(2, 1000)

    # 对混合后的信号矩阵 m 进行中心化和归一化处理
    center_and_norm(m)

    # 定义一个测试函数 g_test，用于作为 fastica 函数的参数
    def g_test(x):
        return x**3, (3 * x**2).mean(axis=-1)

    # 设置算法、非线性函数和白化方式的组合列表
    algos = ["parallel", "deflation"]
    nls = ["logcosh", "exp", "cube", g_test]
    whitening = ["arbitrary-variance", "unit-variance", False]
    
    # 使用 itertools.product 遍历所有算法、非线性函数和白化方式的组合
    for algo, nl, whiten in itertools.product(algos, nls, whitening):
        if whiten:
            # 对混合后的信号矩阵 m 调用 fastica 函数进行独立成分分析
            k_, mixing_, s_ = fastica(
                m.T, fun=nl, whiten=whiten, algorithm=algo, random_state=rng
            )
            # 使用 pytest 来测试预期的 ValueError 是否被触发
            with pytest.raises(ValueError):
                fastica(m.T, fun=np.tanh, whiten=whiten, algorithm=algo)
        else:
            # 使用 PCA 进行主成分分析，并对信号矩阵 X 进行白化处理
            pca = PCA(n_components=2, whiten=True, random_state=rng)
            X = pca.fit_transform(m.T)
            # 对白化后的信号矩阵 X 调用 fastica 函数进行独立成分分析
            k_, mixing_, s_ = fastica(
                X, fun=nl, algorithm=algo, whiten=False, random_state=rng
            )
            # 使用 pytest 来测试预期的 ValueError 是否被触发
            with pytest.raises(ValueError):
                fastica(X, fun=np.tanh, algorithm=algo)
        
        # 转置 s_，以便后续处理
        s_ = s_.T
        
        # 检查是否混合模型在测试中描述的一致性被保持
        if whiten:
            # XXX: 精确重建到标准的相对容差不可能。这可能是因为 add_noise 为 True，
            # 但在 add_noise 为 False 时，我们也需要在 float32 的情况下使用一个非平凡的 atol。
            #
            # 注意，在这个测试中，这两个信号都是非高斯分布的。
            atol = 1e-5 if global_dtype == np.float32 else 0
            # 使用 assert_allclose 检查重构信号与预期信号之间的接近程度
            assert_allclose(np.dot(np.dot(mixing_, k_), m), s_, atol=atol)
        
        # 对估计出的独立成分 s_ 进行中心化和归一化处理
        center_and_norm(s_)
        s1_, s2_ = s_
        
        # 检查估计出的信号是否按照预期的顺序进行了估计
        if abs(np.dot(s1_, s2)) > abs(np.dot(s1_, s1)):
            s2_, s1_ = s_
        
        # 对估计出的第一和第二信号进行缩放
        s1_ *= np.sign(np.dot(s1_, s1))
        s2_ *= np.sign(np.dot(s2_, s2))

        # 检查我们是否正确估计了原始信号
        if not add_noise:
            # 使用 assert_allclose 检查第一和第二信号的标准化后的方差是否接近 1
            assert_allclose(np.dot(s1_, s1) / n_samples, 1, atol=1e-2)
            assert_allclose(np.dot(s2_, s2) / n_samples, 1, atol=1e-2)
        else:
            # 在有噪音的情况下，使用较宽松的容差来检查标准化后的方差是否接近 1
            assert_allclose(np.dot(s1_, s1) / n_samples, 1, atol=1e-1)
            assert_allclose(np.dot(s2_, s2) / n_samples, 1, atol=1e-1)

    # 测试 FastICA 类
    # 使用 fastica 函数获取估计的源信号及其对应的函数
    _, _, sources_fun = fastica(
        m.T, fun=nl, algorithm=algo, random_state=global_random_seed
    )
    # 使用 FastICA 类创建实例，对信号 m 进行独立成分分析
    ica = FastICA(fun=nl, algorithm=algo, random_state=global_random_seed)
    sources = ica.fit_transform(m.T)
    # 使用 assert 检查估计的混合矩阵的形状和估计的源信号的形状是否正确
    assert ica.components_.shape == (2, 2)
    assert sources.shape == (1000, 2)

    # 使用 assert_allclose 检查函数估计的源信号和 FastICA 类估计的源信号是否接近
    assert_allclose(sources_fun, sources)
    # 设置容差以考虑源信号元素不同的数量级
    atol = np.max(np.abs(sources)) * (1e-5 if global_dtype == np.float32 else 1e-7)
    # 使用 assert_allclose 函数检查 sources 和 ica.transform(m.T) 是否在指定的容差范围内接近
    assert_allclose(sources, ica.transform(m.T), atol=atol)
    
    # 断言检查 ica.mixing_ 的形状是否为 (2, 2)
    assert ica.mixing_.shape == (2, 2)
    
    # 创建 FastICA 对象 ica，指定混合函数为双曲正切函数 np.tanh，并指定算法为 algo
    ica = FastICA(fun=np.tanh, algorithm=algo)
    
    # 使用 pytest 的 raises 函数检查是否会引发 ValueError 异常
    with pytest.raises(ValueError):
        # 对 ica 对象应用 fit 方法，拟合传入的数据 m.T
        ica.fit(m.T)
# 测试 FastICA 算法在不进行白化的情况下的表现

def test_fastica_nowhiten():
    # 创建一个2x2的矩阵 m
    m = [[0, 1], [1, 0]]

    # 创建 FastICA 对象，设定参数：不进行白化，随机种子为0
    ica = FastICA(n_components=1, whiten=False, random_state=0)
    
    # 设置警告消息字符串
    warn_msg = "Ignoring n_components with whiten=False."
    
    # 使用 pytest 的 warn 方法检查是否会触发 UserWarning，并匹配警告消息
    with pytest.warns(UserWarning, match=warn_msg):
        # 对矩阵 m 进行 FastICA 拟合
        ica.fit(m)
    
    # 检查是否有属性 mixing_ 被创建
    assert hasattr(ica, "mixing_")


def test_fastica_convergence_fail():
    # 测试 FastICA 算法在非常简单的数据上的表现
    # 确保当容差足够低时会引发 ConvergenceWarning

    # 设置随机数种子
    rng = np.random.RandomState(0)

    # 生成1000个样本点的时间序列 t
    n_samples = 1000
    t = np.linspace(0, 100, n_samples)
    
    # 生成两个源信号 s1 和 s2
    s1 = np.sin(t)
    s2 = np.ceil(np.sin(np.pi * t))
    s = np.c_[s1, s2].T
    
    # 对源信号进行中心化和归一化处理
    center_and_norm(s)

    # 生成混合矩阵 mixing
    mixing = rng.randn(6, 2)
    m = np.dot(mixing, s)

    # 使用 tolerance 为 0 运行 FastICA，以确保不收敛
    warn_msg = (
        "FastICA did not converge. Consider increasing tolerance "
        "or the maximum number of iterations."
    )
    
    # 使用 pytest 的 warn 方法检查是否会触发 ConvergenceWarning，并匹配警告消息
    with pytest.warns(ConvergenceWarning, match=warn_msg):
        ica = FastICA(
            algorithm="parallel", n_components=2, random_state=rng, max_iter=2, tol=0.0
        )
        ica.fit(m.T)


@pytest.mark.parametrize("add_noise", [True, False])
def test_non_square_fastica(add_noise):
    # 测试 FastICA 算法在非方形数据上的表现

    # 设置随机数种子
    rng = np.random.RandomState(0)

    # 生成1000个样本点的时间序列 t
    n_samples = 1000
    t = np.linspace(0, 100, n_samples)
    
    # 生成两个源信号 s1 和 s2
    s1 = np.sin(t)
    s2 = np.ceil(np.sin(np.pi * t))
    s = np.c_[s1, s2].T
    
    # 对源信号进行中心化和归一化处理
    center_and_norm(s)
    
    # 将 s1 和 s2 分开
    s1, s2 = s

    # 生成混合矩阵 mixing
    mixing = rng.randn(6, 2)
    m = np.dot(mixing, s)

    # 如果需要添加噪声，则向 m 中添加噪声
    if add_noise:
        m += 0.1 * rng.randn(6, n_samples)

    # 对混合信号进行中心化和归一化处理
    center_and_norm(m)

    # 使用 fastica 函数进行 FastICA，设定参数
    k_, mixing_, s_ = fastica(
        m.T, n_components=2, whiten="unit-variance", random_state=rng
    )
    s_ = s_.T

    # 检查文档字符串中描述的混合模型是否成立
    assert_allclose(s_, np.dot(np.dot(mixing_, k_), m))

    # 对 s_ 进行中心化和归一化处理
    center_and_norm(s_)
    s1_, s2_ = s_

    # 检查源信号是否以错误的顺序被估计
    if abs(np.dot(s1_, s2)) > abs(np.dot(s1_, s1)):
        s2_, s1_ = s_
    
    # 通过乘以符号来调整 s1_ 和 s2_
    s1_ *= np.sign(np.dot(s1_, s1))
    s2_ *= np.sign(np.dot(s2_, s2))

    # 检查是否已正确估计原始源信号
    if not add_noise:
        assert_allclose(np.dot(s1_, s1) / n_samples, 1, atol=1e-3)
        assert_allclose(np.dot(s2_, s2) / n_samples, 1, atol=1e-3)


def test_fit_transform(global_random_seed, global_dtype):
    """使用 FastICA 算法测试转换后数据的单位方差。

    检查 `fit_transform` 是否与应用 `fit` 和 `transform` 后的结果相同。

    Bug #13056
    """
    # 生成全局随机数种子的随机数生成器
    rng = np.random.RandomState(global_random_seed)
    
    # 生成数据 X，是一个100x10的随机浮点数数组
    X = rng.random_sample((100, 10)).astype(global_dtype)
    
    # 最大迭代次数
    max_iter = 300
    # 对于每个独立成分分析（ICA）的参数组合，依次执行以下操作
    for whiten, n_components in [["unit-variance", 5], [False, None]]:
        # 如果未指定 n_components，则使用输入数据 X 的列数作为默认值
        n_components_ = n_components if n_components is not None else X.shape[1]

        # 创建 FastICA 对象，配置参数包括独立成分数、最大迭代次数、是否白化、随机种子
        ica = FastICA(
            n_components=n_components, max_iter=max_iter, whiten=whiten, random_state=0
        )
        
        # 忽略 RuntimeWarning，确保数值误差不导致对负数求平方根
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            # XXX: 对于某些种子值，模型可能不会收敛。
            # 然而，这并不是我们在此测试中要验证的。
            warnings.simplefilter("ignore", ConvergenceWarning)
            # 对输入数据 X 进行独立成分分析，返回转换后的结果 Xt
            Xt = ica.fit_transform(X)
        
        # 断言独立成分向量的形状为 (n_components_, 10)
        assert ica.components_.shape == (n_components_, 10)
        # 断言转换后的数据 Xt 的形状为 (样本数, n_components_)
        assert Xt.shape == (X.shape[0], n_components_)

        # 创建另一个 FastICA 对象，使用相同的配置参数
        ica2 = FastICA(
            n_components=n_components, max_iter=max_iter, whiten=whiten, random_state=0
        )
        
        # 再次忽略 RuntimeWarning 和 ConvergenceWarning
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            warnings.simplefilter("ignore", ConvergenceWarning)
            # 对输入数据 X 执行独立成分分析，不返回转换后的结果
            ica2.fit(X)
        
        # 断言第二次独立成分向量的形状为 (n_components_, 10)
        assert ica2.components_.shape == (n_components_, 10)
        # 使用第二个模型对输入数据 X 进行转换，返回结果 Xt2
        Xt2 = ica2.transform(X)

        # XXX: 对于 float32 类型的数据，为了确保测试通过，需要设置 atol。
        # 这是否暴露了一个 bug？
        if global_dtype:
            atol = np.abs(Xt2).mean() / 1e6
        else:
            # 对于 float64 类型的数据，默认的相对容差（rtol）已足够
            atol = 0.0
        # 使用 assert_allclose 断言 Xt 和 Xt2 之间的所有元素都在指定的容差范围内
        assert_allclose(Xt, Xt2, atol=atol)
# 使用 pytest.mark.filterwarnings 忽略特定警告信息
# 设置测试参数化，包括 whiten 类型、n_components 和预期的混合形状
@pytest.mark.filterwarnings("ignore:Ignoring n_components with whiten=False.")
@pytest.mark.parametrize(
    "whiten, n_components, expected_mixing_shape",
    [
        ("arbitrary-variance", 5, (10, 5)),  # 使用任意方差，预期混合矩阵形状为 (10, 5)
        ("arbitrary-variance", 10, (10, 10)),  # 使用任意方差，预期混合矩阵形状为 (10, 10)
        ("unit-variance", 5, (10, 5)),  # 使用单位方差，预期混合矩阵形状为 (10, 5)
        ("unit-variance", 10, (10, 10)),  # 使用单位方差，预期混合矩阵形状为 (10, 10)
        (False, 5, (10, 10)),  # 不使用白化，预期混合矩阵形状为 (10, 10)
        (False, 10, (10, 10)),  # 不使用白化，预期混合矩阵形状为 (10, 10)
    ],
)
def test_inverse_transform(
    whiten, n_components, expected_mixing_shape, global_random_seed, global_dtype
):
    # 测试 FastICA.inverse_transform 方法
    n_samples = 100
    rng = np.random.RandomState(global_random_seed)
    X = rng.random_sample((n_samples, 10)).astype(global_dtype)

    # 创建 FastICA 对象
    ica = FastICA(n_components=n_components, random_state=rng, whiten=whiten)
    
    # 捕获警告，忽略收敛警告，因为对于有效的逆转换定义不会影响
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        # 拟合并转换数据
        Xt = ica.fit_transform(X)
    
    # 断言混合矩阵的形状符合预期
    assert ica.mixing_.shape == expected_mixing_shape
    
    # 反向转换并断言形状匹配
    X2 = ica.inverse_transform(Xt)
    assert X.shape == X2.shape

    # 在非降维情况下进行可逆性测试
    if n_components == X.shape[1]:
        # 对于 float32 数据，必须设置 atol 以使测试通过，是否揭示了错误？
        if global_dtype:
            atol = np.abs(X2).mean() / 1e5
        else:
            atol = 0.0  # 对于 float64 数据，默认的 rtol 足够了
        assert_allclose(X, X2, atol=atol)


# 测试 FastICA 函数的错误情况
def test_fastica_errors():
    n_features = 3
    n_samples = 10
    rng = np.random.RandomState(0)
    X = rng.random_sample((n_samples, n_features))
    w_init = rng.randn(n_features + 1, n_features + 1)
    
    # 使用 pytest.raises 检测 ValueError 异常，并匹配相应的错误消息
    with pytest.raises(ValueError, match=r"alpha must be in \[1,2\]"):
        fastica(X, fun_args={"alpha": 0})
    with pytest.raises(
        ValueError, match="w_init has invalid shape.+" r"should be \(3L?, 3L?\)"
    ):
        fastica(X, w_init=w_init)


# 测试 FastICA 算法使用单位方差转换后的数据方差是否接近 1.0
def test_fastica_whiten_unit_variance():
    """Test unit variance of transformed data using FastICA algorithm.

    Bug #13056
    """
    rng = np.random.RandomState(0)
    X = rng.random_sample((100, 10))
    n_components = X.shape[1]
    
    # 创建 FastICA 对象，并设置使用单位方差
    ica = FastICA(n_components=n_components, whiten="unit-variance", random_state=0)
    Xt = ica.fit_transform(X)

    # 断言转换后的数据方差接近 1.0
    assert np.var(Xt) == pytest.approx(1.0)


# 参数化测试，测试 FastICA 输出的形状是否符合预期
@pytest.mark.parametrize("whiten", ["arbitrary-variance", "unit-variance", False])
@pytest.mark.parametrize("return_X_mean", [True, False])
@pytest.mark.parametrize("return_n_iter", [True, False])
def test_fastica_output_shape(whiten, return_X_mean, return_n_iter):
    n_features = 3
    n_samples = 10
    rng = np.random.RandomState(0)
    X = rng.random_sample((n_samples, n_features))
    # 计算期望的输出长度，包括固定的常数3以及传入的return_X_mean和return_n_iter的值
    expected_len = 3 + return_X_mean + return_n_iter
    
    # 调用fastica函数进行独立成分分析，传入参数X作为输入数据，whiten表示是否要进行白化处理，
    # return_n_iter和return_X_mean分别控制是否返回迭代次数和均值数据
    out = fastica(
        X, whiten=whiten, return_n_iter=return_n_iter, return_X_mean=return_X_mean
    )
    
    # 断言输出的结果长度与预期的长度相等
    assert len(out) == expected_len
    
    # 如果不需要进行白化处理，断言第一个输出结果应为None
    if not whiten:
        assert out[0] is None
@pytest.mark.parametrize("add_noise", [True, False])
def test_fastica_simple_different_solvers(add_noise, global_random_seed):
    """Test FastICA is consistent between whiten_solvers."""
    # 使用参数化测试，测试 FastICA 在不同白化解算器下的一致性

    rng = np.random.RandomState(global_random_seed)
    n_samples = 1000
    # 生成两个信号源：
    s1 = (2 * np.sin(np.linspace(0, 100, n_samples)) > 0) - 1
    s2 = stats.t.rvs(1, size=n_samples, random_state=rng)
    s = np.c_[s1, s2].T
    # 中心化和归一化信号源
    center_and_norm(s)
    s1, s2 = s

    # 混合角度
    phi = rng.rand() * 2 * np.pi
    mixing = np.array([[np.cos(phi), np.sin(phi)], [np.sin(phi), -np.cos(phi)]])
    m = np.dot(mixing, s)

    if add_noise:
        m += 0.1 * rng.randn(2, 1000)

    # 再次中心化和归一化混合后的信号
    center_and_norm(m)

    outs = {}
    for solver in ("svd", "eigh"):
        # 使用不同的解算器初始化 FastICA 对象
        ica = FastICA(random_state=0, whiten="unit-variance", whiten_solver=solver)
        # 拟合并转换混合信号
        sources = ica.fit_transform(m.T)
        outs[solver] = sources
        assert ica.components_.shape == (2, 2)
        assert sources.shape == (1000, 2)

    # 比较结果，使用小的容差以减少测试的脆弱性
    assert_allclose(outs["eigh"], outs["svd"], atol=1e-12)


def test_fastica_eigh_low_rank_warning(global_random_seed):
    """Test FastICA eigh solver raises warning for low-rank data."""
    # 测试 FastICA 的 eigh 解算器在低秩数据时是否会发出警告

    rng = np.random.RandomState(global_random_seed)
    A = rng.randn(10, 2)
    X = A @ A.T  # 创建一个低秩数据矩阵
    ica = FastICA(random_state=0, whiten="unit-variance", whiten_solver="eigh")
    msg = "There are some small singular values"
    with pytest.warns(UserWarning, match=msg):
        # 拟合低秩数据，期望会触发警告
        ica.fit(X)
```