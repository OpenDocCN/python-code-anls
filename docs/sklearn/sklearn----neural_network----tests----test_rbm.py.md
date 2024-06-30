# `D:\src\scipysrc\scikit-learn\sklearn\neural_network\tests\test_rbm.py`

```
import re  # 导入正则表达式模块
import sys  # 导入系统相关模块
from io import StringIO  # 导入字符串IO模块

import numpy as np  # 导入NumPy库
import pytest  # 导入pytest测试框架

from sklearn.datasets import load_digits  # 导入加载手写数字数据集的函数
from sklearn.neural_network import BernoulliRBM  # 导入伯努利受限玻尔兹曼机模型
from sklearn.utils._testing import (  # 导入测试工具函数
    assert_allclose,  # 断言所有元素近似相等
    assert_almost_equal,  # 断言所有元素近似相等
    assert_array_equal,  # 断言所有元素完全相等
)
from sklearn.utils.fixes import (  # 导入修复函数
    CSC_CONTAINERS,  # CSC格式容器
    CSR_CONTAINERS,  # CSR格式容器
    LIL_CONTAINERS,  # LIL格式容器
)
from sklearn.utils.validation import assert_all_finite  # 导入断言所有元素为有限数的函数

Xdigits, _ = load_digits(return_X_y=True)  # 加载手写数字数据集，仅返回特征X
Xdigits -= Xdigits.min()  # 特征X归一化，减去最小值
Xdigits /= Xdigits.max()  # 特征X归一化，除以最大值


def test_fit():  # 定义测试函数test_fit
    X = Xdigits.copy()  # 复制特征X数据为X

    rbm = BernoulliRBM(  # 创建伯努利受限玻尔兹曼机模型对象rbm
        n_components=64, learning_rate=0.1, batch_size=10, n_iter=7, random_state=9
    )
    rbm.fit(X)  # 使用X拟合模型rbm

    assert_almost_equal(rbm.score_samples(X).mean(), -21.0, decimal=0)  # 断言模型得分的均值近似为-21.0

    # 检查是否有任何in-place操作修改了X
    assert_array_equal(X, Xdigits)  # 断言X与Xdigits完全相等


def test_partial_fit():  # 定义测试函数test_partial_fit
    X = Xdigits.copy()  # 复制特征X数据为X
    rbm = BernoulliRBM(  # 创建伯努利受限玻尔兹曼机模型对象rbm
        n_components=64, learning_rate=0.1, batch_size=20, random_state=9
    )
    n_samples = X.shape[0]  # 获取样本数
    n_batches = int(np.ceil(float(n_samples) / rbm.batch_size))  # 计算批次数
    batch_slices = np.array_split(X, n_batches)  # 将X分割为多个批次

    for i in range(7):  # 循环7次
        for batch in batch_slices:  # 遍历每个批次
            rbm.partial_fit(batch)  # 部分拟合模型rbm

    assert_almost_equal(rbm.score_samples(X).mean(), -21.0, decimal=0)  # 断言模型得分的均值近似为-21.0
    assert_array_equal(X, Xdigits)  # 断言X与Xdigits完全相等


def test_transform():  # 定义测试函数test_transform
    X = Xdigits[:100]  # 获取前100个样本特征作为X
    rbm1 = BernoulliRBM(n_components=16, batch_size=5, n_iter=5, random_state=42)  # 创建伯努利受限玻尔兹曼机模型对象rbm1
    rbm1.fit(X)  # 使用X拟合模型rbm1

    Xt1 = rbm1.transform(X)  # 对X进行变换得到Xt1
    Xt2 = rbm1._mean_hiddens(X)  # 计算X的隐藏层均值得到Xt2

    assert_array_equal(Xt1, Xt2)  # 断言Xt1与Xt2完全相等


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_small_sparse(csr_container):  # 参数化测试函数test_small_sparse，接收csr_container参数
    # BernoulliRBM应该能够处理小的稀疏矩阵。
    X = csr_container(Xdigits[:4])  # 使用csr_container处理前4个样本特征形成稀疏矩阵X
    BernoulliRBM().fit(X)  # 创建伯努利受限玻尔兹曼机模型并拟合X，不应抛出异常


@pytest.mark.parametrize("sparse_container", CSC_CONTAINERS + CSR_CONTAINERS)
def test_small_sparse_partial_fit(sparse_container):  # 参数化测试函数test_small_sparse_partial_fit，接收sparse_container参数
    X_sparse = sparse_container(Xdigits[:100])  # 使用sparse_container处理前100个样本特征形成稀疏矩阵X_sparse
    X = Xdigits[:100].copy()  # 复制前100个样本特征为X

    rbm1 = BernoulliRBM(  # 创建伯努利受限玻尔兹曼机模型对象rbm1
        n_components=64, learning_rate=0.1, batch_size=10, random_state=9
    )
    rbm2 = BernoulliRBM(  # 创建伯努利受限玻尔兹曼机模型对象rbm2
        n_components=64, learning_rate=0.1, batch_size=10, random_state=9
    )

    rbm1.partial_fit(X_sparse)  # 使用稀疏矩阵X_sparse部分拟合模型rbm1
    rbm2.partial_fit(X)  # 使用X部分拟合模型rbm2

    assert_almost_equal(  # 断言rbm1和rbm2对X得分的均值近似相等
        rbm1.score_samples(X).mean(), rbm2.score_samples(X).mean(), decimal=0
    )


def test_sample_hiddens():  # 定义测试函数test_sample_hiddens
    rng = np.random.RandomState(0)  # 创建随机数生成器rng
    X = Xdigits[:100]  # 获取前100个样本特征作为X
    rbm1 = BernoulliRBM(n_components=2, batch_size=5, n_iter=5, random_state=42)  # 创建伯努利受限玻尔兹曼机模型对象rbm1
    rbm1.fit(X)  # 使用X拟合模型rbm1

    h = rbm1._mean_hiddens(X[0])  # 计算第一个样本的隐藏层均值h
    hs = np.mean([rbm1._sample_hiddens(X[0], rng) for i in range(100)], 0)  # 计算多次采样后的隐藏层均值hs

    assert_almost_equal(h, hs, decimal=1)  # 断言h与hs的值近似相等


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_fit_gibbs(csc_container):  # 参数化测试函数test_fit_gibbs，接收csc_container参数
    # XXX: this test is very seed-dependent! It probably needs to be rewritten.
    # Gibbs on the RBM hidden layer should be able to recreate [[0], [1]]
    # from the same input
    # 使用`
        # 初始化一个随机数生成器，设置随机种子为42，确保结果可复现
        rng = np.random.RandomState(42)
        # 创建一个包含两个样本的数据数组，数据格式为二维数组，每个样本一个特征
        X = np.array([[0.0], [1.0]])
        # 初始化一个 BernoulliRBM 模型，设置组件数量为2，批处理大小为2，迭代次数为42，使用上面初始化的随机数生成器
        rbm1 = BernoulliRBM(n_components=2, batch_size=2, n_iter=42, random_state=rng)
        # 使用数据 X 拟合模型，进行训练
        rbm1.fit(X)
        # 验证模型的权重，检查组件是否接近预期值
        assert_almost_equal(
            rbm1.components_, np.array([[0.02649814], [0.02009084]]), decimal=4
        )
        # 使用 gibbs 方法生成数据，检查其与输入数据 X 是否相等
        assert_almost_equal(rbm1.gibbs(X), X)
    
        # 初始化另一个随机数生成器，设置随机种子为42，确保结果可复现
        rng = np.random.RandomState(42)
        # 创建一个稀疏矩阵形式的输入数据，包含两个样本，每个样本一个特征
        X = csc_container([[0.0], [1.0]])
        # 初始化另一个 BernoulliRBM 模型，设置组件数量为2，批处理大小为2，迭代次数为42，使用上面初始化的随机数生成器
        rbm2 = BernoulliRBM(n_components=2, batch_size=2, n_iter=42, random_state=rng)
        # 使用稀疏矩阵数据 X 拟合模型，进行训练
        rbm2.fit(X)
        # 验证模型的权重，检查组件是否接近预期值
        assert_almost_equal(
            rbm2.components_, np.array([[0.02649814], [0.02009084]]), decimal=4
        )
        # 使用 gibbs 方法生成数据，检查其与输入稀疏矩阵数据 X 转换为数组后是否相等
        assert_almost_equal(rbm2.gibbs(X), X.toarray())
        # 验证两个模型的组件是否相等
        assert_almost_equal(rbm1.components_, rbm2.components_)
def test_gibbs_smoke():
    # 检查在完整数字数据集上采样时是否避免出现 NaN。
    # 同时检查再次采样是否会产生不同的结果。
    X = Xdigits
    # 创建一个 BernoulliRBM 对象，设置参数并拟合数据 X
    rbm1 = BernoulliRBM(n_components=42, batch_size=40, n_iter=20, random_state=42)
    rbm1.fit(X)
    # 使用 Gibbs 采样方法对 X 进行采样
    X_sampled = rbm1.gibbs(X)
    # 断言采样后的数据没有 NaN
    assert_all_finite(X_sampled)
    # 再次使用 Gibbs 采样方法对 X 进行采样
    X_sampled2 = rbm1.gibbs(X)
    # 断言两次采样的结果在每行中是否有不同
    assert np.all((X_sampled != X_sampled2).max(axis=1))


@pytest.mark.parametrize("lil_containers", LIL_CONTAINERS)
def test_score_samples(lil_containers):
    # 测试 score_samples 方法（伪似然）。
    # 断言伪似然计算时没有进行裁剪。
    # 参考 Fabian 的博客，http://bit.ly/1iYefRk
    rng = np.random.RandomState(42)
    # 创建一个二维数组 X，包含两行分别是全零和全一
    X = np.vstack([np.zeros(1000), np.ones(1000)])
    # 创建一个 BernoulliRBM 对象，设置参数并拟合数据 X
    rbm1 = BernoulliRBM(n_components=10, batch_size=2, n_iter=10, random_state=rng)
    rbm1.fit(X)
    # 断言 score_samples 的结果是否全部小于 -300
    assert (rbm1.score_samples(X) < -300).all()

    # 稀疏与密集输入不应影响输出。同时测试稀疏输入验证。
    rbm1.random_state = 42
    d_score = rbm1.score_samples(X)
    rbm1.random_state = 42
    s_score = rbm1.score_samples(lil_containers(X))
    # 断言稠密输入和稀疏输入的伪似然得分几乎相等
    assert_almost_equal(d_score, s_score)

    # 测试数值稳定性 (#2785): 以前可能会生成无穷大并导致异常
    with np.errstate(under="ignore"):
        rbm1.score_samples([np.arange(1000) * 100])


def test_rbm_verbose():
    # 测试 BernoulliRBM 的 verbose 参数设置为输出迭代过程信息
    rbm = BernoulliRBM(n_iter=2, verbose=10)
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        # 执行拟合过程，并将标准输出重定向到 StringIO 对象
        rbm.fit(Xdigits)
    finally:
        # 恢复原来的标准输出
        sys.stdout = old_stdout


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_sparse_and_verbose(csc_container):
    # 确保 RBM 在 verbose=True 时能处理稀疏输入
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    # 创建一个稀疏输入 X，以 csc_container 的方式组织
    X = csc_container([[0.0], [1.0]])
    # 创建一个 BernoulliRBM 对象，设置参数并拟合数据 X，同时启用 verbose 输出
    rbm = BernoulliRBM(
        n_components=2, batch_size=2, n_iter=1, random_state=42, verbose=True
    )
    try:
        # 执行拟合过程，并将标准输出重定向到 StringIO 对象
        rbm.fit(X)
        s = sys.stdout.getvalue()
        # 确保输出信息的格式正确
        assert re.match(
            r"\[BernoulliRBM\] Iteration 1,"
            r" pseudo-likelihood = -?(\d)+(\.\d+)?,"
            r" time = (\d|\.)+s",
            s,
        )
    finally:
        # 恢复原来的标准输出
        sys.stdout = old_stdout


@pytest.mark.parametrize(
    "dtype_in, dtype_out",
    [(np.float32, np.float32), (np.float64, np.float64), (int, np.float64)],
)
def test_transformer_dtypes_casting(dtype_in, dtype_out):
    # 测试数据类型转换的一致性
    X = Xdigits[:100].astype(dtype_in)
    # 创建一个 BernoulliRBM 对象，设置参数并拟合数据 X
    rbm = BernoulliRBM(n_components=16, batch_size=5, n_iter=5, random_state=42)
    # 对数据 X 进行拟合转换
    Xt = rbm.fit_transform(X)

    # 断言转换后的数据类型与期望的输出类型一致
    assert Xt.dtype == dtype_out, "transform dtype: {} - original dtype: {}".format(
        Xt.dtype, X.dtype
    )


def test_convergence_dtype_consistency():
    # 测试拟合过程中的数据类型一致性
    # 使用 np.float64 类型的输入数据 X_64
    X_64 = Xdigits[:100].astype(np.float64)
    # 创建一个 BernoulliRBM 对象，使用 64 位精度进行训练和转换
    rbm_64 = BernoulliRBM(n_components=16, batch_size=5, n_iter=5, random_state=42)
    # 对输入数据 X_64 进行拟合和转换，返回转换后的数据 Xt_64
    Xt_64 = rbm_64.fit_transform(X_64)
    
    # 将 Xdigits 数据的前 100 行转换为 float32 类型
    X_32 = Xdigits[:100].astype(np.float32)
    # 创建一个 BernoulliRBM 对象，使用 32 位精度进行训练和转换
    rbm_32 = BernoulliRBM(n_components=16, batch_size=5, n_iter=5, random_state=42)
    # 对输入数据 X_32 进行拟合和转换，返回转换后的数据 Xt_32
    Xt_32 = rbm_32.fit_transform(X_32)
    
    # 断言：确保在 32 位和 64 位精度下，转换后的结果和属性足够接近
    assert_allclose(Xt_64, Xt_32, rtol=1e-06, atol=0)
    # 断言：确保在 32 位和 64 位精度下，隐藏层截距（intercept_hidden_）足够接近
    assert_allclose(rbm_64.intercept_hidden_, rbm_32.intercept_hidden_, rtol=1e-06, atol=0)
    # 断言：确保在 32 位和 64 位精度下，可见层截距（intercept_visible_）足够接近
    assert_allclose(rbm_64.intercept_visible_, rbm_32.intercept_visible_, rtol=1e-05, atol=0)
    # 断言：确保在 32 位和 64 位精度下，组件（components_）足够接近
    assert_allclose(rbm_64.components_, rbm_32.components_, rtol=1e-03, atol=0)
    # 断言：确保在 32 位和 64 位精度下，隐藏样本（h_samples_）足够接近
    assert_allclose(rbm_64.h_samples_, rbm_32.h_samples_)
@pytest.mark.parametrize("method", ["fit", "partial_fit"])
def test_feature_names_out(method):
    """Check `get_feature_names_out` for `BernoulliRBM`."""
    # 定义测试函数，用于验证 `BernoulliRBM` 的 `get_feature_names_out` 方法

    n_components = 10
    # 创建 BernoulliRBM 模型对象，设定组件数为 10
    rbm = BernoulliRBM(n_components=n_components)

    # 调用指定的方法（fit 或 partial_fit）来拟合模型，Xdigits 是输入数据
    getattr(rbm, method)(Xdigits)

    # 获取模型的输出特征名列表
    names = rbm.get_feature_names_out()

    # 生成预期的特征名列表，命名规则为 bernoullirbm{i}，其中 i 从 0 到 n_components-1
    expected_names = [f"bernoullirbm{i}" for i in range(n_components)]

    # 断言模型输出的特征名列表与预期的特征名列表相等
    assert_array_equal(expected_names, names)
```