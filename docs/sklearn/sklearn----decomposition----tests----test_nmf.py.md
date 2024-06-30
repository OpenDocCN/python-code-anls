# `D:\src\scipysrc\scikit-learn\sklearn\decomposition\tests\test_nmf.py`

```
# 导入必要的库和模块
import re
import sys
import warnings
from io import StringIO

import numpy as np
import pytest
from scipy import linalg

# 导入 sklearn 中的相关模块和函数
from sklearn.base import clone
from sklearn.decomposition import NMF, MiniBatchNMF, non_negative_factorization
from sklearn.decomposition import _nmf as nmf  # 用于测试内部实现的别名
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    ignore_warnings,
)
from sklearn.utils.extmath import squared_norm
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS

# 使用 pytest 的参数化装饰器定义测试函数
@pytest.mark.parametrize(
    ["Estimator", "solver"],
    [[NMF, {"solver": "cd"}], [NMF, {"solver": "mu"}], [MiniBatchNMF, {}]],
)
def test_convergence_warning(Estimator, solver):
    # 设置收敛警告的消息内容
    convergence_warning = (
        "Maximum number of iterations 1 reached. Increase it to improve convergence."
    )
    A = np.ones((2, 2))
    # 使用 pytest 的 warn 函数检查是否触发 ConvergenceWarning，并匹配特定的警告消息
    with pytest.warns(ConvergenceWarning, match=convergence_warning):
        Estimator(max_iter=1, n_components="auto", **solver).fit(A)


def test_initialize_nn_output():
    # 测试初始化过程中不会返回负值的情况
    rng = np.random.mtrand.RandomState(42)
    data = np.abs(rng.randn(10, 10))
    # 针对不同的初始化方式进行测试
    for init in ("random", "nndsvd", "nndsvda", "nndsvdar"):
        # 调用 nmf 模块中的 _initialize_nmf 函数进行初始化
        W, H = nmf._initialize_nmf(data, 10, init=init, random_state=0)
        # 断言 W 和 H 中不含有负值
        assert not ((W < 0).any() or (H < 0).any())


# TODO(1.6): remove the warning filter for `n_components`
@pytest.mark.filterwarnings(
    r"ignore:The multiplicative update \('mu'\) solver cannot update zeros present in"
    r" the initialization",
    "ignore:The default value of `n_components` will change",
)
def test_parameter_checking():
    # 在这里，我们只检查那些未在通用测试中自动测试的无效参数值。

    A = np.ones((2, 2))

    # 测试对 solver='cd' 时 beta_loss 参数为 1.0 的情况是否会触发 ValueError
    msg = "Invalid beta_loss parameter: solver 'cd' does not handle beta_loss = 1.0"
    with pytest.raises(ValueError, match=msg):
        NMF(solver="cd", beta_loss=1.0).fit(A)
    
    # 测试数据中是否存在负值会触发 ValueError 的情况
    msg = "Negative values in data passed to"
    with pytest.raises(ValueError, match=msg):
        NMF().fit(-A)
    clf = NMF(2, tol=0.1).fit(A)
    with pytest.raises(ValueError, match=msg):
        clf.transform(-A)
    with pytest.raises(ValueError, match=msg):
        nmf._initialize_nmf(-A, 2, "nndsvd")

    # 对于不同的初始化方式，测试当 n_components > min(n_samples, n_features) 时是否会触发 ValueError
    for init in ["nndsvd", "nndsvda", "nndsvdar"]:
        msg = re.escape(
            "init = '{}' can only be used when "
            "n_components <= min(n_samples, n_features)".format(init)
        )
        with pytest.raises(ValueError, match=msg):
            NMF(3, init=init).fit(A)
        with pytest.raises(ValueError, match=msg):
            MiniBatchNMF(3, init=init).fit(A)
        with pytest.raises(ValueError, match=msg):
            nmf._initialize_nmf(A, 3, init)


def test_initialize_close():
    # 测试 NNDSVD 初始化过程中的错误
    # 使用随机种子为42初始化随机数生成器
    rng = np.random.mtrand.RandomState(42)
    # 生成一个形状为(10, 10)的随机矩阵，元素取绝对值
    A = np.abs(rng.randn(10, 10))
    # 使用非负矩阵分解的方法初始化 NMF 模型的 W 和 H 矩阵
    W, H = nmf._initialize_nmf(A, 10, init="nndsvd")
    # 计算重构误差，即原始矩阵 A 与 W * H 的差的范数
    error = linalg.norm(np.dot(W, H) - A)
    # 计算矩阵 A 的标准差
    sdev = linalg.norm(A - A.mean())
    # 断言重构误差小于等于矩阵 A 的标准差
    assert error <= sdev
def test_initialize_variants():
    # Test NNDSVD variants correctness
    # 测试 NNDSVD 变体的正确性

    rng = np.random.mtrand.RandomState(42)
    # 创建一个随机数生成器实例 rng，种子为 42

    data = np.abs(rng.randn(10, 10))
    # 生成一个 10x10 的随机数组，然后取绝对值，确保数据非负

    W0, H0 = nmf._initialize_nmf(data, 10, init="nndsvd")
    # 使用 NMF 初始化函数 _initialize_nmf 初始化 W0 和 H0，使用 NNDSVD 方法

    Wa, Ha = nmf._initialize_nmf(data, 10, init="nndsvda")
    # 使用 NMF 初始化函数 _initialize_nmf 初始化 Wa 和 Ha，使用 NNDSVDA 方法

    War, Har = nmf._initialize_nmf(data, 10, init="nndsvdar", random_state=0)
    # 使用 NMF 初始化函数 _initialize_nmf 初始化 War 和 Har，使用 NNDSVDAR 方法，种子为 0

    for ref, evl in ((W0, Wa), (W0, War), (H0, Ha), (H0, Har)):
        assert_almost_equal(evl[ref != 0], ref[ref != 0])
        # 检查参考值 ref 和评估值 evl 在非零位置的近似相等性
        

# ignore UserWarning raised when both solver='mu' and init='nndsvd'
@ignore_warnings(category=UserWarning)
@pytest.mark.parametrize(
    ["Estimator", "solver"],
    [[NMF, {"solver": "cd"}], [NMF, {"solver": "mu"}], [MiniBatchNMF, {}]],
)
@pytest.mark.parametrize("init", (None, "nndsvd", "nndsvda", "nndsvdar", "random"))
@pytest.mark.parametrize("alpha_W", (0.0, 1.0))
@pytest.mark.parametrize("alpha_H", (0.0, 1.0, "same"))
def test_nmf_fit_nn_output(Estimator, solver, init, alpha_W, alpha_H):
    # Test that the decomposition does not contain negative values
    # 测试分解结果不包含负值

    A = np.c_[5.0 - np.arange(1, 6), 5.0 + np.arange(1, 6)]
    # 创建一个包含两列的矩阵 A，每列由 5.0 减去和加上 1 到 5 的序列组成

    model = Estimator(
        n_components=2,
        init=init,
        alpha_W=alpha_W,
        alpha_H=alpha_H,
        random_state=0,
        **solver,
    )
    # 初始化一个估计器实例 model，设置参数 n_components=2, init=init, alpha_W=alpha_W, alpha_H=alpha_H, random_state=0, 并传递 solver 的其余参数

    transf = model.fit_transform(A)
    # 对数据 A 进行拟合并转换

    assert not ((model.components_ < 0).any() or (transf < 0).any())
    # 断言：模型的成分和转换后的结果不包含负值


@pytest.mark.parametrize(
    ["Estimator", "solver"],
    [[NMF, {"solver": "cd"}], [NMF, {"solver": "mu"}], [MiniBatchNMF, {}]],
)
def test_nmf_fit_close(Estimator, solver):
    rng = np.random.mtrand.RandomState(42)
    # 创建一个随机数生成器实例 rng，种子为 42

    # Test that the fit is not too far away
    # 测试拟合结果与真实值之间的误差不会太大

    pnmf = Estimator(
        5,
        init="nndsvdar",
        random_state=0,
        max_iter=600,
        **solver,
    )
    # 初始化一个估计器实例 pnmf，设置参数 n_components=5, init="nndsvdar", random_state=0, max_iter=600，并传递 solver 的其余参数

    X = np.abs(rng.randn(6, 5))
    # 生成一个 6x5 的随机数组，然后取绝对值，确保数据非负

    assert pnmf.fit(X).reconstruction_err_ < 0.1
    # 断言：拟合后的重建误差小于 0.1


def test_nmf_true_reconstruction():
    # Test that the fit is not too far away from an exact solution
    # (by construction)
    # 测试拟合结果与精确解之间的误差不会太大（通过构造）

    n_samples = 15
    n_features = 10
    n_components = 5
    beta_loss = 1
    batch_size = 3
    max_iter = 1000

    rng = np.random.mtrand.RandomState(42)
    # 创建一个随机数生成器实例 rng，种子为 42

    W_true = np.zeros([n_samples, n_components])
    W_array = np.abs(rng.randn(n_samples))
    for j in range(n_components):
        W_true[j % n_samples, j] = W_array[j % n_samples]
    # 构造真实的矩阵 W_true，保证其非负

    H_true = np.zeros([n_components, n_features])
    H_array = np.abs(rng.randn(n_components))
    for j in range(n_features):
        H_true[j % n_components, j] = H_array[j % n_components]
    # 构造真实的矩阵 H_true，保证其非负

    X = np.dot(W_true, H_true)
    # 生成矩阵 X，为 W_true 和 H_true 的乘积

    model = NMF(
        n_components=n_components,
        solver="mu",
        beta_loss=beta_loss,
        max_iter=max_iter,
        random_state=0,
    )
    # 初始化一个 NMF 模型实例 model，设置参数 n_components=n_components, solver="mu", beta_loss=beta_loss, max_iter=max_iter, random_state=0

    transf = model.fit_transform(X)
    # 对数据 X 进行拟合并转换

    X_calc = np.dot(transf, model.components_)
    # 计算重建的矩阵 X_calc

    assert model.reconstruction_err_ < 0.1
    # 断言：重建误差小于 0.1

    assert_allclose(X, X_calc)
    # 断言：X 与 X_calc 在所有元素上都非常接近
    # 使用 MiniBatchNMF 进行非负矩阵分解，设置参数如下：
    # - n_components: 分解后的成分数量
    # - beta_loss: 损失函数类型
    # - batch_size: 小批量处理的批大小
    # - random_state: 随机数生成器的种子，用于重现结果
    # - max_iter: 最大迭代次数
    mbmodel = MiniBatchNMF(
        n_components=n_components,
        beta_loss=beta_loss,
        batch_size=batch_size,
        random_state=0,
        max_iter=max_iter,
    )

    # 使用拟合后的模型对输入数据 X 进行变换，得到转换后的结果
    transf = mbmodel.fit_transform(X)

    # 根据转换后的结果和模型的成分，重构原始数据 X_calc
    X_calc = np.dot(transf, mbmodel.components_)

    # 断言重构误差小于 0.1，如果不满足则抛出异常
    assert mbmodel.reconstruction_err_ < 0.1
    
    # 断言 X 和 X_calc 在指定的容差下（atol=1）非常接近，否则抛出异常
    assert_allclose(X, X_calc, atol=1)
@pytest.mark.parametrize("solver", ["cd", "mu"])
# 使用参数化测试来测试不同的求解器（cd和mu）
def test_nmf_transform(solver):
    # 测试 fit_transform 和 fit.transform 在 NMF 中的等效性
    # 测试 NMF.transform 返回的数值是否接近
    rng = np.random.mtrand.RandomState(42)
    # 生成随机数种子
    A = np.abs(rng.randn(6, 5))
    # 创建一个6x5的随机矩阵 A，并取其绝对值
    m = NMF(
        solver=solver,
        n_components=3,
        init="random",
        random_state=0,
        tol=1e-6,
    )
    # 使用给定的求解器、成分数、随机初始化、随机种子、容忍度初始化 NMF 模型 m
    ft = m.fit_transform(A)
    # 对 A 进行拟合转换，得到 ft
    t = m.transform(A)
    # 对 A 进行转换，得到 t
    assert_allclose(ft, t, atol=1e-1)
    # 使用 assert_allclose 断言 ft 和 t 的接近度，允许的误差为 1e-1


def test_minibatch_nmf_transform():
    # 测试 MiniBatchNMF 中 fit_transform 和 fit.transform 的等效性
    # 仅在进行新的重新启动时才保证
    rng = np.random.mtrand.RandomState(42)
    # 生成随机数种子
    A = np.abs(rng.randn(6, 5))
    # 创建一个6x5的随机矩阵 A，并取其绝对值
    m = MiniBatchNMF(
        n_components=3,
        random_state=0,
        tol=1e-3,
        fresh_restarts=True,
    )
    # 使用给定的成分数、随机种子、容忍度和新的重新启动初始化 MiniBatchNMF 模型 m
    ft = m.fit_transform(A)
    # 对 A 进行拟合转换，得到 ft
    t = m.transform(A)
    # 对 A 进行转换，得到 t
    assert_allclose(ft, t)
    # 使用 assert_allclose 断言 ft 和 t 的接近度


@pytest.mark.parametrize(
    ["Estimator", "solver"],
    [[NMF, {"solver": "cd"}], [NMF, {"solver": "mu"}], [MiniBatchNMF, {}]],
)
# 使用参数化测试来测试不同的估计器（NMF和MiniBatchNMF）和求解器（cd和mu）
def test_nmf_transform_custom_init(Estimator, solver):
    # 烟雾测试，检查是否 NMF.transform 能够与自定义初始化一起工作
    random_state = np.random.RandomState(0)
    # 生成随机数种子
    A = np.abs(random_state.randn(6, 5))
    # 创建一个6x5的随机矩阵 A，并取其绝对值
    n_components = 4
    avg = np.sqrt(A.mean() / n_components)
    # 计算 A 的均值的平方根除以成分数
    H_init = np.abs(avg * random_state.randn(n_components, 5))
    # 使用给定的均值初始化 H_init
    W_init = np.abs(avg * random_state.randn(6, n_components))
    # 使用给定的均值初始化 W_init

    m = Estimator(
        n_components=n_components, init="custom", random_state=0, tol=1e-3, **solver
    )
    # 使用给定的成分数、自定义初始化、随机种子、容忍度和求解器初始化估计器 m
    m.fit_transform(A, W=W_init, H=H_init)
    # 对 A 进行拟合转换，使用给定的 W_init 和 H_init
    m.transform(A)
    # 对 A 进行转换


@pytest.mark.parametrize("solver", ("cd", "mu"))
# 使用参数化测试来测试不同的求解器（cd和mu）
def test_nmf_inverse_transform(solver):
    # 测试 NMF.inverse_transform 返回的数值是否接近
    random_state = np.random.RandomState(0)
    # 生成随机数种子
    A = np.abs(random_state.randn(6, 4))
    # 创建一个6x4的随机矩阵 A，并取其绝对值
    m = NMF(
        solver=solver,
        n_components=4,
        init="random",
        random_state=0,
        max_iter=1000,
    )
    # 使用给定的求解器、成分数、随机初始化、随机种子、最大迭代次数初始化 NMF 模型 m
    ft = m.fit_transform(A)
    # 对 A 进行拟合转换，得到 ft
    A_new = m.inverse_transform(ft)
    # 对 ft 进行逆转换，得到 A_new
    assert_array_almost_equal(A, A_new, decimal=2)
    # 使用 assert_array_almost_equal 断言 A 和 A_new 的接近度，精确度为 2 位小数


# TODO(1.6): remove the warning filter
@pytest.mark.filterwarnings("ignore:The default value of `n_components` will change")
# 忽略警告："n_components" 的默认值将会改变
def test_mbnmf_inverse_transform():
    # 测试 MiniBatchNMF.transform 后接 MiniBatchNMF.inverse_transform
    # 结果接近于单位矩阵
    rng = np.random.RandomState(0)
    # 生成随机数种子
    A = np.abs(rng.randn(6, 4))
    # 创建一个6x4的随机矩阵 A，并取其绝对值
    nmf = MiniBatchNMF(
        random_state=rng,
        max_iter=500,
        init="nndsvdar",
        fresh_restarts=True,
    )
    # 使用给定的随机种子、最大迭代次数、初始化和新的重新启动初始化 MiniBatchNMF 模型 nmf
    ft = nmf.fit_transform(A)
    # 对 A 进行拟合转换，得到 ft
    A_new = nmf.inverse_transform(ft)
    # 对 ft 进行逆转换，得到 A_new
    assert_allclose(A, A_new, rtol=1e-3, atol=1e-2)
    # 使用 assert_allclose 断言 A 和 A_new 的接近度，相对误差和绝对误差分别为 1e-3 和 1e-2


@pytest.mark.parametrize("Estimator", [NMF, MiniBatchNMF])
# 使用参数化测试来测试不同的估计器（NMF和MiniBatchNMF）
def test_n_components_greater_n_features(Estimator):
    # 烟雾测试，检查当成分数大于特征数时的情况
    rng = np.random.mtrand.RandomState(42)
    # 生成随机数种子
    A = np.abs(rng.randn(30, 10))
    # 创建一个30x10的随机矩阵 A，并取其绝对值
    # 使用Estimator类创建一个实例，设置参数n_components为15，随机数种子为0，收敛容差(tol)为1e-2，
    # 然后调用fit方法，将数据集A用于拟合模型。
    Estimator(n_components=15, random_state=0, tol=1e-2).fit(A)
@pytest.mark.parametrize(
    ["Estimator", "solver"],
    [[NMF, {"solver": "cd"}], [NMF, {"solver": "mu"}], [MiniBatchNMF, {}]],
)
@pytest.mark.parametrize("sparse_container", CSC_CONTAINERS + CSR_CONTAINERS)
@pytest.mark.parametrize("alpha_W", (0.0, 1.0))
@pytest.mark.parametrize("alpha_H", (0.0, 1.0, "same"))
def test_nmf_sparse_input(Estimator, solver, sparse_container, alpha_W, alpha_H):
    # Test that sparse matrices are accepted as input

    # 创建随机数生成器，并生成一个10x10的非负随机矩阵A
    rng = np.random.mtrand.RandomState(42)
    A = np.abs(rng.randn(10, 10))
    # 将A的每一列中的偶数列置零
    A[:, 2 * np.arange(5)] = 0
    # 使用给定的稀疏容器将A转换为稀疏矩阵A_sparse
    A_sparse = sparse_container(A)

    # 使用给定的参数创建第一个估计器对象est1
    est1 = Estimator(
        n_components=5,
        init="random",
        alpha_W=alpha_W,
        alpha_H=alpha_H,
        random_state=0,
        tol=0,
        max_iter=100,
        **solver,
    )
    # 克隆第一个估计器对象，创建第二个估计器对象est2
    est2 = clone(est1)

    # 使用est1拟合A，得到W1和H1
    W1 = est1.fit_transform(A)
    H1 = est1.components_
    # 使用est2拟合A_sparse，得到W2和H2
    W2 = est2.fit_transform(A_sparse)
    H2 = est2.components_

    # 断言W1和W2以及H1和H2在数值上相近
    assert_allclose(W1, W2)
    assert_allclose(H1, H2)


@pytest.mark.parametrize(
    ["Estimator", "solver"],
    [[NMF, {"solver": "cd"}], [NMF, {"solver": "mu"}], [MiniBatchNMF, {}]],
)
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_nmf_sparse_transform(Estimator, solver, csc_container):
    # Test that transform works on sparse data.  Issue #2124

    # 创建随机数生成器，并生成一个3x2的非负随机矩阵A
    rng = np.random.mtrand.RandomState(42)
    A = np.abs(rng.randn(3, 2))
    # 将A的索引为(1, 1)的元素置零
    A[1, 1] = 0
    # 使用给定的稀疏容器将A转换为稀疏矩阵A
    A = csc_container(A)

    # 使用给定的参数创建NMF模型对象model
    model = Estimator(random_state=0, n_components=2, max_iter=400, **solver)
    # 在A上拟合并变换得到A_fit_tr
    A_fit_tr = model.fit_transform(A)
    # 在A上仅进行变换得到A_tr
    A_tr = model.transform(A)

    # 断言A_fit_tr和A_tr在数值上相近，容差为1e-1
    assert_allclose(A_fit_tr, A_tr, atol=1e-1)


# TODO(1.6): remove the warning filter
@pytest.mark.filterwarnings("ignore:The default value of `n_components` will change")
@pytest.mark.parametrize("init", ["random", "nndsvd"])
@pytest.mark.parametrize("solver", ("cd", "mu"))
@pytest.mark.parametrize("alpha_W", (0.0, 1.0))
@pytest.mark.parametrize("alpha_H", (0.0, 1.0, "same"))
def test_non_negative_factorization_consistency(init, solver, alpha_W, alpha_H):
    # Test that the function is called in the same way, either directly
    # or through the NMF class

    # 设置最大迭代次数和随机数生成器
    max_iter = 500
    rng = np.random.mtrand.RandomState(42)
    # 生成一个10x10的非负随机矩阵A
    A = np.abs(rng.randn(10, 10))
    # 将A的每一列中的偶数列置零
    A[:, 2 * np.arange(5)] = 0

    # 使用非负矩阵分解函数non_negative_factorization拟合A，返回W_nmf和H
    W_nmf, H, _ = non_negative_factorization(
        A,
        init=init,
        solver=solver,
        max_iter=max_iter,
        alpha_W=alpha_W,
        alpha_H=alpha_H,
        random_state=1,
        tol=1e-2,
    )
    # 使用非负矩阵分解函数non_negative_factorization拟合A，指定H，返回W_nmf_2和H
    W_nmf_2, H, _ = non_negative_factorization(
        A,
        H=H,
        update_H=False,
        init=init,
        solver=solver,
        max_iter=max_iter,
        alpha_W=alpha_W,
        alpha_H=alpha_H,
        random_state=1,
        tol=1e-2,
    )

    # 使用NMF类创建模型对象model_class
    model_class = NMF(
        init=init,
        solver=solver,
        max_iter=max_iter,
        alpha_W=alpha_W,
        alpha_H=alpha_H,
        random_state=1,
        tol=1e-2,
    )
    # 使用model_class拟合A，得到W_cls
    W_cls = model_class.fit_transform(A)
    # 使用模型对象 model_class 对输入数据 A 进行转换，生成 W_cls_2
    W_cls_2 = model_class.transform(A)

    # 断言检查 W_nmf 和 W_cls 是否近似相等
    assert_allclose(W_nmf, W_cls)
    
    # 断言检查 W_nmf_2 和 W_cls_2 是否近似相等
    assert_allclose(W_nmf_2, W_cls_2)
def test_non_negative_factorization_checking():
    # Note that the validity of parameter types and range of possible values
    # for scalar numerical or str parameters is already checked in the common
    # tests. Here we only check for problems that cannot be captured by simple
    # declarative constraints on the valid parameter values.

    A = np.ones((2, 2))
    # 创建一个2x2的全1数组A作为测试数据
    nnmf = non_negative_factorization
    # 将non_negative_factorization函数引用赋值给nnmf变量
    msg = re.escape("Negative values in data passed to NMF (input H)")
    # 设置匹配的错误消息，用于检测是否抛出指定的ValueError异常
    with pytest.raises(ValueError, match=msg):
        # 使用pytest模块的raises函数检测是否抛出ValueError异常，并匹配指定的错误消息
        nnmf(A, A, -A, 2, init="custom")
        # 调用nnmf函数，传入参数A, A, -A, 2, init="custom"

    msg = re.escape("Negative values in data passed to NMF (input W)")
    # 设置匹配的错误消息，用于检测是否抛出指定的ValueError异常
    with pytest.raises(ValueError, match=msg):
        # 使用pytest模块的raises函数检测是否抛出ValueError异常，并匹配指定的错误消息
        nnmf(A, -A, A, 2, init="custom")
        # 调用nnmf函数，传入参数A, -A, A, 2, init="custom"

    msg = re.escape("Array passed to NMF (input H) is full of zeros")
    # 设置匹配的错误消息，用于检测是否抛出指定的ValueError异常
    with pytest.raises(ValueError, match=msg):
        # 使用pytest模块的raises函数检测是否抛出ValueError异常，并匹配指定的错误消息
        nnmf(A, A, 0 * A, 2, init="custom")
        # 调用nnmf函数，传入参数A, A, 0 * A, 2, init="custom"


def _beta_divergence_dense(X, W, H, beta):
    """Compute the beta-divergence of X and W.H for dense array only.

    Used as a reference for testing nmf._beta_divergence.
    """
    WH = np.dot(W, H)
    # 计算矩阵W和H的乘积，赋值给WH变量

    if beta == 2:
        # 如果beta等于2
        return squared_norm(X - WH) / 2
        # 返回(X - WH)的平方范数除以2的结果作为beta=2时的beta-divergence值

    WH_Xnonzero = WH[X != 0]
    # 从WH中提取X非零位置对应的元素，赋值给WH_Xnonzero变量
    X_nonzero = X[X != 0]
    # 提取X中非零元素，赋值给X_nonzero变量
    np.maximum(WH_Xnonzero, 1e-9, out=WH_Xnonzero)
    # 将WH_Xnonzero中的每个元素与1e-9比较取较大值，结果写回WH_Xnonzero数组

    if beta == 1:
        # 如果beta等于1
        res = np.sum(X_nonzero * np.log(X_nonzero / WH_Xnonzero))
        # 计算beta=1时的beta-divergence值
        res += WH.sum() - X.sum()
        # 累加WH和X的总和之差到res变量

    elif beta == 0:
        # 如果beta等于0
        div = X_nonzero / WH_Xnonzero
        # 计算X_nonzero与WH_Xnonzero的元素除法结果，赋值给div变量
        res = np.sum(div) - X.size - np.sum(np.log(div))
        # 计算beta=0时的beta-divergence值，并赋给res变量
    else:
        # 对于其他的beta值
        res = (X_nonzero**beta).sum()
        # 计算(X_nonzero的每个元素的beta次方)的总和，赋给res变量
        res += (beta - 1) * (WH**beta).sum()
        # 计算(WH的每个元素的beta次方)的总和，乘以(beta-1)后累加到res变量
        res -= beta * (X_nonzero * (WH_Xnonzero ** (beta - 1))).sum()
        # 计算(X_nonzero与WH_Xnonzero的元素乘积)的总和，乘以beta后累减到res变量
        res /= beta * (beta - 1)
        # 将res除以beta和(beta-1)的乘积，更新res变量为最终的beta-divergence值

    return res
    # 返回计算得到的beta-divergence值


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_beta_divergence(csr_container):
    # Compare _beta_divergence with the reference _beta_divergence_dense
    n_samples = 20
    n_features = 10
    n_components = 5
    beta_losses = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

    # initialization
    rng = np.random.mtrand.RandomState(42)
    # 创建随机数生成器rng，种子为42
    X = rng.randn(n_samples, n_features)
    # 生成n_samples x n_features大小的标准正态分布随机数组X
    np.clip(X, 0, None, out=X)
    # 将X中的所有元素限制在0到正无穷之间，并写回X
    X_csr = csr_container(X)
    # 使用csr_container函数将X转换为稀疏矩阵表示，赋给X_csr变量
    W, H = nmf._initialize_nmf(X, n_components, init="random", random_state=42)
    # 调用nmf._initialize_nmf函数初始化NMF模型，得到W和H矩阵

    for beta in beta_losses:
        # 遍历每个beta值
        ref = _beta_divergence_dense(X, W, H, beta)
        # 调用_beta_divergence_dense函数计算稠密情况下的beta-divergence值，作为参考值
        loss = nmf._beta_divergence(X, W, H, beta)
        # 调用nmf._beta_divergence函数计算当前beta值下的beta-divergence值
        loss_csr = nmf._beta_divergence(X_csr, W, H, beta)
        # 调用nmf._beta_divergence函数计算当前beta值下稀疏表示的beta-divergence值

        assert_almost_equal(ref, loss, decimal=7)
        # 断言参考值和当前计算值在小数点后7位精度上近似相等
        assert_almost_equal(ref, loss_csr, decimal=7)
        # 断言参考值和稀疏表示的计算值在小数点后7位精度上近似相等


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_special_sparse_dot(csr_container):
    # Test the function that computes np.dot(W, H), only where X is non zero.
    n_samples = 10
    n_features = 5
    n_components = 3
    rng = np.random.mtrand.RandomState(42)
    # 创建随机数生成器rng，种子为42
    X = rng.randn(n_samples, n_features)
    # 生成n_samples x n_features大小的标准正态分布随机数组X
    np.clip(X, 0, None, out=X)
    # 将X中的所有元素限制在0到正无穷之间，并写回X
    X_csr = csr_container(X)
    # 使用csr_container函数将X转换为稀疏矩阵表示，赋给X_csr变量
    # 使用 NumPy 的随机数生成器 rng 生成一个形状为 (n_samples, n_components) 的非负随机矩阵 W
    W = np.abs(rng.randn(n_samples, n_components))
    # 使用 NumPy 的随机数生成器 rng 生成一个形状为 (n_components, n_features) 的非负随机矩阵 H
    H = np.abs(rng.randn(n_components, n_features))

    # 使用 nmf 模块中的 _special_sparse_dot 函数计算稀疏矩阵 W 和 H 与稀疏矩阵 X_csr 的乘积，存储在 WH_safe 中
    WH_safe = nmf._special_sparse_dot(W, H, X_csr)
    # 使用 nmf 模块中的 _special_sparse_dot 函数计算稀疏矩阵 W 和 H 与稠密矩阵 X 的乘积，存储在 WH 中
    WH = nmf._special_sparse_dot(W, H, X)

    # 检验 WH_safe 和 WH 在 X_csr 中非零元素位置的数值是否近似相等，精度为 10 位小数
    ii, jj = X_csr.nonzero()
    WH_safe_data = np.asarray(WH_safe[ii, jj]).ravel()
    assert_array_almost_equal(WH_safe_data, WH[ii, jj], decimal=10)

    # 检验 WH_safe 的稀疏结构是否与 X_csr 相同：比较它们的索引、指针和形状
    assert_array_equal(WH_safe.indices, X_csr.indices)
    assert_array_equal(WH_safe.indptr, X_csr.indptr)
    assert_array_equal(WH_safe.shape, X_csr.shape)
# 忽略收敛警告，并对给定的 CSR_CONTAINERS 参数化测试
@ignore_warnings(category=ConvergenceWarning)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_nmf_multiplicative_update_sparse(csr_container):
    # 比较稀疏和密集输入在乘法更新的非负矩阵分解 (NMF) 中的效果
    # 同时测试结果在 beta_loss 参数变化时的连续性

    # 定义样本数、特征数、分量数、正则化参数、L1 比例和迭代次数
    n_samples = 20
    n_features = 10
    n_components = 5
    alpha = 0.1
    l1_ratio = 0.5
    n_iter = 20

    # 初始化随机数生成器
    rng = np.random.mtrand.RandomState(1337)
    # 生成随机正态分布的矩阵 X，并取其绝对值
    X = rng.randn(n_samples, n_features)
    X = np.abs(X)
    # 将 X 转换成 CSR 格式
    X_csr = csr_container(X)
    # 使用 nmf._initialize_nmf 初始化 NMF，得到初始的 W 和 H
    W0, H0 = nmf._initialize_nmf(X, n_components, init="random", random_state=42)

    # 在不同的 beta_loss 值上进行循环测试
    for beta_loss in (-1.2, 0, 0.2, 1.0, 2.0, 2.5):
        # 使用非负矩阵分解 (NMF) 进行计算，参照稠密数组 X
        W, H = W0.copy(), H0.copy()
        W1, H1, _ = non_negative_factorization(
            X,
            W,
            H,
            n_components,
            init="custom",
            update_H=True,
            solver="mu",
            beta_loss=beta_loss,
            max_iter=n_iter,
            alpha_W=alpha,
            l1_ratio=l1_ratio,
            random_state=42,
        )

        # 使用稀疏矩阵 X_csr 进行计算
        W, H = W0.copy(), H0.copy()
        W2, H2, _ = non_negative_factorization(
            X_csr,
            W,
            H,
            n_components,
            init="custom",
            update_H=True,
            solver="mu",
            beta_loss=beta_loss,
            max_iter=n_iter,
            alpha_W=alpha,
            l1_ratio=l1_ratio,
            random_state=42,
        )

        # 断言稠密数组和稀疏数组的结果在给定的容差下相似
        assert_allclose(W1, W2, atol=1e-7)
        assert_allclose(H1, H2, atol=1e-7)

        # 在接近相同的 beta_loss 值上进行测试，确保结果在给定的容差下仍然连续
        beta_loss -= 1.0e-5
        W, H = W0.copy(), H0.copy()
        W3, H3, _ = non_negative_factorization(
            X_csr,
            W,
            H,
            n_components,
            init="custom",
            update_H=True,
            solver="mu",
            beta_loss=beta_loss,
            max_iter=n_iter,
            alpha_W=alpha,
            l1_ratio=l1_ratio,
            random_state=42,
        )

        # 断言在接近的 beta_loss 值上，结果在更大的容差下也相似
        assert_allclose(W1, W3, atol=1e-4)
        assert_allclose(H1, H3, atol=1e-4)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_nmf_negative_beta_loss(csr_container):
    # 测试当 beta_loss < 0 且输入 X 包含零时是否引发错误
    # 测试输入包含零时，输出是否不包含 NaN 值

    # 定义样本数、特征数和分量数
    n_samples = 6
    n_features = 5
    n_components = 3

    # 初始化随机数生成器
    rng = np.random.mtrand.RandomState(42)
    # 生成随机正态分布的矩阵 X，并将其截断为非负数
    X = rng.randn(n_samples, n_features)
    np.clip(X, 0, None, out=X)
    # 将 X 转换成 CSR 格式
    X_csr = csr_container(X)
    # 定义一个内部函数，用于验证非负矩阵分解结果中是否包含 NaN，确保结果符合预期
    def _assert_nmf_no_nan(X, beta_loss):
        # 执行非负矩阵分解，返回分解后的矩阵 W、H 和重建误差
        W, H, _ = non_negative_factorization(
            X,
            init="random",  # 使用随机初始化方式
            n_components=n_components,  # 指定分解的组件数目
            solver="mu",  # 使用乘法更新法作为求解器
            beta_loss=beta_loss,  # 损失函数的 beta 参数
            random_state=0,  # 设置随机数生成器的种子以保证可重现性
            max_iter=1000,  # 设置最大迭代次数
        )
        # 断言矩阵 W 中不包含 NaN 值
        assert not np.any(np.isnan(W))
        # 断言矩阵 H 中不包含 NaN 值
        assert not np.any(np.isnan(H))

    # 当 beta_loss <= 0 且 X 包含零值时，求解器可能会发散，给出相应的错误信息
    msg = "When beta_loss <= 0 and X contains zeros, the solver may diverge."
    # 对于每个 beta_loss 的取值，执行以下操作
    for beta_loss in (-0.6, 0.0):
        # 使用 pytest 来断言在给定条件下会抛出 ValueError，并匹配特定的错误信息
        with pytest.raises(ValueError, match=msg):
            _assert_nmf_no_nan(X, beta_loss)
        # 在 X 上加上一个极小值，以避免零值并重新验证非负矩阵分解结果
        _assert_nmf_no_nan(X + 1e-9, beta_loss)

    # 对于其他 beta_loss 的取值，直接验证非负矩阵分解结果
    for beta_loss in (0.2, 1.0, 1.2, 2.0, 2.5):
        _assert_nmf_no_nan(X, beta_loss)
        _assert_nmf_no_nan(X_csr, beta_loss)
# TODO(1.6): remove the warning filter
# 使用 pytest 的 mark 来忽略特定的警告消息
@pytest.mark.filterwarnings("ignore:The default value of `n_components` will change")
# 使用 pytest 的 parametrize 装饰器来定义多组参数化测试
@pytest.mark.parametrize("beta_loss", [-0.5, 0.0])
def test_minibatch_nmf_negative_beta_loss(beta_loss):
    """Check that an error is raised if beta_loss < 0 and X contains zeros."""
    # 设定随机数生成器的种子，以便结果可重复
    rng = np.random.RandomState(0)
    # 生成一个 6x5 的正态分布矩阵 X，并将小于 0 的元素置为 0
    X = rng.normal(size=(6, 5))
    X[X < 0] = 0

    # 创建 MiniBatchNMF 对象，指定 beta_loss 参数和随机种子
    nmf = MiniBatchNMF(beta_loss=beta_loss, random_state=0)

    # 当期望引发 ValueError 异常，并包含特定消息时，测试通过
    msg = "When beta_loss <= 0 and X contains zeros, the solver may diverge."
    with pytest.raises(ValueError, match=msg):
        nmf.fit(X)


@pytest.mark.parametrize(
    ["Estimator", "solver"],
    [[NMF, {"solver": "cd"}], [NMF, {"solver": "mu"}], [MiniBatchNMF, {}]],
)
def test_nmf_regularization(Estimator, solver):
    # Test the effect of L1 and L2 regularizations
    n_samples = 6
    n_features = 5
    n_components = 3
    # 设定随机数生成器的种子
    rng = np.random.mtrand.RandomState(42)
    # 生成一个大小为 (n_samples, n_features) 的非负随机矩阵 X
    X = np.abs(rng.randn(n_samples, n_features))

    # 创建正则化 Estimator 对象，使用指定的参数和 solver
    # 这里测试 L1 正则化效果，alpha_W 设置为 0.5，l1_ratio 设置为 1.0
    l1_ratio = 1.0
    regul = Estimator(
        n_components=n_components,
        alpha_W=0.5,
        l1_ratio=l1_ratio,
        random_state=42,
        **solver,
    )
    # 创建模型 Estimator 对象，不使用正则化，alpha_W 设置为 0.0，l1_ratio 也设置为 1.0
    model = Estimator(
        n_components=n_components,
        alpha_W=0.0,
        l1_ratio=l1_ratio,
        random_state=42,
        **solver,
    )

    # 使用正则化对象拟合数据 X，返回变换后的矩阵 W
    W_regul = regul.fit_transform(X)
    # 使用模型对象拟合数据 X，返回变换后的矩阵 W
    W_model = model.fit_transform(X)

    # 获取正则化对象的成分矩阵 H
    H_regul = regul.components_
    # 获取模型对象的成分矩阵 H
    H_model = model.components_

    # 计算非零元素的个数，用于后续断言比较
    eps = np.finfo(np.float64).eps
    W_regul_n_zeros = W_regul[W_regul <= eps].size
    W_model_n_zeros = W_model[W_model <= eps].size
    H_regul_n_zeros = H_regul[H_regul <= eps].size
    H_model_n_zeros = H_model[H_model <= eps].size

    # 断言：正则化后的非零元素个数应该比模型少
    assert W_regul_n_zeros > W_model_n_zeros
    assert H_regul_n_zeros > H_model_n_zeros

    # 测试 L2 正则化效果，alpha_W 设置为 0.5，l1_ratio 设置为 0.0
    l1_ratio = 0.0
    regul = Estimator(
        n_components=n_components,
        alpha_W=0.5,
        l1_ratio=l1_ratio,
        random_state=42,
        **solver,
    )
    model = Estimator(
        n_components=n_components,
        alpha_W=0.0,
        l1_ratio=l1_ratio,
        random_state=42,
        **solver,
    )

    # 再次拟合数据 X，返回变换后的矩阵 W
    W_regul = regul.fit_transform(X)
    # 再次拟合数据 X，返回变换后的矩阵 W
    W_model = model.fit_transform(X)

    # 获取正则化对象的成分矩阵 H
    H_regul = regul.components_
    # 获取模型对象的成分矩阵 H
    H_model = model.components_

    # 断言：使用 L2 正则化后，矩阵 W 和 H 的平方范数之和应该减少
    assert (linalg.norm(W_model)) ** 2.0 + (linalg.norm(H_model)) ** 2.0 > (
        linalg.norm(W_regul)
    ) ** 2.0 + (linalg.norm(H_regul)) ** 2.0


@ignore_warnings(category=ConvergenceWarning)
# 使用 pytest 的 parametrize 装饰器定义多组参数化测试
@pytest.mark.parametrize("solver", ("cd", "mu"))
def test_nmf_decreasing(solver):
    # test that the objective function is decreasing at each iteration
    n_samples = 20
    n_features = 15
    n_components = 10
    alpha = 0.1
    l1_ratio = 0.5
    tol = 0.0

    # initialization
    # 设定随机数生成器的种子
    rng = np.random.mtrand.RandomState(42)
    # 生成一个大小为 (n_samples, n_features) 的随机矩阵 X，并取其绝对值
    X = rng.randn(n_samples, n_features)
    np.abs(X, X)  # 将 X 中的每个元素取绝对值，更新到 X 中
    # 使用非负矩阵分解初始化 W0 和 H0，采用随机初始化方式，指定随机种子为 42
    W0, H0 = nmf._initialize_nmf(X, n_components, init="random", random_state=42)

    # 针对不同的 beta_loss 值进行迭代，依次为 (-1.2, 0, 0.2, 1.0, 2.0, 2.5)
    for beta_loss in (-1.2, 0, 0.2, 1.0, 2.0, 2.5):
        # 如果求解器不是 "mu" 并且 beta_loss 不等于 2，则跳过当前循环
        if solver != "mu" and beta_loss != 2:
            # not implemented
            continue
        
        # 复制 W0 和 H0 到 W 和 H，作为当前迭代的起始点
        W, H = W0.copy(), H0.copy()
        previous_loss = None
        
        # 执行 30 次迭代
        for _ in range(30):
            # 使用非负矩阵分解进行一次迭代，从上一次的结果开始
            W, H, _ = non_negative_factorization(
                X,
                W,
                H,
                beta_loss=beta_loss,
                init="custom",
                n_components=n_components,
                max_iter=1,
                alpha_W=alpha,
                solver=solver,
                tol=tol,
                l1_ratio=l1_ratio,
                verbose=0,
                random_state=0,
                update_H=True,
            )

            # 计算当前损失函数值，包括 beta 分歧度项和正则化项
            loss = (
                nmf._beta_divergence(X, W, H, beta_loss)
                + alpha * l1_ratio * n_features * W.sum()
                + alpha * l1_ratio * n_samples * H.sum()
                + alpha * (1 - l1_ratio) * n_features * (W**2).sum()
                + alpha * (1 - l1_ratio) * n_samples * (H**2).sum()
            )
            
            # 如果之前有记录上一次的损失函数值，断言当前损失值应小于之前的损失值
            if previous_loss is not None:
                assert previous_loss > loss
            previous_loss = loss
# 定义用于测试 NMF 下溢问题的函数
def test_nmf_underflow():
    # 使用种子值 0 创建随机数生成器
    rng = np.random.RandomState(0)
    # 定义样本数、特征数和分量数
    n_samples, n_features, n_components = 10, 2, 2
    # 生成随机矩阵 X，并取其绝对值后乘以 10
    X = np.abs(rng.randn(n_samples, n_features)) * 10
    # 生成随机矩阵 W，并取其绝对值后乘以 10
    W = np.abs(rng.randn(n_samples, n_components)) * 10
    # 生成随机矩阵 H
    H = np.abs(rng.randn(n_components, n_features))

    # 修改 X 的第一个元素为 0，用于测试下溢情况
    X[0, 0] = 0
    # 计算使用修改后的 X 调用 _beta_divergence 的结果
    ref = nmf._beta_divergence(X, W, H, beta=1.0)
    # 将 X 的第一个元素修改为极小值，再次计算 _beta_divergence 的结果
    X[0, 0] = 1e-323
    res = nmf._beta_divergence(X, W, H, beta=1.0)
    # 断言结果 res 应近似于 ref
    assert_almost_equal(res, ref)


# TODO(1.6): 移除警告过滤器
@pytest.mark.filterwarnings("ignore:The default value of `n_components` will change")
# 参数化测试，检查不同数据类型的输入和输出是否一致
@pytest.mark.parametrize(
    "dtype_in, dtype_out",
    [
        (np.float32, np.float32),
        (np.float64, np.float64),
        (np.int32, np.float64),
        (np.int64, np.float64),
    ],
)
# 参数化测试，检查不同的估计器和求解器组合
@pytest.mark.parametrize(
    ["Estimator", "solver"],
    [[NMF, {"solver": "cd"}], [NMF, {"solver": "mu"}], [MiniBatchNMF, {}]],
)
def test_nmf_dtype_match(Estimator, solver, dtype_in, dtype_out):
    # 检查 NMF 是否能保持数据类型 (float32 和 float64)
    X = np.random.RandomState(0).randn(20, 15).astype(dtype_in, copy=False)
    np.abs(X, out=X)

    # 创建 NMF 对象
    nmf = Estimator(
        alpha_W=1.0,
        alpha_H=1.0,
        tol=1e-2,
        random_state=0,
        **solver,
    )

    # 断言拟合后的转换和组件的数据类型与预期一致
    assert nmf.fit(X).transform(X).dtype == dtype_out
    assert nmf.fit_transform(X).dtype == dtype_out
    assert nmf.components_.dtype == dtype_out


# TODO(1.6): 移除警告过滤器
@pytest.mark.filterwarnings("ignore:The default value of `n_components` will change")
# 参数化测试，检查 float32 和 float64 下 NMF 的结果一致性
@pytest.mark.parametrize(
    ["Estimator", "solver"],
    [[NMF, {"solver": "cd"}], [NMF, {"solver": "mu"}], [MiniBatchNMF, {}]],
)
def test_nmf_float32_float64_consistency(Estimator, solver):
    # 检查 NMF 在 float32 和 float64 下的结果是否一致
    X = np.random.RandomState(0).randn(50, 7)
    np.abs(X, out=X)
    # 使用 float32 进行拟合和转换
    nmf32 = Estimator(random_state=0, tol=1e-3, **solver)
    W32 = nmf32.fit_transform(X.astype(np.float32))
    # 使用 float64 进行拟合和转换
    nmf64 = Estimator(random_state=0, tol=1e-3, **solver)
    W64 = nmf64.fit_transform(X)

    # 断言两种数据类型下得到的结果近似
    assert_allclose(W32, W64, atol=1e-5)


# TODO(1.6): 移除警告过滤器
@pytest.mark.filterwarnings("ignore:The default value of `n_components` will change")
# 参数化测试，检查自定义初始化时是否会引发数据类型错误
@pytest.mark.parametrize("Estimator", [NMF, MiniBatchNMF])
def test_nmf_custom_init_dtype_error(Estimator):
    # 检查当自定义 H 和/或 W 的数据类型与 X 不一致时是否会引发错误
    rng = np.random.RandomState(0)
    X = rng.random_sample((20, 15))
    H = rng.random_sample((15, 15)).astype(np.float32)
    W = rng.random_sample((20, 15))

    # 使用 pytest 断言预期的 TypeError 异常被引发
    with pytest.raises(TypeError, match="should have the same dtype as X"):
        Estimator(init="custom").fit(X, H=H, W=W)

    with pytest.raises(TypeError, match="should have the same dtype as X"):
        non_negative_factorization(X, H=H, update_H=False)
@pytest.mark.parametrize("beta_loss", [-0.5, 0, 0.5, 1, 1.5, 2, 2.5])
# 使用 pytest 的 parametrize 装饰器，用不同的 beta_loss 参数执行该测试函数
def test_nmf_minibatchnmf_equivalence(beta_loss):
    # 测试当 batch_size = n_samples 且 forget_factor = 0.0 时，MiniBatchNMF 是否等同于 NMF

    rng = np.random.mtrand.RandomState(42)
    # 使用种子为 42 的随机数生成器创建随机状态对象

    X = np.abs(rng.randn(48, 5))
    # 创建一个形状为 (48, 5) 的随机数数组 X，并取其绝对值

    nmf = NMF(
        n_components=5,
        beta_loss=beta_loss,
        solver="mu",
        random_state=0,
        tol=0,
    )
    # 创建一个 NMF 对象，设置参数包括 n_components, beta_loss, solver 等，并使用随机状态为 0

    mbnmf = MiniBatchNMF(
        n_components=5,
        beta_loss=beta_loss,
        random_state=0,
        tol=0,
        max_no_improvement=None,
        batch_size=X.shape[0],
        forget_factor=0.0,
    )
    # 创建一个 MiniBatchNMF 对象，设置参数包括 n_components, beta_loss, batch_size 等

    W = nmf.fit_transform(X)
    # 使用 NMF 对象拟合并变换 X，得到矩阵 W

    mbW = mbnmf.fit_transform(X)
    # 使用 MiniBatchNMF 对象拟合并变换 X，得到矩阵 mbW

    assert_allclose(W, mbW)
    # 断言矩阵 W 和 mbW 是否近似相等


def test_minibatch_nmf_partial_fit():
    # 检查 fit 和 partial_fit 的等价性。仅适用于新启动。

    rng = np.random.mtrand.RandomState(42)
    # 使用种子为 42 的随机数生成器创建随机状态对象

    X = np.abs(rng.randn(100, 5))
    # 创建一个形状为 (100, 5) 的随机数数组 X，并取其绝对值

    n_components = 5
    batch_size = 10
    max_iter = 2

    mbnmf1 = MiniBatchNMF(
        n_components=n_components,
        init="custom",
        random_state=0,
        max_iter=max_iter,
        batch_size=batch_size,
        tol=0,
        max_no_improvement=None,
        fresh_restarts=False,
    )
    # 创建一个 MiniBatchNMF 对象 mbnmf1，设置参数包括 n_components, init, max_iter 等

    mbnmf2 = MiniBatchNMF(n_components=n_components, init="custom", random_state=0)
    # 创建另一个 MiniBatchNMF 对象 mbnmf2，设置参数包括 n_components, init 等

    W, H = nmf._initialize_nmf(
        X, n_components=n_components, init="random", random_state=0
    )
    # 使用 NMF 对象的 _initialize_nmf 方法初始化矩阵 W 和 H

    mbnmf1.fit(X, W=W, H=H)
    # 使用 mbnmf1 对象拟合 X，传入 W 和 H

    for i in range(max_iter):
        for j in range(batch_size):
            mbnmf2.partial_fit(X[j : j + batch_size], W=W[:batch_size], H=H)
    # 循环进行部分拟合，使用 mbnmf2 对象的 partial_fit 方法

    assert mbnmf1.n_steps_ == mbnmf2.n_steps_
    # 断言 mbnmf1 和 mbnmf2 的步骤数是否相等

    assert_allclose(mbnmf1.components_, mbnmf2.components_)
    # 断言 mbnmf1 和 mbnmf2 的成分矩阵是否近似相等


def test_feature_names_out():
    """Check feature names out for NMF."""
    # 检查 NMF 的特征名称输出

    random_state = np.random.RandomState(0)
    # 使用种子为 0 的随机数生成器创建随机状态对象

    X = np.abs(random_state.randn(10, 4))
    # 创建一个形状为 (10, 4) 的随机数数组 X，并取其绝对值

    nmf = NMF(n_components=3).fit(X)
    # 创建一个 NMF 对象，设置 n_components=3 并拟合 X

    names = nmf.get_feature_names_out()
    # 调用 NMF 对象的 get_feature_names_out 方法获取特征名称

    assert_array_equal([f"nmf{i}" for i in range(3)], names)
    # 断言获取的特征名称是否符合预期


# TODO(1.6): remove the warning filter
@pytest.mark.filterwarnings("ignore:The default value of `n_components` will change")
# 使用 pytest 的 filterwarnings 装饰器忽略特定警告信息
def test_minibatch_nmf_verbose():
    # 检查 MiniBatchNMF 的详细模式以提高覆盖率。

    A = np.random.RandomState(0).random_sample((100, 10))
    # 使用种子为 0 的随机数生成器创建一个形状为 (100, 10) 的随机数数组 A

    nmf = MiniBatchNMF(tol=1e-2, random_state=0, verbose=1)
    # 创建一个 MiniBatchNMF 对象，设置参数包括 tol=1e-2, random_state=0, verbose=1

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        nmf.fit(A)
        # 使用 nmf 对象拟合数组 A
    finally:
        sys.stdout = old_stdout
        # 恢复标准输出流


# TODO(1.7): remove this test
@pytest.mark.parametrize("Estimator", [NMF, MiniBatchNMF])
# 使用 pytest 的 parametrize 装饰器，对 Estimator 参数化
def test_NMF_inverse_transform_Xt_deprecation(Estimator):
    rng = np.random.RandomState(42)
    # 使用种子为 42 的随机数生成器创建随机状态对象

    A = np.abs(rng.randn(6, 5))
    # 创建一个形状为 (6, 5) 的随机数数组 A，并取其绝对值

    est = Estimator(
        n_components=3,
        init="random",
        random_state=0,
        tol=1e-6,
    )
    # 创建一个 Estimator 对象，可能是 NMF 或 MiniBatchNMF，设置参数包括 n_components, init 等

    X = est.fit_transform(A)
    # 使用 Estimator 对象拟合并变换数组 A，得到变换后的矩阵 X
    # 使用 pytest 模块验证在调用 est.inverse_transform() 时是否会引发 TypeError 异常，并检查异常消息是否包含指定文本
    with pytest.raises(TypeError, match="Missing required positional argument"):
        est.inverse_transform()
    
    # 使用 pytest 模块验证在调用 est.inverse_transform() 时是否会引发 TypeError 异常，并检查异常消息是否包含指定文本，确保同时传递 X 和 Xt 会引发异常
    with pytest.raises(TypeError, match="Cannot use both X and Xt. Use X only"):
        est.inverse_transform(X=X, Xt=X)
    
    # 使用 warnings 模块捕获所有警告，并设置警告过滤器以将警告转换为错误
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("error")
        # 调用 est.inverse_transform()，预期会触发警告转换为错误的机制
        est.inverse_transform(X)
    
    # 使用 pytest 模块验证在调用 est.inverse_transform() 时是否会引发 FutureWarning 警告，并检查警告消息是否包含指定文本
    with pytest.warns(FutureWarning, match="Xt was renamed X in version 1.5"):
        est.inverse_transform(Xt=X)
@pytest.mark.parametrize("Estimator", [NMF, MiniBatchNMF])
# 使用 pytest 的参数化装饰器，定义测试函数参数为 NMF 和 MiniBatchNMF
def test_nmf_n_components_auto(Estimator):
    # 检查 n_components 是否从自定义初始化中正确推断出来
    rng = np.random.RandomState(0)
    # 使用种子 0 初始化随机数生成器
    X = rng.random_sample((6, 5))
    # 创建一个 6x5 的随机数矩阵 X
    W = rng.random_sample((6, 2))
    # 创建一个 6x2 的随机数矩阵 W
    H = rng.random_sample((2, 5))
    # 创建一个 2x5 的随机数矩阵 H
    est = Estimator(
        n_components="auto",
        init="custom",
        random_state=0,
        tol=1e-6,
    )
    # 使用 Estimator 类的自动 n_components、自定义初始化、种子 0 和容忍度 1e-6 创建估计器对象
    est.fit_transform(X, W=W, H=H)
    # 对 X 进行拟合变换，传入 W 和 H
    assert est._n_components == H.shape[0]
    # 断言估计器的 _n_components 属性等于 H 矩阵的行数


def test_nmf_non_negative_factorization_n_components_auto():
    # 检查从提供的自定义初始化中正确推断出 n_components
    rng = np.random.RandomState(0)
    # 使用种子 0 初始化随机数生成器
    X = rng.random_sample((6, 5))
    # 创建一个 6x5 的随机数矩阵 X
    W_init = rng.random_sample((6, 2))
    # 创建一个 6x2 的随机数矩阵 W_init
    H_init = rng.random_sample((2, 5))
    # 创建一个 2x5 的随机数矩阵 H_init
    W, H, _ = non_negative_factorization(
        X, W=W_init, H=H_init, init="custom", n_components="auto"
    )
    # 使用自定义初始化和 n_components="auto" 调用非负矩阵分解函数
    assert H.shape == H_init.shape
    # 断言输出的 H 矩阵形状与 H_init 相同
    assert W.shape == W_init.shape
    # 断言输出的 W 矩阵形状与 W_init 相同


# TODO(1.6): remove
# 待办事项标签，版本 1.6 时移除该函数
def test_nmf_n_components_default_value_warning():
    rng = np.random.RandomState(0)
    # 使用种子 0 初始化随机数生成器
    X = rng.random_sample((6, 5))
    # 创建一个 6x5 的随机数矩阵 X
    H = rng.random_sample((2, 5))
    # 创建一个 2x5 的随机数矩阵 H
    with pytest.warns(
        FutureWarning, match="The default value of `n_components` will change from"
    ):
        # 检查未来警告，匹配警告信息中的 n_components 默认值将发生变化
        non_negative_factorization(X, H=H)


def test_nmf_n_components_auto_no_h_update():
    # 测试非负矩阵分解在设置 n_components="auto" 时不会失败，同时验证推断出的 n_component 值是否正确
    rng = np.random.RandomState(0)
    # 使用种子 0 初始化随机数生成器
    X = rng.random_sample((6, 5))
    # 创建一个 6x5 的随机数矩阵 X
    H_true = rng.random_sample((2, 5))
    # 创建一个真实的 2x5 的随机数矩阵 H_true
    W, H, _ = non_negative_factorization(
        X, H=H_true, n_components="auto", update_H=False
    )  # 应该不会失败
    # 使用非负矩阵分解函数，设置 n_components="auto" 和 update_H=False
    assert_allclose(H, H_true)
    # 断言输出的 H 矩阵与 H_true 在数值上相近
    assert W.shape == (X.shape[0], H_true.shape[0])
    # 断言输出的 W 矩阵形状为 (X 的行数, H_true 的行数)


def test_nmf_w_h_not_used_warning():
    # 检查是否在提供的 W 和 H 未被使用时会引发警告，并且初始化会覆盖 W 或 H 的值
    rng = np.random.RandomState(0)
    # 使用种子 0 初始化随机数生成器
    X = rng.random_sample((6, 5))
    # 创建一个 6x5 的随机数矩阵 X
    W_init = rng.random_sample((6, 2))
    # 创建一个 6x2 的随机数矩阵 W_init
    H_init = rng.random_sample((2, 5))
    # 创建一个 2x5 的随机数矩阵 H_init
    with pytest.warns(
        RuntimeWarning,
        match="When init!='custom', provided W or H are ignored",
    ):
        # 检查运行时警告，匹配警告信息中的当 init!='custom' 时，提供的 W 或 H 会被忽略
        non_negative_factorization(X, H=H_init, update_H=True, n_components="auto")

    with pytest.warns(
        RuntimeWarning,
        match="When init!='custom', provided W or H are ignored",
    ):
        # 检查运行时警告，匹配警告信息中的当 init!='custom' 时，提供的 W 或 H 会被忽略
        non_negative_factorization(
            X, W=W_init, H=H_init, update_H=True, n_components="auto"
        )

    with pytest.warns(
        RuntimeWarning, match="When update_H=False, the provided initial W is not used."
    ):
        # 检查运行时警告，匹配警告信息中的当 update_H=False 时，提供的初始 W 不会被使用
        non_negative_factorization(X, W=W_init, H=H_init, update_H=False)
        # 当 update_H 参数为 False 时，不论 init 参数如何设置，都会忽略 W 参数
        # TODO: 在 init 参数为 "custom" 时，使用提供的 W 参数。
        non_negative_factorization(
            X, W=W_init, H=H_init, update_H=False, n_components="auto"
        )
# 定义一个测试函数，用于测试 NMF 类中自定义初始化时的形状错误提示
def test_nmf_custom_init_shape_error():
    # 创建一个随机数生成器对象，种子值为 0
    rng = np.random.RandomState(0)
    # 生成一个形状为 (6, 5) 的随机数矩阵 X
    X = rng.random_sample((6, 5))
    # 生成一个形状为 (2, 5) 的随机数矩阵 H
    H = rng.random_sample((2, 5))
    # 创建一个 NMF 对象，设置成 2 个成分，自定义初始化方式，并指定随机状态为 0
    nmf = NMF(n_components=2, init="custom", random_state=0)

    # 使用 pytest 框架的断言，检查当传递的 W 矩阵形状不合适时是否会引发 ValueError，并匹配特定错误信息
    with pytest.raises(ValueError, match="Array with wrong first dimension passed"):
        nmf.fit(X, H=H, W=rng.random_sample((5, 2)))

    # 再次使用 pytest 框架的断言，检查当传递的 W 矩阵形状不合适时是否会引发 ValueError，并匹配特定错误信息
    with pytest.raises(ValueError, match="Array with wrong second dimension passed"):
        nmf.fit(X, H=H, W=rng.random_sample((6, 3)))
```