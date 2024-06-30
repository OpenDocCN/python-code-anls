# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\tests\test_linear_loss.py`

```
"""
Tests for LinearModelLoss

Note that correctness of losses (which compose LinearModelLoss) is already well
covered in the _loss module.
"""

import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于编写和运行测试
from numpy.testing import assert_allclose  # 导入assert_allclose函数，用于检查数组是否接近
from scipy import linalg, optimize  # 导入linalg和optimize子模块，用于线性代数和优化

from sklearn._loss.loss import (  # 导入sklearn库中定义的损失函数
    HalfBinomialLoss,
    HalfMultinomialLoss,
    HalfPoissonLoss,
)
from sklearn.datasets import make_low_rank_matrix  # 导入make_low_rank_matrix函数，用于生成低秩矩阵
from sklearn.linear_model._linear_loss import LinearModelLoss  # 导入LinearModelLoss类，测试目标
from sklearn.utils.extmath import squared_norm  # 导入squared_norm函数，用于计算向量范数
from sklearn.utils.fixes import CSR_CONTAINERS  # 导入CSR_CONTAINERS，用于兼容性修复

# We do not need to test all losses, just what LinearModelLoss does on top of the
# base losses.
LOSSES = [HalfBinomialLoss, HalfMultinomialLoss, HalfPoissonLoss]  # 损失函数的列表


def random_X_y_coef(  # 定义函数random_X_y_coef，生成随机的X，y和coef
    linear_model_loss, n_samples, n_features, coef_bound=(-2, 2), seed=42
):
    """Random generate y, X and coef in valid range."""
    rng = np.random.RandomState(seed)  # 使用给定种子创建随机数生成器
    n_dof = n_features + linear_model_loss.fit_intercept  # 自由度数量，考虑是否拟合截距
    X = make_low_rank_matrix(  # 使用make_low_rank_matrix生成低秩矩阵X
        n_samples=n_samples,
        n_features=n_features,
        random_state=rng,
    )
    coef = linear_model_loss.init_zero_coef(X)  # 使用linear_model_loss初始化coef

    if linear_model_loss.base_loss.is_multiclass:  # 如果基础损失函数是多分类的
        n_classes = linear_model_loss.base_loss.n_classes  # 获取类别数量
        coef.flat[:] = rng.uniform(  # 用均匀分布随机初始化coef
            low=coef_bound[0],
            high=coef_bound[1],
            size=n_classes * n_dof,
        )
        if linear_model_loss.fit_intercept:  # 如果拟合截距
            raw_prediction = X @ coef[:, :-1].T + coef[:, -1]  # 计算原始预测值
        else:
            raw_prediction = X @ coef.T  # 计算原始预测值
        proba = linear_model_loss.base_loss.link.inverse(raw_prediction)  # 获取概率值

        # y = rng.choice(np.arange(n_classes), p=proba) does not work.
        # See https://stackoverflow.com/a/34190035/16761084
        def choice_vectorized(items, p):  # 定义向量化的选择函数
            s = p.cumsum(axis=1)  # 沿着行的累积和
            r = rng.rand(p.shape[0])[:, None]  # 随机数矩阵
            k = (s < r).sum(axis=1)  # 累积和比随机数小的个数
            return items[k]  # 返回选择的项目

        y = choice_vectorized(np.arange(n_classes), p=proba).astype(np.float64)  # 选择y值

    else:  # 如果不是多分类
        coef.flat[:] = rng.uniform(  # 使用均匀分布随机初始化coef
            low=coef_bound[0],
            high=coef_bound[1],
            size=n_dof,
        )
        if linear_model_loss.fit_intercept:  # 如果拟合截距
            raw_prediction = X @ coef[:-1] + coef[-1]  # 计算原始预测值
        else:
            raw_prediction = X @ coef  # 计算原始预测值
        y = linear_model_loss.base_loss.link.inverse(  # 使用链接函数获取y值
            raw_prediction + rng.uniform(low=-1, high=1, size=n_samples)
        )

    return X, y, coef  # 返回生成的X，y和coef


@pytest.mark.parametrize("base_loss", LOSSES)  # 参数化测试基础损失函数
@pytest.mark.parametrize("fit_intercept", [False, True])  # 参数化测试是否拟合截距
@pytest.mark.parametrize("n_features", [0, 1, 10])  # 参数化测试特征数量
@pytest.mark.parametrize("dtype", [None, np.float32, np.float64, np.int64])  # 参数化测试数据类型
def test_init_zero_coef(base_loss, fit_intercept, n_features, dtype):
    """Test that init_zero_coef initializes coef correctly."""
    loss = LinearModelLoss(base_loss=base_loss(), fit_intercept=fit_intercept)  # 创建LinearModelLoss对象
    rng = np.random.RandomState(42)  # 使用给定种子创建随机数生成器
    # 生成一个形状为 (5, n_features) 的正态分布随机数组 X
    X = rng.normal(size=(5, n_features))
    
    # 使用损失函数中的初始化零系数方法初始化系数 coef
    coef = loss.init_zero_coef(X, dtype=dtype)
    
    # 如果基础损失函数是多分类的，则获取类别数并进行断言检查
    if loss.base_loss.is_multiclass:
        n_classes = loss.base_loss.n_classes
        assert coef.shape == (n_classes, n_features + fit_intercept)  # 确保 coef 的形状符合预期
        assert coef.flags["F_CONTIGUOUS"]  # 断言 coef 是 Fortran（列优先）连续的
    else:
        assert coef.shape == (n_features + fit_intercept,)  # 确保 coef 的形状符合预期
    
    # 如果未指定 dtype，则断言 coef 的数据类型与 X 的数据类型相同
    if dtype is None:
        assert coef.dtype == X.dtype
    else:
        assert coef.dtype == dtype  # 否则，断言 coef 的数据类型与指定的 dtype 相同
    
    # 断言 coef 中非零元素的数量为 0
    assert np.count_nonzero(coef) == 0
# 使用 pytest.mark.parametrize 装饰器，对 base_loss 参数进行参数化测试，参数为 LOSSES 列表中的值
# 对 fit_intercept 参数进行参数化测试，参数为 False 和 True
# 对 sample_weight 参数进行参数化测试，参数为 None 和 "range"
# 对 l2_reg_strength 参数进行参数化测试，参数为 0 和 1
# 对 csr_container 参数进行参数化测试，参数为 CSR_CONTAINERS 列表中的值
@pytest.mark.parametrize("base_loss", LOSSES)
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("sample_weight", [None, "range"])
@pytest.mark.parametrize("l2_reg_strength", [0, 1])
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_loss_grad_hess_are_the_same(
    base_loss, fit_intercept, sample_weight, l2_reg_strength, csr_container
):
    """Test that loss and gradient are the same across different functions."""
    # 使用指定的 base_loss 和 fit_intercept 创建 LinearModelLoss 实例
    loss = LinearModelLoss(base_loss=base_loss(), fit_intercept=fit_intercept)
    # 调用 random_X_y_coef 函数生成随机的 X, y 数据和系数 coef
    X, y, coef = random_X_y_coef(
        linear_model_loss=loss, n_samples=10, n_features=5, seed=42
    )

    # 如果 sample_weight 参数为 "range"，则生成一个从 1 到样本数量的等差数列作为 sample_weight
    if sample_weight == "range":
        sample_weight = np.linspace(1, y.shape[0], num=y.shape[0])

    # 计算 loss 和 gradient，使用指定的参数和权重
    l1 = loss.loss(
        coef, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength
    )
    g1 = loss.gradient(
        coef, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength
    )
    # 调用 loss_gradient 方法计算 loss 和 gradient
    l2, g2 = loss.loss_gradient(
        coef, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength
    )
    # 调用 gradient_hessian_product 方法计算 gradient 和 hessian 乘积
    g3, h3 = loss.gradient_hessian_product(
        coef, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength
    )
    # 如果 base_loss 不是多分类模型，则调用 gradient_hessian 方法计算 gradient 和 hessian
    if not base_loss.is_multiclass:
        g4, h4, _ = loss.gradient_hessian(
            coef, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength
        )
    else:
        # 如果是多分类模型，则断言抛出 NotImplementedError 异常
        with pytest.raises(NotImplementedError):
            loss.gradient_hessian(
                coef,
                X,
                y,
                sample_weight=sample_weight,
                l2_reg_strength=l2_reg_strength,
            )

    # 断言 loss 和 l2 的值接近
    assert_allclose(l1, l2)
    # 断言 g1 和 g2 的值接近
    assert_allclose(g1, g2)
    # 断言 g1 和 g3 的值接近
    assert_allclose(g1, g3)
    # 如果不是多分类模型，断言 g1 和 g4 的值接近，并断言 h4 @ g4 与 h3(g3) 的值接近
    if not base_loss.is_multiclass:
        assert_allclose(g1, g4)
        assert_allclose(h4 @ g4, h3(g3))

    # 对稀疏矩阵 X 进行相同的测试
    X = csr_container(X)
    # 计算稀疏矩阵下的 loss
    l1_sp = loss.loss(
        coef, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength
    )
    # 计算稀疏矩阵下的 gradient
    g1_sp = loss.gradient(
        coef, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength
    )
    # 计算稀疏矩阵下的 loss 和 gradient
    l2_sp, g2_sp = loss.loss_gradient(
        coef, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength
    )
    # 计算稀疏矩阵下的 gradient 和 hessian 乘积
    g3_sp, h3_sp = loss.gradient_hessian_product(
        coef, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength
    )
    # 如果不是多分类模型，计算稀疏矩阵下的 gradient 和 hessian
    if not base_loss.is_multiclass:
        g4_sp, h4_sp, _ = loss.gradient_hessian(
            coef, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength
        )

    # 断言稀疏矩阵和原始矩阵下的 loss 值接近
    assert_allclose(l1, l1_sp)
    assert_allclose(l1, l2_sp)
    # 断言稀疏矩阵和原始矩阵下的 gradient 值接近
    assert_allclose(g1, g1_sp)
    assert_allclose(g1, g2_sp)
    assert_allclose(g1, g3_sp)
    # 断言稀疏矩阵下的 hessian 乘积与原始矩阵下的 hessian 乘积的值接近
    assert_allclose(h3(g1), h3_sp(g1_sp))
    # 如果不是多分类模型，断言稀疏矩阵和原始矩阵下的 g4_sp 和 h4_sp 的值接近
    if not base_loss.is_multiclass:
        assert_allclose(g1, g4_sp)
        assert_allclose(h4 @ g4, h4_sp @ g1_sp)
# 使用 pytest.mark.parametrize 装饰器为 test_loss_gradients_hessp_intercept 函数添加参数化测试
@pytest.mark.parametrize("l2_reg_strength", [0, 1])
@pytest.mark.parametrize("X_container", CSR_CONTAINERS + [None])
def test_loss_gradients_hessp_intercept(
    base_loss, sample_weight, l2_reg_strength, X_container
):
    """Test that loss and gradient handle intercept correctly."""
    # 初始化不带截距的线性模型损失函数
    loss = LinearModelLoss(base_loss=base_loss(), fit_intercept=False)
    # 初始化带有截距的线性模型损失函数
    loss_inter = LinearModelLoss(base_loss=base_loss(), fit_intercept=True)
    n_samples, n_features = 10, 5
    # 生成随机的特征矩阵 X, 目标值 y 和系数 coef
    X, y, coef = random_X_y_coef(
        linear_model_loss=loss, n_samples=n_samples, n_features=n_features, seed=42
    )

    X[:, -1] = 1  # 将最后一列设置为1，模拟截距项
    X_inter = X[
        :, :-1
    ]  # 排除截距列，因为它会被 loss_inter 自动添加

    if X_container is not None:
        X = X_container(X)

    if sample_weight == "range":
        sample_weight = np.linspace(1, y.shape[0], num=y.shape[0])

    # 计算不带截距的损失和梯度
    l, g = loss.loss_gradient(
        coef, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength
    )
    # 计算不带截距的梯度和海森矩阵乘积
    _, hessp = loss.gradient_hessian_product(
        coef, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength
    )
    # 计算带截距的损失和梯度
    l_inter, g_inter = loss_inter.loss_gradient(
        coef, X_inter, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength
    )
    # 计算带截距的梯度和海森矩阵乘积
    _, hessp_inter = loss_inter.gradient_hessian_product(
        coef, X_inter, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength
    )

    # 验证带截距的损失与不带截距的损失加上L2惩罚后的关系
    assert l == pytest.approx(
        l_inter + 0.5 * l2_reg_strength * squared_norm(coef.T[-1])
    )

    # 调整带截距梯度，添加L2正则化惩罚
    g_inter_corrected = g_inter
    g_inter_corrected.T[-1] += l2_reg_strength * coef.T[-1]
    # 验证不带截距的梯度与调整后的带截距梯度相等
    assert_allclose(g, g_inter_corrected)

    # 生成随机状态下的随机向量 s，并计算海森矩阵乘以 s
    s = np.random.RandomState(42).randn(*coef.shape)
    h = hessp(s)
    h_inter = hessp_inter(s)
    # 调整带截距的海森矩阵，添加L2正则化惩罚
    h_inter_corrected = h_inter
    h_inter_corrected.T[-1] += l2_reg_strength * s.T[-1]
    # 验证不带截距的海森矩阵与调整后的带截距海森矩阵相等
    assert_allclose(h, h_inter_corrected)


# 使用 pytest.mark.parametrize 装饰器为 test_gradients_hessians_numerically 函数添加参数化测试
@pytest.mark.parametrize("base_loss", LOSSES)
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("sample_weight", [None, "range"])
@pytest.mark.parametrize("l2_reg_strength", [0, 1])
def test_gradients_hessians_numerically(
    base_loss, fit_intercept, sample_weight, l2_reg_strength
):
    """Test gradients and hessians with numerical derivatives.

    Gradient should equal the numerical derivatives of the loss function.
    Hessians should equal the numerical derivatives of gradients.
    """
    # 初始化线性模型损失函数，可以选择是否带截距
    loss = LinearModelLoss(base_loss=base_loss(), fit_intercept=fit_intercept)
    n_samples, n_features = 10, 5
    # 生成随机的特征矩阵 X, 目标值 y 和系数 coef
    X, y, coef = random_X_y_coef(
        linear_model_loss=loss, n_samples=n_samples, n_features=n_features, seed=42
    )
    coef = coef.ravel(order="F")  # 对于多项式损失，此步骤很重要

    if sample_weight == "range":
        sample_weight = np.linspace(1, y.shape[0], num=y.shape[0])

    # 1. 检查数值梯度
    eps = 1e-6
    g, hessp = loss.gradient_hessian_product(
        coef, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength
    )
    # 调用函数计算损失函数关于参数 coef 的梯度和 Hessian 乘积，返回梯度 g 和 Hessian 乘积 hessp

    # Use a trick to get central finite difference of accuracy 4 (five-point stencil)
    # 使用技巧来获取精度为 4 的中心有限差分（五点式）
    # https://en.wikipedia.org/wiki/Numerical_differentiation
    # https://en.wikipedia.org/wiki/Finite_difference_coefficient

    # approx_g1 = (f(x + eps) - f(x - eps)) / (2*eps)
    approx_g1 = optimize.approx_fprime(
        coef,
        lambda coef: loss.loss(
            coef - eps,
            X,
            y,
            sample_weight=sample_weight,
            l2_reg_strength=l2_reg_strength,
        ),
        2 * eps,
    )
    # 使用 optimize.approx_fprime 函数来近似计算损失函数 loss 对于参数 coef 的梯度，使用中心差分法

    # approx_g2 = (f(x + 2*eps) - f(x - 2*eps)) / (4*eps)
    approx_g2 = optimize.approx_fprime(
        coef,
        lambda coef: loss.loss(
            coef - 2 * eps,
            X,
            y,
            sample_weight=sample_weight,
            l2_reg_strength=l2_reg_strength,
        ),
        4 * eps,
    )
    # 使用 optimize.approx_fprime 函数来近似计算损失函数 loss 对于参数 coef 的梯度，使用中心差分法

    # Five-point stencil approximation
    # 使用五点式逼近法来计算梯度
    # See: https://en.wikipedia.org/wiki/Five-point_stencil#1D_first_derivative

    approx_g = (4 * approx_g1 - approx_g2) / 3
    # 使用五点式逼近法计算梯度的数值逼近值

    assert_allclose(g, approx_g, rtol=1e-2, atol=1e-8)
    # 断言：验证计算得到的梯度 g 与数值逼近值 approx_g 的接近程度，允许相对误差 rtol 和绝对误差 atol

    # 2. Check hessp numerically along the second direction of the gradient
    # 2. 在梯度的第二方向上数值检查 hessp

    vector = np.zeros_like(g)
    vector[1] = 1
    # 创建一个与 g 形状相同的零向量，并将第二个元素设为 1，表示在梯度的第二个方向上的向量

    hess_col = hessp(vector)
    # 调用 hessp 函数计算在给定向量上的 Hessian 乘积 hess_col

    # Computation of the Hessian is particularly fragile to numerical errors when doing
    # simple finite differences. Here we compute the grad along a path in the direction
    # of the vector and then use a least-square regression to estimate the slope
    # 计算 Hessian 矩阵时在进行简单有限差分时特别容易受到数值误差的影响。这里我们沿着向量方向计算梯度，
    # 然后使用最小二乘回归来估计斜率

    eps = 1e-3
    # 设置一个小的步长 eps

    d_x = np.linspace(-eps, eps, 30)
    # 在 [-eps, eps] 区间内生成 30 个均匀分布的点作为步长序列 d_x

    d_grad = np.array(
        [
            loss.gradient(
                coef + t * vector,
                X,
                y,
                sample_weight=sample_weight,
                l2_reg_strength=l2_reg_strength,
            )
            for t in d_x
        ]
    )
    # 计算在 coef 沿着 vector 方向上的梯度的序列 d_grad

    d_grad -= d_grad.mean(axis=0)
    # 减去 d_grad 在列方向上的均值

    approx_hess_col = linalg.lstsq(d_x[:, np.newaxis], d_grad)[0].ravel()
    # 使用最小二乘法回归来估计 Hessian 列的数值逼近值 approx_hess_col

    assert_allclose(approx_hess_col, hess_col, rtol=1e-3)
    # 断言：验证计算得到的 Hessian 列的数值逼近值 approx_hess_col 与 hess_col 的接近程度，允许相对误差 rtol
# 使用 pytest 的 mark.parametrize 装饰器，定义了一个参数化测试函数，用于测试多项式线性模型损失函数的系数形状
@pytest.mark.parametrize("fit_intercept", [False, True])
def test_multinomial_coef_shape(fit_intercept):
    """Test that multinomial LinearModelLoss respects shape of coef."""
    # 创建一个多项式线性模型损失函数对象，基于 HalfMultinomialLoss，指定是否拟合截距
    loss = LinearModelLoss(base_loss=HalfMultinomialLoss(), fit_intercept=fit_intercept)
    # 设定样本数和特征数
    n_samples, n_features = 10, 5
    # 生成随机的 X, y 和系数 coef，使用指定的种子
    X, y, coef = random_X_y_coef(
        linear_model_loss=loss, n_samples=n_samples, n_features=n_features, seed=42
    )
    # 生成一个与 coef 相同形状的随机数组 s
    s = np.random.RandomState(42).randn(*coef.shape)

    # 计算损失值 l 和梯度 g
    l, g = loss.loss_gradient(coef, X, y)
    # 计算梯度 g1
    g1 = loss.gradient(coef, X, y)
    # 计算梯度 g2 和 Hessian 矩阵乘积 hessp
    g2, hessp = loss.gradient_hessian_product(coef, X, y)
    # 计算 Hessian 矩阵乘积结果 h
    h = hessp(s)
    # 断言梯度 g 的形状与 coef 相同
    assert g.shape == coef.shape
    # 断言 Hessian 矩阵乘积结果 h 的形状与 coef 相同
    assert h.shape == coef.shape
    # 使用 assert_allclose 检查 g 和 g1 的值是否相近
    assert_allclose(g, g1)
    # 使用 assert_allclose 检查 g 和 g2 的值是否相近
    assert_allclose(g, g2)

    # 将 coef 和 s 按列展平成一维数组
    coef_r = coef.ravel(order="F")
    s_r = s.ravel(order="F")
    # 计算展平后的损失值 l_r 和梯度 g_r
    l_r, g_r = loss.loss_gradient(coef_r, X, y)
    # 计算展平后的梯度 g1_r
    g1_r = loss.gradient(coef_r, X, y)
    # 计算展平后的梯度 g2_r 和 Hessian 矩阵乘积 hessp_r
    g2_r, hessp_r = loss.gradient_hessian_product(coef_r, X, y)
    # 计算展平后的 Hessian 矩阵乘积结果 h_r
    h_r = hessp_r(s_r)
    # 断言展平后的梯度 g_r 的形状与 coef_r 相同
    assert g_r.shape == coef_r.shape
    # 断言展平后的 Hessian 矩阵乘积结果 h_r 的形状与 coef_r 相同
    assert h_r.shape == coef_r.shape
    # 使用 assert_allclose 检查 g_r 和 g1_r 的值是否相近
    assert_allclose(g_r, g1_r)
    # 使用 assert_allclose 检查 g_r 和 g2_r 的值是否相近
    assert_allclose(g_r, g2_r)

    # 使用 assert_allclose 检查 g 和 g_r 的值是否相近，并将 g_r 转换为与 loss.base_loss.n_classes 一致的形状
    assert_allclose(g, g_r.reshape(loss.base_loss.n_classes, -1, order="F"))
    # 使用 assert_allclose 检查 h 和 h_r 的值是否相近，并将 h_r 转换为与 loss.base_loss.n_classes 一致的形状
    assert_allclose(h, h_r.reshape(loss.base_loss.n_classes, -1, order="F"))
```