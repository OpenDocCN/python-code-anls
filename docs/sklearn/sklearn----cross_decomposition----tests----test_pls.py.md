# `D:\src\scipysrc\scikit-learn\sklearn\cross_decomposition\tests\test_pls.py`

```
# 导入警告模块，用于处理警告信息
import warnings

# 导入 NumPy 库，并将其重命名为 np
import numpy as np

# 导入 pytest 模块，用于编写和运行测试用例
import pytest

# 从 NumPy 测试模块中导入断言函数，用于检查数组的近似相等性
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal

# 导入 sklearn 中的交叉分解模块
from sklearn.cross_decomposition import CCA, PLSSVD, PLSCanonical, PLSRegression

# 导入 sklearn 中交叉分解模块的私有函数和类
from sklearn.cross_decomposition._pls import (
    _center_scale_xy,
    _get_first_singular_vectors_power_method,
    _get_first_singular_vectors_svd,
    _svd_flip_1d,
)

# 导入 sklearn 中的数据集加载函数和生成函数
from sklearn.datasets import load_linnerud, make_regression

# 导入 sklearn 中的集成学习模块中的回归器
from sklearn.ensemble import VotingRegressor

# 导入 sklearn 中的异常类
from sklearn.exceptions import ConvergenceWarning

# 导入 sklearn 中的线性回归模型
from sklearn.linear_model import LinearRegression

# 导入 sklearn 中的工具函数
from sklearn.utils import check_random_state

# 导入 sklearn 中的数学扩展函数，用于矩阵操作
from sklearn.utils.extmath import svd_flip


# 定义一个函数，用于断言矩阵的正交性
def assert_matrix_orthogonal(M):
    # 计算矩阵 M 的转置与自身的乘积
    K = np.dot(M.T, M)
    # 断言矩阵 K 与其对角线元素相等，即 M 是正交矩阵
    assert_array_almost_equal(K, np.diag(np.diag(K)))


# 定义一个测试函数，用于测试 PLSCanonical 类的基本功能
def test_pls_canonical_basics():
    # 加载 linnerud 数据集
    d = load_linnerud()
    # 获取数据集的特征和目标值
    X = d.data
    Y = d.target

    # 创建 PLSCanonical 对象，设置主成分数量为 X 的特征数
    pls = PLSCanonical(n_components=X.shape[1])
    # 对数据集进行拟合
    pls.fit(X, Y)

    # 断言 X 的权重向量矩阵为正交矩阵
    assert_matrix_orthogonal(pls.x_weights_)
    # 断言 Y 的权重向量矩阵为正交矩阵
    assert_matrix_orthogonal(pls.y_weights_)
    # 断言 X 的分数矩阵为正交矩阵
    assert_matrix_orthogonal(pls._x_scores)
    # 断言 Y 的分数矩阵为正交矩阵
    assert_matrix_orthogonal(pls._y_scores)

    # 检查 X = TP' 和 Y = UQ' 是否成立
    T = pls._x_scores
    P = pls.x_loadings_
    U = pls._y_scores
    Q = pls.y_loadings_

    # 需要先进行缩放
    Xc, Yc, x_mean, y_mean, x_std, y_std = _center_scale_xy(
        X.copy(), Y.copy(), scale=True
    )
    # 断言中心化和缩放后的 Xc 与 T * P' 的近似相等性
    assert_array_almost_equal(Xc, np.dot(T, P.T))
    # 断言中心化和缩放后的 Yc 与 U * Q' 的近似相等性
    assert_array_almost_equal(Yc, np.dot(U, Q.T))

    # 检查对训练数据的旋转是否导致相应的分数
    Xt = pls.transform(X)
    assert_array_almost_equal(Xt, pls._x_scores)
    Xt, Yt = pls.transform(X, Y)
    assert_array_almost_equal(Xt, pls._x_scores)
    assert_array_almost_equal(Yt, pls._y_scores)

    # 检查逆变换是否有效
    X_back = pls.inverse_transform(Xt)
    assert_array_almost_equal(X_back, X)
    _, Y_back = pls.inverse_transform(Xt, Yt)
    assert_array_almost_equal(Y_back, Y)


# 定义一个测试函数，用于对 PLSRegression 进行健全性检查
def test_sanity_check_pls_regression():
    # 加载 linnerud 数据集
    d = load_linnerud()
    # 获取数据集的特征和目标值
    X = d.data
    Y = d.target

    # 创建 PLSRegression 对象，设置主成分数量为 X 的特征数
    pls = PLSRegression(n_components=X.shape[1])
    # 对数据集进行拟合，并执行变换
    X_trans, _ = pls.fit_transform(X, Y)

    # FIXME: 预期 y_trans == pls.y_scores_，但实际情况并非如此。
    # 这是一个问题，参考 https://github.com/scikit-learn/scikit-learn/issues/22420
    # 断言 X_trans 与 pls.x_scores_ 的近似相等性
    assert_allclose(X_trans, pls.x_scores_)

    # 预期的 X 权重向量
    expected_x_weights = np.array(
        [
            [-0.61330704, -0.00443647, 0.78983213],
            [-0.74697144, -0.32172099, -0.58183269],
            [-0.25668686, 0.94682413, -0.19399983],
        ]
    )

    # 预期的 X 负荷矩阵
    expected_x_loadings = np.array(
        [
            [-0.61470416, -0.24574278, 0.78983213],
            [-0.65625755, -0.14396183, -0.58183269],
            [-0.51733059, 1.00609417, -0.19399983],
        ]
    )
    # 定义预期的 Y 权重矩阵，是一个 NumPy 数组
    expected_y_weights = np.array(
        [
            [+0.32456184, 0.29892183, 0.20316322],
            [+0.42439636, 0.61970543, 0.19320542],
            [-0.13143144, -0.26348971, -0.17092916],
        ]
    )
    
    # 定义预期的 Y 载荷矩阵，也是一个 NumPy 数组
    expected_y_loadings = np.array(
        [
            [+0.32456184, 0.29892183, 0.20316322],
            [+0.42439636, 0.61970543, 0.19320542],
            [-0.13143144, -0.26348971, -0.17092916],
        ]
    )
    
    # 断言 PLS 模型的 X 载荷矩阵近似等于预期的 X 载荷矩阵
    assert_array_almost_equal(np.abs(pls.x_loadings_), np.abs(expected_x_loadings))
    # 断言 PLS 模型的 X 权重矩阵近似等于预期的 X 权重矩阵
    assert_array_almost_equal(np.abs(pls.x_weights_), np.abs(expected_x_weights))
    # 断言 PLS 模型的 Y 载荷矩阵近似等于预期的 Y 载荷矩阵
    assert_array_almost_equal(np.abs(pls.y_loadings_), np.abs(expected_y_loadings))
    # 断言 PLS 模型的 Y 权重矩阵近似等于预期的 Y 权重矩阵
    assert_array_almost_equal(np.abs(pls.y_weights_), np.abs(expected_y_weights))
    
    # 在 R / Python 间差异中，载荷、权重等符号应该保持一致
    # 计算 PLS 模型 X 载荷矩阵的符号翻转情况
    x_loadings_sign_flip = np.sign(pls.x_loadings_ / expected_x_loadings)
    # 计算 PLS 模型 X 权重矩阵的符号翻转情况
    x_weights_sign_flip = np.sign(pls.x_weights_ / expected_x_weights)
    # 计算 PLS 模型 Y 权重矩阵的符号翻转情况
    y_weights_sign_flip = np.sign(pls.y_weights_ / expected_y_weights)
    # 计算 PLS 模型 Y 载荷矩阵的符号翻转情况
    y_loadings_sign_flip = np.sign(pls.y_loadings_ / expected_y_loadings)
    # 断言 X 载荷矩阵的符号翻转与 X 权重矩阵的符号翻转近似等价
    assert_array_almost_equal(x_loadings_sign_flip, x_weights_sign_flip)
    # 断言 Y 载荷矩阵的符号翻转与 Y 权重矩阵的符号翻转近似等价
    assert_array_almost_equal(y_loadings_sign_flip, y_weights_sign_flip)
# 定义一个函数用于测试 PLS 回归在 Y 的第一列为常数时的行为
def test_sanity_check_pls_regression_constant_column_Y():
    # 从数据集加载 Linnerud 数据集
    d = load_linnerud()
    # 获取数据集中的 X 和 Y 数据
    X = d.data
    Y = d.target
    # 将 Y 的第一列设为常数 1
    Y[:, 0] = 1
    # 创建一个 PLSRegression 对象，指定主成分数为 X 的列数
    pls = PLSRegression(n_components=X.shape[1])
    # 使用 X 和修改后的 Y 训练 PLS 模型
    pls.fit(X, Y)

    # 期望的 X 权重向量
    expected_x_weights = np.array(
        [
            [-0.6273573, 0.007081799, 0.7786994],
            [-0.7493417, -0.277612681, -0.6011807],
            [-0.2119194, 0.960666981, -0.1794690],
        ]
    )

    # 期望的 X 负荷向量
    expected_x_loadings = np.array(
        [
            [-0.6273512, -0.22464538, 0.7786994],
            [-0.6643156, -0.09871193, -0.6011807],
            [-0.5125877, 1.01407380, -0.1794690],
        ]
    )

    # 期望的 Y 负荷向量
    expected_y_loadings = np.array(
        [
            [0.0000000, 0.0000000, 0.0000000],
            [0.4357300, 0.5828479, 0.2174802],
            [-0.1353739, -0.2486423, -0.1810386],
        ]
    )

    # 检查计算得到的 X 权重向量与期望值的绝对值之间的近似相等关系
    assert_array_almost_equal(np.abs(expected_x_weights), np.abs(pls.x_weights_))
    # 检查计算得到的 X 负荷向量与期望值的绝对值之间的近似相等关系
    assert_array_almost_equal(np.abs(expected_x_loadings), np.abs(pls.x_loadings_))
    # 对于 PLSRegression 默认参数情况下，y_loadings == y_weights
    assert_array_almost_equal(np.abs(pls.y_loadings_), np.abs(expected_y_loadings))
    assert_array_almost_equal(np.abs(pls.y_weights_), np.abs(expected_y_loadings))

    # 计算 X 负荷向量的符号翻转情况
    x_loadings_sign_flip = np.sign(expected_x_loadings / pls.x_loadings_)
    # 计算 X 权重向量的符号翻转情况
    x_weights_sign_flip = np.sign(expected_x_weights / pls.x_weights_)
    # 忽略 Y 的第一行全零的情况
    y_loadings_sign_flip = np.sign(expected_y_loadings[1:] / pls.y_loadings_[1:])

    # 检查 X 负荷向量和 X 权重向量的符号翻转情况是否相等
    assert_array_equal(x_loadings_sign_flip, x_weights_sign_flip)
    # 检查 X 负荷向量和 Y 负荷向量的符号翻转情况是否相等（忽略第一行）
    assert_array_equal(x_loadings_sign_flip[1:], y_loadings_sign_flip)


# 定义一个函数用于测试 PLSCanonical 的基本一致性
def test_sanity_check_pls_canonical():
    # 从数据集加载 Linnerud 数据集
    d = load_linnerud()
    # 获取数据集中的 X 和 Y 数据
    X = d.data
    Y = d.target

    # 创建一个 PLSCanonical 对象，指定主成分数为 X 的列数
    pls = PLSCanonical(n_components=X.shape[1])
    # 使用 X 和 Y 训练 PLSCanonical 模型
    pls.fit(X, Y)

    # 期望的 X 权重向量
    expected_x_weights = np.array(
        [
            [-0.61330704, 0.25616119, -0.74715187],
            [-0.74697144, 0.11930791, 0.65406368],
            [-0.25668686, -0.95924297, -0.11817271],
        ]
    )

    # 期望的 X 旋转向量
    expected_x_rotations = np.array(
        [
            [-0.61330704, 0.41591889, -0.62297525],
            [-0.74697144, 0.31388326, 0.77368233],
            [-0.25668686, -0.89237972, -0.24121788],
        ]
    )

    # 期望的 Y 权重向量
    expected_y_weights = np.array(
        [
            [+0.58989127, 0.7890047, 0.1717553],
            [+0.77134053, -0.61351791, 0.16920272],
            [-0.23887670, -0.03267062, 0.97050016],
        ]
    )

    # 期望的 Y 旋转向量
    expected_y_rotations = np.array(
        [
            [+0.58989127, 0.7168115, 0.30665872],
            [+0.77134053, -0.70791757, 0.19786539],
            [-0.23887670, -0.00343595, 0.94162826],
        ]
    )
    # 断言两个数组的元素近似相等，使用绝对值进行比较
    assert_array_almost_equal(np.abs(pls.x_rotations_), np.abs(expected_x_rotations))
    assert_array_almost_equal(np.abs(pls.x_weights_), np.abs(expected_x_weights))
    assert_array_almost_equal(np.abs(pls.y_rotations_), np.abs(expected_y_rotations))
    assert_array_almost_equal(np.abs(pls.y_weights_), np.abs(expected_y_weights))

    # 计算每个旋转系数的符号翻转，以检查是否与预期符号一致
    x_rotations_sign_flip = np.sign(pls.x_rotations_ / expected_x_rotations)
    x_weights_sign_flip = np.sign(pls.x_weights_ / expected_x_weights)
    y_rotations_sign_flip = np.sign(pls.y_rotations_ / expected_y_rotations)
    y_weights_sign_flip = np.sign(pls.y_weights_ / expected_y_weights)
    assert_array_almost_equal(x_rotations_sign_flip, x_weights_sign_flip)
    assert_array_almost_equal(y_rotations_sign_flip, y_weights_sign_flip)

    # 断言矩阵是正交的
    assert_matrix_orthogonal(pls.x_weights_)
    assert_matrix_orthogonal(pls.y_weights_)

    # 断言矩阵是正交的，这些矩阵可能是类内私有变量
    assert_matrix_orthogonal(pls._x_scores)
    assert_matrix_orthogonal(pls._y_scores)
def test_sanity_check_pls_canonical_random():
    # 对 PLSCanonical 在随机数据上进行健全性检查
    # 结果已经与 R 包 plspm 进行了验证

    # 设定数据维度
    n = 500  # 样本数
    p_noise = 10  # X 中的噪声变量数目
    q_noise = 5   # Y 中的噪声变量数目

    # 生成随机数种子并生成随机数据
    rng = check_random_state(11)
    l1 = rng.normal(size=n)
    l2 = rng.normal(size=n)
    latents = np.array([l1, l1, l2, l2]).T
    X = latents + rng.normal(size=4 * n).reshape((n, 4))
    Y = latents + rng.normal(size=4 * n).reshape((n, 4))

    # 添加噪声变量到 X 和 Y 中
    X = np.concatenate((X, rng.normal(size=p_noise * n).reshape(n, p_noise)), axis=1)
    Y = np.concatenate((Y, rng.normal(size=q_noise * n).reshape(n, q_noise)), axis=1)

    # 创建 PLSCanonical 对象并拟合数据
    pls = PLSCanonical(n_components=3)
    pls.fit(X, Y)

    # 预期的 X 权重矩阵
    expected_x_weights = np.array(
        [
            [0.65803719, 0.19197924, 0.21769083],
            [0.7009113, 0.13303969, -0.15376699],
            [0.13528197, -0.68636408, 0.13856546],
            [0.16854574, -0.66788088, -0.12485304],
            [-0.03232333, -0.04189855, 0.40690153],
            [0.1148816, -0.09643158, 0.1613305],
            [0.04792138, -0.02384992, 0.17175319],
            [-0.06781, -0.01666137, -0.18556747],
            [-0.00266945, -0.00160224, 0.11893098],
            [-0.00849528, -0.07706095, 0.1570547],
            [-0.00949471, -0.02964127, 0.34657036],
            [-0.03572177, 0.0945091, 0.3414855],
            [0.05584937, -0.02028961, -0.57682568],
            [0.05744254, -0.01482333, -0.17431274],
        ]
    )

    # 预期的 X 加载矩阵
    expected_x_loadings = np.array(
        [
            [0.65649254, 0.1847647, 0.15270699],
            [0.67554234, 0.15237508, -0.09182247],
            [0.19219925, -0.67750975, 0.08673128],
            [0.2133631, -0.67034809, -0.08835483],
            [-0.03178912, -0.06668336, 0.43395268],
            [0.15684588, -0.13350241, 0.20578984],
            [0.03337736, -0.03807306, 0.09871553],
            [-0.06199844, 0.01559854, -0.1881785],
            [0.00406146, -0.00587025, 0.16413253],
            [-0.00374239, -0.05848466, 0.19140336],
            [0.00139214, -0.01033161, 0.32239136],
            [-0.05292828, 0.0953533, 0.31916881],
            [0.04031924, -0.01961045, -0.65174036],
            [0.06172484, -0.06597366, -0.1244497],
        ]
    )

    # 预期的 Y 权重矩阵
    expected_y_weights = np.array(
        [
            [0.66101097, 0.18672553, 0.22826092],
            [0.69347861, 0.18463471, -0.23995597],
            [0.14462724, -0.66504085, 0.17082434],
            [0.22247955, -0.6932605, -0.09832993],
            [0.07035859, 0.00714283, 0.67810124],
            [0.07765351, -0.0105204, -0.44108074],
            [-0.00917056, 0.04322147, 0.10062478],
            [-0.01909512, 0.06182718, 0.28830475],
            [0.01756709, 0.04797666, 0.32225745],
        ]
    )
    # 定义期望的 Y 负载因子矩阵，包含预期值的数组
    expected_y_loadings = np.array(
        [
            [0.68568625, 0.1674376, 0.0969508],
            [0.68782064, 0.20375837, -0.1164448],
            [0.11712173, -0.68046903, 0.12001505],
            [0.17860457, -0.6798319, -0.05089681],
            [0.06265739, -0.0277703, 0.74729584],
            [0.0914178, 0.00403751, -0.5135078],
            [-0.02196918, -0.01377169, 0.09564505],
            [-0.03288952, 0.09039729, 0.31858973],
            [0.04287624, 0.05254676, 0.27836841],
        ]
    )
    
    # 检查 PLS 模型的 X 负载因子是否与期望值接近，使用 assert_array_almost_equal 进行比较
    assert_array_almost_equal(np.abs(pls.x_loadings_), np.abs(expected_x_loadings))
    # 检查 PLS 模型的 X 权重是否与期望值接近
    assert_array_almost_equal(np.abs(pls.x_weights_), np.abs(expected_x_weights))
    # 检查 PLS 模型的 Y 负载因子是否与期望值接近
    assert_array_almost_equal(np.abs(pls.y_loadings_), np.abs(expected_y_loadings))
    # 检查 PLS 模型的 Y 权重是否与期望值接近
    assert_array_almost_equal(np.abs(pls.y_weights_), np.abs(expected_y_weights))
    
    # 计算 PLS 模型中 X 负载因子的符号翻转情况，用于后续比较
    x_loadings_sign_flip = np.sign(pls.x_loadings_ / expected_x_loadings)
    # 计算 PLS 模型中 X 权重的符号翻转情况
    x_weights_sign_flip = np.sign(pls.x_weights_ / expected_x_weights)
    # 计算 PLS 模型中 Y 权重的符号翻转情况
    y_weights_sign_flip = np.sign(pls.y_weights_ / expected_y_weights)
    # 计算 PLS 模型中 Y 负载因子的符号翻转情况
    y_loadings_sign_flip = np.sign(pls.y_loadings_ / expected_y_loadings)
    # 检查 X 负载因子和权重的符号翻转是否一致
    assert_array_almost_equal(x_loadings_sign_flip, x_weights_sign_flip)
    # 检查 Y 负载因子和权重的符号翻转是否一致
    assert_array_almost_equal(y_loadings_sign_flip, y_weights_sign_flip)
    
    # 断言 PLS 模型中 X 权重矩阵是正交的
    assert_matrix_orthogonal(pls.x_weights_)
    # 断言 PLS 模型中 Y 权重矩阵是正交的
    assert_matrix_orthogonal(pls.y_weights_)
    
    # 断言 PLS 模型中的 X 得分矩阵是正交的
    assert_matrix_orthogonal(pls._x_scores)
    # 断言 PLS 模型中的 Y 得分矩阵是正交的
    assert_matrix_orthogonal(pls._y_scores)
def test_convergence_fail():
    # 确保在 max_iter 设置过小时会引发 ConvergenceWarning
    # 载入 linnerud 数据集
    d = load_linnerud()
    # 提取数据和目标变量
    X = d.data
    Y = d.target
    # 使用 max_iter=2 初始化 PLSCanonical 模型
    pls_nipals = PLSCanonical(n_components=X.shape[1], max_iter=2)
    # 使用 pytest 的 warn 断言检测是否会引发 ConvergenceWarning
    with pytest.warns(ConvergenceWarning):
        pls_nipals.fit(X, Y)


@pytest.mark.parametrize("Est", (PLSSVD, PLSRegression, PLSCanonical))
def test_attibutes_shapes(Est):
    # 确保不同估计器的属性在 n_components 设置下的形状正确
    # 载入 linnerud 数据集
    d = load_linnerud()
    # 提取数据和目标变量
    X = d.data
    Y = d.target
    # 设置 n_components 为 2
    n_components = 2
    # 初始化对应的估计器 Est
    pls = Est(n_components=n_components)
    # 拟合模型
    pls.fit(X, Y)
    # 使用断言确保属性的第二维度等于 n_components
    assert all(
        attr.shape[1] == n_components for attr in (pls.x_weights_, pls.y_weights_)
    )


@pytest.mark.parametrize("Est", (PLSRegression, PLSCanonical, CCA))
def test_univariate_equivalence(Est):
    # 确保使用 2D Y 和仅有 1 列的 Y 等效
    # 载入 linnerud 数据集
    d = load_linnerud()
    # 提取数据和目标变量
    X = d.data
    Y = d.target

    # 初始化估计器 Est，设置 n_components 为 1
    est = Est(n_components=1)
    # 对 Y 的第一列进行拟合，获取系数
    one_d_coeff = est.fit(X, Y[:, 0]).coef_
    # 对 Y 的前 1 列进行拟合，获取系数
    two_d_coeff = est.fit(X, Y[:, :1]).coef_

    # 使用断言确保系数的形状一致
    assert one_d_coeff.shape == two_d_coeff.shape
    assert_array_almost_equal(one_d_coeff, two_d_coeff)


@pytest.mark.parametrize("Est", (PLSRegression, PLSCanonical, CCA, PLSSVD))
def test_copy(Est):
    # 检查 "copy" 关键字的工作情况
    # 载入 linnerud 数据集
    d = load_linnerud()
    # 提取数据和目标变量
    X = d.data
    Y = d.target
    # 复制 X 的原始版本
    X_orig = X.copy()

    # 使用 copy=True 时不会修改原始数据
    pls = Est(copy=True).fit(X, Y)
    assert_array_equal(X, X_orig)

    # 使用 copy=False 时会修改原始数据，使用断言检查是否引发 AssertionError
    with pytest.raises(AssertionError):
        Est(copy=False).fit(X, Y)
        assert_array_almost_equal(X, X_orig)

    # 如果 Est 是 PLSSVD，则直接返回，因为 PLSSVD 不支持 copy 参数
    if Est is PLSSVD:
        return

    # 进行 transform 和 predict 操作时，使用 copy=False 会修改原始数据，使用断言检查是否引发 AssertionError
    X_orig = X.copy()
    with pytest.raises(AssertionError):
        pls.transform(X, Y, copy=False),
        assert_array_almost_equal(X, X_orig)

    X_orig = X.copy()
    with pytest.raises(AssertionError):
        pls.predict(X, copy=False),
        assert_array_almost_equal(X, X_orig)

    # 确保 copy=True 给出的 transform 和 predict 结果与 copy=False 时一致
    assert_array_almost_equal(
        pls.transform(X, Y, copy=True), pls.transform(X.copy(), Y.copy(), copy=False)
    )
    assert_array_almost_equal(
        pls.predict(X, copy=True), pls.predict(X.copy(), copy=False)
    )


def _generate_test_scale_and_stability_datasets():
    """生成用于 test_scale_and_stability 的数据集"""
    # 生成非回归测试数据集 7818
    rng = np.random.RandomState(0)
    n_samples = 1000
    n_targets = 5
    n_features = 10
    Q = rng.randn(n_targets, n_features)
    Y = rng.randn(n_samples, n_targets)
    X = np.dot(Y, Q) + 2 * rng.randn(n_samples, n_features) + 1
    X *= 1000
    yield X, Y

    # 数据集，其中一个特征被约束
    X, Y = load_linnerud(return_X_y=True)
    # 导致 X[:, -1].std() 为零
    X[:, -1] = 1.0
    yield X, Y
    # 创建一个 NumPy 数组 X，包含四行三列的数据
    X = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [2.0, 2.0, 2.0], [3.0, 5.0, 4.0]])
    # 创建一个 NumPy 数组 Y，包含四行两列的数据
    Y = np.array([[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]])
    # 生成器函数的第一个 yield 语句，返回数组 X 和 Y
    yield X, Y

    # 用于执行非回归测试的种子，测试#18746中 CCA 失败的情况
    seeds = [530, 741]
    # 遍历种子列表，依次执行以下操作
    for seed in seeds:
        # 使用指定种子创建随机数生成器对象
        rng = np.random.RandomState(seed)
        # 使用随机数生成器生成一个形状为 (4, 3) 的随机数数组 X
        X = rng.randn(4, 3)
        # 使用随机数生成器生成一个形状为 (4, 2) 的随机数数组 Y
        Y = rng.randn(4, 2)
        # 生成器函数的另一个 yield 语句，返回当前生成的数组 X 和 Y
        yield X, Y
@pytest.mark.parametrize("Est", (CCA, PLSCanonical, PLSRegression, PLSSVD))
@pytest.mark.parametrize("X, Y", _generate_test_scale_and_stability_datasets())
def test_scale_and_stability(Est, X, Y):
    """scale=True is equivalent to scale=False on centered/scaled data
    This allows to check numerical stability over platforms as well"""

    X_s, Y_s, *_ = _center_scale_xy(X, Y)  # 对输入数据进行居中和标准化处理

    X_score, Y_score = Est(scale=True).fit_transform(X, Y)  # 使用指定估计器对数据进行拟合和变换，启用标准化
    X_s_score, Y_s_score = Est(scale=False).fit_transform(X_s, Y_s)  # 使用指定估计器对居中和标准化后的数据进行拟合和变换，禁用标准化

    assert_allclose(X_s_score, X_score, atol=1e-4)  # 断言两个数组的所有元素在给定容差下相等
    assert_allclose(Y_s_score, Y_score, atol=1e-4)


@pytest.mark.parametrize("Estimator", (PLSSVD, PLSRegression, PLSCanonical, CCA))
def test_n_components_upper_bounds(Estimator):
    """Check the validation of `n_components` upper bounds for `PLS` regressors."""
    rng = np.random.RandomState(0)
    X = rng.randn(10, 5)
    Y = rng.randn(10, 3)
    est = Estimator(n_components=10)  # 创建估计器实例，设置超过允许值的 n_components
    err_msg = "`n_components` upper bound is .*. Got 10 instead. Reduce `n_components`."
    with pytest.raises(ValueError, match=err_msg):  # 断言抛出 ValueError 异常，并检查异常消息匹配预期的模式
        est.fit(X, Y)


@pytest.mark.parametrize("n_samples, n_features", [(100, 10), (100, 200)])
def test_singular_value_helpers(n_samples, n_features, global_random_seed):
    # Make sure SVD and power method give approximately the same results
    X, Y = make_regression(
        n_samples, n_features, n_targets=5, random_state=global_random_seed
    )
    u1, v1, _ = _get_first_singular_vectors_power_method(X, Y, norm_y_weights=True)  # 使用功率方法获取第一奇异向量
    u2, v2 = _get_first_singular_vectors_svd(X, Y)  # 使用SVD方法获取第一奇异向量对

    _svd_flip_1d(u1, v1)  # 原地修改 u1, v1 的方向使其一致
    _svd_flip_1d(u2, v2)  # 原地修改 u2, v2 的方向使其一致

    rtol = 1e-3
    # 设置 atol 是因为某些坐标非常接近零
    assert_allclose(u1, u2, atol=u2.max() * rtol)  # 断言两个数组的所有元素在给定容差下相等
    assert_allclose(v1, v2, atol=v2.max() * rtol)


def test_one_component_equivalence(global_random_seed):
    # PLSSVD, PLSRegression and PLSCanonical should all be equivalent when
    # n_components is 1
    X, Y = make_regression(100, 10, n_targets=5, random_state=global_random_seed)
    svd = PLSSVD(n_components=1).fit(X, Y).transform(X)  # 对数据进行拟合和变换，使用单个主成分
    reg = PLSRegression(n_components=1).fit(X, Y).transform(X)  # 对数据进行拟合和变换，使用单个主成分
    canonical = PLSCanonical(n_components=1).fit(X, Y).transform(X)  # 对数据进行拟合和变换，使用单个主成分

    rtol = 1e-3
    # 设置 atol 是因为某些元素非常接近零
    assert_allclose(svd, reg, atol=reg.max() * rtol)  # 断言两个数组的所有元素在给定容差下相等
    assert_allclose(svd, canonical, atol=canonical.max() * rtol)


def test_svd_flip_1d():
    # Make sure svd_flip_1d is equivalent to svd_flip
    u = np.array([1, -4, 2])
    v = np.array([1, 2, 3])

    u_expected, v_expected = svd_flip(u.reshape(-1, 1), v.reshape(1, -1))  # 获取 u, v 的一维 SVD 翻转后的结果
    _svd_flip_1d(u, v)  # 原地修改 u, v 的方向使其一致

    assert_allclose(u, u_expected.ravel())  # 断言两个数组的所有元素在给定容差下相等
    assert_allclose(u, [-1, 4, -2])  # 断言 u 的值与预期一致

    assert_allclose(v, v_expected.ravel())  # 断言两个数组的所有元素在给定容差下相等
    assert_allclose(v, [-1, -2, -3])  # 断言 v 的值与预期一致


def test_loadings_converges(global_random_seed):
    """Test that CCA converges. Non-regression test for #19549."""
    # 使用 make_regression 函数生成一个具有200个样本，20个特征和20个目标的回归数据集，使用全局随机种子来确保结果可重复性
    X, y = make_regression(
        n_samples=200, n_features=20, n_targets=20, random_state=global_random_seed
    )

    # 创建一个CCA对象，指定10个主成分和最大迭代次数为500
    cca = CCA(n_components=10, max_iter=500)

    # 使用警告过滤器捕获收敛警告，确保在收敛警告时抛出异常
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        
        # 对CCA模型进行拟合，以寻找X和y之间的最佳线性投影
        cca.fit(X, y)

    # 断言：所有的X的加载值都应该小于1的绝对值，表明加载值收敛到合理的范围内
    assert np.all(np.abs(cca.x_loadings_) < 1)
# 检查当 y 是常数时是否发出警告。非回归测试 #19831
def test_pls_constant_y():
    rng = np.random.RandomState(42)  # 创建具有确定种子的随机数生成器
    x = rng.rand(100, 3)  # 生成形状为 (100, 3) 的随机数组 x
    y = np.zeros(100)  # 创建长度为 100 的全零数组 y

    pls = PLSRegression()  # 创建一个 PLSRegression 对象

    msg = "y residual is constant at iteration"
    with pytest.warns(UserWarning, match=msg):  # 检查是否发出特定警告信息
        pls.fit(x, y)  # 使用 x, y 训练 PLSRegression 模型

    assert_allclose(pls.x_rotations_, 0)  # 检查属性 x_rotations_ 是否接近于零


# 使用参数化测试检查 coef_ 属性的形状。非回归测试链接 #12410
@pytest.mark.parametrize("PLSEstimator", [PLSRegression, PLSCanonical, CCA])
def test_pls_coef_shape(PLSEstimator):
    d = load_linnerud()  # 加载 linnerud 数据集
    X = d.data  # 提取数据集的数据部分
    Y = d.target  # 提取数据集的目标部分

    pls = PLSEstimator(copy=True).fit(X, Y)  # 使用指定的估计器拟合数据

    n_targets, n_features = Y.shape[1], X.shape[1]  # 计算目标和特征的数量
    assert pls.coef_.shape == (n_targets, n_features)  # 断言 coef_ 的形状是否符合预期


# 使用参数化测试检查预测函数的行为
@pytest.mark.parametrize("scale", [True, False])
@pytest.mark.parametrize("PLSEstimator", [PLSRegression, PLSCanonical, CCA])
def test_pls_prediction(PLSEstimator, scale):
    d = load_linnerud()  # 加载 linnerud 数据集
    X = d.data  # 提取数据集的数据部分
    Y = d.target  # 提取数据集的目标部分

    pls = PLSEstimator(copy=True, scale=scale).fit(X, Y)  # 使用指定的估计器拟合数据

    Y_pred = pls.predict(X, copy=True)  # 进行预测

    y_mean = Y.mean(axis=0)  # 计算目标的均值
    X_trans = X - X.mean(axis=0)  # 中心化输入数据

    assert_allclose(pls.intercept_, y_mean)  # 检查截距是否接近目标的均值
    assert_allclose(Y_pred, X_trans @ pls.coef_.T + pls.intercept_)  # 检查预测值是否接近预期值


# 使用参数化测试检查 get_feature_names_out 方法的行为
@pytest.mark.parametrize("Klass", [CCA, PLSSVD, PLSRegression, PLSCanonical])
def test_pls_feature_names_out(Klass):
    X, Y = load_linnerud(return_X_y=True)  # 加载 linnerud 数据集的数据和目标

    est = Klass().fit(X, Y)  # 使用指定的估计器拟合数据
    names_out = est.get_feature_names_out()  # 获取输出特征的名称

    class_name_lower = Klass.__name__.lower()  # 获取估计器类名的小写形式
    expected_names_out = np.array(
        [f"{class_name_lower}{i}" for i in range(est.x_weights_.shape[1])],  # 生成预期的输出名称列表
        dtype=object,
    )
    assert_array_equal(names_out, expected_names_out)  # 断言输出的名称是否符合预期


# 使用参数化测试检查 cross_decomposition 模块中的 set_output 方法
@pytest.mark.parametrize("Klass", [CCA, PLSSVD, PLSRegression, PLSCanonical])
def test_pls_set_output(Klass):
    pd = pytest.importorskip("pandas")  # 导入并检查 pandas 库是否可用
    X, Y = load_linnerud(return_X_y=True, as_frame=True)  # 加载 linnerud 数据集的数据和目标，并作为 DataFrame 返回

    est = Klass().set_output(transform="pandas").fit(X, Y)  # 使用指定的估计器拟合数据并设置输出为 pandas 格式
    X_trans, y_trans = est.transform(X, Y)  # 进行数据转换

    assert isinstance(y_trans, np.ndarray)  # 断言 y_trans 是否为 numpy 数组
    assert isinstance(X_trans, pd.DataFrame)  # 断言 X_trans 是否为 pandas DataFrame
    assert_array_equal(X_trans.columns, est.get_feature_names_out())  # 断言 DataFrame 的列是否与预期的特征名称相匹配


# 检查当使用 1 维 y 进行拟合时，预测结果是否也是 1 维的。非回归测试 #26549
def test_pls_regression_fit_1d_y():
    X = np.array([[1, 1], [2, 4], [3, 9], [4, 16], [5, 25], [6, 36]])  # 创建输入特征 X
    y = np.array([2, 6, 12, 20, 30, 42])  # 创建目标值 y
    expected = y.copy()  # 复制目标值作为预期结果

    plsr = PLSRegression().fit(X, y)  # 使用 PLSRegression 对象拟合数据
    y_pred = plsr.predict(X)  # 进行预测

    assert y_pred.shape == expected.shape  # 断言预测结果的形状是否与预期一致

    # 检查该功能是否适用于 VotingRegressor
    lr = LinearRegression().fit(X, y)  # 使用线性回归模型拟合数据
    # 创建一个投票回归器对象 `vr`，由线性回归器 `lr` 和偏最小二乘回归器 `plsr` 组成
    vr = VotingRegressor([("lr", lr), ("plsr", plsr)])
    # 使用训练数据 `X` 和标签 `y` 对投票回归器进行训练，并进行预测得到预测结果 `y_pred`
    y_pred = vr.fit(X, y).predict(X)
    # 断言预测结果 `y_pred` 的形状与期望的形状 `expected.shape` 相同
    assert y_pred.shape == expected.shape
    # 断言预测结果 `y_pred` 与期望结果 `expected` 在数值上近似相等
    assert_allclose(y_pred, expected)
# 测试 PLSRegression 类在使用 scale=True 时，确保系数使用了 X 和 Y 的标准差

def test_pls_regression_scaling_coef():
    """Check that when using `scale=True`, the coefficients are using the std. dev. from
    both `X` and `Y`.
    
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/27964
    """

    # 使用种子0初始化随机数生成器
    rng = np.random.RandomState(0)
    # 生成一个形状为 (3, 5) 的系数矩阵，系数从均匀分布中抽取
    coef = rng.uniform(size=(3, 5))
    # 生成一个形状为 (30, 5) 的随机正态分布矩阵 X，每个元素乘以标准差 10
    X = rng.normal(scale=10, size=(30, 5))
    # 通过矩阵乘法生成 Y，Y = X @ coef^T
    Y = X @ coef.T

    # 创建 PLSRegression 对象，使用 5 个主成分并设置 scale=True，对 X 和 Y 进行拟合
    pls = PLSRegression(n_components=5, scale=True).fit(X, Y)
    # 断言拟合后的系数与预期的系数 coef 相近
    assert_allclose(pls.coef_, coef)

    # 断言能够从 X 预测出 Y
    assert_allclose(pls.predict(X), Y)


# TODO(1.7): Remove
# 参数化测试函数，测试在使用 Y 参数而不是 y 参数时是否显示警告消息
@pytest.mark.parametrize("Klass", [PLSRegression, CCA, PLSSVD, PLSCanonical])
def test_pls_fit_warning_on_deprecated_Y_argument(Klass):
    # Test warning message is shown when using Y instead of y

    # 加载 linnerud 数据集
    d = load_linnerud()
    X = d.data
    Y = d.target
    y = d.target

    # 定义警告消息内容
    msg = "`Y` is deprecated in 1.5 and will be removed in 1.7. Use `y` instead."

    # 断言使用 Y 参数时会显示 FutureWarning 警告消息
    with pytest.warns(FutureWarning, match=msg):
        Klass().fit(X=X, Y=Y)

    # 定义错误消息内容
    err_msg1 = "Cannot use both `y` and `Y`. Use only `y` as `Y` is deprecated."

    # 断言同时使用 y 和 Y 参数会引发 ValueError 错误
    with (
        pytest.warns(FutureWarning, match=msg),
        pytest.raises(ValueError, match=err_msg1),
    ):
        Klass().fit(X, y, Y)

    # 定义错误消息内容
    err_msg2 = "y is required."

    # 断言没有提供 y 参数时会引发 ValueError 错误
    with pytest.raises(ValueError, match=err_msg2):
        Klass().fit(X)


# TODO(1.7): Remove
# 参数化测试函数，测试在使用 Y 参数而不是 y 参数时是否显示警告消息
@pytest.mark.parametrize("Klass", [PLSRegression, CCA, PLSSVD, PLSCanonical])
def test_pls_transform_warning_on_deprecated_Y_argument(Klass):
    # Test warning message is shown when using Y instead of y

    # 加载 linnerud 数据集
    d = load_linnerud()
    X = d.data
    Y = d.target
    y = d.target

    # 创建 Klass 模型对象，并使用 y 参数对 X 进行拟合
    plsr = Klass().fit(X, y)

    # 定义警告消息内容
    msg = "`Y` is deprecated in 1.5 and will be removed in 1.7. Use `y` instead."

    # 断言在使用 Y 参数进行数据转换时会显示 FutureWarning 警告消息
    with pytest.warns(FutureWarning, match=msg):
        plsr.transform(X=X, Y=Y)

    # 定义错误消息内容
    err_msg1 = "Cannot use both `y` and `Y`. Use only `y` as `Y` is deprecated."

    # 断言同时使用 y 和 Y 参数进行数据转换会引发 ValueError 错误
    with (
        pytest.warns(FutureWarning, match=msg),
        pytest.raises(ValueError, match=err_msg1),
    ):
        plsr.transform(X, y, Y)


# TODO(1.7): Remove
# 参数化测试函数，测试在使用 Y 参数而不是 y 参数时是否显示警告消息
@pytest.mark.parametrize("Klass", [PLSRegression, CCA, PLSCanonical])
def test_pls_inverse_transform_warning_on_deprecated_Y_argument(Klass):
    # Test warning message is shown when using Y instead of y

    # 加载 linnerud 数据集
    d = load_linnerud()
    X = d.data
    y = d.target

    # 创建 Klass 模型对象，并使用 y 参数对 X 进行拟合
    plsr = Klass().fit(X, y)

    # 对 X 和 y 进行数据转换
    X_transformed, y_transformed = plsr.transform(X, y)

    # 定义警告消息内容
    msg = "`Y` is deprecated in 1.5 and will be removed in 1.7. Use `y` instead."

    # 断言在使用 Y 参数进行逆转换时会显示 FutureWarning 警告消息
    with pytest.warns(FutureWarning, match=msg):
        plsr.inverse_transform(X=X_transformed, Y=y_transformed)

    # 定义错误消息内容
    err_msg1 = "Cannot use both `y` and `Y`. Use only `y` as `Y` is deprecated."
    # 使用 pytest 提供的上下文管理器来测试函数 plsr.inverse_transform 的行为：
    # 1. 检查是否会发出 FutureWarning 警告，并匹配预期的警告消息 `msg`
    # 2. 检查是否会引发 ValueError 异常，并匹配预期的错误消息 `err_msg1`
    with (
        pytest.warns(FutureWarning, match=msg),  # 检查是否会发出 FutureWarning 警告
        pytest.raises(ValueError, match=err_msg1),  # 检查是否会引发 ValueError 异常
    ):
        # 调用 plsr.inverse_transform 函数，传入参数 X_transformed, y_transformed, Y=y_transformed
        plsr.inverse_transform(X=X_transformed, y=y_transformed, Y=y_transformed)
```