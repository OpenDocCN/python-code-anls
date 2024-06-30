# `D:\src\scipysrc\scikit-learn\sklearn\preprocessing\tests\test_polynomial.py`

```
import sys  # 导入sys模块，用于系统相关操作

import numpy as np  # 导入NumPy库，并使用别名np
import pytest  # 导入pytest测试框架
from numpy.testing import assert_allclose, assert_array_equal  # 导入NumPy测试工具函数
from scipy import sparse  # 导入SciPy稀疏矩阵模块
from scipy.interpolate import BSpline  # 导入SciPy的B样条插值函数
from scipy.sparse import random as sparse_random  # 导入SciPy稀疏随机矩阵生成函数

from sklearn.linear_model import LinearRegression  # 导入线性回归模型
from sklearn.pipeline import Pipeline  # 导入管道模型
from sklearn.preprocessing import (  # 导入数据预处理模块中的多个类
    KBinsDiscretizer,
    PolynomialFeatures,
    SplineTransformer,
)
from sklearn.preprocessing._csr_polynomial_expansion import (  # 导入CSR格式多项式展开相关函数
    _calc_expanded_nnz,
    _calc_total_nnz,
    _get_sizeof_LARGEST_INT_t,
)
from sklearn.utils._testing import assert_array_almost_equal  # 导入用于测试的数组近似相等函数
from sklearn.utils.fixes import (  # 导入一些修复工具函数
    CSC_CONTAINERS,
    CSR_CONTAINERS,
    parse_version,
    sp_version,
)

@pytest.mark.parametrize("est", (PolynomialFeatures, SplineTransformer))
def test_polynomial_and_spline_array_order(est):
    """测试多项式特征和样条变换的数组顺序是否正确输出。"""
    X = np.arange(10).reshape(5, 2)

    def is_c_contiguous(a):
        return np.isfortran(a.T)

    assert is_c_contiguous(est().fit_transform(X))  # 断言多项式特征或样条变换的结果数组是C连续的
    assert is_c_contiguous(est(order="C").fit_transform(X))  # 断言指定C顺序后的结果数组是C连续的
    assert np.isfortran(est(order="F").fit_transform(X))  # 断言指定Fortran顺序后的结果数组是Fortran连续的


@pytest.mark.parametrize(
    "params, err_msg",
    [
        ({"knots": [[1]]}, r"Number of knots, knots.shape\[0\], must be >= 2."),  # 参数化测试：节点数少于2时应抛出异常
        ({"knots": [[1, 1], [2, 2]]}, r"knots.shape\[1\] == n_features is violated"),  # 参数化测试：节点矩阵列数不符合要求时应抛出异常
        ({"knots": [[1], [0]]}, "knots must be sorted without duplicates."),  # 参数化测试：节点必须按顺序且无重复
    ],
)
def test_spline_transformer_input_validation(params, err_msg):
    """测试SplineTransformer对无效输入的异常处理是否正确。"""
    X = [[1], [2]]

    with pytest.raises(ValueError, match=err_msg):
        SplineTransformer(**params).fit(X)  # 断言在给定参数下，SplineTransformer应该抛出特定异常


@pytest.mark.parametrize("extrapolation", ["continue", "periodic"])
def test_spline_transformer_integer_knots(extrapolation):
    """测试SplineTransformer是否接受整数值节点位置。"""
    X = np.arange(20).reshape(10, 2)
    knots = [[0, 1], [1, 2], [5, 5], [11, 10], [12, 11]]
    _ = SplineTransformer(
        degree=3, knots=knots, extrapolation=extrapolation
    ).fit_transform(X)  # 测试SplineTransformer是否能够处理整数值节点位置的情况


def test_spline_transformer_feature_names():
    """测试SplineTransformer是否能正确生成特征名称。"""
    X = np.arange(20).reshape(10, 2)
    splt = SplineTransformer(n_knots=3, degree=3, include_bias=True).fit(X)
    feature_names = splt.get_feature_names_out()
    assert_array_equal(
        feature_names,
        [
            "x0_sp_0",
            "x0_sp_1",
            "x0_sp_2",
            "x0_sp_3",
            "x0_sp_4",
            "x1_sp_0",
            "x1_sp_1",
            "x1_sp_2",
            "x1_sp_3",
            "x1_sp_4",
        ],
    )

    splt = SplineTransformer(n_knots=3, degree=3, include_bias=False).fit(X)
    feature_names = splt.get_feature_names_out(["a", "b"])  # 测试SplineTransformer在指定自定义名称时的特征生成
    # 使用 assert_array_equal 函数比较 feature_names 和给定的列表是否相等
    assert_array_equal(
        feature_names,
        [
            "a_sp_0",  # 第一个元素应该是 "a_sp_0"
            "a_sp_1",  # 第二个元素应该是 "a_sp_1"
            "a_sp_2",  # 第三个元素应该是 "a_sp_2"
            "a_sp_3",  # 第四个元素应该是 "a_sp_3"
            "b_sp_0",  # 第五个元素应该是 "b_sp_0"
            "b_sp_1",  # 第六个元素应该是 "b_sp_1"
            "b_sp_2",  # 第七个元素应该是 "b_sp_2"
            "b_sp_3",  # 第八个元素应该是 "b_sp_3"
        ],
    )
@pytest.mark.parametrize(
    "extrapolation",
    ["constant", "linear", "continue", "periodic"],
)
@pytest.mark.parametrize("degree", [2, 3])
def test_split_transform_feature_names_extrapolation_degree(extrapolation, degree):
    """Test feature names are correct for different extrapolations and degree.

    Non-regression test for gh-25292.
    """
    # 创建一个测试用的输入矩阵 X，包含20个元素，reshape成10x2的矩阵
    X = np.arange(20).reshape(10, 2)
    # 使用 SplineTransformer 对象，设置 degree 和 extrapolation 参数，对 X 进行拟合
    splt = SplineTransformer(degree=degree, extrapolation=extrapolation).fit(X)
    # 获取转换后的特征名称
    feature_names = splt.get_feature_names_out(["a", "b"])
    # 断言特征名称列表长度与输出特征数目一致
    assert len(feature_names) == splt.n_features_out_

    # 对 X 进行转换
    X_trans = splt.transform(X)
    # 断言转换后的数据形状第二维度与特征名称列表长度一致
    assert X_trans.shape[1] == len(feature_names)


@pytest.mark.parametrize("degree", range(1, 5))
@pytest.mark.parametrize("n_knots", range(3, 5))
@pytest.mark.parametrize("knots", ["uniform", "quantile"])
@pytest.mark.parametrize("extrapolation", ["constant", "periodic"])
def test_spline_transformer_unity_decomposition(degree, n_knots, knots, extrapolation):
    """Test that B-splines are indeed a decomposition of unity.

    Splines basis functions must sum up to 1 per row, if we stay in between boundaries.
    """
    # 创建一个从0到1的等间隔的包含100个点的列向量 X
    X = np.linspace(0, 1, 100)[:, None]
    # 构造包含边界0和1的训练集 X_train
    X_train = np.r_[[[0]], X[::2, :], [[1]]]
    # 构造测试集 X_test
    X_test = X[1::2, :]

    if extrapolation == "periodic":
        n_knots = n_knots + degree  # 对于周期性样条，要求 degree < n_knots

    # 创建 SplineTransformer 对象，设置各种参数
    splt = SplineTransformer(
        n_knots=n_knots,
        degree=degree,
        knots=knots,
        include_bias=True,
        extrapolation=extrapolation,
    )
    # 在训练集上拟合 SplineTransformer
    splt.fit(X_train)
    # 对于 X_train 和 X_test 断言基函数的每行和为1，以验证样条基函数确实是单位分解
    for X in [X_train, X_test]:
        assert_allclose(np.sum(splt.transform(X), axis=1), 1)


@pytest.mark.parametrize(["bias", "intercept"], [(True, False), (False, True)])
def test_spline_transformer_linear_regression(bias, intercept):
    """Test that B-splines fit a sinusodial curve pretty well."""
    # 创建一个包含100个点的列向量 X
    X = np.linspace(0, 10, 100)[:, None]
    # 创建对应的正弦曲线作为目标值 y，避免零值
    y = np.sin(X[:, 0]) + 2
    # 创建 Pipeline 对象，包含 SplineTransformer 和线性回归器
    pipe = Pipeline(
        steps=[
            (
                "spline",
                SplineTransformer(
                    n_knots=15,
                    degree=3,
                    include_bias=bias,
                    extrapolation="constant",
                ),
            ),
            ("ols", LinearRegression(fit_intercept=intercept)),
        ]
    )
    # 在 X 和 y 上拟合 Pipeline
    pipe.fit(X, y)
    # 断言预测值和真实值非常接近
    assert_allclose(pipe.predict(X), y, rtol=1e-3)


@pytest.mark.parametrize(
    ["knots", "n_knots", "sample_weight", "expected_knots"],
    # 定义一个包含多个元组的列表，每个元组描述了不同的分箱方式和相关参数
    [
        # 第一个元组描述使用均匀分箱方式，分为3个箱子，不使用预设的分箱边界，数据为一个2x2的NumPy数组
        ("uniform", 3, None, np.array([[0, 2], [3, 8], [6, 14]])),
        # 第二个元组描述使用均匀分箱方式，分为3个箱子，使用指定的分箱边界数组，数据为一个3x2的NumPy数组
        (
            "uniform",
            3,
            np.array([0, 0, 1, 1, 0, 3, 1]),
            np.array([[2, 2], [4, 8], [6, 14]]),
        ),
        # 第三个元组描述使用均匀分箱方式，分为4个箱子，不使用预设的分箱边界，数据为一个4x2的NumPy数组
        ("uniform", 4, None, np.array([[0, 2], [2, 6], [4, 10], [6, 14]])),
        # 第四个元组描述使用分位数分箱方式，分为3个箱子，不使用预设的分箱边界，数据为一个3x2的NumPy数组
        ("quantile", 3, None, np.array([[0, 2], [3, 3], [6, 14]])),
        # 第五个元组描述使用分位数分箱方式，分为3个箱子，使用指定的分箱边界数组，数据为一个3x2的NumPy数组
        (
            "quantile",
            3,
            np.array([0, 0, 1, 1, 0, 3, 1]),
            np.array([[2, 2], [5, 8], [6, 14]]),
        ),
    ],
# 定义测试函数，用于测试 SplineTransformer 类中的 _get_base_knot_positions 方法
def test_spline_transformer_get_base_knot_positions(
    knots, n_knots, sample_weight, expected_knots
):
    """Check the behaviour to find knot positions with and without sample_weight."""
    # 创建示例数据集 X
    X = np.array([[0, 2], [0, 2], [2, 2], [3, 3], [4, 6], [5, 8], [6, 14]])
    # 调用 _get_base_knot_positions 方法计算基础结点位置
    base_knots = SplineTransformer._get_base_knot_positions(
        X=X, knots=knots, n_knots=n_knots, sample_weight=sample_weight
    )
    # 使用 assert_allclose 断言检查计算结果与期望值的接近程度
    assert_allclose(base_knots, expected_knots)


# 使用 pytest 的参数化装饰器来定义多组参数化测试
@pytest.mark.parametrize(["bias", "intercept"], [(True, False), (False, True)])
def test_spline_transformer_periodic_linear_regression(bias, intercept):
    """Test that B-splines fit a periodic curve pretty well."""

    # 定义周期曲线函数 f(x)
    # "+ 3" 是为了避免在 assert_allclose 中出现值为 0 的情况
    def f(x):
        return np.sin(2 * np.pi * x) - np.sin(8 * np.pi * x) + 3

    # 创建输入数据 X
    X = np.linspace(0, 1, 101)[:, None]
    
    # 创建管道对象，包含 SplineTransformer 和 LinearRegression
    pipe = Pipeline(
        steps=[
            (
                "spline",
                SplineTransformer(
                    n_knots=20,
                    degree=3,
                    include_bias=bias,
                    extrapolation="periodic",
                ),
            ),
            ("ols", LinearRegression(fit_intercept=intercept)),
        ]
    )
    
    # 在数据 X 上拟合管道
    pipe.fit(X, f(X[:, 0]))

    # 生成更大的数据数组以检查周期外推
    X_ = np.linspace(-1, 2, 301)[:, None]
    
    # 对扩展数据进行预测
    predictions = pipe.predict(X_)
    
    # 使用 assert_allclose 断言检查预测值与期望值的接近程度，指定容差
    assert_allclose(predictions, f(X_[:, 0]), atol=0.01, rtol=0.01)
    
    # 使用 assert_allclose 断言检查前 100 个预测值和后 100 个预测值的接近程度
    assert_allclose(predictions[0:100], predictions[100:200], rtol=1e-3)


# 测试 SplineTransformer 中周期样条回溯的情况
def test_spline_transformer_periodic_spline_backport():
    """Test that the backport of extrapolate="periodic" works correctly"""
    # 创建输入数据 X
    X = np.linspace(-2, 3.5, 10)[:, None]
    degree = 2

    # 使用 SplineTransformer 中的周期外推回溯功能
    transformer = SplineTransformer(
        degree=degree, extrapolation="periodic", knots=[[-1.0], [0.0], [1.0]]
    )
    
    # 对数据 X 进行变换
    Xt = transformer.fit_transform(X)

    # 使用 BSpline 进行周期外推
    coef = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    spl = BSpline(np.arange(-3, 4), coef, degree, "periodic")
    Xspl = spl(X[:, 0])
    
    # 使用 assert_allclose 断言检查变换结果与 BSpline 计算结果的接近程度
    assert_allclose(Xt, Xspl)


# 测试移动结点是否导致相同的转换结果（除了排列）
def test_spline_transformer_periodic_splines_periodicity():
    """Test if shifted knots result in the same transformation up to permutation."""
    # 创建输入数据 X
    X = np.linspace(0, 10, 101)[:, None]

    # 创建两个使用不同结点的 SplineTransformer 实例
    transformer_1 = SplineTransformer(
        degree=3,
        extrapolation="periodic",
        knots=[[0.0], [1.0], [3.0], [4.0], [5.0], [8.0]],
    )

    transformer_2 = SplineTransformer(
        degree=3,
        extrapolation="periodic",
        knots=[[1.0], [3.0], [4.0], [5.0], [8.0], [9.0]],
    )

    # 对数据 X 分别进行转换
    Xt_1 = transformer_1.fit_transform(X)
    Xt_2 = transformer_2.fit_transform(X)
    
    # 使用 assert_allclose 断言检查两次转换结果在经过重新排列后的接近程度
    assert_allclose(Xt_1, Xt_2[:, [4, 0, 1, 2, 3]])


# 测试样条转换在首尾结点处的平滑性
@pytest.mark.parametrize("degree", [3, 5])
def test_spline_transformer_periodic_splines_smoothness(degree):
    """Test that spline transformation is smooth at first / last knot."""
    # 创建一个包含从 -2 到 10 的等间距的 10000 个点的一维数组，形状为 (10000, 1)
    X = np.linspace(-2, 10, 10_000)[:, None]

    # 使用 SplineTransformer 进行数据转换
    # degree 参数指定样条插值的阶数
    # extrapolation 参数设置为 "periodic" 表示周期外推
    # knots 参数指定插值节点的位置
    transformer = SplineTransformer(
        degree=degree,
        extrapolation="periodic",
        knots=[[0.0], [1.0], [3.0], [4.0], [5.0], [8.0]],
    )
    # 对输入数据 X 进行转换
    Xt = transformer.fit_transform(X)

    # 计算输入数据 X 的间隔
    delta = (X.max() - X.min()) / len(X)
    # 设置容差 tol 为 10 倍的间隔 delta

    tol = 10 * delta

    # 初始化 dXt 为 Xt
    dXt = Xt

    # 对每个阶数从 1 到 degree 的插值进行检查
    for d in range(1, degree + 1):
        # 计算 dXt 的数值差分
        diff = np.diff(dXt, axis=0)
        # 断言数值差分的最大绝对值小于 tol，以确保连续性
        assert np.abs(diff).max() < tol

        # 计算 dXt 的 d-th 数值导数
        dXt = diff / delta

    # 检查 degree 阶数的插值在节点处是否 `degree` 次连续可微
    # 如果不是，degree+1 阶数的数值导数在节点处应该有尖峰
    diff = np.diff(dXt, axis=0)
    # 断言数值差分的最大绝对值大于 1，以确保存在尖峰
    assert np.abs(diff).max() > 1
# 使用 pytest.mark.parametrize 装饰器为 test_spline_transformer_extrapolation 函数添加参数化测试用例
@pytest.mark.parametrize(["bias", "intercept"], [(True, False), (False, True)])
@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
def test_spline_transformer_extrapolation(bias, intercept, degree):
    """Test that B-spline extrapolation works correctly."""
    # 创建一个长度为 100 的一维数组，表示 X 轴数据范围从 -1 到 1
    X = np.linspace(-1, 1, 100)[:, None]
    # 将 X 转换为一维数组，作为 y 值
    y = X.squeeze()

    # 构建数据处理管道 Pipeline 对象
    pipe = Pipeline(
        [
            # 第一个步骤为 B-spline 变换器 SplineTransformer
            [
                "spline",
                SplineTransformer(
                    n_knots=4,
                    degree=degree,
                    include_bias=bias,
                    extrapolation="constant",
                ),
            ],
            # 第二个步骤为普通最小二乘线性回归 LinearRegression
            ["ols", LinearRegression(fit_intercept=intercept)],
        ]
    )
    # 对管道进行拟合
    pipe.fit(X, y)
    # 验证预测结果是否接近预期值 [-1, 1]
    assert_allclose(pipe.predict([[-10], [5]]), [-1, 1])

    # 构建另一个数据处理管道对象
    pipe = Pipeline(
        [
            # 第一个步骤为 B-spline 变换器 SplineTransformer
            [
                "spline",
                SplineTransformer(
                    n_knots=4,
                    degree=degree,
                    include_bias=bias,
                    extrapolation="linear",
                ),
            ],
            # 第二个步骤为普通最小二乘线性回归 LinearRegression
            ["ols", LinearRegression(fit_intercept=intercept)],
        ]
    )
    # 对管道进行拟合
    pipe.fit(X, y)
    # 验证预测结果是否接近预期值 [-10, 5]
    assert_allclose(pipe.predict([[-10], [5]]), [-10, 5])

    # 构建一个单独的 B-spline 变换器对象
    splt = SplineTransformer(
        n_knots=4, degree=degree, include_bias=bias, extrapolation="error"
    )
    # 对变换器进行拟合
    splt.fit(X)
    # 预期 ValueError 错误消息
    msg = "X contains values beyond the limits of the knots"
    # 使用 pytest.raises 来验证是否抛出预期的 ValueError 异常
    with pytest.raises(ValueError, match=msg):
        splt.transform([[-10]])
    with pytest.raises(ValueError, match=msg):
        splt.transform([[5]])


# 定义测试函数 test_spline_transformer_kbindiscretizer，测试 B-spline 变换器与 KBinsDiscretizer 的等效性
def test_spline_transformer_kbindiscretizer():
    """Test that a B-spline of degree=0 is equivalent to KBinsDiscretizer."""
    # 设定随机数种子
    rng = np.random.RandomState(97531)
    # 生成一个 200x1 的随机正态分布数组 X
    X = rng.randn(200).reshape(200, 1)
    # 确定分箱数和结点数
    n_bins = 5
    n_knots = n_bins + 1

    # 创建 B-spline 变换器对象
    splt = SplineTransformer(
        n_knots=n_knots, degree=0, knots="quantile", include_bias=True
    )
    # 对数据进行变换
    splines = splt.fit_transform(X)

    # 创建 KBinsDiscretizer 对象
    kbd = KBinsDiscretizer(n_bins=n_bins, encode="onehot-dense", strategy="quantile")
    # 对数据进行分箱转换
    kbins = kbd.fit_transform(X)

    # 验证两者是否近似相等，设置相对误差容忍度为 1e-13
    assert_allclose(splines, kbins, rtol=1e-13)


# 使用 pytest.mark.skipif 装饰器，条件为 scipy 版本小于 1.8.0 时跳过测试
@pytest.mark.skipif(
    sp_version < parse_version("1.8.0"),
    reason="The option `sparse_output` is available as of scipy 1.8.0",
)
# 为 test_spline_transformer_sparse_output 函数添加参数化测试用例
@pytest.mark.parametrize("degree", range(1, 3))
@pytest.mark.parametrize("knots", ["uniform", "quantile"])
@pytest.mark.parametrize(
    "extrapolation", ["error", "constant", "linear", "continue", "periodic"]
)
@pytest.mark.parametrize("include_bias", [False, True])
def test_spline_transformer_sparse_output(
    degree, knots, extrapolation, include_bias, global_random_seed
):
    # 设定随机数种子
    rng = np.random.RandomState(global_random_seed)
    # 生成一个 200x5 的随机正态分布数组 X
    X = rng.randn(200).reshape(40, 5)
    # 创建一个用于密集输出的样条变换器对象
    splt_dense = SplineTransformer(
        degree=degree,
        knots=knots,
        extrapolation=extrapolation,
        include_bias=include_bias,
        sparse_output=False,
    )
    # 创建一个用于稀疏输出的样条变换器对象
    splt_sparse = SplineTransformer(
        degree=degree,
        knots=knots,
        extrapolation=extrapolation,
        include_bias=include_bias,
        sparse_output=True,
    )

    # 使用密集输出的样条变换器对象拟合数据集 X
    splt_dense.fit(X)
    # 使用稀疏输出的样条变换器对象拟合数据集 X
    splt_sparse.fit(X)

    # 对数据集 X 进行稀疏输出的样条变换
    X_trans_sparse = splt_sparse.transform(X)
    # 对数据集 X 进行密集输出的样条变换
    X_trans_dense = splt_dense.transform(X)
    
    # 断言稀疏输出的变换结果是稀疏矩阵，并且格式为 "csr"（压缩稀疏行格式）
    assert sparse.issparse(X_trans_sparse) and X_trans_sparse.format == "csr"
    # 断言密集输出的变换结果与稀疏输出转换为密集数组后的结果非常接近
    assert_allclose(X_trans_dense, X_trans_sparse.toarray())

    # 如果设置了 "extrapolation" 为 "error"，则进行外推检查
    X_min = np.amin(X, axis=0)
    X_max = np.amax(X, axis=0)
    X_extra = np.r_[
        np.linspace(X_min - 5, X_min, 10), np.linspace(X_max, X_max + 5, 10)
    ]
    if extrapolation == "error":
        # 对于密集输出的样条变换器对象，检查是否会引发 ValueError 异常
        msg = "X contains values beyond the limits of the knots"
        with pytest.raises(ValueError, match=msg):
            splt_dense.transform(X_extra)
        # 对于稀疏输出的样条变换器对象，检查是否会引发 ValueError 异常
        msg = "Out of bounds"
        with pytest.raises(ValueError, match=msg):
            splt_sparse.transform(X_extra)
    else:
        # 否则，断言密集输出和稀疏输出的变换结果非常接近
        assert_allclose(
            splt_dense.transform(X_extra), splt_sparse.transform(X_extra).toarray()
        )
@pytest.mark.skipif(
    sp_version >= parse_version("1.8.0"),
    reason="The option `sparse_output` is available as of scipy 1.8.0",
)
# 标记为测试用例，如果 scipy 版本大于等于 1.8.0，则跳过测试，原因是 `sparse_output` 选项仅在 scipy 1.8.0 及以上版本可用
def test_spline_transformer_sparse_output_raise_error_for_old_scipy():
    """Test that SplineTransformer with sparse=True raises for scipy<1.8.0."""
    X = [[1], [2]]
    # 使用 pytest 来检测是否会抛出 ValueError 异常，并且异常消息中包含 "scipy>=1.8.0"
    with pytest.raises(ValueError, match="scipy>=1.8.0"):
        SplineTransformer(sparse_output=True).fit(X)


@pytest.mark.parametrize("n_knots", [5, 10])
@pytest.mark.parametrize("include_bias", [True, False])
@pytest.mark.parametrize("degree", [3, 4])
@pytest.mark.parametrize(
    "extrapolation", ["error", "constant", "linear", "continue", "periodic"]
)
@pytest.mark.parametrize("sparse_output", [False, True])
# 标记为测试用例，测试 SplineTransformer 转换后的特征数量是否正确
def test_spline_transformer_n_features_out(
    n_knots, include_bias, degree, extrapolation, sparse_output
):
    """Test that transform results in n_features_out_ features."""
    # 如果 sparse_output 为 True 并且 scipy 版本小于 1.8.0，则跳过测试
    if sparse_output and sp_version < parse_version("1.8.0"):
        pytest.skip("The option `sparse_output` is available as of scipy 1.8.0")

    # 创建 SplineTransformer 对象
    splt = SplineTransformer(
        n_knots=n_knots,
        degree=degree,
        include_bias=include_bias,
        extrapolation=extrapolation,
        sparse_output=sparse_output,
    )
    X = np.linspace(0, 1, 10)[:, None]
    # 对数据进行拟合
    splt.fit(X)

    # 断言转换后的特征数量是否等于 splt.n_features_out_
    assert splt.transform(X).shape[1] == splt.n_features_out_


@pytest.mark.parametrize(
    "params, err_msg",
    [
        ({"degree": (-1, 2)}, r"degree=\(min_degree, max_degree\) must"),
        ({"degree": (0, 1.5)}, r"degree=\(min_degree, max_degree\) must"),
        ({"degree": (3, 2)}, r"degree=\(min_degree, max_degree\) must"),
        ({"degree": (1, 2, 3)}, r"int or tuple \(min_degree, max_degree\)"),
    ],
)
# 标记为测试用例，测试 PolynomialFeatures 中的输入验证是否能正确抛出错误
def test_polynomial_features_input_validation(params, err_msg):
    """Test that we raise errors for invalid input in PolynomialFeatures."""
    X = [[1], [2]]

    # 使用 pytest 来检测是否会抛出 ValueError 异常，并且异常消息匹配 err_msg
    with pytest.raises(ValueError, match=err_msg):
        PolynomialFeatures(**params).fit(X)


@pytest.fixture()
# 创建一个测试夹具，返回一个包含单个特征和其三次幂的数据
def single_feature_degree3():
    X = np.arange(6)[:, np.newaxis]
    P = np.hstack([np.ones_like(X), X, X**2, X**3])
    return X, P


@pytest.mark.parametrize(
    "degree, include_bias, interaction_only, indices",
    [
        (3, True, False, slice(None, None)),
        (3, False, False, slice(1, None)),
        (3, True, True, [0, 1]),
        (3, False, True, [1]),
        ((2, 3), True, False, [0, 2, 3]),
        ((2, 3), False, False, [2, 3]),
        ((2, 3), True, True, [0]),
        ((2, 3), False, True, []),
    ],
)
@pytest.mark.parametrize("X_container", [None] + CSR_CONTAINERS + CSC_CONTAINERS)
# 标记为测试用例，测试在单个特征上使用 PolynomialFeatures 的情况，最高次为三次幂
def test_polynomial_features_one_feature(
    single_feature_degree3,
    degree,
    include_bias,
    interaction_only,
    indices,
    X_container,
):
    """Test PolynomialFeatures on single feature up to degree 3."""
    X, P = single_feature_degree3
    if X_container is not None:
        X = X_container(X)
    # 使用多项式特征转换器 PolynomialFeatures 对象进行多项式特征生成
    tf = PolynomialFeatures(
        degree=degree, include_bias=include_bias, interaction_only=interaction_only
    ).fit(X)
    # 对输入数据 X 进行多项式特征转换，得到转换后的输出 out
    out = tf.transform(X)
    # 如果 X_container 不为空，则将 out 转换为稀疏矩阵表示
    if X_container is not None:
        out = out.toarray()
    # 使用 assert_allclose 函数验证转换后的输出 out 与预期的多项式特征矩阵 P[:, indices] 是否相近
    assert_allclose(out, P[:, indices])
    # 如果多项式特征转换器 tf 生成的输出特征数量大于 0，则进行进一步的断言验证
    if tf.n_output_features_ > 0:
        # 断言 tf.powers_ 的形状应为 (输出特征数量, 输入特征数量)
        assert tf.powers_.shape == (tf.n_output_features_, tf.n_features_in_)
@pytest.fixture()
# 定义一个测试夹具，生成一个包含两个特征的 3x2 的 NumPy 数组 X 和对应的多项式特征矩阵 P
def two_features_degree3():
    X = np.arange(6).reshape((3, 2))  # 创建一个 3x2 的数组 X，内容为 [0, 1, 2, 3, 4, 5]
    x1 = X[:, :1]  # 提取 X 的第一列作为 x1，形状为 (3, 1)
    x2 = X[:, 1:]  # 提取 X 的第二列作为 x2，形状为 (3, 1)
    P = np.hstack(
        [
            x1**0 * x2**0,  # 第0项：常数项
            x1**1 * x2**0,  # 第1项：x1
            x1**0 * x2**1,  # 第2项：x2
            x1**2 * x2**0,  # 第3项：x1^2
            x1**1 * x2**1,  # 第4项：x1*x2
            x1**0 * x2**2,  # 第5项：x2^2
            x1**3 * x2**0,  # 第6项：x1^3
            x1**2 * x2**1,  # 第7项：x1^2*x2
            x1**1 * x2**2,  # 第8项：x1*x2^2
            x1**0 * x2**3,  # 第9项：x2^3
        ]
    )  # 按列将以上项连接成 P，形状为 (3, 10)
    return X, P  # 返回 X 和 P


@pytest.mark.parametrize(
    "degree, include_bias, interaction_only, indices",
    [
        (2, True, False, slice(0, 6)),  # 情况1
        (2, False, False, slice(1, 6)),  # 情况2
        (2, True, True, [0, 1, 2, 4]),  # 情况3
        (2, False, True, [1, 2, 4]),  # 情况4
        ((2, 2), True, False, [0, 3, 4, 5]),  # 情况5
        ((2, 2), False, False, [3, 4, 5]),  # 情况6
        ((2, 2), True, True, [0, 4]),  # 情况7
        ((2, 2), False, True, [4]),  # 情况8
        (3, True, False, slice(None, None)),  # 情况9
        (3, False, False, slice(1, None)),  # 情况10
        (3, True, True, [0, 1, 2, 4]),  # 情况11
        (3, False, True, [1, 2, 4]),  # 情况12
        ((2, 3), True, False, [0, 3, 4, 5, 6, 7, 8, 9]),  # 情况13
        ((2, 3), False, False, slice(3, None)),  # 情况14
        ((2, 3), True, True, [0, 4]),  # 情况15
        ((2, 3), False, True, [4]),  # 情况16
        ((3, 3), True, False, [0, 6, 7, 8, 9]),  # 情况17
        ((3, 3), False, False, [6, 7, 8, 9]),  # 情况18
        ((3, 3), True, True, [0]),  # 情况19
        ((3, 3), False, True, []),  # 情况20
    ],
)
@pytest.mark.parametrize("X_container", [None] + CSR_CONTAINERS + CSC_CONTAINERS)
# 使用参数化测试的装饰器，定义多个测试参数组合
def test_polynomial_features_two_features(
    two_features_degree3,
    degree,
    include_bias,
    interaction_only,
    indices,
    X_container,
):
    """Test PolynomialFeatures on 2 features up to degree 3."""
    X, P = two_features_degree3  # 获取两个特征的 3x2 数组 X 和多项式特征矩阵 P
    if X_container is not None:
        X = X_container(X)  # 如果指定了 X_container，则使用其对 X 进行转换

    # 使用 PolynomialFeatures 进行拟合
    tf = PolynomialFeatures(
        degree=degree, include_bias=include_bias, interaction_only=interaction_only
    ).fit(X)
    out = tf.transform(X)  # 对 X 进行转换，得到多项式特征矩阵

    if X_container is not None:
        out = out.toarray()  # 如果使用了稀疏矩阵容器，将结果转换为稠密数组

    assert_allclose(out, P[:, indices])  # 断言转换后的结果与预期的多项式特征矩阵 P 的指定列部分相等
    if tf.n_output_features_ > 0:
        assert tf.powers_.shape == (tf.n_output_features_, tf.n_features_in_)
        # 断言生成的多项式特征的幂次矩阵的形状符合预期


def test_polynomial_feature_names():
    X = np.arange(30).reshape(10, 3)  # 创建一个 10x3 的数组 X，内容为 [0, 1, ..., 29]

    # 测试 degree=2、include_bias=True 时的多项式特征名是否正确
    poly = PolynomialFeatures(degree=2, include_bias=True).fit(X)
    feature_names = poly.get_feature_names_out()
    assert_array_equal(
        ["1", "x0", "x1", "x2", "x0^2", "x0 x1", "x0 x2", "x1^2", "x1 x2", "x2^2"],
        feature_names,
    )
    assert len(feature_names) == poly.transform(X).shape[1]

    # 测试 degree=3、include_bias=False 时的多项式特征名是否正确
    poly = PolynomialFeatures(degree=3, include_bias=False).fit(X)
    feature_names = poly.get_feature_names_out(["a", "b", "c"])  # 自定义特征名
    # 断言：验证特征名称列表是否与预期相等
    assert_array_equal(
        [
            "a",
            "b",
            "c",
            "a^2",
            "a b",
            "a c",
            "b^2",
            "b c",
            "c^2",
            "a^3",
            "a^2 b",
            "a^2 c",
            "a b^2",
            "a b c",
            "a c^2",
            "b^3",
            "b^2 c",
            "b c^2",
            "c^3",
        ],
        feature_names,
    )
    # 断言：验证特征名称列表的长度是否与多项式变换后的列数相等
    assert len(feature_names) == poly.transform(X).shape[1]

    # 创建多项式特征对象，设置最高次数为2和3，不包含偏置项，并对输入数据进行拟合
    poly = PolynomialFeatures(degree=(2, 3), include_bias=False).fit(X)
    # 获取多项式特征的输出特征名称列表
    feature_names = poly.get_feature_names_out(["a", "b", "c"])
    # 断言：验证特征名称列表是否与预期相等
    assert_array_equal(
        [
            "a^2",
            "a b",
            "a c",
            "b^2",
            "b c",
            "c^2",
            "a^3",
            "a^2 b",
            "a^2 c",
            "a b^2",
            "a b c",
            "a c^2",
            "b^3",
            "b^2 c",
            "b c^2",
            "c^3",
        ],
        feature_names,
    )
    # 断言：验证特征名称列表的长度是否与多项式变换后的列数相等
    assert len(feature_names) == poly.transform(X).shape[1]

    # 创建多项式特征对象，设置最高次数为3，包含偏置项，只考虑交互作用，并对输入数据进行拟合
    poly = PolynomialFeatures(
        degree=(3, 3), include_bias=True, interaction_only=True
    ).fit(X)
    # 获取多项式特征的输出特征名称列表
    feature_names = poly.get_feature_names_out(["a", "b", "c"])
    # 断言：验证特征名称列表是否与预期相等
    assert_array_equal(["1", "a b c"], feature_names)
    # 断言：验证特征名称列表的长度是否与多项式变换后的列数相等
    assert len(feature_names) == poly.transform(X).shape[1]

    # 测试一些Unicode字符
    # 创建多项式特征对象，设置最高次数为1，包含偏置项，并对输入数据进行拟合
    poly = PolynomialFeatures(degree=1, include_bias=True).fit(X)
    # 获取多项式特征的输出特征名称列表，包括Unicode字符
    feature_names = poly.get_feature_names_out(["\u0001F40D", "\u262e", "\u05d0"])
    # 断言：验证特征名称列表是否与预期相等，包括Unicode字符
    assert_array_equal(["1", "\u0001F40D", "\u262e", "\u05d0"], feature_names)
@pytest.mark.parametrize(
    ["deg", "include_bias", "interaction_only", "dtype"],
    [  # 参数化测试的参数列表
        (1, True, False, int),       # 第一个参数组合
        (2, True, False, int),       # 第二个参数组合
        (2, True, False, np.float32),# 第三个参数组合
        (2, True, False, np.float64),# 第四个参数组合
        (3, False, False, np.float64),# 第五个参数组合
        (3, False, True, np.float64),# 第六个参数组合
        (4, False, False, np.float64),# 第七个参数组合
        (4, False, True, np.float64),# 第八个参数组合
    ],
)
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_polynomial_features_csc_X(
    deg, include_bias, interaction_only, dtype, csc_container
):
    rng = np.random.RandomState(0)  # 使用种子0初始化随机数生成器
    X = rng.randint(0, 2, (100, 2))  # 生成100行2列的随机整数矩阵
    X_csc = csc_container(X)  # 将生成的矩阵转换为指定的压缩稀疏列格式

    est = PolynomialFeatures(  # 创建多项式特征转换器对象
        deg, include_bias=include_bias, interaction_only=interaction_only
    )
    Xt_csc = est.fit_transform(X_csc.astype(dtype))  # 对转换后的数据进行拟合和转换
    Xt_dense = est.fit_transform(X.astype(dtype))  # 对原始数据进行拟合和转换

    assert sparse.issparse(Xt_csc) and Xt_csc.format == "csc"  # 断言转换后的数据为稀疏矩阵并且格式为"csc"
    assert Xt_csc.dtype == Xt_dense.dtype  # 断言稀疏格式和密集格式的数据类型一致
    assert_array_almost_equal(Xt_csc.toarray(), Xt_dense)  # 断言稀疏格式和密集格式的数据数组近似相等


@pytest.mark.parametrize(
    ["deg", "include_bias", "interaction_only", "dtype"],
    [  # 参数化测试的参数列表
        (1, True, False, int),       # 第一个参数组合
        (2, True, False, int),       # 第二个参数组合
        (2, True, False, np.float32),# 第三个参数组合
        (2, True, False, np.float64),# 第四个参数组合
        (3, False, False, np.float64),# 第五个参数组合
        (3, False, True, np.float64),# 第六个参数组合
    ],
)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_polynomial_features_csr_X(
    deg, include_bias, interaction_only, dtype, csr_container
):
    rng = np.random.RandomState(0)  # 使用种子0初始化随机数生成器
    X = rng.randint(0, 2, (100, 2))  # 生成100行2列的随机整数矩阵
    X_csr = csr_container(X)  # 将生成的矩阵转换为指定的压缩稀疏行格式

    est = PolynomialFeatures(  # 创建多项式特征转换器对象
        deg, include_bias=include_bias, interaction_only=interaction_only
    )
    Xt_csr = est.fit_transform(X_csr.astype(dtype))  # 对转换后的数据进行拟合和转换
    Xt_dense = est.fit_transform(X.astype(dtype, copy=False))  # 对原始数据进行拟合和转换

    assert sparse.issparse(Xt_csr) and Xt_csr.format == "csr"  # 断言转换后的数据为稀疏矩阵并且格式为"csr"
    assert Xt_csr.dtype == Xt_dense.dtype  # 断言稀疏格式和密集格式的数据类型一致
    assert_array_almost_equal(Xt_csr.toarray(), Xt_dense)  # 断言稀疏格式和密集格式的数据数组近似相等


@pytest.mark.parametrize("n_features", [1, 4, 5])  # 参数化测试的参数列表
@pytest.mark.parametrize(
    "min_degree, max_degree", [(0, 1), (0, 2), (1, 3), (0, 4), (3, 4)]
)  # 参数化测试的参数列表
@pytest.mark.parametrize("interaction_only", [True, False])  # 参数化测试的参数列表
@pytest.mark.parametrize("include_bias", [True, False])  # 参数化测试的参数列表
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)  # 参数化测试的参数列表
def test_num_combinations(
    n_features, min_degree, max_degree, interaction_only, include_bias, csr_container
):
    """
    Test that n_output_features_ is calculated correctly.
    """
    x = csr_container(([1], ([0], [n_features - 1])))  # 创建指定压缩稀疏行格式的数据
    est = PolynomialFeatures(  # 创建多项式特征转换器对象
        degree=max_degree,
        interaction_only=interaction_only,
        include_bias=include_bias,
    )
    est.fit(x)  # 对数据进行拟合
    num_combos = est.n_output_features_  # 获取输出特征数量

    combos = PolynomialFeatures._combinations(  # 调用静态方法生成特征组合
        n_features=n_features,
        min_degree=0,
        max_degree=max_degree,
        interaction_only=interaction_only,
        include_bias=include_bias,
    )
    # 断言：验证 num_combos 是否等于组合列表 combos 的长度
    assert num_combos == sum([1 for _ in combos])
# 使用 pytest.mark.parametrize 装饰器指定参数化测试的参数列表，用于多次运行相同测试函数
@pytest.mark.parametrize(
    ["deg", "include_bias", "interaction_only", "dtype"],  # 参数列表包括 deg, include_bias, interaction_only 和 dtype
    [
        (2, True, False, np.float32),   # 第一个参数化测试组合
        (2, True, False, np.float64),   # 第二个参数化测试组合
        (3, False, False, np.float64),  # 第三个参数化测试组合
        (3, False, True, np.float64),   # 第四个参数化测试组合
    ],
)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)  # 参数化 csr_container，使用预定义的 CSR_CONTAINERS
def test_polynomial_features_csr_X_floats(
    deg, include_bias, interaction_only, dtype, csr_container
):
    X_csr = csr_container(sparse_random(1000, 10, 0.5, random_state=0))  # 生成稀疏随机矩阵 X_csr
    X = X_csr.toarray()  # 将稀疏矩阵 X_csr 转换为密集矩阵 X

    est = PolynomialFeatures(
        deg, include_bias=include_bias, interaction_only=interaction_only
    )  # 创建多项式特征转换器 est

    # 在稀疏矩阵 X_csr 上拟合并转换为新的稀疏矩阵 Xt_csr，数据类型转换为指定 dtype
    Xt_csr = est.fit_transform(X_csr.astype(dtype))
    # 在密集矩阵 X 上拟合并转换为新的密集矩阵 Xt_dense，数据类型转换为指定 dtype
    Xt_dense = est.fit_transform(X.astype(dtype))

    # 断言确保 Xt_csr 是稀疏矩阵且格式为 "csr"
    assert sparse.issparse(Xt_csr) and Xt_csr.format == "csr"
    # 断言确保 Xt_csr 和 Xt_dense 的数据类型相同
    assert Xt_csr.dtype == Xt_dense.dtype
    # 断言确保 Xt_csr 和 Xt_dense 数值上近似相等
    assert_array_almost_equal(Xt_csr.toarray(), Xt_dense)


# 参数化测试，测试稀疏矩阵中是否包含零行的情况
@pytest.mark.parametrize(
    ["zero_row_index", "deg", "interaction_only"],
    [
        (0, 2, True),   # 第一个参数化测试组合
        (1, 2, True),   # 第二个参数化测试组合
        (2, 2, True),   # 第三个参数化测试组合
        (0, 3, True),   # 第四个参数化测试组合
        (1, 3, True),   # 第五个参数化测试组合
        (2, 3, True),   # 第六个参数化测试组合
        (0, 2, False),  # 第七个参数化测试组合
        (1, 2, False),  # 第八个参数化测试组合
        (2, 2, False),  # 第九个参数化测试组合
        (0, 3, False),  # 第十个参数化测试组合
        (1, 3, False),  # 第十一个参数化测试组合
        (2, 3, False),  # 第十二个参数化测试组合
    ],
)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)  # 参数化 csr_container，使用预定义的 CSR_CONTAINERS
def test_polynomial_features_csr_X_zero_row(
    zero_row_index, deg, interaction_only, csr_container
):
    X_csr = csr_container(sparse_random(3, 10, 1.0, random_state=0))  # 生成稀疏随机矩阵 X_csr
    X_csr[zero_row_index, :] = 0.0  # 将指定行索引的行置为零向量
    X = X_csr.toarray()  # 将稀疏矩阵 X_csr 转换为密集矩阵 X

    est = PolynomialFeatures(
        deg, include_bias=False, interaction_only=interaction_only
    )  # 创建多项式特征转换器 est，不包含偏置项

    # 在稀疏矩阵 X_csr 上拟合并转换为新的稀疏矩阵 Xt_csr
    Xt_csr = est.fit_transform(X_csr)
    # 在密集矩阵 X 上拟合并转换为新的密集矩阵 Xt_dense
    Xt_dense = est.fit_transform(X)

    # 断言确保 Xt_csr 是稀疏矩阵且格式为 "csr"
    assert sparse.issparse(Xt_csr) and Xt_csr.format == "csr"
    # 断言确保 Xt_csr 和 Xt_dense 的数据类型相同
    assert Xt_csr.dtype == Xt_dense.dtype
    # 断言确保 Xt_csr 和 Xt_dense 数值上近似相等
    assert_array_almost_equal(Xt_csr.toarray(), Xt_dense)


# 参数化测试，测试多项式特征扩展到最高阶数为 4 的情况
@pytest.mark.parametrize(
    ["include_bias", "interaction_only"],
    [(True, True), (True, False), (False, True), (False, False)],
)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)  # 参数化 csr_container，使用预定义的 CSR_CONTAINERS
def test_polynomial_features_csr_X_degree_4(
    include_bias, interaction_only, csr_container
):
    X_csr = csr_container(sparse_random(1000, 10, 0.5, random_state=0))  # 生成稀疏随机矩阵 X_csr
    X = X_csr.toarray()  # 将稀疏矩阵 X_csr 转换为密集矩阵 X

    est = PolynomialFeatures(
        4, include_bias=include_bias, interaction_only=interaction_only
    )  # 创建多项式特征转换器 est，最高阶数为 4

    # 在稀疏矩阵 X_csr 上拟合并转换为新的稀疏矩阵 Xt_csr
    Xt_csr = est.fit_transform(X_csr)
    # 在密集矩阵 X 上拟合并转换为新的密集矩阵 Xt_dense
    Xt_dense = est.fit_transform(X)

    # 断言确保 Xt_csr 是稀疏矩阵且格式为 "csr"
    assert sparse.issparse(Xt_csr) and Xt_csr.format == "csr"
    # 断言确保 Xt_csr 和 Xt_dense 的数据类型相同
    assert Xt_csr.dtype == Xt_dense.dtype
    # 断言确保 Xt_csr 和 Xt_dense 数值上近似相等
    assert_array_almost_equal(Xt_csr.toarray(), Xt_dense)
    [
        # 创建一个包含多个元组的列表，每个元组表示一个三元组 (a, b, c)，其中：
        # a 表示第一个元素，取值为 2 或 3；
        # b 表示第二个元素，取值为 1、2 或 3；
        # c 表示第三个元素，取值为 True 或 False。
        (2, 1, True),
        (2, 2, True),
        (3, 1, True),
        (3, 2, True),
        (3, 3, True),
        (2, 1, False),
        (2, 2, False),
        (3, 1, False),
        (3, 2, False),
        (3, 3, False),
    ],
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
# 使用 pytest 的 parametrize 装饰器，为 csr_container 参数化测试
def test_polynomial_features_csr_X_dim_edges(deg, dim, interaction_only, csr_container):
    # 创建稀疏矩阵 X_csr，维度为 1000 × dim，稀疏度为 0.5，随机种子为 0
    X_csr = csr_container(sparse_random(1000, dim, 0.5, random_state=0))
    # 将稀疏矩阵 X_csr 转换为密集矩阵 X
    X = X_csr.toarray()

    # 初始化多项式特征生成器，指定阶数 deg 和是否仅交互特征 interaction_only
    est = PolynomialFeatures(deg, interaction_only=interaction_only)
    # 在稀疏矩阵 X_csr 上拟合并转换数据
    Xt_csr = est.fit_transform(X_csr)
    # 在密集矩阵 X 上拟合并转换数据
    Xt_dense = est.fit_transform(X)

    # 断言转换后的 Xt_csr 是稀疏矩阵且格式为 "csr"
    assert sparse.issparse(Xt_csr) and Xt_csr.format == "csr"
    # 断言转换后的 Xt_csr 的数据类型与 Xt_dense 相同
    assert Xt_csr.dtype == Xt_dense.dtype
    # 断言转换后的 Xt_csr 和 Xt_dense 的数值近似相等
    assert_array_almost_equal(Xt_csr.toarray(), Xt_dense)


@pytest.mark.parametrize("interaction_only", [True, False])
@pytest.mark.parametrize("include_bias", [True, False])
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
# 使用 pytest 的 parametrize 装饰器，为 interaction_only, include_bias 和 csr_container 参数化测试
def test_csr_polynomial_expansion_index_overflow_non_regression(
    interaction_only, include_bias, csr_container
):
    """Check the automatic index dtype promotion to `np.int64` when needed.

    This ensures that sufficiently large input configurations get
    properly promoted to use `np.int64` for index and indptr representation
    while preserving data integrity. Non-regression test for gh-16803.

    Note that this is only possible for Python runtimes with a 64 bit address
    space. On 32 bit platforms, a `ValueError` is raised instead.
    """
    # 定义一个函数 degree_2_calc，根据 interaction_only 参数计算索引
    def degree_2_calc(d, i, j):
        if interaction_only:
            return d * i - (i**2 + 3 * i) // 2 - 1 + j
        else:
            return d * i - (i**2 + i) // 2 + j

    # 定义样本数 n_samples 和特征数 n_features
    n_samples = 13
    n_features = 120001
    # 定义数据类型为 np.float32 的数据 data
    data_dtype = np.float32
    # 创建数据数组 data，范围从 1 到 5，数据类型为 np.int64
    data = np.arange(1, 5, dtype=np.int64)
    # 创建行索引数组 row，值为 n_samples - 2 和 n_samples - 1
    row = np.array([n_samples - 2, n_samples - 2, n_samples - 1, n_samples - 1])
    # 创建列索引数组 col，值为 n_features - 2 和 n_features - 1，数据类型为 np.int64
    col = np.array(
        [n_features - 2, n_features - 1, n_features - 2, n_features - 1], dtype=np.int64
    )
    # 使用 csr_container 创建稀疏矩阵 X，指定形状、数据类型和数据
    X = csr_container(
        (data, (row, col)),
        shape=(n_samples, n_features),
        dtype=data_dtype,
    )
    # 初始化多项式特征生成器，指定 interaction_only、include_bias 和 degree=2
    pf = PolynomialFeatures(
        interaction_only=interaction_only, include_bias=include_bias, degree=2
    )

    # 计算预期的组合数目 num_combinations
    num_combinations = pf._num_combinations(
        n_features=n_features,
        min_degree=0,
        max_degree=2,
        interaction_only=pf.interaction_only,
        include_bias=pf.include_bias,
    )
    # 如果 num_combinations 超过 np.intp 的最大值，则期望抛出 ValueError
    if num_combinations > np.iinfo(np.intp).max:
        msg = (
            r"The output that would result from the current configuration would have"
            r" \d* features which is too large to be indexed"
        )
        with pytest.raises(ValueError, match=msg):
            pf.fit(X)
        return
    # 否则，对稀疏矩阵 X 进行拟合和转换
    X_trans = pf.fit_transform(X)
    # 获取 X_trans 中非零元素的行和列索引
    row_nonzero, col_nonzero = X_trans.nonzero()
    # 计算一次项特征的数量 n_degree_1_features_out
    n_degree_1_features_out = n_features + include_bias
    # 计算最大的二次项索引，根据非交互模式和列索引计算
    max_degree_2_idx = (
        degree_2_calc(n_features, col[int(not interaction_only)], col[1])
        + n_degree_1_features_out
    )

    # 处理除了最后一个样本外的所有样本的偏置项，最后一个样本将单独处理
    # 因为在它之前有不同的数据值
    data_target = [1] * (n_samples - 2) if include_bias else []
    col_nonzero_target = [0] * (n_samples - 2) if include_bias else []

    # 遍历两次数据的循环
    for i in range(2):
        x = data[2 * i]
        y = data[2 * i + 1]
        x_idx = col[2 * i]
        y_idx = col[2 * i + 1]
        if include_bias:
            data_target.append(1)
            col_nonzero_target.append(0)
        # 扩展数据目标和非零列目标列表
        data_target.extend([x, y])
        col_nonzero_target.extend(
            [x_idx + int(include_bias), y_idx + int(include_bias)]
        )
        if not interaction_only:
            # 如果不是仅交互模式，则添加二次项和交互项
            data_target.extend([x * x, x * y, y * y])
            col_nonzero_target.extend(
                [
                    degree_2_calc(n_features, x_idx, x_idx) + n_degree_1_features_out,
                    degree_2_calc(n_features, x_idx, y_idx) + n_degree_1_features_out,
                    degree_2_calc(n_features, y_idx, y_idx) + n_degree_1_features_out,
                ]
            )
        else:
            # 如果是仅交互模式，则只添加交互项
            data_target.extend([x * y])
            col_nonzero_target.append(
                degree_2_calc(n_features, x_idx, y_idx) + n_degree_1_features_out
            )

    # 计算每行的非零元素数量
    nnz_per_row = int(include_bias) + 3 + 2 * int(not interaction_only)

    # 断言确保输出特征数等于最大二次项索引加一
    assert pf.n_output_features_ == max_degree_2_idx + 1
    # 断言确保转换后的数据类型与指定数据类型一致
    assert X_trans.dtype == data_dtype
    # 断言确保转换后的形状与指定形状一致
    assert X_trans.shape == (n_samples, max_degree_2_idx + 1)
    # 断言确保行指针和列索引数据类型为 np.int64
    assert X_trans.indptr.dtype == X_trans.indices.dtype == np.int64
    # 确保实际需要进行 dtype 提升：
    assert X_trans.indices.max() > np.iinfo(np.int32).max

    # 设置每行的非零行索引目标列表
    row_nonzero_target = list(range(n_samples - 2)) if include_bias else []
    row_nonzero_target.extend(
        [n_samples - 2] * nnz_per_row + [n_samples - 1] * nnz_per_row
    )

    # 断言确保转换后的数据与目标数据一致
    assert_allclose(X_trans.data, data_target)
    # 断言确保行非零目标与目标列表一致
    assert_array_equal(row_nonzero, row_nonzero_target)
    # 断言确保列非零目标与目标列表一致
    assert_array_equal(col_nonzero, col_nonzero_target)
@pytest.mark.parametrize(
    "degree, n_features",
    [
        # 需要在 interaction_only=False 时提升为 int64
        (2, 65535),
        (3, 2344),
        # 这确保了在计算输出列时，中间操作会溢出 C-long，因此需要使用 python-longs。
        (2, int(np.sqrt(np.iinfo(np.int64).max) + 1)),
        (3, 65535),
        # 这个测试用例测试了溢出检查的第二个条件，考虑了 `n_features` 的值。
        (2, int(np.sqrt(np.iinfo(np.int64).max))),
    ],
)
@pytest.mark.parametrize("interaction_only", [True, False])
@pytest.mark.parametrize("include_bias", [True, False])
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_csr_polynomial_expansion_index_overflow(
    degree, n_features, interaction_only, include_bias, csr_container
):
    """Tests known edge-cases to the dtype promotion strategy and custom
    Cython code, including a current bug in the upstream
    `scipy.sparse.hstack`.
    """
    data = [1.0]
    row = [0]
    col = [n_features - 1]

    # First degree index
    # 第一次度的索引
    expected_indices = [
        n_features - 1 + int(include_bias),
    ]
    # Second degree index
    # 第二次度的索引
    expected_indices.append(n_features * (n_features + 1) // 2 + expected_indices[0])
    # Third degree index
    # 第三次度的索引
    expected_indices.append(
        n_features * (n_features + 1) * (n_features + 2) // 6 + expected_indices[1]
    )

    X = csr_container((data, (row, col)))
    pf = PolynomialFeatures(
        interaction_only=interaction_only, include_bias=include_bias, degree=degree
    )

    # Calculate the number of combinations a-priori, and if needed check for
    # the correct ValueError and terminate the test early.
    # 预先计算组合数，并在必要时检查正确的 ValueError 并提前终止测试。
    num_combinations = pf._num_combinations(
        n_features=n_features,
        min_degree=0,
        max_degree=degree,
        interaction_only=pf.interaction_only,
        include_bias=pf.include_bias,
    )
    if num_combinations > np.iinfo(np.intp).max:
        msg = (
            r"The output that would result from the current configuration would have"
            r" \d* features which is too large to be indexed"
        )
        with pytest.raises(ValueError, match=msg):
            pf.fit(X)
        return

    # In SciPy < 1.8, a bug occurs when an intermediate matrix in
    # `to_stack` in `hstack` fits within int32 however would require int64 when
    # combined with all previous matrices in `to_stack`.
    # 在 SciPy < 1.8 中，当 `to_stack` 中的中间矩阵适合于 int32 时，但与 `to_stack` 中的所有先前矩阵组合时需要 int64 时会出现 bug。
    # 检查 SciPy 版本是否低于 1.8.0，以确定是否存在已知 bug
    if sp_version < parse_version("1.8.0"):
        # 初始化 bug 标志为 False
        has_bug = False
        # 获取 np.int32 的最大值
        max_int32 = np.iinfo(np.int32).max
        # 计算累积大小，包括特征数和是否包含偏置
        cumulative_size = n_features + include_bias
        # 遍历需要计算的多项式次数
        for deg in range(2, degree + 1):
            # 计算当前多项式次数下的最大 indptr
            max_indptr = _calc_total_nnz(X.indptr, interaction_only, deg)
            # 计算当前多项式次数下的最大 indices
            max_indices = _calc_expanded_nnz(n_features, interaction_only, deg) - 1
            # 更新累积大小
            cumulative_size += max_indices + 1
            # 检查是否需要使用 np.int64 类型，如果超过 np.int32 的范围
            needs_int64 = max(max_indices, max_indptr) > max_int32
            # 如果累积大小超过 np.int32 的范围但不需要使用 np.int64，则存在 bug
            has_bug |= not needs_int64 and cumulative_size > max_int32
        # 如果存在 bug，则抛出异常并返回
        if has_bug:
            msg = r"In scipy versions `<1.8.0`, the function `scipy.sparse.hstack`"
            with pytest.raises(ValueError, match=msg):
                X_trans = pf.fit_transform(X)
            return

    # 当 `n_features>=65535` 时，可能会出现 bug，需检查特定情况
    if (
        sp_version < parse_version("1.9.2")
        and n_features == 65535
        and degree == 2
        and not interaction_only
    ):  # pragma: no cover
        # 如果满足特定条件，抛出异常并返回
        msg = r"In scipy versions `<1.9.2`, the function `scipy.sparse.hstack`"
        with pytest.raises(ValueError, match=msg):
            X_trans = pf.fit_transform(X)
        return

    # 应用多项式特征转换
    X_trans = pf.fit_transform(X)

    # 预期的数据类型取决于组合数是否超过 np.int32 的最大值
    expected_dtype = np.int64 if num_combinations > np.iinfo(np.int32).max else np.int32
    # 计算非偏置项的数量
    non_bias_terms = 1 + (degree - 1) * int(not interaction_only)
    # 预期的非零元素数量
    expected_nnz = int(include_bias) + non_bias_terms

    # 断言转换后的特征矩阵的属性和预期值相符
    assert X_trans.dtype == X.dtype
    assert X_trans.shape == (1, pf.n_output_features_)
    assert X_trans.indptr.dtype == X_trans.indices.dtype == expected_dtype
    assert X_trans.nnz == expected_nnz

    # 如果包含偏置，断言第一个元素应接近 1.0
    if include_bias:
        assert X_trans[0, 0] == pytest.approx(1.0)

    # 断言非偏置项的位置应接近 1.0
    for idx in range(non_bias_terms):
        assert X_trans[0, expected_indices[idx]] == pytest.approx(1.0)

    # 计算偏置的偏移量
    offset = interaction_only * n_features
    # 如果是三次多项式，需进一步计算偏移量
    if degree == 3:
        offset *= 1 + n_features

    # 断言多项式特征转换后的输出特征数量与预期一致
    assert pf.n_output_features_ == expected_indices[degree - 1] + 1 - offset
@pytest.mark.parametrize("interaction_only", [True, False])
@pytest.mark.parametrize("include_bias", [True, False])
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_csr_polynomial_expansion_too_large_to_index(
    interaction_only, include_bias, csr_container
):
    # 定义测试函数，用于测试在指定条件下的 CSR 矩阵多项式展开是否会导致索引超出范围的错误

    # 设定一个接近 int64 最大值的特征数
    n_features = np.iinfo(np.int64).max // 2
    data = [1.0]
    row = [0]
    col = [n_features - 1]
    # 创建 CSR 格式矩阵 X
    X = csr_container((data, (row, col)))
    
    # 创建 PolynomialFeatures 对象，指定参数
    pf = PolynomialFeatures(
        interaction_only=interaction_only, include_bias=include_bias, degree=(2, 2)
    )
    
    # 定义匹配的错误信息，检测是否抛出预期的 ValueError 异常
    msg = (
        r"The output that would result from the current configuration would have \d*"
        r" features which is too large to be indexed"
    )
    
    # 使用 pytest 的 raises 方法检测是否抛出 ValueError 异常，并匹配错误信息
    with pytest.raises(ValueError, match=msg):
        pf.fit(X)
    with pytest.raises(ValueError, match=msg):
        pf.fit_transform(X)


@pytest.mark.parametrize("sparse_container", CSR_CONTAINERS + CSC_CONTAINERS)
def test_polynomial_features_behaviour_on_zero_degree(sparse_container):
    """Check that PolynomialFeatures raises error when degree=0 and include_bias=False,
    and output a single constant column when include_bias=True
    """
    # 创建一个形状为 (10, 2) 的全为 1 的数组 X
    X = np.ones((10, 2))
    
    # 创建 PolynomialFeatures 对象，指定 degree=0, include_bias=False
    poly = PolynomialFeatures(degree=0, include_bias=False)
    
    # 定义匹配的错误信息，检测是否抛出预期的 ValueError 异常
    err_msg = (
        "Setting degree to zero and include_bias to False would result in"
        " an empty output array."
    )
    
    # 使用 pytest 的 raises 方法检测是否抛出 ValueError 异常，并匹配错误信息
    with pytest.raises(ValueError, match=err_msg):
        poly.fit_transform(X)

    # 创建 PolynomialFeatures 对象，指定 degree=(0, 0), include_bias=False
    poly = PolynomialFeatures(degree=(0, 0), include_bias=False)
    
    # 定义匹配的错误信息，检测是否抛出预期的 ValueError 异常
    err_msg = (
        "Setting both min_degree and max_degree to zero and include_bias to"
        " False would result in an empty output array."
    )
    
    # 使用 pytest 的 raises 方法检测是否抛出 ValueError 异常，并匹配错误信息
    with pytest.raises(ValueError, match=err_msg):
        poly.fit_transform(X)

    # 循环检测不同的输入数组（包括稀疏矩阵），确保在 degree=0, include_bias=True 下输出符合预期
    for _X in [X, sparse_container(X)]:
        poly = PolynomialFeatures(degree=0, include_bias=True)
        output = poly.fit_transform(_X)
        # 如果输出是稀疏矩阵，则转换为稠密数组进行比较
        if sparse.issparse(output):
            output = output.toarray()
        # 使用 assert_array_equal 检查输出是否符合预期结果（全为 1）
        assert_array_equal(output, np.ones((X.shape[0], 1)))


def test_sizeof_LARGEST_INT_t():
    # 在 Windows 下，scikit-learn 通常使用 MSVC 编译，不支持 int128 算术运算
    # 参考：https://stackoverflow.com/a/6761962/163740
    if sys.platform == "win32" or (
        sys.maxsize <= 2**32 and sys.platform != "emscripten"
    ):
        expected_size = 8
    else:
        expected_size = 16

    # 断言 _get_sizeof_LARGEST_INT_t() 返回的结果是否符合预期大小
    assert _get_sizeof_LARGEST_INT_t() == expected_size


@pytest.mark.xfail(
    sys.platform == "win32",
    reason=(
        "On Windows, scikit-learn is typically compiled with MSVC that does not support"
        " int128 arithmetic (at the time of writing)"
    ),
    run=True,
)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_csr_polynomial_expansion_windows_fail(csr_container):
    # 确保在 Windows 下会发生整数溢出，同时保证输出是可以用 int64 索引的
    
    # 计算特征数，使用 np.iinfo(np.int64).max 的立方根加 3
    n_features = int(np.iinfo(np.int64).max ** (1 / 3) + 3)
    # 创建包含单一元素 1.0 的数据列表
    data = [1.0]
    # 创建包含单一元素 0 的行索引列表
    row = [0]
    # 创建列索引列表，最后一个元素为 n_features - 1
    col = [n_features - 1]

    # 第一阶索引
    expected_indices = [
        n_features - 1,
    ]
    # 第二阶索引
    expected_indices.append(
        int(n_features * (n_features + 1) // 2 + expected_indices[0])
    )
    # 第三阶索引
    expected_indices.append(
        int(n_features * (n_features + 1) * (n_features + 2) // 6 + expected_indices[1])
    )

    # 使用 data, row, col 创建 CSR 格式的稀疏矩阵 X
    X = csr_container((data, (row, col)))
    # 初始化多项式特征转换器，设定交互项为 False，包含偏差项为 False，多项式阶数为 3
    pf = PolynomialFeatures(interaction_only=False, include_bias=False, degree=3)
    
    # 根据系统最大整数大小进行条件判断
    if sys.maxsize <= 2**32:
        # 如果条件成立，设定错误消息正则表达式匹配规则
        msg = (
            r"The output that would result from the current configuration would"
            r" have \d*"
            r" features which is too large to be indexed"
        )
        # 使用 pytest 检查是否会抛出 ValueError，并匹配设定的错误消息
        with pytest.raises(ValueError, match=msg):
            pf.fit_transform(X)
    else:
        # 如果条件不成立，对 X 进行多项式特征转换
        X_trans = pf.fit_transform(X)
        # 验证转换后的稀疏矩阵 X_trans 中的指定索引位置的值近似为 1.0
        for idx in range(3):
            assert X_trans[0, expected_indices[idx]] == pytest.approx(1.0)
```