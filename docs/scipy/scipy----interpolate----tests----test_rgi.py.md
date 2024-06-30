# `D:\src\scipysrc\scipy\scipy\interpolate\tests\test_rgi.py`

```
# 导入 itertools 库，用于生成迭代器的函数
import itertools

# 导入 pytest 库，用于编写和运行测试用例
import pytest

# 导入 numpy 库，并使用 np 别名进行引用
import numpy as np

# 从 numpy.testing 模块中导入多个断言函数，用于测试 numpy 数组的相等性和近似性
from numpy.testing import (
    assert_allclose,     # 断言所有元素在误差范围内接近
    assert_equal,        # 断言两个数组或值相等
    assert_warns,        # 断言某个警告被触发
    assert_array_almost_equal,  # 断言两个数组在指定的小数位数上近似相等
    assert_array_equal   # 断言两个数组完全相等
)

# 从 pytest 模块中导入 raises 函数，并使用 assert_raises 别名进行引用
from pytest import raises as assert_raises

# 从 scipy.interpolate 模块中导入多个插值方法和类
from scipy.interpolate import (
    RegularGridInterpolator,  # 正则网格插值器
    interpn,                  # n维插值
    RectBivariateSpline,      # 二维矩形样条插值
    NearestNDInterpolator,    # 最近邻n维插值器
    LinearNDInterpolator      # 线性n维插值器
)

# 从 scipy.sparse._sputils 模块中导入 matrix 类
from scipy.sparse._sputils import matrix

# 从 scipy._lib._util 模块中导入 ComplexWarning 类
from scipy._lib._util import ComplexWarning

# 使用 pytest 的 parametrize 标记，为 RegularGridInterpolator 类中的所有插值方法创建参数化测试
parametrize_rgi_interp_methods = pytest.mark.parametrize(
    "method", RegularGridInterpolator._ALL_METHODS
)

# 定义 TestRegularGridInterpolator 类，用于测试 RegularGridInterpolator 类的功能
class TestRegularGridInterpolator:
    # 定义 _get_sample_4d 方法，返回一个四维网格的示例点和值
    def _get_sample_4d(self):
        # 创建一个每个维度包含三个点的四维网格
        points = [(0., .5, 1.)] * 4
        values = np.asarray([0., .5, 1.])
        values0 = values[:, np.newaxis, np.newaxis, np.newaxis]
        values1 = values[np.newaxis, :, np.newaxis, np.newaxis]
        values2 = values[np.newaxis, np.newaxis, :, np.newaxis]
        values3 = values[np.newaxis, np.newaxis, np.newaxis, :]
        values = (values0 + values1 * 10 + values2 * 100 + values3 * 1000)
        return points, values

    # 定义 _get_sample_4d_2 方法，返回另一个四维网格的示例点和值
    def _get_sample_4d_2(self):
        # 创建另一个每个维度包含两组不同点的四维网格
        points = [(0., .5, 1.)] * 2 + [(0., 5., 10.)] * 2
        values = np.asarray([0., .5, 1.])
        values0 = values[:, np.newaxis, np.newaxis, np.newaxis]
        values1 = values[np.newaxis, :, np.newaxis, np.newaxis]
        values2 = values[np.newaxis, np.newaxis, :, np.newaxis]
        values3 = values[np.newaxis, np.newaxis, np.newaxis, :]
        values = (values0 + values1 * 10 + values2 * 100 + values3 * 1000)
        return points, values

    # 定义 _get_sample_4d_3 方法，返回另一个四维网格的示例点和值
    def _get_sample_4d_3(self):
        # 创建另一个每个维度包含七个点的四维网格
        points = [(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)] * 4
        values = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        values0 = values[:, np.newaxis, np.newaxis, np.newaxis]
        values1 = values[np.newaxis, :, np.newaxis, np.newaxis]
        values2 = values[np.newaxis, np.newaxis, :, np.newaxis]
        values3 = values[np.newaxis, np.newaxis, np.newaxis, :]
        values = (values0 + values1 * 10 + values2 * 100 + values3 * 1000)
        return points, values

    # 定义 _get_sample_4d_4 方法，返回另一个四维网格的示例点和值
    def _get_sample_4d_4(self):
        # 创建另一个每个维度包含两个点的四维网格
        points = [(0.0, 1.0)] * 4
        values = np.asarray([0.0, 1.0])
        values0 = values[:, np.newaxis, np.newaxis, np.newaxis]
        values1 = values[np.newaxis, :, np.newaxis, np.newaxis]
        values2 = values[np.newaxis, np.newaxis, :, np.newaxis]
        values3 = values[np.newaxis, np.newaxis, np.newaxis, :]
        values = (values0 + values1 * 10 + values2 * 100 + values3 * 1000)
        return points, values

    # 使用 parametrize_rgi_interp_methods 标记的测试方法，测试 RegularGridInterpolator 的不同插值方法
    @parametrize_rgi_interp_methods
    # 定义一个测试方法，接受一个方法参数用于指定插值方法
    def test_list_input(self, method):
        # 从辅助方法中获取一个4维示例数据集的点和值
        points, values = self._get_sample_4d_3()

        # 创建一个示例数据集，转换为NumPy数组
        sample = np.asarray([[0.1, 0.1, 1., .9], [0.2, 0.1, .45, .8],
                             [0.5, 0.5, .5, .5]])

        # 使用指定的插值方法创建RegularGridInterpolator对象，输入数据转换为列表形式
        interp = RegularGridInterpolator(points,
                                         values.tolist(),
                                         method=method)
        # 对示例数据进行插值计算，转换为列表形式
        v1 = interp(sample.tolist())

        # 使用指定的插值方法创建RegularGridInterpolator对象，输入数据为原始NumPy数组形式
        interp = RegularGridInterpolator(points,
                                         values,
                                         method=method)
        # 对示例数据进行插值计算，保持数据形式为NumPy数组
        v2 = interp(sample)

        # 断言两种插值结果的近似性
        assert_allclose(v1, v2)

    @pytest.mark.parametrize('method', ['cubic', 'quintic', 'pchip'])
    # 定义一个参数化测试方法，参数为不同的插值方法
    def test_spline_dim_error(self, method):
        # 从辅助方法中获取一个4维示例数据集的点和值
        points, values = self._get_sample_4d_4()
        match = "points in dimension"

        # 检查在创建插值器时是否会引发错误
        with pytest.raises(ValueError, match=match):
            RegularGridInterpolator(points, values, method=method)

        # 创建RegularGridInterpolator对象，检查在提供不正确的方法时是否会引发错误
        interp = RegularGridInterpolator(points, values)
        sample = np.asarray([[0.1, 0.1, 1., .9], [0.2, 0.1, .45, .8],
                             [0.5, 0.5, .5, .5]])
        with pytest.raises(ValueError, match=match):
            interp(sample, method=method)

    @pytest.mark.parametrize(
        "points_values, sample",
        [
            (
                _get_sample_4d,
                np.asarray(
                    [[0.1, 0.1, 1.0, 0.9],
                     [0.2, 0.1, 0.45, 0.8],
                     [0.5, 0.5, 0.5, 0.5]]
                ),
            ),
            (_get_sample_4d_2, np.asarray([0.1, 0.1, 10.0, 9.0])),
        ],
    )
    # 定义一个参数化测试方法，测试线性和slinear插值方法的结果接近程度
    def test_linear_and_slinear_close(self, points_values, sample):
        # 从辅助方法中获取一个4维示例数据集的点和值
        points, values = points_values(self)

        # 使用线性插值方法创建RegularGridInterpolator对象
        interp = RegularGridInterpolator(points, values, method="linear")
        v1 = interp(sample)

        # 使用slinear插值方法创建RegularGridInterpolator对象
        interp = RegularGridInterpolator(points, values, method="slinear")
        v2 = interp(sample)

        # 断言线性和slinear插值方法的结果接近程度
        assert_allclose(v1, v2)

    # 定义一个测试方法，用于测试插值函数的导数计算功能
    def test_derivatives(self):
        # 从辅助方法中获取一个4维示例数据集的点和值
        points, values = self._get_sample_4d()

        # 创建RegularGridInterpolator对象，使用slinear插值方法
        interp = RegularGridInterpolator(points, values, method="slinear")

        # 断言在提供错误数量的导数时是否会引发错误
        with assert_raises(ValueError):
            interp(sample, nu=1)

        # 断言导数计算的结果与预期值的近似性，atol表示绝对误差的阈值
        assert_allclose(interp(sample, nu=(1, 0, 0, 0)),
                        [1, 1, 1], atol=1e-15)
        assert_allclose(interp(sample, nu=(0, 1, 0, 0)),
                        [10, 10, 10], atol=1e-15)

        # 断言线性函数的二阶导数为零
        assert_allclose(interp(sample, nu=(0, 1, 1, 0)),
                        [0, 0, 0], atol=2e-12)

    @parametrize_rgi_interp_methods
    # 测试复杂插值方法的函数，根据给定的方法选择是否跳过使用 pchip 方法
    def test_complex(self, method):
        if method == "pchip":
            # 如果选择了 pchip 方法，则跳过测试，因为对复杂数据没有意义
            pytest.skip("pchip does not make sense for complex data")
        # 获取四维样本数据的点和值
        points, values = self._get_sample_4d_3()
        # 将值部分转换为复数，实部不变，虚部乘以 -2
        values = values - 2j * values
        # 设置采样点
        sample = np.asarray([[0.1, 0.1, 1., .9], [0.2, 0.1, .45, .8],
                             [0.5, 0.5, .5, .5]])

        # 使用给定方法创建 RegularGridInterpolator 对象
        interp = RegularGridInterpolator(points, values, method=method)
        # 创建一个只包含实部的 RegularGridInterpolator 对象
        rinterp = RegularGridInterpolator(points, values.real, method=method)
        # 创建一个只包含虚部的 RegularGridInterpolator 对象
        iinterp = RegularGridInterpolator(points, values.imag, method=method)

        # 分别对三个插值器进行插值计算
        v1 = interp(sample)
        v2 = rinterp(sample) + 1j * iinterp(sample)
        # 检查两种方式得到的结果是否非常接近
        assert_allclose(v1, v2)

    # 测试 cubic 和 pchip 方法的差异
    def test_cubic_vs_pchip(self):
        # 创建一维坐标向量
        x, y = [1, 2, 3, 4], [1, 2, 3, 4]
        # 使用 meshgrid 函数生成网格
        xg, yg = np.meshgrid(x, y, indexing='ij')

        # 定义值的计算方式
        values = (lambda x, y: x ** 4 * y ** 4)(xg, yg)
        # 创建 cubic 方法的 RegularGridInterpolator 对象
        cubic = RegularGridInterpolator((x, y), values, method='cubic')
        # 创建 pchip 方法的 RegularGridInterpolator 对象
        pchip = RegularGridInterpolator((x, y), values, method='pchip')

        # 分别对两种插值方法进行计算
        vals_cubic = cubic([1.5, 2])
        vals_pchip = pchip([1.5, 2])
        # 检查两种方法得到的结果是否在一定误差范围内不相等
        assert not np.allclose(vals_cubic, vals_pchip, atol=1e-14, rtol=0)

    # 测试线性插值方法在一维输入下的计算
    def test_linear_xi1d(self):
        # 获取四维样本数据的点和值
        points, values = self._get_sample_4d_2()
        # 创建 RegularGridInterpolator 对象
        interp = RegularGridInterpolator(points, values)
        # 设置输入的采样点
        sample = np.asarray([0.1, 0.1, 10., 9.])
        # 预期得到的插值结果
        wanted = 1001.1
        # 检查插值计算结果是否接近预期
        assert_array_almost_equal(interp(sample), wanted)

    # 测试线性插值方法在三维输入下的计算
    def test_linear_xi3d(self):
        # 获取四维样本数据的点和值
        points, values = self._get_sample_4d()
        # 创建 RegularGridInterpolator 对象
        interp = RegularGridInterpolator(points, values)
        # 设置输入的采样点
        sample = np.asarray([[0.1, 0.1, 1., .9], [0.2, 0.1, .45, .8],
                             [0.5, 0.5, .5, .5]])
        # 预期得到的插值结果
        wanted = np.asarray([1001.1, 846.2, 555.5])
        # 检查插值计算结果是否接近预期
        assert_array_almost_equal(interp(sample), wanted)

    # 使用参数化测试，测试 nearest 方法的插值计算
    @pytest.mark.parametrize(
        "sample, wanted",
        [
            (np.asarray([0.1, 0.1, 0.9, 0.9]), 1100.0),
            (np.asarray([0.1, 0.1, 0.1, 0.1]), 0.0),
            (np.asarray([0.0, 0.0, 0.0, 0.0]), 0.0),
            (np.asarray([1.0, 1.0, 1.0, 1.0]), 1111.0),
            (np.asarray([0.1, 0.4, 0.6, 0.9]), 1055.0),
        ],
    )
    def test_nearest(self, sample, wanted):
        # 获取四维样本数据的点和值
        points, values = self._get_sample_4d()
        # 创建 nearest 方法的 RegularGridInterpolator 对象
        interp = RegularGridInterpolator(points, values, method="nearest")
        # 检查插值计算结果是否接近预期
        assert_array_almost_equal(interp(sample), wanted)

    # 测试线性插值方法在边缘点上的计算
    def test_linear_edges(self):
        # 获取四维样本数据的点和值
        points, values = self._get_sample_4d()
        # 创建 RegularGridInterpolator 对象
        interp = RegularGridInterpolator(points, values)
        # 设置输入的采样点
        sample = np.asarray([[0., 0., 0., 0.], [1., 1., 1., 1.]])
        # 预期得到的插值结果
        wanted = np.asarray([0., 1111.])
        # 检查插值计算结果是否接近预期
        assert_array_almost_equal(interp(sample), wanted)
    def test_valid_create(self):
        # 创建一个二维网格，每个维度有三个点
        points = [(0., .5, 1.), (0., 1., .5)]
        values = np.asarray([0., .5, 1.])
        values0 = values[:, np.newaxis]  # 将 values 转换为列向量
        values1 = values[np.newaxis, :]  # 将 values 转换为行向量
        values = (values0 + values1 * 10)  # 执行向量化运算
        assert_raises(ValueError, RegularGridInterpolator, points, values)  # 检查是否引发 ValueError 异常
        points = [((0., .5, 1.), ), (0., .5, 1.)]
        assert_raises(ValueError, RegularGridInterpolator, points, values)  # 检查是否引发 ValueError 异常
        points = [(0., .5, .75, 1.), (0., .5, 1.)]
        assert_raises(ValueError, RegularGridInterpolator, points, values)  # 检查是否引发 ValueError 异常
        points = [(0., .5, 1.), (0., .5, 1.), (0., .5, 1.)]
        assert_raises(ValueError, RegularGridInterpolator, points, values)  # 检查是否引发 ValueError 异常
        points = [(0., .5, 1.), (0., .5, 1.)]
        assert_raises(ValueError, RegularGridInterpolator, points, values,
                      method="undefmethod")  # 检查是否引发 ValueError 异常，并指定 method 参数为 "undefmethod"

    def test_valid_call(self):
        points, values = self._get_sample_4d()  # 获取 4 维样本点和值
        interp = RegularGridInterpolator(points, values)  # 创建 RegularGridInterpolator 实例
        sample = np.asarray([[0., 0., 0., 0.], [1., 1., 1., 1.]])  # 创建示例样本
        assert_raises(ValueError, interp, sample, "undefmethod")  # 检查插值调用是否引发 ValueError 异常，指定 method 参数为 "undefmethod"
        sample = np.asarray([[0., 0., 0.], [1., 1., 1.]])  # 创建示例样本
        assert_raises(ValueError, interp, sample)  # 检查插值调用是否引发 ValueError 异常
        sample = np.asarray([[0., 0., 0., 0.], [1., 1., 1., 1.1]])  # 创建示例样本
        assert_raises(ValueError, interp, sample)  # 检查插值调用是否引发 ValueError 异常

    def test_out_of_bounds_extrap(self):
        points, values = self._get_sample_4d()  # 获取 4 维样本点和值
        interp = RegularGridInterpolator(points, values, bounds_error=False,
                                         fill_value=None)  # 创建 RegularGridInterpolator 实例，关闭边界错误检查，填充值为 None
        sample = np.asarray([[-.1, -.1, -.1, -.1], [1.1, 1.1, 1.1, 1.1],
                             [21, 2.1, -1.1, -11], [2.1, 2.1, -1.1, -1.1]])  # 创建超出边界的示例样本
        wanted = np.asarray([0., 1111., 11., 11.])  # 期望的插值结果
        assert_array_almost_equal(interp(sample, method="nearest"), wanted)  # 检查最近邻插值方法的插值结果是否与期望一致
        wanted = np.asarray([-111.1, 1222.1, -11068., -1186.9])  # 期望的插值结果
        assert_array_almost_equal(interp(sample, method="linear"), wanted)  # 检查线性插值方法的插值结果是否与期望一致

    def test_out_of_bounds_extrap2(self):
        points, values = self._get_sample_4d_2()  # 获取第二组 4 维样本点和值
        interp = RegularGridInterpolator(points, values, bounds_error=False,
                                         fill_value=None)  # 创建 RegularGridInterpolator 实例，关闭边界错误检查，填充值为 None
        sample = np.asarray([[-.1, -.1, -.1, -.1], [1.1, 1.1, 1.1, 1.1],
                             [21, 2.1, -1.1, -11], [2.1, 2.1, -1.1, -1.1]])  # 创建超出边界的示例样本
        wanted = np.asarray([0., 11., 11., 11.])  # 期望的插值结果
        assert_array_almost_equal(interp(sample, method="nearest"), wanted)  # 检查最近邻插值方法的插值结果是否与期望一致
        wanted = np.asarray([-12.1, 133.1, -1069., -97.9])  # 期望的插值结果
        assert_array_almost_equal(interp(sample, method="linear"), wanted)  # 检查线性插值方法的插值结果是否与期望一致
    # 定义测试函数，用于测试 RegularGridInterpolator 的边界条件处理功能
    def test_out_of_bounds_fill(self):
        # 获取一个4维样本数据，包括点集和值集
        points, values = self._get_sample_4d()
        # 创建 RegularGridInterpolator 对象，设置边界错误处理为忽略并使用 NaN 填充
        interp = RegularGridInterpolator(points, values, bounds_error=False,
                                         fill_value=np.nan)
        # 定义一个超出边界的样本点集
        sample = np.asarray([[-.1, -.1, -.1, -.1], [1.1, 1.1, 1.1, 1.1],
                             [2.1, 2.1, -1.1, -1.1]])
        # 预期的结果是全为 NaN
        wanted = np.asarray([np.nan, np.nan, np.nan])
        # 断言使用最近邻插值方法时的结果近似等于预期结果
        assert_array_almost_equal(interp(sample, method="nearest"), wanted)
        # 断言使用线性插值方法时的结果近似等于预期结果
        assert_array_almost_equal(interp(sample, method="linear"), wanted)
        # 定义一个在边界内的样本点集
        sample = np.asarray([[0.1, 0.1, 1., .9], [0.2, 0.1, .45, .8],
                             [0.5, 0.5, .5, .5]])
        # 预期的结果是根据插值得出的数值
        wanted = np.asarray([1001.1, 846.2, 555.5])
        # 断言插值结果与预期结果近似相等
        assert_array_almost_equal(interp(sample), wanted)

    # 定义测试函数，比较 RegularGridInterpolator 和 NearestNDInterpolator 的最近邻插值效果
    def test_nearest_compare_qhull(self):
        # 获取一个4维样本数据，包括点集和值集
        points, values = self._get_sample_4d()
        # 创建 RegularGridInterpolator 对象，使用最近邻插值方法
        interp = RegularGridInterpolator(points, values, method="nearest")
        # 生成 points 的所有组合点集
        points_qhull = itertools.product(*points)
        points_qhull = [p for p in points_qhull]
        points_qhull = np.asarray(points_qhull)
        # 将 values 重塑为一维数组
        values_qhull = values.reshape(-1)
        # 创建 NearestNDInterpolator 对象，用于进行最近邻插值比较
        interp_qhull = NearestNDInterpolator(points_qhull, values_qhull)
        # 定义一个样本点集
        sample = np.asarray([[0.1, 0.1, 1., .9], [0.2, 0.1, .45, .8],
                             [0.5, 0.5, .5, .5]])
        # 断言 RegularGridInterpolator 和 NearestNDInterpolator 的插值结果近似相等
        assert_array_almost_equal(interp(sample), interp_qhull(sample))

    # 定义测试函数，比较 RegularGridInterpolator 和 LinearNDInterpolator 的线性插值效果
    def test_linear_compare_qhull(self):
        # 获取一个4维样本数据，包括点集和值集
        points, values = self._get_sample_4d()
        # 创建 RegularGridInterpolator 对象，使用线性插值方法
        interp = RegularGridInterpolator(points, values)
        # 生成 points 的所有组合点集
        points_qhull = itertools.product(*points)
        points_qhull = [p for p in points_qhull]
        points_qhull = np.asarray(points_qhull)
        # 将 values 重塑为一维数组
        values_qhull = values.reshape(-1)
        # 创建 LinearNDInterpolator 对象，用于进行线性插值比较
        interp_qhull = LinearNDInterpolator(points_qhull, values_qhull)
        # 定义一个样本点集
        sample = np.asarray([[0.1, 0.1, 1., .9], [0.2, 0.1, .45, .8],
                             [0.5, 0.5, .5, .5]])
        # 断言 RegularGridInterpolator 和 LinearNDInterpolator 的插值结果近似相等
        assert_array_almost_equal(interp(sample), interp_qhull(sample))

    # 定义参数化测试函数，测试 RegularGridInterpolator 处理不同方法的值类型
    @pytest.mark.parametrize("method", ["nearest", "linear"])
    def test_duck_typed_values(self, method):
        # 生成一个线性空间的 x 和 y
        x = np.linspace(0, 2, 5)
        y = np.linspace(0, 1, 7)
        # 创建一个自定义的 MyValue 对象
        values = MyValue((5, 7))
        # 创建 RegularGridInterpolator 对象，根据不同方法插值
        interp = RegularGridInterpolator((x, y), values, method=method)
        # 插值计算结果 v1
        v1 = interp([0.4, 0.7])
        # 创建 RegularGridInterpolator 对象，根据不同方法插值
        interp = RegularGridInterpolator((x, y), values._v, method=method)
        # 插值计算结果 v2
        v2 = interp([0.4, 0.7])
        # 断言两次插值结果近似相等
        assert_allclose(v1, v2)

    # 定义测试函数，测试 RegularGridInterpolator 处理无效填充值的情况
    def test_invalid_fill_value(self):
        # 设置随机种子
        np.random.seed(1234)
        # 生成一个线性空间的 x 和 y
        x = np.linspace(0, 2, 5)
        y = np.linspace(0, 1, 7)
        # 生成一个随机的5x7数组作为 values
        values = np.random.rand(5, 7)

        # 测试整数可以强制转换为浮点数填充值
        RegularGridInterpolator((x, y), values, fill_value=1)

        # 测试复数值不能作为填充值
        assert_raises(ValueError, RegularGridInterpolator,
                      (x, y), values, fill_value=1+2j)
    def test_fillvalue_type(self):
        # from #3703; test that interpolator object construction succeeds
        values = np.ones((10, 20, 30), dtype='>f4')
        points = [np.arange(n) for n in values.shape]
        # 创建 RegularGridInterpolator 对象，使用默认的填充值
        RegularGridInterpolator(points, values)
        # 创建 RegularGridInterpolator 对象，指定填充值为 0.0
        RegularGridInterpolator(points, values, fill_value=0.)

    def test_length_one_axis(self):
        # gh-5890, gh-9524 : length-1 axis is legal for method='linear'.
        # Along the axis it's linear interpolation; away from the length-1
        # axis, it's an extrapolation, so fill_value should be used.
        
        # 定义一个简单的函数 f(x, y)，返回 x + y
        def f(x, y):
            return x + y
        # 创建长度为 1 的数组 x，包含一个元素 1
        x = np.linspace(1, 1, 1)
        # 创建一个包含 10 个元素的数组 y，从 1 到 10
        y = np.linspace(1, 10, 10)
        # 用函数 f 在网格点上计算数据
        data = f(*np.meshgrid(x, y, indexing="ij", sparse=True))

        # 创建 RegularGridInterpolator 对象，使用线性插值方法
        interp = RegularGridInterpolator((x, y), data, method="linear",
                                         bounds_error=False, fill_value=101)

        # 检查在网格点上的插值结果
        assert_allclose(interp(np.array([[1, 1], [1, 5], [1, 10]])),
                        [2, 6, 11],
                        atol=1e-14)

        # 检查网格外插值是否为线性插值
        assert_allclose(interp(np.array([[1, 1.4], [1, 5.3], [1, 10]])),
                        [2.4, 6.3, 11],
                        atol=1e-14)

        # 检查使用 fill_value 进行外推插值
        assert_allclose(interp(np.array([1.1, 2.4])),
                        interp.fill_value,
                        atol=1e-14)

        # 检查外推插值：在 y 轴上线性，x 轴上常量
        interp.fill_value = None
        assert_allclose(interp([[1, 0.3], [1, 11.5]]),
                        [1.3, 12.5], atol=1e-15)

        assert_allclose(interp([[1.5, 0.3], [1.9, 11.5]]),
                        [1.3, 12.5], atol=1e-15)

        # 使用最近邻方法进行外推
        interp = RegularGridInterpolator((x, y), data, method="nearest",
                                         bounds_error=False, fill_value=None)
        assert_allclose(interp([[1.5, 1.8], [-4, 5.1]]),
                        [3, 6],
                        atol=1e-15)

    @pytest.mark.parametrize("fill_value", [None, np.nan, np.pi])
    @pytest.mark.parametrize("method", ['linear', 'nearest'])
    # 定义一个测试方法，用于测试 RegularGridInterpolator 在一个轴上的长度为1的情况
    def test_length_one_axis2(self, fill_value, method):
        # 设置选项字典，包括填充值、允许边界错误以及插值方法
        options = {"fill_value": fill_value, "bounds_error": False,
                   "method": method}

        # 生成一个长度为20的 x 值数组，范围从 0 到 2π
        x = np.linspace(0, 2*np.pi, 20)
        # 计算 sin(x) 的值，生成 z 数组
        z = np.sin(x)

        # 创建 RegularGridInterpolator 对象 fa，使用 x 数组作为单一轴
        fa = RegularGridInterpolator((x,), z[:], **options)
        # 创建 RegularGridInterpolator 对象 fb，使用 x 和 [0] 作为两个轴
        fb = RegularGridInterpolator((x, [0]), z[:, None], **options)

        # 生成一个从 -1 到 2π+1 的长度为100的 x1a 数组
        x1a = np.linspace(-1, 2*np.pi+1, 100)
        # 使用 fa 对象计算在 x1a 上的插值结果，生成 za 数组
        za = fa(x1a)

        # 对于给定的 y1b 值进行评估，fb 应该与 fa 表现一致
        y1b = np.zeros(100)
        zb = fb(np.vstack([x1a, y1b]).T)
        # 使用 assert_allclose 检查 zb 和 za 的近似程度
        assert_allclose(zb, za)

        # 对于不同的 y1b 值进行评估，fb 应返回填充值
        y1b = np.ones(100)
        zb = fb(np.vstack([x1a, y1b]).T)
        # 根据填充值是否为 None 使用 assert_allclose 检查 zb 和 za 的近似程度
        if fill_value is None:
            assert_allclose(zb, za)
        else:
            assert_allclose(zb, fill_value)

    @pytest.mark.parametrize("method", ['nearest', 'linear'])
    # 定义一个参数化测试方法，测试 RegularGridInterpolator 处理一维输入中的 NaN 值的情况
    def test_nan_x_1d(self, method):
        # 创建 RegularGridInterpolator 对象 f，使用一维输入和预设的参数
        f = RegularGridInterpolator(([1, 2, 3],), [10, 20, 30], fill_value=1,
                                    bounds_error=False, method=method)
        # 断言当输入包含 NaN 时，f 返回 NaN
        assert np.isnan(f([np.nan]))

        # 测试任意的 NaN 模式
        rng = np.random.default_rng(8143215468)
        x = rng.random(size=100) * 4
        i = rng.random(size=100) > 0.5
        x[i] = np.nan
        with np.errstate(invalid='ignore'):
            # 使用 f 对 x 进行插值计算，处理 NaN 值时可能会生成 numpy 警告
            # 这些警告应该传播到用户，我们在此简单地过滤掉它们
            res = f(x)

        # 使用 assert_equal 检查 res 中对应于 NaN 和非 NaN 值的一致性
        assert_equal(res[i], np.nan)
        assert_equal(res[~i], f(x[~i]))

        # 还要测试长度为1的轴上的情况 f(nan)
        x = [1, 2, 3]
        y = [1, ]
        data = np.ones((3, 1))
        f = RegularGridInterpolator((x, y), data, fill_value=1,
                                    bounds_error=False, method=method)
        # 断言当输入包含 NaN 时，f 返回 NaN
        assert np.isnan(f([np.nan, 1]))
        assert np.isnan(f([1, np.nan]))

    @pytest.mark.parametrize("method", ['nearest', 'linear'])
    # 参数化测试方法的标记
    # 定义测试函数，用于测试二维插值方法中处理 NaN 值的情况
    def test_nan_x_2d(self, method):
        # 创建输入数据 x 和 y，分别为 [0, 1, 2] 和 [1, 3, 7]
        x, y = np.array([0, 1, 2]), np.array([1, 3, 7])

        # 定义测试函数 f(x, y)，计算 x^2 + y^2
        def f(x, y):
            return x**2 + y**2

        # 使用 np.meshgrid 创建二维网格 xg 和 yg，通过 'ij' 索引方式，且稀疏表示
        xg, yg = np.meshgrid(x, y, indexing='ij', sparse=True)
        
        # 计算二维数据的函数值，即 f(xg, yg)
        data = f(xg, yg)
        
        # 使用 RegularGridInterpolator 创建插值器 interp，传入数据 (x, y) 和 data，
        # 方法为 method，并关闭边界错误的报错
        interp = RegularGridInterpolator((x, y), data,
                                         method=method, bounds_error=False)

        # 忽略 NaN 值产生的无效操作警告
        with np.errstate(invalid='ignore'):
            # 在插值器上进行插值操作，传入一个包含 NaN 的数组 [[1.5, np.nan], [1, 1]]
            res = interp([[1.5, np.nan], [1, 1]])
        
        # 使用 assert_allclose 断言，验证 res[1] 约等于 2，允许误差为 1e-14
        assert_allclose(res[1], 2, atol=1e-14)
        
        # 使用 assert 断言，验证 res[0] 是 NaN
        assert np.isnan(res[0])

        # 测试任意 NaN 模式
        # 使用随机数生成器 rng 创建随机数据
        rng = np.random.default_rng(8143215468)
        
        # 生成大小为 100 的随机数数组 x 和 y，分别在 [-1, 3) 和 [0, 8) 之间
        x = rng.random(size=100)*4-1
        y = rng.random(size=100)*8
        
        # 使用随机数生成器 rng 生成大小为 100 的随机布尔值数组 i1 和 i2
        i1 = rng.random(size=100) > 0.5
        i2 = rng.random(size=100) > 0.5
        
        # 创建布尔数组 i，其中 i[i1] 或 i[i2] 是 True
        i = i1 | i2
        
        # 将数组 x 中 i1 位置的元素设置为 NaN
        x[i1] = np.nan
        
        # 将数组 y 中 i2 位置的元素设置为 NaN
        y[i2] = np.nan
        
        # 创建二维数组 z，其转置为 [x, y]，用于插值
        z = np.array([x, y]).T
        
        # 忽略 NaN 值产生的无效操作警告
        with np.errstate(invalid='ignore'):
            # 在插值器 interp 上进行插值操作，传入数组 z
            res = interp(z)

        # 使用 assert_equal 断言，验证 res 中 i 对应的元素是 NaN
        assert_equal(res[i], np.nan)
        
        # 使用 assert_equal 断言，验证 res 中非 i 对应的元素与 interp(z[~i]) 相等
        assert_equal(res[~i], interp(z[~i]))

    # 使用 pytest.mark.fail_slow(10) 标记测试用例为慢速失败，失败率达到 10% 时触发
    @pytest.mark.fail_slow(10)
    
    # 使用 parametrize_rgi_interp_methods 装饰器，参数化测试插值方法
    @parametrize_rgi_interp_methods
    
    # 使用 pytest.mark.parametrize 参数化测试用例，设置不同维度 ndims 和相应的函数 func
    @pytest.mark.parametrize(("ndims", "func"), [
        (2, lambda x, y: 2 * x ** 3 + 3 * y ** 2),
        (3, lambda x, y, z: 2 * x ** 3 + 3 * y ** 2 - z),
        (4, lambda x, y, z, a: 2 * x ** 3 + 3 * y ** 2 - z + a),
        (5, lambda x, y, z, a, b: 2 * x ** 3 + 3 * y ** 2 - z + a * b),
    ])
    # 定义测试函数，用于测试 RegularGridInterpolator 的降序点插值是否与升序点插值结果一致
    def test_descending_points_nd(self, method, ndims, func):

        # 如果维度大于等于4且方法是 "cubic" 或 "quintic"，则跳过测试并输出相应信息
        if ndims >= 4 and method in {"cubic", "quintic"}:
            pytest.skip("too slow; OOM (quintic); or nearly so (cubic)")

        # 创建随机数生成器对象
        rng = np.random.default_rng(42)
        sample_low = 1
        sample_high = 5
        # 生成测试点集合
        test_points = rng.uniform(sample_low, sample_high, size=(2, ndims))

        # 生成升序点的列表
        ascending_points = [np.linspace(sample_low, sample_high, 12)
                            for _ in range(ndims)]

        # 根据函数计算升序点的值
        ascending_values = func(*np.meshgrid(*ascending_points,
                                             indexing="ij",
                                             sparse=True))

        # 创建 RegularGridInterpolator 对象，用于升序点的插值
        ascending_interp = RegularGridInterpolator(ascending_points,
                                                   ascending_values,
                                                   method=method)
        # 对测试点进行升序点插值
        ascending_result = ascending_interp(test_points)

        # 生成降序点的列表
        descending_points = [xi[::-1] for xi in ascending_points]
        # 根据函数计算降序点的值
        descending_values = func(*np.meshgrid(*descending_points,
                                              indexing="ij",
                                              sparse=True))
        # 创建 RegularGridInterpolator 对象，用于降序点的插值
        descending_interp = RegularGridInterpolator(descending_points,
                                                    descending_values,
                                                    method=method)
        # 对测试点进行降序点插值
        descending_result = descending_interp(test_points)

        # 断言升序点插值结果与降序点插值结果相等
        assert_array_equal(ascending_result, descending_result)

    # 测试无效点顺序的函数
    def test_invalid_points_order(self):
        # 定义一个二维值函数
        def val_func_2d(x, y):
            return 2 * x ** 3 + 3 * y ** 2

        # 设置 x 和 y 的值数组，其中 x 的值未严格升序或降序
        x = np.array([.5, 2., 0., 4., 5.5])  # not ascending or descending
        y = np.array([.5, 2., 3., 4., 5.5])
        points = (x, y)
        # 计算点集对应的值
        values = val_func_2d(*np.meshgrid(*points, indexing='ij',
                                          sparse=True))
        # 定义匹配错误信息的字符串
        match = "must be strictly ascending or descending"
        # 使用 pytest 检测是否会抛出 ValueError 异常，异常信息需匹配指定字符串
        with pytest.raises(ValueError, match=match):
            RegularGridInterpolator(points, values)

    # 测试 RegularGridInterpolator 对象的填充值
    @parametrize_rgi_interp_methods
    def test_fill_value(self, method):
        # 创建 RegularGridInterpolator 对象，值全为 1
        interp = RegularGridInterpolator([np.arange(6)], np.ones(6),
                                         method=method, bounds_error=False)
        # 断言对超出范围的点插值结果为 NaN
        assert np.isnan(interp([10]))

    # 标记测试为慢速失败的测试，并使用 parametrize_rgi_interp_methods 装饰器
    @pytest.mark.fail_slow(5)
    @parametrize_rgi_interp_methods
    def test_nonscalar_values(self, method):
        # 如果方法为 "quintic"，则跳过测试，因为速度太慢
        if method == "quintic":
            pytest.skip("Way too slow.")

        # 验证非标量值也能正常工作
        points = [(0.0, 0.5, 1.0, 1.5, 2.0, 2.5)] * 2 + [
            (0.0, 5.0, 10.0, 15.0, 20, 25.0)
        ] * 2

        # 使用默认种子创建随机数生成器
        rng = np.random.default_rng(1234)
        # 生成一个 6x6x6x6x8 的随机数数组
        values = rng.random((6, 6, 6, 6, 8))
        # 生成一个 7x3x4 的随机数数组
        sample = rng.random((7, 3, 4))

        # 创建 RegularGridInterpolator 对象进行插值
        interp = RegularGridInterpolator(points, values, method=method,
                                         bounds_error=False)
        # 对样本进行插值
        v = interp(sample)
        # 断言插值结果的形状为 (7, 3, 8)，并提供错误消息
        assert_equal(v.shape, (7, 3, 8), err_msg=method)

        # 初始化一个空列表 vs
        vs = []
        # 循环处理每个索引 j
        for j in range(8):
            # 创建 RegularGridInterpolator 对象，对单个轴进行插值
            interp = RegularGridInterpolator(points, values[..., j],
                                             method=method,
                                             bounds_error=False)
            # 将插值结果添加到 vs 列表中
            vs.append(interp(sample))
        # 将 vs 转置为 (1, 7, 3, 8) 的数组
        v2 = np.array(vs).transpose(1, 2, 0)

        # 使用绝对误差容忍度 1e-14 断言 v 和 v2 的所有元素近似相等，提供方法名称作为错误消息
        assert_allclose(v, v2, atol=1e-14, err_msg=method)

    @parametrize_rgi_interp_methods
    @pytest.mark.parametrize("flip_points", [False, True])
    def test_nonscalar_values_2(self, method, flip_points):
        # 如果方法为 "cubic" 或 "quintic"，则跳过测试，因为速度太慢
        if method in {"cubic", "quintic"}:
            pytest.skip("Way too slow.")

        # 验证非标量值也能正常工作：使用不同长度的轴简化内部跟踪
        points = [(0.0, 0.5, 1.0, 1.5, 2.0, 2.5),
                  (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
                  (0.0, 5.0, 10.0, 15.0, 20, 25.0, 35.0, 36.0),
                  (0.0, 5.0, 10.0, 15.0, 20, 25.0, 35.0, 36.0, 47)]

        # 如果 flip_points 为 True，则颠倒 points 列表中每个元组的顺序
        if flip_points:
            points = [tuple(reversed(p)) for p in points]

        # 使用默认种子创建随机数生成器
        rng = np.random.default_rng(1234)

        # trailing_points 为 (3, 2)
        trailing_points = (3, 2)
        # NB: values 具有 trailing dimension `num_trailing_dims`
        # 生成一个 (6, 7, 8, 9, 3, 2) 的随机数数组
        values = rng.random((6, 7, 8, 9, *trailing_points))
        # 生成一个长度为 4 的随机数数组作为样本点
        sample = rng.random(4)

        # 创建 RegularGridInterpolator 对象进行插值
        interp = RegularGridInterpolator(points, values, method=method,
                                         bounds_error=False)
        # 对样本进行插值
        v = interp(sample)

        # v 拥有每个 trailing dimension 条目的单一样本点
        assert v.shape == (1, *trailing_points)

        # 手动遍历 trailing dimensions 来检查值
        vs = np.empty(values.shape[-2:])
        for i in range(values.shape[-2]):
            for j in range(values.shape[-1]):
                # 创建 RegularGridInterpolator 对象，对单个轴进行插值
                interp = RegularGridInterpolator(points, values[..., i, j],
                                                 method=method,
                                                 bounds_error=False)
                # 获取插值结果并将其转换为标量值，然后放入 vs 数组中
                vs[i, j] = interp(sample).item()
        # 将 vs 扩展为形状为 (1, 3, 2) 的数组
        v2 = np.expand_dims(vs, axis=0)
        # 使用绝对误差容忍度 1e-14 断言 v 和 v2 的所有元素近似相等，提供方法名称作为错误消息
        assert_allclose(v, v2, atol=1e-14, err_msg=method)
    def test_nonscalar_values_linear_2D(self):
        # 验证在二维快速路径中非标量数值的工作情况
        method = 'linear'  # 使用线性插值方法
        points = [(0.0, 0.5, 1.0, 1.5, 2.0, 2.5),  # 定义数据点坐标
                  (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0), ]

        rng = np.random.default_rng(1234)  # 使用种子为1234的随机数生成器

        trailing_points = (3, 4)
        # NB: values has a `num_trailing_dims` trailing dimension
        # 定义具有尾部维度的数据，注意到这里的尾部维度数目
        values = rng.random((6, 7, *trailing_points))
        sample = rng.random(2)   # 单个样本点！

        interp = RegularGridInterpolator(points, values, method=method,
                                         bounds_error=False)
        v = interp(sample)  # 进行插值计算

        # v has a single sample point *per entry in the trailing dimensions*
        # v每个尾部维度条目都有一个单独的样本点
        assert v.shape == (1, *trailing_points)

        # check the values, too : manually loop over the trailing dimensions
        # 也检查数值：手动遍历尾部维度
        vs = np.empty(values.shape[-2:])
        for i in range(values.shape[-2]):
            for j in range(values.shape[-1]):
                interp = RegularGridInterpolator(points, values[..., i, j],
                                                 method=method,
                                                 bounds_error=False)
                vs[i, j] = interp(sample).item()
        v2 = np.expand_dims(vs, axis=0)
        assert_allclose(v, v2, atol=1e-14, err_msg=method)

    @pytest.mark.parametrize(
        "dtype",
        [np.float32, np.float64, np.complex64, np.complex128]
    )
    @pytest.mark.parametrize("xi_dtype", [np.float32, np.float64])
    def test_float32_values(self, dtype, xi_dtype):
        # regression test for gh-17718: values.dtype=float32 fails
        # 对于gh-17718的回归测试：values.dtype=float32 失败的问题
        def f(x, y):
            return 2 * x**3 + 3 * y**2

        x = np.linspace(1, 4, 11)
        y = np.linspace(4, 7, 22)

        xg, yg = np.meshgrid(x, y, indexing='ij', sparse=True)
        data = f(xg, yg)

        data = data.astype(dtype)  # 将数据转换为指定的数据类型

        interp = RegularGridInterpolator((x, y), data)

        pts = np.array([[2.1, 6.2],
                        [3.3, 5.2]], dtype=xi_dtype)

        # the values here are just what the call returns; the test checks that
        # that the call succeeds at all, instead of failing with cython not
        # having a float32 kernel
        # 这里的值是调用返回的值；测试检查调用是否成功，而不是由于cython没有float32内核而失败
        assert_allclose(interp(pts), [134.10469388, 153.40069388], atol=1e-7)
    # 定义测试方法 test_bad_solver，用于测试不良的解算器输入情况
    def test_bad_solver(self):
        # 创建一维数组 x，包含从 0 到 3 的 7 个均匀间隔的数值
        x = np.linspace(0, 3, 7)
        # 创建一维数组 y，包含从 0 到 3 的 7 个均匀间隔的数值
        y = np.linspace(0, 3, 7)
        # 使用 meshgrid 函数创建网格 xg 和 yg，采用 'ij' 索引顺序，sparse=True 表示创建稀疏矩阵
        xg, yg = np.meshgrid(x, y, indexing='ij', sparse=True)
        # 计算网格数据的和，存储在 data 中
        data = xg + yg

        # 使用 assert_raises 上下文管理器来验证 ValueError 是否被抛出
        with assert_raises(ValueError):
            # 创建 RegularGridInterpolator 对象，指定网格和数据，并传入一个仅返回其输入的无效解算器
            RegularGridInterpolator((x, y), data, solver=lambda x: x)

        # 使用 assert_raises 上下文管理器来验证 TypeError 是否被抛出
        with assert_raises(TypeError):
            # 创建 RegularGridInterpolator 对象，指定网格、数据、方法为 'slinear'，并传入一个无效解算器
            RegularGridInterpolator(
                (x, y), data, method='slinear', solver=lambda x: x
            )

        # 使用 assert_raises 上下文管理器来验证 TypeError 是否被抛出
        with assert_raises(TypeError):
            # 创建 RegularGridInterpolator 对象，指定网格、数据、方法为 'slinear'，传入一个无效解算器并带有额外的未知参数
            RegularGridInterpolator(
                (x, y), data, method='slinear', solver=lambda x: x, woof='woof'
            )

        # 使用 assert_raises 上下文管理器来验证 TypeError 是否被抛出
        with assert_raises(TypeError):
            # 创建 RegularGridInterpolator 对象，指定网格、数据、方法为 'slinear'，传入一个解算器参数字典包含未知的参数
            RegularGridInterpolator(
                (x, y), data, method='slinear', solver_args={'woof': 42}
            )
class MyValue:
    """
    Minimal indexable object
    """

    def __init__(self, shape):
        # 初始化对象，设置对象的维度和形状
        self.ndim = 2
        self.shape = shape
        # 创建一个包含指定形状的 NumPy 数组，并赋给对象的私有变量 _v
        self._v = np.arange(np.prod(shape)).reshape(shape)

    def __getitem__(self, idx):
        # 返回私有变量 _v 的指定索引的元素
        return self._v[idx]

    def __array_interface__(self):
        # 返回 None，表示对象没有 NumPy 数组接口
        return None

    def __array__(self, dtype=None, copy=None):
        # 抛出运行时异常，指示对象没有数组表示形式
        raise RuntimeError("No array representation")


class TestInterpN:
    def _sample_2d_data(self):
        # 返回用于测试的样本二维数据 x、y、z
        x = np.array([.5, 2., 3., 4., 5.5, 6.])
        y = np.array([.5, 2., 3., 4., 5.5, 6.])
        z = np.array(
            [
                [1, 2, 1, 2, 1, 1],
                [1, 2, 1, 2, 1, 1],
                [1, 2, 3, 2, 1, 1],
                [1, 2, 2, 2, 1, 1],
                [1, 2, 1, 2, 1, 1],
                [1, 2, 2, 2, 1, 1],
            ]
        )
        return x, y, z

    def test_spline_2d(self):
        # 测试二维样本数据的样条插值
        x, y, z = self._sample_2d_data()
        # 创建二维样条插值对象 lut
        lut = RectBivariateSpline(x, y, z)

        # 定义要插值的坐标点 xi
        xi = np.array([[1, 2.3, 5.3, 0.5, 3.3, 1.2, 3],
                       [1, 3.3, 1.2, 4.0, 5.0, 1.0, 3]]).T
        # 使用插值函数 interpn 对比插值结果与 lut 的评估值
        assert_array_almost_equal(interpn((x, y), z, xi, method="splinef2d"),
                                  lut.ev(xi[:, 0], xi[:, 1]))

    @parametrize_rgi_interp_methods
    def test_list_input(self, method):
        # 测试以列表形式输入的不同插值方法
        x, y, z = self._sample_2d_data()
        # 定义要插值的坐标点 xi
        xi = np.array([[1, 2.3, 5.3, 0.5, 3.3, 1.2, 3],
                       [1, 3.3, 1.2, 4.0, 5.0, 1.0, 3]]).T
        # 使用 interpn 进行插值并比较不同输入方式的结果 v1 和 v2
        v1 = interpn((x, y), z, xi, method=method)
        v2 = interpn(
            (x.tolist(), y.tolist()), z.tolist(), xi.tolist(), method=method
        )
        assert_allclose(v1, v2, err_msg=method)

    def test_spline_2d_outofbounds(self):
        # 测试二维样本数据的越界情况下的样条插值
        x = np.array([.5, 2., 3., 4., 5.5])
        y = np.array([.5, 2., 3., 4., 5.5])
        z = np.array([[1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 2, 3, 2, 1],
                      [1, 2, 2, 2, 1], [1, 2, 1, 2, 1]])
        # 创建二维样条插值对象 lut
        lut = RectBivariateSpline(x, y, z)

        # 定义要插值的坐标点 xi，包括越界的情况
        xi = np.array([[1, 2.3, 6.3, 0.5, 3.3, 1.2, 3],
                       [1, 3.3, 1.2, -4.0, 5.0, 1.0, 3]]).T
        # 使用 interpn 进行越界情况下的插值，设置边界错误为 False，填充值为 999.99
        actual = interpn((x, y), z, xi, method="splinef2d",
                         bounds_error=False, fill_value=999.99)
        # 获取 lut 对象对 xi 的评估值作为期望值
        expected = lut.ev(xi[:, 0], xi[:, 1])
        # 将期望值中第2到第3个元素设为 999.99，因为这些值超出了边界
        expected[2:4] = 999.99
        # 断言插值结果与期望值的接近程度
        assert_array_almost_equal(actual, expected)

        # 对于 splinef2d 方法，不允许外推，因此断言在边界错误为 False 且填充值为 None 时抛出 ValueError
        assert_raises(ValueError, interpn, (x, y), z, xi, method="splinef2d",
                      bounds_error=False, fill_value=None)
    # 定义一个方法来生成一个包含样本数据点和对应数值的4维数据集
    def _sample_4d_data(self):
        # 创建包含两个相同数据点的列表，每个数据点有三个坐标值
        points = [(0., .5, 1.)] * 2 + [(0., 5., 10.)] * 2
        # 创建包含三个数值的数组
        values = np.asarray([0., .5, 1.])
        # 扩展数组以匹配4维数据的形状
        values0 = values[:, np.newaxis, np.newaxis, np.newaxis]
        values1 = values[np.newaxis, :, np.newaxis, np.newaxis]
        values2 = values[np.newaxis, np.newaxis, :, np.newaxis]
        values3 = values[np.newaxis, np.newaxis, np.newaxis, :]
        # 创建包含组合数值的4维数组
        values = (values0 + values1 * 10 + values2 * 100 + values3 * 1000)
        return points, values

    # 测试线性插值在4维数据上的效果
    def test_linear_4d(self):
        # 获取样本数据点和对应数值
        points, values = self._sample_4d_data()
        # 创建 RegularGridInterpolator 对象，用于线性插值
        interp_rg = RegularGridInterpolator(points, values)
        # 创建一个待插值的样本点
        sample = np.asarray([[0.1, 0.1, 10., 9.]])
        # 使用 interpn 进行线性插值，得到期望结果
        wanted = interpn(points, values, sample, method="linear")
        # 断言插值结果与期望值接近
        assert_array_almost_equal(interp_rg(sample), wanted)

    # 测试带有超出边界情况的线性插值在4维数据上的效果
    def test_4d_linear_outofbounds(self):
        # 获取样本数据点和对应数值
        points, values = self._sample_4d_data()
        # 创建一个超出边界的样本点
        sample = np.asarray([[0.1, -0.1, 10.1, 9.]])
        # 设置期望值为一个指定的填充值
        wanted = 999.99
        # 使用 interpn 进行线性插值，并设置边界错误时的处理方式和填充值
        actual = interpn(points, values, sample, method="linear",
                         bounds_error=False, fill_value=999.99)
        # 断言插值结果与期望值接近
        assert_array_almost_equal(actual, wanted)

    # 测试最近邻插值在4维数据上的效果
    def test_nearest_4d(self):
        # 获取样本数据点和对应数值
        points, values = self._sample_4d_data()
        # 创建 RegularGridInterpolator 对象，用于最近邻插值
        interp_rg = RegularGridInterpolator(points, values, method="nearest")
        # 创建一个待插值的样本点
        sample = np.asarray([[0.1, 0.1, 10., 9.]])
        # 使用 interpn 进行最近邻插值，得到期望结果
        wanted = interpn(points, values, sample, method="nearest")
        # 断言插值结果与期望值接近
        assert_array_almost_equal(interp_rg(sample), wanted)

    # 测试带有超出边界情况的最近邻插值在4维数据上的效果
    def test_4d_nearest_outofbounds(self):
        # 获取样本数据点和对应数值
        points, values = self._sample_4d_data()
        # 创建一个超出边界的样本点
        sample = np.asarray([[0.1, -0.1, 10.1, 9.]])
        # 设置期望值为一个指定的填充值
        wanted = 999.99
        # 使用 interpn 进行最近邻插值，并设置边界错误时的处理方式和填充值
        actual = interpn(points, values, sample, method="nearest",
                         bounds_error=False, fill_value=999.99)
        # 断言插值结果与期望值接近
        assert_array_almost_equal(actual, wanted)

    # 测试在1维样本点情况下插值的效果
    def test_xi_1d(self):
        # 获取样本数据点和对应数值
        points, values = self._sample_4d_data()
        # 创建一个1维样本点
        sample = np.asarray([0.1, 0.1, 10., 9.])
        # 使用 interpn 进行插值，测试在不同维度下的一致性，无边界错误
        v1 = interpn(points, values, sample, bounds_error=False)
        v2 = interpn(points, values, sample[None,:], bounds_error=False)
        # 断言两种方式的插值结果应当接近
        assert_allclose(v1, v2)

    # 测试在高维样本点情况下插值的效果
    def test_xi_nd(self):
        # 获取样本数据点和对应数值
        points, values = self._sample_4d_data()

        # 使用随机数生成器设置一个高维样本点
        np.random.seed(1234)
        sample = np.random.rand(2, 3, 4)

        # 使用 interpn 进行最近邻插值，测试高维情况下的效果，无边界错误
        v1 = interpn(points, values, sample, method='nearest',
                     bounds_error=False)
        # 断言插值结果的形状应为 (2, 3)
        assert_equal(v1.shape, (2, 3))

        # 将样本点重塑为2维，并进行插值测试
        v2 = interpn(points, values, sample.reshape(-1, 4),
                     method='nearest', bounds_error=False)
        # 断言两种方式的插值结果应当接近
        assert_allclose(v1, v2.reshape(v1.shape))
    # 定义测试方法 `test_xi_broadcast`，用于验证插值器在广播 xi 时的行为
    def test_xi_broadcast(self, method):
        # 生成二维数据样本 x, y, values
        x, y, values = self._sample_2d_data()
        # 将 x 和 y 组合成 points 元组
        points = (x, y)

        # 创建 xi 和 yi，分别为 0 到 1 等间距的两个值，0 到 3 等间距的三个值
        xi = np.linspace(0, 1, 2)
        yi = np.linspace(0, 3, 3)

        # 构造采样点 sample
        sample = (xi[:, None], yi[None, :])

        # 进行插值计算，使用 interpn 函数，方法为 method，允许边界错误
        v1 = interpn(points, values, sample, method=method, bounds_error=False)

        # 断言 v1 的形状为 (2, 3)
        assert_equal(v1.shape, (2, 3))

        # 创建网格 xx 和 yy
        xx, yy = np.meshgrid(xi, yi)

        # 将网格点组合成采样点 sample
        sample = np.c_[xx.T.ravel(), yy.T.ravel()]

        # 进行插值计算，使用 interpn 函数，方法为 method，允许边界错误
        v2 = interpn(points, values, sample,
                     method=method, bounds_error=False)

        # 断言 v1 和重塑后的 v2 形状相等
        assert_allclose(v1, v2.reshape(v1.shape))

    # 为测试方法标记 'fail_slow'，指定失败次数为 5
    @pytest.mark.fail_slow(5)
    # 使用 parametrize_rgi_interp_methods 装饰器，动态参数化测试方法
    @parametrize_rgi_interp_methods
    def test_nonscalar_values(self, method):

        # 如果方法为 "quintic"，则跳过测试，因为速度太慢
        if method == "quintic":
            pytest.skip("Way too slow.")

        # 验证非标量值的插值也能正常工作
        points = [(0.0, 0.5, 1.0, 1.5, 2.0, 2.5)] * 2 + [
            (0.0, 5.0, 10.0, 15.0, 20, 25.0)
        ] * 2

        # 使用随机数生成器创建 values 和 sample
        rng = np.random.default_rng(1234)
        values = rng.random((6, 6, 6, 6, 8))
        sample = rng.random((7, 3, 4))

        # 进行插值计算，使用 interpn 函数，方法为 method，允许边界错误
        v = interpn(points, values, sample, method=method,
                    bounds_error=False)

        # 断言 v 的形状为 (7, 3, 8)，错误消息为 method
        assert_equal(v.shape, (7, 3, 8), err_msg=method)

        # 逐个维度插值，生成多个结果 vs
        vs = [interpn(points, values[..., j], sample, method=method,
                      bounds_error=False) for j in range(8)]

        # 将 vs 转置为 v2，并使用 allclose 断言它们在给定容差下相等
        v2 = np.array(vs).transpose(1, 2, 0)
        assert_allclose(v, v2, atol=1e-14, err_msg=method)

    # 使用 parametrize_rgi_interp_methods 装饰器，动态参数化测试方法
    @parametrize_rgi_interp_methods
    def test_nonscalar_values_2(self, method):

        # 如果方法为 "cubic" 或 "quintic"，则跳过测试，因为速度太慢
        if method in {"cubic", "quintic"}:
            pytest.skip("Way too slow.")

        # 验证非标量值的插值也能正常工作，简化轴的长度以追踪内部操作
        points = [(0.0, 0.5, 1.0, 1.5, 2.0, 2.5),
                  (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
                  (0.0, 5.0, 10.0, 15.0, 20, 25.0, 35.0, 36.0),
                  (0.0, 5.0, 10.0, 15.0, 20, 25.0, 35.0, 36.0, 47)]

        # 使用随机数生成器创建 values 和 sample
        rng = np.random.default_rng(1234)

        # trailing_points 为 (3, 2)
        trailing_points = (3, 2)

        # values 具有 num_trailing_dims 个尾随维度
        values = rng.random((6, 7, 8, 9, *trailing_points))

        # 创建单一样本点的 sample
        sample = rng.random(4)

        # 进行插值计算，使用 interpn 函数，方法为 method，允许边界错误
        v = interpn(points, values, sample, method=method, bounds_error=False)

        # v 每个尾随维度条目都有单一样本点
        assert v.shape == (1, *trailing_points)

        # 手动循环尾随维度，检查值 vs 是否与 v 相等，使用 allclose 断言
        vs = [[
                interpn(points, values[..., i, j], sample, method=method,
                        bounds_error=False) for i in range(values.shape[-2])
              ] for j in range(values.shape[-1])]

        assert_allclose(v, np.asarray(vs).T, atol=1e-14, err_msg=method)
    def test_non_scalar_values_splinef2d(self):
        # 测试对于向量值样本，使用 splinef2d 方法应该引发 ValueError 异常
        # 获取样本数据的点和值
        points, values = self._sample_4d_data()

        # 设置随机种子并生成一个随机的 3x3x3x3x6 的值数组
        np.random.seed(1234)
        values = np.random.rand(3, 3, 3, 3, 6)

        # 生成一个随机的 7x11x4 的样本数据数组，并验证是否引发 ValueError 异常
        sample = np.random.rand(7, 11, 4)
        assert_raises(ValueError, interpn, points, values, sample,
                      method='splinef2d')

    @parametrize_rgi_interp_methods
    def test_complex(self, method):
        # 对于复杂数据测试
        if method == "pchip":
            # 如果方法是 pchip，则跳过测试，因为 pchip 不支持复杂数据
            pytest.skip("pchip does not make sense for complex data")

        # 获取二维样本数据的 x、y 坐标和值
        x, y, values = self._sample_2d_data()
        points = (x, y)

        # 将值数组转换为复数形式（实部不变，虚部乘以 -2）
        values = values - 2j*values

        # 生成一个二维的样本数据数组
        sample = np.array([[1, 2.3, 5.3, 0.5, 3.3, 1.2, 3],
                           [1, 3.3, 1.2, 4.0, 5.0, 1.0, 3]]).T

        # 分别用指定方法计算插值结果 v1 和 v2，并验证它们的近似性
        v1 = interpn(points, values, sample, method=method)
        v2r = interpn(points, values.real, sample, method=method)
        v2i = interpn(points, values.imag, sample, method=method)
        v2 = v2r + 1j*v2i
        assert_allclose(v1, v2)

    def test_complex_pchip(self):
        # 测试复杂数据使用 pchip 方法是否引发 ValueError 异常
        x, y, values = self._sample_2d_data()
        points = (x, y)

        # 将值数组转换为复数形式（实部不变，虚部乘以 -2）
        values = values - 2j*values

        # 生成一个二维的样本数据数组，并验证是否引发 ValueError 异常
        sample = np.array([[1, 2.3, 5.3, 0.5, 3.3, 1.2, 3],
                           [1, 3.3, 1.2, 4.0, 5.0, 1.0, 3]]).T
        with pytest.raises(ValueError, match='real'):
            interpn(points, values, sample, method='pchip')

    def test_complex_spline2fd(self):
        # 测试复杂数据使用 spline2fd 方法是否发出复杂警告
        x, y, values = self._sample_2d_data()
        points = (x, y)

        # 将值数组转换为复数形式（实部不变，虚部乘以 -2）
        values = values - 2j*values

        # 生成一个二维的样本数据数组，并验证是否发出复杂警告
        sample = np.array([[1, 2.3, 5.3, 0.5, 3.3, 1.2, 3],
                           [1, 3.3, 1.2, 4.0, 5.0, 1.0, 3]]).T
        with assert_warns(ComplexWarning):
            interpn(points, values, sample, method='splinef2d')

    @pytest.mark.parametrize(
        "method",
        ["linear", "nearest"]
    )
    def test_duck_typed_values(self, method):
        # 测试鸭子类型的值输入
        x = np.linspace(0, 2, 5)
        y = np.linspace(0, 1, 7)

        # 创建一个自定义值对象 MyValue，并生成两个插值结果 v1 和 v2，验证它们的近似性
        values = MyValue((5, 7))
        v1 = interpn((x, y), values, [0.4, 0.7], method=method)
        v2 = interpn((x, y), values._v, [0.4, 0.7], method=method)
        assert_allclose(v1, v2)

    @parametrize_rgi_interp_methods
    def test_matrix_input(self, method):
        # 测试矩阵输入的情况
        x = np.linspace(0, 2, 6)
        y = np.linspace(0, 1, 7)

        # 创建一个矩阵对象，并生成两个插值结果 v1 和 v2，验证它们的近似性
        values = matrix(np.random.rand(6, 7))
        sample = np.random.rand(3, 7, 2)
        v1 = interpn((x, y), values, sample, method=method)

        # 对于 quintic 方法，验证 v1 和 v2 的近似性（允许较大的误差）
        v2 = interpn((x, y), np.asarray(values), sample, method=method)
        if method == "quintic":
            assert_allclose(v1, v2, atol=5e-5, rtol=2e-6)
        else:
            assert_allclose(v1, v2)
    def test_length_one_axis(self):
        # gh-5890, gh-9524 : length-1 axis is legal for method='linear'.
        # Along the axis it's linear interpolation; away from the length-1
        # axis, it's an extrapolation, so fill_value should be used.

        # 创建一个包含单个长度轴的测试用例，用于 method='linear' 的合法性测试
        values = np.array([[0.1, 1, 10]])
        xi = np.array([[1, 2.2], [1, 3.2], [1, 3.8]])

        # 在给定点 xi 上进行插值计算
        res = interpn(([1], [2, 3, 4]), values, xi)
        wanted = [0.9*0.2 + 0.1,   # 在 [2, 3) 区间上是 0.9*(x-2) + 0.1
                  9*0.2 + 1,       # 在 [3, 4] 区间上是 9*(x-3) + 1
                  9*0.8 + 1]

        # 检查计算结果与期望结果的接近程度
        assert_allclose(res, wanted, atol=1e-15)

        # 检查外推情况
        xi = np.array([[1.1, 2.2], [1.5, 3.2], [-2.3, 3.8]])
        res = interpn(([1], [2, 3, 4]), values, xi,
                      bounds_error=False, fill_value=None)

        # 检查计算结果与期望结果的接近程度
        assert_allclose(res, wanted, atol=1e-15)

    def test_descending_points(self):
        def value_func_4d(x, y, z, a):
            return 2 * x ** 3 + 3 * y ** 2 - z - a

        x1 = np.array([0, 1, 2, 3])
        x2 = np.array([0, 10, 20, 30])
        x3 = np.array([0, 10, 20, 30])
        x4 = np.array([0, .1, .2, .30])
        points = (x1, x2, x3, x4)

        # 生成四维函数值数组
        values = value_func_4d(
            *np.meshgrid(*points, indexing='ij', sparse=True))

        # 创建测试点
        pts = (0.1, 0.3, np.transpose(np.linspace(0, 30, 4)),
               np.linspace(0, 0.3, 4))

        # 使用正序点进行插值计算
        correct_result = interpn(points, values, pts)

        # 对 x1, x2, x3, x4 进行倒序处理
        x1_descend = x1[::-1]
        x2_descend = x2[::-1]
        x3_descend = x3[::-1]
        x4_descend = x4[::-1]
        points_shuffled = (x1_descend, x2_descend, x3_descend, x4_descend)

        # 使用倒序点进行插值计算
        values_shuffled = value_func_4d(
            *np.meshgrid(*points_shuffled, indexing='ij', sparse=True))
        test_result = interpn(points_shuffled, values_shuffled, pts)

        # 检查正序与倒序计算结果是否一致
        assert_array_equal(correct_result, test_result)

    def test_invalid_points_order(self):
        x = np.array([.5, 2., 0., 4., 5.5])  # 点未严格升序或降序排列
        y = np.array([.5, 2., 3., 4., 5.5])
        z = np.array([[1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 2, 3, 2, 1],
                      [1, 2, 2, 2, 1], [1, 2, 1, 2, 1]])
        xi = np.array([[1, 2.3, 6.3, 0.5, 3.3, 1.2, 3],
                       [1, 3.3, 1.2, -4.0, 5.0, 1.0, 3]]).T

        # 检查插值函数对于非严格升序或降序排列的输入点是否引发 ValueError
        match = "must be strictly ascending or descending"
        with pytest.raises(ValueError, match=match):
            interpn((x, y), z, xi)

    def test_invalid_xi_dimensions(self):
        # https://github.com/scipy/scipy/issues/16519
        points = [(0, 1)]
        values = [0, 1]
        xi = np.ones((1, 1, 3))

        # 检查插值函数对于维度不匹配的 xi 是否引发正确的 ValueError
        msg = ("The requested sample points xi have dimension 3, but this "
               "RegularGridInterpolator has dimension 1")
        with assert_raises(ValueError, match=msg):
            interpn(points, values, xi)
    def test_readonly_grid(self):
        # 测试只读网格的功能

        # 创建一维坐标点
        x = np.linspace(0, 4, 5)
        y = np.linspace(0, 5, 6)
        z = np.linspace(0, 6, 7)
        points = (x, y, z)

        # 创建数据值数组，并设为只读
        values = np.ones((5, 6, 7))
        values.flags.writeable = False

        # 创建一个测试点，并设为只读
        point = np.array([2.21, 3.12, 1.15])
        point.flags.writeable = False

        # 将每个坐标点数组设为只读
        for d in points:
            d.flags.writeable = False

        # 使用 interpn 进行插值计算
        interpn(points, values, point)

        # 使用 RegularGridInterpolator 进行插值计算
        RegularGridInterpolator(points, values)(point)

    def test_2d_readonly_grid(self):
        # 测试特殊的二维只读网格场景

        # 创建二维坐标点
        x = np.linspace(0, 4, 5)
        y = np.linspace(0, 5, 6)
        points = (x, y)

        # 创建数据值数组，并设为只读
        values = np.ones((5, 6))
        values.flags.writeable = False

        # 创建一个测试点，并设为只读
        point = np.array([2.21, 3.12])
        point.flags.writeable = False

        # 将每个坐标点数组设为只读
        for d in points:
            d.flags.writeable = False

        # 使用 interpn 进行插值计算
        interpn(points, values, point)

        # 使用 RegularGridInterpolator 进行插值计算
        RegularGridInterpolator(points, values)(point)

    def test_non_c_contiguous_grid(self):
        # 测试非 C 连续网格的情况

        # 创建一维坐标点，并确保其非 C 连续
        x = np.linspace(0, 4, 5)
        x = np.vstack((x, np.empty_like(x))).T.copy()[:, 0]
        assert not x.flags.c_contiguous

        # 创建二维坐标点
        y = np.linspace(0, 5, 6)
        z = np.linspace(0, 6, 7)
        points = (x, y, z)

        # 创建数据值数组
        values = np.ones((5, 6, 7))

        # 创建一个测试点
        point = np.array([2.21, 3.12, 1.15])

        # 使用 interpn 进行插值计算
        interpn(points, values, point)

        # 使用 RegularGridInterpolator 进行插值计算
        RegularGridInterpolator(points, values)(point)

    @pytest.mark.parametrize("dtype", ['>f8', '<f8'])
    def test_endianness(self, dtype):
        # 测试数据的字节序

        # 创建一维坐标点，并指定数据类型及字节序
        x = np.linspace(0, 4, 5, dtype=dtype)
        y = np.linspace(0, 5, 6, dtype=dtype)
        points = (x, y)

        # 创建数据值数组，并指定数据类型及字节序
        values = np.ones((5, 6), dtype=dtype)

        # 创建一个测试点，并指定数据类型及字节序
        point = np.array([2.21, 3.12], dtype=dtype)

        # 使用 interpn 进行插值计算
        interpn(points, values, point)

        # 使用 RegularGridInterpolator 进行插值计算
        RegularGridInterpolator(points, values)(point)
```