# `D:\src\scipysrc\scipy\scipy\spatial\tests\test_slerp.py`

```
import numpy as np  # 导入NumPy库，用于科学计算
from numpy.testing import assert_allclose  # 导入NumPy的测试模块，用于断言测试结果

import pytest  # 导入pytest库，用于编写和运行测试用例
from scipy.spatial import geometric_slerp  # 导入scipy库的geometric_slerp模块，用于几何插值


def _generate_spherical_points(ndim=3, n_pts=2):
    # 生成在单位球面上均匀分布的点
    # 参考：https://stackoverflow.com/a/23785326
    # 扩展到任意维度的球面
    # 对于0维球面，始终产生对踵点
    np.random.seed(123)
    points = np.random.normal(size=(n_pts, ndim))
    points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
    return points[0], points[1]


class TestGeometricSlerp:
    # 测试几何slerp代码的各种属性

    @pytest.mark.parametrize("n_dims", [2, 3, 5, 7, 9])
    @pytest.mark.parametrize("n_pts", [0, 3, 17])
    def test_shape_property(self, n_dims, n_pts):
        # geometric_slerp的输出形状应与
        # 输入的维度和请求的插值点数相匹配
        start, end = _generate_spherical_points(n_dims, 2)

        actual = geometric_slerp(start=start,
                                 end=end,
                                 t=np.linspace(0, 1, n_pts))

        assert actual.shape == (n_pts, n_dims)

    @pytest.mark.parametrize("n_dims", [2, 3, 5, 7, 9])
    @pytest.mark.parametrize("n_pts", [3, 17])
    def test_include_ends(self, n_dims, n_pts):
        # geometric_slerp应返回包含起点和终点坐标的数据结构
        # 当t包含0和1时
        # 这在例如表示插值的曲面绘制时非常方便

        # 生成器对单位球面的表现不佳（始终产生对踵点），因此在这里使用自定义值
        start, end = _generate_spherical_points(n_dims, 2)

        actual = geometric_slerp(start=start,
                                 end=end,
                                 t=np.linspace(0, 1, n_pts))

        assert_allclose(actual[0], start)
        assert_allclose(actual[-1], end)

    @pytest.mark.parametrize("start, end", [
        # 两个数组都不是平坦的
        (np.zeros((1, 3)), np.ones((1, 3))),
        # 只有起始数组不是平坦的
        (np.zeros((1, 3)), np.ones(3)),
        # 只有终点数组不是平坦的
        (np.zeros(1), np.ones((3, 1))),
        ])
    def test_input_shape_flat(self, start, end):
        # geometric_slerp应适当处理不是平坦的输入数组
        with pytest.raises(ValueError, match='one-dimensional'):
            geometric_slerp(start=start,
                            end=end,
                            t=np.linspace(0, 1, 10))

    @pytest.mark.parametrize("start, end", [
        # 7维和3维终点
        (np.zeros(7), np.ones(3)),
        # 2维和1维终点
        (np.zeros(2), np.ones(1)),
        # 空数组，"3D"也会以这种方式被捕获
        (np.array([]), np.ones(3)),
        ])
    def test_input_dim_mismatch(self, start, end):
        # geometric_slerp函数必须能够处理维度不匹配的情况，
        # 即在尝试在两个不同维度之间进行插值时
        with pytest.raises(ValueError, match='dimensions'):
            # 断言调用geometric_slerp时会引发ValueError，并匹配'dimensions'字符串
            geometric_slerp(start=start,
                            end=end,
                            t=np.linspace(0, 1, 10))

    @pytest.mark.parametrize("start, end", [
        # both empty
        (np.array([]), np.array([])),
        ])
    def test_input_at_least1d(self, start, end):
        # geometric_slerp函数对于空输入必须能够适当处理，
        # 即在未被维度不匹配检测到时
        with pytest.raises(ValueError, match='at least two-dim'):
            # 断言调用geometric_slerp时会引发ValueError，并匹配'at least two-dim'字符串
            geometric_slerp(start=start,
                            end=end,
                            t=np.linspace(0, 1, 10))

    @pytest.mark.parametrize("start, end, expected", [
        # North and South Poles are definitely antipodes
        # but should be handled gracefully now
        (np.array([0, 0, 1.0]), np.array([0, 0, -1.0]), "warning"),
        # this case will issue a warning & be handled
        # gracefully as well;
        # North Pole was rotated very slightly
        # using r = R.from_euler('x', 0.035, degrees=True)
        # to achieve Euclidean distance offset from diameter by
        # 9.328908379124812e-08, within the default tol
        (np.array([0.00000000e+00,
                  -6.10865200e-04,
                  9.99999813e-01]), np.array([0, 0, -1.0]), "warning"),
        # this case should succeed without warning because a
        # sufficiently large
        # rotation was applied to North Pole point to shift it
        # to a Euclidean distance of 2.3036691931821451e-07
        # from South Pole, which is larger than tol
        (np.array([0.00000000e+00,
                  -9.59930941e-04,
                  9.99999539e-01]), np.array([0, 0, -1.0]), "success"),
        ])
    def test_handle_antipodes(self, start, end, expected):
        # 处理对极点必须适当；
        # 在高维空间中存在无数可能的测地插值路径
        if expected == "warning":
            with pytest.warns(UserWarning, match='antipodes'):
                # 断言调用geometric_slerp时会引发UserWarning，并匹配'antipodes'字符串
                res = geometric_slerp(start=start,
                                      end=end,
                                      t=np.linspace(0, 1, 10))
        else:
            # 此处预期结果为'success'，不会引发警告
            res = geometric_slerp(start=start,
                                  end=end,
                                  t=np.linspace(0, 1, 10))

        # 对极点或接近对极点应该仍然在球面上产生slerp路径
        assert_allclose(np.linalg.norm(res, axis=1), 1.0)
    @pytest.mark.parametrize("start, end, expected", [
        # 参数化测试用例：二维空间，n_pts=4（两个新的插值点）
        # 这是一个实际的圆
        (np.array([1, 0]),  # 起始点
         np.array([0, 1]),  # 结束点
         np.array([[1, 0],  # 预期的插值点
                   [np.sqrt(3) / 2, 0.5],  # 单位圆上的30度角
                   [0.5, np.sqrt(3) / 2],  # 单位圆上的60度角
                   [0, 1]])),  # 结束点
        # 同样适用于三维空间（添加z = 0平面）
        # 这是一个普通的球体
        (np.array([1, 0, 0]),  # 起始点
         np.array([0, 1, 0]),  # 结束点
         np.array([[1, 0, 0],  # 预期的插值点
                   [np.sqrt(3) / 2, 0.5, 0],
                   [0.5, np.sqrt(3) / 2, 0],
                   [0, 1, 0]])),  # 结束点
        # 对于5维空间，用常量填充更多列
        # 使用零最简单--在单位圆上的非零值更难推理
        # 在更高维度上
        (np.array([1, 0, 0, 0, 0]),  # 起始点
         np.array([0, 1, 0, 0, 0]),  # 结束点
         np.array([[1, 0, 0, 0, 0],  # 预期的插值点
                   [np.sqrt(3) / 2, 0.5, 0, 0, 0],
                   [0.5, np.sqrt(3) / 2, 0, 0, 0],
                   [0, 1, 0, 0, 0]])),  # 结束点
    ])
    # 测试直接的例子
    def test_straightforward_examples(self, start, end, expected):
        # 一些直接的插值测试，足够简单，使用单位圆来推断预期值；
        # 对于更大的维度，用常量填充，使数据是N维但更容易推理
        actual = geometric_slerp(start=start,
                                 end=end,
                                 t=np.linspace(0, 1, 4))
        # 断言实际值与预期值的接近度，允许误差为1e-16
        assert_allclose(actual, expected, atol=1e-16)

    @pytest.mark.parametrize("t", [
        # 区间端点明显违反限制
        np.linspace(-20, 20, 300),
        # 只有一个区间端点略微违反限制
        np.linspace(-0.0001, 0.0001, 17),
        ])
    # 测试t值的边界情况
    def test_t_values_limits(self, t):
        # geometric_slerp() 应适当处理插值参数 < 0 和 > 1
        with pytest.raises(ValueError, match='interpolation parameter'):
            _ = geometric_slerp(start=np.array([1, 0]),
                                end=np.array([0, 1]),
                                t=t)

    @pytest.mark.parametrize("start, end", [
        (np.array([1]),  # 起始点
         np.array([0])),  # 结束点
        (np.array([0]),  # 起始点
         np.array([1])),  # 结束点
        (np.array([-17.7]),  # 起始点
         np.array([165.9])),  # 结束点
     ])
    # 测试0-球的处理
    def test_0_sphere_handling(self, start, end):
        # 没有意义插值0-球的两个点集
        with pytest.raises(ValueError, match='at least two-dim'):
            _ = geometric_slerp(start=start,
                                end=end,
                                t=np.linspace(0, 1, 4))
    @pytest.mark.parametrize("tol", [
        # 整数会引发异常
        5,
        # 字符串会引发异常
        "7",
        # 列表和数组也会引发异常
        [5, 6, 7], np.array(9.0),
        ])
    def test_tol_type(self, tol):
        # 当 tol 不是合适的浮点类型时，geometric_slerp() 应该引发异常
        with pytest.raises(ValueError, match='must be a float'):
            _ = geometric_slerp(start=np.array([1, 0]),
                                end=np.array([0, 1]),
                                t=np.linspace(0, 1, 5),
                                tol=tol)

    @pytest.mark.parametrize("tol", [
        -5e-6,
        -7e-10,
        ])
    def test_tol_sign(self, tol):
        # geometric_slerp() 目前可以处理负的 tol 值，只要它们是浮点数
        _ = geometric_slerp(start=np.array([1, 0]),
                            end=np.array([0, 1]),
                            t=np.linspace(0, 1, 5),
                            tol=tol)

    @pytest.mark.parametrize("start, end", [
        # 1-球（圆）上有一个点在原点，另一个点在圆上
        (np.array([1, 0]), np.array([0, 0])),
        # 2-球（普通球体）上两个点略微偏离球面，但偏离的方向不同
        (np.array([1 + 1e-6, 0, 0]),
         np.array([0, 1 - 1e-6, 0])),
        # 4-球（四维球体）上同样的情况
        (np.array([1 + 1e-6, 0, 0, 0]),
         np.array([0, 1 - 1e-6, 0, 0])),
        ])
    def test_unit_sphere_enforcement(self, start, end):
        # geometric_slerp() 应该在输入明显不可能在半径为1的n-球面上的情况下引发异常
        with pytest.raises(ValueError, match='unit n-sphere'):
            geometric_slerp(start=start,
                            end=end,
                            t=np.linspace(0, 1, 5))

    @pytest.mark.parametrize("start, end", [
        # 1-球 45度角情况
        (np.array([1, 0]),
         np.array([np.sqrt(2) / 2.,
                   np.sqrt(2) / 2.])),
        # 2-球 135度角情况
        (np.array([1, 0]),
         np.array([-np.sqrt(2) / 2.,
                   np.sqrt(2) / 2.])),
        ])
    @pytest.mark.parametrize("t_func", [
        np.linspace, np.logspace])
    def test_order_handling(self, start, end, t_func):
        # geometric_slerp() should handle scenarios with
        # ascending and descending t value arrays gracefully;
        # results should simply be reversed

        # for scrambled / unsorted parameters, the same values
        # should be returned, just in scrambled order

        num_t_vals = 20
        np.random.seed(789)
        # Generate an array of t values using the provided t_func
        forward_t_vals = t_func(0, 10, num_t_vals)
        # Normalize t values to range [0, 1]
        forward_t_vals /= forward_t_vals.max()
        # Reverse the array of t values
        reverse_t_vals = np.flipud(forward_t_vals)
        # Create an array of indices and shuffle them
        shuffled_indices = np.arange(num_t_vals)
        np.random.shuffle(shuffled_indices)
        # Create scrambled t values using the shuffled indices
        scramble_t_vals = forward_t_vals.copy()[shuffled_indices]

        # Calculate results using geometric_slerp function for different t arrays
        forward_results = geometric_slerp(start=start,
                                          end=end,
                                          t=forward_t_vals)
        reverse_results = geometric_slerp(start=start,
                                          end=end,
                                          t=reverse_t_vals)
        scrambled_results = geometric_slerp(start=start,
                                            end=end,
                                            t=scramble_t_vals)

        # Check if results for reverse_t_vals are reversed correctly compared to forward_results
        assert_allclose(forward_results, np.flipud(reverse_results))
        # Check if scrambled_results match forward_results with shuffled indices
        assert_allclose(forward_results[shuffled_indices],
                        scrambled_results)

    @pytest.mark.parametrize("t", [
        # string:
        "15, 5, 7",
        # complex numbers currently produce a warning
        # but not sure we need to worry about it too much:
        # [3 + 1j, 5 + 2j],
        ])
    def test_t_values_conversion(self, t):
        # Test that geometric_slerp raises a ValueError when given invalid t values
        with pytest.raises(ValueError):
            _ = geometric_slerp(start=np.array([1]),
                                end=np.array([0]),
                                t=t)

    def test_accept_arraylike(self):
        # array-like support requested by reviewer
        # in gh-10380
        # Calculate actual results using geometric_slerp for specified start, end, and t values
        actual = geometric_slerp([1, 0], [0, 1], [0, 1/3, 0.5, 2/3, 1])

        # expected values are based on visual inspection
        # of the unit circle for the progressions along
        # the circumference provided in t
        expected = np.array([[1, 0],
                             [np.sqrt(3) / 2, 0.5],
                             [np.sqrt(2) / 2,
                              np.sqrt(2) / 2],
                             [0.5, np.sqrt(3) / 2],
                             [0, 1]], dtype=np.float64)
        # Tyler's original Cython implementation of geometric_slerp
        # can pass at atol=0 here, but on balance we will accept
        # 1e-16 for an implementation that avoids Cython and
        # makes up accuracy ground elsewhere
        # Assert that actual results are close to expected results with a tolerance of 1e-16
        assert_allclose(actual, expected, atol=1e-16)
    def test_scalar_t(self):
        # 当 t 是一个标量时，返回的值是请求的适当维度的单个插值点
        # 由审阅者在 gh-10380 中要求
        actual = geometric_slerp([1, 0], [0, 1], 0.5)
        expected = np.array([np.sqrt(2) / 2,
                             np.sqrt(2) / 2], dtype=np.float64)
        assert actual.shape == (2,)
        assert_allclose(actual, expected)

    @pytest.mark.parametrize('start', [
        np.array([1, 0, 0]),
        np.array([0, 1]),
    ])
    @pytest.mark.parametrize('t', [
        np.array(1),
        np.array([1]),
        np.array([[1]]),
        np.array([[[1]]]),
        np.array([]),
        np.linspace(0, 1, 5),
    ])
    def test_degenerate_input(self, start, t):
        if np.asarray(t).ndim > 1:
            with pytest.raises(ValueError):
                geometric_slerp(start=start, end=start, t=t)
        else:
            # 计算期望的形状
            shape = (t.size,) + start.shape
            expected = np.full(shape, start)

            # 执行插值计算
            actual = geometric_slerp(start=start, end=start, t=t)
            assert_allclose(actual, expected)

            # 检查退化和非退化输入是否产生相同的大小
            non_degenerate = geometric_slerp(start=start, end=start[::-1], t=t)
            assert actual.size == non_degenerate.size

    @pytest.mark.parametrize('k', np.logspace(-10, -1, 10))
    def test_numerical_stability_pi(self, k):
        # geometric_slerp 在接近 pi 角度的情况下应具有优秀的数值稳定性
        # 在起始点和结束点之间
        angle = np.pi - k
        ts = np.linspace(0, 1, 100)
        P = np.array([1, 0, 0, 0])
        Q = np.array([np.cos(angle), np.sin(angle), 0, 0])
        # 仅对 geometric_slerp 确定输入实际在单位球面上的情况进行测试
        with np.testing.suppress_warnings() as sup:
            sup.filter(UserWarning)
            result = geometric_slerp(P, Q, ts, 1e-18)
            norms = np.linalg.norm(result, axis=1)
            error = np.max(np.abs(norms - 1))
            assert error < 4e-15

    @pytest.mark.parametrize('t', [
     [[0, 0.5]],
     [[[[[[[[[0, 0.5]]]]]]]]],
    ])
    def test_interpolation_param_ndim(self, t):
        # gh-14465 的回归测试
        arr1 = np.array([0, 1])
        arr2 = np.array([1, 0])

        with pytest.raises(ValueError):
            geometric_slerp(start=arr1,
                            end=arr2,
                            t=t)

        with pytest.raises(ValueError):
            geometric_slerp(start=arr1,
                            end=arr1,
                            t=t)
```