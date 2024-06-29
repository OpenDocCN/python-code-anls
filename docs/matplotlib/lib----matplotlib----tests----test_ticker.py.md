# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_ticker.py`

```py
# 导入必要的库：nullcontext 用于创建空的上下文管理器，itertools 提供迭代工具，locale 处理地区设置，logging 提供日志功能，re 提供正则表达式操作，parse_version 用于解析版本号。
from contextlib import nullcontext
import itertools
import locale
import logging
import re
from packaging.version import parse as parse_version

# 导入 NumPy 库及其测试模块，导入 pytest 用于单元测试。
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
import pytest

# 导入 Matplotlib 库及其相关模块，用于图形绘制和轴标尺设置。
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# 定义一个测试类 TestMaxNLocator，用于测试 MaxNLocator 类的功能。
class TestMaxNLocator:
    # 定义基本数据列表，每个元素包含 (vmin, vmax, 期望的 tick 值数组)
    basic_data = [
        (20, 100, np.array([20., 40., 60., 80., 100.])),
        (0.001, 0.0001, np.array([0., 0.0002, 0.0004, 0.0006, 0.0008, 0.001])),
        (-1e15, 1e15, np.array([-1.0e+15, -5.0e+14, 0e+00, 5e+14, 1.0e+15])),
        (0, 0.85e-50, np.arange(6) * 2e-51),
        (-0.85e-50, 0, np.arange(-5, 1) * 2e-51),
    ]

    # 定义整数数据列表，每个元素包含 (vmin, vmax, steps, 期望的 tick 值数组)
    integer_data = [
        (-0.1, 1.1, None, np.array([-1, 0, 1, 2])),
        (-0.1, 0.95, None, np.array([-0.25, 0, 0.25, 0.5, 0.75, 1.0])),
        (1, 55, [1, 1.5, 5, 6, 10], np.array([0, 15, 30, 45, 60])),
    ]

    # 使用 pytest 的 parametrize 装饰器，对 test_basic 方法进行参数化测试
    @pytest.mark.parametrize('vmin, vmax, expected', basic_data)
    def test_basic(self, vmin, vmax, expected):
        # 创建 MaxNLocator 实例 loc，指定 nbins 参数为 5
        loc = mticker.MaxNLocator(nbins=5)
        # 断言调用 loc 的 tick_values 方法返回的 tick 值近似等于 expected
        assert_almost_equal(loc.tick_values(vmin, vmax), expected)

    # 使用 pytest 的 parametrize 装饰器，对 test_integer 方法进行参数化测试
    @pytest.mark.parametrize('vmin, vmax, steps, expected', integer_data)
    def test_integer(self, vmin, vmax, steps, expected):
        # 创建 MaxNLocator 实例 loc，指定 nbins 参数为 5，integer 参数为 True，steps 参数为给定的 steps
        loc = mticker.MaxNLocator(nbins=5, integer=True, steps=steps)
        # 断言调用 loc 的 tick_values 方法返回的 tick 值近似等于 expected
        assert_almost_equal(loc.tick_values(vmin, vmax), expected)

    # 使用 pytest 的 parametrize 装饰器，对 test_errors 方法进行参数化测试
    @pytest.mark.parametrize('kwargs, errortype, match', [
        ({'foo': 0}, TypeError,
         re.escape("set_params() got an unexpected keyword argument 'foo'")),
        ({'steps': [2, 1]}, ValueError, "steps argument must be an increasing"),
        ({'steps': 2}, ValueError, "steps argument must be an increasing"),
        ({'steps': [2, 11]}, ValueError, "steps argument must be an increasing"),
    ])
    def test_errors(self, kwargs, errortype, match):
        # 使用 pytest 的 raises 方法检查是否抛出指定类型的异常，并匹配异常信息
        with pytest.raises(errortype, match=match):
            mticker.MaxNLocator(**kwargs)

    # 使用 pytest 的 parametrize 装饰器，对 test_padding 方法进行参数化测试
    @pytest.mark.parametrize('steps, result', [
        ([1, 2, 10], [1, 2, 10]),
        ([2, 10], [1, 2, 10]),
        ([1, 2], [1, 2, 10]),
        ([2], [1, 2, 10]),
    ])
    def test_padding(self, steps, result):
        # 创建 MaxNLocator 实例 loc，指定 steps 参数为给定的 steps
        loc = mticker.MaxNLocator(steps=steps)
        # 断言 loc 的 _steps 属性的所有元素与 result 相等
        assert (loc._steps == result).all()

# 定义一个测试类 TestLinearLocator，用于测试 LinearLocator 类的功能。
class TestLinearLocator:
    # 定义 test_basic 方法，测试基本情况下 LinearLocator 的功能
    def test_basic(self):
        # 创建 LinearLocator 实例 loc，指定 numticks 参数为 3
        loc = mticker.LinearLocator(numticks=3)
        # 定义测试值数组 test_value
        test_value = np.array([-0.8, -0.3, 0.2])
        # 断言调用 loc 的 tick_values 方法返回的 tick 值近似等于 test_value
        assert_almost_equal(loc.tick_values(-0.8, 0.2), test_value)

    # 定义 test_zero_numticks 方法，测试 numticks 为 0 时 LinearLocator 的功能
    def test_zero_numticks(self):
        # 创建 LinearLocator 实例 loc，指定 numticks 参数为 0
        loc = mticker.LinearLocator(numticks=0)
        # 断言调用 loc 的 tick_values 方法返回空列表
        loc.tick_values(-0.8, 0.2) == []
    def test_set_params(self):
        """
        Create linear locator with presets={}, numticks=2 and change it to
        something else. See if change was successful. Should not exception.
        """
        # 创建一个线性定位器，初始参数为 presets={}，numticks=2
        loc = mticker.LinearLocator(numticks=2)
        # 设置定位器的参数为 numticks=8, presets={(0, 1): []}
        loc.set_params(numticks=8, presets={(0, 1): []})
        # 断言确认参数设置成功
        assert loc.numticks == 8
        assert loc.presets == {(0, 1): []}

    def test_presets(self):
        # 使用给定的 presets 初始化线性定位器
        loc = mticker.LinearLocator(presets={(1, 2): [1, 1.25, 1.75],
                                             (0, 2): [0.5, 1.5]})
        # 断言不同输入范围下的 tick 值是否符合预期
        assert loc.tick_values(1, 2) == [1, 1.25, 1.75]
        assert loc.tick_values(2, 1) == [1, 1.25, 1.75]
        assert loc.tick_values(0, 2) == [0.5, 1.5]
        assert loc.tick_values(0.0, 2.0) == [0.5, 1.5]
        # 断言在区间 [0, 1] 内生成的 tick 值是否正确
        assert (loc.tick_values(0, 1) == np.linspace(0, 1, 11)).all()
class TestMultipleLocator:
    # 多重定位器的测试类

    def test_basic(self):
        # 测试基本情况下的多重定位器
        loc = mticker.MultipleLocator(base=3.147)
        # 设置基准值为3.147的多重定位器
        test_value = np.array([-9.441, -6.294, -3.147, 0., 3.147, 6.294,
                               9.441, 12.588])
        # 预期的测试数值数组
        assert_almost_equal(loc.tick_values(-7, 10), test_value)
        # 断言几乎相等，验证 tick_values 方法的输出是否与预期数组几乎相等

    def test_basic_with_offset(self):
        # 测试带偏移的多重定位器
        loc = mticker.MultipleLocator(base=3.147, offset=1.2)
        # 设置基准值为3.147、偏移为1.2的多重定位器
        test_value = np.array([-8.241, -5.094, -1.947, 1.2, 4.347, 7.494,
                               10.641])
        # 预期的测试数值数组
        assert_almost_equal(loc.tick_values(-7, 10), test_value)
        # 断言几乎相等，验证 tick_values 方法的输出是否与预期数组几乎相等

    def test_view_limits(self):
        """
        Test basic behavior of view limits.
        """
        # 测试视图限制的基本行为
        with mpl.rc_context({'axes.autolimit_mode': 'data'}):
            # 在 'data' 模式下创建多重定位器
            loc = mticker.MultipleLocator(base=3.147)
            # 设置基准值为3.147的多重定位器
            assert_almost_equal(loc.view_limits(-5, 5), (-5, 5))
            # 断言几乎相等，验证 view_limits 方法的输出是否符合预期 (-5, 5)

    def test_view_limits_round_numbers(self):
        """
        Test that everything works properly with 'round_numbers' for auto
        limit.
        """
        # 测试 'round_numbers' 自动限制模式下的正确工作
        with mpl.rc_context({'axes.autolimit_mode': 'round_numbers'}):
            # 在 'round_numbers' 模式下创建多重定位器
            loc = mticker.MultipleLocator(base=3.147)
            # 设置基准值为3.147的多重定位器
            assert_almost_equal(loc.view_limits(-4, 4), (-6.294, 6.294))
            # 断言几乎相等，验证 view_limits 方法的输出是否符合预期 (-6.294, 6.294)

    def test_view_limits_round_numbers_with_offset(self):
        """
        Test that everything works properly with 'round_numbers' for auto
        limit.
        """
        # 测试 'round_numbers' 自动限制模式下带偏移的正确工作
        with mpl.rc_context({'axes.autolimit_mode': 'round_numbers'}):
            # 在 'round_numbers' 模式下创建基准值为3.147、偏移为1.3的多重定位器
            loc = mticker.MultipleLocator(base=3.147, offset=1.3)
            assert_almost_equal(loc.view_limits(-4, 4), (-4.994, 4.447))
            # 断言几乎相等，验证 view_limits 方法的输出是否符合预期 (-4.994, 4.447)

    def test_view_limits_single_bin(self):
        """
        Test that 'round_numbers' works properly with a single bin.
        """
        # 测试 'round_numbers' 自动限制模式下单一区间的正确工作
        with mpl.rc_context({'axes.autolimit_mode': 'round_numbers'}):
            # 在 'round_numbers' 模式下创建最大N定位器，设定区间数为1
            loc = mticker.MaxNLocator(nbins=1)
            assert_almost_equal(loc.view_limits(-2.3, 2.3), (-4, 4))
            # 断言几乎相等，验证 view_limits 方法的输出是否符合预期 (-4, 4)

    def test_set_params(self):
        """
        Create multiple locator with 0.7 base, and change it to something else.
        See if change was successful.
        """
        # 测试设置参数的功能
        mult = mticker.MultipleLocator(base=0.7)
        # 创建基准值为0.7的多重定位器
        mult.set_params(base=1.7)
        # 设置基准值为1.7
        assert mult._edge.step == 1.7
        # 断言多重定位器的 _edge.step 属性是否为1.7，验证设置是否成功
        mult.set_params(offset=3)
        # 设置偏移为3
        assert mult._offset == 3
        # 断言多重定位器的 _offset 属性是否为3，验证设置是否成功


class TestAutoMinorLocator:
    def test_basic(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 1.39)
        ax.minorticks_on()
        test_value = np.array([0.05, 0.1, 0.15, 0.25, 0.3, 0.35, 0.45,
                               0.5, 0.55, 0.65, 0.7, 0.75, 0.85, 0.9,
                               0.95, 1.05, 1.1, 1.15, 1.25, 1.3, 1.35])
        # 预期的测试数值数组
        assert_almost_equal(ax.xaxis.get_ticklocs(minor=True), test_value)
        # 断言几乎相等，验证获取 x 轴次要刻度位置是否与预期数组几乎相等

    # NB: the following values are assuming that *xlim* is [0, 5]
    params = [
        (0, 0),  # no major tick => no minor tick either
        (1, 0)   # a single major tick => no minor tick
    ]
    # 参数列表，假设 *xlim* 为 [0, 5] 下的情况
    def test_first_and_last_minorticks(self):
        """
        Test that first and last minor tick appear as expected.
        """
        # This test is related to issue #22331
        
        # 创建一个新的图形和轴对象
        fig, ax = plt.subplots()
        
        # 设置 x 轴的数据范围
        ax.set_xlim(-1.9, 1.9)
        
        # 设置 x 轴的次要刻度定位器为自动
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        
        # 定义预期的测试数值
        test_value = np.array([-1.9, -1.8, -1.7, -1.6, -1.4, -1.3, -1.2, -1.1,
                               -0.9, -0.8, -0.7, -0.6, -0.4, -0.3, -0.2, -0.1,
                               0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.1,
                               1.2, 1.3, 1.4, 1.6, 1.7, 1.8, 1.9])
        
        # 断言 x 轴的次要刻度位置与预期值相近
        assert_almost_equal(ax.xaxis.get_ticklocs(minor=True), test_value)
        
        # 设置 x 轴的新数据范围
        ax.set_xlim(-5, 5)
        
        # 更新预期的测试数值
        test_value = np.array([-5.0, -4.5, -3.5, -3.0, -2.5, -1.5, -1.0, -0.5,
                               0.5, 1.0, 1.5, 2.5, 3.0, 3.5, 4.5, 5.0])
        
        # 再次断言 x 轴的次要刻度位置与更新后的预期值相近
        assert_almost_equal(ax.xaxis.get_ticklocs(minor=True), test_value)

    @pytest.mark.parametrize('nb_majorticks, expected_nb_minorticks', params)
    def test_low_number_of_majorticks(
            self, nb_majorticks, expected_nb_minorticks):
        # This test is related to issue #8804
        
        # 创建一个新的图形和轴对象
        fig, ax = plt.subplots()
        
        # 设置 x 轴的数据范围
        xlims = (0, 5)  # easier to test the different code paths
        ax.set_xlim(*xlims)
        
        # 设置 x 轴的主刻度位置
        ax.set_xticks(np.linspace(xlims[0], xlims[1], nb_majorticks))
        
        # 打开次要刻度
        ax.minorticks_on()
        
        # 设置 x 轴的次要刻度定位器为自动
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        
        # 断言 x 轴的次要刻度数目等于预期的数目
        assert len(ax.xaxis.get_minorticklocs()) == expected_nb_minorticks

    majorstep_minordivisions = [(1, 5),
                                (2, 4),
                                (2.5, 5),
                                (5, 5),
                                (10, 5)]

    # This test is meant to verify the parameterization for
    # test_number_of_minor_ticks
    def test_using_all_default_major_steps(self):
        # 在非经典模式下创建一个新的图形和轴对象
        with mpl.rc_context({'_internal.classic_mode': False}):
            # 提取主刻度步长
            majorsteps = [x[0] for x in self.majorstep_minordivisions]
            
            # 断言所有默认主步长与自动定位器的步长相近
            np.testing.assert_allclose(majorsteps,
                                       mticker.AutoLocator()._steps)

    @pytest.mark.parametrize('major_step, expected_nb_minordivisions',
                             majorstep_minordivisions)
    def test_number_of_minor_ticks(
            self, major_step, expected_nb_minordivisions):
        # 创建一个新的图形和轴对象
        fig, ax = plt.subplots()
        
        # 设置 x 轴的数据范围
        xlims = (0, major_step)
        ax.set_xlim(*xlims)
        
        # 设置 x 轴的主刻度位置
        ax.set_xticks(xlims)
        
        # 打开次要刻度
        ax.minorticks_on()
        
        # 设置 x 轴的次要刻度定位器为自动
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        
        # 计算次要分割的数目
        nb_minor_divisions = len(ax.xaxis.get_minorticklocs()) + 1
        
        # 断言次要分割数目等于预期的数目
        assert nb_minor_divisions == expected_nb_minordivisions

    limits = [(0, 1.39), (0, 0.139),
              (0, 0.11e-19), (0, 0.112e-12),
              (-2.0e-07, -3.3e-08), (1.20e-06, 1.42e-06),
              (-1.34e-06, -1.44e-06), (-8.76e-07, -1.51e-06)]
    # 创建一个包含多个子列表的参考列表，每个子列表包含一组数字
    reference = [
        [0.05, 0.1, 0.15, 0.25, 0.3, 0.35, 0.45, 0.5, 0.55, 0.65, 0.7,
         0.75, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.25, 1.3, 1.35],
        [0.005, 0.01, 0.015, 0.025, 0.03, 0.035, 0.045, 0.05, 0.055, 0.065,
         0.07, 0.075, 0.085, 0.09, 0.095, 0.105, 0.11, 0.115, 0.125, 0.13,
         0.135],
        [5.00e-22, 1.00e-21, 1.50e-21, 2.50e-21, 3.00e-21, 3.50e-21, 4.50e-21,
         5.00e-21, 5.50e-21, 6.50e-21, 7.00e-21, 7.50e-21, 8.50e-21, 9.00e-21,
         9.50e-21, 1.05e-20, 1.10e-20],
        [5.00e-15, 1.00e-14, 1.50e-14, 2.50e-14, 3.00e-14, 3.50e-14, 4.50e-14,
         5.00e-14, 5.50e-14, 6.50e-14, 7.00e-14, 7.50e-14, 8.50e-14, 9.00e-14,
         9.50e-14, 1.05e-13, 1.10e-13],
        [-1.95e-07, -1.90e-07, -1.85e-07, -1.75e-07, -1.70e-07, -1.65e-07,
         -1.55e-07, -1.50e-07, -1.45e-07, -1.35e-07, -1.30e-07, -1.25e-07,
         -1.15e-07, -1.10e-07, -1.05e-07, -9.50e-08, -9.00e-08, -8.50e-08,
         -7.50e-08, -7.00e-08, -6.50e-08, -5.50e-08, -5.00e-08, -4.50e-08,
         -3.50e-08],
        [1.21e-06, 1.22e-06, 1.23e-06, 1.24e-06, 1.26e-06, 1.27e-06, 1.28e-06,
         1.29e-06, 1.31e-06, 1.32e-06, 1.33e-06, 1.34e-06, 1.36e-06, 1.37e-06,
         1.38e-06, 1.39e-06, 1.41e-06, 1.42e-06],
        [-1.435e-06, -1.430e-06, -1.425e-06, -1.415e-06, -1.410e-06,
         -1.405e-06, -1.395e-06, -1.390e-06, -1.385e-06, -1.375e-06,
         -1.370e-06, -1.365e-06, -1.355e-06, -1.350e-06, -1.345e-06],
        [-1.48e-06, -1.46e-06, -1.44e-06, -1.42e-06, -1.38e-06, -1.36e-06,
         -1.34e-06, -1.32e-06, -1.28e-06, -1.26e-06, -1.24e-06, -1.22e-06,
         -1.18e-06, -1.16e-06, -1.14e-06, -1.12e-06, -1.08e-06, -1.06e-06,
         -1.04e-06, -1.02e-06, -9.80e-07, -9.60e-07, -9.40e-07, -9.20e-07,
         -8.80e-07]
    ]
    
    # 将两个列表进行压缩，形成元组的列表，每个元组包含一个限制值和一个参考值列表
    additional_data = list(zip(limits, reference))
    
    # 使用 pytest.mark.parametrize 装饰器，定义一个参数化测试函数 test_additional
    @pytest.mark.parametrize('lim, ref', additional_data)
    def test_additional(self, lim, ref):
        # 创建一个包含单个图形和轴对象的图形对象
        fig, ax = plt.subplots()
    
        # 启用次要刻度线
        ax.minorticks_on()
        # 绘制次要网格线，限定在 y 轴上，线宽为 1
        ax.grid(True, 'minor', 'y', linewidth=1)
        # 绘制主要网格线，线条颜色为黑色，线宽为 1
        ax.grid(True, 'major', color='k', linewidth=1)
        # 设置 y 轴的数值范围
        ax.set_ylim(lim)
    
        # 使用 assert_almost_equal 检查轴对象的次要刻度位置是否接近于给定的参考值列表
        assert_almost_equal(ax.yaxis.get_ticklocs(minor=True), ref)
    
    # 使用 pytest.mark.parametrize 装饰器，定义一个参数化测试函数，测试不同的 rcparam 设置
    @pytest.mark.parametrize('use_rcparam', [False, True])
    @pytest.mark.parametrize(
        'lim, ref', [
            # 第一个参数化组合，设置 y 轴范围为 (0, 1.39)，参考值为特定的列表
            ((0, 1.39),
             [0.05, 0.1, 0.15, 0.25, 0.3, 0.35, 0.45, 0.5, 0.55, 0.65, 0.7,
              0.75, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.25, 1.3, 1.35]),
            # 第二个参数化组合，设置 y 轴范围为 (0, 0.139)，参考值为特定的列表
            ((0, 0.139),
             [0.005, 0.01, 0.015, 0.025, 0.03, 0.035, 0.045, 0.05, 0.055,
              0.065, 0.07, 0.075, 0.085, 0.09, 0.095, 0.105, 0.11, 0.115,
              0.125, 0.13, 0.135]),
        ])
    def test_number_of_minor_ticks_int(self, n, lim, ref, use_rcparam):
        # 根据是否使用 rc 参数来设置上下文和关键字参数
        if use_rcparam:
            # 使用 rc 参数设置上下文，控制 x 和 y 轴的次要刻度数
            context = {'xtick.minor.ndivs': n, 'ytick.minor.ndivs': n}
            kwargs = {}  # 空字典，用于传递给 AutoMinorLocator 的关键字参数
        else:
            context = {}  # 不使用 rc 参数时，上下文为空字典
            kwargs = {'n': n}  # 指定 AutoMinorLocator 的刻度数为 n

        # 进入上下文环境，应用设置的 rc 参数或空字典
        with mpl.rc_context(context):
            # 创建新的图形和轴对象
            fig, ax = plt.subplots()
            # 设置 x 和 y 轴的主要刻度范围
            ax.set_xlim(*lim)
            ax.set_ylim(*lim)
            # 设置 x 轴的主要刻度为 1 的倍数刻度
            ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
            # 设置 x 轴的次要刻度为自动计算的刻度
            ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(**kwargs))
            # 设置 y 轴的主要刻度为 1 的倍数刻度
            ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
            # 设置 y 轴的次要刻度为自动计算的刻度
            ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(**kwargs))
            # 断言 x 轴和 y 轴的次要刻度位置与预期值 ref 几乎相等
            assert_almost_equal(ax.xaxis.get_ticklocs(minor=True), ref)
            assert_almost_equal(ax.yaxis.get_ticklocs(minor=True), ref)
class TestLogLocator:
    def test_basic(self):
        # 创建一个对数定位器对象，设置期望的刻度数目为5
        loc = mticker.LogLocator(numticks=5)
        # 断言调用tick_values方法时，当起始和结束值为0和1000时，会引发值错误异常
        with pytest.raises(ValueError):
            loc.tick_values(0, 1000)

        # 准备一个测试值数组，用于断言调用tick_values方法时返回的结果近似等于该数组
        test_value = np.array([1.00000000e-05, 1.00000000e-03, 1.00000000e-01,
                               1.00000000e+01, 1.00000000e+03, 1.00000000e+05,
                               1.00000000e+07, 1.000000000e+09])
        assert_almost_equal(loc.tick_values(0.001, 1.1e5), test_value)

        # 创建一个基数为2的对数定位器对象
        loc = mticker.LogLocator(base=2)
        # 准备一个测试值数组，用于断言调用tick_values方法时返回的结果近似等于该数组
        test_value = np.array([0.5, 1., 2., 4., 8., 16., 32., 64., 128., 256.])
        assert_almost_equal(loc.tick_values(1, 100), test_value)

    def test_polar_axes(self):
        """
        Polar Axes have a different ticking logic.
        极坐标轴具有不同的刻度逻辑。
        """
        # 创建一个极坐标子图
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        # 设置y轴的比例为对数尺度，限制y轴的范围为1到100
        ax.set_yscale('log')
        ax.set_ylim(1, 100)
        # 断言获取y轴刻度值的结果与预期的数组相等
        assert_array_equal(ax.get_yticks(), [10, 100, 1000])

    def test_switch_to_autolocator(self):
        # 创建一个所有子刻度都显示的对数定位器对象
        loc = mticker.LogLocator(subs="all")
        # 断言调用tick_values方法时，起始值和结束值为0.45和0.55时，返回的刻度数组等于预期的数组
        assert_array_equal(loc.tick_values(0.45, 0.55),
                           [0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56])
        # 检查确保1.0和10不在刻度值数组中，因为这是一个次要定位器
        loc = mticker.LogLocator(subs=np.arange(2, 10))
        assert 1.0 not in loc.tick_values(0.9, 20.)
        assert 10.0 not in loc.tick_values(0.9, 20.)

    def test_set_params(self):
        """
        Create log locator with default value, base=10.0, subs=[1.0],
        numdecs=4, numticks=15 and change it to something else.
        See if change was successful. Should not raise exception.
        创建一个基于默认值的对数定位器，base=10.0, subs=[1.0],
        numdecs=4, numticks=15，并将其更改为其他值。
        检查更改是否成功。不应引发异常。
        """
        # 创建一个默认参数的对数定位器对象
        loc = mticker.LogLocator()
        # 使用pytest的警告上下文检查，设置参数numticks=7, numdecs=8, subs=[2.0], base=4，预期会有MatplotlibDeprecationWarning警告
        with pytest.warns(mpl.MatplotlibDeprecationWarning, match="numdecs"):
            loc.set_params(numticks=7, numdecs=8, subs=[2.0], base=4)
        # 断言检查numticks参数是否被成功设置为7
        assert loc.numticks == 7
        # 使用pytest的警告上下文检查，检查numdecs参数是否被成功设置为8
        with pytest.warns(mpl.MatplotlibDeprecationWarning, match="numdecs"):
            assert loc.numdecs == 8
        # 断言检查_base属性是否被成功设置为4
        assert loc._base == 4
        # 断言检查_subs属性是否被成功设置为[2.0]
        assert list(loc._subs) == [2.0]

    def test_tick_values_correct(self):
        # 创建一个带有特定子刻度的对数定位器对象
        ll = mticker.LogLocator(subs=(1, 2, 5))
        # 准备一个测试值数组，用于断言调用tick_values方法时返回的结果近似等于该数组
        test_value = np.array([1.e-01, 2.e-01, 5.e-01, 1.e+00, 2.e+00, 5.e+00,
                               1.e+01, 2.e+01, 5.e+01, 1.e+02, 2.e+02, 5.e+02,
                               1.e+03, 2.e+03, 5.e+03, 1.e+04, 2.e+04, 5.e+04,
                               1.e+05, 2.e+05, 5.e+05, 1.e+06, 2.e+06, 5.e+06,
                               1.e+07, 2.e+07, 5.e+07, 1.e+08, 2.e+08, 5.e+08])
        assert_almost_equal(ll.tick_values(1, 1e7), test_value)
    # 设置 Matplotlib 的配置项，关闭内部经典模式
    mpl.rcParams['_internal.classic_mode'] = False
    
    # 创建一个对数定位器，指定子标签为 (1, 2, 5)
    ll = mticker.LogLocator(subs=(1, 2, 5))
    
    # 定义用于测试的数值数组，包含对数尺度下的特定测试值
    test_value = np.array([1.e-01, 2.e-01, 5.e-01, 1.e+00, 2.e+00, 5.e+00,
                           1.e+01, 2.e+01, 5.e+01, 1.e+02, 2.e+02, 5.e+02,
                           1.e+03, 2.e+03, 5.e+03, 1.e+04, 2.e+04, 5.e+04,
                           1.e+05, 2.e+05, 5.e+05, 1.e+06, 2.e+06, 5.e+06,
                           1.e+07, 2.e+07, 5.e+07, 1.e+08, 2.e+08, 5.e+08,
                           1.e+09, 2.e+09, 5.e+09])
    
    # 断言调用对数定位器的 tick_values 方法生成的刻度值与预期测试值接近
    assert_almost_equal(ll.tick_values(1, 1e8), test_value)

    # 创建一个随机数生成器对象
    rng = np.random.default_rng(19680801)
    
    # 创建一个包含随机数据和空列表的虚拟数据集
    dummy_data = [rng.normal(size=100), [], []]
    
    # 创建具有共享 X 和 Y 轴的子图
    fig, axes = plt.subplots(len(dummy_data), sharex=True, sharey=True)

    # 遍历每个子图及其对应的数据
    for ax, data in zip(axes.flatten(), dummy_data):
        # 在当前子图上绘制数据的直方图，设定分布的 bin 数量为 10
        ax.hist(data, bins=10)
        
        # 设置 Y 轴为对数尺度，并剪切非正数部分
        ax.set_yscale('log', nonpositive='clip')

    # 遍历每个子图，断言它们的 Y 轴刻度与第一个子图相同
    for ax in axes.flatten():
        assert all(ax.get_yticks() == axes[0].get_yticks())
        assert ax.get_ylim() == axes[0].get_ylim()
class TestNullLocator:
    def test_set_params(self):
        """
        Create null locator, and attempt to call set_params() on it.
        Should not exception, and should raise a warning.
        """
        # 创建空定位器对象
        loc = mticker.NullLocator()
        # 使用 pytest 来检查是否会发出 UserWarning 警告
        with pytest.warns(UserWarning):
            # 调用空定位器的 set_params() 方法
            loc.set_params()


class _LogitHelper:
    @staticmethod
    def isclose(x, y):
        # 检查两个数是否在相对误差范围内相等
        return (np.isclose(-np.log(1/x-1), -np.log(1/y-1))
                if 0 < x < 1 and 0 < y < 1 else False)

    @staticmethod
    def assert_almost_equal(x, y):
        # 断言 x 和 y 都在 (0, 1) 范围内
        ax = np.array(x)
        ay = np.array(y)
        assert np.all(ax > 0) and np.all(ax < 1)
        assert np.all(ay > 0) and np.all(ay < 1)
        # 计算对数变换后的值，并断言它们近似相等
        lx = -np.log(1/ax-1)
        ly = -np.log(1/ay-1)
        assert_almost_equal(lx, ly)


class TestLogitLocator:
    ref_basic_limits = [
        (5e-2, 1 - 5e-2),
        (5e-3, 1 - 5e-3),
        (5e-4, 1 - 5e-4),
        (5e-5, 1 - 5e-5),
        (5e-6, 1 - 5e-6),
        (5e-7, 1 - 5e-7),
        (5e-8, 1 - 5e-8),
        (5e-9, 1 - 5e-9),
    ]

    ref_basic_major_ticks = [
        1 / (10 ** np.arange(1, 3)),
        1 / (10 ** np.arange(1, 4)),
        1 / (10 ** np.arange(1, 5)),
        1 / (10 ** np.arange(1, 6)),
        1 / (10 ** np.arange(1, 7)),
        1 / (10 ** np.arange(1, 8)),
        1 / (10 ** np.arange(1, 9)),
        1 / (10 ** np.arange(1, 10)),
    ]

    ref_maxn_limits = [(0.4, 0.6), (5e-2, 2e-1), (1 - 2e-1, 1 - 5e-2)]

    @pytest.mark.parametrize(
        "lims, expected_low_ticks",
        zip(ref_basic_limits, ref_basic_major_ticks),
    )
    def test_basic_major(self, lims, expected_low_ticks):
        """
        Create logit locator with huge number of major, and tests ticks.
        """
        # 期望的 ticks 包括预期的低值 ticks、0.5 和预期的高值 ticks
        expected_ticks = sorted(
            [*expected_low_ticks, 0.5, *(1 - expected_low_ticks)]
        )
        # 创建 LogitLocator 对象
        loc = mticker.LogitLocator(nbins=100)
        # 使用 LogitHelper 中的方法来比较实际 ticks 和期望 ticks 的近似性
        _LogitHelper.assert_almost_equal(
            loc.tick_values(*lims),
            expected_ticks
        )

    @pytest.mark.parametrize("lims", ref_maxn_limits)
    def test_maxn_major(self, lims):
        """
        When the axis is zoomed, the locator must have the same behavior as
        MaxNLocator.
        """
        # 创建 LogitLocator 和 MaxNLocator 对象
        loc = mticker.LogitLocator(nbins=100)
        maxn_loc = mticker.MaxNLocator(nbins=100, steps=[1, 2, 5, 10])
        # 对不同的 nbins 值进行测试
        for nbins in (4, 8, 16):
            # 设置 LogitLocator 和 MaxNLocator 的 nbins 参数
            loc.set_params(nbins=nbins)
            maxn_loc.set_params(nbins=nbins)
            # 获取 LogitLocator 和 MaxNLocator 的 ticks
            ticks = loc.tick_values(*lims)
            maxn_ticks = maxn_loc.tick_values(*lims)
            # 断言两者的形状相同，且所有元素都相等
            assert ticks.shape == maxn_ticks.shape
            assert (ticks == maxn_ticks).all()

    @pytest.mark.parametrize("lims", ref_basic_limits + ref_maxn_limits)
    def test_nbins_major(self, lims):
        """
        Assert logit locator for respecting nbins param.
        """
        
        # 计算基本所需的刻度数
        basic_needed = int(-np.floor(np.log10(lims[0]))) * 2 + 1
        # 创建 LogitLocator 对象，设定 nbins 参数为 100
        loc = mticker.LogitLocator(nbins=100)
        # 逐步减少 nbins 参数直到基本所需刻度数，进行断言
        for nbins in range(basic_needed, 2, -1):
            loc.set_params(nbins=nbins)
            # 断言生成的刻度值不超过 nbins + 2
            assert len(loc.tick_values(*lims)) <= nbins + 2

    @pytest.mark.parametrize(
        "lims, expected_low_ticks",
        zip(ref_basic_limits, ref_basic_major_ticks),
    )
    def test_minor(self, lims, expected_low_ticks):
        """
        In large scale, test the presence of minor,
        and assert no minor when major are subsampled.
        """
        
        # 组装预期的所有刻度值
        expected_ticks = sorted(
            [*expected_low_ticks, 0.5, *(1 - expected_low_ticks)]
        )
        # 计算基本所需的刻度数
        basic_needed = len(expected_ticks)
        # 创建 LogitLocator 对象，设定 nbins 参数为 100
        loc = mticker.LogitLocator(nbins=100)
        # 创建用于生成次要刻度的 LogitLocator 对象，设定 nbins 参数为 100，同时启用次要刻度
        minor_loc = mticker.LogitLocator(nbins=100, minor=True)
        # 逐步减少 nbins 参数直到基本所需刻度数，进行断言
        for nbins in range(basic_needed, 2, -1):
            loc.set_params(nbins=nbins)
            minor_loc.set_params(nbins=nbins)
            major_ticks = loc.tick_values(*lims)
            minor_ticks = minor_loc.tick_values(*lims)
            if len(major_ticks) >= len(expected_ticks):
                # 如果未进行主刻度的抽样，次要刻度应当较多
                assert (len(major_ticks) - 1) * 5 < len(minor_ticks)
            else:
                # 如果进行了主刻度的抽样
                _LogitHelper.assert_almost_equal(
                    sorted([*major_ticks, *minor_ticks]), expected_ticks)

    def test_minor_attr(self):
        """
        Test minor attribute of LogitLocator object.
        """
        
        # 创建 LogitLocator 对象
        loc = mticker.LogitLocator(nbins=100)
        # 断言初始时 minor 属性为 False
        assert not loc.minor
        # 将 minor 属性设为 True，并断言为 True
        loc.minor = True
        assert loc.minor
        # 调用 set_params 方法将 minor 属性设为 False，并断言为 False
        loc.set_params(minor=False)
        assert not loc.minor

    acceptable_vmin_vmax = [
        *(2.5 ** np.arange(-3, 0)),
        *(1 - 2.5 ** np.arange(-3, 0)),
    ]

    @pytest.mark.parametrize(
        "lims",
        [
            (a, b)
            for (a, b) in itertools.product(acceptable_vmin_vmax, repeat=2)
            if a != b
        ],
    )
    def test_nonsingular_ok(self, lims):
        """
        Test the nonsingular method of LogitLocator for acceptable values.
        """
        
        # 创建 LogitLocator 对象
        loc = mticker.LogitLocator()
        # 调用 nonsingular 方法，验证返回值与输入值排序后相同
        lims2 = loc.nonsingular(*lims)
        assert sorted(lims) == sorted(lims2)

    @pytest.mark.parametrize("okval", acceptable_vmin_vmax)
    def test_nonsingular_nok(self, okval):
        """
        Test the nonsingular method of LogitLocator for non-acceptable values.
        """
        
        # 创建 LogitLocator 对象
        loc = mticker.LogitLocator()
        # 设定 vmin 和 vmax 的值，调用 nonsingular 方法，验证返回值符合预期
        vmin, vmax = (-1, okval)
        vmin2, vmax2 = loc.nonsingular(vmin, vmax)
        assert vmax2 == vmax
        assert 0 < vmin2 < vmax2
        vmin, vmax = (okval, 2)
        vmin2, vmax2 = loc.nonsingular(vmin, vmax)
        assert vmin2 == vmin
        assert vmin2 < vmax2 < 1
class TestFixedLocator:
    def test_set_params(self):
        """
        创建一个固定定位器，使用5个bins，并将其更改为其他值。
        检查更改是否成功。
        不应该抛出异常。
        """
        # 创建固定定位器对象，定义bins为0到23之间的整数，初始使用5个bins
        fixed = mticker.FixedLocator(range(0, 24), nbins=5)
        # 设置固定定位器的参数，将bins的数量更改为7
        fixed.set_params(nbins=7)
        # 断言检查bins的数量是否变为了7
        assert fixed.nbins == 7


class TestIndexLocator:
    def test_set_params(self):
        """
        创建一个索引定位器，使用基数3和偏移4，并将其更改为其他值。
        检查更改是否成功。
        不应该抛出异常。
        """
        # 创建索引定位器对象，定义基数为3，偏移为4
        index = mticker.IndexLocator(base=3, offset=4)
        # 设置索引定位器的参数，将基数和偏移都更改为7
        index.set_params(base=7, offset=7)
        # 断言检查基数和偏移是否都变为了7
        assert index._base == 7
        assert index.offset == 7


class TestSymmetricalLogLocator:
    def test_set_params(self):
        """
        创建对称对数定位器，默认subs为[1.0]，numticks为15，并将其更改为其他值。
        检查更改是否成功。
        不应该抛出异常。
        """
        # 创建对称对数定位器对象，基数为10，线性阈值为1
        sym = mticker.SymmetricalLogLocator(base=10, linthresh=1)
        # 设置对称对数定位器的参数，将subs更改为[2.0]，numticks更改为8
        sym.set_params(subs=[2.0], numticks=8)
        # 断言检查subs和numticks是否分别变为了[2.0]和8
        assert sym._subs == [2.0]
        assert sym.numticks == 8

    @pytest.mark.parametrize(
            'vmin, vmax, expected',
            [
                (0, 1, [0, 1]),
                (-1, 1, [-1, 0, 1]),
            ],
    )
    def test_values(self, vmin, vmax, expected):
        # https://github.com/matplotlib/matplotlib/issues/25945
        # 创建对称对数定位器对象，基数为10，线性阈值为1
        sym = mticker.SymmetricalLogLocator(base=10, linthresh=1)
        # 调用tick_values方法计算ticks
        ticks = sym.tick_values(vmin=vmin, vmax=vmax)
        # 断言检查计算得到的ticks是否符合预期的expected值
        assert_array_equal(ticks, expected)

    def test_subs(self):
        # 创建对称对数定位器对象，基数为10，线性阈值为1，subs为[2.0, 4.0]
        sym = mticker.SymmetricalLogLocator(base=10, linthresh=1, subs=[2.0, 4.0])
        # 创建虚拟轴
        sym.create_dummy_axis()
        # 设置虚拟轴的视图间隔为-10到10
        sym.axis.set_view_interval(-10, 10)
        # 断言检查计算得到的ticks是否符合预期的数组
        assert (sym() == [-20., -40.,  -2.,  -4.,   0.,   2.,   4.,  20.,  40.]).all()

    def test_extending(self):
        # 创建对称对数定位器对象，基数为10，线性阈值为1
        sym = mticker.SymmetricalLogLocator(base=10, linthresh=1)
        # 创建虚拟轴
        sym.create_dummy_axis()
        # 设置虚拟轴的视图间隔为8到9
        sym.axis.set_view_interval(8, 9)
        # 断言检查计算得到的ticks是否符合预期的数组
        assert (sym() == [1.0]).all()
        # 设置虚拟轴的视图间隔为8到12
        sym.axis.set_view_interval(8, 12)
        # 断言检查计算得到的ticks是否符合预期的数组
        assert (sym() == [1.0, 10.0]).all()
        # 断言检查view_limits方法对给定范围的视图限制是否符合预期
        assert sym.view_limits(10, 10) == (1, 100)
        assert sym.view_limits(-10, -10) == (-100, -1)
        assert sym.view_limits(0, 0) == (-0.001, 0.001)


class TestAsinhLocator:
    def test_init(self):
        # 创建反正弦定位器对象，线性宽度为2.718，numticks为19
        lctr = mticker.AsinhLocator(linear_width=2.718, numticks=19)
        # 断言检查初始化参数是否符合预期
        assert lctr.linear_width == 2.718
        assert lctr.numticks == 19
        assert lctr.base == 10
    def test_set_params(self):
        # 创建一个 AsinhLocator 对象，设置其参数：线性宽度为 5，刻度数量为 17，符号阈值为 0.125，基数为 4，替换值为 (2.5, 3.25)
        lctr = mticker.AsinhLocator(linear_width=5,
                                    numticks=17, symthresh=0.125,
                                    base=4, subs=(2.5, 3.25))
        # 断言验证设置的参数是否正确
        assert lctr.numticks == 17
        assert lctr.symthresh == 0.125
        assert lctr.base == 4
        assert lctr.subs == (2.5, 3.25)

        # 设置参数 numticks 为 23，并验证设置结果
        lctr.set_params(numticks=23)
        assert lctr.numticks == 23
        # 设置参数为 None，并验证 numticks 是否保持不变
        lctr.set_params(None)
        assert lctr.numticks == 23

        # 设置参数 symthresh 为 0.5，并验证设置结果
        lctr.set_params(symthresh=0.5)
        assert lctr.symthresh == 0.5
        # 设置参数为 None，并验证 symthresh 是否保持不变
        lctr.set_params(symthresh=None)
        assert lctr.symthresh == 0.5

        # 设置参数 base 为 7，并验证设置结果
        lctr.set_params(base=7)
        assert lctr.base == 7
        # 设置参数为 None，并验证 base 是否保持不变
        lctr.set_params(base=None)
        assert lctr.base == 7

        # 设置参数 subs 为 (2, 4.125)，并验证设置结果
        lctr.set_params(subs=(2, 4.125))
        assert lctr.subs == (2, 4.125)
        # 设置参数为 None，并验证 subs 是否被设为 None
        lctr.set_params(subs=None)
        assert lctr.subs == (2, 4.125)
        # 设置参数为 []，并验证 subs 是否被设为 None
        lctr.set_params(subs=[])
        assert lctr.subs is None

    def test_linear_values(self):
        # 创建一个 AsinhLocator 对象，设置其参数：线性宽度为 100，刻度数量为 11，基数为 0
        lctr = mticker.AsinhLocator(linear_width=100, numticks=11, base=0)

        # 验证 tick_values 方法的输出是否接近预期的结果
        assert_almost_equal(lctr.tick_values(-1, 1),
                            np.arange(-1, 1.01, 0.2))
        assert_almost_equal(lctr.tick_values(-0.1, 0.1),
                            np.arange(-0.1, 0.101, 0.02))
        assert_almost_equal(lctr.tick_values(-0.01, 0.01),
                            np.arange(-0.01, 0.0101, 0.002))

    def test_wide_values(self):
        # 创建一个 AsinhLocator 对象，设置其参数：线性宽度为 0.1，刻度数量为 11，基数为 0
        lctr = mticker.AsinhLocator(linear_width=0.1, numticks=11, base=0)

        # 验证 tick_values 方法的输出是否接近预期的结果
        assert_almost_equal(lctr.tick_values(-100, 100),
                            [-100, -20, -5, -1, -0.2,
                             0, 0.2, 1, 5, 20, 100])
        assert_almost_equal(lctr.tick_values(-1000, 1000),
                            [-1000, -100, -20, -3, -0.4,
                             0, 0.4, 3, 20, 100, 1000])

    def test_near_zero(self):
        """Check that manually injected zero will supersede nearby tick"""
        # 创建一个 AsinhLocator 对象，设置其参数：线性宽度为 100，刻度数量为 3，基数为 0
        lctr = mticker.AsinhLocator(linear_width=100, numticks=3, base=0)

        # 验证 tick_values 方法的输出是否接近预期的结果
        assert_almost_equal(lctr.tick_values(-1.1, 0.9), [-1.0, 0.0, 0.9])

    def test_fallback(self):
        # 创建一个 AsinhLocator 对象，设置其参数：线性宽度为 1.0，刻度数量为 11
        lctr = mticker.AsinhLocator(1.0, numticks=11)

        # 验证 tick_values 方法的输出是否接近预期的结果
        assert_almost_equal(lctr.tick_values(101, 102),
                            np.arange(101, 102.01, 0.1))

    def test_symmetrizing(self):
        # 创建一个 AsinhLocator 对象，设置其参数：线性宽度为 1，刻度数量为 3，符号阈值为 0.25，基数为 0
        lctr = mticker.AsinhLocator(linear_width=1, numticks=3,
                                    symthresh=0.25, base=0)
        # 创建一个虚拟坐标轴
        lctr.create_dummy_axis()

        # 设置坐标轴的视图间隔为 (-1, 2)，并验证其输出是否接近预期的结果
        lctr.axis.set_view_interval(-1, 2)
        assert_almost_equal(lctr(), [-1, 0, 2])

        # 设置坐标轴的视图间隔为 (-1, 0.9)，并验证其输出是否接近预期的结果
        lctr.axis.set_view_interval(-1, 0.9)
        assert_almost_equal(lctr(), [-1, 0, 1])

        # 设置坐标轴的视图间隔为 (-0.85, 1.05)，并验证其输出是否接近预期的结果
        lctr.axis.set_view_interval(-0.85, 1.05)
        assert_almost_equal(lctr(), [-1, 0, 1])

        # 设置坐标轴的视图间隔为 (1, 1.1)，并验证其输出是否接近预期的结果
        lctr.axis.set_view_interval(1, 1.1)
        assert_almost_equal(lctr(), [1, 1.05, 1.1])
    # 定义一个测试方法，用于测试 AsinhLocator 类在不同配置下的行为
    def test_base_rounding(self):
        # 创建一个 AsinhLocator 对象，指定线性宽度为 1，生成 8 个刻度点，
        # 基数为 10，辅助刻度为 (1, 3, 5)
        lctr10 = mticker.AsinhLocator(linear_width=1, numticks=8,
                                      base=10, subs=(1, 3, 5))
        # 断言 lctr10 对象生成的刻度值与期望值几乎相等
        assert_almost_equal(lctr10.tick_values(-110, 110),
                            [-500, -300, -100, -50, -30, -10, -5, -3, -1,
                             -0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5,
                             1, 3, 5, 10, 30, 50, 100, 300, 500])

        # 创建一个 AsinhLocator 对象，指定线性宽度为 1，生成 20 个刻度点，
        # 基数为 5
        lctr5 = mticker.AsinhLocator(linear_width=1, numticks=20, base=5)
        # 断言 lctr5 对象生成的刻度值与期望值几乎相等
        assert_almost_equal(lctr5.tick_values(-1050, 1050),
                            [-625, -125, -25, -5, -1, -0.2, 0,
                             0.2, 1, 5, 25, 125, 625])
class TestScalarFormatter:
    # 偏移量测试数据，每个元组包含左值、右值和预期偏移量
    offset_data = [
        (123, 189, 0),
        (-189, -123, 0),
        (12341, 12349, 12340),
        (-12349, -12341, -12340),
        (99999.5, 100010.5, 100000),
        (-100010.5, -99999.5, -100000),
        (99990.5, 100000.5, 100000),
        (-100000.5, -99990.5, -100000),
        (1233999, 1234001, 1234000),
        (-1234001, -1233999, -1234000),
        (1, 1, 1),
        (123, 123, 0),
        # Test cases courtesy of @WeatherGod
        (.4538, .4578, .45),
        (3789.12, 3783.1, 3780),
        (45124.3, 45831.75, 45000),
        (0.000721, 0.0007243, 0.00072),
        (12592.82, 12591.43, 12590),
        (9., 12., 0),
        (900., 1200., 0),
        (1900., 1200., 0),
        (0.99, 1.01, 1),
        (9.99, 10.01, 10),
        (99.99, 100.01, 100),
        (5.99, 6.01, 6),
        (15.99, 16.01, 16),
        (-0.452, 0.492, 0),
        (-0.492, 0.492, 0),
        (12331.4, 12350.5, 12300),
        (-12335.3, 12335.3, 0),
    ]

    # 是否使用偏移量的测试数据，True 或 False
    use_offset_data = [True, False]

    # 是否使用数学文本的测试数据，True 或 False
    useMathText_data = [True, False]

    # 科学计数法的限制和配置的测试数据
    # (sci_type, scilimits, lim, orderOfMag, fewticks)
    scilimits_data = [
        (False, (0, 0), (10.0, 20.0), 0, False),
        (True, (-2, 2), (-10, 20), 0, False),
        (True, (-2, 2), (-20, 10), 0, False),
        (True, (-2, 2), (-110, 120), 2, False),
        (True, (-2, 2), (-120, 110), 2, False),
        (True, (-2, 2), (-.001, 0.002), -3, False),
        (True, (-7, 7), (0.18e10, 0.83e10), 9, True),
        (True, (0, 0), (-1e5, 1e5), 5, False),
        (True, (6, 6), (-1e5, 1e5), 6, False),
    ]

    # 光标位置的测试数据，每个元素为一个列表，包含浮点数和预期字符串格式
    cursor_data = [
        [0., "0.000"],
        [0.0123, "0.012"],
        [0.123, "0.123"],
        [1.23,  "1.230"],
        [12.3, "12.300"],
    ]

    # 格式化数据的测试数据，每个元组包含浮点数和其预期字符串格式
    format_data = [
        (.1, "1e-1"),
        (.11, "1.1e-1"),
        (1e8, "1e8"),
        (1.1e8, "1.1e8"),
    ]

    @pytest.mark.parametrize('unicode_minus, result',
                             [(True, "\N{MINUS SIGN}1"), (False, "-1")])
    def test_unicode_minus(self, unicode_minus, result):
        # 设置 Matplotlib 的 Unicode 减号显示设置
        mpl.rcParams['axes.unicode_minus'] = unicode_minus
        # 断言获取的数据短格式化结果与预期结果一致
        assert (
            plt.gca().xaxis.get_major_formatter().format_data_short(-1).strip()
            == result)

    @pytest.mark.parametrize('left, right, offset', offset_data)
    def test_offset_value(self, left, right, offset):
        # 创建图形和坐标轴对象
        fig, ax = plt.subplots()
        # 获取 X 轴主要格式化器
        formatter = ax.xaxis.get_major_formatter()

        # 如果左右值相等，则使用 UserWarning 捕获环境，以确保警告匹配
        with (pytest.warns(UserWarning, match='Attempting to set identical')
              if left == right else nullcontext()):
            ax.set_xlim(left, right)
        # 更新 X 轴刻度
        ax.xaxis._update_ticks()
        # 断言格式化器的偏移量是否符合预期
        assert formatter.offset == offset

        # 再次设置反向的左右值
        with (pytest.warns(UserWarning, match='Attempting to set identical')
              if left == right else nullcontext()):
            ax.set_xlim(right, left)
        # 再次更新 X 轴刻度
        ax.xaxis._update_ticks()
        # 断言格式化器的偏移量是否符合预期
        assert formatter.offset == offset

    @pytest.mark.parametrize('use_offset', use_offset_data)
    # 使用参数 use_offset 来测试 ScalarFormatter 的偏移设置
    def test_use_offset(self, use_offset):
        # 在上下文中设置 matplotlib 的 rc 参数，控制坐标轴格式化器是否使用偏移量
        with mpl.rc_context({'axes.formatter.useoffset': use_offset}):
            # 创建 ScalarFormatter 对象
            tmp_form = mticker.ScalarFormatter()
            # 断言 ScalarFormatter 是否使用了指定的偏移设置
            assert use_offset == tmp_form.get_useOffset()
            # 断言 ScalarFormatter 的偏移量是否为 0
            assert tmp_form.offset == 0

    # 使用参数 use_math_text 来测试 ScalarFormatter 是否使用数学文本
    @pytest.mark.parametrize('use_math_text', useMathText_data)
    def test_useMathText(self, use_math_text):
        # 在上下文中设置 matplotlib 的 rc 参数，控制坐标轴格式化器是否使用数学文本
        with mpl.rc_context({'axes.formatter.use_mathtext': use_math_text}):
            # 创建 ScalarFormatter 对象
            tmp_form = mticker.ScalarFormatter()
            # 断言 ScalarFormatter 是否使用了指定的数学文本设置
            assert use_math_text == tmp_form.get_useMathText()

    # 测试设置 ScalarFormatter 的偏移量为浮点数时的情况
    def test_set_use_offset_float(self):
        # 创建 ScalarFormatter 对象
        tmp_form = mticker.ScalarFormatter()
        # 设置 ScalarFormatter 的偏移量为 0.5
        tmp_form.set_useOffset(0.5)
        # 断言 ScalarFormatter 是否未使用偏移量
        assert not tmp_form.get_useOffset()
        # 断言 ScalarFormatter 的偏移量是否为 0.5
        assert tmp_form.offset == 0.5

    # 测试使用区域设置进行数值格式化
    def test_use_locale(self):
        # 获取当前环境的区域设置信息
        conv = locale.localeconv()
        sep = conv['thousands_sep']
        # 如果千位分隔符不存在或分组信息的最后一位为默认值，则跳过测试
        if not sep or conv['grouping'][-1:] in ([], [locale.CHAR_MAX]):
            pytest.skip('Locale does not apply grouping')  # pragma: no cover

        # 在上下文中设置 matplotlib 的 rc 参数，启用区域设置
        with mpl.rc_context({'axes.formatter.use_locale': True}):
            # 创建 ScalarFormatter 对象
            tmp_form = mticker.ScalarFormatter()
            # 断言 ScalarFormatter 是否使用了区域设置
            assert tmp_form.get_useLocale()

            # 创建虚拟轴
            tmp_form.create_dummy_axis()
            # 设置虚拟轴的数据区间
            tmp_form.axis.set_data_interval(0, 10)
            # 设置虚拟轴的刻度位置
            tmp_form.set_locs([1, 2, 3])
            # 断言千位分隔符是否出现在指定数值的格式化结果中
            assert sep in tmp_form(1e9)

    # 使用参数来测试 ScalarFormatter 的科学记数法限制
    @pytest.mark.parametrize(
        'sci_type, scilimits, lim, orderOfMag, fewticks', scilimits_data)
    def test_scilimits(self, sci_type, scilimits, lim, orderOfMag, fewticks):
        # 创建 ScalarFormatter 对象
        tmp_form = mticker.ScalarFormatter()
        # 设置科学记数法类型
        tmp_form.set_scientific(sci_type)
        # 设置科学记数法的限制
        tmp_form.set_powerlimits(scilimits)
        # 创建图形和坐标轴
        fig, ax = plt.subplots()
        ax.yaxis.set_major_formatter(tmp_form)
        ax.set_ylim(*lim)
        # 如果需要少量刻度，则设置主要刻度定位器
        if fewticks:
            ax.yaxis.set_major_locator(mticker.MaxNLocator(4))

        # 获取当前坐标轴的主要刻度位置
        tmp_form.set_locs(ax.yaxis.get_majorticklocs())
        # 断言期望的数量级是否与 ScalarFormatter 的数量级匹配
        assert orderOfMag == tmp_form.orderOfMagnitude

    # 使用参数来测试 ScalarFormatter 的数据格式化
    @pytest.mark.parametrize('value, expected', format_data)
    def test_format_data(self, value, expected):
        # 禁用 matplotlib 的 unicode 减号显示设置
        mpl.rcParams['axes.unicode_minus'] = False
        # 创建 ScalarFormatter 对象
        sf = mticker.ScalarFormatter()
        # 断言指定值的格式化结果是否符合预期
        assert sf.format_data(value) == expected

    # 使用参数来测试坐标轴精度
    @pytest.mark.parametrize('data, expected', cursor_data)
    def test_cursor_precision(self, data, expected):
        # 创建图形和坐标轴
        fig, ax = plt.subplots()
        # 设置 x 轴的数据区间，设置指向精度为 0.001
        ax.set_xlim(-1, 1)
        # 获取 x 轴的主要刻度格式化器的数据短格式化方法
        fmt = ax.xaxis.get_major_formatter().format_data_short
        # 断言格式化给定数据后的结果是否符合预期
        assert fmt(data) == expected

    # 使用参数来测试虚拟轴的坐标精度
    @pytest.mark.parametrize('data, expected', cursor_data)
    def test_cursor_dummy_axis(self, data, expected):
        # 创建 ScalarFormatter 对象
        sf = mticker.ScalarFormatter()
        # 创建虚拟轴
        sf.create_dummy_axis()
        # 设置虚拟轴的视图区间
        sf.axis.set_view_interval(0, 10)
        # 获取格式化数据的短格式化方法
        fmt = sf.format_data_short
        # 断言格式化给定数据后的结果是否符合预期
        assert fmt(data) == expected
        # 断言虚拟轴的刻度空间是否为 9
        assert sf.axis.get_tick_space() == 9
        # 断言虚拟轴的最小位置是否为 0
        assert sf.axis.get_minpos() == 0
    # 测试 Matplotlib 的数学文本刻度功能
    def test_mathtext_ticks(self):
        # 更新全局参数，设置使用衬线字体 cmr10，禁用数学文本
        mpl.rcParams.update({
            'font.family': 'serif',
            'font.serif': 'cmr10',
            'axes.formatter.use_mathtext': False
        })
    
        # 根据 pytest 版本执行不同的测试分支
        if parse_version(pytest.__version__).major < 8:
            # 如果 pytest 版本小于 8，期望收到用户警告，匹配 'cmr10 font should ideally'
            with pytest.warns(UserWarning, match='cmr10 font should ideally'):
                # 创建图形和坐标轴对象
                fig, ax = plt.subplots()
                # 设置 x 轴刻度
                ax.set_xticks([-1, 0, 1])
                # 绘制图形
                fig.canvas.draw()
        else:
            # 如果 pytest 版本大于等于 8，期望收到两个用户警告，分别匹配 'Glyph 8722' 和 'cmr10 font should ideally'
            with (pytest.warns(UserWarning, match="Glyph 8722"),
                  pytest.warns(UserWarning, match='cmr10 font should ideally')):
                # 创建图形和坐标轴对象
                fig, ax = plt.subplots()
                # 设置 x 轴刻度
                ax.set_xticks([-1, 0, 1])
                # 绘制图形
                fig.canvas.draw()
    
    # 测试 cmr10 字体替换功能
    def test_cmr10_substitutions(self, caplog):
        # 更新全局参数，设置使用 cmr10 字体，数学文本使用 cm 字体集，启用数学文本
        mpl.rcParams.update({
            'font.family': 'cmr10',
            'mathtext.fontset': 'cm',
            'axes.formatter.use_mathtext': True,
        })
    
        # 测试不会记录缺失字形的警告
        with caplog.at_level(logging.WARNING, logger='matplotlib.mathtext'):
            # 创建图形和坐标轴对象
            fig, ax = plt.subplots()
            # 绘制折线图
            ax.plot([-0.03, 0.05], [40, 0.05])
            # 设置 y 轴为对数坐标
            ax.set_yscale('log')
            # 设置 y 轴刻度
            yticks = [0.02, 0.3, 4, 50]
            # 创建科学计数法格式化器
            formatter = mticker.LogFormatterSciNotation()
            # 设置 y 轴刻度及其格式化
            ax.set_yticks(yticks, map(formatter, yticks))
            # 绘制图形
            fig.canvas.draw()
            # 断言不会有日志记录
            assert not caplog.text
    
    # 测试 ScalarFormatter 的空刻度情况
    def test_empty_locs(self):
        # 创建 ScalarFormatter 对象
        sf = mticker.ScalarFormatter()
        # 设置刻度为空列表
        sf.set_locs([])
        # 断言在位置 0.5 处的刻度为空字符串
        assert sf(0.5) == ''
class TestLogFormatterExponent:
    param_data = [
        (True, 4, np.arange(-3, 4.0), np.arange(-3, 4.0),
         ['-3', '-2', '-1', '0', '1', '2', '3']),
        # With labelOnlyBase=False, non-integer powers should be nicely
        # formatted.
        (False, 10, np.array([0.1, 0.00001, np.pi, 0.2, -0.2, -0.00001]),
         range(6), ['0.1', '1e-05', '3.14', '0.2', '-0.2', '-1e-05']),
        (False, 50, np.array([3, 5, 12, 42], dtype=float), range(6),
         ['3', '5', '12', '42']),
    ]

    base_data = [2.0, 5.0, 10.0, np.pi, np.e]

    @pytest.mark.parametrize(
            'labelOnlyBase, exponent, locs, positions, expected', param_data)
    @pytest.mark.parametrize('base', base_data)
    def test_basic(self, labelOnlyBase, base, exponent, locs, positions,
                   expected):
        # 创建 LogFormatterExponent 对象，设置基数和是否仅标签基数
        formatter = mticker.LogFormatterExponent(base=base,
                                                 labelOnlyBase=labelOnlyBase)
        # 创建虚拟轴对象
        formatter.create_dummy_axis()
        # 设置轴视图区间
        formatter.axis.set_view_interval(1, base**exponent)
        # 计算值为基数的指数次方
        vals = base**locs
        # 为每个值生成格式化后的标签
        labels = [formatter(x, pos) for (x, pos) in zip(vals, positions)]
        # 将期望的标签中的减号替换为 Unicode 减号
        expected = [label.replace('-', '\N{Minus Sign}') for label in expected]
        # 断言生成的标签与期望的标签一致
        assert labels == expected

    def test_blank(self):
        # 当 labelOnlyBase=True 时，非整数幂的情况应返回空字符串
        formatter = mticker.LogFormatterExponent(base=10, labelOnlyBase=True)
        # 创建虚拟轴对象
        formatter.create_dummy_axis()
        # 设置轴视图区间
        formatter.axis.set_view_interval(1, 10)
        # 断言对于 10 的 0.1 次方应为空字符串
        assert formatter(10**0.1) == ''


class TestLogFormatterMathtext:
    fmt = mticker.LogFormatterMathtext()
    test_data = [
        (0, 1, '$\\mathdefault{10^{0}}$'),
        (0, 1e-2, '$\\mathdefault{10^{-2}}$'),
        (0, 1e2, '$\\mathdefault{10^{2}}$'),
        (3, 1, '$\\mathdefault{1}$'),
        (3, 1e-2, '$\\mathdefault{0.01}$'),
        (3, 1e2, '$\\mathdefault{100}$'),
        (3, 1e-3, '$\\mathdefault{10^{-3}}$'),
        (3, 1e3, '$\\mathdefault{10^{3}}$'),
    ]

    @pytest.mark.parametrize('min_exponent, value, expected', test_data)
    def test_min_exponent(self, min_exponent, value, expected):
        # 在给定的上下文中设置最小指数值
        with mpl.rc_context({'axes.formatter.min_exponent': min_exponent}):
            # 断言格式化后的值与期望值相等
            assert self.fmt(value) == expected


class TestLogFormatterSciNotation:
    # 定义测试数据，包含多个元组，每个元组包括 base、value 和 expected 三个元素
    test_data = [
        (2, 0.03125, '$\\mathdefault{2^{-5}}$'),
        (2, 1, '$\\mathdefault{2^{0}}$'),
        (2, 32, '$\\mathdefault{2^{5}}$'),
        (2, 0.0375, '$\\mathdefault{1.2\\times2^{-5}}$'),
        (2, 1.2, '$\\mathdefault{1.2\\times2^{0}}$'),
        (2, 38.4, '$\\mathdefault{1.2\\times2^{5}}$'),
        (10, -1, '$\\mathdefault{-10^{0}}$'),
        (10, 1e-05, '$\\mathdefault{10^{-5}}$'),
        (10, 1, '$\\mathdefault{10^{0}}$'),
        (10, 100000, '$\\mathdefault{10^{5}}$'),
        (10, 2e-05, '$\\mathdefault{2\\times10^{-5}}$'),
        (10, 2, '$\\mathdefault{2\\times10^{0}}$'),
        (10, 200000, '$\\mathdefault{2\\times10^{5}}$'),
        (10, 5e-05, '$\\mathdefault{5\\times10^{-5}}$'),
        (10, 5, '$\\mathdefault{5\\times10^{0}}$'),
        (10, 500000, '$\\mathdefault{5\\times10^{5}}$'),
    ]
    
    # 使用 matplotlib 的样式上下文 'default'，并对测试参数化
    @mpl.style.context('default')
    @pytest.mark.parametrize('base, value, expected', test_data)
    # 定义测试函数 test_basic，参数为 base、value、expected
    def test_basic(self, base, value, expected):
        # 创建一个科学记数法格式化对象，指定基数为 base
        formatter = mticker.LogFormatterSciNotation(base=base)
        # 使用 matplotlib 的上下文管理器设置 'text.usetex' 为 False
        with mpl.rc_context({'text.usetex': False}):
            # 断言 formatter 处理 value 后的结果与预期的 expected 相等
            assert formatter(value) == expected
class TestLogFormatter:
    # 测试 LogFormatter 类的功能

    @pytest.mark.parametrize('value, domain, expected', pprint_data)
    # 使用 pytest 的参数化测试来测试 pprint 方法
    def test_pprint(self, value, domain, expected):
        # 创建 LogFormatter 对象
        fmt = mticker.LogFormatter()
        # 调用 _pprint_val 方法进行格式化
        label = fmt._pprint_val(value, domain)
        # 断言格式化后的结果是否符合预期
        assert label == expected

    @pytest.mark.parametrize('value, long, short', [
        (0.0, "0", "0"),
        (0, "0", "0"),
        (-1.0, "-10^0", "-1"),
        (2e-10, "2x10^-10", "2e-10"),
        (1e10, "10^10", "1e+10"),
    ])
    # 使用 pytest 的参数化测试来测试 format_data 和 format_data_short 方法
    def test_format_data(self, value, long, short):
        # 创建一个图表对象
        fig, ax = plt.subplots()
        # 设置 x 轴的比例为对数尺度
        ax.set_xscale('log')
        # 获取当前轴的主要格式化器
        fmt = ax.xaxis.get_major_formatter()
        # 断言使用 format_data 方法格式化后的结果是否符合预期
        assert fmt.format_data(value) == long
        # 断言使用 format_data_short 方法格式化后的结果是否符合预期
        assert fmt.format_data_short(value) == short

    def _sub_labels(self, axis, subs=()):
        """Test whether locator marks subs to be labeled."""
        # 测试轴的次要格式化器是否标记次要刻度需要标记
        fmt = axis.get_minor_formatter()
        # 获取次要刻度的位置
        minor_tlocs = axis.get_minorticklocs()
        # 设置格式化器的位置
        fmt.set_locs(minor_tlocs)
        # 计算次要刻度位置的系数
        coefs = minor_tlocs / 10**(np.floor(np.log10(minor_tlocs)))
        # 预期的标签结果，检查是否需要标记
        label_expected = [round(c) in subs for c in coefs]
        # 测试实际的标签结果
        label_test = [fmt(x) != '' for x in minor_tlocs]
        # 断言实际的标签结果是否与预期一致
        assert label_test == label_expected

    @mpl.style.context('default')
    # 使用默认样式上下文来测试 sublabel 方法
    def test_sublabel(self):
        # 创建一个图表对象
        fig, ax = plt.subplots()
        # 设置 x 轴的比例为对数尺度
        ax.set_xscale('log')
        # 设置主要定位器为对数定位器，子定位器为空
        ax.xaxis.set_major_locator(mticker.LogLocator(base=10, subs=[]))
        # 设置次要定位器为对数定位器，子定位器为 2 到 9
        ax.xaxis.set_minor_locator(mticker.LogLocator(base=10,
                                                      subs=np.arange(2, 10)))
        # 设置主要格式化器为只显示基数的对数格式化器
        ax.xaxis.set_major_formatter(mticker.LogFormatter(labelOnlyBase=True))
        # 设置次要格式化器为显示所有内容的对数格式化器
        ax.xaxis.set_minor_formatter(mticker.LogFormatter(labelOnlyBase=False))
        # 设置 x 轴的范围在大于 3 个数量级时，只显示基数
        ax.set_xlim(1, 1e4)
        # 获取当前轴的主要格式化器
        fmt = ax.xaxis.get_major_formatter()
        # 设置格式化器的位置
        fmt.set_locs(ax.xaxis.get_majorticklocs())
        # 检查是否所有的主要刻度都显示标签
        show_major_labels = [fmt(x) != ''
                             for x in ax.xaxis.get_majorticklocs()]
        # 断言所有的主要刻度是否都显示标签
        assert np.all(show_major_labels)
        # 测试次要标签的方法
        self._sub_labels(ax.xaxis, subs=[])

        # 对于接下来的两个情况，如果 LogFormatter.set_locs 中的 numdec 阈值为 3，
        # 那么对于 2-3 个数量级，标签将为 3，而对于 1-2 个数量级，标签将为 (2, 5)。
        # 当阈值为 1 时，不会标记子刻度。
        
        # 设置 x 轴的范围在 2 到 3 个数量级
        ax.set_xlim(1, 800)
        # 测试次要标签的方法
        self._sub_labels(ax.xaxis, subs=[])

        # 设置 x 轴的范围在 1 到 2 个数量级
        ax.set_xlim(1, 80)
        # 测试次要标签的方法
        self._sub_labels(ax.xaxis, subs=[])

        # 设置 x 轴的范围在 0.4 到 1 个数量级，标签子刻度为 2, 3, 4, 6
        ax.set_xlim(1, 8)
        # 测试次要标签的方法
        self._sub_labels(ax.xaxis, subs=[2, 3, 4, 6])

        # 设置 x 轴的范围在 0 到 0.4 个数量级，标签所有子刻度
        ax.set_xlim(0.5, 0.9)
        # 测试次要标签的方法
        self._sub_labels(ax.xaxis, subs=np.arange(2, 10, dtype=int))

    @pytest.mark.parametrize('val', [1, 10, 100, 1000])
    # 使用 pytest 的参数化测试来测试 val 参数
    def test_LogFormatter_call(self, val):
        # 测试 __call__ 方法中使用的 _num_to_string 方法
        # 创建一个临时的 LogFormatter 实例
        temp_lf = mticker.LogFormatter()
        # 创建一个虚拟的坐标轴
        temp_lf.create_dummy_axis()
        # 设置坐标轴的视图区间
        temp_lf.axis.set_view_interval(1, 10)
        # 断言调用 LogFormatter 实例时返回字符串形式的输入值
        assert temp_lf(val) == str(val)

    @pytest.mark.parametrize('val', [1e-323, 2e-323, 10e-323, 11e-323])
    def test_Log
    # 定义一个测试日志格式化类 TestLogitFormatter
    class TestLogitFormatter:
        
        # 静态方法：用于解析字符串，将其转换为浮点数表示，支持科学计数法和分数形式
        @staticmethod
        def logit_deformatter(string):
            r"""
            Parser to convert string as r'$\mathdefault{1.41\cdot10^{-4}}$' in
            float 1.41e-4, as '0.5' or as r'$\mathdefault{\frac{1}{2}}$' in float
            0.5,
            """
            # 匹配科学计数法格式的字符串
            match = re.match(
                r"[^\d]*"
                r"(?P<comp>1-)?"
                r"(?P<mant>\d*\.?\d*)?"
                r"(?:\\cdot)?"
                r"(?:10\^\{(?P<expo>-?\d*)})?"
                r"[^\d]*$",
                string,
            )
            if match:
                # 判断是否存在负号
                comp = match["comp"] is not None
                # 提取幂次的系数，若未指定默认为 0
                mantissa = float(match["mant"]) if match["mant"] else 1
                expo = int(match["expo"]) if match["expo"] is not None else 0
                # 计算最终的数值表示
                value = mantissa * 10 ** expo
                if match["mant"] or match["expo"] is not None:
                    if comp:
                        return 1 - value
                    return value
            # 匹配分数形式的字符串
            match = re.match(
                r"[^\d]*\\frac\{(?P<num>\d+)\}\{(?P<deno>\d+)\}[^\d]*$", string
            )
            if match:
                # 将分子和分母转换为浮点数，并返回它们的比值
                num, deno = float(match["num"]), float(match["deno"])
                return num / deno
            # 若未匹配到任何格式，抛出数值错误异常
            raise ValueError("Not formatted by LogitFormatter")

        # 参数化测试：测试 logit_deformatter 方法的不同输入和预期输出
        @pytest.mark.parametrize(
            "fx, x",
            [
                (r"STUFF0.41OTHERSTUFF", 0.41),
                (r"STUFF1.41\cdot10^{-2}OTHERSTUFF", 1.41e-2),
                (r"STUFF1-0.41OTHERSTUFF", 1 - 0.41),
                (r"STUFF1-1.41\cdot10^{-2}OTHERSTUFF", 1 - 1.41e-2),
                (r"STUFF", None),
                (r"STUFF12.4e-3OTHERSTUFF", None),
            ],
        )
        # 测试 logit_deformatter 方法处理不同输入的行为
        def test_logit_deformater(self, fx, x):
            if x is None:
                # 当期望值为 None 时，断言会引发 ValueError 异常
                with pytest.raises(ValueError):
                    TestLogitFormatter.logit_deformatter(fx)
            else:
                # 当期望值不为 None 时，断言方法返回值与期望值相近
                y = TestLogitFormatter.logit_deformatter(fx)
                assert _LogitHelper.isclose(x, y)

        # 定义一个列表，包含一系列用于 logit 空间理想刻度的数值
        decade_test = sorted(
            [10 ** (-i) for i in range(1, 10)]
            + [1 - 10 ** (-i) for i in range(1, 10)]
            + [1 / 2]
        )

        # 参数化测试：测试 logit 空间中基本值对应于理想刻度的值
        @pytest.mark.parametrize("x", decade_test)
        def test_basic(self, x):
            """
            Test the formatted value correspond to the value for ideal ticks in
            logit space.
            """
            # 创建一个 LogitFormatter 实例
            formatter = mticker.LogitFormatter(use_overline=False)
            # 设置 LogitFormatter 的刻度位置
            formatter.set_locs(self.decade_test)
            # 对输入值进行格式化
            s = formatter(x)
            # 将格式化后的字符串转换回数值
            x2 = TestLogitFormatter.logit_deformatter(s)
            # 断言格式化后的数值与原始输入值在允许误差范围内相等
            assert _LogitHelper.isclose(x, x2)

        # 参数化测试：测试 LogitFormatter 处理无效值的行为
        @pytest.mark.parametrize("x", (-1, -0.5, -0.1, 1.1, 1.5, 2))
        def test_invalid(self, x):
            """
            Test that invalid value are formatted with empty string without
            raising exception.
            """
            # 创建一个 LogitFormatter 实例
            formatter = mticker.LogitFormatter(use_overline=False)
            # 设置 LogitFormatter 的刻度位置
            formatter.set_locs(self.decade_test)
            # 对输入值进行格式化
            s = formatter(x)
            # 断言格式化后的字符串为空字符串
            assert s == ""

        # 参数化测试：测试 LogitFormatter 处理 sigmoid 函数结果的行为
        @pytest.mark.parametrize("x", 1 / (1 + np.exp(-np.linspace(-7, 7, 10))))
    @pytest.mark.parametrize("method, lims, cases", lims_minor_major)
    def test_minor_vs_major(self, method, lims, cases):
        """
        使用 pytest 的参数化装饰器，传入 method, lims, cases 作为参数进行测试。
        测试 minor vs major 的显示效果。
        """

        # 根据 method 的值选择使用 LogitLocator(minor=True) 或直接使用给定的 ticks
        if method:
            min_loc = mticker.LogitLocator(minor=True)
            ticks = min_loc.tick_values(*lims)
        else:
            ticks = np.array(lims)

        # 创建 LogitFormatter 对象，并设置为使用 minor ticks
        min_form = mticker.LogitFormatter(minor=True)

        # 遍历 cases 中的阈值和是否包含 minor tick 的情况
        for threshold, has_minor in cases:
            # 设置 LogitFormatter 的 minor threshold
            min_form.set_minor_threshold(threshold)

            # 格式化 ticks
            formatted = min_form.format_ticks(ticks)

            # 筛选出有标签的部分
            labelled = [f for f in formatted if len(f) > 0]

            # 断言：如果 has_minor 为 True，则应该有标签；否则应该没有标签
            if has_minor:
                assert len(labelled) > 0, (threshold, has_minor)
            else:
                assert len(labelled) == 0, (threshold, has_minor)

    def test_minor_number(self):
        """
        测试参数 minor_number 的影响。
        """
        # 创建 LogitLocator 和 LogitFormatter 对象，并设置为使用 minor ticks
        min_loc = mticker.LogitLocator(minor=True)
        min_form = mticker.LogitFormatter(minor=True)

        # 获取指定范围内的 ticks
        ticks = min_loc.tick_values(5e-2, 1 - 5e-2)

        # 遍历不同的 minor_number 值
        for minor_number in (2, 4, 8, 16):
            # 设置 LogitFormatter 的 minor number
            min_form.set_minor_number(minor_number)

            # 格式化 ticks
            formatted = min_form.format_ticks(ticks)

            # 筛选出有标签的部分
            labelled = [f for f in formatted if len(f) > 0]

            # 断言：标签的数量应该等于 minor_number
            assert len(labelled) == minor_number

    def test_use_overline(self):
        """
        测试参数 use_overline 的影响。
        """
        # 设置初始值和预期结果
        x = 1 - 1e-2
        fx1 = r"$\mathdefault{1-10^{-2}}$"
        fx2 = r"$\mathdefault{\overline{10^{-2}}}$"

        # 创建 LogitFormatter 对象，初始不使用 overline
        form = mticker.LogitFormatter(use_overline=False)

        # 断言：初始情况下应该与 fx1 一致
        assert form(x) == fx1

        # 使用 overline，并断言结果应与 fx2 一致
        form.use_overline(True)
        assert form(x) == fx2

        # 关闭 overline，并断言结果应与 fx1 一致
        form.use_overline(False)
        assert form(x) == fx1

    def test_one_half(self):
        """
        测试参数 one_half 的影响。
        """
        # 创建 LogitFormatter 对象
        form = mticker.LogitFormatter()

        # 断言：在默认情况下应该包含 LaTeX 分数表达式
        assert r"\frac{1}{2}" in form(1/2)

        # 设置为 "1/2"，并断言结果中应包含 "1/2"
        form.set_one_half("1/2")
        assert "1/2" in form(1/2)

        # 设置为 "one half"，并断言结果中应包含 "one half"
        form.set_one_half("one half")
        assert "one half" in form(1/2)
    # 使用 pytest 的参数化装饰器，为测试方法 test_format_data_short 提供多组参数 N = 100, 253, 754
    @pytest.mark.parametrize("N", (100, 253, 754))
    # 定义测试方法 test_format_data_short，接受参数 N
    def test_format_data_short(self, N):
        # 生成一个包含 N 个均匀分布在 [0, 1] 间的数列，并去除首尾两个数
        locs = np.linspace(0, 1, N)[1:-1]
        # 创建 LogitFormatter 对象，用于格式化数据
        form = mticker.LogitFormatter()
        # 遍历 locs 中的每个数 x
        for x in locs:
            # 对当前数 x 进行格式化处理，得到 fx
            fx = form.format_data_short(x)
            # 如果 fx 以 "1-" 开头
            if fx.startswith("1-"):
                # 则将 fx 中第三个字符到末尾的部分转换为浮点数，并将其取 1 减
                x2 = 1 - float(fx[2:])
            else:
                # 否则直接将 fx 转换为浮点数
                x2 = float(fx)
            # 断言当前处理后的数 x2 与原始数 x 的差值绝对值小于 1/N
            assert abs(x - x2) < 1 / N
class TestFormatStrFormatter:
    def test_basic(self):
        # 创建一个以 % 格式进行格式化的格式化器对象
        tmp_form = mticker.FormatStrFormatter('%05d')
        # 断言格式化后的结果为 '00002'
        assert '00002' == tmp_form(2)


class TestStrMethodFormatter:
    test_data = [
        ('{x:05d}', (2,), False, '00002'),
        ('{x:05d}', (2,), True, '00002'),
        ('{x:05d}', (-2,), False, '-0002'),
        ('{x:05d}', (-2,), True, '\N{MINUS SIGN}0002'),
        ('{x:03d}-{pos:02d}', (2, 1), False, '002-01'),
        ('{x:03d}-{pos:02d}', (2, 1), True, '002-01'),
        ('{x:03d}-{pos:02d}', (-2, 1), False, '-02-01'),
        ('{x:03d}-{pos:02d}', (-2, 1), True, '\N{MINUS SIGN}02-01'),
    ]

    @pytest.mark.parametrize('format, input, unicode_minus, expected', test_data)
    def test_basic(self, format, input, unicode_minus, expected):
        # 在上下文中设置 rc 参数 'axes.unicode_minus' 的布尔值
        with mpl.rc_context({"axes.unicode_minus": unicode_minus}):
            # 创建一个基于字符串方法的格式化器对象
            fmt = mticker.StrMethodFormatter(format)
            # 断言格式化后的结果与预期结果相符
            assert fmt(*input) == expected


class TestEngFormatter:
    # (unicode_minus, input, expected) where ''expected'' corresponds to the
    # outputs respectively returned when (places=None, places=0, places=2)
    # unicode_minus is a boolean value for the rcParam['axes.unicode_minus']
    # 定义一个原始格式数据的列表，包含了各种测试用例，每个元素是一个元组，元组包含了三个值：布尔值unicode_minus，浮点数input，元组expected
    raw_format_data = [
        (False, -1234.56789, ('-1.23457 k', '-1 k', '-1.23 k')),  # 测试负数，包括几种格式化输出结果
        (True, -1234.56789, ('\N{MINUS SIGN}1.23457 k', '\N{MINUS SIGN}1 k', '\N{MINUS SIGN}1.23 k')),  # 测试负数，包括带Unicode负号的几种格式化输出结果
        (False, -1.23456789, ('-1.23457', '-1', '-1.23')),  # 测试负小数，包括几种格式化输出结果
        (True, -1.23456789, ('\N{MINUS SIGN}1.23457', '\N{MINUS SIGN}1', '\N{MINUS SIGN}1.23')),  # 测试负小数，包括带Unicode负号的几种格式化输出结果
        (False, -0.123456789, ('-123.457 m', '-123 m', '-123.46 m')),  # 测试负小数，包括几种格式化输出结果
        (True, -0.123456789, ('\N{MINUS SIGN}123.457 m', '\N{MINUS SIGN}123 m', '\N{MINUS SIGN}123.46 m')),  # 测试负小数，包括带Unicode负号的几种格式化输出结果
        (False, -0.00123456789, ('-1.23457 m', '-1 m', '-1.23 m')),  # 测试负小数，包括几种格式化输出结果
        (True, -0.00123456789, ('\N{MINUS SIGN}1.23457 m', '\N{MINUS SIGN}1 m', '\N{MINUS SIGN}1.23 m')),  # 测试负小数，包括带Unicode负号的几种格式化输出结果
        (True, -0.0, ('0', '0', '0.00')),  # 测试零，包括几种格式化输出结果
        (True, -0, ('0', '0', '0.00')),  # 测试零，包括几种格式化输出结果
        (True, 0, ('0', '0', '0.00')),  # 测试零，包括几种格式化输出结果
        (True, 1.23456789e-6, ('1.23457 µ', '1 µ', '1.23 µ')),  # 测试小正数，包括几种格式化输出结果
        (True, 0.123456789, ('123.457 m', '123 m', '123.46 m')),  # 测试正小数，包括几种格式化输出结果
        (True, 0.1, ('100 m', '100 m', '100.00 m')),  # 测试正小数，包括几种格式化输出结果
        (True, 1, ('1', '1', '1.00')),  # 测试整数，包括几种格式化输出结果
        (True, 1.23456789, ('1.23457', '1', '1.23')),  # 测试正小数，包括几种格式化输出结果
        (True, 999.9, ('999.9', '1 k', '999.90')),  # 测试接近1000的数，包括几种格式化输出结果
        (True, 999.9999, ('1 k', '1 k', '1.00 k')),  # 测试接近1000的数，包括几种格式化输出结果
        (False, -999.9999, ('-1 k', '-1 k', '-1.00 k')),  # 测试负接近1000的数，包括几种格式化输出结果
        (True, -999.9999, ('\N{MINUS SIGN}1 k', '\N{MINUS SIGN}1 k', '\N{MINUS SIGN}1.00 k')),  # 测试负接近1000的数，包括带Unicode负号的几种格式化输出结果
        (True, 1000, ('1 k', '1 k', '1.00 k')),  # 测试1000，包括几种格式化输出结果
        (True, 1001, ('1.001 k', '1 k', '1.00 k')),  # 测试大于1000的数，包括几种格式化输出结果
        (True, 100001, ('100.001 k', '100 k', '100.00 k')),  # 测试大于1000的数，包括几种格式化输出结果
        (True, 987654.321, ('987.654 k', '988 k', '987.65 k')),  # 测试接近1000k的数，包括几种格式化输出结果
        (True, 1.23e33, ('1230 Q', '1230 Q', '1230.00 Q'))  # 测试超过1000 Q的数，包括几种格式化输出结果
    ]
    
    # 使用pytest的参数化测试装饰器标记，将raw_format_data作为参数传入测试函数
    @pytest.mark.parametrize('unicode_minus, input, expected', raw_format_data)
def test_engformatter_usetex_useMathText():
    # 创建一个图形和轴对象
    fig, ax = plt.subplots()
    # 绘制一条直线
    ax.plot([0, 500, 1000], [0, 500, 1000])
    # 设置 x 轴的刻度位置
    ax.set_xticks([0, 500, 1000])
    
    # 遍历两种 EngFormatter 格式化器，一个使用 LaTeX，一个使用 MathText
    for formatter in (mticker.EngFormatter(usetex=True),
                      mticker.EngFormatter(useMathText=True)):
        # 设置 x 轴主刻度的格式化器
        ax.xaxis.set_major_formatter(formatter)
        # 绘制图形
        fig.canvas.draw()
        # 获取 x 轴刻度标签的文本内容
        x_tick_label_text = [labl.get_text() for labl in ax.get_xticklabels()]
        # 检查是否在刻度标签中的数字周围插入了美元符号 `$`
        # 这个断言用于确认标签是否符合预期格式
        assert x_tick_label_text == ['$0$', '$500$', '$1$ k']


class TestPercentFormatter:
    # 百分比数据列表，包含不同设置的测试用例
    percent_data = [
        (100, 0, '%', 120, 100, '120%'),
        (100, 0, '%', 100, 90, '100%'),
        (100, 0, '%', 90, 50, '90%'),
        (100, 0, '%', -1.7, 40, '-2%'),
        (100, 1, '%', 90.0, 100, '90.0%'),
        (100, 1, '%', 80.1, 90, '80.1%'),
        (100, 1, '%', 70.23, 50, '70.2%'),
        (100, 1, '%', -60.554, 40, '-60.6%'),
        (100, None, '%', 95, 1, '95.00%'),
        (1.0, None, '%', 3, 6, '300%'),
        (17.0, None, '%', 1, 8.5, '6%'),
        (17.0, None, '%', 1, 8.4, '5.9%'),
        (5, None, '%', -100, 0.000001, '-2000.00000%'),
        (1.0, 2, None, 1.2, 100, '120.00'),
        (75, 3, '', 50, 100, '66.667'),
        (42, None, '^^Foobar$$', 21, 12, '50.0^^Foobar$$'),
    ]

    # 百分比数据的标识列表，用于参数化测试
    percent_ids = [
        'decimals=0, x>100%',
        'decimals=0, x=100%',
        'decimals=0, x<100%',
        'decimals=0, x<0%',
        'decimals=1, x>100%',
        'decimals=1, x=100%',
        'decimals=1, x<100%',
        'decimals=1, x<0%',
        'autodecimal, x<100%, display_range=1',
        'autodecimal, x>100%, display_range=6 (custom xmax test)',
        'autodecimal, x<100%, display_range=8.5 (autodecimal test 1)',
        'autodecimal, x<100%, display_range=8.4 (autodecimal test 2)',
        'autodecimal, x<-100%, display_range=1e-6 (tiny display range)',
        'None as percent symbol',
        'Empty percent symbol',
        'Custom percent symbol',
    ]

    # LaTeX 数据列表，用于测试不同设置下的 LaTeX 表达式
    latex_data = [
        (False, False, r'50\{t}%'),
        (False, True, r'50\\\{t\}\%'),
        (True, False, r'50\{t}%'),
        (True, True, r'50\{t}%'),
    ]

    @pytest.mark.parametrize(
            'xmax, decimals, symbol, x, display_range, expected',
            percent_data, ids=percent_ids)
    # 定义一个测试方法，用于测试基本功能
    def test_basic(self, xmax, decimals, symbol,
                   x, display_range, expected):
        # 创建一个 PercentFormatter 对象，用于格式化百分比显示
        formatter = mticker.PercentFormatter(xmax, decimals, symbol)
        # 使用临时的 matplotlib 配置上下文，确保在非 LaTeX 模式下进行测试
        with mpl.rc_context(rc={'text.usetex': False}):
            # 断言调用 PercentFormatter 对象的 format_pct 方法后的输出是否等于预期值
            assert formatter.format_pct(x, display_range) == expected

    # 使用参数化测试，参数为 'is_latex', 'usetex', 'expected'，从 latex_data 中获取
    @pytest.mark.parametrize('is_latex, usetex, expected', latex_data)
    # 定义一个测试方法，用于测试 LaTeX 模式下的百分比格式化
    def test_latex(self, is_latex, usetex, expected):
        # 创建一个 PercentFormatter 对象，使用 LaTeX 格式化符号，并设置是否为 LaTeX 模式
        fmt = mticker.PercentFormatter(symbol='\\{t}%', is_latex=is_latex)
        # 使用临时的 matplotlib 配置上下文，确保在指定的 LaTeX 模式下进行测试
        with mpl.rc_context(rc={'text.usetex': usetex}):
            # 断言调用 PercentFormatter 对象的 format_pct 方法后的输出是否等于预期值
            assert fmt.format_pct(50, 100) == expected
def _impl_locale_comma():
    try:
        # 尝试设置本地化环境为德语（德国）UTF-8
        locale.setlocale(locale.LC_ALL, 'de_DE.UTF-8')
    except locale.Error:
        # 如果设置失败，输出跳过信息并返回
        print('SKIP: Locale de_DE.UTF-8 is not supported on this machine')
        return
    
    # 创建一个使用数学文本和本地化的标量格式化器
    ticks = mticker.ScalarFormatter(useMathText=True, useLocale=True)
    # 定义数学文本格式
    fmt = '$\\mathdefault{%1.1f}$'
    # 使用格式化函数 `_format_maybe_minus_and_locale` 处理格式化，并验证结果
    x = ticks._format_maybe_minus_and_locale(fmt, 0.5)
    assert x == '$\\mathdefault{0{,}5}$'
    
    # 确保格式字符串中的逗号不被改变
    fmt = ',$\\mathdefault{,%1.1f},$'
    # 再次使用格式化函数 `_format_maybe_minus_and_locale` 处理格式化，并验证结果
    x = ticks._format_maybe_minus_and_locale(fmt, 0.5)
    assert x == ',$\\mathdefault{,0{,}5},$'
    
    # 创建一个不使用数学文本但使用本地化的标量格式化器
    ticks = mticker.ScalarFormatter(useMathText=False, useLocale=True)
    # 定义格式字符串
    fmt = '%1.1f'
    # 使用格式化函数 `_format_maybe_minus_and_locale` 处理格式化，并验证结果
    x = ticks._format_maybe_minus_and_locale(fmt, 0.5)
    assert x == '0,5'


def test_locale_comma():
    # 在某些系统或 pytest 版本中，异常处理器中的 `pytest.skip` 不会跳过测试，而是被当作异常处理，
    # 因此直接运行该测试可能会错误地失败而非跳过。
    # 为避免此问题，使用子进程运行测试，可以避免这个问题，并且无需修复本地化设置。
    proc = mpl.testing.subprocess_run_helper(_impl_locale_comma, timeout=60,
                                             extra_env={'MPLBACKEND': 'Agg'})
    # 检查是否有跳过信息，如果有则跳过测试，并将跳过信息打印出来
    skip_msg = next((line[len('SKIP:'):].strip()
                     for line in proc.stdout.splitlines()
                     if line.startswith('SKIP:')),
                    '')
    if skip_msg:
        pytest.skip(skip_msg)


def test_majformatter_type():
    fig, ax = plt.subplots()
    with pytest.raises(TypeError):
        # 测试主要刻度格式化器的类型是否正确
        ax.xaxis.set_major_formatter(mticker.LogLocator())


def test_minformatter_type():
    fig, ax = plt.subplots()
    with pytest.raises(TypeError):
        # 测试次要刻度格式化器的类型是否正确
        ax.xaxis.set_minor_formatter(mticker.LogLocator())


def test_majlocator_type():
    fig, ax = plt.subplots()
    with pytest.raises(TypeError):
        # 测试主要刻度定位器的类型是否正确
        ax.xaxis.set_major_locator(mticker.LogFormatter())


def test_minlocator_type():
    fig, ax = plt.subplots()
    with pytest.raises(TypeError):
        # 测试次要刻度定位器的类型是否正确
        ax.xaxis.set_minor_locator(mticker.LogFormatter())


def test_minorticks_rc():
    fig = plt.figure()

    def minorticksubplot(xminor, yminor, i):
        # 定义一组 rc 参数，控制是否显示次要刻度
        rc = {'xtick.minor.visible': xminor,
              'ytick.minor.visible': yminor}
        with plt.rc_context(rc=rc):
            ax = fig.add_subplot(2, 2, i)

        # 验证是否根据 rc 参数正确显示次要刻度
        assert (len(ax.xaxis.get_minor_ticks()) > 0) == xminor
        assert (len(ax.yaxis.get_minor_ticks()) > 0) == yminor

    # 执行四个子图的测试，分别测试不同的次要刻度显示情况
    minorticksubplot(False, False, 1)
    minorticksubplot(True, False, 2)
    minorticksubplot(False, True, 3)
    minorticksubplot(True, True, 4)


def test_minorticks_toggle():
    """
    Test toggling minor ticks

    Test `.Axis.minorticks_on()` and `.Axis.minorticks_off()`. Testing is
    limited to a subset of built-in scales - `'linear'`, `'log'`, `'asinh'`
    and `'logit'`. `symlog` scale does not seem to have a working minor
    """
    # 创建一个新的图形对象
    fig = plt.figure()
    
    # 定义一个函数，用于在图形上设置不同的刻度类型和次刻度显示方式
    def minortickstoggle(xminor, yminor, scale, i):
        # 添加一个子图到指定位置
        ax = fig.add_subplot(2, 2, i)
        # 设置 x 轴和 y 轴的刻度类型
        ax.set_xscale(scale)
        ax.set_yscale(scale)
        
        # 根据参数设置是否显示次刻度
        if not xminor and not yminor:
            ax.minorticks_off()
        if xminor and not yminor:
            ax.xaxis.minorticks_on()
            ax.yaxis.minorticks_off()
        if not xminor and yminor:
            ax.xaxis.minorticks_off()
            ax.yaxis.minorticks_on()
        if xminor and yminor:
            ax.minorticks_on()
    
        # 断言检查是否正确设置了次刻度
        assert (len(ax.xaxis.get_minor_ticks()) > 0) == xminor
        assert (len(ax.yaxis.get_minor_ticks()) > 0) == yminor
    
    # 定义不同的刻度类型列表
    scales = ['linear', 'log', 'asinh', 'logit']
    
    # 对每种刻度类型调用 minortickstoggle 函数，并清除图形对象
    for scale in scales:
        minortickstoggle(False, False, scale, 1)
        minortickstoggle(True, False, scale, 2)
        minortickstoggle(False, True, scale, 3)
        minortickstoggle(True, True, scale, 4)
        fig.clear()
    
    # 关闭图形对象，释放资源
    plt.close(fig)
@pytest.mark.parametrize('remove_overlapping_locs, expected_num',
                         ((True, 6),    # Test case with remove_overlapping_locs=True, expects 6 ticks
                          (None, 6),    # Test case with remove_overlapping_locs=None (default), expects 6 ticks
                          (False, 9)))  # Test case with remove_overlapping_locs=False, expects 9 ticks
def test_remove_overlap(remove_overlapping_locs, expected_num):
    # Generate a datetime array from "2018-11-03" to "2018-11-06"
    t = np.arange("2018-11-03", "2018-11-06", dtype="datetime64")
    # Create an array of ones with the same length as t
    x = np.ones(len(t))

    # Create a new figure and axis
    fig, ax = plt.subplots()
    # Plot x against t on the axis ax
    ax.plot(t, x)

    # Set major tick locators and formatters for the x-axis
    ax.xaxis.set_major_locator(mpl.dates.DayLocator())
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('\n%a'))

    # Set minor tick locators and formatters for the x-axis
    ax.xaxis.set_minor_locator(mpl.dates.HourLocator((0, 6, 12, 18)))
    ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%H:%M'))

    # Force there to be extra ticks
    ax.xaxis.get_minor_ticks(15)

    # Set/remove overlapping locations based on the parameter
    if remove_overlapping_locs is not None:
        ax.xaxis.remove_overlapping_locs = remove_overlapping_locs

    # Check getter/setter functionality for remove_overlapping_locs
    current = ax.xaxis.remove_overlapping_locs
    assert (current == ax.xaxis.get_remove_overlapping_locs())
    plt.setp(ax.xaxis, remove_overlapping_locs=current)
    new = ax.xaxis.remove_overlapping_locs
    assert (new == ax.xaxis.remove_overlapping_locs)

    # Check that the accessors filter correctly
    assert len(ax.xaxis.get_minorticklocs()) == expected_num  # Check minor tick locations
    assert len(ax.xaxis.get_minor_ticks()) == expected_num     # Check number of minor ticks
    assert len(ax.xaxis.get_minorticklabels()) == expected_num # Check minor tick labels
    assert len(ax.xaxis.get_minorticklines()) == expected_num*2 # Check minor tick lines


@pytest.mark.parametrize('sub', [
    ['hi', 'aardvark'],     # Test case with a list of strings for subs, expects ValueError
    np.zeros((2, 2))])      # Test case with a 2x2 array of zeros for subs, expects ValueError
def test_bad_locator_subs(sub):
    # Create a LogLocator instance
    ll = mticker.LogLocator()
    # Ensure ValueError is raised when setting parameters with invalid subs
    with pytest.raises(ValueError):
        ll.set_params(subs=sub)


@pytest.mark.parametrize('numticks', [1, 2, 3, 9])
@mpl.style.context('default')
def test_small_range_loglocator(numticks):
    # Create a LogLocator instance
    ll = mticker.LogLocator()
    # Set the number of ticks for LogLocator
    ll.set_params(numticks=numticks)
    # Iterate over different top values to test tick generation
    for top in [5, 7, 9, 11, 15, 50, 100, 1000]:
        # Assert that the difference between consecutive log10 tick values is 1
        assert (np.diff(np.log10(ll.tick_values(6, 150))) == 1).all()


def test_NullFormatter():
    # Create a NullFormatter instance
    formatter = mticker.NullFormatter()
    # Assert that formatting and format_data methods return an empty string
    assert formatter(1.0) == ''
    assert formatter.format_data(1.0) == ''
    assert formatter.format_data_short(1.0) == ''


@pytest.mark.parametrize('formatter', (
    mticker.FuncFormatter(lambda a: f'val: {a}'),    # Test FuncFormatter with a lambda function
    mticker.FixedFormatter(('foo', 'bar'))))         # Test FixedFormatter with a tuple of strings
def test_set_offset_string(formatter):
    # Assert that initial offset string is empty
    assert formatter.get_offset() == ''
    # Set a new offset string 'mpl'
    formatter.set_offset_string('mpl')
    # Assert that the offset string is now 'mpl'
    assert formatter.get_offset() == 'mpl'


def test_minorticks_on_multi_fig():
    """
    Turning on minor gridlines in a multi-Axes Figure
    that contains more than one boxplot and shares the x-axis
    should not raise an exception.
    """
    # Create a new figure and axis
    fig, ax = plt.subplots()

    # Plot three boxplots on the axis ax
    ax.boxplot(np.arange(10), positions=[0])
    ax.boxplot(np.arange(10), positions=[0])
    ax.boxplot(np.arange(10), positions=[1])

    # Turn on major gridlines
    ax.grid(which="major")
    # 在坐标轴上显示次要网格线
    ax.grid(which="minor")
    # 开启次要刻度
    ax.minorticks_on()
    # 在不进行渲染的情况下绘制图形
    fig.draw_without_rendering()

    # 断言：检查是否存在次要 X 轴网格线
    assert ax.get_xgridlines()
    # 断言：检查 X 轴次要刻度定位器是否为自动次要定位器
    assert isinstance(ax.xaxis.get_minor_locator(), mpl.ticker.AutoMinorLocator)
```