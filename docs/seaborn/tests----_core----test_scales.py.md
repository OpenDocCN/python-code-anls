# `D:\src\scipysrc\seaborn\tests\_core\test_scales.py`

```
import re  # 导入正则表达式模块

import numpy as np  # 导入NumPy库，并将其命名为np
import pandas as pd  # 导入Pandas库，并将其命名为pd
import matplotlib as mpl  # 导入Matplotlib库，并将其命名为mpl

import pytest  # 导入pytest测试框架
from numpy.testing import assert_array_equal  # 导入NumPy测试工具中的数组相等断言函数
from pandas.testing import assert_series_equal  # 导入Pandas测试工具中的Series相等断言函数

from seaborn._core.plot import Plot  # 从seaborn中导入Plot类
from seaborn._core.scales import (  # 从seaborn中导入多个尺度相关的类
    Nominal,  # 名义尺度
    Continuous,  # 连续尺度
    Boolean,  # 布尔尺度
    Temporal,  # 时间尺度
    PseudoAxis,  # 伪坐标轴
)
from seaborn._core.properties import (  # 从seaborn中导入多个属性相关的类
    IntervalProperty,  # 区间属性
    ObjectProperty,  # 对象属性
    Coordinate,  # 坐标属性
    Alpha,  # 透明度属性
    Color,  # 颜色属性
    Fill,  # 填充属性
)
from seaborn.palettes import color_palette  # 从seaborn中导入颜色调色板函数
from seaborn.utils import _version_predates  # 从seaborn中导入版本比较函数


class TestContinuous:  # 定义测试类TestContinuous

    @pytest.fixture  # 定义pytest的装置（fixture），用于提供测试数据
    def x(self):
        return pd.Series([1, 3, 9], name="x", dtype=float)  # 返回一个包含浮点数的Pandas Series

    def setup_ticks(self, x, *args, **kwargs):  # 定义设置ticks的方法

        s = Continuous().tick(*args, **kwargs)._setup(x, Coordinate())  # 创建连续尺度对象，并设置ticks
        a = PseudoAxis(s._matplotlib_scale)  # 创建伪坐标轴对象，使用Matplotlib的尺度
        a.set_view_interval(0, 1)  # 设置视图间隔为[0, 1]
        return a  # 返回设置好的伪坐标轴对象

    def setup_labels(self, x, *args, **kwargs):  # 定义设置标签的方法

        s = Continuous().label(*args, **kwargs)._setup(x, Coordinate())  # 创建连续尺度对象，并设置标签
        a = PseudoAxis(s._matplotlib_scale)  # 创建伪坐标轴对象，使用Matplotlib的尺度
        a.set_view_interval(0, 1)  # 设置视图间隔为[0, 1]
        locs = a.major.locator()  # 获取主要刻度定位器
        return a, locs  # 返回设置好的伪坐标轴对象和刻度定位器对象

    def test_coordinate_defaults(self, x):  # 定义测试默认坐标方法

        s = Continuous()._setup(x, Coordinate())  # 创建连续尺度对象，并设置默认坐标
        assert_series_equal(s(x), x)  # 断言计算的结果与预期的Series相等

    def test_coordinate_transform(self, x):  # 定义测试坐标转换方法

        s = Continuous(trans="log")._setup(x, Coordinate())  # 创建连续尺度对象，并设置对数坐标转换
        assert_series_equal(s(x), np.log10(x))  # 断言计算的结果与预期的对数转换结果相等

    def test_coordinate_transform_with_parameter(self, x):  # 定义测试带参数的坐标转换方法

        s = Continuous(trans="pow3")._setup(x, Coordinate())  # 创建连续尺度对象，并设置三次幂坐标转换
        assert_series_equal(s(x), np.power(x, 3))  # 断言计算的结果与预期的三次幂转换结果相等

    def test_coordinate_transform_error(self, x):  # 定义测试坐标转换错误处理方法

        s = Continuous(trans="bad")  # 创建连续尺度对象，使用错误的转换参数
        with pytest.raises(ValueError, match="Unknown value provided"):  # 断言抛出值错误，并匹配错误信息
            s._setup(x, Coordinate())  # 设置坐标

    def test_interval_defaults(self, x):  # 定义测试默认区间方法

        s = Continuous()._setup(x, IntervalProperty())  # 创建连续尺度对象，并设置默认区间属性
        assert_array_equal(s(x), [0, .25, 1])  # 断言计算的结果与预期的默认区间结果相等

    def test_interval_with_range(self, x):  # 定义测试带范围的区间方法

        s = Continuous((1, 3))._setup(x, IntervalProperty())  # 创建连续尺度对象，并设置指定范围的区间属性
        assert_array_equal(s(x), [1, 1.5, 3])  # 断言计算的结果与预期的指定范围区间结果相等

    def test_interval_with_norm(self, x):  # 定义测试带标准化的区间方法

        s = Continuous(norm=(3, 7))._setup(x, IntervalProperty())  # 创建连续尺度对象，并设置标准化的区间属性
        assert_array_equal(s(x), [-.5, 0, 1.5])  # 断言计算的结果与预期的标准化区间结果相等

    def test_interval_with_range_norm_and_transform(self, x):  # 定义测试带范围、标准化和转换的区间方法

        x = pd.Series([1, 10, 100])  # 创建包含不同值的Pandas Series
        # TODO param order?
        s = Continuous((2, 3), (10, 100), "log")._setup(x, IntervalProperty())  # 创建连续尺度对象，并设置范围、标准化和对数转换的区间属性
        assert_array_equal(s(x), [1, 2, 3])  # 断言计算的结果与预期的区间结果相等

    def test_interval_with_bools(self):  # 定义测试布尔值区间方法

        x = pd.Series([True, False, False])  # 创建包含布尔值的Pandas Series
        s = Continuous()._setup(x, IntervalProperty())  # 创建连续尺度对象，并设置布尔值区间属性
        assert_array_equal(s(x), [1, 0, 0])  # 断言计算的结果与预期的布尔值区间结果相等

    def test_color_defaults(self, x):  # 定义测试默认颜色方法

        cmap = color_palette("ch:", as_cmap=True)  # 使用seaborn中的调色板函数生成颜色映射
        s = Continuous()._setup(x, Color())  # 创建连续尺度对象，并设置默认颜色属性
        assert_array_equal(s(x), cmap([0, .25, 1])[:, :3])  # 断言计算的结果与预期的颜色映射结果相等，FIXME RGBA
    # 测试颜色映射的命名数值情况
    def test_color_named_values(self, x):
        # 使用 "viridis" 调色板创建颜色映射对象
        cmap = color_palette("viridis", as_cmap=True)
        # 创建 Continuous 对象并设置其属性
        s = Continuous("viridis")._setup(x, Color())
        # 断言实际输出与预期颜色映射的 RGBA 值相等
        assert_array_equal(s(x), cmap([0, .25, 1])[:, :3])  # FIXME RGBA

    # 测试颜色映射的元组数值情况
    def test_color_tuple_values(self, x):
        # 使用 "blend:b,g" 调色板创建混合颜色映射对象
        cmap = color_palette("blend:b,g", as_cmap=True)
        # 创建 Continuous 对象并设置其属性
        s = Continuous(("b", "g"))._setup(x, Color())
        # 断言实际输出与预期颜色映射的 RGBA 值相等
        assert_array_equal(s(x), cmap([0, .25, 1])[:, :3])  # FIXME RGBA

    # 测试颜色映射的可调用数值情况
    def test_color_callable_values(self, x):
        # 使用 "light:r" 调色板创建颜色映射对象
        cmap = color_palette("light:r", as_cmap=True)
        # 创建 Continuous 对象并设置其属性
        s = Continuous(cmap)._setup(x, Color())
        # 断言实际输出与预期颜色映射的 RGBA 值相等
        assert_array_equal(s(x), cmap([0, .25, 1])[:, :3])  # FIXME RGBA

    # 测试带有归一化的颜色映射情况
    def test_color_with_norm(self, x):
        # 使用 "ch:" 调色板创建颜色映射对象
        cmap = color_palette("ch:", as_cmap=True)
        # 创建 Continuous 对象并设置其属性
        s = Continuous(norm=(3, 7))._setup(x, Color())
        # 断言实际输出与预期颜色映射的 RGBA 值相等
        assert_array_equal(s(x), cmap([-.5, 0, 1.5])[:, :3])  # FIXME RGBA

    # 测试带有变换的颜色映射情况
    def test_color_with_transform(self, x):
        # 创建包含对数变换的 Continuous 对象
        x = pd.Series([1, 10, 100], name="x", dtype=float)
        # 使用 "ch:" 调色板创建颜色映射对象
        cmap = color_palette("ch:", as_cmap=True)
        # 创建 Continuous 对象并设置其属性
        s = Continuous(trans="log")._setup(x, Color())
        # 断言实际输出与预期颜色映射的 RGBA 值相等
        assert_array_equal(s(x), cmap([0, .5, 1])[:, :3])  # FIXME RGBA

    # 测试刻度定位器的设置情况
    def test_tick_locator(self, x):
        # 设置刻度位置列表
        locs = [.2, .5, .8]
        # 创建固定刻度定位器对象
        locator = mpl.ticker.FixedLocator(locs)
        # 使用设置刻度的方法并获取结果
        a = self.setup_ticks(x, locator)
        # 断言主要刻度定位器的实际输出与预期刻度位置列表相等
        assert_array_equal(a.major.locator(), locs)

    # 测试刻度定位器输入检查情况
    def test_tick_locator_input_check(self, x):
        # 定义预期的错误信息
        err = "Tick locator must be an instance of .*?, not <class 'tuple'>."
        # 使用 pytest 检查异常并捕获特定错误信息
        with pytest.raises(TypeError, match=err):
            # 调用 Continuous 类的 tick 方法，并传入非预期类型的参数
            Continuous().tick((1, 2))

    # 测试刻度定位器 upto 参数的情况
    def test_tick_upto(self, x):
        # 遍历不同的 upto 值进行测试
        for n in [2, 5, 10]:
            # 使用 upto 参数设置刻度
            a = self.setup_ticks(x, upto=n)
            # 断言主要刻度定位器的刻度数量不超过 (n + 1)
            assert len(a.major.locator()) <= (n + 1)

    # 测试刻度定位器 every 参数的情况
    def test_tick_every(self, x):
        # 遍历不同的 every 值进行测试
        for d in [.05, .2, .5]:
            # 使用 every 参数设置刻度
            a = self.setup_ticks(x, every=d)
            # 断言主要刻度定位器的刻度间距与预期值相近
            assert np.allclose(np.diff(a.major.locator()), d)

    # 测试刻度定位器 between 参数的情况
    def test_tick_every_between(self, x):
        # 定义区间范围的上下限
        lo, hi = .2, .8
        # 遍历不同的 every 值进行测试
        for d in [.05, .2, .5]:
            # 使用 between 参数设置刻度
            a = self.setup_ticks(x, every=d, between=(lo, hi))
            # 生成预期的刻度位置数组
            expected = np.arange(lo, hi + d, d)
            # 断言主要刻度定位器的实际输出与预期刻度位置数组相等
            assert_array_equal(a.major.locator(), expected)

    # 测试刻度定位器 at 参数的情况
    def test_tick_at(self, x):
        # 设置刻度位置列表
        locs = [.2, .5, .9]
        # 使用 at 参数设置刻度
        a = self.setup_ticks(x, at=locs)
        # 断言主要刻度定位器的实际输出与预期刻度位置列表相等
        assert_array_equal(a.major.locator(), locs)

    # 测试刻度定位器 count 参数的情况
    def test_tick_count(self, x):
        # 设置刻度数量
        n = 8
        # 使用 count 参数设置刻度
        a = self.setup_ticks(x, count=n)
        # 断言主要刻度定位器的实际输出与预期均匀分布的刻度位置数组相等
        assert_array_equal(a.major.locator(), np.linspace(0, 1, n))

    # 测试刻度定位器 count 和 between 参数的情况
    def test_tick_count_between(self, x):
        # 设置刻度数量
        n = 5
        # 定义区间范围的上下限
        lo, hi = .2, .7
        # 使用 count 和 between 参数设置刻度
        a = self.setup_ticks(x, count=n, between=(lo, hi))
        # 断言主要刻度定位器的实际输出与预期均匀分布的刻度位置数组相等
        assert_array_equal(a.major.locator(), np.linspace(lo, hi, n))
    # 定义一个测试方法，用于测试设置小刻度的情况
    def test_tick_minor(self, x):
        # 设定小刻度的数量为3
        n = 3
        # 调用设置刻度的辅助方法，获取设置好的刻度数组
        a = self.setup_ticks(x, count=2, minor=n)
        # 生成一个期望的刻度值数组，从0到1等分为n+2个点
        expected = np.linspace(0, 1, n + 2)
        # 如果 matplotlib 版本早于 "3.8.0rc1"，则调整期望的刻度值数组
        if _version_predates(mpl, "3.8.0rc1"):
            # 在 matplotlib 版本小于3.8时，小刻度包括最大的主刻度位置但不包括最小的主刻度位置
            expected = expected[1:]
        # 断言实际计算得到的小刻度定位器的结果与期望的刻度值数组相等
        assert_array_equal(a.minor.locator(), expected)

    # 定义一个测试方法，测试默认的对数刻度设置情况
    def test_log_tick_default(self, x):
        # 使用对数变换初始化一个连续型的刻度设置
        s = Continuous(trans="log")._setup(x, Coordinate())
        # 创建一个伪轴对象，使用 matplotlib 的对数刻度
        a = PseudoAxis(s._matplotlib_scale)
        # 设置轴的视图间隔为0.5到1050
        a.set_view_interval(.5, 1050)
        # 获取主刻度的定位器
        ticks = a.major.locator()
        # 断言所有刻度的对数值差都接近1
        assert np.allclose(np.diff(np.log10(ticks)), 1)

    # 定义一个测试方法，测试直到指定数目的对数刻度设置情况
    def test_log_tick_upto(self, x):
        # 设定小于等于3的对数刻度
        n = 3
        # 使用对数变换初始化一个连续型的刻度设置，并设置刻度直到小于等于n
        s = Continuous(trans="log").tick(upto=n)._setup(x, Coordinate())
        # 创建一个伪轴对象，使用 matplotlib 的对数刻度
        a = PseudoAxis(s._matplotlib_scale)
        # 断言主刻度的数量等于n
        assert a.major.locator.numticks == n

    # 定义一个测试方法，测试指定数量的对数刻度设置情况
    def test_log_tick_count(self, x):
        # 使用 pytest 断言引发运行时错误，要求包含 "count" 字段
        with pytest.raises(RuntimeError, match="`count` requires"):
            Continuous(trans="log").tick(count=4)

        # 使用对数变换初始化一个连续型的刻度设置，并设置刻度数量为4，范围在1到1000之间
        s = Continuous(trans="log").tick(count=4, between=(1, 1000))
        # 创建一个伪轴对象，使用 matplotlib 的对数刻度
        a = PseudoAxis(s._setup(x, Coordinate())._matplotlib_scale)
        # 设置轴的视图间隔为0.5到1050
        a.set_view_interval(.5, 1050)
        # 断言主刻度的定位器的结果与预期的刻度值数组相等
        assert_array_equal(a.major.locator(), [1, 10, 100, 1000])

    # 定义一个测试方法，测试禁用刻度标签格式的对数刻度设置情况
    def test_log_tick_format_disabled(self, x):
        # 使用对数变换初始化一个连续型的刻度设置，并禁用基数
        s = Continuous(trans="log").label(base=None)._setup(x, Coordinate())
        # 创建一个伪轴对象，使用 matplotlib 的对数刻度
        a = PseudoAxis(s._matplotlib_scale)
        # 设置轴的视图间隔为20到20000
        a.set_view_interval(20, 20000)
        # 获取主刻度的格式化标签
        labels = a.major.formatter.format_ticks(a.major.locator())
        # 对每个标签文本进行断言，检查其格式是否符合数字的正则表达式
        for text in labels:
            assert re.match(r"^\d+$", text)

    # 定义一个测试方法，测试不支持的每个刻度设置情况
    def test_log_tick_every(self, x):
        # 使用 pytest 断言引发运行时错误，不支持 "every" 字段
        with pytest.raises(RuntimeError, match="`every` not supported"):
            Continuous(trans="log").tick(every=2)

    # 定义一个测试方法，测试对称对数刻度的默认设置情况
    def test_symlog_tick_default(self, x):
        # 使用对称对数变换初始化一个连续型的刻度设置
        s = Continuous(trans="symlog")._setup(x, Coordinate())
        # 创建一个伪轴对象，使用 matplotlib 的对称对数刻度
        a = PseudoAxis(s._matplotlib_scale)
        # 设置轴的视图间隔为-1050到1050
        a.set_view_interval(-1050, 1050)
        # 获取主刻度的定位器
        ticks = a.major.locator()
        # 断言第一个刻度值为负数，且绝对值等于最后一个刻度值
        assert ticks[0] == -ticks[-1]
        # 对所有正数刻度值取绝对值并排序
        pos_ticks = np.sort(np.unique(np.abs(ticks)))
        # 断言所有正数刻度的对数值差都接近1
        assert np.allclose(np.diff(np.log10(pos_ticks[1:])), 1)
        # 断言第一个正数刻度值为0
        assert pos_ticks[0] == 0

    # 定义一个测试方法，测试标签格式化器设置情况
    def test_label_formatter(self, x):
        # 使用指定格式初始化一个 matplotlib 的格式化器
        fmt = mpl.ticker.FormatStrFormatter("%.3f")
        # 调用设置标签的辅助方法，并获取设置好的轴对象和位置数组
        a, locs = self.setup_labels(x, fmt)
        # 获取主刻度的格式化标签
        labels = a.major.formatter.format_ticks(locs)
        # 对每个标签文本进行断言，检查其格式是否符合小数点后三位的格式
        for text in labels:
            assert re.match(r"^\d\.\d{3}$", text)

    # 定义一个测试方法，测试类似于指定模式的标签格式化器设置情况
    def test_label_like_pattern(self, x):
        # 调用设置标签的辅助方法，使用类似于 ".4f" 的模式设置标签，并获取设置好的轴对象和位置数组
        a, locs = self.setup_labels(x, like=".4f")
        # 获取主刻度的格式化标签
        labels = a.major.formatter.format_ticks(locs)
        # 对每个标签文本进行断言，检查其格式是否符合小数点后四位的格式
        for text in labels:
            assert re.match(r"^\d\.\d{4}$", text)

    # 定义一个测试方法，测试类似于指定字符串的标签格式化器设置情况
    def test_label_like_string(self, x):
        # 调用设置标签的辅助方法，使用类似于 "x = {x:.1f}" 的字符串设置标签，并获取设置好的轴对象和位置数组
        a, locs = self.setup_labels(x, like="x = {x:.1f}")
        # 获取主刻度的格式化标签
        labels = a.major.formatter.format_ticks(locs)
        # 对每个标签文本进行断言，检查其格式是否符合 "x = 数字.数字" 的格式
        for text in labels:
            assert re.match(r"^x = \d\.\d$", text)
    # 测试用例：测试类中的标签生成函数
    def test_label_like_function(self, x):
        # 调用 `setup_labels` 方法设置标签，使用 `like="{:^5.1f}".format` 格式化标签
        a, locs = self.setup_labels(x, like="{:^5.1f}".format)
        # 使用标签的主要格式化器格式化刻度位置 `locs`
        labels = a.major.formatter.format_ticks(locs)
        # 对每个标签文本进行断言，检查其格式是否符合 "^ \d\.\d $" 的正则表达式
        for text in labels:
            assert re.match(r"^ \d\.\d $", text)
    
    # 测试用例：测试类中的基数标签生成函数
    def test_label_base(self, x):
        # 调用 `setup_labels` 方法设置标签，使用 100*x 和基数为 2
        a, locs = self.setup_labels(100 * x, base=2)
        # 使用标签的主要格式化器格式化刻度位置 `locs`
        labels = a.major.formatter.format_ticks(locs)
        # 对除第一个标签外的每个标签文本进行断言，检查其是否为空或包含 "2^"
        for text in labels[1:]:
            assert not text or "2^" in text
    
    # 测试用例：测试类中的单位标签生成函数
    def test_label_unit(self, x):
        # 调用 `setup_labels` 方法设置标签，使用 1000*x 和单位为 "g"
        a, locs = self.setup_labels(1000 * x, unit="g")
        # 使用标签的主要格式化器格式化刻度位置 `locs`
        labels = a.major.formatter.format_ticks(locs)
        # 对除第一个和最后一个标签外的每个标签文本进行断言，检查其格式是否匹配 "^\d+ mg$"
        for text in labels[1:-1]:
            assert re.match(r"^\d+ mg$", text)
    
    # 测试用例：测试类中带分隔符的单位标签生成函数
    def test_label_unit_with_sep(self, x):
        # 调用 `setup_labels` 方法设置标签，使用 1000*x 和单位为 ("", "g")
        a, locs = self.setup_labels(1000 * x, unit=("", "g"))
        # 使用标签的主要格式化器格式化刻度位置 `locs`
        labels = a.major.formatter.format_ticks(locs)
        # 对除第一个和最后一个标签外的每个标签文本进行断言，检查其格式是否匹配 "^\d+mg$"
        for text in labels[1:-1]:
            assert re.match(r"^\d+mg$", text)
    
    # 测试用例：测试类中空单位的标签生成函数
    def test_label_empty_unit(self, x):
        # 调用 `setup_labels` 方法设置标签，使用 1000*x 和空单位
        a, locs = self.setup_labels(1000 * x, unit="")
        # 使用标签的主要格式化器格式化刻度位置 `locs`
        labels = a.major.formatter.format_ticks(locs)
        # 对除第一个和最后一个标签外的每个标签文本进行断言，检查其格式是否匹配 "^\d+m$"
        for text in labels[1:-1]:
            assert re.match(r"^\d+m$", text)
    
    # 测试用例：测试类中从转换设置基数的标签生成函数
    def test_label_base_from_transform(self, x):
        # 创建连续标度 `Continuous`，转换设置为 "log"
        s = Continuous(trans="log")
        # 使用 `s._setup(x, Coordinate())` 返回的设置结果创建伪轴 `PseudoAxis`
        a = PseudoAxis(s._setup(x, Coordinate())._matplotlib_scale)
        # 设置视图间隔为 (10, 1000)
        a.set_view_interval(10, 1000)
        # 使用标签的主要格式化器格式化刻度位置 `[100]`
        label, = a.major.formatter.format_ticks([100])
        # 断言标签文本中是否包含 "10^{2}"
        assert r"10^{2}" in label
    
    # 测试用例：测试类中标签类型检查
    def test_label_type_checks(self):
        # 创建连续标度 `Continuous`
        s = Continuous()
        # 断言调用 `s.label("{x}")` 时是否抛出 `TypeError` 异常，且异常信息匹配 "Label formatter must be"
        with pytest.raises(TypeError, match="Label formatter must be"):
            s.label("{x}")
        
        # 断言调用 `s.label(like=2)` 时是否抛出 `TypeError` 异常，且异常信息匹配 "`like` must be"
        with pytest.raises(TypeError, match="`like` must be"):
            s.label(like=2)
# 定义一个测试类 TestNominal，用于测试 Nominal 类的功能
class TestNominal:

    # 使用 pytest 的 fixture 装饰器定义 x fixture，返回一个包含字符串的 Pandas Series 对象
    @pytest.fixture
    def x(self):
        return pd.Series(["a", "c", "b", "c"], name="x")

    # 使用 pytest 的 fixture 装饰器定义 y fixture，返回一个包含数值的 Pandas Series 对象
    @pytest.fixture
    def y(self):
        return pd.Series([1, -1.5, 3, -1.5], name="y")

    # 测试默认设置下的 Nominal 类 _setup 方法
    def test_coordinate_defaults(self, x):
        # 创建一个 Nominal 类实例 s，并调用其 _setup 方法，返回结果并与预期结果进行比较
        s = Nominal()._setup(x, Coordinate())
        assert_array_equal(s(x), np.array([0, 1, 2, 1], float))

    # 测试带有指定顺序的 Nominal 类 _setup 方法
    def test_coordinate_with_order(self, x):
        # 创建一个带有指定顺序的 Nominal 类实例 s，并调用其 _setup 方法，返回结果并与预期结果进行比较
        s = Nominal(order=["a", "b", "c"])._setup(x, Coordinate())
        assert_array_equal(s(x), np.array([0, 2, 1, 2], float))

    # 测试带有子集顺序的 Nominal 类 _setup 方法
    def test_coordinate_with_subset_order(self, x):
        # 创建一个带有子集顺序的 Nominal 类实例 s，并调用其 _setup 方法，返回结果并与预期结果进行比较
        s = Nominal(order=["c", "a"])._setup(x, Coordinate())
        assert_array_equal(s(x), np.array([1, 0, np.nan, 0], float))

    # 测试带有坐标轴参数的 Nominal 类 _setup 方法
    def test_coordinate_axis(self, x):
        # 创建一个坐标轴对象 ax，并使用其创建一个 Nominal 类实例 s，并调用其 _setup 方法，返回结果并与预期结果进行比较
        ax = mpl.figure.Figure().subplots()
        s = Nominal()._setup(x, Coordinate(), ax.xaxis)
        assert_array_equal(s(x), np.array([0, 1, 2, 1], float))
        # 获取坐标轴对象 ax 的主要格式化器 f，并验证格式化后的结果与预期结果一致
        f = ax.xaxis.get_major_formatter()
        assert f.format_ticks([0, 1, 2]) == ["a", "c", "b"]

    # 测试带有指定顺序和坐标轴参数的 Nominal 类 _setup 方法
    def test_coordinate_axis_with_order(self, x):
        # 创建一个带有指定顺序和坐标轴参数的 Nominal 类实例 s，并调用其 _setup 方法，返回结果并与预期结果进行比较
        order = ["a", "b", "c"]
        ax = mpl.figure.Figure().subplots()
        s = Nominal(order=order)._setup(x, Coordinate(), ax.xaxis)
        assert_array_equal(s(x), np.array([0, 2, 1, 2], float))
        # 获取坐标轴对象 ax 的主要格式化器 f，并验证格式化后的结果与预期结果一致
        f = ax.xaxis.get_major_formatter()
        assert f.format_ticks([0, 1, 2]) == order

    # 测试带有子集顺序和坐标轴参数的 Nominal 类 _setup 方法
    def test_coordinate_axis_with_subset_order(self, x):
        # 创建一个带有子集顺序和坐标轴参数的 Nominal 类实例 s，并调用其 _setup 方法，返回结果并与预期结果进行比较
        order = ["c", "a"]
        ax = mpl.figure.Figure().subplots()
        s = Nominal(order=order)._setup(x, Coordinate(), ax.xaxis)
        assert_array_equal(s(x), np.array([1, 0, np.nan, 0], float))
        # 获取坐标轴对象 ax 的主要格式化器 f，并验证格式化后的结果与预期结果一致
        f = ax.xaxis.get_major_formatter()
        assert f.format_ticks([0, 1, 2]) == [*order, ""]

    # 测试带有分类数据类型的 Nominal 类 _setup 方法
    def test_coordinate_axis_with_category_dtype(self, x):
        # 创建一个带有分类数据类型的 Nominal 类实例 s，并调用其 _setup 方法，返回结果并与预期结果进行比较
        order = ["b", "a", "d", "c"]
        x = x.astype(pd.CategoricalDtype(order))
        ax = mpl.figure.Figure().subplots()
        s = Nominal()._setup(x, Coordinate(), ax.xaxis)
        assert_array_equal(s(x), np.array([1, 3, 0, 3], float))
        # 获取坐标轴对象 ax 的主要格式化器 f，并验证格式化后的结果与预期结果一致
        f = ax.xaxis.get_major_formatter()
        assert f.format_ticks([0, 1, 2, 3]) == order

    # 测试带有数值数据的 Nominal 类 _setup 方法
    def test_coordinate_numeric_data(self, y):
        # 创建一个带有数值数据的 Nominal 类实例 s，并调用其 _setup 方法，返回结果并与预期结果进行比较
        ax = mpl.figure.Figure().subplots()
        s = Nominal()._setup(y, Coordinate(), ax.yaxis)
        assert_array_equal(s(y), np.array([1, 0, 2, 0], float))
        # 获取坐标轴对象 ax 的主要格式化器 f，并验证格式化后的结果与预期结果一致
        f = ax.yaxis.get_major_formatter()
        assert f.format_ticks([0, 1, 2]) == ["-1.5", "1.0", "3.0"]

    # 测试带有指定顺序和数值数据的 Nominal 类 _setup 方法
    def test_coordinate_numeric_data_with_order(self, y):
        # 创建一个带有指定顺序和数值数据的 Nominal 类实例 s，并调用其 _setup 方法，返回结果并与预期结果进行比较
        order = [1, 4, -1.5]
        ax = mpl.figure.Figure().subplots()
        s = Nominal(order=order)._setup(y, Coordinate(), ax.yaxis)
        assert_array_equal(s(y), np.array([0, 2, np.nan, 2], float))
        # 获取坐标轴对象 ax 的主要格式化器 f，并验证格式化后的结果与预期结果一致
        f = ax.yaxis.get_major_formatter()
        assert f.format_ticks([0, 1, 2]) == ["1.0", "4.0", "-1.5"]
    # 测试默认颜色情况下的功能
    def test_color_defaults(self, x):
        # 使用默认的 Nominal 对象和 Color 对象设置
        s = Nominal()._setup(x, Color())
        # 获取默认颜色调色板
        cs = color_palette()
        # 断言处理后的结果与预期的数组相等
        assert_array_equal(s(x), [cs[0], cs[1], cs[2], cs[1]])

    # 测试命名调色板的颜色情况
    def test_color_named_palette(self, x):
        # 指定命名调色板
        pal = "flare"
        s = Nominal(pal)._setup(x, Color())
        # 获取指定命名调色板的颜色列表
        cs = color_palette(pal, 3)
        # 断言处理后的结果与预期的数组相等
        assert_array_equal(s(x), [cs[0], cs[1], cs[2], cs[1]])

    # 测试列表形式的调色板
    def test_color_list_palette(self, x):
        # 指定列表形式的调色板
        cs = color_palette("crest", 3)
        s = Nominal(cs)._setup(x, Color())
        # 断言处理后的结果与预期的数组相等
        assert_array_equal(s(x), [cs[0], cs[1], cs[2], cs[1]])

    # 测试字典形式的调色板
    def test_color_dict_palette(self, x):
        # 指定字典形式的调色板
        cs = color_palette("crest", 3)
        # 创建对应字典
        pal = dict(zip("bac", cs))
        s = Nominal(pal)._setup(x, Color())
        # 断言处理后的结果与预期的数组相等
        assert_array_equal(s(x), [cs[1], cs[2], cs[0], cs[2]])

    # 测试数值数据情况下的颜色处理
    def test_color_numeric_data(self, y):
        # 使用默认的 Nominal 对象和 Color 对象设置
        s = Nominal()._setup(y, Color())
        # 获取默认颜色调色板
        cs = color_palette()
        # 断言处理后的结果与预期的数组相等
        assert_array_equal(s(y), [cs[1], cs[0], cs[2], cs[0]])

    # 测试数值数据与指定顺序子集的颜色处理
    def test_color_numeric_with_order_subset(self, y):
        # 指定顺序子集
        s = Nominal(order=[-1.5, 1])._setup(y, Color())
        # 获取两种颜色的调色板
        c1, c2 = color_palette(n_colors=2)
        # 定义空值
        null = (np.nan, np.nan, np.nan)
        # 断言处理后的结果与预期的数组相等
        assert_array_equal(s(y), [c2, c1, null, c1])

    @pytest.mark.xfail(reason="Need to sort out float/int order")
    # 测试数值数据中包含整数和浮点数混合的颜色处理（预期失败）
    def test_color_numeric_int_float_mix(self):
        # 创建包含整数和浮点数的 Series
        z = pd.Series([1, 2], name="z")
        s = Nominal(order=[1.0, 2])._setup(z, Color())
        # 获取两种颜色的调色板
        c1, c2 = color_palette(n_colors=2)
        # 定义空值
        null = (np.nan, np.nan, np.nan)
        # 断言处理后的结果与预期的数组相等
        assert_array_equal(s(z), [c1, null, c2])

    # 测试带有透明度的调色板处理
    def test_color_alpha_in_palette(self, x):
        # 指定带有透明度的颜色列表
        cs = [(.2, .2, .3, .5), (.1, .2, .3, 1), (.5, .6, .2, 0)]
        s = Nominal(cs)._setup(x, Color())
        # 断言处理后的结果与预期的数组相等
        assert_array_equal(s(x), [cs[0], cs[1], cs[2], cs[1]])

    # 测试未知调色板名称的情况
    def test_color_unknown_palette(self, x):
        # 指定未知的调色板名称
        pal = "not_a_palette"
        err = f"'{pal}' is not a valid palette name"
        # 断言抛出 ValueError 异常并匹配特定错误消息
        with pytest.raises(ValueError, match=err):
            Nominal(pal)._setup(x, Color())

    # 测试默认对象处理
    def test_object_defaults(self, x):
        # 定义 MockProperty 类来模拟对象属性
        class MockProperty(ObjectProperty):
            def _default_values(self, n):
                return list("xyz"[:n])

        s = Nominal()._setup(x, MockProperty())
        # 断言处理后的结果与预期的数组相等
        assert s(x) == ["x", "y", "z", "y"]

    # 测试对象列表处理
    def test_object_list(self, x):
        # 指定对象列表
        vs = ["x", "y", "z"]
        s = Nominal(vs)._setup(x, ObjectProperty())
        # 断言处理后的结果与预期的数组相等
        assert s(x) == ["x", "y", "z", "y"]

    # 测试对象字典处理
    def test_object_dict(self, x):
        # 指定对象字典
        vs = {"a": "x", "b": "y", "c": "z"}
        s = Nominal(vs)._setup(x, ObjectProperty())
        # 断言处理后的结果与预期的数组相等
        assert s(x) == ["x", "z", "y", "z"]

    # 测试对象顺序处理
    def test_object_order(self, x):
        # 指定对象列表和顺序
        vs = ["x", "y", "z"]
        s = Nominal(vs, order=["c", "a", "b"])._setup(x, ObjectProperty())
        # 断言处理后的结果与预期的数组相等
        assert s(x) == ["y", "x", "z", "x"]

    # 测试对象顺序子集处理
    def test_object_order_subset(self, x):
        # 指定对象列表和顺序子集
        vs = ["x", "y"]
        s = Nominal(vs, order=["a", "c"])._setup(x, ObjectProperty())
        # 断言处理后的结果与预期的数组相等
        assert s(x) == ["x", "y", None, "y"]
    # 定义一个测试方法，用于测试特殊对象的行为
    def test_objects_that_are_weird(self, x):
        # 初始化一组奇怪的对象
        vs = [("x", 1), (None, None, 0), {}]
        # 使用给定的对象属性创建 Nominal 实例，并进行初始化设置
        s = Nominal(vs)._setup(x, ObjectProperty())
        # 断言调用 s(x) 返回的结果与预期相符
        assert s(x) == [vs[0], vs[1], vs[2], vs[1]]

    # 定义一个测试方法，用于测试默认的 Alpha 属性
    def test_alpha_default(self, x):
        # 使用默认的 Alpha 属性创建 Nominal 实例，并进行初始化设置
        s = Nominal()._setup(x, Alpha())
        # 断言调用 s(x) 返回的结果与预期相符
        assert_array_equal(s(x), [.95, .625, .3, .625])

    # 定义一个测试方法，测试填充功能
    def test_fill(self):
        # 创建一个包含字符串的 Series 对象
        x = pd.Series(["a", "a", "b", "a"], name="x")
        # 使用 Fill 属性创建 Nominal 实例，并进行初始化设置
        s = Nominal()._setup(x, Fill())
        # 断言调用 s(x) 返回的结果与预期相符
        assert_array_equal(s(x), [True, True, False, True])

    # 定义一个测试方法，测试使用字典进行填充
    def test_fill_dict(self):
        # 创建一个包含字符串的 Series 对象
        x = pd.Series(["a", "a", "b", "a"], name="x")
        # 定义填充值的字典
        vs = {"a": False, "b": True}
        # 使用给定的填充字典创建 Nominal 实例，并进行初始化设置
        s = Nominal(vs)._setup(x, Fill())
        # 断言调用 s(x) 返回的结果与预期相符
        assert_array_equal(s(x), [False, False, True, False])

    # 定义一个测试方法，测试填充功能时的 nunique 警告
    def test_fill_nunique_warning(self):
        # 创建一个包含字符串的 Series 对象
        x = pd.Series(["a", "b", "c", "a", "b"], name="x")
        # 使用 Fill 属性创建 Nominal 实例，并进行初始化设置，同时捕获 UserWarning
        with pytest.warns(UserWarning, match="The variable assigned to fill"):
            s = Nominal()._setup(x, Fill())
        # 断言调用 s(x) 返回的结果与预期相符
        assert_array_equal(s(x), [True, False, True, True, False])

    # 定义一个测试方法，测试默认区间设置
    def test_interval_defaults(self, x):
        # 定义一个 MockProperty 类，继承自 IntervalProperty，设置默认区间为 (1, 2)
        class MockProperty(IntervalProperty):
            _default_range = (1, 2)
        # 使用 MockProperty 创建 Nominal 实例，并进行初始化设置
        s = Nominal()._setup(x, MockProperty())
        # 断言调用 s(x) 返回的结果与预期相符
        assert_array_equal(s(x), [2, 1.5, 1, 1.5])

    # 定义一个测试方法，测试使用元组作为区间设置
    def test_interval_tuple(self, x):
        # 使用元组 (1, 2) 创建 Nominal 实例，并进行初始化设置
        s = Nominal((1, 2))._setup(x, IntervalProperty())
        # 断言调用 s(x) 返回的结果与预期相符
        assert_array_equal(s(x), [2, 1.5, 1, 1.5])

    # 定义一个测试方法，测试使用数字作为区间设置
    def test_interval_tuple_numeric(self, y):
        # 使用元组 (1, 2) 创建 Nominal 实例，并进行初始化设置
        s = Nominal((1, 2))._setup(y, IntervalProperty())
        # 断言调用 s(y) 返回的结果与预期相符
        assert_array_equal(s(y), [1.5, 2, 1, 2])

    # 定义一个测试方法，测试使用列表作为区间设置
    def test_interval_list(self, x):
        # 定义一个包含数字的列表
        vs = [2, 5, 4]
        # 使用给定的列表创建 Nominal 实例，并进行初始化设置
        s = Nominal(vs)._setup(x, IntervalProperty())
        # 断言调用 s(x) 返回的结果与预期相符
        assert_array_equal(s(x), [2, 5, 4, 5])

    # 定义一个测试方法，测试使用字典作为区间设置
    def test_interval_dict(self, x):
        # 定义一个包含字符串和对应数值的字典
        vs = {"a": 3, "b": 4, "c": 6}
        # 使用给定的字典创建 Nominal 实例，并进行初始化设置
        s = Nominal(vs)._setup(x, IntervalProperty())
        # 断言调用 s(x) 返回的结果与预期相符
        assert_array_equal(s(x), [3, 6, 4, 6])

    # 定义一个测试方法，测试带有转换功能的区间设置
    def test_interval_with_transform(self, x):
        # 定义一个 MockProperty 类，继承自 IntervalProperty，设置转换函数为平方和平方根
        class MockProperty(IntervalProperty):
            _forward = np.square
            _inverse = np.sqrt
        # 使用元组 (2, 4) 创建 Nominal 实例，并进行初始化设置
        s = Nominal((2, 4))._setup(x, MockProperty())
        # 断言调用 s(x) 返回的结果与预期相符
        assert_array_equal(s(x), [4, np.sqrt(10), 2, np.sqrt(10)])

    # 定义一个测试方法，测试空数据的处理
    def test_empty_data(self):
        # 创建一个空的 Series 对象
        x = pd.Series([], dtype=object, name="x")
        # 使用 Coordinate 属性创建 Nominal 实例，并进行初始化设置
        s = Nominal()._setup(x, Coordinate())
        # 断言调用 s(x) 返回的结果为空列表
        assert_array_equal(s(x), [])

    # 定义一个测试方法，测试最终化过程
    def test_finalize(self, x):
        # 创建一个 Figure 对象的子图
        ax = mpl.figure.Figure().subplots()
        # 使用 Coordinate 属性创建 Nominal 实例，并进行初始化设置，同时指定 y 轴
        s = Nominal()._setup(x, Coordinate(), ax.yaxis)
        # 调用 _finalize 方法进行最终化处理，使用 Plot() 和 ax.yaxis 作为参数
        s._finalize(Plot(), ax.yaxis)

        # 获取唯一的级别值
        levels = x.unique()
        # 断言 y 轴的限制值与预期相符
        assert ax.get_ylim() == (len(levels) - .5, -.5)
        # 断言 y 轴的刻度值与预期相符
        assert_array_equal(ax.get_yticks(), list(range(len(levels))))
        # 遍历级别，断言 y 轴主要格式化程序的结果与预期相符
        for i, expected in enumerate(levels):
            assert ax.yaxis.major.formatter(i) == expected
    # 定义测试类 TestTemporal，用于测试 Temporal 类的各个方法
class TestTemporal:

    # 创建 pytest fixture t，返回一个包含指定日期的 Pandas Series 对象
    @pytest.fixture
    def t(self):
        dates = pd.to_datetime(["1972-09-27", "1975-06-24", "1980-12-14"])
        return pd.Series(dates, name="x")

    # 创建 pytest fixture x，返回一个将日期转换为 matplotlib 数字格式的 Pandas Series 对象
    @pytest.fixture
    def x(self, t):
        return pd.Series(mpl.dates.date2num(t), name=t.name)

    # 测试 Temporal 类的 _setup 方法与 Coordinate 参数的默认设置
    def test_coordinate_defaults(self, t, x):

        # 创建 Temporal 类的实例 s，使用 t 和默认的 Coordinate() 参数进行初始化
        s = Temporal()._setup(t, Coordinate())
        # 断言 s(t) 的结果与预期的 x 数组相等
        assert_array_equal(s(t), x)

    # 测试 Temporal 类的 _setup 方法与 IntervalProperty 参数的默认设置
    def test_interval_defaults(self, t, x):

        # 创建 Temporal 类的实例 s，使用 t 和默认的 IntervalProperty() 参数进行初始化
        s = Temporal()._setup(t, IntervalProperty())
        # 将 x 数组归一化处理，使其值范围在 [0, 1] 内
        normed = (x - x.min()) / (x.max() - x.min())
        # 断言 s(t) 的结果与归一化后的 normed 数组相等
        assert_array_equal(s(t), normed)

    # 测试 Temporal 类的 _setup 方法与 IntervalProperty 参数及给定数值范围的设置
    def test_interval_with_range(self, t, x):

        # 定义给定的数值范围
        values = (1, 3)
        # 创建 Temporal 类的实例 s，使用给定的数值范围 (1, 3) 和 IntervalProperty() 参数进行初始化
        s = Temporal((1, 3))._setup(t, IntervalProperty())
        # 将 x 数组归一化处理，使其值范围在 [0, 1] 内
        normed = (x - x.min()) / (x.max() - x.min())
        # 根据给定的数值范围 values 计算预期结果
        expected = normed * (values[1] - values[0]) + values[0]
        # 断言 s(t) 的结果与预期的 expected 数组相等
        assert_array_equal(s(t), expected)

    # 测试 Temporal 类的 _setup 方法与 IntervalProperty 参数及给定归一化范围的设置
    def test_interval_with_norm(self, t, x):

        # 定义归一化范围
        norm = t[1], t[2]
        # 创建 Temporal 类的实例 s，使用给定的归一化范围 norm 和 IntervalProperty() 参数进行初始化
        s = Temporal(norm=norm)._setup(t, IntervalProperty())
        # 将 x 数组归一化处理，使其值范围在 [0, 1] 内
        n = mpl.dates.date2num(norm)
        normed = (x - n[0]) / (n[1] - n[0])
        # 断言 s(t) 的结果与预期的 normed 数组相等
        assert_array_equal(s(t), normed)

    # 测试 Temporal 类的 _setup 方法与 Color 参数的默认设置
    def test_color_defaults(self, t, x):

        # 创建色彩映射 cmap，使用 "ch:" 颜色调色板
        cmap = color_palette("ch:", as_cmap=True)
        # 创建 Temporal 类的实例 s，使用默认的 Color() 参数进行初始化
        s = Temporal()._setup(t, Color())
        # 将 x 数组归一化处理，使其值范围在 [0, 1] 内
        normed = (x - x.min()) / (x.max() - x.min())
        # 断言 s(t) 的结果与 cmap 对归一化结果的前三列 RGBA 数组相等
        assert_array_equal(s(t), cmap(normed)[:, :3])  # FIXME RGBA

    # 测试 Temporal 类的 _setup 方法与 Color 参数及给定颜色映射名字的设置
    def test_color_named_values(self, t, x):

        # 定义颜色映射名字
        name = "viridis"
        # 创建色彩映射 cmap，使用指定的颜色映射名字
        cmap = color_palette(name, as_cmap=True)
        # 创建 Temporal 类的实例 s，使用给定的颜色映射名字 name 和 Color() 参数进行初始化
        s = Temporal(name)._setup(t, Color())
        # 将 x 数组归一化处理，使其值范围在 [0, 1] 内
        normed = (x - x.min()) / (x.max() - x.min())
        # 断言 s(t) 的结果与 cmap 对归一化结果的前三列 RGBA 数组相等
        assert_array_equal(s(t), cmap(normed)[:, :3])  # FIXME RGBA

    # 测试 Temporal 类的 _setup 方法与 Coordinate 参数及给定坐标轴的设置
    def test_coordinate_axis(self, t, x):

        # 创建一个 Matplotlib Figure 对象的子图 ax
        ax = mpl.figure.Figure().subplots()
        # 创建 Temporal 类的实例 s，使用默认的 Coordinate() 参数和指定的 x 轴 ax 进行初始化
        s = Temporal()._setup(t, Coordinate(), ax.xaxis)
        # 断言 s(t) 的结果与预期的 x 数组相等
        assert_array_equal(s(t), x)
        # 获取 ax x 轴的主定位器 locator 和主格式化器 formatter
        locator = ax.xaxis.get_major_locator()
        formatter = ax.xaxis.get_major_formatter()
        # 断言 locator 是 AutoDateLocator 类的实例，formatter 是 AutoDateFormatter 类的实例
        assert isinstance(locator, mpl.dates.AutoDateLocator)
        assert isinstance(formatter, mpl.dates.AutoDateFormatter)

    # 测试 Temporal 类的 tick 方法与给定定位器的设置
    def test_tick_locator(self, t):

        # 创建年定位器 locator，指定月份和日期
        locator = mpl.dates.YearLocator(month=3, day=15)
        # 创建 Temporal 类的实例 s，使用 YearLocator 定位器进行初始化
        s = Temporal().tick(locator)
        # 创建伪轴 a，使用 t 和 Coordinate() 进行初始化，并使用 Matplotlib 比例
        a = PseudoAxis(s._setup(t, Coordinate())._matplotlib_scale)
        a.set_view_interval(0, 365)
        # 断言 73 存在于伪轴 a 的主定位器中
        assert 73 in a.major.locator()

    # 测试 Temporal 类的 tick 方法与最大刻度数的设置
    def test_tick_upto(self, t, x):

        # 定义最大刻度数 n
        n = 8
        # 创建一个 Matplotlib Figure 对象的子图 ax
        ax = mpl.figure.Figure().subplots()
        # 创建 Temporal 类的实例，使用 upto=n 参数进行初始化，并指定 x 轴 ax
        Temporal().tick(upto=n)._setup(t, Coordinate(), ax.xaxis)
        # 获取 ax x 轴的主定位器 locator
        locator = ax.xaxis.get_major_locator()
        # 断言 locator 的最大刻度数值是 {n}
        assert set(locator.maxticks.values()) == {n}

    # 测试 Temporal 类的 label 方法与给定的格式化器的设置
    def test_label_formatter(self, t):

        # 创建日期格式化器 formatter，指定日期格式为 "%Y"
        formatter = mpl.dates.DateFormatter("%Y")
        # 创建 Temporal 类的实例 s，使用指定的 formatter 进行初始化
        s = Temporal().label(formatter)
        # 创建伪轴 a，使用 t 和 Coordinate() 进行初始化，并使用 Matplotlib 比例
        a = PseudoAxis(s._setup(t, Coordinate())._matplotlib_scale)
        a.set_view_interval(10, 1000)
        # 获取格式化后的标签 label
        label, = a.major.formatter.format_ticks([100])
        # 断言 label 的值为 "1970"
        assert label == "1970"
    # 定义一个测试方法，用于测试时间标签的简洁性设置
    def test_label_concise(self, t, x):
        # 创建一个 Matplotlib 图形对象，并获取其子图对象
        ax = mpl.figure.Figure().subplots()
        # 使用 Temporal 类的 label 方法设置时间标签，启用简洁模式，并初始化设置
        Temporal().label(concise=True)._setup(t, Coordinate(), ax.xaxis)
        # 获取 x 轴的主要格式化器
        formatter = ax.xaxis.get_major_formatter()
        # 断言 formatter 是 ConciseDateFormatter 类的实例
        assert isinstance(formatter, mpl.dates.ConciseDateFormatter)
    # 定义一个测试类 TestBoolean
class TestBoolean:
    
    # 定义一个 pytest fixture，返回一个包含布尔值的 Pandas Series
    @pytest.fixture
    def x(self):
        return pd.Series([True, False, False, True], name="x", dtype=bool)

    # 测试函数：测试 _setup 方法对坐标类 Coordinate 的处理
    def test_coordinate(self, x):
        # 创建 Boolean 对象并调用其 _setup 方法，设置数据和坐标对象
        s = Boolean()._setup(x, Coordinate())
        # 断言处理后的结果与期望结果相等
        assert_array_equal(s(x), x.astype(float))

    # 测试函数：测试 _setup 方法对带坐标轴的坐标类 Coordinate 的处理
    def test_coordinate_axis(self, x):
        # 创建一个 Matplotlib 的 figure 对象，并获取其子图
        ax = mpl.figure.Figure().subplots()
        # 创建 Boolean 对象并调用其 _setup 方法，设置数据、坐标对象和坐标轴对象
        s = Boolean()._setup(x, Coordinate(), ax.xaxis)
        # 断言处理后的结果与期望结果相等
        assert_array_equal(s(x), x.astype(float))
        # 获取坐标轴的主要格式化程序
        f = ax.xaxis.get_major_formatter()
        # 断言格式化后的刻度与预期列表相等
        assert f.format_ticks([0, 1]) == ["False", "True"]

    # 测试函数：测试 _setup 方法对带缺失值的坐标类 Coordinate 的处理，使用参数化
    @pytest.mark.parametrize(
        "dtype,value",
        [
            (object, np.nan),
            (object, None),
            ("boolean", pd.NA),
        ]
    )
    def test_coordinate_missing(self, x, dtype, value):
        # 将数据类型转换为指定的类型，并将指定位置的值设置为缺失值
        x = x.astype(dtype)
        x[2] = value
        # 创建 Boolean 对象并调用其 _setup 方法，设置数据和坐标对象
        s = Boolean()._setup(x, Coordinate())
        # 断言处理后的结果与期望结果相等
        assert_array_equal(s(x), x.astype(float))

    # 测试函数：测试 _setup 方法对颜色类 Color 的默认处理
    def test_color_defaults(self, x):
        # 创建 Boolean 对象并调用其 _setup 方法，设置数据和颜色对象
        s = Boolean()._setup(x, Color())
        # 获取默认的调色板
        cs = color_palette()
        # 根据数据的取反结果，构造期望的结果列表
        expected = [cs[int(x_i)] for x_i in ~x]
        # 断言处理后的结果与期望结果相等
        assert_array_equal(s(x), expected)

    # 测试函数：测试 _setup 方法对颜色类 Color 的列表调色板处理
    def test_color_list_palette(self, x):
        # 创建指定调色板和 Boolean 对象并调用其 _setup 方法，设置数据和颜色对象
        cs = color_palette("crest", 2)
        s = Boolean(cs)._setup(x, Color())
        # 根据数据的取反结果，构造期望的结果列表
        expected = [cs[int(x_i)] for x_i in ~x]
        # 断言处理后的结果与期望结果相等
        assert_array_equal(s(x), expected)

    # 测试函数：测试 _setup 方法对颜色类 Color 的元组调色板处理
    def test_color_tuple_palette(self, x):
        # 创建指定调色板元组和 Boolean 对象并调用其 _setup 方法，设置数据和颜色对象
        cs = tuple(color_palette("crest", 2))
        s = Boolean(cs)._setup(x, Color())
        # 根据数据的取反结果，构造期望的结果列表
        expected = [cs[int(x_i)] for x_i in ~x]
        # 断言处理后的结果与期望结果相等
        assert_array_equal(s(x), expected)

    # 测试函数：测试 _setup 方法对颜色类 Color 的字典调色板处理
    def test_color_dict_palette(self, x):
        # 创建指定调色板字典和 Boolean 对象并调用其 _setup 方法，设置数据和颜色对象
        cs = color_palette("crest", 2)
        pal = {True: cs[0], False: cs[1]}
        s = Boolean(pal)._setup(x, Color())
        # 根据数据的取值结果，构造期望的结果列表
        expected = [pal[x_i] for x_i in x]
        # 断言处理后的结果与期望结果相等
        assert_array_equal(s(x), expected)

    # 测试函数：测试 _setup 方法对对象属性类 ObjectProperty 的默认处理
    def test_object_defaults(self, x):
        # 定义一个包含属性名称的列表
        vs = ["x", "y", "z"]

        # 定义一个 MockProperty 类，继承自 ObjectProperty，用于模拟属性处理
        class MockProperty(ObjectProperty):
            # 重写 _default_values 方法，返回指定数量的默认属性名称
            def _default_values(self, n):
                return vs[:n]

        # 创建 Boolean 对象并调用其 _setup 方法，设置数据和 MockProperty 对象
        s = Boolean()._setup(x, MockProperty())
        # 根据数据的取反结果，构造期望的结果列表
        expected = [vs[int(x_i)] for x_i in ~x]
        # 断言处理后的结果与期望结果相等
        assert s(x) == expected

    # 测试函数：测试 _setup 方法对对象属性类 ObjectProperty 的列表处理
    def test_object_list(self, x):
        # 定义一个包含属性名称的列表
        vs = ["x", "y"]
        # 创建 Boolean 对象并调用其 _setup 方法，设置数据和 ObjectProperty 对象
        s = Boolean(vs)._setup(x, ObjectProperty())
        # 根据数据的取反结果，构造期望的结果列表
        expected = [vs[int(x_i)] for x_i in ~x]
        # 断言处理后的结果与期望结果相等
        assert s(x) == expected

    # 测试函数：测试 _setup 方法对对象属性类 ObjectProperty 的字典处理
    def test_object_dict(self, x):
        # 定义一个包含属性名称的字典
        vs = {True: "x", False: "y"}
        # 创建 Boolean 对象并调用其 _setup 方法，设置数据和 ObjectProperty 对象
        s = Boolean(vs)._setup(x, ObjectProperty())
        # 根据数据的取值结果，构造期望的结果列表
        expected = [vs[x_i] for x_i in x]
        # 断言处理后的结果与期望结果相等
        assert s(x) == expected

    # 测试函数：测试 _setup 方法对填充类 Fill 的默认处理
    def test_fill(self, x):
        # 创建 Boolean 对象并调用其 _setup 方法，设置数据和填充对象
        s = Boolean()._setup(x, Fill())
        # 断言处理后的结果与原始数据相等
        assert_array_equal(s(x), x)

    # 测试函数：测试 _setup 方法对区间属性类 IntervalProperty 的默认处理
    def test_interval_defaults(self, x):
        # 定义一个包含区间范围的元组
        vs = (1, 2)

        # 定义一个 MockProperty 类，继承自 IntervalProperty，用于模拟区间属性处理
        class MockProperty(IntervalProperty):
            # 定义一个类属性 _default_range，表示默认的区间范围
            _default_range = vs

        # 创建 Boolean 对象并调用其 _setup 方法，设置数据和 MockProperty 对象
        s = Boolean()._setup(x, MockProperty())
        # 根据数据的取值结果，构造期望的结果列表
        expected = [vs[int(x_i)] for x_i in x]
        # 断言处理后的结果与期望结果相等
        assert_array_equal(s(x), expected)
    # 定义一个测试方法，用于测试带有区间元组的布尔运算
    def test_interval_tuple(self, x):
        # 设定区间元组的值
        vs = (3, 5)
        # 创建一个 Boolean 对象，并用 x 和 IntervalProperty 设置它
        s = Boolean(vs)._setup(x, IntervalProperty())
        # 预期结果是根据 x 的值从 vs 中选取相应的元素组成的列表
        expected = [vs[int(x_i)] for x_i in x]
        # 断言调用 s(x) 得到的结果与预期列表相等
        assert_array_equal(s(x), expected)

    # 定义一个测试方法，用于测试 finalize 方法
    def test_finalize(self, x):
        # 创建一个 Figure 对象，并获取其子图
        ax = mpl.figure.Figure().subplots()
        # 创建一个 Boolean 对象，并用 x、Coordinate 对象和 ax 的 x 轴设置它
        s = Boolean()._setup(x, Coordinate(), ax.xaxis)
        # 调用 _finalize 方法，用 Plot 对象和 ax 的 x 轴作为参数
        s._finalize(Plot(), ax.xaxis)
        # 断言获取的 x 轴的限制范围是 (1.5, -0.5)
        assert ax.get_xlim() == (1.5, -0.5)
        # 断言获取的 x 轴刻度值是 [0, 1]
        assert_array_equal(ax.get_xticks(), [0, 1])
        # 断言 x 轴的主要格式化器对于索引 0 返回 "False"
        assert ax.xaxis.major.formatter(0) == "False"
        # 断言 x 轴的主要格式化器对于索引 1 返回 "True"
        assert ax.xaxis.major.formatter(1) == "True"
```