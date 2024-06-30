# `D:\src\scipysrc\seaborn\tests\_marks\test_base.py`

```
from dataclasses import dataclass  # 导入 dataclass 模块，用于创建数据类

import numpy as np  # 导入 NumPy 库，并用 np 别名表示
import pandas as pd  # 导入 Pandas 库，并用 pd 别名表示
import matplotlib as mpl  # 导入 Matplotlib 库，并用 mpl 别名表示

import pytest  # 导入 pytest 测试框架
from numpy.testing import assert_array_equal  # 从 NumPy 测试模块中导入数组相等断言函数

from seaborn._marks.base import Mark, Mappable, resolve_color  # 从 seaborn 库中的 _marks.base 模块导入 Mark, Mappable 类以及 resolve_color 函数


class TestMappable:

    def mark(self, **features):
        # 定义一个内部数据类 MockMark，继承自 Mark 类，用于模拟标记对象
        @dataclass
        class MockMark(Mark):
            linewidth: float = Mappable(rc="lines.linewidth")  # 线宽，默认从 rc 参数 lines.linewidth 中获取
            pointsize: float = Mappable(4)  # 点大小，默认为 4
            color: str = Mappable("C0")  # 颜色，默认为 "C0"
            fillcolor: str = Mappable(depend="color")  # 填充颜色，默认依赖于 color 参数
            alpha: float = Mappable(1)  # 透明度，默认为 1
            fillalpha: float = Mappable(depend="alpha")  # 填充透明度，默认依赖于 alpha 参数

        m = MockMark(**features)  # 使用传入的特征参数创建 MockMark 对象
        return m

    def test_repr(self):
        # 测试 __repr__ 方法，验证 Mappable 类的字符串表示形式
        assert str(Mappable(.5)) == "<0.5>"
        assert str(Mappable("CO")) == "<'CO'>"
        assert str(Mappable(rc="lines.linewidth")) == "<rc:lines.linewidth>"
        assert str(Mappable(depend="color")) == "<depend:color>"
        assert str(Mappable(auto=True)) == "<auto>"

    def test_input_checks(self):
        # 测试输入检查功能，确保 Mappable 类能正确处理无效参数的情况
        with pytest.raises(AssertionError):
            Mappable(rc="bogus.parameter")
        with pytest.raises(AssertionError):
            Mappable(depend="nonexistent_feature")

    def test_value(self):
        # 测试值解析功能，验证 Mark 对象能正确解析和返回指定特征的值
        val = 3
        m = self.mark(linewidth=val)
        assert m._resolve({}, "linewidth") == val  # 断言解析后的线宽值等于预期值

        df = pd.DataFrame(index=pd.RangeIndex(10))
        assert_array_equal(m._resolve(df, "linewidth"), np.full(len(df), val))  # 断言解析后的 DataFrame 线宽值数组与预期相等

    def test_default(self):
        # 测试默认值设置功能，确保 Mark 对象能正确处理默认值设定
        val = 3
        m = self.mark(linewidth=Mappable(val))
        assert m._resolve({}, "linewidth") == val  # 断言解析后的线宽值等于预期值

        df = pd.DataFrame(index=pd.RangeIndex(10))
        assert_array_equal(m._resolve(df, "linewidth"), np.full(len(df), val))  # 断言解析后的 DataFrame 线宽值数组与预期相等

    def test_rcparam(self):
        # 测试从 rc 参数读取值的功能，验证 Mark 对象能正确从 Matplotlib rc 参数获取线宽值
        param = "lines.linewidth"
        val = mpl.rcParams[param]  # 获取当前 Matplotlib 参数 lines.linewidth 的值

        m = self.mark(linewidth=Mappable(rc=param))
        assert m._resolve({}, "linewidth") == val  # 断言解析后的线宽值等于预期值

        df = pd.DataFrame(index=pd.RangeIndex(10))
        assert_array_equal(m._resolve(df, "linewidth"), np.full(len(df), val))  # 断言解析后的 DataFrame 线宽值数组与预期相等

    def test_depends(self):
        # 测试依赖关系功能，确保 Mark 对象能正确处理依赖于其他特征的情况
        val = 2
        df = pd.DataFrame(index=pd.RangeIndex(10))

        m = self.mark(pointsize=Mappable(val), linewidth=Mappable(depend="pointsize"))
        assert m._resolve({}, "linewidth") == val  # 断言解析后的线宽值等于预期值
        assert_array_equal(m._resolve(df, "linewidth"), np.full(len(df), val))  # 断言解析后的 DataFrame 线宽值数组与预期相等

        m = self.mark(pointsize=val * 2, linewidth=Mappable(depend="pointsize"))
        assert m._resolve({}, "linewidth") == val * 2  # 断言解析后的线宽值等于预期值乘以2
        assert_array_equal(m._resolve(df, "linewidth"), np.full(len(df), val * 2))  # 断言解析后的 DataFrame 线宽值数组与预期相等
    def test_mapped(self):
        # 定义一个字典，映射了字符到整数的关系
        values = {"a": 1, "b": 2, "c": 3}

        # 定义一个函数 f(x)，接受一个参数 x，并返回一个 numpy 数组，
        # 数组的每个元素根据 values 字典中对应 x_i 的值来构建
        def f(x):
            return np.array([values[x_i] for x_i in x])

        # 创建一个标记对象 m，指定 linewidth 为 Mappable 对象的 2 倍
        m = self.mark(linewidth=Mappable(2))

        # 定义一个字典 scales，将 "linewidth" 映射到函数 f
        scales = {"linewidth": f}

        # 使用 assert 断言语句，验证 _resolve 方法的返回值与预期值相等
        assert m._resolve({"linewidth": "c"}, "linewidth", scales) == 3

        # 创建一个包含 "linewidth" 列的 DataFrame 对象 df
        df = pd.DataFrame({"linewidth": ["a", "b", "c"]})

        # 创建预期结果 expected，是一个 numpy 数组，元素分别对应 values 中的值
        expected = np.array([1, 2, 3], float)

        # 使用 assert_array_equal 断言语句，验证 _resolve 方法的返回值与预期值相等
        assert_array_equal(m._resolve(df, "linewidth", scales), expected)

    def test_color(self):
        # 设置颜色 c 和透明度 a
        c, a = "C1", .5

        # 创建一个标记对象 m，指定颜色和透明度
        m = self.mark(color=c, alpha=a)

        # 使用 resolve_color 函数验证颜色的解析结果
        assert resolve_color(m, {}) == mpl.colors.to_rgba(c, a)

        # 创建一个包含 10 行索引的 DataFrame 对象 df
        df = pd.DataFrame(index=pd.RangeIndex(10))

        # 创建一个包含多个 c 元素的列表 cs
        cs = [c] * len(df)

        # 使用 assert_array_equal 断言语句，验证 resolve_color 方法的返回值与预期值相等
        assert_array_equal(resolve_color(m, df), mpl.colors.to_rgba_array(cs, a))

    def test_color_mapped_alpha(self):
        # 设置颜色 c 和透明度映射 values
        c = "r"
        values = {"a": .2, "b": .5, "c": .8}

        # 创建一个标记对象 m，指定颜色和透明度映射
        m = self.mark(color=c, alpha=Mappable(1))

        # 创建一个字典 scales，将 "alpha" 映射到一个 lambda 表达式
        scales = {"alpha": lambda s: np.array([values[s_i] for s_i in s])}

        # 使用 assert 断言语句，验证 resolve_color 方法的返回值与预期值相等
        assert resolve_color(m, {"alpha": "b"}, "", scales) == mpl.colors.to_rgba(c, .5)

        # 创建一个包含 "alpha" 列的 DataFrame 对象 df
        df = pd.DataFrame({"alpha": list(values.keys())})

        # 创建预期结果 expected，通过 to_rgba_array 设置颜色和 alpha 值
        expected = mpl.colors.to_rgba_array([c] * len(df))
        expected[:, 3] = list(values.values())

        # 使用 assert_array_equal 断言语句，验证 resolve_color 方法的返回值与预期值相等
        assert_array_equal(resolve_color(m, df, "", scales), expected)

    def test_color_scaled_as_strings(self):
        # 设置颜色列表 colors
        colors = ["C1", "dodgerblue", "#445566"]

        # 创建一个标记对象 m
        m = self.mark()

        # 创建一个字典 scales，将 "color" 映射到一个 lambda 表达式
        scales = {"color": lambda s: colors}

        # 使用 resolve_color 函数解析颜色，验证返回值与预期值相等
        actual = resolve_color(m, {"color": pd.Series(["a", "b", "c"])}, "", scales)
        expected = mpl.colors.to_rgba_array(colors)

        # 使用 assert_array_equal 断言语句，验证解析结果与预期值相等
        assert_array_equal(actual, expected)

    def test_fillcolor(self):
        # 设置颜色 c、透明度 a 和填充透明度 fa
        c, a = "green", .8
        fa = .2

        # 创建一个标记对象 m，指定颜色、透明度和填充属性
        m = self.mark(
            color=c, alpha=a,
            fillcolor=Mappable(depend="color"), fillalpha=Mappable(fa),
        )

        # 使用 resolve_color 函数解析颜色，验证返回值与预期值相等
        assert resolve_color(m, {}) == mpl.colors.to_rgba(c, a)

        # 使用 resolve_color 函数解析颜色（填充属性），验证返回值与预期值相等
        assert resolve_color(m, {}, "fill") == mpl.colors.to_rgba(c, fa)

        # 创建一个包含 10 行索引的 DataFrame 对象 df
        df = pd.DataFrame(index=pd.RangeIndex(10))

        # 创建一个包含多个 c 元素的列表 cs
        cs = [c] * len(df)

        # 使用 assert_array_equal 断言语句，验证 resolve_color 方法的返回值与预期值相等
        assert_array_equal(resolve_color(m, df), mpl.colors.to_rgba_array(cs, a))

        # 使用 assert_array_equal 断言语句，验证 resolve_color 方法的返回值与预期值相等（填充属性）
        assert_array_equal(
            resolve_color(m, df, "fill"), mpl.colors.to_rgba_array(cs, fa)
        )
```