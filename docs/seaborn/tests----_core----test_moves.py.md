# `D:\src\scipysrc\seaborn\tests\_core\test_moves.py`

```
# 引入 itertools 模块中的 product 函数，用于生成多个可迭代对象的笛卡尔积
from itertools import product

# 引入 numpy 库，并将其命名为 np，用于科学计算和数组操作
import numpy as np

# 引入 pandas 库，并将其命名为 pd，用于数据处理和分析
import pandas as pd

# 从 pandas.testing 模块中引入 assert_series_equal 函数，用于比较两个 Series 是否相等
from pandas.testing import assert_series_equal

# 从 numpy.testing 模块中引入 assert_array_equal 和 assert_array_almost_equal 函数，
# 用于比较两个数组是否相等或几乎相等
from numpy.testing import assert_array_equal, assert_array_almost_equal

# 从 seaborn._core.moves 模块中引入 Dodge, Jitter, Shift, Stack, Norm 类，
# 这些类用于实现不同的数据移动方式
from seaborn._core.moves import Dodge, Jitter, Shift, Stack, Norm

# 从 seaborn._core.rules 模块中引入 categorical_order 函数，用于获取分类变量的顺序信息
from seaborn._core.rules import categorical_order

# 从 seaborn._core.groupby 模块中引入 GroupBy 类，用于对数据进行分组操作
from seaborn._core.groupby import GroupBy

# 引入 pytest 库，用于编写和运行测试
import pytest


class MoveFixtures:

    @pytest.fixture
    def df(self, rng):
        # 使用 rng 生成随机数种子，用于生成数据
        n = 50
        data = {
            "x": rng.choice([0., 1., 2., 3.], n),  # 随机选择 [0., 1., 2., 3.] 中的值作为 'x' 列的数据
            "y": rng.normal(0, 1, n),  # 从正态分布 N(0, 1) 生成 n 个随机数作为 'y' 列的数据
            "grp2": rng.choice(["a", "b"], n),  # 随机选择 ["a", "b"] 中的值作为 'grp2' 列的数据
            "grp3": rng.choice(["x", "y", "z"], n),  # 随机选择 ["x", "y", "z"] 中的值作为 'grp3' 列的数据
            "width": 0.8,  # 设置 'width' 列的值为 0.8
            "baseline": 0,  # 设置 'baseline' 列的值为 0
        }
        return pd.DataFrame(data)  # 返回生成的 DataFrame 对象

    @pytest.fixture
    def toy_df(self):
        # 生成一个简化版的 DataFrame 对象
        data = {
            "x": [0, 0, 1],  # 'x' 列的数据
            "y": [1, 2, 3],  # 'y' 列的数据
            "grp": ["a", "b", "b"],  # 'grp' 列的数据
            "width": .8,  # 设置 'width' 列的值为 0.8
            "baseline": 0,  # 设置 'baseline' 列的值为 0
        }
        return pd.DataFrame(data)  # 返回生成的 DataFrame 对象

    @pytest.fixture
    def toy_df_widths(self, toy_df):
        # 复制 toy_df，修改其 'width' 列的值
        toy_df["width"] = [.8, .2, .4]
        return toy_df  # 返回修改后的 DataFrame 对象

    @pytest.fixture
    def toy_df_facets(self):
        # 生成一个带有多个分面的 DataFrame 对象
        data = {
            "x": [0, 0, 1, 0, 1, 2],  # 'x' 列的数据
            "y": [1, 2, 3, 1, 2, 3],  # 'y' 列的数据
            "grp": ["a", "b", "a", "b", "a", "b"],  # 'grp' 列的数据
            "col": ["x", "x", "x", "y", "y", "y"],  # 'col' 列的数据
            "width": .8,  # 设置 'width' 列的值为 0.8
            "baseline": 0,  # 设置 'baseline' 列的值为 0
        }
        return pd.DataFrame(data)  # 返回生成的 DataFrame 对象


class TestJitter(MoveFixtures):

    def get_groupby(self, data, orient):
        # 根据 orient 参数确定要分组的变量，并返回 GroupBy 对象
        other = {"x": "y", "y": "x"}[orient]
        variables = [v for v in data if v not in [other, "width"]]
        return GroupBy(variables)

    def check_same(self, res, df, *cols):
        # 检查 res 中的列与 df 中指定的列是否相同
        for col in cols:
            assert_series_equal(res[col], df[col])

    def check_pos(self, res, df, var, limit):
        # 检查 res 中的 var 列相对于 df 中 var 列的位置是否满足限制
        assert (res[var] != df[var]).all()
        assert (res[var] < df[var] + limit / 2).all()
        assert (res[var] > df[var] - limit / 2).all()

    def test_default(self, df):
        # 测试 Jitter 类的默认行为

        orient = "x"  # 设定 orient 参数为 "x"
        groupby = self.get_groupby(df, orient)  # 获取分组信息
        res = Jitter()(df, groupby, orient, {})  # 调用 Jitter 类进行数据处理
        self.check_same(res, df, "y", "grp2", "width")  # 检查处理后的结果与原始 df 是否相同
        self.check_pos(res, df, "x", 0.2 * df["width"])  # 检查 'x' 列的位置是否满足偏移限制
        assert (res["x"] - df["x"]).abs().min() > 0  # 断言处理后的 'x' 列与原始 'x' 列的最小绝对差大于 0

    def test_width(self, df):
        # 测试 Jitter 类在指定宽度下的行为

        width = .4  # 设置宽度为 0.4
        orient = "x"  # 设定 orient 参数为 "x"
        groupby = self.get_groupby(df, orient)  # 获取分组信息
        res = Jitter(width=width)(df, groupby, orient, {})  # 调用 Jitter 类进行数据处理
        self.check_same(res, df, "y", "grp2", "width")  # 检查处理后的结果与原始 df 是否相同
        self.check_pos(res, df, "x", width * df["width"])  # 检查 'x' 列的位置是否满足指定宽度的偏移限制

    def test_x(self, df):
        # 测试 Jitter 类在指定 x 偏移下的行为

        val = .2  # 设置 x 偏移为 0.2
        orient = "x"  # 设定 orient 参数为 "x"
        groupby = self.get_groupby(df, orient)  # 获取分组信息
        res = Jitter(x=val)(df, groupby, orient, {})  # 调用 Jitter 类进行数据处理
        self.check_same(res, df, "y", "grp2", "width")  # 检查处理后的结果与原始 df 是否相同
        self.check_pos(res, df, "x", val)  # 检查 'x' 列的位置是否满足指定的 x 偏移
    # 定义一个测试方法，用于测试 Jitter 类的 y 参数设置
    def test_y(self, df):
        # 设置一个浮点型变量 val 为 0.2
        val = .2
        # 设置方向 orient 为 "x"
        orient = "x"
        # 调用 self 对象的 get_groupby 方法，获取基于 orient 方向的分组
        groupby = self.get_groupby(df, orient)
        # 使用 Jitter 类初始化并调用，传入参数 df, groupby, orient, 空字典 {}
        res = Jitter(y=val)(df, groupby, orient, {})
        # 调用 self 对象的 check_same 方法，验证 res 对象的 "x", "grp2", "width" 属性与 df 的对应属性是否相同
        self.check_same(res, df, "x", "grp2", "width")
        # 调用 self 对象的 check_pos 方法，验证 res 对象的 "y" 属性与预期值 val 是否相符
        self.check_pos(res, df, "y", val)

    # 定义一个测试方法，用于测试 Jitter 类的 seed 参数设置
    def test_seed(self, df):
        # 定义一个字典 kws，包含 width=.2, y=.1, seed=0 这些参数设置
        kws = dict(width=.2, y=.1, seed=0)
        # 设置方向 orient 为 "x"
        orient = "x"
        # 调用 self 对象的 get_groupby 方法，获取基于 orient 方向的分组
        groupby = self.get_groupby(df, orient)
        # 使用 Jitter 类分别初始化两次，传入参数 df, groupby, orient, 空字典 {}
        res1 = Jitter(**kws)(df, groupby, orient, {})
        res2 = Jitter(**kws)(df, groupby, orient, {})
        # 遍历字符串 "xy"，对 res1 和 res2 的 "x" 和 "y" 属性进行相等性断言
        for var in "xy":
            assert_series_equal(res1[var], res2[var])
class TestDodge(MoveFixtures):
    # 继承自 MoveFixtures 的测试类 TestDodge

    # First some very simple toy examples

    def test_default(self, toy_df):
        # 测试默认情况下的 Dodge 函数行为，使用 toy_df 数据
        groupby = GroupBy(["x", "grp"])
        # 创建 GroupBy 对象，按照 "x" 和 "grp" 列分组
        res = Dodge()(toy_df, groupby, "x", {})
        # 调用 Dodge 类处理 toy_df 数据，返回处理结果

        assert_array_equal(res["y"], [1, 2, 3]),
        # 断言处理结果中 "y" 列的值为 [1, 2, 3]
        assert_array_almost_equal(res["x"], [-.2, .2, 1.2])
        # 断言处理结果中 "x" 列的值近似为 [-0.2, 0.2, 1.2]
        assert_array_almost_equal(res["width"], [.4, .4, .4])
        # 断言处理结果中 "width" 列的值近似为 [0.4, 0.4, 0.4]

    def test_fill(self, toy_df):
        # 测试使用 empty="fill" 参数的 Dodge 函数行为，使用 toy_df 数据
        groupby = GroupBy(["x", "grp"])
        # 创建 GroupBy 对象，按照 "x" 和 "grp" 列分组
        res = Dodge(empty="fill")(toy_df, groupby, "x", {})
        # 调用 Dodge 类处理 toy_df 数据，使用 "fill" 模式处理空值，返回处理结果

        assert_array_equal(res["y"], [1, 2, 3]),
        # 断言处理结果中 "y" 列的值为 [1, 2, 3]
        assert_array_almost_equal(res["x"], [-.2, .2, 1])
        # 断言处理结果中 "x" 列的值近似为 [-0.2, 0.2, 1]
        assert_array_almost_equal(res["width"], [.4, .4, .8])
        # 断言处理结果中 "width" 列的值近似为 [0.4, 0.4, 0.8]

    def test_drop(self, toy_df):
        # 测试使用 empty="drop" 参数的 Dodge 函数行为，使用 toy_df 数据
        groupby = GroupBy(["x", "grp"])
        # 创建 GroupBy 对象，按照 "x" 和 "grp" 列分组
        res = Dodge("drop")(toy_df, groupby, "x", {})
        # 调用 Dodge 类处理 toy_df 数据，使用 "drop" 模式处理空值，返回处理结果

        assert_array_equal(res["y"], [1, 2, 3])
        # 断言处理结果中 "y" 列的值为 [1, 2, 3]
        assert_array_almost_equal(res["x"], [-.2, .2, 1])
        # 断言处理结果中 "x" 列的值近似为 [-0.2, 0.2, 1]
        assert_array_almost_equal(res["width"], [.4, .4, .4])
        # 断言处理结果中 "width" 列的值近似为 [0.4, 0.4, 0.4]

    def test_gap(self, toy_df):
        # 测试使用 gap=.25 参数的 Dodge 函数行为，使用 toy_df 数据
        groupby = GroupBy(["x", "grp"])
        # 创建 GroupBy 对象，按照 "x" 和 "grp" 列分组
        res = Dodge(gap=.25)(toy_df, groupby, "x", {})
        # 调用 Dodge 类处理 toy_df 数据，设置间隙为 0.25，返回处理结果

        assert_array_equal(res["y"], [1, 2, 3])
        # 断言处理结果中 "y" 列的值为 [1, 2, 3]
        assert_array_almost_equal(res["x"], [-.2, .2, 1.2])
        # 断言处理结果中 "x" 列的值近似为 [-0.2, 0.2, 1.2]
        assert_array_almost_equal(res["width"], [.3, .3, .3])
        # 断言处理结果中 "width" 列的值近似为 [0.3, 0.3, 0.3]

    def test_widths_default(self, toy_df_widths):
        # 测试默认情况下的 Dodge 函数行为，使用 toy_df_widths 数据
        groupby = GroupBy(["x", "grp"])
        # 创建 GroupBy 对象，按照 "x" 和 "grp" 列分组
        res = Dodge()(toy_df_widths, groupby, "x", {})
        # 调用 Dodge 类处理 toy_df_widths 数据，返回处理结果

        assert_array_equal(res["y"], [1, 2, 3])
        # 断言处理结果中 "y" 列的值为 [1, 2, 3]
        assert_array_almost_equal(res["x"], [-.08, .32, 1.1])
        # 断言处理结果中 "x" 列的值近似为 [-0.08, 0.32, 1.1]
        assert_array_almost_equal(res["width"], [.64, .16, .2])
        # 断言处理结果中 "width" 列的值近似为 [0.64, 0.16, 0.2]

    def test_widths_fill(self, toy_df_widths):
        # 测试使用 empty="fill" 参数的 Dodge 函数行为，使用 toy_df_widths 数据
        groupby = GroupBy(["x", "grp"])
        # 创建 GroupBy 对象，按照 "x" 和 "grp" 列分组
        res = Dodge(empty="fill")(toy_df_widths, groupby, "x", {})
        # 调用 Dodge 类处理 toy_df_widths 数据，使用 "fill" 模式处理空值，返回处理结果

        assert_array_equal(res["y"], [1, 2, 3])
        # 断言处理结果中 "y" 列的值为 [1, 2, 3]
        assert_array_almost_equal(res["x"], [-.08, .32, 1])
        # 断言处理结果中 "x" 列的值近似为 [-0.08, 0.32, 1]
        assert_array_almost_equal(res["width"], [.64, .16, .4])
        # 断言处理结果中 "width" 列的值近似为 [0.64, 0.16, 0.4]

    def test_widths_drop(self, toy_df_widths):
        # 测试使用 empty="drop" 参数的 Dodge 函数行为，使用 toy_df_widths 数据
        groupby = GroupBy(["x", "grp"])
        # 创建 GroupBy 对象，按照 "x" 和 "grp" 列分组
        res = Dodge(empty="drop")(toy_df_widths, groupby, "x", {})
        # 调用 Dodge 类处理 toy_df_widths 数据，使用 "drop" 模式处理空值，返回处理结果

        assert_array_equal(res["y"], [1, 2, 3])
        # 断言处理结果中 "y" 列的值为 [1, 2, 3]
        assert_array_almost_equal(res["x"], [-.08, .32, 1])
        # 断言处理结果中 "x" 列的值近似为 [-0.08, 0.32, 1]
        assert_array_almost_equal(res["width"], [.64, .16, .2])
        # 断言处理结果中 "width" 列的值近似为 [0.64, 0.16, 0.2]

    def test_faceted_default(self, toy_df_facets):
        # 测试默认情况下的 Dodge 函数行为，使用 toy_df_facets 数据
        groupby = GroupBy(["x", "grp", "col"])
        # 创建 GroupBy 对象，按照 "x", "grp", "col" 列分组
        res = Dodge()(toy_df_facets, groupby, "x", {})
        # 调用 Dodge 类处理 toy_df_facets 数据，返回处理结果

        assert_array_equal(res["y"], [1, 2, 3, 1, 2, 3])
        # 断言处理结果中 "y" 列的值为 [1, 2, 3, 1, 2, 3]
        assert_array_almost_equal(res["x"], [-.2, .2, .8, .2, .8, 2.2
    # 定义一个测试方法，用于测试带有分面数据框的 faceted drop 操作
    def test_faceted_drop(self, toy_df_facets):
        # 创建一个 GroupBy 对象，按照指定的列进行分组
        groupby = GroupBy(["x", "grp", "col"])
        # 使用 Dodge 策略进行处理，返回处理后的结果
        res = Dodge(empty="drop")(toy_df_facets, groupby, "x", {})

        # 断言处理后的结果的 y 列与预期值一致
        assert_array_equal(res["y"], [1, 2, 3, 1, 2, 3])
        # 断言处理后的结果的 x 列近似于预期值
        assert_array_almost_equal(res["x"], [-.2, .2, 1, 0, 1, 2])
        # 断言处理后的结果的 width 列近似于预期值
        assert_array_almost_equal(res["width"], [.4] * 6)

    # 定义一个测试方法，用于测试方向变换
    def test_orient(self, toy_df):
        # 复制 toy_df，交换 x 和 y 列的数据
        df = toy_df.assign(x=toy_df["y"], y=toy_df["x"])

        # 创建一个 GroupBy 对象，按照指定的列进行分组
        groupby = GroupBy(["y", "grp"])
        # 使用 Dodge 策略进行处理，返回处理后的结果
        res = Dodge("drop")(df, groupby, "y", {})

        # 断言处理后的结果的 x 列与预期值一致
        assert_array_equal(res["x"], [1, 2, 3])
        # 断言处理后的结果的 y 列近似于预期值
        assert_array_almost_equal(res["y"], [-.2, .2, 1])
        # 断言处理后的结果的 width 列近似于预期值
        assert_array_almost_equal(res["width"], [.4, .4, .4])

    # 现在测试稍微复杂的数据情况
    @pytest.mark.parametrize("grp", ["grp2", "grp3"])
    # 定义一个测试方法，测试单一语义的 dodge 操作
    def test_single_semantic(self, df, grp):
        # 创建一个 GroupBy 对象，按照指定的列进行分组
        groupby = GroupBy(["x", grp])
        # 使用 Dodge 策略进行处理，返回处理后的结果
        res = Dodge()(df, groupby, "x", {})

        # 获取数据框中 grp 列的类别顺序
        levels = categorical_order(df[grp])
        w, n = 0.8, len(levels)

        # 计算偏移量数组，使得 dodge 的结果能够展示出分组效果
        shifts = np.linspace(0, w - w / n, n)
        shifts -= shifts.mean()

        # 断言处理后的结果的 y 列与原始数据的 y 列一致
        assert_series_equal(res["y"], df["y"])
        # 断言处理后的结果的 width 列为原始数据的 width 列除以组数 n 的结果
        assert_series_equal(res["width"], df["width"] / n)

        # 遍历每个类别和其对应的偏移量，检查处理后的 x 列是否正确偏移
        for val, shift in zip(levels, shifts):
            rows = df[grp] == val
            assert_series_equal(res.loc[rows, "x"], df.loc[rows, "x"] + shift)

    # 定义一个测试方法，测试两个语义的 dodge 操作
    def test_two_semantics(self, df):
        # 创建一个 GroupBy 对象，按照指定的列进行分组
        groupby = GroupBy(["x", "grp2", "grp3"])
        # 使用 Dodge 策略进行处理，返回处理后的结果
        res = Dodge()(df, groupby, "x", {})

        # 获取数据框中 grp2 和 grp3 列的类别顺序
        levels = categorical_order(df["grp2"]), categorical_order(df["grp3"])
        w, n = 0.8, len(levels[0]) * len(levels[1])

        # 计算偏移量数组，使得 dodge 的结果能够展示出分组效果
        shifts = np.linspace(0, w - w / n, n)
        shifts -= shifts.mean()

        # 断言处理后的结果的 y 列与原始数据的 y 列一致
        assert_series_equal(res["y"], df["y"])
        # 断言处理后的结果的 width 列为原始数据的 width 列除以组数 n 的结果
        assert_series_equal(res["width"], df["width"] / n)

        # 遍历每个组合和其对应的偏移量，检查处理后的 x 列是否正确偏移
        for (v2, v3), shift in zip(product(*levels), shifts):
            rows = (df["grp2"] == v2) & (df["grp3"] == v3)
            assert_series_equal(res.loc[rows, "x"], df.loc[rows, "x"] + shift)
class TestStack(MoveFixtures):

    # 测试基本情况下的堆叠操作
    def test_basic(self, toy_df):

        # 创建一个GroupBy对象，指定分组列为["color", "group"]
        groupby = GroupBy(["color", "group"])
        # 调用Stack类的实例进行堆叠操作，将结果存储在res中
        res = Stack()(toy_df, groupby, "x", {})

        # 断言堆叠后"x"列的结果是否与期望的数组相等
        assert_array_equal(res["x"], [0, 0, 1])
        # 断言堆叠后"y"列的结果是否与期望的数组相等
        assert_array_equal(res["y"], [1, 3, 3])
        # 断言堆叠后"baseline"列的结果是否与期望的数组相等
        assert_array_equal(res["baseline"], [0, 1, 0])

    # 测试多面板情况下的堆叠操作
    def test_faceted(self, toy_df_facets):

        # 创建一个GroupBy对象，指定分组列为["color", "group"]
        groupby = GroupBy(["color", "group"])
        # 调用Stack类的实例进行堆叠操作，将结果存储在res中
        res = Stack()(toy_df_facets, groupby, "x", {})

        # 断言堆叠后"x"列的结果是否与期望的数组相等
        assert_array_equal(res["x"], [0, 0, 1, 0, 1, 2])
        # 断言堆叠后"y"列的结果是否与期望的数组相等
        assert_array_equal(res["y"], [1, 3, 3, 1, 2, 3])
        # 断言堆叠后"baseline"列的结果是否与期望的数组相等
        assert_array_equal(res["baseline"], [0, 1, 0, 0, 0, 0])

    # 测试包含缺失数据情况下的堆叠操作
    def test_misssing_data(self, toy_df):

        # 创建一个包含缺失数据的DataFrame
        df = pd.DataFrame({
            "x": [0, 0, 0],
            "y": [2, np.nan, 1],
            "baseline": [0, 0, 0],
        })
        # 调用Stack类的实例进行堆叠操作，将结果存储在res中
        res = Stack()(df, None, "x", {})
        
        # 断言堆叠后"y"列的结果是否与期望的数组相等
        assert_array_equal(res["y"], [2, np.nan, 3])
        # 断言堆叠后"baseline"列的结果是否与期望的数组相等
        assert_array_equal(res["baseline"], [0, np.nan, 2])

    # 测试基线一致性检查
    def test_baseline_homogeneity_check(self, toy_df):

        # 修改toy_df的"baseline"列
        toy_df["baseline"] = [0, 1, 2]
        # 创建一个GroupBy对象，指定分组列为["color", "group"]
        groupby = GroupBy(["color", "group"])
        # 创建一个Stack类的实例
        move = Stack()
        err = "Stack move cannot be used when baselines"
        # 使用pytest断言捕获RuntimeError异常，确保在指定条件下抛出异常
        with pytest.raises(RuntimeError, match=err):
            # 调用Stack类的实例进行堆叠操作，预期会抛出异常
            move(toy_df, groupby, "x", {})


``` 
class TestShift(MoveFixtures):

    # 测试默认情况下的Shift操作
    def test_default(self, toy_df):

        # 创建一个GroupBy对象，指定分组列为["color", "group"]
        gb = GroupBy(["color", "group"])
        # 调用Shift类的实例进行Shift操作，将结果存储在res中
        res = Shift()(toy_df, gb, "x", {})
        # 遍历toy_df的所有列，断言每一列的数据是否与res中对应列的数据相等
        for col in toy_df:
            assert_series_equal(toy_df[col], res[col])

    # 使用pytest的参数化测试，测试不同移动值的Shift操作
    @pytest.mark.parametrize("x,y", [(.3, 0), (0, .2), (.1, .3)])
    def test_moves(self, toy_df, x, y):

        # 创建一个GroupBy对象，指定分组列为["color", "group"]
        gb = GroupBy(["color", "group"])
        # 调用Shift类的实例进行Shift操作，将结果存储在res中，传入不同的移动值x和y
        res = Shift(x=x, y=y)(toy_df, gb, "x", {})
        # 断言Shift后"x"列的结果是否与原始数据加上移动值x相等
        assert_array_equal(res["x"], toy_df["x"] + x)
        # 断言Shift后"y"列的结果是否与原始数据加上移动值y相等
        assert_array_equal(res["y"], toy_df["y"] + y)


class TestNorm(MoveFixtures):

    # 使用pytest的参数化测试，测试在无分组情况下的默认Norm操作
    @pytest.mark.parametrize("orient", ["x", "y"])
    def test_default_no_groups(self, df, orient):

        # 根据orient选择对应的另一个方向的列名
        other = {"x": "y", "y": "x"}[orient]
        # 创建一个GroupBy对象，指定分组列为["null"]
        gb = GroupBy(["null"])
        # 调用Norm类的实例进行Norm操作，将结果存储在res中
        res = Norm()(df, gb, orient, {})
        # 断言另一个方向的列的最大值是否接近于1
        assert res[other].max() == pytest.approx(1)

    # 使用pytest的参数化测试，测试在有分组情况下的默认Norm操作
    @pytest.mark.parametrize("orient", ["x", "y"])
    def test_default_groups(self, df, orient):

        # 根据orient选择对应的另一个方向的列名
        other = {"x": "y", "y": "x"}[orient]
        # 创建一个GroupBy对象，指定分组列为["grp2"]
        gb = GroupBy(["grp2"])
        # 调用Norm类的实例进行Norm操作，将结果存储在res中
        res = Norm()(df, gb, orient, {})
        # 遍历分组后的结果，断言每个分组中另一个方向的列的最大值是否接近于1
        for _, grp in res.groupby("grp2"):
            assert grp[other].max() == pytest.approx(1)

    # 测试使用"sum"参数的Norm操作
    def test_sum(self, df):

        # 创建一个GroupBy对象，指定分组列为["null"]
        gb = GroupBy(["null"])
        # 调用Norm类的实例进行Norm操作，将结果存储在res中，使用"sum"作为参数
        res = Norm("sum")(df, gb, "x", {})
        # 断言"y"列的总和是否接近于1
        assert res["y"].sum() == pytest.approx(1)

    # 测试使用"where"参数的Norm操作
    def test_where(self, df):

        # 创建一个GroupBy对象，指定分组列为["null"]
        gb = GroupBy(["null"])
        # 调用Norm类的实例进行Norm操作，将结果存储在res中，使用"where"条件为"x == 2"
        res = Norm(where="x == 2")(df, gb, "x", {})
        # 断言在"x"列等于2的行中"y"列的最大值是否接近于1
        assert res.loc[res["x"] == 2, "y"].max() == pytest.approx(1)

    # 测试使用"percent=True"参数的Norm操作
    def test_percent(self, df):

        # 创建一个GroupBy对象，指定分组列为["null"]
        gb = GroupBy(["null"])
        # 调用Norm类的实例进行Norm操作，将结果存储在res中，使用
```