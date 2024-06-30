# `D:\src\scipysrc\seaborn\tests\_stats\test_counting.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pandas as pd  # 导入 Pandas 库，用于数据处理和分析

import pytest  # 导入 pytest 库，用于编写和运行测试用例
from numpy.testing import assert_array_equal  # 导入 NumPy 的测试工具，用于比较数组是否相等

from seaborn._core.groupby import GroupBy  # 导入 seaborn 库中的 GroupBy 类，用于数据分组操作
from seaborn._stats.counting import Hist, Count  # 导入 seaborn 库中的 Hist 和 Count 类，用于统计和计数操作


class TestCount:  # 定义测试类 TestCount

    @pytest.fixture  # 声明 pytest 的测试固件装饰器
    def df(self, rng):  # 定义一个固件函数 df，依赖于 rng 参数

        n = 30  # 设置数据框的行数为 30
        return pd.DataFrame(dict(  # 返回一个 Pandas 数据框，包含以下列
            x=rng.uniform(0, 7, n).round(),  # x 列为在 [0, 7) 上均匀分布的随机数，四舍五入取整
            y=rng.normal(size=n),  # y 列为服从标准正态分布的随机数
            color=rng.choice(["a", "b", "c"], n),  # color 列为从列表中随机选择的值
            group=rng.choice(["x", "y"], n),  # group 列为从列表中随机选择的值
        ))

    def get_groupby(self, df, orient):  # 定义一个方法 get_groupby，接受数据框和方向参数 orient

        other = {"x": "y", "y": "x"}[orient]  # 根据 orient 参数选择另一列的名称
        cols = [c for c in df if c != other]  # 列出除了 other 列以外的所有列名
        return GroupBy(cols)  # 返回根据列名进行分组的 GroupBy 对象

    def test_single_grouper(self, df):  # 定义单一分组器的测试方法，接受数据框 df 参数

        ori = "x"  # 设置方向为 "x"
        df = df[["x"]]  # 从数据框中选择只包含 "x" 列的子集
        gb = self.get_groupby(df, ori)  # 调用 get_groupby 方法获取 GroupBy 对象
        res = Count()(df, gb, ori, {})  # 调用 Count 类的实例，对数据框进行计数操作
        expected = df.groupby("x").size()  # 获取按 "x" 列分组后的大小
        assert_array_equal(res.sort_values("x")["y"], expected)  # 断言计算结果与预期结果相等

    def test_multiple_groupers(self, df):  # 定义多个分组器的测试方法，接受数据框 df 参数

        ori = "x"  # 设置方向为 "x"
        df = df[["x", "group"]].sort_values("group")  # 从数据框中选择包含 "x" 和 "group" 列的子集，并按 "group" 列排序
        gb = self.get_groupby(df, ori)  # 调用 get_groupby 方法获取 GroupBy 对象
        res = Count()(df, gb, ori, {})  # 调用 Count 类的实例，对数据框进行计数操作
        expected = df.groupby(["x", "group"]).size()  # 获取按 ["x", "group"] 列分组后的大小
        assert_array_equal(res.sort_values(["x", "group"])["y"], expected)  # 断言计算结果与预期结果相等


class TestHist:  # 定义测试类 TestHist

    @pytest.fixture  # 声明 pytest 的测试固件装饰器
    def single_args(self):  # 定义一个固件函数 single_args

        groupby = GroupBy(["group"])  # 创建一个 GroupBy 对象，按 "group" 列进行分组

        class Scale:  # 定义一个 Scale 类
            scale_type = "continuous"  # 设置 scale_type 属性为 "continuous"

        return groupby, "x", {"x": Scale()}  # 返回 GroupBy 对象、字符串 "x" 和包含 Scale 对象的字典

    @pytest.fixture  # 声明 pytest 的测试固件装饰器
    def triple_args(self):  # 定义一个固件函数 triple_args

        groupby = GroupBy(["group", "a", "s"])  # 创建一个 GroupBy 对象，按 ["group", "a", "s"] 列进行分组

        class Scale:  # 定义一个 Scale 类
            scale_type = "continuous"  # 设置 scale_type 属性为 "continuous"

        return groupby, "x", {"x": Scale()}  # 返回 GroupBy 对象、字符串 "x" 和包含 Scale 对象的字典

    def test_string_bins(self, long_df):  # 定义测试字符串类型的分箱方法，接受长数据框 long_df 参数

        h = Hist(bins="sqrt")  # 创建 Hist 类的实例，使用 "sqrt" 字符串作为分箱参数
        bin_kws = h._define_bin_params(long_df, "x", "continuous")  # 调用 _define_bin_params 方法定义分箱参数
        assert bin_kws["range"] == (long_df["x"].min(), long_df["x"].max())  # 断言分箱的范围等于长数据框中 "x" 列的最小和最大值
        assert bin_kws["bins"] == int(np.sqrt(len(long_df)))  # 断言分箱的数量等于长数据框行数的平方根取整数

    def test_int_bins(self, long_df):  # 定义测试整数类型的分箱方法，接受长数据框 long_df 参数

        n = 24  # 设置分箱的数量为 24
        h = Hist(bins=n)  # 创建 Hist 类的实例，使用整数 n 作为分箱参数
        bin_kws = h._define_bin_params(long_df, "x", "continuous")  # 调用 _define_bin_params 方法定义分箱参数
        assert bin_kws["range"] == (long_df["x"].min(), long_df["x"].max())  # 断言分箱的范围等于长数据框中 "x" 列的最小和最大值
        assert bin_kws["bins"] == n  # 断言分箱的数量等于 n

    def test_array_bins(self, long_df):  # 定义测试数组类型的分箱方法，接受长数据框 long_df 参数

        bins = [-3, -2, 1, 2, 3]  # 设置分箱的数组
        h = Hist(bins=bins)  # 创建 Hist 类的实例，使用 bins 数组作为分箱参数
        bin_kws = h._define_bin_params(long_df, "x", "continuous")  # 调用 _define_bin_params 方法定义分箱参数
        assert_array_equal(bin_kws["bins"], bins)  # 断言分箱的数组与预期的 bins 数组相等

    def test_binwidth(self, long_df):  # 定义测试分箱宽度的方法，接受长数据框 long_df 参数

        binwidth = .5  # 设置分箱的宽度为 0.5
        h = Hist(binwidth=binwidth)  # 创建 Hist 类的实例，使用 binwidth 作为分箱宽度参数
        bin_kws = h._define_bin_params(long_df, "x", "continuous")  # 调用 _define_bin_params 方法定义分箱参数
        n_bins = bin_kws["bins"]  # 获取分箱的数量
        left, right = bin_kws["range"]  # 获取分箱的范围
        assert (right - left) / n_bins == pytest.approx(binwidth)  # 使用 pytest 的近似断言，验证分箱的宽度是否等于 binwidth

    def test_binrange(self, long_df):  # 定义测试分箱范围的方法，接受长数据框 long_df 参数

        binrange = (-4, 4)  # 设置分箱的范围为 (-4, 4
    # 测试离散数据的直方图分箱功能
    def test_discrete_bins(self, long_df):
        # 创建一个离散数据直方图对象
        h = Hist(discrete=True)
        # 从长格式数据中选择列"x"并转换为整数类型
        x = long_df["x"].astype(int)
        # 使用长格式数据和指定的列"x"，定义离散数据的直方图分箱参数
        bin_kws = h._define_bin_params(long_df.assign(x=x), "x", "continuous")
        # 断言分箱参数的范围是否正确
        assert bin_kws["range"] == (x.min() - .5, x.max() + .5)
        # 断言分箱参数的箱数是否正确
        assert bin_kws["bins"] == (x.max() - x.min() + 1)

    # 测试从名义尺度数据生成的离散数据直方图分箱功能
    def test_discrete_bins_from_nominal_scale(self, rng):
        # 创建一个直方图对象
        h = Hist()
        # 生成随机整数数组"x"，范围为0到4，共10个数
        x = rng.randint(0, 5, 10)
        # 创建包含"x"列的数据框
        df = pd.DataFrame({"x": x})
        # 使用数据框和指定的列"x"，定义名义尺度数据的直方图分箱参数
        bin_kws = h._define_bin_params(df, "x", "nominal")
        # 断言分箱参数的范围是否正确
        assert bin_kws["range"] == (x.min() - .5, x.max() + .5)
        # 断言分箱参数的箱数是否正确
        assert bin_kws["bins"] == (x.max() - x.min() + 1)

    # 测试计数统计功能
    def test_count_stat(self, long_df, single_args):
        # 创建一个计数统计的直方图对象
        h = Hist(stat="count")
        # 对长格式数据应用计数统计功能
        out = h(long_df, *single_args)
        # 断言统计结果中y列的总和是否等于长格式数据的长度
        assert out["y"].sum() == len(long_df)

    # 测试概率统计功能
    def test_probability_stat(self, long_df, single_args):
        # 创建一个概率统计的直方图对象
        h = Hist(stat="probability")
        # 对长格式数据应用概率统计功能
        out = h(long_df, *single_args)
        # 断言统计结果中y列的总和是否等于1
        assert out["y"].sum() == 1

    # 测试比例统计功能
    def test_proportion_stat(self, long_df, single_args):
        # 创建一个比例统计的直方图对象
        h = Hist(stat="proportion")
        # 对长格式数据应用比例统计功能
        out = h(long_df, *single_args)
        # 断言统计结果中y列的总和是否等于1
        assert out["y"].sum() == 1

    # 测试百分比统计功能
    def test_percent_stat(self, long_df, single_args):
        # 创建一个百分比统计的直方图对象
        h = Hist(stat="percent")
        # 对长格式数据应用百分比统计功能
        out = h(long_df, *single_args)
        # 断言统计结果中y列的总和是否等于100
        assert out["y"].sum() == 100

    # 测试密度统计功能
    def test_density_stat(self, long_df, single_args):
        # 创建一个密度统计的直方图对象
        h = Hist(stat="density")
        # 对长格式数据应用密度统计功能
        out = h(long_df, *single_args)
        # 断言统计结果中y列乘以space列的总和是否等于1
        assert (out["y"] * out["space"]).sum() == 1

    # 测试频率统计功能
    def test_frequency_stat(self, long_df, single_args):
        # 创建一个频率统计的直方图对象
        h = Hist(stat="frequency")
        # 对长格式数据应用频率统计功能
        out = h(long_df, *single_args)
        # 断言统计结果中y列乘以space列的总和是否等于长格式数据的长度
        assert (out["y"] * out["space"]).sum() == len(long_df)

    # 测试无效统计参数的情况
    def test_invalid_stat(self):
        # 使用pytest的断言检查是否引发指定异常和错误消息
        with pytest.raises(ValueError, match="The `stat` parameter for `Hist`"):
            Hist(stat="invalid")

    # 测试累积计数统计功能
    def test_cumulative_count(self, long_df, single_args):
        # 创建一个累积计数统计的直方图对象
        h = Hist(stat="count", cumulative=True)
        # 对长格式数据应用累积计数统计功能
        out = h(long_df, *single_args)
        # 断言统计结果中y列的最大值是否等于长格式数据的长度
        assert out["y"].max() == len(long_df)

    # 测试累积比例统计功能
    def test_cumulative_proportion(self, long_df, single_args):
        # 创建一个累积比例统计的直方图对象
        h = Hist(stat="proportion", cumulative=True)
        # 对长格式数据应用累积比例统计功能
        out = h(long_df, *single_args)
        # 断言统计结果中y列的最大值是否等于1
        assert out["y"].max() == 1

    # 测试累积密度统计功能
    def test_cumulative_density(self, long_df, single_args):
        # 创建一个累积密度统计的直方图对象
        h = Hist(stat="density", cumulative=True)
        # 对长格式数据应用累积密度统计功能
        out = h(long_df, *single_args)
        # 断言统计结果中y列的最大值是否等于1
        assert out["y"].max() == 1

    # 测试默认常规化方式下的百分比统计功能
    def test_common_norm_default(self, long_df, triple_args):
        # 创建一个百分比统计的直方图对象
        h = Hist(stat="percent")
        # 对长格式数据应用百分比统计功能
        out = h(long_df, *triple_args)
        # 断言统计结果中y列的总和是否接近100
        assert out["y"].sum() == pytest.approx(100)

    # 测试关闭常规化方式下的百分比统计功能
    def test_common_norm_false(self, long_df, triple_args):
        # 创建一个关闭常规化的百分比统计的直方图对象
        h = Hist(stat="percent", common_norm=False)
        # 对长格式数据应用关闭常规化的百分比统计功能
        out = h(long_df, *triple_args)
        # 对每组（"a", "s"）检查统计结果中y列的总和是否接近100
        for _, out_part in out.groupby(["a", "s"]):
            assert out_part["y"].sum() == pytest.approx(100)
    # 定义测试函数，用于测试 Histogram 对象的 common_norm_subset 方法
    def test_common_norm_subset(self, long_df, triple_args):
        # 创建 Histogram 对象 h，指定统计方式为百分比，指定公共归一化变量为 "a"
        h = Hist(stat="percent", common_norm=["a"])
        # 调用 Histogram 对象的方法，计算处理后的输出
        out = h(long_df, *triple_args)
        # 根据 "a" 列进行分组，检查每组中 "y" 列的和是否近似等于 100%
        for _, out_part in out.groupby("a"):
            assert out_part["y"].sum() == pytest.approx(100)

    # 定义测试函数，用于测试 Histogram 对象的 common_norm_warning 方法
    def test_common_norm_warning(self, long_df, triple_args):
        # 创建 Histogram 对象 h，指定公共归一化变量为 "b"
        h = Hist(common_norm=["b"])
        # 使用 pytest 的 warns 方法，检查是否会引发 UserWarning，且其消息匹配指定的正则表达式
        with pytest.warns(UserWarning, match=r"Undefined variable\(s\)"):
            h(long_df, *triple_args)

    # 定义测试函数，用于测试 Histogram 对象的 common_bins_default 方法
    def test_common_bins_default(self, long_df, triple_args):
        # 创建 Histogram 对象 h，使用默认参数
        h = Hist()
        # 调用 Histogram 对象的方法，计算处理后的输出
        out = h(long_df, *triple_args)
        bins = []
        # 根据 ["a", "s"] 列进行分组，收集每组中 "x" 列的值构成的元组
        for _, out_part in out.groupby(["a", "s"]):
            bins.append(tuple(out_part["x"]))
        # 断言 bins 集合中的元素数量是否为 1，即所有分组的 "x" 列取值应相同
        assert len(set(bins)) == 1

    # 定义测试函数，用于测试 Histogram 对象的 common_bins_false 方法
    def test_common_bins_false(self, long_df, triple_args):
        # 创建 Histogram 对象 h，禁用公共 binning 功能
        h = Hist(common_bins=False)
        # 调用 Histogram 对象的方法，计算处理后的输出
        out = h(long_df, *triple_args)
        bins = []
        # 根据 ["a", "s"] 列进行分组，收集每组中 "x" 列的值构成的元组
        for _, out_part in out.groupby(["a", "s"]):
            bins.append(tuple(out_part["x"]))
        # 断言 bins 集合中的元素数量是否等于 out 中 ["a", "s"] 组的数量，即每组的 "x" 列取值可以不同
        assert len(set(bins)) == len(out.groupby(["a", "s"]))

    # 定义测试函数，用于测试 Histogram 对象的 common_bins_subset 方法
    def test_common_bins_subset(self, long_df, triple_args):
        # 创建 Histogram 对象 h，禁用公共 binning 功能
        h = Hist(common_bins=False)
        # 调用 Histogram 对象的方法，计算处理后的输出
        out = h(long_df, *triple_args)
        bins = []
        # 根据 "a" 列进行分组，收集每组中 "x" 列的值构成的元组
        for _, out_part in out.groupby("a"):
            bins.append(tuple(out_part["x"]))
        # 断言 bins 集合中的元素数量是否等于 out 中 "a" 列的唯一值数量，即每组的 "x" 列取值应相同
        assert len(set(bins)) == out["a"].nunique()

    # 定义测试函数，用于测试 Histogram 对象的 common_bins_warning 方法
    def test_common_bins_warning(self, long_df, triple_args):
        # 创建 Histogram 对象 h，指定公共 binning 功能为 "b"
        h = Hist(common_bins=["b"])
        # 使用 pytest 的 warns 方法，检查是否会引发 UserWarning，且其消息匹配指定的正则表达式
        with pytest.warns(UserWarning, match=r"Undefined variable\(s\)"):
            h(long_df, *triple_args)

    # 定义测试函数，用于测试 Histogram 对象的 histogram_single 方法
    def test_histogram_single(self, long_df, single_args):
        # 创建 Histogram 对象 h，使用默认参数
        h = Hist()
        # 调用 Histogram 对象的方法，计算处理后的输出
        out = h(long_df, *single_args)
        # 计算 long_df 中 "x" 列的直方图及其边缘
        hist, edges = np.histogram(long_df["x"], bins="auto")
        # 断言处理后的输出中 "y" 列是否与直方图 hist 数组相等
        assert_array_equal(out["y"], hist)
        # 断言处理后的输出中 "space" 列是否与直方图边缘的差分相等
        assert_array_equal(out["space"], np.diff(edges))

    # 定义测试函数，用于测试 Histogram 对象的 histogram_multiple 方法
    def test_histogram_multiple(self, long_df, triple_args):
        # 创建 Histogram 对象 h，使用默认参数
        h = Hist()
        # 调用 Histogram 对象的方法，计算处理后的输出
        out = h(long_df, *triple_args)
        # 使用 "auto" 算法计算 long_df 中 "x" 列的直方图边缘
        bins = np.histogram_bin_edges(long_df["x"], "auto")
        # 根据 ["a", "s"] 列进行分组，对每组数据计算直方图及其边缘
        for (a, s), out_part in out.groupby(["a", "s"]):
            x = long_df.loc[(long_df["a"] == a) & (long_df["s"] == s), "x"]
            hist, edges = np.histogram(x, bins=bins)
            # 断言处理后的输出中每组的 "y" 列是否与计算得到的直方图 hist 数组相等
            assert_array_equal(out_part["y"], hist)
            # 断言处理后的输出中每组的 "space" 列是否与计算得到的直方图边缘的差分相等
            assert_array_equal(out_part["space"], np.diff(edges))
```