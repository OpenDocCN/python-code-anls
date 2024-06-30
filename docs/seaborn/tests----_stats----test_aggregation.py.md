# `D:\src\scipysrc\seaborn\tests\_stats\test_aggregation.py`

```
# 导入必要的库
import numpy as np
import pandas as pd

# 导入 pytest 测试框架和用于断言 DataFrame 相等的函数
import pytest
from pandas.testing import assert_frame_equal

# 导入 seaborn 中的 GroupBy 和 Agg 类
from seaborn._core.groupby import GroupBy
from seaborn._stats.aggregation import Agg, Est


# AggregationFixtures 类定义了测试数据的 fixture
class AggregationFixtures:

    # pytest fixture，生成包含随机数据的 DataFrame
    @pytest.fixture
    def df(self, rng):
        n = 30
        return pd.DataFrame(dict(
            x=rng.uniform(0, 7, n).round(),
            y=rng.normal(size=n),
            color=rng.choice(["a", "b", "c"], n),
            group=rng.choice(["x", "y"], n),
        ))

    # 根据 DataFrame 和方向 orient 获取 GroupBy 对象
    def get_groupby(self, df, orient):
        other = {"x": "y", "y": "x"}[orient]
        cols = [c for c in df if c != other]
        return GroupBy(cols)


# TestAgg 类包含测试 Agg 类的功能
class TestAgg(AggregationFixtures):

    # 测试默认聚合操作
    def test_default(self, df):
        ori = "x"
        df = df[["x", "y"]]
        gb = self.get_groupby(df, ori)
        res = Agg()(df, gb, ori, {})

        expected = df.groupby("x", as_index=False)["y"].mean()
        assert_frame_equal(res, expected)

    # 测试多维度默认聚合操作
    def test_default_multi(self, df):
        ori = "x"
        gb = self.get_groupby(df, ori)
        res = Agg()(df, gb, ori, {})

        # 构建多维度索引并计算期望结果
        grp = ["x", "color", "group"]
        index = pd.MultiIndex.from_product(
            [sorted(df["x"].unique()), df["color"].unique(), df["group"].unique()],
            names=["x", "color", "group"]
        )
        expected = (
            df
            .groupby(grp)
            .agg("mean")
            .reindex(index=index)
            .dropna()
            .reset_index()
            .reindex(columns=df.columns)
        )
        assert_frame_equal(res, expected)

    # 使用参数化测试不同的聚合函数
    @pytest.mark.parametrize("func", ["max", lambda x: float(len(x) % 2)])
    def test_func(self, df, func):
        ori = "x"
        df = df[["x", "y"]]
        gb = self.get_groupby(df, ori)
        res = Agg(func)(df, gb, ori, {})

        expected = df.groupby("x", as_index=False)["y"].agg(func)
        assert_frame_equal(res, expected)


# TestEst 类包含测试 Est 类的功能
class TestEst(AggregationFixtures):

    # 注意：大部分底层代码在 tests/test_statistics 中已经测试
    # 使用参数化测试不同的统计函数
    @pytest.mark.parametrize("func", [np.mean, "mean"])
    def test_mean_sd(self, df, func):
        ori = "x"
        df = df[["x", "y"]]
        gb = self.get_groupby(df, ori)
        res = Est(func, "sd")(df, gb, ori, {})

        # 计算均值和标准差，构建期望的 DataFrame
        grouped = df.groupby("x", as_index=False)["y"]
        est = grouped.mean()
        err = grouped.std().fillna(0)  # 填充 NaN 值，仅在固定测试时需要
        expected = est.assign(ymin=est["y"] - err["y"], ymax=est["y"] + err["y"])
        assert_frame_equal(res, expected)

    # 测试单一观测值的标准差情况
    def test_sd_single_obs(self):
        y = 1.5
        ori = "x"
        df = pd.DataFrame([{"x": "a", "y": y}])
        gb = self.get_groupby(df, ori)
        res = Est("mean", "sd")(df, gb, ori, {})
        expected = df.assign(ymin=y, ymax=y)
        assert_frame_equal(res, expected)
    # 定义一个测试函数，用于测试 Est 类中的 median_pi 方法
    def test_median_pi(self, df):
        # 初始化原始字符串 ori
        ori = "x"
        # 从数据框 df 中选择 "x" 和 "y" 列
        df = df[["x", "y"]]
        # 调用自定义方法 get_groupby 获取按 ori 分组的结果 gb
        gb = self.get_groupby(df, ori)
        # 调用 Est 类的实例化对象，使用 median 方法和参数 ("pi", 100)，对 df 和 gb 执行估计
        res = Est("median", ("pi", 100))(df, gb, ori, {})
        
        # 对 df 按 "x" 列分组，并计算 "y" 列的中位数
        grouped = df.groupby("x", as_index=False)["y"]
        est = grouped.median()
        # 为估计结果添加 "ymin" 和 "ymax" 列，分别存放每组最小和最大值
        expected = est.assign(ymin=grouped.min()["y"], ymax=grouped.max()["y"])
        # 断言 res 和 expected 数据框相等
        assert_frame_equal(res, expected)

    # 定义一个测试函数，用于测试 Est 类中的 weighted_mean 方法
    def test_weighted_mean(self, df, rng):
        # 从均匀分布 rng 中生成权重数组，长度与 df 相同
        weights = rng.uniform(0, 5, len(df))
        # 调用自定义方法 get_groupby 获取按 "x" 列分组的结果 gb
        gb = self.get_groupby(df[["x", "y"]], "x")
        # 在 df 中添加一列 "weight"，存放权重数组
        df = df.assign(weight=weights)
        # 调用 Est 类的实例化对象，使用 mean 方法对 df 和 gb 执行加权平均估计
        res = Est("mean")(df, gb, "x", {})
        
        # 对 res 中的每一行进行遍历
        for _, res_row in res.iterrows():
            # 选择 df 中 "x" 列与 res_row["x"] 相等的行组成 rows
            rows = df[df["x"] == res_row["x"]]
            # 计算 rows["y"] 的加权平均值，权重为 rows["weight"]
            expected = np.average(rows["y"], weights=rows["weight"])
            # 断言 res_row["y"] 等于预期的加权平均值 expected
            assert res_row["y"] == expected

    # 定义一个测试函数，用于测试 Est 类中的 seed 方法
    def test_seed(self, df):
        # 初始化原始字符串 ori
        ori = "x"
        # 调用自定义方法 get_groupby 获取按 ori 分组的结果 gb
        gb = self.get_groupby(df, ori)
        # 准备参数 args，包括 df, gb, ori, 空字典 {}
        args = df, gb, ori, {}
        # 使用 seed=99 参数调用 Est 类的实例化对象，执行 mean 和 ci 方法
        res1 = Est("mean", "ci", seed=99)(*args)
        # 再次使用相同参数和 seed=99 调用 Est 类的实例化对象，执行 mean 和 ci 方法
        res2 = Est("mean", "ci", seed=99)(*args)
        # 断言两次调用的结果 res1 和 res2 的数据框内容完全相同
        assert_frame_equal(res1, res2)
```